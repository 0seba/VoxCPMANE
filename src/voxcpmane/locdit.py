"""Pure numpy runtime wrapper around the converted LocDiT CoreML package.

The converted ``.mlpackage`` takes CFG batch = 2 inputs in channels-first
NCHW layout (see :mod:`qeml.conversion.voxcpm2.locdit`) and returns the
velocity prediction for both the conditional and unconditional branches
in one predict call.

This module is numpy-only — no torch dependency. For the torch-compatible
subclass used when monkey-patching a live VoxCPM2 ``nn.Module`` graph,
see :mod:`qeml.voxcpm2.locdit_torch`.
"""

from __future__ import annotations

import math
import time as _time

import coremltools as ct
import numpy as np

from pathlib import Path

from ._coreml_utils import (
    get_feature_info,
    load_coreml_model,
)


def _sinusoidal_pos_emb_np(
    t: np.ndarray, dim: int, scale: float = 1000.0
) -> np.ndarray:
    """Numpy twin of ``voxcpm.modules.locdit.SinusoidalPosEmb`` — runs in
    fp32 on the caller side so the ``sin``/``cos`` stay out of the fp16
    CoreML graph (fp16 ``sin`` of values near 1000 loses precision)."""
    half_dim = dim // 2
    emb_step = math.log(10000.0) / (half_dim - 1)
    freqs = np.exp(np.arange(half_dim, dtype=np.float32) * -emb_step)
    t = np.asarray(t, dtype=np.float32)
    emb = scale * t[..., None] * freqs
    return np.concatenate([np.sin(emb), np.cos(emb)], axis=-1)


class CoreMLUnifiedCFM:
    """CoreML-backed replacement for ``voxcpm.modules.locdit.UnifiedCFM``.

    The ``estimator`` traced inside the mlpackage expects inputs in
    channels-first NCHW at a fixed batch of 2. The solver below fills
    both halves of that batch (cond + uncond) per step, then splits the
    output to recover ``dphi_dt`` and ``cfg_dphi_dt``.

    Pure numpy — no torch. For a torch ``nn.Module`` wrapper that can be
    installed as a ``tts.feat_decoder`` submodule, see
    :class:`qeml.voxcpm2.locdit_torch.CoreMLUnifiedCFM`.
    """

    def __init__(
        self,
        mlmodel_path: Path,
        *,
        in_channels: int = 64,
        sigma_min: float = 1e-6,
        t_scheduler: str = "log-norm",
        mean_mode: bool = False,
        compute_units: ct.ComputeUnit = ct.ComputeUnit.CPU_AND_NE,
    ):
        self.mlmodel = load_coreml_model(mlmodel_path, compute_units=compute_units)
        self.in_channels = in_channels
        self.sigma_min = sigma_min
        self.t_scheduler = t_scheduler
        self.mean_mode = mean_mode
        self.input_dtype = get_feature_info(self.mlmodel, mlmodel_path, "x")["dtype"]
        self._timestep_cache: dict[
            tuple[int, int, float, bool, str],
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ] = {}

    # ------------------------------------------------------------------ #
    # solver internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _optimized_scale(
        positive_flat: np.ndarray, negative_flat: np.ndarray
    ) -> np.ndarray:
        dot_product = np.sum(positive_flat * negative_flat, axis=1, keepdims=True)
        squared_norm = (
            np.sum(negative_flat * negative_flat, axis=1, keepdims=True) + 1e-8
        )
        return dot_product / squared_norm

    def _cached_timestep_tables(
        self,
        n_timesteps: int,
        hidden_size: int,
        sway_sampling_coef: float,
        model_dtype: np.dtype,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dtype = np.dtype(model_dtype)
        key = (
            int(n_timesteps),
            int(hidden_size),
            float(sway_sampling_coef),
            self.mean_mode,
            dtype.str,
        )
        cached = self._timestep_cache.get(key)
        if cached is not None:
            return cached

        t_span = np.linspace(1, 0, n_timesteps + 1, dtype=np.float32)
        t_span = t_span + sway_sampling_coef * (np.cos(np.pi / 2 * t_span) - 1 + t_span)

        t_emb = _sinusoidal_pos_emb_np(t_span[:-1], hidden_size).astype(dtype)
        t_emb = np.ascontiguousarray(t_emb.reshape(n_timesteps, hidden_size, 1, 1))

        dt_val = float(t_span[0] - t_span[1])
        dt_emb_t = np.array(dt_val if self.mean_mode else 0.0, dtype=np.float32)
        dt_emb = _sinusoidal_pos_emb_np(dt_emb_t, hidden_size).astype(dtype)
        dt_emb = np.ascontiguousarray(dt_emb.reshape(hidden_size, 1, 1))

        cached = (t_span, t_emb, dt_emb)
        self._timestep_cache[key] = cached
        return cached

    # ------------------------------------------------------------------ #
    # numpy-only call surface
    # ------------------------------------------------------------------ #

    def predict_numpy(
        self,
        mu: np.ndarray,
        n_timesteps: int,
        patch_size: int,
        cond: np.ndarray,
        temperature: float = 1.0,
        cfg_value: float = 1.0,
        sway_sampling_coef: float = 1.0,
        use_cfg_zero_star: bool = True,
        rng: np.random.Generator | None = None,
        timings: dict | None = None,
    ) -> np.ndarray:
        """Numpy-only diffusion sampling — no torch dependency.

        Args:
            mu: ``(B, 2 * dit_hidden)`` flat concatenated projections.
            cond: ``(B, in_channels, cond_len)`` prefix condition.
            rng: Optional numpy random generator for reproducibility.
            timings: Optional dict to accumulate sub-timings into keys
                ``"predict"``, ``"math"``, ``"n_predict"``.
        Returns:
            ``(B, in_channels, patch_size)`` numpy array.
        """
        if rng is None:
            rng = np.random.default_rng()
        b = mu.shape[0]
        z = (
            rng.standard_normal((b, self.in_channels, patch_size)).astype(np.float32)
            * temperature
        )

        hidden_size = mu.shape[1] // 2
        model_dtype = self.input_dtype.type
        t_span, t_emb, dt_emb = self._cached_timestep_tables(
            n_timesteps=n_timesteps,
            hidden_size=hidden_size,
            sway_sampling_coef=sway_sampling_coef,
            model_dtype=model_dtype,
        )

        return self._solve_euler_numpy(
            x=z,
            t_span=t_span,
            mu=mu,
            cond=cond,
            t_emb=t_emb,
            dt_emb=dt_emb,
            cfg_value=float(cfg_value),
            use_cfg_zero_star=bool(use_cfg_zero_star),
            timings=timings,
        )

    def _solve_euler_numpy(
        self,
        x: np.ndarray,
        t_span: np.ndarray,
        mu: np.ndarray,
        cond: np.ndarray,
        t_emb: np.ndarray,
        dt_emb: np.ndarray,
        cfg_value: float,
        use_cfg_zero_star: bool,
        timings: dict | None = None,
    ) -> np.ndarray:
        t_predict = 0.0
        t_math = 0.0
        n_predict = 0
        b = x.shape[0]
        patch_size = x.shape[2]
        cond_len = cond.shape[2]

        hidden_size = mu.shape[1] // 2
        mu_cf = mu.reshape(b, 2, hidden_size).transpose(0, 2, 1)[:, :, None, :]
        model_dtype = self.input_dtype.type
        mu_cf = np.ascontiguousarray(mu_cf, dtype=model_dtype)

        cond_cf = cond[:, :, None, :].astype(model_dtype)
        cond_cf = np.ascontiguousarray(cond_cf)

        t_val = float(t_span[0])
        dt_val = float(t_span[0] - t_span[1])

        two_b = 2 * b
        x_in = np.zeros((two_b, self.in_channels, 1, patch_size), dtype=model_dtype)
        mu_in = np.zeros((two_b, hidden_size, 1, 2), dtype=model_dtype)
        cond_in = np.zeros((two_b, self.in_channels, 1, cond_len), dtype=model_dtype)
        t_emb_in = np.zeros((two_b, hidden_size, 1, 1), dtype=model_dtype)
        dt_emb_in = np.zeros((two_b, hidden_size, 1, 1), dtype=model_dtype)

        cond_in[:b] = cond_cf
        cond_in[b:] = cond_cf
        mu_in[:b] = mu_cf

        dt_emb_in[:] = dt_emb

        zero_init_steps = max(1, int(len(t_span) * 0.04))
        x_cur = x.astype(np.float32)

        for step in range(1, len(t_span)):
            if use_cfg_zero_star and step <= zero_init_steps:
                dphi_dt = np.zeros_like(x_cur)
            else:
                t_m0 = _time.perf_counter()
                x_in[:b] = x_cur[:, :, None, :]
                x_in[b:] = x_cur[:, :, None, :]

                t_emb_in[:] = t_emb[step - 1]
                t_math += _time.perf_counter() - t_m0

                t_p0 = _time.perf_counter()
                out = self.mlmodel.predict(
                    {
                        "x": x_in,
                        "mu": mu_in,
                        "t_emb": t_emb_in,
                        "cond": cond_in,
                        "dt_emb": dt_emb_in,
                    }
                )["output"]
                t_predict += _time.perf_counter() - t_p0
                n_predict += 1

                t_m0 = _time.perf_counter()
                out = out.astype(np.float32).reshape(
                    two_b, self.in_channels, patch_size
                )

                dphi, cfg_dphi = out[:b], out[b:]

                if use_cfg_zero_star:
                    positive_flat = dphi.reshape(b, -1)
                    negative_flat = cfg_dphi.reshape(b, -1)
                    st_star = self._optimized_scale(positive_flat, negative_flat)
                    st_star = st_star.reshape(b, *([1] * (dphi.ndim - 1)))
                else:
                    st_star = 1.0

                dphi_dt = cfg_dphi * st_star + cfg_value * (dphi - cfg_dphi * st_star)
                t_math += _time.perf_counter() - t_m0

            t_m0 = _time.perf_counter()
            x_cur = x_cur - dt_val * dphi_dt
            t_val = t_val - dt_val
            if step < len(t_span) - 1:
                dt_val = t_val - float(t_span[step + 1])
            t_math += _time.perf_counter() - t_m0

        if timings is not None:
            timings["predict"] = timings.get("predict", 0.0) + t_predict
            timings["math"] = timings.get("math", 0.0) + t_math
            timings["n_predict"] = timings.get("n_predict", 0) + n_predict

        return np.ascontiguousarray(x_cur)
