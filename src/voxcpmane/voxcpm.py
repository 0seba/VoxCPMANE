import os
import re
import math
import time
import queue
import tempfile
import warnings
import threading
import concurrent.futures
from typing import List, Optional, Tuple, Generator, Union

import coremltools as ct

import numpy as np
from einops import rearrange
from transformers import PreTrainedTokenizer, LlamaTokenizerFast
from tqdm import tqdm

import soxr
import soundfile

# import librosa
# from scipy import signal

from .text_normalize import TextNormalizer


STATE_MAX_LENGTH = 2048


def mask_multichar_chinese_tokens(tokenizer: PreTrainedTokenizer):
    """Create a tokenizer wrapper that converts multi-character Chinese tokens to single characters.

    This function creates a wrapper around the provided tokenizer that automatically
    splits multi-character Chinese tokens into individual characters. This is useful
    for ensuring consistent tokenization of Chinese text.

    Args:
        tokenizer: The base tokenizer to wrap

    Returns:
        A CharTokenizerWrapper instance that handles multi-character Chinese tokens

    Example:
        >>> from transformers import LlamaTokenizerFast
        >>> tokenizer = LlamaTokenizerFast.from_pretrained("path/to/tokenizer")
        >>> wrapped_tokenizer = mask_multichar_chinese_tokens(tokenizer)
        >>> tokens = wrapped_tokenizer("你好世界")
    """
    # Pre-compute multi-character tokens (length >= 2, pure Chinese characters)
    multichar_tokens = {
        token
        for token in tokenizer.vocab.keys()
        if len(token) >= 2 and all("\u4e00" <= c <= "\u9fff" for c in token)
    }

    class CharTokenizerWrapper:
        """Wrapper class for tokenizers that handles multi-character Chinese tokens.

        This wrapper automatically splits multi-character Chinese tokens into
        individual characters while preserving the original tokenizer's interface.
        """

        def __init__(self, base_tokenizer: PreTrainedTokenizer) -> None:
            """Initialize the wrapper with a base tokenizer.

            Args:
                base_tokenizer: The tokenizer to wrap
            """
            self.tokenizer = base_tokenizer
            self.multichar_tokens = multichar_tokens

        def tokenize(self, text: str, **kwargs) -> List[str]:
            """Tokenize text and split multi-character Chinese tokens into single characters.

            Args:
                text: Input text to tokenize
                **kwargs: Additional arguments passed to the base tokenizer

            Returns:
                List of processed tokens with multi-character Chinese tokens split

            Example:
                >>> wrapper = CharTokenizerWrapper(tokenizer)
                >>> tokens = wrapper.tokenize("你好世界")
                >>> # Returns ["你", "好", "世", "界"] instead of ["你好", "世界"]
            """
            if not isinstance(text, str):
                raise TypeError(f"Expected string input, got {type(text)}")

            tokens = self.tokenizer.tokenize(text, **kwargs)
            processed = []

            for token in tokens:
                # Remove possible subword prefix
                clean_token = token.replace(" ", "")

                if clean_token in self.multichar_tokens:
                    # Split multi-character token into single characters
                    chars = list(clean_token)
                    processed.extend(chars)
                else:
                    processed.append(token)

            return processed

        def __call__(self, text: str, **kwargs) -> List[int]:
            """Call the tokenizer and return token IDs.

            This method provides the same interface as the original tokenizer
            but with multi-character Chinese token handling.

            Args:
                text: Input text to tokenize
                **kwargs: Additional arguments passed to the base tokenizer

            Returns:
                List of token IDs

            Raises:
                TypeError: If input is not a string
                ValueError: If tokenization fails
            """
            try:
                tokens = self.tokenize(text, **kwargs)
                result = self.tokenizer.convert_tokens_to_ids(tokens)
                return result
            except Exception as e:
                raise ValueError(f"Tokenization failed: {str(e)}") from e

    return CharTokenizerWrapper(tokenizer)


class BaseLMANEWrapperWithCache:
    def __init__(self, embed_tokens, lm_mlmodel_model: ct.models.MLModel, chunk_size=4):
        self.lm_mlmodel_model = lm_mlmodel_model
        self.state: Optional[ct.models.model.MLState] = None
        self.current_position = 0
        self.chunk_size = chunk_size

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def reset_state(self):
        self.state = self.lm_mlmodel_model.make_state()
        self.current_position = 0

    def forward(self, inputs_embeds, is_causal=True, reset_state=True):
        bsz, _, _, length = inputs_embeds.shape
        if reset_state or self.state is None:
            self.reset_state()

        if length == self.chunk_size:
            position_id = np.array([self.current_position], dtype=np.int32)
            outputs = self.lm_mlmodel_model.predict(
                {
                    "inputs_embeds": inputs_embeds,
                    "position_id": position_id,
                },
                state=self.state,
            )
            self.current_position += self.chunk_size
            return outputs["output"], None

        num_steps = math.ceil(length / self.chunk_size)
        pad_size = num_steps * self.chunk_size - length
        inputs_embeds = np.pad(inputs_embeds, ((0, 0), (0, 0), (0, 0), (0, pad_size)))

        output_chunks = []
        chunks = np.split(inputs_embeds, num_steps, axis=-1)
        for i in range(num_steps):
            position_id = np.array([self.current_position], dtype=np.int32)
            outputs = self.lm_mlmodel_model.predict(
                {
                    "inputs_embeds": chunks[i],
                    "position_id": position_id,
                },
                state=self.state,
            )
            output_chunks.append(outputs["output"])
            self.current_position += self.chunk_size

        self.current_position -= pad_size

        output = np.concatenate(output_chunks, axis=-1)[:, :, :, :length]
        return output, None

    def forward_step(self, inputs_embeds, position_id):
        return self.forward(inputs_embeds, is_causal=True, reset_state=False)[0]


class FeatureEncoderANEWrapper:
    def __init__(self, mlmodel: ct.models.MLModel, chunk_size=14):
        self.mlmodel = mlmodel
        self.chunk_size = chunk_size

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputs_embeds: np.ndarray):
        bsz, dim, p, length = inputs_embeds.shape
        total_bsz = bsz * length
        inputs_embeds = rearrange(inputs_embeds, "b d p t -> (b t) d 1 p")

        if total_bsz == 1 or total_bsz == self.chunk_size:
            outputs = self.mlmodel.predict(
                {
                    "x": inputs_embeds,
                },
            )
            output = outputs["output"].transpose(2, 1, 3, 0)
            return output

        num_steps = math.ceil(total_bsz / self.chunk_size)
        pad_size = num_steps * self.chunk_size - total_bsz
        inputs_embeds = np.pad(inputs_embeds, ((0, pad_size), (0, 0), (0, 0), (0, 0)))

        output_chunks = []
        for i in range(num_steps):
            outputs = self.mlmodel.predict(
                {
                    "x": inputs_embeds[i * self.chunk_size : (i + 1) * self.chunk_size],
                },
            )
            output_chunks.append(outputs["output"])

        output = np.concatenate(output_chunks, axis=0)[:total_bsz].reshape(
            bsz, length, -1, 1
        )
        output = output.transpose(0, 2, 3, 1)
        return output

    def forward_step(self, inputs_embeds: np.ndarray, position_id):
        return self.forward(
            np.expand_dims(inputs_embeds, 1), is_causal=True, reset_state=False
        )[0].squeeze(1)


class MLModelWithPad:
    def __init__(
        self, mlmodel: ct.models.MLModel, input_name: str, chunk_size: int = 4
    ):
        self.mlmodel = mlmodel
        self.input_name = input_name
        self.chunk_size = chunk_size

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputs: np.ndarray):
        bsz, dim, c, length = inputs.shape

        if length == 1 or length == self.chunk_size:
            return self.mlmodel.predict(
                {
                    self.input_name: inputs,
                },
            )["output"]
        else:
            num_steps = math.ceil(length / self.chunk_size)
            pad_size = num_steps * self.chunk_size - length
            inputs = np.pad(inputs, ((0, 0), (0, 0), (0, 0), (0, pad_size)))
            chunk_size = self.chunk_size

        output_chunks = []
        for i in range(num_steps):
            outputs = self.mlmodel.predict(
                {
                    self.input_name: inputs[..., i * chunk_size : (i + 1) * chunk_size],
                },
            )
            output_chunks.append(outputs["output"])

        if length == 1:
            output = output_chunks[0]
        else:
            output = np.concatenate(output_chunks, axis=-1)[..., :length]

        return output


class Projections:
    def __init__(self, mlmodel: ct.models.MLModel):
        self.mlmodel = mlmodel

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, lm_hidden: np.ndarray, residual_hidden: np.ndarray):
        output = self.mlmodel.predict(
            {
                "lm_hidden": lm_hidden,
                "residual_hidden": residual_hidden,
                "t": np.array([0.5], dtype=np.float32),
            }
        )

        dit_hidden = output["dit_hidden"]
        stop_flag = output["stop_flag"]
        return dit_hidden, stop_flag


class VAEEncoderANEWrapper:
    def __init__(
        self,
        mlmodel: ct.models.MLModel,
        chunk_size=15_360,
        overlap_size=2560,
        hop_length=640,
    ):
        self.mlmodel = mlmodel
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.step_size = chunk_size - overlap_size
        self.hop_length = hop_length

        # Encoder output rate (samples per frame)
        # 17920 -> 28  ⇒  17920 / 28 = 640 samples per frame
        self.samples_per_frame = 640

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def preprocess(self, audio_data, sample_rate):
        pad_to = self.hop_length
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / pad_to) * pad_to - length
        if right_pad > 0:
            audio_data = np.pad(audio_data, ((0, 0), (0, 0), (0, right_pad)))
        return audio_data

    def forward(self, audio_data: np.ndarray, sample_rate):
        if audio_data.ndim == 2:
            audio_data = np.expand_dims(audio_data, 1)

        audio_data = self.preprocess(audio_data, sample_rate)
        audio_data = audio_data.astype(np.float32)
        *_, length = audio_data.shape

        if length == self.chunk_size:
            return self.mlmodel.predict({"x": audio_data})["output"]

        num_steps = math.ceil((length - self.overlap_size) / self.step_size)
        total_length = (num_steps - 1) * self.step_size + self.chunk_size
        pad_size = total_length - length

        if pad_size > 0:
            audio_data = np.pad(audio_data, ((0, 0), (0, 0), (0, pad_size)))

        chunks = []
        for i in range(num_steps):
            start = i * self.step_size
            end = start + self.chunk_size
            chunk = audio_data[..., start:end]
            output = self.mlmodel.predict({"x": chunk})["output"]
            chunks.append(output)

        output = np.concatenate(chunks, axis=-1)

        if pad_size > 0:
            pad_frames = math.ceil(pad_size / self.samples_per_frame)
            if pad_frames > 0:
                output = output[..., :-pad_frames]

        return output


class AudioVAEDecoderANEWrapper:
    def __init__(self, mlmodel: ct.models.MLModel, input_seq_length=24):
        self.mlmodel = mlmodel
        self.input_seq_length = input_seq_length

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: np.ndarray, patch_len: int, chunk_size: int):
        pad_size = self.input_seq_length - x.shape[2]
        if pad_size > 0:
            x = np.pad(x, ((0, 0), (0, 0), (0, pad_size)))

        decode_audio = self.mlmodel.predict(
            {
                "latent_pred": x,
            }
        )["decoded_audio"]

        if pad_size > 0:
            end = self.input_seq_length * chunk_size - pad_size * chunk_size
            start = -pad_size * chunk_size - patch_len
            # decode_audio = decode_audio[..., start:end]
            decode_audio = decode_audio[..., :end]

        return decode_audio.reshape(-1)


class UnifiedCFMANE:
    def __init__(
        self,
        in_channels: int,
        sigma_min,
        t_scheduler,
        estimator: ct.models.MLModel,
        mean_mode=False,
    ):
        self.in_channels = in_channels
        self.sigma_min = sigma_min
        self.t_scheduler = t_scheduler
        self.in_channels = in_channels
        self.mean_mode = mean_mode
        self.estimator = estimator

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        mu: np.ndarray,
        n_timesteps: int,
        patch_size: int,
        cond: np.ndarray,
        temperature: float = 1.0,
        cfg_value: float = 1.0,
        sway_sampling_coef: float = 1.0,
        use_cfg_zero_star: bool = True,
    ):
        mu = mu.squeeze((-1, -2))
        b, c = mu.shape
        t = patch_size
        z = np.random.normal(scale=temperature, size=(b, self.in_channels, t))
        t_span = np.linspace(1, 0, n_timesteps + 1, dtype=mu.dtype)
        t_span = t_span + sway_sampling_coef * (np.cos(np.pi / 2 * t_span) - 1 + t_span)
        return self.solve_euler(
            z,
            t_span=t_span,
            mu=mu,
            cond=cond,
            cfg_value=cfg_value,
            use_cfg_zero_star=use_cfg_zero_star,
        )

    def optimized_scale(self, positive_flat, negative_flat):
        dot_product = np.sum(positive_flat * negative_flat, axis=1, keepdims=True)
        squared_norm = np.sum(negative_flat**2, axis=1, keepdims=True) + 1e-8
        st_star = dot_product / squared_norm
        return st_star

    def solve_euler(
        self,
        x: np.ndarray,
        t_span: np.ndarray,
        mu: np.ndarray,
        cond: np.ndarray,
        cfg_value: float = 1.0,
        use_cfg_zero_star: bool = True,
    ):
        t, _, dt = t_span[0], t_span[-1], t_span[0] - t_span[1]
        sol = []
        zero_init_steps = max(1, int(len(t_span) * 0.04))
        for step in range(1, len(t_span)):
            if use_cfg_zero_star and step <= zero_init_steps:
                dphi_dt = 0.0
            else:
                b = x.shape[0]
                x_in = np.zeros([2 * b, self.in_channels, x.shape[2]], dtype=x.dtype)
                mu_in = np.zeros([2 * b, mu.shape[1]], dtype=x.dtype)
                t_in = np.zeros([2 * b], dtype=x.dtype)
                dt_in = np.zeros([2 * b], dtype=x.dtype)
                cond_in = np.zeros([2 * b, self.in_channels, x.shape[2]], dtype=x.dtype)
                x_in[:b], x_in[b:] = x, x
                mu_in[:b] = mu
                t_in[:b], t_in[b:] = t, t
                dt_in[:b], dt_in[b:] = dt, dt

                if not self.mean_mode:
                    dt_in = np.zeros_like(dt_in)
                cond_in[:b], cond_in[b:] = cond, cond

                dphi_dt = self.estimator.predict(
                    {
                        "x": x_in,
                        "mu": mu_in,
                        "t": t_in,
                        "cond": cond_in,
                        "dt": dt_in,
                    }
                )["output"]

                dphi_dt, cfg_dphi_dt = np.split(dphi_dt, 2, axis=0)

                if use_cfg_zero_star:
                    positive_flat = dphi_dt.reshape(b, -1)
                    negative_flat = cfg_dphi_dt.reshape(b, -1)
                    st_star = self.optimized_scale(positive_flat, negative_flat)
                    st_star = st_star.reshape(b, *([1] * (len(dphi_dt.shape) - 1)))
                else:
                    st_star = 1.0

                dphi_dt = cfg_dphi_dt * st_star + cfg_value * (
                    dphi_dt - cfg_dphi_dt * st_star
                )

            x = x - dt * dphi_dt
            t = t - dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t - t_span[step + 1]

        return np.expand_dims(x, axis=-1)


class AudioVAEHolder:
    def __init__(self, encoder, decoder, latent_dim):
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)


class Embeddings:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __call__(self, indices):
        return self.embeddings[indices]


class VoxCPMModelANE:
    def __init__(
        self,
        tokenizer: LlamaTokenizerFast,
        # base_lm_mlmodel: ct.models.MLModel,
        base_lm_mlmodel_path: str,
        residual_lm_mlmodel_path: str,
        fsq_mlmodel: ct.models.MLModel,
        locdit_mlmodel: ct.models.MLModel,
        projections_mlmodel: ct.models.MLModel,
        audio_vae_decoder_mlmodel: ct.models.MLModel,
        audio_vae_encoder_mlmodel: ct.models.MLModel,
        feature_encoder_mlmodel: ct.models.MLModel,
        base_lm_embed_tokens: np.ndarray,
        vae_encoder_hop_length: int,
        patch_size=2,
        latent_dim=64,
        sigma_min=1e-6,
        t_scheduler="log-norm",
        mean_mode=False,
        audio_vae_latent_dim=64,
        audio_vae_chunk_size=640,
        audio_vae_sample_rate=16_000,
        scale_emb=12.0,
        use_mup=False,
        # audio_vae_encoder_chunk_size=17_920,
        audio_vae_encoder_chunk_size=15_360,
        base_lm_chunk_size=4,
        residual_lm_chunk_size=4,
        feature_encoder_chunk_size=14,
        fsq_layer_chunk_size=32,
        audio_vae_decoder_chunk_size=24,
    ):
        self.text_tokenizer = mask_multichar_chinese_tokens(tokenizer)
        self.audio_start_token = 101
        self.audio_end_token = 102
        self.feat_dim = latent_dim
        self.patch_size = patch_size
        # self.embed_tokens = Embeddings(base_lm_embed_tokens)
        self.embed_tokens = base_lm_embed_tokens
        # self.base_lm = BaseLMANEWrapperWithCache(Embeddings(base_lm_embed_tokens), base_lm_mlmodel, chunk_size=base_lm_chunk_size)
        # self.residual_lm = BaseLMANEWrapperWithCache(None, residual_lm_mlmodel, chunk_size=residual_lm_chunk_size)
        self.base_lm_model_path = base_lm_mlmodel_path
        self.residual_lm_model_path = residual_lm_mlmodel_path
        self.base_lm_chunk_size = base_lm_chunk_size
        self.residual_lm_chunk_size = residual_lm_chunk_size
        # self.base_lm = ct.models.CompiledMLModel(base_lm_mlmodel_path, compute_units=ct.ComputeUnit.CPU_AND_NE, function_name=f"length_{self.base_lm_chunk_size}")
        # self.residual_lm = ct.models.CompiledMLModel(residual_lm_mlmodel_path, compute_units=ct.ComputeUnit.CPU_AND_NE, function_name=f"length_{self.residual_lm_chunk_size}")
        self.base_lm: BaseLMANEWrapperWithCache = None
        self.residual_lm: BaseLMANEWrapperWithCache = None
        self.base_lm_inference: BaseLMANEWrapperWithCache = None
        self.residual_lm_inference: BaseLMANEWrapperWithCache = None
        self.load_preprocessing_models()
        self.load_inference_models()
        self.feat_encoder = FeatureEncoderANEWrapper(
            feature_encoder_mlmodel, chunk_size=feature_encoder_chunk_size
        )
        self.feat_encoder_step = self.feat_encoder
        self.feat_decoder = UnifiedCFMANE(
            in_channels=latent_dim,
            sigma_min=sigma_min,
            t_scheduler=t_scheduler,
            estimator=locdit_mlmodel,
            mean_mode=mean_mode,
        )
        self.fsq_layer = MLModelWithPad(fsq_mlmodel, "x", fsq_layer_chunk_size)
        self.lm_to_dit_proj = Projections(projections_mlmodel)
        self.audio_vae_encoder = VAEEncoderANEWrapper(
            audio_vae_encoder_mlmodel,
            # chunk_size=15_360,
            chunk_size=audio_vae_encoder_chunk_size,
            overlap_size=2_560,
            hop_length=vae_encoder_hop_length,
        )
        self.audio_vae_decoder = AudioVAEDecoderANEWrapper(
            audio_vae_decoder_mlmodel,
            input_seq_length=audio_vae_decoder_chunk_size,
        )
        self.audio_vae = AudioVAEHolder(
            self.audio_vae_encoder,
            self.audio_vae_decoder,
            audio_vae_latent_dim,
        )
        self.audio_vae_latent_dim = audio_vae_latent_dim
        self.audio_vae_chunk_size = audio_vae_chunk_size
        self.chunk_size = audio_vae_chunk_size
        self.sample_rate = audio_vae_sample_rate
        self.scale_emb = scale_emb
        self.use_mup = use_mup

    def _generate(
        self,
        target_text: str,
        prompt_text: str = "",
        prompt_wav_path: str = "",
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        retry_badcase: bool = False,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        streaming: bool = False,
    ) -> Generator[np.ndarray, None, None]:
        if retry_badcase and streaming:
            warnings.warn(
                "Retry on bad cases is not supported in streaming mode, setting retry_badcase=False."
            )
            retry_badcase = False
        if len(prompt_wav_path) == 0:
            text = target_text
            text_token = np.array(self.text_tokenizer(text), dtype=np.int32)
            text_token = np.concatenate(
                [
                    text_token,
                    np.array([self.audio_start_token], dtype=np.int32),
                ],
                axis=-1,
            )
            text_length = text_token.shape[0]
            audio_feat = np.zeros(
                (text_length, self.patch_size, self.audio_vae_latent_dim),
                dtype=np.float32,
            )
            text_mask = np.ones(text_length, dtype=np.int32)
            audio_mask = np.zeros(text_length, dtype=np.int32)

        else:
            text = prompt_text + target_text
            text_token = np.array(self.text_tokenizer(text), dtype=np.int32)
            text_token = np.concatenate(
                [
                    text_token,
                    np.array([self.audio_start_token], dtype=np.int32),
                ],
                axis=-1,
            )
            text_length = text_token.shape[0]

            audio, sr = librosa.load(prompt_wav_path, sr=None)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=0, keepdims=True)

            if sr != self.sample_rate:
                audio = librosa.resample(
                    y=audio, orig_sr=sr, target_sr=self.sample_rate
                )

            patch_len = self.patch_size * self.chunk_size
            if audio.shape[1] % patch_len != 0:
                pad_width = patch_len - audio.shape[1] % patch_len
                audio = np.pad(audio, ((0, 0), (0, pad_width)))

            audio_feat = self.audio_vae.encode(audio, self.sample_rate)
            audio_feat = audio_feat.reshape(
                self.audio_vae_latent_dim,
                -1,
                self.patch_size,
            ).transpose(1, 2, 0)
            audio_feat = audio_feat[:-1, ...]
            audio_length = audio_feat.shape[0]
            text_pad_token = np.zeros(audio_length, dtype=np.int32)
            text_token = np.concatenate([text_token, text_pad_token])
            audio_pad_feat = np.zeros(
                (text_length, self.patch_size, self.audio_vae_latent_dim),
                dtype=np.float32,
            )
            audio_feat = np.concatenate([audio_pad_feat, audio_feat], axis=0)
            text_mask = np.concatenate(
                [np.ones(text_length), np.zeros(audio_length)]
            ).astype(np.int32)
            audio_mask = np.concatenate(
                [np.zeros(text_length), np.ones(audio_length)]
            ).astype(np.int32)

        text_token = np.expand_dims(text_token, 0)
        text_mask = np.expand_dims(text_mask, 0)
        audio_feat = np.expand_dims(audio_feat, 0).astype(np.float32)
        audio_mask = np.expand_dims(audio_mask, 0)
        target_text_length = len(self.text_tokenizer(target_text))

        retry_badcase_times = 0
        while retry_badcase_times < retry_badcase_max_times:
            inference_result = self._inference_np(
                text_token,
                text_mask,
                audio_feat,
                audio_mask,
                min_len=min_len,
                max_len=(
                    int(target_text_length * retry_badcase_ratio_threshold + 10)
                    if retry_badcase
                    else max_len
                ),
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                streaming=streaming,
            )
            if streaming:
                patch_len = self.patch_size * self.chunk_size
                for latent_pred, _ in inference_result:
                    decode_audio = self.audio_vae.decode(latent_pred)
                    decode_audio = decode_audio[..., -patch_len:]
                    yield decode_audio
                break
            else:
                latent_pred, pred_audio_feat = next(inference_result)
                if retry_badcase:
                    if (
                        pred_audio_feat.shape[0]
                        >= target_text_length * retry_badcase_ratio_threshold
                    ):
                        print(
                            f"Badcase detected, audio_text_ratio={pred_audio_feat.shape[0] / target_text_length}, retrying..."
                        )
                        retry_badcase_times += 1
                        continue
                    else:
                        break
                else:
                    break

        if not streaming:
            decode_audio = self.audio_vae.decode(latent_pred.astype(np.float32))
            decode_audio = np.squeeze(decode_audio, 1)
            decode_audio = decode_audio[..., 640:-640]
            yield decode_audio

    def _generate_with_prompt_cache(
        self,
        target_text: str,
        prompt_cache: dict,
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        retry_badcase: bool = False,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        streaming: bool = False,
    ) -> Generator[
        Tuple[np.ndarray, np.ndarray, Union[np.ndarray, List[np.ndarray]]], None, None
    ]:
        if retry_badcase and streaming:
            warnings.warn(
                "Retry on bad cases is not supported in streaming mode, setting retry_badcase=False."
            )
            retry_badcase = False

        if prompt_cache is None:
            prompt_text_token = np.empty(0, dtype=np.int32)
            prompt_audio_feat = np.empty(
                (0, self.patch_size, self.audio_vae_latent_dim), dtype=np.float32
            )
        else:
            prompt_text_token = prompt_cache["text_token"]
            prompt_audio_feat = prompt_cache["audio_feat"]

        target_text_token = np.array(self.text_tokenizer(target_text), dtype=np.int32)
        text_token = np.concatenate([prompt_text_token, target_text_token], axis=0)
        text_token = np.concatenate(
            [
                text_token,
                np.array([self.audio_start_token], dtype=np.int32),
            ],
            axis=-1,
        )

        audio_length = prompt_audio_feat.shape[0]
        text_length = text_token.shape[0]
        text_pad_token = np.zeros(audio_length, dtype=np.int32)
        audio_pad_feat = np.zeros(
            (text_token.shape[0], self.patch_size, self.audio_vae_latent_dim),
            dtype=np.float32,
        )
        text_token = np.concatenate([text_token, text_pad_token])
        audio_feat = np.concatenate([audio_pad_feat, prompt_audio_feat], axis=0)
        text_mask = np.concatenate(
            [np.ones(text_length), np.zeros(audio_length)]
        ).astype(np.int32)
        audio_mask = np.concatenate(
            [np.zeros(text_length), np.ones(audio_length)]
        ).astype(np.int32)

        text_token = np.expand_dims(text_token, 0)
        text_mask = np.expand_dims(text_mask, 0)
        audio_feat = np.expand_dims(audio_feat, 0).astype(np.float32)
        audio_mask = np.expand_dims(audio_mask, 0)

        target_text_length = len(self.text_tokenizer(target_text))
        retry_badcase_times = 0
        while retry_badcase_times < retry_badcase_max_times:
            inference_result = self._inference_np(
                text_token,
                text_mask,
                audio_feat,
                audio_mask,
                min_len=min_len,
                max_len=(
                    int(target_text_length * retry_badcase_ratio_threshold + 10)
                    if retry_badcase
                    else max_len
                ),
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                streaming=streaming,
            )
            if streaming:
                patch_len = self.patch_size * self.chunk_size

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    decode_future = None
                    pred_audio_feat_prev = None
                    target_text_token_prev = None

                    for i, (latent_pred, pred_audio_feat) in enumerate(
                        inference_result
                    ):
                        if i == 0:
                            decode_future = executor.submit(
                                self.audio_vae_decoder,
                                latent_pred,
                                patch_len,
                                self.chunk_size,
                            )
                            pred_audio_feat_prev = pred_audio_feat
                            target_text_token_prev = target_text_token
                            continue

                        decode_audio = decode_future.result()
                        next_decode_future = executor.submit(
                            self.audio_vae_decoder,
                            latent_pred,
                            patch_len,
                            self.chunk_size,
                        )

                        if i > 1:
                            decode_audio = decode_audio[patch_len * 2 :]

                        yield (
                            decode_audio,
                            target_text_token_prev,
                            pred_audio_feat_prev,
                        )

                        decode_future = next_decode_future
                        pred_audio_feat_prev = pred_audio_feat
                        target_text_token_prev = target_text_token

                    if decode_future:
                        decode_audio = decode_future.result()
                        decode_audio = decode_audio[patch_len * 2 :]
                        yield (
                            decode_audio,
                            target_text_token_prev,
                            pred_audio_feat_prev,
                        )
                break
            else:
                latent_pred, pred_audio_feat = next(inference_result)
                if retry_badcase:
                    if (
                        pred_audio_feat.shape[0]
                        >= target_text_length * retry_badcase_ratio_threshold
                    ):
                        print(
                            f"Badcase detected, audio_text_ratio={pred_audio_feat.shape[0] / target_text_length}, retrying..."
                        )
                        retry_badcase_times += 1
                        continue
                    else:
                        break
                else:
                    break
        if not streaming:
            decode_audio = self.audio_vae.decode(latent_pred).squeeze(1)
            decode_audio = decode_audio[..., 640:-640]
            yield (decode_audio, target_text_token, pred_audio_feat)

    def _generate_threaded_prompt_processing(
        self,
        text,
        audio: Optional[np.ndarray] = None,
        audio_cache: Optional[np.ndarray] = None,
        max_length: int = 2048,
        min_len: int = 2,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        # streaming=True,
    ) -> Generator[
        # Tuple[np.ndarray, np.ndarray, Union[np.ndarray, List[np.ndarray]]], None, None
        Tuple[np.ndarray],
        None,
        None,
    ]:
        # assert (audio is not None) or (audio_cache is not None)

        generate_start_time = time.time()
        streaming = True

        text = np.concatenate(
            [
                text,
                np.array([[self.audio_start_token]], dtype=np.int32),
            ],
            axis=-1,
        )

        start = time.perf_counter()
        inference_result = self._process_prompt_fully_pipelined(
            text,
            audio,
            audio_cache=audio_cache,
            max_length=max_length,
            min_len=min_len,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            streaming=True,
        )
        if streaming:
            patch_len = self.patch_size * self.chunk_size

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                decode_future = None
                pred_audio_feat_prev = None
                target_text_token_prev = None

                for i, (latent_pred, pred_audio_feat, stop_flag) in enumerate(
                    inference_result
                ):
                    # if i == 0:
                    #     decode_future = executor.submit(
                    #         self.audio_vae_decoder,
                    #         latent_pred,
                    #         patch_len,
                    #         self.chunk_size,
                    #     )
                    #     pred_audio_feat_prev = pred_audio_feat
                    # target_text_token_prev = target_text_token
                    # continue

                    # tried threaded processing but i think it is currently not working
                    # will have to try later when python 3.14 is supported by coreml without gil
                    # next_decode_future = executor.submit(
                    decode_future = executor.submit(
                        self.audio_vae_decoder, latent_pred, patch_len, self.chunk_size
                    )
                    decode_audio = decode_future.result()

                    if i > 0:
                        decode_audio = decode_audio[patch_len * 2 :]
                    if stop_flag:
                        decode_audio = decode_audio[:-1280]

                    if generate_start_time is not None:
                        generate_start_time = None
                    yield (
                        decode_audio,
                        # target_text_token_prev,
                        # pred_audio_feat_prev
                    )

                    # decode_future = next_decode_future
                    # pred_audio_feat_prev = pred_audio_feat
                    # target_text_token_prev = target_text_token

                # if decode_future:
                #     print("decode future")
                #     decode_audio = decode_future.result()
                #     decode_audio = decode_audio[patch_len * 2 :-1280]
                #     yield (
                #         decode_audio,
                #         # target_text_token_prev,
                #         # pred_audio_feat_prev
                #     )
        else:
            latent_pred, pred_audio_feat = next(inference_result)
        if not streaming:
            decode_audio = self.audio_vae.decode(latent_pred).squeeze(1)
            decode_audio = decode_audio[..., 640:-640]
            yield (
                decode_audio,
                # target_text_token,
                # pred_audio_feat
            )

    def _inference_np(
        self,
        text: np.ndarray,
        text_mask: np.ndarray,
        feat: np.ndarray,
        feat_mask: np.ndarray,
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        streaming: bool = False,
    ) -> Generator[Tuple[np.ndarray, Union[np.ndarray, List[np.ndarray]]], None, None]:
        B, T, P, D = feat.shape
        feat = feat.transpose(0, 3, 2, 1).astype(np.float32)
        text_mask: np.ndarray = np.expand_dims(text_mask, axis=(1, 2)).astype(
            np.float32
        )
        feat_mask: np.ndarray = np.expand_dims(feat_mask, axis=(1, 2)).astype(
            np.float32
        )
        feat_embed = self.feat_encoder(feat)

        if self.use_mup:
            scale_emb = self.scale_emb
        else:
            scale_emb = 1.0

        text_embed = self.embed_tokens[text] * scale_emb
        text_embed = rearrange(text_embed, "b t d -> b d 1 t")
        combined_embed = text_mask * text_embed + feat_mask * feat_embed

        prefix_feat_cond = feat[:, ..., -1:]
        pred_feat_seq = []

        enc_outputs, _ = self.base_lm(
            inputs_embeds=combined_embed,
            is_causal=True,
        )
        enc_outputs = self.fsq_layer(enc_outputs) * feat_mask + enc_outputs * text_mask
        lm_hidden = enc_outputs[:, ..., -1:]

        residual_enc_outputs, _ = self.residual_lm(
            inputs_embeds=enc_outputs + feat_mask * feat_embed,
            is_causal=True,
        )
        residual_hidden = residual_enc_outputs[:, ..., -1:]

        for i in range(max_len):
            dit_hidden, stop_flag = self.lm_to_dit_proj(lm_hidden, residual_hidden)

            pred_feat = self.feat_decoder(
                mu=dit_hidden,
                patch_size=self.patch_size,
                cond=prefix_feat_cond.transpose(0, 3, 1, 2),
                n_timesteps=inference_timesteps,
                cfg_value=cfg_value,
            )

            curr_embed = self.feat_encoder(pred_feat)
            pred_feat_seq.append(pred_feat.transpose(0, 3, 2, 1))
            prefix_feat_cond = pred_feat

            if streaming and len(pred_feat_seq) == 12:
                pred_feat_chunk = np.concatenate(pred_feat_seq, axis=1)
                feat_pred = rearrange(
                    pred_feat_chunk, "b t p d -> b d (t p)", b=B, p=self.patch_size
                )
                pred_feat_seq = pred_feat_seq[-2:]
                yield feat_pred, pred_feat_seq

            stop_flag = stop_flag.item()
            if i > min_len and stop_flag == 1:
                if len(pred_feat_seq) > 2:
                    pred_feat_chunk = np.concatenate(pred_feat_seq, axis=1)
                    feat_pred = rearrange(
                        pred_feat_chunk, "b t p d -> b d (t p)", b=B, p=self.patch_size
                    )
                    yield feat_pred, pred_feat_seq

                break

            lm_hidden = self.base_lm.forward_step(
                curr_embed,
                None,
            )

            lm_hidden = self.fsq_layer(lm_hidden)

            start_time = time.perf_counter()
            residual_hidden = self.residual_lm.forward_step(
                lm_hidden + curr_embed,
                None,
            )

        if not streaming:
            pred_feat_seq = np.concatenate(pred_feat_seq, axis=1)
            feat_pred = rearrange(
                pred_feat_seq, "b t p d -> b d (t p)", b=B, p=self.patch_size
            )
            yield feat_pred, np.squeeze(pred_feat_seq, 0)

    @classmethod
    def from_local(
        cls,
        path: str,
        # base_lm_mlmodel: ct.models.MLModel,
        base_lm_mlmodel_path: str,
        residual_lm_mlmodel_path: str,
        fsq_mlmodel: ct.models.MLModel,
        locdit_mlmodel: ct.models.MLModel,
        projections_mlmodel: ct.models.MLModel,
        audio_vae_decoder_mlmodel: ct.models.MLModel,
        audio_vae_encoder_mlmodel: ct.models.MLModel,
        feature_encoder_mlmodel: ct.models.MLModel,
        base_lm_embed_tokens: np.ndarray,
        vae_encoder_hop_length: int,
        base_lm_chunk_size: int,
        residual_lm_chunk_size: int,
        audio_vae_encoder_chunk_size: int,
        feature_encoder_chunk_size: int,
        fsq_layer_chunk_size: int,
    ) -> "VoxCPMModelANE":
        tokenizer = LlamaTokenizerFast.from_pretrained(path)
        model = cls(
            tokenizer,
            base_lm_mlmodel_path,
            residual_lm_mlmodel_path,
            fsq_mlmodel,
            locdit_mlmodel,
            projections_mlmodel,
            audio_vae_decoder_mlmodel,
            audio_vae_encoder_mlmodel,
            feature_encoder_mlmodel,
            base_lm_embed_tokens,
            vae_encoder_hop_length,
            base_lm_chunk_size=base_lm_chunk_size,
            residual_lm_chunk_size=residual_lm_chunk_size,
            audio_vae_encoder_chunk_size=audio_vae_encoder_chunk_size,
            feature_encoder_chunk_size=feature_encoder_chunk_size,
            fsq_layer_chunk_size=fsq_layer_chunk_size,
        )
        return model

    def build_prompt_cache(
        self,
        prompt_text: str,
        prompt_wav_path: str,
    ):
        start = time.time()

        if not prompt_text or not prompt_wav_path:
            raise ValueError("prompt_text and prompt_wav_path are required")

        text_token = np.array(self.text_tokenizer(prompt_text), dtype=np.int32)

        # audio, sr = librosa.load(prompt_wav_path, sr=None, mono=False)
        audio, sr = soundfile.read(prompt_wav_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1, keepdims=False)
        # else:
        #     audio = np.expand_dims(audio, 0)

        if sr != self.sample_rate:
            resample_start = time.perf_counter()
            # audio = librosa.resample(y=audio, orig_sr=sr, target_sr=self.sample_rate)
            # audio = signal.resample_poly(audio, up=self.sample_rate, down=sr, axis=1)
            # audio = signal.resample_poly(audio, self.sample_rate, sr,
            #                        window=('kaiser', 8.0),
            #                        padtype='constant', cval=0.0, axis=1)
            audio = soxr.resample(audio, sr, self.sample_rate, quality="HQ")[None, :]

        patch_len = self.patch_size * self.chunk_size
        if audio.shape[1] % patch_len != 0:
            pad_width = patch_len - audio.shape[1] % patch_len
            audio = np.pad(audio, ((0, 0), (0, pad_width)))

        audio_feat = self.audio_vae_encoder(audio, self.sample_rate)

        audio_feat = audio_feat.reshape(
            self.audio_vae_latent_dim,
            -1,
            self.patch_size,
        ).transpose(1, 2, 0)
        audio_feat = audio_feat[:-1, ...]

        prompt_cache = {
            "text_token": text_token,
            "audio_feat": audio_feat,
        }

        return prompt_cache

    def load_preprocessing_models(self):
        if self.base_lm is None:
            base_lm = ct.models.CompiledMLModel(
                self.base_lm_model_path,
                compute_units=ct.ComputeUnit.CPU_AND_NE,
                function_name=f"length_{self.base_lm_chunk_size}",
            )
            self.base_lm = BaseLMANEWrapperWithCache(
                self.embed_tokens, base_lm, self.base_lm_chunk_size
            )
        if self.residual_lm is None:
            residual_lm = ct.models.CompiledMLModel(
                self.residual_lm_model_path,
                compute_units=ct.ComputeUnit.CPU_AND_NE,
                function_name=f"length_{self.residual_lm_chunk_size}",
            )
            self.residual_lm = BaseLMANEWrapperWithCache(
                None, residual_lm, self.residual_lm_chunk_size
            )

    def load_inference_models(self, event: threading.Event = None):
        if event is not None:
            event.wait()
        if self.base_lm_inference is None:
            base_lm_inference = ct.models.CompiledMLModel(
                self.base_lm_model_path,
                compute_units=ct.ComputeUnit.CPU_AND_NE,
                function_name=f"length_1",
            )
            self.base_lm_inference = BaseLMANEWrapperWithCache(
                self.embed_tokens, base_lm_inference, 1
            )
        if self.residual_lm_inference is None:
            residual_lm_inference = ct.models.CompiledMLModel(
                self.residual_lm_model_path,
                compute_units=ct.ComputeUnit.CPU_AND_NE,
                function_name=f"length_1",
            )
            self.residual_lm_inference = BaseLMANEWrapperWithCache(
                None, residual_lm_inference, 1
            )

        return True

    def _process_prompt_fully_pipelined(
        self,
        text_token: np.ndarray,
        audio_data: Optional[np.ndarray] = None,
        audio_cache: Optional[np.ndarray] = None,
        max_length: int = 2048,
        min_len: int = 2,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        streaming=True,
    ) -> Generator[
        Tuple[np.ndarray, np.ndarray, Union[np.ndarray, List[np.ndarray]]], None, None
    ]:
        B = 1  # Batch size

        prompt_start_time = time.perf_counter()
        q_audio_vae_out = queue.Queue(maxsize=32)
        q_final_state = queue.Queue(maxsize=1)

        def _text_producer():
            self.load_preprocessing_models()

            self.base_lm.reset_state()
            self.residual_lm.reset_state()
            try:
                B, text_len = text_token.shape
                if text_len == 0:
                    return
                scale_emb = self.scale_emb if self.use_mup else 1.0
                text_embed = self.embed_tokens[text_token] * scale_emb
                text_embed = rearrange(text_embed, "b t d -> b d 1 t")

                base_lm_output, _ = self.base_lm(text_embed, reset_state=False)
                residual_lm_output, _ = self.residual_lm(
                    base_lm_output, reset_state=False
                )

                fsq_output = np.zeros_like(base_lm_output[..., -1:])
                _feat_chunk = np.zeros((1, 64, 2, 1))

                keep_iter = True
                remainder_chunk = None

                while keep_iter:
                    if remainder_chunk is not None:
                        feat_chunk = remainder_chunk
                        remainder_chunk = None
                    else:
                        feat_chunk = q_audio_vae_out.get()

                    if feat_chunk is None:
                        feat_chunk = _feat_chunk
                        break

                    feat_chunks = [feat_chunk]

                    current_length_size = feat_chunk.shape[-1]
                    feat_embed_chunk = self.feat_encoder(feat_chunk)
                    feat_embed_chunks = [feat_embed_chunk]

                    try:
                        while True:
                            # we'll try to feed the base lm and residual lm in chunks of size 32 (or 64),
                            # the audio vae encoder and feat_encoder generate in chunks of size 16,
                            # so we try to run them twice, but we do not wait if we only have a single chunk of 16 available
                            maybe_next_feat_chunk = q_audio_vae_out.get_nowait()
                            if maybe_next_feat_chunk is None:
                                keep_iter = False
                                break
                            else:
                                if (
                                    current_length_size
                                    + maybe_next_feat_chunk.shape[-1]
                                ) > self.base_lm_chunk_size:
                                    remainder_chunk = maybe_next_feat_chunk
                                    break
                                else:
                                    current_length_size += maybe_next_feat_chunk.shape[
                                        -1
                                    ]
                                    feat_chunk = maybe_next_feat_chunk
                                    feat_chunks.append(feat_chunk)
                                    feat_embed_chunk = self.feat_encoder(feat_chunk)
                                    feat_embed_chunks.append(feat_embed_chunk)
                    except queue.Empty:
                        pass

                    feat_chunk = np.concatenate(feat_chunks, axis=-1)
                    feat_embed_chunk = np.concatenate(feat_embed_chunks, axis=-1)

                    base_lm_output, _ = self.base_lm(
                        feat_embed_chunk, reset_state=False
                    )
                    fsq_output = self.fsq_layer(base_lm_output)
                    residual_lm_output, _ = self.residual_lm(
                        fsq_output + feat_embed_chunk, reset_state=False
                    )

            finally:
                last_lm_hidden = fsq_output[..., -1:]
                last_residual_hidden = residual_lm_output[..., -1:]
                last_prefix_feat_cond = feat_chunk[..., -1:]
                q_final_state.put(
                    (last_lm_hidden, last_residual_hidden, last_prefix_feat_cond)
                )

        def _audio_producer():
            try:
                if audio_cache is not None:
                    FEAT_ENCODER_CHUNK_SIZE = 16
                    length = audio_cache.shape[-1]
                    num_steps = math.ceil(length / FEAT_ENCODER_CHUNK_SIZE)
                    for i in range(num_steps):
                        q_audio_vae_out.put(
                            audio_cache[
                                ...,
                                i
                                * FEAT_ENCODER_CHUNK_SIZE : (i + 1)
                                * FEAT_ENCODER_CHUNK_SIZE,
                            ]
                        )
                    return

                if audio_data is None or audio_data.shape[1] == 0:
                    return
                audio_data_processed = self.audio_vae_encoder.preprocess(
                    audio_data, self.sample_rate
                )

                length = audio_data_processed.shape[-1]
                step_size = self.audio_vae_encoder.step_size
                chunk_size = self.audio_vae_encoder.chunk_size
                # chunk_size = 17920
                # step_size = chunk_size - 2560
                num_steps = math.ceil(length / step_size)
                if num_steps == 0:
                    return
                for i in range(num_steps):
                    start, end = i * step_size, i * step_size + chunk_size
                    audio_chunk = audio_data_processed[..., start:end]
                    if audio_chunk.shape[-1] < chunk_size:
                        pad_width = chunk_size - audio_chunk.shape[-1]
                        assert (pad_width % (self.patch_size * self.chunk_size)) == 0
                        audio_chunk = np.pad(
                            audio_chunk, ((0, 0), (0, 0), (0, pad_width))
                        )
                    else:
                        pad_width = 0
                    vae_latent = self.audio_vae_encoder.mlmodel.predict(
                        {"x": audio_chunk}
                    )["output"]
                    if pad_width > 0:
                        vae_latent = vae_latent[..., : pad_width // self.chunk_size]
                    vae_latent = vae_latent.reshape(
                        1, self.audio_vae_latent_dim, -1, self.patch_size
                    )
                    vae_latent = vae_latent.transpose(0, 1, 3, 2)
                    if i > 0:
                        vae_latent = [vae_latent[..., 2:]]
                    else:
                        vae_latent = [vae_latent[..., :2], vae_latent[..., 2:]]

                    for chunk in vae_latent:
                        q_audio_vae_out.put(chunk)
            finally:
                q_audio_vae_out.put(None)

        text_workers = [
            threading.Thread(target=_text_producer),
        ]
        audio_workers = [
            threading.Thread(target=_audio_producer),
        ]
        for t in text_workers + audio_workers:
            t.start()

        lm_hidden, residual_hidden, prefix_feat_cond = q_final_state.get()

        for t in text_workers + audio_workers:
            t.join()

        pred_feat_seq = []
        is_first_iter = True

        max_len = min(max_length, STATE_MAX_LENGTH - self.base_lm.current_position)

        for i in range(max_len):
            dit_hidden, stop_flag = self.lm_to_dit_proj(lm_hidden, residual_hidden)

            pred_feat = self.feat_decoder(
                mu=dit_hidden,
                patch_size=self.patch_size,
                cond=prefix_feat_cond.transpose(0, 3, 1, 2),
                n_timesteps=inference_timesteps,
                cfg_value=cfg_value,
            )

            curr_embed = self.feat_encoder(pred_feat)
            pred_feat_seq.append(pred_feat.transpose(0, 3, 2, 1))
            prefix_feat_cond = pred_feat

            stop = (stop_flag.item() == 1) and (i > min_len)
            last_iter = i == (max_len - 1)
            # slightly shorter first iter for faster ttft
            # if streaming and (len(pred_feat_seq) == 12 or (len(pred_feat_seq) == 7 and is_first_iter)):
            if streaming and (len(pred_feat_seq) == 12 or stop or last_iter):
                is_first_iter = False
                pred_feat_chunk = np.concatenate(pred_feat_seq, axis=1)
                feat_pred = rearrange(
                    pred_feat_chunk, "b t p d -> b d (t p)", b=B, p=self.patch_size
                )
                pred_feat_seq = pred_feat_seq[-2:]
                yield feat_pred, pred_feat_seq, stop

                if stop or last_iter:
                    break

            # if i > min_len and stop_flag == 1:
            #     if len(pred_feat_seq) > 2:
            #         pred_feat_chunk = np.concatenate(pred_feat_seq, axis=1)
            #         feat_pred = rearrange(
            #             pred_feat_chunk, "b t p d -> b d (t p)", b=B, p=self.patch_size
            #         )
            #         yield feat_pred, pred_feat_seq
            #     break

            lm_hidden = self.base_lm.forward_step(
                curr_embed,
                None,
            )
            lm_hidden = self.fsq_layer(lm_hidden)

            residual_hidden = self.residual_lm.forward_step(
                lm_hidden + curr_embed,
                None,
            )


class VoxCPMANE:
    def __init__(
        self,
        voxcpm_model_path: str,
        base_lm_mlmodel: ct.models.MLModel,
        fsq_mlmodel: ct.models.MLModel,
        locdit_mlmodel: ct.models.MLModel,
        projections_mlmodel: ct.models.MLModel,
        audio_vae_decoder_mlmodel: ct.models.MLModel,
        audio_vae_encoder_mlmodel: ct.models.MLModel,
        feature_encoder_mlmodel: ct.models.MLModel,
        residual_lm_mlmodel: ct.models.MLModel,
        base_lm_embed_tokens: np.ndarray,
        zipenhancer_model_path: str = "iic/speech_zipenhancer_ans_multiloss_16k_base",
        enable_denoiser: bool = True,
        vae_encoder_hop_length=640,
        optimize: bool = True,
        base_lm_chunk_size=4,
        residual_lm_chunk_size=4,
        audio_vae_encoder_chunk_size=15_360,
        feature_encoder_chunk_size=12,
        fsq_layer_chunk_size=32,
    ):
        print(
            f"voxcpm_model_path: {voxcpm_model_path}, zipenhancer_model_path: {zipenhancer_model_path}, enable_denoiser: {enable_denoiser}"
        )
        self.tts_model = VoxCPMModelANE.from_local(
            voxcpm_model_path,
            base_lm_mlmodel,
            fsq_mlmodel,
            locdit_mlmodel,
            projections_mlmodel,
            audio_vae_decoder_mlmodel,
            audio_vae_encoder_mlmodel,
            feature_encoder_mlmodel,
            residual_lm_mlmodel,
            base_lm_embed_tokens,
            vae_encoder_hop_length=vae_encoder_hop_length,
            base_lm_chunk_size=base_lm_chunk_size,
            residual_lm_chunk_size=residual_lm_chunk_size,
            audio_vae_encoder_chunk_size=audio_vae_encoder_chunk_size,
            feature_encoder_chunk_size=feature_encoder_chunk_size,
            fsq_layer_chunk_size=fsq_layer_chunk_size,
        )
        self.text_normalizer = TextNormalizer()
        if enable_denoiser and zipenhancer_model_path is not None:
            # from voxcpm.zipenhancer import ZipEnhancer

            # self.denoiser = ZipEnhancer(zipenhancer_model_path)
            print("Denoiser currently not supported")
        else:
            self.denoiser = None

    def _generate(
        self,
        text: str,
        prompt_wav_path: str = None,
        prompt_text: str = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        max_length: int = 4096,
        normalize: bool = True,
        # denoise : bool = True,
        denoise: bool = False,
        # retry_badcase : bool = True,
        retry_badcase: bool = False,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        streaming: bool = False,
    ) -> Generator[np.ndarray, None, None]:

        if not text.strip() or not isinstance(text, str):
            raise ValueError("target text must be a non-empty string")

        if prompt_wav_path is not None:
            if not os.path.exists(prompt_wav_path):
                raise FileNotFoundError(
                    f"prompt_wav_path does not exist: {prompt_wav_path}"
                )

        if (prompt_wav_path is None) != (prompt_text is None):
            raise ValueError(
                "prompt_wav_path and prompt_text must both be provided or both be None"
            )

        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        temp_prompt_wav_path = None

        try:
            if prompt_wav_path is not None and prompt_text is not None:
                if denoise and self.denoiser is not None:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".wav"
                    ) as tmp_file:
                        temp_prompt_wav_path = tmp_file.name
                    self.denoiser.enhance(
                        prompt_wav_path, output_path=temp_prompt_wav_path
                    )
                    prompt_wav_path = temp_prompt_wav_path
                prompt_cache_start_time = time.time()
                fixed_prompt_cache = self.tts_model.build_prompt_cache(
                    # fixed_prompt_cache = self.tts_model.build_prompt_cache_parallel(
                    prompt_wav_path=prompt_wav_path,
                    prompt_text=prompt_text,
                )
            else:
                fixed_prompt_cache = None

            if normalize:
                text = self.text_normalizer.normalize(text)

            generate_result = self.tts_model._generate_with_prompt_cache(
                target_text=text,
                prompt_cache=fixed_prompt_cache,
                min_len=2,
                max_len=max_length,
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                retry_badcase=retry_badcase,
                retry_badcase_max_times=retry_badcase_max_times,
                retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                streaming=streaming,
            )

            for wav, _, _ in generate_result:
                yield wav

        finally:
            if temp_prompt_wav_path and os.path.exists(temp_prompt_wav_path):
                try:
                    os.unlink(temp_prompt_wav_path)
                except OSError:
                    pass

    def generate(self, *args, **kwargs) -> np.ndarray:
        return next(self._generate(*args, streaming=False, **kwargs))

    def generate_streaming(self, *args, **kwargs) -> Generator[np.ndarray, None, None]:
        return self._generate(*args, streaming=True, **kwargs)

    def create_custom_voice(
        self,
        voice_name: str,
        prompt_wav_path: str,
        prompt_text: str,
        cache_dir: str,
    ):
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        # Generate cache using the model
        cache = self.tts_model.build_prompt_cache(
            prompt_text=prompt_text,
            prompt_wav_path=prompt_wav_path,
        )

        # Extract audio features (the .npy content)
        audio_feat = cache["audio_feat"]
        audio_feat = rearrange(audio_feat, 't p d -> 1 d p t')

        # Paths
        npy_path = os.path.join(cache_dir, f"{voice_name}.npy")
        txt_path = os.path.join(cache_dir, f"{voice_name}.txt")

        # Save
        np.save(npy_path, audio_feat)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(prompt_text)

        return npy_path, txt_path
