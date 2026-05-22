import re
import regex
import inflect
from wetext import Normalizer

chinese_char_pattern = re.compile(r"[\u4e00-\u9fff]+")


# whether contain chinese character
def contains_chinese(text):
    return bool(chinese_char_pattern.search(text))


# replace special symbol
def replace_corner_mark(text):
    text = text.replace("²", "平方")
    text = text.replace("³", "立方")
    text = text.replace("√", "根号")
    text = text.replace("≈", "约等于")
    text = text.replace("<", "小于")
    return text


# remove meaningless symbol
def remove_bracket(text):
    text = text.replace("（", " ").replace("）", " ")
    text = text.replace("【", " ").replace("】", " ")
    text = text.replace("`", "").replace("`", "")
    text = text.replace("——", " ")
    return text


# spell Arabic numerals
def spell_out_number(text: str, inflect_parser):
    new_text = []
    st = None
    for i, c in enumerate(text):
        if not c.isdigit():
            if st is not None:
                num_str = inflect_parser.number_to_words(text[st:i])
                new_text.append(num_str)
                st = None
            new_text.append(c)
        else:
            if st is None:
                st = i
    if st is not None and st < len(text):
        num_str = inflect_parser.number_to_words(text[st:])
        new_text.append(num_str)
    return "".join(new_text)


# remove blank between chinese character
def replace_blank(text: str):
    out_str = []
    for i, c in enumerate(text):
        if c == " ":
            if (
                0 < i < len(text) - 1
                and text[i + 1].isascii()
                and text[i + 1] != " "
                and text[i - 1].isascii()
                and text[i - 1] != " "
            ):
                out_str.append(c)
        else:
            out_str.append(c)
    return "".join(out_str)


def clean_markdown(md_text: str) -> str:
    md_text = re.sub(r"```.*?```", "", md_text, flags=re.DOTALL)
    md_text = re.sub(r"`[^`]*`", "", md_text)
    md_text = re.sub(r"!\[[^\]]*\]\([^\)]+\)", "", md_text)
    md_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", md_text)
    md_text = re.sub(r"^(\s*)-\s+", r"\1", md_text, flags=re.MULTILINE)
    md_text = re.sub(r"<[^>]+>", "", md_text)
    md_text = re.sub(r"^#{1,6}\s*", "", md_text, flags=re.MULTILINE)
    return re.sub(r"\n\s*\n", "\n", md_text).strip()


def clean_text(text):
    text = clean_markdown(text)
    text = regex.compile(
        r"\p{Emoji_Presentation}|\p{Emoji}\uFE0F", flags=regex.UNICODE
    ).sub("", text)
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.replace('"', "“")
    return text


class TextNormalizer:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.zh_tn_model = Normalizer(lang="zh", operator="tn", remove_erhua=True)
        self.en_tn_model = Normalizer(lang="en", operator="tn")
        self.inflect_parser = inflect.engine()

    def normalize(self, text, split=False):
        # 去除 Markdown 语法，去除表情符号，去除换行符
        lang = "zh" if contains_chinese(text) else "en"
        text = clean_text(text)
        if lang == "zh":
            text = text.replace(
                "=", "等于"
            )  # 修复 ”550 + 320 等于 870 千卡。“ 被错误正则为 ”五百五十加三百二十等于八七十千卡.“
            if re.search(r"([\d$%^*_+≥≤≠×÷?=])", text):  # 避免 英文连字符被错误正则为减
                text = re.sub(
                    r"(?<=[a-zA-Z0-9])-(?=\d)", " - ", text
                )  # 修复 x-2 被正则为 x负2
            text = self.zh_tn_model.normalize(text)
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = remove_bracket(text)
        else:
            text = self.en_tn_model.normalize(text)
            text = spell_out_number(text, self.inflect_parser)
        return text if split is False else [text]
