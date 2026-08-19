"""Text normalization for the Bangla dataset pipeline.

Steps: unicode normalization -> HTML/markup stripping -> whitespace cleanup.
All functions are pure and operate on ``str``.
"""

from __future__ import annotations

import html
import re
import unicodedata

from .punctuation import convert_digits, normalize_bangla_punctuation

__all__ = [
    "clean_whitespace",
    "fix_bengali_yaphala",
    "nfkc_latin_only",
    "normalize_text",
    "strip_html",
    "strip_markup",
    "unicode_normalize",
]

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.DOTALL | re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]+`")
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_HRULE_RE = re.compile(r"^\s*([-*_])\1{2,}\s*$", re.MULTILINE)
_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?", re.MULTILINE)
_LIST_MARKER_RE = re.compile(r"^\s*([-*+]|\d+[.)])\s+", re.MULTILINE)

# Variation selectors, zero-width space, soft hyphen and other invisible codepoints.
# The zero-width joiner (U+200D) is deliberately kept: it is meaningful in Bangla
# conjunct formation.
_VARIATION_SELECTOR_RE = re.compile("[\ufe00-\ufe0f\U000e0100-\U000e01ef]")
_ZERO_WIDTH_RE = re.compile("[\u200b\u200e\u200f\u2060\ufeff]")

_NBSP_CHARS = ("\u00a0", "\u202f", "\u2007", "\u2009")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Bangla consonants (ক-হ plus the extended forms রড র্ড় ঢ় য়) followed by a
# zero-width joiner and য. Typed Bengali sometimes uses ZWJ to force the ya-phala
# conjunct (e.g. ক‍্য = kya); the canonical NFC encoding is hasanta (U+09CD) + য.
_ZWJ_YAPHALA_RE = re.compile("([\u0995-\u09b9\u09dc\u09dd\u09df])\u200d\u09af")


def fix_bengali_yaphala(text: str) -> str:
    """Normalize ZWJ-forced ya-phala sequences to hasanta + য.

    Converts ``<consonant> ZWJ য`` -> ``<consonant> ্ য`` so that conjuncts
    use the canonical hasanta form and NFC round-trips cleanly.
    """
    return _ZWJ_YAPHALA_RE.sub(lambda m: m.group(1) + "\u09cd\u09af", text)


_LATIN_RUN_RE = re.compile(
    "[\u0041-\u007a\u00c0-\u02af\u1e00-\u1eff\u0300-\u036f\ufb00-\ufb06\uff21-\uff5a]"
)


def nfkc_latin_only(text: str) -> str:
    """Apply NFKC only to Latin-script runs, leaving Bangla text intact.

    NFKC across the whole document would decompose Bangla conjuncts; restricting
    it to Latin segments lets English loanwords and punctuation (e.g. fullwidth
    forms, ligatures like ``ﬁ``) normalise without touching Bengali glyphs.
    """
    out: list[str] = []
    buf: list[str] = []
    for ch in text:
        if _LATIN_RUN_RE.match(ch):
            buf.append(ch)
        else:
            if buf:
                out.append(unicodedata.normalize("NFKC", "".join(buf)))
                buf = []
            out.append(ch)
    if buf:
        out.append(unicodedata.normalize("NFKC", "".join(buf)))
    return "".join(out)


def unicode_normalize(text: str, form: str = "NFC") -> str:
    """Apply Unicode normalization and drop invisible/special characters.

    Uses NFC by default so Bangla conjuncts and composed vowel signs survive
    unchanged. NFKC may be requested explicitly for Latin-heavy text.
    """
    text = unicodedata.normalize(form, text)
    text = _VARIATION_SELECTOR_RE.sub("", text)
    # Zero-width joiner is meaningful in Bangla (conjuncts) so it is kept;
    # zero-width space and the BOM are removed.
    text = _ZERO_WIDTH_RE.sub("", text)
    for ch in _NBSP_CHARS:
        text = text.replace(ch, " ")
    text = _CONTROL_CHARS.sub("", text)
    return text


def strip_html(text: str) -> str:
    """Remove HTML/XML markup, comments and script/style bodies."""
    text = _SCRIPT_STYLE_RE.sub(" ", text)
    text = _HTML_COMMENT_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return text.strip()


def strip_markup(text: str) -> str:
    """Remove common Markdown/lightweight markup syntax."""
    text = _CODE_FENCE_RE.sub(" ", text)
    text = _INLINE_CODE_RE.sub(" ", text)
    text = _IMAGE_RE.sub(" ", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _HEADING_RE.sub("", text)
    text = _HRULE_RE.sub("", text)
    text = _BLOCKQUOTE_RE.sub("", text)
    text = _LIST_MARKER_RE.sub("", text)
    return text


def clean_whitespace(text: str, strip_lines: bool = True) -> str:
    """Collapse runs of whitespace and normalise newlines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u2028", "\n")
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() if strip_lines else text


def normalize_text(
    text: str,
    *,
    unicode_form: str = "NFC",
    remove_html: bool = True,
    remove_markup: bool = True,
    collapse_whitespace: bool = True,
    fix_yaphala: bool = True,
    normalize_punctuation: bool = True,
    digit_style: str = "keep",
) -> str:
    """Run the full normalization stack on a single text.

    ``digit_style`` controls number style: ``"keep"`` (default), ``"bengali"``
    or ``"ascii"``. ``normalize_punctuation`` applies the Bangla punctuation
    rules (danda/quotes/dashes/ellipsis) from ``punctuation``.
    """
    text = unicode_normalize(text, unicode_form)
    if fix_yaphala:
        text = fix_bengali_yaphala(text)
    if remove_html:
        text = strip_html(text)
    if remove_markup:
        text = strip_markup(text)
    if normalize_punctuation:
        text = normalize_bangla_punctuation(text)
    if digit_style != "keep":
        text = convert_digits(text, target=digit_style)
    if collapse_whitespace:
        text = clean_whitespace(text)
    return text.strip()


def tokenize_words(text: str) -> list[str]:
    """Split text into whitespace-delimited tokens (approximation only)."""
    return text.split()
