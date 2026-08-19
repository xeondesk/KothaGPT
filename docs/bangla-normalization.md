# Bangla Normalization Policy (WS-2 / WS-3)

Canonical normalization rules for Bangla text, implemented in
`data/pipeline/normalize.py` and `data/pipeline/punctuation.py`.

## Unicode form

- Default **NFC** — Bangla conjuncts and composed vowel signs (ে, ো, ৈ, ৌ, ৃ)
  survive unchanged. NFC handles the standard composed forms automatically.
- **NFKC is NOT applied to Bangla text** — it decomposes conjuncts and can
  corrupt glyphs. Use `nfkc_latin_only`-style logic for Latin segments only.
- Zero-width joiner (U+200D) is **kept** (meaningful for conjuncts).
- Zero-width space (U+200B), BOM (U+FEFF), RTL/LTR marks (U+200E/200F) and
  variation selectors (U+FE00..U+FE0F) are removed.

## Bangla-specific fixes

- **Ya-phala (WS-2):** `consonant + ZWJ + য` (typed form) normalizes to
  `consonant + ্ + য` (canonical hasanta form) via `fix_bengali_yaphala`.
  Covers ক-হ plus রড় র্ড় ঢ় য়.
- **Digit policy:** Bengali (০-৯) and Arabic-Indic (٠-٩) digits can be mapped
  to ASCII (default) or back to Bengali via `convert_digits(text, target)`.
  Pass `digit_style="ascii"|"bengali"` to `normalize_text`.

## Punctuation (WS-3) — canonical forms

| Source | Canonical |
| --- | --- |
| ॥ (U+0965) and runs of । | single danda । (U+0964) |
| whitespace before । | removed |
| । directly followed by a letter | । + space |
| “ ” „ « » | `"` |
| ‘ ’ ‚ | `'` |
| – — (en/em dash) | `-` |
| … (U+2026) | `...` |

## Open questions

- **Reph (রেফ) reordering:** ে + ্ + র is the Unicode NFC canonical form and is
  *not* reordered (reordering would break rendering). If collation-time
  reordering is needed later, add it as an explicit opt-in transform and keep it
  out of the default normalization path.
- **Vowel-sign reordering (েয etc.):** left as NFC; revisit if evaluation shows
  tokenizer regressions on orthographic-variant text.