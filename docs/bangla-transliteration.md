# Bangla Transliteration Policy (WS-5)

Deterministic, rule-based Bangla <-> Latin transliteration in
`ml/tokenizer/transliterate.py`. Entry points:

- `latin_to_bangla(text)` — romanized Bangla -> Bangla script (Avro-style).
- `bangla_to_latin(text)` — Bangla script -> readable Latin (lossy).

The CLI `encode` command accepts `--transliterate` to convert romanized input
before tokenization:
`python -m ml.tokenizer.cli encode --tokenizer DIR --text "ami bangla valobasi" --transliterate`.

## Rules

- **Longest-match table:** digraphs (`kh`, `gh`, `chh`, `ng`, `sh`, `th`,
  `dh`, `bh`, `ph`, `Th`, `Dh`) match before single letters.
- **Vowels:** `a i u e o` -> matras (া ি ু ে ো) after a consonant; the same
  letters -> standalone vowels (আ ই উ এ অ) at word start / after non-consonant.
- **Consonants:** a consonant followed by another consonant or a boundary gets
  a hasanta (্), producing conjuncts (`kkha` -> ক্‌খা).
- **Signs:** `ng` -> anusvara ং (the common romanized spelling, e.g. `bangla`),
  and signs do not force a hasanta on the next letter.
- **Digits, spaces and ASCII punctuation are preserved** and passed through.

## Ambiguity defaults (documented limitations)

Bangla has more letters than Latin, so several spellings are shared. The
deterministic defaults are:

| Latin | Bangla | Note |
| --- | --- | --- |
| `t` / `T` | ত / ট | lowercase = dental, capital = retroflex (ISO-15919 style) |
| `th` / `Th` | থ / ঠ | as above (`kotha` -> কোথা) |
| `d` / `D` | দ / ড | as above |
| `n` / `N` | ন / ণ | as above |
| `sh` / `S` | শ / ষ | `s` = স |
| `y` / `Y` | য / য় | |
| `ng` | ং | anusvara default |

This is a **generic** transliterator, not full Avro: context-sensitive choices
(e.g. `shokal` -> সকাল vs শোকাল, implicit অ between consonants) are not
attempted. Full Avro accuracy is out of scope for WS-5.

## Coverage targets

- Golden fixture: >= 500 romanization pairs at >= 95% accuracy (fixture lives
  in `tests/fixtures/`; seed set curated during WS-8).
- `unk` on transliterated input should match plain-Bangla levels once the
  vocabulary is frozen (WS-6).