# Decision — Tokenizer & Vocabulary Freeze

- **date**: 2026-08-20T01:46:42
- **algorithm**: `bpe` (vocab 16,000) — canonical production vocab
- **corpus digest**: `7bf1a6e740e3`

## Why this tokenizer/vocab

- Trained on the normalized Bangla corpus (33,319 docs of 133,275, stride 4).
- Coverage: **98.48%** of script characters on a 5,000-doc sample; unk rate 0.00%.
- Efficiency gate passes: tpc 0.3361 (<= 2.0), unk 0.00% (<= 0.005), fidelity 100% (>= 1.0).

## Vocab-size comparison (same corpus, same stride-4 sample)

| vocab size | version                                  | tokens/char | script-char coverage | unk  | decode fidelity | gate |
| ---------- | ---------------------------------------- | ----------- | -------------------- | ---- | --------------- | ---- |
| 16,000     | `1.0.0+7bf1a6e740e3.5d53c485`            | 0.3361      | 98.48%               | 0.0% | 100%            | pass |
| 32,000     | `1.0.0+7bf1a6e740e3.c5df90bc`            | 0.2790      | 98.43%               | 0.0% | 100%            | pass |
| 50,000     | `1.0.0+7bf1a6e740e3.29927a15`            | 0.2720      | 98.48%               | 0.0% | 100%            | pass |

Larger vocabs shave tokens/char (~19% from 16k→32k, then marginal) at the cost
of a heavier embedding table and slower sampling; coverage is flat. The 16k
vocab is kept as the canonical production vocab (committed under
`ml/tokenizer/vocab/`); 32k/50k runs are reference points.

## Open questions

- Compare against sentencepiece / tokenizers reference baselines on the same corpus (dev-only dependency) and record the numbers here.
