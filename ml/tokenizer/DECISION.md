# Decision — Tokenizer & Vocabulary Freeze

- **date**: 2026-08-20T01:46:42
- **algorithm**: `bpe` (vocab 16,000)
- **corpus digest**: `7bf1a6e740e3`

## Why this tokenizer/vocab

- Trained on the normalized Bangla corpus (33,319 docs of 133,275, stride 4).
- Coverage: **98.48%** of script characters on a 5,000-doc sample; unk rate 0.00%.
- Efficiency gate passes: tpc 0.3361 (<= 2.0), unk 0.00% (<= 0.005), fidelity 100% (>= 1.0).

## Open questions

- Compare against sentencepiece / tokenizers reference baselines on the same corpus (dev-only dependency) and record the numbers here.
- Re-freeze at 32k / 50k vocab sizes once the 16k baseline is stable.
