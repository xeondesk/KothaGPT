# Bangla Benchmark — v1

Deterministic benchmark datasets for Bangla language evaluation (WS-8), generated
by `data/benchmarks/bangla/generate.py`.

| Task | File | Records | Min (plan) |
| --- | --- | --- | --- |
| QA | `bangla_qa.jsonl` | 576 | 500 |
| Translation (bn↔en) | `bangla_translation.jsonl` | 1120 | 1000 |
| Summarization | `bangla_summarization.jsonl` | 108 | 100 |
| Generation | `bangla_generation.jsonl` | 26 | 20 |

## Determinism

The generator is fully deterministic: the same checkout produces byte-identical
files. Every record has a stable `record_id`, and the dev/test split is derived
as `sha256(record_id) % 10 < 8` (80% dev / 20% test), so regenerating never
reshuffles the split. `tests/test_bangla_benchmark.py` asserts the committed
JSONL matches the generator output exactly.

## Record shape

- `task`: one of `bangla_qa`, `bangla_translation`, `bangla_summarization`,
  `bangla_generation`.
- `record_id`: stable unique id.
- `split`: `dev` or `test`.
- QA: `prompt` + `passage` + `reference` (a verbatim span of `passage`).
- Translation: `source_text` + `reference` + `source_lang` / `target_lang`.
- Summarization: `prompt` + `source` + `reference` (hand-authored gold).
- Generation: `prompt` + `reference` (open-ended; scored for language quality).

## Regenerate

```bash
python -m data.benchmarks.bangla.generate
```

This rewrites `v1/*.jsonl` and `MANIFEST.json`. Do not hand-edit the JSONL —
edit the generator and regenerate.