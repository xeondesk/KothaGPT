# data/processed

This folder stores processed, deduplicated, and sharded text corpora used for training and evaluation.

Layout conventions

- `data/processed/CURRENT` — short file with the current dataset name (single line). Example: `bangla-validated-v1`
- `data/processed/<dataset>/train/` — training shards (text files or JSONL)
- `data/processed/<dataset>/dev/` — development/devset
- `data/processed/<dataset>/test/` — test set

Expectations

- Text should be UTF-8 normalized (NFC/NFKC as chosen by project).
- Shards should be reasonably sized (e.g., 10–100 MB) to support parallel ingestion.
- A `metadata.json` next to each dataset is recommended containing: source list, token counts, dedup rate, license summary.

Quick commands

Validate current processed dataset (uses pipeline CLI):

```bash
make data-validate
```

Build tokenizer corpus path (example):

```bash
cat data/processed/CURRENT
# outputs: bangla-validated-v1
ls -1 data/processed/bangla-validated-v1/train | head
```
