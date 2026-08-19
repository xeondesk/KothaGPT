# Data — Phase 1A Bangla Dataset Pipeline

Pipeline: raw → normalize → quality filter → deduplicate → train/validation
split → stats → versioned shards.

```
data/raw/              source corpus (drop text/jsonl/html files here)
data/scripts/          corpus fetch tool + manifest template
data/pipeline/         the pipeline implementation (pure stdlib)
data/processed/        versioned output + reports
```

## Usage

```bash
# Run the full pipeline over data/raw -> data/processed/<version_id>/
python -m data.pipeline.cli run

# Inspect a single text
python -m data.pipeline.cli check  my_file.txt
python -m data.pipeline.cli normalize some_raw.txt > cleaned.txt

# Show the current dataset version
python -m data.pipeline.cli version
```

Every run emits an immutable, content-addressed version under
`data/processed/<version_id>/` with:

- `train/*.jsonl.gz`, `validation/*.jsonl.gz` — the split shards
- `report/REPORT.md`, `report/stats.json` — dataset statistics
- `MANIFEST.json` — config, counts, shards, and file map
- `data/processed/CURRENT` — pointer to the latest version

Key flags: `--min-chars/--max-chars/--min-words` (length limits),
`--no-bangla-check` (disable language gating), `--allow-pii`,
`--no-exact-dedup`, `--near-dedup` (MinHash LSH), `--validation-ratio`,
`--shard-size`, `--no-gzip`. See `python -m data.pipeline.cli run --help`.

## Collecting a legally usable Bangla corpus

Do **not** commit raw data to this repository. Download or place corpora under
`data/raw/` and verify licenses yourself. `data/scripts/corpora.manifest.json`
is a template for `scripts/fetch_corpora.py`:

```bash
# Edit data/scripts/corpora.manifest.json to list {name, url, license, ...}
python data/scripts/fetch_corpora.py --manifest data/scripts/corpora.manifest.json
```

Open Bangla-language datasets worth evaluating (verify the license of each):

| Corpus | Notes | License to verify |
| --- | --- | --- |
| Samanantar (AI4Bharat) | Large en→bn parallel corpus | CC-BY-SA-4.0 |
| OSCAR / CC-100 (bn) | Web crawl, noisy, must filter | Common Crawl terms |
| Bengali Wikipedia dump | High-quality encyclopedia text | CC-BY-SA-4.0 |
| bn-wikinews / bn-wikiquote | Smaller, clean wiki projects | CC-BY-SA-4.0 |
| BanglaStory / BanglaNLG | Curated creative / NLG collections | varies |
| LEAF (language eval) | Evaluations, not pretraining | varies |

The pipeline's default filters assume a web/encyclopedia-style corpus. For
conversational or code-heavy data, loosen `--min-words`/`--min-chars`.

## What each step does

- **Normalize** — Unicode NFC, HTML/markup stripping, whitespace collapsing.
  Zero-width joiner (U+200D) is preserved because it is meaningful in Bangla
  conjuncts.
- **Quality filter** — Bangla language gate (script ratio + common-word boost),
  PII scan (email/URL/phone/IP/card/NID), short/long length limits.
- **Deduplicate** — exact sha256 dedup by default; optional MinHash + banded
  LSH near-duplicate removal (`--near-dedup`).
- **Split** — deterministic content-addressed train/validation split so the
  split is reproducible regardless of file order.
- **Stats / report** — per-doc length distributions, language and source
  breakdown, split sizes.
- **Version** — content-addressed version id derived from the full config and
  counts; every run produces a new immutable snapshot.

## Tests

```bash
python -m pytest tests/test_data_pipeline.py
```

Do not place private, copyrighted, or sensitive raw data in this repository.