# Dataset Implementation Plan (ডেটাসেট পাইপলাইন)

Maps the 18 dataset items to concrete work. Existing pipeline modules live in
`data/pipeline/`; collection scripts in `data/scripts/`. Status legend:
✅ built · 🔶 partial / needs extension · 🆕 new.

Goal: a reproducible, versioned, legally-clean, Bangla-first multilingual corpus
with train/validation/test splits, plus quality-scored instruction-style data
for later SFT.

---

## Phase A — Collection & Ingestion (সমগ্র সংগ্রহ)

### A1. Bangla text collection (বাংলা text dataset সংগ্রহ) 🔶
**Status:** `data/scripts/fetch_corpora.py` + `corpora.manifest.json` exist but
the manifest is an empty template.
- **Do:** populate the manifest with verified sources (Bengali Wikipedia dump,
  Samanantar bn side, OSCAR/CC-100 bn, bn-wikinews, bn-wikiquote,
  BanglaStory/BanglaNLG). Add an optional HF Hub loader
  (`data/scripts/load_hf.py`) so `--source hf:AI4Bharat/samanantar` works.
- **Success:** `make data` completes on ≥ 3 sources; every record carries
  `{source, license, license_url, fetched_at}`.
- **Rule:** raw data stays in `data/raw/` (git-ignored), never committed.

### A2. English dataset collection (ইংরেজি dataset সংগ্রহ) 🆕
- **Do:** add English/mixed corpora (e.g., Wikipedia en, English web subset)
  and relax the pipeline's Bangla gate with a `--lang en|bn|mixed|all` flag in
  `data/pipeline/cli.py` (`quality.detect_language` already classifies `en`).
- **Success:** pipeline can emit an `en` or `mixed` dataset with language split
  reported in stats.

### A3. Open-source dataset collection (Open-source dataset সংগ্রহ) 🔶
- **Do:** extend the manifest schema to `{source_type: url|hf|hub, name, url,
  license, license_url, split, config}` and validate entries at ingest
  (`--strict-license`). Add `--no-download` dry-run.
- **Success:** one manifest drives all downloads; license is mandatory and
  machine-checked.

### A4. Web corpus creation (Web corpus তৈরি) 🆕
- **Do:** ingest OSCAR / CC-100 `bn` subsets and a Common Crawl `bn` language
  filter pass. Optionally a focused crawler (`data/scripts/crawl.py`) limited
  to allow-listed Bangla domains with `robots.txt` respect.
- **Success:** ≥ 1 web corpus source; reported web-vs-encyclopedia share in the
  report. Web data is expected to be noisy — must pass Phase C hard.

### A5. Books / documents collection (বই/ডকুমেন্ট dataset সংগ্রহ) 🆕
- **Do:** Bangla books from verified open sources (license-checked only) +
  document extractors for PDF/EPUB/DOCX → normalized text
  (`data/scripts/extract_docs.py`). Cross-check titles against a
  known-copyrighted blocklist before ingest.
- **Success:** extractor covers PDF/EPUB; ingested docs carry full provenance.

### A6. Code dataset collection (Code dataset সংগ্রহ) 🆕
- **Do:** ingest a Bangla-relevant code subset (The Stack `bn`, or a curated
  GitHub scrape) as JSONL `{path, language, code, content}`. Code gets its own
  dedup (PCCC-style) in Phase C1 and is **not** run through the prose
  language gate.
- **Success:** separate `code` shard stream in the versioned output.

### A7. Question → Answer dataset (Question → Answer dataset তৈরি) 🆕
- **Do:** ingest public Bangla QA (XQuAD-bn, BanglaQA-style) and optionally a
  synthetic QA generator with answer-extraction verification.
- **Success:** each record validated as `{question, answer, context?, source,
  verified}`; ≥ 95% answers non-empty after filters.

### A8. Instruction dataset (Instruction dataset তৈরি) 🆕
- **Do:** build a small curated instruction set (Alpaca-style, Bangla) and a
  synthetic generation pipeline (`data/synthetic/`), gated by Phase C quality
  scoring before it is allowed into a release.
- **Success:** `data/processed/<ver>/instruction/` shards; instruction quality
  score (≥ threshold) logged in the manifest.

### A9. Conversation dataset (Conversation dataset তৈরি) 🆕
- **Do:** multi-turn extraction from Bangla forums/subreddits (license-checked)
  and/or synthetic dialogue chains; schema
  `{messages: [{role, content}], source}`.
- **Success:** ≥ 2-turn conversations only; role/content schema validated by a
  `pydantic`-style check in `data/pipeline/schema.py`.

---

## Phase B — Filtering & Quality (ফিল্টারিং ও গুণমান)

### B1. Dataset deduplication (Dataset deduplication) 🔶
**Status:** `data/pipeline/dedup.py` has exact sha256 + MinHash LSH.
- **Do:** scale to large corpora (sharded MinHash, optional Bloom filter for
  exact dedup), **cross-source** global dedup so one corpus doesn't leak into
  another, and PCCC-style dedup for the code stream.
- **Success:** dedup rate < 3% reported in `report/`; cross-source duplicates
  removed.

### B2. Spam filtering (Spam filtering) ✅
**Status:** implemented — `data/pipeline/spam.py` (domain blocklist, Bangla+
English promo phrases, token-repetition, char-burst, ALLCAPS, URL-density
scoring; `spam_gate` with `--spam-threshold`, disabled via `--no-spam-check`),
wired into `run_pipeline` between quality filter and dedup.
- **Do:** grow the domain/phrase blocklists from real corpus samples.
- **Success:** per-doc spam score; `--no-spam-check` escape hatch.

### B3. Toxic-content filtering (Toxic-content filtering) ✅
**Status:** implemented — `data/pipeline/toxic.py` (word-boundary slur/hate/
profanity blocklists in Bangla + English with rule ids, always-on by default;
optional `--toxic-classifier module:callable` toxicity hook with
`--toxic-threshold`; disabled via `--no-toxic-check`). Bangla-safe boundaries
use lookarounds (Python `\b` breaks on Bangla combining marks).
- **Do:** expand blocklists from real corpus samples; plug in a trained
  toxicity classifier as the hook.
- **Success:** blocklist rules unit-tested; classifier scores over threshold
  reject; escape hatch for research corpora.

### B4. PII filtering (PII filtering) 🔶→✅
**Status:** extended — `data/pipeline/quality.py` now also detects passport and
street-address PII and adds `redact_pii()`; `--pii-mode mask` masks spans
(`[PII]`) and keeps the doc, `drop` (default) rejects it.
- **Do:** add remaining localized identifiers (bank account, driving license)
  and tune recall on a real Bangla sample.
- **Success:** PII recall test suite on Bangla-language samples; zero PII in
  released train split.

### B5. Copyright/licensing validation (Copyright/licensing validation) ✅
**Status:** implemented — `data/pipeline/copyright.py` (license allow-list with
version-insensitive matching, per-source license map `--license-map`, title
blocklist `--copyrighted-titles`), wired into `run_pipeline` as the license
gate (`--require-license`); license counts reported in `report/REPORT.md`.
- **Do:** automate known-copyrighted-title fingerprinting against a growing
  blocklist, and emit the provenance map into `MANIFEST.json`.
- **Success:** no record without a validated license; provenance map written to
  every version (`MANIFEST.json`).

### B6. Dataset quality scoring (Dataset quality scoring) 🔶
**Status:** `data/pipeline/stats.py` reports aggregate stats only.
- **Do:** `data/pipeline/score.py` — per-doc score combining language
  confidence, repetition ratio, length fit, source trust, and (optional) LM
  perplexity; write `report/quality.json` with score histogram and a
  `--min-score` threshold.
- **Success:** every doc carries `quality_score`; low-scoring tail is
  inspectable and threshold-tunable.

---

## Phase C — Split & Versioning (বিভাগ ও সংস্করণ)

### C1. Train/validation/test split (Train/validation/test split তৈরি) 🔶
**Status:** `data/pipeline/split.py` emits deterministic train/validation.
- **Do:** add a held-out **test** set (`--test-ratio`, default 0.01),
  **stratify by source/domain** so rare sources aren't starved, and
  **contamination prevention** (train/test hash intersection check).
- **Success:** train/val/test reported; zero cross-split near-duplicate matches.

### C2. Dataset versioning system (Dataset versioning ব্যবস্থা তৈরি) 🔶
**Status:** `data/pipeline/version.py` already content-addresses every run and
updates `data/processed/CURRENT`.
- **Do:** attach provenance (source → record ids), a quality-score histogram,
  and a human-readable `CHANGELOG` field; add optional publish step
  (`data/scripts/publish.py` → HF Hub or local registry).
- **Success:** every version is immutable + reproducible from config; upstream
  pointer (`CURRENT`) and full metadata present in `MANIFEST.json`.

---

## Build order (dependencies)

1. **B5 + B4** first — license/PII gates protect everything downstream.
2. **A1 → A3** collection, then **C1/C2** split+version (already functional) so
   each new source lands in a versioned snapshot.
3. **B1 → B2 → B3 → B6** filtering stack, applied before split.
4. **A4 → A6** web/books/code (noisier sources) once filters are solid.
5. **A7 → A9** instruction-style data last; synthetic outputs must pass B6.

## Acceptance criteria (done = all green)

- [ ] `make data` runs on the filled manifest with **no license/PII failures**
- [ ] report shows: dedup rate < 3%, toxic/spam drop rates, quality histogram
- [ ] train/val/test splits with zero cross-contamination
- [ ] every version reproducible via `MANIFEST.json` config
- [ ] CI gate (`make data-validate`) passes against the real dataset

## Risks

- **License grey areas** → strict allow-list + human review per new source.
- **Synthetic data quality** → B6 scoring + sample review before release.
- **Web corpus noise** → hard Phase-B filtering + source scoring.
- **Code/instruction mixing with prose** → separate shard streams, separate
  dedup.

Owners: `data/` (collection + filtering), `data/scripts/` (ingest), `data/pipeline/`
(pipeline), `infra/` (CI gates), `evals/` (split/QA validation).