"""Pipeline configuration and the full-pipeline driver."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

from data.pipeline import dedup, io, normalize, quality, split, stats, version

__all__ = ["PipelineConfig", "run_pipeline"]


@dataclass
class PipelineConfig:
    """Knobs for the Phase 1A data pipeline.

    Defaults favour aggressive but safe filtering for a small first corpus.
    """

    # Inputs / outputs
    raw_dir: str = "data/raw"
    out_root: str = "data/processed"
    version_label: str | None = None

    # Normalization
    unicode_form: str = "NFC"
    remove_html: bool = True
    remove_markup: bool = True

    # Quality filter
    min_chars: int = 100
    max_chars: int = 1_000_000
    min_words: int = 20
    require_bangla: bool = True
    min_bangla_ratio: float = 0.5
    allow_pii: bool = False

    # Deduplication
    dedup_exact: bool = True
    dedup_near: bool = False
    dedup_threshold: float = 0.8

    # Split
    validation_ratio: float = 0.02

    # Shards
    shard_size: int = 100_000
    gzip_output: bool = True

    # Reporting
    sample_note: str | None = None

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def run_pipeline(cfg: PipelineConfig) -> dict:
    """Execute raw -> normalize -> filter -> dedup -> split -> version.

    Returns a summary dict with per-step counts and the emitted version.
    """
    cfg_dict = cfg.as_dict()
    source = Path(cfg.raw_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"raw directory not found: {source}")

    summary: dict = {}

    # 1. Read raw records.
    records = list(io.iter_records(source))
    summary["raw"] = len(records)

    # 2. Normalize.
    normalized = []
    for record in records:
        text = normalize.normalize_text(
            record["text"],
            unicode_form=cfg.unicode_form,
            remove_html=cfg.remove_html,
            remove_markup=cfg.remove_markup,
        )
        if not text:
            continue
        record = dict(record)
        record["text"] = text
        normalized.append(record)
    summary["normalized"] = len(normalized)

    # 3. Quality filter.
    filtered = []
    for record in normalized:
        keep, reasons = quality.quality_filter(
            record["text"],
            min_chars=cfg.min_chars,
            max_chars=cfg.max_chars,
            min_words=cfg.min_words,
            require_bangla=cfg.require_bangla,
            min_bangla_ratio=cfg.min_bangla_ratio,
            allow_pii=cfg.allow_pii,
        )
        if keep:
            filtered.append(record)
    summary["after_filter"] = len(filtered)

    # 4. Deduplicate.
    deduped = dedup.deduplicate(
        filtered,
        exact=cfg.dedup_exact,
        near=cfg.dedup_near,
        threshold=cfg.dedup_threshold,
    )
    summary["after_dedup"] = len(deduped)

    # 5. Deterministic train/validation split.
    split_records = [split.split_record(r, validation_ratio=cfg.validation_ratio) for r in deduped]
    summary["train"] = sum(1 for r in split_records if r["split"] == "train")
    summary["validation"] = sum(1 for r in split_records if r["split"] == "validation")

    # 6. Statistics report.
    stats_data = stats.compute_stats(iter(split_records))
    summary["stats"] = stats_data

    # 7. Versioned output.
    out_root = Path(cfg.out_root)
    version_obj = version.DatasetVersion.create(
        config=cfg_dict,
        counts=summary,
        shards=[],
        files={},
        version_label=cfg.version_label,
    )
    version_dir = out_root / version_obj.version_id
    stats.write_report(stats_data, version_dir / "report")

    train_dir = version_dir / "train"
    val_dir = version_dir / "validation"
    train_shards = io.write_shards(
        (r for r in split_records if r["split"] == "train"),
        train_dir,
        shard_size=cfg.shard_size,
        gzip_output=cfg.gzip_output,
    )
    val_shards = io.write_shards(
        (r for r in split_records if r["split"] == "validation"),
        val_dir,
        shard_size=cfg.shard_size,
        gzip_output=cfg.gzip_output,
    )
    version_obj.shards = train_shards + val_shards
    version_obj.files = {
        "train_dir": str(train_dir.relative_to(out_root)),
        "validation_dir": str(val_dir.relative_to(out_root)),
        "report_dir": str((version_dir / "report").relative_to(out_root)),
    }
    version_obj.config = cfg_dict

    manifest_path = version.write_manifest(version_obj, version_dir)
    summary["version_id"] = version_obj.version_id
    summary["manifest"] = str(manifest_path)
    summary["train_shards"] = train_shards
    summary["validation_shards"] = val_shards
    return summary
