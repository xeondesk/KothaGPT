"""CLI entry point for the Phase 1A Bangla data pipeline.

Usage:
    python -m data.pipeline.cli run [options]
    python -m data.pipeline.cli normalize FILE
    python -m data.pipeline.cli check TEXT
    python -m data.pipeline.cli version
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data.pipeline import normalize as norm
from data.pipeline import quality
from data.pipeline.config import PipelineConfig, run_pipeline

_MAIN = (
    "raw -> normalize -> quality filter -> deduplicate -> train/validation "
    "split -> stats -> versioned shards"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kothagpt-data",
        description="Phase 1A — Bangla dataset pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the full pipeline (" + _MAIN + ").")
    run_p.add_argument("--raw-dir", default="data/raw")
    run_p.add_argument("--out-root", default="data/processed")
    run_p.add_argument("--version-label", default=None)
    run_p.add_argument("--unicode-form", default="NFC")
    run_p.add_argument("--keep-html", action="store_true", help="skip HTML stripping")
    run_p.add_argument("--keep-markup", action="store_true", help="skip markup stripping")
    run_p.add_argument("--min-chars", type=int, default=100)
    run_p.add_argument("--max-chars", type=int, default=1_000_000)
    run_p.add_argument("--min-words", type=int, default=20)
    run_p.add_argument("--no-bangla-check", action="store_true")
    run_p.add_argument("--min-bangla-ratio", type=float, default=0.5)
    run_p.add_argument("--allow-pii", action="store_true")
    run_p.add_argument("--no-exact-dedup", action="store_true")
    run_p.add_argument("--near-dedup", action="store_true", help="enable MinHash LSH dedup")
    run_p.add_argument("--dedup-threshold", type=float, default=0.8)
    run_p.add_argument("--validation-ratio", type=float, default=0.02)
    run_p.add_argument("--shard-size", type=int, default=100_000)
    run_p.add_argument("--no-gzip", action="store_true")

    norm_p = sub.add_parser("normalize", help="Normalize a text file or stdin.")
    norm_p.add_argument("file", nargs="?", default="-")
    norm_p.add_argument("--unicode-form", default="NFC")
    norm_p.add_argument("--no-html", action="store_true")
    norm_p.add_argument("--no-markup", action="store_true")

    check_p = sub.add_parser(
        "check", help="Run language/PII/length checks on a text file or stdin."
    )
    check_p.add_argument("file", nargs="?", default="-")

    version_p = sub.add_parser("version", help="Show the CURRENT dataset version.")
    version_p.add_argument("--out-root", default="data/processed")

    return parser


def _read_input(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def _cmd_normalize(args: argparse.Namespace) -> int:
    text = _read_input(args.file)
    result = norm.normalize_text(
        text,
        unicode_form=args.unicode_form,
        remove_html=not args.no_html,
        remove_markup=not args.no_markup,
    )
    sys.stdout.write(result + "\n")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    text = _read_input(args.file)
    lang = quality.detect_language(text)
    ratio = quality.bengali_ratio(text)
    pii = quality.contains_pii(text)
    kept, reasons = quality.length_filter(text, min_chars=1, max_chars=10**12, min_words=0)
    report = {
        "language": lang,
        "bengali_ratio": round(ratio, 3),
        "chars": len(text),
        "words": len(text.split()),
        "pii": pii,
        "length_ok": kept,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _cmd_version(args: argparse.Namespace) -> int:
    current = Path(args.out_root) / "CURRENT"
    if not current.exists():
        print("No dataset version yet. Run 'kothagpt-data run'.", file=sys.stderr)
        return 1
    version_id = current.read_text(encoding="utf-8").strip()
    manifest = Path(args.out_root) / version_id / "MANIFEST.json"
    print(f"version_id: {version_id}")
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        counts = data.get("counts", {})
        print(f"docs (after dedup): {counts.get('after_dedup', 'n/a')}")
        print(f"train: {counts.get('train', 'n/a')}")
        print(f"validation: {counts.get('validation', 'n/a')}")
        print(f"manifest: {manifest}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = PipelineConfig(
        raw_dir=args.raw_dir,
        out_root=args.out_root,
        version_label=args.version_label,
        unicode_form=args.unicode_form,
        remove_html=not args.keep_html,
        remove_markup=not args.keep_markup,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        min_words=args.min_words,
        require_bangla=not args.no_bangla_check,
        min_bangla_ratio=args.min_bangla_ratio,
        allow_pii=args.allow_pii,
        dedup_exact=not args.no_exact_dedup,
        dedup_near=args.near_dedup,
        dedup_threshold=args.dedup_threshold,
        validation_ratio=args.validation_ratio,
        shard_size=args.shard_size,
        gzip_output=not args.no_gzip,
    )
    summary = run_pipeline(cfg)
    print(f"Pipeline complete: {_MAIN}")
    print(f"  raw docs:        {summary['raw']:,}")
    print(f"  normalized:      {summary['normalized']:,}")
    print(f"  after filter:    {summary['after_filter']:,}")
    print(f"  after dedup:     {summary['after_dedup']:,}")
    print(f"  train / val:     {summary['train']:,} / {summary['validation']:,}")
    print(f"  version id:      {summary['version_id']}")
    print(f"  manifest:        {summary['manifest']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "normalize":
        return _cmd_normalize(args)
    if args.command == "check":
        return _cmd_check(args)
    if args.command == "version":
        return _cmd_version(args)
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
