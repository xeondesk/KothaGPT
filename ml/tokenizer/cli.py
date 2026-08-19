"""CLI for Phase 1B — Bangla tokenizer training and benchmarking.

Usage:
    python -m ml.tokenizer.cli train --corpus PATH --algorithm bpe|unigram --vocab-size N --out DIR
    python -m ml.tokenizer.cli experiments --corpus PATH [--out DIR]
    python -m ml.tokenizer.cli encode --tokenizer DIR --text "..." | --file PATH
    python -m ml.tokenizer.cli benchmark --tokenizer DIR --file PATH
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ml.tokenizer import load_tokenizer, train_bpe, train_unigram
from ml.tokenizer.benchmark import SAMPLE_TEXTS, run_benchmark
from ml.tokenizer.corpus import load_corpus

DEFAULT_VOCAB_SIZES = (16000, 32000, 50000)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kothagpt-tokenizer",
        description="Phase 1B — Bangla tokenizer (BPE / Unigram) experiments.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train a single tokenizer.")
    train_p.add_argument("--corpus", required=True, help="corpus dir or file")
    train_p.add_argument("--algorithm", choices=("bpe", "unigram"), default="bpe")
    train_p.add_argument("--vocab-size", type=int, default=16000)
    train_p.add_argument("--out", required=True, help="output artifact dir")
    train_p.add_argument("--min-frequency", type=int, default=2)
    train_p.add_argument("--max-subword-len", type=int, default=8)
    train_p.add_argument("--iterations", type=int, default=8)

    exp_p = sub.add_parser("experiments", help="Run the BPE/Unigram x vocab matrix.")
    exp_p.add_argument("--corpus", required=True)
    exp_p.add_argument("--out", default="ml/tokenizer/artifacts")
    exp_p.add_argument("--algorithms", default="bpe,unigram", help="comma-separated algorithms")
    exp_p.add_argument(
        "--vocab-sizes",
        default=",".join(str(v) for v in DEFAULT_VOCAB_SIZES),
        help="comma-separated target vocab sizes",
    )
    exp_p.add_argument("--min-frequency", type=int, default=2)
    exp_p.add_argument("--max-subword-len", type=int, default=8)
    exp_p.add_argument("--iterations", type=int, default=8)

    enc_p = sub.add_parser("encode", help="Encode text with a saved tokenizer.")
    enc_p.add_argument("--tokenizer", required=True)
    enc_p.add_argument("--text", default=None)
    enc_p.add_argument("--file", default=None)
    enc_p.add_argument("--show-tokens", action="store_true", dest="show_tokens")

    bench_p = sub.add_parser("benchmark", help="Benchmark a saved tokenizer.")
    bench_p.add_argument("--tokenizer", required=True)
    bench_p.add_argument("--file", default=None)

    return parser


def _train_one(
    algorithm: str,
    corpus: list[str],
    vocab_size: int,
    min_frequency: int,
    max_subword_len: int,
    iterations: int,
):
    if algorithm == "bpe":
        return train_bpe(corpus, vocab_size, min_frequency=min_frequency, log=None)
    return train_unigram(
        corpus,
        vocab_size,
        min_frequency=min_frequency,
        max_subword_len=max_subword_len,
        iterations=iterations,
        log=None,
    )


def _cmd_train(args: argparse.Namespace) -> int:
    corpus = load_corpus(args.corpus)
    tokenizer = _train_one(
        args.algorithm,
        corpus,
        args.vocab_size,
        args.min_frequency,
        args.max_subword_len,
        args.iterations,
    )
    path = tokenizer.save(Path(args.out))
    stats = run_benchmark(tokenizer)
    print(f"trained {args.algorithm} tokenizer")
    print(f"  vocab: {len(tokenizer.vocab)} tokens (target {args.vocab_size})")
    print(f"  saved: {path}")
    print(f"  avg tokens/char: {stats['avg_tokens_per_char']:.4f}")
    return 0


def _cmd_experiments(args: argparse.Namespace) -> int:
    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    vocab_sizes = [int(v) for v in args.vocab_sizes.split(",") if v.strip()]
    corpus = load_corpus(args.corpus)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for algorithm in algorithms:
        for vocab_size in vocab_sizes:
            name = f"{algorithm}-{vocab_size}"
            print(f"[experiments] {name}: training on {len(corpus)} docs ...")
            tokenizer = _train_one(
                algorithm,
                corpus,
                vocab_size,
                args.min_frequency,
                args.max_subword_len,
                args.iterations,
            )
            exp_dir = out_root / "experiments" / name
            tokenizer.save(exp_dir)
            bench = run_benchmark(tokenizer)
            rows.append(
                {
                    "algorithm": algorithm,
                    "target_vocab": vocab_size,
                    "actual_vocab": len(tokenizer.vocab),
                    "avg_tokens_per_char": bench["avg_tokens_per_char"],
                    "per_set": {k: v["tokens_per_char"] for k, v in bench["per_set"].items()},
                    "unk": {k: v["unk"] for k, v in bench["per_set"].items()},
                    "dir": str(exp_dir),
                }
            )

    best = min(rows, key=lambda r: r["avg_tokens_per_char"])
    best_tokenizer = load_tokenizer(Path(best["dir"]) / "tokenizer.json")
    best_dir = out_root / "best"
    best_tokenizer.save(best_dir)

    comparison = {
        "results": rows,
        "best": {"name": f"{best['algorithm']}-{best['target_vocab']}", "dir": str(best_dir)},
    }
    (out_root / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(out_root, rows, best)
    _print_summary(rows, best)
    return 0


def _write_report(out_root: Path, rows: list[dict], best: dict) -> None:
    lines = ["# Tokenizer Experiment Report", ""]
    lines.append("Lower tokens/char is better (fewer tokens for the same text).")
    lines.append("")
    lines.append("## Overall (average across test sets)")
    lines.append("| algorithm | target vocab | actual vocab | tokens/char | best |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in rows:
        marker = " *" if row is best else ""
        lines.append(
            f"| {row['algorithm']} | {row['target_vocab']:,} | {row['actual_vocab']:,} "
            f"| {row['avg_tokens_per_char']:.4f} | {marker} |"
        )
    lines.append("")
    names = ["bangla", "mixed", "punctuation", "emoji", "code"]
    lines.append("## Per test set (tokens/char)")
    lines.append("| algorithm | vocab | " + " | ".join(names) + " |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in names) + " |")
    for row in rows:
        cells = " | ".join(f"{row['per_set'][n]:.4f}" for n in names if n in row["per_set"])
        lines.append(f"| {row['algorithm']} | {row['target_vocab']:,} | {cells} |")
    lines.append("")
    lines.append("## Best (frozen)")
    lines.append(f"- `{best['algorithm']}-{best['target_vocab']}` → `{out_root / 'best'}`")
    lines.append("")
    lines.append("## Unk coverage (total unknown tokens per set)")
    lines.append("| algorithm | vocab | " + " | ".join(names) + " |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in names) + " |")
    for row in rows:
        cells = " | ".join(str(row["unk"].get(n, "-")) for n in names)
        lines.append(f"| {row['algorithm']} | {row['target_vocab']:,} | {cells} |")
    lines.append("")
    (out_root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _print_summary(rows: list[dict], best: dict) -> None:
    print("")
    print("Experiment summary (avg tokens/char, lower is better):")
    for row in sorted(rows, key=lambda r: r["avg_tokens_per_char"]):
        marker = " <- best" if row is best else ""
        print(
            f"  {row['algorithm']:<8} vocab={row['target_vocab']:>6} "
            f"actual={row['actual_vocab']:>6} tpc={row['avg_tokens_per_char']:.4f}{marker}"
        )
    print(f"Frozen best: {best['algorithm']}-{best['target_vocab']}")


def _cmd_encode(args: argparse.Namespace) -> int:
    tokenizer = load_tokenizer(args.tokenizer)
    if args.text is not None:
        text = args.text
    elif args.file is not None:
        text = Path(args.file).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()
    ids = tokenizer.encode(text)
    print(f"tokens: {len(ids)}  chars: {len(text)}  tokens/char: {len(ids) / len(text):.4f}")
    if args.show_tokens:
        print(json.dumps(tokenizer.tokenize(text), ensure_ascii=False, indent=2))
    else:
        print(ids)
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    tokenizer = load_tokenizer(args.tokenizer)
    texts = SAMPLE_TEXTS
    if args.file is not None:
        text = Path(args.file).read_text(encoding="utf-8")
        texts = {"file": text}
    result = run_benchmark(tokenizer, texts)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "train":
        return _cmd_train(args)
    if args.command == "experiments":
        return _cmd_experiments(args)
    if args.command == "encode":
        return _cmd_encode(args)
    if args.command == "benchmark":
        return _cmd_benchmark(args)
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
