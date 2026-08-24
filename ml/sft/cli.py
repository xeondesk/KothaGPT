"""SFT CLI: python -m ml.sft.cli run --config ml/configs/sft.yaml"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.instruction.dataset import load_jsonl
from ml.models import KothaGPT, load_config
from ml.tokenizer import load_tokenizer

from ml.sft.trainer import run_sft_trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SFT training")
    parser.add_argument("--train", required=True, help="instruction jsonl")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base", help="base checkpoint (optional)")
    parser.add_argument("--out", default="ml/sft/artifacts/run")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--template", default="default")
    args = parser.parse_args()

    config = load_config(args.config)
    tok_path = Path(args.tokenizer)
    if tok_path.is_dir():
        tok_path = tok_path / "tokenizer.json"
    tokenizer = load_tokenizer(tok_path)
    records = load_jsonl(args.train)
    model = KothaGPT(config.model)
    if args.base and Path(args.base).is_file():
        try:
            import torch
            state = torch.load(args.base, map_location="cpu")
            model.load_state_dict(state.get("model", state), strict=False)
        except Exception as e:
            print(f"warn: could not load base {args.base}: {e}")

    metrics = run_sft_trainer(
        model,
        records,
        tokenizer,
        device=args.device,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        max_length=config.model.max_position_embeddings,
        template=args.template,
    )
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(metrics)


if __name__ == "__main__":
    main()
