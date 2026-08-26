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
    if args.base:
        base_path = Path(args.base)
        if not base_path.exists():
            raise SystemExit(f"base checkpoint not found: {args.base}")
        import torch

        try:
            state = torch.load(base_path, map_location="cpu", weights_only=True)
        except Exception as e:
            raise SystemExit(f"failed to load base checkpoint {args.base}: {e}") from e
        # Trainer saves {"model_state": ...}, not "model"
        if isinstance(state, dict) and "model_state" in state:
            model_state = state["model_state"]
        elif isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
            # Assume direct state dict (weights-only)
            model_state = state
        else:
            raise SystemExit(f"base checkpoint {args.base} has no model_state")
        try:
            model.load_state_dict(model_state, strict=True)
        except Exception as e:
            raise SystemExit(f"incompatible base checkpoint {args.base}: {e}") from e

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
