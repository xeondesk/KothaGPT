"""Training CLI: ``python -m ml.trainer.cli run --config ...``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ml.models import KothaGPT, load_config
from ml.tokenizer import load_tokenizer

from .dataset import CausalLMDataset, build_blocks
from .distributed import DistributedConfig, destroy_distributed, init_distributed, unwrap_module
from .evaluate import sample_text
from .loop import train


def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)

    if args.tokenizer:
        config.data.tokenizer_path = args.tokenizer

    dist = DistributedConfig(
        world_size=args.world_size,
        rank=args.rank,
        local_rank=args.local_rank,
        master_addr=args.master_addr,
        master_port=args.master_port,
        backend=args.backend,
        use_fsdp=args.use_fsdp,
        distributed=args.distributed,
        init_method=args.init_method,
    )
    dist = init_distributed(dist)

    tokenizer = load_tokenizer(config.data.tokenizer_path)
    block_size = config.model.max_position_embeddings

    train_blocks = build_blocks(config.data.train, tokenizer, block_size)
    train_dataset = CausalLMDataset(train_blocks)
    validation_dataset = None
    if Path(config.data.validation).exists():
        try:
            validation_dataset = CausalLMDataset(
                build_blocks(config.data.validation, tokenizer, block_size)
            )
        except (FileNotFoundError, ValueError):
            validation_dataset = None

    model = KothaGPT(config.model, gradient_checkpointing=config.training.gradient_checkpointing)

    resume_from = args.out if args.resume else None

    result = train(
        model,
        config,
        train_dataset,
        validation_dataset,
        out_dir=args.out,
        device=args.device,
        resume_from=resume_from,
        max_steps=args.max_steps,
        dist=dist,
    )

    if (dist.rank or 0) == 0:
        print(json.dumps(result))
        if validation_dataset is not None:
            text = sample_text(
                unwrap_module(model),
                tokenizer,
                "বাংলা ভাষা",
                max_new_tokens=8,
                temperature=1.0,
                device=args.device,
            )
            print(f"sample: {text}")
    destroy_distributed()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ml.trainer", description="KothaGPT base-model training")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a training job")
    run.add_argument("--config", required=True, help="path to YAML config")
    run.add_argument("--tokenizer", default=None, help="override tokenizer artifact path")
    run.add_argument("--out", default="ml/pretrain/artifacts/run", help="run output directory")
    run.add_argument("--device", default="cpu", help="device (cpu or cuda[:N])")
    run.add_argument("--max-steps", type=int, default=None, help="override max steps")
    run.add_argument("--resume", action="store_true", help="resume from latest checkpoint")
    run.add_argument("--distributed", action="store_true", help="enable distributed training")
    run.add_argument("--world-size", type=int, default=None, help="number of ranks (default from WORLD_SIZE)")
    run.add_argument("--rank", type=int, default=None, help="rank of this process (default from RANK)")
    run.add_argument("--local-rank", type=int, default=None, help="local rank (default from LOCAL_RANK)")
    run.add_argument("--master-addr", default="127.0.0.1", help="master address")
    run.add_argument("--master-port", default="29500", help="master port")
    run.add_argument("--backend", default=None, help="distributed backend (nccl|gloo)")
    run.add_argument("--init-method", default=None, help="torch.distributed init method URL")
    run.add_argument("--use-fsdp", action="store_true", help="use FSDP instead of DDP (requires CUDA)")
    run.set_defaults(func=_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())