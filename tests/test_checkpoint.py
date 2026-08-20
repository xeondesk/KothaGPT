"""Tests for checkpoint save/load/resume semantics (WS-6)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ml.models import KothaGPT, ModelConfig
from ml.trainer import (
    CausalLMDataset,
    build_blocks,
    group_parameters,
    latest_checkpoint,
    load_checkpoint,
    resume,
    save_checkpoint,
    step_of,
    train,
)
from tests.conftest import make_corpus


def test_save_and_load_roundtrip(config, tokenizer, tmp_path: Path) -> None:
    model = KothaGPT(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = save_checkpoint(
        tmp_path / "run",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        step=7,
        config=config,
        metadata={"config_digest": "abc"},
    )
    assert path.name == "step-0000007.pt"
    assert step_of(path) == 7
    state = load_checkpoint(path)
    assert state["step"] == 7
    assert state["config"].model.hidden_size == config.model.hidden_size


def test_latest_checkpoint_returns_newest(config, tokenizer, tmp_path: Path) -> None:
    model = KothaGPT(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in (1, 2, 3):
        save_checkpoint(
            tmp_path / "run",
            model=model,
            optimizer=optimizer,
            scheduler=None,
            step=step,
            config=config,
            metadata={},
        )
    latest = latest_checkpoint(tmp_path / "run")
    assert latest is not None
    assert step_of(latest) == 3


def test_resume_restores_step(config, tokenizer, tmp_path: Path) -> None:
    model = KothaGPT(config.model)
    out = tmp_path / "run"
    train(
        model,
        config,
        CausalLMDataset(build_blocks(make_corpus(tmp_path, docs=8), tokenizer, block_size=8)),
        None,
        out_dir=out,
        device="cpu",
    )
    restored = KothaGPT(config.model)
    restored_opt = torch.optim.AdamW(
        group_parameters(restored, weight_decay=config.training.weight_decay),
        lr=1e-3,
    )
    step = resume(out, model=restored, optimizer=restored_opt, scheduler=None, config=config)
    assert step == 4
    for a, b in zip(model.parameters(), restored.parameters()):
        assert torch.equal(a.detach(), b.detach())


def test_resume_continues_training(config, tokenizer, tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path, docs=8)
    dataset = CausalLMDataset(build_blocks(corpus, tokenizer, block_size=8))
    out = tmp_path / "run"
    model = KothaGPT(config.model)
    first = train(model, config, dataset, None, out_dir=out, device="cpu")
    assert first["step"] == 4

    continued = config.model
    resumed_model = KothaGPT(continued)
    resumed_config = config
    second = train(
        resumed_model,
        resumed_config,
        dataset,
        None,
        out_dir=out,
        device="cpu",
        max_steps=6,
    )
    assert second["step"] == 6


def test_resume_model_mismatch_raises(config, tokenizer, tmp_path: Path) -> None:
    dataset = CausalLMDataset(build_blocks(make_corpus(tmp_path, docs=8), tokenizer, block_size=8))
    out = tmp_path / "run"
    model = KothaGPT(config.model)
    train(model, config, dataset, None, out_dir=out, device="cpu")

    mismatched = ModelConfig(**{**config.model.to_dict(), "hidden_size": 48})
    wrong = KothaGPT(mismatched)
    opt = torch.optim.AdamW(wrong.parameters(), lr=1e-3)

    from ml.models import BaseModelConfig

    mismatched_cfg = BaseModelConfig(
        model=mismatched, training=config.training, data=config.data
    )
    with pytest.raises(ValueError, match="model config mismatch"):
        resume(out, model=wrong, optimizer=opt, scheduler=None, config=mismatched_cfg)


def test_checkpoint_pruning_keeps_last(config, tokenizer, tmp_path: Path) -> None:
    from dataclasses import replace


    dataset = CausalLMDataset(build_blocks(make_corpus(tmp_path, docs=8), tokenizer, block_size=8))
    pruned_cfg = replace(
        config, training=replace(config.training, keep_last=1)
    )
    model = KothaGPT(pruned_cfg.model)
    out = tmp_path / "run"
    train(model, pruned_cfg, dataset, None, out_dir=out, device="cpu")
    checkpoints = sorted((out / "checkpoints").glob("step-*.pt"))
    assert len(checkpoints) == 1
    assert step_of(checkpoints[0]) == 4