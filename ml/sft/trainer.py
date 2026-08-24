"""SFT trainer wrapping ml.trainer loop (WS-2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ml.instruction.dataset import InstructionCollator
from ml.models import KothaGPT
from ml.sft.dataset import SFTDataset


def run_sft_trainer(
    model: KothaGPT,
    records,
    tokenizer,
    *,
    device: str = "cpu",
    max_steps: int = 20,
    batch_size: int = 2,
    learning_rate: float = 3e-4,
    max_length: int = 512,
    template: str = "default",
) -> dict[str, Any]:
    dataset = SFTDataset(records, tokenizer, max_length, template=template)
    pad_id = getattr(tokenizer, "vocab", {}).get("<pad>", 3)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=InstructionCollator(pad_id, max_length),
    )
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    steps, total = 0, 0.0
    initial_loss = None
    while steps < max_steps:
        progressed = False
        for batch in loader:
            progressed = True
            optimizer.zero_grad(set_to_none=True)
            outputs = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
            loss = outputs["loss"]
            if initial_loss is None:
                initial_loss = float(loss.detach())
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            steps += 1
            if steps >= max_steps:
                break
        if not progressed:
            break
    avg_loss = total / max(steps, 1)
    return {"steps": steps, "loss": avg_loss, "initial_loss": initial_loss, "final_loss": avg_loss}
