"""SFT dataset mixing and assistant-masked batching (WS-2).

Wraps ml.instruction.dataset with mixing weights and chat-template awareness.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ml.instruction.dataset import InstructionCollator, InstructionDataset, InstructionRecord, load_jsonl
from ml.sft.templates import apply_chat_template

__all__ = ["SFTDataset", "SFTMix", "load_mix"]


class SFTMix:
    """Builds a weighted mixture across task types/sources with contamination check."""

    def __init__(self, records: list[InstructionRecord], weights: dict[str, float] | None = None, seed: int = 0):
        self.records = records
        self.weights = weights or {}
        self.seed = seed

    def build(self) -> list[InstructionRecord]:
        if not self.weights:
            return list(self.records)
        # Group by category
        by_cat: dict[str, list[InstructionRecord]] = {}
        for r in self.records:
            by_cat.setdefault(r.category, []).append(r)
        # Weighted sampling (oversampling small categories)
        rng = random.Random(self.seed)
        mixed: list[InstructionRecord] = []
        total_weight = sum(self.weights.get(cat, 1.0) for cat in by_cat)
        # Simple: duplicate small categories to match weight
        for cat, recs in by_cat.items():
            w = self.weights.get(cat, 1.0)
            # target count proportional to weight
            target = max(len(recs), int(len(self.records) * w / total_weight))
            if len(recs) >= target:
                mixed.extend(rng.sample(recs, target))
            else:
                # oversample with replacement
                for _ in range(target):
                    mixed.append(rng.choice(recs))
        rng.shuffle(mixed)
        return mixed

    def check_contamination(self, eval_records: list[InstructionRecord]) -> list[str]:
        """Return list of eval instruction hashes that appear in train."""
        train_set = {r.instruction for r in self.records}
        return [r.instruction for r in eval_records if r.instruction in train_set]


def load_mix(paths: list[str | Path], weights: dict[str, float] | None = None) -> list[InstructionRecord]:
    records: list[InstructionRecord] = []
    for p in paths:
        records.extend(load_jsonl(p))
    return SFTMix(records, weights).build()


class SFTDataset(InstructionDataset):
    """InstructionDataset with chat-template and EOS-aware masking.

    Uses templates.apply_chat_template for prompt construction when messages present,
    otherwise falls back to InstructionRecord.prompt().
    """

    def __init__(self, records: Iterable[InstructionRecord], tokenizer, max_length: int, template: str = "default"):
        super().__init__(records, tokenizer, max_length)
        self.template = template

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        # Use chat template if messages present
        if record.messages:
            prompt = apply_chat_template([{"role": m["role"], "content": m["content"]} for m in record.messages if m["role"] != "assistant"], template=self.template)
            completion = record.output + "\n<eos>"
        else:
            prompt = record.prompt()
            completion = record.completion()
        prompt_ids = self.tokenizer.encode(prompt)
        completion_ids = self.tokenizer.encode(completion)
        ids = (prompt_ids + completion_ids)[: self.max_length]
        prompt_len = min(len(prompt_ids), len(ids))
        labels = [-100] * prompt_len + ids[prompt_len:]
        return {"input_ids": ids, "labels": labels, "record": record}
