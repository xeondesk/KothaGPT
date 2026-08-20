"""Streaming corpus dataset for causal-LM pretraining.

Records are tokenized in deterministic order and packed contiguously into
``block_size`` blocks (standard LM packing across document boundaries). The
final partial block is dropped to avoid padding. Blocks are materialized as a
single ``LongTensor``; for very large corpora, build per shard.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from ml.tokenizer import BaseTokenizer
from ml.tokenizer.corpus import load_corpus


def build_blocks(
    corpus: str | Path,
    tokenizer: BaseTokenizer,
    block_size: int,
) -> torch.Tensor:
    """Tokenize the corpus and pack it into ``[N, block_size]`` token blocks."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    buffer: list[int] = []
    blocks: list[torch.Tensor] = []
    for text in load_corpus(corpus):
        ids = tokenizer.encode(text)
        if not ids:
            continue
        buffer.extend(ids)
        while len(buffer) >= block_size:
            blocks.append(torch.tensor(buffer[:block_size], dtype=torch.long))
            del buffer[:block_size]
    if not blocks:
        raise ValueError(f"corpus too small for block_size={block_size}: {corpus}")
    return torch.stack(blocks)


class CausalLMDataset(Dataset):
    """Returns ``(input_ids, labels)`` with a one-token causal shift."""

    def __init__(self, blocks: torch.Tensor) -> None:
        self.blocks = blocks

    def __len__(self) -> int:
        return self.blocks.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        block = self.blocks[index]
        return block[:-1], block[1:]