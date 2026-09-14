"""WordPiece tokenizer: trainer and encoder.

Training selects merges that maximise the likelihood increase, i.e. the pair
(a, b) with the highest ``freq(ab) / (freq(a) * freq(b))``.  Encoding uses a
greedy longest-match-first scan over the merged subwords in the vocabulary.

This implementation follows the existing BPE/Unigram conventions: word-based
tokenisation with a leading ``▁`` (U+2581) space marker and the full Bengali
block bootstrapped into every vocabulary.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from itertools import pairwise
from typing import Any

from .base import _WORD_MARKER, BOOTSTRAP_CHARS, SPECIAL_TOKENS, UNK, BaseTokenizer

__all__ = ["WordPieceTokenizer", "train_wordpiece"]


def train_wordpiece(
    texts: list[str],
    vocab_size: int,
    *,
    min_frequency: int = 2,
    log: Callable[[str], None] | None = None,
) -> WordPieceTokenizer:
    """Train a WordPiece tokenizer from a list of texts."""
    if vocab_size <= len(SPECIAL_TOKENS):
        raise ValueError(f"vocab_size must exceed {len(SPECIAL_TOKENS)} special tokens")
    word_counts: Counter[str] = Counter()
    for text in texts:
        for word in text.split():
            word_counts[_WORD_MARKER + word] += 1
    if not word_counts:
        raise ValueError("empty corpus: no words to train on")

    # Start with individual characters as base vocabulary.
    vocab: set[str] = set()
    for word in word_counts:
        vocab.update(word)
    vocab |= set(BOOTSTRAP_CHARS)

    # Track how each word is segmented and compute symbol-pair frequencies.
    symbols: dict[str, tuple[str, ...]] = {w: tuple(w) for w in word_counts}
    pair_counts: Counter[tuple[str, str]] = Counter()
    symbol_counts: Counter[str] = Counter()
    for word, sym in symbols.items():
        cnt = word_counts[word]
        for s in sym:
            symbol_counts[s] += cnt
        for a, b in pairwise(sym):
            pair_counts[(a, b)] += cnt

    target = vocab_size - len(SPECIAL_TOKENS)
    merges: list[tuple[str, str]] = []

    while len(vocab) < target:
        # Pick the pair maximising freq(ab) / (freq(a) * freq(b)).
        best_pair: tuple[str, str] | None = None
        best_score = -1.0
        for pair, freq in pair_counts.items():
            if freq < min_frequency:
                continue
            a, b = pair
            denom = symbol_counts.get(a, 0) * symbol_counts.get(b, 0)
            if denom == 0:
                continue
            score = freq / denom
            if score > best_score:
                best_score = score
                best_pair = pair
        if best_pair is None:
            break

        a, b = best_pair
        merged = a + b
        vocab.add(merged)
        merges.append(best_pair)

        # Apply the merge to every word that contained this pair.
        freq = pair_counts[best_pair]
        for word in list(word_counts):
            sym = symbols[word]
            new_sym: list[str] = []
            i = 0
            wcnt = word_counts[word]
            while i < len(sym):
                if i + 1 < len(sym) and sym[i] == a and sym[i + 1] == b:
                    new_sym.append(merged)
                    i += 2
                else:
                    new_sym.append(sym[i])
                    i += 1
            new_sym_t = tuple(new_sym)
            if new_sym_t == sym:
                continue
            # Update counts.
            for s in sym:
                symbol_counts[s] -= wcnt
            for a2, b2 in pairwise(sym):
                pair_counts[(a2, b2)] -= wcnt
            for s in new_sym_t:
                symbol_counts[s] += wcnt
            for a2, b2 in pairwise(new_sym_t):
                pair_counts[(a2, b2)] += wcnt
            symbols[word] = new_sym_t

        # Clean up stale entries.
        for key in [k for k, v in pair_counts.items() if v <= 0]:
            del pair_counts[key]

        if log is not None and len(merges) % 1000 == 0:
            log(f"wordpiece: {len(merges)} merges, vocab={len(vocab)}")

    vocab_ids: dict[str, int] = {}
    vocab_ids.update(SPECIAL_TOKENS)
    for offset, token in enumerate(sorted(vocab)):
        vocab_ids[token] = len(SPECIAL_TOKENS) + offset
    return WordPieceTokenizer(vocab_ids, merges)


class WordPieceTokenizer(BaseTokenizer):
    """WordPiece tokenizer; encodes via greedy longest-match-first."""

    type = "wordpiece"

    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]]) -> None:
        super().__init__(vocab)
        self.merges = merges
        self._max_token_len = max(len(t) for t in vocab)

    def _encode_word(self, word: str) -> list[str]:
        tokens: list[str] = []
        start = 0
        n = len(word)
        while start < n:
            end = min(start + self._max_token_len, n)
            found = False
            while start < end:
                substr = word[start:end]
                if substr in self.vocab:
                    tokens.append(substr)
                    found = True
                    break
                end -= 1
            if not found:
                tokens.append(UNK)
                start += 1
            else:
                start = end
        return tokens

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["merges"] = [list(pair) for pair in self.merges]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WordPieceTokenizer:
        vocab = data["vocab"]
        merges = [tuple(pair) for pair in data.get("merges", [])]
        if UNK not in vocab:
            raise ValueError("saved WordPiece tokenizer missing <unk> special token")
        return cls(vocab, merges)
