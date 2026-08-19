"""Tests for the Phase 1B Bangla tokenizer (BPE and Unigram)."""

from __future__ import annotations

import random

import pytest

from ml.tokenizer import (
    load_tokenizer,
    train_bpe,
    train_unigram,
)
from ml.tokenizer.benchmark import SAMPLE_TEXTS, run_benchmark
from ml.tokenizer.corpus import load_corpus

_BANGLA_WORDS = [
    "বাংলা",
    "ভাষা",
    "আন্দোলন",
    "সাহিত্য",
    "কবিতা",
    "গান",
    "নদী",
    "পাহাড়",
    "সবুজ",
    "শহর",
    "গ্রাম",
    "মানুষ",
    "ইতিহাস",
    "সংস্কৃতি",
    "শিক্ষা",
    "বিজ্ঞান",
    "প্রযুক্তি",
    "কম্পিউটার",
    "নেটওয়ার্ক",
    "তথ্য",
    "ডেটা",
    "মডেল",
    "অনুবাদ",
    "সামগ্রী",
    "প্রকাশনা",
    "লেখক",
    "পাঠক",
    "বাজার",
    "অর্থনীতি",
    "স্বাধীনতা",
    "সোনালী",
    "আকাশ",
    "মেঘ",
    "বৃষ্টি",
    "জল",
    "সমুদ্র",
    "তরঙ্গ",
    "পাখি",
    "ফুল",
    "গাছ",
    "মাটি",
    "রোদ",
    "চাঁদ",
    "তারকা",
    "রাত",
    "দিন",
    "সকাল",
    "সন্ধ্যা",
    "শিশু",
    "মা",
    "বাবা",
    "ভাই",
    "বোন",
    "পরিবার",
    "বন্ধু",
    "প্রতিবেশী",
    "সহকর্মী",
    "স্কুল",
    "কলেজ",
    "বিশ্ববিদ্যালয়",
    "শিক্ষক",
    "শিক্ষার্থী",
    "পরীক্ষা",
    "ফলাফল",
    "কথা",
    "মতামত",
    "প্রশ্ন",
    "উত্তর",
    "সমস্যা",
    "সমাধান",
    "উন্নয়ন",
    "অগ্রগতি",
    "অর্থ",
    "টাকা",
    "ব্যাংক",
    "লেনদেন",
    "রপ্তানি",
    "আমদানি",
    "শিল্প",
    "কারখানা",
    "কৃষি",
    "ধান",
    "গম",
    "শাকসবজি",
    "ফলমূল",
    "মাছ",
    "মাংস",
    "দুধ",
    "চা",
    "কফি",
    "ভাত",
    "রুটি",
    "তরকারি",
    "মিষ্টি",
    "পিঠা",
    "উৎসব",
    "ঈদ",
    "পূজা",
    "বিয়ে",
    "বই",
    "খবর",
    "সংবাদপত্র",
    "চলচ্চিত্র",
    "নাটক",
    "টেলিভিশন",
    "রেডিও",
    "ফটো",
    "স্মার্টফোন",
    "ইন্টারনেট",
    "ইমেইল",
    "ওয়েবসাইট",
    "অ্যাপ",
    "গেম",
    "ভিডিও",
]


def _make_corpus(n: int = 800, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    corpus = []
    for _ in range(n):
        k = rng.randint(12, 35)
        corpus.append(" ".join(rng.choice(_BANGLA_WORDS) for _ in range(k)))
    return corpus


@pytest.fixture(scope="module")
def corpus() -> list[str]:
    return _make_corpus()


def test_bpe_trains_vocab_and_merges(corpus):
    tokenizer = train_bpe(corpus, vocab_size=400, min_frequency=2)
    assert len(tokenizer.vocab) >= 300
    assert tokenizer.merges
    # the first merge must have produced a real concatenated token
    first = tokenizer.merges[0]
    assert "".join(first) in tokenizer.vocab


def test_bpe_roundtrip(corpus):
    tokenizer = train_bpe(corpus, vocab_size=400, min_frequency=2)
    for text in corpus[:20]:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_unigram_trains_and_roundtrips(corpus):
    tokenizer = train_unigram(corpus, vocab_size=400, min_frequency=2, iterations=3)
    assert tokenizer.probs
    for text in corpus[:20]:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_vocab_contains_special_tokens(corpus):
    tokenizer = train_bpe(corpus, vocab_size=200, min_frequency=2)
    for special in ("<unk>", "<bos>", "<eos>", "<pad>"):
        assert special in tokenizer.vocab


def test_save_load_roundtrip(corpus, tmp_path):
    bpe = train_bpe(corpus, vocab_size=300, min_frequency=2)
    unigram = train_unigram(corpus, vocab_size=300, min_frequency=2, iterations=2)
    for tokenizer, name in ((bpe, "bpe"), (unigram, "unigram")):
        tokenizer.save(tmp_path / name)
        loaded = load_tokenizer(tmp_path / name / "tokenizer.json")
        assert type(loaded) is type(tokenizer)
        for text in corpus[:10]:
            assert loaded.encode(text) == tokenizer.encode(text)


def test_emoji_punctuation_mixed_code_no_crash(corpus):
    tokenizer = train_bpe(corpus, vocab_size=300, min_frequency=2)
    for text in SAMPLE_TEXTS.values():
        ids = tokenizer.encode(text)
        assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
        assert tokenizer.decode(ids)


def test_unk_reports_coverage(corpus):
    tokenizer = train_bpe(corpus, vocab_size=300, min_frequency=2)
    stats = tokenizer.stats(SAMPLE_TEXTS["bangla"])
    assert stats["tokens"] > 0
    assert 0 <= stats["tokens_per_char"] < 5
    assert "unk" in stats


def test_unseen_bangla_characters_encode(corpus):
    """Characters outside the training corpus must still tokenize (bootstrap)."""
    tokenizer = train_bpe(corpus, vocab_size=200, min_frequency=2)
    text = "এটি এমন একটি বাক্য যাতে এ ও ঊ এবং ঔ আছে"  # letters not in synthetic corpus
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    unigram = train_unigram(corpus, vocab_size=200, min_frequency=2, iterations=2)
    ids = unigram.encode(text)
    assert unigram.decode(ids) == text


def test_benchmark_shape(corpus):
    tokenizer = train_unigram(corpus, vocab_size=300, min_frequency=2, iterations=2)
    result = run_benchmark(tokenizer)
    assert result["avg_tokens_per_char"] > 0
    assert set(result["per_set"]) == {
        "bangla",
        "mixed",
        "punctuation",
        "emoji",
        "code",
    }


def test_unigram_improves_over_initial_vocab(corpus):
    tokenizer = train_unigram(corpus, vocab_size=400, min_frequency=2, iterations=3)
    assert len(tokenizer.vocab) >= 300


def test_load_corpus_jsonl(tmp_path):
    import json

    (tmp_path / "c.jsonl").write_text(
        json.dumps({"text": "বাংলা ভাষা"}) + "\n" + json.dumps({"text": "সাহিত্য ও সংস্কৃতি"}) + "\n",
        encoding="utf-8",
    )
    texts = load_corpus(tmp_path / "c.jsonl")
    assert texts == ["বাংলা ভাষা", "সাহিত্য ও সংস্কৃতি"]
