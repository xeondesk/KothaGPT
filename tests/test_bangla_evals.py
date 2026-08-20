"""Tests for the WS-9 Bangla eval harness (evals/metrics.py, evals/run.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.metrics import (
    bengali_script_ratio,
    exact_match,
    language_detection,
    mean_ci,
    normalize_answer,
    rouge,
)
from evals.run import MockTarget, run_suite

SUITE = Path(__file__).parent.parent / "evals/suites/bangla.yaml"
DATA = Path(__file__).parent.parent / "data/benchmarks/bangla/v1"


def test_exact_match_ignores_punctuation_and_case():
    assert exact_match("ঢাকা।", "ঢাকা") == 1.0
    assert exact_match("Hello!", "hello") == 1.0
    assert exact_match("বাংলা", "ইংরেজি") == 0.0


def test_normalize_answer_collapses_whitespace():
    assert normalize_answer("  বাংলা    ভাষা  ") == "বাংলা ভাষা"


def test_language_detection():
    assert language_detection("বাংলা ভাষা") == "bn"
    assert language_detection("Hello world") == "en"
    assert language_detection("Hello বাংলা") == "bn"


def test_bengali_script_ratio():
    assert bengali_script_ratio("বাংলা") == 1.0
    assert bengali_script_ratio("Bangla") == 0.0
    assert bengali_script_ratio("বাংলা ভাষা") == 1.0


def test_rouge_perfect_match_is_one():
    scores = rouge("বাংলা ভাষা বাংলাদেশের ভাষা", "বাংলা ভাষা বাংলাদেশের ভাষা")
    assert scores["rouge1"] == 1.0
    assert scores["rouge2"] == 1.0
    assert scores["rougeL"] == 1.0


def test_rouge_disjoint_is_zero():
    scores = rouge("এক দুই তিন", "চার পাঁচ ছয়")
    assert scores["rouge1"] == 0.0


def test_rouge_partial_overlap():
    scores = rouge("বাংলা ভাষা সুন্দর", "বাংলা ভাষা")
    assert scores["rouge1"] > 0.0
    assert scores["rougeL"] > 0.0


def test_mean_ci_bounds():
    stats = mean_ci([1.0, 2.0, 3.0])
    assert stats["mean"] == 2.0
    assert stats["ci_low"] <= stats["mean"] <= stats["ci_high"]


def test_mean_ci_empty():
    assert mean_ci([])["mean"] == 0.0


@pytest.mark.parametrize(
    "task",
    ["bangla_qa", "bangla_translation", "bangla_summarization", "bangla_generation"],
)
def test_suite_runs_all_tasks_with_mock_target(task):
    results = run_suite(SUITE, data_dir=DATA, split="all", target=MockTarget())
    summary = results["summary"][task]
    assert summary["instances"] > 0
    assert any(
        key in summary["metrics"] for key in ("exact_match", "rouge1", "bengali_script_ratio")
    )


def test_mock_target_returns_gold_references():
    record = {"reference": "বাংলা ভাষা"}
    assert MockTarget().generate("প্রশ্ন?", record) == "বাংলা ভাষা"


def test_qa_exact_match_is_perfect_under_mock():
    results = run_suite(SUITE, data_dir=DATA, split="dev", target=MockTarget())
    em = results["summary"]["bangla_qa"]["metrics"]["exact_match"]
    assert em["mean"] == 1.0


def test_translation_rouge_is_perfect_under_mock():
    results = run_suite(SUITE, data_dir=DATA, split="dev", target=MockTarget())
    r1 = results["summary"]["bangla_translation"]["metrics"]["rouge1"]
    assert r1["mean"] == 1.0
