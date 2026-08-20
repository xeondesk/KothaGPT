"""WS-9: Bangla benchmark runner.

Usage:
    python -m evals.run --suite evals/suites/bangla.yaml
    python -m evals.run --suite evals/suites/bangla.yaml --split dev --target mock
    python -m evals.run --suite evals/suites/bangla.yaml --target api

Targets:
- ``mock``: returns the gold reference for every prompt (harness smoke test;
  QA exact-match and translation/summarization ROUGE should hit 100%).
- ``api``: calls the services.api ``MockBackend`` over its ChatCompletion
  request path.

Writes ``evals/results/<date>-<name>.json`` and ``.REPORT.md``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .metrics import (
    bengali_script_ratio,
    exact_match,
    language_detection,
    mean_ci,
    rouge,
)

_SUITE_DIR = Path(__file__).parent / "suites"
_RESULTS_DIR = Path(__file__).parent / "results"
_DATA_DIR = Path(__file__).parent.parent / "data/benchmarks/bangla/v1"

# Task -> callable(prediction, reference) -> dict[str, float]
_SCORERS: dict[str, Callable[[str, str], dict[str, float]]] = {
    "bangla_qa": lambda p, r: {"exact_match": exact_match(p, r)},
    "bangla_translation": lambda p, r: {**rouge(p, r), "exact_match": exact_match(p, r)},
    "bangla_summarization": lambda p, r: {**rouge(p, r)},
    "bangla_generation": lambda p, r: {},
}

# Task -> callable(prediction) -> dict[str, float] (language quality, all tasks).
def _lang_metrics(prediction: str) -> dict[str, float]:
    return {
        "bengali_script_ratio": bengali_script_ratio(prediction),
        "language_correct": 1.0 if language_detection(prediction) == "bn" else 0.0,
    }


class Target:
    def generate(self, prompt: str, record: dict[str, Any]) -> str:
        raise NotImplementedError


class MockTarget(Target):
    """Returns the gold reference; validates the harness end to end."""

    def generate(self, prompt: str, record: dict[str, Any]) -> str:
        return record["reference"]


class ApiTarget(Target):
    """Calls services.api.MockBackend through its ChatCompletion request path."""

    def __init__(self) -> None:
        from services.api.api.schemas import ChatCompletionRequest, Message
        from services.api.core.mock_backend import MockBackend

        self._backend = MockBackend()
        self._req = ChatCompletionRequest
        self._msg = Message

    def generate(self, prompt: str, record: dict[str, Any]) -> str:
        request = self._req(
            model="kothagpt",
            messages=[self._msg(role="user", content=prompt)],
        )
        response = self._backend.chat(request)
        return response.choices[0].message.content


def _load_yaml_suite(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_instances(task: str, data_dir: Path, split: str) -> list[dict[str, Any]]:
    path = data_dir / f"{task}.jsonl"
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if split == "all" or record.get("split") == split:
                records.append(record)
    return records


def _score(prediction: str, record: dict[str, Any]) -> dict[str, float]:
    task = record["task"]
    scorer = _SCORERS[task]
    return {**scorer(prediction, record.get("reference", "")), **_lang_metrics(prediction)}


def run_suite(
    suite_path: Path,
    *,
    data_dir: Path = _DATA_DIR,
    split: str = "all",
    target: Target | None = None,
) -> dict[str, Any]:
    suite = _load_yaml_suite(suite_path)
    name = suite.get("name", "suite")
    target = target or MockTarget()

    per_task: dict[str, list[dict[str, Any]]] = {}
    for task in suite.get("tasks", []):
        instances = _load_instances(task, data_dir, split)
        scored: list[dict[str, Any]] = []
        for record in instances:
            prompt = record.get("prompt") or record.get("source_text") or record.get("source") or ""
            prediction = target.generate(prompt, record)
            scored.append(
                {
                    "record_id": record["record_id"],
                    "split": record.get("split", "dev"),
                    "prediction": prediction,
                    "reference": record.get("reference", ""),
                    "scores": _score(prediction, record),
                }
            )
        per_task[task] = scored
        print(f"  {task}: {len(instances)} instances")

    summary: dict[str, Any] = {}
    for task, scored in per_task.items():
        keys: set[str] = set()
        for item in scored:
            keys.update(item["scores"])
        task_summary: dict[str, dict[str, float]] = {}
        for key in sorted(keys):
            values = [item["scores"][key] for item in scored]
            task_summary[key] = mean_ci(values)
        summary[task] = {"instances": len(scored), "metrics": task_summary}

    return {
        "suite": name,
        "target": type(target).__name__.lower().removesuffix("target"),
        "split": split,
        "date": _dt.date.today().isoformat(),
        "summary": summary,
        "tasks": {task: scored for task, scored in per_task.items()},
    }


def _render_report(results: dict[str, Any]) -> str:
    lines = [
        "# Bangla Evaluation Report",
        "",
        f"- **suite**: {results['suite']} v1",
        f"- **target**: {results['target']}",
        f"- **split**: {results['split']}",
        f"- **date**: {results['date']}",
        f"- **instances**: {sum(t['instances'] for t in results['summary'].values())}",
        "",
        "## Results",
        "",
    ]
    for task, summary in results["summary"].items():
        lines.append(f"### {task} ({summary['instances']} instances)")
        lines.append("| metric | mean | std | 95% CI |")
        lines.append("| --- | --- | --- | --- |")
        for metric, ci in summary["metrics"].items():
            lines.append(
                f"| {metric} | {ci['mean']:.4f} | {ci['std']:.4f} | "
                f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}] |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default=str(_SUITE_DIR / "bangla.yaml"))
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument("--target", choices=("mock", "api"), default="mock")
    parser.add_argument("--data-dir", default=str(_DATA_DIR))
    parser.add_argument("--out-dir", default=str(_RESULTS_DIR))
    parser.add_argument("--name", default="bangla")
    args = parser.parse_args(argv)

    target: Target = MockTarget() if args.target == "mock" else ApiTarget()
    results = run_suite(
        Path(args.suite),
        data_dir=Path(args.data_dir),
        split=args.split,
        target=target,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{results['date']}-{args.name}.json"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {json_path}")
    report_path = out_dir / f"{results['date']}-{args.name}.REPORT.md"
    report_path.write_text(_render_report(results), encoding="utf-8")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())