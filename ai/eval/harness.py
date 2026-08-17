"""A small evaluation harness: loads JSONL task files (prompt + expected
answer/keywords), runs them through any callable "solver", and scores
results with pluggable metrics (exact match, keyword coverage).

Usage:
    from ai.eval.harness import EvalHarness, exact_match, keyword_coverage

    async def solver(prompt: str) -> str:
        return await my_model.complete(prompt)

    harness = EvalHarness.from_jsonl("ai/eval/tasks/qa_basic.jsonl")
    report = await harness.run(solver, metric=keyword_coverage)
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

Solver = Callable[[str], Awaitable[str]]
Metric = Callable[[str, "Task"], float]


@dataclass
class Task:
    id: str
    prompt: str
    expected: str = ""
    keywords: list[str] = field(default_factory=list)


@dataclass
class TaskResult:
    task_id: str
    prompt: str
    output: str
    score: float


@dataclass
class EvalReport:
    results: list[TaskResult]

    @property
    def mean_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    def to_dict(self) -> dict:
        return {
            "mean_score": self.mean_score,
            "n_tasks": len(self.results),
            "results": [r.__dict__ for r in self.results],
        }


def exact_match(output: str, task: Task) -> float:
    return 1.0 if output.strip().lower() == task.expected.strip().lower() else 0.0


def keyword_coverage(output: str, task: Task) -> float:
    if not task.keywords:
        return 0.0
    lowered = output.lower()
    hits = sum(1 for kw in task.keywords if kw.lower() in lowered)
    return hits / len(task.keywords)


class EvalHarness:
    def __init__(self, tasks: list[Task]) -> None:
        self.tasks = tasks

    @classmethod
    def from_jsonl(cls, path: str | Path) -> EvalHarness:
        tasks = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                tasks.append(
                    Task(
                        id=row["id"],
                        prompt=row["prompt"],
                        expected=row.get("expected", ""),
                        keywords=row.get("keywords", []),
                    )
                )
        return cls(tasks)

    async def run(self, solver: Solver, metric: Metric = keyword_coverage) -> EvalReport:
        results = []
        for task in self.tasks:
            output = await solver(task.prompt)
            score = metric(output, task)
            results.append(TaskResult(task_id=task.id, prompt=task.prompt, output=output, score=score))
        return EvalReport(results=results)
