"""`Checks`: the one check-and-mutation registry of an exact script, with its printed lines, JSON verdict and exit code.

A script makes one registry, records each claim with `check` and each mutation with `mutation` (or a block of checks under
`mutant`), and ends with ``sys.exit(checks.finish())``::

    checks = Checks()                      # reads --mutations and --verdict PATH from sys.argv; the rest is checks.args
    checks.check("A rank 2L", rank == 2 * L, f"rank {rank}")
    if checks.mutations:
        checks.mutation("A with even weights", rank_even == 0, f"rank {rank_even}")
    sys.exit(checks.finish())

Lines print as ``[OK  ] name: detail`` and ``[FAIL] name: detail``; a detail of None prints no colon. A mutation prints as
``[OK  ] mutation caught: name`` or ``[FAIL] mutation NOT caught: name``. `finish` prints ``N/N checks OK`` (with
``, M/M mutations caught`` once a mutation is recorded), writes the verdict, and returns 0 only when every check passed and
every mutation was caught. The verdict JSON holds no times or paths, so two runs of one script agree byte for byte.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path

SCHEMA = "exact-kit/verdict/v1"


class Checks:
    """The registry. ``argv`` defaults to ``sys.argv[1:]``; ``script`` (the file the verdict names) to ``sys.argv[0]``."""

    def __init__(self, argv: list[str] | None = None, script: str | None = None, out=None):
        args = list(sys.argv[1:] if argv is None else argv)
        self.mutations = "--mutations" in args
        self.verdict_path = None
        rest = []
        it = iter(args)
        for a in it:
            if a == "--mutations":
                continue
            if a == "--verdict":
                self.verdict_path = next(it, None)
                if self.verdict_path is None:
                    raise SystemExit("--verdict needs a path")
            elif a.startswith("--verdict="):
                self.verdict_path = a.split("=", 1)[1]
            else:
                rest.append(a)
        self.args = rest
        self.script = sys.argv[0] if script is None else script
        self.out = out
        self.checks: list[dict] = []
        self.mutants: list[dict] = []
        self._inside: list[dict] | None = None

    def _print(self, line: str) -> None:
        print(line, file=self.out or sys.stdout, flush=True)

    @staticmethod
    def _line(ok: bool, text: str, detail) -> str:
        return f"[{'OK  ' if ok else 'FAIL'}] {text}" + ("" if detail is None else f": {detail}")

    def check(self, name: str, ok, detail=None) -> bool:
        """Record and print one claim; returns it as a bool."""
        ok = bool(ok)
        entry = {"name": name, "ok": ok, "detail": None if detail is None else str(detail)}
        (self.checks if self._inside is None else self._inside).append(entry)
        self._print(self._line(ok, name, detail))
        return ok

    def mutation(self, name: str, caught, detail=None) -> bool:
        """Record and print one mutation: ``caught`` says the claim it breaks failed under it."""
        caught = bool(caught)
        self.mutants.append({"name": name, "caught": caught, "detail": None if detail is None else str(detail)})
        self._print(self._line(caught, f"mutation {'caught' if caught else 'NOT caught'}: {name}", detail))
        return caught

    @contextlib.contextmanager
    def mutant(self, name: str):
        """Run checks under a mutation: they count as the mutant's, not the script's, and it is caught iff one fails."""
        if self._inside is not None:
            raise RuntimeError("mutants do not nest")
        self._inside = inner = []
        try:
            yield inner
        finally:
            self._inside = None
        failed = sum(not c["ok"] for c in inner)
        self.mutation(name, failed > 0, f"{failed} of {len(inner)} checks fail")

    @property
    def ok(self) -> bool:
        return all(c["ok"] for c in self.checks) and all(m["caught"] for m in self.mutants)

    def summary(self) -> str:
        n_ok = sum(c["ok"] for c in self.checks)
        s = f"{n_ok}/{len(self.checks)} checks OK"
        if self.mutants:
            s += f", {sum(m['caught'] for m in self.mutants)}/{len(self.mutants)} mutations caught"
        return s + ("" if self.ok else "  <-- FAILURES")

    def verdict(self) -> dict:
        script = Path(self.script)
        digest = hashlib.sha256(script.read_bytes()).hexdigest() if script.is_file() else None
        return {"schema": SCHEMA, "script": {"name": script.name, "sha256": digest}, "args": self.args,
                "mutations_run": self.mutations, "checks": self.checks, "mutations": self.mutants,
                "counts": {"checks": len(self.checks), "checks_ok": sum(c["ok"] for c in self.checks),
                           "mutations": len(self.mutants), "mutations_caught": sum(m["caught"] for m in self.mutants)},
                "ok": self.ok}

    def finish(self) -> int:
        """Print the summary, write the verdict (``--verdict PATH``, else ``$OUT/<script>.verdict.json`` under a run), and
        under `research run` the run's result.json. Returns the exit code."""
        self._print(f"\n{self.summary()}")
        doc = self.verdict()
        path = self.verdict_path
        if path is None and os.environ.get("OUT"):
            path = os.path.join(os.environ["OUT"], f"{Path(self.script).stem}.verdict.json")
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(dumps(doc), encoding="utf-8")
        _result(doc, path, self.summary())
        return 0 if self.ok else 1


def dumps(doc: dict) -> str:
    """The verdict's one byte form."""
    return json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def _result(doc: dict, verdict_path: str | None, detail: str) -> None:
    """The run's ``research/result/v0.1`` document (tools/research/src/research/result.py), written without importing
    `research`, as benchmarks/pouw/run_result.py does."""
    run_dir, run_id = os.environ.get("RESEARCH_RUN_DIR"), os.environ.get("RESEARCH_RUN_ID")
    if not run_dir or not run_id:
        return
    rel = None
    if verdict_path:
        try:
            rel = str(Path(verdict_path).resolve().relative_to(Path(run_dir).resolve()))
        except ValueError:
            rel = None
    out = {"schema": "research/result/v0.1", "run_id": run_id,
           "workload_fingerprint": {"script": doc["script"], "args": doc["args"], "mutations_run": doc["mutations_run"],
                                    "source_sha": os.environ.get("RESEARCH_SOURCE_SHA")},
           "validation": {"status": "passed" if doc["ok"] else "failed", "checker": "exact_kit.checks", "detail": detail,
                          "evidence": [rel] if rel else []},
           "measurements": [{"name": k, "value": float(v), "unit": "count"} for k, v in doc["counts"].items()],
           "measurement_files": [rel] if rel else [], "artifacts": []}
    d = Path(run_dir)
    tmp = d / f"result.json.tmp.{os.getpid()}"
    tmp.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, d / "result.json")
