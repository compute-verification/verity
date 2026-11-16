"""The kit tests' stand-ins: a host whose nvidia-smi and git answer as a test says, and the result a writer wrote."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from verity_bench import kit

SHA = "c" * 40
RUN = "run-1"
CARD = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
GPU0, GPU1 = "GPU-11111111-2222-3333-4444-555555555555", "GPU-66666666-7777-8888-9999-000000000000"
HEADER = ("+-----------------------------------------------------------------------------------------+\n"
          "| NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0     |\n")


def _row(uuid: str, sm: str = "2100", mem: str = "9501", name: str = CARD, driver: str = "580.95.05") -> str:
    return f"{name}, {uuid}, {driver}, {sm}, {mem}\n"


def _host(monkeypatch: pytest.MonkeyPatch, answers: dict[str, tuple[int, str]] | None, sha: str | None = SHA) -> list[list[str]]:
    """A host whose nvidia-smi answers `answers` (by `query` or `header`; None: a host without one) and whose git answers
    nothing; the queries asked, in order."""

    asked: list[list[str]] = []

    def run(argv: list[str]) -> tuple[int, str]:
        asked.append(list(argv))
        if argv[0] == "git":
            return 128, "fatal: not a git repository"
        assert argv[0] == "nvidia-smi" and answers is not None, argv
        return answers["query" if any(a.startswith("--query-gpu=") for a in argv) else "header"]

    monkeypatch.setattr(kit.shutil, "which", lambda name: None if answers is None else f"/usr/bin/{name}")
    monkeypatch.setattr(kit, "_run", run)
    monkeypatch.delenv("GPU_LEASE_UUID", raising=False)
    # a pytest started inside a research run holds that run's directory, whose result no test may write
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    monkeypatch.setenv("RESEARCH_RUN_ID", RUN)
    if sha is None:
        monkeypatch.delenv("RESEARCH_SOURCE_SHA", raising=False)
    else:
        monkeypatch.setenv("RESEARCH_SOURCE_SHA", sha)
    return asked


@pytest.fixture
def no_gpu(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    return _host(monkeypatch, None)


def _result(out: Path) -> dict:
    return json.loads((out / kit.RESULT).read_text())
