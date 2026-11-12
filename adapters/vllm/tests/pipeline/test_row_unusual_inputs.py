"""What two of a row's readers give for an input no run has written: the card's reading of a log whose first line nests past the
interpreter's depth (`verity_vllm.pipeline.report.log_header`), and the manifest build asked for a workload Program whose recorded
digest is the text ``-`` with no function to ask the digest of (`verity_vllm.pipeline.row_build.required_manifest`).

The first has no run header, as any first line that is not one; the second fails where the digest is asked for.
"""
from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import verity_vllm.pipeline.report as CARD
import verity_vllm.pipeline.row_build as RB

#: more levels than any interpreter decodes
DEPTH = 100_000


@pytest.mark.parametrize("first", [b"[" * DEPTH + b"]" * DEPTH, b'{"a":' * DEPTH + b"1" + b"}" * DEPTH, b"[" * DEPTH],
                         ids=["lists", "objects", "lists-left-open"])
def test_the_card_has_no_run_header_for_a_first_line_nested_past_the_interpreters_depth(tmp_path: Path, first: bytes) -> None:
    (tmp_path / "log.jsonl").write_bytes(first + b"\n")
    (tmp_path / "log.jsonl.gz").write_bytes(gzip.compress(first + b"\n"))
    assert CARD.log_header(tmp_path / "log.jsonl") is None and CARD.log_header(tmp_path / "log.jsonl.gz") is None


#: `manifest build-global --workload-digest -` as far as the row sees it: it writes what its stdin gave it, whole
_READS_ITS_STDIN = """
import sys
a = sys.argv[1:]
open(a[a.index("--out") + 1] + ".stdin", "w").write(sys.stdin.read())
"""


def test_a_manifest_asked_for_a_digest_that_reads_as_the_pipe_with_no_function_fails_at_the_call(tmp_path: Path, monkeypatch) -> None:
    """A batch's manifest takes the workload digest from its stdin when the row asks it with ``-``, the text the row passes when it has
    a function to ask the digest of.  A workload Program whose recorded digest is that text, with no function, is asked the same way:
    the call fails as a call of None does, and the child still gets its line and the end of its stdin."""
    (tmp_path / "build_request_LP4_T2").mkdir()
    (tmp_path / "build_workload").mkdir()
    (tmp_path / "build_workload" / "workload_program.json").write_text(json.dumps({"workload_digest": "-"}))
    started: list[subprocess.Popen[bytes]] = []
    popen = subprocess.Popen

    def recorded(*a: Any, **k: Any) -> subprocess.Popen[bytes]:
        started.append(popen(*a, **k))
        return started[-1]

    monkeypatch.setattr(RB.subprocess, "Popen", recorded)
    row = SimpleNamespace(ck_=None, B=2, d=str(tmp_path), wl="w.json", tree=str(tmp_path), child_env=None, log=lambda line: None,
                          tl=lambda *a: None, cli=lambda *a: [sys.executable, "-c", _READS_ITS_STDIN, *a])
    try:
        with pytest.raises(TypeError, match="'NoneType' object is not callable"):
            RB.required_manifest(row, "build")
    finally:
        for child in started:
            if child.stdin is not None and not child.stdin.closed:
                child.stdin.close()
            child.wait()
    assert len(started) == 1 and started[0].returncode == 0
    assert (tmp_path / "manifest.json.stdin").read_text() == "\n"
