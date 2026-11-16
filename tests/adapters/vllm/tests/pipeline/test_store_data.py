"""What `verity_vllm.pipeline.store_data` makes of `research data`'s answer: the command it runs, the source a select's summary line
names, the path a fetch printed last, and a refusal with what the CLI said, which names the bundle where the store has no remote."""
from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from verity_vllm.pipeline import store_data as SD

ART = "art:" + "1" * 64


@pytest.fixture
def research_data(monkeypatch):
    """`research data` stood in for: each call's command is recorded and answered by the next (exit code, stdout, stderr) given."""
    commands: list[list[str]] = []
    answers: list[tuple[int, str, str]] = []

    def run(command, capture_output, text):
        assert capture_output and text
        commands.append(command)
        return subprocess.CompletedProcess(command, *answers.pop(0))

    monkeypatch.setattr(SD, "subprocess", SimpleNamespace(run=run))
    return commands, answers


def test_a_select_says_whether_it_read_the_index_or_the_catalog(research_data):
    commands, answers = research_data
    answers += [(0, "", "2 artifact(s) from the index (/s/index.sqlite, complete as of 2026-10-11T18:00:00Z), 1 of them only here\n"),
                (0, "", "warning: no index\n0 artifact(s) from the catalog alone (this store's, /s/catalog; no index: none built)\n")]
    assert [SD.selected_from("--kind", "vllm-build/v1"), SD.selected_from("--kind", "vllm-build/v1")] == ["the index", "the catalog"]
    assert commands == [[sys.executable, "-m", "research", "data", "select", "--kind", "vllm-build/v1"]] * 2
    answers.append((0, f"{ART}\n", "1 artifact(s) from a mirror\n"))
    with pytest.raises(RuntimeError, match=r"^research data select --kind k --ids: its summary names neither the index nor the catalog: "
                                           r"'1 artifact\(s\) from a mirror'$"):
        SD.selected_from("--kind", "k", "--ids")


def test_a_fetch_gives_the_path_printed_last_and_a_refusal_says_what_the_cli_said(research_data, tmp_path):
    commands, answers = research_data
    answers.append((0, f"fetching 2 files\n{tmp_path}/payload\n", ""))
    assert SD.fetch(ART, tmp_path, ["a/b.json", "c.json"]) == tmp_path / "payload"
    assert commands[-1][4:] == ["fetch", ART, "--to", str(tmp_path), "--path=a/b.json", "--path=c.json"]
    answers.append((1, "", f"{ART}: not in this store, and no remote configured\n"))
    with pytest.raises(RuntimeError, match=rf"^research data fetch {ART}: rc=1: {ART}: not in this store, and no remote configured: "
                                           r"a research run job has no store; the launcher writes the row's bundle \(`verity-vllm "
                                           r"records fetch ROW --to FILE`\)"):
        SD.fetch(ART)
    assert commands[-1][4:] == ["fetch", ART]
    answers.append((2, "", f"{ART}: the remote refused the read\n"))
    with pytest.raises(RuntimeError, match=rf"^research data fetch {ART}: rc=2: {ART}: the remote refused the read$"):
        SD.fetch(ART)
    answers.append((0, '[{"path": "a/b.json"}, {"path": "c.json"}]', ""))
    assert SD.listing(ART) == ["a/b.json", "c.json"] and commands[-1][4:] == ["fetch", ART, "--list", "--json"]
