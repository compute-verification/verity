"""The lint sites' readings (`lint_sites.py`) on made-up `lint_counts.py` output: each file goes to the lead whose `owns`
entry is its longest prefix, or to `unowned`; the sites past their entries are the lines' own `past`, never found minus
allowed; a list that couldn't be counted is named, with its reason, and summed into nothing; a list marked as no
ratchet's is named and makes nothing partial; the result holds the total and one reading for each lead of each of the
three numbers, the total last; and a run that counted no list writes none."""
from __future__ import annotations

import json
from pathlib import Path

import lint_sites as S
import pytest

RUN = "r20000101-000000-aaaa"
STAMP = {"card": "none", "gpus": [], "driver": None, "cuda": None, "node": "test", "commit": "d" * 40}
HANDLES = [{"handle": "top", "owns": ["tests/", "AGENTS.md"]},
           {"handle": "vllm", "owns": ["adapters/vllm/"]},
           {"handle": "circuits", "owns": ["adapters/", "tests/circuits/"]},
           {"handle": "dashboard-goals"}]
STDOUT = "".join(f"{line}\n" for line in (
    "p10_size\tadapters/vllm/a.py\t3\t2\t1",
    "p10_size\tadapters/torch/b.py\t1\t1\t0",
    "repo_env_reads\ttests/circuits/c.py\t4\t4\t0",
    "repo_env_reads\ttests/d.py\t2\t3\t1",
    "repo_env_reads\tAGENTS.md\t1\t0\t1",
    "repo_env_reads\tAGENTS.mdx\t5\t5\t2",
    "repo_ruff\t-\t-\t-\t-",
    "repo_python_verdicts\t-\tn/a\tn/a\tn/a",
))
STDERR = ("p10_size: 4 found in 2 file(s), 3 allowed in 2; 1 key(s) hold 1 more than their entries, 0 entr(y/ies) allow 0 "
          "more than is found\n"
          "p02_value_checks: 0 found in 0 file(s), 0 allowed in 0; 0 key(s) hold 0 more than their entries, 0 entr(y/ies) "
          "allow 0 more than is found\n"
          "repo_ruff: not counted: the tree pins ruff==0.17.2 and python has no ruff\n  and a second line\n"
          "repo_python_verdicts: not a ratchet's: its module has no allowed()\n"
          "lint_counts: dddddddddddd: 2 list(s) counted, 1 not counted, 1 no ratchet's; 4 found, 3 allowed; 5 past their "
          "entries by key, 4 allowed past what is found\n")
FOUND = {None: 16, "circuits": 5, "top": 3, "vllm": 3, S.UNOWNED: 5}
ALLOWED = {None: 15, "circuits": 5, "top": 3, "vllm": 2, S.UNOWNED: 5}
PAST = {None: 5, "circuits": 0, "top": 2, "vllm": 1, S.UNOWNED: 2}
NOT_COUNTED = "not counted: repo_ruff (the tree pins ruff==0.17.2 and python has no ruff)"


def test_a_file_goes_to_the_longest_owns_prefix_and_else_to_unowned() -> None:
    assert S.owner(HANDLES, "adapters/vllm/a.py") == "vllm"
    assert S.owner(HANDLES, "adapters/torch/b.py") == "circuits"
    assert S.owner(HANDLES, "tests/circuits/c.py") == "circuits"
    assert S.owner(HANDLES, "tests/d.py") == "top"
    assert S.owner(HANDLES, "AGENTS.md") == "top"
    assert S.owner(HANDLES, "AGENTS.mdx") == S.UNOWNED
    assert S.owner(HANDLES, "testsuite/e.py") == S.UNOWNED


def test_an_uncounted_list_is_named_and_not_summed() -> None:
    rows, uncounted, none = S.parse(STDOUT)
    assert uncounted == ["repo_ruff"] and none == ["repo_python_verdicts"]
    assert all(lint not in ("repo_ruff", "repo_python_verdicts") for lint, *_ in rows)
    assert S.sums(rows, HANDLES)[None] == (16, 15, 5)
    detail = S.detail(rows, uncounted, none, STDERR)
    assert ("p02_value_checks 0 found, 0 allowed, 0 past their entries; p10_size 4 found, 3 allowed, 1 past their entries; "
            "repo_env_reads 12 found, 12 allowed, 4 past their entries") in detail
    assert "Not counted, so left out of every sum: repo_ruff (the tree pins ruff==0.17.2 and python has no ruff)" in detail
    assert "No ratchet's, so never counted: repo_python_verdicts" in detail
    assert "second line" not in detail


def test_past_entries_is_the_lines_past_and_not_found_minus_allowed() -> None:
    got = S.sums(S.parse(STDOUT)[0], HANDLES)
    assert {m: p for m, (_, _, p) in got.items()} == PAST
    assert got[S.UNOWNED][0] - got[S.UNOWNED][1] == 0 < got[S.UNOWNED][2]


def result(tmp_path: Path, stdout: str, stderr: str) -> dict:
    S.write(S.kit.Writer(tmp_path, RUN, STAMP), *S.parse(stdout), HANDLES, stderr)
    return json.loads((tmp_path / S.kit.RESULT).read_text(encoding="utf-8"))


def test_the_result_holds_the_totals_and_each_leads_readings(tmp_path: Path) -> None:
    doc = result(tmp_path, STDOUT, STDERR)
    got = {(m["name"], m.get("member")): (m["value"], m["target"], m["mark"], m["unit"]) for m in doc["measurements"]}
    assert got == {**{(S.FOUND, m): (v, 0, "partial", "sites") for m, v in FOUND.items()},
                   **{(S.ALLOWED, m): (v, None, "partial", "sites") for m, v in ALLOWED.items()},
                   **{(S.PAST, m): (v, 0, "partial", "sites") for m, v in PAST.items()}}
    assert doc["validation"]["status"] == "passed" and "repo_ruff" in doc["validation"]["detail"]
    for name in (S.FOUND, S.ALLOWED, S.PAST):
        mine = [m for m in doc["measurements"] if m["name"] == name]
        assert mine[-1].get("member") is None and all(m.get("member") for m in mine[:-1]), "the total is written last"
        assert {tuple(m["inputs"]) for m in mine} == {("partial", NOT_COUNTED)}


def test_a_list_that_is_no_ratchets_is_named_and_makes_nothing_partial(tmp_path: Path) -> None:
    doc = result(tmp_path, "repo_env_reads\ttests/x.py\t2\t1\t1\nrepo_python_verdicts\t-\tn/a\tn/a\tn/a\n",
                 "repo_python_verdicts: not a ratchet's: its module has no allowed()\n")
    assert {m["mark"] for m in doc["measurements"]} == {"measured"}
    assert all(m["inputs"] == ["measured"] for m in doc["measurements"])
    assert "No ratchet's, so never counted: repo_python_verdicts" in doc["validation"]["detail"]


def test_a_run_that_counted_no_list_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="counted no list.*repo_ruff"):
        result(tmp_path, "repo_ruff\t-\t-\t-\t-\n", "repo_ruff: not counted: no ruff\n")
    assert not (tmp_path / S.kit.RESULT).exists()


def test_a_count_with_every_list_counted_is_measured(tmp_path: Path) -> None:
    doc = result(tmp_path, "repo_env_reads\ttests/x.py\t0\t1\t0\n", "")
    assert {m["mark"] for m in doc["measurements"]} == {"measured"}
