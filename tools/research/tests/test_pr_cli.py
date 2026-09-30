"""`research pr` against the reference model of the PR routes (`research.jobs.prs`), through `ModelClient`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.jobs import prs, trains
from research.jobs.cli import pr_main
from research.jobs.client import ModelClient, Unreachable

VECTORS = json.loads(Path(prs.__file__).with_name("pr_vectors.json").read_text())


@pytest.fixture
def site():
    setup = VECTORS["setup"]
    s = prs.PRService(epoch=VECTORS["epoch"], coordinators=json.loads(json.dumps(setup["coordinators"])))
    for t in setup["tokens"]:
        s.mint(t["secret"], t["name"], t["scopes"])
    gh = setup["github"]
    s.github = trains.FakeGitHub(commits=json.loads(json.dumps(gh["commits"])), refs=dict(gh["refs"]),
                                 prs={p["number"]: {**p, "labels": list(p["labels"])} for p in gh["prs"]})
    s.sync(0)
    return s


def cli(site, token="tok-pous", at=60.0):
    return lambda *argv: pr_main(list(argv), ModelClient(site, token, lambda: at))


def test_list_and_show_print_the_rows(site, capsys):
    assert cli(site)("list", "--coordinator", "pous") == 0
    out = capsys.readouterr().out.splitlines()
    assert [line.split()[0] for line in out] == ["#15", "#10"]
    assert "pous/pous-circuits" in out[1] and "by a push" in out[1]
    assert cli(site)("list", "--state", "unowned", "--json") == 0
    assert [p["number"] for p in json.loads(capsys.readouterr().out)] == [14]
    assert cli(site)("show", "10") == 0
    out = capsys.readouterr().out
    assert "POUS circuits" in out and "owner from body" in out and "pr.seen" in out


def test_writes_send_owner_blocker_renew_and_close(site, capsys):
    run = cli(site)
    assert run("blocker", "10", "RC's rerun of #364's check", "--ref", "note:pous/x") == 0
    assert (site.rows[10].blocker, site.rows[10].ref, site.rows[10].blocker_by) == ("RC's rerun of #364's check", "note:pous/x", "pous")
    assert run("blocker", "10", "--clear") == 0 and site.rows[10].blocker is None
    assert run("renew", "15", "--blocker", "the audit") == 0
    assert (site.rows[15].blocker, site.rows[15].touched_by) == ("the audit", "pous")
    assert run("own", "14", "pous/pous-circuits") == 0 and (site.rows[14].coordinator, site.rows[14].owner_from) == ("pous", "set")
    capsys.readouterr()
    assert run("close", "14", "--reason", "superseded by #16") == 0
    assert capsys.readouterr().out == "#14 closed, with the comment: superseded by #16\n"
    assert site.github.prs[14]["state"] == "closed"
    assert site.github.comments == [{"pr": 14, "body": "Closed by pous (pous): superseded by #16"}]


def test_a_refusal_exits_1_with_the_sites_reason_and_a_bad_invocation_2(site, capsys):
    assert cli(site, token="tok-coordinator")("close", "10", "--reason", "stale") == 1
    assert "#10 is pous's to close" in capsys.readouterr().err
    assert cli(site, token="tok-reader")("renew", "10") == 1
    assert "403 the token lacks prs:write" in capsys.readouterr().err
    assert cli(site)("own", "10", "pous") == 1
    for argv in (["blocker", "10"], ["blocker", "10", "x", "--clear"], ["blocker", "10", "--clear", "--ref", "r"], ["renew", "10", "--ref", "r"]):
        assert cli(site)(*argv) == 2, argv
    assert site.rows[10].touched_by is None


def test_an_unreachable_site_exits_4(capsys):
    class Down:
        def call(self, method, path, body=None):
            raise Unreachable(f"{method} {path}: HTTP 503")
    assert pr_main(["show", "10"], Down()) == 4
    assert "unreachable" in capsys.readouterr().err
