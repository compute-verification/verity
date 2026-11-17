"""The remote application's cryptographic error: its arithmetic against hand-computed cases, each term's Lean names
against the tree, each basis against its parameters and the lock, and what a research run records."""
from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path

import pytest
import run as R

BASES = {"proved", "assumed", "open", "excluded"}


def test_a_term_pools_over_plays_and_calls() -> None:
    t = R.Term("x", "", "assumed", (), coeffs=(Fraction(1, 8), Fraction(1, 16), Fraction(1, 32)))
    assert t.at(3, 4) == Fraction(3, 8) + Fraction(4, 16) + Fraction(16, 32)
    assert R.Term("y", "", "open", ()).at(3, 4) is None


def test_figures_by_hand_on_a_one_gpu_fleet() -> None:
    t = R.Term("x", "", "assumed", (), coeffs=(Fraction(1, 2 ** 10), Fraction(1, 2 ** 40), Fraction(0)))
    fl = {"world": {"id": "w", "machines": 1, "gpus": 1, "window_s": 1}}
    f = R.figures([t, R.Term("y", "", "open", ())], fl)["world"]["periods"]
    # a window: one play at 2^33 calls, 2^-10 + 2^33·2^-40 = 9·2^-10
    assert f["window"]["plays"] == 1 and f["window"]["sha512_calls_log2"] == R.HASH_RATE_LOG2
    assert f["window"]["total_log2"] == round(R.log2(Fraction(9, 2 ** 10)), 2) == f["window"]["share_log2"]
    week = dict(R.MONITORING_PERIODS)["week"]
    assert f["week"]["plays"] == week and f["week"]["terms_log2"] == {"x": f["week"]["total_log2"]}
    assert f["week"]["total_log2"] == round(R.log2(week * Fraction(9, 2 ** 10)), 2)


def test_the_closed_terms_by_hand() -> None:
    by = {t.name: t for t in R.terms()}
    assert by["hiding"].coeffs == (Fraction(1, 2 ** 207), 0, 0)
    assert by["file_finders"].coeffs == (Fraction(1, 2 ** 510), 0, 0)
    assert by["read_finder"].coeffs == (0, Fraction(1, 2 ** 255), 0)
    assert by["binding"].coeffs[1] == 2 ** 20 * Fraction(8, 5) * 2 * (1 + 2 ** 23) / 2 ** 256
    assert by["roots"].coeffs == (Fraction(2 ** 40, 2 ** 256), 0, 0)


def test_irrationals_are_bounded_on_the_safe_side() -> None:
    for x in (Fraction(4), Fraction(1, 2 ** 513), Fraction(2 ** 23 * 277)):
        s = R.sqrt_up(x)
        assert x <= s * s <= x * (1 + Fraction(1, 2 ** 60)) ** 2
    assert R.E_LOW < Fraction(27182819, 10 ** 7)


def _declared(name: str, path: Path) -> bool:
    last = name.rsplit(".", 1)[-1]
    pat = re.compile(rf"^\s*(?:(?:noncomputable|private|protected)\s+)*(?:def|theorem|lemma|abbrev|structure|class)\s+"
                     rf"(?:[\w.]*\.)?{re.escape(last)}(?![\w.])", re.M)
    return bool(pat.search(path.read_text(encoding="utf-8")))


def test_each_term_names_lean_declarations_that_exist() -> None:
    missing = [f"{t.name}: {n} in {f}" for t in R.terms() for n, f in t.lean if not _declared(n, R.ROOT / f)]
    assert not missing, "a term names a Lean declaration its file doesn't declare (fix the name or the file):\n" + "\n".join(missing)
    assert _declared("Verity.Sampled.remote_soundness_deployedReady",
                     R.ROOT / "security/Proofs/Properties/Verity/Sampled/OpensAt.lean") is True


def test_a_term_is_proved_only_at_parameters_lean_fixes() -> None:
    for t in R.terms():
        assert t.basis in BASES and t.lean, t.name
        assert (t.coeffs is None) == (t.basis in ("open", "excluded")), t.name
        if t.basis == "proved":
            assert all(R.PARAMS[p].basis in ("lean", "code") for p in t.params), t.name
        assert set(t.params) <= set(R.PARAMS), t.name


def test_the_guarantee_and_its_assumptions_come_from_the_locks() -> None:
    g = R.guarantee()
    mine = g[R.GUARANTEE]
    assert {"ecr/sha-512", "cr/sha-512", "euf-cma/lms-sha512-n64"} <= set(mine["assumptions"])
    assert "Verity.Vocab.PouwSound" in mine["open"] and "Verity.Vocab.SpotSound" in g[R.COMPOSITE]["open"]
    for t in R.terms():
        assert set(t.assumptions) <= set(mine["assumptions"]), t.name


def test_a_research_run_records_each_metric_with_the_headline_last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                  capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20261114-000000-0000")
    assert R.main(["--positions-log2", "10"]) == 0 and not list(tmp_path.glob("*/result.json"))
    assert "--positions-log2" in capsys.readouterr().out and R.PARAMS["positions"].value == 2 ** 20
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path))
    assert R.main([]) == 0
    res = json.loads((tmp_path / "result.json").read_text())
    ms = res["measurements"]
    assert len(ms) == len(R.METRICS) * 2 * len(R.MONITORING_PERIODS) and {m["mark"] for m in ms} == {"modeled"}
    for metric in R.METRICS:
        mine = [m for m in ms if m["name"] == metric]
        assert (mine[-1]["over"], mine[-1]["period"]) == R.HEADLINE
    fig = json.loads((tmp_path / R.KEPT).read_text())
    head = fig["fleets"]["world"]["periods"]["month"]
    assert [m["value"] for m in ms if m["name"] == "soundness/remote-crypto-error-log2"][-1] == head["total_log2"]
    assert res["validation"]["status"] == "passed" and "evaluated in Python" in fig["headline"]
