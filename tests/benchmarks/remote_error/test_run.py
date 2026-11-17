"""SPOT's cryptographic error in the remote application: its arithmetic against hand-computed cases, the parameters it
reads against their files, each term's Lean names against the tree, each basis against its parameters' kinds and the lock,
and what its output and a research run say."""
from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path

import pytest
import run as R

P = R.params()


def test_a_term_pools_over_plays_and_calls() -> None:
    t = R.Term("x", "", (), coeffs=(Fraction(1, 8), Fraction(1, 16), Fraction(1, 32)))
    assert t.at(3, 4) == Fraction(3, 8) + Fraction(4, 16) + Fraction(16, 32)
    assert R.Term("y", "", (), left_out="open").at(3, 4) is None


def test_figures_by_hand_on_a_one_gpu_fleet() -> None:
    t = R.Term("x", "", (), coeffs=(Fraction(1, 2 ** 10), Fraction(1, 2 ** 40), Fraction(0)))
    fl = {"world": {"id": "w", "machines": 1, "gpus": 1, "window_s": 1}}
    f = R.figures([t, R.Term("y", "", (), left_out="open")], fl, Fraction(2 ** 33))["world"]["periods"]
    # a window: one play at 2^33 calls, 2^-10 + 2^33·2^-40 = 9·2^-10
    assert f["window"]["plays"] == 1 and f["window"]["sha512_calls_log2"] == 33
    assert f["window"]["total_log2"] == round(R.log2(Fraction(9, 2 ** 10)), 2) == f["window"]["share_log2"]
    week = dict(R.MONITORING_PERIODS)["week"]
    assert f["week"]["plays"] == week and f["week"]["terms_log2"] == {"x": f["week"]["total_log2"]}
    assert f["week"]["total_log2"] == round(R.log2(week * Fraction(9, 2 ** 10)), 2)


def test_the_parameters_the_application_determines_are_read_from_their_files() -> None:
    s = R.schedule()
    assert (s["m_max"], s["queries"], s["levels"], s["k0"], s["log_len"]) == (35, 277, 7, 6, 23)
    assert (s["k_pad"], s["inner_log_inv_rate"], s["rounds_log"]) == (192, 3, 9)
    assert R.composer() == (18, 2 ** 8) and R.design() == (1, 3481648, 1)
    # PadLayout.len at m = kLog = 35, k₀ = 6: 204 + 2·29 + 2·29 + 4·6 = 344; nextPow2(344 + 192) · 2^3
    assert R.inner_n(35, 35, 6, 192, 3) == 1024 * 8
    want = {"positions": 2 ** 17, "N": 2 ** 13, "levels": 7, "log_len": 23, "queries": 277, "rounds": 1025,
            "max_leaves": 2 ** 49, "calls_mul": 2, "table_bound": Fraction(1, 2 ** 205)}
    assert {k: P[k].value for k in want} == want
    assert all(p.kind in R.KINDS and p.source for p in P.values())
    assert P["positions"].kind == "a, upper bound" and P["hash_rate"].kind == "c"
    assert {k for k, p in P.items() if p.kind == "b"} == {"M", "Rw", "k", "rho"}


def test_a_count_of_proof_blocks_tightens_the_commit_strings_read() -> None:
    assert R.params(2 * 3481648)["positions"].value == 2 * 2 ** 8
    assert R.params(2 ** 40)["positions"].value == 2 ** 17


def test_k_is_where_the_total_is_least_and_rho_the_least_it_admits() -> None:
    k, rho = P["k"].value, P["rho"].value
    assert rho == Fraction(2 ** 23) / (R.E_LOW * k) < 1
    best = R.headline_total(P)
    for other in (int(k * Fraction(65, 64)), int(k * Fraction(63, 64)), 2 ** 23, 2 ** 25):
        assert R.headline_total(R.with_k(P, other)) >= best


def test_the_closed_terms_by_hand() -> None:
    by = {t.name: t for t in R.terms(P)}
    assert by["hiding"].coeffs == (Fraction(1, 2 ** 207), 0, 0)
    assert by["file_finders"].coeffs == (Fraction(1, 2 ** 510), 0, 0)
    assert by["read_finder"].coeffs == (0, Fraction(1, 2 ** 255), 0)
    assert by["forgery"].coeffs == (0, Fraction(1, 2 ** 255), 0)
    k, rho = P["k"].value, P["rho"].value
    assert by["binding"].coeffs[1] == 2 ** 17 / (1 - rho) * 2 * (1 + k) / 2 ** 256
    assert by["binding_w"].coeffs == (2 ** 17 / (1 - rho) * 2 * 2 ** 40 * (1 + k) / 2 ** 256, 0, 0)
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
    missing = [f"{t.name}: {n} in {f}" for t in R.terms(P) for n, f in t.lean if not _declared(n, R.ROOT / f)]
    assert not missing, "a term names a Lean declaration its file doesn't declare (fix the name or the file):\n" + "\n".join(missing)
    assert _declared(R.GUARANTEE, R.ROOT / "security/Proofs/Properties/Verity/Sampled/OpensAt.lean")


def test_a_basis_follows_from_its_parameters_kinds() -> None:
    by = {t.name: R.basis(t, P) for t in R.terms(P)}
    assert set(by.values()) <= set(R.BASES) and "proved at these parameters" not in by.values()
    for name in ("hiding", "read_finder", "file_finders", "binding", "table_collisions", "two_continuation"):
        assert by[name] == "formula proved, value evaluated here", name
    for name in ("forgery", "table_iop", "roots", "binding_w", "table_collisions_v", "two_continuation_h"):
        assert by[name] == "assumed", name
    assert (by["pouw"], by["policy"], by["sample_miss"]) == ("open", "open", "excluded")
    for t in R.terms(P):
        assert t.lean and set(t.params) <= set(P), t.name


def test_the_guarantee_and_its_assumptions_come_from_the_locks() -> None:
    g = R.guarantee()
    mine = g[R.GUARANTEE]
    assert {"ecr/sha-512", "cr/sha-512", "euf-cma/lms-sha512-n64"} <= set(mine["assumptions"])
    assert "Verity.Vocab.PouwSound" in mine["open"] and "Verity.Vocab.SpotSound" in g[R.COMPOSITE]["open"]
    assert len(mine["open"]) == 4 and "four open premises" in R.DESCRIPTION
    for t in R.terms(P):
        assert set(t.assumptions) <= set(mine["assumptions"]), t.name


def test_the_output_says_which_part_it_is_and_what_is_left_out(capsys: pytest.CaptureFixture[str]) -> None:
    assert R.main([]) == 0
    out = capsys.readouterr().out
    for phrase in ("bound's formula proved in Lean under four open premises", "value evaluated in Python",
                   "SPOT's part only", "pouw (Pearl+): open, not in the total", "Verity.Vocab.PouwSound",
                   "inference-only policy: open, not in the total", "Verity.Vocab.InferenceOnlySound",
                   "without the forgery term"):
        assert phrase in out, phrase
    head = R.report(P)["fleets"]["world"]["periods"]["month"]
    assert head["total_without_forgery_log2"] <= head["total_log2"]


def test_a_research_run_records_each_metric_with_the_headline_last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20261114-000000-0000")
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
    assert [m["value"] for m in ms if m["name"] == "soundness/remote-spot-crypto-error-log2"][-1] == head["total_log2"]
    assert res["validation"]["status"] == "passed" and fig["description"] == R.DESCRIPTION
