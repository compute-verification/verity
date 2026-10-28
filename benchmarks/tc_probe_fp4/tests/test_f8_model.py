"""The B200 FP8 probe's CPU side: ``f8_model``'s vectorized step held word for word to the scalar references (``GroupSum.step`` for one
format unscaled, ``k32.GroupSumCandidate.step_scaled`` for mixed formats and UE8M0 scales) on every family ``f8_sm100`` generates, and
a fake-silicon dry run of ``f8_sm100``'s capture and eval: the layout gate passes on a device that computes a candidate, the eval
names that candidate alone among the floors on the floor probe, and one flipped word is counted.  No kernel is built or run."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import f8_model as FM  # noqa: E402
import f8_sm100 as S  # noqa: E402
import k32  # noqa: E402
from verity.errors import InvalidArtifact  # noqa: E402
from verity.ml.circuits.silicon import models as M  # noqa: E402

PER_FAMILY = 48


def _sample(kind: str, fam: str, seed: int):
    kk, fa, fb = S.KINDS[kind]
    fn, _, _ = S.FAMILIES[fam] if fam != "bulk" else (S.fam_bulk, 1, False)
    t = fn(np.random.default_rng(seed), 2, fa, fb, kk == 1)
    c, a, b, sa, sb, _ = S.expand(t, 0, 2)
    pick = np.random.default_rng(seed + 1).choice(c.size, PER_FAMILY, replace=False)
    a32, b32 = a[pick, :32], b[pick, :32]
    return kk, fa, fb, c[pick], a32, b32, sa[pick, 0], sb[pick, 0]


def _families(kind: str) -> list[str]:
    scaled = S.KINDS[kind][0] == 1
    return [f for f, (_, _, needs) in S.FAMILIES.items() if scaled or not needs] + ["bulk"]


@pytest.mark.parametrize("kind", list(S.KINDS))
def test_step_matches_scalar_reference(kind):
    for fi, fam in enumerate(_families(kind)):
        kk, fa, fb, c, a, b, sa, sb = _sample(kind, fam, 1000 + fi)
        for cand in list(FM.candidates(fa, fb, kk == 1).values())[::3]:
            words, dom = cand.step(c, a, b, sa if kk else None, sb if kk else None)
            ref = k32.GroupSumCandidate(cand.name, fa, fb, cand.groups, cand.width, cand.zero_exponent, scale="ue8m0" if kk else None)
            same = fa == fb and not kk
            gs = M.GroupSum(cand.name, "test", fa, 8, cand.groups, cand.width, cand.zero_exponent, "test") if same else None
            for i in range(len(c)):
                if not dom[i]:
                    continue
                if gs is not None:
                    want = gs.step(int(c[i]), [int(x) for x in a[i]], [int(x) for x in b[i]])
                else:
                    try:
                        want = ref.step_scaled(int(c[i]), a[i], b[i], (int(sa[i]),), (int(sb[i]),))
                    except InvalidArtifact:
                        assert (int(words[i]) & 0x7FFFFFFF) == 0x7F800000, (kind, fam, cand.name, i)
                        continue
                assert int(words[i]) == want, (kind, fam, cand.name, i, f"{int(words[i]):08X}", f"{want:08X}")


class FakeLib:
    """A device that computes ``truth`` (a candidate name; sm_120's step where a kind has no such candidate) on every kind."""

    def __init__(self, truth: str, misrouted: frozenset[str] = frozenset()):
        self.truth, self.misrouted = truth, misrouted

    def identity(self) -> dict:
        return {"rc": 0, "cc": "10.0", "name": "fake"}

    def run(self, kind: str, t) -> np.ndarray:
        kk, fa, fb = S.KINDS[kind]
        n = t["A"].shape[0]
        c, a, b, sa, sb, _ = S.expand(t, 0, n)
        cands = FM.candidates(fa, fb, kk == 1)
        cand = cands.get(self.truth, cands["sm120_gs32_w26_f133"])
        w, _ = FM.chain(cand, c, a, b, sa if kk else None, sb if kk else None, int(t["steps"]))
        D = w.reshape(n, S.M_T, S.N_T)
        return D[:, ::-1] if kind in self.misrouted else D


def test_dry_run_names_the_floor(tmp_path):
    scratch, out = tmp_path / "scratch", tmp_path / "out"
    S.gen(Namespace(scratch=str(scratch), seed=5, tiles=1, bulk_scale=0.0))
    keep = {("mx_e4m3", "scale_floor_probe"), ("mx_e5m2", "acc_walk"), ("f8_e4m3_e5m2", "cancellation")}
    plan = json.loads((scratch / "plan.json").read_text())
    plan["families"] = [p for p in plan["families"] if (p["kind"], p["family"]) in keep]
    (scratch / "plan.json").write_text(json.dumps(plan))
    assert S.capture(Namespace(scratch=str(scratch), out=str(out)), lib=FakeLib("gs32_w26_f140")) == 0
    doc = json.loads((out / "capture.json").read_text())
    assert doc["layout_ok"] and len(doc["captured"]) == 3
    D = scratch / "mx_e4m3__scale_floor_probe" / "D.npy"
    d = np.load(D)
    d.reshape(-1)[17] ^= 1
    np.save(D, d)
    assert S.evaluate(Namespace(scratch=str(scratch), out=str(out), procs=1, chunk_outputs=1 << 13, bulk_models="",
                                bulk_all_models=False)) == 0
    totals = json.loads((out / "summary.json").read_text())["totals"]
    floor = totals["mx_e4m3"]
    assert floor["gs32_w26_f140"]["mismatches"] == 1
    assert all(v["mismatches"] > 1 for k, v in floor.items() if k.startswith("gs32_w26_f") and k != "gs32_w26_f140")
    assert totals["mx_e5m2"]["sm120_gs32_w26_f133"]["mismatches"] == 0
    assert totals["f8_e4m3_e5m2"]["sm120_gs32_w26_f133"]["mismatches"] == 0
    assert totals["f8_e4m3_e5m2"]["gs32_w26_f149"]["mismatches"] == 0


def test_a_misrouted_kind_fails_its_gate_and_the_others_are_still_captured(tmp_path):
    scratch, out = tmp_path / "scratch", tmp_path / "out"
    S.gen(Namespace(scratch=str(scratch), seed=6, tiles=1, bulk_scale=0.0))
    plan = json.loads((scratch / "plan.json").read_text())
    plan["families"] = [p for p in plan["families"] if p["family"] == "acc_walk" and p["kind"] in ("f8_e4m3", "mx_e4m3")]
    (scratch / "plan.json").write_text(json.dumps(plan))
    assert S.capture(Namespace(scratch=str(scratch), out=str(out)), lib=FakeLib("sm120_gs32_w26_f133", frozenset({"mx_e4m3"}))) == 3
    doc = json.loads((out / "capture.json").read_text())
    assert doc["kind_layout_ok"]["mx_e4m3"] is False and doc["kind_layout_ok"]["f8_e4m3"] is True
    assert [p["kind"] for p in doc["captured"]] == ["f8_e4m3"]
    assert S.evaluate(Namespace(scratch=str(scratch), out=str(out), procs=1, chunk_outputs=1 << 13, bulk_models="",
                                bulk_all_models=False)) == 0
    totals = json.loads((out / "summary.json").read_text())["totals"]
    assert set(totals) == {"f8_e4m3"} and totals["f8_e4m3"]["sm120_gs32_w26_f133"]["mismatches"] == 0


B200_CAPTURE = HERE / "tests" / "fixtures" / "tc-sm100-f8-2026-10-19.json"


@pytest.mark.parametrize("kind", list(S.KINDS))
def test_b200_capture_records_are_sm120s_step_and_refute_every_other_candidate(kind):
    """The B200's captured words of every kind (``f8_sm100.py pin``): sm_120's QMMA step reproduces
    each, and every other candidate misses one, except the floors a plain kind can't see (any floor <= -126 is its words)."""
    kk, fa, fb = S.KINDS[kind]
    records = json.loads(B200_CAPTURE.read_text())["kinds"][kind]
    cands = FM.candidates(fa, fb, kk == 1)
    misses = dict.fromkeys(cands, 0)
    for r in records:
        for name, cand in cands.items():
            word, dom = S.replay(r, cand)
            assert dom, r
            misses[name] += word != int(r["d"], 16)
    unrefuted = {name for name, n in misses.items() if n == 0}
    assert unrefuted == ({"sm120_gs32_w26_f133"} if kk == 1 else {"sm120_gs32_w26_f133", "gs32_w26_f126", "gs32_w26_f149"})
