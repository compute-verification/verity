"""The GLM-5.3 FA3 prefill capture without a GPU: the launch check reads the tile, the flags and the combine kernel from FA3's kernel
names, the cases reach every sampled row and both served classes, a comparison job takes the twin's words as the kernel's and finds a
flipped one (a split sequence's the split twin's, with `DsaDenseMha`'s split count checked on a one-sequence launch), a non-finite
head takes the reference evaluator's words, the verdict counts the split sequences where the plan splits and needs the backend's
launches to write the served words, and `main` end to end (the GPU run is `tests/properties/glm53_fa3_prefill_capture_gpu.py --out
...` on an H200)."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import json
from pathlib import Path

import numpy as np
import pytest

from tests.properties import glm53_fa3_prefill_capture_gpu as X
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.reference import evaluate
from verity_vllm.program.frontend.rules import vllm_dsa_dense
from verity.ml.definitions.numerics import fa2_relation, fa3_model as FM
from verity.ml.definitions.registry import attention_vocab, hopper
from verity.ml.definitions.registry import fa3_check_inf as F3
from verity_vllm.properties.record import REGISTRY, digest_of

COMBINE = ("void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdCombine<cute::tuple<cute::C<8>, cute::C<128> >, "
           "5, 256, 1, false, true, cutlass::bfloat16_t, float, cutlass::arch::Sm90> > >(flash::FlashAttnFwdCombine<...>::Params)")
PREPARE = "void flash::prepare_varlen_num_blocks_kernel<1, 1, true>(int, int, int, int const*, int const*)"


def _fwd(tile=(128, 80, 256), dv=256, **flags) -> str:
    f = dict(X.WANT_FLAGS, PackGQA=True, Split=False) | flags
    b = ", ".join("true" if f[k] else "false" for k in X.FLAGS)
    return ("void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, "
            f"cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<{tile[0]}>, cute::C<{tile[1]}>, cute::C<{tile[2]}> >, "
            f"{dv}, cutlass::bfloat16_t, float, cutlass::arch::Sm90, {b}, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<...> > > >"
            "(flash::FlashAttnFwdSm90<...>::Params)")


def _bf16(rng, shape):
    return (rng.standard_normal(shape).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def test_the_launch_check_reads_the_tile_the_flags_and_the_combine() -> None:
    pinned = X.launch_geometry([PREPARE, _fwd(), _fwd()], 1, 81)
    assert pinned["ok"] and pinned["instantiations"] == [{"stages": 2, "tile": [128, 80, 256], "headdim_v": 256,
                                                          "flags": dict(X.WANT_FLAGS, PackGQA=True, Split=False)}]
    assert X.launch_geometry([PREPARE, _fwd(Split=True), COMBINE], 5, 512)["ok"]
    assert X.launch_geometry([PREPARE, _fwd(Is_causal=False)], 1, 1)["ok"]   # mha_fwd drops causal at max_seqlen_q 1
    for names, static, sq in (([_fwd()], 5, 512), ([_fwd(Split=True)], 5, 512), ([_fwd(Split=True), COMBINE], 1, 81),
                              ([_fwd(), COMBINE], 1, 81), ([_fwd(tile=(128, 128, 128), dv=128)], 1, 81), ([_fwd(IntraWGOverlap=False)], 1, 81),
                              ([_fwd(PagedKVNonTMA=True)], 1, 81), ([_fwd(), _fwd(tile=(128, 64, 256))], 1, 81), ([PREPARE], 1, 81), ([], 1, 81),
                              ([_fwd()], 1, 1), ([_fwd(Is_causal=False)], 1, 2)):
        assert not X.launch_geometry(names, static, sq)["ok"], (names, static, sq)
    assert X.launch_geometry([_fwd(PackGQA=False)], 1, 81)["ok"]       # recorded only: one query head a KV head packs nothing


def test_the_cases_reach_every_sampled_row_and_both_served_classes() -> None:
    cases = X.case_list()
    assert [c["name"] for c in X.case_list(quick=True)] == list(X.QUICK)
    longest = max(L for c in cases for L in c["seqlens"])
    assert all(r < longest for r in X.SAMPLE)
    seqs = [(c["name"], s, d) for c in cases for s, d in enumerate(X.plan(c["seqlens"], 0)["dynamic"])]
    assert any(d > 1 for *_, d in seqs) and any(d == 1 for *_, d in seqs)
    assert X.plan([512], 0) == {"static": 4, "dynamic": [4]} and X.plan([512, 512, 64], 1) == {"static": 1, "dynamic": [1, 1, 1]}
    quick = [d for c in X.case_list(quick=True) for d in X.plan(c["seqlens"], 0)["dynamic"]]
    assert 1 in quick and max(quick) > 1


def _launch(tmp: Path, name: str, seqlens: list[int], seed: int, mode: str = "pinned", plant=None) -> list[dict]:
    """A launch's saved words with the C++ twin as the kernel, the split twin on a sequence the plan splits (``plant(q, out)``
    edits them first), and its jobs."""
    rng = np.random.default_rng(seed)
    N = sum(seqlens)
    q, k, v = _bf16(rng, (N, X.NH, X.D)), _bf16(rng, (N, X.NH, X.D)), _bf16(rng, (N, X.NH, X.D))
    out, start = np.zeros_like(q), 0
    pl = X.plan(seqlens, 0 if mode == "served" else 1)
    for L, ns in zip(seqlens, pl["dynamic"]):
        sl = slice(start, start + L)
        if ns > 1:
            out[sl] = X.split_twin(q[sl], k[sl], v[sl], np.arange(L), L, ns)[0]
        else:
            out[sl] = FM.attention_call(q[sl], k[sl], v[sl], np.arange(L), L, L, X.SCALE, fa2_relation.tables(), kblock_n=X.BN).out
        start += L
    if plant:
        plant(q, k, v, out)
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, a in (("q", q), ("k", k), ("v", v), ("out", out)):
        paths[key] = str(d / f"{key}.npy")
        np.save(paths[key], a)
    return X.jobs_for({"name": name, "seqlens": seqlens}, mode, pl, paths)


def _split_plan(splits: dict[tuple, list[int]]):
    """A stub of `plan` that splits the given served launches (seqlens -> each sequence's count)."""
    def plan(seqlens, ns):
        dyn = splits.get(tuple(seqlens)) if ns == 0 else None
        return {"static": max(dyn), "dynamic": dyn} if dyn else {"static": 1, "dynamic": [1] * len(seqlens)}
    return plan


@pytest.mark.slow
def test_a_split_job_takes_the_split_twins_words_and_checks_the_rules_split_count(tmp_path, monkeypatch) -> None:
    """A 100-row sequence the stubbed plan runs as 3 splits of one 80-key block each (rows 80..99 see two splits), alone and
    beside a 3-row one: its rows against the split twin, its sampled rows' heads against `AttentionHeadSplit_v1`, a flipped word
    found; on the one-sequence launch the rule's own count (1 at 100 rows) disagrees with the stub's, which the binding check
    counts, and agrees once the rule is stubbed too."""
    monkeypatch.setattr(X, "plan", _split_plan({(100,): [3], (100, 3): [3, 1]}))
    jobs = _launch(tmp_path, "s", [100, 3], 5, "served")
    clean = [X.compare(j) for j in jobs]
    assert [(r["seq"], r["split"]) for r in clean] == [(0, 3), (1, 1)] and not any("error" in r for r in clean)
    assert {k: clean[0][k] for k in X.TOTALS} == {"rows": 100, "heads": 800, "mismatch": 0, "nonfinite": 0, "reference": 6,
                                                   "twin_mismatch": 0, "binding_mismatch": 0, "sampled_rows": 6}
    out = np.load(jobs[0]["paths"]["out"])
    out[90, 3, 7] ^= 1
    np.save(jobs[0]["paths"]["out"], out)
    assert X.compare(jobs[0])["mismatch"] == 1
    (one,) = _launch(tmp_path, "one", [100], 6, "served")
    assert X.compare(one)["binding_mismatch"] == 6
    monkeypatch.setattr(vllm_dsa_dense, "_dense_splits", lambda tp, T, NH, D: 3)
    assert X.compare(one)["binding_mismatch"] == 0


def test_a_job_takes_the_twins_words_and_finds_a_flipped_one(tmp_path) -> None:
    jobs = _launch(tmp_path, "w", [83, 3], 1)
    clean = [X.compare(j) for j in jobs]
    assert [(r["seq"], r["span"]) for r in clean] == [(0, [0, 83]), (1, [0, 3])] and not any("error" in r for r in clean)
    t = {k: sum(r[k] for r in clean) for k in X.TOTALS}
    assert t == {"rows": 86, "heads": 86 * X.NH, "mismatch": 0, "nonfinite": 0, "reference": 9, "twin_mismatch": 0,
                 "binding_mismatch": 0, "sampled_rows": 9}
    out = np.load(jobs[1]["paths"]["out"])
    want = int(out[83 + 1, 5, 17])
    out[83 + 1, 5, 17] ^= 1
    np.save(jobs[1]["paths"]["out"], out)
    bad = X.compare(jobs[1])
    assert bad["mismatch"] == 1 and bad["first_mismatch"] == {"row": 1, "head": 5, "T": 2, "col": 17, "want": want, "kernel": want ^ 1}


def test_a_non_finite_head_takes_the_reference_evaluators_words(tmp_path) -> None:
    """q = +inf in one head of row 40: the twin flags it, and the kernel's words there are `AttentionHead_v4`'s."""
    fa3 = {"DOT": hopper.HopperBF16WgmmaDot16, "INV": hopper.Fa3InvSum}
    r, h = 40, 2

    def plant(q, k, v, out):
        q[r, h, 0] = 0x7F80
        T = r + 1
        head = bind(F3.AttentionHeadV4, T=T, D=X.D, BN=X.BN, MASKED_FROM=attention_vocab.fa3_masked_from(T, X.BN, r, X.BM, 1), **fa3)
        out[r, h] = evaluate(head, q[r, h].tolist(), k[:T, h].reshape(-1).tolist(), v[:T, h].reshape(-1).tolist())

    (res,) = [X.compare(j) for j in _launch(tmp_path, "inf", [90], 2, plant=plant)]
    assert res["nonfinite"] == 1 and res["mismatch"] == 0 and res["twin_mismatch"] == 0 and res["reference"] == 1 + res["sampled_rows"]


def _rec(mode: str, split: int, **over) -> dict:
    return {"case": "c", "mode": mode, "seq": 0, "span": [0, 1], "split": split, "first_mismatch": None,
            **{k: 1 for k in X.TOTALS}, "mismatch": 0, "twin_mismatch": 0, "binding_mismatch": 0} | over


BACKEND = {"mode": "backend", "ok": True, "deterministic": True, "same_as_served": True}


SERVED = {"mode": "served", "ok": True, "deterministic": True, "plan": {"static": 4, "dynamic": [4, 1]}}


def _doc(comparisons, served=SERVED) -> dict:
    return {"gate": {"ok": True}, "errors": {}, "selection": {"ok": True}, "mufu": {"ok": True},
            "launches": [served, BACKEND], "classes": X.summary(comparisons)}


def test_the_verdict_counts_the_split_sequences_where_the_plan_splits() -> None:
    good = [_rec("pinned", 1), _rec("served", 1), _rec("served", 4)]
    assert X._verdict(_doc(good))
    assert not X._verdict(_doc(good[:2] + [_rec("served", 4, mismatch=1)]))
    assert not X._verdict(_doc(good[:2] + [_rec("served", 4, binding_mismatch=1)]))
    assert not X._verdict(_doc(good[:2] + [_rec("served", 4, reference=0)]))
    assert not X._verdict(_doc(good[:2]))                                          # the plan splits, no split row compared
    unsplit = SERVED | {"plan": {"static": 1, "dynamic": [1]}}
    assert X._verdict(_doc(good[:2], unsplit)) and not X._verdict(_doc(good, unsplit))
    assert not X._verdict(_doc([_rec("pinned", 1, mismatch=1), _rec("served", 1), _rec("served", 4)]))
    assert not X._verdict(_doc([_rec("pinned", 1), _rec("served", 1, binding_mismatch=1), _rec("served", 4)]))
    assert not X._verdict(_doc([_rec("pinned", 1), _rec("served", 3)]))          # no served sequence ran as one split
    assert not X._verdict(_doc([_rec("pinned", 1, reference=0), _rec("served", 1), _rec("served", 4)]))
    for key, bad in (("mufu", {"ok": False}), ("selection", {"ok": False}), ("launches", [{"ok": False, "deterministic": True}, BACKEND]),
                     ("launches", [SERVED | {"deterministic": False}, BACKEND]), ("errors", {"compare/x": "boom"}),
                     ("launches", [SERVED]), ("launches", [BACKEND, BACKEND | {"same_as_served": False}])):
        assert not X._verdict(_doc(good) | {key: bad}), (key, bad)


@pytest.mark.slow
@pytest.mark.parametrize("where,code,line", [(None, 0, "GLM53-FA3-PREFILL GO"), ("served_split", 1, "GLM53-FA3-PREFILL NO-GO")])
def test_main_end_to_end_on_stubbed_launches(tmp_path, monkeypatch, capsys, where, code, line) -> None:
    """`main` with the GPU stood in for (the twins' words as the kernel's, a pinned and a served launch of [83, 3], and a served
    launch of [300, 3] whose 300-row sequence the stubbed plan splits; one word flipped ``where``): the jobs through the spawned
    worker pool, the classes, the sealed record, the GO / NO-GO line and the exit status.  Slow (about 25 s): the spawned
    workers import the evaluator, and the 300-row sequence's sampled rows run `AttentionHeadSplit_v1` at up to 300 keys.  A
    varlen launch, so the workers, which the stub does not reach, read no rule's split count."""
    monkeypatch.setattr(X.C, "gate", lambda *a: {"ok": True})
    monkeypatch.setattr(X.G, "_environment", lambda: {})
    plan = _split_plan({(300, 3): [3, 1]})

    def gpu(work, outdir, quick, only, doc):
        doc["selection"], doc["mufu"] = {"ok": True}, {"ok": True}
        doc["launches"] += [{"case": n, "mode": m, "ok": True, "deterministic": True, "plan": plan(s, int(m == "pinned"))}
                            for n, m, s in (("a", "pinned", [83, 3]), ("b", "served", [83, 3]), ("c", "served", [300, 3]))]
        doc["launches"].append(BACKEND | {"case": "b"})
        flip = (lambda q, k, v, out: out.__setitem__((7, 0, 0), out[7, 0, 0] ^ 1))
        return (_launch(work, "a", [83, 3], 3, "pinned") + _launch(work, "b", [83, 3], 3, "served")
                + _launch(work, "c", [300, 3], 4, "served", plant=flip if where else None))

    monkeypatch.setattr(X, "gpu", gpu)
    monkeypatch.setattr(X, "plan", plan)
    out = tmp_path / "out"
    assert X.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1"]) == code
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["digest"] == digest_of(doc) and doc["ok"] is (code == 0) and not doc["errors"]
    cl = doc["classes"]
    assert (cl["pinned"]["rows"], cl["served"]["rows"], cl["served_split"]["rows"]) == (86, 89, 300)
    assert cl["served_split"]["reference"] > 0 and sum(c["mismatch"] for c in cl.values()) == int(where is not None)
    printed = capsys.readouterr().out
    split = f"served-split (AttentionSplit_v1) sequences=1 rows=300 heads_differing={int(where is not None)} of 2400"
    assert line in printed and split in printed


def test_the_record_is_a_registered_property() -> None:
    assert REGISTRY["glm53_fa3_prefill_capture"].schema == X.SCHEMA
