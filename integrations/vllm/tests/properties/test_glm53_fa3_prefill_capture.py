"""The GLM-5.3 FA3 prefill capture without a GPU: the launch check reads the tile, the flags, the combine instantiation and the kernel
counts from FA3's kernel names, the cases reach every sampled row and all three classes, a comparison job takes the twin's words as
the kernel's (unsplit, and split by `fa3_split_twins.split_call`) and finds a flipped one, a non-finite head takes the reference
evaluator's words, the verdict needs every class, and `main` end to end (the GPU run is
`tests/properties/glm53_fa3_prefill_capture_gpu.py --out ...` on an H200)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.properties import glm53_fa3_prefill_capture_gpu as X
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.reference import evaluate
from verity_vllm.program.kernels import fa2_relation, fa3_model as FM
from verity_vllm.program.kernels import fa3_split_twins as FT
from verity_vllm.program.registry import attention_vocab, hopper
from verity_vllm.program.registry import fa3_check_inf as F3
from verity_vllm.program.registry import fa3_split as FS
from verity_vllm.program.registry import sparse_mla_split as W
from verity_vllm.properties.record import REGISTRY, digest_of

FA3 = {"DOT": hopper.HopperBF16WgmmaDot16, "INV": hopper.Fa3InvSum}
PREPARE = "void flash::prepare_varlen_num_blocks_kernel<1, 1, true>(int, int, int, int const*, int const*)"
SPLIT = {"fwd": X.CALLS, "combine": X.CALLS}
UNSPLIT = {"fwd": X.CALLS, "combine": 0}


def _combine(tile=(8, 128), log=5, threads=256, align=1, even=False, varlen=True, wrapped=False) -> str:
    """A `FlashAttnFwdCombine` kernel name as the profiler reports it (GLM-SPLIT's H200 run, r20261013-001656-57d6: no
    `enable_sm90_or_later` wrapper; ``wrapped`` the forward kernel's spelling)."""
    t = (f"flash::FlashAttnFwdCombine<cute::tuple<cute::C<{tile[0]}>, cute::C<{tile[1]}> >, {log}, {threads}, {align}, "
         f"{str(even).lower()}, {str(varlen).lower()}, cutlass::bfloat16_t, float, cutlass::arch::Sm90>")
    inner = f"flash::enable_sm90_or_later<{t} >" if wrapped else t
    return f"void cutlass::device_kernel<{inner} >({t}::Params)"


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
    pinned = X.launch_geometry([PREPARE, _fwd(), _fwd()], 1, UNSPLIT)
    assert pinned["ok"] and pinned["instantiations"] == [{"stages": 2, "tile": [128, 80, 256], "headdim_v": 256,
                                                          "flags": dict(X.WANT_FLAGS, PackGQA=True, Split=False)}]
    split = X.launch_geometry([PREPARE, _fwd(Split=True), _combine()], 5, SPLIT)
    assert split["ok"] and split["combine"] == [X.WANT_COMBINE] and split["counts"] == SPLIT
    assert X.launch_geometry([PREPARE, _fwd(Split=True), _combine(wrapped=True)], 5, SPLIT)["ok"]
    assert X.launch_geometry([_fwd(Split=True), _combine()], 5)["ok"]                  # counts not read
    assert X.launch_geometry([_fwd(Split=True), _combine(), _combine(wrapped=True)], 32, SPLIT)["ok"]   # one instantiation, two spellings
    for names, static, counts in (
            ([_fwd()], 5, SPLIT), ([_fwd(Split=True)], 5, SPLIT), ([_fwd(Split=True), _combine()], 1, UNSPLIT),
            ([_fwd(), _combine()], 1, UNSPLIT), ([_fwd(tile=(128, 128, 128), dv=128)], 1, UNSPLIT),
            ([_fwd(IntraWGOverlap=False)], 1, UNSPLIT), ([_fwd(PagedKVNonTMA=True)], 1, UNSPLIT),
            ([_fwd(), _fwd(tile=(128, 64, 256))], 1, UNSPLIT), ([PREPARE], 1, UNSPLIT), ([], 1, UNSPLIT),
            ([_fwd(Split=True), _combine(log=4)], 5, SPLIT),                     # the <= 16 splits combine (kBlockM >= 16 only)
            ([_fwd(Split=True), _combine(tile=(16, 64))], 5, SPLIT),             # dv <= 64's
            ([_fwd(Split=True), _combine(even=True)], 5, SPLIT), ([_fwd(Split=True), _combine(varlen=False)], 5, SPLIT),
            ([_fwd(Split=True), _combine(), _combine(log=6)], 5, SPLIT),         # two instantiations
            ([_fwd(Split=True), "void flash::FlashAttnFwdCombine<?>(int)"], 5, SPLIT),            # a combine the check can't read
            ([_fwd(Split=True), _combine()], 33, SPLIT),                         # more splits than the combine's 32 lanes
            ([_fwd(Split=True), _combine()], 5, {"fwd": X.CALLS, "combine": 1}),
            ([_fwd(Split=True), _combine()], 5, {"fwd": 1, "combine": X.CALLS}),
            ([_fwd()], 1, {"fwd": X.CALLS, "combine": X.CALLS})):
        assert not X.launch_geometry(names, static, counts)["ok"], (names, static, counts)
    assert X.launch_geometry([_fwd(PackGQA=False)], 1, UNSPLIT)["ok"]   # recorded only: one query head a KV head packs nothing
    assert 1 << X.WANT_COMBINE["log_max_splits"] == W.LANES


def test_the_cases_reach_every_sampled_row_and_all_three_classes() -> None:
    cases = X.case_list()
    assert [c["name"] for c in X.case_list(quick=True)] == list(X.QUICK)
    longest = max(L for c in cases for L in c["seqlens"])
    assert all(r < longest for r in X.SAMPLE)
    plans = [X.plan(c["seqlens"], 0) for c in cases]
    seqs = [d for p in plans for d in p["dynamic"]]
    assert any(d > 1 for d in seqs) and any(d == 1 for d in seqs)
    assert all(p["static"] <= W.LANES for p in plans)
    assert X.plan([512], 0) == {"static": 4, "dynamic": [4]} and X.plan([512, 512, 64], 1) == {"static": 1, "dynamic": [1, 1, 1]}
    quick = [d for c in X.case_list(quick=True) for d in X.plan(c["seqlens"], 0)["dynamic"]]
    assert 1 in quick and max(quick) > 1


def _launch(tmp: Path, name: str, seqlens: list[int], seed: int, mode: str = "pinned", plant=None) -> list[dict]:
    """A launch's saved words with the twin as the kernel (`fa3_model.attention_call` for a sequence the plan runs as one split,
    `fa3_split_twins.split_call` for one it splits; ``plant(q, k, v, out)`` edits them first), and its jobs."""
    pl = X.plan(seqlens, 0 if mode == "served" else 1)
    rng = np.random.default_rng(seed)
    N = sum(seqlens)
    q, k, v = _bf16(rng, (N, X.NH, X.D)), _bf16(rng, (N, X.NH, X.D)), _bf16(rng, (N, X.NH, X.D))
    out, start = np.zeros_like(q), 0
    for L, ns in zip(seqlens, pl["dynamic"]):
        sl = slice(start, start + L)
        if ns > 1:
            out[sl] = FT.split_call(q[sl], k[sl], v[sl], np.arange(L), L, ns, fa2_relation.tables(), kblock_n=X.BN, kblock_m=X.BM)[0]
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


def _split_plan(monkeypatch, L: int, ns: int) -> None:
    """The served plan splits a sequence of L tokens into ``ns`` (the real plan first splits at 321 tokens, too slow for the reference
    evaluator here)."""
    real = X.plan
    monkeypatch.setattr(X, "plan", lambda seqlens, n: ({"static": ns, "dynamic": [ns if x == L else 1 for x in seqlens]}
                                                       if n == 0 and L in seqlens else real(seqlens, n)))


def test_a_job_takes_the_twins_words_and_finds_a_flipped_one(tmp_path) -> None:
    jobs = _launch(tmp_path, "w", [83, 3], 1)
    clean = [X.compare(j) for j in jobs]
    assert [(r["seq"], r["span"]) for r in clean] == [(0, [0, 83]), (1, [0, 3])] and not any("error" in r for r in clean)
    t = {k: sum(r[k] for r in clean) for k in X.TOTALS}
    assert t == {"rows": 86, "heads": 86 * X.NH, "mismatch": 0, "nonfinite": 0, "lg2_open": 0, "reference": 9, "twin_mismatch": 0,
                 "binding_mismatch": 0, "sampled_rows": 9, "ns_sampled": 0, "ns_unique": 0}
    out = np.load(jobs[1]["paths"]["out"])
    want = int(out[83 + 1, 5, 17])
    out[83 + 1, 5, 17] ^= 1
    np.save(jobs[1]["paths"]["out"], out)
    bad = X.compare(jobs[1])
    assert bad["mismatch"] == 1 and bad["first_mismatch"] == {"row": 1, "head": 5, "T": 2, "col": 17, "want": want, "kernel": want ^ 1}


def test_a_split_job_takes_the_split_twins_words_and_finds_a_flipped_one(tmp_path, monkeypatch) -> None:
    """A served sequence of 100 tokens the plan splits in 2 (n_block_max 2: rows below 80 read one split, the rest two): every row
    against `split_call`, the sampled ones against `head_split_rows`, `AttentionHeadSplit_v1` on one head and `AttentionSplit_v1` as
    the binding, with the split count diagnosed; then one flipped word."""
    _split_plan(monkeypatch, 100, 2)
    (job,) = _launch(tmp_path, "s", [100], 5, "served")
    assert (job["split"], job["static"]) == (2, 2)
    clean = X.compare(job)
    assert "error" not in clean, clean.get("error")
    assert {k: clean[k] for k in X.TOTALS} == {"rows": 100, "heads": 100 * X.NH, "mismatch": 0, "nonfinite": 0, "lg2_open": 0,
                                               "reference": 6, "twin_mismatch": 0, "binding_mismatch": 0, "sampled_rows": 6,
                                               "ns_sampled": 6, "ns_unique": clean["ns_unique"]}
    assert [m["row"] for m in clean["ns_matches"]] == [0, 1, 79, 80, 81, 99] and all(2 in m["matches"] for m in clean["ns_matches"])
    assert clean["ns_matches"][-1]["matches"] == [2] and clean["ns_unique"] >= 1     # row 99: some head's words are the split's only
    assert X.klass(clean) == "served_split"
    out = np.load(job["paths"]["out"])
    want = int(out[90, 3, 200])
    out[90, 3, 200] ^= 0x40
    np.save(job["paths"]["out"], out)
    bad = X.compare(job)
    assert bad["mismatch"] == 1 and bad["first_mismatch"] == {"row": 90, "head": 3, "T": 91, "col": 200, "want": want,
                                                              "kernel": want ^ 0x40}


def test_a_non_finite_head_takes_the_reference_evaluators_words(tmp_path) -> None:
    """q = +inf in one head of row 40: the twin flags it, and the kernel's words there are `AttentionHead_v4`'s."""
    r, h = 40, 2

    def plant(q, k, v, out):
        q[r, h, 0] = 0x7F80
        T = r + 1
        head = bind(F3.AttentionHeadV4, T=T, D=X.D, BN=X.BN, MASKED_FROM=attention_vocab.fa3_masked_from(T, X.BN, r, X.BM, 1), **FA3)
        out[r, h] = evaluate(head, q[r, h].tolist(), k[:T, h].reshape(-1).tolist(), v[:T, h].reshape(-1).tolist())

    (res,) = [X.compare(j) for j in _launch(tmp_path, "inf", [90], 2, plant=plant)]
    assert res["nonfinite"] == 1 and res["mismatch"] == 0 and res["twin_mismatch"] == 0 and res["reference"] == 1 + res["sampled_rows"]


def test_a_non_finite_split_head_takes_the_reference_evaluators_words(tmp_path, monkeypatch) -> None:
    """k = +inf in key 85, head 1, of a sequence split in 2: every row from 85 on is flagged in head 1, and the kernel's words there
    are `AttentionHeadSplit_v1`'s."""
    _split_plan(monkeypatch, 90, 2)
    h, key = 1, 85

    def plant(q, k, v, out):
        k[key, h, 7] = 0x7F80
        nbm = FM.kernel_geometry(90, 90, np.arange(90), X.BN, kblock_m=X.BM, qh_per_kvh=1)[1]
        for r in range(key, 90):
            T = r + 1
            head = bind(FS.AttentionHeadSplit, T=T, NBM=int(nbm[r]), NS=2, D=X.D, BN=X.BN,
                        MASKED_FROM=attention_vocab.fa3_masked_from(T, X.BN, r, X.BM, 1), **FA3)
            out[r, h] = evaluate(head, q[r, h].tolist(), k[:T, h].reshape(-1).tolist(), v[:T, h].reshape(-1).tolist())

    (res,) = [X.compare(j) for j in _launch(tmp_path, "inf", [90], 2, "served", plant=plant)]
    assert "error" not in res, res.get("error")
    assert res["nonfinite"] == 90 - key and res["mismatch"] == 0 and res["twin_mismatch"] == 0 and res["lg2_open"] == 0


def _rec(mode: str, split: int, **over) -> dict:
    return {"case": "c", "mode": mode, "seq": 0, "span": [0, 1], "split": split, "first_mismatch": None,
            **{k: 1 for k in X.TOTALS}, "mismatch": 0, "lg2_open": 0, "twin_mismatch": 0, "binding_mismatch": 0} | over


def _doc(comparisons) -> dict:
    return {"gate": {"ok": True}, "errors": {}, "selection": {"ok": True}, "mufu": {"ok": True},
            "launches": [{"ok": True, "deterministic": True}], "classes": X.summary(comparisons)}


def test_the_verdict_needs_every_class() -> None:
    good = [_rec("pinned", 1), _rec("served", 1), _rec("served", 4)]
    assert X._verdict(_doc(good)) and X.summary(good)["served_split"]["rows"] == 1
    for bad in ([_rec("pinned", 1, mismatch=1), _rec("served", 1), _rec("served", 4)],
                [_rec("pinned", 1), _rec("served", 1, binding_mismatch=1), _rec("served", 4)],
                [_rec("pinned", 1), _rec("served", 1), _rec("served", 4, mismatch=1)],
                [_rec("pinned", 1), _rec("served", 1), _rec("served", 4, lg2_open=1)],
                [_rec("pinned", 1), _rec("served", 1), _rec("served", 4, twin_mismatch=1)],
                [_rec("pinned", 1), _rec("served", 1), _rec("served", 4, reference=0)],
                [_rec("pinned", 1, reference=0), _rec("served", 1), _rec("served", 4)],
                [_rec("pinned", 1), _rec("served", 3)],                  # no served sequence ran as one split
                [_rec("pinned", 1), _rec("served", 1)]):                 # none the plan splits
        assert not X._verdict(_doc(bad)), bad
    for key, bad in (("mufu", {"ok": False}), ("selection", {"ok": False}), ("launches", [{"ok": False, "deterministic": True}]),
                     ("launches", [{"ok": True, "deterministic": False}]), ("errors", {"compare/x": "boom"})):
        assert not X._verdict(_doc(good) | {key: bad}), key


@pytest.mark.slow
@pytest.mark.parametrize("where,code,line", [(None, 0, "GLM53-FA3-PREFILL GO"), ("served_split", 1, "GLM53-FA3-PREFILL NO-GO")])
def test_main_end_to_end_on_stubbed_launches(tmp_path, monkeypatch, capsys, where, code, line) -> None:
    """`main` with the GPU stood in for (the twin's words as the kernel's, a pinned and a served launch of [83, 3], and a served
    launch of [100] that the stubbed plan splits in 2; one word flipped ``where``): the jobs through the spawned worker pool, the
    classes, the sealed record, the GO / NO-GO line and the exit status.  Slow (about 8 s): the spawned workers import the
    evaluator, and the sampled rows run the reference evaluator at up to 100 keys, split and unsplit."""
    monkeypatch.setattr(X.C, "gate", lambda *a: {"ok": True})
    monkeypatch.setattr(X.G, "_environment", lambda: {})
    _split_plan(monkeypatch, 100, 2)

    def gpu(work, outdir, quick, only, doc):
        doc["selection"], doc["mufu"] = {"ok": True}, {"ok": True}
        doc["launches"] += [{"case": n, "ok": True, "deterministic": True, "plan": {"static": s}}
                            for n, s in (("a", 1), ("b", 1), ("c", 2))]
        flip = (lambda q, k, v, out: out.__setitem__((7, 0, 0), out[7, 0, 0] ^ 1))
        return (_launch(work, "a", [83, 3], 3, "pinned") + _launch(work, "b", [83, 3], 3, "served")
                + _launch(work, "c", [100], 4, "served", plant=flip if where == "served_split" else None))

    monkeypatch.setattr(X, "gpu", gpu)
    out = tmp_path / "out"
    assert X.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1"]) == code
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["digest"] == digest_of(doc) and doc["ok"] is (code == 0) and not doc["errors"]
    cl = doc["classes"]
    assert (cl["pinned"]["rows"], cl["served"]["rows"], cl["served_split"]["rows"]) == (86, 86, 100)
    assert sum(c["mismatch"] for c in cl.values()) == int(where is not None) and cl["served_split"]["ns_sampled"] == 6
    printed = capsys.readouterr().out
    assert line in printed and "split_launches_ok=1/1" in printed and 'served_split={"rows": 100' in printed


def test_the_record_is_a_registered_property() -> None:
    assert REGISTRY["glm53_fa3_prefill_capture"].schema == X.SCHEMA
