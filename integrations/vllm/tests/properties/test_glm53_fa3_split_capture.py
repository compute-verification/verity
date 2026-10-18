"""The GLM-5.3 FA3 split capture without a GPU: the launches split as the kernel derives it, the operands, each comparison job
takes the kernel's words and finds a flipped one (the attention job names the split count the words are), the reference sample,
the verdict and `main` end to end (the GPU run is `tests/properties/glm53_fa3_split_capture_gpu.py --out ...` on an H200)."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from tests.properties import glm53_fa3_split_capture_gpu as X
from verity_vllm.program.kernels import derived_rows as DR
from verity_vllm.program.kernels import sparse_mla_split_twins as TW
from verity_vllm.program.registry import sparse_mla as M
from verity_vllm.program.registry import sparse_mla_split as W
from verity_vllm.properties.record import REGISTRY, digest_of, seal

U32 = np.uint32


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


SPLITS = {
    "dsa14": [1, 1, 1, 1, 1, 1, 1, 2, 2, 8, 16, 16, 16, 16],     # 181 blocks: 2 per SM
    "probe5": [3, 3, 16, 28, 28],                                 # 86 blocks: 1 per SM
    "ladder32": [_ceil(k, 5) for k in range(1, 33)],              # 528 blocks: 5 per SM
    "wide8": [11] * 8,                                            # 256 blocks: 3 per SM, the last split 2 blocks
    "wide24": [5] * 24,                                           # 768 blocks: 7 per SM
    "edge992": [1] * 991 + [4],                                   # 1023 blocks: 9 per SM
    "edge993": [1] * 993,                                         # past MAX_SPLIT_BATCH: one split each
    "pin1": [1] * 14, "bound28": [1, 1, 1, 1, 1, 1, 1, 2, 2, 8, 16, 16, 16, 16], "bound5": [3, 3, 5, 5, 5],
    "solo1": [1], "solo64": [1], "solo65": [2], "solo320": [5], "solo1000": [16], "solo1793": [28], "solo2048": [28],
}


@pytest.mark.parametrize("name", sorted(SPLITS))
def test_the_launches_split_as_the_kernel_derives(name: str) -> None:
    s, arg = X.LAUNCHES[name]
    assert W.launch_num_splits(X.SETS[s], 132, X.TOPK, arg) == SPLITS[name]


def test_the_launches_reach_uneven_and_empty_splits() -> None:
    """A 2048-key token in 11 splits ends with a split of 2 blocks; 29 blocks in 28 splits leave 13 of them empty."""
    assert W.split_blocks(2048, 11)[-1] == (30, 32) and len(W.split_blocks(2048, 11)) == 11
    assert len(W.split_blocks(1793, 28)) == 15
    counts = {ns for name, (s, arg) in X.LAUNCHES.items() for ns in W.launch_num_splits(X.SETS[s], 132, X.TOPK, arg)}
    assert {1, 2, 3, 4, 5, 7, 8, 10, 11, 16, 24, 28} <= counts


def test_the_plan_is_consistent() -> None:
    for a, b in X.SAME.items():
        assert X.LAUNCHES[a][0] == X.LAUNCHES[b][0]
        sa, aa = X.LAUNCHES[a]
        assert W.launch_num_splits(X.SETS[sa], 132, X.TOPK, aa) == W.launch_num_splits(X.SETS[sa], 132, X.TOPK, X.LAUNCHES[b][1])
    for name, i in X.REFERENCE:
        assert 0 <= i < len(X.SETS[X.LAUNCHES[name][0]]) and i in X.SAMPLE.get(X.LAUNCHES[name][0], range(i + 1))
    for s, tokens in X.SAMPLE.items():
        assert all(0 <= i < len(X.SETS[s]) for i in tokens) and len(X.SETS[s]) - 1 in tokens
    assert set(X.QUICK_LAUNCHES) <= set(X.LAUNCHES) and any(n in X.QUICK_LAUNCHES for n, _ in X.REFERENCE)
    assert max(max(v) for v in X.SETS.values()) <= X.TOPK and set(range(126, 139)) <= set(X.LG2_EXPONENTS)


def test_the_statics_are_glm53s() -> None:
    G = M.SPARSE_MLA_GLM53
    assert (X.NH, X.DR, X.DV, X.SCALE) == (G["NH"], G["DR"], G["DV"], G["SCALE"]) and X.SCALE == 0.0625


def test_the_operands() -> None:
    a, b = X.operands(["dsa14", "solo1"]), X.operands(["pin1"])
    assert set(a["sets"]) == {"dsa14", "solo1"} and set(b["sets"]) == {"dsa14"}
    assert np.array_equal(a["pool"], b["pool"]) and all(np.array_equal(a["sets"]["dsa14"][k], b["sets"]["dsa14"][k]) for k in b["sets"]["dsa14"])
    s = a["sets"]["dsa14"]
    assert s["qn"].shape == (14, X.NH, X.DV) and s["qr"].shape == (14, X.NH, X.DR) and s["rows"].shape == (14, X.TOPK)
    for t, row in zip(s["valid"], s["rows"]):
        assert (row[t:] == -1).all() and len(set(row[:t].tolist())) == t and row[:t].min() >= 0 and row[:t].max() < X.POOL
    v = (s["qn"].astype(np.uint32) << 16).view(np.float32)
    assert 2.5 < v[1].std() / v[0].std() < 3.5


def _attn(tmp_path, name: str, out_of=None):
    """``name``'s launch record and jobs, the kernel's words the twin's at each token's split count (``out_of(i, words)`` edits
    them), for the tokens of `SAMPLE` only."""
    o = X.operands([name])
    s = o["sets"][X.LAUNCHES[name][0]]
    ns = W.launch_num_splits(s["valid"], 132, X.TOPK, X.LAUNCHES[name][1])
    out = np.zeros((len(ns), X.NH, X.DV), np.uint16)
    tokens = X.SAMPLE.get(X.LAUNCHES[name][0], range(len(ns)))
    for i in tokens:
        kv = o["pool"][s["rows"][i, :s["valid"][i]]]
        out[i] = TW.head_split_rows(s["qn"][i], s["qr"][i], kv, T=int(s["valid"][i]), NS=ns[i], scale=X.SCALE)[0]
        if out_of is not None:
            out[i] = out_of(i, out[i], kv)
    pool = X._save(tmp_path / "pool", pool=o["pool"])
    return X.attn_jobs(tmp_path, name, s, out, 132, pool, {"deterministic": True, "shape": True})


def test_the_attn_jobs_take_the_kernel_words(tmp_path, monkeypatch) -> None:
    """Token 8 (T = 130, two splits) with one word flipped is one mismatch that no split count explains; token 7 (T = 129, two
    splits) given its words at three splits differs, and the diagnosis names 3."""
    monkeypatch.setattr(X, "SAMPLE", {"dsa14": (7, 8)})
    s = X.operands(["dsa14"])["sets"]["dsa14"]

    def edit(i, words, kv):
        if i == 8:
            words = words.copy()
            words[3, 100] ^= 1
        if i == 7:
            words = TW.head_split_rows(s["qn"][i], s["qr"][i], kv, T=129, NS=3, scale=X.SCALE)[0]
        return words

    rec, jobs = _attn(tmp_path, "dsa14", edit)
    assert rec["num_splits"] == SPLITS["dsa14"] and rec["static_bound"] == 28 and rec["splits"] == {"1": 7, "2": 2, "8": 1, "16": 4}
    res = {j["token"]: X.compare(j) for j in jobs}
    assert all("error" not in r for r in res.values()), res
    assert res[8]["words"] == X.NH * X.DV and res[8]["mismatch"] == 1 and res[8]["first_mismatch"]["head"] == 3 and res[8]["first_mismatch"]["word"] == 100
    assert res[8]["first_mismatch"]["matching_ns"] == []
    assert res[7]["mismatch"] > 0 and res[7]["differing"][0]["NS"] == 2 and res[7]["differing"][0]["matching_ns"] == [3]


def test_the_reference_sample_agrees_with_the_twin(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(X, "SAMPLE", {"dsa14": (8,)})
    monkeypatch.setattr(X, "REFERENCE", (("dsa14", 8),))
    rec, jobs = _attn(tmp_path, "dsa14")
    assert [j["reference"] for j in jobs] == [True]
    r = X.compare(jobs[0])
    assert (r["mismatch"], r["reference"], r["twin_mismatch"]) == (0, 1, 0)


def test_the_lg2_jobs_take_the_kernel_words_and_skip_the_open_ones(tmp_path, monkeypatch) -> None:
    """The start of exponent 126's band: the open word 0.5 (here the exact -1) is skipped, a flipped word is one mismatch."""
    assert X.LG2_JOB <= 1 << 23 and (1 << 23) % X.LG2_JOB == 0
    monkeypatch.setattr(X, "LG2_JOB", 1 << 14)
    x = (U32(126) << U32(23)) | np.arange(1 << 15, dtype=U32)
    y = TW.mufu_lg2_words(x)
    y[0] = 0xBF800000
    y[12345] ^= 2
    jobs = X.lg2_jobs(tmp_path, "E126", y, e=126)
    assert [j["range"] for j in jobs] == [[0, 1 << 14], [1 << 14, 1 << 15]]
    r = X.compare(jobs[0])
    assert (r["words"], r["mismatch"], r["reference"], r["twin_mismatch"]) == (X.LG2_JOB - 1, 1, 2, 0)
    assert r["first_mismatch"] == {"x": int(x[12345]), "want": int(y[12345] ^ 2), "kernel": int(y[12345])}
    sweep = X.sweep_words()[:4096]
    r = X.compare(X.lg2_jobs(tmp_path, "sweep", TW.mufu_lg2_words(sweep), x=sweep)[0])
    assert r["words"] == 4096 - int(TW.undetermined_words(sweep).sum()) and r["mismatch"] == 0


@pytest.mark.parametrize("op,twin", [("ex2", DR.mufu_ex2_ftz), ("rcp", DR.mufu_rcp_ftz)])
def test_the_mufu_jobs_take_the_kernel_words(tmp_path, op: str, twin) -> None:
    x = X.sweep_words()[:4096]
    y = twin(x).astype(U32)
    y[7] ^= 1
    r = X.compare({"phase": "mufu", "case": op, "kind": "mufu", "paths": X._save(tmp_path / op, x=x, y=y)})
    assert (r["words"], r["mismatch"], r["first_mismatch"]["x"]) == (4096, 1, int(x[7]))


def _doc(quick: bool = False) -> dict:
    names = X.QUICK_LAUNCHES if quick else tuple(X.LAUNCHES)
    launches = [{"phase": "mufu", "case": c, "checks": {"ok": True}} for c in ("lg2", "ex2", "rcp")]
    launches += [{"phase": "attn", "case": n, "checks": {"deterministic": True, "shape": True}} for n in names]
    comps = [{"phase": x["phase"], "case": x["case"], "words": 10, "mismatch": 0, "unresolved": 0, "reference": 1, "twin_mismatch": 0,
              "first_mismatch": None, "differing": []} for x in launches]
    doc = {"schema": X.SCHEMA, "quick": quick, "gate": {"ok": True}, "errors": {}, "bind": {"ok": True}, "launches": launches,
           "comparisons": comps}
    doc["totals"] = {k: sum(c[k] for c in comps) for k in X.TOTALS}
    return doc


def _set(doc: dict, path: tuple, value) -> dict:
    doc = copy.deepcopy(doc)
    d = doc
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value
    if path[0] == "comparisons":
        doc["totals"] = {k: sum(c[k] for c in doc["comparisons"]) for k in X.TOTALS}
    return doc


def test_the_verdict_holds_on_a_passing_doc() -> None:
    assert X._verdict(_doc()) and X._verdict(_doc(quick=True))
    assert all(c["ok"] for c in X.cases(_doc()).values())


@pytest.mark.parametrize("path,value", [(("gate", "ok"), False), (("bind", "ok"), False), (("errors", "attn"), "boom"),
                                        (("errors", "compare/attn/dsa14"), "boom"), (("launches", 0, "checks", "ok"), False),
                                        (("launches", 5, "checks", "deterministic"), False), (("launches", 4, "checks"), {}),
                                        (("comparisons", 3, "mismatch"), 1), (("comparisons", 0, "unresolved"), 2),
                                        (("comparisons", 6, "twin_mismatch"), 1), (("comparisons", 2, "words"), 0)])
def test_the_verdict_needs_every_check(path: tuple, value) -> None:
    assert not X._verdict(_set(_doc(), path, value))


def test_the_verdict_needs_every_launch_and_a_reference() -> None:
    doc = _doc()
    doc["launches"] = [x for x in doc["launches"] if x["case"] != "edge993"]
    assert not X._verdict(doc)
    doc = _doc()
    for c in doc["comparisons"]:
        c["reference"] = 0
    doc["totals"]["reference"] = 0
    assert not X._verdict(doc)


def test_each_launch_has_its_own_verdict() -> None:
    doc = _set(_set(_doc(), ("comparisons", 4, "mismatch"), 3), ("launches", 1, "checks", "ok"), False)
    assert {k for k, c in X.cases(doc).items() if not c["ok"]} == {"mufu/ex2", f"attn/{doc['launches'][4]['case']}"}


def test_main_end_to_end_on_stubbed_launches(tmp_path, monkeypatch, capsys) -> None:
    """`main --quick` with the GPU's phases stood in for (the jobs of the tests above, the attention words the twin's with one
    flipped): the jobs through the spawned worker pool, the per-launch verdicts, the sealed record and the exit status."""
    monkeypatch.setattr(X.C, "gate", lambda *a: {"ok": True, "sms": 132})
    monkeypatch.setattr(X.G, "_environment", lambda: {})
    monkeypatch.setattr(X, "QUICK_LAUNCHES", ("dsa14",))
    monkeypatch.setattr(X, "SAMPLE", {"dsa14": (0, 8)})
    monkeypatch.setattr(X, "REFERENCE", (("dsa14", 8),))
    sweep = X.sweep_words()[:2048]

    def flip(i, words, kv):
        if i == 8:
            words = words.copy()
            words[0, 0] ^= 1
        return words

    def gpu(work, quick, doc):
        assert quick
        rec, jobs = _attn(work, "dsa14", flip)
        doc["launches"] += [{"phase": "mufu", "case": "lg2", "checks": {"sweep_input": True}}, rec]
        jobs += X.lg2_jobs(work, "sweep", TW.mufu_lg2_words(sweep), x=sweep)
        for op, twin in (("ex2", DR.mufu_ex2_ftz), ("rcp", DR.mufu_rcp_ftz)):
            doc["launches"].append({"phase": "mufu", "case": op, "checks": {"table": True}})
            jobs.append({"phase": "mufu", "case": op, "kind": "mufu", "paths": X._save(work / op, x=sweep, y=twin(sweep).astype(U32))})
        return jobs

    monkeypatch.setattr(X, "gpu", gpu)
    out = tmp_path / "out"
    assert X.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1", "--quick"]) == 1
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["digest"] == digest_of(doc) and doc["ok"] is False and not doc["errors"] and doc["bind"]["ok"]
    assert {c: v["ok"] for c, v in doc["cases"].items()} == {"mufu/lg2": True, "attn/dsa14": False, "mufu/ex2": True, "mufu/rcp": True}
    assert doc["totals"]["mismatch"] == 1 and doc["totals"]["reference"] == 1 + 2 and doc["totals"]["twin_mismatch"] == 0
    att = next(c for c in doc["comparisons"] if c["case"] == "dsa14")
    assert att["words"] == 2 * X.NH * X.DV and att["first_mismatch"]["token"] == 8 and att["first_mismatch"]["NS"] == 2
    printed = capsys.readouterr().out
    assert "GLM53-FA3-SPLIT NO-GO" in printed and "no-go cases=['attn/dsa14']" in printed


def test_the_record_is_a_registered_property() -> None:
    doc = seal(_doc(), X._verdict(_doc()))
    assert REGISTRY["glm53_fa3_split_capture"].schema == X.SCHEMA and doc["ok"] and doc["digest"] == digest_of(doc)


def test_the_programs_bind_at_glm53s_statics() -> None:
    got = X.bind_check()
    assert got["ok"] and "SparseMlaHeadSplit_v1{T=2048,NS=16,...}" in got["bindings"] and "MufuLg2Ftz_v1" in got["bindings"]
