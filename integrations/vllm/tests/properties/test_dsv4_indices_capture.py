"""The H200 capture of DeepSeek-V4-Pro's prefill index builders (`dsv4_indices_capture_gpu`) on the CPU: its chunks' positions and
words, the rows it reads back, its launch checks and comparison jobs on what the twin says the kernels write, its PTX check, its
verdicts and its main loop with the GPU stood in for."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from tests.properties import dsv4_indices_capture_gpu as X
from verity_vllm.program.kernels import dsv4_indices_twin as T
from verity_vllm.properties.record import REGISTRY, digest_of

#: the cases simulated here (ratio 128's widest rows are left to the H200)
SIM = ["c4", "c128-256"]


def simulate_c128a(case: str, pos: np.ndarray) -> dict:
    c = X.CASES_L[case]
    pf = np.full((pos.size + 2, c.buffer), X.SENTINEL, np.uint32)
    pf[:pos.size, :c.topk] = X.short_rows(pos, c.topk, c.ratio)
    run = {"pf": pf, "gd": np.full((1, c.buffer), X.SENTINEL, np.uint32), "dl": np.full(1, X.SENTINEL, np.uint32)}
    return {"runs": [run, copy.deepcopy(run)], "view": True, "pos": pos}


def simulate_combine(case: str, o: dict, topk: np.ndarray) -> dict:
    st, n = X.statics(case), o["pos"].size
    rows, lens, declined = T.combine(topk, X._w(o["pos"]), o["mb"], o["n"], o["gs"], **st)
    assert not declined.any()
    ws = np.full((n + X.WS_OFF + 2, st["WIDTH"]), X.SENTINEL, np.uint32)
    ws[X.WS_OFF:X.WS_OFF + n] = rows
    wl = np.full(n + X.LENS_OFF + 4, X.SENTINEL, np.uint32)
    wl[X.LENS_OFF:X.LENS_OFF + n] = lens
    run = {"ws": ws, "wl": wl}
    return {"runs": [run, copy.deepcopy(run)], "views": True, "alloc": {"rows": rows.copy(), "lens": lens.copy()}, "topk": topk}


def simulate(case: str, seeds=(7, 8)):
    """What `launches` would return were the kernels the twin and the restatement."""
    ops = {kind: X.requests(case, kind, seed) for kind, seed in zip(("served", "edge"), seeds)}
    c128a, topk = None, ops["served"]["topk"]
    if X.CASES_L[case].ratio == 128:
        c128a = simulate_c128a(case, X.c128a_positions(case, ops["served"]["pos"], seeds[0]))
        topk = c128a["runs"][0]["pf"][:ops["served"]["pos"].size, :X.CASES_L[case].topk]
    return ops, c128a, {"served": simulate_combine(case, ops["served"], topk),
                        "edge": simulate_combine(case, ops["edge"], ops["edge"]["topk"])}


@pytest.fixture(scope="module")
def observed(tmp_path_factory):
    """Each simulated case's operands, the twin's words as the GPU's, and `observe`'s records and quick jobs on them."""
    out = {}
    for case in SIM:
        ops, c128a, combine = simulate(case)
        work = tmp_path_factory.mktemp(case)
        recs, jobs = X.observe(work, case, ops, c128a, combine, quick=True)
        out[case] = SimpleNamespace(ops=ops, c128a=c128a, combine=combine, recs=recs, jobs=jobs)
    return out


def test_the_statics_are_dsv4s() -> None:
    """The window 128, rows padded to 128: ratio 4's 1024 in 1152, ratio 128's K in K + 128 at K 256 (a buffer of 8192), 896 (its
    own) and 8192; each K serves its context (K >= max_len // 128) and the registry's served statics are two of the cases."""
    assert (X.WINDOW, X.ALIGN) == (128, 128)
    assert {k: (c.topk, c.ratio, c.buffer, X.width(c.topk)) for k, c in X.CASES_L.items()} == {
        "c4": (1024, 4, 1024, 1152), "c128-256": (256, 128, 8192, 384), "c128-896": (896, 128, 896, 1024),
        "c128-8192": (8192, 128, 8192, 8320)}
    for k, c in X.CASES_L.items():
        if c.ratio == 128:
            need = max(c.max_len // c.ratio, 1)
            assert min(max(1 << (need - 1).bit_length(), X.ALIGN), c.buffer) == c.topk >= need, k
    assert X.statics("c4") == X.R.DSV4_INDICES["c4"] and X.statics("c128-8192") == X.R.DSV4_INDICES["c128"]
    assert set(X.PTX_COUNTS) == set(X.KERNELS) and len(X.SOURCES) == 3


@pytest.mark.parametrize("case", list(X.CASES_L))
def test_the_chunks_hold_the_positions_they_mean(case: str) -> None:
    """The served chunk: request 0 from position 0, request 1 ending at the case's longest context (ratio 4: past the top-k's
    saturation), gather lengths, N and M as the chunk plan has them.  The edge chunk: both int32 ends of the Definition's domain,
    negative positions, every K's saturation, M * b and N + swa_start wrapping, a gather longer than its sequence.  No token of
    either is one the twin declines, and each top-k row is TOP_K words."""
    c = X.CASES_L[case]
    s, e = X.requests(case, "served", 1), X.requests(case, "edge", 2)
    assert s["pos"].min() == 0 and s["pos"].max() < c.max_len and s["pos"].size == sum(q for _, q in X.SERVED[case])
    q = (s["pos"] + 1) // c.ratio
    assert (q.max() > c.topk) if c.ratio == 4 else (q.max() == c.topk)
    seq, ql = np.array(X.SERVED[case]).T
    assert s["gather"].tolist() == (ql + np.clip(seq - ql, 0, 127)).tolist()
    assert s["N"] == max(seq // c.ratio) and s["M"] == s["N"] + max(s["gather"])
    assert (s["topk"] is None) == (c.ratio == 128)
    assert e["pos"].max() == X.I32_MAX - 1 and e["pos"].min() == X.I32_MIN + X.WINDOW - 1 and (e["pos"] < 0).sum() > 50
    assert ((e["pos"] + 1) // c.ratio == c.topk).any() and ((e["pos"] + 1) // c.ratio > c.topk).any()
    assert (e["M"] * e["b"] > X.I32_MAX).any() and (T.s32(e["gs"]) < 0).any()
    start = np.maximum(e["pos"] - 127, 0)
    assert (e["N"] + start > X.I32_MAX).any()
    for o in (s, e):
        topk = o["topk"] if o["topk"] is not None else np.zeros((o["pos"].size, c.topk), np.uint32)
        assert topk.shape == (o["pos"].size, c.topk)
        assert not T.combine(topk, X._w(o["pos"]), o["mb"], o["n"], o["gs"], **X.statics(case))[2].any()
        assert np.array_equal(T.s32(o["pos"]), o["pos"])


def test_ratio_4s_selection_is_ids_below_the_count_and_junk_past_it() -> None:
    """The served ratio-4 top-k: slots below the token's count ids in [-1, count), the rest random words the kernel must not read."""
    o = X.requests("c4", "served", 3)
    q = np.minimum((o["pos"] + 1) // 4, 1024)
    t = T.s32(o["topk"])
    inside = np.arange(1024)[None, :] < q[:, None]
    assert ((t >= -1) & (t < np.maximum(q, 1)[:, None]))[inside].all()
    assert (np.abs(t[~inside]) > 1 << 20).mean() > 0.9


@pytest.mark.parametrize("case", ["c128-256", "c128-896"])
def test_short_rows_are_the_reference_at_every_special_position(case: str) -> None:
    """`short_rows` against `Dsv4ShortContextTopK_v1{K,128}` on the c128a builder's special positions (the int64 ends, around
    0 and K * 128) and a few served ones; the served positions come first, as the combine reads their rows."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate

    c = X.CASES_L[case]
    served = X.requests(case, "served", 1)["pos"]
    pos = X.c128a_positions(case, served, 1)
    assert np.array_equal(pos[:served.size], served) and pos[served.size:].size == len(X.SPECIAL_POSITIONS) + 6 + 64
    rows = X.short_rows(pos, c.topk, c.ratio)
    fn = bind(X.I.Dsv4ShortContextTopK, TOPK=c.topk, RATIO=c.ratio)
    pick = list(range(served.size, served.size + len(X.SPECIAL_POSITIONS) + 6)) + [0, served.size - 1]
    for i in pick:
        assert [int(v) for v in evaluate(fn, int(pos[i]) & ((1 << 64) - 1))] == rows[i].tolist(), int(pos[i])
    full = {int(p): r for p, r in zip(pos, rows)}
    assert (T.s32(full[X.I64_MAX]) == -1).all() and (T.s32(full[c.topk * 128 - 1]) == np.arange(c.topk)).all()
    assert (T.s32(full[127])[:2] == [0, -1]).all() and (T.s32(full[-1]) == -1).all()


def test_observe_passes_the_twins_words(observed) -> None:
    """On the twin's words every launch check holds: the runs agree, the views were returned, the allocated rows are the
    workspace's, and every other word of every buffer is the sentinel."""
    for case, ob in observed.items():
        want = [("combine", case), ("combine", f"{case}-edge")] + ([("c128a", case)] if X.CASES_L[case].ratio == 128 else [])
        assert sorted((x["phase"], x["case"]) for x in ob.recs) == sorted(want)
        for x in ob.recs:
            assert X._launch_ok(x), x


@pytest.mark.parametrize("what", ["sentinel", "second_run", "alloc", "view", "c128a_past_k", "c128a_decode"])
def test_the_launch_checks_catch_each_slip(observed, tmp_path, what: str) -> None:
    """A word written past the token rows, a second launch that differs, an allocated row that differs from the workspace's, a
    launch that did not return its views, a c128a word past K, and a decode buffer written each fail their check."""
    case = "c128-256"
    ob = observed[case]
    combine, c128a = copy.deepcopy(ob.combine), copy.deepcopy(ob.c128a)
    s = combine["served"]
    if what == "sentinel":
        s["runs"][0]["ws"][0, 3] = 0
        s["runs"][1]["ws"][0, 3] = 0
    elif what == "second_run":
        s["runs"][1]["wl"][X.LENS_OFF] ^= 1
    elif what == "alloc":
        s["alloc"]["rows"][2, 5] ^= 1
    elif what == "view":
        s["views"] = False
    elif what == "c128a_past_k":
        for r in c128a["runs"]:
            r["pf"][0, X.CASES_L[case].topk] = 0
    else:
        for r in c128a["runs"]:
            r["dl"][0] = 0
    recs, _ = X.observe(tmp_path, case, ob.ops, c128a, combine, quick=True)
    failed = {(x["phase"], x["case"], k) for x in recs for k, v in x["checks"].items() if v is not True}
    want = {"sentinel": ("combine", case, "untouched"), "second_run": ("combine", case, "deterministic"),
            "alloc": ("combine", case, "alloc_is_out"), "view": ("combine", case, "views"),
            "c128a_past_k": ("c128a", case, "untouched"), "c128a_decode": ("c128a", case, "untouched")}[what]
    assert failed == {want}


def _no_refs(monkeypatch) -> None:
    monkeypatch.setattr(X, "_ref_rows", lambda lo, hi, every: [])


def _flipped(tmp_path, job: dict, key: str, at: tuple) -> dict:
    a = {k: np.array(v) for k, v in X._load(job["paths"]).items()}
    a[key][at] ^= 1
    return {**job, "paths": X._save(tmp_path, **a)}


@pytest.mark.parametrize("case", SIM)
def test_the_jobs_take_every_word_and_find_a_flipped_one(observed, case: str, tmp_path, monkeypatch) -> None:
    """Every job against the twin (the references off): every token's WIDTH + 1 words and every c128a row's K, 0 mismatches; a
    word flipped in a row, a length and a c128a row each found at its position and word."""
    _no_refs(monkeypatch)
    ob = observed[case]
    st = X.statics(case)
    tot: dict = {}
    for j in ob.jobs:
        r = X.compare(j)
        assert "error" not in r, r
        t = tot.setdefault((r["phase"], r["case"]), [0, 0])
        t[0] += r["words"]
        t[1] += r["mismatch"]
    rec = {(x["phase"], x["case"]): x for x in ob.recs}
    want = {("combine", n): rec[("combine", n)]["tokens"] * (st["WIDTH"] + 1) for n in (case, f"{case}-edge")}
    if ("c128a", case) in rec:
        want[("c128a", case)] = rec[("c128a", case)]["rows"] * st["TOPK"]
    assert tot == {k: [v, 0] for k, v in want.items()}
    j = [j for j in ob.jobs if j["kind"] == "combine" and j["case"] == f"{case}-edge"][1]
    lo = j["rows"][0]
    pos = T.s32(X._load(j["paths"])["pos"])
    r = X.compare(_flipped(tmp_path / "rows", j, "got_rows", (lo + 1, 7)))
    assert r["mismatch"] == 1 and r["first_mismatch"] == {"position": int(pos[lo + 1]), "word": 7, "want": r["first_mismatch"]["want"],
                                                          "kernel": r["first_mismatch"]["want"] ^ 1}
    r = X.compare(_flipped(tmp_path / "lens", j, "got_lens", (lo + 2,)))
    assert r["mismatch"] == 1 and r["first_mismatch"]["word"] == st["WIDTH"] and r["first_mismatch"]["position"] == int(pos[lo + 2])
    if ("c128a", case) in rec:
        j = [j for j in ob.jobs if j["kind"] == "c128a"][-1]
        lo = j["rows"][0]
        r = X.compare(_flipped(tmp_path / "c128a", j, "got", (lo, 3)))
        assert r["mismatch"] == 1 and r["first_mismatch"]["position"] == int(X._load(j["paths"])["pos"][lo])


def test_the_references_agree_with_the_twins(observed) -> None:
    """The reference Definitions on one token of each job kind (the first served token, an edge token at each int32 end, a c128a
    row at INT64_MAX) agree with the twins on every word."""
    for case, ob in observed.items():
        seen = set()
        for j in ob.jobs:
            a = X._load(j["paths"])
            pos = np.asarray(a["pos"]) if j["kind"] == "c128a" else T.s32(a["pos"])
            lo, hi = j["rows"]
            picks = [i for i in range(lo, hi) if int(pos[i]) in (0, X.I32_MAX - 1, X.I32_MIN + 127, X.I64_MAX)]
            for i in picks[:4]:
                r = X.compare({**j, "rows": [i, i + 1], "ref_every": 1})
                assert r["reference"] == 1 and r["twin_mismatch"] == 0 and r["mismatch"] == 0, (case, j["case"], int(pos[i]))
                seen.add((j["kind"], int(pos[i])))
        want = {("combine", p) for p in (0, X.I32_MAX - 1, X.I32_MIN + 127)}
        assert want | ({("c128a", 0), ("c128a", X.I64_MAX)} if X.CASES_L[case].ratio == 128 else set()) <= seen, seen


def test_the_jobs_reference_every_instance_or_a_sample(observed, tmp_path) -> None:
    """A full run references every token; --quick every 16th served one and each job's first, and every edge one and every
    c128a special."""
    for case, ob in observed.items():
        served = ob.ops["served"]["pos"].size
        for j in ob.jobs:
            edge = j["case"].endswith("-edge") or (j["kind"] == "c128a" and j["rows"][1] > served)
            assert j["ref_every"] == (1 if edge else X.QUICK_REF_EVERY), j
        _, full = X.observe(tmp_path / case, case, ob.ops, ob.c128a, ob.combine, quick=False)
        assert {j["ref_every"] for j in full} == {1}


def test_a_token_outside_the_domain_is_an_error(observed, tmp_path) -> None:
    """A job holding a position the twin declines (INT32_MAX) is an error, not a pass."""
    j = [j for j in observed["c4"].jobs if j["kind"] == "combine"][0]
    a = {k: np.array(v) for k, v in X._load(j["paths"]).items()}
    a["pos"][j["rows"][0]] = X.I32_MAX
    r = X.compare({**j, "paths": X._save(tmp_path, **a)})
    assert "outside the Definition's domain" in r["error"]


def _ptx(counts: dict, target: str = "sm_90a") -> str:
    body = "\n".join(f"\t{op} \t%r{i}, %r1, %r2;" for op, k in counts.items() for i in range(k))
    return f".version 8.7\n.target {target}\n.address_size 64\n{body}\n\tadd.s32 \t%r9, %r1, 4;\n"


@pytest.mark.parametrize("slip", [None, "fewer", "combine_div", "c128a_floor", "target"])
def test_the_ptx_check(tmp_path, slip) -> None:
    """One combine file per TOP_K with PTX_COUNTS["combine"] and a c128a file with its division and min pass; one TOP_K's compile
    missing, a division in the combine, the c128a builder without its int64 division, or another target fail."""
    cache = tmp_path / "triton_cache"
    combine = dict(X.PTX_COUNTS["combine"])
    files = [combine] * (3 if slip == "fewer" else 4)
    if slip == "combine_div":
        files = files[:-1] + [{**combine, "div.s32": 1}]
    for i, cnt in enumerate(files):
        (cache / f"h{i}").mkdir(parents=True)
        (cache / f"h{i}" / "kernel.ptx").write_text(_ptx(cnt, "sm_80" if slip == "target" and i == 0 else "sm_90a"))
    c128a = {"div.s64": 0 if slip == "c128a_floor" else 1, "min.s64": 1, "div.s32": 8, "mul.lo.s32": 11}
    (cache / "h9").mkdir()
    (cache / "h9" / "_build_c128a_topk_metadata_kernel.ptx").write_text(_ptx(c128a))
    rec = X.ptx(cache, tmp_path / "out")
    failed = {k for k, v in rec["checks"].items() if not v}
    assert failed == {None: set(), "fewer": {"combine"}, "combine_div": {"combine"}, "c128a_floor": {"c128a"},
                      "target": {"target"}}[slip]
    assert len(list((tmp_path / "out" / "ptx").glob("*.ptx"))) == len(files) + 1


def test_the_sources_are_pinned_by_hash(tmp_path) -> None:
    rel = list(X.SOURCES)
    (tmp_path / rel[0]).parent.mkdir(parents=True)
    (tmp_path / rel[0]).write_text("x")
    rec = X.sources(tmp_path)
    assert rec["ok"] is False and rec["files"][rel[0]] is False and rec["files"][rel[1]] is None
    assert hashlib.sha256(b"x").hexdigest() not in X.SOURCES.values()


def test_the_programs_bind() -> None:
    rec = X.bind_check()
    assert rec["ok"], rec
    assert len(rec["bindings"]) == len(X.CASES_L) + 3


def _doc() -> dict:
    launches = [{"phase": p, "case": c, "checks": {"a": True, "b": True}} for p, c in X.CASES]
    comps = [{"phase": p, "case": c, "words": 10, "mismatch": 0, "reference": 1, "twin_mismatch": 0, "first_mismatch": None}
             for p, c in X.CASES]
    return {"schema": X.SCHEMA, "gate": {"ok": True}, "errors": {}, "selection": {"ok": True}, "ptx": {"ok": True},
            "bind": {"ok": True}, "launches": launches, "comparisons": comps, "totals": {k: sum(r[k] for r in comps) for k in X.TOTALS}}


@pytest.mark.parametrize("path,value", [(("gate", "ok"), False), (("selection", "ok"), False), (("ptx", "ok"), False),
                                        (("bind", "ok"), False), (("errors", "stack"), "boom"),
                                        (("launches", 0, "checks", "b"), False), (("launches", 10, "checks", "a"), False),
                                        (("comparisons", 1, "mismatch"), 1), (("comparisons", 4, "twin_mismatch"), 3),
                                        (("comparisons", 5, "reference"), 0), (("comparisons", 6, "words"), 0),
                                        (("totals", "mismatch"), 1)])
def test_the_verdict_needs_every_check(path: tuple, value) -> None:
    assert X._verdict(_doc())
    doc = _doc()
    node = doc
    for p in path[:-1]:
        node = node[p]
    node[path[-1]] = value
    assert not X._verdict(doc)


def test_each_case_has_its_own_verdict() -> None:
    """A case is its phase and its name: a layer case's launch error fails that case's c128a, served and edge cases and no other, a
    comparison's error its own case, a missing launch record its case."""
    want = [f"{p}/{c}" for p, c in X.CASES]
    assert len(want) == 11 and list(X.cases(_doc())) == want and all(c["ok"] for c in X.cases(_doc()).values())
    doc = _doc()
    doc["errors"]["c128-896"] = "boom"
    doc["errors"]["compare/combine/c4-edge"] = "boom"
    doc["launches"] = [x for x in doc["launches"] if (x["phase"], x["case"]) != ("combine", "c128-256")]
    assert {k for k, c in X.cases(doc).items() if not c["ok"]} == {"c128a/c128-896", "combine/c128-896", "combine/c128-896-edge",
                                                                   "combine/c4-edge", "combine/c128-256"}


def test_main_end_to_end_on_stubbed_launches(observed, tmp_path, monkeypatch, capsys) -> None:
    """`main` with the GPU stood in for (the ratio-4 case's twin-made words: its first served job and an edge job with one word
    flipped) and vLLM's config module stubbed: the jobs through the spawned worker pool, the per-case verdicts, the sealed record
    and the exit status, so a slip in the orchestration shows here and not on the H200."""
    ob = observed["c4"]
    for k in ("TRITON_CACHE_DIR", "VLLM_ENABLE_V1_MULTIPROCESSING"):
        monkeypatch.setenv(k, "")
    vc = types.ModuleType("vllm.config")
    vc.CompilationConfig = vc.VllmConfig = lambda **kw: kw
    vc.set_current_vllm_config = lambda cfg: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.config", vc)
    monkeypatch.setattr(X.C, "gate", lambda *a: {"ok": True})
    monkeypatch.setattr(X.G, "_environment", lambda: {})
    monkeypatch.setattr(X, "bind_check", lambda: {"bindings": {}, "ok": True})
    monkeypatch.setattr(X, "ptx", lambda cache, outdir: {"ok": True})
    by = {}
    for j in ob.jobs:
        by.setdefault(j["case"], j)
    edge = _flipped(tmp_path / "bad", {**by["c4-edge"], "rows": [0, 2]}, "got_rows", (1, 1000))

    def gpu(work, outdir, quick, doc):
        doc["selection"] = {"ok": True}
        doc["launches"] += [x for x in ob.recs]
        return [{**by["c4"], "rows": [0, 3]}, edge]

    monkeypatch.setattr(X, "gpu", gpu)
    out = tmp_path / "out"
    assert X.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1", "--quick"]) == 1
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["digest"] == digest_of(doc) and doc["ok"] is False and not doc["errors"]
    assert {c: v["ok"] for c, v in doc["cases"].items()} == {c: c == "combine/c4" for c in X.cases(_doc())}
    words = X.statics("c4")["WIDTH"] + 1
    assert doc["cases"]["combine/c4-edge"]["totals"] == {"words": 2 * words, "mismatch": 1, "reference": 2, "twin_mismatch": 0}
    assert doc["cases"]["combine/c4"]["totals"] == {"words": 3 * words, "mismatch": 0, "reference": 1, "twin_mismatch": 0}
    printed = capsys.readouterr().out
    assert "DSV4-INDICES NO-GO" in printed
    assert f"no-go cases={[c for c in X.cases(_doc()) if c != 'combine/c4']}" in printed


def test_the_record_is_a_registered_property() -> None:
    from verity_vllm.properties.record import seal

    doc = seal(_doc(), X._verdict(_doc()))
    assert REGISTRY["dsv4_indices_capture"].schema == X.SCHEMA and doc["ok"] and doc["digest"] == digest_of(doc)
