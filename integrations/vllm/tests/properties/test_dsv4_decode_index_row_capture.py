"""The H200 capture of DeepSeek-V4-Pro's ratio-4 decode index row (`dsv4_decode_index_row_capture_gpu`) on the CPU: its steps'
operands, its launch checks and comparison jobs on what the twin says the kernel writes, its PTX check, its verdicts and its main
loop with the GPU stood in for."""
from __future__ import annotations

__models__ = ("deepseek-ai/DeepSeek-V4-Pro",)

import contextlib
import copy
import hashlib
import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from tests.properties import dsv4_decode_index_row_capture_gpu as X
from verity.ml.definitions.numerics import dsv4_decode_indices_twin as T
from verity_vllm.properties.record import REGISTRY, digest_of


def simulate_launch(case: str, o: dict) -> dict:
    """What `launch` would return were the kernel the twin."""
    st, n = X.statics(case), o["topk"].shape[0]
    rows, lens, declined = T.global_topk(o["topk"], o["table"][o["req"]], o["valid"], **st)
    assert not declined.any()
    ws = np.full((n + X.WS_OFF + 1, st["TOPK"]), X.SENTINEL, np.uint32)
    ws[X.WS_OFF:X.WS_OFF + n] = rows
    wl = np.full(n + X.LENS_OFF + 4, X.SENTINEL, np.uint32)
    wl[X.LENS_OFF:X.LENS_OFF + n] = lens
    run = {"rows": rows, "lens": lens, "shapes": [[n, st["TOPK"]], [n]]}
    return {"runs": [run, copy.deepcopy(run)], "direct": {"ws": ws, "wl": wl}, "operands": True}


def simulate(case: str, seeds=(7, 8)):
    ops = {kind: X.requests(case, kind, seed) for kind, seed in zip(("served", "edge"), seeds)}
    return ops, {kind: simulate_launch(case, ops[kind]) for kind in ops}


@pytest.fixture(scope="module")
def observed(tmp_path_factory):
    """Each case's operands, the twin's words as the GPU's, and `observe`'s records and quick jobs on them."""
    out = {}
    for case in X.CASES_L:
        ops, rows = simulate(case)
        work = tmp_path_factory.mktemp(case)
        recs, jobs = X.observe(work, case, ops, rows, quick=True)
        out[case] = SimpleNamespace(ops=ops, rows=rows, recs=recs, jobs=jobs)
    return out


def test_the_statics_are_dsv4s() -> None:
    """Top-k 1024, BS the kernel block 256 over ratio 4, NB cdiv(max_model_len, 256): the registry's served statics; one PTX count
    per kernel, the four sources, 16 served requests (max_num_seqs)."""
    for case, max_len in X.MAX_LEN.items():
        assert X.statics(case) == {"TOPK": 1024, "BS": X.KERNEL_BLOCK // X.RATIO, "NB": -(-max_len // X.KERNEL_BLOCK)}
        assert X.statics(case) == X.R.DSV4_DECODE_INDICES[case]
        assert len(X.POSITIONS[case]) == 16 and X.POSITIONS[case][0] == 0 and X.POSITIONS[case][-1] == max_len - 1
    assert X.statics("c4")["NB"] == 4096 and X.statics("c4-4096")["NB"] == 16
    assert set(X.PTX_COUNTS) == set(X.KERNELS) and len(X.SOURCES) == 4 and X.TRITON_BLOCK == X.statics("c4")["TOPK"]


@pytest.mark.parametrize("case", X.CASES_L)
def test_the_served_step_holds_what_it_means(case: str) -> None:
    """16 requests of one token each and two padding tokens (byte 0, request 0); each served row distinct ids below its token's
    compressed count, the newest among them, -1 past them; an empty row at position 0, a row one short of full and a full one, and
    the row at max_model_len - 1 reaching the row's last block; every valid index's block among its request's allocated ones."""
    st = X.statics(case)
    o = X.requests(case, "served", 1)
    nreq = len(X.POSITIONS[case])
    assert o["topk"].shape == (nreq + X.PADDING, st["TOPK"]) and o["table"].shape == (nreq, st["NB"])
    assert o["req"].tolist() == list(range(nreq)) + [0] * X.PADDING
    assert o["valid"].tolist() == [1] * nreq + [0] * X.PADDING
    x = T.s32(o["topk"])
    for r, p in enumerate(X.POSITIONS[case]):
        q = (p + 1) // X.RATIO
        k = min(q, st["TOPK"])
        row = x[r]
        assert (row[k:] == -1).all() and len(set(row[:k].tolist())) == k and (row[:k] < q).all() and (row[:k] >= 0).all()
        assert k == 0 or q - 1 in row[:k]
        assert (x[r, :k] // st["BS"] < -(-q // st["BS"])).all()
    lens = (x[:nreq] >= 0).sum(axis=1)
    assert lens[0] == 0 and (lens == st["TOPK"]).any() and (lens == st["TOPK"] - 1).any()
    assert x[nreq - 1].max() // st["BS"] == st["NB"] - 1
    assert not T.global_topk(o["topk"], o["table"][o["req"]], o["valid"], **st)[2].any()
    pad = x[nreq:]
    assert ((pad >= 0) & (pad < st["NB"] * st["BS"]) | (pad == -1)).all() and (pad >= 0).any()


@pytest.mark.parametrize("case", X.CASES_L)
def test_the_edge_step_holds_what_it_means(case: str) -> None:
    """Six requests' tokens, every request's row of each kind, in a shuffled map; every byte 0, 1, 2, 0x80, 0xFF; empty, full and
    short rows, the negative kinds, the row's last slot; block numbers whose product by BS wraps int32, and negative ones, read by
    a valid index; no token the twin declines."""
    st = X.statics(case)
    o = X.requests(case, "edge", 2)
    n = X.EDGE_REQUESTS * X.EDGE_PER_REQUEST
    assert o["topk"].shape == (n, st["TOPK"]) and o["table"].shape == (X.EDGE_REQUESTS, st["NB"])
    assert sorted(set(zip(o["req"].tolist(), o["kinds"].tolist()))) == sorted(
        (r, k) for r in range(X.EDGE_REQUESTS) for k in set(X.EDGE_KINDS))
    assert (np.diff(o["req"]) < 0).any() and set(o["valid"].tolist()) == set(X.EDGE_BYTES)
    x = T.s32(o["topk"])
    live = x >= 0
    assert (~live).all(axis=1).any() and live.all(axis=1).any() and ((live.sum(axis=1) > 0) & (live.sum(axis=1) < 64)).any()
    assert {-1, -2, -st["BS"], X.I32_MIN} <= set(x[~live].tolist()) and st["NB"] * st["BS"] - 1 in x
    bn = T.s32(o["table"][o["req"]][np.arange(n)[:, None], np.where(live, x // st["BS"], 0)])[live]
    assert (bn * st["BS"] > X.I32_MAX).any() and (bn * st["BS"] < X.I32_MIN).any() and (bn < 0).any()
    assert not T.global_topk(o["topk"], o["table"][o["req"]], o["valid"], **st)[2].any()


def test_observe_passes_the_twins_words(observed) -> None:
    """On the twin's words every launch check holds: the runs agree, the shapes are the decode's, the direct launch wrote the
    wrapper's words and nothing around them, the operands are unchanged."""
    for case, ob in observed.items():
        assert sorted(x["case"] for x in ob.recs) == sorted([case, f"{case}-edge"])
        for x in ob.recs:
            assert X._launch_ok(x), x
        served = next(x for x in ob.recs if x["case"] == case)
        assert served["padding"] == X.PADDING and served["requests"] == 16


@pytest.mark.parametrize("what", ["sentinel", "lens_sentinel", "second_run", "direct", "shape", "operands"])
def test_the_launch_checks_catch_each_slip(observed, tmp_path, what: str) -> None:
    """A word written past the token rows, a length past the tokens, a second launch that differs, a direct launch that differs
    from the wrapper, an output of another shape, and an operand written each fail their check."""
    case = "c4-4096"
    ob = observed[case]
    rows = copy.deepcopy(ob.rows)
    s = rows["served"]
    if what == "sentinel":
        s["direct"]["ws"][0, 3] = 0
    elif what == "lens_sentinel":
        s["direct"]["wl"][-1] = 0
    elif what == "second_run":
        s["runs"][1]["lens"][0] ^= 1
    elif what == "direct":
        s["direct"]["ws"][X.WS_OFF + 2, 5] ^= 1
    elif what == "shape":
        s["runs"][0]["shapes"] = [[1, 1024], [1]]
    else:
        s["operands"] = False
    recs, _ = X.observe(tmp_path, case, ob.ops, rows, quick=True)
    failed = {(x["case"], k) for x in recs for k, v in x["checks"].items() if v is not True}
    want = {"sentinel": "untouched", "lens_sentinel": "untouched", "second_run": "deterministic", "direct": "direct",
            "shape": "shapes", "operands": "operands"}[what]
    assert failed == {(case, want)}


def _stand_in_cache_utils(monkeypatch, calls: list) -> None:
    """vLLM's `cache_utils` stood in for on the CPU: the wrapper and the kernel the twin over the tensors they are handed, each
    call's arguments recorded."""
    torch = pytest.importorskip("torch")

    def stores(topk, req, table, bs, valid):
        n, nb = topk.shape[0], table.shape[1]
        rows, lens, declined = T.global_topk(topk.numpy().view(np.uint32), table.numpy().view(np.uint32)[req[:n].numpy()],
                                             valid.view(torch.uint8).numpy(), topk.shape[1], bs, nb)
        assert not declined.any()
        return torch.from_numpy(rows.view(np.int32)), torch.from_numpy(lens.view(np.int32))

    def wrapper(topk, req, table, bs, valid):
        calls.append({"op": "wrapper", "topk": (tuple(topk.shape), topk.stride(0)), "req": tuple(req.shape),
                      "table": (tuple(table.shape), table.stride(0)), "bs": bs, "valid": (valid.dtype, tuple(valid.shape))})
        return stores(topk, req, table, bs, valid)

    class Kernel:
        def __getitem__(self, grid):
            def run(out, out_stride, lens, topk, topk_stride, k, req, table, table_stride, bs, valid, TRITON_BLOCK_SIZE):
                calls.append({"op": "kernel", "grid": grid, "strides": (out_stride, topk_stride, k, table_stride), "bs": bs,
                              "block": TRITON_BLOCK_SIZE, "valid": valid.dtype})
                rows, n = stores(topk, req, table, bs, valid)
                out.copy_(rows)
                lens.copy_(n)
            return run

    cu = types.ModuleType("vllm.models.deepseek_v4.common.ops.cache_utils")
    cu.compute_global_topk_indices_and_lens = wrapper
    cu._compute_global_topk_indices_and_lens_kernel = Kernel()
    for name in ("vllm", "vllm.models", "vllm.models.deepseek_v4", "vllm.models.deepseek_v4.common",
                 "vllm.models.deepseek_v4.common.ops"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, cu.__name__, cu)
    monkeypatch.setattr(X, "DEVICE", "cpu")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)


@pytest.mark.slow
def test_launch_hands_the_kernel_the_decodes_operands(tmp_path, monkeypatch) -> None:
    """`launch` on CPU tensors with the kernel the twin: the wrapper gets the [:tokens] top-k slice of a TOPK-wide buffer, the
    whole `token_to_req_indices` buffer, the [:requests] table of stride NB, block size 64 and a bool valid of the tokens' bytes
    (2, 0x80 and 0xFF kept); the direct launch one program per token at the wrapper's constexprs; and `observe` passes every check
    and every word on what comes back."""
    calls: list = []
    _stand_in_cache_utils(monkeypatch, calls)
    torch = sys.modules["torch"]
    case = "c4-4096"
    st = X.statics(case)
    ops = {kind: X.requests(case, kind, seed) for kind, seed in zip(("served", "edge"), (3, 4))}
    rows = {kind: X.launch(case, ops[kind]) for kind in ops}
    for kind, o in ops.items():
        n, nreq = o["topk"].shape[0], o["table"].shape[0]
        mine = calls[:3] if kind == "served" else calls[3:]
        assert [c["op"] for c in mine] == ["wrapper", "wrapper", "kernel"]
        assert mine[0] == {"op": "wrapper", "topk": ((n, st["TOPK"]), st["TOPK"]), "req": (n + X.REQ_PAD,),
                           "table": ((nreq, st["NB"]), st["NB"]), "bs": st["BS"], "valid": (torch.bool, (n,))}
        assert mine[2] == {"op": "kernel", "grid": (n,), "strides": (st["TOPK"], st["TOPK"], st["TOPK"], st["NB"]), "bs": st["BS"],
                           "block": X.TRITON_BLOCK, "valid": torch.bool}
    recs, jobs = X.observe(tmp_path, case, ops, rows, quick=False)
    assert all(X._launch_ok(x) for x in recs) and len(recs) == 2, recs
    for j in jobs:
        r = X.compare(j)
        assert "error" not in r and r["mismatch"] == 0 and r["twin_mismatch"] == 0 and r["reference"] == j["rows"][1] - j["rows"][0], r
    assert set(rows["edge"]["runs"][0]["lens"][ops["edge"]["valid"] == 0].tolist()) == {0}
    assert (rows["edge"]["runs"][0]["lens"][np.isin(ops["edge"]["valid"], [2, 0x80, 0xFF])] > 0).any()


def _no_refs(monkeypatch) -> None:
    monkeypatch.setattr(X, "_ref_rows", lambda lo, hi, every: [])


def _flipped(tmp_path, job: dict, key: str, at: tuple) -> dict:
    a = {k: np.array(v) for k, v in X._load(job["paths"]).items()}
    a[key][at] ^= 1
    return {**job, "paths": X._save(tmp_path, **a)}


@pytest.mark.parametrize("case", X.CASES_L)
def test_the_jobs_take_every_word_and_find_a_flipped_one(observed, case: str, tmp_path, monkeypatch) -> None:
    """Every job against the twin (the references off): every token's TOPK + 1 words, 0 mismatches; a word flipped in a row and
    a length each found at its token and word."""
    _no_refs(monkeypatch)
    ob = observed[case]
    st = X.statics(case)
    tot: dict = {}
    for j in ob.jobs:
        r = X.compare(j)
        assert "error" not in r, r
        t = tot.setdefault(r["case"], [0, 0])
        t[0] += r["words"]
        t[1] += r["mismatch"]
    assert tot == {x["case"]: [x["tokens"] * (st["TOPK"] + 1), 0] for x in ob.recs}
    j = [j for j in ob.jobs if j["case"] == f"{case}-edge"][1]
    lo = j["rows"][0]
    r = X.compare(_flipped(tmp_path / "rows", j, "got_rows", (lo + 1, 7)))
    assert r["mismatch"] == 1 and r["first_mismatch"] == {"token": lo + 1, "word": 7, "want": r["first_mismatch"]["want"],
                                                          "kernel": r["first_mismatch"]["want"] ^ 1}
    r = X.compare(_flipped(tmp_path / "lens", j, "got_lens", (lo + 2,)))
    assert r["mismatch"] == 1 and r["first_mismatch"]["word"] == st["TOPK"] and r["first_mismatch"]["token"] == lo + 2


@pytest.mark.slow
def test_the_references_agree_with_the_twin(observed) -> None:
    """The reference `Dsv4GlobalTopk_v1` on every edge token of the small row and on the served step's first, last and padding
    tokens at both statics agrees with the twin on every word."""
    for case, ob in observed.items():
        n_served = ob.ops["served"]["topk"].shape[0]
        picks = {case: [0, len(X.POSITIONS[case]) - 1, n_served - 1]}
        if case == "c4-4096":
            picks[f"{case}-edge"] = list(range(X.EDGE_REQUESTS * X.EDGE_PER_REQUEST))
        for j in ob.jobs:
            lo, hi = j["rows"]
            for i in (i for i in picks.get(j["case"], []) if lo <= i < hi):
                r = X.compare({**j, "rows": [i, i + 1], "ref_every": 1})
                assert r["reference"] == 1 and r["twin_mismatch"] == 0 and r["mismatch"] == 0, (j["case"], i)


def test_the_jobs_reference_every_token_or_a_sample(observed, tmp_path) -> None:
    """A full run references every token; --quick every 16th served one and each job's first, and every edge one."""
    for case, ob in observed.items():
        for j in ob.jobs:
            assert j["ref_every"] == (1 if j["case"].endswith("-edge") else X.QUICK_REF_EVERY), j
        _, full = X.observe(tmp_path / case, case, ob.ops, ob.rows, quick=False)
        assert {j["ref_every"] for j in full} == {1}


def test_a_token_outside_the_domain_is_an_error(observed, tmp_path) -> None:
    """A job holding a valid index whose block is past its row (the kernel would read the next request's) is an error, not a
    pass."""
    j = [j for j in observed["c4-4096"].jobs if j["case"] == "c4-4096"][0]
    a = {k: np.array(v) for k, v in X._load(j["paths"]).items()}
    a["topk"][j["rows"][0], 0] = X.statics("c4-4096")["NB"] * X.statics("c4-4096")["BS"]
    r = X.compare({**j, "paths": X._save(tmp_path, **a)})
    assert "past their block row" in r["error"]


def _ptx(counts: dict, target: str = "sm_90a") -> str:
    body = "\n".join(f"\t{op} \t%r{i}, %r1, %r2;" for op, k in counts.items() for i in range(k))
    return f".version 8.7\n.target {target}\n.address_size 64\n{body}\n\tadd.s32 \t%r9, %r1, 4;\n\tld.global.b32 \t%r3, [%rd1];\n"


@pytest.mark.parametrize("slip", [None, "fewer", "div", "rem", "no_popc", "target"])
def test_the_ptx_check(tmp_path, slip) -> None:
    """One file per NB with PTX_COUNTS["row"] passes; one NB's compile missing, a division or a remainder in it, the count without
    its popc, or another target fail."""
    cache = tmp_path / "triton_cache"
    row = dict(X.PTX_COUNTS["row"])
    files = [row] * (1 if slip == "fewer" else 2)
    if slip == "div":
        files = [row, {**row, "div.s32": 1}]
    elif slip == "rem":
        files = [row, {**row, "rem.s32": 1}]
    elif slip == "no_popc":
        files = [row, {k: v for k, v in row.items() if k != "popc.b32"}]
    for i, cnt in enumerate(files):
        (cache / f"h{i}").mkdir(parents=True)
        (cache / f"h{i}" / f"{X.KERNELS['row']}.ptx").write_text(_ptx(cnt, "sm_80" if slip == "target" and i == 0 else "sm_90a"))
    rec = X.ptx(cache, tmp_path / "out")
    failed = {k for k, v in rec["checks"].items() if not v}
    assert failed == ({"target"} if slip == "target" else set() if slip is None else {"row"})
    assert len(list((tmp_path / "out" / "ptx").glob("*.ptx"))) == len(files)


def test_the_ptx_counts_are_the_kernels_ops() -> None:
    """The counted ops are the ones the module docstring reads: `//` and `%` by BS as shifts and masks (no division, remainder or
    mul.lo), the count as a popc, a warp reduction and two shuffles, the byte as one 8-bit load."""
    want = X.PTX_COUNTS["row"]
    assert (want["shr.s32"], want["shr.u32"], want["sub.s32"]) == (16, 10, 8)
    assert (want["popc.b32"], want["redux.sync.add.s32"], want["shfl.sync.bfly.b32"], want["ld.global.b8"]) == (1, 1, 2, 1)
    assert not {k for k in want if k.startswith(("div.", "rem.", "mul.lo", "mul.hi", "min.", "max."))}
    assert X.ptx_counts(_ptx(want)) == want


def test_the_sources_are_pinned_by_hash(tmp_path) -> None:
    rel = list(X.SOURCES)
    (tmp_path / rel[0]).parent.mkdir(parents=True)
    (tmp_path / rel[0]).write_text("x")
    rec = X.sources(tmp_path)
    assert rec["ok"] is False and rec["files"][rel[0]] is False and rec["files"][rel[1]] is None
    assert hashlib.sha256(b"x").hexdigest() not in X.SOURCES.values()


def test_the_program_binds() -> None:
    rec = X.bind_check()
    assert rec["ok"], rec
    assert len(rec["bindings"]) == len(X.CASES_L)


def _doc() -> dict:
    launches = [{"phase": p, "case": c, "checks": {"a": True, "b": True}} for p, c in X.CASES]
    comps = [{"phase": p, "case": c, "words": 10, "mismatch": 0, "reference": 1, "twin_mismatch": 0, "first_mismatch": None}
             for p, c in X.CASES]
    return {"schema": X.SCHEMA, "gate": {"ok": True}, "errors": {}, "selection": {"ok": True}, "ptx": {"ok": True},
            "bind": {"ok": True}, "launches": launches, "comparisons": comps, "totals": {k: sum(r[k] for r in comps) for k in X.TOTALS}}


@pytest.mark.parametrize("path,value", [(("gate", "ok"), False), (("selection", "ok"), False), (("ptx", "ok"), False),
                                        (("bind", "ok"), False), (("errors", "stack"), "boom"),
                                        (("launches", 0, "checks", "b"), False), (("launches", 3, "checks", "a"), False),
                                        (("comparisons", 1, "mismatch"), 1), (("comparisons", 2, "twin_mismatch"), 3),
                                        (("comparisons", 3, "reference"), 0), (("comparisons", 0, "words"), 0),
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
    """A case is its phase and its name: a case's launch error fails its served and edge cases and no other, a comparison's error
    its own case, a missing launch record its case."""
    want = [f"{p}/{c}" for p, c in X.CASES]
    assert want == ["row/c4", "row/c4-edge", "row/c4-4096", "row/c4-4096-edge"]
    assert list(X.cases(_doc())) == want and all(c["ok"] for c in X.cases(_doc()).values())
    doc = _doc()
    doc["errors"]["c4-4096"] = "boom"
    assert {k for k, c in X.cases(doc).items() if not c["ok"]} == {"row/c4-4096", "row/c4-4096-edge"}
    doc = _doc()
    doc["errors"]["compare/row/c4-edge"] = "boom"
    doc["launches"] = [x for x in doc["launches"] if x["case"] != "c4"]
    assert {k for k, c in X.cases(doc).items() if not c["ok"]} == {"row/c4-edge", "row/c4"}


def test_main_end_to_end_on_stubbed_launches(observed, tmp_path, monkeypatch, capsys) -> None:
    """`main` with the GPU stood in for (the small row's twin-made words: its first served job and an edge job with one word
    flipped) and vLLM's config module stubbed: the jobs through the spawned worker pool, the per-case verdicts, the sealed record
    and the exit status, so a slip in the orchestration shows here and not on the H200."""
    ob = observed["c4-4096"]
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
    edge = _flipped(tmp_path / "bad", {**by["c4-4096-edge"], "rows": [0, 2]}, "got_rows", (1, 1000))

    def gpu(work, outdir, quick, doc):
        doc["selection"] = {"ok": True}
        doc["launches"] += [x for x in ob.recs]
        return [{**by["c4-4096"], "rows": [0, 3]}, edge]

    monkeypatch.setattr(X, "gpu", gpu)
    out = tmp_path / "out"
    assert X.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1", "--quick"]) == 1
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["digest"] == digest_of(doc) and doc["ok"] is False and not doc["errors"]
    assert {c: v["ok"] for c, v in doc["cases"].items()} == {c: c == "row/c4-4096" for c in X.cases(_doc())}
    words = X.statics("c4-4096")["TOPK"] + 1
    assert doc["cases"]["row/c4-4096-edge"]["totals"] == {"words": 2 * words, "mismatch": 1, "reference": 2, "twin_mismatch": 0}
    assert doc["cases"]["row/c4-4096"]["totals"] == {"words": 3 * words, "mismatch": 0, "reference": 1, "twin_mismatch": 0}
    printed = capsys.readouterr().out
    assert "DSV4-DECODE-INDEX-ROW NO-GO" in printed
    assert f"no-go cases={[c for c in X.cases(_doc()) if c != 'row/c4-4096']}" in printed


def test_a_failed_gate_writes_a_sealed_record_and_exits_3(tmp_path, monkeypatch, capsys) -> None:
    """The device gate failing writes the sealed NO-GO record, prints GATE-FAIL and exits 3 before any launch."""
    for k in ("TRITON_CACHE_DIR", "VLLM_ENABLE_V1_MULTIPROCESSING"):
        monkeypatch.setenv(k, "")
    vc = types.ModuleType("vllm.config")
    vc.CompilationConfig = vc.VllmConfig = lambda **kw: kw
    vc.set_current_vllm_config = lambda cfg: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.config", vc)
    monkeypatch.setattr(X.C, "gate", lambda *a: {"ok": False, "sms": 114})
    monkeypatch.setattr(X.G, "_environment", lambda: {})
    monkeypatch.setattr(X, "gpu", lambda *a: pytest.fail("launched past a failed gate"))
    with pytest.raises(SystemExit) as e:
        X.main(["--out", str(tmp_path / "out"), "--scratch", str(tmp_path)])
    assert e.value.code == 3
    doc = json.loads((tmp_path / "out" / X.RECORD).read_text())
    assert doc["ok"] is False and doc["digest"] == digest_of(doc) and doc["gate"] == {"ok": False, "sms": 114}
    assert "DSV4-DECODE-INDEX-ROW GATE-FAIL" in capsys.readouterr().out


def test_the_record_is_a_registered_property() -> None:
    from verity_vllm.properties.record import seal

    doc = seal(_doc(), X._verdict(_doc()))
    assert REGISTRY["dsv4_decode_index_row_capture"].schema == X.SCHEMA and doc["ok"] and doc["digest"] == digest_of(doc)
