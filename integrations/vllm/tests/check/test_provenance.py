"""The exhaustive operand-provenance check (G5, D82): the independent last-writer map reproduces every read of a
consistent fold, and detects a wrong wiring / a stale generation / a partial clobber when one is planted."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity_vllm.check import operand_provenance as P
from verity_vllm.observe.events import DispatchEnter, DispatchExit, ViewDesc
from verity_vllm.observe.fold.fold import fold
from verity_vllm.observe.fold.views import byte_extent, collapsed_dims, element_runs, row_view
from verity_vllm.observe.log import LogWriter
from verity_vllm.observe.fold.memory import access_runs
from tests.observe import test_fold_synthetic as tfs

M1 = Path(__file__).resolve().parents[2] / "data" / "logs" / "m1.jsonl.gz"
M6 = Path(__file__).resolve().parents[2] / "data" / "logs" / "m6" / "log.jsonl.gz"


def V(alloc, offset, shape, strides=None, dtype="bfloat16"):
    if strides is None:
        strides = []
        s = 1
        for n in reversed(shape):
            strides.append(s)
            s *= n
        strides = list(reversed(strides))
    return ViewDesc(alloc=alloc, offset=offset, shape=tuple(shape), strides=tuple(strides), dtype=dtype, device="cuda:0")


def acc(seq, kind, view=None, refs=None, alloc=None, rows=None, event=1, step=0):
    return {"seq": seq, "event": event, "step": step, "kind": kind, "view": view, "alloc": alloc if alloc is not None else (view.alloc if view else None),
            "rows": rows, "refs": refs, "what": ""}


# --- run arithmetic -----------------------------------------------------------------------------------------------


def test_canon_and_slice():
    rs = [("n", 3, 0, 4), ("n", 3, 4, 8), ("p", 0, 10, 12), ("n", 3, 8, 9)]
    assert P._canon(rs) == [("n", 3, 0, 8), ("p", 0, 10, 12), ("n", 3, 8, 9)]
    assert P._slice_runs(P._canon(rs), 6, 4) == [("n", 3, 6, 8), ("p", 0, 10, 12)]
    assert P._slice_runs(P._canon(rs), 0, 0) == []
    assert P._count(P._canon(rs)) == 11


# --- the byte map -------------------------------------------------------------------------------------------------


def test_last_writer_reuse_and_partial_clobber():
    a = P._Alloc()
    a.write(0, 16, 2, [("n", 1, 0, 8)], seq=1)            # 8 bf16 elements from node 1
    assert a.read(0, 16, 2) == ([("n", 1, 0, 8)], 0)
    assert a.read(4, 12, 2) == ([("n", 1, 2, 6)], 0)       # a sub-view
    a.write(4, 8, 2, [("n", 2, 0, 2)], seq=2)              # elements 2,3 rewritten by node 2
    assert a.read(0, 16, 2) == ([("n", 1, 0, 2), ("n", 2, 0, 2), ("n", 1, 4, 8)], 0)
    got, missing = a.read(0, 16, 4)                        # width reinterpretation: unresolvable
    assert missing == 16 and got == []
    a.write(1, 3, 1, [("n", 3, 0, 2)], seq=3)              # a byte write inside element 0 and 1: both elements are clobbered
    got, missing = a.read(0, 4, 2)
    assert missing > 0


def test_checker_reproduces_and_detects():
    x = V(1, 0, [4])
    y = V(2, 0, [4])
    base = [
        acc(0, "alloc", alloc=1), acc(1, "alloc", alloc=2),
        acc(2, "root", x, [("p", 0, 0, 4)]),
        acc(3, "read", x, [("p", 0, 0, 4)]),
        acc(4, "write", y, [("n", 0, 0, 4)]),
        acc(5, "read", y, [("n", 0, 0, 4)]),
        acc(6, "read", V(2, 1, [2]), [("n", 0, 1, 3)]),
    ]
    ck = P.ProvenanceChecker()
    for r in base:
        ck.apply(r)
    assert ck.n_mismatch == 0 and ck.reads == 3 and ck.uncovered == 0
    # wrong producer wired
    bad = copy.deepcopy(base)
    bad[5]["refs"] = [("n", 7, 0, 4)]
    ck = P.ProvenanceChecker()
    for r in bad:
        ck.apply(r)
    assert ck.n_mismatch == 1 and ck.mismatches[0]["seq"] == 5
    # off-by-one slice
    bad = copy.deepcopy(base)
    bad[6]["refs"] = [("n", 0, 0, 2)]
    ck = P.ProvenanceChecker()
    for r in bad:
        ck.apply(r)
    assert ck.n_mismatch == 1
    # a new generation of allocation 2 before the read: the contents are unreachable (uncovered)
    stale = copy.deepcopy(base)
    stale.insert(5, acc(9, "alloc", alloc=2))
    ck = P.ProvenanceChecker()
    for r in stale:
        ck.apply(r)
    assert ck.uncovered == 2 and ck.n_mismatch == 2
    # a read of a freed allocation
    freed = copy.deepcopy(base)
    freed.insert(5, acc(9, "free", alloc=2))
    ck = P.ProvenanceChecker()
    for r in freed:
        ck.apply(r)
    assert ck.uncovered >= 1


def test_gather_many_rows_and_strided_views():
    kv = V(5, 0, [8, 4])                                   # 8 rows of 4
    steps = []
    steps.append(acc(0, "alloc", alloc=5))
    for i in range(8):
        steps.append(acc(1 + i, "write", V(5, 4 * i, [4]), [("n", 10 + i, 0, 4)]))
    # gather rows 6, 2, 7 -> refs in that order
    steps.append(acc(20, "read", kv, [("n", 16, 0, 4), ("n", 12, 0, 4), ("n", 17, 0, 4)], rows=[6, 2, 7]))
    # a column slice (strided view): column 1 of every row
    col = V(5, 1, [8], strides=[4])
    steps.append(acc(21, "read", col, [("n", 10 + i, 1, 2) for i in range(8)]))
    ck = P.ProvenanceChecker()
    for r in steps:
        ck.apply(r)
    assert ck.n_mismatch == 0 and ck.reads == 2


def test_schema_parsing():
    assert P._schema_mutable_names("_C::fused_add_rms_norm(Tensor($0! -> ) input, Tensor($1! -> ) residual, Tensor? weight, float epsilon) -> ()") == ["input", "residual"]
    assert P._schema_mutable_names("aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)") == ["self"]
    assert P._schema_mutable_names("aten::mm(Tensor self, Tensor mat2) -> Tensor") == []
    assert P._schema_arg_names("aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)") == ["self", "src", "non_blocking"]
    assert P._schema_arg_names("_C::rotary_embedding(Tensor positions, Tensor($0! -> ) query, Tensor($1! -> )? key, int head_size) -> ()") == ["positions", "query", "key", "head_size"]


# --- schema-declared write regions ----------------------------------------------------------------------------------


def _declared_regions_before(v):
    """The region rule before stride-ordered runs: the dims of stride != 1 counted as rows for the guard, runs in index order."""
    if v.numel == 0:
        return []
    shape, strides = collapsed_dims(v)
    rows = 1
    for n, st in zip(shape, strides):
        if st != 1:
            rows *= n
    if rows > 20_000:
        lo, hi = byte_extent(v)
        return [(v.alloc, lo, hi)]
    return [(v.alloc, start * v.itemsize, (start + n) * v.itemsize) for start, n in element_runs(v)]


def _bytes(regions):
    return sorted({(al, b) for al, lo, hi in regions for b in range(lo, hi)})


def test_declared_regions():
    wt = V(7, 0, [2048, 3072], strides=[1, 2048])            # aten.t of a contiguous [3072, 2048] weight
    assert P._declared_regions(wt) == [(7, 0, 2 * 2048 * 3072)]
    col = V(5, 1, [8], strides=[4])                          # a column: the holes stay out
    assert P._declared_regions(col) == [(5, 2 * (1 + 4 * i), 2 * (2 + 4 * i)) for i in range(8)]
    assert P._declared_regions(V(3, 0, [8, 4], strides=[0, 1])) == [(3, 0, 8)] * 8      # broadcast rows: one region each, as before
    assert P._declared_regions(V(3, 2, [3, 4], strides=[1, 0])) == [(3, 4, 10)] * 4      # a broadcast inner dim walks outermost
    holes = V(4, 0, [2, 30_000], strides=[1, 4])             # 30,000 runs > DECLARED_RUNS_MAX: the byte extent
    assert P._declared_regions(holes) == [(4, 0, 2 * (4 * 29_999 + 2))]
    assert P._declared_regions(V(6, 0, [0, 4])) == []
    for v in (col, V(9, 3, [4, 6]), V(9, 0, [4, 6], strides=[12, 2]), V(2, 0, [5, 3, 4], strides=[24, 1, 3]),
              V(7, 0, [4, 6], strides=[1, 4]), V(3, 0, [8, 4], strides=[0, 1])):
        assert _bytes(P._declared_regions(v)) == _bytes(_declared_regions_before(v)), v       # the same bytes


def _linear_log(path, t_shape):
    """A linear root (event 1) over a contiguous weight `w`, as vLLM dispatches it: `aten.t(w)` (2), then `aten.mm` (3)."""
    k, n = t_shape
    w, x, y = V(7, 0, [n, k]), V(8, 0, [4, k]), V(9, 0, [4, n])
    wt = V(7, 0, [k, n], strides=[1, k])

    def enter(i, parent, op, schema, args):
        return DispatchEnter(id=i, parent=parent, step=0, phase="forward", op=op, schema=schema, args=args, kwargs={}, stream=0)
    with LogWriter(path) as lw:
        for ev in (enter(1, None, "aten.linear.default", "aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor", [x, w, None]),
                   enter(2, 1, "aten.t.default", "aten::t(Tensor(a) self) -> Tensor(a)", [w]), DispatchExit(id=2, outs=wt),
                   enter(3, 1, "aten.mm.default", "aten::mm(Tensor self, Tensor mat2) -> Tensor", [x, wt]), DispatchExit(id=3, outs=y),
                   DispatchExit(id=1, outs=y)):
            lw.write(ev)
    return 2 * 4 * n                                         # the output's bytes


def test_transposed_weight_is_one_declared_region(tmp_path):
    log = tmp_path / "linear.jsonl.gz"
    ny = _linear_log(log, (2048, 3072))                      # GLM-4.7-Flash's [2048, 3072] aten.t outs: 6,291,456 regions before
    _, sc = P.stream_and_schema_facts(log, {1: [(9, 0, ny), (7, 2, 4)]})
    assert sc["schema_roots"] == 1 and sc["writes_outside_schema"] == 0
    assert sc["declared_unwritten"] == 1                     # the weight, as one region
    _, sc = P.stream_and_schema_facts(log, {1: [(9, 0, ny), (7, 2, 4), (11, 0, 2)]})
    assert sc["writes_outside_schema"] == 1 and sc["outside_examples"][0]["write"] == (11, 0, 2)


def test_transposed_weight_verdicts_unchanged(tmp_path, monkeypatch):
    log = tmp_path / "linear.jsonl.gz"
    ny = _linear_log(log, (4, 6))
    cases = {"output": [(9, 0, ny)], "output and one weight element": [(9, 0, ny), (7, 6, 8)], "a row of the output": [(9, 12, 24)],
             "an undeclared alloc": [(9, 0, ny), (11, 0, 2)], "past the weight": [(7, 48, 50)], "past the output": [(9, ny, ny + 2)]}
    now = {c: P.stream_and_schema_facts(log, {1: ws})[1]["writes_outside_schema"] for c, ws in cases.items()}
    monkeypatch.setattr(P, "_declared_regions", _declared_regions_before)
    before = {c: P.stream_and_schema_facts(log, {1: ws})[1]["writes_outside_schema"] for c, ws in cases.items()}
    assert now == before == {"output": 0, "output and one weight element": 0, "a row of the output": 0, "an undeclared alloc": 1,
                             "past the weight": 1, "past the output": 1}
    # The one verdict that moves: a write over adjacent weight elements, which index order had split into one-element regions.
    span = {1: [(7, 0, 4)]}
    assert P.stream_and_schema_facts(log, span)[1]["writes_outside_schema"] == 1
    monkeypatch.undo()
    assert P.stream_and_schema_facts(log, span)[1]["writes_outside_schema"] == 0


def _writes_as_extents(r):
    """`check`'s journaled write regions before they took the declared side's rule: each write's byte extent."""
    wbe = {}
    for a in r.accesses:
        if a["kind"] == "write" and a["event"] is not None and a["event"] >= 0:
            v, rows = a["view"], a.get("rows")
            for u in ([v] if rows is None else [row_view(v, i) for i in rows]):
                lo, hi = byte_extent(u)
                wbe.setdefault(a["event"], []).append((u.alloc, lo, hi))
    return wbe


def _rope_log(path, t, h):
    """A rotary root (event 1) as MLA calls it: `query` is `q_pe`, per token h heads of 64 at a head stride of 256, after each
    head's 192 `q_nope` elements; `key` is `k_pe`, the last 64 of a 576-wide row."""
    q_pe = V(5, 192, [t, h, 64], strides=[h * 256, 256, 1])
    k_pe = V(6, 512, [t, 1, 64], strides=[576, 64, 1])
    schema = ("_C::rotary_embedding(Tensor positions, Tensor($0! -> ) query, Tensor($1! -> )? key, int head_size, Tensor cos_sin_cache, "
              "bool is_neox, int rope_dim_offset=0, bool inverse=False) -> ()")
    with LogWriter(path) as lw:
        lw.write(DispatchEnter(id=1, parent=None, step=0, phase="forward", op="_C.rotary_embedding.default", schema=schema,
                               args=[V(4, 0, [t], dtype="int64"), q_pe, k_pe, 64, V(3, 0, [64, 64]), False], kwargs={}, stream=0))
        lw.write(DispatchExit(id=1, outs=None))
    return q_pe, k_pe


def test_strided_q_pe_write_is_inside_its_declared_runs(tmp_path):
    t, h = 3, 4
    log = tmp_path / "rope.jsonl.gz"
    q_pe, k_pe = _rope_log(log, t, h)
    accs = [acc(i, "write", row_view(q_pe, i), refs=[("n", 0, i * h * 64, (i + 1) * h * 64)]) for i in range(t)]   # a row per token
    accs.append(acc(t, "write", k_pe, refs=[("n", 1, 0, t * 64)], rows=list(range(t))))
    r = SimpleNamespace(accesses=accs, instances=[])
    rep = P.check(r, log)
    assert rep["status"] == "pass" and rep["schema"]["schema_roots"] == 1
    assert rep["schema"]["writes_outside_schema"] == 0 and rep["schema"]["declared_unwritten"] == 0
    # As its byte extent, each token's q_pe row spans the other heads' q_nope: one write outside per token, as GLM-4.7-Flash had.
    assert P.stream_and_schema_facts(log, _writes_as_extents(r))[1]["writes_outside_schema"] == t
    nope = acc(t + 1, "write", V(5, 0, [192]), refs=[("n", 2, 0, 192)])                                       # into the holes
    rep = P.check(SimpleNamespace(accesses=[*accs, nope], instances=[]), log)
    assert rep["status"] == "fail" and rep["schema"]["writes_outside_schema"] == 1


# --- a real fold --------------------------------------------------------------------------------------------------


def test_toy_fold_journal_is_reproduced():
    if not tfs._real_memory_available():
        tfs.install_stub_memory()
    r = fold(tfs.toy_log(), tfs.toy_profile(), strict=True)
    assert r.accesses and any(a["kind"] == "read" for a in r.accesses)
    rep = P.check(r)
    assert rep["status"] == "pass" and rep["n_mismatch"] == 0 and rep["reads"] > 0, rep
    # plant a wrong wire: the last read points at a different producer
    reads = [a for a in r.accesses if a["kind"] == "read" and access_runs(a["refs"]) and access_runs(a["refs"])[0][0] == "n"]
    victim = reads[-1]
    sp, ix, lo, hi = access_runs(victim["refs"])[0]
    victim["refs"] = [(sp, ix + 1, lo, hi)] + [tuple(x) for x in access_runs(victim["refs"])[1:]]
    rep = P.check(r)
    assert rep["status"] == "fail" and rep["n_mismatch"] == 1 and rep["mismatches"][0]["seq"] == victim["seq"]


@pytest.mark.skipif(not M1.exists(), reason="golden M1 log not present")
def test_m1_exhaustive_provenance():
    from verity_vllm.check.fold_compare import fold_log, load_profile
    r = fold_log(M1, load_profile("vllm_d9105ea80_sm89_eager"), with_program=False)
    rep = P.check(r, M1)
    assert rep["status"] == "pass", rep["failures"]
    assert rep["reads"] > 20000 and rep["n_mismatch"] == 0 and rep["uncovered"] == 0
    assert rep["streams"]["single_stream"] is True
    assert rep["schema"]["writes_outside_schema"] == 0 and rep["schema"]["schema_roots"] > 1000


@pytest.mark.slow
@pytest.mark.parametrize("log, profile", [(M1, "vllm_d9105ea80_sm89_eager"), (M6, "vllm_d9105ea80_sm89_eager_qwen15")], ids=["m1", "m6"])
def test_fixture_wiring_unchanged_by_the_region_rules(log, profile, monkeypatch):
    if not log.exists():
        pytest.skip(f"{log} not present")
    from verity_vllm.check.fold_compare import fold_log, load_profile
    r = fold_log(log, load_profile(profile), with_program=False)
    now = P.check(r, log)
    assert now["status"] == "pass"
    monkeypatch.setattr(P, "_declared_regions", _declared_regions_before)
    assert P.stream_and_schema_facts(log, _writes_as_extents(r)) == (now["streams"], now["schema"])
