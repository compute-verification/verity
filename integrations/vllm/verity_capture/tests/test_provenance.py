"""The exhaustive operand-provenance check (G5, D82): the independent last-writer map reproduces every read of a
consistent fold, and detects a wrong wiring / a stale generation / a partial clobber when one is planted."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from verity_capture import provenance as P
from verity_capture.events import ViewDesc
from verity_capture.fold import fold
from verity_capture.tests import test_fold_synthetic as tfs

M1 = Path(__file__).resolve().parents[2] / "data/logs/m1.jsonl.gz"


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


# --- a real fold --------------------------------------------------------------------------------------------------


def test_toy_fold_journal_is_reproduced():
    if not tfs._real_memory_available():
        tfs.install_stub_memory()
    r = fold(tfs.toy_log(), tfs.toy_profile(), strict=True)
    assert r.accesses and any(a["kind"] == "read" for a in r.accesses)
    rep = P.check(r)
    assert rep["status"] == "pass" and rep["n_mismatch"] == 0 and rep["reads"] > 0, rep
    # plant a wrong wire: the last read points at a different producer
    reads = [a for a in r.accesses if a["kind"] == "read" and a["refs"] and a["refs"][0][0] == "n"]
    victim = reads[-1]
    sp, ix, lo, hi = victim["refs"][0]
    victim["refs"] = [(sp, ix + 1, lo, hi)] + [tuple(x) for x in victim["refs"][1:]]
    rep = P.check(r)
    assert rep["status"] == "fail" and rep["n_mismatch"] == 1 and rep["mismatches"][0]["seq"] == victim["seq"]


@pytest.mark.skipif(not M1.exists(), reason="golden M1 log not present")
def test_m1_exhaustive_provenance():
    from verity_capture.compare import fold_log, load_profile
    r = fold_log(M1, load_profile("vllm_d9105ea80_sm89_eager"), with_program=False)
    rep = P.check(r, M1)
    assert rep["status"] == "pass", rep["failures"]
    assert rep["reads"] > 20000 and rep["n_mismatch"] == 0 and rep["uncovered"] == 0
    assert rep["streams"]["single_stream"] is True
    assert rep["schema"]["writes_outside_schema"] == 0 and rep["schema"]["schema_roots"] > 1000
