"""`binding.coverage_check`'s member-`*` path (the hidden family: one `*` identity per request x step x attention layer) must not
scan the WHOLE map index for every `*` identity: O(|*| x |map|) is 16,512 x 111,239 at B8 llama32-1b 1024/128 (minutes) and
132,096 x ~890k at B64 (more than 13 min at 100% CPU on the serving thread AFTER the instrumented run).  It builds one op-level key
index (`_op_key_index`) and looks each `*` identity up in O(1); these tests pin that the verdict, the counts and the named lists are those of
the scan on a B64-shaped fixture with every outcome present (bound, missing, ambiguous, address-mismatch, extra), and that the B64 shape,
hours for the old scan, is counted exactly."""
from __future__ import annotations

import pytest

from verity_vllm.commit.binding import (
    _op_key_index,
    coverage_check,
    identity,
    identity_dict,
    map_index,
)

PD = "cd" * 32
HID = "fa2_hidden_m1"


def _leaf(rid: str, step: int, op: str, mem: str, *, lo: int = 0, hi: int = 8, off: int = 0, inv: int = 0, rank: int = 0, ordinal: int = 0) -> dict:
    return {"identity": identity_dict(identity(PD, rid, step, inv, op, mem, lo, hi, rank)), "kind": "leaf", "step_root": "00" * 32,
            "layout": "v2b", "chunk": 64, "leaf_lo": ordinal, "leaf_hi": ordinal + 1, "stream_offset": ordinal * 64, "width_bytes": (hi - lo) * 2,
            "encoding": {"dtype": "bfloat16", "shape": [hi - lo]}, "element_offset": off, "tensor_ordinal": ordinal, "name": f"{op}/{mem}"}


def _fixture(b: int, steps: int, layers: int, *, planes: int = 3):
    """B requests x steps x layers: a `*` identity per (request, step, layer) for the hidden family + one `out` member per (request, step)
    of a dense module; the map carries `planes` hidden members per layer.  Perturbations: request r1 step 1 layer 0 has NO hidden entry
    (missing); r2 step 0 layer 1 carries plane 0 TWICE (ambiguous); r3 step 0 dense `out` bound at the wrong element range (address
    mismatch); a stray entry under an unrequired step (extra)."""
    pop: list[tuple] = []
    entries: list[dict] = []
    k = 0
    for i in range(b):
        rid = f"r{i}"
        for st in range(steps):
            if rid == "r3" and st == 0:
                pop.append(identity(PD, rid, st, 0, "model.norm", "out", 0, 8, 0))                                # required [0, 8)
                entries.append(_leaf(rid, st, "model.norm", "out", lo=8, hi=16, off=0, ordinal=k)); k += 1      # bound at [8, 16): mismatch
            else:
                pop.append(identity(PD, rid, st, 0, "model.norm", "out", 0, -1, 0))
                entries.append(_leaf(rid, st, "model.norm", "out", lo=0, hi=-1, ordinal=k)); k += 1
            for L in range(layers):
                op = f"{HID}/L{L}"
                pop.append(identity(PD, rid, st, 0, op, "*", 0, -1, 0))
                if rid == "r1" and st == 1 and L == 0:
                    continue                                                                                    # missing
                for p in range(planes):
                    entries.append(_leaf(rid, st, op, f"g{p}", ordinal=k)); k += 1
                if rid == "r2" and st == 0 and L == 1:
                    entries.append(_leaf(rid, st, op, "g0", ordinal=k)); k += 1                                  # second binding: ambiguous
    entries.append(_leaf("r0", steps + 7, "model.norm", "out", lo=0, hi=-1, ordinal=k))                            # extra (unrequired step)
    return pop, {"digest": "22" * 32, "entries_n": len(entries), "entries": entries}


def _reference_star_keys(idx, rid, st, inv, op, rk):
    """The pre-fix scan, kept here as the oracle for the index."""
    return [k for k in idx if k[0] == rid and k[1] == st and k[2] == inv and k[3] == op and k[5] == rk]


def test_op_key_index_equals_the_scan_for_every_op_in_map_order():
    _pop, bm = _fixture(4, 3, 2)
    idx = map_index(bm)
    ok = _op_key_index(idx)
    ops = {(k[0], k[1], k[2], k[3], k[5]) for k in idx}
    assert set(ok) == ops
    for (rid, st, inv, op, rk) in ops:
        assert ok[(rid, st, inv, op, rk)] == _reference_star_keys(idx, rid, st, inv, op, rk)
    assert ok.get(("r9", 0, 0, f"{HID}/L0", 0), []) == []


def test_coverage_verdict_and_named_lists_on_every_outcome():
    pop, bm = _fixture(4, 3, 2)
    cov = coverage_check(pop, bm)
    n_star = 4 * 3 * 2
    n_dense = 4 * 3
    assert cov["expected_n"] == n_star + n_dense
    assert cov["missing_n"] == 2                       # r1/1/L0 (no entry) + r3/0 norm (address mismatch counts as unbound)
    assert cov["address_mismatch_n"] == 1
    assert cov["ambiguous_n"] == 1 and cov["ambiguous"][0]["op_path"] == f"{HID}/L1" and cov["ambiguous"][0]["request_id"] == "r2"
    assert cov["ambiguous"][0]["bindings_n"] == 2
    assert cov["extra_n"] == 2                          # the unrequired-step entry + the mismatched-range entry (unused = extra, as before)
    assert sorted((e[0], e[1]) for e in cov["extra"]) == [("r0", 3 + 7), ("r3", 0)]
    assert cov["physical_copies_n"] == 1
    assert cov["ok"] is False
    miss = {(m["request_id"], m["step"], m["op_path"]) for m in cov["missing"]}
    assert miss == {("r1", 1, f"{HID}/L0"), ("r3", 0, "model.norm")}


def test_clean_fixture_is_ok_and_every_star_member_counts_as_used():
    pop, bm = _fixture(2, 2, 2)
    # drop the perturbations: keep only r0 (no perturbation touches r0 within the required steps) -> restrict population and map to r0
    pop = [p for p in pop if p[1] == "r0"]
    bm["entries"] = [e for e in bm["entries"] if e["identity"]["request_id"] == "r0" and e["identity"]["step"] < 2]
    bm["entries_n"] = len(bm["entries"])
    cov = coverage_check(pop, bm)
    assert cov["ok"] is True and cov["missing_n"] == 0 and cov["ambiguous_n"] == 0
    assert cov["extra_n"] == 0                          # every plane member under a `*` identity is USED (not reported extra)


@pytest.mark.slow
def test_b64_shape_is_counted_exactly():
    """The B64 shape: 64 requests x 129 steps x 16 layers = 132,096 `*` identities against ~900k map entries (3 planes + 1 dense here:
    64 x 129 x (16 x 3 + 1) = 404,544 entries).  Pre-fix this scan was ~1.2e11 tuple compares (hours); the index makes it linear."""
    pop, bm = _fixture(64, 129, 16)
    cov = coverage_check(pop, bm)
    assert cov["expected_n"] == 64 * 129 * 17 and cov["missing_n"] == 2 and cov["ambiguous_n"] == 1
