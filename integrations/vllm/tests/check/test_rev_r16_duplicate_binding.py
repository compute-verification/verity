"""One identity, TWO leaf bindings: coverage passes, the replay silently reads the first.

"every required identity has ONE unambiguous binding".  `binding.coverage_check` enforces ">= 1" (duplicates of ANOTHER
identity cannot hide a missing one -- that part holds) and only REPORTS `physical_copies_n`; `opening.CommittedStore` keys the
committed words by (request, step, op_path, member) and keeps the first entry (min invocation), so a second binding of the SAME identity
to a different buffer is never read by form (A) and is not a named failure anywhere on the (A) path.  Honest records have 0 copies
(153/153 coverage records), so gating on it costs nothing.

This file documented the hole with two plain tests + two strict xfails; the fix flips them: `coverage_check` counts a
required identity bound by >1 covering leaf entry as AMBIGUOUS (`ambiguous_n`, `unambiguous`, first 12 named with where each copy lives)
and `ok` requires `ambiguous_n == 0`; `CommittedStore` keeps every entry of a key and `words` refuses by name when more than one entry of
the lowest invocation covers the requested slice.  The honest shapes (one entry per identity; chunked entries with DISJOINT ranges; a
duplicate under a NON-required identity = extra, reported) stay `ok`.
"""
from __future__ import annotations

import pytest

from verity_vllm.commit.binding import coverage_check, identity, identity_dict
from verity_vllm.check.replay.opening import CommittedStore

PD = "ab" * 32


def _leaf(name: str, *, lo: int, hi: int, stream_offset: int, invocation: int = 0, dtype: str = "bfloat16") -> dict:
    return {"identity": identity_dict(identity(PD, "r0", 0, invocation, "model.norm", "out", lo, hi, 0)), "kind": "leaf",
            "step_root": "00" * 32, "layout": "v2b", "chunk": 64, "leaf_lo": stream_offset // 64, "leaf_hi": stream_offset // 64 + 1,
            "stream_offset": stream_offset, "width_bytes": (hi - lo) * 2, "encoding": {"dtype": dtype, "shape": [hi - lo]},
            "element_offset": 0, "tensor_ordinal": 0, "name": name}


def _bmap(entries: list[dict]) -> dict:
    return {"digest": "11" * 32, "entries_n": len(entries), "entries": entries}


def _population():
    return [identity(PD, "r0", 0, 0, "model.norm", "out", 0, 8, 0)]


def test_two_bindings_of_one_required_identity_are_ambiguous_and_coverage_is_not_ok():
    bm = _bmap([_leaf("model.norm/out", lo=0, hi=8, stream_offset=0), _leaf("model.norm/out#2", lo=0, hi=8, stream_offset=64)])
    cov = coverage_check(_population(), bm)
    assert cov["ok"] is False and cov["bound_n"] == 1 and cov["missing_n"] == 0 and cov["physical_copies_n"] == 1
    assert cov["ambiguous_n"] == 1 and cov["unambiguous"] is False
    a = cov["ambiguous"][0]
    assert a["op_path"] == "model.norm" and a["bindings_n"] == 2 and a["bound_at"] == ["model.norm/out@0[0,1)", "model.norm/out#2@64[1,2)"]


def test_committed_store_no_longer_reads_the_first_binding_silently():
    """The attack: honest bytes first, arbitrary bytes second under the same identity -- neither copy is read; the VU fails by name."""
    honest = bytes(range(16))
    forged = bytes(16)
    stream = {0: honest, 64: forged}

    def committed(step: int, name: str, lo: int, hi: int):
        base = 0 if name == "model.norm/out" else 64
        return stream[base][lo:hi]

    bm = _bmap([_leaf("model.norm/out", lo=0, hi=8, stream_offset=0), _leaf("model.norm/out#2", lo=0, hi=8, stream_offset=64)])
    st = CommittedStore(bm, committed)
    w, why = st.words("r0", 0, "model.norm", "out", 0, 8)
    assert w is None and why.startswith("ambiguous binding: 2 entries for (r0, step 0, model.norm/out) [invocation 0] cover [0, 8)")
    assert "model.norm/out@0" in why and "model.norm/out#2@64" in why and "one unambiguous binding per required identity" in why
    # the order of the entries does not matter (forged first): still refused, never 'first wins'
    bm2 = _bmap([_leaf("model.norm/out#2", lo=0, hi=8, stream_offset=64), _leaf("model.norm/out", lo=0, hi=8, stream_offset=0)])
    w, why = CommittedStore(bm2, committed).words("r0", 0, "model.norm", "out", 2, 6)
    assert w is None and "ambiguous binding" in why
    # one binding: read as before
    w, why = CommittedStore(_bmap([_leaf("model.norm/out", lo=0, hi=8, stream_offset=0)]), committed).words("r0", 0, "model.norm", "out", 0, 8)
    assert w is not None and w.tobytes() == honest and why == "bfloat16"


def test_fixed_coverage_refuses_an_ambiguous_binding():
    """The former strict xfail, flipped."""
    bm = _bmap([_leaf("model.norm/out", lo=0, hi=8, stream_offset=0), _leaf("model.norm/out#2", lo=0, hi=8, stream_offset=64)])
    cov = coverage_check(_population(), bm)
    assert cov["ok"] is False and cov.get("unambiguous") is False


def test_fixed_committed_store_names_the_ambiguity():
    """The former strict xfail, flipped."""
    bm = _bmap([_leaf("model.norm/out", lo=0, hi=8, stream_offset=0), _leaf("model.norm/out#2", lo=0, hi=8, stream_offset=64)])
    st = CommittedStore(bm, lambda *_: bytes(16))
    w, why = st.words("r0", 0, "model.norm", "out", 0, 8)
    assert w is None and "ambiguous" in why


def test_ambiguity_is_gated_on_required_identities_and_the_honest_shapes_stay_ok():
    """(a) a duplicate under a NON-required identity is an extra (reported), not ambiguity; (b) chunked entries with DISJOINT ranges under
    one key are one binding per element -- coverage of a slice inside one chunk is unambiguous and the store reads the covering chunk;
    (c) a member `*` requirement is ambiguous when any member of that op is bound twice; (d) a wider and a narrower entry both covering the
    required range are still two bindings of it."""
    # (a)
    bm = _bmap([_leaf("model.norm/out", lo=0, hi=8, stream_offset=0),
                {**_leaf("x/out", lo=0, hi=8, stream_offset=64), "identity": identity_dict(identity(PD, "r0", 0, 0, "x", "out", 0, 8, 0))},
                {**_leaf("x/out#2", lo=0, hi=8, stream_offset=128), "identity": identity_dict(identity(PD, "r0", 0, 0, "x", "out", 0, 8, 0))}])
    cov = coverage_check(_population(), bm)
    assert cov["ok"] is True and cov["ambiguous_n"] == 0 and cov["unambiguous"] is True and cov["physical_copies_n"] == 1 and cov["extra_n"] == 1
    # (b)
    chunks = _bmap([_leaf("model.norm/out", lo=0, hi=4, stream_offset=0), _leaf("model.norm/out.1", lo=4, hi=8, stream_offset=64)])
    cov = coverage_check([identity(PD, "r0", 0, 0, "model.norm", "out", 0, 4, 0), identity(PD, "r0", 0, 0, "model.norm", "out", 4, 8, 0)], chunks)
    assert cov["ok"] is True and cov["ambiguous_n"] == 0 and cov["bound_n"] == 2
    seen = []

    def committed(step, name, lo, hi):
        seen.append(name)
        return bytes(range(16))[lo:hi]

    w, why = CommittedStore(chunks, committed).words("r0", 0, "model.norm", "out", 4, 8)
    assert w is not None and seen == ["model.norm/out.1"] and why == "bfloat16"
    w, why = CommittedStore(chunks, committed).words("r0", 0, "model.norm", "out", 2, 6)          # spans both chunks: no single covering entry
    assert w is None and "outside the bound identity range" in why
    # (c)
    star = [identity(PD, "r0", 0, 0, "model.norm", "*", 0, -1, 0)]
    assert coverage_check(star, _bmap([_leaf("model.norm/out", lo=0, hi=8, stream_offset=0)]))["ok"] is True
    cov = coverage_check(star, _bmap([_leaf("model.norm/out", lo=0, hi=8, stream_offset=0), _leaf("model.norm/out#2", lo=0, hi=8, stream_offset=64)]))
    assert cov["ok"] is False and cov["ambiguous_n"] == 1
    # (d)
    cov = coverage_check(_population(), _bmap([_leaf("model.norm/out", lo=0, hi=8, stream_offset=0), _leaf("model.norm/out.w", lo=0, hi=16, stream_offset=64)]))
    assert cov["ok"] is False and cov["ambiguous_n"] == 1 and cov["ambiguous"][0]["bindings_n"] == 2


def test_duplicate_of_another_identity_does_not_hide_a_missing_one():
    """The half of the C1 rule that DOES hold today (kept as a regression)."""
    bm = _bmap([_leaf("model.norm/out", lo=0, hi=8, stream_offset=0), _leaf("model.norm/out#2", lo=0, hi=8, stream_offset=64)])
    pop = _population() + [identity(PD, "r0", 0, 0, "model.embed_tokens", "out", 0, 8, 0)]
    cov = coverage_check(pop, bm)
    assert cov["ok"] is False and cov["missing_n"] == 1 and cov["missing"][0]["op_path"] == "model.embed_tokens"


def test_commit_verdict_names_the_ambiguous_binding():
    """The row verdict reads coverage.ok: an ambiguous binding (all bound, none missing) FAILs with the C1 text naming the identity + copies."""
    from verity_vllm.check.commit_pass import commit_verdict
    bm = _bmap([_leaf("model.norm/out", lo=0, hi=8, stream_offset=0), _leaf("model.norm/out#2", lo=0, hi=8, stream_offset=64)])
    cov = coverage_check(_population(), bm)
    rec = {"extra": {"expected_tensors": 3, "committed_tensors": 3, "errors_n": 0}, "openings": {"n": 64, "verified": 64},
           "binding": {"context": {"ok": True}, "coverage": cov, "openings_population": {"skipped": "test", "unbound_n": 0}, "population": {"requests": 1}}}
    ok, why = commit_verdict(rec, True, min_openings=64)
    assert not ok and why.startswith("identity coverage: 1 of 1 required identities have more than one binding (C1 requires one unambiguous binding; first: step 0 model.norm/out inv 0 req r0 rank 0 bound 2 x at ['model.norm/out@0[0,1)', 'model.norm/out#2@64[1,2)'])"), why
    ok_cov = coverage_check(_population(), _bmap([_leaf("model.norm/out", lo=0, hi=8, stream_offset=0)]))
    assert ok_cov["ok"] is True and ok_cov["unambiguous"] is True
