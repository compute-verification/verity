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

from verity_vllm.check.replay.opening import CommittedStore
from verity_vllm.commit.binding import identity, identity_dict

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




def test_fixed_committed_store_names_the_ambiguity():
    """The former strict xfail, flipped."""
    bm = _bmap([_leaf("model.norm/out", lo=0, hi=8, stream_offset=0), _leaf("model.norm/out#2", lo=0, hi=8, stream_offset=64)])
    st = CommittedStore(bm, lambda *_: bytes(16))
    w, why = st.words("r0", 0, "model.norm", "out", 0, 8)
    assert w is None and "ambiguous" in why






