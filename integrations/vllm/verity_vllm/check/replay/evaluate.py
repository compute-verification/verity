"""Evaluating one sampled VU: its operands from the committed store and the weights, the registered row kernel
(`verity_vllm.program.kernels.rows`), and the comparison of every member it produces with the committed words.  A VU whose
inputs cannot be resolved is not evaluable by name (`_Unresolved`), never skipped."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from verity_vllm.query.manifest import format as RM
from verity_vllm.check.replay.replay_dump import dump_row_words
from verity_vllm.commit.opened import OpeningNotVerified, opened_check
from verity_vllm.program.kernels.kernel_registry import Declined
from verity_vllm.program.kernels.rows import EVALUATORS, ROWS
from verity_vllm.check.replay.index import IDS_MEMBER, SAMPLER_OP, SEED_PARAM, ProgramIndex, const_word
from verity_vllm.check.replay.opening import _ITEMSIZE, CommittedStore
from verity_vllm.check.replay.compare import _interior_slice, _tp_member_alias, lifted_compare, padded_slot_enable


#: [R16 fp8] a `weights` field's leaf width (its `Value<b>`) -> the word dtype the registered slice is read as (bf16 -> u16, e4m3 -> u8, f32 -> u32)
_NP_BY_BITS = {8: np.uint8, 16: np.uint16, 32: np.uint32}


class _Operands:
    """The row's operands for `rows.ROWS[family].fn`: `inputs[g]` = the committed words of arg group g (one array per slice,
    concatenated in order), `weights[g]` = the registered slice when arg group g is a `weights` slice; built on first use."""

    def __init__(self, inputs: list[list[np.ndarray]], weights: list[np.ndarray | None]) -> None:
        self.inputs, self.weights = inputs, weights

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, g: int) -> np.ndarray:
        if self.weights[g] is not None:
            return self.weights[g]
        return np.concatenate(self.inputs[g]) if len(self.inputs[g]) > 1 else self.inputs[g][0]


#: observers of every row evaluation, interior producers included: tap(family, statics, operands, outputs, weight_refs), weight_refs[g]
#: = {"field", "range"} when operand g is a registered weight slice, else None.  Empty on every record; the VU exporter
#: (`verity_vllm.pipeline.vu_export`) installs one around its own evaluations and removes it after.
EVALUATION_TAPS: list[Callable[..., None]] = []


def evaluate(family: str, statics: dict[str, Any], inputs: list[list[np.ndarray]], weights: list[np.ndarray | None], *,
             refs: list[dict[str, Any] | None] | None = None) -> dict[str, np.ndarray]:
    """F_V for the supported families: the family's row kernel (`verity_vllm.program.kernels.rows`) on the row's operands (`_Operands`).
    Returns {member: words}."""
    k = ROWS.get(family)
    if k is None:
        raise KeyError(family)
    ops = _Operands(inputs, weights)
    out = k.fn(statics, ops)
    for tap in EVALUATION_TAPS:
        tap(family, statics, [ops[g] for g in range(len(ops))], out, refs or [None] * len(ops))
    return out


#: an input of the VU is not available to the replay, or its row kernel declines the instance (partial, never PASS): the message
#: names the slice or the reason
_Unresolved = Declined

INTERIOR_DEPTH = 4     # composed interior producers per VU (Gemm under a BiasAdd = 1); a deeper chain is 'not evaluable', named


def _evaluate_row(pi: ProgramIndex, store: CommittedStore, weights: Callable[[str, int, int], np.ndarray | None], r: dict, rid: str, arrive: int,
                  inputs_rec: list, *, depth: int, interior: dict[int, dict[str, np.ndarray]] | None = None) -> dict[str, np.ndarray]:
    """F_row on the row's resolved inputs: committed member words, registered weight slices, committed prompt ids -- or, for an arg that
    names an INTERIOR producer row (a member-less row such as the pre-bias Gemm of a biased linear, MAN-07, or the Fp8GroupQuant of an
    FP8 linear whose two ports the GEMM reads as two arg slices), that row's OWN evaluation from ITS committed inputs (bounded depth;
    `interior` memoises it per VU).  Raises _Unresolved with the slice named when an input is not available."""
    if interior is None:
        interior = {}
    d = pi.describe(r)
    if d is None:
        raise _Unresolved(f"row {r.get('i')} ({RM.spec_family(r['spec'])}) is unmodelled")
    fam, st = d["family"], d["statics"]
    inputs: list[list[np.ndarray]] = []
    wslices: list[np.ndarray | None] = []
    wrefs: list[dict[str, Any] | None] = []
    for group in r.get("args") or []:
        words: list[np.ndarray] = []
        wsl = None
        wref = None
        for sl in group:
            kind, a, lo, hi = sl[0], int(sl[1]), int(sl[2]), int(sl[3])
            if kind == "p":
                pname = pi.pnames[a] if a < len(pi.pnames) else str(a)
                if pname == "weights":
                    ws = pi.weight_slice(lo, hi)
                    if ws is None:
                        raise _Unresolved(f"weights slice [{lo}, {hi}) spans fields")
                    fname, flo, fhi, _shape = ws
                    arr = weights(fname, flo, fhi)
                    if arr is None:
                        raise _Unresolved(f"registered weight field {fname} not available to the replay")
                    # [R16 fp8] the slice is read in the field's DECLARED leaf width (bf16 u16 / e4m3 u8 / f32 u32): a 16-bit cast of an
                    # f32 block scale or of an e4m3 byte would be a silent corruption of a load-bearing input
                    bits = pi.weight_bits(fname)
                    arr = np.asarray(arr).reshape(-1)
                    if arr.dtype.itemsize * 8 != bits:
                        raise _Unresolved(f"registered weight field {fname} is provided as {arr.dtype} but the Program declares Value<{bits}>")
                    wsl = np.asarray(arr, dtype=_NP_BY_BITS.get(bits, np.uint16)).reshape(-1)
                    inputs_rec.append({"kind": "weights", "field": fname, "range": [flo, fhi], "bits": bits})
                    wref = {"field": fname, "range": [flo, fhi]}
                elif pname in (SEED_PARAM, RM.SPLITS_PARAM):
                    # [R16 stoch, F-r16-20] the sampler's named INPUTS: the committed runner.sampler/seed word of THIS (step, request) and,
                    # under derived S, the committed runner.sampler/splits word of the event the Program's slice names (`splits[s]` = the
                    # request's step s; the identity is per engine step, request-local element 0).  Never the declared / observed value.
                    if hi - lo != 1:
                        raise _Unresolved(f"{pname} slice [{lo}, {hi}) is not one word")
                    step = d["step"] if pname == SEED_PARAM else lo
                    if pname == RM.SPLITS_PARAM and lo != d["step"]:
                        raise _Unresolved(f"splits slice [{lo}, {hi}) names step {lo}, the selector is at step {d['step']}")
                    w, why = store.words(rid, arrive + step, SAMPLER_OP, pname, 0, 1)
                    if w is None:
                        raise _Unresolved(why)
                    words.append(w)
                    inputs_rec.append({"kind": "committed", "identity": [rid, arrive + step, SAMPLER_OP, pname, [0, 1]]})
                else:                                                             # the declared prompt: committed input_token_ids of the arrival step
                    eop = pi.embed_op_path()
                    if eop is None:
                        raise _Unresolved("Program has no Embedding row: prompt ids have no committed identity")
                    w, why = store.words(rid, arrive + 0, eop, IDS_MEMBER, lo, hi)
                    if w is None:
                        raise _Unresolved(why)
                    words.append(w)
                    inputs_rec.append({"kind": "prompt", "identity": [rid, arrive, eop, IDS_MEMBER, [lo, hi]]})
            elif kind == "n":
                m = pi.member_at(a, lo, hi)
                if m is None:
                    pr = pi.by_i.get(a)
                    pd = pi.describe(pr) if pr is not None else None
                    if pd is not None and pd["members"] == [] and pd["family"] in EVALUATORS:      # an interior producer: compose it
                        if depth >= INTERIOR_DEPTH:
                            raise _Unresolved(f"interior chain deeper than {INTERIOR_DEPTH} at row {a} ({pd['family']})")
                        sub_rec: list = []
                        if a in interior:                                              # [R16 fp8] one evaluation per interior row per VU
                            got = interior[a]
                        else:
                            got = interior[a] = _evaluate_row(pi, store, weights, pr, rid, arrive, sub_rec, depth=depth + 1, interior=interior)
                        out = _interior_slice(got, lo, hi)
                        if out is None:
                            raise _Unresolved(f"arg slice ['n', {a}, {lo}, {hi}) exceeds the interior {pd['family']} output "
                                              f"({sum(int(np.asarray(v).size) for v in got.values())} words over {len(got)} member(s))")
                        words.append(out)
                        inputs_rec.append({"kind": "interior", "row": a, "family": pd["family"], "op_path": pd["op_path"], "range": [lo, hi], "inputs": sub_rec})
                        continue
                    cst = const_word(pr) if pr is not None else None
                    if cst is not None:                                                # a Program constant (Const32/64[0x..]_v1: top_p, pos, temperature)
                        if (lo, hi) != (0, 1):
                            raise _Unresolved(f"arg slice ['n', {a}, {lo}, {hi}) of a one-word constant row")
                        words.append(cst)
                        inputs_rec.append({"kind": "const", "row": a, "spec": pr["spec"]})
                        continue
                    raise _Unresolved(f"arg slice ['n', {a}, {lo}, {hi}) is not inside one committed member of row {a}")
                pdsc, member, mlo, mhi, _dt = m
                width = next(int(mm[1]) for mm in pdsc["members"] if mm[0] == member)
                prow = pdsc["row"]
                w, why = store.words(rid, arrive + pdsc["step"], pdsc["op_path"], member, prow * width + mlo, prow * width + mhi)
                if w is None:
                    raise _Unresolved(why)
                words.append(w)
                inputs_rec.append({"kind": "committed", "identity": [rid, arrive + pdsc["step"], pdsc["op_path"], member, [prow * width + mlo, prow * width + mhi]]})
            else:
                raise _Unresolved(f"unknown arg kind {kind!r}")
        inputs.append(words)
        wslices.append(wsl)
        wrefs.append(wref if wsl is not None else None)
    return evaluate(fam, st, inputs, wslices, refs=wrefs)


def replay_vu(pi: ProgramIndex, store: CommittedStore, weights: Callable[[str, int, int], np.ndarray | None], vu: dict, *,
              arrive: int) -> dict[str, Any]:
    """One VU: resolve its inputs to committed words / registered weights, evaluate, compare with its committed outputs (`opened_check`)."""
    return opened_check(getattr(store, "committed", None), lambda rec: _replay_vu(pi, store, weights, vu, arrive, rec))


def _replay_vu(pi: ProgramIndex, store: CommittedStore, weights: Callable[[str, int, int], np.ndarray | None], vu: dict, arrive: int,
               rec: dict[str, Any]) -> dict[str, Any]:
    r = pi.by_i[vu["i"]]
    d = pi.describe(r)
    rid = vu["request_id"]
    fam, st = d["family"], d["statics"]
    rec.update({"i": vu["i"], "spec": r["spec"], "request_id": rid, "step": d["step"], "engine_step": arrive + d["step"], "op_path": d["op_path"],
                "row": d["row"], "phase": vu["phase"], "inputs": []})
    try:
        got = _evaluate_row(pi, store, weights, r, rid, arrive, rec["inputs"], depth=0, interior={})
    except _Unresolved as u:
        rec["why"] = str(u)
        return rec | {"result": None}
    except OpeningNotVerified:
        raise
    except Exception as e:  # noqa: BLE001
        rec["why"] = f"evaluator error: {e!r}"[:300]
        return rec | {"result": None}
    if d.get("plane") and d["plane"] not in got and len(got) == 1:
        got = {d["plane"]: next(iter(got.values()))}                             # [MAN-06f v3] a dense relation's `out` IS this row's plane (act = SiluMul out)
        rec["plane"] = d["plane"]
    # the committed outputs: every member the manifest requires of this row
    outs, mism = [], []
    for m in d["members"]:
        member, width, dtype = m[0], int(m[1]), str(m[2])
        if member not in got:
            # [tp TP-08d] a rank Program's row-parallel Gemm / lm_head shard Gemm: the relation's ONE output is this rank's PARTIAL / SHARD
            # member (part_rank<r> / shard_rank<r>, the collective's INPUT the rank committed) -- the evaluator names it "0"/"out";
            # [MOE-TP-01 / moe] the MoeSum relation names its one output `sum`: on a TP rank that sum of the rank's committed `down`
            # words (the rank's K-sharded expert partials, routed weight applied) IS the MoE block's part_rank<r>, the all-reduce input
            alias = _tp_member_alias(got, member)
            if alias is None and d.get("copies") and len(got) == 1:
                alias = next(iter(got.values()))                                 # [F-r19-int-24] `<experts>/out` = the MoeSum relation's one output
            if alias is not None:
                got[member] = alias
        if member not in got:
            rec["why"] = f"evaluator produced no member {member!r}"
            return rec | {"result": None}
        elo, ehi = d["row"] * width, (d["row"] + 1) * width
        w, why = store.words(rid, arrive + d["step"], d["op_path"], member, elo, ehi)
        if w is None:
            rec["why"] = why
            return rec | {"result": None}
        want = np.asarray(got[member]).reshape(-1)
        have = np.asarray(w).reshape(-1)
        if RM.is_token_select(fam):                                              # TokenSelect_v1 / GumbelTopPTokenSelect_v1: the one sampled id
            eq = int(have[0]) == int(want[0])
            outs.append({"member": member, "committed": int(have[0]), "evaluated": int(want[0]), "equal": eq})
        elif dtype in RM.LIFTED_DTYPES:
            # [revb F-r17b-52 (ii)] a LIFTED member of the padded MoE construction (Q_module_pad_v1): the slot enable a_e is derived from the
            # row's COMMITTED router_ids (never the twin's), the expected words are enc(bot) exactly for a disabled slot and the Active word of
            # the twin's value for an enabled one; the router members are Active in every executed step
            active, why_a = padded_slot_enable(member, got, store, rid, arrive + d["step"], d["op_path"], d["row"], int(st["TOPK"]))
            if active is None:
                rec["why"] = why_a
                return rec | {"result": None}
            if _ITEMSIZE.get(str(why), 0) != RM.LIFTED_DTYPES[dtype][1]:
                # (iii) the plane must be COMMITTED at the lifted word's width (u32 / u64): a bf16 plane whose values happen to equal the Active
                # payloads is a different representation of the identity -- refused by name, never read as Active words
                rec["why"] = (f"{member}: committed as {why!r} ({_ITEMSIZE.get(str(why))} B words), the query requires the lifted word {dtype} "
                              f"({RM.LIFTED_DTYPES[dtype][1]} B: Value<{RM.LIFTED_DTYPES[dtype][0] + 1}>) -- representation mismatch (F-r17b-52 (iii))")
                return rec | {"result": None}
            eq, nd, first, verdict = lifted_compare(have, want, active, dtype)
            outs.append({"member": member, "width": width, "equal": eq, "differing_elements": nd, "first_diff": first, "lifted": dtype,
                         "slot_enable": int(active), "verdict": verdict})
        else:
            if _ITEMSIZE.get(dtype, 2) == 4:                                     # [MAN-06f v3] f32 / i32 plane members (router_w / router_ids): whole words
                have16 = have.astype(np.uint32) if have.dtype != np.uint32 else have
                want = np.asarray(want, dtype=np.uint32)
            else:
                have16 = have.astype(np.uint16) if have.dtype != np.uint16 else have
            eq = have16.shape == want.shape and bool(np.array_equal(have16, want))
            nd = int(np.count_nonzero(have16 != want)) if have16.shape == want.shape else -1
            outs.append({"member": member, "width": width, "equal": eq, "differing_elements": nd,
                         "first_diff": (int(np.flatnonzero(have16 != want)[0]) if (not eq and nd > 0) else None)})
        if not eq:
            mism.append(member)
        dump_row_words(fam, rid, arrive + d["step"], d["op_path"], d["row"], member, have, want, eq, store=store, elo=elo, ehi=ehi)
    for c in d.get("copies") or []:
        # [MOE-TP-02] the same output committed under its second required name (the TP rank's moe/L<k>/sum plane beside part_rank<r>):
        # read back at the plane address and compared against the SAME evaluated words -- a plane copy that disagrees is a mismatch by name
        want = np.asarray(got[c["of"]]).reshape(-1)
        elo, ehi = int(c["row"]) * int(c["width"]), (int(c["row"]) + 1) * int(c["width"])
        w, why = store.words(rid, arrive + d["step"], c["op_path"], c["member"], elo, ehi)
        label = f"{c['op_path']}/{c['member']} (copy of {c['of']})"
        if w is None:
            rec["why"] = why
            return rec | {"result": None}
        have = np.asarray(w).reshape(-1)
        if _ITEMSIZE.get(str(c["dtype"]), 2) == 4:                              # [revb n-moe-tp02-1] the primary path's word-width switch, mirrored
            have16 = have.astype(np.uint32) if have.dtype != np.uint32 else have
            want = np.asarray(want, dtype=np.uint32)
        else:
            have16 = have.astype(np.uint16) if have.dtype != np.uint16 else have
        eq = have16.shape == want.shape and bool(np.array_equal(have16, want))
        nd = int(np.count_nonzero(have16 != want)) if have16.shape == want.shape else -1
        outs.append({"member": label, "width": int(c["width"]), "equal": eq, "differing_elements": nd,
                     "first_diff": (int(np.flatnonzero(have16 != want)[0]) if (not eq and nd > 0) else None)})
        if not eq:
            mism.append(label)
        dump_row_words(fam, rid, arrive + d["step"], c["op_path"], int(c["row"]), c["member"], have, want, eq, store=store, elo=elo, ehi=ehi)
    rec["outputs"] = outs
    rec["result"] = not mism
    if mism:
        rec["why"] = f"{fam} {rid} step {d['step']} {d['op_path']}/{','.join(mism)} row {d['row']}: F_V(committed inputs) != committed output"
    return rec
