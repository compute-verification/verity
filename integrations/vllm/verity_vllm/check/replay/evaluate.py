"""Evaluating one sampled VU: its operands from the committed store and the weights, the registered row kernel
(`verity.ml.definitions.numerics.rows`), and the comparison of every member it produces with the committed words.  A VU whose
inputs cannot be resolved is not evaluable by name (`_Unresolved`), never skipped."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from verity_vllm.query.manifest import format as RM
from verity_vllm.check.replay.replay_dump import dump_row_words
from verity_vllm.commit.opened import OpeningNotVerified, opened_check
from verity.ml.definitions.numerics.kernel_registry import Declined
from verity.ml.definitions.numerics.rows import EVALUATORS, ROWS
from verity_vllm.check.replay.index import IDS_MEMBER, SAMPLER_OP, SEED_PARAM, ProgramIndex, const_word, row_args
from verity_vllm.check.replay.opening import _ITEMSIZE, CommittedStore
from verity_vllm.check.replay.compare import _interior_slice, _tp_member_alias, lifted_compare, padded_slot_enable
from verity_vllm.check.replay.delegate import DELEGATES, Place
from verity_vllm.program.compact import is_file_prog


#: a `weights` field's leaf width (its `Value<b>`) -> the word dtype the registered slice is read as (bf16 -> u16, e4m3 -> u8, f32 -> u32)
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

#: the evaluation of every row on another IR, the last one installed: evaluator(pi, row, describe(row), operands) -> {member or port:
#: words}, with optional ``prewarm(picked)`` (run in the parent before the replay forks), ``seconds_per_vu`` (the pool's time per
#: pick) and ``peak_bytes(vu)`` (the bytes one pick's evaluation allocates, which `driver.fit_pool` fits the pool to).  Empty on every
#: record; the Boolean re-verification (`verity_vllm.pipeline.boolean_replay`) installs one around its replay.
ROW_EVALUATORS: list[Callable[..., dict[str, np.ndarray]]] = []


def evaluate(family: str, statics: dict[str, Any], inputs: list[list[np.ndarray]], weights: list[np.ndarray | None], *,
             refs: list[dict[str, Any] | None] | None = None) -> dict[str, np.ndarray]:
    """F_V for the supported families: the family's row kernel (`verity.ml.definitions.numerics.rows`) on the row's operands (`_Operands`).
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


def _weight_words(pi: ProgramIndex, weights: Callable[[str, int, int], np.ndarray | None], fname: str, flo: int, fhi: int) -> tuple[np.ndarray, int]:
    """The registered words [flo, fhi) of `weights` field `fname` in the field's declared leaf width, and that width."""
    arr = weights(fname, flo, fhi)
    if arr is None:
        raise _Unresolved(f"registered weight field {fname} not available to the replay")
    # the slice is read in the field's DECLARED leaf width (bf16 u16 / e4m3 u8 / f32 u32): a 16-bit cast of an
    # f32 block scale or of an e4m3 byte would be a silent corruption of a load-bearing input
    bits = pi.weight_bits(fname)
    arr = np.asarray(arr).reshape(-1)
    if arr.dtype.itemsize * 8 != bits:
        raise _Unresolved(f"registered weight field {fname} is provided as {arr.dtype} but the Program declares Value<{bits}>")
    return np.asarray(arr, dtype=_NP_BY_BITS.get(bits, np.uint16)).reshape(-1), bits


def _weight_runs(pi: ProgramIndex, group: list) -> tuple[np.ndarray, np.ndarray] | None:
    """A `weights` operand of more than one run as its runs' (lo, hi), in operand order, straight from the stored items: a strided
    view's "P" progressions (GLM-5.3's MLA-absorb W_UK_T / W_UV reads, ~1e6 runs in ~2048 progressions, never expanded) or several
    "p" runs.  None for an operand of one run or one that reads anything but `weights`."""
    if not any(it and it[0] in ("p", "P") for it in group) or "weights" not in pi.pnames:
        return None
    wp = pi.pnames.index("weights")
    los: list[np.ndarray] = []
    his: list[np.ndarray] = []
    lo1: list[int] = []
    hi1: list[int] = []
    for it in group:
        if it[0] == "p" and int(it[1]) == wp:
            lo1.append(int(it[2]))
            hi1.append(int(it[3]))
        elif is_file_prog(it) and it[0] == "P" and int(it[1]) == wp and int(it[4]) == 0:
            if lo1:
                los.append(np.asarray(lo1, dtype=np.int64))
                his.append(np.asarray(hi1, dtype=np.int64))
                lo1, hi1 = [], []
            k = np.arange(int(it[7]), dtype=np.int64)
            los.append(int(it[2]) + k * int(it[5]))
            his.append(int(it[3]) + k * int(it[6]))
        else:
            return None
    if lo1:
        los.append(np.asarray(lo1, dtype=np.int64))
        his.append(np.asarray(hi1, dtype=np.int64))
    lo, hi = np.concatenate(los), np.concatenate(his)
    return (lo, hi) if lo.size > 1 else None


def _weight_gather(pi: ProgramIndex, weights: Callable[[str, int, int], np.ndarray | None], lo: np.ndarray, hi: np.ndarray,
                   inputs_rec: list) -> tuple[np.ndarray, dict[str, Any]]:
    """The leaves of runs [lo, hi) of the `weights` struct, in order: one read of the span of the one field they lie in, then a gather.
    Recorded once, the field and span with the run and leaf counts (the row's stored args name every leaf)."""
    n = hi - lo
    if (n < 0).any() or not n.any():
        raise _Unresolved(f"weights operand of {lo.size} runs reads {int(n.clip(0).sum())} leaves (a run with hi < lo, or none)")
    idx = np.repeat(lo - (np.cumsum(n) - n), n) + np.arange(int(n.sum()), dtype=np.int64)
    mn, mx = int(idx.min()), int(idx.max()) + 1
    ws = pi.weight_slice(mn, mx)
    if ws is None:
        raise _Unresolved(f"weights operand of {lo.size} runs over [{mn}, {mx}) spans fields")
    fname, flo, fhi, _shape = ws
    words, bits = _weight_words(pi, weights, fname, flo, fhi)
    gather = {"runs": int(lo.size), "leaves": int(idx.size)}
    inputs_rec.append({"kind": "weights", "field": fname, "range": [flo, fhi], "bits": bits, "gather": gather})
    return words[idx - mn], {"field": fname, "range": [flo, fhi], "gather": gather}


def _cached_prefix_words(pi: ProgramIndex, store: CommittedStore, rid: str, pname: str, lo: int, hi: int, inputs_rec: list) -> list[np.ndarray]:
    """A warm request's cached rows [lo, hi) of the kv-prefix input: what its producing requests of record committed."""
    src = pi.cached_prefix.get(rid)
    if src is None:
        raise _Unresolved(f"{pname} slice [{lo}, {hi}): no producing requests of record for {rid}'s cached rows")
    out = []
    for q, qstep, qop, qmem, qlo, qhi in src.identities(lo, hi):
        w, why = store.words(q, qstep, qop, qmem, qlo, qhi)
        if w is None:
            raise _Unresolved(why)
        out.append(w)
        inputs_rec.append({"kind": "cached", "identity": [q, qstep, qop, qmem, [qlo, qhi]]})
    return out


def _schedule_witness(pi: ProgramIndex, store: CommittedStore, r: dict, d: dict, rid: str, arrive: int, pname: str, lo: int, hi: int,
                      inputs_rec: list) -> np.ndarray:
    """The witness keys of a MoeAlignRanks_v1 row (`format.SCHEDULE_PARAM`): its OWN committed rank, so the row reproduces that rank
    exactly when it ranks each expert's pairs."""
    fam = d["family"]
    own = pi.member_of_port(int(r["i"])).get("0") if fam in RM.SCHEDULE_READERS else None
    width = next((int(m[1]) for m in d["members"] if m[0] == own), None)
    if width is None or width != hi - lo:
        raise _Unresolved(f"{pname} slice [{lo}, {hi}) read by {fam}: only a {sorted(RM.SCHEDULE_READERS)} row's own "
                          f"committed rank (port 0, one key per pair) binds it")
    at = d["row"] * width
    w, why = store.words(rid, arrive + d["step"], d["op_path"], own, at, at + width)
    if w is None:
        raise _Unresolved(why)
    inputs_rec.append({"kind": "witness", "param": pname, "range": [lo, hi],
                       "identity": [rid, arrive + d["step"], d["op_path"], own, [at, at + width]]})
    return w


def _evaluate_row(pi: ProgramIndex, store: CommittedStore, weights: Callable[[str, int, int], np.ndarray | None], r: dict, rid: str, arrive: int,
                  inputs_rec: list, *, depth: int, interior: dict[int, dict[str, np.ndarray]] | None = None,
                  row: int | None = None, plan: bool = False) -> dict[str, np.ndarray]:
    """F_row on the row's resolved inputs: committed member words, registered weight slices, committed prompt ids -- or, for an arg that
    names an INTERIOR producer row (a member-less row such as the pre-bias Gemm of a biased linear, or the Fp8GroupQuant of an
    FP8 linear whose two ports the GEMM reads as two arg slices), that row's OWN evaluation from ITS committed inputs (bounded depth;
    `interior` memoises it per VU).  Raises _Unresolved with the slice named when an input is not available.  ``plan``: every committed
    input is read and nothing is evaluated (no weights, no relation; `plan_vu`)."""
    if interior is None:
        interior = {}
    d = pi.describe(r)
    if d is None:
        raise _Unresolved(f"row {r.get('i')} ({RM.spec_family(r['spec'])}) is unmodelled")
    fam, st = d["family"], d["statics"]
    inputs: list[list[np.ndarray]] = []
    wslices: list[np.ndarray | None] = []
    wrefs: list[dict[str, Any] | None] = []
    for stored in r.get("args") or []:
        runs = _weight_runs(pi, stored)
        if runs is not None:
            wsl, wref = (None, None) if plan else _weight_gather(pi, weights, *runs, inputs_rec)
            inputs.append([])
            wslices.append(wsl)
            wrefs.append(wref)
            continue
        (group,) = row_args({"args": [stored]})
        words: list[np.ndarray] = []
        wsl = None
        wref = None
        for sl in group:
            kind, a, lo, hi = sl[0], int(sl[1]), int(sl[2]), int(sl[3])
            if kind == "p":
                pname = pi.pnames[a] if a < len(pi.pnames) else str(a)
                if pname == "weights":
                    if plan:
                        continue
                    ws = pi.weight_slice(lo, hi)
                    if ws is None:
                        raise _Unresolved(f"weights slice [{lo}, {hi}) spans fields")
                    fname, flo, fhi, _shape = ws
                    wsl, bits = _weight_words(pi, weights, fname, flo, fhi)
                    inputs_rec.append({"kind": "weights", "field": fname, "range": [flo, fhi], "bits": bits})
                    wref = {"field": fname, "range": [flo, fhi]}
                elif a in pi.supplied:
                    binding, fields = pi.supplied[a]
                    f = fields.get(lo)
                    if f is None or hi - lo != 1:
                        raise _Unresolved(f"{pname} slice [{lo}, {hi}) is not one field of the supplied layout")
                    if binding != "registration":
                        raise _Unresolved(f"run value {pname}.{f['name']}: the run Record's values are not read by the replay yet")
                    # the registered value as the build's layout records it: what registration binds, never a committed word
                    words.append(np.array([f["bits"]], dtype=_NP_BY_BITS[int(f["width"])] if int(f["width"]) in _NP_BY_BITS else np.uint64))
                    inputs_rec.append({"kind": "registered", "param": pname, "field": f["name"], "bits": f"0x{f['bits']:x}"})
                elif pname in (SEED_PARAM, RM.SPLITS_PARAM):
                    # the sampler's named INPUTS: the committed runner.sampler/seed word of THIS (step, request) and,
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
                elif pi.kv_prefix and a == pi.kv_prefix["input"]:
                    words.extend(_cached_prefix_words(pi, store, rid, pname, lo, hi, inputs_rec))
                elif pname == RM.SCHEDULE_PARAM:
                    words.append(_schedule_witness(pi, store, r, d, rid, arrive, pname, lo, hi, inputs_rec))
                else:                                                             # the declared prompt: committed input_token_ids of the arrival step
                    eop = pi.embed_op_path()
                    if eop is None:
                        raise _Unresolved("Program has no Embedding row: prompt ids have no committed identity")
                    c = int(pi.kv_prefix["C"]) if pi.kv_prefix else 0          # a warm request's arrival step embeds prompt[C:]
                    if lo < c:
                        raise _Unresolved(f"prompt slice [{lo}, {hi}) reads a row the prefix cache served (C = {c}): no committed input id")
                    w, why = store.words(rid, arrive + 0, eop, IDS_MEMBER, lo - c, hi - c)
                    if w is None:
                        raise _Unresolved(why)
                    words.append(w)
                    inputs_rec.append({"kind": "prompt", "identity": [rid, arrive, eop, IDS_MEMBER, [lo - c, hi - c]]})
            elif kind == "n":
                m = pi.member_at(a, lo, hi)
                if m is None:
                    pr = pi.by_i.get(a)
                    pd = pi.describe(pr) if pr is not None else None
                    if pd is not None and pd["members"] == [] and (pd["family"] in EVALUATORS or pd["family"] in DELEGATES):  # an interior producer
                        if depth >= INTERIOR_DEPTH:
                            raise _Unresolved(f"interior chain deeper than {INTERIOR_DEPTH} at row {a} ({pd['family']})")
                        sub_rec: list = []
                        if a in interior:                                              # one evaluation per interior row per VU
                            got = interior[a]
                        else:
                            got = interior[a] = _evaluate_row(pi, store, weights, pr, rid, arrive, sub_rec, depth=depth + 1, interior=interior,
                                                              row=d["row"] if row is None else row, plan=plan)
                        if plan:
                            continue
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
    if plan:
        return {}
    if ROW_EVALUATORS:
        return ROW_EVALUATORS[-1](pi, r, d, _Operands(inputs, wslices))
    fn = DELEGATES.get(fam)
    if fn is not None:                                                   # a protocol's replay answers for the row (`delegate`)
        member = d["members"][0][0] if d["members"] else "out"
        at = d["row"] if d["members"] or row is None else row                   # an interior producer is at its consumer's row
        return fn(st, inputs, wslices, Place(rid, arrive + d["step"], d["op_path"], at, member, store))
    return evaluate(fam, st, inputs, wslices, refs=wrefs)


def replay_vu(pi: ProgramIndex, store: CommittedStore, weights: Callable[[str, int, int], np.ndarray | None], vu: dict, *,
              arrive: int) -> dict[str, Any]:
    """One VU: resolve its inputs to committed words / registered weights, evaluate, compare with its committed outputs (`opened_check`)."""
    return opened_check(getattr(store, "committed", None), lambda rec: _replay_vu(pi, store, weights, vu, arrive, rec))


def plan_vu(pi: ProgramIndex, store: CommittedStore, vu: dict, *, arrive: int) -> str | None:
    """Reads what `replay_vu` would read of the committed store for `vu` -- its inputs (interior rows included), its committed output
    members and their plane copies -- without evaluating it; the first input it cannot resolve, named (the replay will name it too)."""
    r = pi.by_i[vu["i"]]
    d = pi.describe(r)
    if d is None:
        return f"row {r.get('i')} is unmodelled"
    why = None
    try:
        _evaluate_row(pi, store, None, r, vu["request_id"], arrive, [], depth=0, interior={}, plan=True)  # type: ignore[arg-type]
    except _Unresolved as u:
        why = str(u)
    for m in d["members"]:
        width = int(m[1])
        store.words(vu["request_id"], arrive + d["step"], d["op_path"], m[0], d["row"] * width, (d["row"] + 1) * width)
    for c in d.get("copies") or []:
        width = int(c["width"])
        store.words(vu["request_id"], arrive + d["step"], c["op_path"], c["member"], int(c["row"]) * width, (int(c["row"]) + 1) * width)
    return why


def by_member(got: dict[str, Any], ports: dict[str, str]) -> dict[str, Any]:
    """A multi-output relation's values (keyed by its Definition's ports) under the member each port is committed as
    (`ProgramIndex.member_of_port`; DeepGemmFp8Linear_v1: 0 -> x_q, 1 -> x_s, 2 -> 0 or part_rank<r>).  A port with no committed member
    keeps its name unless that name is another port's member: it never stands in for it."""
    out = {ports[p]: v for p, v in got.items() if p in ports}
    return out | {p: v for p, v in got.items() if p not in ports and p not in out}


def plane_ports(d: dict) -> dict[str, str]:
    """A MoE plane row's output port j -> its j-th plane member (in_quant: 0 -> in_q, 1 -> in_scale): `_describe_plane` lists the
    members in the relation's return order.  {} for any other row."""
    return {str(j): m[0] for j, m in enumerate(d["members"])} if d.get("plane_kind") else {}


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
    if len(got) > 1:
        got = by_member(got, pi.member_of_port(vu["i"]) or plane_ports(d))
    if d.get("plane") and d["plane"] not in got and len(got) == 1:
        got = {d["plane"]: next(iter(got.values()))}                             # a dense relation's `out` IS this row's plane (act = SiluMul out)
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
                alias = next(iter(got.values()))                                 # `<experts>/out` = the MoeSum relation's one output
            if alias is None and len(d["members"]) == 1 and len({id(v) for v in got.values()}) == 1:
                alias = next(iter(got.values()))                                 # a one-output relation's words ARE the row's one member, however spelt
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
            # a LIFTED member of the padded MoE construction (Q_module_pad_v1): the slot enable a_e is derived from the
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
                              f"({RM.LIFTED_DTYPES[dtype][1]} B: Value<{RM.LIFTED_DTYPES[dtype][0] + 1}>) -- representation mismatch")
                return rec | {"result": None}
            eq, nd, first, verdict = lifted_compare(have, want, active, dtype)
            outs.append({"member": member, "width": width, "equal": eq, "differing_elements": nd, "first_diff": first, "lifted": dtype,
                         "slot_enable": int(active), "verdict": verdict})
        else:
            if _ITEMSIZE.get(dtype, 2) == 4:                                     # f32 / i32 plane members (router_w / router_ids): whole words
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
        if _ITEMSIZE.get(str(c["dtype"]), 2) == 4:                              # the primary path's word-width switch, mirrored
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
