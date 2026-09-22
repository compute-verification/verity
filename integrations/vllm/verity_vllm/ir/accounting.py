"""Commitment accounting (GUIDANCE-2026-09-07 §18.5).

For a plan (a set of verification units with boundaries) over a program, and a commitment layout (the leaves
that were actually committed, with their bytes), compute

  (i)   the *required* distinct boundary values: the union of every unit's `in ∪ out` plus the protocol
        boundary (prompt / output leaves, the weights the program reads);
  (ii)  the *committed* bytes: every committed leaf's bytes;
  (iii) the *extra* bytes: committed bytes that (i) does not need, classified as `padding` (bytes of a leaf
        beyond the widths of the values it holds) or `conservative_capture` (whole values committed that no
        unit boundary needs — e.g. the `norm.rmsnorm` rows at non-sampled prefill positions and the fp8 scale
        scalars TA1 recorded as `not_in_program`);

and assert, fail-closed, that every committed logical value is either required by (i) or explicitly declared
extra (`declared_extra`), listing the offending values otherwise (`AccountingError`). A value that is
*internal* to a unit (produced and consumed inside it, on no boundary) must never be committed; the adapters
pass the set of internal values they know so that the assertion names them separately.

Everything here is at the granularity of *logical values* (one gate output = one `Value<w>`); a leaf is a
physical container for a run of values (IR §C: authentication locations are not semantic components). Leaf
containers that hold a run of `n` values of one width are described by (n, width) rather than enumerated, so
the 271,350-leaf B1 layout is accounted in seconds.

Adapters (each returns an `Account`):

* `structural_plan_fixture(doc, plan_index)` — RA's `veritor/structural-plans/v0` fixtures: units = the plan's
  verification units (`expected.vu_boundaries`), committed = the commitment units' boundaries.
* `identity_fixture(doc, plan, layout)` — the cross-representation identity fixture: units = the plan's
  `vu/*`, committed = the named layout's domains/leaves.
* `w3c_fixture(dir)` — one authenticated GEMM-coordinate fixture (W3c): unit = the coordinate, committed = the
  three opened row leaves.
* `ta1_results(dir, plan)` — TA1's B0/B1 capture: committed leaves reconstructed from `description.json` ×
  `commitments.json` (checked against the B0 store's per-leaf index when it is present) plus the weights index;
  plans `per-op-row` (TA1's replay-unit plan: one unit per captured op row, `required.json`) and `named-calls`
  (the typed program's `Embedding`/`LayerPre`/`Attention`/`LayerPost`/`Final` calls).

    PYTHONPATH=. python -m verity_vllm.ir.accounting [--out accounting.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ValueRun:
    """`count` consecutive logical values of `width` bits sharing the id prefix `name` (`name` alone when
    count == 1; otherwise `name[k]`). Two runs with the same name are the same values."""
    name: str
    count: int = 1
    width: int = 16

    @property
    def bits(self) -> int:
        return self.count * self.width

    @property
    def bytes(self) -> int:
        return self.count * bytes_of_width(self.width)


def bytes_of_width(w: int) -> int:
    return (w + 7) // 8


@dataclass
class Leaf:
    """A committed leaf: the value runs it holds and the bytes it occupies in the commitment (>= the values' bytes)."""
    id: str
    domain: str
    values: list[ValueRun]
    nbytes: int

    @property
    def value_bytes(self) -> int:
        return sum(v.bytes for v in self.values)

    @property
    def padding(self) -> int:
        return self.nbytes - self.value_bytes


@dataclass
class Unit:
    """A verification unit's boundary, as value runs (`in_ ∪ out` is the boundary; `internal` are values
    produced and consumed inside — never to be committed)."""
    id: str
    in_: list[ValueRun] = field(default_factory=list)
    out: list[ValueRun] = field(default_factory=list)
    internal: list[ValueRun] = field(default_factory=list)


class AccountingError(AssertionError):
    pass


@dataclass
class Account:
    case: str
    plan: str
    # (i)
    required_values: int
    required_bytes: int
    required_runs: int
    # (ii)
    committed_leaves: int
    committed_values: int
    committed_bytes: int
    # (iii)
    extra_bytes: int
    padding_bytes: int
    conservative_capture_bytes: int
    conservative_capture_values: int
    declared_extra_bytes: int
    # required values with no committed container (opened directly / not committed): informative
    required_uncommitted_values: int
    required_uncommitted_bytes: int
    # the assertion
    offending: list[dict]
    internal_committed: list[dict]
    extra_detail: list[dict]
    notes: list[str] = field(default_factory=list)
    domains: dict[str, dict] = field(default_factory=dict)  # top-level commitment domain -> {"leaves", "bytes"}

    @property
    def ok(self) -> bool:
        return not self.offending and not self.internal_committed

    def to_json(self, max_detail: int | None = None) -> dict:
        """The record. `max_detail` truncates the per-value lists (the counts and bytes are always complete)."""
        d = dict(self.__dict__)
        d["ok"] = self.ok
        if max_detail is not None:
            for k in ("offending", "internal_committed", "extra_detail"):
                if len(d[k]) > max_detail:
                    d[k + "_total"] = len(d[k])
                    d[k] = d[k][:max_detail]
        return d

    def summary_row(self) -> str:
        return (f"{self.case:44s} {self.plan:18s} req {self.required_values:>12,} vals {self.required_bytes:>14,} B | "
                f"committed {self.committed_leaves:>7,} leaves {self.committed_bytes:>14,} B | extra {self.extra_bytes:>13,} B "
                f"(pad {self.padding_bytes:,}, conservative {self.conservative_capture_bytes:,})"
                + (f" | required-but-uncommitted {self.required_uncommitted_values:,} vals" if self.required_uncommitted_values else "")
                + f" | {'OK' if self.ok else 'FAIL'}")


def account(case: str, plan: str, units: Iterable[Unit], protocol: Iterable[ValueRun], leaves: Iterable[Leaf],
            declared_extra: dict[str, str] | None = None, notes: Iterable[str] = ()) -> Account:
    """The three counts and the assertion. `declared_extra` maps a value-run name (or a leaf id) to the reason
    it is committed without being required; anything committed, unrequired and undeclared is *offending*."""
    declared_extra = dict(declared_extra or {})
    required: dict[str, ValueRun] = {}
    internal: dict[str, tuple[ValueRun, str]] = {}
    for u in units:
        for v in list(u.in_) + list(u.out):
            _merge(required, v)
        for v in u.internal:
            internal[v.name] = (v, u.id)
    for v in protocol:
        _merge(required, v)
    for name in list(internal):  # a value on some other unit's boundary is not internal to the plan
        if name in required:
            del internal[name]

    committed_values = committed_bytes = padding = 0
    conservative_b = conservative_v = declared_b = 0
    covered: dict[str, int] = {}  # required run name -> committed count
    offending: list[dict] = []
    internal_committed: list[dict] = []
    extra_detail: list[dict] = []
    n_leaves = 0
    domains: dict[str, dict] = {}
    for leaf in leaves:
        n_leaves += 1
        committed_bytes += leaf.nbytes
        dm = domains.setdefault(leaf.domain.split("/")[0], {"leaves": 0, "bytes": 0})
        dm["leaves"] += 1
        dm["bytes"] += leaf.nbytes
        if leaf.padding < 0:
            raise ValueError(f"leaf {leaf.id}: {leaf.nbytes} bytes hold {leaf.value_bytes} bytes of values")
        padding += leaf.padding
        if leaf.padding:
            extra_detail.append({"leaf": leaf.id, "kind": "padding", "bytes": leaf.padding})
        for v in leaf.values:
            committed_values += v.count
            req = required.get(v.name)
            if req is not None:
                if req.width != v.width:
                    raise ValueError(f"{v.name}: committed width {v.width} != required width {req.width}")
                use = min(v.count, req.count)  # a leaf may hold more values under the same name than are required
                covered[v.name] = covered.get(v.name, 0) + use
                if v.count > req.count:
                    over = ValueRun(v.name, v.count - req.count, v.width)
                    conservative_b += over.bytes
                    conservative_v += over.count
                    reason = declared_extra.get(v.name) or declared_extra.get(leaf.id)
                    extra_detail.append({"leaf": leaf.id, "value": v.name, "kind": "conservative_capture", "values": over.count,
                                         "bytes": over.bytes, "declared": reason})
                    if reason is None:
                        offending.append({"leaf": leaf.id, "value": v.name, "values": over.count, "bytes": over.bytes,
                                          "why": "committed beyond the required run and not declared extra"})
                continue
            conservative_b += v.bytes
            conservative_v += v.count
            reason = declared_extra.get(v.name) or declared_extra.get(leaf.id)
            if v.name in internal:
                internal_committed.append({"leaf": leaf.id, "value": v.name, "values": v.count, "bytes": v.bytes,
                                           "unit": internal[v.name][1], "why": "internal to a unit (an accumulator or intermediate), yet committed"})
            extra_detail.append({"leaf": leaf.id, "value": v.name, "kind": "conservative_capture", "values": v.count, "bytes": v.bytes,
                                 "declared": reason})
            if reason is not None:
                declared_b += v.bytes
            else:
                offending.append({"leaf": leaf.id, "value": v.name, "values": v.count, "bytes": v.bytes,
                                  "why": "committed, required by no unit boundary or protocol boundary, and not declared extra"})
    req_values = sum(r.count for r in required.values())
    req_bytes = sum(r.bytes for r in required.values())
    unc_v = sum(r.count - covered.get(n, 0) for n, r in required.items())
    unc_b = sum((r.count - covered.get(n, 0)) * bytes_of_width(r.width) for n, r in required.items())
    return Account(case, plan, req_values, req_bytes, len(required), n_leaves, committed_values, committed_bytes,
                   padding + conservative_b, padding, conservative_b, conservative_v, declared_b, unc_v, unc_b,
                   offending, internal_committed, extra_detail, list(notes), domains)


def _merge(into: dict[str, ValueRun], v: ValueRun) -> None:
    prev = into.get(v.name)
    if prev is None or prev.count < v.count:
        into[v.name] = v


def assert_accounted(acc: Account) -> Account:
    """Fail closed: raise `AccountingError` naming every committed value that is neither required nor declared."""
    if acc.internal_committed:
        raise AccountingError(f"{acc.case} [{acc.plan}]: {len(acc.internal_committed)} internal value(s) committed: "
                              + "; ".join(f"{d['value']} ({d['unit']})" for d in acc.internal_committed[:20]))
    if acc.offending:
        raise AccountingError(f"{acc.case} [{acc.plan}]: {len(acc.offending)} committed value run(s) neither required nor declared extra: "
                              + "; ".join(f"{d['value']}x{d['values']}@{d['leaf']}" for d in acc.offending[:20]))
    return acc


# ---------------------------------------------------------------------------------------------------------
# adapters: RA's structural-plan fixtures
# ---------------------------------------------------------------------------------------------------------

def structural_plan_fixture(doc: dict, plan_index: int) -> Account | None:
    """Units = the plan's verification units (boundaries from `expected`), committed = the commitment units'
    boundary values (each as a one-value leaf of ceil(width/8) bytes), protocol = `Input` gates + protected
    outputs. Returns None for plans the validator rejected (they have no boundaries)."""
    plan = doc["plans"][plan_index]
    exp = plan["expected"]
    if exp["status"] != "valid":
        return None
    gates = doc["expanded"]["gates"]
    w = lambda g: gates[g]["width"]
    vu = [Unit(f"vu/{i}", [ValueRun(f"g{g}", 1, w(g)) for g in b["in"]], [ValueRun(f"g{g}", 1, w(g)) for g in b["out"]])
          for i, b in enumerate(exp["vu_boundaries"])]
    on_boundary = {g for b in exp["vu_boundaries"] for g in b["in"] + b["out"]}
    for i, u in enumerate(plan["verification_units"]):
        if "gates" in u:
            vu[i].internal = [ValueRun(f"g{g}", 1, w(g)) for g in u["gates"] if g not in on_boundary]
    protocol = [ValueRun(f"g{i}", 1, g["width"]) for i, g in enumerate(gates) if g["op"] == "Input"]
    protocol += [ValueRun(f"g{g}", 1, w(g)) for g in doc["expanded"]["protected_outputs"]]
    cu_of: dict[int, int] = {}  # a value on several commitment-unit boundaries is committed once (by its first unit)
    for i, b in enumerate(exp["cu_boundaries"]):
        for g in b["in"] + b["out"]:
            cu_of.setdefault(g, i)
    leaves = [Leaf(f"cu{i}/g{g}", f"cu{i}", [ValueRun(f"g{g}", 1, w(g))], bytes_of_width(w(g))) for g, i in sorted(cu_of.items())]
    return account(f"structural-plans/{doc['name']}", plan["name"], vu, protocol, leaves)


# ---------------------------------------------------------------------------------------------------------
# adapters: the identity fixture
# ---------------------------------------------------------------------------------------------------------

def identity_fixture(doc: dict, plan: str, layout: str = "scalar") -> Account:
    width = {g["ref"]: g["width"] for g in doc["gates"]}
    by_id = {u["id"]: u for u in doc["units"]}
    units = []
    for uid in doc["plans"][plan]:
        u = by_id[uid]
        b = u["boundary"]
        body = {doc["gates"][i]["ref"] for i in u["gates"]}
        internal = [ValueRun(r, 1, width[r]) for r in sorted(body - set(b["in"]) - set(b["out"]))]
        units.append(Unit(uid, [ValueRun(r, 1, width[r]) for r in b["in"]], [ValueRun(r, 1, width[r]) for r in b["out"]], internal))
    protocol = [ValueRun(g["ref"], 1, g["width"]) for g in doc["gates"] if g["op"] == "Input"]
    protocol += [ValueRun(doc["gates"][i]["ref"], 1, doc["gates"][i]["width"]) for i in doc["protected"]]
    leaves = []
    for dom in doc["layouts"][layout]:
        for rank, words in enumerate(dom["leaves"]):
            vals = [ValueRun(r, 1, width[r]) for r in words]
            leaves.append(Leaf(f"{dom['id']}/{rank}", dom["id"], vals, sum(v.bytes for v in vals)))
    # the layout commits every value of the program; under a coarse plan some are internal to a unit — the
    # assertion must name them (nothing is declared extra here on purpose)
    internal_names = {v.name for u in units for v in u.internal}
    return account(f"identity/{layout}", plan, units, protocol, leaves, declared_extra={},
                   notes=[f"internal under this plan: {sorted(internal_names)}"] if internal_names else [])


# ---------------------------------------------------------------------------------------------------------
# adapters: W3c authenticated GEMM-coordinate fixtures
# ---------------------------------------------------------------------------------------------------------

def w3c_fixture(d: Path) -> Account:
    """One coordinate unit `out_row[j] = bf16(x[arg_cols] . w_row)`; committed: the three opened row leaves."""
    f = json.loads((d / "fixture.json").read_text())
    K, N, j = f["K"], f["N"], f["j"]
    x_words = os.path.getsize(d / "x.u16") // 2
    cols = f.get("x_arg_cols")
    lo, hi = (cols if isinstance(cols, list) and len(cols) == 2 else [0, K])  # None: the GEMM consumes the whole row
    x_vals = [ValueRun("x[arg_cols]", hi - lo, 16)] + ([ValueRun("x[other]", x_words - (hi - lo), 16)] if x_words > hi - lo else [])
    unit = Unit(f"coordinate/{f['op']}/j={j}", [ValueRun("x[arg_cols]", hi - lo, 16), ValueRun("w_row", K, 16)], [ValueRun(f"out_row[{j}]", 1, 16)])
    leaves = [Leaf(f["x_id"].split("/t=", 1)[1], "ru", x_vals, os.path.getsize(d / "x.u16")),
              Leaf(f["w_id"], "weights", [ValueRun("w_row", K, 16)], os.path.getsize(d / "w_row.u16")),
              Leaf(f["canonical_id"].split("/t=", 1)[1], "ru", [ValueRun(f"out_row[{j}]", 1, 16), ValueRun("out_row[other]", N - 1, 16)],
                   os.path.getsize(d / "out_row.u16"))]
    declared = {"out_row[other]": "row leaf; the coordinate VU is a view on it (vu-integration.md §6): the other N-1 coordinates are other units' outputs",
                "x[other]": "row leaf; the unit reads x[arg_cols] only"}
    return account(f"w3c/{d.name}", "one-coordinate", [unit], [], leaves, declared_extra=declared)


# ---------------------------------------------------------------------------------------------------------
# adapters: TA1's B0/B1 captures (description × commitments × weights index)
# ---------------------------------------------------------------------------------------------------------

_ITEM = {"bfloat16": 2, "float32": 4, "int32": 4}

# Returns of the typed program's named calls, per captured op suffix and port (b1.py `LayerPre`/`LayerPost`/
# `Final`/`Embedding`; `Attention` returns its whole output). `slice` = the words of the leaf that are returns.
def _named_call_returns(C: dict) -> dict[tuple[str, int], tuple[str, tuple[int, int] | None]]:
    NH, KVH, D, H = C["NH"], C["KVH"], C["D"], C["H"]
    qkv_v = ((NH + KVH) * D, (NH + 2 * KVH) * D)  # the V rows are returned by LayerPre (the K/Q rows go through RoPE inside it)
    m = {
        ("embed_tokens.embedding", 0): ("Embedding", None),
        ("input_layernorm.rmsnorm", 1): ("LayerPre", None),        # residual' (fused norm; layer 0 has no port 1)
        ("self_attn.rotary_emb.rope", 0): ("LayerPre", None),      # q
        ("self_attn.rotary_emb.rope", 1): ("LayerPre", None),      # k
        ("self_attn.attn.attn", 0): ("Attention", None),
        ("mlp.down_proj.gemm", 0): ("LayerPost", None),
        ("post_attention_layernorm.rmsnorm", 1): ("LayerPost", None),
        ("sampler.select", 0): ("Final", None),
    }
    m[("self_attn.qkv_proj.add" if C.get("BIAS", True) else "self_attn.qkv_proj.gemm", 0)] = ("LayerPre", qkv_v)
    return m


def _leaf_shape_words(shape: list[int]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def ta1_results(d: Path, plan: str = "per-op-row", store: Path | None = None) -> Account:
    """`d` = `fixtures/results/TA1-typed-b1-20260907/B0` or `/B1`. Committed leaves are reconstructed per
    replay unit from `description.json` (ops with `positions: all` at every position of the step, `sampled` ops
    at the step's sampled position) and the weights index; when the extracted store (`store/ru/<request>/<step>/
    index.json`) is present, the reconstruction is checked against it leaf for leaf."""
    from verity_vllm.registry import b1
    desc = json.loads((d / "description.json").read_text())
    comm = json.loads((d / "commitments.json").read_text())
    widx = json.loads((d / "weights" / "index.json").read_text())
    C = b1.B1_CONFIG if "Qwen" in desc["model"] else b1.B0_CONFIG
    cfg = "B1" if C is b1.B1_CONFIG else "B0"
    notes: list[str] = []
    nodes = [n for n in desc["nodes"] if n["origin"] == "captured"]
    ports = {}  # op -> (node, [(port, shape, dtype)])
    for n in nodes:
        outs = [(0, list(n["shape"]), n["dtype"])] + [(o["port"], list(o["shape"]), o["dtype"]) for o in n.get("outputs", [])]
        ports[n["op"]] = (n, outs)
    named = _named_call_returns(C)
    leaves: list[Leaf] = []
    units: list[Unit] = []
    declared: dict[str, str] = {}
    protocol: list[ValueRun] = []
    n_positions = n_steps = n_nip = 0
    requests = sorted(p.name for p in (d / "ru").iterdir() if p.is_dir())
    for request in requests:
        boundary = json.loads((d / "ru" / request / "boundary.json").read_text())
        steps = boundary["steps"]
        sampled = {r["t"] for r in boundary["output_records"]}
        LP = len(boundary["prompt_token_ids"])
        rq = request[:8]
        n_steps += len(steps)
        for st in steps:
            step, ph, positions = st["step"], st["phase"], st["positions"]
            tree = comm["trees"].get(f"ru/{request}/{step:06d}")
            dom = f"ru/{request}/{step:06d}"
            cnt = 0
            for t in positions:
                n_positions += 1
                for op, (n, outs) in ports.items():
                    if ph not in n["phases"] or (n["positions"] == "sampled" and t not in sampled):
                        continue
                    L = n["layer"]
                    for port, shape, dtype in outs:
                        vid = f"{rq}/t={t}/{op}/port={port}"
                        words = _leaf_shape_words(shape)
                        width = 8 * _ITEM[dtype]
                        leaves.append(Leaf(vid, dom, [ValueRun(vid, words, width)], words * _ITEM[dtype]))
                        cnt += 1
                        if plan == "per-op-row":
                            # TA1's replay-unit plan: one unit per captured op row; out = its ports, in = its args (other
                            # ops' ports at this position, weights, the token) — every activation leaf is some unit's `out`
                            if port == 0:
                                ins: list[ValueRun] = []
                                for a in n["args"]:
                                    if a["source"] == "node":
                                        sh = next(s for p, s, _ in ports[a["op"]][1] if p == a["port"])
                                        ins.append(ValueRun(f"{rq}/t={t}/{a['op']}/port={a['port']}", _leaf_shape_words(sh), 16))
                                    elif a["source"] == "weight":
                                        e = _windex(widx, "w=" + a["param"])
                                        if e is not None:
                                            ins.append(ValueRun("w=" + a["param"], _leaf_shape_words(e["shape"]), 8 * _ITEM[e["dtype"]]))
                                    elif a["source"] == "token":  # the previous position's sampled token or a prompt word
                                        ins.append(ValueRun(f"{rq}/t={t - 1}/sampler.select/port=0", 1, 32) if t >= LP
                                                   else ValueRun(f"{rq}/prompt_token_ids", LP, 32))
                                units.append(Unit(vid, ins, [ValueRun(f"{rq}/t={t}/{op}/port={p}", _leaf_shape_words(s), 8 * _ITEM[dt]) for p, s, dt in outs]))
                            if L == "norm" and t not in sampled:
                                n_nip += 1
                        else:  # named-calls: the typed program's Embedding / LayerPre / Attention / LayerPost / Final
                            suffix = op if L in ("embed", "norm", "lm_head", "sampler") else op[len(f"layers.{L}."):]
                            ret = named.get((suffix, port))
                            if L in ("norm", "lm_head", "sampler") and t not in sampled:
                                declared[vid] = "not_in_program: production runs the final norm/LM head at every prefill row; the program has a Final only at sampled positions"
                            elif L in ("lm_head", "norm"):
                                declared[vid] = "internal to Final (its return is the token)"
                            elif ret is None:
                                declared[vid] = "internal to the named call (captured at production kernel granularity, not a return of the composite)"
                            else:
                                call, sl = ret
                                if sl is None:
                                    units.append(Unit(f"{rq}/t={t}/{L}/{call}", [], [ValueRun(vid, words, width)]))
                                else:  # only the words [lo, hi) of the leaf are the return (the V rows); the rest is internal
                                    lo, hi = sl
                                    leaves[-1] = Leaf(vid, dom, [ValueRun(vid + "[q,k]", lo, width), ValueRun(vid + "[v]", hi - lo, width)], words * _ITEM[dtype])
                                    units.append(Unit(f"{rq}/t={t}/{L}/{call}", [], [ValueRun(vid + "[v]", hi - lo, width)]))
                                    declared[vid + "[q,k]"] = "internal to LayerPre (rotated inside the call); only the V rows are returned"
            if tree is not None and tree["leaf_count"] != cnt:
                notes.append(f"{rq} step {step}: reconstructed {cnt} leaves, commitments.json says {tree['leaf_count']}")
        # protocol boundary of this request: the prompt leaf and the sampled tokens
        protocol.append(ValueRun(f"{rq}/prompt_token_ids", LP, 32))
        leaves.append(Leaf(f"boundary/{request}/prompt_token_ids", "boundary", [ValueRun(f"{rq}/prompt_token_ids", LP, 32)], 4 * LP))
        for t in sorted(sampled):
            protocol.append(ValueRun(f"{rq}/t={t}/sampler.select/port=0", 1, 32))
        # check against the extracted store, if present
        if store is not None and (store / "ru" / request).is_dir():
            idx_bytes = 0
            idx_ids = set()
            for st in steps:
                ix = json.loads((store / "ru" / request / f"{st['step']:06d}" / "index.json").read_text())
                for e in ix["entries"]:
                    idx_bytes += e["nbytes"]
                    m = _CID.match(e["canonical_id"])
                    idx_ids.add(f"{rq}/t={m['t']}/{m['op']}/port={m['port']}")
            rec = [lf for lf in leaves if lf.domain.startswith(f"ru/{request}/")]
            rec_ids = {lf.id for lf in rec}
            notes.append(f"{rq}: store index {len(idx_ids)} leaves / {idx_bytes:,} B vs reconstruction {len(rec_ids)} / {sum(lf.nbytes for lf in rec):,} B; "
                         f"ids {'identical' if idx_ids == rec_ids else f'differ ({len(idx_ids ^ rec_ids)})'}")
    if plan == "per-op-row":
        notes.append(f"{n_nip} norm.rmsnorm leaves at non-sampled positions are units of this plan (required here) but not_in_program for the typed Serve")

    # --- weights ---------------------------------------------------------------------------------------
    read_params = {"w=" + a["param"] for n in nodes for a in n["args"] if a["source"] == "weight"}
    read_params.add("w=layers.0.self_attn.rotary_emb.cos_sin_cache")
    for e in widx["entries"]:
        cid = e["canonical_id"]
        words = _leaf_shape_words(e["shape"])
        width = 8 * _ITEM[e["dtype"]]
        leaves.append(Leaf(cid, "weights", [ValueRun(cid, words, width)], words * _ITEM[e["dtype"]]))
        if cid in read_params:
            protocol.append(ValueRun(cid, words, width))
        else:
            declared[cid] = "not_in_program: no gate of the bf16 program reads this leaf (fp8 scale scalar)"
    ru_leaves = sum(1 for lf in leaves if lf.domain.startswith("ru/"))
    ru_bytes = sum(lf.nbytes for lf in leaves if lf.domain.startswith("ru/"))
    notes.append(f"{len(requests)} request(s), {n_steps} replay units, {n_positions} positions; replay-unit leaves {ru_leaves:,} / {ru_bytes:,} B "
                 f"(capture_summary: {_capture_summary(d)})")
    return account(f"TA1/{cfg}/{desc['profile_id']}", plan, units, protocol, leaves, declared_extra=declared, notes=notes)


import re as _re

_CID = _re.compile(r"^p=(?P<p>[^/]+)/r=(?P<r>[^/]+)/t=(?P<t>\d+)/ph=(?P<ph>[^/]+)/l=(?P<l>[^/]+)/op=(?P<op>[^/]+)/port=(?P<port>\d+)$")


def _capture_summary(d: Path) -> str:
    p = d / "capture_summary.json"
    if not p.exists():
        return "n/a"
    s = json.loads(p.read_text())
    return f"{s.get('leaves'):,} leaves / {s.get('bytes_captured'):,} B"


def _windex(widx: dict, cid: str):
    for e in widx["entries"]:
        if e["canonical_id"] == cid:
            return e
    return None


# ---------------------------------------------------------------------------------------------------------
# run everything local
# ---------------------------------------------------------------------------------------------------------

def run_all(store_b0: Path | None = None) -> list[Account]:
    out: list[Account] = []
    sp = REPO / "fixtures/verity-ir/structural-plans/v0"
    for p in sorted(sp.glob("*.json")):
        doc = json.loads(p.read_text())
        for i in range(len(doc["plans"])):
            a = structural_plan_fixture(doc, i)
            if a is not None:
                out.append(a)
    ident = json.loads((REPO / "fixtures/verity-ir/identity/identity-v0.json").read_text())
    for plan in ident["plans"]:
        for layout in ("scalar", "packed"):
            out.append(identity_fixture(ident, plan, layout))
    w3 = REPO / "fixtures/B1-b3eba618b1b5-20260907T1731Z"
    if w3.is_dir():
        for p in sorted(x for x in w3.iterdir() if x.is_dir() and (x / "fixture.json").exists()):
            out.append(w3c_fixture(p))
    ta1 = REPO / "fixtures/results/TA1-typed-b1-20260907"
    for cfg in ("B0", "B1"):
        if (ta1 / cfg / "description.json").exists():
            for plan in ("per-op-row", "named-calls"):
                out.append(ta1_results(ta1 / cfg, plan, store=store_b0 if cfg == "B0" else None))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--store-b0", type=Path, default=None, help="extracted B0 commit store (checks the leaf reconstruction)")
    ap.add_argument("--strict", action="store_true", help="exit 1 on the first failed assertion")
    ap.add_argument("--max-detail", type=int, default=64, help="per-value detail lists in --out are truncated to this many entries")
    a = ap.parse_args(argv)
    accs = run_all(a.store_b0)
    fails = 0
    for acc in accs:
        print(acc.summary_row())
        for n in acc.notes:
            print(f"    note: {n}")
        try:
            assert_accounted(acc)
        except AccountingError as e:
            fails += 1
            print(f"    ASSERTION FAILED: {e}")
            if a.strict:
                return 1
    if a.out:
        a.out.write_text(json.dumps([x.to_json(max_detail=a.max_detail) for x in accs], indent=1))
    print(f"{len(accs)} cases, {fails} failed assertions")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
