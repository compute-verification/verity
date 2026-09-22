"""Padded request rectangle (CB-B): the mechanically derived flat request body (prefill | decode 1 … decode D_MAX —
every step present, since the meta execution takes no EOS / data-dependent branch) rewritten into the `Serve@3`
form: liveness as a circuit value with the prescribed no-op relation, using the *registered* padding primitives and
nothing architecture-shaped.

  * sampling events: after each step's sample node, `PadGuard_v1{EOS}(live, y) -> (tok_out, live_out)` (serve3.py);
    the next step's read of the raw sample is redirected to `tok_out`; `live_0 = Const1[0x1]`.
  * state rows: for every decode step s ≥ 1, every collection produced in step s and read by a *later* step — found
    mechanically as the cross-step reference edges of the flat body (that set IS the KV state: here the RoPE'd K row
    and the V slice of the QKV GEMM output, per layer, plus the sampled token) — is muxed
    `batch(SelectBf16_v1, live_{s-1}, row, zero-row)` against one broadcast `Const16[0x0]`, and every later-step
    read is redirected to the muxed row.  Same-step reads stay raw (the step's own attention reads its fresh row);
    prefill rows are not muxed (prefill is always live; spec §3).  The mux keys on the liveness *entering* the step
    (`live_{s-1}`), as `PadDecodeStep_v1` does.
  * return: `(out: [D_MAX+1] i32, live: [D_MAX+1] bit)` — `out[g]` the muxed token of event g, `live[g]` the flag
    after event g (spec §8).

Why `PadDecodeStep_v1` is not bound as registered: its body (a) is `b1._forward_positions_v2` — the hand-written
Serve_v2 step — so binding it would *replace* the derived step body with the hand emitter.  The padding primitive
needed is "PadDecodeStep with the step body as a parameter" (guard + row muxes around a derived step function);
here the same gates are composed flat over the derived body and checked against `serve3.serve_v3` by gate count.

`peel_check` verifies the active region: dropping the inserted nodes and undoing the redirects reproduces the flat
body node-for-node (spec id, operand runs, output type).
"""

from __future__ import annotations

from typing import Any

from verity.ir.defs import Body, Builder, CompositeDefinition, Node, bind
from verity.ir.refs import Affine, Coll, concat, runs, tuple_of
from verity_vllm.registry import prims as P
from verity_vllm.registry.b1 import ZERO16, const
from verity_vllm.registry.prims import BIT, I32
from verity_vllm.registry.serve3 import PadGuard
from verity.ir.types import Array, Tuple, Value


def step_index(step_counts: list[int]) -> list[int]:
    out: list[int] = []
    for si, c in enumerate(step_counts):
        out += [si] * c
    return out


def state_ranges(nodes: list[Node], step: list[int]) -> dict[int, list[tuple[int, int]]]:
    """producer node k -> merged leaf ranges [lo, hi) of its output read by nodes of a later step."""
    raw: dict[int, list[tuple[int, int]]] = {}
    for k, n in enumerate(nodes):
        for c in n.args:
            for sp, idx, lo, hi in runs(c.refs):
                if sp == "n" and step[idx] < step[k]:
                    raw.setdefault(idx, []).append((lo, hi))
    out: dict[int, list[tuple[int, int]]] = {}
    for k, rs in raw.items():
        rs.sort()
        merged: list[list[int]] = []
        for lo, hi in rs:
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        out[k] = [(lo, hi) for lo, hi in merged]
    return out


def leaf_type_of(t, lo: int, hi: int):
    """The single leaf type of leaves [lo, hi) of type t (raises if mixed)."""
    ws = set(t.leaf_widths()[lo:hi])
    if len(ws) != 1:
        raise TypeError(f"state range [{lo},{hi}) of {t!r} has mixed leaf widths {sorted(ws)}")
    return Value(ws.pop())


def pad_body(body: Body, step_counts: list[int], eos: int, *, name_prefix: str = "", mux_last_step: bool = True) -> tuple[Body, dict]:
    """`mux_last_step`: Serve@3 appends (and muxes) the cache rows of *every* decode step (spec §3), but the last
    step's rows have no reader in the flat body, so they are not cross-step edges.  When True, the state signature
    learned from the earlier decode steps — (definition id → leaf ranges) — is applied to the last step's producers
    too, so the padded suffix has the spec's structure; when False only rows that are actually read are muxed."""
    nodes = body.nodes
    step = step_index(step_counts)
    assert len(step) == len(nodes), (len(step), len(nodes))
    n_steps = len(step_counts)
    D_MAX = n_steps - 1
    sample_of_step: dict[int, int] = {}
    for sp, idx, lo, hi in runs(body.ret):
        assert sp == "n" and (lo, hi) == (0, 1), "the request body returns one sampled token per step"
        sample_of_step[step[idx]] = idx
    assert sorted(sample_of_step) == list(range(n_steps)), sample_of_step
    st = state_ranges(nodes, step)
    if mux_last_step and n_steps >= 3:
        kinds: dict[str, list[tuple[int, int]]] = {}
        for k, rs in st.items():
            if step[k] >= 1 and k != sample_of_step.get(step[k]):
                kinds.setdefault(nodes[k].fn.id, rs)
        last = n_steps - 1
        for k, n in enumerate(nodes):
            if step[k] == last and n.fn.id in kinds and k not in st and k != sample_of_step.get(last):
                st[k] = list(kinds[n.fn.id])
    stats = {"steps": n_steps, "D_MAX": D_MAX, "guards": 0, "mux_batches": 0, "mux_members": 0, "zero_consts": 0,
             "state_rows_by_step": {}, "state_kinds": {}}
    for k, rs in st.items():
        kind = f"{nodes[k].fn.id}:{rs}"
        stats["state_kinds"][kind] = stats["state_kinds"].get(kind, 0) + 1

    ret_type = Tuple(Array(D_MAX + 1, I32), Array(D_MAX + 1, BIT))
    B = Builder(body.params, ret_type)
    env: dict[tuple[str, int], Any] = {("p", i): Affine("p", i, 0, t.leaves) for i, (_, t) in enumerate(body.params)}
    redirect: dict[int, list[tuple[int, int, Any]]] = {}     # old producer -> [(lo, hi, refs replacing that range for later steps)]
    guard = bind(PadGuard, EOS=eos)
    live_by_event: dict[int, Coll] = {}
    outs: list[Coll] = []
    lives: list[Coll] = []
    zero_of_step: dict[int, Coll] = {}
    inserted_at: dict[int, str] = {}                        # new index -> kind (for peel_check)
    new_of_old: dict[int, int] = {}

    def subst(refs, reader_step: int):
        parts = []
        for sp, idx, lo, hi in runs(refs):
            if sp == "n" and step[idx] < reader_step and idx in redirect:
                pos = lo
                for rlo, rhi, rrefs in redirect[idx]:
                    if rhi <= pos or rlo >= hi:
                        continue
                    if rlo > pos:
                        parts.append(env[("n", idx)].slice(pos, rlo))
                        pos = rlo
                    seg_hi = min(hi, rhi)
                    parts.append(rrefs.slice(pos - rlo, seg_hi - rlo))
                    pos = seg_hi
                if pos < hi:
                    parts.append(env[("n", idx)].slice(pos, hi))
            else:
                parts.append(env[(sp, idx)].slice(lo, hi))
        return concat(parts) if parts else refs

    for k, node in enumerate(nodes):
        s = step[k]
        colls = [Coll(a.type, subst(a.refs, s)) for a in node.args]
        if node.form in ("call", "primitive"):
            out = B.call(node.fn, *colls, name=node.name)
        elif node.form == "batch":
            out = B.batch(node.fn, *colls, axes=node.axes, n=node.n, name=node.name)
        else:
            out = B.scan(node.fn, colls[0], xs=colls[1:1 + node.n_xs], shared=colls[1 + node.n_xs:], n=node.n, name=node.name)
        env[("n", k)] = out.refs
        new_of_old[k] = len(B.nodes) - 1
        # (c) row muxes: state produced by a decode step, keyed on the liveness entering the step
        if s >= 1 and k in st and k != sample_of_step.get(s):
            live_in = live_by_event[s - 1]
            if s not in zero_of_step:
                zero_of_step[s] = B.call(ZERO16, name=f"{name_prefix}s={s}/zero")
                inserted_at[len(B.nodes) - 1] = "zero"
                stats["zero_consts"] += 1
            z = zero_of_step[s]
            for lo, hi in st[k]:
                lt = leaf_type_of(node.out, lo, hi)
                if lt != Value(16):
                    raise TypeError(f"state row of node {k} ({node.fn.id}) has leaf type {lt!r}; only bf16 rows have a registered "
                                    f"select (SelectBf16_v1) — a Select<{lt.w}> primitive would be needed")
                n = hi - lo
                row = Coll(Array(n, lt), out.refs.slice(lo, hi))
                zrow = Coll(Array(n, lt), concat([z.refs] * n))
                mux = B.batch(P.SelectBf16, live_in, row, zrow, axes=(None, 0, 0), name=f"{name_prefix}s={s}/mux/{node.name or node.fn.id}")
                inserted_at[len(B.nodes) - 1] = f"mux:{k}:{lo}:{hi}"
                redirect.setdefault(k, []).append((lo, hi, mux.refs))
                stats["mux_batches"] += 1
                stats["mux_members"] += n
                stats["state_rows_by_step"][s] = stats["state_rows_by_step"].get(s, 0) + 1
        # (b) the guard at the step's sampling event
        if k == sample_of_step.get(s):
            if s == 0:
                live_prev = B.call(const(1, 1), name=f"{name_prefix}live_0")
                inserted_at[len(B.nodes) - 1] = "live0"
            else:
                live_prev = live_by_event[s - 1]
            y = Coll(I32, out.refs.slice(0, 1))
            g = B.call(guard, live_prev, y, name=f"{name_prefix}s={s}/guard")
            inserted_at[len(B.nodes) - 1] = f"guard:{k}"
            stats["guards"] += 1
            tok_out, live_out = g[0], g[1]
            redirect.setdefault(k, []).append((0, 1, tok_out.refs))
            live_by_event[s] = live_out
            outs.append(tok_out)
            lives.append(live_out)
    for k in redirect:
        redirect[k].sort(key=lambda t: t[0])
    ret = tuple_of(Coll(Array(D_MAX + 1, I32), concat([t.refs for t in outs])),
                   Coll(Array(D_MAX + 1, BIT), concat([x.refs for x in lives])))
    padded = B.finish(ret)
    stats["nodes_flat"] = len(nodes)
    stats["nodes_padded"] = len(padded.nodes)
    stats["gates_flat"] = body.gates
    stats["gates_padded"] = padded.gates
    stats["inserted"] = {kind.split(":")[0]: 0 for kind in inserted_at.values()}
    for kind in inserted_at.values():
        stats["inserted"][kind.split(":")[0]] += 1
    return padded, {"stats": stats, "inserted_at": inserted_at, "new_of_old": new_of_old, "redirect_keys": sorted(redirect)}


def peel_check(flat: Body, padded: Body, info: dict) -> dict:
    """The active region: the padded body minus the inserted nodes, with redirects undone, equals the flat body."""
    inserted_at: dict[int, str] = info["inserted_at"]
    new_of_old: dict[int, int] = info["new_of_old"]
    old_of_new = {v: k for k, v in new_of_old.items()}
    # inverse map for redirected refs: mux node -> (producer old index, lo); guard node -> sample old index (field 0)
    inv: dict[int, tuple[int, int]] = {}
    for new_k, kind in inserted_at.items():
        parts = kind.split(":")
        if parts[0] == "mux":
            inv[new_k] = (int(parts[1]), int(parts[2]))
        elif parts[0] == "guard":
            inv[new_k] = (int(parts[1]), 0)

    def unmap(refs) -> list[tuple[str, int, int, int]]:
        out = []
        for sp, idx, lo, hi in runs(refs):
            if sp == "n":
                if idx in inv:
                    k_old, base = inv[idx]
                    out.append((sp, k_old, lo + base, hi + base))
                else:
                    out.append((sp, old_of_new[idx], lo, hi))
            else:
                out.append((sp, idx, lo, hi))
        # re-merge adjacent runs on the same producer
        merged: list[list] = []
        for r in out:
            if merged and merged[-1][0] == r[0] and merged[-1][1] == r[1] and merged[-1][3] == r[2]:
                merged[-1][3] = r[3]
            else:
                merged.append(list(r))
        return [tuple(m) for m in merged]

    mism = []
    kept = [k for k in range(len(padded.nodes)) if k not in inserted_at]
    if len(kept) != len(flat.nodes):
        return {"ok": False, "reason": f"{len(kept)} kept nodes vs {len(flat.nodes)} flat nodes"}
    for k_old, k_new in zip(range(len(flat.nodes)), kept):
        a, b = flat.nodes[k_old], padded.nodes[k_new]
        if a.fn.id != b.fn.id or a.out != b.out or a.form != b.form:
            mism.append({"k": k_old, "what": "spec/out/form", "flat": a.fn.id, "padded": b.fn.id})
            continue
        ra = [[list(r) for r in runs(c.refs)] for c in a.args]
        rb = [[list(r) for r in unmap(c.refs)] for c in b.args]
        if ra != rb:
            mism.append({"k": k_old, "what": "wiring", "flat": ra, "padded_unmapped": rb})
        if len(mism) > 5:
            break
    return {"ok": not mism, "kept": len(kept), "mismatches": mism}


def padded_program(padded: Body, *, profile_id: str, workload_digest: str, eos: int, D_MAX: int):
    """Wrap the padded body as `PaddedCaptured_v1{PROFILE, WORKLOAD, EOS, D_MAX}` (like the fold's Captured_v1)."""
    from verity_capture.fold import _replayer
    from verity.ir.program import Program

    params, ret_type = padded.params, padded.ret_type
    defn = CompositeDefinition("PaddedCaptured", 1, ("PROFILE", "WORKLOAD", "EOS", "D_MAX"),
                               lambda S, _p=params, _r=ret_type: (_p, _r), _replayer(padded),
                               doc="CB-B: derived flat request body in Serve@3 padded form", conformance="derived: cb-b pad", register=False)
    spec = bind(defn, PROFILE=profile_id, WORKLOAD=workload_digest, EOS=eos, D_MAX=D_MAX)
    spec.set_body(padded)
    return Program(spec), spec


def gate_histogram(fn, memo: dict | None = None) -> dict[str, int]:
    """Expanded-circuit histogram: primitive id -> number of gates, over the fully unfolded definition (memoised per
    specialisation id; composites are unfolded, `batch`/`scan` multiply by n).  The unfolding comparator's view."""
    from collections import Counter

    memo = {} if memo is None else memo
    fid = getattr(fn, "id", str(fn))
    if fid in memo:
        return memo[fid]
    if not hasattr(fn, "body") or getattr(fn, "form", None) == "primitive" or type(fn).__name__ == "PrimitiveDefinition":
        memo[fid] = {fid: 1}
        return memo[fid]
    h: Counter = Counter()
    for n in fn.body.nodes:
        if n.form == "primitive":
            h[n.fn.id] += 1
        else:
            sub = gate_histogram(n.fn, memo)
            mult = 1 if n.form == "call" else n.n
            for k, v in sub.items():
                h[k] += v * mult
    memo[fid] = dict(h)
    return memo[fid]


def histogram_diff(a: dict[str, int], b: dict[str, int]) -> dict:
    keys = set(a) | set(b)
    return {"equal": a == b, "gates": [sum(a.values()), sum(b.values())],
            "only_a": {k: a[k] for k in keys if k not in b}, "only_b": {k: b[k] for k in keys if k not in a},
            "count_diff": {k: [a[k], b[k]] for k in keys if k in a and k in b and a[k] != b[k]}}


def node_histogram(fn, memo: dict | None = None) -> dict[str, int]:
    """Like `gate_histogram` but counting *composite calls* by specialisation id at every nesting level (a
    structural view: which registered Definitions are instantiated how often, in either nesting)."""
    from collections import Counter

    memo = {} if memo is None else memo
    fid = getattr(fn, "id", str(fn))
    if fid in memo:
        return memo[fid]
    if not hasattr(fn, "body") or type(fn).__name__ == "PrimitiveDefinition":
        memo[fid] = {}
        return memo[fid]
    h: Counter = Counter()
    for n in fn.body.nodes:
        if n.form == "primitive":
            continue
        mult = 1 if n.form == "call" else n.n
        h[n.fn.id] += mult
        for k, v in node_histogram(n.fn, memo).items():
            h[k] += v * mult
    memo[fid] = dict(h)
    return memo[fid]
