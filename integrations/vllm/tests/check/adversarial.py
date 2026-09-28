"""tests/check/adversarial.py — lane ov-adversarial (Agent 10, swarm 2026-09-17): the mutation harness.

NOT a checker and NOT a fix.  This module plays the *untrusted adapter author*: it takes a genuine profile, wraps its
patterns so that the Resolutions they return are rewritten (wrong producer, wrong-but-written slice, swallowed Compute,
weight-root substitution, ...), refolds the trusted record (log) with the hostile profile exactly as the sealed CPU
phase does (`run_config.py --phase cpu` refolds from the log with the profile named in `run.json`; it never reads the
lane's `program.json` / `instances.jsonl` / `accesses.jsonl.gz`), and runs the sealed checkers over the result:

    provenance.check      (G5 wiring: brute-force last-writer byte map over the fold's own journal + stream/schema facts)
    replay.tier_a         (G4: stratified k-per-(spec, regime) bit-exact VU evaluation from snapshots + served tokens)
    replay.tier_chain     (G5 chaining smoke test)
    root_policy.evaluate  (G3)
    vu_query              (the exhaustive VU population + `production_sample`)
    census.reconcile      (kernel residue against the census)
    gates.evaluate_gates  (over a synthesized run directory holding the reports above)

Two trusted records: the synthetic `consistent_toy` (test_replay_synthetic; CPU, snapshots derived from the GENUINE
Program's reference evaluation) and the real B0 capture `data/logs/m1.jsonl.gz` + `data/logs/m1/`
(snapshots of steps 0, 1, 8; HF safetensors; the B0 cos_sin fixture).  Every mutation is a function
`(parts, node, ctx) -> parts | None` installed on ONE genuine pattern; `MutatingPattern` records how often it fired.

    uv run python -m tests.check.adversarial --harness toy --mutation all --out /tmp/adv/toy.json
    uv run python -m tests.check.adversarial --harness m1 --mutation M01 --k 2 --out /tmp/adv/m1_M01.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.refs import Coll as _Coll, concat
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import prims as P
from verity.ml.circuits.types import Array

from verity_vllm.check import operand_provenance as provenance
from verity_vllm.check.replay import replay
from verity_vllm.check.replay import vu_query as vq
from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.fold import FoldResult, fold
from verity_vllm.observe.fold.resolver import Compute, Gather, GatherMany, Group, PerRow, Profile, Shared, Transparent, Unsupported, parts_of

REPO = Path(__file__).resolve().parents[2]
BF16 = P.BF16

# ---------------------------------------------------------------------------------------------------------
# the adapter author's lever: a pattern that rewrites what a genuine pattern returns
# ---------------------------------------------------------------------------------------------------------

Rewrite = Callable[[list, Any, Any, dict], "list | None"]      # (parts, node, ctx, state) -> new parts | None (= leave alone)


def plain(p):
    """Strip `ComputeGroup` / `TransparentGroup` down to the plain dataclass (members are re-emitted as a `Group`)."""
    if isinstance(p, Compute):
        return Compute(p.spec, p.rows, p.operands, p.outputs, p.consumed, p.hint, p.kernels, p.attribution, p.pattern)
    if isinstance(p, Transparent):
        return Transparent(p.effect, p.consumed, p.src, p.dst, p.rows, p.hint, p.kernels, p.pattern)
    return p


class MutatingPattern:
    """Delegates to `inner`; when `rewrite` returns new parts, emits them as a `resolver.Group` (the fold and the census
    read both spellings through `parts_of`).  `hits` counts the rewritten resolutions."""

    def __init__(self, inner, rewrite: Rewrite, state: dict | None = None) -> None:
        self.inner, self.rewrite, self.state = inner, rewrite, state if state is not None else {}
        self.name = inner.name
        self.hits = 0

    def match(self, node, ctx):
        r = self.inner.match(node, ctx)
        if r is None or isinstance(r, Unsupported):
            return r
        parts = [plain(p) for p in parts_of(r)]
        new = self.rewrite(parts, node, ctx, self.state)
        if new is None:
            return r
        self.hits += 1
        return Group(tuple(new), tuple(r.consumed), getattr(r, "hint", ""), getattr(r, "pattern", "") or self.name)


class RecordingPattern:
    """Delegates to `inner`; records every Compute's output views by hint into `state["out"][hint]` so a later
    mutation in the same step can point an operand at another instance's output (what an adapter author can do)."""

    def __init__(self, inner, state: dict) -> None:
        self.inner, self.state, self.name = inner, state, inner.name

    def match(self, node, ctx):
        r = self.inner.match(node, ctx)
        if r is None or isinstance(r, Unsupported):
            return r
        for p in parts_of(r):
            if isinstance(p, Compute):
                self.state.setdefault("out", {})[(ctx.start.step, p.hint.split("/")[0])] = p
        return r


def mutate_profile(profile: Profile, target: str | Callable[[Any], bool], rewrite: Rewrite, *, record: bool = True,
                   pid: str | None = None, weight_field=None) -> tuple[Profile, dict]:
    """A hostile copy of `profile`: the pattern(s) selected by `target` (a name or a predicate) are wrapped in
    `MutatingPattern(rewrite)`, every other pattern in `RecordingPattern` (so rewrites can consult `state`)."""
    state: dict = {"hits": []}
    sel = (lambda p: p.name == target) if isinstance(target, str) else target
    pats = []
    for p in profile.patterns:
        if sel(p):
            mp = MutatingPattern(p, rewrite, state)
            state["hits"].append(mp)
            pats.append(mp)
        else:
            pats.append(RecordingPattern(p, state) if record else p)
    prof = dataclasses.replace(profile, patterns=pats, id=pid or (profile.id + "+adv"),
                               weight_field=weight_field or profile.weight_field)
    return prof, state


def hits(state: dict) -> int:
    return sum(m.hits for m in state["hits"])


# ---------------------------------------------------------------------------------------------------------
# view helpers
# ---------------------------------------------------------------------------------------------------------


def half_cols(v: ViewDesc, lo: int, hi: int) -> ViewDesc:
    """Columns [lo, hi) of a rank-2 view (a sub-view of the same allocation — written by the same producer)."""
    return ViewDesc(v.alloc, v.offset + lo * v.strides[1], (v.shape[0], hi - lo), v.strides, v.dtype, v.device)


def spec_name(spec) -> str:
    return getattr(spec, "id", str(spec)).split("{")[0]


def layer_of(hint: str) -> int | None:
    import re
    m = re.search(r"layers\.(\d+)\.", hint or "")
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------------------------------------
# hostile definitions (what a lane can author in ITS OWN module; NOT under verity_vllm/program/registry/quarantine)
# ---------------------------------------------------------------------------------------------------------


@composite("AdvGemmHalfTiled", 1, ["K", "N", "BAD"],
           lambda S: ((("x", Array(S.K, BF16)), ("w", Array(S.N, Array(S.K, BF16)))), Array(S.N, BF16)),
           conformance="adversarial: NOT a registered semantics")
def AdvGemmHalfTiled(B, S, x, w):
    """Mutation 7/9/10: `x @ W^T` whose FIRST half of the outputs is a batch of `GemmCoordinate` (admissible child units)
    and whose SECOND half is computed through an interior value `t` (`x + x` when BAD=1, `x + 0` when BAD=0), so no
    child unit of the oversize instance ever covers those returns.  BAD=1 makes the un-tiled half WRONG."""
    h = S.N // 2
    a = B.batch(bind(b1.GemmCoordinate, K=S.K), x, w[0:h], axes=(None, 0))
    other = x if S.BAD else _Coll(Array(S.K, BF16), concat([B.call(b1.ZERO16).refs] * S.K))
    t = B.batch(P.Bf16Add, x, other, axes=(0, 0))                    # interior [K]
    c = B.batch(bind(b1.GemmCoordinate, K=S.K), t, w[h:S.N], axes=(None, 0))
    return _Coll(Array(S.N, BF16), concat([a.refs, c.refs]))


@composite("AdvGemmZeroMass", 1, ["K", "N"],
           lambda S: ((("x", Array(S.K, BF16)), ("w", Array(S.N, Array(S.K, BF16)))), Array(S.N, BF16)),
           conformance="adversarial: NOT a registered semantics")
def AdvGemmZeroMass(B, S, x, w):
    """Mutation 8: every output coordinate is computed from an interior value, so an oversize instance has NO
    admissible child unit (zero mass)."""
    t = B.batch(P.Bf16Add, x, _Coll(Array(S.K, BF16), concat([B.call(b1.ZERO16).refs] * S.K)), axes=(0, 0))
    return B.batch(bind(b1.GemmCoordinate, K=S.K), t, w, axes=(None, 0))


BAKED: dict[int, tuple[int, ...]] = {}       # TAG -> the observed output words the hostile body returns


@composite("AdvBakedOutput", 1, ["N", "TAG"], lambda S: ((("x", Array(S.N, BF16)),), Array(S.N, BF16)),
           conformance="adversarial: NOT a registered semantics")
def AdvBakedOutput(B, S, x):
    """Mutation 6b: a body that ignores its operand and returns constants — the bytes the runtime was observed to
    produce (one specialization per instance via TAG, so every instance is its own stratum and tier a evaluates it)."""
    words = BAKED[S.TAG]
    return _Coll(Array(S.N, BF16), concat([B.call(b1.const(16, int(w))).refs for w in words]))


# ---------------------------------------------------------------------------------------------------------
# the mutation catalogue (each: (name, target pattern predicate, rewrite) built for a given record)
# ---------------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class Mutation:
    id: str
    title: str
    injection: str                           # one line: where/how the adapter author injects it
    target: Callable[[Any], bool]
    rewrite: Rewrite
    weight_field: Any = None                 # a hostile weight-name map instead of / in addition to the rewrite
    max_arg_leaves: int | None = None        # tier a budget this mutation needs (child descent)
    notes: str = ""


def _is(name_sub: str):
    return lambda p: name_sub in p.name


def _compute_with(parts, pred):
    return [i for i, p in enumerate(parts) if isinstance(p, Compute) and pred(p)]


def catalogue(*, layer: int, o_hint: str, big_hint: str, ln1_hint: str, alt_hint: str, silu_hint: str, attn_hint: str, gemm_pat: str, rms_pat: str,
              silu_pat: str, attn_pat: str, equal_rows: tuple[int, int] | None, genuine_weight_field, swap_fields,
              toy: bool) -> dict[str, Mutation]:
    """`layer`: the layer the single-instance mutations hit; `*_hint`: the module paths of that layer's o_proj /
    input_layernorm / silu / attention resolutions; `*_pat`: the genuine pattern names; `equal_rows`: two prefill rows
    whose token ids are equal (mutation 11); `swap_fields`: (name_a, name_b) two same-shaped weight Roots."""
    M: dict[str, Mutation] = {}

    # (1) wrong producer: o_proj's activation operand := the LAST output of this layer's input_layernorm resolution
    #     (toy: the normed activation; B0: the fused norm's residual stream — a buffer still ALIVE at o_proj; a freed
    #     buffer is an `Unresolved` read, see the report) — same [M,H] bf16 shape, written, a *different* instance's output.
    def m01(parts, node, ctx, st):
        idx = _compute_with(parts, lambda c: c.hint.startswith(o_hint) and spec_name(c.spec) == "Gemm_v1")
        if not idx:
            return None
        alt = (st.get("out") or {}).get((ctx.start.step, alt_hint))
        if alt is None:
            return None
        nv = alt.outputs[-1].view
        c = parts[idx[0]]
        a = c.operands[0].view
        if nv.shape != a.shape or nv.dtype != a.dtype:
            return None
        parts[idx[0]] = dataclasses.replace(c, operands=(PerRow(nv),) + c.operands[1:])
        return parts
    M["M01"] = Mutation("M01", "wrong producer (same shape/dtype)",
                        f"pattern `{gemm_pat}`: layer-{layer} o_proj operand[0] := `{alt_hint}` last output view", _is(gemm_pat), m01)

    # (2) wrong-but-written runtime slice: attention K operand := KV slots (1..T-1, T-1) instead of (0..T-1) — every slot
    #     written this step or earlier, so provenance finds a writer for every leaf.
    def m02(parts, node, ctx, st):
        out = None
        for i, p in enumerate(parts):
            if isinstance(p, Compute) and spec_name(p.spec).startswith("Attention") and p.hint.startswith(attn_hint):
                k = p.operands[1]
                if not isinstance(k, GatherMany):
                    continue
                new_idx = tuple((s[1:] + (s[-1],)) if len(s) >= 2 else s for s in k.indices)
                if new_idx == k.indices:
                    continue
                parts[i] = dataclasses.replace(p, operands=(p.operands[0], GatherMany(k.view, new_idx), p.operands[2]))
                out = parts
        return out
    M["M02"] = Mutation("M02", "wrong-but-written runtime slice",
                        f"pattern `{attn_pat}`: layer-{layer} K operand GatherMany slots (0..T-1) -> (1..T-1, T-1)", _is(attn_pat), m02)

    # (3) swallowed runtime operation: silu_and_mul := Transparent(move gate-half(gu) -> act) claiming the same kernel.
    def m03(parts, node, ctx, st):
        idx = _compute_with(parts, lambda c: spec_name(c.spec) == "SiluMul_v1" and c.hint.startswith(silu_hint))
        if not idx:
            return None
        c = parts[idx[0]]
        gu = c.operands[0].view
        I = c.outputs[0].view.shape[1]
        parts[idx[0]] = Transparent("move", c.consumed, PerRow(half_cols(gu, 0, I)), c.outputs[0], c.rows, c.hint,
                                    c.kernels, c.pattern)
        return parts
    M["M03"] = Mutation("M03", "swallowed runtime operation (Compute -> Transparent move)",
                        f"pattern `{silu_pat}`: layer-{layer} SiluMul Compute := Transparent('move', gu[:, :I] -> act), kernels kept",
                        _is(silu_pat), m03)

    # (3b) swallowed with NO rewiring: silu_and_mul := Transparent('noop').
    def m03b(parts, node, ctx, st):
        idx = _compute_with(parts, lambda c: spec_name(c.spec) == "SiluMul_v1" and c.hint.startswith(silu_hint))
        if not idx:
            return None
        c = parts[idx[0]]
        parts[idx[0]] = Transparent("noop", c.consumed, None, None, 1, c.hint, c.kernels, c.pattern)
        return parts
    M["M03b"] = Mutation("M03b", "swallowed runtime operation (Compute -> Transparent noop, consumer reads stale buffer)",
                         f"pattern `{silu_pat}`: layer-{layer} SiluMul Compute := Transparent('noop')", _is(silu_pat), m03b)

    # (4) spurious dependency on an existing spec: an extra operand.
    def m04(parts, node, ctx, st):
        idx = _compute_with(parts, lambda c: spec_name(c.spec) == "SiluMul_v1" and c.hint.startswith(silu_hint))
        if not idx:
            return None
        c = parts[idx[0]]
        parts[idx[0]] = dataclasses.replace(c, operands=c.operands + (c.operands[0],))
        return parts
    M["M04"] = Mutation("M04", "spurious dependency (extra operand on an existing spec)",
                        f"pattern `{silu_pat}`: layer-{layer} SiluMul operands += (gu,)", _is(silu_pat), m04)

    # (5) weight-root substitution: two same-shaped parameters swapped in the adapter's weight-name map.
    a_name, b_name = swap_fields
    def swapped_weight_field(name: str):
        if name == a_name:
            return genuine_weight_field(b_name)
        if name == b_name:
            return genuine_weight_field(a_name)
        return genuine_weight_field(name)
    M["M05"] = Mutation("M05", "weight-root substitution (same-shaped parameters)",
                        f"profile.weight_field: `{a_name}` <-> `{b_name}`", lambda p: False, lambda parts, node, ctx, st: None,
                        weight_field=swapped_weight_field)

    # (6a) intermediate smuggled as a "root": the o_proj weight operand := the layer's input_layernorm OUTPUT ... shapes
    #      differ ([N,K] vs [M,H]) for every real spec, so the lever is (6b): bake the observed output as constants.
    def m06(parts, node, ctx, st):
        idx = _compute_with(parts, lambda c: spec_name(c.spec) == "SiluMul_v1" and c.hint.startswith(silu_hint))
        if not idx:
            return None
        c = parts[idx[0]]
        if c.rows != 1:
            return None                                              # one row per instance keeps the constant table small
        I = c.outputs[0].view.shape[1]
        tag = int(node.id)
        words = BAKED.get(tag)
        if words is None:
            return None                                              # the record (snapshot) was not consulted for this event
        parts[idx[0]] = dataclasses.replace(c, spec=bind(AdvBakedOutput, N=I, TAG=tag),
                                            operands=(PerRow(half_cols(c.operands[0].view, 0, I)),))
        return parts
    M["M06"] = Mutation("M06", "output smuggled as a Definition constant (per-instance specialization)",
                        f"pattern `{silu_pat}`: layer-{layer} decode SiluMul := AdvBakedOutput{{N,TAG=event}} whose body returns the observed bytes",
                        _is(silu_pat), m06, notes="requires the adversary to have read the snapshot store (they hold the run dir)")

    # (7)/(9)/(10) partial VU tiling: o_proj Gemm := AdvGemmHalfTiled (first half tiled by admissible GemmCoordinate
    #      children, second half through an interior value; BAD=1 makes the un-tiled half wrong).
    def m07(bad):
        def rw(parts, node, ctx, st):
            idx = _compute_with(parts, lambda c: c.hint.startswith(big_hint) and spec_name(c.spec) == "Gemm_v1")
            if not idx:
                return None
            c = parts[idx[0]]
            K, N = c.spec.statics["K"], c.spec.statics["N"]
            parts[idx[0]] = dataclasses.replace(c, spec=bind(AdvGemmHalfTiled, K=K, N=N, BAD=bad))
            return parts
        return rw
    M["M07"] = Mutation("M07", "partial VU tiling (un-tiled half correct)",
                        f"pattern `{gemm_pat}`: `{big_hint}` Gemm_v1 := AdvGemmHalfTiled{{BAD=0}}", _is(gemm_pat), m07(0),
                        max_arg_leaves=100 if toy else None)
    M["M09"] = Mutation("M09", "biased child-VU decomposition (tiled half right, un-tiled half WRONG)",
                        f"pattern `{gemm_pat}`: `{big_hint}` Gemm_v1 := AdvGemmHalfTiled{{BAD=1}}", _is(gemm_pat), m07(1),
                        max_arg_leaves=100 if toy else None)

    # (8) zero-mass instance.
    def m08(parts, node, ctx, st):
        idx = _compute_with(parts, lambda c: c.hint.startswith(big_hint) and spec_name(c.spec) == "Gemm_v1")
        if not idx:
            return None
        c = parts[idx[0]]
        K, N = c.spec.statics["K"], c.spec.statics["N"]
        parts[idx[0]] = dataclasses.replace(c, spec=bind(AdvGemmZeroMass, K=K, N=N))
        return parts
    M["M08"] = Mutation("M08", "zero-mass instance (oversize, no admissible child unit)",
                        f"pattern `{gemm_pat}`: `{big_hint}` Gemm_v1 := AdvGemmZeroMass", _is(gemm_pat), m08,
                        max_arg_leaves=100 if toy else None)

    # (11) equal bytes, wrong producer: the layer-0 input_layernorm reads prefill row a's embedding for row b where
    #      token[a] == token[b] (Embedding rows are the same table row => byte-identical outputs of DIFFERENT instances).
    if equal_rows is not None:
        ra, rb = equal_rows
        def m11(parts, node, ctx, st):
            idx = _compute_with(parts, lambda c: spec_name(c.spec).startswith("RMSNorm") and c.hint.startswith(ln1_hint))
            if not idx or ctx.start.step != 0:
                return None
            c = parts[idx[0]]
            x = c.operands[0]
            if not isinstance(x, PerRow) or x.view.shape[0] <= max(ra, rb):
                return None
            order = list(range(x.view.shape[0]))
            order[rb] = ra
            parts[idx[0]] = dataclasses.replace(c, operands=(Gather(x.view, tuple(order)),) + c.operands[1:])
            return parts
        M["M11"] = Mutation("M11", "two runtime values with equal bytes: operand pointed at the wrong producer",
                            f"pattern `{rms_pat}`: step-0 layer-0 input_layernorm row {rb} reads embedding row {ra} (token[{ra}] == token[{rb}])",
                            _is(rms_pat), m11)
    return M


# ---------------------------------------------------------------------------------------------------------
# the checkers, exactly as the sealed CPU phase runs them (in-process)
# ---------------------------------------------------------------------------------------------------------


def run_checkers(result: FoldResult, *, store, weights, log_path=None, seed: int = 0, k: int = 2, max_arg_leaves: int = 4_000_000,
                 chain_n: int = 3, root_policy_fn: Callable[[], dict] | None = None, census_fn: Callable[[], dict] | None = None,
                 tier_a_steps=None) -> dict:
    out: dict[str, Any] = {}
    t0 = time.time()
    out["provenance"] = provenance.check(result, log_path)
    out["tier_a"] = replay.tier_a(result, store, weights, k=k, seed=seed, max_arg_leaves=max_arg_leaves, steps=tier_a_steps)
    try:
        out["tier_chain"] = replay.tier_chain(result, store, weights, n=chain_n, seed=seed, max_arg_leaves=max_arg_leaves, steps=tier_a_steps)
    except Exception as e:                                   # a mutated Program may break the chain planner
        out["tier_chain"] = {"status": "error", "error": repr(e)}
    if root_policy_fn is not None:
        out["root_policy"] = root_policy_fn()
    if census_fn is not None:
        out["census"] = census_fn()
    out["seconds"] = round(time.time() - t0, 2)
    return out


def summarize(rep: dict) -> dict:
    """The verdict-bearing lines of each checker (what the report table quotes)."""
    s: dict[str, Any] = {}
    p = rep.get("provenance")
    if p:
        s["provenance"] = {"status": p["status"], "failures": p["failures"], "n_mismatch": p["n_mismatch"], "uncovered": p["uncovered"],
                           "schema": {k: (p.get("schema") or {}).get(k) for k in ("writes_outside_schema", "declared_unwritten")}}
    a = rep.get("tier_a")
    if a:
        fails = [e for e in a.get("instances", []) if e["status"] == "fail"]
        s["tier_a"] = {"status": a["status"], "reason": a.get("reason"), "uncovered_specs": a.get("uncovered_specs"),
                       "n_fail": len(fails), "fail_examples": [{k: e.get(k) for k in ("name", "spec", "granularity", "n_diff", "reason")} for e in fails[:4]],
                       "served_tokens": (a.get("served_tokens") or {}).get("status"),
                       "vu_population": {k: (a.get("vu_population") or {}).get(k) for k in ("status", "partial_tiling_specs", "zero_mass_specs", "count", "oversize_instances")}}
    c = rep.get("tier_chain")
    if c:
        s["tier_chain"] = {"status": c.get("status"), "reason": c.get("reason") or c.get("error"),
                           "fails": [sl.get("reason") or sl.get("failures") for sl in c.get("slices", []) if sl.get("status") == "fail"][:3]}
    r = rep.get("root_policy")
    if r:
        s["root_policy"] = {"status": r["status"], "first_failures": r.get("first_failures", [])[:3], "problems": r.get("problems", [])[:3]}
    ce = rep.get("census")
    if ce:
        s["census"] = {"residue_zero": ce.get("residue_zero"), "residue_names": (ce.get("totals") or {}).get("residue_names")}
    return s


# ---------------------------------------------------------------------------------------------------------
# harness 1: the synthetic consistent toy (CPU, snapshots derived from the GENUINE Program)
# ---------------------------------------------------------------------------------------------------------


class ToyRecord:
    """The trusted record of the toy: a consistent log, snapshots of the GENUINE fold's transcript, toy weights."""

    def __init__(self, snapdir: Path, prompt: list[int] | None = None) -> None:
        from tests.observe import test_fold_synthetic as tfs
        from tests.check.test_compare_synthetic import ToyWeights, _evaluate_program, _write_snapshots
        self.tfs = tfs
        if not tfs._real_memory_available():
            tfs.install_stub_memory()
        self._orig = (list(tfs.SAMPLED), list(tfs.PROMPT))
        if prompt is not None:
            tfs.PROMPT[:] = prompt
        r = fold(tfs.toy_log(), tfs.toy_profile(), strict=True)
        weights = ToyWeights(r.params[-1][1].leaves)
        t0 = self._tokens(r, weights, _evaluate_program)[0]
        tfs.SAMPLED[:] = [t0, tfs.SAMPLED[1]]
        r = fold(tfs.toy_log(), tfs.toy_profile(), strict=True)
        t1 = self._tokens(r, weights, _evaluate_program)[1]
        tfs.SAMPLED[:] = [t0, t1]
        r = fold(tfs.toy_log(), tfs.toy_profile(), strict=True)
        t = _evaluate_program(r, weights)
        snapdir.mkdir(parents=True, exist_ok=True)
        _write_snapshots(r, t, snapdir)
        from verity_vllm.check.fold_compare import SnapshotStore
        import copy
        self.genuine, self.weights, self.transcript = r, weights, t
        self.store = SnapshotStore(snapdir)
        self.events = copy.deepcopy(tfs.toy_log())      # the toy header references the module's live PROMPT list
        self.prompt = list(tfs.PROMPT)
        self.profile = tfs.toy_profile()
        self.snapdir = snapdir
        self.restore()                                    # the record is complete; hand the module globals back

    @staticmethod
    def _tokens(r, weights, ev):
        t = ev(r, weights)
        return [int(t[g]) for g in r.program.circuit.root_outputs()]

    def restore(self) -> None:
        self.tfs.SAMPLED[:], self.tfs.PROMPT[:] = self._orig

    def fold(self, profile: Profile) -> FoldResult:
        return fold(self.events, profile, strict=True)

    def weight_source(self, result: FoldResult):
        return self.weights

    def catalogue(self) -> dict[str, Mutation]:
        tfs = self.tfs
        eq = None
        if len(self.prompt) >= 2 and self.prompt[0] == self.prompt[1]:
            eq = (0, 1)
        return catalogue(layer=0, o_hint="model.layers.0.self_attn.o_proj", big_hint="model.layers.0.self_attn.o_proj", ln1_hint="model.layers.0.input_layernorm", alt_hint="model.layers.0.input_layernorm",
                         silu_hint="model.layers.0.mlp.act_fn", attn_hint="model.layers.0.self_attn.attn", gemm_pat="mm",
                         rms_pat="rms_triton", silu_pat="silu", attn_pat="attention", equal_rows=eq,
                         genuine_weight_field=tfs.weight_field,
                         swap_fields=("model.layers.0.input_layernorm.weight", "model.layers.0.post_attention_layernorm.weight"), toy=True)

    def bake(self) -> None:
        """Mutation 6b needs the observed outputs: read them from the trusted record's snapshots (the adversary holds the run dir)."""
        for m in self.genuine.instances:
            if spec_name(self.genuine.body.nodes[m["node"]].fn) == "SiluMul_v1" and m["step"] > 0:
                call_outs = [self.transcript[g] for g in self._returns(m["node"])]
                BAKED[int(m["event"])] = tuple(int(w) for w in call_outs)

    def _returns(self, node: int):
        from verity.ml.circuits.layout import FunctionCall
        circuit = self.genuine.program.circuit
        n_in = len(self.genuine.program.root.body.nodes) - len(self.genuine.body.nodes)
        return FunctionCall(circuit, circuit.root_scope, n_in + node).returns

    def root_policy(self, profile: Profile) -> Callable[[], dict]:
        return lambda: {"status": "n/a", "note": "toy log has no RunHeader weights identity; root policy is exercised on M1"}


# ---------------------------------------------------------------------------------------------------------
# harness 2: the real B0 capture (M1 log + snapshots of steps 0/1/8 + HF safetensors + cos_sin fixture)
# ---------------------------------------------------------------------------------------------------------

M1_LOG = REPO / "data" / "logs" / "m1.jsonl.gz"
M1_SNAPS = REPO / "data" / "logs" / "m1"
M1_CENSUS = REPO / "data" / "logs" / "l5" / "census.json"
B0_PROFILE = "vllm_d9105ea80_sm89_eager"


class M1Record:
    def __init__(self) -> None:
        from verity_vllm.check.fold_compare import DEFAULT_COS_SIN, SnapshotStore, WeightSource, load_profile
        from verity_vllm.observe.log import read_log
        self.profile = load_profile(B0_PROFILE)
        self.events = list(read_log(M1_LOG))
        self.store = SnapshotStore(M1_SNAPS)
        self.cos_sin = DEFAULT_COS_SIN
        self._ws = WeightSource
        self.header = self.events[0]
        self.prompt = self.header.workload["requests"][0]["prompt_token_ids"]

    def fold(self, profile: Profile) -> FoldResult:
        return fold(self.events, profile, strict=True)

    def weight_source(self, result: FoldResult):
        return self._ws(result.params[-1][1], None, self.cos_sin, repo=self.header.model["repo"])

    def equal_rows(self) -> tuple[int, int] | None:
        seen: dict[int, int] = {}
        for i, t in enumerate(self.prompt):
            if t in seen:
                return (seen[t], i)
            seen[t] = i
        return None

    def catalogue(self, layer: int = 7) -> dict[str, Mutation]:
        from verity_vllm.engine.profiles.vllm_d9105ea80_sm89_eager import make_weight_field
        wf = self.profile.weight_field
        return catalogue(layer=layer, o_hint=f"model.layers.{layer}.self_attn.o_proj", big_hint="logits_processor", ln1_hint="model.layers.0.input_layernorm", alt_hint=f"model.layers.{layer}.input_layernorm",
                         silu_hint=f"model.layers.{layer}.mlp.act_fn", attn_hint=f"model.layers.{layer}.self_attn.attn",
                         gemm_pat="gemm", rms_pat="rms_norm", silu_pat="act.silu", attn_pat="attention", equal_rows=self.equal_rows(),
                         genuine_weight_field=wf,
                         swap_fields=("model.layers.3.self_attn.qkv_proj.weight", "model.layers.5.self_attn.qkv_proj.weight"), toy=False)

    def root_policy(self, profile: Profile) -> Callable[[], dict]:
        from verity_vllm.check import root_policy
        h = self.header
        events = [dataclasses.replace(h, notes={**h.notes, "weights_verified": True})] + self.events[1:]
        return lambda: root_policy.evaluate(events, M1_SNAPS, profile, cos_sin=self.cos_sin, weight_bytes=False)

    def census(self, profile: Profile) -> Callable[[], dict] | None:
        """census.reconcile over the HOSTILE profile (it loads profiles by module name; the hostile one lives in memory)."""
        if not M1_CENSUS.exists():
            return None
        from verity_vllm.properties import census
        from verity_vllm.observe.fold import resolve_log

        def run() -> dict:
            orig = resolve_log.load_profile
            resolve_log.load_profile = lambda name: (profile, None)
            try:
                return census.reconcile(json.loads(M1_CENSUS.read_text()), str(M1_LOG), profile.id)
            finally:
                resolve_log.load_profile = orig
        return run

    def bake(self, layer: int) -> FoldResult:
        """Mutation 6b on the real record: the observed SiluMul outputs of `layer` at the snapshot decode steps, read
        from the trusted snapshot store through the GENUINE fold (the adversary holds the run directory)."""
        g = self.fold(self.profile)
        rep = replay.Replayer(g, self.store, self.weight_source(g), chain=False)
        for m in g.instances:
            if m["spec"].startswith("SiluMul_v1") and f"layers.{layer}/" in m["name"] and m["step"] in self.store.steps and m["step"] > 0:
                BAKED[int(m["event"])] = tuple(int(w) for w in rep.expected_words(m))
        return g


# ---------------------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------------------


def strata(result: FoldResult) -> dict[str, dict[str, int]]:
    """spec -> regime -> instance count (the tier-a strata; the miss probability of an n-instance mutation inside a
    stratum of size S under k draws without replacement is C(S-n, k)/C(S, k) ~ (1 - n/S)^k)."""
    out: dict[str, dict[str, int]] = {}
    for m in result.instances:
        spec = result.body.nodes[m["node"]].fn.id
        reg = "prefill" if m["step"] == 0 else "decode"
        out.setdefault(spec, {}).setdefault(reg, 0)
        out[spec][reg] += 1
    return out


def miss_probability(S: int, n: int, k: int) -> float:
    from math import comb
    if n >= S or k >= S:
        return 0.0 if n > 0 else 1.0
    return comb(S - n, k) / comb(S, k)


def run_one(record, mut: Mutation, *, k: int, seed: int, max_arg_leaves: int, chain_n: int = 3, log_path=None,
            census: bool = False, tier_a_steps=None) -> dict:
    prof, state = mutate_profile(record.profile, mut.target, mut.rewrite, weight_field=mut.weight_field)
    census_fn = record.census(prof) if census and hasattr(record, "census") else None
    t0 = time.time()
    try:
        result = record.fold(prof)
    except Exception as e:
        return {"mutation": mut.id, "title": mut.title, "injection": mut.injection, "hits": hits(state),
                "fold": {"status": "error", "error": f"{type(e).__name__}: {e}"}, "classification": "deterministically rejected (fold)"}
    budget = mut.max_arg_leaves or max_arg_leaves
    weights = record.weight_source(result)
    rep = run_checkers(result, store=record.store, weights=weights, log_path=log_path, seed=seed, k=k, max_arg_leaves=budget,
                       chain_n=chain_n, root_policy_fn=record.root_policy(prof), census_fn=census_fn, tier_a_steps=tier_a_steps)
    s = summarize(rep)
    s["fold"] = {"status": "ok", "digest": result.digest, "instances": len(result.instances), "seconds": round(time.time() - t0, 2)}
    return {"mutation": mut.id, "title": mut.title, "injection": mut.injection, "hits": hits(state), "summary": s, "strata": strata(result),
            "full": rep}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--harness", choices=("toy", "m1"), default="toy")
    ap.add_argument("--mutation", default="all", help="comma-separated ids or 'all'")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer", type=int, default=7, help="m1: the layer single-instance mutations hit")
    ap.add_argument("--max-arg-leaves", type=int, default=4_000_000)
    ap.add_argument("--steps", default=None, help="m1: tier a steps (comma-separated) to bound memory")
    ap.add_argument("--census", action="store_true", help="m1: also run census.reconcile with the genuine profile module (kernels are kept)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default="/tmp/adv")
    a = ap.parse_args(argv)
    scratch = Path(a.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    rows = []
    if a.harness == "toy":
        rec = ToyRecord(scratch / "toy_snaps")
        try:
            cat = rec.catalogue()
            rec.bake()
            ids = list(cat) if a.mutation == "all" else a.mutation.split(",")
            for mid in ids:
                if mid == "M11" and "M11" not in cat:
                    continue
                r = run_one(rec, cat[mid], k=a.k, seed=a.seed, max_arg_leaves=a.max_arg_leaves)
                rows.append(r)
                print(json.dumps({k: v for k, v in r.items() if k != "full"}, default=str)[:1500], flush=True)
            if a.mutation == "all" or "M11" in a.mutation.split(","):
                rec.restore()
                rec2 = ToyRecord(scratch / "toy_snaps_eq", prompt=[5, 5])
                try:
                    cat2 = rec2.catalogue()
                    r = run_one(rec2, cat2["M11"], k=a.k, seed=a.seed, max_arg_leaves=a.max_arg_leaves)
                    rows.append(r)
                    print(json.dumps({k: v for k, v in r.items() if k != "full"}, default=str)[:1500], flush=True)
                finally:
                    rec2.restore()
        finally:
            rec.restore()
    else:
        rec = M1Record()
        cat = rec.catalogue(a.layer)
        ids = list(cat) if a.mutation == "all" else a.mutation.split(",")
        steps = [int(s) for s in a.steps.split(",")] if a.steps else None
        if "M06" in ids:
            rec.bake(a.layer)
        for mid in ids:
            r = run_one(rec, cat[mid], k=a.k, seed=a.seed, max_arg_leaves=a.max_arg_leaves, log_path=str(M1_LOG),
                        census=a.census, tier_a_steps=steps)
            rows.append(r)
            print(json.dumps({k: v for k, v in r.items() if k not in ("full", "strata")}, default=str)[:2000], flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=1, default=str) + "\n")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
