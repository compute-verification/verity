"""X-03 per-layer attention plane classes: the expected FA2/FA3 hidden-plane inventory of a step is DERIVED per attention
layer from the Program's attention Call statics x the step's active-sequence set; uniformity across layers is a consequence, never a rule.

The seven questions for the Value `fa2.m1/L<k>` (the FA3 twin `fa3.m1/L<k>` differs only in the launch geometry):
 1. What Values: the M1 export stream of attention layer k's launch in engine step s -- the hidden rows R5-R8 (S scores, P weights, RP
    bf16 P fragments, ROW max/rescale/lane sums, O accumulators, FIN epilogue) of that layer's `Attention_v3{T,NH,KVH,D,BN}` /
    `AttentionSoftcap_v1{..,CAP}` Calls at the step's positions: the AttnBlock interiors (s[c] dots, PV chains, max chain).
 2. Why the query requires them: Q_module_v1's attention override names those interiors (`required_manifest` "attention override"); a
    module-boundary value does not determine them, so the hidden family is required in full, per layer per step.
 3. Runtime occurrence: ONE `varlen_fwd` launch per attention layer per engine step over all co-scheduled sequences; the kernel picks
    its geometry by its own rule (`seqlenq_ngroups_swapped`, flash_api.cpp:646): a decode launch (max_seqlen_q == 1) of a GQA layer
    (NH > KVH) folds the query heads into rows ONLY without a sliding window -- the window is a per-layer static vLLM passes on every
    launch of that layer (`attention_impls_observed[layer].sliding_window`, the fact `attn_kind_for` records as the Call's `inert_window`).
 4. Acquisition: the tap writes six planes into the declared buffer; `native_collect.cpp Fa2Wrap::call` stages the WHOLE buffer as one
    tensor (`g<HB>x<M>x<NB>[x<BN>x<D>]`) or, on the compacted path (b >= 2 sequences, un-swapped), six per-sequence rectangles
    (`s<i>/g<h>x<sl>x<NB>[..]/{S,P,RP,ROW,O,FIN}`) -- so a step's plane class per layer is (kernel rule) x (staging rule) x (active set).
 5. Occurrence resolution: layer k = the step's k-th launch (`launches_in_step`); sequence i = the global index in `cu_seqlens_q`
    order = the step's row order (`step_rows`): (request, seqlen_q) per active sequence.
 6. Binding: the name splits into (op_path `fa2.m1/L<k>`, member) -> the manifest identity of that layer at that engine step
    (`verity_vllm.query.manifest.bind`; at B>1 coverage holds the op's tensors at the step to the launch's whole stream, each tensor
    the words its name's geometry declares, `format.batch_launch_shortfall`); leaves are position-identified in the packed step stream.
 7. Independent check (this module): `check_step` recomputes each layer's class and count from the declarations and the active set and
    names every layer whose observed inventory disagrees; the old uniformity rule is kept as `uniform_rule` for runs without declarations.
No model names anywhere: a layer is described only by its statics (NH, KVH, D, window); classes are derived, never tabulated.
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

PLANES = ("S", "P", "RP", "ROW", "O", "FIN")          # native_collect.cpp `name_id_seq` planes[6]; hidden_stream.StreamLayout order
PLANES_MS = PLANES + ("MS",)                           # a build that writes the MS class (`LayoutPolicy.ms`): the seventh plane
PLANES_PER_SEQ = len(PLANES)
TAP_PREFIXES = ("fa2.m1/L", "fa3.m1/L")
WHOLE, PER_SEQ, NONE = "whole", "per_seq", "none"

_NAME_RE = re.compile(r"^(?P<tap>fa[23])\.m1/L(?P<layer>\d+)/(?P<rest>.+)$")
_GEOM_RE = re.compile(r"^g(?P<a>\d+)x(?P<b>\d+)x(?P<nb>\d+)(?:x(?P<bn>\d+)x(?P<d>\d+))?$")


# ---------------------------------------------------------------------------------------------------------------------------------
# declarations: what the Program says about each attention layer
# ---------------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class AttnLayerDecl:
    """The attention Call statics of one layer: `Attention_v3{T,NH,KVH,D,BN}`'s NH/KVH/D and the Call's `inert_window` (None = no window;
    W = vLLM passes `window_size=(W-1, 0)` on every launch of the layer).  `label` is free text for reports (never read by the rule)."""
    layer: int
    NH: int
    KVH: int
    D: int
    window: int | None = None
    label: str = ""

    def swaps_on_decode(self) -> bool:
        """The kernel's own fold rule (mirror of `hidden_source.fa2_swapped` / flash_api.cpp:646) restricted to the per-layer statics:
        a decode launch of this layer runs the swapped kernel iff GQA and no window (dropout 0 / alibi None / D % 8 are launch-invariant pins)."""
        return self.NH > self.KVH and self.window is None and self.D % 8 == 0


def _layer_index(name: str) -> int | None:
    m = re.search(r"\.layers\.(\d+)\.", name + ".")
    return int(m.group(1)) if m else None


def _window_of(v: Any) -> int | None:
    """`sliding_window` as observed on the impl: None / -1 / [-1, -1] -> no window; [W-1, 0] or W -> a window (the value is not read by the class)."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        if not v:
            return None
        w = v[0]
    else:
        w = v
    try:
        w = int(w)
    except (TypeError, ValueError):
        return None
    return None if w < 0 else w


def declarations_from_attention_impls(impls: Mapping[str, Mapping[str, Any]]) -> list[AttnLayerDecl]:
    """Per-layer declarations from the Build record's `attention_impls_observed` ({layer_name: {num_heads, num_kv_heads, head_size,
    sliding_window, ..}}, `vllm_meta.attention_impls`): the same per-layer facts the attention rule selected each Call's Definition and
    `inert_window` from.  Entries without a `.layers.<k>.` index or without the three head statics are skipped (not guessed)."""
    out: dict[int, AttnLayerDecl] = {}
    for name, obs in impls.items():
        k = _layer_index(str(name))
        if k is None or not isinstance(obs, Mapping):
            continue
        nh, kvh, d = obs.get("num_heads"), obs.get("num_kv_heads"), obs.get("head_size")
        if nh is None or kvh is None or d is None:
            continue
        out[k] = AttnLayerDecl(k, int(nh), int(kvh), int(d), _window_of(obs.get("sliding_window")), label=str(name))
    return [out[k] for k in sorted(out)]


def attention_impls_of_record(program_dir: str | None = None, programs_root: str | None = None) -> tuple[dict | None, str | None]:
    """The Build record's `attention_impls_observed` for a Commit: `<program_dir>/result.json` (B=1) or the first `build_request*/result.json`
    under `programs_root` that carries it (B>=2: every component Program of a workload is the same model; the record is per layer, not per shape).
    -> (impls, path) or (None, None).  json only (no torch)."""
    cands: list[str] = []
    if program_dir:
        cands.append(os.path.join(str(program_dir), "result.json"))
    if programs_root:
        cands += sorted(glob.glob(os.path.join(str(programs_root), "build_request*", "result.json")))
    for p in cands:
        try:
            with open(p) as f:
                res = json.load(f)
        except (OSError, ValueError):
            continue
        obs = res.get("attention_impls_observed") if isinstance(res, dict) else None
        if isinstance(obs, dict) and obs:
            return obs, p
    return None, None


# ---------------------------------------------------------------------------------------------------------------------------------
# the step's active-sequence set and the collector's staging policy
# ---------------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class StepSequences:
    """The step's active sequences in launch (row) order: `seqlen_q[i]` = query rows of sequence i this step (prefill: its scheduled
    tokens; decode: 1).  `exact` False = only (n_seq, total rows) were known (per-sequence geometry is then not compared)."""
    seqlen_q: tuple[int, ...]
    exact: bool = True

    @property
    def n_seq(self) -> int:
        return len(self.seqlen_q)

    @property
    def max_seqlen_q(self) -> int:
        return max(self.seqlen_q, default=0)

    @property
    def active(self) -> tuple[int, ...]:
        return tuple(i for i, sl in enumerate(self.seqlen_q) if sl > 0)

    @classmethod
    def from_rows(cls, rows: Iterable[tuple[Any, int, int]]) -> "StepSequences":
        """From the collector's `step_rows[step]` = [(request_id, row_lo, row_hi), ..] (launch order)."""
        return cls(tuple(int(hi) - int(lo) for _rid, lo, hi in rows), exact=True)

    @classmethod
    def from_counts(cls, n_seq: int, tokens: int) -> "StepSequences":
        """From (co-scheduled sequences, token rows) only: all-ones when tokens == n_seq (a pure decode step), else an inexact split
        (one sequence carries the surplus) -- enough to decide every layer's class, not its per-sequence geometry."""
        n_seq, tokens = int(n_seq), int(tokens)
        if n_seq <= 0:
            return cls((), exact=True)
        if tokens == n_seq:
            return cls((1,) * n_seq, exact=True)
        return cls((max(1, tokens - (n_seq - 1)),) + (1,) * (n_seq - 1), exact=False)


@dataclass(frozen=True)
class LayoutPolicy:
    """The collector's staging rule (a collector fact, not a model fact): FA2 per-sequence compaction of multi-sequence un-swapped launches
    (`VERITY_FA2_COMPACT`, native_collect.cpp `compacted_path = (b >= 2 || in_split) && compact && !swapped`); FA3 stages whole launches."""
    backend: str = "fa2"                                   # "fa2" | "fa3"
    compact: bool = True
    ms: bool = False                                       # the tap build writes the MS class (hidden_source.tap_ms)


@dataclass(frozen=True)
class PlaneClass:
    """The derived expectation of one layer at one step."""
    kind: str                                              # WHOLE | PER_SEQ | NONE
    count: int                                             # tensors the layer must carry
    slabs: int                                             # WHOLE: HB of the launch; PER_SEQ: h per sequence (rows of a plane rectangle)
    rows: int                                              # WHOLE: M (row pitch); PER_SEQ: 0 (per-sequence sl, see `seqlen_q`)
    swapped: bool
    why: str
    planes: tuple[str, ...] = PLANES                       # PER_SEQ: the planes every active sequence carries

    def as_dict(self) -> dict[str, Any]:
        return {"class": self.kind, "count": self.count, "slabs": self.slabs, "rows": self.rows, "swapped": self.swapped, "why": self.why}


def plane_class(decl: AttnLayerDecl, seqs: StepSequences, policy: LayoutPolicy = LayoutPolicy()) -> PlaneClass:
    """The class of layer `decl` at a step with active set `seqs` under `policy` -- the composition of the kernel's fold rule (statics x
    max_seqlen_q) and the collector's staging rule (b, swapped, compact).  Pure."""
    b, mq = seqs.n_seq, seqs.max_seqlen_q
    if b <= 0 or mq <= 0:
        return PlaneClass(NONE, 0, 0, 0, False, "no active sequence: the layer launches nothing")
    if policy.backend == "fa3":
        g = max(1, decl.NH // max(1, decl.KVH))
        return PlaneClass(WHOLE, 1, b * decl.KVH, mq * g, False, "fa3: one PackGQA launch staged whole (HB = b*KVH, M = max_seqlen_q*NH/KVH)")
    swapped = mq == 1 and decl.swaps_on_decode()
    if swapped:
        return PlaneClass(WHOLE, 1, b * decl.KVH, decl.NH // decl.KVH, True,
                          f"decode launch of a GQA layer without a window: swapped kernel (HB = b*KVH = {b}*{decl.KVH}, M = NH/KVH); staged whole")
    if b >= 2 and policy.compact:
        n, planes = len(seqs.active), (PLANES_MS if policy.ms else PLANES)
        return PlaneClass(PER_SEQ, len(planes) * n, decl.NH, 0, False,
                          f"un-swapped multi-sequence launch ({'window ' + str(decl.window) if decl.window is not None else 'NH == KVH or prefill'}): "
                          f"per-sequence compaction, {len(planes)} planes x {n} sequence(s) with rows, h = NH = {decl.NH}", planes)
    return PlaneClass(WHOLE, 1, b * decl.NH, mq, False,
                      f"un-swapped launch staged whole ({'single sequence' if b == 1 else 'compaction off'}: HB = b*NH = {b}*{decl.NH}, M = max_seqlen_q = {mq})")


# ---------------------------------------------------------------------------------------------------------------------------------
# the observed inventory
# ---------------------------------------------------------------------------------------------------------------------------------

@dataclass
class ObservedLayer:
    layer: int
    whole: list[tuple[int, int, int, int | None, int | None]] = field(default_factory=list)      # (HB, M, NB, BN, D) per whole tensor
    per_seq: dict[int, Counter] = field(default_factory=dict)                                    # seq -> Counter(plane)
    per_seq_geom: dict[int, set[tuple[int, int, int, int | None, int | None]]] = field(default_factory=dict)   # seq -> {(h, sl, NB, BN, D)}
    padding: int = 0                                                                              # `/<rid>/bot` planes (B9.1), counted apart
    unparsed: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.whole) + sum(sum(c.values()) for c in self.per_seq.values())

    @property
    def kind(self) -> str:
        if self.per_seq and not self.whole:
            return PER_SEQ
        if self.whole and not self.per_seq:
            return WHOLE
        if self.whole and self.per_seq:
            return "mixed"
        return NONE


def _geom(s: str) -> tuple[int, int, int, int | None, int | None] | None:
    m = _GEOM_RE.match(s)
    if not m:
        return None
    return (int(m["a"]), int(m["b"]), int(m["nb"]), int(m["bn"]) if m["bn"] else None, int(m["d"]) if m["d"] else None)


def observe(names: Iterable[str]) -> dict[int, ObservedLayer]:
    """Parse the step's hidden-plane names (`fa2.m1/L<k>/..`, `fa3.m1/L<k>/..`, with multiplicity) into per-layer observations.  Names of
    other families are ignored; a padding plane (`../<rid>/bot`) is counted apart from the executed inventory."""
    out: dict[int, ObservedLayer] = {}
    for n in names:
        m = _NAME_RE.match(n)
        if not m:
            continue
        k = int(m["layer"])
        ol = out.setdefault(k, ObservedLayer(k))
        rest = m["rest"]
        if rest.endswith("/bot"):
            ol.padding += 1
            continue
        parts = rest.split("/")
        if len(parts) == 1:
            g = _geom(parts[0])
            if g is None:
                ol.unparsed.append(n)
            else:
                ol.whole.append(g)
        elif len(parts) == 3 and parts[0].startswith("s") and parts[0][1:].isdigit() and parts[2] in PLANES_MS:
            i, g = int(parts[0][1:]), _geom(parts[1])
            if g is None:
                ol.unparsed.append(n)
                continue
            ol.per_seq.setdefault(i, Counter())[parts[2]] += 1
            ol.per_seq_geom.setdefault(i, set()).add(g)
        else:
            ol.unparsed.append(n)
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------------------------------------------------------------

def _layer_diffs(decl: AttnLayerDecl, exp: PlaneClass, obs: ObservedLayer | None, seqs: StepSequences) -> list[str]:
    """Every named disagreement between the layer's derived class and its observed inventory (empty = the layer agrees)."""
    if obs is None or (obs.count == 0 and not obs.unparsed):
        return [] if exp.count == 0 else [f"no plane committed (expected {exp.kind} x{exp.count})"]
    d: list[str] = []
    if obs.unparsed:
        d.append(f"{len(obs.unparsed)} unparsable plane name(s), e.g. {obs.unparsed[0]!r}")
    if obs.kind != exp.kind:
        d.append(f"class {obs.kind} observed, {exp.kind} declared ({exp.why})")
    if obs.count != exp.count:
        d.append(f"count {obs.count} observed, {exp.count} expected")
    if exp.kind == WHOLE and obs.whole:
        for HB, M, _NB, _BN, D in obs.whole:
            if (HB, M) != (exp.slabs, exp.rows):
                d.append(f"whole geometry g{HB}x{M} observed, g{exp.slabs}x{exp.rows} expected")
            if D is not None and D != decl.D:
                d.append(f"head_dim {D} in the name, {decl.D} declared")
    if exp.kind == PER_SEQ and obs.per_seq:
        want = set(seqs.active)
        got = set(obs.per_seq)
        if got != want:
            d.append(f"sequences {sorted(got)} observed, {sorted(want)} active")
        short = {i: dict(c) for i, c in obs.per_seq.items() if sorted(c.elements()) != sorted(exp.planes)}
        if short:
            i0 = min(short)
            d.append(f"sequence s{i0} carries planes {short[i0]} (each active sequence carries exactly {list(exp.planes)})")
        for i, gs in sorted(obs.per_seq_geom.items()):
            for h, sl, _NB, _BN, D in gs:
                if h != exp.slabs:
                    d.append(f"s{i}: {h} slabs observed, NH = {exp.slabs} expected")
                if seqs.exact and i < seqs.n_seq and sl != seqs.seqlen_q[i]:
                    d.append(f"s{i}: {sl} rows observed, seqlen_q {seqs.seqlen_q[i]} scheduled")
                if D is not None and D != decl.D:
                    d.append(f"s{i}: head_dim {D} in the name, {decl.D} declared")
    return d


def check_step(decls: Iterable[AttnLayerDecl], seqs: StepSequences, names: Iterable[str], policy: LayoutPolicy = LayoutPolicy()) -> dict[str, Any]:
    """The X-03 attention-plane verdict of ONE step: for every declared layer the derived class/count (`plane_class`) vs the observed
    inventory (`observe`), plus every observed layer no declaration covers.  Pure; `names` may repeat (multiplicity counts).
    -> {ok, n_seq, max_seqlen_q, expected: {layer: class}, classes: {kind: [layers]} (uniformity as a consequence), layers_bad: {layer:
    [named differences]}, fa2_layers_bad: {layer: observed count} (the legacy record shape), undeclared_layers, padding_planes}."""
    decls = sorted(decls, key=lambda d: d.layer)
    obs = observe(names)
    expected: dict[int, PlaneClass] = {d.layer: plane_class(d, seqs, policy) for d in decls}
    layers_bad: dict[int, list[str]] = {}
    for d in decls:
        diffs = _layer_diffs(d, expected[d.layer], obs.get(d.layer), seqs)
        if diffs:
            layers_bad[d.layer] = diffs
    declared = {d.layer for d in decls}
    undeclared = {k: o.count for k, o in obs.items() if k not in declared and o.count > 0}
    classes: dict[str, list[int]] = {}
    for k, e in expected.items():
        classes.setdefault(f"{e.kind}x{e.count}", []).append(k)
    legacy = {k: (obs[k].count if k in obs else 0) for k in layers_bad}
    legacy.update(undeclared)
    return {"ok": not layers_bad and not undeclared, "rule": "program-derived per-layer class", "backend": policy.backend, "compact": policy.compact,
            "n_seq": seqs.n_seq, "max_seqlen_q": seqs.max_seqlen_q, "exact_seqlens": seqs.exact,
            "expected": {k: e.as_dict() for k, e in expected.items()}, "classes": classes,
            "layers_bad": layers_bad, "undeclared_layers": undeclared, "fa2_layers_bad": legacy,
            "padding_planes": {k: o.padding for k, o in obs.items() if o.padding}}


def uniform_rule(counts_by_layer: Mapping[int, int], attention_layers: Iterable[int]) -> dict[int, int]:
    """The uniform X-03 rule as a pure function (the no-declaration fallback of `identity_check`): every attention layer of the step
    must carry the SAME positive count -- the modal count (ties -> smallest); layers outside `attention_layers` are reported with their count."""
    layers = list(attention_layers)
    counts = {k: int(counts_by_layer.get(k, 0)) for k in layers}
    mode = Counter(counts.values()).most_common()
    ref_c = min(c for c, n in mode if n == mode[0][1]) if mode else 0
    bad = {k: c for k, c in counts.items() if c <= 0 or c != ref_c}
    bad.update({k: int(c) for k, c in counts_by_layer.items() if k not in counts})
    return bad


def counts_by_layer(names: Iterable[str]) -> Counter:
    """Hidden-plane tensors per layer, every plane counted (executed and padding alike -- what `uniform_rule` reads)."""
    c: Counter = Counter()
    for n in names:
        m = _NAME_RE.match(n)
        if m:
            c[int(m["layer"])] += 1
    return c
