"""The identity check of a native committer's run: the identities every committed step must cover (`expected_manifest`, from
the install-time module set and the attached sources) and each committed step's tensor names held to them (`identity_check`).
Each is a function of what it is handed (the acquired modules, the sources, the steps' layouts, the driver's attention
declarations); it reads no committer and imports no torch."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from verity_vllm.commit.committer.acquired import _mro_names

# either FlashAttention tap's hidden-value rows (`fa2.m1/L<k>/...`, `fa3.m1/L<k>/...`): one family for the identity check, the trace and the footprint
HIDDEN_M1_PREFIXES = ("fa2.m1/", "fa3.m1/")
TP_PARTIAL_RE = re.compile(r"/(part|shard)_rank\d+$")   # <module path>/part_rank<r>: the rank-local input of that module's collective
CALL_DEPENDENT_RE = re.compile(r"/(all_reduce|x_q|x_s)$")   # bound only on the calls whose token count takes that path; the manifest requires them


def is_hidden_value_row(name: str) -> bool:
    """Whether a committed tensor's name is a hidden-value row of either FlashAttention tap."""
    return name.startswith(HIDDEN_M1_PREFIXES)


def tp_partials_bad(man: dict[str, Any], names: Counter[str]) -> dict[str, int]:
    """TP rank-local collective inputs: exactly one `tp/<site>/{part,shard}_rank<r>` per expected site, none from elsewhere
    (`identity_check`'s per-step TP rule over the step's tensor names)."""
    tp_bad: dict[str, int] = {}
    tp_sites = man.get("tp_sites")                                    # none for a run with no TP source: left out, null or empty
    tp_counts = Counter(n.rsplit("/", 1)[0] for n in names.elements() if TP_PARTIAL_RE.search(n))
    if tp_sites:
        tp_want = man.get("tp_site_counts")                           # a mid-module site binds its static call count per step, any other site one
        tp_bad = {s: tp_counts.get(s, 0) for s in tp_sites if tp_counts.get(s, 0) != int(tp_want.get(s, 1) if tp_want else 1)}
        tp_bad.update({s: c for s, c in tp_counts.items() if s not in tp_sites})
        rk = man.get("tp_rank")
        if rk is not None and any(not n.endswith(f"_rank{rk}") for n in names if TP_PARTIAL_RE.search(n)):
            tp_bad["<foreign rank tag>"] = sum(c for n, c in names.items() if TP_PARTIAL_RE.search(n) and not n.endswith(f"_rank{rk}"))
    elif tp_counts:
        tp_bad = {"<undeclared tp source>": sum(tp_counts.values())}
    return tp_bad


def _split_step_names(srcs: list[Any], names: Any, step: int) -> tuple[Any, dict[str, tuple[int, int]]]:
    """The step's names with each call-boundary source's own split off in turn, and every source's mismatches."""
    bad: dict[str, tuple[int, int]] = {}
    for src in srcs:
        names, mine = src.split_step_names(names, step, False)
        bad.update(mine)
    return names, bad


def _listed(man: dict[str, Any], field: str) -> Any:
    """What an expected manifest lists under `field` (its TP sites, its MoE layers, the layers whose router commits one more
    tensor, the sites ruled out); nothing for a manifest that leaves the field out or gives it null or empty: a run without that
    source declares none."""
    got = man.get(field)
    return got if got else []


def module_of(name: str) -> str:
    """Committed tensor name -> the producing module name (tuple slot `/k`, acquired input `/in`, sampler ids stripped)."""
    return name.split("/", 1)[0]


def expected_manifest(acquired: list[tuple[str, Any]], sources: list[Any], runner: Any, external_acquisition: bool = False) -> dict[str, Any]:
    """The identities every committed step MUST cover, derived from the INSTALL-TIME module set (`acquired`: the acquired
    Instance-output modules of the model, each with its name, selected by class) and the attached hidden sources, not from what
    the collector emitted: every hooked module contributes >= 1 tensor per step, the sampler exactly one `sampled_token_ids`, each
    FA2 hidden source exactly one `fa2.m1/L<k>/...` row per attention layer.  Counts alone cannot see 'missing one, duplicated
    another'; identities can."""
    modules = [n for n, _m in acquired]
    # under external acquisition (compiled kernel taps) the module forwards are NOT wrapped: the per-step identities
    # are the source's (P_compiled's kernel launches, `c<piece>.<kernel>`), declared by the source from the required manifest
    ext = [s for s in sources if hasattr(s, "expected_modules")]
    if ext and external_acquisition:
        modules = sorted({m for s in ext for m in s.expected_modules()})
    if runner is not None and hasattr(runner, "sample") and "runner.sampler" not in modules:
        modules.append("runner.sampler")        # sampled ids are acquired from the wrapped `runner.sample`, not from a module hook
    attn_layers = sorted({int(n.split(".layers.")[1].split(".")[0]) for n, m in acquired if "Attention" in _mro_names(m) and ".layers." in n})
    fa2 = any(getattr(s, "short_name", "") in ("hidden_m1", "hidden_m1_fa3") or type(s).__name__.endswith(("HiddenM1Source", "HiddenM1SourceFA3"))
              for s in sources)                                            # the FA3 tap binds the same one-row-per-layer identity
    # the MoE block source binds exactly TENSORS_PER_LAYER tensors per MoE layer per step (router_w, router_ids, up, act, down, sum), + router_softmax per tapped layer
    moe_src, rs_src = (next((s for s in sources if getattr(s, "short_name", "") == n), None) for n in ("moe", "router_softmax"))
    moe_layers = sorted(set(moe_src.layers.values())) if moe_src is not None else []
    # the TP partial source binds exactly ONE rank-local collective input per hooked site per step (install-time site set)
    tp_src = next((s for s in sources if getattr(s, "short_name", "") == "tp_partial"), None)
    # the TP source is also the ONE binder of the fused-MoE block's RECEIVED all-reduce output
    # (`model.layers.<L>.mlp.experts/out`; MoERunner is not an ACQUIRE_CLASSES module), the MoE source of a module that does not reduce it
    for sites in (getattr(tp_src, "received_site_names", None), getattr(moe_src, "out_site_names", None)):
        modules += [s for s in (sites() if sites is not None else ()) if s not in modules]
    return {"modules": modules, "attention_layers": attn_layers, "fa2_hidden_per_layer": 1 if fa2 else 0, "sampler_ids": 1 if any(n == "runner.sampler" for n in modules) else 0,
            "moe_layers": moe_layers, "moe_tensors_per_layer": (getattr(moe_src, "stats", {}).get("tensors_per_layer", 6) if moe_src is not None else 0),
            "moe_hash_layers": sorted(getattr(moe_src, "hash_layers", ())),        # a hash layer's router commits one more (`moe_source`: ids, in)
            "router_softmax_layers": sorted(int(k) for k in rs_src.layers) if rs_src is not None else [], "tp_sites": sorted(tp_src.site_names()) if tp_src is not None else [], "tp_rank": (tp_src.rank if tp_src is not None else None),
            # partials per site per step: 1, or a mid-module site's call count (OlmoeAttention: 2 gathers); the calls its instance's flags rule out are listed
            "tp_site_counts": (dict(tp_src.site_counts()) if tp_src is not None and hasattr(tp_src, "site_counts") else {}), "tp_sites_ruled_out": list(getattr(tp_src, "mid_sites_ruled_out", None) or [])}


def moe_layer_of(sources: list[Any]) -> Any:
    """The MoE source's `layer_of`: the layer of each name it commits (its planes, or its values at calls under a MoE module),
    which it holds to its own rule; None for every name without one.  Without the source (an offline committer), a plane's layer."""
    from verity_vllm.acquire.sources.moe_source import plane_layer
    src = next((s for s in sources if getattr(s, "short_name", "") == "moe"), None)
    return getattr(src, "layer_of", None) or plane_layer


def identity_check(man: dict[str, Any], layouts: dict[int, Any], sources: list[Any], decls: list[Any],
                   step_rows: dict[int, Any] | None = None, policy: Any = None) -> dict[str, Any]:
    """The identity check of the steps in `layouts` (a step's metas: its `TensorMeta`s, or a cached plan's record) against the
    expected manifest `man`.  For every committed step: (1) every module of the manifest produced >= 1 committed tensor and no
    tensor came from a module outside it; (2) the FA2/FA3 hidden source produced, per attention layer, exactly the plane inventory the
    layer's class implies -- DERIVED per layer from the Program's attention Call statics x the step's active sequences when the driver
    handed `decls` (fa2_plane_classes: a windowed layer and a global GQA layer of one step legitimately
    differ), else the uniform rule "every layer the same positive count"; (3) exactly one sampled-ids tensor; (4) the multiset of
    module-tensor NAMES (with multiplicity: a shared RotaryEmbedding is legitimately called once per layer) is the same in every step of
    the run -- the per-step call structure of the model does not depend on the token count.  A step that drops one tensor and
    duplicates another keeps its count and FAILS (1) or (4).  With a route source (`acquire.sources.route_source`: the capture map's
    model), the names the map lists are held to the map's names for the step's routes (`routes_bad`) instead of (1) and (4), and (1)
    covers the hooked modules the map has no entry for.  `step_rows` is the collector's row record of each step's active sequences
    and `policy` the committer's own staging rule, each None where it has none (`step_sequences`, `plane_policy`)."""

    from verity_vllm.acquire import fa2_plane_classes as FP
    expected_modules = set(man["modules"])
    if not expected_modules:
        return {"ok": None, "skipped": "no install-time module set (offline core)", "steps_checked": len(layouts)}
    violations: list[dict[str, Any]] = []
    ref_names: Counter | None = None
    fa2_rules_used: Counter = Counter()                          # per step: "per-layer" (declarations + active set) or "uniform"
    # per-step names held to the source's own rule, not rule (4): the call-boundary targets; the capture map's names on the step's routes
    rt_src = next((s for s in sources if hasattr(s, "step_modules")), None)
    cb_srcs = [s for s in sources if hasattr(s, "split_step_names") and s is not rt_src]
    own = frozenset().union(*(getattr(s, "own_names", ()) for s in sources))   # what no module produced (KV slots)
    moe_of = moe_layer_of(sources)

    # no step-reading source or attention declarations: a step's verdict reads its names, step 0's and the manifest only
    verdicts: dict[Any, Any]
    step_free, verdicts = rt_src is None and not cb_srcs and not (decls and man["fa2_hidden_per_layer"]), {}
    for step in sorted(layouts):
        metas = layouts[step]
        cols = metas.names if hasattr(metas, "scan") else None   # a cached plan's record (plan_cache): its names, no TensorMeta built
        if step_free and cols in verdicts:                         # a plan-name tuple already judged (None is never a key)
            fa2_rules_used.update({"uniform": 1} if man["fa2_hidden_per_layer"] else {})
            violations.extend([{**verdicts[cols], "step": step}] if verdicts[cols] else [])
            continue
        names = Counter(m.name for m in metas) if cols is None else Counter(cols)
        mod_names = Counter({n: c for n, c in names.items() if not is_hidden_value_row(n) and not n.startswith("moe/") and moe_of(n) is None and not TP_PARTIAL_RE.search(n) and not CALL_DEPENDENT_RE.search(n) and n not in own})
        tp_bad = tp_partials_bad(man, names)
        mod_names, rt_bad = rt_src.split_step_names(mod_names, step, False, metas) if rt_src is not None else (mod_names, {})
        by_module: Counter[str] = Counter()
        for n, c in mod_names.items():
            by_module[module_of(n)] += c
        step_expected = rt_src.step_modules(expected_modules) if rt_src is not None else expected_modules
        missing = sorted(step_expected - set(by_module))
        unknown = sorted(set(by_module) - expected_modules)
        fa2_layers = FP.counts_by_layer(names.elements())
        fa2_bad: dict[int, int] = {}
        fa2_detail: dict[str, Any] = {}
        if man["fa2_hidden_per_layer"]:
            seqs = step_sequences(step_rows, step, metas if cols is None else metas.scan()) if decls else None
            if seqs is not None:
                # each layer's class is DERIVED from the Program's attention Call statics x the step's active-sequence set
                # (fa2_plane_classes.plane_class): a windowed layer's decode launch is un-swapped (6 planes x sequences) while a global GQA
                # layer's is swapped (one whole row) -- uniformity across layers is a consequence for one-class Programs, not a rule
                v = FP.check_step(decls, seqs, names.elements(), plane_policy(policy, sources))
                fa2_bad = v["fa2_layers_bad"]
                fa2_rules_used["per-layer"] += 1
                if fa2_bad:
                    fa2_detail = {"rule": v["rule"], "n_seq": v["n_seq"], "max_seqlen_q": v["max_seqlen_q"], "exact_seqlens": v["exact_seqlens"],
                                  "classes": v["classes"], "layers_bad": {k: d[:3] for k, d in list(v["layers_bad"].items())[:4]},
                                  "undeclared_layers": dict(list(v["undeclared_layers"].items())[:4])}
            else:
                # uniformity rule (no declarations, or no record of the step's active sequences): one whole-buffer row per layer, or (per-sequence
                # compaction) 6 planes x sequences per layer -- every attention layer of the step must carry the SAME positive count
                fa2_bad = FP.uniform_rule(fa2_layers, man["attention_layers"])
                fa2_rules_used["uniform"] += 1
        moe_bad: dict[int, int] = {}
        if man.get("moe_layers"):
            moe_counts = Counter(k for n in names.elements() if (k := moe_of(n)) is not None)
            want, rs_layers, hash_layers = (int(man.get("moe_tensors_per_layer") or 0), set(_listed(man, "router_softmax_layers")),
                                            set(_listed(man, "moe_hash_layers")))
            moe_bad = {k: moe_counts.get(k, 0) for k in man["moe_layers"] if moe_counts.get(k, 0) != want + (k in rs_layers) + (k in hash_layers)}
            moe_bad.update({k: c for k, c in moe_counts.items() if k not in man["moe_layers"]})
        elif any(n.startswith("moe/") for n in names):
            moe_bad = {-1: sum(c for n, c in names.items() if n.startswith("moe/"))}      # MoE tensors from no declared source
        n_ids = names.get("runner.sampler/sampled_token_ids", 0)
        sampler_bad = bool(man["sampler_ids"]) and n_ids != man["sampler_ids"]
        mod_names, cb_bad = _split_step_names(cb_srcs, mod_names, step)
        split_bad = {k: bad for k, bad in (("call_boundaries", cb_bad), ("routes", rt_bad)) if bad}
        ref_names = mod_names if ref_names is None else ref_names
        diff = {n: (mod_names.get(n, 0), ref_names.get(n, 0)) for n in set(mod_names) | set(ref_names) if mod_names.get(n, 0) != ref_names.get(n, 0)}
        if missing or unknown or fa2_bad or moe_bad or tp_bad or sampler_bad or diff or split_bad:
            violations.append({"step": step, "missing_modules": missing[:8], "unknown_modules": unknown[:8], "fa2_layers_bad": fa2_bad,
                               "moe_layers_bad": moe_bad, "tp_sites_bad": dict(list(tp_bad.items())[:8]), "tp_sites_bad_n": len(tp_bad),
                               "sampler_ids": n_ids, "multiplicity_vs_step0": dict(sorted(diff.items())[:8]),
                               **{k: v for sn, bad in split_bad.items() for k, v in ((f"{sn}_bad", dict(sorted(bad.items())[:8])), (f"{sn}_bad_n", len(bad)))},
                               **({"fa2_detail": fa2_detail} if fa2_detail else {})})
        if step_free and cols is not None:
            verdicts[cols] = violations[-1] if violations and violations[-1]["step"] == step else None
    return {"ok": not violations, "steps_checked": len(layouts),
            "manifest": {"modules": len(expected_modules), "attention_layers": len(man["attention_layers"]), "fa2_hidden_per_layer": man["fa2_hidden_per_layer"],
                         "sampler_ids": man["sampler_ids"], "moe_layers": len(_listed(man, "moe_layers")), "moe_tensors_per_layer": man.get("moe_tensors_per_layer", 0),
                         "tp_sites": len(_listed(man, "tp_sites")), "tp_rank": man.get("tp_rank"), "tp_sites_ruled_out": _listed(man, "tp_sites_ruled_out")},
            # which attention-plane rule each step ran and what it was derived from
            "fa2_rule": {"steps": dict(fa2_rules_used), "attention_decls": len(decls),
                         "policy": {k: v for k, v in plane_policy(policy, sources).__dict__.items() if k != "ms" or v} if decls else None,
                         "classes_declared": (sorted(Counter(("windowed" if d.window is not None else "global") + ("-gqa" if d.NH > d.KVH else "-mha")
                                                             for d in decls).items()) if decls else None)},
            "violations": violations[:5], "violations_n": len(violations)}


def plane_policy(policy: Any, sources: list[Any] | None) -> Any:
    """The collector's hidden-plane staging rule (`fa2_plane_classes.LayoutPolicy`): the explicit override `policy`, else read
    from the installed hidden source -- the native FA2 source's `Fa2Wrap.compact` (per-sequence compaction), the Python source
    stages whole buffers, the FA3 sources stage whole PackGQA launches."""
    from verity_vllm.acquire import fa2_plane_classes as FP

    if policy is not None:
        return policy
    for s in sources or []:
        sn = getattr(s, "short_name", "")
        tn = type(s).__name__
        if sn == "hidden_m1_fa3" or tn.endswith("FA3"):
            return FP.LayoutPolicy(backend="fa3", compact=False, ms=bool(getattr(s, "ms", False)))
        if sn == "hidden_m1" or tn.endswith("HiddenM1Source"):
            w = getattr(s, "_w", None)
            return FP.LayoutPolicy(backend="fa2", compact=bool(getattr(w, "compact", False)) if w is not None else False, ms=bool(getattr(s, "ms", False)))
    return FP.LayoutPolicy()


def step_sequences(step_rows: dict[int, Any] | None, step: int, metas: list) -> Any:
    """The step's active-sequence set (`fa2_plane_classes.StepSequences`): the collector's row record `step_rows[step]`
    (BIND-01: [(request, row_lo, row_hi)] in launch order) when present; else (n_seq, tokens) read from the executed layout -- the
    sampled-ids member has one slot per co-scheduled sequence, the integer ids input one row per token; None when neither is known
    (the caller then runs the uniform rule for the step).  Never read from the hidden planes themselves."""
    from verity_vllm.acquire import fa2_plane_classes as FP

    rows = (step_rows or {}).get(int(step))
    if rows:
        return FP.StepSequences.from_rows(rows)
    n_seq = tokens = None
    for m in metas:
        numel = int(math.prod(m.shape)) if m.shape else 0
        if m.name == "runner.sampler/sampled_token_ids":
            n_seq = numel
        elif m.name.endswith("/in") and tokens is None and str(m.dtype).replace("torch.", "") in ("int32", "int64", "i32", "i64", "long", "int"):
            tokens = numel
    if n_seq is None or tokens is None:
        return None
    return FP.StepSequences.from_counts(n_seq, tokens)
