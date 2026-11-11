"""Resolve a raw observation log against a pinned profile (torch-free).

    uv run verity-vllm resolve-log data/logs/m1.jsonl.gz \
        --profile vllm_d9105ea80_sm89_eager [--step N] [--dump-trees] [--max-depth 6] [--out PATH] [--strict]

Per step: `StepContext` (header, StepStart, SampleEnd, StepLinks, Roots by allocation — the weight/buffer Roots of the
preamble plus the step's prompt-row Roots —, the profile's C, the profile), `resolve_step`, and the op histogram with
the resolution of every event.  Prints the per-step summary, the op histogram, the `ResolutionReport` JSON (primary
resolutions, the integrator's counter) and the expanded counts (grouped members: one `Attention_v3{T}` per T, the k
rotation of RoPE, the v move of the KV update), then writes `<out>.json` and `<out>.md`
(default `out/capture/reports/resolution_<log stem>.{json,md}`).  `--dump-trees` prints `TreeNode.render` for one step
(`--step`, default 0), collapsing roots whose subtree *shape* (ops only) was already printed.  Exit status 1 with
`--strict` when any step has an Unsupported resolution or a `ResolutionError`.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import sys
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from verity_vllm.config import option
from verity_vllm.observe.events import Event, Note, Root, RunHeader, StepLink, StorageAlloc, ViewDesc
from verity_vllm.observe.log import read_log, split_steps
from verity_vllm.observe.fold.patterns import GEMM_TARGET_SCRATCH_KEY, X01_SCRATCH_KEY, label, members
from verity_vllm.observe.fold.resolver import (Compute, Group, Profile, Resolution, ResolutionError, ResolutionReport, StepContext, Transparent,
                                               external_buffers, spec_module, Unsupported, resolve_step)
from verity_vllm.observe.fold.tree import StepEvents, TreeNode, step_events

def load_profile(name: str) -> tuple[Profile, Any]:
    mod_name = name if "." in name else f"verity_vllm.observe.fold.profiles.{name}"
    mod = importlib.import_module(mod_name)
    return mod.PROFILE, mod


# ---------------------------------------------------------------------------------------------------------
# per-step resolution
# ---------------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class StepResult:
    step: int
    phase_hint: str
    rows: int
    roots: int
    events: int
    resolutions: list[Resolution]
    error: str | None = None
    seconds: float = 0.0
    target: dict | None = None                 # [X-01] per-step target-fact acceptances (patterns.X01_SCRATCH_KEY), None when none fired
    gemm_target: dict | None = None            # per-step registered GEMM target selection (patterns.GEMM_TARGET_SCRATCH_KEY)

    @property
    def unsupported(self) -> list[Unsupported]:
        return [r for r in self.resolutions if isinstance(r, Unsupported)]

    def computes(self) -> list[Compute]:
        return [m for r in self.resolutions for m in members(r) if isinstance(m, Compute)]

    def transparents(self) -> list[Transparent]:
        return [m for r in self.resolutions for m in members(r) if isinstance(m, Transparent)]

    def spec_rows(self) -> dict[str, int]:
        c: Counter = Counter()
        for m in self.computes():
            c[m.spec.id] += m.rows
        return dict(sorted(c.items()))

    def spec_computes(self) -> dict[str, int]:
        c: Counter = Counter()
        for m in self.computes():
            c[m.spec.id] += 1
        return dict(sorted(c.items()))

    def spec_family_rows(self) -> dict[str, int]:
        """Rows per definition (`Gemm_v1`, `Attention_v3`, ...) — the per-step counts G2 is stated in."""
        c: Counter = Counter()
        for m in self.computes():
            c[m.spec.id.split("{")[0]] += m.rows
        return dict(sorted(c.items()))


def collect_roots(events: list[Event], into: dict[int, list[Root]]) -> None:
    for ev in events:
        if isinstance(ev, Root):
            into.setdefault(ev.view.alloc, []).append(ev)


def make_context(header: RunHeader, se: StepEvents, roots: dict[int, list[Root]], profile: Profile,
                 buffers: dict[str, ViewDesc] | None = None) -> StepContext:
    step_roots: dict[int, list[Root]] = {k: list(v) for k, v in roots.items()}
    collect_roots(se.other, step_roots)
    return StepContext(header, se.start, se.sample, list(se.links), step_roots, dict(profile.config), profile, buffers=buffers or {})


def resolve_log(path: str | Path, profile: Profile, only_step: int | None = None,
                on_step: Any = None) -> tuple[RunHeader | None, list[StepResult], dict[int, list[Root]], list[Note]]:
    header: RunHeader | None = None
    roots: dict[int, list[Root]] = {}
    notes: list[Note] = []
    results: list[StepResult] = []
    devices: dict[int, str] = {}
    buffers: dict[str, ViewDesc] = {}
    for pre, step in split_steps(read_log(path)):
        for ev in pre:
            if isinstance(ev, RunHeader):
                header = ev
            elif isinstance(ev, StorageAlloc):
                devices[ev.alloc] = ev.device
            elif isinstance(ev, Note):
                notes.append(ev)
                buffers.update(external_buffers(ev, lambda a: devices.get(a, "")))
        collect_roots(pre, roots)
        if not step:
            continue
        se = step_events(step)
        if only_step is not None and se.start.step != only_step:
            continue
        if header is None:
            header = RunHeader("?", "", "", {}, {}, {})
        ctx = make_context(header, se, roots, profile, buffers)
        t0 = time.perf_counter()
        n_events = sum(len(r.ids()) for r in se.roots)
        try:
            res = resolve_step(se, ctx)
            err = None
        except ResolutionError as e:
            res, err = [], str(e)
        sr = StepResult(se.start.step, se.start.phase_hint, len(se.start.rows), len(se.roots), n_events, res, err,
                        time.perf_counter() - t0, target=ctx.scratch.get(X01_SCRATCH_KEY), gemm_target=ctx.scratch.get(GEMM_TARGET_SCRATCH_KEY))
        results.append(sr)
        if on_step is not None:
            on_step(se, sr)
    return header, results, roots, notes


# ---------------------------------------------------------------------------------------------------------
# histograms and reports
# ---------------------------------------------------------------------------------------------------------


def op_resolution_histogram(se: StepEvents, res: list[Resolution]) -> dict[str, Counter]:
    """op -> Counter of resolution labels; nested events count as `consumed by <pattern>`."""
    h: dict[str, Counter] = OrderedDict()
    for root, r in zip(se.roots, res):
        h.setdefault(root.op, Counter())[short_label(r)] += 1
        pat = getattr(r, "pattern", "") or ("unsupported" if isinstance(r, Unsupported) else "?")
        for n in root.walk():
            if n is root:
                continue
            h.setdefault(n.op, Counter())[f"consumed by {pat}"] += 1
    return h


def short_label(r: Resolution) -> str:
    if isinstance(r, Compute):
        extra = getattr(r, "also", ())
        base = f"Compute {r.spec.id}"
        return base + (f" (+{len(extra)} grouped)" if extra else "")
    if isinstance(r, Transparent):
        extra = getattr(r, "also", ())
        return f"Transparent {r.effect}" + (f" (+{len(extra)} grouped)" if extra else "")
    if isinstance(r, Group):   # a `resolver.Group` root is neither spelling above
        kinds = Counter("Compute" if isinstance(p, Compute) else "Transparent" if isinstance(p, Transparent) else type(p).__name__
                        for p in r.parts)
        return f"Group of {len(r.parts)}: " + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))
    return f"Unsupported: {r.reason}"


def merge_hist(into: dict[str, Counter], h: dict[str, Counter]) -> None:
    for op, c in h.items():
        into.setdefault(op, Counter()).update(c)


def shape_signature(node: TreeNode) -> str:
    return node.op + "(" + ",".join(shape_signature(c) for c in node.children) + ")"


def dump_trees(se: StepEvents, res: list[Resolution] | None, max_depth: int, out=sys.stdout) -> None:
    seen: dict[str, int] = {}
    print(f"# step {se.start.step} phase={se.start.phase_hint} rows={len(se.start.rows)} roots={len(se.roots)} "
          f"logits_rows={len(se.start.logits_rows)} block_size={se.start.block_size}", file=out)
    print(f"# input_ids_view={se.start.input_ids_view!r} positions_view={se.start.positions_view!r}", file=out)
    for i, root in enumerate(se.roots):
        sig = shape_signature(root)
        r = res[i] if res is not None and i < len(res) else None
        tag = f"  => {short_label(r)}" if r is not None else ""
        if sig in seen:
            print(f"[{i}] {root.op}  (same shape as root #{seen[sig]}; {len(root.ids())} events){tag}", file=out)
            continue
        seen[sig] = i
        print(f"[{i}] phase={root.phase} module={root.enter.module}{tag}", file=out)
        print(root.render(1, max_depth), file=out)
        if root.is_triton:
            e = root.enter
            args = {k: (repr(v) if not isinstance(v, (int, float, str, bool, type(None))) else v) for k, v in e.args.items()}
            print(f"    args={args} num_warps={e.num_warps} num_stages={e.num_stages}", file=out)
        else:
            print(f"    schema={root.enter.schema}", file=out)
            scalars = [a for a in root.enter.args if not hasattr(a, "alloc")]
            if scalars or root.enter.kwargs:
                print(f"    scalar args={scalars} kwargs={root.enter.kwargs}", file=out)


def build_report(header: RunHeader | None, results: list[StepResult], roots: dict[int, list[Root]], hist: dict[str, Counter],
                 profile: Profile, prof_mod: Any, log_path: str, notes: list[Note]) -> dict:
    primary = ResolutionReport()
    for sr in results:
        primary.steps += 1
        for r in sr.resolutions:
            primary.add(r)
    computes: Counter = Counter()
    rows: Counter = Counter()
    effects: Counter = Counter()
    spec_modules: dict[str, str] = {}
    for sr in results:
        for m in sr.computes():
            computes[m.spec.id] += 1
            rows[m.spec.id] += m.rows
            if m.spec.id not in spec_modules:
                spec_modules[m.spec.id] = spec_module(m.spec)
        for t in sr.transparents():
            effects[t.effect] += 1
    unsupported_examples: list[dict] = []
    per_op_examples: Counter = Counter()
    for sr in results:
        for u in sr.unsupported:
            if per_op_examples[u.op] < 3:
                per_op_examples[u.op] += 1
                unsupported_examples.append({"step": sr.step, "id": u.event_id, "op": u.op, "reason": u.reason,
                                             "detail": _jsonable(u.detail)})
    root_classes: Counter = Counter()
    unknown_roots: list[str] = []
    classify = getattr(prof_mod, "root_class", None)
    for rs in roots.values():
        for r in rs:
            if r.role in ("weight", "buffer"):
                cls = classify(r.name) if classify else ("weight" if profile.weight_field(r.name) else "unknown")
                root_classes[cls] += 1
                if cls == "unknown":
                    unknown_roots.append(r.name)
            else:
                root_classes[r.role] += 1
    header_mismatch = []
    check = getattr(prof_mod, "check_header", None)
    if header is not None and check is not None:
        header_mismatch = check(header)
    return {
        "log": str(log_path),
        "profile": profile.id,
        "log_profile_id": header.profile_id if header else None,
        "header_mismatches": header_mismatch,
        "steps": [{"step": sr.step, "phase": sr.phase_hint, "rows": sr.rows, "roots": sr.roots, "events": sr.events,
                   "computes": len(sr.computes()), "transparent": len(sr.transparents()), "unsupported": len(sr.unsupported),
                   "error": sr.error, "seconds": round(sr.seconds, 4), "spec_rows": sr.spec_rows(),
                   "spec_computes": sr.spec_computes(), "family_rows": sr.spec_family_rows()} for sr in results],
        "totals": {"steps": len(results), "unsupported": sum(len(sr.unsupported) for sr in results),
                   "errors": sum(1 for sr in results if sr.error), "computes": sum(computes.values()),
                   "compute_rows": sum(rows.values()), "transparent": sum(effects.values())},
        "by_spec": {k: {"computes": computes[k], "rows": rows[k], "module": spec_modules.get(k)} for k in sorted(computes)},
        "spec_modules": dict(sorted(spec_modules.items())),
        "quarantine_specs": sorted(k for k, m in spec_modules.items() if ".quarantine." in (m or "")),
        "transparent_effects": dict(sorted(effects.items())),
        "op_histogram": {op: dict(c.most_common()) for op, c in hist.items()},
        "unsupported_examples": unsupported_examples,
        "primary_report": primary.to_dict(),
        "roots": {"classes": dict(root_classes), "unknown": unknown_roots[:32]},
        "notes": [{"text": n.text, "data": _jsonable(n.data)} for n in notes[:64]],
        "target": _target_facts(header, results),
    }


def _target_facts(header: RunHeader | None, results: list[StepResult]) -> dict:
    """[X-01] The target the resolver pinned the persistent GEMM against: the header's GPU document (the capture host's device query,
    `TargetProfile.from_gpu_document`) and the per-run total of GEMM launches checked against that target's `NUM_SMS` (0 launches /
    `target_profile` None when the header carried no device: the profile default was the pin)."""
    from verity_vllm.program.frontend.target_profile import TargetProfile
    gpu = ((header.profile if header else None) or {}).get("gpu") or {}
    tp = TargetProfile.from_gpu_document(gpu) if gpu else None
    acc = [sr.target for sr in results if sr.target]
    out: dict = {"host_gpu": {"name": gpu.get("name"), "compute_capability": gpu.get("compute_capability", gpu.get("cc")),
                              "multi_processor_count": gpu.get("multi_processor_count", gpu.get("sms"))},
                 "target_profile": tp.to_json() if tp else None,
                 "num_sms_pinned_from_target": {"launches": sum(int(a.get("launches", 0)) for a in acc), "steps": len(acc)}}
    if acc:
        out["num_sms_pinned_from_target"].update({k: acc[0][k] for k in ("pinned", "source", "profile_default", "basis") if k in acc[0]})
    gts = [sr.gemm_target for sr in results if sr.gemm_target]   # which registered GEMM specialization pinned the launch geometry
    out["gemm_target"] = None if not gts else {**{k: gts[0].get(k) for k in ("target", "dot", "refusal", "compute_capability", "header_tuned_gemm_arch_family",
                                                                              "tuned_table_sha256", "basis")},
                                               "launches": sum(int(g.get("launches", 0)) for g in gts), "steps": len(gts)}
    return out


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    return repr(x)


def report_markdown(rep: dict) -> str:
    L: list[str] = []
    L.append(f"# Resolution report: `{Path(rep['log']).name}` × profile `{rep['profile']}`")
    L.append("")
    L.append(f"- log profile_id: `{rep['log_profile_id']}`; header mismatches vs C: {rep['header_mismatches'] or 'none'}")
    t = rep["totals"]
    L.append(f"- steps {t['steps']}, computes {t['computes']} ({t['compute_rows']} rows), transparent {t['transparent']}, "
             f"**Unsupported {t['unsupported']}**, resolution errors {t['errors']}")
    L.append(f"- roots: {rep['roots']['classes']}" + (f"; unknown: {rep['roots']['unknown']}" if rep['roots']['unknown'] else ""))
    L.append("")
    L.append("## Specializations")
    L.append("")
    L.append("| specialization | computes | rows |")
    L.append("| --- | ---: | ---: |")
    for k, v in rep["by_spec"].items():
        L.append(f"| `{k}` | {v['computes']} | {v['rows']} |")
    L.append("")
    L.append(f"Transparent effects: {rep['transparent_effects']}")
    L.append("")
    L.append("## Steps")
    L.append("")
    L.append("| step | phase | rows | roots | events | computes | transparent | unsupported | family rows |")
    L.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for s in rep["steps"]:
        fam = ", ".join(f"{k}={v}" for k, v in s["family_rows"].items())
        err = f" ERROR: {s['error']}" if s["error"] else ""
        L.append(f"| {s['step']} | {s['phase']} | {s['rows']} | {s['roots']} | {s['events']} | {s['computes']} | "
                 f"{s['transparent']} | {s['unsupported']} | {fam}{err} |")
    L.append("")
    L.append("## Op histogram (all steps; root events resolve, nested events are consumed by their scope)")
    L.append("")
    L.append("| op | count | resolutions |")
    L.append("| --- | ---: | --- |")
    for op, c in sorted(rep["op_histogram"].items(), key=lambda kv: (-sum(kv[1].values()), kv[0])):
        total = sum(c.values())
        parts = "; ".join(f"{k} ×{v}" for k, v in c.items())
        L.append(f"| `{op}` | {total} | {parts} |")
    L.append("")
    if rep["unsupported_examples"]:
        L.append("## Unsupported (examples, ≤3 per op)")
        L.append("")
        for u in rep["unsupported_examples"]:
            L.append(f"- step {u['step']} id {u['id']} `{u['op']}`: {u['reason']}")
            L.append(f"  - `{json.dumps(u['detail'])[:600]}`")
        L.append("")
    if rep["notes"]:
        L.append("## Notes in the log (preamble)")
        L.append("")
        for n in rep["notes"][:16]:
            L.append(f"- {n['text']}: `{json.dumps(n['data'])[:300]}`")
        L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Options:
    log: str = option("log")
    profile: str = option("--profile", help="the profile to resolve against: a module of verity_vllm.observe.fold.profiles, or a dotted path")
    step: int | None = option("--step", type=int, default=None, help="resolve (and dump) only this step")
    dump_trees: bool = option("--dump-trees", default=False, action="store_true")
    max_depth: int = option("--max-depth", type=int, default=6)
    out: str | None = option("--out", default=None, help="report base path (default out/capture/reports/resolution_<stem>)")
    strict: bool = option("--strict", default=False, action="store_true", help="exit 1 on any Unsupported / ResolutionError")
    quiet: bool = option("--quiet", default=False, action="store_true")


def main(args: Options) -> int:
    profile, prof_mod = load_profile(args.profile)
    hist: dict[str, Counter] = OrderedDict()
    dump_step = args.step if args.step is not None else 0

    def on_step(se: StepEvents, sr: StepResult) -> None:
        if sr.error is None:
            merge_hist(hist, op_resolution_histogram(se, sr.resolutions))
        if not args.quiet:
            fam = " ".join(f"{k}={v}" for k, v in sr.spec_family_rows().items())
            print(f"step {sr.step:>3} {sr.phase_hint:<7} rows={sr.rows:<3} roots={sr.roots:<4} events={sr.events:<5} "
                  f"computes={len(sr.computes()):<4} transparent={len(sr.transparents()):<4} unsupported={len(sr.unsupported):<3} "
                  f"{sr.seconds:.2f}s  {fam}" + (f"  ERROR {sr.error}" if sr.error else ""))
        if args.dump_trees and se.start.step == dump_step:
            print()
            dump_trees(se, sr.resolutions if sr.error is None else None, args.max_depth)
            print()

    header, results, roots, notes = resolve_log(args.log, profile, args.step, on_step)
    rep = build_report(header, results, roots, hist, profile, prof_mod, args.log, notes)

    if not args.quiet:
        print()
        print("## op histogram")
        for op, c in sorted(hist.items(), key=lambda kv: (-sum(kv[1].values()), kv[0])):
            print(f"  {op:<60} {sum(c.values()):>6}  " + "; ".join(f"{k} ×{v}" for k, v in c.most_common()))
        print()
        print("## ResolutionReport (primary resolutions)")
        print(json.dumps(rep["primary_report"], indent=1, default=repr)[:20000])
        print()
        print("## expanded (grouped members)")
        print(json.dumps({"by_spec": rep["by_spec"], "transparent_effects": rep["transparent_effects"], "totals": rep["totals"],
                          "roots": rep["roots"], "header_mismatches": rep["header_mismatches"]}, indent=1))
        if rep["unsupported_examples"]:
            print()
            print("## Unsupported examples")
            for u in rep["unsupported_examples"]:
                print(f"  step {u['step']} id {u['id']} {u['op']}: {u['reason']}  {json.dumps(u['detail'])[:400]}")

    stem = Path(args.log).name.split(".")[0]
    base = Path(args.out) if args.out else Path("out/capture/reports") / f"resolution_{stem}"
    base.parent.mkdir(parents=True, exist_ok=True)
    Path(str(base) + ".json").write_text(json.dumps(rep, indent=1, default=repr))
    Path(str(base) + ".md").write_text(report_markdown(rep))
    if not args.quiet:
        print(f"\nwrote {base}.json and {base}.md")
    bad = rep["totals"]["unsupported"] or rep["totals"]["errors"]
    return 1 if (args.strict and bad) else 0
