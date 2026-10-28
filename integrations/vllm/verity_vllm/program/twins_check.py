"""`verity-vllm twins --check`: the twins' equivalence evidence (`program.kernels.twins`) over a served log's specs, with captured
instances checked against their output snapshots (`check.fold_compare`), and the family-level small-statics sweep.

    uv run verity-vllm twins --check --log LOG --snapshots DIR --profile P --seed S --out twins.json
"""
from __future__ import annotations

import glob
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from verity.ml.circuits.defs import SpecializedDefinition
from verity_vllm.config import UsageError, option
from verity.ml.definitions.numerics.twins import LIBRARIES, SCHEMA, U64, check_families, check_spec


def _captured_providers(result, vals, store, spec_id: str, rng: np.random.Generator, limit: int) -> list[Callable]:
    """Callables yielding (node, args, expected words) for real instances of `spec_id` at snapshot steps: operands by
    provenance exactly as tier a resolves them (`compare.Values`), expected from the instance's own output snapshot."""
    from verity_vllm.check.fold_compare import _component_ranges
    from verity.ml.circuits.refs import runs
    steps = set(store.steps)
    metas = [m for m in result.instances if m["spec"] == spec_id and m["step"] in steps]
    idx = list(range(len(metas)))
    rng.shuffle(idx)
    out = []
    for i in idx[:limit]:
        m = metas[i]

        def provider(m=m):
            node = m["node"]
            args = []
            for a in result.body.nodes[node].args:
                parts = [np.asarray(vals.of_run(space, ix, lo, hi)) for space, ix, lo, hi in runs(a.refs)]
                args.append(np.concatenate(parts) if len(parts) != 1 else parts[0])
            exp = []
            for j, view in enumerate(m["output_views"]):
                if view is None:
                    raise LookupError(f"output {j} view unknown")
                s = store.find(view, at=m["event"])
                if s is None:
                    raise LookupError(f"no snapshot of event {m['event']} covers output {j}")
                exp.append(store.extract(s, view).astype(U64))
            expect = np.concatenate(exp) if exp else np.zeros(0, dtype=U64)
            total = sum(hi - lo for lo, hi in _component_ranges(m))
            if expect.size != total:
                raise LookupError(f"snapshot has {expect.size} words for {total} outputs")
            return node, args, expect
        out.append(provider)
    return out


def check_log(log: str, profile_name: str, snapshots: str | None, *, seed: int, n_random: int, n_edge: int, k: int,
              safetensors: str | None = None, cos_sin: str | None = None, only: set[str] | None = None) -> dict:
    """Every spec of the log: `check_spec`; plus the family-level small-statics sweep.  Fold WITHOUT Program."""
    from verity_vllm.check.fold_compare import SnapshotStore, Values, WeightSource, _attach_specs, _max_rss_gb, fold_log, load_profile
    t0 = time.time()
    profile = load_profile(profile_name)
    repo = (profile.config or {}).get("model")
    result = fold_log(log, profile, with_program=False)
    _attach_specs(result)
    specs: dict[str, SpecializedDefinition] = {}
    for m in result.instances:
        specs.setdefault(m["spec"], m["_spec"])
    snaps = Path(snapshots) if snapshots else None
    store = SnapshotStore(snaps) if snaps is not None and (snaps / "snapshots.json").exists() else None
    cs = Path(cos_sin) if cos_sin else ((snaps / "cos_sin_cache.npy") if snaps is not None and (snaps / "cos_sin_cache.npy").exists() else None)
    weights = WeightSource(result.params[-1][1], [Path(p) for p in sorted(glob.glob(safetensors))] if safetensors else None, cs, repo=repo,
                           config=dict(profile.config or {}))
    vals = Values(result, store, weights) if store is not None else None
    captured_note = None
    if store is None:
        captured_note = "no snapshots.json: captured instances not checked"
    elif snaps is not None and not (snaps / "values").exists():
        captured_note = f"{snaps}/values missing: captured instances not checked"
        vals = None
    report: dict[str, Any] = {"schema": SCHEMA, "log": str(log), "profile": profile_name, "model": repo, "snapshots": str(snaps) if snaps else None,
                              "seed": seed, "n_random_requested": n_random, "n_edge_requested": n_edge, "k_captured": k,
                              "weights_available": weights.available(), "captured_note": captured_note, "specs": {}, "families": {}}
    rng = np.random.default_rng(seed)
    for sid in sorted(specs):
        if only and sid.split("{")[0] not in only and sid not in only:
            continue
        spec = specs[sid]
        prov = _captured_providers(result, vals, store, sid, rng, k * 4) if vals is not None else None
        report["specs"][sid] = check_spec(spec, rng, n_random=n_random, n_edge=n_edge, captured=prov, k=k)
        e = report["specs"][sid]
        print(f"  {sid:52s} {e['twin_id'] or '-':28s} {e['status']:8s} random {e['n_random']:3d} captured {e['n_captured']:2d} "
              f"edge {e['n_edge']:3d} declined {e['n_declined']:2d} {e['seconds']}s", file=sys.stderr, flush=True)
    report["families"] = check_families(rng, n_random=n_random, n_edge=n_edge)
    st = Counter(e["status"] for e in report["specs"].values())
    fam_fail = [f for f, e in report["families"].items() if e["status"] == "fail"]
    report["summary"] = {"n_specs": len(report["specs"]), "n_twinned": sum(1 for e in report["specs"].values() if e["twin_id"]),
                         "by_status": dict(st), "words_compared": sum(e.get("words_random", 0) for e in report["specs"].values()),
                         "families_failing": fam_fail}
    report["status"] = "fail" if st.get("fail") or fam_fail else "pass"
    report["libraries"] = dict(LIBRARIES)
    report["seconds"] = round(time.time() - t0, 3)
    report["max_rss_gb"] = round(_max_rss_gb(), 3)
    return report


@dataclass(frozen=True, kw_only=True)
class Options:
    check: bool = option("--check", default=False, action="store_true",
                         help="equivalence evidence against the gate evaluator (writes --out)")
    log: str | None = option("--log", default=None)
    snapshots: str | None = option("--snapshots", default=None)
    profile: str | None = option("--profile", default=None)
    seed: int = option("--seed", type=int, default=0)
    n_random: int = option("--n-random", type=int, default=12, help="random instances per spec (and per family)")
    n_edge: int = option("--n-edge", type=int, default=8, help="edge-word instances per spec (and per family)")
    k: int = option("--k", type=int, default=3, help="captured instances per spec")
    only: str | None = option("--only", default=None, help="comma-separated base ids or spec ids to check")
    safetensors: str | None = option("--safetensors", default=None)
    cos_sin: str | None = option("--cos-sin", default=None)
    out: str | None = option("--out", default=None)


def main(a: Options) -> int:
    if not a.check:
        raise UsageError("nothing to do: --check")
    t0 = time.time()
    if a.log:
        if not a.profile:
            raise UsageError("--log needs --profile")
        rep = check_log(a.log, a.profile, a.snapshots, seed=a.seed, n_random=a.n_random, n_edge=a.n_edge, k=a.k,
                        safetensors=a.safetensors, cos_sin=a.cos_sin, only=set(a.only.split(",")) if a.only else None)
    else:
        from verity_vllm.check.fold_compare import _max_rss_gb
        rng = np.random.default_rng(a.seed)
        fams = check_families(rng, n_random=a.n_random, n_edge=a.n_edge)
        rep = {"schema": SCHEMA, "log": None, "seed": a.seed, "specs": {}, "families": fams,
               "status": "fail" if any(e["status"] == "fail" for e in fams.values()) else "pass",
               "libraries": dict(LIBRARIES), "seconds": round(time.time() - t0, 3), "max_rss_gb": round(_max_rss_gb(), 3)}
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=1, default=str))
        print(str(out))
    print(f"twins {rep['status']}: {rep.get('summary')} {rep['seconds']}s {rep.get('max_rss_gb')}GB", file=sys.stderr)
    return 0 if rep["status"] == "pass" else 1


__all__ = ["Options", "check_log", "main"]
