"""[E6 NIGHT 2026-09-12] The rows pipeline of a Serve_v1 registry case, in two memory-bounded phases (the laptop's guardian caps a lane process at 2.5 GB):

  python scripts/serve_rows_pipeline.py plan <CASE> <LP> <STEPS> <OUT>   # build program + P1/cu=whole plan; STREAMED plan-rule/v1 -> OUT/plan_digest.json
  python scripts/serve_rows_pipeline.py rows <CASE> <LP> <STEPS> <OUT>   # build again; rows (population_v2.build_rows) + partition digest -> OUT/population.compact.json, OUT/population.v5.json

then ``python -m e2e.population_v3 freeze serve --case <CASE> --lp LP --steps STEPS --pipeline-dir OUT --out <instance dir> --literals compiled --token-select whole-scan``.

Why two phases: ``veritor.plans.summary.plan_digest`` materialises ``plan.to_json()`` (one dict per run; 2.1 M runs at 4x8 = +1.1 GB, MEASURED: the single-phase
pipeline was killed at 2.88 GB).  :func:`plan_rule_digest_streamed` yields the canonical JSON of ``plan.to_json()`` chunk by chunk (two passes: the tagged frame
carries the payload length first) and is byte-identical to ``identity_digest("veritor/plans/plan-rule/v1", plan.to_json())`` -- MEASURED equal on B0 4x8 (TINY's
``c4e04cd6…``); 0.60 GB peak on SmolLM2-360M 4x8 (182 s).  The rows phase re-builds the program (15-40 s) rather than sharing a process with the digest.
"""
from __future__ import annotations

import gc
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent
for p in (str(ROOT), str(POC), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
sys.setrecursionlimit(100_000)

from veritor.core.identity import _TAGGED_HASH_PREFIX, canonical_json_text  # noqa: E402
from veritor.plans.runs import UnitRuns  # noqa: E402

PLAN_RULE_TAG = b"veritor/plans/plan-rule/v1"
CHECK_HEAD, CHECK_EVERY = 4096, 1024  # column-JSON cross-check sample (see _chunks)


def _canon(v) -> bytes:
    return canonical_json_text(v).encode("utf-8")


def _chunks(plan):
    """The canonical JSON bytes of ``plan.to_json()`` in order (mirrors VerificationPlan.to_json's keys: name, schema, declared_domains, commitment_units,
    verification_units), a UnitRuns side run by run."""
    keys = {"commitment_units": plan.commitment_units, "verification_units": plan.verification_units}
    top = {"name": plan.name}
    if plan.compact:
        top["schema"] = "veritor/plans/verification-plan/v1"
    if plan.declared_domains:
        top["schema"] = "veritor/plans/verification-plan/v1"
        top["declared_domains"] = {str(g): int(b) for g, b in sorted(plan.declared_domains.items())}
    yield b"{"
    first = True
    for k in sorted(set(top) | set(keys)):
        if not first:
            yield b","
        first = False
        yield _canon(k) + b":"
        if k in top:
            yield _canon(top[k])
            continue
        side = keys[k]
        if isinstance(side, UnitRuns):
            yield b'{"runs":['
            f2 = True
            # [E2 SCALE 2026-09-13] the run JSON comes straight from the columns (UnitRuns.iter_rule_json: no dict per
            # run); the first CHECK_HEAD runs and every CHECK_EVERY-th run are cross-checked against _canon(run.to_json())
            views = side.runs
            for r, enc in enumerate(side.iter_rule_json()):
                if not f2:
                    yield b","
                f2 = False
                if r < CHECK_HEAD or r % CHECK_EVERY == 0:
                    ref = _canon(views[r].to_json())
                    if ref != enc:
                        raise SystemExit(f"REFUSED: run {r} column JSON {enc!r} != canonical {ref!r}")
                yield enc
            yield b"]}"
        else:
            yield b"["
            f2 = True
            for d in side:
                if not f2:
                    yield b","
                f2 = False
                yield _canon(d.to_json())
            yield b"]"
    yield b"}"


def plan_rule_digest_streamed(plan) -> str:
    """``plan-rule/v1`` of a compact plan without materialising ``plan.to_json()`` (O(1) memory beyond the plan)."""
    if not plan.compact or not isinstance(plan.verification_units, UnitRuns):
        raise SystemExit("REFUSED: the streamed digest is plan-rule/v1 of a compact UnitRuns plan only")
    n = sum(len(c) for c in _chunks(plan))
    h = hashlib.sha256()
    h.update(_TAGGED_HASH_PREFIX + len(PLAN_RULE_TAG).to_bytes(4, "big") + PLAN_RULE_TAG + n.to_bytes(8, "big"))
    for c in _chunks(plan):
        h.update(c)
    return h.hexdigest()


def _rss() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)


def main(argv: list[str]) -> int:
    from e2e import population as pop, population_v2 as p2
    from e2e.population_v3 import PLAN_NAME_P1
    from verity_vllm.registry import b1

    # [E6 CURVE 2026-09-13, D57] optional trailing `--attention '<json>'`: the declared kernel geometry {"kernel","kBlockM","kBlockN","num_splits"} of
    # the pinned identity -> E7's `Serve_v4{C,LP,STEPS,BN}`; omitted = the legacy `Serve_v1` (kBlockN 64).  Recorded in plan_digest.json / population.compact.json.
    attention = None
    if "--attention" in argv:
        i = argv.index("--attention"); attention = json.loads(argv[i + 1]); argv = argv[:i] + argv[i + 2:]
    if len(argv) != 5 or argv[0] not in ("plan", "rows", "check"):
        print(__doc__)
        return 2
    phase, case, lp, steps, out = argv[0], argv[1], int(argv[2]), int(argv[3]), Path(argv[4])
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    program, adapted, circuit, pdig = pop.build(dict(getattr(b1, f"{case}_CONFIG")), lp, steps, family="Serve_v4" if attention else None, attention=attention)
    plan, meta = pop.make_plan(PLAN_NAME_P1, adapted, circuit)
    print(f"{phase}: build+plan {time.time() - t:.1f}s rss {_rss()}", flush=True)
    if phase in ("plan", "check"):
        pd = plan_rule_digest_streamed(plan)
        print(f"plan_digest {pd} (plan-rule/v1) {time.time() - t:.1f}s rss {_rss()}", flush=True)
        if phase == "check":  # small shapes only: the reference materialises plan.to_json()
            from veritor.plans.summary import plan_digest
            pk, ref = plan_digest(plan)
            print(f"reference {ref} {pk} {'EQUAL' if str(ref) == pd else 'DIFFERENT'} rss {_rss()}", flush=True)
            return 0 if str(ref) == pd else 1
        json.dump({"plan_digest": pd, "plan_digest_kind": "plan-rule/v1", "program_digest": pdig, "plan_name": PLAN_NAME_P1, "meta": meta, "case": case, "LP": lp,
                   "STEPS": steps, "attention": attention, "method": "scripts/serve_rows_pipeline.py plan_rule_digest_streamed: canonical JSON of plan.to_json() streamed (two passes); "
                                             "equal to veritor.plans.summary.plan_digest (MEASURED on B0 4x8)", "seconds": round(time.time() - t, 1), "peak_rss_gb": _rss()},
                  open(out / "plan_digest.json", "w"), indent=1, sort_keys=True)
        return 0
    pdj = json.loads((out / "plan_digest.json").read_text())
    if pdj["program_digest"] != pdig or pdj.get("attention") != attention:
        raise SystemExit(f"REFUSED: plan_digest.json is of program {pdj['program_digest'][:16]} (attention {pdj.get('attention')}), this build is {pdig[:16]} ({attention})")
    from veritor.plans.summary import PARTITION_DIGEST, partition_digest
    t1 = time.time()
    kinds, rows, cu_ivs, n_v = p2.build_rows(plan, circuit)
    if n_v != len(plan.verification_units):
        raise SystemExit(f"REFUSED: rows count {n_v} != plan units {len(plan.verification_units)}")
    print(f"rows {len(rows)} N_V {n_v} {time.time() - t1:.1f}s rss {_rss()}", flush=True)
    gc.collect()
    part = partition_digest(plan)
    doc = {
        "schema": p2.POPULATION_V2, "plan_name": plan.name, "instances": 1,
        "vu_count": n_v, "ru_count": len(plan.commitment_units), "per_instance": {"vu_count": n_v, "ru_count": len(plan.commitment_units)},
        "gates": circuit.size, "program_digest": pdig, "plan_digest": pdj["plan_digest"], "plan_digest_kind": pdj["plan_digest_kind"],
        **({"attention": attention} if attention is not None else {}),
        "partition_digest": part, "partition_digest_kind": PARTITION_DIGEST,
        "kinds": {"schema": p2.KIND_SCHEMA, "key": "(stopped-node callee definition name, node form); 'batch' for a member of an adapter-unrolled batch",
                  "entries": [k.to_json() for k in kinds]},
        "row_fields": list(p2.ROW_FIELDS), "rows": rows, "cu_intervals": cu_ivs,
        "label": "MEASURED (built from the compact P1/cu=whole plan in two memory-bounded phases [scripts/serve_rows_pipeline.py]: rows = UnitRuns split at CU "
                 "boundaries and coalesced; ru = CU owning the unit's gates; plan NOT validated by validate_plan -- the rows are the plan's own UnitRuns)",
        "seconds": round(time.time() - t, 3),
    }
    po = p2.outside_record_from_meta(meta)
    if po is not None:
        doc["prescribed_outside_partition"] = po
    doc["content_sha256"] = p2.content_sha256(doc)
    p2.write(doc, out / p2.COMPACT_FILE)
    del plan, circuit, adapted, program, rows, kinds
    gc.collect()
    from record_v5 import population_v5 as p5
    v5 = p5.from_v2(out / p2.COMPACT_FILE)
    p5.write(v5, out / "population.v5.json")
    print(f"v5 N_V={v5['vu_count']} rows={len(v5['rows'])} digest={v5['partition_digest'][:16]} size={(out / 'population.v5.json').stat().st_size / 1e6:.1f}MB "
          f"total {time.time() - t:.1f}s rss {_rss()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
