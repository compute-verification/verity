"""`tp-commit`'s import path of record, read from the running processes."""
from __future__ import annotations

import json
from pathlib import Path


def write_import_path_runtime(llm, out: Path, rpc_timeout: float) -> None:
    """The import path OF RECORD, from the RUNNING processes: this driver's repo-local sys.modules + every rank process's (after
    finalize: replay, linkage, pin, partial source all imported) + the collector's JIT C++/CUDA sources -- what ran, not a static
    closure; the tag-candidate equivalence rule reads `import_path_runtime.json` when it exists."""
    try:
        from verity_vllm.pipeline.tp.rank_worker import repo_modules_loaded
        drv = repo_modules_loaded()
        ranks_mods = llm.collective_rpc("tp2_repo_modules", timeout=rpc_timeout)
        union = set(drv["files"])
        for rm in ranks_mods:
            union.update((rm or {}).get("files") or [])
        (out / "import_path_runtime.json").write_text(json.dumps({"schema": "verity/tp2/import-path-runtime/v1", "method": drv["method"],
                                                                   "driver": drv, "ranks": ranks_mods, "union_n": len(union), "union": sorted(union)},
                                                                  indent=1, default=str) + "\n")
        print(f"[tp2commit] import path of record (runtime): driver {drv['n']} files, ranks {[r.get('n') for r in ranks_mods]}, union {len(union)} -> import_path_runtime.json", flush=True)
    except Exception as e:  # noqa: BLE001 -- evidence, not a gate
        (out / "import_path_runtime.json").write_text(json.dumps({"schema": "verity/tp2/import-path-runtime/v1", "error": repr(e)[:300]}) + "\n")
