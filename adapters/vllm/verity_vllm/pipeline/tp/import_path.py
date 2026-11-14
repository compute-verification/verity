"""`tp-commit`'s import path of record, read from the running processes."""
from __future__ import annotations

import json
from pathlib import Path

from verity_vllm.records.schema_ids import IMPORT_PATH_RUNTIME_SCHEMA


def write_import_path_runtime(llm, out: Path, rpc_timeout: float) -> None:
    """The import path OF RECORD, from the RUNNING processes: this driver's repo-local sys.modules + every rank process's (after
    finalize: replay, linkage, pin, partial source all imported) + the collector's JIT C++/CUDA sources -- what ran, not a static
    closure; the tag-candidate equivalence rule reads `import_path_runtime.json` when it exists."""
    try:
        from verity_vllm.pipeline.tp.rank_worker import repo_modules_loaded
        drv = repo_modules_loaded()
        ranks_mods = llm.collective_rpc("tp2_repo_modules", timeout=rpc_timeout)
        union = set(drv["files"])
        for rm in ranks_mods:               # each rank's `repo_modules_loaded()`; a rank that answered without its files adds none
            files = rm.get("files") if rm else None
            union.update(files if files else [])
        (out / "import_path_runtime.json").write_text(json.dumps({"schema": IMPORT_PATH_RUNTIME_SCHEMA, "method": drv["method"],
                                                                   "driver": drv, "ranks": ranks_mods, "union_n": len(union), "union": sorted(union)},
                                                                  indent=1, default=str) + "\n")
        print(f"[tp2commit] import path of record (runtime): driver {drv['n']} files, ranks {[r.get('n') for r in ranks_mods]}, union {len(union)} -> import_path_runtime.json", flush=True)
    except Exception as e:  # noqa: BLE001 -- evidence, not a gate
        (out / "import_path_runtime.json").write_text(json.dumps({"schema": IMPORT_PATH_RUNTIME_SCHEMA, "error": repr(e)[:300]}) + "\n")
