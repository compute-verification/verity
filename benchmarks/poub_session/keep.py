"""What a served line keeps of the row's Commit directory: the files `ROW` names, each public by the rule beside it, and
nothing else, so a private file the row writes (its tokens, replays, timelines, unsalted token digests) stays in run.sh's
scratch and is removed with it. Stdlib only; run.sh's PY runs it.

    keep.py row SRC DST     # each file of ROW that SRC holds, copied to DST under its relative path

A name private by the vLLM integration's list (`private_names.json`, one list for every keeper and for custody) is
refused by name, exit 3, before anything is copied. The matcher is the integration's own, `private_names.py` beside the
list, loaded by path: it needs only the standard library, and run.sh's PY need not hold `verity_vllm`.
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_MATCHER = ROOT / "adapters" / "vllm" / "verity_vllm" / "pipeline" / "private_names.py"
_spec = importlib.util.spec_from_file_location("verity_private_names", _MATCHER)
names = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(names)
PRIVATE_NAMES = names.LIST
private = names.private

#: the row's records a served line keeps, each with the public kind it is (AGENTS.md's four)
ROW = {
    "stages.txt": "the protocol's outputs: each stage's verdict line",
    "build_summary.json": "protocol structure: the Build's Program digests",
    "config_record.json": "the protocol's outputs: the config run's verdict",
    "commit/verdict.json": "the protocol's outputs: the Commit's verdict of record",
    "commit/remote_session.json": "the protocol's outputs: the PoUB session's verdict, the store's layout and counts",
}


#: an engine export's fields that travel to the prover through run.sh's scratch only (`engine.<g>.json` in --dir):
#: the arena's CUDA IPC handle opens the GPU's memory to whoever holds it while the engine runs
EXPORT_SCRATCH_ONLY = ("handle",)


def export_kept(export: dict) -> dict:
    """An engine's export as kept under the run's output: without its IPC handle."""
    return {k: v for k, v in export.items() if k not in EXPORT_SCRATCH_ONLY}


def checks_kept(checks: dict | None) -> dict | None:
    """An engine's decode checks as kept under the run's output: the probe's verdict (`probe_tokens_equal`, which the
    engine computes and `node.py`'s gate reads from the status in scratch) and each probe request's token count, never
    the tokens the model served it."""
    if checks is None:
        return None
    probe = checks.get("probe") or {}
    return {**{k: v for k, v in checks.items() if k != "probe"},
            "probe": {side: [len(t) for t in probe.get(side) or []] for side in ("plain", "decoded")}}


def keep(src: Path, dst: Path, rels: list[str]) -> list[str]:
    """Copy each of `rels` that `src` holds to `dst`; the copied ones. Refused (ValueError, naming them) where any is
    private, before anything is copied."""
    listed = names.load()
    refused = [r for r in rels if private(r, listed)]
    if refused:
        raise ValueError(f"refused: {', '.join(refused)} private by {PRIVATE_NAMES.relative_to(ROOT)}: never kept")
    kept = []
    for rel in rels:
        f = src / rel
        if f.is_file() and not f.is_symlink():
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dst / rel)
            kept.append(rel)
    return kept


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[0] != "row":
        print(__doc__, file=sys.stderr)
        return 2
    try:
        kept = keep(Path(argv[1]), Path(argv[2]), list(ROW))
    except ValueError as e:
        print(f"keep.py: {e}", file=sys.stderr)
        return 3
    print(f"keep.py: kept {len(kept)} of the row's public records: {' '.join(kept) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
