"""What this directory's GPU scripts share to serve an int8 pin: `pin` finds the registered int8 pin of a role
(`checkpoints-int8.json`) or makes one with `int8-checkpoint`, `scratch_checkpoints` writes the scratch copies of the
manifests the engine reads it through, and PROMPTS are the prompts they generate on."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Any

from verity_vllm.config import checkpoint_entries
from verity_vllm.engine import int8_checkpoint as I

PROMPTS = ["The capital of France is", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
           "The three primary colors are", "Water boils at a temperature of", "Once upon a time, in a small village,"]


def pin(a: argparse.Namespace) -> tuple[dict[str, Any], str]:
    """The int8 pin of `a.role`: the registered one (re-made and checked by `int8-checkpoint make` when its snapshot is missing
    here), else a new one (`int8-checkpoint pin`)."""
    out = io.StringIO()
    entry = next((c for c in checkpoint_entries(a.checkpoints) if c.get("role") == a.role), None)
    if entry is not None and Path(entry["local_path"]).is_dir():
        return entry, "registered"
    with contextlib.redirect_stdout(out):
        rc = I.main(I.MakeOptions(role=a.role, checkpoints=a.checkpoints, hf_home=a.hf_home) if entry is not None else
                    I.PinOptions(base_role=a.base_role, role=a.role, repo=a.repo, checkpoints=a.checkpoints, hf_home=a.hf_home,
                                 by=a.by, at=a.at))
    if rc:
        raise SystemExit(f"int8-checkpoint exited {rc}: {out.getvalue()[-2000:]}")
    return (entry, "remade") if entry is not None else (json.loads(out.getvalue()), "made")


def scratch_checkpoints(checkpoints: str, entry: dict[str, Any], scratch: Path) -> Path:
    p = scratch / "checkpoints.json"
    p.write_text(Path(checkpoints).read_text())
    sib = Path(checkpoints).with_name("checkpoints-int8.json")
    doc = json.loads(sib.read_text()) if sib.is_file() else {"checkpoints": []}
    doc["checkpoints"] = [c for c in doc["checkpoints"] if c.get("role") != entry["role"]] + [{**entry, "downloaded": True}]
    (scratch / "checkpoints-int8.json").write_text(json.dumps(doc, indent=1) + "\n")
    return p
