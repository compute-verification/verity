"""Every Definition a built Program reaches that is not Boolean, or with --dry-run what a word Program's Boolean Program would
reach today (`program.boolean.purity`, `dry_run`).

    verity-vllm boolean-purity DESCRIPTOR [--dry-run] [--out JSON]
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from verity_vllm.config import option
from verity.ml.definitions.boolean import dry_run, load_program, purity


@dataclass
class Options:
    """Every Definition a built Program reaches that is not Boolean (the floor's count), or with --dry-run what a word
    Program's Boolean Program would reach today."""
    desc: Path = option("desc", type=Path, help="a Program descriptor (descriptor.json[.gz]: a Build's request or workload Program)")
    dry_run: bool = option("--dry-run", default=False, action="store_true",
                           help="read a word Program: each derived Call's Boolean version or gap, and the would-be count")
    out: Path | None = option("--out", default=None, type=Path, help="write the report here (JSON) as well as to stdout")


def main(a: Options) -> int:
    program, digest = load_program(a.desc)
    rep = dry_run(program) if a.dry_run else purity(program)
    rep = {**rep, "program_digest": digest, "descriptor": str(a.desc)}
    text = json.dumps(rep, indent=1, sort_keys=True)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)
    nb = rep["non_boolean"]
    print(f"[boolean-purity] {rep['mode']}: {nb['definitions']} non-Boolean Definitions ({nb['specializations']} specializations)"
          + (f"; gaps: {', '.join(rep['gaps'])}" if rep.get("gaps") else ""), file=sys.stderr)
    return 0


__all__ = ["Options", "main"]
