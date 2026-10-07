"""Confirm the compiled extension contains native route-U and signed-control tensor-core instructions."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from kernels import load  # noqa: E402


def main() -> None:
    ext = load()
    sass = subprocess.run(
        ["/usr/local/cuda/bin/cuobjdump", "-sass", ext.__file__],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    result = {
        "extension": ext.__file__,
        "imma_u8_s8": sass.count("IMMA.16816.U8.S8"),
        "imma_s8_s8": sass.count("IMMA.16816.S8.S8"),
    }
    print(json.dumps(result, sort_keys=True))
    if not result["imma_u8_s8"] or not result["imma_s8_s8"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
