"""Writes `lean_decode_texts.json`, Lean's refusal text on each refusing case of `decode_difftest.cases()`:

    uv run python tests/tools/difftest/write_lean_texts.py tests/tools/difftest/lean_decode_texts.json
"""
from __future__ import annotations

import json
import sys

import test_program_decode_port as T
import verity_difftest as D


def texts() -> dict[str, str]:
    out = {}
    with D.Server(T.DECODE.argv()) as s:
        for name, text, _registry in T.CASES:
            verdict = json.loads(s.call("decode_text", {"descriptor": text})[1])
            if not verdict["ok"]:
                out[name] = verdict["message"]
    return out


if __name__ == "__main__":
    with open(sys.argv[1], "w") as f:
        json.dump(texts(), f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")
