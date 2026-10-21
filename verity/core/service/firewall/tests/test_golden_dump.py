"""Scratch, never landed: freezes the Rust reference's outer sessions for the Lean (`FIREWALL_GOLDEN_OUT_DIR=DIR`): the
inputs `test_the_outer_session_agrees_with_rust` draws at its seed, each with Rust's answer, one JSON line each."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import _release as R
import verity_difftest as D
from test_firewall_release_agreement import SEED, lean, port, rust  # noqa: F401

OUT = os.environ.get("FIREWALL_GOLDEN_OUT_DIR")


@pytest.mark.skipif(not OUT, reason="FIREWALL_GOLDEN_OUT_DIR names where the frozen cases go")
def test_dump_the_outer_sessions(lean, rust):  # noqa: F811
    got = []
    through = R.through(rust, "outer")

    def ref(value):
        try:
            out = through(value)
        except D.Reject as e:
            got.append({"input": value, "reject": str(e)})
            raise
        got.append({"input": value, "output": out})
        return out

    stats = D.run(port("outer", ref, R.outer_input()), examples=1000, seed=SEED, server=lean)
    print(f"outer vs Rust, seed {SEED}: {stats}")
    Path(OUT).mkdir(parents=True, exist_ok=True)
    (Path(OUT) / "outer.jsonl").write_text("".join(json.dumps(g, sort_keys=True) + "\n" for g in got))
