"""`Menu.Staged` at F6's pinned menus, checked by running V*: `vstar menu` runs compiled `VStar.statements` over every
pinned menu's inner session (`VStar.Menu.sessions`, each shape's largest entry at its entries' most claims), which is
the premise `FlockSampled.Menu.ofEntries_staged` takes. The proof, a bound on `VStar.parts`, is deferred.

These run V* at the deployed sizes, which nothing cheaper does: the kernel cannot evaluate `statements` there, and the
difftest compares Lean with Python on drawn small shapes, not on the pinned ones. A menu or a parts limit that leaves a
pinned shape unstaged fails here and nowhere else before an audit refuses it."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import verity_difftest as D
from verity_flock import rec_algebra as A
from verity_flock import rec_residuals as RR

LEAN = Path(__file__).resolve().parents[1]
#: an inner session whose batched-T residual alone needs 300 sha512x3 slots (past `MAX_LONE_SLOTS`), under `MAX_REGIONS`
CONTROL = (26, 24, 28)
pytestmark = [pytest.mark.slow, pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")]


def _menu(*shape: int) -> tuple[int, list[dict]]:
    p = subprocess.run([str(D.build(LEAN, "vstar")), "menu", *map(str, shape)], capture_output=True, text=True)
    assert p.returncode in (0, 1), p.stderr
    return p.returncode, [json.loads(line) for line in p.stdout.splitlines()]


def test_v_star_stages_every_pinned_inner_session():
    code, rows = _menu()
    assert [r for r in rows if "refused" in r] == [] and code == 0
    assert all(r["statements"] > 0 for r in rows)
    tail = [(r["m"], r["k_log"], r["claims"]) for r in rows
            if any(w.startswith("vllm/llama32-1b__") and "__i256__" in w for w in r["models"])]
    assert (35, 27, 16) in tail


def test_v_star_refuses_a_session_past_the_parts_limit():
    code, rows = _menu(*CONTROL)
    assert code == 1 and len(rows) == 1
    S = RR.spec(A.Shape(*CONTROL))
    with pytest.raises(ValueError) as e:
        RR.parts(S)
    assert rows[0]["refused"] == str(e.value)
    assert RR.parts(S, max_lone_slots=300) and RR.MAX_LONE_SLOTS < 300
