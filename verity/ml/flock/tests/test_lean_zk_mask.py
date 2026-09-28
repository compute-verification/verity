"""D12 (PROTOCOL.md §16.14): under `--zk`, `Zk.shapeOk` refuses a statement whose Δ or regions meet the mask slot
(`Zk.maskFree`), the condition the zero-knowledge views take from the setup (`FailClosed.avoidsMask_of_shapeOk`). Upstream
checks neither; honest `--zk` statements avoid the slot, so no vector exercises it and this runs it on the Lean verifier's
check, at a mask range of words `[512, 1024)` in a block of `2^20` bits."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_lean_verifier import LEAN  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

CASES = [
    # (Δ pair, region free bits, region fixed bits, refused)
    ((0, 128), [0, 1, 2, 3, 4, 5, 6], 0, False),
    ((511 * 128 + 127, 1024 * 128), [0, 1, 2, 3, 4, 5, 6], 0, False),
    ((512 * 128 + 5, 0), [0, 1, 2, 3, 4, 5, 6], 0, True),
    ((0, 1023 * 128), [0, 1, 2, 3, 4, 5, 6], 0, True),
    ((0, 128), [0, 1, 2, 3, 4, 5, 6], 600 * 128, True),
    ((0, 128), [0, 1, 2, 3, 4, 5, 6, 16], 0, True),
    ((0, 128), [0, 1, 2, 3, 4, 5, 6, 19], 0, False),
]


def test_a_delta_pair_or_region_in_the_mask_slot_is_refused(tmp_path):
    subprocess.run(["lake", "build", "Flock"], cwd=LEAN, check=True, capture_output=True)
    f = tmp_path / "Mask.lean"
    evals = "".join(f"#eval verdict ({a}, {b}) #{free} {fixed}\n" for (a, b), free, fixed, _ in CASES)
    f.write_text("import Flock\nopen Flock\n"
                 "def stmtOf (p : Nat × Nat) (reg : Region) : Stmt :=\n"
                 "  { (default : Stmt) with\n"
                 "    m := 25, nbl := 5, da := #[p], db := #[(0, 128)], regions := #[reg],\n"
                 "    c := { (default : Circuit) with\n"
                 "      kLog := 20, ranges := #[{ kind := .mask, slotLog := 16, pos0 := 1, count := 1, role := .none }] } }\n"
                 "def verdict (p : Nat × Nat) (free : Array Nat) (fixed : Nat) : String :=\n"
                 "  match mkRegion 20 \"r\" free fixed fun _ => .empty with\n"
                 "  | .error e => e\n"
                 "  | .ok reg => match Zk.shapeOk (stmtOf p reg) with | .ok () => \"ok\" | .error e => e\n" + evals)
    p = subprocess.run(["lake", "env", "lean", str(f)], cwd=LEAN, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    refused = '"zk: a pair of Δ or a region meets the mask slot (D12: the zero-knowledge proof covers no such statement)"'
    assert p.stdout.split("\n")[:len(CASES)] == [refused if r else '"ok"' for *_, r in CASES], p.stdout
