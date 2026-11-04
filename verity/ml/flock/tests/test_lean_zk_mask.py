"""D12 (`Flock.Zk`): under `--zk`, `Zk.shapeOk` refuses a statement whose Δ or regions meet the mask slot
(`Zk.maskFree`), the condition the zero-knowledge views take from the setup (`FailClosed.avoidsMask_of_shapeOk`). Upstream
checks neither; honest `--zk` statements avoid the slot, so no vector exercises it and this runs it on the Lean verifier's
check, at a mask range of words `[512, 1024)` in a block of `2^20` bits. And the slot's floor (`Flock.Zk`): `slot_log ≥ 13`,
which both sides check and no honest statement is below (the deployed `mask_log` is 14)."""
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


SLOTS = [
    # (slot_log, blocks log, verdict)
    (12, 5, '"slot_log = 12: --zk needs a mask slot of slot_log >= 13"'),
    (12, 10, '"slot_log = 12: --zk needs a mask slot of slot_log >= 13"'),
    (7, 15, '"slot_log = 7: --zk needs a mask slot of slot_log >= 13"'),
    (13, 5, '"ok"'),
    (14, 5, '"ok"'),
    (16, 5, '"ok"'),
    (13, 3, '"512 mask words: the rank check needs 512 for its 4 claim points, and 64 more"'),
]


def test_a_mask_slot_below_2_13_bits_is_refused(tmp_path):
    """`Zk.shapeOk` refuses `slot_log < 13` (`Zk.SLOT_LOG_MIN`) however many mask words the blocks give, ahead of the
    words check, and takes the deployed `mask_log = 14`."""
    subprocess.run(["lake", "build", "Flock"], cwd=LEAN, check=True, capture_output=True)
    f = tmp_path / "Slot.lean"
    f.write_text("import Flock\nopen Flock\n"
                 "def stmtOf (s b : Nat) : Stmt :=\n"
                 "  { (default : Stmt) with\n"
                 "    m := 25, nbl := b,\n"
                 "    c := { (default : Circuit) with\n"
                 "      kLog := 20, ranges := #[{ kind := .mask, slotLog := s, pos0 := 1, count := 1, role := .none }] } }\n"
                 "def verdict (s b : Nat) : String :=\n"
                 "  match Zk.shapeOk (stmtOf s b) with | .ok () => \"ok\" | .error e => e\n"
                 + "".join(f"#eval verdict {s} {n}\n" for s, n, _ in SLOTS))
    p = subprocess.run(["lake", "env", "lean", str(f)], cwd=LEAN, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.split("\n")[:len(SLOTS)] == [v for *_, v in SLOTS], p.stdout
