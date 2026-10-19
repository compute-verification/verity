"""fp_step_bounds.py, run as written: every served route's MMAs inside premise (a)'s regime on every input and order, the
words' reads and folds exact, the H200 and rebase controls outside, and every mutation caught. Its transcription of the
routes (planes, chains, starts, restarts, folds) is held here to the kernels and emu8.py, which the stdlib script can't
read."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

POUW = Path(__file__).resolve().parents[1]
SCRIPT = POUW / "theory" / "fp_step_bounds.py"
DC = POUW / "lowbit_dc"


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_fp_step_bounds(tmp_path):
    """The bounds, controls and replays are claims only running the script checks."""
    verdict = tmp_path / "verdict.json"
    proc = subprocess.run([sys.executable, "-B", str(SCRIPT), "--mutations", "--verdict", str(verdict)],
                          capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    doc = json.loads(verdict.read_text(encoding="utf-8"))
    assert doc["ok"]
    assert doc["counts"]["mutations"] == doc["counts"]["mutations_caught"] >= 9
    assert doc["counts"]["checks"] == doc["counts"]["checks_ok"] >= 120


def test_transcription_matches_emu8():
    """k3's and k4's planes and starts are emu8's."""
    fs = _load("fp_step_bounds", SCRIPT)
    emu = _load("emu8_emu8", POUW / "emu8" / "emu8.py")
    assert (fs.C0_BITS, fs.C1_BITS, fs.C2_BITS) == (emu.C0_BITS, emu.C1_BITS, emu.C2_BITS)
    for x in range(-128, 128):
        for side in (0, 1):
            assert list(fs.k3_planes(x, side)) == [int(v) for v in emu.planes(x, "k3", side)]
            assert list(fs.k4_planes(x, side)) == [int(v) for v in emu.planes(x, "k4", side)]
    k3, k4 = fs.ROUTES["k3"], fs.ROUTES["k4"]
    assert [a.start for a in k3.accs] == [emu.C1_BITS, emu.C2_BITS] and k4.accs[0].start == emu.C0_BITS
    assert k3.accs[0].mmas == ((0, 0), (2, 2)) and k3.accs[1].mmas == ((1, 1),) and k3.read == ("bits", (15, 2), 0)


def test_transcription_matches_kernels():
    """The B200 and H200 routes as lowbit_dc's kernels state them: if a kernel changes one of these lines, the script's
    route table must change with it."""
    fs = _load("fp_step_bounds", SCRIPT)
    fpx, egemm, pgemm = ((DC / f).read_text() for f in ("fpx.cu", "egemm.cu", "pgemm.cu"))
    assert "if (fmt == F_K3B || fmt == F_K3Z) return p == 0 ? 4 * h : p == 1 ? m : (bside ? l : -l);" in fpx
    assert "if (fmt == F_K3H) return p == 0 ? h : p == 1 ? m : l;" in fpx
    assert "return p == 0 ? 16 * h : l;" in fpx and "const int h = (x + 8) >> 4, l = x - 16 * h, m = h + l;" in fpx
    assert ("umma_f8f6f4(tmem, da(0), db(0), idesc, z);\n              umma_f8f6f4(tmem, da(2), db(2), idesc, 1);\n"
            "              umma_f8f6f4(tmem + 32, da(1), db(1), idesc, z);") in fpx  # k3: acc1 (0,0) then (2,2); acc2 (1,1)
    assert "__fadd_rn(__uint_as_float(g0), 12582912.f)" in fpx  # k4z's fold
    assert "if (!is_fp4(fmt)) return (int)(r % 33u) - 16;" in fpx and "c.s = grid == 0 ? 1 : grid == 1 ? 2 : 4;" in fpx
    assert "__fmaf_rn(__uint_as_float(g), ldexpf(1.f, -c.e), 12582912.f)" in fpx  # e4r's fold, 2^-e = s^2
    assert "fmaf(16.f, __uint_as_float(a1), fmaf(15.f, __uint_as_float(a0), MAGIC))" in egemm  # k3z
    assert "fmaf(-15.f, g2, fmaf(16.f, g1, fmaf(240.f, g0, MAGIC)))" in egemm  # k3h
    assert "4 L / 32 kind::f8f6f4 E4M3 MMAs a word onto one FP32 accumulator" in pgemm
    assert "constexpr int RB = L == 32 ? 7 : 3;" in pgemm
    r = fs.ROUTES
    assert r["k3z"].read == ("fold", ((0, 15), (1, 16))) and r["k3h"].read == ("fold", ((0, 240), (1, 16), (2, -15)))
    assert r["k4z"].read == ("fold", ((0, 1),)) and r["k4z"].accs[0].start == 0
    assert (r["k4t-L32"].words, r["k4t-L32"].slices, r["k4t-L64"].words, r["k4t-L64"].slices) == (7, 1, 3, 2)
    assert (r["k4s-L32"].slices, r["k4s-L64"].slices, r["k4s-L64"].words) == (1, 2, 1)
    assert [r[f"e4r-s{s}"].unit for s in (1, 2, 4)] == [1, fs.Q(1, 4), fs.Q(1, 16)]
