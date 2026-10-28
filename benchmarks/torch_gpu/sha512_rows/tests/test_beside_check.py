"""beside's gate and summary on synthetic gate files: a digest as the kernel stores it (eight native u64 state words) reads
back as SHA-512's bytes, one flipped byte fails its kernel alone, and the summary's co-issue verdict reads the arms."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import beside_check
import beside_summary
from lowreg_check import row_v2, tile_rows

M, N, K, GRID, HASH_TILES = 64, 512, 128, 2, 2


def stored(digest: bytes) -> bytes:
    """SHA-512's 64 bytes as beside.cu writes them: state word i, a native little-endian u64, at byte 8 i."""
    return b"".join(int.from_bytes(digest[8 * i:8 * i + 8], "big").to_bytes(8, "little") for i in range(8))


def gate_dir(tmp: Path, flip: str | None = None) -> Path:
    g = tmp / "gate"
    g.mkdir()
    rows = tile_rows(5, 128, 512)
    (g / "seed.bin").write_bytes(b"".join(rows))
    y = bytes((7 * i + 3) & 255 for i in range(M * N * 2))
    head = {"sms": 132, "device": "test", "M": M, "N": N, "K": K, "grid": GRID}
    lines = []
    for k in ("h72_r", "h64_rs"):
        dig = bytearray(b"".join(stored(row_v2(y[2 * N * i:2 * N * (i + 1)])) for i in range(M)))
        if k == flip:
            dig[100] ^= 1
        (g / f"{k}-handoff.y").write_bytes(y)
        (g / f"{k}-handoff.digests").write_bytes(bytes(dig))
        (g / f"{k}-hash.digests").write_bytes(b"".join(stored(row_v2(rows[i % 128] * HASH_TILES))
                                                       for i in range(GRID * 128)))
        common = {"kernel": k, "regs": 128, "local_bytes": 0, "hash_tiles": HASH_TILES}
        lines += [head, {**common, "arm": "handoff", "cells": 8, "cells_wrong": 0, "rows": M},
                  {**common, "arm": "hash", "cells": 0, "cells_wrong": 0, "rows": GRID * 128}]
    (tmp / "gate.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    return g


def test_stored_digest_reads_back_as_sha512() -> None:
    d = hashlib.sha512(b"abc").digest()
    assert beside_check.digest_bytes(stored(d)) == d


def test_gate_passes_exact_digests(tmp_path: Path) -> None:
    g = gate_dir(tmp_path)
    assert beside_check.main(str(tmp_path / "gate.jsonl"), str(g), str(tmp_path / "passed.txt")) == 0
    assert (tmp_path / "passed.txt").read_text().split() == ["h72_r", "h64_rs"]


def test_a_flipped_byte_fails_its_kernel_alone(tmp_path: Path) -> None:
    g = gate_dir(tmp_path, flip="h64_rs")
    assert beside_check.main(str(tmp_path / "gate.jsonl"), str(g), str(tmp_path / "passed.txt")) == 1
    assert (tmp_path / "passed.txt").read_text().split() == ["h72_r"]


def test_hash_arm_slots_follow_the_cta_tile_and_the_half_buffer(tmp_path: Path) -> None:
    """Three hash warpgroups on a 64-row tile: slot g hashed row g % 64; the half O buffer: the row's first 256 bytes twice
    a tile. A digest of the whole row where the half was hashed fails."""
    g = tmp_path / "gate"
    g.mkdir()
    rows = tile_rows(5, 128, 512)
    (g / "seed.bin").write_bytes(b"".join(rows))
    head = {"sms": 132, "device": "test", "M": M, "N": N, "K": K, "grid": GRID}
    m1 = b"".join(stored(row_v2(rows[(i % 384) % 64] * HASH_TILES)) for i in range(GRID * 384))
    half = b"".join(stored(row_v2(rows[i % 128][:256] * 2 * HASH_TILES)) for i in range(GRID * 128))
    whole = b"".join(stored(row_v2(rows[i % 128] * HASH_TILES)) for i in range(GRID * 128))
    (g / "m1h3_rs-hash.digests").write_bytes(m1)
    (g / "h128_r_s4-hash.digests").write_bytes(half)
    (g / "h96_r_s4-hash.digests").write_bytes(whole)
    common = {"arm": "hash", "regs": 96, "local_bytes": 0, "hash_tiles": HASH_TILES, "cells": 0, "cells_wrong": 0}
    lines = [head, {**common, "kernel": "m1h3_rs", "nm": 1, "nh": 3, "tm": 64, "slots": 384, "half": 0, "rows": GRID * 384},
             {**common, "kernel": "h128_r_s4", "nm": 2, "nh": 1, "tm": 128, "slots": 128, "half": 1, "rows": GRID * 128},
             {**common, "kernel": "h96_r_s4", "nm": 2, "nh": 1, "tm": 128, "slots": 128, "half": 1, "rows": GRID * 128}]
    (tmp_path / "gate.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    assert beside_check.main(str(tmp_path / "gate.jsonl"), str(g), str(tmp_path / "passed.txt")) == 1
    assert (tmp_path / "passed.txt").read_text().split() == ["m1h3_rs", "h128_r_s4"]


def test_summary_reads_the_co_issue_from_the_arms(tmp_path: Path) -> None:
    def arm(kernel: str, name: str, ms: float, mma_us: float, hash_us: float, comps: float, **kw: object) -> dict:
        return {"kernel": kernel, "arm": name, "nwg": 4, "nform": 128, "stages": 3, "P": 24, "Y": 208, "H": 72,
                "var": "r", "regs": 128, "local_bytes": 0, "ms": ms, "mma_us": mma_us, "hash_us": hash_us,
                "compressions": comps, "tops": 1000.0 if mma_us else 0.0, "hash_gcomp_per_s": 0.0, "mhz": 1830.0,
                "watts": 500.0, "launches": 10, **kw}

    rows = [{"sms": 132, "device": "test", "M": 16384, "N": 4096, "K": 4096, "grid": 128},
            arm("wgl_n256_s4", "mma", 0.30, 290.0, 0.0, 0.0, nwg=3, nform=256, stages=4, P=40, Y=232, H=128),
            arm("h72_r", "mma", 0.31, 300.0, 0.0, 0.0), arm("h72_r", "hash", 0.50, 0.0, 480.0, 1e6),
            arm("h72_r", "free", 0.32, 315.0, 316.0, 6e5), arm("h72_r", "handoff", 0.52, 500.0, 505.0, 1.1e6)]
    (tmp_path / "beside-time.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = "\n".join(beside_summary.beside(tmp_path, {}))
    assert "takes 315.0 us against 300.0 us alone (+5.0%)" in out
    assert "1.040x the larger of its GEMM alone" in out
    assert "VERDICT beside budget: the fastest handoff is h72_r, P 24 + 2 x Y 208 + H 72 = 512" in out
    # served: 4 x 16384 x 4096 / 16384 x 133 = 2,179,072 compressions; 6e5 / 316 us beside the GEMM's 315 us
    # (1.899 G/s, 598,101 done), the rest at 1e6 / 480 us = 2.083 G/s: 315 + 1,580,971 / 2,083.3 = 1073.9 us
    assert "VERDICT beside served model h72_r" in out and "2.18 M compressions owed" in out
    assert "1073.9 us against serial" in out


def test_summary_budget_names_every_hash_warpgroup(tmp_path: Path) -> None:
    rows = [{"sms": 132, "device": "test", "M": 16384, "N": 4096, "K": 4096, "grid": 128},
            *({"kernel": "m1h3_rs", "arm": a, "nwg": 5, "nm": 1, "nh": 3, "tm": 64, "grid": 132, "nform": 64,
               "stages": 4, "ot": 1, "P": 24, "Y": 256, "H": 64, "var": "rs", "regs": 96, "local_bytes": 0, "ms": ms,
               "mma_us": mu, "hash_us": hu, "compressions": c, "tops": 900.0 if mu else 0.0, "hash_gcomp_per_s": 0.0,
               "mhz": 1980.0, "watts": 500.0, "launches": 10}
              for a, ms, mu, hu, c in (("mma", 0.7, 690.0, 0.0, 0.0), ("hash", 0.3, 0.0, 290.0, 1.2e6),
                                       ("free", 0.7, 692.0, 693.0, 2.4e6)))]
    (tmp_path / "beside-time.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = "\n".join(beside_summary.beside(tmp_path, {}))
    assert "VERDICT beside co-issue m1h3_rs (P 24 + Y 256 + 3 x H 64 = 472 <= 480 registers" in out
    assert "the GEMM beside 3 hash warpgroup(s)" in out


def test_stalls_beside_the_gemm_are_free_less_mma(tmp_path: Path) -> None:
    cols = ["ID", "Kernel Name", "smsp__warps_active.avg.per_cycle_active",
            "smsp__warps_issue_stalled_selected.avg.per_cycle_active",
            "smsp__warps_issue_stalled_wait.avg.per_cycle_active",
            "smsp__warps_issue_stalled_gmma.avg.per_cycle_active", "smsp__issue_active.avg.pct_of_peak_sustained_elapsed"]
    arms = {"mma": ("3", "0.10", "0.20", "1.50", "12"), "hash": ("1", "0.40", "0.30", "0", "41"),
            "free": ("4", "0.35", "0.75", "1.52", "42")}
    for arm, v in arms.items():
        (tmp_path / f"ncu-h72_r-{arm}.csv").write_text(
            "\n".join(",".join(f'"{x}"' for x in r) for r in (cols, [""] * len(cols), ["0", "beside", *v])) + "\n")
    stalls = {f"h72_r-{a}": beside_summary.ncu_stalls(tmp_path / f"ncu-h72_r-{a}.csv") for a in arms}
    assert stalls["h72_r-free"] == {"active": 4.0, "selected": 0.35, "wait": 0.75, "gmma": 1.52}
    (v,) = beside_summary.stall_verdicts(stalls)
    assert v.startswith("VERDICT beside stalls h72_r: the hash warps beside the GEMM (free - mma)")
    assert "wait 0.550 (alone 0.300), selected 0.250 (alone 0.400), gmma 0.020 (alone 0.000)" in v
