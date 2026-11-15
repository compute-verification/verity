"""dc/pipe.py's build gate and its ε_tensor at L = 32, on synthetic inputs: the gate refuses a pg100 kernel whose
registers a block of its warps can't launch with, and the adversary's two bounds are the load balance and that held to
the round trip. The kernels themselves need a B200 (dc/pipe.sh)."""
from __future__ import annotations

import types
from pathlib import Path

import pipe

#: pgemm.cu's 38 kernels (NB, LW, MODE, MI): four (NB, LW) in four modes and (4, 4) without the honest mode, each with
#: one MMA warp and with one a tile
KERNELS = [(nb, lw, mode, mi) for nb, lw in ((2, 1), (2, 2), (4, 1), (4, 2), (4, 4)) for mode in range(4)
           for mi in (0, 1) if (lw, mode) != (4, 2)]


def _build(out: Path, registers: dict[tuple[int, int, int, int], int]) -> None:
    """A build's log and SASS as pipe.sh leaves them: ptxas's register and spill reports, and each kernel's opcodes."""
    log, sass = [], []
    for nb, lw, mode, mi in KERNELS:
        name = f"_Z5pg100ILi{nb}ELi{lw}ELi{mode}ELb{mi}EEvPKaS1_iiiijPiPjPx"
        log += [f"ptxas info    : Compiling entry function '{name}' for 'sm_100a'",
                f"ptxas info    : Used {registers.get((nb, lw, mode, mi), 64)} registers, used 1 barriers",
                "ptxas info    : 0 bytes spill stores, 0 bytes spill loads"]
        ops = ["UTCIMMA"] + ([] if mode in (0, 3) else ["LDTM.x32"]) + ["EXIT"]
        sass.append(f"\t\tFunction : {name}")
        sass += [f"        /*{16 * i:04x}*/                   {op} R0, R1 ;  /* 0x0 */" for i, op in enumerate(ops)]
    (out / "sass").mkdir(parents=True, exist_ok=True)
    (out / "build.log").write_text("\n".join(log) + "\n")
    (out / "sass" / "pgemm.sass").write_text("\n".join(sass) + "\n")


def test_gate_build_holds_each_kernel_to_the_registers_its_block_launches_with(tmp_path, capsys):
    """An SMSP's 16,384 registers over the warps it holds, in threads of 32, rounded down to 8 and at most 255: 12 warps
    (NB = 4, LW = 2, an MMA warp a tile) launch with 168 registers and 5 warps (NB = 2, LW = 1) with 255."""
    assert len(KERNELS) == pipe.KERNELS
    _build(tmp_path, {(4, 2, 2, 1): 168, (2, 1, 2, 0): 255})
    assert pipe.gate_build(tmp_path) == 0
    passed = capsys.readouterr().out
    assert "PASS pg100<4,2,2,1>: UTCIMMA 1, LDTM.x32 1, LDL/STL 0, registers 168 (launchable 168)" in passed
    assert "registers 255 (launchable 255)" in passed and "REFUSED" not in passed
    _build(tmp_path, {(4, 2, 2, 1): 169})
    assert pipe.gate_build(tmp_path) == 1
    refused = capsys.readouterr()
    assert "REFUSED pg100<4,2,2,1>: UTCIMMA 1, LDTM.x32 1, LDL/STL 0, registers 169 (launchable 168)" in refused.out
    assert refused.err == "REFUSED pg100<4,2,2,1>\n"


def test_eps_tensor_l32_bounds_the_adversary_by_the_balance_and_by_the_round_trip(monkeypatch):
    """With ρ_T5 = 256, honest's TMEM stage at 128 words/SM/clk and the best of the pipeline and self-wait arms at 160
    slots/SM/clk: T1 is 1/2; the adversary at the load balance (0.9) keeps 230.4 tcgen05 words, and held to the round
    trip min(0.9, 160/256) = 0.625 of ρ_T5, each with the shared pipe's slack at its t in mma.sync (here 64 (1 − t))."""
    class Device:
        def __init__(self, family, dirs):
            assert family == "sm100"

        def model(self):
            return {"tmem_points": [(1.0, 0.5)], "sync_points": "sync", "honest_fraction_L32": 0.9, "eps_tensor_L32": 0.1}

        def tensor(self):
            return {"IMMA_16832": {"rate": 2.0, "slots_per": 32.0}, "UTCIMMA_M128_N256": {"rate": 1.0, "slots_per": 1e6}}

    analyze = types.SimpleNamespace(balanced=lambda region: 0.9, upper=lambda points, t: 1.0 - t)
    monkeypatch.setattr(pipe, "_module", lambda name, path: types.SimpleNamespace(analyze=analyze, Device=Device))
    rates = [{"arm": 1, "L": 32, "slots_per_clk": 128.0, "committed_words_per_clk": 128.0, "nb": 4, "lw": 2, "mi": 1},
             {"arm": 0, "L": 32, "slots_per_clk": 150.0, "committed_words_per_clk": 0.0, "nb": 4, "lw": 1, "mi": 1},
             {"arm": 3, "L": 32, "slots_per_clk": 160.0, "committed_words_per_clk": 0.0, "nb": 2, "lw": 1, "mi": 0}]
    e = pipe.eps_tensor_l32([], rates, 256.0)
    assert e["T1"] == 0.5 and e["honest_config"] == {"nb": 4, "lw": 2, "mi": 1} and e["rho_sync"] == 64.0
    assert e["round_trip_config"] == {"nb": 2, "lw": 1, "mi": 0, "arm": 3}
    assert e["bandwidth"] == {"t": 0.9, "tcgen05_words": 0.9 * 256, "mma_sync_words": (1.0 - 0.9) * 64,
                              "adversary_words": 0.9 * 256 + (1.0 - 0.9) * 64,
                              "eps": 1 - 128 / (0.9 * 256 + (1.0 - 0.9) * 64)}
    assert e["round_trip"]["t"] == 0.625 and e["round_trip"]["adversary_words"] == 160.0 + 0.375 * 64
    assert e["round_trip"]["eps"] == 1 - 128 / 184
    assert "round_trip" not in pipe.eps_tensor_l32([], rates[:1], 256.0)
    assert pipe.eps_tensor_l32([], rates[1:], 256.0) == {}
