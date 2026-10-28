"""bench.py's host side and sass_gate.py, no GPU: `Params` and the arms against u2_gemm.cu, every tile's shared memory, the
default shape's balance on 132 SMs, the clock and power arithmetic, and the SASS gate on small listings."""
from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import bench as B  # noqa: E402
import sass_gate as SG  # noqa: E402

SRC = (HERE / "u2_gemm.cu").read_text()
#: the H100's and H200's opt-in shared memory a CTA (227 KiB)
SM90_SMEM = 232448


def const(name: str) -> int:
    """One of the GEMM's constants (`constexpr int BN = 256, BK = 128, ..., SMEM_BUDGET = 224 * 1024, ...`)."""
    m = re.search(rf"\b{name} = (\d+)(?: \* (\d+))?[,;]", SRC)
    return int(m.group(1)) * int(m.group(2) or 1)


def tile(nwg: int, prod: int) -> dict:
    """Tile<NWG, PROD>'s threads, shared memory and stages, as u2mm_layout reports them."""
    bk, budget, max_stages = const("BK"), const("SMEM_BUDGET"), const("MAX_STAGES")
    stage = (64 * nwg + 256) * bk
    stages = min(budget // stage, max_stages)
    return {"threads": 128 * (nwg + prod), "smem": 1024 + stages * stage + 16 * stages, "stages": stages}


def kernels() -> dict[str, tuple[int, int, int, int]]:
    """u2_gemm.cu's GEMM kernels: name -> (consumer warpgroups, producer, cluster, lag)."""
    out = {n: (int(w), int(p), int(c), 0) for n, w, p, c in re.findall(r"^U2MM\((\w+), (\d), (\d), (\d),", SRC, re.M)}
    out |= {n: (int(w), 0, int(c), 1) for n, w, c in re.findall(r"^U2MML\((\w+), (\d), (\d),", SRC, re.M)}
    return out


def test_params_is_the_kernels_layout():
    o, m, started, size = map(int, re.search(r"offsetof\(Params, o\) == (\d+) && offsetof\(Params, M\) == (\d+) && "
                                             r"offsetof\(Params, started\) == (\d+) &&\s+sizeof\(Params\) == (\d+)",
                                             SRC).groups())
    assert (B.Params.o.offset, B.Params.M.offset, B.Params.started.offset, B.C.sizeof(B.Params)) == (o, m, started, size)


def test_every_arm_is_a_kernel_of_the_build_and_the_gate_wants_it():
    built = kernels()
    assert {name for name, *_ in B.KERNELS.values()} == set(built)
    for name, nwg, prod, cl in B.KERNELS.values():
        assert built[name][:3] == (nwg, prod, cl)
    assert set(built) | {"hash_k", "stamp_k", "wait_k"} == SG.WANT
    assert all(SG.REG_CAP[n] == 168 for n in built if re.match(r"u2mm_ci[12]l?_", n))


def test_every_tile_fits_the_cards_shared_memory_and_its_swizzle():
    for nwg, prod in ((1, 0), (2, 0), (3, 0), (2, 1)):
        t = tile(nwg, prod)
        assert 3 <= t["stages"] <= const("MAX_STAGES") and t["smem"] <= SM90_SMEM
        assert (64 * nwg * const("BK")) % 1024 == 0 and ((64 * nwg + 256) * const("BK")) % 1024 == 0


class FakeLib:
    sms = 132

    def __init__(self):
        self.tiles = {k: tile(*k) for k in ((1, 0), (2, 0), (3, 0), (2, 1))}
        self.fns = dict.fromkeys(B.KERNELS)

    def clusters(self, arm: str) -> int:
        return 66

    def tmap(self, ptr: int, rows: int, depth: int, box_rows: int) -> bytes:
        assert box_rows <= 256
        return bytes(128)


def operand(*shape: int):
    return types.SimpleNamespace(shape=shape, data_ptr=lambda: 0)


@pytest.mark.parametrize("arm", sorted(B.KERNELS))
def test_the_default_shape_gives_every_cta_the_same_tiles(arm):
    m, n, k = B.parse_shape("25344x16384x32768")
    g = B.Gemm(FakeLib(), arm, operand(m, k), operand(n, k), operand(m, n))
    cl = B.KERNELS[arm][3]
    assert g.grid == 132 and g.rounds * g.grid // cl == g.units and g.describe()["balance"] == 1
    assert g.macs == m * n * k


def test_a_shape_off_the_tiles_is_refused():
    with pytest.raises(ValueError):
        B.Gemm(FakeLib(), "ci3_m2", operand(25344 - 192, 8192), operand(16384, 8192), operand(25344 - 192, 16384))


def stamps(nseg: int, nsm: int, ns: int, clocks: int) -> torch.Tensor:
    st = torch.zeros((nseg + 1, nsm, 6), dtype=torch.int64)
    for s in range(nseg + 1):
        st[s, :, 0] = 10**12 + s * ns
        st[s, :, 2] = 7 + s * clocks
        st[s, :, 4] = torch.arange(nsm - 1, -1, -1)
    return st


def run(st: torch.Tensor) -> B.Run:
    r = B.Run.__new__(B.Run)
    r.nsm, r.stamps = st.shape[1], st
    return r


def test_segments_count_every_sms_clocks_and_the_rates_follow():
    nsm, ns, clocks = 4, 10**6, 1830 * 10**3
    r = run(stamps(3, nsm, ns, clocks))
    segs = r.segments(3, [2, 2, 2], 4096 * nsm * clocks // 2, 10**12)
    assert [s["mhz"] for s in segs] == [1830.0] * 3 and segs[0]["clocks"] == nsm * clocks
    rates = B.Run.rates(segs, nsm)
    assert rates["of_4096"] == 1 and rates["macs_per_smclk"] == 4096 and rates["sm_mhz"] == pytest.approx(1830)
    assert rates["of_theta"] == pytest.approx(nsm / 132)
    assert B.Run.rates([], nsm) == {}


@pytest.mark.parametrize("bad", ["twice", "timed_out"])
def test_segments_refuse_stamps_not_one_on_every_sm(bad):
    st = stamps(2, 4, 10**6, 10**6)
    st[1, 0, 4] = st[1, 1, 4] if bad == "twice" else -1  # stamp_k's ~0 after its wait for every CTA timed out
    with pytest.raises(RuntimeError, match="one on every SM"):
        run(st).segments(2, [1, 1], 1, 10**12)


def test_mean_watts_are_the_energy_counters_slope():
    nv = B.Nvml.__new__(B.Nvml)
    # sampled every 10 ms, the counter stepping every 20 ms at 700 W
    nv.samples = [(t * 10**7, 1000 + 0.7 * 20 * (t // 2), 690.0, 650.0 + t, 1830, 0x1, 60) for t in range(101)]
    w = nv.window(0, 10**9)
    assert w["mean_w"] == pytest.approx(700) and w["max_w"] == 750 and w["avg_w_mean"] == 690
    assert w["throttle"] == ["0x1"] and w["nvml_sm_mhz"] == 1830
    assert nv.window(0, 10**7) == {"samples": 2}


def test_the_summary_has_a_row_an_item():
    rows = [{"name": "hash x1", "tag": "t", "round": 0, "of_bar": 0.5, "sm_mhz": 1980, "power_steady": {}},
            {"name": "ci3_m2", "tag": "t", "round": 0, "steady": {"of_4096": 0.98, "of_theta": 0.97, "sm_mhz": 1810},
             "power_steady": {"mean_w": 690.0, "max_w": 700.0, "throttle": ["0x4"]}, "pj_per_mac": 0.7},
            {"name": "ci2_m2+h1", "tag": "t", "round": 1, "hash": {"of_bar": 0.4}, "beside": {"of_4096": 0.9}}]
    text = B.summary(rows)
    assert text.count("\n") == 5 and "| ci3_m2 | t | 0 | 0.9800 | 0.9700 | 1810 | 690 | 700 | 0.700 | - | 0x4 |" in text
    assert "| 0.400 |" in text and "| 0.500 |" in text


PTXAS = """ptxas info    : Compiling entry function '{name}' for 'sm_90a'
    0 bytes stack frame, {spill} bytes spill stores, 0 bytes spill loads
ptxas info    : Used {regs} registers, used 2 barriers
"""
GEMM_SASS = "\t\tFunction : {name}\n" + "        /*0100*/   UTMALDG.2D [UR8], [UR4] ;\n" + \
    "        /*0110*/   {atom} R24, gdesc[UR4], R24 ;\n" * 4
OTHER_SASS = "\t\tFunction : {name}\n        /*0100*/   EXIT ;\n"


def listing(**change) -> tuple[str, str]:
    ptxas, sass = "", "\tcode for sm_90a\n"
    for name in sorted(SG.WANT):
        if name in change.get("drop", ()):
            continue
        c = change.get(name, {})
        regs = c.get("regs", 79 if name == "hash_k" else 160 if name.startswith("u2mm_") else 12)
        ptxas += PTXAS.format(name=name, spill=c.get("spill", 0), regs=regs)
        if name.startswith("u2mm_"):
            sass += GEMM_SASS.format(name=name, atom=c.get("atom", "IGMMA.64x256x32.S8.S8"))
        else:
            sass += OTHER_SASS.format(name=name)
    return ptxas + change.get("warn", ""), sass


def test_the_sass_gate_passes_the_s8_atom_and_every_kernel():
    funcs, problems = SG.gate(*listing())
    assert problems == [] and funcs["u2mm_ci3_m2"]["IGMMA"] == 4 and funcs["u2mm_ci3_m2"]["UTMALDG"] == 1


@pytest.mark.parametrize("change", [
    {"u2mm_ci2_m2": {"spill": 8}}, {"u2mm_ci2l_m2": {"regs": 170}}, {"hash_k": {"regs": 96}},
    {"u2mm_ws_m1": {"atom": "IGMMA.64x256x32.U8.U8"}}, {"u2mm_ws_m1": {"atom": "HGMMA.64x256x16.F32.BF16"}},
    {"drop": ("u2mm_ci3l_m1",)}, {"warn": "ptxas warning : wgmma serialized\n"}])
def test_the_sass_gate_refuses(change):
    _, problems = SG.gate(*listing(**change))
    assert problems
