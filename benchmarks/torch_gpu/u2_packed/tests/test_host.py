"""bench.py's host side, the packed format and sass_gate.py, no GPU: `Params` and the arms against packed.cu, the format's
pack and unpack, the kernel's unpacking arithmetic and its swizzled stores, every layout's shared memory, and the SASS gate
on small listings."""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import bench as B  # noqa: E402
import sass_gate as SG  # noqa: E402

SRC = (HERE / "packed.cu").read_text()
#: the H100's and H200's opt-in shared memory a CTA (227 KiB), an SM's (228 KiB) and its register file
SM90_CTA_SMEM, SM90_SM_SMEM, SM90_REGS = 232448, 233472, 65536


def kernels() -> dict[str, tuple[str, str, int, int]]:
    """packed.cu's GEMM kernels: name -> (load, mma, unpacked slots, packed slots)."""
    return {n: (load, mma, int(su), int(sp))
            for n, load, mma, su, sp in re.findall(r"^PK\((\w+), (\w+), (\w+), (\d), (\d)\)", SRC, re.M)}


def smem(su: int, sp: int, load: str) -> int:
    """Layout<LOAD, SU, SP>::SMEM."""
    stage = (B.TM + B.BN) * B.BK
    nbar = su if load == "I8" else sp
    return 1024 + su * stage + sp * stage // 4 + 8 * nbar


def test_params_is_the_kernels_layout():
    o, m, started, size = map(int, re.search(r"offsetof\(Params, o\) == (\d+) && offsetof\(Params, M\) == (\d+) && "
                                             r"offsetof\(Params, started\) == (\d+) &&\s+sizeof\(Params\) == (\d+)",
                                             SRC).groups())
    assert (B.Params.o.offset, B.Params.M.offset, B.Params.started.offset, B.C.sizeof(B.Params)) == (o, m, started, size)


def test_every_arm_is_a_kernel_of_the_build_and_the_gate_wants_it():
    built = kernels()
    assert {name for name, *_ in B.KERNELS.values()} == set(built) == set(SG.GEMMS)
    for arm, (name, kind, lay, mma) in B.KERNELS.items():
        load, kmma, su, sp = built[name]
        assert (kmma == "true") == mma and (SG.GEMMS[name] is not None) == mma
        assert (kind == "i8") == (load == "I8")
        assert SG.GEMMS[name] in (None, "U8" if load == "P2U" else "S8")
        assert SG.REG_CAP[name] == 168
    layouts = re.findall(r"out\[\d\] = Layout<(\w+), (\d), (\d)>::SMEM;", SRC)
    assert [(load, int(su), int(sp)) for load, su, sp in layouts] == [("I8", 4, 0), ("P2", 3, 6), ("RES", 4, 1)]
    for name, (load, _, su, sp) in built.items():
        lay = next(v[2] for v in B.KERNELS.values() if v[0] == name)
        assert layouts[lay][1:] == (str(su), str(sp))


def test_every_layout_fits_beside_a_hash_cta():
    for load, su, sp in (("I8", 4, 0), ("P2", 3, 6), ("RES", 4, 1)):
        s = smem(su, sp, load)
        assert s <= SM90_CTA_SMEM
        assert s + 1024 + 1024 <= SM90_SM_SMEM  # each CTA's reserved kilobyte, hash_k's no dynamic shared memory
    assert 256 * 168 + B.HASH_THREADS * SG.REG_CAP["hash_k"] <= SM90_REGS


def test_pack_and_unpack_are_inverse_and_the_format_is_the_documented_one():
    g = torch.Generator().manual_seed(1)
    v = torch.randint(0, 4, (5, 64), dtype=torch.int8, generator=g)
    p = B.pack(v)
    assert p.shape == (5, 16) and p.dtype == torch.uint8
    assert torch.equal(B.unpack(p), v)
    for r in range(5):
        for k in range(64):
            byte, i = 4 * (k // 16) + k % 4, (k // 4) % 4
            assert (int(p[r, byte]) >> (2 * i)) & 3 == int(v[r, k])
    q = torch.randint(0, 256, (3, 32), dtype=torch.uint8, generator=g)
    assert torch.equal(B.pack(B.unpack(q)), q)


def words(x: int, u8: bool) -> list[int]:
    """The kernel's four unpacked words of packed word x (packed.cu's unpack)."""
    if u8:
        return [((x << s) & 0xFFFFFFFF) & 0xC0C0C0C0 for s in (6, 4, 2, 0)]
    return [(x >> s) & 0x03030303 for s in (0, 2, 4, 6)]


def test_the_kernels_unpacking_is_the_formats():
    assert "sts128(at, x & M2, (x >> 2) & M2, (x >> 4) & M2, (x >> 6) & M2)" in SRC
    assert "sts128(at, (x << 6) & H2, (x << 4) & H2, (x << 2) & H2, x & H2)" in SRC
    rnd = random.Random(2)
    for _ in range(200):
        x = rnd.getrandbits(32)
        row = torch.tensor([list(x.to_bytes(4, "little"))], dtype=torch.uint8)
        vals = B.unpack(row)[0].tolist()  # 16 values of one word
        plain = b"".join(w.to_bytes(4, "little") for w in words(x, False))
        assert list(plain) == vals
        scaled = b"".join(w.to_bytes(4, "little") for w in words(x, True))
        assert list(scaled) == [64 * v for v in vals]
    # u8 x 64 sums stay below 2^31 at the largest K the kernel's comment allows
    assert 58254 * (3 * 64) ** 2 < 2 ** 31 <= 58255 * (3 * 64) ** 2


def test_the_unpacking_writes_every_chunk_of_the_swizzled_slot_once_without_bank_conflicts():
    """packed.cu's unpack: thread tid, pass j: row (tid >> 1) + 128 j, half tid & 1, chunks 4 half + q at
    row * 128 + ((c ^ (row % 8)) << 4), SWIZZLE_128B's place for chunk c of the row."""
    seen = {}
    for j in range(3):
        for q in range(4):
            for warp in range(8):
                banks = []
                for lane in range(32):
                    tid = 32 * warp + lane
                    r, h = (tid >> 1) + 128 * j, tid & 1
                    c = 4 * h + q
                    at = r * B.BK + ((c ^ ((tid >> 1) & 7)) << 4)
                    assert (r & 7) == ((tid >> 1) & 7) and at == r * 128 + ((c ^ (r % 8)) << 4)
                    assert (r, c) not in seen
                    seen[(r, c)] = at
                    banks.append((at % 128) // 16)
                # 32 stores of 16 bytes: four wavefronts at best, each 16-byte bank group hit four times
                assert sorted(banks.count(b) for b in set(banks)) == [4] * 8
    assert set(seen) == {(r, c) for r in range(B.TM + B.BN) for c in range(8)}


LISTING_PTXAS = "\n".join(
    f"ptxas info    : Compiling entry function '{n}' for 'sm_90a'\nptxas info    : Function properties for {n}\n"
    f"    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\nptxas info    : Used {r} registers"
    for n, r in [*((g, 168) for g in SG.GEMMS), ("pk_layout", 8), ("hash_k", 80), ("stamp_k", 16), ("wait_k", 8)])


def listing(mma: dict[str, str]) -> str:
    out = []
    for n in sorted(SG.WANT):
        out.append(f"\t\tFunction : {n}")
        if n in SG.GEMMS:
            out.append("        /*0100*/                   UTMALDG.2D [UR8], [UR4] ;")
            for i in range(4 if SG.GEMMS[n] else 0):
                form = mma.get(n, f"IGMMA.64x256x32.{SG.GEMMS[n]}.{SG.GEMMS[n]}")
                out.append(f"        /*02{i}0*/                   {form} R24, gdesc[UR4], R24 ;")
            out.append("        /*0300*/                   LOP3.LUT R3, R2, 0x3030303, RZ, 0xc0, !PT ;")
    return "\n".join(out)


def test_the_gate_passes_a_clean_build_and_counts_opcodes():
    funcs, problems = SG.gate(LISTING_PTXAS, listing({}))
    assert problems == []
    assert funcs["pk_p2"]["ops"]["LOP3"] == 1 and funcs["pk_p2"]["IGMMA"] == 4 and funcs["pk_p2x"]["IGMMA"] == 0


def test_the_gate_refuses_spills_wrong_atoms_and_registers():
    _, problems = SG.gate(LISTING_PTXAS.replace("0 bytes spill stores", "8 bytes spill stores", 1), listing({}))
    assert any("local memory" in p for p in problems)
    _, problems = SG.gate(LISTING_PTXAS, listing({"pk_p2": "IGMMA.64x256x32.U8.U8"}))
    assert any("pk_p2:" in p and "atom" in p for p in problems)
    _, problems = SG.gate(LISTING_PTXAS.replace("Used 80 registers", "Used 88 registers"), listing({}))
    assert any("hash_k: 88 registers" in p for p in problems)
    _, problems = SG.gate(LISTING_PTXAS + "\nptxas warning : wgmma serialized", listing({}))
    assert problems
