"""gpu_kit: the bank model, the dump format, the twin compare, and the timing gate on real sm_120a SASS.

The SASS fixtures (`fixtures/gpu_kit/*.nvdisasm.txt`) are `nvdisasm -c` listings cut to a few kernels: `round7_lm` is
RS-W′'s round-7 lm.cubin, whose LDSM loop ptxas merged and whose STSM addresses conflict 8-way;
the others are `opcode_probe/probes/*.cu` built with nvcc 13.0 for sm_120a. `*.list.json` are those binaries' `--list`
tables (the shared-memory offsets the bank check reads), cut to the same kernels.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

POUW = Path(__file__).resolve().parents[1]
FIX = Path(__file__).resolve().parent / "fixtures" / "gpu_kit"


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


G = _load("gpu_kit_sass_gate", POUW / "gpu_kit" / "sass_gate.py")
banks = _load("gpu_kit_banks", POUW / "gpu_kit" / "banks.py")
dump = _load("gpu_kit_dump", POUW / "gpu_kit" / "dump.py")
compare = _load("gpu_kit_compare", POUW / "gpu_kit" / "compare.py")
H = G.H

R7_LDSM, R7_STSM = "_Z4k_lmILi1EEvP8BlockClkPii", "_Z4k_lmILi8EEvP8BlockClkPii"


def functions(name: str, edit=None) -> dict:
    text = (FIX / f"{name}.nvdisasm.txt").read_text()
    return {f["name"]: H.Function(f) for f in H.parse_nvdisasm(edit(text) if edit else text)}


def table(name: str) -> dict:
    return {p["name"]: p for p in json.loads((FIX / f"{name}.list.json").read_text())}


def checks(rec: dict) -> set[str]:
    return {r.split(":", 1)[0] for r in rec["reasons"]}


# ---- banks -------------------------------------------------------------------------------------------------------------


def lm_off(slot: int, lane: int, warp: int, it: int) -> int:
    """probes/lm.cu's ring: lane l's row is base + l + 8 slot, the base moving 64 rows a warp and an iteration."""
    return (((lane + 64 * (warp + it)) & 2047) + 8 * slot) * 16


def r7_off(slot: int, lane: int, warp: int, it: int = 0) -> int:
    """Round 7's lm.cu address, sh16(smem, threadIdx.x * 8 + u): a 128-B lane stride."""
    return ((((warp * 32 + lane) * 8 + slot) * 4) & 4095) * 4


def test_ring_rows_are_conflict_free_and_round7_is_8_way():
    for op in ("LDSM.16.M88", "LDSM.16.M88.2", "LDSM.16.M88.4", "STSM.16.M88.4", "LDS.128", "STS.128"):
        for w, it in ((0, 0), (1, 1), (31, 7)):
            assert all(banks.degree(op, [lm_off(s, l, w, it) for l in range(32)]) == 1 for s in range(8)), op
    for op in ("LDSM.16.M88", "STSM.16.M88"):
        d, phase = banks.worst(op, [r7_off(0, l, 0) for l in range(32)])
        assert d == 8 and phase == [128 * i for i in range(8)]


def test_word_accesses_by_width():
    assert banks.degree("LDS", [4 * l for l in range(32)]) == 1
    assert banks.degree("LDS", [8 * l for l in range(32)]) == 2
    assert banks.degree("LDS", [0] * 32) == 1  # one word: a broadcast
    assert banks.degree("LDS.64", [8 * l for l in range(32)]) == 1  # 16 lanes a phase
    assert banks.degree("LDS.128", [16 * l for l in range(32)]) == 1  # 8 lanes a phase
    assert banks.degree("LDS.128", [32 * l for l in range(32)]) == 2
    assert banks.degree("ATOMS.ADD", [128 * l for l in range(32)]) == 32
    assert banks.degree("LDS", [None] * 31 + [4]) == 1


def test_matrix_registers():
    assert [banks.matrix_registers(op) for op in ("LDSM.16.M88", "LDSM.16.MT88.2", "STSM.16.M88.4")] == [1, 2, 4]
    assert banks.matrix_registers("LDSM.8.MT1616.2") == 4
    with pytest.raises(ValueError, match="31 lane offsets"):
        banks.phases("LDS", [0] * 31)


# ---- the dump format and the twin compare ------------------------------------------------------------------------------


def native_words(digest: bytes) -> bytes:
    """A SHA-512 digest as a GPU holds it: 8 uint64 words, each stored little-endian."""
    return b"".join(struct.pack("<Q", int.from_bytes(digest[i:i + 8], "big")) for i in range(0, 64, 8))


def test_dump_round_trip_and_its_checks(tmp_path):
    d = [hashlib.sha512(bytes([i])).digest() for i in range(3)]
    assert dump.digests_from_words(b"".join(native_words(x) for x in d)) == b"".join(d)
    dump.write(tmp_path, "out", b"\x01\x02\x03\x04")
    dump.write(tmp_path, "leaves", b"".join(d), "sha512-be")
    got = dump.read(tmp_path)
    assert got["out"].data == b"\x01\x02\x03\x04" and got["leaves"].layout == "sha512-be"
    with pytest.raises(dump.DumpError, match="64-byte digests"):
        dump.write(tmp_path, "x", b"\0" * 63, "sha512-be")
    with pytest.raises(dump.DumpError, match="layout"):
        dump.write(tmp_path, "x", b"", "le")
    (tmp_path / "out.bin").write_bytes(b"\x01")
    with pytest.raises(dump.DumpError, match="cut short"):
        dump.read(tmp_path)
    dump.write(tmp_path / "two", "a", b"")
    dump.write(tmp_path / "two", "a", b"")
    with pytest.raises(dump.DumpError, match="dumped twice"):
        dump.read(tmp_path / "two")
    with pytest.raises(dump.DumpError, match="no dump.jsonl"):
        dump.read(tmp_path / "none")


def test_items_reads_bench_lines():
    out = 'GK arm=a reps=1\n{"gk": "item", "arm": "a", "rep": 0, "ms": 1.5}\n{"op": "time"}\n'
    assert dump.items(out) == [{"gk": "item", "arm": "a", "rep": 0, "ms": 1.5}]


def test_compare_names_the_known_faults():
    want = bytes(range(64))
    assert compare.compare(want, want)["equal"]
    unwritten = want[:16] + b"\xa5" * 48
    r = compare.compare(unwritten, want)
    assert r["poisoned"] == r["words"] == 12 and r["first"] == 16 and r["hint"].startswith("not written")
    egemm = native_words(hashlib.sha512(b"x").digest())  # 63f78c35a: the native words, not the digest's bytes
    assert compare.compare(egemm, hashlib.sha512(b"x").digest())["hint"].startswith("byte order: the device's bytes "
                                                                                    "are the twin's with every 8-byte")
    swapped4 = b"".join(want[i:i + 4][::-1] for i in range(0, 64, 4))
    assert "4-byte" in compare.compare(swapped4, want)["hint"]
    assert compare.compare(want[:32], want)["hint"].startswith("size")
    assert compare.compare(want[:-1] + b"\0", want)["hint"] is None


def test_compare_dumps(tmp_path):
    digest = hashlib.sha512(b"leaf").digest()
    dump.write(tmp_path / "dev", "leaves", native_words(digest))
    dump.write(tmp_path / "dev", "out", b"\xa5" * 8)
    dump.write(tmp_path / "twin", "leaves", digest, "sha512-be")
    dump.write(tmp_path / "twin", "out", b"\0" * 8)
    dump.write(tmp_path / "twin", "extra", b"")
    v = compare.compare_dumps(tmp_path / "dev", tmp_path / "twin")
    assert not v["equal"] and v["missing"] == ["extra: not in the device dump"]
    assert v["buffers"]["leaves"]["hint"] == "layout: the device dumped bytes, the twin sha512-be"
    assert v["buffers"]["out"]["hint"].startswith("not written")
    assert compare.main([str(tmp_path / "dev"), str(tmp_path / "dev")]) == 0
    assert compare.main([str(tmp_path / "dev"), str(tmp_path / "twin")]) == compare.EXIT_DIFF


# ---- the gate ----------------------------------------------------------------------------------------------------------


def r7_smem(op: str) -> list[dict]:
    return [{"sample": s, "op": op, "lanes": [r7_off(slot, l, w) for l in range(32)]}
            for s, w in enumerate((0, 1, 31)) for slot in range(8)]


def test_round7_ldsm_is_refused_by_loops_and_banks():
    """Last night's LDSM probe: one address set every iteration, so ptxas unrolled the loop and kept 9 of its loads."""
    rec = G.judge_kernel(functions("round7_lm")[R7_LDSM], [{"LDSM.16.M88": 8}], smem=r7_smem("LDSM.16.M88"))
    assert not rec["ok"] and {"loops", "smem"} <= checks(rec)
    assert any(r.startswith("loops: 3 loops between the clock reads, 1 declared") for r in rec["reasons"])
    assert any("up to 8-way" in r for r in rec["reasons"])


def test_round7_stsm_is_refused_by_banks():
    """Last night's STSM probe: a 128-B lane stride puts each 8-row phase on 4 banks."""
    rec = G.judge_kernel(functions("round7_lm")[R7_STSM], [{"STSM.16.M88": 8}], smem=r7_smem("STSM.16.M88"))
    assert not rec["ok"] and "smem" in checks(rec)
    assert any(r.startswith("smem: STSM.16.M88:") and "up to 8-way" in r for r in rec["reasons"])


def test_ring_probe_passes_with_its_sink_and_banks():
    fn, p = functions("lm")["ldsm_x1"], table("lm")["ldsm_x1"]
    rec = G.judge_kernel(fn, [G.parse_expect(p["expect"])], smem=p["smem"])
    assert rec["ok"], rec["reasons"]
    assert rec["loops"][0]["counts"] == {"LDSM.16.M88": 8} and rec["sink"]["by"] == "store"
    assert {a["degree"] for a in rec["smem"]} == {1} and len(rec["smem"]) == 4 * 8
    assert rec["sass_sha256"] == G.sass_sha256(fn)


def test_the_in_tree_known_invalid_probes():
    fns, t = functions("lm"), table("lm")
    merged = G.judge_kernel(fns["bad_ldsm_merged"], [{"LDSM.16.M88": 8}], smem=t["bad_ldsm_merged"]["smem"])
    assert "loops" in checks(merged)
    conflict = G.judge_kernel(fns["bad_stsm_conflict"], [{"STSM.16.M88": 8}], smem=t["bad_stsm_conflict"]["smem"])
    assert checks(conflict) == {"smem"}


def test_count_mismatch_names_what_the_loop_holds():
    rec = G.judge_kernel(functions("lm")["ldsm_x1"], [{"LDSM.16.M88": 16}])
    (why,) = rec["reasons"]
    assert why.startswith("loops: LDSM.16.M88: 8 in the loop at") and "16 expected: ptxas merged" in why
    assert "the loop holds" in why and "8 LDSM.16.M88" in why


def test_side_effect_sink_and_two_loops():
    atom = functions("atom")
    red = G.judge_kernel(atom["redg_add32"], [{"REDG.E.ADD.STRONG.GPU": 8}])
    assert red["ok"] and red["sink"]["by"] == "side effect"
    assert G.judge_kernel(atom["atomg_cas64"], [{"ATOMG.E.CAS.64.STRONG.GPU": 4}])["ok"]
    spec = functions("issue")["spec_dadd_r2"]
    both = [{"IMMA.16832.S8.S8": 8}, {"DADD": 2}]
    assert G.judge_kernel(spec, both)["ok"]
    assert any(r.startswith("loops: 2 loops between the clock reads, 1 declared")
               for r in G.judge_kernel(spec, both[:1])["reasons"])


ELECT_LISTING = """\t.section\t.text.blkred,"ax",@progbits
        .global         blkred
        .type           blkred,@function
        .size           blkred,(.L_x_9 - blkred)
        .other          blkred,@"STO_CUDA_ENTRY STV_DEFAULT"
blkred:
.text.blkred:
        /*0000*/                   CS2UR UR10, SR_CLOCKLO ;
.L_x_1:
        /*0010*/                   PLOP3.LUT P0, PT, PT, PT, PT, 0x80, 0x8 ;
.L_x_2:
        /*0020*/               @P0 ELECT P2, URZ, PT ;
        /*0030*/                   UBLKRED.G.S.ADD [UR4], [UR7], UR8 ;
        /*0040*/               @P2 PLOP3.LUT P0, PT, P2, PT, PT, 0x8, 0x80 ;
        /*0050*/                   PLOP3.LUT P2, PT, PT, PT, PT, 0x8, 0x80 ;
        /*0060*/               @P0 BRA.U.ANY `(.L_x_2) ;
        /*0070*/                   UIADD3 UR6, UPT, UPT, UR6, 0x1, URZ ;
        /*0080*/                   UISETP.NE.AND UP0, UPT, UR6, URZ, UPT ;
        /*0090*/                   BRA.U UP0, `(.L_x_1) ;
        /*00a0*/                   CS2R R2, SR_CLOCKLO ;
        /*00b0*/                   EXIT ;
.L_x_9:
"""


TMEM_LISTING = """\t.section\t.text.tm,"ax",@progbits
        .global         tm
        .type           tm,@function
tm:
.text.tm:
        /*0000*/                   CS2R R2, SR_CLOCKLO ;
.L_x_1:
        /*0010*/                   LDTM.x128 R4, tmem[UR6] ;
        /*0020*/                   STG.E [R8], R4 ;
        /*0030*/                   UIADD3 UR6, UPT, UPT, UR6, 0x1, URZ ;
        /*0040*/                   UISETP.NE.AND UP0, UPT, UR6, URZ, UPT ;
        /*0050*/                   BRA.U UP0, `(.L_x_1) ;
        /*0060*/                   CS2R R3, SR_CLOCKLO ;
        /*0070*/                   EXIT ;
.L_x_9:
"""


def test_a_tmem_read_must_be_an_ldtm():
    """Every read out of tensor memory is an LDTM (tcgen05.ld), the one TMEM→register move; a tmem operand on any other
    opcode is a new, unproven path and is refused."""
    fn = {f["name"]: H.Function(f) for f in H.parse_nvdisasm(TMEM_LISTING)}["tm"]
    rec = G.judge_kernel(fn, [{"LDTM.x128": 1, "STG.E": 1}])
    assert rec["ok"], rec["reasons"]
    assert rec["tmem"] == ["0x0010 LDTM.x128"] and "tmem" not in checks(rec)
    sneaky = TMEM_LISTING.replace("LDTM.x128 R4, tmem[UR6]", "LD.E.128 R4, tmem[UR6]")
    fn2 = {f["name"]: H.Function(f) for f in H.parse_nvdisasm(sneaky)}["tm"]
    rec2 = G.judge_kernel(fn2, [{"LD.E.128": 1, "STG.E": 1}])
    assert "tmem" in checks(rec2) and not rec2["ok"]
    assert any("LD" in r and "LDTM" in r for r in rec2["reasons"])


def test_an_elect_loop_is_one_trip_and_listed():
    """ptxas wraps UBLKRED (sm_120) in an ELECT loop that runs once per active lane."""
    fn = {f["name"]: H.Function(f) for f in H.parse_nvdisasm(ELECT_LISTING)}["blkred"]
    rec = G.judge_kernel(fn, [{"UBLKRED*": 1}])
    assert rec["ok"], rec["reasons"]
    assert rec["elect"] == ["0x0020"] and rec["sink"]["by"] == "side effect"
    plain = ELECT_LISTING.replace("BRA.U.ANY", "BRA.U")
    fn = {f["name"]: H.Function(f) for f in H.parse_nvdisasm(plain)}["blkred"]
    assert any("is nested in the timed loop" in r for r in G.judge_kernel(fn, [{"UBLKRED*": 1}])["reasons"])


def test_ldgsts_is_a_side_effect():
    assert G.is_write(["LDGSTS", "E", "BYPASS", "128"])


def test_forbid_clock_and_sink_refusals():
    rec = G.judge_kernel(functions("lm")["ldsm_x1"], [{"LDSM.16.M88": 8}], forbid=("LDSM",))
    assert checks(rec) == {"forbid"}
    no_clock = functions("lm", lambda s: s.replace("SR_CLOCKLO", "SR_TID.X"))["ldsm_x1"]
    assert checks(G.judge_kernel(no_clock, [{"LDSM.16.M88": 8}])) == {"clock"}
    no_store = functions("lm", lambda s: s.replace("STG.E ", "NOP ; // "))["ldsm_x1"]
    assert checks(G.judge_kernel(no_store, [{"LDSM.16.M88": 8}])) == {"sink"}


def test_smem_offsets_must_match_the_loop():
    p = table("lm")["ldsm_x1"]
    short = [a for a in p["smem"] if a["sample"] != 0 or a is not p["smem"][0]]
    rec = G.judge_kernel(functions("lm")["ldsm_x1"], [{"LDSM.16.M88": 8}], smem=short)
    assert any(r.startswith("smem: sample 0 gives offsets for") for r in rec["reasons"])


def test_gate_selects_by_name_and_reports_missing():
    fns = list(functions("lm").values()) + list(functions("round7_lm").values())
    v = G.gate(Path("lm"), ["k_lm", "nope"], [{"LDSM.16.M88": 8}], fns=fns)
    assert {r["kernel"] for r in v["kernels"]} == {R7_LDSM, R7_STSM} and v["missing"] == ["nope"] and not v["ok"]
    assert G.refusals(v)[0] == "nope: no such kernel in lm"
    assert G.base_name(R7_LDSM) == "k_lm" and G.base_name("ldsm_x1") == "ldsm_x1"
    with pytest.raises(G.GateError, match="OP=N"):
        G.parse_expect("LDSM.16.M88")
