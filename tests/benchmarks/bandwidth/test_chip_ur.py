"""`chip_ur.py` without a card: the rewrite of a cubin's four uniform-register bytes, over a stand-in for cuobjdump
whose listing names the registers the cubin's bytes hold."""

import sys
from pathlib import Path

import chip_ur
import pytest

#: where the stand-in cubin holds each instruction's low word, and the word it starts with: v1's load into UR4 and its
#: read, v2's load into UR5 and its read (an LDCU's destination is byte 2, an ALU instruction's uniform source byte 4)
WORDS = {16: 0x0000_0000_0004_7AB9, 40: 0x0000_0004_0000_7C24, 64: 0x0000_0000_0005_7AB9, 88: 0x0000_0005_0000_7C12}
SIZE = 128


def cubin(words: dict[int, int] = WORDS) -> bytes:
    blob = bytearray(range(SIZE))
    for at, word in words.items():
        blob[at:at + 8] = word.to_bytes(8, "little")
    return bytes(blob)


def listing(_cuobjdump: str, path: str, extra: tuple[tuple[str, int], ...] = ()) -> list[tuple[str, int]]:
    """What cuobjdump says of a stand-in cubin: the four instructions, each naming the register its byte holds."""
    blob = Path(path).read_bytes()
    v1_load, v1_read, v2_load, v2_read = (int.from_bytes(blob[at:at + 8], "little") for at in WORDS)
    return [(f"LDCU UR{v1_load >> 16 & 0xFF}, c[0x0][0x388]", v1_load),
            (f"IMAD R5, R0, UR{v1_read >> 32 & 0xFF}, RZ", v1_read),
            (f"LDCU UR{v2_load >> 16 & 0xFF}, c[0x0][0x398]", v2_load),
            (f"LOP3.LUT R7, R0, UR{v2_read >> 32 & 0xFF}, RZ, 0x3c, !PT", v2_read),
            *extra]


@pytest.fixture
def rewrite(tmp_path, monkeypatch, capsys):
    """Run the command on a stand-in cubin; gives (what it printed, the cubin before, the cubin it wrote or None)."""

    def run(a, b, blob: bytes | None = None, sass=listing):
        blob = cubin() if blob is None else blob
        src, dst = tmp_path / "urt.cubin", tmp_path / "out.cubin"
        src.write_bytes(blob)
        monkeypatch.setattr(chip_ur, "sass", sass)
        monkeypatch.setattr(sys, "argv", ["chip_ur.py", "cuobjdump", str(src), str(a), str(b), str(dst)])
        chip_ur.main()
        return capsys.readouterr().out, blob, dst.read_bytes()

    return run


def test_the_rewrite_sets_the_four_register_bytes_and_no_other(rewrite):
    out, before, after = rewrite(126, 62)
    assert out == "urt: v1 in UR126, v2 in UR62\n"
    changed = {i: after[i] for i in range(SIZE) if after[i] != before[i]}
    assert changed == {16 + 2: 126, 40 + 4: 126, 64 + 2: 62, 88 + 4: 62} and len(after) == len(before)


@pytest.mark.parametrize("a, b", [(6, 6), (255, 6), (6, -1)])
def test_registers_that_are_not_two_of_0_to_254_are_refused(rewrite, a, b):
    with pytest.raises(SystemExit) as stop:
        rewrite(a, b)
    assert stop.value.code == "REFUSED: a and b are distinct uniform registers, 0..254"


@pytest.mark.parametrize("a, b", [(8, 6), (6, 9)])
def test_a_register_the_kernel_uses_for_something_else_or_the_one_after_it_is_refused(rewrite, a, b):
    def other(cuobjdump, path):
        return listing(cuobjdump, path, (("LDCU.64 UR8, c[0x0][0x358]", 0x0000_0000_0008_7AB9),))

    with pytest.raises(SystemExit) as stop:
        rewrite(a, b, sass=other)
    assert stop.value.code == f"REFUSED: UR{a} or UR{b} is one the kernel uses for something else ([8, 9])"


def test_a_cubin_with_two_loads_of_a_parameter_is_refused(rewrite):
    def twice(cuobjdump, path):
        return listing(cuobjdump, path, (("LDCU UR9, c[0x0][0x388]", 0x0000_0000_0009_7AB9),))

    with pytest.raises(SystemExit) as stop:
        rewrite(6, 7, sass=twice)
    assert stop.value.code == "REFUSED: 2 loads of c[0x0][0x388]"


def test_a_load_whose_byte_is_not_the_register_it_names_is_refused(rewrite):
    def misnamed(cuobjdump, path):
        (_, word), *rest = listing(cuobjdump, path)
        return [("LDCU UR3, c[0x0][0x388]", word), *rest]

    with pytest.raises(SystemExit) as stop:
        rewrite(6, 7, sass=misnamed)
    assert stop.value.code == "REFUSED: LDCU UR3, c[0x0][0x388]: byte 2 of 0x0000000000047ab9 is not UR3"


def test_a_register_read_twice_is_refused(rewrite):
    def twice(cuobjdump, path):
        return listing(cuobjdump, path, (("IADD3 R6, R6, UR4, RZ", 0x0000_0004_0000_7C10),))

    with pytest.raises(SystemExit) as stop:
        rewrite(6, 7, sass=twice)
    assert stop.value.code.startswith("REFUSED: UR4's reads are [('IMAD R5, R0, UR4, RZ', ")


def test_an_instruction_word_the_cubin_holds_twice_is_refused(rewrite):
    twice = cubin({**WORDS, 104: WORDS[16]})
    with pytest.raises(SystemExit) as stop:
        rewrite(6, 7, twice)
    assert stop.value.code == "REFUSED: 0x0000000000047ab9 is not unique in the cubin"


def test_a_rewrite_the_listing_does_not_show_is_refused(rewrite, tmp_path):
    def unmoved(cuobjdump, _path):
        return listing(cuobjdump, str(tmp_path / "urt.cubin"))

    with pytest.raises(SystemExit) as stop:
        rewrite(6, 7, sass=unmoved)
    assert stop.value.code == "REFUSED: the rewritten SASS names [4, 4, 5, 5], not UR6, UR7"


def test_the_functions_refuse_by_name_and_write_nothing(tmp_path, monkeypatch):
    src, dst = tmp_path / "urt.cubin", tmp_path / "out.cubin"
    src.write_bytes(cubin())
    monkeypatch.setattr(chip_ur, "sass", listing)
    with pytest.raises(chip_ur.Refused, match="a and b are distinct uniform registers, 0..254"):
        chip_ur.rewrite("cuobjdump", str(src), 6, 6, str(dst))
    with pytest.raises(chip_ur.Refused, match=r"2 loads of c\[0x0\]\[0x388\]"):
        chip_ur.plan([*listing("cuobjdump", str(src)), ("LDCU UR9, c[0x0][0x388]", 0x0000_0000_0009_7AB9)])
    assert not dst.exists()
