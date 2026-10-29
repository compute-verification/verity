"""Set the last-use hint (LDL -> LDL.LU) on every local load of the named kernels of a CUDA binary, in place.

PTX can't ask for LDL.LU: ptxas drops `.lu` on ld.local and emits LDL.LU only for its own spill reloads. On sm_90 the hint
is the load's cache-op field, bits 84-86 of the 128-bit instruction: 1 for LDL, 3 for LDL.LU, every other bit the same
(as ptxas encodes its spill reloads beside its ld.local loads). This finds each CUDA ELF the binary embeds (uncompressed,
nvcc's default), each named kernel's `.text` section, and each LDL in it (opcode 0x983 in bits 0-11), and sets the field
to 3. The SASS gate then reads the patched binary as any other.

    python3 ldl_lu.py BINARY KERNEL...

Exit: 0 every named kernel's every LDL was LDL and is now LDL.LU; 1 a kernel has none, or one holds another cache op.
"""

import struct
import sys
from pathlib import Path

EM_CUDA = 190
LDL = 0x983
CACHE_OP_SHIFT, CACHE_OP_MASK = 84 - 64, 0x7
DEFAULT, LAST_USE = 1, 3


def cuda_elfs(b: bytes):
    i = -1
    while (i := b.find(b"\x7fELF", i + 1)) >= 0:
        if b[i + 4] == 2 and struct.unpack_from("<H", b, i + 18)[0] == EM_CUDA:
            yield i


def text_sections(b: bytes, base: int):
    """Each `.text.<kernel>` section of the ELF at `base`: kernel, file offset, size."""
    shoff = struct.unpack_from("<Q", b, base + 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", b, base + 0x3A)
    secs = [struct.unpack_from("<IIQQQQIIQQ", b, base + shoff + k * shentsize) for k in range(shnum)]
    strtab = base + secs[shstrndx][4]
    for name_off, _, _, _, off, size, *_ in secs:
        name = b[strtab + name_off:b.index(b"\0", strtab + name_off)].decode()
        if name.startswith(".text."):
            yield name.removeprefix(".text."), base + off, size


def patch(b: bytearray, kernels: list[str]) -> dict[str, list[str]]:
    done: dict[str, list[str]] = {k: [] for k in kernels}
    for base in list(cuda_elfs(b)):
        for kernel, off, size in text_sections(b, base):
            if kernel not in done:
                continue
            for at in range(off, off + size, 16):
                lo, hi = struct.unpack_from("<QQ", b, at)
                if lo & 0xFFF != LDL:
                    continue
                op = (hi >> CACHE_OP_SHIFT) & CACHE_OP_MASK
                if op != DEFAULT:
                    raise SystemExit(f"ldl_lu: {kernel}+{at - off:#x}: an LDL with cache op {op}, not {DEFAULT}")
                new = hi & ~(CACHE_OP_MASK << CACHE_OP_SHIFT) | LAST_USE << CACHE_OP_SHIFT
                struct.pack_into("<Q", b, at + 8, new)
                done[kernel].append(f"{at - off:#06x} {hi:#018x} -> {new:#018x}")
    return done


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__.split("\n\n")[2].strip(), file=sys.stderr)
        return 1
    path = Path(argv[0])
    b = bytearray(path.read_bytes())
    done = patch(b, argv[1:])
    rc = 0
    for kernel, edits in done.items():
        print(f"{kernel}: {len(edits)} LDL -> LDL.LU" + "".join(f"\n  {e}" for e in edits))
        if not edits:
            print(f"ldl_lu: {kernel}: no LDL in any CUDA ELF of {path}", file=sys.stderr)
            rc = 1
    if rc == 0:
        path.write_bytes(bytes(b))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
