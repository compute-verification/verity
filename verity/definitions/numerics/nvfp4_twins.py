"""The device layouts a Commit opens for the NVFP4 linear (`registry.nvfp4`): the codes are packed two per byte, element 2i in
the low nibble; the scale bytes are the swizzled [ceil(M/128)*128, round_up(K/16, 4)] layout of
`cvt_quant_to_fp4_get_sf_out_offset`."""
from __future__ import annotations


def sf_offset(row: int, kblock: int, K: int) -> int:
    """Byte offset of the scale of (row, 16-block) in the swizzled layout: [M tiles of 128][K tiles of 4 blocks][32][4][4]."""
    k_tiles = (K + 63) // 64
    return (((row >> 7) * k_tiles + (kblock >> 2)) << 9) | ((row & 31) << 4) | (((row >> 5) & 3) << 2) | (kblock & 3)


def unpack_codes(packed: bytes | list[int]) -> list[int]:
    """The E2M1 codes of a packed row: element 2i is the low nibble of byte i (`cvt.rn.satfinite.e2m1x2.f32 d, x[2i+1], x[2i]`)."""
    return [c for b in packed for c in (b & 0xF, b >> 4)]
