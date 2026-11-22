"""A multi-output relation's values compare against the members its ports are committed as.  `DeepGemmFp8Linear_v1` returns (q, s, y) on
ports 0 / 1 / 2 and the manifest commits them as `x_q` / `x_s` / `0` (`part_rank<r>` on a row-parallel TP linear): the GLM-5.3 TP8 Commit
left every DeepGemm prefill stratum 'evaluator produced no member x_q' (411 per rank), and port 0's q stood where
member 0 (y) was.  The committed q is e4m3 (`float8_e4m3fn`, one byte a word), read like any other 8-bit plane.  A MoE plane row has no
port map: its ports are its plane members in return order (`Fp8GroupQuantUe8m0_v1` on the in_quant plane: 0 -> in_q, 1 -> in_scale)."""
from __future__ import annotations

import numpy as np

from verity_vllm.check.replay import opening as OP

K, N, ROW, ROWS = 256, 128, 1, 3
OP_PATH = "model.layers.0.mlp.down_proj"
PORTS = {"0": "x_q", "1": "x_s", "2": "0"}


def _operands(rng):
    x = (np.asarray(rng.standard_normal(K), dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
    b = rng.integers(0, 256, N * K)
    w = np.where((b & 0x7F) == 0x7F, b & ~0x40, b).astype(np.uint8)
    ws: np.ndarray = np.full((N // 128) * (K // 128), 2.0 ** -8, dtype=np.float32).view(np.uint32)
    return [x, w, ws]


class _Index:
    def __init__(self, members, ports):
        self.by_i = {7: {"i": 7, "spec": f"DeepGemmFp8Linear_v1{{K={K},N={N}}}"}}
        self._d = {"family": "DeepGemmFp8Linear_v1", "statics": {"K": K, "N": N}, "step": 0, "op_path": OP_PATH, "row": ROW,
                   "members": members}
        self._ports = ports

    def describe(self, r):
        return self._d

    def member_of_port(self, i):
        return dict(self._ports)


def _store(planes):
    """A binding map with one entry per member (ROWS rows) and the reader over the committed bytes."""
    entries, blobs = [], {}
    for member, (dtype, words) in planes.items():
        name = f"{OP_PATH}/{member}"
        blobs[name] = np.ascontiguousarray(words).tobytes()
        entries.append({"name": name, "element_offset": 0, "encoding": {"dtype": dtype},
                        "identity": {"request_id": "r0", "step": 0, "op_path": OP_PATH, "output_member": member,
                                     "element_range": [0, words.size]}})
    return OP.CommittedStore({"entries": entries}, lambda step, name, b0, b1: blobs[name][b0:b1])


def _committed(got, y_member="0"):
    """ROWS rows of committed words, row ROW the honest one."""
    def rows(w, dt):
        out = np.zeros((ROWS, w.size), dtype=dt)
        out[ROW] = w
        return out.reshape(-1)
    return {"x_q": ("float8_e4m3fn", rows(got["0"], np.uint8)), "x_s": ("float32", rows(got["1"], np.uint32)),
            y_member: ("bfloat16", rows(got["2"], np.uint16))}


MEMBERS = [("x_q", K, "w8"), ("x_s", K // 128, "f32"), ("0", N, "bf16")]


def test_e4m3_plane_reads_one_byte_a_word():
    words: np.ndarray = np.arange(12, dtype=np.uint8)
    got, dtype = _store({"x_q": ("float8_e4m3fn", words)}).words("r0", 0, OP_PATH, "x_q", 4, 8)
    assert dtype == "float8_e4m3fn" and got.dtype == np.uint8 and got.tolist() == [4, 5, 6, 7]


PLANE = "moe/L3"
PLANE_MEMBERS = [("in_q", K, "w8"), ("in_scale", K // 128, "f32")]


class _PlaneIndex(_Index):
    """A MoE in_quant plane row as `ProgramIndex._describe_plane` describes it: no v2 port map, the plane members in return order."""

    def __init__(self, plane_kind="in_quant"):
        super().__init__(PLANE_MEMBERS, {})
        self.by_i = {7: {"i": 7, "spec": f"Fp8GroupQuantUe8m0_v1{{K={K},G=128}}"}}
        self._d = {"family": "Fp8GroupQuantUe8m0_v1", "statics": {"K": K, "G": 128}, "step": 0, "op_path": PLANE, "row": ROW,
                   "members": PLANE_MEMBERS, "plane": "in_q"}
        if plane_kind:
            self._d["plane_kind"] = plane_kind


def _plane_store(q, s, flip=None):
    planes = {"in_q": ("float8_e4m3fn", q, np.uint8), "in_scale": ("float32", s, np.uint32)}
    entries, blobs = [], {}
    for member, (dtype, w, dt) in planes.items():
        rows = np.zeros((ROWS, w.size), dtype=dt)
        rows[ROW] = w
        if member == flip:
            rows[ROW, 1] ^= 1
        blobs[member] = rows.tobytes()
        entries.append({"name": f"{PLANE}/{member}", "element_offset": 0, "encoding": {"dtype": dtype},
                        "identity": {"request_id": "r0", "step": 0, "op_path": PLANE, "output_member": member,
                                     "element_range": [0, rows.size]}})
    return OP.CommittedStore({"entries": entries}, lambda step, name, b0, b1: blobs[name.rsplit("/", 1)[1]][b0:b1])


