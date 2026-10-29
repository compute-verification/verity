"""A multi-output relation's values compare against the members its ports are committed as.  `DeepGemmFp8Linear_v1` returns (q, s, y) on
ports 0 / 1 / 2 and the manifest commits them as `x_q` / `x_s` / `0` (`part_rank<r>` on a row-parallel TP linear): the GLM-5.3 TP8 Commit
left every DeepGemm prefill stratum 'evaluator produced no member x_q' (411 per rank), and port 0's q stood where
member 0 (y) was.  The committed q is e4m3 (`float8_e4m3fn`, one byte a word), read like any other 8-bit plane.  A MoE plane row has no
port map: its ports are its plane members in return order (`Fp8GroupQuantUe8m0_v1` on the in_quant plane: 0 -> in_q, 1 -> in_scale)."""
from __future__ import annotations

import numpy as np
import pytest

from verity_vllm.check.replay import evaluate as EV, opening as OP
from verity.ml.definitions.numerics.deepgemm_rows import deepgemm_linear

K, N, ROW, ROWS = 256, 128, 1, 3
OP_PATH = "model.layers.0.mlp.down_proj"
PORTS = {"0": "x_q", "1": "x_s", "2": "0"}


def test_ports_are_renamed_to_their_members_and_port_zero_never_stands_in_for_member_zero():
    q, s, y = (np.full(2, v, dtype=np.uint16) for v in (1, 2, 3))
    got = EV.by_member({"0": q, "1": s, "2": y}, PORTS)
    assert got == {"x_q": q, "x_s": s, "0": y}
    tp = EV.by_member({"0": q, "1": s, "2": y}, {"0": "x_q", "1": "x_s", "2": "part_rank3"})
    assert tp == {"x_q": q, "x_s": s, "part_rank3": y}
    assert EV.by_member({"0": q, "1": s, "2": y}, {"0": "x_q", "2": "0"}) == {"x_q": q, "0": y, "1": s}


def test_identity_and_empty_port_maps_change_nothing():
    a, b = np.zeros(2, dtype=np.uint32), np.ones(2, dtype=np.uint32)
    got = {"router_w": a, "router_ids": b}
    assert EV.by_member(got, {"router_w": "router_w", "router_ids": "router_ids"}) == got
    assert EV.by_member(got, {}) == got


def _operands(rng):
    x = (np.asarray(rng.standard_normal(K), dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
    b = rng.integers(0, 256, N * K)
    w = np.where((b & 0x7F) == 0x7F, b & ~0x40, b).astype(np.uint8)
    ws = np.full((N // 128) * (K // 128), 2.0 ** -8, dtype=np.float32).view(np.uint32)
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


@pytest.fixture
def honest(monkeypatch):
    ops = _operands(np.random.default_rng(3))
    got = deepgemm_linear({"K": K, "N": N}, ops)
    monkeypatch.setattr(EV, "_evaluate_row", lambda *a, **k: deepgemm_linear({"K": K, "N": N}, ops))
    return got


MEMBERS = [("x_q", K, "w8"), ("x_s", K // 128, "f32"), ("0", N, "bf16")]


def _replay(members, ports, planes):
    return EV._replay_vu(_Index(members, ports), _store(planes), lambda *a: None, {"i": 7, "request_id": "r0", "phase": "prefill"}, 0, {})


def test_honest_deepgemm_row_replays_every_committed_member(honest):
    rec = _replay(MEMBERS, PORTS, _committed(honest))
    assert rec["result"] is True, rec.get("why")
    assert [o["member"] for o in rec["outputs"]] == ["x_q", "x_s", "0"] and all(o["equal"] for o in rec["outputs"])


def test_row_parallel_rank_compares_y_as_its_partial(honest):
    members = MEMBERS[:2] + [("part_rank0", N, "bf16")]
    rec = _replay(members, {"0": "x_q", "1": "x_s", "2": "part_rank0"}, _committed(honest, "part_rank0"))
    assert rec["result"] is True, rec.get("why")
    assert rec["outputs"][2]["member"] == "part_rank0"


@pytest.mark.parametrize("member,word", [("x_q", 5), ("x_s", 0), ("0", 9)])
def test_a_changed_committed_word_of_any_member_is_a_mismatch_by_name(honest, member, word):
    planes = _committed(honest)
    dtype, words = planes[member]
    words = words.copy()
    width = words.size // ROWS
    words[ROW * width + word] ^= 1
    planes[member] = (dtype, words)
    rec = _replay(MEMBERS, PORTS, planes)
    assert rec["result"] is False
    assert f"{OP_PATH}/{member} row {ROW}" in rec["why"]


def test_without_the_port_map_the_row_is_not_evaluated_by_name(honest):
    rec = _replay(MEMBERS, {}, _committed(honest))
    assert rec["result"] is None and rec["why"] == "evaluator produced no member 'x_q'"


def test_e4m3_plane_reads_one_byte_a_word():
    words = np.arange(12, dtype=np.uint8)
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


@pytest.fixture
def quant(monkeypatch):
    from verity.ml.definitions.numerics.glm53_rows import fp8_group_quant_ue8m0
    x = _operands(np.random.default_rng(5))[0]
    got = fp8_group_quant_ue8m0({"K": K, "G": 128}, [x])
    monkeypatch.setattr(EV, "_evaluate_row", lambda *a, **k: fp8_group_quant_ue8m0({"K": K, "G": 128}, [x]))
    return got


def _replay_plane(index, store):
    return EV._replay_vu(index, store, lambda *a: None, {"i": 7, "request_id": "r0", "phase": "prefill"}, 0, {})


def test_a_moe_quant_plane_row_compares_its_ports_as_its_plane_members(quant):
    rec = _replay_plane(_PlaneIndex(), _plane_store(quant["0"], quant["1"]))
    assert rec["result"] is True, rec.get("why")
    assert [o["member"] for o in rec["outputs"]] == ["in_q", "in_scale"] and all(o["equal"] for o in rec["outputs"])


@pytest.mark.parametrize("member", ["in_q", "in_scale"])
def test_a_changed_committed_plane_word_is_a_mismatch_by_name(quant, member):
    rec = _replay_plane(_PlaneIndex(), _plane_store(quant["0"], quant["1"], flip=member))
    assert rec["result"] is False
    assert f"{PLANE}/{member} row {ROW}" in rec["why"]


def test_a_row_that_is_no_plane_row_gets_no_plane_port_map(quant):
    rec = _replay_plane(_PlaneIndex(plane_kind=None), _plane_store(quant["0"], quant["1"]))
    assert rec["result"] is None and rec["why"] == "evaluator produced no member 'in_q'"
