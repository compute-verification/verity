"""`stage.Wires` on a committed capture: a staged instance's input is wired to the latest earlier call that output its bytes
(``block`` when a population gates that call, ``call`` when none does), or that output a tensor it is a strided view of, or
whose output is a strided view into its real rows, its padding rows uncharged; a parameter is ``registered`` with its
committed SHA-512; an `ADVICE` input is charged at its width; anything else, and an output of the call enclosing the reader,
is `Unwired` with the fix. `traced` records what an instance's arguments read, and `wiring` charges each advice argument
once."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
BF16 = "torch.bfloat16"


def _stage(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "setrecursionlimit", lambda n: None)
    spec = importlib.util.spec_from_file_location("spot_stage_wires", HERE / "stage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(b: bytes) -> str:
    return hashlib.sha512(b).hexdigest()


class _Blobs:
    def __init__(self, d: Path) -> None:
        self.d = d

    def ref(self, a: np.ndarray, stride: list[int] | None = None) -> dict:
        raw = np.ascontiguousarray(a).tobytes()
        h = _sha(raw)
        p = self.d / "rank0" / "blobs" / h[:2] / h
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
        return {"shape": list(a.shape), "dtype": BF16, "stride": stride or [int(np.prod(a.shape[i + 1:])) for i in range(a.ndim)],
                "sha512": h}


def _commit(d: Path, events: list[dict], params: dict) -> None:
    sd = d / "rank0" / "step0"
    sd.mkdir(parents=True, exist_ok=True)
    ops = "".join(json.dumps(e, sort_keys=True) + "\n" for e in events).encode()
    pj = json.dumps(params).encode()
    (sd / "ops.jsonl").write_bytes(ops)
    (sd / "params.json").write_bytes(pj)
    root = _sha(f"verity/spot/rank-root\0{_sha(ops)}\0{_sha(pj)}".encode())
    (sd / "meta.json").write_text(json.dumps({"scheduled": {"req": 4}, "ops_sha512": _sha(ops), "params_sha512": _sha(pj),
                                              "rank_root": root, "check": {}}))
    (sd / "commit.json").write_text(json.dumps({"root": _sha(f"verity/spot/step-root\0{root}".encode()), "coins": "00" * 64,
                                                "rank_roots": [root], "layers": {"dense": [0]}, "opened": {}}))


def _ev(seq: int, op: str, ins: dict, outs: dict, parent: int | None = None) -> dict:
    return {"seq": seq, "op": op, "parent": parent, "layer": 0, "in": ins, "out": outs}


@pytest.fixture()
def step(tmp_path, monkeypatch):
    stage = _stage(monkeypatch)
    rng = np.random.default_rng(3)
    b = _Blobs(tmp_path)
    x, y, z, q, kv, p = (rng.integers(0, 2 ** 16, s, dtype=np.uint16) for s in ((4, 8), (4, 8), (4, 8), (4, 8), (2, 6, 4), (4, 8)))
    pad, short = (rng.integers(0, 2 ** 16, (6, 8), dtype=np.uint16) for _ in range(2))
    w = {"shape": [8, 8], "dtype": BF16, "sha512": "ab" * 64}
    heads = [4, 8, 1]  # (head, token, column) of a [tokens, 2 heads x 4] buffer, as MLA's absorb writes o_proj's input
    events = [
        _ev(0, "aten::mm", {"self": b.ref(x), "mat2": {"param": "w", "shape": [8, 8], "dtype": BF16}}, {"ret0": b.ref(y)}),
        _ev(1, "aten::gelu", {"self": b.ref(y)}, {"ret0": b.ref(z)}),
        _ev(2, "aten::add", {"self": b.ref(z), "other": b.ref(q)}, {"ret0": b.ref(z + 1)}),
        _ev(3, "aten::bmm", {"self": b.ref(y[:, 2:6], stride=[8, 1])}, {"ret0": b.ref(y[:, :1])}),
        _ev(4, "fa3_sparse_mla", {"kv_gathered": b.ref(kv)}, {}),
        _ev(5, "vllm::outer", {}, {"ret0": b.ref(p)}),
        _ev(6, "aten::mm", {"self": b.ref(p)}, {"ret0": b.ref(p + 2)}, parent=5),
        _ev(7, "aten::bmm", {}, {"out": b.ref(pad[:4].reshape(4, 2, 4).transpose(1, 0, 2), stride=heads)}),
        _ev(8, "vllm::linear", {"input": b.ref(pad)}, {}),
        _ev(9, "_C::per_token_group_fp8_quant", {"input": b.ref(pad)}, {}, parent=8),
        _ev(10, "aten::bmm", {}, {"out": b.ref(short[:3].reshape(3, 2, 4).transpose(1, 0, 2), stride=heads)}),
        _ev(11, "aten::gelu", {"self": b.ref(short)}, {}),
    ]
    _commit(tmp_path, events, {"w": w})
    cap = stage.Capture(tmp_path, 0)
    return stage, cap, stage.Wires(cap)


def test_an_input_is_wired_to_the_call_that_output_its_bytes_or_a_tensor_it_views(step) -> None:
    stage, cap, wires = step
    ev = cap.events[0]
    assert wires.source(0, ev[1]["in"]["self"]) == {"kind": "block", "rank": 0, "seq": 0, "op": "aten::mm", "out": "ret0",
                                                    "into": {"seq": 1, "op": "aten::gelu", "in": "self"}}
    assert wires.source(0, ev[2]["in"]["self"])["kind"] == "call", "aten::gelu is no population's: committed, not proved"
    view = wires.source(0, ev[3]["in"]["self"])
    assert (view["kind"], view["seq"], view["view"]) == ("block", 0, {"offset": 2, "stride": [8, 1]})
    assert wires.source(0, ev[3]["in"]["self"]) is view, "a reference's source is found once"
    assert wires.source(0, ev[0]["out"]["ret0"]) == {"kind": "block", "rank": 0, "seq": 0, "op": "aten::mm", "out": "ret0"}


def test_a_buffer_a_call_wrote_through_a_view_is_wired_to_it_and_its_padding_rows_cost_nothing(step) -> None:
    stage, cap, wires = step
    ev = cap.events[0]
    src = wires.source(0, ev[8]["in"]["input"])
    assert src == {"kind": "block", "rank": 0, "seq": 7, "op": "aten::bmm", "out": "out",
                   "into": {"seq": 8, "op": "vllm::linear", "in": "input"}, "cover": {"stride": [4, 8, 1], "rows": 4}}
    quant = wires.source(0, ev[9]["in"]["input"])
    assert (quant["seq"], quant["into"]["seq"], quant["cover"]) == (7, 9, src["cover"]), "the enclosing call is skipped"
    w = stage.wiring({"A": {"gate": [{"key": [0, 8, 0], "sources": [src]}, {"key": [0, 9, 0], "sources": [quant]}]}})
    assert w["advice_bits"] == 0, "the padding rows past the 4 real rows are in no population"
    with pytest.raises(stage.Unwired, match="reads self"):
        wires.source(0, ev[11]["in"]["self"])


def test_a_parameter_is_registered_and_an_advice_input_charged_at_its_width(step) -> None:
    stage, cap, wires = step
    ev = cap.events[0]
    assert wires.source(0, ev[0]["in"]["mat2"]) == {"kind": "registered", "rank": 0, "param": "w", "sha512": "ab" * 64}
    adv = wires.source(0, ev[4]["in"]["kv_gathered"])
    assert (adv["kind"], adv["bits"]) == ("advice", 2 * 6 * 4 * 16) and adv["why"] == stage.ADVICE[("fa3_sparse_mla", "kv_gathered")]


def test_an_input_no_earlier_call_output_is_unwired_with_the_fix_and_the_enclosing_calls_output_too(step) -> None:
    stage, cap, wires = step
    ev = cap.events[0]
    for _ in range(2):
        with pytest.raises(stage.Unwired, match=r"reads other, .*declare \('aten::add', 'other'\) in stage\.ADVICE"):
            wires.source(0, ev[2]["in"]["other"])
    with pytest.raises(stage.Unwired, match="reads self"):
        wires.source(0, ev[6]["in"]["self"])


def test_traced_records_the_reads_of_an_instances_arguments_and_wiring_charges_advice_once(step) -> None:
    stage, cap, wires = step
    ev = cap.events[0]
    reads: list = []
    args = stage.traced(cap, lambda: [cap.tensor(0, ev[1]["in"]["self"])[0], cap.tensor(0, ev[1]["in"]["self"])[1],
                                      cap.tensor(0, ev[4]["in"]["kv_gathered"])[0]], reads)()
    assert len(args) == 3 and len(reads) == 3 and cap.reads is None
    sources = wires.of(reads)
    assert [s["kind"] for s in sources] == ["block", "advice"]
    types = {"A": {"gate": [{"key": [0, 1, 0], "sources": sources}, {"key": [0, 1, 1], "sources": sources},
                            {"key": [0, 2, 0], "unwired": "rank 0 ..."}]}}
    w = stage.wiring(types)
    assert w["advice_bits"] == 2 * 6 * 4 * 16 and len(w["inputs"]["A"]) == 2 and w["unwired"] == {"A": ["rank 0 ..."]}
