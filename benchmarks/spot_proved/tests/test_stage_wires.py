"""`stage.Wires` on a committed capture: a staged instance's input is wired to the latest earlier call that output its bytes
(``block`` when a population reads that call, ``call`` when none does; a call that only passed its input's bytes on is passed
over), or that output a tensor it is a strided view of, or to the calls whose outputs are strided views into its real rows,
one or several, its padding rows uncharged; a parameter is ``registered`` with its
committed SHA-512; a `GAPS` input (a computed value the capture doesn't record) is counted at the union of the parts
instances read (`Capture.part`), else at its width, as gap bits, never as advice, and an `ADVICE` input (a free choice)
is charged so; anything else, and an output of the call enclosing the reader, is `Unwired` with the fix. `traced` records
what an instance's arguments read, and `wiring` counts each such argument once."""
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
DTYPES = {np.dtype(np.uint16): BF16, np.dtype(np.int32): "torch.int32", np.dtype(np.int64): "torch.int64"}


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
        return {"shape": list(a.shape), "dtype": DTYPES[a.dtype],
                "stride": stride or [int(np.prod(a.shape[i + 1:])) for i in range(a.ndim)], "sha512": h}


def _batch(req_ids: list, scheduled: dict) -> str:
    return _sha(b"verity/spot/batch\0" + json.dumps({"req_ids": req_ids, "scheduled": scheduled}, sort_keys=True,
                                                     separators=(",", ":")).encode())


def _commit(d: Path, events: list[dict], params: dict, scheduled: dict | None = None, req_ids: list | None = None,
            tamper: dict | None = None) -> None:
    """The step committed as `capture.py` commits it: with ``req_ids``, the batch under the rank root too; ``tamper`` then
    rewrites ``meta.json``'s fields after the commit."""
    sd = d / "rank0" / "step0"
    sd.mkdir(parents=True, exist_ok=True)
    ops = "".join(json.dumps(e, sort_keys=True) + "\n" for e in events).encode()
    pj = json.dumps(params).encode()
    (sd / "ops.jsonl").write_bytes(ops)
    (sd / "params.json").write_bytes(pj)
    scheduled = scheduled or {"req": 4}
    batch = {"req_ids": req_ids, "batch_sha512": _batch(req_ids, scheduled)} if req_ids is not None else {}
    root = _sha("\0".join(["verity/spot/rank-root", _sha(ops), _sha(pj)] + ([batch["batch_sha512"]] if batch else [])).encode())
    (sd / "meta.json").write_text(json.dumps({"scheduled": scheduled, "ops_sha512": _sha(ops), "params_sha512": _sha(pj),
                                              "rank_root": root, "check": {}, **batch, **(tamper or {})}))
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
    rec = stage.Capture(tmp_path, 0)
    return stage, rec, stage.Wires(rec)


def test_an_input_is_wired_to_the_call_that_output_its_bytes_or_a_tensor_it_views(step) -> None:
    stage, rec, wires = step
    ev = rec.events[0]
    assert wires.source(0, ev[1]["in"]["self"]) == {"kind": "block", "rank": 0, "seq": 0, "op": "aten::mm", "out": "ret0",
                                                    "into": {"seq": 1, "op": "aten::gelu", "in": "self"}}
    assert wires.source(0, ev[2]["in"]["self"])["kind"] == "call", "aten::gelu is no population's: committed, not proved"
    view = wires.source(0, ev[3]["in"]["self"])
    assert (view["kind"], view["seq"], view["view"]) == ("block", 0, {"offset": 2, "stride": [8, 1]})
    assert wires.source(0, ev[3]["in"]["self"]) is view, "a reference's source is found once"
    assert wires.source(0, ev[0]["out"]["ret0"]) == {"kind": "block", "rank": 0, "seq": 0, "op": "aten::mm", "out": "ret0"}


def test_a_buffer_a_call_wrote_through_a_view_is_wired_to_it_and_its_padding_rows_cost_nothing(step) -> None:
    stage, rec, wires = step
    ev = rec.events[0]
    src = wires.source(0, ev[8]["in"]["input"])
    assert src == {"kind": "block", "rank": 0, "seq": 7, "op": "aten::bmm", "out": "out",
                   "into": {"seq": 8, "op": "vllm::linear", "in": "input"}, "cover": {"stride": [4, 8, 1], "offset": 0, "rows": 4}}
    quant = wires.source(0, ev[9]["in"]["input"])
    assert (quant["seq"], quant["into"]["seq"], quant["cover"]) == (7, 9, src["cover"]), "the enclosing call is skipped"
    w = stage.wiring({"A": {"gate": [{"key": [0, 8, 0], "sources": [src]}, {"key": [0, 9, 0], "sources": [quant]}]}})
    assert w["advice_bits"] == 0, "the padding rows past the 4 real rows are in no population"
    with pytest.raises(stage.Unwired, match="reads self"):
        wires.source(0, ev[11]["in"]["self"])


@pytest.fixture()
def mixed(tmp_path, monkeypatch):
    """A mixed step's attention output: 2 decode rows written by the absorb's ``out=``, 2 prefill rows copied in from dense
    FA3's output, 2 padding rows nothing writes; a sparse FA3 output an identity ``aten::to`` passes on; and a value only an
    identity call outputs."""
    stage = _stage(monkeypatch)
    rng = np.random.default_rng(5)
    b = _Blobs(tmp_path)
    o, sp, u, q = (rng.integers(0, 2 ** 16, s, dtype=np.uint16) for s in ((6, 8), (2, 8), (2, 8), (2, 8)))
    dense = o[2:4].copy()
    events = [
        _ev(0, "_vllm_fa3_C::fwd", {"q": b.ref(q)}, {"ret0": b.ref(dense)}),
        _ev(1, "aten::copy_", {"self": b.ref(q + 1), "src": b.ref(dense)}, {"self": b.ref(dense)}),
        _ev(2, "aten::bmm", {}, {"out": b.ref(o[:2].reshape(2, 2, 4).transpose(1, 0, 2), stride=[4, 8, 1])}),
        _ev(3, "vllm::linear", {"input": b.ref(o)}, {}),
        _ev(4, "_vllm_fa3_C::fwd", {"q": b.ref(q + 2), "q_v": b.ref(q + 3)}, {"ret0": b.ref(sp)}),
        _ev(5, "aten::to", {"self": b.ref(sp)}, {"ret0": b.ref(sp)}),
        _ev(6, "aten::gelu", {"self": b.ref(sp)}, {}),
        _ev(7, "aten::gelu", {"self": b.ref(dense)}, {}),
        _ev(8, "aten::to", {"self": b.ref(u)}, {"ret0": b.ref(u)}),
        _ev(9, "aten::gelu", {"self": b.ref(u)}, {}),
    ]
    _commit(tmp_path, events, {})
    rec = stage.Capture(tmp_path, 0)
    return stage, rec, stage.Wires(rec)


def test_a_buffer_several_calls_wrote_in_parts_is_wired_to_each_call_for_its_rows(mixed) -> None:
    stage, rec, wires = mixed
    ev = rec.events[0]
    into = {"seq": 3, "op": "vllm::linear", "in": "input"}
    assert wires.source(0, ev[3]["in"]["input"]) == [
        {"kind": "block", "rank": 0, "seq": 2, "op": "aten::bmm", "out": "out", "into": into,
         "cover": {"stride": [4, 8, 1], "offset": 0, "rows": 4}},
        {"kind": "call", "rank": 0, "seq": 1, "op": "aten::copy_", "out": "self", "into": into,
         "cover": {"stride": [8, 1], "offset": 16, "rows": 4}}]
    sources = wires.of([(0, ev[3]["in"]["input"])])
    assert [s["seq"] for s in sources] == [2, 1], "`of` lists a source per call"
    w = stage.wiring({"A": {"gate": [{"key": [0, 3, 0], "sources": sources}]}})
    assert (w["unproved_calls"], w["advice_bits"], w["gap_bits"]) == ({"aten::copy_": 1}, 0, 0)


@pytest.fixture()
def full_graph(tmp_path, monkeypatch):
    """A full-graph decode step's MLA layer as GLM-5.3 TP8 records it: 5 tokens padded to 8, attention run at the padded
    batch, so the absorb's ``out=`` (heads x tokens x 256, a transposed view of o_proj's ``[8, 2048]`` input) writes all 8
    rows, padding rows included, and the FP8 o_proj reads the buffer whole."""
    stage = _stage(monkeypatch)
    rng = np.random.default_rng(7)
    b = _Blobs(tmp_path)
    attn, o, stale, y, hidden = (rng.integers(0, 2 ** 16, s, dtype=np.uint16)
                                 for s in ((8, 8, 512), (8, 2048), (8, 2048), (8, 6144), (8, 6144)))
    heads = [256, 2048, 1]
    out = b.ref(o.reshape(8, 8, 256).transpose(1, 0, 2), stride=heads)
    events = [
        _ev(0, "fa3_sparse_mla", {}, {"r0": b.ref(attn)}),
        _ev(1, "aten::bmm", {"self": b.ref(attn.transpose(1, 0, 2), stride=[512, 4096, 1]),
                             "out": b.ref(stale.reshape(8, 8, 256).transpose(1, 0, 2), stride=heads)},
            {"out": out, "ret0": dict(out)}),
        _ev(2, "aten::new_empty", {"self": b.ref(o)}, {"ret0": b.ref(o[:1, :1].reshape(1))}),
        _ev(3, "vllm::dynamic_flashinfer_deepgemm_blockscale_gemm",
            {"input": b.ref(o), "weight": {"param": "w", "shape": [6144, 2048], "dtype": "torch.float8_e4m3fn"}},
            {"ret0": b.ref(y)}),
        _ev(4, "model_output", {}, {"hidden": b.ref(hidden)}),
    ]
    _commit(tmp_path, events, {"w": {"shape": [6144, 2048], "dtype": "torch.float8_e4m3fn", "sha512": "cd" * 64}},
            {f"req{i}": 1 for i in range(5)})
    rec = stage.Capture(tmp_path, 0)
    return stage, rec, stage.Wires(rec)


def test_a_buffer_a_full_graph_step_wrote_with_its_padding_rows_is_wired_to_the_writer_for_its_real_rows(full_graph) -> None:
    stage, rec, wires = full_graph
    ev = rec.events[0]
    assert (rec.padded(0), rec.real_rows(0, 8)) == (8, 5)
    src = wires.source(0, ev[3]["in"]["input"])
    assert src == {"kind": "block", "rank": 0, "seq": 1, "op": "aten::bmm", "out": "ret0",
                   "into": {"seq": 3, "op": "vllm::dynamic_flashinfer_deepgemm_blockscale_gemm", "in": "input"},
                   "cover": {"stride": [256, 2048, 1], "offset": 0, "rows": 5}}
    w = stage.wiring({"A": {"gate": [{"key": [0, 3, 0], "sources": [src]}]}})
    assert (w["unproved_calls"], w["advice_bits"], w["gap_bits"]) == ({}, 0, 0)


def test_a_call_that_passes_its_inputs_bytes_on_is_passed_over_and_dense_fa3_is_no_populations(mixed) -> None:
    stage, rec, wires = mixed
    ev = rec.events[0]
    sparse = wires.source(0, ev[6]["in"]["self"])
    assert (sparse["kind"], sparse["seq"], sparse["op"]) == ("block", 4, "_vllm_fa3_C::fwd"), "aten::to only passed it on"
    dense = wires.source(0, ev[7]["in"]["self"])
    assert (dense["kind"], dense["seq"]) == ("call", 0), "copy_ passed FA3's bytes on; dense FA3 is committed, not proved"
    only = wires.source(0, ev[9]["in"]["self"])
    assert (only["kind"], only["seq"], only["op"]) == ("call", 8, "aten::to"), "no earlier call output them"
    gated, other, _ = stage.op_coverage(rec)
    assert (gated.get("_vllm_fa3_C::fwd"), other.get("_vllm_fa3_C::fwd")) == (1, 1)


def test_a_parameter_is_registered_and_a_gap_input_counted_at_its_width(step) -> None:
    stage, rec, wires = step
    ev = rec.events[0]
    assert wires.source(0, ev[0]["in"]["mat2"]) == {"kind": "registered", "rank": 0, "param": "w", "sha512": "ab" * 64}
    gap = wires.source(0, ev[4]["in"]["kv_gathered"])
    assert (gap["kind"], gap["bits"]) == ("gap", 2 * 6 * 4 * 16) and gap["why"] == stage.GAPS[("fa3_sparse_mla", "kv_gathered")]
    assert not stage.ADVICE, "no staged population reads a free choice today"


def test_a_free_choice_is_charged_as_advice_and_a_gap_is_counted_apart_never_charged(step, monkeypatch) -> None:
    stage, rec, wires = step
    ev = rec.events[0]
    monkeypatch.setitem(stage.ADVICE, ("aten::add", "other"), "a choice nothing computes")
    adv, gap = wires.source(0, ev[2]["in"]["other"]), wires.source(0, ev[4]["in"]["kv_gathered"])
    assert (adv["kind"], adv["bits"], adv["why"]) == ("advice", 4 * 8 * 16, "a choice nothing computes")
    w = stage.wiring({"A": {"gate": [{"key": [0, 2, 0], "sources": [adv, gap]}]}})
    assert (w["advice_bits"], w["gap_bits"]) == (4 * 8 * 16, 2 * 6 * 4 * 16)


def test_an_input_no_earlier_call_output_is_unwired_with_the_fix_and_the_enclosing_calls_output_too(step) -> None:
    stage, rec, wires = step
    ev = rec.events[0]
    for _ in range(2):
        with pytest.raises(stage.Unwired, match=r"reads other, .*declare \('aten::add', 'other'\) in stage\.GAPS if the step "
                                                r"computes it .* or in stage\.ADVICE if nothing does"):
            wires.source(0, ev[2]["in"]["other"])
    with pytest.raises(stage.Unwired, match="reads self"):
        wires.source(0, ev[6]["in"]["self"])


def test_traced_records_the_reads_of_an_instances_arguments_and_wiring_counts_a_gap_once(step) -> None:
    stage, rec, wires = step
    ev = rec.events[0]
    reads: list = []
    args = stage.traced(rec, lambda: [rec.tensor(0, ev[1]["in"]["self"])[0], rec.tensor(0, ev[1]["in"]["self"])[1],
                                      rec.tensor(0, ev[4]["in"]["kv_gathered"])[0]], reads)()
    assert len(args) == 3 and len(reads) == 3 and rec.reads is None
    sources = wires.of(reads)
    assert [s["kind"] for s in sources] == ["block", "gap"]
    types = {"A": {"gate": [{"key": [0, 1, 0], "sources": sources}, {"key": [0, 1, 1], "sources": sources},
                            {"key": [0, 2, 0], "unwired": "rank 0 ..."}]}}
    w = stage.wiring(types)
    assert (w["gap_bits"], w["advice_bits"]) == (2 * 6 * 4 * 16, 0)
    assert len(w["inputs"]["A"]) == 2 and w["unwired"] == {"A": ["rank 0 ..."]}


def test_a_gap_input_is_counted_at_the_union_of_the_parts_staged_instances_read(step) -> None:
    stage, rec, wires = step
    kv = rec.events[0][4]["in"]["kv_gathered"]
    row = 4 * 16

    def read(*keys):
        reads: list = []
        got = stage.traced(rec, lambda: [rec.part(0, kv, k) for k in keys], reads)()
        assert all(np.array_equal(x, rec.tensor(0, kv)[k]) for x, k in zip(got, keys))
        return wires.of(reads)

    a, b, whole = read((0, slice(1, 4)), (0, slice(2, 5))), read((1, slice(0, 2))), read((1, slice(0, 2)), (0,))
    assert a[0]["parts"] == [[[0, 1], [1, 4]], [[0, 1], [2, 5]]] and a[0]["bits"] == 4 * row, "token 0's rows 1 to 4, once"
    assert b[0]["bits"] == 2 * row and "parts" not in wires.source(0, kv), "a reference's own source stays whole"
    assert whole[0]["parts"] == [[[1, 2], [0, 2]], [[0, 1]]] and whole[0]["bits"] == 8 * row

    def counted(*instances):
        return stage.wiring({"S": {"gate": [{"key": [0, 4, i], "sources": s} for i, s in enumerate(instances)]}})["gap_bits"]

    assert counted(a, b, a) == 6 * row, "the union over instances, each row once: 6 of the 12 rows"
    assert counted(a, whole) == 8 * row, "a part over fewer axes takes the rest of them whole: all of token 0's rows"
    reads: list = []
    stage.traced(rec, lambda: [rec.tensor(0, kv)[0, 1:4]], reads)()
    assert counted(a, wires.of(reads)) == 12 * row, "a read through `Capture.tensor` takes the input whole"


KV_W = 6  # a cache row's words: kv_c_out's 4, then k_pe_out's 2
TOPK = 8
#: the step's tokens as (request, position): decodes a (10 keys, past the top-k) and b (6 keys), then c's 2-token prefill
TOKENS = [("a", 9), ("b", 5), ("c", 0), ("c", 1)]
BLOCKS = {"a": [3, 1, 6], "b": [0, 5], "c": [2]}  # each request's cache blocks, of `stage.KV_BLOCK` = 4 slots in the tests
PICKED = {0: [0, 1, 2, 4, 5, 7, 8, 9], 1: [0, 1, 2, 3, 4, 5], 2: [0], 3: [0, 1]}  # each token's top-k, as positions


def _slot(req: str, p: int) -> int:
    return BLOCKS[req][p // 4] * 4 + p % 4


@pytest.fixture()
def gather(tmp_path, monkeypatch):
    """A step whose tap gathers tokens 0, 1 and 3's K/V rows through its page table (top-k positions to slots), and the
    capture of it: with the step's cache write (``write``) and its positions (``positions``) recorded or not, a gathered row's
    bytes changed (``corrupt``: token 1's row 5, its own position), and the requests' scheduled counts. With ``convert`` the
    step records the call that converts the top-k to slots (`stage.CONVERT`), and these change what is served: a request's
    page-table blocks (``blocks``), a token's top-k (``picked``), the slot the write puts a token's row at (``moved``), an
    entry of the slots the call outputs and the tap reads (``page``, which its inputs don't give), the tap's valid counts
    (``used``), and the request the call takes a token as, whose page table it then reads (``owner``). It commits the batch
    (``req_ids`` in batch order, by default ``a``, ``b``, ``c``) under the rank root unless ``batch`` is False, and
    ``tamper`` rewrites ``meta.json``'s fields after the commit. ``pad`` appends that many padding tokens to the call and the
    tap, which vLLM's conversion takes as request 0 with no top-k."""
    stage = _stage(monkeypatch)
    monkeypatch.setattr(stage, "KV_BLOCK", 4)
    made = iter(range(100))

    def make(write=True, positions=True, corrupt=False, scheduled=None, convert=False, blocks=None, picked=None, moved=None,
             page=None, used=None, owner=None, req_ids=("a", "b", "c"), batch=True, tamper=None, pad=0):
        d = tmp_path / f"s{next(made)}"
        blocks, picked = {**BLOCKS, **(blocks or {})}, {**PICKED, **(picked or {})}
        owner = [{**dict(enumerate(q for q, _ in TOKENS)), **(owner or {})}[t] for t in range(len(TOKENS))]
        rng = np.random.default_rng(5)
        cache = {s: rng.integers(0, 2 ** 16, KV_W, dtype=np.uint16) for s in range(28)}
        kv_c, k_pe = (rng.integers(0, 2 ** 16, (len(TOKENS), w), dtype=np.uint16) for w in (4, 2))
        slots = np.array([_slot(q, p) for q, p in TOKENS], np.int64)
        for t, s in (moved or {}).items():
            slots[t] = s
        for t, s in enumerate(slots.tolist()):
            if s >= 0:
                cache[s] = np.concatenate([kv_c[t], k_pe[t]])
        table = np.full((len(TOKENS) + pad, TOPK), -1, np.int32)
        for t, ps in picked.items():
            table[t, :len(ps)] = [(blocks[owner[t]] + [0] * 3)[p // 4] * 4 + p % 4 for p in ps]
        served = table.copy()
        for (t, j), s in (page or {}).items():
            served[t, j] = s
        kept = [0, 1, 3]
        kv = np.zeros((len(kept), TOPK, KV_W), np.uint16)
        for i, t in enumerate(kept):
            for j, s in enumerate(served[t].tolist()):
                if s >= 0:
                    kv[i, j] = cache[s]
        if corrupt:
            kv[1, 5, 0] ^= 1
        b = _Blobs(d)
        events = [_ev(0, "_C::cooperative_topk", {"lengths": b.ref(np.array([[10], [6]], np.int32))}, {})]
        if write:
            ins = {"slot_mapping": b.ref(slots), **({"positions": b.ref(np.array([p for _, p in TOKENS]))} if positions else {})}
            events.append(_ev(1, "fused_norm_rope", ins, {"kv_c_out": b.ref(kv_c), "k_pe_out": b.ref(k_pe.reshape(-1, 1, 2))}))
        counts = np.array([len(picked[t]) for t in range(len(TOKENS))] + [0] * pad, np.int32)
        if convert:
            reqs = list(blocks)
            req = np.array([reqs.index(q) for q in owner] + [0] * pad, np.int32)
            bt = np.zeros((len(reqs), 3), np.int32)
            for k, q in enumerate(reqs):
                bt[k, :len(blocks[q])] = blocks[q]
            topk = np.full((len(TOKENS) + pad, TOPK), -1, np.int32)
            for t, ps in picked.items():
                topk[t, :len(ps)] = ps
            events.append(_ev(2, stage.CONVERT, {"req_id": b.ref(req), "block_table": b.ref(bt), "token_indices": b.ref(topk),
                                                  "BLOCK_SIZE": 4, "BLOCK_STRIDE_ROWS": 4, "NUM_TOPK_TOKENS": TOPK,
                                                  "return_valid_counts": True}, {"r0": b.ref(served), "r1": b.ref(counts)}))
        tap = {"block_table": b.ref(served), "seqused_k": b.ref(counts if used is None else np.array(used, np.int32)),
               "kv_gathered": b.ref(kv), "kv_gathered_rows": kept, "kv_gathered_tokens": len(kept)}
        events.append(_ev(len(events) + int(not write), "fa3_sparse_mla", tap, {}))
        _commit(d, events, {}, scheduled or {"a": 1, "b": 1, "c": 2}, list(req_ids) if convert and batch else None, tamper)
        rec = stage.Capture(d, 0)
        wires = stage.Wires(rec)
        kv_ref = rec.events[0][-1]["in"]["kv_gathered"]

        def of(*keys):
            reads: list = []
            stage.traced(rec, lambda: [rec.part(0, kv_ref, k) for k in keys], reads)()
            return wires.of(reads)
        return of
    return stage, make


def test_a_gathered_row_is_wired_to_the_steps_cache_write_by_slot_else_to_history(gather) -> None:
    stage, make = gather
    of = make()
    got = of((0, slice(4, 8)), (1, slice(2, 6)), (2, slice(0, 2)))
    assert [s["kind"] for s in got] == ["call", "history", "gap", "selection"]
    write, history, table, pick = got
    assert (write["seq"], write["op"], write["out"]) == (1, "fused_norm_rope", "kv_c_out+k_pe_out")
    assert write["rows"] == [[0, 7, 0], [1, 5, 1], [2, 0, 2], [2, 1, 3]], "each decode's own position, and the prefill's two"
    assert (history["port"], history["parts"]) == (stage.HISTORY_PORT, [[[0, 1], [4, 7]], [[1, 2], [2, 5]]])
    assert history["bits"] == 6 * KV_W * 16 and history["why"] == stage.HISTORY_WHY
    assert (table["in"], table["parts"]) == ("block_table", [[[0, 1], [4, 8]], [[1, 2], [2, 6]], [[3, 4], [0, 2]]])
    assert table["bits"] == 10 * 32, "every page-table entry a wired or history row is read through"
    from verity.ml.definitions.registry.dsa_topk import waksman_switches
    assert pick == {"kind": "selection", "rank": 0, "seq": 0, "token": 0, "via": "DsaWaksmanRows_v1{N=10,K=8,W=6,E=16}",
                    "bits": waksman_switches(10, TOPK), "why": stage.SELECTION_WHY}, "token 0's 10 keys pass a selection"
    w = stage.wiring({"S": {"gate": [{"key": [0, 2, 0], "sources": got}, {"key": [0, 2, 1], "sources": got}]}})
    assert (w["advice_bits"], w["gap_bits"], w["history_bits"], w["selection_bits"]) == (
        0, 10 * 32, 6 * KV_W * 16, waksman_switches(10, TOPK)), "the page-table entries are a gap, not advice"
    assert w["unproved_calls"] == {"fused_norm_rope": 1}


def test_a_row_at_a_slot_the_write_filled_with_other_bytes_or_a_long_row_without_positions_is_unwired(gather) -> None:
    stage, make = gather
    with pytest.raises(stage.Unwired, match=r"row 5 at slot 21, which fused_norm_rope \(seq 1\) wrote from its row 1, but not"):
        make(corrupt=True)((1, slice(2, 6)))
    of = make(positions=False)
    assert [s["kind"] for s in of((1, slice(2, 6)))] == ["call", "history", "gap"], "a short row needs no positions"
    with pytest.raises(stage.Unwired, match="token 0 reads 8 rows, and its keys"):
        of((0, slice(4, 8)))


def test_without_the_steps_cache_write_a_single_token_requests_other_rows_are_history_by_the_layout(gather) -> None:
    stage, make = gather
    history, left, table = make(write=False)((0, slice(4, 8)), (1, slice(2, 6)), (2, slice(0, 2)))
    assert history["parts"] == [[[1, 2], [2, 5]]] and history["bits"] == 3 * KV_W * 16, "token 1's rows but its own"
    assert left["parts"] == [[[0, 1], [4, 8]], [[1, 2], [5, 6]], [[2, 3], [0, 2]]], (
        "token 0 reads the top-k's 8 rows (its keys unknown), token 1's row 5 is its own position, token 3 is a prefill's")
    assert (table["in"], table["parts"], table["bits"]) == ("block_table", [[[1, 2], [2, 5]]], 3 * 32)
    every = make(write=False, scheduled={"a": 1, "b": 1, "c": 1, "d": 1})((1, slice(2, 6)))
    assert [s["kind"] for s in every] == ["gap"], "requests the top-k's decode rows don't match: no token is known single"


def test_own_rows_reads_the_paged_layout_in_any_order_and_refuses_a_row_that_breaks_it(gather) -> None:
    stage, _ = gather
    row = np.array

    assert stage.own_rows(row([21, 0, 20, 1, 2, 3, -1, -1]), 6, 4) == {0}, "the partly filled block's highest residue"
    assert stage.own_rows(row([4, 5, 6, 7, 0, 1, 2, 3, -1]), 8, 4) == {3, 7}, "whole blocks: each block's last residue"
    for bad, n in (([1, 2, -1], 2), ([0, 4, -1], 2), ([0, 1, 7], 2), ([0, 0, -1], 2), ([0, 1, 2, 3, 5, 6], 6)):
        assert stage.own_rows(row(bad), n, 4) is None, bad


def test_with_the_conversion_recorded_a_row_is_wired_by_request_and_position_and_no_page_table_entry_is_read(gather) -> None:
    stage, make = gather
    got = make(convert=True)((0, slice(4, 8)), (1, slice(2, 6)), (2, slice(0, 2)))
    assert [s["kind"] for s in got] == ["call", "history", "selection"], "no block_table source"
    write, history, pick = got
    assert (write["seq"], write["out"]) == (1, "kv_c_out+k_pe_out")
    assert write["rows"] == [[0, 7, 0], [1, 5, 1], [2, 0, 2], [2, 1, 3]], "each decode's own position, and the prefill's two"
    assert (history["port"], history["parts"]) == (stage.HISTORY_PORT, [[[0, 1], [4, 7]], [[1, 2], [2, 5]]])
    assert history["at"] == [[0, "a", [[5, 6], [7, 9]]], [1, "b", [[2, 5]]]], "each history row's request and position"
    assert pick["via"] == "DsaWaksmanRows_v1{N=10,K=8,W=6,E=16}" and pick["seq"] == 0
    w = stage.wiring({"S": {"gate": [{"key": [0, 3, 0], "sources": got}]}})
    from verity.ml.definitions.registry.dsa_topk import waksman_switches
    assert (w["advice_bits"], w["gap_bits"], w["history_bits"], w["selection_bits"]) == (
        0, 0, 6 * KV_W * 16, waksman_switches(10, TOPK)), "the page-table entries leave the accounting"


def test_convert_rows_is_the_served_conversion_and_a_capture_without_it_keeps_the_slot_path(gather) -> None:
    stage, make = gather
    topk = np.array([[0, 5, -1, 9, 4], [-1, -1, 1, 0, -1]])
    kept, slots, counts = stage.convert_rows(topk, np.array([0, 1]), np.array([[3, 1], [2, 0]], np.int32), 4, 6)
    assert [k.tolist() for k in kept] == [[0, 5, 4], [1, 0]], "an entry past the request's blocks is dropped, as -1 is"
    assert slots.tolist() == [[18, 7, 6, -1, -1], [13, 12, -1, -1, -1]] and counts.tolist() == [3, 2]
    assert [s["kind"] for s in make(convert=True, positions=False)((1, slice(2, 6)))] == ["call", "history", "gap"], (
        "without the write's positions a row isn't wired by position")
    assert [s["kind"] for s in make(convert=True, write=False)((1, slice(2, 6)))][-1] == "gap"


@pytest.mark.parametrize(("change", "message"), [
    ({"page": {(1, 3): 4}}, r"output slots or counts that its inputs don't give"),
    ({"used": [8, 5, 1, 2]}, r"reads its slots but not its valid counts as seqused_k"),
    ({"picked": {1: [0, 1, 2, 3, 4]}}, r"token 1 at position 5 has 5 valid keys, not min\(6, 8\)"),
    ({"picked": {1: [1, 0, 2, 3, 4, 5]}}, r"token 1's top-k row doesn't read its 6 keys in order"),
    ({"picked": {0: [0, 1, 2, 4, 5, 7, 8, 10]}}, r"token 0's top-k row isn't a selection of its 10 keys"),
    ({"moved": {1: 22}}, r"token 1 reads position 5 of its request at slot 21, but fused_norm_rope \(seq 1\) wrote it at slot 22"),
    ({"moved": {1: -1}}, r"token 1 reads position 5 of its request, which the step computes, but .* records no write"),
    ({"blocks": {"b": [2, 5]}}, r"token 1 reads position 0 of its request, an earlier step's, at slot 8, which .* filled"),
])
def test_a_conversion_or_a_read_the_wiring_by_position_does_not_hold_is_unwired(gather, change, message) -> None:
    stage, make = gather
    with pytest.raises(stage.Unwired, match=message):
        make(convert=True, **change)((0, slice(0, 8)), (1, slice(0, 6)))


@pytest.mark.parametrize(("change", "message"), [
    ({"owner": {0: "b", 1: "a"}}, r"takes token 0 as request 1, but the committed batch puts it at request 0 \(a\)"),
    ({"req_ids": ("c", "a", "b")}, r"takes token 1 as request 1, but the committed batch puts it at request 0 \(c\)"),
    ({"req_ids": ("a", "b")}, r"the committed batch's request ids aren't its scheduled requests, each once"),
    ({"req_ids": ("a", "b", "b", "c")}, r"the committed batch's request ids aren't its scheduled requests, each once"),
])
def test_a_token_the_conversion_puts_in_another_request_than_the_committed_batch_does_is_unwired(gather, change, message) -> None:
    """A swapped ``req_id`` whose slots its own page-table rows give, and a committed batch whose request ids put the tokens
    elsewhere or aren't the scheduled requests."""
    stage, make = gather
    with pytest.raises(stage.Unwired, match=message):
        make(convert=True, **change)((0, slice(0, 8)), (1, slice(0, 6)))


def test_padding_past_the_committed_batch_is_wired_from_nothing_and_a_gathered_token_there_is_unwired(gather) -> None:
    """A padded call wires as the unpadded one does; scheduled counts that leave a gathered token past the batch can't hand
    it a request."""
    stage, make = gather
    parts = (0, slice(4, 8)), (1, slice(2, 6)), (2, slice(0, 2))
    assert [(s["kind"], s.get("bits"), s.get("rows")) for s in make(convert=True, pad=2)(*parts)] == [
        (s["kind"], s.get("bits"), s.get("rows")) for s in make(convert=True)(*parts)]
    with pytest.raises(stage.Unwired, match=r"token 3 is past the committed batch's 3 tokens, so it has no request"):
        make(convert=True, scheduled={"a": 1, "b": 1, "c": 1})(*parts)


def test_a_batch_changed_after_its_commit_is_refused_and_an_uncommitted_one_keeps_the_slot_path(gather) -> None:
    stage, make = gather
    swapped, sched = ["b", "a", "c"], {"a": 1, "b": 1, "c": 2}
    for tamper in ({"req_ids": swapped}, {"scheduled": {"a": 2, "b": 1, "c": 1}}):
        with pytest.raises(SystemExit, match="req_ids or scheduled counts are not the batch it committed"):
            make(convert=True, tamper=tamper)
    with pytest.raises(SystemExit, match=r"rank root \w+ is not SHA-512 of its files"):
        make(convert=True, tamper={"req_ids": swapped, "batch_sha512": _batch(swapped, sched)})
    got = make(convert=True, batch=False)((0, slice(4, 8)), (1, slice(2, 6)), (2, slice(0, 2)))
    assert [s["kind"] for s in got] == ["call", "history", "gap", "selection"], "the page-table entries stay a gap"
