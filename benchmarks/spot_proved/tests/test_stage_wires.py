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


def _commit(d: Path, events: list[dict], params: dict, scheduled: dict | None = None) -> None:
    sd = d / "rank0" / "step0"
    sd.mkdir(parents=True, exist_ok=True)
    ops = "".join(json.dumps(e, sort_keys=True) + "\n" for e in events).encode()
    pj = json.dumps(params).encode()
    (sd / "ops.jsonl").write_bytes(ops)
    (sd / "params.json").write_bytes(pj)
    root = _sha(f"verity/spot/rank-root\0{_sha(ops)}\0{_sha(pj)}".encode())
    (sd / "meta.json").write_text(json.dumps({"scheduled": scheduled or {"req": 4}, "ops_sha512": _sha(ops), "params_sha512": _sha(pj),
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
    cap = stage.Capture(tmp_path, 0)
    return stage, cap, stage.Wires(cap)


def test_a_buffer_several_calls_wrote_in_parts_is_wired_to_each_call_for_its_rows(mixed) -> None:
    stage, cap, wires = mixed
    ev = cap.events[0]
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


def test_a_call_that_passes_its_inputs_bytes_on_is_passed_over_and_dense_fa3_is_no_populations(mixed) -> None:
    stage, cap, wires = mixed
    ev = cap.events[0]
    sparse = wires.source(0, ev[6]["in"]["self"])
    assert (sparse["kind"], sparse["seq"], sparse["op"]) == ("block", 4, "_vllm_fa3_C::fwd"), "aten::to only passed it on"
    dense = wires.source(0, ev[7]["in"]["self"])
    assert (dense["kind"], dense["seq"]) == ("call", 0), "copy_ passed FA3's bytes on; dense FA3 is committed, not proved"
    only = wires.source(0, ev[9]["in"]["self"])
    assert (only["kind"], only["seq"], only["op"]) == ("call", 8, "aten::to"), "no earlier call output them"
    gated, other, _ = stage.op_coverage(cap)
    assert (gated.get("_vllm_fa3_C::fwd"), other.get("_vllm_fa3_C::fwd")) == (1, 1)


def test_a_parameter_is_registered_and_a_gap_input_counted_at_its_width(step) -> None:
    stage, cap, wires = step
    ev = cap.events[0]
    assert wires.source(0, ev[0]["in"]["mat2"]) == {"kind": "registered", "rank": 0, "param": "w", "sha512": "ab" * 64}
    gap = wires.source(0, ev[4]["in"]["kv_gathered"])
    assert (gap["kind"], gap["bits"]) == ("gap", 2 * 6 * 4 * 16) and gap["why"] == stage.GAPS[("fa3_sparse_mla", "kv_gathered")]
    assert not stage.ADVICE, "no staged population reads a free choice today"


def test_a_free_choice_is_charged_as_advice_and_a_gap_is_counted_apart_never_charged(step, monkeypatch) -> None:
    stage, cap, wires = step
    ev = cap.events[0]
    monkeypatch.setitem(stage.ADVICE, ("aten::add", "other"), "a choice nothing computes")
    adv, gap = wires.source(0, ev[2]["in"]["other"]), wires.source(0, ev[4]["in"]["kv_gathered"])
    assert (adv["kind"], adv["bits"], adv["why"]) == ("advice", 4 * 8 * 16, "a choice nothing computes")
    w = stage.wiring({"A": {"gate": [{"key": [0, 2, 0], "sources": [adv, gap]}]}})
    assert (w["advice_bits"], w["gap_bits"]) == (4 * 8 * 16, 2 * 6 * 4 * 16)


def test_an_input_no_earlier_call_output_is_unwired_with_the_fix_and_the_enclosing_calls_output_too(step) -> None:
    stage, cap, wires = step
    ev = cap.events[0]
    for _ in range(2):
        with pytest.raises(stage.Unwired, match=r"reads other, .*declare \('aten::add', 'other'\) in stage\.GAPS if the step "
                                                r"computes it .* or in stage\.ADVICE if nothing does"):
            wires.source(0, ev[2]["in"]["other"])
    with pytest.raises(stage.Unwired, match="reads self"):
        wires.source(0, ev[6]["in"]["self"])


def test_traced_records_the_reads_of_an_instances_arguments_and_wiring_counts_a_gap_once(step) -> None:
    stage, cap, wires = step
    ev = cap.events[0]
    reads: list = []
    args = stage.traced(cap, lambda: [cap.tensor(0, ev[1]["in"]["self"])[0], cap.tensor(0, ev[1]["in"]["self"])[1],
                                      cap.tensor(0, ev[4]["in"]["kv_gathered"])[0]], reads)()
    assert len(args) == 3 and len(reads) == 3 and cap.reads is None
    sources = wires.of(reads)
    assert [s["kind"] for s in sources] == ["block", "gap"]
    types = {"A": {"gate": [{"key": [0, 1, 0], "sources": sources}, {"key": [0, 1, 1], "sources": sources},
                            {"key": [0, 2, 0], "unwired": "rank 0 ..."}]}}
    w = stage.wiring(types)
    assert (w["gap_bits"], w["advice_bits"]) == (2 * 6 * 4 * 16, 0)
    assert len(w["inputs"]["A"]) == 2 and w["unwired"] == {"A": ["rank 0 ..."]}


def test_a_gap_input_is_counted_at_the_union_of_the_parts_staged_instances_read(step) -> None:
    stage, cap, wires = step
    kv = cap.events[0][4]["in"]["kv_gathered"]
    row = 4 * 16

    def read(*keys):
        reads: list = []
        got = stage.traced(cap, lambda: [cap.part(0, kv, k) for k in keys], reads)()
        assert all(np.array_equal(x, cap.tensor(0, kv)[k]) for x, k in zip(got, keys))
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
    stage.traced(cap, lambda: [cap.tensor(0, kv)[0, 1:4]], reads)()
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
    bytes changed (``corrupt``: token 1's row 5, its own position), and the requests' scheduled counts."""
    stage = _stage(monkeypatch)
    monkeypatch.setattr(stage, "KV_BLOCK", 4)

    def make(write=True, positions=True, corrupt=False, scheduled=None):
        d = tmp_path / f"w{int(write)}p{int(positions)}c{int(corrupt)}{len(scheduled or ())}"
        rng = np.random.default_rng(5)
        cache = {s: rng.integers(0, 2 ** 16, KV_W, dtype=np.uint16) for s in range(28)}
        kv_c, k_pe = (rng.integers(0, 2 ** 16, (len(TOKENS), w), dtype=np.uint16) for w in (4, 2))
        slots = np.array([_slot(q, p) for q, p in TOKENS], np.int64)
        for t, s in enumerate(slots.tolist()):
            cache[s] = np.concatenate([kv_c[t], k_pe[t]])
        page = np.full((len(TOKENS), TOPK), -1, np.int32)
        for t, ps in PICKED.items():
            page[t, :len(ps)] = [_slot(TOKENS[t][0], p) for p in ps]
        kept = [0, 1, 3]
        kv = np.zeros((len(kept), TOPK, KV_W), np.uint16)
        for i, t in enumerate(kept):
            for j, s in enumerate(page[t].tolist()):
                if s >= 0:
                    kv[i, j] = cache[s]
        if corrupt:
            kv[1, 5, 0] ^= 1
        b = _Blobs(d)
        events = [_ev(0, "_C::cooperative_topk", {"lengths": b.ref(np.array([[10], [6]], np.int32))}, {})]
        if write:
            ins = {"slot_mapping": b.ref(slots), **({"positions": b.ref(np.array([p for _, p in TOKENS]))} if positions else {})}
            events.append(_ev(1, "fused_norm_rope", ins, {"kv_c_out": b.ref(kv_c), "k_pe_out": b.ref(k_pe.reshape(-1, 1, 2))}))
        used = np.array([len(PICKED[t]) for t in range(len(TOKENS))], np.int32)
        events.append(_ev(2, "fa3_sparse_mla", {"block_table": b.ref(page), "seqused_k": b.ref(used), "kv_gathered": b.ref(kv),
                                                "kv_gathered_rows": kept, "kv_gathered_tokens": len(kept)}, {}))
        _commit(d, events, {}, scheduled or {"a": 1, "b": 1, "c": 2})
        cap = stage.Capture(d, 0)
        wires = stage.Wires(cap)
        kv_ref = cap.events[0][-1]["in"]["kv_gathered"]

        def of(*keys):
            reads: list = []
            stage.traced(cap, lambda: [cap.part(0, kv_ref, k) for k in keys], reads)()
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
