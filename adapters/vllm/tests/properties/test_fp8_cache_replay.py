"""The FP8 cache stand-in (`fp8_cache_replay_gpu`) on the CPU, with Model Runner V2 stood in for: `Rows` reads each request's
cached rows by position, in a bf16 cache and in a 656-byte FP8 one, across chunked prefill, a prefix-cache hit and a recompute
after a preemption; what of ours it records (the history tables' guard passes the bf16 cache and refuses the FP8 one by name,
and the tables' hook reads `Rows`' bytes); its comparison of a replay with its serve, which finds one differing byte of
one row, a replay on another cache format, and a differing token; and its count of the attention's routes and the write
kernel's PTX it keeps."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from tests.properties import fp8_cache_replay_gpu as X

BLOCK = 4
#: a cache format -> its tensor's dtype and a row's last dimension (vLLM's ``fp8_ds_mla`` row: 512 e4m3, four f32 scales, 64 bf16)
FORMATS = {"auto": (torch.bfloat16, 576), "fp8_ds_mla": (torch.uint8, 656)}


class Layer(torch.nn.Module):
    """A layer with a bound cache, named as vLLM names it: an MLA attention layer (``fmt``) by its ``layer_name``, an
    indexer's key cache by its ``prefix`` alone."""

    def __init__(self, name: str, dtype: torch.dtype, width: int, fmt: str | None) -> None:
        super().__init__()
        self.name, self.kv_cache = name, torch.zeros(16, BLOCK, width, dtype=dtype)
        if fmt is None:
            self.prefix = name
        else:
            self.layer_name, self.kv_lora_rank, self.qk_rope_head_dim, self.kv_cache_dtype = name, 512, 64, fmt


class Runner:
    """Model Runner V2 stood in for: `execute_model` writes each scheduled position's row into every cache at its slot and
    leaves the step's batch and slots in ``execute_model_state``, padded past the batch. ``flip``: one (request, position,
    byte) of the first layer written with its low bit flipped."""

    def __init__(self, fmt: str, flip: tuple[str, int, int] | None = None) -> None:
        dtype, width = FORMATS[fmt]
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([Layer("model.layers.0.self_attn.attn", dtype, width, fmt),
                                                 Layer("model.layers.0.self_attn.indexer.k_cache", torch.uint8, 132, None),
                                                 Layer("model.layers.3.self_attn.attn", dtype, width, fmt)])
        self.bytes = {m.name: m.kv_cache.shape[-1] * m.kv_cache.element_size() for m in self.model.layers}
        self.blocks: dict[str, list[int]]
        self.gen: dict[str, int]
        self.execute_model_state: NS | None
        self.blocks, self.free, self.gen, self.flip, self.execute_model_state = {}, list(range(16)), {}, flip, None

    def row(self, request: str, position: int, layer: str) -> np.ndarray:
        seed = [sum(map(ord, request)), position, sum(map(ord, layer)), self.gen.get(request, 0)]
        out = np.random.default_rng(seed).integers(0, 256, self.bytes[layer], dtype=np.uint8)
        if self.flip is not None and self.flip[:2] == (request, position) and layer == self.model.layers[0].name:
            out[self.flip[2]] ^= 1
        return out

    def share(self, request: str, other: str, blocks: int) -> None:
        self.blocks[request] = self.blocks[other][:blocks]

    def preempt(self, request: str) -> None:
        self.free += self.blocks.pop(request)
        self.gen[request] = self.gen.get(request, 0) + 1

    def execute_model(self, so: NS) -> None:
        ids, qsl, nct, slots = list(so.num_scheduled_tokens), [0], [], []
        for rid in ids:
            start, n = so.starts[rid], so.num_scheduled_tokens[rid]
            for p in range(start, start + n):
                bs = self.blocks.setdefault(rid, [])
                while len(bs) <= p // BLOCK:
                    bs.append(self.free.pop(0))
                slot = bs[p // BLOCK] * BLOCK + p % BLOCK
                for m in self.model.layers:
                    m.kv_cache.view(torch.uint8).view(-1, self.bytes[m.name])[slot] = torch.from_numpy(self.row(rid, p, m.name))
                slots.append(slot)
            qsl.append(qsl[-1] + n)
            nct.append(start)
        self.execute_model_state = NS(
            input_batch=NS(req_ids=ids, query_start_loc_np=np.array(qsl + [qsl[-1]] * 3, dtype=np.int32),
                           num_computed_tokens_np=np.array(nct + [0, 0], dtype=np.int32)),
            slot_mappings_by_layer=dict.fromkeys(self.bytes, torch.tensor(slots + [-1, -1], dtype=torch.int32)))


def step(**sched: tuple[int, int]) -> NS:
    """A scheduler output: each request's (start, tokens), in batch order."""
    return NS(num_scheduled_tokens={r: n for r, (_s, n) in sched.items()}, starts={r: s for r, (s, _n) in sched.items()},
              total_num_scheduled_tokens=sum(n for _s, n in sched.values()), scheduled_spec_decode_tokens={})


#: request -> (its first computed position, how many it computed): a prefills in two chunks and decodes; b is preempted after
#: three positions and recomputed from 0; c hits a's first block and computes from there
SERVED = {"a": (0, 6), "b": (0, 4), "c": (BLOCK, 3)}


def serve(fmt: str, flip: tuple[str, int, int] | None = None, hook: bool = False) -> tuple[Runner, X.Rows, dict, object]:
    runner = Runner(fmt, flip)
    rows = X.Rows(runner)
    said, tables = X.ours(runner, {"kv_cache_dtype": None if fmt == "auto" else fmt})

    def go(**sched: tuple[int, int]) -> None:
        so = step(**sched)
        runner.execute_model(so)
        rows.after(runner)
        if hook and tables is not None:
            tables.after(runner, so)

    go(a=(0, 3))
    go(b=(0, 3), a=(3, 2))
    runner.share("c", "a", 1)
    go(a=(5, 1), c=(BLOCK, 2))
    runner.preempt("b")
    go(b=(0, 4), c=(BLOCK + 2, 1))
    return runner, rows, said, tables


@pytest.mark.parametrize("fmt", FORMATS)
def test_rows_are_each_requests_cache_rows_by_position_across_chunks_a_hit_and_a_recompute(fmt):
    runner, rows, _said, _tables = serve(fmt)
    digests, raw = rows.digest()
    assert rows.steps == 4 and rows.skipped == {} and sorted(digests) == sorted(SERVED)
    for rid, (first, n) in SERVED.items():
        for name, width in runner.bytes.items():
            want = np.stack([runner.row(rid, p, name) for p in range(first, first + n)])
            assert raw[f"{rid}|{name}"].tobytes() == want.tobytes()
            assert digests[rid][name] == {"first": first, "positions": n, "row_bytes": width,
                                          "sha512": hashlib.sha512(want.tobytes()).hexdigest()}
    assert {x["row_bytes"] for by in digests.values() for n, x in by.items() if "indexer" not in n} == {FORMATS[fmt][1] * (2 if fmt == "auto" else 1)}


def test_the_tables_guard_passes_a_bf16_cache_and_refuses_an_fp8_one_by_name():
    _runner, rows, said, tables = serve("auto", hook=True)
    assert said == {"header_kv_cache_dtype": None, "tables_guard": {"verdict": "ok", "layers": [0, 3]}}
    # seven blocks of sixteen rows in all at each of two layers, c's from its hit on, each `Rows`' bytes
    assert X.tables_rows(tables, rows) == {"failed": None, "blocks": 14, "rows": 32, "differing": []}
    _runner, _rows, said, tables = serve("fp8_ds_mla")
    assert tables is None and said["header_kv_cache_dtype"] == "fp8_ds_mla" and said["tables_guard"]["verdict"] == "refuses"
    assert "uint8" in said["tables_guard"]["message"] and "--kv-cache-dtype auto" in said["tables_guard"]["message"]


def doc(tmp_path, tag: str, fmt: str, flip: tuple[str, int, int] | None = None, tokens: dict | None = None) -> tuple[dict, Path]:
    runner, rows, _said, _tables = serve(fmt, flip)
    digests, raw = rows.digest()
    np.savez(tmp_path / f"{tag}.npz", **raw)
    eng = NS(vllm_config=NS(cache_config=NS(cache_dtype=fmt, block_size=BLOCK)))
    steps = [{**{k: i for k in X.STEP_KEYS}, "cache_hit": {"c": 4} if i == 2 else {}} for i in range(rows.steps)]
    return ({"steps": steps, "tokens": tokens or {"a": [5, 6, 7], "b": [8]}, "rows": digests, "chosen": X.chosen(eng, runner),
             "empty_schedules": 0, "routes": {"top-k": rows.steps}, "topk": None, "rows_skipped": rows.skipped},
            tmp_path / f"{tag}.npz")


def test_a_replay_equals_its_serve_only_with_every_step_token_and_row_byte_on_the_same_engine(tmp_path):
    served, a = doc(tmp_path, "serve", "fp8_ds_mla")
    layers = served["chosen"]["layers"]
    assert served["chosen"]["cache_dtype"] == "fp8_ds_mla" and [(x["mla"], x["dtype"], x["row_bytes"]) for x in layers] == [
        (True, "uint8", 656), (False, "uint8", 132), (True, "uint8", 656)]
    same, b = doc(tmp_path, "same", "fp8_ds_mla")
    got = X.compare(served, same, (a, b))
    assert got["equal"] and got["rows"] == {"sets": 9, "differing": 0, "first": []} and json.dumps(got)
    assert got["routes_replayed"] == {"top-k": 4}
    said = X.summary(served)
    assert said["routes"] == {"top-k": 4} and said["topk"] is None and said["cache_hits"] == {"c": 4} and json.dumps(said)
    assert (said["cache_dtype"], said["steps"], said["tokens"], said["distinct_tokens"]) == ("fp8_ds_mla", 4, 4, 4)

    one, b = doc(tmp_path, "one", "fp8_ds_mla", flip=("a", 2, 513))
    got = X.compare(served, one, (a, b))
    assert not got["equal"] and got["same_engine"] and got["first_step_differing"] is None and not got["tokens_differing"]
    assert got["rows"]["differing"] == 1 and got["rows"]["first"] == [{
        "request": "a", "layer": "model.layers.0.self_attn.attn", "served": 6, "replayed": 6, "rows_differing": 1,
        "first_position": 2, "first_byte": 513}]

    bf16, b = doc(tmp_path, "bf16", "auto")
    got = X.compare(served, bf16, (a, b))
    assert not got["equal"] and not got["same_engine"] and got["replay_cache_dtype"] == "auto" and got["rows"]["differing"] == 6
    assert {(x.get("served_row_bytes"), x.get("replayed_row_bytes")) for x in got["rows"]["first"]} == {(656, 1152)}

    other, b = doc(tmp_path, "tokens", "fp8_ds_mla", tokens={"a": [5, 6, 9], "b": [8]})
    got = X.compare(served, other, (a, b))
    assert not got["equal"] and got["tokens_differing"] == [{"request": "a", "tokens": 3, "first_position": 2}]


def test_routes_count_each_decision_by_what_it_reads_and_a_write_kernels_ptx_is_kept_once(tmp_path):
    from verity_vllm.engine.hooks import Hooks

    class Attn(Layer):
        def _use_sparse_mha(self, attn_metadata):
            return attn_metadata.prefill is not None and attn_metadata.prefill.dense

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([Layer("indexer", torch.uint8, 132, None), Attn("attn", torch.uint8, 656, "fp8_ds_mla")])
    hooks = Hooks(tag="test")
    routes = X.Routes(hooks, model)
    asked = [NS(prefill=None), NS(prefill=NS(dense=True, chunked_context=None)), NS(prefill=NS(dense=True, chunked_context=object())),
             NS(prefill=NS(dense=False, chunked_context=object())), NS(prefill=None)]
    assert [model.layers[1]._use_sparse_mha(m) for m in asked] == [False, True, True, False, False]
    assert routes.count == {"top-k": 3, "dense": 1, "dense, cached context": 1}
    hooks.uninstall()
    assert model.layers[1]._use_sparse_mha(asked[1]) and sum(routes.count.values()) == 5
    bare = torch.nn.Module()
    bare.layers = torch.nn.ModuleList([Layer("attn", torch.uint8, 656, "fp8_ds_mla")])
    assert X.Routes(Hooks(tag="test"), bare).count == {}

    seen: set[str]
    cache, seen = tmp_path / "triton", set()
    for key, text in (("a" * 40, "div.full.f32 %f1, %f2, %f3;\ndiv.full.f32 %f4, %f5, %f6;\nrem.s32 %r1, %r2, %r3;"),
                      ("b" * 40, "div.rn.f32 %f1, %f2, %f3;\ndiv.s64 %rd1, %rd2, %rd3;")):
        (cache / key).mkdir(parents=True)
        (cache / key / f"{X.WRITE_KERNEL}.ptx").write_text(text)
        (cache / key / "_fused_q_kernel.ptx").write_text("div.rn.f32")
        got = X.keep_ptx(cache, tmp_path / "ptx" / key[0], seen)
        assert [(x["file"], x["div"]) for x in got] == [
            (f"{key[:12]}-{X.WRITE_KERNEL}.ptx", {"div.full.f32": 2} if key[0] == "a" else {"div.rn.f32": 1})]
        assert (tmp_path / "ptx" / key[0] / got[0]["file"]).read_text() == text
    assert X.keep_ptx(cache, tmp_path / "ptx" / "c", seen) == [] and not (tmp_path / "ptx" / "c").exists()


def test_the_plan_shares_a_prefix_holds_one_prompt_past_the_top_k_and_samples_half(tmp_path, monkeypatch):
    plan = X.plan(7)
    # without a canonical top-k no prompt reaches the indexer's top-k, and the rest of the plan keeps its shape
    short = X.plan(7, long=False)
    assert max(len(r["prompt_ids"]) for r in short) <= 610 and [r["id"] for r in short] == [r["id"] for r in plan]
    assert X.canonical_topk() is None

    def wrapped():
        pass

    wrapped.__dict__["dsa_topk_canon"] = canon = object()
    monkeypatch.setattr(torch.ops._C, X.TOPK_OPS[2], wrapped, raising=False)
    assert X.canonical_topk() is canon
    assert plan == X.plan(7) and len(plan) == X.REQUESTS and len({r["id"] for r in plan}) == X.REQUESTS
    shared = [r for i, r in enumerate(plan) if i % 3 == 2]
    assert len(shared) == 4 and len({tuple(r["prompt_ids"][:200]) for r in shared}) == 1
    assert max(len(r["prompt_ids"]) for r in plan) == X.LONG > 2048
    assert [r["sampling"] for r in plan[:2]] == [{"temperature": 0.0, "seed": 7}, {"temperature": 1.0, "top_p": 0.95, "seed": 1001}]
    X.stand_in(tmp_path)
    cut = json.loads((tmp_path / "config.json").read_text())
    assert (cut["num_hidden_layers"], cut["n_routed_experts"], len(cut["indexer_types"]), len(cut["mlp_layer_types"])) == (
        X.LAYERS, X.EXPERTS, X.LAYERS, X.LAYERS)
    # a rank's heads at tensor-parallel 8, below the 32 at which vLLM reads an FP8 cache's prefill rows through a bf16 copy
    assert cut["num_attention_heads"] == cut["num_key_value_heads"] == X.HEADS == 64 // 8 < 32
