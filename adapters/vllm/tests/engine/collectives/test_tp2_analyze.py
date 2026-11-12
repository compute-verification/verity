"""The torch-free helpers of bench/tp2_analyze (the collective body arithmetic and the shard / position classifiers), and its five
reports over a two-rank run written here with `verity_vllm.observe.log.LogWriter` in the layout `tp.capture` leaves."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tests.engine.collectives import analyze as ta
from verity_vllm.observe.events import (
    SCHEMA,
    DispatchEnter,
    DispatchExit,
    Root,
    RunHeader,
    StepStart,
    TritonLaunch,
    ViewDesc,
)
from verity_vllm.observe.log import LogWriter


def test_bf16_round_trip_and_rne_ties():
    vals = np.array([0.0, 1.0, -2.5, 1.0 / 3.0, 65504.0, 1e-3], dtype=np.float32)
    bits = ta.f32_to_bf16_rne(vals)
    back = ta.bf16_to_f32(bits)
    # every bf16 value is exactly representable in f32, and re-rounding is idempotent
    assert np.array_equal(ta.f32_to_bf16_rne(back), bits)
    assert np.all(np.abs(back - vals) <= np.abs(vals) * 2.0 ** -8 + 1e-30)
    # 1 + 2^-8 is exactly halfway between the bf16 neighbours 1.0 and 1 + 2^-7: RNE rounds to the even mantissa (1.0)
    tie = np.array([1.0 + 2.0 ** -8], dtype=np.float32)
    assert ta.bf16_to_f32(ta.f32_to_bf16_rne(tie))[0] == 1.0
    # 1 + 3 * 2^-8 is halfway between 1 + 2^-7 and 1 + 2^-6: rounds to even = 1 + 2^-6
    tie2 = np.array([1.0 + 3.0 * 2.0 ** -8], dtype=np.float32)
    assert ta.bf16_to_f32(ta.f32_to_bf16_rne(tie2))[0] == 1.0 + 2.0 ** -6


def test_all_reduce_body_is_single_rounding_of_the_f32_sum():
    rng = np.random.default_rng(0)
    a = ta.bf16_to_f32(ta.f32_to_bf16_rne(rng.standard_normal(4096).astype(np.float32)))
    b = ta.bf16_to_f32(ta.f32_to_bf16_rne(rng.standard_normal(4096).astype(np.float32)))
    body = ta.f32_to_bf16_rne(a + b)
    # commutative for world 2 (the order of the two shards cannot matter)
    assert np.array_equal(body, ta.f32_to_bf16_rne(b + a))
    # and equals the sequential bf16 chain, which for two operands is the same single rounding
    assert np.array_equal(body, ta.f32_to_bf16_rne(ta.bf16_to_f32(ta.f32_to_bf16_rne(a)) + b))


def test_classify_shard_rules():
    w = 2
    assert ta.classify_shard((1536,), (1536,), w, "model.norm.weight", {}) == "replicated"
    assert ta.classify_shard((151936, 1536), (75968, 1536), w, "model.embed_tokens.weight", {}).startswith("dim0: contiguous vocab range")
    assert ta.classify_shard((2048, 1536), (1024, 1536), w, "model.layers.0.self_attn.qkv_proj.weight", {}).startswith("dim0: fused [q | k | v]")
    assert ta.classify_shard((2048,), (1024,), w, "model.layers.0.self_attn.qkv_proj.bias", {}).startswith("dim0: fused [q | k | v] bias")
    assert ta.classify_shard((17920, 1536), (8960, 1536), w, "model.layers.0.mlp.gate_up_proj.weight", {}).startswith("dim0: fused [gate | up]")
    assert ta.classify_shard((1536, 1536), (1536, 768), w, "model.layers.0.self_attn.o_proj.weight", {}).startswith("dim1: contiguous input-feature")
    assert ta.classify_shard((1536, 8960), (1536, 4480), w, "model.layers.0.mlp.down_proj.weight", {}).startswith("dim1:")
    assert ta.classify_shard((3, 4), (2, 4), w, "x", {}).startswith("other:")
    assert ta.classify_shard((8, 4), (4, 4), w, "model.layers.0.linear.weight", {}) == "dim0: contiguous output-feature slice (ColumnParallelLinear)"
    assert ta.classify_shard((8,), (4,), w, "model.layers.0.linear.bias", {}) == "dim0: output-feature slice (bias)"
    assert ta.classify_shard((2, 9, 8, 16, 64), (2, 9, 4, 16, 64), w, "kv_cache", {}) == "KV cache: heads split across ranks"


def test_collective_positions_and_op_filter():
    assert ta.is_collective_op("vllm.all_reduce.default")
    assert ta.is_collective_op("vllm.all_gather.default")
    assert ta.is_collective_op("_C.vocab_parallel_embedding.default")
    assert not ta.is_collective_op("aten.mm.default")
    assert ta.position_of("model.layers.3.self_attn.o_proj", "vllm.all_reduce.default").startswith("layer 3: after o_proj")
    assert ta.position_of("model.layers.27.mlp.down_proj", "vllm.all_reduce.default").startswith("layer 27: after down_proj")
    assert ta.position_of("model.embed_tokens", "vllm.all_reduce.default").startswith("embedding: sum")
    assert ta.position_of("model.embed_tokens", "_C.vocab_parallel_embedding.default").startswith("embedding: fused masked lookup")
    assert ta.position_of("logits_processor", "vllm.all_gather.default").startswith("logits: gather")


REDUCE, GATHER, OTHER = "vllm.all_reduce.default", "vllm.all_gather.default", "c10d.broadcast"
#: bfloat16 bit patterns, written out: 1, 2, 0.5, -1, 3, 0.25 on rank 0 and 1, 1, 0.5, 1, -3, 0.75 on rank 1; their sums 2, 3, 1, 0, 0, 1
SHARDS = ((0x3F80, 0x4000, 0x3F00, 0xBF80, 0x4040, 0x3E80), (0x3F80, 0x3F80, 0x3F00, 0x3F80, 0xC040, 0x3F40))
SUMMED = (0x4000, 0x4040, 0x3F80, 0x0000, 0x0000, 0x3F80)
GATHERED = ((1, 2, 5, 6), (3, 4, 7, 8))
#: the six collectives of each rank's log, in order: (event id, op, module path)
REDUCE_OK, REDUCE_WRONG, REDUCE_BARE, REDUCE_LOST, GATHER_OK, BROADCAST = 10, 20, 30, 40, 50, 60


def _view(alloc: int, shape: tuple[int, ...], dtype: str = "bfloat16", offset: int = 0) -> ViewDesc:
    strides = tuple(int(np.prod(shape[i + 1:])) for i in range(len(shape)))
    return ViewDesc(alloc, offset, shape, strides, dtype, "cuda:0")


def _step() -> StepStart:
    return StepStart(step=0, rows=[("r0", 0)], input_token_ids=[1], positions=[0], seq_lens={"r0": 1}, num_computed={"r0": 0},
                     block_tables={"r0": [0]}, block_size=16, slot_mapping=[0], logits_rows=[("r0", 0)], logits_indices=[0], num_reqs=1,
                     num_tokens=1, num_tokens_padded=1)


def _header(rank: int) -> RunHeader:
    return RunHeader(SCHEMA, "2026-01-01T00:00:00Z", "profile-of-the-run", {}, {}, {}, {"tp2": {"world_size": 2, "rank": rank, "facts": {"big": 1}}})


def _snapshot(rank_dir: Path, entries: list[dict], event: int, path: str, view: ViewDesc, values: np.ndarray, keep_file: bool = True) -> None:
    """One entry of the snapshot index as `verity_vllm.observe.capture` writes it, and its content-addressed file."""
    data = values.tobytes()
    sha = hashlib.sha256(data).hexdigest()
    name = f"values/0/{sha[:16]}.bin"
    if keep_file:
        (rank_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (rank_dir / name).write_bytes(data)
    entries.append({"event": event, "path": path, "view": view.to_json(), "dtype": view.dtype, "shape": list(view.shape),
                    "strides": list(view.strides), "byte_offset": view.offset * view.itemsize, "nbytes": len(data),
                    "layout": "spanned-storage-bytes", "sha256": sha, "file": name, "step": 0})


def _rank(run: Path, rank: int, wrong_out: int) -> None:
    """One rank of a run: its log, its snapshot index and the snapshots' bytes.  ``wrong_out`` is the first element of this
    rank's result of the second all-reduce (the true sum on rank 0)."""
    shard, summed = np.array(SHARDS[rank], np.uint16).reshape(2, 3), np.array(SUMMED, np.uint16).reshape(2, 3)
    wrong = summed.copy()
    wrong[0, 0] = wrong_out
    part = np.array(GATHERED, np.int32)[:, 2 * rank:2 * rank + 2]
    half, whole, wide = _view(1, (2, 3)), _view(2, (2, 3), offset=6), _view(4, (2, 4), "int32")
    narrow = _view(3, (2, 2), "int32")
    with LogWriter(run / f"log.rank{rank}.jsonl.gz") as log:
        log.write(_header(rank))
        log.write(Root(_view(90, (4, 2)), "weight", "model.layers.0.self_attn.o_proj.weight"))
        log.write(Root(_view(91, (4,)), "weight", "model.norm.weight"))
        log.write(Root(_view(92, (3,)), "buffer", "model.only_rank"))
        log.write(_step())
        log.write(DispatchEnter(REDUCE_OK, None, 0, "forward", REDUCE, "", [half], {}, 0, "model.layers.0.self_attn.o_proj"))
        log.write(DispatchEnter(11, REDUCE_OK, 0, "forward", "aten.add.Tensor", "", [half, half], {}, 0))
        log.write(TritonLaunch(12, REDUCE_OK, 0, "forward", "pkg.kernels.add_kernel", "sha", {}, {}, [1], None, None, 0))
        log.write(DispatchExit(11, whole))
        log.write(DispatchExit(REDUCE_OK, whole))
        log.write(DispatchEnter(REDUCE_WRONG, None, 0, "forward", REDUCE, "", [half], {}, 0, "model.layers.1.mlp.down_proj"))
        log.write(DispatchExit(REDUCE_WRONG, whole))
        log.write(DispatchEnter(REDUCE_BARE, None, 0, "forward", REDUCE, "", [half], {}, 0, "model.embed_tokens"))
        log.write(DispatchEnter(REDUCE_LOST, None, 0, "forward", REDUCE, "", [half], {}, 0))
        log.write(DispatchExit(REDUCE_LOST, whole))
        log.write(DispatchEnter(GATHER_OK, None, 0, "sample", GATHER, "", [narrow, -1], {}, 0, "logits_processor"))
        log.write(DispatchExit(GATHER_OK, wide))
        log.write(DispatchEnter(BROADCAST, None, 0, "other", OTHER, "", [half], {}, 0, "x.y"))
        log.write(DispatchExit(BROADCAST, half))
        log.write(DispatchEnter(70, None, 0, "forward", "aten.mm.default", "", [half, half], {}, 0))
        log.write(DispatchExit(70, half))
    entries: list[dict] = []
    rank_dir = run / f"rank{rank}"
    _snapshot(rank_dir, entries, REDUCE_OK, "args[0]", half, shard)
    _snapshot(rank_dir, entries, REDUCE_OK, "outs", whole, summed)
    _snapshot(rank_dir, entries, REDUCE_WRONG, "args[0]", half, shard)
    _snapshot(rank_dir, entries, REDUCE_WRONG, "outs", whole, wrong)
    _snapshot(rank_dir, entries, REDUCE_LOST, "args[0]", half, shard + 1, keep_file=rank == 0)
    _snapshot(rank_dir, entries, REDUCE_LOST, "outs", whole, summed + 1)
    _snapshot(rank_dir, entries, GATHER_OK, "args[0]", narrow, part)
    _snapshot(rank_dir, entries, GATHER_OK, "outs", wide, np.array(GATHERED, np.int32))
    _snapshot(rank_dir, entries, BROADCAST, "args[0]", half, shard)
    _snapshot(rank_dir, entries, BROADCAST, "outs", half, shard)
    (rank_dir / "snapshots.json").write_text(json.dumps({"entries": entries}))


def _run(run: Path, wrong_out: int = 0x4001) -> Path:
    """A two-rank run whose second all-reduce has a wrong first element on rank 1, and whose fourth lost a file on rank 1."""
    _rank(run, 0, SUMMED[0])
    _rank(run, 1, wrong_out)
    (run / "tokens.json").write_text(json.dumps({"requests": [{"generated_token_ids": [5, 6]}]}))
    return run


def test_inventory_lists_each_ranks_collectives_with_their_place(tmp_path):
    doc = ta.inventory(_run(tmp_path))
    assert doc["schema"] == "verity-capture/tp2-inventory/v0" and sorted(doc["ranks"]) == ["0", "1"]
    for rank in ("0", "1"):
        rk = doc["ranks"][rank]
        assert (rk["log"], rk["events"], rk["steps"], rk["roots"], rk["root_numel"]) == (f"log.rank{rank}.jsonl.gz", 21, 1, 3, 15)
        assert (rk["profile_id"], rk["tp2_notes"]) == ("profile-of-the-run", {"world_size": 2, "rank": int(rank)})
        assert (rk["dispatch_ops"], rk["distinct_ops"], rk["triton_launches"], rk["collectives_total"]) == (8, 5, 1, 6)
        assert rk["collectives_by_op"] == {REDUCE: 4, GATHER: 1, OTHER: 1}
        assert rk["collectives_per_step"] == {"0": {REDUCE: 4, GATHER: 1, OTHER: 1}}
        assert rk["collectives_by_position"] == {
            "layer i: after o_proj (attention output, row-parallel)": 1, "layer i: after down_proj (MLP output, row-parallel)": 1,
            "embedding: sum of the vocab-sharded lookups": 1, "?": 1, "logits: gather of the vocab-sharded lm_head output": 1, "x.y": 1}
        assert rk["triton_histogram"] == {"pkg.kernels.add_kernel": 1}
        rows = {row["id"]: row for row in rk["collectives"]}
        assert [rows[i]["ordinal"] for i in (REDUCE_OK, REDUCE_WRONG, REDUCE_BARE, REDUCE_LOST, GATHER_OK, BROADCAST)] == [0, 1, 2, 3, 0, 0]
        assert rows[REDUCE_OK]["children"] == ["aten.add.Tensor", "triton:add_kernel"]
        assert rows[REDUCE_OK]["arg0"] == {"alloc": 1, "offset": 0, "shape": [2, 3], "strides": [3, 1], "dtype": "bfloat16", "numel": 6}
        assert rows[REDUCE_OK]["out"]["offset"] == 6 and not rows[REDUCE_OK]["out_variant"]
        # a collective whose exit the log does not hold has no result view
        assert rows[REDUCE_BARE]["out"] is None and rows[REDUCE_BARE]["out_variant"]
        assert rows[GATHER_OK]["scalar_args"] == [-1] and rows[GATHER_OK]["phase"] == "sample"


def test_a_directory_with_no_rank_log_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="no log.rank<r>.jsonl.gz under"):
        ta.inventory(tmp_path)


def test_view_array_reads_a_strided_view_in_logical_order():
    stored = np.array([0, 1, 2, 3, 4, 5], np.uint16)
    turned = ViewDesc(1, 0, (3, 2), (1, 3), "bfloat16", "cuda:0")
    assert ta.view_array(stored.tobytes(), {"view": turned.to_json(), "byte_offset": 0}).tolist() == [[0, 3], [1, 4], [2, 5]]
    # the spanned bytes start at the view's first element, wherever that is in the storage
    late = ViewDesc(1, 4, (2,), (1,), "int32", "cuda:0")
    assert ta.view_array(np.array([7, 9], np.int32).tobytes(), {"view": late.to_json(), "byte_offset": 16}).tolist() == [7, 9]


def test_a_rank_with_no_index_and_an_entry_with_no_file_read_as_nothing(tmp_path):
    run = _run(tmp_path)
    assert ta.load_index(run, 2) == []
    index = ta.load_index(run, 1)
    kept, lost = index[1], index[4]
    assert ta.entry_bytes(run, 1, kept) == np.array(SUMMED, np.uint16).tobytes()
    assert ta.entry_bytes(run, 1, lost) is None and ta.entry_bytes(run, 1, {"skipped": "oversize"}) is None


def test_an_index_of_a_log_with_no_rank_in_its_name_is_refused(tmp_path):
    with pytest.raises(AssertionError, match="log.jsonl.gz"):
        ta.collective_entries(tmp_path, {"log": "log.jsonl.gz", "collectives": []})


def test_body_report_checks_each_collective_against_its_arithmetic(tmp_path):
    run = _run(tmp_path)
    doc = ta.body_report(run)
    assert doc["schema"] == "verity-capture/tp2-body/v0" and doc["ranks"] == [0, 1]
    assert doc["summary"] == {"body_ok": 2, "body_mismatch": 1, "unchecked": 1, "unmodelled": 2}
    by = {(c["op"], c["ordinal"]): c for c in doc["checked"]}
    good = by[REDUCE, 0]
    assert good["status"] == "ok" and good["body"] == "bf16(sum_r f32(shard_r)) round-to-nearest-even"
    assert good["body_matches_rank_outs"] == {"0": True, "1": True} and good["out_sha_equal_across_ranks"]
    assert (good["dtype"], good["shape_in"], good["shape_out"], good["module"]) == ("bfloat16", [2, 3], [2, 3], "model.layers.0.self_attn.o_proj")
    assert good["bf16_chain_equals_f32_sum"] and good["mismatching_elements_vs_rank0_out"] == 0 and not good["shards_identical"]
    # rank 1's result differs in one element: the body holds on rank 0 alone, and the two results' digests differ
    bad = by[REDUCE, 1]
    assert bad["status"] == "ok" and bad["body_matches_rank_outs"] == {"0": True, "1": False} and not bad["out_sha_equal_across_ranks"]
    assert by[REDUCE, 2]["status"] == "no snapshot bytes (root alloc, oversize, or non-snapshot step)"
    # the index names a file that rank 1's directory does not hold: an error of that collective, and the report goes on
    assert by[REDUCE, 3]["status"] == """error: TypeError("a bytes-like object is required, not 'NoneType'")"""
    assert "body" not in by[REDUCE, 3]
    gathered = by[GATHER, 0]
    assert gathered["status"] == "ok" and gathered["body"] == "concat(shard_0..shard_1, dim=-1)"
    assert gathered["body_matches_rank_outs"] == {"0": True, "1": True} and gathered["shape_out"] == [2, 4]
    assert by[OTHER, 0]["status"] == "ok" and by[OTHER, 0]["body"] == "not modelled here"


def test_determinism_report_compares_two_runs_snapshot_by_snapshot(tmp_path):
    a, b = _run(tmp_path / "a"), _run(tmp_path / "b", wrong_out=0x4002)
    doc = ta.determinism_report(a, b)
    assert doc["schema"] == "verity-capture/tp2-determinism/v0" and doc["tokens_equal"] is True
    same, differs = doc["ranks"]["0"], doc["ranks"]["1"]
    assert (same["collectives_compared"], same["collectives_only_in_a"], same["collectives_only_in_b"]) == (6, 0, 0)
    assert same["collective_shas"] == {"arg0_equal": 5, "out_equal": 5, "arg0_unsnapshotted": 1, "out_unsnapshotted": 1}
    assert same["collective_mismatches"] == []
    assert same["all_snapshot_entries"] == {"a": 10, "b": 10, "common_keys": 10, "equal": 10, "differ": 0, "differ_per_step": {}}
    # rank 1's wrong element is another wrong element in the second run: one result differs, and it is named
    assert differs["collective_shas"] == {"arg0_equal": 5, "out_equal": 4, "out_differ": 1, "arg0_unsnapshotted": 1, "out_unsnapshotted": 1}
    assert [(m["key"], m["part"]) for m in differs["collective_mismatches"]] == [([0, REDUCE, 1], "out")]
    assert differs["all_snapshot_entries"] == {"a": 10, "b": 10, "common_keys": 10, "equal": 9, "differ": 1, "differ_per_step": {"0": 1}}
    assert (differs["events_a"], differs["dispatch_ops_b"]) == (21, 8)
    (b / "tokens.json").write_text(json.dumps({"requests": [{"generated_token_ids": [5, 7]}]}))
    assert ta.determinism_report(a, b)["tokens_equal"] is False
    (b / "tokens.json").unlink()
    assert "tokens_equal" not in ta.determinism_report(a, b)


def test_roots_report_classes_each_ranks_roots_against_the_one_rank_log(tmp_path):
    run = _run(tmp_path / "run")
    with LogWriter(tmp_path / "tp1.jsonl.gz") as log:
        log.write(_header(0))
        log.write(Root(_view(90, (4, 4)), "weight", "model.layers.0.self_attn.o_proj.weight"))
        log.write(Root(_view(91, (4,)), "weight", "model.norm.weight"))
        log.write(Root(_view(93, (2,)), "buffer", "model.only_tp1"))
        log.write(_step())
        log.write(Root(_view(94, (9,)), "const", "after the first step"))
    doc = ta.roots_report(run, tmp_path / "tp1.jsonl.gz")
    assert (doc["schema"], doc["tp1_roots"], doc["tp1_root_numel"]) == ("verity-capture/tp2-roots/v0", 3, 22)
    for rank in ("0", "1"):
        rk = doc["ranks"][rank]
        assert (rk["roots"], rk["root_numel"], rk["numel_ratio_vs_tp1"]) == (3, 15, 0.6818)
        assert rk["kinds"] == {"dim1": 1, "replicated": 1, "not in TP=1 log": 1}
        assert (rk["missing_vs_tp1"], rk["extra_vs_tp1"]) == (["model.only_tp1"], ["model.only_rank"])
        assert {row["name"]: row["kind"] for row in rk["rows"]} == {
            "model.layers.0.self_attn.o_proj.weight": "dim1: contiguous input-feature slice (RowParallelLinear)",
            "model.norm.weight": "replicated", "model.only_rank": "not in TP=1 log"}
        assert [row["name"] for row in rk["layer0_and_globals"]] == [row["name"] for row in rk["rows"]]


def test_unsupported_report_counts_the_ops_a_resolution_could_not_place(tmp_path):
    res = tmp_path / "res.json"
    res.write_text(json.dumps({
        "totals": {"events": 9}, "notes": [f"note {i}" for i in range(12)], "roots": 3,
        "unsupported_examples": [{"op": "aten.foo", "reason": "no rule"}, {"op": "aten.foo"}, {"op": "aten.bar", "reason": "no rule"}],
        "op_histogram": {"aten.foo": {"unsupported": 2, "Compute": 1}, "aten.bar": {"UnsupportedShape": 3, "Compute": 4},
                         "aten.fine": {"Compute": 5}, "aten.count": 4},
        "steps": [{"step": 0, "phase": "prefill", "rows": 1, "unsupported": 5, "not kept": 1}]}))
    doc = ta.unsupported_report([res])
    assert doc["schema"] == "verity-capture/tp2-unsupported/v0" and list(doc["resolutions"]) == [str(res)]
    got = doc["resolutions"][str(res)]
    assert got["unsupported_ops"] == {"aten.bar": 3, "aten.foo": 2}
    assert got["reasons"] == {"aten.foo": {"no rule": 1, "": 1}, "aten.bar": {"no rule": 1}}
    assert (got["totals"], got["roots"], got["header_mismatches"], len(got["notes"]), len(got["examples"])) == ({"events": 9}, 3, None, 10, 3)
    assert got["steps"] == [{"step": 0, "phase": "prefill", "rows": 1, "roots": None, "events": None, "computes": None, "transparent": None,
                             "unsupported": 5, "error": None}]


def test_each_command_writes_its_report_where_it_says(tmp_path, capsys):
    a, b = _run(tmp_path / "a"), _run(tmp_path / "b")
    with LogWriter(tmp_path / "tp1.jsonl.gz") as log:
        log.write(_header(0))
    res = tmp_path / "res.json"
    res.write_text(json.dumps({"totals": {"events": 1}}))
    wrote = {"inventory": (["inventory", str(a)], a / "tp2_inventory.json"),
             "roots": (["roots", str(a), "--tp1-log", str(tmp_path / "tp1.jsonl.gz")], a / "tp2_roots.json"),
             "body": (["body", str(a)], a / "tp2_body.json"),
             "determinism": (["determinism", str(a), str(b)], a / "tp2_determinism.json"),
             "unsupported": (["unsupported", str(res)], tmp_path / "tp2_unsupported.json")}
    for name, (argv, path) in wrote.items():
        assert ta.main(argv) == 0
        assert capsys.readouterr().out.rstrip().endswith(f"-> {path}")
        assert json.loads(path.read_text())["schema"] == f"verity-capture/tp2-{name}/v0"
    # --out names another place, and its directory is made
    assert ta.main(["body", str(a), "--out", str(tmp_path / "kept" / "body.json")]) == 0
    # the summary counts in the order the collectives sort: the broadcast first, then the gather, then the four all-reduces
    assert capsys.readouterr().out.splitlines()[0] == "body: {'unmodelled': 2, 'body_ok': 2, 'body_mismatch': 1, 'unchecked': 1}"
    assert json.loads((tmp_path / "kept" / "body.json").read_text())["summary"]["body_ok"] == 2
    with pytest.raises(SystemExit):
        ta.main(["roots", str(a)])
