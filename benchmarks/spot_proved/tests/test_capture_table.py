"""`capture.py --table`: at a layer with its own indexer, each request's last token past ``table_keys`` keys gets its history
table (its latent rows below n as the paged cache holds them, the step's own included), the scores its top-k call ranked,
the top-k the slot conversion maps as the canonical top-k left it, and the rows attention read; and `table.py` holds them
together. vLLM's kernels are stubs: a decode row ranked by a top-k over lengths, a prefill chunk's last row by a top-k over
[rowStarts, rowEnds), and a short row that keeps every key."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

HERE = Path(__file__).resolve().parents[1]
ATTN = "vllm.models.deepseek_v32.attention"
FAMS = "vllm.v1.attention.backends.mla"
FA = f"{FAMS}.flashattn_mla_sparse"
KVL, ROPE, BS, K, LAYER = 8, 4, 4, 4, 2
IDS, COUNTS = ["d", "s", "p"], {"d": 1, "s": 1, "p": 3}
POS = [6, 2, 6, 7, 8]
TABLE = [[5, 0, 2], [1, 7, 7], [3, 4, 6]]


def fused_norm_rope(positions, q_c, q_rms_norm_w, q_rms_eps, kv_c, kv_rms_norm_w, kv_rms_eps, k_pe, k_rope_cos_sin_cache,
                    index_k, index_k_layer_norm_w, index_k_layer_norm_bias, index_k_layer_norm_eps,
                    index_k_rope_cos_sin_cache, topk_indices_buffer, slot_mapping=None, mla_kv_cache=None, kv_c_out=None,
                    k_pe_out=None, **kw):
    kv_c_out.copy_(kv_c)
    k_pe_out.copy_(k_pe)
    mla_kv_cache.view(-1, KVL + ROPE)[slot_mapping] = torch.cat([kv_c_out, k_pe_out], dim=-1)
    return q_c


def convert(req_id, block_table, token_indices, BLOCK_SIZE=64, NUM_TOPK_TOKENS=2048, **kw):
    blk = block_table[req_id.long()[:, None], (token_indices // BLOCK_SIZE).clamp(min=0).long()]
    return torch.where(token_indices >= 0, blk * BLOCK_SIZE + token_indices % BLOCK_SIZE, -1).int()


def varlen(q=None, k=None, v=None, block_table=None, **kw):
    return torch.zeros(block_table.shape[0], 1)


def _stub(monkeypatch) -> None:
    names = ["vllm", "vllm.models", "vllm.models.deepseek_v32", ATTN, "vllm.v1", "vllm.v1.attention", "vllm.v1.attention.backends",
             FAMS, f"{FAMS}.flashmla_sparse", FA]
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules[ATTN].fused_norm_rope = fused_norm_rope
    sys.modules[FA].triton_convert_req_index_to_global_index = convert
    sys.modules[FA].flash_attn_varlen_func = varlen
    sys.modules[f"{FAMS}.flashmla_sparse"].FlashMLASparseImpl = type("FlashMLASparseImpl", (), {})


def _module(monkeypatch):
    spec = importlib.util.spec_from_file_location("spot_capture_table", HERE / "capture.py")
    capture = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, capture)
    spec.loader.exec_module(capture)
    return capture


def _capture(tmp_path, monkeypatch):
    capture = _module(monkeypatch)
    tmp_path.chmod(0o700)
    S = capture._State({"capture": str(tmp_path), "steps": [3], "mla_tokens": 0, "open": {}, "keep": "none",
                        "table": [LAYER], "table_keys": K})
    monkeypatch.setattr(capture, "S", S)
    capture._install_py_kernels()
    capture._install_mla()
    S.recording, S.layer, S.step = True, LAYER, 3
    S.input_batch, S.sched = types.SimpleNamespace(req_ids=IDS), types.SimpleNamespace(num_scheduled_tokens=COUNTS)
    return capture, S


def _table():
    spec = importlib.util.spec_from_file_location("spot_table", HERE / "table.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _schema(*names):
    return types.SimpleNamespace(arguments=[types.SimpleNamespace(name=n, alias_info=None) for n in names])


def _canonical(scores, n):
    return sorted(sorted(range(n), key=lambda j: (-scores[j], j))[:K]) if n > K else list(range(n)) + [-1] * (K - n)


def _step(capture, *, copied: bool = False, other_request: bool = False, dtype: str = "auto", logits=torch.float32):
    """One layer of the step: its cache write, a decode top-k over lengths, a prefill top-k over row bounds, the slot
    conversion and FA3's read. Returns the cache and what each kept row's scores and top-k must be."""
    g = torch.Generator().manual_seed(3)
    cache = torch.randn(8, BS, KVL + ROPE, generator=g).bfloat16()
    pos = torch.tensor(POS)
    req = torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32)
    table = torch.tensor(TABLE, dtype=torch.int32)
    slots = table[req.long(), pos // BS] * BS + pos % BS
    buf = torch.full((8, K), -7, dtype=torch.int32)
    kv_c, k_pe = torch.randn(5, KVL, generator=g).bfloat16(), torch.randn(5, ROPE, generator=g).bfloat16()
    sys.modules[ATTN].fused_norm_rope(pos, torch.zeros(5, 2), None, 1e-6, kv_c, None, 1e-6, k_pe, None, None, None, None, 1e-6,
                                      None, buf, slot_mapping=slots, mla_kv_cache=cache, mla_kv_cache_dtype=dtype,
                                      kv_c_out=torch.empty_like(kv_c), k_pe_out=torch.empty_like(k_pe))

    dec = torch.randn(2, 16, generator=g).to(logits)
    lengths = torch.tensor([[7], [3]], dtype=torch.int32)
    dec_out = torch.empty(2, K, dtype=torch.int32) if copied else buf[0:2]
    want = {0: (dec[0, :7], _canonical(dec[0].tolist(), 7))}

    def run_dec(seq):
        for r, n in enumerate((7, 3)):
            dec_out[r] = torch.tensor(sorted(range(n), key=lambda j: -dec[r, j].item())[:K] + [-1] * (K - min(n, K)))
    capture._record_op("_C::cooperative_topk", _schema("logits", "lengths", "output", "workspace", "k", "max_seq_len"),
                       (dec, lengths, dec_out, None, K, 16), {}, run_dec)
    for r, n in enumerate((7, 3)):
        dec_out[r] = torch.tensor(_canonical(dec[r].tolist(), n), dtype=torch.int32)
    if copied:
        buf[0:2] = dec_out

    pre = torch.randn(3, 20, generator=g).to(logits)
    starts, ends = torch.tensor([2, 2, 2], dtype=torch.int32), torch.tensor([9, 10, 11], dtype=torch.int32)
    want[4] = (pre[2, 2:11], _canonical(pre[2, 2:11].tolist(), 9))

    def run_pre(seq):
        for r in range(3):
            row = pre[r, 2:int(ends[r])].tolist()
            buf[2 + r] = torch.tensor(sorted(range(len(row)), key=lambda j: -row[j])[:K])
    capture._record_op("_C::top_k_per_row_prefill", _schema("logits", "rowStarts", "rowEnds", "indices", "numRows", "stride0",
                                                            "stride1", "topK"),
                       (pre, starts, ends, buf[2:5], 3, 20, 1, K), {}, run_pre)
    for r in range(3):
        buf[2 + r] = torch.tensor(_canonical(pre[r, 2:int(ends[r])].tolist(), int(ends[r]) - 2), dtype=torch.int32)

    conv_req = torch.tensor([0, 1, 2, 2, 0], dtype=torch.int32) if other_request else req
    page = sys.modules[FA].triton_convert_req_index_to_global_index(conv_req, table, buf[:5], BLOCK_SIZE=BS, NUM_TOPK_TOKENS=K)
    flat = cache.view(-1, KVL + ROPE)
    sys.modules[FA].flash_attn_varlen_func(q=torch.zeros(5, 1), k=flat[:, KVL:].unsqueeze(1).unsqueeze(1),
                                           v=flat[:, :KVL].unsqueeze(1).unsqueeze(1), block_table=page,
                                           seqused_k=torch.tensor([4, 3, 4, 4, 4], dtype=torch.int32))
    return cache, want


def _u16(t):
    return t.contiguous().view(torch.int16).numpy().view(np.uint16)


def _finish(capture, S, tmp_path) -> Path:
    d = S.root() / f"step{S.step}"
    S.written = capture._write_tables(d)
    capture._drain()
    return d / "table"


@pytest.mark.parametrize("copied", [False, True])
def test_each_long_rows_table_scores_and_canonical_top_k_are_written_and_join(tmp_path, monkeypatch, copied) -> None:
    _stub(monkeypatch)
    capture, S = _capture(tmp_path, monkeypatch)
    cache, want = _step(capture, copied=copied)
    assert S.errors == []
    td = _finish(capture, S, tmp_path)
    doc = json.loads((td / "table.json").read_text())
    assert [(e["request"], e["token"], e["n"], e["op"], e["by"]) for e in doc["tables"]] == [
        ("d", 0, 7, "_C::cooperative_topk", "keys" if copied else "view"), ("p", 4, 9, "_C::top_k_per_row_prefill", "view")]
    for k, e in enumerate(doc["tables"]):
        r, n = IDS.index(e["request"]), e["n"]
        rows = torch.stack([cache[TABLE[r][p // BS], p % BS] for p in range(n)])
        assert np.array_equal(np.fromfile(td / f"{k}.kv", "<u2").reshape(n, -1), _u16(rows))
        scores, topk = want[e["token"]]
        assert np.array_equal(np.fromfile(td / f"{k}.scores", "<f4"), scores.numpy())
        assert np.fromfile(td / f"{k}.topk", "<i4").tolist() == topk
        assert np.array_equal(np.fromfile(td / f"{k}.read", "<u2").reshape(K, -1), _u16(rows[topk]))

    res = _table().check(tmp_path, {"request": "p", "steps": [{"output": 0, "n": 9, "step": 3}]})
    assert res["ok"] and res["tables"] == 2 and res["failed"] == res["missing"] == []
    import hashlib
    assert S.written == {"layers": [LAYER], "keys": K, "rows": 2,
                         "sha512": hashlib.sha512((td / "table.json").read_bytes()).hexdigest()}
    assert td.stat().st_mode & 0o777 == 0o700
    assert {f.name: f.stat().st_mode & 0o777 for f in td.iterdir()} == {
        f"{k}.{x}": 0o600 for k in (0, 1) for x in ("kv", "scores", "topk", "read")} | {"table.json": 0o600}


def test_a_top_k_over_bf16_logits_writes_no_table(tmp_path, monkeypatch) -> None:
    _stub(monkeypatch)
    capture, S = _capture(tmp_path, monkeypatch)
    _step(capture, logits=torch.bfloat16)
    assert S.tables == [] and len(S.errors) == 2
    assert all(e.endswith("ranks torch.bfloat16 logits, not the f32 scores DsaTopKMaskCounted ranks") for e in S.errors)


@pytest.mark.parametrize("where", ["open", "run"])
def test_a_capture_directory_others_can_read_or_inside_a_run_directory_gets_no_table(tmp_path, monkeypatch, where) -> None:
    _stub(monkeypatch)
    capture, S = _capture(tmp_path, monkeypatch)
    _step(capture)
    if where == "open":
        tmp_path.chmod(0o755)
    else:
        (tmp_path / "job.json").write_text("{}")
    td = _finish(capture, S, tmp_path)
    why = "mode 755, not 0700" if where == "open" else f"inside the run directory {tmp_path.resolve()}"
    assert not td.exists() and S.tables == [] and S.written == {"layers": [LAYER], "keys": K, "rows": 0, "refused": why}
    assert S.errors == [f"table: no table written, the capture directory is {why}"]


def test_site_with_tables_refuses_a_run_directory_and_makes_its_directories_private(tmp_path, monkeypatch) -> None:
    capture = _module(monkeypatch)
    run = tmp_path / "run"
    run.mkdir()
    (run / "remote.json").write_text("{}")
    with pytest.raises(SystemExit, match="lies inside the run directory"):
        capture.main(["site", str(tmp_path / "site"), "--capture", str(run / "out" / "cap"), "--steps", "3", "--table", "2"])
    steps = tmp_path / "steps.json"
    steps.write_text(json.dumps({"request": "x", "prompt": 9, "steps": [{"output": 0, "n": 9, "step": 3},
                                                                        {"output": 1, "n": 10, "step": 4}]}))
    priv = tmp_path / "priv"
    capture.main(["site", str(priv / "site"), "--capture", str(priv / "cap"), "--steps-from", str(steps), "--table", "2"])
    assert [(priv / x).stat().st_mode & 0o777 for x in ("site", "cap")] == [0o700, 0o700]
    assert json.loads((priv / "site" / capture.CONFIG).read_text())["steps"] == [3, 4]


def test_table_py_writes_its_result_privately_and_prints_counts_only(tmp_path, monkeypatch, capsys) -> None:
    _written(tmp_path, monkeypatch)
    table, out = _table(), tmp_path / "check.json"
    assert table.main(["check", str(tmp_path), "--out", str(out)]) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "tables": 2, "ranks": 1, "failed": 0, "missing": 0}
    assert out.stat().st_mode & 0o777 == 0o600 and json.loads(out.read_text())["rows"][0][2] == "d"
    run = tmp_path / "run"
    run.mkdir()
    (run / "job.json").write_text("{}")
    with pytest.raises(SystemExit, match="lies inside the run directory"):
        table.main(["check", str(tmp_path), "--out", str(run / "check.json")])
    assert not (run / "check.json").exists()
    with pytest.raises(SystemExit):
        table.main(["check", str(tmp_path)])


def test_a_conversion_that_takes_a_kept_token_as_another_request_writes_no_table_and_says_so(tmp_path, monkeypatch) -> None:
    _stub(monkeypatch)
    capture, S = _capture(tmp_path, monkeypatch)
    _step(capture, other_request=True)
    assert [e["request"] for e in S.tables] == ["d"]
    assert S.errors == ["step 3 table layer 2 request p token 4: the conversion's token 4 is not of batch index 2"]


def test_a_cache_written_as_fp8_writes_no_table(tmp_path, monkeypatch) -> None:
    _stub(monkeypatch)
    capture, S = _capture(tmp_path, monkeypatch)
    _step(capture, dtype="fp8_ds_mla")
    assert S.tables == [] and len(S.errors) == 2
    assert all(e.endswith("a torch.bfloat16 cache written as fp8_ds_mla: a table holds bf16 rows only") for e in S.errors)


def _rewrite(td: Path, k: int, name: str, fn) -> None:
    doc = json.loads((td / "table.json").read_text())
    f = doc["tables"][k]["files"][name]
    a = fn(np.fromfile(td / f"{k}.{name}", {"kv": "<u2", "read": "<u2", "scores": "<f4", "topk": "<i4"}[name]).copy())
    (td / f"{k}.{name}").write_bytes(a.tobytes())
    import hashlib
    f["sha512"] = hashlib.sha512(a.tobytes()).hexdigest()
    (td / "table.json").write_text(json.dumps(doc))


def _written(tmp_path, monkeypatch) -> Path:
    _stub(monkeypatch)
    capture, S = _capture(tmp_path, monkeypatch)
    _step(capture)
    return _finish(capture, S, tmp_path)


def test_check_refuses_a_top_k_that_is_not_the_scores_canonical_one(tmp_path, monkeypatch) -> None:
    td = _written(tmp_path, monkeypatch)
    _rewrite(td, 1, "topk", lambda a: a[::-1])
    res = _table().check(tmp_path)
    assert not res["ok"] and [f["why"] for f in res["failed"]] == [["its top-k is not the canonical top-4 of its scores"]]


def test_check_refuses_scores_that_are_not_f32(tmp_path, monkeypatch) -> None:
    td = _written(tmp_path, monkeypatch)
    doc = json.loads((td / "table.json").read_text())
    doc["tables"][1]["files"]["scores"]["dtype"] = "torch.bfloat16"
    (td / "table.json").write_text(json.dumps(doc))
    _rewrite(td, 1, "scores", lambda a: (a.view("<u4") >> 16).astype("<u2"))
    res = _table().check(tmp_path)
    assert [f["why"] for f in res["failed"]] == [["1.scores is torch.bfloat16, not torch.float32"]]


def _tie(td: Path, scores: np.ndarray, pick: list[int]) -> None:
    kv = np.fromfile(td / "1.kv", "<u2").reshape(len(scores), -1)
    _rewrite(td, 1, "scores", lambda a: scores)
    _rewrite(td, 1, "topk", lambda a: np.asarray(pick, "<i4"))
    _rewrite(td, 1, "read", lambda a: kv[pick].reshape(-1))


def test_a_tie_across_the_cut_goes_to_the_smaller_positions_as_the_counted_mask_breaks_it(tmp_path, monkeypatch) -> None:
    """Five keys tie for the last three of four slots: `DsaTopKMaskCounted_v2` keeps positions 1 to 3, and the check accepts
    that top-k and refuses one keeping position 5 in place of 3."""
    from verity.circuits import evaluation as E
    from verity.circuits.boolean import trace as TR
    from verity.circuits.defs import bind
    from verity.definitions.registry import boolean_dsa_topk_mask as BM
    td = _written(tmp_path, monkeypatch)
    scores, N = np.array([2, 1, 1, 1, 1, 1, 0, 0, 0], "<f4"), 16
    row = scores.view("<u4").tolist() + [0] * (N - len(scores))
    mask = np.asarray(E.evaluate_batch(TR.word_of(bind(BM.DsaTopKMaskCountedV2, N=N, TOPK=K)), [row], [len(scores)]))
    assert np.nonzero(mask.reshape(-1))[0].tolist() == [0, 1, 2, 3]
    _tie(td, scores, [0, 1, 2, 3])
    assert _table().check(tmp_path)["ok"]
    _tie(td, scores, [0, 1, 2, 5])
    assert [f["why"] for f in _table().check(tmp_path)["failed"]] == [["its top-k is not the canonical top-4 of its scores"]]


def test_check_refuses_a_table_row_attention_did_not_read(tmp_path, monkeypatch) -> None:
    td = _written(tmp_path, monkeypatch)
    topk = np.fromfile(td / "0.topk", "<i4")

    def flip(a):
        a.reshape(7, -1)[topk[1], 0] ^= 1
        return a
    _rewrite(td, 0, "kv", flip)
    (fail,) = _table().check(tmp_path)["failed"]
    assert fail["request"] == "d" and fail["why"][0].startswith("1 rows attention read are not the table's rows")


def test_check_refuses_ranks_that_disagree_and_a_named_step_without_its_table(tmp_path, monkeypatch) -> None:
    td = _written(tmp_path, monkeypatch)
    other = tmp_path / "rank1" / "step3" / "table"
    other.mkdir(parents=True)
    for f in td.iterdir():
        (other / f.name).write_bytes(f.read_bytes())
    (other / "table.json").write_text(json.dumps({**json.loads((td / "table.json").read_text()), "rank": 1}))
    _rewrite(other, 0, "scores", lambda a: a * 2)
    res = _table().check(tmp_path, {"request": "p", "steps": [{"output": 1, "n": 10, "step": 4}]})
    assert {f["request"]: f["why"][0] for f in res["failed"]}["d"] == "ranks [0, 1] hold different tables"
    assert res["missing"] == [{"output": 1, "n": 10, "step": 4, "why": "no table"}]


def test_decode_steps_names_the_step_sampling_each_output() -> None:
    w = {"steps": [{"step": 0, "rows": [["a", 0, 4]]}, {"step": 1, "rows": [["a", 4, 2], ["b", 0, 3]]},
                   {"step": 2, "rows": [["a", 6, 1], ["b", 3, 1]]}],
         "requests": [{"engine_id": "a", "prompt_ids": [0] * 6}, {"engine_id": "b", "prompt_ids": [0] * 3}]}
    assert _table().decode_steps(w) == {"request": "a", "prompt": 6, "steps": [{"output": 0, "n": 6, "step": 1},
                                                                               {"output": 1, "n": 7, "step": 2}]}
