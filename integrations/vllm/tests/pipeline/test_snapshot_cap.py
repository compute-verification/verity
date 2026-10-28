"""run_config derives the Match capture's per-tensor snapshot cap from the workload's largest forward and the
checkpoint's widths when no --snapshot-max-bytes is given.  Torch-free.

A fixed 32 MiB cap skips the prefill-step gate_up output of every 1024-class row ([1024+B, 2I] bf16 = 33.7 MB for Llama-3.2-1B,
36.7 MB for Qwen2.5-1.5B; snapshots.json 'skipped: spanned bytes 33685504 > snapshot_max_bytes 33554432'), and the Commit's form (B)
oracle-compare then reports 'no snapshot of alloc ... (0 entries of that alloc)' for those rows (a B16 mixed-arrivals row: 16 layers
x 5 rows of the [1028, 16384] gate_up at engine step 36, the step a 1024-token prompt arrives).  256/32 rows never hit it."""
from __future__ import annotations

import json
import sys

import pytest

from verity_vllm.pipeline import match as RC
from verity_vllm.pipeline import cli

LLAMA_1B = {"H": 2048, "I": 8192, "GU": 16384, "QKV": 3072, "V": 128256}         # analytic.config_of keys
QWEN_15B = {"H": 1536, "I": 8960, "GU": 17920, "QKV": 2048, "V": 151936}
OLMOE = {"H": 2048, "I": 1024, "GU": 2048, "QKV": 4096, "V": 50304, "TOPK": 8, "NUM_EXPERTS": 64}


def _wl(prompt_lens, max_tokens=128, mnbt=None):
    doc = {"requests": [{"request_id": f"r{i}", "prompt_len": n, "prompt_token_ids": [7] * n, "max_tokens": max_tokens} for i, n in enumerate(prompt_lens)]}
    if mnbt is not None:
        doc["engine"] = {"args": {"max_num_batched_tokens": mnbt, "max_model_len": 1152}}
    return doc


def test_the_32_mib_cap_of_record_skipped_the_1024_class_prefill_gate_up_and_the_derived_cap_covers_it():
    # the record's numbers: llama int-2 B16 mixed-arrivals, the [1028, 16384] bf16 gate_up at the 1024-token arrival = 33,685,504 B
    assert 1028 * LLAMA_1B["GU"] * 2 == 33_685_504 > RC.SNAPSHOT_CAP_FLOOR == 33_554_432
    # qwen H100 B1 1024/128: [1024, 17920] bf16 = 36,700,160 B
    assert 1024 * QWEN_15B["GU"] * 2 == 36_700_160 > RC.SNAPSHOT_CAP_FLOOR
    assert 35_840 == QWEN_15B["GU"] * 2       # one row of that tensor = the committed identity's byte range
    # the derived caps: rows = the engine's max_num_batched_tokens of the workload (the bound the scheduler enforces)
    cap, why = RC.snapshot_cap_bytes(_wl([1024] * 16, mnbt=5348), LLAMA_1B)
    assert cap == 5348 * 16384 * 2 and why["raised"] and why["rows_source"] == "engine max_num_batched_tokens" and why["rows"] == 5348
    assert why["row_bytes"] == 32768 and why["row_bytes_source"] == "hf config GU=16384 x bf16"
    assert cap >= 33_685_504
    cap_q, why_q = RC.snapshot_cap_bytes(_wl([1024], mnbt=2048), QWEN_15B)
    assert cap_q == 2048 * 17920 * 2 >= 36_700_160 and why_q["raised"]
    # the 4096/512 long-context row: 4097 rows x 32 KiB = 134 MB
    cap_l, why_l = RC.snapshot_cap_bytes(_wl([4096], max_tokens=512, mnbt=4097), LLAMA_1B)
    assert cap_l == 4097 * 32768 and why_l["raised"]


def test_256_32_rows_and_moe_rows_keep_the_32_mib_floor():
    # 256/32 B8 llama: (256 + 8) x 32 KiB = 8.6 MB -> the floor stands (identical capture to the rows of record)
    cap, why = RC.snapshot_cap_bytes(_wl([256] * 8, max_tokens=32, mnbt=None), LLAMA_1B)
    assert cap == RC.SNAPSHOT_CAP_FLOOR and not why["raised"] and why["rows"] == 256 + 8 and why["rows_source"].startswith("longest prompt")
    # OLMoE B1 256/32 with the cov-moe contract (routed intermediates are covered by --snapshot-max-bytes-op, not by this width)
    cap_m, why_m = RC.snapshot_cap_bytes(_wl([256], max_tokens=32, mnbt=2048), OLMOE)
    assert cap_m == RC.SNAPSHOT_CAP_FLOOR and not why_m["raised"] and why_m["row_bytes"] == 4096 * 2      # QKV is OLMoE's widest dense row
    assert "TOPK" not in json.dumps(why_m)


def test_rows_and_widths_fall_back_by_name_and_the_cli_engine_arg_wins():
    # no HF config readable: 64 KiB per row (2 x I x bf16 for I <= 16384), the workload's own bound for rows
    cap, why = RC.snapshot_cap_bytes(_wl([1024, 700, 10]), None)
    assert why["rows"] == 1024 + 3 and why["row_bytes"] == RC.SNAPSHOT_ROW_BYTES_FALLBACK and why["row_bytes_source"].startswith("no HF config")
    assert cap == 1027 * 65536
    # an explicit CLI engine arg overrides the workload's engine.args (run_config's merge order)
    cap2, why2 = RC.snapshot_cap_bytes(_wl([1024], mnbt=2048), LLAMA_1B, {"max_num_batched_tokens": 1025})
    assert why2["rows"] == 1025 and cap2 == 1025 * 32768
    # empty / missing workload: floor
    cap3, why3 = RC.snapshot_cap_bytes(None, None)
    assert cap3 == RC.SNAPSHOT_CAP_FLOOR and why3["rows"] == 0


def test_main_derives_the_cap_only_when_none_is_given_and_records_the_derivation(tmp_path, monkeypatch):
    """run_config --dry-run with a workload and an HF config: the capture command carries the derived --snapshot-max-bytes; an explicit
    value is passed through untouched with derivation None."""
    wl = tmp_path / "wl.json"
    wl.write_text(json.dumps(_wl([1024] * 4, mnbt=2048)))
    hf = tmp_path / "config.json"
    hf.write_text(json.dumps({"model_type": "llama", "hidden_size": 2048, "num_hidden_layers": 16, "num_attention_heads": 32, "num_key_value_heads": 8,
                              "head_dim": 64, "intermediate_size": 8192, "vocab_size": 128256, "rms_norm_eps": 1e-5, "rope_theta": 500000.0,
                              "max_position_embeddings": 131072}))
    seen = {}

    class _Plan(RC.Plan):
        def __init__(self, a):
            seen["a"] = a
            raise SystemExit(0)            # stop before any stage runs; the derivation happens before Plan()

    monkeypatch.setattr(RC, "Plan", _Plan)
    monkeypatch.setattr(RC, "profile_config", lambda name: None)
    base = ["--case", "LLAMA32_1B", "--profile", "derived_LLAMA32_1B", "--workload", str(wl), "--out", str(tmp_path / "run"),
            "--hf-config", str(hf), "--dry-run"]
    with pytest.raises(SystemExit):
        cli.main(["match", *base])
    a = seen["a"]
    assert a.snapshot_max_bytes == 2048 * 16384 * 2 and a.snapshot_cap_derivation["raised"] is True
    assert a.snapshot_cap_derivation["row_bytes_source"] == "hf config GU=16384 x bf16"
    with pytest.raises(SystemExit):
        cli.main(["match", *base, "--snapshot-max-bytes", "4096"])
    a2 = seen["a"]
    assert a2.snapshot_max_bytes == 4096 and a2.snapshot_cap_derivation is None
    # a run.json re-run (--run-dir) keeps the cap the capture ran with, never a re-derived one
    run = tmp_path / "run"; run.mkdir()
    (run / RC.G.ARTIFACTS.get("run", "run.json")).write_text(json.dumps({"case": "LLAMA32_1B", "workload_path": str(wl), "profile": "derived_LLAMA32_1B",
                                                                       "config_id": "x", "snapshot_max_bytes": 777, "snapshot_cap_derivation": {"rows": 1}}))
    with pytest.raises(SystemExit):
        cli.main(["match", "--run-dir", str(run), "--phase", "cpu", "--dry-run"])
    a3 = seen["a"]
    assert a3.snapshot_max_bytes == 777 and a3.snapshot_cap_derivation == {"rows": 1}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
