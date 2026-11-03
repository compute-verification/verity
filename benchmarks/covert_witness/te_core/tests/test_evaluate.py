"""covert_witness's evaluation job (evaluate.py) on a stub prepare: what it makes of the submitted checkpoint's score, and
its tier's budget."""
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

import evaluate

BLOB = b"E"
FINAL_SCORE = {"bpb": 1.2, "start_bpb": 1.3}
SEARCH_SCORE = {"bpb": 1.25, "start_bpb": 1.3, "slice": {"coin": "c", "weights_digest": "w", "windows": 1}}


def stub_prepare(final: bool, tier: str = "search") -> SimpleNamespace:
    theta = {"w": torch.zeros(2)}
    budgets = {"search": {"tier": "search", "train_flops": 1e12, "gpu_seconds": 600.0},
               "long": {"tier": "long", "train_flops": 1e14, "gpu_seconds": 6e4, "c0_flops": 1e12, "c0_multiple": 100}}
    return SimpleNamespace(
        VERSION="3.0", DATA="v3",
        TRAIN_FLOP_BUDGET=1e12, HARNESS_FLOP_SLACK=1.1, GPU_SECONDS_CAP=600.0, FINAL_MODE=final, FINAL="final",
        SEARCH_POOL="search", DEVICE="cpu", budget=lambda name: budgets[tier],
        start_record=lambda name: {"digest": "d", "config": {"n_layer": 2}, "tokenizer": "gpt2", "parameters": 2},
        load_theta=lambda name: theta, format_of=lambda blob: "full", decode=lambda th, blob: dict(th),
        rebuild_macs=lambda th, blob: 0, same_bits=torch.equal, deep_share=lambda rec, names: {},
        model_config=lambda rec, blob: rec["config"], model_record=lambda rec, config, state: rec,
        read_steps=lambda path: [], train_flops=lambda rec, trainable, steps: {"flops": 2e6},
        held_out=lambda split, tok: ([0] * 8, {"bytes": 32}), eval_cost=lambda rec, n, nbytes: {})


def checked(tmp_path, monkeypatch, final: bool, score: dict, tier: str = "search", macs: float = 1e6,
            wall_s: float = 1.0) -> dict:
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "E.bin").write_bytes(BLOB)
    (sub / "meta.json").write_text(json.dumps({"start": "s50", "theta_digest": "d", "changed": [],
                                               "trainable": {"full": []}}))
    run = tmp_path / "run.json"
    run.write_text(json.dumps({"totals": {"macs": macs, "wall_s": wall_s, "self_cpu_s": 1.0}, "steps": [{}],
                               "start": {"digest": "d"}, "submitted": {"e_sha256": hashlib.sha256(BLOB).hexdigest()},
                               "checkpoints": [{"step": 0, "macs": macs}]}))
    tree = tmp_path / "tree.json"
    tree.write_text(json.dumps({"cpu_s": 1.0, "main_pid": 1, "gpu_apps": []}))
    timing = tmp_path / "timing.json"
    timing.write_text(json.dumps({"rc": 0}))
    island = tmp_path / "island.json"
    island.write_text(json.dumps({"name": "e100kb", "commit": "c", "digest": "i"}))

    def score_run(prepare, rec, run_dir, theta, start, device):
        rec["checkpoints"][0] |= {"submitted": True, "val_bpb": score["bpb"], "score": json.loads(json.dumps(score))}
        rec["start"]["val_bpb"] = score["start_bpb"]
        rec["scored"] = {"split": "final" if final else "search", "final_mode": final, "problems": []}
        return rec

    monkeypatch.setattr(evaluate, "prepare", stub_prepare(final, tier))
    monkeypatch.setattr(evaluate, "score_run", score_run)
    monkeypatch.setattr(evaluate, "source_clean", lambda: True)
    return evaluate.check(sub, timing, run, tree, island, tmp_path / "scored")


@pytest.mark.parametrize("final, score", [(True, FINAL_SCORE), (False, SEARCH_SCORE)])
def test_the_submitted_score_is_recorded_with_or_without_a_search_slice(tmp_path, monkeypatch, final, score):
    out = checked(tmp_path, monkeypatch, final, score)
    assert out["valid"], out["checks"]
    assert not {"e_size", "e_format", "shallow", "class_named"} & set(out["checks"]) and "caps" not in out
    assert out["score"]["bpb"] == score["bpb"] and ("slice" in out["score"]) == (not final)
    if not final:
        assert out["score"]["slice"]["digest"] == "te_core.run.theta_sha256"
    scored = json.loads((tmp_path / "scored" / "run.json").read_text())
    assert scored["checkpoints"][0]["score"]["bpb"] == score["bpb"]


@pytest.mark.parametrize("by", ["01:20Z coordinator", None])
def test_a_long_run_has_its_tiers_budget_and_is_valid_only_when_a_board_line_assigns_it(tmp_path, monkeypatch, by):
    if by:
        monkeypatch.setenv("CW_TIER_BY", by)
    else:
        monkeypatch.delenv("CW_TIER_BY", raising=False)
    # 100x the search tier's FLOPs and seconds: over the search budget, within the long one
    out = checked(tmp_path, monkeypatch, False, SEARCH_SCORE, tier="long", macs=4e13, wall_s=3e4)
    assert out["budget"]["tier"] == "long" and out["budget"]["assigned_by"] == by
    assert out["checks"]["flop_budget"] is True and out["checks"]["gpu_seconds"] is True
    assert out["checks"]["tier_assigned"] is bool(by) and out["valid"] is bool(by)
