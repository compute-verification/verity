"""The ladder's registration, sample, compile detection and summary, on synthetic manifests and model directories."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import ladder as L


def manifest(tmp: Path, **paths: str) -> Path:
    rows = [{"role": role, "repo": f"org/{role.lower()}", "revision": "a" * 40, "local_path": p} for role, p in paths.items()]
    rows.append({"role": "QWEN3_8B", "repo": "Qwen/Qwen3-8B", "revision": "b" * 40})
    f = tmp / "checkpoints.json"
    f.write_text(json.dumps({"checkpoints": rows}))
    return f


def test_register_adds_the_rung_to_serve_tax_at_its_width(tmp_path):
    m = manifest(tmp_path, LLAMA31_70B="/ck/70b", OLMOE="/ck/olmoe")
    assert L.register("llama31-70b", m) == Path("/ck/70b")
    model = L.S.MODELS["llama31-70b"]
    assert (model.repo, model.served) == ("org/llama31_70b", "llama-3.1-70b")
    assert model.args == ("--dtype", "bfloat16", "--tensor-parallel-size", "2")
    L.register("olmoe-1b-7b", m)
    assert L.S.MODELS["olmoe-1b-7b"].args == ("--dtype", "bfloat16")


def test_a_role_without_a_local_copy_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="QWEN3_8B"):
        L.entry("qwen3-8b", manifest(tmp_path))


def test_every_rung_names_a_role_the_integration_pins():
    roles = {e["role"] for e in json.loads(L.CHECKPOINTS.read_text())["checkpoints"]}
    assert {r.role for r in L.LADDER.values()} <= roles


class Tok:
    def __init__(self, bos, eos, blank):
        self.bos_token_id, self.eos_token_id, self.blank = bos, eos, blank

    def encode(self, text, add_special_tokens=True):
        assert text == "\n\n" and not add_special_tokens
        return self.blank


def test_the_plan_starts_documents_with_the_checkpoint_s_own_tokens():
    assert L.native_tokens(Tok(128000, 128001, [271])) == (128000, 271)
    assert L.native_tokens(Tok(None, 50279, [187])) == (50279, 187)
    with pytest.raises(SystemExit, match="blank line"):
        L.native_tokens(Tok(1, 2, [13, 13]))


def test_the_alone_sample_is_spread_over_the_window():
    reqs = [{"before_step": s, "engine_id": f"r{s}"} for s in reversed(range(10))]
    assert [r["engine_id"] for r in L.sample(reqs, 3)] == ["r0", "r3", "r6"]
    assert len(L.sample(reqs, 64)) == 10


def test_compile_mode_reads_vllm_s_log(tmp_path):
    fresh, load, none = tmp_path / "a.log", tmp_path / "b.log", tmp_path / "c.log"
    fresh.write_text("INFO Compiling a graph for compile range (1, 8192) takes 12.1 s\n")
    load.write_text("INFO Directly load AOT compilation from path /x\n")
    none.write_text("INFO nothing\n")
    assert [L.compile_mode(p) for p in (fresh, load, none, tmp_path / "missing.log")] == ["compile", "load", None, None]


class Config:
    def __init__(self, r0: int):
        self.kwargs, self.num_warps, self.num_stages = {"XBLOCK": 2, "R0_BLOCK": r0}, 8, 1


class Launcher:
    def __init__(self, r0: int):
        self.config = Config(r0)


class Autotuner:
    """Inductor's CachingAutotuner as far as the tiling log reads it: a kernel with two candidate tilings keeps one."""

    def __init__(self, keep: int):
        self.launchers, self.keep = [Launcher(1024), Launcher(512)], keep
        self.inductor_meta, self.fn = {"kernel_name": "triton_red_fused_add_rms_norm_1"}, None

    def autotune_to_one_config(self, *args, **kwargs):
        self.launchers = [x for x in self.launchers if x.config.kwargs["R0_BLOCK"] == self.keep]


def test_the_tiling_log_records_each_runtime_choice(tmp_path, monkeypatch):
    class Tuner(Autotuner):
        pass

    (tmp_path / "replay-1").mkdir()
    monkeypatch.setenv(L.TILINGS_ENV, str(tmp_path / "replay-1"))
    L.patch_autotuner(Tuner)
    L.patch_autotuner(Tuner)
    Tuner(512).autotune_to_one_config()
    recs = [json.loads(x) for f in (tmp_path / "replay-1").glob("tilings.*.jsonl") for x in f.read_text().splitlines()]
    assert recs == [{"kernel": "triton_red_fused_add_rms_norm_1", "kept": {"XBLOCK": 2, "R0_BLOCK": 512, "num_warps": 8,
                                                                          "num_stages": 1},
                     "candidates": [{"XBLOCK": 2, "R0_BLOCK": r, "num_warps": 8, "num_stages": 1} for r in (1024, 512)]}]


def tiling_log(d: Path, *kept: int) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for pid, r0 in enumerate(kept):
        (d / f"tilings.{pid}.jsonl").write_text(json.dumps({"kernel": "k", "candidates": [], "kept": {"R0_BLOCK": r0}}) + "\n")


def test_replays_that_kept_different_tilings_get_different_digests(tmp_path):
    tiling_log(tmp_path / "a", 512)
    tiling_log(tmp_path / "b", 512)
    tiling_log(tmp_path / "c", 1024)
    tiling_log(tmp_path / "tp", 1024, 512)
    a, b, c, tp = (L.tilings(tmp_path / n) for n in ("a", "b", "c", "tp"))
    assert a == b and a["digest"] != c["digest"] and (a["processes"], a["choices"]) == (1, 1)
    assert tp["processes"] == 2 and L.tilings(tmp_path / "none") is None


def replay_json(d: Path, differing: int, against: int | None = None) -> None:
    d.mkdir(parents=True)
    r = {"run_id": "r1", "requests": 4, "steps_served": 10, "rows_served": 40, "steps_differing": 0, "rows_differing": 0,
         "tokens_served": 40, "requests_tokens_differing": min(differing, 1), "token_positions_differing": differing,
         "bit_exact": differing == 0, "replay_over_served": 1.01}
    if against is not None:
        r["against"] = {"requests_tokens_differing": against}
    (d / "replay.json").write_text(json.dumps(r))


def test_the_pin_is_read_from_the_environment(monkeypatch):
    monkeypatch.delenv("TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK", raising=False)
    assert not L.pinned()
    monkeypatch.setenv("TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK", "0")
    assert L.pinned()


def test_summary_reads_a_model_directory(tmp_path):
    d = tmp_path / "out" / "pinned" / "qwen3-8b"
    d.mkdir(parents=True)
    (d / "serve.json").write_text(json.dumps({"model": "qwen3-8b", "checkpoint": "Qwen/Qwen3-8B@b", "tp": 1, "run_id": "r1",
                                              "rate": 87.0, "inductor_pin": True, "capacity": {"req_per_s": 60.0},
                                              "window": {"sent": 4, "ok": 4}}))
    replay_json(d / "replay-1", 0)
    replay_json(d / "replay-2", 4, against=1)
    tiling_log(d / "replay-1", 512)
    tiling_log(d / "replay-2", 1024)
    (d / "replay-2.log").write_text("Compiling a graph for compile range\n")
    (d / "alone").mkdir()
    (d / "alone" / "alone.json").write_text(json.dumps({"run_id": "r1", "requests": 2, "tokens_served": 20,
                                                        "requests_tokens_differing": 1, "token_positions_differing": 5,
                                                        "against_replay": {"requests_tokens_differing": 1}}))
    row = L.model_row(d)
    assert (row["model"], row["pin"]) == ("qwen3-8b", True)
    assert [r["bit_exact"] for r in row["replays"]] == [True, False]
    assert row["replays"][1]["rows_exact_share"] == pytest.approx(0.9)
    assert row["replays"][1]["compile"] == "compile"
    assert row["alone"]["requests_tokens_differing"] == 1
    md = L.markdown([row])
    assert row["tilings"] == {"replays": 2, "distinct": 2, "choices": 1}
    assert "| qwen3-8b | 1 | yes | 4, 10, 40 | 1 of 2 | 90.000000% | 0, 1 (fresh compile) | 1 | 2 of 2 (1) | 1 of 2 |" in md
