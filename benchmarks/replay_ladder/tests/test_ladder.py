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


def replay_json(d: Path, differing: int, against: int | None = None) -> None:
    d.mkdir(parents=True)
    r = {"run_id": "r1", "requests": 4, "steps_served": 10, "rows_served": 40, "steps_differing": 0, "rows_differing": 0,
         "tokens_served": 40, "requests_tokens_differing": min(differing, 1), "token_positions_differing": differing,
         "bit_exact": differing == 0, "replay_over_served": 1.01}
    if against is not None:
        r["against"] = {"requests_tokens_differing": against}
    (d / "replay.json").write_text(json.dumps(r))


def test_summary_reads_a_model_directory(tmp_path):
    d = tmp_path / "out" / "qwen3-8b"
    d.mkdir(parents=True)
    (d / "serve.json").write_text(json.dumps({"checkpoint": "Qwen/Qwen3-8B@b", "tp": 1, "run_id": "r1", "rate": 87.0,
                                              "capacity": {"req_per_s": 60.0}, "window": {"sent": 4, "ok": 4}}))
    replay_json(d / "replay-1", 0)
    replay_json(d / "replay-2", 4, against=1)
    (d / "replay-2.log").write_text("Compiling a graph for compile range\n")
    (d / "alone").mkdir()
    (d / "alone" / "alone.json").write_text(json.dumps({"run_id": "r1", "requests": 2, "tokens_served": 20,
                                                        "requests_tokens_differing": 1, "token_positions_differing": 5,
                                                        "against_replay": {"requests_tokens_differing": 1}}))
    row = L.model_row(d)
    assert [r["bit_exact"] for r in row["replays"]] == [True, False]
    assert row["replays"][1]["rows_exact_share"] == pytest.approx(0.9)
    assert row["replays"][1]["compile"] == "compile"
    assert row["alone"]["requests_tokens_differing"] == 1
    md = L.markdown([row])
    assert "| qwen3-8b | 1 | 4, 10, 40 | 1 of 2 | 90.000000% | 0, 1 (fresh compile) | 1 | 1 of 2 |" in md
