"""The ladder's registration, sample, compile detection and summary, on synthetic manifests and model directories."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_a_replay_run_sh_started_from_empty_caches_compiled_where_vllm_logs_neither(tmp_path):
    log, mark = tmp_path / "replay-4.log", tmp_path / "replay-4.fresh"
    log.write_text("INFO Graph capturing finished in 29 secs\n")
    assert L.compile_mode(log, mark) is None
    mark.touch()
    assert L.compile_mode(log, mark) == "compile"
    log.write_text("INFO Directly load AOT compilation from path /x\n")
    assert L.compile_mode(log, mark) == "load"


def tiling_log(d: Path, *kept: int, xblock: int = 1024, only: bool = False) -> None:
    """One process a kept R0_BLOCK: each keeps it on reduction k (and 2048 on j unless ``only``), and ``xblock`` on a
    pointwise kernel."""
    d.mkdir(parents=True, exist_ok=True)
    for pid, r0 in enumerate(kept):
        recs = [{"kernel": "k", "heuristic": "REDUCTION", "candidates": [], "kept": {"R0_BLOCK": r0}},
                {"kernel": "p", "heuristic": "POINTWISE", "candidates": [], "kept": {"XBLOCK": xblock}}]
        if not only:
            recs.insert(0, {"kernel": "j", "heuristic": "REDUCTION", "candidates": [], "kept": {"R0_BLOCK": 2048}})
        (d / f"tilings.{pid}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))


def test_a_reduction_that_kept_two_configs_is_split(tmp_path):
    tiling_log(tmp_path / "a", 512)
    tiling_log(tmp_path / "b", 512, xblock=512, only=True)
    tiling_log(tmp_path / "c", 1024)
    tiling_log(tmp_path / "tp", 1024, 512)
    a, b, c, tp = (L.kept(tmp_path / n) for n in ("a", "b", "c", "tp"))
    assert L.split([a, b]) == {"runs": 2, "reductions": 2, "split": 0}
    assert L.split([a, c])["split"] == 1 and L.tilings(tmp_path / "tp") == {"runs": 1, "reductions": 2, "split": 1}
    assert L.kept(tmp_path / "none") is None and L.tilings(tmp_path / "none") is None


def replay_json(d: Path, differing: int, against: int | None = None) -> None:
    d.mkdir(parents=True)
    r = {"run_id": "r1", "requests": 4, "steps_served": 10, "rows_served": 40, "steps_differing": 0, "rows_differing": 0,
         "tokens_served": 40, "requests_tokens_differing": min(differing, 1), "token_positions_differing": differing,
         "bit_exact": differing == 0, "replay_over_served": 1.01}
    if against is not None:
        r["against"] = {"requests_tokens_differing": against}
    (d / "replay.json").write_text(json.dumps(r))


def test_only_a_reductions_choice_can_change_a_sum():
    assert L.sums({"kernel": "k", "heuristic": "PERSISTENT_REDUCTION", "candidates": []})
    assert not L.sums({"kernel": "triton_red_x", "heuristic": "POINTWISE", "candidates": []})
    assert L.sums({"kernel": "triton_red_fused_rms_norm_0", "candidates": [{"XBLOCK": 1}]})
    assert L.sums({"kernel": "Placeholder.DESCRIPTIVE_NAME", "candidates": [{"XBLOCK": 1, "R0_BLOCK": 2048}]})
    assert not L.sums({"kernel": "triton_poi_fused_1", "candidates": [{"XBLOCK": 1024}, {"XBLOCK": 512}]})


def test_the_arm_is_read_from_the_ladder_s_settings(monkeypatch):
    monkeypatch.delenv("LADDER_ARM", raising=False)
    assert L.pin() == "unpinned"
    for arm in L.PIN_ENV:
        monkeypatch.setenv("LADDER_ARM", arm)
        assert L.pin() == arm
    monkeypatch.setenv("LADDER_ARM", "deterministic")
    with pytest.raises(ValueError, match="pinned, rblock, unpinned"):
        L.pin()


def serve_capacity_only(tmp_path, monkeypatch) -> tuple[Path, list[Path]]:
    """`serve` for qwen3-8b with its capacity server failing at once: the sites serve_tax would have built."""
    out = tmp_path / "qwen3-8b"
    monkeypatch.setattr(L.S, "plugin_site", lambda site, arm: site)
    sites = []

    def main(argv):
        o = Path(argv[argv.index("--out") + 1])
        sites.append(L.S.plugin_site(o / ".site", L.S.ARMS["compiled"]))
        return 1

    monkeypatch.setattr(L.S, "main", main)
    monkeypatch.setattr(L, "register", lambda model: tmp_path)
    monkeypatch.setattr(L, "native_tokens", lambda tok: (1, 2))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(
        from_pretrained=lambda *a, **k: None)))
    monkeypatch.setattr(L.S, "corpus", lambda: SimpleNamespace(traffic=SimpleNamespace()))
    a = SimpleNamespace(model="qwen3-8b", c4="c4.json.gz", out=str(out), docs=10, burst=8, over=1.45, span_s=60.0, plan=None,
                        max_model_len=None)
    assert L.serve(a) == 1
    return out, sites


def served_env(site: Path) -> dict[str, str]:
    """The environment serve_tax starts a server with, after its plugin site."""
    return L.S.server_env(L.S.ARMS["compiled"], site.parent, site, None, 1.0, 1.0)


def test_serving_loads_the_pin_and_the_capacity_server_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("LADDER_ARM", "pinned")
    monkeypatch.delenv("LADDER_ROUTE_CAPTURE", raising=False)
    monkeypatch.delenv("ROUTECAP_OUT", raising=False)
    out, sites = serve_capacity_only(tmp_path, monkeypatch)
    assert json.loads((sites[0] / L.P.CONFIG).read_text()) == {"pin": True, "tilings": str(out / L.SERVING_TILINGS)}
    assert L.S.plugin_site(out / "record" / ".site", L.S.ARMS["record"]) == out / "record" / ".site"
    assert json.loads((out / "record" / ".site" / L.P.CONFIG).read_text()) == {"pin": True, "tilings": None}
    assert not (sites[0] / "route_capture_plugin.py").exists() and "ROUTECAP_OUT" not in served_env(sites[0])


def test_serving_with_the_route_capture_loads_it_in_each_server_writing_its_own_capture(tmp_path, monkeypatch):
    src = tmp_path / "route_capture"
    src.mkdir()
    (src / "route_capture_plugin.py").write_text("def register():\n    pass\n")
    (src / "settings.py").write_text("")
    monkeypatch.setenv("LADDER_ROUTE_CAPTURE", str(src))
    monkeypatch.setenv("ROUTECAP_OUT", "")  # so that the test's end unsets what serve sets
    out, sites = serve_capacity_only(tmp_path, monkeypatch)
    assert served_env(sites[0])["ROUTECAP_OUT"] == str(out / "capacity" / "route_capture")
    entry = (sites[0] / "verity_route_capture-0.0.dist-info" / "entry_points.txt").read_text()
    assert entry == "[vllm.general_plugins]\nverity_route_capture = route_capture_plugin:register\n"
    assert {f.name for f in sites[0].glob("*.py")} >= {"route_capture_plugin.py", "settings.py", "inductor_pin.py"}
    site = L.S.plugin_site(out / "record" / ".site", L.S.ARMS["record"])
    assert served_env(site)["ROUTECAP_OUT"] == str(out / "record" / "route_capture")


def test_summary_reads_a_model_directory(tmp_path):
    d = tmp_path / "out" / "pinned" / "qwen3-8b"
    d.mkdir(parents=True)
    (d / "serve.json").write_text(json.dumps({"model": "qwen3-8b", "checkpoint": "Qwen/Qwen3-8B@b", "tp": 1, "run_id": "r1",
                                              "rate": 87.0, "inductor_pin": "pinned", "capacity": {"req_per_s": 60.0},
                                              "window": {"sent": 4, "ok": 4}}))
    replay_json(d / "replay-1", 0)
    replay_json(d / "replay-2", 4, against=1)
    tiling_log(d / L.SERVING_TILINGS, 512)
    tiling_log(d / "replay-1", 512)
    tiling_log(d / "replay-2", 1024)
    (d / "replay-2.log").write_text("Compiling a graph for compile range\n")
    (d / "alone").mkdir()
    (d / "alone" / "alone.json").write_text(json.dumps({"run_id": "r1", "requests": 2, "tokens_served": 20,
                                                        "requests_tokens_differing": 1, "token_positions_differing": 5,
                                                        "against_replay": {"requests_tokens_differing": 1}}))
    row = L.model_row(d)
    assert (row["model"], row["pin"]) == ("qwen3-8b", "pinned")
    assert [r["bit_exact"] for r in row["replays"]] == [True, False]
    assert row["replays"][1]["rows_exact_share"] == pytest.approx(0.9)
    assert row["replays"][1]["compile"] == "compile"
    assert row["alone"]["requests_tokens_differing"] == 1
    md = L.markdown([row])
    assert row["serve_tilings"] == row["replays"][0]["tilings"] == {"runs": 1, "reductions": 2, "split": 0}
    assert row["tilings"] == {"runs": 3, "reductions": 2, "split": 1}
    assert ("| qwen3-8b | 1 | inductor_pin | 4, 10, 40 | 1 of 2 | "
            "90.000000% | 0, 1 (fresh compile) | 1 | 1 of 2 | 1 of 2 |") in md
    for legacy, arm in ((True, "rblock"), (False, "unpinned"), (None, "unpinned")):
        (d / "serve.json").write_text(json.dumps({"model": "qwen3-8b", "inductor_pin": legacy}))
        assert L.model_row(d)["pin"] == arm
