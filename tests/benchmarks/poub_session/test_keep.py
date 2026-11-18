"""`keep.py`: what a served line keeps under the run's output. The row's records it keeps are an explicit list of public
files, none private by the vLLM integration's list, and a private name asked for is refused by name; an engine's kept
record holds neither the arena's IPC handle nor the tokens its probe was served, and keeps the probe's verdict."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load_keep():
    spec = importlib.util.spec_from_file_location("poub_session_keep", ROOT / "benchmarks" / "poub_session" / "keep.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


keep = load_keep()
HANDLE = "ab" * 64
PLAIN = [[90211, 4417, 118003], [7, 88, 9901, 31337]]


def test_the_private_names_are_the_integrations_list_as_data():
    doc = json.loads(keep.PRIVATE_NAMES.read_text())
    assert keep.PRIVATE_NAMES == ROOT / "adapters/vllm/verity_vllm/pipeline/private_names.json"
    kept_apart, private = keep.names.load()
    assert kept_apart == tuple(doc["kept_apart"]) and "replay_bundle_p" in kept_apart
    assert private == tuple(doc["private"]) and {"tokens_p*.json", "timeline.jsonl", "timeline_report.json"} <= set(private)


def test_every_keeper_uses_the_one_matcher():
    """keep.py's `private` is `private_names.py`'s, loaded by path, and research_outputs.py's calls it: neither has a
    matcher of its own."""
    pipeline = ROOT / "adapters/vllm/verity_vllm/pipeline"
    assert Path(keep.private.__code__.co_filename) == pipeline / "private_names.py"
    for f in (ROOT / "benchmarks/poub_session/keep.py", pipeline / "research_outputs.py"):
        assert "fnmatch" not in f.read_text(), f"{f.name} matches names itself; call private_names.private"
    assert "private_names.private(rel" in (pipeline / "research_outputs.py").read_text()


@pytest.mark.parametrize("rel", ["commit/tokens_p0.json", "commit/sampled_replay_p0.json", "timeline.jsonl",
                                 "timeline_report.json", "replay_bundle_p0/commit/values.json", "load_0.jsonl"])
def test_a_private_file_is_private_wherever_it_sits(rel):
    assert keep.private(rel)


def test_every_record_a_row_keeps_is_public_and_says_why():
    assert [r for r in keep.ROW if keep.private(r)] == []
    assert all(why.split(":")[0] in ("protocol structure", "the protocol's outputs") for why in keep.ROW.values())


def test_a_private_name_asked_for_is_refused_by_name_and_nothing_is_copied(tmp_path):
    src, dst = tmp_path / "row", tmp_path / "out"
    (src / "commit").mkdir(parents=True)
    for rel in ("stages.txt", "commit/tokens_p0.json", "timeline.jsonl"):
        (src / rel).write_text("x")
    with pytest.raises(ValueError, match=r"refused: commit/tokens_p0.json, timeline.jsonl private by "
                                         r"adapters/vllm/verity_vllm/pipeline/private_names.json"):
        keep.keep(src, dst, ["stages.txt", "commit/tokens_p0.json", "timeline.jsonl"])
    assert not dst.exists()
    assert keep.keep(src, dst, ["stages.txt", "commit/verdict.json"]) == ["stages.txt"]


def test_an_engines_kept_record_holds_no_handle_and_no_served_tokens(tmp_path):
    export = {"gpu": 0, "uuid": "GPU-1", "handle": HANDLE, "offset": 0, "allocation": 1 << 30, "weights_blocks": 9}
    checks = {"arena_decodes": True, "probe_tokens_equal": True, "first_block": 12, "hooked_modules": 3,
              "decode_calls": 0, "probe": {"plain": PLAIN, "decoded": PLAIN}}
    record = {**keep.export_kept(export), "checks": keep.checks_kept(checks)}
    (tmp_path / "engine.0.json").write_text(json.dumps(record))
    text = (tmp_path / "engine.0.json").read_text()
    assert HANDLE not in text and "handle" not in record
    assert all(str(t) not in text for t in (90211, 118003, 9901, 31337))
    assert record["checks"]["probe_tokens_equal"] is True and record["checks"]["probe"] == {"plain": [3, 4], "decoded": [3, 4]}
    assert {k: record[k] for k in ("offset", "allocation", "weights_blocks")} == {"offset": 0, "allocation": 1 << 30,
                                                                                  "weights_blocks": 9}
    assert keep.checks_kept(None) is None


def test_the_engine_and_the_prover_keep_through_keep_py():
    """engine.py writes its kept record through `export_kept` and `checks_kept`; node.py's prover.json its engines
    through `export_kept`: the handle and the probe's tokens reach only the scratch directory."""
    engine = (ROOT / "benchmarks/poub_session/engine.py").read_text()
    node = (ROOT / "benchmarks/poub_session/node.py").read_text()
    assert '**keep.export_kept(info), "checks": keep.checks_kept(checks)' in engine
    assert 'write_json(d / f"engine.{g}.json", info)' in engine       # the full export: --dir, run.sh's scratch
    assert 'write_json(a.out / f"engine.{g}.json", out)' in engine and "out = record(info, checks, a, d," in engine
    assert 'out["engines"] = prover_engines(exports)' in node
    assert '(a.out / "prover.json").write_text(json.dumps(out' in node


def test_the_commits_run_table_and_memory_map_are_refused_by_name(tmp_path):
    """`runs.jsonl` carries an unsalted digest of the served tokens and `memory_map.jsonl` digests of KV payloads: a keeper
    that asks for either (as one on the shared branch did) is refused before anything is copied."""
    src, dst = tmp_path / "row", tmp_path / "out"
    (src / "commit").mkdir(parents=True)
    for rel in ("commit/verdict.json", "commit/runs.jsonl", "commit/memory_map.jsonl", "commit/step_trace.7.jsonl.gz"):
        (src / rel).write_text("{}")
    with pytest.raises(ValueError, match="commit/runs.jsonl, commit/memory_map.jsonl, commit/step_trace.7.jsonl.gz"):
        keep.keep(src, dst, ["commit/verdict.json", "commit/runs.jsonl", "commit/memory_map.jsonl",
                             "commit/step_trace.7.jsonl.gz"])
    assert not dst.exists() or not any(dst.rglob("*"))


def test_the_command_line_keeps_the_rows_list(tmp_path):
    src, dst = tmp_path / "row", tmp_path / "out"
    (src / "commit").mkdir(parents=True)
    for rel in ("stages.txt", "commit/remote_session.json", "commit/runs.jsonl", "commit/tokens_p0.json"):
        (src / rel).write_text("{}")
    assert keep.main(["row", str(src), str(dst)]) == 0
    assert sorted(p.relative_to(dst).as_posix() for p in dst.rglob("*") if p.is_file()) == [
        "commit/remote_session.json", "stages.txt"]
    assert keep.main(["rows", str(src)]) == 2
