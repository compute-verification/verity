"""The glue inventory (`verity_vllm.glue`): every `__models__` marker names a model the tree pins, a model's inventory joins its
marked modules to their rules, allowlist entries, family facts, profile patterns (from the `config.json` the Build reads) and capture
maps, the hours run from the checkpoint's staging event to the first PASS record, and a malformed marker or an unpinned model refuses."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from verity_vllm import glue as G
from verity_vllm.config import UsageError
from verity_vllm.engine.profiles import dense_generic
from verity_vllm.observe.fold.profiles import generic

GLM = "zai-org/GLM-5.3"
DSV4 = "deepseek-ai/DeepSeek-V4-Pro"
DSV4_CONFIG = Path(__file__).parents[1] / "check" / "fixtures" / "deepseek_v4" / "config.json"


@pytest.fixture(scope="module")
def marks() -> dict[str, tuple[str, ...]]:
    return G.markers()


def test_every_marker_names_a_pinned_model_and_both_seeds_are_marked(marks):
    pinned = G.known()
    assert {m for ms in marks.values() for m in ms} <= pinned
    assert {GLM, DSV4} <= {m for ms in marks.values() for m in ms}
    assert marks["verity/ml/definitions/registry/dsa.circuit-check.py"] == marks["verity/ml/definitions/registry/dsa.py"] == (GLM,)


def test_a_models_inventory_joins_its_marked_modules(marks, tmp_path, monkeypatch):
    inv = G.inventory(GLM, marks, sql=None, fetch=None)
    paths = {f["path"] for f in inv["files"]}
    assert "verity_vllm/program/frontend/rules/vllm_dsa.py" in paths and all(GLM in f["models"] for f in inv["files"])
    assert "DsaQ[vllm-dsa]_v1" in {r["rule"] for r in inv["rules"]}
    assert inv["allowlists"] and all(e["file"] in paths for e in inv["allowlists"])
    assert inv["family"]["model_type"] == "glm_moe_dsa" and inv["family"]["facts"] > 0
    assert "dsa.fused_q" in {e["pattern"] for e in inv["profile"]["entries"]} and inv["profile"]["config"] == "package:hf_configs"
    assert {m["id"] for m in inv["capture_maps"]} == {"glm-moe-dsa-eager-hopper-tp8-v0"}
    assert inv["hours"] is None and "hours" not in G.render(inv)

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    dsv4 = G.inventory(DSV4, marks, sql=None, fetch=None)
    assert "verity/ml/definitions/registry/mhc.py" in {f["path"] for f in dsv4["files"]}
    assert "deepseek-v4-eager-hopper-tp8-v0" in {m["id"] for m in dsv4["capture_maps"]}
    assert dsv4["model_type"] == "deepseek_v4"
    assert generic.structural_refusal(json.loads(DSV4_CONFIG.read_text())) is None
    assert {k: dsv4["profile"][k] for k in ("module", "config")} == {"module": "derived_DSV4_PRO_tp8", "config": "package:hf_configs"}
    assert "family facts  deepseek_v4 (mhc-moe)" in G.render(dsv4)


def test_the_config_is_the_builds_and_a_fetched_one_must_be_the_pinned_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    raw = DSV4_CONFIG.read_bytes()
    pin = hashlib.sha256(raw).hexdigest()
    rev = "b5968e9190ef611bbf34a7229255be88a0e937c1"
    snap = dense_generic.hf_snapshot(DSV4, rev) / "config.json"
    assert G.hf_config(DSV4, None, rev, pin, fetch=lambda repo, r: raw + b" ")[0] is None and not snap.exists()
    assert G.hf_config(DSV4, None, rev, None, fetch=lambda repo, r: raw) == (None, f"no {snap}, and no config_sha256 pinned to fetch it against")
    assert G.hf_config(DSV4, None, rev, pin, fetch=None) == (None, f"no {snap}")

    def down(repo: str, r: str) -> bytes:
        raise OSError("unreachable")
    assert G.hf_config(DSV4, None, rev, pin, fetch=down) == (None, f"no {snap}, and the hub's config.json at {rev[:12]}: unreachable")
    cfg, src = G.hf_config(DSV4, None, rev, pin, fetch=lambda repo, r: raw)
    assert (cfg["model_type"], src, snap.read_bytes()) == ("deepseek_v4", "hf_home:snapshot", raw)
    assert G.hf_config(DSV4, None, rev, pin, fetch=None)[1] == "hf_home:snapshot", "a snapshot present is read with no fetch"
    snap.write_bytes(raw + b" ")
    assert G.hf_config(DSV4, None, rev, pin, fetch=None) == (None, f"{snap} is not the pinned config_sha256 {pin[:12]}")
    assert G.hf_config(GLM, "GLM53", None, None)[1] == "package:hf_configs", "a role's own config comes first"


def test_hours_run_from_the_checkpoints_staging_event_to_the_first_pass_record():
    asked: list[str] = []

    def sql(q: str) -> list[dict]:
        asked.append(q)
        if "'staged'" in q:
            return [{"value": "2026-10-12T21:42:55Z", "ref": "vy-glm53-eu:/fs/x.verify"}]
        return [{"target": "art:" + "ab" * 32, "value": "row-a", "ts": "2026-10-17T02:34:55Z"}]

    h = G.hours("ckpt:o/it's@r", ["row-a", "row-b"], sql)
    assert h["staged"] == {"at": "2026-10-12T21:42:55Z", "ref": "vy-glm53-eu:/fs/x.verify"} and h["record"]["row"] == "row-a"
    assert h["hours"] == 100.9 and "'ckpt:o/it''s@r'" in asked[0] and "'PASS'" in asked[1] and "('row-a', 'row-b')" in asked[1]
    assert G.hours(None, [], sql) == {"checkpoint": None, "staged": None, "record": None, "hours": None} and len(asked) == 2
    assert "no staging event on ckpt:x" in G.render({**G.inventory(GLM, {}, sql=None, fetch=None),
                                                      "hours": {"checkpoint": "ckpt:x", "staged": None, "record": None, "hours": None}})


def test_a_definition_is_tied_by_the_manifest_and_carries_no_marker(marks, tmp_path, monkeypatch):
    verity = G.VERITY_HOME / "verity"
    assert not [p for p in verity.rglob("*.py") if "__pycache__" not in p.parts and G.marked(p, G.VERITY_HOME)]
    rows = json.loads(G.DEFINITIONS.read_text())["models"]
    assert rows and all(marks[rel] == tuple(ms) and not G.marked(G.source(rel), G.VERITY_HOME) for rel, ms in rows.items())
    stale = tmp_path / "definitions.json"
    stale.write_text(json.dumps({"models": {"verity_vllm/gone.py": [GLM]}}))
    monkeypatch.setattr(G, "DEFINITIONS", stale)
    with pytest.raises(ValueError, match="gone.py, which is not a module"):
        G.markers()


@pytest.mark.parametrize("src", ["__models__ = 'zai-org/GLM-5.3'\n", "__models__ = ()\n", "__models__ = (REPO,)\n"])
def test_a_malformed_marker_refuses(tmp_path, src):
    (tmp_path / "m.py").write_text(src)
    with pytest.raises(ValueError, match="tuple of HF repo ids"):
        G.marked(tmp_path / "m.py", tmp_path)


def test_an_unpinned_model_refuses():
    with pytest.raises(UsageError, match="no checkpoint or capture map pins it"):
        G.main(G.Options(models=["org/none"], store=False))
