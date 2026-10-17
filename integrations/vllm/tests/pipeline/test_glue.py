"""The glue inventory (`verity_vllm.glue`): every `__models__` marker names a model the tree pins, a model's inventory joins its
marked modules to their rules, allowlist entries, family facts, profile patterns and capture maps, the hours come from the store's
first run on the staged weights and first PASS record, and a malformed marker or an unpinned model refuses."""
from __future__ import annotations

import pytest

from verity_vllm import glue as G
from verity_vllm.config import UsageError

GLM = "zai-org/GLM-5.3"
DSV4 = "deepseek-ai/DeepSeek-V4-Pro"


@pytest.fixture(scope="module")
def marks() -> dict[str, tuple[str, ...]]:
    return G.markers()


def test_every_marker_names_a_pinned_model_and_both_seeds_are_marked(marks):
    pinned = G.known()
    assert {m for ms in marks.values() for m in ms} <= pinned
    assert {GLM, DSV4} <= {m for ms in marks.values() for m in ms}
    assert marks["verity_vllm/program/registry/dsa.circuit-check.py"] == marks["verity_vllm/program/registry/dsa.py"] == (GLM,)


def test_a_models_inventory_joins_its_marked_modules(marks):
    inv = G.inventory(GLM, marks, sql=None)
    paths = {f["path"] for f in inv["files"]}
    assert "verity_vllm/program/frontend/rules/vllm_dsa.py" in paths and all(GLM in f["models"] for f in inv["files"])
    assert "DsaQ[vllm-dsa]_v1" in {r["rule"] for r in inv["rules"]}
    assert inv["allowlists"] and all(e["file"] in paths for e in inv["allowlists"])
    assert inv["family"]["model_type"] == "glm_moe_dsa" and inv["family"]["facts"] > 0
    assert "dsa.fused_q" in {e["pattern"] for e in inv["profile"]["entries"]}
    assert {m["id"] for m in inv["capture_maps"]} == {"glm-moe-dsa-eager-hopper-tp8-v0"}
    assert inv["hours"] is None and "hours" not in G.render(inv)

    dsv4 = G.inventory(DSV4, marks, sql=None)
    assert "verity_vllm/program/registry/mhc.py" in {f["path"] for f in dsv4["files"]}
    assert "deepseek-v4-eager-hopper-tp8-v0" in {m["id"] for m in dsv4["capture_maps"]}
    assert G._profile(None, 8, set()) == {"refused": "no checkpoint role in manifests/checkpoints.json"}


def test_hours_run_from_the_first_run_on_the_staged_weights_to_the_first_pass_record():
    asked: list[str] = []

    def sql(q: str) -> list[dict]:
        asked.append(q)
        if "attempt_params" in q:
            return [{"id": "r1", "start_utc": "2026-10-12T23:26:10.265Z"}]
        return [{"target": "art:" + "ab" * 32, "value": "row-a", "ts": "2026-10-17T02:34:55Z"}]

    h = G.hours("/fs/it's", ["row-a", "row-b"], sql)
    assert h["staged"] == {"run": "r1", "at": "2026-10-12T23:26:10.265Z"} and h["record"]["row"] == "row-a" and h["hours"] == 99.1
    assert "'/fs/it''s'" in asked[0] and "'PASS'" in asked[1] and "('row-a', 'row-b')" in asked[1]
    assert G.hours(None, [], sql) == {"staged_dir": None, "staged": None, "record": None, "hours": None} and len(asked) == 2


@pytest.mark.parametrize("src", ["__models__ = 'zai-org/GLM-5.3'\n", "__models__ = ()\n", "__models__ = (REPO,)\n"])
def test_a_malformed_marker_refuses(tmp_path, src):
    (tmp_path / "m.py").write_text(src)
    with pytest.raises(ValueError, match="tuple of HF repo ids"):
        G.marked(tmp_path / "m.py", tmp_path)


def test_an_unpinned_model_refuses():
    with pytest.raises(UsageError, match="no checkpoint or capture map pins it"):
        G.main(G.Options(models=["org/none"], store=False))
