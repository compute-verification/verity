"""The carried pool's row export over recorded traces, through `pool_row_trace.drive`: vLLM d9105ea80's ``BlockPool``,
transcribed and keyed by BlockKey_v1, makes each of the replay's pool operations, and the two pools agree after each
one. At each point no block is held, `verity_vllm.engine.pool_row.export`'s row and root are the ones `_KV` gives; after
the trace the genesis reset is the empty row."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PRT = _load("enc_pool_row_trace", HERE.parent / "pool_row_trace.py")

PREFIX = {"cached": 185, "cached at an edge": 157, "checks": 1261, "copies at an edge": 1, "edges": 1, "hit blocks": 52, "keys": 185, "lookups": 14,
          "ops": 1261, "steps": 155}
ASYNC_CRUSH = {"cached": 1939, "cached at an edge": 158, "checks": 7514, "copies at an edge": 1, "edges": 1, "hit blocks": 5579, "keys": 1499,
               "lookups": 1368, "ops": 7514, "steps": 1506}
SYNC_CRUSH = {"cached": 1932, "cached at an edge": 158, "checks": 7466, "copies at an edge": 1, "edges": 1, "hit blocks": 5553, "keys": 1499,
              "lookups": 1355, "ops": 7466, "steps": 1493}


@pytest.mark.parametrize("name, want", [("async-prefix-cache.jsonl", PREFIX), ("async-crush.json.gz", ASYNC_CRUSH),
                                        ("sync-crush.json.gz", SYNC_CRUSH)])
def test_vllm_s_pool_exports_the_row_the_replay_s_pool_gives(name, want) -> None:
    """Every operation agrees (`drive` raises otherwise), with each hit the hash map's first copy, through the crush
    traces' preemptions. The trace's end, the one point at which none of these traces' requests runs, exports a row
    caching all but one or two blocks, one copy a key: these traces never compute one block twice."""
    got = PRT.drive(HERE / "fixtures" / name)
    assert {k: v for k, v in got.items() if k != "roots"} == want and len(got["roots"]) == want["edges"]
