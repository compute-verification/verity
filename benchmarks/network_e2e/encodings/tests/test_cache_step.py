"""CacheStep_v1's specification (`verity.core.protocols.certifier.cache_step.reference`) against the derived rules'
prefix rules (`derived.Prefixes`) over whole traces, through `cache_vectors.drive`, and the certifier's difftest vectors
as `cache_vectors` makes them now."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
REPO = HERE.parents[3]


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


V = _load("enc_cache_vectors", HERE.parent / "cache_vectors.py")
D = V.D
FIXTURES = HERE / "fixtures"
VECTORS = REPO / "verity" / "core" / "protocols" / "certifier" / "tests" / "fixtures" / "cache_step_vectors.json"


def _trace(name):
    return V.load(FIXTURES / name) if (FIXTURES / name).is_dir() else D.load_trace(FIXTURES / name)


@pytest.mark.parametrize("name", ["sync-crush.json.gz", "corpus-smallkv"])
def test_the_cache_is_derived_s_prefix_rules_over_a_whole_trace(name) -> None:
    """Every call of the trace, in scheduling order: each derived hit is the one `Prefixes` derives (vLLM's own hit) and
    one unit either side is refused; each sent hit is a cached prefix (vLLM's eviction never hits past the cache that
    never evicts) and one unit past the cache is refused; each hit's sources at its step's end are `Prefixes.sources`
    and the advice's; and the lookup and the computed units in one call leave the same table as in two."""
    tr = _trace(name)
    cfg = D.config_of(tr)
    adv, order = D.advice_of(tr)
    held = D.holdings_of(tr)
    tokens = D.tokens_of(held, order, [q.prompt_len for q in adv.requests])
    sent_from = V.first_sent(adv, cfg, held, order)
    got = V.drive(adv, tokens, cfg, sent_from, V.statics_for(adv, tokens, cfg, 1 << 14))["counts"]
    assert got["steps"] == len(adv.steps) and got["derived"] >= 2 and got["sent"] >= 40
    assert got["refused"] >= got["derived"] + got["sent"]
    assert got["hit_units"] > 100 and got["entries"] > 900


def test_the_vectors_are_current() -> None:
    """The certifier's vectors (`cache_step_vectors.json`) are what `cache_vectors` makes from these fixtures now."""
    want = json.loads(VECTORS.read_text())
    for name, sl in want["slices"].items():
        assert V.slice_of(_trace(name), sl["statics"]["N"], sl["steps"]) == sl, name
