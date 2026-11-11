"""Flock's committed synthetic sets (``verity/protocols/flock/prover/tests/fixtures/synthetic_sets``) are what the generator writes for
their recipes (``verity_numerical.bench.fixture_sets``)."""
from __future__ import annotations

import json

import pytest

from verity_numerical.bench import fixture_sets as FS
from verity_flock import subcircuit_templates as TM


def test_the_committed_synthetic_sets_are_the_generators():
    assert FS.committed(), f"no synthetic sets under {FS.FIXTURES}"
    assert FS.check() == [], "rewrite them: python -m verity_numerical.bench.fixture_sets"


def test_a_place_holding_another_recipe_is_refused(tmp_path):
    _d, sub, n, seed = FS.committed()[0]
    manifest = FS.write(sub, n, seed, tmp_path) / "manifest.json"
    doc = json.loads(manifest.read_text())
    doc["provenance"]["recipe"]["parameters"] = {"other": 1}
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="two recipes share the place"):
        FS.write(sub, n, seed, tmp_path)


def test_an_argument_the_samplers_record_away_from_its_default_names_the_place():
    assert FS.place(TM.attention_head(16, 16), 16, 1).name == "n16-seed1"
    assert FS.place(TM.attention_head(16, 16, T=17), 16, 1).name == "n16-seed1-T17-17"
