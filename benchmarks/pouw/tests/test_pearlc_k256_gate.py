"""The k = 256 gate: the cross-group cancellation families mirror a word's two promotion groups, the exhaustive searches
replay the served -h3 bit for bit (`cell` asserts it), and the slice tool's vectorized rules are the scheme's own."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

POUW = Path(__file__).parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


G = _load("pouw_pearlc_k256_gate", POUW / "pearlc_k256_gate.py")
S = _load("pouw_glm53_slices", POUW / "pearl_c_sm90" / "glm53_slices.py")


def test_mirror_family_cancels_across_the_two_groups():
    K = G.K
    a, b = K.family("xcancel-mirror", 4, 4, 256, K.lines(256, K._rng(1, "FA", 256)), K._rng(1, "xcancel-mirror", 256, 1.0))
    assert (a[:, :128] == a[:, 128:]).all() and (b[:, :128] == -b[:, 128:]).all()


def test_gate_cell_replays_the_served_scheme_and_finds_no_shortcut():
    r = G.cell("gaussian", 1, rows=16, cols=8)
    assert r["in_domain"] and r["credited_words"] > 0
    assert r["fragment_share_of_credit"] == 0.0 and not r["falsifier"]
    assert r["peel_words_u_equals_c"] == 0 and r["peel_fp8_route_slices"] == 0


def test_slice_rules_agree_with_the_scheme():
    rng = np.random.default_rng(7)
    x = rng.standard_t(3, (48, 256)).astype(np.float32)
    x[:8, :3] *= 60.0                     # spikes: liveness fails on some rows
    x[8:12] = 0.0                         # zero rows: the floor fails
    x[12, ::8] = 1e-30
    f_ok, l_ok, _ = S.rules(x)
    exact = [S.exact(row) for row in x]
    assert f_ok.tolist() == [e[0] for e in exact] and l_ok.tolist() == [e[1] for e in exact]
    assert not all(l_ok) and not all(f_ok)
