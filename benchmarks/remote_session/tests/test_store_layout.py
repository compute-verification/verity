"""The store a checkpoint lays out, end to end (its layout is core's `remote_bandwidth.store`, whose own tests are beside
it). On a small synthetic checkpoint (`store_layout.synthetic`): the arena is the engine's packing (`engine.pack`); a
prover that registers that store under the verifier's payload key is accepted, with the store's regions; the node's
layout names its host tier as one region; verifier.py takes no checkpoint; `store_layout`'s pin is the one core reads,
and its SHA-256s are the arena's; and the paced line reports weights held and filler held apart."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))

from verity.protocols.accounting.poub.verifier import store as rstore  # noqa: E402

import engine  # noqa: E402
import store_layout as sl  # noqa: E402
from verity.protocols.accounting.poub.verifier.session import verifier  # noqa: E402
from test_session import run  # noqa: E402

ROLE = "synthetic-bf16"
B = rstore.BLOCK
TINY = {"layers": 1, "hidden": 16, "heads": 2, "kv_heads": 1, "intermediate": 32, "vocab": 64}
"""A checkpoint of 6 weights blocks an arena, so two arenas of 7 and 2 host blocks make `test_session`'s 16-block store
(the prover's encoding is most of a session's time)."""


@pytest.fixture
def ckpt(tmp_path) -> Path:
    sl.synthetic(tmp_path / "m", name=ROLE, **TINY)
    return tmp_path / "m"


@pytest.mark.parametrize("tied", [False, True])
def test_the_arena_is_the_engines_packing_of_vllms_weights(tmp_path, tied):
    """Each weight of core's arena sits where `engine.pack` places it in an engine's arena of its blocks."""
    pin = sl.synthetic(tmp_path / "m", name=ROLE, tied=tied)
    a = rstore.arena(tmp_path / "m", pin["checkpoint"])
    assert len(a.entries) == 9 + (not tied)
    assert [e.offset for e in a.entries] == engine.pack([e.nbytes for e in a.entries], a.weights_blocks * B)


def test_store_layouts_pin_is_the_one_core_reads_and_its_sha256s_the_arenas(ckpt, tmp_path, monkeypatch):
    """`store_layout.py pin` pins every file of a checkpoint as core's `read_pin` reads a pin, and `arena_sha256`, which
    `node.py --model-dir` holds each engine's plain hashes to, is the SHA-256 of each span of the arena as it is read,
    in worker processes too."""
    want = json.loads((ckpt / "pin.json").read_text())
    (ckpt / "pin.json").unlink()
    assert sl.main(["pin", str(ckpt), "--name", ROLE, "--out", str(tmp_path / "local.json")]) == 0
    assert rstore.read_pin(tmp_path / "local.json", ROLE) == want
    a = rstore.arena_of(ckpt)
    monkeypatch.setattr(sl, "GIB", 4 * B)
    nbytes = (a.weights_blocks + 3) * B
    spans = [hashlib.sha256(a.read(lo, min(4 * B, nbytes - lo))).hexdigest() for lo in range(0, nbytes, 4 * B)]
    assert len(spans) == 3 and sl.arena_sha256(a, nbytes) == sl.arena_sha256(a, nbytes, 2) == spans


def test_a_store_registered_as_the_layout_lays_it_out_is_accepted_with_its_regions(ckpt):
    """The prover's store is the checkpoint's (two arenas of weights and zero filler, then a host tier), salted under
    the payload key the verifier sent: the verifier takes its regions from the layout, and the honest session is
    accepted."""
    store = rstore.Store(rstore.arena_of(ckpt), 2, 7, 2)
    got = {}
    (r,), (p,) = run("honest", store=store, then=lambda v: got.update(regions=v.regions))
    assert r["accepted"] and p["accepted"]
    assert got["regions"] == tuple((x["kind"], x["tier"], x["first"], x["blocks"]) for x in store.regions())


def test_the_nodes_layout_names_a_host_tier_in_several_segments_as_one_region(ckpt):
    """The serving node's store holds its host tier in a segment a GPU; its layout (`store_layout.regions`) names that
    tier as one region, the store's own regions, and a store with no host tier has no host region."""
    a = rstore.arena_of(ckpt)
    store = rstore.Store(a, 2, 7, 2)
    n, base, on_host = [7, 7, 1, 1], [0, 7, 14, 15], [False, False, True, True]
    layout = sl.regions(n, base, on_host, [6, 6])
    assert layout == store.regions()
    assert verifier.regions_of(layout, 16) == verifier.regions_of(store.regions(), 16)
    assert sl.regions(n[:2], base[:2], on_host[:2], [6, 6]) == rstore.Store(a, 2, 7).regions()


def test_verifier_py_takes_no_checkpoint(tmp_path):
    """verifier.py holds no checkpoint: `--model` and the store's shape are no arguments of it."""
    for flag, value in (("--model", ROLE), ("--checkpoint", str(tmp_path)), ("--arena-blocks", "7")):
        with pytest.raises(SystemExit):
            verifier.main(["--out", str(tmp_path / "out"), flag, value])


def test_the_paced_line_reports_weights_held_and_filler_held_apart():
    """A stretch's credit is each kind held less its lost blocks: weights, and filler, which is the rest of HBM and the
    host tier, each in both forms; filler never counts toward the weights line."""
    N, b = 1000, 4096
    held = {"weights": 600 * b, "hbm": 700 * b, "host": 300 * b}
    mem = {"windows": [{"held": held, "average": {"held": held}} for _ in range(12)]}
    seen = [{"released_ns": 10 * g, "passes": 2, "arrivals_ns": [10 * g + 5, 10 * g + 10]} for g in range(12)]

    def line(wrong: float) -> dict:
        return verifier.paced_line(mem, seen, groups=4, wrong=wrong, blocks=N, rebuild_per_ns=0.0, chip_bytes=0,
                                   block_bytes=b, node_bytes=10 * N * b, burst_bytes=0.0)

    whole = line(0.0)
    for form in ("time", "snapshot"):
        assert whole[form]["weights_bytes"] == 600 * b and whole[form]["filler_bytes"] == 400 * b
        assert whole[form]["share"] == 0.06 and whole[form]["filler_share"] == 0.04
    lossy = line(80.0)
    lost = lossy["snapshot"]["lost_blocks"] * b
    assert lost > 0 and lossy["snapshot"]["weights_bytes"] == pytest.approx(600 * b - lost)
    assert lossy["snapshot"]["filler_bytes"] == pytest.approx(400 * b - lost)
    assert lossy["stretches"][0][verifier.TIME_HELD]["credited"]["filler"] == pytest.approx(
        400 * b - lossy["stretches"][0][verifier.TIME_HELD]["lost_blocks"] * b)
