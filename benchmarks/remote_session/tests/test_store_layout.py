"""The store binding end to end (`verifier.py --model`; its layout and decision are core's `remote_bandwidth.store`, whose
own tests are beside it). On a small synthetic checkpoint (`store_layout.synthetic`): the arena is the engine's packing
(`engine.pack`); a prover that registers the store the pin lays out under the verifier's payload key is accepted and
bound, and one whose root is any other (its own salts, another checkpoint, another shape) is refused with the reason
before the registry draws anything; `verifier.py --model` refuses a pin whose files aren't the checkpoint's before it
listens; `store_layout`'s pin is the one core reads, and its SHA-256s are the arena's; and the paced line reports weights
held and filler held apart."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import socket
import sys
import threading
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))

from verity.core.protocols.remote_bandwidth import store as rstore  # noqa: E402

import engine  # noqa: E402
import prover  # noqa: E402
import store_layout as sl  # noqa: E402
import verifier  # noqa: E402
import wire  # noqa: E402
from test_session import P512, run  # noqa: E402

ROLE = "synthetic-bf16"
NODE = b"test-node"
B = rstore.BLOCK
TINY = {"layers": 1, "hidden": 16, "heads": 2, "kv_heads": 1, "intermediate": 32, "vocab": 64}
"""A checkpoint of 6 weights blocks an arena, so two arenas of 7 and 2 host blocks make `test_session`'s 16-block store
(the prover's encoding is most of a session's time)."""


@pytest.fixture
def ckpt(tmp_path) -> Path:
    sl.synthetic(tmp_path / "m", name=ROLE, **TINY)
    return tmp_path / "m"


def binding(ckpt: Path, engines: int = 2, arena_blocks: int = 7, **kw) -> rstore.Binding:
    return rstore.Binding(ROLE, ckpt, ckpt / "pin.json", engines, arena_blocks, log=lambda m: None, **kw)


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


def test_a_store_registered_as_the_layout_lays_it_out_is_accepted_and_bound(ckpt):
    """The prover's store is the pinned checkpoint's (two arenas of weights and zero filler, then a host tier), salted
    under the payload key the verifier sent: the verifier recomputes the same root, binds the registration, takes its
    regions from the layout, and the honest session is accepted."""
    b = binding(ckpt, workers=2)
    store = rstore.Store(b.arena, 2, 7, 2)
    got = {}
    (r,), (p,) = run("honest", store=store, binding=b, verdict="python",
                     then=lambda v: got.update(reg=v.registration, regions=v.regions, key=v.payload_key))
    assert r["accepted"] and p["accepted"]
    rec = b.record()
    assert rec["bound"] is True and rec["name"] == ROLE and rec["checkpoint_sha512"] == b.pin["checkpoint_sha512"]
    assert rec["expected_root"] == rec["payload_root"] == got["reg"].payload_root.hex()
    assert rec["payload_key"] == got["key"].hex() and rec["host_blocks"] == 2 and rec["blocks"] == 16
    assert got["regions"] == tuple((x["kind"], x["tier"], x["first"], x["blocks"]) for x in store.regions())


def greet(backend, b: rstore.Binding | None, **kw) -> tuple[verifier.Verifier, ValueError | None, list[str]]:
    """A prover over `backend` greeting a verifier that binds to `b`: the verifier, its refusal, and the `bye` the prover
    read."""
    x, y = socket.socketpair()
    p = prover.ProverSide(wire.Conn(y), backend, NODE, prover.parse_plan("honest"), seed=11, log=lambda m: None, **kw)
    heard = []

    def prove() -> None:
        try:
            p.register(codec=P512)
        except wire.Closed as e:
            heard.append(str(e))

    th = threading.Thread(target=prove, daemon=True)
    th.start()
    v = verifier.Verifier(wire.Conn(x), source=os.urandom(32), windows=3, window_ns=20, samples=6, delta=2.0 ** -40,
                          jitter_ns=10 ** 9, log=lambda m: None, verdict="python", binding=b)
    refused = None
    try:
        v.register()
    except ValueError as e:
        refused = e
    th.join()
    v.stop()
    return v, refused, heard


def test_a_store_whose_root_isnt_the_pinned_models_is_refused_with_why_before_the_registry_draws(ckpt, tmp_path):
    """Each of: the right store salted with the prover's own key (`--seed-payload-key`), another checkpoint's arenas
    of the same shape, a store of other arenas, and a store short of the arenas. The verifier answers `bye` with the
    reason and the registry has drawn nothing for the node."""
    b = binding(ckpt)
    store = rstore.Store(b.arena, 2, 7, 2)
    assert b.arena.weights_blocks == 6
    sl.synthetic(tmp_path / "other", name=ROLE, seed=1, **TINY)
    cases = [
        (prover.CpuBackend(16, random.Random(7), store=store), {"seed_payload_key": True},
         re.escape("is not the pinned model synthetic-bf16's store (2 arenas of 7 blocks, 6 of weights each, then 2 "
                   "host filler blocks) under this verifier's payload key")),
        (prover.CpuBackend(16, random.Random(7), store=rstore.Store(rstore.arena_of(tmp_path / "other"), 2, 7, 2)), {},
         "is not the pinned model synthetic-bf16's store"),
        (prover.CpuBackend(16, random.Random(7), store=rstore.Store(b.arena, 2, 8)), {},
         "the hello's layout .* is not the pinned model synthetic-bf16's"),
        (prover.CpuBackend(12, random.Random(7)), {}, "the hello's store is 12 blocks, fewer than the pinned model's 2 "
                                                      "arenas of 7"),
    ]
    for backend, kw, why in cases:
        v, refused, heard = greet(backend, b, **kw)
        assert isinstance(refused, rstore.Refused), why
        assert re.search(why, str(refused)), (why, str(refused))
        assert heard == [f"refused: {refused}"] and v.registry.drawn(NODE) == 0
        assert b.record()["bound"] is False
    assert "register the store `remote_bandwidth.store` lays out, salted with the key the verifier sent" in str(
        greet(cases[0][0], b, seed_payload_key=True)[1])
    v, refused, heard = greet(prover.CpuBackend(16, random.Random(7), store=store), b)
    assert refused is None and heard == [] and v.registry.drawn(NODE) == 1 and b.record()["bound"] is True


def test_verifier_py_refuses_a_pin_whose_files_arent_the_checkpoints_before_it_listens(ckpt, tmp_path):
    """`verifier.py --model` holds every pinned file to its SHA-512 before it listens: a pin naming another hash for a
    file exits 3, and verifier.json keeps why and an unbound binding."""
    pin = json.loads((ckpt / "pin.json").read_text())
    shard = sorted(f for f in pin["checkpoint"] if f.endswith(".safetensors"))[0]
    other = {**pin["checkpoint"], shard: "ab" * 64}
    pin_path = tmp_path / "other-pin.json"
    pin_path.write_text(json.dumps({"name": ROLE, "checkpoint": other, "checkpoint_sha512": rstore.checkpoint_sha512(other)}))
    out = tmp_path / "out"
    sys_argv = sys.argv
    try:
        sys.argv = ["verifier.py", "--out", str(out), "--verdict", "python", "--model", ROLE, "--checkpoint", str(ckpt),
                    "--pin", str(pin_path), "--arena-blocks", "7", "--layout-workers", "1"]
        assert verifier.main() == 3
    finally:
        sys.argv = sys_argv
    doc = json.loads((out / "verifier.json").read_text())
    assert "not the pin's" in doc["refused"] and doc["binding"]["bound"] is False


def test_the_paced_line_reports_weights_held_and_filler_held_apart():
    """A stretch's credit is each kind held less its lost blocks: weights, and filler, which is the rest of HBM and the
    host tier, each in both forms; filler never counts toward the weights line."""
    N, b = 1000, 4096
    held = {"weights": 600 * b, "hbm": 700 * b, "host": 300 * b}
    mem = {"windows": [{"held": held, "average": {"held": held}} for _ in range(12)]}
    seen = [{"released_ns": 10 * g, "passes": 2, "arrivals_ns": [10 * g + 5, 10 * g + 10]} for g in range(12)]

    def line(wrong: float) -> dict:
        return verifier.paced_line(mem, seen, groups=4, wrong=wrong, blocks=N, rebuild_per_ns=0.0, chip_bytes=0,
                                   block_bytes=b, node_bytes=10 * N * b)

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
