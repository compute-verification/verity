"""The soundness package's layout of the live coin tree (`verifier/lean/soundness/FlockSoundness/Discharge/ZkTree/Layout.lean`:
`drLive`, `dlLive`, `keyBits`, `saltBits`, `tapeAt`, and `maskBytes` through `CROnly.hankel`), which
`zk_session_sound_real` is cited at, against the live crate's `CoinTree` (`live/src/coin_tree.rs`), through vectors the
crate draws (`fixtures/coin_tree_rust.json`) and the layout's reading of the same tapes (`fixtures/coin_tree_lean.json`,
`lean/coin_tree.sh`)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUST = json.loads((Path(__file__).parent / "fixtures" / "coin_tree_rust.json").read_text())
LEAN = json.loads((Path(__file__).parent / "fixtures" / "coin_tree_lean.json").read_text())


def test_the_draws_cover_the_live_spec():
    """The live draw is `Zk.coinSpec`'s shape (the session stream and one table's reps, 2^8 rounds of 512 coins), and the
    openings reach a block's last coin, a stream's last round, every stream, and `n = 0`."""
    live = next(d for d in RUST["draws"] if d["name"] == "live")
    assert live["spec"] == {"streams": ["session", "t/rep0", "t/rep1"], "rounds_log": 8, "block": 512}
    ops = {(o["stream"], o["round"], o["n"]) for o in live["openings"]}
    assert (2, 255, 512) in ops and (2, 3, 0) in ops and {s for s, _, _ in ops} == {0, 1, 2}
    assert live["points"]["m"] == 27 and len(RUST["draws"]) == 2


def test_lean_reads_the_key_coins_salts_and_points_the_coin_tree_draws():
    """On each tape, the layout's key (bit 2047 zero), each opening's coins and salt, and the link points are the ones
    `CoinTree::draw` drew and `CoinTree::open` opened."""
    assert [d["name"] for d in LEAN["draws"]] == [d["name"] for d in RUST["draws"]]
    for lean, rust in zip(LEAN["draws"], RUST["draws"]):
        assert lean["key"] == rust["key"]
        assert int(rust["key"][510], 16) < 8
        for lo, ro in zip(lean["openings"], rust["openings"], strict=True):
            assert (lo["stream"], lo["round"], lo["n"]) == (ro["stream"], ro["round"], ro["n"])
            assert lo["coins"] == ro["coins"] and len(lo["coins"]) == lo["n"]
            assert lo["salt"] == ro["salt"]
        assert lean["points"] == rust["points"]


def test_lean_hankel_is_the_coin_trees_mask():
    """`CROnly.hankel` on `keyBits` and `saltBits` of §9's key and salt is the leaf's mask `M(K)·y` (`x ⊕ b`), so the
    2047 key bits `Key` keeps, and the salt's 1536, are read as `Rules::mask` reads them."""
    assert LEAN["mask"]["mask"] == RUST["mask"]["mask"]
    assert len(RUST["mask"]["key"]) == 512 and len(RUST["mask"]["salt"]) == 384


def test_the_vectors_were_generated_from_this_tree():
    """Each fixture records the sha256 of the files its vectors depend on (the coin tree and its generator; the layout,
    its script, the Rust vectors and the toolchain pins). A change to any of them fails here until the vectors are
    regenerated."""
    stale = [p for f in (RUST, LEAN) for p, h in f["inputs"].items()
             if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h]
    assert not stale, f"changed since the vectors were generated, rerun backends/flock/tests/lean/coin_tree.sh: {stale}"
