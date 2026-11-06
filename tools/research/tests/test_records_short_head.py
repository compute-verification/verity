"""The records gate keys a head's Lean inputs from its tree, whatever form the head came in."""

from test_queue import LOCK, _gate, _key, _keyed, _lock, _published, world  # noqa: F401  (world: the fixture)


def test_a_short_head_is_vouched_for_exactly_as_its_full_sha(world):
    """A 12-character head is keyed from the tree its name resolves to, so it has its full sha's key: a pass on its parent's key
    vouches for neither form, and a pass on the head's own key vouches for both.  A gate that read no Lean inputs at a short
    head would vouch for every candidate."""
    world.with_store_remote()
    a = _keyed(world, {LOCK: _lock()})
    b = world.change({LOCK: _lock(A="2")})
    _published(world, a, passes={_key(world.dev, a): "pkg"})
    gate = _gate(world)
    gate._read()
    assert gate._vouch(b, "pkg", 0) is None and gate._vouch(b[:12], "pkg", 0) is None
    key = _key(world.dev, b)
    _published(world, b, passes={key: "pkg"})
    gate = _gate(world)
    gate._read()
    full, short = gate._vouch(b, "pkg", 0), gate._vouch(b[:12], "pkg", 0)
    assert full and short and f"(key {key[:12]})" in full and f"(key {key[:12]})" in short, (full, short)
