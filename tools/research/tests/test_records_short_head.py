"""The records gate compares inputs by full shas, whatever form the head came in."""

from test_queue import _gate, world  # noqa: F401  (world: the fixture)


def test_a_short_head_compares_inputs_like_its_full_sha(world):  # noqa: F811
    """`git diff-tree --stdin` reads a pair whose head isn't a full sha as one commit and prints no paths, so a 12-character head
    vouched for every candidate (#6699).  The gate resolves every name first: the short head's diff is the full one's."""
    a = world.change({"pkg/A.lean": "def a := 1\n"})
    b = world.change({"pkg/A.lean": "def a := 2\n"}, base=a)
    gate = _gate(world)
    assert gate._differs(b, [a], ["pkg"]) == {a}
    assert gate._differs(b[:12], [a], ["pkg"]) == {a}
    assert gate._differs(b[:12], [b], ["pkg"]) == set()
