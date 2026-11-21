from verity.circuits.replay_links import Link
from verity.protocols.proof.spot.verifier import second_stage as SS

ROOT1, DOM1 = "aa" * 64, "bb" * 64


def _own() -> SS.Own:
    boundary = {"roots": [{"port": "logits", "root": ROOT1, "domain": DOM1, "leaves": 12}]}
    return SS.Own(sample=(1, 3), parts=None, interior={}, boundary=boundary, drawn=(),
                  links=(Link("Pick_v0", "x", "logits"),))


def _stmt(value: str, root: str, domain: str, leaves: int) -> dict:
    port = {"value": value, "root": root, "domain": domain, "leaves": leaves}
    return {"template": "Pick_v0", "header": {}, "draw": {}, "registered": {"ports": {"x": port}}}


def test_a_linked_read_of_a_tree_committed_after_the_sample_is_refused_by_name():
    """A port a link names reads round 1's rows at round 1's root: an interior run input's tree ``input/n0``, committed
    after the first stage's sample at a root of its own, is refused (``second-link-root``); at round 1's root, domain
    and leaves it is not, whatever value it names."""
    own = _own()
    fresh = SS.link_root_failures(own, [_stmt("input/n0", "cc" * 64, "dd" * 64, 4)])
    assert [(c, d["port"]) for c, d in fresh] == [("second-link-root", "x")]
    assert SS.link_root_failures(own, [_stmt("input/n0", ROOT1, DOM1, 12)]) == []
    assert SS.link_root_failures(own, [_stmt("logits", ROOT1, DOM1, 12)]) == []
