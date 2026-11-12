"""`verity.definitions.ops.boolean.rolled`: a group written as calls of its words' circuits is the `specialized` group: the
same gates, cut into its classes of words and the gates they share, a body whose nodes don't grow with the group, and the
same values."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from verity.circuits import boolean as BL
from verity.circuits.boolean.forms import or_reduce
from verity.circuits.defs import bind
from verity.circuits.evaluation import bits as BS
from verity.definitions.ops.boolean import elementwise as EW
from verity.definitions.ops.boolean import rolled as RO
from verity.circuits.types import Array, Tuple


def _group(n: int):
    """A group of n 2-bit words with a shared s (the OR of each word's AND): word 0 its own, words 1 and 2 one circuit, the rest
    another, each reading s, and s the group's last output."""
    def sig(S):
        return (("x", Array(n, Array(2, BL.BIT))),), Tuple(Array(n, Array(2, BL.BIT)), Array(1, BL.BIT))

    def build(S):
        def run(C, *x):
            d = [C.AND(a[0], a[1]) for a in x]
            s = or_reduce(C, d)
            q = [[C.AND(x[0][0], s), s]]
            q += [[C.AND(d[j], s) ^ x[j][0], C.AND(x[j][1], s)] for j in (1, 2)]
            q += [[C.AND(C.AND(x[j][0], s), x[j][1] ^ s), d[j]] for j in range(3, n)]
            return q + [[s]]
        return run
    return build, sig


@pytest.mark.parametrize("n", [6, 12])
def test_a_rolled_group_is_the_specialized_group(n) -> None:
    """`rolled` cuts the group into its classes of words, the larger first, and its core, with exactly the `specialized`
    circuit's gates; its body is the classes' In batches, the core's Call and their Out batches whatever n is; and it is the
    `specialized` Definition on random groups."""
    build, sig = _group(n)
    whole = EW.specialized("RolledWholeCheck", 1, [], build, sig, register=False)
    d = RO.rolled("RolledCheck", 1, [], build, sig, register=False)
    S = SimpleNamespace()
    R = d.cut(S)
    assert [c["elements"] for c in R["classes"]] == [list(range(3, n)), [1, 2]] and R["rest"] == [0, 1, 2 * n]
    parts = [(len(c["elements"]), c["in"].counts, c["out"].counts) for c in R["classes"]]
    counts = whole.traced(S).counts
    assert {g: R["core"].counts[g] + sum(m * (i[g] + o[g]) for m, i, o in parts) for g in counts} == counts
    assert [nd.fn.id.split("{")[0] for nd in bind(d).body.nodes] == ["RolledCheckIn_v1", "RolledCheckIn_v1", "RolledCheckCore_v1",
                                                                     "RolledCheckOut_v1", "RolledCheckOut_v1"]
    x = np.random.default_rng(28).integers(0, 4, (2000, n)).astype(np.uint8)
    assert np.array_equal(np.asarray(BS.evaluate_words(bind(d), x)), np.asarray(BS.evaluate_words(bind(whole), x)))


def test_a_class_is_named_by_its_index() -> None:
    """A class's parts take its index V, and refuse one the group doesn't have."""
    build, sig = _group(6)
    d = RO.rolled("RolledClassCheck", 1, [], build, sig, register=False)
    In, _core, Out = d.parts
    assert In.signature(SimpleNamespace(V=1))[0][0][1] == Array(2, BL.BIT)
    with pytest.raises(ValueError, match="no class V=2 of 2"):
        Out.signature(SimpleNamespace(V=2))
    with pytest.raises(ValueError, match="V names a class"):
        RO.rolled("RolledStaticCheck", 1, ["V"], build, sig, register=False)
