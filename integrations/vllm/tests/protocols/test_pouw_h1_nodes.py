"""`pouw_h1.nodes.Nodes`: a registered value's strings and levels read out of the array the hash wrote are the tuple of
the same byte strings to every reader (indexing, slices, iteration, equality and hash either way round, a pickle), and the
CPU twin's `Registered` built from them opens every row under core's root and paths."""
from __future__ import annotations

import pickle
import shutil

import numpy as np
import pytest

from verity.core.protocols.pouw.audit import hm96_salt
from verity.core.protocols.sampled_proofs.one_stage import registered as G
from verity_vllm.protocols.pouw.h1 import registered as NR
from verity_vllm.protocols.pouw.h1 import registered_device as D
from verity_vllm.protocols.pouw.h1.nodes import Nodes

PROGRAM, WINDOW = "ab" * 64, "cd" * 64
KEY = bytes(range(32))


def test_nodes_are_the_tuple_of_their_bytes():
    raw = np.random.default_rng(0).integers(0, 256, 5 * 64, dtype=np.uint8)
    want = tuple(raw[64 * j:64 * (j + 1)].tobytes() for j in range(5))
    got = Nodes(raw, 64)
    assert len(got) == 5 and got[0] == want[0] and got[-1] == want[-1] and got[1:4] == want[1:4] and got[::-2] == want[::-2]
    assert tuple(got) == want and list(got) == list(want) and got == want and want == got and hash(got) == hash(want)
    assert got != want[:4] and want[:4] != got and got != (*want[:4], bytes(64)) and got != list(want)
    assert (got, 1) == (want, 1) and {want: 1}[got] == 1
    assert pickle.loads(pickle.dumps(got, protocol=5)) == want
    assert Nodes(raw.copy(), 64) == got and Nodes(raw, 32) != got and Nodes(raw[:0], 64) == ()
    with pytest.raises(IndexError):
        got[5]
    with pytest.raises(IndexError):
        got[-6]
    for raw_, width in ((raw[:65], 64), (raw.view(np.uint16), 64), (raw, 0)):
        with pytest.raises(ValueError):
            Nodes(raw_, width)


@pytest.mark.skipif(not any(shutil.which(c) for c in ("g++", "c++", "clang++")), reason="no C++ compiler")
@pytest.mark.parametrize("n,w", [(1, 1), (5, 3), (33, 70)])
def test_the_cpu_twins_registered_value_opens_under_cores_root(n, w):
    words = np.random.default_rng(n).integers(0, 1 << 16, (n, w), dtype=np.uint16)
    core = G.Registered.commit(PROGRAM, "h1/v", words.tolist(), [hm96_salt(KEY, 3, j) for j in range(n)],
                               schema=G.SCHEMA_V2, window=WINDOW)
    bcs, levels = D.host_commit(words, KEY, 3, None, core.domain.domain_id)
    assert isinstance(bcs, Nodes) and all(isinstance(lv, Nodes) for lv in levels)
    assert (bcs, levels) == (core.bcs, core.levels) and NR._numpy(words, KEY, 3, None, core.domain.domain_id) == (bcs, levels)
    got = G.Registered(program=PROGRAM, value="h1/v", words=w, word_bits=G.WORD_BITS, rows=core.rows, salts=core.salts,
                       bcs=bcs, levels=levels, schema=G.SCHEMA_V2, window=WINDOW)
    assert got.root == core.root and got.entry() == core.entry()
    assert all(got.path(p) == core.path(p) and G.verify_read(core.domain, core.root, p, got.bcs[p], got.path(p), G.SCHEMA_V2)
               for p in range(n))
