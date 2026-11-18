"""A served call's y tree where the kernels take no rows of its width (`served_y_device`), CPU only, on the sm_90a kernels'
`run.py` with no device: GLM-5.3's merged q_a/kv_a projection's y is 2,624 words wide, no k the kernels' rows take, so the
run's device trees give no tree for it and check none, and `y_tree` makes it by `commit_y`, the tree every device tree is
held to.  A width the kernels take is the card's (its first tree held to `commit_y` there).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from verity.protocols.resource_accounting.compute.pouw.prover.kernels.pearl_c_sm90 import run as R
from verity.protocols.resource_accounting.compute.pouw.prover.serving.pearl_c.served_y import HM96_SALT, commit_y
from verity_torch.pouw.pearl_c.served_y_device import DeviceTrees, y_tree

torch = pytest.importorskip("torch")


def _run():
    return SimpleNamespace(kernel=SimpleNamespace(run=R, dev=None), variant="h100", const_keys={}, device="cpu",
                           salt=bytes(range(64)), _y_trees=None, cur=None)


def test_a_width_the_kernels_refuse_gets_no_device_tree_and_no_check():
    """`DeviceTrees`: the kernels refuse rows of k = 2,624 words (k a multiple of 256), so its pipeline is None, held
    so with the kernels' refusal, and the call gives no tree: nothing is launched or held to `commit_y` at that width."""
    trees = DeviceTrees(_run())
    trees._check = lambda *a: pytest.fail("a refused width is checked")
    y = torch.zeros(3, 2624, dtype=torch.bfloat16)
    salts = torch.zeros(3 * HM96_SALT, dtype=torch.uint8)
    assert trees(y, 5, salts) is None and trees.pipes == {2624: None} and "k of 256" in trees.refused[2624]


def test_y_tree_at_a_refused_width_is_commit_ys():
    """`y_tree` over a 2,624-wide y: `commit_y`'s tree, byte for byte, its root the last node."""
    run = _run()
    y = torch.linspace(-3, 3, 7 * 2624).reshape(7, 2624).to(torch.bfloat16)
    salts = [bytes([i]) * HM96_SALT for i in range(7)]
    root, want = commit_y(y.view(torch.uint8).numpy().tobytes(), 2624, run.salt, 9, salts)
    got = y_tree(run, y, 9, torch.frombuffer(bytearray(b"".join(salts)), dtype=torch.uint8)).numpy().tobytes()
    assert got == want and got[-64:] == root
