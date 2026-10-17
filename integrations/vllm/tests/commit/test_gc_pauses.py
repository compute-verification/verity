"""The committer's collection monitor (`native_host.GcPauses`): while installed, each Python collection of at least `min_s` is a pause
(generation, start, seconds, thread) that a run's finalize carries, so a step's tail can be laid against the collections that stalled it."""

from __future__ import annotations

import gc

from verity_vllm.commit.committer.native_host import GcPauses


def test_a_collection_while_installed_is_a_pause_on_its_thread_and_none_after_uninstall():
    g = GcPauses(min_s=0.0)
    g.install()
    g.install()
    try:
        assert gc.callbacks.count(g) == 1
        gc.collect()
    finally:
        g.uninstall()
    assert g not in gc.callbacks
    gen, _, seconds, thread = g.pauses[-1]
    assert (gen, thread) == (2, "MainThread") and seconds >= 0.0
    rec = g.record()
    assert rec["min_s"] == 0.0 and rec["total_s"] == sum(p[2] for p in g.pauses) and rec["pauses"] == g.pauses
    n = len(g.pauses)
    gc.collect()
    assert len(g.pauses) == n


def test_collections_shorter_than_the_floor_are_not_pauses():
    g = GcPauses(min_s=3600.0)
    g.install()
    try:
        gc.collect()
    finally:
        g.uninstall()
    assert g.record() == {"min_s": 3600.0, "pauses": [], "total_s": 0}
