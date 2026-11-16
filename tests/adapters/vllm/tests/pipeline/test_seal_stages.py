"""The two stages that seal serve-openai's closed windows off the engine loop (`verity_vllm.pipeline.seal_stages.Sealer`): the
threads it starts and ends, the text it keeps of a failure, and what its summary shows of them. What the stages do with a
window's job and its seal is in `tests/pipeline/test_serve_offloop.py`."""
from __future__ import annotations

import threading

from verity_vllm.pipeline import seal_stages as SS


def test_the_two_stages_are_daemon_threads_named_for_serving_that_run_until_the_close():
    sealer = SS.Sealer(8)
    stages = (sealer.thread, sealer.seal_thread)
    assert [t.name for t in stages] == ["serve-seal", "serve-seal-2"]
    assert all(t.daemon and t.is_alive() for t in stages), "a daemon thread does not hold a server that is ending"
    sealer.drain()
    assert all(t.is_alive() for t in stages), "a drain leaves both stages running for the next window"
    sealer.close()
    assert not any(t.is_alive() for t in stages)


def test_a_failure_is_kept_as_its_class_and_300_characters_and_the_summary_shows_the_first_ten_of_them():
    def fail(i, _waited):
        raise ValueError(f"{i}:" + "x" * 400)

    sealer = SS.Sealer("64")
    assert sealer.summary() == {"hold_bytes": 64, "held": 0, "errors": [], "errors_n": 0}
    for i in range(12):
        sealer.submit(lambda waited, i=i: fail(i, waited), 0)
    sealer.drain()
    kept = [(f"ValueError: {i}:" + "x" * 400)[:300] for i in range(12)]
    assert sealer.errors == kept and {len(text) for text in kept} == {300}
    assert sealer.summary() == {"hold_bytes": 64, "held": 0, "errors": kept[:10], "errors_n": 12}
    sealer.close()


def test_the_summary_shows_the_bytes_held_by_a_window_whose_seal_has_not_run():
    sealing, go = threading.Event(), threading.Event()

    def seal():
        sealing.set()
        go.wait()

    sealer = SS.Sealer(100, clock=lambda: 5.0)
    assert sealer.submit(lambda _waited: seal, 40.9) == 0.0
    sealing.wait()
    assert sealer.summary() == {"hold_bytes": 100, "held": 40, "errors": [], "errors_n": 0}
    go.set()
    sealer.close()
    assert sealer.summary() == {"hold_bytes": 100, "held": 0, "errors": [], "errors_n": 0}
