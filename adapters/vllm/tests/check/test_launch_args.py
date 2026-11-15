"""`verity_vllm.check.replay.launch_args` on snapshots written here and read by the capture's own store: a view's words as the fp32
of their values under the dtype the view states (`as_f32`), a launch's small argument from the snapshot taken at the launch or else
the latest before it (`small_arg`), and an integer argument as int64 values (`small_int_arg`)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from verity_vllm.check.fold_compare import SnapshotStore
from verity_vllm.check.replay.launch_args import as_f32, small_arg, small_int_arg
from verity_vllm.observe.events import TritonLaunch, ViewDesc

VALUES = np.array([1.5, -0.375, 0.0, 2.0 ** -10, -np.inf], dtype=np.float32)      # each exact in all four floating dtypes


def _view(shape: tuple[int, ...], dtype: str) -> ViewDesc:
    return ViewDesc(7, 0, shape, (shape[1], 1) if len(shape) == 2 else (1,), dtype, "cuda:0")


def _store(root: Path, snapshots: list[tuple[int, ViewDesc, np.ndarray]]) -> SnapshotStore:
    """A capture's store under `root` holding `snapshots`: (the event that took it, its view, its words in row-major order)."""
    entries = []
    for k, (event, view, words) in enumerate(snapshots):
        words.tofile(root / f"{k}.bin")
        entries.append({"event": event, "arg": "x", "view": view.to_json(), "file": f"{k}.bin"})
    (root / "snapshots.json").write_text(json.dumps({"layout": "logical-row-major", "entries": entries}))
    return SnapshotStore(root)


def _launch(event: int, args: dict) -> TritonLaunch:
    return TritonLaunch(id=event, parent=None, step=0, phase="sample", fn="k", src_sha="", args=args, constexprs={}, grid=[1],
                        num_warps=None, num_stages=None, stream=0)


@pytest.mark.parametrize("dtype, words", [
    ("bfloat16", (VALUES.view(np.uint32) >> 16).astype(np.uint16)),
    ("float16", VALUES.astype(np.float16).view(np.uint16)),
    ("float32", VALUES.view(np.uint32)),
    ("float64", VALUES.astype(np.float64).view(np.uint64)),
])
def test_a_views_words_are_read_as_the_fp32_of_their_values_under_its_dtype(dtype, words):
    got = as_f32(words, dtype)
    assert got.dtype == np.float32 and got.tobytes() == VALUES.tobytes()


def test_words_of_a_dtype_that_is_not_floating_are_refused_by_its_name():
    with pytest.raises(ValueError, match="^int32$"):
        as_f32(np.zeros(2, dtype=np.uint32), "int32")


def test_a_small_argument_is_read_at_its_launch_or_else_from_the_latest_snapshot_before_it(tmp_path):
    view = _view((2,), "float32")
    store = _store(tmp_path, [(event, view, np.array([event, -event], dtype=np.float32)) for event in (3, 5, 9)])

    def read(event: int, name: str = "t", arg: object = view):
        return small_arg(store, _launch(event, {"t": arg}), name, np.float32)

    assert read(5).tolist() == [5.0, -5.0] and read(9).tolist() == [9.0, -9.0]            # the launch's own
    assert read(4).tolist() == [3.0, -3.0] and read(8).tolist() == [5.0, -5.0]            # none at the launch: the latest before it
    assert read(2) is None                                                                # one taken after the launch is not its argument
    assert read(5, name="absent") is None and read(5, arg=0.5) is None                    # a name it lacks, a value that is no view


def test_a_small_argument_comes_flat_as_fp32_values_or_as_the_words_it_holds(tmp_path):
    view = _view((2, 2), "bfloat16")
    words = (VALUES[:4].view(np.uint32) >> 16).astype(np.uint16)
    launch = _launch(5, {"t": view})
    store = _store(tmp_path, [(5, view, words)])
    values = small_arg(store, launch, "t", np.float32)
    assert values.dtype == np.float32 and values.tolist() == VALUES[:4].tolist()
    raw = small_arg(store, launch, "t", None)
    assert raw.dtype == np.uint16 and raw.tolist() == words.tolist()


@pytest.mark.parametrize("dtype, held", [("int64", np.int64), ("int32", np.int32), ("uint32", np.uint32), ("int16", np.int16),
                                         ("uint8", np.uint8)])
def test_an_integer_argument_is_read_as_int64_values_under_its_dtype(tmp_path, dtype, held):
    values = np.array([0, 1, np.iinfo(held).min, np.iinfo(held).max], dtype=held)
    view = _view((4,), dtype)
    got = small_int_arg(_store(tmp_path, [(5, view, values)]), _launch(5, {"n": view}), "n")
    assert got.dtype == np.int64 and got.tolist() == values.tolist()


def test_a_floating_argument_has_no_integer_reading_and_neither_has_one_the_launch_lacks(tmp_path):
    view = _view((2,), "float32")
    store = _store(tmp_path, [(5, view, VALUES[:2])])
    launch = _launch(5, {"n": view})
    assert small_arg(store, launch, "n", None) is not None
    assert small_int_arg(store, launch, "n") is None and small_int_arg(store, launch, "absent") is None
