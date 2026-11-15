"""The largest-tensor bound a window ring reports (`verity_vllm.commit.committer.native_collect`,
`NativeCollectCommitter.enable_windows`), read from an arena sizing record: its prefill hidden bound over the model's layers, or its
decode step's bytes where those are more. A record may leave each of the three out, give it null or give it zero, and the bound
is then what the others give: a run with no record reports none."""
from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("torch")

from tests.commit import test_native_collect_plan_cache as PC

RECORDS: list[tuple[str, dict[str, Any] | None, int]] = [
    ("no record", None, 0),
    ("an empty record", {}, 0),
    ("no model", {"prefill_hidden_bytes": 4096}, 4096),
    ("a null model", {"model": None, "prefill_hidden_bytes": 4096}, 4096),
    ("an empty model", {"model": {}, "prefill_hidden_bytes": 4096}, 4096),
    ("a model with null layers", {"model": {"layers": None}, "prefill_hidden_bytes": 4096}, 4096),
    ("a model with no layers", {"model": {"layers": 0}, "prefill_hidden_bytes": 4096}, 4096),
    ("a model with a negative layer count", {"model": {"layers": -2}, "prefill_hidden_bytes": 4096}, 4096),
    ("four layers", {"model": {"layers": 4}, "prefill_hidden_bytes": 4096}, 1024),
    ("four layers as text", {"model": {"layers": "4"}, "prefill_hidden_bytes": 4096}, 1024),
    ("a decode step above a layer's share", {"model": {"layers": 4}, "prefill_hidden_bytes": 4096, "decode_step_bytes": 2000}, 2000),
    ("a decode step below a layer's share", {"model": {"layers": 4}, "prefill_hidden_bytes": 4096, "decode_step_bytes": 1000}, 1024),
    ("null bounds", {"model": {"layers": 4}, "prefill_hidden_bytes": None, "decode_step_bytes": None}, 0),
    ("a model alone", {"model": {"layers": 4}}, 0),
    ("a decode step alone", {"decode_step_bytes": 512}, 512),
]


@pytest.mark.parametrize(("rec", "want"), [(rec, want) for _, rec, want in RECORDS], ids=[name for name, _, _ in RECORDS])
def test_the_ring_reports_the_records_largest_tensor_bound(monkeypatch: pytest.MonkeyPatch, rec: dict[str, Any] | None,
                                                           want: int) -> None:
    com = PC.committer(monkeypatch)
    info = com.enable_windows(rec)
    assert info["largest_tensor_bound"] == want
    assert info["slots"] == max(PC.SLOTS, 3) and info["windowed"] is True          # the ring itself does not depend on the record


def test_a_model_that_is_no_mapping_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    com = PC.committer(monkeypatch)
    with pytest.raises(AttributeError):
        com.enable_windows({"model": [4], "prefill_hidden_bytes": 4096})
