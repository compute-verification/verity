"""The kernel mode of the top-p pin check (`topp_pin_check.kernel`), whole, on a CPU.

On a GPU the mode launches vLLM's `_apply_topp_split`.  Here that is a stand-in on `tests.fake_vllm`'s module, which picks its
split count by vLLM's doubling loop and keeps a set of lanes that depends on the count, as vLLM's kernels do; the device is
the CPU.  The pin is the real one (`engine.topp_pin`), and so are the rows (`topp_split_probe.gen_ties`) and the batch sizes.
So these cases show what the mode launches and what it makes of the answers: its rows, each batch size in both orders under
both arms, the lone row each row is held to, the verdict and the file.  That vLLM's kernels keep one nucleus at one split
count is what the mode says when a GPU runs it.
"""
from __future__ import annotations

import json
import types
from typing import Any

import pytest

pytest.importorskip("torch")

import numpy as np
import torch

from tests import fake_vllm
from verity_vllm.engine import topp_pin
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import topp_pin_check as tc
from verity_vllm.pipeline import topp_split_probe as probe

V, K, HEAD, ROWS = 128256, 24, 8, 64
SEED = 20260920 + 99991
PLATEAU = 3.5
GPU, VERSION = "a GPU, device 0", "0.0-stand-in"
#: The L40S's SM count, its batch sizes (the drain's, and 35 and 36, where vLLM's S goes from 4 to 2) and vLLM's S at each.
L40S = 142
L40S_SPLITS = {64: 2, 36: 2, 35: 4, 18: 4, 17: 8, 9: 8, 8: 16, 7: 16, 6: 16, 5: 16, 4: 32, 3: 32, 2: 32, 1: 32}

_made: dict[tuple[Any, ...], tuple[np.ndarray, float]] = {}


@pytest.fixture(autouse=True)
def _no_pin_before_or_after():
    topp_pin.uninstall()
    yield
    topp_pin.uninstall()


def _kept(s: int) -> int:
    """How many lanes the stand-in keeps of a row at split count ``s``: its head, and the first ``s`` of its plateau."""
    return HEAD + min(s, K)


class _Case:
    """The mode on a device of ``num_sm`` SMs.  The stand-in for `_apply_topp_split` keeps a row's lanes above the plateau
    and the plateau's first S, S by vLLM's loop to ``max_splits``; ``short`` names, by batch size, the places in a batch
    whose row keeps one lane fewer, which no pin mends."""

    def __init__(self, monkeypatch, num_sm: int = L40S, *, max_splits: int = 32, short: dict[int, list[int]] | None = None) -> None:
        self.max_splits, self.short = max_splits, short or {}
        self.log: list[tuple[Any, ...]] = []
        self.rows: list[tuple[Any, ...]] = []
        self.row_of: dict[float, int] = {}
        fake = fake_vllm.install(monkeypatch)
        fake["vllm"].__version__ = VERSION
        self.site = fake[topp_pin.SITE]
        self.site._SPLIT_MAX_SPLITS = max_splits
        self.site._apply_topp_split = self._split
        made = probe.gen_ties

        def gen_ties(seed: int, vocab: int, splits: int, k: int, keep_frac: float) -> tuple[np.ndarray, float]:
            args = (seed, vocab, splits, k, keep_frac)
            if args not in _made:
                _made[args] = made(*args)
            self.row_of[_made[args][1]] = len(self.rows)
            self.rows.append(args)
            return _made[args]

        def properties(device: int) -> types.SimpleNamespace:
            assert device == 0
            return types.SimpleNamespace(multi_processor_count=num_sm)

        cpu = torch.device("cpu")

        def device(name: str) -> torch.device:
            self.log.append(("device", name))
            return cpu

        monkeypatch.setattr(probe, "gen_ties", gen_ties)
        monkeypatch.setattr(torch, "device", device)
        monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: f"a GPU, device {device}")
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: self.log.append(("sync",)))

    def _split(self, logits: torch.Tensor, k: Any, p: torch.Tensor, mask_value: float, num_sm: int) -> None:
        batch, s = int(logits.shape[0]), 1
        while s < self.max_splits and s * 2 * batch <= num_sm:
            s *= 2
        self.log.append(("launch", batch, s, k, mask_value, [self.row_of[float(x)] for x in p]))
        plateau = logits == PLATEAU
        budget = torch.full((batch, 1), min(s, K), dtype=torch.int32)
        budget[self.short.get(batch, [])] -= 1
        keep = (logits > PLATEAU) | (plateau & (plateau.cumsum(1, dtype=torch.int32) <= budget))
        logits.masked_fill_(~keep, mask_value)

    def launches(self) -> list[tuple[int, int, list[int]]]:
        """Each launch's batch size, split count and rows, in order: nothing but launches, each waited for, after the device
        is named, and every one with no top-k and -inf for what it drops."""
        assert self.log[0] == ("device", "cuda") and self.log[2::2] == [("sync",)] * (len(self.log) // 2), self.log[:6]
        assert all(e[0] == "launch" and e[3] is None and e[4] == float("-inf") for e in self.log[1::2])
        return [(e[1], e[2], e[5]) for e in self.log[1::2]]


def _doc(num_sm: int, registered: int, splits: dict[int, int]) -> dict:
    """What the mode writes where no row moves under the pin: ``splits`` is vLLM's own S at each batch size, largest first."""
    batches = [{"B": B, "order": order,
                "pinned": {"S": registered, "rows_moved": 0, "kept": [_kept(registered)] * min(B, 4)},
                "vllm": {"S": s, "rows_moved": 0 if s == registered else B, "kept": [_kept(s)] * min(B, 4)}}
               for B, s in splits.items() for order in ("in order", "reversed")]
    return {"ok": True, "num_sm": num_sm, "registered_splits": registered, "V": V, "rows": ROWS, "gpu": GPU, "vllm": VERSION,
            "vllm_moved_some_row": any(s != registered for s in splits.values()), "batches": batches}


def _launches(registered: int, splits: dict[int, int]) -> list[tuple[int, int, list[int]]]:
    """Every row alone under the pin, then each batch size in order and reversed, the pin's arm before vLLM's own."""
    want = [(1, registered, [r]) for r in range(ROWS)]
    for B, s in splits.items():
        for idx in (list(range(B)), list(range(B))[::-1]):
            want += [(B, registered, idx), (B, s, idx)]
    return want


def _command(out) -> int | str | None:
    with pytest.raises(SystemExit) as e:
        tc.main(cli.parse("topp-pin-check", ["kernel", "--out", str(out)], {}))
    return e.value.code


def test_a_pin_that_holds_passes_on_the_drains_rows_each_alone_then_in_every_batch_under_both_arms(tmp_path, monkeypatch, capsys):
    case = _Case(monkeypatch)
    assert _command(tmp_path) == 0
    want = _doc(L40S, 32, L40S_SPLITS)
    assert want["vllm_moved_some_row"] is True and len(want["batches"]) == 28
    assert (tmp_path / "kernel.json").read_text() == json.dumps(want, indent=1) + "\n"
    assert capsys.readouterr().out.splitlines() == [json.dumps(b) for b in want["batches"]]
    assert case.rows == [(SEED + b, V, 16 if b % 2 == 0 else 32, K, 0.5) for b in range(ROWS)] and len(case.row_of) == ROWS
    assert case.launches() == _launches(32, L40S_SPLITS)
    assert topp_pin.pinned() and case.site._apply_topp_split.verity_topp_pin is True


@pytest.mark.parametrize("short", [{64: [0]}, {17: [16], 3: [2], 2: [0, 1]}])
def test_a_row_that_a_batch_moves_under_the_pin_fails_whichever_batch_it_is(tmp_path, monkeypatch, short):
    _Case(monkeypatch, short=short)
    assert _command(tmp_path) == 1
    want = _doc(L40S, 32, L40S_SPLITS)
    want["ok"] = False
    for b in want["batches"]:
        places = short.get(b["B"], [])
        for arm in ("pinned", "vllm"):
            if b[arm]["S"] == 32:
                b[arm]["rows_moved"] = len(places)
            for at in places:
                if at < 4:
                    b[arm]["kept"][at] -= 1
    assert json.loads((tmp_path / "kernel.json").read_text()) == want


def test_a_device_on_which_vllm_picks_the_registered_s_at_every_batch_moves_no_row(tmp_path, monkeypatch):
    case = _Case(monkeypatch, 4096)
    splits = dict.fromkeys((64, 18, 17, 9, 8, 7, 6, 5, 4, 3, 2, 1), 32)
    doc = tc.kernel(tmp_path)
    assert doc == json.loads((tmp_path / "kernel.json").read_text()) == _doc(4096, 32, splits)
    assert doc["ok"] is True and doc["vllm_moved_some_row"] is False
    assert case.launches() == _launches(32, splits)


def test_the_registered_s_is_the_devices(tmp_path, monkeypatch):
    case = _Case(monkeypatch, 16)
    splits = {64: 1, 18: 1, 17: 1, 9: 1, 8: 2, 7: 2, 6: 2, 5: 2, 4: 4, 3: 4, 2: 8, 1: 16}
    assert tc.kernel(tmp_path) == _doc(16, 16, splits)
    assert case.launches() == _launches(16, splits)


def test_vllms_own_s_is_its_loop_to_the_largest_split_count_its_module_states(tmp_path, monkeypatch):
    case = _Case(monkeypatch, max_splits=8)
    splits = {B: min(s, 8) for B, s in L40S_SPLITS.items()}
    doc = tc.kernel(tmp_path)
    assert {b["B"]: b["vllm"]["S"] for b in doc["batches"]} == splits
    assert case.launches() == _launches(8, splits)
