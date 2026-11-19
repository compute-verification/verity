"""A served window's record made at the serve's close (`close_record.Record.make`), on the processor."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from verity.commitments import docket_digest as DD

from tests.program.predict.test_window_rows import (
    CLASS,
    GRAIN,
    LANES,
    RUN,
    _located,
    _read,
    _registered,
    _served,
    _window,
)
from verity_vllm.pipeline import advice as ADV
from verity_vllm.pipeline import advice_windows as AW
from verity_vllm.program.predict import scan_state as SS
from verity_vllm.program.predict import window_plan as WP
from verity_vllm.program.predict import window_rows as WR
from verity_vllm.program.predict import window_weights as WW
from verity_vllm.program.predict.close_record import Record


def _committer(held):
    """A committer as the close reads it: each step's members in one uint8 block, each at its offset (``dev_off``)."""
    layouts, blocks = {}, {}
    for (step, name), a in sorted(held.items()):
        b = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
        at = sum(x.size for x in blocks.get(step, []))
        layouts.setdefault(step, []).append(SimpleNamespace(name=name, ordinal=0, dev_off=at, stream_off=-1))
        blocks.setdefault(step, []).append(b)
    return SimpleNamespace(_layouts=layouts, _gpu_blocks={s: (torch.from_numpy(np.concatenate(bs)),) for s, bs in blocks.items()})


@pytest.mark.slow
def test_the_serves_close_records_its_window_with_strings_from_where_its_rows_are_committed(tmp_path, monkeypatch):
    """`Record.make` over TINY's window at class 2,4,2: its slots placed from the window's bodies, their served values and
    staged rows read from a committer holding each member at its offset in its step's block, the scan state the stand-in
    its slots' rows give, the window recorded, closed and written under ``windows/w0/record/``, its strings' cost and the
    twin's comparison on the sampled ports returned. Stood in: the committer, the checkpoint's words (`record_words`),
    the weights registration (every weight unrooted) and the advice opening. What no cheaper test reaches: the serve's own
    path from its close's plan to a written record. Slow (minutes) for the close."""
    low, fields, _values, _weights, slots = _window()
    served, held = [], {}
    for k, s in enumerate(slots):
        rows, store = _served(low, s, T=max(WR.positions(low.programs[k].fn)) + 1 - s.row[1])
        held.update({(step, f"s{k}:{op}/{m}"): a for (step, op, m), a in store.items()})
        served.append(WR.Served(rows, *_read(store), locate=_located(store, f"s{k}:")))
    units = WP.Plan(low, served)
    ids = [f"q{k}" for k in range(len(LANES))]
    bodies = {i: {"prompt": p[:lp], "max_tokens": m} for i, (_o, lp, m, p) in zip(ids, LANES)}
    bound = {"compiler": {"name": "c", "digest": "cd" * 64}, "advice": {"root": "ef" * 64, "bits": 0}}
    weight_values = WW.weight_values
    monkeypatch.setattr(ADV, "bodies", lambda out, window: bodies)
    monkeypatch.setattr(AW, "opened", lambda compiler, doc, codec=None: (None, bound))
    monkeypatch.setattr(WW, "record_words", lambda *a, **k: fields)
    monkeypatch.setattr(WW, "weight_values", lambda r, s, _reg, keep_rows=False: weight_values(r, s, None, GRAIN, keep_rows=keep_rows))
    (tmp_path / "advice" / "w0" / "private").mkdir(parents=True)
    (tmp_path / "advice" / "w0" / "private" / "opening.json").write_text("{}")
    plan = SimpleNamespace(cls=CLASS, out=tmp_path, compiler=None, _registered=_registered)
    binding = SimpleNamespace(rc=SimpleNamespace(run_root=bytes.fromhex(RUN)), com=_committer(held))
    compiled = SimpleNamespace(requests=[SimpleNamespace(id=i) for i in ids])

    got = Record("rate:1/4", "unread", tmp_path / "salt_stream", sample=2, workers=2).make(
        plan, 0, compiled, low, units, served, binding)
    to = tmp_path / "windows" / "w0" / "record"
    assert got["out"] == str(to) and got["run"] == RUN and got["population"] > 0
    s = got["strings"]
    assert s["where"] == "HostTwin" and s["acquired"] > 0 and s["acquired"] + s["host"] > s["acquired"]
    assert s["twin"]["equal"] and s["twin"]["ports"] == min(2, s["twin"]["of"]) > 0
    reg = json.loads((to / "registration.json").read_text())
    assert SS.SCAN_STATE in [e["value"] for e in reg["registered"]] and DD.digest(reg)
    assert {"rows", "record", "close", "write"} <= set(got["seconds"])
