"""The strings a served window's close hands `Window.close`, computed as the card computes them, on the processor."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from verity.commitments import randomness as RND
from verity.commitments.cuda.salt_stream import engine as SE
from verity.commitments.periods import interface as IF
from verity.commitments.periods import rows as CS
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2

from tests.program.predict.test_window_rows import (
    CLASS,
    SLOTS,
    _located,
    _read,
    _served,
    _window,
)
from verity_vllm.program.predict import window_plan as WP
from verity_vllm.program.predict import window_rows as WR
from verity_vllm.program.predict import window_strings as WS


@pytest.fixture(scope="module")
def twin(tmp_path_factory):
    return WS.HostTwin(SE.host_lib(tmp_path_factory.mktemp("salt_stream")))


def _ports(low, filled):
    """Each output port as `Window.leaf_rows` gives it: its template's ordinal, its own after the template's input rows, and
    its units' rows in unit order."""
    c = low.circuit
    out = {}
    for t, (tid, fn) in enumerate(zip(c.ids, (fn for fn, _ in c.templates))):
        lay = IF.layout(fn)
        for o, (name, _w) in enumerate(lay.outs):
            rows = np.asarray(filled.outs[tid][o], dtype=np.uint16)
            out[(tid, name)] = SimpleNamespace(template=t, port=len(lay.rows) + o, schema=SCHEMA_HM96_SHA512_ROW_V2,
                                               units=list(range(len(rows))), rows=rows.reshape(len(rows), -1))
    return out


def _served_window():
    """The test window's ports (`_ports`), its plan, and its served members' bytes as a committer's blocks would hold them."""
    low, _f, tables, weights, slots = _window()
    served, got, held = [], [], {}
    for k, s in enumerate(slots):
        rows, store = _served(low, s, T=max(WR.positions(low.programs[k].fn)) + 1 - s.row[1])
        held.update({(step, f"s{k}:{op}/{m}"): a for (step, op, m), a in store.items()})
        served.append(WR.Served(rows, *_read(store), locate=_located(store, f"s{k}:")))
        got.append(WR.Slot(low.programs[k].fn, CLASS, SLOTS[k], weights, served[-1]))
    filled = WR.Rows(low, got, tables)
    blocks = {key: torch.from_numpy(np.ascontiguousarray(a).view(np.uint8).reshape(-1)) for key, a in held.items()}
    return _ports(low, filled), WP.Plan(low, served), lambda step, name: (blocks[(step, name)], 0)


def test_a_served_windows_strings_are_its_rows_strings_under_the_windows_key_where_acquired_or_not(twin):
    """Every leaf port's salts are the window key's stream at (template, port), leaf u for unit u, and its strings each row's
    ``b || c`` as the period computes them (`rows.row_strings`): the same whether a unit's row is hashed from where the serve
    committed it (`Plan.feeds` through `stage`) or on the host from its recorded row, and the cost counts each unit once."""
    ports, plan, block = _served_window()
    key = RND.derive(bytes(64), "verity/window-commit/salts/v1", {"window": "t"})
    staged = WS.acquired(plan, block, ports)
    assert 0 < sum(map(len, staged.values())) == plan.describe()["fed_rows"]
    fed, cost = WS.leaf_strings(ports, key, twin, {k: {u: r.numpy() for u, r in v.items()} for k, v in staged.items()})
    host, cost_host = WS.leaf_strings(ports, key, twin, {})
    total = sum(len(lr.units) for lr in ports.values())
    assert cost["acquired"] == plan.describe()["fed_rows"] and cost["acquired"] + cost["host"] == total == cost_host["host"]
    for k, lr in ports.items():
        n = len(lr.units)
        ys = key.stream((lr.template, lr.port), SE.LEAF_BYTES * n)
        assert fed[k] == host[k] == (ys, CS.row_strings(lr.rows, ys, lr.schema)), k


class _Flipped(WS.HostTwin):
    def strings(self, rows, lb, ys):
        out = super().strings(rows, lb, ys).copy()
        out[0, 0] ^= 1
        return out


def test_the_twin_recomputes_a_sample_of_ports_and_says_whether_the_hashers_strings_are_its_own(twin, capsys):
    """`strings_of` with a twin: the sampled ports, drawn among those with acquired units, are recomputed by the twin from host
    copies of the same rows, and the line it prints says equal, or names the ports whose strings differ."""
    ports, plan, block = _served_window()
    win = SimpleNamespace(leaf_rows=lambda: ports)
    key = RND.derive(bytes(64), "verity/window-commit/salts/v1", {"window": "t"})
    cost: dict = {}
    WS.strings_of(win, key, twin, plan, block, twin=twin, sample=3, cost=cost)
    fed = sum(1 for k in WS.acquired(plan, block, ports) if k in ports)
    assert cost["twin"]["equal"] and cost["twin"]["ports"] == min(3, fed) > 0 and cost["twin"]["of"] == fed
    assert "WINDOW-STRINGS" in capsys.readouterr().err
    bad = _Flipped(twin.lib)
    WS.strings_of(win, key, bad, plan, block, twin=twin, sample=fed, cost=cost)
    assert not cost["twin"]["equal"] and cost["twin"]["unequal"]


def test_a_committers_block_is_read_at_its_members_device_offset_else_its_stream_offset():
    blk = torch.zeros(16, dtype=torch.uint8)
    metas = [SimpleNamespace(name="a", ordinal=0, dev_off=-1, stream_off=8), SimpleNamespace(name="b", ordinal=0, dev_off=4,
                                                                                             stream_off=12)]
    block = WS.block_of(SimpleNamespace(_layouts={3: metas}, _gpu_blocks={3: (blk, None)}))
    assert block(3, "a")[1] == 8 and block(3, "b")[1] == 4 and block(3, "a")[0] is blk
    with pytest.raises(ValueError, match="commits no member c"):
        block(3, "c")


def test_a_port_of_a_schema_the_row_hash_doesnt_compute_is_refused_by_name(twin):
    lr = SimpleNamespace(template=0, port=0, schema="hm96-sha512/row/v1", units=[0], rows=np.zeros((1, 4), np.uint16))
    with pytest.raises(ValueError, match="is hm96-sha512/row/v1, and the row hash computes"):
        WS.leaf_strings({("t", "o0"): lr}, RND.derive(bytes(64), "verity/window-commit/salts/v1", {}), twin, {})
