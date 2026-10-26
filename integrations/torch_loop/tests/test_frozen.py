"""The frozen H200 shape's decode step (`frozen`), on a tiny shape of the same code: its units are its templates' functions
(and a tampering step's are not), one window per template each with its own root, the circuits it registers before running
are the ones it writes and core's registration check accepts each, the f64 linear is `loop.linear` at the extremes, and the
W report's bounds are core's."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from verity.core.protocols.profile import Stratified, Stratum
from verity.core.protocols.sampled_proofs.one_stage import draw as D, registration as R
from verity.ml.windows import interface as IF, rows as CS
from verity_torch_loop import frozen as F, frozen_w as FW, loop as L, serve_frozen as S

SHAPE = ["--small", "--law", "subset:4", "--cap", "64", "--cap-of", "s8-mlp=16", "--position", "3"]


def test_the_steps_units_are_its_templates_and_a_tampering_steps_are_not():
    for tamper in (False, True):
        step = F.FrozenStep(F.Model(S.SMALL, 2), b"abcd", 3, tamper=tamper, check=10 ** 6)
        wins = step.windows(D.parse_law("subset:4"), 64)
        assert {t: w.window.population for t, w in wins.items()} == step.units == F.units(S.SMALL, 3, 2)
        assert step.checked == step.units
        assert step.mismatched == ({t: n for t, n in step.units.items()} if tamper else {t: 0 for t in F.TAGS})
        assert len({step.run_root(t) for t in F.TAGS}) == 3


def test_keep_records_the_first_units_of_each_op():
    step = F.FrozenStep(F.Model(S.SMALL, 1), b"ab", 1, keep=8)
    wins = step.windows(D.parse_law("subset:4"), 64)
    assert {t: w.window.population for t, w in wins.items()} == F.units(S.SMALL, 1, 1, keep=8) == {
        "s8-width": 8 * 5 + 8, "s8-mlp": 8, "bf16-dot": 4 + 8}


def test_the_circuits_it_registers_before_running_are_the_ones_it_writes_and_core_accepts_each(tmp_path, capsys):
    assert S.main(["--circuit-only", *SHAPE]) == 0
    before = json.loads(capsys.readouterr().out)
    assert S.main(["--out", str(tmp_path), *SHAPE]) == 0
    got = json.loads(capsys.readouterr().out)
    assert set(got["windows"]) == set(before) == set(F.TAGS) and set(got["mismatched"].values()) == {0}
    for tag, w in got["windows"].items():
        d = tmp_path / tag
        assert w["circuit"] == before[tag] == hashlib.sha512((d / "circuit.json").read_bytes()).hexdigest()
        mine, _doc = IF.of_records(d, D.parse_law("subset:4"), IF.resolve)
        rec = json.loads((d / "registration.json").read_text())
        layer = CS.leaf_layer({m.key: (d / f"m{k}-public.bin").read_bytes() for k, m in enumerate(mine.members)})
        v = R.check(rec, mine.program, mine.query, mine.law, CS.SCHEME, CS.expected_ports(mine), leaf_layer=layer)
        assert v.ok, v.to_json()
        assert rec["window"]["run"] == w["run"] and mine.population == w["population"]
        assert w["members"] == len(mine.members) == -(-w["population"] // (16 if tag == "s8-mlp" else 64))


def test_the_f64_linear_is_the_loops_at_the_extremes():
    g = torch.Generator().manual_seed(1)
    for k in (128, 4096):
        wq = torch.randint(-128, 128, (64, k), generator=g, dtype=torch.int8)
        wq[0], wq[1] = -128, 127
        xq = torch.full((1, k), -128, dtype=torch.int8)
        sw = torch.rand(64, generator=g)
        sx = torch.tensor([0.37])
        assert torch.equal(F.linear(xq, sx, F.Linear(wq, sw)).view(torch.int16),
                           L.linear(xq, sx, wq, sw)[0].view(torch.int16))


def test_the_w_report_is_cores_stratified_bound_per_block_and_density_bound_pooled():
    sizes = {"a": 1000, "b": 300}
    got = FW.report(sizes, 16, [0.01], 20)
    law = Stratified((Stratum("a", 1000, 16), Stratum("b", 300, 16)))
    row = got["delta"]["0.01"]
    assert row["block"]["W"] == law.count_bound(0.01) and row["block"]["by_window"]["b"] == law.stratum_bound("b", 0.01)
    assert row["pooled"]["week"]["blocks"] == 504 and row["pooled"]["year"]["blocks"] == 26280
    assert row["pooled"]["week"]["W"] == row["pooled"]["year"]["W"] >= row["block"]["W"]
    assert row["pooled"]["year"]["block_at_delta_over_T"] == law.count_bound(0.01 / 26280)
    assert np.isclose(row["pooled"]["week"]["W"], 1000 * np.log(100) / 16)
