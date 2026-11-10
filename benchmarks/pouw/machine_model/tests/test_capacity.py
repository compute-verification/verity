"""`machine-model capacity`: the certified share σ of a window's capacity, read on the pooled certified load."""
from __future__ import annotations

import json

import pytest

from verity_machine_model import capacity as C
from verity_machine_model import eps as E
from verity_machine_model.cli import main
from verity_machine_model.model import device

VEC = "VF3n-16"


@pytest.fixture(scope="module")
def pi():
    return E.machine(VEC)


def test_the_empty_gpu_certifies_nothing_and_offers_a_full_tensor_pipe(pi):
    x = C.cell(pi, [(1.0, C.Work("empty", {}))])
    assert x.sigma() == pytest.approx(0.0) and x.credit() == 0.0
    assert x.pooled.mix[C.MMA] == pytest.approx(1 / C.TENSOR_CYCLES)
    assert x.empty > 1 / C.TENSOR_CYCLES  # the scalar pipes add useful work beside the tensor pipe


@pytest.mark.parametrize("dev", ["sm_80", "sm_89", "sm_90", "sm_100"])
def test_every_devices_empty_gpu_offers_its_own_tensor_pipe(dev):
    """USEFUL names sm_120's MMA only: on another device the hidden program must still find that device's tensor class,
    or σ ignores the tensor pipe (sm_100's empty GPU read 0.0195 slots a SMSP-cycle, its lanes, against 0.52)."""
    data = device(dev)
    pi_d = E.machine("", data)
    cls = {E.honest(p, data).cls for p in E.profiles(data)}
    assert C.MMA not in cls
    u = C.useful_of(data)
    x = C.cell(pi_d, [(1.0, C.Work("empty", {}))], u)
    tensor = sum(u[k] * v for k, v in x.pooled.mix.items() if k in cls)
    assert tensor > 0.75 * x.empty and x.empty > 3 * C.hidden(pi_d, C.Work("empty", {})).hidden


def test_useful_of_sm120_is_useful():
    assert C.useful_of() == C.USEFUL


def test_the_tensor_pipe_alone_leaves_the_scalar_pipes_open(pi):
    full = C.cell(pi, [(1.0, C.tensor_filler(1.0, "none"))])
    assert full.tensor() == pytest.approx(1.0) and full.credit() == pytest.approx(1.0)
    assert 0.7 < full.sigma() < 0.85 and C.MMA not in full.pooled.mix


def test_certified_lane_chains_that_saturate_issue_leave_nothing(pi):
    x = C.cell(pi, [(1.0, C.lane_filler(1.0))])
    assert x.sigma() == pytest.approx(1.0) and "issue" in [r for r, v in x.pooled.load.items() if v >= 1 - 1e-9]


def test_sigma_grows_with_certified_work(pi):
    s = [C.cell(pi, [(1.0, C.tensor_filler(0.6, "duo") + C.lane_filler(x))]).sigma() for x in (0, 0.1, 0.2, 0.3)]
    assert s == sorted(s) and s[0] == pytest.approx(0.6411, abs=1e-4) and s[-1] > 0.95


def test_the_pooled_reading_never_credits_more_than_an_attested_placement(pi):
    groups = [(0.75, C.tensor_filler(0.6, "duo") + C.lane_filler(0.3)), (0.25, C.served(0.05))]
    x = C.cell(pi, groups)
    assert x.sigma() <= x.sigma_placed() + 1e-12
    assert x.sigma() < x.sigma_placed() - 0.01  # here the placement matters: the step SMs' open MMAs move
    same = C.cell(pi, [(0.5, C.tensor_filler(0.6, "duo")), (0.5, C.tensor_filler(0.6, "duo"))])
    assert same.sigma() == pytest.approx(same.sigma_placed())


def test_a_load_over_a_row_is_reported_not_credited(pi):
    h = C.hidden(pi, C.tensor_filler(1.0, "duo"))
    assert h.over and h.hidden == 0.0


def test_the_shares_of_the_cell_must_sum_to_one(pi):
    with pytest.raises(SystemExit, match="sum to 0.5"):
        C.cell(pi, [(0.5, C.lane_filler(0.1))])


def test_parse_reads_each_term_as_its_constructor():
    w = C.parse("duo:0.6+lane:0.3+served:0.16:xin13+IMAD=0.01")
    m = 0.6 / 16
    assert w.n[C.MMA] == pytest.approx(m + 0.01)
    assert w.n["ALU"] == pytest.approx(m * 8 + 0.15 + 0.01 * 4)
    assert w.n["IMAD"] == pytest.approx(m * 4 + 0.15 + 0.01 * 2 + 0.01)
    assert C.parse("xin13:0.8*2").n["ALU"] == pytest.approx(0.05 * 4 * 2)
    assert C.parse("arx:0.8").n == pytest.approx({C.MMA: 0.05, "ALU": 0.05 * 4})
    with pytest.raises(SystemExit, match="can't read"):
        C.parse("duo")


def test_the_cli_reports_the_pooled_sigma(capsys):
    assert main(["capacity", "--vector", VEC, "--json", "0.745/duo:0.6+lane:0.3", "0.255/served:0.15+lane:0.6"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["vector"] == VEC and len(d["groups"]) == 2 and d["binding"] == ["JI3n_0"]
    assert d["sigma"] == pytest.approx(0.8679, abs=1e-4) and d["sigma"] <= d["sigma_placed"] + 1e-12
    assert main(["capacity", "--vector", VEC, "duo:0.6"]) == 0
    assert "σ 0.6411" in capsys.readouterr().out
