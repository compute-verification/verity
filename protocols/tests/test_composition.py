"""The three protocols composed by `protocols/example.py`, honest and cheating."""

import importlib.util
import sys
from pathlib import Path

import pytest

import verity_pouw
from verity_pous import P
import verity_sampled_proofs

_spec = importlib.util.spec_from_file_location("protocols_example", Path(__file__).parents[1] / "example.py")
example = sys.modules["protocols_example"] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example)

N = 64


def epoch(tamper=None, noise_keys=None, device=None):
    dep = example.deploy(N, filler_sectors=64)
    if device:
        device(dep)
    return dep, example.serve(dep, tamper=tamper, noise_keys=noise_keys)


def test_honest_epoch_accounts_work_and_space_with_delta_counted_once():
    dep, ev = epoch()
    report = example.audit(dep, ev)
    a = report["allowance"]
    assert (a.n_r, a.n_v) == (N, example.M)
    assert all(w.forgeable == int(report["incorrect_vus_max"]) for w in report["work"])
    assert all(report["composed_macs"][w.device] >= w.macs for w in report["work"])
    assert all(s.accounted > 0.95 * s.declared for s in report["space"])
    forgery = sum(report["audits"].values()) / P
    assert report["failure"] == pytest.approx(a.delta + 2 * dep.pouw.eta + 2 * dep.pous.delta + forgery)


def _tile_outputs(dep, rus):
    return [dep.sp.vus[i][0].returns[1] for i in rus]


def test_fabricated_noised_outputs_are_caught_by_sampled_proofs():
    def tamper(dep, values):
        for g in _tile_outputs(dep, range(N)):
            values[g] ^= 1
    dep, ev = epoch(tamper)
    with pytest.raises(verity_sampled_proofs.Rejected, match="replay"):
        example.audit(dep, ev)


def test_noise_from_the_wrong_device_key_is_caught_through_prescribed_inputs():
    dep = example.deploy(N, filler_sectors=64)
    swapped = {"gpu0": dep.keys["gpu1"], "gpu1": dep.keys["gpu0"]}
    with pytest.raises(verity_sampled_proofs.Rejected, match="prescribed"):
        example.audit(dep, example.serve(dep, noise_keys=swapped))


def test_one_unsampled_incorrect_unit_passes_and_stays_inside_the_allowance():
    for r in range(N):
        dep, ev = epoch(lambda dep, values, r=r: values.__setitem__(g := _tile_outputs(dep, [r])[0], values[g] ^ 1))
        try:
            report = example.audit(dep, ev)
        except verity_sampled_proofs.Rejected:
            continue
        assert report["incorrect_vus_max"] >= 1  # the claim already charged for this one
        return
    pytest.fail("every single-unit corruption was sampled; the law is not the configured one")


def test_a_ticket_for_an_uncommitted_value_is_rejected_by_pouw():
    dep, ev = epoch()
    tk = ev.tickets[0]
    forged = type(tk.opening)(tk.opening.position, bytes([tk.opening.value[0] ^ 1, tk.opening.value[1]]), tk.opening.path)
    ev.tickets[0] = type(tk)(tk.request, tk.gate, forged)
    with pytest.raises(verity_pouw.Rejected):
        example.audit(dep, ev)


def test_pous_runs_alongside_and_only_the_offloading_device_loses_its_space():
    honest = example.audit(*epoch())
    dep, ev = epoch(device=lambda dep: dep.devices["gpu1"].offloaded.update(("filler", s) for s in range(64)))
    report = example.audit(dep, ev)
    assert set(ev.audits["gpu1"]) == {"late"}
    by_device = {(s.device, s.region): s.accounted for s in report["space"]}
    assert by_device[("gpu1", "filler")] == 0 and by_device[("gpu0", "filler")] > 0
    assert report["work"] == honest["work"]
