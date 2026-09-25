"""The toy engine with all three protocols: phases, wiring shapes, contention, and cheating provers."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))

from toy_engine import demo  # noqa: E402
from toy_engine.engine import Conflict  # noqa: E402
import verity_pouw  # noqa: E402
import verity_sampled_proofs  # noqa: E402

CFG = demo.Config(n_requests=16)


def epoch(cfg=CFG, after_wiring=None):
    world = demo.setup(cfg)
    if after_wiring:
        after_wiring(world)
    demo.serve(world)
    return world


@pytest.fixture(scope="module")
def honest():
    world = epoch()
    return world, demo.audit(world)


def test_honest_epoch_runs_all_three_phases(honest):
    world, report = honest
    p = report["profile"]
    assert p.linked and p.prescribed == {"prompt", "kv-init", "weights", "noise"}
    assert p.n_r == sum(len(s.requests) for s in world.engine.steps) == report["policy"]["tokens"]
    assert {a[0] for a in world.audits} == {"accepted"}
    assert all(s.accounted > 0.95 * s.declared for s in report["space"].values())
    assert report["planned"] == world.pous.certified("filler", world.engine.cfg.yield_latency, len(world.audits))
    assert all(w.macs > 0 for w in report["work"])
    assert set(report["events"]) == {"integrity", "pous/audits/gpu0", "pouw/tickets/gpu0"}  # integrity once


@pytest.mark.parametrize("shape", ["phases", "subscriptions"])
def test_every_wiring_shape_gives_the_same_epoch(honest, shape):
    world, report = honest
    other = epoch(replace(CFG, wiring=shape))
    assert other.log.entries == world.log.entries and other.tickets == world.tickets
    assert demo.audit(other)["space"] == report["space"]


def test_sampling_at_audit_retains_every_step_and_does_not_fit_the_workspace(honest):
    world, _ = honest
    audit_mode = epoch(replace(CFG, sample="audit"))
    assert demo.audit(audit_mode)["profile"].n_r == demo.audit(world)["profile"].n_r
    assert audit_mode.memory.peak["sp/retained"] > 3 * world.memory.peak["sp/retained"]
    epoch(replace(CFG, retention="hbm"))  # serving-mode retention fits the unaudited workspace
    with pytest.raises(Conflict, match="sp/retained.*filler \\(pous\\)"):
        epoch(replace(CFG, sample="audit", retention="hbm"))


STRESS = replace(CFG, epoch=0.05, audit_rate=400.0, mean_gap=5e-5)  # audits land inside steps


def test_yield_latency_costs_certified_fraction_instead_of_failing_audits():
    reports = {}
    for launch in (1e-6, 1e-3):
        world = epoch(replace(STRESS, launch=launch))
        assert {a[0] for a in world.audits} == {"accepted"}
        assert max(a[1] for a in world.audits) <= world.engine.cfg.yield_latency  # never waits past the declared latency
        reports[launch] = demo.audit(world)
    fast, slow = (reports[k]["space"]["filler"] for k in (1e-6, 1e-3))
    assert 0 < slow.accounted < fast.accounted and slow.window > fast.window + 5e-4
    assert reports[1e-3]["work"][0].macs > 0  # the other protocols are unaffected


def test_offloaded_filler_answers_slowly_and_only_pous_loses(honest):
    _, report = honest
    world = epoch(after_wiring=lambda w: w.memory.offloaded.update(("filler", s) for s in range(64)))
    cheat = demo.audit(world)
    assert {a[0] for a in world.audits} == {"accepted"}
    assert cheat["space"]["filler"].accounted < report["space"]["filler"].accounted
    assert cheat["work"] == report["work"]


def test_fabricated_noised_outputs_are_caught_by_sampled_proofs():
    def cheat(world):
        def tamper(c):
            for call in world.sp.layout(c.step.program).vus:
                c.values[call[1].returns[1]] ^= 1  # every RU's first tile y', a ticket fabricated per request
        world.engine.hooks.on("commit", tamper)
    with pytest.raises(verity_sampled_proofs.Rejected, match="replay"):
        demo.audit(epoch(replace(CFG, cheat=cheat)))


def test_noise_not_from_the_issued_seed_is_caught_through_prescribed_inputs():
    def wrong_noise(world):
        world.engine.hooks.on("step_start", lambda s: s.noise.update(
            {r: verity_pouw.noise(b"not the seed", s.id, demo.K) for r in s.placement}))
    with pytest.raises(verity_sampled_proofs.Rejected, match="neither prescribed nor linked"):
        demo.audit(epoch(after_wiring=wrong_noise))


def test_a_forged_ticket_is_rejected_by_pouw():
    world = epoch()
    t = world.tickets[0]
    world.tickets[0] = verity_pouw.Ticket(t.step, t.request, t.gate - 2)
    with pytest.raises(verity_pouw.Rejected):
        demo.audit(world)
