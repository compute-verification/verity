"""The N3 check on runs served in memory: an honest run holds, and each way of breaking the relation is refused by name."""
import asyncio
import copy

import pytest

from verity.core.protocols.certifier.active import decode_frame
from workload import model as M
from workload import n3
from workload.block import Context, MemoryNet
from workload.hardware import OUTSIDE, Hardware
from workload.run import statement

CONFIG = {"spec": {"d": 2, "layers": 3, "requests": 2}, "blocks": 3, "devices": 2, "frame_bytes": 16, "seed": 1}
INGRESS = "b1.uplink>b1.gpu0"


def serve(config):
    ctx = Context(config)
    results = asyncio.run(MemoryNet(ctx).run())
    public, witness = statement(ctx, results)
    records, disagreements = n3.merge(results)
    assert disagreements == []
    return ctx, results, public, witness, records


@pytest.fixture(scope="module")
def honest():
    return serve(CONFIG)


def codes(public, witness, records):
    return set(n3.check(public, witness, records).codes())


def edited(records, key, edit):
    out = copy.deepcopy(records)
    out[key] = edit(list(out[key]))
    return out


def first_data(log):
    return next(i for i, (_, f) in enumerate(log) if decode_frame(f)[1] != 0)


def test_an_honest_run_holds_and_serves_the_reference(honest):
    ctx, results, public, witness, records = honest
    report = n3.check(public, witness, records)
    assert report.ok, report.violations[:3]
    s = report.stats
    assert s["frames"] == sum(map(sum, public["occupancy"].values())) == s["explained"] + s["declared"]
    assert s["appearances"] == sum(len(ctx.phi.route(m)) * len(m.edges) for m in ctx.phi.messages)
    assert s["values"] == len(ctx.edges)
    spec = ctx.spec
    assert results[OUTSIDE]["answers"] == M.reference(spec, M.requests(spec, 1), M.weights(spec, 1))


@pytest.mark.parametrize("mode, refused", [("extra-frame", {"unexplained", "schedule"}), ("flip-bit", {"content", "value"}),
                                           ("fill", {"fill"})])
def test_a_tampering_block_is_refused_by_name(mode, refused):
    _, _, public, witness, records = serve(dict(CONFIG, tamper={"unit": "b0", "mode": mode}))
    report = n3.check(public, witness, records)
    assert set(report.codes()) == refused
    assert {v.where for v in report.violations} == {"b0.uplink>b0.cert", INGRESS}


def test_a_dropped_frame_is_missing(honest):
    *_, public, witness, records = honest
    assert codes(public, witness, edited(records, INGRESS, lambda log: log[:first_data(log)] + log[first_data(log) + 1:])) \
        == {"missing", "schedule"}


def test_a_frame_in_another_bucket_is_refused(honest):
    *_, public, witness, records = honest

    def later(log):
        i = first_data(log)
        return log[:i] + [(log[i][0] + 1, log[i][1])] + log[i + 1:]

    assert codes(public, witness, edited(records, INGRESS, later)) == {"bucket", "schedule"}


def test_a_malformed_frame_is_refused_and_its_frame_is_missing(honest):
    *_, public, witness, records = honest

    def truncate(log):
        i = first_data(log)
        return log[:i] + [(log[i][0], log[i][1][:-1])] + log[i + 1:]

    assert codes(public, witness, edited(records, INGRESS, truncate)) == {"malformed", "missing"}


def test_a_replayed_hello_is_unexplained(honest):
    *_, public, witness, records = honest
    assert codes(public, witness, edited(records, INGRESS, lambda log: log + [log[0]])) == {"unexplained", "schedule"}


def test_another_placement_is_refused(honest):
    *_, public, witness, records = honest
    assert codes(dict(public, placement="00" * 64), witness, records) == {"placement"}


def test_values_that_dont_register_to_their_root_are_refused(honest):
    *_, public, witness, records = honest
    bad = copy.deepcopy(witness)
    row = bad["values"]["b1"]["rows"][0]
    bad["values"]["b1"]["rows"][0] = ("1" if row[0] == "0" else "0") + row[1:]
    assert codes(public, bad, records) == {"values"}


def test_a_profile_that_isnt_the_placements_is_refused(honest):
    *_, public, witness, records = honest
    profile = copy.deepcopy(public["occupancy"])
    profile[INGRESS][-1] += 1
    assert codes(dict(public, occupancy=profile), witness, records) == {"occupancy", "schedule"}


def test_two_logs_of_one_link_direction_must_agree(honest):
    _, results, *_ = honest
    twin = copy.deepcopy(results["b1"])
    twin["unit"] = "b1-twin"
    key = next(iter(twin["records"]))
    twin["records"][key] = twin["records"][key][:-1]
    _, bad = n3.merge(dict(results, **{"b1-twin": twin}))
    assert [(v.code, v.where) for v in bad] == [("records", key)]


def test_a_run_on_a_declaration_holds_with_one_step_a_bucket(datacenter):
    hw = Hardware.from_datacenter(datacenter(3))
    ctx, results, public, witness, records = serve({"spec": CONFIG["spec"], "hardware": hw.to_json(), "frame_bytes": 16,
                                                    "seed": 1})
    assert ctx.placement.ticks[0][:3] == (0, 10, 20)
    assert n3.check(public, witness, records).ok
    assert public["occupancy"]["b1.nic0-b1.cert>b1.nic0"] == [0, 0, 2, 1]


def test_a_block_with_no_values_commits_nothing():
    ctx, results, public, witness, records = serve(dict(CONFIG, blocks=4))
    assert results["b3"]["values"] is None and "b3" not in public["values"]
    assert n3.check(public, witness, records).ok
