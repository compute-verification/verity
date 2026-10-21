"""The N3 check on runs served in memory: an honest run holds, and each way of breaking the relation is refused by name."""
import asyncio
import copy

import pytest

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


def first_data(ctx, key=INGRESS):
    """The position of the direction's first frame that isn't a HELLO, in grid order: no frame names its message."""
    slots = sorted(ctx.phi.slots()[key], key=lambda s: s.bucket)
    return next(i for i, s in enumerate(slots) if s.kind == "data")


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


@pytest.mark.parametrize("mode, refused", [("extra-frame", {"extra", "content", "value", "schedule"}),
                                           ("flip-bit", {"content", "value"}), ("fill", {"fill"})])
def test_a_tampering_block_is_refused_by_name(mode, refused):
    """A frame is matched to φ by its position, so an extra canonical frame takes the next one's place: that one is
    content, and the last one past φ's is extra."""
    _, _, public, witness, records = serve(dict(CONFIG, tamper={"unit": "b0", "mode": mode}))
    report = n3.check(public, witness, records)
    assert set(report.codes()) == refused
    assert {v.where for v in report.violations} == {"b0.uplink>b0.cert", INGRESS}


def test_a_dropped_frame_is_missing_and_shifts_the_next(honest):
    ctx, _, public, witness, records = honest
    i = first_data(ctx)
    assert codes(public, witness, edited(records, INGRESS, lambda log: log[:i] + log[i + 1:])) \
        == {"missing", "content", "value", "schedule"}


def test_a_frame_in_another_bucket_is_refused(honest):
    ctx, _, public, witness, records = honest
    i = first_data(ctx)

    def later(log):
        return log[:i] + [(log[i][0] + 1, log[i][1])] + log[i + 1:]

    assert codes(public, witness, edited(records, INGRESS, later)) == {"bucket", "schedule"}


def test_a_malformed_frame_is_refused_and_its_values_are_missing(honest):
    ctx, _, public, witness, records = honest
    i = first_data(ctx)

    def truncate(log):
        return log[:i] + [(log[i][0], log[i][1][:-1])] + log[i + 1:]

    assert codes(public, witness, edited(records, INGRESS, truncate)) == {"malformed", "missing"}


def test_a_replayed_hello_is_refused(honest):
    """The replay goes after its bucket's frames, so the next bucket's frame is out of place and the last is extra."""
    *_, public, witness, records = honest
    assert codes(public, witness, edited(records, INGRESS, lambda log: log + [log[0]])) == \
        {"extra", "content", "value", "schedule"}


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
