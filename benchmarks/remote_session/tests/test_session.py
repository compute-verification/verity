"""The verifier's and the prover's parts one at a time: the verifier's answer to an append and its order of a root
message's parts, a registration closed from the verifier's registry file, a layout's regions, openings on the wire,
the timing term's parts and the predicted miss rate."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2]))

from verity.core.protocols.remote_bandwidth import audit, timing  # noqa: E402
from verity.core.protocols.remote_bandwidth import codec as rcodec  # noqa: E402

import prover  # noqa: E402
import verifier  # noqa: E402
import wire  # noqa: E402

P512 = rcodec.Codec(rcodec.P512, 2, False)


def test_the_timing_parts_reproduce_the_whole_node_sessions_split_and_sum_to_its_timing_term():
    """r20261015-203743-d101 (15 Oct), 8 GPUs' HBM and a host tier, every pass the whole store: its parts are #2089's
    split, and at either R they sum to the ledger's timing term."""
    d101 = {"span_ns": 420_074_057_129, "passes": 6430, "reads_bytes": 4_888_145_229_250_560, "pass_ns": 65_259_603,
            "tiers": {"hbm": {"bytes": 751_619_276_800, "ms": 61.502974},
                      "host": {"bytes": 8_589_934_592, "ms": 25.468371}}}
    for r in (timing.READ_BYTES_PER_S, timing.SPEC_BYTES_PER_S):
        parts = verifier.timing_parts(capacity_bytes_per_s=r, **d101)
        assert sum(parts.values()) == pytest.approx(1 - d101["reads_bytes"] / (r * d101["span_ns"] / 1e9), abs=1e-12)
    parts = verifier.timing_parts(capacity_bytes_per_s=timing.READ_BYTES_PER_S, **d101)
    assert sum(parts.values()) == pytest.approx(0.09926702029063428, abs=1e-12)
    assert {k: round(v * 100, 2) for k, v in parts.items()} == {
        "above_stated_tiers": 0.0, "host_dram_idle": 3.68, "hbm_below_stated": 0.66, "receipts_beyond_read": 5.48,
        "boundaries_and_allowance": 0.10, "blocks_left_out": 0.0}
    spec = verifier.timing_parts(capacity_bytes_per_s=timing.SPEC_BYTES_PER_S, **d101)
    assert spec["above_stated_tiers"] == pytest.approx(1 - timing.READ_BYTES_PER_S / timing.SPEC_BYTES_PER_S)


@pytest.mark.parametrize("layout", [
    [{"kind": "weights", "tier": "hbm", "first": 0, "blocks": 8}],
    [{"kind": "weights", "tier": "hbm", "first": 0, "blocks": 8},
     {"kind": "kv", "tier": "hbm", "first": 9, "blocks": 7}],
    [{"kind": "code", "tier": "hbm", "first": 0, "blocks": 16}],
    [{"kind": "kv", "tier": "nvme", "first": 0, "blocks": 16}],
    [{"kind": "kv", "tier": "hbm", "first": 0, "blocks": 0}, {"kind": "kv", "tier": "hbm", "first": 0, "blocks": 16}],
    {"kind": "kv", "tier": "hbm", "first": 0, "blocks": 16},
    ["kv"],
])
def test_a_layout_that_is_not_the_stores_regions_in_order_is_refused(layout):
    with pytest.raises(ValueError):
        verifier.regions_of(layout, 16)


def test_the_verifier_closes_a_registration_left_live_from_its_registry_file(tmp_path, monkeypatch, capsys):
    """`--registry R --close NODE`: a registration a verifier process left live (its session in flight when it ended)
    closes with that session unended, counted as failed, and the node may register again."""
    registry = audit.Registry.from_json({"schema": audit.REGISTRY_SCHEMA, "closed": [], "refused": {},
                                         "live": {b"test-node".hex(): {"registration": "ab" * 64,
                                                                        "sessions": ["accepted", audit.UNENDED]}}})
    path = tmp_path / "registry.json"
    verifier.save_registry(path, registry)
    monkeypatch.setattr(sys, "argv", ["verifier.py", "--registry", str(path), "--close", b"test-node".hex()])
    assert verifier.main() == 0
    back = verifier.load_registry(path)
    assert back.live(b"test-node") is None and back.closed == [
        audit.Closed(b"test-node", bytes.fromhex("ab" * 64), ("accepted", audit.UNENDED))]
    assert "closed" in capsys.readouterr().out


def test_an_append_is_answered_with_its_coin_and_a_root_message_is_applied_in_order():
    """`Verifier.append` answers an append's message with its first index and its coin, drawn then, once: the same
    root sent again draws none and is the session's `coin` rejection.  `Verifier.root`: a message's releases, then
    its root with its plan change and its step; a plan that still holds a block the message released, or one
    appended by no message, is the session's `plan` rejection, a root with no step or one not above the last pass's
    is its `step` rejection, and a message the session cannot read, a root's carrying appends among them, is
    malformed."""
    reg = audit.register(os.urandom(32), b"test-node", 8, bytes(64), P512)
    params = audit.Params(8, 10 ** 9, 1, 2, 2.0 ** -40, window_ns=10, codec=P512, receipt=reg.receipt, rows=reg.rows)

    def session():
        from verity.core.protocols.remote_bandwidth import coins

        s = audit.Session(params, reg, None, coins.Challenger(reg.digest, 0, params.coins), clock=lambda: 0,
                          not_before_ns=0)
        s.start()
        return s

    def appended(s):
        return verifier.Verifier.append(s, {"type": "append", "root": root, "count": 2})

    root = (b"\7" * 64).hex()
    s = session()
    reply = appended(s)
    assert set(reply) == {"type", "first", "coin"} and (reply["type"], reply["first"]) == ("appended", 8)
    assert bytes.fromhex(reply["coin"]) == s._appends[0][3] and len(reply["coin"]) == 64
    msg = {"t": 0, "i": 0, "root": root, "released": [[3, 1]], "change": [[[8, 2]], [[3, 1]]], "step": 5}
    assert verifier.Verifier.root(s, msg) is None
    assert [p.runs for p in s.windows[0].plans] == [((0, 3), (4, 6))] and s.released == 1 and s.extent == 10
    assert s.windows[0].steps == [5]
    assert verifier.Verifier.root(s, {"t": 0, "i": 1, "root": root, "change": [[], []], "step": 9}) is None
    assert appended(s) == audit.Verdict(0, False, "coin") and s.challenger.appends == 1
    for step in (5, 4):
        s = session()
        appended(s)
        assert verifier.Verifier.root(s, msg) is None
        v = verifier.Verifier.root(s, {"t": 0, "i": 1, "root": root, "change": [[], []], "step": step})
        assert v is not None and v.failure == "step"
    s = session()
    appended(s)
    v = verifier.Verifier.root(s, {k: x for k, x in msg.items() if k != "step"})
    assert v is not None and v.failure == "step"
    s = session()
    appended(s)
    v = verifier.Verifier.root(s, {**msg, "change": [[[8, 2]], []]})
    assert v is not None and v.failure == "plan"
    s = session()
    v = verifier.Verifier.root(s, {**msg, "released": []})
    assert v is not None and v.failure == "plan"
    with pytest.raises(ValueError):
        verifier.Verifier.append(session(), {"type": "append", "root": "zz", "count": 2})
    with pytest.raises(ValueError, match="message of its own"):
        verifier.Verifier.root(session(), {**msg, "appends": [[root, 2]]})


def test_a_kv_plan_needs_its_pages_a_step():
    assert prover.parse_plan("honest,kv:4,kvdrop:2,skip:0.5") == [("honest", 0.0), ("kv", 4.0), ("kvdrop", 2.0),
                                                                  ("skip", 0.5)]
    for bad in ("kv", "kv:0", "kv:1.5", "kvdrop:", "evict:2"):
        with pytest.raises(ValueError):
            prover.parse_plan(bad)


def test_openings_round_trip_on_the_wire_and_a_short_body_is_none():
    o = audit.Opening(3, 17, bytes(range(256)) * 16, b"\1" * 1024, b"\2" * 192, (b"\3" * 64,) * 5, b"\4" * 192,
                      (b"\5" * 64,) * 9, b"")
    appended = audit.Opening(4, 18, bytes(range(256)) * 16, b"\1" * 1024, b"\2" * 192, (b"\3" * 64,) * 5, b"\4" * 192,
                             (b"\5" * 64,) * 3, b"", b"\6" * 64)
    body = wire.encode_openings([o, appended, o])
    assert wire.decode_openings(body) == [o, appended, o]
    assert wire.decode_openings(body[:-1]) is None and wire.decode_openings(b"") is None
    with pytest.raises(ValueError):
        wire.encode_openings([audit.Opening(3, 17, o.block, o.leaf, o.leaf_salt, o.receipt_path, o.payload_salt,
                                            o.payload_path, b"", b"\6" * 63)])


@pytest.mark.parametrize("name", ["p1024/k3/masked", "p1024/k2/masked", "p512/k2"])
def test_codec_names_round_trip(name):
    assert wire.codec_named(name).name == name


def test_the_predicted_rate_is_the_draws_chance_to_miss_every_dropped_receipt():
    import summary

    assert summary.miss(100, 0, 10) == 1.0 and summary.miss(10, 1, 10) == 0.0
    assert summary.miss(20, 2, 2) == pytest.approx(18 / 20 * 17 / 19)
    U, f, s = 68 << 22, 2.0 ** -14, 2773
    assert summary.miss(U, round(f * U), s) == pytest.approx((1 - f) ** s, rel=1e-4)
