"""P2's arm of the POUS-only harness (`benchmarks/pous/p2_v1/`, `bench.py --impl p2-v1-gpu | p2-v3-gpu`) on CPU: the reference
(`p2_v1/reference.py`) against the spec's pinned vectors, and every step of the arm with the reference standing in for the GPU
library at the kit's 128-bit width, `test_pous_harness`'s Python responder for the native one, and the arm's memory challenger process
(`p2_v1/memory_challenger.py` for the spec's framing, `band_gpu/memory_challenger.py`'s `protocol.TimedAudit` for `p2-16448/v4`) on its
logical clock, so no outcome depends on a clock."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("verity")

from test_pous_harness import PyResponder  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmarks" / "pous"
WS = {"id": "toy/linears", "bytes": 2560, "tensors": [["q", 16, 32], ["k", 8, 32], ["down", 32, 16]]}
SEED = "5eed" * 8
"""`band_gpu/memory_challenger.py --seed` for the `V3` arm: its challenges, and so its controls' verdicts, are fixed."""


def _bench():
    spec = importlib.util.spec_from_file_location("pous_bench", BENCH / "bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _p2():
    sys.path.insert(0, str(BENCH))
    from p2_v1 import gpu, harness, reference
    return reference, harness, gpu


def _small():
    R, _, _ = _p2()
    return R.Params(128, 173, 14, blocks_per_segment=64, segments=1, profile="test-14B")    # the kit's w = 128 prime


class _Bound:
    """The reference as one setup's kernels (its key prefix), on CPU rows."""

    def __init__(self, params, prefix, flip: bool, says: int = 4) -> None:
        self.params, self.prefix, self.flip, self.keys_version = params, prefix, flip, says

    def _keys(self, i):
        R, _, _ = _p2()
        k, t, _ = R.keys_after(self.prefix, self.params, *divmod(i, self.params.blocks_per_segment))
        return k, t

    def keys(self, first, count):
        n = self.params.width_bytes
        ks = [self._keys(first + j) for j in range(count)]
        return (torch.tensor([list(k.to_bytes(n, "big")) for k, _ in ks], dtype=torch.uint8),
                torch.tensor([list(t.to_bytes(n, "big")) for _, t in ks], dtype=torch.uint8))

    def _map(self, rows, first, fn):
        n = self.params.width_bytes
        out = []
        for j in range(rows.shape[0]):
            k, t = self._keys(first + j)
            out.append(list(fn(self.params, int.from_bytes(bytes(rows[j].numpy()), "big"), k, t).to_bytes(n, "big")))
        return torch.tensor(out, dtype=torch.uint8)

    def encode(self, m, first):
        R, _, _ = _p2()
        c = self._map(m, first, R.encode)
        if self.flip and m.shape[0] > 1:
            c[-1, -1] ^= 1
        return c

    def decode(self, c, first):
        R, _, _ = _p2()
        return self._map(c, first, R.decode)

    def raised(self):
        return 0


def _backend(flip: bool = False, says: int = 4):
    _, harness, _ = _p2()

    def timer(fn):
        fn()
        return 1e-3

    return harness.Backend("cpu", lambda params, prefix: _Bound(params, prefix, flip, says), lambda c, cpu: PyResponder(c), timer)


def _args(tmp_path, **kw):
    base = {"max_segments": 0, "p2_lib": str(tmp_path / "lib"), "build_dir": str(tmp_path / "build"), "p2_encode_blocks": 0,
            "tokens": "1,8", "reps": 1, "load_tokens": 0, "audits": 2, "pings": 20, "allowance": "max", "k": 0, "deadline_ns": 0,
            "clock": "logical", "verifier_cpu": None, "responder_cpu": None, "nvme_dir": "", "nvme_samples": 8, "impl_commit": None}
    return types.SimpleNamespace(**{**base, **kw})


def test_the_reference_matches_every_pinned_vector():
    R, _, _ = _p2()
    got = R.conformance()
    assert got["passed"] and got["checks"]["file_sha512"], got
    assert set(got["checks"]) == {"file_sha512", "p2-v4/w64/0", "p2-v4/w96/0", "p2-v4/w128/0", "p2-v4/w16448/0",
                                  "sigma-rejection/w64", "sigma-rejection/w16448"}


@pytest.mark.slow
def test_the_reference_encodes_the_full_width_vector():
    R, _, _ = _p2()
    assert R.conformance(full_encode=True)["passed"]


def test_the_v4_keys_are_one_compression_a_chunk_past_the_prefixs_midstate():
    """`p2keys4.cuh` takes `expand_prefix` as whole SHA-512 blocks: a chunk's last block is `seg ‖ blk ‖ P ‖ u32be(i)`, SHA-512's
    0x80 byte and the 16-byte length, so every chunk costs one compression."""
    import hashlib

    R, _, _ = _p2()
    salt, model = bytes(range(32)), bytes(range(64))
    prefix = R.expand_prefix(salt, R.DEPLOYMENT, model)
    assert len(prefix) == 256 and 8 + 8 + 1 + 4 + 1 + 16 <= 128
    k, t, attempt = R.keys(salt, R.DEPLOYMENT, model, 5, 1234)
    base = prefix + (5).to_bytes(8, "big") + (1234).to_bytes(8, "big")
    row = lambda p: b"".join(hashlib.sha512(base + p + i.to_bytes(4, "big")).digest() for i in range(33))[:2056]  # noqa: E731
    assert k.to_bytes(2056, "big") == row(b"K") and attempt == 0 and t.to_bytes(2056, "big") == row(b"t")


def test_the_root_exponent_splits_as_the_kernel_takes_it():
    R, _, gpu = _p2()
    for params in (R.DEPLOYMENT, _small()):
        a, n, r = gpu.exponent_split(params)
        assert (((1 << n) - 1) << a) + r == (params.p + 1) // 4 and 0 <= r < 1 << a
    assert gpu.exponent_split(R.DEPLOYMENT) == (13, 16433, 2926)


def test_params_refuse_a_modulus_below_the_payload():
    """Every payload is below p (spec §2, `m < 2^b < p`): a c0 past 2^w (a negative modulus, which is 3 mod 4 in Python's
    arithmetic), or one leaving p at or below the largest payload, is refused rather than rejecting every block."""
    R, _, _ = _p2()
    for c0 in ((1 << 32) + 5, (1 << 32) - 3, (1 << 32) - (1 << 24) + 1):
        with pytest.raises(ValueError, match="payload"):
            R.Params(32, c0, 3)
    assert R.Params(32, 5, 3).p == (1 << 32) - 5


def test_the_tree_and_the_audit_derivation_are_the_specs():
    import hashlib

    from verity.ml.commitments.identity import tagged_sha512
    from verity.ml.commitments.rowleaf import sha512_row_v2_digest
    from verity.ml.randomness import frame

    R, _, _ = _p2()
    leaves = [bytes([i]) * 64 for i in range(3)]
    node = lambda l, r: hashlib.sha512(b"\x01" + l + r).digest()  # noqa: E731
    assert R.root(leaves) == node(node(leaves[0], leaves[1]), leaves[2])                    # RFC 6962: split at 2
    assert R.path(leaves, 2) == [node(leaves[0], leaves[1])] and R.path(leaves, 0) == [leaves[1], leaves[2]]
    block, salt = bytes(range(256)) * 8 + bytes(8), b"t" * 32
    leaf = R.leaf(salt, 3, 513, block)
    assert leaf == hashlib.sha512(b"\x00" + salt + (3).to_bytes(8, "big") + (513).to_bytes(2, "big")
                                  + sha512_row_v2_digest(block, 8 * 2056)).digest()
    assert len(b"\x00" + salt + bytes(10) + leaf) == 107 <= 111                            # the leaf is one compression
    params, w = R.Params(64, 189, 7, 8, 1, "p2-63b"), b"\x01\x02"
    manifest = R.canonical_json_bytes(params.manifest())
    assert R.w_commitment(params, w) == tagged_sha512("verity-pous/w-commitment/v2", len(manifest).to_bytes(8, "big")
                                                      + manifest + w)
    prefix = R.expand_prefix(salt, R.DEPLOYMENT, b"m" * 64)
    body = R.KEYS_TAG + frame("p2/v4") + frame("p2-p16448-c21065") + frame(b"m" * 64) + frame(salt)
    assert len(body) == 176 and prefix == body + bytes(80) and len(prefix) == 256      # two whole blocks
    base = prefix + bytes(16)
    assert R.stream(base, b"t", 65) == b"".join(hashlib.sha512(base + b"t" + i.to_bytes(4, "big")).digest() for i in (0, 1))[:65]
    audit_id, key = R.audit_key(b"s" * 32, b"e" * 64, b"r" * 64, salt, 0)
    again_id, again = R.audit_key(b"s" * 32, b"e" * 64, b"r" * 64, salt, 0)
    assert audit_id == again_id and [key.uniform(1 << 19, j) for j in range(5)] == [again.uniform(1 << 19, j) for j in range(5)]
    other_id, _ = R.audit_key(b"s" * 32, b"e" * 64, b"r" * 64, salt, 1)
    assert other_id != audit_id                                                               # a fresh counter, a fresh audit


def test_the_spec_rtt_cap_is_the_schemes():
    """The arm's Δ and RTT cap are spec v1.1's, as `verity`'s P2 scheme states them, so `rtt_allowance_within_spec`
    judges against the cap whose `Δ + RTT` the certificate admits."""
    from verity.core.protocols.pous import audit
    from verity.core.protocols.pous.schemes import p2

    R, _, _ = _p2()
    assert (R.DELTA_NS, R.MAX_RTT_ALLOWANCE_NS) == (p2.DELTA_NS, p2.RTT_ALLOWANCE_CAP_NS) == (500_000, 354_000)
    assert p2.DELTA_NS == audit.DEADLINE_NS
    assert R.DELTA_NS + R.MAX_RTT_ALLOWANCE_NS < p2.ROOT_CALL_NS <= R.DELTA_NS + R.MAX_RTT_ALLOWANCE_NS + 1_000


def test_the_p2_arm_runs_every_step_and_its_gates_decide(tmp_path):
    R, _, _ = _p2()
    nvme = tmp_path / "nvme"
    nvme.mkdir()
    body = _bench().P2V1Gpu().run(_args(tmp_path, nvme_dir=str(nvme)), WS, tmp_path, backend=_backend(), params=_small())
    gates = body["gates"]
    assert body["segments"] == 3 and body["blocks"] == 192 and body["weight_bytes"] == 2560
    for g in ("vector", "round_trip", "reference_decode", "kernel_flags", "timed_audit", "value_control", "late_control",
              "recompute_control"):
        assert gates[g]["passed"], (g, gates[g])
    assert gates["vector"]["vector"] == "p2-v4/w128/0" and gates["round_trip"]["blocks"] == 192
    point = gates["operating_point"]
    assert point["k"] and point["deadline"] and point["rtt_allowance"] and not point["blocks"]    # 192 < 2^8 at the test point
    m = body["metrics"]
    honest, value, late, rec = m["audit"], m["value_control"], m["late_control"], m["recompute_control"]
    assert honest["audits_accepted"] == 2 and honest["k"] == R.K_AUDIT == 105 and honest["responder"]["answered"] == 2 * 105
    limit = m["calibration"]["limit_ns"]
    assert honest["limit_fixed_before_load"] and honest["limit_ns"] == late["limit_ns"] == rec["limit_ns"] == limit
    assert honest["allowance_kind"] == "loopback" and len(honest["source"]) == 64 and len(honest["records"]) == 2
    assert value["audits_accepted"] == 0 and value["wrong"] > 0
    assert late["audits_accepted"] == 0 and late["late"] > 0 and late["wrong"] == 0
    assert rec["wrong"] == 0 and rec["late"] > 0 and rec["margin_s"]["min"] < 0 and rec["dropped_blocks"] == 20
    assert m["seqroot"]["root_s"] == 1e-3 / 2 and "upper bound" in m["seqroot"]["label"]
    assert m["loop"]["tokens"] == [1, 8] and set(m["loop"]["overhead"]) == {1, 8} and m["nvme"]["block_read_s"]["n"] == 8
    assert body["setup"]["manifest"] == {"scheme": "p2", "w": 128, "c0": 173, "profile": "test-14B", "blocks_per_segment": 64,
                                         "segments": 3}
    assert not body["profile"]["certified"] and "P2–M1-SGI" in body["profile"]["experimental"]
    names = {r["name"] for r in body["measurements"]}
    assert {"encode_GBps", "decode_overhead_T1", "audit_margin_min_s", "recompute_margin_min_s", "seqroot_root_s"} <= names


def test_the_recompute_control_covers_c_past_its_encoded_head(tmp_path):
    """With only the first 64 of 192 blocks encoded (the rest random canonical, as on the device), every dropped block, in the
    head or past it, is re-encoded from its decoded row and answers correctly; the margin is the limit minus each recomputed
    answer's device time."""
    body = _bench().P2V1Gpu().run(_args(tmp_path, p2_encode_blocks=64, audits=6), WS, tmp_path, backend=_backend(),
                                  params=_small())
    m, gates = body["metrics"], body["gates"]
    assert m["encoded_blocks"] == 64 and gates["round_trip"]["blocks"] == 64
    rec = m["recompute_control"]
    assert rec["wrong"] == 0 and gates["recompute_control"]["passed"], gates["recompute_control"]
    limit = m["calibration"]["limit_ns"]
    recomputed = [a for r in rec["records"] for a in r["answers"] if a["index"] % 10 == 0]
    assert rec["recomputed_answers"] == len(recomputed) > 0 and all(a["device_ns"] == 1_000_000 for a in recomputed)
    assert rec["device_margin_s"]["min"] == rec["device_margin_s"]["max"] == (limit - 1_000_000) / 1e9      # the timer's 1 ms
    rows = {r["name"]: r["value"] for r in body["measurements"]}
    assert rows["recompute_margin_min_s"] == rows["recompute_margin_max_s"] == (limit - 1_000_000) / 1e9
    assert m["seqroot"]["block"] % 10 == 0 and len(m["seqroot"]["encode_one_block_reps_s"]) == 3


def test_the_p2_recompute_adversary_reencodes_every_dropped_block_of_c():
    """`plaintext_rows` + `_recomputer` give back `C_i` for dropped blocks in the encoded head and past it."""
    R, harness, _ = _p2()
    P = _small().with_segments(1)
    salt, model = b"s" * 32, b"m" * 64
    bound = _Bound(P, R.expand_prefix(salt, P, model), False)
    blocks, nb = P.blocks, P.width_bytes
    m = torch.zeros(blocks, nb, dtype=torch.uint8)
    m[:, nb - P.payload_bytes:] = torch.randint(0, 256, (blocks, P.payload_bytes), generator=torch.Generator().manual_seed(1),
                                                dtype=torch.uint8)
    c = torch.cat([bound.encode(m[:8], 0), torch.randint(0, 128, (blocks - 8, nb), generator=torch.Generator().manual_seed(2),
                                                         dtype=torch.uint8)])
    dropped = [0, 7, 8, 30, blocks - 1]
    rc = harness._recomputer(_backend(), bound, harness.plaintext_rows(bound, c, dropped), dropped)
    for i in dropped:
        block, device_ns = rc(i)
        assert block == bytes(c[i].numpy()) and device_ns == 1_000_000, i


def test_a_wrong_p2_kernel_stops_the_run_before_anything_is_timed(tmp_path):
    body = _bench().P2V1Gpu().run(_args(tmp_path), WS, tmp_path, backend=_backend(flip=True), params=_small())
    assert body["gates"]["vector"]["passed"] and not body["gates"]["round_trip"]["passed"]
    assert body["metrics"] is None and body["measurements"] == [] and "nothing was timed" in body["stopped"]


def test_the_vector_gate_holds_the_keys_to_the_pinned_v4_keys(tmp_path):
    """The vector gate reads `p2-v4-keys.json`: a library that expands v4's keys passes; one that reports another key version,
    or whose keys differ from the pinned file's, fails it."""
    import json

    _, harness, _ = _p2()
    body = _bench().P2V1Gpu().run(_args(tmp_path, gates_only=True), WS, tmp_path, backend=_backend(), params=_small())
    assert all(g["passed"] for g in body["gates"].values()), body["gates"]
    assert body["gates"]["vector"]["keys_version"] == 4 and body["gates"]["vector"]["keys_v4"]
    body = _bench().P2V1Gpu().run(_args(tmp_path, gates_only=True), WS, tmp_path, backend=_backend(says=1), params=_small())
    assert not body["gates"]["vector"]["passed"] and not body["gates"]["vector"]["keys_v4"]
    keys = json.loads(harness.KEYS.read_text())
    small = _small()
    for v in keys["vectors"]:
        if int(v["w"]) == small.w:
            v["t"] = "00" * small.width_bytes
    gate = harness.vector_gate(_backend(), small, keys=keys)
    assert not gate["passed"] and gate["K"] and not gate["t"]


def test_gates_only_runs_the_kernel_gates_and_times_nothing(tmp_path):
    body = _bench().P2V1Gpu().run(_args(tmp_path, gates_only=True), WS, tmp_path, backend=_backend(), params=_small())
    assert set(body["gates"]) == {"vector", "round_trip", "reference_decode", "kernel_flags"}
    assert all(g["passed"] for g in body["gates"].values())
    assert body["metrics"] is None and body["measurements"] == [] and "--gates-only" in body["stopped"]


def test_audits_0_runs_the_p2_arm_untimed_with_no_limit_audit_or_control_and_says_so(tmp_path, monkeypatch):
    _, harness, _ = _p2()
    monkeypatch.setattr(harness.H, "audit_round", lambda *a, **k: pytest.fail("an untimed run started an audit"))
    monkeypatch.setattr(harness.H, "calibrate_limit", lambda *a, **k: pytest.fail("an untimed run calibrated a limit"))
    body = _bench().P2V1Gpu().run(_args(tmp_path, audits=0), WS, tmp_path, backend=_backend(), params=_small())
    assert body["timed"] is False and body["audits"] == "skipped (untimed placement)" and body["profile"] is None
    assert set(body["gates"]) == {"vector", "round_trip", "reference_decode", "kernel_flags"}
    assert all(g["passed"] for g in body["gates"].values())
    m = body["metrics"]
    assert m["timed"] is False and set(m["loop"]["overhead"]) == {1, 8} and m["seqroot"]["twice_limit_s"] is None
    assert not {"audit", "deadline", "calibration", "value_control", "late_control", "recompute_control"} & set(m)
    names = {r["name"] for r in body["measurements"]}
    assert {"encode_GBps", "stored_GBps", "decode_overhead_T1", "seqroot_root_s"} <= names
    assert not [name for name in names if name.startswith(("audit", "recompute"))]
    assert not (tmp_path / "leaves-p2.bin").exists()


def test_side_by_side_tables_both_arms_and_withholds_a_failed_one(tmp_path, monkeypatch):
    import json

    import test_pous_harness as band_t

    sys.path.insert(0, str(BENCH))
    import side_by_side as sbs

    bp = band_t._small()
    (tmp_path / "b").mkdir()
    (tmp_path / "p").mkdir()
    band = _bench().BandGpu().run(band_t._args(tmp_path / "b"), band_t.WS, tmp_path / "b", backend=band_t._backend(bp), base=bp,
                                  vectors=band_t._vectors(bp))
    p2 = _bench().P2V1Gpu().run(_args(tmp_path / "p"), WS, tmp_path / "p", backend=_backend(), params=_small())
    doc = lambda body, scheme, ok: json.loads(json.dumps({"scheme": scheme, "passed": ok, "weight_set": WS["id"], "hardware": "cpu",  # noqa: E731
                                                          "gates": body["gates"], "metrics": body["metrics"]}, default=str))
    both = sbs.table(doc(band, "band-chain/d12/v1", True), doc(p2, "p2/v4", True))
    rows = {r["metric"]: r for r in both["rows"]}
    for name in ("encode_GBps", "decode_pass_s", "decode_overhead_T1", "decode_overhead_T8", "audit_answer_p99.9_s",
                 "audit_margin_min_s", "audit_limit_s", "recompute_margin_min_s"):
        assert rows[name]["band"] is not None and rows[name]["p2"] is not None, name
    assert rows["decode_pass_s"]["p2_over_band"] == rows["decode_pass_s"]["p2"] / rows["decode_pass_s"]["band"]
    assert "| metric | band (band-chain/d12/v1) | P2 (p2/v4) |" in sbs.markdown(both)
    one = sbs.table(doc(band, "band-chain/d12/v1", True), doc(p2, "p2/v4", False))
    assert all(r["p2"] is None for r in one["rows"]) and not one["p2"]["passed"]
    (tmp_path / "band.json").write_text(json.dumps(doc(band, "band-chain/d12/v1", True)))
    (tmp_path / "p2.json").write_text(json.dumps(doc(p2, "p2/v4", False)))
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    assert sbs.main([str(tmp_path / "band.json"), str(tmp_path / "p2.json"), "--out", str(tmp_path / "side.json")]) == 1
    assert json.loads((tmp_path / "side.json").read_text())["schema"] == sbs.SCHEMA


def test_the_registered_expander_given_the_specs_prefix_gives_every_pinned_key():
    bench = _bench()
    assert bench.IMPLS["p2-v3-gpu"] is bench.P2V3Gpu and bench.IMPLS["p2-v1-gpu"] is bench.P2V1Gpu
    got = bench.P2V3Gpu().conformance_gate()
    assert got["passed"], got
    assert {k for k in got["checks"] if k.startswith("expander/")} == {
        "expander/p2-v4/w64/0", "expander/p2-v4/w96/0", "expander/p2-v4/w128/0", "expander/p2-v4/w16448/0"}


def test_the_v3_arm_serves_v3s_blocks_and_audits_them_through_protocol_memory_challenger(tmp_path):
    """Under `V3` the device's `C` is v3's encoding of `W` (v3's own codec decodes it back), the leaves are `vk`'s, and every
    audit and control is `protocol.TimedAudit`'s.  At the test's 128-bit width P2 has no certificate, so the audit runs as a
    research audit and says so."""
    from verity.core.protocols.pous import protocol, schemes
    from verity.core.protocols.pous.verifier import leaf_hash

    _, harness, _ = _p2()
    body = _bench().P2V3Gpu().run(_args(tmp_path, k=105, seed=SEED), WS, tmp_path, backend=_backend(), params=_small(), research=True)
    gates, m = body["gates"], body["metrics"]
    for g in ("vector", "round_trip", "reference_decode", "kernel_flags", "codec", "timed_audit", "value_control",
              "late_control", "recompute_control"):
        assert gates[g]["passed"], (g, gates[g])
    assert "certificate" not in gates and gates["codec"]["codec"] == "p2-128/v4"
    params = schemes.P2Params(128, 173, 64, 3)
    setup = body["setup"]
    assert setup["scheme"] == "p2-128/v4" and setup["manifest"] == params.manifest() and len(bytes.fromhex(setup["salt"])) == 24
    assert body["implementation"] == {**body["implementation"], "name": "p2-v3-gpu", "memory_challenger": "band_gpu/memory_challenger.py",
                                      "memory_challenger_args": []}
    w = b"".join(_bench()._w_segment(WS["id"], s, params.segment_bytes, WS["bytes"]) for s in range(3))
    commitment = protocol.commit(schemes.P2(params), w)
    salt = protocol.draw_salt(harness.SETUP_SOURCE, commitment)
    assert setup["w_commitment"] == commitment.hex() and setup["salt"] == salt.hex()
    raw = (tmp_path / "leaves-p2.bin").read_bytes()
    encoded = protocol.encode(schemes.P2(params), w, salt)
    assert raw == b"".join(leaf_hash(params, salt, s * 64 + j, b) for s, seg in enumerate(encoded) for j, b in enumerate(seg))
    honest = m["audit"]
    assert honest["scheme"] == "p2-128/v4" and honest["strict"] is False and honest["audits_accepted"] == 2
    assert honest["k"] == 105 and any("no certificate" in g for g in honest["gaps"])
    from band_gpu.memory_challenger import seeded_source
    assert honest["seed"] == SEED and [r["source"] for r in honest["records"]] == [
        seeded_source(bytes.fromhex(SEED), j).hex() for j in range(2)]
    assert m["value_control"]["audits_accepted"] == 0 and m["late_control"]["audits_accepted"] == 0
    assert m["recompute_control"]["wrong"] == 0 and m["recompute_control"]["recomputed_answers"] > 0
    assert body["profile"]["scheme"] == "p2-128/v4" and not body["profile"]["certified"]


def test_a_joint_runs_store_is_set_up_under_the_bandwidth_salt_of_its_device(tmp_path, monkeypatch):
    """`--bandwidth-salt GPU` sets `C` up under `bandwidth.audit.draw_salt` for the GPU's id as the challenger reads it, and
    P2's audit runs over that `C`, so the bandwidth verifier accepts the store P2 audits; a plain run's salt is PoUS's own
    (the test above), and the switch refuses the frozen spec's framing."""
    from verity.core.protocols.bandwidth import audit
    from verity.core.protocols.pous import protocol, schemes
    from verity.core.protocols.pous.verifier import VerifierKey, leaf_hash

    _, harness, _ = _p2()
    device = bytes(range(16))
    asked: list[str] = []
    monkeypatch.setattr(audit, "read_device", lambda gpu: asked.append(gpu) or device)
    gpu = "GPU-00010203-0405-0607-0809-0a0b0c0d0e0f"
    body = _bench().P2V3Gpu().run(_args(tmp_path, k=105, seed=SEED, bandwidth_salt=gpu), WS, tmp_path, backend=_backend(),
                                  params=_small(), research=True)
    assert asked == [gpu] and body["gates"]["timed_audit"]["passed"] and body["gates"]["codec"]["passed"]
    params = schemes.P2Params(128, 173, 64, 3)
    w = b"".join(_bench()._w_segment(WS["id"], s, params.segment_bytes, WS["bytes"]) for s in range(3))
    commitment = protocol.commit(schemes.P2(params), w)
    salt = audit.draw_salt(harness.SETUP_SOURCE, commitment, device)
    setup = body["setup"]
    assert (setup["salt"], setup["salt_from"], setup["device"]) == (salt.hex(), "bandwidth", device.hex())
    assert salt != protocol.draw_salt(harness.SETUP_SOURCE, commitment)
    encoded = protocol.encode(schemes.P2(params), w, salt)
    leaves = [leaf_hash(params, salt, s * 64 + j, b) for s, seg in enumerate(encoded) for j, b in enumerate(seg)]
    assert (tmp_path / "leaves-p2.bin").read_bytes() == b"".join(leaves)
    vk = VerifierKey.from_leaves(params, salt, leaves)
    (tmp_path / "state").mkdir()
    audit.ChallengerState.create(tmp_path / "state", vk)
    with audit.ChallengerState(tmp_path / "state", vk) as state:
        audit.Verifier(vk, audit.Params("ro-blake3/v1", audit.SAMPLES, 1_200_000), b"secret", state, device=device,
                       commitment=commitment, salt_source=harness.SETUP_SOURCE)
        with pytest.raises(ValueError, match="not draw_salt's for this device"):
            audit.Verifier(vk, audit.Params("ro-blake3/v1", audit.SAMPLES, 1_200_000), b"secret", state,
                           device=bytes(16), commitment=commitment, salt_source=harness.SETUP_SOURCE)
    with pytest.raises(SystemExit, match="registered P2 store"):
        _bench().P2V1Gpu().run(_args(tmp_path / "v1", bandwidth_salt=gpu), WS, tmp_path / "v1", backend=_backend(),
                               params=_small())


def test_the_v3_arm_is_strict_by_default_so_an_uncovered_audit_is_refused_and_fails(tmp_path):
    body = _bench().P2V3Gpu().run(_args(tmp_path, k=105, seed=SEED), WS, tmp_path, backend=_backend(), params=_small())
    gates, honest = body["gates"], body["metrics"]["audit"]
    assert body["implementation"]["memory_challenger_args"] == ["--strict"]
    assert "has no certificate" in honest["refused"] and honest["audits"] == 0 and honest["responder"]["answered"] == 0
    assert not gates["timed_audit"]["passed"] and not gates["certificate"]["passed"] and body["profile"] is None


def _live_at_the_p2_point(tmp_path, limit_ns: int, clock=None):
    """`band_gpu/memory_challenger.py`'s audit, strict, of `p2-16448/v4` at its registered point (one segment), against a responder whose
    blocks match `vk`'s leaves, on the logical clock (or `clock`): the memory challenger checks leaves, not codewords."""
    import hashlib

    sys.path.insert(0, str(BENCH))
    from band_gpu import harness as H
    from band_gpu import memory_challenger as live

    from verity.core.protocols.pous import schemes
    from verity.core.protocols.pous.verifier import VerifierKey, leaf_hash

    scheme = schemes.get("p2-16448/v4")
    params, salt = scheme.params, b"s" * 24
    block = lambda i: hashlib.shake_256(i.to_bytes(4, "big")).digest(params.block_bytes)  # noqa: E731
    vk = VerifierKey.from_leaves(params, salt, [leaf_hash(params, salt, i, block(i)) for i in range(params.blocks)])
    responder = H.ControlResponder(params.block_bytes, lambda i: (block(i), 0, 1_000, 0))
    try:
        return live.audit("127.0.0.1", responder.port(), scheme, vk, audits=2, pings=20, limit_ns=limit_ns,
                          clock=clock or live.LogicalClock(), research=False), responder.answered()
    finally:
        responder.stop()


def test_protocol_memory_challenger_audits_p2_v3_at_the_rtt_cap_and_refuses_one_us_past_it(tmp_path):
    """The end-to-end claim's memory challenger: `Δ + 354 µs` (spec v1.1's cap) is inside `P2SlackFamilyUncond`'s
    certificate, so 105 challenges run and are accepted; `Δ + 355 µs` fits one 854,029 ns root call, so the memory challenger
    refuses before sending any challenge."""
    from verity.core.protocols.pous.schemes import p2

    doc, answered = _live_at_the_p2_point(tmp_path, 500_000 + p2.RTT_ALLOWANCE_CAP_NS)
    assert doc["strict"] and "refused" not in doc and doc["audits_accepted"] == doc["audits"] == 2 and answered == 2 * 105
    assert doc["k"] == 105 and doc["gaps"] == ["W has 1 segments; Pous.SecurityProofs.P2SlackFamilyUncond is about 8"]
    assert doc["profile"]["certificate"]["theorem"] == "Pous.SecurityProofs.P2SlackFamilyUncond"
    assert doc["profile"]["limit_ns"] == 854_000 and doc["profile"]["effective_rounds"] == 0
    doc, answered = _live_at_the_p2_point(tmp_path, 500_000 + p2.RTT_ALLOWANCE_CAP_NS + 1_000)
    assert doc["audits"] == 0 and answered == 0 and doc["records"] == []
    assert "Δ + RTT = 855000 ns fits 1 calls of 854029 ns" in doc["refused"]


def test_the_host_trace_splits_each_slow_round_trip_into_where_its_time_went():
    """A ping over `TRACE_OVER_NS` is kept; a challenge is kept when its round trip or the memory challenger's time for it is
    over, and its split adds up: send, the wait for the first byte (the responder's reported time and the rest), the rest of
    the answer, and the memory challenger's own work outside the round trip."""
    sys.path.insert(0, str(BENCH))
    from band_gpu import memory_challenger as live

    ping, challenge = live.OP_PING, live.OP_CHALLENGE
    log = [(ping, 0, 0, 1_000, 5_000, 6_000, 1_000, 2_000, 0),
           (ping, 0, 10_000, 11_000, 90_000, 95_000, 70_000, 15_000, 0),
           (challenge, 7, 100_000, 101_000, 140_000, 141_000, 3_000, 120_000, 30_000),
           (challenge, 9, 200_000, 201_000, 210_000, 211_000, 1_000, 205_000, 5_000)]
    records = [{"answers": [{"elapsed_ns": 600_000}, {"elapsed_ns": 12_000}]}]
    trace = live.host_trace(log, records)
    assert trace["clock"] == "CLOCK_MONOTONIC" and trace["roundtrips"] == 4 and trace["over_ns"] == live.TRACE_OVER_NS
    slow_ping, slow_answer = trace["kept"]
    assert slow_ping == {"kind": "ping", "start_ns": 10_000, "roundtrip_ns": 85_000, "send_ns": 1_000, "wait_ns": 79_000,
                         "server_ns": 0, "responder_other_ns": 79_000, "recv_ns": 5_000, "poll_gap_ns": 70_000,
                         "poll_gap_at_ns": 15_000}
    assert slow_answer == {"kind": "challenge", "audit": 0, "round": 0, "index": 7, "elapsed_ns": 600_000,
                           "outside_ns": 559_000, "start_ns": 100_000, "roundtrip_ns": 41_000, "send_ns": 1_000,
                           "wait_ns": 39_000, "server_ns": 30_000, "responder_other_ns": 9_000, "recv_ns": 1_000,
                           "poll_gap_ns": 3_000, "poll_gap_at_ns": 120_000}


def test_a_wall_clock_audit_traces_every_round_trip_and_a_logical_one_none(tmp_path):
    """On a clock that is not the logical one, the memory challenger logs every round trip (the pings before and after and
    each answer, the 50 warm-up pings before each set included) and its thread's faults and context switches; each kept split
    adds up to its round trip."""
    import itertools

    from verity.core.protocols.pous.schemes import p2

    ticks = itertools.count(0, 7)
    doc, answered = _live_at_the_p2_point(tmp_path, 500_000 + p2.RTT_ALLOWANCE_CAP_NS, clock=lambda: next(ticks))
    assert doc["audits_accepted"] == 2 and answered == 2 * 105
    trace = doc["host_trace"]
    assert trace["roundtrips"] == 2 * (50 + 20) + 2 * 105
    for row in trace["kept"]:
        assert row["send_ns"] + row["wait_ns"] + row["recv_ns"] == row["roundtrip_ns"]
        assert row["server_ns"] + row["responder_other_ns"] == row["wait_ns"]
    assert set(doc["memory_challenger_rusage"]["after"]) == {"minflt", "majflt", "nvcsw", "nivcsw"}
    doc, _ = _live_at_the_p2_point(tmp_path, 500_000 + p2.RTT_ALLOWANCE_CAP_NS)
    assert "host_trace" not in doc


def test_a_cached_encoding_is_reused_for_its_own_setup_and_never_when_its_bytes_changed(tmp_path):
    """`--p2-c-cache`: the first run encodes and stores `C`; a second run of the same setup loads it (the encode time is
    the first run's) and passes every gate a fresh `C` does; a cached `C` whose bytes no longer match its sha256 is
    encoded again."""
    cache = tmp_path / "cache"

    def run(name: str) -> dict:
        out = tmp_path / name
        out.mkdir()
        return _bench().P2V3Gpu().run(_args(out, k=105, seed=SEED, p2_c_cache=str(cache)), WS, out, backend=_backend(),
                                      params=_small(), research=True)

    first = run("first")
    rec = first["metrics"]["encode_cache"]
    assert rec["hit"] is False and (cache / f"{rec['key']}.c").stat().st_size == rec["bytes"]
    second = run("second")
    assert second["metrics"]["encode_cache"] == {**rec, "hit": True}
    assert second["metrics"]["encode_s"] == first["metrics"]["encode_s"]
    for g in ("round_trip", "reference_decode", "codec", "timed_audit", "value_control", "late_control", "recompute_control"):
        assert second["gates"][g]["passed"], (g, second["gates"][g])
    assert (tmp_path / "second" / "leaves-p2.bin").read_bytes() == (tmp_path / "first" / "leaves-p2.bin").read_bytes()
    data = cache / f"{rec['key']}.c"
    raw = bytearray(data.read_bytes())
    raw[5] ^= 1
    data.write_bytes(bytes(raw))
    third = run("third")
    assert third["metrics"]["encode_cache"]["hit"] is False and third["gates"]["round_trip"]["passed"]
    assert hashlib.sha256(data.read_bytes()).hexdigest() == rec["sha256"]


def test_the_memory_challenger_arm_audits_v3s_blocks_and_fails_closed_under_its_test_signer(tmp_path):
    """Under `MC` every audit and control is the memory challenger's: its leaves are each block's `sha512/row/v2` digest,
    the honest sessions pass under the test signer's own mapping and every control is rejected under it (the tampered one
    for a flipped bit in a block's last byte, the bad-signature one for a flipped signature bit and for a receipt time
    edited after signing).  `memory_challenger.SCHEMES` accepts S1's scheme alone, so the real verdict rejects every test-signed session
    and no profile is issued; a session's public part is its commitment, its verdict and its coin openings."""
    bench = _bench()
    assert bench.IMPLS["p2-v3-mc-gpu"] is bench.P2V3McGpu
    body = bench.P2V3McGpu().run(_args(tmp_path, k=105), WS, tmp_path, backend=_backend(), params=_small(), research=True)
    gates, m = body["gates"], body["metrics"]
    for g in ("vector", "round_trip", "reference_decode", "kernel_flags", "codec", "timed_audit", "value_control",
              "late_control", "recompute_control", "tampered_control", "signature_control", "fail_closed"):
        assert gates[g]["passed"], (g, gates[g])
    assert "certificate" not in gates and body["profile"] is None
    assert body["implementation"] == {**body["implementation"], "name": "p2-v3-mc-gpu", "memory_challenger": "memory_challenger/live.py",
                                      "memory_challenger_args": ["--signer", "test"]}
    nb = _small().width_bytes
    raw = (tmp_path / "leaves-p2.bin").read_bytes()
    assert len(raw) == 64 * body["blocks"]
    honest = m["audit"]
    assert honest["memory_challenger"] == "memory_challenger.MemoryChallenger" and honest["limit_ns"] == 854_000 and honest["limit_fixed_before_load"]
    assert honest["audits"] == honest["audits_accepted_under_signer"] == 2 and honest["audits_accepted"] == 0
    assert honest["answers"] == 2 * 105 and honest["profile"] is None and honest["signer"]["test"]
    assert honest["signer"]["scheme"] == "test/lms-sha512/n64-w1-h1/v0"
    for rec in honest["records"]:
        assert set(rec["public"]) == {"commitment", "verdict", "openings"} and rec["public"]["verdict"] == {"accepted": False}
        assert rec["witness"]["reasons"] == ["signature scheme 'test/lms-sha512/n64-w1-h1/v0' is not accepted"]
        assert rec["witness"]["reasons_under_signer"] == []
    tampered = m["tampered_control"]
    assert tampered["wrong"] > 0 and tampered["late"] == 0 and tampered["audits_accepted_under_signer"] == 0
    assert all(any(r.endswith("answers are not their blocks of C") for r in rec["witness"]["reasons_under_signer"])
               for rec in tampered["records"])
    assert m["late_control"]["late"] == 2 and all(len(rec["answers"]) == 10 for rec in m["late_control"]["records"])
    assert m["signature_control"] == {"sessions": 2, "unedited_accepted": 2, "flipped_rejected": 2, "edited_rejected": 2,
                                      "passed": True}
    assert len(bytes.fromhex(honest["records"][0]["public"]["openings"][0]["salt"])) == 192 and nb == 16


def _mc_at_the_p2_point(limit_ns: int, tamper: frozenset[int] = frozenset(), audits: int = 2, **kw):
    """`memory_challenger/live.py`'s audit, strict, of `p2-16448/v4` at its registered point (one segment), on the logical
    clock, against a responder whose blocks are the digests' (`tamper`'s with their last byte's top bit flipped)."""
    sys.path.insert(0, str(BENCH))
    from band_gpu import harness as H
    from memory_challenger import live

    from verity.core.protocols.pous import schemes

    scheme = schemes.get("p2-16448/v4")
    nb = scheme.params.block_bytes
    block = lambda i: i.to_bytes(8, "big") * (nb // 8)  # noqa: E731
    digests = b"".join(_p2()[1].MC.leaf(None, i, block(i)) for i in range(scheme.params.blocks))
    responder = H.ControlResponder(nb, H.control_answers(b"".join(block(i) for i in range(scheme.params.blocks)), nb,
                                                         tamper=set(tamper)))
    try:
        return live.audit("127.0.0.1", responder.port(), scheme, b"s" * 24, digests, audits=audits, pings=20,
                          limit_ns=limit_ns, clock=live.LogicalClock(), **kw), responder.answered()
    finally:
        responder.stop()


def test_the_memory_challenger_audits_p2_v3_at_the_rtt_cap_and_refuses_one_us_past_it():
    """`Δ + 354 µs` is inside `P2SlackFamilyUncond`'s certificate: 105 rounds a session run, the test-signed sessions pass
    under the test signer's mapping and the real verdict rejects them; `Δ + 355 µs` is refused before any round.  Every
    session is bound to the root the setup publishes for the responder's `C*`."""
    from verity.core.protocols.pous import schemes
    from verity.core.protocols.pous.schemes import p2
    from verity.core.protocols.pous.verifier import VerifierKey, leaf_hash

    doc, answered = _mc_at_the_p2_point(500_000 + p2.RTT_ALLOWANCE_CAP_NS)
    assert doc["strict"] and "refused" not in doc and answered == 2 * 105 and doc["k"] == 105
    assert doc["audits"] == doc["audits_accepted_under_signer"] == 2 and doc["audits_accepted"] == 0
    assert doc["late"] == doc["wrong"] == 0 and doc["profile"] is None
    params, salt = schemes.get("p2-16448/v4").params, b"s" * 24
    blocks = [i.to_bytes(8, "big") * (params.block_bytes // 8) for i in range(params.blocks)]
    root = VerifierKey.from_leaves(params, salt, [leaf_hash(params, salt, i, b) for i, b in enumerate(blocks)]).root.hex()
    assert doc["setup_root"] == root and all(rec["public"]["commitment"]["binding"] == root for rec in doc["records"])
    rec = doc["records"][0]
    assert rec["witness"]["expected"] == {str(a["index"]): rec["witness"]["signed"]["rounds"][j]["digest"]
                                          for j, a in enumerate(rec["answers"])}
    doc, answered = _mc_at_the_p2_point(500_000 + p2.RTT_ALLOWANCE_CAP_NS + 1_000)
    assert doc["audits"] == 0 and answered == 0 and doc["records"] == []
    assert "fits 1 calls of 854029 ns" in doc["refused"]


def test_the_memory_challenger_catches_a_flipped_bit_in_a_blocks_last_byte():
    """Every block tampered: each round's digest is wrong, each session rejected, and no round is late or retried."""
    from verity.core.protocols.pous.schemes import p2

    doc, answered = _mc_at_the_p2_point(500_000 + p2.RTT_ALLOWANCE_CAP_NS, tamper=frozenset(range(1 << 16)))
    assert answered == 2 * 105 and doc["wrong"] == 2 * 105 and doc["late"] == 0
    assert doc["audits_accepted_under_signer"] == doc["audits_accepted"] == 0


@pytest.mark.parametrize("record", ["lean", "python"])
def test_the_memory_challenger_records_both_verdicts_whichever_is_the_records(record, monkeypatch):
    """`judge` runs the Lean verdict and the Python's on every session and keeps both in the witness; `RECORD_VERDICT` picks
    the record's, and where the two agree the record is the same either way."""
    from verity.core.protocols.pous.schemes import p2

    live = _live()
    monkeypatch.setattr(live, "RECORD_VERDICT", record)
    doc, _ = _mc_at_the_p2_point(500_000 + p2.RTT_ALLOWANCE_CAP_NS, audits=1)
    assert doc["audits_accepted_under_signer"] == 1 and doc["audits_accepted"] == 0
    for rec in doc["records"]:
        w = rec["witness"]
        assert w["record_verdict"] == record and w["reasons"] == ["signature scheme 'test/lms-sha512/n64-w1-h1/v0' is not accepted"]
        assert w["verdicts"] == {name: {"accepted": False, "reasons": w["reasons"]} for name in ("lean", "python")}
        assert w["verdicts_under_signer"] == {name: {"accepted": True, "reasons": []} for name in ("lean", "python")}


@pytest.mark.parametrize("record", ["lean", "python"])
@pytest.mark.parametrize("forced", ["lean", "python"])
def test_a_disagreement_between_the_verdicts_rejects_the_session(forced, record, monkeypatch):
    """`forced`'s verdict flips its accept bit: under `memory_challenger.SCHEMES` it accepts the test-signed sessions the other
    rejects, and under the signer's mapping it rejects the honest ones the other accepts.  Each session is rejected for the
    disagreement, with both verdicts named, whichever one accepted and whichever is the record's."""
    import json

    from verity.core.protocols.pous import memory_challenger as mc
    from verity.core.protocols.pous.schemes import p2

    live = _live()

    def flipped(verdict):
        def flip(*args, **kw):
            v = verdict(*args, **kw)
            return mc.Verdict(not v.accepted, () if not v.accepted else ("forced",), v.late, v.wrong)
        return flip

    monkeypatch.setattr(live, "RECORD_VERDICT", record)
    if forced == "python":
        monkeypatch.setattr(mc, "verdict", flipped(mc.verdict))
    else:
        lean = live.lean_verdict().verdict
        monkeypatch.setattr(live, "lean_verdict", lambda: types.SimpleNamespace(verdict=flipped(lean), server=lambda: None))
    other = "python" if forced == "lean" else "lean"
    doc, _ = _mc_at_the_p2_point(500_000 + p2.RTT_ALLOWANCE_CAP_NS, audits=1)
    assert doc["audits"] == 1 and doc["audits_accepted"] == doc["audits_accepted_under_signer"] == 0
    for rec in doc["records"]:
        w = rec["witness"]
        for verdicts, reasons, accepted in ((w["verdicts"], w["reasons"], forced), (w["verdicts_under_signer"],
                                                                                    w["reasons_under_signer"], other)):
            assert {name: v["accepted"] for name, v in verdicts.items()} == {forced: accepted == forced,
                                                                             other: accepted == other}
            assert reasons == [f"{live.DISAGREE}: lean {json.dumps(verdicts['lean'])}, python {json.dumps(verdicts['python'])}"]
        assert rec["public"]["verdict"] == {"accepted": False} and not rec["accepted_under_signer"]


def _live():
    sys.path.insert(0, str(BENCH))
    from memory_challenger import live

    return live


def test_the_signed_sessions_are_on_disk_before_any_verdict_runs(tmp_path, monkeypatch):
    """A verdict that fails after the sessions signed (r20261009-034919-c88c: `judge` could not import the Lean's server)
    takes nothing with it: every signed session is in `sessions_out` first, whole, and still verifies under its signer."""
    from verity.core.protocols.pous import memory_challenger as mc
    from verity.core.protocols.pous import schemes
    from verity.core.protocols.pous.schemes import p2

    live = _live()

    def failing(*_a, **_k):
        raise RuntimeError("the verdict failed")

    monkeypatch.setattr(live, "judge", failing)
    with pytest.raises(RuntimeError, match="the verdict failed"):
        _mc_at_the_p2_point(500_000 + p2.RTT_ALLOWANCE_CAP_NS, audits=2, sessions_out=tmp_path / "sessions.json")
    doc = json.loads((tmp_path / "sessions.json").read_text())
    assert doc["signer"]["scheme"].startswith(mc.TEST_PREFIX) and len(doc["sessions"]) == 2
    nb = schemes.get("p2-16448/v4").params.block_bytes
    leaf = lambda i: _p2()[1].MC.leaf(None, i, i.to_bytes(8, "big") * (nb // 8))  # noqa: E731
    for s in doc["sessions"]:
        signed = mc.SignedSession.from_json(s["signed"])
        assert len(s["reports"]) == len(signed.rounds) == 105
        v = mc.verdict(signed, mc.SessionCommitment.from_json(s["commitment"]),
                       [mc.CoinOpening.from_json(o) for o in s["openings"]], leaf,
                       trusted={bytes.fromhex(doc["signer"]["public_key"])}, schemes=live.signer_schemes(signed.scheme))
        assert v.accepted, v.reasons


def test_a_lean_verdict_that_cannot_load_stops_the_audit_before_any_signer_opens(tmp_path, monkeypatch):
    """The Lean verdict is loaded, built and serving before the first session signs: where it can't be, the audit raises
    before the signer's key, and so its leaf state, is opened, and no session file is written."""
    from verity.core.protocols.pous.schemes import p2

    live = _live()

    def missing():
        raise ModuleNotFoundError("No module named 'verity_difftest'")

    monkeypatch.setattr(live, "lean_verdict", missing)
    monkeypatch.setattr(live, "lms_signer", lambda *_a, **_k: pytest.fail("the signer was opened"))
    with pytest.raises(ModuleNotFoundError, match="verity_difftest"):
        _mc_at_the_p2_point(500_000 + p2.RTT_ALLOWANCE_CAP_NS, audits=1, signer="lms", lms_key=tmp_path / "key",
                            sessions_out=tmp_path / "sessions.json", session_log_path=tmp_path / "session-log")
    assert not any((tmp_path / f).exists() for f in ("key", "sessions.json", "session-log"))


def test_the_memory_challengers_lms_key_is_made_once_and_its_leaves_never_reused(tmp_path, monkeypatch):
    """`--lms-key DIR`: the first use makes the key there (seed, public key, tree, leaf state), a later one loads it without a
    keygen and signs at the next leaf, and a tree that is not the public key's is refused.  At a small height, so under a
    `test/` id."""
    from verity.ml.crypto import lms

    live, small = _live(), lms.Params(64, 1, 2)
    first = live.lms_signer(tmp_path / "key", params=small)
    assert first.scheme == "test/lms-sha512/n64-w1-h2/v0" and (tmp_path / "key" / "seed").stat().st_mode & 0o777 == 0o600
    a = lms.Signature.decode(small, first.sign(b"s" * 32, b"r"))
    first.state.close()
    monkeypatch.setattr(lms, "keygen", lambda *_a, **_k: pytest.fail("the key is made again"))
    again = live.lms_signer(tmp_path / "key", params=small)
    assert again.public_key == first.public_key
    assert (a.q, lms.Signature.decode(small, again.sign(b"s" * 32, b"r")).q) == (0, 1)
    again.state.close()
    tree = tmp_path / "key" / "tree.bin"
    raw = tree.read_bytes()
    tree.write_bytes(raw[:-1] + bytes([raw[-1] ^ 0xFF]))
    with pytest.raises(ValueError, match="is not the tree of"):
        live.lms_signer(tmp_path / "key", params=small)


class _SparseTree:
    """A deployed key's tree for leaf 0 alone (its path's siblings arbitrary, its root the one they give): one session
    signed at `lms.DEPLOYED` without its 2^20-leaf keygen."""

    def __init__(self, I: bytes, seed: bytes) -> None:
        from verity.ml.crypto import lms

        self.root = b""
        probe = lms.sign_at(lms.PrivateKey(lms.DEPLOYED, I, seed, self), 0, b"probe")
        self.root = lms.root_of(lms.DEPLOYED, I, b"probe", probe)

    def __getitem__(self, r: int) -> bytes:
        return self.root if r == 1 else hashlib.sha512(b"sibling%d" % r).digest()


def _deployed_signer(tmp_path):
    """An S1 signer at `lms.DEPLOYED` over `_SparseTree`, its leaf state in `tmp_path`, for one session."""
    from verity.ml.crypto import lms
    from verity.core.protocols.pous import memory_challenger as mc

    I, seed = hashlib.sha512(b"I").digest()[:16], hashlib.sha512(b"seed").digest()[:lms.DEPLOYED.N]
    key = lms.PrivateKey(lms.DEPLOYED, I, seed, _SparseTree(I, seed))
    lms.LeafState.create(tmp_path / "state", lms.public_key(key), lms.DEPLOYED.H)
    return mc.LmsSigner(key, lms.LeafState(tmp_path / "state", lms.public_key(key), lms.DEPLOYED.H))


def test_an_lms_signed_session_at_the_p2_point_is_accepted_and_issues_the_profile(tmp_path, monkeypatch):
    """With `--signer lms` the verdict is `memory_challenger.SCHEMES`'s and it accepts the honest session: the profile is
    issued under the certificate; the bad-signature control rejects it flipped or edited under the same mapping.  The
    session's id is in the session log, made with the new key."""
    from verity.core.protocols.pous import memory_challenger as mc
    from verity.core.protocols.pous.schemes import p2

    live, signer = _live(), _deployed_signer(tmp_path)
    monkeypatch.setattr(live, "lms_signer", lambda d, workers=1: signer)
    doc, answered = _mc_at_the_p2_point(500_000 + p2.RTT_ALLOWANCE_CAP_NS, audits=1, signer="lms",
                                        lms_key=tmp_path / "key", session_log_path=tmp_path / "session-log")
    assert answered == 105 and doc["signer"]["scheme"] == mc.LMS and doc["signer"]["test"] is False
    session = bytes.fromhex(doc["records"][0]["public"]["commitment"]["session"])
    assert doc["signer"]["session_log"] == str(tmp_path / "session-log") and session in mc.SessionLog(tmp_path / "session-log")
    assert doc["audits"] == doc["audits_accepted"] == doc["audits_accepted_under_signer"] == 1
    assert doc["records"][0]["public"]["verdict"] == {"accepted": True} and doc["profile"] is not None
    from verity.core.protocols.pous import schemes

    nb = schemes.get("p2-16448/v4").params.block_bytes
    ctl = live.signature_control(doc, lambda i: mc.block_digest(i.to_bytes(8, "big") * (nb // 8)))
    assert ctl == {"sessions": 1, "unedited_accepted": 1, "flipped_rejected": 1, "edited_rejected": 1, "passed": True}
    with pytest.raises(ValueError, match="'lms' with a key directory"):
        _mc_at_the_p2_point(500_000 + p2.RTT_ALLOWANCE_CAP_NS, audits=1, signer="lms")


def test_an_lms_signed_audit_takes_a_session_log_of_its_own_and_an_existing_key_needs_its_log(tmp_path, monkeypatch):
    """`--session-log FILE`: an LMS-signed audit refuses to start without one or with one inside `--lms-key`, and for a
    key that exists (its `seed` there) it refuses a missing log before any session, with no leaf spent and no log made."""
    from verity.core.protocols.pous import memory_challenger as mc
    from verity.core.protocols.pous import schemes
    from verity.core.protocols.pous.schemes import p2

    live, signer = _live(), _deployed_signer(tmp_path)
    monkeypatch.setattr(live, "lms_signer", lambda d, workers=1: signer)
    scheme, key, log = schemes.get("p2-16448/v4"), tmp_path / "key", tmp_path / "session-log"

    def audit(**kw):
        return live.audit("127.0.0.1", 0, scheme, b"s" * 24, bytes(64 * scheme.params.blocks), audits=1, pings=1,
                          limit_ns=500_000 + p2.RTT_ALLOWANCE_CAP_NS, clock=live.LogicalClock(), signer="lms", lms_key=key,
                          **kw)

    with pytest.raises(ValueError, match="takes the memory challenger's session log"):
        audit()
    with pytest.raises(ValueError, match="inside the key's directory"):
        audit(session_log_path=key / "sessions")
    key.mkdir()
    (key / "seed").write_bytes(bytes(64))
    with pytest.raises(mc.SessionLogError, match="missing"):
        audit(session_log_path=log)
    assert not log.exists() and json.loads((tmp_path / "state").read_text())["next"] == 0


def test_decode_bench_result_names_device_rate_setup_and_keys_agreement(tmp_path):
    sys.path.insert(0, str(BENCH))
    from p2_v1 import decode_bench as db

    every = {"blocks": 1 << 19, "ranges": 256, "passed": True, "failed_ranges": []}
    pinned = {"vector": "v4-deployment", "keys_version": 4, "passed": True, "K": True, "t": True}
    lib = {"lib": "/b/v4", "fused": False, "keys_version": 4, "blocks": 1 << 19, "setup_s": 0.5, "keys_s": 0.25,
           "keys_GBps": 4.3, "kernel_GBps": 70.0, "decode_GBps": 50.0, "build": {"name": "v4", "flags": "-DP2_KEYS4"},
           "gate": {"passed": True, "reference_blocks": 58, "reference_keys_equal": 58, "pinned_keys": pinned,
                    "keys_all": every}}
    doc = {"device": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "results": [lib], "passed": True}
    out = tmp_path / "out"
    out.mkdir()
    (out / "decode-bench.json").write_text("{}")
    smi = "3, GPU-abc, NVIDIA RTX PRO 6000 Blackwell Server Edition"
    r = db.result_doc(doc, smi, "r1", tmp_path, out, ["decode_bench.py"])
    assert r["schema"] == "research/result/v0.1" and r["run_id"] == "r1"
    assert r["device"] == {"name": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "index": 3, "uuid": "GPU-abc"}
    assert r["keys_per_s"] == (1 << 19) / 0.25 and r["setup_s"] == 0.5
    assert r["agreement"]["passed"] and r["validation"]["status"] == "passed"
    assert r["measurement_files"] == ["out/decode-bench.json"] == r["validation"]["evidence"]
    assert all({"name", "value", "unit"} <= set(m) for m in r["measurements"])
    lib["gate"]["keys_all"] = None
    assert not db.result_doc(doc, smi, "r1", tmp_path, out, [])["agreement"]["passed"]
    lib["gate"]["keys_all"] = every | {"passed": False}
    assert db.result_doc(doc, smi, "r1", tmp_path, out, [])["validation"]["status"] == "failed"
    lib["gate"]["keys_all"], lib["gate"]["reference_keys_equal"] = every, 57
    assert not db.result_doc(doc, "", "r1", tmp_path, out, [])["agreement"]["passed"]
    assert db.result_doc(doc, "", "r1", tmp_path, out, [])["device"]["index"] is None
