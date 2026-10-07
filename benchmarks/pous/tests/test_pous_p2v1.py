"""P2's arm of the POUS-only harness (`benchmarks/pous/p2_v1/`, `bench.py --impl p2-v1-gpu`) on CPU: the reference
(`p2_v1/reference.py`) against the spec's pinned vectors, and every step of the arm with the reference standing in for the GPU
library at the kit's 128-bit width, `test_pous_harness`'s Python responder for the native one, and the arm's own verifier process
(`p2_v1/memory_challenger.py`) on its logical clock, so no outcome depends on a clock."""
from __future__ import annotations

import importlib.util
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
    """The reference as one setup's kernels, on CPU rows."""

    def __init__(self, params, salt, model, flip: bool, says: int = 4) -> None:
        self.params, self.salt, self.model, self.flip, self.keys_version = params, salt, model, flip, says

    def _keys(self, i):
        R, _, _ = _p2()
        bps = self.params.blocks_per_segment
        k, t, _ = R.keys(self.salt, self.params, self.model, i // bps, i % bps)
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

    return harness.Backend("cpu", lambda params, salt, model: _Bound(params, salt, model, flip, says),
                           lambda c, cpu: PyResponder(c), timer)


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

    from verity.primitives.commitments.identity import tagged_sha512
    from verity.primitives.commitments.rowleaf import sha512_row_v2_digest
    from verity.primitives.randomness import frame

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
    from verity.protocols.accounting.space.pous import audit
    from verity.protocols.accounting.space.pous.schemes import p2

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
    _, harness, _ = _p2()
    P = _small().with_segments(1)
    salt, model = b"s" * 32, b"m" * 64
    bound = _Bound(P, salt, model, False)
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
