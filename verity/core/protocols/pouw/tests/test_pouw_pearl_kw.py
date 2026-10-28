"""``pearl-fp8-v4`` against Pearl's own Rust at 284b147b (``tests/vectors/pearl_kw.json``, B200), and the scheme through
the protocol on the RTX 4090's ``ada`` device."""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import pytest

import pearl_vectors as PV
from verity.ml.crypto.blake3 import blake3
from verity.core.protocols.pouw.protocol import Context, DomainError, Matmul, PoUWScheme, Shape, Tile
from verity.core.protocols.pouw.schemes import get
from verity.core.protocols.pouw.schemes import pearl_kw as P

# the clear audit is a reference tool (benchmarks/pouw/clear), imported by name
sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "benchmarks" / "pouw" / "clear"))

PINNED = json.loads(PV.PINNED.read_text())
OUT = PINNED["outputs"]
IN = PV.inputs()


def f32s_hex(xs) -> str:
    return b"".join(struct.pack("<I", P.f32_bits(x)) for x in xs).hex()


def u16s_hex(xs) -> str:
    return struct.pack(f"<{len(xs)}H", *xs).hex()


def test_pinned_commit():
    assert PINNED["pearl_commit"] == PV.PEARL_COMMIT and PINNED["devices"] == ["b200", "h100"]


def test_hash_labelled_and_noise_lines():
    for c, want in zip(IN["hash_labelled"], OUT["hash_labelled"]):
        key = bytes.fromhex(c["key"]) if c["key"] else None
        assert P.hash_labelled(bytes.fromhex(c["msg"]), c["label"], key).hex() == want
    for c, want in zip(IN["lines"], OUT["lines"]):
        assert bytes(P.sample_line(bytes.fromhex(c["seed"]), c["side"], c["factor"], c["line"])).hex() == want


def test_bf16_ops():
    def safe(f, *a):
        try:
            return f(*a)
        except P.Reject:
            return None

    for (a, b, c), want in zip(IN["bf16_ops"], OUT["bf16_ops"]):
        assert [safe(P.bf16_mul, a, b), safe(P.bf16_div, a, b), safe(P.bf16_fma, a, b, c)] == want


def test_b200_matmul_and_partials():
    for c, want in zip(IN["matmul"], OUT["matmul"]):
        a, b, k = bytes.fromhex(c["a"]), bytes.fromhex(c["b"]), c["k"]
        outs, parts = [], []
        for i in range(c["m"]):
            for j in range(c["n"]):
                p: list[float] = []
                outs.append(P.device_dot(P.B200, a[i * k:(i + 1) * k], b[j * k:(j + 1) * k], p))
                parts.append(f32s_hex(p))
        assert f32s_hex(outs) == want["out"] and parts == want["partials"]
        if want["h100"] is not None:
            assert f32s_hex([P.device_dot(P.H100, a[i * k:(i + 1) * k], b[j * k:(j + 1) * k]) for i in range(c["m"]) for j in range(c["n"])]) == want["h100"]


def test_census_lambda_grid_skip():
    for c, want in zip(IN["census"], OUT["census"]):
        dev = P.B200 if c["device"] == 1 else P.H100
        lam = P.lam(dev, c["alpha"], c["x"], c["l2"])
        g = P.grid_exponent(dev, c["z"], c["m"])
        assert [lam, g, P.skip(lam, c["lb"], g)] == want


def test_layout():
    assert P.PATTERN.hex() == OUT["pattern"] and P.LANES == OUT["lanes"] and P.budget(1024, 16, 16) == OUT["budget"]


def side(c: dict, dev: P.CodeParams) -> tuple[P.Built, list, bytes, bytes]:
    k = PV.K
    ints, scales = bytes.fromhex(c["ints"]), struct.unpack(f"<{len(c['scales']) // 4}H", bytes.fromhex(c["scales"]))
    rows = [P.fp10_row([x - 256 if x >= 128 else x for x in ints[r * k:(r + 1) * k]], scales[r * k // 8:(r + 1) * k // 8])
            for r in range(c["rows"])]
    planes = [P.parse_row(r, k) for r in rows]
    clean = [P.open_row(a, s) for a, s in planes]
    norms = [P.exact_norms(a, s) for a, s in planes]
    seed, f_seed = bytes.fromhex(c["seed"]), bytes.fromhex(c["f_seed"])
    e = [P.sample_line(seed, c["side"], 0, i) for i in c["lines"]]
    f = [list(x) for x in P._f_basis(f_seed, c["side"], k)]
    return P.noisy_quantize(dev, clean, norms, e, f), norms, bytes(x for row in e for x in row), bytes(x for row in f for x in row)


@pytest.mark.slow
@pytest.mark.parametrize("dev,key", [(P.B200, "tiles"), (P.H100, "tiles_h100")], ids=["b200", "h100"])
@pytest.mark.parametrize("t", [0, 1])
def test_tile_matches_pearl(t, dev, key):
    """The full tile on B200: prequant planes -> norms -> noise -> noisy quantization -> device matmul -> extract -> J,
    and the jackpot policy's verdict (tile 1 has 5.9% dead entries, which the policy rejects)."""
    c, want = IN["tiles"][t], OUT[key][t]
    (a, na, ea, fa), (b, nb, eb, fb) = side(c["a"], dev), side(c["b"], dev)
    for got, norms, e, f, exp in ((a, na, ea, fa, want["a"]), (b, nb, eb, fb, want["b"])):
        assert [list(n) for n in norms] == exp["norms"]
        assert hashlib.sha256(e).hexdigest() == exp["e_sha256"]
        assert blake3(f).hex() == exp["f_sha"]
        assert (u16s_hex(got.alpha), u16s_hex(got.beta), u16s_hex(got.l2)) == (exp["alpha"], exp["beta"], exp["l2"])
        assert hashlib.sha256(bytes(x for row in got.codes for x in row)).hexdigest() == exp["codes_sha256"]
    scheme = P.PearlKW(dev.name)
    ops = P.Ops(a, b)
    tile = scheme._replay(ops)
    assert f32s_hex([x for row in tile for x in row]) == want["acc"]
    msg = P.extract(tile)
    assert msg.hex() == want["msg"]
    assert P.hash_labelled(msg, "jackpot", bytes.fromhex(c["seed_a"])).hex() == want["jackpot"]
    ctx = Context(b"\0" * 32, Matmul(0, Shape(16, PV.K, 16), 0), b"\0" * 32, b"\0" * 32)
    assert scheme.admissible(ctx, ops) == want["approved"]


def test_interface_and_domain():
    s = get("pearl-fp8-v4")
    assert isinstance(s, PoUWScheme) and s.name == "pearl-fp8-v4" and s.device is P.ADA and s.certificate.gamma is None
    for shape in ((16, 1000, 16), (16, 1056 - 16, 16), (16, 1 << 17, 16), (8, 1024, 16)):
        with pytest.raises(DomainError):
            s.check(Matmul(0, Shape(*shape), 0))
    for k, n in ((2048, 2560), (2048, 2048), (2048, 22016), (11008, 2048)):   # Qwen2.5-3B
        s.check(Matmul(0, Shape(16, k, n), 0))


@pytest.mark.slow
def test_ada_tile_through_the_protocol():
    """One honest 16 x 16 tile on the 4090's device: admissible, digest reproducible, useful output near A·Bᵀ."""
    from verity.core.protocols.pouw.audit import Epoch, Workload
    from clear_audit import Prover, Sampled, Verifier

    s, k = get("pearl-fp8-v4"), 1024
    c = IN["tiles"][0]
    ints_a, ints_b = bytes.fromhex(c["a"]["ints"]), bytes.fromhex(c["b"]["ints"])
    sa = struct.unpack(f"<{len(c['a']['scales']) // 4}H", bytes.fromhex(c["a"]["scales"]))
    sb = struct.unpack(f"<{len(c['b']['scales']) // 4}H", bytes.fromhex(c["b"]["scales"]))
    rows_a = [P.fp10_row([x - 256 if x >= 128 else x for x in ints_a[r * k:(r + 1) * k]], sa[r * k // 8:(r + 1) * k // 8]) for r in range(16)]
    rows_b = [P.fp10_row([x - 256 if x >= 128 else x for x in ints_b[r * k:(r + 1) * k]], sb[r * k // 8:(r + 1) * k // 8]) for r in range(16)]
    work = Workload((Matmul(3, Shape(16, k, 16), 9),))
    prover = Prover(s, {9: rows_b})
    epoch = Epoch.unverified(work, prover.weight_roots, b"beacon")
    tr = prover.prove(epoch, {3: rows_a})
    audit = Verifier(s, epoch).audit(prover, tr, Sampled(2), b"verifier")
    assert audit.accepted and audit.profile.certified(0.5) is None
    fa = [[P.bf16_to_f32(x) for x in P.open_row(*P.parse_row(r, k))] for r in rows_a]
    fb = [[P.bf16_to_f32(x) for x in P.open_row(*P.parse_row(r, k))] for r in rows_b]

    def fp8_row(row):                              # the baseline: plain per-row FP8 quantization, no noise
        s = 448 / max(abs(x) for x in row)
        return [P.fp8_to_f32(P.f32_to_fp8(x * s)) / s for x in row]

    def rel_rms(got, exact):
        num = sum((g - e) ** 2 for gr, er in zip(got, exact) for g, e in zip(gr, er))
        return (num / sum(e * e for er in exact for e in er)) ** 0.5

    exact = [[sum(x * y for x, y in zip(a, b)) for b in fb] for a in fa]
    plain = [[sum(x * y for x, y in zip(a, b)) for b in map(fp8_row, fb)] for a in map(fp8_row, fa)]
    pearl, fp8 = rel_rms(tr.useful[0], exact), rel_rms(plain, exact)
    assert pearl < 0.12 and pearl < 3 * fp8          # measured: 7.8% against 3.8% for plain FP8 (δ = 1 noise, as large as the signal)
