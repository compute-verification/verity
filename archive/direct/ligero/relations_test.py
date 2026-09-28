"""The Hopper relations on the B-Ligero prover (lane b-hopper), one code path per operand format parametrised by the model:

* the unit at the relation's params against ``model.step`` on 2,000 random finite cases + the D-family edge cases: accepted,
  the outputs the model's words;
* the OTHER pipeline's word (Ampere for Hopper BF16, Ada for Hopper FP8) where it differs: rejected -- the relation pins
  the Hopper adder, not just "a BF16/FP8 dot";
* one witness-row mutation per (gadget scope, row class): rejected;
* the K=1536 chain (two VUs) proved and verified on the CPU, non-ZK interactive and ZK Fiat-Shamir, and the runner's
  negative battery (claimed-word bits, off-domain operands, another statement, broken links, c_0 != 0, accumulator
  bits, witness rows) rejected by the Python verifier;
* the statement files round-trip (v2 for bf16-hopper -- the Rust verifier's format -- and v3 for fp8-hopper) and the
  Python file verifier accepts them and rejects a claimed-word flip.

Needs torch (``importorskip``): runs on the pod, skips on the laptop.  CPU only, small l.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("blake3")
from verity.errors import InvalidArtifact
from verity.ml.circuits.silicon.cast import f32_to_bf16_word
from verity.ml.circuits.silicon.models import ADA_E4M3_M16N8K32, AMPERE_BF16_M16N8K16, tc_dot_chain

from . import protocol as protocol_mod
from .compile import P
from .relations import RELATIONS, Relation, random_finite_bf16
from .relchain import RelationChainRunner, instances, negatives
from .witness import build_witness, check_constraints, eval_expr, tables

HOPPER = [RELATIONS["bf16-hopper"], RELATIONS["fp8-hopper"]]
OTHER = {"bf16-hopper": AMPERE_BF16_M16N8K16, "fp8-hopper": ADA_E4M3_M16N8K32}
N_HONEST = 2000


@pytest.fixture(scope="module", params=HOPPER, ids=lambda r: r.name)
def unit(request):
    rel = request.param
    s = rel.compile(rel.params, False)
    return rel, s, tables(s, "cpu")


def _finite_fp32(rng: np.random.Generator, n: int, exp_lo: int = 60, exp_hi: int = 200) -> np.ndarray:
    kind = rng.integers(0, 16, size=n)
    sign = rng.integers(0, 2, size=n, dtype=np.int64) << 31
    t = rng.integers(exp_lo, exp_hi, size=n, dtype=np.int64)
    f = rng.integers(0, 1 << 23, size=n, dtype=np.int64)
    c = sign | (t << 23) | f
    c = np.where(kind == 0, sign, c)                                   # +-0
    c = np.where(kind == 1, sign | f, c)                               # subnormal
    return c


def _cases(rel: Relation, n: int, seed: int):
    """Random finite operands and accumulators inside the model's domain (a case whose step leaves the finite range is
    redrawn), plus the D-family: zero / negative-zero accumulators, all-zero products, exact cancellation, subnormal
    operands, and a large accumulator against small products (the adder's window)."""
    rng = np.random.default_rng(seed)
    k = rel.k
    a = rel.operands(rng, (n, k))
    b = rel.operands(rng, (n, k))
    c = _finite_fp32(rng, n)
    one = 0x3F80 if rel.params.word_bits == 16 else 0x38
    neg = 0x8000 if rel.params.word_bits == 16 else 0x80
    tiny = 0x0001
    d = 0
    c[d] = 0; a[d] = 0; b[d] = 0; d += 1                               # D1: +0 + 0 products
    c[d] = 0x80000000; a[d] = 0; b[d] = 0; d += 1                      # D2: -0 + 0 products
    c[d] = 0x3F800001; a[d] = 0; b[d] = 0; d += 1                      # D3: pass-through accumulator (truncated on FP8)
    c[d] = 0; a[d] = one; b[d] = one; d += 1                           # D4: k x 1.0
    c[d] = 0; a[d, :] = one; a[d, 1::2] = one | neg; b[d] = one; d += 1  # D5: exact cancellation to zero
    c[d] = 0x80000000; a[d, :] = one; a[d, 1::2] = one | neg; b[d] = one; d += 1  # D6: ... from -0
    c[d] = 0x3F800000; a[d] = tiny; b[d] = one; d += 1                 # D7: subnormal products below the floor
    c[d] = 0x00800001; a[d] = tiny; b[d] = tiny; d += 1                # D8: subnormal accumulator, tiny products
    c[d] = 0x5F000000; a[d] = one; b[d] = one; d += 1                  # D9: large accumulator, products fall off the window
    c[d] = 0x3F800000; a[d, 0] = one; b[d, 0] = one | neg; a[d, 1:] = 0; b[d, 1:] = 0; d += 1  # D10: 1 - 1 = 0 via a product
    for i in range(d, n):
        while True:
            try:
                rel.model.step(int(c[i]), [int(x) for x in a[i]], [int(x) for x in b[i]])
                break
            except InvalidArtifact:
                c[i] = _finite_fp32(rng, 1)[0]
                a[i] = rel.operands(rng, (k,))
    return c, a, b


def _oracle(rel: Relation, c, a, b) -> np.ndarray:
    return np.asarray([rel.model.step(int(c[i]), [int(x) for x in a[i]], [int(x) for x in b[i]]) for i in range(len(c))], dtype=np.int64)


def _witness(unit, c, a, b, y):
    rel, s, tb = unit
    pub = rel.public_vectors(rel.params, c, a, b, y)
    W = build_witness(s, pub, rel.hints(rel.params, s, c, a, b, y, "cpu"), "cpu")
    return W, pub, check_constraints(s, tb, W, pub)


def _words(s, W):
    ys, yt, yf = (eval_expr(s.outputs[n], W) for n in ("y.s", "y.t", "y.f"))
    return ((ys << 31) | (yt << 23) | yf).numpy()


def test_params_come_from_the_model():
    r = RELATIONS["bf16-hopper"]
    assert (r.params.groups, r.params.width, r.params.floor, r.params.sig, r.params.exp) == ((16,), 26, -133, 8, 8)
    assert not r.params.lossy and r.k == 16 and r.steps == 96 and r.y_bits == 16 and r.tag == b""
    r = RELATIONS["fp8-hopper"]
    assert (r.params.groups, r.params.width, r.params.floor, r.params.sig, r.params.op_exp) == ((32,), 14, -139, 4, 4)
    # FP8Params are adder-side (acc_sig == width == 14: the FP32 significand truncated by ``rescale`` = 10 bits first)
    assert r.params.rescale == 10 and r.k == 32 and r.steps == 48 and r.y_bits == 22 and r.tag == b"|rel=fp8-hopper|v1"
    assert RELATIONS["fp8-ada"].params.groups == (16, 16)


def test_honest_random_and_edge_cases_accepted(unit):
    rel, s, _ = unit
    c, a, b = _cases(rel, N_HONEST, 1)
    y = _oracle(rel, c, a, b)
    W, _, bad = _witness(unit, c, a, b, y)
    assert bad == []
    assert (_words(s, W) == y).all()
    if rel.reference_hints is not None:                                # the torch generator is gen_step
        units = [(int(c[i]), [int(x) for x in a[i]], [int(x) for x in b[i]], int(y[i])) for i in range(64)]
        ref = rel.reference_hints(rel.params, s, units, "cpu")
        alt = rel.hints(rel.params, s, c[:64], a[:64], b[:64], y[:64], "cpu")
        assert torch.equal(ref, alt)


def test_the_other_pipelines_word_is_rejected(unit):
    rel, s, tb = unit
    other = OTHER[rel.name]
    c, a, b = _cases(rel, 3000, 2)
    y = _oracle(rel, c, a, b)
    idx = []
    for i in range(len(c)):
        try:
            yo = other.step(int(c[i]), [int(x) for x in a[i]], [int(x) for x in b[i]])
        except InvalidArtifact:
            continue
        if yo != int(y[i]) and (rel.params.word_bits == 16 or (yo & 0x3FF) == 0):
            idx.append((i, yo))
        if len(idx) == 40:
            break
    assert len(idx) >= 20, f"{rel.name}: the cases do not separate {rel.model.name} from {other.name}"
    sel = np.asarray([i for i, _ in idx])
    y_other = np.asarray([yo for _, yo in idx], dtype=np.int64)
    W, pub, bad = _witness(unit, c[sel], a[sel], b[sel], y[sel])
    assert bad == [] and (_words(s, W) == y[sel]).all()
    pub_bad = rel.public_vectors(rel.params, c[sel], a[sel], b[sel], y_other)
    assert check_constraints(s, tb, W, pub_bad), f"{rel.name}: {other.name}'s words accepted"


def test_row_mutations_rejected(unit):
    rel, s, tb = unit
    c, a, b = _cases(rel, 64, 3)
    y = _oracle(rel, c, a, b)
    W, pub, bad = _witness(unit, c, a, b, y)
    assert bad == []
    seen, picks = set(), []
    for r in s.rows:
        key = (".".join(r.name.split(".")[:2]), r.kind)
        if key not in seen:
            seen.add(key)
            picks.append(r)
    rng = np.random.default_rng(7)
    for r in picks:
        Wm = W.clone()
        col = int(rng.integers(0, 64))
        Wm[r.idx, col] = (Wm[r.idx, col] + int(rng.integers(1, 3))) % P
        assert check_constraints(s, tb, Wm, pub), f"mutation of {r.name} ({r.kind}) accepted"


def test_off_domain_operands_are_refused_by_the_public_decode(unit):
    rel, s, tb = unit
    c, a, b = _cases(rel, 16, 4)
    y = _oracle(rel, c, a, b)
    for w in rel.off_domain:
        bad = a.copy()
        bad[3, 4] = w
        with pytest.raises(ValueError):
            rel.public_vectors(rel.params, c, bad, b, y)


@pytest.mark.parametrize("rel", HOPPER, ids=lambda r: r.name)
def test_instances_are_the_model_chain(rel):
    vus = instances(rel, 2, seed=5)
    for a, b, accs, y in vus:
        assert len(a) == len(b) == 1536 and len(accs) == rel.steps
        assert tc_dot_chain(rel.model, 0, a, b) == y == accs[-1]
        k = rel.k
        c = 0
        for s_, acc in enumerate(accs):
            c = rel.model.step(c, a[k * s_:k * s_ + k], b[k * s_:k * s_ + k])
            assert c == acc
    if rel.name == "bf16-hopper":
        assert rel.y_public(vus[0][3]) == f32_to_bf16_word(vus[0][3])


def test_bf16_hopper_operand_recipe_is_finite_and_mixed():
    rng = np.random.default_rng(0)
    w = random_finite_bf16(rng, (4096,))
    assert ((w >> 7) & 0xFF).max() < 0xFF and (w >= 0).all() and (w < 65536).all()
    assert ((w & 0x7FFF) == 0).sum() > 200 and ((((w >> 7) & 0xFF) == 0) & ((w & 0x7F) != 0)).sum() > 200


@pytest.mark.parametrize("rel", HOPPER, ids=lambda r: r.name)
@pytest.mark.parametrize("zk,mode", [(False, protocol_mod.MODE_INTERACTIVE), (True, protocol_mod.MODE_FIAT_SHAMIR)])
def test_chain_two_vus_cpu(rel, zk, mode):
    R = RelationChainRunner(rel, "cpu", -128.0, zk=zk, mode=mode)
    vus = instances(rel, 2, seed=5)
    proof, pubs, lay = R.prove_vus(vus, check=True)
    ok, why = R.verify_vus(proof, pubs, lay)
    assert ok, why
    a, b, ypub = pubs
    assert int(ypub[rel.steps - 1]) == rel.y_public(vus[0][3]) and int(ypub[2 * rel.steps - 1]) == rel.y_public(vus[1][3])


@pytest.mark.parametrize("rel", HOPPER, ids=lambda r: r.name)
def test_negative_battery_rejected(rel):
    R = RelationChainRunner(rel, "cpu", -128.0, zk=False, mode=protocol_mod.MODE_FIAT_SHAMIR)
    negs = negatives(R, instances(rel, 2, seed=5), row_negatives=12)
    accepted = [name for name, (ok, _) in negs if ok]
    assert accepted == [], f"{rel.name}: negatives accepted: {accepted}"
    assert len(negs) >= rel.y_bits + 2 + len(rel.off_domain) + 1 + 1 + 4 + 12


@pytest.mark.parametrize("rel", HOPPER, ids=lambda r: r.name)
def test_statement_files_round_trip_and_verify(rel, tmp_path):
    """Every registered relation's statement file is the relation-named ``ligero-statement/v4`` (main's format, magic
    LIGSTM04: word widths + relation name in the header) -- bf16-hopper included, although its words are BF16: the file
    names the relation, so no verifier can mistake it for the Ampere relation's v2."""
    from .serialize import STMT_MAGIC_V4, _statement_of, proof_bytes, read_statement, statement_bytes, verify_files
    R = RelationChainRunner(rel, "cpu", -128.0, zk=False, mode=protocol_mod.MODE_FIAT_SHAMIR)
    assert R.relation == rel.name
    vus = instances(rel, 1, seed=9)
    proof, pubs, lay = R.prove_vus(vus, check=False)
    st = _statement_of(R, pubs, lay, R.cfg(lay.l), 1)
    assert st.v4 and st.relation == rel.name and st.auth is None
    assert (st.word_bytes, st.y_bytes) == (np.dtype(rel.word_dtype).itemsize, np.dtype(rel.y_dtype).itemsize)
    data = statement_bytes(st)
    assert data[:8] == STMT_MAGIC_V4 == b"LIGSTM04"
    rt = read_statement(data)
    assert np.array_equal(rt.a, st.a) and np.array_equal(rt.b, st.b) and np.array_equal(rt.y16, st.y16)
    assert (rt.word_bytes, rt.y_bytes, rt.relation, rt.auth) == (st.word_bytes, st.y_bytes, rel.name, None)
    sp, pp = tmp_path / "s.stmt", tmp_path / "p.proof"
    sp.write_bytes(data)
    pp.write_bytes(proof_bytes(proof))
    ok, why, _ = verify_files(sp, pp, "cpu", relation=rel.name)
    assert ok, why
    ok, why, _ = verify_files(sp, pp, "cpu")                          # the runner is picked from the relation named in the file
    assert ok, why
    ok, why, _ = verify_files(sp, pp, "cpu", relation="bf16")         # asked for another relation: refused by name, not by digest
    assert not ok and rel.name in why, why
    bad = read_statement(data)
    bad.y16 = bad.y16.copy()
    bad.y16[rel.steps - 1] ^= 1
    (tmp_path / "bad.stmt").write_bytes(statement_bytes(bad))
    ok, why, _ = verify_files(tmp_path / "bad.stmt", pp, "cpu", relation=rel.name)
    assert not ok


def test_registered_relations_take_only_the_vu_commands():
    from .run import main
    with pytest.raises(SystemExit):
        main(["--relation", "bf16-hopper", "gate"])                     # only gate-vu / bench-vu for the registered relations
