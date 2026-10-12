"""The hashed relation (``hashchain.py`` / ``hashauth.py`` / statement v5): census, gadget digest == committer digest,
file round trips, the extended chain tables, and the committed-operand negatives -- on the CPU, two VUs."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from . import poseidon2 as h2  # noqa: E402  (the reference + the numpy row-vectorised evaluations)

from . import chain, hashauth, hashchain  # noqa: E402
from .compile import P  # noqa: E402
from .relations import relation  # noqa: E402
from .relchain import HashedRelationRunner, hashed_negatives, instances, tile_instances  # noqa: E402
from .serialize import (  # noqa: E402
    HASHED_STATEMENT_TRIMMED,
    STMT_MAGIC_V5,
    STMT_MAGIC_V6,
    SYS_MAGIC_V3,
    _statement_of,
    proof_bytes,
    read_proof,
    read_statement,
    read_system,
    statement_bytes,
    system_bytes,
    verify_files,
)

CENSUS = {"bf16-hopper": (3292, 608, 1720), "fp8-ada": (3769, 704, 1720), "fp8-hopper": (3396, 704, 1720),
          # lane hash-compose: the Ampere BF16 v1 unit (== vu.py's REAL system) on the generic runner
          "bf16-ampere": (None, 608, 1720)}


@pytest.mark.parametrize("name", sorted(CENSUS))
def test_compose_census(name: str) -> None:
    hr = hashchain.compose(relation(name))
    base, private, hashed = CENSUS[name]
    c = hr.census
    if base is None:                       # bf16-ampere: the bare unit's row count is vu.py's (compile_unit(REAL))
        from .unit import compile_unit
        from verity_numerical.checker.params import REAL
        assert relation(name).params == REAL
        base = compile_unit(REAL, chain=True).m
    assert (c["rows_base"], c["rows_private_operands"], c["rows_hash"], c["rows_digest_pins"]) == (base, private, hashed, 17)
    assert c["sbox_rows_per_permutation"] == 852 and c["permutations_per_unit"] == 2 and c["linked_rows"] == 19
    assert [n for _, n in hr.sys.pins] == [hashchain.HASH_PIN_END] + [hashchain.hash_pin(x) for x in range(16)]
    assert not any(r.kind == "pin" and not r.name.startswith("hash.") for r in hr.sys.rows)
    assert len(hr.sys.chain["c"]) == len(hr.sys.chain["y"]) == 19


@pytest.fixture(scope="module")
def runner():
    rel = relation("fp8-ada")
    R = HashedRelationRunner(rel, "cpu", -128.0, zk=False, mode="fiat-shamir")
    data = instances(rel, 4)
    R.commit_vus(data)
    return R, data


def test_prove_verify_and_committer_digest(runner) -> None:
    R, data = runner
    proof, pubs, lay = R.prove_vus(data[:2], check=True, vu_ids=[0, 1])
    table, idx, ypub, auth, sibs = pubs
    ends = np.arange(2) * lay.steps + lay.steps - 1
    assert np.array_equal(table[ends, 1:9], R.digests["a"][:2]) and np.array_equal(table[ends, 9:], R.digests["b"][:2])
    assert table[:, 0].sum() == 2 and not table[np.setdiff1d(np.arange(lay.l), ends)].any()
    assert R.verify_vus(proof, pubs, lay) == (True, "")
    # the committer on the words the prover used, independently
    a0 = np.asarray(data[0][0]); b1 = np.asarray(data[1][1])
    assert h2.hash_row(a0.tolist(), 8, h2.ROLE_X) == table[ends[0], 1:9].tolist()
    assert h2.hash_row(b1.tolist(), 8, h2.ROLE_W) == table[ends[1], 9:].tolist()


def test_statement_v5_and_system_v3_round_trip(runner, tmp_path) -> None:
    R, data = runner
    proof, pubs, lay = R.prove_vus([data[1], data[3]], check=False, vu_ids=[1, 3])
    st = _statement_of(R, pubs, lay, R.cfg(lay.l), 1)
    raw = statement_bytes(st)
    # no operand words: ~1.5 kB for two VUs; v6 (the default, lane tier0-bytes) drops the l-vector's zero padding too
    assert raw[:8] == (STMT_MAGIC_V6 if HASHED_STATEMENT_TRIMMED else STMT_MAGIC_V5) and len(raw) < 2000
    st2 = read_statement(raw)
    assert st2.v5 and st2.authentication == "included-hash" and st2.relation == "fp8-ada"
    assert st2.trimmed == HASHED_STATEMENT_TRIMMED and statement_bytes(st2) == raw
    assert np.array_equal(st2.a, st.a) and np.array_equal(st2.b, st.b) and np.array_equal(st2.y16, st.y16)
    assert st2.hash_auth == st.hash_auth and st2.hash_sibs == st.hash_sibs
    sysb = system_bytes(R.sys)
    assert sysb[:8] == SYS_MAGIC_V3
    rd = read_system(sysb)
    assert rd["m"] == R.sys.m and len(rd["chain"]["c"]) == 19 and rd["chain"]["y16"] is not None
    (tmp_path / "s.stmt").write_bytes(raw)
    (tmp_path / "s.proof").write_bytes(proof_bytes(proof))
    assert read_proof((tmp_path / "s.proof").read_bytes()).root == proof.root
    ok, why, _ = verify_files(tmp_path / "s.stmt", tmp_path / "s.proof", "cpu", runner=R)
    assert ok, why
    # a v5 statement against the bare runner, and vice versa, is refused
    from .relchain import RelationChainRunner
    bare = RelationChainRunner(relation("fp8-ada"), "cpu", -128.0, zk=False, mode="fiat-shamir")
    ok, why, _ = verify_files(tmp_path / "s.stmt", tmp_path / "s.proof", "cpu", runner=bare)
    assert not ok and "authentication" in why


def test_negatives_rejected(runner) -> None:
    R, data = runner
    negs = hashed_negatives(R, data, row_negatives=3)
    accepted = [name for name, (ok, _) in negs if ok]
    assert len(negs) >= 20 and accepted == []
    names = " | ".join(n for n, _ in negs)
    for must in ("digest lane", "swapped", "multiproof sibling", "root replaced", "computed on the flipped row", "W column",
                 "hashed word", "carry link", "out of range"):
        assert must in names


def test_tile_sharing_layout(runner) -> None:
    rel = relation("fp8-ada")
    data, x, w = tile_instances(rel, 2, 3, seed=7)
    assert len(data) == 6 and x.shape == (2, 1536) and w.shape == (3, 1536)
    assert data[4][0] == x[1].tolist() and data[4][1] == w[1].tolist()
    R = HashedRelationRunner(rel, "cpu", -128.0, zk=False, mode="fiat-shamir")
    R.commit_vus(data, tile=(2, 3), x_rows=x, w_cols=w)
    assert R.trees["a"].count == 2 and R.trees["b"].count == 3 and R.trees["y"].count == 6
    proof, pubs, lay = R.prove_vus(data[3:6], check=False, vu_ids=[3, 4, 5])
    auth = pubs[3]
    assert auth.x_index == (1, 1, 1) and auth.w_index == (0, 1, 2)
    assert R.verify_vus(proof, pubs, lay) == (True, "")
    # a VU claiming the wrong column of the tile
    bad = dataclasses.replace(auth, w_index=(0, 2, 2))
    idx = bad.indices()
    assert R.verify_vus(proof, (pubs[0], idx, pubs[2], bad, pubs[4]), lay)[0] is False


def test_chain_tables_extend_the_three_link_case() -> None:
    """``chain_coefs`` for a system with 19 linked rows: the first three families are the accumulator's, the extras are
    monomials in the seven base families; a 3-link system's tables are exactly the pre-extension ones."""
    torch.manual_seed(0)
    lay = chain.Layout(16, 2, 8)
    rc = torch.randint(0, P, (2, chain.N_FAMILIES, lay.l), dtype=torch.int64)
    e0, e1, e6 = (chain.extra_coef(rc, e) for e in (0, 1, 6))
    assert torch.equal(e0, rc[:, 0] * rc[:, 6] % P) and torch.equal(e1, rc[:, 1] * rc[:, 6] % P)
    assert torch.equal(e6, rc[:, 0] * rc[:, 6] % P * rc[:, 6] % P)
    # the 3-link tables of a bare relation are the pre-extension ones (no extra virtual families)
    from .witness import tables
    sys3 = relation("fp8-ada").compile(relation("fp8-ada").params, True)
    tabs = chain._chain_tables(sys3, lay, "cpu")
    assert tabs["n_extra"] == 0 and tabs["famsel"].tolist() == [0, 1, 2, 3, 4, 5, 6] and tabs["link_fams"].tolist() == [0, 1, 2]
    hr = hashchain.compose(relation("fp8-ada"))
    tabs19 = chain._chain_tables(hr.sys, lay, "cpu")
    assert tabs19["n_extra"] == 32 and tabs19["link_fams"].tolist() == [0, 1, 2] + list(range(7, 23))


def test_permute_torch_matches_numpy_and_kat() -> None:
    """The whole-state torch permutation (the witness side) is ``permute_np`` / the Plonky3 KAT lane for lane."""
    rng = np.random.default_rng(7)
    st = rng.integers(0, P, size=(h2.WIDTH, 37), dtype=np.int64)
    inp, out = h2.KAT_24
    st[:, 0] = inp
    got = hashchain.permute_torch(torch.as_tensor(st)).numpy()
    assert got[:, 0].tolist() == list(out)
    assert np.array_equal(got, h2.permute_np(st))


def test_numpy_vectorised_poseidon2_equals_the_scalar_reference() -> None:
    """``poseidon2.permute_np`` / ``hash_rows_np`` / ``pack_words_np`` / ``row_digests`` (the committer's numpy paths, kept
    out of ``verity.ml.commitments`` by the package boundary) are the scalar reference lane for lane, row for row."""
    inp, out = h2.KAT_24
    assert h2.permute_np(np.asarray(inp, dtype=np.int64)[:, None])[:, 0].tolist() == list(out)
    rng = np.random.default_rng(11)
    cols = rng.integers(0, P, size=(h2.WIDTH, 5)).astype(np.int64)
    vec = h2.permute_np(cols)
    for j in range(5):
        assert vec[:, j].tolist() == h2.permute([int(x) for x in cols[:, j]])
    rows = rng.integers(0, 2**16, size=(7, 1536)).astype(np.int64)
    vec = h2.hash_rows_np(rows, 16, h2.ROLE_X)
    for i in range(7):
        assert vec[i].tolist() == h2.hash_row(rows[i].tolist(), 16, h2.ROLE_X)
    assert np.array_equal(h2.row_digests(rows, 16, h2.ROLE_X), vec)
    for bits, per in ((16, 1), (8, 2), (4, 4)):
        words = rng.integers(0, 2**bits, size=(50, 16 * per)).astype(np.int64)
        el = h2.pack_words_np(words, bits)
        assert el.shape == (50, 16) and el.max() < 2**16 < P
        assert el[3].tolist() == h2.pack_words(words[3].tolist(), bits)
        back = np.stack([(el >> (bits * q)) & (2**bits - 1) for q in range(per)], 2).reshape(50, 16 * per)
        assert np.array_equal(back, words)
    with pytest.raises(ValueError):
        h2.pack_words_np(np.array([[2**16]]), 16)


def test_fused_witness_source_modes() -> None:
    """Dense programs (the hashed systems) take the table-driven kernel; the v1 units' source is untouched."""
    from . import witness_device as wd
    rel = relation("fp8-ada")
    base = rel.compile(rel.params, True)
    hs = hashchain.compose(rel).sys
    assert wd._n_terms(base.program) <= wd.TAB_THRESHOLD < wd._n_terms(hs.program)
    assert wd.generate(base) == wd.generate(base, None)
    tab: list = []
    src = wd.generate(hs, tab)
    assert len(tab) > 50_000 and "affine(J, C," in src and len(src) < 1_000_000
    assert all(0 <= j < hs.m and 0 <= v < P for j, v in tab)
    # with the Poseidon2 witness hook the fused program loses exactly the 1704 S-box product rows and is v1-sized again
    hook, = hs.witness_hooks
    prog = wd._program(hs)
    assert len(hook.rows) == 1704 and len(hs.program) - len(prog) == 1704
    assert all(not (op[0] == "prod" and op[1] in hook.rows) for op in prog)
    assert wd._n_terms(prog) <= wd.TAB_THRESHOLD
    assert {r.kind for r in hs.rows if r.idx in hook.rows} == {"prod"}


def test_sponge_kernel_source_is_the_permutation() -> None:
    """The straight-line CUDA statements the device hook emits, executed in Python, are Poseidon2 (KAT) and store
    x^2, x^3, x^4, x^7 of each S-box input; the source names every S-box row of the compiled sponges."""
    import re
    em: list = []
    state = [hashchain._Sym(em, f"s[{i}]", P - 1) for i in range(h2.WIDTH)]
    stores: list = []

    def reduce(x):
        if isinstance(x, hashchain._Sym):
            return x if x.bound < P else x._new(f"{x.var} % P", P - 1)
        return int(x) % P

    def sbox(x):
        x2 = x._new(f"({x.var} * {x.var}) % P", P - 1)
        x3 = x._new(f"({x2.var} * {x.var}) % P", P - 1)
        x4 = x._new(f"({x2.var} * {x2.var}) % P", P - 1)
        x7 = x._new(f"({x4.var} * {x3.var}) % P", P - 1)
        stores.append((x.var, x2.var, x3.var, x4.var, x7.var))
        return x7

    out = h2.permute_ring(state, sbox, reduce)
    inp, exp = h2.KAT_24
    env = {f"s{i}": v for i, v in enumerate(inp)}
    pat = re.compile(r"\s*u64 (t\d+) = (.*);")
    for line in em:
        m = pat.match(line)
        assert m, line
        expr = re.sub(r"s\[(\d+)\]", r"s\1", m.group(2).replace("ULL", ""))
        val = eval(expr, {"P": P}, env)  # noqa: S307 -- our own emitted statements
        assert 0 <= val < 2**64
        env[m.group(1)] = val
    assert [env[o.var] for o in out] == list(exp)
    assert len(stores) == 213
    for x, x2, x3, x4, x7 in stores:
        assert env[x7] == pow(env[x], 7, P) and env[x2] == env[x] ** 2 % P and env[x3] == env[x] ** 3 % P and env[x4] == env[x] ** 4 % P
    src = hashchain.sponge_kernel_source(8, 2, 213)
    assert "poseidon2_rows" in src and src.count("W[(size_t)rows[") == 4 * 213
    with pytest.raises(ValueError):
        hashchain.sponge_kernel_source(8, 2, 212)


def test_row_sponge_kernel_permute_is_the_permutation() -> None:
    """The committer kernel's ``permute`` body, executed in Python, is Poseidon2 (KAT)."""
    import re
    from .leaf.poseidon2 import row_sponge_kernel_source
    src = row_sponge_kernel_source()
    body = src.split("permute(u64* s)\n{\n", 1)[1].split("\n}\n", 1)[0].splitlines()
    inp, exp = h2.KAT_24
    env = {f"s{i}": v for i, v in enumerate(inp)}
    out = {}
    tmp, fin = re.compile(r"\s*u64 (t\d+) = (.*);"), re.compile(r"\s*s\[(\d+)\] = (t\d+);")
    for line in body:
        m = tmp.match(line)
        if m:
            env[m.group(1)] = eval(re.sub(r"s\[(\d+)\]", r"s\1", m.group(2).replace("ULL", "")), {"P": P}, env)  # noqa: S307
            assert 0 <= env[m.group(1)] < 2**64
            continue
        m = fin.match(line)
        assert m, line
        out[int(m.group(1))] = env[m.group(2)]
    assert [out[i] for i in range(h2.WIDTH)] == list(exp)


def test_batch_merkle_framing_is_the_domain_framing() -> None:
    """``auth.leaf_hashes`` / ``_node_chunk`` / ``_pad_chunk`` (one hashed prefix per domain) hash the bytes
    ``CommitmentDomain.leaf`` / ``.node`` and the pad leaf do; ``leaf_bytes_many`` is ``row_leaf_value`` per row."""
    from verity.ml.commitments.merkle import _PAD, _hash, _uint
    from verity.ml.commitments.rowleaf import row_leaf_value

    from . import auth
    from .leaf.poseidon2 import Poseidon2Leaf
    rng = np.random.default_rng(9)
    for count in (1, 5, 300):
        dom = auth.domain_for(bytes(range(32)), 1, count)
        vals = [rng.integers(0, 256, int(rng.integers(0, 40)), dtype=np.uint8).tobytes() for _ in range(count)]
        for start in (0, 255, 70000):
            want = b"".join(dom.leaf(start + i, start + i, "sch", v) for i, v in enumerate(vals))
            assert auth.leaf_hashes(dom.domain_id, start, "sch", vals) == want
        pairs = rng.integers(0, 256, (count, 64), dtype=np.uint8)
        for lv, start in ((0, 0), (3, 255), (11, 1 << 20)):
            want = b"".join(dom.node(lv, start + i, pairs[i, :32].tobytes(), pairs[i, 32:].tobytes()) for i in range(count))
            assert auth._node_chunk((dom, lv, start, pairs)) == want
        want = b"".join(_hash(_PAD, dom.domain_id, _uint(r)) for r in range(250, 260))
        assert auth._pad_chunk((dom.domain_id, 250, 260)) == want
    d = rng.integers(0, P, (7, 8))
    assert Poseidon2Leaf().leaf_bytes_many(d) == [row_leaf_value(r) for r in d]
    with pytest.raises(ValueError):
        Poseidon2Leaf().leaf_bytes_many(np.full((1, 8), P))


def test_synthetic_rows_are_the_vu_words(tmp_path) -> None:
    """``instances``' ``x_rows`` / ``w_cols`` (what ``commit_vus`` commits) are the VUs' own ``a`` / ``b`` words, drawn
    and from the npz cache; ``_instance`` is unchanged."""
    from .relchain import _instance, _word_rows
    rel = relation("fp8-ada")
    for data in (instances(rel, 4), instances(rel, 4, cache=tmp_path), instances(rel, 4, cache=tmp_path)):
        x, w = _word_rows(data.x_rows), _word_rows(data.w_cols)
        assert x.shape == w.shape == (4, len(data[0][0]))
        assert x.tolist() == [list(v[0]) for v in data] and w.tolist() == [list(v[1]) for v in data]
    assert _instance((rel.name, rel.instance_seed, 2)) == tuple(data[2])


def test_frozen_rows_are_the_vu_words() -> None:
    """``frozen_instances``' ``x_rows`` / ``w_cols`` are the VUs' own ``a`` / ``b`` words, past the tier's end too
    (recycled ``i mod n``)."""
    from verity_numerical.bench import instances as frozen

    from .relchain import _word_rows, frozen_instances
    rel = relation("bf16-ampere")
    try:
        n_tier = frozen.load(rel.frozen_tier, frozen.DEFAULT_ROOT).n
    except FileNotFoundError:
        pytest.skip("frozen instances not built")
    data = frozen_instances(rel, n_tier + 2)
    x, w = _word_rows(data.x_rows), _word_rows(data.w_cols)
    assert x.shape == w.shape == (n_tier + 2, len(data[0][0]))
    for i in (0, 1, n_tier - 1, n_tier, n_tier + 1):
        assert x[i].tolist() == list(data[i][0]) and w[i].tolist() == list(data[i][1])
    small = frozen_instances(rel, 3)
    assert np.array_equal(small.x_rows, x[:3]) and np.array_equal(small.w_cols, w[:3])


def _fp4_rows(rng, n: int, steps: int) -> np.ndarray:
    from .fp4.hashed import FP4Format
    k = FP4Format().k
    codes = rng.integers(0, 16, (n * steps, k - 4))
    scales = rng.integers(0, 128, (n * steps, 4))
    return np.concatenate([codes, scales], 1).reshape(n, steps * k)


@pytest.mark.parametrize("fmt", ["w8", "w16", "fp4"])
def test_row_sponge_kernel_equals_chain_witness_and_native(fmt) -> None:
    """On CUDA: the committer's one-thread-per-row sponge gives the linked rows of :meth:`chain_witness` (reshaped per
    row) and the digests of :meth:`native` (the tree leaves), for plain 8- and 16-bit words and the fp4 lane format."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA")
    from .leaf.poseidon2 import Poseidon2Leaf, row_sponge_kernel
    if row_sponge_kernel() is None:
        pytest.skip("cupy")
    rng = np.random.default_rng(5)
    n, steps = 37, 5
    if fmt == "fp4":
        from .fp4.hashed import FP4Format
        leaf, word_bits = Poseidon2Leaf().with_lanes(FP4Format()), 4
        k = FP4Format().k
        rows = _fp4_rows(rng, n, steps)
    else:
        leaf, word_bits = Poseidon2Leaf(), int(fmt[1:])
        k = h2.RATE * (16 // word_bits)
        rows = rng.integers(0, 2**word_bits, (n, steps * k))
    for role in (h2.ROLE_X, h2.ROLE_W):
        words = torch.as_tensor(rows.reshape(n * steps, k), dtype=torch.int64, device="cuda")
        got = leaf.row_sponges(words, word_bits=word_bits, role=role, row_words=steps * k, n=n, steps=steps)
        assert got is not None
        cap, dg = leaf.chain_witness(words, word_bits=word_bits, role=role, row_words=steps * k, n_vus=n, steps=steps,
                                     l=n * steps)[:2]
        assert torch.equal(got[0], cap.T.reshape(n, steps, h2.CAPACITY)) and torch.equal(got[1], dg)
        assert np.array_equal(got[1].cpu().numpy(), leaf.native(rows, word_bits=word_bits, role=role))
    bad = torch.as_tensor(rows.reshape(n * steps, k), dtype=torch.int64, device="cuda")
    bad[3, 0] = -1
    with pytest.raises(ValueError):
        leaf.row_sponges(bad, word_bits=word_bits, role=h2.ROLE_X, row_words=steps * k, n=n, steps=steps)


def test_precomputed_row_sponges_equal_the_per_proof_sponge(runner) -> None:
    """``commit`` runs every committed row's sponge once; the gathered capacity rows / digests are what the per-proof
    sponge computes on the same words, and a prover whose words differ from the claimed rows falls back to them."""
    R, data = runner
    assert R.sponges is not None and set(R.sponges) == {"a", "b"}
    vus = [data[1], data[3]]
    auth = R.auth_for([1, 3])
    units = []
    for a, b, accs, _ in vus:
        from .relchain import vu_units
        units += vu_units(R.rel, a, b, accs)
    l = R.layout(2).l
    while len(units) < l:
        units.append(R._pad_unit())
    a = np.asarray([u[1] for u in units], dtype=np.int64)
    b = np.asarray([u[2] for u in units], dtype=np.int64)
    sp = R._sponges_for(auth, a, b, 2)
    assert sp is not None and set(sp) == {"a", "b"}
    slow = hashchain.hash_hints(R.hr, a, b, 2, l, "cpu")
    fast = hashchain.hash_hints(R.hr, a, b, 2, l, "cpu", sponges=sp)
    assert torch.equal(slow[0], fast[0]) and torch.equal(slow[1], fast[1])
    # the block assembly of the hint rows equals the per-name one
    assert R.hr.block_layout is True
    R.hr.block_layout = False
    try:
        by_name = hashchain.hash_hints(R.hr, a, b, 2, l, "cpu", sponges=sp)
    finally:
        R.hr.block_layout = None
    assert torch.equal(by_name[0], fast[0]) and torch.equal(by_name[1], fast[1])
    # claimed x rows are VU 0's and VU 2's: the words do not match -> no precomputed sponge for "a"
    other = R.auth_for([0, 2])
    sp2 = R._sponges_for(hashauth.HashAuth(auth.vu_index, other.x_index, auth.w_index, auth.trees), a, b, 2)
    assert sp2 is not None and set(sp2) == {"b"}


@pytest.mark.parametrize("name", sorted(CENSUS))
def test_private_decode_is_the_public_decode(name: str) -> None:
    """The converted decode pins from the device decode (no domain checks, no host sync) equal the relation's own public
    decode on finite operands, both as the G-side (compose_shared) and the hashed (compose) name lists."""
    rel = relation(name)
    rng = np.random.default_rng(7)
    l = 64
    a = torch.as_tensor(np.asarray([rel.operands(rng, (rel.k,)) for _ in range(l)], dtype=np.int64))
    b = torch.as_tensor(np.asarray([rel.operands(rng, (rel.k,)) for _ in range(l)], dtype=np.int64))
    vals = rel.public_vectors(rel.params, None, a, b, None, device="cpu")
    for hr in (hashchain.compose(rel), hashchain.compose_shared(rel)):
        n_dec = len([n for n in hr.extra_hint_names if n.split(".")[-1] in ("s", "m", "e") and "[" in n and n[0] in "ab"])
        names = hr.extra_hint_names[:n_dec]
        assert names and all(n.split(".")[-1] in ("s", "m", "e") for n in names)
        ref = torch.stack([torch.as_tensor(vals[n]).to(torch.int64) for n in names])
        assert torch.equal(hashchain.private_decode(hr, a, b, names), ref)
