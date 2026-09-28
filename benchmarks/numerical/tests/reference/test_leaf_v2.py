"""The ``leaf/v2`` reference (``verity_numerical.reference.leaf_v2``, spec ``verity/commitments/LEAF_V2.md``) over real
captured rows of ``fixtures/bench-instances/v1/seeds``: an activation tensor (tokens x 1536), a weight tensor (rows x
1536) and an output row, committed under the unchanged ids tree; one sampled VU's ``x`` row / ``w`` row / ``y`` word
bound to the proof tableau by the cross-tableau IRS + linear tests; the negatives that must be caught.  Reduced code
parameters (n = 4096, k = 1024, l = 768) keep the pure-Python suite under 20 s; nothing here is a performance number."""

from __future__ import annotations

import json
import random
from array import array
from pathlib import Path

import pytest
from verity.ml.commitments.leaves import LeafOpening, verify_opening
from verity_numerical.reference import leaf_v2 as v2

REPO = Path(__file__).resolve().parents[4]
SEEDS = REPO / "fixtures" / "bench-instances" / "v1" / "seeds"
CODE = v2.Code(n=4096, k=1024, l=768, R=8)
T_OPEN = 48                       # <= k - l = 256 randomizers: hiding
TOKENS = 24                       # all captured activation rows; every row is 2 tableau rows


def _u16(path: Path) -> list[int]:
    a = array("H")
    a.frombytes(path.read_bytes())
    return list(a)


def _seed_rows() -> tuple[list[str], list[list[int]], list[list[int]], list[int]]:
    doc = json.loads((SEEDS / "seeds.json").read_text())
    names = list(doc["seeds"])[:TOKENS]
    xs = [_u16(SEEDS / n / "x.u16") for n in names]
    ws = [_u16(SEEDS / n / "w_row.u16") for n in names]
    ys = [int(doc["seeds"][n]["out_row_word_j"]) for n in names]
    return names, xs, ws, ys


@pytest.fixture(scope="module")
def world():
    names, xs, ws, ys = _seed_rows()
    K = len(xs[0])
    assert K == 1536 and all(len(w) == K for w in ws)
    x_id = "p=b3eba618b1b5/r=ref/t=0/ph=decode/l=5/op=layers.5.input_layernorm.rmsnorm/port=0"
    w_id = "w=layers.5.self_attn.qkv_proj.weight/rows=ref"
    y_id = "p=b3eba618b1b5/r=ref/t=0/ph=decode/l=5/op=layers.5.self_attn.qkv_proj.gemm/port=0"
    x_t = v2.commit_tensor(x_id, "bfloat16", (TOKENS, K), [w for row in xs for w in row], CODE, seed=b"x" * 32)
    w_t = v2.commit_tensor(w_id, "bfloat16", (TOKENS, K), [w for row in ws for w in row], CODE, seed=b"w" * 32)
    y_t = v2.commit_tensor(y_id, "bfloat16", (1, TOKENS), ys, CODE, seed=b"y" * 32)
    tree = v2.tensor_tree([x_t, w_t, y_t], "ru/ref")
    return {"names": names, "xs": xs, "ws": ws, "ys": ys, "K": K, "x": x_t, "w": w_t, "y": y_t, "tree": tree}


def _claims_for_vu(world, token: int, wrow: int, x_rows=None, w_rows=None) -> tuple[list[v2.Claim], list[list[int]]]:
    """The sampled VU (activation row ``token``, weight row ``wrow``, output word ``token``): witness rows for the
    proof tableau and the equality claims that bind them."""
    code = CODE
    K = world["K"]
    x_row = world["xs"][token] if x_rows is None else x_rows
    w_row = world["ws"][wrow] if w_rows is None else w_rows
    y_word = world["ys"][token]
    per = code.rows_per_tensor_row(K)
    witness = []
    claims = []
    for role, tensor, trow, row in (("x", world["x"], token, x_row), ("w", world["w"], wrow, w_row)):
        for s in range(per):
            piece = row[s * code.l:(s + 1) * code.l]
            claims.append(v2.Claim(tensor.canonical_id, trow * per + s, tuple(range(len(piece))), len(witness), tuple(range(len(piece)))))
            witness.append(list(piece))
    pos = world["y"].position(token)                    # word `token` of the [1, TOKENS] output row
    claims.append(v2.Claim(world["y"].canonical_id, pos.tableau_row, (pos.payload_index,), len(witness), (0,)))
    witness.append([y_word])
    return claims, witness


def _run(world, claims, witness, rng_seed=7, tamper=None):
    rng = random.Random(rng_seed)
    proof = v2.ProofTableau.commit(CODE, witness, rng)
    session = v2.verifier_coins(CODE, claims, T_OPEN, rng)
    tensors = {t.canonical_id: t for t in (world["x"], world["w"], world["y"])}
    v2.prove(session, tensors, proof)
    if tamper:
        tamper(session)
    tree = world["tree"]
    descriptors = {t.canonical_id: t.descriptor() for t in tensors.values()}
    openings = {tid: tree.path(tid) for tid in tensors}
    return v2.verify(session, descriptors, tree, openings, proof.root, len(proof.codewords))


def test_layout_and_digest_structure(world):
    x = world["x"]
    assert CODE.rows_per_tensor_row(1536) == 2 and len(x.payload_rows) == 2 * TOKENS
    assert len(x.chunks) == -(-2 * TOKENS // CODE.R)
    assert all(len(cw) == CODE.n for c in x.chunks for cw in c.codewords)
    # non-systematic: the message is recovered only by decoding; no codeword symbol equals a payload symbol
    assert v2.decode_row(CODE, x.chunks[0].codewords[0])[:CODE.l] == x.payload_rows[0]
    # word addressing is the leaf/v1 row-major index
    pos = x.position(1536 * 3 + 1000)
    assert (pos.tableau_row, pos.payload_index) == (3 * 2 + 1, 1000 - 768)
    assert x.words[1536 * 3 + 1000] == world["xs"][3][1000] == x.payload_rows[pos.tableau_row][pos.payload_index]
    # the digest binds id, dtype, shape, code and every chunk root; the ids tree is leaf/v1's
    d = x.descriptor()
    assert bytes.fromhex(d["digest"]) == v2.leaf_v2_digest(x.canonical_id, "bfloat16", x.shape, CODE, x.chunk_roots)
    assert v2.leaf_v2_digest(x.canonical_id, "float16", x.shape, CODE, x.chunk_roots) != x.digest
    assert v2.leaf_v2_digest(x.canonical_id, "bfloat16", (TOKENS * 2, 768), CODE, x.chunk_roots) != x.digest
    tree = world["tree"]
    assert tree.leaf_count == 3 and tree.index_of(x.canonical_id) == sorted(tree.ids).index(x.canonical_id)
    # a v1 LeafOpening-style path check works on the digest level unchanged
    op = LeafOpening(tree.tree_id, tree.root, 3, tree.index_of(x.canonical_id), x.canonical_id, "bfloat16", x.shape, b"", tree.path(x.canonical_id))
    assert verify_opening(tree.root, 3, op) is False            # v1 would hash the (empty) bytes: v2 digests are not v1 leaves
    from verity.ml.commitments.leaves import fold_path
    assert fold_path(x.digest, op.index, op.path) == tree.root


def test_randomizers_make_the_root_non_canonical(world):
    """Same bytes, different seed -> different chunk roots and digest (hiding); the payload is unchanged."""
    x = world["x"]
    again = v2.commit_tensor(x.canonical_id, "bfloat16", x.shape, x.words, CODE, seed=b"other" + b"\0" * 27)
    assert again.payload_rows == x.payload_rows and again.digest != x.digest
    same = v2.commit_tensor(x.canonical_id, "bfloat16", x.shape, x.words, CODE, seed=b"x" * 32)
    assert same.digest == x.digest


def test_ntt_roundtrip_and_coset_encoding():
    rng = random.Random(1)
    coeffs = [rng.randrange(v2.BABYBEAR) for _ in range(64)]
    assert v2.intt(v2.ntt(coeffs)) == coeffs
    assert v2.coset_intt(v2.coset_ntt(coeffs)) == coeffs
    code = v2.Code(n=64, k=16, l=8, R=2)
    cw = v2.encode_row(code, list(range(8)), [5] * 8)
    # a codeword has degree < k on the coset, and decodes to the message
    assert all(c == 0 for c in v2.coset_intt(cw)[code.k:])
    assert v2.decode_row(code, cw) == list(range(8)) + [5] * 8
    # the codeword domain (31 H_n) is disjoint from the message domain (H_k): no systematic position exists
    assert not set(code.domain()) & set(code.message_domain())
    # evaluating the interpolant directly at every coset point reproduces the codeword
    P = v2.BABYBEAR
    poly = v2.intt(list(range(8)) + [5] * 8)
    for x, y in zip(code.domain(), cw, strict=True):
        assert sum(c * pow(x, i, P) for i, c in enumerate(poly)) % P == y
    # and on the subgroup itself the polynomial IS systematic -- the domain choice is what removes the leak
    assert v2.ntt(poly)[:16] == list(range(8)) + [5] * 8


def test_no_opened_column_is_a_payload_word(world):
    """Red-team F3 (redteam_vu.md): with the systematic encoding an opened column ``(n/k) j`` was payload word ``j`` in
    the clear.  Open EVERY column of every chunk of the activation tensor: no column value equals the payload word of
    its row at any position that the old systematic map would have exposed, and, stronger, no column is a payload
    word at all beyond chance (a 16-bit word among 31-bit field elements: expected ``n * 2^16 / p ~ 0.13`` hits/row)."""
    x = world["x"]
    stride = CODE.n // CODE.k
    systematic_hits = 0
    chance_hits = 0
    for chunk in x.chunks:
        for r, cw in enumerate(chunk.codewords):
            payload = x.payload_rows[chunk.index * CODE.R + r]
            for j in range(CODE.l):
                if cw[stride * j] == payload[j]:
                    systematic_hits += 1
            pl = set(payload)
            chance_hits += sum(1 for v in cw if v in pl)
    rows = len(x.payload_rows)
    assert systematic_hits <= rows * CODE.l * 2 ** -12, systematic_hits      # was == rows * l before the fix
    # coincidences only: a column value equal to SOME word of its row happens ~ n * |payload| / p times per row
    assert chance_hits < rows * CODE.n * CODE.l / v2.BABYBEAR * 8 + 8 * rows, chance_hits
    # the old encoding (on the subgroup) would have exposed every payload word: the check is not vacuous
    old = v2.ntt(v2.intt(x.payload_rows[0] + x.randomizer_rows[0]) + [0] * (CODE.n - CODE.k))
    assert [old[stride * j] for j in range(CODE.l)] == x.payload_rows[0]


def test_honest_vu_verifies(world):
    claims, witness = _claims_for_vu(world, token=2, wrow=4)
    assert _run(world, claims, witness) == v2.Verdict(True)


def test_transparent_word_opening(world):
    """The non-ZK conformance path: disclose the row + randomizers, re-encode, walk every column to the chunk root."""
    x = world["x"]
    idx = 1536 * 4 + 77
    pos = x.position(idx)
    chunk = x.chunks[pos.chunk]
    paths = {c: v2.open_path(chunk.levels, c) for c in range(CODE.n)}
    got = v2.verify_transparent_word(x.descriptor(), CODE, idx, x.payload_rows[pos.tableau_row], x.randomizer_rows[pos.tableau_row],
                                     chunk.root, paths, chunk.codewords)
    assert got == world["xs"][4][77]
    bad = list(x.payload_rows[pos.tableau_row])
    bad[pos.payload_index] ^= 1
    assert v2.verify_transparent_word(x.descriptor(), CODE, idx, bad, x.randomizer_rows[pos.tableau_row], chunk.root, paths, chunk.codewords) is None


def test_negative_swapped_weight_row(world):
    """The prover holds W row 5 but claims row 4 (the sampled coordinate): caught by the linear test."""
    claims, witness = _claims_for_vu(world, token=2, wrow=4, w_rows=world["ws"][5])
    verdict = _run(world, claims, witness)
    assert not verdict.ok and "equality constraints not satisfied" in verdict.reason


def test_negative_one_flipped_word(world):
    x_bad = list(world["xs"][1])
    x_bad[1535] ^= 0x0001
    claims, witness = _claims_for_vu(world, token=1, wrow=1, x_rows=x_bad)
    assert not _run(world, claims, witness).ok


def test_negative_wrong_output_word(world):
    claims, witness = _claims_for_vu(world, token=0, wrow=0)
    witness[-1] = [world["ys"][0] ^ 0x0100]
    assert not _run(world, claims, witness).ok


def test_negative_tampered_column_and_wrong_tensor(world):
    claims, witness = _claims_for_vu(world, token=3, wrow=3)

    def tamper_column(session):
        tid = world["w"].canonical_id
        ci = next(iter(session.tensor_openings[tid]))
        o = session.tensor_openings[tid][ci][0]
        vals = list(o.values)
        vals[0] = (vals[0] + 1) % v2.BABYBEAR
        session.tensor_openings[tid][ci][0] = v2.ColumnOpening(o.chunk, o.column, tuple(vals), o.path)

    verdict = _run(world, claims, witness, tamper=tamper_column)
    assert not verdict.ok and "does not fold to the chunk root" in verdict.reason

    # cross-tensor confusion: the claim names the activation tensor's id for the weight row
    confused = [v2.Claim(world["x"].canonical_id, c.tableau_row, c.tensor_positions, c.proof_row, c.proof_positions)
                if c.tensor == world["w"].canonical_id else c for c in claims]
    assert not _run(world, confused, witness).ok


def test_negative_non_codeword_tensor_row_is_caught_by_irs(world):
    """A dishonest committer puts a far-from-code row into the tensor tableau: the IRS test rejects it even though
    the column openings are consistent with the chunk root."""
    import copy

    w = copy.deepcopy(world["w"])
    chunk = w.chunks[0]
    rng = random.Random(3)
    chunk.codewords[1] = [rng.randrange(v2.BABYBEAR) for _ in range(CODE.n)]
    chunk.build()
    bad_world = dict(world)
    bad_world["w"] = w
    bad_world["tree"] = v2.tensor_tree([world["x"], w, world["y"]], "ru/ref")
    claims, witness = _claims_for_vu(bad_world, token=0, wrow=0)
    # the witness row 1 (second half of W row 0) is honest data; the tensor side is garbage
    verdict = _run(bad_world, claims, witness)
    assert not verdict.ok and ("IRS" in verdict.reason or "equality" in verdict.reason)


def test_hiding_bound_enforced():
    rng = random.Random(0)
    with pytest.raises(Exception, match="not hiding"):
        v2.verifier_coins(CODE, [], CODE.randomizers + 1, rng)
