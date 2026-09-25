"""The conformance suite EVERY leaf scheme must pass (campaign brief §3.1 D1): the static contract, native == chain
witness == gadget on random rows, the leaf framing, statement v5 with the ``+<leaf>`` suffix through the generic runner,
the committed-operand negatives, and the Rust fixture / pin helper.  Parametrized over the registry, so a new scheme
(``leaf/blake3.py``, ``leaf/ajtai.py``) is covered by importing its module below -- nothing else to write.

Run: ``pytest backends/direct/ligero/leaf/conformance_test.py -k <name>`` (CPU, seconds per scheme)."""
from __future__ import annotations

import json
import subprocess

import numpy as np
import os
import pytest

torch = pytest.importorskip("torch")

from .. import hashauth, hashchain, protocol  # noqa: E402
from ..relations import relation  # noqa: E402
from ..relchain import HashedRelationRunner, hashed_negatives, instances  # noqa: E402
from ..serialize import _statement_of, proof_bytes, read_statement, statement_bytes, verify_files  # noqa: E402
from . import base, registry  # noqa: E402
from . import dummy  # noqa: E402, F401  (test-only scheme: registers itself)
from .fixtures import ligero_verify_binary, write_fixture  # noqa: E402

#: k = 32 words of 8 bits, 48 columns per VU: the relation every scheme is exercised on (VERITY_LEAF_CONFORMANCE_REL=bf16-hopper
#: for the 96-column relation)
REL = os.environ.get("VERITY_LEAF_CONFORMANCE_REL", "fp8-ada")
#: every registered scheme, plus family members named in VERITY_LEAF_CONFORMANCE (comma-separated, e.g. "ajtai-n64,ajtai-n128")
SCHEMES = registry.names() + [n for n in os.environ.get("VERITY_LEAF_CONFORMANCE", "").split(",") if n]


class _Runners(dict):
    """Built on first use, so ``-k <scheme>`` on a relation another scheme cannot compose (Poseidon2 on an x4 fold) still runs."""

    def __init__(self) -> None:
        super().__init__()
        self.rel = relation(REL)
        self.data = instances(self.rel, 4)

    def __missing__(self, name: str) -> tuple:
        R = HashedRelationRunner(self.rel, "cpu", -128.0, zk=False, mode="fiat-shamir", leaf=name)
        R.commit_vus(self.data)
        self[name] = (R, self.data)
        return self[name]


@pytest.fixture(scope="module")
def runners() -> dict[str, tuple]:
    """One committed CPU runner per scheme (4 synthetic VUs)."""
    return _Runners()


# ---------------------------------------------------------------------------------------------------------------------
# 1. the static contract
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", SCHEMES)
def test_static_contract(name: str) -> None:
    s = registry.get(name)
    base.check_scheme(s)
    assert isinstance(s, base.LeafScheme)
    assert registry.suffix_of(s) == name == registry.canonical(name) and "+" not in name and name not in registry.ALIASES
    assert name.startswith(s.name)                                            # a family member "ajtai-n64" has name "ajtai"
    assert 1 <= s.digest_elems and s.digest_elems + s.carry_elems <= 1024 - 3  # Rust MAX_LINKS bounds the linked rows
    assert len(s.params_sha256) == 64 and int(s.params_sha256, 16) >= 0
    assert s.schema and s.assumption and s.hash_name
    if name in registry.names():
        assert registry.by_schema(s.schema).schema == s.schema
    assert registry.get(name) is not registry.get(name)                        # fresh instances (gadgets hold state)
    others = [registry.get(n) for n in SCHEMES if n != name]
    # a schema names the committed leaf, the scheme name the circuit proving it: two circuits of one leaf (blake3 /
    # blake3-xob) must commit the same object
    assert all((o.schema == s.schema) == (o.params_sha256 == s.params_sha256) for o in others), "schema / params must identify the leaf"
    for o in others:
        if o.schema == s.schema:
            assert (o.digest_elems, o.carry_elems, o.hash_name) == (s.digest_elems, s.carry_elems, s.hash_name)
            assert type(o).leaf_bytes is type(s).leaf_bytes and type(o).native is type(s).native


# ---------------------------------------------------------------------------------------------------------------------
# 2. native: shape, range, determinism, role separation
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", SCHEMES)
def test_native_shape_range_determinism_and_roles(name: str) -> None:
    s = registry.get(name)
    rel = relation(REL)
    K, nb = rel.steps * rel.k, 8
    rows = np.random.default_rng(1).integers(0, 2**nb, size=(6, K), dtype=np.int64)
    dx = s.native(rows, word_bits=nb, role=hashchain.ROLES[0][2])
    dw = s.native(rows, word_bits=nb, role=hashchain.ROLES[1][2])
    for d in (dx, dw):
        assert d.shape == (6, s.digest_elems) and d.dtype == np.int64 and (0 <= d).all() and (d < protocol.P).all()
    assert np.array_equal(dx, s.native(rows, word_bits=nb, role=hashchain.ROLES[0][2]))
    assert not np.array_equal(dx, dw), "the x and W roles must hash differently (domain separation)"
    assert len({tuple(r) for r in dx}) == 6, "distinct rows, distinct digests"
    rows2 = rows.copy()
    rows2[2, K // 2] ^= 1
    dx2 = s.native(rows2, word_bits=nb, role=hashchain.ROLES[0][2])
    assert not np.array_equal(dx2[2], dx[2]) and np.array_equal(dx2[3], dx[3])


# ---------------------------------------------------------------------------------------------------------------------
# 3. framing
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", SCHEMES)
def test_leaf_bytes_framing(name: str) -> None:
    s = registry.get(name)
    rng = np.random.default_rng(2)
    ds = rng.integers(0, protocol.P, size=(50, s.digest_elems), dtype=np.int64)
    bs = [s.leaf_bytes(d) for d in ds]
    assert all(isinstance(b, bytes) and len(b) == len(bs[0]) > 0 for b in bs)
    assert len(set(bs)) == 50, "leaf_bytes must be injective on digests"
    assert s.leaf_bytes(ds[0]) == s.leaf_bytes(list(int(x) for x in ds[0])) == s.leaf_bytes(torch.as_tensor(ds[0]))
    with pytest.raises((ValueError, OverflowError)):
        s.leaf_bytes(np.array([protocol.P] + [0] * (s.digest_elems - 1)))
    with pytest.raises((ValueError, TypeError)):
        s.leaf_bytes(ds[0][:-1] if s.digest_elems > 1 else np.zeros(2, dtype=np.int64))


# ---------------------------------------------------------------------------------------------------------------------
# 4. native == chain witness (the prover's per-column states and digests)
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", SCHEMES)
def test_chain_witness_matches_native_and_starts_at_zero(name: str) -> None:
    rel = relation(REL)
    hr = hashchain.compose(rel, name)
    s = hr.leaf
    n, steps, k, nb = 3, rel.steps, rel.k, hr.word_bits
    l = n * steps + 5                                                 # 5 pad columns
    rows = np.random.default_rng(3).integers(0, 2**nb, size=(n, steps * k), dtype=np.int64)
    rows[(rows & 0x7F) == 0x7F] = 0                                   # no E4M3 NaN: the relation's decode (hash_hints) refuses it
    words = torch.zeros((l, k), dtype=torch.int64)
    words[:n * steps] = torch.as_tensor(rows.reshape(n * steps, k))
    for _, nm, role in hashchain.ROLES:
        out = s.chain_witness(words, word_bits=nb, role=role, row_words=steps * k, n_vus=n, steps=steps, l=l)
        cap, dg = out[0], out[1]
        assert tuple(cap.shape) == (s.carry_elems, l) and tuple(dg.shape) == (n, s.digest_elems)
        assert np.array_equal(dg.cpu().numpy(), s.native(rows, word_bits=nb, role=role)), f"{nm}: chain digest != native"
        starts = torch.arange(n) * steps
        assert not cap[:, starts].any(), "the carried state at a chain start must be 0"
        assert not cap[:, n * steps:].any(), "pad columns carry 0"
        assert ((cap >= 0) & (cap < protocol.P)).all()
        extras = out[2] if len(out) > 2 else {}
        for k_, v in extras.items():
            assert k_ in hr.extra_hint_names and tuple(torch.as_tensor(v).shape) == (l,)
        # row_chain (the committer's precomputation) is the same computation reshaped
        rc = hashchain.row_chain(hr, rows, role, "cpu")
        assert np.array_equal(rc[0].reshape(n * steps, s.carry_elems).T.numpy(), cap[:, :n * steps].numpy())
        assert np.array_equal(rc[1].numpy(), dg.numpy())
    # every extra hint row the gadget declared has a witness value (hash_hints raises otherwise)
    a = words.clone()
    b = torch.zeros_like(words)
    b[:n * steps] = torch.as_tensor(rows.reshape(n * steps, k)[::-1].copy())
    hints, digests = hashchain.hash_hints(hr, a, b, n, l, "cpu")
    assert tuple(hints.shape) == (len(hr.extra_hint_names), l) and tuple(digests.shape) == (n, 2 * s.digest_elems)


# ---------------------------------------------------------------------------------------------------------------------
# 5. native == gadget: the compiled hashed relation proves and verifies, publishing native's digests
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", SCHEMES)
def test_compose_prove_verify_publishes_the_native_digests(name: str, runners) -> None:
    R, data = runners[name]
    s = R.leaf
    assert R.relation == registry.relation_name(REL, name) and R.hr.census["leaf"] == name
    assert R.hr.census["linked_rows"] == 3 + 2 * s.carry_elems
    assert [n for _, n in R.sys.pins] == [hashchain.HASH_PIN_END] + [hashchain.hash_pin(x) for x in range(2 * s.digest_elems)]
    vus, ids = [data[1], data[3]], [1, 3]
    proof, pubs, lay = R.prove_vus(vus, check=True, vu_ids=ids)       # check=True: the prover evaluates every constraint
    ok, why = R.verify_vus(proof, pubs, lay)
    assert ok, why
    table = np.asarray(pubs[0])
    ends = np.arange(2) * lay.steps + lay.steps - 1
    published = table[ends, 1:]
    x_rows = np.asarray([v[0] for v in vus], dtype=np.int64).reshape(2, -1)
    w_cols = np.asarray([v[1] for v in vus], dtype=np.int64).reshape(2, -1)
    want = np.concatenate([s.native(x_rows, word_bits=R.hr.word_bits, role=hashchain.ROLES[0][2]),
                           s.native(w_cols, word_bits=R.hr.word_bits, role=hashchain.ROLES[1][2])], 1)
    assert np.array_equal(published, want), "the digests the gadget pins are not native()'s"
    assert np.array_equal(R.digests["a"][ids], want[:, :s.digest_elems]) and np.array_equal(R.digests["b"][ids], want[:, s.digest_elems:])
    # the committed trees open at those digests under the scheme's framing
    auth = pubs[3]
    ok, why = hashauth.verify_hash_auth(auth, pubs[4], digests=published, y_words=np.asarray(pubs[2])[ends],
                                        y_bits=R.rel.y_bits, leaf=s)
    assert ok, why
    bad = published.copy()
    bad[1, s.digest_elems] ^= 1
    ok, why = hashauth.verify_hash_auth(auth, pubs[4], digests=bad, y_words=np.asarray(pubs[2])[ends], y_bits=R.rel.y_bits, leaf=s)
    assert not ok and why.startswith("auth: b:")


@pytest.mark.parametrize("name", SCHEMES)
def test_pipelined_prover_publishes_the_sequential_statement(name: str, runners) -> None:
    """``prove_vus_many`` (bench-vu-rel --pipeline N): the statement is fixed from the committer's digests before each
    prover starts; it must be the one ``prove_vus`` publishes, and every sub-batch must verify."""
    R, data = runners[name]
    ids = [[0, 2], [1, 3]]
    many, wall = R.prove_vus_many([[data[i] for i in g] for g in ids], lambda i: None, depth=2, vu_ids_list=ids)
    assert len(many) == 2 and wall > 0
    for (V, proof, pubs, lay), g in zip(many, ids):
        assert R.verify_vus(proof, pubs, lay, coins=V) == (True, "")
        _, ref, _ = R.prove_vus([data[i] for i in g], vu_ids=g)
        assert np.array_equal(pubs[0], ref[0]) and np.array_equal(pubs[1], ref[1]) and pubs[3] == ref[3] and pubs[4] == ref[4]
        assert proof.timings["auth_openings"] >= 0 and proof.timings["total"] + proof.timings.get("hints_host", 0.0) > 0


# sha256 is 47 full proofs: ~15 min on 4 idle vCPUs (art:020560e1), past a runner-wide 1800 s on a loaded host
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("name", SCHEMES)
def test_committed_operand_negatives_rejected(name: str, runners) -> None:
    R, data = runners[name]
    negs = hashed_negatives(R, [data[0], data[2]])
    bad = [(what, why) for what, (ok, why) in negs if ok]
    assert not bad, f"{name}: accepted negatives {bad}"
    assert len(negs) >= 8


# ---------------------------------------------------------------------------------------------------------------------
# 6. statement v5 with the scheme suffix, through the generic runner resolution
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", SCHEMES)
def test_statement_v5_carries_the_scheme_and_verifies_from_files(name: str, runners, tmp_path) -> None:
    R, data = runners[name]
    proof, pubs, lay = R.prove_vus([data[1], data[3]], check=False, vu_ids=[1, 3])
    st = _statement_of(R, pubs, lay, R.cfg(lay.l), 1)
    raw = statement_bytes(st)
    st2 = read_statement(raw)
    assert st2.relation == registry.relation_name(REL, name) and st2.base_relation == REL and st2.leaf == name
    assert st2.leaf_scheme().name == name and statement_bytes(st2) == raw
    assert st2.digests().shape == (2, 2 * R.leaf.digest_elems)
    (tmp_path / "s.stmt").write_bytes(raw)
    (tmp_path / "s.proof").write_bytes(proof_bytes(proof))
    ok, why, _ = verify_files(tmp_path / "s.stmt", tmp_path / "s.proof", "cpu")      # the runner is resolved from the file
    assert ok, why
    # a statement rewritten with the suffix of another circuit of the same leaf parses, but not against that circuit
    for twin in (n for n in SCHEMES if n != name and registry.get(n).schema == R.leaf.schema):
        t_rel = registry.relation_name(REL, twin)
        i = raw.index(st2.relation.encode())
        forged = raw[:i - 4] + len(t_rel).to_bytes(4, "little") + t_rel.encode() + raw[i + len(st2.relation):]
        (tmp_path / "t.stmt").write_bytes(forged)
        (tmp_path / "t.proof").write_bytes(proof_bytes(proof))
        try:
            ok, _why, _ = verify_files(tmp_path / "t.stmt", tmp_path / "t.proof", "cpu")
        except ValueError:
            ok = False
        assert not ok, f"{name} proof accepted as {twin}"
    # the block must name the scheme's schema: a statement rewritten with another scheme's suffix is refused at parse
    other = next((n for n in SCHEMES if n != name and registry.get(n).schema != R.leaf.schema), None)
    if other is not None:
        o_rel = registry.relation_name(REL, other)
        i = raw.index(st2.relation.encode()) if st2.relation.encode() in raw else -1
        assert i > 0
        forged = raw[:i - 4] + len(o_rel).to_bytes(4, "little") + o_rel.encode() + raw[i + len(st2.relation):]
        with pytest.raises(ValueError):
            read_statement(forged)


# ---------------------------------------------------------------------------------------------------------------------
# 7. the Rust fixture / pin helper
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", SCHEMES)
def test_fixture_helper_writes_a_rust_verifiable_set(name: str, tmp_path) -> None:
    man = write_fixture(tmp_path, REL, name, n_vus=4, vus=(0, 2), mode="interactive")
    for f in man["files"]:
        assert (tmp_path / f).stat().st_size > 0
    assert man["relation"]["leaf"] == name and man["relation"]["statement_relation"] == registry.relation_name(REL, name)
    st = read_statement((tmp_path / "sub_00.stmt").read_bytes())
    assert st.leaf == name and st.n_vus == 2
    binary = ligero_verify_binary()
    if binary is None:
        pytest.skip("ligero-verify not built")
    assert man["rust"]["sys_id"] == man["system"]["sys_id"] and "pins_row" in man
    r = subprocess.run([str(binary), "verify", "--system", str(tmp_path / "system.bin"), "--statement", str(tmp_path / "sub_00.stmt"),
                        "--proof", str(tmp_path / "sub_00.proof"), "--coins", str(tmp_path / "sub_00.coins"), "--allow-any-system"],
                       capture_output=True, text=True)
    js = json.loads(r.stdout.strip().splitlines()[-1])
    if getattr(registry.get(name), "test_only", False):
        # the Rust verifier does not know a test-only scheme: the statement is refused for naming it
        assert r.returncode == 1 and not js["accepted"] and "leaf scheme" in js["reason"], js
    else:
        assert r.returncode == 0 and js["accepted"], js
        assert man["rust"]["pinned_relation"] is not None or name != registry.DEFAULT
