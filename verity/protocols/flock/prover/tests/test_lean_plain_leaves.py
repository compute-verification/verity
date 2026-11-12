"""A recursive session's inner statement has plain SHA-512 leaves (`verity_flock.circuit.LEAF_SCHEME_PLAIN`, staged under
`FLOCK_PLAIN_LEAVES=1`). The statement digest binds the prover's identity, which names the leaf scheme and the Merkle hash,
so the Lean verifier's `verity/flock-circuit+plain-leaves` (`Tags.withPlainLeaves`) writes both as upstream's `identity`
does, pins META `leaf_scheme` to the staged object, and decodes each opening's empty `opened_salts`. Upstream's `--zk`
refuses such a statement, and so does Lean's `verify --zk`; without `--zk` the hidden-output headlines cover plain leaves
(`ProvedScope.leafOk`), and the recursive verifier checks the inner statement's sessions."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from verity_flock import circuit as FC

ROOT = Path(__file__).resolve().parents[5]
LEAN = ROOT / "verity/protocols/flock/verifier"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
STATEMENT = "verity/flock-circuit+plain-leaves"


def _lean_def(name: str) -> str:
    src = (LEAN / "Flock" / "Tags.lean").read_text()
    body = src[src.index(f"def {name}"):]
    return body[:body.index("\n\n")]


def _lean_scheme() -> dict:
    body = _lean_def("leafSchemePlain")
    got: dict = dict(re.findall(r'\("(\w+)", s "([^"]*)"\)', body))
    got.update({k: int(v) for k, v in re.findall(r'\("(\w+)", n (\d+)\)', body)})
    return got


def _rust_scheme() -> dict:
    src = (ROOT / "verity/protocols/flock/prover/live/src/circuit.rs").read_text()
    body = src[src.index("pub fn leaf_scheme_plain() -> Value {"):]
    body = body[:body.index("\n}")]
    got: dict = dict(re.findall(r'"(\w+)": "([^"]*)"', body))
    got.update({k: int(v) for k, v in re.findall(r'"(\w+)": (\d+)', body)})
    return got


def _rust_merkle_text() -> str:
    """`identity`'s `hashes.merkle` on a plain-leaf statement: `merkle(c)` is `HashKind::Sha512` there, named as the SHA-512
    patch's `as_str` names it."""
    src = (ROOT / "verity/protocols/flock/prover/live/src/bin/flock-circuit.rs").read_text()
    assert "if c.plain_leaves() { HashKind::Sha512 } else { MERKLE }" in src
    fmt = re.search(r'"merkle": format!\("\{\}([^"]*)", merkle\(c\)\.as_str\(\)\)', src).group(1)
    patch = (ROOT / "verity/protocols/flock/prover/flock-sha512-b684b12.patch").read_text()
    name = re.search(r'HashKind::Sha512 => "([^"]*)",', patch).group(1)
    return name + fmt


def test_the_plain_leaf_scheme_is_the_staged_one():
    assert _lean_scheme() == FC.LEAF_SCHEME_PLAIN == _rust_scheme()


def test_the_plain_leaf_identity_is_the_provers():
    body = _lean_def("withPlainLeaves")
    assert re.findall(r'setObjVal! "merkle" \(s "([^"]*)"\)', body) == [_rust_merkle_text()] == ["sha512 (every Ligerito level)"]
    rust = (ROOT / "verity/protocols/flock/prover/live/src/bin/flock-circuit.rs").read_text()
    assert '"leaf_scheme": c.leaf_scheme,' in rust
    assert 'setObjVal! "leaf_scheme" leafSchemePlain' in body
    assert 'merkleLeaf := "flock-leaf/sha512-unsalted", pinnedLeafScheme := some leafSchemePlain, identity }' in body
    assert "t.pinnedLeafScheme.map Firewall.canonForm == some (Firewall.canonForm leafSchemePlain)" in _lean_def("plainLeaves")


def test_the_pinned_leaf_scheme_is_a_constant():
    """`plainLeaves` compares by `canonForm`, which is coarser than `Json`'s `==` on numbers (`0.0` prints as `0`), and the
    two agree on the constants `pinnedLeafScheme` takes. A scheme read from input would widen the plain-leaves refusal."""
    sets = [m.strip() for p in LEAN.rglob("*.lean") if ".lake" not in p.parts
            for m in re.findall(r"pinnedLeafScheme := ([^,}\n]+)", p.read_text())]
    assert sets and all(re.fullmatch(r"some leafScheme\w+", m) for m in sets), sets


def test_a_plain_leaf_opening_carries_an_empty_salt_vector():
    """The pinned patch serializes `opened_salts` in every opening, "empty for unsalted leaves", so Lean decodes an unsalted
    opening of an hm96-row statement (the inner statement's) with an empty vector, not `631567f7`'s absent one."""
    patch = (ROOT / "verity/protocols/flock/prover/flock-sha512-b684b12.patch").read_text()
    assert "(empty for unsalted leaves).\n+    pub opened_salts: Vec<flock_merkle::hm96::Salt>," in patch
    verify = (LEAN / "Flock" / "Verify.lean").read_text()
    assert "openedSalts := st.tags.hm96Rows" in verify
    assert "saltLen : Option Nat := if st.merkle.saltBytes = 0 && !st.openedSalts then none else some st.merkle.saltBytes" in verify


def test_the_plain_leaf_statement_is_the_default_one_by_name():
    assert f'{{ circuit.withPlainLeaves with name := "{STATEMENT}" }}' in _lean_def("circuitPlainLeaves")
    assert "circuitPlainLeaves]" in _lean_def("all")


def _verify(*args: str) -> subprocess.CompletedProcess:
    if shutil.which("lake") is None:
        pytest.skip("no Lean toolchain")
    subprocess.run(["lake", "build", "flock-verify"], cwd=LEAN, check=True, capture_output=True)
    return subprocess.run([str(BIN), "verify", "--statement", STATEMENT, *args], capture_output=True, text=True, check=False)


@pytest.mark.slow
def test_verify_refuses_the_plain_leaf_statement_under_zk(tmp_path):
    """`--zk`'s hiding needs `hm96-sha512/v1` leaves, so `verify --zk` refuses the statement before it reads an input."""
    p = _verify("--zk", "--circuit", str(tmp_path / "c"), "--public", str(tmp_path / "p"), "--session", str(tmp_path / "s"))
    assert p.returncode == 2, p.stdout + p.stderr
    assert (f"verify --zk: refused: {STATEMENT}, a statement with plain SHA-512 leaves (flock-leaf/sha512-unsalted): "
            "--zk's hiding needs hm96-sha512/v1 leaves") in p.stderr


@pytest.mark.slow
def test_verify_refuses_the_recursive_inner_session_at_its_coins():
    """Without `--zk`, `verify --coins tree` reads a session on the challenger's tree of the statement's own streams
    (`Zk.coinSpec`), so rec-coins' inner session (`fixtures/rec-coins`' `inner-honest`: a plain-leaf statement on the tree
    of the recursive verifier's streams, `rec_live.INNER_COIN_SPEC`) is refused at S2, before any check reads its leaves;
    the recursive verifier checks that session (`test_lean_rec_coins.py`)."""
    d = Path(__file__).resolve().parent / "fixtures" / "rec-coins" / "16-16-8-n64"
    assert (d / "inner-honest" / "session").is_dir(), (
        f"{d} is not fetched: `uv run research data fetch-fixtures verity/protocols/flock/prover/tests/fixtures/rec-coins`")
    p = _verify("--circuit", str(d / "u" / "stage" / "circuit.txt"), "--public", str(d / "u" / "stage" / "pub-64.bin"),
                "--partition", str(d / "u" / "stage" / "partition.json"), "--coins", "tree",
                "--session", str(d / "inner-honest" / "session"))
    verdicts = [json.loads(line[len("VERDICT "):]) for line in p.stdout.splitlines() if line.startswith("VERDICT ")]
    assert p.returncode == 1 and len(verdicts) == 1, p.stdout + p.stderr
    assert not verdicts[0]["accepted"]
    assert verdicts[0]["why"].startswith("S2"), verdicts[0]["why"]
