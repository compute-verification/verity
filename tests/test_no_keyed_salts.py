"""No source under `verity/`, `integrations/` or `adapters/` derives a hiding leaf's salt from a key, by any function: every
leaf's salt is fresh operating-system bytes, drawn once for its leaf (`verity.commitments.hm96.SHA512.fresh_salt()`), and
handed on as data.

The scan reads tracked Python, Lean, CUDA, C and Rust sources for the keyed derivations found by reading the tree, by
name or by the domain label each hashes under (`NAMES`): the hm96 salt function and the ChaCha20 block it is built on,
keyed salt tables, Flock's keyed staging salts, the ZK server's value salts, the SHA-512 salt PRF of `commit_gpu` and
its twins, PoUB's store salts, and a salt key under any spelling. A file that names one fails unless `ALLOWED` lists it
with the lane that owns its switch. `ALLOWED` only shrinks: an entry whose file names none any more is stale, and its
line goes.

The scan catches a new use of a derivation it knows by name; it cannot see a derivation under a name or label it does not
know, so a green run says no known derivation spread, not that the rule holds."""

from __future__ import annotations

import re

import pytest
from test_repository import ROOT, _tracked

SCOPES = ("verity/", "integrations/", "adapters/")
SUFFIXES = (".py", ".lean", ".cu", ".cuh", ".c", ".cc", ".cpp", ".h", ".hpp", ".rs")
#: a salt key in any spelling (`salt_key`, `saltKey`, `hm96_salt_key`, `--salt-key`, `FC_ZK_SERVER_SALT_KEY`), not a salt
#: object's keys; the hm96 salt function and the ChaCha20 block it is built on; keyed salt tables; `stage_salts`' keyed
#: branch and the ZK server's `saltOf` by their domain labels; `commit_gpu`'s SHA-512 salt PRF and its twins; PoUB's
#: `store_salt` under a key; and the vllm committer's `salt(ks, pos)`
NAMES = re.compile(r"(?i:salt[_-]?key)(?![sS])|\bhm96_salt\b|\bhm96Salt\b|\bkeyed_salts\b|\bchacha20_block\b|\bchacha20Block\b"
                   r"|flock-circuit/stage-salt/v1|zk-server/value-salt|\bprf_salt\b|\bsalt_prf\b|\bstore_salt\(\s*key\b"
                   r"|\bsalt\(ks\b")
FRESH = "verity.commitments.hm96.SHA512.fresh_salt()"
FIX = (f"a leaf's salt is fresh operating-system bytes, one draw a leaf ({FRESH}), handed to whatever hashes the leaf as "
       "data: never derived from a key, and no key is written to make them again. Take the salts as an argument "
       "instead of the key; ALLOWED in tests/test_no_keyed_salts.py only shrinks")

#: {file that still names a keyed derivation: the lane that switches it to fresh salts}
ALLOWED: dict[str, str] = {
    "adapters/torch/verity_torch/pouw/h1/registered_device.py": "@compute-accounting",
    "adapters/torch/verity_torch/pouw/pearl_c/ends.py": "@compute-accounting",
    "adapters/torch/verity_torch/pouw/pearl_c/host.py": "@compute-accounting",
    "adapters/torch/verity_torch/pouw/pearl_c/run_device.py": "@compute-accounting",
    "adapters/vllm/verity_vllm/commit/committer/native_rows.cu": "@vllm",
    "adapters/vllm/verity_vllm/commit/step_rows.py": "@vllm",
    "adapters/vllm/verity_vllm/pipeline/interior_links.py": "@vllm",
    "adapters/vllm/verity_vllm/pipeline/interior_rows.py": "@vllm",
    "adapters/vllm/verity_vllm/protocols/pouw/__init__.py": "@vllm",
    "verity/commitments/crypto/chacha20.py": "@proofs",
    "verity/commitments/cuda/commit_gpu/commit_gpu.cuh": "@kernels",
    "verity/commitments/cuda/commit_gpu/reference.py": "@kernels",
    "verity/commitments/cuda/commit_slots/slots.py": "@kernels",
    "verity/commitments/cuda/commit_slots/slots_gpu.py": "@kernels",
    "verity/commitments/cuda/sha512_rows/check.py": "@kernels",
    "verity/commitments/cuda/sha512_rows/engine.py": "@kernels",
    "verity/commitments/cuda/sha512_rows/rows.cuh": "@kernels",
    "verity/commitments/hm96/batch.py": "@proofs",
    "verity/protocols/experimental/dither/prover/dither.py": "@compute-accounting",
    "verity/protocols/proofs/flock/prover/live/src/bin/flock-circuit.rs": "@proofs",
    "verity/protocols/proofs/flock/prover/live/src/zk_hooks.rs": "@proofs",
    "verity/protocols/proofs/flock/prover/live/src/zk_server.rs": "@proofs",
    "verity/protocols/proofs/flock/prover/python/verity_flock/zk_server.py": "@proofs",
    "verity/protocols/proofs/flock/verifier/Flock/ZkServer/Salts.lean": "@proofs",
    "verity/protocols/proofs/flock/verifier/Flock/ZkServer/Session.lean": "@proofs",
    "verity/protocols/proofs/flock/verifier/Flock/ZkServer/Values.lean": "@proofs",
    "verity/protocols/proofs/recursion/verifier/zk_server/ZkServer/Loop.lean": "@proofs",
    "verity/protocols/resource_accounting/compute/pouw/prover/clear/clear_audit.py": "@proofs",
    "verity/protocols/resource_accounting/compute/pouw/prover/kernels/pearl_c/hash_h3.cuh": "@kernels",
    "verity/protocols/resource_accounting/compute/pouw/prover/kernels/pearl_c_sm120/hash_decode.cuh": "@kernels",
    "verity/protocols/resource_accounting/compute/pouw/prover/kernels/pearl_c_sm120/run.py": "@kernels",
    "verity/protocols/resource_accounting/compute/pouw/prover/kernels/pearl_c_sm90/grouped.py": "@kernels",
    "verity/protocols/resource_accounting/compute/pouw/prover/kernels/pearl_c_sm90/run.py": "@kernels",
    "verity/protocols/resource_accounting/compute/pouw/prover/kernels/pouw_cuda/registered_rows.cuh": "@kernels",
    "verity/protocols/resource_accounting/compute/pouw/prover/kernels/pouw_cuda/row_table.cuh": "@kernels",
    "verity/protocols/resource_accounting/compute/pouw/prover/kernels/pouw_cuda/zy_commit.cuh": "@kernels",
    "verity/protocols/resource_accounting/compute/pouw/prover/serving/__init__.py": "@compute-accounting",
    "verity/protocols/resource_accounting/compute/pouw/prover/serving/h1/prover.py": "@compute-accounting",
    "verity/protocols/resource_accounting/compute/pouw/prover/serving/h1/registered.py": "@compute-accounting",
    "verity/protocols/resource_accounting/compute/pouw/prover/serving/h1/requests.py": "@compute-accounting",
    "verity/protocols/resource_accounting/compute/pouw/prover/serving/h1/served.py": "@compute-accounting",
    "verity/protocols/resource_accounting/compute/pouw/prover/serving/ncp.py": "@compute-accounting",
    "verity/protocols/resource_accounting/compute/pouw/prover/serving/pearl_c/__init__.py": "@compute-accounting",
    "verity/protocols/resource_accounting/compute/pouw/verifier/audit.py": "@proofs",
    "verity/protocols/resource_accounting/compute/pouw/verifier/weights.py": "@compute-accounting",
    "verity/protocols/resource_accounting/memory/poub/prover/codec_tensor/reader.py": "@memory-accounting",
    "verity/protocols/resource_accounting/memory/poub/prover/node_store/registry.py": "@memory-accounting",
    "verity/protocols/resource_accounting/memory/poub/prover/poub_session/device.py": "@memory-accounting",
    "verity/protocols/resource_accounting/memory/poub/prover/poub_session/prover.py": "@memory-accounting",
    "verity/protocols/resource_accounting/memory/poub/verifier/store.py": "@memory-accounting",
    "verity/protocols/resource_accounting/memory/pous/prover/served_weights.py": "@memory-accounting",
}
LANES = {"@compute-accounting", "@kernels", "@lean", "@memory-accounting", "@network-accounting", "@proofs", "@vllm"}


def _keyed() -> dict[str, list[str]]:
    """{tracked source in a scope: the keyed derivations it names}."""
    found = {}
    for path in _tracked():
        if not path.startswith(SCOPES) or not path.endswith(SUFFIXES):
            continue
        file = ROOT / path
        if file.is_file() and (hits := sorted(set(NAMES.findall(file.read_text(errors="replace"))))):
            found[path] = hits
    return found


def test_no_file_derives_a_salt_from_a_key() -> None:
    new = {path: hits for path, hits in _keyed().items() if path not in ALLOWED}
    assert not new, f"{FIX}:\n" + "\n".join(f"  {path}: {', '.join(hits)}" for path, hits in sorted(new.items()))


def test_allowed_only_shrinks() -> None:
    keyed = _keyed()
    stale = sorted(path for path in ALLOWED if path not in keyed)
    assert not stale, ("these files no longer name a keyed salt derivation: delete their lines from ALLOWED in "
                       "tests/test_no_keyed_salts.py:\n" + "\n".join(f"  {path}" for path in stale))


def test_every_entry_names_its_owner() -> None:
    assert not {path: lane for path, lane in ALLOWED.items() if lane not in LANES}


@pytest.mark.parametrize(("text", "named"), [
    ("rows = [r + hm96_salt(key, nonce, i) for i, r in enumerate(rows)]", True),
    ("rows[i]! ++ hm96Salt k nonce i", True),
    ("man['hm96_salt_key']", True),
    ("def commit(salt_key: bytes)", True),
    ("(saltKey : ByteArray)", True),
    ("os.environ['FC_ZK_SERVER_SALT_KEY']", True),
    ("chacha20_block(key, 3 * i, nonce)", True),
    ("chacha20Block key (3 * i) nonce", True),
    ("salts = keyed_salts(key, n)", True),
    ("core-zk-server values --state DIR --salt-key FILE", True),
    ('tag = b"verity/flock-circuit/stage-salt/v1\\0" + key', True),
    ('def SALT_LABEL : ByteArray := label "verity/zk-server/value-salt/v1"', True),
    ("def prf_salt(ks: bytes, pos: int) -> bytes:", True),
    ("DEV void salt_prf(u64 pos, u64* Y) {", True),
    ("self.salts = [store_salt(key, PAYLOAD_LABEL, j) for j in range(n)]", True),
    ("hiding_leaf(r, salt(ks, i), n_bits, key)", True),
    ("key = store_salt(setup, doc, digest, source, commitment)", False),
    ("salts = stage_salts(iset, 0, 1, st.ports, None)", False),
    ("WINDOW_SALT_KEYS = ('coins', 'epoch')", False),
    ("HM96_SALT = 192", False),
    ("salt = SHA512.fresh_salt()", False),
])
def test_the_scan_reads_each_spelling(text: str, named: bool) -> None:
    assert bool(NAMES.search(text)) is named
