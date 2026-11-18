"""A hiding leaf's salt under `verity/`, `integrations/` or `adapters/` is one of two things, and either needs no entry
here: fresh operating-system bytes drawn for its leaf (`verity.commitments.hm96.SHA512.fresh_salt()`) and handed on as
data; or the one keyed stream the Lean lists (`KeyedUniform`), `verity.commitments.randomness`'s `stream` under a key its
`derive` makes from fresh operating-system bytes, drawn fresh for the unit it salts (a step, a pass, a registration) and
never reused. Any other derivation of a salt from a key is a third one. A leaf that hides nothing may take a public salt,
derived by SHA-512 from public names alone and said to be public where it is made; it is outside the rule. The rule is
stated in `verity/commitments/hm96/PROTOCOL.md`, section 3, "Randomness" (`RULE`).

The scan reads tracked Python, Lean, CUDA, C and Rust sources for the third derivations found by reading the tree, by name
or by the domain label each hashes under (`NAMES`): the hm96 salt function and the ChaCha20 block it is built on, keyed
salt tables, Flock's keyed staging salts, the ZK server's value salts, the SHA-512 salt PRF (`prf_salt`, `salt_prf`),
PoUB's store salts and payload key, and a salt key under any spelling. A file that names one fails unless a list
holds it:

* `ALLOWED`: derivations still to be removed, each with the lane that removes it. It only shrinks: an entry whose file
  names none any more is stale, and its line goes.
* `RULED`: derivations a ruling keeps for now, each with its owner and the rulings it rests on; each ruling has its
  condition and the place in the tree that states it (`STATED`), and `None` there warns until the owner writes that
  sentence. No line of a ruled module writes, sends, logs or serializes a key it names, except a step-rows key a replay
  must reopen rows with, under the private replay store its condition names (`WRITTEN` holds the files that write one
  anywhere else or where this test cannot tell, and only shrinks). An entry goes when its owner moves to an allowed form or
  the ruling changes.
* `TEST_BUILD`: files a proving build cannot contain, because every declaration of their module carries a test-build
  condition on the line before it and no default feature turns that condition on; the check reads the declaring file
  and the crate's `Cargo.toml`, not the file's name.

The scan catches a new use of a derivation it knows by name; it cannot see a derivation under a name or label it does not
know, nor whether a ruled key is drawn as often as its ruling says, so a green run says no known derivation spread, not
that the rule holds."""

from __future__ import annotations

import re
import tomllib

import pytest
from test_repository import ROOT, _tracked

SCOPES = ("verity/", "integrations/", "adapters/")
SUFFIXES = (".py", ".lean", ".cu", ".cuh", ".c", ".cc", ".cpp", ".h", ".hpp", ".rs")
#: a salt key in any spelling (`salt_key`, `saltKey`, `hm96_salt_key`, `--salt-key`, `FC_ZK_SERVER_SALT_KEY`), not a salt
#: object's keys; the hm96 salt function and the ChaCha20 block it is built on; keyed salt tables; `stage_salts`' keyed
#: branch and the ZK server's `saltOf` by their domain labels; the SHA-512 salt PRF (`prf_salt`, `salt_prf`); PoUB's
#: `store_salt` under a key, and its payload key; and the vllm committer's `salt(ks, pos)`
NAMES = re.compile(r"(?i:(?:salt|payload)[_-]?key)(?![sS])|\bhm96_salt\b|\bhm96Salt\b|\bkeyed_salts\b|\bchacha20_block\b|\bchacha20Block\b"
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
    "adapters/torch/verity_torch/pouw/pearl_c/run_device.py": "@compute-accounting",
    # the step rows' processor twin is ruled (RULED's native_rows.cu); the registered weights' `tensor_key` expansion here is not
    "adapters/vllm/verity_vllm/commit/step_rows.py": "@vllm",
    "adapters/vllm/verity_vllm/protocols/pouw/__init__.py": "@vllm",
    "verity/commitments/hm96/batch.py": "@proofs",
    "verity/protocols/proofs/flock/prover/live/src/bin/flock-circuit.rs": "@proofs",
    "verity/protocols/proofs/flock/prover/live/src/zk_server.rs": "@proofs",
    "verity/protocols/proofs/flock/prover/python/verity_flock/zk_server.py": "@proofs",
    "verity/protocols/proofs/flock/verifier/Flock/ZkServer/Salts.lean": "@proofs",
    "verity/protocols/proofs/flock/verifier/Flock/ZkServer/Session.lean": "@proofs",
    "verity/protocols/proofs/flock/verifier/Flock/ZkServer/Values.lean": "@proofs",
    "verity/protocols/proofs/recursion/verifier/zk_server/ZkServer/Loop.lean": "@proofs",
    # both keep ChaCha20 only for compute-accounting's callers that pass a key and a nonce and no salts
    "verity/protocols/resource_accounting/compute/pouw/prover/kernels/pouw_cuda/registered_rows.cuh": "@vllm",
    "verity/protocols/resource_accounting/compute/pouw/prover/kernels/pouw_cuda/row_table.cuh": "@vllm",
    "verity/protocols/resource_accounting/compute/pouw/prover/serving/__init__.py": "@compute-accounting",
    "verity/protocols/resource_accounting/compute/pouw/prover/serving/h1/prover.py": "@compute-accounting",
    "verity/protocols/resource_accounting/compute/pouw/prover/serving/h1/registered.py": "@compute-accounting",
    "verity/protocols/resource_accounting/compute/pouw/prover/serving/h1/requests.py": "@compute-accounting",
    "verity/protocols/resource_accounting/compute/pouw/prover/serving/h1/served.py": "@compute-accounting",
    "verity/protocols/resource_accounting/compute/pouw/prover/serving/ncp.py": "@compute-accounting",
    "verity/protocols/resource_accounting/compute/pouw/prover/serving/pearl_c/__init__.py": "@compute-accounting",
    "verity/protocols/resource_accounting/compute/pouw/verifier/audit.py": "@compute-accounting",
    "verity/protocols/resource_accounting/compute/pouw/verifier/weights.py": "@compute-accounting",
}
#: the rulings that admit a keyed derivation: {ruling: its condition}
PASS_TREE = ("the bandwidth audit's pass tree: its leaves' salts are expanded on the card from a key drawn fresh from the "
             "operating system for each pass, never reused and never written")
PAYLOAD_TREE = ("the bandwidth audit's payload tree: its salts are expanded from the prover's own key, drawn from the "
                "operating system, which the verifier neither chooses nor receives and which is never written")
STEP_ROWS = ("the step rows: their salts are expanded on the card by their own SHA-512 expansion, until the shared expander "
             "of the listed stream exists, from a key drawn fresh from the operating system for each step and never reused; "
             "a key a replay must reopen rows with may be written to one place only: the store's private directory, outside "
             "every run's output, on the adapter's private list, refused by custody, and never sent to a verifier")
#: where the tree states the salts' rule and its ruled derivations: hm96's protocol, section 3, "Randomness"
RULE = ("verity/commitments/hm96/PROTOCOL.md", "derivations kept for now by a ruling, each with its condition")
#: {ruling: (the file in the tree that states it as a rule, a phrase of that statement), or None while nothing does}
STATED: dict[str, tuple[str, str] | None] = {PASS_TREE: RULE, PAYLOAD_TREE: RULE, STEP_ROWS: RULE}
#: {ruling: (the module it covers, the names its keys go by there)}
SCOPE = {PASS_TREE: ("verity/protocols/resource_accounting/memory/poub/", ("salt_key", "salt-key")),
         PAYLOAD_TREE: ("verity/protocols/resource_accounting/memory/poub/", ("payload_key", "payload-key")),
         STEP_ROWS: ("adapters/vllm/verity_vllm/commit/", ("keys.bin",))}
#: {file whose keyed derivation a ruling admits: (the lane that owns it, the rulings it rests on)}
RULED: dict[str, tuple[str, tuple[str, ...]]] = {
    "adapters/vllm/verity_vllm/commit/committer/native_rows.cu": ("@vllm", (STEP_ROWS,)),
    "verity/protocols/resource_accounting/memory/poub/prover/codec_tensor/reader.py": ("@memory-accounting", (PASS_TREE,)),
    "verity/protocols/resource_accounting/memory/poub/prover/node_store/registry.py": ("@memory-accounting", (PASS_TREE,)),
    "verity/protocols/resource_accounting/memory/poub/prover/poub_session/device.py": ("@memory-accounting",
                                                                                      (PASS_TREE, PAYLOAD_TREE)),
    "verity/protocols/resource_accounting/memory/poub/verifier/store.py": ("@memory-accounting", (PASS_TREE, PAYLOAD_TREE)),
}
#: a line that writes, sends, logs or serializes what it names: a writer's call, a key's hex, or the name as a string's
#: field (a manifest's or a message's)
WRITES = re.compile(r"\.(?:send|write|write_bytes|write_text|dump|dumps|save|savez|tofile)\(|\bprint\(|\blog(?:ger)?\.|\.hex\(\)")
#: {file of a ruled module that still writes or sends a ruled key where its ruling does not admit it: the lane that stops it}
#: A step-rows key is admitted only under the private replay store, which no single line shows: a file can leave only once
#: its writer takes its directory from one function that names that store, outside every run's output, and the file's name
#: is on the adapter's private list (`verity_vllm.pipeline.research_outputs.PRIVATE`), with a check here that reads both.
WRITTEN: dict[str, str] = {
    # `keys.bin` goes into the replay bundle, `<out>/replay_bundle_p<pair>/store/`, inside the commit's output, private by the
    # bundle directory's name alone
    "adapters/vllm/verity_vllm/commit/store_dump.py": "@vllm",
}
LANES = {"@compute-accounting", "@kernels", "@lean", "@memory-accounting", "@network-accounting", "@proofs", "@vllm"}
#: {file that derives salts only in a test build: (the lane that owns it, the file that declares its module, the module's
#: name, the condition every declaration of it carries on the line before)}. A proving build has none of these modules;
#: Flock's verifier refuses a statement made with the `seed-injection` feature by name (`versionWhy` in
#: `verity/protocols/proofs/flock/verifier/Flock/ProvedScope.lean`).
TEST_BUILD: dict[str, tuple[str, str, str, str]] = {
    "verity/protocols/proofs/flock/prover/live/src/seed_injection.rs": (
        "@proofs", "verity/protocols/proofs/flock/prover/live/src/lib.rs", "seed_injection",
        '#[cfg(any(test, feature = "seed-injection"))]'),
}


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
    new = {path: hits for path, hits in _keyed().items()
           if path not in ALLOWED and path not in RULED and path not in TEST_BUILD}
    assert not new, f"{FIX}:\n" + "\n".join(f"  {path}: {', '.join(hits)}" for path, hits in sorted(new.items()))


def _default_features(cargo: str) -> set[str]:
    """The features a `Cargo.toml`'s `default` turns on, through the features it names in turn."""
    features = tomllib.loads(cargo).get("features", {})
    on: set[str] = set()
    todo = list(features.get("default", []))
    while todo:
        f = todo.pop()
        if f not in on and "/" not in f and not f.startswith("dep:"):
            on.add(f)
            todo.extend(features.get(f, []))
    return on


def _test_build_faults(declaring: str, module: str, condition: str, cargo: str) -> list[str]:
    """Why `module`, declared in the text `declaring` of a crate whose manifest is `cargo`, is not in test builds only."""
    lines = declaring.splitlines()
    decl = re.compile(rf"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+{re.escape(module)}\s*;")
    at = [i for i, line in enumerate(lines) if decl.match(line)]
    faults = [] if at else [f"`mod {module};` is not declared"]
    for i in at:
        attrs, j = [], i - 1
        while j >= 0 and lines[j].strip().startswith("#["):
            attrs.append(lines[j].strip())
            j -= 1
        if condition not in attrs:
            faults.append(f"line {i + 1}, `{lines[i].strip()}`, is not right after `{condition}`")
    if on := sorted(set(re.findall(r'feature\s*=\s*"([^"]+)"', condition)) & _default_features(cargo)):
        faults.append(f"the crate's default features turn on {', '.join(on)}")
    return faults


def _cargo_of(path: str) -> str:
    """The `Cargo.toml` of the crate that holds `path`."""
    d = (ROOT / path).parent
    while not (d / "Cargo.toml").is_file():
        d = d.parent
    return (d / "Cargo.toml").read_text()


def test_a_test_build_module_is_declared_only_under_its_condition() -> None:
    faults = {path: _test_build_faults((ROOT / declaring).read_text(), module, condition, _cargo_of(declaring))
              for path, (_, declaring, module, condition) in TEST_BUILD.items()}
    faults = {path: why for path, why in faults.items() if why}
    assert not faults, ("a TEST_BUILD file derives salts only because a proving build lacks its module: every `mod` "
                        "line declaring it comes right after its condition, and no default feature turns the condition "
                        "on. Put the condition back on the declaration and take the feature out of `default`, or give "
                        f"the file fresh salts ({FRESH}) and delete its TEST_BUILD line:\n"
                        + "\n".join(f"  {path}: {why}" for path, whys in sorted(faults.items()) for why in whys))


CFG = '#[cfg(any(test, feature = "seed-injection"))]'
CARGO = '[features]\nseed-injection = []\nglue = []\n'


@pytest.mark.parametrize(("declaring", "cargo", "faults"), [
    (f"pub mod a;\n{CFG}\nmod seed_injection;\npub mod b;\n", CARGO, 0),
    ("pub mod a;\nmod seed_injection;\n", CARGO, 1),
    ('#[cfg(test)]\nmod seed_injection;\n', CARGO, 1),
    (f"{CFG}\nmod seed_injection;\n#[cfg(test)]\npub mod seed_injection;\n", CARGO, 1),
    (f"{CFG}\nmod seed_injection;\n", CARGO.replace("[features]\n", '[features]\ndefault = ["seed-injection"]\n'), 1),
    (f"{CFG}\nmod seed_injection;\n", CARGO.replace("[features]\n", '[features]\ndefault = ["harness"]\n'
                                                    'harness = ["seed-injection"]\n'), 1),
    ("pub mod a;\n", CARGO, 1),
])
def test_the_test_build_check_reads_each_form(declaring: str, cargo: str, faults: int) -> None:
    assert len(_test_build_faults(declaring, "seed_injection", CFG, cargo)) == faults


def test_allowed_only_shrinks() -> None:
    keyed = _keyed()
    stale = sorted(path for path in [*ALLOWED, *TEST_BUILD] if path not in keyed)
    assert not stale, ("these files no longer name a keyed salt derivation: delete their lines from ALLOWED or TEST_BUILD in "
                       "tests/test_no_keyed_salts.py:\n" + "\n".join(f"  {path}" for path in stale))


def test_every_entry_names_its_owner() -> None:
    lanes = {**ALLOWED, **WRITTEN, **{path: lane for path, (lane, _) in RULED.items()},
             **{path: lane for path, (lane, *_) in TEST_BUILD.items()}}
    assert not {path: lane for path, lane in lanes.items() if lane not in LANES}


def test_a_ruled_file_is_not_allowed_too() -> None:
    assert not set(RULED) & set(ALLOWED), "a file is ALLOWED (it switches to fresh salts) or RULED (a ruling admits it)"


def test_every_ruling_is_stated_in_the_tree() -> None:
    """A ruling lives in the tree as a rule, not only in this test: its owner states it where the salts' rule is stated."""
    unstated = [ruling for ruling, at in STATED.items() if at is None or at[1] not in (ROOT / at[0]).read_text()]
    assert not unstated, ("these rulings admit a keyed derivation and no file of the tree states them; their owners write "
                          "the sentence and STATED names it:\n" + "\n".join(f"  {ruling}" for ruling in unstated))


def _writes(line: str, names: tuple[str, ...]) -> bool:
    """`line` names one of `names` and writes, sends, logs or serializes it."""
    return any(n in line for n in names) and bool(WRITES.search(line) or any(f"'{n}'" in line or f'"{n}"' in line
                                                                              for n in names))


def _written() -> dict[str, list[str]]:
    """{tracked source of a ruled module: its lines that write, send, log or serialize a key the ruling names}."""
    found: dict[str, list[str]] = {}
    for path in _tracked():
        if not path.endswith(SUFFIXES) or not (ROOT / path).is_file():
            continue
        names = tuple(n for module, keys in SCOPE.values() if path.startswith(module) for n in keys)
        if names:
            for i, line in enumerate((ROOT / path).read_text(errors="replace").splitlines(), 1):
                if _writes(line, names):
                    found.setdefault(path, []).append(f"{i}: {line.strip()}")
    return found


def test_a_ruled_key_is_never_written() -> None:
    new = {path: lines for path, lines in _written().items() if path not in WRITTEN}
    assert not new, ("a ruled key is never written, sent, logged or serialized: it is drawn from the operating system "
                     "and dropped, except a step-rows key a replay needs, written only under the private replay store "
                     "(STEP_ROWS in tests/test_no_keyed_salts.py):\n"
                     + "\n".join(f"  {path}: {line}" for path, lines in sorted(new.items()) for line in lines))


def test_written_only_shrinks() -> None:
    stale = sorted(set(WRITTEN) - set(_written()))
    assert not stale, ("these files no longer write a ruled key: delete their lines from WRITTEN in "
                       "tests/test_no_keyed_salts.py:\n" + "\n".join(f"  {path}" for path in stale))


@pytest.mark.parametrize(("line", "written"), [
    ("man['hm96_salt_key'] = key.hex()", True),
    ('self.conn.send({"type": "payload-key", "key": self.payload_key.hex()})', True),
    ("Path(out, 'salt_key').write_bytes(self.salt_key)", True),
    ("log.info('salt_key %s', self.salt_key)", True),
    ("self.salt_key = secrets.token_bytes(32)", False),
    ("salts = [store_salt(self.payload_key, PAYLOAD_LABEL, j) for j in range(n)]", False),
])
def test_the_write_check_reads_each_form(line: str, written: bool) -> None:
    assert _writes(line, ("salt_key", "payload_key", "payload-key", "hm96_salt_key")) is written


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
    ('self.conn.send({"type": "payload-key", "key": self.payload_key.hex()})', True),
    ("key = store_salt(setup, doc, digest, source, commitment)", False),
    ("salts = stage_salts(iset, 0, 1, st.ports, None)", False),
    ("WINDOW_SALT_KEYS = ('coins', 'epoch')", False),
    ("HM96_SALT = 192", False),
    ("salt = SHA512.fresh_salt()", False),
    ('salts = derive(os.urandom(32), "verity/vllm/step-salt/v1", {"step": step}).stream(("leaf", i))', False),
])
def test_the_scan_reads_each_spelling(text: str, named: bool) -> None:
    assert bool(NAMES.search(text)) is named
