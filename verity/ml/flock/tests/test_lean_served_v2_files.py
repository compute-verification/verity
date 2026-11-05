"""`ServedFilesParseV2` (`Security/Proofs/Flock/Bridge/Assumptions/ServedV2.lean`), the named premise of `KSByDeployed`'s
honest witness, held to the code on the files it pins: `test_lean_served_registered.py`'s row/v2 session.

* The stage's circuit, public, partition, program and registered files have the premise's SHA-512 digests, and they and
  the session record's draw fit a statement's 8-byte length words (`RawFits`).
* `tests/lean/ServedV2Files.lean`, built against the verifier's own build, computes each of the premise's facts of the
  files with the function the premise states it with: `Desc.parse`, `HmIn.parse` giving no ports, `HmIn.loadPublic` at
  none with 32 instances, every port a `row/v2` row, `PortRoots.check`, `KsScope.check` at one table, and at the session
  record's draw, which proves 16 units, `setupHidden`'s table at the partition's JSON and the program passing
  `Zk.shapeOk` with a `fast100` and a `--zk` schedule.
* `flock-verify verify --zk` on the files and `registered.json` accepts the session: it sets the statement up as
  `FlockDeploy.Ins.setup` states, at that draw, so at 16 instances.

Each step parses the 56 MB circuit and the setup builds an m = 25 table, so the module is slow; it is the one check of the
premise's computation, which the kernel can't run."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
VERIFIER = Path(__file__).resolve().parents[1] / "verifier"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
#: `test_lean_served_registered.py`'s row/v2 fixture
FIXTURE = "art:cd47a8fe806ef14308e22d07275d125334595b3a73d63ef8626ed762a7b9c181"
#: `ServedFilesParseV2`'s pins: each stage file's SHA-512
DIGESTS = {
    "circuit.txt": "0c494f886ebc835e7d2a9050f1017b76e852eed07a799f7e54a01c8e306c7d787365009224f90d5ecafd4140c85a6aad6576e434e279bea2dfc1f4e9a1742ae1",
    "pub-32.bin": "95dc2441381ee37989575ebadffb97c4216ab6e9988bbe2759bdf7f5d5a63fb9d7fbb5adc9f066817e0ed65b001db31ff5947c50ae625e0ee667e4f694f5a912",
    "partition.json": "88587f18568664e1448f319283bc94a4504a3fd99f71b5dfd15bb84d90997676d7c71c8754b752f81a66cf0b469d1320ab59963a4922c644beedfbcd89b69515",
    "program.json": "7c9c10d6e16498f6f3264a7ba48f4aee81b828d8481d56a5d5082c97888a5d8a214920adea30e4998c46b913333bdabcda7ad3abc6faa5022ca2d05422d6b8c4",
    "registered.json": "2f9bfa9260258819e8edee20c9cd5f2d6e04acbe6f87d78baf83c3818d30e272159082749f3aa5580702807865479f0dfe3bbae0ace3745c7e9fe4762b653a3b",
}
STATEMENT = "verity/flock-circuit"

pytestmark = [pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"), pytest.mark.slow]


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    sys.path.insert(0, str(VERIFIER))
    import ci

    try:
        return ci.fetch_sessions(FIXTURE, tmp_path_factory.mktemp("store"))
    except (subprocess.CalledProcessError, OSError) as e:
        pytest.skip(f"the evidence store is not reachable: {e}")


@pytest.fixture(scope="module")
def verifier():
    subprocess.run(["lake", "build", "flock-verify"], cwd=LEAN, check=True, capture_output=True)
    return BIN


@pytest.fixture(scope="module")
def files_bin(tmp_path_factory, verifier):
    """`tests/lean/ServedV2Files.lean` built against the verifier's own build, as a package that requires it by path."""
    pkg = tmp_path_factory.mktemp("served-v2-files")
    (pkg / "lakefile.toml").write_text(
        f'name = "served_v2_files"\ndefaultTargets = ["served-v2-files"]\n\n[[require]]\nname = "flock_verifier"\n'
        f'path = "{LEAN}"\n\n[[lean_exe]]\nname = "served-v2-files"\nroot = "ServedV2Files"\n')
    shutil.copy(Path(__file__).resolve().parent / "lean" / "ServedV2Files.lean", pkg)
    p = subprocess.run(["lake", "-d", str(pkg), "build"], cwd=LEAN, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-4000:]
    return pkg / ".lake" / "build" / "bin" / "served-v2-files"


def test_the_files_are_the_premises_and_fit_a_statements_length_words(fx):
    st = fx / "stage"
    assert {f: hashlib.sha512((st / f).read_bytes()).hexdigest() for f in DIGESTS} == DIGESTS
    draw = json.dumps(json.loads((fx / "session" / "session.json").read_text())["unit_draw"]).encode()
    for b in [*((st / f).read_bytes() for f in DIGESTS), draw, STATEMENT.encode()]:
        assert len(b) < 2 ** 64


def test_each_fact_of_the_premise_holds_on_the_files(fx, files_bin):
    st = fx / "stage"
    p = subprocess.run([str(files_bin), str(st / "circuit.txt"), str(st / "pub-32.bin"), str(st / "partition.json"),
                        str(st / "program.json"), str(fx / "session" / "session.json")],
                       capture_output=True, text=True, check=False)
    assert p.returncode == 0, p.stdout + p.stderr
    assert json.loads(p.stdout) == {"hm_in_none": True, "pub_n": 32, "rows_v2": True, "port_roots": "ok",
                                    "ks_scope": "ok", "proved": 16, "shape_ok": "ok", "fast100": True, "schedule": "ok"}


def test_verify_zk_sets_the_statement_up_and_accepts_the_session(fx, verifier):
    st = fx / "stage"
    p = subprocess.run([str(verifier), "verify", "--zk", "--statement", STATEMENT, "--circuit", str(st / "circuit.txt"),
                        "--public", str(st / "pub-32.bin"), "--partition", str(st / "partition.json"), "--program",
                        str(st / "program.json"), "--registered", str(st / "registered.json"), "--session",
                        str(fx / "session")], capture_output=True, text=True, check=False)
    (line,) = [json.loads(x.split(" ", 1)[1]) for x in p.stdout.splitlines() if x.startswith("VERDICT ")]
    assert (line["accepted"], line["why"]) == (True, ""), line
