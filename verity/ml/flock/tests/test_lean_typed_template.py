"""Item 1e for a template: the Lean verifier reads a typed template statement (`verity/flock-circuit/types`, GEMM) as #273's
Rust prover splits it: the instance type's root one slot per VU and each placed layout's slots its parts, with rows only
through `deriveChecked`, the row wiring derived, and Δ in the prover's order (the flat entries, then per VU the inputs'
copies, the parts' bound rows and the root's cross entries). A typed statement is a form the soundness proof does not
cover (§16.15), so on the prover's recorded sessions the statement sets up and every case is refused: the honest ones for
its form, once every check has passed, and every other case for the check that catches it. That build has no zk patch, so
its plain files are upstream's form, without the `zk_level0 = None` tag the deployed prover writes, and are read under
`--upstream-plain` (PROTOCOL.md §8.1)."""
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
STATEMENT = "verity/flock-circuit/types@210d32e1+seed-injection"
#: the typed GEMM (gemm-coordinate/k64/sm80-mma-bf16, 4 instances) staged at #273, and every CPU selftest case of its
#: flock-circuit (seed-injection build, #273 at 8505c540: the typed id's own tags), recorded
FIXTURE = "art:e9c0209d"

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


@pytest.fixture(scope="module")
def gemm(tmp_path_factory):
    sys.path.insert(0, str(VERIFIER))
    import ci  # noqa: PLC0415

    store = tmp_path_factory.mktemp("store")
    try:
        root = ci.fetch(f"{FIXTURE}/stage", store).parent
    except (subprocess.CalledProcessError, OSError) as e:
        pytest.skip(f"the evidence store is not reachable: {e}")
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    return root, store


def _statement(circuit: Path, public: Path) -> str:
    p = subprocess.run([str(BIN), "statement", "--statement", STATEMENT, "--circuit", str(circuit), "--public", str(public)],
                       capture_output=True, text=True)
    return p.stdout.strip().splitlines()[-1]


def test_the_verifier_reads_a_typed_template(gemm):
    root, _ = gemm
    circuit = root / "stage" / "circuit.txt"
    st = json.loads(_statement(circuit, root / "stage" / "pub-4.bin"))
    assert st["statement"] == "verity/flock-circuit/types"
    assert st["circuit_sha512"] == hashlib.sha512(circuit.read_bytes()).hexdigest() and st["regions"] == 3


@pytest.mark.slow
def test_typed_expand_reads_the_public_output_form(gemm):
    """`typed-expand --statement` parses a `types@210d32e1` file as `verify` does; the default, `verity/flock-circuit/types`
    with hidden outputs (PROTOCOL.md §16.13), refuses it."""
    root, _ = gemm
    circuit = root / "stage" / "circuit.txt"

    def expand(*statement: str) -> dict:
        p = subprocess.run([str(BIN), "typed-expand", str(circuit), *statement], capture_output=True, text=True)
        return json.loads(next(ln for ln in p.stdout.splitlines() if ln.startswith("TYPED ")).split(" ", 1)[1])

    old = expand("--statement", STATEMENT)
    assert old["ok"] and old["sha512"] == hashlib.sha512(circuit.read_bytes()).hexdigest(), old
    assert not expand()["ok"]


@pytest.mark.slow
def test_it_refuses_the_rust_provers_sessions(gemm, tmp_path):
    root, _ = gemm
    subprocess.run([sys.executable, str(VERIFIER / "selftest_records.py"), str(root / "records"), str(tmp_path)], check=True,
                   capture_output=True)
    sessions = sorted(p for p in tmp_path.iterdir() if (p / "case.json").exists())
    p = subprocess.run([str(BIN), "verify", "--statement", STATEMENT, "--circuit", str(root / "stage" / "circuit.txt"),
                        "--public", str(root / "stage" / "pub-4.bin"), "--tables", str(root / "stage" / "tables"),
                        "--upstream-plain", "--session", *map(str, sessions)], capture_output=True, text=True)
    got = {Path(v["session"]).name: v for v in (json.loads(ln[8:]) for ln in p.stdout.splitlines() if ln.startswith("VERDICT "))}
    want = {s.name: json.loads((s / "case.json").read_text())["expect"] == "accept" for s in sessions}
    refused = f"refused: {STATEMENT}, a typed statement: a statement form the soundness proof does not cover"
    assert not any(got[n]["accepted"] for n in want)
    assert {n: got[n]["why"] == refused for n in want} == want, {n: got[n]["why"] for n in want}
    assert sum(want.values()) == 3 and len(want) == 20


def test_the_verifiers_rows_are_the_writers(gemm, tmp_path):
    """The rows the verifier derives through `deriveChecked` (`typed-rows`) are the writer's `rows/` byte for byte: the
    root, the placed layout and `delta.txt`."""
    root, _ = gemm
    p = subprocess.run([str(BIN), "typed-rows", str(root / "stage" / "circuit.txt"), "--out", str(tmp_path / "rows")],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    want = sorted((root / "stage" / "rows").iterdir())
    assert [f.name for f in want] == sorted(f.name for f in (tmp_path / "rows").iterdir())
    for f in want:
        assert (tmp_path / "rows" / f.name).read_bytes() == f.read_bytes(), f.name


def _rewrite(root: Path, out: Path, edit_meta=None, extra: str = "") -> Path:
    head, meta, rest = (root / "stage" / "circuit.txt").read_text().split("\n", 2)
    m = json.loads(meta[len("META "):])
    if edit_meta:
        edit_meta(m)
    out.write_text(head + "\nMETA " + json.dumps(m, sort_keys=True, separators=(",", ":")) + "\n" + extra + rest)
    return out


@pytest.mark.slow
@pytest.mark.parametrize("key", ["wires", "units_per_vu", "leaves_in", "out_net"])
def test_a_template_stating_its_derived_wiring_is_refused(gemm, tmp_path, key):
    root, _ = gemm
    bad = _rewrite(root, tmp_path / "bad.circuit", lambda m: m.__setitem__(key, []))
    assert "a typed template's block wiring is derived" in _statement(bad, root / "stage" / "pub-4.bin")


def test_a_template_carrying_root_rows_is_refused(gemm, tmp_path):
    root, _ = gemm
    bad = _rewrite(root, tmp_path / "bad.circuit", extra="CIRCUIT root 1\nflock-ir-unit/v2 root 0 0 0 0\n")
    assert "carries no unit rows" in _statement(bad, root / "stage" / "pub-4.bin")
