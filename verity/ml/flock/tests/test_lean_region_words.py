"""The region-word check (`docs/region-word-check.md`, the red team's C3 on the public-circuit ZK proof): the Lean verifier
refuses an hm96-sha512 statement in which a column shares a 128-bit word with a region's columns without being one of them,
unless its rows in the combined A and B (the slot type's row plus Δ, summed over GF(2)) are both empty. On the pinned GEMM
coordinate statement (set 13 of `vectors.json`: its tail stage's `Out` region is 16 bits of a word), the 112 other columns
of that word are checked and pass; a Δ entry into one is refused and a cancelling pair passes; a computed row there, which
the current parser accepts, is refused."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from verity.ml.circuits.codec import canonical_json

LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
STATEMENT = "verity/flock-circuit@e51e2b86+seed-injection"
GEMM = "art:3104c2f9/gemm-k1024/stage"
#: the tail stage `stage1`'s range starts at slot 1057 of 2^9 columns; its output word is its word 1, the region its 16 bits
SHARED = 1057 * 512 + 128 + 16

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


@pytest.fixture(scope="module")
def gemm(tmp_path_factory):
    """The GEMM coordinate statement's circuit and public file, from the evidence store as `ci.py` fetches them."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verifier"))
    import ci  # noqa: PLC0415

    store = tmp_path_factory.mktemp("store")
    try:
        circuit, public = ci.fetch(f"{GEMM}/circuit.txt", store), ci.fetch(f"{GEMM}/pub-4.bin", store)
    except (subprocess.CalledProcessError, OSError) as e:
        pytest.skip(f"the evidence store is not reachable: {e}")
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    return circuit, public, store


def _run(cmd: str, circuit: Path, public: Path, *extra: str) -> str:
    p = subprocess.run([str(BIN), cmd, "--statement", STATEMENT, "--circuit", str(circuit), "--public", str(public), *extra],
                       capture_output=True, text=True)
    return p.stdout


def _words(circuit: Path, public: Path, *extra: str) -> dict:
    out = _run("region-words", circuit, public, *extra)
    return json.loads(next(ln for ln in out.splitlines() if ln.startswith("REGIONWORDS ")).split(" ", 1)[1])


def test_the_gemm_statement_passes_with_its_shared_columns(gemm):
    circuit, public, _ = gemm
    assert _words(circuit, public) == {"ok": True, "shared": 112}


def test_a_delta_entry_into_a_shared_column_is_refused_and_a_pair_cancels(gemm):
    circuit, public, _ = gemm
    one = _words(circuit, public, "--delta", f"{SHARED},5")
    assert not one["ok"] and f"column {SHARED} shares a word" in one["error"], one
    assert _words(circuit, public, "--delta", f"{SHARED},5", "--delta", f"{SHARED},5") == {"ok": True, "shared": 112}


def _class(b: bytes) -> str:
    n1 = b.index(b"\n")
    n2 = b.index(b"\n", n1 + 1)
    meta = {k: v for k, v in json.loads(b[n1 + 6:n2]).items() if k not in ("partition", "program_sha512", "program_digests")}
    return hashlib.sha512(b[:n1] + b"\nMETA " + canonical_json(meta) + b"\n" + b[n2 + 1:]).hexdigest()


def test_a_computed_row_in_a_shared_column_is_refused(gemm):
    """`stage1`'s row 144 (bit 16 of its output word) given `A = [const]`; the public file rebound to the altered circuit
    (its digest and class), so the region-word check is what refuses it."""
    circuit, public, store = gemm
    good = circuit.read_bytes()
    lines = good.decode().split("\n")
    i = 2
    while lines[i]:
        _, name, n = lines[i].split(" ")
        if name == "stage1":
            const = int(lines[i + 1].split(" ")[3])
            assert lines[i + 2 + 144] == "0 0"
            lines[i + 2 + 144] = f"1 {const} 0"
        i += 1 + int(n)
    bad = "\n".join(lines).encode()
    pub = public.read_bytes().replace(hashlib.sha512(good).hexdigest().encode(), hashlib.sha512(bad).hexdigest().encode())
    pub = pub.replace(_class(good).encode(), _class(bad).encode())
    (store / "bad.circuit").write_bytes(bad)
    (store / "bad.pub").write_bytes(pub)
    out = _run("statement", store / "bad.circuit", store / "bad.pub")
    assert out.startswith("setup: region out: column") and "is not forced to zero" in out, out
