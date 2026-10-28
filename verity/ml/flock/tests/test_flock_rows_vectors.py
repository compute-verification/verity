"""`flock-rows --archive`, the verifier's `Flock.DeriveAll.deriveAll` compiled, gives the prover's mirror's rows on
the derive vectors (`derive_vectors.json`): each inner layout's physical rows, the unit's own region, its parts and Δ,
and the logical list, by SHA-512: calls placed and inlined, exports, and reads inline and placed. Those rows pass
`Flock.DeriveAll.deriveChecked`'s check, the one `compose_sound` is proved from."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
VECTORS = HERE / "derive_vectors.json"
pytestmark = [pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"),
              pytest.mark.skipif(not VECTORS.exists(), reason="the derive vectors are not in this tree yet")]


def cases():
    return json.loads(VECTORS.read_text())["cases"] if VECTORS.exists() else []


@pytest.fixture(scope="module")
def built():
    subprocess.run(["lake", "build", "flock-rows"], cwd=LEAN, check=True, capture_output=True)


@pytest.mark.parametrize("case", cases(), ids=lambda c: c["name"])
def test_flock_rows_matches_the_mirror(built, case, tmp_path):
    path = tmp_path / "archive.json"
    path.write_text(json.dumps(case["archive"]))
    parts = [("unit", case["physical"]["unit"]["sha512"]), ("delta", case["physical"]["delta"]),
             ("logical", case["logical"]["sha512"])]
    parts += [(digest, v["sha512"]) for digest, v in case["physical"]["layouts"].items()]
    for part, want in parts:
        out = subprocess.run([str(LEAN / ".lake/build/bin/flock-rows"), "--archive", str(path), "--part", part],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert hashlib.sha512(out.stdout.encode()).hexdigest() == want, f"{case['name']}: {part[:16]}"


@pytest.mark.parametrize("case", cases(), ids=lambda c: c["name"])
def test_derived_rows_pass_the_check(built, case, tmp_path):
    """`deriveChecked` accepts every vector's rows: calls and reads, inline or placed."""
    path = tmp_path / "archive.json"
    path.write_text(json.dumps(case["archive"]))
    out = subprocess.run([str(LEAN / ".lake/build/bin/flock-rows"), "--archive", str(path), "--part", "check"],
                         capture_output=True, text=True)
    assert (out.returncode, out.stdout) == (0, "ok\n"), out.stderr
