"""D11 (PROTOCOL.md §17.1): `Net.ofRows` refuses an expanded circuit whose output group ends past its unit's slot,
`col + words > 2^unitLog / 128`, so a region at an output group's column (`Circuit.outRegion`) stays inside the slot. The
statement is `_bit_row_toy`'s: its unit has 1281 useful rows (a 2^11-row slot, 16 words) and one output group of one word,
moved here with the public file rebound to the moved circuit. Without the refusal, setup accepts every one of them.
Upstream's verifier doesn't bound the column; no vector exercises it, so this runs it on the Lean verifier."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_lean_verifier import LEAN, _bit_row_toy  # noqa: E402

from verity.ml.circuits.codec import canonical_json

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


def _class(b: bytes) -> str:
    n1 = b.index(b"\n")
    n2 = b.index(b"\n", n1 + 1)
    meta = {k: v for k, v in json.loads(b[n1 + 6:n2]).items() if k not in ("partition", "program_sha512", "program_digests")}
    return hashlib.sha512(b[:n1] + b"\nMETA " + canonical_json(meta) + b"\n" + b[n2 + 1:]).hexdigest()


def _statement(d: Path, good: str, text: str) -> subprocess.CompletedProcess:
    """`flock-verify statement` on ``text``, with the public file staged for ``good`` rebound to it (its digest and class)."""
    g, b = good.encode(), text.encode()
    pub = (d / "pub.bin").read_bytes().replace(hashlib.sha512(g).hexdigest().encode(), hashlib.sha512(b).hexdigest().encode())
    (d / "c.txt").write_bytes(b)
    (d / "p.bin").write_bytes(pub.replace(_class(g).encode(), _class(b).encode()))
    return subprocess.run([str(LEAN / ".lake/build/bin/flock-verify"), "statement", "--statement", "verity/flock-circuit",
                           "--circuit", str(d / "c.txt"), "--public", str(d / "p.bin")], capture_output=True, text=True)


def _with_out_group(text: str, col: int, words: int) -> str:
    """The circuit with its unit's one output group moved to word ``col``, ``words`` wide (the header's last group)."""
    lines = text.split("\n")
    at = next(i for i, ln in enumerate(lines) if ln.startswith("flock-ir-unit/v2 unit-class/toy "))
    h = lines[at].split()
    assert h[-5:] == ["1", "9", "1", "1", "16"], h[-5:]
    lines[at] = " ".join(h[:-4] + [str(col), str(words), "1", "16"])
    return "\n".join(lines)


@pytest.mark.slow
def test_an_output_group_past_its_units_slot_is_refused(tmp_path):
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    text = _bit_row_toy([1024], tmp_path)
    p = _statement(tmp_path, text, text)
    assert p.returncode == 0, p.stdout + p.stderr
    for col, words in ((16, 1), (31, 1), (16, 16), (0, 32)):
        p = _statement(tmp_path, text, _with_out_group(text, col, words))
        assert p.returncode != 0, (col, words, p.stdout)
        assert f"output group at word {col} ({words} words) overruns its unit's 16 words (D11)" in p.stdout + p.stderr, \
            (col, words, p.stdout + p.stderr)
    p = _statement(tmp_path, text, _with_out_group(text, 15, 1))
    assert p.returncode == 0, p.stdout + p.stderr
