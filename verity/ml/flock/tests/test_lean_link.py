"""The firewall's level-0 recheck in Lean (`Flock/Firewall/Link.lean`, `flock-firewall link`), the root a firewalled table's
`Commit` carries (`flock-circuit`'s `Firewall::new`): its protocol's refusals. A request is `table M LEN` and `LEN` bytes, sized
by level 0 of `M`'s `--zk` schedule; a request of another size, an `M` without a schedule or a malformed line is answered
`err WHY` and ends the process with exit 1, and the end of input ends it with exit 0. The root itself, at `m = 25` and up, is
`live/src/firewall.rs`'s `lean_link_agrees_with_commit_zk`'s, held to `commit_zk`'s."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-firewall"

#: level 0 of m = 25's --zk schedule: 2^18 witness words, four extra lanes of 2^12, 68 lanes' 554 paddings, 2^13 salts
M25 = 16 * 2**18 + 16 * 4 * 2**12 + 16 * 68 * 554 + 192 * 2**13


@pytest.fixture(scope="module", autouse=True)
def built():
    if shutil.which("lake") is None:
        pytest.skip("no Lean toolchain")
    subprocess.run(["lake", "build", "flock-firewall"], cwd=LEAN, check=True, capture_output=True)


def _link(stdin: bytes) -> tuple[str, int]:
    p = subprocess.run([str(BIN), "link"], input=stdin, capture_output=True)
    return p.stdout.decode(errors="replace"), p.returncode


def test_end_of_input_ends_the_session():
    assert _link(b"") == ("", 0)


@pytest.mark.parametrize("line", [b"table 25\n", b"table x 3\n", b"tables 25 3\n", b"table 25 3 4\n"])
def test_malformed_line(line):
    out, rc = _link(line)
    assert rc == 1 and out.startswith("err a request `")


def test_no_schedule_below_25():
    out, rc = _link(b"table 24 16\n" + bytes(16))
    assert rc == 1 and out == "err m = 24: --zk needs m >= 25 (Ligerito level 0's padding outgrows its lanes at rate 1/2)\n"


def test_wrong_size():
    out, rc = _link(f"table 25 {M25 - 16}\n".encode() + bytes(M25 - 16))
    assert rc == 1 and out == f"err link: a {M25 - 16}-byte table where m = 25 takes {M25}\n"


def test_short_input():
    out, rc = _link(f"table 25 {M25}\n".encode() + bytes(100))
    assert rc != 0 and not out.startswith("ok")
