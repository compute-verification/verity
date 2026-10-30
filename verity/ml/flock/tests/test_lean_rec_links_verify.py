"""`flock-verify verify --zk` and `statement` at a recursive statement carrying the inner population's link table
(`--rec-links ROOT:N`), on the fetched fixture `rec-links/k64-n4`: pod 85's run with REC_LINKS=1 at K = 64, N = 4 (its
inner statement N instances of GemmCoordinate_v2{K} wired in a chain, each cut wire a link). It holds the inner statement's
files and the inner session V* was staged over (``inner/``), V*'s inputs as oprove fixed them before any session
(``vstar/``: the inner shape naming the table, the shared rows), and V*'s link part (``links/``) and its forged row
(``forged-link/``: a table of its own whose first row names a writer whose output is not its reader's row), each with
its statement, the verifier's bindings, its index in V*'s list, its table and its session: the honest part's proved
behind the ZK server, the forgery's by a prover that sends its proofs though the unit's outputs are not the rows the
verifier registered.

`verify` accepts the honest part and refuses the forged row, by its proof: setup passes the forgery at its own table, as
`statement` (setup alone) does. Both refuse, at setup and with one reason, the honest part at another root, at another
row count, without the table, and the forgery at the honest table. What the #guards of `Flock/Test/RecLinks.lean` and
`Flock/Test/RecLinkPart.lean` can't catch: the binary's own reading of the table's flag and of the files a link part's
check reads, on a proved session."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from verity_flock import rec_links as RL

HERE = Path(__file__).resolve().parent
BIN = HERE.parents[3] / "verity/core/service/flock/.lake/build/bin/flock-verify"
SET = HERE / "fixtures" / "rec-links" / "k64-n4"

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not BIN.exists() or shutil.which("lake") is None, reason="the Lean verifier is not built")]


@pytest.fixture(scope="module", autouse=True)
def fetched() -> None:
    assert (SET / "links" / "circuit.txt").is_file(), (
        f"{SET} is not fetched: `uv run research data fetch-fixtures verity/ml/flock/tests/fixtures/rec-links`")


def _inner() -> list[str]:
    """V*'s inputs and the inner statement's files, as pod 85's lean step gives every V* part."""
    i, v = SET / "inner", SET / "vstar"
    return ["--rec-inner", str(v / "rec-inner.json"), "--rec-rows", str(v / "rows.json"),
            "--rec-inner-circuit", str(i / "circuit.txt"), "--rec-inner-public", str(i / "pub.bin"),
            "--rec-inner-partition", str(i / "partition.json"), "--rec-inner-program", str(i / "program.json"),
            "--rec-inner-statement", "verity/flock-circuit+plain-leaves", "--rec-inner-statement-coins", "os",
            "--rec-inner-draw", str(i / "draw.json"), "--rec-inner-session", str(i / "session"),
            "--rec-inner-transcript", str(i / "public.json")]


def _part(name: str) -> list[str]:
    p = SET / name
    return ["--statement", "verity/flock-circuit", "--circuit", str(p / "circuit.txt"), "--public", str(p / "pub.bin"),
            "--partition", str(p / "partition.json"), "--program", str(p / "program.json"),
            "--registered", str(p / "registered.json"), "--rec-part", (p / "part.txt").read_text().strip(),
            "--rec-bindings", str(p / "bindings.json"), *_inner()]


def table(name: str) -> tuple[str, int]:
    """The part's table as `--rec-links` takes it: its root in hex and its row count."""
    root, n = (SET / name / "rec-links.txt").read_text().strip().split(":")
    return root, int(n)


def _links(t: tuple[str, int] | None) -> list[str]:
    return [] if t is None else ["--rec-links", f"{t[0]}:{t[1]}"]


def verify(name: str, t: tuple[str, int] | None) -> tuple[int, dict]:
    p = subprocess.run([str(BIN), "verify", "--zk", *_part(name), *_links(t), "--session", str(SET / name / "session")],
                       capture_output=True, text=True)
    (v,) = [json.loads(ln[8:]) for ln in p.stdout.splitlines() if ln.startswith("VERDICT ")]
    return p.returncode, v


def statement(name: str, t: tuple[str, int] | None) -> tuple[int, str]:
    p = subprocess.run([str(BIN), "statement", *_part(name), *_links(t)], capture_output=True, text=True)
    return p.returncode, p.stdout.strip().splitlines()[-1]


def test_the_honest_link_part_is_accepted():
    rc, v = verify("links", table("links"))
    assert (rc, v["accepted"], v["why"]) == (0, True, ""), v
    rc, line = statement("links", table("links"))
    assert rc == 0 and json.loads(line)["statement"] == "verity/flock-circuit", line


def test_the_forged_row_is_refused_by_its_proof():
    """The forgery's table is its own (another root), registered as the link part reads it: setup passes it, in `verify`
    as in `statement`, and its proof does not hold."""
    assert table("forged-link") != table("links")
    rc, v = verify("forged-link", table("forged-link"))
    assert rc == 1 and not v["accepted"] and v["why"] and not v["why"].startswith("setup:"), v
    rc, line = statement("forged-link", table("forged-link"))
    assert rc == 0, line


def _other_root(t: tuple[str, int]) -> tuple[str, int]:
    return f"{int(t[0][0], 16) ^ 1:x}{t[0][1:]}", t[1]


def _other_rows(t: tuple[str, int]) -> tuple[str, int]:
    """The next row count that is a table's (`rec_links.grid`), so the flag parses and setup refuses it."""
    return t[0], next(n for n in range(t[1] + 1, 2 * t[1] + 2) if RL.grid(n) == n)


@pytest.mark.parametrize("name,links,why", [
    ("links", lambda: _other_root(table("links")), "not the statement's table"),
    ("links", lambda: _other_rows(table("links")), "rows (--rec-links), and the verifier's inner shape's"),
    ("links", lambda: None, "and the statement carries none (--rec-links ROOT:N)"),
    ("forged-link", lambda: table("links"), "not the statement's table"),
], ids=["other-root", "other-rows", "no-table", "forgery-at-the-honest-table"])
def test_verify_and_statement_refuse_another_table_at_setup_alike(name, links, why):
    t = links()
    rc, v = verify(name, t)
    src, line = statement(name, t)
    assert (rc, v["accepted"], src) == (1, False, 1), (v, line)
    assert v["why"] == line and line.startswith("setup: rec-links") and why in line, (v["why"], line)
