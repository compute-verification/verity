"""The coin server's record in its own format, held to Lean's total reader (`Flock.RecRecord`, `flock-verify
rec-record`), which `Proofs.Flock.Claims.Recursion.RecRecord` proves reads back what `recordW` writes. The record's writer is Python
(`rec_live.record_bytes`), so these tests are what ties the proof to the deployed record: on the honest records of the
plain session and of 2d61's candidate-3 session, and on one with every escape a stream name can need, `recordW` writes
`record_bytes`' bytes up to the rounds' end, and `readRecord` reads each round as Python holds it. A record in another
layout (compact, or its rounds not first) is refused."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from test_rec_outer import BIN
from verity_flock import rec_live as RL

HERE = Path(__file__).resolve().parent
C3 = HERE / "fixtures" / "rec-c3-inner"
PLAIN = HERE / "fixtures" / "rec-coins" / "16-16-8-n64" / "inner-honest"

pytestmark = pytest.mark.skipif(not BIN.exists(), reason="the Lean verifier is not built")

ROUND_KEYS = ["stream", "k", "tops", "level0", "commits", "n", "coins"]


def _record(d: Path) -> dict:
    p = d / "public.json"
    assert p.is_file(), f"{d} is not fetched: `uv run research data fetch-fixtures {d.relative_to(HERE.parents[3])}`"
    return json.loads(p.read_text())


def _read(tmp_path: Path, text: str) -> tuple[int, dict | None, str]:
    f = tmp_path / "public.json"
    f.write_text(text)
    p = subprocess.run([str(BIN), "rec-record", str(f)], capture_output=True, text=True)
    out = p.stdout.strip()
    return p.returncode, json.loads(out.removeprefix("RECRECORD ")) if out.startswith("RECRECORD ") else None, p.stderr


def _escapes() -> dict:
    r = _record(PLAIN)
    r["rounds"] = [{**r["rounds"][0], "stream": 'a"b\\c/d'}] + r["rounds"][1:3]
    return r


@pytest.mark.parametrize("record", [lambda: _record(PLAIN), lambda: _record(C3), _escapes],
                         ids=["plain", "c3", "escapes"])
def test_the_lean_writer_writes_the_coin_servers_record_and_its_reader_reads_it(tmp_path, record):
    """`recordW` at the rounds `readRecord` read is `record_bytes`' bytes up to the rounds' end (``writes``), and each
    round reads as the record holds it: stream, index, tops, level-0 count, commit strings, coin count and coins."""
    r = record()
    rc, out, err = _read(tmp_path, RL.record_bytes(r))
    assert rc == 0, err
    assert out["writes"] is True
    assert out["rounds"] == [{k: x[k] for k in ROUND_KEYS} for x in r["rounds"]]


@pytest.mark.parametrize("dump", [lambda r: json.dumps(r), lambda r: json.dumps(r, indent=1, sort_keys=True)],
                         ids=["compact", "rounds-not-first"])
def test_a_record_in_another_layout_is_refused(tmp_path, dump):
    rc, out, err = _read(tmp_path, dump(_record(PLAIN)))
    assert (rc, out, err.strip()) == (1, None, "rec-record: not the coin server's record format")
