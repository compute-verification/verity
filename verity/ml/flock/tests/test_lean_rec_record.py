"""The coin server's record in its own format, held to Lean's total reader (`Flock.RecRecord`, `flock-verify
rec-record`), which `Proofs.Flock.Claims.Recursion.RecRecord` proves reads back what `recordW` writes. The record's writer is Python
(`rec_live.record_bytes`), so these tests are what ties the proof to the deployed record: on the honest records of the
plain session and of 2d61's candidate-3 session, and on one with every escape a stream name can need, `recordW` writes
`record_bytes`' bytes up to the rounds' end, and `readRecord` reads each round as Python holds it. A record in another
layout (compact, or its rounds not first) is refused.

The proofs' transcript model (`RecDeploy.recTr`, read by `recRead`) is the record read on past its rounds to the link
points (`readFull`, written by `recordFullW`), and the inner rounds before the last as the record shows them
(`Record.msgs`). These tests hold that model to the coin server's records: `recordFullW` writes `record_bytes`' bytes up
to the link points, the rounds shown are the link's level-0 tops and points, then each recorded round's tops and
`c_k` with its coins, and at the plain session they are the inner verifier's rounds less the last, each as many
128-byte commit strings as the schedule fixes (`RecDeploy.nAtSched`)."""
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


def _shown(r: dict) -> list[dict]:
    """The inner rounds before the last as the coin server's record holds them: the link's level-0 tops (each table's
    nodes, in the record's order) joined, with the link points; then each round's tops' nodes and ``c_k`` joined, with
    its coins."""
    link = r["link"]
    return ([{"msg": "".join(n for nodes in link["roots"].values() for n in nodes), "coins": link["points"]}] +
            [{"msg": "".join(n for top in x["tops"] for n in top) + "".join(x["commits"]), "coins": x["coins"]}
             for x in r["rounds"]])


@pytest.mark.parametrize("record", [lambda: _record(PLAIN), lambda: _record(C3)], ids=["plain", "c3"])
def test_the_transcript_model_shows_the_coin_servers_rounds(tmp_path, record):
    """`recordFullW` at the record `readFull` read is `record_bytes`' bytes up to the link points, and the rounds it shows
    are the coin server's: round 0 from the link, each later one from its recorded round."""
    r = record()
    rc, out, err = _read(tmp_path, RL.record_bytes(r))
    assert rc == 0, err
    assert out["full"]["writes"] is True
    assert out["full"]["msgs"] == _shown(r)


def test_the_transcript_model_has_the_inner_verifiers_rounds_before_the_last(tmp_path):
    """At the plain session's inner shape (m 27, k_log 22, 8 claims; `Flock.Test.RecSchedule`), the inner verifier has
    `2 + rounds` rounds, the two reps' rounds between the link and the last. The model shows all but the last: round 0
    the level-0 top's `2^cap₀` nodes (`level0Nodes`), each later round its tops' nodes and one `c_k` where it absorbs a
    message (`nAtSched`), each 128 bytes, with the schedule's coin count."""
    r = _record(PLAIN)
    rc, out, err = _read(tmp_path, RL.record_bytes(r))
    assert rc == 0, err
    msgs = out["full"]["msgs"]
    rep = RL.rep_rounds(27, 22, 8)
    sched = rep + rep
    assert len(msgs) == (2 + len(sched)) - 1
    assert len(msgs[0]["msg"]) == 2 * 128 * (1 << RL.fast100(27).levels[0].cap)
    assert len(msgs[0]["coins"]) == r["hello"]["link"]["m"] * r["hello"]["link"]["points"]
    for m, (n, tops, _level0, e) in zip(msgs[1:], sched):
        assert (len(m["msg"]), len(m["coins"])) == (2 * 128 * (sum(tops) + int(e > 0)), n)


def test_a_hello_in_another_order_is_not_read_in_full(tmp_path):
    """The record's Hello is read in its fixed keys and order: one in another order reads to its rounds, not past them."""
    r = _record(PLAIN)
    r["hello"] = dict(reversed(list(r["hello"].items())))
    rc, out, err = _read(tmp_path, RL.record_bytes(r))
    assert rc == 0, err
    assert out["writes"] is True and out["full"] is None
