"""The records check over every published record (`NetworkCheck.Records`, `network-records serve`, and `network-records
verdicts` with the grid compared and without it) on `test_network_check_protocol`'s generated spatial units, their records
published as one kind (`KINDS`): all raw (`grid.BlockGrid`), all committed (`commitment.commit`), all attested, or mixed,
each record's kind sampled and the changed record's the kind its change needs. The Lean check gives each change its own
verdict (`EXPECT`), the record-kind changes among them: a raw grid slot changed, a raw record relabelled to another link,
a committed digest changed, and a committed status in the clear that disagrees with its digest either way."""

from __future__ import annotations

import dataclasses
import json
import shutil
from collections import Counter

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import test_network_check_protocol as V2
import verity_difftest as D
from verity.protocols.accounting.network.verifier.certificate import CertifiedRecord
from verity.protocols.accounting.network.verifier.commitment import CommittedRecord, commit, record_digest
from verity.protocols.accounting.network.verifier.grid import Reason, BlockGrid, Status
from verity.protocols.accounting.network.verifier.lean_audit import encode

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

KINDS = ("raw", "committed", "certified", "mixed")

# --- the generator --------------------------------------------------------------------------------------------------------

COMMON = {"honest": "accept", "shuffled": "accept", "extra": "accept", "missing": "no-verdict", "refused": "no-verdict",
          "status": "status", "duplicate": "duplicate", "advice-outside": "advice", "advice-unit": "advice",
          "late": "mismatch", "frames": "mismatch",
          "params": "params", "commitment": "commitment", "evidence": "evidence", "case": "case"}
RAW = {"raw-slot": "mismatch", "raw-other-link": "no-verdict"}
COMMITTED = {"committed-digest": "mismatch", "status-digest": "mismatch", "status-clear": "status"}
EXPECT = {"raw": COMMON | RAW, "committed": COMMON | COMMITTED, "certified": V2.EXPECT,
          "mixed": V2.EXPECT | RAW | COMMITTED}
#: The kind the changed record must have in a mixed unit.
NEEDS = {m: "certified" for m in V2.EXPECT if m not in COMMON} | {m: "raw" for m in RAW} | {m: "committed" for m in COMMITTED}
EGRESS_ONLY = ("raw-slot", "committed-digest", "status-digest", "status-clear")
TAGS: dict[str, tuple[str, str]] = {}


def _as(kind: str, r: CertifiedRecord, d: V2.Sampled):
    raw, g = d.raws[(r.direction.value, r.link, r.window)]
    return r if kind == "certified" else raw if kind == "raw" else commit(raw, g)


def _bad_records(c: bytes, n: int) -> list[bytes]:
    """Commitments `records` refuses: truncated, extended, a record of an unknown kind, a body cut short or with a byte
    left over, a slot tag, a status code or a link out of range, an attested body of no record."""
    def framed(kind, body):
        return bytes([kind]) + len(body).to_bytes(8, "big") + body

    def raw(link=b"a", status=0, slot=b"\0", extra=b""):
        return framed(0, len(link).to_bytes(2, "big") + link + bytes(8) + bytes([status]) + (1).to_bytes(4, "big") +
                      (1).to_bytes(4, "big") + slot + extra)

    out = [c + bytes([n % 256]), c + bytes(9), c + framed(3, b""), c + raw(extra=b"\0"), c + raw(slot=b"\2"),
           c + raw(status=7), c + raw(link=b"\xff"), c + raw()[:-1], c + raw()[:5], c + framed(2, b""),
           c + framed(1, b"\0\1a" + bytes(8) + b"\0\0\1\xff")]
    if c:
        out.append(c[:n % len(c)])
    return out


@st.composite
def record_cases(draw, kinds: tuple[str, ...] = KINDS, only: tuple[str, ...] | None = None) -> dict:
    kind = draw(st.sampled_from(kinds))
    m = draw(st.sampled_from([x for x in EXPECT[kind] if only is None or x in only]))
    d = draw(V2.units((m if m in V2.EXPECT else "honest",), need_egress=m in EGRESS_ONLY))
    aim = ("egress", *d.egress_target) if m in EGRESS_ONLY and d.egress_target else d.target
    lws = [(r.direction.value, r.link, r.window) for r in d.records]
    # in a mixed unit every record of the changed execution phase, either direction, has the kind the change needs: a
    # raw or committed record of the other direction would be a duplicate of it
    published = [_as(NEEDS[m] if kind == "mixed" and m in NEEDS and lw[1:] == aim[1:] else
                     draw(st.sampled_from(KINDS[:3])) if kind == "mixed" else kind, r, d) for r, lw in zip(d.records, lws)]
    at = next((i for i, lw in enumerate(lws) if lw == aim), None) if m in RAW or m in COMMITTED else None
    if at is not None:
        r, n = published[at], d.n
        if m == "raw-slot" and isinstance(r, BlockGrid):
            rows = [list(row) for row in r.grid]
            t, s = n % len(rows), n % len(rows[0])
            f = rows[t][s]
            frame = next((x for row in r.grid for x in row if x is not None), b"\0")
            rows[t][s] = draw(st.sampled_from([None, V2._flip(f, n)] if f is not None else [frame, b""]))
            grid = draw(st.sampled_from([tuple(map(tuple, rows)), r.grid[:-1], r.grid + (r.grid[0],),
                                         (r.grid[0] + (None,),) + r.grid[1:]]))
            published[at] = dataclasses.replace(r, grid=grid)
        elif m == "raw-other-link" and isinstance(r, BlockGrid):
            published[at] = dataclasses.replace(r, link=draw(st.sampled_from(["zz", *d.egress, *d.params["ingress"]["links"]])))
        elif isinstance(r, CommittedRecord):
            raw, g = d.raws[aim]
            reason = draw(st.sampled_from(list(Reason)))
            if m == "committed-digest":
                published[at] = dataclasses.replace(r, digest=draw(st.sampled_from(
                    [V2._flip(r.digest.encode(), n).decode(), r.digest.upper(), r.digest[:-1], ""])))
            elif m == "status-digest" and r.status.complete:
                published[at] = dataclasses.replace(r, digest=record_digest(r.link, r.window, raw.grid, Status(reason), g))
            elif m == "status-clear" and r.status.complete:
                published[at] = dataclasses.replace(r, status=Status(reason))
    c = b"".join(encode(r) for r in published)
    case = V2.malformed(draw, m, d.n, d.params, c, V2.evidence_bytes(d), lambda: _bad_records(c, d.n))
    TAGS[D.canonical(case)] = (kind, m)
    return case


@pytest.fixture(scope="module")
def server():
    with D.Server([str(D.build(V2.LEAN, "network-records")), "serve"]) as s:
        yield s


@pytest.fixture(scope="module")
def verdicts_server():
    with D.Server([str(D.build(V2.LEAN, "network-records")), "verdicts"]) as s:
        yield s


@pytest.mark.parametrize("m", list(EXPECT["mixed"]))
def test_each_change_gets_its_verdict(server, m):
    """On mixed units, the Lean check gives each change its own verdict on some generated unit (a change can miss, as a
    late frame that still fits its bucket)."""
    seen = V2.lean_paths(server, record_cases(("mixed",), (m,)), 80 if m in ("late", "frames") else 30)
    assert seen[EXPECT["mixed"][m]], seen


@pytest.mark.slow
def test_without_the_grid_only_a_grid_mismatch_is_accepted(verdicts_server):
    """Both entries on the same cases, every kind and change: what `verdicts-no-grid` answers, `verdicts` answers too,
    with the same execution phases in order and the same verdicts, except that each egress `mismatch` of `verdicts` is an
    accept (an ingress `mismatch` stays); where `verdicts` answers and `verdicts-no-grid` refuses, the unit's records
    aren't all attested. Some units are accepted whole by `verdicts-no-grid` where `verdicts` gives a `mismatch`: only
    their grid differs from the Program's."""
    seen = Counter()

    @settings(derandomize=True, database=None, deadline=None, max_examples=300)
    @given(st.one_of(record_cases(("certified",)), record_cases(("certified",), ("late", "frames")),
                     record_cases(("certified",), ("status", "advice-outside", "advice-unit", "duplicate")), record_cases()))
    def both(case):
        full, no_grid = (verdicts_server.call(fn, case) for fn in ("verdicts", "verdicts-no-grid"))
        if no_grid[0] == "reject":
            if full[0] == "output":
                assert TAGS[D.canonical(case)][0] != "certified", case
                seen["refused-uncertified"] += 1
            return
        assert full[0] == "output", (full, no_grid)
        a, b = (json.loads(x[1])["link_windows"] for x in (full, no_grid))
        assert [(v["direction"], v["link"], v["window"]) for v in a] == \
            [(v["direction"], v["link"], v["window"]) for v in b]
        for x, y in zip(a, b):
            if x["direction"] == "egress" and x["verdict"] == "mismatch":
                assert y["verdict"] == "accept", (x, y)
                seen["mismatch-accepted"] += 1
            else:
                assert y == x
                seen[x["verdict"].split(":")[0]] += 1
        if any(x["verdict"] == "mismatch" for x in a) and all(y["verdict"] == "accept" for y in b):
            seen["grid-only-unit"] += 1

    both()
    print(dict(seen))
    assert min(seen[k] for k in ("refused-uncertified", "accept", "no-verdict", "signature", "advice", "status",
                                 "duplicate")) > 0, seen
    assert seen["mismatch-accepted"] > 20 and seen["grid-only-unit"] > 20, seen
