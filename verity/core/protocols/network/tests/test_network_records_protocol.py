"""The audit over every published record (`NetworkCheck.Records`, `network-records serve` and `network-records verdicts`)
against `audit.py`, through the difftest harness.

The cases are `test_network_check_protocol`'s generated spatial units, their records published as one kind
(`KINDS`): all raw (`grid.Record`), all committed (`commitment.commit`), all certified, or mixed, each record's kind drawn
and the changed record's the kind its change needs. Each kind gets the changes that apply to it (`EXPECT`), and the
record-kind negative controls: a raw grid slot changed, a raw record relabelled to another link, a committed digest
changed, and a committed status in the clear that disagrees with its digest either way. The reference decodes the
commitment with its own decoder (`records`, from `NetworkCheck/Records.lean`'s module docstring), not with
`lean_audit.encode`, which writes the cases, and runs the real `Audit`."""

from __future__ import annotations

import contextlib
import dataclasses
import shutil
from collections import Counter

import pytest
from hypothesis import strategies as st

import test_network_check_protocol as V2
import verity_difftest as D
from verity_difftest import protocol as P
from verity.ml.commitments.identity import tagged_sha512
from verity.core.protocols.certifier.audit import Audit
from verity.core.protocols.certifier.certificate import CertifiedRecord
from verity.core.protocols.certifier.commitment import CommittedRecord, commit, record_digest
from verity.core.protocols.certifier.grid import COMPLETE, Reason, Record, Status
from verity.core.protocols.certifier.lean_audit import encode
from verity.core.protocols.certifier.schedule import expected_grids

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

ID = "verity/network-records/v2"
COMMITMENT_TAG = "verity-network-certifier/records/v1"
KINDS = ("raw", "committed", "certified", "mixed")

# --- the reference --------------------------------------------------------------------------------------------------------


def records(b: bytes) -> tuple[list, bytes]:
    """The published records, `u8 kind ‖ u64be len ‖ body` one after another, decoded exactly."""
    out, pos = [], 0

    def reader(data: bytes):
        at = 0

        def take(n):
            nonlocal at
            if at + n > len(data):
                raise D.Reject("truncated")
            at += n
            return data[at - n:at]

        def be(n):
            return int.from_bytes(take(n), "big")

        def text(what):
            try:
                return take(be(2)).decode("utf-8")
            except UnicodeDecodeError:
                raise D.Reject(f"{what} is not UTF-8") from None

        def status():
            code = be(1)
            if code not in V2.REASONS:
                raise D.Reject(f"status code {code}")
            return Status(V2.REASONS[code])

        return take, be, text, status, lambda: at

    take, be, _, _, at = reader(b)
    while at() < len(b):
        kind, body = be(1), take(be(8))
        btake, bbe, btext, bstatus, bat = reader(body)
        if kind == 0:
            link, window, status = btext("a link"), bbe(8), bstatus()
            grid = []
            for _ in range(bbe(4)):
                row = []
                for _ in range(bbe(4)):
                    t = bbe(1)
                    if t not in (0, 1):
                        raise D.Reject(f"slot tag {t}")
                    row.append(btake(bbe(4)) if t else None)
                grid.append(tuple(row))
            out.append(Record(link, window, tuple(grid), status))
        elif kind == 1:
            link, window, status = btext("a link"), bbe(8), bstatus()
            out.append(CommittedRecord(link, window, btext("a digest"), status))
        elif kind == 2:
            rs, _ = V2.commitment(body)
            if len(rs) != 1:
                raise D.Reject("a certified body holds one record")
            out.append(rs[0])
            continue
        else:
            raise D.Reject(f"record kind {kind}")
        if bat() != len(body):
            raise D.Reject("bytes left in a record")
    return out, b


def reference(case, check=V2.check) -> dict:
    return P.answer(ID, case, params=V2.params, commitment=records, evidence=V2.evidence, check=check,
                    digest=lambda c: tagged_sha512(COMMITMENT_TAG, c[1]))


def all_verdicts(u, c, e) -> tuple:
    """Every link-window's verdict as `network-records verdicts` writes it, after v2's evidence refusals."""
    out = V2.check(u, c, e)
    if out[0] == "reject" and out[1].startswith("evidence: "):
        return out
    audit = Audit()
    vs = [("egress", v) for link, (frames, syncs) in zip(u.egress, e)
          for v in audit.check_link(link, c[0], frames, syncs, u.sync_set, u.grid, u.key, u.session)]
    vs += [("ingress", v) for link in u.ingress
           for v in audit.check_ingress_link(link, c[0], u.windows, u.key, u.ingress_grid, u.session)]
    return "accept", {}, [{"direction": d, "link": v.link, "verdict": v.violation or "accept", "window": v.window}
                          for d, v in vs]


def verdicts_reference(case) -> dict:
    out = reference(case, all_verdicts)
    if out["verdict"] != "accept":
        raise D.Reject(out["reason"])
    return {"link_windows": out["profile"]}


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
SEEN: Counter = Counter()
#: Cases per kind and change in the full run.
EXAMPLES = 30


def _as(kind: str, r: CertifiedRecord, d: V2.Drawn):
    raw, g = d.raws[(r.direction.value, r.link, r.window)]
    return r if kind == "certified" else raw if kind == "raw" else commit(raw, g)


def _bad_records(c: bytes, n: int) -> list[bytes]:
    """Commitments `records` refuses: truncated, extended, a record of an unknown kind, a body cut short or with a byte
    left over, a slot tag, a status code or a link out of range, a certified body of no record."""
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
    # in a mixed unit every record of the changed link-window, either direction, has the kind the change needs: a raw or
    # committed record of the other direction would be a duplicate of it
    published = [_as(NEEDS[m] if kind == "mixed" and m in NEEDS and lw[1:] == aim[1:] else
                     draw(st.sampled_from(KINDS[:3])) if kind == "mixed" else kind, r, d) for r, lw in zip(d.records, lws)]
    at = next((i for i, lw in enumerate(lws) if lw == aim), None) if m in RAW or m in COMMITTED else None
    if at is not None:
        r, n = published[at], d.n
        if m == "raw-slot" and isinstance(r, Record):
            rows = [list(row) for row in r.grid]
            t, s = n % len(rows), n % len(rows[0])
            f = rows[t][s]
            frame = next((x for row in r.grid for x in row if x is not None), b"\0")
            rows[t][s] = draw(st.sampled_from([None, V2._flip(f, n)] if f is not None else [frame, b""]))
            grid = draw(st.sampled_from([tuple(map(tuple, rows)), r.grid[:-1], r.grid + (r.grid[0],),
                                         (r.grid[0] + (None,),) + r.grid[1:]]))
            published[at] = dataclasses.replace(r, grid=grid)
        elif m == "raw-other-link" and isinstance(r, Record):
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


def counted(case) -> dict:
    out = reference(case)
    kind, m = TAGS.get(D.canonical(case), ("?", "?"))
    SEEN[(kind, m, V2._kind(out))] += 1
    return out


PORT = D.ProtocolPort("network-records", counted, record_cases(), package=V2.LEAN)


@dataclasses.dataclass(frozen=True)
class VerdictsPort(D.Port):
    """`network-records verdicts`, a `Difftest.serve` port behind the protocol's executable."""

    def argv(self) -> list[str]:
        return [*super().argv(), "verdicts"]


VERDICTS = VerdictsPort("network-records", "verdicts", verdicts_reference, record_cases(), package=V2.LEAN)
server = D.server_fixture(PORT)
verdicts_server = D.server_fixture(VERDICTS)


def _table() -> Counter:
    by_path = Counter()
    for (kind, m, path), k in SEEN.items():
        by_path[(kind, m, path)] += k
    print("\n" + "\n".join(f"{kind:10} {m:17} {path:12} {k}" for (kind, m, path), k in sorted(by_path.items())))
    return by_path


@pytest.mark.slow
def test_the_records_check_agrees_with_the_audit(server):
    """Lean and the audit agree on every case, and each kind reaches every path that applies to it: each change gives its
    own verdict at least once on raw, committed, certified and mixed records."""
    SEEN.clear()
    for kind in KINDS:
        for m in EXPECT[kind]:
            # one run per kind and change, since `sampled_from`'s derandomized draws skip some of 88 pairs in thousands;
            # a run of one change gives one verdict, which `run` calls vacuous only after comparing every case
            with contextlib.suppress(D.Vacuous):
                D.run(dataclasses.replace(PORT, inputs=record_cases((kind,), (m,))),
                      examples=EXAMPLES * (2 if m in ("late", "frames") else 1), server=server)
    by_path = _table()
    missed = [(kind, m) for kind in KINDS for m, path in EXPECT[kind].items() if not by_path[(kind, m, path)]]
    assert not missed, missed
    verdicts = Counter()
    for (_, _, path), k in by_path.items():
        verdicts[path] += k
    assert min(verdicts[v] for v in ("accept", "no-verdict", "mismatch", "signature")) > 50, verdicts


@pytest.mark.slow
def test_every_verdict_is_the_audits(verdicts_server):
    """`network-records verdicts` gives every link-window's verdict, not only the first violation, as the audit does."""
    stats = D.run(VERDICTS, examples=1200, seed=1, server=verdicts_server)
    assert stats.outputs > 600 and stats.rejects > 50, stats


def test_each_kind_gives_the_audits_verdicts(server, verdicts_server):
    """A short run per kind over an honest unit, a missing record and a mismatch."""
    for kind in KINDS:
        only = ("honest", "missing", "frames")
        port = dataclasses.replace(PORT, inputs=record_cases((kind,), only))
        stats = D.run(port, examples=20, server=server)
        assert all(stats.counts.get(v) for v in ("accept", "reject", "undecided")), (kind, stats)
        D.run(dataclasses.replace(VERDICTS, inputs=record_cases((kind,), only), need_rejects=False), examples=10,
              server=verdicts_server)


@pytest.mark.slow
def test_a_reference_wrong_on_one_path_is_caught(server):
    """A reference that never checks a committed record's digest (it publishes the Program's in its place) differs."""
    def ignore_digests(u, c, e):
        recs = list(c[0])
        if len(e) == len(u.egress) and all(len(s) == u.windows for _, s in e):
            for link, (frames, syncs) in zip(u.egress, e):
                expected = expected_grids(frames, syncs, u.grid)
                recs = [dataclasses.replace(r, digest=record_digest(link, r.window, expected[r.window], COMPLETE, u.grid))
                        if isinstance(r, CommittedRecord) and r.link == link and r.window < u.windows else r for r in recs]
        return V2.check(u, (recs, c[1]), e)

    port = dataclasses.replace(PORT, reference=lambda case: reference(case, ignore_digests),
                               inputs=record_cases(("committed",), ("honest", "committed-digest", "status-digest")))
    with pytest.raises(D.Mismatch) as x:
        D.run(port, examples=400, server=server)
    assert (x.value.lean[0], x.value.python[0]) == ("reject", "accept")


@pytest.mark.parametrize("kind", KINDS[:3])
def test_the_encoding_round_trips(kind):
    """`records` reads back what `lean_audit.encode` writes, record for record, for every kind."""
    from hypothesis import given, settings

    @settings(derandomize=True, database=None, deadline=None, max_examples=30)
    @given(V2.units(("honest", "status", "extra")))
    def roundtrip(d):
        published = [_as(kind, r, d) for r in d.records]
        assert records(b"".join(encode(r) for r in published))[0] == published

    roundtrip()
