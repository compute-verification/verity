"""PoUB's window decided from a span (`Poub.Span`, `Session.verifySpan`), at 16 sketch rows on a p512 store whose
passes read it whole (the identity plan), against `poub-difftest`.

The verifier's own copy of each sample's two public rows, and of the supplied values it gives the audit (`span_public`),
is held to the Python the prover's window commits them by (`span.public_rows`, `block_check.opening_public_v2`'s, and
`span.supplied`), at an unmasked and a masked codec. The rest is the Lean's alone, which the Python never decides: a
window decided on public files taken as accepted (`span_held`), each built here with the honest unit's `o0` row committed
as the proof system commits it (`opening.commit`), is accepted on the honest span; is rejected at the sample whose unit
says it failed; and is rejected `malformed` on a span of units in another order or not the population's from 0, of a
unit missing, with a verdict word or its salt changed, or of a pass whose plan is not the identity. That the units'
`pub` and `signs` rows are the window's is the audit's R7 against the supplied values, which only `span` reaches: it
rejects `malformed` a span of those same files when the proof system's audit doesn't accept its statements, here
statements that are no audit's at all. No staged statement of the opening unit exists for an audit to accept, so no
test here reaches `span`'s acceptance."""

from __future__ import annotations

import hashlib
import json
import shutil
from typing import Any

import pytest
import ra_cases as M
import verity_difftest as D
from hypothesis import strategies as st

from verity.protocols.resource_accounting.memory.poub.verifier import audit, receipts, span
from verity.protocols.resource_accounting.memory.poub.verifier import opening as O
from verity.protocols.resource_accounting.memory.poub.verifier.audit import AuditRun, SessionError
from verity.protocols.resource_accounting.memory.poub.verifier.receipts import Plan

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

#: the verdict word's row as the layout registers it
VERDICT_BITS = 16
OK, CODEC = 1, 1 << 2
#: a store at 16 sketch rows
WORLD = M.World(rows=16)


def statics(reg: audit.Registration, d: audit.PositionSample):
    return O.statics(reg.codec, receipts.PER_LEAF, 1, O.depth(receipts.leaves(d.plan.length)), O.depth(reg.blocks), 1)


def span_public(v: Any) -> dict:
    """`spanPublicFn`: the case's session through `audit.AuditRun`, then window t's samples, each one's rows and the
    supplied values."""
    with M.decoding():
        reg = M.registration_of(M._get(v, "registration"))
        kw, committed, appends, not_before = M.session_of(v)
        events = [M.event_of(e) for e in M._list(v, "events")]
        params = M.params_of(reg, kw)
        t = M.int_(v, "t")
    clock = M.Clock()
    s = AuditRun(params, reg, None, M.replayed(reg, 0, params.windows, committed, appends, params.groups_per_coin),
                 not_before_ns=not_before, clock=clock)
    for e in events:
        M.apply(s, clock, e)
    try:
        ds = s.drawn(t)
    except SessionError as e:
        raise D.Reject(str(e)) from None
    sup = span.supplied(ds, reg.codec_seed, statics(reg, ds[0])) if ds else None
    return {"draws": [[d.index, d.position] for d in ds],
            "rows": [span.public_rows(d, reg.codec_seed, statics(reg, d)) for d in ds],
            "supplied": None if sup is None else {r: [[{"bytes": b.hex()}] for b in rows] for r, rows in sup.items()}}


PUBLIC = D.Port(M.EXE, "span_public", span_public, st.nothing(), package=M.PACKAGE, of_record=M.OF_RECORD)
server = D.server_fixture(PUBLIC)


def drawn(world: M.World, plans: tuple = (None, None)) -> M.Run:
    """One window of two passes over the world's store, each its plan (None: the whole store), closed and sampled, every
    receipt sampled."""
    r = M.Run(world, windows=1)
    r.start()
    for plan in plans:
        r.run(plan)
    r.boundary()
    assert isinstance(r.close(0), tuple)
    return r


def salt(label: bytes) -> bytes:
    return hashlib.shake_256(label + b"o0").digest(receipts.SALT_BYTES)


def public_file(words: list[int], ss: list[bytes], ports: list[list], indices: range) -> bytes:
    """A public file of one instance a unit, its units ``indices``: each `o0` row's ``b || c`` under its salt, every other
    row's a stand-in."""
    body = b""
    for k, (w, s) in enumerate(zip(words, ss, strict=True)):
        for name, bits in ports:
            body += O.commit(bits, [w], s) if name == "o0" else hashlib.sha512(b"%d/%s" % (k, name.encode())).digest() * 2
    header = {"instances": len(words), "units": {"indices": list(indices)}}
    return json.dumps(header, sort_keys=True).encode() + b"\n" + body


class Span:
    """A sampled window, its honest units' words, and what the prover sends of them."""

    def __init__(self, r: M.Run) -> None:
        self.r, reg = r, r.store.registration
        self.ds = r.s.drawn(0)
        self.ports = span.layout(statics(reg, self.ds[0]), VERDICT_BITS)
        self.words = [OK] * len(self.ds)
        self.salts = [salt(b"unit %d " % k) for k in range(len(self.ds))]

    def held(self, cut: tuple[int, ...] = (), words=None, sent=None, ss=None, indices=None) -> list[dict]:
        """The statements' public files and instances, the units cut into statements at ``cut``; ``words`` and ``ss``
        what the files commit, ``sent`` the words the prover sends (each the committed one unless given), ``indices``
        the units' population indices (0, 1, … unless given)."""
        words, ss = words or self.words, ss or self.salts
        sent, indices = sent or words, indices or list(range(len(words)))
        out, at = [], 0
        for end in [*cut, len(words)]:
            ks = range(at, end)
            out.append({"instances": [span.instance(sent[k], self.salts[k]) for k in ks],
                        "public": public_file([words[k] for k in ks], [ss[k] for k in ks], self.ports,
                                              indices[at:end]).hex()})
            at = end
        return out

    def case(self, held: list[dict]) -> dict:
        return {**self.r.case(), "held": held, "layout": self.ports, "t": 0}


def held(server, case: dict) -> dict:
    kind, out = server.call("span_held", case)
    assert kind == "output", out
    return json.loads(out)


def rejected(at: tuple[int, int] | None, failure: str = "malformed") -> dict:
    return {"outcome": failure, "result": {"at": None if at is None else list(at), "failure": failure, "window": 0}}


ACCEPTED = {"outcome": "accepted", "result": {"accepted": 0}}


@pytest.mark.parametrize("world", [WORLD, M.World(M.P512_K3, rows=16)], ids=["p512-k2", "p512-k3-masked"])
def test_the_verifiers_rows_are_the_ones_the_prover_commits(server, world) -> None:
    case = {**drawn(world).case(), "t": 0}
    kind, out = D.replay(PUBLIC, case, server)
    assert kind == "output" and len(json.loads(out)["draws"]) == 12


def test_a_window_with_no_samples_awaiting_has_no_rows(server) -> None:
    r = M.Run(WORLD, windows=1)
    r.start()
    r.run()
    assert D.replay(PUBLIC, {**r.case(), "t": 0}, server)[0] == "reject"


def test_an_honest_span_is_accepted_in_one_statement_or_several(server) -> None:
    s = Span(drawn(WORLD))
    assert held(server, s.case(s.held())) == ACCEPTED
    assert held(server, s.case(s.held(cut=(5, 9)))) == ACCEPTED


def test_a_unit_that_says_it_failed_rejects_the_window_there(server) -> None:
    s = Span(drawn(WORLD))
    words = list(s.words)
    words[3] = CODEC
    assert held(server, s.case(s.held(words=words))) == rejected(tuple(s.r.draws[0][3]), "codec")
    words[3] = 0
    assert held(server, s.case(s.held(words=words))) == rejected(tuple(s.r.draws[0][3]))


def test_a_verdict_word_or_its_salt_changed_is_malformed_at_its_sample(server) -> None:
    s = Span(drawn(WORLD))
    words = list(s.words)
    words[2] = CODEC
    sent = list(s.words)
    assert held(server, s.case(s.held(words=words, sent=sent))) == rejected(tuple(s.r.draws[0][2]))
    ss = list(s.salts)
    ss[4] = bytes(receipts.SALT_BYTES)
    assert held(server, s.case(s.held(ss=ss))) == rejected(tuple(s.r.draws[0][4]))


def test_units_not_the_populations_in_order_are_malformed(server) -> None:
    """Unit k reads the supplied rows' row k, so a span whose units are not 0, 1, … in order is no window's: two units
    swapped, or a second statement's units counted from 0 again."""
    s = Span(drawn(WORLD))
    order = list(range(len(s.ds)))
    order[0], order[1] = order[1], order[0]
    assert held(server, s.case(s.held(indices=order))) == rejected(None)
    assert held(server, s.case(s.held(cut=(6,), indices=[*range(6), *range(len(s.ds) - 6)]))) == rejected(None)


def test_a_missing_unit_or_statement_is_malformed_at_no_sample(server) -> None:
    s = Span(drawn(WORLD))
    statements = s.held(cut=(6,))
    assert held(server, s.case(statements[:1])) == rejected(None)
    one = s.held()[0]
    short = {"instances": one["instances"][:-1], "public": one["public"]}
    assert held(server, s.case([short])) == rejected(None)
    assert held(server, s.case([])) == rejected(None)


def test_a_pass_whose_plan_is_not_the_identity_is_malformed(server) -> None:
    """v2's unit has no plan row: a sample of a pass that reads the store in another order is no unit's, whatever its
    statements."""
    s = Span(drawn(WORLD, plans=(None, Plan(((1, 3), (0, 1)), 6))))
    k = next(k for k, d in enumerate(s.ds) if d.index == 1)
    assert held(server, s.case(s.held())) == rejected(tuple(s.r.draws[0][k]))


def test_the_whole_check_rejects_a_span_whose_statements_no_audit_accepted(server) -> None:
    """`span_held` accepts these public files; `span` reads the same files from blobs and first asks the proof system's
    audit of each statement, which accepts none of these: the window is `malformed` at no sample."""
    s = Span(drawn(WORLD))
    h = s.held()
    assert held(server, s.case(h)) == ACCEPTED
    public = bytes.fromhex(h[0]["public"])
    st_ = span.statement(public, b"{}", b"{}", h[0]["instances"])
    case = {**s.r.case(), "t": 0, "auditor": span.auditor({}, s.ports), "span": span.span([st_], [public])}
    kind, out = server.call("span", case)
    assert kind == "output" and json.loads(out) == rejected(None), out
