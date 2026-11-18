"""Refresh's slot schedule in `Poub.Session`, held to the outcome each case is built to reach: the Python books have no
refresh yet, so a case is driven by `ra_cases.Run` with refresh off (its coins, passes and draws) and sent to the Lean
with refresh on, its appends naming their slots and origins and a re-key's openings their origin's leaf. With refresh
off nothing in a session's input or record names it."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from fractions import Fraction

import pytest
import ra_cases as M
import verity_difftest as D
from hypothesis import strategies as st

from verity.protocols.resource_accounting.memory.poub.prover import audit_prover
from verity.protocols.resource_accounting.memory.poub.verifier.receipts import Plan

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

PORT = D.Port(M.EXE, "session", M.session, st.nothing(), package=M.PACKAGE, of_record=M.OF_RECORD)
server = D.server_fixture(PORT)

REKEY_DOMAIN = b"verity/remote-bandwidth/rekey/v0\0"
N = 6


def rekey_root(registration: bytes, first: int, count: int, origin: dict) -> bytes:
    """`Poub.rekeyRoot`: a re-key's declaration, which `append` takes as its root."""
    kind = 0 if "registered" in origin else 1
    x = origin.get("registered", origin.get("append"))
    return hashlib.sha512(REKEY_DOMAIN + registration + first.to_bytes(8, "big") + count.to_bytes(8, "big")
                          + bytes([kind]) + x.to_bytes(8, "big")).digest()


class Refresh(M.Run):
    """A session over the 6-block store at L = 2 whose case names `refresh` (unless `refresh` is False) and
    `fresh_slots`. With F = 1 its slots are: window 0 fresh 0; window 1 registered [3, 6) 1, fresh 2; window 2
    registered [0, 3) 3, re-key of 0 4, fresh 5; window 3 registered [3, 6) 6, re-key of 2 7, fresh 8; window 4
    registered [0, 3) 9, re-key of 0 10, re-key of 5 11, fresh 12."""

    def __init__(self, *, refresh: bool = True, fresh_slots: int = 1, **kw) -> None:
        super().__init__(M.World(blocks=N), lifetime=2, **kw)
        self.refresh, self.fresh_slots = refresh, fresh_slots
        self.by_slot: dict[int, audit_prover.Append] = {}
        self.origins: dict[int, tuple[int, audit_prover.Append | None]] = {}
        self.slot_coins: dict[int, bytes] = {}

    @property
    def digest(self) -> bytes:
        return self.store.registration.digest

    def send(self, root: bytes, count: int, slot: int | None, origin: dict | None = None) -> object:
        """An append sent with its slot and origin, which the Python books don't read."""
        event = {"kind": "append", "root": root.hex(), "count": count}
        event |= ({} if slot is None else {"slot": slot}) | ({} if origin is None else {"origin": origin})
        out = self._call(event, lambda: self.s.append(root, count))
        if isinstance(out, tuple):
            first, coin = out
            self.append_coins.append(coin)
            if slot is not None:
                self.slot_coins[slot] = coin
            self.head = M.audit.append_head(self.head, first, count, root)
            if first in self.store.pending:
                self.store.encode(first, coin)
        return out

    def fresh(self, count: int, slot: int | None) -> audit_prover.Append:
        a = self.store.append(M.pages(self.store.extent, count), M.salts(b"page salts %d" % self.store.extent, count))
        self.send(a.root, count, slot)
        if slot is not None:
            self.by_slot[slot] = a
        return a

    def _rekey(self, payloads, salts, slot: int | None, origin: dict, at: int, of: audit_prover.Append | None,
               root: bytes | None = None) -> audit_prover.Append:
        a = self.store.append(payloads, salts)
        self.origins[a.first] = (at, of)
        self.send(root or rekey_root(self.digest, a.first, a.count, origin), a.count, slot, origin)
        return a

    def registered(self, slot: int | None, lo: int, count: int, root: bytes | None = None) -> audit_prover.Append:
        """The re-key of registered blocks `lo … lo + count − 1`, their payloads and salts appended again."""
        return self._rekey(self.store.payloads[lo:lo + count], self.store.salts[lo:lo + count], slot,
                           {"registered": lo}, lo, None, root)

    def rekey(self, slot: int, k: int) -> audit_prover.Append:
        """The whole re-key of the fresh append in slot `k`."""
        b = self.by_slot[k]
        return self._rekey([self.store.payload(j) for j in range(b.first, b.first + b.count)], b.salts, slot,
                           {"append": k}, b.first, b)

    def pass_(self, *runs: tuple[int, int]) -> object:
        return self.run(Plan(runs, self.store.extent))

    def lean_openings(self, t: int) -> list[dict]:
        """Window t's openings as the Lean takes them: a re-key's block names its origin's leaf and opens its payload
        under the origin's root (the registration's payload tree, or the fresh append's tree)."""
        out = []
        for (i, p), o in zip(self.draws[t], self.openings(t)):
            j = self.passes[t, i].plan.block(p)
            a = next((a for a in self.store.appends if a.first <= j < a.first + a.count), None)
            doc = M.opening_json(o)
            if a is not None and a.first in self.origins:
                at, of = self.origins[a.first]
                q = at + (j - a.first)
                tree = self.store.payload_tree if of is None else of.tree
                doc |= {"origin_position": q, "payload_path": [x.hex() for x in tree.open(q).path]}
            out.append(doc)
        return out

    def finish_lean(self, edit=None) -> None:
        """Every window closed and verified in order, its openings in the Lean's form, through `edit` if given."""
        for t in range(len(self.s.windows)):
            if not isinstance(self.close(t), tuple):
                return
            opened = self.lean_openings(t)
            self.events.append({"kind": "verify", "t": t, "openings": edit(t, opened) if edit else opened})

    def case(self) -> dict:
        """The case, with refresh its append coins by slot: each slot's the coin the Python books gave the append sent
        in it, empty for a slot nothing was sent in."""
        out = super().case()
        if self.refresh:
            out["params"] |= {"refresh": True, "fresh_slots": self.fresh_slots}
            n = max(self.slot_coins, default=-1) + 1
            out["append_coins"] = [self.slot_coins[k].hex() if k in self.slot_coins else "" for k in range(n)]
        return out


def lean(server, r: Refresh) -> dict:
    kind, out = PORT.lean(server, r.case())
    assert kind == "output", out
    return json.loads(out)


def honest(edit=None) -> Refresh:
    """Five windows: a fresh append in window 0's slot, each window's registered re-key from window 1, and slot 0's
    re-key in windows 2 and 4; every plan holds only what refresh lets it."""
    r = Refresh(windows=5)
    r.start()
    r.pass_((0, N))
    r.fresh(2, 0)
    r.pass_((0, N))
    r.boundary()
    r.pass_((0, N))
    r.registered(1, 3, 3)
    r.pass_((0, N), (8, 3))
    r.boundary()
    r.registered(3, 0, 3)
    r.rekey(4, 0)
    r.pass_((8, 8))
    r.pass_((8, 8))
    r.boundary()
    r.registered(6, 3, 3)
    r.pass_((11, 8))
    r.pass_((11, 8))
    r.boundary()
    r.registered(9, 0, 3)
    r.rekey(10, 0)
    r.pass_((16, 8))
    r.pass_((16, 8))
    r.boundary()
    r.finish_lean(edit)
    return r


def test_an_honest_refresh_session_is_accepted_with_its_record(server) -> None:
    """Window 0 holds the fresh append in probation from its second pass, window 1 still; from window 2 nothing is."""
    r = honest()
    out = lean(server, r)
    assert out["outcome"] == "accepted", out["results"]
    rec = out["record"]
    assert rec["refresh"] is True and rec["appended"] == 18
    assert [w["probation"] for w in rec["windows"]] == [[0, 2], [2, 2], [0, 0], [0, 0], [0, 0]]
    opened = {o.get("origin_position") for e in r.events if e["kind"] == "verify" for o in e["openings"]}
    assert {0, 1, 2, 3, 4, 5, 6, 7} <= opened


def _at(t: int, j: int, **fields):
    """An `edit` of window t's openings: the first opening of block j with `fields` set (None: removed)."""
    def edit(u: int, opened: list[dict]) -> list[dict]:
        if u != t:
            return opened
        out = copy.deepcopy(opened)
        k = next(k for k, o in enumerate(out) if o.get("origin_position") == j)
        for key, v in fields.items():
            if v is None:
                out[k].pop(key)
            else:
                out[k][key] = v(out[k]) if callable(v) else v
        return out
    return edit


def test_a_rekeys_opening_under_its_own_tree_is_refused_origin_path(server) -> None:
    """Window 2's opening of slot 0's first block, re-keyed in slot 4 at block 14, opened along the re-key append's own
    tree. The sessions are deterministic, so a second one's store gives the first's path."""
    own = [x.hex() for x in honest().store.committed(14)[1]]
    assert lean(server, honest(_at(2, 6, payload_path=own)))["outcome"] == "origin-path"


@pytest.mark.parametrize("fields", [{"origin_position": None}, {"origin_position": lambda o: o["origin_position"] + 1}])
def test_a_rekeys_opening_without_its_origins_leaf_is_malformed(server, fields) -> None:
    assert lean(server, honest(_at(2, 6, **fields)))["outcome"] == "malformed"


def _window0(r: Refresh) -> None:
    r.start()
    r.pass_((0, N))


@pytest.mark.parametrize("build,failure", [
    (lambda r: r.fresh(2, 0), "append"),
    (lambda r: r.registered(None, 3, 3), "append"),
], ids=["slot", "origin"])
def test_without_refresh_a_slot_or_a_rekey_is_refused_append(server, build, failure) -> None:
    r = Refresh(refresh=False)
    _window0(r)
    build(r)
    out = lean(server, r)
    assert out["outcome"] == failure


def _shifted(r: Refresh) -> None:
    r.boundary()
    r.fresh(2, 0)
    r.pass_((0, N))


def _wrong_kind(r: Refresh) -> None:
    r.boundary()
    r.fresh(2, 1)


def _wrong_count(r: Refresh) -> None:
    r.boundary()
    r.registered(1, 3, 2)


def _wrong_lo(r: Refresh) -> None:
    r.boundary()
    r.registered(1, 4, 2)


def _out_of_order(r: Refresh) -> None:
    r.fresh(2, 0)
    r.fresh(1, 0)


def _past_schedule(r: Refresh) -> None:
    r.fresh(2, 13)


def _over_cap(r: Refresh) -> None:
    r.send(hashlib.sha512(b"513 blocks").digest(), 513, 0)


def _empty_slot(r: Refresh) -> None:
    r.boundary()
    r.pass_((0, N))
    r.boundary()
    r.send(rekey_root(r.digest, r.store.extent, 2, {"append": 0}), 2, 4, {"append": 0})


def _wrong_root(r: Refresh) -> None:
    r.boundary()
    r.registered(1, 3, 3, root=rekey_root(r.digest, r.store.extent, 3, {"registered": 0}))


def _registered_late(r: Refresh) -> None:
    for _ in range(2):
        r.boundary()
        r.pass_((0, N))


def _probation(r: Refresh) -> None:
    r.fresh(2, 0)
    r.pass_((0, N))
    r.boundary()
    r.pass_((0, N + 2))


def _fresh_never(r: Refresh) -> None:
    r.fresh(2, 0)
    r.pass_((0, N))
    r.boundary()
    r.registered(1, 3, 3)
    r.pass_((0, N))
    r.boundary()
    r.registered(3, 0, 3)
    r.pass_((6, 5))


@pytest.mark.parametrize("build,failure", [
    (_shifted, "origin"),
    (_wrong_kind, "origin"),
    (_wrong_count, "origin"),
    (_wrong_lo, "origin"),
    (_out_of_order, "origin"),
    (_past_schedule, "origin"),
    (_over_cap, "origin"),
    (_empty_slot, "origin"),
    (_wrong_root, "append"),
    (_registered_late, "lifetime"),
    (_probation, "probation"),
    (_fresh_never, "lifetime"),
], ids=lambda x: x if isinstance(x, str) else x.__name__.strip("_"))
def test_refresh_refuses_a_session_off_its_schedule(server, build, failure) -> None:
    """At W = 5, L = 2, F = 1 (`Refresh`): a fresh append sent in window 1 for window 0's slot; a fresh append in a
    registered slot; a registered re-key short of its class run, or not from its start; a slot sent twice; a slot past
    the schedule's 13; a fresh append over 512 blocks; the re-key of an empty slot; a re-key root that isn't its
    declaration; a registered block planned at t = L; a fresh append's block planned in its probation, and once out of
    it, past its lifetime."""
    r = Refresh(windows=5)
    _window0(r)
    build(r)
    out = lean(server, r)
    assert out["outcome"] == failure, out["results"]
    assert [k for k, x in enumerate(out["results"]) if "failure" in x] == [len(r.events) - 1]


def gamma(server, registration: dict, record: dict, **extra) -> dict:
    """`Poub.Gamma`'s terms of `record` at δ = 2^-40 and four times the least capacity its checks pass at."""
    ws = [(sum(w["lengths"]), w["deadline_ns"] - w["released_ns"]) for w in record["windows"]]
    span = record["windows"][-1]["deadline_ns"] - record["windows"][0]["released_ns"]
    least = max(Fraction(u * M.codec.BLOCK_BYTES * 10 ** 9, s) for u, s in [(sum(u for u, _ in ws), span), *ws])
    kind, text = server.call("gamma", {
        "registration": registration, "record": record, "delta": "1/1099511627776", "block_bytes": M.codec.BLOCK_BYTES,
        "chip_bytes": 0, "capacity_bytes_per_s": str(4 * least.numerator) + "/" + str(least.denominator),
        "squarings_per_s": "0"} | extra)
    assert kind == "output", text
    return json.loads(text)


def test_a_refresh_record_credits_its_appended_blocks_on_its_schedule_check_alone_with_no_carrier(server) -> None:
    """Its post-seed receipts are credited given the schedule check's cap, with no stacking verdict and carrier 0;
    without it only its registered ones are. Its probation share, the most any pass held of what it held and read
    (window 0's second pass: 2 of 2 + 6), is reported either way."""
    r = honest()
    rec = lean(server, r)["record"]
    assert sum(p for w in rec["windows"] for p in w["post_seed"]) > 0
    plain, credited = gamma(server, r.case()["registration"], rec), gamma(server, r.case()["registration"], rec,
                                                                          scheduled=7)
    assert Fraction(credited["terms"]["timing"]) < Fraction(plain["terms"]["timing"])
    assert credited["terms"]["carrier"] == plain["terms"]["carrier"] == "0"
    assert (credited["steering_bits"], plain["steering_bits"]) == (7, None)
    assert credited["probation"] == plain["probation"] == "1/4"


def test_a_schedule_check_credits_nothing_on_a_record_without_refresh(server) -> None:
    case, _ = M.recorded("honest-rekey")
    rec = json.loads(PORT.lean(server, case)[1])["record"]
    assert gamma(server, case["registration"], rec, scheduled=0) == {
        "refused": "only a refresh record credits appended blocks on its schedule check alone"}
    assert "probation" not in gamma(server, case["registration"], rec)


def test_fresh_slots_without_refresh_are_refused(server) -> None:
    r = Refresh(refresh=False)
    _window0(r)
    case = r.case()
    case["params"]["fresh_slots"] = 1
    assert PORT.lean(server, case)[0] == "reject"


def test_refresh_off_named_or_not_gives_the_same_session_and_names_nothing(server) -> None:
    case, _ = M.recorded("honest-rekey")
    plain = PORT.lean(server, case)
    named = copy.deepcopy(case)
    named["params"]["refresh"] = False
    assert PORT.lean(server, named) == plain and plain[0] == "output"
    rec = json.loads(plain[1])["record"]
    assert "refresh" not in rec and all("probation" not in w for w in rec["windows"])


@pytest.mark.parametrize("name", ["honest-refresh", "refresh-skipped-rekey", "refresh-coin-of-empty-slot"])
def test_an_empty_slots_coin_is_never_released_and_a_later_append_takes_its_own_slots(server, name) -> None:
    """Both sides on a refresh session whose slots 2 and 5 (and 6, skipped) stay empty: each later append takes its own
    slot's coin, and R2, in slot 3, encoded under empty slot 2's coin, is refused `codec`."""
    case, outcome = M.recorded(name)
    kind, lean = D.replay(PORT, copy.deepcopy(case), server)
    assert kind == "output" and json.loads(lean)["outcome"] == outcome == M.REFRESH_SCENARIOS[name][3]

