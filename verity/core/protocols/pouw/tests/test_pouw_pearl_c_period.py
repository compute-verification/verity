"""`credit.pearl_c_period_credit` (the Pearl-C period rule): a live Pearl-C pass earns only inside a
Pearl-C period, which earns only when the presented passes' epochs are exactly its close's sessions, every session is a
Pearl-C salt session, credited and pinned, every pass is accepted, each weights' root's epochs are numbered on from the
lane's earlier credited periods with no gap, and no two passes of one epoch hold one call index. An honest sequence of
two periods is credited; each clause, an abandoned complete salt-only session and a gap in k refuse the period."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

from verity.core.protocols.pouw import credit as Cr
from verity.core.protocols.pouw import salt as PS
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.audit import EPOCH_SALT_DOMAIN
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.ml.crypto import lms
from verity.ml.randomness import derive

PKG = Path(__file__).resolve().parents[1] / "lean/check/tests"
for _name in ("pouw_rounds_cases", "pouw_pass_cases"):
    if _name not in sys.modules:
        _spec = importlib.util.spec_from_file_location(_name, PKG / f"{_name}.py")
        sys.modules[_name] = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(sys.modules[_name])
rc, pc = sys.modules["pouw_rounds_cases"], sys.modules["pouw_pass_cases"]

DESCRIPTOR = b"verity/test/pearl-c-period/v1"
CIRCUIT = hashlib.sha512(DESCRIPTOR).hexdigest()


class Lane:
    """One lane's auditor ledger, in time order, signed under key 0 at increasing leaves: Pearl-C periods (`open`,
    `close`) and salt sessions (`epoch`), each session reserved, opened on its epoch's root, its record and close
    signed at the two leaves after its open and ended as the gate ends it."""

    def __init__(self, seed: str) -> None:
        self.rng = random.Random(seed)
        self.events: list[tuple[str, str | None, dict]] = []
        self.leaves: list[int] = []
        self.sessions: dict[str, tuple[rc.Session, bytes, int]] = {}
        self.members: list[str] | None = None
        self.pub = lms.public_key(rc._key(0))
        self._ledger: list[str] | None = None

    def _push(self, event: tuple[str, str | None, dict], skip: int = 0) -> None:
        self.leaves.append(self.leaves[-1] + 1 + skip if self.leaves else 0)
        self.events.append(event)
        self._ledger = None

    def open(self, name: str, circuit: str = CIRCUIT, **fields: Any) -> None:
        """Period `name`'s open, committing `circuit` and the further `fields` (its pin and freshness floor)."""
        self._push(("period-open", None, {"period": name, "lane": rc.LANE, "by": rc.LANE, "cap": 4, "circuit": circuit,
                                          "key_sha512": hashlib.sha512(self.pub).hexdigest(), **fields}))
        self.name, self.members = name, []

    def epoch(self, w: bytes, k: int, h1: bool = False, pinned: str | None = None) -> str:
        """Epoch k over weights' root `w`: a salt-only session (H1's three-family layout, every coin released, when
        `h1`), its salt coin opened after `salt.epoch_open_root(w, k)`, its open naming commit `pinned` (a commit of
        `rc.PINS[0]` pins, which every sealed prefix holds) when given."""
        rng, r, sid = self.rng, PS.epoch_open_root(w, k), rc.session_id(self.rng)
        leaf = self.leaves[-1] + 3 if self.leaves else 2
        if h1:
            ch = Wd.live_challenger(rng.randrange(4), 1, rng.randbytes)
            s = rc.Session(rng, ch, [rng.randbytes(64)], salt=r, sid=sid, leaf=leaf, draw=lambda _: rng.randbytes(64))
        else:
            s = rc.Session(rng, Ch.LiveChallenger(Wd.LIVE_SCHEME, rng.randrange(4), Wd.SALT_FAMILIES, rng.randbytes), [],
                           salt=r, sid=sid, leaf=leaf)
        c = s.ch.commitment
        asked = {"scheme": c.scheme, "epoch": c.epoch, "families": [list(f) for f in c.families]}
        o = {**rc._open(rng, {"root": r.hex()}, s.pub), **asked, **({"pinned": pinned} if pinned else {})}
        reserve, opened = rc._reserved(rng, sid, o)
        self._push(reserve)
        self._push(opened)
        self._push(("end", sid, rc._end(s, **pc.gate_end(s))), skip=2)
        self.sessions[sid] = (s, w, k)
        if self.members is not None:
            self.members.append(sid)
        return sid

    def close(self) -> None:
        roots = {sid: b["window"]["root"] for kind, sid, b in self.events if kind == "open"}
        ended = {sid: b["complete"] is True for kind, sid, b in self.events if kind == "end"}
        windows = [{"root": roots[sid], "session": sid, "complete": ended[sid]} for sid in self.members]
        self._push(("period-close", None, {"period": self.name, "lane": rc.LANE, "by": rc.LANE, "windows": windows,
                                           "complete": all(w["complete"] for w in windows)}))
        self.members = None

    def ledger(self) -> list[str]:
        assert self.leaves[-1] < 1 << rc.PARAMS.H
        if self._ledger is None:
            self._ledger = rc.ledger_lines(rc._bind(self.events), rc._key(0), self.leaves)
        return list(self._ledger)

    def args(self, name: str, passes: Any, *earlier: Any) -> tuple:
        """`pearl_c_period_credit`'s arguments for period `name`'s `passes`, the lane's earlier epochs `earlier` (none
        when not given)."""
        key = next(iter(self.sessions.values()))[0].key()
        return (key, self.ledger(), list(rc.PINS), list(rc.MAIN), name, passes, earlier[0] if earlier else [],
                rc.lms_verify)

    def manifest(self, sid: str, base: dict, first: int) -> dict:
        """A pass of session `sid`'s epoch over `base`'s workload, its call indices from `first` on."""
        s, w, k = self.sessions[sid]
        c, o = s.ch.commitment, s.salt[0]
        salt = derive(o.coin, EPOCH_SALT_DOMAIN, {"epoch": c.epoch, "coins": c.root, "weights": w}).stream("salt", 32)
        man = copy.deepcopy(base)
        man.update(matmuls=[{**u, "index": u["index"] + first} for u in man["matmuls"]], pass_index=k,
                   run=f"r-{self.rng.randrange(1 << 30):x}", coins=c.to_json(), salt_opening=o.to_json(),
                   salt=salt.hex())
        opened = next(e for e in map(json.loads, self.ledger()) if e["kind"] == "open" and e["session"] == sid)
        man["session"] = pc.carried(opened, c)
        return man

    def evidence(self, sid: str, man: dict, descriptor: bytes = DESCRIPTOR) -> dict:
        raw = json.dumps(man).encode()
        return {"context": f"served-zk/{man['run']}/{hashlib.sha512(raw).hexdigest()}", "descriptor": descriptor.hex(),
                "manifest": raw.hex(), "session": self.sessions[sid][0].files()}

    def pass_(self, sid: str, base: dict, first: int = 0) -> dict:
        return self.evidence(sid, self.manifest(sid, base, first))


def _base(rng: random.Random) -> tuple[dict, bytes, int]:
    """A workload (its call indices 0 … n − 1), its weights' root W and n."""
    man = pc._pass_manifest(rng)
    return man, pc.CC.weights_root(pc.Co.SCHEME, pc._roots(man)), len(man["matmuls"])


def _two_periods(seed: str) -> tuple[Lane, list[dict], list[dict], list[bytes]]:
    """Period A: epoch 0 over W (two passes, the second's calls numbered on from the first's) and epoch 1 over W;
    period B: epoch 2 over W and epoch 0 over a second weights' root W2."""
    lane = Lane(seed)
    (b1, w, n), (b2, w2, _) = _base(lane.rng), _base(lane.rng)
    lane.open("p-a")
    a0, a1 = lane.epoch(w, 0), lane.epoch(w, 1)
    lane.close()
    lane.open("p-b")
    b_2, b_0 = lane.epoch(w, 2), lane.epoch(w2, 0)
    lane.close()
    a = [lane.pass_(a0, b1), lane.pass_(a0, b1, n), lane.pass_(a1, b1)]
    b = [lane.pass_(b_2, b1), lane.pass_(b_0, b2), lane.pass_(b_0, b2, 1 << 20)]
    return lane, a, b, [w, w2]


def test_an_honest_two_period_sequence_is_credited():
    for seed in range(2):
        lane, a, b, (w, w2) = _two_periods(f"honest/{seed}")
        got, why = Cr.pearl_c_period_credit(*lane.args("p-a", a))
        assert why is None and got["epochs"] == [[w.hex(), 0], [w.hex(), 1]]
        assert got["passes"] == [json.loads(bytes.fromhex(p["manifest"])) for p in a]
        got_b, why = Cr.pearl_c_period_credit(*lane.args("p-b", b, got["epochs"]))
        assert why is None and got_b["epochs"] == [[w.hex(), 2], [w2.hex(), 0]] and len(got_b["passes"]) == 3


def test_a_pass_whose_epoch_the_close_does_not_list_or_a_session_with_no_pass_refuses_the_period():
    """Clause 1, exact match: the presented epochs are exactly the close's sessions, in its order."""
    lane, a, b, _ = _two_periods("exact")
    p0, p0b, p1 = a
    assert Cr.pearl_c_period_credit(*lane.args("p-a", [p0, p0b]))[1][0] == "missing"
    assert Cr.pearl_c_period_credit(*lane.args("p-a", [p1, p0, p0b]))[1][0] == "order"
    _, why = Cr.pearl_c_period_credit(*lane.args("p-a", [p0, b[0], p0b, p1]))
    assert why[0] == "period" and "only the windows its close lists count" in why[1]
    assert Cr.pearl_c_period_credit(*lane.args("p-c", a))[1][0] == "period"


def test_an_abandoned_complete_salt_only_session_refuses_its_period():
    """A complete salt-only session ends like a used one, so a prover could open several and keep the salt it likes:
    one the close lists that no presented pass names refuses the whole period, wherever it falls in the close."""
    for where in range(3):
        lane = Lane(f"abandoned/{where}")
        base, w, n = _base(lane.rng)
        lane.open("p-a")
        sids = [lane.epoch(w, k) for k in range(3)]
        lane.close()
        abandoned = sids[where]
        assert {e["kind"]: e["body"] for e in map(json.loads, lane.ledger())
                if e["session"] == abandoned}["end"]["complete"] is True
        passes = [lane.pass_(sid, base) for sid in sids if sid != abandoned]
        got, why = Cr.pearl_c_period_credit(*lane.args("p-a", passes))
        assert got is None and why[0] == "missing" and "is missing" in why[1], where


def test_a_session_not_salt_only_refuses_the_period():
    """Clause 2: an H1 window's session (salt, draw and rounds, every coin released, so the period's close is complete)
    opened on an epoch's root is not a Pearl-C salt session, and H1's window check refuses a salt-only session."""
    lane = Lane("salt-only")
    base, w, n = _base(lane.rng)
    lane.open("p-a")
    s0, s1 = lane.epoch(w, 0), lane.epoch(w, 1, h1=True)
    lane.close()
    got, why = Cr.pearl_c_period_credit(*lane.args("p-a", [lane.pass_(s0, base), lane.pass_(s1, base)]))
    assert got is None and why[0] == Cr.SALT_ONLY and "not a Pearl-C salt session's" in why[1]
    salt_only, h1 = (lane.sessions[s][0].ch.commitment for s in (s0, s1))
    assert Wd.live_epoch_reason(salt_only) is not None and Wd.salt_epoch_reason(h1) is not None


def test_a_session_of_another_circuit_refuses_the_period():
    """Clause 3: each pass's session is credited toward its period, the pass's descriptor hashing to the circuit the
    period's open committed."""
    lane, a, _, _ = _two_periods("circuit")
    passes = list(a)
    passes[1] = {**passes[1], "descriptor": b"another circuit".hex()}
    _, why = Cr.pearl_c_period_credit(*lane.args("p-a", passes))
    assert why[0] == "circuit" and "the one period p-a committed" in why[1]


def test_a_session_pinned_behind_its_periods_open_pin_refuses_the_period():
    """Clause 3a, H1's period pin floor: a period opened under MAIN[2] with a freshness floor of one first-parent commit
    refuses a session pinned at MAIN[4], two behind it, with ``pin``; at a floor of two the same session is credited."""
    for floor, refused in ((1, True), (2, False)):
        lane = Lane(f"pin/{floor}")
        base, w, _ = _base(lane.rng)
        lane.open("p-a", **{Cr.PERIOD_PIN_FIELD: rc.MAIN[2], Cr.STALE_FIELD: floor})
        sid = lane.epoch(w, 0, pinned=rc.MAIN[4])
        lane.close()
        got, why = Cr.pearl_c_period_credit(*lane.args("p-a", [lane.pass_(sid, base)]))
        if refused:
            assert got is None and why[0] == "pin" and "pass 0 of period p-a is pinned 2 first-parent" in why[1], why
        else:
            assert why is None and got["epochs"] == [[w.hex(), 0]]


def test_a_refused_pass_refuses_the_period():
    """Clause 4: each pass passes `core_check.pass_epoch` under the live inputs: a context naming another manifest, or
    a salt that isn't its live coin's, refuses the period."""
    lane, a, _, _ = _two_periods("pass")
    p0, p0b, p1 = a
    named = {**p0b, "context": p0["context"]}
    _, why = Cr.pearl_c_period_credit(*lane.args("p-a", [p0, named, p1]))
    assert why[0] == Cr.PASS_REFUSED and "the pass is not the one the window's parameters name" in why[1]
    man = json.loads(bytes.fromhex(p1["manifest"]))
    man["salt"] = ("1" if man["salt"][0] == "0" else "0") + man["salt"][1:]
    sid = man["session"]["session"]
    _, why = Cr.pearl_c_period_credit(*lane.args("p-a", [p0, p0b, lane.evidence(sid, man)]))
    assert why[0] == Cr.PASS_REFUSED and "the pass's salt is not its live coin's" in why[1]


def test_a_context_not_a_string_refuses_the_period_as_a_refused_pass():
    """Clause 4: a ``context`` that isn't a string is refused by name (``pass``), not raised from `pass_epoch`."""
    lane, a, _, _ = _two_periods("context")
    p0, p0b, p1 = a
    for context in (None, 7, [p0b["context"]], {"context": p0b["context"]}):
        got, why = Cr.pearl_c_period_credit(*lane.args("p-a", [p0, {**p0b, "context": context}, p1]))
        assert got is None and why[0] == Cr.PASS_REFUSED and "pass 1 of period p-a" in why[1], context
        assert "context is not a string" in why[1]


def _retained(lane: Lane, sid: str, base: dict) -> bytes:
    """Session `sid`'s pass retained in the vLLM integration's form (`PearlCDeviceRun._write_retained`, `write_manifest`: the
    head, then ``weights`` and ``matmuls``, at indent 1 with a newline), its ``session`` `session_head`'s, the gate's
    open answer's {session, seq, root, commitment} alone, beside ``pass_index``."""
    man = lane.manifest(sid, base, 0)
    rng, s = lane.rng, man["session"]

    def file(path: str, digest: bool = True) -> dict:
        n = rng.randrange(1, 1 << 20)
        return {"path": path, "bytes": n, **({"sha256": rng.randbytes(32).hex()} if digest else {})}

    head = {"schema": man["schema"], "run": man["run"], "entry": "verity_vllm.LLM", "scheme": "pearl-c-sm120-v1-h3",
            "scheme_args": {"forming": "v1", "device": "sm120", "hashing": "h3"}, "coins": man["coins"],
            "salt_opening": man["salt_opening"], "salt": man["salt"],
            "session": {k: s[k] for k in ("session", "seq", "root", "commitment")}, "pass_index": man["pass_index"],
            "cubin_sha256": rng.randbytes(32).hex(), "kernel_source_sha256": rng.randbytes(32).hex(),
            "schedule": "deferred", "records_root": rng.randbytes(64).hex(),
            "commitment": {"root": rng.randbytes(64).hex(), "count": len(man["matmuls"])},
            "hm96_salt_key": rng.randbytes(32).hex(), "leaves": file("leaves"), "tree_t": file("tree_t")}
    weights = {w: {**v, "excluded_b": [], "rows": file(f"weights/{w}.rows.bf16", False),
                   "tree": file(f"weights/{w}.tree")} for w, v in man["weights"].items()}
    matmuls = [{**u, "k_kept": u["k"], "excluded_a": [], "root_a": rng.randbytes(64).hex(),
                "rows": file(f"calls/{u['index']}.rows.bf16", False), "tree": file(f"calls/{u['index']}.tree")}
               for u in man["matmuls"]]
    return (json.dumps({**head, "weights": weights, "matmuls": matmuls}, indent=1) + "\n").encode()


def test_a_pass_retained_in_vllms_form_is_accepted_and_credited_in_a_one_epoch_period():
    """A retained manifest exactly as the vLLM integration writes it passes `core_check.pass_epoch` under the live inputs (``live
    coin``), and its one-epoch period credits it."""
    lane = Lane("retained")
    base, w, _ = _base(lane.rng)
    lane.open("p-a")
    sid = lane.epoch(w, 0)
    lane.close()
    raw = _retained(lane, sid, base)
    ev = {"context": f"served-zk/r-retained/{hashlib.sha512(raw).hexdigest()}", "descriptor": DESCRIPTOR.hex(),
          "manifest": raw.hex(), "session": lane.sessions[sid][0].files()}
    key, ledger, pins, main = lane.args("p-a", [ev])[:4]
    live = {"key": key, "ledger": ledger, "main": main, "pins": pins, "session": ev["session"]}
    assert pc.CC.pass_epoch(None, raw, ev["context"], live=live, verify=rc.lms_verify) == ("live coin", None)
    got, why = Cr.pearl_c_period_credit(*lane.args("p-a", [ev]))
    assert why is None and got == {"epochs": [[w.hex(), 0]], "passes": [json.loads(raw)]}


def test_epochs_not_numbered_on_from_the_earlier_periods_refuse():
    """Clause 5: each weights' root's epochs continue from the lane's earlier credited periods: period B alone (its W
    from 2, not 0), period A again after itself (a repeat) and period B after an earlier list missing epoch 1 refuse."""
    lane, a, b, (w, _) = _two_periods("index")
    for earlier in ([], [[w.hex(), 0], [w.hex(), 1], [w.hex(), 2]], [[w.hex(), 0]]):
        _, why = Cr.pearl_c_period_credit(*lane.args("p-b", b, earlier))
        assert why[0] == Cr.INDEX and "with no gap or repeat" in why[1], earlier
    _, why = Cr.pearl_c_period_credit(*lane.args("p-a", a, [[w.hex(), 1]]))
    assert why[0] == Cr.INDEX and "is not epoch 2" in why[1]


def test_a_gap_in_k_spends_the_weights_root():
    """Epoch 1 over W opened outside every period (so never credited): the next period's epoch 2 over W is refused.
    The ledger opens epoch 1's root only once (`challenger.verify_ledger`), so no later period can credit it either,
    and W is spent."""
    lane = Lane("gap")
    base, w, n = _base(lane.rng)
    lane.open("p-a")
    s0 = lane.epoch(w, 0)
    lane.close()
    lane.epoch(w, 1)
    lane.open("p-b")
    s2 = lane.epoch(w, 2)
    lane.close()
    got, why = Cr.pearl_c_period_credit(*lane.args("p-a", [lane.pass_(s0, base)]))
    assert why is None
    _, why = Cr.pearl_c_period_credit(*lane.args("p-b", [lane.pass_(s2, base)], got["epochs"]))
    assert why[0] == Cr.INDEX and "is not epoch 1" in why[1] and "re-key hm96" in why[1]


def test_two_passes_of_one_epoch_sharing_a_call_index_refuse_the_period():
    """Clause 6 (`passes_distinct`): a call index two passes of one epoch hold repeats its seeds and tickets."""
    lane, a, _, _ = _two_periods("distinct")
    p0, _, p1 = a
    sid = json.loads(bytes.fromhex(p0["manifest"]))["session"]["session"]
    man = json.loads(bytes.fromhex(p0["manifest"]))
    man["run"] = "r-a-replay"
    _, why = Cr.pearl_c_period_credit(*lane.args("p-a", [p0, lane.evidence(sid, man), p1]))
    assert why[0] == Cr.REPEATED_CALL and "passes 0 and 1 both hold call index" in why[1]


def test_malformed_evidence_is_refused():
    lane, a, _, (w, _) = _two_periods("form")
    p0 = a[0]
    for passes in ([], None, [{**p0, "extra": 1}], [{k: v for k, v in p0.items() if k != "descriptor"}],
                   [{**p0, "manifest": "zz"}]):
        assert Cr.pearl_c_period_credit(*lane.args("p-a", passes)) == (None, ("period", Cr.PEARL_C_FORM)), passes
    for earlier in (None, [[w.hex()]], [[w.hex(), -1]], [[w.hex(), True]], [[w.hex().upper(), 0]], [[w.hex(), 1 << 64]]):
        assert Cr.pearl_c_period_credit(*lane.args("p-a", a, earlier)) == (None, (Cr.INDEX, Cr.EARLIER_FORM))
    man = json.loads(bytes.fromhex(p0["manifest"]))
    del man["pass_index"]
    _, why = Cr.pearl_c_period_credit(*lane.args("p-a", [{**p0, "manifest": json.dumps(man).encode().hex()}]))
    assert why[0] == Cr.PASS_REFUSED and "names no salt epoch" in why[1]
