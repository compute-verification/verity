"""Core credit's statement steps over a served H1 window with receipts on (SERVEH1): `pouw_h1_gpu.py --credit`'s check.

A window's ``receipt@<c>``, ``pub@<c>`` and ok rows are bound only where credit verifies a drawn call's statement
(`credit._verified`): on the ``--registered`` file credit builds itself from the descriptor and the window's entries
(`credit.registered_file`: port o0 at the call's registered receipt table, ``pub`` and o1 at rows of known content
credit computes), then `credit._proved` and `credit._registration` over the outcomes. `verdict` runs those steps, after
`credit._a_link` and (with the Commit) `credit._arow`, as `credit._window` orders them, over the window's own evidence.

The verifier of record is a stand-in (`StandIn`), as in core's `test_pouw_credit_statements.py`: there is no Flock
proof of a served K = 4096 unit here. A session's "proof" is the prover's openings of each sampled proof block's ``pub``, o0 and
o1 rows (each row's b || c, its path and, for o0, its salt) from the prover's own tables (`Prover`). The stand-in
accepts a session only when each opening verifies against the root credit's file gives that port, at the row the file
reads (`registered.verify_read`, the check Lean makes of a registered port), and the opened o0 and o1 rows are the
unit's outputs: o0 `h1tile.reference`'s receipt of its A and B rows seeded from credit's public row, o1 the one word 1
under its public ok salt. The A and B rows are the serving GPU's (`operands`), whose binding is `_arow`'s and the
weights', not the stand-in's. Units are chosen by the caller: in-process coins select none.

The negatives tamper with the prover's tables after an honest window (`NEGATIVES`): a receipt word flipped (the table
re-registered and re-listed under its round's root, so the round roots commit it), the same with the round roots left
honest, a public table built under another step's A root, and a unit's ok row 0."""
from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import credit as CR
from verity.core.protocols.pouw.circuit import h1tile as HT
from verity.core.protocols.sampled_proofs.one_stage import proofs as PF
from verity.core.protocols.sampled_proofs.one_stage import registered as G

from verity_vllm.protocols.pouw.h1 import registered as NR
from verity_vllm.protocols.pouw.h1 import served as SV

#: the prover-side tamperings `negatives` applies, each to be refused
NEGATIVES = ("receipt-word", "receipt-word-after-round", "pub-other-root", "ok-zero")


def bc_of(words: Any, salt: bytes) -> bytes:
    """A row/v2 row's b || c under ``salt`` (`registered.Registered.commit`'s)."""
    return G._INST.commit_string(G._INST.default_key, G.row_digest([int(w) for w in words], G.WORD_BITS, G.SCHEMA_V2), salt)


def _root(dom: Any, position: int, bc: bytes, path: bytes) -> bytes:
    """The root ``bc`` at ``position`` reaches through ``path`` under ``dom`` (`registered.verify_read`'s walk)."""
    digest, i = G.leaf(dom, position, bc, G.SCHEMA_V2), position
    for level in range(len(path) // 64):
        sib = path[64 * level:64 * level + 64]
        digest = dom.node(level, i >> 1, *((digest, sib) if i % 2 == 0 else (sib, digest)))
        i >>= 1
    return digest


@dataclass
class Window:
    """What credit reads of a closed window (``raw`` its descriptor, ``entries`` and ``rounds`` its evidence, ``commit``
    the Commit record's registered entries or None), the prover's tables (``tables`` by value, ``ok`` its ok rows) and
    the session round 0's input-ids and request tables bind (the gate's reserved one, `pouw_h1_served.LOCAL_SESSION` under
    in-process coins)."""

    raw: bytes
    entries: list[dict[str, Any]]
    rounds: list[dict[str, Any]]
    commit: list[dict[str, Any]] | None
    window: str
    salt: str
    tables: dict[str, Any]
    ok: Any
    session: str = ""
    program: str = ""
    layout: dict = field(default_factory=dict)
    shape: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.program = hashlib.sha512(self.raw).hexdigest()
        layout, why = CR.layout_of(self.raw)
        if why is not None:
            raise ValueError(f"the window's descriptor has no layout: {why}")
        self.layout, self.shape = layout, CR.shape_of(self.raw)

    @property
    def steps(self) -> int:
        return len(self.rounds)

    @property
    def prefix(self) -> str:
        return self.shape["prefix"]

    def u0(self, c: int) -> int:
        return sum(n for n, _w in CR.call_work(self.layout)[:c])

    def call_of(self, u: int) -> int:
        at = 0
        for c, (n, _w) in enumerate(CR.call_work(self.layout)):
            if u < at + n:
                return c
            at += n
        raise ValueError(f"unit {u} is past the window's {at} units")

    def k_of(self, c: int) -> int:
        return {name: k for name, k, _n in self.layout["linears"]}[self.layout["calls"][c][0]]


def of_run(h1: Any, served: Any) -> Window:
    """The `Window` of a closed `pouw_h1.H1Run` ``h1`` and its `pouw_h1_served.Served` ``served``."""
    got = CR.commit_of(h1.commits.get("commit"))[0] if h1.commits.get("commit") is not None else None
    return Window(SV.descriptor(served.program), [dict(e) for e in served.entries()], [dict(r) for r in h1.round_log],
                  None if got is None else got[1], served.root, h1.salt.hex(), dict(served.registered), served.ok,
                  h1.session or SV.LOCAL_SESSION)


#: ``(call, A [M, K] s8, B [N, K] s8)``: a drawn call's operands, as the serving GPU held them
Operands = Callable[[int], tuple[np.ndarray, np.ndarray]]


class StandIn:
    """The stand-in verifier of record (the module docstring): `__call__` is `credit._verified`'s ``run``."""

    def __init__(self, w: Window, operands: Operands):
        self.w, self.operands = w, operands
        self.rejected: list[str] = []

    def _dom(self, port: dict) -> Any:
        dom = G.domain(self.w.program, port["value"], port["words"], port["leaves"],
                       window=None if CR.is_held(port["value"]) else self.w.window)
        if dom.domain_id.hex() != port["domain"]:
            raise ValueError(f"port {port['value']}'s domain is not the registered one")
        return dom

    def unit(self, reg: dict, c: int, u: int, opened: dict) -> str | None:
        """Why unit ``u`` of call ``c``'s session is rejected, or None."""
        ports, t = reg["ports"], u - self.w.u0(c)
        pub, o0, o1 = ports[CR.PUB_PORT], ports.get(CR.RECEIPT_PORT), ports[CR.OK_PORT]
        if o0 is None:
            return "the file reads no receipt table at o0"
        for name, port in ((CR.PUB_PORT, pub), (CR.RECEIPT_PORT, o0), (CR.OK_PORT, o1)):
            i, (bc, path) = u - port.get(G.READS_FROM, 0), (bytes.fromhex(x) for x in opened[name][:2])
            at = port["reads"][i] if 0 <= i < len(port["reads"]) else None
            if at is None or not G.verify_read(self._dom(port), bytes.fromhex(port["root"]), at, bc, path, G.SCHEMA_V2):
                return f"unit {u}'s {name} row does not open {port['value']} at the file's root"
        if bytes.fromhex(opened[CR.PUB_PORT][0]) != bc_of(pub["known"]["rows"][t], bytes.fromhex(pub["known"]["salts"][t])):
            return f"unit {u}'s public row is not credit's"
        salt, position, roots, j10r = _fields(pub["known"]["rows"][t])
        a, b = self.operands(c)
        r, cc = self.w.layout["tile"]
        i, j = divmod(t, b.shape[0] // cc)
        ta, tb = (a[r * i:r * (i + 1)].view(np.uint8)).tolist(), (b[cc * j:cc * (j + 1)].view(np.uint8)).tolist()
        if j10r is None:
            _z, receipt = HT.reference(ta, tb, salt, position, roots, pairs=[(0, 0)])
        else:
            _z, receipt = HT.j10r_reference(ta, tb, salt, position, roots, *j10r, pairs=[(0, 0)])
        want = SV.receipt_words(np.array([receipt], np.uint32))[0]
        if bytes.fromhex(opened[CR.RECEIPT_PORT][0]) != bc_of(want, bytes.fromhex(opened[CR.RECEIPT_PORT][2])):
            return f"unit {u}'s o0 row is not the receipt it computes"
        if bytes.fromhex(opened[CR.OK_PORT][0]) != bc_of(CC.OK_ROW, CC.ok_salt(self.w.window, u)):
            return f"unit {u}'s o1 row is not the ok = 1 it computes"
        return None

    def __call__(self, proof: PF.Proof, *, circuit_sha512: str, program: bytes, partition: Any, registered: dict
                 ) -> tuple[list[PF.Finding] | None, tuple[str, str] | None]:
        if hashlib.sha512(proof.circuit.read_bytes()).hexdigest() != circuit_sha512:
            return None, ("circuit", "the statement's circuit file is not the committed one")
        if program != self.w.raw or partition["program"] != self.w.program:
            return None, ("proof", "verified against another program")
        out = []
        for s in proof.sessions:
            rec = json.loads((s / "session.json").read_text())
            c, units = rec["call"], tuple(rec["units"])
            why = next((x for u in units if (x := self.unit(registered, c, u, rec["opened"][str(u)])) is not None), None)
            if why is not None:
                self.rejected.append(f"call {c}: {why}")
            out.append(PF.Finding(str(s), why is None, why or "", self.w.program if why is None else None,
                                  {p: {"value": e["value"], "root": e["root"]} for p, e in registered["ports"].items()}
                                  if why is None else {}, units if why is None else ()))
        return out, None


def _fields(words: Any) -> tuple[bytes, int, tuple[bytes, bytes], tuple[int, tuple[int, int]] | None]:
    """A public row's salt, position, (root(A), root(B)) and a J10-R row's (call, blocks), None for a v1 row
    (`core_check.public_words` inverted)."""
    x = b"".join(int(v).to_bytes(2, "little") for v in words)
    x = b"".join(x[i:i + 4][::-1] for i in range(0, len(x), 4))
    v1, s = x[:CC.PUBLIC_ROW_BITS // 8], CC.PUBLIC_ROW_BITS // 8 - 8 - 128
    j10r = None
    if len(x) * 8 == CC.PUBLIC_ROW_J10R_BITS:
        call, b0, b1 = (int.from_bytes(x[len(v1) + 8 * i:len(v1) + 8 * (i + 1)], "big") for i in range(3))
        j10r = call, (b0, b1)
    return v1[:s], int.from_bytes(v1[s:s + 8], "big"), (v1[s + 8:s + 72], v1[s + 72:]), j10r


def statements(w: Window, units: list[int], to: Path) -> tuple[list[dict], dict[str, str]]:
    """The prover's statements of ``units`` (one a drawn call, its session's openings from the prover's tables) under
    ``to``, and the period-open's circuit files (``{k<K>: SHA-512}``) of the stand-in's circuit files."""
    files, out, by_call = {}, [], {}
    for u in sorted(units):
        by_call.setdefault(w.call_of(u), []).append(u)
    for c, us in sorted(by_call.items()):
        k, base = w.k_of(c), to / f"call-{c}"
        (base / "session").mkdir(parents=True)
        circuit = f"stand-in circuit file of K = {k}'s template\n".encode()
        (base / "circuit.txt").write_bytes(circuit)
        files[f"k{k}"] = hashlib.sha512(circuit).hexdigest()
        (base / "pub.bin").write_bytes(json.dumps({"units": us}).encode())
        pub, rc = w.tables[f"{w.prefix}pub@{c}"], w.tables[f"{w.prefix}receipt@{c}"]
        opened = {}
        for u in us:
            t = u - w.u0(c)
            opened[str(u)] = {CR.PUB_PORT: [pub.bcs[t].hex(), pub.path(t).hex()],
                              CR.RECEIPT_PORT: [rc.bcs[t].hex(), rc.path(t).hex(), bytes(rc.salts[t]).hex()],
                              CR.OK_PORT: [w.ok.bcs[u].hex(), w.ok.path(u).hex()]}
        (base / "session" / "session.json").write_text(json.dumps({"call": c, "units": us, "opened": opened}))
        out.append({"call": c, "circuit": str(base / "circuit.txt"), "public": str(base / "pub.bin"),
                    "sessions": [str(base / "session")]})
    return out, files


def verdict(w: Window, units: list[int], operands: Operands) -> dict[str, Any]:
    """Credit's steps over ``w``, ``units`` drawn: ``{step: None or (code, why)}`` in `credit._window`'s order, the first
    refusal stopping the rest, and ``accepted`` when none refused."""
    out: dict[str, Any] = {"a_link": CR._a_link(w.rounds, w.entries, len(w.layout["calls"]))}
    if w.commit is not None and out["a_link"] is None:
        out["arow"] = CR._arow(w.rounds, w.entries, w.commit, w.layout, w.prefix)
    stand_in = StandIn(w, operands)
    if all(v is None for v in out.values()):
        with tempfile.TemporaryDirectory() as d:
            sts, files = statements(w, units, Path(d))
            proofs, why = CR._verified(sts, files, w.raw, w.layout, w.entries, w.window, units, w.steps, stand_in, w.salt)
        out["verified"] = why
        if why is None:
            out["proved"] = CR._proved(units, proofs)
        if why is None and out["proved"] is None:
            ss, _ = CR.strata(w.layout)
            out["registration"] = CR._registration(w.window, w.program, w.prefix, w.entries, ss, w.layout, w.steps, proofs,
                                                   w.session or SV.LOCAL_SESSION)
    out["rejected"] = stand_in.rejected
    out["accepted"] = all(v is None for k, v in out.items() if k not in ("rejected",))
    out["refused"] = next(([k, *v] for k, v in out.items() if k not in ("rejected", "accepted") and v is not None), None)
    return out


def _entry(w: Window, value: str, reg: Any) -> list[dict[str, Any]]:
    return [reg.entry() if e["value"] == value else e for e in w.entries]


def _relist(w: Window, value: str, reg: Any, own: int) -> list[dict[str, Any]]:
    """The round log with round ``own`` listing ``reg``'s segments for ``value`` and its root over them (`credit.round_root`),
    as a prover committing ``reg`` before that round's coin would have recorded it."""
    rounds = [dict(r) for r in w.rounds]
    r = rounds[own]
    a = {**r["a"], value: [[lv, ix, reg.levels[lv][ix].hex()] for lv, ix in CR.cover(0, reg.leaves)]}
    rounds[own] = {**r, "a": a, "root": CR.round_root(own, bytes.fromhex(r["inner"]), CR._segments(a)).hex()}
    return rounds


def tampered(w: Window, kind: str, u: int, receipts: Callable[[int], np.ndarray], key: bytes) -> Window:
    """``w`` with the prover's tables changed at sampled proof block ``u`` (`NEGATIVES`): ``receipt-word``, its call's receipt table
    with u's first word flipped, re-registered under its nonce and re-listed under its round's root (``receipts(c)`` the
    call's honest receipt words, ``key`` the run's); ``receipt-word-after-round``, the same table with the round roots
    left honest; ``pub-other-root``, its call's public table built under another step's A table of its K;
    ``ok-zero``, the window's ok rows with u's row 0."""
    c, pre = w.call_of(u), w.prefix
    t = u - w.u0(c)
    if kind.startswith("receipt-word"):
        rc = f"{pre}receipt@{c}"
        words = np.array(receipts(c), np.uint16)
        words[t, 0] ^= 1
        reg = NR.commit(w.program, rc, words, key, SV.RECEIPT_NONCE + c, window=w.window)
        rounds = w.rounds if kind == "receipt-word-after-round" else _relist(
            w, rc, reg, c // (len(w.layout["calls"]) // w.steps))
        return replace(w, tables={**w.tables, rc: reg}, entries=_entry(w, rc, reg), rounds=rounds)
    if kind == "pub-other-root":
        k, per = w.k_of(c), len(w.layout["calls"]) // w.steps
        s = c // per
        other = f"{pre}a/{k}@{(s + 1) % w.steps}"
        pv, n = f"{pre}pub@{c}", CR.call_work(w.layout)[c][0]
        _rows, _salts, reg = CR.public_table(w.program, pv, w.salt, w.window, w.u0(c), n,
                                             (w.tables[other].root.hex(), w.tables[f"{pre}b/{k}"].root.hex()))
        return replace(w, tables={**w.tables, pv: reg}, entries=_entry(w, pv, reg))
    if kind == "ok-zero":
        return replace(w, ok=_OkRowZero(w.ok, u, bc_of([0], CC.ok_salt(w.window, u))))
    raise ValueError(f"no negative {kind!r}")


@dataclass(frozen=True)
class _OkRowZero:
    """The window's ok rows with row ``u`` 0: its b || c that row's, every path the honest tree's."""

    ok: Any
    u: int
    bc: bytes

    @property
    def bcs(self) -> Any:
        return _Bcs(self.ok.bcs, self.u, self.bc)

    def path(self, position: int) -> bytes:
        return self.ok.path(position)


@dataclass(frozen=True)
class _Bcs:
    bcs: Any
    u: int
    bc: bytes

    def __getitem__(self, i: int) -> bytes:
        return self.bc if i == self.u else self.bcs[i]


def check(w: Window, units: list[int], operands: Operands, receipts: Callable[[int], np.ndarray], key: bytes,
          at: int | None = None) -> dict[str, Any]:
    """`verdict` of the honest window and of each of `NEGATIVES` at sampled proof block ``at`` (default ``units[0]``): ``ok`` when
    the honest window is accepted and every negative refused."""
    honest = verdict(w, units, operands)
    u = units[0] if at is None else at
    negs = {kind: verdict(tampered(w, kind, u, receipts, key), units, operands) for kind in NEGATIVES}
    return {"units": list(units), "at": u, "honest": honest, "negatives": negs,
            "ok": bool(honest["accepted"] and all(not v["accepted"] for v in negs.values()))}


def lines(got: dict[str, Any]) -> list[str]:
    """The accept and refuse lines of `check`'s outcome."""
    def say(v: dict[str, Any]) -> str:
        if v["accepted"]:
            return "ACCEPT"
        step, code, why = v["refused"]
        return f"REFUSE at {step} ({code}): {why[:400]}" + (f" [stand-in: {v['rejected'][0]}]" if v["rejected"] else "")
    return [f"credit honest: {say(got['honest'])}"] + [f"credit {k}: {say(v)}" for k, v in got["negatives"].items()]


__all__ = ["NEGATIVES", "StandIn", "Window", "bc_of", "check", "lines", "of_run", "statements", "tampered", "verdict"]
