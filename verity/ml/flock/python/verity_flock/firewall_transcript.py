"""The firewall's transcript (``verity/firewall-transcript/v0``): what the inner phase's firewall (`rec_live.Proxy`) sends
its challenger (the coin server), in order, each item tagged with the case of the firewall's contract it falls under, with
the firewall's own decision beside it. ``flock-firewall contract`` (``Flock.Firewall.Contract.run`` in the Lean verifier)
decides the same transcript, held to the verifier's public input (``verity/firewall-public/v0``, `public`), never to
anything the transcript says of itself; `firewall_agree.py` compares the two.

The recorder watches the firewall from outside: a coin server that keeps every call made to it (`RecordingCoins`), a salt
source that keeps every salt it hands out (`Tape`), and every answer the firewall gives the workers (`Session.handle`). It
splits each call into what the schedule fixes (its shape: ``fixed`` items) and the commitments it carries (``commit``
items, one a node of a salted top or a message row, each with the message the firewall's own records say it commits to;
a round's copies of a level-0 top carry none, Commit committed them), and each coin answer into
the challenger's coins, the salt it sent with them to open its commitment (when it committed to them), and the coins the
workers received (``coins``). A salt any answer to the workers contains is
exposed. Nothing here trusts the firewall's checks: a call it should have refused is in the transcript as it was made.
"""
from __future__ import annotations

import json
from collections.abc import Callable

from verity.ml.commitments import hm96
from verity.ml.commitments.rowleaf import ROLE_X, row_bytes, sha512_row_prefix

from . import rec_live as RL

FORMAT = "verity/firewall-transcript/v0"
PUBLIC_FORMAT = "verity/firewall-public/v0"
INST = hm96.SHA512


def public(schedule, relations=(), statement=None, computes=(), coins=None) -> dict:
    """The verifier's public input `flock-firewall contract --public` reads (`Flock.Firewall.Contract.PublicInput`): the
    session's schedule (or ``None``), the relations the verifier checks on masked values, the values the public statement
    fixes (name to bytes), the names of the values the firewall fixes by its own computation, and the challenger's
    commitments to its coins, if it made any (`rec_live.CoinServer.commit_coins`: hex, per stream and round)."""
    out = {"kind": PUBLIC_FORMAT, "schedule": schedule, "relations": list(relations),
           "statement": {k: v.hex() for k, v in (statement or {}).items()}, "computes": list(computes)}
    if coins is not None:
        out["coins"] = {s: list(cs) for s, cs in coins.items()}
    return out


def canon(v) -> bytes:
    """`Flock.canon`'s form: compact, keys sorted, every character outside printable ASCII escaped, integers only (a float
    here makes the item differ from the schedule's)."""
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def row_message(row: list[int]) -> bytes:
    """The bytes a registered row's ``sha512/row/v1`` digest hashes (role x, 16-bit words): its prefix block, then the row."""
    return sha512_row_prefix(ROLE_X, 16, len(row)) + row_bytes(row, 16)


class Tape:
    """The firewall's salt source, keeping each salt it hands out, in order: salt number ``i`` is ``salts[i]``."""

    def __init__(self, fresh: Callable[[], bytes] = INST.fresh_salt):
        self.fresh = fresh
        self.salts: list[bytes] = []

    def __call__(self) -> bytes:
        s = self.fresh()
        self.salts.append(s)
        return s


class RecordingCoins(RL.CoinServer):
    """The coin server, keeping every call the firewall makes to it as it was made (before the coin server judges it); a
    call the coin server refused also carries its reason (``refused``)."""

    def __init__(self, rand=None):
        super().__init__(rand) if rand else super().__init__()
        self.calls: list[dict] = []

    def _call(self, c: dict, f, *a):
        self.calls.append(c)
        try:
            return f(*a)
        except Exception as e:
            c["refused"] = str(e)
            raise

    def start(self, hello):
        return self._call({"call": "hello", "hello": hello}, super().start, hello)

    def commit(self, roots):
        c = {"call": "roots", "roots": dict(roots), "coins": None}
        c["coins"] = self._call(c, super().commit, roots)
        c["opening"] = self.openings.get((RL.LINK_COINS, 0))
        return c["coins"]

    def y(self, y):
        return self._call({"call": "y", "y": list(y)}, super().y, y)

    def open(self, stream, parent, at):
        return self._call({"call": "open", "stream": stream, "parent": parent, "at": at}, super().open, stream, parent, at)

    def round(self, stream, tops, commits, n, level0=0):
        st = self.streams.get(stream)
        c = {"call": "round", "stream": stream, "k": st["rounds"] if st else None, "tops": [list(t) for t in tops],
             "level0": level0, "commits": list(commits), "n": n, "coins": None}
        c["coins"] = self._call(c, super().round, stream, tops, commits, n, level0)
        c["opening"] = self.openings.get((stream, c["k"]))
        return c["coins"]


class Session:
    """One inner session through a firewall: ``handle`` answers the workers' requests as the firewall does and keeps what
    the transcript needs. ``proxy`` is the firewall under test (`rec_live.Proxy` or a subclass). The challenger's coins are
    live, as `rec_live.serve`'s are, unless ``commit``: then it commits to them before the session
    (`rec_live.CoinServer.commit_coins`, each commitment under a salt from ``coin_salt``, the challenger's own, never the
    firewall's tape). ``coins`` is the challenger under test (`RecordingCoins` or a subclass)."""

    def __init__(self, schedule: dict, rand=None, fresh: Callable[[], bytes] = INST.fresh_salt, proxy=RL.Proxy,
                 coin_salt: Callable[[], bytes] = INST.fresh_salt, commit: bool = False, coins=None):
        self.schedule = schedule
        self.coins = (coins or RecordingCoins)(rand)
        self.tape = Tape(fresh)
        self.commitments = self.coins.commit_coins(schedule, coin_salt) if commit else None
        self.proxy = proxy(self.coins, self.tape, schedule=schedule, commitments=self.commitments)
        self.answers: list[bytes] = []
        #: per coin-issuing call (by its index in ``coins.calls``), the coins the workers received
        self.relayed: dict[int, list[int]] = {}
        self.stray: list[list[int]] = []

    def handle(self, req: bytes) -> bytes:
        before = len(self.coins.calls)
        resp = self.proxy.handle(req)
        self.answers.append(resp)
        for c in self.coins.calls[before:]:
            if c["call"] == "roots":
                # the firewall's roots record as of this call: a later Commit overwrites it
                c["mine"] = {t: dict(v) for t, v in self.proxy.roots.items()}
        d = RL.decode_resp(resp)
        if d[0] == RL.RESP_COINS:
            new = [i for i in range(before, len(self.coins.calls)) if self.coins.calls[i].get("coins") is not None]
            if new:
                self.relayed[new[-1]] = list(d[1])
            else:
                self.stray.append(list(d[1]))
        return resp

    def accepted(self) -> bool:
        """The firewall's decision: the session ended with Finish, and the coin server's record says so."""
        return self.coins.end == "finish" and self.proxy.stopped is None

    def public(self) -> dict:
        """The inner session's public input: its schedule and the challenger's commitments to its coins, if it committed to
        them, and no relation, statement value or value of the firewall's own (the inner phase masks nothing and fixes
        everything it sends by the schedule)."""
        return public(self.schedule, coins=self.commitments)

    def transcript(self) -> dict:
        tape = {s: i for i, s in reversed(list(enumerate(self.tape.salts)))}
        unknown = len(self.tape.salts)
        items: list[dict] = []
        hexs = lambda xs: [RL.f128_hex(v) for v in xs]

        def commits(sent: list[bytes], rows: list[list[int]], salts: list[str]):
            for j, ch in enumerate(list(sent) or [b""]):
                msg = row_message(rows[j]) if j < len(rows) else b""
                sid = tape.get(bytes.fromhex(salts[j]), unknown) if j < len(salts) else unknown
                items.append({"commit": {"msg": msg.hex(), "salt": sid, "sent": ch.hex()}})

        def coins(stream: str, k: int, i: int, c: dict):
            item = {"stream": stream, "round": k, "coins": hexs(c["coins"] or []), "relayed": hexs(self.relayed.get(i, []))}
            if c.get("opening") is not None:
                item["opening"] = c["opening"].hex()
            items.append({"coins": item})

        priv = iter(self.proxy.private)
        for i, c in enumerate(self.coins.calls):
            kind = c["call"]
            if kind == "hello":
                items.append({"fixed": {"what": "hello", "by": "schedule", "sent": canon(c["hello"]).hex()}})
            elif kind == "roots":
                items.append({"fixed": {"what": "roots", "by": "schedule",
                                        "sent": canon({t: len(r) for t, r in c["roots"].items()}).hex()}})
                for t, r in sorted(c["roots"].items()):
                    mine = c.get("mine", self.proxy.roots).get(t, {})
                    commits(r, RL.top_rows(bytes.fromhex("".join(mine["nodes"]))) if mine else [], mine.get("salts", []))
                coins(RL.LINK_COINS, 0, i, c)
            elif kind == "y":
                items.append({"fixed": {"what": "link.y", "by": "schedule", "sent": b"".join(RL.f128_bytes(v) for v in c["y"]).hex()}})
            elif kind == "open":
                items.append({"fixed": {"what": f"open:{c['stream']}", "by": "schedule",
                                        "sent": canon({"at": c["at"], "parent": c["parent"]}).hex()}})
            elif kind == "round":
                r = len([x for x in self.coins.calls[:i] if x["call"] == "round"])
                items.append({"fixed": {"what": f"round:{r}", "by": "schedule",
                                        "sent": canon({"stream": c["stream"], "k": c["k"], "n": c["n"], "tops": [len(x) for x in c["tops"]],
                                                       "level0": c["level0"], "commits": len(c["commits"])}).hex()}})
                p = next(priv, None) or {}
                clear = p.get("tops", [])
                for j, top in enumerate(c["tops"]):
                    o = clear[j] if j < len(clear) else {}
                    commits(top, RL.top_rows(bytes.fromhex("".join(o["nodes"]))) if o else [], o.get("salts", []))
                rows, salts = p.get("rows", []), p.get("salts", [])
                for j, bc in enumerate(c["commits"]):
                    commits([bc], rows[j:j + 1], salts[j:j + 1])
                coins(c["stream"], c["k"] if c["k"] is not None else 0, i, c)
        for s in self.stray:
            items.append({"coins": {"stream": "", "round": 0, "coins": [], "relayed": hexs(s)}})
        if self.coins.end == "finish":
            rec = self.coins.record()
            items.append({"fixed": {"what": "finish", "by": "schedule",
                                    "sent": canon({"rounds": len(rec["rounds"]), "streams": {s: v["rounds"] for s, v in rec["streams"].items()}}).hex()}})
            items.append({"finish": {}})
        answers = b"".join(self.answers)
        exposed = [i for i, s in enumerate(self.tape.salts) if s in answers or s.hex().encode() in answers]
        return {"kind": FORMAT, "phase": "inner",
                "tape": {"salts": [s.hex() for s in self.tape.salts], "pads": []}, "exposed": {"salts": exposed, "pads": []},
                "items": items, "firewall": {"accepted": self.accepted(), "why": self.proxy.stopped}}
