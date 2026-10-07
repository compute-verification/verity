"""``rec/inner-transcript/v0``: the inner session of a recursive session (hiding model architectures), run by the unmodified
C-Flock prover against a verifier that never sees the hidden statement or the prover's messages.

    python -m verity_flock.rec_live --listen 127.0.0.1:PORT --out DIR [--sessions 1]

The prover (``flock-circuit prove --verifier ADDR`` under ``FC_COINS=os``) speaks the live protocol (``live/src/lib.rs``) to
:class:`Proxy`, which runs on the prover's side. The proxy answers what the live server would, except that every coin comes
from :class:`CoinServer`, the verifier's side, which is shown only the session's public data:

* ``Hello``: the schedule (profile, reps, flavor, tables, the link's shape), without Σ (a digest of the hidden statement);
* ``Commit``: each table's level-0 cap (``root_B``), answered with the link points; root_F (Σ) stays with the prover;
* each ``Round``: its caps (the ``BYTES`` frames that commit a Merkle tree), and ``c_k``, a hiding commitment to the round's
  free message, answered with the round's ``n`` coins.

A round's content is the transcript-v2 frames the prover absorbed since its last squeeze (PROTOCOL.md §6.1). `split_round`
sorts them: the domain and labels are public constants and are dropped; a ``BYTES`` frame is a cap (its bytes public), a
proof-of-work nonce (8 bytes, dropped: with live coins grinding buys nothing) or, in a rep's first round, the statement digest
(dropped: it is a function of the hidden statement); the observed field elements are the free message ``F_k``. ``F_k`` is
laid out as rows of :data:`ROW_WORDS` u16 words (each element 8 words, its 16 bytes lo LE then hi LE; the last row
zero-padded) and each row is committed as a registered row is (``verity.experimental.protocols.sampled_proofs.one_stage.registered``:
``hm96-sha512/row/v1`` under its own fresh 192-byte salt), so ``c_k`` is the list of the rows' ``b || c``. A round with no free
message has an empty ``c_k``. The rows and salts are the prover's: V*'s units open them against the registered value the
verifier builds from every ``c_k`` it received (`message_rows`, `commit_rows`).

At ``Finish`` the proxy writes the session directory (``session.json``, a ``flock-live-session/v1`` record that
``flock-circuit replay`` and the Lean verifier check, and ``<table>.rep<r>.bin``), the coin server's record ``public.json``
(``rec/inner-transcript/v0``: everything the verifier saw and issued) and the prover's ``private.json`` (every round's rows
and salts), and answers the prover with a verdict that says nothing was verified: the inner proof is V*'s witness.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from verity.ml.commitments import hm96

KIND = "rec/inner-transcript/v0"
SESSION_KIND = "flock-live-session/v1"

OP_DOMAIN, OP_LABEL, OP_OBSERVE, OP_SQUEEZE, OP_BYTES = 1, 2, 3, 4, 5
KIND_SCALAR, KIND_SLICE = 1, 2
LABEL_R1CS = b"flock-r1cs-v0"
NONCE_BYTES = 8
CAP_NODE_BYTES = 64

REQ_HELLO, REQ_LINK, REQ_OPEN, REQ_ROUND, REQ_PROOF, REQ_FINISH, REQ_COMMIT, REQ_PRIME, REQ_PING, REQ_REGISTER = range(1, 11)
RESP_OK, RESP_COINS, RESP_REFUSED, RESP_VERDICT, RESP_WORDS, RESP_DRAW = range(1, 7)

#: u16 words per message row: one SHA-512 block, 8 field elements
ROW_WORDS = 64
F_WORDS = 8
F_PER_ROW = ROW_WORDS // F_WORDS
MASK64 = (1 << 64) - 1


# --- wire codec (live/src/lib.rs `Req`, `Resp`) -------------------------------------------------------------------------------

class _Cur:
    def __init__(self, b: bytes, at: int = 1):
        self.b, self.at = b, at

    def u32(self) -> int:
        if self.at + 4 > len(self.b):
            raise ValueError("short frame")
        v = struct.unpack_from("<I", self.b, self.at)[0]
        self.at += 4
        return v

    def u64(self) -> int:
        if self.at + 8 > len(self.b):
            raise ValueError("short frame")
        v = struct.unpack_from("<Q", self.b, self.at)[0]
        self.at += 8
        return v

    def bytes(self) -> bytes:
        n = self.u32()
        if self.at + n > len(self.b):
            raise ValueError("short frame")
        v = self.b[self.at:self.at + n]
        self.at += n
        return v

    def str(self) -> str:
        return self.bytes().decode()


def decode_req(b: bytes) -> tuple:
    """``(tag, *fields)`` in ``Req``'s field order."""
    if not b:
        raise ValueError("empty frame")
    c, t = _Cur(b), b[0]
    if t == REQ_HELLO:
        return t, c.str()
    if t in (REQ_LINK, REQ_PING):
        return t, c.bytes()
    if t == REQ_OPEN:
        return t, c.str(), c.str(), c.bytes(), c.u64()
    if t == REQ_ROUND:
        return t, c.str(), c.bytes(), c.u32()
    if t == REQ_PROOF:
        return t, c.str(), c.u32(), c.bytes()
    if t in (REQ_FINISH, REQ_REGISTER):
        return (t,)
    if t == REQ_COMMIT:
        root_f = c.bytes()
        lists = []
        for _ in range(2):
            lists.append([(c.str(), c.bytes()) for _ in range(c.u32())])
        return t, root_f, lists[0], lists[1]
    if t == REQ_PRIME:
        return t, c.bytes(), c.u32()
    raise ValueError(f"unknown request tag {t}")


def _put_bytes(b: bytes) -> bytes:
    return struct.pack("<I", len(b)) + b


def encode_req(t: int, *f) -> bytes:
    """``Req::encode`` (tests drive the proxy with it)."""
    if t == REQ_HELLO:
        return bytes([t]) + _put_bytes(f[0].encode())
    if t in (REQ_LINK, REQ_PING):
        return bytes([t]) + _put_bytes(f[0])
    if t == REQ_OPEN:
        return bytes([t]) + _put_bytes(f[0].encode()) + _put_bytes(f[1].encode()) + _put_bytes(f[2]) + struct.pack("<Q", f[3])
    if t == REQ_ROUND:
        return bytes([t]) + _put_bytes(f[0].encode()) + _put_bytes(f[1]) + struct.pack("<I", f[2])
    if t == REQ_PROOF:
        return bytes([t]) + _put_bytes(f[0].encode()) + struct.pack("<I", f[1]) + _put_bytes(f[2])
    if t in (REQ_FINISH, REQ_REGISTER):
        return bytes([t])
    if t == REQ_COMMIT:
        out = bytes([t]) + _put_bytes(f[0])
        for lst in (f[1], f[2]):
            out += struct.pack("<I", len(lst)) + b"".join(_put_bytes(n.encode()) + _put_bytes(v) for n, v in lst)
        return out
    raise ValueError(f"no encoder for request tag {t}")


def f128_bytes(v: int) -> bytes:
    return struct.pack("<QQ", v & MASK64, v >> 64)


def f128_hex(v: int) -> str:
    """The record's form: hi then lo, each 16 hex digits (``f128_hex`` in lib.rs)."""
    return f"{v >> 64:016x}{v & MASK64:016x}"


def resp_ok() -> bytes:
    return bytes([RESP_OK])


def resp_coins(coins: list[int]) -> bytes:
    return bytes([RESP_COINS]) + struct.pack("<I", len(coins)) + b"".join(f128_bytes(c) for c in coins)


def resp_str(tag: int, s: str) -> bytes:
    return bytes([tag]) + _put_bytes(s.encode())


def decode_resp(b: bytes) -> tuple:
    c, t = _Cur(b), b[0]
    if t == RESP_OK:
        return (t,)
    if t == RESP_COINS:
        n = c.u32()
        return t, [c.u64() | (c.u64() << 64) for _ in range(n)]
    if t in (RESP_REFUSED, RESP_VERDICT, RESP_DRAW):
        return t, c.str()
    raise ValueError(f"unknown response tag {t}")


def read_frame(f) -> bytes | None:
    """One length-prefixed frame, or None at a clean end of stream."""
    head = f.read(4)
    if not head:
        return None
    if len(head) < 4:
        raise ValueError("truncated frame length")
    n = struct.unpack("<I", head)[0]
    b = f.read(n)
    if len(b) != n:
        raise ValueError("truncated frame")
    return b


def write_frame(f, b: bytes) -> None:
    f.write(struct.pack("<I", len(b)) + b)
    f.flush()


# --- a round's content ----------------------------------------------------------------------------------------------------------

def frames(content: bytes) -> list[tuple[int, int, int, bytes]]:
    """The transcript-v2 frames of a round's content: ``(op, kind, len, payload)``; a squeeze's payload is empty."""
    out, at = [], 0
    while at < len(content):
        if at + 16 > len(content):
            raise ValueError("a truncated frame header")
        op, kind, n = content[at], content[at + 1], struct.unpack_from("<Q", content, at + 8)[0]
        at += 16
        size = 16 * n if op == OP_OBSERVE else 0 if op == OP_SQUEEZE else n
        if op not in (OP_DOMAIN, OP_LABEL, OP_OBSERVE, OP_SQUEEZE, OP_BYTES):
            raise ValueError(f"unknown frame op {op}")
        if at + size > len(content):
            raise ValueError("a frame overruns its round")
        out.append((op, kind, n, content[at:at + size]))
        at += -(-size // 16) * 16
    return out


@dataclass
class Split:
    """A round sorted: its caps (public), its free message (field elements, the prover's), what was dropped, its coin count."""

    caps: list[bytes] = field(default_factory=list)
    free: list[int] = field(default_factory=list)
    nonces: list[bytes] = field(default_factory=list)
    digest: bytes | None = None
    n: int = 0


def split_round(content: bytes, first_of_rep: bool) -> Split:
    """`frames` sorted (the module docstring). ``first_of_rep``: the round binds the statement (a rep stream's first round:
    ``LABEL_R1CS``, the statement digest, then the level-0 cap)."""
    s, binding = Split(), None
    fr = frames(content)
    if not fr or fr[-1][0] != OP_SQUEEZE:
        raise ValueError("a round without its squeeze")
    for op, kind, n, p in fr:
        if op == OP_LABEL:
            if first_of_rep and p == LABEL_R1CS and binding is None:
                binding = 0
        elif op == OP_BYTES:
            if binding == 0:
                s.digest, binding = p, 1
            elif binding == 1:
                if len(p) % CAP_NODE_BYTES or not p:
                    raise ValueError(f"a level-0 cap of {len(p)} bytes")
                s.caps.append(p)
                binding = 2
            elif len(p) == NONCE_BYTES:
                s.nonces.append(p)
            elif p and len(p) % CAP_NODE_BYTES == 0:
                s.caps.append(p)
            else:
                raise ValueError(f"a BYTES frame of {len(p)} bytes is neither a cap nor a nonce")
        elif op == OP_OBSERVE:
            if (kind == KIND_SCALAR and n != 1) or kind not in (KIND_SCALAR, KIND_SLICE):
                raise ValueError(f"an observe of kind {kind} and length {n}")
            s.free += [int.from_bytes(p[16 * i:16 * i + 8], "little") | (int.from_bytes(p[16 * i + 8:16 * i + 16], "little") << 64)
                       for i in range(n)]
    if first_of_rep and binding != 2:
        raise ValueError("a rep's first round binds no statement digest and level-0 cap")
    s.n = fr[-1][2]
    return s


def message_rows(free: list[int]) -> list[list[int]]:
    """``F_k`` as rows of :data:`ROW_WORDS` u16 words: element ``i`` is words ``8 (i mod 8) ..`` of row ``i // 8``, its lo
    then hi 64 bits as little-endian u16 words; the last row zero-padded. No rows for an empty message."""
    words: list[int] = []
    for v in free:
        words += [(v >> (16 * j)) & 0xFFFF for j in range(F_WORDS)]
    words += [0] * (-len(words) % ROW_WORDS)
    return [words[i:i + ROW_WORDS] for i in range(0, len(words), ROW_WORDS)]


def row_elements(row: list[int]) -> list[int]:
    """`message_rows`' inverse on one row: its 8 field elements."""
    return [sum(row[F_WORDS * i + j] << (16 * j) for j in range(F_WORDS)) for i in range(F_PER_ROW)]


def commit_rows(rows: list[list[int]], salts: list[bytes]) -> list[bytes]:
    """Each row's ``b || c`` as a registered row's (``hm96-sha512/row/v1`` over ``sha512/row/v1``, role x)."""
    from verity.ml.commitments.rowleaf import ROLE_X, sha512_row_digest

    inst = hm96.SHA512
    return [inst.commit_string(inst.default_key, sha512_row_digest(r, 16, ROLE_X), s) for r, s in zip(rows, salts, strict=True)]


# --- the verifier's side ----------------------------------------------------------------------------------------------------------

class CoinServer:
    """The verifier's coin server: public data in, coins out, every exchange recorded (``record``). ``rand(n)`` returns n
    bytes (default the operating system's generator)."""

    def __init__(self, rand: Callable[[int], bytes] = os.urandom):
        self.rand = rand
        self.hello: dict | None = None
        self.link: dict | None = None
        self.streams: dict[str, dict] = {}
        self.rounds: list[dict] = []

    def _coins(self, n: int) -> list[int]:
        b = self.rand(16 * n)
        return [int.from_bytes(b[16 * i:16 * i + 8], "little") | (int.from_bytes(b[16 * i + 8:16 * i + 16], "little") << 64)
                for i in range(n)]

    def start(self, hello: dict) -> None:
        if self.hello is not None:
            raise ValueError("a second Hello")
        if "sigma" in hello.get("link", {}):
            raise ValueError("the coin server is never shown Σ")
        self.hello = hello

    def commit(self, roots: dict[str, bytes]) -> list[int]:
        """Each table's level-0 cap in; the link points out (``points × m`` coins)."""
        if self.hello is None or self.link is not None:
            raise ValueError("the link points are drawn once, after Hello")
        spec = self.hello["link"]
        pts = self._coins(spec["points"] * spec["m"])
        self.link = {"roots": {t: r.hex() for t, r in sorted(roots.items())}, "points": [f128_hex(c) for c in pts], "y": None}
        return pts

    def y(self, y: list[int]) -> None:
        if self.link is None or self.link["y"] is not None:
            raise ValueError("y follows the link points, once")
        self.link["y"] = [f128_hex(v) for v in y]

    def open(self, stream: str, parent: str, at: int) -> None:
        if stream in self.streams:
            raise ValueError(f"stream {stream} opened twice")
        self.streams[stream] = {"parent": parent, "at": at, "rounds": 0}

    def round(self, stream: str, caps: list[bytes], commits: list[bytes], n: int) -> list[int]:
        if self.link is None or self.link["y"] is None:
            raise ValueError("a coin before the link exchange")
        st = self.streams.get(stream)
        if st is None:
            raise ValueError(f"a round on stream {stream}, never opened")
        coins = self._coins(n)
        self.rounds.append({"stream": stream, "k": st["rounds"], "caps": [c.hex() for c in caps], "commits": [c.hex() for c in commits],
                            "n": n, "coins": [f128_hex(c) for c in coins]})
        st["rounds"] += 1
        return coins

    def record(self) -> dict:
        return {"kind": KIND, "hello": self.hello, "link": self.link, "streams": self.streams, "rounds": self.rounds,
                "message_rows": sum(len(r["commits"]) for r in self.rounds), "row_words": ROW_WORDS}


# --- the prover's side ------------------------------------------------------------------------------------------------------------

def _dg(b: bytes) -> str:
    return hashlib.sha512(b).hexdigest()


class Proxy:
    """The live server's part on the prover's side: answers the unmodified prover, takes every coin from ``coins`` (shown only
    public data), and keeps what the session record and V*'s witness need."""

    def __init__(self, coins: CoinServer, fresh_salt: Callable[[], bytes] = hm96.SHA512.fresh_salt):
        self.coins = coins
        self.fresh_salt = fresh_salt
        self.hello: str | None = None
        self.hello_v: dict | None = None
        self.commit: tuple | None = None
        self.points: list[list[int]] = []
        self.y: bytes | None = None
        self.streams: dict[str, dict] = {}
        self.proofs: dict[tuple[str, int], bytes] = {}
        self.private: list[dict] = []
        self.verdict: dict | None = None

    def handle(self, req: bytes) -> bytes:
        try:
            return self._handle(decode_req(req))
        except ValueError as e:
            return resp_str(RESP_REFUSED, f"rec proxy: {e}")

    def _handle(self, r: tuple) -> bytes:
        t = r[0]
        if t == REQ_PING:
            return resp_ok()
        if self.verdict is not None:
            raise ValueError("session finished")
        if t == REQ_REGISTER:
            return resp_str(RESP_DRAW, "null")
        if t == REQ_HELLO:
            self.hello = r[1]
            self.hello_v = json.loads(r[1])
            if "coins" in self.hello_v:
                raise ValueError("the inner session runs on live OS coins (FC_COINS=os): no seed, no coin tree")
            pub = json.loads(r[1])
            pub.get("link", {}).pop("sigma", None)
            self.coins.start(pub)
            return resp_ok()
        if t == REQ_COMMIT:
            _, root_f, roots, publics = r
            if publics and any(v for _, v in publics):
                raise ValueError("public words in an inner session")
            self.commit = (root_f, roots, publics)
            pts = self.coins.commit(dict(roots))
            m = self.hello_v["link"]["m"]
            self.points = [pts[i:i + m] for i in range(0, len(pts), m)]
            return resp_coins(pts)
        if t == REQ_LINK:
            self.y = r[1]
            self.coins.y([int.from_bytes(self.y[i:i + 8], "little") | (int.from_bytes(self.y[i + 8:i + 16], "little") << 64)
                          for i in range(0, len(self.y), 16)])
            return resp_ok()
        if t == REQ_OPEN:
            _, stream, parent, label, at = r
            self.coins.open(stream, parent, at)
            self.streams[stream] = {"parent": parent, "label": label, "at": at, "rounds": [], "cap": None}
            return resp_ok()
        if t == REQ_ROUND:
            _, stream, content, n = r
            st = self.streams.get(stream)
            if st is None:
                raise ValueError(f"a round on stream {stream}, never opened")
            k = len(st["rounds"])
            s = split_round(content, k == 0 and not st["parent"])
            if s.n != n:
                raise ValueError(f"the round squeezes {s.n} coins, the prover asked for {n}")
            if k == 0 and not st["parent"]:
                st["cap"] = s.caps[0]
            rows = message_rows(s.free)
            salts = [self.fresh_salt() for _ in rows]
            bcs = commit_rows(rows, salts)
            coins = self.coins.round(stream, s.caps, bcs, n)
            st["rounds"].append({"content": content, "n": n, "coins": coins})
            self.private.append({"stream": stream, "k": k, "elements": [f128_hex(v) for v in s.free], "rows": rows,
                                 "salts": [x.hex() for x in salts], "nonces": [x.hex() for x in s.nonces]})
            return resp_coins(coins)
        if t == REQ_PROOF:
            _, table, rep, b = r
            self.proofs[(table, rep)] = b
            return resp_ok()
        if t == REQ_FINISH:
            self.verdict = {"accepted": None, "why": "rec/inner-transcript/v0: the inner proof is V*'s witness; nothing verified here"}
            return resp_str(RESP_VERDICT, json.dumps(self.verdict))
        raise ValueError(f"request tag {t} has no place in an inner session")

    def session_record(self) -> dict:
        """The ``flock-live-session/v1`` record the live server would have written (``Server::record_json``): what
        ``Server::from_record`` replays."""
        root_f, roots, publics = self.commit
        link = {"mode": "exchange", "sigma": self.hello_v["link"]["sigma"], "points": [[f128_hex(c) for c in p] for p in self.points],
                "points_g": None, "y": [f128_hex(int.from_bytes(self.y[i:i + 8], "little") | (int.from_bytes(self.y[i + 8:i + 16], "little") << 64))
                                        for i in range(0, len(self.y), 16)],
                "root_f": root_f.hex(), "root_b": {t: v.hex() for t, v in roots}, "publics": {t: v.hex() for t, v in publics},
                "root_f_sha512": _dg(root_f), "root_b_sha512": {t: _dg(v) for t, v in roots},
                "publics_sha512": {t: _dg(v) for t, v in publics}}
        lb = root_f + b"".join(v for _, v in sorted(roots)) + b"".join(f128_bytes(c) for p in self.points for c in p) + self.y
        streams = []
        for name in sorted(self.streams):
            st = self.streams[name]
            streams.append({"stream": name, "parent": st["parent"], "label": st["label"].decode("utf-8", "replace"), "at": st["at"],
                            "rounds": [{"k": k, "g": None, "n": rd["n"], "msg_len": len(rd["content"]),
                                        "coins": [f128_hex(c) for c in rd["coins"]], "msg_sha512": _dg(rd["content"]),
                                        "msg": rd["content"].hex()} for k, rd in enumerate(st["rounds"])],
                            "root_sha512": _dg(st["cap"]) if not st["parent"] else None})
        h = self.hello_v
        return {"kind": SESSION_KIND, "hello": self.hello, "link": link, "link_mode": "exchange",
                "config": {"profile": h["profile"], "reps": h["reps"], "flavor": h["flavor"], "require_link": True, "prime_live": False,
                           "tables": [{"name": t, "binding": "R1cs (rec proxy: the statement digest is the prover's)"} for t in h["tables"]]},
                "unit_draw": None, "aborted": None, "verdict": self.verdict, "streams": streams, "prime": None, "coin_tree": None,
                "coin_seed": None, "link_sha512": _dg(lb),
                "proof_sha512": [{"stream": f"{t}/rep{r}", "bytes": len(b), "sha512": _dg(b)} for (t, r), b in sorted(self.proofs.items())]}

    def write(self, out: Path) -> Path:
        """The session directory (``session.json``, ``<table>.rep<r>.bin``), ``public.json`` and ``private.json``."""
        out = Path(out)
        sess = out / "session"
        sess.mkdir(parents=True, exist_ok=True)
        (sess / "session.json").write_text(json.dumps(self.session_record(), indent=1))
        for (t, r), b in self.proofs.items():
            (sess / f"{t}.rep{r}.bin").write_bytes(b)
        (out / "public.json").write_text(json.dumps(self.coins.record(), indent=1))
        (out / "private.json").write_text(json.dumps({"kind": KIND + "/private", "rounds": self.private}))
        return sess


def serve(listen: str, out: Path, rand: Callable[[int], bytes] = os.urandom, ready: Callable[[str], None] | None = None,
          sessions: int = 1, done: Callable[[dict], None] | None = None) -> list[dict]:
    """Accept connections on ``listen`` until ``sessions`` of them have each carried a whole session (``Finish``), write each
    (`Proxy.write`: into ``out``, or ``out/s<k>`` when there are several, a prover's ``--warm`` and ``--runs`` sessions, one
    connection each) and return their summaries, each also passed to ``done`` as it finishes. ``ready`` is called with the
    bound address."""
    host, port = listen.rsplit(":", 1)
    res: list[dict] = []
    with socket.create_server((host, int(port))) as srv:
        h, p = srv.getsockname()[:2]
        addr = f"{h}:{p}"
        if ready:
            ready(addr)
        while len(res) < sessions:
            conn, _ = srv.accept()
            proxy = Proxy(CoinServer(rand))
            with conn, conn.makefile("rb") as rf, conn.makefile("wb") as wf:
                while (req := read_frame(rf)) is not None:
                    write_frame(wf, proxy.handle(req))
                    if proxy.verdict is not None:
                        break
            if proxy.verdict is not None:
                sess = proxy.write(out if sessions == 1 else Path(out) / f"s{len(res)}")
                rec = proxy.coins.record()
                res.append({"session": str(sess), "rounds": len(rec["rounds"]), "message_rows": rec["message_rows"],
                            "coins": sum(r["n"] for r in rec["rounds"]), "caps": sum(len(r["caps"]) for r in rec["rounds"]),
                            "proof_bytes": {f"{t}/rep{r}": len(b) for (t, r), b in proxy.proofs.items()}})
                if done:
                    done(res[-1])
    return res


def main(argv=None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="127.0.0.1:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sessions", type=int, default=1, help="sessions to serve, one connection each (several: out/s<k>/)")
    a = ap.parse_args(argv)
    serve(a.listen, Path(a.out), ready=lambda addr: print(f"SERVING {addr}", flush=True), sessions=a.sessions,
          done=lambda res: print("REC_INNER\t" + json.dumps(res), flush=True))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
