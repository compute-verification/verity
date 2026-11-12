"""The connection rows' difftest's data: random instances, and `connection_rows.py`'s results on each, for
`NetworkCheck/ConnectionRowsDifftest.lean`.

    python scripts/connection_rows_vectors.py <verity checkout> --out DIR [--count N]

Writes `DIR/connection-rows/instances.jsonl`, one instance per line as `NetworkCheck.ConnectionRowsDifftest.evaluate`
reads it, and `DIR/connection-rows/python.jsonl`, the same line's results from the checkout's
`verity.protocols.certifier.verifier.connection_rows` (and `content_rows`, `warden` and `certificate` through it), with
`null` where Python raises `ValueError`. The Lean audit runs `NetworkCheck/ConnectionRowsDifftestMain.lean` on both
files (`runs` in `lean-audit.json`, which pins their SHA-256), and
`verity/protocols/certifier/verifier/tests/test_network_certifier_connection_rows_lean_difftest.py` checks that this
script still writes them.

An instance has a schedule with lengths rows of either form, with content classes or none, about one in four broken in
one way `WardenParams` or `content_rows.classes` refuses (a cell an id request's first prompt field doesn't fit,
`ingress_buckets` past the delay, a connection that outlives a window, a window that isn't whole strides, a row of too
many slots, a response cap that is no cap, a form that is none, a class of the other form, a size that is no power of
two, ids past a response row's, rows that are none or past the lengths table's slots). Its queries: `request_room` and
`request_bytes` at cells an id request does and doesn't fit, `response_bytes`, domains of the schedule's tables and of
others at any count, heads built field by field (a table listed twice or none, a digest of another length, a count or an
execution phase past its field), and registrations of the schedule's tables signed by an LMS key of the instance: the
honest one, with every count from none to the most under form 2, and ones changed in one place. A change to what the
head lists is signed again, so the registration is refused for what it states and not for its certificate: a class's
count past an execution phase's releases, a class's three tables of two counts, a table of no row with a root, a table
of the schedule's rows with another count. The others leave the certificate as it was, or change it: a field, a key, a
table's entry or layout of another value, missing or extra; a certificate of another digest, key id, session, link or
execution phase, or with a signature that doesn't decode or verify.

Every registration has the values' types `connection_rows.registration` writes: hex in lower case, integers written as
integers, a certificate of three keys. Where `check_registration` reads more than that (`bytes.fromhex`'s upper case
and spaces, a number equal to the integer, a certificate's further keys) the Lean refuses, and no instance has one.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from functools import lru_cache
from pathlib import Path

import verity.protocols.certifier.verifier
from verity.commitments.crypto import lms
from verity.protocols.certifier.verifier import connection_rows as CR
from verity.protocols.certifier.verifier import content_rows as CT
from verity.protocols.certifier.verifier.certificate import Certificate, Direction, LmsKey, certificate_message
from verity.protocols.certifier.verifier.warden import WardenParams

COUNT = 240
#: the instances' LMS parameters: the deployed N on a tree of 2^6 leaves, W = 4 for a short signature
LMS = lms.Params(64, 4, 6)
#: the instances that share a key, and the signatures an instance takes of it: a leaf signs once
SHARE, SIGNED = 16, 4
LINKS = ["responses", "eth0", "", "uplink/0", "é-link", "链路", "tab\there", "q\"uote\\"]
NAMES = ["lengths", "fails", "request.0", "response.0", "end.0", "request.99", "", "é", "链", "end.1"]


@lru_cache(maxsize=4)
def key(n: int) -> lms.PrivateKey:
    return lms.keygen(LMS, hashlib.sha512(b"connection-rows-difftest/%d" % n).digest())


def _hex(rng: random.Random, n: int = 64) -> str:
    return rng.randbytes(n).hex()


def _flip(text: str, at: int = 0) -> str:
    """`text`, a hex string, with the digit at `at` another."""
    return text[:at] + ("0" if text[at] != "0" else "1") + text[at + 1:]


def schedule(rng: random.Random) -> tuple[dict, int | None, list | None]:
    """A schedule's integer fields, its lengths rows' form and its classes: what `WardenParams` takes, or about one
    time in four broken in one way."""
    every = rng.choice([1, 1, 2, 2, 4, 8])
    window = every * rng.randint(2, 12)
    delay = rng.randint(1, min(10, window))
    slots = rng.randint(1, 4)
    s = {"bucket_ms": 100, "cell_bytes": rng.choice([387, 400, 512, 512, 600, 1024, 4096, rng.randint(387, 700)]),
         "delay": delay, "egress_cells": rng.randint(1, min(64, (window - delay + 1) * slots)), "every": every,
         "ingress_buckets": rng.randint(1, delay), "ingress_cells": rng.randint(1, 8), "lanes": rng.randint(1, 4),
         "queue": rng.randint(1, 4), "slots": slots, "window": window}
    if s["egress_cells"] > 1 and rng.random() < 0.15:
        s["max_tokens_cap"] = rng.randint(1, s["egress_cells"] - 1)
    lengths = rng.choice([1, 2, 2])
    held = lengths == 2
    most = window // every * WardenParams(**s, lengths=lengths).stride_releases
    content = None
    if rng.random() < 0.85:
        content = [[rng.choice([2, 4, 8, 16, 64, 128, 256]), rng.choice([2, 4, 8, 64, 4096])]
                   + ([] if held else [rng.randint(1, most)]) for _ in range(rng.randint(1, 4))]
    if rng.random() < 0.25:
        breaks = ["cell", "ingress", "span", "slots", "cap", "form", "none"]
        breaks += ["stride"] * (every > 1) + ["class"] * 3 * bool(content) + ["empty"]
        how = rng.choice(breaks)
        if how == "cell":
            s["cell_bytes"] = rng.choice([11, 100, 139, 300, 386])
        elif how == "ingress":
            s["ingress_buckets"] = delay + rng.randint(1, 3)
        elif how == "span":
            s["egress_cells"] = (window - delay + 1) * slots + rng.randint(1, 5)
        elif how == "slots":
            s["lanes"] = 2731 + rng.randint(0, 50)
        elif how == "cap":
            s["max_tokens_cap"] = s["egress_cells"] + rng.randint(0, 2)
        elif how == "form":
            lengths = rng.choice([0, 3, 4])
        elif how == "none":
            lengths = None
        elif how == "stride":
            s["window"] = window + rng.randint(1, every - 1)
        elif how == "empty":
            content = []
        else:
            c = rng.choice(content)
            what = rng.choice(["P", "O", "ids", "big", "form"] + ["rows"] * (not held))
            if what == "P":
                c[0] = rng.choice([0, 1, 3, 6, 100])
            elif what == "O":
                c[1] = rng.choice([0, 1, 3, 6, 100])
            elif what == "ids":
                c[1] = rng.choice([8192, 65536])
            elif what == "big":
                c[0] = 1 << 16
            elif what == "form":
                c[:] = c[:2] + [1] if held else c[:2]
            else:
                c[2] = rng.choice([0, most + 1, most + 7])
    return s, lengths, content


def _maybe(f, *args):
    try:
        return f(*args)
    except ValueError:
        return None


def _head_result(head) -> dict | None:
    return None if head is None else {"encoding": head.encode().hex(), "digest": head.digest().hex()}


def _given(h: dict) -> dict | None:
    """The result for a head given field by field."""
    def build():
        entries = tuple(CR.Entry(t["name"], t["rows"], t["row_bytes"], bytes.fromhex(t["domain"]),
                                 bytes.fromhex(t["root"])) for t in h["tables"])
        return _head_result(CR.Head(h["link"], h["window"], bytes.fromhex(h["parameters"]), entries))
    return _maybe(build)


def _a_head(rng: random.Random) -> dict:
    """A head field by field: one that encodes, its counts up to their fields' most, or half the time one with one
    flaw."""
    def count(bits: int, usual: list[int]) -> int:
        return rng.choice(usual * 3 + [(1 << bits) - 1])
    names = rng.sample(NAMES, rng.choice([1, 2, 3, 5, 8]))
    h = {"link": rng.choice(LINKS), "window": count(64, [0, 1, 5, 1 << 40]), "parameters": _hex(rng),
         "tables": [{"name": name, "rows": count(32, [0, 1, 8, 600]), "row_bytes": count(32, [4, 20, 1802]),
                     "domain": _hex(rng), "root": _hex(rng)} for name in names]}
    if rng.random() < 0.5:
        t = rng.choice(h["tables"])
        flaw = rng.choice(["parameters", "domain", "root", "rows", "row_bytes", "window", "twice", "none"] * 4
                          + ["link", "name"])
        if flaw in ("parameters", "domain", "root"):
            (h if flaw == "parameters" else t)[flaw] = _hex(rng, rng.choice([0, 63, 65]))
        elif flaw in ("rows", "row_bytes"):
            t[flaw] = 1 << 32
        elif flaw == "window":
            h["window"] = 1 << 64
        elif flaw == "twice":
            h["tables"].append(dict(rng.choice(h["tables"]), root=_hex(rng)))
        elif flaw == "none":
            h["tables"] = []
        else:
            (h if flaw == "link" else t)[flaw] = "x" * 65_536
    return h


def _class(name: str) -> int | None:
    """The class a content table's name is of, none for the lengths and fails tables."""
    return int(name.partition(".")[2]) if "." in name else None


class Signing:
    """An instance's signatures: its share of its key's leaves, each used once."""

    def __init__(self, seed: int) -> None:
        self.key = key(seed // SHARE)
        self.next = seed % SHARE * SIGNED
        self.left = SIGNED

    def certificate(self, head: CR.Head, session: bytes, *, link: str | None = None,
                    window: int | None = None) -> Certificate:
        """`head`'s digest signed in the session; `link` and `window` are what the signed message names when it is
        another than the head's."""
        if not self.left:
            raise RuntimeError(f"an instance signs at most {SIGNED} times")
        q, self.next, self.left = self.next, self.next + 1, self.left - 1
        message = certificate_message(self.key.I, session, head.link if link is None else link, Direction.EGRESS,
                                      head.window if window is None else window, head.digest())
        return Certificate(self.key.I, head.link, Direction.EGRESS, head.window, head.digest(),
                           lms.sign_at(self.key, q, message).encode())


def _entries(rng: random.Random, params: WardenParams, session: bytes, link: str, window: int,
             rows_of: dict[str, int]) -> tuple[CR.Entry, ...]:
    """The head's entries with each table's rows from `rows_of`, its domain derived at them and a random root, the
    zero root where it has no row."""
    return tuple(CR.Entry(name, rows_of[name], size,
                          CR.domain(params, session, link, window, name, rows_of[name], size).domain_id,
                          rng.randbytes(64) if rows_of[name] else CR.NO_ROWS_ROOT)
                 for name, _, size in CR.tables_of(params))


def registrations(rng: random.Random, params: WardenParams, session: bytes, link: str, window: int,
                  signing: Signing) -> list[dict]:
    """The honest registration of the schedule's tables at random counts, then ones changed in one place."""
    listed = CR.tables_of(params)
    held, R = CT.counted(params), params.phase_releases
    counts = {}
    rows_of = {}
    for name, rows, _ in listed:
        i = _class(name)
        if held and i is not None:
            rows = counts.setdefault(i, rng.choice([0, 0, 1, R, rng.randint(0, R)]))
        rows_of[name] = rows

    def doc(rows: dict[str, int], change=None, **signed) -> dict:
        entries = _entries(rng, params, session, link, window, rows)
        if change:
            entries = change(entries)
        head = CR.Head(link, window, params.digest, entries)
        return CR.registration(head, signing.certificate(head, session, **signed), params)

    honest = doc(rows_of)
    out = [honest]
    content = [name for name, _, _ in listed if _class(name) is not None]
    stated = ["rows"] + (["over", "split", "rooted", "unrooted"] if held and content else []) + ["unrooted"]
    for how in rng.sample(stated, 2):
        rows = dict(rows_of)
        change = None
        if how == "rows":
            name = rng.choice([n for n, _, _ in listed if not (held and _class(n) is not None)])
            rows[name] += rng.choice([-1, 1])
        elif how == "over":
            i = _class(rng.choice(content))
            rows |= {n: R + rng.randint(1, 3) for n in CT.names(i)}
        elif how == "split":
            name = rng.choice([n for n in content if not n.startswith(CT.REQUEST)])
            rows[name] = rows[name] + 1 if rows[name] < R else rows[name] - 1
        elif how == "rooted":
            name = rng.choice(content)
            rows |= {n: 0 for n in CT.names(_class(name))}
            change = lambda es: tuple(CR.Entry(e.name, e.rows, e.row_bytes, e.domain, rng.randbytes(64))  # noqa: E731
                                      if e.name == name else e for e in es)
        else:
            name = rng.choice([n for n in rows if rows[n]])
            change = lambda es: tuple(CR.Entry(e.name, e.rows, e.row_bytes, e.domain, CR.NO_ROWS_ROOT)  # noqa: E731
                                      if e.name == name else e for e in es)
        out.append(doc(rows, change))

    def changed(how: str) -> dict:
        d = copy.deepcopy(honest)
        t = rng.choice(d["tables"])
        c = d["certificate"]
        if how == "field":
            k = rng.choice(["format", "link", "window", "parameters", "word_bits", "row_schema"])
            d[k] = d[k] + 1 if isinstance(d[k], int) else _flip(d[k]) if k == "parameters" else d[k] + "x"
        elif how == "key":
            if rng.random() < 0.5:
                d["extra"] = 0
            else:
                del d[rng.choice(sorted(d))]
        elif how == "tables":
            d["tables"] = rng.choice([d["tables"][:-1], d["tables"] + d["tables"][-1:], d["tables"][::-1], {}, None])
        elif how == "entry":
            k = rng.choice(["name", "value", "rows", "words", "domain", "root", "root", "layout"])
            if k == "root":
                t[k] = rng.choice([_flip(t[k], rng.randrange(128)), t[k][:-2], t[k] + "00", "zz" + t[k][2:], 7])
            elif k == "layout":
                field = rng.choice(sorted(t[k]))
                t[k] = rng.choice([t[k] | {field: t[k][field] + 1}, t[k] | {"extra": 0},
                                   {f: v for f, v in t[k].items() if f != field}, [], "layout"])
            elif k == "rows":
                t[k] = rng.choice([str(t[k]), -1, None, [t[k]]])
            else:
                t[k] = t[k] + 1 if isinstance(t[k], int) else _flip(t[k]) if k == "domain" else t[k] + "x"
        elif how == "shape":
            at = d["tables"].index(t)
            gone = rng.choice(sorted(t))
            d["tables"][at] = rng.choice([t | {"extra": 0}, {k: v for k, v in t.items() if k != gone}, "table", None])
        elif how == "certificate":
            k = rng.choice(["digest", "signature", "signature", "key"])
            at = rng.randrange(len(c[k]))
            d["certificate"] = rng.choice([c | {k: _flip(c[k], at)}, c | {k: c[k][:-2]}, c | {k: "zz" + c[k][2:]},
                                           {f: v for f, v in c.items() if f != k}, None, c | {k: 0}])
        return d

    for how in rng.sample(["field", "key", "tables", "entry", "entry", "shape", "certificate", "certificate"], 2):
        out.append(changed(how))
    how = rng.choice(["session", "link", "window"])
    if how == "session":
        entries = _entries(rng, params, session, link, window, rows_of)
        head = CR.Head(link, window, params.digest, entries)
        out.append(CR.registration(head, signing.certificate(head, bytes(reversed(session))), params))
    else:
        out.append(doc(rows_of, **({"link": link + "x"} if how == "link" else {"window": window + 1})))
    return out


def instance(seed: int) -> tuple[dict, dict]:
    """Instance `seed` and the Python's results on it."""
    rng = random.Random(seed)
    s, lengths, content = schedule(rng)
    session, link = rng.randbytes(32), rng.choice(LINKS)
    window = rng.choice([0, 1, 3, 17, 1 << 33])
    signing = Signing(seed)

    def built():
        params = WardenParams(**s, lengths=lengths, content=None if content is None else tuple(map(tuple, content)))
        return params, CR.tables_of(params)

    made = _maybe(built)
    rooms = [[rng.choice([20, 100, 115, 116, 128, 200, 256, 386, 387, 512, 4096, rng.randint(10, 700)]),
              rng.choice([0, 1, 2, 3, 4, 8, 16, 100, 128, 1024])] for _ in range(6)]
    ids = [rng.choice([0, 1, 2, 3, 4, 6, 8, 1024, 4096, 8192, 65536]) for _ in range(4)]
    heads = [_a_head(rng) for _ in range(3)]
    domains, docs = [], []
    if made:
        params, listed = made
        for _ in range(4):
            name, rows, size = rng.choice(listed)
            domains.append(rng.choice([[name, rows, size], [name, 0, size], [name, rng.randint(0, 1 << 20), size],
                                       [rng.choice(NAMES), rng.randint(0, 1 << 20), rng.randint(1, 5000)]]))
        docs = registrations(rng, params, session, link, window, signing)
    inst = {"schedule": s, "lengths": lengths, "content": content,
            "parameters": (made[0].digest if made else bytes(64)).hex(), "session": session.hex(), "link": link,
            "window": window, "key": lms.public_key(signing.key).hex(), "lms": {"N": LMS.N, "W": LMS.W, "H": LMS.H},
            "rooms": rooms, "ids": ids, "domains": domains, "heads": heads, "registrations": docs}
    public = LmsKey(LMS, lms.public_key(signing.key))
    results = {
        "tables": [list(t) for t in made[1]] if made else None,
        "rooms": [_maybe(lambda B, P: [CT.request_room(B, P), CT.request_bytes(B, P)], B, P) for B, P in rooms],
        "ids": [_maybe(CT.response_bytes, O) for O in ids],
        "domains": [CR.domain(made[0], session, link, window, *q).domain_id.hex() for q in domains],
        "heads": [_given(h) for h in heads],
        "registrations": [_head_result(_maybe(
            lambda d: CR.check_registration(d, public, session, made[0], link, window)[0], d)) for d in docs]}
    return inst, results


def line(obj: dict) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False) + "\n"


def write(out: Path, count: int = COUNT) -> dict[str, Path]:
    """Both files for instances 0 … count − 1 under ``out/connection-rows``."""
    d = out / "connection-rows"
    d.mkdir(parents=True, exist_ok=True)
    paths = {f"connection-rows/{name}": d / name for name in ("instances.jsonl", "python.jsonl")}
    with (open(paths["connection-rows/instances.jsonl"], "w", encoding="utf-8") as fi,
          open(paths["connection-rows/python.jsonl"], "w", encoding="utf-8") as fp):
        for seed in range(count):
            inst, results = instance(seed)
            fi.write(line(inst))
            fp.write(line(results))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("verity", type=Path, help="the checkout whose verity.protocols.certifier.verifier is tested")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=COUNT)
    a = ap.parse_args()
    here = Path(verity.protocols.certifier.verifier.__file__).resolve()
    if not here.is_relative_to(a.verity.resolve()):
        raise SystemExit(f"verity.protocols.certifier.verifier comes from {here}, not from the checkout {a.verity}")
    write(a.out, a.count)


if __name__ == "__main__":
    main()
