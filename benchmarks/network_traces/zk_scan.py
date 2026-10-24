"""A ZK driver's leak scan: what the verifier received, scanned for the witness, less the bytes the statement fixes.

    import zk_scan                                    # benchmarks/network_traces on sys.path
    rc = zk_scan.received(vdir, control, blobs, held, circuit)
    ok = not rc["leaked"] and rc["control"]["holds_all"]

**The rule** (REDTEAM phase 18, every ZK driver's): a witness piece found in what the verifier received is excluded if
and only if its value occurs, raw or as hex, in the statement-determined bytes the verifier receives (`public_runs`);
every other piece found is leaked and fails the run. No piece is excluded by its content (a byte count, a class of
witness): a decision bit serialized into a zero piece is as leaked as a salt is. The statement-determined bytes are runs,
each matched within itself, so a value straddling two runs, or a run and a message, isn't in them:

* the committed circuit's text (the caller's ``circuit.txt``) and the public rows (``pub-*.bin``), whole;
* each transcript round message's framing (`framing`: ``rounds[*].msg`` of a session record, hex-decoded; C-Flock's
  ``verity/ml/flock/verifier/PROTOCOL.md`` §6.1): each record's 16-byte header, the domain's and labels' bytes and every
  padding, not the field elements or observed bytes, which are the prover's messages;
* each proof stream's (``*.repK.bin``) container fields (`format_runs`, §8): the integers its layout reads that are no
  message (vector lengths, the parameters) and each ring switch's grinding nonce, which §8.3 and §12.2's O4 fix at 0;
  not its elements, hashes, salts, paths, zk words or any other nonce.

**Matching.** A blob is cut into aligned pieces of `PIECE` bytes, its last `PIECE` bytes too when its length isn't a
multiple (`pieces`), and a blob shorter than `PIECE` (a carry row) is one piece, whole; each distinct value is scanned
once, under its first name. A piece is found where its bytes occur at any offset, or its hex (either case) at an even
offset into a run of hex digits (`find`): a hex match straddling nibbles encodes no piece and isn't counted. Every file
the verifier received is scanned, the transcript as the proof streams are. The prover's instance file (``control``) is
to hold every held piece, so a scan that finds nothing has shown it finds them.
"""
from __future__ import annotations

import json
import re
import struct
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PIECE = 16
#: §6.1's record ops: domain, label, field elements, squeeze, observed bytes
DOMAIN, LABEL, ELEMENTS, SQUEEZE, BYTES = 1, 2, 3, 4, 5
_HEX = np.zeros(256, dtype=bool)
_HEX[list(b"0123456789abcdefABCDEF")] = True


def offsets(n: int) -> list[int]:
    """Where a blob of `n` bytes is cut into pieces: every `PIECE` bytes, and its last `PIECE` when `n` isn't a multiple
    (0 alone under `PIECE`)."""
    return [0] if n <= PIECE else list(range(0, n - PIECE + 1, PIECE)) + ([n - PIECE] if n % PIECE else [])


def pieces(blobs: dict[str, bytes], held: Iterable[str] = ()) -> tuple[dict[str, bytes], set[str]]:
    """`blobs` cut into the scan's pieces (`offsets`), each distinct value under its first name, ``"<blob> @<offset>"``;
    and the held ones, the pieces of a blob in `held`."""
    out: dict[str, bytes] = {}
    seen: set[bytes] = set()
    for name, b in blobs.items():
        if len(b) < 8:
            raise ValueError(f"{name}: {len(b)} bytes, under the scan's 8-byte key")
        for o in offsets(len(b)):
            x = b[o:o + PIECE]
            if x not in seen:
                seen.add(x)
                out[f"{name} @{o}"] = x
    held = set(held)
    return out, {k for k in out if k.rsplit(" @", 1)[0] in held}


def needles(pieces_: dict[str, bytes]) -> dict[bytes, list[tuple[str, bool]]]:
    """Each piece raw and as hex in both cases, to its (name, whether hex)."""
    out: dict[bytes, list[tuple[str, bool]]] = {}
    for name, x in pieces_.items():
        for nd, hx in ((x, False), (x.hex().encode(), True), (x.hex().upper().encode(), True)):
            if (name, hx) not in out.get(nd, []):
                out.setdefault(nd, []).append((name, hx))
    return out


def find(data: bytes, table: dict[bytes, list[tuple[str, bool]]]) -> set[str]:
    """The names whose needles occur in `data`: raw at any offset, hex at an even offset into its run of hex digits
    (keyed on each needle's first 8 bytes)."""
    if not table or len(data) < 8:
        return set()
    lens = sorted({len(nd) for nd in table})
    keys = np.unique(np.fromiter((int.from_bytes(nd[:8], "little") for nd in table), dtype=np.uint64, count=len(table)))
    a, out, starts = np.frombuffer(data, dtype=np.uint8), set(), None
    for s in range(8):
        k = (len(a) - s) // 8
        if k <= 0:
            continue
        for at in np.flatnonzero(np.isin(a[s:s + 8 * k].view("<u8"), keys)):
            off = s + 8 * int(at)
            for n in lens:
                for name, hx in table.get(data[off:off + n], ()):
                    if hx:
                        if starts is None:
                            h = _HEX[a]
                            starts = np.flatnonzero(h & ~np.concatenate(([False], h[:-1])))
                        if (off - int(starts[np.searchsorted(starts, off, "right") - 1])) % 2:
                            continue
                    out.add(name)
    return out


def framing(msg: bytes) -> list[bytes]:
    """A transcript round message's framing runs (§6.1): headers, the domain's and labels' bytes and padding, contiguous
    ones joined; the field elements and observed bytes are messages. A record that doesn't parse ends it: the rest is
    message."""
    runs, cur, i = [], bytearray(), 0

    def cut():
        if cur:
            runs.append(bytes(cur))
            cur.clear()

    while i + 16 <= len(msg):
        op, kind, n = msg[i], msg[i + 1], int.from_bytes(msg[i + 8:i + 16], "little")
        pad = -n % 16
        body = {DOMAIN: n + pad, LABEL: n + pad, ELEMENTS: 16 * n, SQUEEZE: 0, BYTES: n + pad}.get(op)
        if (body is None or msg[i + 2:i + 8] != bytes(6) or i + 16 + body > len(msg)
                or (kind != 0) != (op in (ELEMENTS, SQUEEZE)) or (op == ELEMENTS and kind == 1 and n != 1)):
            break
        cur += msg[i:i + 16]
        i += 16
        if op in (DOMAIN, LABEL):
            cur += msg[i:i + body]
        elif op in (ELEMENTS, BYTES):
            cut()
            if op == BYTES:
                cur += msg[i + n:i + body]
        i += body
    cut()
    return runs


def _layout():
    here = str(REPO / "verity/ml/flock/verifier")
    if here not in sys.path:
        sys.path.insert(0, here)
    import redigest
    return redigest


def format_mask(stream: bytes) -> bytes:
    """A ``--zk`` proof stream's container fields, 1 a byte (`format_runs`); all 0 when C-Flock's layout
    (`redigest.layout`) doesn't parse it."""
    RD, readers = _layout(), []

    class Reads(RD.tc.R):
        def __init__(s, b, hlen=32):
            super().__init__(b, hlen)
            s.fmt, s.ints = bytearray(len(b)), False
            readers.append(s)

        def take(s, n):
            at, v = s.i, super().take(n)
            if s.ints:
                s.fmt[at:at + n] = b"\1" * n
            return v

        def _int(s, read):
            s.ints = True
            try:
                return read()
            finally:
                s.ints = False

        def u8(s): return s._int(super().u8)
        def u32(s): return s._int(super().u32)
        def u64(s): return s._int(super().u64)

    R, RD.tc.R = RD.tc.R, Reads
    try:
        off = RD.layout(stream, zk=True)
    except (AssertionError, ValueError, struct.error):
        return bytes(len(stream))
    finally:
        RD.tc.R = R
    mask = next(r for r in reversed(readers) if r.i == len(stream)).fmt
    for key, ats in off.items():
        if re.fullmatch(r"rs\.\d+\.nonce", key):
            for at in ats:
                mask[at:at + 8] = b"\1" * 8
    return bytes(mask)


def format_runs(stream: bytes) -> list[bytes]:
    """A ``--zk`` proof stream's container fields (§8) as runs: the bytes its layout reads as integers that are no message
    (`redigest.layout`'s unkeyed reads: vector lengths, the parameters) and each ring switch's grinding nonce, which §8.3
    and §12.2's O4 fix at 0. [] when the layout doesn't parse it."""
    m = np.frombuffer(format_mask(stream), dtype=np.uint8).astype(bool)
    edges = np.flatnonzero(np.diff(np.concatenate(([False], m, [False]))))
    return [stream[s:e] for s, e in zip(edges[::2], edges[1::2])]


def public_runs(vdir: Path, circuit: bytes = b"", extra: Iterable[bytes] = ()) -> dict[str, list[bytes]]:
    """The statement-determined bytes the verifier received, by kind (the module's docstring): ``circuit`` (the committed
    circuit's text, `circuit`, which the verifier holds by its digest rather than under `vdir`), and under `vdir`
    ``pub`` (``pub-*.bin``), ``framing`` (each session record's rounds) and ``format`` (each ``*.repK.bin``); and
    `extra`, runs a caller knows public (a rescan's circuit pieces, when the circuit wasn't kept)."""
    out: dict[str, list[bytes]] = {"circuit": [bytes(circuit)] if circuit else [], "pub": [], "framing": [],
                                   "format": [], "extra": list(extra)}
    for f in sorted(x for x in vdir.iterdir() if x.is_file()):
        if re.fullmatch(r"pub-\d+\.bin", f.name):
            out["pub"].append(f.read_bytes())
        elif re.search(r"\.rep\d+\.bin$", f.name):
            out["format"] += format_runs(f.read_bytes())
        elif f.name.endswith("session.json"):
            rec = json.loads(f.read_text())
            for st in rec.get("streams") or []:
                for r in st.get("rounds") or []:
                    if r.get("msg"):
                        out["framing"] += framing(bytes.fromhex(r["msg"]))
    return out


def received(vdir: Path, control: Path | None, blobs: dict[str, bytes], held: Iterable[str], circuit: bytes = b"",
             public: Iterable[bytes] = ()) -> dict:
    """Every file under `vdir` (what the verifier received) and the prover's `control` scanned for `blobs`' pieces
    (`pieces`, `find`), the found ones split by the rule (the module's docstring) into ``excluded`` (name to its hex and
    the kinds of `public_runs`, of `circuit` and with `public` as ``extra``, that hold its value) and ``leaked``; the
    control is to hold every held piece (``control`` is None without one: a rescan of a run whose instance file wasn't
    kept)."""
    ps, held_ = pieces(blobs, held)
    table = needles(ps)
    files = {}
    for f in sorted(x for x in vdir.iterdir() if x.is_file()):
        files[f.name] = {"bytes": f.stat().st_size, "found": sorted(find(f.read_bytes(), table))}
    found = sorted({k for x in files.values() for k in x["found"]})
    runs = public_runs(vdir, circuit, public)
    sub = needles({k: ps[k] for k in found})
    pub: dict[str, set[str]] = {}
    for kind, rs in runs.items():
        for r in rs:
            for k in find(r, sub):
                pub.setdefault(k, set()).add(kind)
    ctl = find(control.read_bytes(), table) if control else set()
    return {"pieces": len(ps), "needles": len(table), "files": files,
            "verifier_bytes": sum(x["bytes"] for x in files.values()), "verifier_found": found,
            "excluded": {k: {"hex": ps[k].hex(), "public": sorted(pub[k])} for k in found if k in pub},
            "leaked": [k for k in found if k not in pub],
            "public": {kind: {"runs": len(rs), "bytes": sum(map(len, rs))} for kind, rs in runs.items()},
            "control": control and {"file": control.name, "bytes": control.stat().st_size, "pieces_found": len(ctl),
                                    "of": len(ps), "holds_all": held_ <= ctl, "missing": sorted(held_ - ctl)}}
