"""The received-pieces leak scan every zero-knowledge driver runs: which witness pieces found in what the verifier received
(`a8.received`'s ``verifier_found``, each piece raw or as hex) leak, read against C-Flock's proof format.

    sys.path.insert(0, "benchmarks/network_traces"); import leak_scan as LS
    pieces, held, public = LS.public_pieces(a8, blobs, held, circuit)  # each witness row cut whole, less the public
    LS.scan(received, pieces, vdir)    # {"leaked": [k], "structured": {k: "across" | "public"}, "placed": {k: [...]}}
    LS.find_whole(received, short, vdir, control)  # each row shorter than a piece found whole, then `scan`'s rule

It reads the format with C-Flock's own reader (``verity/ml/flock/verifier``: `redigest.layout`, `transcript_check`).

The pieces (`cut`, the one cutter every driver's `public_pieces` runs): each witness row cut into 16-byte pieces that
cover it whole, its aligned ones and, past its last 16-byte boundary, its last 16 bytes and its last aligned chunk
zero-padded (the row's last element, packed from an element boundary). The residual: a row packed
from an element boundary whose last element pads its tail with anything but zeros matches neither, its last 16 bytes
straddling two elements, so a leak in that tail is missed. A row shorter than a piece (a unit's carry rows, the served
rows of a block of a few steps) is found whole (`find_whole`), raw or as the hex of whole bytes, and its finds are
judged by the same rule. It is at least 6 bytes (`WHOLE`): a uniform L-byte row matches a given offset raw, or an even
offset of hex text, with chance 2^-8L, so R rows over N received bytes expect about 2·N·R·2^-8L chance finds, 2^-7 at 6
bytes for a GiB and 1024 rows but 2 at 5. A low-entropy row matches the format's structure more often: in a field the
reader fixes it reads public, which also excuses a real find of it, and anywhere else leaked.

The rule:
* A find counts only where it is one field's bytes: inside one field of a repetition's proof stream (``S.repK.bin``,
  every read C-Flock's proof format makes of it, `redigest.layout`), inside one 16-byte record of a round message of
  the session transcript (``streams[K].rounds[r].msg``, as `transcript_check.T` frames it), or anywhere in a file whose
  format isn't read (the rest of a transcript, a stream the format doesn't read whole, any other file). A find that
  straddles two fields, or a hex match a nibble off a message's bytes, doesn't count.
* A counted find is excused only if the piece's bytes are in what the statement fixes: the public rows (``pub-*.bin``),
  the statement and the circuit definition (``STATEMENT``), or a field the reader fixes: a stream's integers (its lengths
  and parameters, reads of at most 8 bytes, not its nonces, Ligerito's grinding nonces ``GRIND`` among them, but a ring
  switch's nonce that is 0, as the verifier requires) and a round message's headers, domain and labels. Otherwise the
  piece is ``leaked`` and fails its statement, whatever its row, its bytes or how few values they take: there is no
  byte-count or witness-class exemption, so a switch row found in one field of one repetition fails.
* The excuse is by value, so a low-entropy row equal to some reader-fixed field's bytes is excused wherever it lies. The
  rule that excuses a find only inside a field the reader fixes waits for a class of fields
  whose value is the same for every witness, decided by a differential proof with a different witness: without it an
  honest all-zero private row fails wherever an honest prover writes a zero element (861e's carry row ``cin 6``, inside
  the zero high limbs of Ligerito's intro messages, ``lig.msg``, which the reader doesn't check).

A find across fields isn't a missed leak: the reads tile a stream with no gaps and C-Flock's writer
writes each field whole, so an honest writer's leak lands inside one field. A match across two fields is a coincidental
alignment of their bytes, and a secret split across adjacent fields the prover picks freely (its salts) needs a modified
prover, outside this scan's model. A differential against a proof of a different witness is evidence for a manual
analysis of a flagged piece; it is not part of this rule.
"""
from __future__ import annotations

import bisect
import functools
import json
import re
import struct
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "verity/ml/flock/verifier"))
import redigest as RD  # noqa: E402

#: The verifier's files the statement fixes before any witness: its program and partition, and the staged circuit.
STATEMENT = ("program.json", "partition.json", "circuit.txt")
_SWAP = threading.Lock()

#: Ligerito's grinding nonces, which `redigest.layout` keys without a "nonce" suffix (consumed by
#: `transcript_check`'s ``t.pow``, and prover-controlled: `redigest`'s ``lig-query-nonce``/``lig-claim-nonce``
#: mutants flip them and the verifier accepts), so `reads` counts them among the nonces, not the reader-fixed integers.
#: ``lig.fold`` (``fold_grinding_nonces``) is one too: the verifier requires it empty, but the scan reads what it got.
GRIND = ("lig.qn", "lig.fold", "lig.cbn", "lig.consn")

#: A ring switch's grinding nonce, which the verifier requires to be 0 (C-Flock's PROTOCOL.md §8.3): a zero one is a
#: field the reader fixes, and a nonzero one, which the verifier refuses, stays the prover's.
RS_NONCE = re.compile(r"rs\.\d+\.nonce")

#: The rows `find_whole` takes, by length: shorter than a piece, which `cut` covers, and long enough that chance matches
#: stay rare (2^-7 expected at 6 bytes over a GiB and 1024 rows, 2 at 5; the module's docstring).
WHOLE = range(6, 16)
HEX = b"0123456789abcdefABCDEF"


def cut(blobs: dict[str, bytes], size: int) -> dict[str, bytes]:
    """`blobs` cut into `size`-byte pieces that cover each whole, each distinct piece once under its first name
    (``"<blob> @<offset>"``): its aligned pieces and, when its length isn't a multiple of `size`, its last `size` bytes,
    which overlap the piece before them, and its last aligned chunk zero-padded, the last element of the row packed into
    the format's 16-byte elements from an element boundary, where the overlapping piece straddles two elements."""
    pieces: dict[str, bytes] = {}
    seen: set[bytes] = set()
    for name, b in blobs.items():
        if len(b) < size:
            raise ValueError(f"{name!r} is {len(b)} bytes, shorter than the scan's {size}-byte piece, which `a8.needles` "
                             "won't cut: find it whole (`find_whole`)")
        for o in [*range(0, len(b) - size + 1, size)] + [len(b) - size, len(b) - len(b) % size] * (len(b) % size > 0):
            x = b[o:o + size].ljust(size, b"\0")
            if x not in seen:
                seen.add(x)
                pieces[f"{name} @{o}"] = x
    return pieces


def public_pieces(a8, blobs: dict[str, bytes], held: set[str], public: bytes,
                  fields: bool = False) -> tuple[dict[str, bytes], set[str], list[str]]:
    """`blobs` cut into the scan's pieces (`cut`, at `a8.needles`' size), less the pieces `public` holds raw or as hex:
    what the statement fixes before any witness (the staged circuit, which the stager writes from the Definition before
    any instance), which the verifier has whatever the witness. Returns the pieces to scan for, the held ones (pieces of
    a blob in `held`, which the prover's instance holds as it is) and the excluded pieces' hex. A held blob is a whole
    row, which the instance packs into 16-byte elements, its last one zero-padded; with `fields` it is a field inside a
    row (a head in N2's ``wit`` row), followed there by the next field's bytes (the head's salt), so its zero-padded
    last chunk isn't held, while its last 16 bytes, which cover the same tail, are."""
    pieces = cut(blobs, a8.PIECE)
    out = {pieces[k] for k in a8.find(public, a8.needles(pieces))}
    kept = {k: x for k, x in pieces.items() if x not in out}
    padded = {f"{k} @{len(b) - len(b) % a8.PIECE}" for k, b in blobs.items() if fields and len(b) % a8.PIECE}
    return kept, {k for k in kept if k.rsplit(" @", 1)[0] in held and k not in padded}, sorted(x.hex() for x in out)


def is_stream(name: str) -> bool:
    """Whether `name` is a repetition's proof stream, ``S.repK.bin``."""
    return re.fullmatch(r".*\.rep\d+\.bin", name) is not None


@functools.lru_cache(maxsize=8)
def reads(stream: bytes) -> tuple[tuple[tuple[int, int], ...], frozenset[int]] | None:
    """A ``--zk`` proof stream's fields, (offset, bytes) in order: every read C-Flock's proof format makes of it
    (`redigest.layout`: its elements, hashes, salts and paths and the integers between them), and its nonces' offsets
    (the ``*nonce`` fields and the Ligerito grinding nonces `GRIND`, but a ring switch's that is 0, `RS_NONCE`); None
    when the format doesn't read the whole stream."""
    readers = []
    # `transcript_check.R` is swapped process-wide while `redigest.layout` reads: one caller at a time, or a recorder
    # subclasses another's and stays installed (n2_zk scans its members in threads).
    with _SWAP:
        class Reads(RD.tc.R):
            def __init__(s, b, hlen=32):
                super().__init__(b, hlen)
                s.reads = []
                readers.append(s)

            def take(s, n):
                s.reads.append((s.i, n))
                return super().take(n)

        R, RD.tc.R = RD.tc.R, Reads
        try:
            off = RD.layout(stream, zk=True)
        except (AssertionError, ValueError, KeyError, IndexError, struct.error):
            return None
        finally:
            RD.tc.R = R
    got = next((r.reads for r in reversed(readers) if r.i == len(stream)), None)
    nonces = frozenset(o for k, os in off.items() if k.endswith("nonce") or k in GRIND for o in os
                       if not (RS_NONCE.fullmatch(k) and not any(stream[o:o + 8])))
    return None if got is None else (tuple(got), nonces)


def records(msg: bytes) -> list[tuple[int, int, bool]] | None:
    """A round message's fields as `transcript_check.T` frames them, (offset, bytes, public): each 16-byte header (op,
    kind, n), public, then its payload: one element or n (op 3), the domain's or a label's n bytes padded to 16 (ops 1
    and 2, public) or a nonce's (op 5), none for a squeeze (op 4). None when the message isn't so framed."""
    out, at = [], 0
    while at < len(msg):
        h = msg[at:at + 16]
        if len(h) < 16 or not 1 <= h[0] <= 5 or any(h[2:8]):
            return None
        op, n = h[0], int.from_bytes(h[8:], "little")
        size = 16 * (1 if h[1] == 1 else n) if op == 3 else n + -n % 16 if op in (1, 2, 5) else 0
        if at + 16 + size > len(msg):
            return None
        out.append((at, 16, True))
        if op == 3:
            out += [(at + 16 + 16 * i, 16, False) for i in range(size // 16)]
        elif size:
            out.append((at + 16, size, op != 5))
        at += 16 + size
    return out


def transcript(data: bytes) -> tuple[dict[tuple[int, int], bytes], bytes] | None:
    """A live session's transcript as its streams' round messages by (repetition, round), and the rest of it re-serialised
    without them; None if it isn't one."""
    try:
        t = json.loads(data)
        rounds = [(int(re.search(r"rep(\d+)$", s["label"])[1]), r, m)
                  for s in t["streams"] for r, m in enumerate(s["rounds"])]
        msgs = {(K, r): bytes.fromhex(m["msg"]) for K, r, m in rounds}
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    if len(msgs) != len(rounds):
        return None
    for _K, _r, m in rounds:
        m["msg"] = ""
    return msgs, json.dumps(t).encode()


def _offsets(data: bytes, piece: bytes) -> list[int]:
    out, at = [], data.find(piece) if piece else -1
    while at >= 0:
        out.append(at)
        at = data.find(piece, at + 1)
    return out


def _forms(piece: bytes) -> tuple[bytes, bytes, bytes]:
    return piece, piece.hex().encode(), piece.hex().upper().encode()


def _inside(fields, at: int, n: int) -> bool:
    """Whether bytes at..at+n lie in one of `fields`, (offset, bytes, ...) in order and not overlapping."""
    i = bisect.bisect_right([f[0] for f in fields], at) - 1
    return i >= 0 and at + n <= fields[i][0] + fields[i][1]


def _spans(data: bytes, fields) -> list[bytes]:
    """`data`'s runs of consecutive public fields, (offset, bytes, public) in order."""
    out, run = [], None
    for o, n, public in fields:
        if public and run and run[1] == o:
            run[1] = o + n
            continue
        if run:
            out.append(data[run[0]:run[1]])
        run = [o, o + n] if public else None
    return out + ([data[run[0]:run[1]]] if run else [])


def _read(vdir: Path | None, f: str) -> bytes:
    return (vdir / f).read_bytes() if vdir is not None and (vdir / f).is_file() else b""


def _known(received: dict, pieces: dict[str, bytes]) -> dict:
    return {**received, "verifier_found": [k for k in received["verifier_found"] if k in pieces]}


def placements(received: dict, pieces: dict[str, bytes], vdir: Path | None,
               reads=reads) -> dict[str, list[tuple[str, int | None, str]]]:
    """Where each found piece is in what the verifier received (read from ``vdir``): per file it is found in (a
    transcript's round message as ``FILE#repK.rR``), each raw offset, and how it lies there: ``field``, inside one field of
    a proof stream's format (`reads`) or of a message's (`records`); ``across``, over two or more of them; ``file``, in a
    file whose format isn't read, the rest of a transcript, or (offset None) only as hex. A hex match a nibble off a
    message's bytes is no place."""
    out = {}
    for k in received["verifier_found"]:
        piece, rows = pieces[k], []
        for f, rec in received["files"].items():
            if k not in rec.get("found", {}):
                continue
            b = _read(vdir, f)
            tr = transcript(b) if f.endswith("session.json") else None
            if tr is not None:
                msgs, rest = tr
                for (K, r), m in sorted(msgs.items()):
                    recs = records(m)
                    rows += [(f"{f}#rep{K}.r{r}", at, "file" if recs is None else "field" if _inside(recs, at, len(piece)) else "across")
                             for at in _offsets(m, piece)]
                rows += [(f, None, "file")] * any(x in rest for x in _forms(piece))
                continue
            fields = reads(b) if is_stream(f) else None
            rows += [(f, at, "file" if fields is None else "field" if _inside(fields[0], at, len(piece)) else "across")
                     for at in _offsets(b, piece)] or [(f, None, "file")]
        out[k] = rows
    return out


def public_bytes(files, vdir: Path | None, reads=reads) -> list[bytes]:
    """What the statement fixes in what the verifier received: the public rows (``pub-*.bin``), the statement and circuit
    definition (``STATEMENT``), each proof stream's runs of integers (its lengths and parameters, `reads` of at most 8
    bytes, not its nonces) and each round message's runs of headers, domain and labels (`records`)."""
    out = []
    for f in files:
        b = _read(vdir, f)
        if re.fullmatch(r"pub-\d+\.bin", f) or f in STATEMENT:
            out.append(b)
        elif is_stream(f) and (p := reads(b)) is not None:
            out += _spans(b, [(o, n, n <= 8 and o not in p[1]) for o, n in p[0]])
        elif f.endswith("session.json") and (tr := transcript(b)) is not None:
            out += [s for m in tr[0].values() if (recs := records(m)) is not None for s in _spans(m, recs)]
    return out


def classify(received: dict, pieces: dict[str, bytes], vdir: Path | None = None,
             reads=reads) -> tuple[list[str], dict[str, str]]:
    """`received`'s found pieces as the ``leaked`` ones, which fail a statement, and the ``structured`` rest with why:
    ``across``, no find of it counts (`placements`: each straddles fields of the format, or is a nibble off a message's
    hex), or ``public``, its bytes, raw or as hex, are in what the statement fixes (`public_bytes`). A found piece
    `pieces` doesn't name, or a find where ``vdir`` doesn't show the format, is leaked."""
    placed = placements(_known(received, pieces), pieces, vdir, reads)
    public = None
    leaked, structured = [], {}
    for k in received["verifier_found"]:
        if k not in placed:
            leaked.append(k)
            continue
        if all(why == "across" for _f, _at, why in placed[k]):
            structured[k] = "across"
            continue
        if public is None:
            public = public_bytes(received["files"], vdir, reads)
        if any(x in p for p in public for x in _forms(pieces[k])):
            structured[k] = "public"
        else:
            leaked.append(k)
    return leaked, structured


def scan(received: dict, pieces: dict[str, bytes], vdir: Path | None, reads=reads) -> dict:
    """`received` scanned: its ``leaked`` and ``structured`` pieces (`classify`) and where each known one was found
    (``placed``, `placements`)."""
    leaked, structured = classify(received, pieces, vdir, reads)
    return {"leaked": leaked, "structured": structured, "placed": placements(_known(received, pieces), pieces, vdir, reads)}


def rescan(received: dict, pieces: dict[str, bytes], vdir: Path | None, reads=reads, a8=None, rows=()) -> dict:
    """A finished run's scan (its summary's ``received``) taken again under this rule from the verifier's files it kept,
    with the pieces the driver recomputes: `scan`'s result, and the run's own ``leaked`` as ``was``. A found piece the
    driver can't recompute (a salt the run drew) reads leaked. With `a8`, each kept file is searched afresh for `pieces`
    (`a8.find`), so a piece the run's cutter didn't cut is searched too, and of the run's own finds only those of a blob
    the driver didn't recompute (not in `rows`) stay."""
    if a8 is not None and vdir is not None:
        table, files = a8.needles(pieces), {}
        for f, rec in received["files"].items():
            if not (vdir / f).is_file():
                files[f] = rec
                continue
            old = {k: n for k, n in rec.get("found", {}).items() if k.rsplit(" @", 1)[0] not in rows}
            files[f] = {**rec, "found": {**old, **a8.find((vdir / f).read_bytes(), table)}}
        received = {**received, "files": files, "verifier_found": sorted({k for x in files.values() for k in x["found"]})}
    return {"was": received.get("leaked"), **scan(received, pieces, vdir, reads)}


def _run(data: bytes, at: int) -> tuple[int, int]:
    """The run of hex digits in `data` around offset `at`, [start, end)."""
    s = e = at
    while s:
        lo = max(0, s - 4096)
        left = len(data[lo:s].rstrip(HEX))
        s = lo + left
        if left:
            break
    while e < len(data):
        hi = min(len(data), e + 4096)
        right = len(data[e:hi].lstrip(HEX))
        e = hi - right
        if right:
            break
    return s, e


def _hex_in(data: bytes, hx: bytes) -> bool:
    """Whether hex `hx` is in `data` as the hex of whole bytes: at an even offset into its run of hex digits, not a
    nibble off."""
    run, at = (0, 0), data.find(hx)
    while at >= 0:
        if not run[0] <= at < run[1]:
            run = _run(data, at)
        if (at - run[0]) % 2 == 0:
            return True
        at = data.find(hx, at + 1)
    return False


def _whole(data: bytes, row: bytes) -> int:
    """How many of `row`'s forms are in `data`: raw, or as the hex of whole bytes in either case."""
    raw, *hx = _forms(row)
    return (raw in data) + sum(_hex_in(data, h) for h in set(hx))


def find_whole(received: dict | None, rows: dict[str, bytes], vdir: Path | None, control: Path | None = None,
               held: set[str] | None = None, reads=reads) -> dict:
    """`rows`, each shorter than a piece (`WHOLE`), found whole in the files the verifier received (``received``'s, a
    piece scan's record, or every file in `vdir`): raw, or as the hex of whole bytes (`_hex_in`), each distinct row once
    under its first name. The finds are judged by `scan`'s rule (``leaked``, ``structured``, ``placed``), and with the
    prover's `control` file, whether it holds every row in `held` (all of them by default)."""
    if bad := {k: len(x) for k, x in rows.items() if len(x) not in WHOLE}:
        raise ValueError(f"rows {bad} (their bytes) aren't {WHOLE.start} to {WHOLE.stop - 1} bytes: cut a row of 16 or more "
                         f"(`public_pieces`), so a leak of any piece of it is found; a row under {WHOLE.start} bytes "
                         "can't be told from chance matches, so scan the longer row holding it (a block of more steps)")
    first: dict[bytes, str] = {}
    for k, x in rows.items():
        first.setdefault(x, k)
    names = sorted(received["files"]) if received is not None else sorted(f.name for f in vdir.iterdir() if f.is_file())
    files = {}
    for f in names:
        b = _read(vdir, f)
        files[f] = {"bytes": len(b), "found": {k: n for x, k in first.items() if (n := _whole(b, x))}}
    out: dict = {"rows": len(first), "files": files,
                 "verifier_found": sorted({k for x in files.values() for k in x["found"]})}
    if control is not None:
        b = control.read_bytes()
        missing = sorted({first[rows[k]] for k in (rows if held is None else held) if not _whole(b, rows[k])})
        out["control"] = {"file": control.name, "holds_all": not missing, "missing": missing}
    return {**out, **scan(out, {k: x for x, k in first.items()}, vdir, reads)}
