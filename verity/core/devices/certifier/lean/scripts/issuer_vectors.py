"""The issuer difftest's data: records to certify, and what `certificate.Issuer` does with each, for `CertifierDevice/IssuerDifftest.lean`.

    python scripts/issuer_vectors.py <verity checkout> --out DIR [--count N]

Writes `DIR/issuer/instances.jsonl`, one instance per line as `CertifierDevice.IssuerDifftest.evaluate` reads it (an LMS
key, made by `lms.keygen` from a seed, a certifier session's id, and a sequence of calls to one issuer in that session),
and `DIR/issuer/python.jsonl`, the same line's results from the checkout's real `certificate.Issuer`. A call is a record (link, window, grid and status), its
direction, grid parameters and Σ_sync (or none), the parameters digest it is certified under (`parameters_digest` of
those, or other bytes), the deadline bit, the leaf the host reserves for it (or a refusal), a tamper to check
`CertifiedRecord.verify` against, and which way, if any, the call was made malformed (`MALFORMED`). The session id is 32
bytes, or in some instances 31 or 33, which `Issuer.certify` refuses on every call. The issuer's signer is a stand-in
that starts any session and signs at the call's leaf with `lms.sign_at`, or raises `SignerRefused` where the reservation
is refused, since the signer's state (`lms.LeafState.start` and `reserve`) is the host's.
Each call's result is the parameters digest, and either `null` where `Issuer.certify` raises `ValueError` or the
certified record (`null` when refused): its head's encoding and digest, its certificate's fields, `verify` under the
key, and `verify` after the tamper; at the end, the issuer's `refused` list.

The keys are the deployed scheme's N = 64, W = 1 at height 1 and `TOY_N` at W = 1, 2 and 4, small enough to sign
interpreted (`Lms.SignDifftest` holds signing itself to `lms.sign_at` at more sets and heights). The records are grids of B-byte frames, left-packed, under random parameters and statuses, and grids
`record_head` refuses (a row too many or too few, a row too long, a row not left-packed, a frame of the wrong size), a
window past a u64, a link past 65,535 bytes, parameters bytes not 64 long, and leaves past 2^H. The tampers change the
certificate's signature (a byte, or its length), link, direction, window or digest, the head's deadline, or the public
key's identifier or root, or check the certificate in another session; or they change the certificate's link, direction,
window or key identifier and sign it again at the same leaf, so that only the checks that it names the head's link-window and the key reject it. The Lean audit runs `CertifierDevice/IssuerDifftestMain.lean` on both files (`runs` in
`lean-audit.json`, which pins their SHA-256), and `verity/core/protocols/certifier/tests/
test_network_certifier_issuer_lean_difftest.py` checks that this script still writes them.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
from fractions import Fraction
from pathlib import Path

import verity.core.protocols.certifier
from verity.core.protocols.certifier.certificate import (Direction, Issuer, LmsKey, SignerRefused,
                                                                 parameters_digest)
from verity.core.protocols.certifier.grid import GridParams, Reason, Record, Status
from verity.core.protocols.certifier.schedule import SyncSet
from verity.ml.crypto import lms

COUNT = 160
#: (N, W, H, toy) by weight: the audit runs the difftest interpreted, so N = 64 is W = 1 at height 1, and W = 8, whose
#: chains are 255 hashes long, is `Lms.SignDifftest`'s alone
KEYS = [((64, 1, 1, False), 3), ((8, 1, 2, True), 2), ((8, 2, 3, True), 2), ((8, 4, 2, True), 2)]
RHOS = [Fraction(1), Fraction(1, 2), Fraction(3, 2), Fraction(999, 1000), Fraction(1001, 1000), Fraction(7)]
LINKS = ["responses", "requests", "", "réponses/λ"]
TAMPERS = ["signature", "signature-length", "link", "direction", "window", "digest", "deadline", "public-id",
           "public-root", "session", "resign-link", "resign-direction", "resign-window", "resign-key-id"]
MALFORMED = ["rows", "row", "packing", "frame", "window", "link", "parameters", "leaf"]


class LeafSigner:
    """The issuer's signer at the leaf the host reserved for the call (`leaf`), or refused when the host refused it."""

    def __init__(self, key: lms.PrivateKey) -> None:
        self.key, self.leaf = key, None

    @property
    def key_id(self) -> bytes:
        return self.key.I

    def start(self, session: bytes) -> None:
        if self.leaf is None:
            raise SignerRefused("the signer's state refused the session's start")

    def sign(self, message: bytes) -> bytes:
        if self.leaf is None:
            raise SignerRefused("the leaf state refused the reservation")
        return lms.sign_at(self.key, self.leaf, message).encode()


def grid_of(rng: random.Random, p: GridParams) -> list[list[str | None]]:
    rows = []
    for _ in range(p.buckets):
        k = rng.choice([0, p.allocation, rng.randint(0, p.allocation)])
        rows.append([rng.randbytes(p.frame_bytes).hex() for _ in range(k)] + [None] * (p.allocation - k))
    return rows


def call(rng: random.Random, height: int) -> dict:
    p = GridParams(rng.randint(1, 5), rng.randint(1, 4), rng.choice([9, 16, 32, 64]), rng.choice([1, 20, 100]),
                   rng.choice([1, 16, 1 << 16]))
    direction = rng.choice(list(Direction))
    sync_set = None
    if direction is Direction.EGRESS and rng.random() < 0.8:
        sync_set = {"deltas": rng.sample(range(0, 6), rng.randint(1, 3)),
                    "rhos": [[r.numerator, r.denominator] for r in rng.sample(RHOS, rng.randint(1, 3))]}
    grid = grid_of(rng, p)
    record = {"link": rng.choice(LINKS), "window": rng.choice([0, 1, rng.randrange(1000), 2 ** 64 - 1]), "grid": grid,
              "status": None if rng.random() < 0.5 else rng.choice(list(Reason)).value}
    c = {"record": record, "direction": direction.value, "params": {"T": p.buckets, "r": p.allocation,
         "B": p.frame_bytes, "tau": p.bucket_ms, "Q": p.queue_capacity}, "sync_set": sync_set, "parameters": None,
         "deadline": rng.random() < 0.8, "leaf": None if rng.random() < 0.15 else rng.randrange(2 ** height),
         "tamper": None if rng.random() < 0.3 else [rng.choice(TAMPERS), rng.randrange(1 << 16)], "malformed": None}
    if rng.random() < 0.15:
        bad = c["malformed"] = rng.choice(MALFORMED)
        if bad == "rows":
            grid.append([None] * p.allocation) if rng.random() < 0.5 else grid.pop()
        elif bad == "row":
            grid[rng.randrange(len(grid))].append(None)
        elif bad == "packing":
            row = grid[rng.randrange(len(grid))]
            row[-1] = rng.randbytes(p.frame_bytes).hex()
            row[0] = None
        elif bad == "frame":
            grid[rng.randrange(len(grid))][0] = rng.randbytes(p.frame_bytes + rng.choice([-1, 1])).hex()
        elif bad == "window":
            record["window"] = 2 ** 64 + rng.randrange(3)
        elif bad == "link":
            record["link"] = "x" * 65536
        elif bad == "parameters":
            c["parameters"] = rng.randbytes(rng.choice([0, 63, 65])).hex()
        else:
            c["leaf"] = 2 ** height + rng.randrange(3)
    return c


def instance(seed: int) -> dict:
    rng = random.Random(seed)
    (n, w, h, toy), = rng.choices([k for k, _ in KEYS], [x for _, x in KEYS])
    session = rng.randbytes(32 if rng.random() < 0.9 else rng.choice([31, 33]))
    return {"key": {"params": [n, w, h, toy], "seed": rng.randbytes(n).hex()}, "session": session.hex(),
            "calls": [call(rng, h) for _ in range(rng.randint(1, 5))]}


def record_of(x: dict) -> Record:
    grid = tuple(tuple(None if s is None else bytes.fromhex(s) for s in row) for row in x["grid"])
    return Record(x["link"], x["window"], grid, Status(None if x["status"] is None else Reason(x["status"])))


def tampered(c, key: LmsKey, private: lms.PrivateKey, q: int, session: bytes, tamper: list) -> bool:
    kind, at = tamper
    x, h = c.certificate, c.head
    flip = lambda b: b[:at % len(b)] + bytes([b[at % len(b)] ^ 1]) + b[at % len(b) + 1:]
    if kind == "signature":
        x = dataclasses.replace(x, signature=flip(x.signature))
    elif kind == "signature-length":
        x = dataclasses.replace(x, signature=x.signature[:-1])
    elif kind == "link":
        x = dataclasses.replace(x, link=x.link + "x")
    elif kind == "direction":
        x = dataclasses.replace(x, direction=Direction.INGRESS if x.direction is Direction.EGRESS else Direction.EGRESS)
    elif kind == "window":
        x = dataclasses.replace(x, window=x.window ^ 1)
    elif kind == "digest":
        x = dataclasses.replace(x, digest=flip(x.digest))
    elif kind == "deadline":
        h = dataclasses.replace(h, deadline=not h.deadline)
    elif kind.startswith("resign-"):
        x = dataclasses.replace(x, **{
            "link": {"link": x.link + "x"},
            "direction": {"direction": Direction.INGRESS if x.direction is Direction.EGRESS else Direction.EGRESS},
            "window": {"window": x.window ^ 1}, "key-id": {"key_id": flip(x.key_id)}}[kind.removeprefix("resign-")])
        x = dataclasses.replace(x, signature=lms.sign_at(private, q, x.message(session)).encode())
    elif kind == "session":
        session = flip(session)
    elif kind == "public-id":
        key = LmsKey(key.params, flip(key.public[:16]) + key.public[16:])
    else:
        key = LmsKey(key.params, key.public[:16] + flip(key.public[16:]))
    return dataclasses.replace(c, head=h, certificate=x).verify(key, session)


def python_results(inst: dict) -> dict:
    n, w, h, toy = inst["key"]["params"]
    params = lms.Params(n, w, h, toy=toy)
    private = lms.keygen(params, bytes.fromhex(inst["key"]["seed"]))
    key = LmsKey(params, lms.public_key(private))
    signer = LeafSigner(private)
    session = bytes.fromhex(inst["session"])
    issuer = Issuer(signer, session)
    out = []
    for c in inst["calls"]:
        p = GridParams(**dict(zip(["buckets", "allocation", "frame_bytes", "bucket_ms", "queue_capacity"],
                                  [c["params"][k] for k in ["T", "r", "B", "tau", "Q"]])))
        ss = c["sync_set"]
        ss = None if ss is None else SyncSet(tuple(ss["deltas"]), tuple(Fraction(a, b) for a, b in ss["rhos"]))
        parameters = parameters_digest(p, ss) if c["parameters"] is None else bytes.fromhex(c["parameters"])
        signer.leaf = c["leaf"]
        try:
            certified = issuer.certify(record_of(c["record"]), Direction(c["direction"]), p, parameters,
                                       deadline=c["deadline"])
        except ValueError:
            out.append({"parameters": parameters.hex(), "certify": None})
            continue
        if certified is None:
            out.append({"parameters": parameters.hex(), "certify": {"certified": None}})
            continue
        x = certified.certificate
        out.append({"parameters": parameters.hex(), "certify": {"certified": {
            "head": [certified.head.encode().hex(), certified.head.digest().hex()],
            "certificate": [x.key_id.hex(), x.link, x.direction.value, x.window, x.digest.hex(), x.signature.hex()],
            "verify": certified.verify(key, session),
            "tampered": None if c["tamper"] is None else tampered(certified, key, private, c["leaf"], session,
                                                                  c["tamper"])}}})
    return {"public": key.public.hex(), "calls": out,
            "refused": [[link, d.value, window] for link, d, window in issuer.refused]}


def line(obj: dict) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True) + "\n"


def write(out: Path, count: int = COUNT) -> dict[str, Path]:
    """Both files for instances 0 … count − 1 under ``out/issuer``."""
    d = out / "issuer"
    d.mkdir(parents=True, exist_ok=True)
    paths = {"issuer/instances.jsonl": d / "instances.jsonl", "issuer/python.jsonl": d / "python.jsonl"}
    with open(paths["issuer/instances.jsonl"], "w") as fi, open(paths["issuer/python.jsonl"], "w") as fp:
        for seed in range(count):
            inst = instance(seed)
            fi.write(line(inst))
            fp.write(line(python_results(inst)))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("verity", type=Path, help="the checkout whose verity.core.protocols.certifier is tested")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=COUNT)
    a = ap.parse_args()
    here = Path(verity.core.protocols.certifier.__file__).resolve()
    if not here.is_relative_to(a.verity.resolve()):
        raise SystemExit(f"verity.core.protocols.certifier comes from {here}, not from the checkout {a.verity}")
    write(a.out, a.count)


if __name__ == "__main__":
    main()
