"""Onsite's or remote's case from its subprotocols' cases, and the Lean verdict on it (`onsite check`, `remote check`).

    composite.py onsite|remote --piece DIR ... [--registration DIR] --out DIR [--exe PATH]

Each ``--piece`` is one subprotocol's directory, holding its case as its own `serve` reads it, ``case.json``:
``{"params", "commitment", "evidence"}`` with the commitment and evidence hex, and ``blobs``, the paths of the files it
reads (sampled proofs v2's ``case-v2.json`` is one); and its own Lean answer on that case, ``answer.json``, whose
``protocol`` is its id and ``commitment`` its digest (null when its commitment didn't decode, and then the case's own
commitment stands in, which a rejection never binds). ``--registration`` is NCI's (`verity/nci/v1`) for onsite, which
registers before composing; its digest is the one sampled proofs' case commits to.

It writes into ``--out`` the product's params (``params.json``), commitment (``commitment.bin``: `bind` of the digests
in the order of the ids, then the registration's digest) and evidence (``evidence.json``), all canonical, and the blobs'
paths (``blobs.txt``), as `Core.Composition.registeredB` and `productB` read them (the composition README's formats),
from the cases that exist. It prints each piece's own answer line (``PIECE``). With ``--exe``, it runs ``EXE check
params.json commitment.bin evidence.json BLOB...``, writes its outcome line to ``answer.json``, prints it
(``VERDICT``) and exits with its code: 0 accept, 1 reject, 2 undecided. It decides nothing itself: a subprotocol with no
piece, or whose directory lacks its case or its answer, is left out of the case, and Lean's verdict on that case is
the composite's (a rejection naming the missing case). Such a run is labelled ``NOT_ESTABLISHED``, naming what is
missing. 64 is a usage error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PRODUCTS = {"onsite": "verity/onsite/v0", "remote": "verity/remote/v0"}
#: Each product's subprotocols, as `Core.Onsite.components` and `Core.Remote.components` list them; read only to name
#: the missing ones.
COMPONENTS = {
    "onsite": ("verity/certifier/v0", "verity/pous/p2/v4/run", "verity/pouw/period/v1", "verity/remote-bandwidth/v0",
               "verity/sampled-proofs/one-stage/v2", "verity/wipe/v0"),
    "remote": ("verity/pouw/v1", "verity/sampled-proofs/one-stage/v2"),
}
REGISTRATION = "verity/nci/v1"
USAGE = 64


class Parser(argparse.ArgumentParser):
    """argparse's, but a usage error exits 64, not 2, which is Lean's undecided."""

    def error(self, message: str):
        self.print_usage(sys.stderr)
        self.exit(USAGE, f"{self.prog}: error: {message}\n")


def canonical(x: Any) -> bytes:
    """`x` as `Core.canonicalJson` reads it, for the integers, strings, lists and objects of ASCII keys cases hold."""
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def bind(id: str, digests: Sequence[tuple[str, bytes]]) -> bytes:
    """`Core.Composition.bind`: SHA-512 of the product's id, a zero byte and the digests in order as canonical JSON."""
    body = canonical({"components": [{"commitment": d.hex(), "id": i} for i, d in digests]})
    return hashlib.sha512(id.encode() + b"\0" + body).digest()


def ran(d: Path) -> bool:
    return (d / "case.json").is_file() and (d / "answer.json").is_file()


def piece(d: Path) -> dict[str, Any]:
    """A subprotocol's case, digest and own answer line from its directory: ``{"id", "params", "commitment", "evidence",
    "digest", "blobs", "answer"}``, the commitment, evidence and digest bytes."""
    case = json.loads((d / "case.json").read_text())
    line = (d / "answer.json").read_text().strip()
    answer = json.loads(line)
    commitment = bytes.fromhex(case["commitment"])
    digest = answer.get("commitment")
    return {"id": answer["protocol"], "params": case["params"], "commitment": commitment,
            "evidence": bytes.fromhex(case["evidence"]), "digest": bytes.fromhex(digest) if digest else commitment,
            "blobs": [str(b) for b in case.get("blobs") or []], "answer": line}


def compose(product: str, pieces: Sequence[dict[str, Any]], registration: dict[str, Any] | None = None
            ) -> tuple[dict, bytes, bytes, list[str]]:
    """The product's params, commitment, evidence and blobs, the subprotocols in the order of their ids."""
    xs = sorted(pieces, key=lambda p: p["id"])
    params: dict[str, Any] = {"components": [{"id": p["id"], "params": p["params"]} for p in xs]}
    evidence: dict[str, Any] = {"components": [{"commitment": p["commitment"].hex(), "evidence": p["evidence"].hex(),
                                                "id": p["id"], "params": p["params"]} for p in xs]}
    commitment = bind(PRODUCTS[product], [(p["id"], p["digest"]) for p in xs])
    if registration is not None:
        params["registration"] = {"id": registration["id"], "params": registration["params"]}
        evidence["registration"] = {"commitment": registration["commitment"].hex(),
                                    "evidence": registration["evidence"].hex(), "id": registration["id"],
                                    "params": registration["params"]}
        commitment += registration["digest"]
    blobs = list(dict.fromkeys(b for p in xs for b in p["blobs"]))
    return params, commitment, canonical(evidence), blobs


def missing(product: str, pieces: Sequence[dict[str, Any]], registration: dict[str, Any] | None) -> list[str]:
    """The product's subprotocols with no piece that ran, then onsite's registration when it has none."""
    ids = {p["id"] for p in pieces}
    out = [i for i in COMPONENTS[product] if i not in ids]
    return out + ([REGISTRATION] if product == "onsite" and registration is None else [])


def main(argv: Sequence[str] | None = None) -> int:
    ap = Parser(description=__doc__.split("\n\n")[0])
    ap.add_argument("product", choices=sorted(PRODUCTS))
    ap.add_argument("--piece", type=Path, action="append", default=[], help="a subprotocol's case.json and answer.json")
    ap.add_argument("--registration", type=Path, help="NCI's case.json and answer.json (onsite)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exe", help="the product's executable, to run its check")
    a = ap.parse_args(argv)
    reg = piece(a.registration) if a.registration is not None and ran(a.registration) else None
    pieces = [piece(d) for d in a.piece if ran(d)]
    params, commitment, evidence, blobs = compose(a.product, pieces, reg)
    a.out.mkdir(parents=True, exist_ok=True)
    files = [a.out / "params.json", a.out / "commitment.bin", a.out / "evidence.json"]
    for f, b in zip(files, (canonical(params), commitment, evidence)):
        f.write_bytes(b)
    (a.out / "blobs.txt").write_text("".join(f"{b}\n" for b in blobs))
    for p in sorted(pieces, key=lambda p: p["id"]) + ([reg] if reg is not None else []):
        print(f"PIECE {p['answer']}", flush=True)
    print("COMPOSITE " + json.dumps({"product": PRODUCTS[a.product], "components": [x["id"] for x in params["components"]],
                                     "commitment": commitment.hex(), "blobs": len(blobs)}, sort_keys=True), flush=True)
    if gone := missing(a.product, pieces, reg):
        print(f"NOT_ESTABLISHED missing {', '.join(gone)}", flush=True)
    cmd = [a.exe or a.product, "check", *map(str, files), *blobs]
    if not a.exe:
        print("CHECK " + " ".join(cmd), flush=True)
        return 0
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    (a.out / "answer.json").write_text(line + "\n")
    print(f"VERDICT {line or p.stderr.strip()[-2000:]}", flush=True)
    return p.returncode


if __name__ == "__main__":
    sys.exit(main())
