"""Onsite's or remote's case from its subprotocols' cases, and the Lean verdict on it (`onsite check`, `remote check`).

    composite.py onsite|remote --piece DIR ... [--registration DIR] --out DIR [--exe PATH]

Each ``--piece`` is one subprotocol's directory, holding its case as its own `serve` reads it, ``case.json``:
``{"params", "commitment", "evidence"}`` with the commitment and evidence hex, and ``blobs``, the paths of the files it
reads (sampled proofs v2's ``case-v2.json`` is one); and its own Lean answer on that case, ``answer.json``, a
`Core.Answer` (read by `verity.core.protocols.answer`, which refuses any other shape), whose ``protocol`` is its id and
``commitment`` its digest (null when its commitment didn't decode, and then the case's own commitment stands in, which a
rejection never binds). ``--registration`` is NCI's (`verity/nci/v1`) for onsite, which
registers before composing; its digest is the one sampled proofs' case commits to.

It writes into ``--out`` the product's params (``params.json``), commitment (``commitment.bin``: `bind` of the digests
in the order of the ids, then the registration's digest) and evidence (``evidence.json``), all canonical, and the blobs'
paths (``blobs.txt``), as `Core.Composition.registeredB` and `productB` read them (the composition README's formats).
With ``--exe``, it runs ``EXE check params.json commitment.bin evidence.json BLOB...``, writes its outcome line to
``answer.json`` and exits with its code: 0 accept, 1 reject, 2 undecided.
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

from verity.core.protocols import answer as CA

PRODUCTS = {"onsite": "verity/onsite/v0", "remote": "verity/remote/v0"}
REGISTRATION = "verity/nci/v1"


def canonical(x: Any) -> bytes:
    """`x` as `Core.canonicalJson` reads it, for the integers, strings, lists and objects of ASCII keys cases hold."""
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def bind(id: str, digests: Sequence[tuple[str, bytes]]) -> bytes:
    """`Core.Composition.bind`: SHA-512 of the product's id, a zero byte and the digests in order as canonical JSON."""
    body = canonical({"components": [{"commitment": d.hex(), "id": i} for i, d in digests]})
    return hashlib.sha512(id.encode() + b"\0" + body).digest()


def piece(d: Path) -> dict[str, Any]:
    """A subprotocol's case and digest from its directory: ``{"id", "params", "commitment", "evidence", "digest",
    "blobs"}``, the commitment, evidence and digest bytes."""
    case = json.loads((d / "case.json").read_text())
    answer = CA.of_json(json.loads((d / "answer.json").read_text()), str(d / "answer.json"))
    commitment = bytes.fromhex(case["commitment"])
    return {"id": answer.protocol, "params": case["params"], "commitment": commitment,
            "evidence": bytes.fromhex(case["evidence"]),
            "digest": commitment if answer.digest is None else answer.digest,
            "blobs": [str(b) for b in case.get("blobs") or []]}


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


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("product", choices=sorted(PRODUCTS))
    ap.add_argument("--piece", type=Path, action="append", default=[], help="a subprotocol's case.json and answer.json")
    ap.add_argument("--registration", type=Path, help="NCI's case.json and answer.json (onsite)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exe", help="the product's executable, to run its check")
    a = ap.parse_args(argv)
    if (a.product == "onsite") != (a.registration is not None):
        ap.error("onsite takes NCI's registration (--registration), and remote none")
    reg = piece(a.registration) if a.registration else None
    if reg is not None and reg["id"] != REGISTRATION:
        ap.error(f"--registration is {reg['id']}'s case, not {REGISTRATION}'s")
    params, commitment, evidence, blobs = compose(a.product, [piece(d) for d in a.piece], reg)
    a.out.mkdir(parents=True, exist_ok=True)
    files = [a.out / "params.json", a.out / "commitment.bin", a.out / "evidence.json"]
    for f, b in zip(files, (canonical(params), commitment, evidence)):
        f.write_bytes(b)
    (a.out / "blobs.txt").write_text("".join(f"{b}\n" for b in blobs))
    cmd = [a.exe or a.product, "check", *map(str, files), *blobs]
    print("COMPOSITE " + json.dumps({"product": PRODUCTS[a.product], "components": [x["id"] for x in params["components"]],
                                     "commitment": commitment.hex(), "blobs": len(blobs)}, sort_keys=True), flush=True)
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
