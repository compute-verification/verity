"""Onsite's or remote's case from its subprotocols' cases, and the Lean verdict on it (`onsite check`, `remote check`).

    composite.py onsite|remote --piece DIR ... [--registration DIR] --out DIR [--exe PATH]

Each ``--piece`` is one subprotocol's directory, holding its case as its own `serve` reads it, ``case.json``:
``{"params", "commitment", "evidence"}`` with the commitment and evidence hex, and ``blobs``, the paths of the files it
reads (sampled proofs v2's ``case-v2.json`` is one); and its own Lean answer on that case, ``answer.json``, whose
``protocol`` is its id, ``verdict`` its verdict and ``commitment`` its digest (null when its commitment didn't decode,
and then the case's own commitment stands in, which a rejection never binds). ``--registration`` is NCI's
(`verity/nci/v1`) for onsite, which registers before composing; its digest is the one sampled proofs' case commits to.

A subprotocol of the product with no piece, or whose directory lacks its case or its answer, didn't run, and enters as
undecided ("not run"), as does onsite's registration without ``--registration``. Then nothing is composed, and the
verdict is `Core.Composition.judge`'s over the answers that exist with the ones that didn't run undecided: a refused
registration or a run piece's rejection rejects, and otherwise the product is undecided. ``answer.json`` in ``--out``
gets that answer, in `Core.Answer`'s form, and it exits 1 or 2.

When every one ran, it writes into ``--out`` the product's params (``params.json``), commitment (``commitment.bin``:
`bind` of the digests in the order of the ids, then the registration's digest) and evidence (``evidence.json``), all
canonical, and the blobs' paths (``blobs.txt``), as `Core.Composition.registeredB` and `productB` read them (the
composition README's formats). With ``--exe``, it runs ``EXE check params.json commitment.bin evidence.json BLOB...``,
writes its outcome line to ``answer.json`` and exits with its code: 0 accept, 1 reject, 2 undecided. 64 is a usage
error.
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
#: Each product's subprotocols, as `Core.Onsite.components` and `Core.Remote.components` list them.
COMPONENTS = {
    "onsite": ("verity/certifier/v0", "verity/pous/p2/v4/run", "verity/pouw/period/v1", "verity/remote-bandwidth/v0",
               "verity/sampled-proofs/one-stage/v2", "verity/wipe/v0"),
    "remote": ("verity/pouw/v1", "verity/sampled-proofs/one-stage/v2"),
}
REGISTRATION = "verity/nci/v1"
NOT_RUN = "not run"
USAGE = 64


class Usage(Exception):
    pass


class Parser(argparse.ArgumentParser):
    """argparse's, but a usage error exits 64, not 2, which is undecided here."""

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
    """A subprotocol's case, digest and own verdict from its directory: ``{"id", "params", "commitment", "evidence",
    "digest", "blobs", "verdict", "reason"}``, the commitment, evidence and digest bytes."""
    case = json.loads((d / "case.json").read_text())
    answer = json.loads((d / "answer.json").read_text())
    commitment = bytes.fromhex(case["commitment"])
    digest = answer.get("commitment")
    return {"id": answer["protocol"], "params": case["params"], "commitment": commitment,
            "evidence": bytes.fromhex(case["evidence"]), "digest": bytes.fromhex(digest) if digest else commitment,
            "blobs": [str(b) for b in case.get("blobs") or []], "verdict": answer.get("verdict"),
            "reason": answer.get("reason")}


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


def reasons(xs: Sequence[tuple[str, str, str | None]]) -> str:
    """`Core.Composition`'s reasons: each answer's ``ID VERDICT: REASON``, joined by ``; ``."""
    return "; ".join(f"{i} {v}: {r or ''}" for i, v, r in xs)


def undecided_answer(product: str, pieces: Sequence[dict[str, Any]], missing: Sequence[str],
                     registration: dict[str, Any] | None, needs_registration: bool) -> dict[str, Any]:
    """The product's answer when some subprotocol didn't run, in `Core.Answer`'s form: a refused registration rejects,
    then any run piece's rejection; otherwise it is undecided, naming the undecided and the not-run ones."""
    def answer(verdict: str, reason: str) -> dict[str, Any]:
        return {"commitment": None, "profile": None, "protocol": PRODUCTS[product], "public": None, "reason": reason,
                "verdict": verdict}
    if registration is not None and registration["verdict"] in ("reject", "undecided"):
        return answer("reject" if registration["verdict"] == "reject" else "undecided",
                      "registration: " + reasons([(registration["id"], registration["verdict"],
                                                   registration["reason"])]))
    if needs_registration and registration is None:
        return answer("undecided", "registration: " + reasons([(REGISTRATION, "undecided", NOT_RUN)]))
    xs = sorted([(p["id"], p["verdict"], p["reason"]) for p in pieces] + [(i, "undecided", NOT_RUN) for i in missing])
    rejected = [x for x in xs if x[1] == "reject"]
    if rejected:
        return answer("reject", reasons(rejected))
    return answer("undecided", reasons([x for x in xs if x[1] != "accept"]))


def run(a: argparse.Namespace) -> int:
    if a.product == "remote" and a.registration is not None:
        raise Usage("remote takes no registration (--registration)")
    reg = piece(a.registration) if a.registration is not None and ran(a.registration) else None
    if reg is not None and reg["id"] != REGISTRATION:
        raise Usage(f"--registration is {reg['id']}'s case, not {REGISTRATION}'s")
    pieces = [piece(d) for d in a.piece if ran(d)]
    ids = [p["id"] for p in pieces]
    if unknown := sorted(set(ids) - set(COMPONENTS[a.product])):
        raise Usage(f"{', '.join(unknown)} is not a subprotocol of {PRODUCTS[a.product]}")
    if twice := sorted({i for i in ids if ids.count(i) > 1}):
        raise Usage(f"{', '.join(twice)} has two pieces")
    missing = [i for i in COMPONENTS[a.product] if i not in ids]
    a.out.mkdir(parents=True, exist_ok=True)
    if missing or (a.product == "onsite" and reg is None):
        ans = undecided_answer(a.product, pieces, missing, reg, a.product == "onsite")
        line = canonical(ans).decode()
        (a.out / "answer.json").write_text(line + "\n")
        print("NOT_RUN " + json.dumps(missing + ([] if a.product == "remote" or reg else [REGISTRATION])), flush=True)
        print(f"VERDICT {line}", flush=True)
        return 1 if ans["verdict"] == "reject" else 2
    params, commitment, evidence, blobs = compose(a.product, pieces, reg)
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


def main(argv: Sequence[str] | None = None) -> int:
    ap = Parser(description=__doc__.split("\n\n")[0])
    ap.add_argument("product", choices=sorted(PRODUCTS))
    ap.add_argument("--piece", type=Path, action="append", default=[], help="a subprotocol's case.json and answer.json")
    ap.add_argument("--registration", type=Path, help="NCI's case.json and answer.json (onsite)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exe", help="the product's executable, to run its check")
    try:
        return run(ap.parse_args(argv))
    except Usage as e:
        print(f"composite.py: {e}", file=sys.stderr)
        return USAGE


if __name__ == "__main__":
    sys.exit(main())
