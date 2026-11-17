"""Onsite's or remote's case from its subprotocols' cases, and the Lean verdict on it (`onsite check`, `remote check`,
`remote-v1 check`).

    composite.py onsite|remote|remote-v1 --cases DIR ... --private-out DIR --public-out DIR [--exe PATH]
                 [--development WHY] [--key-sha512 HEX] [--coins-on MACHINE]

``remote-v1`` is remote's composite of record (`Core.Remote.V1`, ``verity/remote/v1``), whose work subprotocol is
Pearl+'s run verdict (`verity/pouw/pearl-plus/v1`) in the period's place; ``remote`` is v0, H1's development one.

Each ``--cases`` is a private directory (0700, its files 0600, as the pieces write it) holding one subprotocol's case,
or a directory whose subdirectories each hold one. A case is the subprotocol's case as its own `serve` reads it,
``case.json``: ``{"params", "commitment", "evidence"}`` with the commitment and evidence hex, and ``blobs``, the paths
of the files it reads (SPOT's is a5's ``case-sampled.json``, one-stage v2's case in its one-stage form);
beside its own Lean answer on that case, ``answer.json``, whose ``protocol`` is its id and ``commitment`` its digest
(null when its commitment didn't decode, and then the case's own commitment stands in, which a rejection never binds).
The inference-only policy's case (`verity/inference-only/v0`) is one subprotocol's like any other; its record's digest
must be the one SPOT's case commits to (and PoUW's, at remote), which the product checks.

A case's evidence is witness, and so are its params' bodies, so the output is split. ``--private-out`` (created 0700,
its files 0600) gets the product's params (``params.json``), commitment (``commitment.bin``: `bind` of the digests in
the order of the ids) and evidence (``evidence.json``), all canonical, as `Core.Composition.sameB` reads them (the
composition README's formats), the blobs' paths
(``blobs.txt``), each piece's answer (``pieces/N.answer.json``) and the product's (``answer.json``). An answer's
query, digests, profile and reason are private too (a partition's digest is the query), and so is a case's commitment
(Pearl+'s holds its passes' manifests, their salt key among them), so ``--public-out`` gets only
``composite.json``: the product, the composite's commitment, each case's id, params digest (SHA-512 of its canonical
params) and commitment's digest (its answer's; null where it names none), what is missing, and each piece's and Lean's
outcome (`outcome`: the id, the verdict and the
refusal's name) and its stated exceptions (`EXCEPTIONS`: the policy's v0, composed in the clear for one run of record
only); and ``verdicts.txt``, its ``PIECE ID VERDICT [NAME]``, ``NOT_ESTABLISHED``, ``VERDICT
ID VERDICT [NAME]`` and ``EXCEPTION LINE`` lines, which it also prints. Before writing them it refuses (exit 65) public bytes that share 64 bytes or more
with any case's commitment or evidence, raw or hex, the digests the answers name aside. The public directory may not be
or contain the private one.

With ``--exe``, it runs ``EXE check params.json commitment.bin evidence.json BLOB...`` and exits with its code: 0
accept, 1 reject, 2 undecided. Its record then says what ran the check (`provenance`, ``ran``, and a ``RAN`` line):
the commit of the tree it was built from (``--tree``), the executable's SHA-512, and the guarantee the accept cites
(`CITES`) with what that commit's lock says of it. An accept whose cited guarantee that lock doesn't list exits 66
(``UNCITED``); one with no cited guarantee, no commit or lock, or a tree that isn't its commit's bytes is labelled
``NOT_ESTABLISHED``. It decides nothing itself: a subprotocol with no case, or whose directory lacks its case
or its answer, is left out of the product's case, and Lean's verdict on that case is the composite's (a rejection
naming the missing case). Such a run is labelled ``NOT_ESTABLISHED``, naming what is missing. 64 is a usage error.

``--development WHY`` labels a composite that claims no record (one composed on the served node, which the prover
controls: `served_composite.sh`): ``composite.json`` carries ``development`` WHY, and ``DEVELOPMENT WHY`` is the first
line of its ``verdicts.txt`` and of what it prints after its ``COMPOSITE`` line.

``composite.json`` says the sampled proofs' ``coins``: ``live`` only when their case's challenger key is the one this
machine pins (``--key-sha512``), else ``simulated``, and ``coins_on`` the machine (``--coins-on``). On simulated coins an
accept's cite is kept and qualified: ``ran.not_covered`` names the coins and the machine, as does a ``NOT_COVERED``
line.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

PRODUCTS = {"onsite": "verity/onsite/v0", "remote": "verity/remote/v0", "remote-v1": "verity/remote/v1"}
#: Each product's subprotocols, as `Core.Onsite.components`, `Core.Remote.components` and `Core.Remote.V1.components`
#: list them; read only to name the missing ones.
COMPONENTS = {
    "onsite": ("verity/certifier/v0", "verity/inference-only/v0", "verity/pous/p2/v4/run", "verity/pouw/period/v1",
               "verity/remote-bandwidth/p2/v0", "verity/sampled-proofs/v1", "verity/wipe/v0"),
    "remote": ("verity/inference-only/v0", "verity/pouw/period/v1", "verity/sampled-proofs/v1"),
    "remote-v1": ("verity/inference-only/v0", "verity/pouw/pearl-plus/v1", "verity/sampled-proofs/v1"),
}
#: The subprotocols a product composes in the clear, as a stated exception to everything-zero-knowledge, each with the
#: line its record carries (`Core.Remote`'s and `Core.Onsite`'s docstrings say the same).
EXCEPTIONS = {
    "verity/inference-only/v0": (
        "stated exception: verity/inference-only/v0 publishes the declared division and each inference block's profile "
        "in the clear, composed for the run of record of 28 Oct only (Daniel's ruling, 27 Oct, 9:39 PM PDT: \"Let's run "
        "it in the clear tonight as a stated exception.\"); v1's private form (division committed under hiding leaves, "
        "compliance proved, padded profile) replaces it after 15:00Z on 28 Oct"),
}
#: The listed guarantee each product's accept rests on, as its lock names it: the one theorem over the compiled check its
#: executable runs. A product with none listed accepts with no theorem behind it, and its record says so.
CITES = {"remote-v1": "Verity.remote_soundness_of_subprotocols"}
#: The lock that lists `CITES`' guarantees, at the tree's commit.
LOCK = "security/Properties/lean-audit.json"
USAGE = 64
LEAK = 65
#: An accept whose cited guarantee the lock at the tree's commit doesn't list.
UNCITED = 66
#: Public bytes sharing a run of twice this with a case's evidence are refused: the evidence's windows at this stride,
#: looked up among every window of the public bytes, find any such run.
WINDOW = 32
VERDICTS = ("accept", "reject", "undecided")
#: A refusal's name is its reason up to the first colon (``params``, ``proofs``), public only when it is a short name and
#: carries no value.
NAME = re.compile(r"[A-Za-z][A-Za-z0-9_./ -]{0,63}")
VALUE = re.compile(r"[0-9A-Fa-f]{16}")
SAMPLED = "verity/sampled-proofs/v1"


class Parser(argparse.ArgumentParser):
    """argparse's, but a usage error exits 64, not 2, which is Lean's undecided."""

    def error(self, message: str):
        self.print_usage(sys.stderr)
        self.exit(USAGE, f"{self.prog}: error: {message}\n")


class Leak(Exception):
    pass


def canonical(x: Any) -> bytes:
    """`x` as `Core.canonicalJson` reads it, for the integers, strings, lists and objects of ASCII keys cases hold."""
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def bind(id: str, digests: Sequence[tuple[str, bytes]]) -> bytes:
    """`Core.Composition.bind`: SHA-512 of the product's id, a zero byte and the digests in order as canonical JSON."""
    body = canonical({"components": [{"commitment": d.hex(), "id": i} for i, d in digests]})
    return hashlib.sha512(id.encode() + b"\0" + body).digest()


def ran(d: Path) -> bool:
    return (d / "case.json").is_file() and (d / "answer.json").is_file()


def case_dirs(roots: Iterable[Path]) -> list[Path]:
    """Each ``--cases`` directory that holds a case, else its subdirectories that do."""
    out: list[Path] = []
    for r in roots:
        out += [r] if ran(r) else sorted(d for d in r.iterdir() if d.is_dir() and ran(d)) if r.is_dir() else []
    return out


def piece(d: Path) -> dict[str, Any]:
    """A subprotocol's case, digest and own answer line from its directory: ``{"id", "params", "commitment", "evidence",
    "digest", "named", "blobs", "answer"}``, the commitment, evidence and digest bytes, and ``named`` the digest its
    answer names, or None."""
    case = json.loads((d / "case.json").read_text())
    line = (d / "answer.json").read_text().strip()
    answer = json.loads(line)
    commitment = bytes.fromhex(case["commitment"])
    digest = answer.get("commitment")
    return {"id": answer["protocol"], "params": case["params"], "commitment": commitment,
            "evidence": bytes.fromhex(case["evidence"]), "digest": bytes.fromhex(digest) if digest else commitment,
            "named": digest or None, "blobs": [str(b) for b in case.get("blobs") or []], "answer": line}


def compose(product: str, pieces: Sequence[dict[str, Any]]) -> tuple[dict, bytes, bytes, list[str]]:
    """The product's params, commitment, evidence and blobs, the subprotocols in the order of their ids."""
    xs = sorted(pieces, key=lambda p: p["id"])
    params: dict[str, Any] = {"components": [{"id": p["id"], "params": p["params"]} for p in xs]}
    evidence: dict[str, Any] = {"components": [{"commitment": p["commitment"].hex(), "evidence": p["evidence"].hex(),
                                                "id": p["id"], "params": p["params"]} for p in xs]}
    commitment = bind(PRODUCTS[product], [(p["id"], p["digest"]) for p in xs])
    blobs = list(dict.fromkeys(b for p in xs for b in p["blobs"]))
    return params, commitment, canonical(evidence), blobs


def missing(product: str, pieces: Sequence[dict[str, Any]]) -> list[str]:
    """The product's subprotocols with no piece that ran."""
    ids = {p["id"] for p in pieces}
    return [i for i in COMPONENTS[product] if i not in ids]


def outcome(id: str, answer: str | None) -> dict[str, str]:
    """What of a Lean answer line is public: ``{"id", "verdict"}`` (``none`` when there is no answer), and ``refusal``,
    the name of the reason a verdict other than accept gives (``refused`` when that is not a short name, ``unparsed``
    when the line isn't JSON)."""
    try:
        a = json.loads(answer) if answer else {}
    except ValueError:
        return {"id": id, "verdict": "none", "refusal": "unparsed"}
    a = a if isinstance(a, dict) else {}
    out = {"id": id, "verdict": a["verdict"] if a.get("verdict") in VERDICTS else "none"}
    reason = a.get("reason")
    if out["verdict"] != "accept" and isinstance(reason, str) and reason:
        name = reason.split(":", 1)[0].strip()
        out["refusal"] = name if NAME.fullmatch(name) and not VALUE.search(name) else "refused"
    return out


def line(kind: str, o: dict[str, str]) -> str:
    return " ".join([kind, o["id"], o["verdict"], *([o["refusal"]] if "refusal" in o else [])])


def exceptions(product: str) -> list[str]:
    """The `EXCEPTIONS` lines of the product's subprotocols, in the order of their ids."""
    return [EXCEPTIONS[i] for i in COMPONENTS[product] if i in EXCEPTIONS]


def file_sha512(path: Path) -> str:
    h = hashlib.sha512()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def provenance(product: str, exe: Path, tree: Path) -> dict[str, Any]:
    """What ran the check: the tree's commit (``commit``, None outside a git checkout) and whether its tracked files are
    that commit's (``clean``), the executable's SHA-512 (``verifier_sha512``), and the guarantee the product's accept
    cites (`CITES`, ``cites``) with what the lock at that commit says of it (``lock``: ``listed``, ``deployed``, and
    its premises still open, ``open``; None where the commit holds no lock)."""
    git = lambda *args: subprocess.run(["git", "-C", str(tree), *args], capture_output=True, text=True)  # noqa: E731
    head = git("rev-parse", "HEAD")
    commit = head.stdout.strip() if head.returncode == 0 else None
    clean = commit is not None and git("status", "--porcelain", "--untracked-files=no").stdout == ""
    cites, lock = CITES.get(product), None
    shown = git("show", f"{commit}:{LOCK}") if commit is not None else None
    if cites is not None and shown is not None and shown.returncode == 0:
        audit = json.loads(shown.stdout)
        g = audit.get("guarantees", {}).get(cites)
        lock = {"listed": g is not None, "deployed": cites in audit.get("deployed", []),
                "open": sorted(((g or {}).get("premises") or {}).get("open", []))}
    return {"commit": commit, "clean": clean, "verifier_sha512": file_sha512(exe), "cites": cites, "lock": lock}


def members_reached(cases: Sequence[dict[str, Any]]) -> tuple[int, int] | None:
    """Of the sampled-proofs case in the per-member form, the members its accepted sample reached and the members it
    registers; None for any other case."""
    for p in cases:
        proofs = p["params"].get("proofs") if p["id"] == SAMPLED else None
        if not isinstance(proofs, dict) or not isinstance(proofs.get("members"), dict):
            continue
        try:
            a = json.loads(p["answer"])
        except ValueError:
            return None
        if not isinstance(a, dict) or a.get("verdict") != "accept":
            return None
        drawn = (a.get("public") or {}).get("drawn") or []
        ms = proofs["members"].values()
        return sum(any(lo <= u < hi for u in drawn for lo, hi in m["units"]) for m in ms), len(ms)
    return None


def uncited(p: dict[str, Any]) -> str | None:
    """Why the lock at provenance `p`'s commit refutes the guarantee its accept cites: it lists no such guarantee."""
    if p["cites"] is not None and p["lock"] is not None and not p["lock"]["listed"]:
        return f"{LOCK} at {p['commit']} lists no {p['cites']}"
    return None


def unestablished(p: dict[str, Any]) -> str | None:
    """What provenance `p` leaves unshown about its accept's theorem, short of a refutation (`uncited`)."""
    if p["cites"] is None:
        return "the product cites no listed guarantee"
    if p["commit"] is None:
        return "the tree is no git checkout"
    if p["lock"] is None:
        return f"{p['commit']} holds no {LOCK}"
    if not p["clean"]:
        return f"the tree's tracked files differ from {p['commit']}"
    return None


def sampled_coins(cases: Sequence[dict[str, Any]], key_sha512: str | None) -> str | None:
    """``live`` when the sampled-proofs case's challenger key (its params' ``challenger.key``) is the one this machine pins
    (``key_sha512``), else ``simulated``, a case naming no key and no pin included; None with no such case."""
    for p in cases:
        if p["id"] != SAMPLED:
            continue
        ch = p["params"].get("challenger")
        key = ch.get("key") if isinstance(ch, dict) else None
        public = key.get("public") if isinstance(key, dict) else None
        if not key_sha512 or not isinstance(public, str) or not re.fullmatch(r"(?:[0-9a-fA-F]{2})+", public):
            return "simulated"
        return "live" if hashlib.sha512(bytes.fromhex(public)).hexdigest() == key_sha512.lower() else "simulated"
    return None


def public_record(product: str, commitment: bytes, cases: Sequence[dict[str, Any]], gone: Sequence[str],
                  verdict: dict[str, str] | None, development: str | None = None,
                  ran: dict[str, Any] | None = None) -> dict[str, Any]:
    """What ``--public-out`` may hold: no evidence, no params bodies, no answer bodies and no case's commitment, only the
    digest its answer names; with ``ran``, what ran the check (`provenance`)."""
    return {"product": PRODUCTS[product], "commitment": commitment.hex(), "missing": list(gone), "verdict": verdict,
            "exceptions": exceptions(product), **({"development": development} if development else {}),
            **({"ran": ran} if ran is not None else {}),
            "cases": [{"params_sha512": hashlib.sha512(canonical(p["params"])).hexdigest(),
                       "commitment": p["named"]} | outcome(p["id"], p["answer"]) for p in cases]}


def check_public(public: bytes, cases: Sequence[dict[str, Any]]) -> None:
    """Refuse ``public`` when it shares a run of ``2 * WINDOW`` bytes or more with any case's commitment or evidence, raw
    or hex. The digests the cases' answers name are public by definition, and evidence may carry the same roots, so they
    are masked first; a case's commitment is held to it like its evidence, since one may be witness (Pearl+'s holds its
    passes' manifests and their salt key), and so is everything else, the pieces' own answers included."""
    for p in cases:
        if p["named"] and len(p["digest"]) >= WINDOW:
            public = public.replace(p["digest"].hex().encode(), b"*").replace(p["digest"], b"*")
    seen = {public[i:i + WINDOW] for i in range(len(public) - WINDOW + 1)}
    for p in cases:
        for what, b in (("commitment", p["commitment"]), ("evidence", p["evidence"])):
            for e in (b, b.hex().encode()):
                if any(e[i:i + WINDOW] in seen for i in range(0, len(e) - WINDOW + 1, WINDOW)):
                    raise Leak(f"the public output holds bytes of {p['id']}'s {what}")


def write_private(d: Path, files: dict[str, bytes]) -> None:
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(d, 0o700)
    for name, b in files.items():
        f = d / name
        f.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(f.parent, 0o700)
        fd = os.open(f, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(b)
        os.chmod(f, 0o600)


def run(a: argparse.Namespace) -> int:
    pub, priv = a.public_out.resolve(), a.private_out.resolve()
    if priv == pub or pub in priv.parents:
        print("composite.py: --private-out may not be or lie inside --public-out", file=sys.stderr)
        return USAGE
    found = [piece(d) for d in case_dirs(a.cases)]
    params, commitment, evidence, blobs = compose(a.product, found)
    files = {"params.json": canonical(params), "commitment.bin": commitment, "evidence.json": evidence,
             "blobs.txt": "".join(f"{b}\n" for b in blobs).encode()}
    files |= {f"pieces/{n}.answer.json": (p["answer"] + "\n").encode() for n, p in enumerate(found)}
    write_private(priv, files)
    cases = sorted(found, key=lambda p: p["id"])
    coins = sampled_coins(cases, a.key_sha512)
    gone = missing(a.product, found)
    lines = [f"DEVELOPMENT {a.development}"] if a.development else []
    lines += [line("PIECE", outcome(p["id"], p["answer"])) for p in cases]
    lines += [f"NOT_ESTABLISHED missing {', '.join(gone)}"] if gone else []
    cmd = [a.exe or a.product, "check", *(str(priv / n) for n in ("params.json", "commitment.bin", "evidence.json")),
           *blobs]
    rc, verdict, ran = 0, None, None
    if a.exe:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        answer = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else None
        write_private(priv, {"answer.json": f"{answer or ''}\n".encode(), "check.err": p.stderr.encode()})
        verdict = outcome(PRODUCTS[a.product], answer)
        lines.append(line("VERDICT", verdict) + ("" if answer else f" (exit {p.returncode}; its stderr is in --private-out)"))
        rc = p.returncode
        ran = provenance(a.product, Path(shutil.which(a.exe) or a.exe), a.tree)
        lines.append(f"RAN {ran['commit']} clean={ran['clean']} sha512={ran['verifier_sha512']} cites={ran['cites']}")
        if ran["cites"] is not None and (got := members_reached(cases)) is not None and got[0] < got[1]:
            ran["not_covered"] = {"id": SAMPLED, "members": {"sampled": got[0], "registered": got[1]}}
            lines.append(f"NOT_COVERED {SAMPLED} {got[0]} of {got[1]} members sampled: {ran['cites']} does not cover it")
        lines += [f"NOT_ESTABLISHED {why}"] if (why := unestablished(ran)) else []
        if rc == 0 and (why := uncited(ran)) is not None:
            lines.append(f"UNCITED {why}")
            rc = UNCITED
        if ran["cites"] is not None and coins == "simulated":
            ran.setdefault("not_covered", {"id": SAMPLED}).update(coins=coins, coins_on=a.coins_on)
            lines.append(f"NOT_COVERED {SAMPLED} coins: simulated on {a.coins_on}: {ran['cites']} does not cover it")
    lines += [f"EXCEPTION {x}" for x in exceptions(a.product)]
    record = canonical(public_record(a.product, commitment, cases, gone, verdict, a.development, ran)
                       | ({"coins": coins, "coins_on": {SAMPLED: a.coins_on}} if coins else {}))
    text = "".join(f"{ln}\n" for ln in lines).encode()
    try:
        check_public(record + b"\n" + text, found)
    except Leak as e:
        print(f"composite.py: {e}; nothing written to --public-out", file=sys.stderr)
        return LEAK
    pub.mkdir(parents=True, exist_ok=True)
    (pub / "composite.json").write_bytes(record + b"\n")
    (pub / "verdicts.txt").write_bytes(text)
    print("COMPOSITE " + json.dumps({"product": PRODUCTS[a.product], "components": [x["id"] for x in params["components"]],
                                     "commitment": commitment.hex(), "blobs": len(blobs)}, sort_keys=True), flush=True)
    for ln in lines:
        print(ln, flush=True)
    if not a.exe:
        print("CHECK " + " ".join(cmd), flush=True)
    return rc


def main(argv: Sequence[str] | None = None) -> int:
    ap = Parser(description=__doc__.split("\n\n")[0])
    ap.add_argument("product", choices=sorted(PRODUCTS))
    ap.add_argument("--cases", type=Path, action="append", default=[],
                    help="a private directory holding a subprotocol's case.json and answer.json, or such directories")
    ap.add_argument("--private-out", type=Path, required=True, help="the composed case and the answers (0700)")
    ap.add_argument("--public-out", type=Path, required=True, help="verdicts, commitment digests and params digests only")
    ap.add_argument("--exe", help="the product's executable, to run its check")
    ap.add_argument("--development", help="why this composite claims no record: its outputs lead with it")
    ap.add_argument("--key-sha512", help="the challenger key this machine pins: sampled proofs' coins are simulated unless "
                                         "their case's challenger key is it")
    ap.add_argument("--coins-on", default="unnamed", help="the machine sampled proofs' challenger flipped its coins on")
    ap.add_argument("--tree", type=Path, default=Path(__file__).resolve().parents[2],
                    help="the checkout the executable was built from (default: this file's)")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
