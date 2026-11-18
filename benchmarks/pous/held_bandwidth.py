"""PoUB-P2's session over the ``C*`` the PoUS piece holds (`served_piece.py`): PoUB's prover
(`p2_party_prover.prove`) over PoUS's store as the vLLM adapter encoded it on the card, run by the same process after PoUS's
session and before the store is released, so both audits are of one store that one process holds, and no second store
is encoded. The session needs the card (`held`); its answer needs none, and is made once the card is released
(`decide`).

**The registration** is the auditor's, never the prover's: the bandwidth party (the auditor gate's ``bw_party
--pous-session SID``) builds it from the memory attester's own record of PoUS session SID on its host
(`challenger_party.keep`, `p2_parties.pous_record`): the node, P2's parameters, setup's commitment and salt as the request
declared them before that session, and the root the session bound. Its plan is the whole store. The prover (`bandwidth.Held`)
refuses a registration that isn't its store's (`hold`), and the session's case names the root (`p2_case.store_root`),
which `p2_case.same_store` holds to PoUS's run case. The party is given a fresh session id (``--session``), which the
memory attester claims in its session log before the first coin and under which it signs the case's close of the
verifier's log (`p2_parties.Closer`), under the key ``--trusted-sha512`` pins (`key_of`). `decide` trusts that key
and never one the case lists, since the case is kept in the developer's private directory.

**A pass** reads every unit of the plan from ``C*``'s rows on their device: σ by the exact int8 product against the
pass's signs (`bandwidth.Sketch`), y on the device (`receipts_of`), a few units held to `p2.Sketcher` (`spot`), and
the receipt tree (`ReceiptTree`). An opening is the unit's four rows from the device with their PoUS vk paths, from the
tree over the challenger's table (`row_leaf` of each row digest), whose root must be setup's.

**The outputs**: `held` leaves the session's record and its case, not yet answered (`HELD`), in the private directory;
`decide` answers that case. The session's core case and its answer (`p2_case.write`, the Lean's with
``poub-p2-protocol``) and everything that names the store (the registration, the verifier's logs, the session's
record, the hiding commitment's opening) go to the private directory; the public answer is the case's answer in the core
form under a hiding commitment to the case's digest (one ``hm96-sha512/row/v2`` row in a ``frame-v3-sha512`` tree,
`verity.commitments.advice`, bound to the window), its profile without the registration's digest.

**The session's length** is the request's: ``windows`` of ``window_s`` each (`WINDOWS`, `WINDOW_S`: two minutes), each of
``passes`` passes (`PASSES`) and ``samples`` openings (`SAMPLES`), by default the auditor's (`p2_parties.SESSION`, which
its party holds every request to). At δ = 2^-40 an accepted session's ε is δ + Σ samples/2^64, which leaves out
Σ_t adv_t (SHA-512 stream as PRF: not priced, not assumed; `p2_parties.EPS_TERMS`), and the pooled bound leaves at most
0.45% of its receipts wrong at those defaults over a store of 2^17
units (6 windows of 4 passes, 1,024 samples each; `pooled.bound`), so q_bytes is at least 99.5% of the reads.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "verity/protocols/resource_accounting/memory/poub/prover/p2_stage/bandwidth.py").parent))

import bandwidth as B

from verity.protocols.resource_accounting.memory.pous.attester import memory_attester as mc
from verity.protocols.resource_accounting.memory.pous.verifier import period
from verity.protocols.resource_accounting.memory.poub.verifier import ledger, p2_case, p2_parties
from verity.commitments import advice
from verity.commitments.identity import canonical_json_bytes

WINDOWS, PASSES, SAMPLES = (p2_parties.SESSION[k] for k in ("windows", "passes", "samples"))
WINDOW_S, DEADLINE_S = p2_parties.SESSION["window_ns"] / 1e9, p2_parties.SESSION["deadline_ns"] / 1e9
BINDING_TAG = b"verity/remote-bandwidth/served-window/v1"
#: the session's record, and its case until `decide` answers it, in the private directory
SESSION, HELD = "session.json", "held-case.json"
NO_KEY = ("no key is named for the bandwidth session's close: the configuration pins none for the bandwidth verifier's "
          "bandwidth sessions, and the run names none")
SCOPE = ("beside the served window: the session ran on the card after window {k} was served and PoUS's session bound the "
         "store, over that store as the PoUS piece encoded and held it, and was not interleaved with the engine's reads "
         "(serving is not hooked), so it certifies the session's own reads of the store, not the window's")


def binding(model: str, revision: str, window: int) -> bytes:
    """What the public answer's hiding commitment is bound to: the served model and the window."""
    doc = {"model": model, "revision": revision, "window": window}
    return hashlib.sha512(BINDING_TAG + canonical_json_bytes(doc)).digest()


def request(node: bytes, *, windows: int = WINDOWS, passes: int = PASSES, samples: int = SAMPLES,
            window_s: float = WINDOW_S, deadline_s: float = DEADLINE_S) -> dict[str, Any]:
    return p2_parties.request_of(node, windows=windows, passes=passes, samples=samples, window_ns=int(window_s * 1e9),
                               deadline_ns=int(deadline_s * 1e9))


def _no(verdict: str, reason: str) -> dict[str, Any]:
    return {"commitment": None, "profile": None, "protocol": p2_parties.PROTOCOL, "public": None, "reason": reason,
            "verdict": verdict}


def _write(path: Path, doc: Any, mode: int | None = None) -> None:
    path.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    if mode is not None:
        os.chmod(path, mode)


def public_answer(judged: Mapping[str, Any], bind: bytes, scope: str, r_bytes_per_s: int | None,
                  private: Path) -> dict[str, Any]:
    """The case's answer `judged` in public: its commitment a hiding one to the case's digest under `bind`, whose
    opening goes to `private`, its profile without the registration's digest and with `scope`, and γ against R. A
    refusal with no commitment, the Lean's of a case it decodes no commitment of, has none to hide."""
    if judged.get("commitment") is None and judged.get("verdict") != "accept":
        return dict(judged) | {"reason": f"{judged.get('reason')}; {scope}"}
    row = bytes.fromhex(judged["commitment"])
    hidden = advice.commit([row], 8 * len(row), bind)
    o = hidden.open(0)
    _write(private / "opening.json", {"binding": bind.hex(), "count": 1, "index": o.index, "n_bits": 8 * len(row),
                                      "path": [p.hex() for p in o.path], "root": hidden.root.hex(), "row": row.hex(),
                                      "salt": o.salt.hex()}, 0o600)
    out = dict(judged) | {"commitment": hidden.root.hex()}
    if judged.get("verdict") != "accept":
        out["reason"] = f"{judged.get('reason')}; {scope}"
        return out
    public = dict(judged["public"])
    profile = {k: v for k, v in judged["profile"].items() if k != "registration"} | {
        "scope": scope, "eps_terms": p2_case.EPS_TERMS, "named_terms": ledger.NAMED_TERMS,
        "commitment": {"scheme": "hm96-sha512/row/v2 in frame-v3-sha512 (verity.commitments.advice)", "rows": 1,
                       "n_bits": 8 * len(row), "binding": bind.hex(),
                       "of": "the session's case digest (p2_case.digest), its registration and plan"}}
    if r_bytes_per_s is not None and public["span_ns"] > 0:
        from fractions import Fraction

        public |= {"gamma": str(1 - Fraction(public["q_bytes"] * 10**9, r_bytes_per_s * public["span_ns"])),
                   "r_bytes_per_s": r_bytes_per_s}
        profile["gamma_terms"] = p2_parties.GAMMA_TERMS
    return out | {"public": public, "profile": profile}


def key_of(deployment: Mapping[str, Any], trusted_sha512: str | None = None) -> str | None:
    """`trusted_sha512`, a rehearsal's, else the configuration's pin for the close."""
    pinned = mc.pinned(deployment["pins"], p2_parties.CLOSER_PARTY, p2_parties.COMPONENT)
    return trusted_sha512 or min(pinned, default=None)


def held(store: Any, *, node: str, card: str, party: str, party_host: str, sid: bytes, request_: Mapping[str, Any],
         private: Path, trusted_sha512: str | None = None) -> None:
    """One PoUB-P2 session over the held `store` with the bandwidth party `party` (a shell command, to which
    ``--pous-session`` and PoUS's session id `sid` are added, and ``--session`` and a fresh id for this session, under
    which the memory attester signs the case's close; without `trusted_sha512`, no session has a case; its port reached
    at `party_host`), under `request_`:
    the session's record (`SESSION`: the party's answer and its own record of it, the registration, each pass and
    opening, the seconds) and, where the session made one, its case not yet answered (`HELD`), with the logs, to
    `private` (0700, files 0600), for `decide`. This is the part that reads the store. γ is against the data sheet's
    stated memory bandwidth of `card`, the name nvidia-smi gives the card (`kit.datasheet_bandwidth`), read before the
    session starts and refused by name where its record has none."""
    from verity_bench import kit

    r_bytes_per_s = kit.datasheet_bandwidth(card)
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(private, 0o700)
    (private / HELD).unlink(missing_ok=True)   # an earlier session's case is never this one's
    p2_session = None if trusted_sha512 is None else secrets.token_bytes(32)
    command = f"{party} --pous-session {sid.hex()}"
    if p2_session is not None:
        command += f" --session {p2_session.hex()} --trusted-sha512 {trusted_sha512}"
    with open(private / "prover.log", "a") as log_f:

        def log(m: str) -> None:
            print(m, file=log_f, flush=True)

        t0 = time.perf_counter()
        record = rc = None
        prover = None
        try:
            prover = B.Held(store, node.encode(), log=log)
            setup_s = time.perf_counter() - t0
            answer, record, rc = B._party(SimpleNamespace(party=command, party_host=party_host), private, prover,
                                          request_)
        except Exception as e:  # noqa: BLE001 - the prover's own failure, its answer, never the verifier's
            log(json.dumps({"prover_error": f"{type(e).__name__}: {e}"}))
            answer, setup_s = {"verdict": "prover-error", "reason": f"{type(e).__name__}: {e}"[:1000]}, None
        wall = time.perf_counter() - t0
    if prover is not None and prover.kept is not None:
        fd = os.open(private / HELD, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(canonical_json_bytes(dict(prover.kept)) + b"\n")
    _write(private / SESSION, {
        "answer": answer, "verifier_record": record, "verifier_rc": rc, "request": dict(request_),
        "registration": None if prover is None else p2_parties.registration_json(prover.registration, prover.plan),
        "steps": None if prover is None else prover.steps, "opened": None if prover is None else prover.opened,
        "pous_session": sid.hex(), "session": None if p2_session is None else p2_session.hex(), "party": command,
        "party_host": party_host, "card": card, "r_bytes_per_s": r_bytes_per_s,
        "setup_s": None if setup_s is None else round(setup_s, 3), "session_wall_s": round(wall, 3)}, 0o600)


def key_record(case: Mapping[str, Any], trusted_sha512: str, deployment: Mapping[str, Any], host: str) -> dict[str, Any]:
    """``key_pin`` is ``run``: the pins this machine was handed are used only to refuse, and ``pins_matched`` says
    whether they give the key `trusted_sha512` names for the close."""
    key = p2_case.close_key(case)
    key_sha = None if key is None else hashlib.sha512(key).hexdigest()
    pinned = mc.pinned(deployment["pins"], p2_parties.CLOSER_PARTY, p2_parties.COMPONENT)
    return {"key_sha512": key_sha, "key_pinned": key_sha == trusted_sha512,
            "key_pin": "run", "pins_matched": trusted_sha512 in pinned,
            mc.PREMISE: mc.trusted_device(
                deployment["pins"], p2_parties.CLOSER_PARTY, p2_parties.COMPONENT, key_sha,
                f"the bandwidth verifier, which signs the close with the memory attester's key on {host}",
                configuration=deployment["configuration"], committed=deployment["committed"])}


def decide(private: Path, public: Path, *, bind: bytes, window: int, pous_case: Mapping[str, Any] | None = None,
           case_exe: str | None = None, trusted_sha512: str | None = None,
           deployment: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The answer to the session `held` left in `private`, with no store and no card: the case and its answer to
    `private` (`p2_case.write`), the public answer and a summary to `public` (``answer.json``, ``bandwidth.json``); the
    public answer, also returned. With `pous_case` (PoUS's run case), a case whose store isn't PoUS's
    (`p2_case.same_store`) is refused. With `case_exe` (``poub-p2-protocol``) the Lean answers the case and
    its answer is the one written, in ``lean_s`` seconds; a Lean that doesn't answer leaves the session undecided and no
    case written; without it the session's reject or undecided stands and its accept is undecided
    (`p2_case.unjudged`). With no key `key_of` gives, the session is undecided (`NO_KEY`). The summary names the program whose verdict the answer is and the host it ran on
    (``decided_by``, `period.decided_by`): the Lean where it answered the case, else this Python. `HELD` stays, so the
    same session can be answered again; the caller removes it once it needs no other answer."""
    public.mkdir(parents=True, exist_ok=True)
    scope = SCOPE.format(k=window)
    deployed = period.deployment({}) if deployment is None else deployment
    trusted_sha512 = key_of(deployed, trusted_sha512)
    doc = json.loads((private / SESSION).read_text())
    answer, record, steps = doc["answer"], doc["verifier_record"], doc["steps"] or []
    r_bytes_per_s = doc["r_bytes_per_s"]
    kept = json.loads((private / HELD).read_text()) if (private / HELD).is_file() else None
    summary: dict[str, Any] = {"window": window, "request": doc["request"], "party_host": doc["party_host"],
                               "pous_session": "the PoUS session the PoUS piece ran (its id private)",
                               "private": str(private), "decided_by": period.decided_by(None),
                               "setup_s": doc["setup_s"], "session_wall_s": doc["session_wall_s"],
                               "verifier_rc": doc["verifier_rc"], "passes": len(steps),
                               "pass_s": [s["pass_s"] for s in steps]}
    if record is not None and record != answer:
        out = _no("reject", f"the verifier's own record of its answer ({record.get('verdict')}) is not the session's "
                            f"({answer.get('verdict')}); {scope}")
    elif trusted_sha512 is None:
        out = _no("undecided", f"{NO_KEY}; {scope}")
    elif kept is None:
        verdict = "undecided" if answer.get("verdict") in ("prover-error", "refused", None) else "reject"
        out = _no(verdict, f"the session made no case: {answer.get('verdict')}: {answer.get('reason')}"[:1000]
                  + f"; {scope}")
    elif pous_case is not None and (why := p2_case.same_store(kept, pous_case)) is not None:
        out = _no("reject", f"the bandwidth session's store is not PoUS's: {why}; {scope}")
    else:
        summary["same_store"] = pous_case is not None
        try:
            kept = p2_case.trusting(kept, trusted_sha512)
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            out = _no("reject", f"the session's close doesn't decode: {type(e).__name__}: {e}"[:1000] + f"; {scope}")
        else:
            summary |= key_record(kept, trusted_sha512, deployed, doc["party_host"])
            t = time.perf_counter()
            try:
                judged = (p2_case.judged(kept, answer, case_exe) if case_exe is not None
                          else p2_case.unjudged(kept, answer))
            except (OSError, IndexError, ValueError, subprocess.CalledProcessError) as e:
                out = _no("undecided",
                          f"{case_exe} doesn't answer the case: {type(e).__name__}: {e}"[:1000] + f"; {scope}")
            else:
                if case_exe is not None:
                    summary |= {"decided_by": period.decided_by(case_exe), "lean_s": round(time.perf_counter() - t, 3)}
                p2_case.write(private, kept, judged)
                out = public_answer(judged, bind, scope, r_bytes_per_s, private)
    _write(public / "answer.json", out)
    _write(public / "bandwidth.json", summary | {"verdict": out["verdict"], "reason": out["reason"]})
    return out


def missing(public: Path, window: int, reason: str) -> dict[str, Any]:
    """No session ran: the public answer undecided, saying why, by this Python (``decided_by``)."""
    public.mkdir(parents=True, exist_ok=True)
    out = _no("undecided", f"{reason}; {SCOPE.format(k=window)}")
    _write(public / "answer.json", out)
    _write(public / "bandwidth.json", {"window": window, "verdict": out["verdict"], "reason": out["reason"],
                                       "decided_by": period.decided_by(None)})
    return out


def line(answer: Mapping[str, Any]) -> str:
    if answer["verdict"] == "accept":
        return f"BANDWIDTH: accept: {json.dumps(answer['public'], sort_keys=True)}"
    return f"BANDWIDTH: {answer['verdict']}: {answer['reason']}"


__all__ = ["HELD", "NO_KEY", "SCOPE", "SESSION", "binding", "decide", "held", "key_of", "key_record", "line",
           "missing", "public_answer", "request"]
