"""PoUB-P2's session over the ``C*`` the PoUS piece holds (`served_piece.py`): the remote bandwidth audit's prover
(`p2_party.prove`) over PoUS's store as the vLLM adapter encoded it on the card, run by the same process after PoUS's
session and before the store is released, so both audits are of one store that one process holds, and no second store
is encoded.

**The registration** is the auditor's, never the prover's: the bandwidth party (the auditor gate's ``bw_party
--pous-session SID``) builds it from the memory challenger's own record of PoUS session SID on its host
(`challenger_party.keep`, `p2_party.pous_record`): the node, P2's parameters, setup's commitment and salt as the request
declared them before that session, and the root the session bound. Its plan is the whole store. The prover's `Held`
refuses a registration that isn't its store's (`hold`), and the session's case names the root (`p2_case.store_root`),
which `p2_case.same_store` holds to PoUS's run case.

**A pass** reads every unit of the plan from ``C*``'s rows on their device: σ by the exact int8 product against the
pass's signs (`p2_v1.bandwidth.Sketch`), y on the device (`receipts_of`), a few units held to `p2.Sketcher` (`spot`), and
the receipt tree (`ReceiptTree`). An opening is the unit's four rows from the device with their PoUS vk paths, from the
tree over the challenger's table (`row_leaf` of each row digest), whose root must be setup's.

**The outputs** (`session`): the session's core case and its answer (`p2_case.write`, the Lean's with
``remote-audit-p2-protocol``) and everything that names the store (the registration, the verifier's logs, the session's
record, the hiding commitment's opening) go to the private directory; the public answer is the case's answer in the core
form under a hiding commitment to the case's digest (one ``hm96-sha512/row/v2`` row in a ``frame-v3-sha512`` tree,
`verity.ml.commitments.advice`, bound to the window), its profile without the registration's digest.

**The session's length** is the request's: ``windows`` of ``window_s`` each (`WINDOWS`, `WINDOW_S`: two minutes), each of
``passes`` passes (`PASSES`) and ``samples`` openings (`SAMPLES`), by default the auditor's (`p2_party.SESSION`, which
its party holds every request to). At δ = 2^-40 an accepted session's ε is δ + Σ draws/2^64, and the pooled bound leaves at most 0.45% of its receipts wrong at those defaults over a store of 2^17
units (6 windows of 4 passes, 1,024 draws each; `pooled.bound`), so q_bytes is at least 99.5% of the reads.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from p2_v1 import bandwidth as B

from verity.core.protocols.pous import verifier as V
from verity.core.protocols.remote_bandwidth import coins, p2, p2_case, p2_party, receipts
from verity.ml.commitments import advice
from verity.ml.commitments.identity import canonical_json_bytes

UNIT, BLOCK, DIGEST = p2.UNIT_BLOCKS, p2.BLOCK_BYTES, receipts.DIGEST_BYTES
SALT, PER_LEAF = receipts.SALT_BYTES, receipts.PER_LEAF
WINDOWS, PASSES, SAMPLES = (p2_party.SESSION[k] for k in ("windows", "passes", "samples"))
WINDOW_S, DEADLINE_S = p2_party.SESSION["window_ns"] / 1e9, p2_party.SESSION["deadline_ns"] / 1e9
#: an H200's specified HBM read rate, bytes a second: what γ is against
R_BYTES_PER_S = 4_800_000_000_000
BINDING_TAG = b"verity/remote-bandwidth/served-window/v1"
SCOPE = ("beside the served window: the session ran on the card after window {k} was served and PoUS's session bound the "
         "store, over that store as the PoUS piece encoded and held it, and was not interleaved with the engine's reads "
         "(serving is not hooked), so it certifies the session's own reads of the store, not the window's")


class Held:
    """`p2_party.Prover` over the adapter's store ``store`` (`verity_vllm.protocols.pous._Store`): ``C*``'s rows on
    their device, the challenger's table of their digests, setup's salt, commitment and root, under the node ``node``."""

    def __init__(self, store: Any, node: bytes, *, chunk: int = 1 << 16, spot: int = 16,
                 log: Callable[[str], None] = lambda _m: None) -> None:
        import torch

        self.torch, self.rows, self.chunk, self.spot, self.log = torch, store.rows, chunk, spot, log
        params = store.params
        self.registration = p2.Registration(node, p2.codec_of(params), store.commitment, store.salt, store.root)
        digests = store.digests
        self.tree = V.MerkleTree([V.row_leaf(params, store.salt, j, digests[DIGEST * j:DIGEST * (j + 1)])
                                  for j in range(params.blocks)])
        if self.tree.root != store.root:
            raise ValueError("the challenger's table of the held C* is not setup's root")
        self.plan = receipts.Plan.whole(self.registration.blocks)
        self.passes: dict[tuple[int, int], tuple[np.ndarray, bytes, B.ReceiptTree]] = {}
        self.received: dict[int, coins.Released] = {}
        self.steps: list[dict[str, Any]] = []
        self.opened: list[dict[str, Any]] = []
        self.kept: dict[str, Any] | None = None

    def case(self, doc: dict[str, Any]) -> None:
        self.kept = doc

    def hold(self, registration: Mapping[str, Any]) -> None:
        if p2_party.registration_of(registration) != self.registration:
            raise ValueError("the verifier's registration is not the held store's")
        self.plan = p2_party.plan_of(registration, self.registration)

    def end(self, commitment: coins.LiveCommitment, openings: Sequence[coins.LiveOpening]) -> None:
        why = coins.received_reason(commitment, [self.received[t] for t in sorted(self.received)], openings)
        if why is not None:
            raise ValueError(f"the session's coins are not the ones this prover was sent: {why}")

    def unit(self, g: int) -> bytes:
        return self.rows[UNIT * g:UNIT * (g + 1)].cpu().numpy().tobytes()

    def sigma(self, seed: bytes) -> Any:
        """σ of every unit up to the plan's last, from the held rows on their device."""
        torch, units = self.torch, max(f + c for f, c in self.plan.runs)
        torch.backends.cuda.matmul.allow_tf32 = False
        sk = B.Sketch(torch, seed, self.registration.rows, units, self.rows.device)
        for j0 in range(0, UNIT * units, self.chunk):
            sk.add(self.rows[j0:min(j0 + self.chunk, UNIT * units)], j0)
        if sk.covered(self.plan.runs) != self.plan.length:
            raise RuntimeError(f"the pass sketched {sk.covered(self.plan.runs)} of the plan's {self.plan.length} units")
        return sk.sigma

    def _spot(self, seed: bytes, sigma: Any, ys: np.ndarray) -> int:
        """The plan's first and last positions and `spot` drawn from the seed: σ held to `p2.Sketcher` on the unit's
        bytes and y to hashlib."""
        plan, sk = self.plan, p2.Sketcher(seed, self.registration.rows)
        rng = np.random.default_rng(int.from_bytes(seed[:8], "big"))
        positions = sorted({0, plan.length - 1, *rng.integers(0, plan.length, self.spot).tolist()})
        for p in positions:
            g = plan.block(p)
            got = bytes(sigma[g].cpu().numpy().astype("<i4").tobytes())
            if got != sk.sketch(self.unit(g)) or bytes(ys[p]) != sk.receipt(g, got):
                raise RuntimeError(f"unit {g}'s σ or y is not the reference's")
        return len(positions)

    def run(self, challenge: coins.Released, t: int, i: int, previous: bytes) -> bytes:
        if challenge.label != coins.CHALLENGE or self.received.setdefault(challenge.index, challenge) != challenge:
            raise ValueError(f"not window {challenge.index}'s one challenge")
        seed = receipts.seed(coins.coin(challenge), i, previous)
        t0 = time.perf_counter()
        sigma = self.sigma(seed)
        a = time.perf_counter()
        parts = [B.receipts_of(self.torch, seed, sigma[f:f + c], first=f) for f, c in self.plan.runs]
        ys = parts[0] if len(parts) == 1 else np.concatenate(parts)
        b = time.perf_counter()
        spotted = self._spot(seed, sigma, ys)
        del sigma
        salts = os.urandom(SALT * receipts.leaves(self.plan.length))
        tree = B.ReceiptTree(seed, self.plan, ys, salts)
        self.passes[t, i] = (ys, salts, tree)
        rec = {"window": t, "pass": i, "sketch_s": round(a - t0, 3), "receipts_s": round(b - a, 3),
               "spot_units": spotted, "pass_s": round(time.perf_counter() - t0, 3)}
        self.steps.append(rec)
        self.log(json.dumps({"pass": rec}))
        return tree.root

    def open(self, t: int, draws: Sequence[tuple[int, int]]) -> list[p2.Opening]:
        t0 = time.perf_counter()
        out = []
        for i, p in draws:
            ys, salts, tree = self.passes[t, i]
            c, g = p // PER_LEAF, self.plan.block(p)
            unit = self.unit(g)
            out.append(p2.Opening(i, p, tuple(unit[BLOCK * b:BLOCK * (b + 1)] for b in range(UNIT)),
                                  tuple(self.tree.path(UNIT * g + b) for b in range(UNIT)),
                                  ys[PER_LEAF * c:min(PER_LEAF * (c + 1), self.plan.length)].tobytes(),
                                  salts[SALT * c:SALT * (c + 1)], tree.path(c)))
        for key in [k for k in self.passes if k[0] <= t]:
            del self.passes[key]
        rec = {"window": t, "openings": len(out), "open_s": round(time.perf_counter() - t0, 3)}
        self.opened.append(rec)
        self.log(json.dumps({"open": rec}))
        return out


def binding(model: str, revision: str, window: int) -> bytes:
    """What the public answer's hiding commitment is bound to: the served model and the window."""
    doc = {"model": model, "revision": revision, "window": window}
    return hashlib.sha512(BINDING_TAG + canonical_json_bytes(doc)).digest()


def request(node: bytes, *, windows: int = WINDOWS, passes: int = PASSES, samples: int = SAMPLES,
            window_s: float = WINDOW_S, deadline_s: float = DEADLINE_S) -> dict[str, Any]:
    return p2_party.request_of(node, windows=windows, passes=passes, samples=samples, window_ns=int(window_s * 1e9),
                               deadline_ns=int(deadline_s * 1e9))


def _no(verdict: str, reason: str) -> dict[str, Any]:
    return {"commitment": None, "profile": None, "protocol": p2_party.PROTOCOL, "public": None, "reason": reason,
            "verdict": verdict}


def _write(path: Path, doc: Any, mode: int | None = None) -> None:
    path.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    if mode is not None:
        os.chmod(path, mode)


def public_answer(judged: Mapping[str, Any], bind: bytes, scope: str, r_bytes_per_s: int | None,
                  private: Path) -> dict[str, Any]:
    """The case's answer `judged` in public: its commitment a hiding one to the case's digest under `bind`, whose
    opening goes to `private`, its profile without the registration's digest and with `scope`, and γ against R."""
    row = bytes.fromhex(judged["commitment"])
    hidden = advice.register([row], 8 * len(row), bind)
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
        "scope": scope, "eps_terms": p2_case.EPS_TERMS,
        "commitment": {"scheme": "hm96-sha512/row/v2 in frame-v3-sha512 (verity.ml.commitments.advice)", "rows": 1,
                       "n_bits": 8 * len(row), "binding": bind.hex(),
                       "of": "the session's case digest (p2_case.digest), its registration and plan"}}
    if r_bytes_per_s is not None and public["span_ns"] > 0:
        from fractions import Fraction

        public |= {"gamma": str(1 - Fraction(public["q_bytes"] * 10**9, r_bytes_per_s * public["span_ns"])),
                   "r_bytes_per_s": r_bytes_per_s}
        profile["gamma_terms"] = p2_party.GAMMA_TERMS
    return out | {"public": public, "profile": profile}


def session(store: Any, *, node: str, party: str, party_host: str, sid: bytes, request_: Mapping[str, Any],
            private: Path, public: Path, bind: bytes, window: int, pous_case: Mapping[str, Any] | None = None,
            case_exe: str | None = None, r_bytes_per_s: int | None = R_BYTES_PER_S) -> dict[str, Any]:
    """One PoUB-P2 session over the held `store` with the bandwidth party `party` (a shell command, to which
    ``--pous-session`` and PoUS's session id `sid` are added; its port reached at `party_host`), under `request_`: the
    case and everything naming the store to `private` (0700, files 0600), the public answer and a summary to `public`
    (``answer.json``, ``bandwidth.json``); the public answer, also returned. With `pous_case` (PoUS's run case), a case
    whose store isn't PoUS's (`p2_case.same_store`) is refused."""
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(private, 0o700)
    public.mkdir(parents=True, exist_ok=True)
    scope = SCOPE.format(k=window)
    command = f"{party} --pous-session {sid.hex()}"
    summary: dict[str, Any] = {"window": window, "request": dict(request_), "party_host": party_host,
                               "pous_session": "the PoUS session the PoUS piece ran (its id private)",
                               "private": str(private), "judge": "lean" if case_exe else "python"}
    with open(private / "prover.log", "a") as log_f:

        def log(m: str) -> None:
            print(m, file=log_f, flush=True)

        t0 = time.perf_counter()
        record = rc = None
        prover = None
        try:
            prover = Held(store, node.encode(), log=log)
            setup_s = time.perf_counter() - t0
            answer, record, rc = B._party(SimpleNamespace(party=command, party_host=party_host), private, prover,
                                          request_)
        except Exception as e:  # noqa: BLE001 - the prover's own failure, its answer, never the verifier's
            log(json.dumps({"prover_error": f"{type(e).__name__}: {e}"}))
            answer, setup_s = {"verdict": "prover-error", "reason": f"{type(e).__name__}: {e}"[:1000]}, None
        wall = time.perf_counter() - t0
    kept = prover.kept if prover is not None else None
    _write(private / "session.json", {
        "answer": answer, "verifier_record": record, "verifier_rc": rc, "request": dict(request_),
        "registration": None if prover is None else p2_party.registration_json(prover.registration, prover.plan),
        "steps": None if prover is None else prover.steps, "opened": None if prover is None else prover.opened,
        "pous_session": sid.hex(), "party": command, "party_host": party_host}, 0o600)
    summary |= {"setup_s": None if setup_s is None else round(setup_s, 3), "session_wall_s": round(wall, 3),
                "verifier_rc": rc, "passes": len(prover.steps) if prover else 0,
                "pass_s": [s["pass_s"] for s in prover.steps] if prover else []}
    if record is not None and record != answer:
        out = _no("reject", f"the verifier's own record of its answer ({record.get('verdict')}) is not the session's "
                            f"({answer.get('verdict')}); {scope}")
    elif kept is None:
        verdict = "undecided" if answer.get("verdict") in ("prover-error", "refused", None) else "reject"
        out = _no(verdict, f"the session made no case: {answer.get('verdict')}: {answer.get('reason')}"[:1000]
                  + f"; {scope}")
    elif pous_case is not None and (why := p2_case.same_store(kept, pous_case)) is not None:
        out = _no("reject", f"the bandwidth session's store is not PoUS's: {why}; {scope}")
    else:
        judged = p2_case.judged(kept, answer, case_exe)
        p2_case.write(private, kept, judged)
        out = public_answer(judged, bind, scope, r_bytes_per_s, private)
        summary["same_store"] = pous_case is not None
    _write(public / "answer.json", out)
    _write(public / "bandwidth.json", summary | {"verdict": out["verdict"], "reason": out["reason"]})
    return out


def missing(public: Path, window: int, reason: str) -> dict[str, Any]:
    """No session ran: the public answer undecided, saying why."""
    public.mkdir(parents=True, exist_ok=True)
    out = _no("undecided", f"{reason}; {SCOPE.format(k=window)}")
    _write(public / "answer.json", out)
    _write(public / "bandwidth.json", {"window": window, "verdict": out["verdict"], "reason": out["reason"]})
    return out


def line(answer: Mapping[str, Any]) -> str:
    if answer["verdict"] == "accept":
        return f"BANDWIDTH: accept: {json.dumps(answer['public'], sort_keys=True)}"
    return f"BANDWIDTH: {answer['verdict']}: {answer['reason']}"


__all__ = ["Held", "SCOPE", "binding", "line", "missing", "public_answer", "request", "session"]
