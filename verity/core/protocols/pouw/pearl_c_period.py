"""The Pearl-C period credit rule (CMP-82): a live Pearl-C pass earns credit only inside a Pearl-C period, and a period
earns only when every clause below holds, else nothing of it does. The Python reference of the Lean
`PouwCheck.pearlCPeriodCredit` (`lean/check/PouwCheck/Period.lean`), held to it by the difftest
(`lean/check/tests/test_difftest.py`'s ``pearl-c-period``).

A Pearl-C salt epoch is one challenger session the auditor's ledger reserves for the salt coin alone
(`window.SALT_FAMILIES`) and opens on the epoch's root R = `salt.epoch_open_root(W, k)`; every pass of the epoch reads
it (`core_check.live_salt`). Under that layout an abandoned session ends complete, so without this rule a prover could
open n epochs on fresh roots and present only the salt it liked (best-of-n). Here every session a credited period holds
is an accepted epoch's, a session outside every period earns nothing, and a root opens once.

The refusals are numbered by the clause each enforces (`REFUSALS`; 0, the inputs' form, is none of them):

1. ``exact``: `credit.period` passes with the presented passes' epoch roots (each pass's manifest's ``session.root``,
   the distinct ones in the order the passes first name them): the ledger opens and closes the period, its close is
   complete, and those roots are exactly its windows, in its order. So a complete salt-only session the close lists
   but no pass presents refuses the period.
2. ``salt-only``: every session the close lists is a Pearl-C salt session: the ledger reserves it for the salt coin
   alone and its open binds that reserve. An H1 window's session is reserved for H1's layout, so H1 and Pearl-C never
   share a period; H1's own rules refuse a Pearl-C session (`window.live_epoch_reason`).
3. ``session``: each pass's session counts toward the period (`challenger.credited_session`: the pin, then its
   ``descriptor``, hex, hashing to the circuit the period's open committed), within the period's pin floor
   (`credit._pin_hygiene`).
4. ``pass``: each pass passes `core_check.pass_epoch` with the verifier's live inputs; one refused pass refuses the
   period.
5. ``index``: for each weights' root W, the period's epochs on W, in the close's order, have k = n_W, n_W + 1, … with
   no gap or repeat, n_W one more than the largest k `earlier` (the lane's epochs credited in earlier Pearl-C periods,
   ``[W, k]`` pairs) lists on W, else 0. A W whose numbering broke (a root of it opened outside a credited period) is
   spent: that root opens once, so no later period continues it.
6. ``repeated-call``: no two passes of one epoch share a call index, and every call index is below 2^32 (clause 4's
   `core_check.CALL_RANGE` already refuses one that isn't; the period states it again so that its own check is what
   `Properties.Pouw.LiveSalt.PassesDistinct` reads). A pass presented twice shares its indices with itself.

The credit is the passes' manifests, in the order presented. Each pass reads the ledger through `pinned_session` twice
(clauses 3 and 4), so a period of P passes verifies the ledger 2P + 1 times."""
from __future__ import annotations

import json
from typing import Any

from verity.core.protocols.pouw import coins as Co
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import credit as Cr
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.sampled_proofs import challenger as Ch

#: the period's refusals, numbered by the rule's clause each enforces (``form``, 0, is its inputs')
FORM, EXACT, SALT_ONLY, SESSION, PASS, INDEX, REPEATED_CALL = REFUSALS = (
    "form", "exact", "salt-only", "session", "pass", "index", "repeated-call")
#: a presented pass: the window's context naming its manifest, the descriptor of the circuit its period committed (hex),
#: its manifest's bytes (hex) and its epoch's session's files (`core_check.LIVE`'s ``session``)
PASS_KEYS = ("context", "descriptor", "manifest", "session")
PASSES_FORM = ("the period's passes are a list of {context, descriptor, manifest, session}: the window's context naming "
               "the pass's manifest by its SHA-512 (a string), the circuit descriptor's bytes and the manifest's bytes "
               "in lowercase hex, and its epoch's session's files")
EARLIER_FORM = ("the lane's earlier credited epochs are a list of [W, k] pairs: a weights' root (a SHA-512 digest in "
                "lowercase hex) and an epoch number, an integer in [0, 2^64)")
NOT_SALT_ONLY = ("is not a Pearl-C salt session: the auditor's ledger doesn't reserve it for the salt coin alone, or "
                 "opens it binding no such reserve, and H1's sessions and Pearl-C's never share a period")
OUT_OF_TURN = ("a weights' root's epochs run 0, 1, 2 and on across its lane's credited periods with no gap or repeat, "
               "so a root whose numbering broke is spent")
REPEATED = "a call index names one call across its epoch, so no call's work is credited twice"
_SALT_LAYOUT = [list(f) for f in Wd.SALT_FAMILIES]
_U64 = 1 << 64


def _presented(passes: Any) -> tuple[list[tuple[bytes, dict, str]] | None, tuple[str, str] | None]:
    """Each pass's manifest bytes, the manifest and the epoch root it names, or the ``form`` refusal."""
    if not (type(passes) is list and all(isinstance(p, dict) and sorted(p) == list(PASS_KEYS)
                                         and type(p["context"]) is str for p in passes)):
        return None, (FORM, PASSES_FORM)
    out = []
    for i, p in enumerate(passes):
        h = p["manifest"]
        if not (type(h) is str and Cr._HEX.fullmatch(h)):
            return None, (FORM, f"pass {i}'s manifest is not its bytes in lowercase hex")
        raw = bytes.fromhex(h)
        try:
            m = CC._manifest(raw)
        except CC.Refused:
            return None, (FORM, f"pass {i}'s manifest is not a JSON object")
        s = m.get("session")
        r = s.get("root") if isinstance(s, dict) else None
        if type(r) is not str:
            return None, (FORM, f"pass {i}'s manifest names no epoch: its session's root is not a string")
        out.append((raw, m, r))
    return out, None


def _next_epochs(earlier: Any) -> dict[str, int] | None:
    """n_W for each W `earlier` lists, or None when it isn't a list of ``[W, k]`` pairs (`EARLIER_FORM`)."""
    if type(earlier) is not list:
        return None
    n: dict[str, int] = {}
    for e in earlier:
        if not (type(e) is list and len(e) == 2 and Cr._digest(e[0]) is not None and type(e[1]) is int
                and 0 <= e[1] < _U64):
            return None
        n[e[0]] = max(n.get(e[0], 0), e[1] + 1)
    return n


def _salt_only(ledger: list[str], name: str) -> tuple[str, str] | None:
    """Clause 2 over the close of period `name` in `ledger`, which `credit.period` has trusted."""
    entries = [json.loads(line) for line in ledger]
    close = next(e["body"] for e in entries if e["kind"] == "period-close" and e["body"]["period"] == name)
    for j, w in enumerate(close["windows"]):
        named = {e["kind"]: e for e in entries if e["session"] == w["session"]}
        reserve, opened = named.get("reserve"), named.get("open")
        if not (reserve is not None and opened is not None
                and CC._jeq(reserve["body"].get("families"), _SALT_LAYOUT)
                and CC._jeq(opened["body"].get("reserve"), reserve["seq"])):
            return SALT_ONLY, f"window {j} of period {name} (session {w['session']}) {NOT_SALT_ONLY}"
    return None


def pearl_c_period_credit(key: Any, ledger: Any, pins: Any, main: Any, name: Any, passes: Any, earlier: Any,
                          verify: Ch.Verify) -> tuple[list[dict] | None, tuple[str, str] | None]:
    """Pearl-C period `name`'s credit, the manifests of `passes` in order, and None; or None and the refusal (a code of
    `REFUSALS` and why), the first clause of the module's rule that fails, under the pinned challenger key `key`, the
    auditor's ledger `ledger`, the host's pins log `pins` and the verifier's copy of main's first-parent history
    `main`, the challenger's signatures checked with `verify`. Each of `passes` is ``{context, descriptor, manifest,
    session}`` (`PASS_KEYS`), and `earlier` the lane's epochs credited in earlier Pearl-C periods, ``[W, k]`` pairs."""
    got, why = _presented(passes)
    if why is not None:
        return None, why
    n_w = _next_epochs(earlier)
    if n_w is None:
        return None, (FORM, EARLIER_FORM)
    roots = list(dict.fromkeys(r for _, _, r in got))
    why = Cr.period(key, ledger, name, roots, verify)
    if why is not None:
        return None, (EXACT, f"the period's sessions are not exactly its passes' epochs, so it earns nothing: "
                             f"{why[0]}: {why[1]}")
    why = _salt_only(ledger, name)
    if why is not None:
        return None, why
    k = Ch.ChallengerKey.from_json(key)
    period_pin, floor = Cr.committed_pin(ledger, name), Cr.committed_floor(ledger, name)
    for i, (p, (_, _, r)) in enumerate(zip(passes, got)):
        j = roots.index(r)
        d = p["descriptor"]
        if not (type(d) is str and Cr._HEX.fullmatch(d)):
            return None, (SESSION, f"pass {i} of period {name}: {Cr.DESCRIPTOR_HEX}")
        try:
            audited = Ch.credited_session(ledger, {"root": r}, k, pins, main, bytes.fromhex(d), verify)
        except Ch.CoinError as e:
            return None, (SESSION, f"pass {i} of period {name}, of its window {j}, doesn't count toward it, so the "
                                   f"period earns nothing: {e}")
        why = Cr._pin_hygiene(name, j, main, audited.pinned, period_pin, floor)
        if why is not None:
            return None, (SESSION, f"{why[0]}: {why[1]}")
    for i, (p, (raw, _, _)) in enumerate(zip(passes, got)):
        live = {"key": key, "ledger": ledger, "main": main, "pins": pins, "session": p["session"]}
        _, why = CC.pass_epoch(None, raw, p["context"], live=live, verify=verify)
        if why is not None:
            return None, (PASS, f"pass {i} of period {name} is refused, so the period earns nothing: {why[0]}: {why[1]}")
    for j, r in enumerate(roots):
        m = next(m for _, m, q in got if q == r)
        w = CC.weights_root(Co.SCHEME, CC._workload(m)[1]).hex()
        want = n_w.get(w, 0)
        if m["pass_index"] != want:
            return None, (INDEX, f"window {j} of period {name} is epoch {m['pass_index']} of weights' root {w[:16]}..., "
                                 f"not epoch {want}: {OUT_OF_TURN}")
        n_w[w] = want + 1
    held: dict[str, dict[int, int]] = {}
    for i, (_, m, r) in enumerate(got):
        seen = held.setdefault(r, {})
        us = [u["index"] for u in CC._workload(m)[0]]
        for u in us:
            if not 0 <= u < CC.CALL_BOUND:
                return None, (REPEATED_CALL, f"pass {i} of period {name} holds call index {u}: {CC.CALL_RANGE}")
            q = seen.get(u)
            if q is not None:
                return None, (REPEATED_CALL, f"passes {q} and {i} of period {name} both hold call index {u} of epoch "
                                             f"{r[:16]}...: {REPEATED}")
        seen.update((u, i) for u in us)
    return [m for _, m, _ in got], None


__all__ = ["EARLIER_FORM", "EXACT", "FORM", "INDEX", "NOT_SALT_ONLY", "OUT_OF_TURN", "PASS", "PASSES_FORM", "PASS_KEYS",
           "REFUSALS", "REPEATED", "REPEATED_CALL", "SALT_ONLY", "SESSION", "pearl_c_period_credit"]
