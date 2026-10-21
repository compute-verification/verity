"""A served window's credited set, its drawn units and their outcome, its A rows, and the period it counts toward, as
PoUW's verifier reads them (RT4-INT-32, -34 and -42; PRF-132): the rules a served window's verifier applies beside
`core_check.rounds`. This is the Python
reference of the Lean `PouwCheck.Credit` (`lean/check`), held to it by the difftest (`lean/check/tests/test_difftest.py`).

* **The credited set is fixed before the draw** (`credited`). Each row of the window's manifest (the serving process's
  call rows, in order, ``index`` 0, 1, …) carries its call's verdict as ``credit`` (true or false), and a credited row
  carries its z table's root (``tables.z``). `calls_root` hashes the rows (and the served trees' digest when the window
  has them), and `draw_root` hashes that, the last round's receipts and every round's root and coin. The challenger's
  draw coin must have followed that one root, as its signed log shows, read through the session the auditor's ledger
  opens for the window (`challenger.trusted_session_of`), never a commitment the prover hands in. So a credit bit
  changed after the draw coin changes the draw root and is refused, and the credited set is the rows whose bit is true.
* **The drawn units are the draw coin's** (`drawn`): `core_check.draw`'s rule under the law ``work:k``
  (`core_check.work_units`), over the window's own units. The window's layout is its Program's,
  ``h1window.program(linears, calls, m, tile)``, and its units are in that Program's order: call by call, each call's
  tiles row block major. There is one stratum per template, that is per K (`strata`), each unit's work its tile's
  R·C·K served MACs, at K = min(`RECORD_K`, N) (`law`). The served window's arguments are fixed (`window`): the stream
  is the window's root, the registration its draw root, and the receipt ``{"receipts": …}``, its last round's receipts.
* **No partial credit** (`outcome`): the window is accepted only when every unit the draw coin draws has exactly one
  proof outcome and it is an accept. A drawn unit whose proof failed or is missing refuses the whole window, not only
  its call. A unit with two outcomes is refused, since a retry never reproves, and so is an outcome for a unit the draw
  didn't select. `window` is the whole rule on one window: `credited`, then `outcome` of the coin it read.
* **A period counts only closed and complete** (`period`, RT4-INT-42): its windows are exactly those the auditor's
  ledger lists in the period's close, in that order, each opened and ended complete. A period with no close, or with
  a window missing, out of order or not its own, earns nothing, so no window is chosen after the fact. Under
  `period_credit` each window must also prove the circuit the period's open committed (`challenger.credited_session`:
  the SHA-512 of the descriptor its proofs were verified against), and a refused window refuses its period.
* **The A rows are the round roots'** (`a_link`): each registered A value of the window (a value named ``…/a/<K>``,
  `h1window.roles`) over L rows and S steps is the tree whose rows ``[s·L/S, (s+1)·L/S)`` round s's root commits as
  aligned subtrees. Each round's ``a`` lists, for every A value, its ``[level, index, root]`` subtrees in row order, and
  the root the challenger received for round s is `round_root` of the serving process's own round root (``inner``) and
  those segments. The verifier recomputes each value's root from the segments and its domain's pads. So the rows a
  unit reads are the ones fixed before each round's coin.

`calls_root` and `draw_root` are vllm's `pouw_h1.calls_root` and `pouw_h1_requests.draw_root`, byte for byte. Every
refusal is (code, why), the first rule the input breaks, its why naming the fix; the Lean port writes the same strings.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.protocol import tagged
from verity.core.protocols.sampled_proofs import challenger as Ch

CALLS_TAG = b"verity/pouw/h1/calls/v2\0"
DRAW_TAG = "verity/pouw/h1/draw/v1"
ROUND_TAG = "verity/pouw/h1/round-a/v1"
#: each manifest row's credit bit
CREDIT = "credit"
#: how deep a manifest row nests
DEPTH = 64
VERDICTS = ("accept", "reject")
#: a segment's level is below this
LEVELS = 64
#: an A value's rows: the verifier hashes each pad past them, at most as many again
MAX_ROWS = 1 << 24
_DIGEST = re.compile(r"[0-9a-f]{128}")
_DIGITS = re.compile(r"[0-9]+")
_HEX = re.compile(r"(?:[0-9a-f]{2})+")
_FRAME = b"veritor/protocol/merkle/frame/v3\0"

SALT_FORM = "the window's Program salt is not a SHA-512 digest (128 lowercase hex)"
CALLS_FORM = "the window's manifest is not a list of call rows"
SERVED_FORM = "the served trees' digest is neither null nor a SHA-512 digest (128 lowercase hex)"
DRAW_FORM = "the window's draw is not an object whose receipts are a SHA-512 digest"
NO_DRAW = ("the challenger's session closed with no draw coin opened: the window opens its draw coin once, after its "
           "draw root")
DRAW_ROOT = ("the window's draw root is not over its manifest (each call's credit bit among it), its last round's "
             "receipts and every round's root and coin")
CHANGED = ("the draw root over the manifest (each call's credit bit among it), its last round's receipts and every round's "
           "root and coin is not the one root the challenger's log shows its draw coin followed: a credit bit or a call "
           "changed after the draw coin, or the draw opened after another root. Commit every call's credit bit in its "
           "manifest row before the draw root, and never change it after the draw coin")
#: a served H1 window's draw budget K: `circuit.plan.RECORD_K`, the least K with (1 − ε)^K ≤ δ at ε = 0.1% and
#: δ = 2⁻⁴⁰ (`plan.record_k`), capped at the window's units
RECORD_K = 27_713
#: a window's layout: `circuit.h1window.program`'s arguments
LAYOUT_KEYS = ("calls", "linears", "m", "tile")
LAYOUT_FORM = ("the window's layout is not {calls, linears, m, tile}, h1window.program's arguments: its calls "
               "[[linear, A block], ...] in order, its linears [[name, K, n], ...] each once, its A slot m and its tile "
               "[R, C], every number a natural number and R, C at least one")
COIN_FORM = "the draw coin is not a SHA-512 digest (128 lowercase hex)"
DRAW_ARGS = ("the draw's stream is not a string, its registration not a SHA-512 digest, or its receipt not an object of "
             "printable ASCII strings")
#: a window's evidence, as `window` reads it
WINDOW_KEYS = ("calls", "draw", "layout", "proofs", "rounds", "salt", "served", "session", "window")
#: a window's evidence in a period: `WINDOW_KEYS` and the descriptor of the circuit its proofs were verified against
PERIOD_KEYS = ("calls", "descriptor", "draw", "layout", "proofs", "rounds", "salt", "served", "session", "window")
WINDOWS_FORM = ("the period's windows are not a list of each window's evidence {" + ", ".join(PERIOD_KEYS) + "}, "
                "in its close's order")
DESCRIPTOR_HEX = ("the window's descriptor is not the bytes of the circuit its proofs were verified against in "
                  "lowercase hex (its canonical JSON without annotations, the bytes program_digest hashes)")
PROOFS_FORM = "the drawn units' proof outcomes are not a list of {unit, verdict}, verdict accept or reject"
LOG_FORM = "the round log is not a nonempty list"
ENTRIES_FORM = "the window's registered entries are not a list of objects"
NO_A = ("the window registers no A value (…/a/<K>), so nothing ties its units' A rows to the round roots: register the "
        "window's A values")


def _digest(x: Any) -> bytes | None:
    return bytes.fromhex(x) if isinstance(x, str) and _DIGEST.fullmatch(x) else None


def _nat(x: Any) -> bool:
    return type(x) is int and x >= 0


def _printable(x: Any) -> bool:
    """A string of printable ASCII but ``"`` and ``\\``, as a receipt's keys and values are (`core_check.draw_key_context`)."""
    return type(x) is str and all(" " <= ch <= "~" and ch not in '"\\' for ch in x)


def _plain(x: Any, depth: int = DEPTH) -> bool:
    """JSON of objects, lists, strings, integers, booleans and null nested at most `depth` deep: what the Lean writes
    as `json.dumps` does."""
    if depth == 0:
        return False
    if x is None or type(x) in (bool, int, str):
        return True
    if type(x) is list:
        return all(_plain(v, depth - 1) for v in x)
    return type(x) is dict and all(type(k) is str and _plain(v, depth - 1) for k, v in x.items())


def calls_root(salt: bytes, rows: Any, served: bytes | None = None) -> bytes:
    """SHA-512 over the manifest rows `rows` in order, each as ``json.dumps(row, sort_keys=True)`` and a newline, under
    `CALLS_TAG` and the window's Program salt; with `served`, SHA-512 of that and the served trees' digest."""
    h = hashlib.sha512(CALLS_TAG + salt)
    for row in rows:
        h.update(json.dumps(row, sort_keys=True).encode() + b"\n")
    return h.digest() if served is None else hashlib.sha512(h.digest() + served).digest()


def draw_root(calls: bytes, receipts: bytes, rounds: Any) -> bytes:
    """What the draw coin opens after: the manifest's `calls_root`, the last round's receipts and every round's root and
    coin, so the draw follows every round coin by hash."""
    return tagged(DRAW_TAG, calls, receipts, *(p for x in rounds for p in (
        bytes.fromhex(x["root"]), bytes.fromhex((x.get("coin") or {}).get("coin", "")))))


def no_bit(i: int) -> str:
    return (f"manifest row {i} has no credit bit: the serving process commits each call's verdict as `{CREDIT}` (true or "
            "false) in its manifest row, under calls_root, before the draw coin")


def no_z(i: int) -> str:
    return (f"call {i} is credited with no z table root: a credited call's manifest row carries its z table's root "
            f"(`tables.z`), or its credit bit is false")


def credited(key: Any, ledger: Any, window: Any, session: Any, salt: Any, calls: Any, served: Any, rounds: Any, draw: Any,
             verify: Ch.Verify) -> tuple[list[int] | None, tuple[str, str] | None]:
    """The credited calls (their indices) of a window whose manifest `calls` (its rows, `served` the served trees'
    digest or None, under the window's Program salt `salt`) fixed every call's credit bit before the draw coin, and None;
    or None and the refusal (``coin``, ``manifest``, ``credit``, ``rounds`` or ``draw``, and why). The draw coin is read
    through the session the auditor's signed ledger `ledger` opens for `window` under the pinned challenger key `key`,
    from that session's files `session` (`challenger.trusted_session_of`, its signatures checked with `verify`): its log
    must show `draw_root` over the manifest, ``draw``'s ``receipts`` and the round log `rounds` (each round's ``root``
    and ``coin``) as the one root the draw coin followed, ``draw``'s ``root`` must be it, and ``draw``'s ``coin`` must be
    the challenger's draw coin 0 opened after it."""
    try:
        k = Ch.ChallengerKey.from_json(key)
    except Ch.CoinError as e:
        return None, ("coin", f"the verifier's challenger key is malformed: {e}")
    try:
        c, received = Ch.trusted_session_of(ledger, window, k, session, verify)
    except Ch.CoinError as e:
        return None, ("coin", f"the challenger's session is not trusted: {e}")
    s = _digest(salt)
    if s is None:
        return None, ("manifest", SALT_FORM)
    if not isinstance(calls, list):
        return None, ("manifest", CALLS_FORM)
    out = []
    for i, row in enumerate(calls):
        if not (isinstance(row, dict) and _plain(row)):
            return None, ("manifest", f"manifest row {i} is not an object of objects, lists, strings, integers, booleans "
                                      "and null")
        if not (type(row.get("index")) is int and row["index"] == i):
            return None, ("manifest", f"manifest row {i} is not call {i}: the rows are the window's calls in order, index "
                                      "0, 1, …")
        bit = row.get(CREDIT)
        if type(bit) is not bool:
            return None, ("credit", no_bit(i))
        if bit:
            tables = row.get("tables")
            if not (isinstance(tables, dict) and _digest(tables.get("z")) is not None):
                return None, ("credit", no_z(i))
            out.append(i)
    if served is not None and _digest(served) is None:
        return None, ("manifest", SERVED_FORM)
    if not isinstance(rounds, list):
        return None, ("rounds", "the round log is not a list")
    parts = []
    for r, x in enumerate(rounds):
        root = _digest(x.get("root")) if isinstance(x, dict) else None
        if root is None:
            return None, ("rounds", f"round {r}'s root is not a SHA-512 digest")
        try:
            parts += [root, Ch.LiveOpening.from_json(x.get("coin")).coin]
        except Ch.CoinError as e:
            return None, ("rounds", f"round {r}'s coin: {e}")
    receipts = _digest(draw.get("receipts")) if isinstance(draw, dict) else None
    if receipts is None:
        return None, ("draw", DRAW_FORM)
    root = tagged(DRAW_TAG, calls_root(s, calls, _digest(served)), receipts, *parts)
    if draw.get("root") != root.hex():
        return None, ("draw", DRAW_ROOT)
    got = tuple(received.get(Wd.DRAW, ()))
    if not got:
        return None, ("draw", NO_DRAW)
    if got != (root,):
        return None, ("draw", CHANGED)
    try:
        why = Ch.live_reason(c, Ch.LiveOpening.from_json(draw.get("coin")), Wd.DRAW, 0, root)
    except Ch.CoinError as e:
        why = str(e)
    if why is not None:
        return None, ("coin", f"the draw coin: {why}")
    return out, None


def strata(layout: Any) -> tuple[list[dict] | None, tuple[str, str] | None]:
    """The work strata of the served H1 window whose Program is ``h1window.program(linears, calls, m, tile)`` (`layout`,
    ``{calls, linears, m, tile}``), and None; or None and the refusal (``layout``, and why). Its units are the Program's,
    in its order: call by call, each call's (m/R)·(n/C) tiles row block major. K's stratum is every unit of the calls of
    K's linears, template ``k<K>``, as ``[lo, hi)`` ranges in order (adjacent ones merged), the strata in the order of
    their first unit; a unit's work is its tile's served MACs, R·C·K, and every floor is one. Refused as h1window refuses
    the layout. Its cost is the layout's calls, never its units."""
    def no(why: str) -> tuple[None, tuple[str, str]]:
        return None, ("layout", why)

    if not (isinstance(layout, dict) and sorted(layout) == list(LAYOUT_KEYS)):
        return no(LAYOUT_FORM)
    tile, m, linears, calls = layout["tile"], layout["m"], layout["linears"], layout["calls"]
    if not (type(tile) is list and len(tile) == 2 and all(_nat(x) and x >= 1 for x in tile) and _nat(m)
            and type(linears) is list and linears and type(calls) is list and calls):
        return no(LAYOUT_FORM)
    r, c = tile
    ks: dict[str, tuple[int, int]] = {}
    for x in linears:
        if not (type(x) is list and len(x) == 3 and type(x[0]) is str and _nat(x[1]) and _nat(x[2])):
            return no(LAYOUT_FORM)
        if x[0] in ks:
            return no(f"linear {x[0]} is named twice: a window names each of its weights once")
        ks[x[0]] = (x[1], x[2])
    for x in calls:
        if not (type(x) is list and len(x) == 2 and type(x[0]) is str and _nat(x[1])):
            return no(LAYOUT_FORM)
        if x[0] not in ks:
            return no(f"a call of {x[0]}, which is no linear of the window: every call is of one of its linears")
    for name in ks:
        if not any(lin == name for lin, _blk in calls):
            return no(f"linear {name} is never called: every linear of the window is called")
    if m < r or m & (m - 1):
        return no(f"the A slot {m} is no power of two of at least the tile's {r} rows (h1window.slot)")
    for name, (k, n) in ks.items():
        if k < 1 or n < c or n % c:
            return no(f"linear {name}: K = {k} and n = {n} are not K >= 1 and n whole blocks of the tile's {c} columns")
    for k in sorted({k for k, _n in ks.values()}):
        used = sorted({blk for lin, blk in calls if ks[lin][0] == k})
        if used != list(range(len(used))):
            return no(f"the A blocks of K = {k} are {used}: each of 0, 1, ... is read by some call of K")
    out: list[dict] = []
    at = 0
    for lin, _blk in calls:
        k, n = ks[lin]
        size, t = (m // r) * (n // c), f"k{k}"
        s = next((s for s in out if s["template"] == t), None)
        if s is None:
            s = {"template": t, "units": [], "work": r * c * k, "floor": 1}
            out.append(s)
        if s["units"] and s["units"][-1][1] == at:
            s["units"][-1][1] = at + size
        else:
            s["units"].append([at, at + size])
        at += size
    return out, None


def law(ss: list[dict]) -> int:
    """`ss` (`strata`'s) under the work law ``work:k`` at K = min(`RECORD_K`, N), N its units: each stratum's k_s
    (`core_check.work_k`, over the population's work W = Σ w_s·n_s), set in place. Returns K."""
    sizes = [sum(hi - lo for lo, hi in s["units"]) for s in ss]
    k, total = min(RECORD_K, sum(sizes)), sum(s["work"] * n for s, n in zip(ss, sizes))
    for s, n in zip(ss, sizes):
        s["k"] = CC.work_k(k, s["work"], n, total, s["floor"])
    return k


def _drawn(coin: bytes, stream: Any, registration: Any, receipt: Any, ss: list[dict]
           ) -> tuple[list[int] | None, tuple[str, str] | None]:
    if not (type(stream) is str and _digest(registration) is not None and isinstance(receipt, dict)
            and all(_printable(x) and _printable(v) for x, v in receipt.items())):
        return None, ("draw", DRAW_ARGS)
    law(ss)
    got = CC.work_units(coin, stream, registration, receipt, ss)
    return (None, ("sampler", "the draw's stream ran out")) if got is None else (got, None)


def _proved(drawn: list[int], proofs: Any) -> tuple[str, str] | None:
    if not isinstance(proofs, list):
        return "outcome", PROOFS_FORM
    seen: dict[int, str] = {}
    picked = set(drawn)
    for p in proofs:
        if not (isinstance(p, dict) and sorted(p) == ["unit", "verdict"] and _nat(p["unit"])
                and p["verdict"] in VERDICTS):
            return "outcome", PROOFS_FORM
        u = p["unit"]
        if u in seen:
            return "outcome", (f"unit {u} has two proof outcomes: a drawn unit is proved and verified once, and a retry "
                               "never reproves it")
        if u not in picked:
            return "outcome", f"unit {u} has a proof outcome and was not drawn: only the draw's units are proved"
        seen[u] = p["verdict"]
    for u in drawn:
        if u not in seen:
            return "missing", (f"drawn unit {u} has no proof outcome: every drawn unit's proof is verified, and a window "
                               "with one missing is refused whole (no partial credit). Prove and verify every drawn unit")
        if seen[u] != "accept":
            return "failed", (f"drawn unit {u}'s proof was rejected: a failed drawn unit refuses the whole window, not "
                              "only its call (no partial credit, and no retry)")
    return None


def drawn(coin: Any, stream: Any, registration: Any, receipt: Any, layout: Any
          ) -> tuple[list[int] | None, tuple[str, str] | None]:
    """The units the draw coin `coin` (its value, a SHA-512 digest in hex) draws from the window `layout` (`strata`)
    under `law`, ascending (`core_check.work_units` under the stream's digest `stream`, the registration `registration`
    and its `receipt`), and None; or None and the refusal (``layout``, ``draw`` or ``sampler``, and why)."""
    ss, why = strata(layout)
    if why is not None:
        return None, why
    c = _digest(coin)
    if c is None:
        return None, ("draw", COIN_FORM)
    return _drawn(c, stream, registration, receipt, ss)


def outcome(coin: Any, stream: Any, registration: Any, receipt: Any, layout: Any, proofs: Any) -> tuple[str, str] | None:
    """None when every unit the draw coin draws from the window (`drawn`, never a list the prover hands in) has exactly
    one outcome in `proofs` (each ``{unit, verdict}``, verdict ``accept`` or ``reject``) and it is ``accept``, and no
    other unit has one; else the refusal (``layout``, ``draw``, ``sampler``, ``outcome``, ``missing`` or ``failed``, and
    why): the whole window is refused, with no partial credit."""
    got, why = drawn(coin, stream, registration, receipt, layout)
    return why if got is None else _proved(got, proofs)


def window(key: Any, ledger: Any, window: Any, session: Any, salt: Any, calls: Any, served: Any, rounds: Any, draw: Any,
           layout: Any, proofs: Any, verify: Ch.Verify) -> tuple[list[int] | None, tuple[str, str] | None]:
    """A served H1 window's credited calls and None, or None and the refusal: `credited` (the credit bits fixed before
    the draw coin, read through the challenger's session), the manifest one row per call of `layout` (``manifest``),
    and `outcome` of the draw coin `credited` read, under the served window's arguments: the stream is the window's
    root (``window.root``, the root the auditor's ledger opens its session for), the registration its draw root
    (``draw.root``, the one root the draw coin followed) and the receipt ``{"receipts": draw.receipts}``, the last
    round's receipts."""
    got, why = credited(key, ledger, window, session, salt, calls, served, rounds, draw, verify)
    if why is not None:
        return None, why
    ss, why = strata(layout)
    if why is not None:
        return None, why
    if len(calls) != len(layout["calls"]):
        return None, ("manifest", f"the manifest's {len(calls)} rows are not the layout's {len(layout['calls'])} calls: "
                                  "each call of the window's Program is one manifest row, in order")
    units, why = _drawn(Ch.LiveOpening.from_json(draw["coin"]).coin, window["root"], draw["root"],
                        {"receipts": draw["receipts"]}, ss)
    if why is None:
        why = _proved(units, proofs)
    return (None, why) if why is not None else (got, None)


def period(key: Any, ledger: Any, name: Any, roots: Any, verify: Ch.Verify) -> tuple[str, str] | None:
    """None when `roots` are exactly the windows the auditor's ledger `ledger` lists in period `name`'s close, in its
    order, and that close is complete: every window between the period's open and its close, each opened and ended
    complete (`challenger.verify_periods`, under the pinned challenger key `key`). Else the refusal (``coin``,
    ``period``, ``missing`` or ``order``, and why): a period with no close, closed incomplete, or with a window missing,
    out of order or not its own earns nothing, so no window is chosen after the fact."""
    try:
        k = Ch.ChallengerKey.from_json(key)
    except Ch.CoinError as e:
        return "coin", f"the verifier's challenger key is malformed: {e}"
    try:
        Ch.ledger_head([ledger], k, verify)
    except Ch.CoinError as e:
        return "coin", f"the auditor's ledger is not trusted: {e}"
    if type(name) is not str:
        return "period", "the period is not named"
    entries = [json.loads(line) for line in ledger]
    if not any(e["kind"] == "period-open" and e["body"]["period"] == name for e in entries):
        return "period", (f"the auditor's ledger opens no period {name}: credit counts only a period the ledger opens "
                          "and closes")
    close = next((e["body"] for e in entries if e["kind"] == "period-close" and e["body"]["period"] == name), None)
    if close is None:
        return "period", (f"period {name} has no close in the auditor's ledger, so it earns nothing: close it "
                          "(period-close) and present the ledger after its close")
    ws = close["windows"]
    if not close["complete"]:
        i = next((i for i, w in enumerate(ws) if not w["complete"]), None)
        return "period", (f"period {name} closed with no window, so it earns nothing" if i is None else
                          f"period {name} closed incomplete: its window {i} (session {ws[i]['session']}) never opened "
                          "or didn't end complete before the close, so the period earns nothing")
    if not (type(roots) is list and all(_digest(r) is not None for r in roots)):
        return "period", "the period's windows are not a list of window roots (SHA-512 digests)"
    want = [w["root"] for w in ws]
    wanted, presented = set(want), set(roots)
    for i in range(max(len(want), len(roots))):
        if i >= len(want):
            return "period", (f"the windows presented run past the {len(want)} that period {name}'s close lists: only "
                              "its close's windows count, each once")
        if i >= len(roots) or (roots[i] != want[i] and want[i] not in presented):
            return "missing", (f"window {i} of period {name} ({want[i][:16]}...) is missing: every window between the "
                               "period's open and its close counts, so present them all, in the close's order")
        if roots[i] != want[i]:
            if roots[i] not in wanted:
                return "period", (f"window {roots[i][:16]}... is not one of period {name}'s windows: only the windows "
                                  "its close lists count")
            return "order", (f"window {i} of period {name} is {roots[i][:16]}..., not {want[i][:16]}..., the close's "
                             f"window {i}: present the period's windows in the order its close lists them")
    return None


def period_credit(key: Any, ledger: Any, name: Any, windows: Any, verify: Ch.Verify
                  ) -> tuple[list[list[int]] | None, tuple[str, str] | None]:
    """Each window's credited calls, in the close's order, of period `name`, and None; or None and the refusal: the
    period's windows are exactly its close's (`period`), each window proves the circuit the period's open committed
    (``circuit``: its ``descriptor``, the bytes in hex of the circuit its proofs were verified against, hashes to it,
    `challenger.credited_session`), and each window's evidence (``{window, session, salt, calls, served, rounds, draw,
    layout, proofs}``, as `window` reads it, under the same ledger) is credited by `window`. A refused window refuses
    the whole period (``circuit`` or ``window``), so a period earns only when every window in it does."""
    if not (type(windows) is list and all(isinstance(w, dict) and sorted(w) == list(PERIOD_KEYS) for w in windows)):
        return None, ("period", WINDOWS_FORM)
    roots = [w["window"].get("root") if isinstance(w["window"], dict) else None for w in windows]
    why = period(key, ledger, name, roots, verify)
    if why is not None:
        return None, why
    k = Ch.ChallengerKey.from_json(key)
    out = []
    for i, w in enumerate(windows):
        if not (type(w["descriptor"]) is str and _HEX.fullmatch(w["descriptor"])):
            return None, ("circuit", f"window {i} of period {name}: {DESCRIPTOR_HEX}")
        try:
            Ch.credited_session(ledger, w["window"], k, bytes.fromhex(w["descriptor"]), verify)
        except Ch.CoinError as e:
            return None, ("circuit", f"window {i} of period {name} doesn't count toward it, so the period earns "
                                     f"nothing: {e}")
        got, why = window(key, ledger, w["window"], w["session"], w["salt"], w["calls"], w["served"], w["rounds"],
                          w["draw"], w["layout"], w["proofs"], verify)
        if why is not None:
            return None, ("window", f"window {i} of period {name} is refused, so the period earns nothing: {why[0]}: "
                                    f"{why[1]}")
        out.append(got)
    return out, None


def _uint(n: int) -> bytes:
    return n.to_bytes(max(1, (n.bit_length() + 7) // 8), "big")


def _frame(tag: bytes, *parts: bytes) -> bytes:
    """frame-v3's SHA-512 hash (`verity.ml.commitments.merkle`'s ``_hash`` under ``sha512``)."""
    return hashlib.sha512(_FRAME + len(tag).to_bytes(4, "big") + tag
                          + b"".join(len(p).to_bytes(8, "big") + p for p in parts)).digest()


def merkle_depth(count: int) -> int:
    return 0 if count <= 1 else (count - 1).bit_length()


def tree_root(domain: bytes, count: int, have: dict[tuple[int, int], bytes]) -> bytes | None:
    """The root of the frame-v3 SHA-512 tree of domain id `domain` over `count` leaves, from the subtree roots `have`
    (``(level, index)`` to its root, level 0 a leaf) and the domain's pads past `count`; None where a node below
    `count` is in none of them."""
    def at(level: int, index: int) -> bytes | None:
        if (level, index) in have:
            return have[(level, index)]
        if level == 0:
            return _frame(b"pad", domain, _uint(index)) if index >= count else None
        left, right = at(level - 1, 2 * index), at(level - 1, 2 * index + 1)
        return None if left is None or right is None else _frame(b"node", domain, _uint(level - 1), _uint(index), left,
                                                                 right)
    return at(merkle_depth(count), 0)


def cover(lo: int, hi: int) -> list[tuple[int, int]]:
    """Rows ``[lo, hi)`` as the fewest aligned subtrees, ``(level, index)`` in row order: what the serving process lists
    for a round's rows of an A value."""
    out = []
    while lo < hi:
        level = 0
        while level + 1 < LEVELS and lo % (1 << (level + 1)) == 0 and lo + (1 << (level + 1)) <= hi:
            level += 1
        out.append((level, lo >> level))
        lo += 1 << level
    return out


def round_root(r: int, inner: bytes, segments: dict[str, list[tuple[int, int, bytes]]]) -> bytes:
    """The root the challenger receives for round `r`: the serving process's own round root `inner` and, for each A
    value by name, its segments (level, index and root) in row order."""
    return tagged(ROUND_TAG, r, inner, *(p for v in sorted(segments) for level, index, root in segments[v]
                                         for p in (v, level, index, root)))


def _segments(a: Any) -> dict[str, list[tuple[int, int, bytes]]] | None:
    if not isinstance(a, dict):
        return None
    out = {}
    for v, segs in a.items():
        if not isinstance(segs, list):
            return None
        got = []
        for s in segs:
            if not (isinstance(s, list) and len(s) == 3 and _nat(s[0]) and s[0] < LEVELS and _nat(s[1])
                    and _digest(s[2]) is not None):
                return None
            got.append((s[0], s[1], _digest(s[2])))
        out[v] = got
    return out


def is_a(value: Any) -> bool:
    """A registered A value: a name whose last two parts are ``a`` and a contraction (``…/a/<K>``)."""
    parts = value.split("/") if isinstance(value, str) else []
    return len(parts) >= 2 and parts[-2] == "a" and bool(_DIGITS.fullmatch(parts[-1]))


def a_link(rounds: Any, entries: Any) -> tuple[str, str] | None:
    """None when every registered A value among `entries` (the window's registered entries, ``{value, root, leaves,
    domain, …}``) is the tree over the segments the round log `rounds` commits (each round's ``root``, the one the
    challenger received, over its ``inner`` root and its segments ``a``, `round_root`), round s's segments of a value
    of L rows tiling its rows ``[s·L/S, (s+1)·L/S)`` of S rounds; else the refusal (``a-link``, and why)."""
    if not (isinstance(rounds, list) and rounds):
        return "a-link", LOG_FORM
    segs = []
    for r, x in enumerate(rounds):
        root, inner = ((_digest(x.get(k)) if isinstance(x, dict) else None) for k in ("root", "inner"))
        if root is None or inner is None:
            return "a-link", f"round {r}'s root or inner root is not a SHA-512 digest"
        p = _segments(x.get("a"))
        if p is None:
            return "a-link", (f"round {r}'s A segments are not an object from each A value to its [level, index, root] "
                              f"subtrees, each level below {LEVELS}")
        if round_root(r, inner, p) != root:
            return "a-link", (f"round {r}'s root is not over its inner root and its A segments: the serving process "
                              "commits each round's segments of every registered A value under that round's root "
                              "(`credit.round_root`), before that round's coin")
        segs.append(p)
    if not (isinstance(entries, list) and all(isinstance(e, dict) for e in entries)):
        return "a-link", ENTRIES_FORM
    a = [e for e in entries if is_a(e.get("value"))]
    if not a:
        return "a-link", NO_A
    names: list[str] = []
    for e in a:
        if e["value"] in names:
            return "a-link", f"the window registers A value {e['value']} twice"
        names.append(e["value"])
    for r, p in enumerate(segs):
        for v in sorted(p):
            if v not in names:
                return "a-link", f"round {r} lists segments of {v}, which is no registered A value of the window"
    steps = len(rounds)
    for e in a:
        v, root, domain, count = e["value"], _digest(e.get("root")), _digest(e.get("domain")), e.get("leaves")
        if root is None or domain is None or not _nat(count) or not 1 <= count <= MAX_ROWS:
            return "a-link", (f"registered A value {v}'s entry has no root, domain or leaf count from 1 to {MAX_ROWS} (the "
                              "verifier hashes its pads)")
        if count % steps:
            return "a-link", f"registered A value {v}'s {count} rows are not the window's {steps} rounds' equal blocks"
        per, have = count // steps, {}
        for r, p in enumerate(segs):
            at, end = r * per, (r + 1) * per
            untiled = (f"round {r}'s segments of {v} don't tile its rows [{at}, {end}) in order, each an aligned subtree: "
                       "the serving process lists each round's rows of every A value as aligned subtrees (`credit.cover`)")
            for level, index, h in p.get(v, []):
                if index << level != at or at + (1 << level) > end:
                    return "a-link", untiled
                have[(level, index)] = h
                at += 1 << level
            if at != end:
                return "a-link", untiled
        if tree_root(domain, count, have) != root:
            return "a-link", (f"registered A value {v}'s root is not the tree over the segments the round roots commit: "
                              "its rows are not the ones fixed under each round's root before that round's coin. Register "
                              "the A value whose rows each round's root committed")
    return None


__all__ = ["CALLS_TAG", "CREDIT", "DEPTH", "DRAW_TAG", "LAYOUT_KEYS", "LEVELS", "PERIOD_KEYS", "RECORD_K", "ROUND_TAG",
           "VERDICTS", "WINDOW_KEYS", "a_link", "calls_root", "cover", "credited", "draw_root", "drawn", "is_a", "law",
           "merkle_depth", "outcome", "period", "period_credit", "round_root", "strata", "tree_root", "window"]
