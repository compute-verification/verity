"""PoUW's filler windows and the ``fill`` a period credits beside its ``work``: the Python the Lean
`PouwCheck.Fill` is held to through ``pouw-period`` (`period_check`, the difftest ``lean/check/tests/test_period_difftest.py``).

A filler window is Dither calls (`circuit.dither`) the prover runs on tensor pipes its served windows leave idle, each a
call whose A is its per-call pad (`dither.fill_seeds`, `dither.pad_rows`) and whose B is a published checkpoint's s8
weights, committing its T_1 .. T_S (`dither.Shape` with ``o=False``). Its proof blocks are `PouwDitherBlock_v1`'s,
one cell and one segment each. It counts toward the companion period ``<name>:fill`` (`period_name`), which the
auditor's ledger opens and closes as it does a served period: its open commits the circuit (the SHA-512 of the filler
descriptor, `descriptor_of`) and the draw K (`credit.K_FIELD`). `work` and its rule (`credit.period_credit`) are unchanged.

* **The circuit is public structure** (`descriptor_of`): the descriptor is canonical JSON ``{L, calls, checkpoint, k, lp,
  m, n, row, unit}``, the Dither filler's shape (t = min(L, K)/L′ at most `T_MAX`), its call count, the SHA-512
  `dither.checkpoint_digest` of B and the unit ``PouwDitherBlock_v1``. The window's open root is `open_root` of the
  circuit and the window's session, so it is fixed before every coin and no two windows share one.
* **The pad and noise are live coins** (`_coins`): the window's challenger session (`families`: one salt coin, one call
  coin a call, one draw coin) is read through the auditor's ledger (`challenger.trusted_session_of`); its log shows the
  open root as the one root its salt coin and every call coin followed. The window's salt is the salt coin's round-0
  salt after it (`h1tile.live_salt`), and call c's seeds are `dither.fill_seeds` of that salt and call coin c, so a
  call's A and noise are drawn after its circuit and B are fixed, and the prover can table nothing ahead of them.
* **Every value is fixed before the draw** (`receipts`, `draw_root`): each call's value root (``roots``) is under the
  draw root, over the window's salt and `receipts` of the roots, and the challenger's log shows it as the one root the
  draw coin followed. The draw is `credit._window_drawn`'s over one stratum (`stratum`) of every call's blocks at the fill
  period's K, block (c, i, j, s) unit c·m·n·S + (i·n + j)·S + s − 1 (`block_of`), each block L MACs.
* **No partial credit**: every drawn block has exactly one proof outcome and it is an accept (`credit._proved`), each
  verified against its block's statement (`statement`: the circuit, the salt, its call's coin and the block) with its
  ``t`` port opened at its call's value root (`registered`), and a refused window refuses the whole fill period. The
  credit is ``{work, k, delta, eps}``: each window's work (`work`, m·n·K a call), the fill period's K, `credit.DELTA`
  and the ε its draw established over all of it (`eps`).

A filler window's evidence is `FILL_KEYS`, each drawn block's proof outcome the verifier of record's verdict."""
from __future__ import annotations

import hashlib
from fractions import Fraction
from typing import Any

from verity.core.protocols.pouw import credit as Cr
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.circuit import dither as D
from verity.core.protocols.pouw.circuit import h1tile as HT
from verity.core.protocols.pouw.protocol import tagged
from verity.core.protocols.sampled_proofs import challenger as Ch

#: the companion period a served period's filler windows count toward: ``<name>:fill``
SUFFIX = ":fill"
#: a filler window's call coins, one a call, between its salt coin and its draw coin
CALL = "pouw/fill/call"
UNIT = "PouwDitherBlock_v1"
#: the most L′-tiles a block's words span, t = min(L, K) / L′ (Dither's NOFOLD-OPT pick, `dither.Shape`)
T_MAX = 8
DESCRIPTOR_KEYS = ("L", "calls", "checkpoint", "k", "lp", "m", "n", "row", "unit")
OPEN_TAG = "verity/pouw/fill/open/v2"
ROOTS_TAG = "verity/pouw/fill/roots/v1"
DRAW_TAG = "verity/pouw/fill/draw/v2"
STATEMENT_TAG = "verity/pouw/fill/statement/v1"
#: the one stratum of a filler window's draw
TEMPLATE = "dither"
FILL_KEYS = ("calls", "descriptor", "draw", "opening", "proofs", "roots", "salt", "session", "window")
FILLS_FORM = ("the period's filler windows are not a list of each filler window's evidence {" + ", ".join(FILL_KEYS)
              + "}, in the order its fill period's close lists them")
DESCRIPTOR_HEX = "the filler window's descriptor is not the bytes in lowercase hex of the circuit its fill period committed"
DESCRIPTOR_FORM = ("the filler window's descriptor is not canonical JSON {" + ", ".join(DESCRIPTOR_KEYS) + "}: the "
                   f"Dither filler's shape and call count as positive integers, B's checkpoint digest (128 lowercase "
                   f"hex) and the unit {UNIT}")
SHAPE = (f"the filler window's shape is not a Dither filler's: m and n whole tiles of {D.TILE}, lp a multiple of 32 "
         f"dividing k, L a multiple of lp dividing k with t = min(L, k) / lp at most {T_MAX}, k below {D.K_LIMIT}, "
         f"and row a multiple of 16 dividing {D.TILE_WORD_BYTES}")
OPEN = ("the filler window's open root is not open_root of its circuit and its session: open each filler window on "
        "its own root, after the circuit is fixed")
FILL_EPOCH = ("the coin commitment's coins are not a filler window's: its salt coin, a call coin for each of its "
              "calls, then its draw coin")
SALT_NOT_RECEIVED = "the challenger's log does not show the filler window's open root as the one root its salt coin followed"
FILL_SALT = ("the filler window's salt is not its live salt coin's round-0 salt after its open root (h1tile.live_salt): "
             "seed its calls from the coin the challenger opened")
CALLS_FORM = "the filler window's calls are not a list of each call's coin opening, one a call of its descriptor"
CALLS_NOT_RECEIVED = ("the challenger's log does not show the filler window's open root as the one root each of its call "
                      "coins followed: open every call coin after the open root, before the call's values")
ROOTS_FORM = "the filler window's roots are not a list of each call's value root (SHA-512 digests), one a call"
RECEIPTS = ("the filler window's draw receipts are not the digest of its calls' value roots (receipts): commit every "
            "call's values before the draw coin and draw over them")
DRAW_ROOT = "the filler window's draw root is not over its salt and its calls' value roots (draw_root)"
CHANGED = ("the draw root over the filler window's salt and its calls' value roots is not the one root the challenger's "
           "log shows its draw coin followed: a value changed after the draw coin")


def period_name(name: str) -> str:
    """The fill period of served period `name`."""
    return name + SUFFIX


def families(calls: int) -> tuple[tuple[str, int], ...]:
    """A filler window's coin layout: its salt coin, `calls` call coins, then its draw coin."""
    return (Wd.SALT, 1), (CALL, calls), (Wd.DRAW, 1)


def descriptor_of(raw: bytes) -> tuple[tuple[D.Shape, int] | None, tuple[str, str] | None]:
    """The filler shape and call count the descriptor `raw` names, and None; or None and the refusal (``descriptor``)."""
    try:
        ok, j = Cr._loads(raw.decode())
    except UnicodeDecodeError:
        ok, j = False, None
    if not (ok and isinstance(j, dict) and sorted(j) == list(DESCRIPTOR_KEYS) and j["unit"] == UNIT
            and Cr._digest(j["checkpoint"]) is not None
            and all(Cr._nat(j[x]) and j[x] >= 1 for x in ("L", "calls", "k", "lp", "m", "n", "row"))):
        return None, ("descriptor", DESCRIPTOR_FORM)
    try:
        sh = D.Shape(j["m"], j["n"], j["k"], j["lp"], j["L"], j["row"], o=False)
    except ValueError:
        return None, ("descriptor", SHAPE)
    if j["k"] % j["L"] or min(sh.L, sh.k) // sh.lp > T_MAX:
        return None, ("descriptor", SHAPE)
    return (sh, j["calls"]), None


def descriptor(sh: D.Shape, calls: int, checkpoint: bytes) -> bytes:
    """The filler descriptor of `calls` calls of shape `sh` against the checkpoint whose digest is `checkpoint`."""
    return Cr._canonical({"L": sh.L, "calls": calls, "checkpoint": checkpoint.hex(), "k": sh.k, "lp": sh.lp, "m": sh.m,
                          "n": sh.n, "row": sh.row, "unit": UNIT}).encode()


def open_root(circuit: bytes, session: str) -> bytes:
    """A filler window's open root: `tagged` of its circuit (its descriptor's SHA-512) and its session."""
    return tagged(OPEN_TAG, circuit, session)


def receipts(roots: list[bytes]) -> bytes:
    """The digest of a filler window's calls' value roots, in call order."""
    return tagged(ROOTS_TAG, *roots)


def draw_root(salt: bytes, receipts_: bytes) -> bytes:
    """A filler window's draw root, over its salt and `receipts` of its calls' value roots."""
    return tagged(DRAW_TAG, salt, receipts_)


def blocks(sh: D.Shape) -> int:
    """A call's proof blocks: m·n cells of S segments each."""
    return sh.m * sh.n * sh.S


def block_of(sh: D.Shape, u: int) -> tuple[int, int, int, int]:
    """Unit `u`'s block (c, i, j, s): call c, cell (i, j), segment s."""
    c, r = divmod(u, blocks(sh))
    cell, s = divmod(r, sh.S)
    i, j = divmod(cell, sh.n)
    return c, i, j, s + 1


def stratum(sh: D.Shape, calls: int) -> list[dict]:
    """The draw's one stratum: every call's blocks, L MACs each."""
    return [{"template": TEMPLATE, "units": [[0, calls * blocks(sh)]], "work": sh.L, "floor": 1}]


def statement(circuit: bytes, salt: bytes, coin: bytes, block: tuple[int, int, int, int]) -> bytes:
    """The statement a drawn block's proof is verified against: its circuit, the window's salt, its call's coin and the
    block (c, i, j, s), from which the verifier of record derives its public inputs (its pad row, its noise and its
    Definition)."""
    return tagged(STATEMENT_TAG, circuit, salt, coin, *block)


def registered(c: int, root: bytes) -> dict:
    """A block proof's registered finding: its ``t`` port's words opened at call c's value root."""
    return {"t": {"root": root.hex(), "value": f"fill/T@{c}"}}


def work(sh: D.Shape, calls: int) -> int:
    """A filler window's work: m·n·K MACs a call."""
    return calls * sh.m * sh.n * sh.k


def eps(sh: D.Shape, calls: int, k: int) -> Fraction | None:
    """The ε a draw of K = `k` establishes at `credit.DELTA` over the whole of a filler window's work (`window.eps_at`),
    or None when it establishes none (not even ε = 1: `window.k_prime` at 1 above K; K and the work are positive)."""
    w = work(sh, calls)
    if Wd.k_prime(Fraction(1), Cr.DELTA, w, w, 1) > k:
        return None
    return Wd.eps_at(k, Cr.DELTA, w, w, 1)


def _live(c: Ch.LiveCommitment, doc: Any, label: str, index: int, after: bytes, what: str) -> tuple[str, str] | None:
    try:
        why = Ch.live_reason(c, Ch.LiveOpening.from_json(doc), label, index, after)
    except Ch.CoinError as e:
        why = str(e)
    return None if why is None else ("coin", f"{what}: {why}")


def _coins(key: Any, ledger: Any, pins: Any, main: Any, window: dict, w: dict, calls: int, verify: Ch.Verify
           ) -> tuple[list[bytes] | None, tuple[str, str] | None]:
    """Each call's coin, and None, when the filler window's salt, call and draw coins are live, read through its
    challenger's session; else None and the refusal (``coin``, ``salt``, ``calls`` or ``draw``)."""
    k = Ch.ChallengerKey.from_json(key)
    r = bytes.fromhex(window["root"])
    try:
        c, received = Ch.trusted_session_of(ledger, window, k, pins, main, w["session"], verify)
    except Ch.CoinError as e:
        return None, ("coin", f"the challenger's session is not trusted: {e}")
    if c.scheme != Wd.LIVE_SCHEME:
        return None, ("coin", f"the coin commitment's scheme is {c.scheme}, not {Wd.LIVE_SCHEME}")
    if tuple(c.families) != families(calls):
        return None, ("coin", FILL_EPOCH)
    if tuple(received.get(Wd.SALT, ())) != (r,):
        return None, ("salt", SALT_NOT_RECEIVED)
    why = _live(c, w["opening"], Wd.SALT, 0, r, "the salt's coin")
    if why is not None:
        return None, why
    if w["salt"] != HT.live_salt(c, bytes.fromhex(w["opening"]["coin"]), 0, r).hex():
        return None, ("salt", FILL_SALT)
    if not (type(w["calls"]) is list and len(w["calls"]) == calls):
        return None, ("calls", CALLS_FORM)
    if tuple(received.get(CALL, ())) != (r,) * calls:
        return None, ("calls", CALLS_NOT_RECEIVED)
    coins = []
    for i, o in enumerate(w["calls"]):
        why = _live(c, o, CALL, i, r, f"call {i}'s coin")
        if why is not None:
            return None, why
        coins.append(bytes.fromhex(o["coin"]))
    roots = w["roots"]
    if not (type(roots) is list and len(roots) == calls and all(Cr._digest(x) is not None for x in roots)):
        return None, ("roots", ROOTS_FORM)
    rc = receipts([bytes.fromhex(x) for x in roots])
    draw = w["draw"]
    if not (isinstance(draw, dict) and draw.get("receipts") == rc.hex()):
        return None, ("draw", RECEIPTS)
    root = draw_root(bytes.fromhex(w["salt"]), rc)
    if draw.get("root") != root.hex():
        return None, ("draw", DRAW_ROOT)
    got = tuple(received.get(Wd.DRAW, ()))
    if not got:
        return None, ("draw", Cr.NO_DRAW)
    if got != (root,):
        return None, ("draw", CHANGED)
    why = _live(c, draw.get("coin"), Wd.DRAW, 0, root, "the draw coin")
    return (None, why) if why is not None else (coins, None)


def other_statement(u: int) -> str:
    return (f"drawn block {u}'s proof was verified against another statement than its own (statement: the circuit, the "
            "window's salt, its call's coin and the block): verify every drawn block against its own statement")


def other_registered(u: int, c: int) -> str:
    return (f"drawn block {u}'s proof opened something other than its t port at call {c}'s value root fill/T@{c}: "
            "verify every drawn block with its call's values registered")


def _window(key: Any, ledger: Any, pins: Any, main: Any, w: dict, session: str, circuit: bytes, k: int,
            verify: Ch.Verify) -> tuple[tuple[D.Shape, int] | None, tuple[str, str] | None]:
    """A filler window's shape and call count and None, or None and the refusal of the first step that refuses: its
    root's form, its descriptor (`descriptor_of`), its open root (`open_root`, ``open``), `_coins`, the draw at the
    fill period's committed K `k` (`credit._window_drawn` over `stratum`), each drawn block's one accepted outcome
    (`credit._proved`), its statement (``statement``) and its registered finding (``registered``)."""
    window = w["window"]
    if not (isinstance(window, dict) and sorted(window) == ["root"] and Cr._digest(window["root"]) is not None):
        return None, ("fill", Cr.WINDOW_ROOT_FORM)
    got, why = descriptor_of(bytes.fromhex(w["descriptor"]))
    if why is not None:
        return None, why
    sh, calls = got
    if window["root"] != open_root(circuit, session).hex():
        return None, ("open", OPEN)
    coins, why = _coins(key, ledger, pins, main, window, w, calls, verify)
    if why is not None:
        return None, why
    units, why = Cr._window_drawn(window, w["draw"], stratum(sh, calls), k)
    if why is None:
        why = Cr._proved(units, w["proofs"])
    if why is not None:
        return None, why
    salt = bytes.fromhex(w["salt"])
    for p in w["proofs"]:
        u = p["unit"]
        b = block_of(sh, u)
        if p["program"] != statement(circuit, salt, coins[b[0]], b).hex():
            return None, ("statement", other_statement(u))
        if p["registered"] != registered(b[0], bytes.fromhex(w["roots"][b[0]])):
            return None, ("registered", other_registered(u, b[0]))
    return got, None


def period_fill(key: Any, ledger: Any, pins: Any, main: Any, name: Any, fills: Any, verify: Ch.Verify
                ) -> tuple[dict | None, tuple[str, str] | None]:
    """Served period `name`'s fill ``{work, k, delta, eps}`` and None, or None and the refusal, the form the Lean
    `PouwCheck.Fill.periodFill` reads. Its windows are exactly the close's of the fill period (`period_name`,
    `credit.period`), whose open committed K (``budget``); each window's session ran a commit the host's pins log `pins`
    records on main's first-parent history `main`, the window proves the circuit the fill period's open committed
    (``circuit``, `challenger.credited_session`) within its freshness floor (``pin``), and `_window` credits it
    (``fill``). A refused window refuses the fill."""
    if not (type(name) is str and type(fills) is list
            and all(isinstance(w, dict) and sorted(w) == list(FILL_KEYS) for w in fills)):
        return None, ("fill", FILLS_FORM)
    nm = period_name(name)
    why = Cr.period(key, ledger, nm, [w["window"].get("root") if isinstance(w["window"], dict) else None for w in fills],
                    verify)
    if why is not None:
        return None, why
    big_k, why = Cr.committed_k(ledger, nm)
    if why is not None:
        return None, why
    period_pin, floor = Cr.committed_pin(ledger, nm), Cr.committed_floor(ledger, nm)
    k = Ch.ChallengerKey.from_json(key)
    out, eps_ = [], []
    for i, w in enumerate(fills):
        if not (type(w["descriptor"]) is str and Cr._HEX.fullmatch(w["descriptor"])):
            return None, ("circuit", f"filler window {i} of period {nm}: {DESCRIPTOR_HEX}")
        raw = bytes.fromhex(w["descriptor"])
        try:
            audited = Ch.credited_session(ledger, w["window"], k, pins, main, raw, verify)
        except Ch.CoinError as e:
            return None, ("circuit", f"filler window {i} of period {nm} doesn't count toward it, so the fill earns "
                                     f"nothing: {e}")
        why = Cr._pin_hygiene(nm, i, main, audited.pinned, period_pin, floor)
        if why is not None:
            return None, why
        got, why = _window(key, ledger, pins, main, w, audited.session, hashlib.sha512(raw).digest(), big_k,
                            verify)
        if why is not None:
            return None, ("fill", f"filler window {i} of period {nm} is refused, so the fill earns nothing: {why[0]}: "
                                  f"{why[1]}")
        out.append(work(*got))
        e = eps(*got, big_k)
        eps_.append(None if e is None else str(e))
    return {"work": out, "k": big_k, "delta": str(Cr.DELTA), "eps": eps_}, None


__all__ = ["CALL", "CALLS_FORM", "CALLS_NOT_RECEIVED", "CHANGED", "DESCRIPTOR_FORM", "DESCRIPTOR_HEX", "DESCRIPTOR_KEYS",
           "DRAW_ROOT", "DRAW_TAG", "FILL_EPOCH", "FILL_KEYS", "FILL_SALT", "FILLS_FORM", "OPEN", "OPEN_TAG", "RECEIPTS",
           "ROOTS_FORM", "ROOTS_TAG", "SALT_NOT_RECEIVED", "SHAPE", "STATEMENT_TAG", "SUFFIX", "T_MAX", "TEMPLATE",
           "UNIT", "block_of", "blocks", "descriptor", "descriptor_of", "draw_root", "eps", "families", "open_root",
           "period_fill", "period_name", "receipts", "registered", "statement", "stratum", "work"]
