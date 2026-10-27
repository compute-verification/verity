"""PoUW's filler windows and the ``fill`` a period credits beside its ``work`` (CMP-172; Daniel's card 5d4997de, option
(a)): the Python the Lean `PouwCheck.Fill` is held to through ``pouw-period`` (`period_check`, the difftest
``lean/check/tests/test_period_difftest.py``).

A filler window is an H1 window the prover runs on tensor pipes its served windows leave idle. It is the served window's
Program (``h1window.program(linears, calls, m, tile)``, its descriptor's layout, `credit.layout_of`) at one round, its
units J10-R tiles of the period's template, and its credit is every unit's work, M·n·K a call: there are no served rows,
since every row of its A is the pad. It counts toward the companion period ``<name>:fill`` (`period_name`), which the
auditor's ledger opens and closes as it does a served period: its open commits the circuit, the draw K (`credit.K_FIELD`),
the weights' root and model (`credit.WEIGHTS_FIELD`, `credit.MODEL_FIELD`) and, for statements, the circuit files.
`work` and its rule (`credit.window`, `credit.period_credit`) are unchanged.

* **The pad is the verifier's** (`pad_entry`): K's A table ``<PREFIX>a/<K>@0`` holds M rows for each call of K, row i
  its K int8 codes SHAKE-256(`PAD_TAG` ‖ u64be(K) ‖ u64be(i)) and the f32 scale 1.0 (`PAD_SCALE`), each row
  `weights.chunks` (K) leaves as the served A tables' (`credit.registered_file` reads them so), under the public salts
  `pad_salt` of the window's open root, registered binding it. The verifier builds that registration itself and the
  window's entry must be it, so the A rows a filler unit reads are public structure, dense and never a served block, and
  a unit can't make its tile cheap by choosing its A (zero rows prove trivially).
* **The salt is the window's live coin** (`_coins`): the window's challenger session (`FILL_FAMILIES`: one salt coin and
  one draw coin) is read through the auditor's ledger (`challenger.trusted_session_of`); its log shows the window's open
  root as the one root its salt coin followed, and the window's Program salt is that coin's round-0 salt after it
  (`h1tile.live_salt`), so every unit's noise is drawn after its operands are fixed and no salt serves two windows.
* **The open root binds the weights and the session** (`open_root`): `salt.window_open_root` of the weights' root, the
  constant `FILL_REQUESTS` in the request table's place, and the session. `credit._period_weights` holds each K's
  ``<PREFIX>b/<K>`` to it and to the weights' root the fill period's open signed, so a filler unit reads the period's
  registered B rows.
* **Every value is fixed before the draw** (`entries_root`, `draw_root`): the draw root is over the Program salt and the
  digest of every registered entry (each unit's z/y record table ``<PREFIX>zy@<c>`` and receipt table
  ``<PREFIX>receipt@<c>`` among them); the challenger's log shows it as the one root the draw coin followed, and the draw
  is `credit._window_drawn`'s over the window's strata at the fill period's K.
* **No partial credit**: every drawn unit has exactly one proof outcome and it is an accept (`credit._proved`), each
  verified against the committed circuit and opening its K's pad and weights at their registered roots
  (`credit._registration`), and a refused window refuses the whole fill period. The credit is ``{work, k, delta, eps}``:
  each window's work (`work`, every unit's R·C·K), the fill period's K, `credit.DELTA` and the ε its draw established
  over all of it (`eps`).

A filler window's evidence is `FILL_KEYS` (each drawn unit's proof outcome, the Lean's form) or `FILL_STATEMENT_KEYS`
(its statements' proof files, which `fill_period` verifies with the verifier of record, `credit._runner`)."""
from __future__ import annotations

import hashlib
import struct
from typing import Any

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import credit as Cr
from verity.core.protocols.pouw import salt as PS
from verity.core.protocols.pouw import verifier as V
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.circuit import h1tile as HT
from verity.core.protocols.pouw.protocol import tagged
from verity.core.protocols.pouw.weights import chunks, row_bytes, row_words
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.sampled_proofs.one_stage import proofs as PF
from verity.core.protocols.sampled_proofs.one_stage import registered as G

#: the companion period a served period's filler windows count toward: ``<name>:fill``
SUFFIX = ":fill"
#: a filler window's coin layout: its salt coin, then its draw coin
FILL_FAMILIES = ((Wd.SALT, 1), (Wd.DRAW, 1))
#: what stands in a filler window's open root for a served window's round-0 request table's root
FILL_REQUESTS = hashlib.sha512(b"verity/pouw/fill/requests/v1\0").digest()
DRAW_TAG = "verity/pouw/fill/draw/v1"
ENTRIES_TAG = "verity/pouw/fill/entries/v1"
PAD_TAG = b"verity/pouw/fill/pad/v1\0"
PAD_SALT_TAG = b"verity/pouw/fill/pad-salt/v1\0"
#: every pad row's scale s_a, an f32 little-endian
PAD_SCALE = struct.pack("<f", 1.0)
FILL_KEYS = ("descriptor", "draw", "entries", "opening", "proofs", "salt", "session", "window")
FILL_STATEMENT_KEYS = ("descriptor", "draw", "entries", "opening", "salt", "session", "statements", "window")
FILLS_FORM = ("the period's filler windows are not a list of each filler window's evidence {" + ", ".join(FILL_KEYS)
              + "}, in the order its fill period's close lists them")
FILL_EPOCH = "the coin commitment's coins are not a filler window's: its salt coin, then its draw coin"
SALT_NOT_RECEIVED = "the challenger's log does not show the filler window's open root as the one root its salt coin followed"
FILL_SALT = ("the filler window's Program salt is not its live salt coin's round-0 salt after its open root "
             "(h1tile.live_salt): seed its units from the coin the challenger opened")
ENTRIES_FORM = ("the filler window's registered entries are not a list of one-stage's {value, root, leaves, words, "
                "domain}")
RECEIPTS = ("the filler window's draw receipts are not the digest of its registered entries (entries_root): register "
            "every table before the draw coin and draw over them")
DRAW_ROOT = "the filler window's draw root is not over its Program salt and its registered entries (draw_root)"
CHANGED = ("the draw root over the filler window's Program salt and registered entries is not the one root the "
           "challenger's log shows its draw coin followed: a value changed after the draw coin")


def period_name(name: str) -> str:
    """The fill period of served period `name`."""
    return name + SUFFIX


def fill_epoch_reason(c: Ch.LiveCommitment) -> str | None:
    """Why `c` is not a filler window's coin commitment (`window.LIVE_SCHEME`, `FILL_FAMILIES`); None when it is."""
    if c.scheme != Wd.LIVE_SCHEME:
        return f"the coin commitment's scheme is {c.scheme}, not {Wd.LIVE_SCHEME}"
    return None if tuple(c.families) == FILL_FAMILIES else FILL_EPOCH


def open_root(weights_root: bytes, session: str) -> bytes:
    """A filler window's open root: `salt.window_open_root` of its weights' root, `FILL_REQUESTS` and its session."""
    return PS.window_open_root(weights_root, FILL_REQUESTS, session)


def pad_codes(k: int, row: int) -> bytes:
    """Pad row `row`'s K int8 codes at K = `k`."""
    return hashlib.shake_256(PAD_TAG + k.to_bytes(8, "big") + row.to_bytes(8, "big")).digest(k)


def pad_leaves(k: int, rows: int) -> list[list[int]]:
    """The leaves of `rows` pad rows at K = `k`: row i's x = `weights.chunks` (K) leaves, chunk c its K/x codes then
    `PAD_SCALE`, as u16 words."""
    x = chunks(k)
    kc = k // x
    out = []
    for i in range(rows):
        codes = pad_codes(k, i)
        out += [row_words(row_bytes(codes[c * kc:(c + 1) * kc], PAD_SCALE)) for c in range(x)]
    return out


def pad_salt(window: str, leaf: int) -> bytes:
    """Pad leaf `leaf`'s public hm96 salt in the window whose open root is `window` (hex): SHAKE-256 of `PAD_SALT_TAG` ‖
    window ‖ 0 ‖ u64be(leaf), `core_check.SALT_BYTES` bytes."""
    return hashlib.shake_256(PAD_SALT_TAG + window.encode() + b"\0" + leaf.to_bytes(8, "big")).digest(CC.SALT_BYTES)


def pad_rows(layout: dict) -> dict[int, int]:
    """Each K of `layout` (one `credit.strata` accepted) and its pad table's rows: M for each call of K."""
    ks = {name: k for name, k, _n in layout["linears"]}
    out: dict[int, int] = {}
    for lin, _blk in layout["calls"]:
        out[ks[lin]] = out.get(ks[lin], 0) + layout["m"]
    return out


def pad_entry(program: str, prefix: str, window: str, k: int, rows: int) -> dict:
    """K's pad table ``<prefix>a/<K>@0`` of `rows` rows as the filler window whose open root is `window` registers it,
    under the circuit whose descriptor's SHA-512 is `program`."""
    leaves = pad_leaves(k, rows)
    v = f"{prefix}a/{k}@0"
    return G.Registered.commit(program, v, leaves, [pad_salt(window, i) for i in range(len(leaves))], G.WORD_BITS,
                               G.SCHEMA_V2, window=window).entry()


def entries_root(entries: list[dict]) -> bytes:
    """The digest of a filler window's registered entries, in order: `tagged` of each one's canonical JSON."""
    return tagged(ENTRIES_TAG, *(Cr._canonical(e).encode() for e in entries))


def draw_root(salt: bytes, receipts: bytes) -> bytes:
    """A filler window's draw root, over its Program salt and `entries_root` of its entries."""
    return tagged(DRAW_TAG, salt, receipts)


def work(layout: dict) -> int:
    """A filler window's work: every unit's R·C·K, M·n·K a call."""
    return sum(size * w for size, w in Cr.call_work(layout))


def eps(layout: dict, k: int) -> Any:
    """The ε a draw of K = `k` establishes at `credit.DELTA` over the whole of a filler window's work (`window.eps_at`),
    or None when it establishes none."""
    try:
        return Wd.eps_at(k, Cr.DELTA, work(layout), work(layout), 1)
    except ValueError:
        return None


def _coins(key: Any, ledger: Any, pins: Any, main: Any, window: Any, session: Any, salt: Any, opening: Any, draw: Any,
           entries: Any, verify: Ch.Verify) -> tuple[str, str] | None:
    """None when the filler window's salt and draw coins are live, read through its challenger's session; else the
    refusal (``coin``, ``salt``, ``fill`` or ``draw``)."""
    try:
        k = Ch.ChallengerKey.from_json(key)
    except Ch.CoinError as e:
        return "coin", f"the verifier's challenger key is malformed: {e}"
    r = Cr._digest(window.get("root")) if isinstance(window, dict) else None
    if r is None:
        return "fill", Cr.WINDOW_ROOT_FORM
    try:
        c, received = Ch.trusted_session_of(ledger, window, k, pins, main, session, verify)
    except Ch.CoinError as e:
        return "coin", f"the challenger's session is not trusted: {e}"
    why = fill_epoch_reason(c)
    if why is not None:
        return "coin", why
    if tuple(received.get(Wd.SALT, ())) != (r,):
        return "salt", SALT_NOT_RECEIVED
    try:
        o = Ch.LiveOpening.from_json(opening)
    except Ch.CoinError as e:
        return "coin", f"the salt's {e}"
    why = Ch.live_reason(c, o, Wd.SALT, 0, r)
    if why is not None:
        return "coin", f"the salt's {why}"
    if salt != HT.live_salt(c, o.coin, 0, r).hex():
        return "salt", FILL_SALT
    if not (type(entries) is list and all(Cr._entry(e) for e in entries)):
        return "fill", ENTRIES_FORM
    receipts = entries_root(entries)
    if not (isinstance(draw, dict) and draw.get("receipts") == receipts.hex()):
        return "draw", RECEIPTS
    root = draw_root(bytes.fromhex(salt), receipts)
    if draw.get("root") != root.hex():
        return "draw", DRAW_ROOT
    got = tuple(received.get(Wd.DRAW, ()))
    if not got:
        return "draw", Cr.NO_DRAW
    if got != (root,):
        return "draw", CHANGED
    try:
        why = Ch.live_reason(c, Ch.LiveOpening.from_json(draw.get("coin")), Wd.DRAW, 0, root)
    except Ch.CoinError as e:
        why = str(e)
    return None if why is None else ("coin", f"the draw coin: {why}")


def no_pad(v: str) -> str:
    return (f"the filler window registers no pad table {v}, or registers it twice: each K's A rows are its one pad "
            "table, which the verifier builds (pad_entry)")


def other_pad(v: str) -> str:
    return (f"the filler window's {v} is not the pad (pad_entry: each row its public codes and scale 1.0 under the "
            "public pad salts of the window's open root, registered binding it): a filler unit's A rows are the pad")


def other_value(v: str, prefix: str) -> str:
    return (f"the filler window registers {v}, which is none of its values: a filler window registers each K's pad "
            f"{prefix}a/<K>@0 and weights {prefix}b/<K>, and each call's {prefix}zy@<c> and {prefix}receipt@<c>, "
            "nothing else")


def _pad(entries: list[dict], layout: dict, program: str, prefix: str, window: str) -> tuple[str, str] | None:
    """None when the filler window's `entries` register each K's pad table (`pad_entry`) once, and no value but those,
    each K's weights and each call's z/y and receipt tables; else the refusal (``pad``)."""
    rows = pad_rows(layout)
    pads = {f"{prefix}a/{k}@0": k for k in rows}
    others = {f"{prefix}b/{k}" for k in rows} | {f"{prefix}{t}@{c}" for t in ("zy", "receipt")
                                                  for c in range(len(layout["calls"]))}
    names = [e["value"] for e in entries]
    for v in names:
        if v not in pads and v not in others:
            return "pad", other_value(v, prefix)
    for v, k in pads.items():
        if names.count(v) != 1:
            return "pad", no_pad(v)
        if entries[names.index(v)] != pad_entry(program, prefix, window, k, rows[k]):
            return "pad", other_pad(v)
    return None


#: what a filler window's drawn units' proof outcomes are, given its layout, descriptor, drawn units and round count
Proved = Cr.Proved


def _window(key: Any, ledger: Any, pins: Any, main: Any, window: Any, session: Any, salt: Any, opening: Any, draw: Any,
            descriptor: Any, entries: Any, proved: Proved, k: Any, verify: Ch.Verify
            ) -> tuple[dict | None, tuple[str, str] | None]:
    """A filler window's layout and None, or None and the refusal of the first step that refuses: `_coins`, the layout
    of its circuit (`credit.layout_of` of `descriptor`), its weights (`credit._period_weights` at `open_root`), `_pad`,
    the draw at the fill period's committed K `k` (`credit._window_drawn`), each drawn unit's one accepted outcome
    (`credit._proved`) and its registration (`credit._registration`, one round)."""
    why = _coins(key, ledger, pins, main, window, session, salt, opening, draw, entries, verify)
    if why is not None:
        return None, why
    if not (type(descriptor) is str and Cr._HEX.fullmatch(descriptor)):
        return None, ("layout", Cr.DESCRIPTOR_HEX)
    raw = bytes.fromhex(descriptor)
    layout, why = Cr.layout_of(raw)
    if why is not None:
        return None, why
    program, prefix = hashlib.sha512(raw).hexdigest(), Cr._prefix(raw)
    (sid, _period), why = Cr._period_weights(key, ledger, window, [{"requests": FILL_REQUESTS.hex()}], layout, entries,
                                             program, prefix, verify)
    if why is not None:
        return None, why
    ss, _ = Cr.strata(layout)
    why = _pad(entries, layout, program, prefix, window["root"])
    if why is not None:
        return None, why
    units, why = Cr._window_drawn(window, draw, ss, k)
    proofs = None
    if why is None:
        proofs, why = proved(layout, raw, units, 1)
    if why is None:
        why = Cr._proved(units, proofs)
    if why is None:
        why = Cr._registration(window["root"], program, prefix, entries, ss, layout, 1, proofs, sid)
    return (None, why) if why is not None else (layout, None)


def fill_period(key: Any, ledger: Any, pins: Any, main: Any, name: Any, fills: Any, verify: Ch.Verify,
                run: Cr.Run = PF.verify) -> tuple[dict | None, tuple[str, str] | None]:
    """Served period `name`'s fill and None, or None and the refusal: `_period_outcomes`' rules, each filler window's
    evidence (`FILL_STATEMENT_KEYS`) carrying its statements' proof files, which the verifier of record `run` verifies on
    the circuit files the fill period's open committed (`credit.committed_files`)."""
    return _period_of(key, ledger, pins, main, name, fills, verify, run)


def _period_outcomes(key: Any, ledger: Any, pins: Any, main: Any, name: Any, fills: Any, verify: Ch.Verify
                     ) -> tuple[dict | None, tuple[str, str] | None]:
    """`fill_period` on each drawn unit's proof outcome (`FILL_KEYS`), the form the Lean `PouwCheck.Fill.periodFill`
    reads. The fill ``{work, k, delta, eps}`` of served period `name`, and None; or None and the refusal. Its windows are
    exactly the close's of the fill period (`period_name`, `credit.period`), whose open committed K (``budget``) and
    signed the weights' root (``weights``); each window's session ran a commit the host's pins log `pins` records on
    main's first-parent history `main`, the window proves the circuit the fill period's open committed (``circuit``,
    `challenger.credited_session`) within its freshness floor (``pin``), and `_window` credits it (``fill``). A refused
    window refuses the fill."""
    return _period_of(key, ledger, pins, main, name, fills, verify, None)


def _period_of(key: Any, ledger: Any, pins: Any, main: Any, name: Any, fills: Any, verify: Ch.Verify,
               run: Cr.Run | None) -> tuple[dict | None, tuple[str, str] | None]:
    keys = FILL_KEYS if run is None else FILL_STATEMENT_KEYS
    if not (type(name) is str and type(fills) is list
            and all(isinstance(w, dict) and sorted(w) == list(keys) for w in fills)):
        return None, ("fill", FILLS_FORM)
    nm = period_name(name)
    why = Cr.period(key, ledger, nm, [w["window"].get("root") if isinstance(w["window"], dict) else None for w in fills],
                    verify)
    if why is not None:
        return None, why
    big_k, why = Cr.committed_k(ledger, nm)
    if why is None:
        _, why = Cr.committed_weights(ledger, nm)
    if why is not None:
        return None, why
    period_pin, floor = Cr.committed_pin(ledger, nm), Cr.committed_floor(ledger, nm)
    files = None
    if run is not None:
        files, why = (Cr._lean_committed_files if V.RECORD_VERDICT == "lean" else Cr.committed_files)(ledger, nm)
        if why is not None:
            return None, why
    k = Ch.ChallengerKey.from_json(key)
    out, eps_ = [], []
    for i, w in enumerate(fills):
        if not (type(w["descriptor"]) is str and Cr._HEX.fullmatch(w["descriptor"])):
            return None, ("circuit", f"filler window {i} of period {nm}: {Cr.DESCRIPTOR_HEX}")
        try:
            audited = Ch.credited_session(ledger, w["window"], k, pins, main, bytes.fromhex(w["descriptor"]), verify)
        except Ch.CoinError as e:
            return None, ("circuit", f"filler window {i} of period {nm} doesn't count toward it, so the fill earns "
                                     f"nothing: {e}")
        why = Cr._pin_hygiene(nm, i, main, audited.pinned, period_pin, floor, "filler window")
        if why is not None:
            return None, why
        proved = ((lambda *_, p=w["proofs"]: (p, None)) if run is None
                  else Cr._runner(w["statements"], files, w["entries"], w["window"], w["salt"], run))
        layout, why = _window(key, ledger, pins, main, w["window"], w["session"], w["salt"], w["opening"], w["draw"],
                              w["descriptor"], w["entries"], proved, big_k, verify)
        if why is not None:
            return None, ("fill", f"filler window {i} of period {nm} is refused, so the fill earns nothing: {why[0]}: "
                                  f"{why[1]}")
        out.append(work(layout))
        e = eps(layout, big_k)
        eps_.append(None if e is None else str(e))
    return {"work": out, "k": big_k, "delta": str(Cr.DELTA), "eps": eps_}, None


__all__ = ["CHANGED", "DRAW_ROOT", "DRAW_TAG", "ENTRIES_FORM", "ENTRIES_TAG", "FILL_EPOCH", "FILL_FAMILIES", "FILL_KEYS",
           "FILL_REQUESTS", "FILL_SALT", "FILL_STATEMENT_KEYS", "FILLS_FORM", "PAD_SALT_TAG", "PAD_SCALE", "PAD_TAG",
           "RECEIPTS", "SALT_NOT_RECEIVED", "SUFFIX", "draw_root", "entries_root", "eps", "fill_epoch_reason",
           "fill_period", "open_root", "pad_codes", "pad_entry", "pad_leaves", "pad_rows", "pad_salt", "period_name",
           "work"]
