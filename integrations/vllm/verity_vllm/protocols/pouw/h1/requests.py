"""H1's requests and rounds (`Accounting.Served`: a request entry is an `act` value, a run input under the root of the round that admits it, committed before that round's coin).
`pouw_h1.H1Run` builds them as it serves; this module holds what they are and how they are checked.

* **Rounds.** Round s is step s, one round a step. vLLM's scheduler admits a request in the step that first schedules it
  (`SchedulerOutput.scheduled_new_reqs`), and `pouw_h1_hooks.admitting`, the option's wrap of the model runner's
  ``execute_model``, hands those (`new_requests`) and the step's batch order (`batch_order`: Model Runner V2's own, each
  scheduled request's rows in the order its A tables hold them) to `pouw_h1.H1Run.admit` before the step's forward. Round s closes
  once step s's last linear is recorded: its root (`round_root`) is over the step's calls (`step_digest`, their manifest
  rows with what the listeners bound), its request table's root and its slot table's root, and only then does the round
  coins' `LiveRecord` (the auditor host's session, the pouw setting ``challenger``; ``rounds`` in tests) open round s's coin.
  Round 0's request table comes before the salt: its root is in the window's open root (`salt.window_open_root`, over
  the weights' root, it and the session), which the gate opens the session on and the salt coin opens after
  (`pouw_h1.H1Run.open_window`), so its binding reads no salt (`requests_binding`) and every later round's does.
* **Entries and slots.** Each `RequestEntry` (its ordinal, round, first slot, cap and prompt tokens) is a hiding leaf of its
  round's request table (`request_table`: T hm96 leaves, the unused ones `EMPTY_REQUEST`), and slot i of round s's slot
  table (`slot_table`, T hm96 leaves) names the request whose row is slot i of step s's A tables, or none (`EMPTY_SLOT`). A
  request is keyed by its vLLM ``req_id``: one resumed after a preemption (Model Runner V2 puts it back in
  ``scheduled_new_reqs``) keeps its entry and is counted readmitted. A served row of a request with no entry (refused at
  admission: prompt embeddings, a pooling request, past the window; or never admitted) is a gap
  (`pouw_h1.H1Run.request_gaps`), and so is a step whose batch isn't its served rows; a window with a gap credits nothing.
* **The checks.** `request_reason` refuses an entry opened under another round's root or under none, a root whose coin
  isn't the challenger's opened after it (`live_reason`), and an entry whose first slot in its round doesn't name it.
  `slot_reason` refuses a served slot that names no request, or one whose entry is under a later round than the slot's
  step, so an entry deferred past the step that first serves it is refused. `rounds_reason` holds the whole log to the
  challenger's coins, and `pouw_h1_challenger.trusted_rounds_reason` holds those coins to the session the auditor's signed
  ledger opens and ends for the window, its record and log signed by the verifier's own challenger, and the rounds to
  PoUW's round rule of record (`core_check.rounds`).
* **Admission** (`ADMISSIONS`, the pouw setting ``admission``): the period fixes a window's live tiles before its
  first coin, so by default (``round-0``) a window admits requests only in round 0, and an entry under a later round is
  refused (`request_reason`); ``per-round`` admits in every round, for a statement that fixes each round's live tiles
  before that round's coin.
* **Round values** (the live order): each call's A table (its T slots' s8 rows and f32 scales) and, when
  its arm hands them over, its z table (one leaf a proof block, its z/y record: z, then y = dq(z)) are hiding tables of the call (`pouw_h1.H1Run`, ``tables`` in its manifest row),
  so the step digest, and with it round s's root, commits them before round s's coin, and slot i of round s's slot
  table names leaf i of those A tables. `tables_reason` refuses a window where it names leaves of an A table no round
  root commits: a call with no A table root, at a step no round closes, or off its round's step digest. Round s + 1's
  root also commits round s's receipt roots (`receipts_digest`), and the draw coin opens after `draw_root`, over the
  manifest, the last round's receipts and every round's root and coin, so the draw follows every round coin by hash.
* **The A rows under the round roots** (`credit.a_link`): the root round s's coin follows is over the round's
  ``inner`` root (all of the above, `round_root`) and its ``a`` segments: step s's A table ``a/<K>@<s>`` of each K, the
  rows of its calls of K call after call, whole, as the aligned subtrees of its registered tree (`round_row`,
  `pouw_h1_served.Served.segments`), registered as the run records the step's last call of K. So each call's A rows, the
  ones its proof blocks read, are fixed under its own round's root before that round's coin, and no other round lists them.
  Round s's input-ids table ``ids@<s>`` (ROOTS1, `pouw_h1_commit`) is listed there whole too. Round 0's is built at
  admission, before the window opens: its prompts' tokens in step 0's batch order (`first_ids`), each request's first
  rows at the first slot its entry names. Its root is inside round 0's request root (`credit.requests_root` of the
  request table's root and it; round 0's row keeps the table's as ``request_table``), so the window's open root fixes
  the prompt tokens before the salt; its domain binds the window's session in place of that root.
* **Not yet.** The round root has no admit root (H1 has no admit proof block yet), no producer hands the run a receipt root
  yet, and no proof block reads the round coins yet.
"""
from __future__ import annotations

import json
import struct
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.audit import HM96, HM96_SALT, ROW_SCHEMA, rows_domain
from verity.core.protocols.pouw import credit as CR
from verity.core.protocols.pouw.credit import CREDIT
from verity.core.protocols.pouw.protocol import tagged
from verity.core.protocols.sampled_proofs.challenger import CoinError, LiveCommitment, LiveOpening, live_reason
from verity.ml.commitments import Commitment, MerkleTree, Opening, VerificationLimits, verify_opening


@dataclass(frozen=True)
class RequestEntry:
    """An admitted request's entry (`Accounting.Served.Cls.act`, "a request entry its round admits"): its ordinal
    among the window's admitted requests, the round that admitted it (its engine step), its first slot there (the first of
    its rows in that step's A tables, which round ``round``'s slot table names it at), its token cap and its prompt's token
    ids. It is a value, a hiding leaf of its round's request table under that round's root."""

    index: int
    round: int
    slot: int
    max_tokens: int
    prompt: tuple[int, ...]

    def row(self) -> bytes:
        """The leaf's preimage: index, round, slot, cap and prompt length as u32 LE, then the prompt's token ids as u32 LE."""
        return struct.pack("<5I", self.index, self.round, self.slot, self.max_tokens, len(self.prompt)) + \
            np.asarray(self.prompt, dtype="<u4").tobytes()


#: an unused leaf of a round's request table: every entry's row is longer (it has a prompt token)
EMPTY_REQUEST = bytes(20)
#: a slot table's leaf for a slot no request's row is in: a named slot's is its request's index + 1
EMPTY_SLOT = bytes(4)
#: the hm96 salt stream's nonces of round r's request and slot tables are REQUESTS_NONCE + r and SLOTS_NONCE + r. Both sit
#: inside `audit`'s activation nonces (below 2^62); the streams are apart from the activation trees' only because each run
#: draws its own salt key (`pouw_h1.H1Run`'s ``key``), which nothing persists
REQUESTS_NONCE = 1 << 61
SLOTS_NONCE = REQUESTS_NONCE + (1 << 40)
#: call c's A and z tables' nonces are TABLES_NONCE + 3c + 0 and 2 (`TABLE_NONCES`; 1 was the retired y table's)
TABLES_NONCE = REQUESTS_NONCE + (1 << 41)
#: a call's hiding tables under the run's salt key (`pouw_h1.H1Run`): A's T slot rows and scales, z's records
#: (`pouw_h1_records`)
TABLE_TAGS = {"a": "verity/pouw/h1/a-table/v1", "z": "verity/pouw/h1/z-table/v2"}
TABLE_NONCES = {"a": 0, "z": 2}    # j of each table's hm96 nonces
ROUND_TAG = "verity/pouw/h1/round/v3"
STEP_TAG = "verity/pouw/h1/step-calls/v1"
REQUESTS_TAG = "verity/pouw/h1/requests/v2"
OPEN_REQUESTS_TAG = "verity/pouw/h1/requests/round-0/v1"
SLOTS_TAG = "verity/pouw/h1/slots/v1"
RECEIPTS_TAG = "verity/pouw/h1/receipts/v1"
DRAW_TAG = "verity/pouw/h1/draw/v1"
#: round 0's ``receipts``: no round precedes it
NO_RECEIPTS = bytes(64)
#: the pouw setting ``admission``: requests admitted in round 0 only (the default), or in every round
ROUND0, PER_ROUND = "round-0", "per-round"
ADMISSIONS = (ROUND0, PER_ROUND)
#: the credit refusal of a call in a window that served a row of a request with no entry (`pouw_h1.H1Run.request_gaps`)
UNENTERED = "credit: request-without-entry: the window served a row of a request with no entry (details.requests.gaps)"


def late_admission(r: int) -> str:
    return (f"admitted in round {r}, after round 0: the period fixes a window's live tiles before its first coin, "
            f"so a window admits only in round 0 (the pouw setting 'admission': '{PER_ROUND}' admits in every round)")


def requests_binding(salt: bytes | None, r: int) -> bytes:
    """Round r's request table's binding. Round 0's reads no salt, since its root is committed before the salt opens."""
    return tagged(OPEN_REQUESTS_TAG, 0) if r == 0 else tagged(REQUESTS_TAG, salt, r)


def slots_binding(salt: bytes, r: int) -> bytes:
    return tagged(SLOTS_TAG, salt, r)


def slot_leaf(owner: int | None) -> bytes:
    return EMPTY_SLOT if owner is None else struct.pack("<I", owner + 1)


def slot_owner(leaf: bytes) -> int | None:
    """The request index a slot leaf's preimage names, None for `EMPTY_SLOT`; ValueError on any other preimage."""
    if len(leaf) != len(EMPTY_SLOT):
        raise ValueError(f"a slot leaf is {len(EMPTY_SLOT)} bytes, not {len(leaf)}")
    (n,) = struct.unpack("<I", leaf)
    return n - 1 if n else None


def request_table(salt: bytes | None, r: int, rows: int, entries: Sequence[RequestEntry], key: bytes) -> Any:
    """Round r's request table: `rows` (T) hiding leaves, the entries the round admits in admission order, then
    `EMPTY_REQUEST`. Each admitted request schedules a token row in the step that admits it, so a step admits at most T."""
    if len(entries) > rows:
        raise ValueError(f"round {r} admits {len(entries)} requests, past its {rows} leaves")
    values = [e.row() for e in entries] + [EMPTY_REQUEST] * (rows - len(entries))
    return hiding_rows(requests_binding(salt, r), r, values, key, REQUESTS_NONCE + r)


def first_ids(prompts: Mapping[str, Sequence[int]], order: Sequence[tuple[str, int]]) -> list[int | None]:
    """Step 0's input token ids in its batch order (`batch_order`): each request's first rows are its first prompt tokens
    (a window opens on new requests, with no prefix cached), None for a row past its prompt or of a request with none."""
    out: list[int | None] = []
    for req_id, rows in order:
        p = prompts.get(req_id) or ()
        out += [int(p[i]) if i < len(p) else None for i in range(int(rows))]
    return out


def slot_table(salt: bytes, r: int, rows: int, owners: Sequence[int | None], key: bytes) -> Any:
    """Round r's slot table: `rows` (T) hiding leaves, leaf i naming the request whose row is slot i of step r's A tables
    (`owners`, in the step's batch order), `EMPTY_SLOT` past them."""
    values = [slot_leaf(o) for o in owners[:rows]] + [EMPTY_SLOT] * (rows - min(len(owners), rows))
    return hiding_rows(slots_binding(salt, r), r, values, key, SLOTS_NONCE + r)


def table_leaves(rows: int, width: int, m: int, *parts: np.ndarray) -> list[bytes]:
    """A call's table leaves: slot i < m the bytes of row i of each of `parts` in turn, every later slot `width` zero bytes."""
    out = [b"".join(p[i].tobytes() for p in parts) for i in range(m)]
    return out + [bytes(width)] * (rows - m)


def hiding_rows(binding: bytes, owner: int, values: Sequence[bytes], key: bytes, nonce: int) -> MerkleTree:
    """``audit.commit_rows(binding, owner, values, hash=HM96, salt_key=key, nonce=nonce)``, its leaf salts drawn as one
    keystream: leaf i's (`audit.hm96_salt`) is ChaCha20 blocks 3i..3i + 2 at a 64-bit counter and nonce, which is RFC
    8439's stream (`ml.windows.rows.chacha20`) with the counter's high word, 0 below 2^32 blocks, first in its nonce.
    The per-leaf block function costs about 0.2 ms a leaf, a z table's thousands of leaves a call."""
    from verity.ml.windows.rows import chacha20
    y = chacha20(key, bytes(4) + nonce.to_bytes(8, "little"), len(values) * HM96_SALT)
    rows = {i: v + y[i * HM96_SALT:(i + 1) * HM96_SALT] for i, v in enumerate(values)}
    return MerkleTree(rows_domain(binding, owner, len(values), HM96), rows, lambda _: ROW_SCHEMA)


def step_digest(salt: bytes, s: int, rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Step s's calls: SHA-512 over their manifest rows (`pouw_h1.H1Run.manifest`), in order, with what each call's listeners
    bound (root(A) among it, once AROW binds it), each without its credit bit (`credit.CREDIT`): the step's last verdict
    comes after its round closes, and calls_root commits the bits before the sample root."""
    return tagged(STEP_TAG, salt, s, *(json.dumps({k: v for k, v in row.items() if k != CREDIT}, sort_keys=True)
                                       for row in rows))


def _root64(h: Any) -> bool:
    try:
        return isinstance(h, str) and len(bytes.fromhex(h)) == 64
    except ValueError:
        return False


def tables_reason(salt: bytes, log: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> str | None:
    """Why a slot of some round's slot table names an A-table leaf that no round root commits, given the window's round log
    and its manifest rows (`pouw_h1.H1Run.manifest`): a call at a step no round of the log closes, a logged round whose step
    has no call, a call whose row carries no A table root (``tables``' ``a``, 64 bytes), or a step whose rows aren't the
    ones its round's ``calls`` digest is over (`step_digest`); None when each call's A table is under its own round's
    root, and so recorded before that round's coin."""
    steps: dict[Any, list[Mapping[str, Any]]] = {}
    for row in rows:
        steps.setdefault(row.get("step"), []).append(row)
    logged = [x.get("round") for x in log]
    for s, got in steps.items():
        if s not in logged:
            return (f"call {got[0].get('index')} is at step {s}, which no round of the log closes, so its A table is under "
                    "no round root")
    for x in log:
        r, got = x.get("round"), steps.get(x.get("round"), [])
        if not got:
            return f"round {r}'s step has no call, so its slot table names leaves of no committed A table"
        bare = next((m for m in got if not _root64((m.get("tables") or {}).get("a"))), None)
        if bare is not None:
            return (f"call {bare.get('index')} of step {r} has no A table root, so round {r}'s slot table names A-table "
                    "leaves that no round root commits")
        if step_digest(salt, r, got).hex() != x.get("calls"):
            return f"step {r}'s calls in the record (their tables' roots among them) are not the ones under round {r}'s root"
    return None


def round_root(salt: bytes, r: int, calls: bytes, requests: bytes, slots: bytes, receipts: bytes) -> bytes:
    """Round r's root, over step r's calls (`step_digest`, their A and z tables among them), the root of the request
    table of the entries it admits, the root of its slot table and round r - 1's receipts (`receipts_digest`;
    `NO_RECEIPTS` in round 0)."""
    return tagged(ROUND_TAG, salt, r, calls, requests, slots, receipts)


def round_row(salt: bytes, r: int, calls: bytes, requests: bytes, slots: bytes, receipts: bytes,
              segments: Mapping[str, Sequence[tuple[int, int, bytes]]]) -> dict[str, Any]:
    """Round r's row of the log before its coin: its parts, its ``inner`` root (`round_root`) and ``root``, the one the
    challenger receives, over that and each A value's ``a`` segments (`credit.round_root`: ``[level, index, root]``
    subtrees of the round's rows, `pouw_h1_served.Served.segments`)."""
    inner = round_root(salt, r, calls, requests, slots, receipts)
    return {"round": r, "root": CR.round_root(r, inner, {v: list(x) for v, x in segments.items()}).hex(),
            "inner": inner.hex(), "a": {v: [[lv, i, h.hex()] for lv, i, h in segments[v]] for v in sorted(segments)},
            "calls": calls.hex(), "requests": requests.hex(), "slots": slots.hex(), "receipts": receipts.hex()}


def round_reason(salt: bytes, row: Mapping[str, Any]) -> str | None:
    """Why a round's row (`round_row`) isn't its root over its parts: its inner root over its calls, its request and
    slot tables and the receipts before it, and its root over that and its A segments; None when it is."""
    try:
        r = row["round"]
        calls, requests, slots, receipts, inner, root = (bytes.fromhex(row[k]) for k in (
            "calls", "requests", "slots", "receipts", "inner", "root"))
        segments = {v: [(int(lv), int(i), bytes.fromhex(h)) for lv, i, h in x] for v, x in row["a"].items()}
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        return f"a round of the log is malformed ({e})"
    if round_root(salt, r, calls, requests, slots, receipts) != inner:
        return f"round {r}'s root is not over its calls, its request table, its slot table and the receipts before it"
    if CR.round_root(r, inner, segments) != root:
        return f"round {r}'s root is not over its inner root and its A segments (`credit.round_root`)"
    return None


def receipts_digest(salt: bytes, r: int, roots: Sequence[tuple[int, bytes | None]]) -> bytes:
    """Round r's receipts: each of its calls' (index, receipt root), in call order, an empty root where none was handed
    over; committed under round r + 1's root, or for the last round in `draw_root`."""
    return tagged(RECEIPTS_TAG, salt, r, *(p for i, root in roots for p in (i, root or b"")))


def draw_root(calls_root: bytes, receipts: bytes, log: Sequence[Mapping[str, Any]]) -> bytes:
    """What the draw coin opens after: the window's manifest (``calls_root``), its last round's receipts and every round's
    root and coin, so the draw follows every round coin by hash."""
    return tagged(DRAW_TAG, calls_root, receipts,
                  *(p for x in log for p in (bytes.fromhex(x["root"]), bytes.fromhex((x.get("coin") or {}).get("coin", "")))))


@dataclass(frozen=True)
class RoundOpening:
    """A leaf opened under round ``round``'s request or slot table: its position, its value (the leaf's preimage and its
    hm96 salt) and its path."""

    round: int
    position: int
    value: bytes
    path: tuple[bytes, ...]


def _round(log: Sequence[Mapping[str, Any]], commitment: LiveCommitment, salt: bytes, r: int
           ) -> tuple[Mapping[str, Any] | None, str | None]:
    """Round r's row of the log, once its root is over its parts and its coin is the challenger's round coin r, opened
    after that root (`live_reason`); else why not."""
    row = next((x for x in log if x.get("round") == r), None)
    if row is None:
        return None, f"the round log has no round {r}"
    why = round_reason(salt, row)
    if why is not None:
        return None, why
    root = bytes.fromhex(row["root"])
    if row.get("coin") is None:
        return None, f"round {r} has no live coin, so nothing shows its root committed before one"
    try:
        why = live_reason(commitment, LiveOpening.from_json(row["coin"]), Wd.ROUND, r, root)
    except CoinError as e:
        why = str(e)
    return (row, None) if why is None else (None, f"round {r}'s coin: {why}")


def _leaf(binding: bytes, r: int, root_hex: str, rows: int, opening: RoundOpening) -> bool:
    return verify_opening(rows_domain(binding, r, rows, HM96), Commitment(bytes.fromhex(root_hex), rows),
                          Opening(opening.position, opening.value, opening.path), ROW_SCHEMA, VerificationLimits())


def _slot(row: Mapping[str, Any], salt: bytes, rows: int, opening: RoundOpening) -> tuple[int | None, str | None]:
    """The request slot `opening` names under its round's slot table, or why it doesn't open there."""
    s, p = opening.round, opening.position
    if not _leaf(slots_binding(salt, s), s, row["slots"], rows, opening):
        return None, f"slot {p} doesn't open under round {s}'s slot table"
    try:
        return slot_owner(opening.value[:-HM96_SALT]), None
    except ValueError as e:
        return None, f"slot {p} of round {s}: {e}"


def request_reason(log: Sequence[Mapping[str, Any]], commitment: LiveCommitment, salt: bytes, rows: int,
                   entry: RequestEntry, opening: RoundOpening, first: RoundOpening, admission: str = ROUND0) -> str | None:
    """Why `opening` doesn't show `entry` as a leaf under the root of the round that admitted it, a root the round coins'
    `commitment` shows recorded before that round's coin, with `first` (its first slot, opened under that round's slot
    table) naming it, and admitted in round 0 unless `admission` is `PER_ROUND`; None when it does. Alone it says nothing
    of a round the entry isn't under: `slot_reason` refuses an entry under a later round than a step that serves it, and
    `rounds_reason` holds the whole log to the coins."""
    r, i = opening.round, entry.index
    if admission not in ADMISSIONS:
        raise ValueError(f"an admission is one of {ADMISSIONS}, not {admission!r}")
    if not any(x.get("round") == r for x in log):
        return f"request {i}: the round log has no round {r}, so its entry is under no round root"
    if entry.round != r:
        return f"request {i} was admitted in round {entry.round}, and its entry is opened under round {r}'s root"
    if admission == ROUND0 and r != 0:
        return f"request {i} was {late_admission(r)}"
    if opening.value[:-HM96_SALT] != entry.row():
        return f"request {i}: the leaf opened under round {r} is not its entry"
    row, why = _round(log, commitment, salt, r)
    if why is not None:
        return f"request {i}: {why}"
    # round 0's request root folds its input-ids table's root into the table's (credit.requests_root, held by credit._arow)
    table = row.get("request_table", row["requests"]) if r == 0 else row["requests"]
    if not _leaf(requests_binding(salt, r), r, table, rows, opening):
        return f"request {i}'s entry doesn't open under round {r}'s request table"
    if (first.round, first.position) != (r, entry.slot):
        return f"request {i}'s first slot is slot {entry.slot} of step {r}, and slot {first.position} of step {first.round} " \
               "is opened"
    owner, why = _slot(row, salt, rows, first)
    if why is not None:
        return f"request {i}: {why}"
    if owner != i:
        return f"slot {entry.slot} of step {r} names {'no request' if owner is None else f'request {owner}'}, not request " \
               f"{i}, whose entry says it was first served there"
    return None


def slot_reason(log: Sequence[Mapping[str, Any]], commitment: LiveCommitment, salt: bytes, rows: int,
                calls: Sequence[Mapping[str, Any]], slot: RoundOpening, entry: RequestEntry | None = None,
                opening: RoundOpening | None = None, first: RoundOpening | None = None, admission: str = ROUND0
                ) -> str | None:
    """Why slot ``slot.position`` of step ``slot.round`` isn't what it must be, given that step's manifest rows `calls` (its
    served rows are linear 0's): a served slot names an admitted request, whose entry (`entry`, opened by `opening` and
    `first` as `request_reason` takes them under `admission`) is under the slot's round or an earlier one, and a slot past
    the served rows names none; None when it is."""
    s, p = slot.round, slot.position
    row, why = _round(log, commitment, salt, s)
    if why is not None:
        return why
    if step_digest(salt, s, calls) != bytes.fromhex(row["calls"]):
        return f"step {s}'s calls are not the ones under round {s}'s root"
    owner, why = _slot(row, salt, rows, slot)
    if why is not None:
        return why
    served = next((int(c["served"]) for c in calls if c.get("layer") == 0), 0)
    if p >= served:
        return None if owner is None else f"slot {p} of step {s} is past the step's {served} served rows and names request " \
                                          f"{owner}"
    if owner is None:
        return f"slot {p} of step {s} is a served row and names no admitted request: a request served without an entry"
    if entry is None or opening is None or first is None or entry.index != owner:
        return f"slot {p} of step {s} names request {owner}, and request {owner}'s entry isn't opened beside it"
    why = request_reason(log, commitment, salt, rows, entry, opening, first, admission)
    if why is not None:
        return why
    if entry.round > s:
        return f"request {owner} is served at slot {p} of step {s}, before round {entry.round}, the round its entry is under"
    return None


def rounds_reason(log: Sequence[Mapping[str, Any]], commitment: LiveCommitment) -> str | None:
    """Why the round log isn't the window's rounds in order, each round's coin the challenger's, opened once after its
    root (`window.rounds_reason`); None when it is."""
    if [x.get("round") for x in log] != list(range(len(log))):
        return "the round log is not rounds 0, 1, ... in order"
    if any(x.get("coin") is None for x in log):
        return "a round of the log has no live coin"
    try:
        openings = [LiveOpening.from_json(x["coin"]) for x in log]
        return Wd.rounds_reason(commitment, [bytes.fromhex(x["root"]) for x in log], openings)
    except CoinError as e:
        return str(e)


def new_requests(scheduler_output: Any) -> list[tuple[str, Sequence[int] | None, int]]:
    """The requests vLLM's scheduler admitted in this step (`SchedulerOutput.scheduled_new_reqs`, where Model Runner V2
    also puts requests resumed after a preemption), each (req_id, prompt token ids, token cap). A request with prompt
    embeddings has no prompt token ids here, and a pooling request no cap."""
    out = []
    for r in getattr(scheduler_output, "scheduled_new_reqs", None) or ():
        params = getattr(r, "sampling_params", None)
        prompt = None if getattr(r, "prompt_embeds", None) is not None else getattr(r, "prompt_token_ids", None)
        out.append((str(r.req_id), prompt, int(getattr(params, "max_tokens", 0) or 0)))
    return out


def sort_batch_req_ids(num_tokens_per_req: Mapping[str, int], draft_tokens: Mapping[str, Sequence[int]],
                       decode_query_len: int) -> list[str]:
    """Model Runner V2's batch order at vLLM d9105ea8 (``vllm/v1/worker/gpu/model_runner.py``, `sort_batch_req_ids`):
    verification and decode first, then short extends, then prefills, by rows; the runner's own is used when its module
    has one (`batch_order`)."""
    return sorted(num_tokens_per_req, key=lambda r: (not draft_tokens.get(r), num_tokens_per_req[r] != decode_query_len,
                                                     num_tokens_per_req[r]))


def batch_order(runner: Any, scheduler_output: Any) -> list[tuple[str, int]]:
    """The step's scheduled requests, each (req_id, rows), in the order Model Runner V2 lays their rows out in the step's
    inputs, and so in its linears' A tables (``gather_batch_req_state``, then ``prepare_inputs``' cumulative rows)."""
    tokens = dict(getattr(scheduler_output, "num_scheduled_tokens", None) or {})
    drafts = getattr(scheduler_output, "scheduled_spec_decode_tokens", None) or {}
    sort = getattr(sys.modules.get(type(runner).__module__), "sort_batch_req_ids", None) or sort_batch_req_ids
    return [(str(r), int(tokens[r])) for r in sort(tokens, drafts, runner.decode_query_len)]


__all__ = ["ADMISSIONS", "EMPTY_REQUEST", "EMPTY_SLOT", "HM96_SALT", "NO_RECEIPTS", "OPEN_REQUESTS_TAG", "PER_ROUND",
           "REQUESTS_NONCE", "ROUND0", "RequestEntry", "RoundOpening", "SLOTS_NONCE", "TABLES_NONCE", "TABLE_NONCES",
           "TABLE_TAGS", "UNENTERED", "batch_order", "draw_root", "first_ids", "hiding_rows", "late_admission",
           "new_requests", "receipts_digest", "request_reason", "request_table", "requests_binding", "round_reason", "round_root",
           "round_row", "rounds_reason", "slot_leaf", "slot_owner", "slot_reason", "slot_table", "slots_binding",
           "sort_batch_req_ids", "step_digest", "table_leaves", "tables_reason"]
