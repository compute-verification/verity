"""H1's requests and rounds (#2677's `Accounting.Served`, `cursor/int-tie-ea24` at 8f161917ab, not on main: a request entry is
an `act` value, a run input under the root of the round that admits it, committed before that round's coin; RT3-INT-9).
`pouw_h1.H1Run` builds them as it serves; this module holds what they are and how they are checked.

* **Rounds.** Round s is step s, one round a step. vLLM's scheduler admits a request in the step that first schedules it
  (`SchedulerOutput.scheduled_new_reqs`), and `pouw_h1.admitting`, the option's wrap of the model runner's
  ``execute_model``, hands those (`new_requests`) and the step's batch order (`batch_order`: Model Runner V2's own, each
  scheduled request's rows in the order its A tables hold them) to `pouw_h1.H1Run.admit` before the step's forward. Round s closes
  once step s's last linear is recorded: its root (`round_root`) is over the step's calls (`step_digest`, their manifest
  rows with what the listeners bound), its request table's root and its slot table's root, and only then does the round
  coins' `LiveRecord` (the pouw setting ``rounds``) open round s's coin. The salt is opened at install, before any
  admission, so no request entry precedes it.
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
  challenger's coins.
* **Not yet.** The slot table names A-table leaves only once the round root commits root(A): the step digest covers what
  the listeners bind, and no served run binds ``root_a`` yet (``details.requests.a_roots``). The round root has no admit
  root (H1 has no admit unit yet) and no receipt root (receipts aren't committed here), and no unit reads the round coins
  yet.
"""
from __future__ import annotations

import json
import struct
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.audit import HM96, HM96_SALT, ROW_SCHEMA, commit_rows, rows_domain
from verity.core.protocols.pouw.protocol import tagged
from verity.core.protocols.sampled_proofs.challenger import CoinError, LiveCommitment, LiveOpening, live_reason
from verity.ml.commitments import Commitment, Opening, VerificationLimits, verify_opening


@dataclass(frozen=True)
class RequestEntry:
    """An admitted request's entry (#2677's `Accounting.Served.Cls.act`, "a request entry its round admits"): its ordinal
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
ROUND_TAG = "verity/pouw/h1/round/v2"
STEP_TAG = "verity/pouw/h1/step-calls/v1"
REQUESTS_TAG = "verity/pouw/h1/requests/v2"
SLOTS_TAG = "verity/pouw/h1/slots/v1"
#: the credit refusal of a call in a window that served a row of a request with no entry (`pouw_h1.H1Run.request_gaps`)
UNENTERED = "credit: request-without-entry: the window served a row of a request with no entry (details.requests.gaps)"


def requests_binding(salt: bytes, r: int) -> bytes:
    return tagged(REQUESTS_TAG, salt, r)


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


def request_table(salt: bytes, r: int, rows: int, entries: Sequence[RequestEntry], key: bytes) -> Any:
    """Round r's request table: `rows` (T) hiding leaves, the entries the round admits in admission order, then
    `EMPTY_REQUEST`. Each admitted request schedules a token row in the step that admits it, so a step admits at most T."""
    if len(entries) > rows:
        raise ValueError(f"round {r} admits {len(entries)} requests, past its {rows} leaves")
    values = [e.row() for e in entries] + [EMPTY_REQUEST] * (rows - len(entries))
    return commit_rows(requests_binding(salt, r), r, values, hash=HM96, salt_key=key, nonce=REQUESTS_NONCE + r)


def slot_table(salt: bytes, r: int, rows: int, owners: Sequence[int | None], key: bytes) -> Any:
    """Round r's slot table: `rows` (T) hiding leaves, leaf i naming the request whose row is slot i of step r's A tables
    (`owners`, in the step's batch order), `EMPTY_SLOT` past them."""
    values = [slot_leaf(o) for o in owners[:rows]] + [EMPTY_SLOT] * (rows - min(len(owners), rows))
    return commit_rows(slots_binding(salt, r), r, values, hash=HM96, salt_key=key, nonce=SLOTS_NONCE + r)


def step_digest(salt: bytes, s: int, rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Step s's calls: SHA-512 over their manifest rows (`pouw_h1.H1Run.manifest`), in order, with what each call's listeners
    bound (root(A) among it, once AROW binds it)."""
    return tagged(STEP_TAG, salt, s, *(json.dumps(row, sort_keys=True) for row in rows))


def round_root(salt: bytes, r: int, calls: bytes, requests: bytes, slots: bytes) -> bytes:
    """Round r's root, over step r's calls (`step_digest`), the root of the request table of the entries it admits and the
    root of its slot table."""
    return tagged(ROUND_TAG, salt, r, calls, requests, slots)


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
    calls, requests, slots, root = (bytes.fromhex(row[k]) for k in ("calls", "requests", "slots", "root"))
    if round_root(salt, r, calls, requests, slots) != root:
        return None, f"round {r}'s root is not over its calls, its request table and its slot table"
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
                   entry: RequestEntry, opening: RoundOpening, first: RoundOpening) -> str | None:
    """Why `opening` doesn't show `entry` as a leaf under the root of the round that admitted it, a root the round coins'
    `commitment` shows recorded before that round's coin, with `first` (its first slot, opened under that round's slot
    table) naming it; None when it does. Alone it says nothing of a round the entry isn't under: `slot_reason` refuses an
    entry under a later round than a step that serves it, and `rounds_reason` holds the whole log to the coins."""
    r, i = opening.round, entry.index
    if not any(x.get("round") == r for x in log):
        return f"request {i}: the round log has no round {r}, so its entry is under no round root"
    if entry.round != r:
        return f"request {i} was admitted in round {entry.round}, and its entry is opened under round {r}'s root"
    if opening.value[:-HM96_SALT] != entry.row():
        return f"request {i}: the leaf opened under round {r} is not its entry"
    row, why = _round(log, commitment, salt, r)
    if why is not None:
        return f"request {i}: {why}"
    if not _leaf(requests_binding(salt, r), r, row["requests"], rows, opening):
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
                opening: RoundOpening | None = None, first: RoundOpening | None = None) -> str | None:
    """Why slot ``slot.position`` of step ``slot.round`` isn't what it must be, given that step's manifest rows `calls` (its
    served rows are linear 0's): a served slot names an admitted request, whose entry (`entry`, opened by `opening` and
    `first` as `request_reason` takes them) is under the slot's round or an earlier one, and a slot past the served rows
    names none; None when it is."""
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
    why = request_reason(log, commitment, salt, rows, entry, opening, first)
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


__all__ = ["EMPTY_REQUEST", "EMPTY_SLOT", "HM96_SALT", "REQUESTS_NONCE", "RequestEntry", "RoundOpening", "SLOTS_NONCE",
           "UNENTERED", "batch_order", "new_requests", "request_reason", "request_table", "requests_binding", "round_root",
           "rounds_reason", "slot_leaf", "slot_owner", "slot_reason", "slot_table", "slots_binding", "sort_batch_req_ids",
           "step_digest"]
