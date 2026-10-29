"""A served window's credited set, its drawn units and their outcome, its A rows, and the period it counts toward, as
PoUW's verifier reads them. `_window_outcomes` and `_period_outcomes` are the Python
reference of the Lean `PouwCheck.Credit` (`lean/check`), held to it by the difftest (`lean/check/tests/test_difftest.py`);
`window` and `period_credit` are they with each drawn unit's outcome from the verifier of record's own run, which the
Lean `PouwCheck.Credit.window` takes as given. Those outcomes are the Lean's (`PouwCheck.Verified`, `_lean_verified`):
the Lean decides which statements the verifier runs on, with which inputs, and what its answers credit; `_verified` is
its Python reference, held to it by the difftest (`lean/check/tests/test_verified_difftest.py`). Each round's live
outcome is the Lean's too (`PouwCheck.LiveVerified`, `_lean_live_verified`), with `_live_verified` its reference
(`lean/check/tests/test_live_verified_difftest.py`).

**`window` and `period_credit` are the only entries.** On one window's evidence and one round log, `window` runs every
step and credits only when all of them accept: `_credited`, `core_check.rounds`, `_a_link`, `layout_of`, its period's
`_weights`, `_arow`, the draw, `_verified`, `_outcome`'s rule, `_registration` and `_live`. `period_credit` runs `window` on each window
of a closed, complete period. A live Pearl-C pass is credited only through `pearl_c_period_credit`, its period holding
Pearl-C salt sessions alone. The steps are internal, so no caller credits a window with a step skipped: `_credited`
returns the credit bits whatever the proofs' outcome, and `_a_link` trusts the round log's roots, which only
`core_check.rounds` holds to the ones the challenger received before each coin.

* **The verdicts are credit's own** (`_verified`). A window's evidence carries its statements' proof files, one
  statement a call with a drawn unit (`STATEMENT_KEYS`: its circuit file, its public file and its sessions), never a
  verdict. Credit runs the verifier of record (`proofs.verify`, `flock-verify verify --zk`) on each: on the circuit file
  whose SHA-512 its period's open committed for the call's K (`CIRCUIT_FILES`, ``{k<K>: SHA-512}``, refused before the
  verifier runs when it differs), against the committed descriptor, its partition object (`partition_of`) and the
  call's ``--registered`` file credit builds from the descriptor and the window's entries (`registered_file`,
  `h1window.table_reads_file`'s, main's per-call tables). A rejected session refuses the window, and so does a
  verifier that can't run. The circuit file's tie to the descriptor's Definitions rests on the auditor's lowering,
  which built the committed file, until the verifier derives the circuit from the descriptor. The verdict
  over the statements is the Lean's (`PouwCheck.Verified.verified`, through `verifier.lean`): it names each statement
  to verify and the inputs (`_asks`: up to the first statement a check before the verifier refuses), this module only
  runs the verifier on them, and the Lean judges the answers (`_lean_verified`).
* **The seed's inputs are the verifier's** (`registered_file`'s ``pub``): each unit's public row, the salt, position
  and roots its seed and its receipt's fold keys are derived from, is a row of its call's public table
  ``<prefix>pub@<c>``, which credit builds itself from the window's Program salt, the unit's position and the call's
  own roots, root(``a/<K>@<s>``) and root(``b/<K>``) as the window's entries register them, and names at that known
  content. The window Program's one ``roots`` parameter is the layout's, not what a call's units are seeded under. A
  J10-R window's templates (`SHARED_J10R`) take two more, ``call`` and ``blocks``, its noise's call stream and its
  unit's place in it, and its public row holds those as credit derives them from the layout (`core_check.j10r_blocks`),
  every K's template in one form.
* **A unit's ok is the verifier's** (`registered_file`'s ``o1``): the statement's output row ``ok`` reads the window's
  ok rows ``<prefix>ok``, row u unit u's, the one word 1 under its public `core_check.ok_salt` (h1_prove's
  registration, #3124), which credit registers itself and names at that known content. A unit that computed 0 opens no
  row of it, and a session that opened ok at another root is refused.

* **The credited set is fixed before the draw** (`_credited`). Each row of the window's manifest (the serving process's
  call rows, in order, ``index`` 0, 1, …) carries its call's verdict as ``credit`` (true or false), and a credited row
  carries its z table's root (``tables.z``). `calls_root` hashes the rows (and the served trees' digest when the window
  has them) and the window's Commit record's digest (`commit_of`), and `draw_root` hashes that, the last round's
  receipts and every round's root and coin. The challenger's draw coin must have followed that one root, as its signed
  log shows, read through the session the auditor's ledger opens for the window, run from a commit the host's pins log
  records on main's first-parent line (`challenger.trusted_session_of`, over `challenger.pinned_session`), never a
  commitment the prover hands in. So a credit bit changed after the draw coin changes the draw root and is refused, and
  the credited set is the rows whose bit is true.
* **The drawn units are the draw coin's** (`drawn`): `core_check.draw`'s rule under the law ``work:k``
  (`core_check.work_units`), over the window's own units. The window's layout is its Program's,
  ``h1window.program(linears, calls, m, tile)``, and its units are in that Program's order: call by call, each call's
  tiles row block major. There is one stratum per template, that is per K (`strata`), each unit's work its tile's
  R·C·K served MACs, at the K its period's open committed (`law`). The served window's arguments are fixed (`window`):
  the stream is the window's root, the registration its draw root, and the receipt ``{"receipts": …}``, its last
  round's receipts.
* **The layout is the committed circuit's** (`layout_of`): `window` reads it from the descriptor whose SHA-512 the
  period's open committed, never from the window's evidence. Its program function's record (``TILES``, ``LINEARS``,
  ``CALLS``, ``M``) gives the calls, the linears and the slot, and K's template's parameters ``a_k<K>`` and ``b_k<K>``
  the tile's R and C. Each K of its ``TILES`` is one the tile's proof covers, a multiple of 32 below 2^17
  (`TILE_K_STEP`, `TILE_K_LIMIT`: the K its ``GemmAccumulatorS8`` binds), else the window is refused (``tile-k``,
  `tile_k`, its fix the K to pad to). Its root's nodes other than inputs must be, in order, each call's (M/R)·(n/C)
  instances of K's template, so the record names the units the circuit has. The cost is the descriptor's nodes and the
  record's calls, never the gates they unroll to.
* **K is committed, and ε is what it established** (`committed_k`, `established`): K is the proof service's selection,
  committed in the period-open (`K_FIELD`, in the auditor's signed ledger beside the period's circuit and cap) before
  any window's draw. A period whose open commits none is refused, and so is a window with fewer than K units, whose
  draw would not be K's. PoUW carries no ε target: `period_credit` reports each window's credited served work C and
  the ε its draw of K established over it at `DELTA`, `window.eps_at(K, δ, C, W, 1)` (`window.k_prime` inverted), W
  the window's work.
* **Credit counts served rows, not slots** (`served_rows`, `credited_work`): every call has the window's M slot rows,
  its step's m_s served rows first and empty rows after, and empty rows' units prove trivially, so no draw tells them
  from served ones. Each manifest row states its call's m_s (`SERVED_ROWS`, `pouw_h1.Call.summary`'s ``served``), under
  `calls_root`, which the draw root binds, so m_s is fixed before the draw coin. A row whose m_s is not a natural at
  most M is refused, and so is a step whose calls state different m_s (call c at step ``c // (C/S)``, as `_arow`
  places it). A credited call is m_s·n·K of credited work (`served_work`), and one with m_s = 0 is no credit; the
  draw's population stays the Program's units, padded slots included (`call_work`). Each step's m_s is proved (`_live`).
* **Each step's served rows are the live Program's** (`_live`, `live_of`): the window's evidence carries its live
  Program (`circuit.live.program_v2`, ``live``: its descriptor, and each round's live@<s> proof outcome, or in `window`
  its proof files, `_live_verified`, whose verdict is the Lean's, `_lean_live_verified`), whose record's T, S and
  PREFIX must be the period's committed T, the window's round count and its circuit's PREFIX, its root's units S calls
  of `PouwLiveSlots_v2{T}`. Every round is proved, never
  sampled: round s's one accepted proof, verified against that descriptor, opened its slot ports at round s's slot table
  ``<prefix>slots@<s>`` and its header ports at round 0's request table ``<prefix>requests@0`` at their registered roots
  (each registered once with T rows; `_a_link` holds the slot table to round s's root and `_arow` the request table to
  ``request_table``), its ``m`` at the known ``<prefix>m@<s>`` of the manifest's m_s, its ``s`` at the known
  ``<prefix>round@<s>`` of s, its caps ``cap_<i>`` at the known ``<prefix>caps`` of the max_tokens its period's open
  committed (`committed_length`) and its ok ``o0`` at the known ``<prefix>live-ok`` rows (`live_reads.reads_file`). So m_s
  is rule A's count of round s's slots assigned to round 0's admitted requests (`live.live_ok_v2`), the admitted requests
  are the fixed length's, each admitted in round 0 with its public max_tokens as its cap, so no request takes a decode row
  past its own max_tokens, and a step whose m_s is any other refuses the window. The live statements' circuit file is the
  period-open's ``circuit_files`` ``live``.
* **No partial credit** (`_outcome`): the window is accepted only when every unit the draw coin draws has exactly one
  proof outcome and it is an accept. A drawn unit whose proof failed or is missing refuses the whole window, not only
  its call. A unit with two outcomes is refused, since a retry never reproves, and so is an outcome for a unit the draw
  didn't select.
* **The round log is the challenger's** (`core_check.rounds`): the round roots `window` reads are the ones the
  challenger's signed log shows it received, in order, each round's coin opened once after its root. A log the
  challenger never received is refused, however its draw root and A trees agree with it.
* **A period counts only closed and complete** (`period`, RT4-INT-42): its windows are exactly those the auditor's
  ledger lists in the period's close, in that order, each opened and ended complete. A period with no close, or with
  a window missing, out of order or not its own, earns nothing, so no window is chosen after the fact. Under
  `period_credit` each window's session must have run main's code and the window must prove the circuit the period's
  open committed (`challenger.credited_session`, the pin first, then the circuit: the SHA-512 of the descriptor its
  proofs were verified against, the same bytes `layout_of` reads its layout from) and is drawn at the period's K, and
  a refused window refuses its period. Every path here that credits reads its session through
  `challenger.pinned_session`, against the host's pins log (``pins``, its lines) and the verifier's own copy of main's
  first-parent history (``main``, its ``git rev-list --first-parent origin/main``).
* **The A rows are the round roots'** (`_a_link`): each registered A value of the window, among its registered
  ``entries``, is a tree whose rows the round roots commit as aligned subtrees. A window's ``…/a/<K>``
  (`h1window.roles`) over L rows and S steps has rows ``[s·L/S, (s+1)·L/S)`` under round s's root. Round s's A table of
  K ``…/a/<K>@<s>`` (`h1window.table`: the M rows of the A block of each of its calls of K, call after call, which
  `h1window.table_reads` opens those calls' units at) is committed whole by round s and listed by no other round. Each
  round's ``a`` lists, for every A value it commits, its ``[level, index, root]`` subtrees in row order, and the root the
  challenger received for round s is `round_root` of the serving process's own round root (``inner``) and those
  segments. The verifier recomputes each value's root from the segments and its domain's pads. So the rows a unit reads
  are the ones fixed before its round's coin. Round s's input-ids table ``…/ids@<s>`` (slot i's input token id) is
  committed whole by round s, as its A tables are, and so is each of its calls' receipt table ``…/receipt@<c>`` (one
  row a unit of the call, its receipt, which a call's statement then reads as its output row o0, `registered_file`).
  The z and y records are not yet bound to the round roots.
* **The A rows and input ids are the served Commit's** (`_arow`, ROOTS1): the window's evidence carries the record of the
  Commit that ran in the window's engine (``commit``, one-stage's `verity/registration/v2` record, its canonical JSON in
  hex), whose digest `calls_root` binds before the draw coin, so it is fixed before any coin that selects a unit. Round
  s's A table of K ``…/a/<K>@<s>`` holds M rows for each of the round's calls of K (`step_tables`) and must have the
  root, domain and leaf count of the Commit's `commit_rows` value of its round and K, and round s's ``…/ids@<s>`` those
  of its `commit_ids` value: the Commit registers the window's own tables, so one tree is both. So the A rows a call's
  units read are its linear's input as the Commit commits it, and
  the input token ids under each round's root are its step's. Round 0's ``…/ids@0`` is also inside round 0's request
  table: the round-0 request root the window's open root binds is `requests_root` of the table's root and ids@0's, so
  the prompt tokens are fixed before the salt coin. Its domain binds the window's session in place of the open root,
  which contains it (`domain_of`). The window's request table ``<prefix>requests@0`` (`is_requests`, the live unit's
  ``req``), when it registers one, is round 0's: its root is ``request_table`` and its domain binds the session alike.
  That the Commit's values are the model's computation is the Commit's draw's, not credit's.
* **The window serves its period's fixed length** (`committed_length`, `_length`): correct
  tokens served past the honest run's end of sequence are not useful work, so a row of record serves each request's
  public max_tokens whatever it samples (ignore_eos, circuits' ``public-cap`` done source). The period's open commits
  them before any window (`LENGTH_FIELD`, ``{max_tokens, stop}``, stop `window.STOP_FIXED`), so they are the verifier's
  own; an open with none (``no-fixed-length``) or an EOS-keyed stop mode (``eos-keyed-row``) credits nothing. Every
  request is admitted in round 0, so round s ≥ 1 serves one decode row for each request whose max_tokens is past s,
  and each step's served rows m_s (`served_rows`, its calls' ``served``, fixed under `calls_root` before the draw coin)
  must be that in every round: a round serving more ran a request past its public max_tokens, and one serving fewer stopped a
  request early (``length-mismatch``). Each round's live statement proves its m_s (`_live`), so the rows credited are
  the public schedule's.
* **The proofs are the window's registration's** (`_registration`): each drawn unit's proof outcome is the verifier of
  record's verdict for its session (`PROOF_KEYS`): the SHA-512 of the descriptor it verified the session against
  (``program``) and the session's own registered finding (``registered``, flock-verify's: each registered port's value
  and the root its reads were opened against). The entries are the registration's: each A value's domain, and each
  other value's a proof opened, is `domain_of` the committed descriptor's SHA-512 and, for every value but the weights'
  ``…/b/<K>`` (held across windows, `is_held`) and round 0's input-ids and request tables, the window's open root
  (``window.root``, the root the auditor's ledger opens its session for, which a per-window value's domain binds);
  round 0's input-ids and request tables, whose roots that open root contains, bind the window's session
  instead. Each proof was
  verified against that
  descriptor, opened only values the window registers once, each at its registered root, and opened its unit's K's A
  value, the window's ``…/a/<K>`` or its own call's round's ``…/a/<K>@<s>``. So the A rows `_a_link` ties to the round roots
  are the ones each drawn unit's proof opened, under this window's registration: entries of another window or circuit,
  a proof verified against another registration, and an entry whose root is not the one its sessions opened are
  refused.
* **The weights are the period's, registered before the salt** (`_weights`, `committed_weights`): each K's
  ``<PREFIX>b/<K>``, all the B rows (int8 codes and their s_w) of the window's linears of that K, is registered once
  under the registered domain of its name and the window's circuit, binding no window, and
  `weights_root` of their roots is the weights' root the window's open root binds (`salt.window_open_root`, with its
  round-0 request table's root, round 0's ``requests`` in its round log, and the session the auditor's ledger opened for
  it), which its salt coin opens after. That root is the one its period's open signed (`WEIGHTS_FIELD`, the gate's
  ``--model``) before any window, so every window of a period opens the same B rows: a B row the prover changes
  after the salt, or between two windows of a period, its codes or its s_w, is refused, and so is a period whose open
  signs no weights' root. The open names the model the root is of (`MODEL_FIELD`, the auditor's registration of its
  int8 checkpoint), and its role and checkpoint are one `PINNED_MODELS` pins (``lean/check/models.json``), else the
  period counts nothing.

`calls_root` and `draw_root` are vllm's `pouw_h1.calls_root` and `pouw_h1_requests.draw_root`, byte for byte. Every
refusal is (code, why), the first rule the input breaks, its why naming the fix; the Lean port writes the same strings.
"""
from __future__ import annotations

import bisect
import functools
import hashlib
import itertools
import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

from verity.core.protocols.pouw import coins as Co
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import salt as PS
from verity.core.protocols.pouw import verifier as V
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.circuit import live_reads as LR
from verity.core.protocols.pouw.protocol import CALL_RANGE, tagged
from verity.core.protocols.pouw.weights import CHUNK_ALIGN, CHUNK_MAX, chunks, leaf_words, weights_root
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.sampled_proofs.one_stage import proofs as PF
from verity.core.protocols.sampled_proofs.one_stage import registered as G

#: the verifier of record over a statement's proof files (`proofs.verify`)
Run = Callable[..., tuple[list | None, tuple[str, str] | None]]

CALLS_TAG = b"verity/pouw/h1/calls/v2\0"
DRAW_TAG = "verity/pouw/h1/draw/v1"
ROUND_TAG = "verity/pouw/h1/round-a/v1"
#: each manifest row's credit bit
CREDIT = "credit"
#: each manifest row's served rows m_s: its step's rows in slots 0 .. m_s - 1 of its call's M, the rest empty
SERVED_ROWS = "served"
#: how deep a manifest row nests
DEPTH = 64
VERDICTS = ("accept", "reject")
#: a segment's level is below this
LEVELS = 64
#: a row's words, and the rows of a value the round roots commit whose rows the layout doesn't fix (an input-ids
#: table): the verifier hashes each pad past them, at most as many again. Every A value's and receipt table's rows are
#: the layout's own (`table_rows`), every weights' value's too (`_weights`), and no entry's leaves are bounded otherwise
MAX_ROWS = 1 << 24
_DIGEST = re.compile(r"[0-9a-f]{128}")
#: a contraction K in a registered value's name: a canonical decimal, no leading zero
_DIGITS = re.compile(r"[1-9][0-9]*")
_A = re.compile(r"([1-9][0-9]*)(?:@([0-9]+))?")
_HEX = re.compile(r"(?:[0-9a-f]{2})+")
_FRAME = b"veritor/protocol/merkle/frame/v3\0"

SALT_FORM = "the window's Program salt is not a SHA-512 digest (128 lowercase hex)"
CALLS_FORM = "the window's manifest is not a list of call rows"
SERVED_FORM = "the served trees' digest is neither null nor a SHA-512 digest (128 lowercase hex)"
DRAW_FORM = "the window's draw is not an object whose receipts are a SHA-512 digest"
NO_DRAW = ("the challenger's session closed with no draw coin opened: the window opens its draw coin once, after its "
           "draw root")
DRAW_ROOT = ("the window's draw root is not over its manifest (each call's credit bit among it), its Commit record, its "
             "last round's receipts and every round's root and coin")
CHANGED = ("the draw root over the manifest (each call's credit bit among it), its Commit record, its last round's "
           "receipts and every round's root and coin is not the one root the challenger's log shows its draw coin "
           "followed: a credit bit, a call or the Commit record changed after the draw coin, or the draw opened after "
           "another root. Commit every call's credit bit in its manifest row, and the Commit record, before the draw "
           "root, and never change them after the draw coin")
#: the period-open's field holding the draw's K, the proof service's selection, committed before any draw
K_FIELD = "draw_k"
#: the most a committed K may be
MAX_K = 1 << 24
#: the period-open's field holding the served tables' leaf count T (the pouw setting ``rows``: each round's request and
#: slot tables' leaves), committed before any draw; a window's A slot M is `h1window.slot` of it
ROWS_FIELD = "rows"
#: the δ the period's ε is reported at
DELTA = Fraction(1, 1 << 40)
#: the period-open's field holding the weights' root (`weights_root` of each K's registered ``b/<K>``, the gate's
#: ``--model``), signed before any window's salt
WEIGHTS_FIELD = "weights_root"
#: the period-open's field naming the model whose weights its root is of (the gate's ``--model``, from the auditor's
#: model registry): ``{name, checkpoint_sha512}``, its int8 checkpoint's role and that checkpoint's SHA-512
MODEL_FIELD = "model"
MODEL_KEYS = ("checkpoint_sha512", "name")
#: each checkpoint role a period's open may name, with its checkpoint's SHA-512 (``checkpoint_sha512``, of the canonical
#: JSON of each of its files' SHA-512); the Lean embeds the same file. Credit counts only an int8 role's weights: the
#: gate signs a weights' root only from an H1 registration, so an open naming another role (the remote memory audit's
#: BF16 checkpoint) signs none and its period counts nothing
MODELS = Path(__file__).parent / "lean/check/models.json"
PINNED_MODELS = {role: e["checkpoint_sha512"] for role, e in json.loads(MODELS.read_text()).items()}
MODEL_FIX = ("open the period with the gate's --model, which signs the auditor's registration of a checkpoint credit "
             "pins (models.json)")
#: the period-open's field holding the commit it opened under (40 lowercase hex on main's first-parent line): every
#: window of the period must be pinned within the freshness floor (`STALE_FIELD`) of it, counted in first-parent commits
#: from this pin and never from main's head (`committed_pin`, `_pin_hygiene`, CMP74-FOLLOW). Optional: a period-open that
#: records none isn't held to it
PERIOD_PIN_FIELD = "pinned"
#: the period-open's field holding the freshness floor (a nonnegative integer): a window counts only when its pin is at
#: most this many first-parent commits behind the period's open pin on main's line (`_pin_hygiene`, measured from that
#: pin, never from main's head, so main moving on never refuses a window fresh at its open, CMP74-FOLLOW). Optional,
#: default 0 when the open pins but records no floor; infra sets it generously, about a day of first-parent commits
STALE_FIELD = "pin_floor"
#: the period-open's field holding the rows of record's fixed length: ``{max_tokens, stop}``,
#: each request's public max_tokens in admission order and the stop mode `window.STOP_FIXED` (ignore_eos, circuits'
#: ``public-cap`` done source), signed before any window, so it is the verifier's own and never the prover's
LENGTH_FIELD = "length"
LENGTH_KEYS = ("max_tokens", "stop")
K_FORM = f"the draw's K is not a positive integer at most 2^24 (the K its period's open committed as {K_FIELD})"
#: a window's layout: `circuit.h1window.program`'s arguments
LAYOUT_KEYS = ("calls", "linears", "m", "tile")
LAYOUT_FORM = ("the window's layout is not {calls, linears, m, tile}, h1window.program's arguments: its calls "
               "[[linear, A block], ...] in order, its linears [[name, K, n], ...] each once, its A slot m and its tile "
               "[R, C], every number a natural number and R, C at least one")
COIN_FORM = "the draw coin is not a SHA-512 digest (128 lowercase hex)"
DRAW_ARGS = ("the draw's stream is not a string, its registration not a SHA-512 digest, or its receipt not an object of "
             "printable ASCII strings")
#: a window's evidence, as `window` reads it, its descriptor the circuit's its proofs were verified against, its
#: entries the values it registered and its statements each call's proof files
WINDOW_KEYS = ("calls", "commit", "descriptor", "draw", "entries", "live", "rounds", "salt", "served", "session",
               "statements", "window")
#: a window's evidence in a period
PERIOD_KEYS = WINDOW_KEYS
PERIOD_FORM = ("the period's windows are not a list of each window's evidence {" + ", ".join(PERIOD_KEYS) + "}, "
               "in its close's order: each window carries its statements' proof files, never verdicts")
#: a window's evidence with each drawn unit's proof outcome in place of its statements (`_window_outcomes`), the Lean's
PROVED_KEYS = ("calls", "commit", "descriptor", "draw", "entries", "live", "proofs", "rounds", "salt", "served",
               "session", "window")
WINDOWS_FORM = ("the period's windows are not a list of each window's evidence {" + ", ".join(PROVED_KEYS) + "}, "
                "in its close's order")
#: a call's statement: its index, the paths of its circuit file and public file, and its sessions' directories
STATEMENT_KEYS = ("call", "circuit", "public", "sessions")
STATEMENTS_FORM = ("the window's statements are not a list of {call, circuit, public, sessions}, each a call of the "
                   "window, the paths of its circuit file and its public file, and its sessions' directories (at least "
                   "one): the window carries each call's proof files, and credit verifies them itself, never a verdict")
#: the period-open's field holding each K's template's circuit file's SHA-512 and the live Program's template's,
#: ``{k<K>: SHA-512, live: SHA-512}``
CIRCUIT_FILES = "circuit_files"
FILES_FORM = (f"the committed circuit files are not {{k<K>: SHA-512, live: SHA-512}} (the period-open's "
              f"{CIRCUIT_FILES}, each digest 128 lowercase hex)")
_TEMPLATE = re.compile(r"k[1-9][0-9]*|live")
DESCRIPTOR_HEX = ("the window's descriptor is not the bytes of the circuit its proofs were verified against in "
                  "lowercase hex (its canonical JSON without annotations, the bytes program_digest hashes)")
#: the descriptor schema `layout_of` reads
SCHEMA = "verity-ir/descriptor/v1"
#: h1window's Program function, `h1window.PouwH1Window`, and its record's bindings in order
WINDOW_FN = "PouwH1Window_v3"
RECORD = ("TILES", "LINEARS", "CALLS", "M", "PREFIX")
#: K's template, `h1window.template`, and its parameters after its A and B rows (`h1window.SHARED`)
TILE_FN = "PouwH1WindowTile_v1"
SHARED = ("zy", "salt", "position", "roots")
#: a J10-R template's (`h1window.SHARED_J10R`): its call's first position and its unit's blocks after them
SHARED_J10R = (*SHARED, "call", "blocks")
#: a window's tile is at a K its proof covers, a multiple of `TILE_K_STEP` below `TILE_K_LIMIT`: the K the tile's
#: `GemmAccumulatorS8` binds (its 32-wide step, and `S8_K_LIMIT`, where an s8 sum can leave s32)
TILE_K_STEP = 32
TILE_K_LIMIT = 1 << 17
_INPUT = re.compile(r"(?:Registered)?Input[1-9][0-9]*_v1")
DESCRIPTOR_FORM = ("the window's descriptor is not a verity-ir/descriptor/v1 descriptor in canonical JSON (keys sorted, "
                   "compact, ASCII: the bytes program_digest hashes) with its callees, definitions, program function and "
                   "root")
NOT_WINDOW = ("the window's circuit is not an H1 window: its program function is not PouwH1Window_v3{TILES=…,LINEARS=…,"
              "CALLS=…,M=…,PREFIX=…}, each binding canonical JSON, its TILES [[K, {\"fn\": template}], ...] by "
              "ascending K and its PREFIX a string (h1window.program's Program)")
BODY = ("the window's circuit's units are not its record's tiles: its root's nodes other than its inputs are not, in "
        "order, each of its CALLS' (M/R)·(n/C) instances of K's template, each node a call of it or a batch of at least "
        "one (h1window.program's body)")
#: a drawn unit's proof outcome: the verifier of record's verdict for the session that proved it
PROOF_KEYS = ("program", "registered", "unit", "verdict")
PROOFS_FORM = ("the drawn units' proof outcomes are not a list of {program, registered, unit, verdict}, each the verifier "
               "of record's verdict for a drawn unit's session: verdict accept or reject, program the SHA-512 of the "
               "circuit's descriptor it verified the session against, and registered its finding, each registered port's "
               "{value, root}, port and value printable ASCII and root a SHA-512 digest")
LOG_FORM = "the round log is not a nonempty list"
ENTRIES_FORM = "the window's registered entries are not a list of objects"
NO_A = ("the window registers no A value (…/a/<K>, or a round's …/a/<K>@<s>), so nothing ties its units' A rows to "
        "the round roots: register the window's A values")
COMMIT_TAG = b"verity/pouw/h1/commit/v1\0"
#: the served Commit's record: one-stage's registration record with registered values
COMMIT_FORMAT = "verity/registration/v2"
COMMIT_FORM = ("the window's Commit record is not the bytes in lowercase hex of a verity/registration/v2 record in "
               "canonical JSON whose registered values are a list of objects: the served Commit runs in the window's "
               "engine, and its record is bound under the window's calls_root before the draw coin")
_IDS = re.compile(r"ids@(0|[1-9][0-9]*)")
_SLOTS = re.compile(r"slots@(0|[1-9][0-9]*)")
_RECEIPT = re.compile(r"receipt@(0|[1-9][0-9]*)")
#: a tile unit's receipt, its return's field (`h1window.OUTPUTS`), and the statement's output row of it
#: (`h1window.STATEMENT_OUT_ROWS`), which reads call c's receipt table when the window registers one
RECEIPT_FIELD, RECEIPT_PORT = "receipt", "o0"
_PUB = re.compile(r"pub@(0|[1-9][0-9]*)")
#: the statement's public row (`h1window.PUB_PORT`), which reads call c's public table, the verifier's own
PUB_PORT = "pub"
#: the statement's output row of its unit's ok (`h1window.STATEMENT_OUT_ROWS`), and the window Program's output it is
#: (`h1window.OUTPUTS`, after the Program's prefix): the window's ok rows, row u unit u's, the verifier's own
OK_PORT, OK_VALUE = "o1", "ok"
#: round 0's request root (``requests``, the one the window's open root binds): its request table's root with round 0's
#: input-ids table's root inside (`requests_root`)
REQUESTS_IDS_TAG = "verity/pouw/h1/requests/ids/v1"
IDS_OUTSIDE = ("round 0's request root (requests, the one the window's open root binds) is not requests_root of its "
               "request table's root (request_table) and its input-ids table's root, so round 0's input ids are not "
               "fixed before the salt: commit round 0's input-ids table inside round 0's request table, under the "
               "window's open root")
#: round 0's request table's name's last part (`circuit.live`'s ``<prefix>requests@0``, the live unit's ``req`` port)
REQUESTS_TABLE = "requests@0"
REQUESTS_OUTSIDE = ("the window's request table's root is not round 0's request_table, the one round 0's request root "
                    "(requests) binds under the window's open root, so its rows are not the requests admitted before "
                    "the salt: register round 0's request table at its request_table root, once")
#: the window evidence's live Program (`circuit.live`), and the period-open's circuit file of its template
#: (`CIRCUIT_FILES`' ``live``)
LIVE = "live"
#: the live Program's function and its record's bindings in order, and its one template
LIVE_FN = "PouwLiveWindow_v2"
LIVE_RECORD = ("T", "S", "PREFIX")
LIVE_UNIT = "PouwLiveSlots_v2"
#: the live evidence with each round's proof outcome (`_window_outcomes`, the Lean's), and with its statements (`window`)
LIVE_KEYS = ("descriptor", "proofs")
LIVE_STATEMENTS = ("descriptor", "statements")
#: a round's live statement: its round, the paths of its circuit file and public file, and its sessions' directories
LIVE_STATEMENT_KEYS = ("circuit", "public", "round", "sessions")
LIVE_FORM = ("the window's live evidence is not {descriptor, proofs}: its live Program's descriptor in lowercase hex "
             "(circuit.live.program_v2 of its tables' T, its S rounds and its PREFIX, canonical JSON without annotations) "
             "and each round's live@<s> proof outcome, unit s round s's")
LIVE_STATEMENTS_FORM = ("the window's live evidence is not {descriptor, statements}: its live Program's descriptor in "
                        "lowercase hex and a list of {circuit, public, round, sessions}, each round's live@<s> proof "
                        "files: the paths of its circuit file and its public file, and its sessions' directories (at "
                        "least one)")
LIVE_DESCRIPTOR_FORM = ("the window's live descriptor is not a verity-ir/descriptor/v1 descriptor in canonical JSON "
                        "with its callees, definitions, program function and root")
NOT_LIVE = ("the window's live circuit is not a live Program: its program function is not PouwLiveWindow_v2{T=…,S=…,"
            "PREFIX=…}, each binding canonical JSON, T and S positive integers and PREFIX a string "
            "(circuit.live.program_v2's Program, whose units bind each request's cap to its public max_tokens)")
LIVE_BODY = ("the window's live circuit's units are not its rounds: its root's nodes other than its inputs are not S "
             "calls of PouwLiveSlots_v2{T=…}, a composite of that id bound to its record's T alone, S its record's "
             "(circuit.live.program_v2's body)")
LIVE_PROOFS_FORM = ("the window's live proof outcomes are not a list of {program, registered, unit, verdict}, each the "
                    "verifier of record's verdict for round s's live@<s> session, unit s: verdict accept or reject, "
                    "program the SHA-512 of the live descriptor, registered its finding")
LIVE_PREFIX = ("the window's live Program's PREFIX is not its circuit's, so the tables its units read are not the "
               "window's: build the live Program under the window Program's PREFIX")


def commit_rows(s: int, k: int) -> str:
    """The Commit's name for the A rows (its AROW) of the window's round `s`'s calls of contraction `k`: what it registers
    as that round's A table of K ``<prefix>a/<K>@<s>`` (`step_tables`)."""
    return f"row-table/{s}/k{k}"


def step_tables(layout: dict, steps: int) -> list[tuple[int, int, list[int]]]:
    """Each round's A tables of a layout `strata` accepted, over its `steps` rounds (its calls the rounds' equal blocks,
    in order): ``(s, K, calls)`` for each K round s's calls have, in order of its first call there, `calls` the round's
    calls of K in call order. Round s's A table of K ``<prefix>a/<K>@<s>`` holds the M rows of each of their A blocks,
    call after call, so the j-th of them reads its rows from row M·j (`step_row`). Empty when `steps` doesn't divide
    the calls."""
    calls, ks = layout["calls"], {name: k for name, k, _n in layout["linears"]}
    if steps < 1 or len(calls) % steps:
        return []
    per, out = len(calls) // steps, []
    for s in range(steps):
        mine: dict[int, list[int]] = {}
        for c in range(s * per, (s + 1) * per):
            mine.setdefault(ks[calls[c][0]], []).append(c)
        out += [(s, k, cs) for k, cs in mine.items()]
    return out


def step_row(layout: dict, steps: int, call: int) -> tuple[int, int, int]:
    """Call `call`'s round of the layout's `steps`, its round's calls of its K, and its first row in its round's A table
    of K (`step_tables`): ``(s, n, M·j)``."""
    s, n, j = next(((s, len(cs), cs.index(call)) for s, _k, cs in step_tables(layout, steps) if call in cs), (0, 0, 0))
    return s, n, layout["m"] * j


def table_rows(layout: dict, steps: int, prefix: str) -> dict[str, int]:
    """The rows the layout `layout` (one `layout_of` accepted) gives each value the round roots commit but the
    input-ids tables, by name under `prefix` (the descriptor's ``PREFIX``), over its `steps` rounds: the window's
    ``a/<K>`` x·blocks·M (its blocks of K, x = `weights.chunks` (K) leaves a row, `h1window`'s), round s's A table of K
    ``a/<K>@<s>`` x·M·|round s's calls of K| (`step_tables`), and call c's receipt table ``receipt@<c>`` c's units
    (`call_work`). Empty when `steps` doesn't divide the calls, a window `_arow` refuses. `_a_link` holds each such
    entry to these rows before it hashes any pad."""
    calls = layout["calls"]
    if not (type(steps) is int and steps >= 1 and len(calls) % steps == 0):
        return {}
    ks, m = {name: k for name, k, _n in layout["linears"]}, layout["m"]
    out = {f"{prefix}a/{k}": chunks(k) * m * (1 + max(blk for lin, blk in calls if ks[lin] == k))
           for k in sorted(set(ks.values()))}
    out |= {f"{prefix}a/{k}@{s}": chunks(k) * m * len(cs) for s, k, cs in step_tables(layout, steps)}
    return out | {f"{prefix}receipt@{c}": size for c, (size, _work) in enumerate(call_work(layout))}


def requests_root(table: bytes, ids: bytes) -> bytes:
    """Round 0's request root, the one the window's open root binds (`salt.window_open_root`): `tagged` of
    `REQUESTS_IDS_TAG`, the root of round 0's request table `table` and the root of its input-ids table ``…/ids@0``
    `ids`. So the window's open root contains ids@0's root, and its rows are fixed before the salt coin."""
    return tagged(REQUESTS_IDS_TAG, table, ids)


def commit_ids(s: int) -> str:
    """The Commit's name for its step's input token ids at the window's round `s`: what it registers as round s's
    input-ids table ``<prefix>ids@<s>``."""
    return f"input-ids/{s}"


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


def calls_root(salt: bytes, rows: Any, served: bytes | None = None, commit: bytes | None = None) -> bytes:
    """SHA-512 over the manifest rows `rows` in order, each as ``json.dumps(row, sort_keys=True)`` and a newline, under
    `CALLS_TAG` and the window's Program salt; with `served`, SHA-512 of that and the served trees' digest; with
    `commit`, SHA-512 under `COMMIT_TAG` of that and the Commit record's digest."""
    h = hashlib.sha512(CALLS_TAG + salt)
    for row in rows:
        h.update(json.dumps(row, sort_keys=True).encode() + b"\n")
    d = h.digest() if served is None else hashlib.sha512(h.digest() + served).digest()
    return d if commit is None else hashlib.sha512(COMMIT_TAG + d + commit).digest()


def commit_of(commit: Any) -> tuple[tuple[bytes, list[dict]] | None, tuple[str, str] | None]:
    """The digest of the window's Commit record `commit` (the bytes in lowercase hex of its canonical JSON, the bytes
    `registration.digest` hashes under its format) and its registered values, and None; or None and the refusal
    (``commit``)."""
    try:
        ok, rec = (_loads(bytes.fromhex(commit).decode()) if type(commit) is str and _HEX.fullmatch(commit)
                   else (False, None))
    except UnicodeDecodeError:
        ok, rec = False, None
    if not (ok and isinstance(rec, dict) and rec.get("format") == COMMIT_FORMAT and type(rec.get("registered")) is list
            and all(isinstance(e, dict) for e in rec["registered"])):
        return None, ("commit", COMMIT_FORM)
    d = hashlib.sha512(COMMIT_FORMAT.encode() + b"\0" + bytes.fromhex(commit)).digest()
    return (d, rec["registered"]), None


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


def no_served(i: int, t: int) -> str:
    return (f"manifest row {i} states no served rows m_s at most the period's T = {t} table rows (`{SERVED_ROWS}`, a "
            "natural number): the serving process commits each call's served rows in its manifest row, under "
            "calls_root, before the draw coin, and a step serves at most its tables' T rows")


def no_rows(name: str) -> str:
    return (f"period {name}'s open commits no table rows T ({ROWS_FIELD}, a positive integer: each round's request and "
            "slot tables' leaves), so no window's served rows m_s are bounded by its tables: the proof service commits T "
            "in the period-open, beside its draw K, before any draw")


ROWS_FORM = (f"the period's table rows T are not a positive integer (the T its open committed as {ROWS_FIELD}: each "
             "round's request and slot tables' leaves)")


def other_slot(t: int, r: int, m: int) -> str:
    return (f"the period's T = {t} table rows give an A slot of {max(r, 1 << (t - 1).bit_length())} rows, not the "
            f"window's M = {m}: a window's slot is h1window.slot of its tables' T rows (the least power of two at or "
            f"above T, at least the tile's R = {r}), so its slot rows past T are no table's")


def other_served(i: int, first: int, s: int, got: int, want: int) -> str:
    return (f"call {i} states {got} served rows and call {first} {want}, both of step {s}: every dense call of a step "
            "serves the step's m_s rows, so its calls state one m_s")


def _credited(key: Any, ledger: Any, pins: Any, main: Any, window: Any, session: Any, salt: Any, calls: Any, served: Any,
              commit: Any, rounds: Any, draw: Any, verify: Ch.Verify) -> tuple[list[int] | None, tuple[str, str] | None]:
    """The credited calls (their indices) of a window whose manifest `calls` (its rows, `served` the served trees'
    digest or None, under the window's Program salt `salt`, and its Commit record `commit`, `commit_of`'s) fixed every
    call's credit bit and the Commit record before the draw coin, and None; or None and the refusal (``coin``,
    ``manifest``, ``credit``, ``commit``, ``rounds`` or ``draw``, and why). The draw coin is read through the session the
    auditor's signed ledger `ledger` opens for `window` under the pinned challenger key `key`, run from a commit the
    host's pins log `pins` records and the verifier's copy of main's first-parent history `main` holds, from that
    session's files `session` (`challenger.trusted_session_of`, its signatures checked with `verify`): its log must show
    `draw_root` over the manifest and the Commit record's digest (`calls_root`), ``draw``'s ``receipts`` and the round
    log `rounds` (each round's ``root`` and ``coin``) as the one root the draw coin followed, ``draw``'s ``root`` must be
    it, and ``draw``'s ``coin`` must be the challenger's draw coin 0 opened after it. A step of `window`: its indices are
    no credit until every other step accepts."""
    try:
        k = Ch.ChallengerKey.from_json(key)
    except Ch.CoinError as e:
        return None, ("coin", f"the verifier's challenger key is malformed: {e}")
    try:
        c, received = Ch.trusted_session_of(ledger, window, k, pins, main, session, verify)
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
    got, why = commit_of(commit)
    if why is not None:
        return None, why
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
    root = tagged(DRAW_TAG, calls_root(s, calls, _digest(served), got[0]), receipts, *parts)
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


def _canonical(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def _loads(text: str) -> tuple[bool, Any]:
    """`text` read as JSON, and whether it is that value's canonical JSON (`_canonical`, Python's `canonical_json`), so
    every reading of it, the Lean's among them, is the same."""
    try:
        x = json.loads(text, parse_constant=_no_constant)
        return _canonical(x) == text, x
    except (ValueError, RecursionError):
        return False, None


def _record(fid: str) -> list[tuple[str, str]]:
    """`codec._split_record` (`Flock.Program.recordPairs`): the ``k=<json>`` pairs of a specialization id
    ``Name_vN{k1=..,k2=..}`` in id order, commas inside strings and brackets skipped; none for an id without a record."""
    if "{" not in fid or not fid.endswith("}"):
        return []
    rec = fid[fid.index("{") + 1:-1]
    pairs: list[tuple[str, str]] = []
    depth, in_str, esc, start = 0, False, False, 0
    for pos, ch in enumerate(rec + ","):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        elif ch == "," and depth == 0:
            k, _, v = rec[start:pos].partition("=")
            pairs.append((k, v))
            start = pos + 1
    return pairs


def no_template(k: int) -> str:
    return (f"the window's template at K = {k} is not h1window's: a composite {TILE_FN} bound to its UNIT and K = {k}, "
            f"its id {TILE_FN}{{UNIT=…,K={k}}}, whose parameters are a_k{k} and b_k{k}, the tile's R and C rows of K's n "
            f"chunk rows each (n = weights.chunks(K), each row 8·K/n + 32 bits: K/n codes and the row's scale, h1zy.ROW), "
            f"then {', '.join(SHARED)} (with J10-R's noise, then {', '.join(SHARED_J10R[len(SHARED):])}), at the same "
            "R × C and in the same form at every K")


def _chunks(k: int) -> int | None:
    """`weights.chunks` of `k`, or None for a K it refuses."""
    try:
        return chunks(k)
    except ValueError:
        return None


def tile_k(k: int) -> str:
    """Why `layout_of` refuses the window's tile at a K = `k` of at least one that its proof doesn't cover
    (``tile-k``), with its fix: the K to pad its linears' rows to, or past `TILE_K_LIMIT` the split."""
    up = (k + TILE_K_STEP - 1) // TILE_K_STEP * TILE_K_STEP
    fix = (f"pad its linears' A and B rows with zero codes to K = {up}" if up < TILE_K_LIMIT else
           f"split its linears' k range into linears of at most K = {TILE_K_LIMIT - TILE_K_STEP} each, each padded "
           f"with zero codes to a multiple of {TILE_K_STEP}")
    return (f"the window's tile at K = {k} is outside the K its proof covers, multiples of {TILE_K_STEP} below "
            f"{TILE_K_LIMIT} (the K its GemmAccumulatorS8 binds): {fix}")


def _tile(dj: Any, tid: str, k: int) -> tuple[int, int, bool] | None:
    """The R × C of K = `k`'s template `dj` (its definition, under the id `tid`) and whether it is J10-R's (its
    parameters end in `SHARED_J10R`), or None when it is not `h1window.template`'s: its A and B rows are n·R and n·C chunk
    rows of 8·K/n + 32 bits each, n = `weights.chunks` (K)."""
    if not (isinstance(dj, dict) and dj.get("form") == "composite" and dj.get("id") == TILE_FN):
        return None
    b, body = dj.get("bindings"), dj.get("body")
    if not (isinstance(b, dict) and sorted(b) == ["K", "UNIT"] and type(b["K"]) is int and b["K"] == k
            and tid == f"{TILE_FN}{{UNIT={_canonical(b['UNIT'])},K={k}}}" and isinstance(body, dict)):
        return None
    ps = body.get("params")
    if not (type(ps) is list and all(type(p) is list and len(p) == 2 for p in ps)
            and [p[0] for p in ps] in ([f"a_k{k}", f"b_k{k}", *SHARED], [f"a_k{k}", f"b_k{k}", *SHARED_J10R])):
        return None
    rows, n = [p[1] for p in ps[:2]], _chunks(k)
    if n is None or not all(type(t) is list and len(t) == 3 and t[0] == "Array" and _nat(t[1]) and t[1] >= 1
                            and t[1] % n == 0 and _bits(t[2]) == 8 * (k // n) + 32 for t in rows):
        return None
    return rows[0][1] // n, rows[1][1] // n, len(ps) == 2 + len(SHARED_J10R)


def _runs(xs: Any) -> list[list]:
    """`(template, count)` pairs with adjacent ones of one template merged."""
    out: list[list] = []
    for t, n in xs:
        if out and out[-1][0] == t:
            out[-1][1] += n
        else:
            out.append([t, n])
    return out


def layout_of(descriptor: Any) -> tuple[dict | None, tuple[str, str] | None]:
    """The layout (``{calls, linears, m, tile}``, `strata`'s) of the H1 window whose circuit's descriptor is
    `descriptor` (its bytes, canonical JSON without annotations, the bytes the period's ``circuit`` is the SHA-512 of),
    and None; or None and the refusal (``layout`` or ``tile-k``, and why). The calls, linears and slot are its program
    function's record (`WINDOW_FN`'s ``CALLS``, ``LINEARS`` and ``M``), each K's template (``TILES``) is
    `h1window.template`'s, whose A and B rows give R and C, at a K its tile's proof covers (else ``tile-k``, `tile_k`: a
    multiple of `TILE_K_STEP` below `TILE_K_LIMIT`, checked before K's template), and the root's nodes other than inputs
    are each call's (M/R)·(n/C) instances of its K's template, in the calls' order: a call one instance and a batch its
    members, as the template query counts units (`Q_template_instances`). Its cost is the descriptor's bytes, never the
    gates its nodes unroll to. The Lean reads the descriptor through C-Flock's decoder (`Credit.programOf`), so it also
    refuses (`DESCRIPTOR_FORM`) a descriptor that decoder refuses; this checks only its form (`_parsed`)."""
    def no(why: str) -> tuple[None, tuple[str, str]]:
        return None, ("layout", why)

    d = _parsed(descriptor)
    if d is None:
        return no(DESCRIPTOR_FORM)
    callees, defs, fn = d["callees"], d["definitions"], d["program_fn"]
    rec = _record(fn) if fn.startswith(WINDOW_FN + "{") else []
    if [k for k, _v in rec] != list(RECORD):
        return no(NOT_WINDOW)
    vals = {}
    for k, s in rec:
        ok, vals[k] = _loads(s)
        if not ok:
            return no(NOT_WINDOW)
    tiles = vals["TILES"]
    if not (type(tiles) is list and tiles and type(vals["PREFIX"]) is str
            and all(type(t) is list and len(t) == 2 and _nat(t[0]) and t[0] >= 1 and isinstance(t[1], dict)
                    and sorted(t[1]) == ["fn"] and type(t[1]["fn"]) is str for t in tiles)
            and all(a[0] < b[0] for a, b in zip(tiles, tiles[1:]))):
        return no(NOT_WINDOW)
    rc = None
    for k, t in tiles:
        if k % TILE_K_STEP or k >= TILE_K_LIMIT:
            return None, ("tile-k", tile_k(k))
        got = _tile(defs.get(t["fn"]), t["fn"], k)
        if got is None or rc not in (None, got):
            return no(no_template(k))
        rc = got
    layout = {"calls": vals["CALLS"], "linears": vals["LINEARS"], "m": vals["M"], "tile": list(rc[:2])}
    _ss, why = strata(layout)
    if why is not None:
        return None, why
    tks, lks = [k for k, _t in tiles], sorted({k for _name, k, _n in layout["linears"]})
    if tks != lks:
        return no(f"the window's TILES are at K = {tks}, not at its linears' K = {lks}: h1window binds one template "
                  "per K its linears have")
    (r, c, _j10r), m, of = rc, layout["m"], {k: t["fn"] for k, t in tiles}
    ks = {name: (k, n) for name, k, n in layout["linears"]}
    want = _runs((of[ks[lin][0]], (m // r) * (ks[lin][1] // c)) for lin, _blk in layout["calls"])
    root = defs.get(d["root"])
    body = root.get("body") if isinstance(root, dict) and root.get("form") == "composite" else None
    nodes = body.get("nodes") if isinstance(body, dict) else None
    if type(nodes) is not list:
        return no(BODY)
    got = []
    for x in nodes:
        if not (isinstance(x, dict) and _nat(x.get("fn")) and x["fn"] < len(callees)):
            return no(BODY)
        t = callees[x["fn"]]
        if _INPUT.fullmatch(t):
            continue
        n = 1 if x.get("form") == "call" else x.get("n") if x.get("form") == "batch" else None
        if not (_nat(n) and n >= 1):
            return no(BODY)
        got.append((t, n))
    if _runs(got) != want:
        return no(BODY)
    return layout, None


def _leaves(t: Any) -> int:
    """A descriptor type's leaves (`types.from_json`'s): ``["Value", w]`` one, ``["Array", n, t]`` n of t's, a
    ``["Tuple", [t, ...]]`` or ``["Record", [[name, t], ...]]`` its fields'; zero for any other form."""
    if not (type(t) is list and t):
        return 0
    if t[0] == "Value":
        return 1
    if t[0] == "Array" and len(t) == 3 and _nat(t[1]):
        return t[1] * _leaves(t[2])
    if t[0] == "Tuple" and len(t) == 2 and type(t[1]) is list:
        return sum(_leaves(x) for x in t[1])
    if t[0] == "Record" and len(t) == 2 and type(t[1]) is list:
        return sum(_leaves(x[1]) for x in t[1] if type(x) is list and len(x) == 2)
    return 0


def shape_of(descriptor: bytes) -> dict:
    """What a statement over the window's units reads of the descriptor `descriptor`, one `layout_of` accepted: its
    parameters' ``prefix`` (its record's ``PREFIX``) and, for each K, its template's id (``TILES``), K's chunks ``n``
    (`weights.chunks`: each A and B row n leaves) and the 16-bit words of its A leaf, its B leaf and its z/y record (``a_k<K>``'s element, ``b_k<K>``'s and ``zy``, h1window's widths:
    `h1window.table_reads_file`'s), and of its unit's receipt (the ``receipt`` field of its return, the statement's
    output row o0; 0 when it returns none of whole 16-bit words), and whether it is J10-R's (``j10r``, its parameters
    ending in `SHARED_J10R`). Its cost is the descriptor's bytes."""
    d = json.loads(descriptor.decode())
    vals = {k: json.loads(v) for k, v in _record(d["program_fn"])}
    tiles = {}
    for k, t in vals["TILES"]:
        body = d["definitions"][t["fn"]]["body"]
        a, b, zy = (p[1] for p in body["params"][:3])
        tiles[k] = {"fn": t["fn"], "n": chunks(k), "a": _leaves(a[2]) // 16, "b": _leaves(b[2]) // 16, "zy": _leaves(zy) // 16,
                    "receipt": _receipt_words(body.get("ret_type")),
                    "j10r": [p[0] for p in body["params"][2:]] == list(SHARED_J10R)}
    return {"prefix": vals["PREFIX"], "tiles": tiles}


def _bits(t: Any) -> int:
    """A descriptor type's bits, its leaves' widths (`types.leaf_widths`' sum): ``["Value", w]`` w, and `_leaves`' rule
    for the rest; zero for any other form."""
    if not (type(t) is list and t):
        return 0
    if t[0] == "Value":
        return t[1] if len(t) == 2 and _nat(t[1]) else 0
    if t[0] == "Array" and len(t) == 3 and _nat(t[1]):
        return t[1] * _bits(t[2])
    if t[0] == "Tuple" and len(t) == 2 and type(t[1]) is list:
        return sum(_bits(x) for x in t[1])
    if t[0] == "Record" and len(t) == 2 and type(t[1]) is list:
        return sum(_bits(x[1]) for x in t[1] if type(x) is list and len(x) == 2)
    return 0


def _receipt_words(ret: Any) -> int:
    """The 16-bit words of a template return's ``receipt`` field (``["Record", [["receipt", t], ...]]``, its first), its
    bits as the statement's output row o0 holds them (`h1window.statement_rows`' rule, `_bits`); 0 for another form or
    bits of no whole words."""
    fields = ret[1] if type(ret) is list and len(ret) == 2 and ret[0] == "Record" and type(ret[1]) is list else []
    t = next((x[1] for x in fields if type(x) is list and len(x) == 2 and x[0] == RECEIPT_FIELD), None)
    n = _bits(t)
    return n // 16 if n % 16 == 0 else 0


#: the partition object a window's statements are verified under: every instance of each K's template one unit
PARTITION_FORMAT = "verity/partition/v1"
QUERY = ("Q_template_instances", 0)


def partition_of(program: str, shape: dict) -> dict:
    """The partition object (`partition_object.build`'s) of the window whose circuit's descriptor has SHA-512 `program`
    and whose `shape_of` is `shape`: its templates by ascending K, as `h1window.query` names them."""
    return {"format": PARTITION_FORMAT, "program": program,
            "query": {"name": QUERY[0], "version": QUERY[1],
                      "params": {"templates": [shape["tiles"][k]["fn"] for k in sorted(shape["tiles"])]}}}


def registered_file(program: str, layout: dict, shape: dict, call: int, roots: dict, window: str, steps: int, salt: str
                    ) -> tuple[dict | None, tuple[str, str] | None]:
    """The verifier's ``--registered`` file (`registered.READS_V1`) of call `call`'s statement in the window `layout` (one
    `strata` accepted, its circuit's descriptor of SHA-512 `program` and `shape_of` `shape`, its calls `steps` rounds'
    equal blocks), and None; or None and the refusal (``registration``: the window registers no root, or two, for a value
    the statement reads). It is `h1window.table_reads_file`'s, built from the descriptor alone: unit u, tile t =
    ti·(n/C) + tj of the call, reads its A row i (port ``a_k<K>_<i>``) at row o + R·ti + i of its round s's A table of
    K ``<prefix>a/<K>@<s>`` (M rows for each of the round's calls of K, o the call's first, `step_row`), its B row j
    (``b_k<K>_<j>``) at row b0 + C·tj + j of the weights' ``<prefix>b/<K>`` (b0 the call's linear's first row in it,
    its rows each of K's linears' n in turn); at a K of x = `weights.chunks` (K) > 1 chunks each row is x leaves, so
    its chunk row i of x·R (port ``a_k<K>_<i>``) is leaf x·(o + R·ti) + i, its chunk row j of x·C leaf
    x·(b0 + C·tj) + j, and each of those values has x leaves a row; and its record (``zy``) at row t of the call's z table ``<prefix>zy@<call>``
    (a row a unit of the call); and, when the window registers the call's receipt table ``<prefix>receipt@<call>``,
    the statement's output row o0 (`RECEIPT_PORT`, the unit's receipt, `shape_of`'s width) at its row t, so the
    receipts the round roots commit are the ones the units return; every other unit reads none of them: each port's
    reads start at the call's first unit u0 (``reads_from``, `core_check.call_reads`) and end at its last (the verifier
    reads none below ``reads_from`` or past ``reads``). Each value is a row/v2 value of 16-bit leaves under
    `registered.domain` of its name, rows and words, binding the window's open root `window` but for the weights',
    held across windows (`domain_of`), at the root `roots` (``{value: [root, ...]}``, each value's registered roots)
    gives it. Its reads cost the call's units, once a port; only the ok rows' known content is the window's.

    The statement's public row ``pub`` (`PUB_PORT`), each unit's seed's inputs, is the verifier's own: unit u0 + t reads
    row t of the call's public table ``<prefix>pub@<call>`` (`h1window.public_table`), which the file names at its known
    content (``known``, `Flock.Registered.checkKnownAll`): the window's Program salt `salt` (hex), the unit's position
    u0 + t and the call's roots, root(``a/<K>@<s>``) and root(``b/<K>``) as the window registers them
    (`core_check.public_words`), each under `core_check.public_row_salt` of `window` and its position, at the root of
    their registration. So a unit seeded under another salt, position or roots (another call's, or made up) opens no row
    of it, and the prover never chooses its own noise or fold keys. A J10-R window's (`shape_of`'s ``j10r``) row is that
    and then the unit's ``call`` and ``blocks`` (`core_check.public_row`'s J10-R form), which the verifier derives from
    the window's layout too: ``call`` the call's first position u0 (its seed_c's) and ``blocks`` tile (I, J) = (t // (n/C),
    t % (n/C))'s `core_check.j10r_blocks` at the call's slot M, its K and the tile, so a unit that draws its noise from
    another call's stream or another unit's blocks of it opens no row either.

    The statement's output row ``ok`` (`OK_PORT`, o1) is the verifier's own too: unit u reads row u of the window's ok
    rows ``<prefix>ok`` (`OK_VALUE`, `ok_table`, one row a unit of the window), named at its known content, the one word
    1 under `core_check.ok_salt` of `window` and u. So a unit that computed 0 opens no row of it; and a window whose entries
    register those rows themselves is refused (``registration``)."""
    (r, c), m, prefix = layout["tile"], layout["m"], shape["prefix"]
    ov = f"{prefix}{OK_VALUE}"
    if ov in roots:
        return None, ("registration", f"the window registers {ov}, its units' ok rows, which are the verifier's own (the "
                                      "one word 1 under each unit's public ok salt): no entry registers them")
    lin = layout["calls"][call][0]
    ks = {name: (k, n) for name, k, n in layout["linears"]}
    k, n = ks[lin]
    tile = shape["tiles"][k]
    b0 = 0
    for name, kk, nn in layout["linears"]:
        if name == lin:
            break
        b0 += nn if kk == k else 0
    sizes = [size for size, _work in call_work(layout)]
    u0, size, per, x = sum(sizes[:call]), sizes[call], n // c, tile["n"]
    s, calls_k, at = step_row(layout, steps, call)
    a, zy, b = f"{prefix}a/{k}@{s}", f"{prefix}zy@{call}", f"{prefix}b/{k}"
    rows_of = {a: (x * m * calls_k, tile["a"]), zy: (size, tile["zy"]),
               b: (x * sum(nn for _name, kk, nn in layout["linears"] if kk == k), tile["b"])}
    rc = f"{prefix}receipt@{call}"
    if rc in roots:
        if not tile["receipt"]:
            return None, ("registration", f"the window registers {rc}, and K = {k}'s template returns no receipt of "
                                          "whole 16-bit words for its statement's output row o0 to read")
        rows_of[rc] = (size, tile["receipt"])
    for v in rows_of:
        if len(roots.get(v, ())) != 1:
            return None, ("registration", f"call {call}'s statement reads {v}, which the window registers "
                                          f"{'twice' if roots.get(v) else 'not'}: register each of a call's tables and "
                                          "its weights once")
    if any(_digest(roots[v][0]) is None for v in (a, b)):
        return None, ("registration", f"call {call}'s seed reads the roots of {a} and {b}, which are not both SHA-512 "
                                      "digests (128 lowercase hex): register each table at its tree's root")
    ports: list[tuple[str, str, Any]] = (
        [(f"a_k{k}_{i}", a, lambda t, i=i: x * (at + r * (t // per)) + i) for i in range(x * r)]
        + [(f"b_k{k}_{j}", b, lambda t, j=j: x * (b0 + c * (t % per)) + j) for j in range(x * c)]
        + [("zy", zy, lambda t: t)] + ([(RECEIPT_PORT, rc, lambda t: t)] if rc in rows_of else []))
    params = {f"{p}_k{k}_{i}": {G.PARAM_KEY: f"{p}_k{k}" if n == 1 else f"{p}_k{k}[{i}]"}
              for p, n in (("a", x * r), ("b", x * c)) for i in range(n)}
    out = {}
    for port, v, row in ports:
        leaves, words = rows_of[v]
        out[port] = {**params.get(port, {}), "value": v, "root": roots[v][0], "leaves": leaves, "words": words,
                     "word_bits": G.WORD_BITS,
                     "domain": G.domain(program, v, words, leaves,
                                        window=None if is_held(v) else window).domain_id.hex(),
                     **G.schema_entry(G.SCHEMA_V2), **CC.call_reads(u0, [row(t) for t in range(size)])}
    pv = f"{prefix}pub@{call}"
    blocks = [CC.j10r_blocks(m, k, (r, c), t // per, t % per) for t in range(size)] if tile["j10r"] else None
    rows, salts, reg = public_table(program, pv, salt, window, u0, size, (roots[a][0], roots[b][0]), blocks)
    out[PUB_PORT] = {"value": pv, "root": reg.root.hex(), "leaves": size, "words": len(rows[0]), "word_bits": G.WORD_BITS,
                     "domain": reg.domain.domain_id.hex(), **G.schema_entry(G.SCHEMA_V2),
                     **CC.call_reads(u0, list(range(size))), "known": {"rows": rows, "salts": [x.hex() for x in salts]}}
    rows, salts, reg = ok_table(program, ov, window, sum(sizes))
    out[OK_PORT] = {"value": ov, "root": reg.root.hex(), "leaves": sum(sizes), "words": len(CC.OK_ROW),
                    "word_bits": G.WORD_BITS, "domain": reg.domain.domain_id.hex(), **G.schema_entry(G.SCHEMA_V2),
                    **CC.call_reads(u0, list(range(u0, u0 + size))),
                    "known": {"rows": rows, "salts": [x.hex() for x in salts]}}
    return {"format": G.READS_V1, "ports": out}, None


def public_table(program: str, value: str, salt: str, window: str, u0: int, n: int, roots: tuple[str, str],
                 blocks: list[tuple[int, int]] | None = None) -> tuple[list[list[int]], list[bytes], G.Registered]:
    """`h1window.public_table`: a call's public rows ``value`` over its `n` units from unit `u0`, their salts and their
    registration, unit u's row the window's Program salt `salt`, its position u and its call's `roots` (each hex,
    `core_check.public_words`), and with `blocks` (a J10-R window's, one a unit) the call's first position `u0` and unit
    u's blocks, under `core_check.public_row_salt` of the window's open root `window` and u, registered under the
    descriptor's SHA-512 `program` binding `window`."""
    s, rs = bytes.fromhex(salt), (bytes.fromhex(roots[0]), bytes.fromhex(roots[1]))
    rows = [CC.public_words(s, u0 + t, rs, None if blocks is None else (u0, blocks[t])) for t in range(n)]
    salts = [CC.public_row_salt(window, u) for u in range(u0, u0 + n)]
    return rows, salts, G.Registered.commit(program, value, rows, salts, G.WORD_BITS, G.SCHEMA_V2, window=window)


@functools.lru_cache(maxsize=8)
def ok_table(program: str, value: str, window: str, population: int) -> tuple[list[list[int]], list[bytes], G.Registered]:
    """`h1window.ok_table`: the window's ok rows ``value`` over its `population` units, their salts and their
    registration (h1_prove's `register_ok`): row u the one word 1 (`core_check.OK_ROW`) under `core_check.ok_salt` of
    the window's open root `window` and u, registered under the descriptor's SHA-512 `program` binding `window`. Every
    input is public, so the verifier and the prover compute the same root."""
    rows = [list(CC.OK_ROW) for _u in range(population)]
    salts = [CC.ok_salt(window, u) for u in range(population)]
    return rows, salts, G.Registered.commit(program, value, rows, salts, G.WORD_BITS, G.SCHEMA_V2, window=window)


def law(ss: list[dict], k: Any) -> tuple[str, str] | None:
    """`ss` (`strata`'s) under the work law ``work:k`` at the committed K `k`: each stratum's k_s (`core_check.work_k`,
    over the population's work W = Σ w_s·n_s), set in place; else the refusal (``budget``): `k` isn't a positive integer
    at most `MAX_K`, or the window has fewer than K units, so its draw would not be K's."""
    if not (type(k) is int and 1 <= k <= MAX_K):
        return "budget", K_FORM
    sizes = [sum(hi - lo for lo, hi in s["units"]) for s in ss]
    if sum(sizes) < k:
        return "budget", (f"the window's {sum(sizes)} units are fewer than K = {k}, the K its period committed, so its "
                          f"draw would not be K's: the proof service commits a K at most every window's units "
                          f"({K_FIELD} in the period-open)")
    total = sum(s["work"] * n for s, n in zip(ss, sizes))
    for s, n in zip(ss, sizes):
        s["k"] = CC.work_k(k, s["work"], n, total, s["floor"])
    return None


def call_work(layout: dict) -> list[tuple[int, int]]:
    """Each call's units and the work of each of them, ``(size, work)`` in call order, of a layout `strata` accepted."""
    (r, c), m = layout["tile"], layout["m"]
    ks = {name: (k, n) for name, k, n in layout["linears"]}
    return [((m // r) * (ks[lin][1] // c), r * c * ks[lin][0]) for lin, _blk in layout["calls"]]


def served_rows(calls: list[dict], layout: dict, steps: int, t: int
                ) -> tuple[list[int] | None, tuple[str, str] | None]:
    """Each call's served rows m_s, in call order, and None; or None and the refusal (``manifest``). Call c's m_s is
    its manifest row's `SERVED_ROWS` (`calls`, rows `_credited` accepted, one a call of `layout`, one `strata`
    accepted), a natural number at most the period's T table rows `t` (`committed_rows`, whose `h1window.slot` is the
    A slot M); and call c is at step ``c // (C/S)`` of the window's `steps` rounds, as `_arow` places it, so every
    dense call of a step states the step's one m_s. The rows are under `calls_root`, which the draw root binds, so
    each m_s is fixed before the draw coin. Its cost is the calls."""
    per = len(layout["calls"]) // steps
    out: list[int] = []
    for i, row in enumerate(calls):
        got = row.get(SERVED_ROWS)
        if not (_nat(got) and got <= t):
            return None, ("manifest", no_served(i, t))
        first = i - i % per
        if i > first and got != out[first]:
            return None, ("manifest", other_served(i, first, i // per, got, out[first]))
        out.append(got)
    return out, None


def served_work(layout: dict, served: list[int]) -> list[tuple[int, int]]:
    """Each call's served positions and the work of each, ``(positions, work)`` in call order, of a layout `strata`
    accepted at its calls' served rows `served` (`served_rows`'): a call of a linear of K and n at m_s rows is m_s·(n/C)
    live (row, column tile) positions of C·K MACs each, m_s·n·K in all, where `call_work` counts its M slot rows."""
    c = layout["tile"][1]
    ks = {name: (k, n) for name, k, n in layout["linears"]}
    return [(m * (ks[lin][1] // c), c * ks[lin][0]) for (lin, _blk), m in zip(layout["calls"], served)]


def credited_work(layout: dict, credited_calls: list[int], served: list[int]) -> int:
    """The credited calls' served work: `served_work` over `credited_calls`."""
    picked = set(credited_calls)
    return sum(size * work for i, (size, work) in enumerate(served_work(layout, served)) if i in picked)


def established(layout: dict, credited_calls: list[int], served: list[int], k: int) -> Fraction | None:
    """The ε a draw of K = `k` under the work law over the window `layout` (one `strata` accepted) establishes at
    `DELTA` for its credited calls `credited_calls` at its calls' served rows `served`: a wrong set holding a share ε of
    the credited work C goes undrawn with probability at most δ; None when it establishes no ε <= 1 (C = 0, or K too
    small for C's share). It is `window.eps_at(K, δ, C, W, 1)`, C the credited calls' served work (`credited_work`,
    m_s·n·K a call) and W the window's units' (`call_work`, the draw's population, padded slots included): the work
    law's draw of K misses a set of work εC with probability at most (1 − εC/W)^K, each stratum drawing K·W_s/W of its
    units rounded up. Its cap is K >= W, where every stratum is drawn whole; never the unit count n, since a draw of
    K = n draws a stratum of little work only in part."""
    try:
        return Wd.eps_at(k, DELTA, credited_work(layout, credited_calls, served),
                         sum(size * work for size, work in call_work(layout)), 1)
    except ValueError:
        return None


def _drawn(coin: bytes, stream: Any, registration: Any, receipt: Any, ss: list[dict], k: Any
           ) -> tuple[list[int] | None, tuple[str, str] | None]:
    if not (type(stream) is str and _digest(registration) is not None and isinstance(receipt, dict)
            and all(_printable(x) and _printable(v) for x, v in receipt.items())):
        return None, ("draw", DRAW_ARGS)
    why = law(ss, k)
    if why is not None:
        return None, why
    got = CC.work_units(coin, stream, registration, receipt, ss)
    return (None, ("sampler", "the draw's stream ran out")) if got is None else (got, None)


def _window_drawn(window: dict, draw: dict, ss: list[dict], k: Any
                  ) -> tuple[list[int] | None, tuple[str, str] | None]:
    """The units the window's draw coin (`draw`'s ``coin``, a live opening) draws over the strata `ss` at the period's
    committed K `k`, its stream the window's root, its registration the draw's root and its receipt ``{"receipts": …}``
    (`_drawn`), and None; or None and the refusal."""
    try:
        coin = Ch.LiveOpening.from_json(draw.get("coin")).coin
    except Ch.CoinError as e:
        return None, ("coin", f"the draw coin: {e}")
    return _drawn(coin, window.get("root"), draw.get("root"), {"receipts": draw.get("receipts")}, ss, k)


def _finding(x: Any) -> list[tuple[str, str, str]] | None:
    """A session's registered finding (flock-verify's ``registered``, ``{port: {value, root}}``: each registered port's
    value and the root its reads were opened against) as ``(port, value, root)`` in port order, or None."""
    if not isinstance(x, dict):
        return None
    out = []
    for port in sorted(x):
        e = x[port]
        if not (_printable(port) and isinstance(e, dict) and sorted(e) == ["root", "value"] and _printable(e["value"])
                and _digest(e["root"]) is not None):
            return None
        out.append((port, e["value"], e["root"]))
    return out


def _proved(drawn: list[int], proofs: Any) -> tuple[str, str] | None:
    if not isinstance(proofs, list):
        return "outcome", PROOFS_FORM
    seen: dict[int, str] = {}
    picked = set(drawn)
    for p in proofs:
        if not (isinstance(p, dict) and sorted(p) == list(PROOF_KEYS) and _nat(p["unit"]) and p["verdict"] in VERDICTS
                and _digest(p["program"]) is not None and _finding(p["registered"]) is not None):
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


def drawn(coin: Any, stream: Any, registration: Any, receipt: Any, layout: Any, k: Any
          ) -> tuple[list[int] | None, tuple[str, str] | None]:
    """The units the draw coin `coin` (its value, a SHA-512 digest in hex) draws from the window `layout` (`strata`)
    under `law` at the committed K `k`, ascending (`core_check.work_units` under the stream's digest `stream`, the
    registration `registration` and its `receipt`), and None; or None and the refusal (``layout``, ``draw``,
    ``budget`` or ``sampler``, and why)."""
    ss, why = strata(layout)
    if why is not None:
        return None, why
    c = _digest(coin)
    if c is None:
        return None, ("draw", COIN_FORM)
    return _drawn(c, stream, registration, receipt, ss, k)


def _outcome(coin: Any, stream: Any, registration: Any, receipt: Any, layout: Any, k: Any, proofs: Any
            ) -> tuple[str, str] | None:
    """None when every unit the draw coin draws from the window (`drawn`, never a list the prover hands in) has exactly
    one outcome in `proofs` (each of `PROOF_KEYS`, verdict ``accept`` or ``reject``) and it is ``accept``, and no
    other unit has one; else the refusal (``layout``, ``draw``, ``sampler``, ``outcome``, ``missing`` or ``failed``, and
    why): the whole window is refused, with no partial credit. `window`'s rule on the draw coin it read, the units
    drawn under the served window's arguments."""
    got, why = drawn(coin, stream, registration, receipt, layout, k)
    return why if got is None else _proved(got, proofs)


def _entry(e: Any) -> bool:
    """A registered value's entry as one-stage's registration names it (`registered.ENTRY_KEYS`, with ``word_bits`` only
    when it is not 16): its value printable ASCII, its root a SHA-512 digest, its rows at least 1 and its words from 1
    to `MAX_ROWS`. Its rows have no other bound here: the verifier hashes pads only in `_a_link`, which holds each value
    it hashes to the layout's rows (`table_rows`) or `MAX_ROWS`."""
    bits = e.get(G.WIDTH_KEY, G.WORD_BITS) if isinstance(e, dict) else None
    return (isinstance(e, dict) and set(G.ENTRY_KEYS) <= set(e) <= {*G.ENTRY_KEYS, G.WIDTH_KEY}
            and _printable(e["value"]) and _digest(e["root"]) is not None and type(e["domain"]) is str
            and _nat(e["leaves"]) and e["leaves"] >= 1 and _nat(e["words"]) and 1 <= e["words"] <= MAX_ROWS
            and type(bits) is int and bits in G.WIDTHS and e.get(G.WIDTH_KEY) != G.WORD_BITS)


def domain_of(program: str, window: str, e: dict, session: str | None = None) -> str:
    """The domain one-stage's registration gives the registered value `e` (an entry `_entry` accepts) of the window whose
    open root is `window`, under the circuit whose descriptor's SHA-512 is `program` (`registered.domain`): the
    weights' (``…/b/<K>``, `is_held`) are held across windows and bind none; round 0's input-ids table (``…/ids@0``,
    `ids_step` 0) and request table (``…/requests@0``, `is_requests`), whose roots the window's open root contains
    (`requests_root`, `_arow`), bind the window's session `session` in that root's place; every other value's rows are
    the window's own, its A rows and its calls' A and z tables (`h1window.table`) among them, and its domain binds the
    window's open root."""
    v, first = e["value"], ids_step(e["value"]) == 0 or is_requests(e["value"])
    return G.domain(program, v, e["words"], e["leaves"], e.get(G.WIDTH_KEY, G.WORD_BITS),
                    None if is_held(v) or first else window, session=session if first else None).domain_id.hex()


def other_registration(v: str) -> str:
    return (f"registered value {v}'s domain is not the one the window's registration gives it (the registered domain of "
            f"{v} under the circuit the window's proofs were verified against, the SHA-512 of its descriptor, and, for a "
            "value other than the weights' …/b/<K> and round 0's …/ids@0 and …/requests@0, the window's open root, the "
            "root the auditor's ledger opened its session for, for those two that session): the entry is another "
            "registration's, "
            "another window's or another circuit's")


def _registration(window: str, program: str, prefix: str, entries: list, ss: list[dict], layout: dict, steps: int,
                  proofs: list, session: str | None = None) -> tuple[str, str] | None:
    """None when the window's registered `entries` are its own registration's and each drawn unit's proof (`proofs`, one
    accepted outcome a unit, as `_proved` holds them) is a session verified against it, else the refusal
    (``registration``, and why). Each A value's, input-ids table's and request table's (`is_requests`) domain, and each
    other value's a proof opened, is `domain_of` the window's
    open root `window`, its session `session` and the descriptor's SHA-512 `program`; each proof was verified against
    `program`; every
    registered port its session opened (its ``registered`` finding) is of a value the window registers once, at that
    value's registered root, but for its call's public table (`pub_call`) and the window's ok rows (``<prefix>ok``,
    `OK_VALUE`), which no entry registers and `_verified` holds to the rows `registered_file` built; and it opened its unit's K's A value: the window's ``…/a/<K>`` (K the unit's stratum `ss`'s)
    or its call's round's A table of K ``…/a/<K>@<s>`` (the call `layout`'s units place it in, s its round of `steps`,
    its calls the rounds' equal blocks), and its K's B rows, the window's ``<prefix>b/<K>`` (`prefix` the descriptor's
    ``PREFIX``). A step of `window`, after `_a_link` has held each
    A value's root to the segments the round roots commit and `_period_weights` each ``<prefix>b/<K>`` to the weights'
    root its period signed."""
    def held(v: str, e: dict) -> str | None:
        if not _entry(e):
            return (f"registered value {v}'s entry is not one-stage's {{value, root, leaves, words, domain}} (with "
                    f"word_bits only when it is not 16), its rows at least 1 and its words from 1 to {MAX_ROWS}")
        return None if e["domain"] == domain_of(program, window, e, session) else other_registration(v)

    for e in entries:
        if is_tied(e.get("value")) or is_requests(e.get("value")):
            why = held(e["value"], e)
            if why is not None:
                return "registration", why
    names = [e.get("value") for e in entries]
    checked: set[str] = set()
    starts, at, c, per = [], 0, 0, max(1, len(layout["calls"]) // steps)
    for size, _work in call_work(layout):
        starts.append(at)
        at += size
    for p in sorted(proofs, key=lambda p: p["unit"]):
        u = p["unit"]
        k = next(s["template"][1:] for s in ss if any(lo <= u < hi for lo, hi in s["units"]))
        while c + 1 < len(starts) and starts[c + 1] <= u:
            c += 1
        if p["program"] != program:
            return "registration", (f"drawn unit {u}'s proof was verified against another circuit than the window's (its "
                                    "program is not the SHA-512 of the descriptor whose layout the window was drawn over): "
                                    "verify every drawn unit's session against the window's committed circuit")
        opened, weights = False, False
        for port, v, root in _finding(p["registered"]):
            if pub_call(v) is not None or v == f"{prefix}{OK_VALUE}":
                continue
            if names.count(v) != 1:
                return "registration", (f"drawn unit {u}'s proof opened port {port} against {v}, which the window "
                                        f"registers {'twice' if v in names else 'not'}: its session was verified against "
                                        "another registration than the window's")
            e = entries[names.index(v)]
            if v not in checked and not is_tied(v):
                why = held(v, e)
                if why is not None:
                    return "registration", why
            checked.add(v)
            if root != e["root"]:
                return "registration", (f"drawn unit {u}'s proof opened port {port}'s rows of {v} against another root "
                                        f"than the one the window registered for {v}: its session was verified against "
                                        "another registration, or the window's entry is not the root its sessions opened")
            got = a_value(v)
            opened = opened or (got is not None and got[0] == k and got[1] in (None, str(c // per)))
            weights = weights or v == f"{prefix}b/{k}"
        if not opened:
            return "registration", (f"drawn unit {u}'s proof opened no A value …/a/{k} or …/a/{k}@{c // per} (its "
                                    "call's round's) of "
                                    "the window, so nothing ties the A rows it proved to the ones the round roots "
                                    "committed: verify each drawn unit's session with the window's A value registered")
        if not weights:
            return "registration", (f"drawn unit {u}'s proof opened no weights {prefix}b/{k} of the window, so nothing "
                                    "ties the B rows it proved to the ones its period registered: verify each drawn "
                                    "unit's session with the window's weights registered")
    return None


def _same(e: dict, c: dict) -> bool:
    """`c` has the root, domain and leaf count of `e`, an entry `_a_link` accepted (both strings, and a natural)."""
    return (c.get("root") == e["root"] and c.get("domain") == e["domain"] and _nat(c.get("leaves"))
            and c["leaves"] == e["leaves"])


def _arow(rounds: list, entries: list, registered: list, layout: dict, prefix: str) -> tuple[str, str] | None:
    """None when each round's A table of each K and each round's input-ids table among the window's registered
    `entries` is the Commit's value of the same rows among its record's `registered` (`commit_of`'s): same root, domain
    and leaf count; else the refusal (``arow``, and why). The window's `layout` (`strata`'s) calls are its round log
    `rounds`' S rounds' equal blocks, and round s's ``<prefix>a/<K>@<s>`` (`prefix` the descriptor's ``PREFIX``,
    `step_tables`) holds M rows for each of the round's calls of K, call after call, and is the Commit's `commit_rows`
    of s and K; round s's ``<prefix>ids@<s>`` is the Commit's `commit_ids` of s. The Commit registers each of those
    values once. So the A rows a call's units read are its linear's input as the served Commit commits it, and the
    input token ids under each round's root are its step's (ROOTS1). The window registers no other input-ids table, and
    round 0's request root (``requests``, the one its open root binds) is `requests_root` of its request table's root
    (``request_table``) and ``<prefix>ids@0``'s, so round 0's input ids are fixed before the salt coin. The window
    registers at most one request table (`is_requests`), ``<prefix>requests@0``, at root ``request_table``, so the
    live unit's ``req`` rows are round 0's admitted requests; `_registration` holds its domain to the session. A step of
    `window`, after `_a_link` has held each to the segments the round roots commit, `_period_weights` round 0's request
    root to the open root, and the manifest to the layout's calls. Its cost is the calls and the records' entries."""
    steps, calls = len(rounds), layout["calls"]
    if len(calls) % steps:
        return "arow", (f"the window's {len(calls)} calls are not its {steps} rounds' equal blocks, so no call's round, "
                        "and no Commit value of its rows, is fixed")
    ours = {e["value"]: e for e in entries if is_tied(e.get("value"))}
    theirs: dict[str, list[dict]] = {}
    for e in registered:
        if type(e.get("value")) is str:
            theirs.setdefault(e["value"], []).append(e)
    pairs = [(f"{prefix}a/{k}@{s}", commit_rows(s, k), f"round {s}'s A table of K = {k}",
              chunks(k) * layout["m"] * len(cs)) for s, k, cs in step_tables(layout, steps)]
    pairs += [(f"{prefix}ids@{s}", commit_ids(s), f"round {s}'s input-ids table", None) for s in range(steps)]
    for v, name, whose, rows in pairs:
        if v not in ours:
            return "arow", (f"the window registers no {whose} {v}, so nothing ties it to the Commit's {name}: the "
                            "served window registers each round's A table of each K and its input-ids table")
        if rows is not None and ours[v]["leaves"] != rows:
            return "arow", (f"{whose} {v} has {ours[v]['leaves']} rows, not {rows}: it holds the M rows of each of the "
                            "round's calls of K, call after call, each row K's chunks' leaves (weights.chunks), so a "
                            "call missing from it reads another call's rows (ROOTS1)")
        got = theirs.get(name, [])
        if len(got) != 1:
            return "arow", (f"the Commit record registers {name} {'twice' if got else 'not'}: the served Commit "
                            "registers each round's A table of each K as row-table/<round>/k<K> and each round's "
                            "input-ids table as input-ids/<round>, once")
        if not _same(ours[v], got[0]):
            return "arow", (f"{whose} {v} is not the Commit's {name}: their root, domain or leaf count differ, so its "
                            "rows are not the ones the served Commit commits (ROOTS1). The Commit registers the "
                            "window's own table as its value, with no tree of its own")
    want = {v for v, _name, _whose, _rows in pairs}
    for v in ours:
        if ids_step(v) is not None and v not in want:
            return "arow", (f"the window registers input-ids table {v}, which is not {prefix}ids@<s> of any of its "
                            f"{steps} rounds: it registers each round's input-ids table once, under its descriptor's "
                            "PREFIX")
    table, ids = _digest(rounds[0].get("request_table")), _digest(ours[f"{prefix}ids@0"].get("root"))
    if table is None or ids is None or rounds[0].get("requests") != requests_root(table, ids).hex():
        return "arow", IDS_OUTSIDE
    tables = [e for e in entries if is_requests(e.get("value"))]
    for e in tables:
        if e["value"] != f"{prefix}{REQUESTS_TABLE}":
            return "arow", (f"the window registers request table {e['value']}, which is not {prefix}{REQUESTS_TABLE}: "
                            "it registers round 0's request table once, under its descriptor's PREFIX")
    if len(tables) > 1 or any(e.get("root") != table.hex() for e in tables):
        return "arow", REQUESTS_OUTSIDE
    return None


def no_statement_files(k: Any) -> str:
    return (f"the period's open commits no circuit file for K = {k}'s template ({CIRCUIT_FILES} k{k}, the SHA-512 of the "
            "circuit file the auditor's lowering built for it), so no statement of its calls can be verified: commit "
            f"each K's circuit file in the period-open")


def _verified(statements: Any, files: Any, raw: bytes, layout: dict, entries: list, window: str, units: list[int],
              steps: int, run: Run, salt: str) -> tuple[list[dict] | None, tuple[str, str] | None]:
    """Each drawn unit's proof outcome (`PROOF_KEYS`) from the verifier of record's own run over the window's
    `statements` (`STATEMENT_KEYS`: one a call with a drawn unit among `units`, its proof files), and None; or None and
    the refusal. Call c's statement is verified by `run` (`proofs.verify`) on its circuit file, which must hash to
    `files`' digest for its K (``k<K>``, the period-open's `CIRCUIT_FILES`, else ``circuit``), against the circuit's
    descriptor `raw`, its partition object (`partition_of`) and the call's ``--registered`` file credit builds itself
    (`registered_file`, at the roots the window's `entries` register, its open root `window`, its `steps` rounds and its
    Program salt `salt`). A session whose finding opened the call's public table at another root than that file's, or
    opened the window's ok rows (port ``o1``) anywhere but at that file's value and root, is refused (``registration``):
    `_registration` holds every other value to the window's entries, and these two, which no entry registers, to the
    rows credit built. A file missing or an
    answer that is not one verdict a session refuses (``proof``), and so does a verifier that can't run: the window stays
    refused. A session the verifier rejects refuses the window (``failed``: no partial credit, and no retry), and one
    that proved a unit of another call refuses it (``proof``). Each unit an accepted session proved is one outcome:
    accepted, the program its verdict names and its registered finding, which `_proved` and `_registration` then hold
    to the draw and to the window's registration."""
    asks, refused = _asks(statements, files, raw, layout, entries, window, units, steps, salt)
    out: list[dict] = []
    for a in asks:
        s = statements[a["statement"]]
        try:
            got, why = run(PF.Proof(Path(s["circuit"]), Path(s["public"]), tuple(map(Path, s["sessions"]))),
                           circuit_sha512=a["circuit_sha512"], program=raw, partition=a["partition"],
                           registered=a["registered"])
        except (PF.LeanError, OSError, ValueError, KeyError, TypeError) as e:
            return None, ("proof", f"call {s['call']}'s statement could not be verified, so the window stays refused: "
                                   f"{e}")
        if why is not None:
            return None, (why[0], f"call {s['call']}'s statement: {why[1]}")
        why = _judged(layout, s["call"], a["registered"], got, out)
        if why is not None:
            return None, why
    return (None, refused) if refused is not None else (out, None)


def _asks(statements: Any, files: Any, raw: bytes, layout: dict, entries: list, window: str, units: list[int],
          steps: int, salt: str) -> tuple[list[dict], tuple[str, str] | None]:
    """The verifier runs `_verified` makes, in statement order, up to the first statement a check before the verifier
    refuses, and that refusal (or None): each ``{statement, circuit_sha512, registered, partition}``, the statement's
    index among `statements` and the verifier's inputs for it. The Python reference of the Lean's
    `PouwCheck.Verified.asks`, which `_lean_verified` runs the verifier on."""
    if not (isinstance(files, dict) and all(type(x) is str and _digest(v) is not None for x, v in files.items())):
        return [], ("circuit", FILES_FORM)
    if not (isinstance(statements, list)
            and all(isinstance(s, dict) and sorted(s) == list(STATEMENT_KEYS) and _nat(s["call"])
                    and type(s["circuit"]) is str and type(s["public"]) is str and type(s["sessions"]) is list
                    and s["sessions"] and all(type(x) is str for x in s["sessions"]) for s in statements)):
        return [], ("proof", STATEMENTS_FORM)
    sizes = [size for size, _work in call_work(layout)]
    if any(s["call"] >= len(sizes) for s in statements):
        return [], ("proof", STATEMENTS_FORM)
    starts = list(itertools.accumulate(sizes, initial=0))[:-1]
    drawn = {bisect.bisect_right(starts, u) - 1 for u in units}
    program, shape = hashlib.sha512(raw).hexdigest(), shape_of(raw)
    partition = partition_of(program, shape)
    roots: dict[str, list[str]] = {}
    for e in entries:
        roots.setdefault(e.get("value"), []).append(e.get("root"))
    ks = {name: k for name, k, _n in layout["linears"]}
    seen: set[int] = set()
    out: list[dict] = []
    for i, s in enumerate(statements):
        c = s["call"]
        if c in seen:
            return out, ("outcome", f"call {c} has two statements: a call's drawn units are proved in one statement "
                                    "and verified once, and a retry never reproves them")
        seen.add(c)
        if c not in drawn:
            return out, ("outcome", f"call {c} has a statement and no drawn unit: only the draw's units are proved")
        k = ks[layout["calls"][c][0]]
        if f"k{k}" not in files:
            return out, ("circuit", no_statement_files(k))
        reg, why = registered_file(program, layout, shape, c, roots, window, steps, salt)
        if why is not None:
            return out, why
        out.append({"statement": i, "circuit_sha512": files[f"k{k}"], "registered": reg, "partition": partition})
    return out, None


def _judged(layout: dict, c: int, reg: dict, got: list[PF.Finding], out: list[dict]) -> tuple[str, str] | None:
    """Call `c`'s statement's findings `got`, verified against the ``--registered`` file `reg`, appended to `out` as
    outcomes, and None; or the refusal of the first finding that credits nothing."""
    sizes = [size for size, _work in call_work(layout)]
    start = sum(sizes[:c])
    for f in got:
        if not f.accepted:
            return ("failed", f"call {c}'s session {f.session} was rejected by the verifier of record ({f.why}): a "
                              "failed drawn unit refuses the whole window, not only its call (no partial credit, and "
                              "no retry)")
        pub = reg["ports"][PUB_PORT]
        if any(pub_call(v) is not None and (port, v, root) != (PUB_PORT, pub["value"], pub["root"])
               for port, v, root in _finding(f.registered) or ()):
            return ("registration", f"call {c}'s session {f.session} opened a public table at another port, value or "
                                    f"root than the verifier's own {pub['value']} at port {PUB_PORT}: a unit's public "
                                    "row is the window's salt, its position and its call's roots, as credit builds "
                                    "them")
        why = _ok_held(c, f.session, f.registered, reg["ports"][OK_PORT])
        if why is not None:
            return why
        for u in f.units:
            if not start <= u < start + sizes[c]:
                return ("proof", f"call {c}'s session {f.session} proved unit {u}, which is no unit of call {c}: a "
                                 "call's statement proves its own call's units")
            out.append({"unit": u, "verdict": "accept", "program": f.program, "registered": f.registered})
    return None


def _ok_held(c: int, session: Any, registered: Any, ok: dict) -> tuple[str, str] | None:
    """None when call `c`'s session `session`, whose registered finding is `registered`, opened the window's ok rows
    once, at port ``o1`` (`OK_PORT`) and at the value and root of `ok` (the ok port of credit's ``--registered`` file,
    `registered_file`'s); else the refusal (``registration``). The ok rows are the verifier's own, registered under the
    window's open root, so a session that opened them at another root (rows registered without that root, another
    window's, or a unit's row 0) or not at all credits nothing. `_judged`'s step for each finding."""
    if [(p, v, root) for p, v, root in _finding(registered) or () if p == OK_PORT or v == ok["value"]] != [
            (OK_PORT, ok["value"], ok["root"])]:
        return ("registration", f"call {c}'s session {session} opened the ok rows at another port, value or root "
                                f"than the verifier's own {ok['value']} at port {OK_PORT}, or not at all: a unit's "
                                "ok row is the one word 1 under its public ok salt, as credit registers it, so a "
                                "unit that computed 0 opens none of it")
    return None


def _finding_json(f: PF.Finding) -> dict:
    """A verifier finding as `PouwCheck.Verified` reads it: ``{session, accepted, why, program, registered, units}``."""
    return {"session": f.session, "accepted": f.accepted, "why": f.why, "program": f.program,
            "registered": f.registered, "units": list(f.units)}


def _answer(run: Run, s: dict, ask: dict, raw: bytes) -> dict:
    """The verifier of record's answer for one of the Lean's asks (`_asks`'s form) on statement `s`'s proof files: its
    findings, its refusal, or why it could not run, as `PouwCheck.Verified` reads them."""
    try:
        got, why = run(PF.Proof(Path(s["circuit"]), Path(s["public"]), tuple(map(Path, s["sessions"]))),
                       circuit_sha512=ask["circuit_sha512"], program=raw, partition=ask["partition"],
                       registered=ask["registered"])
    except (PF.LeanError, OSError, ValueError, KeyError, TypeError) as e:
        return {"error": str(e)}
    return {"refused": list(why)} if why is not None else {"findings": [_finding_json(f) for f in got]}


def _lean_verified(statements: Any, files: Any, raw: bytes, entries: list, window: str, units: list[int], steps: int,
                   run: Run, salt: str) -> tuple[list[dict] | None, tuple[str, str] | None]:
    """`_verified`'s verdict from the Lean verifier of record (`PouwCheck.Verified.verified`, `verifier.lean`): the Lean
    names the verifier runs (`_asks`), this runs `run` on each in order until an answer the Lean will refuse at
    (an error, a refusal or a rejected session), and the Lean judges the answers. A Lean that can't run leaves the
    window refused (``proof``)."""
    q = {"statements": statements, "files": files, "descriptor": raw.hex(), "entries": entries, "window": window,
         "units": units, "steps": steps, "salt": salt, "runs": None}
    try:
        got = V.lean("credit-verified", q)
        if "run" in got:
            runs = []
            for a in got["run"]:
                runs.append(_answer(run, statements[a["statement"]], a, raw))
                if "findings" not in runs[-1] or not all(f["accepted"] for f in runs[-1]["findings"]):
                    break
            got = V.lean("credit-verified", {**q, "runs": runs})
    except V.LeanError as e:
        return None, ("proof", f"the Lean verifier of record could not judge the window's statements, so the window "
                               f"stays refused: {e}")
    return (None, tuple(got["refused"])) if "refused" in got else (got["outcomes"], None)


def _parsed(descriptor: Any) -> dict | None:
    """A descriptor's bytes read as `layout_of` reads them: canonical JSON of schema `SCHEMA` with its callees, definitions,
    program function and root; None for any other."""
    try:
        ok, d = _loads(descriptor.decode()) if type(descriptor) is bytes else (False, None)
    except UnicodeDecodeError:
        ok, d = False, None
    if not (ok and isinstance(d, dict) and d.get("schema") == SCHEMA and type(d.get("callees")) is list
            and all(type(x) is str for x in d["callees"]) and isinstance(d.get("definitions"), dict)
            and type(d.get("program_fn")) is str and type(d.get("root")) is str):
        return None
    return d


def live_of(descriptor: Any) -> tuple[dict | None, tuple[str, str] | None]:
    """The live Program (`circuit.live.program_v2`) whose descriptor is `descriptor` (its bytes, canonical JSON without
    annotations), ``{T, S, prefix, unit}``, and None; or None and the refusal (``live``, and why). T, S and the prefix are
    its program function's record (`LIVE_FN`'s ``T``, ``S`` and ``PREFIX``), its template ``unit`` is
    ``PouwLiveSlots_v2{T=<T>}``, a composite of id `LIVE_UNIT` bound to T alone, and its root's nodes other than inputs
    are S calls of it, unit s round s's. Its cost is the descriptor's bytes, never the gates its nodes unroll to."""
    def no(why: str) -> tuple[None, tuple[str, str]]:
        return None, (LIVE, why)

    d = _parsed(descriptor)
    if d is None:
        return no(LIVE_DESCRIPTOR_FORM)
    callees, defs, fn = d["callees"], d["definitions"], d["program_fn"]
    rec = _record(fn) if fn.startswith(LIVE_FN + "{") else []
    if [k for k, _v in rec] != list(LIVE_RECORD):
        return no(NOT_LIVE)
    vals = {}
    for k, s in rec:
        ok, vals[k] = _loads(s)
        if not ok:
            return no(NOT_LIVE)
    t, steps, prefix = vals["T"], vals["S"], vals["PREFIX"]
    if not (_nat(t) and t >= 1 and _nat(steps) and steps >= 1 and type(prefix) is str):
        return no(NOT_LIVE)
    unit = f"{LIVE_UNIT}{{T={t}}}"
    dj = defs.get(unit)
    b = dj.get("bindings") if isinstance(dj, dict) else None
    if not (isinstance(b, dict) and dj.get("form") == "composite" and dj.get("id") == LIVE_UNIT and sorted(b) == ["T"]
            and _nat(b["T"]) and b["T"] == t):
        return no(LIVE_BODY)
    root = defs.get(d["root"])
    body = root.get("body") if isinstance(root, dict) and root.get("form") == "composite" else None
    nodes = body.get("nodes") if isinstance(body, dict) else None
    if type(nodes) is not list:
        return no(LIVE_BODY)
    calls = 0
    for x in nodes:
        if not (isinstance(x, dict) and _nat(x.get("fn")) and x["fn"] < len(callees)):
            return no(LIVE_BODY)
        if _INPUT.fullmatch(callees[x["fn"]]):
            continue
        if not (x.get("form") == "call" and callees[x["fn"]] == unit):
            return no(LIVE_BODY)
        calls += 1
    if calls != steps:
        return no(LIVE_BODY)
    return {"T": t, "S": steps, "prefix": prefix, "unit": unit}, None


def other_live_rows(got: int, t: int) -> str:
    return (f"the window's live Program is at T = {got}, not the {t} table rows its period's open committed "
            f"({ROWS_FIELD}): its units read each round's T slot leaves and the T request headers")


def other_live_rounds(got: int, steps: int) -> str:
    return (f"the window's live Program has {got} rounds, not the window's {steps}: every round's served rows m_s is "
            "proved, one live@<s> a round")


def no_live_table(v: str, n: int) -> str:
    return (f"the window registers {v} {'twice' if n else 'not'}: each round's live unit reads round s's slot table "
            f"{LR.SLOTS}@<s> and round 0's request table {LR.REQUESTS}@0, each registered once")


def other_live_table(v: str, leaves: int, words: int, t: int, want: int) -> str:
    return (f"{v} has {leaves} rows of {words} words, not the T = {t} rows of {want} 16-bit words its live unit reads "
            "(circuit.live_reads)")


def other_live_finding(s: int, m: int) -> str:
    return (f"round {s}'s live@{s} proof opened other ports, values or roots than its statement reads: its slot ports "
            f"slot_<i> at round {s}'s slot table's registered root, its header ports req_<i> at round 0's request "
            f"table's, its m at the known m@{s} of the manifest's m_s = {m} served rows, its s at the known round@{s} "
            f"of {s}, its caps cap_<i> at the known caps of the max_tokens its period's open committed and its ok o0 at "
            "the known live-ok rows, the one word 1: a proof at another m, round or caps, of other tables, or whose unit "
            "computed 0, proves no step's served rows")


def _live_roots(entries: list, prefix: str, t: int, steps: int) -> tuple[dict | None, tuple[str, str] | None]:
    """The roots of the tables the window's live units read among its registered `entries` (``{value: root}``), and None;
    or None and the refusal (``live``): each round s's slot table ``<prefix>slots@<s>`` and round 0's request table
    ``<prefix>requests@0`` registered once, T = `t` rows of `live_reads.SLOT_WORDS` and `live_reads.HEADER_WORDS`
    16-bit words. `_a_link` holds each slot table to its round's root and `_arow` the request table to round 0's
    ``request_table``; `_registration` holds their domains to the open root and the session."""
    want = [(LR.table(prefix, LR.SLOTS, s), LR.SLOT_WORDS) for s in range(steps)]
    want.append((LR.table(prefix, LR.REQUESTS), LR.HEADER_WORDS))
    out = {}
    for v, words in want:
        got = [e for e in entries if e.get("value") == v]
        if len(got) != 1:
            return None, (LIVE, no_live_table(v, len(got)))
        e = got[0]
        if not (_entry(e) and G.WIDTH_KEY not in e and e["leaves"] == t and e["words"] == words):
            return None, (LIVE, other_live_table(v, e.get("leaves"), e.get("words"), t, words))
        out[v] = e["root"]
    return out, None


def _live_finding(program: str, prefix: str, window: str, t: int, steps: int, s: int, m: int, roots: dict,
                  caps: str) -> dict[str, tuple[str, str]]:
    """Round s's live@<s> finding as its statement reads it (`live_reads.reads_file`), ``{port: (value, root)}``: each
    slot port at round s's slot table, each header port at round 0's request table (their roots `roots`), ``m`` at the
    known ``m@<s>`` of `m`, ``s`` at the known ``round@<s>`` of s, each ``cap_<i>`` at the known ``caps``, whose root is
    `caps`, and ``o0`` at the known ``live-ok`` rows, each under the window Program's digest `program` binding its open
    root `window`."""
    sv, rv, cv = LR.table(prefix, LR.SLOTS, s), LR.table(prefix, LR.REQUESTS), LR.table(prefix, LR.CAPS)
    out = {f"slot_{i}": (sv, roots[sv]) for i in range(t)}
    out |= {f"req_{i}": (rv, roots[rv]) for i in range(t)}
    out["m"] = (LR.table(prefix, LR.M_ROWS, s), LR.m_table(program, prefix, window, s, m)[2].root.hex())
    out["s"] = (LR.table(prefix, LR.ROUND_ROWS, s), LR.round_table(program, prefix, window, s)[2].root.hex())
    out |= {f"cap_{i}": (cv, caps) for i in range(t)}
    out[LR.OK_PORT] = (LR.table(prefix, LR.OK_VALUE), LR.ok_table(program, prefix, window, steps)[2].root.hex())
    return out


Lived = Callable[[dict, bytes, dict, dict, list[int], str, str, str, list[int]], tuple[Any, tuple[str, str] | None]]


def _live(live: Any, lived: Lived, keys: tuple[str, ...], t: int, steps: int, prefix: str, program: str, window: str,
          session: str, entries: list, served: list[int], max_tokens: list[int]) -> tuple[str, str] | None:
    """None when the window's live Program proves each round's served rows, else the refusal (``live``, or `lived`'s).
    Its live evidence `live` (`keys`: `LIVE_KEYS`, or `LIVE_STATEMENTS` in `window`) carries the live Program's
    descriptor (`live_of`), at the period's T table rows `t`, the window's `steps` rounds and its circuit's PREFIX
    `prefix`, and `lived` gives each round's proof outcome (`PROOF_KEYS`, unit s round s's) from it, the descriptor's
    bytes and `live_of`, the live tables' roots (`_live_roots`), each step's m_s (``served[s·C/S]``), the window
    Program's digest `program`, its open root `window`, its session `session` and the max_tokens its period's open
    committed (`max_tokens`, `committed_length`). Each round has exactly one, accepted, verified against the live
    descriptor, its finding `_live_finding`'s at its step's m_s, round s and the public caps of `max_tokens`
    (`live_reads.caps_table`): a statement at an m other than the manifest's opens no ``m@<s>`` row, one at another round
    no ``round@<s>`` row, one at other caps no ``caps`` row, and one whose unit computed 0 no ``live-ok`` row. A step of
    `window`, after `_registration`. Its cost is the descriptor's bytes and the rounds' statements' ports."""
    if not (isinstance(live, dict) and sorted(live) == list(keys) and type(live["descriptor"]) is str
            and _HEX.fullmatch(live["descriptor"])):
        return LIVE, LIVE_FORM if keys == LIVE_KEYS else LIVE_STATEMENTS_FORM
    raw = bytes.fromhex(live["descriptor"])
    got, why = live_of(raw)
    if why is not None:
        return why
    if got["T"] != t:
        return LIVE, other_live_rows(got["T"], t)
    if got["S"] != steps:
        return LIVE, other_live_rounds(got["S"], steps)
    if got["prefix"] != prefix:
        return LIVE, LIVE_PREFIX
    roots, why = _live_roots(entries, prefix, t, steps)
    if why is not None:
        return why
    per = len(served) // steps
    ms = [served[s * per] for s in range(steps)]
    proofs, why = lived(live, raw, got, roots, ms, program, window, session, max_tokens)
    if why is not None:
        return why
    if not (isinstance(proofs, list)
            and all(isinstance(p, dict) and sorted(p) == list(PROOF_KEYS) and _nat(p["unit"])
                    and p["verdict"] in VERDICTS and _digest(p["program"]) is not None
                    and _finding(p["registered"]) is not None for p in proofs)):
        return LIVE, LIVE_PROOFS_FORM
    digest, seen = hashlib.sha512(raw).hexdigest(), {}
    caps = LR.caps_table(program, prefix, window, t, max_tokens)[2].root.hex()
    for p in proofs:
        s = p["unit"]
        if s >= steps:
            return LIVE, (f"live unit {s} has a proof outcome and the window has {steps} rounds: unit s is round s's "
                          "live@<s>")
        if s in seen:
            return LIVE, (f"round {s}'s live@{s} has two proof outcomes: each round's is proved and verified once, and "
                          "a retry never reproves it")
        seen[s] = p
    for s in range(steps):
        p = seen.get(s)
        if p is None:
            return LIVE, (f"round {s}'s live@{s} has no proof outcome: every round's served rows m_s is proved (the live "
                          "unit is never sampled), so a window with one missing is refused")
        if p["verdict"] != "accept":
            return LIVE, (f"round {s}'s live@{s} proof was rejected: its served rows m_s = {ms[s]} are not proved, so the "
                          "window is refused (no partial credit, and no retry)")
        if p["program"] != digest:
            return LIVE, (f"round {s}'s live@{s} proof was verified against another circuit than the window's live "
                          "Program: its program is not the SHA-512 of the live descriptor")
        f = {port: (v, root) for port, v, root in _finding(p["registered"])}
        if f != _live_finding(program, prefix, window, t, steps, s, ms[s], roots, caps):
            return LIVE, other_live_finding(s, ms[s])
    return None


def no_live_files() -> str:
    return (f"the period's open commits no circuit file for the live Program's template ({CIRCUIT_FILES} {LIVE}, the "
            "SHA-512 of the circuit file the auditor's lowering built for it), so no round's live@<s> can be verified: "
            "commit it in the period-open beside each K's")


def _live_verified(files: Any, run: Run) -> Lived:
    """`_live`'s `lived` over the live evidence's ``statements`` (`LIVE_STATEMENT_KEYS`, one a round): round s's verified by
    `run` (`proofs.verify`) on its circuit file, which must hash to `files`' ``live`` (the period-open's
    `CIRCUIT_FILES`, else ``circuit``), against the live descriptor, its partition object (every instance of its
    template one unit) and the ``--registered`` file credit builds itself (`live_reads.reads_file`, under the window
    Program's digest, its open root and its session, at the step's m_s and the period's committed max_tokens). A
    session the verifier rejects refuses the window (``live``), and so does one that proved another unit than its
    round's. The Python reference of the Lean's `PouwCheck.LiveVerified` (`_lean_live_verified`), held to it by the
    difftest (`lean/check/tests/test_live_verified_difftest.py`)."""
    def lived(live: dict, raw: bytes, got: dict, roots: dict, ms: list[int], program: str, window: str, session: str,
              max_tokens: list[int]) -> tuple[list | None, tuple[str, str] | None]:
        statements = live["statements"]
        asks, refused = _live_asks(files, statements, raw, got, roots, ms, program, window, session, max_tokens)
        out: list[dict] = []
        for a in asks:
            st = statements[a["statement"]]
            s = st["round"]
            try:
                found, why = run(PF.Proof(Path(st["circuit"]), Path(st["public"]), tuple(map(Path, st["sessions"]))),
                                 circuit_sha512=a["circuit_sha512"], program=raw, partition=a["partition"],
                                 registered=a["registered"])
            except (PF.LeanError, OSError, ValueError, KeyError, TypeError) as e:
                return None, (LIVE, f"round {s}'s live@{s} could not be verified, so the window stays refused: {e}")
            if why is not None:
                return None, (why[0], f"round {s}'s live@{s}: {why[1]}")
            for f in found:
                if not f.accepted:
                    return None, (LIVE, f"round {s}'s live@{s} session {f.session} was rejected by the verifier of "
                                        f"record ({f.why}): its served rows m_s = {ms[s]} are not proved (no retry)")
                if tuple(f.units) != (s,):
                    return None, (LIVE, f"round {s}'s live@{s} session {f.session} proved units {list(f.units)}, not "
                                        f"unit {s}: round s's statement proves its own round's unit")
                out.append({"unit": s, "verdict": "accept", "program": f.program, "registered": f.registered})
        return (None, refused) if refused is not None else (out, None)
    return lived


def _live_asks(files: Any, statements: Any, raw: bytes, got: dict, roots: dict, ms: list[int], program: str,
               window: str, session: str, max_tokens: list[int]) -> tuple[list[dict], tuple[str, str] | None]:
    """The verifier runs `_live_verified` makes, in statement order, up to the first statement a check before the
    verifier refuses, and that refusal (or None): each ``{statement, circuit_sha512, registered, partition}``, the
    statement's index among `statements` and the verifier's inputs for it, its ``--registered`` file at the period's
    committed `max_tokens`. The Python reference of the Lean's `PouwCheck.LiveVerified.asks`, which
    `_lean_live_verified` runs the verifier on."""
    if not (isinstance(files, dict) and all(type(x) is str and _digest(v) is not None for x, v in files.items())):
        return [], ("circuit", FILES_FORM)
    if LIVE not in files:
        return [], ("circuit", no_live_files())
    if not (isinstance(statements, list)
            and all(isinstance(s, dict) and sorted(s) == list(LIVE_STATEMENT_KEYS) and _nat(s["round"])
                    and type(s["circuit"]) is str and type(s["public"]) is str and type(s["sessions"]) is list
                    and s["sessions"] and all(type(x) is str for x in s["sessions"]) for s in statements)):
        return [], (LIVE, LIVE_STATEMENTS_FORM)
    partition = {"format": PARTITION_FORMAT, "program": hashlib.sha512(raw).hexdigest(),
                 "query": {"name": QUERY[0], "version": QUERY[1], "params": {"templates": [got["unit"]]}}}
    out: list[dict] = []
    seen: set[int] = set()
    for i, st in enumerate(statements):
        s = st["round"]
        if s >= got["S"] or s in seen:
            return out, (LIVE, f"round {s} has {'two live statements' if s in seen else 'a live statement'} and the "
                               f"window has {got['S']} rounds: each round's live@<s> is one statement, verified once")
        seen.add(s)
        reg = LR.reads_file(program, got["T"], got["S"], s, roots, window, session, ms[s], got["prefix"], max_tokens)
        out.append({"statement": i, "circuit_sha512": files[LIVE], "registered": reg, "partition": partition})
    return out, None


def _lean_live_verified(files: Any, entries: Any, run: Run) -> Lived:
    """`_live_verified`'s `lived` from the Lean verifier of record (`PouwCheck.LiveVerified.liveVerified`,
    `verifier.lean`), over the window's registered `entries`: the Lean names the verifier runs (`_live_asks`), this runs
    `run` on each in order until an answer the Lean will refuse at (an error, a refusal or a rejected session), and the
    Lean judges the answers. A Lean that can't run leaves the window refused (``live``)."""
    def lived(live: dict, raw: bytes, _got: dict, _roots: dict, ms: list[int], program: str, window: str,
              session: str, max_tokens: list[int]) -> tuple[list | None, tuple[str, str] | None]:
        statements = live["statements"]
        q = {"files": files, "statements": statements, "descriptor": raw.hex(), "entries": entries, "ms": ms,
             "program": program, "window": window, "session": session, "max_tokens": max_tokens, "runs": None}
        try:
            got = V.lean("credit-live-verified", q)
            if "run" in got:
                runs = []
                for a in got["run"]:
                    runs.append(_answer(run, statements[a["statement"]], a, raw))
                    if "findings" not in runs[-1] or not all(f["accepted"] for f in runs[-1]["findings"]):
                        break
                got = V.lean("credit-live-verified", {**q, "runs": runs})
        except V.LeanError as e:
            return None, (LIVE, f"the Lean verifier of record could not judge the window's live statements, so the "
                                f"window stays refused: {e}")
        return (None, tuple(got["refused"])) if "refused" in got else (got["outcomes"], None)
    return lived


def _live_runner(files: Any, entries: Any, run: Run) -> Lived:
    return _lean_live_verified(files, entries, run) if V.RECORD_VERDICT == "lean" else _live_verified(files, run)


Proved = Callable[[dict, bytes, list[int], int], tuple[Any, tuple[str, str] | None]]


def _live_outcomes(live: dict, *_: Any) -> tuple[Any, None]:
    """`_live`'s `lived` on the live evidence's own ``proofs``: the form the Lean reads."""
    return live["proofs"], None


def _window(key: Any, ledger: Any, pins: Any, main: Any, window: Any, session: Any, salt: Any, calls: Any, served: Any,
            commit: Any, rounds: Any, draw: Any, descriptor: Any, entries: Any, proved: Proved, live: Any, lived: Lived,
            keys: tuple[str, ...], k: Any, t: Any, verify: Ch.Verify
            ) -> tuple[tuple[list[int], dict, list[int]] | None, tuple[str, str] | None]:
    got, why = _credited(key, ledger, pins, main, window, session, salt, calls, served, commit, rounds, draw, verify)
    if why is not None:
        return None, why
    why = CC.rounds(key, ledger, pins, main, window, session, [x["root"] for x in rounds], [x["coin"] for x in rounds],
                    verify)
    if why is not None:
        return None, why
    if not (type(descriptor) is str and _HEX.fullmatch(descriptor)):
        return None, ("layout", DESCRIPTOR_HEX)
    layout, why = layout_of(bytes.fromhex(descriptor))
    if why is not None:
        return None, why
    if not (_nat(t) and t >= 1):
        return None, ("layout", ROWS_FORM)
    if max(layout["tile"][0], 1 << (t - 1).bit_length()) != layout["m"]:
        return None, ("layout", other_slot(t, layout["tile"][0], layout["m"]))
    program, prefix = hashlib.sha512(bytes.fromhex(descriptor)).hexdigest(), _prefix(bytes.fromhex(descriptor))
    if len(calls) != len(layout["calls"]):
        return None, ("manifest", f"the manifest's {len(calls)} rows are not the circuit's {len(layout['calls'])} "
                                  "calls: each call of the window's Program is one manifest row, in order")
    why = _a_link(rounds, entries, len(calls), table_rows(layout, len(rounds), prefix))
    if why is not None:
        return None, why
    (sid, period), why = _period_weights(key, ledger, window, rounds, layout, entries, program, prefix, verify)
    if why is not None:
        return None, why
    ss, _ = strata(layout)
    why = _arow(rounds, entries, commit_of(commit)[0][1], layout, prefix)
    if why is not None:
        return None, why
    rows, why = served_rows(calls, layout, len(rounds), t)
    if why is not None:
        return None, why
    max_tokens, why = committed_length(ledger, period)
    if why is None:
        why = _length(max_tokens, rows, layout, len(rounds))
    if why is not None:
        return None, why
    units, why = _window_drawn(window, draw, ss, k)
    proofs = None
    if why is None:
        proofs, why = proved(layout, bytes.fromhex(descriptor), units, len(rounds))
    if why is None:
        why = _proved(units, proofs)
    if why is None:
        why = _registration(window["root"], program, prefix, entries, ss, layout, len(rounds), proofs, sid)
    if why is None:
        why = _live(live, lived, keys, t, len(rounds), prefix, program, window["root"], sid, entries, rows, max_tokens)
    return (None, why) if why is not None else (([i for i in got if rows[i] > 0], layout, rows), None)


def _prefix(descriptor: bytes) -> str:
    """The ``PREFIX`` of the program function's record of a descriptor `layout_of` accepted."""
    return _loads(dict(_record(_loads(descriptor.decode())[1]["program_fn"]))["PREFIX"])[1]


def no_weights(name: str) -> str:
    return (f"period {name}'s open signs no weights' root ({WEIGHTS_FIELD}, a SHA-512 digest, the gate's --model): "
            "the period registers each K's B rows before its first window's salt and opens with their root")


def no_model(name: str) -> str:
    return (f"period {name}'s open signs a weights' root but no model ({MODEL_FIELD}, {{name, checkpoint_sha512}}), so "
            f"nothing says which checkpoint it is the root of: {MODEL_FIX}")


def unpinned_model(name: str, model: str) -> str:
    return f"period {name}'s open names model {model}, which is no checkpoint role credit pins: {MODEL_FIX}"


def other_checkpoint(name: str, model: str) -> str:
    return (f"period {name}'s open names model {model} with another checkpoint than the one credit pins for it: "
            f"{MODEL_FIX}")


def _period_weights(key: Any, ledger: Any, window: dict, rounds: list, layout: dict, entries: Any, program: str,
                    prefix: str, verify: Ch.Verify) -> tuple[tuple[str, str] | tuple[None, None], tuple[str, str] | None]:
    """The session the auditor's ledger opened for the window and the period holding it, and None, when the window's
    registered weights (`_weights`, over round 0's ``requests`` in its round log and that session) fold to the weights'
    root its period's open signed (`committed_weights`); else (None, None) and the refusal (``weights``, ``layout`` or
    ``coin``). A step of `window`, after `_credited` trusted the ledger."""
    try:
        audited = Ch.audited_session(ledger, window, Ch.ChallengerKey.from_json(key), verify)
    except Ch.CoinError as e:
        return (None, None), ("coin", f"the challenger's session is not trusted: {e}")
    if audited.period is None:
        return (None, None), ("weights", (f"the window is in no period of the auditor's ledger, so no period-open signed "
                                          f"the weights' root it opens ({WEIGHTS_FIELD}): reserve its session inside a "
                                          "period whose open signs it"))
    signed, why = committed_weights(ledger, audited.period)
    if why is not None:
        return (None, None), why
    requests = rounds[0].get("requests") if rounds else None
    w, why = _weights(window, layout, entries, requests, audited.session, program, prefix)
    if why is not None:
        return (None, None), why
    if w.hex() != signed:
        return (None, None), ("weights", (f"the window's registered weights' root {w.hex()[:16]}... is not "
                                          f"{signed[:16]}..., the one period {audited.period}'s open signed: a B row's "
                                          "codes or s_w differ from the ones the period registered before its first "
                                          "window's salt. Open every window of a period on its registered B rows"))
    return (audited.session, audited.period), None


def no_fixed_length(name: str) -> str:
    return (f"no-fixed-length: period {name}'s open commits no fixed length ({LENGTH_FIELD}, {{max_tokens, stop}}), so "
            "its windows are not of record: a row of record serves each request's public max_tokens whatever it "
            f"samples (ignore_eos; Daniel, 19 Oct, ASK-39). Open the period with the row's max_tokens and stop "
            f"{Wd.STOP_FIXED}")


def eos_keyed_row(name: str, stop: str) -> str:
    return (f"eos-keyed-row: period {name}'s open commits the stop mode {stop}, not {Wd.STOP_FIXED} (ignore_eos, "
            "circuits' public-cap done source): a row of record is done only at each request's public max_tokens, "
            "never at an EOS token")


def length_form(name: str) -> str:
    return (f"period {name}'s open commits a fixed length ({LENGTH_FIELD}) that is not {{max_tokens, stop}}, max_tokens "
            f"a non-empty list of positive integers (each request's public max_tokens, in admission order) and stop "
            f"{Wd.STOP_FIXED}")


def committed_length(ledger: list, name: str) -> tuple[list[int] | None, tuple[str, str] | None]:
    """Each request's public max_tokens period `name`'s open committed (`LENGTH_FIELD`, ``{max_tokens, stop}`` with stop
    `window.STOP_FIXED`) in the auditor's ledger `ledger`, one the challenger's checks have trusted, and None; or None
    and the refusal (``length``): an open with no fixed length (its windows are not of record), one whose stop mode is
    EOS-keyed, or a malformed one."""
    body = next((e["body"] for e in map(json.loads, ledger)
                 if e["kind"] == "period-open" and e["body"]["period"] == name), {})
    ln = body.get(LENGTH_FIELD)
    if ln is None:
        return None, ("length", no_fixed_length(name))
    if isinstance(ln, dict) and type(ln.get("stop")) is str and ln["stop"] != Wd.STOP_FIXED:
        return None, ("length", eos_keyed_row(name, ln["stop"]))
    m = ln.get("max_tokens") if isinstance(ln, dict) else None
    if not (isinstance(ln, dict) and sorted(ln) == list(LENGTH_KEYS) and ln["stop"] == Wd.STOP_FIXED
            and type(m) is list and m and all(type(n) is int and n >= 1 for n in m)):
        return None, ("length", length_form(name))
    return list(m), None


def _length(max_tokens: list[int], rows: list[int], layout: dict, steps: int) -> tuple[str, str] | None:
    """None when the window serves its period's fixed length, `max_tokens` each request's public max_tokens
    (`committed_length`), at its calls' served rows `rows` (`served_rows`': one m_s a step, fixed before the draw coin);
    else the refusal (``length``). The window's `layout` calls are its `steps` rounds' equal blocks (`_arow` held them
    so), every request is admitted in round 0, and a request of max_tokens n holds its prompt's rows in round 0 and one
    decode row in each round 1 … n − 1. So round 0 serves at least one row a request, and round s ≥ 1 serves exactly the
    requests whose max_tokens is past s. A round serving more rows than that ran a request past its public max_tokens,
    and one serving fewer stopped a request before it (an EOS-keyed done source, not the public cap)."""
    big_m, n = layout["m"], len(max_tokens)
    if n > big_m:
        return "length", (f"length-mismatch: the fixed length's {n} requests are more than the window's {big_m} slots, "
                          "so round 0 can't hold a row of each")
    q = max_tokens.index(max(max_tokens))
    if max_tokens[q] > steps:
        return "length", (f"length-mismatch: request {q}'s max_tokens {max_tokens[q]} is past the window's {steps} "
                          "rounds, so the window can't serve it to its public cap")
    per = len(rows) // steps
    if rows[0] < n:
        return "length", (f"length-mismatch: round 0 served {rows[0]} rows, fewer than the fixed length's {n} "
                          "requests, each admitted there with its prompt")
    for s in range(1, steps):
        got, want = rows[s * per], sum(1 for t in max_tokens if t > s)
        if got > want:
            return "length", (f"length-mismatch: round {s} served {got} rows, past the {want} the fixed length "
                              f"schedules there (one decode row for each request whose max_tokens is past {s}): a "
                              "request ran past its public max_tokens")
        if got < want:
            return "length", (f"length-mismatch: round {s} served {got} rows, short of the {want} the fixed length "
                              f"schedules there (one decode row for each request whose max_tokens is past {s}): a "
                              "request stopped before its public max_tokens, so its done source is not the public cap")
    return None


def window(key: Any, ledger: Any, pins: Any, main: Any, window: Any, session: Any, salt: Any, calls: Any, served: Any,
           commit: Any, rounds: Any, draw: Any, descriptor: Any, entries: Any, statements: Any, live: Any,
           circuit_files: Any, k: Any, rows: Any, verify: Ch.Verify, run: Run = PF.verify
           ) -> tuple[list[int] | None, tuple[str, str] | None]:
    """A served H1 window's credited calls and None, or None and the refusal of the first step that refuses, every step
    over the same evidence and the one round log `rounds`: `_window_outcomes`' steps, each drawn unit's proof outcome
    the verifier of record's own run over the window's `statements` (`_verified`: each call's proof files, on the
    circuit file whose SHA-512 `circuit_files` gives its K, ``{k<K>: SHA-512}``, the period-open's `CIRCUIT_FILES`, and
    the ``--registered`` file credit builds from the descriptor and the window's entries). The evidence carries proof
    files, never verdicts, and so does its `live` evidence (`LIVE_STATEMENTS`: the live Program's descriptor and each
    round's live@<s> proof files, `_live_verified`, on the circuit file `circuit_files`' ``live``). Both verdicts are
    the Lean's when `verifier.RECORD_VERDICT` is ``"lean"`` (`_lean_verified`, `_lean_live_verified`). `run` is the
    verifier of record (`proofs.verify`), the verifier's own, never the evidence's.

    `period_credit` ties `descriptor` to the circuit the window's period committed; a window counts only there."""
    got, why = _window(key, ledger, pins, main, window, session, salt, calls, served, commit, rounds, draw, descriptor,
                       entries, _runner(statements, circuit_files, entries, window, salt, run), live,
                       _live_runner(circuit_files, entries, run), LIVE_STATEMENTS, k, rows, verify)
    return (None, why) if got is None else (got[0], None)


def _runner(statements: Any, files: Any, entries: Any, window: Any, salt: Any, run: Run) -> Proved:
    if V.RECORD_VERDICT == "lean":
        return lambda _layout, raw, units, steps: _lean_verified(statements, files, raw, entries, window["root"], units,
                                                                 steps, run, salt)
    return lambda layout, raw, units, steps: _verified(statements, files, raw, layout, entries, window["root"], units,
                                                       steps, run, salt)


def _window_outcomes(key: Any, ledger: Any, pins: Any, main: Any, window: Any, session: Any, salt: Any, calls: Any,
                     served: Any, commit: Any, rounds: Any, draw: Any, descriptor: Any, entries: Any, proofs: Any,
                     live: Any, k: Any, rows: Any, verify: Ch.Verify) -> tuple[list[int] | None, tuple[str, str] | None]:
    """`window` on each drawn unit's proof outcome `proofs` in place of its statements: the form the Lean
    `PouwCheck.Credit.window` reads until it runs the verifier of record itself, which the difftest holds it to. A
    served H1 window's credited calls and None, or None and the refusal of the first step that refuses, every step
    over the same evidence and the one round log `rounds`:

    1. `_credited`: the credit bits fixed before the draw coin, read through the challenger's session, one the host's
       pins log `pins` records as run from a commit the verifier's copy of main's first-parent history `main` holds
       (`challenger.pinned_session`);
    2. `core_check.rounds`: the round log's roots are the ones the challenger's log shows it received, in order, each
       round's coin (its ``coin``) opened once after its root;
    3. `_a_link`: each A value among the window's registered `entries` is the tree over the segments those round roots
       commit;
    4. the layout of its circuit (`layout_of` of `descriptor`, the bytes in hex of the circuit its proofs were verified
       against), whose A slot M is `h1window.slot` of the T table rows `rows` its period's open committed
       (`committed_rows`, ``layout``);
    5. `_weights`, its period's: the window's registered ``<PREFIX>b/<K>`` (``PREFIX`` the descriptor's) are each K's
       B rows once, binding no window, and their `weights_root` is the one the open of the period holding the window
       signed (`committed_weights`), and with round 0's ``requests`` in `rounds` and the session the auditor's ledger
       opened for the window it is the window's open root (`salt.window_open_root`). So every window of a period opens
       the B rows, codes and s_w, the period registered before its first window's salt, and a window of no period, or
       of one whose open signs no weights' root, is refused before any of its proofs is read;
    6. the manifest one row per call of the layout (``manifest``), `_arow` (each round's A tables and its
       input-ids table the Commit record `commit`'s value of the same rows), `served_rows` (each call's served rows
       m_s at most the period's T table rows `rows`, one m_s a step, ``manifest``; a credited call with m_s = 0 is no
       credit), `_length` (each step's m_s the fixed length the period's open committed, `committed_length`), and
       `_outcome`'s rule on the draw coin
       `_credited` read, at its period's committed K `k`, under the served window's
       arguments: the stream is the window's root (``window.root``, the root the auditor's ledger opens its session
       for), the registration its draw root (``draw.root``, the one root the draw coin followed) and the receipt
       ``{"receipts": draw.receipts}``, the last round's receipts. Each of `proofs` is the verifier of record's verdict
       for a drawn unit's session (`PROOF_KEYS`);
    7. `_registration`: the `entries` are the window's own registration's, their domains the ones its open root
       (``window.root``) and the descriptor's SHA-512 give them, and each drawn unit's session was verified against
       that descriptor and opened its registered values at their registered roots, its K's A value and its K's
       ``<PREFIX>b/<K>`` among them;
    8. `_live`: the window's `live` evidence (`LIVE_KEYS`) is its live Program's descriptor, at the period's T, its S
       rounds and its PREFIX (`live_of`), and each round's one accepted proof outcome, verified against it, opened round
       s's slot table and round 0's request table at their registered roots, ``m@<s>`` at the manifest's m_s,
       ``round@<s>`` at s, ``caps`` at the max_tokens its period's open committed and the ``live-ok`` rows: each step's
       served rows are proved, each served by a request of the fixed length within its public max_tokens."""
    got, why = _window(key, ledger, pins, main, window, session, salt, calls, served, commit, rounds, draw, descriptor,
                       entries, lambda *_: (proofs, None), live, _live_outcomes, LIVE_KEYS, k, rows, verify)
    return (None, why) if got is None else (got[0], None)


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


def committed_k(ledger: list, name: str) -> tuple[int | None, tuple[str, str] | None]:
    """The K period `name`'s open committed (`K_FIELD`, a positive integer at most `MAX_K`) in the auditor's ledger
    `ledger`, one `period` has trusted, and None; or None and the refusal (``budget``)."""
    body = next(e["body"] for e in map(json.loads, ledger)
                if e["kind"] == "period-open" and e["body"]["period"] == name)
    k = body.get(K_FIELD)
    if not (type(k) is int and 1 <= k <= MAX_K):
        return None, ("budget", f"period {name}'s open commits no draw K ({K_FIELD}, a positive integer at most 2^24), "
                                "so its windows have no K to be drawn with: the proof service commits K in the "
                                "period-open, beside its circuit and cap, before any draw")
    return k, None


def committed_rows(ledger: list, name: str) -> tuple[int | None, tuple[str, str] | None]:
    """The table rows T period `name`'s open committed (`ROWS_FIELD`, a positive integer: each round's request and slot
    tables' leaves, the pouw setting ``rows``) in the auditor's ledger `ledger`, one `period` has trusted, and None; or
    None and the refusal (``layout``). Every window of the period has the A slot `h1window.slot` of it, and each of its
    steps serves at most T rows (`served_rows`)."""
    body = next(e["body"] for e in map(json.loads, ledger)
                if e["kind"] == "period-open" and e["body"]["period"] == name)
    t = body.get(ROWS_FIELD)
    if not (_nat(t) and t >= 1):
        return None, ("layout", no_rows(name))
    return t, None


def committed_weights(ledger: list, name: str) -> tuple[str | None, tuple[str, str] | None]:
    """The weights' root period `name`'s open signed (`WEIGHTS_FIELD`, a SHA-512 digest: `weights_root` of each K's
    registered ``b/<K>``) in the auditor's ledger `ledger`, one the challenger's checks have trusted, and None; or None
    and the refusal (``weights``). The open names the model that root is of (`MODEL_FIELD`), a role `PINNED_MODELS`
    pins with the same checkpoint, so an open with a bare root, as the gate signed before its model registry, counts
    nothing."""
    body = next(e["body"] for e in map(json.loads, ledger)
                if e["kind"] == "period-open" and e["body"]["period"] == name)
    w, m = body.get(WEIGHTS_FIELD), body.get(MODEL_FIELD)
    if _digest(w) is None:
        return None, ("weights", no_weights(name))
    if not (isinstance(m, dict) and tuple(sorted(m)) == MODEL_KEYS and type(m["name"]) is str):
        return None, ("weights", no_model(name))
    if m["name"] not in PINNED_MODELS:
        return None, ("weights", unpinned_model(name, m["name"]))
    if m["checkpoint_sha512"] != PINNED_MODELS[m["name"]]:
        return None, ("weights", other_checkpoint(name, m["name"]))
    return w, None


def committed_files(ledger: list, name: str) -> tuple[dict | None, tuple[str, str] | None]:
    """The circuit files period `name`'s open committed (`CIRCUIT_FILES`, ``{k<K>: SHA-512}``: each K's template's
    circuit file, as the auditor's lowering built it) in the auditor's ledger `ledger`, one `period` has trusted, and
    None; or None and the refusal (``circuit``). The Python reference of the Lean's `PouwCheck.CommittedFiles`
    (`_lean_committed_files`), which `period_credit` reads when `verifier.RECORD_VERDICT` is ``"lean"``."""
    body = next(e["body"] for e in map(json.loads, ledger)
                if e["kind"] == "period-open" and e["body"]["period"] == name)
    files = body.get(CIRCUIT_FILES)
    if not (isinstance(files, dict) and files
            and all(type(x) is str and _TEMPLATE.fullmatch(x) and _digest(v) is not None for x, v in files.items())):
        return None, ("circuit", f"period {name}'s open commits no circuit files ({CIRCUIT_FILES}, each K's template "
                                 f"k<K>, and the live Program's template {LIVE}, to the SHA-512 of the circuit file "
                                 "the auditor's lowering built for it), so no "
                                 "window's statements can be verified: commit them in the period-open, beside its "
                                 "circuit and K")
    return files, None


def _lean_committed_files(ledger: list, name: str) -> tuple[dict | None, tuple[str, str] | None]:
    """`committed_files` from the Lean verifier of record (`PouwCheck.CommittedFiles.committedFiles`, `verifier.lean`).
    A Lean that can't run leaves the period refused (``circuit``)."""
    try:
        got = V.lean("credit-committed-files", {"ledger": ledger, "name": name})
    except V.LeanError as e:
        return None, ("circuit", f"the Lean verifier of record could not read period {name}'s committed circuit files, "
                                 f"so the period earns nothing: {e}")
    return (None, tuple(got["refused"])) if "refused" in got else (got["files"], None)


def committed_pin(ledger: list, name: str) -> str | None:
    """The commit period `name`'s open recorded it opened under (`PERIOD_PIN_FIELD`, a commit string), or None when its
    open records none: `_period_of` holds every window's session to within the freshness floor of it on main's
    first-parent line (`_pin_hygiene`).  Optional, so a period-open that records none keeps main's behavior of holding
    each window only to the pins log and main (`challenger.credited_session`)."""
    body = next(e["body"] for e in map(json.loads, ledger)
                if e["kind"] == "period-open" and e["body"]["period"] == name)
    pin = body.get(PERIOD_PIN_FIELD)
    return pin if type(pin) is str else None


def committed_floor(ledger: list, name: str) -> int | None:
    """The freshness floor period `name`'s open recorded (`STALE_FIELD`, a nonnegative integer), or None when its open
    records none: `_period_of` counts no window whose pin is more than this many first-parent commits behind the period's
    open pin (`committed_pin`), default 0 when the open pins but records no floor.  Optional, like `committed_pin`."""
    body = next(e["body"] for e in map(json.loads, ledger)
                if e["kind"] == "period-open" and e["body"]["period"] == name)
    floor = body.get(STALE_FIELD)
    return floor if type(floor) is int and floor >= 0 else None


def _main_index(main: Any, commit: Any) -> int | None:
    """`commit`'s position in the verifier's copy of main's first-parent history `main` (newest first, so position 0 is
    main's head and position p is p commits behind it), or None when `main` doesn't hold it."""
    return main.index(commit) if isinstance(main, list) and commit in main else None


def _pin_hygiene(name: str, i: int, main: Any, pinned: str, period_pin: str | None, floor: int | None,
                 unit: str = "window") -> tuple[str, str] | None:
    """Why window `i` of period `name`, whose session ran commit `pinned` on main's first-parent line `main`, is too old
    for the period: its pin more than the period's freshness floor of first-parent commits behind the commit the period's
    open recorded (`period_pin`, `committed_pin`; `floor`, `committed_floor`, default 0 when the open pins but records no
    floor), measured from that pin and never from main's head, so main moving on never refuses a window fresh at its open;
    None when it is recent enough or the open records no pin.  The per-window pin and circuit are
    `challenger.credited_session`'s; this is the period-level hygiene on top (CMP74-FOLLOW).  `unit` names what `i`
    counts (``pass`` in a Pearl-C period, `pearl_c_period_credit`)."""
    if period_pin is None:
        return None
    pos = _main_index(main, pinned)
    if pos is None:
        return None
    base = _main_index(main, period_pin)
    if base is None:
        return ("pin", f"period {name}'s open pins commit {period_pin[:16]}..., which is not on main's first-parent "
                       f"line in the verifier's copy ({PERIOD_PIN_FIELD}): open the period under a commit "
                       "git rev-list --first-parent origin/main holds")
    f = floor if floor is not None else 0
    if pos > base + f:
        return ("pin", f"{unit} {i} of period {name} is pinned {pos - base} first-parent commit(s) before the commit "
                       f"its period opened under, past its freshness floor of {f} ({STALE_FIELD}): {pinned[:16]}... is "
                       f"that far behind {period_pin[:16]}... on main's first-parent line, so the period earns nothing; "
                       "open every window within the floor of its period's open pin")
    return None


CALL_RANGE_CODE, REPEATED_CALL = "call-range", "repeated-call"


def passes_distinct(passes: Any) -> tuple[str, str] | None:
    """Why the Pearl-C pass manifests `passes`, credited together, are refused, or None: every call index below
    `protocol.CALL_RANGE` (``call-range``), and no index held by two passes of one epoch, nor twice by one pass
    (``repeated-call``). The epoch is the manifest's salt, since -h3's row seed binds the salt and the index but no pass,
    so a repeated index under one salt and root_B repeats its seeds, noise and tickets and would be credited twice
    (`Properties.Pouw.LiveSalt.PassesDistinct`). A pass's calls are its manifest's ``matmuls``, each an
    ``index``."""
    seen: dict[str, dict[int, int]] = {}
    for p, m in enumerate(passes):
        salt = m.get("salt") if isinstance(m, dict) else None
        us = m.get("matmuls") if isinstance(m, dict) else None
        if not isinstance(salt, str) or not isinstance(us, list) or not all(isinstance(u, dict) for u in us):
            return ("manifest", f"pass {p} is not a Pearl-C manifest with a string salt and a list of matmuls")
        held = seen.setdefault(salt, {})
        for u in us:
            i = u.get("index")
            if type(i) is not int or not 0 <= i < CALL_RANGE:
                return (CALL_RANGE_CODE, f"pass {p} holds call index {i!r}, outside [0, 2^32) where -h3's row seed "
                                         "packs it: end the epoch (a new salt) before its 2^32nd call")
            q = held.get(i)
            if q is not None:
                both = f"pass {p} holds call index {i} twice" if q == p else \
                    f"passes {q} and {p} both hold call index {i}"
                return (REPEATED_CALL, f"{both} under salt {salt[:16]}...: one epoch's calls repeat their seeds and "
                                       "tickets, so the device numbers calls across the whole epoch and no index "
                                       "is credited twice")
            held[i] = p
    return None


#: a Pearl-C pass's evidence in its period (`pearl_c_period_credit`)
PEARL_C_KEYS = ("context", "descriptor", "manifest", "session")
PEARL_C_FORM = ("the period's passes are not a nonempty list of each pass's evidence {context, descriptor, manifest, "
                "session}: the window context naming its manifest by SHA-512, the descriptor in lowercase hex of the "
                "circuit it proves, its manifest's bytes in lowercase hex and its salt epoch's session's files")
EARLIER_FORM = ("the lane's earlier credited Pearl-C epochs are not a list of [weights' root, k] pairs: a SHA-512 digest "
                "(128 lowercase hex) and an integer in [0, 2^64)")
SALT_ONLY, PASS_REFUSED, INDEX = "salt-only", "pass", "index"


def pearl_c_period_credit(key: Any, ledger: Any, pins: Any, main: Any, name: Any, passes: Any, earlier: Any,
                          verify: Ch.Verify) -> tuple[dict | None, tuple[str, str] | None]:
    """Pearl-C period `name`'s credit and None, or None and the refusal (the Pearl-C period rule). A
    live Pearl-C pass earns only inside a Pearl-C period, and the period earns only when every clause holds, each
    refused with its own code:

    1. **Exact match** (`period`: ``coin``, ``period``, ``missing``, ``order``). The presented passes' salt epochs'
       roots R = `salt.epoch_open_root` (W, k), of each manifest's weights' root and ``pass_index``
       (`core_check.live_epoch`) and in the order they first appear, are exactly the windows the close of period `name`
       lists, in its order, and the close is complete. So every session the period holds has a pass presented with it,
       and a complete salt-only session no presented pass names (an abandoned one) refuses the period.
    2. **Salt-only sessions** (``salt-only``). Each pass's session is a Pearl-C salt session: `core_check.live_salt`
       refuses none with ``salt`` (its commitment and reserve the salt coin alone, ended complete having released it,
       its log showing R). H1's window checks refuse such a session (`window.live_epoch_reason`), so the two never
       share a period.
    3. **Credited sessions** (``circuit``). Each pass's session passes `challenger.credited_session`: run from a commit
       the host's pins log `pins` records on main's first-parent line `main`, and the pass's ``descriptor`` hashing to
       the circuit the period's open committed.
    3a. **Fresh pins** (``pin``). Each pass's session is pinned within the period's freshness floor of the commit its
       open recorded, H1's period-level pin hygiene (`committed_pin`, `committed_floor`, `_pin_hygiene`): a session
       pinned more than the floor of first-parent commits behind the period's open pin refuses it.
    4. **Accepted passes** (``pass``). Each pass's ``context`` is a string, and the pass passes `core_check.pass_epoch`
       under the verifier's live inputs (``{key, ledger, main, pins, session}``), salted by its live coin, its
       ``context`` naming its manifest.
    5. **Consecutive epochs** (``index``). For each W, the period's epochs on W, in the close's order, are numbered
       n_W, n_W + 1, … with no gap or repeat, where n_W is one more than the largest k on W in `earlier` (the lane's
       epochs credited in earlier Pearl-C periods, ``[W, k]`` pairs) or 0 when it has none. A W whose numbering broke,
       since one of its roots was opened outside a credited period, is spent: the prover re-keys hm96 for a new W.
    6. **Distinct calls** (`passes_distinct`: ``call-range``, ``repeated-call``). No two passes of one epoch hold one
       call index, and none is 2^32 or more.

    The credit is ``{epochs, passes}``: the period's epochs as ``[W, k]`` (W hex) in the close's order, which the
    lane's next Pearl-C period takes in its `earlier`, and the credited passes' manifests in the order presented. The
    challenger's signatures are checked with `verify`."""
    if not (type(passes) is list and passes
            and all(isinstance(p, dict) and sorted(p) == list(PEARL_C_KEYS) for p in passes)):
        return None, ("period", PEARL_C_FORM)
    if not (type(earlier) is list and all(type(e) is list and len(e) == 2 and _digest(e[0]) is not None
                                          and _nat(e[1]) and e[1] < 1 << 64 for e in earlier)):
        return None, (INDEX, EARLIER_FORM)
    if not all(type(p["manifest"]) is str and _HEX.fullmatch(p["manifest"]) for p in passes):
        return None, ("period", PEARL_C_FORM)
    manifests, epochs = [], []
    for i, p in enumerate(passes):
        try:
            m, w, k = CC.live_epoch(bytes.fromhex(p["manifest"]))
        except (CC.Refused, Co.DomainError) as e:
            return None, (PASS_REFUSED, f"pass {i} of period {name} names no salt epoch, so the period earns nothing: {e}")
        manifests.append(m)
        epochs.append((w, k, PS.epoch_open_root(w, k).hex()))
    roots = list(dict.fromkeys(r for _, _, r in epochs))
    why = period(key, ledger, name, roots, verify)
    if why is not None:
        return None, why
    ck, audited = Ch.ChallengerKey.from_json(key), []
    for i, (p, m) in enumerate(zip(passes, manifests)):
        _, why = CC.live_salt(key, ledger, pins, main, p["session"], m, verify)
        if why is not None and why[0] == "salt":
            return None, (SALT_ONLY, f"pass {i} of period {name} is not of a Pearl-C salt session, so the period earns "
                                     f"nothing: {why[1]}")
    for i, (p, (_, _, r)) in enumerate(zip(passes, epochs)):
        d = bytes.fromhex(p["descriptor"]) if type(p["descriptor"]) is str and _HEX.fullmatch(p["descriptor"]) else None
        if d is None:
            return None, ("circuit", f"pass {i} of period {name}: the descriptor is not the bytes in lowercase hex of "
                                     "the circuit it proves")
        try:
            audited.append(Ch.credited_session(ledger, {"root": r}, ck, pins, main, d, verify))
        except Ch.CoinError as e:
            return None, ("circuit", f"pass {i} of period {name} doesn't count toward it, so the period earns "
                                     f"nothing: {e}")
    period_pin, floor = committed_pin(ledger, name), committed_floor(ledger, name)
    for i, a in enumerate(audited):
        why = _pin_hygiene(name, i, main, a.pinned, period_pin, floor, "pass")
        if why is not None:
            return None, why
    for i, p in enumerate(passes):
        if type(p["context"]) is not str:
            return None, (PASS_REFUSED, f"pass {i} of period {name} is refused, so the period earns nothing: its "
                                        "context is not a string, the window context naming its manifest by SHA-512")
        live = {"key": key, "ledger": ledger, "main": main, "pins": pins, "session": p["session"]}
        _, why = CC.pass_epoch(None, bytes.fromhex(p["manifest"]), p["context"], live=live, verify=verify)
        if why is not None:
            return None, (PASS_REFUSED, f"pass {i} of period {name} is refused, so the period earns nothing: {why[0]}: "
                                        f"{why[1]}")
    nxt: dict[str, int] = {}
    for w, k in earlier:
        nxt[w] = max(nxt.get(w, 0), k + 1)
    out = []
    for r in roots:
        w, k = next((w.hex(), k) for w, k, x in epochs if x == r)
        want = nxt.get(w, 0)
        if k != want:
            return None, (INDEX, f"period {name}'s epoch {k} over weights' root {w[:16]}... is not epoch {want}: a "
                                 "weights' root's salt epochs are numbered on from its lane's earlier credited periods "
                                 "with no gap or repeat, so a root whose numbering broke is spent: re-key hm96 for a new "
                                 "weights' root, its epochs numbered from 0")
        nxt[w] = k + 1
        out.append([w, k])
    why = passes_distinct(manifests)
    if why is not None:
        return None, why
    return {"epochs": out, "passes": manifests}, None


def period_credit(key: Any, ledger: Any, pins: Any, main: Any, name: Any, windows: Any, verify: Ch.Verify,
                  run: Run = PF.verify) -> tuple[dict | None, tuple[str, str] | None]:
    """Period `name`'s credit and None, or None and the refusal: `_period_outcomes`' rules, each window's evidence
    (`PERIOD_KEYS`) carrying its statements' proof files, which `window` verifies on the circuit files the period's
    open committed (`committed_files`, else ``circuit``) with the verifier of record `run`."""
    return _period_of(key, ledger, pins, main, name, windows, verify, run)


def _period_outcomes(key: Any, ledger: Any, pins: Any, main: Any, name: Any, windows: Any, verify: Ch.Verify
                     ) -> tuple[dict | None, tuple[str, str] | None]:
    """`period_credit` on each window's drawn units' proof outcomes (`PROVED_KEYS`) in place of its statements, by
    `_window_outcomes`: the form the Lean `PouwCheck.Credit.periodCredit` reads until it runs the verifier of record
    itself. Period `name`'s credit and None, or None and the refusal. The credit is ``{windows, work, k, delta, eps}``:
    each window's credited calls in the close's order (a call with m_s = 0 served rows is none), each window's credited
    served work (`credited_work`, m_s·n·K a credited call, never its M slot rows'), the K the period's open committed
    (`committed_k`), the δ the ε is reported at (`DELTA`), and each window's ε, what its draw of K established over its
    credited work (`established`; null when it establishes no ε <= 1). The period's windows are exactly its close's (`period`), its open committed K
    (``budget``) and the table rows T (`committed_rows`, ``layout``) and signed the weights' root (`committed_weights`, ``weights``) every window's registered B rows fold
    to, each window's session ran a commit the host's pins log `pins` records on the verifier's copy of main's
    first-parent history `main` and the window proves the circuit the period's open committed (``circuit``: the pin
    first, then its ``descriptor``, the bytes in hex of the circuit its proofs were verified against, hashing to it,
    `challenger.credited_session`), and each window's evidence (``{window, session, salt, calls, commit, descriptor,
    entries, served, rounds, draw, proofs}``, as `_window_outcomes` reads it, under the same ledger) is credited by it,
    every step of it, drawn at K over the layout of that same descriptor. Each window's session is also pinned within the
    period's freshness floor (`committed_floor`, first-parent commits, default 0) of the commit the period's open
    recorded (`committed_pin`), counted from that pin and never from main's head, both optional and refused with ``pin``;
    a refused window refuses the whole period (``circuit``, ``pin`` or ``window``), so a period earns only when every
    window in it does."""
    return _period_of(key, ledger, pins, main, name, windows, verify, None)


def _period_of(key: Any, ledger: Any, pins: Any, main: Any, name: Any, windows: Any, verify: Ch.Verify,
               run: Run | None) -> tuple[dict | None, tuple[str, str] | None]:
    keys, form = (PROVED_KEYS, WINDOWS_FORM) if run is None else (PERIOD_KEYS, PERIOD_FORM)
    if not (type(windows) is list and all(isinstance(w, dict) and sorted(w) == list(keys) for w in windows)):
        return None, ("period", form)
    roots = [w["window"].get("root") if isinstance(w["window"], dict) else None for w in windows]
    why = period(key, ledger, name, roots, verify)
    if why is not None:
        return None, why
    big_k, why = committed_k(ledger, name)
    if why is None:
        t, why = committed_rows(ledger, name)
    if why is None:
        _, why = committed_weights(ledger, name)
    if why is not None:
        return None, why
    period_pin, floor = committed_pin(ledger, name), committed_floor(ledger, name)
    files = None
    if run is not None:
        files, why = (_lean_committed_files if V.RECORD_VERDICT == "lean" else committed_files)(ledger, name)
        if why is not None:
            return None, why
    k = Ch.ChallengerKey.from_json(key)
    out, work, eps = [], [], []
    for i, w in enumerate(windows):
        if not (type(w["descriptor"]) is str and _HEX.fullmatch(w["descriptor"])):
            return None, ("circuit", f"window {i} of period {name}: {DESCRIPTOR_HEX}")
        try:
            audited = Ch.credited_session(ledger, w["window"], k, pins, main, bytes.fromhex(w["descriptor"]), verify)
        except Ch.CoinError as e:
            return None, ("circuit", f"window {i} of period {name} doesn't count toward it, so the period earns "
                                     f"nothing: {e}")
        why = _pin_hygiene(name, i, main, audited.pinned, period_pin, floor)
        if why is not None:
            return None, why
        proved = ((lambda *_, p=w["proofs"]: (p, None)) if run is None
                  else _runner(w["statements"], files, w["entries"], w["window"], w["salt"], run))
        lived, lk = ((_live_outcomes, LIVE_KEYS) if run is None
                     else (_live_runner(files, w["entries"], run), LIVE_STATEMENTS))
        got, why = _window(key, ledger, pins, main, w["window"], w["session"], w["salt"], w["calls"], w["served"],
                           w["commit"], w["rounds"], w["draw"], w["descriptor"], w["entries"], proved, w["live"], lived,
                           lk, big_k, t, verify)
        if why is not None:
            return None, ("window", f"window {i} of period {name} is refused, so the period earns nothing: {why[0]}: "
                                    f"{why[1]}")
        out.append(got[0])
        work.append(credited_work(got[1], got[0], got[2]))
        e = established(got[1], got[0], got[2], big_k)
        eps.append(None if e is None else str(e))
    return {"windows": out, "work": work, "k": big_k, "delta": str(DELTA), "eps": eps}, None


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


def a_value(value: Any) -> tuple[str, str | None] | None:
    """A registered A value's contraction and round, as their digits: a name whose last two parts are ``a`` and either a
    contraction (``…/a/<K>``, the window's A rows of K, `h1window.roles`), its round None, or a contraction and a round
    (``…/a/<K>@<s>``, that round's A table of K, `h1window.table`); None for any other name.
    K is a canonical decimal, no leading zero."""
    parts = value.split("/") if isinstance(value, str) else []
    m = _A.fullmatch(parts[-1]) if len(parts) >= 2 and parts[-2] == "a" else None
    return None if m is None else (m[1], m[2])


def is_a(value: Any) -> bool:
    """A registered A value (`a_value`): ``…/a/<K>`` or a round's ``…/a/<K>@<s>``."""
    return a_value(value) is not None


def ids_step(value: Any) -> int | None:
    """An input-ids table's round: a name whose last part is ``ids@<s>`` (``<prefix>ids@<s>``, slot i's input token id
    at round s, s a canonical decimal); None for any other name."""
    m = _IDS.fullmatch(value.split("/")[-1]) if isinstance(value, str) else None
    return None if m is None else int(m[1])


def slots_step(value: Any) -> int | None:
    """A slot table's round: a name whose last part is ``slots@<s>`` (``<prefix>slots@<s>``, `circuit.live`'s: slot i's
    request at round s, the live unit's ``slot`` port, s a canonical decimal); None for any other name."""
    m = _SLOTS.fullmatch(value.split("/")[-1]) if isinstance(value, str) else None
    return None if m is None else int(m[1])


def is_requests(value: Any) -> bool:
    """A request table: a name whose last part is ``requests@0`` (`REQUESTS_TABLE`, ``<prefix>requests@0``, request q's
    header at row q, round 0's)."""
    return isinstance(value, str) and value.split("/")[-1] == REQUESTS_TABLE


def receipt_call(value: Any) -> int | None:
    """A receipt table's call: a name whose last part is ``receipt@<c>`` (``<prefix>receipt@<c>``, `h1window.table`'s:
    row t call c's t-th unit's receipt, c a canonical decimal); None for any other name."""
    m = _RECEIPT.fullmatch(value.split("/")[-1]) if isinstance(value, str) else None
    return None if m is None else int(m[1])


def pub_call(value: Any) -> int | None:
    """A public table's call: a name whose last part is ``pub@<c>`` (``<prefix>pub@<c>``, `h1window.table`'s
    `h1window.PUBS`: row t call c's t-th unit's public row, which `registered_file` builds), c a canonical decimal; None
    for any other name."""
    m = _PUB.fullmatch(value.split("/")[-1]) if isinstance(value, str) else None
    return None if m is None else int(m[1])


def is_tied(value: Any) -> bool:
    """A value the round roots commit (`_a_link`): an A value (`is_a`), an input-ids table (`ids_step`), a slot table
    (`slots_step`) or a receipt table (`receipt_call`)."""
    return (is_a(value) or ids_step(value) is not None or slots_step(value) is not None
            or receipt_call(value) is not None)


def _what(value: str) -> str:
    return ("A value" if is_a(value) else "input-ids table" if ids_step(value) is not None
            else "slot table" if slots_step(value) is not None else "receipt table")


def is_held(value: Any) -> bool:
    """A value held across windows: the weights' rows, a name whose last two parts are ``b`` and a contraction
    (``…/b/<K>``, K a canonical decimal, `h1window.roles`). Every other value a window registers, its A rows and its calls' A and z tables
    among them, is the window's own."""
    parts = value.split("/") if isinstance(value, str) else []
    return len(parts) >= 2 and parts[-2] == "b" and bool(_DIGITS.fullmatch(parts[-1]))


def _a_link(rounds: Any, entries: Any, calls: int = 0, rows: dict[str, int] | None = None) -> tuple[str, str] | None:
    """None when every registered A value among `entries` (the window's registered entries, ``{value, root, leaves,
    domain, …}``) is the tree over the segments the round log `rounds` commits (each round's ``root``, the one the
    challenger received, over its ``inner`` root and its segments ``a``, `round_root`); else the refusal (``a-link``,
    and why). Of S rounds, round s's segments of a window's A value ``…/a/<K>`` of L rows tile its rows
    ``[s·L/S, (s+1)·L/S)``; round s's A table of K ``…/a/<K>@<s>`` is committed whole by round s, whose segments tile
    all its rows, and no other round lists it. Round s's input-ids table ``…/ids@<s>`` (`ids_step`) is committed whole
    by round s alike, as is its slot table ``…/slots@<s>`` (`slots_step`), and so is call c's receipt table
    ``…/receipt@<c>`` (`receipt_call`) by c's round, of the window's `calls` calls (its manifest's rows; 0, the default,
    admits no receipt table) the rounds' equal blocks. Each entry's leaf count is its rows in `rows` (`table_rows`, the
    layout's) when `rows` names it, else from 1 to `MAX_ROWS`, and every entry's count is checked before the verifier
    hashes any pad. A step of `window`, which first holds the round roots to the ones the challenger received
    (`core_check.rounds`) and `layout_of` the layout."""
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
    a = [e for e in entries if is_tied(e.get("value"))]
    if not any(is_a(e["value"]) for e in a):
        return "a-link", NO_A
    names: list[str] = []
    for e in a:
        if e["value"] in names:
            return "a-link", f"the window registers {_what(e['value'])} {e['value']} twice"
        names.append(e["value"])
    for r, p in enumerate(segs):
        for v in sorted(p):
            if v not in names:
                return "a-link", (f"round {r} lists segments of {v}, which is no registered A value, input-ids table "
                                  "or receipt table of the window")
    steps, rows, plan = len(rounds), rows or {}, []
    for e in a:
        v, root, domain, count = e["value"], _digest(e.get("root")), _digest(e.get("domain")), e.get("leaves")
        if root is None or domain is None or not _nat(count):
            return "a-link", f"registered {_what(v)} {v}'s entry has no root, domain or leaf count"
        if v in rows and count != rows[v]:
            return "a-link", (f"registered {_what(v)} {v} has {count} rows, not the {rows[v]} the window's layout gives "
                              "it (credit.table_rows: each K's blocks or round's calls of M rows, each row K's chunks' "
                              "leaves, or each call's units)")
        if v not in rows and not 1 <= count <= MAX_ROWS:
            return "a-link", (f"registered {_what(v)} {v}'s leaf count {count} is not from 1 to {MAX_ROWS} (the verifier "
                              "hashes its pads)")
        step, slots, call = ids_step(v), slots_step(v), receipt_call(v)
        at = None if step is not None or slots is not None or call is not None else a_value(v)[1]
        if call is not None:
            if not (calls % steps == 0 and call < calls):
                return "a-link", (f"registered receipt table {v} is call {call}'s, and the window's {calls} calls are "
                                  f"not its {steps} rounds' equal blocks with a call {call}")
            own = call // (calls // steps)
            per, whose = count, f"call {call}'s receipt table, its round {own}'s"
        elif step is not None:
            if step >= steps:
                return "a-link", f"registered input-ids table {v} is round {step}'s, and the window has {steps} rounds"
            per, own, whose = count, step, f"round {step}'s input-ids table"
        elif slots is not None:
            if slots >= steps:
                return "a-link", f"registered slot table {v} is round {slots}'s, and the window has {steps} rounds"
            per, own, whose = count, slots, f"round {slots}'s slot table"
        elif at is None:
            if count % steps:
                return "a-link", f"registered A value {v}'s {count} rows are not the window's {steps} rounds' equal blocks"
            per, own, whose = count // steps, None, ""
        else:
            if int(at) >= steps:
                return "a-link", f"registered A value {v} is round {int(at)}'s A table, and the window has {steps} rounds"
            per, own, whose = count, int(at), f"round {int(at)}'s A table"
        plan.append((v, root, domain, count, per, own, whose))
    for v, root, domain, count, per, own, whose in plan:
        have = {}
        for r, p in enumerate(segs):
            if own is not None and r != own:
                if p.get(v):
                    return "a-link", f"round {r} lists segments of {v}, {whose}, which only round {own} commits"
                continue
            at, end = (r * per, (r + 1) * per) if own is None else (0, count)
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
            return "a-link", (f"registered {_what(v)} {v}'s root is not the tree over the segments the round roots "
                              "commit: its rows are not the ones fixed under each round's root before that round's coin. "
                              f"Register the {_what(v)} whose rows each round's root committed")
    return None


WINDOW_ROOT_FORM = "the window is not {root}, its open root a SHA-512 digest (128 lowercase hex)"
REQUESTS_FORM = "the window's round-0 request table's root is not a SHA-512 digest (128 lowercase hex)"
SESSION_FORM = ("the window's session is not the id of the session the auditor's ledger opened for it (1 to 64 of "
                "[A-Za-z0-9._:-], the first alphanumeric)")
CIRCUIT_FORM = ("the window's circuit is not its descriptor's SHA-512 (128 lowercase hex) and its PREFIX (a string of "
                "printable ASCII but \" and \\)")
OTHER_WEIGHTS = ("the window's registered weights are not the ones its open root binds: weights_root of each K's …/b/<K> "
                 "root, its round-0 request table's root and its session are not its open root (salt.window_open_root), "
                 "so a B row, its codes or its s_w, is not the one registered before the window's salt. Register each "
                 "K's B rows once, before the window opens, and open them at that root")


def other_weights_entry(v: str, k: int, n: int) -> str:
    x = chunks(k)
    return (f"registered weights {v}'s entry is not the registration of K = {k}'s B rows: one-stage's {{value, root, "
            f"leaves, words, domain}}, its root a SHA-512 digest, its {n} rows (K's linears' n in turn) of {x} leaves "
            f"each (weights.chunks), {x * n} leaves of {leaf_words(k)} 16-bit words (a chunk's K/{x} codes then the "
            f"row's s_w, h1zy.ROW) under the registered domain of {v} and the window's circuit (the SHA-512 of its "
            "descriptor), binding no window (registered.domain)")


def _weights(window: Any, layout: Any, entries: Any, requests: Any, session: Any, program: Any, prefix: Any
             ) -> tuple[bytes | None, tuple[str, str] | None]:
    """The weights' root the window registered and None; or None and the refusal (``weights`` or ``layout``, and why).
    `entries` (the window's registered entries, ``{value, root, …}``) must register, for each K of the window's
    `layout` (`strata`), exactly ``<prefix>b/<K>`` (`prefix` its descriptor's ``PREFIX``) once, and no other ``…/b/<K>``;
    each such entry must be the registration of all of K's B rows (K's linears' n in turn, each x = `weights.chunks` (K)
    leaves of `weights.leaf_words` (K) words, x·Σn leaves),
    under `registered.domain` of its name and the circuit whose descriptor's SHA-512 is `program`, binding no window
    (`other_weights_entry`); and `weights_root` of their roots, the window's round-0 request table's root `requests`
    (hex) and `session` (the id of the session the auditor's ledger opened for `window`, as `core_check.salt` reads it)
    must be the window's open root (``window.root``, `salt.window_open_root`). The salt coin opens only after that root,
    so every code and s_w of a B row a drawn unit opens at its ``…/b/<K>`` root was fixed before the window's salt, and
    a window whose B rows differ from the ones it opened on is refused. A step of `window` (`_period_weights`), which
    also holds the root to the one the window's period signed."""
    def no(why: str) -> tuple[None, tuple[str, str]]:
        return None, ("weights", why)

    r = _digest(window.get("root")) if isinstance(window, dict) else None
    if r is None:
        return no(WINDOW_ROOT_FORM)
    t = _digest(requests)
    if t is None:
        return no(REQUESTS_FORM)
    if not (type(session) is str and PS.SESSION.fullmatch(session)):
        return no(SESSION_FORM)
    if _digest(program) is None or not _printable(prefix):
        return no(CIRCUIT_FORM)
    _ss, why = strata(layout)
    if why is not None:
        return None, why
    rows: dict[int, int] = {}
    for _name, k, n in layout["linears"]:
        rows[k] = rows.get(k, 0) + n
    if bad := sorted(k for k in rows if _chunks(k) is None):
        return None, ("layout", f"the window's K = {bad} have no chunks (weights.chunks: K <= {CHUNK_MAX}, or K/n a "
                                f"multiple of {CHUNK_ALIGN} codes for some n >= K/{CHUNK_MAX}), so no row of them is leaves")
    names = {f"{prefix}b/{k}": k for k in rows}
    if not (isinstance(entries, list) and all(isinstance(e, dict) for e in entries)):
        return no(ENTRIES_FORM)
    roots: dict[int, bytes] = {}
    for e in entries:
        v = e.get("value")
        if not is_held(v):
            continue
        if v not in names:
            return no(f"the window registers weights {v}, which is not {prefix}b/<K> of a K of its circuit: it "
                      f"registers each of its K's B rows as {prefix}b/<K> ({prefix} its descriptor's PREFIX), and no "
                      "other …/b/<K>")
        k, n = names[v], rows[names[v]]
        if k in roots:
            return no(f"the window registers K = {k}'s weights twice ({v}): each K's B rows are one registered value "
                      f"{prefix}b/<K>")
        h, x, w = _digest(e.get("root")), chunks(k), leaf_words(k)
        if not (h is not None and sorted(e) == sorted(G.ENTRY_KEYS) and e["leaves"] == x * n and type(e["leaves"]) is int
                and e["words"] == w and type(e["words"]) is int
                and e["domain"] == G.domain(program, v, w, x * n).domain_id.hex()):
            return no(other_weights_entry(v, k, n))
        roots[k] = h
    for k in sorted(rows):
        if k not in roots:
            return no(f"the window registers no weights {prefix}b/{k} of K = {k}: each K's B rows, their codes and s_w, "
                      "are registered under the weights' root its open root binds, before its salt")
    w = weights_root(roots)
    if PS.window_open_root(w, t, session) != r:
        return no(OTHER_WEIGHTS)
    return w, None


__all__ = ["CALL_RANGE_CODE", "CALLS_TAG", "CIRCUIT_FILES", "COMMIT_FORMAT", "COMMIT_TAG", "CREDIT", "DELTA", "DEPTH", "DRAW_TAG", "EARLIER_FORM", "INDEX", "K_FIELD", "LAYOUT_KEYS", "LENGTH_FIELD",
           "LENGTH_KEYS", "ROWS_FIELD", "LEVELS",
           "LIVE", "LIVE_FN", "LIVE_KEYS", "LIVE_RECORD", "LIVE_STATEMENT_KEYS", "LIVE_STATEMENTS", "LIVE_UNIT", "MAX_K", "MODEL_FIELD", "MODEL_KEYS", "MODELS", "PARTITION_FORMAT", "PASS_REFUSED", "PEARL_C_FORM", "PEARL_C_KEYS", "PERIOD_KEYS", "PINNED_MODELS", "PROOF_KEYS", "PROVED_KEYS", "QUERY", "RECORD", "REPEATED_CALL", "REQUESTS_IDS_TAG", "REQUESTS_OUTSIDE", "REQUESTS_TABLE", "ROUND_TAG",
           "SALT_ONLY", "SCHEMA", "SERVED_ROWS", "STATEMENT_KEYS", "TILE_FN", "VERDICTS", "WEIGHTS_FIELD", "WINDOW_FN", "WINDOW_KEYS",
           "call_work", "calls_root", "commit_ids", "commit_of", "commit_rows", "committed_files", "committed_k", "committed_length", "committed_rows", "committed_weights", "cover",
           "credited_work", "domain_of", "draw_root", "drawn", "established", "a_value", "ids_step", "is_a", "is_held", "is_requests", "is_tied", "law", "layout_of", "live_of",
           "merkle_depth", "partition_of", "passes_distinct", "pearl_c_period_credit", "period", "period_credit", "registered_file", "requests_root", "round_root", "shape_of",
           "served_rows", "served_work", "slots_step", "step_row", "step_tables", "strata", "table_rows", "tree_root",
           "window"]
