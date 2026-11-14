"""scan-zk: the network certifier's step-indexed scan composed in zero knowledge, in the generator form, one
link-window a registration. A link-window's start unit (`ingress_scan.IngressScanStart`) copies the state its scan
starts from, the one the record before it committed (the run's first: the empty state); ingress bucket t
(`ingress_scan.IngressScanBucket`) reads the warden's ingress row t, the scan's state S_{t−1} and its boundary's slot
entries, and computes S_t and its egress view; egress bucket t (`egress.EgressBucket`) reads that view and computes
the warden's egress row t. Every row of every unit, input or output, is a registered read, proved by Flock
`prove --zk` behind the Lean ZK server and held by Lean `verify --zk --registered`. Standalone: in no Spec, except
the honest round on the challenger's coins (``--challenger``), a `service.Stream`'s.

    python benchmarks/network_traces/scan_zk.py prove --bin flock-circuit --lean flock-verify --out DIR [--dry]
        [--class w63|w63k4|test|test2] [--group G] [--warden-run RUN] [--v3c WINDOW] [--draw K [--challenger
        local|gate:USER@HOST --main FILE [--peer NAME --challenger-key RECORD] [--gate-key KEY]]]
    python benchmarks/network_traces/scan_zk.py schedule [--class w63|w63k4|test|test2]    # as warden_serve.py's flags

* **The setups** (`SETUPS`, ``--class``): ``w63`` (`W63`, whose `CLASS`, `SHAPE` and `PARAMS` these are), w63's class
  (B 512, r 4, 4 lanes, C 47, IDB 17, EVERY 16, WITHIN 12: 4 Program slots a slot stride, 8 an execution phase) with the
  network certifier's delay D 27 (WITHIN + EVERY − 1), KO 64 outputs and KI 1,024 prompt ids a request,
  egress at `CLASS.egress` (16 cells in lanes of 4, 4 request slots of 64 17-bit ids), the warden at `PARAMS`, 32-bucket
  link-windows; ``w63k4`` (`W63K4`), w63's class at KO 4 and KI 32, a window circuit's of 8 slots, 32 prompt ids and
  4 outputs a request (egress 4 request slots of 4 ids), its certifier w63's at egress_cells 5 (KO + 1); ``test``
  (`TEST`), the certifier tests' class (B 512, r 2, 2 lanes, C 4, EVERY 2, WITHIN 3, D 5, KO 8, KI 100) and 16-bucket
  link-windows, whose units a CPU proves in minutes; and ``test2`` (`TEST2`), test's over two link-windows, with a
  fourth request admitted in the first and anchored at the second's first boundary.
* **The run** (`scan_traffic.serve`: the warden's own `_ingress` and `_egress`, ticked as `Warden._tick` ticks them)
  carries the setup's warden-form id requests and their responses: w63's five (3, 40, 64, 8 and 5 ids; the fifth waits
  in the queue until request 1's egress ends at bucket 27 and takes its slot at 28), w63k4's five at w63's arrivals
  (prompts of at most 32 ids, responses of 3, 4, 1, 4 and 2, the fifth in request 1's slot at 28 as w63's), test's
  three (the third in the first's slot past its egress). Each link-window of each link is committed by the warden and
  its head signed by a test-size LMS key (`warden_rows.attest`); each registration is read back by the verifier
  (`warden_rows.check_registration`).
* **A network certifier's run** (``--warden-run RUN``, `warden_run`) in its place: a warden_serve run's directory
  (``serve``'s, or the gate's ``warden-rehearse --scan``'s with its witness under ``witness/``) at the setup's schedule
  (w63's: ``scan_zk.py schedule``) carrying the scan's traffic (``rehearse --ids``). The verifier reads
  ``deployment.json`` (the network certifier's public key, the session, the schedule and its digest) and each link's
  registration of each execution phase of the group in ``cells.jsonl``, its head's signature verified under that key;
  the prover's rows are the records under ``cells.salts``, held to that head. Each is refused by name where it doesn't
  hold, as is a schedule the class doesn't serve and a response that isn't id-form.
* **The group** (``--group G``, default 1; `link`'s ``start``, its first link-window, 0): G link-windows of each
  link, each a registration of its own (`window_of`) and proved in rounds of its own, ``<kind>-<its number>``
  (`round_name`), in link-window order on one verifier's log (`prove`). The scan's chain runs over the group's
  buckets from the empty state, T a link-window, and link-window w's rounds read its buckets w·T … (w + 1)·T − 1 as
  their buckets 0 … T − 1.
* **A link-window's rows**, committed by the prover before the sample as hiding rows (`verity.commitments.advice`,
  each tree's root in the registration, bound to the session and the link-window's ingress head): ``state``, T
  leaves, leaf t the state bucket t reads, S_{t−1}, and leaf 0 the state the link-window starts from, the rows a
  window circuit of the link-window reads as its `SCAN_STATE` (`ingress_scan.window_state`), so the state bucket
  T − 1 computes, which no bucket of the link-window reads, has no leaf; ``prog<j>`` for each lane j < NR, lane j's
  slot entries (`slot_entries`), nb leaves for the link-window's nb = T / EVERY boundaries, leaf b slot j's entry at
  boundary b (the outputs of the Program anchored there, read and its response ids, as `SlotEntry_v1` writes them),
  the o0 rows of the block round's lane-j member; and ``view``, T leaves, leaf t S_t's egress view.
* **The block round** (`BLOCK`): the slot entries' stand-in (`SlotEntry{KO, IDB}`, whose Definition id,
  ``SlotEntry_v1``, is the one `NetworkCheck.Period`'s `wired` reads: o0 its `entry_bits` input ``x`` through two
  NOTs, a committed row the verifier never opens; ``x`` a row-seg/v1 row, `ROW_SCHEMAS`), one unit a slot entry of the
  link-window, cut as one-stage cuts a template into NR members of nb units, a lane each (``block-c<j>``,
  `members_of`): unit j·nb + b's ``x`` slot entry b of lane j, its free input (the member's own rows, `Frame`), and
  its o0 a registered read of ``prog<j>`` leaf b, standing in for the anchored Programs' `SlotEntry_v1` units, whose
  outputs the slot entries are. Its entries, read bit included, are the prover's: the stand-in copies its input.
  Lane j's ``prog<j>`` tree is committed under the binding of its block member's o0 frame domain (`block_binding`,
  over that member's statement's input set `block_set`, bound to the lane's value, the session and the link-window's
  ingress head), so its root is the o0 frame root Flock commits under the same salts, and `stage` refuses a block
  member whose o0 frame root and domain aren't its lane's tree's: the registration's ``SlotEntry_v1{…}/c<j>/o0`` (its
  ports named as one-stage names a template's chunk j's, `entries_port`) and the honest round's ``scan/prog<j>`` are
  one root and domain, the tie `NetworkCheck.Period`'s rows form checks for each lane (`progTied`, `wired`). Its
  registration also names the scan's ``state`` tree as ``scan/state``, as an anchored Program's names the
  `SCAN_STATE` it reads, the tie the rows form checks next (`stateTied`); the stand-in reads no state, so here that
  tie is its registration's alone. A link-window's first round; accepted, with no output opened.
* **V3c's trees** (``--v3c WINDOW``, `v3c_trees`; a group of 1): in place of the scan's own, the trees V3c's window
  (`prove_path.py window-commit`'s directory, at w63k4's class) committed before its coins: ``state`` its
  `SCAN_STATE` (its registration's window activations entry, the rows and salts its prover's `SCAN_STATE_FILE`) and lane
  j's ``prog<j>`` chunk j's o0 of its `SELECT_ENTRY` template (``<descriptor id>/c<j>/o0`` in its ``roots``, the rows
  and salts its member's prover body), each a `Held` tree at that registration's root and domain, refused by name where
  its rows aren't the scan's own. The honest round's ``scan/state`` and ``scan/prog<j>`` are then V3c's roots and
  domains, which the consistency check (`NetworkCheck.Period`'s rows form: `stateTied`, `wired`) holds to V3c's
  audit at any sample of it, and there is no block round: V3c's audit proves the anchored Programs' slot entries, and
  ``answers.json`` carries no ``audits`` (the period reads V3c's ``period.json``, `scan_period.rows_case`). `stateTied`
  reads an audit's state entry among its ``roots`` and its ``registered`` alone, so the check refuses V3c's audit,
  whose `SCAN_STATE` is in its window's activations alone, at the state tie.
* **The honest round**: one Program of three templates (`TEMPLATES`) in the generator form (`scan_root`, the form
  `NetworkCheck.Period`'s `scanOf` reads), 2T + 1 units: the start unit (`IngressScanStart`), T ingress units
  (`IngressScanBucket`, buckets 0 … T − 1) and T egress units, each template one batch with its row, state and egress
  parameters inputs of its own and every ingress unit's ``prog<j>`` a row of lane j's input of its slot entries, its
  members ``start``, ``ingress-0``, ``last`` (the link-window's last ingress unit, whose o1 has no leaf to open) and
  ``egress-0`` (`members_of`), every unit sampled by the verifier's Lean. `program` refuses a Program whose one-stage
  partition or reads (`registered.reads`) aren't the members'. The start unit reads ``sin``, the state the
  link-window starts from (the start, below), and outputs ``o0``, that row through two NOTs, committed as state leaf
  0, a committed row the verifier never opens: the row bucket 0's ``sin`` and the first boundary's Programs read is
  the start by gates. Ingress unit t reads ``row`` (the warden's ingress row t), ``sin`` (state leaf t, S_{t−1}) and
  each slot j's ``prog<j>`` (``prog<j>`` leaf ⌊t / EVERY⌋, its boundary's entry of the slot, one table row a boundary,
  `NetworkCheck.Period`'s `readsEntries`), and outputs ``o0``, its ok bit, ``o1``, S_t, committed as state leaf t + 1,
  and ``o2``, its view, committed as view leaf t. Egress unit t reads ``state`` (view leaf t) and outputs ``o0``, its
  render bit, and ``o1``, the row it computes, committed as the warden's egress row t. Each output row but a public
  o0 is a registered read of the row it is committed as (`Registered.checkPort` holds a hidden-output statement's
  output rows, the verifier's PROTOCOL.md §16.13), so S_t is the one leaf unit t computes and unit t + 1 reads, the
  view the one leaf ingress t computes and egress t reads, and the egress row the one the warden signed: the chains
  are wires through the registered roots, not the prover's word. `Registered.checkPort` holds every instance of a
  registered port, so the last unit's o1, the state the link-window ends in, which no state leaf holds, is a
  member's own rows (`Frame`), committed and never opened: the port the next link-window's start reads
  (`CARRY_PORT`). The registration's window states the state's reads, one run a port (``reads``, `state_reads`: the
  start unit's ``o0`` at leaf 0, the ingress units' ``sin`` from leaf 0 and their ``o1``, but the last's, from leaf
  1, as `NetworkCheck.Period`'s `stateRead` holds them). Accepted: every public o0 opens to 1. Its registration names
  each lane's ``prog<j>`` tied, which R8 lets through though `EVERY` units read each slot entry, and both verifiers'
  verdict publishes them (``rests_on_period_check``, the lanes' ``prog<j>``): the acceptance rests on a period check,
  `NetworkCheck.Period`'s `wired`, which ties each ``prog<j>``'s rows to the audit that registered them and refuses
  this run's stand-in block, no pinned `SlotEntry_v1` (`period_case.UNWIRED`). The registration's roots also name, as
  ``rows/<direction>/<w>`` (`warden_roots`), the root and domain of each warden row registration a member's warden port
  reads (ingress's ``row``, egress's ``o1``), so a period check holds the answer to them.
* **The start**: the start unit's ``sin`` is one-stage's registered value `CARRY` (the Program's parameter ``0.sin``,
  one row/v2 row), which the link-window's record, a v2 record, names with its root and domain. The run's first
  record holds `CARRY` at the root the verifier computes and pins, the empty state's row under one-stage's registered
  domain and the public salt of a value the verifier fills (`empty_start`), and its `window` opens it (``empty``,
  `NetworkCheck.Period`'s `startEmpty`: the state's zero words under that salt), public as that state is. Each later
  record's ``previous`` is the record before it and its `CARRY` entry is that record's ``last/o1`` (`CARRY_PORT`: its
  last ingress unit's o1), root and domain, which the verifier holds in the window-history form ``{"port": P}``, the
  port of its log's last record (`carry_of`); the start member's statement opens its ``sin`` row there, under the
  salt and at the frame domain the earlier statement committed it with, and the verifier's ``--registered`` port of
  it is one-stage's (`registered.reads_file`, its ``domains``). The verifier's log takes a link-window's record once
  its honest round's audit accepted it with every opened output 1 (`prove`), so no start is read from a link-window
  the verifier refused. A record's changed forms are refused by the registration check (another root, `R7-held`; no
  earlier record in the verifier's log, `R7-output`), and the start's leaf as a registered read frames it is the
  earlier statement's frame leaf byte for byte (`frame_leaf`). One-stage accepts a record that names an older
  record's port in the explicit form ``{"record": D, "port": P}`` (``explicit-older``, recorded, not refused): which
  record a start may name is for the check that reads the records after it.
* **Every port registered**: the verifier's registered reads name every port of every member's circuit, inputs and
  output rows (`Flock.Registered.checkComplete`, which `flock-verify` runs whenever ``--registered`` is given), so no
  output reads a row the statement commits afresh. A port no scan tree or network certifier row registration holds
  (each proof block's public o0, the last ingress proof block's o1, a block member's ``x``) is the member's own rows of
  it (`Frame`, value ``frame``), committed under its statement's frame binding (`frame_binding`, over the member's
  input set), so its root is the member's frame root of the port, which the round's registration names before any
  coin; `stage` refuses a member whose frame root isn't the tree's. A scan member's input set is its template's
  (`set_of`), naming no round and no link-window, so the frame domain of such a port is its template's and its unit
  count's in every round. A public o0's rows are committed under the verifier's public salts (`ok_salts`,
  `scan_ok.salt`), so the o0 frame root of a member whose every proof block is ok is a constant of its template and
  count, which a period check computes from the schedule's T and holds the registration's ``<member>/o0`` to
  (`scan_ok.constants`, the period check's `okChecks`); it is opened as before, under that salt, which the
  prover sends.
* **The answers** (``answers.json``, `answers`): for each link-window W of the group, the scan's registration of W,
  the network certifier's two row registrations of W and the scan's one-stage answer for W (`window_answer`), which
  `verity/certifier/v0`'s rows form (`NetworkCheck.Period`) checks for W with W's three records' certificates:
  accepted, one-stage's profile and bound at ``--delta`` (every unit sampled, the bound 0); refused, a reject naming
  the members Lean refused. Beside them, ``audits``: each link-window's block round's, the compute audit whose lanes'
  o0 are that link-window's scan's ``prog<j>`` (its registration and answer, its Program's descriptor, its free
  inputs NR·nb·NE, every bit of every slot entry, which the rows form's K charges to the link-window tied to it, and
  no steps).
* **A void run** (a warden run whose warden failed a connection closed, its `FAIL` in the egress row it signed, its
  reason in the ``fails`` record): the scan renders the connection's end where the warden wrote `FAIL`, and that bucket
  alone differs from the warden's row. Its FAILs are in the group's last link-window (`link` refuses a group with
  one earlier: the link-window after a refused one has no start the verifier holds). That link-window's block round
  and its honest round are proved, the honest one expected refused by Lean's registered reads ("does not open") at
  its egress member (`rounds_of`, `targets`), every other member held, and its answer is that refusal, so the period
  check rejects the link-window by name. No negative is placed in a void run.
* **Negatives**, each a registration of one link-window of the group in its honest round's place (`link` places
  them), chain's, tampered's and stale's refused by Lean's registered reads ("does not open"), unread's by one unit's
  ok bit. ``chain``, in bucket ``A``'s link-window: the same Program on a state tree with S_a and S_{a+1} swapped
  (leaves a + 1 and a + 2, a that bucket in its link-window): ingress unit a's o1, the S_a it computes, doesn't open
  leaf a + 1, nor units a + 1's and a + 2's theirs. ``tampered``, in the first link-window with a `DATA` cell: an
  egress-only Program of T units, one egress row's `DATA` id changed and the link-window re-signed (a warden that
  saw a byte the scan doesn't render; a warden run's re-signed by the driver's own key, under which the verifier
  reads it): egress unit t's o1, the row its view renders, doesn't open the signed row t. ``unread`` (`_unread`), in
  the link-window of the group's first bucket U ≥ 1 where a request's Program starts (its slot's `occ` 1): a Program
  of the T ingress units alone, its slot j's ``prog<j>`` tree the honest one but for its entry at U's boundary, its
  read bit cleared (no Program reads the request): every registered read holds, since the read bit enters the unit's
  ok bit alone and its S_U and view are the honest ones, and Lean holds its proofs, but bucket U's o0 opens to 0, so
  the link-window is refused. Its units are every bucket's, not U's alone, so that unit m of the ingress template
  is bucket m, which reads the warden's row m, as in every Program that batches the template; every one is sampled,
  whatever ``--draw``. A prover that sets the bit with n 0 and no outputs passes here, since the stand-in's entries
  are the prover's; an anchored Program's entry is its outputs on the `SCAN_STATE` it reads, which the period check
  holds to the scan's state tree (`stateTied`). ``stale`` (`STALE`, ``stale-<the last link-window's number + 1>``):
  a start unit alone, registered after the group's last record as the link-window after it would be, given the
  state the last link-window started from in place of the one it ended in, which opens no leaf of the last record's
  ``last/o1``.
* **A sampled sample** (``--draw K``): Lean ``draw --stratified K`` on a development challenger's coins opened after
  the round's registration (`a0.challenger_draw`: the verifier has none of its own), each template's stratum its share
  and at least one unit; only the sampled proof blocks are proved and opened, and an accepted audit carries ε at δ
  (``wrong_units``: at most ``bound`` of ``of`` units wrong, except with probability δ), its charge terms
  (`charge_terms`) and each member's cost per sampled proof block (`per_unit`). The chain and tampered rounds sample
  too and are refused all the same: Lean holds every table row of a registered port, sampled or not, an output port's
  table its instances' rows.
* **The challenger's coins** (``--challenger local|gate:USER@HOST --main FILE``, with ``--draw K``): the honest round
  sampled at the stratified rate of K in its N (`rule_of`: a stream's rule is a rate) as prove_path's records audit
  samples (`a5_coins`): for each link-window, one session of the auditor's challenger on the link-window's root
  (`window_root`: the warden's session and the link-window's signed registrations), and the round's `service.Stream`
  (`spec_of`, `stream_of`: its own copy of the round's Program the generator form the round registers) on its coins
  (`a5_coins.RankCoins`): the stream registers the round, the coins commit to the registration's digest, and its
  select samples once (Lean's `Select.select` on the bytes read). Once the session has ended, the outcome of record is
  the stream's with the coins' evidence, the Lean's whole run (`Run.run`, coin source the challenger) held to the
  Python's, and each altered evidence (`a5_coins.negatives`, the evidence for another window) refused by
  `coins.coins_reason` and the Lean's run (`coins_outcome`). ``local`` is a development session on this machine (its
  own key; it shows the path, not an auditor's coins); ``gate:USER@HOST`` is the auditor's challenger's through its
  gate, its challenger on ``--peer``'s 127.0.0.1, under the deployment's key record ``--challenger-key``. The gate
  opens a root once, ever: a run's link-window is sampled once (the synthetic run's registrations carry fresh salts,
  so each synthetic run has a root of its own). The other rounds keep a development challenger's coins: one session
  serves one sample, and their refusal is Lean's registered reads, whatever the coins.
* **What the verifier receives** is scanned for every piece of every unit's rows, inputs and outputs, and every salt
  but a public ``o0``'s, which the prover sends so the verifier opens each unit's output (`a8.received`; the slot
  entries' o0 is a committed row, `ROWS_O0`, whose salts it never sends), less the pieces public
  anyway (the circuits, zeros, the window's pads and `END` cells) and the sparse pieces (at most one nonzero byte); the
  prover's instance file, which holds them all, is the control. The instance holds each row's u16 words followed by
  the next row's, so a row's zero-padded last chunk isn't one the control holds; its last 16 bytes, which cover the
  same tail, are (`leak_scan.public_pieces`' ``fields``). A row shorter than a piece (w63k4's slot entries, 14 bytes)
  is found whole, raw or as hex (`received`, `leak_scan.find_whole`), and held to the same: none found, the control
  holding each.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import socket
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, NamedTuple, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import leak_scan as LS  # noqa: E402
import warden_zk as WZ  # noqa: E402
from verity.circuits import boolean as BL  # noqa: E402
from verity.circuits.boolean import trace as TR  # noqa: E402
from verity.circuits.codec import canonical_json  # noqa: E402
from verity.circuits.defs import CompositeDefinition, bind  # noqa: E402
from verity.circuits.program import Program  # noqa: E402
from verity.circuits.refs import Coll, array_of, coll_concat  # noqa: E402
from verity.circuits.types import Array, Tuple  # noqa: E402
from verity.commitments import advice as ADV, domains  # noqa: E402
from verity.commitments.frame_v3 import BitRowLeaf, FrameV3, Port  # noqa: E402
from verity.commitments.identity import identity_digest_sha512  # noqa: E402
from verity.commitments.indexed import RangeIndexedDomain  # noqa: E402
from verity.commitments.merkle import CommitmentDomain  # noqa: E402
from verity.commitments.rowleaf import (SCHEMA_HM96_SHA512_ROW_SEG, bit_row, row_leaf,  # noqa: E402
                                           sha512_row_seg_digest)
from verity.commitments.crypto import lms  # noqa: E402
from verity.protocols.accounting.certifier.prover import egress as EG, ingress_scan as IS  # noqa: E402
from verity.protocols.accounting.certifier.verifier import io_tie as IT, scan_ok, warden as W  # noqa: E402
from verity.protocols.accounting.certifier.verifier import warden_rows as WR  # noqa: E402
from verity.protocols.accounting.certifier.verifier.active import DATA, id_request  # noqa: E402
from verity.protocols.accounting.certifier.verifier.blocks import SCAN_STATE  # noqa: E402
from verity.protocols.accounting.certifier.verifier.certificate import Direction, Issuer, LmsSigner  # noqa: E402
from verity.protocols.proofs.spot.device import coins as CC
from verity.protocols.proofs.spot.verifier import service as S  # noqa: E402
from verity.protocols.proofs.spot.verifier.consumers import exfiltration_bound, location_bits  # noqa: E402
from verity.protocols.proofs.spot.verifier.one_stage import audit as AU, draw as DR, partition as PA, produced  # noqa: E402
from verity.protocols.proofs.spot.verifier.one_stage import registered as REG, registration as RG  # noqa: E402
from verity.protocols.proofs.spot.verifier.one_stage.draw import stratum_k  # noqa: E402

_TRAFFIC = HERE.parents[1] / "verity/protocols/accounting/certifier/verifier/tests/scan_traffic.py"
_spec = importlib.util.spec_from_file_location("scan_traffic", _TRAFFIC)
traffic = sys.modules["scan_traffic"] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(traffic)


class Setup(NamedTuple):
    """A class the scan is proved at, the network certifier's schedule it serves and the synthetic run's traffic:
    request i's prompt ``prompts[i]`` (max_tokens ``max_tokens`` + i, seed ``seed``·i) arrives at bucket ``arrivals[i]``
    and is answered ``responses[i]``, and the network certifier serves ``windows`` execution phases, every connection
    ended by the last's end; the chain round swaps S_A and S_{A+1}."""

    name: str
    cls: IS.Class
    params: W.WardenParams
    prompts: tuple[tuple[int, ...], ...]
    arrivals: tuple[int, ...]
    responses: tuple[tuple[int, ...], ...]
    windows: int
    A: int
    max_tokens: int
    seed: int


W63 = Setup(
    "w63", IS.Class(B=512, r=4, lanes=4, C=47, IDB=17, EVERY=16, WITHIN=12, D=27, KO=64, KI=1024),
    W.WardenParams(cell_bytes=512, slots=4, bucket_ms=100, delay=27, lanes=4, queue=4, window=32, ingress_cells=47,
                   ingress_buckets=12, egress_cells=65, every=16),
    (tuple((7 * k + 11) % 128_256 for k in range(1024)), tuple(100_000 + k for k in range(600)),
     tuple(131_071 - 3 * k for k in range(40)), tuple((k * k) % 128_256 for k in range(300)),
     tuple(5_000 + k for k in range(100))),
    (0, 2, 2, 12, 15),
    tuple(tuple((97 * k + 13 * q) % (1 << 17) for k in range(n)) for q, n in enumerate((3, 40, 64, 8, 5), 1)),
    windows=2, A=9, max_tokens=16, seed=7)
W63K4 = Setup(
    "w63k4", replace(W63.cls, KO=4, KI=32), replace(W63.params, egress_cells=5),
    (tuple((7 * k + 11) % 128_256 for k in range(32)), tuple(100_000 + k for k in range(20)),
     tuple(131_071 - 3 * k for k in range(5)), tuple((k * k) % 128_256 for k in range(32)),
     tuple(5_000 + k for k in range(9))),
    W63.arrivals,
    tuple(tuple((97 * k + 13 * q) % (1 << 17) for k in range(n)) for q, n in enumerate((3, 4, 1, 4, 2), 1)),
    windows=2, A=9, max_tokens=4, seed=7)
TEST = Setup(
    "test", IS.Class(B=512, r=2, lanes=2, C=4, IDB=17, EVERY=2, WITHIN=3, D=5, KO=8, KI=100),
    W.WardenParams(cell_bytes=512, slots=2, bucket_ms=100, delay=5, lanes=2, queue=2, window=16, ingress_cells=4,
                   ingress_buckets=3, egress_cells=9, every=2),
    ((1000, 1007, 1014), tuple(2000 + 3 * k for k in range(60)), tuple(131071 - k for k in range(100))),
    (0, 1, 1), ((11, 22, 33), (131071, 0, 5, 6, 7, 8, 9, 10), (40, 41, 42, 43)), windows=1, A=6, max_tokens=5, seed=11)
#: test's class and its three requests, and a fourth admitted at bucket 12 of link-window 0 and anchored at bucket 16,
#: link-window 1's first boundary: the state link-window 1 starts from holds it live, its Program due at that boundary
TEST2 = TEST._replace(name="test2", prompts=(*TEST.prompts, (7, 8, 9)), arrivals=(*TEST.arrivals, 12),
                      responses=(*TEST.responses, (51, 52, 53)), windows=2)
SETUPS = {s.name: s for s in (W63, W63K4, TEST, TEST2)}
#: w63's: the schedule ``scan_zk.py schedule`` prints and the gate's ``warden-rehearse --scan`` serves
CLASS, PARAMS = W63.cls, W63.params
SHAPE = CLASS.egress
SESSION = bytes(range(32))
#: a link-window's scan's templates in its Program's order: the start unit (`IngressScanStart`), then every bucket's
#: ingress unit (`IngressScanBucket`, bucket 0's too) and its egress unit, 2T + 1 units, one link-window a registration
TEMPLATES = ("start", "ingress", "egress")
#: the unread round's: the ingress units alone, every bucket's
UNREAD = ("ingress",)
#: the block round's: the slot entries' stand-in (`SlotEntry`), one unit a slot entry, whose o0 rows are the scan's
#: slot entries
BLOCK = ("block",)
#: the stale round's: a start unit alone
STALE = ("start",)
#: the start: the start unit's ``sin`` as the Program names it (`scan_root`'s ``<i>.<name>``; a population's too), the
#: one-stage registered value a link-window's record names the state its scan starts from by
CARRY = "0.sin"
#: the port of a link-window's record the next one's start reads: its last ingress unit's o1, the last state
CARRY_PORT = "last/o1"
#: the templates whose o0 is a committed row the verifier never opens (the slot entries', the start's copy), every
#: other's its public word
ROWS_O0 = frozenset({*BLOCK, "start"})
#: an input row's schema where it isn't hm96-sha512/row/v2: the block's ``x`` is a row-seg/v1 row, which takes a
#: compression before its last block at any length, as Flock's statement needs of one row (`verity_flock.circuit.
#: compose`); at the test class every block row is shorter than a block (NE = 176 bits)
ROW_SCHEMAS = {("block", "x"): SCHEMA_HM96_SHA512_ROW_SEG}
#: each scan template's parameters before its entry ports (`scan_root`'s inputs of its own): row (and sin), or state
FIXED = {"start": 1, "ingress": 2, "egress": 1}
#: each kind of round's templates, and its outcome: accepted, refused by Lean's registered reads, or proved with every
#: read held and one unit's ok bit 0; the block first, since the scan reads its outputs. A round is one link-window's,
#: ``<kind>-<w>`` (`round_name`)
ROUNDS = {"block": (BLOCK, "accepted"), "honest": (TEMPLATES, "accepted"), "chain": (TEMPLATES, "does not open"),
          "tampered": (("egress",), "does not open"), "unread": (UNREAD, "ok 0"), "stale": (STALE, "does not open")}
#: the kinds of round with a start unit, whose record names the state its scan starts from (`carry_of`)
STARTED = frozenset(k for k, (templates, _o) in ROUNDS.items() if "start" in templates)
#: each round Program's directory, by its templates
PROGRAMS = {TEMPLATES: "joint", ("egress",): "solo", UNREAD: "unread", BLOCK: "block", STALE: "stale"}
#: Flock's tag of a frame domain's binding (`verity_flock.circuit.BINDING_TAG`), under which a block member's o0 rows
#: are committed and so its lane's ``prog<j>`` tree (`block_binding`), and a member's own rows of a port (`Frame`;
#: `stage` refuses a member whose frame root of such a port isn't its tree's)
FRAME_BINDING = domains.FLOCK_CIRCUIT_BINDING


def value(kind: str) -> str:
    """A scan registration's value: a tree's, ``state``'s, ``view``'s or lane j's slot entries' (``prog<j>``), or
    ``frame``, a member's own rows of a port no tree or network certifier row registration holds (`Frame`), registered
    as the member's frame root of the port; ``start``'s is the state the scan starts from, the record's own
    registered value (`CARRY`)."""
    return CARRY if kind == "start" else f"verity/certifier/scan-{kind}/v1"


def round_name(kind: str, w: int) -> str:
    """Link-window ``w``'s round of ``kind`` (`ROUNDS`), as the summary and the rounds' directories name it."""
    return f"{kind}-{w}"


def kind_of(name: str) -> str:
    """A round's kind (`round_name`'s)."""
    return name.rsplit("-", 1)[0]


def lane_kinds(cls: IS.Class) -> tuple[str, ...]:
    """The slot entries' trees, one a lane: ``prog<j>``, lane j's."""
    return tuple(f"prog{j}" for j in range(cls.NR))
TIMING = ("e2e_s", "wait_s", "witness_s", "prove_total_s", "proof_bytes", "wire_sent", "wire_recv", "k_log", "nbl",
          "dense_m", "units", "ands_per_vu", "prover", "gpu")
#: an honest round's stream on the auditor's challenger's coins (``--challenger``): its challenger, the format of the
#: link-window's root its session opens on (`window_root`), and each member's share of the proof deadline
CHALLENGER = "scan_zk.py on the auditor's challenger's coins"
WINDOW_ROOT = "verity/certifier/scan-window-root/v3"
MEMBER_DEADLINE_S = 5 * 3600
#: the format of ``answers.json`` (`answers`): each link-window's registration, answer and rows, and the blocks'
#: audits
ANSWERS = "verity/certifier/scan-answers/v3"
#: in V3c's window directory (``--v3c``), its `SCAN_STATE`'s rows (u16 little-endian, row-major) and then their salts,
#: prover only (`prove_path.py window-commit` writes it): its ``v<j>-private.bin`` holds only the rows its members read
SCAN_STATE_FILE = "scan-state-private.bin"
#: the Definition id of V3c's slot entries, whose chunk j's o0 is lane j's ``prog<j>`` (`NetworkCheck.Period`'s `wired`)
SELECT_ENTRY = "SelectEntry_v1"


def log(msg: str) -> None:
    print(f"SCZK {time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def _request(setup: Setup, i: int, prompt) -> bytes:
    return id_request(model="unsloth/Llama-3.2-1B", user=f"user-{i}", prompt=list(prompt),
                      max_tokens=setup.max_tokens + i, seed=setup.seed * i, temperature=0.7, top_p=0.95,
                      frame_bytes=setup.cls.B, offset=0)


def _words(bits) -> list[int]:
    return [sum(int(b) << i for i, b in enumerate(bits[w:w + 16])) for w in range(0, len(bits), 16)]


def _bytes(x) -> bytes:
    """A row as its port's bytes: a byte row itself, a bit row its u16 words little-endian."""
    return x if isinstance(x, bytes) else b"".join(w.to_bytes(2, "little") for w in _words(x))


def ports_of(cls: IS.Class) -> dict[str, tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]]:
    """Each template's input rows and its output rows past o0 (the slot entries', `ROWS_O0`, from o0), (port, bits) in
    its signature's order."""
    NE = IS.entry_bits(cls)
    row, progs = ("row", 8 * cls.S * cls.B), tuple((f"prog{j}", NE) for j in range(cls.NR))
    outs = (("o1", IS.state_bits(cls)), ("o2", IS.view_bits(cls)))
    return {"ingress": ((row, ("sin", IS.state_bits(cls)), *progs), outs),
            "egress": ((("state", IS.view_bits(cls)),), (("o1", 8 * cls.egress.R * cls.B),)),
            "block": ((("x", NE),), (("o0", NE),)),
            "start": ((("sin", IS.state_bits(cls)),), (("o0", IS.state_bits(cls)),))}


def entries_of(cls: IS.Class, T: int) -> int:
    """The slot entries of a link-window of T buckets: NR a boundary, T / EVERY boundaries."""
    return cls.NR * (T // cls.EVERY)


def entry_at(cls: IS.Class, t: int) -> int:
    """The row of lane j's slot entries bucket t's ``prog<j>`` reads (`NetworkCheck.Period`'s `readsEntries`): slot
    j's at t's boundary, row ⌊t / EVERY⌋ of the lane's tree."""
    return t // cls.EVERY


def slot_entries(cls: IS.Class, progs: list[list[int]]) -> list[list[list[int]]]:
    """The slot entries of the scan's Programs' rows `progs` (one a bucket, `scan_traffic.chain`'s), each lane's
    ``prog<j>`` tree's leaves: lane j's leaf b slot j's entry of the Programs' row at boundary b
    (`ingress_scan.prog_rows`), the Programs anchored there, slot j · nb + b of the window's. A bucket between
    boundaries has no due slot, so its row reads none of its entries."""
    nb = len(progs) // cls.EVERY
    return [[IS.prog_rows(cls, progs[b * cls.EVERY])[j] for b in range(nb)] for j in range(cls.NR)]


def members_of(templates: tuple[str, ...], T: int, entries: int = 0, lanes: int = 1) -> list[SimpleNamespace]:
    """A link-window's round's members in its Program's order: each one's name, its first unit (``base``) and the
    link-window's buckets its units compute, unit ``base + k`` bucket ``buckets[k]``. The scan's (`TEMPLATES`):
    ``start``, the start unit, which computes no bucket (its one unit is listed as bucket 0, the bucket whose ``sin``
    its o0 is); ``ingress-0``, buckets 0 … T − 2, and ``last``, bucket T − 1, a member of its own (its S_t has no
    state leaf, and a registered port holds every instance's row); ``egress-0``, buckets 0 … T − 1: the members
    `NetworkCheck.ScanOk` names. A round of some of those templates has their members: the stale round's (`STALE`)
    the start unit alone, the tampered round's the egress units, the unread round's (`UNREAD`) the ingress units, every
    bucket's, so that unit m of a template is bucket m in every Program that batches it. The block round's (`BLOCK`)
    the link-window's ``entries`` slot entries (`entries_of`) cut into ``lanes`` members of consecutive units, as
    one-stage cuts a template (``block-c<j>``, lane j's, its ``lane``), unit j·nb + b slot entry b of lane j, since
    its o0 rows are the lane's ``prog<j>`` tree."""
    if templates == BLOCK:
        if entries < 1 or lanes < 1 or entries % lanes:
            raise ValueError("the block round's members are the link-window's slot entries, a lane each: give "
                             "entries (entries_of) and lanes, the lanes dividing the entries")
        nb = entries // lanes
        return [SimpleNamespace(template="block", name=f"block-c{j}", base=j * nb,
                                buckets=range(j * nb, (j + 1) * nb), lane=j) for j in range(lanes)]
    cut = {"start": ((range(1), "start"),), "ingress": ((range(T - 1), "ingress-0"), (range(T - 1, T), "last")),
           "egress": ((range(T), "egress-0"),)}
    out, at = [], 0
    for tp in templates:
        for part, name in cut[tp]:
            if part:
                out.append(SimpleNamespace(template=tp, name=name, base=at, buckets=part))
                at += len(part)
    return out


def counts_of(members: list[SimpleNamespace]) -> dict[str, int]:
    """Each template's proof blocks over its members."""
    out: dict[str, int] = {}
    for m in members:
        out[m.template] = out.get(m.template, 0) + len(m.buckets)
    return out


# --- the scan's rows: hiding rows the prover registers -----------------------------------------------------------------

def _bound(kind: str, heads: list, session: bytes) -> bytes:
    return hashlib.sha512(b"\0".join([value(kind).encode(), session,
                                      *(x for h in heads for x in (h.value.encode(), h.root))])).digest()


def block_set(heads: list, session: bytes, j: int) -> str:
    """Lane j's block member's statement's input set, which its frame domains bind: bound to the lane's value, the
    session and the group's ingress link-windows' heads."""
    return f"scan-block-{_bound(f'prog{j}', heads, session).hex()}"


#: A scan template's statements' input set, which their frame domains bind (`frame_binding`): the template's alone
#: (`scan_ok.set_of`, the verifier's, which computes each member's ok bits' frame domain from it), naming no round and
#: no execution phase, so a member's frame domain of a port is its template's, its unit count's and the port's, the
#: same in every round and in every execution phase of a group. A block member's is its run's (`block_set`).
set_of = scan_ok.set_of


def frame_binding(set_name: str, n: int, port: str, schema: str = BitRowLeaf.hiding_schema) -> bytes:
    """The binding of a statement's frame domain of ``port`` over its ``n`` units (`verity_flock.circuit.statement`'s,
    at instances 0 … n of input set ``set_name``, its rows of leaf schema ``schema``): a tree under it has the leaves,
    so the root, of the port's rows committed under the same salts, since both are leaves of owner −1 over ranks
    0 … n."""
    return identity_digest_sha512(FRAME_BINDING, {"set": set_name, "lo": 0, "hi": int(n), "port": port,
                                                  "schema": schema})


def block_binding(set_name: str, n: int) -> bytes:
    """The binding of the block's o0 frame domain over its ``n`` units (`frame_binding`): an advice tree under it, of
    ``hm96-sha512/row/v2`` leaves, is the block's o0."""
    return frame_binding(set_name, n, "o0")


class Frame(ADV.Committed):
    """A member's own rows of one port, committed as Flock's statement commits the port (`verity_flock.circuit.
    statement`): under its frame binding (`frame_binding`), its leaves of the port's schema (row/v2's, or a row-seg/v1
    row's, `ROW_SCHEMAS`), each row's b‖c under the salt the stager is given (`stage`), so its root and domain are the
    member's frame port's, which the round's registration names before any coin (`audit`)."""

    def __init__(self, binding: bytes, n_bits: int, rows: tuple[bytes, ...], salts: tuple[bytes, ...],
                 schema: str = BitRowLeaf.hiding_schema) -> None:
        object.__setattr__(self, "schema", schema)
        super().__init__(binding, n_bits, rows, salts)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.schema != BitRowLeaf.hiding_schema:
            d, n = ADV.domain(self.binding, len(self.rows)), len(self.rows)
            leaves = tuple(row_leaf(d, i, ADV.INST.tree_leaf(self.key, tree_bc(self, i)), self.schema)
                           for i in range(n))
            object.__setattr__(self, "leaves", leaves)
            object.__setattr__(self, "root", ADV._scheme(self.binding, n, self.n_bits).root(ADV.PORT, leaves))


def frame(set_name: str, port: str, rows: list, n_bits: int, schema: str = BitRowLeaf.hiding_schema,
          salts: Sequence[bytes] | None = None) -> Frame:
    """A member's own rows of ``port``, each unit's (bits or bytes), as `Frame` commits them, under fresh salts, or
    under ``salts`` (a public word's public ones, `ok_salts`)."""
    return Frame(frame_binding(set_name, len(rows), port, schema), n_bits, tuple(_bytes(r) for r in rows),
                 tuple(ADV.INST.fresh_salt() for _ in rows) if salts is None else tuple(salts), schema)


def ok_salts(n: int) -> list[bytes]:
    """The salts a member's ``n`` proof blocks' public o0 rows are committed under: the verifier's public ones
    (`scan_ok.salt`), row i's by its index in the member. A proof block's public word is its ok bit, so a member
    whose every proof block is ok has the o0 frame root anyone computes from its template and count, and a period
    check holds the round's registration to it (`scan_ok.constants` at the schedule's T, `NetworkCheck.ScanOk`): a
    proof block's ok bit is then no row the prover stated."""
    return [scan_ok.salt(i) for i in range(n)]


def _schema(reg: ADV.Committed) -> str:
    """A registration's leaf schema: a member's own rows' (`Frame`) their port's, the scan's trees' row/v2."""
    return getattr(reg, "schema", BitRowLeaf.hiding_schema)


def tree(kind: str, rows: list[list[int]], heads: list, session: bytes) -> ADV.Committed:
    """`kind`'s rows (bit rows of whole u16 words) committed as hiding rows, bound to the session and the group's
    ingress link-windows' heads: lane j's ``prog<j>`` under its block member's o0 frame binding (`block_binding`), so
    the tree is that member's o0, the others' by their value (`value`)."""
    n = len(rows)
    binding = block_binding(block_set(heads, session, int(kind[4:])), n) if kind.startswith("prog") else \
        _bound(kind, heads, session)
    return ADV.commit([bit_row((_words(r), 16))[0] for r in rows], len(rows[0]), binding)


class Held:
    """A tree another registration committed, which the scan's proof blocks read in place of one of their own
    (``--v3c``, `v3c_trees`): bit rows of ``n_bits`` under ``salts`` at ``domain``, of any owner, each leaf framed as
    `advice` frames a row (`advice.leaf`), so its root is that registration's wherever its rows and salts are."""

    def __init__(self, domain: CommitmentDomain, n_bits: int, rows: Sequence[bytes], salts: Sequence[bytes],
                 key: bytes = ADV.INST.default_key) -> None:
        if len(rows) != len(salts) or len(rows) != domain.count:
            raise ValueError(f"a held tree is its domain's {domain.count} rows, a salt each")
        self.domain, self.n_bits, self.rows, self.salts, self.key = domain, n_bits, tuple(rows), tuple(salts), key
        self.leaves = tuple(ADV.leaf(domain, n_bits, i, r, s, key)
                            for i, (r, s) in enumerate(zip(self.rows, self.salts)))
        self.root = tree_scheme(self).root(ADV.PORT, self.leaves)

    def open(self, i: int) -> ADV.Opening:
        return ADV.Opening(i, self.rows[i], self.salts[i], tree_scheme(self).path(ADV.PORT, self.leaves, i))


def tree_domain(reg: ADV.Committed | Held) -> CommitmentDomain:
    """A tree's domain: a held tree's own, else its binding's over its rows (`advice.domain`)."""
    return reg.domain if isinstance(reg, Held) else ADV.domain(reg.binding, len(reg.rows))


def tree_scheme(reg: ADV.Committed | Held) -> FrameV3:
    return FrameV3({ADV.PORT: Port(tree_domain(reg), BitRowLeaf(reg.n_bits))})


def tree_bc(reg: ADV.Committed, i: int, bits: list[int] | None = None) -> bytes:
    """Row ``bits`` (leaf i's own row by default) committed under leaf i's salt, b‖c: as a statement's table row of it
    is, which opens leaf i only when the row is leaf i's."""
    row = reg.rows[i] if bits is None else bit_row((_words(bits), 16))[0]
    digest = sha512_row_seg_digest(row, reg.n_bits) if _schema(reg) == SCHEMA_HM96_SHA512_ROW_SEG else \
        BitRowLeaf(reg.n_bits).layout().digest(row)
    return ADV.INST.commit_string(reg.key, digest, reg.salts[i])


def tree_port(kind: str, reg: ADV.Committed, reads: list[int], reads_from: int) -> dict:
    """The verifier's own ``--registered`` port of `kind`'s registration: unit ``reads_from + i`` reads leaf reads[i]."""
    n = len(reg.rows)
    e = {"leaves": n, "words": reg.n_bits // 16, "word_bits": 16, "domain": tree_domain(reg).domain_id.hex()}
    port = produced.port(value(kind), e, reg.root.hex(), reads, _schema(reg))
    return port | ({REG.READS_FROM: int(reads_from)} if reads_from else {})


def tree_entry(kind: str, reg: ADV.Committed, positions) -> dict:
    return {"value": value(kind), "positions": [int(p) for p in positions],
            "paths": [b"".join(reg.open(p).path).hex() for p in positions]}


def state_reads(ports: Sequence[Mapping[str, dict]]) -> dict:
    """The scan's state reads as its registration's `window` states them (``reads``, `NetworkCheck.Period`'s
    `stateRead`), from each member's ``--registered`` ports (`tree_port`: unit ``reads_from + i`` at leaf ``reads[i]``):
    its ``o0`` (the start unit's), ``sin`` and ``o1`` of the state's value, each one run ``{"leaf": l, "units": [lo,
    hi]}``, unit lo + i at leaf l + i. SystemExit when a port's reads are no one run."""
    at: dict[str, dict[int, int]] = {}
    for ps in ports:
        for p in ("o0", "sin", "o1"):
            if p in ps and ps[p]["value"] == value("state"):
                at.setdefault(p, {}).update({ps[p].get(REG.READS_FROM, 0) + i: r for i, r in enumerate(ps[p]["reads"])})
    runs = {}
    for p, got in sorted(at.items()):
        lo, hi = min(got), max(got) + 1
        if got != {u: got[lo] + u - lo for u in range(lo, hi)}:
            raise SystemExit(f"the scan's {p} reads {got} are no one run, unit lo + i at leaf l + i, as its "
                             "registration states them")
        runs[p] = {"leaf": got[lo], "units": [lo, hi]}
    return runs


def tree_refusal(port: dict, reg: ADV.Committed, entry: dict, bcs: list[bytes | None],
                 refs: list[int] | None = None) -> str | None:
    """The host's read of a statement's entry on a scan registration, as Lean's registered reads make it (in
    `warden_rows.refusal`'s order): None when it holds. ``bcs`` is each instance's b‖c and ``refs`` its table row (its
    own by default; a shared row's, `registered.table`, the one its instances read). A b‖c of None is a row the host
    can't name (an output the reference gives none of), which opens no leaf."""
    refs = list(range(len(bcs))) if refs is None else list(refs)
    if entry["value"] != port["value"]:
        return f"registered reads: the port reads {port['value']}, not another value ({entry['value']})"
    n, root = len(reg.rows), bytes.fromhex(port["root"])
    d, scheme = tree_domain(reg), tree_scheme(reg)
    table: dict[int, bytes | None] = {}
    for i, r in enumerate(refs):
        table.setdefault(r, bcs[i])
    for r, (p, path) in enumerate(zip(entry["positions"], entry["paths"], strict=True)):
        nodes, bc = tuple(bytes.fromhex(path[i:i + 128]) for i in range(0, len(path), 128)), table.get(r)
        if bc is None or not scheme.verify_path(ADV.PORT, root, n, p, row_leaf(
                d, p, ADV.INST.tree_leaf(reg.key, bc), _schema(reg)), nodes):
            return f"registered reads: table row {r} does not open the registered root at position {p}"
    base = int(port.get(REG.READS_FROM, 0))
    for i, (r, want) in enumerate(zip(refs, port["reads"], strict=True)):
        if entry["positions"][r] != want:
            return (f"registered reads: unit {base + i} reads position {entry['positions'][r]}, and the verifier's "
                    f"program reads {want}")
    return None


# --- the run --------------------------------------------------------------------------------------------------------

def synthetic(out: Path, setup: Setup, windows: list[int]) -> SimpleNamespace:
    """The synthetic run (`scan_traffic.serve`), each execution phase of `windows` of each link committed by the network
    certifier and its head signed by the driver's test-size LMS key (`warden_rows.attest`), each registration read back
    by the verifier."""
    setup.cls.serves(setup.params)
    reqs = [_request(setup, i, p) for i, p in enumerate(setup.prompts)]
    served = traffic.serve(setup.params, reqs, setup.arrivals, [1 << 20] * len(reqs), setup.responses,
                           max(setup.windows, windows[-1] + 1))
    if served.fails:
        raise SystemExit(f"the warden failed {len(served.fails)} connections")
    out.mkdir(parents=True, exist_ok=True)
    height = max(6, (2 * len(windows) + 1).bit_length())
    signer = LmsSigner.create(lms.keygen(lms.Params(64, 1, height), hashlib.sha512(b"scan-zk").digest()),
                              out / "signer-state.json")
    issuer, sides = Issuer(signer, SESSION), {}
    for side, records, direction in (("ingress", served.ingress, Direction.INGRESS),
                                     ("egress", served.egress, Direction.EGRESS)):
        sides[side] = []
        for w in windows:
            cells, cert = WR.attest(issuer, records[w], setup.params, direction)
            if cert is None:
                raise SystemExit("the test signer refused a link-window")
            registration = WR.registration(cells.head, cert, setup.params)
            head, dom = WR.check_registration(registration, signer.public_key(), SESSION, setup.params,
                                              records[w].link, direction, w)
            if list(cells.rows) != WR.window_rows(records[w], setup.params):
                raise SystemExit(f"{side}: the warden's committed rows of link-window {w} aren't its record's")
            sides[side].append(SimpleNamespace(cells=cells, registration=registration, head=head, dom=dom))
    return SimpleNamespace(source="the synthetic run (scan_traffic.serve), signed with the driver's own key",
                           params=setup.params, session=SESSION, sides=sides, outputs=served.outputs, run_id=None,
                           issuer=issuer, key=signer.public_key(), key_of="the run's (the driver's)", failed=0,
                           fail_buckets=[], failed_why=[])


def _found(run: Path, name: str) -> Path:
    """A network certifier run's file: in the run's directory (`warden_serve.py serve --out`), else its ``witness/``
    (the gate's `warden-rehearse --scan`)."""
    for d in (run, run / "witness"):
        if (d / name).is_file():
            return d / name
    raise SystemExit(f"--warden-run {run}: no {name} there or in its witness/ (warden_serve.py serve writes it)")


def warden_run(run: Path, setup: Setup, windows: list[int], out: Path) -> SimpleNamespace:
    """A warden_serve run's execution phases `windows` as the verifier reads them, refused by name where they don't
    hold: the deployment's schedule (its digest, and the class serving it) and public key; each link's registration of
    each execution phase in ``cells.jsonl`` verified under that key and session (`warden_rows.check_registration`: the
    network certifier's
    signature over the head of the rows' root); and the records' rows under ``cells.salts`` (`warden_rows.Cells.of`)
    that head. The units read those rows. The Programs' outputs (the slot entries' ids) are each connection's ids
    as its egress record carries them (`io_tie.parse_cell`), every egress cell an id-form one or a `FAIL` (a void run:
    ``failed`` the reasons its ``fails`` record has in the execution phases, ``fail_buckets`` the buckets they're in,
    and ``failed_why`` the run's ``check.json`` problems that name a connection failed closed). The tampered round's
    re-signed execution phase is signed by a key of the driver's own (``out``), under which the verifier reads it."""
    import warden_serve as WS

    dep = json.loads(_found(run, "deployment.json").read_text())
    params = W.WardenParams(**dep["params"])
    if params.digest.hex() != dep["digest"]:
        raise SystemExit(f"--warden-run {run}: deployment.json's digest isn't its parameters'")
    try:
        setup.cls.serves(params)
    except ValueError as e:
        raise SystemExit(f"--warden-run {run}: {e} (warden_serve.py rehearse at scan_zk's schedule, PARAMS, or the "
                         "gate's warden-rehearse --scan)")
    key, session = WS.cs.read_public(dep["key"]), bytes.fromhex(dep["session"])
    records = {name: [WS.ar.unpack_record(b, params.cell_bytes) for b in WS.ar.read_blobs(_found(run, f"{name}.records"))]
               for name in WS.CERTIFIED}
    lines = [json.loads(x) for x in _found(run, "cells.jsonl").read_text().splitlines()]
    salts = WS.ar.read_blobs(_found(run, "cells.salts"))
    if len(salts) != len(lines):
        raise SystemExit(f"--warden-run {run}: cells.jsonl has {len(lines)} lines and cells.salts {len(salts)} blobs")
    sides = {}
    for side, name, direction in (("ingress", WS.INGRESS, Direction.INGRESS), ("egress", WS.EGRESS, Direction.EGRESS)):
        sides[side] = []
        for w in windows:
            got = [(doc["registration"], blob) for doc, blob in zip(lines, salts) if "registration" in doc and
                   (doc["registration"]["link"], doc["registration"]["direction"], doc["registration"]["window"])
                   == (WS.LINK, direction.value, w)]
            record = next((r for r in records[name] if r.window == w), None)
            if not got or record is None:
                raise SystemExit(f"--warden-run {run}: {side} link-window {w} has no registration or no record (the "
                                 "signer refused it, or the run ended before it)")
            registration, blob = got[0]
            try:
                head, dom = WR.check_registration(registration, key, session, params, WS.LINK, direction, w)
            except ValueError as e:
                raise SystemExit(f"--warden-run {run}: {side} link-window {w}'s registration doesn't verify under the "
                                 f"deployment's key and session: {e}")
            row_salts, wiring_salt = WR.read_witness(blob)
            cells = WR.Cells.of(WS.LINK, direction, record, params, session, row_salts, wiring_salt)
            if cells.head != head:
                raise SystemExit(f"--warden-run {run}: {side} link-window {w}'s record under cells.salts isn't the rows "
                                 "whose root the warden signed")
            sides[side].append(SimpleNamespace(cells=cells, registration=registration, head=head, dom=dom))
    fails = [((r.window - windows[0]) * params.window + t, sum(f is not None for f in row))
             for r in records[WS.FAILS] if r.window in windows for t, row in enumerate(r.grid)]
    fail_buckets = sorted(t for t, n in fails if n)
    check = next((d / "check.json" for d in (run, run / "witness", run / "out") if (d / "check.json").is_file()), None)
    named = [p.removeprefix("audit: ") for p in (json.loads(check.read_text()).get("problems") or [])
             if "failed closed" in p] if fail_buckets and check else []
    B, outputs = params.cell_bytes, {}
    for rec in sorted(records[WS.EGRESS], key=lambda r: r.window):
        for t, row in enumerate(WR.window_rows(rec, params)):
            for s in range(params.lanes * params.slots):
                if row[s * B] == W.FAIL:
                    continue
                cell = IT.parse_cell(row[s * B:(s + 1) * B])
                if cell is None:
                    raise SystemExit(f"--warden-run {run}: egress bucket {rec.window * params.window + t} slot {s} "
                                     "isn't an id-form cell (io_tie.parse_cell): the run's responses aren't the "
                                     "scan's (warden_serve.py rehearse --ids)")
                if cell.data:
                    outputs.setdefault(cell.stream, []).append(cell.id)
    out.mkdir(parents=True, exist_ok=True)
    signer = LmsSigner.create(lms.keygen(lms.Params(64, 1, 6), hashlib.sha512(b"scan-zk tampered").digest()),
                              out / "tamper-signer-state.json")
    return SimpleNamespace(source=f"warden run {dep.get('run_id')} (warden_serve.py, its deployment's key)",
                           params=params, session=session, sides=sides,
                           outputs={q: tuple(xs) for q, xs in outputs.items()}, run_id=dep.get("run_id"),
                           issuer=Issuer(signer, session), key=signer.public_key(), key_of="the driver's own",
                           failed=sum(n for _, n in fails), fail_buckets=fail_buckets, failed_why=named)


def _tamper(got: SimpleNamespace, egress: list[bytes], T: int, B: int, R: int) -> SimpleNamespace:
    """The tampered round's egress execution phase: the group's first egress row with a `DATA` cell, the last digit of
    that cell's id changed, its execution phase's rows committed afresh and their head signed by ``got.issuer``, the
    registration read back under ``got.key``."""
    t, s = next(((t, s) for t, row in enumerate(egress) for s in range(R) if row[s * B] == DATA), (None, None))
    if t is None:
        raise SystemExit("the group has no DATA cell")
    x = bytearray(egress[t])
    k = s * B + IT.ID_AT + IT.ID_DIGITS - 1
    x[k] = ord("0") + (x[k] - ord("0") + 1) % 10
    w = t // T
    c = got.sides["egress"][w].cells
    rows = list(c.rows)
    rows[t % T] = bytes(x)
    cells = WR.Cells.commit(c.link, c.direction, c.window, rows, c.wiring, c.params, c.session)
    cert = got.issuer.sign(c.link, c.direction, c.window, cells.head.digest())
    if cert is None:
        raise SystemExit("the test signer refused the tampered link-window")
    registration = WR.registration(cells.head, cert, c.params)
    head, dom = WR.check_registration(registration, got.key, c.session, c.params, c.link, c.direction, c.window)
    return SimpleNamespace(cells=cells, registration=registration, head=head, dom=dom, window=w, bucket=t,
                           row=bytes(x), key_of=got.key_of)


def _unread(cls: IS.Class, ch) -> SimpleNamespace | None:
    """The unread round's bucket: the group's first ingress bucket b ≥ 1 at which a slot's Program starts (a request's
    anchor, the slot due and its `occ` 1), the first such slot j, and lane j's slot entries with slot j's at b's
    boundary (row ``entry``) absent, its read bit cleared and its outputs kept (``rows``, which b's link-window
    commits in place of its ``prog<j>``, `window_of`); None when no request is anchored in the group past bucket
    0."""
    for b in range(1, len(ch.progs)):
        reads = [d and s.occ & 1 for d, s in zip(IS.due(cls, ch.scan[b]), ch.scan[b].slots)]
        if any(reads):
            j = reads.index(True)
            entries = IS.prog_decode(cls, ch.progs[b])
            entries[j] = (0, *entries[j][1:])
            rows, k = slot_entries(cls, ch.progs)[j], entry_at(cls, b)
            rows[k] = IS.prog_rows(cls, IS.prog_encode(cls, entries))[j]
            return SimpleNamespace(bucket=b, slot=j, entry=k, q=ch.scan[b].slots[j].q, rows=rows)
    return None


def _u16(data: bytes) -> list[int]:
    return [int.from_bytes(data[i:i + 2], "little") for i in range(0, len(data), 2)]


def _v3c_refused(window: Path, why: str) -> SystemExit:
    return SystemExit(f"--v3c {window}: {why}")


def v3c_state(window: Path, rec: dict, states: list[list[int]]) -> Held:
    """V3c's `SCAN_STATE` (the certifier's `blocks.SCAN_STATE`): its entry in its registration's window activations
    (the ``window`` provenance window-commit writes: ``activations.values``), its rows and salts the prover's
    `SCAN_STATE_FILE`, under the domain that provenance gives (the window circuit's program and the window's run root,
    `registered.domain`), held to the entry's root and domain and to the scan's ``states``."""
    prov = rec.get("window") or {}
    acts = prov.get("activations") or {}
    e = (acts.get("values") or {}).get(SCAN_STATE)
    if not e or e.get("value") != SCAN_STATE:
        raise _v3c_refused(window, f"its registration's window activations register no {SCAN_STATE}")
    n, W = int(e["leaves"]), int(e["words"])
    if n != len(states) or 16 * W != len(states[0]):
        raise _v3c_refused(window, f"its {SCAN_STATE} is {n} rows of {16 * W} bits, not the scan's {len(states)} "
                                   f"of {len(states[0])}")
    dom = REG.domain(acts.get("program", ""), SCAN_STATE, W, n, window=prov.get("run"))
    data = (window / SCAN_STATE_FILE).read_bytes()
    if len(data) != n * (2 * W + ADV.INST.salt_bytes):
        raise _v3c_refused(window, f"{SCAN_STATE_FILE} is {len(data)} bytes, not {n} rows of {W} words and their salts")
    words, at, step = [_u16(data[2 * W * t:2 * W * (t + 1)]) for t in range(n)], 2 * W * n, ADV.INST.salt_bytes
    held = Held(dom, 16 * W, tuple(bit_row((r, 16))[0] for r in words),
                tuple(data[at + step * t:at + step * (t + 1)] for t in range(n)))
    if (held.root.hex(), dom.domain_id.hex()) != (e["root"], e["domain"]):
        raise _v3c_refused(window, f"{SCAN_STATE_FILE} under its provenance's domain isn't its registered "
                                   f"{SCAN_STATE}'s root and domain")
    if bad := [t for t in range(n) if words[t] != _words(states[t])]:
        raise _v3c_refused(window, f"its {SCAN_STATE} rows {bad[:4]} aren't the scan's (row t S_(t-1) from the "
                                   "empty state of link-window 0): it scanned another warden run or link-window")
    return held


def v3c_entries(window: Path, rec: dict, win: dict, cls: IS.Class, entries: list[list[list[int]]]) -> dict[str, Held]:
    """Each lane j's ``prog<j>``: chunk j's o0 of V3c's `SELECT_ENTRY` template, cut into one member a lane of nb
    proof blocks (``<descriptor id>/c<j>``, block b slot j·nb + b's entry, the window's slots lane-major), its rows and
    salts that member's prover body ``m<k>-private.bin`` (each instance's rows of its ports but the wired ones, then
    their salts, `verity.commitments.windows.rows.write`), under its o0 frame domain, held to the root its ``window.json`` and
    its registration's ``roots`` name and to the scan's lane j ``entries``."""
    nb, NE = len(entries[0]), IS.entry_bits(cls)
    ms = [m for m in win.get("members") or () if str(m.get("template", "")).startswith(SELECT_ENTRY + "{")]
    tids = sorted({m["template"] for m in ms})
    shape = [(m["name"], m["n"]) for m in ms]
    if len(tids) != 1 or shape != [(f"{tids[0]}/c{j}", nb) for j in range(cls.NR)]:
        raise _v3c_refused(window, f"its {SELECT_ENTRY} members are {shape}, not one template's {cls.NR} chunks of "
                                   f"{nb} proof blocks, a lane each")
    roots = {r["port"]: r for r in rec.get("roots") or ()}
    out = {}
    for j, m in enumerate(ms):
        if m.get("shared_rows") or m.get("value_rows"):
            raise _v3c_refused(window, f"{m['name']} is a shared-row member, whose body lays its ports out whole")
        wired = set(m.get("wired_rows") or ())
        left = [(p, int(w)) for p, w in m["ports"] if p not in wired]
        names = [p for p, _ in left]
        width, w0 = sum(w for _, w in left), dict(left).get("o0")
        fv, n = m["frame_v3"], int(m["n"])
        if w0 is None or 16 * w0 != NE or fv["schemas"].get("o0") != BitRowLeaf.hiding_schema:
            raise _v3c_refused(window, f"{m['name']}'s o0 isn't a {BitRowLeaf.hiding_schema} row of the scan's {NE}-bit "
                                       "entries")
        data = (window / f"m{m['k']}-private.bin").read_bytes()
        if len(data) != n * (2 * width + ADV.INST.salt_bytes * len(left)):
            raise _v3c_refused(window, f"m{m['k']}-private.bin is {len(data)} bytes, not {n} instances' rows of "
                                       f"{names} and their salts")
        off, k = sum(w for p, w in left[:names.index("o0")]), names.index("o0")
        rows = [_u16(data[2 * (width * i + off):2 * (width * i + off + w0)]) for i in range(n)]
        at, step = 2 * width * n, ADV.INST.salt_bytes
        salts = tuple(data[at + step * (len(left) * i + k):at + step * (len(left) * i + k + 1)] for i in range(n))
        dom = CommitmentDomain(bytes.fromhex(fv["bindings"]["o0"]), int(fv["owner"]), RangeIndexedDomain(0, n),
                               hash="sha512")
        held = Held(dom, NE, tuple(bit_row((r, 16))[0] for r in rows), salts)
        theirs = roots.get(f"{m['name']}/o0") or {}
        if not (held.root.hex() == fv["roots"]["o0"] == theirs.get("root")
                and dom.domain_id.hex() == fv["domain_ids"]["o0"] == theirs.get("domain") and theirs.get("leaves") == n):
            raise _v3c_refused(window, f"m{m['k']}-private.bin's o0 under its frame domain isn't the root and domain "
                                       f"its registration names for {m['name']}/o0")
        if bad := [b for b in range(n) if rows[b] != _words(entries[j][b])]:
            raise _v3c_refused(window, f"{m['name']}'s o0 rows {bad[:4]} aren't the scan's lane {j} slot entries")
        out[f"prog{j}"] = held
    return out


def v3c_trees(window: Path, cls: IS.Class, states: list[list[int]],
              entries: list[list[list[int]]]) -> dict[str, Held]:
    """``--v3c``: the trees the scan reads of V3c's window ``window`` (`prove_path.py window-commit`'s directory), in
    place of its own: ``state`` its `SCAN_STATE` (`v3c_state`) and each ``prog<j>`` chunk j's o0 of its
    `SELECT_ENTRY` template (`v3c_entries`), each refused by name where it isn't the scan's."""
    try:
        rec = json.loads((window / "registration.json").read_text())
        win = json.loads((window / "window.json").read_text())
        return {"state": v3c_state(window, rec, states), **v3c_entries(window, rec, win, cls, entries)}
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise _v3c_refused(window, f"{type(e).__name__}: {e}") from None


def link(out: Path, run: Path | None = None, setup: Setup = W63, group: int = 1, start: int = 0,
         v3c: Path | None = None) -> SimpleNamespace:
    """The run (`synthetic`, or `warden_run`'s of a warden_serve run), link-windows start … start + group − 1 of each
    link registered by the warden and read back by the verifier, and the scan's chain over their rows from the empty
    state (`scan_traffic.chain`): what each link-window's rounds are cut from (`window_of`). The group's negatives are
    placed here, each in one link-window: the chain round's swap in bucket ``setup.A``'s, the unread round's unit
    (`_unread`) and the tampered round's egress link-window (`_tamper`); none in a void run, whose FAILs are in its
    last link-window (the one after a refused link-window has no start the verifier holds). With ``v3c`` the state
    and slot entries' trees are V3c's (`v3c_trees`)."""
    t0 = time.time()
    if group < 1 or start < 0:
        raise SystemExit(f"a group is 1 or more link-windows from link-window 0 on, not {group} from {start}")
    if v3c is not None and (group, start) != (1, 0):
        raise SystemExit(f"--v3c: V3c's window is link-window 0's, a group of 1, not {group} from {start}")
    windows = list(range(start, start + group))
    got = synthetic(out, setup, windows) if run is None else warden_run(run, setup, windows, out)
    cls, T = setup.cls, got.params.window
    ingress, egress = ([x for w in got.sides[s] for x in w.cells.rows] for s in ("ingress", "egress"))
    try:
        ch = traffic.chain(cls, ingress, egress, got.outputs)
    except (ValueError, KeyError) as e:
        raise SystemExit(f"the scan refuses link-windows {windows}' rows: {e!r} (the scan reads warden-form id "
                         "requests at its class, answered id-form: warden_serve.py rehearse --ids)")
    bad = [t for t in range(group * T) if EG.outputs(cls.egress, ch.view(t)) != (EG.ACCEPT, ch.egress[t])]
    void = [t for t, row in enumerate(ch.egress) if any(row[s * cls.B] == W.FAIL for s in range(cls.egress.R))]
    if bad != void:
        raise SystemExit(f"egress buckets {bad} don't compute the warden's rows from the views their ingress buckets "
                         f"wrote, and the warden failed a connection in buckets {void}")
    if void != got.fail_buckets:
        raise SystemExit(f"egress buckets {void} hold a FAIL and the warden's fails record has a reason in buckets "
                         f"{got.fail_buckets}, not the same: a failed connection's reason is in its FAIL's bucket"
                         + (f" ({'; '.join(got.failed_why)})" if got.failed_why else ""))
    if early := sorted({windows[t // T] for t in void if t // T != group - 1}):
        raise SystemExit(f"the warden failed a connection in link-windows {early}, which the scan refuses, and the "
                         "link-window after a refused one has no start the verifier holds: prove the group up to "
                         f"the first of them (--group {early[0] - start + 1})")
    if not 0 <= setup.A < group * T or setup.A % T + 2 >= T:
        raise SystemExit(f"the chain round swaps S_A and S_(A+1), its link-window's state leaves a + 1 and a + 2, "
                         f"a = A mod {T}: 0 <= A < {group * T} and a < {T - 2}, not A = {setup.A}")
    admitted: dict[int, int] = {}
    for st in ch.scan:
        for s in st.slots:
            if s.live:
                admitted.setdefault(s.q, s.admitted)
    return SimpleNamespace(setup=setup, chain=ch, T=T, G=group, windows=windows, params=got.params,
                           session=got.session, source=got.source, run_id=got.run_id, outputs=got.outputs,
                           sides=got.sides, swap=None if void else setup.A,
                           unread=None if void else _unread(cls, ch),
                           tampered=None if void else _tamper(got, ch.egress, T, cls.B, cls.egress.R),
                           ordinals=sorted(admitted), admitted=dict(sorted(admitted.items())), void=void,
                           failed=got.failed, failed_why=got.failed_why, v3c=v3c, run_s=round(time.time() - t0, 1))


def _bits(reg: ADV.Committed, i: int) -> list[int]:
    row = reg.rows[i]
    return [row[k // 8] >> k % 8 & 1 for k in range(reg.n_bits)]


def header_refusal(ports: dict, entries: dict) -> str | None:
    """The first rule of Lean's registered reads (`Flock.Registered.check`): a statement's header names a port
    registered only where the verifier's own ports do, since a header entry carries no root and its rows would open
    none the verifier named. None when it holds."""
    extra = next((p for p in entries if p not in ports), None)
    return None if extra is None else (f"registered reads: the statement names port {extra} registered, and the "
                                       "verifier's registered reads do not")


def registered_file(mr: SimpleNamespace) -> dict:
    """Member `mr`'s `--registered` file, the verifier's own (`stage` writes it, `a0.lean_verify` passes it, and a Lean
    verdict's `registered` reports its roots): each network certifier port at the head `warden_rows.check_registration`
    read under the deployment's key, never a root of the prover's."""
    return WR.reads_file(mr.ports)


def member(at: SimpleNamespace, name: str, mb: SimpleNamespace) -> SimpleNamespace:
    """Round `name`'s member `mb` (`members_of`) of link-window `at` (`window_of`): each unit's input rows (in its
    signature's order, as bits or bytes), the salts its rows, inputs and outputs, are committed under, its header's
    `registered` entries and the verifier's own ports, one for every port of its circuit
    (`Flock.Registered.checkComplete`), the host's read of them (`warden_rows.refusal`, `tree_refusal`: the ports in
    its signature's order, inputs, o0, then the other outputs), each unit's o0 word and its output rows as the
    references compute them (``made``, None where the ingress reference gives none), and the network certifier's row
    registration each of its ports reads (``warden``). An ingress unit's ``prog<j>`` is one table row a slot entry it
    reads of lane j's tree (``refs``, `registered.table`), since a timestep's units read their boundary's; its ``sin``
    reads state leaf t and its o1 opens leaf t + 1, the link-window's last unit's none. The start unit's ``sin`` reads
    the one leaf of the link-window's start (``at.start``: the verifier's own row of the empty state, or the last o1
    of the link-window before), and its o0, that row, is a registered read of state leaf 0, no public word. Lane j's
    block member's unit j·nb + b reads ``x``, slot entry b of the lane, its free input, and its o0, that entry, is a
    registered read of ``prog<j>``'s leaf b, no public word (``outputs`` None). A port no tree or network certifier
    row registration holds (each proof block's public o0, the last proof block's o1, a block member's x) is the
    member's own rows of it (`Frame`, value ``frame``), registered as its frame root in its statement's input set
    ``set_name`` (its template's, `set_of`, which names no round and no execution phase; a block member's its lane's
    `block_set`), which the round's registration names before any coin; a public o0's rows are under the public salts
    (`ok_salts`), every other own row's under fresh ones; ``framed`` names the ports `stage` holds to their frame
    roots (those, and a block member's o0)."""
    ch, cls = at.chain, at.setup.cls
    kind, tp, bs = kind_of(name), mb.template, list(mb.buckets)
    state, view = (at.swapped if kind == "chain" else at.trees["state"]), at.trees["view"]
    progs = {p: at.trees[p] for p in lane_kinds(cls)} | (
        {f"prog{at.unread.slot}": at.unread.tree} if kind == "unread" else {})
    ins, outs = ports_of(cls)[tp]
    if tp == "block":
        p, leaves = f"prog{mb.lane}", [b - mb.base for b in bs]
        reads, warden = {"o0": (p, progs[p], leaves)}, {}
        outputs, made = None, {"o0": [_bits(progs[p], i) for i in leaves]}
    elif tp == "start":
        reads, warden = {"sin": ("start", at.start, [0]), "o0": ("state", state, [0])}, {}
        outputs, made = None, {"o0": [at.claimed or _bits(at.start, 0)]}
    elif tp == "egress":
        side = at.tampered if kind == "tampered" else at.sides["egress"]
        reads = {"state": ("view", view, bs)}
        warden = {"o1": side}
        if any(_bits(view, b) != EG.state_bits(cls.egress, ch.view(b)) for b in bs):
            raise SystemExit("a view leaf isn't its bucket's egress view")
        got = [EG.outputs(cls.egress, ch.view(b)) for b in bs]
        outputs, made = [o for o, _ in got], {"o1": [r for _, r in got]}
    else:
        lanes = {p: [entry_at(cls, b) for b in bs] for p in lane_kinds(cls)}
        o1 = [b + 1 for b in bs]
        if o1[0] < len(state.rows) <= o1[-1]:
            raise SystemExit(f"{mb.name}: bucket {len(state.rows) - 1}'s S_t has no state leaf, and its other units' "
                             "have: a registered port holds every instance's row, so it is a member of its own "
                             "(members_of's last)")
        reads = {"sin": ("state", state, bs), **{p: (p, progs[p], ps) for p, ps in lanes.items()},
                 **({"o1": ("state", state, o1)} if o1[-1] < len(state.rows) else {}), "o2": ("view", view, bs)}
        warden = {"row": at.sides["ingress"]}
        got = [IS.reference(cls, ch.ingress[b], _bits(state, b),
                            [x for p, ps in lanes.items() for x in _bits(progs[p], ps[k])]) for k, b in enumerate(bs)]
        outputs = [o for o, _s, _v in got]
        if kind == "unread":  # the read bit enters ok alone: the unit computes the honest S_t and view
            got = [(o, *ch.outputs(b)) for o, b in zip(outputs, bs)]
        made = {"o1": [s for _o, s, _v in got], "o2": [v for _o, _s, v in got]}
    units = [tuple(ch.ingress[b] if r == "row" else made["o0"][k] if tp in ROWS_O0 else
                   _bits(reads[r][1], reads[r][2][k]) for r, _b in ins) for k, b in enumerate(bs)]
    set_name = at.block_sets[mb.lane] if tp == "block" else set_of(tp)
    sig = (*ins, *(() if tp in ROWS_O0 else (("o0", 16),)), *outs)
    own = {**{r: [u[i] for u in units] for i, (r, _b) in enumerate(ins)}, **made,
           **({} if outputs is None else {"o0": [[o] for o in outputs]})}
    for p, n_bits in sig:
        if p not in reads and p not in warden:
            if any(x is None for x in own[p]):
                raise SystemExit(f"{name}/{mb.name}: a unit's {p} row is none the reference gives, and a port no tree "
                                 "or network certifier row holds is registered as the member's own rows before any "
                                 "coin")
            public = p == scan_ok.PORT and tp not in ROWS_O0
            reads[p] = ("frame", frame(set_name, p, own[p], n_bits, ROW_SCHEMAS.get((tp, p), BitRowLeaf.hiding_schema),
                                       ok_salts(len(bs)) if public else None),
                        list(range(len(bs))))
    salts, entries, ports, bcs, refs = {}, {}, {}, {}, {}
    for p, side in warden.items():
        _w, salts[p], entries[p] = side.cells.statement(bs)
        ports[p] = {**WR.bucket_port(side.head, at.params, mb.base), "reads": bs}
        bcs[p] = [WR.bc(made[p][k] if p in made else side.cells.rows[r], side.cells.salts[r]) for k, r in enumerate(bs)]
    for p, (tk, reg, ps) in reads.items():
        table, refs[p] = REG.table(ps) if p.startswith("prog") else (ps, list(range(len(ps))))
        salts[p], entries[p] = [reg.salts[i] for i in table], tree_entry(tk, reg, table)
        ports[p] = tree_port(tk, reg, ps, mb.base)
        said = made.get(p) or (own[p] if tp == "start" else None)
        bcs[p] = [None if said and said[k] is None else tree_bc(reg, i, said[k] if said else None)
                  for k, i in enumerate(ps)]
    refused = header_refusal(ports, entries)
    for p, _b in sig:
        refused = refused or (WR.refusal(ports[p], warden[p].dom, entries[p], bcs[p]) if p in warden else
                              tree_refusal(ports[p], reads[p][1], entries[p], bcs[p], refs[p]))
    changed = at.tampered.bucket if kind == "tampered" and tp == "egress" else None
    return SimpleNamespace(round=name, template=tp, name=mb.name, base=mb.base, n=len(bs), buckets=bs,
                           units=units, made=made, salts=salts, entries=entries, ports=ports, bcs=bcs, refused=refused,
                           refs={p: r for p, r in refs.items() if r != list(range(len(bs)))}, outputs=outputs,
                           changed=changed, warden={p: s.registration for p, s in warden.items()},
                           lane=mb.lane if tp == "block" else None, set_name=set_name,
                           framed=[p for p, (kind, *_r) in reads.items()
                                   if kind == "frame" or (tp, p) == ("block", "o0")],
                           frames={p: reg for p, (kind, reg, _ps) in reads.items() if kind == "frame"})


def rounds_of(at: SimpleNamespace) -> dict[str, tuple[tuple[str, ...], str]]:
    """The rounds link-window `at` (`window_of`) is proved in, by kind, each with its templates and outcome: the block
    and the honest round (`ROUNDS`), and the group's negatives placed in it (`link`): chain, tampered and unread. A
    void link-window's (the warden failed a connection closed, a `FAIL` in the egress row it signed) the block and
    its honest round alone, refused by Lean's registered reads ("does not open") at its egress member, where the scan
    renders the connection's end. With V3c's trees (``--v3c``) no block round: V3c's own audit proves its slot
    entries. The stale round's link-window (``at.claimed``, `host`) has that round alone."""
    if at.claimed is not None:
        return {"stale": ROUNDS["stale"]}
    placed = {"chain": at.swapped, "tampered": at.tampered, "unread": at.unread}
    rounds = {"block": ROUNDS["block"], "honest": (TEMPLATES, "does not open")} if at.void else \
        {k: v for k, v in ROUNDS.items() if k != "stale" and (k not in placed or placed[k] is not None)}
    return {k: v for k, v in rounds.items() if k != "block" or at.v3c is None}


def targets(at: SimpleNamespace, name: str) -> list[str]:
    """The members a refused round's refusal names: chain's, those of ingress units a, a + 1 and a + 2 (a the swap's
    bucket in the link-window) whose o1 opens a state leaf (the link-window's last unit's opens none); stale's, the
    start unit's; unread's, the member of the ingress unit whose slot entry is unread; tampered's and a void
    link-window's honest round's, the egress member."""
    kind = kind_of(name)
    if kind == "chain":
        a = at.setup.A % at.T
        return sorted({mb.name for mb in members_of(TEMPLATES, at.T) if mb.template == "ingress"
                       for b in mb.buckets if a <= b <= a + 2 and b + 1 < at.T})
    if kind == "unread":
        return [mb.name for mb in members_of(UNREAD, at.T) if at.unread.bucket in mb.buckets]
    return ["start" if kind == "stale" else "egress-0"]


def unread_words(at: SimpleNamespace) -> dict[str, list[int]]:
    """The unread round's public words by member: every ingress unit's 1 but the one at the unread bucket, 0."""
    return {mb.name: [int(b != at.unread.bucket) for b in mb.buckets] for mb in members_of(UNREAD, at.T)}


def predicted(at: SimpleNamespace, name: str, members: list[SimpleNamespace]) -> bool:
    """The host's read of a round before any proof, as the round's: accepted, every read held and every output 1 (the
    slot entries' and the start's none: their o0 is no public word); unread's, every read held and every output 1 but
    the unread bucket's unit's, 0 (`unread_words`); refused, its target members' reads refused ("does not open") and
    every other member's held."""
    outcome = rounds_of(at)[kind_of(name)][1]
    if outcome == "accepted":
        return all(m.refused is None and (m.outputs is None or set(m.outputs) == {1}) for m in members)
    if outcome == "ok 0":
        return all(m.refused is None for m in members) and {m.name: m.outputs for m in members} == unread_words(at)
    t = targets(at, name)
    return all((outcome in (m.refused or "")) if m.name in t else m.refused is None for m in members)


def wired(members) -> list[str]:
    """Where a round's chains don't hold, on its members' b‖c (`member`'s ``bcs``, port to each proof block's): bucket b's
    o1 is bucket b + 1's ``sin`` and its o2 egress bucket b's ``state``, the one leaf of the registered roots both
    commit to (the link-window's last bucket's o1 none). The start unit stands where the bucket before bucket 0
    would: its o0 is bucket 0's ``sin``. Empty when they hold."""
    o1, o2, sin, state = {}, {}, {}, {}
    for m in members:
        for k, b in enumerate(m.buckets):
            if m.template == "egress":
                state[b] = m.bcs["state"][k]
            elif m.template == "start":
                o1[b - 1] = m.bcs["o0"][k]
            else:
                o2[b] = m.bcs["o2"][k]
                for p, at in (("o1", o1), ("sin", sin)):
                    if p in m.bcs:
                        at[b] = m.bcs[p][k]
    return [f"S_{b - 1}" for b in sorted(sin) if o1.get(b - 1) != sin[b]] + \
        [f"view {b}" for b in sorted(state) if o2 and o2.get(b) != state[b]]


def empty_start(program: str, cls: IS.Class) -> REG.Registered:
    """The start of a run's first link-window, as the verifier computes it: the empty state's row as one-stage's
    registered value `CARRY` of the Program with digest ``program``, a row/v2 row under the public salt of a value the
    verifier fills (`registered.supplied_salts`). Nothing of it is the prover's, so the verifier pins its root."""
    return REG.Registered.commit(program, CARRY, [_words(IS.encode(cls, IS.empty(cls)))],
                                 REG.supplied_salts(program, CARRY, 1), schema=BitRowLeaf.hiding_schema)


def window_of(whole: SimpleNamespace, w: int, start, claimed: list[int] | None = None) -> SimpleNamespace:
    """Link-window w of `link`'s run as its rounds register it, alone: its number in the run (``number``), its rows of
    each link (``sides``), the chain's buckets w·T … (w + 1)·T − 1 as its buckets 0 … T − 1, and the prover's trees
    over them, bound to the session and its own ingress head: ``state``, T leaves, leaf t the state bucket t reads and
    leaf 0 the one it starts from; each lane's slot entries at its T / EVERY boundaries; ``view``; with V3c's window,
    its trees in their place (`v3c_trees`). ``start`` is the tree its start unit's ``sin`` opens, leaf 0:
    `empty_start`'s for the run's first link-window, else the link-window before's last o1 (its member's `Frame`).
    The group's negatives placed in it, each None elsewhere: ``swapped``, the chain round's state tree, leaves a + 1
    and a + 2 swapped; ``unread``, the unread round's unit and slot entries' tree; ``tampered``, the tampered round's
    egress rows; and ``void``, its buckets with a FAIL. With ``claimed`` (the stale round) the start unit is given
    that row for its ``sin`` in place of ``start``'s, and the state tree is the one row it copies."""
    cls, T, ch = whole.setup.cls, whole.T, whole.chain
    lo, nb, sides = w * T, T // cls.EVERY, {s: ws[w] for s, ws in whole.sides.items()}
    heads = [sides["ingress"].head]
    sub = SimpleNamespace(ingress=ch.ingress[lo:lo + T], egress=ch.egress[lo:lo + T],
                          view=lambda t: ch.view(lo + t), outputs=lambda t: ch.outputs(lo + t))
    states = IS.window_state(cls, ch.states[1:], lo, T)
    entries = [e[w * nb:(w + 1) * nb] for e in slot_entries(cls, ch.progs)]
    trees = {k: tree(k, r, heads, whole.session)
             for k, r in (("state", states), *zip(lane_kinds(cls), entries), ("view", ch.views[lo:lo + T]))}
    if whole.v3c is not None:
        trees.update(v3c_trees(whole.v3c, cls, states, entries))
    swapped = unread = tampered = None
    if claimed is not None:
        trees["state"] = tree("state", [claimed], [s.head for s in whole.sides["ingress"]], whole.session)
    else:
        if whole.swap is not None and whole.swap // T == w:
            a, rows = whole.swap - lo, list(states)
            rows[a + 1], rows[a + 2] = rows[a + 2], rows[a + 1]
            swapped = tree("state", rows, heads, whole.session)
        if (u := whole.unread) is not None and u.bucket // T == w:
            unread = SimpleNamespace(bucket=u.bucket - lo, slot=u.slot, entry=u.entry - w * nb, q=u.q,
                                     tree=tree(f"prog{u.slot}", u.rows[w * nb:(w + 1) * nb], heads, whole.session))
        if (t := whole.tampered) is not None and t.window == w:
            tampered = SimpleNamespace(**{**vars(t), "bucket": t.bucket - lo})
    return SimpleNamespace(setup=whole.setup, chain=sub, T=T, number=whole.windows[w], params=whole.params,
                           session=whole.session, source=whole.source, trees=trees, start=start, claimed=claimed,
                           sides=sides, ordinals=whole.ordinals, swapped=swapped, unread=unread, tampered=tampered,
                           void=[t - lo for t in whole.void if t // T == w] if claimed is None else [],
                           v3c=whole.v3c, block_sets=[block_set(heads, whole.session, j) for j in range(cls.NR)])


def round_trees(at: SimpleNamespace, name: str, templates: tuple[str, ...]) -> dict:
    """The prover's trees a round's registration names (``scan/<kind>``): the link-window's, the chain round's state
    tree and the unread round's slot entries' in their places; an egress-only round's the view alone, the block
    round's the state and each lane's slot entries, the stale round's the one state row its start unit copies."""
    kind = kind_of(name)
    trees = {**at.trees, **({"state": at.swapped} if kind == "chain" else {}),
             **({f"prog{at.unread.slot}": at.unread.tree} if kind == "unread" else {})}
    reads = {("egress",): ("view",), BLOCK: ("state", *lane_kinds(at.setup.cls)), STALE: ("state",)}.get(templates)
    return {k: v for k, v in trees.items() if reads is None or k in reads}


def host(out: Path, run: Path | None = None, draw: int | None = None, setup: Setup = W63, group: int = 1,
         start: int = 0, v3c: Path | None = None, program: str | None = None) -> SimpleNamespace:
    """The run (`link`'s, of `run` when given, on V3c's trees with `v3c`) and every round's members as the host reads
    them (no lowering, no proof): what the test holds. Each of the group's link-windows is a registration of its own
    (`window_of`), its rounds `rounds_of`'s, named ``<kind>-<its number>`` (`round_name`), in link-window order: its
    start unit's ``sin`` the first's `empty_start` and each later one's the honest round before's ``last`` member's
    o1, the `Frame` that round's record names `CARRY_PORT`; an honest round's chains are wired from the start unit's
    o0. Then, unless the run is void, the stale round (``stale-<last + 1>``, `STALE`): a start unit alone, as a
    link-window after the group's last would register it, its ``sin`` given the state the last link-window started
    from in place of the one it ended in, which opens no leaf of the last record's port ("does not open"). A sample
    (`draw`) changes no round: each is refused by Lean's registered reads, which hold every table row, sampled or
    not. ``program``: the digest of the Program the link-windows are registered under, the scan's own
    (`scan_program`) by default, which the first start's domain and salt bind.

    -> ``whole`` (`link`'s), ``rounds`` (by name, in order: its ``kind``, link-window ``number``, ``templates``,
    ``outcome``, link-window ``at``, ``members`` and ``trees``, `round_trees`) and ``draw``."""
    whole = link(out, run, setup, group, start, v3c)
    cls, T, ch = setup.cls, whole.T, whole.chain
    first = empty_start(program or PA.program_digest(scan_program(cls, TEMPLATES, T)), cls)
    begin = Held(first.domain, IS.state_bits(cls), (bit_row((list(first.rows[0]), 16))[0],), first.salts)
    if begin.root != first.root or _bits(begin, 0) != ch.states[0]:
        raise SystemExit("the empty state's row under one-stage's registered domain isn't the root the verifier pins "
                         "(empty_start), or isn't the state the chain starts from")
    rounds: dict[str, SimpleNamespace] = {}

    def read(at: SimpleNamespace, number: int) -> None:
        for kind, (templates, outcome) in rounds_of(at).items():
            name = round_name(kind, number)
            ms = [member(at, name, mb) for mb in members_of(templates, T, entries_of(cls, T), cls.NR)]
            if not predicted(at, name, ms) or (kind == "honest" and wired(ms)):
                raise SystemExit(f"{name}: the host's read of the round isn't {outcome}: "
                                 f"{[(m.name, m.refused, m.outputs) for m in ms]}, chains {wired(ms)}")
            rounds[name] = SimpleNamespace(name=name, kind=kind, number=number, templates=templates, outcome=outcome,
                                           at=at, members=ms, trees=round_trees(at, name, templates))

    for w in range(group):
        at = window_of(whole, w, begin)
        read(at, at.number)
        begin = next(m for m in rounds[round_name("honest", at.number)].members if m.name == "last").frames["o1"]
    if not whole.void:
        claimed = ch.states[(group - 1) * T]
        if claimed == _bits(begin, 0):
            raise SystemExit("the last link-window ends in the state it started from, so the stale round would claim "
                             "the honest start")
        at = window_of(whole, group - 1, begin, claimed)
        read(at, at.number + 1)
    return SimpleNamespace(whole=whole, T=T, rounds=rounds, draw=draw)


def carry_of(pg: SimpleNamespace, cls: IS.Class, log: list[dict], port: dict) -> SimpleNamespace:
    """A round's start, on both sides, given the verifier's log of the run's earlier records and the start member's
    ``sin`` port as the host read it (``port``, `member`).

    The verifier's (``check``, `verifier.check`'s parameters beside the round's own): it expects `CARRY` registered, one
    row/v2 row of the state's words, and holds it, with no earlier record, to the root it computes itself (`empty_start`,
    ``pinned``), and after one to that record's `CARRY_PORT` in the window-history form, ``{"port": P}``, which names
    the log's last record and no other (``outputs``). Its own ``--registered`` port of the start unit's ``sin`` is
    one-stage's (`registered.reads_file` on its Program, the value's root and domain that port's, ``domains``), and
    the start member's file must be it: SystemExit otherwise.

    The prover's (``record``): the record's ``previous`` and its entry of `CARRY`; and, where the start is the empty
    state's, its opening as the record's `window` gives it (``empty``, `NetworkCheck.Period`'s `startEmpty`): the
    state's ``words`` zero u16 words under the verifier's public ``salt``.

    ``negatives(rec)``: the round's record ``rec`` changed, or checked another way, as ``{name: (record, parameters,
    code)}``, each refused with ``code``: ``another-root``, the entry at a root that isn't the held one (`R7-held`);
    after an earlier record, ``no-earlier-record``, the verifier's log empty (`R7-output`). And, where the log holds
    two records, ``explicit-older`` with no code: the entry at the root and domain of the record before the last's
    `CARRY_PORT`, the verifier naming that record in the explicit form ``{"record": D, "port": P}``, which one-stage
    accepts, so that refusing an explicit form for a carried start is left to whoever reads the records after it."""
    values, schemas = {CARRY: IS.state_bits(cls) // 16}, {CARRY: BitRowLeaf.hiding_schema}
    if log:
        e = next((r for r in log[-1]["roots"] if r["port"] == CARRY_PORT), None)
        if e is None:
            raise SystemExit(f"the log's last record names no {CARRY_PORT}, and the start of the link-window after "
                             "it reads it")
        root, dom, how, empty = e["root"], e["domain"], {"outputs": {CARRY: {"port": CARRY_PORT}}}, None
    else:
        first = empty_start(PA.program_digest(pg.program), cls)
        root, dom, how = first.root.hex(), first.domain.domain_id.hex(), {"pinned": {CARRY: first.root.hex()}}
        empty = {"words": values[CARRY], "salt": first.salts[0].hex()}
    mine = REG.reads_file(pg.program, pg.query, values, {CARRY: root}, domains={CARRY: dom}, schemas=schemas)["ports"]
    if list(mine) != ["sin"] or {**mine["sin"], "reads": mine["sin"]["reads"][:1]} != port:
        raise SystemExit(f"the start member's --registered port of sin isn't the verifier's own "
                         f"(registered.reads_file at the start's root and domain): {port} != {mine}")
    check = {"log": list(log), "registered": values, "schemas": schemas, **how}
    entry = {"value": CARRY, "root": root, "leaves": port["leaves"], "words": port["words"], "domain": dom}

    def negatives(rec: dict) -> dict[str, tuple[dict, dict, str | None]]:
        other = hashlib.sha512(bytes.fromhex(root)).hexdigest()
        out = {"another-root": ({**rec, "registered": [{**entry, "root": other}]}, check, "R7-held")}
        if log:
            out["no-earlier-record"] = (rec, {**check, "log": []}, "R7-output")
        old = next((r for r in log[-2]["roots"] if r["port"] == CARRY_PORT), None) if len(log) > 1 else None
        if old is not None:
            out["explicit-older"] = (
                {**rec, "registered": [{**entry, "root": old["root"], "domain": old["domain"]}]},
                {**check, "outputs": {CARRY: {"record": RG.digest(log[-2]), "port": CARRY_PORT}}}, None)
        return out

    return SimpleNamespace(check=check, record={"previous": RG.digest(log[-1]) if log else None, "registered": [entry]},
                           root=root, domain=dom, empty=empty, negatives=negatives)


# --- the statements ---------------------------------------------------------------------------------------------------

def _entry_signature(S):
    return (("x", Array(IS.entry_bits(S), BL.BIT)),), Tuple(Array(IS.entry_bits(S), BL.BIT))


def _entry_body(B, S, x):
    y = B.batch(BL.NOT, B.batch(BL.NOT, x, axes=(0,)), axes=(0,))
    return Coll(_entry_signature(S)[1], TR._refs([y.refs[k] for k in range(IS.entry_bits(S))]))


SlotEntry = CompositeDefinition(
    "SlotEntry", 1, ("KO", "IDB"), _entry_signature, _entry_body, conformance="exact-model-tested", register=False,
    doc="SlotEntry{KO, IDB}: the stand-in of a slot's Program's entry (io_shell's SlotEntry_v1, as the period check "
        "names it): o0 its input x, an entry row of `ingress_scan.entry_bits`, through two NOTs, so its o0 rows are "
        "any entries the prover gives it, read bit included (scan_zk.py)")


def _defs(cls: IS.Class) -> dict[str, object]:
    st = {k: getattr(cls, k) for k in IS.STATICS}
    sh = cls.egress
    return {"ingress": bind(IS.IngressScanBucket, **st),
            "egress": bind(EG.EgressBucket, B=sh.B, R=sh.R, r=sh.r, NR=sh.NR, K=sh.K, IB=sh.IB, D=sh.D),
            "block": bind(SlotEntry, KO=cls.KO, IDB=cls.IDB), "start": bind(IS.IngressScanStart, **st)}


def scan_root(members: list[tuple[object, int, int]], nb: int, every: int) -> Program:
    """The scan's root in the generator form (`NetworkCheck.Period`'s `scanOf`): each `(template, n, fixed)` of
    `members` one batch over its n units, in order, its first `fixed` parameters inputs of its own (``<i>.<name>``),
    and each later one, lane j's ``prog<j>``, at bucket t (the first unit that reads slot entries bucket 0, each next
    such unit the next bucket) row ⌊t / every⌋ (`entry_at`) of lane j's input ``prog<j>``, its slot entries of nb
    boundaries. A template that reads none (the egress units, the start unit) counts no bucket."""
    sigs = [t.signature() for t, _, _ in members]
    lanes = [len(ps) - fixed for (ps, _), (_, _, fixed) in zip(sigs, members)]
    entry = next(ps[fixed][1] for (ps, _), (_, _, fixed), L in zip(sigs, members, lanes) if L)
    L = max(lanes)

    def signature(S):
        params = tuple((f"{i}.{name}", Array(n, ty)) for i, ((ps, _), (_, n, fixed)) in enumerate(zip(sigs, members))
                       for name, ty in ps[:fixed])
        return (*params, *((f"prog{j}", Array(nb, entry)) for j in range(L))), \
            Tuple(*(Array(n, ret) for (_, ret), (_, n, _) in zip(sigs, members)))

    def body(B, S, *args):
        entries, outs, at, start = args[len(args) - L:], [], 0, 0
        for (t, n, fixed), (ps, _) in zip(members, sigs):
            gathered = [array_of([entries[j][(start + m) // every] for m in range(n)])
                        for j in range(len(ps) - fixed)]
            outs.append(B.batch(t, *args[at:at + fixed], *gathered, axes=(0,) * len(ps)))
            at, start = at + fixed, start + (n if gathered else 0)
        return coll_concat(signature(S)[1], outs)

    defn = CompositeDefinition("ScanZkRoot", 1, ("templates", "counts", "nb", "every"), signature, body,
                               register=False)
    return Program(bind(defn, templates=tuple(PA.template_id(t) for t, _, _ in members),
                        counts=tuple(n for _, n, _ in members), nb=nb, every=every))


def scan_program(cls: IS.Class, templates: tuple[str, ...], T: int) -> Program:
    """The scan's Program of `templates` (`TEMPLATES`) over a link-window of T buckets, in the generator form
    (`scan_root`)."""
    defs, counts = _defs(cls), counts_of(members_of(templates, T))
    return scan_root([(defs[tp], counts[tp], FIXED[tp]) for tp in templates], T // cls.EVERY, cls.EVERY)


def entry_reads(prog: Program, query: dict, members: list[SimpleNamespace], cls: IS.Class, nb: int) -> dict:
    """One-stage's reads of the slot entries off the round's Program (`registered.reads`, each lane's ``prog<j>`` a
    registered value of `entry_bits` bit rows), each unit's ``prog<j>`` row: SystemExit unless each lane's port reads
    its own lane's value at the members' rows (`entry_at`), so the verifier's own Program reads at one root a lane what
    the statements open."""
    try:
        got = {p: (r["value"], r["rows"]) for p, r in REG.reads(
            prog, query, {p: IS.entry_bits(cls) // 16 for p in lane_kinds(cls)},
            {p: BitRowLeaf.hiding_schema for p in lane_kinds(cls)}).items()}
    except ValueError as e:
        raise SystemExit(f"one-stage's reads refuse the scan's Program: {e}") from e
    want = {p: (p, [None] * sum(len(m.buckets) for m in members)) for p in lane_kinds(cls)}
    for m in members:
        for k, b in enumerate(m.buckets if m.template == "ingress" else ()):
            for p in lane_kinds(cls):
                want[p][1][m.base + k] = entry_at(cls, b)
    if got != want:
        raise SystemExit(f"one-stage's reads of the scan's Program aren't each unit's slot entries: {got} != {want}")
    return got


def program(M, cls: IS.Class, templates: tuple[str, ...], T: int, out: Path) -> SimpleNamespace:
    """A round's Program and partition (`members_of`'s templates and counts) and each template's stager, every output
    its own row. The scan's three templates' (`TEMPLATES`) is the generator form (`scan_root`), its one-stage reads
    of the slot entries each unit's (`entry_reads`), and its ``prog<j>``, one a lane, the values it names tied
    (``tied``): every `EVERY` units read one slot entry of each lane, the output of the block that registered it,
    which `NetworkCheck.Period`'s rows form ties to the scan (`progTied`, `wired`), so R8 lets them through and the
    registration verdict publishes them (`rests_on_period_check`): the round's acceptance rests on that period
    check, which refuses the block round's stand-in; every other round's a population of its templates, which names
    none."""
    defs, members = _defs(cls), members_of(templates, T, entries=entries_of(cls, T), lanes=cls.NR)
    counts = counts_of(members)
    nb, scan = T // cls.EVERY, templates == TEMPLATES
    prog = scan_program(cls, templates, T) if scan else \
        M.P.mixed_population_program([(defs[tp], counts[tp]) for tp in templates])
    query = M.P.template_instances_query([defs[tp] for tp in templates])
    part = M.P.partition(prog, query)
    if scan:
        entry_reads(prog, query, members, cls, nb)
    out.mkdir(parents=True, exist_ok=True)
    (out / "partition.json").write_text(json.dumps(part, sort_keys=True))
    (out / "program.json").write_bytes(M.canonical_json(M.encode_program(prog)))
    bound = {"partition": M.P.digest(part), "program": part["program"]}
    stagers = {}
    for tp, (tid, base, cnt) in zip(templates, M.P.ranges(prog, query), strict=True):
        assert tid == M.P.template_id(defs[tp]) and cnt == counts[tp]
        assert base == next(m.base for m in members if m.template == tp)
        t = time.time()
        ins, outs = ports_of(cls)[tp]
        sdir = out / tp
        sdir.mkdir(exist_ok=True)
        n_outs = len(outs) + (tp not in ROWS_O0)
        st = M.CS.UnitStager(defs[tp], f"unit/scan-{tp}", [(r, ROW_SCHEMAS.get((tp, r), M.V2), b, [r]) for r, b in ins],
                             [(M.V2, (j,)) for j in range(n_outs)], circuit=sdir / "circuit.txt", bind=bound)
        st.pin
        stagers[tp] = SimpleNamespace(defn=defs[tp], st=st, base=base, n=cnt, out=sdir, ins=ins, outs=outs,
                                      lower_s=round(time.time() - t, 1))
    return SimpleNamespace(program=prog, query=query, templates=templates, members=members, stagers=stagers,
                           units=sum(counts.values()), partition_file=out / "partition.json",
                           program_file=out / "program.json", tied=lane_kinds(cls) if scan else ())


def port_prefix(defn, mr: SimpleNamespace) -> str:
    """Member mr's ports' prefix in its round's registration: its name, the slot entries' (`BLOCK`) their lane's
    chunk's, ``<descriptor id>/c<j>``, so lane j's o0 is the port `NetworkCheck.Period`'s `wired` holds lane j's
    ``scan/prog<j>`` to (`entries_port`)."""
    return f"{PA.template_id(defn)}/c{mr.lane}" if mr.template in BLOCK else mr.name


def stage(M, pg: SimpleNamespace, mr: SimpleNamespace, out: Path) -> dict:
    """Member mr's statement: each unit's input rows in a shared-row file (each instance its own table row, a slot
    entry's one a boundary, ``mr.refs``), every row, input or output, under the salts and paths of its registered
    entry, in its input set ``mr.set_name``; and the verifier's own `--registered` file. Its outputs are the
    references' (o0 every unit's where it is public, each output row where the reference gives one), and each row's
    b‖c the one the host read. Each framed port's frame root and domain (``mr.framed``) must be its tree's: lane j's
    block member's o0 the lane's ``prog<j>`` tree's, since the scan reads lane j's slot entries only where they are
    one tree, and a member's own rows of a port (`Frame`) its registered root's. Its ports in the registration are
    ``at``'s (`port_prefix`)."""
    t0 = time.time()
    sg = pg.stagers[mr.template]
    st, n = sg.st, mr.n
    nbl = max(3, (-(-n // st.meta["vus_per_block"]) - 1).bit_length())
    if not 25 <= st.meta["k_log"] + nbl <= M.a7.M_MAX:
        raise SystemExit(f"{mr.name}: m = {st.meta['k_log']} + {nbl} is outside 25 .. {M.a7.M_MAX}")
    words = {r: M.np.asarray([WR.words(u[k]) if isinstance(u[k], bytes) else _words(u[k]) for u in mr.units],
                             dtype=M.np.uint16) for k, (r, _b) in enumerate(sg.ins)}
    refs = {r: mr.refs.get(r, list(range(n))) for r in words}
    tables = {r: x[[refs[r].index(i) for i in range(max(refs[r]) + 1)]] for r, x in words.items()}
    set_name, salts = mr.set_name, mr.salts
    got = st.stage(words, out, os.urandom(32), salts=salts, indices=range(mr.base, mr.base + n), set_name=set_name,
                   tables=tables, refs=refs, registered=mr.entries)
    o0 = None if mr.outputs is None else [int(x[0]) for x in M.np.asarray(got["outs"]["o0"])]
    if o0 != mr.outputs:
        raise SystemExit(f"{mr.round}/{mr.name}: the units' outputs {o0} are not the reference's {mr.outputs}")
    made = {p: [M.np.asarray(x, dtype="<u2").tobytes() for x in M.np.asarray(got["outs"][p])] for p, _b in sg.outs}
    for p, col in mr.made.items():
        bad = [mr.base + k for k, x in enumerate(col) if x is not None and made[p][k] != _bytes(x)]
        if bad:
            raise SystemExit(f"{mr.round}/{mr.name}: units {bad}' {p} rows are not the reference's")
    (out / "inst.bin").rename(out / f"inst-{n}.bin")
    (out / "pub.bin").rename(out / f"pub-{n}.bin")
    (out / "tables").mkdir(exist_ok=True)
    os.link(sg.out / "circuit.txt", out / "circuit.txt")
    (out / "registered.json").write_text(json.dumps(registered_file(mr), sort_keys=True))
    m = {"tag": f"{mr.round}-{mr.name}", "name": mr.name, "at": port_prefix(sg.defn, mr), "template": mr.template,
         "buckets": mr.buckets, "dir": out, "meta": st.meta, "pin": st.pin,
         "class": st.class_sha512, "set": set_name, "tables": {r: len(x) for r, x in tables.items()},
         "n": n, "base": mr.base, "defn": sg.defn, "ands": got["ands"], "unit_nnz": got["unit_nnz"],
         "salts": got["salts"], "ports": st.ports, "outputs": o0, "units": mr.units, "ins": sg.ins, "made": made,
         "partition_file": pg.partition_file, "program_file": pg.program_file,
         "registered_file": out / "registered.json", "k_log": st.meta["k_log"], "nbl": nbl, "warden": mr.warden,
         "stage_s": round(time.time() - t0, 1)}
    m["header"] = M.a0.public_header(out, n)
    fv = m["header"]["frame_v3"]
    bad = [p for p in mr.framed if (fv["roots"][p], fv["domain_ids"][p]) != (mr.ports[p]["root"],
                                                                            mr.ports[p]["domain"])]
    if bad:
        raise SystemExit(f"{mr.round}/{mr.name}: ports {bad}' frame roots and domains are not their registered trees': "
                         "a tree's binding is not its frame domain's (frame_binding, FRAME_BINDING)")
    m["bcs"] = shared_bcs(out / f"pub-{n}.bin", st.ports)
    if any(bc is not None and m["bcs"][i][p] != bc for p, col in mr.bcs.items() for i, bc in enumerate(col)):
        raise SystemExit(f"{mr.round}/{mr.name}: a staged row's b‖c isn't the one the host read")
    if not all(M.a7.opens(next(p for p in st.ports if p.name == "o0"), [w], salts["o0"][i], m["bcs"][i]["o0"])
               for i, w in enumerate(o0 or ())):
        raise SystemExit(f"{mr.round}/{mr.name}: a unit's output doesn't open against its public o0 commitment")
    return m


def shared_bcs(path: Path, ports: list) -> list[dict[str, bytes]]:
    """Each instance's b‖c by port from a shared-row public file (its header line, each port's table rows' b‖c,
    port-major in table order, then every instance's refs, u32 little-endian, instance-major, ports in order)."""
    import numpy as np

    head, body = path.read_bytes().split(b"\n", 1)
    h, w = json.loads(head), 128
    n, at = h["instances"], [0]
    for p in ports:
        at.append(at[-1] + int(h["shared_rows"][p.name]))
    refs = np.frombuffer(body[at[-1] * w:at[-1] * w + 4 * n * len(ports)], dtype="<u4").reshape(n, len(ports))
    return [{p.name: body[(at[q] + int(refs[i, q])) * w:(at[q] + int(refs[i, q]) + 1) * w] for q, p in enumerate(ports)}
            for i in range(n)]


def staged_wires(members: list[dict]) -> list[str]:
    """`wired` on a staged round's public files."""
    return wired([SimpleNamespace(template=m["template"], buckets=m["buckets"],
                                  bcs={p.name: [x[p.name] for x in m["bcs"]] for p in m["ports"]}) for m in members])


# --- the challenger's coins ---------------------------------------------------------------------------------------------

def window_root(at: SimpleNamespace) -> str:
    """The root the challenger's session opens on (128 hex digits) for link-window `at` (`window_of`): the SHA-512 of
    the warden's session and the link-window's registrations, one a link, whose heads the warden signed. The gate
    opens a root once, ever, so a warden run's link-window is sampled once."""
    return hashlib.sha512(canonical_json({"format": WINDOW_ROOT, "session": at.session.hex(),
                                          "registrations": {s: x.registration
                                                            for s, x in sorted(at.sides.items())}})).hexdigest()


@contextmanager
def challenger(a, at: SimpleNamespace | None, out: Path):
    """The challenger's session for a link-window's honest round's sample (``--challenger``, as prove_path's
    `records_audit.audit` starts it): `a5_coins.start` on `window_root` for one rank, its evidence copied to ``out``;
    once the link-window's rounds are done the rank's turn ends whether or not it sampled (`Draws.skip`) and the
    session's evidence is published, which the yielded namespace's ``published`` then holds. None without
    ``--challenger``, and for no link-window (``at`` None: the stale round, which has no honest round). Needs
    `WZ._one_stage`'s path (a5_coins is one-stage's)."""
    if not a.challenger or at is None:
        yield None
        return
    import a5_coins as AC

    root = window_root(at)
    manager, turns, finisher = AC.start(a.challenger, root, [0], out, a.main, ssh_key=a.gate_key,
                                        peer=a.peer, key_record=a.challenger_key)
    coins = SimpleNamespace(AC=AC, turns=turns, root=root, challenger=a.challenger, published=None)
    try:
        yield coins
    finally:
        turns.skip(0)
        finisher.join()
        coins.published = turns.evidence()
        manager.shutdown()


def rule_of(k: int, n: int) -> str:
    """The round's stream rule over its ``n`` proof blocks: the stratified rate of ``k`` in ``n`` (`service.Law`), at least
    ceil(k n_s / n) of each template's n_s, so each proof block is sampled with probability at least k/n however many the prover
    registers."""
    return f"stratified-rate:{min(k, n)}/{n}"


def spec_of(pg: SimpleNamespace, members: list[dict], want: dict, schemas: dict[str, str], k: int,
            root: str) -> tuple[S.Spec, list, dict[str, int]]:
    """The honest round's `service.Spec` (prove_path's `a5.stream_of`): its unit the round's templates in order, its
    values the round's ports (each member's frame ports and the scan's trees, under their hiding ``schemas``), its
    public items what the registration shows (N, each template's stratum, the group, the leaf layer and each port's
    layout), its selection the stratified rate of ``k`` in N (`rule_of`), its proof deadline `MEMBER_DEADLINE_S` a member
    and its context the group's root. -> (the spec, its proof blocks, `Stream.register`'s counts: each template's proof
    blocks, over its members)."""
    units = [pg.stagers[tp].defn for tp in pg.templates]
    n = sum(m["n"] for m in members)
    law = S.Law(rule_of(k, n))
    named = S.selection_law(law, n, S.selection_sizes(law, pg.program, pg.query))
    strata = [s["template"] for s in DR.derive(pg.program, pg.query, named,
                                               S.selection_table(law, pg.program, pg.query, None)).get("strata") or ()]
    public = [("population", "N, the round's proof units"), *((f"population:{t}", "a template's stratum") for t in strata),
              ("window", "the round's provenance: its link-window of the warden's run"),
              ("leaf_layer", "the leaf layer over the members' public files"),
              *((f"{x}:{p}", "a port's layout") for p in sorted(want) for x in ("leaves", "domain"))]
    check = S.Check(tuple(PA.program_digest(Program(u)) for u in units), pg.query,
                    proof_deadline=float(MEMBER_DEADLINE_S * len(members)), tied=pg.tied)
    spec = S.Spec(tuple(S.Value(p, schemas[p], "serve") for p in sorted(want)), law, check,
                  tuple(S.Public(item, "structure", why) for item, why in public), bytes.fromhex(root))
    counts: dict[str, int] = {}
    for m in members:
        tid = PA.template_id(m["defn"])
        counts[tid] = counts.get(tid, 0) + m["n"]
    return spec, units, counts


def stream_of(a, spec: S.Spec, units: list, pg: SimpleNamespace, coins, held: dict | None = None) -> S.Stream:
    """The honest round's `service.Stream` on ``coins``: the challenger's own copy of the round's Program is the
    generator form the round registers (`scan_root`, ``Stream(program=)``), not its templates' mixed population.
    ``held``: the challenger's key and the verifier's copy of main's history the stream is made with
    (`a5_coins.held`), which it reads every outcome's coin evidence under."""
    return S.Stream(spec, units, budget=1, delta=a.delta, challenger=CHALLENGER, coins=coins, program=pg.program,
                    **(held or {}))


def coins_outcome(a, coins: SimpleNamespace, stream_on, counts: dict, stream: S.Stream, draw: S.Draw,
                  rec: dict, want: dict, leaf_layer: str, read: bytes, segment: dict | None, sessions: list[str],
                  verdicts: dict, statement: dict, given: dict) -> tuple[dict | None, dict, dict]:
    """The honest sample's outcome of record once the challenger's session has ended (prove_path's `a5.coins_outcome`):
    its proof delivered and its outcome with the coins' evidence (`a5_coins.evidence`), the Lean's whole run (`Run.run`,
    its window `window_root`) held to the Python's, the audit failing ``coins`` where
    `coins.coins_reason` refuses; then each altered evidence (`a5_coins.negatives`, each with the key and the copy of
    main's history its stream holds, and the evidence for another window)
    as its own stream's outcome on the honest sample's bytes (`a5_coins.Replay`), refused where `coins_reason` refuses and
    the Lean's run refuses at its select (``stream_on``: the round's stream on given coins and what it holds, `stream_of`). -> (the audit
    record, None where the stream refused or holds the outcome; the negatives; the coins' record)."""
    AC, root, after = coins.AC, coins.root, bytes.fromhex(draw.registration.digest)
    published = coins.turns.evidence()
    ev = AC.evidence(published, segment) if segment is not None else None

    def outcome(s: S.Stream, d: S.Draw, evidence: dict | None) -> tuple[S.Outcome, dict | None]:
        late = s.deliver(d, sessions)
        return (late if late is not None else s.outcome(d, verdicts, statement, coins=evidence, **given)), \
            s.records[-1]["audit"]

    family, ours = stream.family(draw), AC.held(coins.turns)

    def reason(evidence: dict | None, window: str = root, held: dict = ours) -> str | None:
        return CC.coins_reason(held["key"], held["main"], evidence, window, family, after, read, AC.lms_verify)

    why = (published.get("error") or "the draw committed no segment") if ev is None else reason(ev)
    try:
        o, result = outcome(stream, draw, ev)
        told = {"stream": stream.digest, "record": o.record, "accepted": o.accepted, "codes": list(o.verdict.codes)}
    except (S.Refused, S.Pending) as e:
        result, told = None, {"stream": stream.digest, "accepted": False, "refused": AC._refusal(e)}
    if result is not None and why is not None:
        result = {**result, "accepted": False, "profile": None, "integrity_profile": None, "wrong_units": None,
                  "checks": {**result["checks"], "coins": {"ok": False, "codes": ["coins"], "detail": {"coins": why}}}}
    neg: dict[str, dict] = {}
    bad = {**AC.negatives(ev, coins.turns.pre(), **ours), "coins-for-another-window": (ev, ours)} if ev is not None else {}
    for name, (evidence, held) in bad.items():
        window = "00" * 64 if name == "coins-for-another-window" else root
        s = stream_on(AC.Replay(read, family), held)
        try:
            d = s.select(s.register(rec, want, leaf_layer=leaf_layer, window_root=window, counts=counts))
            if d.draw != draw.draw:
                raise ValueError(f"the replayed draw {d.units} is not the honest draw {draw.units}")
            got, audit = outcome(s, d, evidence)
            accepted, side = got.accepted, AC.lean_side(audit)
        except (S.Refused, S.Pending, ValueError, AC.Ch.CoinError) as e:
            accepted, side = None, {"refused": AC._refusal(e)}
        py = reason(evidence, window, held)
        step = ((side or {}).get("step"), (side or {}).get("codes"))
        neg[name] = {"refused": py is not None and accepted is False and step == AC.SELECT_COINS, "accepted": accepted,
                     "python": py, "lean": side}
    checks = (result or {}).get("checks") or {}
    record = {"challenger": coins.challenger, "source": "challenger", "window_root": root, "segment": segment,
              "read": read.hex(), "coins_reason": why, "outcome": told,
              "run": None if result is None or not result["verifiers"]["agree"] else checks.get("run", {"ok": True})}
    return result, neg, record


# --- the audit ----------------------------------------------------------------------------------------------------------

def warden_roots(members) -> list[dict]:
    """The registration's ``rows/<direction>/<w>`` roots: each network certifier row registration its members' network
    certifier ports read (an ingress proof block's ``row``, an egress proof block's ``o1``), its root and domain as the
    network certifier registered them (`warden_rows.registration`, read back by the verifier), its T rows the leaves. A
    period check (`NetworkCheck.Period`'s rows form) holds an execution phase's answer to its rows by these, by root and
    domain.

    A port is never the row registration's value (``warden/<direction>/<w>``, the name Lean reports the reads under):
    the audit takes a root port named as a value for a registered value of the record (`audit.session_registered`),
    which this registration lists none of, and refuses the round."""
    out = {}
    for m in members:
        for doc in m["warden"].values():
            p = f"rows/{doc['direction']}/{doc['window']}"
            out[p] = {"port": p, "root": doc["root"], "leaves": doc["rows"], "domain": doc["domain"]}
    return [out[p] for p in sorted(out)]


def _free_port() -> int:
    """A loopback port the OS has free now. A pool's slots share a machine, and the nodes' 7811-7899 are the challenger
    gate's tunnels, so a fixed port collides across runs."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def audit(M, a, at: SimpleNamespace, pg: SimpleNamespace, name: str, members: list[dict], trees: dict, out: Path,
          port: int, k: int | None = None, coins: SimpleNamespace | None = None,
          carry: SimpleNamespace | None = None) -> dict:
    """One round registered (its members' ports under their prefix, `port_prefix`, each a frame domain of a leaf an
    instance or a shared port's table row; the scan's registrations' roots; and the warden's row registrations its
    warden ports read, `warden_roots`, before any coin), its units sampled by the verifier's Lean (`draw --stratified k`
    on a development challenger's coins, `a0.challenger_draw`; every unit when `k` is None) or, given the challenger's
    ``coins``, by the round's `service.Stream` on them (`stream_of`; `a5_coins.RankCoins`, rank 0), each member's
    sampled proof blocks proved --zk behind the ZK server and held by Lean with the verifier's own registered reads;
    then each held unit's public ``o0`` (none of the slot entries', `ROWS_O0`) opened under the salt the prover sends. A
    member with no unit sampled runs no session. Member j's session listens on ``port + j``, or with ``port`` 0 on a
    port the OS picks free. Writes what the verifier received under ``out / "verifier"``. An accepted audit carries its
    charge terms (`charge_terms`) and each member's cost per sampled proof block; on the challenger's coins, the outcome
    of record is the stream's (`coins_outcome`). Every outcome carries the tied values R8 let through
    (``rests_on_period_check``, the registration verdict's), on whose period check its acceptance rests. The record
    of a round with a start unit (``carry``, `carry_of`) is a v2 record, its ``previous`` the log's last record and
    its one registered value the start, checked with the verifier's log and its hold of that value; its `window`
    opens a start that is the empty state's (``empty``); its outcome carries the record's changed forms' codes
    (``registration_negatives``) and the root the verdicts of record opened the start unit's ``sin`` against
    (``carried_start``, `audit.session_registered`)."""
    from prove_leg import frame_ports  # one-stage's (`WZ._one_stage` puts it on the path)

    a0, a7, P, R, V = M.a0, M.a7, M.P, M.R, M.V
    out.mkdir(parents=True, exist_ok=True)
    want, roots, publics, schemas = {}, [], {}, {}
    for m in members:
        fv = m["header"]["frame_v3"]
        for p_, w in frame_ports(m["set"], m["meta"], m["n"], {p: range(k) for p, k in m["tables"].items()}).items():
            q = f"{m['at']}/{p_}"
            want[q], schemas[q] = w, fv["schemas"][p_]
            roots.append({"port": q, "root": fv["roots"][p_], "leaves": w["leaves"], "domain": fv["domain_ids"][p_]})
        publics[m["name"]] = a0.sha512_file(m["dir"] / f"pub-{m['n']}.bin")
    for kind, reg in trees.items():
        d = tree_domain(reg).domain_id.hex()
        want[f"scan/{kind}"] = {"leaves": len(reg.rows), "domain": d}
        schemas[f"scan/{kind}"] = BitRowLeaf.hiding_schema
        roots.append({"port": f"scan/{kind}", "root": reg.root.hex(), "leaves": len(reg.rows), "domain": d})
    for r in warden_roots(members):
        want[r["port"]] = {"leaves": r["leaves"], "domain": r["domain"]}
        schemas[r["port"]] = WR.SCHEMA
        roots.append(r)
    leaf_layer = hashlib.sha512(M.canonical_json(publics)).hexdigest()
    n = sum(m["n"] for m in members)
    k = n if k is None else k
    rule = f"stratified:{k}" if coins is None else rule_of(k, n)
    law, table = (M.D.parse_design(rule), None) if coins is None else S.params_law(rule, pg.program, pg.query, None)
    win = {"kind": "stand-in", "of": f"round {name}: link-window {at.number} of {at.source}, its ingress and egress "
                                     f"at {at.setup.name}'s class (scan_zk.py)",
           "members": {m["name"]: {"template": P.template_id(m["defn"]), "units": [m["base"], m["base"] + m["n"]]}
                       for m in members},
           "link": "registered: each warden row port, input or output, at the warden's signed root of its "
                   "link-window, unit base + k row k of the member's rows; the scan's state, each lane's prog<j> and "
                   "view, inputs and outputs, at the prover's registered roots (scan/*)"}
    reads = state_reads([json.loads(Path(m["registered_file"]).read_text())["ports"] for m in members])
    win |= ({"reads": {"scan/state": reads}} if reads else {}) | (
        {"empty": {CARRY: carry.empty}} if carry and carry.empty else {})
    rec = R.record(pg.program, pg.query, law, a0.SCHEME, roots, win, leaf_layer=leaf_layer, work_table=table,
                   **(carry.record if carry else {}))
    (out / "registration.json").write_text(json.dumps(rec, indent=1, sort_keys=True))

    def registration(r: dict, given: dict) -> dict:
        return V.check(r, pg.program, pg.query, law, a0.SCHEME, want, leaf_layer=leaf_layer, work_table=table,
                       tied=pg.tied or None, **given).to_json()

    checks = {"registration": registration(rec, carry.check if carry else {})}
    for m in members:
        checks[f"statement/{m['name']}"] = a7.statement_checks({**m, "name": m["at"]}, rec, want)
    receipt = R.receipt(rec, "one-stage-e2e verifier (scan_zk.py)")
    res: dict = {"checks_before_draw": {c: v["ok"] for c, v in checks.items()},
                 "rests_on_period_check": checks["registration"].get("rests_on_period_check", [])}
    if carry:
        res["registration_negatives"] = {x: {"codes": registration(r, given)["codes"], "expected": code}
                                         for x, (r, given, code) in carry.negatives(rec).items()}
    vdir = out / "verifier"
    vdir.mkdir(exist_ok=True)
    if not all(v["ok"] for v in checks.values()):
        return {**res, "checks": checks, "accepted": False, "window_accepted": False, "verdicts": {}, "why": {},
                "opened": {}, "failed": {c: v.get("codes") for c, v in checks.items() if not v["ok"]}}
    stream = None
    if coins is None:
        p = a0.challenger_draw(a.lean, ["--program", str(pg.program_file), "--partition", str(pg.partition_file),
                                        "--stratified", str(k)], rec["law"], rec["population"], R.digest(rec),
                               out / "draw.log")
        line = next((x for x in p.stdout.splitlines() if x.startswith("DRAW ")), None)
        if line is None:
            raise SystemExit(f"{name}: flock-verify draw gave no draw (rc {p.returncode}): {p.stderr[-400:]}")
        lean_draw = json.loads(line.split(" ", 1)[1])
    else:
        spec, units, counts = spec_of(pg, members, want, schemas, k, coins.root)
        rank = coins.AC.RankCoins(coins.turns, 0)
        draw = refused = None
        try:
            stream = stream_of(a, spec, units, pg, rank, coins.AC.held(coins.turns))
            rank.plan(coins.AC.need(rule, pg.program, pg.query, rec["population"]))
            reg = stream.register(rec, want, leaf_layer=leaf_layer, window_root=coins.root, counts=counts)
            rank.commit(bytes.fromhex(reg.digest))
            draw = stream.select(reg)
        except S.Refused as e:
            refused = e.verdict.to_json()
        except (ValueError, coins.AC.Ch.CoinError, V.LeanError) as e:
            refused = {"ok": False, "codes": ["stream"], "detail": {"stream": str(e)}}
        finally:
            segment = rank.drawn()
        read = bytes(rank.read)
        (out / "segment.json").write_text(json.dumps({"challenger": coins.challenger, "window_root": coins.root,
                                                      "segment": segment, "read": read.hex(), "need": rank.need},
                                                     sort_keys=True))
        if draw is None:
            coins.turns.skip(0)
            log(f"{name}: the stream drew nothing on the challenger's coins: {refused}")
            for f in (out / "registration.json", out / "segment.json"):
                os.link(f, vdir / f.name)
            return {**res, "accepted": False, "window_accepted": False, "verdicts": {}, "why": {"draw": refused},
                    "opened": {}, "draw": {"law": rule, "coins": coins.challenger, "refused": refused}}
        lean_draw, receipt = dict(draw.draw), dict(draw.registration.receipt)
        (out / "receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True))
    (out / "draw.json").write_text(json.dumps(lean_draw, sort_keys=True))
    proved, served, n_inst, verdicts, statement, sessions, session_roots = [], [], 0, {}, {}, [], {}
    for j, m in enumerate(members):
        local = sorted(u - m["base"] for u in lean_draw["units"] if m["base"] <= u < m["base"] + m["n"])
        if not local:
            log(f"{name}/{m['name']}: no unit drawn, no session")
            continue
        sdir = out / f"session-{m['name']}"
        record, live, proc = M.a8.session(a, m, sdir, local, port + j if port else _free_port(), env=WZ.firewall(M, m))
        draw_t = (record or {}).get("unit_draw") or {}
        v = V.check_draw(draw_t, {"population": m["n"]}, {"law": "subset", "k": len(local)})
        verdicts[f"m0-verifier/{m['name']}"] = {
            "accepted": ((record or {}).get("verdict") or {}).get("accepted") is True
            and (live or {}).get("roots") == m["header"]["frame_v3"]["roots"] and v.ok,
            "why": ((record or {}).get("verdict") or {}).get("why"), "draw_check": v.to_json()}
        lv = a0.lean_verify(a.lean, m, m["n"], sdir, zk=True)
        verdicts[f"lean-verify/{m['name']}"] = {**lv, "tool": "flock-verify verify --zk --archive --partition "
                                                              "--program --registered"}
        proved += a0.proved_units(m["header"], live or {})
        served += [m["base"] + u for u in draw_t.get("units", [])]
        n_inst += (live or {}).get("instances", 0)
        session_roots.update({f"{m['at']}/{p_}": r for p_, r in ((live or {}).get("roots") or {}).items()})
        timing = {x: (live or {}).get(x) for x in TIMING}
        statement[m["name"]] = {"session_s": proc["wall_s"], "timing": timing, "process": proc,
                                "lean_verify_s": lv.get("verify_s"), "drawn": len(local),
                                **{x: (live or {}).get(x) for x in ("circuit_sha512", "statement_digest", "sigma")}}
        sessions += [a0.sha512_file(f) for f in sorted((sdir / "sessions").glob("*/session.json"))]
        log(f"{name}/{m['name']}: {len(local)} units drawn, Lean {lv.get('accepted')} {lv.get('why') or ''}, "
            f"prove {timing.get('prove_total_s')} s, session {proc['wall_s']} s")
    neg, told = {}, None
    if stream is None:
        result = V.audit_record(rec, receipt, lean_draw, rec["law"], sorted(proved), n_inst, statement, verdicts,
                                a.delta, sessions=sessions, served=sorted(served), session_roots=session_roots)
    else:  # every verdict of record names the sessions delivered (`service.check_verdicts`)
        verdicts = {x: {**y, "sessions": sessions} if y.get("of_record") else y for x, y in verdicts.items()}
        given = {"proved_units": sorted(proved), "instances": n_inst, "served": sorted(served),
                 "session_roots": session_roots}
        result, neg, told = coins_outcome(a, coins, lambda c, held: stream_of(a, spec, units, pg, c, held), counts,
                                          stream, draw, rec, want, leaf_layer, read, segment, sessions, verdicts,
                                          statement, given)
        (out / "coins.json").write_text(json.dumps({**told, "negatives": neg}, indent=1, sort_keys=True, default=str))
        log(f"{name}: the challenger's coins ({coins.challenger}): outcome {told['outcome']}, coins_reason "
            f"{told['coins_reason']}, run {told['run']}, negatives refused "
            f"{ {x: y['refused'] for x, y in neg.items()} }")
        result = result or {"accepted": False, "outcome": told["outcome"]}
    (out / "audit.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    held = set(proved) if result.get("accepted") else set()
    public = {m["base"] + i for m in members if m["outputs"] is not None for i in range(m["n"])}
    opened, sent = {}, []
    for m in [m for m in members if m["outputs"] is not None]:
        o0 = next(p for p in m["ports"] if p.name == "o0")
        opened[m["name"]] = [w if (m["base"] + i) in held and M.a7.opens(o0, [w], m["salts"]["o0"][i], m["bcs"][i]["o0"])
                             else None for i, w in enumerate(m["outputs"])]
        sent += [{"unit": m["base"] + i, "o0": w, "salt/o0": m["salts"]["o0"][i].hex()}
                 for i, w in enumerate(m["outputs"]) if (m["base"] + i) in held]
    (vdir / "openings.json").write_text(json.dumps(sent, indent=1))
    for side, s in at.sides.items():
        (vdir / f"warden-registration-{side}-{at.number}.json").write_text(json.dumps(s.registration, indent=1,
                                                                                      sort_keys=True))
    if kind_of(name) == "tampered":
        (vdir / f"warden-registration-egress-{at.number}-tampered.json").write_text(
            json.dumps(at.tampered.registration, indent=1, sort_keys=True))
    streamed = (out / "receipt.json", out / "segment.json", out / "coins.json") if stream is not None else ()
    for f in (out / "registration.json", pg.partition_file, pg.program_file, out / "draw.json", out / "audit.json",
              *streamed):
        os.link(f, vdir / f.name)
    for m in members:
        for f in (m["dir"] / "circuit.txt", m["dir"] / f"pub-{m['n']}.bin", m["registered_file"]):
            os.link(f, vdir / f"{m['name']}-{f.name}")
        for f in sorted((out / f"session-{m['name']}" / "sessions").rglob("*")):
            if f.is_file():
                os.link(f, vdir / f"{m['name']}-session-{f.parent.name}-{f.name}")
    got = [w for col in opened.values() for w in col if w is not None]
    window_ok = bool(result.get("accepted")) and len(got) == len(held & public) and all(w == 1 for w in got)
    source = "a development challenger's (flock-verify draw, a0.challenger_draw)" if stream is None else \
        f"the auditor's challenger's ({coins.challenger}: a5_coins, the round's service.Stream)"
    return {**res, "accepted": bool(result.get("accepted")), "window_accepted": window_ok, "opened": opened,
            "draw": {"law": rule, "coins": source, "units": lean_draw["units"], "of": rec["population"]},
            **({"coins": told, "coin_negatives": {x: y["refused"] for x, y in neg.items()}} if told else {}),
            **({"carried_start": AU.session_registered(verdicts, rec)} if carry else {}),
            "audit": {x: result.get(x) for x in ("accepted", "complete", "wrong_units")},
            "failed": {c: v.get("codes") for c, v in (result.get("checks") or {}).items() if not v.get("ok")},
            "charge": charge_terms(M, result, members, a.delta),
            "per_unit": per_unit(members, statement),
            "verdicts": {x: y.get("accepted") for x, y in verdicts.items()},
            "why": {x: y.get("why") for x, y in verdicts.items()}, "timing": statement}


def charge_terms(M, result: dict, members: list[dict], delta: float) -> dict | None:
    """An accepted audit's terms (prove_path's `charge_terms`): ε, at most ``wrong_units.bound`` of its ``of`` proof
    blocks wrong except with probability δ; λ_δ (``location_bits``), β_δ (``harm_bound``: each template's proof blocks
    weighing the bits one of its proof blocks commits, `partition._template_bits`) and ``exfiltration_bound`` = β_δ +
    λ_δ. None where the audit isn't accepted, which bounds nothing."""
    if not result.get("accepted"):
        return None
    p = AU.profile(result["registration"], result["draw"], delta)
    bits = {M.P.template_id(m["defn"]): float(M.P._template_bits(m["defn"])) for m in members}
    wu = result.get("wrong_units") or {}
    return {"delta": delta, "wrong_units": wu, "epsilon": wu["bound"] / wu["of"] if wu.get("of") else None,
            "template_bits": bits, "location_bits": location_bits(p), "harm_bound": p.harm_bound(bits),
            "exfiltration_bound": exfiltration_bound(p, bits)}


def per_unit(members: list[dict], statement: dict) -> dict[str, dict]:
    """Each member's session per sampled proof block (prove_path's `per_unit`): Flock's ``prove_total_s``, the session's wall
    and Lean's verify, over the proof blocks the session proved; and a proof block's ANDs and m."""
    out = {}
    for m in members:
        t = statement.get(m["name"]) or {}
        n, tm = t.get("drawn") or 0, t.get("timing") or {}
        pb = tm.get("proof_bytes")
        out[m["name"]] = {"units": n, "ands": m["ands"], "m": m["k_log"] + m["nbl"],
                          "proof_bytes": sum(pb) if isinstance(pb, list) else pb,
                          **{f"{x}_per_unit": round(v / n, 3) if n and isinstance(v, int | float) else None
                             for x, v in (("prove_s", tm.get("prove_total_s")), ("session_s", t.get("session_s")),
                                          ("lean_verify_s", t.get("lean_verify_s")))}}
    return out


def window_answer(reg: dict, record: dict | None, delta: Fraction, reason: str, rests: Sequence[str] = ()) -> dict:
    """One-stage's answer for the group's registration ``reg`` as a period check reads an execution phase's scan answer
    (`NetworkCheck.Period`'s rows form, `pouw.core_check.answer_of`): from an accepted audit record (``record``,
    `audit`'s ``audit.json``), its sample and proved units with one-stage's profile and bound on incorrect proof
    blocks at ``delta`` (`audit.profile`, `audit.wrong_units`: every unit sampled, the bound 0), and the tied values
    its registration's R8 let through when there are any (``rests``, the registration verdict's
    ``rests_on_period_check``), on whose period check the acceptance rests; else a reject naming ``reason``."""
    commitment = RG.digest(reg)
    if record is None:
        return {"commitment": commitment, "profile": None, "protocol": RG.PROTOCOL, "public": None, "reason": reason,
                "verdict": "reject"}
    if record.get("accepted") is not True or RG.digest(record["registration"]) != commitment:
        raise SystemExit("an accepted scan's audit record is for another registration, or not accepted")
    draw = record["draw"]
    profile = AU.profile(reg, draw, delta).to_json()
    profile["delta"] = str(delta)
    wrong = AU.wrong_units(reg, draw, delta)
    return {"commitment": commitment, "profile": profile, "protocol": RG.PROTOCOL,
            "public": {"bound": wrong["bound"], "drawn": sorted(draw["units"]),
                       "per_template": wrong.get("per_template"), "proved": sorted(record["proved_units"]),
                       **({"rests_on_period_check": list(rests)} if rests else {})},
            "reason": None, "verdict": "accept"}


def _answer(out: Path, name: str, res: dict, delta: Fraction, what: str) -> tuple[dict, dict]:
    """Round `name`'s registration (its files under ``out / name``) and its answer (`window_answer`), a refusal
    naming the members Lean refused, else the checks that failed."""
    reg = json.loads((out / name / "registration.json").read_text())
    f = out / name / "audit.json"
    record = json.loads(f.read_text()) if res.get("window_accepted") and f.exists() else None
    why = "; ".join(f"{k.split('/', 1)[1]}: {v}" for k, v in sorted(res.get("why", {}).items())
                    if v and k.startswith("lean-verify/"))
    failed = "; ".join(f"check {c}: {', '.join(codes or ())}" for c, codes in sorted((res.get("failed") or {}).items()))
    reason = f"{what} refused: " + (why or failed or f"audit {res.get('audit')}, outputs {res.get('opened')}")
    return reg, window_answer(reg, record, delta, reason, res.get("rests_on_period_check") or ())


def answers(ats: Sequence[SimpleNamespace], out: Path, results: Mapping[str, dict], delta: Fraction) -> dict:
    """``answers.json`` (`ANSWERS`) of the honest rounds' audits of link-windows ``ats`` (``results``, by round, each
    round's files under ``out / <round>``): for each link-window W, in order, the scan's registration of W, its answer
    (`window_answer`: accepted where the round's outputs all open to 1; a refused round's, a void link-window's, names
    the members Lean refused) and the warden's ingress and egress row registrations of W, the rows form's evidence of
    W but its certificates. With W's block round's audit, the compute audit whose lanes' o0 are W's scan's
    ``prog<j>`` (``audits``, in link-window order): its registration and answer, and the terms a period's parameters
    give it, its Program's descriptor, its free inputs (every bit of every slot entry, the stand-in's ``x``: it
    copies its input) and no steps. -> each link-window's answer in short, by its number."""
    audits, windows, said = [], [], {}
    for at in ats:
        cls, honest, block = at.setup.cls, round_name("honest", at.number), round_name("block", at.number)
        reg, answer = _answer(out, honest, results[honest], delta, f"the scan's audit of link-window {at.number}")
        if block in results:
            breg, banswer = _answer(out, block, results[block], delta, "the block's audit")
            audits.append({"answer": banswer, "free": entries_of(cls, at.T) * IS.entry_bits(cls),
                           "program": json.loads((out / PROGRAMS[BLOCK] / "program.json").read_text()),
                           "registration": breg, "steps": 0})
        windows.append({"window": at.number, "registration": reg, "answer": answer,
                        "registrations": [at.sides[s].registration for s in ("ingress", "egress")]})
        said[at.number] = {"verdict": answer["verdict"], "bound": (answer["public"] or {}).get("bound"),
                           "reason": answer["reason"], "commitment": answer["commitment"]}
    doc = {"format": ANSWERS, "delta": str(delta), "audits": audits, "windows": windows}
    (out / "answers.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    return said


def expected(at: SimpleNamespace, name: str, res: dict) -> bool:
    """An accepted round: accepted, every output 1; the unread round: its proofs held and its audit accepted, and
    the unread bucket's unit's output opened to 0 (`unread_words`), so the link-window is refused; a refused one
    (`rounds_of`): refused by Lean's registered reads ("does not open") at each of its target members (`targets`),
    every other member's proof held."""
    lean = {k.split("/", 1)[1]: v for k, v in res["verdicts"].items() if k.startswith("lean-verify/")}
    outcome = rounds_of(at)[kind_of(name)][1]
    if outcome == "accepted":
        return res["window_accepted"] and all(lean.values())
    if outcome == "ok 0":
        return (res["accepted"] and not res["window_accepted"] and bool(lean) and all(lean.values())
                and res["opened"] == unread_words(at))
    ts = targets(at, name)
    whys = [res["why"].get(f"lean-verify/{t}") or "" for t in ts]
    return (not res["accepted"] and all(lean.get(t) is False for t in ts)
            and all("registered reads" in why and outcome in why for why in whys)
            and all(v for x, v in lean.items() if x not in ts))


def blobs_of(members: list[dict]) -> tuple[dict[str, bytes], set[str]]:
    """The witness pieces the verifier must not receive: every unit's rows, inputs and outputs but a public o0, and
    every salt but a public ``o0``'s (the slot entries' o0, `ROWS_O0`, is a committed row, salts and all)."""
    blobs = {}
    for m in members:
        for i, u in enumerate(m["units"]):
            blobs.update({f"{m['name']} {r} {i}": _bytes(x) for (r, _b), x in zip(m["ins"], u)})
        blobs.update({f"{m['name']} {p} {i}": x for p, col in m["made"].items() for i, x in enumerate(col)})
        blobs.update({f"{m['name']} salt {p} {i}": s for p, ss in m["salts"].items()
                      if p != "o0" or m["template"] in ROWS_O0 for i, s in enumerate(ss)})
    return blobs, set(blobs)


def known(members: list[dict], at: SimpleNamespace) -> bytes:
    """What the scan leaves out: the circuits, zeros, the window's pads and ENDs (fixed by the ordinals), and each
    member's refs (`refs_of`)."""
    B = at.setup.cls.B
    cells = [IT.pad_cell(B, q) + IT.end_cell(q, B) for q in [0, *at.ordinals]]
    circuits = {m["pin"]: m["dir"] / "circuit.txt" for m in members}
    return (b"".join(f.read_bytes() for f in circuits.values()) + bytes(WR.SALT_BYTES) + b"".join(cells)
            + b"".join(refs_of(m) for m in members))


def refs_of(m: dict) -> bytes:
    """Member m's refs in its public file (`shared_bcs`' format): each instance's table row of each port, u32
    little-endian, instance-major, which the statement fixes before any witness (a unit's own row, or its boundary's
    slot entry in first-read order, `registered.table`). They are small integers, which a witness row's small fields
    can equal (S_0's ``now``, ``phase`` and an empty slot). Its header and b‖c stay scanned: its roots, paths and
    commitments are computed from the witness. Refused unless each port's refs are its table's rows in first-read
    order: each at most one past the largest before it, ending at its last row."""
    head, body = (m["dir"] / f"pub-{m['n']}.bin").read_bytes().split(b"\n", 1)
    h = json.loads(head)
    rows = [int(h["shared_rows"][p.name]) for p in m["ports"]]
    at, n = 128 * sum(rows), int(h["instances"])
    refs = body[at:at + 4 * n * len(rows)]
    top, ok = [-1] * len(rows), len(refs) == 4 * n * len(rows)
    for k in range(n * len(rows) if ok else 0):
        x, q = int.from_bytes(refs[4 * k:4 * k + 4], "little"), k % len(rows)
        ok = ok and x <= top[q] + 1
        top[q] = max(top[q], x)
    if not ok or n and top != [r - 1 for r in rows]:
        raise SystemExit(f"{m['name']}: its public file's refs aren't each port's table rows in first-read order")
    return refs


def sparse(pieces: dict[str, bytes], held: set[str]) -> tuple[dict[str, bytes], set[str], int]:
    out = {k for k, x in pieces.items() if sum(map(bool, x)) <= 1}
    return {k: x for k, x in pieces.items() if k not in out}, held - out, len(out)


def received(a8, vdir: Path, control: Path, blobs: dict[str, bytes], held: set[str], public: bytes) -> dict:
    """What the verifier received (`vdir`) scanned for the witness `blobs`, less the sparse pieces: each row of at least
    a piece cut whole less what `public` holds (`leak_scan.public_pieces`, found by `a8.received`), and each shorter
    row (a slot entry at a small KO) found whole (`leak_scan.find_whole`, as ``whole_rows``); the `control` (the
    instances) is to hold every one in `held`."""
    short = {k: x for k, x in blobs.items() if len(x) < a8.PIECE}
    pieces, kept, out = LS.public_pieces(a8, {k: x for k, x in blobs.items() if k not in short}, held, public,
                                         fields=True)
    pieces, kept, nsparse = sparse(pieces, kept)
    short, short_held, nshort = sparse(short, held & set(short))
    rec = {**a8.received(vdir, control, pieces, kept), "public_pieces": len(out), "sparse_pieces": nsparse + nshort}
    if short:
        rec["whole_rows"] = LS.find_whole(rec, short, vdir, control, short_held)
    return rec


def unseen(rec: dict) -> bool:
    """No witness piece nor short row in what the verifier received (`received`), and the control holds them all."""
    return all(not r["verifier_found"] and r["control"]["holds_all"] for r in (rec, rec.get("whole_rows")) if r)


def frame_leaf(earlier: dict, start: dict) -> dict:
    """A start's leaf on the staged files of two link-windows' honest rounds: ``earlier``, the one before's staged
    ``last`` member, whose statement commits its o1 as a frame port of one leaf, so that port's root is that leaf as
    Flock's statement frames it; ``start``, this one's staged start member, whose ``sin`` row's b‖c one-stage's
    registered read frames at leaf 0 of that port's domain (`registered.leaf`, the leaf Lean's registered reads
    open). The two leaves, and whether they and the two rows' b‖c are the same bytes."""
    fv, dom = earlier["header"]["frame_v3"], ADV.domain(frame_binding(earlier["set"], 1, "o1"), 1)
    read = REG.leaf(dom, 0, start["bcs"][0]["sin"], BitRowLeaf.hiding_schema).hex()
    return {"statement": fv["roots"]["o1"], "registered_read": read,
            "same": (read == fv["roots"]["o1"] and dom.domain_id.hex() == fv["domain_ids"]["o1"]
                     and start["bcs"][0]["sin"] == earlier["bcs"][0]["o1"])}


def prove(a) -> int:
    """``prove``: every round of the group's link-windows (`host`), in link-window order on one verifier's log: each
    member proved --zk and held by Lean, a start member's ``sin`` against the root and domain the verifier took from
    its log (`carry_of`). A link-window's rounds with a start unit (`STARTED`) are registered against the log as that
    link-window found it, and the log takes the honest round's record only once its audit accepted it with every
    opened output 1, so no start is read from a link-window the verifier refused: the link-window after one stops at
    `carry_of`. A round is as expected when its outcome is its kind's (`expected`) and what the verifier received
    holds no witness piece (`unseen`); with a start unit, when its record's changed forms are also refused as named;
    and an accepted honest round, when the verdicts of record also opened its start against that root and, after
    the first link-window, its start's leaf is the round before's frame leaf byte for byte (`frame_leaf`)."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    setup = SETUPS[a.setup]
    h = host(out / "warden", a.warden_run, a.draw, setup, a.group, v3c=a.v3c)
    whole, T = h.whole, h.T
    ch, cls, N = whole.chain, setup.cls, 2 * T + 1
    if a.draw is not None and not 3 <= a.draw <= N:
        raise SystemExit(f"--draw {a.draw}: a round's draw is 3 .. {N} of its {N} units (one a template at least)")
    ks = {"honest": a.draw, "chain": a.draw, "tampered": None if a.draw is None else stratum_k(a.draw, T, N)}
    sh = cls.egress
    last = next(r.at for r in reversed(h.rounds.values()) if r.kind == "honest")
    summary: dict = {"source": whole.source, "warden_run": whole.run_id,
                     "draw": ks if a.draw is not None else "every unit",
                     "setup": setup.name, "class": asdict(cls), "params": asdict(whole.params), "shape": asdict(sh),
                     "T": T, "group": whole.G, "windows": whole.windows,
                     "bits": {f"{tp}/{r}": b for tp, (ins, outs) in ports_of(cls).items() for r, b in ins + outs},
                     "members": {name: [{"name": m.name, "base": m.base, "n": m.n} for m in r.members]
                                 for name, r in h.rounds.items()},
                     "admitted": whole.admitted,
                     "anchors": {q: cls.anchor(s, s % cls.EVERY) for q, s in whole.admitted.items()},
                     "responses": {q: len(x) for q, x in sorted(whole.outputs.items())}, "run_s": whole.run_s,
                     "swap": None if whole.swap is None else [whole.swap, whole.swap + 1],
                     "unread": whole.unread and {"bucket": whole.unread.bucket, "slot": whole.unread.slot,
                                                 "q": whole.unread.q},
                     "tampered": whole.tampered and {"bucket": whole.tampered.bucket,
                                                     "window": whole.windows[whole.tampered.window],
                                                     "key": whole.tampered.key_of},
                     "void": whole.void and {"buckets": whole.void, "failed": whole.failed, "why": whole.failed_why,
                                             "refused": targets(last, round_name("honest", last.number))},
                     "cells": {side: {"data": sum(r[s * cls.B] == DATA for r in rows for s in range(sh.R)),
                                      "end": sum(r[s * cls.B] == W.END for r in rows for s in range(sh.R))}
                               for side, rows in (("ingress", ch.ingress), ("egress", ch.egress))},
                     "roots": {r.number: {k: t.root.hex() for k, t in r.at.trees.items()}
                               for r in h.rounds.values() if r.kind == "honest"},
                     "start": {name: {x: r.members[0].ports["sin"][x] for x in ("value", "root", "domain")}
                               for name, r in h.rounds.items() if r.kind in STARTED},
                     "v3c": whole.v3c and {"window": str(whole.v3c),
                                           "trees": sorted(k for k, t in last.trees.items() if isinstance(t, Held))},
                     "host": {name: [{"member": m.name, "refused": m.refused, "outputs": m.outputs}
                                     for m in r.members] for name, r in h.rounds.items()}}
    void = (f" but its FAIL buckets {whole.void} (void: link-window {last.number}'s honest round refused at "
            f"{summary['void']['refused']})" if whole.void else "")
    log(f"run: {whole.source}; {setup.name}'s class, link-windows {whole.windows} of {T} buckets, each a "
        f"registration, admitted {whole.admitted}, responses {summary['responses']}, anchors {summary['anchors']}, "
        f"cells {summary['cells']}; every egress bucket computes the warden's row from its ingress bucket's "
        f"view{void}; host reads as expected ({whole.run_s} s); draw {summary['draw']}")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    if a.dry:
        log("dry: done")
        return 0
    M = WZ._one_stage()
    pgs = {templates: program(M, cls, templates, T, out / PROGRAMS[templates])
           for templates in dict.fromkeys(r.templates for r in h.rounds.values())}
    summary["stagers"] = {f"{PROGRAMS[k]}/{tp}": {
        "unit": g.defn.id, "lower_s": g.lower_s, "k_log": g.st.meta["k_log"], "vus_per_block": g.st.meta["vus_per_block"],
        "pin": g.st.pin, "base": g.base, "n": g.n} for k, pg in pgs.items() for tp, g in pg.stagers.items()}
    log(f"stagers: {summary['stagers']}")
    av = SimpleNamespace(bin=a.bin, lean=a.lean, lease=[], gpu=False, delta=a.delta)
    by_window: dict[int, list[SimpleNamespace]] = {}
    for r in h.rounds.values():
        by_window.setdefault(r.number, []).append(r)
    stride = max(len(r.members) for r in h.rounds.values())
    ok, results, records, done, before, j = True, {}, [], [], None, 0
    for number, rs in by_window.items():
        honest = next((r for r in rs if r.kind == "honest"), None)
        found = list(records)
        with challenger(a, honest and honest.at, out / f"challenger-{number}") as coins:
            if coins is not None:
                summary.setdefault("coins", {"challenger": coins.challenger, "sessions": {}})["sessions"][
                    honest.name] = {"window_root": coins.root}
                log(f"{honest.name} draws on the challenger's coins ({coins.challenger}), its session on the "
                    f"link-window's root {coins.root[:16]}…; the other rounds on a development challenger's")
            for r in rs:
                name, at, pg = r.name, r.at, pgs[r.templates]
                members = [stage(M, pg, mr, out / name / f"stage-{mr.name}") for mr in r.members]
                if r.kind == "honest" and (wires := staged_wires(members)):
                    raise SystemExit(f"{name}'s chains don't hold on its public files: {wires}")
                summary.setdefault("rounds", {})[name] = {
                    m["name"]: {"ands": m["ands"], "k_log": m["k_log"], "nbl": m["nbl"], "m": m["k_log"] + m["nbl"],
                                "stage_s": m["stage_s"], "outputs": m["outputs"]} for m in members}
                log(f"staged {name}: " + "; ".join(f"{m['name']} {m['ands']} ANDs a unit, m = {m['k_log']} + "
                                                   f"{m['nbl']}, outputs {m['outputs']}" for m in members))
                (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
                carry = carry_of(pg, cls, found, r.members[0].ports["sin"]) if r.kind in STARTED else None
                res = audit(M, av, at, pg, name, members, r.trees, out / name, a.port and a.port + stride * j,
                            ks.get(r.kind), coins if r.kind == "honest" else None, carry)
                results[name], j = res, j + 1
                if r.kind == "honest":
                    if res["window_accepted"]:
                        records.append(json.loads((out / name / "registration.json").read_text()))
                    done.append(at)
                    summary["answers"] = answers(done, out, results, Fraction(a.delta))
                    log(f"answers: link-window {number}: {summary['answers'][number]}")
                inst = out / name / "instances.bin"
                inst.write_bytes(b"".join((m["dir"] / f"inst-{m['n']}.bin").read_bytes() for m in members))
                rec = res["received"] = received(M.a8, out / name / "verifier", inst, *blobs_of(members),
                                                 known(members, at))
                inst.unlink()
                good = expected(at, name, res) and unseen(rec)
                negatives = res.get("registration_negatives") or {}
                if carry is not None:
                    good = good and bool(negatives) and all(
                        x["expected"] in x["codes"] if x["expected"] else not x["codes"] for x in negatives.values())
                if r.kind == "honest" and r.outcome == "accepted":
                    leaf = res["frame_leaf"] = frame_leaf(before, members[0]) if before else None
                    good = good and res.get("carried_start") == {CARRY: carry.root} and (leaf is None or leaf["same"])
                    before = next(m for m in members if m["name"] == "last")
                if coins is not None and r.kind == "honest":
                    good = good and bool(res.get("coin_negatives")) and all(res["coin_negatives"].values())
                ok &= good
                summary["rounds"][name]["audit"] = res
                rows = rec.get("whole_rows") or {"rows": 0, "verifier_found": []}
                started = "" if carry is None else (
                    f"start {CARRY} at {carry.root[:16]}… ("
                    + ("pinned, the empty state's" if "pinned" in carry.check else f"{CARRY_PORT} of the record before")
                    + f"), changed forms { {x: y['codes'] for x, y in negatives.items()} }, the verdicts of record "
                    f"opened { {k: (v or '')[:16] for k, v in (res.get('carried_start') or {}).items()} }, frame "
                    f"leaf {res.get('frame_leaf')}; ")
                log(f"round {name}: {started}drawn {len(res.get('draw', {}).get('units') or [])} of {pg.units}, "
                    f"charge {res.get('charge')}, per unit {res.get('per_unit')}; proofs {res['verdicts']}, audit "
                    f"{res.get('audit')}, opened {res.get('opened', {})}, Lean {res['why']}, verifier received "
                    f"{rec['verifier_bytes']} bytes, witness pieces found there: {rec['verifier_found'] or 'none'} "
                    f"(of {rec['blobs']}; {rec['public_pieces']} public, {rec['sparse_pieces']} sparse left out), "
                    f"rows shorter than a piece found whole there: {rows['verifier_found'] or 'none'} (of "
                    f"{rows['rows']}), control {rec['control']['blobs_found']}; "
                    f"{'as expected' if good else 'UNEXPECTED'}")
                (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
        if coins is not None:
            pub = coins.published or {}
            session = {x: pub.get(x) for x in ("trusted", "error", "plan")}
            summary["coins"]["sessions"][honest.name]["session"] = session
            ok &= pub.get("error") is None and (pub.get("trusted") or {}).get("pinned_session") is None
            log(f"the challenger's session for link-window {number}: {session}")
    summary["ok"] = ok
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    log("done: " + ("ok" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="scan_zk.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prove", help="each link-window's start, ingress and egress proof blocks chained by registered "
                                     "reads, --zk")
    p.add_argument("--bin")
    p.add_argument("--lean")
    p.add_argument("--out", required=True)
    p.add_argument("--port", type=int, default=0, help="round i's member j listens on PORT + S·i + j, S the most "
                                                       "members a round has; 0 (the default): a free port for "
                                                       "each session")
    p.add_argument("--delta", type=float, default=2.0 ** -20)
    p.add_argument("--dry", action="store_true", help="the run and the host's read of every round: nothing proved")
    p.add_argument("--class", dest="setup", choices=sorted(SETUPS), default=W63.name,
                   help="the class and the network certifier's schedule (default w63's, PARAMS; w63k4: w63's at KO 4 "
                        "and KI 32, its certifier's at egress cells 5; test: the certifier tests', a CPU window; "
                        "test2: test's over two link-windows, a request anchored at the second's first boundary)")
    p.add_argument("--group", type=int, default=1,
                   help="G: link-windows 0 .. G - 1, each a registration of its own, its scan started from the "
                        "state the record before it committed (default 1)")
    p.add_argument("--warden-run", type=Path,
                   help="a warden_serve run's directory (rehearse --ids at the class's schedule): its group's "
                        "execution phases, verified under its deployment's key, in place of the synthetic run")
    p.add_argument("--draw", type=int,
                   help="K: Lean draw --stratified K on a development challenger's coins, only the SPOT proof "
                        "blocks it draws proved; default every unit")
    p.add_argument("--challenger", help="with --draw: the honest round drawn at the rate K/N (stratified-rate:K/N) on the "
                   "auditor's challenger's coins "
                   "(a5_coins, the round's service.Stream), not the verifier's: local, or gate:USER@HOST "
                   "(the auditor's challenger through its gate, with --peer and --challenger-key)")
    p.add_argument("--peer", help="with --challenger gate:USER@HOST: this machine as the gate's /etc/vy/peers names it, "
                   "whose 127.0.0.1 the session's challenger listens on")
    p.add_argument("--challenger-key", help="with --challenger gate:USER@HOST: the challenger's key record of its "
                   "deployment (challenger-key/v1), never the session's key.json")
    p.add_argument("--gate-key", help="with --challenger gate:USER@HOST: the lane's own gate key (default ssh's own)")
    p.add_argument("--main", help="with --challenger: the verifier's own copy of main's first-parent history (git "
                   "rev-list --first-parent origin/main), which must hold the commit the session's host pins")
    p.add_argument("--v3c", type=Path, metavar="WINDOW",
                   help="V3c's window directory (prove_path.py window-commit's): the scan reads its SCAN_STATE and "
                        "its SelectEntry_v1 chunks' o0 as its state and prog<j> trees, and proves no block round")
    s = sub.add_parser("schedule", help="a setup's schedule (w63's, PARAMS, by default) as warden_serve.py's schedule "
                                        "flags (serve, rehearse): the network certifier runs --warden-run reads, in "
                                        "the id egress form, serve's and rehearse --ids's")
    s.add_argument("--class", dest="setup", choices=sorted(SETUPS), default=W63.name)
    a = ap.parse_args(argv)
    if a.command == "schedule":
        params = SETUPS[a.setup].params
        print(" ".join(f"--{k.replace('_', '-')} {getattr(params, k)}" for k in W.INTEGER_FIELDS))
        return 0
    if a.group < 1:
        p.error(f"--group is 1 or more link-windows, not {a.group}")
    if a.challenger:
        if a.challenger != "local" and not a.challenger.startswith("gate:"):
            p.error(f"--challenger is local or gate:USER@HOST, not {a.challenger!r}")
        if a.draw is None:
            p.error("--challenger draws the honest round's --draw K on the challenger's coins: give --draw")
        if a.challenger.startswith("gate:") and not (a.peer and a.challenger_key):
            p.error("--challenger gate:USER@HOST needs --peer and --challenger-key")
        if not a.main:
            p.error("--challenger needs --main: the verifier's git rev-list --first-parent origin/main, which must hold "
                    "the commit the session's host pins")
    if not a.dry and not (a.bin and a.lean):
        raise SystemExit("prove needs --bin and --lean, or --dry")
    return prove(a)


if __name__ == "__main__":
    raise SystemExit(main())
