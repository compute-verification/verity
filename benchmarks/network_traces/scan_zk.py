"""scan-zk: the network certifier's step-indexed scan composed in zero knowledge, in the generator form. Ingress bucket t
(`ingress_scan.IngressScanBucket`; bucket 0's `IngressScanFirst`, from the empty state) reads the warden's ingress row
t, the scan's state S_{t−1} and its boundary's slot entries, and computes S_t and its egress view; egress bucket t
(`egress.EgressBucket`) reads that view and computes the warden's egress row t. A group of link-windows is one
registration, every row of every unit, input or output, a registered read, proved by C-Flock `prove --zk` behind the
Lean firewall and held by Lean `verify --zk --registered`. Standalone: in no Spec, except the honest round on the
challenger's coins (``--challenger``), a `service.Stream`'s.

    python benchmarks/network_traces/scan_zk.py prove --bin flock-circuit --lean flock-verify --out DIR [--dry]
        [--class w63|test] [--group G] [--warden-run RUN] [--draw K [--challenger local|gate:USER@HOST --main FILE
        [--peer NAME --challenger-key RECORD] [--gate-key KEY]]]
    python benchmarks/network_traces/scan_zk.py schedule [--class w63|test]    # its schedule as warden_serve.py's flags

* **The setups** (`SETUPS`, ``--class``): ``w63`` (`W63`, whose `CLASS`, `SHAPE` and `PARAMS` these are), w63's class
  (B 512, r 4, 4 lanes, C 47, IDB 17, EVERY 16, WITHIN 12: 4 Program slots a timestep, 8 a link-window) with the
  warden's delay D 27 (WITHIN + EVERY − 1), KO 64 outputs and KI 1,024 prompt ids a request,
  egress at `CLASS.egress` (16 cells in lanes of 4, 4 request slots of 64 17-bit ids), the warden at `PARAMS`, 32-bucket
  link-windows; and ``test`` (`TEST`), the certifier tests' class (B 512, r 2, 2 lanes, C 4, EVERY 2, WITHIN 3, D 5,
  KO 8, KI 100) and 16-bucket link-windows, whose units a CPU proves in minutes.
* **The run** (`scan_traffic.serve`: the warden's own `_ingress` and `_egress`, ticked as `Warden._tick` ticks them)
  carries the setup's warden-form id requests and their responses: w63's five (3, 40, 64, 8 and 5 ids; the fifth waits
  in the queue until request 1's egress ends at bucket 27 and takes its slot at 28), test's three (the third in the
  first's slot past its egress). Each link-window of each link is committed by the warden and its head signed by a
  test-size LMS key (`warden_rows.attest`); each registration is read back by the verifier
  (`warden_rows.check_registration`).
* **A warden's run** (``--warden-run RUN``, `warden_run`) in its place: a warden_serve run's directory (``serve``'s, or
  the gate's ``warden-rehearse --scan``'s with its witness under ``witness/``) at the setup's schedule (w63's:
  ``scan_zk.py schedule``) carrying the scan's traffic (``rehearse --ids``). The verifier reads ``deployment.json`` (the
  warden's public key, the session, the schedule and its digest) and each link's registration of each link-window of
  the group in ``cells.jsonl``, its head's signature verified under that key; the prover's rows are the records under
  ``cells.salts``, held to that head. Each is refused by name where it doesn't hold, as is a schedule the class doesn't
  serve and a response that isn't id-form.
* **The group** (``--group G``, default 1; `link`'s ``start``, its first link-window, 0): G link-windows of each link,
  one registration, the scan's buckets 0 … G·T − 1 from the empty state, T a link-window's.
* **The scan's rows**, committed by the prover before the draw as hiding rows (`verity.ml.commitments.advice`, each
  tree's root in the registration): ``state``, G·T leaves, leaf t S_{t−1} and leaf 0 the empty state, the rows a
  window circuit of the group reads as its `SCAN_STATE` (`ingress_scan.window_state` from bucket 0), so S_{G·T−1},
  which no bucket of the group reads, has no leaf; ``prog``, the slot entries (`slot_entries`),
  NR·nb leaves for the group's nb = G·T / EVERY boundaries, leaf j·nb + b slot j's entry at boundary b (the outputs of
  the Program anchored there, read and its response ids, as `SlotEntry_v1` writes them once it carries the read bit),
  the block round's o0 rows; and ``view``, G·T leaves, leaf t S_t's egress view.
* **The block round** (`BLOCK`): the slot entries' stand-in (`SlotEntry{KO, IDB}`, whose Definition id,
  ``SlotEntry_v1``, is the one `NetworkCheck.Period`'s `wired` reads: o0 its `entry_bits` input ``x`` through two
  NOTs, a committed row the verifier never opens; ``x`` a row-seg/v1 row, `ROW_SCHEMAS`), one unit a slot entry of the
  group, unit k's ``x`` slot entry k, its free input (the member's own rows, `Frame`), and its o0 a registered read
  of ``prog`` leaf k, standing in for
  the anchored Programs' `SlotEntry_v1` units, whose outputs the slot entries are. Its entries, read bit included, are
  the prover's: the stand-in copies its input. The ``prog`` tree is committed under the binding of the block's o0
  frame domain (`block_binding`, over the block's statement's input set `block_set`, bound to the session and the
  group's ingress heads), so its root is the o0 frame root C-Flock commits under the same salts, and `stage` refuses a
  block whose o0 frame root and domain aren't the tree's: the registration's ``SlotEntry_v1{…}/c0/o0`` (its ports
  named as one-stage names a template's first chunk's, `entries_port`) and the honest round's ``scan/prog`` are one
  root and domain, the tie `NetworkCheck.Period`'s rows form checks (`progTied`, `wired`). Its registration also names
  the scan's ``state`` tree as ``scan/state``, as an anchored Program's names the `SCAN_STATE` it reads, the tie the
  rows form checks next (`stateTied`); the stand-in reads no state, so here that tie is its registration's alone.
  Proved first; accepted, with no output opened.
* **The honest round, one registration**: one Program of three templates (`TEMPLATES`) in the generator form
  (`scan_root`, the form `NetworkCheck.Period`'s `scanOf` reads): the first unit (`IngressScanFirst`, bucket 0), G·T − 1
  ingress units (buckets 1 … G·T − 1) and G·T egress units, each template one batch with its row, state and egress
  parameters inputs of its own and every ingress unit's ``prog<j>`` a row of one input of the slot entries, its members
  split a link-window (`members_of`: ``first``, ``ingress-w`` and ``egress-w``, and ``last``, the group's last ingress
  unit, whose o1 has no leaf to open), every unit drawn by the verifier's Lean. `program` refuses a Program whose
  one-stage partition or reads (`registered.reads`) aren't the members'. Ingress unit t reads ``row`` (the warden's
  ingress row t), ``sin`` (state leaf t, S_{t−1}; the first unit none) and each slot j's ``prog<j>`` (``prog`` leaf
  j·nb + ⌊t / EVERY⌋, its boundary's entry of the slot, one table row a boundary, `NetworkCheck.Period`'s
  `readsEntries`), and outputs ``o0``, its ok bit, ``o1``, S_t, committed as state leaf t + 1, and ``o2``, its view,
  committed as view leaf t. Egress unit t reads ``state`` (view leaf t) and outputs ``o0``, its render bit, and
  ``o1``, the row it computes, committed as the warden's egress row t. Each output row but o0 is a registered read of
  the row it is committed as (`Registered.checkPort` holds a hidden-output statement's output rows, the verifier's
  PROTOCOL.md §16.13), so S_t is the one leaf unit t computes and unit t + 1 reads, the view the one leaf ingress t
  computes and egress t reads, and the egress row the one the warden signed: the chains are wires through the
  registered roots, not the prover's word. `Registered.checkPort` holds every instance of a registered port, so the
  last unit's o1, S_{G·T−1}, which no state leaf holds, is a member's own rows (`Frame`), committed and never opened.
  Leaf 0 is opened by
  no unit, and the entries boundary 0's Programs write on it are read by none: bucket 0 reads the empty state, in
  which no slot is live, so none is due. Accepted: every o0 opens to 1. The registration's roots also name, as
  ``rows/<direction>/<w>`` (`warden_roots`), the root and domain of each warden row registration a member's warden port
  reads (ingress's ``row``, egress's ``o1``), so a period check holds the answer to them.
* **Every port registered**: the verifier's registered reads name every port of every member's circuit, inputs and
  output rows (`Flock.Registered.checkComplete`, which `flock-verify` runs whenever ``--registered`` is given), so no
  output reads a row the statement commits afresh. A port no scan tree or warden row registration holds (each unit's
  public o0, the last unit's o1, the block's ``x``) is the member's own rows of it (`Frame`, value ``frame``),
  committed under its statement's frame binding (`frame_binding`, over the member's input set), so its root is the
  member's frame root of the port, which the round's registration names before any coin; `stage` refuses a member
  whose frame root isn't the tree's. A public o0 is opened as before, under the salt the prover sends.
* **The answers** (``answers.json``, `answers`): the honest round's registration and, for each link-window W of the
  group, the warden's two row registrations of W and the scan's one-stage answer (`window_answer`), the same for every
  W, which `verity/certifier/v0`'s rows form (`NetworkCheck.Period`) checks for W with W's three records'
  certificates: accepted, one-stage's profile and bound at ``--delta`` (every unit drawn, the bound 0); refused, a
  reject naming the members Lean refused. Beside them, ``audits``: the block round's, the compute audit whose o0 is the
  scan's ``prog`` (its registration and answer, its Program's descriptor, its free inputs NR·nb·NE, every bit of every
  slot entry, which the rows form's K charges to each link-window tied to it, and no steps).
* **A void run** (a warden run whose warden failed a connection closed, its `FAIL` in the egress row it signed, its
  reason in the ``fails`` record): the scan renders the connection's end where the warden wrote `FAIL`, and that bucket
  alone differs from the warden's row. Its block round and its honest round are proved, the honest one expected
  refused by Lean's registered reads ("does not open") at the egress members of its FAIL buckets (`rounds_of`,
  `targets`), every other member held, and its answer is that refusal, so the period check rejects the link-windows by
  name.
* **Negatives**, chain's and tampered's refused by Lean's registered reads ("does not open"), unread's by its unit's ok
  bit. ``chain``: the same Program on a state tree with S_A and S_{A+1} swapped (leaves A + 1 and A + 2): ingress unit
  A's o1, the S_A it computes, doesn't open leaf A + 1, nor units A + 1's and A + 2's theirs. ``tampered``: an
  egress-only Program of G·T units, one egress row's `DATA` id changed and its link-window re-signed (a warden that
  saw a byte the scan doesn't render; a warden run's
  re-signed by the driver's own key, under which the verifier reads it): egress unit t's o1, the row its view renders,
  doesn't open the signed row t. ``unread`` (`_unread`): a Program of one ingress unit, at the first bucket U ≥ 1 where
  a request's Program starts (its slot's `occ` 1), its ``prog`` tree the honest one but for that slot's entry at U's
  boundary, its read bit cleared (no Program reads the request): every registered read holds, since the read bit
  enters the unit's ok bit alone and its S_U and view are the honest ones, and Lean holds its proof, but its o0 opens
  to 0, so the link-window is refused. Every unit (the one) is drawn. A prover that sets the bit with n 0 and no
  outputs passes here, since the stand-in's entries are the prover's; an anchored Program's entry is its outputs on the
  `SCAN_STATE` it reads, which the period check holds to the scan's state tree (`stateTied`).
* **A sampled draw** (``--draw K``): Lean ``draw --stratified K`` on the verifier's own coins (its OS randomness, the
  rehearsal's live coins), each template's stratum its share and at least one unit; only the drawn units are proved
  and opened, and an accepted audit carries ε at δ (``wrong_units``: at most ``bound`` of ``of`` units wrong, except
  with probability δ), its charge terms (`charge_terms`) and each member's cost per drawn unit (`per_unit`). The chain
  and tampered rounds draw too and are refused all the same: Lean holds every table row of a registered port, drawn or
  not, an output port's table its instances' rows. A member none of whose units is drawn (G > 1) runs no session.
* **The challenger's coins** (``--challenger local|gate:USER@HOST --main FILE``, with ``--draw K``): the honest round's
  K drawn as prove_path's records audit draws (`a5_coins`): one session of the auditor's challenger on the group's root
  (`window_root`: the warden's session and the group's signed registrations), and the round's `service.Stream`
  (`spec_of`) on its coins (`a5_coins.RankCoins`): the stream registers the round, the coins commit to the
  registration's digest, and its select draws once (Lean's `Select.select` on the bytes read). Once the session has
  ended, the outcome of record is the stream's with the coins' evidence, the Lean's whole run (`Run.run`, coin source
  the challenger) held to the Python's, and each altered evidence (`a5_coins.negatives`, the evidence for another
  window) refused by `coins.coins_reason` and the Lean's run (`coins_outcome`). ``local`` is a development session on
  this machine (its own key; it shows the path, not an auditor's coins); ``gate:USER@HOST`` is the auditor's
  challenger's through its gate, its challenger on ``--peer``'s 127.0.0.1, under the deployment's key record
  ``--challenger-key``. The gate opens a root once, ever: a run's group is drawn once (the synthetic run's registrations
  carry fresh salts, so each synthetic run has a root of its own). The other rounds keep the verifier's Lean's own
  coins: one session serves one draw, and their refusal is Lean's registered reads, whatever the coins.
* **What the verifier receives** is scanned for every piece of every unit's rows, inputs and outputs, and every salt
  but a public ``o0``'s, which the prover sends so the verifier opens each unit's output (`a8.received`; the slot
  entries' o0 is a committed row, `ROWS_O0`, whose salts it never sends), less the pieces public
  anyway (the circuits, zeros, the window's pads and `END` cells) and the sparse pieces (at most one nonzero byte); the
  prover's instance file, which holds them all, is the control. The instance holds each row's u16 words followed by
  the next row's, so a row's zero-padded last chunk isn't one the control holds; its last 16 bytes, which cover the
  same tail, are (`leak_scan.public_pieces`' ``fields``).
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
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import leak_scan as LS  # noqa: E402
import warden_zk as WZ  # noqa: E402
from verity.ml.circuits import boolean as BL  # noqa: E402
from verity.ml.circuits.boolean import trace as TR  # noqa: E402
from verity.ml.circuits.codec import canonical_json  # noqa: E402
from verity.ml.circuits.defs import CompositeDefinition, bind  # noqa: E402
from verity.ml.circuits.program import Program  # noqa: E402
from verity.ml.circuits.refs import Coll, array_of, coll_concat  # noqa: E402
from verity.ml.circuits.types import Array, Tuple  # noqa: E402
from verity.ml.commitments import advice as ADV  # noqa: E402
from verity.ml.commitments.frame_v3 import BitRowLeaf  # noqa: E402
from verity.ml.commitments.identity import identity_digest_sha512  # noqa: E402
from verity.ml.commitments.rowleaf import (SCHEMA_HM96_SHA512_ROW_SEG, bit_row, row_leaf,  # noqa: E402
                                           sha512_row_seg_digest)
from verity.ml.crypto import lms  # noqa: E402
from verity.core.protocols.certifier import egress as EG, ingress_scan as IS, io_tie as IT, warden as W  # noqa: E402
from verity.core.protocols.certifier import warden_rows as WR  # noqa: E402
from verity.core.protocols.certifier.active import DATA, id_request  # noqa: E402
from verity.core.protocols.certifier.certificate import Direction, Issuer, LmsSigner  # noqa: E402
from verity.core.protocols.sampled_proofs import coins as CC, service as S  # noqa: E402
from verity.core.protocols.sampled_proofs.consumers import exfiltration_bound, location_bits  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import audit as AU, draw as DR, partition as PA, produced  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import registered as REG, registration as RG  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage.draw import stratum_k  # noqa: E402

_TRAFFIC = HERE.parents[1] / "verity/core/protocols/certifier/tests/scan_traffic.py"
_spec = importlib.util.spec_from_file_location("scan_traffic", _TRAFFIC)
traffic = sys.modules["scan_traffic"] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(traffic)


class Setup(NamedTuple):
    """A class the scan is proved at, the warden's schedule it serves and the synthetic run's traffic: request i's
    prompt ``prompts[i]`` (max_tokens ``max_tokens`` + i, seed ``seed``·i) arrives at bucket ``arrivals[i]`` and is
    answered ``responses[i]``, and the warden serves ``windows`` link-windows, every connection ended by the last's end;
    the chain round swaps S_A and S_{A+1}."""

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
TEST = Setup(
    "test", IS.Class(B=512, r=2, lanes=2, C=4, IDB=17, EVERY=2, WITHIN=3, D=5, KO=8, KI=100),
    W.WardenParams(cell_bytes=512, slots=2, bucket_ms=100, delay=5, lanes=2, queue=2, window=16, ingress_cells=4,
                   ingress_buckets=3, egress_cells=9, every=2),
    ((1000, 1007, 1014), tuple(2000 + 3 * k for k in range(60)), tuple(131071 - k for k in range(100))),
    (0, 1, 1), ((11, 22, 33), (131071, 0, 5, 6, 7, 8, 9, 10), (40, 41, 42, 43)), windows=1, A=6, max_tokens=5, seed=11)
SETUPS = {s.name: s for s in (W63, TEST)}
#: w63's: the schedule ``scan_zk.py schedule`` prints and the gate's ``warden-rehearse --scan`` serves
CLASS, PARAMS = W63.cls, W63.params
SHAPE = CLASS.egress
SESSION = bytes(range(32))
#: a round's templates in its Program's order
TEMPLATES = ("first", "ingress", "egress")
#: the unread round's: one ingress unit
UNREAD = ("ingress",)
#: the block round's: the slot entries' stand-in (`SlotEntry`), one unit a slot entry, whose o0 rows are the scan's
#: slot entries
BLOCK = ("block",)
#: the templates whose o0 is a committed row the verifier never opens (the slot entries'), every other's its public word
ROWS_O0 = frozenset(BLOCK)
#: an input row's schema where it isn't hm96-sha512/row/v2: the block's ``x`` is a row-seg/v1 row, which takes a
#: compression before its last block at any length, as C-Flock's statement needs of one row (`verity_flock.circuit.
#: compose`); at the test class every block row is shorter than a block (NE = 176 bits)
ROW_SCHEMAS = {("block", "x"): SCHEMA_HM96_SHA512_ROW_SEG}
#: each scan template's parameters before its entry ports (`scan_root`'s inputs of its own): row (and sin), or state
FIXED = {"first": 1, "ingress": 2, "egress": 1}
#: each round's templates, and its outcome: accepted, refused by Lean's registered reads, or proved with every read
#: held and its unit's ok bit 0; the block first, since the scan reads its outputs
ROUNDS = {"block": (BLOCK, "accepted"), "honest": (TEMPLATES, "accepted"), "chain": (TEMPLATES, "does not open"),
          "tampered": (("egress",), "does not open"), "unread": (UNREAD, "ok 0")}
#: each round Program's directory, by its templates
PROGRAMS = {TEMPLATES: "joint", ("egress",): "solo", UNREAD: "unread", BLOCK: "block"}
#: C-Flock's tag of a frame domain's binding (`verity_flock.circuit.BINDING_TAG`), under which the block's o0 rows are
#: committed and so the scan's `prog` tree (`block_binding`), and a member's own rows of a port (`Frame`; `stage`
#: refuses a member whose frame root of such a port isn't its tree's)
FRAME_BINDING = "verity/flock-circuit/binding/v2"
#: the scan's registered values: its three trees, and ``frame``, a member's own rows of a port no tree or warden row
#: registration holds (`Frame`), registered as the member's frame root of the port
VALUES = {k: f"verity/certifier/scan-{k}/v1" for k in ("state", "prog", "view", "frame")}
TIMING = ("e2e_s", "wait_s", "witness_s", "prove_total_s", "proof_bytes", "wire_sent", "wire_recv", "k_log", "nbl",
          "dense_m", "units", "ands_per_vu", "prover", "gpu")
#: the honest round's stream on the auditor's challenger's coins (``--challenger``): its challenger, the format of the
#: group's root its session opens on (`window_root`), and each member's share of the proof deadline
CHALLENGER = "scan_zk.py on the auditor's challenger's coins"
WINDOW_ROOT = "verity/certifier/scan-window-root/v2"
MEMBER_DEADLINE_S = 5 * 3600
#: the format of ``answers.json`` (`answers`): the honest round's registration, the block's audit and each
#: link-window's answer and rows
ANSWERS = "verity/certifier/scan-answers/v2"


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
    return {"first": ((row, *progs), outs), "ingress": ((row, ("sin", IS.state_bits(cls)), *progs), outs),
            "egress": ((("state", IS.view_bits(cls)),), (("o1", 8 * cls.egress.R * cls.B),)),
            "block": ((("x", NE),), (("o0", NE),))}


def entries_of(cls: IS.Class, G: int, T: int) -> int:
    """The slot entries of a group of G link-windows of T buckets: NR a boundary, G·T / EVERY boundaries."""
    return cls.NR * (G * T // cls.EVERY)


def entry_at(cls: IS.Class, nb: int, j: int, t: int) -> int:
    """The slot entry bucket t's ``prog<j>`` reads (`NetworkCheck.Period`'s `readsEntries`): slot j's at t's boundary,
    row j·nb + ⌊t / EVERY⌋ of a group's nb boundaries."""
    return j * nb + t // cls.EVERY


def slot_entries(cls: IS.Class, progs: list[list[int]]) -> list[list[int]]:
    """The slot entries of the scan's Programs' rows `progs` (one a bucket, `scan_traffic.chain`'s), the ``prog`` tree's
    leaves: leaf `entry_at` (j, b·EVERY) slot j's entry of the Programs' row at boundary b (`ingress_scan.prog_rows`),
    the Programs anchored there. A bucket between boundaries has no due slot, so its row reads none of its entries."""
    nb = len(progs) // cls.EVERY
    return [IS.prog_rows(cls, progs[b * cls.EVERY])[j] for j in range(cls.NR) for b in range(nb)]


def members_of(templates: tuple[str, ...], G: int, T: int, bucket: int = 1, entries: int = 0) -> list[SimpleNamespace]:
    """A round's members in its Program's order, one a template and link-window w of the group: its name, its first
    unit (``base``) and the scan's buckets its units compute, unit ``base + k`` bucket ``buckets[k]``. The first unit
    computes bucket 0, the ingress template buckets 1 … G·T − 1 (with no first unit, 0 …), bucket G·T − 1 a member of
    its own (``last``: its S_t has no state leaf, and a registered port holds every instance's row), the egress
    template 0 … G·T − 1; the unread round's one unit (`UNREAD`) bucket ``bucket``; the block round's one member
    (`BLOCK`) the group's ``entries`` slot entries (`entries_of`), unit k slot entry k, since the `prog` tree its o0
    rows are is one tree over the group."""
    if templates == UNREAD:
        return [SimpleNamespace(template="ingress", window=bucket // T, name=f"ingress-{bucket // T}", base=0,
                                buckets=range(bucket, bucket + 1))]
    if templates == BLOCK:
        if entries < 1:
            raise ValueError("the block round's member is the group's slot entries: give entries (entries_of)")
        return [SimpleNamespace(template="block", window=0, name="block", base=0, buckets=range(entries))]
    out, at = [], 0
    for tp in templates:
        lo = 1 if tp == "ingress" and "first" in templates else 0
        for w in range(1 if tp == "first" else G):
            bs = range(1) if tp == "first" else range(max(lo, w * T), (w + 1) * T)
            parts = ((bs[:-1], f"{tp}-{w}"), (bs[-1:], "last")) if tp == "ingress" and w == G - 1 else \
                ((bs, "first" if tp == "first" else f"{tp}-{w}"),)
            out += [SimpleNamespace(template=tp, window=w, name=name, base=at + part.start - lo, buckets=part)
                    for part, name in parts if part]
        at += counts_of(out)[tp]
    return out


def counts_of(members: list[SimpleNamespace]) -> dict[str, int]:
    """Each template's units over its members."""
    out: dict[str, int] = {}
    for m in members:
        out[m.template] = out.get(m.template, 0) + len(m.buckets)
    return out


# --- the scan's rows: hiding rows the prover registers -----------------------------------------------------------------

def _bound(kind: str, heads: list, session: bytes) -> bytes:
    return hashlib.sha512(b"\0".join([VALUES[kind].encode(), session,
                                      *(x for h in heads for x in (h.value.encode(), h.root))])).digest()


def block_set(heads: list, session: bytes) -> str:
    """The block's statement's input set, which its frame domains bind: bound to the session and the group's ingress
    link-windows' heads."""
    return f"scan-block-{_bound('prog', heads, session).hex()}"


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
    """A member's own rows of one port, committed as C-Flock's statement commits the port (`verity_flock.circuit.
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


def frame(set_name: str, port: str, rows: list, n_bits: int, schema: str = BitRowLeaf.hiding_schema) -> Frame:
    """A member's own rows of ``port``, each unit's (bits or bytes), as `Frame` commits them, under fresh salts."""
    return Frame(frame_binding(set_name, len(rows), port, schema), n_bits, tuple(_bytes(r) for r in rows),
                 tuple(ADV.INST.fresh_salt() for _ in rows), schema)


def _schema(reg: ADV.Committed) -> str:
    """A registration's leaf schema: a member's own rows' (`Frame`) their port's, the scan's trees' row/v2."""
    return getattr(reg, "schema", BitRowLeaf.hiding_schema)


def tree(kind: str, rows: list[list[int]], heads: list, session: bytes) -> ADV.Committed:
    """`kind`'s rows (bit rows of whole u16 words) committed as hiding rows, bound to the session and the group's
    ingress link-windows' heads: `prog`'s under the block's o0 frame binding (`block_binding`), so the tree is the
    block's o0, the others' by their value (`VALUES`)."""
    n = len(rows)
    binding = block_binding(block_set(heads, session), n) if kind == "prog" else _bound(kind, heads, session)
    return ADV.commit([bit_row((_words(r), 16))[0] for r in rows], len(rows[0]), binding)


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
    e = {"leaves": n, "words": reg.n_bits // 16, "word_bits": 16, "domain": ADV.domain(reg.binding, n).domain_id.hex()}
    port = produced.port(VALUES[kind], e, reg.root.hex(), reads, _schema(reg))
    return port | ({REG.READS_FROM: int(reads_from)} if reads_from else {})


def tree_entry(kind: str, reg: ADV.Committed, positions) -> dict:
    return {"value": VALUES[kind], "positions": [int(p) for p in positions],
            "paths": [b"".join(reg.open(p).path).hex() for p in positions]}


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
    d, scheme = ADV.domain(reg.binding, n), ADV._scheme(reg.binding, n, reg.n_bits)
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
    """The synthetic run (`scan_traffic.serve`), each link-window of `windows` of each link committed by the warden and
    its head signed by the driver's test-size LMS key (`warden_rows.attest`), each registration read back by the
    verifier."""
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
    """A warden run's file: in the run's directory (`warden_serve.py serve --out`), else its ``witness/`` (the gate's
    `warden-rehearse --scan`)."""
    for d in (run, run / "witness"):
        if (d / name).is_file():
            return d / name
    raise SystemExit(f"--warden-run {run}: no {name} there or in its witness/ (warden_serve.py serve writes it)")


def warden_run(run: Path, setup: Setup, windows: list[int], out: Path) -> SimpleNamespace:
    """A warden_serve run's link-windows `windows` as the verifier reads them, refused by name where they don't hold:
    the deployment's schedule (its digest, and the class serving it) and public key; each link's registration of each
    link-window in ``cells.jsonl`` verified under that key and session (`warden_rows.check_registration`: the warden's
    signature over the head of the rows' root); and the records' rows under ``cells.salts`` (`warden_rows.Cells.of`)
    that head. The units read those rows. The Programs' outputs (the slot entries' ids) are each connection's ids
    as its egress record carries them (`io_tie.parse_cell`), every egress cell an id-form one or a `FAIL` (a void run:
    ``failed`` the reasons its ``fails`` record has in the link-windows, ``fail_buckets`` the buckets they're in, and
    ``failed_why`` the run's ``check.json`` problems that name a connection failed closed). The tampered round's
    re-signed link-window is signed by a key of the driver's own (``out``), under which the verifier reads it."""
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
    """The tampered round's egress link-window: the group's first egress row with a `DATA` cell, the last digit of
    that cell's id changed, its link-window's rows committed afresh and their head signed by ``got.issuer``, the
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


def _unread(cls: IS.Class, ch, heads: list, session: bytes) -> SimpleNamespace | None:
    """The unread round's unit: the first ingress bucket b ≥ 1 at which a slot's Program starts (a request's anchor,
    the slot due and its `occ` 1), the first such slot j, and the slot entries committed with slot j's at b's boundary
    (leaf ``entry``) absent, its read bit cleared and its outputs kept; None when no request is anchored in the group
    past bucket 0."""
    for b in range(1, len(ch.progs)):
        reads = [d and s.occ & 1 for d, s in zip(IS.due(cls, ch.scan[b]), ch.scan[b].slots)]
        if any(reads):
            j = reads.index(True)
            entries = IS.prog_decode(cls, ch.progs[b])
            entries[j] = (0, *entries[j][1:])
            rows, k = slot_entries(cls, ch.progs), entry_at(cls, len(ch.progs) // cls.EVERY, j, b)
            rows[k] = IS.prog_rows(cls, IS.prog_encode(cls, entries))[j]
            return SimpleNamespace(bucket=b, slot=j, entry=k, q=ch.scan[b].slots[j].q,
                                   tree=tree("prog", rows, heads, session))
    return None


def link(out: Path, run: Path | None = None, setup: Setup = W63, group: int = 1, start: int = 0) -> SimpleNamespace:
    """The run (`synthetic`, or `warden_run`'s of a warden_serve run), link-windows start … start + group − 1 of each
    link registered by the warden and read back by the verifier, the scan's chain over their rows from the empty state
    (`scan_traffic.chain`), the prover's three registrations (the state tree the group's `ingress_scan.window_state`),
    the chain round's state tree, the unread round's unit and slot entries' tree (`_unread`) and the tampered round's
    egress link-window (`_tamper`)."""
    t0 = time.time()
    if group < 1 or start < 0:
        raise SystemExit(f"a group is 1 or more link-windows from link-window 0 on, not {group} from {start}")
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
    if not 1 <= setup.A < setup.A + 2 < group * T:
        raise SystemExit(f"the chain round swaps S_A and S_(A+1), state leaves A + 1 and A + 2, 1 <= A < "
                         f"{group * T - 2}")
    heads = [w.head for w in got.sides["ingress"]]
    states = IS.window_state(cls, ch.states[1:], 0, group * T)
    trees = {k: tree(k, rows, heads, got.session) for k, rows in (("state", states),
                                                                 ("prog", slot_entries(cls, ch.progs)),
                                                                 ("view", ch.views))}
    swapped = list(states)
    swapped[setup.A + 1], swapped[setup.A + 2] = swapped[setup.A + 2], swapped[setup.A + 1]
    admitted: dict[int, int] = {}
    for st in ch.scan:
        for s in st.slots:
            if s.live:
                admitted.setdefault(s.q, s.admitted)
    return SimpleNamespace(setup=setup, chain=ch, T=T, G=group, windows=windows, params=got.params,
                           session=got.session, source=got.source, run_id=got.run_id, outputs=got.outputs,
                           sides=got.sides, trees=trees, swapped=tree("state", swapped, heads, got.session),
                           block_set=block_set(heads, got.session),
                           unread=_unread(cls, ch, heads, got.session),
                           tampered=None if void else _tamper(got, ch.egress, T, cls.B, cls.egress.R),
                           ordinals=sorted(admitted), admitted=dict(sorted(admitted.items())), void=void,
                           failed=got.failed, failed_why=got.failed_why, run_s=round(time.time() - t0, 1))


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
    verdict's `registered` reports its roots): each warden port at the head `warden_rows.check_registration` read
    under the deployment's key, never a root of the prover's."""
    return WR.reads_file(mr.ports)


def member(at: SimpleNamespace, name: str, mb: SimpleNamespace) -> SimpleNamespace:
    """Round `name`'s member `mb` (`members_of`): each unit's input rows (in its signature's order, as bits or bytes),
    the salts its rows, inputs and outputs, are committed under, its header's `registered` entries and the verifier's
    own ports, one for every port of its circuit (`Flock.Registered.checkComplete`), the host's read of them
    (`warden_rows.refusal`, `tree_refusal`: the ports in its signature's order, inputs, o0, then the other outputs),
    each unit's o0 word and its output rows as the references compute them (``made``, None where the ingress reference
    gives none), and the warden's row registration each warden port reads (``warden``). An ingress unit's ``prog<j>``
    is one table row a slot entry it reads (``refs``, `registered.table`), since a timestep's units read their
    boundary's; its ``sin`` reads state leaf t and its o1 opens leaf t + 1, the group's last unit's none. The block's
    unit k reads ``x``, slot entry k, its free input, and its o0, that entry, is a registered read of `prog`'s leaf k,
    no public word (``outputs`` None). A port no tree or warden row registration holds (each unit's public o0, the last
    unit's o1, the block's x) is the member's own rows of it (`Frame`, value ``frame``), registered as its frame root
    in its statement's input set ``set_name`` (the block's `block_set`), which the round's registration names before
    any coin; ``framed`` names the ports `stage` holds to their frame roots (those, and the block's o0)."""
    ch, T, cls = at.chain, at.T, at.setup.cls
    tp, w, bs = mb.template, mb.window, list(mb.buckets)
    rows = [b - w * T for b in bs]
    state, view = (at.swapped if name == "chain" else at.trees["state"]), at.trees["view"]
    prog = at.unread.tree if name == "unread" else at.trees["prog"]
    nb = len(ch.progs) // cls.EVERY
    ins, outs = ports_of(cls)[tp]
    if tp == "block":
        reads, warden = {"o0": ("prog", prog, bs)}, {}
        outputs, made = None, {"o0": [_bits(prog, b) for b in bs]}
    elif tp == "egress":
        side = at.tampered if name == "tampered" and at.tampered.window == w else at.sides["egress"][w]
        reads = {"state": ("view", view, bs)}
        warden = {"o1": side}
        if any(_bits(view, b) != EG.state_bits(cls.egress, ch.view(b)) for b in bs):
            raise SystemExit("a view leaf isn't its bucket's egress view")
        got = [EG.outputs(cls.egress, ch.view(b)) for b in bs]
        outputs, made = [o for o, _ in got], {"o1": [r for _, r in got]}
    else:
        lanes = {f"prog{j}": [entry_at(cls, nb, j, b) for b in bs] for j in range(cls.NR)}
        o1 = [b + 1 for b in bs]
        if o1[0] < len(state.rows) <= o1[-1]:
            raise SystemExit(f"{mb.name}: bucket {len(state.rows) - 1}'s S_t has no state leaf, and its other units' "
                             "have: a registered port holds every instance's row, so it is a member of its own "
                             "(members_of's last)")
        reads = {**({"sin": ("state", state, bs)} if tp == "ingress" else {}),
                 **{p: ("prog", prog, ps) for p, ps in lanes.items()},
                 **({"o1": ("state", state, o1)} if o1[-1] < len(state.rows) else {}), "o2": ("view", view, bs)}
        warden = {"row": at.sides["ingress"][w]}
        got = [IS.reference(cls, ch.ingress[b], _bits(state, b) if tp == "ingress" else None,
                            [x for ps in lanes.values() for x in _bits(prog, ps[k])]) for k, b in enumerate(bs)]
        outputs = [o for o, _s, _v in got]
        if name == "unread":  # the read bit enters ok alone: the unit computes the honest S_t and view
            got = [(o, *ch.outputs(b)) for o, b in zip(outputs, bs)]
        made = {"o1": [s for _o, s, _v in got], "o2": [v for _o, _s, v in got]}
    units = [tuple(ch.ingress[b] if r == "row" else made["o0"][k] if r == "x" else _bits(reads[r][1], reads[r][2][k])
                   for r, _b in ins) for k, b in enumerate(bs)]
    set_name = at.block_set if tp == "block" else f"unit-scan-{tp}-{name}-{mb.name}"
    sig = (*ins, *(() if tp in ROWS_O0 else (("o0", 16),)), *outs)
    own = {**{r: [u[i] for u in units] for i, (r, _b) in enumerate(ins)}, **made,
           **({} if outputs is None else {"o0": [[o] for o in outputs]})}
    for p, n_bits in sig:
        if p not in reads and p not in warden:
            if any(x is None for x in own[p]):
                raise SystemExit(f"{name}/{mb.name}: a unit's {p} row is none the reference gives, and a port no tree "
                                 "or warden row holds is registered as the member's own rows before any coin")
            reads[p] = ("frame", frame(set_name, p, own[p], n_bits, ROW_SCHEMAS.get((tp, p), BitRowLeaf.hiding_schema)),
                        list(range(len(bs))))
    salts, entries, ports, bcs, refs = {}, {}, {}, {}, {}
    for p, side in warden.items():
        _w, salts[p], entries[p] = side.cells.statement(rows)
        ports[p] = {**WR.bucket_port(side.head, at.params, mb.base), "reads": rows}
        bcs[p] = [WR.bc(made[p][k] if p in made else side.cells.rows[r], side.cells.salts[r]) for k, r in enumerate(rows)]
    for p, (kind, reg, ps) in reads.items():
        table, refs[p] = REG.table(ps) if p.startswith("prog") else (ps, list(range(len(ps))))
        salts[p], entries[p] = [reg.salts[i] for i in table], tree_entry(kind, reg, table)
        ports[p] = tree_port(kind, reg, ps, mb.base)
        bcs[p] = [None if p in made and made[p][k] is None else tree_bc(reg, i, made[p][k] if p in made else None)
                  for k, i in enumerate(ps)]
    refused = header_refusal(ports, entries)
    for p, _b in sig:
        refused = refused or (WR.refusal(ports[p], warden[p].dom, entries[p], bcs[p]) if p in warden else
                              tree_refusal(ports[p], reads[p][1], entries[p], bcs[p], refs[p]))
    changed = at.tampered.bucket if name == "tampered" and tp == "egress" and at.tampered.window == w else None
    return SimpleNamespace(round=name, template=tp, window=w, name=mb.name, base=mb.base, n=len(bs), buckets=bs,
                           units=units, made=made, salts=salts, entries=entries, ports=ports, bcs=bcs, refused=refused,
                           refs={p: r for p, r in refs.items() if r != list(range(len(bs)))}, outputs=outputs,
                           changed=changed, warden={p: s.registration for p, s in warden.items()}, set_name=set_name,
                           framed=[p for p, (kind, *_r) in reads.items()
                                   if kind == "frame" or (tp, p) == ("block", "o0")])


def rounds_of(at: SimpleNamespace) -> dict[str, tuple[tuple[str, ...], str]]:
    """The rounds a run is proved in, each with its templates and outcome: `ROUNDS`, the unread round where a request
    is anchored in the group past bucket 0; a void run's (the warden failed a connection closed, a `FAIL` in the
    egress row it signed) the block and its honest round alone, refused by Lean's registered reads ("does not open")
    at the egress members of its FAIL cells, where the scan renders the connection's end."""
    if at.void:
        return {"block": ROUNDS["block"], "honest": (TEMPLATES, "does not open")}
    return {k: v for k, v in ROUNDS.items() if k != "unread" or at.unread is not None}


def targets(at: SimpleNamespace, name: str) -> list[str]:
    """The members a refused round's refusal names: chain's, those of ingress units A, A + 1 and A + 2 whose o1 opens a
    state leaf (the group's last unit's opens none); tampered's, the egress member of the re-signed link-window;
    unread's, its one ingress unit's; a void run's honest round, the egress member of each FAIL's link-window."""
    if name == "chain":
        A, n = at.setup.A, at.G * at.T
        return sorted({mb.name for mb in members_of(TEMPLATES, at.G, at.T) if mb.template != "egress"
                       for b in mb.buckets if A <= b <= A + 2 and b + 1 < n})
    if name == "tampered":
        return [f"egress-{at.tampered.window}"]
    if name == "unread":
        return [f"ingress-{at.unread.bucket // at.T}"]
    return sorted({f"egress-{t // at.T}" for t in at.void})


def predicted(at: SimpleNamespace, name: str, members: list[SimpleNamespace]) -> bool:
    """The host's read of a round before any proof, as the round's: accepted, every read held and every output 1 (the
    slot entries' none: their o0 is no public word); unread's, every read held and its one unit's output 0; refused,
    its target members' reads refused ("does not open") and every other member's held."""
    outcome = rounds_of(at)[name][1]
    if outcome == "accepted":
        return all(m.refused is None and (m.outputs is None or set(m.outputs) == {1}) for m in members)
    if outcome == "ok 0":
        return [(m.name, m.refused, m.outputs) for m in members] == [(targets(at, name)[0], None, [0])]
    t = targets(at, name)
    return all((outcome in (m.refused or "")) if m.name in t else m.refused is None for m in members)


def wired(members) -> list[str]:
    """Where a round's chains don't hold, on its members' b‖c (`member`'s ``bcs``, port to each unit's): bucket b's
    o1 is bucket b + 1's ``sin`` and its o2 egress bucket b's ``state``, the one leaf of the registered roots both
    commit to (the group's last bucket's o1 none). Empty when they hold."""
    o1, o2, sin, state = {}, {}, {}, {}
    for m in members:
        for k, b in enumerate(m.buckets):
            if m.template == "egress":
                state[b] = m.bcs["state"][k]
            else:
                o2[b] = m.bcs["o2"][k]
                for p, at in (("o1", o1), ("sin", sin)):
                    if p in m.bcs:
                        at[b] = m.bcs[p][k]
    return [f"S_{b - 1}" for b in sorted(sin) if o1.get(b - 1) != sin[b]] + \
        [f"view {b}" for b in sorted(state) if o2 and o2.get(b) != state[b]]


def host(out: Path, run: Path | None = None, draw: int | None = None, setup: Setup = W63, group: int = 1,
         start: int = 0) -> SimpleNamespace:
    """The run (`link`'s, of `run` when given) and every round's members as the host reads them (no lowering, no
    proof): what the test holds. A sampled draw (`draw`) changes no round: each is refused by Lean's registered reads,
    which hold every table row, drawn or not."""
    at = link(out, run, setup, group, start)
    got = {}
    for name, (templates, outcome) in rounds_of(at).items():
        got[name] = [member(at, name, mb) for mb in members_of(templates, group, at.T,
                                                               at.unread.bucket if at.unread else 1,
                                                               len(at.trees["prog"].rows))]
        if not predicted(at, name, got[name]) or (name == "honest" and wired(got[name])):
            raise SystemExit(f"{name}: the host's read of the round isn't {outcome}: "
                             f"{[(m.name, m.refused, m.outputs) for m in got[name]]}, chains {wired(got[name])}")
    return SimpleNamespace(at=at, rounds=got, draw=draw)


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
    return {"first": bind(IS.IngressScanFirst, **st), "ingress": bind(IS.IngressScanBucket, **st),
            "egress": bind(EG.EgressBucket, B=sh.B, R=sh.R, r=sh.r, NR=sh.NR, K=sh.K, IB=sh.IB, D=sh.D),
            "block": bind(SlotEntry, KO=cls.KO, IDB=cls.IDB)}


def scan_root(members: list[tuple[object, int, int]], nb: int, every: int) -> Program:
    """The scan's root in the generator form (`NetworkCheck.Period`'s `scanOf`): each `(template, n, fixed)` of
    `members` one batch over its n units, in order, its first `fixed` parameters inputs of its own (``<i>.<name>``),
    and each later one, lane j's ``prog<j>``, at bucket t (the first template's first unit bucket 0, each next unit the
    next bucket) row j·nb + ⌊t / every⌋ (`entry_at`) of one input, ``prog``, the slot entries of nb boundaries."""
    sigs = [t.signature() for t, _, _ in members]
    lanes = [len(ps) - fixed for (ps, _), (_, _, fixed) in zip(sigs, members)]
    entry = next(ps[fixed][1] for (ps, _), (_, _, fixed), L in zip(sigs, members, lanes) if L)
    rows = max(lanes) * nb

    def signature(S):
        params = tuple((f"{i}.{name}", Array(n, ty)) for i, ((ps, _), (_, n, fixed)) in enumerate(zip(sigs, members))
                       for name, ty in ps[:fixed])
        return (*params, ("prog", Array(rows, entry))), Tuple(*(Array(n, ret) for (_, ret), (_, n, _) in
                                                               zip(sigs, members)))

    def body(B, S, *args):
        entries, outs, at, start = args[-1], [], 0, 0
        for (t, n, fixed), (ps, _) in zip(members, sigs):
            gathered = [array_of([entries[j * nb + (start + m) // every] for m in range(n)])
                        for j in range(len(ps) - fixed)]
            outs.append(B.batch(t, *args[at:at + fixed], *gathered, axes=(0,) * len(ps)))
            at, start = at + fixed, start + n
        return coll_concat(signature(S)[1], outs)

    defn = CompositeDefinition("ScanZkRoot", 1, ("templates", "counts", "nb", "every"), signature, body,
                               register=False)
    return Program(bind(defn, templates=tuple(PA.template_id(t) for t, _, _ in members),
                        counts=tuple(n for _, n, _ in members), nb=nb, every=every))


def entry_reads(prog: Program, query: dict, members: list[SimpleNamespace], cls: IS.Class, nb: int) -> dict:
    """One-stage's reads of the slot entries off the round's Program (`registered.reads`, ``prog`` a registered value
    of `entry_bits` bit rows), each unit's ``prog<j>`` row: SystemExit unless they are the members' (`entry_at`), so the
    verifier's own Program reads at one root what the statements open."""
    try:
        got = {p: r["rows"] for p, r in REG.reads(prog, query, {"prog": IS.entry_bits(cls) // 16},
                                                  {"prog": BitRowLeaf.hiding_schema}).items()}
    except ValueError as e:
        raise SystemExit(f"one-stage's reads refuse the scan's Program: {e}") from e
    want = {f"prog{j}": [None] * sum(len(m.buckets) for m in members) for j in range(cls.NR)}
    for m in members:
        for k, b in enumerate(m.buckets if m.template != "egress" else ()):
            for j in range(cls.NR):
                want[f"prog{j}"][m.base + k] = entry_at(cls, nb, j, b)
    if got != want:
        raise SystemExit(f"one-stage's reads of the scan's Program aren't each unit's slot entries: {got} != {want}")
    return got


def program(M, cls: IS.Class, templates: tuple[str, ...], G: int, T: int, out: Path) -> SimpleNamespace:
    """A round's Program and partition (`members_of`'s templates and counts) and each template's stager, every output
    its own row. The scan's three templates' (`TEMPLATES`) is the generator form (`scan_root`), its one-stage reads
    of the slot entries each unit's (`entry_reads`); every other round's a population of its templates."""
    defs, members = _defs(cls), members_of(templates, G, T, entries=entries_of(cls, G, T))
    counts = counts_of(members)
    nb = G * T // cls.EVERY
    prog = scan_root([(defs[tp], counts[tp], FIXED[tp]) for tp in templates], nb, cls.EVERY) \
        if templates == TEMPLATES else M.P.mixed_population_program([(defs[tp], counts[tp]) for tp in templates])
    query = M.P.template_instances_query([defs[tp] for tp in templates])
    part = M.P.partition(prog, query)
    if templates == TEMPLATES:
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
                           program_file=out / "program.json")


def port_prefix(defn, mr: SimpleNamespace) -> str:
    """Member mr's ports' prefix in its round's registration: its name, the slot entries' (`ROWS_O0`) their template's
    first chunk's, ``<descriptor id>/c0``, so its o0 is the port `NetworkCheck.Period`'s `wired` reads
    (`entries_port`)."""
    return f"{PA.template_id(defn)}/c0" if mr.template in ROWS_O0 else mr.name


def stage(M, pg: SimpleNamespace, mr: SimpleNamespace, out: Path) -> dict:
    """Member mr's statement: each unit's input rows in a shared-row file (each instance its own table row, a slot
    entry's one a boundary, ``mr.refs``), every row, input or output, under the salts and paths of its registered
    entry, in its input set ``mr.set_name``; and the verifier's own `--registered` file. Its outputs are the
    references' (o0 every unit's where it is public, each output row where the reference gives one), and each row's
    b‖c the one the host read. Each framed port's frame root and domain (``mr.framed``) must be its tree's: the
    block's o0 the `prog` tree's, since the scan reads the slot entries only where they are one tree, and a member's
    own rows of a port (`Frame`) its registered root's. Its ports in the registration are ``at``'s (`port_prefix`)."""
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
         "window": mr.window, "buckets": mr.buckets, "dir": out, "meta": st.meta, "pin": st.pin,
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
    """The root the challenger's session opens on (128 hex digits): the SHA-512 of the warden's session and the group's
    registrations, each link's in link-window order, whose heads the warden signed. The gate opens a root once, ever,
    so a warden run's group is drawn once."""
    return hashlib.sha512(canonical_json({"format": WINDOW_ROOT, "session": at.session.hex(),
                                          "registrations": {s: [x.registration for x in ws]
                                                            for s, ws in sorted(at.sides.items())}})).hexdigest()


@contextmanager
def challenger(a, at: SimpleNamespace, out: Path):
    """The challenger's session for the honest round's draw (``--challenger``, as prove_path's `records_audit.audit`
    starts it): `a5_coins.start` on `window_root` for one rank, its evidence copied to ``out / "challenger"``; once the
    rounds are done the rank's turn ends whether or not it drew (`Draws.skip`) and the session's evidence is published,
    which the yielded namespace's ``published`` then holds. None without ``--challenger``. Needs `WZ._one_stage`'s path
    (a5_coins is one-stage's)."""
    if not a.challenger:
        yield None
        return
    import a5_coins as AC

    root = window_root(at)
    manager, draws, finisher = AC.start(a.challenger, root, [0], out / "challenger", a.main, ssh_key=a.gate_key,
                                        peer=a.peer, key_record=a.challenger_key)
    coins = SimpleNamespace(AC=AC, draws=draws, root=root, challenger=a.challenger, published=None)
    try:
        yield coins
    finally:
        draws.skip(0)
        finisher.join()
        coins.published = draws.evidence()
        manager.shutdown()


def spec_of(pg: SimpleNamespace, members: list[dict], want: dict, schemas: dict[str, str], k: int,
            root: str) -> tuple[S.Spec, list, dict[str, int]]:
    """The honest round's `service.Spec` (prove_path's `a5.stream_of`): its unit the round's templates in order, its
    values the round's ports (each member's frame ports and the scan's trees, under their hiding ``schemas``), its
    public items what the registration shows (N, each template's stratum, the group, the leaf layer and each port's
    layout), its selection ``stratified:k``, its proof deadline `MEMBER_DEADLINE_S` a member and its context the
    group's root. -> (the spec, its units, `Stream.register`'s counts: each template's units, over its members)."""
    units = [pg.stagers[tp].defn for tp in pg.templates]
    law = S.Law(f"stratified:{k}")
    n = sum(m["n"] for m in members)
    strata = [s["template"] for s in DR.derive(pg.program, pg.query, S.selection_law(law, n)).get("strata") or ()]
    public = [("population", "N, the round's proof units"), *((f"population:{t}", "a template's stratum") for t in strata),
              ("window", "the round's provenance: the group's link-windows of the warden's run"),
              ("leaf_layer", "the leaf layer over the members' public files"),
              *((f"{x}:{p}", "a port's layout") for p in sorted(want) for x in ("leaves", "domain"))]
    check = S.Check(tuple(PA.program_digest(Program(u)) for u in units), pg.query,
                    proof_deadline=float(MEMBER_DEADLINE_S * len(members)))
    spec = S.Spec(tuple(S.Value(p, schemas[p], "serve") for p in sorted(want)), law, check,
                  tuple(S.Public(item, "structure", why) for item, why in public), bytes.fromhex(root))
    counts: dict[str, int] = {}
    for m in members:
        tid = PA.template_id(m["defn"])
        counts[tid] = counts.get(tid, 0) + m["n"]
    return spec, units, counts


def coins_outcome(a, coins: SimpleNamespace, spec: S.Spec, units: list, counts: dict, stream: S.Stream, draw: S.Draw,
                  rec: dict, want: dict, leaf_layer: str, read: bytes, segment: dict | None, sessions: list[str],
                  verdicts: dict, statement: dict, given: dict) -> tuple[dict | None, dict, dict]:
    """The honest draw's outcome of record once the challenger's session has ended (prove_path's `a5.coins_outcome`):
    its proof delivered and its outcome with the coins' evidence (`a5_coins.evidence`), the Lean's whole run (`Run.run`,
    `coin_source` challenger, its window `window_root`) held to the Python's, the audit failing ``coins`` where
    `coins.coins_reason` refuses; then each altered evidence (`a5_coins.negatives`, and the evidence for another window)
    as its own stream's outcome on the honest draw's bytes (`a5_coins.Replay`), refused where `coins_reason` refuses and
    the Lean's run refuses at its select. -> (the audit record, None where the stream refused or holds the outcome; the
    negatives; the coins' record)."""
    AC, root, after = coins.AC, coins.root, bytes.fromhex(draw.registration.digest)
    published = coins.draws.evidence()
    ev = AC.evidence(published, segment) if segment is not None else None

    def outcome(s: S.Stream, d: S.Draw, evidence: dict | None) -> tuple[S.Outcome, dict | None]:
        late = s.deliver(d, sessions)
        return (late if late is not None else s.outcome(d, verdicts, statement, coins=evidence, **given)), \
            s.records[-1]["audit"]

    def reason(evidence: dict | None, window: str = root) -> str | None:
        return CC.coins_reason(evidence, window, after, read, AC.lms_verify)

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
    bad = {**AC.negatives(ev, coins.draws.pre()), "coins-for-another-window": ev} if ev is not None else {}
    for name, evidence in bad.items():
        window = "00" * 64 if name == "coins-for-another-window" else root
        s = S.Stream(spec, units, budget=1, delta=a.delta, challenger=CHALLENGER, coins=AC.Replay(read))
        try:
            d = s.select(s.register(rec, want, leaf_layer=leaf_layer, window_root=window, counts=counts))
            if d.draw != draw.draw:
                raise ValueError(f"the replayed draw {d.units} is not the honest draw {draw.units}")
            got, audit = outcome(s, d, evidence)
            accepted, side = got.accepted, AC.lean_side(audit)
        except (S.Refused, S.Pending, ValueError, AC.Ch.CoinError) as e:
            accepted, side = None, {"refused": AC._refusal(e)}
        py = reason(evidence, window)
        step = ((side or {}).get("step"), (side or {}).get("codes"))
        neg[name] = {"refused": py is not None and accepted is False and step == AC.SELECT_COINS, "accepted": accepted,
                     "python": py, "lean": side}
    checks = (result or {}).get("checks") or {}
    record = {"challenger": coins.challenger, "source": stream.coin_source, "window_root": root, "segment": segment,
              "read": read.hex(), "coins_reason": why, "outcome": told,
              "run": None if result is None or not result["verifiers"]["agree"] else checks.get("run", {"ok": True})}
    return result, neg, record


# --- the audit ----------------------------------------------------------------------------------------------------------

def warden_roots(members) -> list[dict]:
    """The registration's ``rows/<direction>/<w>`` roots: each warden row registration its members' warden ports read
    (an ingress unit's ``row``, an egress unit's ``o1``), its root and domain as the warden registered them
    (`warden_rows.registration`, read back by the verifier), its T rows the leaves. A period check
    (`NetworkCheck.Period`'s rows form) holds a link-window's answer to its rows by these, by root and domain.

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
          port: int, k: int | None = None, coins: SimpleNamespace | None = None) -> dict:
    """One round registered (its members' ports under their prefix, `port_prefix`, each a frame domain of a leaf an
    instance or a shared port's table row; the scan's registrations' roots; and the warden's row registrations its
    warden ports read, `warden_roots`, before any coin), its units drawn by the verifier's Lean (`draw --stratified k`
    on its own coins; every unit when `k` is None) or, given the challenger's ``coins``, by the round's `service.Stream`
    on them (`spec_of`; `a5_coins.RankCoins`, rank 0), each member's drawn units proved --zk behind the firewall and
    held by Lean with the verifier's own registered reads; then each held unit's public ``o0`` (none of the slot
    entries', `ROWS_O0`) opened under the salt the prover sends. A member with no unit drawn runs no session. Member j's
    session listens on ``port + j``, or with ``port`` 0 on a port the OS picks free. Writes what the verifier received
    under ``out / "verifier"``. An accepted audit carries its charge terms (`charge_terms`) and each member's cost per
    drawn unit; on the challenger's coins, the outcome of record is the stream's (`coins_outcome`)."""
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
        d = ADV.domain(reg.binding, len(reg.rows)).domain_id.hex()
        want[f"scan/{kind}"] = {"leaves": len(reg.rows), "domain": d}
        schemas[f"scan/{kind}"] = BitRowLeaf.hiding_schema
        roots.append({"port": f"scan/{kind}", "root": reg.root.hex(), "leaves": len(reg.rows), "domain": d})
    for r in warden_roots(members):
        want[r["port"]] = {"leaves": r["leaves"], "domain": r["domain"]}
        schemas[r["port"]] = WR.SCHEMA
        roots.append(r)
    leaf_layer = hashlib.sha512(M.canonical_json(publics)).hexdigest()
    k = sum(m["n"] for m in members) if k is None else k
    law = M.D.parse_law(f"stratified:{k}")
    win = {"kind": "stand-in", "of": f"round {name}: link-windows {at.windows} of {at.source}, their ingress and egress "
                                     f"at {at.setup.name}'s class (scan_zk.py)",
           "members": {m["name"]: {"template": P.template_id(m["defn"]), "units": [m["base"], m["base"] + m["n"]]}
                       for m in members},
           "link": "registered: each warden row port, input or output, at the warden's signed root of its "
                   "link-window, unit base + k row k of the member's rows; the scan's state, prog and view, inputs and "
                   "outputs, at the prover's registered roots (scan/*)"}
    rec = R.record(pg.program, pg.query, law, a0.SCHEME, roots, win, leaf_layer=leaf_layer)
    (out / "registration.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    checks = {"registration": V.check(rec, pg.program, pg.query, law, a0.SCHEME, want, leaf_layer=leaf_layer).to_json()}
    for m in members:
        checks[f"statement/{m['name']}"] = a7.statement_checks({**m, "name": m["at"]}, rec, want)
    receipt = R.receipt(rec, "one-stage-e2e verifier (scan_zk.py)")
    res: dict = {"checks_before_draw": {c: v["ok"] for c, v in checks.items()}}
    vdir = out / "verifier"
    vdir.mkdir(exist_ok=True)
    if not all(v["ok"] for v in checks.values()):
        return {**res, "checks": checks, "accepted": False, "window_accepted": False, "verdicts": {}, "why": {},
                "opened": {}, "failed": {c: v.get("codes") for c, v in checks.items() if not v["ok"]}}
    stream = None
    if coins is None:
        p = a0.run([a.lean, "draw", "--program", str(pg.program_file), "--partition", str(pg.partition_file),
                    "--stratified", str(k)], out / "draw.log")
        lean_draw = json.loads(next(x for x in p.stdout.splitlines() if x.startswith("DRAW ")).split(" ", 1)[1])
    else:
        spec, units, counts = spec_of(pg, members, want, schemas, k, coins.root)
        rank = coins.AC.RankCoins(coins.draws, 0)
        draw = refused = None
        try:
            stream = S.Stream(spec, units, budget=1, delta=a.delta, challenger=CHALLENGER, coins=rank)
            rank.plan(coins.AC.need(f"stratified:{k}", pg.program, pg.query, rec["population"]))
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
            coins.draws.skip(0)
            log(f"{name}: the stream drew nothing on the challenger's coins: {refused}")
            for f in (out / "registration.json", out / "segment.json"):
                os.link(f, vdir / f.name)
            return {**res, "accepted": False, "window_accepted": False, "verdicts": {}, "why": {"draw": refused},
                    "opened": {}, "draw": {"law": f"stratified:{k}", "coins": coins.challenger, "refused": refused}}
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
        result, neg, told = coins_outcome(a, coins, spec, units, counts, stream, draw, rec, want, leaf_layer, read,
                                          segment, sessions, verdicts, statement, given)
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
    for side, ws in at.sides.items():
        for w, s in zip(at.windows, ws):
            (vdir / f"warden-registration-{side}-{w}.json").write_text(json.dumps(s.registration, indent=1,
                                                                                  sort_keys=True))
    if name == "tampered":
        (vdir / f"warden-registration-egress-{at.windows[at.tampered.window]}-tampered.json").write_text(
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
    source = "the verifier's Lean's own (flock-verify draw, OS randomness)" if stream is None else \
        f"the auditor's challenger's ({coins.challenger}: a5_coins, the round's service.Stream)"
    return {**res, "accepted": bool(result.get("accepted")), "window_accepted": window_ok, "opened": opened,
            "draw": {"law": f"stratified:{k}", "coins": source, "units": lean_draw["units"], "of": rec["population"]},
            **({"coins": told, "coin_negatives": {x: y["refused"] for x, y in neg.items()}} if told else {}),
            "audit": {x: result.get(x) for x in ("accepted", "complete", "wrong_units")},
            "failed": {c: v.get("codes") for c, v in (result.get("checks") or {}).items() if not v.get("ok")},
            "charge": charge_terms(M, result, members, a.delta),
            "per_unit": per_unit(members, statement),
            "verdicts": {x: y.get("accepted") for x, y in verdicts.items()},
            "why": {x: y.get("why") for x, y in verdicts.items()}, "timing": statement}


def charge_terms(M, result: dict, members: list[dict], delta: float) -> dict | None:
    """An accepted audit's terms (prove_path's `charge_terms`): ε, at most ``wrong_units.bound`` of its ``of`` units
    wrong except with probability δ; λ_δ (``location_bits``), β_δ (``harm_bound``: each template's units weighing the
    bits one of its units commits, `partition._template_bits`) and ``exfiltration_bound`` = β_δ + λ_δ. None where the
    audit isn't accepted, which bounds nothing."""
    if not result.get("accepted"):
        return None
    p = AU.profile(result["registration"], result["draw"], delta)
    bits = {M.P.template_id(m["defn"]): float(M.P._template_bits(m["defn"])) for m in members}
    wu = result.get("wrong_units") or {}
    return {"delta": delta, "wrong_units": wu, "epsilon": wu["bound"] / wu["of"] if wu.get("of") else None,
            "template_bits": bits, "location_bits": location_bits(p), "harm_bound": p.harm_bound(bits),
            "exfiltration_bound": exfiltration_bound(p, bits)}


def per_unit(members: list[dict], statement: dict) -> dict[str, dict]:
    """Each member's session per drawn unit (prove_path's `per_unit`): C-Flock's ``prove_total_s``, the session's wall
    and Lean's verify, over the units the session proved; and a unit's ANDs and m."""
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


def window_answer(reg: dict, record: dict | None, delta: Fraction, reason: str) -> dict:
    """One-stage's answer for the group's registration ``reg`` as a period check reads a link-window's scan answer
    (`NetworkCheck.Period`'s rows form, `pouw.core_check.answer_of`): from an accepted audit record (``record``,
    `audit`'s ``audit.json``), its draw and proved units with one-stage's profile and bound on wrong units at
    ``delta`` (`audit.profile`, `audit.wrong_units`: every unit drawn, the bound 0); else a reject naming ``reason``."""
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
                       "per_template": wrong.get("per_template"), "proved": sorted(record["proved_units"])},
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
    return reg, window_answer(reg, record, delta, reason)


def answers(at: SimpleNamespace, out: Path, res: dict, delta: Fraction, block: dict | None = None) -> dict:
    """``answers.json`` (`ANSWERS`) of the honest round's audit (``res``, its files under ``out / "honest"``): its
    registration and, for each link-window W of the group, the scan's answer for W and the warden's ingress and egress
    row registrations of W, the rows form's evidence of W but its certificates. The answer is the group's for every W
    (`window_answer`: accepted where the round's outputs all open to 1), since its registration holds each W's rows
    (`warden_roots`). A refused round's (a void run's) names the members Lean refused. With the block round's audit
    (``block``), the compute audit whose o0 is the scan's `prog` (``audits``): its registration and answer, and the
    terms a period's parameters give it, its Program's descriptor, its free inputs (every bit of every slot entry, the
    stand-in's ``x``: it copies its input) and no steps."""
    reg, answer = _answer(out, "honest", res, delta, f"the scan's audit of link-windows {at.windows}")
    audits = []
    if block is not None:
        breg, banswer = _answer(out, "block", block, delta, "the block's audit")
        audits.append({"answer": banswer, "free": entries_of(at.setup.cls, at.G, at.T) * IS.entry_bits(at.setup.cls),
                       "program": json.loads((out / PROGRAMS[BLOCK] / "program.json").read_text()),
                       "registration": breg, "steps": 0})
    doc = {"format": ANSWERS, "delta": str(delta), "registration": reg, "audits": audits,
           "windows": [{"window": w, "answer": answer,
                        "registrations": [at.sides[s][i].registration for s in ("ingress", "egress")]}
                       for i, w in enumerate(at.windows)]}
    (out / "answers.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    return {"verdict": answer["verdict"], "bound": (answer["public"] or {}).get("bound"), "reason": answer["reason"],
            "commitment": answer["commitment"]}


def expected(at: SimpleNamespace, name: str, res: dict) -> bool:
    """An accepted round: accepted, every output 1; the unread round: its proof held and its audit accepted, and its
    unit's output opened to 0, so the link-window is refused; a refused one (`rounds_of`): refused by Lean's
    registered reads ("does not open") at each of its target members (`targets`), every other member's proof held."""
    lean = {k.split("/", 1)[1]: v for k, v in res["verdicts"].items() if k.startswith("lean-verify/")}
    outcome = rounds_of(at)[name][1]
    if outcome == "accepted":
        return res["window_accepted"] and all(lean.values())
    if outcome == "ok 0":
        return (res["accepted"] and not res["window_accepted"] and bool(lean) and all(lean.values())
                and res["opened"] == {targets(at, name)[0]: [0]})
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


def prove(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    setup = SETUPS[a.setup]
    h = host(out / "warden", a.warden_run, a.draw, setup, a.group)
    at, T, G = h.at, h.at.T, h.at.G
    ch, cls, N = at.chain, setup.cls, 2 * G * T
    if a.draw is not None and not 3 <= a.draw <= N:
        raise SystemExit(f"--draw {a.draw}: a round's draw is 3 .. {N} of its {N} units (one a template at least)")
    ks = {"honest": a.draw, "chain": a.draw, "tampered": None if a.draw is None else stratum_k(a.draw, G * T, N),
          "unread": None, "block": None}
    sh = cls.egress
    summary: dict = {"source": at.source, "warden_run": at.run_id, "draw": ks if a.draw is not None else "every unit",
                     "setup": setup.name, "class": asdict(cls), "params": asdict(at.params), "shape": asdict(sh),
                     "T": T, "group": G, "windows": at.windows,
                     "bits": {f"{tp}/{r}": b for tp, (ins, outs) in ports_of(cls).items() for r, b in ins + outs},
                     "members": {name: [{"name": m.name, "base": m.base, "n": m.n} for m in ms]
                                 for name, ms in h.rounds.items()},
                     "admitted": at.admitted, "anchors": {q: cls.anchor(s, s % cls.EVERY) for q, s in at.admitted.items()},
                     "responses": {q: len(x) for q, x in sorted(at.outputs.items())}, "run_s": at.run_s,
                     "swap": [setup.A, setup.A + 1],
                     "unread": at.unread and {"bucket": at.unread.bucket, "slot": at.unread.slot, "q": at.unread.q},
                     "tampered": at.tampered and {"bucket": at.tampered.bucket,
                                                  "window": at.windows[at.tampered.window], "key": at.tampered.key_of},
                     "void": at.void and {"buckets": at.void, "failed": at.failed, "why": at.failed_why,
                                          "refused": targets(at, "honest")},
                     "cells": {side: {"data": sum(r[s * cls.B] == DATA for r in rows for s in range(sh.R)),
                                      "end": sum(r[s * cls.B] == W.END for r in rows for s in range(sh.R))}
                               for side, rows in (("ingress", ch.ingress), ("egress", ch.egress))},
                     "roots": {k: r.root.hex() for k, r in at.trees.items()},
                     "host": {name: [{"member": m.name, "refused": m.refused, "outputs": m.outputs} for m in ms]
                              for name, ms in h.rounds.items()}}
    void = (f" but its FAIL buckets {at.void} (void: the honest round alone, refused at {targets(at, 'honest')})"
            if at.void else "")
    log(f"run: {at.source}; {setup.name}'s class, link-windows {at.windows} of {T} buckets, admitted {at.admitted}, "
        f"responses {summary['responses']}, anchors {summary['anchors']}, cells {summary['cells']}; every egress "
        f"bucket computes the warden's row from its ingress bucket's view{void}; host reads as expected "
        f"({at.run_s} s); draw {summary['draw']}")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    if a.dry:
        log("dry: done")
        return 0
    M = WZ._one_stage()
    rounds = rounds_of(at)
    pgs = {templates: program(M, cls, templates, G, T, out / PROGRAMS[templates])
           for templates in {t for t, _o in rounds.values()}}
    summary["stagers"] = {f"{PROGRAMS[k]}/{tp}": {
        "unit": g.defn.id, "lower_s": g.lower_s, "k_log": g.st.meta["k_log"], "vus_per_block": g.st.meta["vus_per_block"],
        "pin": g.st.pin, "base": g.base, "n": g.n} for k, pg in pgs.items() for tp, g in pg.stagers.items()}
    log(f"stagers: {summary['stagers']}")
    ok, results = True, {}
    with challenger(a, at, out) as coins:
        if coins is not None:
            summary["coins"] = {"challenger": coins.challenger, "window_root": coins.root, "rounds": ["honest"]}
            log(f"the honest round draws on the challenger's coins ({coins.challenger}), its session on the group's "
                f"root {coins.root[:16]}…; the other rounds on the verifier's Lean's own")
        for j, (name, mrs) in enumerate(h.rounds.items()):
            pg = pgs[rounds[name][0]]
            members = [stage(M, pg, mr, out / name / f"stage-{mr.name}") for mr in mrs]
            if name == "honest":
                summary["wires"] = staged_wires(members)
                if summary["wires"]:
                    raise SystemExit(f"the honest round's chains don't hold on its public files: {summary['wires']}")
            summary.setdefault("rounds", {})[name] = {
                m["name"]: {"ands": m["ands"], "k_log": m["k_log"], "nbl": m["nbl"], "m": m["k_log"] + m["nbl"],
                            "stage_s": m["stage_s"], "outputs": m["outputs"]} for m in members}
            log(f"staged {name}: " + "; ".join(f"{m['name']} {m['ands']} ANDs a unit, m = {m['k_log']} + {m['nbl']}, "
                                               f"outputs {m['outputs']}" for m in members))
            (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
            av = SimpleNamespace(bin=a.bin, lean=a.lean, lease=[], gpu=False, delta=a.delta)
            trees = {**at.trees, **({"state": at.swapped} if name == "chain" else {}),
                     **({"prog": at.unread.tree} if name == "unread" else {})}
            reads = {("egress",): ("view",), BLOCK: ("prog", "state")}.get(pg.templates)
            trees = {k: v for k, v in trees.items() if reads is None or k in reads}
            res = audit(M, av, at, pg, name, members, trees, out / name, a.port and a.port + (2 * G + 2) * j, ks[name],
                        coins if name == "honest" else None)
            results[name] = res
            if name == "honest":
                summary["answers"] = answers(at, out, res, Fraction(a.delta), results.get("block"))
                log(f"answers: link-windows {at.windows}, each {summary['answers']}")
            blobs, held = blobs_of(members)
            blobs, held, public = LS.public_pieces(M.a8, blobs, held, known(members, at), fields=True)
            blobs, held, nsparse = sparse(blobs, held)
            inst = out / name / "instances.bin"
            inst.write_bytes(b"".join((m["dir"] / f"inst-{m['n']}.bin").read_bytes() for m in members))
            res["received"] = {**M.a8.received(out / name / "verifier", inst, blobs, held),
                               "public_pieces": len(public), "sparse_pieces": nsparse}
            inst.unlink()
            rec = res["received"]
            good = expected(at, name, res) and not rec["verifier_found"] and rec["control"]["holds_all"]
            if coins is not None and name == "honest":
                good = good and bool(res.get("coin_negatives")) and all(res["coin_negatives"].values())
            ok &= good
            summary["rounds"][name]["audit"] = res
            log(f"round {name}: drawn {len(res.get('draw', {}).get('units') or [])} of {pg.units}, "
                f"charge {res.get('charge')}, per unit {res.get('per_unit')}; proofs {res['verdicts']}, audit "
                f"{res.get('audit')}, opened "
                f"{ {k: v for k, v in res.get('opened', {}).items()} }, Lean {res['why']}, verifier received "
                f"{rec['verifier_bytes']} bytes, witness pieces found there: {rec['verifier_found'] or 'none'} (of "
                f"{len(blobs)}; {len(public)} public, {nsparse} sparse left out), control "
                f"{rec['control']['blobs_found']}; {'as expected' if good else 'UNEXPECTED'}")
            (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    if coins is not None:
        pub = coins.published or {}
        summary["coins"]["session"] = {x: pub.get(x) for x in ("trusted", "error", "plan")}
        ok &= pub.get("error") is None and (pub.get("trusted") or {}).get("pinned_session") is None
        log(f"the challenger's session: {summary['coins']['session']}")
    summary["ok"] = ok
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    log("done: " + ("ok" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="scan_zk.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prove", help="a group's ingress and egress units chained by registered reads, --zk")
    p.add_argument("--bin")
    p.add_argument("--lean")
    p.add_argument("--out", required=True)
    p.add_argument("--port", type=int, default=0, help="round i's member j listens on PORT + (2G + 2)i + j; 0 (the "
                                                       "default): a free port for each session")
    p.add_argument("--delta", type=float, default=2.0 ** -20)
    p.add_argument("--dry", action="store_true", help="the run and the host's read of every round: nothing proved")
    p.add_argument("--class", dest="setup", choices=sorted(SETUPS), default=W63.name,
                   help="the class and the warden's schedule (default w63's, PARAMS; test: the certifier tests', a CPU "
                        "window)")
    p.add_argument("--group", type=int, default=1, help="G: link-windows 0 .. G - 1 in one registration (default 1)")
    p.add_argument("--warden-run", type=Path,
                   help="a warden_serve run's directory (rehearse --ids at the class's schedule): its group's "
                        "link-windows, verified under its deployment's key, in place of the synthetic run")
    p.add_argument("--draw", type=int,
                   help="K: Lean draw --stratified K on the verifier's own coins, only the drawn units proved; default "
                        "every unit")
    p.add_argument("--challenger", help="with --draw: the honest round's K drawn on the auditor's challenger's coins "
                   "(a5_coins, the round's service.Stream), not the verifier's: local, or gate:USER@HOST "
                   "(the auditor's challenger through its gate, with --peer and --challenger-key)")
    p.add_argument("--peer", help="with --challenger gate:USER@HOST: this machine as the gate's /etc/vy/peers names it, "
                   "whose 127.0.0.1 the session's challenger listens on")
    p.add_argument("--challenger-key", help="with --challenger gate:USER@HOST: the challenger's key record of its "
                   "deployment (challenger-key/v1), never the session's key.json")
    p.add_argument("--gate-key", help="with --challenger gate:USER@HOST: the lane's own gate key (default ssh's own)")
    p.add_argument("--main", help="with --challenger: the verifier's own copy of main's first-parent history (git "
                   "rev-list --first-parent origin/main), which must hold the commit the session's host pins")
    s = sub.add_parser("schedule", help="a setup's schedule (w63's, PARAMS, by default) as warden_serve.py's schedule "
                                        "flags (serve, rehearse): the warden runs --warden-run reads")
    s.add_argument("--class", dest="setup", choices=sorted(SETUPS), default=W63.name)
    a = ap.parse_args(argv)
    if a.command == "schedule":
        print(" ".join(f"--{k.replace('_', '-')} {v}" for k, v in asdict(SETUPS[a.setup].params).items()))
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
