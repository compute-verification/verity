"""scan-zk: the network certifier's step-indexed scan composed in zero knowledge, in the generator form. Ingress bucket t
(`ingress_scan.IngressScanBucket`; bucket 0's `IngressScanFirst`, from the empty state) reads the warden's ingress row
t, the scan's state S_{t−1} and the Programs' row t, and computes S_t and its egress view; egress bucket t
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
  tree's root in the registration), G·T leaves each: ``state``, leaf t S_t; ``prog``, leaf t the Programs' row t (each
  request's response ids at its anchor, standing in for the anchored Programs' outputs); and ``view``, leaf t S_t's
  egress view.
* **The honest round, one registration**: one Program of three templates (`TEMPLATES`): the first unit
  (`IngressScanFirst`, bucket 0), G·T − 1 ingress units (buckets 1 … G·T − 1) and G·T egress units, its members split
  a link-window (`members_of`: ``first``, ``ingress-w`` and ``egress-w``), every unit drawn by the verifier's Lean.
  Ingress unit t reads ``row`` (the warden's ingress row t), ``sin`` (state leaf t − 1; the first unit none) and
  ``prog`` (leaf t), and outputs ``o0``, its ok bit, ``o1``, S_t, committed as state leaf t, and ``o2``, its view,
  committed as view leaf t. Egress unit t reads ``state`` (view leaf t) and outputs ``o0``, its render bit, and ``o1``,
  the row it computes, committed as the warden's egress row t. Each output row but o0 is a registered read of the row
  it is committed as (`Registered.checkPort` holds a hidden-output statement's output rows, the verifier's PROTOCOL.md
  §16.13), so S_t is the one leaf unit t computes and unit t + 1 reads, the view the one leaf ingress t computes and
  egress t reads, and the egress row the one the warden signed: the chains are wires through the registered roots, not
  the prover's word. Accepted: every o0 opens to 1. The registration's roots also name, as
  ``rows/<direction>/<w>`` (`warden_roots`), the root and domain of each warden row registration a member's warden
  port reads (ingress's ``row``, egress's ``o1``), so a period check holds the answer to them.
* **The answers** (``answers.json``, `answers`): the honest round's registration and, for each link-window W of the
  group, the warden's two row registrations of W and the scan's one-stage answer (`window_answer`), the same for every
  W, which `verity/certifier/v0`'s rows form (`NetworkCheck.Period`) checks for W with W's three records'
  certificates: accepted, one-stage's profile and bound at ``--delta`` (every unit drawn, the bound 0); refused, a
  reject naming the members Lean refused.
* **A void run** (a warden run whose warden failed a connection closed, its `FAIL` in the egress row it signed, its
  reason in the ``fails`` record): the scan renders the connection's end where the warden wrote `FAIL`, and that bucket
  alone differs from the warden's row. Its honest round alone is proved, expected refused by Lean's registered reads
  ("does not open") at the egress members of its FAIL buckets (`rounds_of`, `targets`), every other member held, and
  its answer is that refusal, so the period check rejects the link-windows by name.
* **Negatives**, each refused by Lean's registered reads ("does not open"). ``chain``: the same Program on a state tree
  with S_A and S_{A+1} swapped (leaves A and A + 1): ingress unit A's o1, the S_A it computes, doesn't open leaf A.
  ``tampered``: an egress-only Program of G·T units, one egress row's `DATA` id changed and its link-window re-signed (a
  warden that saw a byte the scan doesn't render; a warden run's re-signed by the driver's own key, under which the
  verifier reads it): egress unit t's o1, the row its view renders, doesn't open the signed row t.
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
  this machine (its own key; it shows the path, not an auditor's coins); ``gate:USER@HOST`` is vy-challenger-1's through
  its gate, its challenger on ``--peer``'s 127.0.0.1, under the deployment's key record ``--challenger-key``. The
  gate opens a root once, ever: a run's group is drawn once (the synthetic run's registrations carry fresh salts, so
  each synthetic run has a root of its own). The other rounds keep the verifier's Lean's own coins: one session serves
  one draw, and their refusal is Lean's registered reads, whatever the coins.
* **What the verifier receives** is scanned for every piece of every unit's rows, inputs and outputs, and every salt
  but ``o0``'s, which the prover sends so the verifier opens each unit's output (`a8.received`), less the pieces public
  anyway (the circuits, zeros, the window's pads and `END` cells) and the sparse pieces (at most one nonzero byte); the
  prover's instance file, which holds them all, is the control.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
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
from verity.ml.circuits.codec import canonical_json  # noqa: E402
from verity.ml.circuits.defs import bind  # noqa: E402
from verity.ml.circuits.program import Program  # noqa: E402
from verity.ml.commitments import advice as ADV  # noqa: E402
from verity.ml.commitments.frame_v3 import BitRowLeaf  # noqa: E402
from verity.ml.commitments.rowleaf import bit_row, row_leaf  # noqa: E402
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
#: each round's templates, and its outcome: accepted, or refused by Lean's registered reads
ROUNDS = {"honest": (TEMPLATES, "accepted"), "chain": (TEMPLATES, "does not open"),
          "tampered": (("egress",), "does not open")}
VALUES = {k: f"verity/certifier/scan-{k}/v1" for k in ("state", "prog", "view")}
TIMING = ("e2e_s", "wait_s", "witness_s", "prove_total_s", "proof_bytes", "wire_sent", "wire_recv", "k_log", "nbl",
          "dense_m", "units", "ands_per_vu", "prover", "gpu")
#: the honest round's stream on the auditor's challenger's coins (``--challenger``): its challenger, the format of the
#: group's root its session opens on (`window_root`), and each member's share of the proof deadline
CHALLENGER = "scan_zk.py on the auditor's challenger's coins"
WINDOW_ROOT = "verity/certifier/scan-window-root/v2"
MEMBER_DEADLINE_S = 5 * 3600
#: the format of ``answers.json`` (`answers`): the honest round's registration and each link-window's answer and rows
ANSWERS = "verity/certifier/scan-answers/v1"


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
    """Each template's input rows and its output rows past o0, (port, bits) in its signature's order."""
    row, prog = ("row", 8 * cls.S * cls.B), ("prog", IS.prog_bits(cls))
    outs = (("o1", IS.state_bits(cls)), ("o2", IS.view_bits(cls)))
    return {"first": ((row, prog), outs), "ingress": ((row, ("sin", IS.state_bits(cls)), prog), outs),
            "egress": ((("state", IS.view_bits(cls)),), (("o1", 8 * cls.egress.R * cls.B),))}


def members_of(templates: tuple[str, ...], G: int, T: int) -> list[SimpleNamespace]:
    """A round's members in its Program's order, one a template and link-window w of the group: its name, its first
    unit (``base``) and the scan's buckets its units compute, unit ``base + k`` bucket ``buckets[k]``. The first unit
    computes bucket 0, the ingress template buckets 1 … G·T − 1 (with no first unit, 0 …), the egress template
    0 … G·T − 1."""
    out, at = [], 0
    for tp in templates:
        lo = 1 if tp == "ingress" and "first" in templates else 0
        for w in range(1 if tp == "first" else G):
            bs = range(1) if tp == "first" else range(max(lo, w * T), (w + 1) * T)
            out.append(SimpleNamespace(template=tp, window=w, name="first" if tp == "first" else f"{tp}-{w}",
                                       base=at + bs.start - lo, buckets=bs))
        at += counts_of(out)[tp]
    return out


def counts_of(members: list[SimpleNamespace]) -> dict[str, int]:
    """Each template's units over its members."""
    out: dict[str, int] = {}
    for m in members:
        out[m.template] = out.get(m.template, 0) + len(m.buckets)
    return out


# --- the scan's rows: hiding rows the prover registers -----------------------------------------------------------------

def tree(kind: str, rows: list[list[int]], heads: list, session: bytes) -> ADV.Registration:
    """`kind`'s rows (bit rows of whole u16 words) committed as hiding rows, bound to the session and the group's
    ingress link-windows' heads."""
    binding = hashlib.sha512(b"\0".join([VALUES[kind].encode(), session,
                                         *(x for h in heads for x in (h.value.encode(), h.root))])).digest()
    return ADV.register([bit_row((_words(r), 16))[0] for r in rows], len(rows[0]), binding)


def tree_bc(reg: ADV.Registration, i: int, bits: list[int] | None = None) -> bytes:
    """Row ``bits`` (leaf i's own row by default) committed under leaf i's salt, b‖c: as a statement's table row of it
    is, which opens leaf i only when the row is leaf i's."""
    row = reg.rows[i] if bits is None else bit_row((_words(bits), 16))[0]
    return ADV.INST.commit_string(reg.key, BitRowLeaf(reg.n_bits).layout().digest(row), reg.salts[i])


def tree_port(kind: str, reg: ADV.Registration, reads: list[int], reads_from: int) -> dict:
    """The verifier's own ``--registered`` port of `kind`'s registration: unit ``reads_from + i`` reads leaf reads[i]."""
    n = len(reg.rows)
    e = {"leaves": n, "words": reg.n_bits // 16, "word_bits": 16, "domain": ADV.domain(reg.binding, n).domain_id.hex()}
    port = produced.port(VALUES[kind], e, reg.root.hex(), reads, BitRowLeaf(reg.n_bits).hiding_schema)
    return port | ({REG.READS_FROM: int(reads_from)} if reads_from else {})


def tree_entry(kind: str, reg: ADV.Registration, positions) -> dict:
    return {"value": VALUES[kind], "positions": [int(p) for p in positions],
            "paths": [b"".join(reg.open(p).path).hex() for p in positions]}


def tree_refusal(port: dict, reg: ADV.Registration, entry: dict, bcs: list[bytes | None]) -> str | None:
    """The host's read of a statement's entry on a scan registration, as Lean's registered reads make it (in
    `warden_rows.refusal`'s order): None when it holds. A b‖c of None is a row the host can't name (an output the
    reference gives none of), which opens no leaf."""
    if entry["value"] != port["value"]:
        return f"registered reads: the port reads {port['value']}, not another value ({entry['value']})"
    n, root = len(reg.rows), bytes.fromhex(port["root"])
    d, scheme = ADV.domain(reg.binding, n), ADV._scheme(reg.binding, n, reg.n_bits)
    for r, (p, bc, path) in enumerate(zip(entry["positions"], bcs, entry["paths"], strict=True)):
        nodes = tuple(bytes.fromhex(path[i:i + 128]) for i in range(0, len(path), 128))
        if bc is None or not scheme.verify_path(ADV.PORT, root, n, p, row_leaf(
                d, p, ADV.INST.tree_leaf(reg.key, bc), BitRowLeaf(reg.n_bits).hiding_schema), nodes):
            return f"registered reads: table row {r} does not open the registered root at position {p}"
    base = int(port.get(REG.READS_FROM, 0))
    for i, (p, want) in enumerate(zip(entry["positions"], port["reads"], strict=True)):
        if p != want:
            return f"registered reads: unit {base + i} reads position {p}, and the verifier's program reads {want}"
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
    that head. The units read those rows. The Programs' outputs (the ``prog`` row's stand-in) are each connection's ids
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


def link(out: Path, run: Path | None = None, setup: Setup = W63, group: int = 1, start: int = 0) -> SimpleNamespace:
    """The run (`synthetic`, or `warden_run`'s of a warden_serve run), link-windows start … start + group − 1 of each
    link registered by the warden and read back by the verifier, the scan's chain over their rows from the empty state
    (`scan_traffic.chain`), the prover's three registrations, the chain round's state tree and the tampered round's
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
    if not 1 <= setup.A < setup.A + 1 < group * T:
        raise SystemExit(f"the chain round swaps S_A and S_(A+1) of ingress units, 1 <= A < {group * T - 1}")
    heads = [w.head for w in got.sides["ingress"]]
    states = ch.states[1:]
    trees = {k: tree(k, rows, heads, got.session) for k, rows in (("state", states), ("prog", ch.progs),
                                                                 ("view", ch.views))}
    swapped = list(states)
    swapped[setup.A], swapped[setup.A + 1] = swapped[setup.A + 1], swapped[setup.A]
    admitted: dict[int, int] = {}
    for st in ch.scan:
        for s in st.slots:
            if s.live:
                admitted.setdefault(s.q, s.admitted)
    return SimpleNamespace(setup=setup, chain=ch, T=T, G=group, windows=windows, params=got.params,
                           session=got.session, source=got.source, run_id=got.run_id, outputs=got.outputs,
                           sides=got.sides, trees=trees, swapped=tree("state", swapped, heads, got.session),
                           tampered=None if void else _tamper(got, ch.egress, T, cls.B, cls.egress.R),
                           ordinals=sorted(admitted), admitted=dict(sorted(admitted.items())), void=void,
                           failed=got.failed, failed_why=got.failed_why, run_s=round(time.time() - t0, 1))


def _bits(reg: ADV.Registration, i: int) -> list[int]:
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
    the salts its rows, inputs and outputs but o0, are committed under, its header's `registered` entries and the
    verifier's own ports, the host's read of them (`warden_rows.refusal`, `tree_refusal`: the ports in its signature's
    order, inputs then outputs), each unit's o0 word and its output rows as the references compute them (``made``,
    None where the ingress reference gives none), and the warden's row registration each warden port reads
    (``warden``)."""
    ch, T, cls = at.chain, at.T, at.setup.cls
    tp, w, bs = mb.template, mb.window, list(mb.buckets)
    rows = [b - w * T for b in bs]
    state, view, prog = (at.swapped if name == "chain" else at.trees["state"]), at.trees["view"], at.trees["prog"]
    ins, outs = ports_of(cls)[tp]
    if tp == "egress":
        side = at.tampered if name == "tampered" and at.tampered.window == w else at.sides["egress"][w]
        reads = {"state": ("view", view, bs)}
        warden = {"o1": side}
        if any(_bits(view, b) != EG.state_bits(cls.egress, ch.view(b)) for b in bs):
            raise SystemExit("a view leaf isn't its bucket's egress view")
        got = [EG.outputs(cls.egress, ch.view(b)) for b in bs]
        outputs, made = [o for o, _ in got], {"o1": [r for _, r in got]}
    else:
        reads = {**({"sin": ("state", state, [b - 1 for b in bs])} if tp == "ingress" else {}),
                 "prog": ("prog", prog, bs), "o1": ("state", state, bs), "o2": ("view", view, bs)}
        warden = {"row": at.sides["ingress"][w]}
        got = [IS.reference(cls, ch.ingress[b], _bits(state, b - 1) if tp == "ingress" else None, _bits(prog, b))
               for b in bs]
        outputs, made = [o for o, _s, _v in got], {"o1": [s for _o, s, _v in got], "o2": [v for _o, _s, v in got]}
    units = [tuple(ch.ingress[b] if r == "row" else _bits(reads[r][1], reads[r][2][k]) for r, _b in ins)
             for k, b in enumerate(bs)]
    salts, entries, ports, bcs = {}, {}, {}, {}
    for p, side in warden.items():
        _w, salts[p], entries[p] = side.cells.statement(rows)
        ports[p] = {**WR.bucket_port(side.head, at.params, mb.base), "reads": rows}
        bcs[p] = [WR.bc(made[p][k] if p in made else side.cells.rows[r], side.cells.salts[r]) for k, r in enumerate(rows)]
    for p, (kind, reg, ps) in reads.items():
        salts[p], entries[p] = [reg.salts[i] for i in ps], tree_entry(kind, reg, ps)
        ports[p] = tree_port(kind, reg, ps, mb.base)
        bcs[p] = [None if p in made and made[p][k] is None else tree_bc(reg, i, made[p][k] if p in made else None)
                  for k, i in enumerate(ps)]
    refused = header_refusal(ports, entries)
    for p in [r for r, _b in ins] + [o for o, _b in outs]:
        refused = refused or (WR.refusal(ports[p], warden[p].dom, entries[p], bcs[p]) if p in warden else
                              tree_refusal(ports[p], reads[p][1], entries[p], bcs[p]))
    changed = at.tampered.bucket if name == "tampered" and tp == "egress" and at.tampered.window == w else None
    return SimpleNamespace(round=name, template=tp, window=w, name=mb.name, base=mb.base, n=len(bs), buckets=bs,
                           units=units, made=made, salts=salts, entries=entries, ports=ports, bcs=bcs, refused=refused,
                           outputs=outputs, changed=changed, warden={p: s.registration for p, s in warden.items()})


def rounds_of(at: SimpleNamespace) -> dict[str, tuple[tuple[str, ...], str]]:
    """The rounds a run is proved in, each with its templates and outcome: `ROUNDS`; a void run's (the warden failed a
    connection closed, a `FAIL` in the egress row it signed) its honest round alone, refused by Lean's registered reads
    ("does not open") at the egress members of its FAIL cells, where the scan renders the connection's end."""
    return {"honest": (TEMPLATES, "does not open")} if at.void else ROUNDS


def targets(at: SimpleNamespace, name: str) -> list[str]:
    """The members a refused round's refusal names: chain's, the ingress member of unit A; tampered's, the egress
    member of the re-signed link-window; a void run's honest round, the egress member of each FAIL's link-window."""
    if name == "chain":
        return [f"ingress-{at.setup.A // at.T}"]
    if name == "tampered":
        return [f"egress-{at.tampered.window}"]
    return sorted({f"egress-{t // at.T}" for t in at.void})


def predicted(at: SimpleNamespace, name: str, members: list[SimpleNamespace]) -> bool:
    """The host's read of a round before any proof, as the round's: accepted, every read held and every output 1;
    refused, its target members' reads refused ("does not open") and every other member's held."""
    outcome = rounds_of(at)[name][1]
    if outcome == "accepted":
        return all(m.refused is None and set(m.outputs) == {1} for m in members)
    t = targets(at, name)
    return all((outcome in (m.refused or "")) if m.name in t else m.refused is None for m in members)


def wired(members) -> list[str]:
    """Where a round's chains don't hold, on its members' b‖c (`member`'s ``bcs``, port to each unit's): bucket b's
    o1 is bucket b + 1's ``sin`` and its o2 egress bucket b's ``state``, the one leaf of the registered roots both
    commit to. Empty when they hold."""
    o1, o2, sin, state = {}, {}, {}, {}
    for m in members:
        for k, b in enumerate(m.buckets):
            if m.template == "egress":
                state[b] = m.bcs["state"][k]
            else:
                o1[b], o2[b] = m.bcs["o1"][k], m.bcs["o2"][k]
                if "sin" in m.bcs:
                    sin[b] = m.bcs["sin"][k]
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
        got[name] = [member(at, name, mb) for mb in members_of(templates, group, at.T)]
        if not predicted(at, name, got[name]) or (name == "honest" and wired(got[name])):
            raise SystemExit(f"{name}: the host's read of the round isn't {outcome}: "
                             f"{[(m.name, m.refused, m.outputs) for m in got[name]]}, chains {wired(got[name])}")
    return SimpleNamespace(at=at, rounds=got, draw=draw)


# --- the statements ---------------------------------------------------------------------------------------------------

def _defs(cls: IS.Class) -> dict[str, object]:
    st = {k: getattr(cls, k) for k in IS.STATICS}
    sh = cls.egress
    return {"first": bind(IS.IngressScanFirst, **st), "ingress": bind(IS.IngressScanBucket, **st),
            "egress": bind(EG.EgressBucket, B=sh.B, R=sh.R, r=sh.r, NR=sh.NR, K=sh.K, IB=sh.IB, D=sh.D)}


def program(M, cls: IS.Class, templates: tuple[str, ...], G: int, T: int, out: Path) -> SimpleNamespace:
    """A round's Program and partition (`members_of`'s templates and counts) and each template's stager, every output
    its own row."""
    defs, members = _defs(cls), members_of(templates, G, T)
    counts = counts_of(members)
    prog = M.P.mixed_population_program([(defs[tp], counts[tp]) for tp in templates])
    query = M.P.template_instances_query([defs[tp] for tp in templates])
    part = M.P.partition(prog, query)
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
        st = M.CS.UnitStager(defs[tp], f"unit/scan-{tp}", [(r, M.V2, b, [r]) for r, b in ins],
                             [(M.V2, (j,)) for j in range(1 + len(outs))], circuit=sdir / "circuit.txt", bind=bound)
        st.pin
        stagers[tp] = SimpleNamespace(defn=defs[tp], st=st, base=base, n=cnt, out=sdir, ins=ins, outs=outs,
                                      lower_s=round(time.time() - t, 1))
    return SimpleNamespace(program=prog, query=query, templates=templates, members=members, stagers=stagers,
                           units=sum(counts.values()), partition_file=out / "partition.json",
                           program_file=out / "program.json")


def stage(M, pg: SimpleNamespace, mr: SimpleNamespace, out: Path) -> dict:
    """Member mr's statement: each unit's input rows in a shared-row file (each instance its own table row), every row
    but o0, input or output, under the salts and paths of its registered entry, ``o0`` under a fresh salt; and the
    verifier's own `--registered` file. Its outputs are the references' (o0 every unit's, each output row where the
    reference gives one), and each row's b‖c the one the host read."""
    t0 = time.time()
    sg = pg.stagers[mr.template]
    st, n = sg.st, mr.n
    nbl = max(3, (-(-n // st.meta["vus_per_block"]) - 1).bit_length())
    if not 25 <= st.meta["k_log"] + nbl <= M.a7.M_MAX:
        raise SystemExit(f"{mr.name}: m = {st.meta['k_log']} + {nbl} is outside 25 .. {M.a7.M_MAX}")
    words = {r: M.np.asarray([WR.words(u[k]) if isinstance(u[k], bytes) else _words(u[k]) for u in mr.units],
                             dtype=M.np.uint16) for k, (r, _b) in enumerate(sg.ins)}
    set_name = f"{st.iset.name}-{mr.round}-{mr.window}"
    salts = {**mr.salts, "o0": [os.urandom(WR.SALT_BYTES) for _ in range(n)]}
    got = st.stage(words, out, os.urandom(32), salts=salts, indices=range(mr.base, mr.base + n), set_name=set_name,
                   tables=words, refs={r: list(range(n)) for r in words}, registered=mr.entries)
    o0 = [int(x[0]) for x in M.np.asarray(got["outs"]["o0"])]
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
    m = {"tag": f"{mr.round}-{mr.name}", "name": mr.name, "template": mr.template, "window": mr.window,
         "buckets": mr.buckets, "dir": out, "meta": st.meta, "pin": st.pin, "class": st.class_sha512, "set": set_name,
         "n": n, "base": mr.base, "defn": sg.defn, "ands": got["ands"], "unit_nnz": got["unit_nnz"],
         "salts": got["salts"], "ports": st.ports, "outputs": o0, "units": mr.units, "ins": sg.ins, "made": made,
         "partition_file": pg.partition_file, "program_file": pg.program_file,
         "registered_file": out / "registered.json", "k_log": st.meta["k_log"], "nbl": nbl, "warden": mr.warden,
         "stage_s": round(time.time() - t0, 1)}
    m["header"] = M.a0.public_header(out, n)
    m["bcs"] = shared_bcs(out / f"pub-{n}.bin", st.ports)
    if any(bc is not None and m["bcs"][i][p] != bc for p, col in mr.bcs.items() for i, bc in enumerate(col)):
        raise SystemExit(f"{mr.round}/{mr.name}: a staged row's b‖c isn't the one the host read")
    if not all(M.a7.opens(next(p for p in st.ports if p.name == "o0"), [w], salts["o0"][i], m["bcs"][i]["o0"])
               for i, w in enumerate(o0)):
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


def audit(M, a, at: SimpleNamespace, pg: SimpleNamespace, name: str, members: list[dict], trees: dict, out: Path,
          port: int, k: int | None = None, coins: SimpleNamespace | None = None) -> dict:
    """One round registered (its members' ports, the scan's registrations' roots, and the warden's row registrations
    its warden ports read, `warden_roots`, before any coin), its units drawn by the verifier's Lean (`draw --stratified
    k` on its own coins; every unit when `k` is None) or, given the challenger's ``coins``, by the round's
    `service.Stream` on them (`spec_of`; `a5_coins.RankCoins`, rank 0), each member's drawn units proved --zk behind the
    firewall and held by Lean with the verifier's own registered reads; then each held unit's ``o0`` opened under the
    salt the prover sends. A member with no unit drawn runs no session. Writes what the verifier received under
    ``out / "verifier"``. An accepted audit carries its charge terms (`charge_terms`) and each member's cost per drawn
    unit; on the challenger's coins, the outcome of record is the stream's
    (`coins_outcome`)."""
    a0, a7, P, R, V = M.a0, M.a7, M.P, M.R, M.V
    out.mkdir(parents=True, exist_ok=True)
    want, roots, publics, schemas = {}, [], {}, {}
    for m in members:
        fv = m["header"]["frame_v3"]
        for p_, w in a7.frame_ports(m["set"], m["meta"], m["n"]).items():
            want[f"{m['name']}/{p_}"] = w
            schemas[f"{m['name']}/{p_}"] = fv["schemas"][p_]
            roots.append({"port": f"{m['name']}/{p_}", "root": fv["roots"][p_], "leaves": w["leaves"],
                          "domain": fv["domain_ids"][p_]})
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
        checks[f"statement/{m['name']}"] = a7.statement_checks(m, rec, want)
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
        record, live, proc = M.a8.session(a, m, sdir, local, port + j, env=WZ.firewall(M, m))
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
        session_roots.update({f"{m['name']}/{p_}": r for p_, r in ((live or {}).get("roots") or {}).items()})
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
    opened, sent = {}, []
    for m in members:
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
    window_ok = bool(result.get("accepted")) and len(got) == len(held) and all(w == 1 for w in got)
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


def answers(at: SimpleNamespace, out: Path, res: dict, delta: Fraction) -> dict:
    """``answers.json`` (`ANSWERS`) of the honest round's audit (``res``, its files under ``out / "honest"``): its
    registration and, for each link-window W of the group, the scan's answer for W and the warden's ingress and egress
    row registrations of W, the rows form's evidence of W but its certificates. The answer is the group's for every W
    (`window_answer`: accepted where the round's outputs all open to 1), since its registration holds each W's rows
    (`warden_roots`). A refused round's (a void run's) names the members Lean refused."""
    reg = json.loads((out / "honest" / "registration.json").read_text())
    f = out / "honest" / "audit.json"
    record = json.loads(f.read_text()) if res.get("window_accepted") and f.exists() else None
    why = "; ".join(f"{k.split('/', 1)[1]}: {v}" for k, v in sorted(res.get("why", {}).items())
                    if v and k.startswith("lean-verify/"))
    failed = "; ".join(f"check {c}: {', '.join(codes or ())}" for c, codes in sorted((res.get("failed") or {}).items()))
    reason = (f"the scan's audit of link-windows {at.windows} refused: " +
              (why or failed or f"audit {res.get('audit')}, outputs {res.get('opened')}"))
    answer = window_answer(reg, record, delta, reason)
    doc = {"format": ANSWERS, "delta": str(delta), "registration": reg,
           "windows": [{"window": w, "answer": answer,
                        "registrations": [at.sides[s][i].registration for s in ("ingress", "egress")]}
                       for i, w in enumerate(at.windows)]}
    (out / "answers.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    return {"verdict": answer["verdict"], "bound": (answer["public"] or {}).get("bound"), "reason": answer["reason"],
            "commitment": answer["commitment"]}


def expected(at: SimpleNamespace, name: str, res: dict) -> bool:
    """An accepted round: accepted, every output 1; a refused one (`rounds_of`): refused by Lean's registered reads
    ("does not open") at each of its target members (`targets`), every other member's proof held."""
    lean = {k.split("/", 1)[1]: v for k, v in res["verdicts"].items() if k.startswith("lean-verify/")}
    outcome = rounds_of(at)[name][1]
    if outcome == "accepted":
        return res["window_accepted"] and all(lean.values())
    ts = targets(at, name)
    whys = [res["why"].get(f"lean-verify/{t}") or "" for t in ts]
    return (not res["accepted"] and all(lean.get(t) is False for t in ts)
            and all("registered reads" in why and outcome in why for why in whys)
            and all(v for x, v in lean.items() if x not in ts))


def blobs_of(members: list[dict]) -> tuple[dict[str, bytes], set[str]]:
    """The witness pieces the verifier must not receive: every unit's rows, inputs and outputs but o0, and every salt
    but ``o0``'s."""
    blobs = {}
    for m in members:
        for i, u in enumerate(m["units"]):
            blobs.update({f"{m['name']} {r} {i}": _bytes(x) for (r, _b), x in zip(m["ins"], u)})
        blobs.update({f"{m['name']} {p} {i}": x for p, col in m["made"].items() for i, x in enumerate(col)})
        blobs.update({f"{m['name']} salt {p} {i}": s for p, ss in m["salts"].items() if p != "o0"
                      for i, s in enumerate(ss)})
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
    ks = {"honest": a.draw, "chain": a.draw, "tampered": None if a.draw is None else stratum_k(a.draw, G * T, N)}
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
    pgs = {templates: program(M, cls, templates, G, T, out / ("joint" if templates == TEMPLATES else "solo"))
           for templates in {t for t, _o in rounds.values()}}
    summary["stagers"] = {f"{'joint' if k == TEMPLATES else 'solo'}/{tp}": {
        "unit": g.defn.id, "lower_s": g.lower_s, "k_log": g.st.meta["k_log"], "vus_per_block": g.st.meta["vus_per_block"],
        "pin": g.st.pin, "base": g.base, "n": g.n} for k, pg in pgs.items() for tp, g in pg.stagers.items()}
    log(f"stagers: {summary['stagers']}")
    ok = True
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
            trees = {**at.trees, **({"state": at.swapped} if name == "chain" else {})}
            trees = {k: v for k, v in trees.items() if pg.templates != ("egress",) or k == "view"}
            res = audit(M, av, at, pg, name, members, trees, out / name, a.port + (2 * G + 2) * j, ks[name],
                        coins if name == "honest" else None)
            if name == "honest":
                summary["answers"] = answers(at, out, res, Fraction(a.delta))
                log(f"answers: link-windows {at.windows}, each {summary['answers']}")
            blobs, held = blobs_of(members)
            blobs, held, public = LS.public_pieces(M.a8, blobs, held, known(members, at))
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
    p.add_argument("--port", type=int, default=7811)
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
                   "(vy-challenger-1 through its gate, with --peer and --challenger-key)")
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
