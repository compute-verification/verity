"""The ingress tie in zero knowledge: each ingress link-window's T bucket rows of the warden's cells, each proved as
IoTieRequest (`tie_service.IngressTie`, its unit `NetworkIngressTieBucket_v1{B=512,L=2,R=2}`), registered on the link's
own stream, proved by C-Flock `prove --zk` behind the Lean firewall with each unit's witness row read off the warden's
root (`warden_rows.bucket_port`), delivered, decided by the tie's verifier of record on its own lanes from the advice
and the warden's release (`IngressTie.outcome(advice=…, releases=…, head=…)`), and `advised.IoTieVerdict` on the link
and the stream's registrations built from the stream's outcomes of record alone (`IngressTie.tie`, read through
`tie_service.Ties`, the advised link's tie source).

    python benchmarks/network_traces/n2_tie_zk.py prove --bin flock-circuit --lean flock-verify --out DIR
      [--class test|served|warden] [--warden-run RUN] [--windows K] [--gpu] [--jobs 1] [--debug-clear] [--dry]

* **The link** (`PARAMS`, the pinned size class): a span, three id requests of 5, 40 and 5 prompt tokens
  served through the warden's own ingress (`io_request.serve`) in two windows of T = 4 buckets of two lanes of r = 2
  512-byte cells, request 1's `DATA` in bucket 3 and its `END` in bucket 4, so it spans the windows. The advice is v0
  of one prefill step, each request one kept output, with the link's `received`; it and the warden's release are
  public, and the cells and their salts are witness. ``--class served`` (`CLASSES`) is the served link's ingress
  instead, `n2_advised_zk`'s: one window of T = 60 buckets of one lane of r = 1 4,096-byte cell, the unit
  `IoTieRequest_v1{B=4096,L=1,R=1}`, with the ``honest`` and ``forged`` members alone.
* **A warden's run** (``--class warden --warden-run RUN``, `warden_link`): the gate's ``warden-rehearse``'s real ingress
  in place of the served link, its first K link-windows (``--windows``, 1 by default), every connection in them
  starting and ending there. The verifier reads ``deployment.json`` (the schedule, its digest, the warden's key and
  session) and each window's registration in ``cells.jsonl``, its head's signature verified under that key; the
  prover's rows are the ingress records under ``cells.salts``, held to that head, as `scan_zk.warden_run` reads them.
  The release is the cells' headers (each connection's first bucket, checked against the warden's wiring, and its
  `DATA` and `END` cells a bucket) and the honest advice's prompt lengths the requests' own (their bytes'
  ``prompt``). The unit is the schedule's, ``IoTieRequest_v1{B, L, R}``, and the members are ``honest`` and
  ``forged``: the honest advice on rows of which one has a prompt digit changed, a row IoTieRequest accepts but the
  warden never signed, so Lean's registered reads refuse it ("does not open").
* **Members.** One stream each, over the link's two windows: ``honest``, tied in both; ``carried``, bucket 4 proved on
  its lane's carry-in fields + 5 with its carry-out's + 5, which IoTieRequest accepts but whose
  ``pub`` row is no public row of the verifier's, so Lean's registered read of it refuses window 1; ``forged``, an
  advice whose request 1 is 45 prompt tokens in the same cells, which IoTieRequest refuses in both sampling blocks, so
  its output word 0 fails the registered read of ``o0``; ``split``, prompts of 40 and 5 tokens in cells the advice
  swaps, proved on the cells' own lanes, which the verifier refuses in every window, the release not being the
  swapped advice's schedule, so it has no lanes to register (`EXPECT`).
* **The statement and the rounds** follow `warden_zk`'s: one circuit, `IngressTie.proof_unit` lowered once (the gate
  the Definition its pinned digest names), and each link-window a statement of T instances, unit t bucket t, on rows
  ``pub`` and ``wit`` (`IngressTie.layout`) and the output row ``o0``, the verdict bit: a shared-row file every port
  of which is a registered read, ``wit`` the window's bucket rows under the warden's salts and paths, and ``pub`` and
  ``o0`` the prover's rows under the known values' public salts (`known`). A round registers, selects, proves live
  behind the firewall, delivers, and decides on the proof: the verifier's own Lean run with its own registered reads
  (`IngressTie.registered`: the warden's root, and at known content its own lanes from the advice and the release
  and the output word 1), each ``pub`` row opened to its own copy and each ``o0`` row read. Per window: prove (the
  session's wall time, the prover's ``prove_total_s``, proof bytes) and verify (the verifier of record's run, Lean
  included). The members run one after another, one firewalled session at a time (`n2_zk.ONE_SESSION`: ``--jobs`` is
  1).
* **What the verifier receives** is `RECEIVED` per window and the advice (``advice.json``), scanned for every piece of
  the windows' held cells and their salts but those the public circuit or an empty slot holds, and a member fails on a
  piece `leak_scan.scan` reads as ``leaked``, not on a find it reads as structured (across the format's fields, or
  bytes the statement fixes), the rule every zero-knowledge driver's scan follows; the prover's instance files, which
  hold the cells, are the control.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import certifier_serve as cs  # noqa: E402
import leak_scan as LS  # noqa: E402
import n2_zk as Z  # noqa: E402
from verity.core.protocols.certifier import advice as ADV, advised as AD, head_check as HC  # noqa: E402
from verity.core.protocols.certifier import io_request as Q, service as SVC  # noqa: E402
from verity.core.protocols.certifier import tie_service as TS, warden as W, warden_rows as WR  # noqa: E402
from verity.core.protocols.certifier.active import DATA, HEADER, ID_DIGITS, id_request  # noqa: E402
from verity.core.protocols.certifier.certificate import Direction  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import partition as P  # noqa: E402

B = 512
#: two lanes of r = 2 slots, 4-bucket windows: IoTieRequest_v1{B=512,L=2,R=2}, the pinned size class
PARAMS = W.WardenParams(cell_bytes=B, slots=2, lanes=2, window=4, delay=1)
T = PARAMS.window
LINK = "requests"
SESSION = hashlib.sha512(b"n2-tie-zk").digest()
#: the span: request 1 (40 tokens) takes 2 cells, its DATA in bucket 3 and its END in bucket 4
SPAN, ARRIVALS, RATES = (5, 40, 5), (0, 2, 4), (10 ** 6, 600, 10 ** 6)
#: the forged advice: request 1 45 tokens, in the same cells
FORGED = (5, 45, 5)
#: the split's prompts, whose cells the split advice swaps
SPLIT, SPLIT_ARRIVALS, SPLIT_RATES = (40, 5), (0, 0), (600, 10 ** 6)
#: what the carried member adds to bucket 4's lane 0 carry-in fields and carry-out fields
CARRIED = 5
PROFILE = AD.AdvisedProfile(2, TS.framing(B, 64), 0)
MEMBERS = ("honest", "carried", "forged", "split")
#: each member's windows' outcomes of record, which are its tie's
EXPECT = {"honest": (True, True), "carried": (True, False), "forged": (False, False), "split": (False, False)}
#: what the verifier receives per link-window, by file (`member`)
def _received(T: int) -> dict[str, str]:
    return {"registration.json": "the registration record (the warden's salted root)", "draw.json": "the stream's draw",
            f"pub-{T}.bin": "the statement's public file",
            "registered.json": "the verifier's own registered reads, null where it has no lanes",
            "session-*": "the session's files",
            "delivered.json": "each unit's pub and o0 rows' salts and its output word",
            "outcome.json": "the outcome record", "advice.json": "the link's advice and the warden's release"}


RECEIVED = _received(T)
WINDOWS = 2
#: the size classes `use` sets: ``test``, the pinned test class above with every member; ``served``, the served link's
#: ingress (`n2_advised_zk`: B = 4,096, T = 60 buckets a window, an allocation of one cell, IoTieRequest_v1{B=4096,L=1,
#: R=1}), one window, honest and forged
CLASSES = {
    "test": {"B": B, "params": PARAMS, "windows": WINDOWS, "expect": EXPECT},
    "served": {"B": 4096, "params": W.WardenParams(cell_bytes=4096, slots=1, lanes=1, window=60, delay=1),
               "windows": 1, "expect": {"honest": (True,), "forged": (False,)}},
    "warden": None,
}
#: the warden run ``--class warden`` reads (`warden_link`), None at the other classes
REAL = None


def use(name: str, run: Path | None = None, windows: int = 1) -> None:
    """Set the driver's size class to `CLASSES` [name]: its B, warden parameters, windows, and members with their
    verdicts; ``warden`` reads them off the warden run `run`'s first `windows` link-windows (`warden_link`), the
    forged member's window 0 refused and the rest tied."""
    global B, PARAMS, T, PROFILE, MEMBERS, EXPECT, RECEIVED, WINDOWS, LINK, SESSION, REAL
    if name == "warden":
        if run is None or windows < 1:
            raise SystemExit("--class warden reads a warden run's first link-windows: --warden-run RUN [--windows K]")
        REAL = warden_link(Path(run), windows)
        B, PARAMS, LINK, SESSION, WINDOWS = REAL.params.cell_bytes, REAL.params, REAL.link, REAL.session, windows
        EXPECT = {"honest": (True,) * windows, "forged": (False,) + (True,) * (windows - 1)}
    else:
        c, REAL = CLASSES[name], None
        B, PARAMS, WINDOWS, EXPECT = c["B"], c["params"], c["windows"], dict(c["expect"])
    T, MEMBERS = PARAMS.window, tuple(EXPECT)
    PROFILE = AD.AdvisedProfile(WINDOWS, TS.framing(B, max((64, *(REAL.prompt_lens if REAL else ())))), 0)
    RECEIVED = _received(T)


def log(msg: str) -> None:
    print(f"N2TIE {time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


# --- the link and its members -----------------------------------------------------------------------------------------

def requests(prompt_lens) -> list[bytes]:
    return [id_request(model="glm", user=f"u{i}", prompt=[(7919 * i + 104729 * k) % 151329 for k in range(n)],
                       max_tokens=16, temperature=0.7, top_p=0.95, seed=i, frame_bytes=B, offset=0)
            for i, n in enumerate(prompt_lens)]


def advice(prompt_lens, got) -> AD.LinkAdvice:
    """An advice v0 of one prefill step, each request one kept output, with the link's `received`."""
    reqs = tuple(ADV.RequestAdvice(f"u{j}", n, 1, "eos") for j, n in enumerate(prompt_lens))
    steps = (ADV.StepAdvice(tuple(ADV.Row(j, "prefill", 0, n) for j, n in enumerate(prompt_lens))),)
    return AD.LinkAdvice(ADV.Advice(reqs, steps, tuple(range(len(prompt_lens)))), tuple(got))


def served(prompt_lens, arrivals, rates) -> Q.Link:
    """The requests through the warden's own ingress over the class's windows."""
    return Q.serve(PARAMS, requests(prompt_lens), arrivals, rates, buckets=WINDOWS * T)


def received(link: Q.Link) -> tuple[int, ...]:
    cells = TS.window_cells(link.releases, T)
    return tuple(cells.get(w, 0) for w in range(len(link.records)))


def _found(run: Path, name: str) -> Path:
    """A warden run's file: in the run's directory (`warden_serve.py serve --out`), else its ``witness/`` (the gate's
    ``warden-rehearse``)."""
    for d in (run, run / "witness"):
        if (d / name).is_file():
            return d / name
    raise SystemExit(f"--warden-run {run}: no {name} there or in its witness/")


def warden_link(run: Path, windows: int) -> SimpleNamespace:
    """The warden run's ingress link-windows 0 … windows − 1 (the module docstring, "A warden's run"): each window's
    rows under the warden's salts, held to the head the verifier read off its signed registration, the windows'
    release and their requests' prompt lengths, refused by name where any of it doesn't hold."""
    import warden_serve as WS

    dep = json.loads(_found(run, "deployment.json").read_text())
    params = W.WardenParams(**dep["params"])
    if params.digest.hex() != dep["digest"]:
        raise SystemExit(f"--warden-run {run}: deployment.json's digest isn't its parameters'")
    key, session = WS.cs.read_public(dep["key"]), bytes.fromhex(dep["session"])
    records = {r.window: r for r in (WS.ar.unpack_record(b, params.cell_bytes)
                                     for b in WS.ar.read_blobs(_found(run, f"{WS.INGRESS}.records")))}
    lines = [json.loads(x) for x in _found(run, "cells.jsonl").read_text().splitlines()]
    salts = WS.ar.read_blobs(_found(run, "cells.salts"))
    if len(salts) != len(lines):
        raise SystemExit(f"--warden-run {run}: cells.jsonl has {len(lines)} lines and cells.salts {len(salts)} blobs")
    regs = {doc["registration"]["window"]: (doc["registration"], blob) for doc, blob in zip(lines, salts)
            if "registration" in doc and (doc["registration"]["link"], doc["registration"]["direction"])
            == (WS.LINK, Direction.INGRESS.value)}
    cells = []
    for w in range(windows):
        if w not in regs or w not in records:
            raise SystemExit(f"--warden-run {run}: ingress link-window {w} has no registration or no record")
        registration, blob = regs[w]
        try:
            head, _dom = WR.check_registration(registration, key, session, params, WS.LINK, Direction.INGRESS, w)
        except ValueError as e:
            raise SystemExit(f"--warden-run {run}: ingress link-window {w}'s registration doesn't verify under the "
                             f"deployment's key and session: {e}")
        row_salts, wiring_salt = WR.read_witness(blob)
        c = WR.Cells.of(WS.LINK, Direction.INGRESS, records[w], params, session, row_salts, wiring_salt)
        if c.head != head:
            raise SystemExit(f"--warden-run {run}: ingress link-window {w}'s record under cells.salts isn't the rows "
                             "whose root the warden signed")
        cells.append(c)
    releases, lens = _released(params, cells)
    return SimpleNamespace(source=f"warden run {dep.get('run_id')} (warden_serve.py, its deployment's key)",
                           run_id=dep.get("run_id"), params=params, session=session, link=WS.LINK, cells=cells,
                           records=tuple(records[w] for w in range(windows)), releases=releases, prompt_lens=lens)


def _released(params: W.WardenParams, cells: list) -> tuple[list[Q.Release], tuple[int, ...]]:
    """The windows' release read off their cells' headers, each connection's first bucket and its request cells a
    bucket to its `END`, its first bucket the warden's wiring's; and each request's prompt length, its bytes'
    ``prompt``. Refused where a connection's ingress runs past the windows or the ordinals aren't 1, 2, …"""
    B_, T_, seen = params.cell_bytes, params.window, {}
    for c in cells:
        for t, row in enumerate(c.rows):
            for s in range(0, len(row), B_):
                kind, q, n = HEADER.unpack_from(row, s)
                if row[s:s + B_] == bytes(B_):
                    continue
                d = seen.setdefault(q, {"first": c.window * T_ + t, "cells": {}, "bytes": b"", "end": None})
                if kind in (DATA, W.END):
                    b = c.window * T_ + t
                    d["cells"][b] = d["cells"].get(b, 0) + 1
                    d["bytes"] += row[s + HEADER.size:s + HEADER.size + n]
                    d["end"] = b if kind == W.END else d["end"]
        for q, first, _n in c.wiring.connections:
            if q in seen and seen[q]["first"] // T_ == c.window and seen[q]["first"] != c.window * T_ + first:
                raise SystemExit(f"connection {q}'s first cell is in bucket {seen[q]['first']}, and the warden's "
                                 f"wiring says {c.window * T_ + first}")
    if sorted(seen) != list(range(1, len(seen) + 1)):
        raise SystemExit(f"the windows' connections are {sorted(seen)}, not ordinals 1 … {len(seen)}")
    if past := [q for q, d in seen.items() if d["end"] is None]:
        raise SystemExit(f"connections {past} don't end their ingress in link-windows 0 … {len(cells) - 1}: "
                         "take a --windows they end in")
    releases, lens = [], []
    for q in sorted(seen):
        d = seen[q]
        releases.append(Q.Release(q, d["first"], tuple(d["cells"].get(b, 0) for b in range(d["first"], d["end"] + 1))))
        try:
            lens.append(len(json.loads(d["bytes"].partition(b"\r\n\r\n")[2])["prompt"]))
        except (ValueError, KeyError, TypeError) as e:
            raise SystemExit(f"connection {q}'s request isn't an id request: {e!r}")
    return releases, tuple(lens)


class Forge(NamedTuple):
    """A prover's row in place of the warden's: link-window `window`'s bucket `bucket`, byte `at` changed."""

    window: int
    bucket: int
    at: int
    row: bytes


#: an id request's prompt field (`active.ID_FIELD_BYTES`): `[` or `,`, then the id right-aligned in ten characters
PROMPT_FIELD = re.compile(rb"[\[,] *[0-9]+")


def forge(link: SimpleNamespace) -> Forge:
    """Window 0's first request cell's first prompt field whole in it, its id's last digit changed: an id request of
    the same prompt length, so IoTieRequest accepts the row, but not the row the warden signed."""
    c = link.cells[0]
    for t, row in enumerate(c.rows):
        for s in range(0, len(row), B):
            kind, _q, n = HEADER.unpack_from(row, s)
            if kind not in (DATA, W.END) or row[s:s + B] == bytes(B):
                continue
            lo = s + HEADER.size
            for m in PROMPT_FIELD.finditer(row, lo, lo + n):
                if m.end() - m.start() == 1 + ID_DIGITS:
                    k, x = m.end() - 1, bytearray(row)
                    x[k] = ord("0") + (x[k] - ord("0") + 1) % 10
                    return Forge(0, t, k, bytes(x))
    raise SystemExit("link-window 0 holds no whole prompt field to forge")


def _carried(w: int, lanes):
    if w != 1:
        return lanes
    ln = lanes[0][0]
    return ((ln._replace(fields=ln.fields + CARRIED, fields_out=ln.fields_out + CARRIED),) + lanes[0][1:],) + lanes[1:]


def members() -> dict[str, tuple]:
    """Each member's (served link, its advice, the prover's lanes as an edit of the verifier's, the prompt lengths it
    proves on or its rows' `Forge`, what was changed)."""
    if REAL is not None:
        honest, f = advice(REAL.prompt_lens, received(REAL)), forge(REAL)
        plan = {"honest": (REAL, honest, None, {}),
                "forged": (REAL, honest, f, {"window": f.window, "bucket": f.bucket, "slot": f.at // B,
                                             "byte": f.at % B})}
        return {name: plan[name] for name in MEMBERS}
    link, split = served(SPAN, ARRIVALS, RATES), served(SPLIT, SPLIT_ARRIVALS, SPLIT_RATES)
    honest = advice(SPAN, received(link))
    plan = {"honest": (link, honest, None, {}),
            "carried": (link, honest, _carried, {"bucket": T, "added_fields": CARRIED}),
            "forged": (link, advice(FORGED, received(link)), None, {"prompt_lens": list(FORGED)}),
            "split": (split, advice(SPLIT[::-1], received(split)), SPLIT, {"prompt_lens": list(SPLIT[::-1]),
                                                                          "certified": list(SPLIT)})}
    return {name: plan[name] for name in MEMBERS}


def link_windows(tie: TS.IngressTie, link: Q.Link, adv: AD.LinkAdvice, edit=None) -> SimpleNamespace:
    """Every link-window of the member as the warden and the prover hold it before it registers: the warden's cells
    (`warden_rows.Cells`: a warden run's own, else fresh salts) and its head, the verifier's lanes of `adv` (``own``,
    None where the release isn't its schedule) and the prover's (``lanes``: `edit` of the verifier's, or, a tuple, the
    lanes of those prompt lengths), the prover's rows (the warden's, or a `Forge`'s), each unit's arguments on the
    prover's lanes and rows and the verifier's public arguments on its own, each output word by the gate's reference,
    whether each unit's public row is the verifier's (``opens``) and the host's read of the registered rows (``held``,
    `warden_rows.refusal`: None where every row opens the warden's root)."""
    out = []
    for w, rec in enumerate(link.records):
        cells = link.cells[w] if getattr(link, "cells", None) else \
            WR.Cells.of(LINK, Direction.INGRESS, rec, PARAMS, SESSION)
        own = tie.lanes(adv, link.releases, w)
        if isinstance(edit, Forge):
            lanes = own
        elif isinstance(edit, tuple):
            lanes = tie.lanes(advice(edit, adv.received), link.releases, w)
        else:
            lanes = own if edit is None or own is None else edit(w, own)
        if lanes is None:
            raise SystemExit(f"window {w}: the prover has no lanes of the window to prove on")
        rows = list(cells.rows)
        if isinstance(edit, Forge) and edit.window == w:
            rows[edit.bucket] = edit.row
        args = [tie.arguments(r, ln) for r, ln in zip(rows, lanes)]
        mine = [None] * T if own is None else [tie.public_arguments(ln) for ln in own]
        if any(a is None for a in args):
            raise SystemExit(f"window {w}: the unit takes no instance of the prover's lanes")
        words = [tie.outputs(r, ln) for r, ln in zip(rows, lanes)]
        opens = [m is not None and tie.row(m, "pub") == tie.row(a, "pub") for a, m in zip(args, mine)]
        _words, salts, entry = cells.statement(range(T))
        held = WR.refusal(WR.bucket_port(cells.head, PARAMS), tie.domain(w), entry,
                          [WR.bc(r, s) for r, s in zip(rows, salts)])
        out.append(SimpleNamespace(window=w, cells=cells, head=cells.head, own=own, lanes=lanes, rows=rows, args=args,
                                   mine=mine, words=words, opens=opens, held=held,
                                   expect=int(held is None and all(x == 1 for x in words) and all(opens)),
                                   statement=None))
    return SimpleNamespace(windows=out, advice=adv, releases=tuple(link.releases))


# --- the statement ----------------------------------------------------------------------------------------------------

def stager(M, tie: TS.IngressTie, out: Path) -> SimpleNamespace:
    """The one circuit: `IngressTie.proof_unit` lowered once on rows ``pub`` and ``wit`` with output row ``o0``, bound
    to the registered partition and Program, which are written beside it as the verifier's own."""
    rows = [(r, M.V2, tie.bits(r), list(tie.layout[r])) for r in TS.ROWS]
    out.mkdir(parents=True, exist_ok=True)
    (out / "partition.json").write_text(json.dumps(tie.partition, sort_keys=True))
    (out / "program.json").write_bytes(tie.program)
    bound = {"partition": M.P.digest(tie.partition), "program": tie.partition["program"]}
    t, was = time.time(), gc.isenabled()
    gc.disable()
    try:
        st = M.CS.UnitStager(tie.proof_unit, "unit/ingress-tie-bucket", rows, [(M.V2, (0,))],
                             circuit=out / "circuit.txt", bind=bound)
        st.pin
    finally:
        if was:
            gc.enable()
    got = [(p.name, p.n_bits) for p in st.ports]
    if got != [(p, tie.bits(p)) for p in TS.PORTS] or st.out_words != [1]:
        raise SystemExit(f"the statement's rows {got} are not the ones the verifier opens "
                         f"{[(p, tie.bits(p)) for p in TS.PORTS]}")
    (_tid, base, cnt), = M.P.ranges(tie._program, tie.query)
    if (base, cnt) != (0, T):
        raise SystemExit(f"the registered Program's units are {base} .. {base + cnt}, not the window's 0 .. {T}")
    k_log = st.meta["k_log"]
    nbl = max(3, (-(-T // st.meta["vus_per_block"]) - 1).bit_length())
    if not 25 <= k_log + nbl <= 35:
        raise SystemExit(f"m = {k_log} + {nbl} is outside 25 .. 35 (--zk and flock-circuit's shape)")
    return SimpleNamespace(st=st, out=out, circuit=out / "circuit.txt", partition_file=out / "partition.json",
                           program_file=out / "program.json", lower_s=round(time.time() - t, 1), k_log=k_log, nbl=nbl,
                           circuit_sha512=M.PF.file_sha512(out / "circuit.txt"))


def known(tie: TS.IngressTie, x: SimpleNamespace) -> dict:
    """The prover's ``pub`` and ``o0`` rows of window `x`, its lanes' public rows and its output words, committed as the
    verifier registers its own at known content (`IngressTie.known_tree`): each root is the verifier's exactly where
    the rows are, and a row that isn't fails its port's registered read."""
    return {"pub": tie.known_tree(x.head, "pub", [tie.row(a, "pub") for a in x.args]),
            "o0": tie.known_tree(x.head, "o0", [SVC.output_row(w) for w in x.words])}


def stage(M, sg: SimpleNamespace, tie: TS.IngressTie, rounds: dict[str, SimpleNamespace], out: Path) -> dict:
    """Every link-window's statement on the one circuit, T instances each, unit t bucket t (`UnitStager.stage`: the
    outputs the lowering's, held to the IR reference): a shared-row file every port of which is a registered read, its
    ``wit`` table the window's bucket rows under the warden's salts and paths (`warden_rows.Cells.statement`), and its
    ``pub`` and ``o0`` rows under the prover's known values' public salts and paths (`known`). Each staged ``o0`` row
    must open to its output word, and each ``pub`` row to the verifier's own copy exactly where the prover's lanes are
    the verifier's. Beside it, the verifier's own registered reads (`IngressTie.registered`, null where it has no
    lanes)."""
    np, st, t = M.np, sg.st, time.time()
    ands = None
    for name, rd in rounds.items():
        for x in rd.windows:
            d = out / name / f"w{x.window:03d}" / "statement"
            rows = {r: np.stack([np.frombuffer(tie.row(a, r), "<u2") for a in x.args]) for r in TS.ROWS}
            _words, salts, entry = x.cells.statement(range(T))
            trees = known(tie, x)
            public = {p: list(v.salts) for p, v in trees.items()}
            got = st.stage(rows, d, os.urandom(32), salts={"wit": salts, **public}, indices=range(T),
                           set_name=f"{st.iset.name}-{name}-w{x.window}", tables=rows,
                           refs={r: list(range(T)) for r in TS.ROWS},
                           registered={TS.REGISTERED: entry, **{p: v.statement(range(T))[2] for p, v in trees.items()}})
            ands = got["ands"]
            words = [int(v) for v in np.asarray(got["outs"]["o0"])[:, 0]]
            if words != x.words:
                raise SystemExit(f"{name} window {x.window}: the lowering's output words {words} are not the prover's "
                                 f"references {x.words}")
            (d / "inst.bin").rename(d / f"inst-{T}.bin")
            (d / "pub.bin").rename(d / f"pub-{T}.bin")
            (d / "tables").mkdir(exist_ok=True)
            os.link(sg.circuit, d / "circuit.txt")
            (d / "registered.json").write_text(json.dumps(tie.registered(x.head, x.own), sort_keys=True))
            for u in range(T):
                bcs = SVC._commitments(d / f"pub-{T}.bin", u)
                mine = None if x.mine[u] is None else tie.row(x.mine[u], "pub")
                pub = bool(bcs) and mine is not None and SVC._opens(mine, tie.bits("pub"), public["pub"][u], bcs["pub"])
                if not (bcs and pub == x.opens[u]
                        and SVC._opens(SVC.output_row(words[u]), SVC.OUT_BITS, public["o0"][u], bcs["o0"])):
                    raise SystemExit(f"{name} window {x.window} unit {u}: the staged rows don't open as the prover's "
                                     "lanes say")
            x.statement = SimpleNamespace(dir=d, salts={u: {r: public[r][u] for r in public} for u in range(T)},
                                          outputs=dict(enumerate(words)), private=salts)
    return {"statements": sum(len(rd.windows) for rd in rounds.values()), "instances": T, "ands": ands,
            "stage_s": round(time.time() - t, 1)}


# --- the rounds -------------------------------------------------------------------------------------------------------

def member(M, a, sg: SimpleNamespace, tie: TS.IngressTie, name: str, rd: SimpleNamespace, out: Path,
           port: int) -> dict:
    """The member's link-windows on its own stream, in order: the warden registers, the challenger selects, the prover
    proves the statement live behind the firewall and delivers, and the verifier of record decides on the proof
    (`IngressTie.outcome`); then the tie from the stream's outcomes of record (`IngressTie.tie`). Writes what the
    verifier receives under ``out / "verifier"`` (`RECEIVED`), each window's files prefixed by its window."""
    stream = tie.stream(budget=len(rd.windows), delta=cs.DELTA, challenger="n2-tie-zk", coins=cs._no_coins,
                        clock=time.monotonic)
    # the firewall rechecks a witness the host computed, so a device prover takes the host's (flock-circuit's
    # FIREWALL-REFUSED otherwise), as a8's fw arm does
    env = {**Z.firewall(M, sg), **({"FC_HOST_WITNESS": "1"} if a.gpu else {})}
    av = SimpleNamespace(bin=a.bin, lease=[], gpu=a.gpu)
    vdir = out / "verifier"
    vdir.mkdir(parents=True, exist_ok=True)
    digest = AD.digest(rd.advice)
    (vdir / "advice.json").write_text(json.dumps({"advice": ADV.to_json(rd.advice.advice),
                                                  "received": list(rd.advice.received), "digest": digest,
                                                  "releases": [list(r) for r in rd.releases]}, sort_keys=True))
    previous, rows = None, []
    for j, x in enumerate(rd.windows):
        wdir, pre = out / f"w{x.window:03d}", f"w{x.window:03d}-"
        reg = tie.register(stream, tie.registration(x.head, previous))
        previous = reg.digest
        if reg.record["roots"][0]["root"] != x.head.root.hex():
            raise SystemExit(f"{name} window {x.window}: the registered root is not the warden's")
        draw = stream.select(reg)
        if list(draw.units) != list(range(T)):
            raise SystemExit(f"{name} window {x.window}: the stream drew {list(draw.units)}, not the window's "
                             f"{T} units")
        sdir = wdir / "session"
        m = {"dir": x.statement.dir, "n": T, "pin": sg.st.pin}
        record, live, proc = M.a8.session(av, m, sdir, list(range(T)), port + j, env=env)
        sessions = tuple(sorted(p.parent for p in (sdir / "sessions").glob("*/session.json")))
        delivered = TS.Delivered(M.PF.Proof(sg.circuit, x.statement.dir / f"pub-{T}.bin", sessions),
                                 x.statement.salts, x.statement.outputs)
        late = stream.deliver(draw, delivered.sessions())
        t = time.time()
        outcome = late or tie.outcome(stream, draw, delivered, circuit_sha512=sg.circuit_sha512, advice=rd.advice,
                                      releases=rd.releases, head=x.head, opening=x.cells if a.debug_clear else None)
        verify_s = round(time.time() - t, 1)
        body = stream.records[-1]
        verdict = (body.get("audit") or {}).get("verdicts", {}).get(TS.VERIFIER) or {}
        findings, refused = Z.LEAN_RUNS.get(str(delivered.proof.public), (None, "no Lean run"))
        opened = [TS._opened(delivered, u, None if x.mine[u] is None else tie.row(x.mine[u], "pub"), tie.bits("pub"))
                  for u in range(T)]
        proof = {"lean_accepted": refused is None and bool(findings) and all(f.accepted for f in findings),
                 "refused": refused, "proved": refused is None and tie.proved(findings or [])
                 and all(list(f.units) == list(range(T)) for f in findings), "opened_words": opened}
        files = {"registration.json": reg.record, "draw.json": dict(draw.draw), "outcome.json": body,
                 "delivered.json": {"salts": {u: {r: s.hex() for r, s in v.items()}
                                              for u, v in x.statement.salts.items()},
                                    "outputs": x.statement.outputs}}
        for f, v in files.items():
            (vdir / f"{pre}{f}").write_text(json.dumps(v, indent=1, sort_keys=True, default=str))
        for f in (f"pub-{T}.bin", "registered.json"):
            os.link(x.statement.dir / f, vdir / f"{pre}{f}")
        for f in sorted((sdir / "sessions").rglob("*")):
            if f.is_file():
                os.link(f, vdir / f"{pre}session-{f.parent.name}-{f.name}")
        proofs = live.get("proof_bytes") or []
        row = {"window": x.window, "registration": reg.digest, "accepted": outcome.accepted,
               "public": verdict.get("public"), "service_public": dict(outcome.public), "late": late is not None,
               "decided_by": verdict.get("decided_by"), "proof": proof, "expect": x.expect, "words": x.words,
               "opens": x.opens,
               "prove": {"session_s": proc["wall_s"], "prove_total_s": live.get("prove_total_s"),
                         "e2e_s": live.get("e2e_s"), "proof_bytes": sum(proofs) if proofs else None,
                         "k_log": live.get("k_log"), "prover_accepted": live.get("accepted"),
                         "max_child_rss_gb": proc["max_child_rss_gb"], "cpu_s": proc["cpu_s"]},
               "verify_s": verify_s, "server_verdict": ((record or {}).get("verdict") or {}).get("accepted"),
               **({"clear": verdict["clear"]} if "clear" in verdict else {})}
        # every row is a registered read (the warden's root, the verifier's lanes, the output word 1), so a window the
        # reference refuses is Lean's to refuse, whatever the unit says of it
        held = (proof["proved"] and opened == x.words) if x.expect else not proof["lean_accepted"]
        row["registered_read"] = x.held
        row["as_expected"] = (row["accepted"] == bool(x.expect) and not row["late"] and row["decided_by"] == "proof"
                              and held and row["public"] == {"verdict": x.expect, "advice": digest})
        rows.append(row)
        log(f"{name} window {x.window}: {'accept' if outcome.accepted else 'reject'} public {row['public']}"
            f"{' LATE' if late else ''}, Lean {'proved' if proof['proved'] else 'NOT proved'}"
            f"{'' if refused is None else f' (refused: {str(refused)[:300]})'}, output words "
            f"{opened}, prove {row['prove']['session_s']} s (prove_total {row['prove']['prove_total_s']}"
            f" s, {row['prove']['proof_bytes']} B), verify {verify_s} s; "
            f"{'as expected' if row['as_expected'] else 'UNEXPECTED'}")
    tied = TS.Ties(tie, stream, rd.releases).verdict(tie.link, rd.advice, PROFILE)
    return {"windows": rows, "tie": {"link": tied.link, "registrations": list(tied.registrations),
                                     "advice": tied.advice, "accepted": list(tied.accepted)}}


def blobs_of(rd: SimpleNamespace) -> tuple[dict[str, bytes], set[str]]:
    """The witness pieces the verifier must not receive: each window's held cells (every slot a connection holds, its
    pads included: a pad names its connection's ordinal) in the prover's rows, which its instance files hold as they
    are, and each bucket row's salt. An empty slot, B zero bytes, is public."""
    blobs, held, empty = {}, set(), bytes(B)
    for x in rd.windows:
        for t, row in enumerate(x.rows):
            for i in range(0, len(row), B):
                cell = row[i:i + B]
                if cell != empty:
                    key = f"cell {i // B} t{t} w{x.window}"
                    blobs[key] = cell
                    held.add(key)
        for t, s in enumerate(x.cells.salts):
            blobs[f"salt {t} w{x.window}"] = s
    return blobs, held


def _member_scan(M, sg, rd, out: Path) -> dict:
    public = sg.circuit.read_bytes() + bytes(B)
    blobs, held, excluded = Z.public_pieces(M.a8, *blobs_of(rd), public)
    control = out / "prover-instances.bin"
    with open(control, "wb") as f:
        for x in rd.windows:
            f.write((x.statement.dir / f"inst-{T}.bin").read_bytes())
    got = M.a8.received(out / "verifier", control, blobs, held)
    return {**got, **LS.scan(got, blobs, out / "verifier"), "public_pieces": excluded, "files": len(got["files"])}


def _member_ok(name: str, res: dict, rd) -> bool:
    """Whether member `name`'s run is what `EXPECT` says: every window's verdict its reference's, the tie's on the link
    under its advice with one registration per window, no witness piece `leak_scan.scan` reads as leaked in what the
    verifier received, and the control holding every held cell."""
    rec = res["received"]
    return (all(r["as_expected"] for r in res["windows"]) and tuple(res["tie"]["accepted"]) == EXPECT[name]
            and res["tie"]["advice"] == AD.digest(rd.advice) and res["tie"]["link"] == LINK
            and len(res["tie"]["registrations"]) == len(rd.windows)
            and not rec["leaked"] and rec["control"]["holds_all"])


def prove(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    tie = TS.IngressTie(LINK, PARAMS, SESSION, proof_deadline=cs.PROOF_DEADLINE_S)
    plan = members()
    rounds = {name: link_windows(tie, link, adv, edit) for name, (link, adv, edit, _) in plan.items()}
    summary: dict = {"class": a.cls, "gpu": a.gpu,
                     **({"source": REAL.source, "warden_run": REAL.run_id, "windows": WINDOWS,
                         "prompt_lens": list(REAL.prompt_lens), "releases": [list(r) for r in REAL.releases]}
                        if REAL else {}),
                     "link": {"name": LINK, "shape": list(tie.key), "T": T, "unit": tie.unit.id,
                              "program": tie.partition["program"], "partition": P.digest(tie.partition),
                              "gates": [g.id for g in tie.gates()], "pinned": all(HC.pinned(g) for g in tie.gates()),
                              "bits": {p: tie.bits(p) for p in TS.PORTS}},
                     "changed": {name: what for name, (_, _, _, what) in plan.items()},
                     "advice": {name: AD.digest(rd.advice) for name, rd in rounds.items()},
                     "debug_clear": a.debug_clear, "received": RECEIVED}
    summary["instances_s"] = round(time.time() - t, 1)
    for name, rd in rounds.items():
        summary.setdefault("members", {})[name] = {
            "reference": [{"window": x.window, "words": x.words, "opens": x.opens, "expect": x.expect}
                          for x in rd.windows]}
        log(f"{name}: words {[x.words for x in rd.windows]}, pub rows the verifier's "
            f"{[x.opens for x in rd.windows]}, expected verdicts {[x.expect for x in rd.windows]}")
        if tuple(bool(x.expect) for x in rd.windows) != EXPECT[name]:
            raise SystemExit(f"{name}: the references' verdicts aren't the member's {EXPECT[name]}")
    if not summary["link"]["pinned"]:
        raise SystemExit(f"the gate names no pinned circuit digest: {summary['link']['gates']}")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    if a.dry:
        log("dry: done")
        return 0

    M = Z._one_stage()
    M.PF.executable = lambda: Path(a.lean)
    Z._record_lean_runs(M.PF)
    sg = stager(M, tie, out / "stager")
    summary["stager"] = {"unit": tie.proof_unit.id, "lower_s": sg.lower_s, "k_log": sg.k_log, "nbl": sg.nbl,
                         "pin": sg.st.pin, "circuit_sha512": sg.circuit_sha512,
                         "ands": sg.st.low.layout.counts["ands"]}
    log(f"stager: {tie.proof_unit.id}, {sg.lower_s} s, k_log {sg.k_log} + {sg.nbl}, {summary['stager']['ands']} ANDs, "
        f"pin {sg.st.pin[:16]}")
    summary["staging"] = stage(M, sg, tie, rounds, out)
    log(f"staged: {summary['staging']}")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))

    def go(k: int, name: str) -> tuple[str, dict]:
        res = member(M, a, sg, tie, name, rounds[name], out / name, a.port + 100 * k)
        res["received"] = _member_scan(M, sg, rounds[name], out / name)
        return name, res

    done = dict(go(k, name) for k, name in enumerate(rounds))
    ok = True
    for name, res in done.items():
        rec = res["received"]
        good = _member_ok(name, res, rounds[name])
        ok &= good
        summary["members"][name].update(res, ok=good)
        log(f"member {name}: tie {res['tie']['accepted']}, windows {[int(r['accepted']) for r in res['windows']]}, "
            f"verifier received {rec['verifier_bytes']} bytes in {rec['files']} files, witness pieces found there: "
            f"{rec['verifier_found'] or 'none'} (of {rec['blobs']} blobs; {len(rec['public_pieces'])} public pieces "
            f"left out), leaked {rec['leaked'] or 'none'}, structured {rec['structured'] or 'none'}, "
            f"control {rec['control']['blobs_found']}; {'as expected' if good else 'UNEXPECTED'}")
    summary["tie"] = {name: res["tie"]["accepted"] for name, res in done.items()}
    costs = [r for res in done.values() for r in res["windows"]]
    summary["per_window"] = {
        "of": f"each of the {len(costs)} link-windows, its {T} bucket units proved together and decided by the "
              "verifier of record",
        **{k: [r["prove"][k] for r in costs] for k in ("session_s", "prove_total_s", "e2e_s", "proof_bytes",
                                                       "max_child_rss_gb")},
        "verify_s": [r["verify_s"] for r in costs]}
    log(f"tie {summary['tie']}; per window: {summary['per_window']}")
    summary["ok"] = ok
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    log("done: " + ("ok" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="n2_tie_zk.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prove", help="each ingress link-window's bucket rows proved --zk as IoTieRequest, decided by "
                                     "the tie's verifier of record, and the tie built from the outcomes")
    p.add_argument("--bin")
    p.add_argument("--lean")
    p.add_argument("--out", required=True)
    p.add_argument("--class", dest="cls", choices=sorted(CLASSES), default="test", help="the size class (`CLASSES`)")
    p.add_argument("--warden-run", type=Path, help="--class warden: a warden_serve run's directory (the gate's "
                                                   "warden-rehearse's, its witness under witness/)")
    p.add_argument("--windows", type=int, default=1, help="--class warden: its first link-windows proved")
    p.add_argument("--gpu", action="store_true", help="each session's prover on the GPU (`flock-circuit prove --gpu`)")
    p.add_argument("--port", type=int, default=7981)
    p.add_argument("--jobs", type=int, default=1, help="1: the members proved one after another (n2_zk.ONE_SESSION); "
                                                       "more is refused")
    p.add_argument("--debug-clear", action="store_true",
                   help="each verdict also carries the in-the-clear check on the opening, which decides nothing")
    p.add_argument("--dry", action="store_true", help="the instances and their references only: nothing lowered or "
                                                      "proved")
    a = ap.parse_args(argv)
    Z.one_session(a.jobs)
    if not a.dry and not (a.bin and a.lean):
        raise SystemExit("prove needs --bin and --lean, or --dry")
    use(a.cls, a.warden_run, a.windows)
    return prove(a)


if __name__ == "__main__":
    raise SystemExit(main())
