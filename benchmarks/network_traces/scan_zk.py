"""scan-zk: NET-68's step-indexed scan composed, in zero knowledge at w63's class. Ingress bucket t
(`ingress_scan.IngressScanBucket`) reads the warden's ingress row t, the scan's state S_{t−1} and the Programs' row t,
and writes S_t and its egress view; egress bucket t (`egress.EgressBucket`) reads that view and the warden's egress row
t. A link-window's buckets are chained in one registration, every row of every unit a registered read, proved by C-Flock
`prove --zk` behind the Lean firewall and held by Lean `verify --zk --registered`. Standalone: in no Spec.

    python benchmarks/network_traces/scan_zk.py prove --bin flock-circuit --lean flock-verify --out DIR [--dry]

* **The class** (`CLASS`): w63's (B 512, r 4, 4 lanes, C 47, IDB 17, EVERY 10, WITHIN 12) with the warden's delay D 21
  and KO 64 outputs a request; egress at `CLASS.egress` (16 cells in lanes of 4, 4 request slots of 64 17-bit ids).
  The warden at `PARAMS`, 32-bucket link-windows, which the class serves.
* **The run** (`scan_traffic.serve`: the warden's own `_ingress` and `_egress`, ticked as `Warden._tick` ticks them)
  carries five warden-form id requests (`PROMPTS`, `ARRIVALS`) and their responses
  (`RESPONSES`: 3, 40, 64, 8 and 5 ids). The fifth waits in the queue until request 1's egress ends at bucket 21 and
  takes its slot at 22. Link-window 0 of each link is committed by the warden and its head signed by a test-size LMS
  key (`warden_rows.attest`); each registration is read back by the verifier (`warden_rows.check_registration`).
* **The scan's rows**, committed by the prover before the draw as hiding rows (`verity.ml.commitments.advice`, each
  tree's root in the registration): ``state``, S_{−1} (the empty state) … S_{T−1}, leaf t S_{t−1}; ``prog``, the
  Programs' rows, each request's response ids at its anchor (standing in for the anchored Programs' outputs); and
  ``view``, leaf t S_t's egress view.
* **The honest round, one registration**: one Program of two templates, T ingress units then T egress units, every
  unit drawn by the verifier's Lean. Ingress unit t reads ``row`` (the warden's ingress row t), ``sin`` (state leaf t),
  ``prog`` (leaf t), ``sout`` (state leaf t + 1) and ``view`` (leaf t), and outputs ``o0``, its ok bit. Egress unit t
  reads ``state`` (view leaf t) and ``row`` (the warden's egress row t), and outputs ``o0``, its verdict word. So S_t is
  the one leaf unit t writes and unit t + 1 reads, and the view the one leaf ingress t writes and egress t reads: the
  chain is the registered roots', not the prover's word. Accepted: every ok and every verdict opens to 1.
* **Negatives.** ``chain``: the same Program, its ingress units on a state tree with S_a and S_{a+1} swapped (the
  state rows of two buckets): the proofs hold and units a … a + 2's ok bits open to 0, so the window is refused.
  ``forged``: an egress-only Program, one egress row's `DATA` id changed, a row the warden never signed: refused by
  Lean's registered reads ("does not open").
* **What the verifier receives** is scanned for every piece of every unit's rows and every salt but ``o0``'s, which the
  prover sends so the verifier opens each unit's output (`a8.received`), less the pieces public anyway (the circuits,
  zeros, the window's pads and `END` cells) and the sparse pieces (at most one nonzero byte); the prover's instance
  file, which holds them all, is the control.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import leak_scan as LS  # noqa: E402
import warden_zk as WZ  # noqa: E402
from verity.ml.circuits.defs import bind  # noqa: E402
from verity.ml.commitments import advice as ADV  # noqa: E402
from verity.ml.commitments.frame_v3 import BitRowLeaf  # noqa: E402
from verity.ml.commitments.rowleaf import bit_row, row_leaf  # noqa: E402
from verity.ml.crypto import lms  # noqa: E402
from verity.core.protocols.certifier import egress as EG, ingress_scan as IS, io_tie as IT, warden as W  # noqa: E402
from verity.core.protocols.certifier import warden_rows as WR  # noqa: E402
from verity.core.protocols.certifier.active import DATA, id_request  # noqa: E402
from verity.core.protocols.certifier.certificate import Direction, Issuer, LmsSigner  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import produced, registered as REG  # noqa: E402

_TRAFFIC = HERE.parents[1] / "verity/core/protocols/certifier/tests/scan_traffic.py"
_spec = importlib.util.spec_from_file_location("scan_traffic", _TRAFFIC)
traffic = sys.modules["scan_traffic"] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(traffic)

B = 512
CLASS = IS.Class(B=B, r=4, lanes=4, C=47, IDB=17, EVERY=10, WITHIN=12, D=21, KO=64)
SHAPE = CLASS.egress
PARAMS = W.WardenParams(cell_bytes=B, slots=4, bucket_ms=100, delay=21, lanes=4, queue=4, window=32, ingress_cells=47,
                        ingress_buckets=12, egress_cells=65)
PROMPTS = ([(7 * k + 11) % 128_256 for k in range(1500)], [100_000 + k for k in range(600)],
           [131_071 - 3 * k for k in range(40)], [(k * k) % 128_256 for k in range(300)], [5_000 + k for k in range(100)])
ARRIVALS = (0, 2, 2, 12, 15)
RESPONSES = tuple(tuple((97 * k + 13 * q) % (1 << 17) for k in range(n)) for q, n in enumerate((3, 40, 64, 8, 5), 1))
#: link-windows the warden serves (every connection ends by the second's end); window 0 is proved
WINDOWS = 2
SESSION = bytes(range(32))
#: the chain round's swap: S_A and S_{A+1}, state leaves A + 1 and A + 2
A = 9
#: each round's outcome: accepted; refused on ok bits the verifier opens to 0; refused by Lean's registered reads
ROUNDS = {"honest": "accepted", "chain": "ok", "forged": "does not open"}
VALUES = {k: f"verity/certifier/scan-{k}/v1" for k in ("state", "prog", "view")}
ING_ROWS = (("row", 8 * CLASS.S * B), ("sin", IS.state_bits(CLASS)), ("prog", IS.prog_bits(CLASS)),
            ("sout", IS.state_bits(CLASS)), ("view", IS.view_bits(CLASS)))
EG_ROWS = (("state", SHAPE.state_bits), ("row", 8 * SHAPE.R * B))
TIMING = ("e2e_s", "wait_s", "witness_s", "prove_total_s", "proof_bytes", "wire_sent", "wire_recv", "k_log", "nbl",
          "dense_m", "units", "ands_per_vu", "prover", "gpu")


def log(msg: str) -> None:
    print(f"SCZK {time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def _request(i: int, prompt: list[int]) -> bytes:
    return id_request(model="unsloth/Llama-3.2-1B", user=f"user-{i}", prompt=prompt, max_tokens=16 + i, seed=7 * i,
                      temperature=0.7, top_p=0.95, frame_bytes=B, offset=0)


def _words(bits) -> list[int]:
    return [sum(int(b) << i for i, b in enumerate(bits[w:w + 16])) for w in range(0, len(bits), 16)]


# --- the scan's rows: hiding rows the prover registers -----------------------------------------------------------------

def tree(kind: str, rows: list[list[int]], head) -> ADV.Registration:
    """`kind`'s rows (bit rows of whole u16 words) committed as hiding rows, bound to the session and the ingress
    link-window's head."""
    binding = hashlib.sha512(b"\0".join([VALUES[kind].encode(), SESSION, head.value.encode(), head.root])).digest()
    return ADV.register([bit_row((_words(r), 16))[0] for r in rows], len(rows[0]), binding)


def tree_bc(reg: ADV.Registration, i: int) -> bytes:
    """Leaf i's row commitment b‖c, as a proof that holds the row under its salt commits it."""
    return ADV.INST.commit_string(reg.key, BitRowLeaf(reg.n_bits).layout().digest(reg.rows[i]), reg.salts[i])


def tree_port(kind: str, reg: ADV.Registration, reads: list[int], reads_from: int) -> dict:
    """The verifier's own ``--registered`` port of `kind`'s registration: unit ``reads_from + i`` reads leaf reads[i]."""
    n = len(reg.rows)
    e = {"leaves": n, "words": reg.n_bits // 16, "word_bits": 16, "domain": ADV.domain(reg.binding, n).domain_id.hex()}
    port = produced.port(VALUES[kind], e, reg.root.hex(), reads, BitRowLeaf(reg.n_bits).hiding_schema)
    return port | ({REG.READS_FROM: int(reads_from)} if reads_from else {})


def tree_entry(kind: str, reg: ADV.Registration, positions) -> dict:
    return {"value": VALUES[kind], "positions": [int(p) for p in positions],
            "paths": [b"".join(reg.open(p).path).hex() for p in positions]}


def tree_refusal(port: dict, reg: ADV.Registration, entry: dict, bcs: list[bytes]) -> str | None:
    """The host's read of a statement's entry on a scan registration, as Lean's registered reads make it (in
    `warden_rows.refusal`'s order): None when it holds."""
    if entry["value"] != port["value"]:
        return f"registered reads: the port reads {port['value']}, not another value ({entry['value']})"
    n, root = len(reg.rows), bytes.fromhex(port["root"])
    d, scheme = ADV.domain(reg.binding, n), ADV._scheme(reg.binding, n, reg.n_bits)
    for r, (p, bc, path) in enumerate(zip(entry["positions"], bcs, entry["paths"], strict=True)):
        leaf = row_leaf(d, p, ADV.INST.tree_leaf(reg.key, bc), BitRowLeaf(reg.n_bits).hiding_schema)
        nodes = tuple(bytes.fromhex(path[i:i + 128]) for i in range(0, len(path), 128))
        if not scheme.verify_path(ADV.PORT, root, n, p, leaf, nodes):
            return f"registered reads: table row {r} does not open the registered root at position {p}"
    base = int(port.get(REG.READS_FROM, 0))
    for i, (p, want) in enumerate(zip(entry["positions"], port["reads"], strict=True)):
        if p != want:
            return f"registered reads: unit {base + i} reads position {p}, and the verifier's program reads {want}"
    return None


# --- the run --------------------------------------------------------------------------------------------------------

def link(out: Path) -> SimpleNamespace:
    """The run, window 0 of each link committed and signed by the warden and read back by the verifier, the scan's
    chain over it (`scan_traffic.chain`) and the prover's three registrations."""
    CLASS.serves(PARAMS)
    t0 = time.time()
    reqs = [_request(i, p) for i, p in enumerate(PROMPTS)]
    served = traffic.serve(PARAMS, reqs, ARRIVALS, [1 << 20] * len(reqs), RESPONSES, WINDOWS)
    if served.fails:
        raise SystemExit(f"the warden failed {len(served.fails)} connections")
    T = PARAMS.window
    ingress, egress = served.rows()
    ch = traffic.chain(CLASS, ingress[:T], egress[:T], served.outputs)
    if [EG.verdict(SHAPE, ch.view(t), ch.egress[t]) for t in range(T)] != [EG.ACCEPT] * T:
        raise SystemExit("an egress bucket refuses the view its ingress bucket wrote")
    out.mkdir(parents=True, exist_ok=True)
    signer = LmsSigner.create(lms.keygen(lms.Params(64, 1, 6), hashlib.sha512(b"scan-zk").digest()),
                              out / "signer-state.json")
    sides = {}
    for side, records, direction, rows in (("ingress", served.ingress, Direction.INGRESS, ch.ingress),
                                           ("egress", served.egress, Direction.EGRESS, ch.egress)):
        cells, cert = WR.attest(Issuer(signer, SESSION), records[0], PARAMS, direction)
        if cert is None:
            raise SystemExit("the test signer refused a link-window")
        registration = WR.registration(cells.head, cert, PARAMS)
        head, dom = WR.check_registration(registration, signer.public_key(), SESSION, PARAMS, records[0].link,
                                          direction, 0)
        if list(cells.rows) != rows:
            raise SystemExit(f"{side}: the warden's committed rows aren't the rows the scan read")
        sides[side] = SimpleNamespace(cells=cells, registration=registration, head=head, dom=dom)
    head = sides["ingress"].head
    trees = {"state": tree("state", ch.states, head), "prog": tree("prog", ch.progs, head),
             "view": tree("view", ch.views, head)}
    swapped = list(ch.states)
    swapped[A + 1], swapped[A + 2] = swapped[A + 2], swapped[A + 1]
    ordinals = sorted({s.q for st in ch.scan for s in st.slots if s.live})
    admitted = dict(zip(range(1, len(ARRIVALS) + 1), served.admitted))
    return SimpleNamespace(served=served, chain=ch, T=T, sides=sides, trees=trees, swapped=tree("state", swapped, head),
                           ordinals=ordinals, admitted=admitted, run_s=round(time.time() - t0, 1))


def _forged(at: SimpleNamespace) -> tuple[int, bytes]:
    """The first egress row with a `DATA` cell, the last digit of that cell's id changed."""
    for t, row in enumerate(at.chain.egress):
        for s in range(SHAPE.R):
            if row[s * B] == DATA:
                x = bytearray(row)
                k = s * B + IT.ID_AT + IT.ID_DIGITS - 1
                x[k] = ord("0") + (x[k] - ord("0") + 1) % 10
                return t, bytes(x)
    raise SystemExit("the window has no DATA cell")


def member(at: SimpleNamespace, side: str, name: str, base: int) -> SimpleNamespace:
    """Round `name`'s `side` member: each unit's rows (in its signature's order, as bits or bytes), the salts its rows
    are committed under, its header's `registered` entries and the verifier's own ports, the host's read of them and
    each unit's output."""
    ch, T = at.chain, at.T
    cells = at.sides[side].cells
    _w, row_salts, row_entry = cells.statement(range(T))
    port = {"row": WR.bucket_port(at.sides[side].head, PARAMS, base)}
    if side == "ingress":
        st, prog, view = (at.swapped if name == "chain" else at.trees["state"]), at.trees["prog"], at.trees["view"]
        reads = {"sin": ("state", st, list(range(T))), "prog": ("prog", prog, list(range(T))),
                 "sout": ("state", st, list(range(1, T + 1))), "view": ("view", view, list(range(T)))}
        rows = list(ch.ingress)
        units = [(rows[t], *(_bits(r, p) for _k, r, ps in reads.values() for p in [ps[t]])) for t in range(T)]
        outputs = [IS.reference(CLASS, *u) for u in units]
    else:
        reads = {"state": ("view", at.trees["view"], list(range(T)))}
        rows = list(ch.egress)
        changed = None
        if name == "forged":
            t, rows[t] = _forged(at)
            changed = t
        units = [(_bits(at.trees["view"], t), rows[t]) for t in range(T)]
        if any(u[0] != EG.state_bits(SHAPE, ch.view(t)) for t, u in enumerate(units)):
            raise SystemExit("a view leaf isn't its bucket's egress view")
        outputs = [EG.verdict(SHAPE, ch.view(t), rows[t]) for t in range(T)]
    salts = {"row": row_salts, **{p: [reg.salts[i] for i in ps] for p, (_k, reg, ps) in reads.items()}}
    entries = {"row": row_entry, **{p: tree_entry(k, reg, ps) for p, (k, reg, ps) in reads.items()}}
    port.update({p: tree_port(k, reg, ps, base) for p, (k, reg, ps) in reads.items()})
    bcs = {"row": [WR.bc(r, s) for r, s in zip(rows, row_salts)],
           **{p: [tree_bc(reg, i) for i in ps] for p, (_k, reg, ps) in reads.items()}}
    refused = WR.refusal(port["row"], at.sides[side].dom, row_entry, bcs["row"])
    for p, (_k, reg, _ps) in reads.items():
        refused = refused or tree_refusal(port[p], reg, entries[p], bcs[p])
    return SimpleNamespace(side=side, name=name, units=units, rows=rows, salts=salts, entries=entries, ports=port,
                           bcs=bcs, refused=refused, outputs=outputs, changed=changed if side == "egress" else None)


def _bits(reg: ADV.Registration, i: int) -> list[int]:
    row = reg.rows[i]
    return [row[k // 8] >> k % 8 & 1 for k in range(reg.n_bits)]


def predicted(name: str, members: list[SimpleNamespace]) -> bool:
    """The host's read of a round before any proof, as the round's: honest, every read held and every output 1;
    chain, every read held and the swapped buckets' ok bits 0; forged, its egress row port refused."""
    if ROUNDS[name] == "accepted":
        return all(m.refused is None and set(m.outputs) == {1} for m in members)
    if ROUNDS[name] == "ok":
        ing = next(m for m in members if m.side == "ingress")
        return all(m.refused is None for m in members) and [t for t, o in enumerate(ing.outputs) if not o] == [
            A, A + 1, A + 2]
    return all(ROUNDS[name] in (m.refused or "") for m in members if m.side == "egress")


def host(out: Path) -> SimpleNamespace:
    """The run and every round's members as the host reads them (no lowering, no proof): what the test holds."""
    at = link(out)
    rounds = {"honest": ("ingress", "egress"), "chain": ("ingress", "egress"), "forged": ("egress",)}
    got = {}
    for name, sides in rounds.items():
        got[name] = [member(at, side, name, base) for side, base in zip(sides, _bases(sides))]
        if not predicted(name, got[name]):
            raise SystemExit(f"{name}: the host's read of the round isn't {ROUNDS[name]}: "
                             f"{[(m.side, m.refused, m.outputs) for m in got[name]]}")
    return SimpleNamespace(at=at, rounds=got)


def _bases(sides) -> list[int]:
    """Each member's first unit in its round's Program: the templates in order, T units each."""
    return [k * PARAMS.window for k in range(len(sides))]


# --- the statements ---------------------------------------------------------------------------------------------------

def _defs() -> dict[str, object]:
    return {"ingress": bind(IS.IngressScanBucket, B=CLASS.B, r=CLASS.r, lanes=CLASS.lanes, C=CLASS.C, IDB=CLASS.IDB,
                            EVERY=CLASS.EVERY, WITHIN=CLASS.WITHIN, D=CLASS.D, KO=CLASS.KO),
            "egress": bind(EG.EgressBucket, B=SHAPE.B, R=SHAPE.R, r=SHAPE.r, NR=SHAPE.NR, K=SHAPE.K, IB=SHAPE.IB,
                           D=SHAPE.D)}


def program(M, sides: tuple[str, ...], T: int, out: Path) -> SimpleNamespace:
    """A round's Program and partition (a template a side, T units each) and each side's stager."""
    defs = _defs()
    prog = M.P.mixed_population_program([(defs[s], T) for s in sides])
    query = M.P.template_instances_query([defs[s] for s in sides])
    part = M.P.partition(prog, query)
    out.mkdir(parents=True, exist_ok=True)
    (out / "partition.json").write_text(json.dumps(part, sort_keys=True))
    (out / "program.json").write_bytes(M.canonical_json(M.encode_program(prog)))
    bound = {"partition": M.P.digest(part), "program": part["program"]}
    stagers = {}
    for side, (tid, base, cnt) in zip(sides, M.P.ranges(prog, query), strict=True):
        assert tid == M.P.template_id(defs[side]) and cnt == T and base == _bases(sides)[sides.index(side)]
        t = time.time()
        rows = ING_ROWS if side == "ingress" else EG_ROWS
        sdir = out / side
        sdir.mkdir(exist_ok=True)
        st = M.CS.UnitStager(defs[side], f"unit/scan-{side}", [(r, M.V2, b, [r]) for r, b in rows], [(M.V2, (0,))],
                             circuit=sdir / "circuit.txt", bind=bound)
        st.pin
        stagers[side] = SimpleNamespace(defn=defs[side], st=st, base=base, n=T, out=sdir, rows=rows,
                                        lower_s=round(time.time() - t, 1))
    return SimpleNamespace(program=prog, query=query, sides=sides, stagers=stagers, partition_file=out / "partition.json",
                           program_file=out / "program.json")


def stage(M, pg: SimpleNamespace, mr: SimpleNamespace, out: Path) -> dict:
    """Member mr's statement: each unit's rows in a shared-row file (each instance its own table row), every input row
    under the salts and paths of its registered entry, ``o0`` under a fresh salt; and the verifier's own `--registered`
    file."""
    t0 = time.time()
    sg = pg.stagers[mr.side]
    st, n = sg.st, sg.n
    nbl = max(3, (-(-n // st.meta["vus_per_block"]) - 1).bit_length())
    if not 25 <= st.meta["k_log"] + nbl <= M.a7.M_MAX:
        raise SystemExit(f"{mr.side}: m = {st.meta['k_log']} + {nbl} is outside 25 .. {M.a7.M_MAX}")
    words = {}
    for k, (r, _b) in enumerate(sg.rows):
        col = [u[k] for u in mr.units]
        words[r] = M.np.asarray([WR.words(x) if isinstance(x, bytes) else _words(x) for x in col], dtype=M.np.uint16)
    set_name = f"{st.iset.name}-{mr.name}"
    salts = {**mr.salts, "o0": [os.urandom(WR.SALT_BYTES) for _ in range(n)]}
    got = st.stage(words, out, os.urandom(32), salts=salts, indices=range(sg.base, sg.base + n), set_name=set_name,
                   tables=words, refs={r: list(range(n)) for r in words}, registered=mr.entries)
    o0 = [int(x[0]) for x in M.np.asarray(got["outs"]["o0"])]
    if o0 != mr.outputs:
        raise SystemExit(f"{mr.name}/{mr.side}: the units' outputs {o0} are not the reference's {mr.outputs}")
    (out / "inst.bin").rename(out / f"inst-{n}.bin")
    (out / "pub.bin").rename(out / f"pub-{n}.bin")
    (out / "tables").mkdir(exist_ok=True)
    os.link(sg.out / "circuit.txt", out / "circuit.txt")
    (out / "registered.json").write_text(json.dumps(WR.reads_file(mr.ports), sort_keys=True))
    m = {"tag": f"{mr.name}-{mr.side}", "name": mr.side, "dir": out, "meta": st.meta, "pin": st.pin,
         "class": st.class_sha512, "set": set_name, "n": n, "base": sg.base, "defn": sg.defn, "ands": got["ands"],
         "unit_nnz": got["unit_nnz"], "salts": got["salts"], "ports": st.ports, "outputs": o0, "units": mr.units,
         "partition_file": pg.partition_file, "program_file": pg.program_file,
         "registered_file": out / "registered.json", "k_log": st.meta["k_log"], "nbl": nbl,
         "stage_s": round(time.time() - t0, 1)}
    m["header"] = M.a0.public_header(out, n)
    m["bcs"] = shared_bcs(out / f"pub-{n}.bin", st.ports)
    if any(m["bcs"][i][p] != bc for p, col in mr.bcs.items() for i, bc in enumerate(col)):
        raise SystemExit(f"{mr.name}/{mr.side}: a staged row's b‖c isn't the one its registered entry opens")
    if not all(M.a7.opens(next(p for p in st.ports if p.name == "o0"), [w], salts["o0"][i], m["bcs"][i]["o0"])
               for i, w in enumerate(o0)):
        raise SystemExit(f"{mr.name}/{mr.side}: a unit's output doesn't open against its public o0 commitment")
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


# --- the audit ----------------------------------------------------------------------------------------------------------

def audit(M, a, at: SimpleNamespace, pg: SimpleNamespace, name: str, members: list[dict], trees: dict, out: Path,
          port: int) -> dict:
    """One round registered (its members' ports and the scan's registrations' roots, before any coin), every unit drawn
    by the verifier's Lean, each member proved --zk behind the firewall and held by Lean with the verifier's own
    registered reads; then each held unit's ``o0`` opened under the salt the prover sends. Writes what the verifier
    received under ``out / "verifier"``."""
    a0, a7, P, R, V = M.a0, M.a7, M.P, M.R, M.V
    out.mkdir(parents=True, exist_ok=True)
    want, roots, publics = {}, [], {}
    for m in members:
        fv = m["header"]["frame_v3"]
        for p_, w in a7.frame_ports(m["set"], m["meta"], m["n"]).items():
            want[f"{m['name']}/{p_}"] = w
            roots.append({"port": f"{m['name']}/{p_}", "root": fv["roots"][p_], "leaves": w["leaves"],
                          "domain": fv["domain_ids"][p_]})
        publics[m["name"]] = a0.sha512_file(m["dir"] / f"pub-{m['n']}.bin")
    for kind, reg in trees.items():
        d = ADV.domain(reg.binding, len(reg.rows)).domain_id.hex()
        want[f"scan/{kind}"] = {"leaves": len(reg.rows), "domain": d}
        roots.append({"port": f"scan/{kind}", "root": reg.root.hex(), "leaves": len(reg.rows), "domain": d})
    leaf_layer = hashlib.sha512(M.canonical_json(publics)).hexdigest()
    n_all = sum(m["n"] for m in members)
    law = M.D.parse_law(f"stratified:{n_all}")
    win = {"kind": "stand-in", "of": f"round {name}: link-window 0 of a warden run's ingress and egress at w63's class "
                                     "(scan_zk.py)",
           "members": {m["name"]: {"template": P.template_id(m["defn"]), "units": [m["base"], m["base"] + m["n"]]}
                       for m in members},
           "link": "registered: each row port at the warden's signed root, unit base + t row t; the scan's state, "
                   "prog and view at the prover's registered roots (scan/*)"}
    rec = R.record(pg.program, pg.query, law, a0.SCHEME, roots, win, leaf_layer=leaf_layer)
    (out / "registration.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    checks = {"registration": V.check(rec, pg.program, pg.query, law, a0.SCHEME, want, leaf_layer=leaf_layer).to_json()}
    for m in members:
        checks[f"statement/{m['name']}"] = a7.statement_checks(m, rec, want)
    receipt = R.receipt(rec, "one-stage-e2e verifier (scan_zk.py)")
    res: dict = {"checks_before_draw": {c: v["ok"] for c, v in checks.items()}}
    if not all(v["ok"] for v in checks.values()):
        return {**res, "checks": checks, "accepted": False, "verdicts": {}, "why": {}, "opened": {}}
    p = a0.run([a.lean, "draw", "--program", str(pg.program_file), "--partition", str(pg.partition_file),
                "--stratified", str(n_all)], out / "draw.log")
    lean_draw = json.loads(next(x for x in p.stdout.splitlines() if x.startswith("DRAW ")).split(" ", 1)[1])
    (out / "draw.json").write_text(json.dumps(lean_draw, sort_keys=True))
    proved, served, n_inst, verdicts, statement, sessions, session_roots = [], [], 0, {}, {}, [], {}
    for j, m in enumerate(members):
        local = sorted(u - m["base"] for u in lean_draw["units"] if m["base"] <= u < m["base"] + m["n"])
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
                                "lean_verify_s": lv.get("verify_s"),
                                **{x: (live or {}).get(x) for x in ("circuit_sha512", "statement_digest", "sigma")}}
        sessions += [a0.sha512_file(f) for f in sorted((sdir / "sessions").glob("*/session.json"))]
        log(f"{name}/{m['name']}: {len(local)} units drawn, Lean {lv.get('accepted')} {lv.get('why') or ''}, "
            f"prove {timing.get('prove_total_s')} s, session {proc['wall_s']} s")
    result = V.audit_record(rec, receipt, lean_draw, rec["law"], sorted(proved), n_inst, statement, verdicts, a.delta,
                            sessions=sessions, served=sorted(served), session_roots=session_roots)
    (out / "audit.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    held = set(proved) if result.get("accepted") else set()
    opened, sent = {}, []
    for m in members:
        o0 = next(p for p in m["ports"] if p.name == "o0")
        opened[m["name"]] = [w if (m["base"] + i) in held and M.a7.opens(o0, [w], m["salts"]["o0"][i], m["bcs"][i]["o0"])
                             else None for i, w in enumerate(m["outputs"])]
        sent += [{"unit": m["base"] + i, "o0": w, "salt/o0": m["salts"]["o0"][i].hex()}
                 for i, w in enumerate(m["outputs"]) if (m["base"] + i) in held]
    vdir = out / "verifier"
    vdir.mkdir(exist_ok=True)
    (vdir / "openings.json").write_text(json.dumps(sent, indent=1))
    for side, s in at.sides.items():
        (vdir / f"warden-registration-{side}.json").write_text(json.dumps(s.registration, indent=1, sort_keys=True))
    for f in (out / "registration.json", pg.partition_file, pg.program_file, out / "draw.json", out / "audit.json"):
        os.link(f, vdir / f.name)
    for m in members:
        for f in (m["dir"] / "circuit.txt", m["dir"] / f"pub-{m['n']}.bin", m["registered_file"]):
            os.link(f, vdir / f"{m['name']}-{f.name}")
        for f in sorted((out / f"session-{m['name']}" / "sessions").rglob("*")):
            if f.is_file():
                os.link(f, vdir / f"{m['name']}-session-{f.parent.name}-{f.name}")
    window_ok = bool(result.get("accepted")) and all(w == 1 for ws in opened.values() for w in ws)
    return {**res, "accepted": bool(result.get("accepted")), "window_accepted": window_ok, "opened": opened,
            "audit": {x: result.get(x) for x in ("accepted", "complete", "wrong_units")},
            "verdicts": {x: y.get("accepted") for x, y in verdicts.items()},
            "why": {x: y.get("why") for x, y in verdicts.items()}, "timing": statement}


def expected(name: str, res: dict) -> bool:
    """honest: accepted, every output 1; chain: the proofs held and the swapped buckets' ok bits 0; forged: refused by
    Lean's registered reads."""
    lean = {k.split("/", 1)[1]: v for k, v in res["verdicts"].items() if k.startswith("lean-verify/")}
    if ROUNDS[name] == "accepted":
        return res["window_accepted"] and all(lean.values())
    if ROUNDS[name] == "ok":
        ing = res["opened"].get("ingress") or []
        return res["accepted"] and all(lean.values()) and [t for t, w in enumerate(ing) if w == 0] == [A, A + 1, A + 2]
    why = res["why"].get("lean-verify/egress") or ""
    return not res["accepted"] and lean.get("egress") is False and "registered reads" in why and ROUNDS[name] in why


def blobs_of(members: list[dict]) -> tuple[dict[str, bytes], set[str]]:
    """The witness pieces the verifier must not receive: every unit's rows and every salt but ``o0``'s."""
    blobs = {}
    for m in members:
        for i, u in enumerate(m["units"]):
            for (r, _b), x in zip(ING_ROWS if m["name"] == "ingress" else EG_ROWS, u):
                blobs[f"{m['name']} {r} {i}"] = x if isinstance(x, bytes) else bytes(
                    b for w in _words(x) for b in w.to_bytes(2, "little"))
        blobs.update({f"{m['name']} salt {p} {i}": s for p, ss in m["salts"].items() if p != "o0"
                      for i, s in enumerate(ss)})
    return blobs, set(blobs)


def known(members: list[dict], at: SimpleNamespace) -> bytes:
    """What the scan leaves out: the circuits, zeros, and the window's pads and ENDs (fixed by the ordinals)."""
    cells = [IT.pad_cell(B, q) + IT.end_cell(q, B) for q in [0, *at.ordinals]]
    return b"".join((m["dir"] / "circuit.txt").read_bytes() for m in members) + bytes(WR.SALT_BYTES) + b"".join(cells)


def sparse(pieces: dict[str, bytes], held: set[str]) -> tuple[dict[str, bytes], set[str], int]:
    out = {k for k, x in pieces.items() if sum(map(bool, x)) <= 1}
    return {k: x for k, x in pieces.items() if k not in out}, held - out, len(out)


def prove(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    h = host(out / "warden")
    at, T = h.at, h.at.T
    ch = at.chain
    summary: dict = {"class": asdict(CLASS), "params": asdict(PARAMS), "shape": asdict(SHAPE), "T": T,
                     "bits": {f"{side}/{r}": b for side, rows in (("ingress", ING_ROWS), ("egress", EG_ROWS))
                              for r, b in rows}, "admitted": at.admitted,
                     "anchors": {q: CLASS.anchor(s, s % CLASS.EVERY) for q, s in at.admitted.items()},
                     "responses": [len(x) for x in RESPONSES], "run_s": at.run_s, "swap": [A, A + 1],
                     "cells": {side: {"data": sum(r[s * B] == DATA for r in rows for s in range(SHAPE.R)),
                                      "end": sum(r[s * B] == W.END for r in rows for s in range(SHAPE.R))}
                               for side, rows in (("ingress", ch.ingress), ("egress", ch.egress))},
                     "roots": {k: r.root.hex() for k, r in at.trees.items()},
                     "host": {name: [{"side": m.side, "refused": m.refused, "outputs": m.outputs} for m in ms]
                              for name, ms in h.rounds.items()}}
    log(f"run: {T} buckets of link-window 0, admitted {at.admitted}, anchors {summary['anchors']}, cells "
        f"{summary['cells']}; every egress bucket accepts its ingress bucket's view; host reads as expected "
        f"({at.run_s} s)")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    if a.dry:
        log("dry: done")
        return 0
    M = WZ._one_stage()
    joint = program(M, ("ingress", "egress"), T, out / "joint")
    solo = program(M, ("egress",), T, out / "solo")
    summary["stagers"] = {f"{k}/{s}": {"unit": g.defn.id, "lower_s": g.lower_s, "k_log": g.st.meta["k_log"],
                                       "vus_per_block": g.st.meta["vus_per_block"], "pin": g.st.pin, "base": g.base}
                          for k, pg in (("joint", joint), ("solo", solo)) for s, g in pg.stagers.items()}
    log(f"stagers: {summary['stagers']}")
    ok = True
    for j, (name, mrs) in enumerate(h.rounds.items()):
        pg = solo if name == "forged" else joint
        members = [stage(M, pg, mr, out / name / f"stage-{mr.side}") for mr in mrs]
        if name == "honest":
            same = [x["view"] for x in members[0]["bcs"]] == [x["state"] for x in members[1]["bcs"]]
            summary["view_bcs_shared"] = same
            if not same:
                raise SystemExit("the ingress units' view rows aren't the egress units' state rows")
        summary.setdefault("rounds", {})[name] = {m["name"]: {"ands": m["ands"], "k_log": m["k_log"], "nbl": m["nbl"],
                                                              "m": m["k_log"] + m["nbl"], "stage_s": m["stage_s"],
                                                              "outputs": m["outputs"]} for m in members}
        log(f"staged {name}: " + "; ".join(f"{m['name']} {m['ands']} ANDs a unit, m = {m['k_log']} + {m['nbl']}, "
                                           f"outputs {m['outputs']}" for m in members))
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
        av = SimpleNamespace(bin=a.bin, lean=a.lean, lease=[], gpu=False, delta=a.delta)
        trees = {**at.trees, **({"state": at.swapped} if name == "chain" else {})}
        trees = {k: v for k, v in trees.items() if name != "forged" or k == "view"}
        res = audit(M, av, at, pg, name, members, trees, out / name, a.port + 4 * j)
        blobs, held = blobs_of(members)
        blobs, held, public = LS.public_pieces(M.a8, blobs, held, known(members, at))
        blobs, held, nsparse = sparse(blobs, held)
        inst = out / name / "instances.bin"
        inst.write_bytes(b"".join((m["dir"] / f"inst-{m['n']}.bin").read_bytes() for m in members))
        res["received"] = {**M.a8.received(out / name / "verifier", inst, blobs, held), "public_pieces": len(public),
                           "sparse_pieces": nsparse}
        inst.unlink()
        rec = res["received"]
        good = expected(name, res) and not rec["verifier_found"] and rec["control"]["holds_all"]
        ok &= good
        summary["rounds"][name]["audit"] = res
        log(f"round {name}: proofs {res['verdicts']}, audit {res.get('audit')}, opened "
            f"{ {k: v for k, v in res.get('opened', {}).items()} }, Lean {res['why']}, verifier received "
            f"{rec['verifier_bytes']} bytes, witness pieces found there: {rec['verifier_found'] or 'none'} (of "
            f"{len(blobs)}; {len(public)} public, {nsparse} sparse left out), control {rec['control']['blobs_found']}; "
            f"{'as expected' if good else 'UNEXPECTED'}")
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    summary["ok"] = ok
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    log("done: " + ("ok" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="scan_zk.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prove", help="a link-window's ingress and egress units chained by registered reads, --zk")
    p.add_argument("--bin")
    p.add_argument("--lean")
    p.add_argument("--out", required=True)
    p.add_argument("--port", type=int, default=7811)
    p.add_argument("--delta", type=float, default=2.0 ** -20)
    p.add_argument("--dry", action="store_true", help="the run and the host's read of every round: nothing proved")
    a = ap.parse_args(argv)
    if not a.dry and not (a.bin and a.lean):
        raise SystemExit("prove needs --bin and --lean, or --dry")
    return prove(a)


if __name__ == "__main__":
    raise SystemExit(main())
