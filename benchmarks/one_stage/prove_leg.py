"""The proof leg: units through C-Flock `prove --zk` behind the Lean firewall and the Lean verifier of record, one call
(`prove_units`) that stages each member's units, registers them, takes the verifier's draw, proves each drawn member's
units in one `--zk` session, verifies it with Lean `flock-verify verify --zk` (with `--registered` when the units read
registered values), opens the rows the verifier holds its own copy of, and records every verdict in the one-stage audit
record. Network-accounting's I5 for the arbitrary-vLLM goal (thread 1792816009.557169): the compiled per-step units, N2's
head-check units and the I/O tie's units all ride it.

    python benchmarks/one_stage/prove_leg.py standin --bin flock-circuit --lean flock-verify --firewall flock-firewall \\
      --out DIR [--n 4] [--negatives] [--registered]
    python benchmarks/one_stage/prove_leg.py defs --bin flock-circuit --lean flock-verify --firewall flock-firewall \\
      --out DIR --shape gemm:1b:2048 [--shape attention-v20:1b:384:block ...] [--negatives] [--census] [--reads slots]

* **Members** (`Member`). One template's units, one statement: a bound Definition C-Flock lowers flat (`ir_lower`; Lean
  refuses typed and template statements), `UnitStager`'s rows and output rows, the prover's rows, the rows
  the verifier derives itself (``held``: public inputs, opened by ``b || c`` to its own copy under the salts the prover
  reveals for those rows alone), and the output rows it requires (``expect``: an ok bit). ``stage_kw`` passes a shared-row
  file's tables, refs and registered reads to `UnitStager.stage` for units that read registered values.
* **The program.** The caller's (a window's, whose partition the statements bind), else the population program of the
  members' templates (`partition.mixed_population_program`, `Q_template_instances`), one stratum a member.
* **The registration** (`registration.record`): each statement's frame-v3 ports, the sampling design, the scheme and the
  leaf layer, checked by the verifier of record (R1-R7) and against each statement's header before the draw.
* **The draw** (``coins``): ``"lean"``, the verifier's Lean on its own OS coins (`flock-verify draw --stratified K`, A3);
  a mapping ``{member: [unit, ...]}`` of each member's drawn units, a draw the proof system made on live coins; or a
  callable that takes the registration record and returns that mapping (a `service.Stream`'s register and select).
* **Proving.** Each drawn member's units are one `serve --zk --draw-file` / `prove --zk` session, behind the firewall
  (``FC_FIREWALL=1``, its Lean frame ``FC_FIREWALL_LEAN``) when `Tools.firewall` names one.
* **The verdict of record** is Lean's `verify --zk --archive --partition --program [--registered]` on the session. A unit
  is accepted when its session's Lean and M0 verdicts accept it and its held and expected rows open to the verifier's own.
  The audit record (`verifier.audit_record`) holds every verdict, the draw and the roots.

Negatives (`negatives`), each refused at the proof layer:

    tampered-unit     a unit whose committed output word is wrong, proved with the prover's self-check skipped: the
                      firewall's Lean recheck stops it; unfirewalled, Lean `verify --zk` refuses the proof that reaches it
    tampered-public   a unit staged on a held row other than the verifier's copy: its proof verifies, its opening fails
    tampered-root     a registration whose first root is not the statement's: refused before the draw
    wrong-unit        a member's session checked against another member's circuit (that stratum's committed pin)
    tampered-registered-root  (members with registered reads) the verifier's --registered file with a root changed

Served Definitions (`served`, ``defs``): vLLM's served Definitions as the frontend binds them on the RTX PRO 6000 target
(`targets.gemm_spec`, `attention_spec`, `rmsnorm_fused_spec`, the dead-free rewrites) at a Llama model's statics, each the
template the serving rows commit (`serving_rows.template`: a row, or a ``SLICES`` row's unit), on seeded stand-in values;
and the window circuit's units (`window_circuit`): an attention head's per-key-block Calls on the RTX PRO 6000 and the H200
(``CUT_TARGETS``) and the embedding's registered read. A unit's MUFU table reads are laid out as its template's own
(`chunk_rows.reads`) unless ``--reads`` says otherwise. A shape whose rows' in-circuit hashing alone passes C-Flock's block
(`m1_families.block`) is reported refused, not lowered.

Known limits: units flat or untyped staged; C-Flock's block at most 2^27 (split-K past K = 5041); table-heavy units
need lookup slots; tail lookup slots are the GPU prover's; prover worker threads may need a bigger stack.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import contextlib
import gc
import os
import resource
import shutil
import socket
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import a0  # noqa: E402
import a7  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import draw as D, partition as P, registration as R  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import verifier as V  # noqa: E402
from verity.ml.circuits.codec import canonical_json, encode_program  # noqa: E402
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2  # noqa: E402

FORMAT = "verity/prove-leg/v0"
TIMING = ("e2e_s", "wait_s", "witness_s", "prove_total_s", "proof_bytes", "k_log", "nbl", "dense_m", "units")


def log(msg: str) -> None:
    print(f"LEG {time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def maxrss_gb(who: int) -> float:
    return round(resource.getrusage(who).ru_maxrss / 2**20, 2)


@dataclass
class Member:
    """One template's units, staged as one statement. ``values``: the prover's rows, ``{row: (n, words) u16}``.
    ``held``: rows the verifier derives itself, opened to its copy. ``expect``: output rows (``o<j>``) it requires.
    ``forge``: instances whose first output word has bit 0 flipped before commitment (a negative's prover, its self-check
    skipped). ``stage_kw``: `UnitStager.stage`'s ``tables``, ``refs``, ``registered`` and ``salts``. ``reads``: `UnitStager`'s,
    ``slots`` its reads of the pinned MUFU tables in lookup slots, the tables written beside the statement."""

    name: str
    defn: Any
    rows: list[tuple[str, str, int, list[str]]]
    out_rows: list[tuple[str, tuple[int, ...]]]
    values: dict[str, np.ndarray]
    held: dict[str, np.ndarray] = field(default_factory=dict)
    expect: dict[str, np.ndarray] = field(default_factory=dict)
    forge: tuple[int, ...] = ()
    stage_kw: dict = field(default_factory=dict)
    reads: str = "inline"

    @property
    def n(self) -> int:
        return len(next(iter(self.values.values())))


@dataclass(frozen=True)
class Tools:
    """The binaries: M0's ``flock-circuit`` (the proving build), the Lean ``flock-verify``, and the Lean firewall's frame
    ``flock-firewall`` (None: proved without the firewall)."""

    bin: str
    lean: str
    firewall: str | None = None
    delta: float = 2.0 ** -20


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --- staging ----------------------------------------------------------------------------------------------------------

#: `stage`'s bound stagers by unit and partition inside `reuse_lowerings`, else None
_STAGERS: dict | None = None


class OutsideShape(ValueError):
    """A statement whose m (its block's log plus its block bits) is outside --zk's and flock-circuit's 25 .. `a7.M_MAX`:
    its ``k_log`` and ``per_block`` (units a block) say how many units would fit (`units_for`)."""

    def __init__(self, msg: str, k_log: int, per_block: int):
        super().__init__(msg)
        self.k_log, self.per_block = k_log, per_block

def _stage_forged(st, m: Member, d: Path, salt_key: bytes, indices: list[int], set_name: str) -> dict:
    """`UnitStager.stage` with bit 0 of the first output word of each instance in ``m.forge`` flipped before the set is
    committed (`forged_proofs.stage_case`): its rows and ``b || c`` agree with the outputs it commits, so only the proof
    can refuse it."""
    from types import SimpleNamespace

    from verity_flock import circuit as FC

    n = m.n
    iset = SimpleNamespace(**{**vars(st.iset), "name": set_name})
    flat = np.concatenate([np.asarray(m.values[r], dtype=np.uint16) for r in st.row_ports], axis=1)
    got, _cut, ok = st.low.evaluate(flat.astype(np.uint64))
    if int(ok) != (1 << n) - 1:
        raise AssertionError(f"{m.name}: a unit is unsatisfied on its rows")
    got = got.astype(np.uint16)
    at = np.cumsum([0] + st.out_words)
    outs = {f"o{j}": got[:, at[j]:at[j + 1]].copy() for j in range(len(st.out_words))}
    st.reference_check(m.values, outs)
    for i in m.forge:
        outs["o0"][i, 0] ^= 1
    tables = m.stage_kw.get("tables")
    table_rows = None if tables is None else {p.name: len(tables[p.name]) if p.name in tables else n for p in st.ports}
    salts = {**FC.stage_salts(iset, 0, n, st.ports, salt_key, table_rows), **(m.stage_kw.get("salts") or {})}
    d.mkdir(parents=True, exist_ok=True)
    FC.write(st.text, iset, 0, n, d / "inst.bin", d / "pub.bin", salt_key,
             rows={r: np.asarray(m.values[r], dtype=np.uint16) for r, *_ in st.in_rows}, outs=outs, salts=salts,
             public_words={}, digests=(st.pin, st.class_sha512), indices=indices, share_rows=tables is not None,
             tables=tables, refs=m.stage_kw.get("refs"), registered=m.stage_kw.get("registered"), meta=st.meta)
    counts = st.low.layout.counts
    return {"outs": outs, "salts": salts, "ands": counts["ands"], "unit_nnz": counts["nnz_a"] + counts["nnz_b"]}


def stage(m: Member, base: int, bound: dict, out: Path, salt_key: bytes, files: dict) -> dict:
    """Member ``m``'s statement, bound to the partition (``bound``) at global units ``base ..``: its circuit lowered
    once, its rows committed under salts from ``salt_key``. Returns a0's statement dict (``dir``, ``pin``, the header ...)."""
    from verity_flock import class_statement as CS, ir_lower as IL

    t0 = time.time()
    sd = out / f"stage-{m.name}"
    key = json.dumps([m.defn.id, m.name, m.rows, m.out_rows, bound, m.reads], sort_keys=True)
    st = _STAGERS.get(key) if _STAGERS is not None else None
    if st is None:
        st = CS.UnitStager(m.defn, f"unit/{m.name}", m.rows, m.out_rows, circuit=sd / "circuit.txt", bind=bound, reads=m.reads)
        if _STAGERS is not None:
            _STAGERS[key] = st
    else:
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "circuit.txt").write_text(st.text)
    lower_s = round(time.time() - t0, 1)
    k_log = st.meta["k_log"]
    nbl = max(3, (-(-m.n // st.meta["vus_per_block"]) - 1).bit_length())
    if not 25 <= k_log + nbl <= a7.M_MAX:
        raise OutsideShape(f"{m.name}: m = {k_log} + {nbl} is outside 25 .. {a7.M_MAX} (--zk and flock-circuit's shape)",
                           k_log, st.meta["vus_per_block"])
    d = sd / "statement"
    set_name = f"{st.iset.name}-{m.name}"
    indices = list(range(base, base + m.n))
    if m.forge:
        got = _stage_forged(st, m, d, salt_key, indices, set_name)
    else:
        got = st.stage(m.values, d, salt_key, indices=indices, set_name=set_name, **m.stage_kw)
    (d / "inst.bin").rename(d / f"inst-{m.n}.bin")
    (d / "pub.bin").rename(d / f"pub-{m.n}.bin")
    (d / "tables").mkdir(exist_ok=True)
    if m.reads == "slots":
        IL.write_tables(d / "tables")
    if not (d / "circuit.txt").exists():
        os.link(sd / "circuit.txt", d / "circuit.txt")
    s = {"name": m.name, "tag": m.name, "dir": d, "meta": st.meta, "pin": st.pin, "class": st.class_sha512, "set": set_name,
         "n": m.n, "base": base, "defn": m.defn, "ands": got["ands"], "unit_nnz": got["unit_nnz"], "salts": got["salts"],
         "ports": st.ports, "outs": got["outs"], "lower_s": lower_s, "stage_s": round(time.time() - t0, 1), **files}
    s["header"] = a0.public_header(d, m.n)
    return s


# --- the leg ----------------------------------------------------------------------------------------------------------

def _draw(coins, rec: dict, program_file: Path, partition_file: Path, k: int, lean: str, out: Path,
          bases: dict[str, tuple[int, int]]) -> dict:
    """The verifier's draw as one draw object over global units: Lean's own (``coins == "lean"``), or the caller's."""
    if coins == "lean":
        p = a0.run([lean, "draw", "--program", str(program_file), "--partition", str(partition_file), "--stratified", str(k)],
                   out / "draw.log")
        line = next((x for x in p.stdout.splitlines() if x.startswith("DRAW ")), None)
        if line is None:
            raise RuntimeError(f"flock-verify draw gave no draw (rc {p.returncode}): {p.stderr[-400:]}")
        return json.loads(line.split(" ", 1)[1])
    drawn = coins(rec) if callable(coins) else coins
    units = sorted(bases[name][0] + int(u) for name, us in drawn.items() for u in us)
    return {**rec["law"], "population": rec["population"], "units": units}


def frame_ports(set_name: str, meta: dict, n: int, tables: Mapping[str, Any] | None = None) -> dict[str, dict]:
    """Each port's frame-v3 tree as the verifier derives it: its binding names the set, the instance range, the port and its
    schema; its leaves are the instances' rows, or a shared-row port's table rows (`circuit.statement`)."""
    from verity.ml.commitments.identity import identity_digest_sha512
    from verity.ml.commitments.indexed import RangeIndexedDomain
    from verity.ml.commitments.merkle import CommitmentDomain

    out = {}
    for (port, _w, _c), (schema, _bits) in zip(meta["ports"], meta["row_schemas"], strict=True):
        b = identity_digest_sha512(a0.STAND_IN_BINDING, {"set": set_name, "lo": 0, "hi": n, "port": port, "schema": schema})
        k = len((tables or {}).get(port, range(n)))
        out[port] = {"leaves": k, "domain": CommitmentDomain(b, -1, RangeIndexedDomain(0, k), hash="sha512").domain_id.hex()}
    return out


def public_bcs(path: Path, ports: list) -> tuple[dict[str, list[bytes]], dict[str, list[int]]]:
    """A public file's ``b || c`` per port, one per row of the port's tree, and the row each instance reads: its own (rows
    instance-major), or a shared-row file's ref into the port's table (tables port-major, then the refs; `circuit.write`)."""
    from verity_flock import circuit as FC

    head, body = Path(path).read_bytes().split(b"\n", 1)
    h, w, q = json.loads(head), FC.BC_BYTES, len(ports)
    n = h["instances"]
    if "shared_rows" not in h:
        return ({p.name: [body[(i * q + j) * w:(i * q + j + 1) * w] for i in range(n)] for j, p in enumerate(ports)},
                {p.name: list(range(n)) for p in ports})
    bcs, at = {}, 0
    for p in ports:
        k = h["shared_rows"][p.name]
        bcs[p.name] = [body[(at + j) * w:(at + j + 1) * w] for j in range(k)]
        at += k
    refs = np.frombuffer(body[at * w:at * w + 4 * n * q], dtype="<u4").reshape(n, q)
    return bcs, {p.name: [int(x) for x in refs[:, j]] for j, p in enumerate(ports)}


def opened(s: dict, local: list[int], m: Member) -> dict:
    """Each drawn unit's held rows opened to the verifier's own copy and its expected output rows to the required words,
    under the salts the prover reveals for those rows alone: ``{unit: [failed row, ...]}``."""
    if not (m.held or m.expect):
        return {i: [] for i in local}
    bcs, refs = public_bcs(s["dir"] / f"pub-{s['n']}.bin", s["ports"])
    port = {p.name: p for p in s["ports"]}
    out = {}
    for i in local:
        out[i] = [r for r, want in {**m.held, **m.expect}.items()
                  if not a7.opens(port[r], [int(w) for w in np.asarray(want[i]).reshape(-1)], s["salts"][r][refs[r][i]],
                                  bcs[r][refs[r][i]])]
    return out


def prove_units(members: Sequence[Member], coins: str | Mapping[str, Sequence[int]] | Callable[[dict], Mapping[str, Sequence[int]]],
                registered: Path | None = None, *, tools: Tools, out: Path, k: int | None = None, program=None, query=None,
                window: dict | None = None, mutate: Callable[[dict], dict] | None = None) -> dict:
    """Every member staged, registered, drawn, proved ``--zk`` and verified by Lean; returns the leg's record
    (``verity/prove-leg/v0``): per member its drawn units, each unit's verdict, the session's verdicts and timing; the
    audit record; and ``accepted``. ``registered``: the verifier's own ``--registered`` file. ``k``: the stratified rule's
    K (default every unit). ``program``/``query``: the window's, else the members' population program. ``window``: the
    registration's window (default a stand-in's). ``mutate``: the registration record as the verifier is handed it (a
    negative's), from the honest one."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if program is None:
        program = P.mixed_population_program([(m.defn, m.n) for m in members])
        query = P.template_instances_query([m.defn for m in members])
    part = P.partition(program, query)
    files = {"partition_file": out / "partition.json", "program_file": out / "program.json"}
    files["partition_file"].write_text(json.dumps(part, sort_keys=True))
    files["program_file"].write_bytes(canonical_json(encode_program(program)))
    if registered is not None:
        files["registered_file"] = Path(registered)
    bound = {"partition": P.digest(part), "program": part["program"]}
    bases: dict[str, tuple[int, int]] = {}
    for tid, base, cnt in P.ranges(program, query):
        b0, c0 = bases.get(tid, (base, 0))
        if b0 + c0 != base:
            raise ValueError(f"template {tid}'s units are not one contiguous range of the program's (a member is one statement)")
        bases[tid] = (b0, c0 + cnt)
    population = P.units(program, query)
    leg: dict[str, Any] = {"format": FORMAT, "population": population, "partition": bound["partition"],
                           "program": bound["program"], "firewall": tools.firewall is not None, "members": {},
                           "_out": str(out)}

    salt_key = os.urandom(32)
    staged, by_name = [], {}
    for m in members:
        tid = P.template_id(m.defn)
        if tid not in bases or bases[tid][1] != m.n:
            raise ValueError(f"{m.name}: template {tid} has no range of {m.n} units in the program ({bases.get(tid)})")
        s = stage(m, bases[tid][0], bound, out, salt_key, files)
        staged.append(s)
        by_name[m.name] = (bases[tid][0], m.n)
        leg["members"][m.name] = {"template": tid, "n": m.n, "base": s["base"], "ands": s["ands"], "unit_nnz": s["unit_nnz"],
                                  "k_log": s["meta"].get("k_log"), "pin": s["pin"], "lower_s": s["lower_s"],
                                  "stage_s": s["stage_s"], "forged": list(m.forge)}
        log(f"staged {m.name}: {m.n} units from {s['base']}, {s['ands']} ANDs, k_log {s['meta'].get('k_log')}, "
            f"pin {s['pin'][:16]}, {s['stage_s']} s")

    want, roots, publics = {}, [], {}
    for m, s in zip(members, staged):
        fv = s["header"]["frame_v3"]
        for p_, w in frame_ports(s["set"], s["meta"], s["n"], m.stage_kw.get("tables")).items():
            want[f"{s['name']}/{p_}"] = w
            roots.append({"port": f"{s['name']}/{p_}", "root": fv["roots"][p_], "leaves": w["leaves"],
                          "domain": fv["domain_ids"][p_]})
        publics[s["name"]] = a0.sha512_file(s["dir"] / f"pub-{s['n']}.bin")
    leaf_layer = hashlib.sha512(canonical_json(publics)).hexdigest()
    k = population if k is None else k
    law = D.parse_law(f"stratified:{k}")
    window = window or {"kind": "stand-in", "of": "the proof leg's members (prove_leg.py)",
                        "members": {s["name"]: {"template": P.template_id(s["defn"]), "units": [s["base"], s["base"] + s["n"]]}
                                    for s in staged}, "link": "none"}
    honest = R.record(program, query, law, a0.SCHEME, roots, window, leaf_layer=leaf_layer)
    rec = mutate(copy.deepcopy(honest)) if mutate else honest
    (out / "registration.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    checks = {"registration": V.check(rec, program, query, law, a0.SCHEME, want, leaf_layer=leaf_layer).to_json()}
    for s in staged:
        checks[f"statement/{s['name']}"] = a7.statement_checks(s, rec, want)
    receipt = R.receipt(rec, "prove-leg verifier (prove_leg.py)")
    leg["registration"] = R.digest(rec)
    leg["checks_before_draw"] = {c: {"ok": v["ok"], "codes": v.get("codes")} for c, v in checks.items()}
    if not all(v["ok"] for v in checks.values()):
        leg.update(accepted=False, refused="before the draw", seconds=round(time.time() - t0, 1))
        log(f"refused before the draw: {[c for c, v in checks.items() if not v['ok']]}")
        (out / "leg.json").write_text(json.dumps(leg, indent=1, sort_keys=True, default=str))
        return leg

    draw = _draw(coins, rec, files["program_file"], files["partition_file"], k, tools.lean, out,
                 {s["name"]: (s["base"], s["n"]) for s in staged})
    (out / "draw.json").write_text(json.dumps(draw, sort_keys=True))
    served, proved, n_inst, verdicts, statement, sessions, session_roots = [], [], 0, {}, {}, [], {}
    for m, s in zip(members, staged):
        local = sorted(u - s["base"] for u in draw["units"] if s["base"] <= u < s["base"] + s["n"])
        ml = leg["members"][m.name]
        ml["drawn"] = local
        if not local:
            continue
        sdir = out / f"session-{m.name}"
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "draw.json").write_text(json.dumps({"law": "subset", "population": s["n"], "k": len(local), "units": local},
                                                   sort_keys=True))
        env = {}
        if tools.firewall:
            env = {"FC_FIREWALL": "1", "FC_FIREWALL_LEAN": tools.firewall, **a0.firewall_env(s)}
        if m.forge:
            env["FC_ZK_SELF_CHECK"] = "skip"
        t = time.time()
        record, live = a0.session(tools.bin, s, s["n"], f"subset:{len(local)}", free_port(), sdir,
                                  draw_file=sdir / "draw.json", zk=True, env=env)
        session_s = round(time.time() - t, 1)
        draw_t = (record or {}).get("unit_draw") or {}
        v = V.check_draw(draw_t, {"population": s["n"]}, {"law": "subset", "k": len(local)})
        served += [s["base"] + u for u in draw_t.get("units", [])]
        session_roots.update({f"{m.name}/{p_}": r for p_, r in ((live or {}).get("roots") or {}).items()})
        proved += a0.proved_units(s["header"], live or {})
        n_inst += (live or {}).get("instances", 0)
        m0 = {"accepted": ((record or {}).get("verdict") or {}).get("accepted") is True
              and (live or {}).get("roots") == s["header"]["frame_v3"]["roots"] and v.ok,
              "why": ((record or {}).get("verdict") or {}).get("why") or (live or {}).get("why"), "draw_check": v.to_json()}
        lv = a0.lean_verify(tools.lean, s, s["n"], sdir, zk=True)
        op = opened(s, local, m)
        verdicts[f"m0-verifier/{m.name}"] = m0
        verdicts[f"lean-verify/{m.name}"] = {**lv, "tool": "flock-verify verify --zk --archive --partition --program"
                                             + (" --registered" if registered else "")}
        verdicts[f"openings/{m.name}"] = {"accepted": not any(op.values()), "detail": {i: b for i, b in op.items() if b}}
        statement[m.name] = {"session_s": session_s, **{x: (live or {}).get(x) for x in ("circuit_sha512", "statement_digest",
                                                                                         "sigma")},
                             "timing": {x: (live or {}).get(x) for x in TIMING}, "lean_verify_s": lv.get("verify_s")}
        sessions += [a0.sha512_file(f) for f in sorted((sdir / "sessions").glob("*/session.json"))]
        ml["units"] = [{"unit": s["base"] + i, "verdict": "accept" if m0["accepted"] and lv["accepted"] and not op[i] else
                        "refused: " + "; ".join(x for x in (None if m0["accepted"] else f"m0 {m0['why']}",
                                                            None if lv["accepted"] else f"lean {lv.get('why')}",
                                                            f"opening {op[i]}" if op[i] else None) if x)} for i in local]
        ml.update(session_s=session_s, timing=statement[m.name]["timing"], m0=m0["accepted"], lean=lv["accepted"],
                  prover_stopped=(live or {}).get("prover_stopped"), children_maxrss_gb=maxrss_gb(resource.RUSAGE_CHILDREN),
                  lean_why=lv.get("why"), lean_verify_s=lv.get("verify_s"), openings=verdicts[f"openings/{m.name}"])
        log(f"{m.name}: drawn {local}, M0 {m0['accepted']}, Lean {lv['accepted']} ({lv.get('why')}), openings "
            f"{verdicts[f'openings/{m.name}']['accepted']}, session {session_s} s")

    result = V.audit_record(rec, receipt, draw, rec["law"], sorted(proved), n_inst, statement, verdicts, tools.delta,
                            sessions=sessions, served=sorted(served), session_roots=session_roots)
    (out / "audit.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    leg["audit"] = {x: result.get(x) for x in ("accepted", "complete", "wrong_units")}
    leg["audit"]["failed"] = [c for c, v in (result.get("checks") or {}).items() if not v.get("ok")]
    leg["verdicts"] = {x: y.get("accepted") for x, y in verdicts.items()}
    leg["accepted"] = bool(result.get("accepted"))
    leg["seconds"] = round(time.time() - t0, 1)
    leg["_staged"] = {s["name"]: str(s["dir"]) for s in staged}
    (out / "leg.json").write_text(json.dumps(leg, indent=1, sort_keys=True, default=str))
    log(f"leg: accepted {leg['accepted']}, audit {leg['audit']}, {leg['seconds']} s")
    return leg


# --- negatives --------------------------------------------------------------------------------------------------------

def refused(leg: dict, member: str | None = None) -> bool:
    """The leg refused (not accepted), and, naming ``member``, refused at least one of its units."""
    if leg.get("accepted"):
        return False
    if member is None:
        return True
    units = leg["members"].get(member, {}).get("units")
    return units is None or any(u["verdict"] != "accept" for u in units)


def wrong_unit(honest: dict, a: str, b: str, tools: Tools, out: Path) -> dict:
    """Member ``a``'s honest session checked as member ``b``'s: the auditor's committed circuit for ``b``'s stratum is
    ``b``'s pin, which ``a``'s circuit file doesn't hash to (the proof system's own check, `proofs.verify`'s
    ``circuit``), and Lean verifies ``a``'s session against ``b``'s circuit file."""
    if "_staged" not in honest:
        return {"refused": False, "skipped": "the honest leg has no session (refused before the draw)"}
    da, db = Path(honest["_staged"][a]), Path(honest["_staged"][b])
    n = honest["members"][a]["n"]
    stage_ = out / "stage"
    shutil.rmtree(out, ignore_errors=True)
    (stage_ / "tables").mkdir(parents=True)
    os.link(db / "circuit.txt", stage_ / "circuit.txt")
    os.link(da / f"pub-{n}.bin", stage_ / f"pub-{n}.bin")
    src = Path(honest["_out"]) / f"session-{a}" / "sessions"
    shutil.copytree(src, out / "sessions", copy_function=os.link)
    pin_ok = a0.sha512_file(da / "circuit.txt") == honest["members"][b]["pin"]
    lv = a0.lean_verify(tools.lean, {"dir": stage_, "partition_file": Path(honest["_out"]) / "partition.json",
                                     "program_file": Path(honest["_out"]) / "program.json"}, n, out, zk=True)
    return {"refused": not pin_ok and a0.lean_refused(lv, "staged for another circuit"), "circuit_is_committed_pin": pin_ok,
            "lean": lv.get("accepted"),
            "lean_why": lv.get("why")}


def tampered_registered_root(honest: dict, member: str, registered: Path, tools: Tools, out: Path) -> dict:
    """Member's honest session verified against the verifier's ``--registered`` file with its first port's root changed."""
    if "_staged" not in honest:
        return {"refused": False, "skipped": "the honest leg has no session (refused before the draw)"}
    reads = json.loads(Path(registered).read_text())
    port = next(iter(reads["ports"]))
    reads["ports"][port]["root"] = "00" * 64
    out.mkdir(parents=True, exist_ok=True)
    f = out / "registered.json"
    f.write_text(json.dumps(reads))
    d = Path(honest["_staged"][member])
    n = honest["members"][member]["n"]
    shutil.rmtree(out / "sessions", ignore_errors=True)
    shutil.copytree(Path(honest["_out"]) / f"session-{member}" / "sessions", out / "sessions", copy_function=os.link)
    lv = a0.lean_verify(tools.lean, {"dir": d, "partition_file": Path(honest["_out"]) / "partition.json",
                                     "program_file": Path(honest["_out"]) / "program.json", "registered_file": f}, n, out, zk=True)
    return {"refused": a0.lean_refused(lv, "registered"), "port": port, "lean": lv.get("accepted"), "lean_why": lv.get("why")}


def negatives(members: list[Member], coins, registered: Path | None, tools: Tools, out: Path, honest: dict,
              program=None, query=None, k: int | None = None) -> dict:
    """The proof-layer negatives on the first member (and, for wrong-unit, the second), each expected refused. ``coins``
    must draw unit 0 of the first member (``"lean"`` with every unit drawn does), or ``k`` (the stratified rule's K, fewer
    than the member's units) is given and every unit is forged. ``program``/``query``: the honest leg's, when it was given
    its own."""
    m = members[0]
    neg: dict[str, dict] = {}
    kw = {"tools": tools, "program": program, "query": query, "k": k}
    forge = tuple(range(m.n)) if k is not None and k < m.n else (0,)

    for name, t in (("tampered-unit", tools), ("tampered-unit-unfirewalled", replace(tools, firewall=None) if tools.firewall else None)):
        if t is None:
            continue
        leg = prove_units([replace(m, forge=forge)], coins, registered, out=out / name, **{**kw, "tools": t})
        ml = leg["members"][m.name]
        neg[name] = {"refused": refused(leg, m.name), "units": ml.get("units"), "lean_why": ml.get("lean_why"),
                     "prover_stopped": ml.get("prover_stopped"), "audit": leg.get("audit")}

    if m.held:
        row = next(iter(m.held))
        vals = {r: np.array(v, copy=True) for r, v in m.values.items()}
        vals[row][0, 0] ^= 1
        skw = dict(m.stage_kw)
        if row in (skw.get("tables") or {}):
            t = np.array(skw["tables"][row], copy=True)
            t[skw["refs"][row][0], 0] ^= 1
            skw["tables"] = {**skw["tables"], row: t}
            vals[row] = t[np.asarray(skw["refs"][row])]
        leg = prove_units([replace(m, values=vals, stage_kw=skw)], coins, registered, out=out / "tampered-public", **kw)
        neg["tampered-public"] = {"refused": refused(leg, m.name), "row": row, "units": leg["members"][m.name].get("units"),
                                  "lean": leg["members"][m.name].get("lean"), "openings": leg["members"][m.name].get("openings")}

    def bad_root(rec: dict) -> dict:
        rec["roots"][0]["root"] = "00" * 64
        return rec

    leg = prove_units([m], coins, registered, out=out / "tampered-root", mutate=bad_root, **kw)
    neg["tampered-root"] = {"refused": refused(leg) and leg.get("refused") == "before the draw",
                            "checks_before_draw": leg.get("checks_before_draw")}

    if len(members) > 1:
        neg["wrong-unit"] = wrong_unit(honest, members[0].name, members[1].name, tools, out / "wrong-unit")
    if registered is not None:
        neg["tampered-registered-root"] = tampered_registered_root(honest, m.name, registered, tools,
                                                                   out / "tampered-registered-root")
    for name, r in neg.items():
        log(f"negative {name}: refused {r['refused']}")
    return neg


# --- the stand-in -----------------------------------------------------------------------------------------------------

def bf16(x: np.ndarray) -> np.ndarray:
    """f32 -> bf16 bits, rounded to nearest even."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)).astype(np.uint16)


def standin(n: int, ks: Sequence[int] = (128, 256), seed: int = 20261024) -> list[Member]:
    """A compiled step's matmul coordinates as stand-in units: `LlBf16DotprodDot_v1` (A8's dot product) at each K in
    ``ks``, ``n`` units each, its activation row ``x`` held by the verifier (a certified input: the I/O tie's shape) and
    its weight row ``w`` the prover's; outputs hidden."""
    from circuit_check import targets as TG

    TG.load_registries()
    rng = np.random.default_rng(seed)
    out = []
    for K in ks:
        fn = TG.definition(f"LlBf16DotprodDot_v1{{K={K},BS=128,VEC=8,FMA=1}}")
        x, w = bf16(rng.normal(0, 1, (n, K))), bf16(rng.normal(0, 0.02, (n, K)))
        out.append(Member(f"dot{K}", fn, [("x", V2, 16 * K, ["x"]), ("w", V2, 16 * K, ["w"])], [(V2, (0,))],
                          {"x": x, "w": w}, held={"x": x}))
    return out


def registered_standin(n: int, K: int, out: Path, table_rows: int = 8, seed: int = 20261024):
    """The stand-in with registered weights: a Program whose ``weights`` parameter (``table_rows`` bf16 rows of K, a
    registered value committed once, `registered.Registered.commit`) unit i reads row ``positions[i]`` of as its ``w``,
    ``x`` the unit's own and held by the verifier. Returns ``(member, program, query, registered file)``: the verifier's
    own ``--registered`` file (`registered.reads_file`, at the value's root) written under ``out``."""
    from circuit_check import targets as TG
    from verity.core.protocols.sampled_proofs.one_stage import registered as G
    from verity.ml.circuits.defs import CompositeDefinition, bind
    from verity.ml.circuits.program import Program
    from verity.ml.circuits.refs import coll_concat
    from verity.ml.circuits.types import Array

    TG.load_registries()
    fn = TG.definition(f"LlBf16DotprodDot_v1{{K={K},BS=128,VEC=8,FMA=1}}")
    (xn, xt), (wn, wt) = fn.signature()[0]
    ret = fn.signature()[1]
    rng = np.random.default_rng(seed)
    positions = [int(p) for p in rng.integers(0, table_rows, n)]

    def sig(S):
        return ((xn, Array(n, xt)), ("weights", Array(table_rows, wt))), Array(n, ret)

    def body(B, S, x, weights):
        return coll_concat(sig(S)[1], [B.call(fn, x[i], weights[p]) for i, p in enumerate(positions)])

    program = Program(bind(CompositeDefinition("ProveLegRegisteredDot", 0, (), sig, body, register=False)))
    query = P.template_instance_query(fn)
    x, weights = bf16(rng.normal(0, 1, (n, K))), bf16(rng.normal(0, 0.02, (table_rows, K)))
    reg = G.Registered.commit(P.program_digest(program), "weights", weights.tolist(), None, G.WORD_BITS, V2)
    own = G.reads_file(program, query, {"weights": K}, {"weights": reg.root.hex()}, schemas={"weights": V2})
    out.mkdir(parents=True, exist_ok=True)
    (out / "registered.json").write_text(json.dumps(own, sort_keys=True))
    table, refs = G.table(positions)
    rows, salts, entry = reg.statement(table)
    member = Member(f"dot{K}-registered", fn, [(xn, V2, 16 * K, [xn]), (wn, V2, 16 * K, [wn])], [(V2, (0,))],
                    {xn: x, wn: weights[positions]}, held={xn: x},
                    stage_kw={"tables": {xn: x, wn: np.asarray(rows, dtype=np.uint16)},
                              "refs": {xn: list(range(n)), wn: refs}, "registered": {wn: entry}, "salts": {wn: salts}})
    return member, program, query, out / "registered.json"


def n2_member(name: str, key: tuple[int, int, int], insts: Sequence, pubs: Sequence[Sequence[int]],
              oks: Sequence[int] | None = None) -> Member:
    """N2's head checks (`n2_zk`'s ``NetworkHeadCheckRows`` around `head_check.unit(L, T, COUNTS)`, ``key``) as one
    member: instance i's ``pub`` and ``wit`` rows (`n2_zk.rows_of`), ``pub`` held to ``pubs[i]`` (the verifier's own
    `n2_zk.public_words`: the registered root, the link-window, the signed digest), ``o0`` required to ``oks[i]`` (default
    every check passes)."""
    sys.path.insert(0, str(HERE.parent / "network_traces"))
    import n2_zk as N

    bits = N.row_bits(N.HC.unit(key[0], key[1], bool(key[2])))
    words = [N.rows_of(i) for i in insts]
    values = {r: np.asarray([w[r] for w in words], dtype=np.uint16) for r in N.ROWS}
    oks = [1] * len(insts) if oks is None else list(oks)
    return Member(name, N.unit(key), [(r, V2, bits[r], [r]) for r in N.ROWS], [(V2, (0,))], values,
                  held={"pub": np.asarray(pubs, dtype=np.uint16)}, expect={"o0": np.asarray(oks, dtype=np.uint16)[:, None]})


# --- served Definitions -----------------------------------------------------------------------------------------------

#: the statics of the Llama models the served path runs (Llama-3.2-1B, Llama-3.1-8B; RoPE base 500000)
LLAMA = {"1b": {"H": 2048, "I": 8192, "NH": 32, "KVH": 8, "D": 64, "V": 128256, "EPS": 1e-5},
         "8b": {"H": 4096, "I": 14336, "NH": 32, "KVH": 8, "D": 128, "V": 128256, "EPS": 1e-5}}
#: the --zk statement's least and greatest log size (`stage`)
M_MIN = 25
#: the target each attention cut's decode head is served on, by family: the RTX PRO 6000 (FA2's `Attention_v5`, whose dead-free
#: v18 `window_circuit.CUTS` cuts by `AttentionHead_v19`) and the H200 (FA3's `Attention_v4`, cut by `AttentionHead_v20`)
CUT_TARGETS = {"attention-v19": ((12, 0), 188), "attention-v20": ((9, 0), 132)}
#: an attention cut's Calls by name: its first visited key block, the full block below it and its epilogue
CALLS = {"first": "AttnBlockFirst", "block": "AttnBlock", "end": "AttnHeadEnd"}
#: an attention Call's f32 running state, each parameter's stand-in (scale, mean): the running max, the lane sums and O
STATE = {"m_prev": (2.0, 0.0), "l_prev": (4.0, 16.0), "o_prev": (4.0, 0.0), "l": (4.0, 16.0), "o": (4.0, 0.0)}
#: the window circuit's units that no serving row commits (`serving_rows.template` refuses their one-bit leaves): their
#: Definition is the window's own (`interface.resolve`), and they read no pinned table
WINDOW_ONLY = ("EmbeddingRead_v1",)


def _gen(rng: np.random.Generator, n: int, size: int, scale: float = 1.0, mean: float = 0.0) -> np.ndarray:
    return bf16(rng.normal(mean, scale, (n, size)))


def _f32(rng: np.random.Generator, n: int, size: int, scale: float = 1.0, mean: float = 0.0) -> np.ndarray:
    """``size`` f32 values a unit as u16 words, each value's little-endian bytes (a 32-bit leaf's row words)."""
    return rng.normal(mean, scale, (n, size)).astype("<f4").view("<u2").astype(np.uint16)


def _bit_words(bits: Sequence[int]) -> np.ndarray:
    """A one-bit-leaf row's u16 words: its bits in leaf order, little-endian (`interface`'s bit rows)."""
    return np.packbits(np.asarray(bits, dtype=np.uint8), bitorder="little").view("<u2").astype(np.uint16)


def attention_call(shape: str) -> Any:
    """The Call ``shape`` (``attention-v19|attention-v20:M:T:first|block|end``) names: the served decode head of T keys on its
    ``CUT_TARGETS`` target (`targets.attention_spec`, one query row), cut as `window_circuit.CUTS` cuts it, and its first
    visited block (keys (nb - 1) BN ... T - 1), the full block below it (T > BN) or its epilogue."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import dead_free as DF
    from verity_vllm import TargetProfile
    from verity_vllm.program.frontend import targets as TG
    from verity_vllm.program.predict.window_circuit import CUTS

    fam, model, T, call = shape.split(":")
    L, (cc, sms) = LLAMA[model], CUT_TARGETS[fam]
    row = TG.attention_spec(TargetProfile(compute_capability=cc, num_sms=sms), int(T), L["NH"], L["KVH"], L["D"], 1, row=0, own_rows=1)
    if row.base_id == "Attention_v5":
        row = bind(DF.AttentionV18, **row.bindings)
    cut = bind(CUTS[row.version], **row.bindings)
    if f"v{cut.version}" != fam.rsplit("-", 1)[1]:
        raise SystemExit(f"{shape}: the served head is cut by Attention_v{cut.version}, not {fam}'s")
    head = next(n.fn for n in cut.body.nodes if n.fn.base_id.rsplit("_v", 1)[0] == "AttentionHead")
    want = CALLS.get(call)
    hit = [n.fn for n in head.body.nodes if n.form == "call" and n.fn.base_id.rsplit("_v", 1)[0] == want]
    if not hit:
        raise SystemExit(f"{shape}: {head.id} has no {call!r} Call (calls: {sorted(CALLS)}; a full block needs T > BN)")
    return hit[0]


def served(shape: str) -> tuple[Any, Callable[[np.random.Generator, int], dict[str, np.ndarray]]]:
    """The served Definition ``shape`` names (``family:model[:arg]``, ``model`` a key of `LLAMA`) as the frontend binds it on
    the RTX PRO 6000 (sm_120, batch invariant, FA2 under the per-iteration Check_inf chain; an ``attention-v20`` Call on the
    H200), and its stand-in values (``rng, n -> {parameter: (n, words) u16}``):

        gemm:M[:K]           Gemm_v2{K,N=1,DOT}, the unit of a Gemm_v2 row (``SLICES``), K the hidden size by default
        attention:M:T        Attention_v5, a decode row (one query row, non-causal) over T keys
        attention-v18:M:T    Attention_v18 at Attention_v5's statics (`prune`'s rewrite of Attention_v5)
        rmsnorm-fused:M      RMSNormFusedCuda_v2{N=H,EPS}
        rmsnorm-triton:M     RMSNormTriton_v5{N=H,EPS} (`prune`'s rewrite of RMSNormTriton_v1)
        embedding:M[:V]      Embedding_v1{V,H}
        token-select:M[:V]   TokenSelect_v4{V} (`prune`'s rewrite of TokenSelect_v1)
        token-select-chunk:M[:C]    TokenSelectChunk_v2{C}, a V-chunk of TokenSelect_v4 (C-Flock's ``token-select-chunk``)
        token-select-combine:M[:V]  TokenSelectCombine_v2{V,C}, TokenSelect_v4's combine (``token-select-combine``), its
                                    partials the chunks' own of a stand-in logits row
        rope:M[:NHEADS]      RoPE_v1{NHEADS,D}, NHEADS the query heads by default
        attention-v19:M:T:CALL  a Call of the RTX PRO 6000's decode head over T keys as the window circuit cuts it
                             (`attention_call`): first (AttnBlockFirst_v1), block (AttnBlock_v5) or end (AttnHeadEnd_v1)
        attention-v20:M:T:CALL  the same on the H200 (sm_90, FA3): AttnBlockFirst_v1, AttnBlock_v4 or AttnHeadEnd_v1
        embedding-read:M[:V] EmbeddingRead_v1{V,H}, the window's embedding unit, each stand-in row opening at its token
    """
    from verity.ml.circuits.defs import bind
    from verity_vllm import TargetProfile
    from verity_vllm.program.frontend import targets as TG
    from verity.ml.definitions.registry import b1, dead_free as DF

    fam, model, *arg = shape.split(":")
    L = LLAMA[model]
    pro = TargetProfile(compute_capability=(12, 0), num_sms=188)
    a = int(arg[0]) if arg else None
    H, D, V = L["H"], L["D"], a or L["V"]
    if fam == "gemm":
        K = a or H
        return TG.gemm_spec(pro, K, 1), lambda rng, n: {"x": _gen(rng, n, K), "w": _gen(rng, n, K, 0.02)}
    if fam in ("attention", "attention-v18"):
        fn = TG.attention_spec(pro, a, L["NH"], L["KVH"], D, 1, row=0, own_rows=1)
        if fam == "attention-v18":
            fn = bind(DF.AttentionV18, **fn.bindings)
        return fn, lambda rng, n: {"q": _gen(rng, n, L["NH"] * D), "kc": _gen(rng, n, a * L["KVH"] * D),
                                   "vc": _gen(rng, n, a * L["KVH"] * D)}
    if fam == "rmsnorm-fused":
        return (TG.rmsnorm_fused_spec(pro, H, L["EPS"]),
                lambda rng, n: {"x": _gen(rng, n, H), "res": _gen(rng, n, H), "w": _gen(rng, n, H, 0.05, 1.0)})
    if fam == "rmsnorm-triton":
        return bind(DF.RMSNormTritonV5, N=H, EPS=L["EPS"]), lambda rng, n: {"x": _gen(rng, n, H), "w": _gen(rng, n, H, 0.05, 1.0)}
    if fam == "embedding":
        def emb(rng, n):
            tok = rng.integers(0, V, n, dtype=np.uint32)
            return {"tok": np.stack([tok & 0xFFFF, tok >> 16], axis=1).astype(np.uint16), "table": _gen(rng, n, V * H, 0.02)}
        return bind(b1.Embedding, V=V, H=H), emb
    if fam == "token-select":
        return bind(DF.TokenSelectV4, V=V), lambda rng, n: {"logits": _gen(rng, n, V, 3.0)}
    if fam == "token-select-chunk":
        C = a or DF.CHUNK
        return bind(DF.TokenSelectChunkV2, C=C), lambda rng, n: {"x": _gen(rng, n, C, 3.0)}
    if fam == "token-select-combine":
        from verity.ml.circuits.evaluation import evaluate_batch

        C = DF.CHUNK
        if V % C:
            raise SystemExit(f"{shape}: V = {V} is no whole number of {C}-logit chunks")

        def combine(rng, n):
            logits = _gen(rng, n, V, 3.0)
            part = np.asarray(evaluate_batch(bind(DF.TokenSelectChunkV2, C=C), logits.reshape(-1, C).astype(np.uint64)))
            idx = part[:, 1].astype(np.uint32).reshape(n, -1)
            return {"x0": logits[:, :1], "best": part[:, 0].astype(np.uint16).reshape(n, -1),
                    "idx": np.stack([idx & 0xFFFF, idx >> 16], axis=2).reshape(n, -1).astype(np.uint16)}
        return bind(DF.TokenSelectCombineV2, V=V, C=C), combine
    if fam == "rope":
        heads = a or L["NH"]

        def rope(rng, n):
            pos = rng.integers(0, 8192, (n, 1))
            ang = pos * 500000.0 ** (-np.arange(0, D, 2) / D)
            return {"x": _gen(rng, n, heads * D), "cs": bf16(np.concatenate([np.cos(ang), np.sin(ang)], axis=1))}
        return bind(b1.RoPE, NHEADS=heads, D=D), rope
    if fam in CUT_TARGETS:
        fn = attention_call(shape)

        def call(rng, n):
            out = {}
            for nm, t in fn.signature()[0]:
                words = sum(t.leaf_widths()) // 16
                out[nm] = _gen(rng, n, words) if t.width_set() == {16} else _f32(rng, n, words // 2, *STATE[nm])
            return out
        return fn, call
    if fam == "embedding-read":
        from verity.ml.commitments.gates import embedding as E
        from verity.ml.commitments.gates.frame_v3 import root_ref
        from verity.ml.commitments.gates.hm96 import reference as hm96_leaf
        from verity.ml.commitments.gates.rowleaf import reference as row_digest
        from verity.ml.commitments.merkle import merkle_depth
        from verity.ml.commitments.rowleaf import bit_row

        def emb_read(rng, n):
            names, cols = ("tok", "root", "domain", "row", "salt", "path"), []
            for _ in range(n):
                tok, row = int(rng.integers(0, V)), _gen(rng, 1, H, 0.02)[0].tolist()
                salt, dom = rng.bytes(E.SALT_BYTES), rng.bytes(64)
                path = [rng.bytes(64) for _ in range(merkle_depth(V))]
                root = root_ref(dom, E.SCHEMA, hm96_leaf(row_digest(*bit_row((row, 16))), salt)[2], tok, path)
                if E.reference(V, tok, root, dom, row, salt, path)[1] != 1:
                    raise AssertionError(f"{shape}: the stand-in row doesn't open at token {tok}")
                cols.append([_bit_words(b) for b in E.arguments(V, tok, root, dom, row, salt, path)])
            return {nm: np.stack([c[j] for c in cols]) for j, nm in enumerate(names)}
        return E.read(V, H), emb_read
    raise SystemExit(f"{shape}: no served family {fam!r} (gemm, attention, attention-v18, rmsnorm-fused, rmsnorm-triton, embedding, "
                     f"token-select, token-select-chunk, token-select-combine, rope, {', '.join(CUT_TARGETS)}, embedding-read)")


_LOWERINGS: dict = {}


@contextlib.contextmanager
def reuse_lowerings():
    """`class_statement.unit_lowering` once per unit, and `stage`'s bound `UnitStager` once per unit and partition, while
    inside: the honest leg and its negatives restage the same unit, mostly under the same partition, and neither
    `circuit.compose` nor `UnitStager.stage` changes the lowering or the stager it is given."""
    global _STAGERS
    from verity_flock import class_statement as CS

    _STAGERS = {}
    lower = CS.unit_lowering

    def memo(definition, name, in_rows, *rest, **kw):
        key = (definition.id, name, json.dumps([[r, list(ps)] for r, ps in in_rows]), repr(rest), repr(sorted(kw.items())))
        if key not in _LOWERINGS:
            _LOWERINGS[key] = lower(definition, name, in_rows, *rest, **kw)
        return _LOWERINGS[key]

    CS.unit_lowering = memo
    try:
        yield
    finally:
        CS.unit_lowering = lower
        _LOWERINGS.clear()
        _STAGERS = None
        gc.collect()


def sized(fn) -> dict:
    """`m1_families`' estimate of the unit (an upper bound on its ANDs) and M0's block for it (``rows_log``: its rows'
    in-circuit hashing alone, a lower bound whatever its ANDs)."""
    import math

    import m1_families as MF

    t = time.time()
    e = MF.estimate(fn, {}, {})
    b = MF.block(fn, e["ands"])
    return {"est_ands": e["ands"], "est_log2_ands": round(math.log2(e["ands"]), 2) if e["ands"] else None, "reads": e["reads"],
            "unpieced": e["unpieced"], **b, "refused_block": b["rows_log"] > MF.K_MAX, "k_max": MF.K_MAX,
            "seconds": round(time.time() - t, 1)}


def units_for(k_log: int, per_block: int) -> int:
    """The fewest units a statement of ``k_log`` holds at least `M_MIN` in (`stage`'s m = k_log + its block bits, at least 3)."""
    b = M_MIN - k_log
    return 1 if b <= 3 else per_block * 2 ** (b - 1) + 1


def served_member(shape: str, seed: int, n: int = 1, reads: str | None = None) -> tuple[Member, dict]:
    """``shape``'s served unit as a member of ``n`` units: its parameters the prover's rows (`interface.unit_rows`), its outputs
    hidden, values seeded by ``seed``, its table reads laid out by ``reads`` (`Member`), by default its template's own
    (`chunk_rows.reads`)."""
    from verity.ml.circuits.codec import descriptor_id
    from verity.ml.windows import interface as IF
    from verity_vllm.pipeline import chunk_rows as CR, serving_rows as SRW

    fn, gen = served(shape)
    did = descriptor_id(fn)
    if fn.base_id in WINDOW_ONLY:
        tfn, layout, own = IF.resolve(did), "window", "inline"
    else:
        tp = SRW.template(did)
        tfn, layout, own = tp.fn, tp.layout, CR.reads(tp)
    reads = reads or own
    rows, out_rows = IF.unit_rows(fn)
    values = gen(np.random.default_rng(seed), n)
    if set(values) != {nm for nm, *_ in rows}:
        raise AssertionError(f"{shape}: values for {sorted(values)}, parameters {[nm for nm, *_ in rows]}")
    info = {"descriptor_id": did, "layout": layout, "values": "stand-in: seeded inputs, outputs by the Definition's own evaluator",
            "seed": seed, "rows": [[r, b] for r, _s, b, _p in rows], "n": n, "reads": reads}
    return Member(shape.replace(":", "-"), tfn, rows, out_rows, values, reads=reads), info


def prove_served(shape: str, tools: Tools, out: Path, seed: int, reads: str | None = None) -> tuple[Member, dict, dict]:
    """``shape``'s member through `prove_units`, one unit drawn by Lean: staged at one unit, and again at `units_for`'s when
    one unit's statement is smaller than --zk admits. Call inside `reuse_lowerings`. Returns the member, its info and the leg."""
    import a11_kinds as K

    m, info = served_member(shape, seed, reads=reads)
    t = time.time()
    with K.no_cyclic_gc():
        try:
            leg = prove_units([m], "lean", tools=tools, out=out / m.name / "honest", k=1)
        except OutsideShape as ex:
            n = units_for(ex.k_log, ex.per_block)
            if n == 1:
                raise
            log(f"{shape}: k_log {ex.k_log}, {ex.per_block} a block: staged again at {n} units")
            m, info = served_member(shape, seed, n, reads)
            leg = prove_units([m], "lean", tools=tools, out=out / m.name / "honest", k=1)
    ml = leg["members"][m.name]
    info.update(ands=ml["ands"], k_log=ml["k_log"], leg_s=round(time.time() - t, 1), stage_py_maxrss_gb=maxrss_gb(resource.RUSAGE_SELF))
    return m, info, leg


def defs(shapes: Sequence[str], tools: Tools | None, out: Path, negs: bool, seed: int, max_ands: int,
         lower_refused: bool = False, reads: str | None = None) -> dict:
    """Each served shape sized, and unless only sizing (``tools`` None), proved: one unit drawn by Lean, its member staged at
    `served_member`'s units (its table reads by ``reads``, else its template's own), through `prove_units` and, with ``negs``, `negatives` at that
    draw. A shape whose rows alone pass C-Flock's block is recorded refused and not lowered (with ``lower_refused``: lowered,
    its blocker C-Flock's own refusal), nor one whose estimate passes ``max_ands``."""
    summary: dict[str, dict] = {}
    for shape in shapes:
        rec: dict[str, Any] = {"shape": shape}
        summary[shape] = rec
        try:
            from verity.ml.circuits.codec import descriptor_id

            fn, _gen = served(shape)
            rec["descriptor_id"] = descriptor_id(fn)
            rec["census"] = c = sized(fn)
            log(f"{shape}: census {c}")
            if c["refused_block"]:
                rec["blocker"] = (f"refused by C-Flock's block: its rows ({c['row_bits']} bits) need a block of 2^{c['rows_log']} "
                                  f"for their in-circuit hashing alone, past 2^{c['k_max']}")
            if c["est_ands"] > max_ands:
                rec["blocker"] = rec.get("blocker") or f"not lowered here: estimated {c['est_ands']} ANDs past this run's {max_ands}"
            elif c["refused_block"] and lower_refused:
                rec["lowered_refused"] = True
            if tools is None or ("blocker" in rec and not rec.get("lowered_refused")):
                continue
            with reuse_lowerings():
                try:
                    m, info, honest = prove_served(shape, tools, out, seed, reads)
                except ValueError as ex:
                    if "needs a block of" not in str(ex) and not isinstance(ex, OutsideShape):
                        raise
                    rec["blocker"] = "; ".join(x for x in (rec.get("blocker"), f"C-Flock's stager refused it: {ex}") if x)
                    log(f"{shape}: {rec['blocker']}")
                    continue
                rec.pop("blocker", None)
                rec.update(info)
                rec["honest"] = {x: honest.get(x) for x in ("accepted", "audit", "members", "verdicts", "seconds", "firewall",
                                                            "refused")}
                if negs:
                    rec["negatives"] = negatives([m], "lean", None, tools, out / m.name / "negatives", honest, k=1)
        except Exception as ex:  # noqa: BLE001 -- recorded per shape, the rest still run
            rec["error"] = f"{type(ex).__name__}: {ex}"[:2000]
            log(f"{shape}: {rec['error']}")
        finally:
            gc.collect()
            (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    return summary


def defs_ok(rec: dict) -> bool:
    """A shape's done-condition: its honest unit accepted and every negative refused, or a named blocker."""
    if "blocker" in rec:
        return True
    return bool((rec.get("honest") or {}).get("accepted")) and all(r["refused"] for r in (rec.get("negatives") or {}).values()) \
        and "negatives" in rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="prove_leg.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("standin", help="the stand-in members through the leg, and the negatives")
    s.add_argument("--bin", required=True)
    s.add_argument("--lean", required=True)
    s.add_argument("--firewall", default=None, help="flock-firewall: prove behind the firewall")
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--n", type=int, default=4)
    s.add_argument("--ks", default="128,256")
    s.add_argument("--negatives", action="store_true")
    s.add_argument("--registered", action="store_true",
                   help="the registered-weights stand-in (one member at the first K) in place of the population one")
    d = sub.add_parser("defs", help="vLLM's served Definitions at Llama shapes through the leg (`served`), and the negatives")
    d.add_argument("--bin", default=None)
    d.add_argument("--lean", default=None)
    d.add_argument("--firewall", default=None, help="flock-firewall: prove behind the firewall")
    d.add_argument("--out", required=True, type=Path)
    d.add_argument("--shape", action="append", required=True, help="family:model[:arg] (`served`); repeatable")
    d.add_argument("--negatives", action="store_true")
    d.add_argument("--census", action="store_true", help="size each shape only: no lowering, no proof")
    d.add_argument("--lower-refused", action="store_true",
                   help="lower a shape the census finds past C-Flock's block, for the stager's own refusal")
    d.add_argument("--seed", type=int, default=20261025)
    d.add_argument("--mem-gb", type=int, default=64, help="the job's memory: a shape estimated past mem / 700 B an AND isn't lowered")
    d.add_argument("--reads", choices=("inline", "slots"), default=None,
                   help="UnitStager's reads: a pinned MUFU table's reads inline or in lookup slots (default: the template's "
                        "own, chunk_rows.reads)")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.command == "defs":
        if not a.census and not (a.bin and a.lean):
            ap.error("defs: --bin and --lean, or --census")
        tools = None if a.census else Tools(a.bin, a.lean, a.firewall)
        summary = defs(a.shape, tools, a.out, a.negatives, a.seed, a.mem_gb * 2**30 // 700, a.lower_refused, a.reads)
        ok = all("error" not in r for r in summary.values()) if a.census else all(defs_ok(r) for r in summary.values())
        log("done: " + ("ok" if ok else "FAILED") + " " + json.dumps({k: ("blocker" if "blocker" in r else "error" if "error" in r
                                                                        else "sized" if a.census else defs_ok(r))
                                                                    for k, r in summary.items()}))
        return 0 if ok else 1
    tools = Tools(a.bin, a.lean, a.firewall)
    ks = [int(x) for x in a.ks.split(",")]
    program = query = reg = None
    if a.registered:
        m, program, query, reg = registered_standin(a.n, ks[0], a.out / "verifier")
        members = [m]
    else:
        members = standin(a.n, ks)
    honest = prove_units(members, "lean", reg, tools=tools, out=a.out / "honest", program=program, query=query)
    summary = {"honest": {x: honest.get(x) for x in ("accepted", "audit", "members", "verdicts", "registration", "seconds",
                                                    "firewall")}}
    ok = bool(honest.get("accepted"))
    if a.negatives:
        neg = negatives(members, "lean", reg, tools, a.out / "negatives", honest, program, query)
        summary["negatives"] = neg
        ok &= all(r["refused"] for r in neg.values())
    summary["ok"] = ok
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    log("done: " + ("ok" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
