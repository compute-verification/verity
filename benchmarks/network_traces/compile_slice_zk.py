"""The compile slice (`compile_slice.CompileSlice_v1`) proved in zero knowledge: corpus-smallkv's first seven steps as
one unit, proved by C-Flock `prove --zk` behind the Lean firewall and held by Lean `verify --zk`, through N2-ZK's audit
path (`n2_zk.py`: its stager, firewall and received-pieces scan).

    python benchmarks/network_traces/compile_slice_zk.py prove --bin flock-circuit --lean flock-verify --out DIR
      [--points 1] [--kinds K,...] [--dry]

* **The slice** is `verity/core/protocols/certifier/tests/fixtures/compile_slice_smallkv.json`, and `prove` first checks
  it is the corpus's (`replay_vectors.vectors` on corpus-smallkv: the arrivals, the decisions and the replay's rows;
  `cache_vectors`' tokens of each arrival).
* **The statement**: one template of n units, each the slice's rows under fresh salts, one unit a kind (`KINDS`, or
  `--kinds`' in their order). ``honest``; ``flip``, decision 5 flipped in its window row (the advice row stays the
  registered one); ``charge``, the charge one below the coded length; ``arrival``, arrival 1's prompt one token longer
  in its ``cert`` row (an arrival not in the certified record); ``table``, a made-up first cache table (request 1's
  first block under ROOT), which gives request 1 a 16-token hit at step 6; ``tokens``, request 1's first block made
  request 0's in its ``cert`` row, the same hit through a consistent table; then REDTEAM's phase 9 choices
  (`compile_slice.phase9`: a context, frequencies, a prefix hit, a decision read again, an arrival left in the queue).
  The rest are honest, enough of them that the statement fits `--zk`'s shape (`stage`). Every ``pub`` digest is the
  clear recompile's (the replay's rows, the fixture's, under the unit's digest salt) but ``table``'s, which is the
  digest of the schedule it serves (its own rows, the hit taken), so its ``cache`` check alone refuses it.
* **The bindings**, checked natively by the verifier on the row commitments b‖c the public file holds: each unit's
  ``adv`` row is row i of the advice's hiding registration (`verity.ml.commitments.advice`: its leaf from b‖c, its path
  to the registered root), and its ``cert`` row is the certified record's commitment (a stand-in: one ``cert`` row under
  the certifier's salt, b‖c as `a7.opens` recomputes it).
* **The verifier** opens each unit's ``pub`` row to its own copy (the charge and the digest) and its verdict ``o0``,
  under the salts the prover sends for those two rows; never ``adv``, ``cert`` or ``wit``. A unit passes when its
  verdict is `compile_slice.ACCEPT` and its bindings hold. What the verifier received is scanned for every piece of each
  unit's witness row, every row's salt and the digest's salt (`a8.received`), less the pieces the public circuit holds
  and the sparse ones (`sparse_pieces`); the prover's instance file, which holds the witness rows, is the control.
* **Cost** (``--points``): sessions whose verifier draws the first k units, behind the firewall (`a8.cost`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import n2_zk as N2  # noqa: E402
from verity.core.protocols.certifier import cache_step as CST  # noqa: E402
from verity.core.protocols.certifier import compile_slice as CS  # noqa: E402
from verity.core.protocols.certifier import replay_chain as RC  # noqa: E402
from verity.ml.commitments import advice as ADV  # noqa: E402
from verity.ml.commitments.frame_v3 import BitRowLeaf  # noqa: E402
from verity.ml.commitments.rowleaf import bit_row, row_leaf  # noqa: E402

FIXTURE = REPO / "verity/core/protocols/certifier/tests/fixtures/compile_slice_smallkv.json"
KINDS = ("honest", "flip", "charge", "arrival", "table", "tokens") + CS.PHASE9
REFUSED = {"honest": set(), "flip": {"load", "steps", "rows"}, "charge": {"charge"}, "arrival": {"rows"},
           "table": {"cache"}, "tokens": {"rows"}, "context": {"code", "charge"}, "frequency": {"code", "charge"},
           "hit": {"cache", "rows"}, "reread": {"steps", "rows"}, "untaken": {"steps", "rows"}}
OPENED = ("o0", "pub")


def log(msg: str) -> None:
    print(f"CSZK {time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def load(path: Path = FIXTURE) -> SimpleNamespace:
    fx = json.loads(path.read_text())
    st = CS.statics(fx["R"], fx["W"], fx["A"], fx["engine"], fx["IW"], Q=fx["Q"], D=fx["D"], P=fx["P"], K=fx["K"],
                    SEG=fx["SEG"], CN=fx["CN"], CTW=fx["CTW"], CU=fx["CU"])
    sl = CS.Slice(tuple(map(tuple, fx["arrivals"])), tuple(fx["decisions"]), fx["steps"], tuple(map(tuple, fx["tokens"])))
    return SimpleNamespace(fx=fx, st=st, S=SimpleNamespace(**st), sl=sl, rows=[tuple(map(tuple, r)) for r in fx["rows"]])


def _corpus(tmp: Path) -> Path:
    """corpus-smallkv unpacked as ENC-DERIVED's tests read it (`benchmarks/network_e2e/encodings/tests/test_v1.py`'s
    `_corpus`, which imports pytest): each batch in Model Runner V2's order, each stop-id finish marked `eos`."""
    import gzip
    import shutil

    src = REPO / "benchmarks/network_e2e/encodings/tests/fixtures/corpus-smallkv"
    for f in ("meta.json", "server.log"):
        shutil.copy(src / f, tmp / f)
    out = []
    for line in gzip.decompress((src / "steps.jsonl.gz").read_bytes()).decode().splitlines():
        s = json.loads(line)
        if s.get("batch"):
            sched = ([r for r in s["cached"] if not r["resumed"]] + [r for r in s["cached"] + s["new"] if r["resumed"]]
                     + [r for r in s["new"] if not r["resumed"]])
            s["batch"] = [r["id"] for r in sorted((r for r in sched if r["id"] in set(s["batch"])),
                                                  key=lambda r: r["num_scheduled_tokens"])]
        out.append(json.dumps(s) + "\n")
    (tmp / "steps.jsonl").write_text("".join(out))
    reqs = [json.loads(x) for x in gzip.decompress((src / "requests.jsonl.gz").read_bytes()).decode().splitlines()]
    (tmp / "requests.jsonl").write_text("".join(json.dumps({**q, "eos": q["finish_reason"] == "stop"}) + "\n" for q in reqs))
    return tmp


def corpus_check(x: SimpleNamespace) -> None:
    """The fixture is corpus-smallkv's first P steps: its arrivals, the decisions its replay asks, the replay's rows and
    each arrival's tokens (`cache_vectors`' trace, `derived.tokens_of`)."""
    import importlib
    import tempfile

    enc = REPO / "benchmarks/network_e2e/encodings"
    names = ("replay", "replay_vectors", "cache_vectors", "enc_derived")
    saved, path = {k: sys.modules.pop(k) for k in names if k in sys.modules}, list(sys.path)
    sys.path[:0] = [str(enc)]
    try:
        RP, RV, CV = (importlib.import_module(k) for k in names[:3])
    finally:
        sys.path[:] = path
        for k in names:
            sys.modules.pop(k, None)
        sys.modules.update(saved)
    t = RP.read_corpus_trace(_corpus(Path(tempfile.mkdtemp())))
    _st, cases = RV.vectors(t, R=x.S.R, W=x.S.W, A=x.S.A)
    reqs = t.requests
    arrivals = [(i, reqs[i].prompt_len, reqs[i].output_len) for i in range(x.S.K)]
    dec = [b for _s, inp, want in cases[:x.S.P] for b in inp.bits[:want.used]]
    rows = [tuple(map(tuple, c[2].rows)) for c in cases[:x.S.P]]
    tr = CV.load(enc / "tests/fixtures/corpus-smallkv")
    adv, order = CV.D.advice_of(tr)
    toks = CV.D.tokens_of(CV.D.holdings_of(tr), order, [q.prompt_len for q in adv.requests])
    tokens = [tuple(toks[i]) for i, _P, _O in arrivals]
    if (arrivals, dec, rows, tokens) != (list(x.sl.arrivals), list(x.sl.decisions), x.rows, list(x.sl.tokens)):
        raise SystemExit("the fixture isn't corpus-smallkv's first steps")


def cases(x: SimpleNamespace, n: int, secret: bytes, kinds: tuple[str, ...] = KINDS) -> list[SimpleNamespace]:
    """The statement's n units (the module's docstring): kind, the unit's rows and the reference's verdict."""
    out, red = [], CS.phase9(x.S, x.sl)
    for i in range(n):
        kind = kinds[i] if i < len(kinds) else "honest"
        salt = hashlib.sha512(secret + b"digest salt" + i.to_bytes(4, "big")).digest()[:CS.SALT]
        digest = CS.rows_digest(x.S, x.rows, salt)
        kw = dict(red.get(kind, {}))
        if kind == "flip":
            flip = list(x.sl.decisions)
            flip[5] ^= 1
            kw["window"] = flip
        elif kind == "charge":
            kw["charge"] = CS.case(x.S, x.sl, salt).length - 1
        elif kind == "arrival":
            cert = list(x.sl.arrivals)
            cert[1] = (cert[1][0], cert[1][1] + 1, cert[1][2])
            kw["cert"] = cert
        elif kind == "table":
            kw["fabricate"] = {0: (CST.Entry(x.S.CN, tuple(x.sl.tokens[1][:x.S.B]), 0),)}
            digest = None
        elif kind == "tokens":
            toks = [list(t) for t in x.sl.tokens]
            toks[1][:x.S.B] = toks[0][:x.S.B]
            kw["tokens"] = toks
        c = CS.case(x.S, x.sl, salt, digest=digest, **kw)
        out.append(SimpleNamespace(kind=kind, case=c, salt=salt, words={r: CS.words(c.rows[r]) for r in CS.ROWS}))
    return out


def stager(M, x: SimpleNamespace, n: int, out: Path) -> SimpleNamespace:
    """The statement's program and partition (one template, n units) and its stager, the unit lowered by its builder."""
    defn = CS.unit(**x.st)
    bits = CS.row_bits(x.S)
    rows = [(r, M.V2, bits[r], [r]) for r in CS.ROWS]
    program = M.P.mixed_population_program([(defn, n)])
    query = M.P.template_instances_query([defn])
    part = M.P.partition(program, query)
    out.mkdir(parents=True, exist_ok=True)
    (out / "partition.json").write_text(json.dumps(part, sort_keys=True))
    (out / "program.json").write_bytes(M.canonical_json(M.encode_program(program)))
    bound = {"partition": M.P.digest(part), "program": part["program"]}
    t = time.time()
    st = M.CS.UnitStager(defn, "unit/compile-slice", rows, [(M.V2, (0,))], circuit=out / "circuit.txt", bind=bound)
    st.pin
    (tid, base, cnt), = M.P.ranges(program, query)
    assert tid == M.P.template_id(defn) and cnt == n
    return SimpleNamespace(defn=defn, rows=rows, program=program, query=query, n=n, base=base, st=st, out=out, part=part,
                           lower_s=round(time.time() - t, 1), partition_file=out / "partition.json",
                           program_file=out / "program.json")


def population(meta: dict, kinds: tuple[str, ...]) -> int:
    """The fewest units whose statement fits `--zk`'s shape (k_log + nbl ≥ 25, `n2_zk.stage`), at least one unit a kind."""
    b = 25 - meta["k_log"]
    return len(kinds) if b <= 3 else max(len(kinds), meta["vus_per_block"] * ((1 << (b - 1)) + 1))


def certified(M, port, words: list[int], salt: bytes) -> bytes:
    """The certified record's commitment b‖c (the stand-in: the honest ``cert`` row under the certifier's salt)."""
    return M.CS.FC._commit_strings([port], {port.name: M.np.asarray([words], dtype=M.np.uint16)}, {port.name: [salt]})[port.name][0]


def advice_leaf_ok(reg: ADV.Registration, binding: bytes, n: int, i: int, bc: bytes, n_bits: int) -> bool:
    """The verifier's check of unit i's ``adv`` row: its leaf from the proof's b‖c, walked to the registered root."""
    d = ADV.domain(binding, n)
    kind = BitRowLeaf(n_bits)
    leaf = row_leaf(d, i, ADV.INST.tree_leaf(ADV.INST.default_key, bc), kind.hiding_schema)
    return ADV._scheme(binding, n, n_bits).verify_path(ADV.PORT, reg.root, n, i, leaf, reg.open(i).path)


def stage(M, sg: SimpleNamespace, units: list[SimpleNamespace], salt_key: bytes, salts: dict) -> dict:
    t0 = time.time()
    st, n, out = sg.st, len(units), sg.out / "stage"
    k_log = st.meta["k_log"]
    nbl = max(3, (-(-n // st.meta["vus_per_block"]) - 1).bit_length())
    if not 25 <= k_log + nbl <= M.a7.M_MAX:
        raise SystemExit(f"m = {k_log} + {nbl} is outside 25 .. {M.a7.M_MAX} (--zk and flock-circuit's shape)")
    data = {r: M.np.asarray([u.words[r] for u in units], dtype=M.np.uint16) for r in CS.ROWS}
    set_name = f"{st.iset.name}-slice"
    got = st.stage(data, out, salt_key, salts=salts, indices=range(sg.base, sg.base + n), set_name=set_name)
    (out / "inst.bin").rename(out / f"inst-{n}.bin")
    (out / "pub.bin").rename(out / f"pub-{n}.bin")
    (out / "tables").mkdir(exist_ok=True)
    os.link(sg.out / "circuit.txt", out / "circuit.txt")
    m = {"tag": "slice", "name": "slice", "dir": out, "meta": st.meta, "pin": st.pin, "class": st.class_sha512,
         "set": set_name, "n": n, "base": sg.base, "defn": sg.defn, "ands": got["ands"], "unit_nnz": got["unit_nnz"],
         "salts": got["salts"], "ports": st.ports, "oks": [int(v) for v in M.np.asarray(got["outs"]["o0"])[:, 0]],
         "partition_file": sg.partition_file, "program_file": sg.program_file, "stage_s": round(time.time() - t0, 1)}
    m["header"] = M.a0.public_header(out, n)
    return m


def audit(M, a, sg, m: dict, units: list[SimpleNamespace], binds: dict, out: Path, port: int) -> dict:
    """The statement registered, every unit drawn by the verifier's Lean, proved --zk behind the firewall and held by
    Lean; then the verifier's side per unit: ``pub`` opened to its own copy, the verdict read, the bindings checked on
    b‖c (`advice_leaf_ok`, `certified`). Writes what the verifier received under ``out / "verifier"``."""
    a0, a7, P, R, V = M.a0, M.a7, M.P, M.R, M.V
    out.mkdir(parents=True, exist_ok=True)
    n, name = m["n"], m["name"]
    want, roots = {}, []
    fv = m["header"]["frame_v3"]
    for p_, w in a7.frame_ports(m["set"], m["meta"], n).items():
        want[f"{name}/{p_}"] = w
        roots.append({"port": f"{name}/{p_}", "root": fv["roots"][p_], "leaves": w["leaves"], "domain": fv["domain_ids"][p_]})
    leaf_layer = hashlib.sha512(M.canonical_json({name: a0.sha512_file(m["dir"] / f"pub-{n}.bin")})).hexdigest()
    law = M.D.parse_law(f"stratified:{n}")
    win = {"kind": "stand-in", "of": "the compile slice on corpus-smallkv's first seven steps (compile_slice_zk.py)",
           "members": {name: {"template": P.template_id(m["defn"]), "units": [m["base"], m["base"] + n]}},
           "link": "compile_slice: each unit's adv row is the advice registration's row, its cert row the certified record's"}
    rec = R.record(sg.program, sg.query, law, a0.SCHEME, roots, win, leaf_layer=leaf_layer)
    (out / "registration.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    checks = {"registration": V.check(rec, sg.program, sg.query, law, a0.SCHEME, want, leaf_layer=leaf_layer).to_json(),
              f"statement/{name}": a7.statement_checks(m, rec, want)}
    res: dict = {"checks_before_draw": {c: v["ok"] for c, v in checks.items()}}
    if not all(v["ok"] for v in checks.values()):
        return {**res, "checks": checks, "accepted": False}
    p = a0.run([a.lean, "draw", "--program", str(sg.program_file), "--partition", str(sg.partition_file), "--stratified",
                str(n)], out / "draw.log")
    lean_draw = json.loads(next(x for x in p.stdout.splitlines() if x.startswith("DRAW ")).split(" ", 1)[1])
    local = sorted(u - m["base"] for u in lean_draw["units"] if m["base"] <= u < m["base"] + n)
    sdir = out / "session"
    record, live, proc = M.a8.session(a, m, sdir, local, port, env=N2.firewall(M, m))
    draw_t = (record or {}).get("unit_draw") or {}
    v = V.check_draw(draw_t, {"population": n}, {"law": "subset", "k": len(local)})
    verdicts = {f"m0-verifier/{name}": {"accepted": ((record or {}).get("verdict") or {}).get("accepted") is True
                                        and (live or {}).get("roots") == fv["roots"] and v.ok,
                                        "why": ((record or {}).get("verdict") or {}).get("why"), "draw_check": v.to_json()}}
    lv = a0.lean_verify(a.lean, m, n, sdir, zk=True)
    verdicts[f"lean-verify/{name}"] = {**lv, "tool": "flock-verify verify --zk --archive --partition --program"}
    proved = a0.proved_units(m["header"], live or {})
    timing = {k: (live or {}).get(k) for k in N2.TIMING}
    statement = {name: {"session_s": proc["wall_s"], **{k: (live or {}).get(k) for k in ("circuit_sha512", "statement_digest",
                                                                                         "sigma")},
                        "timing": timing, "process": proc, "lean_verify_s": lv.get("verify_s")}}
    sessions = [a0.sha512_file(f) for f in sorted((sdir / "sessions").glob("*/session.json"))]
    result = V.audit_record(rec, R.receipt(rec, "one-stage-e2e verifier (compile_slice_zk.py)"), lean_draw, rec["law"],
                            sorted(proved), (live or {}).get("instances", 0), statement, verdicts, a.delta, sessions=sessions,
                            served=sorted(m["base"] + u for u in draw_t.get("units", [])),
                            session_roots={f"{name}/{p_}": r for p_, r in ((live or {}).get("roots") or {}).items()})
    (out / "audit.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))

    held = set(proved) if result.get("accepted") else set()
    port_ = {x.name: x for x in m["ports"]}
    bcs = a7.public_bcs(m["dir"] / f"pub-{n}.bin", m["ports"])
    rows, sent = [], []
    for i, u in enumerate(units):
        word = m["oks"][i]
        opened = ((m["base"] + i) in held and a7.opens(port_["pub"], u.words["pub"], m["salts"]["pub"][i], bcs[i]["pub"])
                  and a7.opens(port_["o0"], [word], m["salts"]["o0"][i], bcs[i]["o0"]))
        adv_ok = advice_leaf_ok(binds["advice"], binds["binding"], n, i, bcs[i]["adv"], binds["n_bits"])
        cert_ok = bcs[i]["cert"] == binds["certified"]
        verdict = ("malformed" if not opened else "accept" if word == CS.ACCEPT and adv_ok and cert_ok else "reject")
        rows.append({"unit": m["base"] + i, "kind": u.kind, "verdict_word": word, "flags": u.case.flags, "adv_bound": adv_ok,
                     "cert_bound": cert_ok, "verdict": verdict})
        sent.append({"unit": m["base"] + i, "verdict": word, **{f"salt/{r}": m["salts"][r][i].hex() for r in OPENED}})

    vdir = out / "verifier"
    vdir.mkdir(exist_ok=True)
    for f in (out / "registration.json", sg.partition_file, sg.program_file, m["dir"] / "circuit.txt",
              m["dir"] / f"pub-{n}.bin", sdir / "draw.json", out / "audit.json"):
        os.link(f, vdir / f.name)
    for f in sorted((sdir / "sessions").rglob("*")):
        if f.is_file():
            os.link(f, vdir / f"session-{f.parent.name}-{f.name}")
    (vdir / "openings.json").write_text(json.dumps(sent, indent=1))
    (vdir / "bindings.json").write_text(json.dumps({"advice_root": binds["advice"].root.hex(),
                                                    "certified": binds["certified"].hex(),
                                                    "paths": [[s.hex() for s in binds["advice"].open(i).path] for i in range(n)]},
                                                   indent=1))
    proofs = (live or {}).get("proof_bytes") or []
    return {**res, "accepted": bool(result.get("accepted")), "units": rows,
            "audit": {k: result.get(k) for k in ("accepted", "complete", "wrong_units")},
            "verdicts": {k: y.get("accepted") for k, y in verdicts.items()}, "why": {k: y.get("why") for k, y in verdicts.items()},
            "timing": statement[name], "lean_verify_s": lv.get("verify_s"),
            "per_unit": {"prove_total_s": N2._per(timing.get("prove_total_s"), n), "e2e_s": N2._per(timing.get("e2e_s"), n),
                         "proof_bytes": N2._per(sum(proofs) if proofs else None, n),
                         "lean_verify_s": N2._per(lv.get("verify_s"), n)},
            "opened_rows": list(OPENED)}


def blobs_of(units: list[SimpleNamespace], salts: dict) -> tuple[dict[str, bytes], set[str]]:
    """The witness pieces the verifier must not receive: each unit's ``wit`` row as the instance file holds it (u16
    little-endian), its ``adv``, ``cert`` and ``wit`` salts and its digest salt; the rows are what the control holds."""
    blobs, held = {}, set()
    for i, u in enumerate(units):
        blobs[f"wit {i}"] = b"".join(w.to_bytes(2, "little") for w in u.words["wit"])
        held.add(f"wit {i}")
        blobs[f"digest salt {i}"] = u.salt
        for r in ("adv", "cert", "wit"):
            blobs[f"{r} salt {i}"] = salts[r][i]
    return blobs, held


def sparse_pieces(pieces: dict[str, bytes], held: set[str]) -> tuple[dict[str, bytes], set[str], list[str]]:
    """`pieces` less those with at most one nonzero byte, the held ones among the rest, and the left-out pieces' hex. A
    proof's encoding holds such runs whatever the witness (a small integer in a zero-padded field: 92 of the 4,080 in one
    proof stream of this slice), so finding one is no evidence; the cache rows, mostly zero, have a few."""
    out = {k for k, x in pieces.items() if sum(map(bool, x)) <= 1}
    return {k: x for k, x in pieces.items() if k not in out}, held - out, sorted(pieces[k].hex() for k in out)


def expected(kind: str, flags: dict) -> bool:
    return {f for f, ok in flags.items() if not ok} == REFUSED[kind]


def prove(a) -> int:
    x = load()
    corpus_check(x)
    log(f"slice: corpus-smallkv's first {x.S.P} steps, {x.sl.arrivals}, decisions {x.sl.decisions}, the fixture is the corpus's")
    summary: dict = {"statics": x.st, "row_bits": CS.row_bits(x.S), "slice": x.fx, "kinds": a.kinds}
    secret = os.urandom(32)
    if a.dry:
        units = cases(x, len(a.kinds), secret, a.kinds)
        summary["reference"] = [{"kind": u.kind, "verdict": u.case.verdict, "flags": u.case.flags} for u in units]
        log(f"dry: {summary['reference']}")
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
        return 0 if all(expected(u.kind, u.case.flags) for u in units) else 1
    M = N2._one_stage()
    out = Path(a.out)
    CS.CompileSlice.build = CS._build
    probe = stager(M, x, len(a.kinds), out / "probe")
    n = population(probe.st.meta, a.kinds)
    log(f"probe: k_log {probe.st.meta['k_log']}, vus_per_block {probe.st.meta['vus_per_block']}, lowered in {probe.lower_s} s; "
        f"{n} units")
    sg = probe if n == len(a.kinds) else stager(M, x, n, out / "stager")
    summary["stager"] = {"lower_s": sg.lower_s, "k_log": sg.st.meta["k_log"], "vus_per_block": sg.st.meta["vus_per_block"],
                         "pin": sg.st.pin, "rows": {r: b for r, _s, b, _p in sg.rows}, "n": n}
    units = cases(x, n, secret, a.kinds)
    for u in units:
        if not expected(u.kind, u.case.flags):
            raise SystemExit(f"{u.kind}: the reference's flags {u.case.flags} are not the kind's")
    salt_key = hashlib.sha512(secret + b"row salts").digest()[:32]
    SB = M.CS.FC.SALT_BYTES
    cert_salt = os.urandom(SB)
    salts = {"adv": [os.urandom(SB) for _ in range(n)], "cert": [cert_salt] * n}
    port_ = {p.name: p for p in sg.st.ports}
    honest = units[0].words
    n_bits = 16 * len(honest["adv"])
    binding = bytes.fromhex(M.P.digest(sg.part))
    reg = ADV.register([bit_row((u.words["adv"], 16))[0] for u in units], n_bits, binding, salts["adv"])
    binds = {"advice": reg, "binding": binding, "n_bits": n_bits, "certified": certified(M, port_["cert"], honest["cert"], cert_salt)}
    m = stage(M, sg, units, salt_key, salts)
    refs = [u.case.verdict for u in units]
    if m["oks"] != refs:
        raise SystemExit(f"the staged verdict words {m['oks']} are not the reference's {refs}")
    bcs = M.a7.public_bcs(m["dir"] / f"pub-{n}.bin", m["ports"])
    same = all(bcs[i]["adv"] == ADV.INST.commit_string(ADV.INST.default_key,
                                                       BitRowLeaf(n_bits).layout().digest(bit_row((u.words["adv"], 16))[0]),
                                                       salts["adv"][i]) for i, u in enumerate(units))
    summary["staged"] = {"n": n, "ands": m["ands"], "unit_nnz": m["unit_nnz"], "stage_s": m["stage_s"], "k_log": m["meta"]["k_log"],
                         "pin": m["pin"], "verdict_words": m["oks"], "adv_bc_is_the_advice_leafs": same}
    log(f"staged: {n} units, {m['ands']} ANDs a unit, k_log {m['meta']['k_log']}, verdicts {m['oks'][:len(a.kinds)]}..., "
        f"adv b‖c is advice's: {same}, {m['stage_s']} s")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))

    av = SimpleNamespace(bin=a.bin, lean=a.lean, lease=[], gpu=False, delta=a.delta, points=a.points, arms=["fw"], cold=1)
    res = audit(M, av, sg, m, units, binds, out / "slice", a.port)
    blobs, held, public = N2.public_pieces(M.a8, *blobs_of(units, m["salts"]), (m["dir"] / "circuit.txt").read_bytes())
    blobs, held, sparse = sparse_pieces(blobs, held)
    res["received"] = {**M.a8.received(out / "slice" / "verifier", m["dir"] / f"inst-{n}.bin", blobs, held),
                       "public_pieces": public, "sparse_pieces": sparse}
    summary["audit"] = res
    want = ["accept" if u.kind == "honest" else "reject" for u in units]
    got = [r["verdict"] for r in res.get("units", [])]
    ok = (res["accepted"] and all(res["verdicts"].values()) and got == want and same
          and not res["received"]["verifier_found"] and res["received"]["control"]["holds_all"])
    log(f"proofs {res['verdicts']}, audit {res['audit']}, first units {[(r['kind'], hex(r['verdict_word']), r['adv_bound'], r['cert_bound'], r['verdict']) for r in res.get('units', [])[:len(a.kinds)]]}, "
        f"per unit {res['per_unit']}, verifier received {res['received']['verifier_bytes']} bytes, witness pieces found "
        f"there: {res['received']['verifier_found'] or 'none'} (of {len(blobs)}, {len(sparse)} sparse left out), "
        f"control {res['received']['control']['blobs_found']}; "
        f"{'as expected' if ok else 'UNEXPECTED'}")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    if a.points:
        rows = M.a8.cost(av, m, out / "cost", a.port + 20)
        summary["cost"] = rows
        ok &= all(r["accepted"] for r in rows) and all(r["lean"]["accepted"] for r in rows if "lean" in r)
    summary["ok"] = ok
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    log("done: " + ("ok" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="compile_slice_zk.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prove", help="the slice's units proved --zk and held by Lean")
    p.add_argument("--bin")
    p.add_argument("--lean")
    p.add_argument("--out", required=True)
    p.add_argument("--port", type=int, default=7781)
    p.add_argument("--delta", type=float, default=2.0 ** -20)
    p.add_argument("--points", default="", help="cost points: the k drawn units of each session, comma-separated")
    p.add_argument("--kinds", default=",".join(KINDS),
                   help="the units' kinds in order, comma-separated, honest first (its cert row is the certified record's)")
    p.add_argument("--dry", action="store_true", help="the slice and the reference only: nothing lowered or proved")
    a = ap.parse_args(argv)
    a.points = [int(k) for k in a.points.split(",") if k]
    a.kinds = tuple(k for k in a.kinds.split(",") if k)
    if not a.kinds or a.kinds[0] != "honest" or not set(a.kinds) <= set(KINDS):
        raise SystemExit(f"--kinds: honest first, each of {', '.join(KINDS)}")
    if not a.dry and not (a.bin and a.lean):
        raise SystemExit("prove needs --bin and --lean, or --dry")
    return prove(a)


if __name__ == "__main__":
    sys.exit(main())
