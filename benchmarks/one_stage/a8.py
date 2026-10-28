"""One-stage audit, run A8: one window of PoUB over a registered store of a few thousand chunks, its
openings proved in zero knowledge through the proof system in place of opened (PROTOCOL.md §6's checks 1-3 and 5, as §8
deploys them), so that the verifier receives no chunk bytes; and the prover's cost per opening.

    python benchmarks/one_stage/a8.py --bin flock-circuit --lean flock-verify --out DIR [--blocks 4096] [--plan 64]
      [--samples 16] [--points 1,4,16] [--arms nofw,fw] [--cold 1] [--gpu [--lease CMD]] [--no-audit]

* **Store and window.** ``--blocks`` N seeded payloads (each 64-byte word's top byte zero, p512's words) registered under
  p512/k2 (`a7.P512`: the opening's proof block at the default p1024/k3/masked is past C-Flock's K_MAX), sketch-sha512
  receipts. One window of N / ``--plan`` passes, pass i a stripe of P chunks from P i, so the window's U is N receipts, of
  which its openings coin draws ``--samples`` (`audit.Session.close`). Every salt of the prover's (the payloads', the
  receipt leaves', every row's) comes from its own secret, drawn per run from the OS.
* **Each opening a proof block.** Draw (pass i, position p, chunk j) is one `BlockOpeningCheck_v1` (`block_check`) at P,
  DR (the pass's receipt tree), DP (the payload tree of N) and DV 1 (no appended chunk), its witness `opening.witness`:
  the chunk, its payload, the receipt leaf, the pass's plan row, both salts and paths. The window's proof blocks are one
  statement of the one-stage audit (`one_stage.registration`, the template-instance partition, the stratified sampling
  design, every proof block drawn by the verifier's Lean), proved by `flock-circuit prove --zk` and held by Lean
  `verify --zk --partition --program`.
* **What the verifier opens.** Of each proof block its public rows (``pub`` and ``signs``, which it derives itself:
  `opening.public`) and its verdict row ``o0``, under the salts the prover sends for those three rows. It holds the
  proof block's plan row by ``b || c`` to its own copy of the pass's plan: v0's plans are in the clear (§8, "What the
  plan shows"), so the prover sends the plan row's salt with the plan. It decides the window on the verdicts
  (`opening.Session.decide`).
* **Rounds.** ``honest``, and ``tampered``: the same session (the same coins, so the same draws), the store's chunk at the
  first draw replaced by other bytes before the passes (a prover that lost the chunk and receipts what it holds in its
  place). Its proof block's verdict is codec, its proof is accepted, and the window is rejected. Both rounds are staged on
  one stager, so the circuit is lowered once.
* **What the verifier received** (`scan`): every file the verifier side holds (the registration, partition, program,
  circuit, public file, the session's record and proofs, the draw, the openings it was sent and v0's messages), scanned
  at every offset, raw and as hex, for each drawn chunk and payload cut whole into 16-byte pieces (`leak_scan.cut`, its
  tail included), but the pieces the staged public circuit holds (a run of zero bytes), which it lists. A round fails on
  a piece the zero-knowledge drivers' one rule reads ``leaked`` (`leak_scan.scan`). The prover's input file
  (``inst-*.bin``) is the positive control.
* **Cost** (``--points``): sessions on the honest statement whose verifier draws its first k proof blocks, cold (one prover
  process each, ``--cold`` of them), in each of ``--arms`` (``fw``: the firewall's send check, `FC_FIREWALL=1`, the design's
  path, which rechecks a host witness; ``nofw``), on the device with ``--gpu``: the process's wall and CPU seconds, and the
  session's ``e2e_s``, ``wait_s`` (the prover waiting on the verifier), ``witness_s`` and ``prove_total_s``. ``--no-audit``
  stages only the honest statement and runs the points, Lean verifying each arm's session at the largest k. ``--lease``
  is a command each prover process runs under (`a8_gpu.sh`'s GPU lease), so the device is held for the proofs alone; the
  process's wall then includes the lease's wait, and the session's own times don't.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import resource
import shlex
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "network_traces"))

import a0  # noqa: E402
import a7  # noqa: E402
import leak_scan as LS  # noqa: E402
from verity.core.protocols.remote_bandwidth import audit, codec, coins, opening as C3, receipts  # noqa: E402
from verity.core.protocols.remote_bandwidth.receipts import Plan  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import draw as D, partition as P, registration as R, verifier as V  # noqa: E402
from verity.ml.circuits.codec import canonical_json, encode_program  # noqa: E402
from verity.ml.circuits.defs import bind  # noqa: E402
from verity.ml.commitments.gates import block_check as BC  # noqa: E402
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2  # noqa: E402
from verity_flock import circuit as FC, class_statement as CS  # noqa: E402

NODE, SOURCE = b"a8-node", b"a8 verifier's secret"
P512 = a7.P512
WORD = P512.field.word_bytes
PIECE = 16
OPENED = ("o0", "pub", "signs")


def log(msg: str) -> None:
    print(f"A8 {time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


# --- the remote session -------------------------------------------------------------------------------------------------------

def store_of(blocks: int, seed: int, secret: bytes) -> C3.Store:
    """The registered store: ``blocks`` seeded payloads, each 64-byte word's top byte zero, under salts from ``secret``."""
    rng = random.Random(seed)
    payloads = [b"".join(rng.randbytes(WORD - 1) + b"\0" for _ in range(codec.BLOCK_BYTES // WORD)) for _ in range(blocks)]
    salt = a7.Entropy(secret + b"payload salts")
    ys = [salt(receipts.SALT_BYTES) for _ in range(blocks)]
    root = audit.payload_tree(NODE, payloads, ys).commitment.root
    return C3.Store(audit.register(SOURCE, NODE, blocks, root, P512, nonce=bytes(32)), payloads, ys)


def window(store: C3.Store, L: int, samples: int, delta: float, secret: bytes, seed: int) -> SimpleNamespace:
    """One window over ``store``: N / L passes, pass i the stripe of L chunks from L i, each received as the prover sends
    it (its root, its plan, its step i), then the window closed and drawn. Returns the session, the passes and v0's
    messages, as the verifier received them."""
    reg, N = store.registration, store.registration.blocks
    p = audit.Params(blocks=N, deadline_ns=10**6, windows=1, samples=samples, delta=delta, window_ns=10, codec=P512)
    clock = a7.Clock()
    s = C3.Session(p, reg, None, coins.local(reg.digest, 0, p.windows, a7.Entropy(b"a8 coins %d" % seed)),
                   not_before_ns=0, clock=clock)
    prover = audit.Prover(store, None, s.commitment, entropy=a7.Entropy(secret + b"receipt salts"))
    challenge, previous, passes, msgs = s.start(), reg.digest, {}, []
    for i in range(N // L):
        runs = ((L * i, L),)
        ps = prover.run(challenge, i, previous, Plan(runs, N), head=store.head)
        v = s.receive(0, i, ps.root, runs, step=i)
        if v is not None:
            raise SystemExit(f"pass {i} refused: {v}")
        passes[0, i], previous = ps, ps.root
        msgs.append({"window": 0, "pass": i, "root": ps.root.hex(), "runs": [list(r) for r in runs], "step": i})
    clock.now += p.window_ns
    s.boundary()
    clock.now = s.deadline(0) + 1
    got = s.close(0)
    if isinstance(got, audit.Verdict):
        raise SystemExit(f"the window was refused at its close: {got}")
    return SimpleNamespace(session=s, store=store, passes=passes, messages=msgs, params=p)


def tampered(store: C3.Store, j: int) -> C3.Store:
    """``store`` with chunk j replaced by other bytes (every word still below p): a prover that lost it."""
    out = copy.copy(store)
    out.blocks = list(store.blocks)
    out.blocks[j] = hashlib.shake_256(b"a8 tampered block %d" % j).digest(codec.BLOCK_BYTES)
    return out


def inputs(rs: SimpleNamespace, L: int, secret: bytes) -> SimpleNamespace:
    """What the proof blocks read beside the session: each pass's plan row (its L chunks) under the prover's plan salt,
    and the ``b || c`` the verifier computes from its own copy of the plan under the salt the prover sends with it; the
    statics."""
    N = rs.store.registration.blocks
    salt = a7.Entropy(secret + b"plan salts")
    rows = {i: list(range(L * i, L * (i + 1))) for i in range(N // L)}
    salts = {i: salt(receipts.SALT_BYTES) for i in rows}
    S = C3.statics(P512, receipts.PER_LEAF, L, C3.depth(receipts.leaves(L)), C3.depth(N), 1)
    return SimpleNamespace(plan_rows=rows, plan_salts=salts, plan_bcs={i: C3.commit(16 * L, rows[i], salts[i]) for i in rows},
                           values=C3.Values(bytes(64), None, N, ()), S=S)


# --- staging and the audit ----------------------------------------------------------------------------------------------------

def stager(S, n: int, out: Path) -> SimpleNamespace:
    """The statement's program and partition (one template, n proof blocks) and its one stager (the circuit lowered once)."""
    defn = bind(BC.BlockOpeningCheck, **vars(S))
    rows = BC.opening_rows(S)
    program = P.mixed_population_program([(defn, n)])
    query = P.template_instances_query([defn])
    part = P.partition(program, query)
    n_all, ranges = P.units(program, query), P.ranges(program, query)
    out.mkdir(parents=True, exist_ok=True)
    (out / "partition.json").write_text(json.dumps(part, sort_keys=True))
    (out / "program.json").write_bytes(canonical_json(encode_program(program)))
    bound = {"partition": P.digest(part), "program": part["program"]}
    t = time.time()
    st = CS.UnitStager(defn, "unit/a8-opening", [(r, V2, b, ps) for r, b, ps in rows], [(V2, (0,))], circuit=out / "circuit.txt",
                       bind=bound)
    st.pin
    lower_s = round(time.time() - t, 1)
    (tid, base, cnt), = ranges
    assert tid == P.template_id(defn) and cnt == n == n_all
    return SimpleNamespace(defn=defn, rows=rows, program=program, query=query, part=part, n=n, base=base, st=st, out=out,
                           lower_s=lower_s, partition_file=out / "partition.json", program_file=out / "program.json")


def stage(sg: SimpleNamespace, S, args: list[dict], name: str, plan_salts: list[bytes], salt_key: bytes) -> dict:
    """One round's statement on the stager: proof block i's plan row under ``plan_salts[i]``, every other row under a salt
    from the prover's ``salt_key``."""
    t0 = time.time()
    st, n, out = sg.st, len(args), sg.out / f"stage-{name}"
    k_log = st.meta["k_log"]
    nbl = max(3, (-(-n // st.meta["vus_per_block"]) - 1).bit_length())
    if not 25 <= k_log + nbl <= a7.M_MAX:
        raise SystemExit(f"{name}: m = {k_log} + {nbl} is outside 25 .. {a7.M_MAX} (--zk and flock-circuit's shape)")
    words = [BC.row_words(sg.rows, a) for a in args]
    data = {r: np.asarray([w[r] for w in words], dtype=np.uint16) for r, _b, _ps in sg.rows}
    set_name = f"{st.iset.name}-{name}"
    iset = SimpleNamespace(**{**vars(st.iset), "name": set_name})
    salts = FC.stage_salts(iset, 0, n, st.ports, salt_key)
    salts["plan"] = list(plan_salts)
    got = st.stage(data, out, salt_key, salts=salts, indices=range(sg.base, sg.base + n), set_name=set_name)
    (out / "inst.bin").rename(out / f"inst-{n}.bin")
    (out / "pub.bin").rename(out / f"pub-{n}.bin")
    (out / "tables").mkdir(exist_ok=True)
    os.link(sg.out / "circuit.txt", out / "circuit.txt")
    m = {"tag": f"{name}", "name": f"open-{name}", "dir": out, "meta": st.meta, "pin": st.pin, "class": st.class_sha512,
         "set": set_name, "n": n, "base": sg.base, "defn": sg.defn, "S": S, "rows": sg.rows, "args": args,
         "ands": got["ands"], "unit_nnz": got["unit_nnz"], "salts": got["salts"], "ports": st.ports,
         "oks": [int(x) for x in np.asarray(got["outs"]["o0"])[:, 0]], "partition_file": sg.partition_file,
         "program_file": sg.program_file, "stage_s": round(time.time() - t0, 1)}
    m["header"] = a0.public_header(out, n)
    return m


def session(a, m: dict, sdir: Path, draw: list[int], port: int, env: dict | None = None) -> tuple[dict | None, dict, dict]:
    """One live --zk session on member m's statement: the server serving ``draw`` (the verifier's own, `serve --draw-file`)
    and one prover process (`prove --zk`, with ``--gpu`` on the device). Returns the server's record, the prover's LIVE
    line, and the prover process's wall and CPU seconds."""
    sdir.mkdir(parents=True, exist_ok=True)
    df = sdir / "draw.json"
    df.write_text(json.dumps({"law": "subset", "population": m["n"], "k": len(draw), "units": draw}, sort_keys=True))
    d, n = m["dir"], m["n"]
    env = {**os.environ, **(env or {})}
    with open(sdir / "serve.log", "w") as slog:
        serve = subprocess.Popen([a.bin, "serve", "--zk", "--listen", f"127.0.0.1:{port}", "--out", str(sdir / "sessions"),
                                  "--instances", str(d / f"pub-{n}.bin"), "--circuit", str(d / "circuit.txt"), "--tables",
                                  str(d / "tables"), "--pin", m["pin"], "--operator", "one-stage-a8", "--draw-file", str(df),
                                  "--sessions", "1"], stdout=slog, stderr=subprocess.STDOUT, text=True, env=env)
        try:
            for _ in range(3600):
                if "SERVING" in (sdir / "serve.log").read_text() or serve.poll() is not None:
                    break
                time.sleep(0.5)
            cmd = [*a.lease, a.bin, "prove", "--zk", *(["--gpu"] if a.gpu else []), "--verifier", f"127.0.0.1:{port}",
                   "--instances", str(d / f"inst-{n}.bin"), "--circuit", str(d / "circuit.txt"), "--tables", str(d / "tables"),
                   "--runs", "1", "--warm", "0"]
            r0, t = resource.getrusage(resource.RUSAGE_CHILDREN), time.time()
            p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=4 * 3600)
            wall, r1 = time.time() - t, resource.getrusage(resource.RUSAGE_CHILDREN)
            (sdir / "prove.log").write_text(f"$ {' '.join(cmd)}\n# rc {p.returncode} in {wall:.1f} s\n--- stdout\n{p.stdout}\n"
                                            f"--- stderr\n{p.stderr}")
            live = next((json.loads(x[5:]) for x in p.stdout.splitlines() if x.startswith("LIVE\t")), None)
            if live is None:
                live = {"accepted": False, "why": f"prover exited {p.returncode}: {p.stderr[-600:]}"}
            serve.wait(timeout=3600)
        finally:
            if serve.poll() is None:
                serve.kill()
    recs = sorted((sdir / "sessions").glob("*/session.json"))
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    return (json.loads(recs[-1].read_text()) if recs else None), live, {"wall_s": round(wall, 2), "cpu_s": round(cpu, 1),
                                                                       "max_child_rss_gb": round(r1.ru_maxrss / 2**20, 1)}


TIMING = ("e2e_s", "wait_s", "wait_e2e_s", "witness_s", "prove_total_s", "tp_s", "k_log", "dense_m", "units", "prover", "gpu")


def audit_round(a, sg, m: dict, rs, c3, draws, out: Path, port: int) -> dict:
    """Member m registered, drawn by the verifier's Lean (every proof block), proved --zk and held by Lean; then the
    verifier's side: each proof block's public rows opened to its own copy, its verdict row read, its plan row linked to
    the pass's plan by ``b || c``, and the window decided on the verdicts. Writes what the verifier received under ``out / "verifier"``."""
    out.mkdir(parents=True, exist_ok=True)
    n, name = m["n"], m["name"]
    want, roots = {}, []
    fv = m["header"]["frame_v3"]
    for p_, w in a7.frame_ports(m["set"], m["meta"], n).items():
        want[f"{name}/{p_}"] = w
        roots.append({"port": f"{name}/{p_}", "root": fv["roots"][p_], "leaves": w["leaves"], "domain": fv["domain_ids"][p_]})
    leaf_layer = hashlib.sha512(canonical_json({name: a0.sha512_file(m["dir"] / f"pub-{n}.bin")})).hexdigest()
    law = D.parse_law(f"stratified:{n}")
    win = {"kind": "stand-in", "of": "the openings of one window of a remote bandwidth session (a8.py), on a "
                                      f"{rs.store.registration.blocks}-chunk registered store",
           "members": {name: {"template": P.template_id(m["defn"]), "units": [m["base"], m["base"] + n]}},
           "link": "opening.plan_link: the plan row by its b || c to the verifier's own copy of the pass's plan"}
    rec = R.record(sg.program, sg.query, law, a0.SCHEME, roots, win, leaf_layer=leaf_layer)
    (out / "registration.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    checks = {"registration": V.check(rec, sg.program, sg.query, law, a0.SCHEME, want, leaf_layer=leaf_layer).to_json(),
              f"statement/{name}": a7.statement_checks(m, rec, want)}
    receipt = R.receipt(rec, "one-stage-e2e verifier (a8.py)")
    res = {"checks_before_draw": {c: v["ok"] for c, v in checks.items()}}
    if not all(v["ok"] for v in checks.values()):
        res["checks"] = checks
        return {**res, "accepted": False, "failures": ["malformed"] * len(draws)}
    p = a0.run([a.lean, "draw", "--program", str(sg.program_file), "--partition", str(sg.partition_file), "--stratified", str(n)],
               out / "draw.log")
    lean_draw = json.loads(next(x for x in p.stdout.splitlines() if x.startswith("DRAW ")).split(" ", 1)[1])
    local = sorted(u - m["base"] for u in lean_draw["units"] if m["base"] <= u < m["base"] + n)
    sdir = out / "session"
    record, live, proc = session(a, m, sdir, local, port)
    draw_t = (record or {}).get("unit_draw") or {}
    v = V.check_draw(draw_t, {"population": n}, {"law": "subset", "k": len(local)})
    verdicts = {f"m0-verifier/{name}": {"accepted": ((record or {}).get("verdict") or {}).get("accepted") is True
                                        and (live or {}).get("roots") == fv["roots"] and v.ok,
                                        "why": ((record or {}).get("verdict") or {}).get("why"), "draw_check": v.to_json()}}
    lv = a0.lean_verify(a.lean, m, n, sdir, zk=True)
    verdicts[f"lean-verify/{name}"] = {**lv, "tool": "flock-verify verify --zk --archive --partition --program"}
    proved = a0.proved_units(m["header"], live or {})
    statement = {name: {"session_s": proc["wall_s"], **{x: live.get(x) for x in ("circuit_sha512", "statement_digest", "sigma")},
                        "timing": {x: live.get(x) for x in TIMING}, "process": proc, "lean_verify_s": lv.get("verify_s")}}
    sessions = [a0.sha512_file(f) for f in sorted((sdir / "sessions").glob("*/session.json"))]
    result = V.audit_record(rec, receipt, lean_draw, rec["law"], sorted(proved), live.get("instances", 0), statement, verdicts,
                            a.delta, sessions=sessions, served=sorted(m["base"] + u for u in draw_t.get("units", [])),
                            session_roots={f"{name}/{p_}": r for p_, r in ((live or {}).get("roots") or {}).items()})
    (out / "audit.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))

    held = set(proved) if result.get("accepted") else set()
    port_ = {x.name: x for x in m["ports"]}
    bcs = a7.public_bcs(m["dir"] / f"pub-{n}.bin", m["ports"])
    seed = rs.store.registration.codec_seed
    failures, sent = [], []
    for i, d in enumerate(draws):
        mine = C3.public(d, seed, m["S"], c3.values)
        word = m["oks"][i]
        opened = ((m["base"] + i) in held and a7.opens(port_["pub"], a7.public_words(mine, m["rows"]), m["salts"]["pub"][i], bcs[i]["pub"])
                  and a7.opens(port_["signs"], mine["signs"], m["salts"]["signs"][i], bcs[i]["signs"])
                  and a7.opens(port_["o0"], [word], m["salts"]["o0"][i], bcs[i]["o0"]))
        mine_bc = C3.commit(16 * m["S"].P, c3.plan_rows[d.step], c3.plan_salts[d.step])
        failures.append("malformed" if not opened else BC.verdict(word) or C3.plan_link(bcs[i]["plan"], mine_bc))
        sent.append({"unit": m["base"] + i, "verdict": word, "plan_salt": c3.plan_salts[d.step].hex(),
                     **{f"salt/{r}": m["salts"][r][i].hex() for r in OPENED}})
    verdict = rs.session.decide(0, failures)

    vdir = out / "verifier"
    vdir.mkdir(exist_ok=True)
    for f in (out / "registration.json", sg.partition_file, sg.program_file, m["dir"] / "circuit.txt", m["dir"] / f"pub-{n}.bin",
              sdir / "draw.json", out / "audit.json"):
        os.link(f, vdir / f.name)
    for f in sorted((sdir / "sessions").rglob("*")):
        if f.is_file():
            os.link(f, vdir / f"session-{f.parent.name}-{f.name}")
    (vdir / "openings.json").write_text(json.dumps(sent, indent=1))
    (vdir / "v0.json").write_text(json.dumps({"registration": rs.store.registration.digest.hex(), "messages": rs.messages,
                                              "draws": [[d.index, d.position] for d in draws]}, indent=1))
    return {**res, "accepted": verdict.accepted, "failure": verdict.failure, "failures": failures,
            "audit": {x: result.get(x) for x in ("accepted", "complete", "wrong_units")},
            "verdicts": {x: y.get("accepted") for x, y in verdicts.items()}, "why": {x: y.get("why") for x, y in verdicts.items()},
            "timing": statement, "opened_rows": list(OPENED), "linked_rows": ["plan"], "verdict_words": m["oks"]}


# --- what the verifier received -----------------------------------------------------------------------------------------------

def needles(pieces: dict[str, bytes]) -> dict[bytes, str]:
    """Each named piece (`leak_scan.cut`'s), raw and as hex (both cases), to its name."""
    if bad := {k: len(x) for k, x in pieces.items() if len(x) != PIECE}:
        raise ValueError(f"pieces {bad} (their bytes) aren't {PIECE} bytes: cut each row whole (`leak_scan.public_pieces`)")
    out: dict[bytes, str] = {}
    for name, x in pieces.items():
        for nd in (x, x.hex().encode(), x.hex().upper().encode()):
            out.setdefault(nd, name)
    return out


def find(data: bytes, table: dict[bytes, str]) -> dict[str, int]:
    """For each blob, how many of its needles occur in ``data`` at any offset (keyed on each needle's first 8 bytes)."""
    if len(data) < 8:
        return {}
    keys: dict[int, list[bytes]] = {}
    for nd in table:
        keys.setdefault(int.from_bytes(nd[:8], "little"), []).append(nd)
    karr = np.fromiter(keys, dtype=np.uint64, count=len(keys))
    a, hits = np.frombuffer(data, dtype=np.uint8), {}
    for s in range(8):
        k = (len(a) - s) // 8
        if k <= 0:
            continue
        w = a[s:s + 8 * k].view("<u8")
        for at in np.flatnonzero(np.isin(w, karr)):
            off = s + 8 * int(at)
            for nd in keys[int(w[at])]:
                if data[off:off + len(nd)] == nd:
                    hits.setdefault(table[nd], set()).add(nd)
    return {x: len(v) for x, v in hits.items()}


def received(vdir: Path, control: Path, pieces: dict[str, bytes], held: set[str]) -> dict:
    """Every file under ``vdir`` (what the verifier received) and the prover's ``control`` file, scanned for ``pieces``
    (`needles`): each file's size and the pieces found in it; the control is to hold every piece in ``held``."""
    table = needles(pieces)
    files = {}
    for f in sorted(vdir.iterdir()):
        files[f.name] = {"bytes": f.stat().st_size, "found": find(f.read_bytes(), table)}
    ctl = find(control.read_bytes(), table)
    return {"needles": len(table), "blobs": len(pieces), "files": files,
            "verifier_bytes": sum(x["bytes"] for x in files.values()),
            "verifier_found": sorted({b for x in files.values() for b in x["found"]}),
            "control": {"file": control.name, "bytes": control.stat().st_size, "blobs_found": len(ctl), "of": len(pieces),
                        "holds_all": held <= set(ctl), "missing": sorted(held - set(ctl))}}


def scan(vdir: Path, control: Path, blobs: dict[str, bytes], held: set[str], public: bytes) -> dict:
    """What the verifier received (`received`) scanned for ``blobs``, each cut whole by the zero-knowledge drivers' one
    cutter less the pieces the public circuit ``public`` holds (`leak_scan.public_pieces`, listed as ``public_pieces``),
    and the found pieces judged by their one rule (`leak_scan.scan`: ``leaked``, ``structured``, ``placed``). A blob
    shorter than a piece is refused (`leak_scan.cut`). ``public`` is the staged circuit, which the stager writes from the
    Definition before any input, so what it holds the verifier has whatever the witness."""
    pieces, held, left_out = LS.public_pieces(sys.modules[__name__], blobs, held, public)
    got = {**received(vdir, control, pieces, held), "public_pieces": left_out}
    return {**got, **LS.scan(got, pieces, vdir)}


# --- cost points --------------------------------------------------------------------------------------------------------------

def cost(a, m: dict, out: Path, port: int) -> list[dict]:
    """Sessions on member m's statement whose verifier draws its first k proof blocks, ``a.cold`` prover processes a
    point in each arm; Lean verifies each arm's first session at the largest k."""
    rows = []
    ks = sorted({k for k in a.points if 1 <= k <= m["n"]})
    for arm in a.arms:
        env = {"FC_FIREWALL": "1" if arm == "fw" else "0", **({"FC_HOST_WITNESS": "1"} if a.gpu else {}),
               **(a0.firewall_env(m) if arm == "fw" else {})}
        for k in ks:
            for rep in range(a.cold):
                sdir = out / f"{arm}-k{k}-r{rep}"
                record, live, proc = session(a, m, sdir, list(range(k)), port + len(rows), env)
                row = {"arm": arm, "k": k, "rep": rep, "staged": m["n"], "gpu": a.gpu, **proc,
                       "accepted": bool(live.get("accepted")) and ((record or {}).get("verdict") or {}).get("accepted") is True,
                       "why": live.get("why") or ((record or {}).get("verdict") or {}).get("why"),
                       **{x: live.get(x) for x in TIMING if x != "gpu"}}
                if k == ks[-1] and rep == 0:
                    lv = a0.lean_verify(a.lean, m, m["n"], sdir, zk=True)
                    row["lean"] = {x: lv.get(x) for x in ("accepted", "why", "verify_s")}
                rows.append(row)
                with open(out / "cost.jsonl", "a") as f:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
                log(f"cost {arm} k={k} rep={rep}: accepted {row['accepted']}, wall {row['wall_s']} s, cpu {row['cpu_s']} s, "
                    f"e2e {row.get('e2e_s')}, wait {row.get('wait_s')}, prove {row.get('prove_total_s')}"
                    + (f", lean {row['lean']}" if "lean" in row else ""))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--lean", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", type=int, default=4096)
    ap.add_argument("--plan", type=int, default=64, help="each pass's chunks, P: a stripe of the store")
    ap.add_argument("--samples", type=int, default=16, help="the window's openings, each one proof block")
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--delta", type=float, default=2.0 ** -20)
    ap.add_argument("--points", default="", help="cost points: the k drawn proof blocks of each session, comma-separated")
    ap.add_argument("--arms", default="nofw", help="nofw, fw or both, comma-separated")
    ap.add_argument("--cold", type=int, default=1)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--no-audit", action="store_true", help="stage the honest statement and run the cost points only")
    ap.add_argument("--port", type=int, default=7581)
    ap.add_argument("--lease", default="", help="a command line each prover process runs under (a GPU lease), shell-quoted")
    ap.add_argument("--dry", action="store_true", help="the sessions, witnesses and references only: nothing lowered or proved")
    a = ap.parse_args()
    a.lease = shlex.split(a.lease)
    a.points = [int(x) for x in a.points.split(",") if x]
    a.arms = [x for x in a.arms.split(",") if x]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.blocks % a.plan or a.plan % receipts.PER_LEAF or a.samples > a.blocks:
        raise SystemExit("--blocks is a multiple of --plan, --plan of 16, and --samples at most --blocks")
    secret = os.urandom(32)
    summary: dict = {"blocks": a.blocks, "plan": a.plan, "samples": a.samples, "codec": P512.name, "gpu": a.gpu}

    t = time.time()
    store = store_of(a.blocks, a.seed, secret)
    rs = window(store, a.plan, a.samples, a.delta, secret, a.seed)
    draws = rs.session.drawn(0)
    c3 = inputs(rs, a.plan, secret)
    S = c3.S
    summary["session_s"] = round(time.time() - t, 1)
    summary["statics"] = vars(S)
    summary["draws"] = [{"pass": d.index, "position": d.position, "block": d.block} for d in draws]
    log(f"session: {a.blocks} chunks, {a.blocks // a.plan} passes of {a.plan}, {len(draws)} draws, statics {vars(S)}, "
        f"{summary['session_s']} s")

    rounds = {"honest": (rs, store)}
    if not a.no_audit:
        bad_j = draws[0].block
        bad = tampered(store, bad_j)
        rt = window(bad, a.plan, a.samples, a.delta, secret, a.seed)
        if [(d.index, d.position) for d in rt.session.drawn(0)] != [(d.index, d.position) for d in draws]:
            raise SystemExit("the tampered round drew other receipts than the honest one")
        rounds["tampered"] = (rt, bad)
        summary["tampered_block"] = bad_j

    prepared = {}
    for name, (r, st_) in rounds.items():
        ds = r.session.drawn(0)
        wits = [C3.witness(d, r.passes[d.window, d.index], st_, c3.plan_rows[d.step], S, c3.values) for d in ds]
        ref = [C3.check(d, w, st_.registration.codec_seed, S, c3.values, c3.plan_salts[d.step], c3.plan_bcs[d.step])
               for d, w in zip(ds, wits)]
        if ref != ([None] * len(ds) if name == "honest" else ["codec"] + [None] * (len(ds) - 1)):
            raise SystemExit(f"{name}: the proof block's reference on the prover's witness is {ref}")
        BC.row_words(BC.opening_rows(S), wits[0])
        prepared[name] = (r, st_, ds, wits, ref)
        log(f"{name}: the proof block's reference on the prover's witness: {ref}")
    if a.dry:
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
        log("dry: done")
        return 0

    sg = stager(S, len(draws), out)
    summary["lower_s"] = sg.lower_s
    log(f"stager: {sg.lower_s} s, k_log {sg.st.meta['k_log']}, pin {sg.st.pin[:16]}")
    salt_key = hashlib.sha512(secret + b"row salts").digest()[:32]
    members = {}
    for name, (r, st_, ds, wits, ref) in prepared.items():
        m = stage(sg, S, wits, name, [c3.plan_salts[d.step] for d in ds], salt_key)
        if [BC.verdict(w) for w in m["oks"]] != ref:
            raise SystemExit(f"{name}: the staged verdict words {m['oks']} are not the reference's {ref}")
        members[name] = (m, r, ds, ref)
        summary[name] = {"reference": ref, "verdict_words": m["oks"], "stage_s": m["stage_s"], "ands": m["ands"],
                         "unit_nnz": m["unit_nnz"], "k_log": m["meta"]["k_log"], "pin": m["pin"]}
        log(f"staged {name}: {m['n']} proof blocks, {m['ands']} ANDs, k_log {m['meta']['k_log']}, verdicts {m['oks']}, "
            f"reference {ref}, {m['stage_s']} s")

    ok = True
    if not a.no_audit:
        for j, (name, (m, r, ds, ref)) in enumerate(members.items()):
            res = audit_round(a, sg, m, r, c3, ds, out / name, a.port + j)
            blobs = {}
            for d in ds:
                blobs[f"chunk {d.block}"] = r.store.blocks[d.block]
                blobs[f"payload {d.block}"] = r.store.payload(d.block)
            held = set(blobs)
            if name == "tampered":
                blobs[f"registered chunk {summary['tampered_block']}"] = store.blocks[summary["tampered_block"]]
            res["received"] = scan(out / name / "verifier", m["dir"] / f"inst-{m['n']}.bin", blobs, held,
                                   (m["dir"] / "circuit.txt").read_bytes())
            summary[name].update(res)
            want = name == "honest"
            got = res["received"]
            good = (res["accepted"] is want and res["audit"]["accepted"] is True and all(res["verdicts"].values())
                    and not got["leaked"] and got["control"]["holds_all"] and (want or res["failure"] == "codec"))
            ok &= good
            log(f"round {name}: window {'accepted' if res['accepted'] else 'rejected (' + str(res['failure']) + ')'}, proofs "
                f"{res['verdicts']}, failures {res['failures']}, verifier received {got['verifier_bytes']} bytes in "
                f"{len(got['files'])} files, chunk pieces leaked there: {got['leaked'] or 'none'} (structured: "
                f"{got['structured'] or 'none'}), control found {got['control']['blobs_found']}/{got['blobs']} pieces (holds "
                f"all it should: {got['control']['holds_all']}); {'as expected' if good else 'UNEXPECTED'}")
            (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))

    if a.points:
        m = members["honest"][0]
        rows = cost(a, m, out / "cost", a.port + 10)
        summary["cost"] = {"rows": rows, "ands": m["ands"], "unit_nnz": m["unit_nnz"]}
        ok &= all(r["accepted"] for r in rows) and all(r["lean"]["accepted"] for r in rows if "lean" in r)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    log("done: " + ("ok" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
