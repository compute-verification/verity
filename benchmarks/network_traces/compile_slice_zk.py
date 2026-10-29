"""The compile slice (`compile_slice.CompileSlice_v1`) proved in zero knowledge: one window of a few steps as one unit,
proved by C-Flock `prove --zk` behind the Lean firewall and held by Lean `verify --zk`, through N2's audit path
(`n2_zk.py`: its stager, firewall and received-pieces scan).

    python benchmarks/network_traces/compile_slice_zk.py prove --bin flock-circuit --lean flock-verify --out DIR
      [--slice demo|corpus|w212] [--g G] [--points 1] [--kinds K,...] [--dry]
    python benchmarks/network_traces/compile_slice_zk.py rescan --from PROVE_OUT --out DIR

* **The slice** (``--slice``): ``demo`` (the default), the demonstration class (`compile_slice.demo`, 59,341 ANDs a
  unit), whose candidates take real prefix hits; ``corpus``, corpus-smallkv's first seven steps
  (`verity/core/protocols/certifier/tests/fixtures/compile_slice_smallkv.json`, about 7.6M ANDs a unit), which `prove`
  first checks is the corpus's (`replay_vectors.vectors` on corpus-smallkv: the arrivals, the decisions and the replay's
  rows; `cache_vectors`' tokens of each arrival); or ``w212``, window 212 of vLLM's served run (the encodings'
  ``vllm-8eb1`` fixture) padded into the run's public engine class, (R, W, A) = (8, 1, 9), Q 4, D 32, P 54, K 4, no
  prefix caching, under the run's AsyncScheduler (AS = 1; `fixtures/compile_slice_w212_class_async.json`), with the
  seal's rows as the served ones (`served_rows.step_rows`), which `prove` first checks is the run's (the window's
  decisions as its replay asks them, `RP.decisions` on `replay_vectors.served_window`). The scheduler is the engine
  class's, one AS for every window of the run: `prove` refuses a fixture whose engine isn't the trace header's.
* **The charge** is from the session's carry (`charge_chain`) under the registered book (the arith fixture's
  async-light stream's). The demo's and the corpus's window is the session's first, so its carry-in is the book's
  priors under their public salt; w212's is the carry over windows 0 … 211 of the run, each window's decisions as its
  replay asks them, which `prove` commits window by window (`charge_chain.commit_window`; a stand-in: those windows'
  units aren't proved here), so w212's ``cin`` is window 211's carry-out row under its salt.
* **The statement**: one template of n units, each the slice's rows under fresh salts, one unit a kind (the slice's
  `REFUSED`, or `--kinds`' in their order). ``honest``; ``flip``, a decision flipped in its window row (the advice row
  stays the registered one); ``charge``, the charge one below the advice's; ``arrival``, arrival 1's prompt one token
  longer in its ``cert`` row (an arrival not in the certified record); ``cleared``, advice that leaves the last asked
  decision uncommitted (a cleared enable; on w212 also ``cleared-read``, the window row cut too, so the
  scan reads past the count); ``carry``, a carry-out other than the charge's carry' (the priors on the demo and the
  corpus, counts fitted to the window on w212); ``restart``, the charge from a carry-in of its own (zero counts on the
  demo and the corpus, the fitted counts on w212, which charge 25 bits in place of 27) under a fresh salt. With prefix
  caching (the demo and the corpus): ``table``, a made-up first cache table (`compile_slice.made_up`: request 2's first
  two blocks under ROOT on the demo, request 1's first block on the corpus, which gives it a 16-token hit at step 6);
  ``tokens``, request 1's first block made up in its ``cert`` row (9s on the demo, request 0's on the corpus); on the
  demo ``short``, request 1's admission hit claimed one block of its two, and ``unasked``, the waiting candidate's hit
  claimed 0; the adversarial choices the slice can take (`compile_slice.phase9`: a prefix hit, and on the corpus a
  decision read again and an arrival left in the queue). On the corpus and w212, the passes (advice
  of a pass at every ask where the served window admitted: six on the corpus, twenty-one on w212, one an async unit), with the served rows in
  ``srv`` (``sixpass``) and with its own (``sixpass-own``). The rest are honest, enough of them that the statement fits
  `--zk`'s shape (`stage`).
* **The served rows** (`served_rows`): the Commit side registers each unit's window under the run root (a stand-in: the
  partition's digest) before the draw, a row each block of G steps (``--g``, P by default: one ``srv`` row a unit; at
  G < P the unit is `compile_slice.CompileSliceBlocks_v1`, its rows ``srv0`` …): the served rows, but ``table``'s,
  which are the schedule its server served (its own rows, the hit taken), so its ``cache`` check alone refuses it.
  Every unit's ``srv`` rows are its window's but ``sixpass-own``'s, which are its own run's.
* **The bindings**, checked natively by the verifier on the row commitments b‖c the public file holds: each unit's
  ``adv`` row is row i of the advice's hiding registration (`verity.ml.commitments.advice`: its leaf from b‖c, its path
  to the registered root), its ``srv`` row j is leaf i·(P / G) + j of the served rows' registration
  (`served_rows.leaf`, `served_rows.leaf_ok`), its
  ``cert`` row is the certified record's commitment (a stand-in: one ``cert`` row under the certifier's salt, b‖c as
  `a7.opens` recomputes it), and its ``cin`` and ``cout`` rows are the window's carry rows in the session's chain
  (`charge_chain.check_chain` on the earlier windows' openings and the unit's two).
* **The verifier** opens each unit's ``pub`` row to its own copy (the declared charge and the registered book,
  `compile_slice.pub`) and its verdict ``o0``, under the salts the prover sends for those two rows; never ``adv``,
  ``cert``, ``srv``, ``cin``, ``cout`` or ``wit``. A unit passes when its verdict is `compile_slice.ACCEPT` and its
  bindings hold. What the verifier received is scanned (`received`) for every piece of each unit's witness row, its
  ``srv`` rows and every row's salt (`scan_pieces`: each row cut whole, `leak_scan.public_pieces`, less the pieces the
  staged circuit holds, and each row shorter than the scan's 16-byte piece whole, an ``srv`` row of a few steps: 10
  bytes at the demo's G = 2), and for each unit's carry rows whole (12 bytes; the session's first carry-in, the public
  priors, excepted); the prover's instance file, which holds those rows, is the control. A found piece fails the run
  when the lane's one rule calls it leaked (`leak_scan.scan`): there is no byte-count or witness-class exemption.
* **rescan**: a finished ``prove``'s scan taken again under that rule (`leak_scan.rescan`) from the verifier's files it
  kept, the pieces recomputed from its slice, kinds and partition (a salt the run drew isn't, so a found salt reads
  leaked).
* **Cost** (``--points``): sessions whose verifier draws the first k units, behind the firewall (`a8.cost`).
"""
from __future__ import annotations

import argparse
import dataclasses
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

import leak_scan as LS  # noqa: E402
import n2_zk as N2  # noqa: E402
from verity.core.protocols.certifier import arith_encode as AE  # noqa: E402
from verity.core.protocols.certifier import charge_chain as CC  # noqa: E402
from verity.core.protocols.certifier import compile_slice as CS  # noqa: E402
from verity.core.protocols.certifier import served_rows as SR  # noqa: E402
from verity.ml.commitments import advice as ADV  # noqa: E402
from verity.ml.commitments.frame_v3 import BitRowLeaf  # noqa: E402
from verity.ml.commitments.rowleaf import bit_row, row_leaf  # noqa: E402

FIXTURES = REPO / "verity/core/protocols/certifier/tests/fixtures"
FIXTURE = FIXTURES / "compile_slice_smallkv.json"
W212 = FIXTURES / "compile_slice_w212_class_async.json"
TRACE_8EB1 = REPO / "benchmarks/network_e2e/encodings/tests/fixtures/vllm-8eb1/advice_trace.jsonl.gz"
WINDOW = 212
#: counts fitted to w212, from which its decisions charge 25 bits in place of the session's 27
FITTED = AE.Carry(((1, 0), (16, 0), (64, 16)), 0)
ZERO = AE.Carry(((0, 0),) * AE.RUNS, 0)
_BASE = {"honest": set(), "flip": {"load", "rows"}, "charge": {"charge"}, "arrival": {"rows"}, "cleared": {"load"},
         "carry": {"code"}, "restart": set()}
#: each slice's kinds, in order, and the checks the reference refuses each by (``restart`` the chain alone refuses)
REFUSED = {
    "demo": {**_BASE, "table": {"cache"}, "tokens": {"rows"}, "short": {"cache", "rows"}, "unasked": {"cache"},
             "hit": {"cache", "rows"}},
    "corpus": {**_BASE, "table": {"cache"}, "tokens": {"rows"}, "hit": {"cache", "rows"}, "reread": {"steps", "rows"},
               "untaken": {"steps", "rows"}, "sixpass": {"rows"}, "sixpass-own": set()},
    "w212": {**_BASE, "cleared-read": {"steps", "rows"}, "sixpass": {"rows"}, "sixpass-own": set()}}
#: the decision each slice's ``flip`` flips, and the passes of its ``sixpass``
FLIP = {"demo": 2, "corpus": 5, "w212": 12}
PASSES = {"corpus": 6, "w212": 21}
OPENED = ("o0", "pub")


def book() -> dict:
    """The registered book: the arith fixture's async-light stream's (`arith_vectors.decisions`)."""
    streams = json.loads((FIXTURES / "arith_encode_vectors.json").read_text())["streams"]
    return next(s for s in streams if s["source"].startswith("async-light"))["decisions"]["book"]


BOOK = book()


def log(msg: str) -> None:
    print(f"CSZK {time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def load(name: str, G: int | None = None) -> SimpleNamespace:
    """The slice ``name``, its ``srv`` rows G steps each (P unless given): its statics, its inputs, the served rows (the
    demo's and the corpus's the replay's, w212's the seal's), the carry-out its ``carry`` forges and the carry-in its
    ``restart`` charges from. Its honest carry-in is set by `chain`."""
    if name == "demo":
        st, sl = CS.demo()
        st = CS.with_block(st, G)
        S = SimpleNamespace(**st)
        return SimpleNamespace(name=name, st=st, S=S, sl=sl, rows=list(CS.case(S, sl, BOOK).steps),
                               forged=AE.first_carry(BOOK), restart=ZERO)
    fx = json.loads((W212 if name == "w212" else FIXTURE).read_text())
    cached = {k: fx[k] for k in ("CN", "CTW", "CU") if k in fx}
    st = CS.statics(fx["R"], fx["W"], fx["A"], fx["engine"], fx["IW"], Q=fx["Q"], D=fx["D"], P=fx["P"], K=fx["K"],
                    G=G, **cached)
    S = SimpleNamespace(**st)
    sl = CS.Slice(tuple(map(tuple, fx["arrivals"])), tuple(fx["decisions"]), fx["steps"],
                  tuple(map(tuple, fx.get("tokens", ()))))
    if name == "w212":
        if fx["engine"] != engine_class():
            raise SystemExit(f"w212's fixture's engine {fx['engine']} isn't the run's engine class {engine_class()}: the "
                             "scheduler is the class's, so every window of the run is proved at its one AS")
        return SimpleNamespace(name=name, st=st, S=S, sl=sl, rows=[SR.step_rows(S, s, fx["ids"]) for s in fx["served"]],
                               forged=FITTED, restart=FITTED, fx=fx)
    return SimpleNamespace(name=name, st=st, S=S, sl=sl, rows=[tuple(map(tuple, r)) for r in fx["rows"]],
                           forged=AE.first_carry(BOOK), restart=ZERO)


def _encodings():
    """The encodings' `replay`, `replay_vectors` and `cache_vectors`, imported from their directory and dropped from
    `sys.modules` after, so their bare names clash with nothing."""
    import importlib

    enc = REPO / "benchmarks/network_e2e/encodings"
    names = ("replay", "replay_vectors", "cache_vectors", "enc_derived")
    saved, path = {k: sys.modules.pop(k) for k in names if k in sys.modules}, list(sys.path)
    sys.path[:0] = [str(enc)]
    try:
        return tuple(importlib.import_module(k) for k in names[:3])
    finally:
        sys.path[:] = path
        for k in names:
            sys.modules.pop(k, None)
        sys.modules.update(saved)


def engine_class() -> dict:
    """The run's engine as its trace's header gives it (`RP.Engine.from_header`), its scheduler (``async_scheduling``,
    AS) included: the public class at which every window of the run is proved."""
    RP, _RV, _CV = _encodings()
    e = RP.Engine.from_header(RP.records(TRACE_8EB1)[0]["config"])
    return {k: v for k, v in dataclasses.asdict(e).items() if k != "policy"}


def w212_decisions() -> list[list[int]]:
    """Each of the run's windows 0 … `WINDOW` decisions, as the window's own replay asks them (`RP.decisions` on
    `replay_vectors.served_window`)."""
    RP, RV, _CV = _encodings()
    recs = RP.records(TRACE_8EB1)
    wins = sorted({r["window"] for r in recs if r.get("kind") == "window"})
    if wins[:WINDOW + 1] != list(range(WINDOW + 1)):
        raise SystemExit(f"the run's windows aren't 0 … {WINDOW}")
    return [[int(d.arrived) for d in RP.decisions(RP.load_step_trace(RV.served_window(recs, w)[0]))[0]]
            for w in range(WINDOW + 1)]


def chain(x: SimpleNamespace, program: str, registration: str) -> SimpleNamespace:
    """The session's carry chain before the slice's window: ``values``, the earlier windows' produced carry values
    (`charge_chain.commit_window`, none for the session's first window), the window's honest carry-in ``carry`` and
    the salt of its ``cin`` row (the last carry-out's, or the priors' public `start_salt`)."""
    if x.name != "w212":
        return SimpleNamespace(values=[], carry=AE.first_carry(BOOK), salt=CC.start_salt(program), public=True)
    per = w212_decisions()
    if tuple(per[WINDOW]) != x.sl.decisions:
        raise SystemExit(f"window {WINDOW}'s decisions {per[WINDOW]} aren't the fixture's {x.sl.decisions}")
    D = x.S.D
    ses = CC.session(D, BOOK, [([1] * len(d) + [0] * (D - len(d)), d + [0] * (D - len(d))) for d in per[:WINDOW]])
    values = []
    for _length, _row_in, row_out in ses:
        values.append(CC.commit_window(program, BOOK, CC.carry_of(row_out), registration, values[-1] if values else None))
    return SimpleNamespace(values=values, carry=CC.carry_of(ses[-1][2]), salt=values[-1].salts[-1], public=False,
                           charged=sum(s[0] for s in ses))


def _corpus(tmp: Path) -> Path:
    """corpus-smallkv unpacked as the derived rules' tests read it
    (`benchmarks/network_e2e/encodings/tests/test_v1.py`'s `_corpus`, which imports pytest): each batch in Model Runner
    V2's order, each stop-id finish marked `eos`."""
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
    import tempfile

    enc = REPO / "benchmarks/network_e2e/encodings"
    RP, RV, CV = _encodings()
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


def cases(x: SimpleNamespace, n: int, kinds: tuple[str, ...]) -> list[SimpleNamespace]:
    """The statement's n units (the module's docstring): kind, the unit's rows, its declared charge, the window the
    server registered for it, the reference's verdict and whether its carry-in is the chain's (``chained``). `x.carry`
    is the window's honest carry-in (`chain`)."""
    out, red, demo = [], CS.phase9(x.S, x.sl), x.name == "demo"
    six = CS.Slice(x.sl.arrivals, (0,) * PASSES.get(x.name, 0), x.sl.steps, x.sl.tokens)
    cut = x.sl.decisions[:-1]
    for i in range(n):
        kind = kinds[i] if i < len(kinds) else "honest"
        kw = {"served": x.rows, "carry": x.carry, **red.get(kind, {})}
        sl = six if kind.startswith("sixpass") else x.sl
        if kind == "flip":
            flip = list(x.sl.decisions)
            flip[FLIP[x.name]] ^= 1
            kw["window"] = flip
        elif kind == "charge":
            kw["charge"] = CS.case(x.S, x.sl, BOOK, carry=x.carry).length - 1
        elif kind == "cleared":
            kw["advice"] = cut
        elif kind == "cleared-read":
            kw.update(advice=cut, window=cut)
        elif kind == "carry":
            kw["cout"] = x.forged
        elif kind == "restart":
            kw["carry"] = x.restart
        elif kind == "arrival":
            cert = list(x.sl.arrivals)
            cert[1] = (cert[1][0], cert[1][1] + 1, *cert[1][2:])
            kw["cert"] = cert
        elif kind == "table":
            kw.update(CS.made_up(x.S, x.sl, 2, 2) if demo else CS.made_up(x.S, x.sl, 1), served=None)
        elif kind == "tokens":
            toks = [list(t) for t in x.sl.tokens]
            toks[1][:x.S.B] = [9] * x.S.B if demo else toks[0][:x.S.B]
            kw["tokens"] = toks
        elif kind == "short":
            kw["hits"] = {2: (0, 1, 1)}
        elif kind == "unasked":
            kw["hits"] = {0: (0, 0, 0)}
        elif kind == "sixpass-own":
            kw["served"] = None
        c = CS.case(x.S, sl, BOOK, **kw)
        out.append(SimpleNamespace(kind=kind, case=c, charge=kw.get("charge", c.length),
                                   registered=list(c.steps) if kind == "table" else x.rows,
                                   words={r: CS.words(c.rows[r]) for r in CS.row_names(x.S)}, chained=kind != "restart"))
    return out


def stager(M, x: SimpleNamespace, n: int, out: Path) -> SimpleNamespace:
    """The statement's program and partition (one template, n units) and its stager, the unit lowered by its builder."""
    defn = CS.unit(**x.st)
    bits = CS.row_bits(x.S)
    rows = [(r, M.V2, bits[r], [r]) for r in CS.row_names(x.S)]
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
    return SimpleNamespace(S=x.S, slice=x.name, defn=defn, rows=rows, program=program, query=query, n=n, base=base, st=st,
                           out=out, part=part, lower_s=round(time.time() - t, 1), partition_file=out / "partition.json",
                           program_file=out / "program.json")


def population(meta: dict, kinds: tuple[str, ...]) -> int:
    """The fewest units whose statement fits `--zk`'s shape (k_log + nbl ≥ 25, `n2_zk.stage`), at least one unit a kind."""
    b = 25 - meta["k_log"]
    return len(kinds) if b <= 3 else max(len(kinds), meta["vus_per_block"] * ((1 << (b - 1)) + 1))


def certified(M, port, words: list[int], salt: bytes) -> bytes:
    """The certified record's commitment b‖c (the stand-in: the honest ``cert`` row under the certifier's salt)."""
    return M.CS.FC._commit_strings([port], {port.name: M.np.asarray([words], dtype=M.np.uint16)}, {port.name: [salt]})[port.name][0]


def srv_leaves(S, record: dict, i: int) -> list[tuple[str, int]]:
    """Unit i's ``srv`` rows, each with the served rows' leaf it is held to (`served_rows.leaf`: window i's block j)."""
    return [(r, SR.leaf(S, record, i, j)) for j, r in enumerate(CS.srv_names(S))]


def advice_leaf_ok(reg: ADV.Committed, binding: bytes, n: int, i: int, bc: bytes, n_bits: int) -> bool:
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
    data = {r: M.np.asarray([u.words[r] for u in units], dtype=M.np.uint16) for r in CS.row_names(sg.S)}
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
    b‖c (`advice_leaf_ok`, `served_rows.leaf_ok`, `certified`, `charge_chain.check_chain`). Writes what the verifier
    received under ``out / "verifier"``."""
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
    win = {"kind": "stand-in", "of": f"the compile slice's {sg.slice} window (compile_slice_zk.py)",
           "members": {name: {"template": P.template_id(m["defn"]), "units": [m["base"], m["base"] + n]}},
           "link": "compile_slice: each unit's adv row is the advice registration's row, its cert row the certified "
                   "record's, its cin and cout rows the window's in the session's carry chain"}
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
    served, record = binds["served"], binds["served"].record()
    earlier = [CC.openings(v) for v in binds["chain"].values]
    for i, u in enumerate(units):
        word = m["oks"][i]
        opened = ((m["base"] + i) in held
                  and a7.opens(port_["pub"], CS.pub(sg.S, u.charge, BOOK), m["salts"]["pub"][i], bcs[i]["pub"])
                  and a7.opens(port_["o0"], [word], m["salts"]["o0"][i], bcs[i]["o0"]))
        adv_ok = advice_leaf_ok(binds["advice"], binds["binding"], n, i, bcs[i]["adv"], binds["n_bits"])
        srv_ok = all(SR.leaf_ok(record, binds["run_root"], k, bcs[i][r], served.registration.open(k).path)
                     for r, k in srv_leaves(sg.S, record, i))
        cert_ok = bcs[i]["cert"] == binds["certified"]
        why = CC.check_chain(earlier + [[bcs[i]["cin"], bcs[i]["cout"]]], binds["program"], BOOK)
        bound = adv_ok and srv_ok and cert_ok and why is None
        verdict = ("malformed" if not opened else "accept" if word == CS.ACCEPT and bound else "reject")
        rows.append({"unit": m["base"] + i, "kind": u.kind, "verdict_word": word, "flags": u.case.flags, "adv_bound": adv_ok,
                     "srv_bound": srv_ok, "cert_bound": cert_ok, "chain_bound": why is None, "chain_why": why,
                     "verdict": verdict})
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
    (vdir / "bindings.json").write_text(json.dumps({
        "advice_root": binds["advice"].root.hex(), "certified": binds["certified"].hex(),
        "paths": [[s.hex() for s in binds["advice"].open(i).path] for i in range(n)],
        "served": record, "run_root": binds["run_root"].hex(),
        "served_paths": [[s.hex() for s in served.registration.open(k).path] for k in range(record["count"])],
        "book": BOOK, "program": binds["program"], "carry_chain": [[bc.hex() for bc in w] for w in earlier]}, indent=1))
    proofs = (live or {}).get("proof_bytes") or []
    return {**res, "accepted": bool(result.get("accepted")), "units": rows,
            "audit": {k: result.get(k) for k in ("accepted", "complete", "wrong_units")},
            "verdicts": {k: y.get("accepted") for k, y in verdicts.items()}, "why": {k: y.get("why") for k, y in verdicts.items()},
            "timing": statement[name], "lean_verify_s": lv.get("verify_s"),
            "per_unit": {"prove_total_s": N2._per(timing.get("prove_total_s"), n), "e2e_s": N2._per(timing.get("e2e_s"), n),
                         "proof_bytes": N2._per(sum(proofs) if proofs else None, n),
                         "lean_verify_s": N2._per(lv.get("verify_s"), n)},
            "opened_rows": list(OPENED)}


def _u16(words: list[int]) -> bytes:
    return b"".join(w.to_bytes(2, "little") for w in words)


def _public_cin(u: SimpleNamespace, chain_: SimpleNamespace) -> bool:
    """Whether the unit's ``cin`` is the session's first carry-in, the book's priors under their public salt."""
    return chain_.public and u.chained


def blobs_of(units: list[SimpleNamespace], salts: dict, chain_: SimpleNamespace,
             srv: tuple[str, ...] = ("srv",)) -> tuple[dict[str, bytes], set[str]]:
    """The witness pieces the verifier must not receive: each unit's ``wit`` and ``srv`` rows (`srv`, its rows' names) as
    the instance file holds them (u16 little-endian) and its ``adv``, ``cert``, ``srv``, ``cout``, ``wit`` and (but the
    public priors') ``cin`` salts, when `salts` has them (a rescan's doesn't); the rows are what the control holds."""
    blobs, held = {}, set()
    for i, u in enumerate(units):
        for r in ("wit", *srv):
            blobs[f"{r} {i}"] = _u16(u.words[r])
            held.add(f"{r} {i}")
        for r in ("adv", "cert", *srv, "wit", "cout") + (() if _public_cin(u, chain_) else ("cin",)) if salts else ():
            blobs[f"{r} salt {i}"] = salts[r][i]
    return blobs, held


def carry_rows(units: list[SimpleNamespace], chain_: SimpleNamespace) -> dict[str, bytes]:
    """Each unit's ``cout`` and (but the public priors') ``cin`` rows as the instance file holds them, 12 bytes each."""
    return {f"{r} {i}": _u16(u.words[r]) for i, u in enumerate(units)
            for r in ("cout",) + (() if _public_cin(u, chain_) else ("cin",))}


def scan_pieces(a8, units: list[SimpleNamespace], salts: dict, chain_: SimpleNamespace, srv: tuple[str, ...],
                circuit: bytes) -> tuple[dict[str, bytes], set[str], list[str]]:
    """`blobs_of`'s rows as the scan finds them: each row of at least the scan's 16-byte piece cut whole, less the
    pieces the staged circuit holds (`leak_scan.public_pieces`), and each shorter row whole (an ``srv`` row of a few
    steps, 10 bytes at the demo's G = 2, which `leak_scan.cut` refuses); the held ones, and the left-out pieces' hex."""
    rows, held = blobs_of(units, salts, chain_, srv)
    short = {k: x for k, x in rows.items() if len(x) < a8.PIECE}
    long_ = {k: x for k, x in rows.items() if k not in short}
    cut, kept, public = LS.public_pieces(a8, long_, held - set(short), circuit)
    return {**cut, **short}, kept | (held & set(short)), public


def _find(a8, data: bytes, table: dict[bytes, str]) -> dict[str, int]:
    """`a8.find`, which keys each needle on its first 8 bytes, and each needle shorter than that searched for itself (an
    ``srv`` row is 6 bytes at the demo's G = 1)."""
    hits = a8.find(data, {nd: k for nd, k in table.items() if len(nd) >= 8})
    for nd, k in table.items():
        if len(nd) < 8 and nd in data:
            hits[k] = hits.get(k, 0) + 1
    return hits


def found(a8, vdir: Path, control: Path | None, pieces: dict[str, bytes], held: set[str]) -> dict:
    """Every file under `vdir` (what the verifier received) and the prover's `control` (None on a rescan: a run doesn't
    keep it) scanned for each of `pieces` whole, raw or as hex (`_find`): `scan_pieces`' 16-byte pieces and its rows
    shorter than that, or the 12-byte carry rows, which `a8.needles` doesn't cut. Each distinct value is found under its
    first name (a chained unit's ``cin`` is the ``cout`` before it). The control is to hold every held piece."""
    first: dict[bytes, str] = {}
    for k, x in pieces.items():
        first.setdefault(x, k)
    table = {nd: k for x, k in first.items() for nd in (x, x.hex().encode(), x.hex().upper().encode())}
    files = {f.name: {"bytes": f.stat().st_size, "found": _find(a8, f.read_bytes(), table)}
             for f in sorted(vdir.iterdir()) if f.is_file()}
    out = {"pieces": len(first), "files": files, "verifier_bytes": sum(x["bytes"] for x in files.values()),
           "verifier_found": sorted({k for x in files.values() for k in x["found"]})}
    if control is not None:
        held, ctl = {first[pieces[h]] for h in held}, set(_find(a8, control.read_bytes(), table))
        out["control"] = {"file": control.name, "pieces_found": len(ctl), "of": len(first), "holds_all": held <= ctl,
                          "missing": sorted(held - ctl)}
    return out


def received(a8, vdir: Path, control: Path, pieces: dict[str, bytes], held: set[str]) -> dict:
    """`found`, and the found pieces judged by the lane's one rule (`leak_scan.scan`: ``leaked``, ``structured``,
    ``placed``)."""
    rec = found(a8, vdir, control, pieces, held)
    return {**rec, **LS.scan(rec, pieces, vdir)}


def expected(x: SimpleNamespace, kind: str, flags: dict) -> bool:
    return {f for f, ok in flags.items() if not ok} == REFUSED[x.name][kind]


def slice_check(x: SimpleNamespace, ch: SimpleNamespace) -> None:
    """The slice's log line, and the corpus's check that the fixture is the corpus's (`chain` checks w212's decisions)."""
    if x.name == "corpus":
        corpus_check(x)
        log(f"slice: corpus-smallkv's first {x.S.P} steps, {x.sl.arrivals}, decisions {x.sl.decisions}, the fixture is "
            "the corpus's")
    elif x.name == "w212":
        log(f"slice: vLLM 8eb1's window {WINDOW} at its engine class (P {x.S.P}, D {x.S.D}), {x.sl.arrivals}, decisions "
            f"{x.sl.decisions} (the window's replay's), carry-in {ch.carry} over windows 0 … {WINDOW - 1} "
            f"({ch.charged} bits)")
    else:
        log(f"slice: the demonstration class's {x.S.P} steps, {x.sl.arrivals}, decisions {x.sl.decisions}")


def prove(a) -> int:
    x = load(a.slice, a.g)
    summary: dict = {"statics": x.st, "row_bits": CS.row_bits(x.S), "slice": {"name": x.name, **dataclasses.asdict(x.sl)},
                     "kinds": a.kinds, "book": BOOK}
    secret = os.urandom(32)
    if a.dry:
        ch = chain(x, "compile-slice-zk-dry", "00" * 64)
        slice_check(x, ch)
        x.carry = ch.carry
        units = cases(x, len(a.kinds), a.kinds)
        summary["reference"] = [{"kind": u.kind, "verdict": u.case.verdict, "flags": u.case.flags, "charge": u.case.length}
                                for u in units]
        log(f"dry: {summary['reference']}")
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
        return 0 if all(expected(x, u.kind, u.case.flags) for u in units) else 1
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
    binding = bytes.fromhex(M.P.digest(sg.part))
    program = sg.part["program"]
    ch = chain(x, program, binding.hex())
    slice_check(x, ch)
    x.carry = ch.carry
    units = cases(x, n, a.kinds)
    for u in units:
        if not expected(x, u.kind, u.case.flags):
            raise SystemExit(f"{u.kind}: the reference's flags {u.case.flags} are not the kind's")
    salt_key = hashlib.sha512(secret + b"row salts").digest()[:32]
    SB = M.CS.FC.SALT_BYTES
    cert_salt = os.urandom(SB)
    srv = CS.srv_names(x.S)
    srv_salts = [os.urandom(SB) for _ in range(n * len(srv))]
    salts = {"adv": [os.urandom(SB) for _ in range(n)], "cert": [cert_salt] * n,
             **{r: [srv_salts[i * len(srv) + j] for i in range(n)] for j, r in enumerate(srv)},
             "cin": [ch.salt if u.chained else os.urandom(SB) for u in units], "cout": [os.urandom(SB) for _ in range(n)]}
    port_ = {p.name: p for p in sg.st.ports}
    honest = units[0].words
    n_bits = 16 * len(honest["adv"])
    reg = ADV.commit([bit_row((u.words["adv"], 16))[0] for u in units], n_bits, binding, salts["adv"])
    served = SR.register(x.S, [u.registered for u in units], binding, G=CS.block(x.S), salts=srv_salts)
    binds = {"advice": reg, "binding": binding, "n_bits": n_bits, "served": served, "run_root": binding,
             "certified": certified(M, port_["cert"], honest["cert"], cert_salt), "chain": ch, "program": program}
    m = stage(M, sg, units, salt_key, salts)
    refs = [u.case.verdict for u in units]
    if m["oks"] != refs:
        raise SystemExit(f"the staged verdict words {m['oks']} are not the reference's {refs}")
    bcs = M.a7.public_bcs(m["dir"] / f"pub-{n}.bin", m["ports"])
    same = all(bcs[i]["adv"] == ADV.INST.commit_string(ADV.INST.default_key,
                                                       BitRowLeaf(n_bits).layout().digest(bit_row((u.words["adv"], 16))[0]),
                                                       salts["adv"][i]) for i, u in enumerate(units))
    carry_same = all(bcs[i][r] == CC._opening(u.words[r], salts[r][i]) for i, u in enumerate(units) for r in ("cin", "cout"))
    srv_same = [all(bcs[i][r] == served.bc(k) for r, k in srv_leaves(x.S, served.record(), i)) for i in range(n)]
    srv_want = [u.kind != "sixpass-own" for u in units]
    chain_want = [u.chained for u in units]
    summary["staged"] = {"n": n, "ands": m["ands"], "unit_nnz": m["unit_nnz"], "stage_s": m["stage_s"], "k_log": m["meta"]["k_log"],
                         "pin": m["pin"], "verdict_words": m["oks"], "adv_bc_is_the_advice_leafs": same,
                         "srv_bc_is_the_served_leafs": srv_same, "carry_bc_is_the_chains_openings": carry_same,
                         "charges": [u.case.length for u in units[:len(a.kinds)]]}
    log(f"staged: {n} units, {m['ands']} ANDs a unit, k_log {m['meta']['k_log']}, verdicts {m['oks'][:len(a.kinds)]}..., "
        f"charges {summary['staged']['charges']}..., adv b‖c is advice's: {same}, srv b‖c is the served rows' but "
        f"sixpass-own's: {srv_same == srv_want}, cin/cout b‖c are the chain's openings: {carry_same}, {m['stage_s']} s")
    same &= srv_same == srv_want and carry_same
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))

    av = SimpleNamespace(bin=a.bin, lean=a.lean, lease=[], gpu=False, delta=a.delta, points=a.points, arms=["fw"], cold=1)
    res = audit(M, av, sg, m, units, binds, out / "slice", a.port)
    pieces, held, public = scan_pieces(M.a8, units, m["salts"], ch, srv, (m["dir"] / "circuit.txt").read_bytes())
    short = sum(len(x) < M.a8.PIECE for x in pieces.values())
    vdir, inst, carry = out / "slice" / "verifier", m["dir"] / f"inst-{n}.bin", carry_rows(units, ch)
    res["received"] = rec = {**received(M.a8, vdir, inst, pieces, held), "public_pieces": public,
                             "carry_rows": received(M.a8, vdir, inst, carry, set(carry))}
    summary["audit"] = res
    want = ["accept" if u.kind == "honest" else "reject" for u in units]
    got = [r["verdict"] for r in res.get("units", [])]
    cr = rec["carry_rows"]
    ok = (res["accepted"] and all(res["verdicts"].values()) and got == want and same
          and [r["srv_bound"] for r in res.get("units", [])] == srv_want
          and [r["chain_bound"] for r in res.get("units", [])] == chain_want
          and not rec["leaked"] and rec["control"]["holds_all"] and not cr["leaked"] and cr["control"]["holds_all"])
    log(f"proofs {res['verdicts']}, audit {res['audit']}, first units {[(r['kind'], hex(r['verdict_word']), r['adv_bound'], r['srv_bound'], r['cert_bound'], r['chain_why'] or True, r['verdict']) for r in res.get('units', [])[:len(a.kinds)]]}, "
        f"per unit {res['per_unit']}, verifier received {rec['verifier_bytes']} bytes, witness pieces found there "
        f"(of {len(pieces)}, {short} of them rows shorter than a piece; {len(public)} the circuit holds left out): "
        f"leaked {rec['leaked'] or 'none'}, structured {rec['structured'] or 'none'}, control "
        f"{rec['control']['pieces_found']}; carry rows found there (of "
        f"{len(carry)}): leaked {cr['leaked'] or 'none'}, structured {cr['structured'] or 'none'}, control holds them: "
        f"{cr['control']['holds_all']}; {'as expected' if ok else 'UNEXPECTED'}")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    if a.points:
        rows = M.a8.cost(av, m, out / "cost", a.port + 20)
        summary["cost"] = rows
        ok &= all(r["accepted"] for r in rows) and all(r["lean"]["accepted"] for r in rows if "lean" in r)
    summary["ok"] = ok
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    log("done: " + ("ok" if ok else "FAILED"))
    return 0 if ok else 1


def rescan(a) -> int:
    """`prove`'s scan under `--from` taken again (`leak_scan.rescan`) from the verifier's files it kept: its units rebuilt
    from its slice, kinds and partition (held to its staged verdict words), every piece of their rows found again less
    those its circuit holds (its summary's ``public_pieces``; the circuit isn't kept; `scan_pieces`, at the run's G), and
    each carry row. A salt is drawn by `prove` and not recomputed, so a salt the run found reads leaked."""
    src, out = Path(getattr(a, "from")), Path(a.out)
    summary = json.loads((src / "summary.json").read_text())
    vdir, M = src / "slice" / "verifier", N2._one_stage()
    x = load(summary["slice"]["name"], summary["statics"].get("G"))
    part = json.loads((vdir / "partition.json").read_text())
    ch = chain(x, part["program"], M.P.digest(part))
    x.carry = ch.carry
    units = cases(x, summary["staged"]["n"], tuple(summary["kinds"]))
    if [u.case.verdict for u in units] != summary["staged"]["verdict_words"]:
        raise SystemExit("these units' verdict words aren't the run's: rescan it with the commit that proved it")
    got, circuit = summary["audit"]["received"], set(summary["audit"]["received"].get("public_pieces", []))
    pieces = {k: v for k, v in scan_pieces(M.a8, units, {}, ch, CS.srv_names(x.S), b"")[0].items()
              if v.hex() not in circuit}
    rec = found(M.a8, vdir, None, pieces, set())
    rec["verifier_found"] = sorted(set(rec["verifier_found"]) | set(got["verifier_found"]))
    carry, cg = carry_rows(units, ch), got.get("carry_rows", {})
    res = {**LS.rescan({**rec, "leaked": got.get("leaked", got["verifier_found"])}, pieces, vdir),
           "carry_rows": LS.rescan({**found(M.a8, vdir, None, carry, set()), "leaked": cg.get("leaked", cg.get("verifier_found"))},
                                   carry, vdir)}
    cr = res["carry_rows"]
    ok = not res["leaked"] and not cr["leaked"]
    log(f"rescan of {src}: witness pieces (of {len(pieces)}) leaked {res['leaked'] or 'none'} (was {res['was'] or 'none'}), "
        f"structured {res['structured'] or 'none'}; carry rows (of {len(carry)}) leaked {cr['leaked'] or 'none'} (was "
        f"{cr['was'] or 'none'}), structured {cr['structured'] or 'none'}: {'clean' if ok else 'LEAKED'}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "rescan.json").write_text(json.dumps({"from": str(src), "ok": ok, **res}, indent=1, sort_keys=True))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="compile_slice_zk.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("rescan", help="a finished prove's scan taken again from its summary and verifier files")
    r.add_argument("--from", required=True, help="the prove's --out")
    r.add_argument("--out", required=True)
    p = sub.add_parser("prove", help="the slice's units proved --zk and held by Lean")
    p.add_argument("--bin")
    p.add_argument("--lean")
    p.add_argument("--out", required=True)
    p.add_argument("--port", type=int, default=7781)
    p.add_argument("--delta", type=float, default=2.0 ** -20)
    p.add_argument("--points", default="", help="cost points: the k drawn units of each session, comma-separated")
    p.add_argument("--slice", choices=tuple(REFUSED), default="demo", help="the window the units are of")
    p.add_argument("--g", type=int, default=None, help="the steps a served row holds, dividing P (P by default: one srv "
                   "row a unit)")
    p.add_argument("--kinds", default="", help="the units' kinds in order, comma-separated, honest first (its cert row is "
                   "the certified record's); every kind of the slice by default")
    p.add_argument("--dry", action="store_true", help="the slice and the reference only: nothing lowered or proved")
    a = ap.parse_args(argv)
    if a.command == "rescan":
        return rescan(a)
    a.points = [int(k) for k in a.points.split(",") if k]
    kinds = tuple(REFUSED[a.slice])
    a.kinds = tuple(k for k in a.kinds.split(",") if k) or kinds
    if a.kinds[0] != "honest" or not set(a.kinds) <= set(kinds):
        raise SystemExit(f"--kinds: honest first, each of {', '.join(kinds)}")
    if not a.dry and not (a.bin and a.lean):
        raise SystemExit("prove needs --bin and --lean, or --dry")
    return prove(a)


if __name__ == "__main__":
    sys.exit(main())
