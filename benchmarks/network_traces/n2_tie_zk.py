"""The ingress tie in zero knowledge: each ingress link-window's IoTieRequest (`tie_service.IngressTie`, its unit
`NetworkIngressTieWindow_v1{B=512,T=4,R=2,OWN=1}`) committed by the warden, registered on the link's own stream, proved
by C-Flock `prove --zk` behind the Lean firewall, delivered, decided by the tie's verifier of record on its own
statement from the advice (`IngressTie.outcome(advice=…)`), and `advised.IoTieVerdict` on the link and the stream's
registrations built from the stream's outcomes of record alone (`IngressTie.tie`, read through `tie_service.Ties`, the
advised link's tie source).

    python benchmarks/network_traces/n2_tie_zk.py prove --bin flock-circuit --lean flock-verify --out DIR
      [--jobs 2] [--debug-clear] [--dry]

* **The link** (`SHAPE`, the pinned size class): REDTEAM phase 9's span, three id requests of 5, 40 and 5 prompt tokens
  in two windows of T = 4 buckets of R = 2 512-byte cells, request 1 in two frames, one in each window, behind each
  window's own frame (`io_request.windows_of`). The advice is v0 of one prefill step, each request one kept output,
  with the link's `received`; it is public, and the cells, their salts and the prover's switches are witness.
* **Members.** One stream each, over the link's two windows: ``honest``, tied in both; ``carried``, window 1 proved on
  its carry-in's prompt fields + 5 with its first ended request's + 5 (REDTEAM phase 9 (b)), which IoTieRequest accepts
  but which opens to no public row of the verifier's, so window 1 is refused; ``forged``, an advice whose request 1 is
  45 prompt tokens in the same frames (REDTEAM phase 9's forged advice), which IoTieRequest refuses in both windows;
  ``split``, REDTEAM phase 6's split, prompts of 40 and 5 tokens in cells the advice swaps, refused in window 0
  (`EXPECT`).
* **The statement and the rounds** follow `n2_zk`'s: one circuit, `IngressTie.proof_unit` lowered once (the gate the
  Definition its pinned digest names), and each link-window a statement of one instance on rows ``pub`` and ``wit``
  (`IngressTie.layout`) and the output row ``o0``, the verdict bit. A round registers, selects, proves live behind the
  firewall, delivers, and decides on the proof: the verifier's own Lean run, the ``pub`` row opened only to its own
  copy (its own statement from the advice) and the ``o0`` row read. Per window: prove (the session's wall time, the
  prover's ``prove_total_s``, proof bytes) and verify (the verifier of record's run, Lean included).
* **What the verifier receives** is `RECEIVED` per window and the advice (``advice.json``), scanned for every piece of
  the windows' request frames and their salts but those the public circuit, a pad cell or an own frame holds; the
  prover's instance files, which hold the cells, are the control.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import certifier_serve as cs  # noqa: E402
import n2_zk as Z  # noqa: E402
from verity.core.protocols.certifier import advice as ADV, advised as AD, head_check as HC  # noqa: E402
from verity.core.protocols.certifier import io_request as Q, io_tie as IT, service as SVC  # noqa: E402
from verity.core.protocols.certifier import tie_service as TS  # noqa: E402
from verity.core.protocols.certifier.active import id_request  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import partition as P  # noqa: E402

B = 512
SHAPE = Q.Shape(B, 4, 2, 1)
LINK, RUN = "requests", "n2-tie-zk"
#: REDTEAM phase 9's span: request 1 (40 tokens) takes 2 frames, one in each window
SPAN, COUNTS = (5, 40, 5), (1, 0, 0, 1, 1, 1, 0, 0)
#: REDTEAM phase 9's forged advice: request 1 45 tokens, in the same frames
FORGED = (5, 45, 5)
#: REDTEAM phase 6's prompts, whose frames the split advice swaps
SPLIT, SPLIT_COUNTS = (40, 5), (1, 1, 1, 0, 0, 0, 0, 0)
#: what the carried member adds to window 1's carried prompt fields and its first ended request's
CARRIED = 5
PROFILE = AD.AdvisedProfile(2, TS.framing(B, 64), SHAPE.OWN)
MEMBERS = ("honest", "carried", "forged", "split")
#: each member's windows' outcomes of record, which are its tie's
EXPECT = {"honest": (True, True), "carried": (True, False), "forged": (False, False), "split": (False, True)}
#: what the verifier receives per link-window, by file (`member`)
RECEIVED = {"registration.json": "the registration record (the warden's salted root)", "draw.json": "the stream's draw",
            "pub-1.bin": "the statement's public file", "session-*": "the session's files",
            "delivered.json": "the pub and o0 rows' salts and the output word", "outcome.json": "the outcome record",
            "advice.json": "the link's advice"}


def log(msg: str) -> None:
    print(f"N2TIE {time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


# --- the link and its members -----------------------------------------------------------------------------------------

def requests(prompt_lens) -> list[bytes]:
    return [id_request(model="glm", user=f"u{i}", prompt=[(7919 * i + 104729 * k) % 151329 for k in range(n)],
                       max_tokens=16, temperature=0.7, top_p=0.95, seed=i, frame_bytes=B)
            for i, n in enumerate(prompt_lens)]


def received(counts) -> tuple[int, ...]:
    return tuple(sum(counts[w * SHAPE.T:(w + 1) * SHAPE.T]) for w in range(len(counts) // SHAPE.T))


def advice(prompt_lens, got) -> AD.LinkAdvice:
    """An advice v0 of one prefill step, each request one kept output, with the link's `received`."""
    reqs = tuple(ADV.RequestAdvice(f"u{j}", n, 1, "eos") for j, n in enumerate(prompt_lens))
    steps = (ADV.StepAdvice(tuple(ADV.Row(j, "prefill", 0, n) for j, n in enumerate(prompt_lens))),)
    return AD.LinkAdvice(ADV.Advice(reqs, steps, tuple(range(len(prompt_lens)))), tuple(got))


def _carried(w: int, view: Q.RequestView) -> Q.RequestView:
    if w != 1:
        return view
    return view._replace(carry_in=view.carry_in._replace(fields=view.carry_in.fields + CARRIED),
                         ended=(view.ended[0] + CARRIED,) + view.ended[1:])


def members() -> dict[str, tuple]:
    """Each member's (cells by window, its advice, the prover's edit of the verifier's view, what was changed)."""
    cells, got = Q.windows_of(SHAPE, requests(SPAN), COUNTS), received(COUNTS)
    link = advice(SPAN, got)
    split = Q.windows_of(SHAPE, requests(SPLIT), SPLIT_COUNTS)
    return {"honest": (cells, link, None, {}),
            "carried": (cells, link, _carried, {"window": 1, "added_fields": CARRIED}),
            "forged": (cells, advice(FORGED, got), None, {"prompt_lens": list(FORGED)}),
            "split": (split, advice(SPLIT[::-1], received(SPLIT_COUNTS)), None, {"prompt_lens": list(SPLIT[::-1]),
                                                                                  "certified": list(SPLIT)})}


def link_windows(tie: TS.IngressTie, cells, link: AD.LinkAdvice, edit=None) -> SimpleNamespace:
    """Every link-window of the member as the warden and the prover hold it before it registers: the warden's
    commitment (fresh salts) and its root, the verifier's view of `link` (``own``) and the prover's (``view``, `edit`
    of it), the unit's arguments on the prover's view and the verifier's public arguments on its own, the output word
    by the gate's reference, and whether the prover's public row is the verifier's (``opens``)."""
    out = []
    for w, window in enumerate(cells):
        c = tie.commit(w, window)
        root = bytes.fromhex(tie.root(c))
        own = tie.view(link, w)
        view = own if edit is None or own is None else edit(w, own)
        args = None if view is None else tie.arguments(c, root, view)
        mine = None if own is None else tie.public_arguments(w, root, own)
        if args is None or mine is None:
            raise SystemExit(f"window {w}: the unit takes no instance of the advice's view")
        word = tie.outputs(c, root, view)
        opens = tie.row(mine, "pub") == tie.row(args, "pub")
        out.append(SimpleNamespace(window=w, opening=c, root=root, own=own, view=view, args=args, mine=mine, word=word,
                                   opens=opens, expect=int(word == 1 and opens), statement=None))
    return SimpleNamespace(windows=out, advice=link)


# --- the statement ----------------------------------------------------------------------------------------------------

def stager(M, tie: TS.IngressTie, out: Path) -> SimpleNamespace:
    """The one circuit: `IngressTie.proof_unit` lowered once on rows ``pub`` and ``wit`` with output row ``o0``, bound
    to the registered partition and Program, which are written beside it as the verifier's own."""
    Z.builders()
    rows = [(r, M.V2, tie.bits(r), list(tie.layout[r])) for r in TS.ROWS]
    out.mkdir(parents=True, exist_ok=True)
    (out / "partition.json").write_text(json.dumps(tie.partition, sort_keys=True))
    (out / "program.json").write_bytes(tie.program)
    bound = {"partition": M.P.digest(tie.partition), "program": tie.partition["program"]}
    t, was = time.time(), gc.isenabled()
    gc.disable()
    try:
        st = M.CS.UnitStager(tie.proof_unit, "unit/ingress-tie-window", rows, [(M.V2, (0,))],
                             circuit=out / "circuit.txt", bind=bound)
        st.pin
    finally:
        if was:
            gc.enable()
    got = [(p.name, p.n_bits) for p in st.ports]
    if got != [(p, tie.bits(p)) for p in TS.PORTS] or st.out_words != [1]:
        raise SystemExit(f"the statement's rows {got} are not the ones the verifier opens "
                         f"{[(p, tie.bits(p)) for p in TS.PORTS]}")
    k_log = st.meta["k_log"]
    if not 25 <= k_log + 3 <= 35:
        raise SystemExit(f"m = {k_log} + 3 is outside 25 .. 35 (--zk and flock-circuit's shape)")
    return SimpleNamespace(st=st, out=out, circuit=out / "circuit.txt", partition_file=out / "partition.json",
                           program_file=out / "program.json", lower_s=round(time.time() - t, 1), k_log=k_log,
                           circuit_sha512=M.PF.file_sha512(out / "circuit.txt"))


def stage(M, sg: SimpleNamespace, tie: TS.IngressTie, rounds: dict[str, SimpleNamespace], secret: bytes,
          out: Path) -> dict:
    """Every link-window's statement on the one circuit, one instance each, unit 0, as `n2_zk.stage` stages them: all
    rows evaluated at once (every unit satisfied, the output words the prover's references), held to the IR reference,
    then each written under salts from a key of its own. The staged ``o0`` row must open to the output word, and the
    ``pub`` row to the verifier's own copy exactly where the prover's view is the verifier's."""
    np, st = M.np, sg.st
    items = [(name, x) for name, rd in rounds.items() for x in rd.windows]
    rows = {r: np.stack([np.frombuffer(tie.row(x.args, r), "<u2") for _, x in items]) for r in TS.ROWS}
    t = time.time()
    flat = np.concatenate([rows[r] for r in st.row_ports], axis=1)
    got, _cut, ok = st.low.evaluate(flat.astype(np.uint64))
    if int(ok) != (1 << len(items)) - 1:
        raise SystemExit("a unit is unsatisfied on its rows")
    o0 = got.astype(np.uint16)[:, :1]
    words = [int(w) for w in o0[:, 0]]
    if words != [x.word for _, x in items]:
        raise SystemExit(f"the lowering's output words {words} are not the prover's references "
                         f"{[x.word for _, x in items]}")
    eval_s, t = round(time.time() - t, 1), time.time()
    st.reference_check(rows, {"o0": o0})
    ref_s, t = round(time.time() - t, 1), time.time()
    for k, (name, x) in enumerate(items):
        d = out / name / f"w{x.window:03d}" / "statement"
        iset = SimpleNamespace(**{**vars(st.iset), "name": f"{st.iset.name}-{name}-w{x.window}"})
        key = hashlib.sha512(secret + f"row salts/{name}/{x.window}".encode()).digest()[:32]
        salts = M.FC.stage_salts(iset, 0, 1, st.ports, key)
        d.mkdir(parents=True, exist_ok=True)
        M.FC.write(st.text, iset, 0, 1, d / "inst-1.bin", d / "pub-1.bin", key,
                   rows={r: rows[r][k:k + 1] for r in TS.ROWS}, outs={"o0": o0[k:k + 1]}, salts=salts,
                   public_words={}, digests=(st.pin, st.class_sha512), indices=[0], meta=st.meta)
        (d / "tables").mkdir(exist_ok=True)
        os.link(sg.circuit, d / "circuit.txt")
        bcs = SVC._commitments(d / "pub-1.bin", 0)
        if not (bcs and SVC._opens(tie.row(x.mine, "pub"), tie.bits("pub"), salts["pub"][0], bcs["pub"]) == x.opens
                and SVC._opens(SVC.output_row(words[k]), SVC.OUT_BITS, salts["o0"][0], bcs["o0"])):
            raise SystemExit(f"{name} window {x.window}: the staged rows don't open as the prover's views say")
        x.statement = SimpleNamespace(dir=d, salts={"pub": salts["pub"][0], "o0": salts["o0"][0]}, output=words[k])
    return {"statements": len(items), "evaluate_s": eval_s, "reference_s": ref_s, "write_s": round(time.time() - t, 1)}


# --- the rounds -------------------------------------------------------------------------------------------------------

def member(M, a, sg: SimpleNamespace, tie: TS.IngressTie, name: str, rd: SimpleNamespace, out: Path,
           port: int) -> dict:
    """The member's link-windows on its own stream, in order: the warden registers, the challenger selects, the prover
    proves the statement live behind the firewall and delivers, and the verifier of record decides on the proof
    (`IngressTie.outcome`); then the tie from the stream's outcomes of record (`IngressTie.tie`). Writes what the
    verifier receives under ``out / "verifier"`` (`RECEIVED`), each window's files prefixed by its window."""
    stream = tie.stream(budget=len(rd.windows), delta=cs.DELTA, challenger="n2-tie-zk", coins=cs._no_coins,
                        clock=time.monotonic)
    env, av = Z.firewall(M, sg), SimpleNamespace(bin=a.bin, lease=[], gpu=False)
    vdir = out / "verifier"
    vdir.mkdir(parents=True, exist_ok=True)
    digest = AD.digest(rd.advice)
    (vdir / "advice.json").write_text(json.dumps({"advice": ADV.to_json(rd.advice.advice),
                                                  "received": list(rd.advice.received), "digest": digest},
                                                 sort_keys=True))
    previous, rows = None, []
    for j, x in enumerate(rd.windows):
        wdir, pre = out / f"w{x.window:03d}", f"w{x.window:03d}-"
        reg = tie.register(stream, tie.registration(x.opening, previous))
        previous = reg.digest
        if reg.record["roots"][0]["root"] != x.root.hex():
            raise SystemExit(f"{name} window {x.window}: the registered root is not the statement's")
        draw = stream.select(reg)
        if list(draw.units) != [0]:
            raise SystemExit(f"{name} window {x.window}: the stream drew {list(draw.units)}, not the one unit")
        sdir = wdir / "session"
        m = {"dir": x.statement.dir, "n": 1, "pin": sg.st.pin}
        record, live, proc = M.a8.session(av, m, sdir, [0], port + j, env=env)
        sessions = tuple(sorted(p.parent for p in (sdir / "sessions").glob("*/session.json")))
        delivered = SVC.Delivered(M.PF.Proof(sg.circuit, x.statement.dir / "pub-1.bin", sessions), x.statement.salts,
                                  x.statement.output)
        late = stream.deliver(draw, delivered.sessions())
        t = time.time()
        outcome = late or tie.outcome(stream, draw, delivered, circuit_sha512=sg.circuit_sha512, advice=rd.advice,
                                      opening=x.opening if a.debug_clear else None)
        verify_s = round(time.time() - t, 1)
        body = stream.records[-1]
        verdict = (body.get("audit") or {}).get("verdicts", {}).get(TS.VERIFIER) or {}
        findings, refused = Z.LEAN_RUNS.get(str(delivered.proof.public), (None, "no Lean run"))
        proof = {"lean_accepted": refused is None and bool(findings) and all(f.accepted for f in findings),
                 "refused": refused, "proved": refused is None and tie.proved(findings or [])
                 and all(list(f.units) == [0] for f in findings),
                 "opened_word": TS._opened(delivered, 0, tie.row(x.mine, "pub"), tie.bits("pub"))}
        files = {"registration.json": reg.record, "draw.json": dict(draw.draw), "outcome.json": body,
                 "delivered.json": {"salts": {r: s.hex() for r, s in x.statement.salts.items()},
                                    "output": x.statement.output}}
        for f, v in files.items():
            (vdir / f"{pre}{f}").write_text(json.dumps(v, indent=1, sort_keys=True, default=str))
        os.link(x.statement.dir / "pub-1.bin", vdir / f"{pre}pub-1.bin")
        for f in sorted((sdir / "sessions").rglob("*")):
            if f.is_file():
                os.link(f, vdir / f"{pre}session-{f.parent.name}-{f.name}")
        proofs = live.get("proof_bytes") or []
        row = {"window": x.window, "registration": reg.digest, "accepted": outcome.accepted,
               "public": verdict.get("public"), "service_public": dict(outcome.public), "late": late is not None,
               "decided_by": verdict.get("decided_by"), "proof": proof, "expect": x.expect, "word": x.word,
               "opens": x.opens,
               "prove": {"session_s": proc["wall_s"], "prove_total_s": live.get("prove_total_s"),
                         "e2e_s": live.get("e2e_s"), "proof_bytes": sum(proofs) if proofs else None,
                         "k_log": live.get("k_log"), "prover_accepted": live.get("accepted"),
                         "max_child_rss_gb": proc["max_child_rss_gb"], "cpu_s": proc["cpu_s"]},
               "verify_s": verify_s, "server_verdict": ((record or {}).get("verdict") or {}).get("accepted"),
               **({"clear": verdict["clear"]} if "clear" in verdict else {})}
        row["as_expected"] = (row["accepted"] == bool(x.expect) and not row["late"] and row["decided_by"] == "proof"
                              and proof["proved"] and proof["opened_word"] == (x.word if x.opens else None)
                              and row["public"] == {"verdict": x.expect, "advice": digest})
        rows.append(row)
        log(f"{name} window {x.window}: {'accept' if outcome.accepted else 'reject'} public {row['public']}"
            f"{' LATE' if late else ''}, Lean {'proved' if proof['proved'] else 'NOT proved'}, output word "
            f"{proof['opened_word']}, prove {row['prove']['session_s']} s (prove_total {row['prove']['prove_total_s']}"
            f" s, {row['prove']['proof_bytes']} B), verify {verify_s} s; "
            f"{'as expected' if row['as_expected'] else 'UNEXPECTED'}")
    tied = TS.Ties(tie, stream).verdict(tie.link, rd.advice, PROFILE)
    return {"windows": rows, "tie": {"link": tied.link, "registrations": list(tied.registrations),
                                     "advice": tied.advice, "accepted": list(tied.accepted)}}


def blobs_of(rd: SimpleNamespace) -> tuple[dict[str, bytes], set[str]]:
    """The witness pieces the verifier must not receive: each window's request frames, which the prover's instance
    files hold as they are, and its bucket salts. Pad cells and own frames are public."""
    blobs, held = {}, set()
    for x in rd.windows:
        public = {IT.pad_cell(B), Q.own_frame(x.window, B)}
        for i, cell in enumerate(x.opening.cells):
            if cell not in public:
                blobs[f"cell {i} w{x.window}"] = cell
                held.add(f"cell {i} w{x.window}")
        for t, s in enumerate(x.opening.salts):
            blobs[f"salt {t} w{x.window}"] = s
    return blobs, held


def _member_scan(M, sg, rd, out: Path) -> dict:
    public = sg.circuit.read_bytes() + IT.pad_cell(B) + b"".join(Q.own_frame(x.window, B) for x in rd.windows)
    blobs, held, excluded = Z.public_pieces(M.a8, *blobs_of(rd), public)
    control = out / "prover-instances.bin"
    with open(control, "wb") as f:
        for x in rd.windows:
            f.write((x.statement.dir / "inst-1.bin").read_bytes())
    got = M.a8.received(out / "verifier", control, blobs, held)
    return {**got, "public_pieces": excluded, "files": len(got["files"])}


def prove(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    tie = TS.IngressTie(LINK, SHAPE, RUN, proof_deadline=cs.PROOF_DEADLINE_S)
    plan = members()
    rounds = {name: link_windows(tie, cells, link, edit) for name, (cells, link, edit, _) in plan.items()}
    summary: dict = {"link": {"name": LINK, "shape": list(tie.key), "unit": tie.unit.id,
                              "program": tie.partition["program"], "partition": P.digest(tie.partition),
                              "gates": [g.id for g in tie.gates()], "pinned": all(HC.pinned(g) for g in tie.gates()),
                              "bits": {p: tie.bits(p) for p in TS.PORTS}, "pad": tie.pad},
                     "changed": {name: what for name, (_, _, _, what) in plan.items()},
                     "advice": {name: AD.digest(rd.advice) for name, rd in rounds.items()},
                     "debug_clear": a.debug_clear, "received": RECEIVED}
    summary["instances_s"] = round(time.time() - t, 1)
    for name, rd in rounds.items():
        summary.setdefault("members", {})[name] = {
            "reference": [{"window": x.window, "word": x.word, "opens": x.opens, "expect": x.expect}
                          for x in rd.windows]}
        log(f"{name}: words {[x.word for x in rd.windows]}, pub rows the verifier's {[x.opens for x in rd.windows]}, "
            f"expected verdicts {[x.expect for x in rd.windows]}")
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
    summary["stager"] = {"unit": tie.proof_unit.id, "lower_s": sg.lower_s, "k_log": sg.k_log, "pin": sg.st.pin,
                         "circuit_sha512": sg.circuit_sha512, "ands": sg.st.low.layout.counts["ands"]}
    log(f"stager: {tie.proof_unit.id}, {sg.lower_s} s, k_log {sg.k_log}, {summary['stager']['ands']} ANDs, "
        f"pin {sg.st.pin[:16]}")
    summary["staging"] = stage(M, sg, tie, rounds, os.urandom(32), out)
    log(f"staged: {summary['staging']}")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))

    def go(k: int, name: str) -> tuple[str, dict]:
        res = member(M, a, sg, tie, name, rounds[name], out / name, a.port + 100 * k)
        res["received"] = _member_scan(M, sg, rounds[name], out / name)
        return name, res

    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        done = dict(pool.map(lambda kv: go(*kv), enumerate(rounds)))
    ok = True
    for name, res in done.items():
        rec = res["received"]
        good = (all(r["as_expected"] for r in res["windows"]) and tuple(res["tie"]["accepted"]) == EXPECT[name]
                and res["tie"]["advice"] == AD.digest(rounds[name].advice) and res["tie"]["link"] == LINK
                and len(res["tie"]["registrations"]) == len(rounds[name].windows)
                and not rec["verifier_found"] and rec["control"]["holds_all"])
        ok &= good
        summary["members"][name].update(res, ok=good)
        log(f"member {name}: tie {res['tie']['accepted']}, windows {[int(r['accepted']) for r in res['windows']]}, "
            f"verifier received {rec['verifier_bytes']} bytes in {rec['files']} files, witness pieces found there: "
            f"{rec['verifier_found'] or 'none'} (of {rec['blobs']} blobs; {len(rec['public_pieces'])} public pieces "
            f"left out), control {rec['control']['blobs_found']}; {'as expected' if good else 'UNEXPECTED'}")
    summary["tie"] = {name: res["tie"]["accepted"] for name, res in done.items()}
    costs = [r for res in done.values() for r in res["windows"]]
    summary["per_window"] = {
        "of": f"each of the {len(costs)} link-windows, one unit proved alone and decided by the verifier of record",
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
    p = sub.add_parser("prove", help="each ingress link-window's IoTieRequest proved --zk, decided by the tie's "
                                     "verifier of record, and the tie built from the outcomes")
    p.add_argument("--bin")
    p.add_argument("--lean")
    p.add_argument("--out", required=True)
    p.add_argument("--port", type=int, default=7981)
    p.add_argument("--jobs", type=int, default=2, help="members proved at once")
    p.add_argument("--debug-clear", action="store_true",
                   help="each verdict also carries the in-the-clear check on the opening, which decides nothing")
    p.add_argument("--dry", action="store_true", help="the instances and their references only: nothing lowered or "
                                                      "proved")
    a = ap.parse_args(argv)
    if not a.dry and not (a.bin and a.lean):
        raise SystemExit("prove needs --bin and --lean, or --dry")
    return prove(a)


if __name__ == "__main__":
    raise SystemExit(main())
