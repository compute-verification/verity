"""The two-stage toy (`toy.py`) end to end, its second stage proved with Flock and verified by Lean: serving's boundary
registration of N spans, the first stage's sample S of Q replay blocks after it, the proof server's interior registration of
S's Q·n_rho proof blocks after that, and the interior's audit by `a5.audit`.

* **The stream** (`stream_of`): the challenger's `service.Stream` over the toy's design (`toy.spec`), its `TwoStage` drawing
  both stages with Lean (`Select.select`, `Select.selectSecond`) on one challenger session's coins (`StageCoins`): a
  development session on this machine (`a5_coins.start`'s ``local``, on `ROOT`, pinned on ``--main``), withheld, each
  stage one sample of it after its registration's digest. Its samples show the path, not an auditor's coins.
* **The boundary** (`toy.boundary`): N replay blocks' ingress and egress rows under fresh salts, registered on the stream
  (`Stream.register`, held to the challenger's own window), and the first stage's sample S of Q of them.
* **The interior** (`toy.interior`): S replayed from the boundary's ingress rows, committed template-major and wired as the
  replay block's edges, each wired port its writer's output, one tree a value (`Member.wired_trees` off), each link
  member's boundary port under the boundary's salts at S, registered on the stream after the
  boundary (`Stream.register_interiors`); its window names the boundary registration and S, which the verifier holds it to
  (`source_of`). The stream's second stage samples ``--k`` proof blocks in each (replay block, template) cell.
* **The auditor's params** (`toy.references`), fixed before the run: each template's circuit file and the header of its
  reference statement, `member_stager`'s statement of its member in the interior window of Q replay blocks, staged on
  stand-in rows, and the registered reads it reads, `toy.reads`' with none of the run's fields
  (`service.template_circuits`' ``proofs``).
* **The second stage** (`second_stage`, over any interior window of word-row members, its two records and the verifier's
  reads): `a5.audit` of the stream's second sample: a ``--zk`` session per member with
  sampled proof blocks, run by ``flock-circuit prove --zk`` itself as SPOT[Flock]'s proof server does (`a0.DIRECT`; under
  ``--firewall`` behind the ZK server), Lean's verdicts on each against the verifier's own ``--registered`` reads of every
  port (`toy.reads`: the link ports at the boundary record, every other at the interior record), the audit record, the
  wires of the sampled proof blocks, and a5's negatives (the registered reads' three on b_in's session among them). Each
  session's circuit file is held to the one the params name for its template (`pins`), and its statement as round 2
  takes it (`statements`) to round 2's check (`second_stage.check`, the form allowed) at the params' reference headers
  and registered reads.
* **The outcome** (`outcome_of`): the sessions delivered to the stream and its outcome on what the interior's audit
  record took, with both stages' coin evidence once the session has ended, the sessions' statements and the params:
  the boundary's audit record over the replay sample and the interior's over the second sample, with round 2's check,
  and the run of record over both.
* **Refused run** ``--altered``: the proof server replays span S[0] on an ingress one bit off (`toy.one_bit_off`, a bit the
  fold's AND carries to the egress), so b_in's row and b_out's output are not the boundary's at S[0], every circuit and wire
  holding: Lean's registered reads refuse both link members ("does not open"), the fold and flips accepted.

Exit 0 when the run is as expected: every session's circuit file the one its template's params name, and every Lean
verdict accepted with every negative refused, round 2 accepting the statements, a5's record accepted or refusing only
the sessions' reads of the interior's own ports (`own_reads_only`: one-stage's record takes reads of earlier values
alone), and the outcome accepted or refused by round 2's two-stage form alone (`second_stage.FORM`), its records taking
those reads (the interior's at the interior record and the links' boundary ports at the boundary record); or
(``--altered``) refused at both link members by Lean's registered reads and nowhere else, and the outcome refused.
``--out`` holds the params
(``proofs.json``, and each template's reference statement under ``references/m<k>/``), both windows (``boundary/``,
``interior/``, their private files too), a5's records (``audit/``, the run's statement and verdicts in its
``case.json``), the challenger session's records (``challenger/``), both stages' coin evidence and the coins each read
(``coins.json``), the sessions' statements (``statements.json``) and ``summary.json`` (both stages' registrations, laws
and draws, and the outcome, among it).

    PYTHONPATH=<tree>:<tree>/benchmarks/numerical/python:<tree>/verity/ml/flock/python:<tree>/integrations/vllm \\
      python benchmarks/two_stage/audit.py --m0-pythonpath PATH --bin flock-circuit --lean flock-verify --main FILE \\
        --out DIR [--altered]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "one_stage"))
sys.path.insert(0, str(HERE.parent / "prove_path"))

import a5_coins as AC  # noqa: E402
import toy as T  # noqa: E402
from verity.core.protocols.spot import challenger as Ch, coins as CC, service as S  # noqa: E402

#: what Lean's registered reads refuse each link member of an ``--altered`` run by
ALTERED = "does not open"
#: what round 2 refuses every two-stage run by while the form is off (`second_stage.FORM`)
FORM_REFUSED = "two-stage-form"
#: the run root the run's challenger session opens on, which both registrations name
ROOT = hashlib.sha512(b"verity/two-stage/toy/run/v1").hexdigest()
CHALLENGER = "two-stage toy: a development challenger session on this machine (a5_coins local)"


def composer(defn, member: T.CS.Member):
    """``a5.audit``'s composer of a word-row member's template circuit: Flock's `member_stager`, bound to the window's
    partition and program."""
    def compose(circuit: Path, partition: str, program: str) -> dict:
        from verity_flock.class_statement import member_stager

        st = member_stager(defn, member, {"partition": partition, "program": program}, circuit)
        return {"subcircuit": st.meta.get("subcircuit"), "pin": st.pin, "class_sha512": st.class_sha512, "k_log": st.meta["k_log"]}
    return compose


def source_of(window: T.CS.Binding, boundary: str, sample: list[int]):
    """The verifier's check of the interior registration's window: a stand-in under the interior run of ``boundary`` and
    ``sample``, naming both."""
    def source(w: dict) -> dict:
        ok = {"kind": w.get("kind") == "stand-in", "run": w.get("run") == window.run, "interior": w.get("interior") is True,
              "boundary": w.get("boundary") == boundary, "replay_sample": w.get("replay_sample") == sample}
        return {"ok": all(ok.values()), "codes": [c for c, v in ok.items() if not v],
                "detail": {"kind": w.get("kind"), "run": w.get("run")}}
    return source


def publics(c: T.CS.Committed) -> dict[str, bytes]:
    return {m.key: c.public_body(k) for k, m in enumerate(c.window.members)}


class StageCoins(CC.ChallengerCoins):
    """Both stages' coins of the run's one challenger session (``draws``, `a5_coins.Draws`' proxy) as the stream reads
    them: the first stage's rank 0's, the second's rank 1's, each committed to its registration's digest in turn."""

    def __init__(self, draws) -> None:
        self.stages, self.at = (AC.RankCoins(draws, 0), AC.RankCoins(draws, 1)), -1

    def commit(self, after: bytes) -> None:
        self.at += 1
        self.stages[self.at].commit(after)

    def __call__(self, n: int) -> bytes:
        return self.stages[self.at](n)


def needs(stream: S.Stream, n: int, q: int) -> tuple[int, int]:
    """The most coins each stage's sample reads, known before the session opens (`a5_coins.need`'s rule): the first stage's
    subset of ``q`` of the ``n`` spans, and the second's cells of ``q`` replay blocks, whichever the first stage samples."""
    w, reads = T.boundary_window(n, q), []

    def zeros(k: int) -> bytes:
        reads.append(k)
        return bytes(k)
    S.select_second(list(range(q)), stream.cells, stream.counts, zeros)
    return AC.need(f"rate:{q}/{n}", w.program, w.query, n), -(-(sum(reads) + max(reads, default=0)) // Ch.COIN_BYTES)


def stream_of(a: argparse.Namespace, coins: StageCoins) -> S.Stream:
    """The challenger's two-stage stream over the toy (`toy.spec`, its second stage ``a.k`` proof blocks a cell) on the
    run's challenger session, which opens once both stages have planned the coins their samples read (each plan
    returns only then, so the two plan at once)."""
    stream = S.Stream(T.spec(a.n, a.q, a.k, a.session_timeout * len(T.TEMPLATES)), list(T.TEMPLATES), replay=T.REPLAY,
                      budget=1, delta=a.delta, challenger=CHALLENGER, coins=coins)
    with ThreadPoolExecutor(len(coins.stages)) as ex:
        list(ex.map(lambda c, n: c.plan(n), coins.stages, needs(stream, a.n, a.q)))
    return stream


def drawn(stream: S.Stream, coins: StageCoins, registration: S.WindowCommitmentRecord) -> tuple[S.Draw, dict]:
    """The stream's one sample of ``registration``, on the coins opened after its digest, and their segment (which ends
    the stage's turn on the session's client)."""
    coins.commit(bytes.fromhex(registration.digest))
    try:
        draw = stream.select(registration)
    finally:
        segment = coins.stages[coins.at].drawn()
    return draw, segment


def outcome_of(stream: S.Stream, second: S.Draw, handed: dict, evidence: list[dict | None], statements: list[dict],
               proofs: dict) -> dict:
    """The stream's outcome of the second stage's sample: the sessions delivered (`Stream.deliver`), then
    `Stream.outcome` on what the interior's audit record took (``handed``, a5's), each verdict of record naming the
    sessions, with both stages' coin evidence, the sessions' ``statements`` and the params ``proofs``: accepted, the
    checks each stage's audit record failed (each record taking the sessions' reads of its registration's own ports,
    `verifier.audit_record`'s ``own_reads``), and the codes of round 2's check in the interior's (``statements``)."""
    sessions = handed["sessions"]
    verdicts = {x: {**y, "sessions": sessions} if y.get("of_record") else y for x, y in handed["verdicts"].items()}
    given = {x: handed[x] for x in ("proved_units", "instances", "served", "session_roots")}
    try:
        o = stream.deliver(second, sessions) or stream.outcome(second, verdicts, handed["statement"], coins=evidence[0],
                                                               interior_coins=evidence[1], statements=statements,
                                                               proofs=proofs, **given)
    except (S.Refused, S.Pending) as e:
        return {"accepted": False, "refused": e.verdict.to_json() if isinstance(e, S.Refused) else str(e)}
    body = stream.records[-1]
    audits = {"interior": body["audit"], "replay_sample": (body.get("replay_sample") or {}).get("audit")}
    return {"accepted": o.accepted, "record": o.record, "symbol": body["symbol"],
            **{k: None if r is None else sorted(x for x, c in r["checks"].items() if not c["ok"]) for k, r in audits.items()},
            "registered": {k: ((r or {}).get("checks") or {}).get("registered") for k, r in audits.items()},
            "statements": ((body["audit"].get("checks") or {}).get("statements") or {}).get("codes")}


def stage_of(registration: S.WindowCommitmentRecord, draw: S.Draw) -> dict:
    return {"registration": registration.digest, "law": dict(draw.law), "draw": draw.units}


def own_ports(window: T.CS.Binding, drawn: list[int]) -> list[str]:
    """The interior registration's own ports the sessions read: the port holding each port but its link port of each
    member the second stage drew a unit of (`Binding.held`: a wired port's, its writer's output port), each once."""
    return sorted({window.held(m, p) for t, m in enumerate(window.members) if any(m.base <= u < m.base + m.n for u in drawn)
                   for p, _ in m.all_rows if p != T.link(t)})


def own_reads_only(record: dict | None, ports: list[str]) -> bool:
    """Whether a5's audit record refuses only the sessions' reads of its registration's own ``ports``: its one failed
    check is one-stage's ``registered``, which takes reads of earlier values alone, naming no registered value and
    exactly those ports. `SpotFlock.FrameIs` asks for those reads (an interior port's at the interior record, a link
    port's at the boundary record's leaf S[i]), and the stream's two-stage records take them (`verifier.audit_record`'s
    ``own_reads``)."""
    bad = {k: v for k, v in ((record or {}).get("checks") or {}).items() if not v.get("ok")}
    d = (bad.get("registered") or {}).get("detail") or {}
    return set(bad) == {"registered"} and d.get("registered") == [] == d.get("wrong") and d.get("opened") == ports


def statements(audit_dir: Path, window: T.CS.Binding) -> list[dict]:
    """Each member session's statement as round 2 takes it (``{"template", "header", "draw", "registered"}``): its
    template, its public file's header, its session record's ``unit_draw`` and the reads it was verified against. A
    member the second stage drew nothing of has no session and no statement."""
    import a0

    out = []
    for k, m in enumerate(window.members):
        st, rec = audit_dir / f"m{k}", next((audit_dir / f"session-m{k}" / "sessions").glob("*/session.json"), None)
        if rec is None:
            continue
        reads = st / "registered.json"
        out.append({"template": m.template, "header": a0.public_header(st, m.n),
                    "draw": json.loads(rec.read_text()).get("unit_draw"),
                    "registered": json.loads(reads.read_text()) if reads.exists() else None})
    return out


def pins(proofs: dict, stmts: list[dict]) -> dict[str, bool]:
    """Each session's circuit file against the one the auditor's params ``proofs`` name (`toy.references`: `member_stager`'s
    file for the template, built afresh from the template and its member alone, bound to the interior's partition and
    program): by template, whether the SHA-512 its header names (``circuit_sha512``) is the params'."""
    named = {e["template"]: e["circuit"] for e in proofs["templates"]}
    return {x["template"]: named.get(x["template"]) == x["header"].get("circuit_sha512") for x in stmts}


def round_two(sample: list[int], window: T.CS.Binding, irec: dict, brec: dict, drawn: list[int], stmts: list[dict],
              record: dict, proofs: dict) -> dict:
    """Round 2's statement check of the sessions (`second_stage.check`, the form allowed): over the links and wires the
    replay block's Definition makes (`second_stage.replay_links`, at the boundary record's ports, and
    `second_stage.replay_wires`, as the stream derives them), the reads the audit record's verdicts of record report,
    what they name of each session they accepted (`second_stage.verified_of`), and each template's reference header and
    registered reads the params ``proofs`` give."""
    from verity.core.protocols.spot import second_stage as SS
    from verity.core.protocols.spot.one_stage import audit as A, partition as P
    from verity.ml.circuits.codec import descriptor_id

    replay = T.boundary_window(int(brec["population"]), len(sample)).program
    verdicts = record.get("verdicts") or {}
    own = SS.Own(tuple(sample), SS.parts(len(sample), P.strata(window.program, window.query)), irec, brec, tuple(drawn),
                 SS.replay_links(replay, descriptor_id(T.REPLAY), brec), A.session_registered(verdicts),
                 verified=SS.verified_of(verdicts),
                 refs=tuple((e["template"], e["header"]) for e in proofs["templates"]),
                 ref_reads=tuple((e["template"], e["registered"]) for e in proofs["templates"]),
                 wires=SS.replay_wires(replay, descriptor_id(T.REPLAY)))
    return SS.check(True, own, stmts).to_json()


def second_stage(a: argparse.Namespace, sets: dict, window: T.CS.Binding, window_dir: Path, source, brec: dict, irec: dict,
                 sample: list[int], second: tuple[dict, dict, dict], reads: dict[str, dict], proofs: dict, out: Path,
                 provenance: dict) -> tuple[dict, bool, dict, dict, list[dict] | None]:
    """The second stage over the interior registration ``irec`` of ``window`` (written at ``window_dir``), chained after
    the boundary registration ``brec`` and its replay sample: `a5.audit` of the stream's second sample ``second`` (its
    draw, law and the interior registration's receipt), each member's session verified against ``reads`` (by member
    key, the verifier's own ``--registered`` files), the audit record, and each session's statement held to round 2 at
    the params ``proofs`` (`toy.references`). -> (a5's summary, its acceptance, the audit record, what the run reports,
    the sessions' statements: None without a draw)."""
    import a5
    import records_audit as RA

    a.coins, a.rule, a.window_root, a.compose_cache, a.transfer_gate = None, None, None, None, []
    a.tie = {"log": [brec]}
    port = a.port or RA.free_ports(a5.RANK_PORTS)
    s, ok = a5.audit(a, sets, out / "audit", window, window_dir, source, port, provenance, registered=reads, drawn=second)
    rec_file = out / "audit" / "audit.json"
    record = json.loads(rec_file.read_text()) if rec_file.exists() else {}
    got = {"audit": {x: s.get(x) for x in ("checks_before_draw", "refused", "audit", "draw", "draw_counts", "negatives",
                                           "circuits", "timing")},
           "accepted": ok, "record": {"failed": sorted(k for k, v in (record.get("checks") or {}).items() if not v.get("ok"))}}
    stmts = None
    if s.get("draw") is not None:
        stmts = statements(out / "audit", window)
        (out / "statements.json").write_text(json.dumps(stmts, indent=1, sort_keys=True))
        got["pins"] = pins(proofs, stmts)
        got["round_two"] = round_two(sample, window, irec, brec, s["draw"], stmts, record, proofs)
    return s, ok, record, got, stmts


def expected(a: argparse.Namespace, s: dict, ok: bool, record: dict, window: T.CS.Binding, r2: dict | None,
             o: dict | None, files: dict | None) -> bool:
    """Both: every check before the sample passed and every session's circuit file the one its template's params name
    (`pins`, ``files``). Honest: every Lean verdict accepted, the sessions' registered reads the verifier's own, every
    negative refused, a5's record accepted or refusing only the reads of the interior's own ports (`own_reads_only`),
    round 2 ``r2`` accepting the statements, and the stream's outcome ``o`` accepted or refused by round 2's two-stage
    form alone (`FORM_REFUSED`, while `second_stage.FORM` is off): the interior's audit record failing no check but
    round 2's, by that code at most, and the replay sample's none. ``--altered``: Lean's verify refused both link
    members (m0, m4) by `ALTERED` and accepted the fold and flips, round 2 refuses the statements, and the outcome
    refuses."""
    audit, why = s.get("audit") or {}, (s.get("audit") or {}).get("why") or {}
    before = all((s.get("checks_before_draw") or {}).values()) and bool(files) and all((files or {}).values())
    if not a.altered:
        refused = [v if isinstance(v, bool) else bool(v.get("refused")) for v in (s.get("negatives") or {}).values()]
        own = own_reads_only(record, own_ports(window, s.get("draw") or []))
        return (before and audit.get("complete") is True and audit.get("registered_reads") is True
                and all(v is True for v in (audit.get("verdicts") or {}).values()) and bool(refused) and all(refused)
                and (ok or own) and (r2 or {}).get("ok") is True and o is not None
                and (o.get("accepted") is True
                     or (o.get("interior") in ([], ["statements"]) and o.get("statements") in ([], [FORM_REFUSED])
                         and o.get("replay_sample") == [])))
    links = [f"lean-verify/m{t}" for t in T.LINKS]
    rest = [f"lean-verify/m{t}" for t in range(len(T.TEMPLATES)) if t not in T.LINKS]
    return (before and audit.get("accepted") is False and all(ALTERED in (why.get(x) or "") for x in links)
            and all((audit.get("verdicts") or {}).get(x) is True for x in rest) and (r2 or {}).get("ok") is False
            and o is not None and o.get("accepted") is False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="audit.py", description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=4, help="N: the spans serving commits")
    ap.add_argument("--q", type=int, default=2, help="Q: the replay blocks the first stage samples")
    ap.add_argument("--k", type=int, default=1, help="the second stage's proof blocks in each (replay block, template) cell")
    ap.add_argument("--m0-pythonpath", required=True)
    ap.add_argument("--bin", required=True, help="flock-circuit")
    ap.add_argument("--lean", required=True, help="flock-verify")
    ap.add_argument("--firewall", help="Lean's flock-firewall: every session behind the ZK server; without it each runs "
                    "prove --zk itself, as SPOT[Flock]'s proof server does")
    ap.add_argument("--altered", action="store_true", help="span S[0] replayed on an ingress one bit off")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--port", type=int, default=0, help="the sessions' first port (0: a free block)")
    ap.add_argument("--session-timeout", type=float, default=3600)
    ap.add_argument("--main", required=True, help="the verifier's copy of main's first-parent history (`git rev-list "
                    "--first-parent origin/main`), which the challenger session's pin is held to")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    manager, draws, finisher = AC.start("local", ROOT, [0, 1], a.out / "challenger", a.main)
    try:
        return run(a, a.out, StageCoins(draws), draws)
    finally:
        for rank in (0, 1):
            draws.skip(rank)
        finisher.join()
        manager.shutdown()


def run(a: argparse.Namespace, out: Path, coins: StageCoins, draws) -> int:
    """Both stages on the stream, the second stage's sessions, and the stream's outcome on the session's evidence."""
    stream = stream_of(a, coins)
    summary = {"n": a.n, "q": a.q, "n_rho": T.N_RHO, "k": a.k, "altered": a.altered, "root": ROOT,
               "stream": {"digest": stream.digest, "cells": list(stream.cells), "counts": list(stream.counts)}}

    def refused(stage: str, e: Exception) -> int:
        summary[stage] = {"refused": e.verdict.to_json() if isinstance(e, S.Refused) else str(e)}
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
        print("TWO " + json.dumps({f"{stage}_refused": summary[stage]["refused"]}, default=str))
        return 1
    proofs = T.references(a.q, out / "references")
    S.template_circuits(proofs, stream.spec.check.query)
    (out / "proofs.json").write_text(json.dumps(proofs, indent=1, sort_keys=True))
    bc = T.boundary(T.ingress(a.n), a.q)
    brec = T.write(bc, out / "boundary", {"toy": "two-stage boundary"})
    try:
        reg = stream.register(brec, T.CS.expected_ports(bc.window), leaf_layer=T.CS.leaf_layer(publics(bc)), window_root=ROOT)
        first, segment = drawn(stream, coins, reg)
    except (S.Refused, Ch.CoinError) as e:
        return refused("first_stage", e)
    bd, sample, segments = reg.digest, first.units, [segment]
    summary.update(boundary=bd, first_stage=stage_of(reg, first))
    replayed = None
    if a.altered:
        replayed = T.one_bit_off(T.LK.Port.of(bc, T.boundary_ports(bc.window)["ingress"]).rows_at(sample))
    it = T.interior(bc, bd, sample, {"law": "subset", "k": len(sample) * T.N_RHO}, replayed=replayed)
    irec = T.write(it.committed, out / "interior", {"interior": True, "boundary": bd, "replay_sample": sample}, previous=bd,
                   registered=it.registered)
    try:
        ireg = stream.register_interiors(first, irec, T.CS.expected_ports(it.window),
                                         leaf_layer=T.CS.leaf_layer(publics(it.committed)), window_root=ROOT)
        second, segment = drawn(stream, coins, ireg)
    except (S.Refused, Ch.CoinError) as e:
        return refused("second_stage", e)
    segments.append(segment)
    summary["second_stage"] = stage_of(ireg, second)
    sets = {m.template: (composer(t, m), t) for m, t in zip(it.window.members, T.TEMPLATES)}
    s, ok, record, got, stmts = second_stage(a, sets, it.window, out / "interior", source_of(it.window, bd, sample), brec,
                                             irec, sample, (second.draw, second.law, ireg.receipt),
                                             T.reads(brec, irec, it.window, sample), proofs, out,
                                             {"toy": "two-stage interior", "altered": a.altered})
    summary.update(interior={"population": it.window.population, "law": it.window.law, "run": it.window.run,
                             "members": [{"key": m.key, "n": m.n, "base": m.base} for m in it.window.members],
                             "links": T.links(it.window)},
                   sample=sample, **got)
    drawn_units = s.get("draw") or []
    summary["record"]["own_reads_only"] = own_reads_only(record, own_ports(it.window, drawn_units))
    published = draws.evidence()
    summary["challenger"] = {x: published.get(x) for x in ("trusted", "error", "plan")}
    evidence = [AC.evidence(published, g) for g in segments]
    (out / "coins.json").write_text(json.dumps({"evidence": evidence, "read": [c.read.hex() for c in coins.stages]},
                                               indent=1, sort_keys=True))
    if "handed" in s:
        summary["outcome"] = outcome_of(stream, second, s["handed"], evidence, stmts, proofs)
    summary["as_expected"] = expected(a, s, ok, record, it.window, summary.get("round_two"), summary.get("outcome"),
                                      summary.get("pins"))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    r2, o = summary.get("round_two"), summary.get("outcome") or {}
    print("TWO " + json.dumps({**{x: summary.get(x) for x in ("sample", "accepted", "record", "pins", "as_expected")},
                               "round_two": r2 and {"ok": r2.get("ok"), "codes": r2.get("codes")},
                               "outcome": {x: o.get(x) for x in ("accepted", "interior", "replay_sample", "statements",
                                                                 "refused")}},
                              default=str))
    return 0 if summary["as_expected"] else 1


if __name__ == "__main__":
    sys.exit(main())
