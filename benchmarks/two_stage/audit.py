"""The two-stage toy (`toy.py`) end to end, its second stage proved with Flock and verified by Lean: serving's boundary
registration of N spans, the first stage's sample S of Q replay blocks after it, the proof server's interior registration of
S's Q·n_rho proof blocks after that, and the interior's audit by `a5.audit`.

* **The boundary** (`toy.boundary`): N replay blocks' ingress and egress rows under fresh salts, held to the verifier's own
  window (`verifier.check`) before the first stage.
* **The first stage**: Lean's sample of Q (``flock-verify draw``) on a development challenger's coins opened after the boundary
  registration (`a0.challenger_draw`), held to it by `verifier.check_draw`. Until the stream takes a replay block of several
  templates, the stages are drawn here and not by `service.Stream`'s `TwoStage`.
* **The interior** (`toy.interior`): S replayed from the boundary's ingress rows, committed template-major and wired as the
  replay block's edges, each link member's boundary port under the boundary's salts at S, its registration chained after the
  boundary's; its window names the boundary registration and S, which the verifier holds it to (`source_of`).
* **The second stage**: `a5.audit` of the interior on its own Lean sample (its law ``subset:K``, ``--k``, every unit by
  default) on a development challenger's coins opened after the interior registration: a ``--zk`` session per member with
  sampled proof blocks, run by ``flock-circuit prove --zk`` itself as SPOT[Flock]'s proof server does (`a0.DIRECT`; under
  ``--firewall`` behind the ZK server), Lean's verdicts on each against the verifier's own ``--registered`` reads of every
  port (`toy.reads`: the link ports at the boundary record, every other at the interior record), the audit record, the
  wires of the sampled proof blocks, and a5's negatives (the registered reads' three on b_in's session among them).
* **Refused run** ``--altered``: the proof server replays span S[0] on an ingress one bit off, so b_in's row and b_out's output
  are not the boundary's at S[0], every circuit and wire holding: Lean's registered reads refuse both link members ("does not
  open"), the fold and flips accepted.

Exit 0 when the run is as expected: every Lean verdict accepted with every negative refused, the record accepted or refusing
only the sessions' reads of the interior's own ports (`own_reads_only`: one-stage's record takes reads of earlier values
alone), or (``--altered``) refused at both link members by Lean's registered reads and nowhere else. ``--out`` holds both windows (``boundary/``, ``interior/``, their private files
too), the first stage (``first/``), a5's records (``audit/``) and ``summary.json``.

    PYTHONPATH=<tree>:<tree>/benchmarks/numerical/python:<tree>/verity/ml/flock/python:<tree>/integrations/vllm \\
      python benchmarks/two_stage/audit.py --m0-pythonpath PATH --bin flock-circuit --lean flock-verify --out DIR [--altered]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "one_stage"))
sys.path.insert(0, str(HERE.parent / "prove_path"))

import toy as T  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import registration as R, verifier as V  # noqa: E402

#: what Lean's registered reads refuse each link member of an ``--altered`` run by
ALTERED = "does not open"


def composer(defn, port: str, words: int):
    """``a5.audit``'s composer of a template's circuit: Flock's `UnitStager` over its one input row ``port`` and its ``o0``,
    bound to the window's partition and program."""
    def compose(circuit: Path, partition: str, program: str) -> dict:
        from verity_flock.class_statement import UnitStager

        st = UnitStager(defn, f"unit/two-stage-toy/{defn.id}", [(port, T.V2, words * T.WORD, [port])], [(T.V2, (0,))],
                        circuit=circuit, bind={"partition": partition, "program": program})
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


def first_stage(a: argparse.Namespace, bc: T.CS.Committed, brec: dict, out: Path) -> tuple[list[int] | None, dict]:
    """The boundary registration under the verifier's own window, then Lean's sample of Q after it: -> (S, what it read)."""
    import a0

    out.mkdir(parents=True, exist_ok=True)
    w = bc.window
    v = V.check(brec, w.program, w.query, w.law, T.CS.SCHEME, T.CS.expected_ports(w), leaf_layer=T.CS.leaf_layer(publics(bc)))
    got = {"registration": R.digest(brec), "check": v.to_json()}
    if not v.ok:
        return None, got
    p = a0.challenger_draw(a.lean, ["--population", str(a.n), "--k", str(a.q)], dict(w.law), a.n, R.digest(brec),
                           out / "draw.log")
    draw = json.loads(next(x for x in p.stdout.splitlines() if x.startswith("DRAW ")).split(" ", 1)[1])
    (out / "draw.json").write_text(json.dumps(draw, sort_keys=True))
    d = V.check_draw(draw, brec, dict(w.law))
    got.update(draw=draw["units"], draw_check=d.to_json())
    return (sorted(int(u) for u in draw["units"]) if d.ok else None), got


def own_ports(window: T.CS.Binding) -> list[str]:
    """The interior registration's own ports the sessions read: every member's every port but its link port."""
    return sorted(m.port(p) for t, m in enumerate(window.members) for p, _ in m.all_rows if p != T.link(t))


def own_reads_only(record: dict, window: T.CS.Binding) -> bool:
    """Whether the audit record refuses only the sessions' reads of the interior's own ports: its one failed check is
    one-stage's ``registered``, which takes reads of earlier values alone, naming no registered value and exactly those
    ports. `SpotFlock.FrameIs` asks for those reads, so the two-stage record check is to take them."""
    bad = {k: v for k, v in (record.get("checks") or {}).items() if not v.get("ok")}
    d = (bad.get("registered") or {}).get("detail") or {}
    return set(bad) == {"registered"} and d.get("registered") == [] == d.get("wrong") and d.get("opened") == own_ports(window)


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


def round_two(sample: list[int], window: T.CS.Binding, irec: dict, brec: dict, drawn: list[int], stmts: list[dict]):
    """Round 2's statement check of the sessions (`second_stage.check`, the form allowed), where the tree has it."""
    try:
        from verity.core.protocols.sampled_proofs import second_stage as SS
    except ImportError:
        return None
    from verity.core.protocols.sampled_proofs.one_stage import partition as P

    strata = [(tid, [[lo, lo + n]]) for tid, lo, n in P.ranges(window.program, window.query)]
    return SS.check(True, SS.Own(tuple(sample), SS.parts(len(sample), strata), irec, brec, tuple(drawn)), stmts).to_json()


def expected(a: argparse.Namespace, s: dict, ok: bool, record: dict, window: T.CS.Binding) -> bool:
    """Honest: every check before the sample passed, every Lean verdict accepted, the sessions' registered reads the
    verifier's own, every negative refused, and the record accepted or refusing only the reads of the interior's own
    ports (`own_reads_only`). ``--altered``: every check before the sample passed, and Lean's verify refused both link
    members (m0, m3) by `ALTERED` and accepted the fold and flips."""
    audit, why = s.get("audit") or {}, (s.get("audit") or {}).get("why") or {}
    before = all((s.get("checks_before_draw") or {}).values())
    if not a.altered:
        refused = [v if isinstance(v, bool) else bool(v.get("refused")) for v in (s.get("negatives") or {}).values()]
        return (before and audit.get("complete") is True and audit.get("registered_reads") is True
                and all(v is True for v in (audit.get("verdicts") or {}).values()) and bool(refused) and all(refused)
                and (ok or own_reads_only(record, window)))
    links = [f"lean-verify/m{t}" for t in T.LINKS]
    rest = [f"lean-verify/m{t}" for t in range(len(T.TEMPLATES)) if t not in T.LINKS]
    return (before and audit.get("accepted") is False and all(ALTERED in (why.get(x) or "") for x in links)
            and all((audit.get("verdicts") or {}).get(x) is True for x in rest))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="audit.py", description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=4, help="N: the spans serving commits")
    ap.add_argument("--q", type=int, default=2, help="Q: the replay blocks the first stage samples")
    ap.add_argument("--k", type=int, default=None, help="the second stage's units of the interior's Q·n_rho (default all)")
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
    a = ap.parse_args(argv)
    import a5
    import records_audit as RA

    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    bc = T.boundary(T.ingress(a.n), a.q)
    brec = T.write(bc, out / "boundary", {"toy": "two-stage boundary"})
    bd = R.digest(brec)
    sample, first = first_stage(a, bc, brec, out / "first")
    summary = {"n": a.n, "q": a.q, "n_rho": T.N_RHO, "altered": a.altered, "boundary": bd, "first_stage": first}
    if sample is None:
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
        print("TWO " + json.dumps({"first_stage_refused": first}, default=str))
        return 1
    k = a.k or a.q * T.N_RHO
    replayed = None
    if a.altered:
        port = T.LK.Port.of(bc, T.boundary_ports(bc.window)["ingress"])
        replayed = port.rows_at(sample).copy()
        replayed[0, -1] ^= 1
    it = T.interior(bc, bd, sample, {"law": "subset", "k": k}, replayed=replayed)
    irec = T.write(it.committed, out / "interior", {"interior": True, "boundary": bd, "replay_sample": sample}, previous=bd,
                   registered=it.registered)
    sets = {m.template: (composer(t, *rows[0]), t) for m, t, (rows, _outs) in zip(it.window.members, T.TEMPLATES, T.LAYOUT)}
    a.coins, a.rule, a.window_root, a.compose_cache, a.transfer_gate = None, None, None, None, []
    a.tie = {"log": [brec]}
    port = a.port or RA.free_ports(a5.RANK_PORTS)
    s, ok = a5.audit(a, sets, out / "audit", it.window, out / "interior", source_of(it.window, bd, sample), port,
                     {"toy": "two-stage interior", "altered": a.altered}, registered=T.reads(brec, irec, it.window, sample))
    summary.update(interior={"population": it.window.population, "law": it.window.law, "run": it.window.run,
                             "members": [{"key": m.key, "n": m.n, "base": m.base, "links": T.LINKS.get(t)}
                                         for t, m in enumerate(it.window.members)]},
                   sample=sample, audit={x: s.get(x) for x in ("checks_before_draw", "refused", "audit", "draw", "draw_counts",
                                                                "negatives", "circuits", "timing")},
                   accepted=ok)
    record = json.loads((out / "audit" / "audit.json").read_text()) if (out / "audit" / "audit.json").exists() else {}
    summary["record"] = {"failed": sorted(k for k, v in (record.get("checks") or {}).items() if not v.get("ok")),
                         "own_reads_only": own_reads_only(record, it.window)}
    if s.get("draw") is not None:
        stmts = statements(out / "audit", it.window)
        (out / "statements.json").write_text(json.dumps(stmts, indent=1, sort_keys=True))
        summary["round_two"] = round_two(sample, it.window, irec, brec, s["draw"], stmts)
    summary["as_expected"] = expected(a, s, ok, record, it.window)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    print("TWO " + json.dumps({x: summary.get(x) for x in ("sample", "accepted", "record", "as_expected")}, default=str))
    return 0 if summary["as_expected"] else 1


if __name__ == "__main__":
    sys.exit(main())
