"""The first recursive session's inner side (hiding model architectures; ``80-rec-inner.sh``): the inner
statement staged as M0's ``verity/flock-circuit`` over RoPE d64, proved by the unmodified prover against
``verity_flock.rec_live``'s proxy and challenger.

    python rec_inner.py stage --n N --sets DIR --out DIR [--plain-leaves] [--subcircuit ID] [--seed S]
                              [--sparse SHAPE --stacked FLOCK_CIRCUIT [--forge cell]]
                                                             circuit.txt, inst-N.bin, pub-N.bin, and the verifier's partition.json,
                                                             program.json and draw.json (its full sample, rec_live.full_draw);
                                                             prints the statement's shape
                                                             (--subcircuit: N seeded samples of that census-style id, RoPE d64 by default;
                                                             --plain-leaves: the trees' leaves plain SHA-512, circuit.LEAF_SCHEME_PLAIN;
                                                             --stacked: R registered at staging, rec_session.stage_stacked: the
                                                             circuit registration META binds, circuit-registration.json, and the
                                                             prover's R.bin, ROOT.json and circuit-registration-private.json)
    python rec_inner.py stage-window --window DIR --member K --subcircuit ID --sets DIR --out DIR [--zk-out DIR]
                                                             the same files over every unit of a served window's member K
                                                             (stage_window), plain leaves, the bodies the window's own bytes;
                                                             --zk-out: the member under the hm96 circuit too (85's stage-zk)
    python rec_inner.py forge --what challenge|query --inner DIR [--record session|prover]
                                                             flips one coin of rep 0 in DIR's session.json (session/'s, or
                                                             prover/'s: candidate 3's, challenge only) and public.json,
                                                             written in the challenger's own format (rec_live.record_bytes)
    python rec_inner.py forge --what nodraw|partial --inner DIR [--record session|prover]
                                                             the record without its sample, or with its last sampled proof block dropped
    python rec_inner.py forge --what top|unsalted --inner DIR
                                                             top: flips one byte of rep 0's level-1 top in private.json;
                                                             unsalted: public.json's tops as their clear nodes (no salts)
    python rec_inner.py release-parts --statements REC_STATEMENTS --windows-s CLASS=S,...
                                                             the recursive session's parts (V*'s statements as 85 names them,
                                                             in rec-statements' order) and its release windows (85's RELEASE)
    python rec_inner.py release-report --state DIR --session ID [--times FILE ...]
                                                             that session on the ZK server's record against its schedule:
                                                             each part's release time, when the Lean loop decided it, the
                                                             slack and the outcome; exit 1 unless every part was released
    python rec_inner.py tree-held --challenger DIR SESSION...
                                                             each rec_live session's coins held to the challenger's release
                                                             log (tree_held); exit 1 unless every one is its tree's
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from verity_flock import circuit as FC
from verity_flock import rec_budget as RB
from verity_flock import rec_live as RL
from verity_flock import rec_vstar as V
from verity_flock.record_text import session_text

SEED = 20261004
STREAM = "circuit/rep0"


def _population(sub, n: int, sets: Path, seed: int):
    """The input set of ``sub`` at ``n`` (seeded samples), its lowering, and the `verity/partition/v1` object
    `Q_template_instance` v0 over `population_program` of its template at ``n`` (instance i is unit i), with that program."""
    from verity.protocols.spot.verifier.one_stage import partition as P
    from verity_flock.input_sets import InputSet
    from verity_flock.synthetic import generate

    s = InputSet.open(generate(sub, n, seed, sets)["path"])
    template = s.subcircuit.definition(**FC.set_statics(s))
    program = P.population_program(template, n)
    return s, FC.lowering_for_set(s), P.partition(program, P.template_instance_query(template)), program


def _write_binding(out: Path, part: dict, program) -> None:
    """``partition.json``, ``program.json`` and the sample the challenger answers ``Register`` with, every unit
    (`rec_live.full_draw`), ``draw.json``."""
    from verity.circuits.codec import canonical_json, encode_program

    (out / "partition.json").write_text(json.dumps(part))
    (out / "program.json").write_bytes(canonical_json(encode_program(program)))
    (out / "draw.json").write_text(json.dumps(RL.full_draw(program, part["query"])))


def _shape(out: Path, low, sub_id: str, n: int, part: dict) -> dict:
    from verity.protocols.spot.verifier.one_stage import partition as P

    text = (out / "circuit.txt").read_text()
    meta = json.loads(text.split("\n", 2)[1][5:])
    G = meta["vus_per_block"]
    blocks = -(-n // G)
    return {"subcircuit": sub_id, "unit": low.name, "n": n, "vus_per_block": G, "k_log": meta["k_log"],
            "units_per_vu": meta["units_per_vu"], "unit_log": meta["unit_log"], "unit_rows": low.layout.useful,
            "unit_nnz": low.layout.counts["nnz_a"] + low.layout.counts["nnz_b"], "unit_ands": low.layout.counts["ands"],
            "blocks": blocks,
            "m": meta["k_log"] + max(3, (blocks - 1).bit_length()), "leaf_scheme": meta["leaf_scheme"]["id"],
            "circuit_sha512": FC.digest(text), "class_sha512": FC.class_digest(text), "partition_digest": P.digest(part),
            "program_sha512": part["program"], **({"registration": meta["registration"]} if "registration" in meta else {})}


def stage(n: int, sets: Path, out: Path, plain_leaves: bool = False, sparse=None, stacked: Path | None = None,
          forge: str | None = None, subcircuit: str | None = None, seed: int = SEED) -> dict:
    """The inner statement over ``n`` seeded samples of ``subcircuit`` (a census-style id, RoPE d64 by default), bound to the
    `verity/partition/v1` object `Q_template_instance` v0 over `population_program` of the set's template at ``n``, which
    every verifier of the inner statement holds (``--partition``, ``--rec-inner-partition``), written beside it as
    ``partition.json`` with its program, ``program.json``, and its full sample, ``draw.json``. ``stacked``
    (flock-circuit, with ``sparse``, the bucket's `rec_sparse.Shape`): the stacked table R registered with it here
    (`rec_session.stage_stacked`, ``forge`` ``cell`` its forgery), the circuit registration whose digest META binds."""
    from verity.protocols.spot.verifier.one_stage import partition as P
    from verity_flock import subcircuit_templates as TM

    RB.check_inner(n)
    sub = TM.parse_subcircuit_id(subcircuit) if subcircuit else TM.rope_head(64)
    s, low, part, program = _population(sub, n, sets, seed)
    FC.stage(low, s, 0, n, out, partition=P.digest(part), program=part["program"],
             leaf_scheme=FC.LEAF_SCHEME_PLAIN if plain_leaves else None)
    _write_binding(out, part, program)
    if stacked is not None:
        from verity_flock import rec_session as RS

        RS.stage_stacked(out, stacked, sparse, program, part, forge)
    return _shape(out, low, s.subcircuit.id, n, part)


def stage_window(window: Path, k: int, subcircuit: str, sets: Path, out: Path, zk_out: Path | None = None) -> dict:
    """The inner statement over every unit of a served window's member ``k`` (``prove_path.py commit``'s window dir:
    ``window.json``, ``registration.json``, ``m<k>-public.bin`` and the prover's ``m<k>-private.bin``), with plain SHA-512
    leaves, under the circuit the verifier composes for the member's template from an input set of ``subcircuit`` (its
    lowering alone, as ``compose_circuit.py --input-set`` composes it), bound to the window's own partition object: the one
    `stage` makes at the member's ``n``, which must be the window's (``partition_digest``), so the window holds that one
    member alone. The bodies are the window's bytes (`flock_inputs.flock_input_files`): every row's salt and ``b || c`` are
    the ones the window's registration commits, so ``pub-<n>.bin``'s body is ``m<k>-public.bin`` itself, which the result's
    ``public_body`` states. ``zk_out``: the same member under the hm96 circuit as well (today's ``--zk`` statement, 85's
    ``stage-zk``)."""
    from verity.protocols.spot.verifier.one_stage import partition as P
    from verity.commitments.windows.flock_inputs import flock_input_files
    from verity_flock import subcircuit_templates as TM

    win = json.loads((window / "window.json").read_text())
    mem = win["members"][k]
    n = int(mem["n"])
    RB.check_inner(n)
    s, low, part, program = _population(TM.parse_subcircuit_id(subcircuit), n, sets, SEED)
    if P.digest(part) != win["partition_digest"]:
        raise ValueError(f"{subcircuit} at n = {n}: partition {P.digest(part)[:16]}, not the window's "
                         f"{win['partition_digest'][:16]} (a window of more than member {k}, or another template)")
    public = (window / f"m{k}-public.bin").read_bytes()
    got = {}
    for d, leaves in ((out, FC.LEAF_SCHEME_PLAIN), *(((zk_out, None),) if zk_out else ())):
        d.mkdir(parents=True, exist_ok=True)
        (d / "circuit.txt").write_text(FC.compose(low, s.subcircuit, partition=P.digest(part), program=part["program"],
                                                  leaf_scheme=leaves))
        files = flock_input_files(window, k, (d / "circuit.txt").read_text(), d)
        for kind in ("pub", "inst"):
            Path(files[kind]).rename(d / f"{kind}-{n}.bin")
        body = (d / f"pub-{n}.bin").read_bytes().split(b"\n", 1)[1]
        if body != public:
            raise ValueError(f"{d}/pub-{n}.bin: its body is not the window's m{k}-public.bin")
        _write_binding(d, part, program)
        got[str(d)] = files["circuit_sha512"]
    return {**_shape(out, low, s.subcircuit.id, n, part), "window": {
        "run": win["run"], "registration_digest": win["registration_digest"], "member": k, "template": mem["template"],
        "public_sha512": mem["public_sha512"]}, "public_body": hashlib.sha512(public).hexdigest(),
        "zk_circuit_sha512": got.get(str(zk_out))}


def forge(what: str, inner: Path, level: int = 0, record: str = "session") -> dict:
    """Flip the low bit of one coin of rep 0, in the session record (``record``: ``session/``'s, or ``prover/``'s, which
    candidate 3's relay writes and whose rounds after z_partial are numbered apart, so only ``challenge``) and in the challenger's
    live-coin part's public record alike: the first coin after the binding round (``challenge``) or the first query coin of Ligerito
    level ``level`` (``query``). ``nodraw`` and ``partial`` forge the record's sample (none, or its last unit dropped);
    ``top`` and ``unsalted`` forge the tree tops instead (the module docstring)."""
    if what in ("nodraw", "partial"):
        f = inner / record / "session.json"
        sess = json.loads(f.read_text())
        if what == "nodraw":
            sess["unit_draw"] = None
        else:
            sess["unit_draw"]["units"] = sess["unit_draw"]["units"][:-1]
        f.write_text(session_text(sess))
        return {"forged": what, "record": record}
    if what == "query" and record != "session":
        raise ValueError("query: the prover's record numbers its rounds after z_partial apart from public.json's")
    pub, priv, proofs = V.load(inner)
    if what == "top":
        p = next(p for p in priv["rounds"] if p["stream"] == STREAM and p["k"] > 0 and p.get("tops"))
        nodes = p["tops"][0]["nodes"]
        nodes[0] = format(int(nodes[0][:2], 16) ^ 1, "02x") + nodes[0][2:]
        (inner / "private.json").write_text(json.dumps(priv))
        return {"forged": what, "stream": STREAM, "k": p["k"], "top": 0}
    if what == "unsalted":
        clear = {(p["stream"], p["k"]): [o["nodes"] for o in p.get("tops", [])] for p in priv["rounds"]}
        for r in pub["rounds"]:
            r["tops"] = clear.get((r["stream"], r["k"]), [])
        pub["link"]["roots"] = {t: v["nodes"] for t, v in priv["roots"].items()}
        del pub["tops"]
        (inner / "public.json").write_text(json.dumps(pub))
        return {"forged": what}
    if what == "challenge":
        k = next(r["k"] for r in V._stream_rounds(pub, STREAM) if r["k"] > 0 and r["n"])
    else:
        k = V.query_rounds(pub, STREAM, [len(lv.rows) for lv in V.decode(proofs[0])["levels"]])[level]["k"]
    sess = json.loads((inner / record / "session.json").read_text())
    for rd in (next(r for r in next(s for s in sess["streams"] if s["stream"] == STREAM)["rounds"] if r["k"] == k),
               next(r for r in pub["rounds"] if r["stream"] == STREAM and r["k"] == k)):
        c = rd["coins"][0]
        rd["coins"][0] = c[:-1] + format(int(c[-1], 16) ^ 1, "x")
    (inner / record / "session.json").write_text(json.dumps(sess))
    (inner / "public.json").write_text(RL.record_bytes(pub))
    return {"forged": what, "record": record, "stream": STREAM, "k": k, "coin": 0, **({"level": level} if what == "query" else {})}


def release_parts(units: list[str]) -> list[str]:
    """V*'s statements (``rec-statements``' list, in its order) named as 85 stages them: RecOpen's levels ``L0``, ``L1``, ..
    in order, the algebra's parts ``alg-p<j>``, a later level's cap ``cap-L<l>``, level 0's cap ``cap``, and any other by
    its unit's name. The recursive session's parts, in the order its proofs leave."""
    out, level = [], 0
    for u in units:
        kind, _, rest = u.partition("/")
        if kind == "rec-open":
            out.append(f"L{level}")
            level += 1
        elif kind == "rec-residuals" and (m := re.search(r"-p(\d+)$", rest)):
            out.append(f"alg-p{m[1]}")
        elif kind == "rec-cap":
            out.append(f"cap-L{m[1]}" if (m := re.search(r"-l(\d+)$", rest)) else "cap")
        else:
            out.append(u)
    if len(set(out)) != len(out):
        raise ValueError(f"V*'s statements {units} name the same part twice: {out}")
    return out


def window_class(part: str, spec: dict[str, int]) -> str:
    """The class of ``spec`` (`release_windows_ms`) a part's window is: L0 its own if named, then L, alg, cap, else other."""
    if part == "L0" and "L0" in spec:
        return "L0"
    for c, pat in (("L", r"L\d+"), ("alg", r"alg-p\d+"), ("cap", r"cap(-L\d+)?")):
        if re.fullmatch(pat, part):
            return c
    return "other"


def release_windows_ms(parts: list[str], spec: str) -> list[int]:
    """The release schedule's windows (`verity_flock.firewall`, rule 3), one a part, from ``spec``: comma-separated
    ``CLASS=SECONDS`` (`window_class`), and ``lead=SECONDS``, added to the first part's: what comes between the inner
    session's connection and the first outer session (the inner proof, icheck, vstage, vcap, oprove's setup)."""
    s = {k: int(v) for k, v in (kv.split("=", 1) for kv in spec.split(",") if kv)}
    missing = sorted({window_class(p, s) for p in parts} - set(s))
    if missing:
        raise ValueError(f"RELEASE_WINDOWS_S={spec}: no window for {', '.join(missing)} ({parts})")
    ws = [1000 * s[window_class(p, s)] for p in parts]
    ws[0] += 1000 * s.get("lead", 0)
    if not all(w > 0 for w in ws):
        raise ValueError(f"RELEASE_WINDOWS_S={spec}: every window is positive, not {ws}")
    return ws


def release_report(state: Path, session: str, times: list[dict]) -> dict:
    """The recursive session ``session`` on the ZK server's record in ``state`` against its release schedule: each part's
    window, its release time, when the Lean session loop decided it (``times``, its ``FIREWALL-TIMES`` lines), the
    slack between them, and its outcome on the record. ``ok`` when the inner session finished and every part was released,
    in order, at its scheduled time, with no stop and no refusal."""
    ev = [json.loads(ln) for ln in (state / "record.jsonl").read_text().splitlines() if ln.strip()]
    mine = [e for e in ev if e.get("session") == session]
    (open_,) = [e for e in mine if e["event"] == "open"]
    r, parts = open_.get("release"), open_["parts"]
    at = {t["part"]: t for t in times if t.get("session") == session}
    rows = []
    for j, p in enumerate(parts):
        sched = r["start_ms"] + sum(r["windows_ms"][:j + 1]) if r else None
        outs = [e["event"] for e in mine if e.get("part") == p and e["event"] in ("release", "stop", "refused")]
        t = at.get(p, {})
        rows.append({"part": p, "window_ms": r["windows_ms"][j] if r else None, "release_ms": sched,
                     "ready_ms": t.get("ready_ms"), "decided": t.get("outcome"), "record": outs,
                     "slack_ms": None if sched is None or t.get("ready_ms") is None else sched - t["ready_ms"],
                     "loop_release_ms": t.get("release_ms")})
    order = [e["part"] for e in mine if e["event"] == "release"]
    bad = [e for e in mine if e["event"] in ("stop", "refused")]
    finished = any(e["event"] == "finish" for e in mine)
    ok = (finished and r is not None and order == parts and not bad
          and all(x["record"] == ["release"] and x["decided"] == "go" and x["loop_release_ms"] == x["release_ms"] for x in rows))
    return {"session": session, "stream": open_["stream"], "start_ms": r and r["start_ms"], "parts": rows,
            "released": order, "stops": bad, "inner_finished": finished, "ok": ok,
            "schedule_ms": r and sum(r["windows_ms"])}


def tree_held(challenger: Path, sessions: list[Path]) -> dict:
    """Each rec_live session directory of ``sessions`` held to the release log in ``challenger`` (`rec_live.tree_held_reason`):
    its live-coin part's record (``public.json``) and its coin tree (a stopped session's ``coin_tree.json``, else its session
    record's ``coin_tree``)."""
    from verity.protocols.spot.device import remote as R

    held = []
    for d in sessions:
        ct = d / "coin_tree.json"
        tree = json.loads(ct.read_text()) if ct.exists() else json.loads((d / "session" / "session.json").read_text())["coin_tree"]
        held.append((d.name, tree, json.loads((d / "public.json").read_text())))
    why = RL.tree_held_reason(R.log_lines(challenger / R.RELEASES), held)
    return {"sessions": [n for n, _, _ in held], "reason": why, "ok": why is None,
            "held_by": "rec_live.tree_held_reason, the Python mirror of Spot.Challenger.innerTreesReason, which "
                       "the recursive audit reads"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    st = sp.add_parser("stage")
    st.add_argument("--n", type=int, required=True)
    st.add_argument("--sets", required=True)
    st.add_argument("--out", required=True)
    st.add_argument("--plain-leaves", action="store_true")
    st.add_argument("--sparse", metavar="NU,ELL,S_LOG,T_LOG", help="--stacked: the bucket's RecSparse shape, R's")
    st.add_argument("--stacked", metavar="FLOCK_CIRCUIT", help="register the stacked table R with the statement, by this build")
    st.add_argument("--forge", choices=("cell",), help="--stacked: R registered with cell 0 flipped")
    st.add_argument("--subcircuit", help="a census-style subcircuit id (default RoPE d64)")
    st.add_argument("--seed", type=int, default=SEED)
    sw = sp.add_parser("stage-window")
    sw.add_argument("--window", required=True, help="prove_path.py commit's window dir (rank<r>/), the prover's files included")
    sw.add_argument("--member", type=int, default=0)
    sw.add_argument("--subcircuit", required=True, help="the member template's census-style subcircuit id (its lowering)")
    sw.add_argument("--sets", required=True)
    sw.add_argument("--out", required=True)
    sw.add_argument("--zk-out", help="also the member under the hm96 circuit (today's --zk statement)")
    fg = sp.add_parser("forge")
    fg.add_argument("--what", choices=("challenge", "query", "nodraw", "partial", "top", "unsalted"), required=True)
    fg.add_argument("--inner", required=True)
    fg.add_argument("--level", type=int, default=0)
    fg.add_argument("--record", choices=("session", "prover"), default="session")
    rp = sp.add_parser("release-parts")
    rp.add_argument("--statements", required=True, help="flock-verify rec-statements' output (its RECVSTAR line)")
    rp.add_argument("--windows-s", required=True, help="CLASS=SECONDS,... (release_windows_ms)")
    rr = sp.add_parser("release-report")
    rr.add_argument("--state", required=True)
    rr.add_argument("--session", required=True)
    rr.add_argument("--times", nargs="*", default=[], help="files holding the Lean session loop's FIREWALL-TIMES lines")
    th = sp.add_parser("tree-held")
    th.add_argument("--challenger", required=True, help="the challenger's directory (its releases.jsonl)")
    th.add_argument("sessions", nargs="+", help="rec_live's session directories (public.json beside session/)")
    a = ap.parse_args(argv)
    if a.cmd == "tree-held":
        res = tree_held(Path(a.challenger), [Path(s) for s in a.sessions])
        json.dump(res, sys.stdout, sort_keys=True, indent=1)
        print()
        return 0 if res["ok"] else 1
    if a.cmd == "release-parts":
        (line,) = [json.loads(ln[9:]) for ln in Path(a.statements).read_text().splitlines() if ln.startswith("RECVSTAR ")]
        units = [s[0] for s in line["statements"]]
        parts = release_parts(units)
        json.dump({"units": units, "parts": parts, "windows_ms": release_windows_ms(parts, a.windows_s),
                   "windows_s": a.windows_s}, sys.stdout, sort_keys=True)
        print()
        return 0
    if a.cmd == "release-report":
        times = [json.loads(ln.split(" ", 1)[1]) for f in a.times for ln in Path(f).read_text().splitlines()
                 if ln.startswith("FIREWALL-TIMES ")]
        res = release_report(Path(a.state), a.session, times)
        json.dump(res, sys.stdout, sort_keys=True, indent=1)
        print()
        return 0 if res["ok"] else 1
    if a.cmd == "stage":
        if bool(a.stacked) != bool(a.sparse):
            ap.error("--stacked and --sparse come together: R is the bucket's table")
        sparse = None
        if a.sparse:
            from verity_flock.rec_sparse import Shape
            sparse = Shape(*(int(v) for v in a.sparse.split(",")))
        res = stage(a.n, Path(a.sets), Path(a.out), a.plain_leaves, sparse, a.stacked and Path(a.stacked), a.forge,
                    a.subcircuit, a.seed)
    elif a.cmd == "stage-window":
        res = stage_window(Path(a.window), a.member, a.subcircuit, Path(a.sets), Path(a.out), a.zk_out and Path(a.zk_out))
    else:
        res = forge(a.what, Path(a.inner), a.level, a.record)
    json.dump(res, sys.stdout, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
