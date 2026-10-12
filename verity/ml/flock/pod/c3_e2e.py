"""Scratch driver (cursor/c3-e2e-integration-8c79, never a PR): candidate 3's outer V* statements over one inner session,
staged by `rec_vstage.stage` and each checked by the live Lean verifier's `flock-verify statement`.

    python c3_e2e.py stage KIND --inner I --fold F --points P --circuit C --salts S --out DIR > stage.json
        KIND: honest (every level, the holo algebra, RecSparse, R's levels and opening)
              claim       rep 0's rec-claim row one bit off, registered consistently: the reads name the forged root as the
                          verifier's own entry, in RecSparse and R's opening (gap 4: refused by the claim round's c_k)
              claim-own   the same, read against the honest rec-claim root (the claim round's c_k) as the verifier's own entry
              sparse-v    RecSparse's v of rep 0, element 0 one bit off (check 5: rec-sparse-v)
              stacked-coef  R's level 2 opening 0's coef word 0 one bit off (check 5: rec-coef at stacked/rep0's coins)
              opening-v   R's opening's v element 0 one bit off (check 5: rec-v at stacked/rep0's coins and the claim points)
              h           rep 0's h row one bit off, read against the verifier's own entry from h's c_k
              session     a forged session (88's FORGE=cell|point): RecSparse and R alone, as 88 stages it
    python c3_e2e.py lean --bin B --dir DIR --inner-circuit C --inner-public P --out F.jsonl [--label L]
        `statement` on every staged statement under DIR, plain and, for V*'s listed statements, under --rec-inner
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from verity_flock import rec_algebra as A
from verity_flock import rec_bindings as RB
from verity_flock import rec_sparse as SP
from verity_flock import rec_vstage as S
from verity_flock import rec_vstar as V
from verity_flock.rec_outer import CLAIM_VALUE

SPARSE = SP.Shape(18, 13, 6, 0)


def _flip0(reg):
    from verity.core.protocols.sampled_proofs.one_stage import registered as RG

    rows = [list(r) for r in reg.rows]
    rows[0][0] ^= 1
    return RG.Registered.commit(reg.program, reg.value, rows, list(reg.salts))


def stage(kind: str, a) -> dict:
    inner = Path(a.inner)
    fold = json.loads(Path(a.fold).read_text())
    pts = json.loads(Path(a.points).read_text())
    rng = next(r for r in S.meta(Path(a.circuit))["ranges"] if r["type"] == "unit" and r["net"] in fold.get("hidden", []))
    levels, algebra, info = None, True, {"kind": kind}
    if kind == "h":
        public, _, _ = V.load(inner, S.TABLE)
        hr = f"m{A.h_round(S.verifier(public, fold, pts).sh)}"
        orig = S.messages

        def messages(program, mrs, ports=None):
            regs, own = orig(program, mrs, ports)
            if ports is None:
                regs[hr] = _flip0(regs[hr])
                info["forged"] = {"port": hr, "value": regs[hr].value, "rep": 0, "word": 0, "bit": 0}
            return regs, own

        S.messages = messages
        S.stacked_reads = lambda *x, **k: None
        levels = []
    elif kind in ("claim", "claim-own"):
        orig = S.sparse_messages

        def sparse_messages(*x, **k):
            regs, own = orig(*x, **k)
            honest = regs["claim"]
            assert honest.value == CLAIM_VALUE and own["claim"]["root"] == honest.entry()["root"]
            regs = {**regs, "claim": _flip0(honest)}
            info["forged"] = {"value": CLAIM_VALUE, "rep": 0, "word": 0, "bit": 0, "honest_root": honest.root.hex(),
                              "forged_root": regs["claim"].root.hex()}
            if kind == "claim":
                own = {**own, "claim": regs["claim"].entry()}
            return regs, own

        S.sparse_messages = sparse_messages
        levels, algebra = [], False
    elif kind == "sparse-v":
        orig = SP.verifier_row
        seen = []

        def verifier_row(sh, c):
            v = list(orig(sh, c))
            if not seen:
                v[0] ^= 1
                info["forged"] = {"value": SP.V_VALUE, "rep": 0, "element": 0, "bit": 0}
            seen.append(1)
            return v

        S.SP.verifier_row = verifier_row
        S.stacked_reads = lambda *x, **k: None
        levels, algebra = [], False
    elif kind == "stacked-coef":
        orig = S.coefficients

        def coefficients(vf, qs):
            out = orig(vf, qs)
            ks = sorted(k for k in out if k[0] == S.STACKED_REP and k[1] == 2)
            if ks:
                k = ks[0]
                out[k] = [[out[k][0][0] ^ 1, *out[k][0][1:]], *out[k][1:]] if isinstance(out[k][0], list) else [out[k][0] ^ 1, *out[k][1:]]
                info["forged"] = {"value": "rec-coef", "rep": k[0], "level": k[1], "j": k[2], "word": 0, "bit": 0}
            return out

        S.coefficients = coefficients
        levels, algebra = [], False
    elif kind == "opening-v":
        orig = S.stacked_reads

        def stacked_reads(*x, **k):
            sr = orig(*x, **k)
            sr.v = [sr.v[0] ^ 1, *sr.v[1:]]
            info["forged"] = {"value": "rec-v", "element": 0, "bit": 0}
            return sr

        S.stacked_reads = stacked_reads
        levels, algebra = [], False
    elif kind == "session":
        levels, algebra = [], False
    elif kind == "level3":
        res = S.stage(inner, fold, Path(a.out), [3], False, None, True, pts, Path(a.salts), None, None)
        return info | res
    elif kind != "honest":
        raise SystemExit(f"unknown kind {kind}")
    res = S.stage(inner, fold, Path(a.out), levels, algebra, None, True, pts, Path(a.salts), None, (SPARSE, rng))
    return info | res


def _unit(d: Path) -> str:
    with (d / "circuit.txt").open() as f:
        return f.readline().split()[1]


def lean(a) -> int:
    b, root, out = a.bin, Path(a.dir), Path(a.out)
    inner = {}
    for tag, extra in (("hide", ["--hide"]), ("plain", [])):
        p = subprocess.run([b, "rec-inner", "--circuit", a.inner_circuit, "--public", a.inner_public, *extra],
                           capture_output=True, text=True)
        f = out.parent / f"rec-inner-{tag}.json"
        f.write_text(p.stdout)
        q = subprocess.run([b, "rec-statements", "--rec-inner", str(f)], capture_output=True, text=True)
        line = next((json.loads(ln[9:]) for ln in q.stdout.splitlines() if ln.startswith("RECVSTAR ")), None)
        names = [s[0] for s in line["statements"]] if line and line.get("ok") else None
        inner[tag] = (f, names)
        with out.open("a") as o:
            o.write(json.dumps({"label": a.label, "check": f"rec-inner --{tag}", "rc": p.returncode, "line": p.stdout.strip()[:400],
                                "err": p.stderr.strip()[-400:], "rec_statements": q.stdout.strip()[-1500:] or q.stderr.strip()[-400:]}) + "\n")
    dirs = sorted({c.parent for c in root.rglob("circuit.txt")})
    bad = 0
    for d in dirs:
        unit = _unit(d)
        base = [b, "statement", "--statement", "verity/flock-circuit", "--circuit", str(d / "circuit.txt"), "--public",
                str(d / "pub.bin"), "--partition", str(d / "partition.json"), "--program", str(d / "program.json"),
                "--registered", str(d / "registered.json")]
        runs = [("plain", base)]
        ports = json.loads((d / "registered.json").read_text())["ports"]
        rows = d / "rec-rows.json"
        if "acc" in ports:
            rows.write_text(json.dumps({"format": "verity/registered-reads/v0", "ports": {"acc0": ports["acc"]}}))
        else:
            shared = {}
            for q in sorted(d.parent.glob("p*")):
                for nm, e in json.loads((q / "registered.json").read_text())["ports"].items():
                    if nm not in ("v", "out"):
                        shared.setdefault(nm, e)
            rows.write_text(json.dumps({"format": "verity/registered-reads/v0", "ports": shared}))
        for tag, (f, names) in inner.items():
            if names and unit in names:
                bf = d / "rec-bindings.json"
                if not bf.exists():
                    bf.write_text(json.dumps(RB.bindings(json.loads((d / "partition.json").read_text()),
                                                         json.loads((d / "program.json").read_text()))))
                runs.append((f"rec-inner-{tag}", [*base, "--rec-inner", str(f), "--rec-part", str(names.index(unit)),
                                                  "--rec-bindings", str(bf), "--rec-rows", str(rows)]))
        for mode, argv in runs:
            p = subprocess.run(argv, capture_output=True, text=True)
            first = (p.stdout.strip().splitlines() or [""])[0]
            verdict = "accepted" if p.returncode == 0 else "refused"
            bad += p.returncode != 0 and a.label == "honest" and mode == "plain"
            rec = {"label": a.label, "statement": str(d.relative_to(root)), "unit": unit, "mode": mode, "rc": p.returncode,
                   "verdict": verdict, "line": first[:2000], "err": p.stderr.strip()[-600:]}
            print(json.dumps({k: rec[k] for k in ("label", "statement", "unit", "mode", "rc")} | {"line": first[:300]}), flush=True)
            with out.open("a") as o:
                o.write(json.dumps(rec) + "\n")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("kind")
    for k in ("--inner", "--fold", "--points", "--circuit", "--salts", "--out"):
        s.add_argument(k, required=True)
    l = sub.add_parser("lean")
    for k in ("--bin", "--dir", "--inner-circuit", "--inner-public", "--out"):
        l.add_argument(k, required=True)
    l.add_argument("--label", default="honest")
    a = ap.parse_args()
    if a.cmd == "stage":
        print(json.dumps(stage(a.kind, a), sort_keys=True, default=str))
        return 0
    return lean(a)


if __name__ == "__main__":
    sys.exit(main())
