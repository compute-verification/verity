"""The first recursive session's inner side (hiding model architectures, lane rec-v0; ``80-rec-inner.sh``): the inner
statement staged as M0's ``verity/flock-circuit`` over RoPE d64, proved by the unmodified prover against
``verity_flock.rec_live``'s proxy and coin server.

    python rec_inner.py stage --n N --sets DIR --out DIR      circuit.txt, inst-N.bin, pub-N.bin; prints the statement's shape
    python rec_inner.py universal --cls G,N_IN,N_OUT --n N --out DIR [--seed S] [--netlist FILE] [--fast]
                                                             a hidden netlist as the inner statement (`universal`): DIR/stage,
                                                             and DIR/other (another netlist of the class, the same x and salts);
                                                             --netlist (or REC_NETLIST=FILE): that private circuit's netlist
                                                             (`universal.Netlist` as JSON: n_in, gates, outputs), padded to the
                                                             class, not a random one;
                                                             --fast (or REC_FAST_STAGE=1): the same files from arrays
                                                             (`verity_flock.universal_stage`), for classes today's path can't stage
    python rec_inner.py wired --cls G,N_IN,N_OUT --n N --out DIR [--seed S]
                                                             four linked units (`wired`): DIR/stage, DIR/made-up-input
    python rec_inner.py forge --what challenge|query --inner DIR
                                                             flips one coin of rep 0 in DIR's session.json and public.json
    python rec_inner.py forge --what top|unsalted --inner DIR
                                                             top: flips one byte of rep 0's level-1 top in private.json;
                                                             unsalted: public.json's tops as their clear nodes (no salts)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import secrets
import shutil
import sys
from pathlib import Path

from verity_flock import circuit as FC
from verity_flock import rec_vstar as V

SEED = 20261004
STREAM = "circuit/rep0"
UNIVERSAL = Path(__file__).resolve().parents[4] / "benchmarks" / "private_circuit" / "universal.py"


def stage(n: int, sets: Path, out: Path) -> dict:
    from verity_numerical.bench import templates as TM
    from verity_numerical.bench.generate import generate
    from verity_numerical.bench.input_sets import InputSet

    s = InputSet.open(generate(TM.rope_head(64), n, SEED, sets)["path"])
    low = FC.lowering_for_set(s)
    FC.stage(low, s, 0, n, out)
    text = (out / "circuit.txt").read_text()
    meta = json.loads(text.split("\n", 2)[1][5:])
    G = meta["vus_per_block"]
    blocks = -(-n // G)
    return {"subcircuit": s.subcircuit.id, "unit": low.name, "n": n, "vus_per_block": G, "k_log": meta["k_log"],
            "units_per_vu": meta["units_per_vu"], "unit_log": meta["unit_log"], "unit_rows": low.layout.useful,
            "unit_nnz": low.layout.counts["nnz_a"] + low.layout.counts["nnz_b"], "blocks": blocks,
            "m": meta["k_log"] + max(3, (blocks - 1).bit_length()), "circuit_sha512": FC.digest(text), "class_sha512": FC.class_digest(text)}


def _benchmark():
    """``benchmarks/private_circuit/universal.py`` (protocol 2's universal unit end to end), whose netlists, lanes and wired
    circuit the stagings below reuse, loaded with the recursive session's plain SHA-512 leaves (``FLOCK_PLAIN_LEAVES=1``,
    which `class_statement` reads when it loads)."""
    os.environ["FLOCK_PLAIN_LEAVES"] = "1"
    from verity_flock import class_statement as CS

    if not CS.PLAIN_LEAVES:
        raise SystemExit("class_statement was loaded without FLOCK_PLAIN_LEAVES=1: the inner statement's leaves are plain")
    spec = importlib.util.spec_from_file_location("private_circuit_universal", UNIVERSAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _shape(d: Path, n: int) -> dict:
    with open(d / "circuit.txt") as f:
        f.readline()
        meta = json.loads(f.readline()[5:])
    blocks = -(-n // meta["vus_per_block"])
    return {"k_log": meta["k_log"], "vus_per_block": meta["vus_per_block"], "blocks": blocks,
            "m": meta["k_log"] + max(3, (blocks - 1).bit_length()), "leaf_scheme": (meta.get("leaf_scheme") or {}).get("id")}


def _differing(a: Path, b: Path) -> dict:
    """Where two public files differ: their header's fields by path, and the bytes of the rest."""
    pa, pb = a.read_bytes(), b.read_bytes()
    (ha, ba), (hb, bb) = pa.split(b"\n", 1), pb.split(b"\n", 1)

    def flat(v, at=""):
        return {k: x for key, sub in v.items() for k, x in flat(sub, f"{at}{key}.").items()} if isinstance(v, dict) else {at[:-1]: v}

    fa, fb = flat(json.loads(ha)), flat(json.loads(hb))
    return {"pub_bytes": len(pa), "pub_bytes_differing": sum(x != y for x, y in zip(pa, pb)) + abs(len(pa) - len(pb)),
            "header_differing": sorted(k for k in fa.keys() | fb.keys() if fa.get(k) != fb.get(k)),
            "body_bytes": len(ba), "body_bytes_differing": sum(x != y for x, y in zip(ba, bb)) + abs(len(ba) - len(bb))}


def universal(cls: str, n: int, out: Path, seed: int = 0, netlist: Path | None = None, fast: bool | None = None,
              salt_key: bytes | None = None) -> dict:
    """A hidden netlist as the inner statement: ``UniversalUnit_v1{G, N_IN, N_OUT}``'s class statement (`class_statement.stage`,
    words packed, plain leaves), one Call partitioned by ``Q_template_instance``, of a random netlist of 3/4 of the class's
    gates over 3/4 of its inputs padded to the class (`universal.pad`), on ``n`` random x. The netlist's ``op``, ``src`` and
    ``out`` are one row every instance opens (committed once: the registered program), ``x`` a row per instance, the outputs
    hidden rows. Into ``out/stage`` with its partition object. ``out/other``: another netlist of the class under the same salt
    key and x, the registered one with its last no-op gate made ``(XNOR, 0, 0)`` (a constant nothing reads, so every output
    is the same and only the program's rows differ; with no no-op gate, another random netlist): the statement of a prover
    whose program is not the registered one. ``netlist`` (default $REC_NETLIST): a JSON `universal.Netlist` (``n_in``, ``gates``,
    ``outputs``) in place of the random one; ``fast`` (default REC_FAST_STAGE=1): the same files from
    `verity_flock.universal_stage`."""
    fast = os.environ.get("REC_FAST_STAGE") == "1" if fast is None else fast
    netlist = netlist or os.environ.get("REC_NETLIST") or None
    B = _benchmark()
    from verity_flock import class_statement as CS
    from verity_flock import partition_units as PU

    from verity.ml.circuits import partition_object as PO
    from verity.ml.circuits.defs import bind
    from verity.ml.private_circuits import universal as U

    G, N_IN, N_OUT = (int(v) for v in cls.split(","))
    rng = random.Random(f"rec-private/{seed}/{G}/{N_IN}/{N_OUT}")
    fn = bind(U.UniversalUnit, G=G, N_IN=N_IN, N_OUT=N_OUT)
    small = B._netlist(rng, G, N_IN, N_OUT)
    if netlist is not None:
        nl = json.loads(Path(netlist).read_text())
        small = U.Netlist(nl["n_in"], tuple(tuple(g) for g in nl["gates"]), tuple(nl["outputs"]))
    net = U.pad(small, G, N_IN)
    if len(small.gates) < G:
        other = U.Netlist(N_IN, net.gates[:-1] + ((U.XNOR, 0, 0),), net.outputs)
    else:
        other = U.pad(B._netlist(random.Random(f"rec-private/other/{seed}/{G}/{N_IN}/{N_OUT}"), G, N_IN, N_OUT), G, N_IN)
    xs = [[rng.randrange(2) for _ in range(N_IN)] for _ in range(n)]
    if any(U.interpret(net, x) != U.interpret(small, x[:small.n_in]) for x in xs):
        raise AssertionError("the padded netlist's outputs are not the netlist's")
    salt_key = salt_key or secrets.token_bytes(32)
    res = {"class": {"G": G, "N_IN": N_IN, "N_OUT": N_OUT, "W": N_IN + G, "K": U.index_bits(G, N_IN)}, "n": n, "seed": seed,
           "netlist": {"gates": len(small.gates), "inputs": small.n_in, "outputs": len(small.outputs),
                       "ands": sum(1 for g in small.gates if g[0] & 1), "padded_gates": G - len(small.gates)},
           "unit": {"ands_closed_form": U.and_count(G, N_IN, N_OUT)}}
    keys = ("n", "k_log", "vus_per_block", "w_in", "w_out", "ports", "shared_rows", "circuit_sha512", "class_sha512",
            "circuit_bytes", "ands", "unit_rows", "unit_nnz")
    nets = (("stage", net), ("other", other))
    for name, _nl in nets:
        shutil.rmtree(out / name, ignore_errors=True)
    if fast:
        from verity_flock import universal_stage as US

        gates = US.trace(G, N_IN, N_OUT)
        built, program_sha512 = US.program(gates)[2:]
        st = US.stage(gates, US.unit(gates), PO.digest(built), program_sha512, nets, xs, out, salt_key)
        for name, nl in nets:
            if st[name]["outputs"] != [U.interpret(nl, x) for x in xs]:
                raise AssertionError(f"{name}: the laid-out unit rows disagree with universal.interpret")
            (out / name / "partition.json").write_text(json.dumps(built, sort_keys=True))
            res[name] = {k: st[name][k] for k in keys} | {"outputs": st[name]["outputs"]}
    else:
        program = PU.one_call(fn)
        query = PO.template_instance_query(fn)
        units = PU.PartitionUnits(program, query=query)
        [(shape, cl)] = units.classes().items()
        for name, nl in nets:
            want = [U.interpret(nl, x) for x in xs]
            ins, outs, idents = B.lanes(units, cl, nl, xs)
            if [[outs[j][t] for j in range(N_OUT)] for t in range(n)] != want:
                raise AssertionError(f"{name}: the unit's transcript disagrees with universal.interpret")
            d = out / name
            st = CS.stage(units, B.SPEC, shape, cl, (ins, outs, idents), d, salt_key, packed=True)
            if st["n"] != n:
                raise AssertionError(f"staging capped the instances at {st['n']}, not {n}")
            (d / "partition.json").write_text(json.dumps(PO.build(program, query), sort_keys=True))
            res[name] = {k: st[k] for k in keys} | {"outputs": want}
    res["other"] |= {"same_circuit": res["other"]["circuit_sha512"] == res["stage"]["circuit_sha512"],
                     "same_outputs": res["other"]["outputs"] == res["stage"]["outputs"],
                     "netlist_gates_differing": sum(a != b for a, b in zip(net.gates, other.gates))}
    res["other"] |= _differing(out / "stage" / f"pub-{n}.bin", out / "other" / f"pub-{n}.bin")
    for name in ("stage", "other"):
        del res[name]["outputs"]
    return res | _shape(out / "stage", n)


def wired(cls: str, n: int, out: Path, seed: int = 0) -> dict:
    """``--wired``'s circuit (`universal.wired_topology`: four layers of the unit, a random hidden netlist each, its topology
    the Program's, public) on ``n`` instances as one class statement of ``4 n`` instances with link rows
    (`class_statement.Wiring`), plain leaves: the honest staging in ``out/stage`` and the cheating prover's, whose layer 1
    reads a made-up input in place of layer 0's outputs, every unit correct on what it is given, in ``out/made-up-input``,
    each with the verifier's program descriptor and partition object (`universal.run_wired`, ``--stage-only``)."""
    B = _benchmark()
    G, N_IN, N_OUT = (int(v) for v in cls.split(","))
    row = B.run_wired(argparse.Namespace(n=n, seed=seed, stage_only=True, keep_stage=True, lean=None), G, N_IN, N_OUT, out)
    d = out / f"wired-G{G}-N{N_IN}-O{N_OUT}"
    for name in ("stage", "made-up-input"):
        shutil.rmtree(out / name, ignore_errors=True)
        shutil.move(d / name, out / name)
    for f in ("partition.json", "program.desc"):
        shutil.copyfile(out / "stage" / f, out / "made-up-input" / f)
    shutil.rmtree(d)
    return row | _shape(out / "stage", 4 * n) | {
        "made_up_public": _differing(out / "stage" / f"pub-{4 * n}.bin", out / "made-up-input" / f"pub-{4 * n}.bin")}


def flipped(inner: Path, fold: Path, points: Path, salts: Path, out: Path) -> dict:
    """V*'s statement of the level whose top ``forge --what top`` flipped (level 1: rep 0's first salted top after
    ``root_B``), staged over that forged session by a prover that skips its own checks of its tops against the ``b || c`` the
    coin server received (`rec_vstar.opened_top`, `rec_outer.TopValue.build`), which refuse it by name otherwise: its
    registration of the level's tops holds the flipped node, and the verifier's entry is its own, from the recorded
    ``b || c``. Under ``out/forged-flipped-top/L1``, beside the verifier's files."""
    from verity_flock import rec_live as RL
    from verity_flock import rec_open as R
    from verity_flock import rec_outer as O
    from verity_flock import rec_residuals as RR
    from verity_flock import rec_vstage as S

    from verity.core.protocols.sampled_proofs.one_stage import registered as RG

    def unchecked_top(opening: dict, bcs: list[str], what: str) -> V.Top:
        return V.Top(tuple(bytes.fromhex(x) for x in opening["nodes"]), tuple(bytes.fromhex(x) for x in opening["salts"]),
                     tuple(bytes.fromhex(b) for b in bcs))

    def unchecked_build(cls, program: str, level: int, tops: dict) -> O.TopValue:
        runs, first = [], {}
        for rep, t in sorted(tops.items()):
            if t not in runs:
                runs.append(t)
            first[rep] = sum(len(u.nodes) for u in runs[:runs.index(t)])
        value = O.top_value_name(level)
        reg = RG.Registered.commit(program, value, [r for t in runs for r in RL.top_rows(b"".join(t.nodes))],
                                   [s for t in runs for s in t.salts])
        return cls(reg, RR.commits_entry(program, value, R.OUT_WORDS, [b for t in runs for b in t.bcs]), first)

    V.opened_top, O.TopValue.build = unchecked_top, classmethod(unchecked_build)
    res = S.stage(Path(inner), json.loads(Path(fold).read_text()), Path(out) / "forged-flipped-top", [1], False,
                  pts=json.loads(Path(points).read_text()), salts=Path(salts))
    pub, priv, proofs = V.load(Path(inner))
    p = next(p for p in priv["rounds"] if p["stream"] == STREAM and p["k"] > 0 and p.get("tops"))
    tv = unchecked_build(O.TopValue, O.session_program((Path(inner) / "public.json").read_bytes()), 1,
                         {0: unchecked_top(p["tops"][0], next(r for r in pub["rounds"] if r["stream"] == STREAM
                                                               and r["k"] == p["k"])["tops"][0], "")})
    reads = [q for q in V.queries(pub, priv, proofs) if q.rep == 0 and q.level == 1 and tv.position(q) == 0]
    return res | {"flipped": {"stream": STREAM, "k": p["k"], "level": 1, "node": 0, "openings_reading_it": len(reads),
                              "registered_root_differs": tv.own["root"] != tv.reg.entry()["root"]}}


def forge(what: str, inner: Path, level: int = 0) -> dict:
    """Flip the low bit of one coin of rep 0, in the session record and in the coin server's public record alike: the first
    coin after the binding round (``challenge``) or the first query coin of Ligerito level ``level`` (``query``). ``top`` and
    ``unsalted`` forge the tree tops instead (the module docstring)."""
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
    sess = json.loads((inner / "session" / "session.json").read_text())
    for rd in (next(r for r in next(s for s in sess["streams"] if s["stream"] == STREAM)["rounds"] if r["k"] == k),
               next(r for r in pub["rounds"] if r["stream"] == STREAM and r["k"] == k)):
        c = rd["coins"][0]
        rd["coins"][0] = c[:-1] + format(int(c[-1], 16) ^ 1, "x")
    (inner / "session" / "session.json").write_text(json.dumps(sess))
    (inner / "public.json").write_text(json.dumps(pub))
    return {"forged": what, "stream": STREAM, "k": k, "coin": 0, **({"level": level} if what == "query" else {})}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    st = sp.add_parser("stage")
    st.add_argument("--n", type=int, required=True)
    st.add_argument("--sets", required=True)
    st.add_argument("--out", required=True)
    for cmd in ("universal", "wired"):
        u = sp.add_parser(cmd)
        u.add_argument("--cls", required=True, help="G,N_IN,N_OUT")
        u.add_argument("--n", type=int, required=True, help="instances (wired: per layer, 4 n in all)")
        u.add_argument("--out", required=True)
        u.add_argument("--seed", type=int, default=0)
        if cmd == "universal":
            u.add_argument("--netlist")
            u.add_argument("--fast", action="store_true", default=None)
    fg = sp.add_parser("forge")
    fg.add_argument("--what", choices=("challenge", "query", "top", "unsalted"), required=True)
    fg.add_argument("--inner", required=True)
    fg.add_argument("--level", type=int, default=0)
    fl = sp.add_parser("flipped")
    for k in ("--inner", "--fold", "--points", "--salts", "--out"):
        fl.add_argument(k, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "flipped":
        res = flipped(a.inner, a.fold, a.points, a.salts, a.out)
    elif a.cmd in ("universal", "wired"):
        res = universal(a.cls, a.n, Path(a.out), a.seed, a.netlist, a.fast) if a.cmd == "universal" else wired(a.cls, a.n, Path(a.out), a.seed)
    else:
        res = stage(a.n, Path(a.sets), Path(a.out)) if a.cmd == "stage" else forge(a.what, Path(a.inner), a.level)
    json.dump(res, sys.stdout, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
