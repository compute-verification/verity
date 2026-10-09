"""The first recursive session's inner side (hiding model architectures, lane rec-v0; ``80-rec-inner.sh``): the inner
statement staged as M0's ``verity/flock-circuit`` over RoPE d64, proved by the unmodified prover against
``verity_flock.rec_live``'s proxy and coin server.

    python rec_inner.py stage --n N --sets DIR --out DIR      circuit.txt, inst-N.bin, pub-N.bin; prints the statement's shape
    python rec_inner.py universal --cls G,N_IN,N_OUT --n N --out DIR [--seed S] [--netlist FILE] [--fast] [--outside]
                                                             a hidden netlist as the inner statement (`universal`): DIR/stage,
                                                             and DIR/other (another netlist of the class, the same x and salts);
                                                             --netlist (or REC_NETLIST=FILE): that private circuit's netlist
                                                             (`universal.Netlist` as JSON: n_in, gates, outputs), padded to the
                                                             class, not a random one;
                                                             --fast (or REC_FAST_STAGE=1): the same files from arrays
                                                             (`verity_flock.universal_stage`), for classes today's path can't stage;
                                                             --outside: DIR/stage only, a description that is not a circuit
    python rec_inner.py direct --cls G,N_IN,N_OUT --n N --out DIR [--seed S] [--netlist FILE]
                                                             the private circuit `universal` registers, on the same x, as a
                                                             public circuit (`direct`: the overhead's baseline)
    python rec_inner.py partition --cls G,N_IN,N_OUT --circuit FILE --out FILE
                                                             the partition object of the unit itself as the one Call (a
                                                             staging with no program row: stage_class.py's), as FILE binds it
    python rec_inner.py faulty --stage DIR --n N --out DIR2  DIR's prover file with instance 0's output row one bit off
    python rec_inner.py wired --cls G,N_IN,N_OUT --n N --out DIR [--seed S]
                                                             four linked units (`wired`): DIR/stage, DIR/made-up-input
    python rec_inner.py forge --what challenge|query --inner DIR
                                                             flips one coin of rep 0 in DIR's session.json and public.json
    python rec_inner.py forge --what top|unsalted --inner DIR
                                                             top: flips one byte of rep 0's level-1 top in private.json;
                                                             unsalted: public.json's tops as their clear nodes (no salts)
    python rec_inner.py differ PUB_A PUB_B                   where two public files differ (header fields, body bytes)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import shutil
import sys
from pathlib import Path

from verity_flock import circuit as FC
from verity_flock import rec_vstar as V

SEED = 20261004
STREAM = "circuit/rep0"
#: `circuit.stage_salts` as the module has it, before `universal` or `wired` replace it
STAGE_SALTS = FC.stage_salts
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


def _benchmark(plain: bool = True):
    """``benchmarks/private_circuit/universal.py`` (protocol 2's universal unit end to end), whose netlists, lanes and wired
    circuit the stagings below reuse, loaded with the recursive session's plain SHA-512 leaves (``FLOCK_PLAIN_LEAVES=1``,
    which `class_statement` reads when it loads), or with ``plain`` false C-Flock's salted ones."""
    if plain:
        os.environ["FLOCK_PLAIN_LEAVES"] = "1"
    else:
        os.environ.pop("FLOCK_PLAIN_LEAVES", None)
    from verity_flock import class_statement as CS

    if CS.PLAIN_LEAVES != plain:
        raise SystemExit(f"class_statement was loaded with FLOCK_PLAIN_LEAVES={int(CS.PLAIN_LEAVES)}, not {int(plain)}")
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
    return {"pub_bytes": len(pa), "same_size": len(pa) == len(pb),
            "pub_bytes_differing": sum(x != y for x, y in zip(pa, pb)) + abs(len(pa) - len(pb)),
            "header_differing": sorted(k for k in fa.keys() | fb.keys() if fa.get(k) != fb.get(k)),
            "body_bytes": len(ba), "body_bytes_differing": sum(x != y for x, y in zip(ba, bb)) + abs(len(ba) - len(bb))}


def program_unit(fn):
    """The universal unit ``fn`` over two parameters, ``prog`` (the netlist's ``op``, ``src`` and ``out`` bits in that order,
    which its one Call reads as slices: no gates of its own) and ``x``. A class statement has a row port, and a public region,
    per parameter of its unit (`class_statement.sources`), so the program is one row and the inner statement has three
    regions (``prog``, ``x``, ``out``): the recursion's claims (2 + 2 a region) are rec-v0's eight. With ``fn`` itself the
    unit, five regions' twelve claims put V*'s batched-sum residual alone past a statement's sha512x3 slots
    (`rec_residuals.MAX_SLOTS`)."""
    from verity.ml.circuits.defs import CompositeDefinition, bind
    from verity.ml.circuits.types import Array
    from verity.ml.private_circuits import universal as U

    params, ret = fn.signature()
    prog = sum(t.leaves for _, t in params[:-1])

    def body(B, S, p, x):
        args, at = [], 0
        for _, t in params[:-1]:
            args.append(p[at:at + t.leaves].reshape(t))
            at += t.leaves
        return B.call(fn, *args, x)

    return bind(CompositeDefinition(f"ProgramRow[{fn.id}]", 1, (), lambda S: ((("prog", Array(prog, U.BIT)), params[-1]), ret),
                                    body, register=False))


def _os_salts():
    """`circuit.stage_salts` with every salt fresh from the OS generator whatever key it is given, drawn once per range, ports
    and table sizes: a second staging of the same shape in this process gets the first's salts (``other`` the registered
    statement's), and none is derived from a key."""
    drawn = {}

    def salts(input_set, lo, hi, ports, key, tables=None):
        at = (lo, hi, tuple(p.name for p in ports), tuple(sorted(tables.items())) if tables else None)
        if at not in drawn:
            drawn[at] = STAGE_SALTS(input_set, lo, hi, ports, None, tables)
        return {k: list(v) for k, v in drawn[at].items()}

    return salts


#: a staging's record, as `universal` and `direct` report it
KEYS = ("n", "k_log", "vus_per_block", "w_in", "w_out", "ports", "shared_rows", "circuit_sha512", "class_sha512", "circuit_bytes",
        "ands", "unit_rows", "unit_nnz")


def _private(B, cls: str, n: int, seed: int, netlist: Path | None) -> tuple:
    """The private circuit `universal` registers and its ``n`` x: the class, the netlist (``netlist``'s, else a random one of
    3/4 of the class's gates over 3/4 of its inputs) and each instance's N_IN input bits."""
    from verity.ml.private_circuits import universal as U

    G, N_IN, N_OUT = (int(v) for v in cls.split(","))
    rng = random.Random(f"rec-private/{seed}/{G}/{N_IN}/{N_OUT}")
    small = B._netlist(rng, G, N_IN, N_OUT)
    if netlist is not None:
        nl = json.loads(Path(netlist).read_text())
        small = U.Netlist(nl["n_in"], tuple(tuple(g) for g in nl["gates"]), tuple(nl["outputs"]))
    return (G, N_IN, N_OUT), small, [[rng.randrange(2) for _ in range(N_IN)] for _ in range(n)]


def universal(cls: str, n: int, out: Path, seed: int = 0, netlist: Path | None = None, fast: bool | None = None,
              salt_key: bytes | None = None, outside: bool = False) -> dict:
    """A hidden netlist as the inner statement: ``UniversalUnit_v1{G, N_IN, N_OUT}`` as `program_unit`'s class statement
    (`class_statement.stage`, words packed, plain leaves), one Call partitioned by ``Q_template_instance``, of a random netlist
    of 3/4 of the class's gates over 3/4 of its inputs padded to the class (`universal.pad`), on ``n`` random x. The netlist's
    ``op``, ``src`` and ``out`` are one row every instance opens (committed once: the registered program), ``x`` a row per
    instance, the outputs hidden rows; every salt from the OS (`_os_salts`), or derived from ``salt_key`` when one is given (a
    test's). Into ``out/stage`` with its partition object. ``out/other``: another netlist of the class under the same salts
    and x, the registered one with its last no-op gate made ``(XNOR, 0, 0)`` (a constant nothing reads, so every output is the
    same and only the program's row differs; with no no-op gate, another random netlist): the statement of a prover whose
    program is not the registered one. ``netlist`` (default $REC_NETLIST): a JSON `universal.Netlist` (``n_in``, ``gates``,
    ``outputs``) in place of the random one; ``fast`` (default REC_FAST_STAGE=1): the same files from
    `verity_flock.universal_stage`.

    ``outside``: only ``out/stage``, the netlist with one gate's first input the last wire, ``W - 1``, which no gate before it
    computes (the first gate for which that changes an output on the x, else gate 0): not a circuit (its gates are not in
    order), which the unit runs as reading 0, so it is a member of the class as every description of the class's width is
    (`universal.Netlist`), registered under salts of its own."""
    fast = os.environ.get("REC_FAST_STAGE") == "1" if fast is None else fast
    netlist = netlist or os.environ.get("REC_NETLIST") or None
    B = _benchmark()
    from verity_flock import class_statement as CS
    from verity_flock import partition_units as PU

    from verity.ml.circuits import partition_object as PO
    from verity.ml.circuits.defs import bind
    from verity.ml.private_circuits import universal as U

    (G, N_IN, N_OUT), small, xs = _private(B, cls, n, seed, netlist)
    fn = bind(U.UniversalUnit, G=G, N_IN=N_IN, N_OUT=N_OUT)
    net = U.pad(small, G, N_IN)
    if len(small.gates) < G:
        other = U.Netlist(N_IN, net.gates[:-1] + ((U.XNOR, 0, 0),), net.outputs)
    else:
        other = U.pad(B._netlist(random.Random(f"rec-private/other/{seed}/{G}/{N_IN}/{N_OUT}"), G, N_IN, N_OUT), G, N_IN)
    if any(U.interpret(net, x) != U.interpret(small, x[:small.n_in]) for x in xs):
        raise AssertionError("the padded netlist's outputs are not the netlist's")
    FC.stage_salts = STAGE_SALTS if salt_key is not None else _os_salts()
    res = {"class": {"G": G, "N_IN": N_IN, "N_OUT": N_OUT, "W": N_IN + G, "K": U.index_bits(G, N_IN)}, "n": n, "seed": seed,
           "netlist": {"gates": len(small.gates), "inputs": small.n_in, "outputs": len(small.outputs),
                       "ands": sum(1 for g in small.gates if g[0] & 1), "padded_gates": G - len(small.gates)},
           "unit": {"ands_closed_form": U.and_count(G, N_IN, N_OUT)}, "salts": "key" if salt_key is not None else "os"}
    nets = (("stage", net), ("other", other))
    if outside:
        def reading_last(i):
            op, _, b = net.gates[i]
            return U.Netlist(N_IN, net.gates[:i] + ((op, N_IN + G - 1, b),) + net.gates[i + 1:], net.outputs)

        i = next((i for i in range(G - 1) if any(U.interpret(reading_last(i), x) != U.interpret(net, x) for x in xs)), 0)
        nets = (("stage", reading_last(i)),)
        res["outside"] = {"gate": i, "reads": N_IN + G - 1, "computed_by_gate": G - 1}
    for name, _nl in nets:
        shutil.rmtree(out / name, ignore_errors=True)
    if fast:
        from verity_flock import universal_stage as US

        gates = US.trace(G, N_IN, N_OUT)
        digests = US.program_digests(gates, program_unit)
        st = US.stage(gates, US.unit(gates, program_row=True), digests, nets, xs, out, salt_key)
        built = digests()[0]
        for name, nl in nets:
            if st[name]["outputs"] != [U.interpret(nl, x) for x in xs]:
                raise AssertionError(f"{name}: the laid-out unit rows disagree with universal.interpret")
            (out / name / "partition.json").write_text(json.dumps(built, sort_keys=True))
            res[name] = {k: st[name][k] for k in KEYS} | {"outputs": st[name]["outputs"]}
    else:
        unit = program_unit(fn)
        program = PU.one_call(unit)
        query = PO.template_instance_query(unit)
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
            res[name] = {k: st[k] for k in KEYS} | {"outputs": want}
    if outside:
        res["outside"]["outputs_differ_from_the_netlist"] = sum(
            a != b for a, b in zip(res["stage"]["outputs"], (U.interpret(net, x) for x in xs)))
        del res["stage"]["outputs"]
        return res | _shape(out / "stage", n)
    res["other"] |= {"same_circuit": res["other"]["circuit_sha512"] == res["stage"]["circuit_sha512"],
                     "same_outputs": res["other"]["outputs"] == res["stage"]["outputs"],
                     "netlist_gates_differing": sum(a != b for a, b in zip(net.gates, other.gates))}
    res["other"] |= _differing(out / "stage" / f"pub-{n}.bin", out / "other" / f"pub-{n}.bin")
    for name in ("stage", "other"):
        del res[name]["outputs"]
    return res | _shape(out / "stage", n)


def direct(cls: str, n: int, out: Path, seed: int = 0, netlist: Path | None = None) -> dict:
    """The overhead's baseline: the private circuit `universal` registers (the same netlist, the same x) as a public circuit
    proved directly, with no recursion. `universal.direct`'s Boolean Definition of the netlist, unpadded, on each x's first
    ``n_in`` bits, ``n`` instances of it as one batch (`population_program`) partitioned by ``Q_template_instance``, staged by
    `class_statement.stage` with C-Flock's salted leaves (the ``--zk`` statement Lean covers), META naming the Definition its
    units are instances of (``names_definition``), every instance's rows its own (no shared rows: the form Lean covers for
    a statement without registered reads) and every salt from the OS, into ``out`` as
    rec-private-outer.sh's ``outer`` reads a statement: circuit.txt, inst.bin, pub.bin, partition.json and program.json (the
    program's descriptor)."""
    netlist = netlist or os.environ.get("REC_NETLIST") or None
    B = _benchmark(plain=False)
    from verity_flock import class_statement as CS
    from verity_flock import partition_units as PU

    from verity.core.protocols.sampled_proofs.one_stage.partition import (
        population_program,
    )
    from verity.ml.circuits import partition_object as PO
    from verity.ml.circuits.codec import canonical_json, encode_program
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate
    from verity.ml.private_circuits import universal as U

    (G, N_IN, N_OUT), small, xs = _private(B, cls, n, seed, netlist)
    xs = [x[:small.n_in] for x in xs]
    fn = bind(B.direct(small, f"RecPrivateDirectG{G}N{N_IN}O{N_OUT}S{seed}"))
    program = population_program(fn, n)
    query = PO.template_instance_query(fn)
    units = PU.PartitionUnits(program, query=query)
    [(shape, cl)] = units.classes().items()
    act = units.activation(cl.definition)
    uc = cl.representative
    ins, outs = [[] for _ in uc.inputs], [[] for _ in uc.outputs]
    for x in xs:
        run = PU.evaluate(act, x)
        for port, k in zip(ins, uc.inputs):
            port.append(run[k])
        for port, j in zip(outs, uc.outputs):
            port.append(run[j])
    want = [U.interpret(small, x) for x in xs]
    if any(list(evaluate(fn, x)) != w for x, w in zip(xs, want)):
        raise AssertionError("the Definition disagrees with universal.interpret")
    # the unit's outputs are the netlist's computed ones in the activation's order: an output that is an input bit is not one
    computed = {o: k for k, o in enumerate(small.outputs) if o >= small.n_in}
    if len(outs) != len(computed) or not {tuple(c) for c in outs} <= {tuple(w[k] for w in want) for k in computed.values()}:
        raise AssertionError("the unit's outputs are not the netlist's computed outputs")
    FC.stage_salts = _os_salts()
    shutil.rmtree(out, ignore_errors=True)
    st = CS.stage(units, B.SPEC, shape, cl, (ins, outs), out, None, packed=True, names_definition=True, share_rows=False)
    if st["n"] != n:
        raise AssertionError(f"staging capped the instances at {st['n']}, not {n}")
    (out / "partition.json").write_text(json.dumps(PO.build(program, query), sort_keys=True))
    (out / "program.json").write_bytes(canonical_json(encode_program(program)))
    for f in ("inst", "pub"):
        (out / f"{f}-{n}.bin").rename(out / f"{f}.bin")
    return {"class": {"G": G, "N_IN": N_IN, "N_OUT": N_OUT}, "n": n, "seed": seed, "salts": "os", "definition": fn.id,
            "netlist": {"gates": len(small.gates), "inputs": small.n_in, "outputs": len(small.outputs),
                        "ands": sum(1 for g in small.gates if g[0] & 1)},
            "unit": {"inputs_read": len(uc.inputs), "outputs_computed": len(uc.outputs)}} | {k: st[k] for k in KEYS} | _shape(out, n)


def partition(cls: str, circuit: Path, out: Path) -> dict:
    """The partition object of ``UniversalUnit_v1{cls}`` itself as the program's one Call, ``Q_template_instance``: a staging
    of the unit with no program row (``benchmarks/private_circuit/stage_class.py``'s, which writes none), into ``out``, once
    its digest is the one ``circuit`` binds (META ``partition``)."""
    from verity_flock import partition_units as PU

    from verity.ml.circuits import partition_object as PO
    from verity.ml.circuits.defs import bind
    from verity.ml.private_circuits import universal as U

    G, N_IN, N_OUT = (int(v) for v in cls.split(","))
    fn = bind(U.UniversalUnit, G=G, N_IN=N_IN, N_OUT=N_OUT)
    built = PO.build(PU.one_call(fn), PO.template_instance_query(fn))
    with open(circuit) as f:
        f.readline()
        bound = json.loads(f.readline()[5:])["partition"]["digest"]
    if PO.digest(built) != bound:
        raise SystemExit(f"{circuit} binds partition {bound[:16]}…, not {fn.id}'s one Call ({PO.digest(built)[:16]}…)")
    out.write_text(json.dumps(built, sort_keys=True))
    return {"unit": fn.id, "partition": str(out), "digest": bound}


def faulty(stage_dir: Path, n: int, out: Path, instance: int = 0) -> dict:
    """A faulty unit's prover file: ``stage_dir/inst-n.bin`` with instance ``instance``'s output row one bit off (bit 0 of its
    first word), every other row, salt and ``b || c`` as staged, in ``out/inst-n.bin``: proved against the registered
    statement (``stage_dir``'s public file), its session is of a table one unit of which the witness does not satisfy
    (`rec_shape.forge_filler`'s forgery, at an instance that is not a filler)."""
    from verity_flock import rec_vstage as S

    text = (stage_dir / "circuit.txt").read_text()
    data = bytearray((stage_dir / f"inst-{n}.bin").read_bytes())
    if json.loads(data[:data.index(b"\n")])["shared_rows"]["out"] != n or not 0 <= instance < n:
        raise ValueError(f"instance {instance}: not one of the {n} output rows")
    at = S._offsets(bytes(data), text, "out", instance)[0]
    was = int.from_bytes(data[at:at + 2], "little")
    data[at:at + 2] = (was ^ 1).to_bytes(2, "little")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"inst-{n}.bin").write_bytes(bytes(data))
    return {"instance": instance, "port": "out", "word": 0, "was": was, "now": was ^ 1}


def wired(cls: str, n: int, out: Path, seed: int = 0) -> dict:
    """``--wired``'s circuit (`universal.wired_topology`: four layers of the unit, a random hidden netlist each, its topology
    the Program's, public) on ``n`` instances as one class statement of ``4 n`` instances with link rows
    (`class_statement.Wiring`), plain leaves, every salt from the OS: the honest staging in ``out/stage`` and the cheating prover's, whose layer 1
    reads a made-up input in place of layer 0's outputs, every unit correct on what it is given, in ``out/made-up-input``,
    each with the verifier's program descriptor and partition object (`universal.run_wired`, ``--stage-only``); the
    verifier's own links in ``out/stage/links.json``, and none (``out/no-links.json``: a live verifier that checks no
    Linking, so it records the cheater's session)."""
    B = _benchmark()
    FC.stage_salts = lambda input_set, lo, hi, ports, key, tables=None: STAGE_SALTS(input_set, lo, hi, ports, None, tables)
    G, N_IN, N_OUT = (int(v) for v in cls.split(","))
    row = B.run_wired(argparse.Namespace(n=n, seed=seed, stage_only=True, keep_stage=True, lean=None), G, N_IN, N_OUT, out)
    d = out / f"wired-G{G}-N{N_IN}-O{N_OUT}"
    for name in ("stage", "made-up-input"):
        shutil.rmtree(out / name, ignore_errors=True)
        shutil.move(d / name, out / name)
    for f in ("partition.json", "program.desc"):
        shutil.copyfile(out / "stage" / f, out / "made-up-input" / f)
    shutil.move(d / "no-links.json", out / "no-links.json")
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
    for cmd in ("universal", "wired", "direct"):
        u = sp.add_parser(cmd)
        u.add_argument("--cls", required=True, help="G,N_IN,N_OUT")
        u.add_argument("--n", type=int, required=True, help="instances (wired: per layer, 4 n in all)")
        u.add_argument("--out", required=True)
        u.add_argument("--seed", type=int, default=0)
        if cmd != "wired":
            u.add_argument("--netlist")
        if cmd == "universal":
            u.add_argument("--fast", action="store_true", default=None)
            u.add_argument("--outside", action="store_true")
    pt = sp.add_parser("partition")
    pt.add_argument("--cls", required=True, help="G,N_IN,N_OUT")
    pt.add_argument("--circuit", required=True)
    pt.add_argument("--out", required=True)
    fy = sp.add_parser("faulty")
    fy.add_argument("--stage", required=True)
    fy.add_argument("--n", type=int, required=True)
    fy.add_argument("--out", required=True)
    fg = sp.add_parser("forge")
    fg.add_argument("--what", choices=("challenge", "query", "top", "unsalted"), required=True)
    fg.add_argument("--inner", required=True)
    fg.add_argument("--level", type=int, default=0)
    fl = sp.add_parser("flipped")
    for k in ("--inner", "--fold", "--points", "--salts", "--out"):
        fl.add_argument(k, required=True)
    df = sp.add_parser("differ")
    df.add_argument("a")
    df.add_argument("b")
    a = ap.parse_args(argv)
    if a.cmd == "flipped":
        res = flipped(a.inner, a.fold, a.points, a.salts, a.out)
    elif a.cmd == "differ":
        res = _differing(Path(a.a), Path(a.b))
    elif a.cmd == "faulty":
        res = faulty(Path(a.stage), a.n, Path(a.out))
    elif a.cmd == "partition":
        res = partition(a.cls, Path(a.circuit), Path(a.out))
    elif a.cmd == "universal":
        res = universal(a.cls, a.n, Path(a.out), a.seed, a.netlist, a.fast, outside=a.outside)
    elif a.cmd == "wired":
        res = wired(a.cls, a.n, Path(a.out), a.seed)
    elif a.cmd == "direct":
        res = direct(a.cls, a.n, Path(a.out), a.seed, a.netlist)
    else:
        res = stage(a.n, Path(a.sets), Path(a.out)) if a.cmd == "stage" else forge(a.what, Path(a.inner), a.level)
    json.dump(res, sys.stdout, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
