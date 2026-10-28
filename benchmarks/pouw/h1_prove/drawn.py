"""drawn.py: a served H1 window's drawn J10-R units staged as C-Flock statements, and the statements file core's credit
verifies (``--statements``, `credit.STATEMENT_KEYS`), from the serve dir and the prover's own state.

    python drawn.py stage SERVE_DIR WINDOW --circuits DIR --work W [--prover P] [--k K]
                          [--weights-key FILE | --gate CMD --model ROLE] [--checkpoint DIR | --weights NPZ]
    python drawn.py statements --work W --out FILE
    python drawn.py check SERVE_DIR WINDOW --circuits DIR --statements FILE --lean FLOCK_VERIFY

SERVE_DIR is serve-openai's ``--out`` (`served_verdicts.sh`'s D): the window's record ``h1/w<WINDOW>/pouw-h1-calls.json``,
the declared Program ``h1/window-program.json.gz`` and the period ``period.json`` (its ``draw_k``, else ``--k``); the
prover's state is ``private/h1/w<WINDOW>`` (`pouw_h1_prover`: its salt key and each served call's A codes, scales and z/y
records), or ``--prover``. DIR is `pouw_h1_circuits`' output, whose manifest binds each K's circuit file to the Program.

``stage``:

1. The draw is core's (`pouw_h1_challenger.drawn_of`: `credit.drawn` of the record's draw coin over the descriptor's
   layout at the period's K), grouped by call: a call's drawn units are one statement (a ``--registered`` port holds one
   root).
2. Each drawn call's ``--registered`` file is credit's own (`credit.registered_file` of the record's entries, the
   window's root, its rounds and its Program salt, `pouw_h1_challenger.program_salt_of`), so the statement opens every
   row where credit's verifier reads it.
3. The values those ports read are rebuilt as serve-openai committed them (`pouw_h1_served.Served.commit`): step s's
   ``a/<K>@<s>`` its calls of K's codes over M slots (`Served.a_words`) at `pouw_h1_served.nonce` of its first call,
   call c's ``zy@<c>`` its records (`zy_words`) at c's, both under the prover's key and bound to the window's root, and
   ``b/<K>`` K's linears' weight rows (`chunk_words`, from ``--checkpoint``'s safetensors, `live_challenger.models.parts`,
   or ``--weights``) at ``B_NONCE`` plus K's place under the auditor's key (``--weights-key``, or the gate's
   ``model-key --model ROLE --circuit C``; a server with neither kept no key). Each must have credit's root
   and domain, else the stage is refused; ``pub@<c>`` and ``ok`` are credit's own (`credit.public_table`,
   `credit.ok_table`).
4. K's statement (`h1unit.Shape` ``j10r``) is built bound to the manifest's bind; its circuit file must be the
   manifest's ``k<K>`` byte for byte (the SHA-512 the period-open committed). Each call is staged on it as a shared-row
   statement (`h1unit.statement_tables`), its outputs held to the word reference (`h1zy.reference` under J10-R) and
   every unit's ok 1. W/c<call>/ gets inst.bin, pub.bin, pin.txt and circuit.txt; W/plan.json lists them.

``statements``: after each W/c<call>/ is proved (`hidden_zk/job.sh`'s ``session``: exactly one session, all its
instances drawn), the statements file: each call's `pouw_h1_challenger.statement` (circuit, pub.bin, its session).

``check``: credit's own verification of those statements (`credit._verified` with the Lean ``flock-verify`` at
FLOCK_VERIFY, then `credit._proved`), the prover's look before it hands them over: ``PROVED`` or ``REFUSED``.
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import importlib.util
import itertools
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "hidden_zk"))

import h1unit as U  # noqa: E402

from verity.core.protocols.pouw import core_check as CC  # noqa: E402
from verity.core.protocols.pouw import credit as CR  # noqa: E402
from verity.core.protocols.pouw import weights as Wt  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import registered as G  # noqa: E402

NAME = "unit/pouw-h1-tile"
RECORD = "h1/w{w}/pouw-h1-calls.json"
PROGRAM = "h1/window-program.json.gz"
PERIOD = "period.json"
PROVER = "private/h1/w{w}"
MANIFEST = "circuit-files.json"
PLAN = "plan.json"
MODELS = HERE.parents[1] / "live_challenger" / "models.py"


class Refused(Exception):
    pass


# -- the window ------------------------------------------------------------------------------------------------------------

def window_of(serve_dir: Path, window: int) -> dict[str, Any]:
    """The window's record, descriptor bytes, layout, shape, and what credit verifies its statements under: the window's
    root, its rounds, its Program salt and each value's registered roots."""
    from verity_vllm.protocols.pouw.h1 import challenger as PHC
    record = json.loads((serve_dir / RECORD.format(w=window)).read_text())
    raw = gzip.decompress((serve_dir / PROGRAM).read_bytes())
    layout, why = CR.layout_of(raw)
    if why is not None:
        raise Refused(f"the declared Program's layout: {why[1]}")
    c = record["commitment"]
    served = c.get("served") or {}
    if served.get("refused"):
        raise Refused(f"the window's served trees were refused: {served['refused']}")
    roots: dict[str, list[str]] = {}
    for e in served.get("entries") or ():
        roots.setdefault(e.get("value"), []).append(e.get("root"))
    return {"record": record, "raw": raw, "program": hashlib.sha512(raw).hexdigest(), "layout": layout,
            "shape": CR.shape_of(raw), "root": c["session"]["root"], "steps": len(c["rounds"]),
            "salt": PHC.program_salt_of(c).hex(), "entries": list(served.get("entries") or ()), "roots": roots}


def draw_k(serve_dir: Path, k: int | None) -> int:
    if k is not None:
        return int(k)
    try:
        return int(json.loads((serve_dir / PERIOD).read_text())["draw_k"])
    except (OSError, KeyError, ValueError) as e:
        raise Refused(f"no draw K: give --k, or a serve dir whose {PERIOD} has the open's draw_k ({e})") from None


def drawn_calls(win: dict, k: int) -> dict[int, list[int]]:
    """Core's draw of the window at K, by call: ``{call: [global unit, ...]}``, ascending."""
    from verity_vllm.protocols.pouw.h1 import challenger as PHC
    units, why = PHC.drawn_of(win["record"], {"root": win["root"]}, k, win["raw"])
    if why is not None:
        raise Refused(f"core draws nothing from the window: {why[0]}: {why[1]}")
    starts = list(itertools.accumulate((size for size, _w in CR.call_work(win["layout"])), initial=0))[:-1]
    out: dict[int, list[int]] = {}
    for u in sorted(units):
        out.setdefault(bisect.bisect_right(starts, u) - 1, []).append(int(u))
    return out


# -- the registered values, rebuilt ---------------------------------------------------------------------------------------

def _kept(prover: Path, row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Call `row`'s kept A codes [served, K], scales [served] and z/y records (None for an empty call)."""
    from verity_vllm.protocols.pouw.h1 import prover as PV
    k, m = int(row["k"]), int(row["served"])
    if not m:
        return np.zeros((0, k), np.int8), np.zeros(0, "<f4"), None
    with np.load(prover / PV.CALLS / f"{int(row['index'])}.npz") as x:
        return x["a"].reshape(m, k), x["s_a"].reshape(-1).astype("<f4"), x["records"]


def a_value(win: dict, prover: Path, key: bytes, call: int) -> G.Registered:
    """Call `call`'s step's ``a/<K>@<s>`` as `Served.commit` registered it: its step's calls of K, each its served codes
    over M slots (`Served.a_words`), at the first one's ``a`` nonce, bound to the window's root."""
    from verity_vllm.protocols.pouw.h1 import registered as NR
    from verity_vllm.protocols.pouw.h1 import served as SV
    lay, prefix = win["layout"], win["shape"]["prefix"]
    per, ks = len(lay["linears"]), {name: k for name, k, _n in lay["linears"]}
    s, k = call // per, ks[lay["calls"][call][0]]
    mine = [d for d in range(s * per, (s + 1) * per) if ks[lay["calls"][d][0]] == k]
    rows = {int(r["index"]): r for r in win["record"]["calls"]}
    words = []
    for d in mine:
        codes, scales, _r = _kept(prover, rows[d])
        full, sc = np.zeros((lay["m"], k), np.int8), np.zeros(lay["m"], "<f4")
        full[:len(codes)], sc[:len(codes)] = codes, scales
        words.append(SV.chunk_words(full, sc, Wt.chunks(k)))
    return NR.commit(win["program"], f"{prefix}a/{k}@{s}", np.concatenate(words), key, SV.nonce(mine[0], "a"),
                     keep_rows=True, window=win["root"])


def zy_value(win: dict, prover: Path, key: bytes, call: int) -> G.Registered:
    """Call `call`'s ``zy@<c>`` as `Served.commit` registered it (`zy_words` of its records over M slots, zeros for an
    empty call), at its ``zy`` nonce, bound to the window's root."""
    from verity_vllm.protocols.pouw.h1 import registered as NR
    from verity_vllm.protocols.pouw.h1 import served as SV
    lay = win["layout"]
    row = next(r for r in win["record"]["calls"] if int(r["index"]) == call)
    n, tile = int(row["n"]), tuple(lay["tile"])
    _a, _s, records = _kept(prover, row)
    if records is None or not records.size:
        words = SV.record_words(np.zeros((0, n), "<i4"), np.zeros((0, n), np.int16), lay["m"], tile)
    else:
        words = SV.zy_words(records, int(row["rows"]), n, lay["m"], tile)
    return NR.commit(win["program"], f"{win['shape']['prefix']}zy@{call}", np.asarray(words), key, SV.nonce(call, "zy"),
                     keep_rows=True, window=win["root"])


def _models():
    spec = importlib.util.spec_from_file_location("live_challenger_models", MODELS)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def linear_weights(linears: list[tuple[str, int, int]], k: int, checkpoint: Path | None, weights: Path | None
                   ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each of K's linears' int8 codes (n, K) and f32 s_w (n,), in the window's order: from the checkpoint's safetensors
    (a fused linear its parts' rows in turn, `live_challenger.models.parts`), or from ``weights`` (``w<i>`` and ``s<i>``,
    i the linear's index in the layout)."""
    out = []
    if weights is not None:
        with np.load(weights) as x:
            for i, (_name, kk, n) in enumerate(linears):
                if kk == k:
                    out.append((x[f"w{i}"].astype(np.int8).reshape(n, k), x[f"s{i}"].astype("<f4").reshape(n)))
        return out
    if checkpoint is None:
        raise Refused("b/<K>'s rows are the served model's weights: give --checkpoint (its safetensors) or --weights")
    M = _models()
    ckpt = M.Checkpoint(checkpoint, sorted(p.name for p in checkpoint.iterdir() if p.name.endswith(".safetensors")))
    for name, kk, n in linears:
        if kk != k:
            continue
        codes = [np.asarray(ckpt.tensor(f"{p}.weight")) for p in M.parts(name)]
        scales = [np.asarray(ckpt.tensor(f"{p}.weight_scale"), "<f4").reshape(-1) for p in M.parts(name)]
        w, s = np.concatenate(codes), np.concatenate(scales)
        if w.dtype != np.int8 or w.shape != (n, k) or s.shape != (n,):
            raise Refused(f"{name}: the checkpoint's weight {w.shape} {w.dtype} and scale {s.shape} are not its {n} × {k} "
                          "int8 codes and n scales")
        out.append((w, s))
    return out


def b_value(win: dict, k: int, key: bytes, checkpoint: Path | None, weights: Path | None) -> G.Registered:
    """``b/<K>`` as `pouw_h1_served.register_weights` registered it: K's linears' rows in turn (`chunk_words`), at
    ``B_NONCE`` plus K's place among the window's K's, under `key`, bound to no window."""
    from verity_vllm.protocols.pouw.h1 import registered as NR
    from verity_vllm.protocols.pouw.h1 import served as SV
    lin = [tuple(x) for x in win["layout"]["linears"]]
    words = np.concatenate([SV.chunk_words(c, s, Wt.chunks(k)) for c, s in linear_weights(lin, k, checkpoint, weights)])
    ks = sorted({kk for _n, kk, _x in lin})
    return NR.commit(win["program"], f"{win['shape']['prefix']}b/{k}", words, key, SV.B_NONCE + ks.index(k),
                     keep_rows=True)


def weights_key(a, win: dict) -> tuple[bytes, str]:
    """The key ``b/<K>`` is salted under, and whose: ``--weights-key``'s (32 bytes, raw or hex), else the gate's
    ``model-key`` for ``--model`` under the window's circuit (serve-openai's ``--h1-model``). A server with neither
    salts it under a fresh key it never writes (`serve_h1.H1Windows`), so its windows' b/<K> can't be opened."""
    if a.weights_key:
        raw = Path(a.weights_key).read_bytes()
        key = raw if len(raw) == 32 else bytes.fromhex(raw.decode().strip())
        return key, "--weights-key"
    if a.gate:
        if not a.model:
            raise Refused("--gate reads the auditor's key with model-key --model ROLE: give --model")
        words = ["model-key", "--model", a.model, "--circuit", win["program"], "--weights-root",
                 win["record"]["commitment"]["weights_root"]]
        p = subprocess.run([*shlex.split(a.gate), shlex.join(words)], capture_output=True, check=False)
        try:
            answer = json.loads(p.stdout)
            key = bytes.fromhex(answer["key"])
        except (ValueError, KeyError, TypeError):
            raise Refused(f"the gate's model-key answered no key (exit {p.returncode}): "
                          f"{(p.stdout + p.stderr).decode(errors='replace')[-300:]}") from None
        if hashlib.sha512(key).hexdigest() != answer.get("key_sha512"):
            raise Refused("the gate's model-key key is not the one its key_sha512 names")
        return key, f"the gate's model-key for {a.model}"
    raise Refused("b/<K>'s salt key: give --weights-key, or --gate and --model (the auditor's registration the server "
                  "committed b/<K> under, its --h1-model); a server with neither salted b/<K> under a key it never kept")


def held(reg: G.Registered, entry: dict, what: str) -> G.Registered:
    """`reg` when it has the root and domain credit's ``--registered`` port `entry` reads; else Refused."""
    if (reg.root.hex(), reg.domain.domain_id.hex()) != (entry["root"], entry["domain"]):
        raise Refused(f"{entry['value']}, rebuilt from {what}, is not the tree the window registered (root "
                      f"{reg.root.hex()[:16]}... domain {reg.domain.domain_id.hex()[:16]}..., registered "
                      f"{entry['root'][:16]}... domain {entry['domain'][:16]}...)")
    return reg


# -- staging --------------------------------------------------------------------------------------------------------------

def call_rows(win: dict, call: int, units: list[int], reg: dict, values: dict[str, G.Registered], pub: G.Registered,
              ok: G.Registered) -> dict[str, np.ndarray]:
    """The statement's rows of call `call`'s drawn `units` as `h1unit.statement_tables` takes them (``w_``, ``s_``, and
    for a registered port ``r_``, ``v_``, ``p_``): each port of credit's file `reg` at the row it reads, from `values`
    (each value's rebuilt tree), the public table `pub` and the ok rows `ok`; o0's salts fresh (no entry reads it)."""
    ports = reg["ports"]
    u0 = ports[CR.PUB_PORT][G.READS_FROM]
    z: dict[str, Any] = {"units": np.array(units)}
    for port, e in ports.items():
        src = pub if port == CR.PUB_PORT else ok if port == CR.OK_PORT else values[e["value"]]
        at = [CC.port_reads(e)[u] for u in units]
        if any(x is None for x in at):
            raise Refused(f"port {port} reads no row at a drawn unit of call {call}")
        z[f"w_{port}"] = np.array([src.rows[x] for x in at], np.uint16)
        z[f"s_{port}"] = np.frombuffer(b"".join(bytes(src.salts[x]) for x in at), np.uint8).reshape(len(at), U.SALT_BYTES)
        z[f"r_{port}"] = np.array(at, np.int64)
        z[f"v_{port}"] = np.array([e["value"]] * len(at))
        z[f"p_{port}"] = np.array([src.path(x).hex() for x in at])
    z["s_o0"] = np.frombuffer(os.urandom(U.SALT_BYTES * len(units)), np.uint8).reshape(len(units), U.SALT_BYTES)
    z["u0"] = np.array(u0)
    return z


def reference_check(z: dict, outs: dict, shape: U.Shape, win: dict, k: int, roots: tuple[bytes, bytes]) -> None:
    """Each staged unit's receipt and ok are J10-R's word reference's (`h1zy.reference`, chunked by `chunks_reference`)
    on its rows, the window's Program salt, its position, its call's roots and (u0, blocks); and its ok is 1."""
    from verity.core.protocols.pouw.circuit import h1zy as Z
    lay, (r, c), x = win["layout"], tuple(win["layout"]["tile"]), Wt.chunks(k)
    u0, salt = int(z["u0"]), bytes.fromhex(win["salt"])
    n = next(nn for name, kk, nn in lay["linears"] if name == lay["calls"][int(z["call"])][0])
    for i, u in enumerate(int(v) for v in z["units"]):
        t = u - u0
        a = [U.split_row(z[f"w_a_k{k}_{q}"][i]) for q in range(x * r)]
        b = [U.split_row(z[f"w_b_k{k}_{q}"][i]) for q in range(x * c)]
        j10r = (u0, CC.j10r_blocks(lay["m"], k, (r, c), t // (n // c), t % (n // c)))
        args = ([v for v, _s in a], [v for v, _s in b], [s for _v, s in a], [s for _v, s in b],
                U.split_record(z["w_zy"][i], r * c), salt, u, roots)
        rc, ok = (Z.reference(*args, j10r=j10r) if x == 1 else Z.chunks_reference(*args, x, j10r=j10r))
        if ok != 1 or int(outs["o1"][i][0]) != 1:
            raise Refused(f"unit {u}: its ok is {int(outs['o1'][i][0])} (the word reference's {ok}), not 1: its record is "
                          "not its rows' product, so it is not proved")
        if not np.array_equal(outs["o0"][i], U.receipt_words(rc)):
            print(f"WARNING unit {u}: the lowered circuit's receipt is not the word reference's (no entry reads it)",
                  flush=True)


def stage(a) -> int:
    serve_dir, work = Path(a.serve_dir), Path(a.work)
    prover = Path(a.prover) if a.prover else serve_dir / PROVER.format(w=a.window)
    win = window_of(serve_dir, a.window)
    k_draw = draw_k(serve_dir, a.k)
    drawn = drawn_calls(win, k_draw)
    manifest = json.loads((Path(a.circuits) / MANIFEST).read_text())
    if manifest.get("program") != win["program"]:
        raise Refused(f"--circuits' manifest is for Program {str(manifest.get('program'))[:16]}..., not the window's "
                      f"{win['program'][:16]}...")
    from verity_vllm.protocols.pouw.h1 import prover as PV
    pkey = PV.key_of(prover)
    bkey, bwhose = weights_key(a, win)
    lay, ks = win["layout"], {name: k for name, k, _n in win["layout"]["linears"]}
    by_k: dict[int, list[int]] = {}
    for c in drawn:
        by_k.setdefault(ks[lay["calls"][c][0]], []).append(c)
    work.mkdir(parents=True, exist_ok=True)
    print(f"DRAWN at K = {k_draw}: {sum(map(len, drawn.values()))} units in calls {sorted(drawn)}", flush=True)
    population = sum(size for size, _w in CR.call_work(lay))
    _rows, _salts, ok = CR.ok_table(win["program"], f"{win['shape']['prefix']}{CR.OK_VALUE}", win["root"], population)
    plan = {"window": a.window, "draw_k": k_draw, "program": win["program"], "calls": []}
    for k, calls in sorted(by_k.items()):
        if not win["shape"]["tiles"][k]["j10r"]:
            raise Refused(f"K = {k}'s template is not J10-R's: drawn.py stages J10-R units")
        if f"k{k}" not in manifest.get("files", {}):
            raise Refused(f"--circuits has no k{k} file: build it (pouw_h1_circuits --strata {k})")
        b = None
        values: dict[str, G.Registered] = {}
        regs = {}
        for c in calls:
            reg, why = CR.registered_file(win["program"], lay, win["shape"], c, win["roots"], win["root"], win["steps"],
                                          win["salt"])
            if why is not None:
                raise Refused(f"call {c}: credit builds no registered file: {why[1]}")
            regs[c] = reg
            for port, e in reg["ports"].items():
                v, name = e["value"], e["value"][len(win["shape"]["prefix"]):]
                if port in (CR.PUB_PORT, CR.OK_PORT) or v in values:
                    continue
                if name == f"b/{k}":
                    b = b or held(b_value(win, k, bkey, a.checkpoint and Path(a.checkpoint), a.weights and Path(a.weights)),
                                  e, f"the served weights under {bwhose}")
                    values[v] = b
                elif name == f"zy@{c}":
                    values[v] = held(zy_value(win, prover, pkey, c), e, "the prover's records")
                elif name.startswith(f"a/{k}@"):
                    values[v] = held(a_value(win, prover, pkey, c), e, "the prover's A codes")
                else:
                    raise Refused(f"call {c}'s statement reads {v} at port {port}, which drawn.py doesn't rebuild (a "
                                  "receipt table: stage it as the window registered it)")
        print(f"REBUILT K = {k}: {sorted(values)} at the window's roots", flush=True)
        if os.environ.get("HZ_STAGE", "fast") == "fast":
            import fast_stage
            fast_stage.install(work / f".scratch-k{k}")
        from verity.core.protocols.sampled_proofs.one_stage import proofs as PF
        shape = U.Shape(k, *lay["tile"], j10r=True)
        built = work / f"k{k}.circuit.txt"
        stager = shape.stager(NAME, circuit=built, bind=manifest["bind"])
        got = PF.file_sha512(built)
        if got != manifest["files"][f"k{k}"]:
            raise Refused(f"K = {k}'s statement circuit (SHA-512 {got[:16]}...) is not --circuits' k{k} file "
                          f"({manifest['files'][f'k{k}'][:16]}...), the one the period-open committed")
        built.unlink()
        circuit = (Path(a.circuits) / f"k{k}.circuit.txt").resolve()
        out_ports = [p.name for p in stager.ports if p.name not in stager.row_ports]
        for c in calls:
            reg, units = regs[c], drawn[c]
            pv = reg["ports"][CR.PUB_PORT]
            pub = held(G.Registered.commit(win["program"], pv["value"], pv["known"]["rows"],
                                           [bytes.fromhex(s) for s in pv["known"]["salts"]], G.WORD_BITS, G.SCHEMA_V2,
                                           window=win["root"]), pv, "credit's public table")
            z = call_rows(win, c, units, reg, values, pub, ok)
            z["call"] = np.array(c)
            st = U.statement_tables(z, stager.row_ports, out_ports)
            roots = (bytes.fromhex(reg["ports"][f"a_k{k}_0"]["root"]), bytes.fromhex(reg["ports"][f"b_k{k}_0"]["root"]))
            stager.reference_check = lambda _rows, outs, z=z, roots=roots: reference_check(z, outs, shape, win, k, roots)
            d = work / f"c{c}"
            got = stager.stage({p: z[f"w_{p}"] for p in stager.row_ports}, d, os.urandom(32), salts=st["salts"],
                               indices=units, set_name=f"pouw-h1-w{a.window}-c{c}", tables=st["tables"], refs=st["refs"],
                               registered=st["registered"])
            (d / "pin.txt").write_text(stager.pin)
            (d / "circuit.txt").unlink(missing_ok=True)
            (d / "circuit.txt").symlink_to(circuit)
            (d / "registered.json").write_text(json.dumps(reg))
            plan["calls"].append({"call": c, "k": k, "dir": str(d.resolve()), "units": units, "instances": len(units),
                                  "circuit": str(circuit), "pin": got["circuit_sha512"]})
            print(f"STAGED call {c}: units {units}", flush=True)
    (work / PLAN).write_text(json.dumps(plan, indent=1))
    print("PLAN " + json.dumps({"calls": [(x["call"], x["instances"]) for x in plan["calls"]]}), flush=True)
    return 0


# -- statements and their check ---------------------------------------------------------------------------------------------

def statements(a) -> int:
    from verity_vllm.protocols.pouw.h1 import challenger as PHC
    plan = json.loads((Path(a.work) / PLAN).read_text())
    out = []
    for x in plan["calls"]:
        d = Path(x["dir"])
        sessions = sorted(p for p in (d / "sessions").glob("*/") if (p / "session.json").is_file())
        if len(sessions) != 1:
            raise Refused(f"call {x['call']}: {len(sessions)} sessions in {d}/sessions, not one: a drawn call is proved "
                          "once, every instance drawn (session's subset:N)")
        out.append(PHC.statement(x["call"], x["circuit"], d / "pub.bin", sessions))
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"STATEMENTS {len(out)} calls -> {a.out}", flush=True)
    return 0


def check(a) -> int:
    from verity.core.protocols.sampled_proofs.one_stage import proofs as PF
    PF.executable = lambda: Path(a.lean)
    serve_dir = Path(a.serve_dir)
    win = window_of(serve_dir, a.window)
    k = draw_k(serve_dir, a.k)
    drawn = drawn_calls(win, k)
    units = sorted(u for us in drawn.values() for u in us)
    files = json.loads((Path(a.circuits) / MANIFEST).read_text())["files"]
    sts = json.loads(Path(a.statements).read_text())
    proofs, why = CR._verified(sts, files, win["raw"], win["layout"], win["entries"], win["root"], units, win["steps"],
                               PF.verify, win["salt"])
    if why is None:
        why = CR._proved(units, proofs)
    print(("PROVED " + json.dumps({"units": units, "outcomes": len(proofs)})) if why is None else
          f"REFUSED {why[0]}: {why[1]}", flush=True)
    return 0 if why is None else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("serve_dir")
    s.add_argument("window", type=int)
    s.add_argument("--circuits", required=True)
    s.add_argument("--work", required=True)
    s.add_argument("--prover")
    s.add_argument("--k", type=int)
    s.add_argument("--weights-key")
    s.add_argument("--gate")
    s.add_argument("--model")
    s.add_argument("--checkpoint")
    s.add_argument("--weights")
    t = sub.add_parser("statements")
    t.add_argument("--work", required=True)
    t.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("serve_dir")
    c.add_argument("window", type=int)
    c.add_argument("--circuits", required=True)
    c.add_argument("--statements", required=True)
    c.add_argument("--lean", required=True)
    c.add_argument("--k", type=int)
    a = ap.parse_args(argv)
    try:
        return {"stage": stage, "statements": statements, "check": check}[a.cmd](a)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
