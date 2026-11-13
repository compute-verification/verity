"""A served H1 window's sampled proof blocks staged as Flock statements (a J10-R K's a statement a call, a lanes-form K's each
unit's lanes and combine), and the statements file core's credit verifies
(``--statements``, `credit.STATEMENT_KEYS`), from the serve dir and the prover's own state. ``benchmarks/pouw/h1_prove/drawn.py``
is its CLI, which reads ``--checkpoint`` with the live challenger's model registry (``checkpoint_weights``) and binds
``check`` to credit's own verification (``verified``), running `run` on its arguments.

    sampled stage SERVE_DIR WINDOW --circuits DIR --work W [--prover P] [--k K] [--recompose]
                [--weights-key FILE | --gate CMD --model ROLE] [--checkpoint DIR | --weights NPZ]
    sampled statements --work W --out FILE
    sampled check SERVE_DIR WINDOW --circuits DIR --statements FILE --lean FLOCK_VERIFY

SERVE_DIR is serve-openai's ``--out`` (`served_verdicts.sh`'s D): the window's record ``h1/w<WINDOW>/pouw-h1-calls.json``,
the declared Program ``h1/window-program.json.gz`` and the period ``period.json`` (its ``draw_k``, else ``--k``); the
prover's state is ``private/h1/w<WINDOW>`` (`prover`: its salt key and each served call's A codes, scales and z/y
records), or ``--prover``. DIR is `pouw_h1_circuits`' output, whose manifest binds each K's circuit file to the Program.

``stage``:

1. The sample is core's (`challenger.drawn_of`: `credit.drawn` of the record's sampling coin over the descriptor's layout at the
   period's K), grouped by call: a call's sampled proof blocks are one statement (a ``--registered`` port holds one root).
2. Each sampled call's ``--registered`` file is credit's own (`credit.registered_file` of the record's entries, the
   window's root, its rounds and its Program salt, `challenger.program_salt_of`), so the statement opens every row where
   credit's verifier reads it. Every port of the statement's circuit must be one of its ports (`complete`, the Lean
   verifier's ``registered reads (complete)``), o0 the unit's receipt among them: the window must register each call's
   receipt table ``receipt@<c>`` (serve-openai ``--h1-receipts``), else the stage is refused.
3. The values those ports read are rebuilt as serve-openai committed them (`served.Served.commit`): step s's
   ``a/<K>@<s>`` its calls of K's codes over M slots (`Served.a_words`) at `served.nonce` of its first call, call c's
   ``zy@<c>`` its records (`zy_words`) at c's, both under the prover's key and bound to the window's root, and ``b/<K>``
   K's linears' weight rows (`chunk_words`, from ``--checkpoint``'s safetensors through ``checkpoint_weights``, or
   ``--weights``) at ``B_NONCE`` plus K's place under the auditor's key (``--weights-key``, or the gate's ``model-key
   --model ROLE --circuit C``; a server with neither kept no key); then call c's ``receipt@<c>``, each of its units'
   receipt as J10-R's word reference computes it on those rows (`receipt_table`), at ``RECEIPT_NONCE`` + c under the
   prover's key. Each must have credit's root and domain, else the stage is refused; ``pub@<c>`` and ``ok`` are credit's
   own (`credit.public_table`, `credit.ok_table`).
4. Each call is staged as a shared-row statement (`statement_tables`) on --circuits' ``k<K>`` file, whose SHA-512 must be
   the manifest's (the one the period-open committed), read and never composed (`FileStager`: the K 2048 J10-R
   composition takes hundreds of GB), each unit's outputs J10-R's word reference's (`h1zy.reference`), its ok 1 and both
   outputs the rows the window registered (`outputs_held`), o0 opened at receipt@<c> as ok is at ``ok``.
   ``--recompose`` builds K's flat statement (`stager`, bound to the manifest's bind) instead, its lowered outputs held to
   the word reference's and to those rows. W/c<call>/ gets inst.bin, pub.bin, pin.txt and circuit.txt, and program.json
   and partition.json, the window Program and `credit.partition_of`'s object credit verifies it under (`bind_files`);
   W/plan.json lists them.

**The lanes form** (`credit`'s, `leaf_lanes`): at a K --circuits' manifest commits as ``lane<K>`` and ``combine<K>``
(`credit.form_of`; `pouw_h1_circuits --lanes K`), each sampled proof block is a leaf proved as its (R/4)(C/8) lanes and the combine
(`leaf_lanes.statements`, their rows `drawn_rows`'), each a statement of one instance on its committed file, so a K
whose tile statement is past Flock's block (K 2048 on the 32 × 32 tile) is proved in pieces each within it. The leaf is
v1's (`credit.lanes_shape` refuses any other, J10-R's among them). Its partials (`h1zy.lane_partial` of each lane's
rows) are committed after the sample, ``<prefix>partials@<u>`` under the window's sample root (`produced.commit`), all of them one produced record
(`produced.record`, the sampling coin's); its call's receipts ``receipt@<c>`` are rebuilt as the word reference computes them
(`receipt_value`, `h1zy.reference` of each of the call's units) and held to the root the window registered; each
statement's ``--registered`` file is credit's own (`credit.leaf_registered_file`), its known rows (``pub``, ``admit`` and
``o0``) committed under `leaf_lanes.known_salt`, and its ok the word reference's (`h1zy.lane_reference`,
`h1zy.combine_reference`), refused unless 1. W/u<unit>-<statement>/ gets each statement's files, its program.json and
partition.json its leaf Program and `credit.leaf_partition_of`'s object (`bind_files`); W/plan.json lists them
under ``leaves`` beside the record (``produced``) and each leaf Program's descriptor (``leaf_programs``, the manifest's
``<file>.program.json``).

``statements``: after each W/c<call>/ and W/u<unit>-<statement>/ is proved (`hidden_zk/job.sh`'s ``session``: exactly
one session, all its instances sampled, its prover under the ZK server on those two files, ``h1_prove/drawn.sh``), the
statements file: each call's `challenger.statement` (circuit, pub.bin, its
session), and with leaves the lanes evidence (`credit.LANES_KEYS`: ``leaf_programs``, ``produced`` and those statements
with each leaf statement's ``unit`` and ``statement``).

``check``: credit's own verification of those statements over the window's sampled proof blocks under its sample (``verified``:
the CLI's is `credit._verified` with the Lean ``flock-verify`` at FLOCK_VERIFY, then `credit._proved`), the prover's look
before it hands them over: ``PROVED`` or ``REFUSED``. Credit's verdict of record (`verifier.RECORD_VERDICT` ``lean``,
`PouwCheck.Verified`, the lanes form among it) is the Lean's; this is the Python reference's. Each verifier run is
answered as credit asks for it (`answers.Answers`), one ``LEAN-ANSWER`` line a verdict, and judged in credit's order.
"""
from __future__ import annotations

import bisect
import gzip
import hashlib
import itertools
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from verity.protocols.pouw.prover.circuit import h1tile as HT
from verity.protocols.pouw.prover.circuit import h1window as HW
from verity.protocols.pouw.prover.circuit import h1zy as Z
from verity.protocols.pouw.verifier import core_check as CC
from verity.protocols.pouw.verifier import credit as CR
from verity.protocols.pouw.verifier import leaf_lanes as LL
from verity.protocols.pouw.verifier import weights as Wt
from verity.protocols.spot.device import challenger as Ch
from verity.protocols.spot.verifier.one_stage import produced as PR
from verity.protocols.spot.verifier.one_stage import proofs as PF
from verity.protocols.spot.verifier.one_stage import registered as G

from verity_vllm.errors import Refused
from verity_vllm.protocols.pouw.h1 import challenger as PHC
from verity_vllm.protocols.pouw.h1 import prover as PV
from verity_vllm.protocols.pouw.h1 import registered as NR
from verity_vllm.protocols.pouw.h1 import served as SV
from verity_vllm.protocols.pouw.h1.drawn_rows import (
    LEAF_NAMES,
    LEAF_OUT_ROWS,
    LEAF_PROGRAM,
    complete,
    held,
    leaf_rows,
    leaf_statement,
    operands,
    outputs_held,
    partials,
    receipt_value,
    receipt_words,
    split_record,
    split_row,
)

NAME = "unit/pouw-h1-tile"
RECORD = "h1/w{w}/pouw-h1-calls.json"
PROGRAM = "h1/window-program.json.gz"
PERIOD = "period.json"
PROVER = "private/h1/w{w}"
MANIFEST = "circuit-files.json"
PLAN = "plan.json"
SALT_BYTES = CC.SALT_BYTES
OK = np.array(CC.OK_ROW, np.uint16)
_random = os.urandom

#: ``--checkpoint``'s reader: (checkpoint dir, the window's linears ``(name, K, n)``, K) -> each of K's linears' int8 codes
#: (n, K) and f32 s_w (n,), in the window's order
CheckpointWeights = Callable[[Path, Sequence[tuple[str, int, int]], int], list[tuple[np.ndarray, np.ndarray]]]
#: ``check``'s verifier: (the statements, --circuits' manifest's ``files`` (``k<K>``, or ``lane<K>``, ``combine<K>`` and
#: ``leaf_programs``), the window (`window_of`, its ``draw`` among it), its sampled proof blocks, the Lean ``flock-verify``) -> each
#: sampled proof block's proof outcomes and None, or None and credit's refusal
Verified = Callable[[Any, dict, dict, list[int], Path], tuple[Any, tuple[str, str] | None]]


# -- J10-R's statement ---------------------------------------------------------------------------------------------------

def template(k: int, tile: Sequence[int], m: int) -> Any:
    """J10-R's statement Definition at K: the template a window Program over J10-R's tile unit (`h1tile.PouwH1Tile5`
    under `h1window.tile_unit`) at slot `m` declares for K (its ``TILES``), the one its partition lists."""
    r, c = tile
    program = HW.program([("j10r", k, c)], [("j10r", 0)], m, (r, c), unit=lambda kk: HW.tile_unit(kk, (r, c), HT.PouwH1Tile5))
    return dict(program.fn.bindings["TILES"])[k]


def stager(k: int, tile: Sequence[int], m: int, circuit: Path, bind: dict) -> Any:
    """K's flat statement's `UnitStager` (the form the Lean verifier accepts) on `h1window.statement_rows`, its circuit
    built once and written to `circuit`, bound to `bind` (``{"partition", "program"}``), as `pouw_h1_circuits` builds K's
    file."""
    from verity_flock import class_statement as CS
    t = template(k, tile, m)
    return CS.UnitStager(t, NAME, HW.statement_rows(t), HW.STATEMENT_OUT_ROWS, circuit=circuit, bind=bind)


def _table(positions: Sequence[int]) -> tuple[list[int], list[int], list[int]]:
    """A registered port's shared-row table over its instances' positions (`registered.table`): the table's positions,
    each instance's ref into it, and the first instance that reads each table row."""
    tpos, refs = G.table([int(p) for p in positions])
    first: dict[int, int] = {}
    for i, r in enumerate(refs):
        first.setdefault(r, i)
    return tpos, refs, [first[t] for t in range(len(tpos))]


def _one_value(port: str, values: Sequence[str]) -> str:
    """The one registered value a statement's port reads at all its instances: a port reads one value, so a statement
    holds one call's units."""
    got = sorted({str(v) for v in values})
    if len(got) != 1:
        raise ValueError(f"port {port}: its instances read {got}, and a port reads one registered value: one call's units a "
                         f"statement")
    return got[0]


def statement_tables(z: dict, in_ports: Sequence[str], out_ports: Sequence[str]) -> dict:
    """The shared-row statement of rows ``z`` (`call_rows`): each registered port's table (``r_<port>``, the rows'
    positions in the value ``v_<port>`` names, one value a port) the distinct positions its instances read in first-read
    order, under their registered salts and opened paths (``p_<port>``), and every other input port's one row per
    instance; an output port with positions registered too, its table its instances' own rows; as `UnitStager.stage`
    takes them: ``tables``, ``refs``, ``salts`` (every port's, one per table row) and ``registered``."""
    n = len(z["units"])
    tables, refs, salts, registered = {}, {}, {}, {}
    for p in in_ports:
        w, s = np.asarray(z[f"w_{p}"], np.uint16), [bytes(x) for x in z[f"s_{p}"]]
        if f"r_{p}" not in z:
            tables[p], refs[p], salts[p] = w, list(range(n)), s
            continue
        value = _one_value(p, z[f"v_{p}"])
        tpos, refs[p], first = _table(z[f"r_{p}"])
        paths = [str(x) for x in z[f"p_{p}"]]
        for i, t in enumerate(refs[p]):
            j = first[t]
            if (s[i], paths[i]) != (s[j], paths[j]) or not np.array_equal(w[i], w[j]):
                raise ValueError(f"port {p}: instances {j} and {i} read position {tpos[t]} as two rows")
        tables[p], salts[p] = w[first], [s[j] for j in first]
        registered[p] = {"value": value, "positions": tpos, "paths": [paths[j] for j in first]}
    for p in out_ports:
        salts[p] = [bytes(x) for x in z[f"s_{p}"]]
        if f"r_{p}" in z:
            registered[p] = {"value": _one_value(p, z[f"v_{p}"]), "positions": [int(x) for x in z[f"r_{p}"]],
                             "paths": [str(x) for x in z[f"p_{p}"]]}
    return {"tables": tables, "refs": refs, "salts": salts, "registered": registered}


# -- the window ------------------------------------------------------------------------------------------------------------

def window_of(serve_dir: Path, window: int) -> dict[str, Any]:
    """The window's record, descriptor bytes, layout, shape, and what credit verifies its statements under: the window's
    root, its rounds, its Program salt and each value's registered roots."""
    record = json.loads((serve_dir / RECORD.format(w=window)).read_text())
    raw = gzip.decompress((serve_dir / PROGRAM).read_bytes())
    layout, why = CR.layout_of(raw)
    if why is not None:
        raise Refused(f"the declared Program's layout: {why[1]}")
    c = record["commitment"]
    served = c.get("served")
    if not isinstance(served, dict) or not isinstance(served.get("entries"), list):
        raise Refused("the window's record names no served trees: drawn rebuilds the values they registered")
    if served.get("refused"):
        raise Refused(f"the window's served trees were refused: {served['refused']}")
    roots: dict[str, list[str]] = {}
    for e in served["entries"]:
        roots.setdefault(e.get("value"), []).append(e.get("root"))
    return {"record": record, "raw": raw, "program": hashlib.sha512(raw).hexdigest(), "layout": layout,
            "shape": CR.shape_of(raw), "root": c["session"]["root"], "steps": len(c["rounds"]),
            "salt": PHC.program_salt_of(c).hex(), "entries": list(served["entries"]), "roots": roots,
            "draw": c.get("draw")}


def draw_k(serve_dir: Path, k: int | None) -> int:
    if k is not None:
        return int(k)
    try:
        return int(json.loads((serve_dir / PERIOD).read_text())["draw_k"])
    except (OSError, KeyError, ValueError) as e:
        raise Refused(f"no draw K: give --k, or a serve dir whose {PERIOD} has the open's draw_k ({e})") from None


def drawn_calls(win: dict, k: int) -> dict[int, list[int]]:
    """Core's sample of the window at K, by call: ``{call: [global unit, ...]}``, ascending."""
    units, why = PHC.drawn_of(win["record"], {"root": win["root"]}, k, win["raw"])
    if why is not None or units is None:
        raise Refused(f"core draws nothing from the window: {why[0]}: {why[1]}")
    starts = list(itertools.accumulate((size for size, _w in CR.call_work(win["layout"])), initial=0))[:-1]
    out: dict[int, list[int]] = {}
    for u in sorted(units):
        out.setdefault(bisect.bisect_right(starts, u) - 1, []).append(int(u))
    return out


# -- the registered values, rebuilt ---------------------------------------------------------------------------------------

def _kept(prover: Path, row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Call `row`'s kept A codes [served, K], scales [served] and z/y records (None for an empty call)."""
    k, m = int(row["k"]), int(row["served"])
    if not m:
        return np.zeros((0, k), np.int8), np.zeros(0, "<f4"), None
    with np.load(prover / PV.CALLS / f"{int(row['index'])}.npz") as x:
        return x["a"].reshape(m, k), x["s_a"].reshape(-1).astype("<f4"), x["records"]


def a_value(win: dict, prover: Path, key: bytes, call: int) -> G.Registered:
    """Call `call`'s step's ``a/<K>@<s>`` as `Served.commit` registered it: its step's calls of K, each its served codes
    over M slots (`Served.a_words`), at the first one's ``a`` nonce, bound to the window's root."""
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


def linear_weights(linears: list[tuple[str, int, int]], k: int, checkpoint: Path | None, weights: Path | None,
                   read: CheckpointWeights | None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each of K's linears' int8 codes (n, K) and f32 s_w (n,), in the window's order: from the checkpoint's safetensors
    by `read`, or from ``weights`` (``w<i>`` and ``s<i>``, i the linear's index in the layout)."""
    if weights is not None:
        out = []
        with np.load(weights) as x:
            for i, (_name, kk, n) in enumerate(linears):
                if kk == k:
                    out.append((x[f"w{i}"].astype(np.int8).reshape(n, k), x[f"s{i}"].astype("<f4").reshape(n)))
        return out
    if checkpoint is None:
        raise Refused("b/<K>'s rows are the served model's weights: give --checkpoint (its safetensors) or --weights")
    if read is None:
        raise Refused("--checkpoint is read by benchmarks/pouw/h1_prove/drawn.py's reader: run that CLI, or give --weights")
    return read(checkpoint, linears, k)


def b_value(win: dict, k: int, key: bytes, checkpoint: Path | None, weights: Path | None,
            read: CheckpointWeights | None = None) -> G.Registered:
    """``b/<K>`` as `served.register_weights` registered it: K's linears' rows in turn (`chunk_words`), at ``B_NONCE``
    plus K's place among the window's K's, under `key`, bound to no window."""
    lin = [tuple(x) for x in win["layout"]["linears"]]
    words = np.concatenate([SV.chunk_words(c, s, Wt.chunks(k)) for c, s in linear_weights(lin, k, checkpoint, weights, read)])
    ks = sorted({kk for _n, kk, _x in lin})
    return NR.commit(win["program"], f"{win['shape']['prefix']}b/{k}", words, key, SV.B_NONCE + ks.index(k),
                     keep_rows=True)


def weights_key(a: Any, win: dict) -> tuple[bytes, str]:
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


# -- staging --------------------------------------------------------------------------------------------------------------

def call_rows(win: dict, call: int, units: list[int], reg: dict, values: dict[str, G.Registered], pub: G.Registered,
              ok: G.Registered) -> dict[str, np.ndarray]:
    """The statement's rows of call `call`'s sampled `units` as `statement_tables` takes them (``w_``, ``s_``, and for a
    registered port ``r_``, ``v_``, ``p_``): each port of credit's file `reg` at the row it reads, from `values` (each
    value's rebuilt tree, the call's receipt table among them for o0), the public table `pub` and the ok rows `ok`."""
    ports = reg["ports"]
    u0 = ports[CR.PUB_PORT][G.READS_FROM]
    z: dict[str, Any] = {"units": np.array(units)}
    for port, e in ports.items():
        src = pub if port == CR.PUB_PORT else ok if port == CR.OK_PORT else values[e["value"]]
        at = [CC.port_reads(e)[u] for u in units]
        if any(x is None for x in at):
            raise Refused(f"port {port} reads no row at a drawn unit of call {call}")
        z[f"w_{port}"] = np.array([src.rows[x] for x in at], np.uint16)
        z[f"s_{port}"] = np.frombuffer(b"".join(bytes(src.salts[x]) for x in at), np.uint8).reshape(len(at), SALT_BYTES)
        z[f"r_{port}"] = np.array(at, np.int64)
        z[f"v_{port}"] = np.array([e["value"]] * len(at))
        z[f"p_{port}"] = np.array([src.path(x).hex() for x in at])
    z["u0"] = np.array(u0)
    return z


def unit_outputs(z: dict, win: dict, k: int, roots: tuple[bytes, bytes]) -> dict[str, np.ndarray]:
    """Each staged unit's outputs as J10-R's word reference computes them (`h1zy.reference`, chunked by
    `chunks_reference`) on its rows, the window's Program salt, its position, its call's roots and (u0, blocks): ``o0``
    its receipt's u16 words, ``o1`` its ok. Refused when a unit's ok is not 1 (its record is not its rows' product)."""
    lay, (r, c), x = win["layout"], tuple(win["layout"]["tile"]), Wt.chunks(k)
    u0, salt = int(z["u0"]), bytes.fromhex(win["salt"])
    n = next(nn for name, kk, nn in lay["linears"] if name == lay["calls"][int(z["call"])][0])
    o0, o1 = [], []
    for i, u in enumerate(int(v) for v in z["units"]):
        t = u - u0
        a = [split_row(z[f"w_a_k{k}_{q}"][i]) for q in range(x * r)]
        b = [split_row(z[f"w_b_k{k}_{q}"][i]) for q in range(x * c)]
        j10r = (u0, CC.j10r_blocks(lay["m"], k, (r, c), t // (n // c), t % (n // c)))
        args = ([v for v, _s in a], [v for v, _s in b], [s for _v, s in a], [s for _v, s in b],
                split_record(z["w_zy"][i], r * c), salt, u, roots)
        rc, ok = (Z.reference(*args, j10r=j10r) if x == 1 else Z.chunks_reference(*args, x, j10r=j10r))
        if ok != 1:
            raise Refused(f"unit {u}: the word reference's ok is {ok}, not 1: its record is not its rows' product, so it "
                          "is not proved")
        o0.append(receipt_words(rc))
        o1.append(OK)
    return {"o0": np.array(o0, np.uint16), "o1": np.array(o1, np.uint16)}


def reference_check(z: dict, outs: dict, win: dict, k: int, roots: tuple[bytes, bytes]) -> None:
    """The lowered circuit's outputs (``--recompose``) are the word reference's (`unit_outputs`), unit by unit."""
    want = unit_outputs(z, win, k, roots)
    for p in ("o0", "o1"):
        bad = np.flatnonzero((np.asarray(outs[p], np.uint16) != want[p]).any(axis=1))
        if bad.size:
            raise Refused(f"unit {int(z['units'][bad[0]])}: the lowered circuit's {p} is not the word reference's")


def call_receipts(z: dict, win: dict, k: int, roots: tuple[bytes, bytes],
                  outputs: Callable = unit_outputs) -> np.ndarray:
    """The receipts (u16 words, one row a unit) of the units of `z`'s rows: `unit_outputs`' o0, J10-R's word reference,
    whatever stands in for the staged units' outputs."""
    return outputs(z, win, k, roots)["o0"]


def receipt_table(win: dict, key: bytes, k: int, call: int, reg: dict, values: dict[str, G.Registered]) -> G.Registered:
    """Call `call`'s ``receipt@<c>`` as `Served.commit` registers it with receipts: one row a unit of the call, its
    receipt (`call_receipts` of the unit's A rows, B rows and record at the rows credit's file `reg` reads them, in
    `values`), at ``RECEIPT_NONCE`` + c under the prover's key `key`, bound to the window's root."""
    from verity_vllm.protocols.pouw.h1 import registered as NR
    from verity_vllm.protocols.pouw.h1 import served as SV
    size = CR.call_work(win["layout"])[call][0]
    u0 = reg["ports"][CR.PUB_PORT][G.READS_FROM]
    units = list(range(u0, u0 + size))
    z: dict[str, Any] = {"units": np.array(units), "u0": np.array(u0), "call": np.array(call)}
    for port, e in reg["ports"].items():
        if port not in (CR.PUB_PORT, CR.OK_PORT, CR.RECEIPT_PORT):
            z[f"w_{port}"] = np.array([values[e["value"]].rows[CC.port_reads(e)[u]] for u in units], np.uint16)
    roots = (bytes.fromhex(reg["ports"][f"a_k{k}_0"]["root"]), bytes.fromhex(reg["ports"][f"b_k{k}_0"]["root"]))
    e = reg["ports"][CR.RECEIPT_PORT]
    return NR.commit(win["program"], e["value"], call_receipts(z, win, k, roots), key, SV.RECEIPT_NONCE + call,
                     keep_rows=True, window=win["root"])


class FileStager:
    """`UnitStager`'s ``stage`` on the committed circuit file `path` of the statement of Definition id `definition` over
    the rows `in_rows` and `out_rows` (`UnitStager`'s), read and never composed (the K 2048 J10-R composition takes some
    hundreds of GB): its META (its second line), its pin and class digest (`circuit.digest`, `circuit.class_digest`)
    streamed from the file, and each instance's outputs the word reference's, which ``outs`` gives. `circuit.write` reads
    only these of a circuit; the files it writes are the recomposed stager's byte for byte. J10-R's statement's rows are
    `h1window.statement_rows` of its `template`; a lanes-form leaf's are `leaf_rows`' over its file's template (the
    manifest's ``templates``)."""

    def __init__(self, path: Path, definition: str, in_rows: Sequence[tuple], out_rows: Sequence[tuple],
                 name: str = NAME) -> None:
        from verity_flock import circuit as FC
        with open(path, "rb") as f:
            first, second = f.readline(), f.readline()
            meta = json.loads(second[5:])
            head = {k: v for k, v in meta.items() if k not in FC.BINDING_KEYS}
            h_pin = hashlib.sha512(first + second)
            h_cls = hashlib.sha512(first + b"META " + "".join(FC._COMPACT.iterencode(head)).encode() + b"\n")
            while b := f.read(1 << 26):
                h_pin.update(b)
                h_cls.update(b)
        self.meta, self.pin, self.class_sha512 = meta, h_pin.hexdigest(), h_cls.hexdigest()
        self.ports = FC._meta_ports(meta)
        self.row_ports = [r for r, *_ in in_rows]
        def port(nm: str) -> SimpleNamespace:
            return SimpleNamespace(name=nm, dtype="u16")
        content = json.dumps([definition, [list(r[:3]) + [list(r[3])] for r in in_rows],
                              [[s, list(ps)] for s, ps in out_rows]]).encode()
        outs = sorted({p for _s, ps in out_rows for p in ps})
        self.sub = SimpleNamespace(id=name, template=SimpleNamespace(id="unit-class"), inputs=[port(r) for r in self.row_ports],
                                   outputs=[port(f"o{j}") for j in outs])
        self.iset = SimpleNamespace(subcircuit=self.sub, name=name.replace("/", "-"),
                                    content_digest=hashlib.sha256(content).hexdigest())
        self.outs: dict[str, np.ndarray] | None = None

    def stage(self, rows: dict, out_dir: Path, salt_key: bytes, salts: dict, indices: Sequence[int], set_name: str,
              tables: dict, refs: dict, registered: dict) -> dict:
        from verity_flock import circuit as FC
        iset = SimpleNamespace(**{**vars(self.iset), "name": set_name})
        n = len(rows[self.row_ports[0]])
        for r in self.row_ports:
            if not np.array_equal(np.asarray(tables[r], np.uint16)[np.asarray(refs[r])], np.asarray(rows[r], np.uint16)):
                raise ValueError(f"port {r}'s table rows at the instances' refs are not their rows")
        table_rows = {p.name: len(tables[p.name]) if p.name in tables else n for p in self.ports}
        salts = {**FC.stage_salts(iset, 0, n, self.ports, salt_key, table_rows), **salts}
        out_dir.mkdir(parents=True, exist_ok=True)
        FC.write("", iset, 0, n, out_dir / "inst.bin", out_dir / "pub.bin", salt_key,
                 rows={r: np.asarray(rows[r], np.uint16) for r in self.row_ports}, outs=self.outs, salts=salts,
                 public_words={}, digests=(self.pin, self.class_sha512), indices=[int(u) for u in indices],
                 share_rows=True, tables=tables, refs=refs, registered=registered, meta=self.meta)
        return {"meta": self.meta, "outs": self.outs, "salts": salts, "circuit_sha512": self.pin}


def bind_files(d: Path, program: bytes, partition: dict) -> None:
    """A statement's ``program.json`` and ``partition.json`` in `d`: the Program and partition object credit verifies it
    under (`credit._asks`), which the ZK server's Lean ``Hello`` is set up on (``FC_FIREWALL_HELLO_ARGS``)."""
    (d / "program.json").write_bytes(program)
    (d / "partition.json").write_text(json.dumps(partition))


# -- the lanes form ---------------------------------------------------------------------------------------------------------

def leaf_stagers(k: int, tile: Sequence[int], circuits: Path, manifest: dict) -> dict[str, tuple[FileStager, Path, str]]:
    """Each of K's leaf statements' `FileStager` on --circuits' committed ``<kind><K>`` file, its path and its leaf
    Program's descriptor (hex): each file's SHA-512 the manifest's ``files`` and each descriptor's its ``leaf_programs``
    (what the period-open committed), its Definition id the manifest's ``templates``."""
    files, out = manifest.get("files", {}), {}
    for kind in (LL.LANE, LL.COMBINE):
        name = f"{kind}{k}"
        path = (circuits / f"{name}.circuit.txt").resolve()
        desc = circuits / LEAF_PROGRAM.format(name=name)
        tid = manifest.get("templates", {}).get(name)
        if name not in files or not path.is_file() or not desc.is_file() or tid is None:
            raise Refused(f"--circuits has no {name} file, leaf Program and template id: build them "
                          f"(pouw_h1_circuits --lanes {k})")
        raw = desc.read_bytes()
        if hashlib.sha512(raw).hexdigest() != files.get(CR.LEAF_PROGRAMS, {}).get(name):
            raise Refused(f"--circuits' {desc.name} is not the leaf Program the manifest commits ({CR.LEAF_PROGRAMS} {name})")
        st = FileStager(path, tid, leaf_rows(kind, k, tile), LEAF_OUT_ROWS, LEAF_NAMES[kind])
        if st.pin != files[name]:
            raise Refused(f"{name}'s circuit (SHA-512 {st.pin[:16]}...) is not --circuits' {name} file "
                          f"({files[name][:16]}...), the one the period-open committed")
        st.outs = {LL.OK: np.array([LL.OK_ROW], np.uint16)}
        out[kind] = (st, path, raw.hex())
    return out


def stage_leaves(a: Any, win: dict, k: int, units: list[int], values: dict[str, G.Registered], rebuild: Callable,
                 work: Path, circuits: Path, manifest: dict, plan: dict) -> list[G.Registered]:
    """K's sampled lanes-form leaves `units`, each staged as its lanes and the combine on K's committed files
    (`leaf_stagers`), into ``plan``'s ``leaves`` and ``leaf_programs``; their partials' trees, which the plan's produced
    record holds (`stage`). ``values``: the window's rebuilt values so far, which `rebuild` (value, call) adds to."""
    lay = win["layout"]
    why = CR.lanes_shape(lay, win["shape"], k)
    if why is not None:
        raise Refused(f"K = {k}'s drawn leaves: {why}")
    draw = win["draw"] if isinstance(win.get("draw"), dict) else {}
    draw_root = draw.get("root")
    if not isinstance(draw_root, str):
        raise Refused("the window's record names no draw root, under which a leaf's partials are committed")
    stagers = leaf_stagers(k, lay["tile"], circuits, manifest)
    plan["leaf_programs"].update({f"{kind}{k}": desc for kind, (_s, _p, desc) in stagers.items()})
    trees = []
    for u in units:
        call = CR.leaf_call(lay, u)[0]
        at = CR.leaf_place(lay, win["shape"], u, win["steps"])
        if len(win["roots"].get(at[LL.RECEIPT][0], ())) != 1:
            raise Refused(f"drawn leaf {u}: the window registers its call's {at[LL.RECEIPT][0]} "
                          f"{'twice' if win['roots'].get(at[LL.RECEIPT][0]) else 'not'}, which its combine reads: serve "
                          "the window with its zy receipts (served.Served.receipts) to prove it as lanes")
        for v in (LL.A, LL.B, LL.ZY, LL.RECEIPT):
            if at[v][0] not in values:
                values[at[v][0]] = rebuild(v, call)
        x = operands(win, u, values)
        name = LL.served(win["shape"]["prefix"], LL.PARTIALS, u)
        tree = PR.commit(win["program"], name, partials(x).tolist(), draw_root, schema=G.SCHEMA_V2)
        trees.append(tree)
        for st in LL.statements(*lay["tile"]):
            kind, ln = st
            rows, salts, registered, reg = leaf_statement(win, u, st, x, values, tree, {name: tree.root.hex()}, draw_root)
            stager, circuit, desc = stagers[kind]
            tag = f"u{u}-{kind}" + ("" if ln is None else str(ln))
            d = work / tag
            got = stager.stage(rows, d, _random(32), salts=salts, indices=[0], set_name=f"pouw-h1-w{a.window}-{tag}",
                               tables=rows, refs={p: [0] for p in rows}, registered=registered)
            (d / "pin.txt").write_text(stager.pin)
            (d / "circuit.txt").unlink(missing_ok=True)
            (d / "circuit.txt").symlink_to(circuit)
            (d / "registered.json").write_text(json.dumps(reg))
            leaf_program, unit_id = bytes.fromhex(desc), CR.leaf_unit_id(win["shape"], k, kind)
            if unit_id is None:
                raise Refused(f"K = {k}'s {kind} leaf: the window's template at K is not the zy unit, so credit names no "
                              "leaf partition to verify it under")
            bind_files(d, leaf_program, CR.leaf_partition_of(hashlib.sha512(leaf_program).hexdigest(), unit_id))
            plan["leaves"].append({"unit": u, "statement": [kind, ln], "k": k, "dir": str(d.resolve()), "instances": 1,
                                   "circuit": str(circuit), "pin": got["circuit_sha512"]})
        print(f"STAGED leaf {u}: its {LL.lanes(*lay['tile'])} lanes and combine", flush=True)
    return trees


def stage(a: Any) -> int:
    serve_dir, work = Path(a.serve_dir), Path(a.work)
    prover = Path(a.prover) if a.prover else serve_dir / PROVER.format(w=a.window)
    win = window_of(serve_dir, a.window)
    k_draw = draw_k(serve_dir, a.k)
    drawn = drawn_calls(win, k_draw)
    manifest = json.loads((Path(a.circuits) / MANIFEST).read_text())
    if manifest.get("program") != win["program"]:
        raise Refused(f"--circuits' manifest is for Program {str(manifest.get('program'))[:16]}..., not the window's "
                      f"{win['program'][:16]}...")
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
    plan = {"window": a.window, "draw_k": k_draw, "program": win["program"], "calls": [], "leaves": [],
            "leaf_programs": {}}
    produced: list[G.Registered] = []
    for k, calls in sorted(by_k.items()):
        b = None
        values: dict[str, G.Registered] = {}
        if CR.form_of(manifest.get("files", {}), k) == "lanes":
            def rebuild(v: str, call: int, k: int = k, values: dict = values) -> G.Registered:
                if v == LL.A:
                    return a_value(win, prover, pkey, call)
                if v == LL.B:
                    return b_value(win, k, bkey, a.checkpoint and Path(a.checkpoint), a.weights and Path(a.weights),
                                   a.checkpoint_weights)
                if v == LL.ZY:
                    return zy_value(win, prover, pkey, call)
                return receipt_value(win, pkey, call, values)
            produced += stage_leaves(a, win, k, sorted(u for c in calls for u in drawn[c]), values, rebuild, work,
                                     Path(a.circuits), manifest, plan)
            continue
        if not win["shape"]["tiles"][k]["j10r"]:
            raise Refused(f"K = {k}'s template is not J10-R's: drawn stages J10-R units on a k<K> file, and a v1 unit as "
                          f"its lanes (pouw_h1_circuits --lanes {k})")
        if f"k{k}" not in manifest.get("files", {}):
            raise Refused(f"--circuits has no k{k} file: build it (pouw_h1_circuits --strata {k})")
        circuit = (Path(a.circuits) / f"k{k}.circuit.txt").resolve()
        if a.recompose:
            built = work / f"k{k}.circuit.txt"
            st = stager(k, lay["tile"], lay["m"], built, manifest["bind"])
            got = PF.file_sha512(built)
            built.unlink()
        else:
            t = template(k, lay["tile"], lay["m"])
            st = FileStager(circuit, t.id, HW.statement_rows(t), HW.STATEMENT_OUT_ROWS)
            got = st.pin
        if got != manifest["files"][f"k{k}"]:
            raise Refused(f"K = {k}'s statement circuit (SHA-512 {got[:16]}...) is not --circuits' k{k} file "
                          f"({manifest['files'][f'k{k}'][:16]}...), the one the period-open committed")
        out_ports = [p.name for p in st.ports if p.name not in st.row_ports]
        regs = {}
        for c in calls:
            reg, why = CR.registered_file(win["program"], lay, win["shape"], c, win["roots"], win["root"], win["steps"],
                                          win["salt"])
            if why is not None:
                raise Refused(f"call {c}: credit builds no registered file: {why[1]}")
            complete([p.name for p in st.ports], reg, c)
            regs[c] = reg
            for port, e in reg["ports"].items():
                v, name = e["value"], e["value"][len(win["shape"]["prefix"]):]
                if port in (CR.PUB_PORT, CR.OK_PORT, CR.RECEIPT_PORT) or v in values:
                    continue
                if name == f"b/{k}":
                    b = b or held(b_value(win, k, bkey, a.checkpoint and Path(a.checkpoint), a.weights and Path(a.weights),
                                          a.checkpoint_weights), e, f"the served weights under {bwhose}")
                    values[v] = b
                elif name == f"zy@{c}":
                    values[v] = held(zy_value(win, prover, pkey, c), e, "the prover's records")
                elif name.startswith(f"a/{k}@"):
                    values[v] = held(a_value(win, prover, pkey, c), e, "the prover's A codes")
                else:
                    raise Refused(f"call {c}'s statement reads {v} at port {port}, which drawn doesn't rebuild")
            e = reg["ports"][CR.RECEIPT_PORT]
            values[e["value"]] = held(receipt_table(win, pkey, k, c, reg, values), e,
                                      "the word reference's receipts of the call's units")
        print(f"REBUILT K = {k}: {sorted(values)} at the window's roots", flush=True)
        for c in calls:
            reg, units = regs[c], drawn[c]
            pv = reg["ports"][CR.PUB_PORT]
            pub = held(G.Registered.commit(win["program"], pv["value"], pv["known"]["rows"],
                                           [bytes.fromhex(s) for s in pv["known"]["salts"]], G.WORD_BITS, G.SCHEMA_V2,
                                           window=win["root"]), pv, "credit's public table")
            z = call_rows(win, c, units, reg, values, pub, ok)
            z["call"] = np.array(c)
            tabs = statement_tables(z, st.row_ports, out_ports)
            roots = (bytes.fromhex(reg["ports"][f"a_k{k}_0"]["root"]), bytes.fromhex(reg["ports"][f"b_k{k}_0"]["root"]))
            if a.recompose:
                def lowered(_rows: dict, outs: dict, z: dict = z, roots: tuple = roots) -> None:
                    reference_check(z, outs, win, k, roots)
                    outputs_held(z, outs)
                st.reference_check = lowered
            else:
                st.outs = unit_outputs(z, win, k, roots)
                outputs_held(z, st.outs)
            d = work / f"c{c}"
            got = st.stage({p: z[f"w_{p}"] for p in st.row_ports}, d, _random(32), salts=tabs["salts"],
                           indices=units, set_name=f"pouw-h1-w{a.window}-c{c}", tables=tabs["tables"], refs=tabs["refs"],
                           registered=tabs["registered"])
            (d / "pin.txt").write_text(st.pin)
            (d / "circuit.txt").unlink(missing_ok=True)
            (d / "circuit.txt").symlink_to(circuit)
            (d / "registered.json").write_text(json.dumps(reg))
            bind_files(d, win["raw"], CR.partition_of(win["program"], win["shape"]))
            plan["calls"].append({"call": c, "k": k, "dir": str(d.resolve()), "units": units, "instances": len(units),
                                  "circuit": str(circuit), "pin": got["circuit_sha512"]})
            print(f"STAGED call {c}: units {units}", flush=True)
    if produced:
        coin = Ch.LiveOpening.from_json(win["draw"]["coin"]).coin.hex()
        plan["produced"] = PR.record(win["draw"]["root"], coin, [t.entry() for t in produced],
                                     {t.value: PR.rows_of(t) for t in produced})
    (work / PLAN).write_text(json.dumps(plan, indent=1))
    print("PLAN " + json.dumps({"calls": [(x["call"], x["instances"]) for x in plan["calls"]],
                                "leaves": sorted({x["unit"] for x in plan["leaves"]})}), flush=True)
    return 0


# -- statements and their check ---------------------------------------------------------------------------------------------

def statements(a: Any) -> int:
    plan = json.loads((Path(a.work) / PLAN).read_text())

    def session(x: dict, what: str) -> list[Path]:
        d = Path(x["dir"])
        sessions = sorted(p for p in (d / "sessions").glob("*/") if (p / "session.json").is_file())
        if len(sessions) != 1:
            raise Refused(f"{what}: {len(sessions)} sessions in {d}/sessions, not one: a drawn statement is proved once, "
                          "every instance drawn (session's subset:N)")
        return sessions

    out: Any = [PHC.statement(x["call"], x["circuit"], Path(x["dir"]) / "pub.bin", session(x, f"call {x['call']}"))
                for x in plan["calls"]]
    leaves = plan.get("leaves", [])
    for x in leaves:
        sessions = session(x, f"drawn leaf {x['unit']}'s statement {x['statement']}")
        out.append({"unit": int(x["unit"]), "statement": list(x["statement"]), "circuit": str(x["circuit"]),
                    "public": str(Path(x["dir"]) / "pub.bin"), "sessions": [str(s) for s in sessions]})
    if leaves:
        out = {"leaf_programs": plan["leaf_programs"], "produced": plan["produced"], "statements": out}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"STATEMENTS {len(plan['calls'])} calls, {len(leaves)} leaf statements -> {a.out}", flush=True)
    return 0


def check(a: Any) -> int:
    if a.verified is None:
        raise Refused("check is credit's own verification, which benchmarks/pouw/h1_prove/drawn.py binds: run that CLI")
    serve_dir = Path(a.serve_dir)
    win = window_of(serve_dir, a.window)
    drawn = drawn_calls(win, draw_k(serve_dir, a.k))
    units = sorted(u for us in drawn.values() for u in us)
    files = json.loads((Path(a.circuits) / MANIFEST).read_text())["files"]
    sts = json.loads(Path(a.statements).read_text())
    proofs, why = a.verified(sts, files, win, units, Path(a.lean))
    print(("PROVED " + json.dumps({"units": units, "outcomes": len(proofs)})) if why is None else
          f"REFUSED {why[0]}: {why[1]}", flush=True)
    return 0 if why is None else 1


def run(cmd: str, a: Any) -> int:
    """``stage``, ``statements`` or ``check`` on `a` (the CLI's arguments, with ``checkpoint_weights`` ``--checkpoint``'s
    reader and ``verified`` ``check``'s verifier, each None when not given): 0, or 1 with the refusal on stderr."""
    try:
        return {"stage": stage, "statements": statements, "check": check}[cmd](a)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr, flush=True)
        return 1


__all__ = ["CheckpointWeights", "MANIFEST", "PLAN", "Refused", "Verified", "run"]
