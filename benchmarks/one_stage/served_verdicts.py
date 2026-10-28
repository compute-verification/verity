"""The verdicts on one served window (`served_verdicts.sh`): the window it audits, the subprotocols after
sampled proofs, and the one record of every verdict (the block record).

    served_verdicts.py window SERVE_DIR [--slots N] [--window K]
    served_verdicts.py gate [--gate CMD] [--lane vllm] [--main FILE]
    served_verdicts.py period open SERVE_DIR --name P --circuits DIR --slots N --max-tokens N [--gate CMD] [--draw-k K]
                                 [--model M]
    served_verdicts.py period close SERVE_DIR [--gate CMD]
    served_verdicts.py program OUT [--model M] [--rows T] [--steps S] [--like SERVE_DIR]
    served_verdicts.py pieces --serve-run DIR --window K --out DIR [--node NAME] [--only COMPONENT]... [--skip COMPONENT]...
    served_verdicts.py pouw SERVE_DIR WINDOW DIR [--gate CMD] [--key-sha512 HEX] [--main FILE] [--python PY]
                                [--statements FILE] [--live-evidence FILE]
    served_verdicts.py record --serve-run DIR --window K --sampled-proofs DIR --pieces DIR --out DIR [--node NAME]
    served_verdicts.py public-registration REGISTRATION
    served_verdicts.py sampled-line LOG [--partition FILE]
    served_verdicts.py front-line SESSION

``public-registration`` prints what the run publishes of a window's advice registration (`public_registration`: the
advice's length padded to its compiler's cap), ``sampled-line`` the SAMPLED-PROOFS line from the records audit's log
(`sampled_line`: the partition the draw was over, the members it drew, and no length of the advice or count of the
scope's members), and ``front-line`` the CERTIFIER-FRONT line from the front's session (`front_line`: `FRONT_PUBLIC`).

``window``: the sealed window to audit, from a serve-openai run's ``seals.jsonl``, ``windows.jsonl`` and ``requests.jsonl``
(`pick`): sealed without error, its Commit passed, each of its requests checked and in no other window (V3b's window
circuit refuses a request carried across windows), at most ``--slots`` of them (its class's); the most requests, then the
earliest. ``--window K`` holds K to the same rules. Prints what the run publishes of it (`public_pick`):
``{window, steps, engine_steps, slots, every_slot, refused}``, ``steps`` the window's ``[lo, hi)`` in the run's committed
engine steps, ``every_slot`` whether its requests filled the class's slots, ``refused`` the windows refused, by number. Its
requests' count, list and users stay on the node.

``pieces``: each subprotocol after sampled proofs (`PIECES`: PoUW, the bandwidth audit, PoUS, the network certifier) runs
its piece where one is named (``POUW_CMD``, ``BANDWIDTH_CMD``, ``POUS_CMD``, ``CERTIFIER_CMD``: a command run in the tree as
``bash -c CMD _ SERVE_DIR WINDOW DIR``, which writes its core answer to ``DIR/answer.json``), one at a time, each within
``<COMPONENT>_TIMEOUT`` (else ``PIECE_TIMEOUT``, default `PIECE_TIMEOUT`), else prints a ``MISSING`` line naming it, the
command it lacks and its owner. Each run leaves ``DIR/piece.json`` (`RAN`: the node it ran on, its exit, its limit). A piece
that ran but wrote no answer is its error, unless it refused to run (``DIR/refused.txt``, `REFUSED`, naming why), which
leaves it missing; no verdict is made for a piece that didn't run. A development piece writes ``DIR/development.json``
(`DEVELOPMENT`) in place of a core answer: its verdict is printed as a development run's, and it credits nothing. A piece
that audited another server than the node it ran on says so in ``DIR/served.json`` (`SERVED`: ``{server, source}``,
``source`` ``fixture`` for an answer brought from a recorded run); one that names no server there leaves its answer about
the node it ran on. ``--only`` and ``--skip`` (a component each, repeatable) run those pieces alone, or all but those.

``gate``: before serving with H1's coins from infra's gate (``--gate``, default ``H1_GATE``): refused while the key's lane
(``--lane``) has a period open, which a reserve would join; prints one ``GATE`` line with the gate's latest pin and,
with ``--main``, whether that pin is on the verifier's first-parent line of main (a session pinned off it is refused).

``period open``: the period the served windows join, opened once serve-openai listens and before its first window
reserves (a window opens at its first engine step): its circuit is the Program digest serve-openai declared and wrote
(serve.json's ``protocols.program``, ``h1/window-program.meta.json``), its rows and rounds the server's T and S, and its
circuit files those of ``--circuits`` (``pouw_h1_circuits --program SERVE_DIR/h1 --out DIR --live T,S``'s manifest),
held to that digest, T and S (`period_open_argv`). The live template's file is required. A stratum whose ``k<K>`` file
isn't built yet is left out and named: core's credit refuses a drawn statement at that K (``circuit``), nothing else.
Its fixed length is ``--slots`` requests of ``--max-tokens`` (`credit.LENGTH_FIELD`); its draw K ``--draw-k``, else
`pouw_h1_challenger.draw_k` over the declared layout; its weights the auditor's registration of ``--model`` under the
digest, else none (the gate refuses a root its registry lacks, and serve-openai's own is a local key's, so core's credit
then refuses the window at its weights). The open's answer goes
to ``SERVE_DIR/period.json``. ``period close`` closes that period and adds its answer there.

``program``: serve-openai's declared window Program for ``--model`` at T, S, built on a CPU without serving (`program`,
`DECLARED`) and written as serve-openai writes ``h1/``, so the circuit files can be built before the serve that needs
them (the Program is a function of the linears, T and S, and of the pouw settings ``unit`` and ``ks`` in
``--protocols-config``, which must be serve-openai's own). ``--like`` takes model, T, S and that config from a
serve-openai run and refuses unless the digest is that run's.

``pouw``: PoUW's development piece (``POUW_CMD``): window K's H1 outcome (``windows.jsonl``'s ``pouw_h1``) and record
(``h1/w<K>/pouw-h1-calls.json``); with the auditor host's challenger's coins, the window's session as the gate's ``fetch``
gives it, the auditor's ledger and pins log as the gate answers now and main's first-parent history (``--main``),
checked by `pouw_h1_gpu.check_challenger` under the key the deployment pins (``--key-sha512``). Its verdict is accept when
H1's tables are tied to the window's Commit, its coins are the session the ledger opened and ended for it, and its record
is one manifest under them; core's credit refuses the window and is shown beside it (`DEV_WHY`). Credit reads the
declared Program serve-openai wrote (``h1/window-program.json.gz``), the period ``period open`` opened, and, when given,
the drawn calls' statements (``--statements``) and the window's live evidence (``--live-evidence``,
``h1_prove/live.py evidence``'s), so it reaches each check those reach (`credit_inputs`). The verdict says whether the
window meets the two premises of `Security`'s that no check here discharges, ``AcceptedLive`` and ``SaltIsLiveCoin``
(`premises`): each ``{met, why}``, met only where this run's checks show it.

``record``: sampled proofs' answer from the records audit `prove_path/run.sh` (``PP_RECORDS=1``) kept under
``--sampled-proofs`` (its ``NAME`` dir: ``summary.json``, ``window/audit.json``, ``windows/rank0/registration.json``), read
as `served_integrity.answer` reads it; each other subprotocol's from
``--pieces/<component>/answer.json`` (`piece_row`), else a ``missing`` row naming why. Each answer is about the server
`piece_server` names: sampled proofs' the record's node, a piece's the one its ``served.json`` names, else the node it ran
on, and the wipe's the one its ``served.json`` names. They go into the window report (``verdicts.json``, `row_of`'s rows,
with each one's server) and from it the block record (``block-record.json``, `block_record.from_period`, held by
`block_record.check`), which composes only the answers about the record's node and carries the others as ``elsewhere``;
an answer that names no server isn't carried. An accepted answer whose outputs are not its leg's certificate
(`block_record.leg`) is its subprotocol's error. `block_record` carries only an accepted subprotocol's certificate, so
where one about this node rejects or errs the composed verdict is ``reject`` and no block record is written; it is
``accept`` only when every subprotocol accepted here.
The report's ``composed`` and the block record's ``composed.onsite`` give that verdict with its reason (`said`), one clause
a subprotocol: its verdict and, unless it accepted here, why; an accept here with what it holds beyond its verdict
(`terms`: when its weights were held and their share of the checkpoint, where its profile says, and each δ of the block
record's at which its quantity is not established); PoUW's development piece reads ``development run, credits nothing``
with core's refusal of the window's credit. Prints one ``RECORD`` line, naming each subprotocol's verdict (the
development piece's `DEV_VERDICT`), each missing one and why, each rejecting one, each one elsewhere, and the reason.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import tarfile
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for _p in (Path(__file__).resolve().parent, ROOT / "examples" / "h200" / "interfaces"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

#: the window report's format, kept from the all-protocols runner whose report it was
REPORT_FORMAT = "verity/all-protocols-window/v1"
#: sampled proofs' owner, its row's
INTEGRITY_OWNER = "compute-accounting"

#: each subprotocol after sampled proofs, by its runner name: (its name, its component, its verifier's owner, what its piece
#: does), in the chain's order
PIECES: dict[str, tuple[str, str, str, str]] = {
    "work": ("PoUW", "pouw", "compute-accounting",
             "H1's credit on the served window, its coins from vy-challenger-1's gate (H1_GATE)"),
    "bandwidth": ("the bandwidth audit", "bandwidth", "memory-accounting",
                  ("a remote bandwidth session over this node's registered store, against the auditor's verifier on "
                   "vy-auditor-2; served_verdicts.sh sets it where the tree has benchmarks/one_stage/served_bandwidth.sh "
                   "and this node the verifier's key and the store")),
    "space": ("PoUS", "pous", "memory-accounting",
              ("PoUS over the served model's int8 weights on the serving card, against the memory challenger on "
               "vy-auditor-2; served_verdicts.sh sets it where the tree has benchmarks/pous/served_piece.sh and this node "
               "the challenger's key")),
    "communication": ("the network certifier", "certifier", "network-accounting",
                      "the network certifier in front of the serving endpoint"),
}
#: the record's subprotocol outside this chain: serve() runs its piece (WIPE_CMD) on the card before its engine loads
WIPE = ("the wipe", "wipe", "memory-accounting", "the wipe of the serving card before its engine loads")
NAMES = {"integrity": "sampled proofs", **{k: v[0] for k, v in PIECES.items()}, "wipe": WIPE[0]}
#: a piece's limit unless ``<COMPONENT>_TIMEOUT`` or ``PIECE_TIMEOUT`` sets one, as `timeout` reads it
PIECE_TIMEOUT = "60m"
#: what `pieces` leaves beside a piece it ran: the node it ran on, its exit and its limit
RAN = "piece.json"
#: a piece's refusal to run, naming why: it leaves its subprotocol missing
REFUSED = "refused.txt"
#: the server a piece's answer is about, where it isn't the node `pieces` ran it on (the wipe's always)
SERVED = "served.json"
#: a development piece's document, written in place of its core answer
DEVELOPMENT = "development.json"
#: why a development piece's verdict credits nothing, so its subprotocol's row in the record stays missing
DEV_WHY = ("development run, credits nothing: H1's coins come from vy-challenger-1's gate under the lane's development key, "
           "and a window counts for credit only under the lane's own key; core's credit also needs the window in a period "
           "whose open commits its draw K, rows and weights")
#: a development piece's verdict in the ``RECORD`` line, in place of its row's ``missing``
DEV_VERDICT = "development run, credits nothing"
#: H1's record of a window (`pouw_h1.CALLS_FILE`) under the serve dir
H1_CALLS = "h1/w{k}/pouw-h1-calls.json"
#: serve-openai's declared window Program's meta under the serve dir (`pouw_h1.PROGRAM_META`)
H1_META = "h1/window-program.meta.json"
#: `pouw_h1_circuits`' manifest in its ``--out``
CIRCUITS_MANIFEST = "circuit-files.json"
#: the period this serve dir's windows joined, its open's and close's answers (``period``)
PERIOD = "period.json"
#: the int8 linears serve-openai declares its window Program over, by served model: its layers and each layer's
#: ``(module, n, K)`` in vLLM's module order (serve-openai's H1 arm: each ``weight.shape``, named by `pouw._named`)
DECLARED: dict[str, tuple[int, tuple[tuple[str, int, int], ...]]] = {
    "verity-int8/Llama-3.2-1B-INT8": (16, (("self_attn.qkv_proj", 3072, 2048), ("self_attn.o_proj", 2048, 2048),
                                           ("mlp.gate_up_proj", 16384, 2048), ("mlp.down_proj", 2048, 8192))),
}


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def _refusal(k: int, seal: Mapping[str, Any], line: Mapping[str, Any] | None, mine: Sequence[Mapping[str, Any]],
             slots: int) -> str | None:
    """Why window ``k`` can't be audited, None when it can."""
    if "error" in seal:
        return f"its seal failed: {seal['error']}"[:300]
    if not line or not line.get("passed"):
        return "its Commit did not pass"
    if not mine:
        return "no request"
    carried = sorted(r.get("user") or r.get("engine_id") for r in mine if r.get("windows") != [k])
    if carried:
        return f"requests carried across windows: {carried}"
    unchecked = sorted(r.get("user") or r.get("engine_id") for r in mine if not (r.get("check") or {}).get("passed"))
    if unchecked:
        return f"requests whose check failed: {unchecked}"
    if len(seal.get("requests") or ()) != len(mine):
        return f"{len(seal.get('requests') or ())} sealed requests, {len(mine)} with a record"
    if len(mine) > slots:
        return f"{len(mine)} requests, more than the class's {slots} slots"
    return None


def pick(serve_dir: Path, slots: int, want: int | None = None) -> dict[str, Any]:
    """The window to audit (module docstring); ValueError when none qualifies, or ``want`` doesn't."""
    seals = {int(s["window"]): s for s in _lines(serve_dir / "seals.jsonl")}
    lines = {int(w["window"]): w for w in _lines(serve_dir / "windows.jsonl")}
    reqs = _lines(serve_dir / "requests.jsonl")
    refused: dict[int, str] = {}
    ok: list[tuple[int, list[dict[str, Any]]]] = []
    for k in sorted(seals):
        mine = [r for r in reqs if k in (r.get("windows") or ())]
        why = _refusal(k, seals[k], lines.get(k), mine, slots)
        if why is None:
            ok.append((k, mine))
        else:
            refused[k] = why
    if want is not None:
        if want not in seals:
            raise ValueError(f"window {want} has no seal (sealed: {sorted(seals)})")
        if want in refused:
            raise ValueError(f"window {want}: {refused[want]}")
        ok = [x for x in ok if x[0] == want]
    if not ok:
        raise ValueError(f"no sealed window to audit: {json.dumps(refused, sort_keys=True)}")
    k, mine = max(ok, key=lambda x: (len(x[1]), -x[0]))
    lo = sum(int((w.get("commit") or {}).get("steps") or 0) for j, w in lines.items() if j < k)
    hi = lo + int((lines[k].get("commit") or {}).get("steps") or 0)
    return {"window": k, "requests": len(mine), "users": sorted(r.get("user") or "" for r in mine), "steps": [lo, hi],
            "refused": {str(j): why for j, why in refused.items()}}


def public_pick(got: Mapping[str, Any], slots: int) -> dict[str, Any]:
    """What the run publishes of `pick`'s window: its engine steps and whether its requests filled the class's ``slots``,
    with no count, list or user of them, and the windows refused by number alone."""
    lo, hi = got["steps"]
    return {"window": got["window"], "steps": [lo, hi], "engine_steps": hi - lo, "slots": slots,
            "every_slot": got["requests"] == slots, "refused": sorted(int(j) for j in got["refused"])}


#: The advice registration's fields that state the advice's own length, which `public_registration` pads to the cap.
PADDED = ("binding.advice.bits", "rows", "bits_by_window")


def public_registration(reg: Mapping[str, Any]) -> dict[str, Any]:
    """What the run publishes of a window's advice registration: the advice's length (`PADDED`) at its compiler's
    ``cap_bits``, the one length the class declares (its rows at ``row_bits`` each, every window's bits the cap), no count
    or refusal of its requests, and the rest as registered. ``padded`` names the fields padded."""
    cap, row_bits = int(reg["compiler"]["cap_bits"]), int(reg["row_bits"])
    out = {k: v for k, v in reg.items() if k not in ("requests", "compiled", "refused")}
    out["binding"] = {**reg["binding"], "advice": {**reg["binding"]["advice"], "bits": cap}}
    out["rows"] = -(-cap // row_bits)
    out["bits_by_window"] = {w: cap for w in reg.get("bits_by_window") or {}}
    return {**out, "padded": list(PADDED)}


def sampled_line(records: Mapping[str, Any], partition: Mapping[str, Any] | None) -> str:
    """The SAMPLED-PROOFS line's account of the records audit's ``RECORDS`` row: what the draw was over (the partition's
    templates, scope and law), which members it drew, and the row with no length of the advice (``leak.advice_bits``)
    and no ``population``, the scope's unpadded member count, which counts the window's tokens."""
    row = {k: v for k, v in records.items() if k != "population"}
    if isinstance(row.get("leak"), Mapping):
        row["leak"] = {k: v for k, v in row["leak"].items() if k != "advice_bits"}
    part = partition or {}
    scope = part.get("scope") or {}
    over = (f"over {', '.join(part.get('templates') or ['?'])} at {scope.get('op_path_prefix', '?')} steps "
            f"{scope.get('steps', '?')} (law {part.get('law', '?')})")
    drawn = f"drawn members {row.get('draw')}" if row.get("draw") is not None else "no draw"
    return f"{over}, {drawn}: RECORDS {json.dumps(row, sort_keys=True)}"


#: The certifier front's session fields the CERTIFIER-FRONT line says. The rest (its prompts, request, kept fetch, gate
#: and verdict's cells) count or name the group's requests, and stay in the serve dir.
FRONT_PUBLIC = ("rc", "run", "ended", "ended_utc", "session_s", "pinned", "leaked_removed")


def front_line(session: Mapping[str, Any]) -> str:
    """The CERTIFIER-FRONT line's account of the front's ``session.json``: `FRONT_PUBLIC` alone, or that it refused."""
    if "refused" in session:
        return "refused (its reason in the serve dir's private/certifier-front/session.json)"
    return json.dumps({k: session[k] for k in FRONT_PUBLIC if k in session})


def _unset(comp: str, what: str) -> str:
    return f"{comp.upper()}_CMD unset: {what}"


def pieces(serve_dir: Path, window: int, out: Path, env: Mapping[str, str] | None = None,
           node: str | None = None, only: Sequence[str] | None = None, skip: Sequence[str] = ()) -> dict[str, dict[str, Any]]:
    """Each piece of `PIECES` run where ``<COMPONENT>_CMD`` names it, under `timeout` at its limit, else named missing;
    one line each on stdout, its verdict the one `piece` gives the record. ``only`` and ``skip`` name components: those
    alone, or all but those, so one piece can run beside the others (`served_verdicts.sh` starts the certifier's early)."""
    env = dict(os.environ if env is None else env)
    node = node or socket.gethostname()
    got: dict[str, dict[str, Any]] = {}
    for key, (name, comp, owner, what) in PIECES.items():
        if (only is not None and comp not in only) or comp in skip:
            continue
        cmd = env.get(f"{comp.upper()}_CMD")
        if not cmd:
            got[comp] = {"verdict": "missing", "why": _unset(comp, what)}
            print(f"MISSING {name} ({comp.upper()}_CMD unset; {owner}): {what}", flush=True)
            continue
        d = out / comp
        d.mkdir(parents=True, exist_ok=True)
        limit = env.get(f"{comp.upper()}_TIMEOUT") or env.get("PIECE_TIMEOUT") or PIECE_TIMEOUT
        with open(d / "piece.log", "w") as log:
            rc = subprocess.run(["timeout", "-k", "1m", limit, "bash", "-c", cmd, "_", str(serve_dir), str(window),
                                 str(d)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, check=False).returncode
        (d / RAN).write_text(json.dumps({"server": node, "rc": rc, "limit": limit}, sort_keys=True) + "\n")
        label, (row, _) = name.upper(), piece(key, out)
        if "development" in row:
            got[comp], label = {**development_line(row["development"]), "rc": rc}, f"{label} (development run, credits nothing)"
        else:
            got[comp] = {**{k: row[k] for k in ("verdict", "reason", "why") if row.get(k) is not None}, "rc": rc}
        print(f"{label} {json.dumps(got[comp], sort_keys=True)}", flush=True)
    return got


def development_line(doc: Mapping[str, Any]) -> dict[str, Any]:
    """A development piece's line (`pouw`'s document): its verdict, why it rejects, the window's session and how each
    check came out, core's credit refusal, and that it credits nothing."""
    check, credit = doc.get("check") or {}, doc.get("credit") or {}
    return {"verdict": doc.get("verdict"), "development": True, "credited": False, "reasons": doc.get("reasons"),
            "coin_source": doc.get("coin_source"), "session": (doc.get("session") or {}).get("session"),
            "tied": doc.get("tied"), "coins": check.get("reason", "unchecked"), "manifest": check.get("manifest", "unchecked"),
            "gate_passed": doc.get("gate_passed"), "credit": credit.get("window"),
            **{p: doc.get(p) for p in PREMISES}}


def _gate_out(gate: str, *words: str) -> bytes:
    """The gate's answer to `words` through its command `gate`, the words one shell-quoted line as `commit_coins.ChallengerGate`
    sends them; RuntimeError naming what it said when it refuses."""
    p = subprocess.run([*shlex.split(gate), shlex.join(words)], capture_output=True, check=False)
    if p.returncode:
        said = (p.stdout + p.stderr).decode(errors="replace").strip()
        raise RuntimeError(f"the gate refused {words[0]} (exit {p.returncode}): {said[-400:]}")
    return p.stdout


def gate_ready(gate: str, lane: str, main: Path | None = None) -> tuple[bool, str]:
    """Whether this run may reserve its windows' sessions at `gate` (the ``gate`` subcommand), and its ``GATE`` line."""
    from verity_vllm.protocols.pouw.h1 import window as PHW

    ledger = _gate_out(gate, "ledger").decode().splitlines()
    pins = [json.loads(x) for x in _gate_out(gate, "pins").decode().splitlines() if x.strip()]
    pin, period = (pins[-1] if pins else {}), PHW.open_period(ledger, lane)
    on = None if main is None else pin.get("commit") in main.read_text().split()
    where = "" if on is None else f", {'on' if on else 'NOT on'} the verifier's first-parent line of main"
    line = (f"GATE {'refused' if period else 'ready'}: lane {lane}'s open period "
            f"{period and period.get('period')}{' (a reserve would join it)' if period else ''}; the ledger's "
            f"{len(ledger)} entries; pinned {str(pin.get('commit'))[:12]} (main {str(pin.get('main'))[:12]}, "
            f"{pin.get('pinned_utc')}){where}")
    return period is None, line


def program(out: Path, model: str, rows: int, steps: int, like: Path | None = None,
            config: str | None = None) -> dict[str, Any]:
    """``program``: the window Program serve-openai declares for `model` at T `rows` and S `steps` under `config`, its
    ``--protocols-config`` (`serve_h1.config_of`'s, a JSON object or file), built without serving (`DECLARED`'s linears
    at the pouw setting ``ks``'s K's, over the setting ``unit``'s tile unit, `served.unit_of`) and written as
    serve-openai writes it (`served.write_program`) into `out`; its meta. With `like`, a serve-openai run's dir, its
    model, T, S and config are that run's (serve.json's ``protocols``) and its digest must be the run's
    (``protocols.program``), else ValueError."""
    from verity_vllm.pipeline import serve_h1 as SH
    from verity_vllm.protocols import pouw as PW
    from verity_vllm.protocols.pouw import h1 as PH
    from verity_vllm.protocols.pouw.h1 import served as SV

    want, cfg = None, SH.config_of(config)
    if like is not None:
        serve = json.loads((like / "serve.json").read_text())
        proto = serve.get("protocols") or {}
        model, rows, steps, want = serve.get("served_model_name") or model, proto["rows"], proto["steps"], proto["program"]
        cfg = proto.get("config") or {}
    if model not in DECLARED:
        raise ValueError(f"no declared linears for {model!r}: add them to DECLARED ({sorted(DECLARED)})")
    pw = PW.Settings.of(cfg.get("pouw", {}))
    layers, linears = DECLARED[model]
    named = [(f"model.layers.{i}.{m}", n, k) for i in range(layers) for m, n, k in linears if pw.ks is None or k in pw.ks]
    if not named:
        raise ValueError(f"{model} has no linear at K {list(pw.ks or ())} (the pouw setting 'ks'): its K's are "
                         f"{sorted({k for _m, _n, k in linears})}")
    shapes = [(n, k) for _m, n, k in named]
    modules = {w: m for w, (m, _n, _k) in enumerate(named)}
    digest = SV.write_program(SV.declared(shapes, modules, rows, steps, unit=SV.unit_of(pw.unit)),
                              SV.declared_layout(shapes, modules, rows, steps), out, rows=rows, steps=steps)
    if want is not None and digest != want:
        raise ValueError(f"{model}'s Program at T {rows}, S {steps} is {digest[:16]}..., not {like}'s {str(want)[:16]}...: "
                         "DECLARED's linears aren't the ones that run served")
    return json.loads((out / PH.PROGRAM_META).read_text())


def period_open_argv(serve_dir: Path, name: str, circuits: Path, *, slots: int, max_tokens: int,
                     draw_k: int | None = None, model: str | None = None) -> tuple[list[str], dict[str, Any]]:
    """The gate's ``period-open`` of period `name` for serve-openai's run `serve_dir` (``period open``, module docstring)
    and what it commits: ``{circuit, rows, steps, draw_k, circuit_files, missing, length, weights}``, ``missing`` the
    strata with no ``k<K>`` file yet. ValueError naming what to build or rerun when the serve dir or `circuits` doesn't
    hold the period's Program, T and S. Before serve-openai starts (no ``serve.json`` yet) the Program is ``program``'s,
    and `period_served` holds the one it then declares to it."""
    from verity_vllm.protocols.pouw.h1 import challenger as PHC

    served = serve_dir / "serve.json"
    proto = (json.loads(served.read_text()).get("protocols") or {}) if served.is_file() else None
    try:
        meta = json.loads((serve_dir / H1_META).read_text())
    except (OSError, ValueError) as e:
        raise ValueError(f"{serve_dir} has no declared window Program ({H1_META}, {type(e).__name__}): serve it with "
                         "--protocols pouw:eps1-sm120-h1fx1 from a tree whose serve_h1 writes it") from None
    circuit, rows, steps = meta["program_digest"], int(meta["rows"]), int(meta["steps"])
    if proto is not None and proto.get("program") != circuit:
        raise ValueError(f"serve.json's Program {str(proto.get('program'))[:16]}... is not {H1_META}'s {circuit[:16]}...")
    redo = (f"build them with python -m tests.protocols.pouw_h1_circuits --program {serve_dir / 'h1'} --out {circuits} "
            f"--live {rows},{steps}")
    try:
        doc = json.loads((circuits / CIRCUITS_MANIFEST).read_text())
    except (OSError, ValueError) as e:
        raise ValueError(f"no circuit files at {circuits} ({type(e).__name__}): {redo}") from None
    if doc.get("program") != circuit:
        raise ValueError(f"{circuits}'s circuit files are Program {str(doc.get('program'))[:16]}...'s, not the served "
                         f"{circuit[:16]}...: {redo}")
    lb, files = doc.get("live_bind") or {}, dict(doc.get("files") or {})
    if (lb.get("T"), lb.get("S")) != (rows, steps) or "live" not in files:
        raise ValueError(f"{circuits}'s live circuit file is for T = {lb.get('T')}, S = {lb.get('S')}, not the window's "
                         f"{rows}, {steps}: {redo}")
    strata = sorted({int(k) for _n, k, _w in meta["layout"]["linears"]})
    missing = [k for k in strata if f"k{k}" not in files]
    k = PHC.draw_k(meta["layout"]) if draw_k is None else int(draw_k)
    length = {"max_tokens": [int(max_tokens)] * int(slots), "stop": "fixed"}
    weights = ["--model", model] if model else []
    compact = {"sort_keys": True, "separators": (",", ":")}
    argv = ["period-open", "--name", name, "--cap", "1", "--circuit", circuit, "--draw-k", str(k), "--rows", str(rows),
            *weights, "--circuit-files", json.dumps(files, **compact), "--length", json.dumps(length, **compact)]
    return argv, {"circuit": circuit, "rows": rows, "steps": steps, "draw_k": k, "circuit_files": files,
                  "missing": missing, "length": length, "weights": weights}


def _gate_json(gate: str, argv: Sequence[str]) -> dict[str, Any]:
    """The gate's JSON answer to `argv`; RuntimeError naming what it said when it refuses."""
    raw = _gate_out(gate, *argv)
    said = raw.decode(errors="replace").strip()[-600:]
    try:
        answer = json.loads(raw)
    except ValueError as e:
        raise RuntimeError(f"the gate's answer to {argv[0]} is not JSON ({e}): {said}") from None
    if not isinstance(answer, dict) or "refused" in answer:
        raise RuntimeError(f"the gate refused {argv[0]}: {said}")
    return answer


def period_open(serve_dir: Path, gate: str, name: str, circuits: Path, **kw: Any) -> dict[str, Any]:
    """``period open`` (module docstring): the open sent, its answer and what it committed, written to
    ``serve_dir/PERIOD``."""
    argv, doc = period_open_argv(serve_dir, name, circuits, **kw)
    doc = {"name": name, **doc, "open": _gate_json(gate, argv)}
    (serve_dir / PERIOD).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc


def period_served(serve_dir: Path) -> dict[str, Any]:
    """``period served``: the period opened before serve-openai started, whose Program serve-openai must then have
    declared (serve.json's ``protocols.program``); ValueError naming both when it declared another, since credit refuses
    every window of a period whose open committed another Program."""
    doc = json.loads((serve_dir / PERIOD).read_text())
    got = (json.loads((serve_dir / "serve.json").read_text()).get("protocols") or {}).get("program")
    if got != doc["circuit"]:
        raise ValueError(f"serve-openai declared Program {str(got)[:16]}..., not period {doc['name']}'s "
                         f"{doc['circuit'][:16]}... (`program`'s, built from DECLARED): its windows credit nothing")
    return doc


def period_close(serve_dir: Path, gate: str) -> dict[str, Any]:
    """``period close``: the period `period_open` opened for `serve_dir` closed, its answer added to ``PERIOD``."""
    doc = json.loads((serve_dir / PERIOD).read_text())
    doc["close"] = _gate_json(gate, ["period-close", "--name", doc["name"]])
    (serve_dir / PERIOD).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc


def credit_inputs(serve_dir: Path, statements: Path | None = None, live: Path | None = None) -> dict[str, str]:
    """What core's credit reads of the window beyond its record and session, as ``--check-challenger``'s flags: the
    declared Program serve-openai wrote (``--descriptor``, `pouw_h1.PROGRAM`), the period ``period open`` opened
    (``--period``, `PERIOD`), the drawn calls' statements (``--statements``) and the live evidence (``--live-evidence``,
    ``h1_prove/live.py evidence``'s), each where it exists."""
    got: dict[str, str] = {}
    if (p := serve_dir / "h1" / "window-program.json.gz").is_file():
        got["--descriptor"] = str(p)
    if (p := serve_dir / PERIOD).is_file():
        got["--period"] = json.loads(p.read_text())["name"]
    for flag, p in (("--statements", statements), ("--live-evidence", live)):
        if p is not None:
            got[flag] = str(p)
    return got


def check_challenger(calls: Path, tree: Path, ledger: Path, pins: Path, main: Path | None, key: str, out: Path,
                     inputs: Mapping[str, str], python: str) -> dict[str, Any]:
    """`pouw_h1_gpu.py --check-challenger` (run by `python` in integrations/vllm, so ``tests`` is that package's) of the
    window's record `calls` against its session's tree `tree`, with credit's `inputs` (`credit_inputs`); its
    ``pouw_h1_challenger.json``."""
    out.mkdir(parents=True, exist_ok=True)
    argv = [python, "-m", "tests.protocols.pouw_h1_gpu", "--check-challenger", str(calls), str(tree), "--key-sha512", key,
            "--ledger", str(ledger), "--pins", str(pins), *(["--main", str(main)] if main else []), "--out", str(out),
            *(x for kv in inputs.items() for x in kv)]
    p = subprocess.run(argv, cwd=ROOT / "integrations" / "vllm", capture_output=True, text=True, check=False)
    (out / "check.log").write_text(p.stdout + p.stderr)
    try:
        return json.loads((out / "pouw_h1_challenger.json").read_text())
    except (OSError, ValueError) as e:
        raise RuntimeError(f"pouw_h1_gpu --check-challenger wrote no verdict (exit {p.returncode}): "
                           f"{(p.stdout + p.stderr).strip()[-400:]}") from e


Check = Callable[[Path, Path, Path, Path, "Path | None", str, Path, Mapping[str, str]], Mapping[str, Any]]

#: the premises the pouw piece's verdict says whether the window meets (`premises`)
PREMISES = ("AcceptedLive", "SaltIsLiveCoin")
#: why a window meets ``AcceptedLive`` (`Accounting.TS.Session`'s, ``accounting-ts-accepted-live``)
LIVE_MET = ("credit accepted the window, and it credits a drawn unit or a live round only where the verifier of record "
            "accepted its session (flock-verify verify --zk with the --registered file credit builds)")
#: why a window meets ``SaltIsLiveCoin`` (`Properties.Pouw.Period`'s)
SALT_MET = ("core_check.salt accepted the window's salt as its live salt coin, opened after the window's open root and read "
            "through the session the auditor's ledger opened for it, under the host's pins log and main, and the record's "
            "Program salt is that coin's (manifest_reason)")


def premises(doc: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Whether the window of `pouw`'s document `doc` meets ``AcceptedLive`` and ``SaltIsLiveCoin``, each ``{met, why}``:
    ``AcceptedLive`` only when core's credit accepted the window (a window credit refuses credits no proof, so it
    doesn't), ``SaltIsLiveCoin`` only when the challenger check accepted the window's coins, its salt among them, and its
    record as one manifest under them."""
    check, credit = doc.get("check"), doc.get("credit")
    reasons = list(doc.get("reasons") or ())
    unchecked = next((r for r in reasons if r.startswith("coins: ")), "; ".join(reasons) or "no challenger check")
    if check is None:
        salt = False, f"the challenger check didn't run on the window: {unchecked}"
    elif check.get("reason") is not None:
        salt = False, f"the challenger check refused the window's coins, its salt among them: {check['reason']}"
    elif check.get("manifest") is not None:
        salt = False, f"the window's record isn't one manifest under its salt coin: {check['manifest']}"
    else:
        salt = True, SALT_MET
    if credit is None:
        live = False, f"credit didn't run on the window, so it credited no proof: {unchecked}"
    elif credit.get("credit") is None:
        why = credit.get("window") or credit.get("manifest") or "no credited calls"
        why = f"{why[0]}: {why[1]}" if isinstance(why, (list, tuple)) else str(why)
        live = False, f"credit refused the window ({why}), so it credited no proof"
    else:
        live = True, LIVE_MET
    return {name: {"met": met, "why": why[:600]} for name, (met, why) in zip(PREMISES, (live, salt), strict=True)}


def pouw(serve_dir: Path, window: int, out: Path, *, gate: str | None, key: str | None, main: Path | None,
         check: Check, statements: Path | None = None, live: Path | None = None) -> dict[str, Any]:
    """PoUW's development piece on window `window` (module docstring), its document written to ``out/development.json``;
    `check` is `check_challenger` with its Python bound, given credit's inputs (`credit_inputs` of `serve_dir`, the drawn
    calls' `statements` and the `live` evidence), its document's ``credit_inputs``."""
    out.mkdir(parents=True, exist_ok=True)
    inputs = credit_inputs(serve_dir, statements, live)
    line = next((w for w in _lines(serve_dir / "windows.jsonl") if int(w["window"]) == window), {})
    h1 = line.get("pouw_h1") or {}
    calls = serve_dir / H1_CALLS.format(k=window)
    doc: dict[str, Any] = {"window": window, "credited": False, "why": DEV_WHY, "coin_source": h1.get("coin_source"),
                           "tied": h1.get("tied"), "steps_served": h1.get("steps_served"), "session": h1.get("session"),
                           "gate_passed": (h1.get("gate") or {}).get("gate_passed"), "check": None, "credit": None,
                           "credit_inputs": inputs}
    reasons = []
    if not h1 or "error" in h1:
        reasons.append(f"H1's window: {h1.get('error') or 'its line has no pouw_h1 (served without --protocols)'}")
    elif h1.get("tied") is not True:
        reasons.append(f"ROOTS1: H1's tables are not tied to the window's Commit ({(h1.get('roots1') or {}).get('reason')})")
    if h1 and "error" not in h1 and h1.get("coin_source") != "challenger":
        reasons.append(f"coins: {h1.get('coin_source')}, not the auditor host's challenger's, so nothing checks them")
    elif h1.get("coin_source") == "challenger":
        try:
            if not (gate and key):
                raise ValueError("no gate or no key pin to check the window's session by (H1_GATE, H1_KEY_SHA512)")
            sid = json.loads(calls.read_text())["commitment"]["session"]["session"]
            tgz, tree = out / "session.tgz", out / "session"
            tgz.write_bytes(_gate_out(gate, "fetch", sid))
            with tarfile.open(tgz) as t:
                t.extractall(tree, filter="data")
            (out / "ledger.jsonl").write_bytes(_gate_out(gate, "ledger"))
            (out / "pins.jsonl").write_bytes(_gate_out(gate, "pins"))
            got = check(calls, tree / sid, out / "ledger.jsonl", out / "pins.jsonl", main, key, out / "check", inputs)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, tarfile.TarError) as e:
            reasons.append(f"coins: {type(e).__name__}: {e}"[:600])
        else:
            doc["check"] = {k: got.get(k) for k in ("ok", "reason", "manifest", "rounds", "coin_source")}
            doc["credit"] = {k: (got.get("credit") or {}).get(k) for k in ("window", "manifest", "k", "drawn", "eps", "calls", "credit")}
            reasons += [f"{k}: {got[r]}" for k, r in (("coins", "reason"), ("manifest", "manifest")) if got.get(r)]
    doc.update(verdict="reject" if reasons else "accept", reasons=reasons)
    doc.update(premises(doc))
    (out / DEVELOPMENT).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    return doc


def check_answer(a: Any) -> dict[str, Any]:
    """``a`` when it is an answer in the shapes `Answer.ofJson` reads (`pouw.core_check.answer_of`, the Python mirror of
    the core interface's decoder): an accept has no reason and both its public outputs and its profile, a refusal a
    reason and neither. Anything else is a ValueError, which the row reports as the subprotocol's error."""
    from verity.core.protocols.pouw import core_check as CC

    try:
        CC.answer_of(a)
    except CC.Refused as e:
        raise ValueError(f"not a core answer: {e}") from None
    return dict(a)


def epsilon(a: Mapping[str, Any]) -> str | None:
    """A sampled protocol's bound on wrong units (``public.bound``) over its checked level's size (`Core.Integrity`), or
    None for an answer that has no such bound."""
    public, profile = a.get("public"), a.get("profile")
    levels = profile.get("levels") if isinstance(profile, Mapping) else None
    if not isinstance(public, Mapping) or not isinstance(levels, list) or not levels or not isinstance(levels[-1], Mapping):
        return None
    bound, size = public.get("bound"), levels[-1].get("size")
    return str(Fraction(bound, size)) if isinstance(bound, int) and isinstance(size, int) and size > 0 else None


def row_of(owner: str, a: Any, out: Path, run: str | None = None) -> dict[str, Any]:
    """A subprotocol's report row from its answer (checked to be a core answer first); ``run`` is the recording run."""
    a = check_answer(a)
    runs = json.loads((out / "runs.json").read_text()) if (out / "runs.json").exists() else []
    accepted = a["verdict"] == "accept"
    return {"protocol": a["protocol"], "owner": owner, "verdict": a["verdict"], "reason": a["reason"],
            "commitment": a["commitment"], "quantity": a["public"] if accepted else None,
            "delta": (a["profile"] or {}).get("delta") if isinstance(a["profile"], dict) else None,
            "epsilon": epsilon(a) if accepted else None, "runs": [*([run] if run else []), *runs], "answer": a}


def sampled_proofs_row(audit_dir: Path, run: str | None = None) -> dict[str, Any]:
    """Sampled proofs' row from the records audit kept under ``audit_dir`` (module docstring)."""
    import served_integrity as SI

    reg = audit_dir / "windows" / "rank0" / "registration.json"
    try:
        a = SI.answer(audit_dir, json.loads(reg.read_text()) if reg.exists() else None)
        return row_of(INTEGRITY_OWNER, a, audit_dir, run)
    except (RuntimeError, ValueError) as e:
        return {"verdict": "error", "owner": INTEGRITY_OWNER, "reason": f"{type(e).__name__}: {e}"[:2000]}


def _json(path: Path) -> Any:
    """``path``'s JSON, None where there's no such file; one that isn't JSON raises ValueError."""
    return json.loads(path.read_text()) if path.exists() else None


def piece(key: str, pieces_dir: Path, run: str | None = None) -> tuple[dict[str, Any], tuple[str | None, str]]:
    """A subprotocol's `piece_row` and `piece_server`; a piece that left a file that isn't JSON is its ``error``, about no
    server."""
    try:
        return piece_row(key, pieces_dir, run), piece_server(key, pieces_dir)
    except ValueError as e:
        return ({"verdict": "error", "owner": PIECES.get(key, WIPE)[2],
                 "reason": f"its piece left a file that isn't JSON: {type(e).__name__}: {e}"[:2000]}, (None, "real"))


def piece_row(key: str, pieces_dir: Path, run: str | None = None) -> dict[str, Any]:
    """A subprotocol's runner row from its piece's answer; a development piece's is ``missing`` with its document when it
    accepts (it credits nothing), else ``reject``; a piece that ran (`RAN`) and wrote neither is its ``error``, unless it
    refused to run (`REFUSED`); otherwise ``missing``, naming why."""
    _name, comp, owner, what = PIECES.get(key, WIPE)
    d = pieces_dir / comp
    f, dev, ran = d / "answer.json", d / DEVELOPMENT, _json(d / RAN)
    if not f.exists() and dev.exists():
        doc = _json(dev)
        if not isinstance(doc, dict):
            return {"verdict": "error", "owner": owner, "reason": f"its piece's {DEVELOPMENT} is not a JSON object"}
        if doc.get("verdict") == "accept":
            refused = (doc.get("credit") or {}).get("window")
            if isinstance(refused, list) and len(refused) == 2:
                refused = f"({refused[0]}) {refused[1]}"
            why = DEV_WHY if refused is None else (f"{DEV_WHY}; core's credit refuses the window "
                                                   f"{refused if isinstance(refused, str) else json.dumps(refused)}")
            return {"verdict": "missing", "owner": owner, "why": why[:2000], "development": doc}
        return {"verdict": "reject", "owner": owner, "reason": "; ".join(doc.get("reasons") or ())[:2000],
                "development": doc}
    if not f.exists():
        if (d / REFUSED).exists():
            said = (d / REFUSED).read_text(errors="replace").strip()
            return {"verdict": "missing", "owner": owner, "why": f"its piece refused to run: {said}"[:2000]}
        if isinstance(ran, dict):
            late = f", out of time at {ran.get('limit')}" if ran.get("rc") in (124, 137) else ""
            return {"verdict": "error", "owner": owner,
                    "reason": f"its piece ran (exit {ran.get('rc')}{late}) and wrote no answer; its log is {comp}/piece.log"}
        if key == "wipe" and (d / "piece.log").exists():
            return {"verdict": "missing", "owner": owner, "why": f"its piece wrote no answer; its log is {comp}/piece.log"}
        return {"verdict": "missing", "owner": owner, "why": _unset(comp, what)}
    try:
        return row_of(owner, json.loads(f.read_text()), f.parent, run)
    except ValueError as e:
        return {"verdict": "error", "owner": owner, "reason": f"{type(e).__name__}: {e}"[:2000]}


def piece_server(key: str, pieces_dir: Path) -> tuple[str | None, str]:
    """The server a piece's answer is about and the answer's source (``real`` or ``fixture``): the server its `SERVED`
    record names, else, but for the wipe, the node `pieces` ran it on (`RAN`); None where neither names one."""
    comp = PIECES.get(key, WIPE)[1]
    served, ran = _json(pieces_dir / comp / SERVED), _json(pieces_dir / comp / RAN)
    served = served if isinstance(served, dict) else {}
    ran = ran if isinstance(ran, dict) and key != "wipe" else {}
    return served.get("server") or ran.get("server"), served.get("source") or "real"


def _ratio(e: Fraction, deltas: Sequence[Fraction]) -> str:
    """``e`` exactly, as ``d + 2^-n`` where it is one of ``deltas`` plus a power of two."""
    for d in deltas:
        r = e - d
        if r > 0 and r.numerator == 1 and r.denominator & (r.denominator - 1) == 0:
            return f"{d} + 2^-{r.denominator.bit_length() - 1}"
    return str(e)


def terms(r: Mapping[str, Any], leg: Mapping[str, Any] | None) -> list[str]:
    """What an accepted answer holds beyond its verdict: where its profile says when its weights were held and what share
    of the checkpoint they are, those; where it says which host ran each party, what its `trusted-host` premise rests on
    (the network certifier's ``placement``); and each δ of the block record's at which its leg's quantity is not
    established."""
    profile = (r.get("answer") or {}).get("profile")
    w = profile.get("weights") if isinstance(profile, Mapping) else None
    place = profile.get("placement") if isinstance(profile, Mapping) else None
    out = [f"trusted-host {place['trusted-host']}"] if isinstance(place, Mapping) and place.get("trusted-host") else []
    if isinstance(w, Mapping) and w.get("held"):
        out.append(f"its weights {w['held']}, not concurrently with the block")
    if isinstance(w, Mapping) and w.get("share"):
        outside = (profile.get("outside") or {}).get("bytes") or {}
        out.append(f"they are {float(Fraction(w['share'])):.1%} of the checkpoint ({w.get('covers', 'its weights')})"
                   + (f"; {', '.join(sorted(outside))} are outside its claim" if outside else ""))
    if leg:
        no = [d for d, x in leg["at"].items() if not x["holds"]]
        if no:
            from verity_e2e_demo import block_record as BR
            out.append(f"its quantity is not established at δ = {' or '.join(no)}: the run established it except with "
                       f"{_ratio(Fraction(leg['error']), BR.DELTAS)}")
    return out


def said(rows: Mapping[str, Mapping[str, Any]], at: Mapping[str, tuple[str | None, str]], node: str,
         legs: Mapping[str, Mapping[str, Any]] | None = None) -> str:
    """The composed verdict's reason: one clause a subprotocol, its verdict and, unless it accepted about ``node``, why; an
    accept about ``node`` with its `terms` (``legs``: each accepted one's block record leg, by runner name)."""
    out = []
    for k, r in rows.items():
        v, server = r.get("verdict"), at[k][0]
        if v == "missing":
            s = r.get("why") if "development" in r else f"missing: {r.get('why')}"
        elif server not in (None, node):
            s = f"{v} about {server}, another server: carried, not composed"
        elif v == "accept":
            t = terms(r, (legs or {}).get(k))
            s = ("accept" if server is not None else "accept, naming no server: not carried") + (f" ({'; '.join(t)})" if t else "")
        else:
            s = f"{v}: {r.get('reason')}"
        out.append(f"{NAMES[k]}: {s}")
    return "; ".join(out)


def record(rows: Mapping[str, Mapping[str, Any]], *, window: int, steps: tuple[int, int], node: str,
           run: str | None, servers: Mapping[str, tuple[str | None, str]] | None = None
           ) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    """The window report over ``rows`` (by runner name), its block record (None when a subprotocol about this node
    rejected, erred or was undecided), and the ``RECORD`` line's summary. ``servers``: each piece's `piece_server`, by
    runner name; sampled proofs' answer is about ``node``, and a piece's that names no server isn't carried."""
    from verity_e2e_demo import block_record as BR

    legs = {v: k for k, v in BR.RUNNER_KEYS.items()} | {"wipe": "wipe"}
    at = {k: (None, "real") for k in rows} | dict(servers or {}) | {"integrity": (node, "real")}
    rows = {k: dict(r) for k, r in rows.items()}
    built: dict[str, dict[str, Any]] = {}
    for k, r in rows.items():
        if r.get("verdict") == "accept" and at[k][0] is not None:
            try:
                built[k] = BR.leg(legs[k], r, server=at[k][0], runs=[], source=at[k][1])
            except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as e:
                rows[k] = {**r, "verdict": "error", "quantity": None, "epsilon": None,
                           "reason": f"its answer is not the block record's {legs[k]} leg: {type(e).__name__}: {e}"[:2000]}
    report = {"format": REPORT_FORMAT, "window": window, "node": node, "run": run, "protocols": rows,
              "servers": {k: at[k][0] for k in rows}}
    verdicts = {NAMES[k]: DEV_VERDICT if "development" in r and r.get("verdict") == "missing" else r.get("verdict")
                for k, r in rows.items()}
    missing = {NAMES[k]: r.get("why") for k, r in rows.items() if r.get("verdict") == "missing"}
    elsewhere = {NAMES[k]: at[k][0] for k in rows if at[k][0] not in (None, node)}
    unserved = sorted(NAMES[k] for k, r in rows.items() if r.get("verdict") != "missing" and at[k][0] is None)
    dev = {"development": {NAMES[k]: r["development"]["verdict"] for k, r in rows.items() if "development" in r}}
    dev = dev if dev["development"] else {}
    named = {"verdicts": verdicts, "missing": missing, "elsewhere": elsewhere, **({"unserved": unserved} if unserved else {}),
             **dev}
    bad = sorted(NAMES[k] for k, r in rows.items()
                 if r.get("verdict") not in ("accept", "missing") and NAMES[k] not in elsewhere)
    if bad:
        composed = "reject" if any(rows[k].get("verdict") in ("reject", "error") for k in rows
                                   if NAMES[k] in bad) else "pending"
        report["composed"] = {"protocol": REPORT_FORMAT, "verdict": composed, "reason": said(rows, at, node, built)}
        line = {"composed": composed, "by": bad, **named, "block_record": None, "reason": report["composed"]["reason"],
                "why": "block_record carries an accepted subprotocol's certificate only"}
        return report, None, line
    extra = {legs[k]: {"row": r, "server": at[k][0], "runs": r.get("runs") or [run], "source": at[k][1]}
             for k, r in rows.items() if k != "integrity" and r.get("verdict") == "accept" and at[k][0] is not None}
    composing = {**report, "protocols": {k: r if k == "integrity" else {"verdict": "missing"} for k, r in rows.items()}}
    rec = BR.from_period(composing, steps=steps, deployment=BR.REFERENCE, extra=extra)
    report["composed"] = rec["composed"]["onsite"] = {"protocol": REPORT_FORMAT, "verdict": rec["composed"]["verdict"],
                                                      "reason": said(rows, at, node, built)}
    line = {"composed": rec["composed"]["verdict"], **named, "reason": report["composed"]["reason"],
            "record_missing": rec["composed"]["missing"],
            "check": BR.check(rec), "steps": list(steps), "node": node,
            "deployment": rec["monitoring_period"]["deployment"]["id"]}
    return report, rec, line


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("window")
    w.add_argument("serve_dir", type=Path)
    w.add_argument("--slots", type=int, default=8)
    w.add_argument("--window", type=int)
    g = sub.add_parser("gate")
    g.add_argument("--gate", default=os.environ.get("H1_GATE"))
    g.add_argument("--lane", default="vllm")
    g.add_argument("--main", type=Path, default=os.environ.get("MAIN_FILE") or None)
    pe = sub.add_parser("period")
    pe.add_argument("verb", choices=("open", "served", "close"))
    pe.add_argument("serve_dir", type=Path)
    pe.add_argument("--gate")
    pe.add_argument("--name")
    pe.add_argument("--circuits", type=Path)
    pe.add_argument("--slots", type=int)
    pe.add_argument("--max-tokens", type=int)
    pe.add_argument("--draw-k", type=int)
    pe.add_argument("--model")
    pr = sub.add_parser("program")
    pr.add_argument("out", type=Path)
    pr.add_argument("--model", default="verity-int8/Llama-3.2-1B-INT8")
    pr.add_argument("--rows", type=int, default=512)
    pr.add_argument("--steps", type=int, default=16)
    pr.add_argument("--like", type=Path)
    pr.add_argument("--protocols-config", help="serve-openai's --protocols-config (H1_PROTOCOLS_CONFIG), the same value")
    o = sub.add_parser("pouw")
    o.add_argument("serve_dir", type=Path)
    o.add_argument("window", type=int)
    o.add_argument("out", type=Path)
    o.add_argument("--gate", default=os.environ.get("H1_GATE"))
    o.add_argument("--key-sha512", default=os.environ.get("H1_KEY_SHA512"))
    o.add_argument("--main", type=Path, default=os.environ.get("MAIN_FILE") or None)
    o.add_argument("--python", default=os.environ.get("PY") or sys.executable)
    o.add_argument("--statements", type=Path)
    o.add_argument("--live-evidence", type=Path)
    rg = sub.add_parser("public-registration")
    rg.add_argument("registration", type=Path)
    sl = sub.add_parser("sampled-line")
    sl.add_argument("log", type=Path)
    sl.add_argument("--partition", type=Path)
    fl = sub.add_parser("front-line")
    fl.add_argument("session", type=Path)
    p = sub.add_parser("pieces")
    r = sub.add_parser("record")
    for q in (p, r):
        q.add_argument("--serve-run", type=Path, required=True)
        q.add_argument("--window", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", action="append", choices=[v[1] for v in PIECES.values()])
    p.add_argument("--skip", action="append", default=[], choices=[v[1] for v in PIECES.values()])
    r.add_argument("--sampled-proofs", type=Path, required=True)
    r.add_argument("--pieces", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    for q in (p, r):
        q.add_argument("--node", default=socket.gethostname())
    a = ap.parse_args(argv)
    if a.cmd == "public-registration":
        print(json.dumps(public_registration(json.loads(a.registration.read_text())), indent=1, sort_keys=True))
        return 0
    if a.cmd == "sampled-line":
        rows = [x[len("RECORDS "):] for x in a.log.read_text().splitlines() if x.startswith("RECORDS ")]
        if not rows:
            print("no RECORDS line")
            return 1
        print(sampled_line(json.loads(rows[0]), json.loads(a.partition.read_text()) if a.partition else None))
        return 0
    if a.cmd == "front-line":
        try:
            print(front_line(json.loads(a.session.read_text())))
        except (OSError, ValueError) as e:
            print(f"no session ({type(e).__name__})")
            return 1
        return 0
    if a.cmd == "window":
        try:
            print(json.dumps(public_pick(pick(a.serve_dir, a.slots, a.window), a.slots), sort_keys=True))
        except ValueError as e:
            print(f"served_verdicts.py window: {e}", file=sys.stderr)
            return 1
        return 0
    if a.cmd == "gate":
        if not a.gate:
            print("GATE refused: no gate command (--gate, H1_GATE)", flush=True)
            return 1
        try:
            ok, line = gate_ready(a.gate, a.lane, a.main)
        except (OSError, RuntimeError, ValueError) as e:
            ok, line = False, f"GATE refused: {type(e).__name__}: {e}"[:800]
        print(line, flush=True)
        return 0 if ok else 1
    if a.cmd == "period":
        try:
            if a.verb == "served":
                doc = period_served(a.serve_dir)
                print(f"PERIOD served {doc['name']}: serve-openai declared its Program {doc['circuit'][:16]}...",
                      flush=True)
                return 0
            if not a.gate:
                raise ValueError("no gate command (--gate)")
            if a.verb == "close":
                doc = period_close(a.serve_dir, a.gate)
                print(f"PERIOD closed {doc['name']}: {json.dumps(doc['close'], sort_keys=True)[:600]}", flush=True)
                return 0
            if not (a.name and a.circuits and a.slots and a.max_tokens):
                raise ValueError("period open needs --name, --circuits, --slots and --max-tokens")
            doc = period_open(a.serve_dir, a.gate, a.name, a.circuits, slots=a.slots, max_tokens=a.max_tokens,
                              draw_k=a.draw_k, model=a.model)
        except (OSError, RuntimeError, ValueError, KeyError) as e:
            print(f"PERIOD refused: {type(e).__name__}: {e}"[:1200], flush=True)
            return 1
        print(f"PERIOD opened {doc['name']}: circuit {doc['circuit'][:16]}..., T {doc['rows']}, S {doc['steps']}, draw K "
              f"{doc['draw_k']}, weights {' '.join(doc['weights']) or 'none'}, files {sorted(doc['circuit_files'])}"
              + (f" (no k<K> file yet for K = {doc['missing']}: a drawn statement there is refused)" if doc["missing"]
                 else ""), flush=True)
        return 0
    if a.cmd == "program":
        try:
            meta = program(a.out, a.model, a.rows, a.steps, a.like, a.protocols_config)
        except (OSError, ValueError, KeyError) as e:
            print(f"PROGRAM refused: {type(e).__name__}: {e}"[:1200], flush=True)
            return 1
        print(f"PROGRAM {meta['program_digest'][:16]}... T {meta['rows']}, S {meta['steps']}, {meta['bytes']} bytes"
              + (f", serve-openai's at {a.like}" if a.like else ""), flush=True)
        return 0
    if a.cmd == "pouw":
        doc = pouw(a.serve_dir, a.window, a.out, gate=a.gate, key=a.key_sha512, main=a.main,
                   check=lambda *x: check_challenger(*x, python=a.python), statements=a.statements,
                   live=a.live_evidence)
        print(f"POUW-DEV {json.dumps(development_line(doc), sort_keys=True, default=str)}", flush=True)
        return 0
    if a.cmd == "pieces":
        a.out.mkdir(parents=True, exist_ok=True)
        got = pieces(a.serve_run, a.window, a.out, node=a.node, only=a.only, skip=a.skip)
        return 0 if all(g["verdict"] in ("accept", "missing") for g in got.values()) else 1
    steps = pick(a.serve_run, 1 << 30, a.window)["steps"]
    run = os.environ.get("RESEARCH_RUN_ID")
    keys = [*PIECES, "wipe"]
    got = {k: piece(k, a.pieces, run) for k in keys}
    rows = {"integrity": sampled_proofs_row(a.sampled_proofs, run), **{k: row for k, (row, _) in got.items()}}
    report, rec, line = record(rows, window=a.window, steps=(steps[0], steps[1]), node=a.node, run=run,
                               servers={k: at for k, (_, at) in got.items()})
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "verdicts.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    if rec is not None:
        (a.out / "block-record.json").write_text(json.dumps(rec, separators=(",", ":"), sort_keys=True) + "\n")
    print("RECORD " + json.dumps(line, sort_keys=True), flush=True)
    return 0 if rec is not None and not line["check"] else 1


if __name__ == "__main__":
    sys.exit(main())
