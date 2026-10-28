"""On one GPU: the int8 Llama-3.1-8B pin (as `tests/engine/int8_exact_gpu.py` finds or makes it) served on the eager
int8_exact workload with PoUW's H1 + fx1 (``pouw:eps1-sm120-h1fx1``, `protocol_options.pouw_h1`), checked on the host.

One process writes `<out>/pouw_h1_gpu.json` and the option's record (`<out>/pouw-h1-calls.json`):

  served    `PROMPTS` greedy, batched (a decode batch of len(PROMPTS)) and the first alone (decode at one row), before the
            option is installed and with it: the token ids are the same, since the arm serves its own y. The engine's
            ``max_num_batched_tokens`` is T (``--rows``), with chunked prefill, so no step puts more rows in a linear.
  calls     the static window (`H1Run`, T rows a call, S = ``--steps``): every (step, linear) call in order at s·U + off(w),
            the served ones recorded and the rest empty, nothing refused; the credit (served rows) beside the padding.
  compile   every served shape compiled to `eps1-sm120-h1fx1` (`H1Run.check`): its Program digest, proof blocks, one seed group and
            only full 32 x 32 grid tiles. A shape takes minutes on the host (about 0.13 s a proof block), so only `--compile` does it
            here; otherwise `--compile-from <out>/pouw-h1-calls.json` does it on a CPU node from the record, with each call's
            positions against its shape's compiled proof blocks, and writes `<out>/pouw_h1_compile.json`.
  z         the last retained calls' exact Y (`engine.int8_exact.int_mm` on the GPU) against the s8 product on the host, and
            with `--compile`, a proof block's z (`Call.z`) against its cells.
  hook      a listener (`H1Run.listen`) gets every call with the served tensors on the GPU: A int8 [m, k] and s_a f32 [m]
            (the Commit's ``Int8RowQuant_v1`` /0 and /1), the layer's own int8 weight and scales; what it returns is in the
            manifest; steps begin at linear 0, and every step's calls run over the same layers in order.
  requests  every request the scheduler admits is an entry under the root of its step's round, recorded before the round's
            live coin, and every served row's slot names its request (`h1_settings`: the engine's runner and the window's
            round coins; `requests_check`), one entry a served request and no served row without one.

``--workload fill`` serves `pouw_h1.FILL` instead: 64 one-token prompts (`fill_prompts`) under
the static schedule (``ignore_eos``, every request to its cap, host prefix cache off), with ``max_num_seqs`` 64 and T = 64,
so all 64 prefill at step 0 and decode together for the 16 served steps of an S = 18 window. Without the option it serves
the 64 and then the first alone (the 1-sequence workload); with it, the 64 in one window (`<out>/pouw-h1-calls.json`) and the
first alone in a second (`<out>/solo/`). It records each window's step rows against the declaration (`Static.check`), its
u (`H1Run.utilization`), the credit gate's verdict, each step's seconds (at linear 0 of the next) and tokens/s, and the
static schedule's check (`engine.static_schedule.check`) of every request. Compile is `--compile-from` on each record.

``--challenger HOST:PORT --session JSON --gate CMD`` takes the window's salt and round coins from the auditor host's
challenger, in the challenge phase the challenger gate reserved for the window (``--session``, the gate's reserve answer), which the
window opens through ``CMD open --session S --root R`` on its window-open root once round 0 is admitted
(`gate_command`; `pouw_h1_window.Window`; on PEER, the port the gate forwards), one challenge phase a window, so the fill
workload serves only the 64's window. With ``--peer NAME`` in place of ``--session`` each window the run serves reserves
a challenge phase of its own (`window_sessions`: ``CMD reserve --peer NAME``) just before it is installed, and the next is
reserved only once that challenge phase's end entry is in the gate's ``status`` (`session_end`), so ``--commit``'s windows
and the solo window all take the auditor host's coins. ``--check-challenger CALLS_JSON DIR --key-sha512 HEX --ledger HEAD [--ledger HEAD
...] --pins HEAD [--pins HEAD ...] --main FILE`` is the verifier's side on no GPU, ``--ledger`` given once per head
the gate published, ``--pins`` once per head of the host's pins log the gate published and ``--main`` the auditor's ``git
rev-list --first-parent origin/main`` (without them every window is refused). With ``--window WINDOW_JSON``
(``{root}``, the window's open root) DIR is the challenger run's own ``out/``: the window's coins against the challenge phase
the auditor's signed ledger (the longest of the heads, each extending the others) opens and ends for the window
(`pouw_h1_challenger.trusted_rounds_reason`). Without it the ledger is read as the newest that the others extend
(`pouw_h1_challenger.Ledger.load`, core's check), and DIR the challenge phase's tree as the gate's ``fetch`` gives it: the window held to the
challenge phase the ledger opened for its Commit's root (`pouw_h1_challenger.session_reason`). Either way the record is held to
its coins as one manifest (`pouw_h1_challenger.manifest_reason`), and with ``--period NAME`` the ledger must credit the lane's period
with the window accepted and presented as its close's windows, its open committing ``draw_k``
(`pouw_h1_challenger.period_reason`, core's `credit.period`). The batch workload admits the first prompt alone after the
batch, so it admits per round; the fill workload admits in round 0 only.

``--served`` (fill) commits each call's A and z/y tables and the weights' B as its window Program's registered values
(`pouw_h1_served.Served`, the pouw setting ``served``), under the round roots, which the commitment carries, and records
the trees and `pouw_h1_served.register`'s outcome (`served_check`). With ``--model M --circuit C`` (and ``--gate``) the
weights' ``b/<K>`` are the period's (`pouw_h1_served.Weights`): the gate's ``model-key --model M --circuit C``
answers the auditor's registration of M under the period's circuit C and its salt key (`model_key_of`), and the served
model's rows are registered under that key before the salt, the window refused unless they fold to the registration's
weights' root, the one ``period-open --model M`` signed. ``pouw_h1_rehearsal.py`` runs it all against the challenger gate.
``--baseline-h1`` (fill, with ``--served``) serves an H1 window without `Served` first, in the solo window's place, so
one engine measures plain, H1 and H1 with its registered tables; each window records its host RSS, and a served one
its registration's seconds a step (``registered_s``), the part of them inside `pouw_h1_registered.commit`
(``hashed_s``) and where that hashed (``register_path``: the GPU or NumPy). The period's declared Program and its digest are
declared once before any window (`period_program`), as the period's.

``--receipts SO`` (fill, ``--served``, no ``--handoff``) registers each call's ``receipt@<c>`` too (BRIDGE2), computed on the
serving GPU by kernels' producer (``benchmarks/pouw/served_h1/producer.py``, `receipts_producer`): h1u.cu's proof-block kernel of
the -hbc build SO (``producer_gpu.sh``'s ``h1u_hbc.so``) over each linear's own int8 weight, its seeds taking the
registered roots of the call's step's ``a/<K>@<s>`` and ``b/<K>``. The first layer's linears' calls at step 0 and at the
window's last step are kept, and their first proof block of each row block is held to `h1tile.reference` word for word
(`receipts_check`: live row blocks at step 0, dead at the last step). A ``.cubin`` SO (``served_h1/noise.cu``'s) computes
them with the producer's torch chain over noise.cu's noise instead. ``--j10r`` declares the period's window Program over
J10-R's tile proof block (`period_unit`, `h1tile.PouwH1Tile5`), so the producer computes J10-R's receipts (one noise stream a call)
and they are held to `h1tile.j10r_reference`; v1's proof block is the default.

``--credit`` (with ``--commit`` and ``--receipts``) runs core credit's statement steps over the honest Commit window
(`credit_check`, `pouw_h1_credit`): `credit.registered_file`, `credit._verified` on a stand-in verifier of record that
opens each drawn proof block's pub, o0 and o1 rows against that file's roots and recomputes its receipt, `credit._proved` and
`credit._registration`, the drawn proof blocks the first and last of each kept call; then a receipt word flipped (with and
without its round's root re-listed), a public table under another step's A root and an ok row 0, each to be refused.

``--commit`` (fill, ``--served``) serves the 64's window again with the Commit in the engine
(`acquire.h1_commit.attach`: the rows-sha512 committer, the int8 source at each linear's Int8RowQuant_v1 site, the
step-input port and its tap) into ``<out>/commit/``, each call's A table held to the Commit's AROW and each round over its
input ids (``roots1``), its ``committer`` record its 64-byte run root (`pouw_h1_commit.committed`, which refuses any other),
and records the peak GPU memory and step time of the plain run, the window and the window with the Commit (``commit``). Its
B is registered once, before its first window (under a fresh local key, `local_weights`, unless ``--model`` names the
auditor's registration), and reused by the later windows. Each
Commit window is also held to core's ROOTS1 steps over its own evidence (`core_verdict`: `pouw_h1_challenger.manifest_reason`,
`credit._a_link`, `credit._arow`), which it saves for ``--core-from DIR`` on a CPU node. ``--negatives`` then serves the
window twice more with an honest Commit, H1 handed all-zero A codes or one A row off (`NEGATIVES`), each to be refused at
``arow``, and on a MoE (``--role OLMOE_INT8 --int8-exact linears``) once more with one routed expert call's codes zero
(`EXPERT_NEGATIVE`): the Commit acquires each expert call's rows under its H1 name (`acquire.h1_commit.H1Source`). ``--declare SLOTS,CAP,T,S`` shrinks the fill workload (`declared_of`) so the pure-Python registration fits a run.
Without ``--challenger`` its windows' coins are this process's, so every call's credit gate verdict is
`pouw_h1.IN_PROCESS`; with ``--challenger --gate --peer`` each window draws its salt, round and draw coins from a
challenge phase of its own, and ``commit.coins`` (`coins_summary`) holds each window's coin source, challenge phase and credit gate,
``ok`` only when every window's coins are the challenger's and the honest Commit window's gate refuses none of its calls.

``--handoff SOCK --handoff-state DIR`` (fill, with ``--challenger``) splits the window at its last served step, so the
GPU is held only while it serves: the serving process records what only it sees (`served_measures`), saves the window,
hands its challenge phase's stream to the holder at SOCK and exits (`hand_off`, `pouw_h1_handoff`), and ``--close-from DIR
--stream-fd FD`` closes and releases it on no GPU (`fill_close`): the same close, trees, draw and challenge phase close, the
record the same `RECORD`.
"""
from __future__ import annotations

import argparse
import functools
import gc
import gzip
import resource
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tests.engine.int8_exact_gpu import PROMPTS as SERVE_H1_PROMPTS
from tests.engine.int8_exact_gpu import generate, pin, scratch_checkpoints

SCHEMA = "verity-vllm/pouw-h1-gpu/v1"
RECORD = "pouw_h1_gpu.json"
PROMPTS = ["The capital of France is", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
           "The three primary colors are", "Water boils at a temperature of", "Once upon a time, in a small village,",
           "The speed of light is", "In 1492, Columbus", "The largest planet in the solar system is", "To make a cup of tea,",
           "The chemical symbol for gold is", "A haiku about autumn:", "The square root of 144 is", "Photosynthesis is the process"]


PERIOD_DIGEST = """
import json, resource, sys, time
from verity.core.protocols.sampled_proofs.one_stage import partition as P
from verity_vllm.protocols.pouw.h1 import served as SV
shapes, modules, rows, steps, j10r, arch = json.loads(sys.argv[1])
unit = None
if j10r:
    from verity.ml.protocols.pouw.circuit import h1tile as HT, h1window as HW
    from verity_vllm.protocols.pouw import h1 as PH
    unit = lambda k: HW.tile_unit(k, PH.TILE, HT.PouwH1Tile5, **({"arch": arch} if arch else {}))
t = time.monotonic()
digest = P.program_digest(SV.declared([tuple(s) for s in shapes], {int(w): n for w, n in modules.items()}, rows, steps,
                                      unit=unit, arch=arch))
print(json.dumps({"digest": digest, "s": round(time.monotonic() - t, 1),
                  "peak_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}))
"""


def period_unit(j10r: bool, arch: str | None = None) -> Any:
    """The window Program's tile proof block at each K (`pouw_h1_served.declared`'s ``unit``, as `PERIOD_DIGEST` builds it): None,
    v1's (`h1window.tile_unit`'s on ``arch``, `declared`'s), or with ``j10r`` J10-R's, `h1tile.PouwH1Tile5` under `h1zy`
    on ``arch`` (``--j10r``)."""
    if not j10r:
        return None
    from verity.ml.protocols.pouw.circuit import h1tile as HT
    from verity.ml.protocols.pouw.circuit import h1window as HW
    from verity_vllm.protocols.pouw import h1 as PH
    return lambda k: HW.tile_unit(k, PH.TILE, HT.PouwH1Tile5, **({"arch": arch} if arch else {}))


def period_program(model: Any, rows: int, steps: int, j10r: bool = False,
                   arch: str | None = None) -> tuple[Any, str, dict[str, Any]]:
    """The period's declared Program (`pouw_h1_served.declared` of the shapes and names `pouw._install_h1` gives its run,
    over `period_unit`'s tile proof block on the serving card's ``arch``, which refuses a K outside its H1 record's domain), its
    digest (the period's ``--circuit``) and what declaring them cost, once before any window, as a period declares them.
    The digest is taken in a child process, since at FILL it alone takes minutes and tens of GB, which no window pays, so
    this process's RSS is the windows' own."""
    from verity_vllm.protocols import pouw
    from verity_vllm.protocols.pouw.h1 import hooks as PHK
    from verity_vllm.protocols.pouw.h1 import served as SV

    layers = PHK.layers(model, pouw.linear_types())
    shapes = [tuple(int(d) for d in m.weight.shape) for m in layers]
    modules = {w: f"linear {w}" for w in range(len(layers))} | pouw._named(model, layers)
    child = subprocess.Popen([sys.executable, "-c", PERIOD_DIGEST, json.dumps([shapes, modules, rows, steps, j10r, arch])],
                             stdout=subprocess.PIPE, text=True)
    t = time.monotonic()
    program = SV.declared(shapes, modules, rows, steps, unit=period_unit(j10r, arch), arch=arch)
    declare_s = round(time.monotonic() - t, 1)
    out, _ = child.communicate()
    if child.returncode:
        raise RuntimeError(f"the period's Program digest exited {child.returncode}")
    got = json.loads(out.strip().splitlines()[-1])
    return program, got["digest"], {"shapes": shapes, "modules": modules, "declare_s": declare_s,
                                    "digest_s": got["s"], "digest_peak_gib": got["peak_gib"]}


def gate_argv(argv: list[str], args: Any) -> list[str]:
    """The command running the gate's verb `args` through `argv`: over ssh the gate's forced command reads the remote
    command line with ``shlex.split``, so the words go as one shell-quoted line (a ``--families`` JSON has spaces)."""
    return [*argv, shlex.join(args)] if Path(argv[0]).name == "ssh" else [*argv, *args]


def listening(port: int) -> bool:
    """Whether a socket on this host listens on TCP `port`: ``ss -Hltn``, else the kernel's tables it reads
    (``/proc/net/tcp``, ``tcp6``; state 0A)."""
    import shutil
    if shutil.which("ss"):
        return bool(subprocess.run(["ss", "-Hltn", f"sport = :{port}"], capture_output=True, text=True).stdout.strip())
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            rows = Path(table).read_text().splitlines()[1:]
        except OSError:
            continue
        if any(int(r.split()[1].rsplit(":", 1)[1], 16) == port and r.split()[3] == "0A" for r in rows):
            return True
    return False


def await_listener(port: int, polls: int = 300) -> int | None:
    """The polls (one a second, at most `polls`) until `port` listens here (`listening`), or None: the live challenger
    reaches the serving node through its certifier coin link once the gate's open has answered, and a probe connect
    would take the place of the challenge phase's one client."""
    for i in range(polls):
        if listening(port):
            return i
        time.sleep(1)
    return None


def await_end(status: Any, polls: int = 300) -> dict[str, Any]:
    """`status()` (the gate's ``status S``) once challenge phase S has its ``end`` entry (serve.py's exit after the window's
    close), was never opened or is refused, polled once a second at most `polls` times: the live gate has no ``wait``."""
    for _ in range(polls):
        s = status()
        if s.get("end") is not None or s.get("open") is None or "refused" in s:
            return s
        time.sleep(1)
    return status()


def gate_command(argv: list[str]) -> Any:
    """The `pouw_h1_window.Gate` that runs the gate's verbs as `argv` + VERB (an ``ssh ... research@HOST``): ``open
    --session S --root R``, its JSON answer's ``commitment`` (null until serve.py signs its record) then read from ``status
    S``, and the answer returned once its ``port`` listens on this host (`await_listener`). ``reserve(peer, steps)`` is
    the gate's reserve of a window of S = `steps` rounds served on `peer` (its answer; ValueError naming what the gate
    said when it refuses), and ``ended(session)`` the challenge phase's ``status`` once it has its end entry (`await_end`)."""
    def verb(*args: str) -> dict[str, Any]:
        return json.loads(subprocess.run(gate_argv(argv, args), check=True, capture_output=True, text=True).stdout)

    def gate(session: str, root: str) -> dict[str, Any]:
        answer = verb("open", "--session", session, "--root", root)
        if answer.get("commitment") is None:
            answer["commitment"] = verb("status", session).get("commitment")
        if type(answer.get("port")) is int:
            polls = await_listener(answer["port"])
            print(f"[gate] session {session}: port {answer['port']} "
                  + (f"listening after {polls} polls" if polls is not None else "not listening after 300 polls"), flush=True)
        return answer

    def reserve(peer: str, steps: int) -> dict[str, Any]:
        from verity_vllm.protocols.pouw.h1 import challenger as PHC
        p = subprocess.run(gate_argv(argv, ["reserve", "--peer", peer, "--scheme", PHC.SCHEME, "--families",
                                            PHC.families(steps)]), capture_output=True, text=True)
        try:
            answer = json.loads(p.stdout)
        except ValueError:
            answer = None
        if p.returncode or not isinstance(answer, dict) or "refused" in answer:
            raise ValueError(f"the gate refused the reserve (exit {p.returncode}): {(p.stdout + p.stderr)[-1000:]}")
        return answer

    gate.argv, gate.reserve, gate.ended = argv, reserve, lambda session: await_end(lambda: verb("status", session))
    return gate


def window_sessions(gate: Any, peer: str, steps: int) -> Any:
    """With ``--peer``: each window the run serves gets a challenge phase of its own, reserved at the gate (``gate.reserve``) just
    before the window is installed, since the auditor host's challenger serves one window a challenge phase and a challenge phase opens
    once; ``sessions(name)`` is window `name`'s reserve answer (`pouw_h1_window.reserved_of`), and a name asked for twice
    is refused (ValueError), as is a reserve the gate refuses. ``sessions.held`` maps each window to its answer."""
    from verity_vllm.protocols.pouw.h1 import window as PHW
    held: dict[str, dict[str, Any]] = {}

    def sessions(name: str) -> dict[str, Any]:
        if name in held:
            raise ValueError(f"window {name} has its session ({held[name]['session']}) already: one session a window")
        answer = gate.reserve(peer, steps)
        reserved = PHW.reserved_of({k: v for k, v in answer.items() if k in ("session", *PHW.RESERVE_FIELDS)})
        if reserved is None or reserved["session"] in {r["session"] for r in held.values()}:
            raise ValueError(f"window {name}: the gate's reserve answer {answer!r} is no fresh session")
        held[name] = reserved
        print(f"[gate] window {name}: reserved session {reserved['session']} on {peer}", flush=True)
        return reserved
    sessions.held = held
    return sessions


def session_end(gate: Any, reserved: dict[str, Any]) -> dict[str, Any]:
    """Window's challenge phase `reserved` (its reserve answer) once released: its ``end`` entry as the gate's ``status`` gives it,
    ``ok`` when the challenger's client closed it and every coin family released all its coins (``complete``). The next
    window's challenge phase is reserved only after this one has ended."""
    s = gate.ended(reserved["session"])
    end = ((s.get("end") or {}).get("body") or {}) if isinstance(s.get("end"), dict) else {}
    return {"session": reserved["session"], "seq": reserved.get("seq"), "opened": s.get("open") is not None,
            "end": {k: end.get(k) for k in ("rc", "closed", "complete", "commitment")} if end else None,
            "ok": bool(end.get("closed") and end.get("complete"))}


def fixed_length(decl: Any) -> dict[str, Any]:
    """The fixed length `decl`'s window serves (``period-open --length``, core's `credit.LENGTH_FIELD`): each request's
    public max_tokens, every one admitted at step 0 and run to its cap (``ignore_eos``, the public-cap done source)."""
    return {"max_tokens": [decl.max_tokens] * decl.slots, "stop": "fixed"}


def period_open_argv(name: str, circuit: str, draw_k: int, decl: Any, files: dict[str, str]) -> list[str]:
    """``period-open`` of the Commit window's period `name`: cap 1 (the Commit's window alone, so a negative's refused
    window, which ends complete, is never one of it), its circuit (the period's declared Program's digest), its draw K
    (`pouw_h1_challenger.draw_k` over the declared layout, core's `credit.committed_k`), T, its circuit files
    (`period_files`, core's `credit.committed_files`) and the fixed length (`fixed_length`, core's
    `credit.committed_length`)."""
    return ["period-open", "--name", name, "--cap", "1", "--circuit", circuit, "--draw-k", str(draw_k),
            "--rows", str(decl.rows), "--circuit-files", json.dumps(files, sort_keys=True, separators=(",", ":")),
            "--length", json.dumps(fixed_length(decl), sort_keys=True, separators=(",", ":"))]


def period_files(circuits: Path, circuit: str, layout: dict, decl: Any) -> dict[str, str]:
    """The ``--circuit-files`` of `circuits`' manifest (`pouw_h1_circuits.build`'s, with ``--live``): every stratum's
    ``k<K>`` of `layout` and the live template's, built for the period's `circuit` and the live Program at `decl`'s T
    rows and S rounds. ValueError naming the build that would be, for any other."""
    from tests.protocols import pouw_h1_circuits as PC
    redo = f"build them with pouw_h1_circuits --program <Program {circuit[:16]}...> --out DIR --live {decl.rows},{decl.steps}"
    try:
        doc = json.loads((circuits / PC.MANIFEST).read_text())
    except (OSError, ValueError) as e:
        raise ValueError(f"no circuit files at {circuits} ({type(e).__name__}): {redo}") from None
    if doc.get("program") != circuit:
        raise ValueError(f"{circuits}'s circuit files are Program {str(doc.get('program'))[:16]}...'s, not the period's: {redo}")
    lb = doc.get("live_bind") or {}
    if (lb.get("T"), lb.get("S")) != (decl.rows, decl.steps):
        raise ValueError(f"{circuits}'s live circuit file is for T = {lb.get('T')}, S = {lb.get('S')}, not the window's "
                         f"{decl.rows}, {decl.steps}: {redo}")
    want = {f"k{k}" for k in PC.strata(layout)} | {"live"}
    if set(doc.get("files") or {}) != want:
        raise ValueError(f"{circuits}'s circuit files are {sorted(doc.get('files') or {})}, not {sorted(want)}: {redo}")
    return doc["files"]


def period_verb(gate: Any, argv: list[str]) -> dict[str, Any]:
    """The gate's answer to `argv` (``period-open``/``period-close``); ValueError naming what it said when it refuses."""
    p = subprocess.run(gate_argv(gate.argv, argv), capture_output=True, text=True)
    try:
        answer = json.loads(p.stdout)
    except ValueError:
        answer = None
    if p.returncode or not isinstance(answer, dict) or "refused" in answer:
        raise ValueError(f"the gate refused {argv[0]} (exit {p.returncode}): {(p.stdout + p.stderr)[-1000:]}")
    return answer


def model_key_of(gate: Any, model: str, circuit: str) -> Any:
    """The auditor's registration of `model` under the period's `circuit` and its salt key
    (`pouw_h1_served.ModelKey`), as the gate's ``model-key --model M --circuit C`` answers the lane's gate key; the key
    is never printed or written here."""
    from verity_vllm.protocols.pouw.h1 import served as SV
    p = subprocess.run(gate_argv(gate.argv, ["model-key", "--model", model, "--circuit", circuit]), capture_output=True,
                       text=True)
    try:
        answer = json.loads(p.stdout)
    except ValueError:
        answer = {"refused": f"exit {p.returncode}: {p.stderr[-500:]}"}
    return SV.model_key(answer)


def h1_settings(eng: Any, rows: int, steps: int, challenger: str | None = None, session: Any = None, gate: Any = None,
                **kw: Any) -> dict[str, Any]:
    """The option's settings for one served window: the engine's model runner (where the scheduler's admissions arrive) and
    the window's coins, its salt coin and one live round coin a step: the auditor host's at `challenger` (``HOST:PORT``, one
    challenge phase a window, `pouw_h1_challenger`) in the `session` the challenger gate reserved for it, which `gate` opens on the
    window's open root, else held in this process, which no verifier trusts and which credits nothing (a measurement)."""
    if challenger:
        return {"pouw": {"challenger": challenger, "session": session, "gate": gate, "runner": eng.runner, "rows": rows,
                         "steps": steps, **kw}}
    from verity.core.protocols.pouw import verifier
    from verity.core.protocols.pouw import window as Wd
    from verity.core.protocols.pouw.coins import Challenger
    from verity.core.protocols.sampled_proofs.challenger import LiveRecord
    rounds = LiveRecord(Wd.live_challenger(0, steps, verifier.coins), Wd.ROUND)
    return {"pouw": {"coins": Challenger(0, verifier.coins), "runner": eng.runner, "rounds": rounds, "rows": rows,
                     "steps": steps, **kw}}


def requests_check(h1: Any, requests: int) -> dict[str, Any]:
    """The window's admitted requests: their count against the ones served, each entry opened under the root of the round
    that admitted it, recorded before that round's coin, at a first slot naming it (`pouw_h1_requests.request_reason`), every slot
    of every step naming the request its row is, under a round no later than its step (`slot_reason`), the log against the
    coins (`rounds_reason`), no served row without an entry (``gaps``), and each call's A table under its round's root
    (``a_roots``, `pouw_h1_requests.tables_reason`). With the Commit the request and slot tables are live: their
    leaves open in each round's ``live@<s>``, so ``live`` holds each round's verdict (`pouw_h1_live.verdicts`, its proof block's
    circuit evaluated on its statement) in place of the openings."""
    from verity_vllm.protocols.pouw.h1 import live as LV
    from verity_vllm.protocols.pouw.h1 import requests as RQ
    coins = h1.rounds.challenger.commitment
    live = LV.verdicts(h1)
    why = [] if live else [w for e in h1.requests if (w := RQ.request_reason(
        h1.round_log, coins, h1.salt, h1.rows, e, h1.open_request(e.index), h1.open_first(e.index), h1.admission))]
    manifest, slots = h1.manifest(), [f"{v['reason']} (m = {v['m']})" for v in live if not v["ok"]]
    for s in range(0 if live else len(h1.round_log)):
        calls = [m for m in manifest if m["step"] == s]
        for p in range(h1.rows):
            slot = h1.open_slot(s, p)
            owner = RQ.slot_owner(slot.value[:-RQ.HM96_SALT])
            e = h1.requests[owner] if owner is not None else None
            w = RQ.slot_reason(h1.round_log, coins, h1.salt, h1.rows, calls, slot, e, e and h1.open_request(e.index),
                               e and h1.open_first(e.index), h1.admission)
            if w is not None:
                slots.append(w)
    rounds = RQ.rounds_reason(h1.round_log, coins)
    owners = [h1.owners.get(s, []) for s in range(len(h1.round_log))]
    out = {**h1.details["requests"], "served": requests, "entries_refused": why[:20], "slots_refused": slots[:20],
           "rounds": rounds, "entries_checked": len(h1.requests), "slots_checked": len(h1.round_log) * h1.rows,
           "slot_owners": owners, "slots_are_rows": [len(o) for o in owners] == h1.step_rows()[:len(owners)],
           "entries": [[e.index, e.round, e.slot] for e in h1.requests], "live": live,
           "live_accepted": f"{sum(v['ok'] for v in live)}/{h1.steps}" if live else None}
    out["ok"] = bool(out["admitted"] == requests and not out["refused"] and not out["gaps"] and not why and not slots
                     and rounds is None and out["slots_are_rows"] and str(out.get("a_roots")).startswith("bound")
                     and (not live or len(live) == h1.steps))
    return out


def run(a: argparse.Namespace) -> dict[str, Any]:
    from verity_vllm import protocols as PO
    from verity_vllm.engine.build import build_engine
    from verity_vllm.program.frontend.kernel_readback import int8_exact_of
    from verity_vllm.program.frontend.target_profile import TargetProfile
    from verity_vllm.protocols import pouw
    from verity_vllm.protocols.pouw import h1 as PH
    from verity_vllm.protocols.pouw.h1 import hooks as PHK

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    target = TargetProfile.from_device().with_overrides(int8_exact=a.int8_exact, batch_invariant=a.batch_invariant)
    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
                           "scheme": PH.SCHEME}
    with tempfile.TemporaryDirectory() as scratch:
        entry, how = pin(a)
        doc["pin"] = {"repo": entry["repo"], "revision": entry["revision"], "source": how}
        eng = build_engine(scratch_checkpoints(a.checkpoints, entry, Path(scratch)), case=a.role, target=target,
                           max_num_seqs=len(PROMPTS), engine_args={"max_num_batched_tokens": a.rows,
                                                                   "enable_chunked_prefill": True, "enable_prefix_caching": False,
                                                                   **a.engine_arg})
    steps_public = a.steps or 2 * (a.max_tokens + 2)
    doc["public"] = {"slots": len(PROMPTS), "rows": a.rows, "steps": steps_public, "max_tokens": a.max_tokens}
    doc["engine"] = {"int8_exact": int8_exact_of(eng.model), "enforce_eager": eng.kwargs.get("enforce_eager")}
    print(f"[h1] engine {json.dumps(doc['engine'], default=str)}", flush=True)

    def serve() -> list[list[int]]:
        return [g["token_ids"] for g in generate(eng.llm, PROMPTS, a.max_tokens) + generate(eng.llm, PROMPTS[:1], a.max_tokens)]

    plain = serve()
    entry_name = f"pouw:{PH.SCHEME}"
    ctx = PO.Context(stage="serve", enabled=(entry_name,),
                     settings=h1_settings(eng, a.rows, steps_public, a.challenger, a.session, a.gate, retain_steps=a.retain,
                                          transcript_dir=str(outdir), admission="per-round"))
    t0 = time.monotonic()
    comp = PO.install(eng.model, ctx)
    doc["install_s"] = round(time.monotonic() - t0, 1)
    h1 = pouw.h1_run(entry_name)
    layers = PHK.layers(eng.model, pouw.linear_types())
    seen: list[str] = []

    def listener(call: Any, a_: Any, s_a: Any, weight: Any, weight_scale: Any) -> dict[str, Any]:
        ok = (a_.is_cuda and a_.dtype == torch.int8 and tuple(a_.shape) == (call.m, call.k) and s_a.dtype == torch.float32
              and tuple(s_a.shape) == (call.m,) and weight is layers[call.layer].weight
              and weight_scale is layers[call.layer].weight_scale)
        if not ok:
            seen.append(f"{call.index}: {a_.device} {a_.dtype} {tuple(a_.shape)} {s_a.dtype} {tuple(s_a.shape)}")
        return {"listener": "pouw_h1_gpu"}

    h1.listen(listener)
    try:
        t0 = time.monotonic()
        with_h1 = serve()
        doc["serve_s"] = round(time.monotonic() - t0, 1)
        doc["served"] = {"same_tokens": with_h1 == plain, "requests": len(plain), "tokens": sum(map(len, plain))}
        h1.close()

        steps: dict[int, list[int]] = {}
        for c in h1.calls:
            steps.setdefault(c.step, []).append(c.layer)
        manifest = h1.manifest()
        doc["hook"] = {"bad": seen[:20], "bound": all(r.get("bound") == {"listener": "pouw_h1_gpu"} for r in manifest),
                       "steps": len(steps), "steps_in_layer_order": all(v == list(range(len(layers))) for v in steps.values()),
                       "leaves": all(r["leaves"] == [0, a.rows] for r in manifest)}
        print(f"[h1] hook {json.dumps(doc['hook'])}", flush=True)
        doc["calls"] = {"count": len(h1.calls), "units": sum(c.units for c in h1.calls), "u_step": h1.u_step,
                        "static": h1.static(), "refused": h1.refused[:20], "steps_served": h1.details["steps_served"],
                        "last_step_complete": h1.details["last_step_complete"],
                        "served_rows_per_step": [max((c.m for c in h1.calls if c.step == s), default=0) for s in range(h1.steps)],
                        "gate": h1.credit()}
        print(f"[h1] calls {json.dumps(doc['calls'])}", flush=True)
        doc["requests"] = requests_check(h1, len(plain))
        print(f"[h1] requests {json.dumps(doc['requests'])}", flush=True)

        zs = []
        for index, xq, _sa, y in h1.retained:
            call = h1.call(index)
            w = PH.numpy_of(layers[call.layer].weight).astype(np.int64)
            exact = xq.astype(np.int64) @ w.T
            zs.append({"index": index, "shape": [call.m, call.k, call.n], "y_exact": bool(np.array_equal(y, exact))})
            if a.compile:
                u = call.compiled().units[0]
                full = np.zeros(PH.padded(call.rows, call.n), np.int64)
                full[:call.m, :call.n] = exact
                zs[-1]["unit0_z"] = bool(np.array_equal(call.z(y, 0), [full[u.rows[i], u.cols[j]] for i, j in u.tile.pairs]))
        doc["z"] = zs

        doc["shapes"] = [list(x) for x in h1.shapes()]
        if a.compile:
            t0 = time.monotonic()
            doc["compile"], doc["compile_ok"] = _check(h1)
            doc["compile_s"] = round(time.monotonic() - t0, 1)
            print(f"[h1] compile {doc['compile_ok']} in {doc['compile_s']} s over {len(h1.shapes())} shapes", flush=True)
        else:
            doc["compile"], doc["compile_ok"] = "deferred to --compile-from", None
    finally:
        comp.close()
    doc["commitment"] = comp.installed[0].commitment()
    doc["ok"] = bool(doc["served"]["same_tokens"] and not doc["hook"]["bad"] and doc["hook"]["bound"] and doc["calls"]["static"]
                     and doc["requests"]["ok"]
                     and doc["hook"]["steps_in_layer_order"] and doc["hook"]["leaves"]
                     and doc["calls"]["count"] > 0 and zs and all(z["y_exact"] and z.get("unit0_z", True) for z in zs)
                     and doc["compile_ok"] is not False and doc["engine"]["enforce_eager"] is not False)
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"POUW-H1-GPU {'OK' if doc['ok'] else 'FAIL'} calls={doc['calls']['count']} units={doc['calls']['units']} "
          f"shapes={len(doc['shapes'])} compile={doc['compile_ok']} same_tokens={doc['served']['same_tokens']}", flush=True)
    return doc


#: candidates for `fill_prompts`: each word's first token (no BOS) is one prompt
FILL_WORDS = ("The capital France def Water Once In A To Photosynthesis Columbus Light Gold Tea Autumn Square Planet Primary Speed "
              "Village Number Return Import Class If When Why How What Where Who My Our Your Their This That These Those Every "
              "Some Many Most All Each Today Yesterday Tomorrow Science History Music Art Math Code Data River Mountain Ocean "
              "Forest City Market Doctor Teacher Engine Garden Winter Summer Spring Morning Evening Night Story Letter Question "
              "Answer Problem").split()


def fill_prompts(tokenizer: Any, n: int) -> list[int]:
    """`n` distinct one-token prompts: each `FILL_WORDS` word's first token, in order, first occurrence only."""
    out: list[int] = []
    for w in FILL_WORDS:
        t = tokenizer.encode(w, add_special_tokens=False)[0]
        if t not in out:
            out.append(int(t))
    if len(out) < n:
        raise SystemExit(f"only {len(out)} distinct one-token prompts in FILL_WORDS, {n} needed")
    return out[:n]


def generate_static(llm: Any, prompts: list[int], max_tokens: int) -> list[dict[str, Any]]:
    """Greedy one-token prompts under the static schedule (``ignore_eos``: every request runs to its cap)."""
    from vllm import SamplingParams
    outs = llm.generate([{"prompt_token_ids": [t]} for t in prompts],
                        SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=0, ignore_eos=True), use_tqdm=False)
    return [{"prompt": [t], "token_ids": list(o.outputs[0].token_ids), "finish_reason": str(o.outputs[0].finish_reason),
             "num_cached_tokens": getattr(o, "num_cached_tokens", None)} for t, o in zip(prompts, outs)]


def static_check(gen: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
    """`engine.static_schedule.check` of each request: by length at its cap, no prefix-cache read."""
    from verity_vllm.engine import static_schedule
    reqs = [{"request_id": str(i), "max_tokens": max_tokens} for i in range(len(gen))]
    timing = {"requests": {str(i): {"finish_reason": g["finish_reason"], "n_generated": len(g["token_ids"])} for i, g in enumerate(gen)}}
    out = static_schedule.check(reqs, timing, cached_tokens={str(i): g["num_cached_tokens"] for i, g in enumerate(gen)})
    return {"ok": out["ok"], "problems": out["problems"][:20]}


def rate(gen: list[dict[str, Any]], seconds: float, steps: int) -> dict[str, Any]:
    tokens = sum(len(g["token_ids"]) for g in gen)
    return {"sequences": len(gen), "tokens": tokens, "s": round(seconds, 3), "step_s": round(seconds / steps, 4),
            "tokens_per_s": round(tokens / seconds, 1)}


def tokens_digest(gen: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps([g["token_ids"] for g in gen]).encode()).hexdigest()


def fill(a: argparse.Namespace) -> dict[str, Any]:
    from verity_vllm import protocols as PO
    from verity_vllm.engine.build import build_engine
    from verity_vllm.program.frontend.kernel_readback import int8_exact_of
    from verity_vllm.program.frontend.target_profile import TargetProfile
    from verity_vllm.protocols import pouw
    from verity_vllm.protocols.pouw import h1 as PH
    from verity_vllm.protocols.pouw.h1 import hooks as PHK
    from verity_vllm.protocols.pouw.h1 import served as SV

    decl = declared_of(a)
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    target = TargetProfile.from_device().with_overrides(int8_exact=a.int8_exact, batch_invariant=a.batch_invariant)
    doc: dict[str, Any] = {"schema": SCHEMA + "/fill", "argv": sys.argv, "torch": torch.__version__,
                           "device": torch.cuda.get_device_name(0), "scheme": PH.SCHEME,
                           "public": {"slots": decl.slots, "prompt_lens": list(decl.prompt_lens), "max_tokens": decl.max_tokens,
                                      "rows": decl.rows, "steps": decl.steps, "schedule": decl.schedule(), "u": str(decl.u)}}
    with tempfile.TemporaryDirectory() as scratch:
        entry, how = pin(a)
        doc["pin"] = {"repo": entry["repo"], "revision": entry["revision"], "source": how}
        args = {**decl.engine_args(), **a.engine_arg}
        eng = build_engine(scratch_checkpoints(a.checkpoints, entry, Path(scratch)), case=a.role, target=target,
                           max_num_seqs=args.pop("max_num_seqs"), engine_args=args)
    doc["engine"] = {"int8_exact": int8_exact_of(eng.model), "enforce_eager": eng.kwargs.get("enforce_eager"),
                     "effective": eng.effective, "args": {**decl.engine_args(), **a.engine_arg}}
    entry_name = f"pouw:{PH.SCHEME}"
    registry = model_key_of(a.gate, a.model, a.circuit) if a.model else None
    if registry is not None:
        doc["registry"] = {"model": registry.model, "circuit": registry.circuit, "weights_root": registry.weights_root}
        print(f"[fill] {registry.model}'s registration {registry.weights_root[:16]}... under circuit "
              f"{registry.circuit[:16]}...", flush=True)
    # B registered once, before the first window, and its `Weights` reused by every window after
    shared: list[Any] = [None]
    try:
        cfg = eng.llm.llm_engine.vllm_config
        doc["engine"]["scheduler"] = {"max_num_seqs": cfg.scheduler_config.max_num_seqs,
                                      "max_num_batched_tokens": cfg.scheduler_config.max_num_batched_tokens,
                                      "max_model_len": cfg.model_config.max_model_len,
                                      "num_gpu_blocks": cfg.cache_config.num_gpu_blocks, "block_size": cfg.cache_config.block_size}
    except Exception as e:  # noqa: BLE001
        doc["engine"]["scheduler"] = {"error": repr(e)[:300]}
    print(f"[fill] engine {json.dumps(doc['engine'], default=str)[:1500]}", flush=True)

    tok = eng.llm.get_tokenizer()
    prompts = fill_prompts(tok, decl.slots)
    doc["prompts"] = prompts
    lens = tuple(len(tok.encode(p)) for p in SERVE_H1_PROMPTS)
    try:
        doc["serve_h1"] = {"prompt_lens": list(lens), "u": str(PH.Static(lens, decl.max_tokens, decl.rows, decl.steps).u)}
    except ValueError as e:
        doc["serve_h1"] = {"prompt_lens": list(lens), "u": None, "why": str(e)}
    cap, served_steps = decl.max_tokens, decl.max_tokens
    arch = doc["arch"] = a.arch or (SV.card_arch() if a.served else None)
    period = period_program(eng.model, decl.rows, decl.steps, a.j10r, arch) if a.served else None
    if period is not None:
        doc["period"] = {k: v for k, v in period[2].items() if k not in ("shapes", "modules")} | {
            "circuit": period[1], "noise": "j10r" if a.j10r else "v1", "arch": arch}
        print(f"[fill] period {json.dumps(doc['period'])} rss_gib={rss_gib()}", flush=True)
    gpu_peak()
    t0 = time.monotonic()
    plain = generate_static(eng.llm, prompts, cap)
    t1 = time.monotonic()
    plain_gpu = gpu_peak()
    plain_solo = generate_static(eng.llm, prompts[:1], cap)
    t2 = time.monotonic()
    doc["plain"] = {"fill": rate(plain, t1 - t0, served_steps), "solo": rate(plain_solo, t2 - t1, served_steps),
                    "static": static_check(plain + plain_solo, cap), "tokens_digest": tokens_digest(plain),
                    "solo_is_first_of_fill": plain_solo[0]["token_ids"] == plain[0]["token_ids"], "gpu": plain_gpu}
    print(f"[fill] plain {json.dumps(doc['plain'])}", flush=True)
    layers = None
    sessions = window_sessions(a.gate, a.peer, decl.steps) if a.peer else None

    def window(name: str, want: list[dict[str, Any]], tdir: Path, commit: bool = False, edit: str | None = None
               ) -> dict[str, Any]:
        nonlocal layers
        from verity_vllm.engine.hooks import Hooks
        rss0 = rss_gib()
        reserved = sessions(name) if sessions is not None else a.session
        layers = layers or PHK.layers(eng.model, pouw.linear_types())
        com = tap = None
        if commit:
            from verity_vllm.acquire import h1_commit as HC
            com, _src, tap = HC.attach(eng, layers, run_id=f"pouw-h1-commit-{os.getpid()}")
        # in-process coins open no challenge phase, so a local window's key is its own (`Served.key`)
        if commit and a.served and registry is None and shared[0] is None:
            t0 = time.monotonic()
            shared[0] = local_weights(eng.model, layers, decl.rows, decl.steps, period, arch)
            doc["b_registered"] = {"s": round(time.monotonic() - t0, 1), "path": SV.NR.path()}
            print(f"[fill] B registered under a local key {doc['b_registered']}", flush=True)
        served = (SV.Served(prover=a.prover_dir, registry=registry, weights=shared[0], tap=tap,
                            window=None if a.challenger else os.urandom(64), program=period[0], program_digest=period[1],
                            arch=arch)
                  if a.served and name not in ("solo", "baseline") else None)
        registered_s: dict[int, float] = {}
        hashed_s: dict[int, float] = {}
        nr_commit = SV.NR.commit
        prof = None
        if served is not None and not a.handoff:
            served_commit, at = served.commit, [0]
            if a.profile_register:
                import cProfile

                prof = cProfile.Profile()
                served_commit = profiled(prof, served_commit)

            def hashing(*args: Any, **kw: Any) -> Any:
                t = time.monotonic()
                try:
                    return nr_commit(*args, **kw)
                finally:
                    hashed_s[at[0]] = hashed_s.get(at[0], 0.0) + time.monotonic() - t

            def timed(run: Any, call: Any, *rest: Any) -> dict[str, str]:
                t, at[0] = time.monotonic(), call.step
                if call.layer == 0:
                    print(f"[fill] {name} step {call.step} rss_gib={rss_gib()} registered_s(prev)="
                          f"{round(registered_s.get(call.step - 1, 0.0), 2)} hashed_s(prev)="
                          f"{round(hashed_s.get(call.step - 1, 0.0), 2)} values={len(served.registered)}", flush=True)
                try:
                    return served_commit(run, call, *rest)
                finally:
                    registered_s[call.step] = registered_s.get(call.step, 0.0) + time.monotonic() - t
            served.commit = timed
            SV.NR.commit = hashing
        ctx = PO.Context(stage="serve", enabled=(entry_name,),
                         settings=h1_settings(eng, decl.rows, decl.steps, a.challenger, reserved, a.gate, retain_steps=a.retain,
                                              transcript_dir=str(tdir), admission="round-0", served=served))
        t0 = time.monotonic()
        comp = PO.install(eng.model, ctx)
        out: dict[str, Any] = {"install_s": round(time.monotonic() - t0, 1), "host": host_memory()}
        print(f"[fill] {name} installed install_s={out['install_s']} rss_gib={rss_gib()}", flush=True)
        if served is not None and served.weights is not None:
            shared[0] = served.weights
        lie, ed = Hooks(f"roots1-{edit}"), None
        if edit is not None:
            out["edit"], ed = edit, editor(edit)
            for m in layers:
                lie.wrap(m.quant_method, "observe", lambda orig: edited(orig, ed))
        h1 = pouw.h1_run(entry_name)
        made = receipts_producer(a.receipts, served, h1, layers, decl.steps) if a.receipts and served is not None else None
        if served is not None:
            out["period_is_runs"] = list(h1.shapes_) == period[2]["shapes"] and dict(h1.modules) == period[2]["modules"]
        bad: list[str] = []
        stamps: dict[int, float] = {}
        serving = [True]

        def listener(call: Any, a_: Any, s_a: Any, weight: Any, weight_scale: Any) -> dict[str, Any]:
            if serving[0] and call.layer == 0:
                stamps[call.step] = time.monotonic()
            ok = (a_.is_cuda and a_.dtype == torch.int8 and tuple(a_.shape) == (call.m, call.k) and s_a.dtype == torch.float32
                  and tuple(s_a.shape) == (call.m,) and weight is layers[call.layer].weight
                  and weight_scale is layers[call.layer].weight_scale)
            if not ok:
                bad.append(f"{call.index}: {a_.device} {a_.dtype} {tuple(a_.shape)} {s_a.dtype} {tuple(s_a.shape)}")
            return {"listener": "pouw_h1_gpu"}

        h1.listen(listener)
        try:
            gpu_peak()
            t0 = time.monotonic()
            got = generate_static(eng.llm, prompts[:len(want)], cap)
            end = time.monotonic()
            serving[0] = False
            if ed is not None and ed.at is not None:
                out["edited"] = ed.at
            out["gpu"] = gpu_peak()
            out.update(served_measures(got, want, cap, served_steps, t0, end, stamps))
            if served is not None and not a.handoff:
                out["registered_s"] = [round(registered_s.get(s, 0.0), 3) for s in range(decl.steps)]
                out["hashed_s"] = [round(hashed_s.get(s, 0.0), 3) for s in range(decl.steps)]
                out["register_path"] = SV.NR.path()
                del served.commit
                SV.NR.commit = nr_commit
                if prof is not None:
                    write_profile(prof, tdir / "register-profile.txt")
            print(f"[fill] {name} served step_s={out['step_s']} registered_s={out.get('registered_s')} "
                  f"hashed_s={out.get('hashed_s')} register_path={out.get('register_path')} rss_gib={rss_gib()}", flush=True)
            if com is not None:
                from verity_vllm.protocols.pouw.h1 import commit as CT
                rc = com.finalize()
                out["committer"] = {**CT.committed(com, rc, ("values_retained_bytes", "expected_tensors", "committed_tensors",
                                                             "errors_n", "errors")),
                                    "finalize_s": round(rc.t_finalize_s, 3)}
            if a.commit:
                print(f"[fill] {name} served {json.dumps({k: out.get(k) for k in ('gpu', 'step_s', 'decode_step_s_median', 'serve', 'same_tokens', 'committer')}, default=str)[:3000]}",
                      flush=True)
            if a.handoff:
                comp.installed[0].release = None
            else:
                h1.close()
                out["close_s"] = round(time.monotonic() - end, 1)
                closed_measures(h1, out, len(want), lambda w: layers[w].weight, bad, decl)
                if made is not None:
                    t1 = time.monotonic()
                    out["receipts"] = {**receipts_check(made), "check_s": round(time.monotonic() - t1, 1)}
                    print(f"[fill] {name} receipts {json.dumps(out['receipts'])}", flush=True)
                    out["ok"] = bool(out["ok"] and out["receipts"]["ok"])
                if tap is not None:
                    out["roots1"] = h1.details["roots1"]
                    out["ok"] = bool(out["ok"] and out["roots1"]["tied"])
                    if a.live_dump and out["requests"].get("live"):
                        from verity_vllm.protocols.pouw.h1 import live as LV
                        out["live_dump"] = LV.dump(h1, Path(a.live_dump) / name)
        finally:
            SV.NR.commit = nr_commit
            lie.uninstall()
            comp.close()
            if com is not None:
                com.close()
                tap.untap()
        if a.handoff:
            out["hook_bad"] = bad[:20]
            return out
        out = released(name, h1, out, served, tdir, bool(a.challenger), register=not (a.commit or a.skip_register))
        if sessions is not None:
            out["session"] = session_end(a.gate, reserved)
            out["ok"] = bool(out["ok"] and out["session"]["ok"])
            print(f"[fill] {name} session {json.dumps(out['session'])}", flush=True)
        if tap is not None:
            out["core"] = core_tie({"commitment": dict(h1.commits), "calls": h1.manifest()}, h1.served.program, tdir)
        if a.credit and edit is None and tap is not None and made is not None:
            out["credit"] = credit_check(h1, served, made)
            out["ok"] = bool(out["ok"] and out["credit"]["ok"])
        out["host"] = {"install": out["host"], "end": host_memory()}
        print(f"[fill] {name} core {json.dumps(out.get('core'), default=str)[:2000]} host {json.dumps(out['host'])}",
              flush=True)
        out["rss_gib"] = {"start": rss0, "end": rss_gib(),
                          "peak": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}
        return out

    if a.baseline_h1:
        doc["h1_baseline"] = window("baseline", plain, outdir / "baseline")
    if a.commit:
        if a.period:
            from verity_vllm.protocols.pouw.h1 import challenger as PHC
            from verity_vllm.protocols.pouw.h1 import served as SV
            layout = SV.declared_layout(period[2]["shapes"], period[2]["modules"], decl.rows, decl.steps)
            draw_k, files = PHC.draw_k(layout), period_files(Path(a.circuits), period[1], layout, decl)
            doc["period_gate"] = {"name": a.period, "length": fixed_length(decl), "draw_k": draw_k, "circuit_files": files,
                                  "open": period_verb(a.gate, period_open_argv(a.period, period[1], draw_k, decl, files))}
            print(f"[gate] period {json.dumps(doc['period_gate'], default=str)}", flush=True)
        doc["h1_commit"] = window("commit", plain, outdir / "commit", commit=True)
        if a.period:
            doc["period_gate"]["close"] = period_verb(a.gate, ["period-close", "--name", a.period])
            print(f"[gate] period {a.period} closed {json.dumps(doc['period_gate']['close'], default=str)}", flush=True)
        (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
        if a.negatives:
            doc["negatives"] = {}
            for edit in negatives_of(layers):
                got = window(f"commit-{edit}", plain, outdir / f"commit-{edit}", commit=True, edit=edit)
                doc["negatives"][edit] = {**negative_verdict(got), "window": got}
                gc.collect()
                (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    doc["h1"] = window("fill", plain, outdir)
    if a.handoff:
        return hand_off(pouw.h1_run(entry_name), a, doc)
    if a.commit:
        doc["h1_solo"] = {"skipped": "--commit measures the fill window with and without the Commit"}
        h, c = doc["h1"], doc["h1_commit"]
        negs = {k: {x: v[x] for x in ("refused", "arow", "roots1", "edited")} for k, v in (doc.get("negatives") or {}).items()}
        doc["host"] = host_memory()
        doc["commit"] = {"step_s": {"plain_mean": doc["plain"]["fill"]["step_s"], "h1_decode_median": h["decode_step_s_median"],
                                    "h1_commit_decode_median": c["decode_step_s_median"], "h1_mean": h["serve"]["step_s"],
                                    "h1_commit_mean": c["serve"]["step_s"], "h1_commit_steps": c["step_s"]},
                         "gpu": {"plain": doc["plain"]["gpu"], "h1": h["gpu"], "h1_commit": c["gpu"]},
                         "host": doc["host"], "window": doc["public"],
                         "layout_static": c.get("layout_static"), "same_tokens": c["same_tokens"],
                         "roots1": c.get("roots1"), "core": c.get("core"), "negatives": negs,
                         "committer": c.get("committer"), "coins": coins_summary(doc, bool(a.challenger)), "receipts": c.get("receipts"),
                         "credit": {k: (c.get("credit") or {}).get(k) for k in ("ok", "lines", "units", "s")},
                         "ok": bool(c["ok"] and (c.get("core") or {}).get("tied")
                                    and all(v["refused"] for v in negs.values())
                                    and len(negs) == len(negatives_of(layers)) * a.negatives
                                    and coins_summary(doc, bool(a.challenger))["ok"])}
        print(f"[fill] commit {json.dumps(doc['commit'], default=str)[:3000]}", flush=True)
    elif a.challenger and not a.peer:
        doc["h1_solo"] = {"skipped": "the challenger serves one session, the 64's window"}
    elif a.baseline_h1:
        doc["h1_solo"] = {"skipped": "--baseline-h1 serves the H1 window without Served in its place"}
    else:
        try:
            doc["h1_solo"] = window("solo", plain_solo, outdir / "solo")
        except Exception as e:  # noqa: BLE001
            doc["h1_solo"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:2000]}
    return fill_record(doc, outdir)


def profiled(prof: Any, fn: Any) -> Any:
    """``fn`` run under the profiler ``prof``."""
    def run(*args: Any, **kw: Any) -> Any:
        prof.enable()
        try:
            return fn(*args, **kw)
        finally:
            prof.disable()
    return run


def write_profile(prof: Any, path: Path) -> None:
    """``prof``'s 60 functions of most cumulative time, then of most own time, as text at ``path``."""
    import io
    import pstats

    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    stats = pstats.Stats(prof, stream=buf).strip_dirs()
    stats.sort_stats("cumulative").print_stats(60)
    stats.sort_stats("tottime").print_stats(60)
    path.write_text(buf.getvalue())


def rss_gib() -> float:
    """This process's resident set now, GiB."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return round(int(line.split()[1]) / 2**20, 2)
    return -1.0


def served_measures(got: list[dict[str, Any]], want: list[dict[str, Any]], cap: int, served_steps: int, t0: float, end: float,
                    stamps: dict[int, float]) -> dict[str, Any]:
    """What only the serving process sees of a fill window: its rate, each step's seconds (at linear 0 of the next), its
    tokens against the plain run's and the static schedule's check."""
    step_s = [round((stamps[s + 1] if s + 1 in stamps else end) - stamps[s], 4) for s in sorted(stamps)]
    return {"serve": rate(got, end - t0, served_steps), "step_s": step_s, "served_end_unix": round(time.time(), 3),
            "decode_step_s_median": float(np.median(step_s[1:])) if len(step_s) > 1 else None,
            "same_tokens": [g["token_ids"] for g in got] == [g["token_ids"] for g in want], "tokens_digest": tokens_digest(got),
            "static_schedule": static_check(got, cap)}


def closed_measures(h1: Any, out: dict[str, Any], want: int, weight_of: Any, bad: list[str], decl: Any = None) -> None:
    """A fill window's measures once closed (`want` requests served, linear w's weight `weight_of(w)`, the listener's
    complaints `bad`, `decl` its declaration, default `pouw_h1.FILL`), into `out`, and its ``ok`` before its release."""
    from verity_vllm.protocols.pouw import h1 as PH
    decl = decl or PH.FILL
    cap = decl.max_tokens
    rows = h1.step_rows()
    decl_w = decl if want == decl.slots else PH.Static(decl.prompt_lens[:want], cap, decl.rows, decl.steps)
    zs = []
    for index, xq, _sa, y in h1.retained:
        call = h1.call(index)
        w = PH.numpy_of(weight_of(call.layer)).astype(np.int64)
        zs.append({"index": index, "shape": [call.m, call.k, call.n], "y_exact": bool(np.array_equal(y, xq.astype(np.int64) @ w.T))})
    credit = h1.credit()
    out.update({
        "step_rows": rows, "declared_rows": decl_w.schedule(),
        "schedule_problems": decl_w.check(rows), "u": str(h1.utilization()), "u_float": float(h1.utilization()),
        "u_declared": str(decl_w.u), "layout_static": h1.static(), "refused": h1.refused[:20], "calls": len(h1.calls),
        "units": sum(c.units for c in h1.calls), "u_step": h1.u_step, "hook_bad": bad[:20],
        "steps_in_layer_order": all([c.layer for c in h1.calls if c.step == s] == list(range(len(h1.shapes_)))
                                    for s in range(h1.steps)),
        "z": zs, "shapes": [list(x) for x in h1.shapes()], "requests": requests_check(h1, want),
        "gate": {"gate_passed": credit["gate_passed"], "refused": credit["refused"], "padded": credit["padded"],
                 "units": credit["units"], "served_frac_of_macs": credit["served_frac_of_macs"]}})
    out["ok"] = bool(out["same_tokens"] and out["static_schedule"]["ok"] and not out["schedule_problems"]
                     and out["u"] == out["u_declared"] and out["layout_static"] and not bad and out["steps_in_layer_order"]
                     and zs and all(z["y_exact"] for z in zs) and out["requests"]["ok"]
                     and out["requests"]["per_round"][0] == want)


def released(name: str, h1: Any, out: dict[str, Any], served: Any, tdir: Path, challenger: bool,
             register: bool = True) -> dict[str, Any]:
    """A fill window's commitment and how its challenge phase closed, once released, and its served trees' check (with
    ``register``, the first and last calls' ``--registered`` files too)."""
    out["commitment"] = dict(h1.commits)
    out["coin_source"], out["challenger"] = out["commitment"].get("coin_source"), h1.details.get("challenger")
    if challenger:
        out["ok"] = bool(out["ok"] and out["challenger"] == {"closed": True})
    if served is not None:
        out["served"] = served_check(h1, served, tdir, register)
        out["ok"] = bool(out["ok"] and out["served"]["committed"])
    print(f"[fill] {name} {json.dumps({k: v for k, v in out.items() if k not in ('commitment', 'z')}, default=str)[:3000]}",
          flush=True)
    return out


#: the serving phase's record when it hands its window off (`hand_off`), which the close phase completes
SERVING = "pouw_h1_gpu.serving.json"


def hand_off(h1: Any, a: argparse.Namespace, doc: dict[str, Any]) -> dict[str, Any]:
    """With ``--handoff SOCK --handoff-state DIR``: the served window saved to DIR and its challenge phase's stream handed to the
    holder at SOCK (`pouw_h1_handoff`), its record so far in ``SERVING``; this process then exits, and with it its GPU
    lease, and ``--close-from DIR`` closes and releases the window on the CPU (`fill_close`)."""
    from verity_vllm.protocols.pouw.h1 import handoff as HO
    t0 = time.monotonic()
    HO.suspend(h1, a.handoff_state)
    h = doc["h1"]
    h["handoff"] = {"requests": len(doc["prompts"]), "suspend_s": round(time.monotonic() - t0, 1)}
    h["ok"] = bool(h["same_tokens"] and h["static_schedule"]["ok"] and not h["hook_bad"])
    doc["ok"] = bool(doc["plain"]["static"]["ok"] and h["ok"] and doc["engine"]["enforce_eager"] is not False)
    (Path(a.out) / SERVING).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    HO.send_stream(HO.client_of(h1), a.handoff)
    h["handoff"]["handed_unix"] = round(time.time(), 3)
    print(f"POUW-H1-SERVED {'OK' if doc['ok'] else 'FAIL'} steps={len(h['step_s'])} same_tokens={h['same_tokens']} "
          f"suspend_s={h['handoff']['suspend_s']}: the window and its session's stream are handed off", flush=True)
    return doc


def fill_close(a: argparse.Namespace) -> dict[str, Any]:
    """The close phase, on no GPU: the window ``--close-from`` holds, its challenge phase the stream ``--stream-fd``, closed and
    released as the serving process would have (`pouw_h1_handoff`), and the fill record completed from ``SERVING``."""
    from verity_vllm.protocols.pouw.h1 import handoff as HO
    from verity_vllm.protocols.pouw.h1 import window as PHW
    outdir = Path(a.out)
    doc = json.loads((outdir / SERVING).read_text())
    doc["close_argv"], h = sys.argv, doc["h1"]
    t0 = time.monotonic()
    h1, client = HO.resume(a.close_from, a.stream_fd)
    h["handoff"]["resume_s"] = round(time.monotonic() - t0, 1)
    bad = list(h["hook_bad"])

    def listener(call: Any, a_: Any, s_a: Any, weight: Any, weight_scale: Any) -> dict[str, Any]:
        if not (call.m == 0 and a_.dtype == torch.int8 and tuple(a_.shape) == (0, call.k) and s_a.dtype == torch.float32
                and tuple(s_a.shape) == (0,) and weight is h1._weights[call.layer][0]):
            bad.append(f"{call.index}: {a_.dtype} {tuple(a_.shape)} {s_a.dtype} {tuple(s_a.shape)} at the close")
        return {"listener": "pouw_h1_gpu"}

    h1.listen(listener)
    t0 = time.monotonic()
    try:
        h1.close()
        h["close_s"] = round(time.monotonic() - t0, 1)
        closed_measures(h1, h, h["handoff"]["requests"], lambda w: h1._weights[w][0], bad)
    finally:
        PHW.finish(h1, client)
    h["release_s"] = round(time.monotonic() - t0 - h.get("close_s", 0), 1)
    doc["h1"] = released("fill", h1, h, h1.served, outdir, True)
    doc["h1_solo"] = {"skipped": "the challenger serves one session, the 64's window"}
    return fill_record(doc, outdir)


def gpu_peak() -> dict[str, float]:
    """The device's peak allocated and reserved GiB since the last call (which resets the peak), and its used GiB now."""
    free, total = torch.cuda.mem_get_info()
    out = {"max_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
           "max_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
           "device_used_gib": round((total - free) / 2**30, 3), "device_total_gib": round(total / 2**30, 3)}
    torch.cuda.reset_peak_memory_stats()
    return out


def host_memory() -> dict[str, float | None]:
    """This process's peak and current resident GiB (``VmHWM``, ``VmRSS``)."""
    got = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        k, _, v = line.partition(":")
        if k in ("VmHWM", "VmRSS"):
            got[k] = round(int(v.split()[0]) / 2**20, 2)
    return {"peak_gib": got.get("VmHWM"), "rss_gib": got.get("VmRSS")}


def declared_of(a: argparse.Namespace) -> Any:
    """The fill workload's declaration: `pouw_h1.FILL`, or with ``--declare SLOTS,CAP,T,S`` SLOTS one-token prompts at
    cap CAP, T rows a call and S steps (`pouw_h1.Static`)."""
    from verity_vllm.protocols.pouw import h1 as PH
    if not a.declare:
        return PH.FILL
    slots, cap, rows, steps = a.declare
    return PH.Static((1,) * slots, cap, rows, steps)


def local_weights(model: Any, layers: list[Any], rows: int, steps: int, period: Any = None, arch: str | None = None) -> Any:
    """The served model's ``b/<K>`` registered once under a fresh local key and the Program install declares
    (`pouw._install_h1`'s shapes and names, on the serving card's ``arch``; ``period``'s, from `period_program`, when
    given), for a window with no auditor's registration (no ``--model``)."""
    from verity_vllm.protocols import pouw
    from verity_vllm.protocols.pouw.h1 import served as SV
    shapes = [tuple(int(d) for d in m.weight.shape) for m in layers]
    modules = {w: f"linear {w}" for w in range(len(layers))} | pouw._named(model, layers)
    given = {} if period is None else {"program": period[0], "program_digest": period[1]}
    circuit = SV.Served(arch=arch, **given).declare(shapes, modules, rows, steps)
    return SV.register_weights(circuit, shapes, SV.weights_of(layers), os.urandom(32))


#: the windows ``--negatives`` serves with the Commit in the engine, each handing H1 A codes other than the ones the
#: Commit acquires (`test_pouw_h1_commit`'s edits, here on the served engine)
NEGATIVES = ("a-zero", "a-row-off")
#: the window ``--negatives`` adds on a model with routed experts: one expert call's codes all zero
EXPERT_NEGATIVE = "a-zero-expert"
#: a window's core evidence for ``--core-from`` (`core_tie`)
CORE_EVIDENCE = "core_evidence.json.gz"


def negatives_of(layers: list[Any]) -> tuple[str, ...]:
    """The windows ``--negatives`` serves over H1's linears ``layers``: `NEGATIVES`, and `EXPERT_NEGATIVE` when some are
    routed experts (`pouw_h1_hooks.expert_names`)."""
    from verity_vllm.protocols.pouw.h1 import hooks as PHK
    return NEGATIVES + ((EXPERT_NEGATIVE,) if PHK.expert_names(layers) else ())


def editor(kind: str) -> Any:
    """The codes H1 is handed of the ones the arm quantized: every call's all zero (``a-zero``, the A = 0 probe),
    row 1's first code moved off by one in the first call of two rows or more (``a-row-off``), or the first routed
    expert down call with rows all zero (``a-zero-expert``: its K, the intermediate width, is the experts' alone). The
    edit's ``at`` names the expert site it zeroed."""
    from verity_vllm.engine import int8_experts as X
    done = [False]

    def edit(layer: Any, xq: Any) -> Any:
        if kind == "a-zero":
            return torch.zeros_like(xq)
        if kind == EXPERT_NEGATIVE:
            if done[0] or not xq.shape[0] or not isinstance(layer, X.ExpertSite) or layer.projection != X.DOWN:
                return xq
            done[0], edit.at = True, layer.name
            return torch.zeros_like(xq)
        if done[0] or xq.shape[0] < 2:
            return xq
        done[0], xq = True, xq.clone()
        xq[1, 0] = 1 if int(xq[1, 0]) != 1 else 2
        return xq
    edit.at = None
    return edit


def edited(orig: Any, edit: Any) -> Any:
    """A layer's ``observe`` (H1's) handed ``edit`` of the arm's codes; the served y and z/y records are the arm's own."""
    def observe(layer: Any, xq: Any, s_a: Any, y: Any, z: Any = None) -> Any:
        return orig(layer, edit(layer, xq), s_a, y, z)
    return observe


def core_tie(record: dict[str, Any], program: Any, tdir: Path) -> dict[str, Any]:
    """`core_verdict` of a window's own evidence: its record (commitment and manifest, `pouw_h1.CALLS_FILE`'s) and its
    declared Program's descriptor (`pouw_h1_served.descriptor`), saved to ``<tdir>/CORE_EVIDENCE`` for ``--core-from``."""
    from verity_vllm.protocols.pouw.h1 import served as SV
    c = record["commitment"]
    ev = {"commitment": c, "calls": record["calls"], "descriptor": SV.descriptor(program).hex()}
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / CORE_EVIDENCE).write_bytes(gzip.compress(json.dumps(ev).encode()))
    return core_verdict(ev)


def record_reason(record: dict[str, Any]) -> str | None:
    """`pouw_h1_challenger.manifest_reason` of a window's record, its Program salt read from in-process coins too: their
    coin commitment is `coins.CoinCommitment`'s rather than the live challenger's, and their salt coin opens after the
    weights' root `pouw` keys it on (`audit.Epoch.start`'s, the opening's ``after``), not the served one."""
    from verity.core.protocols.pouw.audit import salt_of
    from verity.core.protocols.pouw.coins import CoinCommitment, CoinOpening
    from verity_vllm.engine.hooks import patched
    from verity_vllm.protocols.pouw import h1 as PH
    from verity_vllm.protocols.pouw.h1 import challenger as PHC
    c = record["commitment"]
    if c.get("coin_source") == PH.CHALLENGER:
        return PHC.manifest_reason(record)
    try:
        opening = CoinOpening.from_json(c["salt_opening"])
        salt = PH.program_salt(salt_of(CoinCommitment.from_json(c["coins"]), opening.coin, opening.after))
    except (KeyError, TypeError, ValueError) as e:
        return f"the window's in-process salt can't be derived from its record ({e})"
    with patched(PHC, "program_salt_of", lambda _c: salt):
        return PHC.manifest_reason(record)


def core_verdict(ev: dict[str, Any]) -> dict[str, Any]:
    """Core credit's ROOTS1 steps over a window's evidence, as `credit.window` runs them once its challenger steps pass
    (a window under in-process coins has none to pass): the record one manifest under calls_root with the Commit's record
    (`record_reason`), each A table and input-ids table committed whole by its round's root (`credit._a_link`), and each
    the Commit's value of its rows (`credit._arow`), each at the rows the layout gives it (`credit.table_rows`: n chunk
    leaves a row past K = 4096). ``tied`` when none refuses."""
    from verity.core.protocols.pouw import credit as CR
    c, raw = ev["commitment"], bytes.fromhex(ev["descriptor"])
    entries, rounds = (c.get("served") or {}).get("entries") or [], c.get("rounds")
    layout, why = CR.layout_of(raw)
    got, cwhy = CR.commit_of(c.get("commit"))
    rows = CR.table_rows(layout, len(rounds), CR._prefix(raw)) if why is None and isinstance(rounds, list) else None
    out: dict[str, Any] = {"manifest": record_reason({"commitment": c, "calls": ev["calls"]}),
                           "a_link": CR._a_link(rounds, entries, len(ev["calls"]), rows), "layout": why, "commit": cwhy}
    out["arow"] = None if why or cwhy else CR._arow(rounds, entries, got[1], layout, CR._prefix(raw))
    out["tied"] = all(v is None for v in out.values())
    out["commit_values"] = len(got[1]) if got else None
    return out


def negative_verdict(out: dict[str, Any]) -> dict[str, Any]:
    """A negative window's outcome: refused by core at ``arow`` (an A table not the Commit's value of its rows), the window
    otherwise one manifest whose A tables its round roots commit, its Commit honest (its tap refused nothing and
    registered trees of its own for the edited rows), and untied by the serving process (``details.roots1``)."""
    core, r1 = out.get("core") or {}, out.get("roots1") or {}
    arow = core.get("arow")
    refused = bool(arow and arow[0] == "arow" and "is not the Commit's" in arow[1] and core.get("manifest") is None
                   and core.get("a_link") is None and r1.get("reason") and r1.get("own_trees"))
    return {"refused": refused, "arow": arow, "roots1": r1.get("reason"), "own_trees": r1.get("own_trees"),
            "edited": out.get("edited"),
            "manifest": core.get("manifest"), "a_link": core.get("a_link"), "same_tokens": out.get("same_tokens")}


def coins_summary(doc: dict[str, Any], challenger: bool) -> dict[str, Any]:
    """A ``--commit`` run's coins, window by window (the Commit window, each negative, the fill window): its coin source,
    its challenge phase and how that ended, and its credit gate (`pouw_h1.H1Run.credit`): the gate-passed rows, the refused calls
    with each reason, and the credited share, gate-passed MACs over served MACs. ``ok`` without ``challenger``; with it,
    when every window's coins are the auditor host's from a challenge phase of its own that ended closed and complete, and the
    honest Commit window's gate passes rows and refuses none of its calls."""
    wins = {"commit": doc.get("h1_commit"), **{f"commit-{k}": v.get("window") for k, v in (doc.get("negatives") or {}).items()},
            "fill": doc.get("h1")}
    rows: dict[str, dict[str, Any]] = {}
    for name, w in wins.items():
        if not w:
            continue
        g, s = w.get("gate") or {}, w.get("session") or {}
        passed, refused = g.get("gate_passed") or {}, g.get("refused") or {}
        served = (passed.get("macs") or 0) + (refused.get("macs") or 0)
        rows[name] = {"coin_source": w.get("coin_source"), "session": s.get("session"), "session_ok": s.get("ok"),
                      "gate_passed_rows": passed.get("rows"), "refused_calls": refused.get("calls"),
                      "credited_share": round((passed.get("macs") or 0) / served, 6) if served else None,
                      "reasons": refused.get("reasons")}
    sessions = [r["session"] for r in rows.values()]
    c = rows.get("commit") or {}
    ok = not challenger or bool(rows and all(r["coin_source"] == "challenger" and r["session_ok"] for r in rows.values())
                                and None not in sessions and len(set(sessions)) == len(sessions)
                                and c.get("refused_calls") == 0 and c.get("gate_passed_rows"))
    return {"windows": rows, "ok": ok}


def core_from(directory: str, out: str) -> dict[str, Any]:
    """No GPU: `core_verdict` of each window's saved evidence under `directory` (``*/CORE_EVIDENCE``)."""
    doc: dict[str, Any] = {"schema": SCHEMA + "/core", "argv": sys.argv, "windows": {}}
    for f in sorted(Path(directory).glob(f"*/{CORE_EVIDENCE}")):
        doc["windows"][f.parent.name] = core_verdict(json.loads(gzip.decompress(f.read_bytes())))
    w = doc["windows"]
    doc["ok"] = bool(w.get("commit", {}).get("tied") and all(
        not v["tied"] and (v["arow"] or ("",))[0] == "arow" and v["manifest"] is None and v["a_link"] is None
        for k, v in w.items() if k.startswith("commit-")))
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "pouw_h1_core.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"POUW-H1-CORE {'OK' if doc['ok'] else 'FAIL'} " + " ".join(
        f"{k}={'tied' if v['tied'] else (v['arow'] or v['a_link'] or v['manifest'])}" for k, v in w.items()), flush=True)
    return doc


def fill_record(doc: dict[str, Any], outdir: Path) -> dict[str, Any]:
    doc["compile"] = "deferred to --compile-from (out/pouw-h1-calls.json and out/solo/pouw-h1-calls.json)"
    doc["ok"] = bool(doc["plain"]["static"]["ok"] and doc["h1"]["ok"] and doc["engine"]["enforce_eager"] is not False
                     and (doc.get("commit") or {}).get("ok", True))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    h = doc["h1"]
    print(f"POUW-H1-FILL {'OK' if doc['ok'] else 'FAIL'} u={h['u']} rows={h['step_rows']} same_tokens={h['same_tokens']} "
          f"gate_passed_rows={h['gate']['gate_passed']['rows']} refused_calls={h['gate']['refused']['calls']} "
          f"step_s(64)={h['decode_step_s_median']} step_s(1)={doc['h1_solo'].get('decode_step_s_median')} "
          f"tok/s(64)={h['serve']['tokens_per_s']} tok/s(1)={doc['h1_solo'].get('serve', {}).get('tokens_per_s')} "
          f"digest={h['tokens_digest'][:16]}", flush=True)
    return doc


def layout_of(linears: Any, calls: Any, m: int, tile: Any) -> dict[str, Any]:
    """A window Program's layout as a record keeps it: its linears, A slot and tile, and the SHA-512 of its calls."""
    return {"linears": [list(x) for x in linears], "m": int(m), "tile": list(tile),
            "calls_sha512": hashlib.sha512(json.dumps([list(c) for c in calls]).encode()).hexdigest()}


def served_check(h1: Any, served: Any, tdir: Path, register: bool = True) -> dict[str, Any]:
    """The window's registered tables (`pouw_h1_served`): each committed value's entry, what was refused, and the outcome of
    `pouw_h1_served.register`, the first and last calls' ``--registered`` files written into ``tdir``: a file holds one
    entry a proof block of the window for each of its ports (`h1window.table_reads_file`), so at FILL's 1.5M proof blocks each call's is
    hundreds of MB, and the window's 2304 about a TB. Without ``register`` (``--commit``'s windows, ``--skip-register``)
    only the entries: at FILL `register`'s template partition (`partition.evaluate`) encodes the window Program again,
    past 197 GB and 17 min."""
    from verity_vllm.protocols.pouw.h1 import served as SV
    held = sum(len(r.bcs) * 128 + 64 * sum(map(len, r.levels)) for r in served.registered.values())
    doc: dict[str, Any] = {"entries": served.entries(), "refused": served.refused[:20], "held_mib": round(held / 2**20, 1),
                           "rows": sum(r.leaves for r in served.registered.values()),
                           "committed": bool(served.registered) and not served.refused,
                           "in_commitment": h1.commits.get("served") == {"entries": served.entries(),
                                                                         "refused": list(served.refused)},
                           "layout": layout_of(served.linears, served.calls, served.m, served.tile)}
    print(f"[fill] served committed={doc['committed']} values={len(doc['entries'])} rows={doc['rows']} "
          f"held_mib={doc['held_mib']} rss_gib={rss_gib()}", flush=True)
    calls = sorted({0, len(served.calls) - 1}) if served.calls else []
    if not register:
        doc["register"] = {"skipped": "--commit or --skip-register: no --registered files"}
        return doc
    try:
        reg = SV.register(h1, calls=calls, directory=tdir)
        doc["register"] = {"registered": len(reg.registered), "files": len(reg.files), "calls": calls}
    except ValueError as e:
        doc["register"] = {"refused": str(e)[:2000]}
    print(f"[fill] served committed={doc['committed']} values={len(doc['entries'])} register={json.dumps(doc['register'])[:600]}",
          flush=True)
    return doc


PRODUCER = Path(__file__).resolve().parents[4] / "benchmarks" / "pouw" / "served_h1" / "producer.py"


@functools.cache
def producer() -> Any:
    """Kernels' producer module (`PRODUCER`), loaded by path: it is a benchmark's, outside this package."""
    spec = importlib.util.spec_from_file_location("served_h1_producer", PRODUCER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def receipts_producer(so: str, served: Any, h1: Any, layers: Any, steps: int) -> Any:
    """``--receipts SO``: `served`'s receipts (`Served.receipts`) from the producer (`producer().Receipts`) for the window
    `h1`, h1u.cu's proof-block kernel of the -hbc build `so` over each linear's int8 weight in `layers` in place (no second copy
    on the GPU), or with a ``.cubin`` SO (noise.cu's) receipts.py's torch chain over noise.cu's noise, keeping the first
    layer's linears' calls at step 0 and at step `steps` - 1 for `receipts_check`. The noise is the window Program's form,
    v1's or J10-R's (``--j10r``)."""
    P = producer()
    kernel = {"cubin": P.RC.Cubin(so)} if so.endswith(".cubin") else {"units": P.Units(so)}
    made = P.Receipts(served, h1, weight=lambda c: layers[c.layer].weight,
                      keep=lambda c: c.layer < 4 and c.step in (0, steps - 1), **kernel)
    served.receipts = made
    return made


def receipts_check(made: Any) -> dict[str, Any]:
    """The kept calls' receipts (`receipts_producer`) at the first proof block of each row block against `h1tile.reference`
    seeded from the proof block's public row as the verifier builds it, word for word (`producer().check`; a proof block is bad unless
    that row is also the producer's salt, position and roots); ``ok`` needs live and dead row blocks both."""
    rows = []
    for c, kept in sorted(made.kept.items()):
        ncb = kept["b"].shape[0] // 32
        rows += [(c, u, v["live"], v["ok"] and v["pub"])
                 for u, v in producer().check(kept, list(range(0, len(kept["receipts"]), ncb))).items()]
    live = sum(r[2] for r in rows)
    return {"calls": sorted(made.kept), "units": len(rows), "live": live, "dead": len(rows) - live,
            "bad": [[c, u] for c, u, _live, ok in rows if not ok][:20],
            "ok": bool(rows) and all(r[3] for r in rows) and 0 < live < len(rows)}


def credit_check(h1: Any, served: Any, made: Any) -> dict[str, Any]:
    """``--credit``: core credit's statement steps (`pouw_h1_credit.check`) over the honest Commit window with receipts
    on, the drawn proof blocks the first and last of each kept call (`receipts_producer`'s: the first layer's linears at step 0
    and the last step), their operands and receipts the producer's on the serving GPU; then each prover-side negative
    (`pouw_h1_credit.NEGATIVES`) at the first of them."""
    from tests.protocols import pouw_h1_credit as PC
    from verity.core.protocols.pouw import credit as CR
    from verity_vllm.protocols.pouw.h1 import served as SV
    t0 = time.monotonic()
    w = PC.of_run(h1, served)
    units = [u for c in sorted(made.kept) for u in (w.u0(c), w.u0(c) + CR.call_work(w.layout)[c][0] - 1)]
    got = PC.check(w, units, lambda c: (made.kept[c]["a"].numpy(), made.kept[c]["b"].numpy()),
                   lambda c: SV.receipt_words(made.kept[c]["receipts"]), h1._key)
    got["s"], got["lines"] = round(time.monotonic() - t0, 1), PC.lines(got)
    got["weights"] = [{x: e[x] for x in ("value", "leaves", "words")} for e in w.entries
                      if e["value"].startswith(f"{w.prefix}b/")]
    got["kept"] = [{"call": c, "k": w.k_of(c), "units": CR.call_work(w.layout)[c][0]} for c in sorted(made.kept)]
    print(f"[fill] credit weights {json.dumps(got['weights'])} kept {json.dumps(got['kept'])}", flush=True)
    for line in got["lines"]:
        print(f"[fill] {line}", flush=True)
    print(f"POUW-H1-CREDIT {'OK' if got['ok'] else 'FAIL'} units={len(units)} s={got['s']}", flush=True)
    return got


def _check(h1: Any) -> tuple[dict[str, Any], bool]:
    try:
        out = h1.check()
        return out, all(v["tiles"] == ["32x32"] and v["groups"] == [[0]] for v in out.values())
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:4000]}, False


def compile_from(calls_file: str, out: str) -> dict[str, Any]:
    """The record's calls compiled on the host: every shape by `H1Run.check`, the window static, and each call (s, w) at
    s·U + off(w) counted in its shapes' compiled proof blocks."""
    from verity_vllm.protocols.pouw import h1 as PH

    rec = json.loads(Path(calls_file).read_text())
    pub = rec["public"]
    h1 = PH.H1Run(rec["entry"], b"", {}, pub["shapes"], pub["rows"], pub["steps"])
    h1.calls = [PH.Call(c["index"], c["step"], c["layer"], c["module"], c["served"], c["k"], c["n"], c["base"], c["rows"])
                for c in rec["calls"]]
    h1.refused[:] = rec["refused"]
    doc: dict[str, Any] = {"schema": SCHEMA + "/compile", "argv": sys.argv, "calls_file": calls_file, "record": rec["schema"],
                           "commitment": rec["commitment"], "public": pub, "calls": len(h1.calls),
                           "shapes": [list(x) for x in h1.shapes()], "static": h1.static()}
    t0 = time.monotonic()
    doc["compile"], doc["compile_ok"] = _check(h1)
    doc["compile_s"] = round(time.monotonic() - t0, 1)
    bad, total = [], 0
    if doc["compile_ok"]:
        sizes = [len(PH.compiled(k, n, pub["rows"]).units) for n, k in pub["shapes"]]
        offs = [sum(sizes[:w]) for w in range(len(sizes))]
        bad = [c.index for c in h1.calls if c.base != c.step * sum(sizes) + offs[c.layer]]
        total = pub["steps"] * sum(sizes)
    doc["positions"] = {"public": not bad, "bad": bad[:20], "units": total, "committed_units": rec["commitment"].get("units")}
    doc["ok"] = bool(doc["compile_ok"] and doc["static"] and not bad and total == rec["commitment"].get("units"))
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "pouw_h1_compile.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"POUW-H1-COMPILE {'OK' if doc['ok'] else 'FAIL'} calls={doc['calls']} shapes={len(doc['shapes'])} units={total} "
          f"static={doc['static']} in {doc['compile_s']} s", flush=True)
    return doc


def check_challenger(calls_file: str, directory: str, public_sha512: str, out: str, ledger: list[str],
                     window_file: str | None = None, period: str | None = None, descriptor: bytes | None = None,
                     pins: list[str] | None = None, main: list[str] | None = None) -> dict[str, Any]:
    """The verifier's side, on no GPU: the served window's coins (`calls_file`'s commitment) against the challenger's own
    directory under the auditor's key, pinned by the SHA-512 of its public key (`pouw_h1_challenger.pinned_key`), `ledger`
    every head the gate published of the auditor's signed ledger, `pins` every head of the host's pins log (the host's
    ``pins.jsonl``, the gate's ``pins``; none refuses every challenge phase: `pouw_h1_challenger.pins_head`) and `main` the
    auditor's own first-parent history of main (none refuses every challenge phase, as `challenger.pinned_session` refuses
    it). With `window_file` (the window's ``{root}``) `directory` is
    the challenger run's ``out/``, fetched whole, as the challenge phase the longest of the heads, each extending
    the others, opens and ends for the window (`trusted_rounds_reason`); without it the ledger is read as the newest that
    the others extend (`Ledger.load`, core's check), the window held to the challenge phase it opened for its Commit's root
    (`session_reason`), `directory` being that challenge phase's tree as the gate's ``fetch SESSION`` gives it (its ``out/``
    within). The record is held to its coins as one manifest (`pouw_h1_challenger.manifest_reason`), and with `period` the ledger
    must credit the lane's period with the window accepted and presented as its close's windows, its open committing
    ``draw_k`` (`pouw_h1_challenger.period_reason`, core's `credit.period`). Its ``credit`` is core's
    (`pouw_h1_challenger.credit_reason`) over `descriptor`, the bytes of the circuit the period committed
    (`pouw_h1_served.descriptor`; none refuses it)."""
    from verity.core.protocols.sampled_proofs import remote
    from verity_vllm.protocols.pouw import h1 as PH
    from verity_vllm.protocols.pouw.h1 import challenger as PHC

    rec, tree = json.loads(Path(calls_file).read_text()), Path(directory)
    d = tree if window_file is not None else tree / "out"
    sid = (rec["commitment"].get("session") or {}).get("session")
    doc: dict[str, Any] = {"schema": SCHEMA + "/challenger", "argv": sys.argv, "calls_file": calls_file, "directory": directory,
                           "pinned_public_sha512": public_sha512, "ledger": ledger, "pins": pins,
                           "window_file": window_file,
                           "coin_source": rec["commitment"].get("coin_source")}
    key = None
    try:
        key = PHC.pinned_key(d / remote.KEY, public_sha512)
        doc["key"] = {"W": key.W, "H": key.H, "public_sha512": hashlib.sha512(key.public).hexdigest()}
        if window_file is not None:
            doc["window"] = json.loads(Path(window_file).read_text())
            doc["reason"] = PHC.trusted_rounds_reason(rec["commitment"], d, ledger, doc["window"], key, pins=pins,
                                                      main=main)
        else:
            newest = PHC.Ledger.load(ledger, key, pins=pins, main=main)
            doc["reason"] = PHC.session_reason(rec["commitment"], newest, lambda s: tree if s == sid else None)
    except Exception as e:  # noqa: BLE001
        doc["reason"] = f"{type(e).__name__}: {e}"
    doc["manifest"] = PHC.manifest_reason(rec)
    if key is not None:
        win = doc.get("window") or {"root": (rec["commitment"].get("session") or {}).get("root")}
        doc["credit"] = PHC.credit_reason(rec, d, ledger, win, key, descriptor, pins=pins, main=main)
    if period is not None:
        try:
            if key is None:
                raise ValueError("no key the pin names, so the auditor's ledger is unread")
            newest = PHC.Ledger.load(ledger, key, pins=pins, main=main)
            root = (doc.get("window") or {}).get("root") or (rec["commitment"].get("session") or {}).get("root")
            accepted = [sid] if doc["reason"] is None else []
            doc["period"] = {"name": period, "reasons": PHC.period_reason(newest, period, accepted, [root])}
        except Exception as e:  # noqa: BLE001
            doc["period"] = {"name": period, "reasons": [f"{type(e).__name__}: {e}"]}
    for name in (remote.RECORD, remote.CLOSE):
        try:
            doc[name] = json.loads((d / name).read_text())
        except (OSError, ValueError) as e:
            doc[name] = {"error": str(e)}
    doc["rounds"] = len(rec["commitment"].get("rounds") or ())
    doc["ok"] = doc["reason"] is None and doc["manifest"] is None and not (doc.get("period") or {}).get("reasons")
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "pouw_h1_challenger.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"POUW-H1-CHALLENGER {'OK' if doc['ok'] else 'FAIL'} rounds={doc['rounds']} host={doc[remote.RECORD].get('host')} "
          f"reason={doc['reason']} manifest={doc['manifest']} period={doc.get('period')} "
          f"credit={json.dumps(doc.get('credit'), default=str)[:2000]}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="int8 Llama served with PoUW's H1 + fx1 on the eager int8_exact row, checked on the host")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json")
    ap.add_argument("--base-role", default="LLAMA31_8B")
    ap.add_argument("--role", default="LLAMA31_8B_INT8")
    ap.add_argument("--repo", default="verity-int8/Llama-3.1-8B-INT8")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME"))
    ap.add_argument("--at", help="resolved_at of the pin entry, if one is made (UTC, ISO 8601)")
    ap.add_argument("--by", help="resolved_by of the pin entry, if one is made")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--retain", type=int, default=4, help="the last calls whose operands and exact Y are kept and checked")
    ap.add_argument("--rows", type=int, default=64, help="T, the public rows a call: vLLM's max_num_batched_tokens")
    ap.add_argument("--steps", type=int, default=0, help="S, the public steps of the window (default 2 (max tokens + 2): the "
                                                         "batch and the one alone)")
    ap.add_argument("--int8-exact", default="all", choices=["linears", "all"],
                    help="TargetProfile.int8_exact: linears alone for a model with no int8 K scales (a MoE: OLMoE)")
    ap.add_argument("--no-batch-invariant", dest="batch_invariant", action="store_false",
                    help="VLLM_BATCH_INVARIANT=0: GLM-5.3's sparse MLA has no batch-invariant backend (sm_90 or sm_120)")
    ap.add_argument("--engine-arg", metavar="KEY=VALUE", action="append", default=[],
                    help="an LLM() keyword over the workload's (repeatable, the value JSON when it parses): "
                         "hf_overrides={\"num_hidden_layers\": L} serves a model's first L layers")
    ap.add_argument("--compile", action="store_true", help="compile every served shape in this process too")
    ap.add_argument("--compile-from", metavar="CALLS_JSON", help="no GPU: compile a served run's record (pouw-h1-calls.json)")
    ap.add_argument("--challenger", metavar="HOST:PORT", help="the window's salt and round coins from the auditor host's "
                                                              "challenger (benchmarks/live_challenger/run.sh, SCHEME "
                                                              "verity/pouw/epoch-coins/live/v1, FAMILIES "
                                                              "pouw_h1_challenger.families(S)); else held in this process")
    ap.add_argument("--check-challenger", nargs=2, metavar=("CALLS_JSON", "CHALLENGER_DIR"),
                    help="no GPU: a served window's coins against the challenger run's out/ (with --window) or the "
                         "session's tree (needs --key-sha512 and --ledger)")
    ap.add_argument("--key-sha512", help="the SHA-512 of the auditor's challenger public key, in hex")
    ap.add_argument("--ledger", metavar="HEAD", action="append",
                    help="with --check-challenger: a published head of the auditor's signed ledger (infra's windows.jsonl), "
                         "once a head infra labelled")
    ap.add_argument("--pins", metavar="HEAD", action="append",
                    help="with --check-challenger: a published head of the host's pins log (infra's pins.jsonl, "
                         "labelled challenger-pins head), once a head; none refuses every session")
    ap.add_argument("--window", metavar="WINDOW_JSON", help="the window's identity, {root}, its window-open root; "
                                                            "CHALLENGER_DIR then the challenger run's out/, else the "
                                                            "session's tree")
    ap.add_argument("--session", type=json.loads, metavar="JSON",
                    help="with --challenger: the challenger gate's reserve answer, {\"session\", \"seq\", \"peer\", "
                         "\"by\", \"expires_utc\", \"period\"}")
    ap.add_argument("--gate", type=lambda v: gate_command(shlex.split(v)), metavar="CMD",
                    help="with --challenger: the command the gate's verbs follow (ssh -i KEY -o BatchMode=yes "
                         "research@HOST), which opens the reserved session on the window's open root")
    ap.add_argument("--peer", metavar="NAME", help="fill with --challenger and --gate, in place of --session: each window "
                                                   "the run serves reserves a session of its own at the gate (reserve --peer "
                                                   "NAME), NAME this node's name in the gate's peers, and the next window's "
                                                   "is reserved once that session has ended (window_sessions)")
    ap.add_argument("--period", metavar="NAME", help="with --check-challenger: the lane's period the window is one of, which "
                                                     "the ledger must credit with it accepted; with --commit and --peer: "
                                                     "the period the run opens at its Commit window (period_open_argv, "
                                                     "with --length) and closes once that window's session has ended")
    ap.add_argument("--circuits", metavar="DIR", help="with --commit and --period: the period's circuit files, "
                                                      "pouw_h1_circuits --live T,S's directory, whose manifest's files "
                                                      "the period-open commits (--circuit-files, period_files)")
    ap.add_argument("--descriptor", metavar="PROGRAM", help="with --check-challenger: the period's declared window Program "
                                                            "(the rehearsal's window-program.json.gz), whose bytes core's "
                                                            "credit reads the window's layout from")
    ap.add_argument("--main", metavar="FILE", help="with --check-challenger: the auditor's own `git rev-list --first-parent "
                                                   "origin/main`, a commit a line; none refuses every session")
    ap.add_argument("--served", action="store_true", help="fill: commit each call's tables and the weights' B as its "
                                                          "Program's registered values (pouw_h1_served.Served) and register them")
    ap.add_argument("--skip-register", action="store_true", help="fill with --served: record the window's entries but "
                    "not pouw_h1_served.register's --registered files (at FILL its partition takes hundreds of GB)")
    ap.add_argument("--baseline-h1", action="store_true", help="fill with --served: serve an H1 window without Served "
                                                               "first, in the solo window's place (the step-time baseline)")
    ap.add_argument("--model", metavar="ROLE", help="fill with --served and --gate: the served model's int8 checkpoint role "
                                                    "(LLAMA31_8B_INT8), whose registration under --circuit the gate's "
                                                    "model-key answers; each K's b/<K> is registered under its key before "
                                                    "the window's salt (pouw_h1_served.Weights)")
    ap.add_argument("--circuit", metavar="SHA512", help="with --model: the period's circuit (its declared Program's digest)")
    ap.add_argument("--workload", choices=("batch", "fill"), default="batch",
                    help="batch: PROMPTS batched and the first alone (--rows, --steps, --max-tokens); fill: pouw_h1.FILL, 64 in "
                         "flight every served step (its own T, S and cap)")
    ap.add_argument("--handoff", metavar="SOCK", help="fill with --challenger: once served, hand the window (to "
                                                     "--handoff-state) and its session's stream (to the holder at SOCK) "
                                                     "off and exit, the close left to --close-from (pouw_h1_handoff)")
    ap.add_argument("--handoff-state", metavar="DIR", help="with --handoff: the prover's scratch the window is saved in")
    ap.add_argument("--prover-dir", metavar="DIR", help="fill with --served: the prover's private directory, kept past the "
                                                       "close, outside --out (pouw_h1_prover: its salt key and values)")
    ap.add_argument("--live-dump", metavar="DIR", help="with --commit: each Commit window's live@<s> statements, the "
                                                       "prover's rows and openings (pouw_h1_live.dump), into DIR/<window>, "
                                                       "outside --out, for benchmarks/pouw/h1_prove/live.sh")
    ap.add_argument("--close-from", metavar="DIR", help="no GPU: close and release the window a --handoff run saved in DIR, "
                                                        "on the session's stream --stream-fd, completing --out's record")
    ap.add_argument("--stream-fd", type=int, metavar="FD", help="with --close-from: the session's stream, as the holder "
                                                                "passed it")
    ap.add_argument("--commit", action="store_true",
                    help="fill with --served: after the 64's window, serve it again with the Commit in the engine "
                         "(pouw_h1_commit, ROOTS1) into <out>/commit/, and record each window's peak GPU memory and step "
                         "time beside the plain run's and the Commit's tie (no solo window, no --registered files); its "
                         "coins are this process's unless --challenger with --peer gives each window a session of its own")
    ap.add_argument("--receipts", metavar="SO",
                    help="fill with --served, without --handoff: each call's receipt@<c> from h1u.cu's units kernel of the "
                         "-hbc build SO (benchmarks/pouw/served_h1/producer_gpu.sh's h1u_hbc.so) on the serving GPU, the "
                         "first block's linears' calls at step 0 and the last step held to h1tile.reference; a .cubin SO "
                         "(served_h1/noise.cu's) takes receipts.py's torch chain over its noise")
    ap.add_argument("--arch", help="fill with --served: the H1 device record (h1device.DEVICES: sm_90, the H200; sm_120, "
                                   "the RTX PRO 6000) whose domain bounds each K's tile unit (default: the serving card's, "
                                   "pouw_h1_served.card_arch)")
    ap.add_argument("--j10r", action="store_true",
                    help="fill with --served: declare the period's window Program over J10-R's tile unit "
                         "(h1tile.PouwH1Tile5: one noise stream a call, seeded by its roots and first position), so its "
                         "receipts and public rows are J10-R's; v1's unit otherwise")
    ap.add_argument("--declare", metavar="SLOTS,CAP,T,S", type=lambda v: tuple(int(x) for x in v.split(",")),
                    help="fill: serve SLOTS one-token prompts at cap CAP, T rows a call, S steps (pouw_h1.Static) in place "
                         "of pouw_h1.FILL's 64,16,64,18")
    ap.add_argument("--negatives", action="store_true",
                    help="with --commit: after the honest window, serve one whose A codes H1 is handed are all zero and "
                         "one with a row off, each with the Commit in the engine (NEGATIVES), and hold each to core's "
                         "ROOTS1 steps (core_verdict)")
    ap.add_argument("--credit", action="store_true",
                    help="with --commit and --receipts: core credit's statement steps over the honest Commit window "
                         "(pouw_h1_credit: registered_file, _verified on a stand-in verifier of record, _proved, "
                         "_registration), accepting it, and refusing a receipt word flipped, a public table under another "
                         "step's A root and an ok row 0")
    ap.add_argument("--profile-register", action="store_true",
                    help="fill with --served: profile each served window's registration (Served.commit) and write its "
                         "functions by cumulative time to <window out>/register-profile.txt")
    ap.add_argument("--core-from", metavar="DIR", help="no GPU: core's ROOTS1 steps (core_verdict) over each window's "
                                                       f"saved evidence, DIR/*/{CORE_EVIDENCE}")
    a = ap.parse_args(argv)
    from verity_vllm.engine.build import parse_engine_args
    a.engine_arg = parse_engine_args(a.engine_arg)
    if a.core_from:
        return 0 if core_from(a.core_from, a.out)["ok"] else 1
    if a.receipts and not (a.served and a.workload == "fill" and not a.handoff):
        ap.error("--receipts is the fill workload's, with --served, without --handoff (the close stage has no GPU for "
                 "the units kernel)")
    if a.j10r and not (a.served and a.workload == "fill"):
        ap.error("--j10r is the fill workload's, with --served (the period's declared Program)")
    if a.commit and not (a.served and a.workload == "fill" and not a.handoff):
        ap.error("--commit is the fill workload's, with --served, without --handoff")
    if a.commit and a.challenger and not a.peer:
        ap.error("--commit with --challenger needs --peer: it serves several windows, and each takes a session of its own "
                 "(one --session is one window's)")
    if a.peer and not (a.challenger and a.gate and a.workload == "fill" and not a.handoff and a.session is None):
        ap.error("--peer is the fill workload's, with --challenger and --gate, without --session or --handoff (each window "
                 "reserves its own session)")
    if a.period and not a.check_challenger and not (a.commit and a.peer):
        ap.error("--period is --check-challenger's, or --commit's with --peer (the period its Commit window opens)")
    if a.period and a.commit and not a.circuits:
        ap.error("--commit with --period needs --circuits: core's credit verifies a window only on the circuit files its "
                 "period's open commits (pouw_h1_circuits --program PROGRAM --out DIR --live T,S)")
    if a.negatives and not a.commit:
        ap.error("--negatives is --commit's")
    if a.credit and not (a.commit and a.receipts):
        ap.error("--credit is --commit's, with --receipts (a window's receipt tables are what its statements' o0 reads)")
    if a.declare and (a.workload != "fill" or len(a.declare) != 4):
        ap.error("--declare is the fill workload's, SLOTS,CAP,T,S")
    if a.close_from:
        if a.stream_fd is None:
            ap.error("--close-from needs --stream-fd (the session's stream the holder took)")
        return 0 if fill_close(a)["ok"] else 1
    if a.model and not (a.circuit and a.gate and a.served and a.workload == "fill"):
        ap.error("--model is the fill workload's, with --served, --circuit and --gate (the gate's model-key)")
    if a.handoff and not (a.handoff_state and a.challenger and a.workload == "fill"):
        ap.error("--handoff is the fill workload's, with --challenger and --handoff-state")
    if a.compile_from:
        return 0 if compile_from(a.compile_from, a.out)["ok"] else 1
    if a.check_challenger:
        if not (a.key_sha512 and a.ledger):
            ap.error("--check-challenger needs --key-sha512 (the auditor's pin from config) and --ledger (once a head)")
        from verity.core.protocols.sampled_proofs import remote
        desc = gzip.decompress(Path(a.descriptor).read_bytes()) if a.descriptor else None
        main = None if a.main is None else remote.main_lines(a.main)
        return 0 if check_challenger(*a.check_challenger, a.key_sha512, a.out, a.ledger, a.window, a.period, desc,
                                     a.pins, main)["ok"] else 1
    if not (a.at and a.by):
        ap.error("serving needs --at and --by (the pin entry's, if one is made)")
    return 0 if (fill if a.workload == "fill" else run)(a)["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
