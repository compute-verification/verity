"""VLLM-20 on one GPU: the int8 Llama-3.1-8B pin (as `tests/engine/int8_exact_gpu.py` finds or makes it) served on the eager
int8_exact row with PoUW's H1 + fx1 (``pouw:eps1-sm120-h1fx1``, `protocol_options.pouw_h1`), checked on the host.

One process writes `<out>/pouw_h1_gpu.json` and the option's record (`<out>/pouw-h1-calls.json`):

  served    `PROMPTS` greedy, batched (a decode batch of len(PROMPTS)) and the first alone (decode at one row), before the
            option is installed and with it: the token ids are the same, since the arm serves its own y. The engine's
            ``max_num_batched_tokens`` is T (``--rows``), with chunked prefill, so no step puts more rows in a linear.
  calls     the static window (`H1Run`, T rows a call, S = ``--steps``): every (step, linear) call in order at s·U + off(w),
            the served ones recorded and the rest empty, nothing refused; the credit (served rows) beside the padding.
  compile   every served shape compiled to `eps1-sm120-h1fx1` (`H1Run.check`): its Program digest, units, one seed group and
            only full 32 x 32 grid tiles. A shape takes minutes on the host (about 0.13 s a unit), so only `--compile` does it
            here; otherwise `--compile-from <out>/pouw-h1-calls.json` does it on a CPU node from the record, with each call's
            positions against its shape's compiled units, and writes `<out>/pouw_h1_compile.json`.
  z         the last retained calls' exact Y (`engine.int8_exact.int_mm` on the GPU) against the s8 product on the host, and
            with `--compile`, a unit's z (`Call.z`) against its cells.
  hook      a listener (`H1Run.listen`) gets every call with the served tensors on the GPU: A int8 [m, k] and s_a f32 [m]
            (the Commit's ``Int8RowQuant_v1`` /0 and /1), the layer's own int8 weight and scales; what it returns is in the
            manifest; steps begin at linear 0, and every step's calls run over the same layers in order.
  requests  every request the scheduler admits is an entry under the root of its step's round, recorded before the round's
            live coin, and every served row's slot names its request (`h1_settings`: the engine's runner and the window's
            round coins; `requests_check`), one entry a served request and no served row without one.

``--workload fill`` serves `pouw_h1.FILL` instead (VLLM-52, PERIOD-PATH's ask 1): 64 one-token prompts (`fill_prompts`) under
VLLM-31's static schedule (``ignore_eos``, every request to its cap, host prefix cache off), with ``max_num_seqs`` 64 and T = 64,
so all 64 prefill at step 0 and decode together for the 16 served steps of an S = 18 window. Without the option it serves
the 64 and then the first alone (the 1-sequence row); with it, the 64 in one window (`<out>/pouw-h1-calls.json`) and the
first alone in a second (`<out>/solo/`). It records each window's step rows against the declaration (`Static.check`), its
u (`H1Run.utilization`), the credit gate's verdict, each step's seconds (at linear 0 of the next) and tokens/s, and the
static schedule's check (`engine.static_schedule.check`) of every request. Compile is `--compile-from` on each record.

``--challenger HOST:PORT --session JSON --gate CMD`` takes the window's salt and round coins from the auditor host's
challenger, in the session infra's gate reserved for the window (``--session``, the gate's reserve answer), which the
window opens through ``CMD open --session S --root R`` on its window-open root once round 0 is admitted
(`gate_command`; `pouw_h1_window.Window`; on PEER, the port the gate forwards), one session a window, so the fill
workload serves only the 64's window. ``--check-challenger CALLS_JSON DIR --key-sha512 HEX --ledger HEAD [--ledger HEAD
...]`` is the verifier's side on no GPU, ``--ledger`` given once per head infra published. With ``--window WINDOW_JSON``
(``{root}``, the window's open root) DIR is the challenger run's own ``out/``: the window's coins against the session
the auditor's signed ledger (the longest of the heads, each extending the others) opens and ends for the window
(`pouw_h1_challenger.trusted_rounds_reason`). Without it the ledger is read as the newest that the others extend
(`pouw_h1_challenger.Ledger.load`, core's check), and DIR the session's tree as the gate's ``fetch`` gives it: the window held to the
session the ledger opened for its Commit's root (`pouw_h1_challenger.session_reason`). Either way the record is held to
its coins as one manifest (`pouw_h1_challenger.manifest_reason`), and with ``--period NAME`` the ledger must credit the lane's period
with the window accepted and presented as its close's windows, its open committing ``draw_k``
(`pouw_h1_challenger.period_reason`, core's `credit.period`). The batch workload admits the first prompt alone after the
batch, so it admits per round; the fill workload admits in round 0 only.

``--served`` (fill) commits each call's A and z/y tables and the weights' B as its window Program's registered values
(`pouw_h1_served.Served`, the pouw setting ``served``), under the round roots, which the commitment carries, and records
the trees and `pouw_h1_served.register`'s outcome (`served_check`). With ``--model M --circuit C`` (and ``--gate``) the
weights' ``b/<K>`` are the period's (`pouw_h1_served.Weights`, VLLM-72): the gate's ``model-key --model M --circuit C``
answers the auditor's registration of M under the period's circuit C and its salt key (`model_key_of`), and the served
model's rows are registered under that key before the salt, the window refused unless they fold to the registration's
weights' root, the one ``period-open --model M`` signed. ``pouw_h1_rehearsal.py`` runs it all against infra's gate.

``--receipts SO`` (fill, ``--served``, no ``--handoff``) registers each call's ``receipt@<c>`` too (BRIDGE2), computed on the
serving GPU by kernels' producer (``benchmarks/pouw/served_h1/producer.py``, `receipts_producer`): h1u.cu's units kernel of
the -hbc build SO (``producer_gpu.sh``'s ``h1u_hbc.so``) over each linear's own int8 weight, its seeds taking the
registered roots of the call's step's ``a/<K>@<s>`` and ``b/<K>``. The first block's linears' calls at step 0 and at the
window's last step are kept, and their first unit of each row block is held to `h1tile.reference` word for word
(`receipts_check`: live row blocks at step 0, dead at the last step).

``--commit`` (fill, ``--served``, in-process coins) serves the 64's window again with the Commit in the engine
(`pouw_h1_commit.attach`: the committer, the int8 source at each linear's Int8RowQuant_v1 site and its tap) into
``<out>/commit/``, each call's A table held to the Commit's AROW and each round over its input ids (``roots1``), and
records the peak GPU memory and step time of the plain run, the window and the window with the Commit (``commit``).

``--handoff SOCK --handoff-state DIR`` (fill, with ``--challenger``) splits the window at its last served step, so the
GPU is held only while it serves: the serving process records what only it sees (`served_measures`), saves the window,
hands its session's stream to the holder at SOCK and exits (`hand_off`, `pouw_h1_handoff`), and ``--close-from DIR
--stream-fd FD`` closes and releases it on no GPU (`fill_close`): the same close, trees, draw and session close, the
record the same `RECORD`.
"""
from __future__ import annotations

import argparse
import functools
import gzip
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
    reaches the serving node through ``ssh -R`` once the gate's open has answered, and a probe connect would take the
    place of the session's one client."""
    for i in range(polls):
        if listening(port):
            return i
        time.sleep(1)
    return None


def gate_command(argv: list[str]) -> Any:
    """The `pouw_h1_window.Gate` that runs the gate's verbs as `argv` + VERB (an ``ssh ... research@HOST``): ``open
    --session S --root R``, its JSON answer's ``commitment`` (null until serve.py signs its record) then read from ``status
    S``, and the answer returned once its ``port`` listens on this host (`await_listener`)."""
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
    gate.argv = argv
    return gate


def model_key_of(gate: Any, model: str, circuit: str) -> Any:
    """The auditor's registration of `model` under the period's `circuit` and its salt key
    (`pouw_h1_served.ModelKey`), as the gate's ``model-key --model M --circuit C`` answers the lane's gate key; the key
    is never printed or written here."""
    from verity_vllm.protocol_options import pouw_h1_served as SV
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
    session a window, `pouw_h1_challenger`) in the `session` infra's gate reserved for it, which `gate` opens on the
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
    (``a_roots``, `pouw_h1_requests.tables_reason`)."""
    from verity_vllm.protocol_options import pouw_h1_requests as RQ
    coins = h1.rounds.challenger.commitment
    why = [w for e in h1.requests if (w := RQ.request_reason(h1.round_log, coins, h1.salt, h1.rows, e, h1.open_request(e.index),
                                                             h1.open_first(e.index), h1.admission))]
    manifest, slots = h1.manifest(), []
    for s in range(len(h1.round_log)):
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
           "entries": [[e.index, e.round, e.slot] for e in h1.requests]}
    out["ok"] = bool(out["admitted"] == requests and not out["refused"] and not out["gaps"] and not why and not slots
                     and rounds is None and out["slots_are_rows"] and str(out.get("a_roots")).startswith("bound"))
    return out


def run(a: argparse.Namespace) -> dict[str, Any]:
    from verity_vllm import protocol_options as PO
    from verity_vllm.engine.build import build_engine
    from verity_vllm.program.frontend.kernel_readback import int8_exact_of
    from verity_vllm.program.frontend.target_profile import TargetProfile
    from verity_vllm.protocol_options import pouw
    from verity_vllm.protocol_options import pouw_h1 as PH
    from verity_vllm.protocol_options import pouw_h1_hooks as PHK

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    target = TargetProfile.from_device().with_overrides(int8_exact=a.int8_exact)
    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
                           "scheme": PH.SCHEME}
    with tempfile.TemporaryDirectory() as scratch:
        entry, how = pin(a)
        doc["pin"] = {"repo": entry["repo"], "revision": entry["revision"], "source": how}
        eng = build_engine(scratch_checkpoints(a.checkpoints, entry, Path(scratch)), case=a.role, target=target,
                           max_num_seqs=len(PROMPTS), engine_args={"max_num_batched_tokens": a.rows,
                                                                   "enable_chunked_prefill": True, "enable_prefix_caching": False})
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
            call = h1.calls[index]
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
    """Greedy one-token prompts under VLLM-31's static schedule (``ignore_eos``: every request runs to its cap)."""
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
    from verity_vllm import protocol_options as PO
    from verity_vllm.engine.build import build_engine
    from verity_vllm.program.frontend.kernel_readback import int8_exact_of
    from verity_vllm.program.frontend.target_profile import TargetProfile
    from verity_vllm.protocol_options import pouw
    from verity_vllm.protocol_options import pouw_h1 as PH
    from verity_vllm.protocol_options import pouw_h1_hooks as PHK
    from verity_vllm.protocol_options import pouw_h1_served as SV

    decl = PH.FILL
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    target = TargetProfile.from_device().with_overrides(int8_exact=a.int8_exact)
    doc: dict[str, Any] = {"schema": SCHEMA + "/fill", "argv": sys.argv, "torch": torch.__version__,
                           "device": torch.cuda.get_device_name(0), "scheme": PH.SCHEME,
                           "public": {"slots": decl.slots, "prompt_lens": list(decl.prompt_lens), "max_tokens": decl.max_tokens,
                                      "rows": decl.rows, "steps": decl.steps, "schedule": decl.schedule(), "u": str(decl.u)}}
    with tempfile.TemporaryDirectory() as scratch:
        entry, how = pin(a)
        doc["pin"] = {"repo": entry["repo"], "revision": entry["revision"], "source": how}
        args = decl.engine_args()
        eng = build_engine(scratch_checkpoints(a.checkpoints, entry, Path(scratch)), case=a.role, target=target,
                           max_num_seqs=args.pop("max_num_seqs"), engine_args=args)
    doc["engine"] = {"int8_exact": int8_exact_of(eng.model), "enforce_eager": eng.kwargs.get("enforce_eager"),
                     "effective": eng.effective, "args": decl.engine_args()}
    entry_name = f"pouw:{PH.SCHEME}"
    registry = model_key_of(a.gate, a.model, a.circuit) if a.model else None
    if registry is not None:
        doc["registry"] = {"model": registry.model, "circuit": registry.circuit, "weights_root": registry.weights_root}
        print(f"[fill] {registry.model}'s registration {registry.weights_root[:16]}... under circuit "
              f"{registry.circuit[:16]}...", flush=True)
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
    doc["serve_h1"] = {"prompt_lens": list(lens), "u": str(PH.Static(lens, decl.max_tokens, decl.rows, decl.steps).u)}
    cap, served_steps = decl.max_tokens, decl.max_tokens
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

    def window(name: str, want: list[dict[str, Any]], tdir: Path) -> dict[str, Any]:
        nonlocal layers
        layers = layers or PHK.layers(eng.model, pouw.linear_types())
        com = tap = None
        if name == "commit":
            from verity_vllm.protocol_options import pouw_h1_commit as CT
            com, _src, tap = CT.attach(eng, layers, run_id=f"pouw-h1-commit-{os.getpid()}")
        # in-process coins open no session, so a local window's key is its own (`Served.key`)
        served = (SV.Served(prover=a.prover_dir, registry=registry, tap=tap, window=None if a.challenger else os.urandom(64))
                  if a.served and name != "solo" else None)
        ctx = PO.Context(stage="serve", enabled=(entry_name,),
                         settings=h1_settings(eng, decl.rows, decl.steps, a.challenger, a.session, a.gate, retain_steps=a.retain,
                                              transcript_dir=str(tdir), admission="round-0", served=served))
        t0 = time.monotonic()
        comp = PO.install(eng.model, ctx)
        out: dict[str, Any] = {"install_s": round(time.monotonic() - t0, 1)}
        h1 = pouw.h1_run(entry_name)
        made = receipts_producer(a.receipts, served, h1, layers, decl.steps) if a.receipts and served is not None else None
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
            out["gpu"] = gpu_peak()
            out.update(served_measures(got, want, cap, served_steps, t0, end, stamps))
            if com is not None:
                rc = com.finalize()
                out["committer"] = {"run_root": rc.run_root.hex(), "name": com.name, "finalize_s": round(rc.t_finalize_s, 3),
                                    **{k: rc.extra.get(k) for k in ("values_retained_bytes", "expected_tensors",
                                                                     "committed_tensors", "errors_n", "errors")}}
            if a.commit:
                print(f"[fill] {name} served {json.dumps({k: out.get(k) for k in ('gpu', 'step_s', 'decode_step_s_median', 'serve', 'same_tokens', 'committer')}, default=str)[:3000]}",
                      flush=True)
            if a.handoff:
                comp.installed[0].release = None
            else:
                h1.close()
                out["close_s"] = round(time.monotonic() - end, 1)
                closed_measures(h1, out, len(want), lambda w: layers[w].weight, bad)
                if made is not None:
                    t1 = time.monotonic()
                    out["receipts"] = {**receipts_check(made), "check_s": round(time.monotonic() - t1, 1)}
                    print(f"[fill] {name} receipts {json.dumps(out['receipts'])}", flush=True)
                    out["ok"] = bool(out["ok"] and out["receipts"]["ok"])
                if tap is not None:
                    out["roots1"] = h1.details["roots1"]
                    out["ok"] = bool(out["ok"] and out["roots1"]["tied"])
        finally:
            comp.close()
            if com is not None:
                com.close()
                tap.untap()
        if a.handoff:
            out["hook_bad"] = bad[:20]
            return out
        return released(name, h1, out, served, tdir, bool(a.challenger), register=not a.commit)

    if a.commit:
        doc["h1_commit"] = window("commit", plain, outdir / "commit")
        (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    doc["h1"] = window("fill", plain, outdir)
    if a.handoff:
        return hand_off(pouw.h1_run(entry_name), a, doc)
    if a.commit:
        doc["h1_solo"] = {"skipped": "--commit measures the 64's window with and without the Commit"}
        h, c = doc["h1"], doc["h1_commit"]
        doc["commit"] = {"step_s": {"plain_mean": doc["plain"]["fill"]["step_s"], "h1_decode_median": h["decode_step_s_median"],
                                    "h1_commit_decode_median": c["decode_step_s_median"], "h1_mean": h["serve"]["step_s"],
                                    "h1_commit_mean": c["serve"]["step_s"]},
                         "gpu": {"plain": doc["plain"]["gpu"], "h1": h["gpu"], "h1_commit": c["gpu"]},
                         "layout_static": c.get("layout_static"), "same_tokens": c["same_tokens"],
                         "roots1": c.get("roots1"), "committer": c.get("committer"), "ok": c["ok"]}
        print(f"[fill] commit {json.dumps(doc['commit'], default=str)[:3000]}", flush=True)
    elif a.challenger:
        doc["h1_solo"] = {"skipped": "the challenger serves one session, the 64's window"}
    else:
        try:
            doc["h1_solo"] = window("solo", plain_solo, outdir / "solo")
        except Exception as e:  # noqa: BLE001
            doc["h1_solo"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:2000]}
    return fill_record(doc, outdir)


def served_measures(got: list[dict[str, Any]], want: list[dict[str, Any]], cap: int, served_steps: int, t0: float, end: float,
                    stamps: dict[int, float]) -> dict[str, Any]:
    """What only the serving process sees of a fill window: its rate, each step's seconds (at linear 0 of the next), its
    tokens against the plain run's and the static schedule's check."""
    step_s = [round((stamps[s + 1] if s + 1 in stamps else end) - stamps[s], 4) for s in sorted(stamps)]
    return {"serve": rate(got, end - t0, served_steps), "step_s": step_s, "served_end_unix": round(time.time(), 3),
            "decode_step_s_median": float(np.median(step_s[1:])) if len(step_s) > 1 else None,
            "same_tokens": [g["token_ids"] for g in got] == [g["token_ids"] for g in want], "tokens_digest": tokens_digest(got),
            "static_schedule": static_check(got, cap)}


def closed_measures(h1: Any, out: dict[str, Any], want: int, weight_of: Any, bad: list[str]) -> None:
    """A fill window's measures once closed (`want` requests served, linear w's weight `weight_of(w)`, the listener's
    complaints `bad`), into `out`, and its ``ok`` before its release."""
    from verity_vllm.protocol_options import pouw_h1 as PH
    decl, cap = PH.FILL, PH.FILL.max_tokens
    rows = h1.step_rows()
    decl_w = decl if want == decl.slots else PH.Static(decl.prompt_lens[:want], cap, decl.rows, decl.steps)
    zs = []
    for index, xq, _sa, y in h1.retained:
        call = h1.calls[index]
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
    """A fill window's commitment and how its session closed, once released, and its served trees' check (with
    ``register``, each call's ``--registered`` file too)."""
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


#: the serving stage's record when it hands its window off (`hand_off`), which the close stage completes
SERVING = "pouw_h1_gpu.serving.json"


def hand_off(h1: Any, a: argparse.Namespace, doc: dict[str, Any]) -> dict[str, Any]:
    """With ``--handoff SOCK --handoff-state DIR``: the served window saved to DIR and its session's stream handed to the
    holder at SOCK (`pouw_h1_handoff`), its record so far in ``SERVING``; this process then exits, and with it its GPU
    lease, and ``--close-from DIR`` closes and releases the window on the CPU (`fill_close`)."""
    from verity_vllm.protocol_options import pouw_h1_handoff as HO
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
    """The close stage, on no GPU: the window ``--close-from`` holds, its session the stream ``--stream-fd``, closed and
    released as the serving process would have (`pouw_h1_handoff`), and the fill record completed from ``SERVING``."""
    from verity_vllm.protocol_options import pouw_h1_handoff as HO
    from verity_vllm.protocol_options import pouw_h1_window as PHW
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
    """The window's registered tables (`pouw_h1_served`): each committed value's entry, what was refused, and with
    ``register`` the outcome of `pouw_h1_served.register`, each call's ``--registered`` file written into ``tdir``."""
    from verity_vllm.protocol_options import pouw_h1_served as SV
    doc: dict[str, Any] = {"entries": served.entries(), "refused": served.refused[:20],
                           "committed": bool(served.registered) and not served.refused,
                           "in_commitment": h1.commits.get("served") == {"entries": served.entries(),
                                                                         "refused": list(served.refused)},
                           "layout": layout_of(served.linears, served.calls, served.m, served.tile)}
    if not register:
        doc["register"] = {"skipped": "--commit's window registers no files"}
        print(f"[fill] served committed={doc['committed']} values={len(doc['entries'])}", flush=True)
        return doc
    try:
        reg = SV.register(h1, directory=tdir)
        doc["register"] = {"registered": len(reg.registered), "files": len(reg.files), "first": SV.file_name(0)}
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
    `h1`, h1u.cu's units kernel of the -hbc build `so` over each linear's int8 weight in `layers` in place (no second copy
    on the GPU), keeping the first block's linears' calls at step 0 and at step `steps` - 1 for `receipts_check`."""
    P = producer()
    made = P.Receipts(served, h1, P.Units(so), weight=lambda c: layers[c.layer].weight,
                      keep=lambda c: c.layer < 4 and c.step in (0, steps - 1))
    served.receipts = made
    return made


def receipts_check(made: Any) -> dict[str, Any]:
    """The kept calls' receipts (`receipts_producer`) at the first unit of each row block against `h1tile.reference`, word
    for word (`producer().check`); ``ok`` needs live and dead row blocks both."""
    rows = []
    for c, kept in sorted(made.kept.items()):
        ncb = kept["b"].shape[0] // 32
        rows += [(c, u, v["live"], v["ok"]) for u, v in producer().check(kept, list(range(0, len(kept["receipts"]), ncb))).items()]
    live = sum(r[2] for r in rows)
    return {"calls": sorted(made.kept), "units": len(rows), "live": live, "dead": len(rows) - live,
            "bad": [[c, u] for c, u, _live, ok in rows if not ok][:20],
            "ok": bool(rows) and all(r[3] for r in rows) and 0 < live < len(rows)}


def _check(h1: Any) -> tuple[dict[str, Any], bool]:
    try:
        out = h1.check()
        return out, all(v["tiles"] == ["32x32"] and v["groups"] == [[0]] for v in out.values())
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:4000]}, False


def compile_from(calls_file: str, out: str) -> dict[str, Any]:
    """The record's calls compiled on the host: every shape by `H1Run.check`, the window static, and each call (s, w) at
    s·U + off(w) counted in its shapes' compiled units."""
    from verity_vllm.protocol_options import pouw_h1 as PH

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
                     pins: list[str] | None = None) -> dict[str, Any]:
    """The verifier's side, on no GPU: the served window's coins (`calls_file`'s commitment) against the challenger's own
    directory under the auditor's key, pinned by the SHA-512 of its public key (`pouw_h1_challenger.pinned_key`), `ledger`
    every head infra published of the auditor's signed ledger and `pins` every head of the host's pins log (infra's
    ``pins.jsonl``, the gate's ``pins``; none refuses every session: `pouw_h1_challenger.pins_head`). With `window_file` (the window's ``{root}``) `directory` is
    the challenger run's ``out/`` (``research fetch RUN --all``), as the session the longest of the heads, each extending
    the others, opens and ends for the window (`trusted_rounds_reason`); without it the ledger is read as the newest that
    the others extend (`Ledger.load`, core's check), the window held to the session it opened for its Commit's root
    (`session_reason`), `directory` being that session's tree as the gate's ``fetch SESSION`` gives it (its ``out/``
    within). The record is held to its coins as one manifest (`pouw_h1_challenger.manifest_reason`), and with `period` the ledger
    must credit the lane's period with the window accepted and presented as its close's windows, its open committing
    ``draw_k`` (`pouw_h1_challenger.period_reason`, core's `credit.period`). Its ``credit`` is core's
    (`pouw_h1_challenger.credit_reason`) over `descriptor`, the bytes of the circuit the period committed
    (`pouw_h1_served.descriptor`; none refuses it)."""
    from verity.core.protocols.sampled_proofs import remote
    from verity_vllm.protocol_options import pouw_h1 as PH
    from verity_vllm.protocol_options import pouw_h1_challenger as PHC

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
            doc["reason"] = PHC.trusted_rounds_reason(rec["commitment"], d, ledger, doc["window"], key, pins=pins)
        else:
            newest = PHC.Ledger.load(ledger, key, pins=pins)
            doc["reason"] = PHC.session_reason(rec["commitment"], newest, lambda s: tree if s == sid else None)
    except Exception as e:  # noqa: BLE001
        doc["reason"] = f"{type(e).__name__}: {e}"
    doc["manifest"] = PHC.manifest_reason(rec)
    if key is not None:
        win = doc.get("window") or {"root": (rec["commitment"].get("session") or {}).get("root")}
        doc["credit"] = PHC.credit_reason(rec, d, ledger, win, key, descriptor, pins=pins)
    if period is not None:
        try:
            if key is None:
                raise ValueError("no key the pin names, so the auditor's ledger is unread")
            newest = PHC.Ledger.load(ledger, key, pins=pins)
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
    ap.add_argument("--period", metavar="NAME", help="with --check-challenger: the lane's period the window is one of, which "
                                                     "the ledger must credit with it accepted")
    ap.add_argument("--descriptor", metavar="PROGRAM", help="with --check-challenger: the period's declared window Program "
                                                            "(the rehearsal's window-program.json.gz), whose bytes core's "
                                                            "credit reads the window's layout from")
    ap.add_argument("--served", action="store_true", help="fill: commit each call's tables and the weights' B as its "
                                                          "Program's registered values (pouw_h1_served.Served) and register them")
    ap.add_argument("--model", metavar="ROLE", help="fill with --served and --gate: the served model's int8 checkpoint role "
                                                    "(LLAMA31_8B_INT8), whose registration under --circuit the gate's "
                                                    "model-key answers; each K's b/<K> is registered under its key before "
                                                    "the window's salt (pouw_h1_served.Weights, VLLM-72)")
    ap.add_argument("--circuit", metavar="SHA512", help="with --model: the period's circuit (its declared Program's digest)")
    ap.add_argument("--workload", choices=("batch", "fill"), default="batch",
                    help="batch: PROMPTS batched and the first alone (--rows, --steps, --max-tokens); fill: pouw_h1.FILL, 64 in "
                         "flight every served step (VLLM-52; its own T, S and cap)")
    ap.add_argument("--handoff", metavar="SOCK", help="fill with --challenger: once served, hand the window (to "
                                                     "--handoff-state) and its session's stream (to the holder at SOCK) "
                                                     "off and exit, the close left to --close-from (pouw_h1_handoff)")
    ap.add_argument("--handoff-state", metavar="DIR", help="with --handoff: the prover's scratch the window is saved in")
    ap.add_argument("--prover-dir", metavar="DIR", help="fill with --served: the prover's private directory, kept past the "
                                                       "close, outside --out (pouw_h1_prover: its salt key and values)")
    ap.add_argument("--close-from", metavar="DIR", help="no GPU: close and release the window a --handoff run saved in DIR, "
                                                        "on the session's stream --stream-fd, completing --out's record")
    ap.add_argument("--stream-fd", type=int, metavar="FD", help="with --close-from: the session's stream, as the holder "
                                                                "passed it")
    ap.add_argument("--commit", action="store_true",
                    help="fill with --served, in-process coins: after the 64's window, serve it again with the Commit in "
                         "the engine (pouw_h1_commit, ROOTS1) into <out>/commit/, and record each window's peak GPU memory "
                         "and step time beside the plain run's and the Commit's tie (no solo window, no --registered files)")
    ap.add_argument("--receipts", metavar="SO",
                    help="fill with --served, without --handoff: each call's receipt@<c> from h1u.cu's units kernel of the "
                         "-hbc build SO (benchmarks/pouw/served_h1/producer_gpu.sh's h1u_hbc.so) on the serving GPU, the "
                         "first block's linears' calls at step 0 and the last step held to h1tile.reference")
    a = ap.parse_args(argv)
    if a.receipts and not (a.served and a.workload == "fill" and not a.handoff):
        ap.error("--receipts is the fill workload's, with --served, without --handoff (the close stage has no GPU for "
                 "the units kernel)")
    if a.commit and not (a.served and a.workload == "fill" and not a.handoff and not a.challenger):
        ap.error("--commit is the fill workload's, with --served, without --challenger or --handoff")
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
        desc = gzip.decompress(Path(a.descriptor).read_bytes()) if a.descriptor else None
        return 0 if check_challenger(*a.check_challenger, a.key_sha512, a.out, a.ledger, a.window, a.period, desc,
                                     a.pins)["ok"] else 1
    if not (a.at and a.by):
        ap.error("serving needs --at and --by (the pin entry's, if one is made)")
    return 0 if (fill if a.workload == "fill" else run)(a)["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
