"""The serve tax: what each change to the production server costs on every token, against stock compiled vLLM at the
integration's pin, on the same node, workload and SLO.

    serve_tax.py plan --model-id ID --c4 C4.json.gz --rates NAME=R,... [--span-s 90] [--requests N] --out PLAN.jsonl
    serve_tax.py run --arm ARM --model-id ID --c4 C4.json.gz --rates NAME=R,... [--checkpoint DIR] [--reps 3] [--span-s 90]
                     [--requests N] [--plan PLAN.jsonl] [--window-s S] [--kv-window-s S]
                     [--tick-ms SAT,FREE | --tick-from STEP_TRACE|RECORD_ARM_DIR] --out DIR
    serve_tax.py run --arm ARM --model-id ID --client vllm-bench --loads NAME[=RUNG+RUNG...],... [--dataset NAME=FILE]
                     [--checkpoint DIR] [--reps 3] [--private DIR] --out DIR
    serve_tax.py run --arm tick-certifier --model-id ID --client vllm-bench --loads NAME=RUNG --tick-ms SAT,FREE
                     --side server [--front PORT] [--cells NAME=N,...] [--stop-at TIME] --out DIR
    serve_tax.py run --arm tick-certifier --model-id ID --client vllm-bench --loads NAME=RUNG [--pass-prompts ...]
                     --side client --certifier-windows N [--front PORT] [--cells NAME=N,...] [--reps 3] --out DIR
    serve_tax.py passes --loads NAME[=RUNG+RUNG...],... [--dataset NAME=FILE] [--fetch] [--arms "ARM ..." --model-id ID]
    serve_tax.py dataset LOAD [--fetch]
    serve_tax.py metrics DIR [--slo-ttft-s S --slo-tpot-s S]
    serve_tax.py table DIR... --out DIR [--slo-ttft-s S --slo-tpot-s S]
    serve_tax.py cost DIR... [--arm ARM] [--reference ARM] [--target PERCENT] [--json] [--out DIR [--result]]
    serve_tax.py tick STEP_TRACE|RECORD_ARM_DIR [--load saturated]
    serve_tax.py fit ARM_DIR --budget-s S --reps N
    serve_tax.py checkpoint --model-id ID

`run` serves one arm (`ARMS`) on one server lifetime of vLLM's own OpenAI server (`vllm.entrypoints.openai.api_server`):
a warm-up load, then each named load REPS times, rep by rep (light, medium, saturated, light, ...), each from an empty prefix
cache (the dev server's ``POST /reset_prefix_cache``). A load is the trace corpus's request plan (C4 prompts of 16 to 1,024
tokens, about 60% behind one of eight shared prefixes, caps 64 or 256: `traffic.plan`) under Poisson arrivals at its rate for
``--span-s`` seconds, sent by the corpus's streaming client (`corpus.drive`), every request decoding to its cap
(``ignore_eos``) so that every arm serves the same tokens. A plan whose requests carry their own sampling (the replay
ladder's production plan: ShareGPT, multi-turn and 8k-32k-token prompts, each greedy or at temperature 0.7 with its seed
by the public rule) is served with it, each request stopping at its end of text or its cap (`serve_body`). Each load's DIR/<load>/rep-<r>/client.jsonl holds the client's
record of each request (send, each streamed chunk's time and tokens, done), and meta.json the load's rate, start and count,
and its schedule: the engine's counters on ``/metrics`` before and after it (`SCHEDULE`).

The arms, each one engine configuration; each adds its change to the arm it names as its parent:

- ``compiled``: stock vLLM at the pin, its defaults (CUDA graphs, with torch.compile on the architectures vLLM
  compiles, which are not the ones it captures breakable CUDA graphs for, GLM-5.3's among them; chunked prefill,
  prefix caching, async scheduling), FP8 where the checkpoint is FP8;
- ``compiled-again``: ``compiled`` in a second server lifetime, which changes nothing: what two starts of one engine
  differ by, the noise floor of a cost (`cost_report`);
- ``no-prefix-cache``: ``--no-enable-prefix-caching``;
- ``published``: ``compiled`` started as vLLM's recipe for the model starts it (`Model.published`): the recipe's flags,
  its speculative decoding among them, the context limit the job names or else the recipe's, vLLM's own limit on the
  requests it runs at once in place of `BASE_ARGS`'s, and behind the front end the job names (`FRONT_ENDS`). What
  stock vLLM serves under the settings its authors publish, where ``compiled`` is stock under the settings the other
  arms share;
- ``eager``: ``--enforce-eager``, the tax if commits or replay force eager;
- ``bi-compiled``: ``compiled`` with ``VLLM_BATCH_INVARIANT=1``, vLLM's batch-invariant kernels;
- ``bi-eager``: ``eager`` with ``VLLM_BATCH_INVARIANT=1``, the committing path's execution with nothing committed;
- ``record``: the scheduler's step trace (`verity_vllm.engine.step_trace.StepTrace`, the recorder serve-openai uses), in
  the engine core, written by a thread of its own (`servetax_plugin`);
- ``release-order``: requests added in the network certifier's release order (`serve_admit.ReleaseOrder` in the engine
  core, `servetax_plugin.ReleaseGate`), the client standing in for the certifier, which numbers them in send order;
- ``tick``: the certified tick (`serve_admit.TickGate` under vLLM's server, `verity_vllm.pipeline.tick_plugin`), τ_sat and
  τ_free given or calibrated on a ``record`` trace (`tick`), the client standing in for the certifier (`corpus.StandIn`);
- ``tokens``: each window's tokens committed: every prompt id and every sampled id, 4 bytes a token, in ``rows-sha512``
  rows (`verity_vllm.commit.step_rows`), hashed off the engine loop at each window's close (`servetax_plugin.TokenCommit`).
- ``kv``: the paged KV cache committed at window edges (every ``--kv-window-s``): the whole pool, between two steps, in
  ``rows-sha512`` rows of 1 MiB hashed on the GPU on a side stream ordered after the step by an event, with no hook inside
  a CUDA graph (`servetax_plugin.KvEdge`);
- ``tokens-only``: ``record`` with each Commit period's tokens committed as ``tokens`` commits them, without the tick;
- ``boundary``: ``tokens-only`` with the boundary commitment at each replay block's edge (every ``--edges-every-s``
  seconds of a load): the engine drains, the KV pool is hashed where it lies by the row-table kernels under a fresh key
  and copied to pinned host memory (on the ranks ``--kv-copy`` names), and the scheduler's state and each rank's runner
  rows and generators are taken, all kept until the next edge (`servetax_plugin.edges_of`, `boundary.rank_take`).
- ``boundary-rows``: ``record`` with the two-stage audit's boundary commitment at serve, each window's finished
  requests committed by size class (`verity_vllm.pipeline.boundary_rows`), vLLM's prefix cache left on, its object the
  two-stage serve's (`ROWS`, with the model's check template, `Model.check`) or ``--boundary-rows``'s JSON instead
  (``check``, ``per_prompt``, ``per_output``, ``drained``, ``inline``, ``verdicts``, ``tables`` for each span's latent
  history tables (`servetax_plugin`) or ``stop``, the serve's stop rule, which has no tables), at ``--window-s``; the
  server is held after its loads until its windows' commits account for every finished request (`await_commits`).
- ``canonical-topk``: ``compiled`` with the canonical DSA top-k (`benchmarks/dsa_topk`'s vLLM plugin, `TOPK`), which
  rewrites only the indexer's rows of more than 2,048 keys, so it acts on long contexts alone;
- ``two-stage``: the two-stage serve's launch line, everything its engine turns on over stock: ``boundary-rows``, the
  canonical DSA top-k and one attention layer's latent history tables in the boundary commitment (`TABLES`), with
  prefix caching off: a table holds its request's rows from position 0, and a request served from the cache computes
  none before its hit's end (`BoundaryWindows.install` refuses the pair). The line is this arm's data and is stated
  nowhere else: a job that serves the two-stage audit serves this arm.
- ``two-stage-tick``: that launch line admitting on the certified tick, as ``tick`` does: the arm whose schedule is a
  function of the certified arrivals, so its windows' advice holds no decision. It is made of ``two-stage``'s data,
  and like every tick arm but the next the corpus's client serves it, standing in for the certifier.
- ``tick-certifier``: ``tick`` with the network certifier itself in front, and vLLM's client behind it. The engine
  admits on the certified tick from the certifier's own releases and watermark, and every request and every answer
  crosses the certifier's cells. It is served by two commands (``--side``, `certified`): the server's on the node,
  the engine and the id front before it (`certified_server`), and the client's where the certifier runs, a host of
  the auditor's and never the node: the certifier, the shim that turns vLLM's client's completions into the
  certifier's id requests (`cell_shim`) and the client's passes (`certified_client`). The certifier's schedule
  (`CELLS`) paces every answer: one id a connection each bucket after a fixed delay, so the tokens a second this
  arm serves are at most its lanes' (`ceiling`) whatever its engine does, and its cost over stock is the
  schedule's and its engine's together, not GPU time a token alone (`cost_text` says so beside it). An id
  request has no ``ignore_eos`` and the tick's server no route that empties the prefix cache, so the plugin holds
  each request to its length and empties the cache for each pass's first request (`servetax_plugin`: ``to_length``,
  ``idle_reset``).
- ``prod``: ``published``'s server, as the stock arm the arms made from it are costed over, at the same settings
  (`STOCK`, `stock_of`);
- ``prod-again``: ``prod`` in a second server lifetime, the noise floor of a cost over ``prod``;
- ``prod-record``: ``prod`` with the scheduler's step trace, as ``record`` is ``compiled`` with it, each request's
  prompt ids and served ids kept (`servetax_plugin.ServedIds`), which a replay of a pass sent by vLLM's client is
  given and compared with (`replay.window`), and the canonical DSA top-k, as ``two-stage`` has it: a row of more than
  2,048 keys is then a function of its inputs, which a replay that is to give the served tokens needs (it runs under
  the same plugin and environment: `replay.served_under`).

An arm is data (`Arm`): its server flags, the changes of `servetax_plugin` it turns on, the other vLLM general plugins
its server loads, its environment and its boundary commitment's object. What an arm keeps on the node and never under
its run's directory goes to ``--private`` (``{private}`` in its data): a boundary commitment's classes
(DIR/boundary/w<window>/p<P>-o<O>/: each class's requests' tokens and its salts, which are values; the run's directory
keeps each class's public commit.json as a line of server/boundary.PID.jsonl), the history tables' rows and salts, and
the top-k's counters. A serve's later stages read them, so the corpus client needs ``--private``; a cost's run by
vLLM's client, which nothing reads them of, gets by default a fresh directory removed when the arm stops.

`plan` fixes the requests every arm serves, so a job's arms and reps read one file (``--plan``).

With ``--client vllm-bench`` the loads are sent by vLLM's own benchmark client instead (`vllm bench serve`, `bench_argv`),
one pass a load and rung: a load (`LOADS`) is one of the client's datasets, and a rung the concurrency the client holds
(``--request-rate inf --max-concurrency RUNG``, the shape of vLLM's standing serving tests), every request greedy and
decoding to its length (``--ignore-eos``); under ``--sampled`` each is sampled instead, at the recipe's temperature
and top-p (`SAMPLED`), naming its client as its ``user``. An arm that keeps the record is sent that user's seed too
(`user_seed`, `sends_seed`; another in each server lifetime), under which its replay samples the request again; a
stock arm is sent none, as a served request carries none. Each pass's DIR/<load>@<rung>/rep-<r>/result.json is the result the client
saved, and `metrics` and `table` take its figures as the client computed them (`bench_metrics`). A load whose dataset
is a file reads the node's one copy of it, pinned by revision and SHA-512 (`Dataset`): `passes --fetch` downloads it
into the node's cache where it is missing, every run checks its SHA-512 and refuses a file that differs, and its
record holds the hash and the path, never the file.

`cost` prints the serving cost of one arm over a reference arm from such passes: how much more GPU time the arm needs
a token, its tax where the reference is saturated: the reference's output tokens a second over the arm's, less one,
both at the rung of a load where the reference's are highest (`serving_cost`, the benchmark kit's paired ratio over
reps: `kit`). Every arm serves the same prompts decoded to the same lengths, so the ratio is of GPU time at equal
tokens, and no SLO enters it. Each load's line says whether the reference's own rungs show it saturated at that rung
(`peak`: it served a rung above, and no more tokens a second at any). One cost a load, the largest the number, with
its 95% interval over reps, its verdict against ``--target`` or the model's target, and every other arm's cost beside
it, as text and as one JSON object whose first keys are value, unit, commit and run. The model's target is the sum of
its parts, each of some of our changes (`Part`, `CHANGES`): an arm that leaves one of those off, or has a change on
that no part counts (`not_in_target`), has no verdict against it but "not comparable" (`NOT_COMPARABLE`), and each
arm's row names the target's part for what its step turns on (`in_target`). Under each load's cost are both sides'
output and prompt tokens a second, the tokens a request and the engine's GPUs, and the load's cost within the SLO
(`within_slo`: the reference's best goodput over the arm's, less one, each at the rung where its own is largest),
which is the cost only where both sides meet the SLO for every
request and which no target or verdict is of. The number's loads are chat and long context (`NUMBER`): where one of
them has not run, the first line names the one that has and the number isn't said to meet its target. Neither repeats
a prefix, so the number is labelled "no repeated prefix" and is never given without the row on a repeated prefix
directly under it (`prefix_row`): the arm's cost over the reference on a third load (`PREFIX`), four planned requests
of which the second and fourth begin as the one before does (`PAIRS`), sent by the client's timed_trace dataset from a
file of lengths and block ids (`write_trace`) with one request in flight, in a run of its own. ``--dataset
prefix=FILE`` serves a table of planned requests instead (`read_plan`: arrival times and lengths, no token and no
text; ``prefix_plan.jsonl`` is the full-size one), each request sent at its arrival time (`ARRIVING`), and the row
then says whether each arm served all that arrived and whether the reference kept up (`KEPT_UP`): where it did, tokens
a second is not a cost and the row's cost is the two latencies. Such a row's figures are their readings, one a rep,
with no mean and no interval. The row has no target
and no verdict; where that load has not run, the report's value is None and its first line says why. Directories of
one arm from two jobs, each a load of its own, are one arm (`joined`), so a load served later joins the number.
``--result`` writes the run's result.json too (`result_of`): each load's cost a measurement, and the number one under
the model's metric id once every load has run.

A sessions load (`Load.sessions`: ``agentic``, ``agentic-full``, ``agentic-256k``) is an agent's recorded sessions (`session_lines`),
each a chain of requests of which each begins as the one before it does. Its rung is its users: each is a client of
its own with one request in flight, which sends its sessions' turns in order, each once the turn before has ended, so
a turn finds its session's prefix computed (`drive_bench`, `merged`). `cost` on such a load is `session_cost`: the
arm's cost over the stock engine it is made from, a server lifetime its unit (`over_lifetimes`), with its interval
over the lifetimes an arm, its verdict against `SESSION_TARGET` by that interval's upper end, and beside them the
cost over the passes taken as independent and the noise floor.

`metrics` reads each load's client records: throughput (output tokens and requests a second over the load's span, its
first send to its last done), TTFT and ITL (the gaps between a request's streamed chunks) at p50 and p99, and goodput, the
requests a second that met the SLO (TTFT at most ``--slo-ttft-s`` and time per output token after the first at most
``--slo-tpot-s``), with the GPUs' busy seconds over the load (NVML's utilization integrated: `GpuBusy`) and the edges
that fell in it. `table` lays the arms' metrics side by side per load, as means over reps with 95% Student-t intervals,
and each arm's tax: its compute per token over stock compiled's at the same load, less one (compiled's throughput over
the arm's), against arm ``compiled`` and against its parent; and each arm's edges, their pause and its parts. `fit` says
how many reps of an arm like an earlier one fit in the seconds a job has left.
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import gzip
import hashlib
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRACES = ROOT / "benchmarks/network_e2e/traces"
sys.path.insert(0, str(ROOT / "verity" / "configuration" / "config"))
sys.path.append(str(ROOT))
from verity_config.settings import SHARED  # noqa: E402 (from the tree's verity/configuration/config, put on sys.path above)
SCHEMA = "verity-bench/serve-tax/v0"
PLUGIN_ENV = "VERITY_SERVETAX"
TICK_MIDDLEWARE = "verity_vllm.pipeline.tick_plugin.TickFeed"
TICK_PLUGIN = ROOT / "adapters/vllm/verity_vllm/pipeline/tick_plugin.py"
#: two-sided 95% Student-t quantiles by degrees of freedom
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 8: 2.306, 10: 2.228, 20: 2.086, 30: 2.042}


@dataclass(frozen=True)
class Arm:
    parent: str | None
    what: str
    args: tuple[str, ...] = ()
    plugin: tuple[str, ...] = ()
    tick: bool = False
    #: the server's environment over the job's (VLLM_BATCH_INVARIANT is "0" unless an arm sets it); `PRIVATE` in a value
    #: is the run's private directory
    env: dict = field(default_factory=dict)
    #: the corpus's sampling (`corpus.SAMPLING`) and whether each request asks for its sampled tokens' logprobs
    sampling: str = "greedy"
    logprobs: bool = False
    #: a replay block's boundary state taken at each of ``--edges-s`` (`servetax_plugin.edges_of`), whether each edge
    #: also dumps every leaf of the engine core's state for the restore's check (`boundary.dump_core`), and whether each
    #: step's GPU busy time is timed for the replay's comparison (`boundary.Busy`)
    edges: bool = False
    edge_dump: bool = True
    edge_busy: bool = True
    #: the other vLLM general plugins its server loads: each one's entry point and its module's directory in this tree
    general: tuple[tuple[str, str], ...] = ()
    #: the boundary commitment's object, for an arm whose plugin has ``boundary_rows`` (`servetax_plugin`), less its check
    #: template, the model's (`Model.check`), and its window, the run's; `PRIVATE` as in ``env``
    rows: dict = field(default_factory=dict)
    #: whether the server starts as vLLM's recipe for the model starts it: the model's published flags added
    #: (`Model.published`), at the job's context limit or else the recipe's (`published_context`), the flags of
    #: `BASE_ARGS` no recipe sets left to vLLM's own defaults (`UNPUBLISHED`), behind the job's front end (`FRONT_ENDS`),
    #: with the flags of vLLM's own the job adds to the line (`published_args`)
    published: bool = False
    #: whether its requests come through the network certifier, which runs on another host than the engine: the arm
    #: is served by two commands, a side each (`certified`)
    certifier: bool = False


BATCH_INVARIANT = {"VLLM_BATCH_INVARIANT": "1"}
#: in an arm's data, the directory its run keeps on the node for what never leaves it (``--private``)
PRIVATE = "{private}"
# The two-stage serve's launch line, what its engine turns on over stock vLLM, is the next four values and the
# ``two-stage`` arm made of them, stated here alone: a job that serves the two-stage audit serves that arm.
#: the canonical DSA top-k: `benchmarks/dsa_topk`'s vLLM plugin on the server's path, switched on, its per-process
#: counters kept on the node
TOPK = (("verity_dsa_topk = dsa_topk:register", "benchmarks/dsa_topk"),)
TOPK_ENV = {"VERITY_DSA_TOPK": "canonical", "VERITY_DSA_TOPK_LOG": f"{PRIVATE}/topk-log"}
#: the boundary commitment's object (`verity_vllm.pipeline.boundary_windows.BoundaryWindows.of`): a replay block of
#: one check a served id with the check template's body in place, a window closed when it drains
ROWS = {"inline": True, "per_prompt": 0, "per_output": 1, "drained": True}
#: one attention layer's latent history tables in the boundary commitment (its ``tables``): the first layer's, their
#: rows and salts kept on the node
TABLES = {"layers": 1, "private": PRIVATE}
ARMS = {
    "compiled": Arm(None, "stock vLLM, its defaults (CUDA graphs, and torch.compile where vLLM compiles the model)"),
    "compiled-again": Arm("compiled", "stock vLLM in a second server lifetime: what two starts of one engine differ by"),
    "no-prefix-cache": Arm("compiled", "prefix caching off", ("--no-enable-prefix-caching",)),
    "published": Arm("compiled", "stock vLLM as its recipe for the model starts it: the recipe's flags, the job's "
                     "context limit or else the recipe's, and vLLM's own limit on the requests it runs at once in place "
                     "of this file's", published=True),
    "eager": Arm("compiled", "stock vLLM, eager", ("--enforce-eager",)),
    "bi-compiled": Arm("compiled", "+ vLLM's batch-invariant kernels", env=BATCH_INVARIANT),
    "bi-eager": Arm("eager", "eager with vLLM's batch-invariant kernels: the committing path's execution (TargetProfile's "
                    "bi-eager), nothing committed", ("--enforce-eager",), env=BATCH_INVARIANT),
    "record": Arm("compiled", "+ the scheduler's step trace (StepTrace)", plugin=("record",)),
    "release-order": Arm("record", "+ admission in the certifier's release order", plugin=("record", "release_order")),
    "tick": Arm("record", "+ admission on the certified tick", plugin=("record",), tick=True),
    "tokens": Arm("tick", "+ each window's tokens committed (rows-sha512)", plugin=("record", "tokens"), tick=True),
    "kv": Arm("tokens", "+ the paged KV cache hashed in place at each window edge", plugin=("record", "tokens", "kv"),
              tick=True),
    "restore-base": Arm("record", "+ seeded sampling, each sampled token's logprob returned", plugin=("record",),
                        sampling="seeded", logprobs=True),
    "restore-pause": Arm("restore-base", "+ a replay block's boundary state taken at each edge (drained, copied, hashed, "
                         "kept), as a server would: no leaf dump, no per-step timing", plugin=("record",), sampling="seeded",
                         logprobs=True, edges=True, edge_dump=False, edge_busy=False),
    "restore": Arm("restore-base", "+ a replay block's boundary state taken at each edge (drained, copied, hashed, kept) "
                   "and every leaf of the engine core's state dumped there for the restore's check",
                   plugin=("record",), sampling="seeded", logprobs=True, edges=True),
    "restore-unseeded": Arm("restore", "the restore arm with requests that carry no seed",
                            plugin=("record",), sampling="unseeded", logprobs=True, edges=True),
    "tokens-only": Arm("record", "+ each Commit period's tokens committed (rows-sha512), without the tick",
                       plugin=("record", "tokens")),
    "boundary": Arm("tokens-only", "+ the boundary commitment at each replay block's edge, as a server would take it: "
                    "drained, the KV pool hashed by the row-table kernels and copied to host memory, the scheduler's "
                    "state and each rank's runner rows and generators taken and kept; no leaf dump, no per-step timing",
                    plugin=("record", "tokens"), edges=True, edge_dump=False, edge_busy=False),
    "boundary-rows": Arm("record", "+ the two-stage boundary commitment at serve: each window's finished requests committed "
                         "by size class (boundary_rows)", plugin=("record", "boundary_rows"), rows=ROWS),
    "canonical-topk": Arm("compiled", "+ the canonical DSA top-k (dsa_topk's plugin, VERITY_DSA_TOPK=canonical)",
                          env=TOPK_ENV, general=TOPK),
    "two-stage": Arm("boundary-rows", "the two-stage serve's launch line: + the canonical DSA top-k and one attention "
                     "layer's history tables in the boundary commitment, prefix caching off",
                     ("--no-enable-prefix-caching",),
                     plugin=("record", "boundary_rows"), env=TOPK_ENV, general=TOPK, rows={**ROWS, "tables": TABLES}),
    "prod": Arm("published", "that server, as the stock arm the arms made from it are costed over", published=True),
    "prod-again": Arm("prod", "that server in a second lifetime: what two starts of its engine differ by",
                      published=True),
    "prod-record": Arm("prod", "+ the scheduler's step trace (StepTrace), each request's prompt ids and served ids, and "
                       "the canonical DSA top-k", plugin=("record", "ids"), env=TOPK_ENV, general=TOPK, published=True),
}
ARMS["two-stage-tick"] = replace(ARMS["two-stage"], parent="two-stage", what="+ admission on the certified tick", tick=True)
ARMS["tick-certifier"] = replace(ARMS["tick"], parent="tick", what="+ the network certifier in front: every request "
                                 "released by it and every answer paced by its schedule, sent by vLLM's client",
                                 plugin=("record", "to_length", "idle_reset"), certifier=True)
#: the stock arms, vLLM with no change of ours: an arm's cost is over the nearest of them above it (`stock_of`)
STOCK = ("compiled", "published", "prod")
#: the changes of ours an arm's engine can have on, each read from the arm's data: what a serving cost is made of, and
#: what a part of a model's target (`Part`) is of
CHANGES: dict[str, Callable[[Arm], bool]] = {
    "the step record": lambda a: "record" in a.plugin,
    "the served ids": lambda a: "ids" in a.plugin,
    "prefix caching off": lambda a: "--no-enable-prefix-caching" in a.args,
    "the two-stage boundary commitment": lambda a: "boundary_rows" in a.plugin,
    "the history tables": lambda a: bool(a.rows.get("tables")),
    "the canonical top-k": lambda a: a.env.get("VERITY_DSA_TOPK") == "canonical",
    "admission on the certified tick": lambda a: a.tick,
    "the edge snapshot": lambda a: a.edges,
    "the network certifier in front": lambda a: a.certifier,
}


def stock_of(arm: str) -> str:
    """The stock arm an arm is made from, which its cost is over: the nearest of `STOCK` above it, so an arm at the
    recipe's settings is costed over stock at those settings. `REFERENCE` for an arm with none above it or that this
    file doesn't define."""
    arm = ARMS[arm].parent if arm in ARMS else None
    while arm is not None and arm not in STOCK:
        arm = ARMS[arm].parent
    return arm or REFERENCE


def changes_of(arm: str) -> list[str]:
    """The changes (`CHANGES`) an arm's engine has on. An arm this file doesn't define has none it can be read to
    have."""
    return [name for name, on in CHANGES.items() if on(ARMS[arm])] if arm in ARMS else []


@dataclass(frozen=True)
class Part:
    """One part of a model's target (`Model.target`): what the cost model counts for some of our changes, in points
    of the target. An arm's cost holds the part only with every one of its changes (`CHANGES`) on, since the points
    are of them together."""
    name: str
    points: float
    changes: tuple[str, ...]
    what: str = ""
    basis: str = "modelled"


#: what every request's body adds to the corpus's (`corpus`), set by `run` from its arm
BODY_EXTRA: dict = {}
#: the corpus's seeded sampling without the request's seed, as production clients send it: vLLM then samples each
#: request's seed from the engine's numpy generator when the request reaches the model runner
UNSEEDED = "unseeded"


@dataclass(frozen=True)
class Model:
    repo: str
    revision: str
    served: str
    args: tuple[str, ...]
    #: the SLO goodput counts against: (TTFT s, time per output token s)
    slo: tuple[float, float]
    env: dict = field(default_factory=dict)
    #: the most serving it with our changes may cost over stock, in percent of stock's serving compute (`cost_report`),
    #: what that figure rests on: modelled (a cost model's share, not a measurement) or measured, and the parts it is
    #: the sum of, each of some of our changes: a cost is compared with the target only where its arm has every part's
    #: changes on (`target_parts`) and no change on that no part counts (`not_in_target`)
    target: float | None = None
    target_basis: str | None = None
    target_parts: tuple[Part, ...] = ()
    #: the id its serving cost is recorded under in a run's result.json (`result_of`)
    metric: str | None = None
    #: the check template its boundary commitment's replay block is made of, a serving template's descriptor id
    check: str | None = None
    #: what the command of vLLM's recipe for the model (recipes.vllm.ai) adds to ``args``, for the cards this file
    #: serves the model on, the context limit that command is served at where the job names none, and the recipe's
    #: tokenizer mode behind each front end (`FRONT_ENDS`): the ``published`` arm's server (`Arm.published`). The
    #: README says where the arm departs from the recipe's command
    published: tuple[str, ...] = ()
    published_context: int | str | None = None
    published_tokenizer: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = [c for p in self.target_parts for c in p.changes if c not in CHANGES]
        if unknown:
            raise ValueError(f"{self.served}: a part of the target names no change of CHANGES: {', '.join(unknown)}")
        total = round(sum(p.points for p in self.target_parts), 2)
        if self.target_parts and total != self.target:
            raise ValueError(f"{self.served}: the target's parts sum to {total}, and the target is {self.target}")

    def checkpoint(self) -> Path:
        hub = Path(SHARED.HF_HOME or "/workspace/hf") / "hub"
        return hub / f"models--{self.repo.replace('/', '--')}" / "snapshots" / self.revision


#: every arm's engine: vLLM's defaults but for the context (the plan's longest request is 1,024 + 256 tokens), the batch
#: cap (pinned, so that a saturated load hits the same cap on every vLLM build) and the seed
BASE_ARGS = ("--max-model-len", "4096", "--max-num-seqs", "256", "--seed", "0", "--disable-uvicorn-access-log")
#: the flags of `BASE_ARGS` no recipe of vLLM's sets, which the ``published`` arm leaves to vLLM's own defaults: the
#: batch limit, so that its server runs as many requests at once as vLLM itself allows on the hardware
UNPUBLISHED = ("--max-num-seqs",)
#: vLLM's word for the longest context limit the cards' KV cache holds, which the engine resolves when it starts
AUTO = "auto"
#: the front ends an arm that starts the server as the recipe does (`Arm.published`) serves behind: vLLM's Python
#: server, which every other arm has, or its Rust one, which ``vllm serve`` starts in front of the engine under
#: `RUST_ENV` on a build that has it
PYTHON, RUST = "python", "rust"
FRONT_ENDS = (PYTHON, RUST)
RUST_ENV = {"VLLM_USE_RUST_FRONTEND": "1"}
#: what makes vLLM write a request's or a response's text to the server's log, which a run publishes: the flags no arm
#: or model may carry, and the variable a server's environment never keeps
LOGS_TEXT_ARGS = ("--enable-log-requests", "--enable-log-outputs")
LOGS_TEXT_ENV = ("VLLM_DEBUG_LOG_API_SERVER_RESPONSE",)
MODELS = {
    "moonlight": Model("moonshotai/Moonlight-16B-A3B-Instruct", "4e735b07a89f73647dfab71ab91b840f362ede5b", "moonlight-16b",
                       ("--dtype", "bfloat16", "--trust-remote-code"), (2.0, 0.08)),
    "llama8b": Model("unsloth/Llama-3.1-8B", "3f0d51f8e5640f98f1a96ea9044a0e55c0a83814", "llama-3.1-8b",
                     ("--dtype", "bfloat16"), (2.0, 0.08)),
    "qwen3-30b": Model("Qwen/Qwen3-30B-A3B", "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39", "qwen3-30b-a3b",
                       ("--dtype", "bfloat16"), (2.0, 0.08)),
    # glm53's target is the cost model's share for serving: extra GPU time at equal tokens, which reads no SLO, for a
    # server that does not cut its cache at a phase edge and keeps vLLM's prefix cache on. Its parts are the model's,
    # none of them measured here. It has none for the step record, the two-stage boundary commitment, the history
    # tables or prefix caching off.
    "glm53": Model("zai-org/GLM-5.3", "aca966e4e02791568aa6a4ced368624b3d897f42", "glm-5.3",
                   ("--tensor-parallel-size", "8"), (2.0, 0.08), target=2.05, target_basis="modelled",
                   target_parts=(
                       Part("the certified tick", 1.63, ("admission on the certified tick",)),
                       Part("the tie order", 0.25, ("the canonical top-k",)),
                       Part("the snapshot bundle", 0.17, ("the edge snapshot",),
                            "the pause to copy and hash the cache at each edge and the carried pages' hash, with "
                            "the prefix cache on")),
                   metric="vllm/serving-cost-glm53", check="TokenSelect_v4{V=154880}",
                   # recipes/models/zai-org/GLM-5.3.yaml in vllm-project/vllm, its command "FP8 on 8xH200 (standard)":
                   # the FP8 checkpoint, tensor parallel, an FP8 KV cache, five draft tokens a step from the model's
                   # own draft layer. The command's tokenizer mode is its Rust front end's, and the recipe names the
                   # other for a build without it. The command names no context limit and the model's own does not
                   # fit these cards' KV cache, so the limit is the one the recipe's command builder sets for these
                   # cards; a job that wants the longest the cache holds names `AUTO`
                   published=("--kv-cache-dtype", "fp8", "--speculative-config.method", "mtp",
                              "--speculative-config.num_speculative_tokens", "5", "--tool-call-parser", "glm47",
                              "--reasoning-parser", "glm47", "--enable-auto-tool-choice"),
                   published_context=262144, published_tokenizer={PYTHON: "auto", RUST: "glm5_next"}),
}


def corpus():
    """The trace corpus's client and plan (`benchmarks/network_e2e/traces`), with every request decoding to its cap."""
    if str(TRACES) not in sys.path:
        sys.path.insert(0, str(TRACES))
    import corpus as C

    if not getattr(C, "_serve_tax", False):
        C.request_body = functools.partial(serve_body, C.request_body)
        C._serve_tax = True
    return C


def serve_body(body: Callable[..., dict], served: str, r: dict, rid: str, sampling: str) -> dict:
    """The corpus's request body (``body``) as the serve tax sends it: decoding to its cap, with `BODY_EXTRA`, and under
    `UNSEEDED` the seeded sampling's parameters without the request's seed. A plan request that carries its own sampling
    (the replay ladder's production plan, `prod_plan`) is sent with it instead, its seed by the public seed rule
    (`corpus.request_seed`) when it is seeded, and stops at its end of text or its cap."""
    if "sampling" in r:
        b = {"model": served, "prompt": r["prompt_ids"], "max_tokens": r["max_tokens"], **r["sampling"], "stream": True,
             "stream_options": {"include_usage": True}, "return_token_ids": True, "user": rid}
        if r.get("seeded"):
            b["seed"] = corpus().request_seed(rid)
        return {**b, **BODY_EXTRA}
    b = body(served, r, rid, "seeded" if sampling == UNSEEDED else sampling)
    if sampling == UNSEEDED:
        b.pop("seed", None)
    return {**b, "ignore_eos": True, **BODY_EXTRA}


# --------------------------------------------------------------------------------------------------------- the server

def entry_points(arm: Arm) -> list[str]:
    """The vLLM general plugins an arm's server loads, as entry points, in order: `servetax_plugin.register` for an arm
    with a change of its, under the tick `tick_plugin.register` after it (the tick's scheduler wrapper is then outside
    the step trace's, so the trace sees a request when the gate adds it), then the arm's others (`Arm.general`)."""
    return [*(["verity_serve_tax = servetax_plugin:register"] if arm.plugin or arm.tick else []),
            *(["verity_tick = verity_vllm.pipeline.tick_plugin:register"] if arm.tick else []),
            *(entry for entry, _ in arm.general)]


def plugin_site(site: Path, arm: Arm) -> Path:
    """A directory whose .dist-info declares the arm's vLLM general plugins (`entry_points`)."""
    d = site / "verity_serve_tax-0.0.dist-info"
    d.mkdir(parents=True, exist_ok=True)
    (d / "METADATA").write_text("Metadata-Version: 2.1\nName: verity-serve-tax\nVersion: 0.0\n")
    (d / "entry_points.txt").write_text("[vllm.general_plugins]\n" + "".join(f"{e}\n" for e in entry_points(arm)))
    return site


def check_plugins(env: dict[str, str], arm: Arm) -> None:
    """Each of the arm's vLLM general plugins loads in the server's interpreter and environment, checked before the
    server starts: vLLM logs a plugin whose import fails and serves on without it, which would be stock vLLM served
    under the arm's name."""
    names = [e.split("=")[0].strip() for e in entry_points(arm)]
    if not names:
        return
    code = ("import sys\nfrom importlib.metadata import entry_points\n"
            "found = {e.name: e for e in entry_points(group='vllm.general_plugins')}\n"
            "for name in sys.argv[1:]:\n    found[name].load()\n")
    p = subprocess.run([sys.executable, "-c", code, *names], env=env, capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"a vLLM general plugin of this arm ({', '.join(names)}) doesn't load in the server's "
                         f"environment: {(p.stderr.strip().splitlines() or ['no output'])[-1]}")


def keeps_private(arm: Arm) -> bool:
    """Whether an arm keeps files on the node: it has a boundary commitment, whose classes' directories hold their
    requests' tokens and their salts, or its environment or its boundary commitment's object names `PRIVATE`."""
    return "boundary_rows" in arm.plugin or PRIVATE in json.dumps([arm.env, arm.rows])


def _private(x: object, private: Path | None) -> object:
    """``x`` with `PRIVATE` in its strings replaced by the run's private directory."""
    if isinstance(x, str):
        return x.replace(PRIVATE, str(private))
    if isinstance(x, dict):
        return {k: _private(v, private) for k, v in x.items()}
    return [_private(v, private) for v in x] if isinstance(x, (list, tuple)) else x


def context_arg(text: str) -> int | str:
    """``--max-model-len``: a number of tokens, or `AUTO`."""
    return AUTO if text == AUTO else int(text)


def base_args(max_model_len: int | str | None = None, without: Sequence[str] = ()) -> tuple[str, ...]:
    """`BASE_ARGS`, with the context limit ``max_model_len`` where one is given (a plan with longer prompts), less the
    flags ``without``, each with its value."""
    args = list(BASE_ARGS)
    if max_model_len is not None:
        args[args.index("--max-model-len") + 1] = str(max_model_len)
    for flag in without:
        i = args.index(flag)
        del args[i:i + 2]
    return tuple(args)


def published_context(arm: str, model_id: str | None, given: int | str | None) -> int | str:
    """The context limit of an arm that starts the server as vLLM's recipe for the model does (`Arm.published`): the
    job's (``--max-model-len``, the same for every arm of the job, so that its arms differ by their flags alone), and
    where the job names none the recipe's (`Model.published_context`) in place of `BASE_ARGS`'s. Refused for a model
    whose recipe this file doesn't hold."""
    model = MODELS.get(model_id or "")
    if model is None or not model.published or model.published_context is None:
        held = ", ".join(n for n, m in MODELS.items() if m.published)
        raise SystemExit(f"arm {arm} starts the server as vLLM's recipe for the model does, and this file holds no recipe "
                         f"for {model_id or 'no model'} (Model.published; it holds {held}'s)")
    return model.published_context if given is None else given


def published_args(text: str | None) -> tuple[str, ...]:
    """``--published-args``: the flags of vLLM's own a job adds to the line of an arm that starts the server as the
    recipe does (the values a recipe leaves to the operator: the requests at once, the cards' share of memory), each one
    ``--FLAG=VALUE``, comma-separated."""
    args = tuple(x for x in (text or "").split(",") if x)
    bad = [x for x in args if not x.startswith("--") or len(x.split("=")) != 2 or not all(x.split("=")) or " " in x]
    if bad:
        raise SystemExit(f"--published-args: {', '.join(bad)}: each flag is --FLAG=VALUE, the flags comma-separated")
    return args


def front_end(arm: Arm, given: str | None) -> str:
    """The front end ``arm``'s server starts behind: the job's (``--published-frontend``) for an arm that starts the
    server as the recipe does, and vLLM's Python server for every other arm, whose plugins and tick it loads."""
    return given or PYTHON if arm.published else PYTHON


def server_argv(arm: Arm, model: Model, checkpoint: Path, port: int, num_gpu_blocks: int | None = None,
                max_model_len: int | str | None = None, front: str = PYTHON, added: Sequence[str] = ()) -> list[str]:
    """The server's command: vLLM's Python server, or behind `RUST` ``vllm serve``, the command that starts the Rust
    front end (under `RUST_ENV`, `server_env`). ``added`` (`published_args`) goes on the line of an arm that starts the
    server as the recipe does, after the recipe's flags and the runner's, and is refused where it names a flag the line
    already sets; no other arm's line takes it."""
    base = base_args(max_model_len, UNPUBLISHED if arm.published else ())
    serve = (["vllm.entrypoints.cli.main", "serve", str(checkpoint)] if front == RUST
             else ["vllm.entrypoints.openai.api_server", "--model", str(checkpoint)])
    mode = model.published_tokenizer.get(front)
    published = (*model.published, *(("--tokenizer-mode", mode) if mode else ())) if arm.published else ()
    line = [sys.executable, "-m", *serve, "--served-model-name",
            model.served, "--host", "127.0.0.1", "--port", str(port), *model.args, *published, *base, *arm.args,
            *(["--num-gpu-blocks-override", str(num_gpu_blocks)] if num_gpu_blocks else []),
            *(["--middleware", TICK_MIDDLEWARE] if arm.tick else [])]
    added = tuple(added) if arm.published else ()
    flags = [x.split("=")[0] for x in added]
    twice = sorted({f for f in flags if f in line or flags.count(f) > 1})
    if twice:
        raise SystemExit(f"--published-args: {', '.join(twice)}: the arm's line sets it already; name each flag once, "
                         "and the context limit with --max-model-len")
    return [*line, *added]


def server_env(arm: Arm, d: Path, site: Path, tick: dict | None, window_s: float, kv_window_s: float,
               edges: dict | None = None, boundary_rows: dict | None = None, check: str | None = None,
               private: Path | None = None, front: str = PYTHON) -> dict[str, str]:
    """The server's environment: the job's, with on its path only what the arm's plugins need (`entry_points`): the
    plugin site, this tree for a change of `servetax_plugin`'s or the tick, and each other plugin's directory, so a
    stock arm's processes import nothing of ours; ``VLLM_PLUGINS``, which would let a job's environment drop one of
    them, is unset, and so is what would put each response's text in the server's log (`LOGS_TEXT_ENV`); `RUST_ENV`
    is set behind the Rust front end and unset behind the Python one, whatever the job's environment says. An arm with
    ``boundary_rows`` gets its object: the arm's (`Arm.rows`), or ``boundary_rows`` in its place, with ``check`` (the
    model's check template) where it names none, at ``window_s``; its classes' directories, which hold values, go
    under ``private`` and never under ``d``. `PRIVATE` in the arm's data is ``private``."""
    ours = [*([str(site)] if entry_points(arm) else []),
            *([str(HERE), str(ROOT / "adapters/vllm"), str(ROOT)] if arm.plugin or arm.tick else []),
            *(str(ROOT / path) for _, path in arm.general)]
    env = dict(os.environ, VLLM_SERVER_DEV_MODE="1", VLLM_BATCH_INVARIANT="0",
               PYTHONPATH=os.pathsep.join([*ours, *SHARED.PYTHONPATH]))
    for name in ("VLLM_PLUGINS", *LOGS_TEXT_ENV, *RUST_ENV):
        env.pop(name, None)
    env.update(RUST_ENV if front == RUST else {})
    if private is None and keeps_private(arm):
        raise SystemExit("this arm keeps files on the node, outside its run's directory: --private DIR")
    env.update(_private(arm.env, private))
    rows = {}
    if "boundary_rows" in arm.plugin:
        rows = {"window_s": window_s, **({"check": check} if check else {}), **_private(boundary_rows or arm.rows, private)}
        if not rows.get("check"):
            raise SystemExit("this arm's boundary commitment needs its check template, and the model names none "
                             "(Model.check): --boundary-rows, a JSON object with check (run.sh: BOUNDARY_ROWS)")
    if arm.plugin:
        env[PLUGIN_ENV] = json.dumps({"out": str(d), "window_s": window_s, "kv_window_s": kv_window_s,
                                      **{k: True for k in arm.plugin}, **(edges if arm.edges and edges else {}),
                                      **({"boundary_rows": rows, "private": str(private)} if rows else {})})
    if arm.tick:
        if tick is None:
            raise SystemExit("the tick arms need τ: --tick-ms SAT,FREE or --tick-from STEP_TRACE")
        env["VERITY_TICK"] = json.dumps({k: tick[k] for k in ("bucket_us", "sat_us", "free_us")} | {
            "hold_ms": 5000, "feed": str(d / "tick-feed.jsonl"), "log": str(d)})
    return env


def value_settings(env: dict[str, str] | None = None) -> dict:
    """The integration's value settings in force in ``env`` (this process's when None), as its registry records them
    (`verity_vllm.settings.record`): each variable set there that can change what an engine computes or commits (the
    ones the registry declares so, and every ``VLLM_``, ``NCCL_`` and ``CUBLAS`` variable), a credential by its name
    with no value, and the SHA-512 of them. An arm's is of the environment its server is started with (`server_env`),
    a replay's of its engine's. What `servetax_plugin` is told (`PLUGIN_ENV`) is no setting of the registry's, so it is
    not among them: an arm's ``plugin``, ``edges`` and ``boundary_rows`` hold it."""
    if "verity_vllm.settings" not in sys.modules:
        sys.path.insert(0, str(ROOT / "adapters/vllm"))
    from verity_vllm import settings

    return settings.record(env)


def wait_ready(server: subprocess.Popen, port: int, startup_s: float) -> None:
    deadline, last = time.monotonic() + startup_s, None
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"the server exited with {server.returncode} before it was ready (server.log)")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except OSError as e:
            last = e
        time.sleep(1.0)
    raise RuntimeError(f"the server was not ready in {startup_s:g} s (its health check last: {last!r})")


def reset_prefix_cache(port: int, tries: int = 30) -> bool:
    """Empty the server's prefix cache between loads (a dev-mode route); False if it never could (blocks still held)."""
    for _ in range(tries):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/reset_prefix_cache", method="POST", data=b"")
        with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
            if json.loads(r.read() or b"{}").get("success", True):
                return True
        time.sleep(1.0)
    return False


#: a load's schedule, from the engine's counters on the server's Prometheus /metrics (summed over engines): its steps
#: (each step with outputs observes ``iteration_tokens_total`` once), the tokens they scheduled and sampled, the prompt
#: tokens the prefix cache served, preemptions, and under speculative decoding its drafts, the tokens they proposed
#: and the tokens of them the model accepted (an engine without it has no such counters, and they read 0)
SCHEDULE = {"steps": "vllm:iteration_tokens_total_count", "step_tokens": "vllm:iteration_tokens_total_sum",
            "generation_tokens": "vllm:generation_tokens_total", "prompt_tokens": "vllm:prompt_tokens_total",
            "prompt_tokens_cached": "vllm:prompt_tokens_cached_total", "preemptions": "vllm:num_preemptions_total",
            "drafts": "vllm:spec_decode_num_drafts_total", "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
            "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total"}


def counters(text: str) -> dict[str, float]:
    """SCHEDULE's counters in a Prometheus text exposition, each summed over its label sets."""
    tot = dict.fromkeys(SCHEDULE.values(), 0.0)
    for line in text.splitlines():
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in tot:
            tot[name] += float(line.rsplit(" ", 1)[1])
    return {k: tot[v] for k, v in SCHEDULE.items()}


def scrape(port: int) -> dict[str, float]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=30) as r:
        return counters(r.read().decode())


# `research cancel` and `--timeout` SIGKILL the job 20 s after its SIGTERM, and the server runs in a session of its own:
# a terminated arm kills the server's group before then, or the engine core and its workers keep the GPUs
TERM_GRACE_S = 5.0


def stop(server: subprocess.Popen, grace_s: float = 120.0) -> str:
    """Stop the server and every process of its group (the engine core), the server given ``grace_s`` to exit on SIGINT:
    ``killed`` if any of the group was left after the server exited, else ``exited``."""
    if server.poll() is None:
        server.send_signal(signal.SIGINT)
        try:
            server.wait(grace_s)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
    try:
        os.killpg(server.pid, signal.SIGKILL)
    except ProcessLookupError:
        return "exited"
    return "killed"


# ----------------------------------------------------------------------------------------------------------- the plan

def load_docs(c4: Path, checkpoint: Path, n: int) -> list[list[int]]:
    """The first n documents of a C4 shard as the checkpoint's token ids: its tokenizer.json where it has one (the corpus's
    reader), else its own tokenizer through transformers (Moonlight's tiktoken)."""
    if (checkpoint / "tokenizer.json").exists():
        return corpus().load_docs(c4, checkpoint / "tokenizer.json", n)
    import gzip

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(checkpoint), trust_remote_code=True)
    out = []
    with gzip.open(c4, "rt", encoding="utf-8") as f:
        for line in f:
            out.append(tok.encode(json.loads(line)["text"], add_special_tokens=False))
            if len(out) == n:
                break
    return out


def plan_of(f: Path, c4: Path, checkpoint: Path, n: int, docs: int, seed: int) -> list[dict]:
    """The plan's first n requests from f, made there first (`traffic.plan` over the shard's first ``docs`` documents)
    unless f exists; refused if f holds fewer than n, never replaced."""
    if f.exists():
        plan = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        if len(plan) < n:
            raise SystemExit(f"the plan at {f} holds {len(plan)} requests and these loads may send {n}: give a longer "
                             "plan, or fewer --rates, a shorter --span-s or --requests")
        return plan[:n]
    plan = corpus().traffic.plan(load_docs(c4, checkpoint, docs), n, seed)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in plan))
    return plan


def plan_need(rates: dict[str, float], span_s: float, requests: int | None) -> int:
    """Requests a run of these loads may send: a load's arrivals over its span, with room for Poisson's spread."""
    return requests or max(math.ceil(r * span_s * 1.5) + 64 for r in rates.values())


def parse_rates(text: str) -> dict[str, float]:
    out = {}
    for part in filter(None, text.split(",")):
        name, _, r = part.partition("=")
        out[name] = float(r)
    return out


# ------------------------------------------------------------------------------------------------------------ run

def nvml() -> tuple[object | None, str | None]:
    """NVML's bindings (vLLM's own copy, else nvidia-ml-py's), initialized, or None and why not (neither installed, or no
    driver answered)."""
    import importlib
    import importlib.util

    def found(m: str) -> bool:
        parts = m.split(".")
        return all(importlib.util.find_spec(".".join(parts[:k])) is not None for k in range(1, len(parts) + 1))

    name = next((m for m in ("vllm.third_party.pynvml", "pynvml") if found(m)), None)
    if name is None:
        return None, "no NVML bindings (vllm.third_party.pynvml, pynvml)"
    lib = importlib.import_module(name)
    try:
        lib.nvmlInit()
    except Exception as e:  # noqa: BLE001 -- NVMLError's class lives in the module that failed to initialize
        return None, f"{name}.nvmlInit: {e!r}"
    return lib, None


class GpuBusy:
    """The busy GPU-seconds of a load: each visible GPU's NVML utilization (the share of its last sample period in which
    a kernel ran) read every ``period_s`` on a thread of its own, integrated over the time between reads. The visible
    GPUs are ``CUDA_VISIBLE_DEVICES``'s, by index or UUID, else all of the node's."""

    def __init__(self, lib: object | None, visible: str | None = None, period_s: float = 0.2,
                 clock: Callable[[], float] = time.monotonic) -> None:
        import threading

        self.lib, self.period_s, self.clock = lib, period_s, clock
        self.handles: list = []
        if lib is not None:
            ids = [x.strip() for x in (visible or "").split(",") if x.strip()] if visible else \
                [str(i) for i in range(lib.nvmlDeviceGetCount())]
            self.handles = [lib.nvmlDeviceGetHandleByUUID(x) if x.startswith(("GPU-", "MIG-")) else
                            lib.nvmlDeviceGetHandleByIndex(int(x)) for x in ids]
        self.busy = [0.0] * len(self.handles)
        self.last: float | None = None
        self.reads = 0
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def sample(self) -> None:
        now = self.clock()
        util = [self.lib.nvmlDeviceGetUtilizationRates(h).gpu / 100.0 for h in self.handles]
        if self.last is not None:
            for i, u in enumerate(util):
                self.busy[i] += u * (now - self.last)
        self.last = now
        self.reads += 1

    def __enter__(self) -> GpuBusy:
        if self.handles:
            import threading

            self.sample()
            self.thread = threading.Thread(target=self._run, name="serve-tax-gpu-busy", daemon=True)
            self.thread.start()
        return self

    def _run(self) -> None:
        while not self.stop.wait(self.period_s):
            self.sample()

    def __exit__(self, *exc: object) -> None:
        if self.thread is not None:
            self.stop.set()
            self.thread.join()
            self.sample()

    def result(self) -> dict | None:
        if not self.handles:
            return None
        return {"gpus": len(self.handles), "busy_gpu_s": round(sum(self.busy), 3),
                "busy_s_per_gpu": [round(b, 3) for b in self.busy], "reads": self.reads, "period_s": self.period_s}


def drive_load(port: int, served: str, plan: list[dict], rate: float, span_s: float, tag: str, d: Path, drain_s: float,
               tick: dict | None, requests: int | None = None, sampling: str = "greedy", lib: object | None = None) -> dict:
    """One load: the plan's first requests under Poisson arrivals at ``rate`` for ``span_s`` (or ``requests`` of them),
    sent by `corpus.drive`; its client records in d/client.jsonl, and with NVML's bindings ``lib`` the GPUs' busy time
    over it (`GpuBusy`)."""
    C = corpus()
    d.mkdir(parents=True, exist_ok=True)
    arr = C.traffic.arrivals(plan, rate)
    k = requests or max(1, sum(1 for a in arr if a <= span_s))
    reqs, arr = plan[:k], arr[:k]
    ids = [f"{tag}-{r['i']:05d}" for r in reqs]
    t = time.monotonic()
    with GpuBusy(lib, SHARED.CUDA_VISIBLE_DEVICES) as busy:
        got = asyncio.run(C.drive(port, served, reqs, arr, ids, d / "client.jsonl", drain_s, sampling,
                                  *([tick["bucket_us"] / 1e6] if tick else [])))
    meta = {"tag": tag, "rate": rate, "span_s": span_s, "requests": k, "arrivals_span_s": round(arr[-1], 3),
            "wall_s": round(time.monotonic() - t, 3), **got, "gpu": busy.result()}
    recs = [json.loads(x) for x in (d / "client.jsonl").read_text().splitlines() if x.strip()]
    meta["unserved"] = len(recs) - len(served_records(recs))
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


# ------------------------------------------------------------------------------------------------- vLLM's own client

#: the clients a load is sent by: the trace corpus's (`corpus.drive`) and vLLM's own (`vllm bench serve`)
CORPUS, BENCH = "corpus", "vllm-bench"
#: the file vLLM's client saves a pass's result to
RESULT = "result.json"
#: the load that repeats a prefix (`LOADS`)
PREFIX = "prefix"
#: in a rung's place, a pass whose requests leave at their planned arrival times, however many are then in flight
ARRIVING = 0


@dataclass(frozen=True)
class Dataset:
    """A dataset file as the Hub holds it: its repository, the revision it is pinned at, its file there, its size and
    its SHA-512. A node keeps one copy, in its Hugging Face cache where a checkpoint's snapshot lies too (`path`),
    outside every run's directory."""
    repo: str
    revision: str
    file: str
    size: int
    sha512: str

    def path(self) -> Path:
        hub = Path(SHARED.HF_HOME or "/workspace/hf") / "hub"
        return hub / f"datasets--{self.repo.replace('/', '--')}" / "snapshots" / self.revision / self.file

    def url(self) -> str:
        return f"https://huggingface.co/datasets/{self.repo}/resolve/{self.revision}/{self.file}"


#: a line of a planned load's table (`read_plan`, `Planned`): a request's arrival in seconds after its pass starts and
#: its prompt's and its output's tokens; and, on a request whose prompt begins as an earlier one's does, that line's
#: place in the table, counted from 0, and the whole blocks the two share
PLAN_KEYS, REPEAT_KEYS = ("at_s", "prompt", "output"), ("after", "blocks")
#: the tokens of one block of a planned prompt. vLLM's client draws a block's tokens from the block's id, so two
#: prompts that begin with the same ids begin with the same tokens; a multiple of the engine's prefix-cache block
BLOCK = 64


@dataclass(frozen=True)
class Planned:
    """One request of a planned load: when it arrives, in seconds after its pass starts; its prompt's and its output's
    tokens; and, where its prompt begins as an earlier request's does, that request's place in the load and the whole
    blocks (`BLOCK`) the two share."""
    at_s: float
    prompt: int
    output: int
    after: int | None = None
    blocks: int = 0


@dataclass(frozen=True)
class Load:
    """A load of vLLM's own client: what it is, as the cost's output says it; its dataset's flags; the prompts a pass
    sends; the concurrencies it is served at (its rungs: the client holds that many requests in flight and sends the
    next when one ends); the context limit its requests fit, which the engines' is at least; its dataset's file, where
    it is one; the bounds of the model's SLO its goodput counts against, by the client's names; its requests, where
    they are planned one by one (`Planned`; the client's file is written from them, `write_trace`, or from the
    table ``--dataset NAME=FILE`` gives, `read_plan`); whether its cost enters the number (`cost_report`), which a
    load that repeats a prefix never does; and whether its requests are sessions' (`session_lines`, from its file):
    its rung is then its users, each a client of its own that holds one request in flight and sends its sessions'
    turns in order, and its prompts are the turns each user sends in a pass."""
    what: str
    dataset: tuple[str, ...]
    prompts: int
    rungs: tuple[int, ...]
    context: int
    file: Dataset | None = None
    slo: tuple[str, ...] = ("ttft", "tpot")
    plan: tuple[Planned, ...] = ()
    number: bool = True
    sessions: bool = False


#: two prompts near a 131,072-token context limit, each sent once and then again with another ending: the second of
#: each pair shares all but its last tokens with the first, in whole blocks. A row of these four alone is scaled down
#: (`prefix_row`): half of their prompt tokens repeat, which is this load's share and no serve's, where such pairs
#: arrive among requests that repeat nothing
PAIRS = (Planned(0.0, 130808, 256), Planned(0.0, 130807, 256, after=0, blocks=2043),
         Planned(0.0, 129016, 256), Planned(0.0, 129016, 256, after=2, blocks=2015))


#: the flags of vLLM's client's timed_trace dataset: a file of lengths and block ids, a block of `BLOCK` tokens
TIMED_TRACE = ("--dataset-name", "timed_trace", "--timed-trace-chunk-hash-size", str(BLOCK))
#: the sessions loads (`Load.sessions`): the agent's sessions cut at the context the serve holds today, whole, and
#: as their publisher cut them for the recipe's context
AGENTIC, AGENTIC_FULL, AGENTIC_256K = "agentic", "agentic-full", "agentic-256k"
#: SemiAnalysis's AgentX traces (Apache-2.0): recorded sessions of a coding agent, a line a session, each request its
#: prompt's and its output's tokens and an id for each block of `BLOCK` prompt tokens, with no token and no text
AGENTX = Dataset("semianalysisai/cc-traces-weka-062126", "23f152f6f0f9399a85901b89a6458def0ef16729", "traces.jsonl",
                 1847151435, "e9737b2661bf90958672b5aa7b133bf009bd2642d43fb3c22060e85b9f15a4ac"
                 "ed11b887571578abdfdc7b14df3b72a0825440628437dad16701697e48386470")
#: the same sessions as their publisher cut them for a context of 262,144 tokens (Apache-2.0), in the same form: no
#: request of it, with its output, passes that context
AGENTX_256K = Dataset("semianalysisai/cc-traces-weka-062126-256k", "8fecd2fc56694469f758f0afbbb6335ad3043740", "traces.jsonl",
                      568864747, "a6c659d9d773a3b00552266103937cad1cd6a7f1233fb98aa00f2ff6f706c4d7"
                      "7ebf1bf7b640485ba0df5cebf2321242e85ecf786182f98d32137d5a173d679d")
#: the fewest turns a session is sent with (`session_lines`): its opening request and the first to carry its context
SESSION_TURNS = 2


#: chat: the load of vLLM's standing serving tests, ShareGPT's first turns, each decoding its recorded answer's length
#: (the client keeps prompts of at most 1,024 tokens, 2,048 with the output). long: long context, the client's synthetic
#: dataset at one long length until a plan of recorded long requests replaces it; its first token takes seconds on any
#: engine, so its goodput counts the time per output token alone. prefix: the load that repeats a prefix, which chat
#: and long do not; its cost is the row under the number and never enters it. One request in flight, so each second
#: request starts once the prompt it repeats is computed; from a table of its own (``--dataset prefix=FILE``), its
#: requests leave as they arrive instead (`ARRIVING`). agentic, agentic-full and agentic-256k: sessions (`Load.sessions`), which
#: repeat most of every prompt; a cost on them is `session_cost` and never enters the number. agentic-256k sends half the
#: turns a user: at the others' count its four users' sessions together pass the KV cache the recipe's server has on
#: eight of these cards, and the prefix cache would evict what a later turn repeats.
LOADS = {
    "chat": Load("ShareGPT's first turns, each decoding its recorded answer's length (vLLM's sharegpt dataset)",
                 ("--dataset-name", "sharegpt"), 200, (4, 8, 12, 16, 24, 32, 48), 4096,
                 Dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", "192ab2185289094fc556ec8ce5ce1e8e587154ca",
                         "ShareGPT_V3_unfiltered_cleaned_split.json", 672837942,
                         "324efe241e9a5a284c94a9cd15776a8a037092528e44b4bc6aa7338f7c93ee80"
                         "62e859ae182c2b9440a1099f6f6f7659b1c3d4ffecf2672e0d9ff8771946d5db")),
    "long": Load("8,192 random tokens in and 128 out (vLLM's random dataset at one long length)",
                 ("--dataset-name", "random", "--random-input-len", "8192", "--random-output-len", "128"), 24,
                 (1, 2, 4, 8), 16384, slo=("tpot",)),
    PREFIX: Load("two prompts near the context limit, each sent again with another ending, of synthetic token ids "
                 "(vLLM's timed_trace dataset)", TIMED_TRACE,
                 len(PAIRS), (1,), 131072, slo=("tpot",), plan=PAIRS, number=False),
    AGENTIC: Load("a coding agent's recorded sessions, each turn's prompt beginning as the turn before it, cut where a "
                  "request passes 131,072 tokens, of synthetic token ids (SemiAnalysis's AgentX traces; vLLM's "
                  "timed_trace dataset)", TIMED_TRACE, 24, (4,), 131072, AGENTX, slo=("tpot",), number=False,
                  sessions=True),
    AGENTIC_FULL: Load("a coding agent's recorded sessions, each turn's prompt beginning as the turn before it, whole "
                       "up to the model's own context limit, of synthetic token ids (SemiAnalysis's AgentX traces; "
                       "vLLM's timed_trace dataset)", TIMED_TRACE, 24, (1,), 1048576, AGENTX, slo=("tpot",),
                       number=False, sessions=True),
    AGENTIC_256K: Load("a coding agent's recorded sessions, each turn's prompt beginning as the turn before it, as their "
                       "publisher cut them for a context of 262,144 tokens, of synthetic token ids (SemiAnalysis's "
                       "AgentX traces, the 256k copy; vLLM's timed_trace dataset)", TIMED_TRACE, 12, (4,), 262144,
                       AGENTX_256K, slo=("tpot",), number=False, sessions=True),
}


#: the loads whose costs the number is the largest of
NUMBER = tuple(n for n, x in LOADS.items() if x.number)


def pass_name(load: str, rung: int) -> str:
    return f"{load}@{rung}"


def at_rung(rung: int) -> str:
    """How a pass at ``rung`` is sent, as the outputs say it."""
    return "as they arrive" if rung == ARRIVING else f"at concurrency {rung}"


def load_tag(name: str, rep: int, bench: bool) -> str:
    """What a load's request ids start with in rep ``rep``. A pass's name ends in its rung, so its rep is set apart."""
    return f"{name}-r{rep}" if bench else f"{name}{rep}"


def trace_lines(plan: Sequence[Planned]) -> list[dict]:
    """A planned load as vLLM's client's timed_trace dataset reads it, a line a request, under the client's names: its
    arrival, its prompt's and its output's tokens, and an id for each block of its prompt. A block's id is its own
    request's, except the blocks a request shares with an earlier one, which carry that request's ids: the client draws
    the same tokens for them. A line holds no token and no text."""
    out: list[dict] = []
    for k, r in enumerate(plan):
        ids = [k << 32 | j for j in range(-(-r.prompt // BLOCK))]
        if r.after is not None:
            if not 0 <= r.after < k or not 0 < r.blocks * BLOCK <= min(r.prompt, plan[r.after].prompt):
                raise ValueError(f"request {k} shares {r.blocks} blocks with request {r.after}: an earlier request, "
                                 "and blocks both prompts have")
            ids[:r.blocks] = out[r.after]["hash_ids"][:r.blocks]
        out.append({"timestamp": r.at_s, "input_length": r.prompt, "output_length": r.output, "hash_ids": ids})
    return out


def planned(lines: Sequence[dict]) -> dict:
    """What a planned load's file (`trace_lines`) plans: its requests, their prompt tokens, the prompt
    tokens that begin a request as an earlier one began (the whole leading blocks whose ids an earlier line begins
    with), which a prefix cache serves where it still holds the earlier prompt, and the seconds its requests arrive
    over."""
    after: dict[tuple[int, int], int] = {}
    repeated = 0
    for x in lines:
        node, shared = 0, True
        for h in x["hash_ids"][:x["input_length"] // BLOCK]:
            shared = shared and (node, h) in after
            repeated += BLOCK * shared
            node = after.setdefault((node, h), len(after) + 1)
    return {"requests": len(lines), "prompt_tokens": sum(x["input_length"] for x in lines),
            "repeated_tokens": repeated, "span_s": max(x["timestamp"] for x in lines)}


def write_trace(f: Path, lines: Sequence[dict]) -> str:
    """Write a planned load's file for vLLM's client (`trace_lines`) at ``f``: its path."""
    f.write_text("".join(json.dumps(x) + "\n" for x in lines))
    return str(f)


def session_lines(f: Path, users: int, turns: int, context: int) -> list[dict]:
    """A sessions load's requests from its corpus ``f`` (`AGENTX`'s form: a line a session, its requests in order,
    each with its prompt's and its output's tokens and an id for each block of its prompt), as vLLM's client's
    timed_trace dataset reads them (`trace_lines`'s names), each with its ``user``, ``turns`` a user. A session is
    its own chain of requests, without the groups its subagents sent, up to the first request that does not fit
    ``context`` with its output; one with fewer than `SESSION_TURNS` such turns is left out. Sessions are dealt in
    the file's order, each to the user with the fewest turns so far, until every user has ``turns``, and a user's
    last session is cut there. A block's id is its session's, so two sessions share no block and a turn repeats only
    its own session's earlier prompts. Refused where the corpus is not in blocks of `BLOCK` tokens or holds too few
    sessions."""
    queues: list[list[dict]] = [[] for _ in range(users)]
    with open(f) as lines:
        for k, text in enumerate(lines):
            if all(len(q) >= turns for q in queues):
                break
            if not text.strip():
                continue
            s = json.loads(text)
            if s.get("block_size") != BLOCK:
                raise SystemExit(f"{f}: line {k + 1}: a session in blocks of {s.get('block_size')} tokens, not {BLOCK}")
            own = []
            for r in s["requests"]:
                if "requests" in r:
                    continue
                if not (r["in"] > 0 and r["out"] > 0 and r["in"] + r["out"] <= context):
                    break
                own.append({"timestamp": 0.0, "input_length": r["in"], "output_length": r["out"],
                            "hash_ids": [k << 32 | h for h in r["hash_ids"]]})
            if len(own) >= SESSION_TURNS:
                min(queues, key=len).extend(own)
    if any(len(q) < turns for q in queues):
        raise SystemExit(f"{f}: too few sessions for {users} users of {turns} turns each within a context of {context:,} "
                         f"tokens: the shortest has {min(map(len, queues))}")
    return [{**x, "user": j} for j, q in enumerate(queues) for x in q[:turns]]


def user_files(dataset: str, users: int) -> list[str]:
    """A sessions load's file (`session_lines`) as its first ``users`` users' own files, beside it and written once:
    a client reads one user's requests, in the user's order. Refused where the file holds fewer users."""
    out, by = [], None
    for j in range(users):
        f = Path(dataset).with_name(f"{Path(dataset).name.removesuffix('.jsonl')}.user-{j}.jsonl")
        if not f.exists():
            if by is None:
                by = {}
                for x in map(json.loads, filter(str.strip, Path(dataset).read_text().splitlines())):
                    by.setdefault(x["user"], []).append(x)
            if j not in by:
                raise SystemExit(f"{dataset} holds {len(by)} users' requests, and the pass is of {users}")
            write_trace(f, by[j])
        out.append(str(f))
    return out


def read_plan(name: str, f: Path) -> tuple[Planned, ...]:
    """Planned load ``name``'s table as given (``--dataset NAME=FILE``), a line a request (`Planned`), checked: its
    arrival, its prompt's and its output's tokens (`PLAN_KEYS`), with the earlier line it begins as and the blocks the
    two share where it repeats one (`REPEAT_KEYS`), and nothing else, so the table holds no token and no text; whole
    lengths, the request within the load's context limit; and arrivals from zero on in the table's order, which is
    the order the client sends in. Refused otherwise."""
    def no(k: int, why: str) -> SystemExit:
        return SystemExit(f"--dataset {name}={f}: line {k}: {why}")

    def whole(x: object) -> bool:
        return isinstance(x, int) and not isinstance(x, bool)

    out: list[Planned] = []
    for k, text in enumerate(filter(str.strip, f.read_text().splitlines()), 1):
        try:
            x = json.loads(text)
        except ValueError:
            raise no(k, "not a JSON object") from None
        if not isinstance(x, dict) or set(x) not in ({*PLAN_KEYS}, {*PLAN_KEYS, *REPEAT_KEYS}):
            raise no(k, f"a request is {', '.join(PLAN_KEYS)}, with {' and '.join(REPEAT_KEYS)} where it begins as an "
                        "earlier one does, and nothing else")
        t, n, m = (x[key] for key in PLAN_KEYS)
        if not (whole(n) and whole(m) and n > 0 and m > 0 and n + m <= LOADS[name].context):
            raise no(k, f"a prompt and an output of whole tokens, {LOADS[name].context:,} at most together")
        if not (isinstance(t, (int, float)) and not isinstance(t, bool) and (out[-1].at_s if out else 0) <= t):
            raise no(k, "an arrival in seconds, from zero on and no earlier than the line's before it")
        i, shared = x.get("after"), x.get("blocks", 0)
        if i is not None and not (whole(i) and whole(shared) and 0 <= i < len(out)
                                  and 0 < shared * BLOCK <= min(n, out[i].prompt)):
            raise no(k, f"after: an earlier line, counted from 0; blocks: whole blocks of {BLOCK} tokens both prompts have")
        out.append(Planned(t, n, m, i, shared))
    if not out:
        raise SystemExit(f"--dataset {name}={f}: no request")
    return tuple(out)


def fetch(ds: Dataset, to: Path, open_url: Callable[[str], object] = urllib.request.urlopen) -> None:
    """Download ``ds`` to ``to``: streamed to a file beside it, hashed as it is written, and moved into place only
    where its SHA-512 is the pinned one, so no other file is ever at ``to``."""
    to.parent.mkdir(parents=True, exist_ok=True)
    tmp, h = to.with_name(f".{to.name}.{os.getpid()}"), hashlib.sha512()
    try:
        with open_url(ds.url()) as r, open(tmp, "wb") as f:
            for chunk in iter(lambda: r.read(1 << 22), b""):
                h.update(chunk)
                f.write(chunk)
        if h.hexdigest() != ds.sha512:
            raise SystemExit(f"{ds.url()} is not the pinned file: its SHA-512 is {h.hexdigest()[:16]}..., not "
                             f"{ds.sha512[:16]}... (nothing kept)")
        tmp.replace(to)
    finally:
        tmp.unlink(missing_ok=True)


def parse_datasets(given: list[str] | None, loads: dict[str, tuple[int, ...]], fetch_missing: bool = False,
                   open_url: Callable[[str], object] = urllib.request.urlopen) -> dict[str, str]:
    """Each of ``loads``' dataset files, for the loads that read one, checked: the node's copy of the pinned file
    (`Dataset.path`), which ``fetch_missing`` downloads where there is none, or ``--dataset NAME=FILE``'s copy of it.
    A file whose SHA-512 is not the pinned one is refused, so every arm and run reads the same bytes. A planned
    load reads a file only where ``--dataset NAME=FILE`` gives its table (`passes_of` reads it)."""
    files = dict(x.split("=", 1) if "=" in x else (x, "") for x in given or ())
    extra = sorted(n for n in files if n not in loads or (LOADS[n].file is None and not LOADS[n].plan))
    if extra:
        raise SystemExit(f"--dataset: {', '.join(extra)}: not a load of --loads that reads a file")
    out = {}
    for name in loads:
        ds = LOADS[name].file
        if ds is None:
            if name in files:
                if not Path(files[name]).is_file():
                    raise SystemExit(f"--dataset {name}={files[name]}: no such file")
                out[name] = files[name]
            continue
        f = Path(files.get(name) or ds.path())
        if not f.is_file() and fetch_missing and name not in files:
            print(f"[serve-tax] load {name}: downloading {ds.url()} to {f}", file=sys.stderr, flush=True)
            fetch(ds, f, open_url)
        if not f.is_file():
            raise SystemExit(f"load {name}: no dataset file at {f}: `serve_tax.py passes --loads {name} --fetch` downloads "
                             f"{ds.repo}'s {ds.file} at {ds.revision[:12]} there, or --dataset {name}=FILE names a copy")
        if (got := file_digest(f, "sha512")) != ds.sha512:
            raise SystemExit(f"load {name}: {f} is not the pinned dataset file ({ds.repo}'s {ds.file} at "
                             f"{ds.revision[:12]}): its SHA-512 is {got[:16]}..., not {ds.sha512[:16]}...")
        out[name] = str(f)
    return out


def client_files(datasets: dict[str, str], traces: dict[str, str]) -> dict[str, str]:
    """The file vLLM's client reads for each load: a planned load's own, written from its plan or from its given table
    (``traces``, `write_trace`), and never that table, which is not in the client's format; else the load's dataset
    file (``datasets``, `parse_datasets`)."""
    return {**datasets, **traces}


def passes_of(loads: dict[str, tuple[int, ...]], datasets: dict[str, str],
              sizes: dict[str, int] | None = None) -> tuple[dict[str, tuple[int, ...]], dict[str, list[dict]]]:
    """``loads`` with the rungs their passes are sent at, and each planned load's lines (`trace_lines`): a planned
    load whose table is given (``datasets``, `parse_datasets`) is sent as its requests arrive (`ARRIVING`,
    `read_plan`); one with none, at its own rung from its own plan. A sessions load's lines are its corpus's
    (`session_lines`), for the users of its highest rung, each sending the load's turns or ``sizes``'s
    (``--pass-prompts``), at its own rungs."""
    lines = {n: trace_lines(read_plan(n, Path(datasets[n])) if n in datasets else LOADS[n].plan)
             for n in loads if LOADS[n].plan}
    held = {n: session_lines(Path(datasets[n]), max(loads[n]), (sizes or {}).get(n) or LOADS[n].prompts, LOADS[n].context)
            for n in loads if LOADS[n].sessions}
    return {n: (ARRIVING,) if n in lines and n in datasets else rungs for n, rungs in loads.items()}, lines | held


def pass_prompts(load: Load, rung: int, lines: Sequence[dict] | None, warm_s: float | None = None,
                 prompts: int | str | None = None) -> int:
    """The prompts a pass of ``load`` at ``rung`` sends: ``prompts`` (``--pass-prompts``, `parse_pass_prompts`: a
    number, the same at every rung, or ``Kx``, K for each request the rung holds in flight and the load's own at
    least) or the load's, or every line of its file where they leave as they arrive (`ARRIVING`). A warm-up
    (``warm_s``) sends two for each request held in flight, the load's own prompts at most, or the lines that arrive
    in its first ``warm_s`` seconds, one at least. A sessions load's are the turns each of its users sends: a
    warm-up's are `SESSION_TURNS`, each user's first session's opening and the first prompt that carries its
    context."""
    if load.sessions:
        return (prompts or load.prompts) if warm_s is None else min(SESSION_TURNS, prompts or load.prompts)
    if rung == ARRIVING:
        return len(lines) if warm_s is None else max(1, sum(x["timestamp"] <= warm_s for x in lines))
    n = max(load.prompts, int(prompts[:-1]) * rung) if isinstance(prompts, str) else prompts or load.prompts
    return n if warm_s is None else min(load.prompts, n, 2 * rung)


def parse_pass_prompts(text: str | None, loads: dict[str, tuple[int, ...]]) -> dict[str, int | str]:
    """``NAME=N,...`` (``--pass-prompts``): the prompts a pass of load NAME sends, in place of the load's own, at
    every rung alike; or ``NAME=Kx``, K prompts for each request the rung holds in flight, the load's own at least
    (`pass_prompts`). A load's own prompts are few for a high rung: the client starts a pass with ``rung`` requests
    and ends it with its last ones draining a few at a time, so 200 prompts at rung 96 hold 96 for part of the pass
    only, and no rung shows more than the load's prompts in flight. A ladder that climbs past them needs longer
    passes. N is the same length at every rung: the draining end is a smaller share of a longer pass, so passes of two
    sizes at one rung differ in tokens a second by their size alone, and under one N that share grows with the rung.
    ``Kx`` sizes each rung's pass by the rung, so that the draining end is the same share of the pass at every rung:
    for a ladder read for the rung where tokens a second stop rising, at a fraction of the time one N large enough
    for its top rung takes at its low ones. Either way a rung has one size, the same for every arm. Refused for a
    load that is not of ``loads`` or that is sent as planned, whose passes are its plan's."""
    out: dict[str, int | str] = {}
    for part in filter(None, (p.strip() for p in (text or "").split(","))):
        name, _, n = part.partition("=")
        if name not in loads:
            raise SystemExit(f"--pass-prompts: {name!r} is not a load of --loads ({', '.join(loads)})")
        if LOADS[name].plan:
            raise SystemExit(f"--pass-prompts: {name}'s requests are sent as planned: a pass of it is its plan's")
        k = n.removesuffix("x")
        if not k.isdigit() or int(k) < 1:
            raise SystemExit(f"--pass-prompts: {part!r}: NAME=N, the prompts a pass of the load sends, or NAME=Kx, K "
                             "for each request its rung holds in flight, 1 or more")
        out[name] = int(n) if n == k else f"{int(k)}x"
    return out


def context_of(loads: dict[str, tuple[int, ...]], given: int | str | None) -> int | str | None:
    """The engines' context limit for vLLM's client's ``loads``: ``given`` (``--max-model-len``), refused where a
    load's requests don't fit it; where none is given, `BASE_ARGS`'s (None), or the largest a load needs past it.
    `AUTO` is the engine's to resolve when it starts, so no load is checked against it: a request past the limit it
    resolves to is one the server refuses, and the pass counts it as failed."""
    if given == AUTO:
        return AUTO
    name = max(loads, key=lambda n: LOADS[n].context)
    need = LOADS[name].context
    if given is not None and given < need:
        raise SystemExit(f"--max-model-len {given}: load {name}'s requests need a context of {need} (run.sh: MAX_MODEL_LEN)")
    return given if given is not None or need <= int(BASE_ARGS[BASE_ARGS.index("--max-model-len") + 1]) else need


def parse_loads(text: str) -> dict[str, tuple[int, ...]]:
    """``NAME[=RUNG+RUNG...],...``: each load (`LOADS`) with the rungs it is served at, its own where none are given."""
    out: dict[str, tuple[int, ...]] = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        name, _, rungs = part.partition("=")
        if name not in LOADS:
            raise SystemExit(f"--loads: {name!r} is not a load ({', '.join(sorted(LOADS))})")
        out[name] = tuple(int(x) for x in rungs.split("+")) if rungs else LOADS[name].rungs
        if not out[name] or min(out[name]) < 1:
            raise SystemExit(f"--loads: {part!r}: each rung is a concurrency of 1 or more")
        if LOADS[name].plan and out[name] != LOADS[name].rungs:
            raise SystemExit(f"--loads: {part!r}: {name}'s requests are sent as planned, at concurrency "
                             f"{'+'.join(map(str, LOADS[name].rungs))}, where each request that repeats a prefix "
                             f"follows the one it repeats: name it without rungs")
    if not out:
        raise SystemExit("--loads: NAME[=RUNG+RUNG...],... names no load")
    # sessions loads share a run: a cost is of each (`session_cost`), and the engine's context, the largest of theirs,
    # holds every one's requests
    beside = [n for n in out if not LOADS[n].number]
    if beside and len(out) > 1 and not all(LOADS[n].sessions for n in out):
        raise SystemExit(f"--loads: {beside[0]} is served in a run of its own: every load of a run is served by one "
                         f"engine, and {beside[0]}'s context limit of {LOADS[beside[0]].context} would be the others' too")
    return out


#: vLLM's client's environment beside the job's own: a planned load's tokens are drawn under Python's hash of each
#: block's id, which differs from one process to the next unless its seed is set
CLIENT_ENV = {"PYTHONHASHSEED": "0"}

#: a sampled pass's sampling (``run --sampled``), the recipe's own: what vLLM's client sends in place of temperature 0
SAMPLED = {"temperature": 1.0, "top_p": 0.95}


def user_seed(user: str, lifetime: int | None = None) -> int:
    """The seed a sampled pass's client sends in each of its requests to an arm that takes one (`sends_seed`), beside
    its name (the body's ``user``): the first 63 bits of SHA-512 over the name, which is the pass's tag and, on a
    sessions load, the user's, and over the server lifetime's number where the arm is served in lifetimes
    (`run --lifetime`). A pass's users have the same names in every lifetime, and under one seed a server samples
    much the same tokens again, so a seed of the name alone would have the arm's lifetimes serve one sample of text
    where a stock arm's serve a new one each, and an interval over the lifetimes would leave that sample's chance
    out. A replay samples the request again under the seed. It is the client's seed: a served request need not send
    one."""
    name = user if lifetime is None else f"{user}/lifetime-{lifetime}"
    return int.from_bytes(hashlib.sha512(b"serve-tax/user-seed/v1/" + name.encode()).digest()[:8], "big") >> 1


def sends_seed(arm: Arm) -> bool:
    """Whether a sampled pass's client sends ``arm`` each user's seed (`user_seed`): where the arm keeps the record,
    whose replay samples each request again under the request's own seed. A stock arm is sent none, as a served
    request carries none, so vLLM samples its requests as it samples any without a seed; what a seed changes in
    vLLM's sampling is then on our side of a cost, as a change of ours."""
    return "record" in arm.plugin


def bench_argv(model: Model, checkpoint: Path, port: int, load: Load, rung: int, d: Path, tag: str,
               dataset: str | None = None, prompts: int | None = None, seed: int = 0, sampled: bool = False,
               seeded: bool = False, lifetime: int | None = None) -> list[str]:
    """One pass of vLLM's client against the server on ``port``: ``load``'s prompts (the same ones under one seed, so
    every arm and rep serves the same requests) at concurrency ``rung``, greedy and each decoding to its length, with
    goodput against the model's SLO and its result saved to ``d``. A ``sampled`` pass's requests are sampled instead
    (`SAMPLED`), each naming its client as its ``user`` (``tag``) and, where ``seeded``, sending that user's seed
    in server lifetime ``lifetime`` (`user_seed`). The client sends no completion of its own before
    the pass (no ready check: `wait_ready` has waited), so every completion it sends is in its result, and request k's
    X-Request-Id is ``{tag}-{k}``, the form the release-order arm reads its release number from
    (`servetax_plugin.release_key`). It names the served model in every request and takes its tokenizer from the
    checkpoint; for its synthetic datasets it first asks the server's ``/tokenize``, which the engine never sees,
    whether the server counts the first prompt's tokens as it does, and trims every prompt to the server's count
    where it doesn't. A planned load (`Load.plan`) is sent from its file (``dataset``, `write_trace`) in the file's
    order, each prompt as the token ids the client draws for its blocks (`CLIENT_ENV`), so the server counts each
    prompt's planned length: held at ``rung`` in flight whatever the file's arrivals (``--no-self-timed``), or, at
    `ARRIVING`, each sent at its arrival after the pass starts, which is the client's way with such a file, with no
    bound on the requests in flight. A sessions load's client is one user's (`drive_bench`): its file the user's own,
    ``rung`` 1."""
    slo = dict(zip(("ttft", "tpot"), model.slo))
    held = [] if rung == ARRIVING else [*(["--no-self-timed"] if load.plan or load.sessions else []),
                                        "--request-rate", "inf", "--max-concurrency", str(rung)]
    sampling = ["--temperature", "0"] if not sampled else [
        "--temperature", str(SAMPLED["temperature"]), "--top-p", str(SAMPLED["top_p"]),
        "--extra-body", json.dumps({"user": tag, **({"seed": user_seed(tag, lifetime)} if seeded else {})})]
    return [sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve", "--backend", "openai",
            "--base-url", f"http://127.0.0.1:{port}", "--endpoint", "/v1/completions",
            "--model", model.served, "--tokenizer", str(checkpoint),
            *(["--trust-remote-code"] if "--trust-remote-code" in model.args else []),
            *load.dataset, *(["--dataset-path", dataset] if dataset else []),
            "--num-prompts", str(prompts or load.prompts), "--seed", str(seed),
            *held, "--ignore-eos", *sampling,
            "--goodput", *(f"{k}:{round(slo[k] * 1e3)}" for k in load.slo),
            "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "99",
            "--request-id-prefix", f"{tag}-", "--ready-check-timeout-sec", "0", "--disable-tqdm",
            "--save-result", "--save-detailed", "--result-dir", str(d), "--result-filename", RESULT]


#: what the client's detailed result holds that no run keeps: each request's generated text, which is a served
#: request's content, and its inter-token times, a number for every token, which the client's percentiles summarize
UNKEPT = frozenset({"generated_texts", "itls"})


def merged(results: Sequence[dict]) -> dict:
    """The results vLLM's clients saved for one pass of a sessions load, a client a user, as one in the client's keys
    (`bench_metrics`): requests and tokens summed, and each rate over the pass's seconds (``duration``), from its
    users' first send to the last one's last answer: the time the engine took to serve the pass's requests, with the
    seconds a client takes to start in no rate. Where each result is the client's detailed one they are read from
    the requests' own start times and latencies, which the clients take from one clock, the host's monotonic one;
    else they are the longest user's own seconds, which leave out how long after the first user it began. The users
    end at different times, so the sum of each user's own rate is another figure, which counts a user that ended early
    as if it served the whole pass: it is kept beside (``users_output_throughput``), as are the users' seconds summed
    (``users_duration``). ``met`` is the requests within the SLO. Where each result is the client's detailed one, the
    time to the first token and the time a token after it (a request's latency past its first token over its other
    tokens, as the client has it) are over every user's requests, under the client's names: their medians, P90s and
    P99s, in milliseconds. It has no other median or percentile, which the users' own don't give."""
    def total(key: str) -> float:
        return sum(r[key] for r in results)

    good = [r.get("request_goodput") for r in results]
    seconds = max(r["duration"] for r in results)
    if all(r.get("start_times") and len(r.get("latencies") or ()) == len(r["start_times"]) for r in results):
        sent = [(at, at + took) for r in results for at, took in zip(r["start_times"], r["latencies"])]
        seconds = max(end for _, end in sent) - min(at for at, _ in sent)
    met = None if None in good else sum(round(g * r["duration"]) for g, r in zip(good, results))
    out = {"users": len(results), "completed": total("completed"), "failed": sum(int(r.get("failed") or 0) for r in results),
           "total_output_tokens": total("total_output_tokens"), "total_input_tokens": total("total_input_tokens"),
           "duration": seconds, "output_throughput": total("total_output_tokens") / seconds,
           "total_token_throughput": (total("total_output_tokens") + total("total_input_tokens")) / seconds,
           "request_throughput": total("completed") / seconds,
           "request_goodput": None if met is None else met / seconds, "met": met,
           "users_output_throughput": total("output_throughput"), "users_duration": total("duration")}
    if all(isinstance(r.get(k), list) for r in results for k in ("ttfts", "latencies", "output_lens")):
        first = [1e3 * x for r in results for x in r["ttfts"]]
        after = [1e3 * (whole - x) / (n - 1) for r in results
                 for whole, x, n in zip(r["latencies"], r["ttfts"], r["output_lens"]) if n > 1]
        for name, xs in (("ttft", first), ("tpot", after)):
            out |= {f"median_{name}_ms": pct(xs, 0.5), f"p90_{name}_ms": pct(xs, 0.9), f"p99_{name}_ms": pct(xs, 0.99)}
    return out


def drive_bench(port: int, model: Model, checkpoint: Path, load: Load, rung: int, tag: str, d: Path,
                dataset: str | None = None, seed: int = 0, lib: object | None = None, prompts: int | None = None,
                sampled: bool = False, seeded: bool = False, lifetime: int | None = None,
                run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                clock: Callable[[], float] = time.monotonic, now: Callable[[], float] = time.time) -> dict:
    """One pass sent by vLLM's client (`bench_argv`): its output in d/bench.log, the result it saved in d/result.json,
    and d/meta.json in `drive_load`'s form (a request the client counted failed is unfinished and unserved). The
    result is the client's detailed one, each request's lengths, start, first-token time and latency in the order
    sent, less what no run keeps (`UNKEPT`): the client saves it in a directory outside the run's, removed when the
    pass ends, so that the generated texts never lie under ``d`` whatever stops the pass. A pass of a sessions load
    is sent by ``rung`` clients at once, a client a user (`user_files`), each with one request in flight and
    ``prompts`` turns to send, its log and result in d/user-<j>/: d/result.json is theirs as one (`merged`)."""
    d.mkdir(parents=True, exist_ok=True)
    if load.sessions:
        jobs = [(d / f"user-{j}", f"{tag}-u{j}", 1, f) for j, f in enumerate(user_files(dataset, rung))]
    else:
        jobs = [(d, tag, rung, dataset)]
    with tempfile.TemporaryDirectory(prefix="serve-tax-result.") as tmp:
        saved, held = Path(tmp).resolve(), SHARED.RESEARCH_RUN_DIR
        if any(saved.is_relative_to(p) for p in (d.resolve(), *([Path(held).resolve()] if held else []))):
            raise RuntimeError(f"the client's result would be saved in {saved}, inside the run's directory, which leaves "
                               "the node with each request's generated text: TMPDIR, outside every run's directory")
        to = [saved / u.name if load.sessions else saved for u, *_ in jobs]
        argvs = [bench_argv(model, checkpoint, port, load, c, to[k], name, f, prompts, seed, sampled, seeded, lifetime)
                 for k, (_, name, c, f) in enumerate(jobs)]

        def client(k: int) -> int:
            jobs[k][0].mkdir(parents=True, exist_ok=True)
            to[k].mkdir(parents=True, exist_ok=True)
            with open(jobs[k][0] / "bench.log", "wb") as log:
                return run(argvs[k], stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **CLIENT_ENV}).returncode

        t, t0 = clock(), now()
        with GpuBusy(lib, SHARED.CUDA_VISIBLE_DEVICES) as busy:
            if len(jobs) == 1:
                rcs = [client(0)]
            else:
                with ThreadPoolExecutor(len(jobs)) as pool:
                    rcs = list(pool.map(client, range(len(jobs))))
        wall = clock() - t
        results = []
        for (u, *_), f, rc in zip(jobs, to, rcs):
            if rc or not (f / RESULT).exists():
                raise RuntimeError(f"vllm bench serve exited {rc}{'' if rc else ' and saved no result'}: {u / 'bench.log'}")
            results.append({k: v for k, v in json.loads((f / RESULT).read_text()).items() if k not in UNKEPT})
    for (u, *_), r in zip(jobs, results):
        (u / RESULT).write_text(json.dumps(r))
    r = results[0]
    if load.sessions:
        r = merged(results)
        (d / RESULT).write_text(json.dumps(r))
    failed = int(r.get("failed") or 0)
    meta = {"tag": tag, "client": BENCH, "concurrency": rung, "rate": None, "requests": r["completed"] + failed,
            "unfinished": failed, "unserved": failed, "wall_s": round(wall, 3), "t0_unix": t0, "argv": argvs[0][1:],
            "gpu": busy.result()}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def bench_metrics(r: dict) -> dict:
    """One pass's metrics from the result vLLM's client saved, under `load_metrics`'s names: the client's own figures
    (its goodput, throughputs, token counts, medians and P99s), none recomputed here; the prompt tokens a second are
    its total throughput less its output throughput. A sessions load's result is its users' as one (`merged`), and
    has beside its rates the sum of its users' own (``users_output_tok_per_s``) and their seconds summed (``users_s``)."""
    sent = r["completed"] + int(r.get("failed") or 0)
    good = r.get("request_goodput")
    met = r["met"] if "met" in r else round(good * r["duration"]) if good is not None else None

    def s(key: str) -> float | None:
        return r[key] / 1e3 if r.get(key) is not None else None

    return {"sent": sent, "ok": r["completed"], "output_tokens": r["total_output_tokens"],
            "prompt_tokens": r["total_input_tokens"], "span_s": r["duration"],
            "output_tok_per_s": r["output_throughput"], "total_tok_per_s": r["total_token_throughput"],
            "prompt_tok_per_s": r["total_token_throughput"] - r["output_throughput"],
            "req_per_s": r["request_throughput"], "goodput_req_per_s": good, "met": met,
            "slo_attained": met / sent if met is not None and sent else None,
            "ttft_p50_s": s("median_ttft_ms"), "ttft_p99_s": s("p99_ttft_ms"), "itl_p50_s": s("median_itl_ms"),
            "itl_p99_s": s("p99_itl_ms"), "tpot_p50_s": s("median_tpot_ms"), "tpot_p99_s": s("p99_tpot_ms"),
            **({"tpot_p90_s": s("p90_tpot_ms")} if "p90_tpot_ms" in r else {}),
            **({"users_output_tok_per_s": r["users_output_throughput"], "users_s": r["users_duration"]}
               if "users_duration" in r else {})}


def await_commits(d: Path, n: int, wait_s: float, clock: Callable[[], float] = time.monotonic,
                  sleep: Callable[[float], None] = time.sleep) -> dict:
    """Hold the server until the boundary commitment's sink lines in ``d`` (``boundary.PID.jsonl``, one a class, committed
    or failed) account for ``n`` spans, the requests its loads finished, or ``wait_s`` passes: each window commits on a
    thread of the engine core's own, and the server's stop leaves the engine core seconds. ``failed`` counts the spans
    of the classes whose commit failed, and ``tables`` those of the classes committed with their history tables, as
    their lines say: what the commitment did, not what the arm asked of it."""
    t0 = clock()

    def spans(lines: list[dict]) -> int:
        return sum(int(x.get("spans", 0)) for x in lines)

    while True:
        lines = [json.loads(x) for f in sorted(d.glob("boundary.*.jsonl")) for x in f.read_text().splitlines() if x.strip()]
        got = spans(lines)
        if got >= n or clock() - t0 >= wait_s:
            bad = [x for x in lines if x.get("error")]
            return {"spans": got, "finished": n, "waited_s": round(clock() - t0, 1), "complete": got >= n,
                    "failed": spans(bad), "tables": spans([x for x in lines if x.get("tables") and not x.get("error")])}
        sleep(1.0)


def open_files() -> int:
    """Raise this process's open-file limit to its hard limit and return it: a saturated load holds a connection open for
    each request in the server's queue, past the usual soft limit of 1,024."""
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft != hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    return resource.getrlimit(resource.RLIMIT_NOFILE)[0]


def versions() -> dict:
    from importlib import metadata

    return {dist: next((d.version for d in metadata.distributions(name=dist)), None)
            for dist in ("vllm", "torch", "aiohttp", "transformers")}


def source_commit() -> str | None:
    """This tree's commit: the research run's source, else the checkout's HEAD."""
    if SHARED.RESEARCH_SOURCE_SHA:
        return SHARED.RESEARCH_SOURCE_SHA
    p = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def file_digest(p: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


#: the ranks that copy their KV pool to host memory at an edge (`boundary.KV_COPY`, which this module doesn't import)
KV_COPY = ("all", "first", "none")


def tp_size(model: Model) -> int:
    args = list(model.args)
    return int(args[args.index("--tensor-parallel-size") + 1]) if "--tensor-parallel-size" in args else 1


def edge_config(arm: Arm, model: Model, a: argparse.Namespace, rates: dict[str, float], tag: str | None = None) -> dict | None:
    """The plugin's edges (`servetax_plugin.edges_of`) for an arm that takes the boundary state: one each
    ``--edges-every-s`` seconds of each load, or at ``--edges-s`` after the first load's first request (its ids start
    with ``tag``: `load_tag`); the KV bytes kept at ``--save-kv``'s edges (by default the first, at fixed times; none,
    every so often), which a tensor-parallel engine keeps in host memory instead."""
    if not arm.edges:
        return None
    if bool(a.edges_s) == bool(a.edges_every_s):
        raise SystemExit(f"arm {a.arm} takes the boundary state at --edges-s S,S,... or every --edges-every-s S: one of them")
    save = a.save_kv if a.save_kv is not None else ("" if a.edges_every_s else "0")
    save_kv = [int(x) for x in save.split(",") if x.strip()]
    if save_kv and tp_size(model) > 1:
        raise SystemExit(f"{a.model_id} runs at TP {tp_size(model)}: its edges keep the KV bytes in host memory (no --save-kv)")
    when = ({"every_s": a.edges_every_s} if a.edges_every_s else
            {"edges_s": [float(x) for x in a.edges_s.split(",")],
             "edge_tag": f"-{tag or load_tag(next(iter(rates)), 1, False)}-"})
    return {**when, "save_kv": save_kv, "kv_dir": str(Path(a.kv_dir) / a.arm) if a.kv_dir else None, "kv_copy": a.kv_copy,
            "dump": arm.edge_dump, "busy": arm.edge_busy}


def run(a: argparse.Namespace) -> int:
    """Serve one arm (`serve`), what it keeps on the node in ``--private``. Where none is given and the arm keeps
    something (`keeps_private`): under vLLM's client, a cost's run, which nothing reads those files of afterwards, a
    fresh directory outside the run's, removed when the arm stops; under the corpus client, a serve whose later stages
    read them, refused before anything starts."""
    if ARMS[a.arm].certifier:
        return certified(a)
    out, held = Path(a.out).resolve(), SHARED.RESEARCH_RUN_DIR

    def outside(private: Path) -> Path:
        for public in (out, *([Path(held).resolve()] if held else [])):
            if private.is_relative_to(public):
                raise SystemExit(f"the private directory {private} is inside {public}, which leaves the node: --private "
                                 "DIR, outside every run's directory")
        return private

    if a.private:
        private = outside(Path(a.private).resolve())
        private.mkdir(mode=0o700, parents=True, exist_ok=True)
        return serve(a, private)
    if not keeps_private(ARMS[a.arm]):
        return serve(a, None)
    if a.client != BENCH:
        raise SystemExit(f"arm {a.arm} keeps on the node what holds values (a boundary commitment's classes hold its "
                         "requests' tokens and its salts), and a serve's later stages read it: --private DIR, outside "
                         "every run's directory (run.sh: PRIVATE)")
    with tempfile.TemporaryDirectory(prefix="serve-tax-private.") as d:
        return serve(a, outside(Path(d).resolve()))


def serve(a: argparse.Namespace, private: Path | None) -> int:
    termed: list[int] = []

    def on_term(s: int, _frame: object) -> None:
        termed.append(s)
        raise SystemExit(128 + s)

    signal.signal(signal.SIGTERM, on_term)
    nofile = open_files()
    stop_at = parse_stop_at(a.stop_at)
    arm, model = ARMS[a.arm], MODELS[a.model_id]
    checkpoint = Path(a.checkpoint) if a.checkpoint else model.checkpoint()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bench = a.client == BENCH
    given = published_context(a.arm, a.model_id, a.max_model_len) if arm.published else a.max_model_len
    if bench:
        if arm.tick or arm.sampling != "greedy" or arm.logprobs:
            raise SystemExit(f"arm {a.arm} needs the corpus client (the certifier's stand-in feed, or its sampling): "
                             f"--client {CORPUS}")
        if not a.loads:
            raise SystemExit(f"--client {BENCH} serves --loads NAME[=RUNG+RUNG...],... ({', '.join(sorted(LOADS))})")
        loads = parse_loads(a.loads)
        datasets = parse_datasets(a.dataset, loads)
        sizes = parse_pass_prompts(a.pass_prompts, loads)
        loads, lines = passes_of(loads, datasets, sizes)
        traces = {n: write_trace(out / f"{n}.trace.jsonl", x) for n, x in lines.items()}
        sent = client_files(datasets, traces)
        context = context_of(loads, given)
        passes = {pass_name(n, c): (n, c) for n, rungs in loads.items() for c in rungs}
        rates, plan = {}, []
    else:
        if not a.rates or not a.c4:
            raise SystemExit(f"--client {CORPUS} serves --rates NAME=REQ_PER_S,... of the --c4 shard's plan")
        passes, context, traces, lines, sent, sizes = {}, given, {}, {}, {}, {}
        if a.sampled:
            raise SystemExit(f"--sampled is a pass of vLLM's client (--client {BENCH}): the corpus client's sampling is "
                             "its arm's")
        rates = parse_rates(a.rates)
        t = time.monotonic()
        plan = plan_of(Path(a.plan) if a.plan else out / "plan.jsonl", Path(a.c4), checkpoint,
                       plan_need(rates, a.span_s, a.requests), a.docs, a.seed)
        print(f"[serve-tax] plan: {len(plan)} requests in {time.monotonic() - t:.1f} s", flush=True)
    tick = tick_args(a.tick_from, a.tick_ms) if arm.tick else None
    srv = out / "server"
    srv.mkdir(exist_ok=True)
    site = plugin_site(out / ".site", arm)
    port = corpus().free_port()
    front = front_end(arm, getattr(a, "published_frontend", None))
    argv = server_argv(arm, model, checkpoint, port, a.num_gpu_blocks, context, front,
                       published_args(getattr(a, "published_args", None)))
    BODY_EXTRA.clear()
    BODY_EXTRA.update({"logprobs": 0} if arm.logprobs else {})
    edges = edge_config(arm, model, a, rates, load_tag(next(iter(passes)), 1, True) if bench else None)
    try:
        env = server_env(arm, srv, site, tick, a.window_s, a.kv_window_s, edges, a.boundary_rows, model.check, private,
                         front)
        check_plugins(env, arm)
    except SystemExit:
        shutil.rmtree(out / ".site", ignore_errors=True)
        raise
    lib, no_nvml = nvml()
    if no_nvml:
        print(f"[serve-tax] no busy GPU-seconds: {no_nvml}", flush=True)
    meta = {"schema": SCHEMA, "arm": a.arm, "what": arm.what, "parent": arm.parent, "model_id": a.model_id,
            "model": f"{model.repo}@{model.revision}", "checkpoint": str(checkpoint), "server_argv": argv[1:],
            "front_end": front,
            "plugin": list(arm.plugin), "env": arm.env, "settings": value_settings(env), "tick": tick, "window_s": a.window_s, "kv_window_s": a.kv_window_s, "rates": rates, "reps": a.reps, "plan": a.plan,
            "lifetime": a.lifetime, "left_out": a.leave_out,
            "general": entry_points(arm),
            "boundary_rows": ({"window_s": a.window_s, "check": model.check, **(a.boundary_rows or arm.rows)}
                              if "boundary_rows" in arm.plugin else None),
            "client": a.client, "passes": {k: {"load": n, "concurrency": c} for k, (n, c) in passes.items()},
            "pass_prompts": sizes,
            "datasets": {n: {"file": f, "sha512": LOADS[n].file.sha512, "repo": LOADS[n].file.repo,
                             "revision": LOADS[n].file.revision}
                         for n, f in (datasets if bench else {}).items() if LOADS[n].file},
            "planned": {n: {**planned(lines[n]), "sha256": file_digest(Path(f)),
                            **({"users": max(loads[n])} if LOADS[n].sessions else {}),
                            **({"sampled": SAMPLED} if a.sampled else {}),
                            **({"table_sha256": file_digest(Path(datasets[n]))} if n in datasets and LOADS[n].plan else {})}
                        for n, f in traces.items()},
            "commit": source_commit(),
            "plan_sha256": file_digest(Path(a.plan)) if a.plan and Path(a.plan).is_file() else None,
            "span_s": a.span_s, "c4": a.c4, "seed": a.seed, "sampled": SAMPLED if a.sampled else None,
            "seeded": a.sampled and sends_seed(arm), "end_of_text_ids": end_of_text_ids(checkpoint),
            "sampling": arm.sampling, "logprobs": arm.logprobs, "edges": edges,
            "gpu_busy": lib is not None, "no_gpu_busy": no_nvml, "versions": versions(), "host": os.uname().nodename,
            "run_id": SHARED.RESEARCH_RUN_ID, "gpus": SHARED.CUDA_VISIBLE_DEVICES, "open_files": nofile, "loads": []}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    t = time.monotonic()
    with open(srv / "server.log", "wb") as log:
        server = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    rc = 0
    try:
        wait_ready(server, port, a.startup_s)
        meta["startup_s"] = round(time.monotonic() - t, 1)
        print(f"[serve-tax] {a.arm}: up in {meta['startup_s']} s", flush=True)
        finished = 0

        def send(name: str, tag: str, d: Path, busy: object | None, warm: bool = False) -> dict:
            if not bench:
                return drive_load(port, model.served, plan, min(rates.values()) if warm else rates[name],
                                  a.warm_s if warm else a.span_s, tag, d, a.drain_s, tick,
                                  None if warm else a.requests, arm.sampling, busy)
            n, c = passes[name]
            return drive_bench(port, model, checkpoint, LOADS[n], c, tag, d, sent.get(n), a.seed, busy,
                               pass_prompts(LOADS[n], c, lines.get(n), a.warm_s if warm else None, sizes.get(n)),
                               a.sampled, a.sampled and sends_seed(arm), a.lifetime)

        if a.warm_s > 0:
            w = send(next(iter(passes or rates)), "warm", out / "warm", None, warm=True)
            finished += w["requests"] - w["unfinished"]
            print(f"[serve-tax] warm-up: {w['requests']} requests, unfinished {w['unfinished']}", flush=True)
        for rep, name in ((r, n) for r in range(1, a.reps + 1) for n in passes or rates):
            need = stop_before(meta["loads"], name, None if stop_at is None else stop_at - time.time())
            if need is not None:
                meta["stopped"] = {"before": name, "rep": rep, "need_s": round(need, 1), "stop_at": a.stop_at}
                print(f"[serve-tax] {a.arm}: stopped before {name} rep {rep}: the pass and the server's stop need "
                      f"{need:.0f} s, more than is left to --stop-at {a.stop_at}", flush=True)
                break
            ok = reset_prefix_cache(port)
            d = out / name / f"rep-{rep}"
            before = scrape(port)
            m = send(name, load_tag(name, rep, bench), d, lib)
            after = scrape(port)
            m |= {"load": name, "rep": rep, "prefix_cache_reset": ok,
                  "schedule": {k: after[k] - before[k] for k in after}}
            (d / "meta.json").write_text(json.dumps(m, indent=1))
            meta["loads"].append({k: m[k] for k in ("load", "rep", "rate", "requests", "unfinished", "wall_s", "t0_unix")})
            (out / "meta.json").write_text(json.dumps(meta, indent=1))
            at = at_rung(m["concurrency"]) if bench else f"at {m['rate']:g}/s"
            print(f"[serve-tax] {name} rep {rep}: {m['requests']} requests {at}, wall {m['wall_s']} s, "
                  f"unfinished {m['unfinished']}, unserved {m['unserved']}", flush=True)
            finished += m["requests"] - m["unfinished"]
        if "boundary_rows" in arm.plugin:
            meta["boundary_commits"] = await_commits(srv, finished, a.commit_wait_s)
            print(f"[serve-tax] boundary commits: {meta['boundary_commits']}", flush=True)
    except Exception as e:  # noqa: BLE001 -- the record says why the arm stopped
        meta["error"] = f"{type(e).__name__}: {e}"
        print(f"[serve-tax] {a.arm}: FAILED {meta['error']}", flush=True)
        rc = 1
    finally:
        meta["server_stop"] = stop(server, TERM_GRACE_S if termed else 120.0)
        meta["server_rc"] = server.returncode
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        shutil.rmtree(out / ".site", ignore_errors=True)
    return rc


# ------------------------------------------------------------------------------ an arm behind the network certifier

#: the two commands an arm behind the network certifier (`Arm.certifier`) is served by (``--side``): the server's on
#: the node, and the client's where the certifier runs, a host of the auditor's that reaches the node's id front
SERVER, CLIENT = "server", "client"
#: the loopback port the id front listens on, on the node, and where the client's side reaches it (a forward's end)
FRONT_PORT = 18746
#: the network certifier's schedule such an arm is served under, by `warden.WardenParams`'s fields; its lanes and its
#: queue are by default the highest rung (`certifier_cells`). One cell, so one id, leaves a connection each slot of a
#: bucket, the first ``delay`` buckets after the connection's admission; a request is at most ``ingress_cells`` cells,
#: read within ``ingress_buckets`` buckets, and an answer at most ``max_tokens_cap`` ids
CELLS = {"cell_bytes": 4096, "slots": 1, "bucket_ms": 250, "delay": 60, "window": 600, "ingress_cells": 4,
         "ingress_buckets": 16, "egress_cells": 4096, "max_tokens_cap": 2048}
CERTIFIER = ROOT / "benchmarks/network_traces/warden_serve.py"
SHIM = HERE / "cell_shim.py"
#: the server's side's record in its directory, which holds no meta.json: no table and no cost reads it as an arm
SERVER_RECORD = "server.json"
#: how long each pass waits before its first request, the engine holding none: past `servetax_plugin.IDLE_RESET_S`,
#: so that the engine empties its prefix cache for the pass
IDLE_S = 6.0
#: how long after its last window the network certifier has to sign its records and end
CERTIFIER_END_S = 600.0


def certified_loads(arm: str, text: str | None) -> dict[str, tuple[int, ...]]:
    """``--loads`` for an arm behind the network certifier: loads of a pinned dataset file's own prompts alone
    (`certifiable`). The shim to the certifier (`cell_shim`) carries completions of a text and nothing else, and
    vLLM's client fits a synthetic dataset's prompts to the server's ``/tokenize`` first, so such a load's prompts
    would not be the ones stock serves; a sessions load's are token ids, a client a user."""
    if not text:
        raise SystemExit(f"arm {arm} is sent by vLLM's client: --loads NAME[=RUNG+RUNG...],...")
    loads = parse_loads(text)
    bad = [n for n in loads if not certifiable(LOADS[n])]
    if bad:
        raise SystemExit(f"arm {arm} serves the loads of a pinned dataset file's own prompts "
                         f"({', '.join(n for n, x in LOADS.items() if certifiable(x))}), not {', '.join(bad)}: the shim "
                         "to the network certifier carries completions of a text alone, and vLLM's client fits a "
                         "synthetic dataset's prompts to the server's /tokenize")
    return loads


def certifiable(load: Load) -> bool:
    """Whether an arm behind the network certifier can be sent ``load`` (`certified_loads`)."""
    return load.file is not None and not load.sessions and not load.plan


def certifier_cells(text: str | None, loads: dict[str, tuple[int, ...]]) -> dict[str, int]:
    """The network certifier's schedule for ``loads``: `CELLS` with ``--cells NAME=N,...`` over it, its lanes and its
    queue by default the loads' highest rung. Refused where it is no schedule of the certifier's in the id form
    (`warden.WardenParams`), or where its queue is under a rung: vLLM's client opens a rung's connections at once,
    each waits in the queue for the next bucket, and the certifier refuses a connection past it and withholds the
    window. Both sides of an arm take the same one: the server's side reads its bucket and its cell from it."""
    from verity.protocols.certifier.verifier.warden import WardenParams

    top = max(c for rungs in loads.values() for c in rungs)
    cells = {**CELLS, "lanes": top, "queue": top}
    for part in filter(None, (p.strip() for p in (text or "").split(","))):
        name, _, n = part.partition("=")
        if name not in cells or not n.isdigit():
            raise SystemExit(f"--cells: {part!r}: NAME=N, NAME one of {', '.join(cells)}")
        cells[name] = int(n)
    try:
        WardenParams(**cells, egress="id")
    except ValueError as e:
        raise SystemExit(f"--cells: {e}") from None
    if cells["queue"] < top:
        raise SystemExit(f"--cells: a queue of {cells['queue']} under the rung {top}: vLLM's client opens a rung's "
                         "connections at once, and the network certifier refuses a connection past its queue")
    return cells


def cell_payload(cells: dict[str, int]) -> int:
    """A cell's payload under a schedule, which is an id event's bytes: the id front's ``--id-payload``."""
    from verity.protocols.certifier.verifier.warden import WardenParams

    return WardenParams(**cells, egress="id").payload


def ceiling(cells: dict[str, int]) -> float:
    """The most ids a second a schedule lets leave, whatever the engine does: each lane's connection one a slot."""
    return cells["lanes"] * cells["slots"] * 1000 / cells["bucket_ms"]


def tree_env() -> dict[str, str]:
    """The job's environment with this tree and the adapter on Python's path: what the id front, the network
    certifier and the shim import."""
    return dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "adapters/vllm"), *SHARED.PYTHONPATH]))


def front_argv(port: int, front: int, payload: int, log: Path) -> list[str]:
    """The id front's command (the adapter's ``verity-vllm ids-front``): on loopback port ``front`` before the server
    on ``port``, it sends the server each id request as a completion under its release headers and answers with the
    stream's ids, an id event of ``payload`` bytes each, and relays every other connection (the watermark's, a
    scrape) as it is. ``log`` takes its record of each id request: lengths and times."""
    return [sys.executable, "-m", "verity_vllm.pipeline.cli", "ids-front", "--upstream", f"127.0.0.1:{port}",
            "--listen", f"127.0.0.1:{front}", "--id-payload", str(payload), "--log", str(log)]


def certifier_argv(cells: dict[str, int], out: Path, front: int, key: Path, session: str, windows: int) -> list[str]:
    """The network certifier's command (`warden_serve.py serve`): before the id front on loopback port ``front``, on a
    port of its own choice, under the schedule ``cells`` in the id form for ``windows`` windows, with the watermark
    stream the certified tick admits by; its records in ``out``, its key in ``key``."""
    return [sys.executable, str(CERTIFIER), "serve", "--out", str(out), "--listen", "127.0.0.1:0", "--upstream",
            f"127.0.0.1:{front}", "--key", str(key), "--session", session, "--windows", str(windows), "--watermark",
            "--egress", "id", *(x for k, v in cells.items() for x in (f"--{k.replace('_', '-')}", str(v)))]


def shim_argv(deployment: Path, checkpoint: Path, log: Path) -> list[str]:
    """The shim's command (`cell_shim`): vLLM's client's completions to the network certifier ``deployment`` names,
    each prompt encoded by the checkpoint's tokenizer; its record of each connection in ``log``."""
    return [sys.executable, str(SHIM), "--deployment", str(deployment), "--tokenizer", str(checkpoint), "--log",
            str(log), "--listen", "127.0.0.1:0"]


def announced(proc: subprocess.Popen, what: str, log: Path) -> dict:
    """The JSON line a process of ours says where it listens with, the first of its output."""
    try:
        return json.loads(proc.stdout.readline())
    except ValueError:
        raise RuntimeError(f"{what} ended before it listened ({log})") from None


def runs_ended(feed: Path) -> int:
    """The network certifier's runs whose watermark the tick's feed (`tick_plugin.TickFeed`) has seen end."""
    if not feed.exists():
        return 0
    return sum(1 for x in feed.read_text().splitlines() if '"end"' in x and json.loads(x).get("end"))


def hold_ends(server: int | None, front: int | None, ended: int, left_s: float | None) -> tuple[str, bool] | None:
    """Whether the server's side stops holding its engine, with why and whether that is its work done: its server or
    its id front exited (their return codes), which is not; the network certifier's run ended; or `STOP_S` or less is
    left to ``--stop-at``. None while it holds."""
    if server is not None:
        return f"the server exited with {server} (server/server.log)", False
    if front is not None:
        return f"the id front exited with {front} (server/front.log)", False
    if ended:
        return "the network certifier's run ended", True
    if left_s is not None and left_s <= STOP_S:
        return "--stop-at", True
    return None


def certifier_record(check: dict | None, lines: Sequence[dict]) -> dict:
    """What the network certifier's offline check (`warden_serve.py check`: its check.json, None where it wrote
    none) found of a run, with the shim's record of each connection (``lines``, `cell_shim.HttpShim`): whether the
    run is accepted and the first of its problems; the completions the shim sent and those it refused unsent; the
    connections that ended, were refused, failed or were cut; and, of those that ended, the ones whose answer is not
    its request's ``max_tokens`` ids (``short``), which `servetax_plugin.to_length` holds at none, so that the arm's
    answers are as long as stock's."""
    sent = [x for x in lines if "unsent" not in x]
    ended = [x for x in sent if x["outcome"] == "end"]
    summary = (check or {}).get("summary") or {}
    return {"accepted": bool(check and check["accepted"]), "problems": (check or {}).get("problems", [])[:5],
            "connections": len(sent), "unsent": len(lines) - len(sent), "ended": len(ended),
            "refused": sum(x["outcome"] == "refused" for x in sent),
            "failed": sum(x["outcome"] not in ("end", "refused") for x in sent),
            "short": sum(x["response"]["completion_tokens"] != x["max_tokens"] for x in ended),
            "certified_windows": summary.get("certified_windows"), "withheld_windows": summary.get("withheld_windows"),
            "cells_in_their_bucket": summary.get("cells_in_their_bucket"),
            "tick_lateness_ms": summary.get("tick_lateness_ms")}


def certified(a: argparse.Namespace) -> int:
    """Serve one side of an arm behind the network certifier (`Arm.certifier`): ``--side server`` on the node
    (`certified_server`) or ``--side client`` where the certifier runs (`certified_client`), both given the same
    ``--loads`` and ``--cells``. Such an arm has no other client than vLLM's behind the certifier's shim."""
    if a.client != BENCH:
        raise SystemExit(f"arm {a.arm} is sent by vLLM's client, through the network certifier: --client {BENCH}")
    if a.side not in (SERVER, CLIENT):
        raise SystemExit(f"arm {a.arm} is served by two commands: --side {SERVER} on the node, which starts the engine "
                         f"and the id front before it, and --side {CLIENT} where the network certifier runs, a host "
                         "that reaches the id front and is never the node")
    if a.sampled:
        raise SystemExit(f"arm {a.arm} is served greedy: the network certifier's id request states temperature 0 and "
                         "its own seed, whatever the client's completion asks (--sampled is another arm's)")
    return (certified_server if a.side == SERVER else certified_client)(a)


def certified_server(a: argparse.Namespace, popen: Callable[..., subprocess.Popen] = subprocess.Popen,
                     sleep: Callable[[float], None] = time.sleep, now: Callable[[], float] = time.time,
                     clock: Callable[[], float] = time.monotonic) -> int:
    """The server's side of an arm behind the network certifier, on the node: the arm's engine on the certified tick
    (`server_argv`, `server_env`; the tick's bucket is the schedule's) and, once it answers, the id front before it on
    loopback port ``--front`` (`front_argv`), both held until the certifier's run has ended (`runs_ended`) or
    ``--stop-at`` (`hold_ends`). It sends nothing: the requests come from the client's side, through the certifier.
    DIR/server.json is its record: its commands, the tick, the schedule, why it stopped, the engine's counters then
    and each time the engine emptied its prefix cache (`servetax_plugin.IdleReset`)."""
    termed: list[int] = []

    def on_term(s: int, _frame: object) -> None:
        termed.append(s)
        raise SystemExit(128 + s)

    arm, model = ARMS[a.arm], MODELS[a.model_id]
    checkpoint = Path(a.checkpoint) if a.checkpoint else model.checkpoint()
    loads = certified_loads(a.arm, a.loads)
    cells = certifier_cells(a.cells, loads)
    stop_at = parse_stop_at(a.stop_at)
    tick = tick_args(a.tick_from, a.tick_ms, bucket_ms=cells["bucket_ms"])
    signal.signal(signal.SIGTERM, on_term)
    nofile = open_files()
    out = Path(a.out)
    srv = out / "server"
    srv.mkdir(parents=True, exist_ok=True)
    site = plugin_site(out / ".site", arm)
    port = corpus().free_port()
    argv = server_argv(arm, model, checkpoint, port, a.num_gpu_blocks, context_of(loads, a.max_model_len))
    front = front_argv(port, a.front, cell_payload(cells), srv / "front.jsonl")
    try:
        env = server_env(arm, srv, site, tick, a.window_s, a.kv_window_s)
        check_plugins(env, arm)
    except SystemExit:
        shutil.rmtree(out / ".site", ignore_errors=True)
        raise
    rec = {"schema": SCHEMA, "arm": a.arm, "what": arm.what, "side": SERVER, "model_id": a.model_id,
           "model": f"{model.repo}@{model.revision}", "checkpoint": str(checkpoint), "server_argv": argv[1:],
           "front_argv": front[1:], "plugin": list(arm.plugin), "general": entry_points(arm),
           "settings": value_settings(env), "tick": tick, "lifetime": a.lifetime,
           "cells": cells, "stop_at": a.stop_at, "commit": source_commit(), "versions": versions(),
           "host": os.uname().nodename, "run_id": SHARED.RESEARCH_RUN_ID, "gpus": SHARED.CUDA_VISIBLE_DEVICES,
           "open_files": nofile}
    (out / SERVER_RECORD).write_text(json.dumps(rec, indent=1))
    t = clock()
    with open(srv / "server.log", "wb") as log:
        server = popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    fronted, rc = None, 0
    try:
        wait_ready(server, port, a.startup_s)
        rec["startup_s"] = round(clock() - t, 1)
        with open(srv / "front.log", "wb") as log:
            fronted = popen(front, stdout=subprocess.PIPE, stderr=log, env=tree_env(), text=True)
        announced(fronted, "the id front", srv / "front.log")
        (out / SERVER_RECORD).write_text(json.dumps(rec, indent=1))
        print(f"[serve-tax] {a.arm}: the engine is up in {rec['startup_s']} s and the id front listens on loopback "
              f"port {a.front}: held for the network certifier's run", flush=True)
        while True:
            why = hold_ends(server.poll(), fronted.poll(), runs_ended(srv / "tick-feed.jsonl"),
                            None if stop_at is None else stop_at - now())
            if why is not None:
                break
            sleep(2.0)
        rec["stopped"] = why[0]
        print(f"[serve-tax] {a.arm}: stopping: {why[0]}", flush=True)
        if not why[1]:
            raise RuntimeError(why[0])
        rec["schedule"] = scrape(port)
    except Exception as e:  # noqa: BLE001 -- the record says why the side stopped
        rec["error"] = f"{type(e).__name__}: {e}"
        print(f"[serve-tax] {a.arm}: FAILED {rec['error']}", flush=True)
        rc = 1
    finally:
        if fronted is not None and fronted.poll() is None:
            fronted.terminate()
            fronted.wait()
        rec["server_stop"] = stop(server, TERM_GRACE_S if termed else 120.0)
        rec["server_rc"] = server.returncode
        rec["prefix_resets"] = [json.loads(x) for f in sorted(srv.glob("prefix_resets.*.jsonl"))
                                for x in f.read_text().splitlines() if x.strip()]
        (out / SERVER_RECORD).write_text(json.dumps(rec, indent=1))
        shutil.rmtree(out / ".site", ignore_errors=True)
    return rc


def served_models(front: int) -> list[str]:
    """The models the server behind the id front on loopback port ``front`` serves, asked through the front's relay."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{front}/v1/models", timeout=30) as r:
            return [x["id"] for x in json.loads(r.read())["data"]]
    except OSError as e:
        raise SystemExit(f"no server answers behind the id front on loopback port {front} ({e!r}): the arm's --side "
                         f"{SERVER} holds one on the node, and this host reaches its front through a forward") from None


def certified_client(a: argparse.Namespace, run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                     popen: Callable[..., subprocess.Popen] = subprocess.Popen,
                     sleep: Callable[[float], None] = time.sleep, now: Callable[[], float] = time.time,
                     clock: Callable[[], float] = time.monotonic) -> int:
    """The client's side of an arm behind the network certifier, where the certifier runs: a host of the auditor's
    that reaches the node's id front on loopback port ``--front`` (a forward's end), never the node. It starts the
    network certifier before that port for ``--certifier-windows`` windows (`certifier_argv`; its key made in a
    directory outside the run's and removed with it) and the shim to it (`shim_argv`), and sends each pass by vLLM's
    client at the shim (`drive_bench`) as `serve` sends an arm's: a warm-up, then each pass REPS times, the engine's
    counters read through the front around each. Each pass first waits `IDLE_S`, the engine holding nothing, so that
    the engine empties its prefix cache for it (`servetax_plugin.IdleReset`). The certifier runs the windows fixed
    when it starts and cuts every connection at their end, so no pass starts that can't be expected to end before
    (`stop_before`), and the side then waits for the certifier's end. Then the certifier's offline check reads the
    run with the shim's record of each connection (DIR/certifier/client.json), and DIR/meta.json, an arm's, holds
    what it found (`certifier_record`). It exits 0 where every pass was sent, the run is accepted and no answer was
    short of its length."""
    import secrets

    import cell_shim

    arm, model = ARMS[a.arm], MODELS[a.model_id]
    checkpoint = Path(a.checkpoint) if a.checkpoint else model.checkpoint()
    loads = certified_loads(a.arm, a.loads)
    if not a.certifier_windows or a.certifier_windows < 1:
        raise SystemExit("--certifier-windows N: the windows the network certifier runs, fixed when it starts: enough "
                         "for the warm-up and every pass, each as long as its longest answer's cells take")
    cells = certifier_cells(a.cells, loads)
    datasets = parse_datasets(a.dataset, loads)
    sizes = parse_pass_prompts(a.pass_prompts, loads)
    passes = {pass_name(n, c): (n, c) for n, rungs in loads.items() for c in rungs}
    served = served_models(a.front)
    if model.served not in served:
        raise SystemExit(f"the server behind the id front on loopback port {a.front} serves {', '.join(served)}, not "
                         f"{model.served}")
    out, run_dir = Path(a.out), SHARED.RESEARCH_RUN_DIR
    cert = out / "certifier"
    cert.mkdir(parents=True, exist_ok=True)
    env = tree_env()
    meta = {"schema": SCHEMA, "arm": a.arm, "what": arm.what, "parent": arm.parent, "side": CLIENT,
            "model_id": a.model_id, "model": f"{model.repo}@{model.revision}", "checkpoint": str(checkpoint),
            "plugin": list(arm.plugin), "client": a.client, "reps": a.reps, "lifetime": a.lifetime,
            "left_out": a.leave_out, "seed": a.seed, "sampled": None, "seeded": False,
            "passes": {k: {"load": n, "concurrency": c} for k, (n, c) in passes.items()}, "pass_prompts": sizes,
            "datasets": {n: {"file": f, "sha512": LOADS[n].file.sha512, "repo": LOADS[n].file.repo,
                             "revision": LOADS[n].file.revision} for n, f in datasets.items()},
            "certifier": {"cells": cells, "windows": a.certifier_windows, "ceiling_tok_per_s": ceiling(cells)},
            "commit": source_commit(), "gpu_busy": False, "no_gpu_busy": "the engine's GPUs are another host's",
            "versions": versions(), "host": os.uname().nodename, "run_id": SHARED.RESEARCH_RUN_ID, "loads": []}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    procs: list[subprocess.Popen] = []
    rc = 0
    with tempfile.TemporaryDirectory(prefix="serve-tax-certifier-key.") as tmp:
        key = Path(tmp).resolve() / "key"
        if any(key.is_relative_to(p) for p in (out.resolve(), *([Path(run_dir).resolve()] if run_dir else []))):
            raise SystemExit(f"the network certifier's key would be made in {key}, inside the run's directory, which "
                             "leaves the host: TMPDIR, outside every run's directory")
        try:
            made = run([sys.executable, str(CERTIFIER), "keys", "--dir", str(key)], stdout=subprocess.DEVNULL,
                       env=env).returncode
            if made:
                raise RuntimeError(f"the network certifier's key was not made (warden_serve.py keys exited {made})")
            with open(out / "certifier.log", "wb") as log:
                certifier = popen(certifier_argv(cells, cert, a.front, key, secrets.token_hex(32), a.certifier_windows),
                                  stdout=subprocess.PIPE, stderr=log, env=env, text=True)
            procs.append(certifier)
            announced(certifier, "the network certifier", out / "certifier.log")
            origin = json.loads((cert / "deployment.json").read_text())["origin_unix"]
            end = origin + a.certifier_windows * cells["window"] * cells["bucket_ms"] / 1000
            with open(out / "shim.log", "wb") as log:
                shim = popen(shim_argv(cert / "deployment.json", checkpoint, out / "shim.jsonl"),
                             stdout=subprocess.PIPE, stderr=log, env=env, text=True)
            procs.append(shim)
            port = announced(shim, "the shim", out / "shim.log")["listening"][1]
            meta["certifier"] |= {"origin_unix": origin, "end_unix": end}
            print(f"[serve-tax] {a.arm}: the network certifier runs {a.certifier_windows} windows, "
                  f"{end - origin:.0f} s, before the id front on loopback port {a.front}", flush=True)
            sleep(max(0.0, origin - now()))

            def send(name: str, tag: str, d: Path, warm: bool = False) -> dict:
                n, c = passes[name]
                sleep(IDLE_S)
                before = scrape(a.front)
                m = drive_bench(port, model, checkpoint, LOADS[n], c, tag, d, datasets.get(n), a.seed, None,
                                pass_prompts(LOADS[n], c, None, a.warm_s if warm else None, sizes.get(n)),
                                run=run, clock=clock, now=now)
                after = scrape(a.front)
                return m | {"schedule": {k: after[k] - before[k] for k in after}}

            if a.warm_s > 0:
                w = send(next(iter(passes)), "warm", out / "warm", warm=True)
                (out / "warm" / "meta.json").write_text(json.dumps(w, indent=1))
                print(f"[serve-tax] warm-up: {w['requests']} requests, unfinished {w['unfinished']}, wall "
                      f"{w['wall_s']} s", flush=True)
            for rep, name in ((r, n) for r in range(1, a.reps + 1) for n in passes):
                need = stop_before(meta["loads"], name, end - now() - IDLE_S)
                if need is not None:
                    meta["stopped"] = {"before": name, "rep": rep, "need_s": round(need, 1)}
                    print(f"[serve-tax] {a.arm}: stopped before {name} rep {rep}: the pass needs {need:.0f} s, more "
                          "than is left of the network certifier's windows (--certifier-windows)", flush=True)
                    rc = 1
                    break
                d = out / name / f"rep-{rep}"
                m = send(name, load_tag(name, rep, True), d) | {"load": name, "rep": rep}
                (d / "meta.json").write_text(json.dumps(m, indent=1))
                meta["loads"].append({k: m[k] for k in ("load", "rep", "rate", "requests", "unfinished", "wall_s", "t0_unix")})
                (out / "meta.json").write_text(json.dumps(meta, indent=1))
                print(f"[serve-tax] {name} rep {rep}: {m['requests']} requests {at_rung(m['concurrency'])}, wall "
                      f"{m['wall_s']} s, unfinished {m['unfinished']}, {m['schedule']['prompt_tokens_cached']:,.0f} of "
                      f"{m['schedule']['prompt_tokens']:,.0f} prompt tokens from the prefix cache", flush=True)
            left = max(0.0, end - now())
            print(f"[serve-tax] {a.arm}: waiting {left:.0f} s for the network certifier's last window", flush=True)
            try:
                meta["certifier"]["exit"] = certifier.wait(left + CERTIFIER_END_S)
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"the network certifier had not ended {CERTIFIER_END_S:.0f} s after its last "
                                   "window (certifier.log)") from None
            lines = [json.loads(x) for x in (out / "shim.jsonl").read_text().splitlines() if x.strip()]
            (cert / "client.json").write_text(json.dumps(cell_shim.client_doc(lines)))
            with open(out / "check.log", "wb") as log:
                meta["certifier"]["check_exit"] = run([sys.executable, str(CERTIFIER), "check", "--run", str(cert)],
                                                      stdout=log, stderr=subprocess.STDOUT, env=env).returncode
            check = cert / "check.json"
            found = certifier_record(json.loads(check.read_text()) if check.exists() else None, lines)
            meta["certifier"] |= found
            print(f"[serve-tax] {a.arm}: the network certifier's run is {'' if found['accepted'] else 'NOT '}accepted: "
                  f"{found['ended']} of {found['connections']} connections ended, {found['refused']} refused, "
                  f"{found['failed']} failed, {found['short']} answers short of their length", flush=True)
            if not found["accepted"] or found["short"]:
                rc = 1
        except Exception as e:  # noqa: BLE001 -- the record says why the arm stopped
            meta["error"] = f"{type(e).__name__}: {e}"
            print(f"[serve-tax] {a.arm}: FAILED {meta['error']}", flush=True)
            rc = 1
        finally:
            for proc in procs:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait()
            (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return rc


# ------------------------------------------------------------------------------------------------------- the tick's τ

def busy_cycle(recs: list[dict]) -> dict:
    """The engine's mean busy cycle over a step trace with times (`StepTrace` with a clock): from one step's output to the
    next's, between consecutive steps that scheduled tokens, where a request was in the engine (arrived, not yet finished)
    once the first's output was back, as `summary.busy_cycle` reads the corpus's traces."""
    flight: set[str] = set()
    outs: list[tuple[float, bool]] = []
    for r in recs:
        if r.get("kind") == "arrive":
            flight.add(r["request"])
        elif r.get("kind") == "step":
            flight -= set(r.get("finished") or ())
            outs.append((float(r["t"]), bool(flight)))
        elif r.get("kind") == "finish":
            flight.discard(r["request"])
    cycles = [b[0] - a[0] for a, b in zip(outs, outs[1:]) if a[1]]
    return {"cycles": len(cycles), "idle": max(0, len(outs) - 1 - len(cycles)),
            "mean_us": round(statistics.fmean(cycles) * 1e6, 1) if cycles else None}


def tick_args(source: str | None, given_ms: str | None, load: str = "saturated", bucket_ms: int | None = None) -> dict:
    """The certified tick's constants in µs: τ_b the corpus's bucket, or ``bucket_ms``, the network certifier's, where
    one releases the requests (the tick admits by the release bucket); τ_sat and τ_free the corpus's multiples of the
    mean busy cycle of a ``record`` step trace (a file, or a ``record`` arm's dir: its trace over ``load``'s first rep),
    or given as ``SAT,FREE`` ms."""
    if not TICK_PLUGIN.is_file():
        raise SystemExit(f"the tick arms need the certified tick ({TICK_PLUGIN.name}), which this tree lacks: run them "
                         "from a tree that has --admit tick")
    C = corpus()
    bucket_us = (bucket_ms or C.TICK_BUCKET_MS) * 1000
    if (source is None) == (given_ms is None):
        raise SystemExit("the tick arms take τ from one of --tick-from STEP_TRACE|RECORD_DIR and --tick-ms SAT,FREE")
    if given_ms is not None:
        sat, free = (float(x) for x in given_ms.split(","))
        return {"bucket_us": bucket_us, "sat_us": round(sat * 1000), "free_us": round(free * 1000),
                "calibrated_from": None}
    src = Path(source)
    if src.is_dir():
        traces = step_traces(src)
        meta = json.loads((src / "meta.json").read_text())
        at = [x for x in meta["loads"] if x["load"] == load and x["rep"] == 1]
        if not traces or not at:
            raise SystemExit(f"{src}: no step trace or no {load} load to calibrate the tick on")
        recs = window_trace(traces[0], at[0]["t0_unix"], at[0]["t0_unix"] + at[0]["wall_s"])
        where = {"trace": str(traces[0]), "load": load, "t0_unix": at[0]["t0_unix"], "wall_s": at[0]["wall_s"]}
    else:
        recs = trace_records(src)
        where = {"trace": str(src)}
    c = busy_cycle(recs)
    if not c["mean_us"]:
        raise SystemExit(f"{source}: no busy cycle to calibrate the tick on")
    return {"bucket_us": bucket_us, "sat_us": round(C.TICK_SAT * c["mean_us"]),
            "free_us": round(C.TICK_FREE * c["mean_us"]), "calibrated_from": {**where, **c}}


def step_traces(arm_dir: Path) -> list[Path]:
    """An arm's step traces, one per engine core, plain or gzipped (run.sh gzips them once the job's table is made)."""
    return sorted((arm_dir / "server").glob("step_trace.*.jsonl*"))


def trace_records(trace: Path) -> list[dict]:
    import gzip

    with (gzip.open if trace.suffix == ".gz" else open)(trace, "rt", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def window_trace(trace: Path, t0: float, t1: float) -> list[dict]:
    """The step trace's records with times in [t0, t1] (one load's), its header first."""
    return in_window(trace_records(trace), t0, t1)


def in_window(recs: list[dict], t0: float, t1: float) -> list[dict]:
    return recs[:1] + [r for r in recs[1:] if "t" in r and t0 <= float(r["t"]) <= t1]


# ------------------------------------------------------------------------------------------------------------ metrics

def pct(xs: list[float], q: float) -> float | None:
    """The q-quantile of xs by linear interpolation between order statistics (None when empty)."""
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def served_records(recs: list[dict]) -> list[dict]:
    """The client records of requests served to their end."""
    return [r for r in recs if r.get("status") == 200 and not r.get("error") and r.get("done_s") is not None]


def load_metrics(recs: list[dict], slo: tuple[float, float]) -> dict:
    """One load's metrics from its client records (module docstring)."""
    ok = served_records(recs)
    if not ok:
        return {"sent": len(recs), "ok": 0}
    span = max(r["send_unix"] + r["done_s"] for r in ok) - min(r["send_unix"] for r in recs)
    tokens = sum(len(r["output_ids"]) for r in ok)
    ttft = [r["first_token_s"] for r in ok if r.get("first_token_s") is not None]
    itl = [b[0] - a[0] for r in ok for a, b in zip(r["chunks"], r["chunks"][1:])]
    good = 0
    tpots = []
    for r in ok:
        n, ft = len(r["output_ids"]), r.get("first_token_s")
        tpot = (r["done_s"] - ft) / (n - 1) if ft is not None and n > 1 else 0.0
        tpots.append(tpot)
        good += ft is not None and ft <= slo[0] and tpot <= slo[1]
    return {"sent": len(recs), "ok": len(ok), "output_tokens": tokens, "span_s": span,
            "output_tok_per_s": tokens / span, "req_per_s": len(ok) / span, "goodput_req_per_s": good / span,
            "slo_attained": good / len(recs), "ttft_p50_s": pct(ttft, 0.5), "ttft_p99_s": pct(ttft, 0.99),
            "itl_p50_s": pct(itl, 0.5), "itl_p99_s": pct(itl, 0.99), "tpot_p50_s": pct(tpots, 0.5),
            "tpot_p99_s": pct(tpots, 0.99)}


def edge_cost(recs: list[dict], windows: list[tuple[float, float]], before_s: float = 10.0) -> list[dict]:
    """What a load's client saw around each of ``windows`` (an edge's start and pause, in the load's seconds; a record's
    chunk times count from its send): the requests streaming across it (sent before it, done after it) and the longest
    gap between consecutive chunks that overlaps it, each request's; and the TTFT of the requests sent from ``before_s``
    before it to its end. Given another arm's records and the same windows, the same requests at the same moments without
    the edge."""
    ok = served_records(recs)
    out = []
    for t, pause in windows:
        gaps = []
        for r in ok:
            if r["send_s"] < t < r["send_s"] + r["done_s"]:
                ts = [r["send_s"], *(r["send_s"] + c[0] for c in r["chunks"])]
                gaps.append(max((b - a for a, b in zip(ts, ts[1:]) if b >= t and a <= t + pause), default=0.0))
        ttft = [r["first_token_s"] for r in ok
                if t - before_s <= r["send_s"] <= t + pause and r.get("first_token_s") is not None]
        out.append({"t_s": t, "pause_s": pause, "in_flight": len(gaps), "gap_p50_s": pct(gaps, 0.5),
                    "gap_max_s": max(gaps, default=None), "sent": len(ttft), "ttft_p50_s": pct(ttft, 0.5),
                    "ttft_p99_s": pct(ttft, 0.99), "ttft_max_s": max(ttft, default=None)})
    return out


def edge_windows(arm_dir: Path, load: str, rep: int = 1) -> list[tuple[float, float]]:
    """A restore arm's edges as `edge_cost` windows in ``load``'s rep ``rep``'s seconds: each from its drain's end, as
    long as its pause."""
    t0 = json.loads((arm_dir / load / f"rep-{rep}" / "meta.json").read_text())["t0_unix"]
    return [(r["drained_unix"] - t0, r["pause_s"]) for r in edge_records(arm_dir)]


def schedule_metrics(s: dict | None) -> dict:
    """A load's schedule from its counters (`SCHEDULE`): its engine steps, the tokens each scheduled and the rows each
    sampled (one per request that emitted a token; under speculative decoding a request emits its accepted draft
    tokens in the same step, so these are tokens a step and no longer rows), its preemptions, the prompt's share served
    from the prefix cache and, where the engine drafted, the draft tokens accepted a draft and their share of the
    draft tokens proposed."""
    if not s or not s.get("steps"):
        return {}
    return {"steps": s["steps"], "tokens_per_step": s["step_tokens"] / s["steps"],
            "rows_per_step": s["generation_tokens"] / s["steps"], "preemptions": s["preemptions"],
            "prompt_cached_share": s["prompt_tokens_cached"] / s["prompt_tokens"] if s["prompt_tokens"] else None,
            **({"accepted_per_draft": s["accepted_tokens"] / s["drafts"],
                "draft_accepted_share": s["accepted_tokens"] / s["draft_tokens"]} if s.get("drafts") else {})}


def edge_records(arm_dir: Path) -> list[dict]:
    """An arm's edges (`servetax_plugin.edges_of`'s records with a step), in order."""
    recs = [json.loads(x) for f in sorted((arm_dir / "server").glob("edges.*.jsonl"))
            for x in f.read_text().splitlines() if x.strip()]
    return sorted((r for r in recs if "next_step" in r), key=lambda r: r["edge"])


def gpu_metrics(m: dict, output_tokens: int | None) -> dict:
    """A load's busy GPU-seconds (`GpuBusy`), and per thousand output tokens."""
    g = m.get("gpu")
    if not g:
        return {}
    return {"busy_gpu_s": g["busy_gpu_s"],
            "busy_gpu_s_per_ktok": g["busy_gpu_s"] / (output_tokens / 1e3) if output_tokens else None}


def load_edges(edges: list[dict], m: dict) -> dict:
    """The edges that fell in a load (from its first send to its wall time's end): how many, and the engine's stall at
    each, the drain to an empty batch queue plus the pause after it, at its median and longest."""
    t0 = m.get("t0_unix")
    if not edges or t0 is None:
        return {}
    stall = [r["drain_s"] + r["pause_s"] for r in edges if t0 <= r["drained_unix"] <= t0 + m["wall_s"]]
    return {"edges": len(stall), "edge_stall_p50_s": pct(stall, 0.5), "edge_stall_max_s": max(stall, default=None),
            "edge_stall_s": sum(stall)}


def edge_summary(edges: list[dict]) -> dict | None:
    """An arm's edges outside its warm-up: how many, each part's median (seconds, the hash kernels' and the copy's
    milliseconds: their longest rank's at TP > 1), the pause's p99 and longest, the bytes hashed and copied, and whether
    every rank's KV root agreed at every edge."""
    es = [r for r in edges if r.get("load") != "warm"]
    if not es:
        return None

    def med(f: Callable[[dict], float | None]) -> float | None:
        return pct([x for x in map(f, es) if x is not None], 0.5)

    pause = [r["pause_s"] for r in es]
    return {"edges": len(es), "pause_p50_s": pct(pause, 0.5), "pause_p99_s": pct(pause, 0.99), "pause_max_s": max(pause),
            "drain_p50_s": med(lambda r: r["drain_s"]), "state_p50_s": med(lambda r: r["state_s"]),
            "kv_p50_s": med(lambda r: r["kv_s"]), "hash_kernel_ms_p50": med(lambda r: r["kv"]["hash_kernel_ms"]),
            "copy_ms_p50": med(lambda r: r["kv"]["copy_ms"]), "runner_s_p50": med(lambda r: r["kv"].get("runner_s")),
            "stall_p50_s": med(lambda r: r["drain_s"] + r["pause_s"]), "kv_bytes": es[0]["kv"]["bytes"],
            "copied_bytes": es[0]["kv"].get("copied_bytes"), "roots_equal": all(r["kv"].get("roots_equal", True) for r in es)}


def end_of_text_ids(checkpoint: Path) -> list[int]:
    """The token ids a checkpoint's generation configuration ends a text at (``eos_token_id``, one or a list; its
    model configuration's where it names none), in order; none where neither file names any."""
    for name in ("generation_config.json", "config.json"):
        f = checkpoint / name
        ids = json.loads(f.read_text()).get("eos_token_id") if f.is_file() else None
        if ids is not None:
            return sorted({ids} if isinstance(ids, int) else set(ids))
    return []


def past_end(d: Path, tags: Sequence[str], ends: Sequence[int]) -> dict[str, dict]:
    """For each pass of an arm's run dir whose server kept its ids (`servetax_plugin.ServedIds`), by the pass's tag:
    the requests it kept, how many of them served an end-of-text id (``ends``) before their last token and went on,
    as every request does that is decoded to its length whatever it generates, and the output tokens after a
    request's first such id. Empty where the server kept none or ``ends`` is."""
    out: dict[str, dict] = {}
    marks = set(ends)
    for path in sorted((d / "server").glob("ids.*.jsonl*")) if marks else ():
        with (gzip.open if path.suffix == ".gz" else open)(path, "rt", encoding="utf-8") as f:
            for line in f:
                head = line.partition(",")[0]                 # a line opens with its request's id
                tag = next((t for t in tags if f"-{t}-" in head), None)
                if tag is None:
                    continue
                ids = json.loads(line)["output_ids"]
                first = next((i for i, x in enumerate(ids) if x in marks), None)
                went_on = first is not None and first < len(ids) - 1
                n = out.setdefault(tag, {"ids_kept": 0, "past_end": 0, "after_end_tokens": 0})
                n["ids_kept"] += 1
                n["past_end"] += went_on
                n["after_end_tokens"] += len(ids) - first - 1 if went_on else 0
    return out


def arm_metrics(d: Path, slo: tuple[float, float] | None = None) -> dict:
    """An arm's run dir: its meta and each load's metrics by rep, with the GPUs' busy time and the edges that fell in
    it, and its edges' summary (`edge_summary`). Where the arm's server kept the served ids and its meta names the
    checkpoint's end-of-text ids, a rep also has how many of its requests served one before their end and the
    output tokens after it (`past_end`)."""
    meta = json.loads((d / "meta.json").read_text())
    ran = MODELS[meta["model_id"]].slo
    slo = slo or ran
    edges = edge_records(d)
    tags = {json.loads(f.read_text()).get("tag"): f.parent for f in d.glob("*/rep-*/meta.json")}
    ended = {tags[t]: n for t, n in past_end(d, [t for t in tags if t], meta.get("end_of_text_ids") or ()).items()}
    loads: dict[str, list[dict]] = {}
    for name in meta.get("passes") or meta["rates"]:
        for rep in sorted((d / name).glob("rep-*"), key=lambda p: int(p.name.split("-")[1])):
            if (rep / RESULT).exists():
                if slo != ran:
                    raise SystemExit(f"{rep}: vLLM's client counted this pass's goodput against the SLO it ran with "
                                     f"(TTFT {ran[0]} s, TPOT {ran[1]} s); serve it again to count against another")
                # a sessions load's pass is read from its users' own results, so that a directory served under
                # another reading of the pass's rate has this one
                users = sorted(rep.glob(f"user-*/{RESULT}"))
                lm = bench_metrics(merged([json.loads(u.read_text()) for u in users]) if users else
                                   json.loads((rep / RESULT).read_text()))
            elif (rep / "client.jsonl").exists():
                lm = load_metrics([json.loads(x) for x in (rep / "client.jsonl").read_text().splitlines() if x.strip()],
                                  slo)
            else:
                continue
            m = json.loads((rep / "meta.json").read_text())
            loads.setdefault(name, []).append({"rep": int(rep.name.split("-")[1]), "rate": m["rate"],
                                               "concurrency": m.get("concurrency"),
                                               "unfinished": m.get("unfinished"), **lm,
                                               **schedule_metrics(m.get("schedule")),
                                               **gpu_metrics(m, lm.get("output_tokens")), **load_edges(edges, m),
                                               **ended.get(rep, {})})
    return {"schema": SCHEMA + "/metrics", "arm": meta["arm"], "parent": meta.get("parent"), "what": meta.get("what"),
            "run_id": meta.get("run_id"), "commit": meta.get("commit"), "vllm": (meta.get("versions") or {}).get("vllm"),
            "model_id": meta["model_id"], "slo": {"ttft_s": slo[0], "tpot_s": slo[1]},
            "startup_s": meta.get("startup_s"), "error": meta.get("error"), "tick": meta.get("tick"), "loads": loads,
            "certifier": meta.get("certifier"),
            "lifetime": meta.get("lifetime"), "left_out": meta.get("left_out") or 0, "reps": meta.get("reps"),
            # a sampled arm served before the arms differed in it was sent a seed
            "seeded": meta.get("seeded", bool(meta.get("sampled"))),
            "edges": edge_summary(edges), "boundary_rows": meta.get("boundary_rows"),
            "boundary_commits": meta.get("boundary_commits"), "planned": meta.get("planned")}


def summary(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    out = {"mean": statistics.fmean(xs), "n": len(xs)}
    if len(xs) >= 2:
        sd = statistics.stdev(xs)
        half = T95[max(k for k in T95 if k <= len(xs) - 1)] * sd / math.sqrt(len(xs))
        out |= {"sd": sd, "ci95": [out["mean"] - half, out["mean"] + half]}
    return out


KEYS = ("output_tok_per_s", "req_per_s", "goodput_req_per_s", "slo_attained", "ttft_p50_s", "ttft_p99_s", "itl_p50_s",
        "itl_p99_s", "steps", "tokens_per_step", "rows_per_step", "preemptions", "prompt_cached_share", "busy_gpu_s",
        "busy_gpu_s_per_ktok", "edges", "edge_stall_p50_s", "edge_stall_max_s", "edge_stall_s")


def table(arms: list[dict]) -> dict:
    """Each arm's metrics per load over its reps, and its tax (module docstring) against ``compiled`` and its parent."""
    by = {a["arm"]: a for a in arms}
    rows = []
    for a in arms:
        for load, reps in a["loads"].items():
            row = {"arm": a["arm"], "load": load, "run_id": a["run_id"], "reps": len(reps),
                   "lifetimes": len(a["lifetimes"]) if "lifetimes" in a else None,
                   "left_out": len((a.get("warm") or {}).get(load, ())),
                   "rate": reps[0]["rate"] if reps else None,
                   "concurrency": reps[0].get("concurrency") if reps else None, "sent": sum(r["sent"] for r in reps),
                   "unserved": sum(r["sent"] - r["ok"] for r in reps), **{k: summary([r.get(k) for r in reps]) for k in KEYS}}
            for ref_name, key in (("compiled", "tax_vs_compiled"), (a.get("parent"), "tax_vs_parent")):
                ref = by.get(ref_name)
                if ref is None or ref_name == a["arm"] or load not in ref["loads"]:
                    continue
                mine = summary([r.get("output_tok_per_s") for r in reps])
                theirs = summary([r.get("output_tok_per_s") for r in ref["loads"][load]])
                good_mine = summary([r.get("goodput_req_per_s") for r in reps])
                good_theirs = summary([r.get("goodput_req_per_s") for r in ref["loads"][load]])
                row[key] = {"of": ref_name, "throughput": tax(theirs, mine), "goodput": tax(good_theirs, good_mine)}
            rows.append(row)
    return {"schema": SCHEMA + "/table", "slo": arms[0]["slo"] if arms else None, "rows": rows,
            "arms": {a["arm"]: {k: a.get(k) for k in ("parent", "what", "run_id", "startup_s", "error", "tick", "edges")}
                     for a in arms}}


def tax(ref: dict, arm: dict) -> dict | None:
    """ref's mean over arm's, less one: the arm's extra compute per token; its interval by the delta method over both
    means' relative standard errors, only where each side has two reps or more (one rep has no spread to go by)."""
    if not ref.get("n") or not arm.get("n") or not arm["mean"]:
        return None
    r = ref["mean"] / arm["mean"]
    out = {"value": r - 1, "n": [ref["n"], arm["n"]]}
    if all(x["n"] >= 2 and x["mean"] for x in (ref, arm)):
        parts = [(x["sd"] / x["mean"]) ** 2 / x["n"] for x in (ref, arm)]
        df = min(ref["n"], arm["n"]) - 1
        half = T95[max(k for k in T95 if k <= df)] * r * math.sqrt(sum(parts))
        out["ci95"] = [r - half - 1, r + half - 1]
    return out


# ---------------------------------------------------------------------------------------------------- the serving cost

#: the arm the serving cost is of (the two-stage serve's launch line) and the arm it is over
OURS, REFERENCE = "two-stage", "compiled"
UNIT = "percent of stock's serving compute"
#: what a served engine of ours can change that the two-stage serve leaves off and no arm here turns on, so no cost
#: here counts it
NO_ARM = (
    "the engine class's served limits (engine_classes.CLASS_G_SERVED)",
    "the sparse attention kernel at one split (VERITY_SPARSE_MLA_NUM_SPLITS=1)",
    "the fixed expert alignment on the Marlin MoE path (verity_moe_align)",
    "the committing endpoint (verity-vllm serve-openai) and all it turns on: its engine profile, the Commit, the seal, "
    "exact int8, proof of useful work, proof of useful space, the memory audit, the network certifier in front of it",
)


def bench_loads(arm: dict) -> list[str]:
    """The loads an arm served by vLLM's client, in its order."""
    return list(dict.fromkeys(name.rpartition("@")[0] for name, reps in arm["loads"].items()
                              if reps and reps[0].get("concurrency") is not None))


def bench_rungs(arm: dict, load: str) -> dict[int, list[dict]]:
    """Each rung of ``load`` an arm served by vLLM's client, with that pass's reps."""
    return {reps[0]["concurrency"]: reps for name, reps in arm["loads"].items()
            if reps and reps[0].get("concurrency") is not None and name.rpartition("@")[0] == load}


def best_pass(arm: dict, load: str) -> tuple[int, list[dict]] | None:
    """The rung of ``load`` an arm serves best within the SLO, with that pass's reps: where its goodput, averaged over
    its reps, is largest (its output tokens a second, where no rung has any goodput), the lowest such rung."""
    rungs = bench_rungs(arm, load)

    def mean(c: int, key: str) -> float:
        return statistics.fmean(r.get(key) or 0.0 for r in rungs[c])

    if not rungs:
        return None
    c = max(sorted(rungs), key=lambda c: (mean(c, "goodput_req_per_s"), mean(c, "output_tok_per_s")))
    return c, rungs[c]


def _tok_per_s(reps: list[dict]) -> float:
    return statistics.fmean(r.get("output_tok_per_s") or 0.0 for r in reps)


def saturated(ref: dict, arm: dict, load: str) -> int | None:
    """The rung of ``load`` a cost is read at: of the rungs both arms served, the one where ``ref``'s output tokens a
    second, averaged over its reps, are highest (the lowest such rung). None where they share none. Whether ``ref``
    is saturated there is `peak`'s to say."""
    rungs = bench_rungs(ref, load)
    shared = sorted(set(rungs) & set(bench_rungs(arm, load)))
    if not shared:
        return None
    return max(shared, key=lambda c: _tok_per_s(rungs[c]))


def peak(ref: dict, load: str, rung: int) -> tuple[int, bool]:
    """Where ``ref`` served ``load`` fastest, and whether that shows it saturated at ``rung``: the rung, of all it
    served, where its output tokens a second are highest (the lowest such), and whether that is ``rung`` with a rung
    served above it. An arm served at its highest rung alone may serve more above: nothing here shows it saturated."""
    rungs = bench_rungs(ref, load)
    top = max(sorted(rungs), key=lambda c: _tok_per_s(rungs[c]))
    return top, top == rung and max(rungs) > rung


def kit() -> tuple:
    """The benchmark kit's door (`verity_bench`, benchmarks/harness): its paired ratio with the interval over its reps
    (`overhead.paired`), and its writer of a run's result.json (`overhead_node.write_result`). Loaded where a cost is
    computed: an arm's server and client import none of it."""
    harness = str(ROOT / "benchmarks/harness")
    if harness not in sys.path:
        sys.path.insert(0, harness)
    from verity_bench import overhead, overhead_node

    return overhead, overhead_node


def paired_cost(ref: list[float], arm: list[float]) -> dict:
    """How much more compute ``arm`` needs than ``ref`` for the same work, a fraction, from each side's rate of work
    (tokens or requests a second) in the reps both have: the kit's paired ratio (`kit`), the median over reps of ref's
    rate over arm's in the same rep, less one, with the kit's 95% interval, the order statistics of those ratios that
    bound their median, which fewer than six reps can't give. Where a rep of either side has no rate the reps don't
    pair: the value is the ratio of the sides' mean rates, with no interval, and None where either mean is zero."""
    out: dict = {"value": None}
    if statistics.fmean(ref) > 0 and statistics.fmean(arm) > 0:
        out["value"] = statistics.fmean(ref) / statistics.fmean(arm) - 1
        if all(ref) and all(arm):
            p = kit()[0].paired([1 / y for y in arm], [1 / x for x in ref])
            out["value"] = p["gamma"]
            if p["ci95"]:
                out["ci95"] = p["ci95"]
    return out


def within_slo(ref: dict, arm: dict, load: str) -> dict | None:
    """What ``arm`` costs over ``ref`` at ``load`` within the SLO, a fraction: how much more serving compute the arm
    needs for the same traffic within it, `paired_cost` of the sides' goodputs, a rep's serving compute a request within the
    SLO being the reciprocal of its goodput. Each side is read at its best pass (`best_pass`), over the reps both
    have. Where either side's mean goodput is zero it has no finite value. None where a side served no pass of the
    load."""
    a, b = best_pass(ref, load), best_pass(arm, load)
    if a is None or b is None:
        return None
    ra, rb = ({r["rep"]: r for r in reps} for _, reps in (a, b))
    reps = sorted(set(ra) & set(rb))
    if not reps:
        return None

    def side(x: dict, c: int, by: dict[int, dict]) -> tuple[dict, list[float]]:
        good = [by[k].get("goodput_req_per_s") or 0.0 for k in reps]
        return {"arm": x["arm"], "concurrency": c, "goodput_req_per_s": statistics.fmean(good),
                "met": sum(by[k].get("met") or 0 for k in reps), "sent": sum(by[k]["sent"] for k in reps)}, good

    (sa, ga), (sb, gb) = side(ref, a[0], ra), side(arm, b[0], rb)
    return {"reps": len(reps), "reference": sa, "arm": sb, **paired_cost(ga, gb)}


def serving_cost(ref: dict, arm: dict, load: str, rung: int | None = None) -> dict | None:
    """What ``arm`` costs over ``ref`` at ``load``, a fraction: how much more GPU time the arm needs a token, its tax.
    Every arm serves the same prompts, each decoded to its length, so at a rung where the reference has no idle GPU
    time its output tokens a second over the arm's, less one, is the arm's extra GPU time a token. Both sides are read
    at one rung, ``rung`` or the one where the reference serves the most tokens a second (`saturated`), over the reps
    both have there (`paired_cost`), each side with its output and prompt tokens a second and the tokens a request it
    served. ``saturated`` says whether the reference's own rungs show it saturated at that rung, and ``peak`` the rung
    where it served the most (`peak`). No SLO enters it: ``within`` is the same traffic's cost within the SLO
    (`within_slo`), which is this cost only where both sides meet the SLO for every request. None where the arms share
    no rung of the load, or no rep at the rung. Refused where a rep's two passes sent different numbers of prompts
    (``--pass-prompts`` on one side alone): they are not the same requests."""
    a, b = bench_rungs(ref, load), bench_rungs(arm, load)
    c = saturated(ref, arm, load) if rung is None else rung
    if c is None or c not in a or c not in b:
        return None
    ra, rb = ({r["rep"]: r for r in x[c]} for x in (a, b))
    reps = sorted(set(ra) & set(rb))
    if not reps:
        return None
    for k in reps:
        if ra[k].get("sent") != rb[k].get("sent"):
            raise SystemExit(f"cost: {ref['arm']} and {arm['arm']} sent passes of different sizes at load {load}, "
                             f"{at_rung(c)} ({ra[k].get('sent')} and {rb[k].get('sent')} prompts in rep {k}; "
                             "--pass-prompts): a cost is of the same requests on both sides")

    def side(x: dict, by: dict[int, dict]) -> tuple[dict, list[float]]:
        tok = [by[k].get("output_tok_per_s") or 0.0 for k in reps]
        served = sum(by[k].get("ok") or 0 for k in reps)

        def mean(key: str) -> float | None:
            return None if any(by[k].get(key) is None for k in reps) else statistics.fmean(by[k][key] for k in reps)

        def each(key: str) -> float | None:
            return None if not served or any(by[k].get(key) is None for k in reps) else sum(by[k][key] for k in reps) / served

        return {"arm": x["arm"], "concurrency": c, "output_tok_per_s": statistics.fmean(tok),
                "prompt_tok_per_s": mean("prompt_tok_per_s"), "req_per_s": mean("req_per_s"),
                "prompt_tokens_per_request": each("prompt_tokens"), "output_tokens_per_request": each("output_tokens")}, tok

    (sa, ta), (sb, tb) = side(ref, ra), side(arm, rb)
    top, shown = peak(ref, load, c)
    return {"load": load, "reps": len(reps), "rung": c, "saturated": shown, "peak": top, "reference": sa, "arm": sb,
            **paired_cost(ta, tb), "within": within_slo(ref, arm, load)}


def verdict(cost: dict, target: float | None, floor: float = 0.0) -> str | None:
    """Whether a cost (`serving_cost`) is within ``target``, a fraction: ``meets`` where its whole interval, widened by
    ``floor`` (the noise floor) at each end, is at or under the target; ``misses`` where all of it is over; else
    ``cannot tell`` (a cost with no finite value, or of fewer than six reps, has no interval). None with no target."""
    if target is None:
        return None
    if cost["value"] is None or "ci95" not in cost:
        return "cannot tell"
    lo, hi = cost["ci95"]
    return "meets" if hi + floor <= target else "misses" if lo - floor > target else "cannot tell"


#: a cost's verdict where its arm leaves off a change the target counts, or has a change on that the target has no
#: part for: the cost and the target are of different changes, so the cost neither meets nor misses it. Not
#: `verdict`'s "cannot tell", which is of an interval that straddles a target it is comparable with.
NOT_COMPARABLE = "not comparable"


def target_parts(model: Model, arm: str) -> list[dict]:
    """A model's target by part, for an arm: each part's points, what it rests on and its changes, and those of its
    changes the arm leaves off (`changes_of`). A part with a change off is not in the arm's cost, so the cost is not
    comparable with the target."""
    on = changes_of(arm)
    return [{"name": p.name, "points": p.points, "basis": p.basis, "what": p.what, "changes": list(p.changes),
             "off": [c for c in p.changes if c not in on]} for p in model.target_parts]


def not_in_target(model: Model, arm: str) -> list[str]:
    """The changes an arm has on (`changes_of`) that no part of a model's target counts: the cost model does not price
    them, so a cost that holds them is not comparable with the target."""
    return [c for c in changes_of(arm) if not any(c in p.changes for p in model.target_parts)]


def in_target(model: Model, arm: str, over: str) -> list[dict]:
    """What a model's target has for the changes ``arm`` turns on over ``over`` (`changes_of`): for each part with a
    change among them, those changes, the part, its points and whether they are the whole part, so that its points
    are of what the step adds; and the changes no part is of, which are not in the cost model."""
    step = [c for c in changes_of(arm) if c not in changes_of(over)]
    out = [{"changes": [c for c in p.changes if c in step], "part": p.name, "points": p.points,
            "whole": all(c in step for c in p.changes)} for p in model.target_parts if any(c in step for c in p.changes)]
    rest = [c for c in step if not any(c in p.changes for p in model.target_parts)]
    return out + ([{"changes": rest, "part": None, "points": None, "whole": False}] if rest else [])


def same_engine(a: str, b: str) -> bool:
    """Whether two arms start the same engine: only their names, parents and descriptions differ."""
    x, y = ARMS.get(a), ARMS.get(b)
    return x is not None and y is not None and replace(x, parent=None, what="") == replace(y, parent=None, what="")


def _pct(x: float | None) -> float | None:
    return None if x is None else round(100 * x, 2)


def _cost(c: dict | None) -> dict:
    return {"value": _pct(c["value"]) if c else None, "ci95": [_pct(x) for x in c["ci95"]] if c and "ci95" in c else None}


def commitment(arm: dict) -> dict | None:
    """What an arm's boundary commitment did, from its record (`await_commits`): the requests its loads finished, the
    spans committed, those whose commit failed and those committed with their history tables, beside whether the
    arm's object asked for tables and the window it committed at. None for an arm with no boundary commitment."""
    b = arm.get("boundary_commits")
    if not b:
        return None
    failed, rows = b.get("failed", 0), arm.get("boundary_rows") or {}
    return {"finished": b["finished"], "committed": b["spans"] - failed, "failed": failed, "tables": b.get("tables", 0),
            "tables_asked": bool(rows.get("tables")), "window_s": rows.get("window_s")}


def lifetimes(arms: list[dict]) -> list[dict]:
    """Arms' metrics with one arm's server lifetimes made one arm's (`run --lifetime`: run.sh's LIFETIMES serves a
    job's arms that many times, in an order that turns round each time, so that two arms' lifetimes are neighbours in
    time). Lifetime k's pass r is the arm's rep (k - 1) * R + r, R the most reps a lifetime given was to serve, so a
    cost pairs the same pass of the same lifetime of its two arms; each rep keeps its ``lifetime`` and its ``pass``.
    The passes a lifetime leaves out (`run --leave-out`: its first, its warm-up, served and recorded like the rest)
    are in no load's reps and so in no figure: they are the arm's ``warm``, by load. An arm with one directory, no
    lifetime and nothing left out is returned as it is. Refused where two directories of an arm name one lifetime, or
    one names none, or they leave out different passes."""
    stride = max((a.get("reps") or 0 for a in arms if a.get("lifetime") is not None), default=0)
    out: list[dict] = []
    by: dict[str, dict] = {}
    for a in arms:
        k, skip = a.get("lifetime"), a.get("left_out") or 0
        if "lifetimes" in a or (k is None and not skip):
            out.append(a)
            continue
        b = by.get(a["arm"])
        if b is None:
            b = by[a["arm"]] = {**a, "loads": {}, "warm": {}, "lifetimes": [], "lifetime": None}
            out.append(b)
        elif k is None or k in b["lifetimes"] or None in b["lifetimes"] or skip != b["left_out"]:
            raise SystemExit(f"cost: two directories of arm {a['arm']} are "
                             + ("lifetimes of which one names no number" if k is None or None in b["lifetimes"] else
                                f"both its lifetime {k}" if k in b["lifetimes"] else
                                f"lifetimes that leave out {b['left_out']} and {skip} passes") + ": give one of them")
        elif differ := [key for key in ("model_id", "slo", "vllm", "planned") if a.get(key) != b.get(key)]:
            raise SystemExit(f"cost: two lifetimes of arm {a['arm']} differ in {', '.join(differ)}: they are not one "
                             "arm's")
        else:
            x, y = b.get("boundary_commits"), a.get("boundary_commits")
            if x and y:
                b["boundary_commits"] = {key: x.get(key, 0) + y.get(key, 0) for key in ("spans", "finished", "failed", "tables")}
            b["error"] = b.get("error") or a.get("error")
        b["lifetimes"].append(k)
        for name, reps in a["loads"].items():
            for r in reps:
                rep = {**r, "lifetime": k or 1, "pass": r["rep"], "rep": ((k or 1) - 1) * stride + r["rep"]}
                (b["warm"] if r["rep"] <= skip else b["loads"]).setdefault(name, []).append(rep)
    for b in by.values():
        b["lifetimes"] = sorted(k or 1 for k in b["lifetimes"])
        for held in (b["loads"], b["warm"]):
            for reps in held.values():
                reps.sort(key=lambda r: r["rep"])
    return out


def joined(arms: list[dict]) -> dict[str, dict]:
    """Arms' metrics by arm, the directories of one arm joined: an arm that two jobs served, each a load of its own,
    has both loads' passes, each load with its own job's commit and run (``of``), and its boundary commitment's counts
    are the two jobs' sums. An arm's server lifetimes are one arm's first (`lifetimes`). Two other directories of one
    arm that served the same load, where a rung's reps would be of two server lifetimes paired by nothing, or that
    differ in the model, the SLO or vLLM, are refused."""
    by: dict[str, dict] = {}
    for a in lifetimes(arms):
        of = {n: {"commit": a.get("commit"), "run": a.get("run_id")} for n in bench_loads(a)}
        b = by.get(a["arm"])
        if b is None:
            by[a["arm"]] = {**a, "loads": dict(a["loads"]), "of": of}
            continue
        twice = sorted(set(of) & set(b["of"])) or sorted(set(a["loads"]) & set(b["loads"]))
        differ = [k for k in ("model_id", "slo", "vllm") if a.get(k) != b.get(k)]
        if twice or differ:
            raise SystemExit(f"cost: two directories of arm {a['arm']} "
                             + (f"both served {', '.join(twice)}" if twice else f"differ in {', '.join(differ)}")
                             + ": give one of them")
        b["loads"] |= a["loads"]
        b["of"] |= of
        x, y = b.get("boundary_commits"), a.get("boundary_commits")
        if x and y:
            b["boundary_commits"] = {k: x.get(k, 0) + y.get(k, 0) for k in ("spans", "finished", "failed", "tables")}
        b["boundary_commits"] = b.get("boundary_commits") or y
        b["boundary_rows"] = b.get("boundary_rows") or a.get("boundary_rows")
        b["planned"] = {**(b.get("planned") or {}), **(a.get("planned") or {})}
    return by


#: the reference with its prefix cache off (`ARMS`): on the load that repeats a prefix, what the cache alone is worth
CACHE_OFF = "no-prefix-cache"
#: what the number is called beside the row under it: its loads repeat no prefix
NO_REPEAT = "no repeated prefix"
#: what the load that repeats a prefix is where its requests are a given table's (`read_plan`), sent as they arrive
GIVEN = ("a table's planned requests, each sent at its arrival time, of synthetic token ids (vLLM's timed_trace "
         "dataset)")
#: where a load's requests are sent as they arrive, the reference kept up where it served all that arrived and each of
#: its passes ended within this share of the time the requests arrive over after the last one arrived
KEPT_UP = 0.1
#: a row's figures that are its cost (`prefix_row`): output tokens a second, or the two latencies
TOKENS, LATENCY = "output tokens a second", "latency"


def latency_cost(ref: list[dict], arm: list[dict], key: str) -> dict:
    """How much longer ``arm``'s ``key``, a latency in seconds, is than ``ref``'s over the same reps, a fraction:
    `paired_cost` of the reciprocals, so the median over reps of the arm's latency over the reference's, less one,
    with its interval. No value where a rep of either side has no such latency."""
    a, b = ([r.get(key) for r in x] for x in (ref, arm))
    return paired_cost([1 / x for x in a], [1 / x for x in b]) if all(a) and all(b) else {"value": None}


def readings(ref: list[dict], arm: list[dict], key: str, longer: bool = False) -> dict:
    """A figure of a row whose requests are sent as they arrive, as its readings in percent, one a rep both sides have,
    with no mean and no interval (``value`` and ``ci95`` are None): the reference's ``key`` over the arm's in the same
    rep, less one, or, where ``longer``, the arm's over the reference's. A rep where a side has no ``key`` reads
    None."""
    pairs = [(y.get(key), x.get(key)) if longer else (x.get(key), y.get(key)) for x, y in zip(ref, arm)]
    return {**_cost(None), "readings": [_pct(a / b - 1) if a and b else None for a, b in pairs]}


def prefix_row(by: dict[str, dict], reference: str, arm: str) -> dict | None:
    """The row under the number, from arms' joined metrics (`joined`), every figure in percent: what ``arm`` costs over
    ``reference`` on the load that repeats a prefix (`PREFIX`), which the number's loads say nothing of. It has no
    target and no verdict. Its figures: the reference's output tokens a second over the arm's, less one
    (``tokens``, `serving_cost`); how much longer the arm's time to the first token and its 99th-percentile gap
    between tokens are (`latency_cost`); the share of the prompt tokens each side's prefix cache served in fact, from
    the engine's own counters, beside the share the load plans to repeat (`planned`); and whether each side served
    every request of every rep. ``cost`` names the figures that are the cost. With one request in flight it is
    `TOKENS`, and ``value`` is that figure. Where the requests are sent as they arrive (`ARRIVING`), each side has
    ``tail_s``, how long after the last request arrived its longest pass ended, ``kept_up`` says whether the
    reference kept up (it served all that arrived, within `KEPT_UP`) and ``why`` says it in words. At arrivals the
    reference keeps up with, tokens a second follows the arrivals and is not a cost: the cost is `LATENCY`, the two
    latencies. Where the reference does not keep up the row is read as the number is: `TOKENS`. Either way each of
    such a row's figures is its readings, one a rep (`readings`), never a mean with an interval, and ``value`` is
    None.
    ``scaled_down`` says the load is the four requests of `PAIRS` and not the plan they are from. ``cache_off`` is
    the same for the reference with its prefix cache off (`CACHE_OFF`), where that arm served the load. None where
    ``reference`` or ``arm`` has not served it; refused where an arm of the row served another plan than
    ``reference`` did."""
    ref = by[reference]
    first = serving_cost(ref, by[arm], PREFIX)
    if first is None:
        return None
    rung = first["rung"]
    plan = (ref.get("planned") or {}).get(PREFIX)
    if plan is None:
        raise SystemExit(f"cost: {reference}'s directory that served load {PREFIX} names no plan (meta.json: planned)")
    arriving = rung == ARRIVING

    def over(name: str) -> dict | None:
        x = by.get(name)
        c = serving_cost(ref, x, PREFIX, rung) if x else None
        if c is None:
            return None
        if (x.get("planned") or {}).get(PREFIX) != plan:
            raise SystemExit(f"cost: {reference} and {name} served load {PREFIX} from different plans, or {name}'s "
                             "directory names none (meta.json: planned)")
        a, b = ({r["rep"]: r for r in bench_rungs(y, PREFIX)[rung]} for y in (ref, x))
        reps = sorted(set(a) & set(b))

        def side(rs: list[dict], of: dict) -> dict:
            cached = [r.get("prompt_cached_share") for r in rs]
            return {"arm": of["arm"], "output_tok_per_s": of["output_tok_per_s"],
                    "cached_share": None if any(v is None for v in cached) else _pct(statistics.fmean(cached)),
                    "served_all": all(r.get("ok") == r.get("sent") == plan["requests"] for r in rs),
                    "tail_s": round(max(r["span_s"] for r in rs) - plan["span_s"], 3) if arriving else None}

        ra, rb = [a[k] for k in reps], [b[k] for k in reps]
        sa, sb = side(ra, c["reference"]), side(rb, c["arm"])
        if not arriving:
            tokens, kept, why = _cost(c), None, None
            ttft, itl = (_cost(latency_cost(ra, rb, key)) for key in ("ttft_p50_s", "itl_p99_s"))
        else:
            tokens = readings(ra, rb, "output_tok_per_s")
            ttft, itl = (readings(ra, rb, key, longer=True) for key in ("ttft_p50_s", "itl_p99_s"))
            short = [] if sa["served_all"] else [f"{sa['arm']} did not serve all that arrived"]
            late = [f"{sa['arm']}'s longest pass ended {sa['tail_s']:,.1f} s after the last arrival, over a tenth of the "
                    f"{plan['span_s']:,.1f} s the requests arrive over"] * (sa["tail_s"] > KEPT_UP * plan["span_s"])
            kept = not short and not late
            why = f"{sa['arm']} kept up" if kept else " and ".join(short + late)
        return {"value": tokens["value"], "ci95": tokens["ci95"], "cost": LATENCY if kept else TOKENS, "kept_up": kept,
                "why": why, "tokens": tokens, "reps": c["reps"], "ttft": ttft, "itl_p99": itl, "reference": sa, "arm": sb}

    four = planned(trace_lines(PAIRS))
    scaled = all(plan.get(k) == four[k] for k in ("requests", "prompt_tokens", "repeated_tokens"))
    return {"load": PREFIX, "what": GIVEN if arriving else LOADS[PREFIX].what, "label": "on a repeated prefix",
            "scaled_down": scaled,
            "planned": {**plan, "repeated_share": _pct(plan["repeated_tokens"] / plan["prompt_tokens"])},
            "rung": rung, **over(arm), "cache_off": over(CACHE_OFF) if arm != CACHE_OFF else None}


#: the most a serve at the recipe's settings may cost with our changes over stock at the same settings and load, in
#: percent
SESSION_TARGET = 1.0
#: a cost report's ``metric`` where it is a sessions load's (`session_cost`)
SESSION_COST = "session_cost"


def over_lifetimes(pairs: Sequence[Sequence[tuple[float, float]]]) -> dict:
    """A cost whose unit is a server lifetime, a fraction. ``pairs`` holds, for each pair of lifetimes (one of the
    reference's and one of the arm's), the two sides' rates of work in each pass both counted. A pair's cost is the
    arm's time for a unit of work (one over its rate), summed over the pair's passes, over the reference's, less one:
    where every pass is of the same work, as a sessions load's are, the arm's seconds for the pair's passes over the
    reference's. The cost is the mean of the pairs' costs, with Student's t 95% interval over them (`T95`), which one
    pair can't give. Passes of one lifetime share whatever
    its server had of its own, so they are not independent, and an interval over them as if they were is too narrow.
    ``each`` has the pairs' costs, ``between`` their standard deviation, and ``within`` the standard deviation of a
    pass's own cost (the reference's rate over the arm's, less one) about its pair's mean, pooled over the pairs
    (absent where no pair has two passes). No value where there is no pair, or a pass of either side has no rate."""
    if not pairs or not all(pairs) or any(not x or not y for p in pairs for x, y in p):
        return {"value": None}
    costs = [[x / y - 1 for x, y in p] for p in pairs]
    each = [sum(1 / y for _, y in p) / sum(1 / x for x, _ in p) - 1 for p in pairs]
    out: dict = {"value": statistics.fmean(each), "each": each}
    if len(each) > 1:
        out["between"] = statistics.stdev(each)
        half = T95[max(k for k in T95 if k <= len(each) - 1)] * out["between"] / math.sqrt(len(each))
        out["ci95"] = [out["value"] - half, out["value"] + half]
    free = sum(len(c) - 1 for c in costs)
    if free:
        out["within"] = math.sqrt(sum((x - statistics.fmean(c)) ** 2 for c in costs for x in c) / free)
    return out


def by_lifetime(ref: dict[int, dict], arm: dict[int, dict]) -> dict[int, list[tuple[float, float]]]:
    """The two sides' output tokens a second in each rep both have, by the server lifetime the rep is a pass of
    (`lifetimes`; one lifetime where the arms were not served in them): `over_lifetimes`' pairs."""
    out: dict[int, list[tuple[float, float]]] = {}
    for k in sorted(set(ref) & set(arm)):
        out.setdefault(ref[k].get("lifetime", 1), []).append((ref[k].get("output_tok_per_s") or 0.0,
                                                               arm[k].get("output_tok_per_s") or 0.0))
    return out


def own_noise(reps: dict[int, dict]) -> dict | None:
    """The noise floor of a cost from the stock arm's own server lifetimes (`lifetimes`), where it has two at least:
    the cost of each second lifetime over the one before it, the same pass of the two paired (`over_lifetimes`, a
    pair of stock's lifetimes its unit), which is stock over stock a lifetime apart, as far apart as a cost's own
    pairs or further. ``reps``: the arm's reps at one rung of a load, by rep. None for an arm served in fewer than
    two lifetimes."""
    of: dict[int, dict[int, float]] = {}
    for r in reps.values():
        if "lifetime" in r:
            of.setdefault(r["lifetime"], {})[r["pass"]] = r.get("output_tok_per_s") or 0.0
    order = sorted(of)
    pairs = {(x, y): [(of[x][n], of[y][n]) for n in sorted(set(of[x]) & set(of[y]))]
             for x, y in zip(order[::2], order[1::2], strict=False)}
    pairs = {k: v for k, v in pairs.items() if v}
    if not pairs:
        return None
    return {**over_lifetimes(list(pairs.values())), "reps": sum(map(len, pairs.values())),
            "lifetimes": [list(k) for k in pairs]}


def session_loads(arms: list[dict], reference: str) -> list[str]:
    """The sessions loads (`Load.sessions`) ``reference`` served, in its order; none for an arm not among ``arms``."""
    ref = joined(arms).get(reference)
    return [n for n in bench_loads(ref) if LOADS[n].sessions] if ref else []


def session_cost(arms: list[dict], arm: str, reference: str, target: float | None = None,
                 load: str | None = None) -> dict | None:
    """The cost of ``arm`` over ``reference`` on the sessions load both served (`Load.sessions`; ``load``, or the
    first the reference served: `session_loads`), from arms' metrics (one arm's directories joined: `joined`), every
    figure in percent: the arm's seconds for the same requests over the reference's, less one, at the users both
    served, a pass's seconds from its users' first send to the last one's last answer (`merged`).
    Its unit is a server lifetime (`over_lifetimes`): the mean over the lifetimes an arm of each one's cost, the arm's
    seconds for the passes both sides counted in it, summed, over the reference's, less one, with Student's t 95%
    interval over the lifetimes. Its verdict is
    against ``target`` (`SESSION_TARGET` where none is given) by that interval's upper end alone: "meets" where it is
    under the target, "misses" where the lower end is over it, else "cannot tell", as with one lifetime an arm, which
    has no such interval. Beside it and in no verdict: ``passes``, the cost over the passes taken as independent, the
    median with its interval over them (`serving_cost`), which is too narrow where the lifetimes' costs spread more
    than a lifetime's passes do; ``spread``, the standard deviation between the lifetimes' costs and of a pass within
    its lifetime; ``readings``, the same cost where a pass's rate of work is read another way (`READINGS`: by the
    seconds its GPUs were busy, by its users' seconds summed, by the sum of its users' own rates), each None where a
    pass has no such figure; ``by_lifetime``, each lifetime's own cost, where the arms were served in lifetimes
    (`lifetimes`), each lifetime's first passes left out (``left_out`` an arm, ``warm`` how many were served); and the noise floor,
    the reference's engine over itself, a lifetime pair its unit too: another arm that starts the reference's engine
    (`same_engine`) over the reference, where one served the load, and else the reference's own lifetimes, each
    second one over the one before it (`own_noise`). ``differs`` says whether the floor's own interval leaves out
    zero: stock then differs from stock. Also beside them: the share of its prompt tokens the load plans to repeat
    (`planned`), the share each side's prefix cache
    served in fact, from the engine's own counters, and whether each side served every request of every rep.
    ``same_work`` says whether both did, served the same count of output tokens in every rep, and their caches served
    the same share of the same prompts in it: the same requests and as many tokens, which need not be the same token
    ids, since two passes of the same prompts can serve other tokens, sampled or greedy. Where it is false the two
    sides did not do the same work, and the verdict is "cannot tell". Each side also has
    its prompt and output tokens a second a GPU, the output tokens a second a user gets at the P90 time a token
    (`merged`), the engine steps a pass took (``steps``) and, where it decodes speculatively, the share of its draft
    tokens accepted and the tokens a draft gave (`schedule_metrics`): a reader sees whether a cost came with more
    steps or fewer accepted drafts, and no figure is adjusted for either. A side whose server kept the served ids
    also has ``past_end``: of the requests of its counted passes, how many served an end-of-text id before their
    end and went on, as a request decoded to its recorded length does, and the share of its output tokens after one
    (`past_end`); None for a side that kept none, as stock does. None where ``reference`` served no sessions
    load; refused where ``arm`` did not serve it or served another plan, or where the arms were served by different
    builds of vLLM."""
    by = joined(arms)
    if reference not in by:
        raise SystemExit(f"cost: no directory of arm {reference} (given: {', '.join(by) or 'none'})")
    ref = by[reference]
    load = next((n for n in bench_loads(ref) if LOADS[n].sessions and load in (None, n)), None)
    if load is None:
        return None
    c = serving_cost(ref, by[arm], load) if arm in by else None
    if c is None:
        raise SystemExit(f"cost: {reference} served load {load} and {arm} did not, at no users both served (given: "
                         f"{', '.join(by)})")
    plan = (ref.get("planned") or {}).get(load)
    if plan is None or (by[arm].get("planned") or {}).get(load) != plan:
        raise SystemExit(f"cost: {reference} and {arm} served load {load} from different plans, or a directory names "
                         "none (meta.json: planned)")
    rung = c["rung"]
    twin = next((x for x in by.values() if x["arm"] not in (reference, arm) and same_engine(x["arm"], reference)
                 and rung in bench_rungs(x, load)), None)
    built = {x["arm"]: x.get("vllm") for x in (ref, by[arm], *([twin] if twin else []))}
    if len(set(built.values())) > 1:
        raise SystemExit("cost: the arms were served by different builds of vLLM ("
                         + ", ".join(f"{n}: {v}" for n, v in built.items()) + "): a cost is over stock from the same "
                         "environment (run.sh: VENV)")
    gpus = tp_size(MODELS[by[arm]["model_id"]])
    a, b = ({r["rep"]: r for r in bench_rungs(x, load)[rung]} for x in (ref, by[arm]))
    reps = sorted(set(a) & set(b))
    lives = by_lifetime(a, b)
    over = over_lifetimes(list(lives.values()))
    if twin:
        again = by_lifetime(a, {r["rep"]: r for r in bench_rungs(twin, load)[rung]})
        noise = {**over_lifetimes(list(again.values())), "reps": sum(map(len, again.values()))} if again else None
    else:
        noise = own_noise(a)
    each = [{"lifetime": n, "value": _pct(x), "reps": len(lives[n])}
            for n, x in zip(lives, over.get("each") or [None] * len(lives), strict=True)]
    warm = [len((x.get("warm") or {}).get(pass_name(load, rung), ())) for x in (ref, by[arm])]
    sent = rung * plan["requests"] // plan["users"]

    def side(rs: dict[int, dict], of: dict) -> dict:
        def mean(key: str) -> float | None:
            xs = [rs[k].get(key) for k in reps]
            return None if None in xs else statistics.fmean(xs)

        whole, slow, accepted = mean("total_tok_per_s"), mean("tpot_p90_s"), mean("accepted_per_draft")
        kept = [rs[k] for k in reps if rs[k].get("ids_kept")]
        tokens = sum(r.get("output_tokens") or 0 for r in kept)
        return {**of, "cached_share": _pct(mean("prompt_cached_share")),
                "served_all": all(rs[k].get("ok") == rs[k].get("sent") == sent for k in reps),
                "tok_per_s_a_gpu": None if whole is None else whole / gpus,
                "user_tok_per_s_p90": 1 / slow if slow else None, "steps": mean("steps"),
                "draft_accepted_share": _pct(mean("draft_accepted_share")),
                "tokens_per_draft": None if accepted is None else 1 + accepted,
                "past_end": {"requests": sum(r["past_end"] for r in kept), "of": sum(r["ids_kept"] for r in kept),
                             "tokens_after_share": _pct(sum(r["after_end_tokens"] for r in kept) / tokens)
                             } if tokens and len(kept) == len(reps) else None}

    def reading(work: Callable[[dict], float | None]) -> float | None:
        """The cost under another figure of a pass's rate of work, the same passes paired the same way."""
        pairs: dict[int, list[tuple[float, float]]] = {}
        for k in reps:
            pairs.setdefault(a[k].get("lifetime", 1), []).append((work(a[k]) or 0.0, work(b[k]) or 0.0))
        return _pct(over_lifetimes(list(pairs.values()))["value"])

    def per(key: str) -> Callable[[dict], float | None]:
        return lambda r: 1 / r[key] if r.get(key) else None

    sa, sb = side(a, c["reference"]), side(b, c["arm"])
    cached = [(a[k].get("prompt_cached_share"), b[k].get("prompt_cached_share")) for k in reps]
    same = sa["served_all"] and sb["served_all"] and all(x is not None and x == y for x, y in cached) \
        and all(a[k].get("output_tokens") == b[k].get("output_tokens") for k in reps)
    target = SESSION_TARGET if target is None else target
    lo, hi = over.get("ci95") or (None, None)
    told = "cannot tell" if not same or hi is None else \
        "meets" if hi < target / 100 else "misses" if lo > target / 100 else "cannot tell"
    floor = noise and (noise.get("ci95") or (None, None))
    of = {n: by[n]["of"][load] for n in (reference, arm)}
    return {"value": _pct(over["value"]), "unit": UNIT, "commit": of[arm]["commit"], "run": of[arm]["run"],
            "metric": SESSION_COST, "better": "lower", "target": target, "target_basis": "given",
            "ci95": _cost(over)["ci95"], "verdict": told,
            "same_work": same, "model": MODELS[by[arm]["model_id"]].served, "arm": arm, "reference": reference,
            "load": load, "what": LOADS[load].what, "users": rung, "reps": c["reps"], "lifetimes": len(lives),
            "passes": {**_cost(c), "reps": c["reps"]},
            "spread": {"between": _pct(over.get("between")), "within": _pct(over.get("within"))},
            "readings": {"busy_gpu_s": reading(per("busy_gpu_s")), "users_s": reading(per("users_s")),
                         "users_rates": reading(lambda r: r.get("users_output_tok_per_s"))},
            "by_lifetime": each if any("lifetime" in a[k] for k in reps) else None,
            "seeded": {"reference": bool(ref.get("seeded")), "arm": bool(by[arm].get("seeded"))},
            "left_out": {"passes": by[arm].get("left_out") or 0, "warm": warm} if any(warm) else None,
            "planned": {**plan, "repeated_share": _pct(plan["repeated_tokens"] / plan["prompt_tokens"])},
            "noise_floor": {"arm": twin["arm"] if twin else reference, **_cost(noise), "reps": noise["reps"],
                            "pairs": len(noise.get("each") or ()),
                            "differs": None if floor[0] is None else not floor[0] <= 0 <= floor[1],
                            **({} if twin else {"lifetimes": noise["lifetimes"]})} if noise else None,
            "sides": {"reference": sa, "arm": sb}, "vllm": by[arm].get("vllm"), "gpus": gpus,
            "reference_commit": of[reference]["commit"], "reference_run": of[reference]["run"]}


#: what a sessions cost says of its requests' lengths: `bench_argv` has vLLM's client send each with its recorded
#: output length and ``ignore_eos``, in every arm, so the sides' counts of output tokens are equal
TO_LENGTH = "each request decoded to its recorded length, past any end-of-text token"


def past_end_text(r: dict) -> list[str]:
    """The line of a sessions cost (`session_cost`'s result) on the requests that went on past an end-of-text token
    (`past_end`), where a side's server kept the served ids; it names a side that kept none. No line where neither
    did."""
    sides = [r["sides"][k] for k in ("reference", "arm")]
    kept = [x for x in sides if x.get("past_end")]
    if not kept:
        return []
    has = [f"{x['past_end']['requests']:,} of {x['arm']}'s {x['past_end']['of']:,} requests served an end-of-text "
           f"token before their end and went on, {x['past_end']['tokens_after_share']:.2f}% of its output tokens "
           "after one" for x in kept]
    none = [f"{x['arm']}'s server keeps no ids, so the count is {kept[0]['arm']}'s alone" for x in sides if x not in kept]
    return ["  past an end-of-text token, in the passes counted: " + "; ".join(has + none)]


#: what else a sessions load's passes can be read by, beside the cost and in no verdict (`session_cost`'s
#: ``readings``): the seconds the GPUs were busy in a pass, the seconds its users each took, summed, and the sum of
#: its users' own rates (`merged`)
READINGS = {"busy_gpu_s": "the GPUs' busy seconds", "users_s": "the users' seconds summed",
            "users_rates": "the users' own rates summed"}


def session_text(r: dict) -> str:
    """A sessions load's cost (`session_cost`) as text: the cost with its interval and verdict first."""
    def rate(x: dict) -> str:
        cached = "its prefix cache's share unread" if x["cached_share"] is None else \
            f"{x['cached_share']:.2f}% of its prompt tokens from its prefix cache"
        each = "" if x["tok_per_s_a_gpu"] is None else f" ({x['tok_per_s_a_gpu']:,.1f} prompt and output tokens a second a GPU)"
        user = "" if x["user_tok_per_s_p90"] is None else \
            f", {x['user_tok_per_s_p90']:,.1f} output tokens a second a user at the P90 time a token"
        steps = "" if x.get("steps") is None else f", {x['steps']:,.0f} engine steps a pass"
        drafts = ", no drafts" if x["tokens_per_draft"] is None else \
            f", {x['draft_accepted_share']:.2f}% of its draft tokens accepted, {x['tokens_per_draft']:.2f} tokens a draft"
        return (f"{x['arm']}: {x['output_tok_per_s']:,.1f} output tokens a second{each}{user}, {cached}{steps}{drafts}, "
                f"{'every request served' if x['served_all'] else 'NOT every request served'}")

    n, plan, spread, other = r["noise_floor"], r["planned"], r["spread"], r.get("readings") or {}

    def points(x: float | None) -> str:
        return "not measured" if x is None else f"{x:.2f} points"

    lines = [
        f"serving cost on load {r['load']}: {r['arm']} over {r['reference']}: {_c(r)} of stock's serving compute over "
        f"{_lifetimes(r['lifetimes'])} an arm, {r['users']} users; target {r['target']:g}%: {r['verdict']}",
        f"  the load: {r['what']}; {plan['requests']:,} requests of {plan['prompt_tokens']:,} prompt tokens, "
        f"{plan['repeated_share']:.2f}% of them planned to repeat"
        + (f"; each sampled at temperature {plan['sampled']['temperature']:g} and top-p {plan['sampled']['top_p']:g}"
           + _seeds(r) if plan.get("sampled") else "; each greedy")
        + f"; {TO_LENGTH}",
        *past_end_text(r),
        f"  the cost is the mean of the lifetimes' costs, a lifetime's being {r['arm']}'s seconds for the passes both arms "
        f"counted in it over {r['reference']}'s, less one, a pass's seconds from its users' first send to the last one's "
        "last answer; "
        + ("one lifetime an arm gives no interval and so no verdict" if r["lifetimes"] == 1 else
           "its interval is Student's t over the lifetimes, and the verdict is its upper end against the target"),
        f"  passes taken as independent: {_c(r['passes'])}, {_reps(r['passes']['reps'])}, the median over them with its "
        "interval; in no verdict",
        *(["  by lifetime: " + ", ".join(f"{_c(x)} ({_reps(x['reps'])})" for x in r["by_lifetime"])
           + (f"; the first {_passes(r['left_out']['passes'])} of each lifetime, its warm-up, served and recorded and "
              f"left out ({r['left_out']['warm'][0]} of {r['reference']}'s, {r['left_out']['warm'][1]} of {r['arm']}'s)"
              if r.get("left_out") else "")]
          if r.get("by_lifetime") else []),
        f"  the spread of the cost (standard deviations): {points(spread['between'])} between lifetimes, "
        f"{points(spread['within'])} between a lifetime's passes",
        *(["  other readings of the same passes, in no verdict: " + ", ".join(
            f"{other[k]:.2f}% by {what}" for k, what in READINGS.items() if other.get(k) is not None)]
          if any(v is not None for v in other.values()) else []),
        "  noise floor: " + ((f"{r['reference']}'s own lifetimes, each second one over the one before it"
                              if n.get("lifetimes") else f"{n['arm']} over {r['reference']}")
                             + f": {_c(n)} over {n['pairs']} pair{'s' if n['pairs'] != 1 else ''} of lifetimes, "
                             f"{_reps(n['reps'])}; beside the cost, not added to its interval" if n else
                             f"not measured: no second lifetime of {r['reference']}'s engine served the load"),
        *([f"  {r['reference']} differs from {r['reference']}: the noise floor's interval leaves out zero, so two "
           "lifetimes of the same engine did not serve at the same rate"] if n and n["differs"] else []),
        *(f"  {rate(r['sides'][k])}" for k in ("reference", "arm")),
        *([] if r["same_work"] else ["  the two sides did not do the same work (a request unserved, another count of "
                                     "output tokens, or their prefix caches served different tokens in a rep): no "
                                     "verdict"]),
        f"  {r['model']} on {r['gpus']} GPUs, vLLM {r['vllm']}; commit {r['commit']}, run {r['run']}; "
        f"{r['reference']}: commit {r['reference_commit']}, run {r['reference_run']}",
    ]
    return "\n".join(lines) + "\n"


def cost_report(arms: list[dict], arm: str = OURS, reference: str = REFERENCE, target: float | None = None) -> dict:
    """The serving cost of ``arm`` over ``reference`` from arms' metrics (`arm_metrics`, one arm's directories joined:
    `joined`), every figure in percent: one cost a load both served by vLLM's client (`serving_cost`), the largest the
    number (a load with no finite cost is the largest). Each load's verdict is against ``target`` (the model's, where
    none is given), its interval widened by the load's noise floor, which is the size of the cost there of another arm
    that starts the reference's engine (`same_engine`: two server lifetimes of one engine). The number's verdict
    follows from them, since it is the largest: it misses where a load misses, meets where every load of `NUMBER` has
    run and meets, and else cannot be told (``not_run`` names the loads that have not run). The model's target is the
    sum of its parts (``target_parts``, `target_parts`), and where ``arm`` leaves off a change one of them counts, or
    has a change on that none of them counts (``not_in_target``, `not_in_target`), every verdict is `NOT_COMPARABLE`
    instead: the cost is not of what the target is of. A target that is given has no parts. Each load's ``within`` is
    its cost within the SLO (`within_slo`), which no target or verdict is of, and its ``saturated`` whether the reference's rungs show it saturated at the load's rung (`peak`), which no verdict
    reads. Beside them: the GPUs an engine serves on, what the arm's boundary commitment did (`commitment`), and every other
    arm's cost at the number's load and rung, over the reference and over the nearest arm above it that ran (its
    parent, its parent's parent...), which for the arms from ``arm`` up to the reference is its step of the whole,
    with its share of it and what the target has for the changes the step turns on (`in_target`). ``commit`` and
    ``run`` are those of the job that served the number's load. The number's loads repeat no prefix (``label``), so it
    is never given without the row on the load that does (``prefix``, `prefix_row`), which comes directly after the
    console's keys: where that load has not run, ``value`` and ``ci95`` are None, each load's own cost is still under
    ``loads``, and the number does not meet its target. The row is given without the number: a job of that load alone
    has it, with no load, no arms' table and no value."""
    by = joined(arms)
    for name in (reference, arm):
        if name not in by:
            raise SystemExit(f"cost: no directory of arm {name} (given: {', '.join(by) or 'none'})")
    ref, ours = by[reference], by[arm]
    per = [c for c in (serving_cost(ref, ours, n) for n in NUMBER if n in bench_loads(ref)) if c]
    row = prefix_row(by, reference, arm)
    if not per and not row:
        raise SystemExit(f"cost: {reference} and {arm} share no load ({', '.join((*NUMBER, PREFIX))}) sent by vLLM's client: serve "
                         f"both with --client {BENCH} and the same --loads")
    model = MODELS[ours["model_id"]]
    parts = target_parts(model, arm) if target is None and model.target is not None else []
    outside = not_in_target(model, arm) if parts else []
    target, basis = (model.target, model.target_basis) if target is None else (target, "given")
    comparable = not any(p["off"] for p in parts) and not outside
    twin = next((x for x in by.values() if x["arm"] != reference and same_engine(x["arm"], reference)), None)
    for c in per:
        c["noise"] = serving_cost(ref, twin, c["load"], c["rung"]) if twin else None
        floor = abs(c["noise"]["value"]) if c["noise"] and c["noise"]["value"] is not None else 0.0
        c["verdict"] = verdict(c, None if target is None else target / 100, floor) if comparable else NOT_COMPARABLE
    worst = max(per, key=lambda c: math.inf if c["value"] is None else c["value"], default=None)
    load, rung, noise = (worst["load"], worst["rung"], worst["noise"]) if worst else (None, None, None)
    not_run = [n for n in NUMBER if n not in {c["load"] for c in per}]
    verdicts = {c["verdict"] for c in per}
    overall = (None if target is None else NOT_COMPARABLE if not comparable else "misses" if "misses" in verdicts else
               "meets" if verdicts == {"meets"} and not not_run and row else "cannot tell")

    def above(name: str) -> str:
        p = ARMS[name].parent if name in ARMS else None
        while p is not None and p not in by:
            p = ARMS[p].parent
        return p or reference

    chain, x = [], arm
    while x != reference:
        chain.append(x)
        x = above(x)
    rows = []
    for name in by if worst else ():
        if name == reference or (twin and name == twin["arm"]):
            continue
        over = above(name)
        step = serving_cost(by[over], by[name], load, rung)
        whole = worst["value"]
        share = (round(step["value"] / whole, 4)
                 if name in chain and step and step["value"] is not None and whole is not None and whole > 0 else None)
        rows.append({"arm": name, **_cost(serving_cost(ref, by[name], load, rung)), "over": over, "step": _cost(step),
                     "share": share, "in_target": in_target(model, name, over) if parts else None})
    slo = {"ttft": ours["slo"]["ttft_s"], "tpot": ours["slo"]["tpot_s"]}

    def within(c: dict) -> dict | None:
        w = c["within"]
        return w and {**_cost(w), "latency_target_ms": {k: round(slo[k] * 1e3) for k in LOADS[c["load"]].slo if k in slo},
                      "reps": w["reps"], "reference": w["reference"], "arm": w["arm"]}

    def job(name: str) -> dict:
        (a, b) = (ours["of"][name], ref["of"][name])
        return {"commit": a["commit"], "run": a["run"], "reference_commit": b["commit"], "reference_run": b["run"]}

    at, given = job(load or PREFIX), bool(row and worst)
    return {"value": _pct(worst["value"]) if given else None, "unit": UNIT, "commit": at["commit"],
            "run": at["run"], "prefix": row and {**row, **job(PREFIX)}, "label": NO_REPEAT,
            "metric": "serving_cost", "id": model.metric, "better": "lower", "target": target,
            "target_basis": basis if target is not None else None, "ci95": _cost(worst)["ci95"] if given else None,
            "verdict": overall,
            "target_parts": parts, "not_in_target": outside, "model": model.served, "arm": arm, "reference": reference,
            "load": load, "not_run": not_run,
            "vllm": ours.get("vllm"), "gpus": tp_size(model), "reference_commit": at["reference_commit"],
            "reference_run": at["reference_run"],
            "noise_floor": {"arm": noise["arm"]["arm"], **_cost(noise)} if noise else None,
            "loads": [{"load": c["load"], "what": LOADS[c["load"]].what, **_cost(c), "verdict": c["verdict"],
                       "noise_floor": _cost(c["noise"])["value"] if c["noise"] else None, "reps": c["reps"],
                       "rung": c["rung"], "saturated": c["saturated"], "peak": c["peak"], "reference": c["reference"],
                       "arm": c["arm"], "within": within(c),
                       **job(c["load"])} for c in per],
            "commitment": commitment(ours), "certifier": ours.get("certifier"), "arms": rows, "no_arm": list(NO_ARM)}


def _c(c: dict | None) -> str:
    if not c or c.get("value") is None:
        return "—"
    return f"{c['value']:.2f}%" + (f" [{c['ci95'][0]:.2f}%, {c['ci95'][1]:.2f}%]" if c.get("ci95") else "")


def _reps(n: int) -> str:
    return f"{n} rep{'s' if n != 1 else ''}"


def _seeds(r: dict) -> str:
    """Which side of a sampled cost was sent a seed (`sends_seed`), in words."""
    ref, arm = r["seeded"]["reference"], r["seeded"]["arm"]
    sent = "under a seed its client sends, one a user"
    if ref == arm:
        return f" {sent}" if arm else " under no seed"
    a, b = (r["arm"], r["reference"]) if arm else (r["reference"], r["arm"])
    return f", {a}'s {sent}, and {b}'s under none"


def _passes(n: int) -> str:
    return "pass" if n == 1 else f"{n} passes"


def _lifetimes(n: int) -> str:
    return f"{n} server lifetime{'s' if n != 1 else ''}"


def _in_target(xs: list[dict] | None) -> str:
    if not xs:
        return "—"
    return "; ".join(f"{', '.join(x['changes'])}: " + ("not in the model" if x["part"] is None else
                                                        f"{'' if x['whole'] else 'part of '}{x['part']}, {x['points']:.2f}%")
                     for x in xs)


def _rates(side: dict) -> str:
    prompt = "" if side["prompt_tok_per_s"] is None else f" and {side['prompt_tok_per_s']:,.1f} prompt"
    return f"{side['arm']} {side['output_tok_per_s']:,.1f} output{prompt} tokens/s"


def _at(c: dict, ref: str) -> str:
    """A load's rung, and whether the reference's rungs show it saturated there."""
    if c["saturated"]:
        return f"at concurrency {c['rung']}, where {ref} is saturated"
    why = (f"it served more tokens a second at concurrency {c['peak']}" if c["peak"] != c["rung"] else
           "it served no rung above")
    return f"at concurrency {c['rung']}, where {ref} is not shown saturated ({why})"


def _share(side: dict) -> str:
    return "not recorded" if side["cached_share"] is None else f"{side['cached_share']:.2f}%"


def prefix_text(row: dict) -> list[str]:
    """The row under the number (`prefix_row`) as text: its cost with what it is of and which figure is the cost, then,
    where its requests are sent as they arrive, what keeping up is and how long after the last arrival each side's
    longest pass ended, then the share planned to repeat beside the share each side's prefix cache served and whether
    each side served every request, then the figures that are not the cost, then the reference with its prefix cache
    off. A figure of a row sent as they arrive is its readings, one a rep."""
    a, b, p, off = row["reference"], row["arm"], row["planned"], row["cache_off"]
    arriving = row["rung"] == ARRIVING

    def fig(c: dict) -> str:
        if "readings" not in c:
            return _c(c)
        return ", ".join("—" if v is None else f"{v:.2f}%" for v in c["readings"]) or "—"

    def served(x: dict) -> str:
        if arriving:
            return ", ".join(f"{s['arm']} {'served' if s['served_all'] else 'did not serve'} all that arrived"
                             for s in (x["reference"], x["arm"]))
        sides = [s["arm"] for s in (x["reference"], x["arm"]) if not s["served_all"]]
        return f"{' and '.join(sides)} did not serve every request" if sides else "every request served"

    def slower(x: dict) -> str:
        return (f"the time to the first token {fig(x['ttft'])} longer, the 99th-percentile gap between tokens "
                f"{fig(x['itl_p99'])} longer")

    def tokens(x: dict) -> str:
        """An arm's figure in tokens a second: the cost, or said not to be one."""
        t = fig(x["tokens"])
        return t if x["cost"] == TOKENS else f"in output tokens a second, which is no cost here, {t}"

    def over(x: dict) -> str:
        return f"{', a reading a rep over' if arriving else ' over'} {_reps(x['reps'])}"

    ratio = f"{a['arm']}'s output tokens a second over {b['arm']}'s, less one"
    rates = f"{a['arm']} {a['output_tok_per_s']:,.1f} output tokens/s, {b['arm']} {b['output_tok_per_s']:,.1f}"
    head = (f"  {row['label']}{', scaled down' if row['scaled_down'] else ''}, {row['what']}; no target, and it judges "
            "nothing: ")
    if not arriving:
        flight = "one request" if row["rung"] == 1 else f"{row['rung']} requests"
        lines = [f"{head}{_c(row)}{over(row)}, {flight} in flight, where the cost is {ratio}: {rates}"]
    elif row["cost"] == TOKENS:
        lines = [f"{head}{fig(row['tokens'])}{over(row)}, as they arrive; {row['why']}, so the row is read as the "
                 f"number is: the cost is {ratio}: {rates}"]
    else:
        lines = [f"{head}{slower(row)}{over(row)}, as they arrive; {row['why']}, so the cost is the two latencies, "
                 f"{b['arm']}'s over {a['arm']}'s, less one: at arrivals {a['arm']} keeps up with, tokens a second "
                 "follows the arrivals and is not a cost"]
    if arriving:
        lines.append(f"    {a['arm']} kept up where it served all that arrived and each of its passes ended within a "
                     f"tenth of the time the requests arrive over after the last one arrived: they arrive over "
                     f"{p['span_s']:,.1f} s, and its longest pass ended {a['tail_s']:,.1f} s after the last arrived, "
                     f"{b['arm']}'s {b['tail_s']:,.1f} s after")
    lines.append(f"    {p['requests']:,} requests, {p['repeated_share']:.2f}% of their prompt tokens planned to repeat; "
                 f"the prefix cache served {_share(a)} ({a['arm']}) and {_share(b)} ({b['arm']}); {served(row)}")
    lines.append(f"    {b['arm']} over {a['arm']}: {slower(row)}" if row["cost"] == TOKENS else
                 f"    {ratio}, which is no cost here: {fig(row['tokens'])}; {rates}")
    if off:
        k = off["arm"]
        lines.append(f"    {k['arm']}, {a['arm']} with its prefix cache off: {tokens(off)}{over(off)}, "
                     f"{k['output_tok_per_s']:,.1f} output tokens/s; {slower(off)}; its prefix cache served {_share(k)}; "
                     f"{served(off)}"
                     + (f"; its longest pass ended {k['tail_s']:,.1f} s after the last arrival" if arriving else ""))
    return lines


def certifier_text(arm: str, k: dict | None) -> list[str]:
    """What the cost of an arm behind the network certifier is of, as text, from its record (`certified_client`,
    `certifier_record`): the schedule's ceiling on its tokens a second, which makes its cost the schedule's and its
    engine's together, and what the certifier's offline check found of its run. Nothing for another arm."""
    if not k:
        return []
    c = k["cells"]
    ids = "one id" if c["slots"] == 1 else f"{c['slots']} ids"
    lines = [f"  {arm} is served through the network certifier, whose schedule lets {ids} leave a connection each "
             f"{c['bucket_ms']} ms, the first {c['delay'] * c['bucket_ms'] / 1000:g} s after its admission: its "
             f"{c['lanes']} lanes carry at most {k['ceiling_tok_per_s']:,.0f} output tokens a second whatever its "
             "engine does, so its cost here is the schedule's and its engine's together, and is not GPU time a token"]
    if "accepted" not in k:
        return [*lines, "    the network certifier's run has no check: the arm stopped before it"]
    return [*lines, f"    the network certifier's run is {'' if k['accepted'] else 'NOT '}accepted by its offline check: "
            f"{k['ended']:,} of {k['connections']:,} connections ended, {k['refused']:,} refused, {k['failed']:,} "
            f"failed, {k['short']:,} answers short of their length"]


def cost_text(r: dict) -> str:
    """`cost_report` as text: the number's line first, naming its load where another has not run, then the row on a
    repeated prefix directly under it (`prefix_text`), then the target's
    parts and, where the arm leaves one off, why its cost is not comparable with the target, each load's cost with
    its rung, whether the reference is shown saturated there and both sides' tokens a second, and, under it, its cost
    within the SLO, the noise floor, what the arm's boundary commitment did, every other arm with what its step turns
    on and the target's part for it, and the changes no arm turns on."""
    by = {c["load"]: c for c in r["loads"]}
    row = r["prefix"]
    if row is None:
        n = f"not given without the row on a repeated prefix under it, and load {PREFIX} has not run"
    elif not by:
        n = "none of its loads has run"
    elif r["value"] is None:
        n = f"no finite cost: a side served no token at concurrency {by[r['load']]['rung']}"
    else:
        n = _c(r)
    target = ("no target" if r["target"] is None else f"target ≤ {r['target']:.2f}%"
              + (f" ({r['target_basis']})" if r["target_basis"] else "") + f"   verdict: {r['verdict']}")
    alone = f", load {r['load']} alone" if r["not_run"] and len(r["loads"]) == 1 else ""
    rule = f"the largest of the costs of its loads ({', '.join(NUMBER)})"
    lines = [f"serving cost, {r['label']}, {r['model']}, {r['arm']} over {r['reference']}{alone}: {n}   {target}",
             *(prefix_text(row) if row else []),
             (f"  not the number yet: it is {rule}, and load {' and load '.join(r['not_run'])} "
              f"{'has' if len(r['not_run']) == 1 else 'have'} not run" if r["not_run"] else
              f"  {rule}, load {r['load']}'s"),
             f"  in {r['unit']}, lower is better: how much more GPU time {r['arm']} needs a token than {r['reference']}, "
             f"an engine on {r['gpus']} GPU{'s' if r['gpus'] != 1 else ''}, both serving the same tokens at the "
             f"concurrency, of those both served, where {r['reference']} serves the most output tokens a second"]
    lines += certifier_text(r["arm"], r.get("certifier"))
    parts = r["target_parts"]
    if parts:
        lines.append("  the target is the sum of its parts: " + ", ".join(
            f"{p['name']} {p['points']:.2f}% ({p['basis']}{'; ' + p['what'] if p['what'] else ''})" for p in parts))
    off, outside = [p for p in parts if p["off"]], r["not_in_target"]
    if off or outside:
        why = []
        if off:
            why.append("leaves off " + " and ".join(
                f"{' and '.join(p['off'])} ({'in ' if len(p['off']) < len(p['changes']) else ''}{p['name']}, "
                f"{p['points']:.2f}%)" for p in off) + ", which the target counts")
        if outside:
            why.append(f"turns on {', '.join(outside[:-1])}{' and ' if len(outside) > 1 else ''}{outside[-1]}, which "
                       "the target has no part for")
        lines.append(f"  {NOT_COMPARABLE}: {r['arm']} {', and '.join(why)}, so no cost of it meets or misses the target")
    for c in r["loads"]:
        a, b = c["reference"], c["arm"]
        each = ("" if a["prompt_tokens_per_request"] is None or a["output_tokens_per_request"] is None else
                f"; {a['prompt_tokens_per_request']:,.0f} prompt and {a['output_tokens_per_request']:,.0f} output "
                "tokens a request")
        lines.append(f"  load {c['load']}, {c['what']}: {_c(c)} over {_reps(c['reps'])}"
                     + (f", {c['verdict']}" if c["verdict"] else "") + f"; {_at(c, r['reference'])}: "
                     f"{_rates(a)}, {_rates(b)}{each}")
        w = c["within"]
        if w:
            x, y = w["reference"], w["arm"]
            bounds = ", ".join(f"{k.upper()} ≤ {v} ms" for k, v in w["latency_target_ms"].items())
            cost = _c(w) if w["value"] is not None else (
                f"no finite cost ({y['arm']} met it for {y['met']} of {y['sent']} requests at its best rung, "
                f"{x['arm']} for {x['met']} of {x['sent']})")
            lines.append(f"    within the latency target ({bounds}), which no target or verdict is of: {cost} over "
                         f"{_reps(w['reps'])}; best goodput {x['goodput_req_per_s']:.4g} req/s at concurrency "
                         f"{x['concurrency']} ({x['arm']}), {y['goodput_req_per_s']:.4g} req/s at concurrency "
                         f"{y['concurrency']} ({y['arm']})")
    f = r["noise_floor"]
    if by:
        lines.append(f"  noise floor ({f['arm']} over {r['reference']}, two server lifetimes of one engine, load "
                     f"{r['load']}): {_c(f)}" if f else
                     f"  noise floor: not measured at load {r['load']} (no second arm of {r['reference']}'s engine "
                     "served it): its verdict reads the interval over reps alone, which no difference between two "
                     "server starts enters")
    k = r["commitment"]
    if k:
        tables = (f"{k['tables']:,} of them with their history tables" if k["tables"] else
                  "none with history tables" + (": the arm asks for them and this tree's boundary commitment builds "
                                                "none, so this cost is without them"
                                                if k["tables_asked"] and k["committed"] and not k["failed"] else ""))
        every = f", a window every {k['window_s']:g} s" if k["window_s"] else ""
        lines.append(f"  {r['arm']}'s boundary commitment{every}: {k['committed']:,} of {k['finished']:,} finished "
                     f"requests committed, {k['failed']:,} failed, {tables}")
    if r["arms"]:
        model = [" what the step turns on, and the target's part for it |", "---|"] if parts else ["", ""]
        lines += ["", f"| arm | cost over {r['reference']} | over | cost over that arm | share of the cost |{model[0]}",
                  f"|---|---:|---|---:|---:|{model[1]}"]
        lines += [f"| {x['arm']} | {_c(x)} | {x['over']} | {_c(x['step'])} | "
                  f"{'—' if x['share'] is None else format(x['share'], '.0%')} |"
                  + (f" {_in_target(x['in_target'])} |" if parts else "") for x in r["arms"]]
    def job(c: dict) -> str:
        return (f"commit {c['commit']}, run {c['run']}"
                + ("" if (c["reference_commit"], c["reference_run"]) == (c["commit"], c["run"]) else
                   f"; {r['reference']}: commit {c['reference_commit']}, run {c['reference_run']}"))

    served = [*r["loads"], *([row] if row else [])]
    jobs = list(dict.fromkeys(job(c) for c in served))
    lines += ["", "changes with no arm, which no cost here counts: " + "; ".join(r["no_arm"]) + ".",
              *([f"{jobs[0]}, vLLM {r['vllm']}"] if len(jobs) == 1 else
                [*(f"load {c['load']}: {job(c)}" for c in served), f"vLLM {r['vllm']}"])]
    return "\n".join(lines) + "\n"


def result_of(r: dict) -> tuple[bool, str, dict, list[dict]]:
    """A cost report (`cost_report`) as a run's result.json holds it (`write_result`): whether it stands as a
    measurement (each load that ran has a finite cost, the row's figures that are its cost are finite too, and the
    arm's boundary commitment committed every finished request), its first line, what ran, and its measurements. Each
    load's cost is a measurement under the load's name, with its rung and whether the reference is shown saturated
    there.
    The number is one under the model's metric id (`Model.metric`), the entry a reader of results takes it from, with
    its target, what the target rests on, the target's parts with the changes the arm left off, the arm's changes no
    part counts, its verdict (one of `verdict`'s three, or `NOT_COMPARABLE`) and each load's cost beside it; it names
    the loads as its inputs. Where a load has not run there is no such entry: one load's cost is not the number. The
    row on a repeated prefix is a measurement under its load's name, with no target and no verdict, whose ``value``
    is its cost in tokens a second with one request in flight, and None where its requests are sent as they arrive:
    its figures are then their readings (``tokens``, ``ttft``, ``itl_p99``), ``cost`` names the ones that are its
    cost and ``why`` says why (`prefix_row`); the
    number's entry carries it (``prefix``) with the number's label; where that load has not run there is no entry
    under the metric's id either. A sessions load's report (`session_cost`) is one measurement under its load's
    name, which stands where it has a value and both sides did the same work."""
    if r["metric"] == SESSION_COST:
        entry = {"name": r["load"], **{key: r[key] for key in ("value", "unit", "target", "target_basis", "better", "commit",
                                                              "run", "ci95", "verdict", "what", "reps", "users",
                                                              "lifetimes")}}
        what = {"benchmark": "serve_tax", "model": r["model"], "arm": r["arm"], "reference": r["reference"],
                "loads": [r["load"]], "vllm": r["vllm"]}
        return r["value"] is not None and r["same_work"], session_text(r).splitlines()[0], what, [entry]
    k, row = r["commitment"], r["prefix"]
    figures = [] if not row else [row["tokens"]] if row["cost"] == TOKENS else [row["ttft"], row["itl_p99"]]
    ok = (all(c["value"] is not None for c in r["loads"])
          and all(c.get("readings", [c["value"]]) and None not in c.get("readings", [c["value"]]) for c in figures)
          and not (k and (k["failed"] or k["committed"] < k["finished"])))
    about = {"unit": r["unit"], "target": r["target"], "target_basis": r["target_basis"], "better": r["better"]}

    def entry(name: str, c: dict, **more: object) -> dict:
        return {"name": name, "value": c["value"], **about, "commit": c["commit"], "run": c["run"], "ci95": c["ci95"],
                "verdict": c["verdict"], **more}

    out = [entry(c["load"], c, what=c["what"], reps=c["reps"], rung=c["rung"], saturated=c["saturated"])
           for c in r["loads"]]
    under = row and {key: row[key] for key in ("load", "label", "scaled_down", "value", "ci95", "cost", "kept_up", "why",
                                               "tokens", "ttft", "itl_p99", "reps", "rung", "commit", "run")}
    if row:
        out.append({"name": row["load"], "unit": r["unit"], "target": None, "target_basis": None, "better": r["better"],
                    "verdict": None, "what": row["what"], **under})
    if r["id"] and not r["not_run"] and row:
        beside = [{key: c[key] for key in ("load", "value", "ci95", "verdict", "reps", "rung", "saturated", "commit", "run")}
                  for c in r["loads"]]
        out.append(entry(r["id"], r, label=r["label"], prefix=under, load=r["load"], loads=beside,
                         inputs=[c["load"] for c in r["loads"]], target_parts=r["target_parts"],
                         not_in_target=r["not_in_target"]))
    what = {"benchmark": "serve_tax", "model": r["model"], "arm": r["arm"], "reference": r["reference"],
            "loads": [*(c["load"] for c in r["loads"]), *([row["load"]] if row else [])], "vllm": r["vllm"]}
    return ok, cost_text(r).splitlines()[0], what, out


def write_result(r: dict, out: Path) -> None:
    """The run's result.json (`result_of`), written by the kit's writer beside the report in ``out``."""
    run_dir, run_id = SHARED.RESEARCH_RUN_DIR, SHARED.RESEARCH_RUN_ID
    if not run_dir or not run_id:
        raise SystemExit("cost --result writes the run's result.json: run it under research run")
    ok, detail, what, measurements = result_of(r)
    kit()[1].write_result(Path(run_dir), run_id, out, "serving_cost.json", ok, detail, what, measurements)


def _f(s: dict, scale: float = 1.0, digits: int = 4) -> str:
    if not s or not s.get("n"):
        return "—"
    return f"{s['mean'] * scale:.{digits}g}"


def _tax(t: dict | None) -> str:
    if not t:
        return "—"
    s = f"{t['value']:+.1%}"
    return s + (f" [{t['ci95'][0]:+.1%}, {t['ci95'][1]:+.1%}]" if "ci95" in t else "")


def markdown(t: dict) -> str:
    slo = t["slo"] or {}
    lines = [f"SLO: TTFT ≤ {slo.get('ttft_s')} s and time per output token ≤ {slo.get('tpot_s')} s. Tax: the arm's compute "
             "per token over the reference's at the same load, less one, from output tokens a second (goodput's beside it). "
             "Unserved: requests sent and not served over the load's reps; a tax beside one isn't over the same requests.",
             "Steps: the engine's steps over the load, with the tokens each scheduled and the rows each sampled.",
             "Busy GPU-s: the GPUs' NVML utilization integrated over the load, summed over GPUs, and per thousand output "
             "tokens. Edges: the replay-block edges in the load, and the engine's stall at each (the drain plus the pause) "
             "at its median and longest.",
             "A pass of vLLM's client (a load at a rung, load@rung) holds its concurrency in place of a rate, or sends "
             "each request of a planned file as it arrives (load@0), and its figures are the client's own.",
             *(["An arm served in server lifetimes has their passes as its reps, less the passes each lifetime leaves "
                "out (its first ones, its warm-up), which are served and recorded and in no figure here: "
                + "; ".join(f"{r['arm']} {r['load']}: {_lifetimes(r['lifetimes'] or 1)}, {r['left_out']} left out"
                            for r in t["rows"] if r.get("lifetimes") or r.get("left_out")) + "."]
               if any(r.get("lifetimes") or r.get("left_out") for r in t["rows"]) else []),
             "", "| arm | load | rate (req/s) | unserved | out tok/s | req/s | goodput req/s | SLO met | TTFT p50 / p99 (ms) | "
             "ITL p50 / p99 (ms) | steps | tokens / rows a step | busy GPU-s (per 1k tok) | edges (stall p50 / max, s) | "
             "tax vs compiled (tok/s) | tax vs compiled (goodput) | tax vs parent (tok/s) | run |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in t["rows"]:
        vc, vp = r.get("tax_vs_compiled") or {}, r.get("tax_vs_parent") or {}
        edges = (f"{_f(r.get('edges'), 1, 3)} ({_f(r.get('edge_stall_p50_s'), 1, 3)} / {_f(r.get('edge_stall_max_s'), 1, 3)})"
                 if (r.get("edges") or {}).get("n") else "—")
        at = f"{r['rate']:.3g}" if r.get("rate") is not None else at_rung(r.get("concurrency")).removeprefix("at ")
        lines.append(f"| {r['arm']} | {r['load']} | {at} | {r['unserved']} of {r['sent']} | "
                     f"{_f(r['output_tok_per_s'])} | {_f(r['req_per_s'])} | "
                     f"{_f(r['goodput_req_per_s'])} | {_f(r['slo_attained'], 100, 3)}% | "
                     f"{_f(r['ttft_p50_s'], 1e3)} / {_f(r['ttft_p99_s'], 1e3)} | {_f(r['itl_p50_s'], 1e3)} / "
                     f"{_f(r['itl_p99_s'], 1e3)} | {_f(r.get('steps'), 1, 5)} | "
                     f"{_f(r.get('tokens_per_step'), 1, 4)} / {_f(r.get('rows_per_step'), 1, 3)} | "
                     f"{_f(r.get('busy_gpu_s'), 1, 4)} ({_f(r.get('busy_gpu_s_per_ktok'), 1, 3)}) | {edges} | "
                     f"{_tax(vc.get('throughput'))} | {_tax(vc.get('goodput'))} | "
                     f"{_tax(vp.get('throughput'))}{' (' + vp['of'] + ')' if vp else ''} | {r['run_id']} |")
    es = {name: a["edges"] for name, a in t.get("arms", {}).items() if a.get("edges")}
    if es:
        lines += ["", "Each arm's edges outside its warm-up: the pause from the drain's end to the next step, its parts' "
                  "medians (at TP > 1 the longest rank's), the bytes hashed and copied at an edge, and whether every rank's KV "
                  "root agreed.", "",
                  "| arm | edges | pause p50 / p99 / max (s) | drain p50 (s) | state p50 (s) | KV p50 (s) | "
                  "hash kernels p50 (ms) | copy p50 (ms) | runner rows p50 (s) | GB hashed / copied | roots equal |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for name, e in es.items():
            lines.append(f"| {name} | {e['edges']} | {_n(e['pause_p50_s'])} / {_n(e['pause_p99_s'])} / {_n(e['pause_max_s'])} | "
                         f"{_n(e['drain_p50_s'])} | {_n(e['state_p50_s'])} | {_n(e['kv_p50_s'])} | "
                         f"{_n(e['hash_kernel_ms_p50'])} | {_n(e['copy_ms_p50'])} | {_n(e['runner_s_p50'])} | "
                         f"{_n(_gb(e['kv_bytes']), 4)} / {_n(_gb(e['copied_bytes']), 4)} | {'yes' if e['roots_equal'] else 'NO'} |")
    return "\n".join(lines) + "\n"


def _n(x: float | None, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}g}"


def _gb(x: float | None) -> float | None:
    return None if x is None else x / 1e9


#: what an arm spends beside its loads' wall time: each load's prefix-cache reset and /metrics scrapes, and the
#: server's stop
LOAD_OVERHEAD_S, STOP_S = 5.0, 60.0


def parse_stop_at(text: str | None) -> float | None:
    """``--stop-at``: a time with its zone, as ``2030-01-01T00:00:00Z``, in seconds since the epoch."""
    if text is None:
        return None
    no = SystemExit(f"--stop-at {text!r}: a time with its zone, as 2030-01-01T00:00:00Z (run.sh: STOP_AT)")
    try:
        at = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise no from None
    if at.tzinfo is None:
        raise no
    return at.timestamp()


def stop_before(served: Sequence[dict], name: str, left_s: float | None, margin: float = 1.1) -> float | None:
    """Whether an arm with ``left_s`` seconds to the time it stops by (``--stop-at``; None: it has none) starts no
    further pass, ``name`` being its next: the seconds that pass and the server's stop need where ``left_s`` is fewer,
    else None. A pass needs its own longest wall time among the passes ``served`` (the arm's ``meta["loads"]``), or
    where it has none yet the longest of any pass, ``margin`` times as long. An arm's first pass has nothing to go by
    and starts whenever the stop alone fits. The arm then stops between two passes and keeps every pass it served,
    where a signal at the time itself would lose the pass in flight."""
    if left_s is None:
        return None
    walls = [x["wall_s"] for x in served if x["load"] == name] or [x["wall_s"] for x in served]
    need = ((max(walls) + LOAD_OVERHEAD_S) * margin if walls else 0.0) + STOP_S
    return need if need > left_s else None


def reps_that_fit(meta: dict, warm_wall_s: float, budget_s: float, reps: int, margin: float = 1.1) -> int:
    """How many of ``reps`` reps an arm like the one ``meta`` records fits in ``budget_s`` seconds: its startup, its
    warm-up's ``warm_wall_s`` and its longest rep's loads, each ``margin`` times as long; ``reps`` when ``meta`` has no
    rep to go by, else 0 to ``reps``."""
    per_rep: dict[int, float] = {}
    for x in meta.get("loads") or ():
        per_rep[x["rep"]] = per_rep.get(x["rep"], 0.0) + x["wall_s"] + LOAD_OVERHEAD_S
    if not per_rep or meta.get("startup_s") is None:
        return reps
    fixed = (meta["startup_s"] + warm_wall_s + STOP_S) * margin
    return max(0, min(reps, int((budget_s - fixed) // (max(per_rep.values()) * margin))))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True, choices=sorted(ARMS))
    r.add_argument("--model-id", required=True, choices=sorted(MODELS))
    r.add_argument("--checkpoint")
    r.add_argument("--client", default=CORPUS, choices=(CORPUS, BENCH),
                   help="who sends the loads: the trace corpus's client (--rates of the --c4 plan) or vLLM's own, "
                        "`vllm bench serve` (--loads)")
    r.add_argument("--c4")
    r.add_argument("--rates", help="NAME=REQ_PER_S,... in the order each rep serves them")
    r.add_argument("--loads", metavar="NAME[=RUNG+RUNG...],...",
                   help=f"vLLM's client: the loads ({', '.join(sorted(LOADS))}) in the order each rep serves them, each at "
                        "its rungs (concurrencies) or the ones given")
    r.add_argument("--dataset", action="append", metavar="NAME=FILE", help="vLLM's client: a load's dataset file")
    r.add_argument("--pass-prompts", metavar="NAME=N|Kx,...",
                   help="vLLM's client: the prompts a pass of a load sends in place of the load's own: N at every rung, "
                        "or Kx, K for each request its rung holds in flight (parse_pass_prompts)")
    r.add_argument("--sampled", action="store_true",
                   help="vLLM's client: each request sampled at the recipe's temperature and top-p (SAMPLED) in place "
                        "of greedy, naming its user; an arm that keeps the record is sent that user's seed too "
                        "(user_seed, sends_seed), another in each server lifetime (--lifetime), and a stock arm none")
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--lifetime", type=int, metavar="K",
                   help="this server lifetime's number among the arm's in the job, from 1 (run.sh's LIFETIMES): a cost "
                        "pairs lifetime K's passes of its two arms (lifetimes)")
    r.add_argument("--leave-out", type=int, default=0, metavar="N",
                   help="the first N reps of each load are the lifetime's warm-up: served and recorded like the "
                        "rest, and left out of every figure (lifetimes)")
    r.add_argument("--stop-at", metavar="TIME",
                   help="the time the arm stops by, with its zone (2030-01-01T00:00:00Z): it starts no pass that it "
                        "can't expect to finish by then, and keeps the passes it served (stop_before)")
    r.add_argument("--span-s", type=float, default=90.0, help="each load's arrivals span")
    r.add_argument("--requests", type=int, help="each load's request count instead of --span-s's")
    r.add_argument("--warm-s", type=float, default=20.0,
                   help="a warm-up load at the lowest rate first, or a short pass of vLLM's client (0: none)")
    r.add_argument("--commit-wait-s", type=float, default=900.0,
                   help="the boundary-rows arm: how long the server is held after its loads for its commits (await_commits)")
    r.add_argument("--drain-s", type=float, default=600.0,
                   help="the trace corpus's client: how long after a load's last arrival it waits for the answers still "
                        "out; a request unanswered by then is cut and counted unserved")
    r.add_argument("--startup-s", type=float, default=1800.0)
    r.add_argument("--window-s", type=float, default=600.0, help="the Commit period the tokens arm closes at")
    r.add_argument("--kv-window-s", type=float, default=600.0, help="the window edge the kv arm hashes the KV cache at")
    r.add_argument("--boundary-rows", type=json.loads, metavar="JSON",
                   help="a boundary commitment's object in place of the arm's own: check, per_prompt, per_output, "
                        "drained, inline, verdicts, tables, stop (the serve's stop rule: ids, max_model_len, outputs)")
    r.add_argument("--private", metavar="DIR",
                   help="where the arm keeps what never leaves the node (a boundary commitment's classes, history tables' "
                        "rows and salts, the top-k's counters), outside every run's directory (under vLLM's client "
                        "by default a fresh directory, removed when the arm stops; the corpus client needs it)")
    r.add_argument("--plan", help="the plan's file, shared by a job's arms (default: OUT/plan.jsonl)")
    r.add_argument("--docs", type=int, default=20000, help="the shard's documents a plan draws from")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--tick-from", metavar="STEP_TRACE|RECORD_DIR", help="τ for the tick arms (ignored by the others)")
    r.add_argument("--tick-ms", metavar="SAT,FREE")
    r.add_argument("--side", choices=(SERVER, CLIENT),
                   help="an arm behind the network certifier: the side this command serves, the engine and the id "
                        "front on the node, or the certifier, its shim and vLLM's client on the certifier's host "
                        "(ignored by the other arms, which one command serves)")
    r.add_argument("--front", type=int, default=FRONT_PORT, metavar="PORT",
                   help="such an arm: the loopback port the id front listens on, on the node, and where the "
                        "client's side reaches it")
    r.add_argument("--cells", metavar="NAME=N,...",
                   help="such an arm: the network certifier's schedule over CELLS, the same on both sides "
                        "(certifier_cells)")
    r.add_argument("--certifier-windows", type=int, metavar="N",
                   help="such an arm's client side: the windows the network certifier runs, fixed when it starts")
    r.add_argument("--num-gpu-blocks", type=int, help="the KV pool's blocks (vLLM's --num-gpu-blocks-override)")
    r.add_argument("--published-frontend", choices=FRONT_ENDS, help="the front end of an arm that starts the server as "
                   "the model's recipe does (default: python, which the other arms have)")
    r.add_argument("--published-args", metavar="FLAG=VALUE,...", help="flags of vLLM's own added to that arm's line, each "
                   "--FLAG=VALUE, given as --published-args=--FLAG=VALUE,... (no other arm's line takes them)")
    r.add_argument("--max-model-len", type=context_arg, help="the engines' context limit, a number of tokens or auto "
                   "(default: BASE_ARGS's)")
    r.add_argument("--edges-s", metavar="S,S,...", help="the restore arms' edges, seconds after the first load's first request")
    r.add_argument("--edges-every-s", type=float, metavar="S", help="an edge each S seconds of each load's serving instead")
    r.add_argument("--save-kv", metavar="K,...", help="the edges whose KV bytes and state a restore arm keeps "
                   "(default: 0 with --edges-s, none with --edges-every-s)")
    r.add_argument("--kv-dir", help="where a restore arm keeps them, in a directory named for the arm (default: its server/edges)")
    r.add_argument("--kv-copy", default="all", choices=KV_COPY,
                   help="the ranks that copy their KV pool to host memory at an edge (first: rank 0, for ranks that hold the same bytes)")
    r.add_argument("--out", required=True)
    pl = sub.add_parser("plan", help="make the plan a job's arms share, before any GPU is held")
    for flag in ("--model-id", "--checkpoint", "--c4", "--rates", "--out"):
        pl.add_argument(flag, required=flag != "--checkpoint")
    pl.add_argument("--span-s", type=float, default=90.0)
    pl.add_argument("--requests", type=int)
    pl.add_argument("--docs", type=int, default=20000)
    pl.add_argument("--seed", type=int, default=0)
    for name in ("metrics", "table"):
        p = sub.add_parser(name)
        p.add_argument("dirs", nargs="+", type=Path)
        p.add_argument("--slo-ttft-s", type=float)
        p.add_argument("--slo-tpot-s", type=float)
        if name == "table":
            p.add_argument("--out", type=Path)
    ps = sub.add_parser("passes", help="the passes vLLM's client serves for --loads, their dataset files and the arms "
                                       "checked, before any GPU is held")
    ps.add_argument("--loads", required=True)
    ps.add_argument("--dataset", action="append")
    ps.add_argument("--pass-prompts", metavar="NAME=N|Kx,...")
    ps.add_argument("--fetch", action="store_true", help="download a pinned dataset file the node's cache doesn't hold")
    ps.add_argument("--arms", default="", help="the arms that will serve them, each checked against --model-id")
    ps.add_argument("--model-id", choices=sorted(MODELS))
    ps.add_argument("--boundary-rows", type=json.loads, metavar="JSON")
    ps.add_argument("--max-model-len", type=context_arg)
    ds = sub.add_parser("dataset", help="a load's pinned dataset file on this node, its SHA-512 checked: its path")
    ds.add_argument("load", choices=sorted(n for n, x in LOADS.items() if x.file))
    ds.add_argument("--fetch", action="store_true", help="download it where the node's cache doesn't hold it")
    co = sub.add_parser("cost", help="the serving cost of --arm over --reference from arms served by vLLM's client")
    co.add_argument("dirs", nargs="+", type=Path)
    co.add_argument("--arm", default=OURS)
    co.add_argument("--reference", help="the arm it is costed over (default: the stock arm --arm is made from, stock_of)")
    co.add_argument("--target", type=float, metavar="PERCENT",
                    help="the most it may cost (default: the model's; on a sessions load, SESSION_TARGET)")
    co.add_argument("--json", action="store_true", help="print the JSON object alone")
    co.add_argument("--out", type=Path, help="write serving_cost.json and serving_cost.txt there")
    co.add_argument("--result", action="store_true",
                    help="with --out, under research run: write the run's result.json too, the number as a measurement "
                         "under the model's metric id")
    k = sub.add_parser("tick")
    k.add_argument("source", metavar="STEP_TRACE|RECORD_DIR")
    k.add_argument("--load", default="saturated")
    e = sub.add_parser("edges", help="each edge of a restore arm as its client saw it, beside other arms at the same moments")
    e.add_argument("arm", type=Path)
    e.add_argument("others", nargs="*", type=Path)
    e.add_argument("--load", default="medium")
    e.add_argument("--rep", type=int, default=1)
    f = sub.add_parser("fit", help="how many reps of an arm like ARM_DIR's fit in the seconds left (reps_that_fit)")
    f.add_argument("arm", type=Path, metavar="ARM_DIR")
    f.add_argument("--budget-s", type=float, required=True)
    f.add_argument("--reps", type=int, required=True)
    c = sub.add_parser("checkpoint", help="the model's pinned snapshot under HF_HOME")
    c.add_argument("--model-id", required=True, choices=sorted(MODELS))
    a = ap.parse_args(argv)
    if a.cmd == "edges":
        windows = edge_windows(a.arm, a.load, a.rep)
        for d in (a.arm, *a.others):
            f = d / a.load / f"rep-{a.rep}" / "client.jsonl"
            recs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
            for k_, row in enumerate(edge_cost(recs, windows)):
                print(json.dumps({"arm": d.name, "edge": k_, **row}))
        return 0
    if a.cmd == "run":
        return run(a)
    if a.cmd == "fit":
        meta = json.loads((a.arm / "meta.json").read_text())
        warm = a.arm / "warm" / "meta.json"
        print(reps_that_fit(meta, json.loads(warm.read_text())["wall_s"] if warm.exists() else 0.0, a.budget_s, a.reps))
        return 0
    if a.cmd == "checkpoint":
        print(MODELS[a.model_id].checkpoint())
        return 0
    if a.cmd == "plan":
        ck = Path(a.checkpoint) if a.checkpoint else MODELS[a.model_id].checkpoint()
        n = plan_need(parse_rates(a.rates), a.span_s, a.requests)
        print(f"[serve-tax] plan: {len(plan_of(Path(a.out), Path(a.c4), ck, n, a.docs, a.seed))} requests in {a.out}")
        return 0
    if a.cmd == "tick":
        print(json.dumps(tick_args(a.source, None, a.load)))
        return 0
    if a.cmd == "dataset":
        print(parse_datasets(None, {a.load: ()}, a.fetch)[a.load])
        return 0
    if a.cmd == "passes":
        loads = parse_loads(a.loads)
        for name in a.arms.split():
            arm = ARMS.get(name)
            if arm is None:
                raise SystemExit(f"--arms: {name!r} is not an arm ({', '.join(ARMS)})")
            if arm.certifier:
                certified_loads(name, a.loads)
            elif arm.tick or arm.sampling != "greedy" or arm.logprobs:
                raise SystemExit(f"--arms: {name} needs the corpus client, not vLLM's")
            if ("boundary_rows" in arm.plugin and not (a.boundary_rows or {}).get("check")
                    and not (a.model_id and MODELS[a.model_id].check)):
                raise SystemExit(f"--arms: {name}'s boundary commitment needs a check template, and "
                                 f"{a.model_id or 'no model'} names none (Model.check): BOUNDARY_ROWS, a JSON object "
                                 "with check")
            if arm.published:
                context_of(loads, published_context(name, a.model_id, a.max_model_len))
        context_of(loads, a.max_model_len)
        files = parse_datasets(a.dataset, loads, a.fetch)
        sizes = parse_pass_prompts(a.pass_prompts, loads)
        loads, lines = passes_of(loads, files, sizes)
        for n, rungs in loads.items():
            plan = planned(lines[n]) if n in lines else None
            arriving = rungs == (ARRIVING,)
            digest = (None if n not in files else f"SHA-512 {LOADS[n].file.sha512[:16]}" if LOADS[n].file else
                      f"SHA-256 {file_digest(Path(files[n]))[:16]}")
            each = [pass_prompts(LOADS[n], c, lines.get(n), prompts=sizes.get(n)) for c in rungs]
            print(f"[serve-tax] load {n}: {GIVEN if arriving else LOADS[n].what}; "
                  f"{', '.join(f'{x:,}' for x in (each if len(set(each)) > 1 else each[:1]))} "
                  f"{'turns a user' if LOADS[n].sessions else 'prompts'} a pass "
                  f"{at_rung(ARRIVING) if arriving else ('of users ' if LOADS[n].sessions else 'at concurrency ') + ', '.join(map(str, rungs))}"
                  + (f" (--pass-prompts: the load's own are {LOADS[n].prompts:,})" if n in sizes else "")
                  + (f", from {files[n]} ({digest}...)" if digest else "")
                  + (f", {plan['repeated_tokens']:,} of their {plan['prompt_tokens']:,} prompt tokens repeated"
                     if plan else "")
                  + (f", arriving over {plan['span_s']:,.1f} s" if arriving else ""))
        return 0
    slo = (a.slo_ttft_s, a.slo_tpot_s) if a.cmd != "cost" and a.slo_ttft_s and a.slo_tpot_s else None
    arms = [json.loads(d.read_text()) if d.is_file() else arm_metrics(d, slo) for d in a.dirs]
    if a.cmd == "cost":
        reference = a.reference or stock_of(a.arm)
        rep = session_cost(arms, a.arm, reference, a.target)
        text = session_text(rep) if rep else None
        # each further sessions load the reference served has a report of its own, printed after the first's
        more = [session_cost(arms, a.arm, reference, a.target, n) for n in session_loads(arms, reference)[1:]] if rep else []
        if rep is None:
            rep = cost_report(arms, a.arm, reference, a.target)
            text = cost_text(rep)
        if a.result and not a.out:
            raise SystemExit("cost --result: with --out DIR, where the report it cites is written")
        if a.out:
            a.out.mkdir(parents=True, exist_ok=True)
            (a.out / "serving_cost.json").write_text(json.dumps(rep, indent=1))
            (a.out / "serving_cost.txt").write_text(text)
            for r in more:
                (a.out / f"serving_cost.{r['load']}.json").write_text(json.dumps(r, indent=1))
                (a.out / f"serving_cost.{r['load']}.txt").write_text(session_text(r))
        if a.result:
            write_result(rep, a.out)
        print(json.dumps(rep) if a.json else text + "".join(map(session_text, more)), end="\n" if a.json else "")
        return 0
    if a.cmd == "metrics":
        print("\n".join(json.dumps(x) for x in arms))
        return 0
    t = table(lifetimes(arms))
    md = markdown(t)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "table.json").write_text(json.dumps(t, indent=1))
        (a.out / "table.md").write_text(md)
    print(md, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
