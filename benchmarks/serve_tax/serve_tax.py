"""The serve tax: what each change to the production server costs on every token, against stock compiled vLLM at the
integration's pin, on the same node, workload and SLO.

    serve_tax.py plan --model-id ID --c4 C4.json.gz --rates NAME=R,... [--span-s 90] [--requests N] --out PLAN.jsonl
    serve_tax.py run --arm ARM --model-id ID --c4 C4.json.gz --rates NAME=R,... [--checkpoint DIR] [--reps 3] [--span-s 90]
                     [--requests N] [--plan PLAN.jsonl] [--window-s S] [--kv-window-s S]
                     [--tick-ms SAT,FREE | --tick-from STEP_TRACE|RECORD_ARM_DIR] --out DIR
    serve_tax.py run --arm ARM --model-id ID --client vllm-bench --loads NAME[=RUNG+RUNG...],... [--dataset NAME=FILE]
                     [--checkpoint DIR] [--reps 3] [--private DIR] --out DIR
    serve_tax.py passes --loads NAME[=RUNG+RUNG...],... [--dataset NAME=FILE] [--fetch] [--arms "ARM ..." --model-id ID]
    serve_tax.py dataset LOAD [--fetch]
    serve_tax.py metrics DIR [--slo-ttft-s S --slo-tpot-s S]
    serve_tax.py table DIR... --out DIR [--slo-ttft-s S --slo-tpot-s S]
    serve_tax.py cost DIR... [--arm ARM] [--reference compiled] [--target PERCENT] [--json] [--out DIR [--result]]
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
  (``check``, ``per_prompt``, ``per_output``, ``drained``, ``inline``, ``tables`` for each span's latent history tables
  (`servetax_plugin`) or ``stop``, the serve's stop rule, which has no tables), at ``--window-s``; the server is held
  after its loads until its windows' commits account for every finished request (`await_commits`).
- ``canonical-topk``: ``compiled`` with the canonical DSA top-k (`benchmarks/dsa_topk`'s vLLM plugin, `TOPK`), which
  rewrites only the indexer's rows of more than 2,048 keys, so it acts on long contexts alone;
- ``two-stage``: the two-stage serve's launch line, everything its engine turns on over stock: ``boundary-rows``, the
  canonical DSA top-k and one attention layer's latent history tables in the boundary commitment (`TABLES`), with
  prefix caching off: a table holds its request's rows from position 0, and a request served from the cache computes
  none before its hit's end (`BoundaryWindows.install` refuses the pair). The line is this arm's data and is stated
  nowhere else: a job that serves the two-stage audit serves this arm.
- ``two-stage-tick``: that launch line admitting on the certified tick, as ``tick`` does: the arm whose schedule is a
  function of the certified arrivals, so its windows' advice holds no decision. It is made of ``two-stage``'s data,
  and like every tick arm the corpus's client serves it, standing in for the certifier.

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
decoding to its length (``--ignore-eos``). Each pass's DIR/<load>@<rung>/rep-<r>/result.json is the result the client
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
}
ARMS["two-stage-tick"] = replace(ARMS["two-stage"], parent="two-stage", what="+ admission on the certified tick", tick=True)
#: the changes of ours an arm's engine can have on, each read from the arm's data: what a serving cost is made of, and
#: what a part of a model's target (`Part`) is of
CHANGES: dict[str, Callable[[Arm], bool]] = {
    "the step record": lambda a: "record" in a.plugin,
    "prefix caching off": lambda a: "--no-enable-prefix-caching" in a.args,
    "the two-stage boundary commitment": lambda a: "boundary_rows" in a.plugin,
    "the history tables": lambda a: bool(a.rows.get("tables")),
    "the canonical top-k": lambda a: a.env.get("VERITY_DSA_TOPK") == "canonical",
    "admission on the certified tick": lambda a: a.tick,
    "the edge snapshot": lambda a: a.edges,
}


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
    table ``--dataset NAME=FILE`` gives, `read_plan`); and whether its cost enters the number (`cost_report`), which a
    load that repeats a prefix never does."""
    what: str
    dataset: tuple[str, ...]
    prompts: int
    rungs: tuple[int, ...]
    context: int
    file: Dataset | None = None
    slo: tuple[str, ...] = ("ttft", "tpot")
    plan: tuple[Planned, ...] = ()
    number: bool = True


#: two prompts near a 131,072-token context limit, each sent once and then again with another ending: the second of
#: each pair shares all but its last tokens with the first, in whole blocks. A row of these four alone is scaled down
#: (`prefix_row`): half of their prompt tokens repeat, which is this load's share and no serve's, where such pairs
#: arrive among requests that repeat nothing
PAIRS = (Planned(0.0, 130808, 256), Planned(0.0, 130807, 256, after=0, blocks=2043),
         Planned(0.0, 129016, 256), Planned(0.0, 129016, 256, after=2, blocks=2015))


#: chat: the load of vLLM's standing serving tests, ShareGPT's first turns, each decoding its recorded answer's length
#: (the client keeps prompts of at most 1,024 tokens, 2,048 with the output). long: long context, the client's synthetic
#: dataset at one long length until a plan of recorded long requests replaces it; its first token takes seconds on any
#: engine, so its goodput counts the time per output token alone. prefix: the load that repeats a prefix, which chat
#: and long do not; its cost is the row under the number and never enters it. One request in flight, so each second
#: request starts once the prompt it repeats is computed; from a table of its own (``--dataset prefix=FILE``), its
#: requests leave as they arrive instead (`ARRIVING`).
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
                 "(vLLM's timed_trace dataset)",
                 ("--dataset-name", "timed_trace", "--timed-trace-chunk-hash-size", str(BLOCK)),
                 len(PAIRS), (1,), 131072, slo=("tpot",), plan=PAIRS, number=False),
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


def passes_of(loads: dict[str, tuple[int, ...]],
              datasets: dict[str, str]) -> tuple[dict[str, tuple[int, ...]], dict[str, list[dict]]]:
    """``loads`` with the rungs their passes are sent at, and each planned load's lines (`trace_lines`): a planned
    load whose table is given (``datasets``, `parse_datasets`) is sent as its requests arrive (`ARRIVING`,
    `read_plan`); one with none, at its own rung from its own plan."""
    lines = {n: trace_lines(read_plan(n, Path(datasets[n])) if n in datasets else LOADS[n].plan)
             for n in loads if LOADS[n].plan}
    return {n: (ARRIVING,) if n in lines and n in datasets else rungs for n, rungs in loads.items()}, lines


def pass_prompts(load: Load, rung: int, lines: Sequence[dict] | None, warm_s: float | None = None,
                 prompts: int | str | None = None) -> int:
    """The prompts a pass of ``load`` at ``rung`` sends: ``prompts`` (``--pass-prompts``, `parse_pass_prompts`: a
    number, the same at every rung, or ``Kx``, K for each request the rung holds in flight and the load's own at
    least) or the load's, or every line of its file where they leave as they arrive (`ARRIVING`). A warm-up
    (``warm_s``) sends two for each request held in flight, the load's own prompts at most, or the lines that arrive
    in its first ``warm_s`` seconds, one at least."""
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
    beside = [n for n in out if not LOADS[n].number]
    if beside and len(out) > 1:
        raise SystemExit(f"--loads: {beside[0]} is served in a run of its own: every load of a run is served by one "
                         f"engine, and {beside[0]}'s context limit of {LOADS[beside[0]].context} would be the others' too")
    return out


#: vLLM's client's environment beside the job's own: a planned load's tokens are drawn under Python's hash of each
#: block's id, which differs from one process to the next unless its seed is set
CLIENT_ENV = {"PYTHONHASHSEED": "0"}


def bench_argv(model: Model, checkpoint: Path, port: int, load: Load, rung: int, d: Path, tag: str,
               dataset: str | None = None, prompts: int | None = None, seed: int = 0) -> list[str]:
    """One pass of vLLM's client against the server on ``port``: ``load``'s prompts (the same ones under one seed, so
    every arm and rep serves the same requests) at concurrency ``rung``, greedy and each decoding to its length, with
    goodput against the model's SLO and its result saved to ``d``. The client sends no completion of its own before
    the pass (no ready check: `wait_ready` has waited), so every completion it sends is in its result, and request k's
    X-Request-Id is ``{tag}-{k}``, the form the release-order arm reads its release number from
    (`servetax_plugin.release_key`). It names the served model in every request and takes its tokenizer from the
    checkpoint; for its synthetic datasets it first asks the server's ``/tokenize``, which the engine never sees,
    whether the server counts the first prompt's tokens as it does, and trims every prompt to the server's count
    where it doesn't. A planned load (`Load.plan`) is sent from its file (``dataset``, `write_trace`) in the file's
    order, each prompt as the token ids the client draws for its blocks (`CLIENT_ENV`), so the server counts each
    prompt's planned length: held at ``rung`` in flight whatever the file's arrivals (``--no-self-timed``), or, at
    `ARRIVING`, each sent at its arrival after the pass starts, which is the client's way with such a file, with no
    bound on the requests in flight."""
    slo = dict(zip(("ttft", "tpot"), model.slo))
    held = [] if rung == ARRIVING else [*(["--no-self-timed"] if load.plan else []), "--request-rate", "inf",
                                        "--max-concurrency", str(rung)]
    return [sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve", "--backend", "openai",
            "--base-url", f"http://127.0.0.1:{port}", "--endpoint", "/v1/completions",
            "--model", model.served, "--tokenizer", str(checkpoint),
            *(["--trust-remote-code"] if "--trust-remote-code" in model.args else []),
            *load.dataset, *(["--dataset-path", dataset] if dataset else []),
            "--num-prompts", str(prompts or load.prompts), "--seed", str(seed),
            *held, "--ignore-eos", "--temperature", "0",
            "--goodput", *(f"{k}:{round(slo[k] * 1e3)}" for k in load.slo),
            "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "99",
            "--request-id-prefix", f"{tag}-", "--ready-check-timeout-sec", "0", "--disable-tqdm",
            "--save-result", "--save-detailed", "--result-dir", str(d), "--result-filename", RESULT]


#: what the client's detailed result holds that no run keeps: each request's generated text, which is a served
#: request's content, and its inter-token times, a number for every token, which the client's percentiles summarize
UNKEPT = frozenset({"generated_texts", "itls"})


def drive_bench(port: int, model: Model, checkpoint: Path, load: Load, rung: int, tag: str, d: Path,
                dataset: str | None = None, seed: int = 0, lib: object | None = None, prompts: int | None = None,
                run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                clock: Callable[[], float] = time.monotonic, now: Callable[[], float] = time.time) -> dict:
    """One pass sent by vLLM's client (`bench_argv`): its output in d/bench.log, the result it saved in d/result.json,
    and d/meta.json in `drive_load`'s form (a request the client counted failed is unfinished and unserved). The
    result is the client's detailed one, each request's lengths, start, first-token time and latency in the order
    sent, less what no run keeps (`UNKEPT`): the client saves it in a directory outside the run's, removed when the
    pass ends, so that the generated texts never lie under ``d`` whatever stops the pass."""
    d.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="serve-tax-result.") as tmp:
        saved, held = Path(tmp).resolve(), SHARED.RESEARCH_RUN_DIR
        if any(saved.is_relative_to(p) for p in (d.resolve(), *([Path(held).resolve()] if held else []))):
            raise RuntimeError(f"the client's result would be saved in {saved}, inside the run's directory, which leaves "
                               "the node with each request's generated text: TMPDIR, outside every run's directory")
        argv = bench_argv(model, checkpoint, port, load, rung, saved, tag, dataset, prompts, seed)
        t, t0 = clock(), now()
        with GpuBusy(lib, SHARED.CUDA_VISIBLE_DEVICES) as busy, open(d / "bench.log", "wb") as log:
            rc = run(argv, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **CLIENT_ENV}).returncode
        wall = clock() - t
        if rc or not (saved / RESULT).exists():
            raise RuntimeError(f"vllm bench serve exited {rc}{'' if rc else ' and saved no result'}: {d / 'bench.log'}")
        r = {k: v for k, v in json.loads((saved / RESULT).read_text()).items() if k not in UNKEPT}
    (d / RESULT).write_text(json.dumps(r))
    failed = int(r.get("failed") or 0)
    meta = {"tag": tag, "client": BENCH, "concurrency": rung, "rate": None, "requests": r["completed"] + failed,
            "unfinished": failed, "unserved": failed, "wall_s": round(wall, 3), "t0_unix": t0, "argv": argv[1:],
            "gpu": busy.result()}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def bench_metrics(r: dict) -> dict:
    """One pass's metrics from the result vLLM's client saved, under `load_metrics`'s names: the client's own figures
    (its goodput, throughputs, token counts, medians and P99s), none recomputed here; the prompt tokens a second are
    its total throughput less its output throughput."""
    sent = r["completed"] + int(r.get("failed") or 0)
    good = r.get("request_goodput")
    met = round(good * r["duration"]) if good is not None else None

    def s(key: str) -> float | None:
        return r[key] / 1e3 if r.get(key) is not None else None

    return {"sent": sent, "ok": r["completed"], "output_tokens": r["total_output_tokens"],
            "prompt_tokens": r["total_input_tokens"], "span_s": r["duration"],
            "output_tok_per_s": r["output_throughput"], "total_tok_per_s": r["total_token_throughput"],
            "prompt_tok_per_s": r["total_token_throughput"] - r["output_throughput"],
            "req_per_s": r["request_throughput"], "goodput_req_per_s": good, "met": met,
            "slo_attained": met / sent if met is not None and sent else None,
            "ttft_p50_s": s("median_ttft_ms"), "ttft_p99_s": s("p99_ttft_ms"), "itl_p50_s": s("median_itl_ms"),
            "itl_p99_s": s("p99_itl_ms"), "tpot_p50_s": s("median_tpot_ms"), "tpot_p99_s": s("p99_tpot_ms")}


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
        loads, lines = passes_of(loads, datasets)
        traces = {n: write_trace(out / f"{n}.trace.jsonl", x) for n, x in lines.items()}
        sent = client_files(datasets, traces)
        context = context_of(loads, given)
        passes = {pass_name(n, c): (n, c) for n, rungs in loads.items() for c in rungs}
        rates, plan = {}, []
    else:
        if not a.rates or not a.c4:
            raise SystemExit(f"--client {CORPUS} serves --rates NAME=REQ_PER_S,... of the --c4 shard's plan")
        passes, context, traces, lines, sent, sizes = {}, given, {}, {}, {}, {}
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
            "plugin": list(arm.plugin), "env": arm.env, "tick": tick, "window_s": a.window_s, "kv_window_s": a.kv_window_s, "rates": rates, "reps": a.reps, "plan": a.plan,
            "general": entry_points(arm),
            "boundary_rows": ({"window_s": a.window_s, "check": model.check, **(a.boundary_rows or arm.rows)}
                              if "boundary_rows" in arm.plugin else None),
            "client": a.client, "passes": {k: {"load": n, "concurrency": c} for k, (n, c) in passes.items()},
            "pass_prompts": sizes,
            "datasets": {n: {"file": f, "sha512": LOADS[n].file.sha512, "repo": LOADS[n].file.repo,
                             "revision": LOADS[n].file.revision}
                         for n, f in (datasets if bench else {}).items() if LOADS[n].file},
            "planned": {n: {**planned(lines[n]), "sha256": file_digest(Path(f)),
                            **({"table_sha256": file_digest(Path(datasets[n]))} if n in datasets else {})}
                        for n, f in traces.items()},
            "commit": source_commit(),
            "plan_sha256": file_digest(Path(a.plan)) if a.plan and Path(a.plan).is_file() else None,
            "span_s": a.span_s, "c4": a.c4, "seed": a.seed, "sampling": arm.sampling, "logprobs": arm.logprobs, "edges": edges,
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
                               pass_prompts(LOADS[n], c, lines.get(n), a.warm_s if warm else None, sizes.get(n)))

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


def tick_args(source: str | None, given_ms: str | None, load: str = "saturated") -> dict:
    """The certified tick's constants in µs: τ_b the corpus's bucket, τ_sat and τ_free the corpus's multiples of the
    mean busy cycle of a ``record`` step trace (a file, or a ``record`` arm's dir: its trace over ``load``'s first rep),
    or given as ``SAT,FREE`` ms."""
    if not TICK_PLUGIN.is_file():
        raise SystemExit(f"the tick arms need the certified tick ({TICK_PLUGIN.name}), which this tree lacks: run them "
                         "from a tree that has --admit tick")
    C = corpus()
    if (source is None) == (given_ms is None):
        raise SystemExit("the tick arms take τ from one of --tick-from STEP_TRACE|RECORD_DIR and --tick-ms SAT,FREE")
    if given_ms is not None:
        sat, free = (float(x) for x in given_ms.split(","))
        return {"bucket_us": C.TICK_BUCKET_MS * 1000, "sat_us": round(sat * 1000), "free_us": round(free * 1000),
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
    return {"bucket_us": C.TICK_BUCKET_MS * 1000, "sat_us": round(C.TICK_SAT * c["mean_us"]),
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
    from the prefix cache and, where the engine drafted, the draft tokens accepted a draft."""
    if not s or not s.get("steps"):
        return {}
    return {"steps": s["steps"], "tokens_per_step": s["step_tokens"] / s["steps"],
            "rows_per_step": s["generation_tokens"] / s["steps"], "preemptions": s["preemptions"],
            "prompt_cached_share": s["prompt_tokens_cached"] / s["prompt_tokens"] if s["prompt_tokens"] else None,
            **({"accepted_per_draft": s["accepted_tokens"] / s["drafts"]} if s.get("drafts") else {})}


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


def arm_metrics(d: Path, slo: tuple[float, float] | None = None) -> dict:
    """An arm's run dir: its meta and each load's metrics by rep, with the GPUs' busy time and the edges that fell in
    it, and its edges' summary (`edge_summary`)."""
    meta = json.loads((d / "meta.json").read_text())
    ran = MODELS[meta["model_id"]].slo
    slo = slo or ran
    edges = edge_records(d)
    loads: dict[str, list[dict]] = {}
    for name in meta.get("passes") or meta["rates"]:
        for rep in sorted((d / name).glob("rep-*"), key=lambda p: int(p.name.split("-")[1])):
            if (rep / RESULT).exists():
                if slo != ran:
                    raise SystemExit(f"{rep}: vLLM's client counted this pass's goodput against the SLO it ran with "
                                     f"(TTFT {ran[0]} s, TPOT {ran[1]} s); serve it again to count against another")
                lm = bench_metrics(json.loads((rep / RESULT).read_text()))
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
                                               **gpu_metrics(m, lm.get("output_tokens")), **load_edges(edges, m)})
    return {"schema": SCHEMA + "/metrics", "arm": meta["arm"], "parent": meta.get("parent"), "what": meta.get("what"),
            "run_id": meta.get("run_id"), "commit": meta.get("commit"), "vllm": (meta.get("versions") or {}).get("vllm"),
            "model_id": meta["model_id"], "slo": {"ttft_s": slo[0], "tpot_s": slo[1]},
            "startup_s": meta.get("startup_s"), "error": meta.get("error"), "tick": meta.get("tick"), "loads": loads,
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
    "exact int8, proof of useful work, proof of useful space, the memory audit, the network certifier in front",
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


def joined(arms: list[dict]) -> dict[str, dict]:
    """Arms' metrics by arm, the directories of one arm joined: an arm that two jobs served, each a load of its own,
    has both loads' passes, each load with its own job's commit and run (``of``), and its boundary commitment's counts
    are the two jobs' sums. Two directories of one arm that served the same load, where a rung's reps would be of two
    server lifetimes, or that differ in the model, the SLO or vLLM, are refused."""
    by: dict[str, dict] = {}
    for a in arms:
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
        raise SystemExit(f"cost: {reference} and {arm} share no load ({', '.join(LOADS)}) sent by vLLM's client: serve "
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
            "commitment": commitment(ours), "arms": rows, "no_arm": list(NO_ARM)}


def _c(c: dict | None) -> str:
    if not c or c.get("value") is None:
        return "—"
    return f"{c['value']:.2f}%" + (f" [{c['ci95'][0]:.2f}%, {c['ci95'][1]:.2f}%]" if c.get("ci95") else "")


def _reps(n: int) -> str:
    return f"{n} rep{'s' if n != 1 else ''}"


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
    under the metric's id either."""
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
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--stop-at", metavar="TIME",
                   help="the time the arm stops by, with its zone (2030-01-01T00:00:00Z): it starts no pass that it "
                        "can't expect to finish by then, and keeps the passes it served (stop_before)")
    r.add_argument("--span-s", type=float, default=90.0, help="each load's arrivals span")
    r.add_argument("--requests", type=int, help="each load's request count instead of --span-s's")
    r.add_argument("--warm-s", type=float, default=20.0,
                   help="a warm-up load at the lowest rate first, or a short pass of vLLM's client (0: none)")
    r.add_argument("--commit-wait-s", type=float, default=900.0,
                   help="the boundary-rows arm: how long the server is held after its loads for its commits (await_commits)")
    r.add_argument("--drain-s", type=float, default=600.0)
    r.add_argument("--startup-s", type=float, default=1800.0)
    r.add_argument("--window-s", type=float, default=600.0, help="the Commit period the tokens arm closes at")
    r.add_argument("--kv-window-s", type=float, default=600.0, help="the window edge the kv arm hashes the KV cache at")
    r.add_argument("--boundary-rows", type=json.loads, metavar="JSON",
                   help="a boundary commitment's object in place of the arm's own: check, per_prompt, per_output, "
                        "drained, inline, tables, stop (the serve's stop rule: ids, max_model_len, outputs)")
    r.add_argument("--private", metavar="DIR",
                   help="where the arm keeps what never leaves the node (a boundary commitment's classes, history tables' "
                        "rows and salts, the top-k's counters), outside every run's directory (under vLLM's client "
                        "by default a fresh directory, removed when the arm stops; the corpus client needs it)")
    r.add_argument("--plan", help="the plan's file, shared by a job's arms (default: OUT/plan.jsonl)")
    r.add_argument("--docs", type=int, default=20000, help="the shard's documents a plan draws from")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--tick-from", metavar="STEP_TRACE|RECORD_DIR", help="τ for the tick arms (ignored by the others)")
    r.add_argument("--tick-ms", metavar="SAT,FREE")
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
    co.add_argument("--reference", default=REFERENCE)
    co.add_argument("--target", type=float, metavar="PERCENT", help="the most it may cost (default: the model's)")
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
            if arm.tick or arm.sampling != "greedy" or arm.logprobs:
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
        loads, lines = passes_of(loads, files)
        for n, rungs in loads.items():
            plan = planned(lines[n]) if n in lines else None
            arriving = rungs == (ARRIVING,)
            digest = (None if n not in files else f"SHA-512 {LOADS[n].file.sha512[:16]}" if LOADS[n].file else
                      f"SHA-256 {file_digest(Path(files[n]))[:16]}")
            each = [pass_prompts(LOADS[n], c, lines.get(n), prompts=sizes.get(n)) for c in rungs]
            print(f"[serve-tax] load {n}: {GIVEN if arriving else LOADS[n].what}; "
                  f"{', '.join(f'{x:,}' for x in (each if len(set(each)) > 1 else each[:1]))} prompts a pass "
                  f"{at_rung(ARRIVING) if arriving else 'at concurrency ' + ', '.join(map(str, rungs))}"
                  + (f" (--pass-prompts: the load's own are {LOADS[n].prompts:,})" if n in sizes else "")
                  + (f", from {files[n]} ({digest}...)" if digest else "")
                  + (f", {plan['repeated_tokens']:,} of their {plan['prompt_tokens']:,} prompt tokens repeated"
                     if plan else "")
                  + (f", arriving over {plan['span_s']:,.1f} s" if arriving else ""))
        return 0
    slo = (a.slo_ttft_s, a.slo_tpot_s) if a.cmd != "cost" and a.slo_ttft_s and a.slo_tpot_s else None
    arms = [json.loads(d.read_text()) if d.is_file() else arm_metrics(d, slo) for d in a.dirs]
    if a.cmd == "cost":
        rep = cost_report(arms, a.arm, a.reference, a.target)
        if a.result and not a.out:
            raise SystemExit("cost --result: with --out DIR, where the report it cites is written")
        if a.out:
            a.out.mkdir(parents=True, exist_ok=True)
            (a.out / "serving_cost.json").write_text(json.dumps(rep, indent=1))
            (a.out / "serving_cost.txt").write_text(cost_text(rep))
        if a.result:
            write_result(rep, a.out)
        print(json.dumps(rep) if a.json else cost_text(rep), end="\n" if a.json else "")
        return 0
    if a.cmd == "metrics":
        print("\n".join(json.dumps(x) for x in arms))
        return 0
    t = table(arms)
    md = markdown(t)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "table.json").write_text(json.dumps(t, indent=1))
        (a.out / "table.md").write_text(md)
    print(md, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
