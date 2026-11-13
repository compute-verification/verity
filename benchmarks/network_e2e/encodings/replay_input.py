"""A trace as the replay's input: `replay_input` reads a step trace (`replay.read_any`) or a derived-format one
(`*.json.gz`, `derived.load_trace`'s) as what `replay.replay` takes, with the trace's own rows to hold the replay to
(`Input`). The pool and window checks here that replay whole traces read their traces through it."""
from __future__ import annotations

import bisect
import gzip
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay as RP


@dataclass
class Input:
    """What the replay takes from a trace (`replay.replay`), and the trace's own rows to hold it to."""
    engine: RP.Engine
    requests: list[RP.RequestIO]
    before: list[int]
    rows: list[list[tuple[int, int, int]]]


def replay_input(path: Path) -> Input:
    """A step trace (`replay.read_any`), or a derived-format trace (`*.json.gz`, `derived.load_trace`'s), as replay input.
    A derived-format request is admitted before the first step to end after its arrival, or at the step it is first
    scheduled in, if that is sooner (an idle engine starts its next step on the arrival)."""
    if path.suffix != ".gz":
        tr = RP.read_any(path)
        return Input(tr.engine, tr.requests, tr.before_step, [list(s.rows) for s in tr.steps])
    t = json.loads(gzip.open(path).read())
    c = t["config"]
    eng = RP.Engine(max_num_seqs=c["max_num_seqs"], max_num_batched_tokens=c["max_num_batched_tokens"],
                    max_model_len=c["max_model_len"], num_gpu_blocks=c["num_gpu_blocks"], block_size=c["block_size"],
                    long_prefill_token_threshold=c["long_prefill_token_threshold"],
                    enable_prefix_caching=c["enable_prefix_caching"], async_scheduling=c["scheduler"] == "async",
                    watermark=c["watermark"])
    rows = [[(x["request"], x["start"], x["n"]) for x in s["rows"]] for s in t["steps"]]
    first: dict[int, int] = {}
    for k, st in enumerate(rows):
        for i, _, _ in st:
            first.setdefault(i, k)
    ends = [s["t_end_s"] for s in t["steps"]]
    reqs, before = [], []
    for i, r in enumerate(t["requests"]):
        out = r["output_ids"]
        reqs.append(RP.RequestIO(prompt_len=len(r["prompt_ids"]), max_tokens=r["max_tokens"], output_len=len(out),
                                 eos=r["finish"] == "eos", tokens=tuple(r["prompt_ids"]) + tuple(out)))
        before.append(min(bisect.bisect_left(ends, r["arrival_s"]) + 1, first[i]))
    return Input(eng, reqs, before, rows)
