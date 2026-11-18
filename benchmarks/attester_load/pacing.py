"""The time a network attester's schedule adds to serving, as a percent of stock's serving time, modeled by replaying a
stored stock run's requests under the schedule.

    uv run python benchmarks/attester_load/pacing.py --from RUN [--card CARD] [--arm prod] [--load agentic@4] [--json]
    uv run python benchmarks/attester_load/pacing.py --dir OUT ...     the same from a run's `out/` on disk

A request of the stored run (vLLM bench's `result.json` of each user: its start, time to first token, latency and
prompt and output lengths) passes through the network attester as `network_attester.py` states its schedule. It is
admitted at the first tick after it arrives, half a bucket later on average. Its id request, cut into cells of
`cell_bytes`, goes in at `slots` cells a bucket from the admission's bucket s, and the server gets it whole at the tick
of its `END`'s bucket. Egress cell i is cut at tick s + D + ⌊i/r⌋ from the server's bytes there by then, and a cell
whose token isn't there fails the connection (late), as a request over the response cap, past the ingress bounds or
past the egress cells does. The server's own times are stock's, from the tick it gets the request. A request that
doesn't fail is answered when its `END` cell, cell k for k tokens, leaves: `paced` after tick s + D + ⌊k/r⌋. Each
user's later turns move by what its earlier turns added, and a pass ends at its last request's end.

So the figure holds the engine's times at stock's: it serves nothing again, and a schedule under which any request
fails has no figure, only its failures, since it doesn't serve the load. Only a request's first and last token have
times, so a request whose tokens between them are late is not seen to fail: the failures are a lower bound. The lanes
are not replayed: a schedule with fewer lanes than the run's most concurrent requests is refused.

Under a research run it writes `METRIC` for each schedule, member `<schedule>/<model>/<card>`, marked modeled.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import statistics
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from verity.protocols.resource_accounting.network.attester.runner_cli import PACED_MS
from verity.protocols.resource_accounting.network.verifier import id_form
from verity.protocols.resource_accounting.network.verifier.link_frame import HEADER

ROOT = Path(__file__).resolve().parents[2]
METRIC = "network-accounting/serving-slowdown-under-pacing"
#: the provisional schedule of the served model's network attester, whose `schedule` a `Schedule` reads
CARD = ROOT / "benchmarks/spot_figures/cards/certifier.glm-5.3.json"
REASONS = ("cap", "ingress", "limit", "late")


@dataclass(frozen=True)
class Schedule:
    """The fields of a network attester's schedule (`NetworkAttesterParams`) the replay reads."""
    bucket_ms: int
    slots: int
    delay: int
    cell_bytes: int
    lanes: int
    ingress_cells: int
    ingress_buckets: int
    egress_cells: int
    max_tokens_cap: int | None = None
    paced_ms: float = PACED_MS

    @classmethod
    def of_card(cls, path: Path = CARD) -> "Schedule":
        s = json.loads(path.read_text())["value"]["schedule"]
        return cls(**{k: s[k] for k in cls.__dataclass_fields__ if k in s})


def schedules(card: Schedule) -> dict[str, Schedule]:
    """The schedules a run is replayed under: the card's, and the card's with cells, ingress bounds and cap wide enough
    for an agent's request and response. Past the card's, the delays are the least under which no request of the
    stored GLM-5.3 four-agent run fails (but the 20 s one, which fails some), and at 250 ms and longer buckets the pace
    keeps 160 ids a second."""
    wide = dict(cell_bytes=4096, ingress_cells=1024, egress_cells=4096, max_tokens_cap=2048)

    def w(bucket_ms: int, slots: int, delay: int) -> Schedule:
        return replace(card, **wide, bucket_ms=bucket_ms, slots=slots, delay=delay, ingress_buckets=min(128, delay))

    return {"card": card,
            "b100-r16-d200": w(100, 16, 200),
            "b100-r4-d335": w(100, 4, 335),
            "b100-r16-d414": w(100, 16, 414),
            "b250-r40-d166": w(250, 40, 166),
            "b500-r80-d84": w(500, 80, 84),
            "b1000-r160-d42": w(1000, 160, 42)}


@functools.cache
def request_cells(prompt_ids: int, max_tokens: int, cell_bytes: int) -> int:
    """The ingress cells of an id request of ``prompt_ids`` ids: byte k in cell ⌊k/P⌋, its last bytes in its `END`."""
    body = id_form.id_request(model="m", user="u", prompt=[0] * prompt_ids, max_tokens=max_tokens,
                              frame_bytes=cell_bytes, offset=0)
    return len(body) // (cell_bytes - HEADER.size) + 1


def outcome(s: Schedule, ttft: float, latency: float, prompt_ids: int, tokens: int) -> tuple[str | None, float]:
    """Why the request fails under ``s`` (one of `REASONS`, the first it meets) or None, and its latency through the
    network attester in seconds."""
    tau = s.bucket_ms / 1000
    answered = tau / 2 + (s.delay + tokens // s.slots) * tau + s.paced_ms / 1000
    if s.max_tokens_cap is not None and tokens > s.max_tokens_cap:
        return "cap", answered
    cells = request_cells(prompt_ids, tokens, s.cell_bytes)
    ingress = math.ceil(cells / s.slots)
    if cells > s.ingress_cells or ingress > s.ingress_buckets:
        return "ingress", answered
    if tokens + 1 > s.egress_cells:
        return "limit", answered
    at = (ingress - 1) * tau
    if at + ttft > s.delay * tau or at + latency > (s.delay + (tokens - 1) // s.slots) * tau:
        return "late", answered
    return None, answered


@dataclass(frozen=True)
class Replay:
    """A schedule's replay over a run's passes: each pass's time through the network attester over stock's, and the
    requests that fail, by reason."""
    passes: int
    requests: int
    concurrent: int
    ratios: tuple[float, ...]
    failed: Mapping[str, int]

    @property
    def percent(self) -> float | None:
        """The median pass's time added, as a percent of stock's; None where a request fails."""
        return None if any(self.failed.values()) else 100 * (statistics.median(self.ratios) - 1)

    @property
    def spread(self) -> list[float] | None:
        return None if self.percent is None else [100 * (min(self.ratios) - 1), 100 * (max(self.ratios) - 1)]


def most_concurrent(spans: Iterable[tuple[float, float]]) -> int:
    edges = sorted((t, d) for a, b in spans for t, d in ((a, 1), (b, -1)))
    n = top = 0
    for _, d in edges:
        n += d
        top = max(top, n)
    return top


def replay(s: Schedule, passes: Sequence[Sequence[Mapping[str, Any]]]) -> Replay:
    """``passes``: each pass's users' vLLM bench results, one request a turn, a user's turns in order."""
    ratios, failed, n, spans = [], dict.fromkeys(REASONS, 0), 0, []
    for users in passes:
        start, stock_end, paced_end = [], [], []
        for u in users:
            moved = 0.0
            for t0, ttft, lat, p, k in zip(u["start_times"], u["ttfts"], u["latencies"], u["input_lens"],
                                          u["output_lens"]):
                n += 1
                why, through = outcome(s, ttft, lat, p, k)
                if why:
                    failed[why] += 1
                start.append(t0)
                stock_end.append(t0 + lat)
                paced_end.append(t0 + moved + through)
                spans.append((t0 + moved, t0 + moved + through))
                moved += through - lat
        first = min(start)
        ratios.append((max(paced_end) - first) / (max(stock_end) - first))
    top = most_concurrent(spans)
    if top > s.lanes:
        raise SystemExit(f"pacing: {top} requests run at once and the schedule has {s.lanes} lanes: queueing for a "
                         "lane is not replayed")
    return Replay(len(passes), n, top, tuple(ratios), failed)


def passes_of(out: Path, arm: str, load: str) -> list[list[dict]]:
    """The run's counted passes of ``arm`` at ``load``: `out/<arm>.<lifetime>/<load>/rep-<k>/user-*/result.json`."""
    reps = sorted(out.glob(f"{arm}.*/{load}/rep-*"), key=lambda p: (int(p.parent.parent.name.rsplit(".", 1)[1]),
                                                                     int(p.name.split("-")[1])))
    got = [[json.loads((u / "result.json").read_text()) for u in sorted(r.glob("user-*"))] for r in reps]
    if not any(got):
        raise SystemExit(f"pacing: no pass of {arm} at {load} under {out}")
    return got


def local_store() -> Any:
    from research.store.local import LocalStore
    return LocalStore()


def fetched(run: str, to: Path, arm: str, load: str, store: Callable[[], Any] = local_store) -> Path:
    """The `out/` of recorded run ``run``'s record, its passes' results alone, fetched from the evidence store."""
    s = store()
    a = s.get_attempt(run)
    art = ((a or {}).get("outputs") or {}).get("run_record")
    if not art:
        raise SystemExit(f"pacing: {run} kept no record in the evidence store this machine reads")
    return Path(s.fetch(art, to, paths=[f"out/{arm}.*/{load}/rep-*/user-*/result.json"])) / "out"


def rows(named: Mapping[str, Schedule], passes: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, Replay]:
    return {name: replay(s, passes) for name, s in named.items()}


def render(doc: dict) -> str:
    lines = [f"# The time the network attester's schedule adds to serving, modeled from {doc['run']}",
             f"{doc['model']} on {doc['card']}, stock arm {doc['arm']} at {doc['load']}: {doc['passes']} passes, "
             f"{doc['requests']} requests; the engine's times are stock's, and failures are a lower bound", "",
             "| schedule | bucket ms | ids a bucket | delay buckets | cell bytes | ingress cells, buckets | cap | "
             "failed (cap, ingress, limit, late) | added, % of stock | over the passes |",
             "| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |"]
    for name, r in doc["rows"].items():
        s, f = r["schedule"], r["failed"]
        pct = "none: fails" if r["percent"] is None else f"{r['percent']:.1f}"
        spread = "" if r["spread"] is None else f"{r['spread'][0]:.1f} to {r['spread'][1]:.1f}"
        lines.append(f"| {name} | {s['bucket_ms']} | {s['slots']} | {s['delay']} | {s['cell_bytes']} | "
                     f"{s['ingress_cells']}, {s['ingress_buckets']} | {s['max_tokens_cap']} | "
                     f"{sum(f.values())}: {', '.join(str(f[k]) for k in REASONS)} | {pct} | {spread} |")
    return "\n".join(lines) + "\n"


def document(run: str, card: str, arm: str, load: str, named: Mapping[str, Schedule],
             passes: Sequence[Sequence[Mapping[str, Any]]]) -> dict:
    got = rows(named, passes)
    models = {u.get("model_id") for p in passes for u in p}
    one = next(iter(got.values()))
    return {"run": run, "card": card, "arm": arm, "load": load, "model": "/".join(sorted(m for m in models if m)),
            "passes": one.passes, "requests": one.requests,
            "rows": {name: {"schedule": asdict(named[name]), "percent": r.percent, "spread": r.spread,
                            "failed": dict(r.failed), "concurrent": r.concurrent} for name, r in got.items()}}


def write(doc: dict, out: Path) -> int:
    """Each schedule's figure as `METRIC` through the harness kit's writer: under a research run its result, else
    `out/result.json`."""
    from verity_bench import kit

    w = kit.writer(str(out))
    for name, r in doc["rows"].items():
        failed = sum(r["failed"].values())
        w.measurement(METRIC, r["percent"], "percent", r["spread"], mark="modeled",
                      member=f"{name}/{doc['model']}/{doc['card']}",
                      why=None if not failed else f"fails at least {failed} of {doc['requests']} requests: it doesn't "
                                                  "serve the load",
                      inputs=[f"modeled: {doc['run']}'s {doc['arm']} at {doc['load']} replayed under the schedule; the "
                              "engine's times are stock's", json.dumps(r["schedule"], sort_keys=True)])
    return len(doc["rows"])


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pacing.py", description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from", dest="run", metavar="RUN", help="a stored serving run, read from the evidence store")
    src.add_argument("--dir", type=Path, metavar="OUT", help="a serving run's out/ on disk")
    ap.add_argument("--card", default="h200x8", help="the GPU card the run served on, as the member names it")
    ap.add_argument("--arm", default="prod", help="the stock arm whose passes are replayed")
    ap.add_argument("--load", default="agentic@4", help="the load whose passes are replayed")
    ap.add_argument("--json", action="store_true", help="the document as JSON in place of the table")
    ap.add_argument("--out", type=Path, help="also write the result through the harness kit's writer for DIR")
    a = ap.parse_args(argv)
    named = schedules(Schedule.of_card())
    if a.run:
        with tempfile.TemporaryDirectory(prefix="pacing-") as tmp:
            doc = document(a.run, a.card, a.arm, a.load, named, passes_of(fetched(a.run, Path(tmp), a.arm, a.load),
                                                                          a.arm, a.load))
    else:
        doc = document(str(a.dir), a.card, a.arm, a.load, named, passes_of(a.dir, a.arm, a.load))
    print(json.dumps(doc, indent=1) if a.json else render(doc), end="" if not a.json else "\n")
    from verity_bench import settings

    if a.out is not None or settings.is_set("RESEARCH_RUN_DIR"):
        print(f"wrote {write(doc, a.out or Path('out'))} schedule(s)' figures", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
