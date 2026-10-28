"""The network certifier's honest path, end to end: honest traces through the reference protocol at calibrated parameters.

`network-certifier-calibrate` sizes the certifier from honest traces (`record.py`'s) and estimates the honest miss rate with
its own model of the prover. This runs the protocol's code on the same windows, at the parameters of a calibration
report (its JSON), and reports what the audit itself decides. Each configuration of a trace is one run of one spatial
unit, so it gets its own `audit.Audit` and one clock:
- **the honest prover** declares one sync per window for all the configuration's egress links (`--syncs`): by default
  in hindsight, met on every one, with `calibration.choose_syncs` over the report's Σ_sync; `envelope`, the
  calibration's envelope in every window; or an online rule of `online` (`last`, `cover-K-M[-rhoR][-fallback]`,
  `sticky-K-M-N[-rhoR]`), declared at each window's first bucket from the frames ready by then, with its history
  carried across a run's configurations (`--fresh` starts each with none). An online rule's egress rows carry
  `choices`, the values a sender timing its writes could have made it declare in the window (`online.Rule.choices`);
- **the honest sender** is `schedule.run_shaper`, driving a `grid.Certifier` at the report's egress r. A late frame goes
  when ready, so a window that no sync meets deviates from the Program's grid;
- **the ingress certifier** is a `grid.Certifier` at the report's r_in and τ_in. A frame is enqueued at the first ingress
  bucket boundary after it arrives (`calibration.size_ingress_ms`'s model);
- **the published records** are the raw network certificates, their commitments (`commitment.commit`) and their network
  certificates (`certificate.Issuer`), signed by `ReplaySigner`;
- **the verifier** is `Audit.check_link` on egress and `Audit.check_ingress_link` on ingress, once on the raw records,
  once on the committed ones and once on the certified ones. The three must agree on every link-window, except that
  the issuer certifies no incomplete status, so the certified audit gives a raw `status:<reason>` as a `no-verdict`
  (`certified_of`).

`ReplaySigner` is a stand-in: real LMS signatures under a toy key (`REPLAY_LMS`), leaves reused and every configuration
signed in one certifier session (`REPLAY_SESSION`), as no certifier may.

A frame's content is its position in its link's canonical order, as 8 bytes (`REPLAY_FRAME_BYTES`), so the release
bucket of every frame is read back from the records. Frames of the report's B (4 KiB) would hold gigabytes per window.
The audit compares contents exactly but never reads their size, so the verdicts are those of B-byte frames. Bandwidth
is reported at the report's B.

A rejected window gets a cause:
- `miss`: one of its own frames was late under the declared sync, so no value of Σ_sync met it (a calibrator miss);
- `carried`: none of its frames was late, but a late frame anchored in an earlier window was handed before it ended,
  so it reached the window through the shaper or the queue. Under a lenient sync a missed window's frames can be
  logically due in the next window, and then only that one deviates;
- `unexplained`: every frame went at its logical bucket and the grid still differs from the Program's;
- a status violation's own reason (`status:overflow`), or the violation itself (`advice`).

Latencies are in ms, counted in buckets: shaping is hand minus ready (as the calibrator reports it), queueing is
release minus hand, and end to end is release minus ready. A frame not released by the last window is counted apart.

    python replay.py --calibration REPORT.json --out DIR [--jobs N] [--syncs POLICY [--fresh]] TRACE.json|DIR ...

It writes `verdicts.jsonl`, one line per link-window, and `summary.json`.
"""

from __future__ import annotations

import argparse
import functools
import gzip
import hashlib
import json
import math
import multiprocessing
from collections import Counter
from collections.abc import Iterable, Sequence
from fractions import Fraction
from pathlib import Path

from verity.ml.crypto import lms
from verity.core.protocols.certifier import online
from verity.core.protocols.certifier.lean_audit import Audit
from verity.core.protocols.certifier.calibration import choose_syncs
from verity.core.protocols.certifier.certificate import Direction, Issuer, LmsKey, parameters_digest
from verity.core.protocols.certifier.commitment import commit
from verity.core.protocols.certifier.grid import GridParams, BlockGrid, Certifier
from verity.core.protocols.certifier.schedule import ClockSync, LogicalFrame, SyncSet, expected_grids, handing_buckets, run_shaper

REPLAY_FRAME_BYTES = 8
REPLAY_LMS = lms.Params(8, 2, 2, toy=True)
REPLAY_SESSION = hashlib.sha512(b"verity-network-traces/replay session").digest()[:lms.SESSION_BYTES]
PERCENTILES = (50, 99, 99.9)
AUDITS = ("raw", "committed", "certified")


class ReplaySigner:
    """The replay's signer: it starts any session, and its n-th signature is at leaf n mod 2^H of the toy key (the module
    docstring)."""

    def __init__(self) -> None:
        self.key, self.n = _replay_key(), 0

    @property
    def key_id(self) -> bytes:
        return self.key.I

    def public_key(self) -> LmsKey:
        return LmsKey(self.key.params, lms.public_key(self.key))

    def start(self, session: bytes) -> None:
        pass

    def sign(self, message: bytes) -> bytes:
        self.n += 1
        return lms.sign_at(self.key, (self.n - 1) % 2 ** self.key.params.H, message).encode()


@functools.cache
def _replay_key() -> lms.PrivateKey:
    return lms.keygen(REPLAY_LMS, hashlib.sha512(b"verity-network-traces/replay").digest()[:REPLAY_LMS.N])


def certify(records: Sequence[BlockGrid], direction: Direction, params: GridParams, parameters: bytes,
            signer: ReplaySigner) -> list:
    issuer = Issuer(signer, REPLAY_SESSION)
    return [c for r in records if (c := issuer.certify(r, direction, params, parameters)) is not None]


def certified_of(raw: str | None) -> str | None:
    """The raw audit's verdict of a link-window as the certified audit gives it: a window the raw audit rejects on its
    status has no certificate, so it is a `no-verdict`."""
    return "no-verdict" if raw is not None and raw.startswith("status:") else raw


class Setup:
    """A calibration report's parameters: the egress grid per link, Σ_sync, the ingress grid per link, and the frame
    size B the bandwidth is reported at."""

    def __init__(self, report: dict) -> None:
        g = report["grid"]
        window_ms = g["window_buckets"] * g["bucket_ms"]
        self.grid, self.frame_bytes = g, g["frame_bytes"]
        self.egress = {e["link"]: GridParams(g["window_buckets"], e["allocation"], REPLAY_FRAME_BYTES, g["bucket_ms"],
                                             g["queue_capacity"]) for e in report["egress"]}
        self.ingress = {}
        for i in report["ingress"]:
            if window_ms % i["bucket_ms"]:
                raise ValueError(f"the ingress bucket {i['bucket_ms']} ms doesn't divide the {window_ms} ms window")
            self.ingress[i["link"]] = GridParams(window_ms // i["bucket_ms"], i["allocation"], REPLAY_FRAME_BYTES,
                                                 i["bucket_ms"], g["queue_capacity"])
        cs = report["clock_sync"]
        self.sync_set = SyncSet(tuple(cs["deltas"]), tuple(Fraction(r) for r in cs["rhos"]))
        declared = cs.get("declared")
        self.envelope = (ClockSync(max(declared["delta"]), min(Fraction(r) for r in declared["rho"])) if declared
                         else ClockSync(max(self.sync_set.deltas), min(self.sync_set.rhos)))

    def parameters(self) -> dict:
        def bps(p: GridParams) -> float:
            return p.allocation * self.frame_bytes * 8 / (p.bucket_ms / 1000)

        return {
            "grid": self.grid, "replay_frame_bytes": REPLAY_FRAME_BYTES,
            "egress": {k: {"allocation": p.allocation, "bandwidth_bps": bps(p)} for k, p in self.egress.items()},
            "sync_set": {"deltas": list(self.sync_set.deltas), "rhos": [str(r) for r in self.sync_set.rhos],
                         "size": self.sync_set.size, "bits": self.sync_set.bits},
            "ingress": {k: {"allocation": p.allocation, "bucket_ms": p.bucket_ms, "buckets": p.buckets,
                            "bandwidth_bps": bps(p)} for k, p in self.ingress.items()},
        }


def payload(k: int) -> bytes:
    return k.to_bytes(REPLAY_FRAME_BYTES, "big")


def releases(records: Sequence[BlockGrid], buckets: int) -> dict[int, int]:
    """Each real frame's release bucket, read from the records' slots."""
    out = {}
    for w, record in enumerate(records):
        for t, row in enumerate(record.grid):
            for slot in row:
                if slot is None:
                    break
                out[int.from_bytes(slot, "big")] = w * buckets + t
    return out


def padding(record: BlockGrid) -> tuple[int, int]:
    """Real frames and slots of a record."""
    real = sum(sum(slot is not None for slot in row) for row in record.grid)
    return real, sum(len(row) for row in record.grid)


def config_windows(cfg: dict, buckets: int) -> int:
    if "windows" in cfg.get("labels", {}):
        lo, hi = cfg["labels"]["windows"]
        return hi - lo
    anchors = [f["anchor"] for link in cfg.get("egress", []) for f in link["frames"]]
    return max(anchors, default=-1) // buckets + 1


def diagnose(w: int, buckets: int, own_late: int, deviating: Sequence[tuple[int, int, int]], expected,
             record: BlockGrid) -> tuple[str, dict]:
    """A mismatched egress window's cause, and the first bucket where its grid leaves the Program's. `deviating` holds
    (anchor window, logical hand, actual hand) for every frame handed off its logical bucket. An earlier window's late
    frame handed before this window ends can reach it through the FIFO queue; `carried_late_frames` counts those
    whose hand buckets span part of this window."""
    lo, hi = w * buckets, (w + 1) * buckets
    earlier = [(h, a) for aw, h, a in deviating if aw < w and min(h, a) < hi]
    detail: dict = {"own_late_frames": own_late,
                    "carried_late_frames": sum(1 for h, a in earlier if max(h, a) >= lo),
                    "first_bucket": next((t for t, (a, b) in enumerate(zip(record.grid, expected[w], strict=True))
                                          if a != b), None)}
    if own_late:
        return "miss", detail
    if earlier:
        return "carried", detail
    return "unexplained", detail


def replay_configuration(cfg: dict, setup: Setup, syncs: str = "hindsight", prior: Sequence[online.Past] = (),
                         lag: int = 0, *, audit=Audit) -> tuple[list[dict], dict, list[online.Past]]:
    """One configuration through prover, shaper, certifier, records and audit, raw, committed and certified. The prover's
    syncs are `syncs` (`parse_sync_policy`), and an online rule starts from `prior` and reads the frames ready by `lag`
    buckets before each window (`online.declare`). The verifier is `audit` (a class with `audit.Audit`'s interface).
    Returns one row per link-window, the configuration's tallies and the online prover's history (empty for the other
    policies)."""
    sync_set = setup.sync_set
    eager = ClockSync(min(sync_set.deltas), max(sync_set.rhos))
    raw, committed, certified = audit(), audit(), audit()
    signer = ReplaySigner()
    key = signer.public_key()
    rows: list[dict] = []
    tally = new_tally()
    run = cfg.get("labels", {}).get("run")
    egress = cfg.get("egress", [])
    T = setup.grid["window_buckets"]
    windows = config_windows(cfg, T)
    links = [([LogicalFrame(payload(k), f["anchor"], f["index"]) for k, f in enumerate(link["frames"])],
              [f["ready"] for f in link["frames"]]) for link in egress]
    history: list[online.Past] = []
    choices: list[int] | None = None
    hands_per_link = None
    if syncs == "hindsight":
        declared, hands_per_link, misses = choose_syncs(links, sync_set, T)
        declared = declared + [eager] * (windows - len(declared))
        tally["egress"]["calibrator_misses"] += misses
    elif syncs == "envelope":
        declared = [setup.envelope] * windows
    else:
        declared, history, choices = online.declare(links, sync_set, T, online.rule(syncs), windows, setup.envelope,
                                                     prior, lag)
    syncs = declared
    for k, (link, (frames, ready)) in enumerate(zip(egress, links, strict=True)):
        name, params = link["link"], setup.egress[link["link"]]
        logical = handing_buckets(frames, syncs, T)
        actual, last = [], 0
        for h, r in zip(logical, ready, strict=True):
            last = max(h, r, last)
            actual.append(last)
        if hands_per_link is not None and actual != hands_per_link[k]:
            raise AssertionError(f"{cfg.get('name')} {name}: the shaper's hand times differ from the calibrator's")
        own_late = Counter(f.anchor // T for f, h, r in zip(frames, logical, ready, strict=True) if r > h)
        records = run_shaper(frames, ready, syncs, Certifier(name, params))
        published = [commit(r, params) for r in records]
        signed = certify(records, Direction.EGRESS, params, parameters_digest(params, sync_set), signer)
        v_raw = raw.check_link(name, records, frames, syncs, sync_set, params)
        v_com = committed.check_link(name, published, frames, syncs, sync_set, params)
        v_cer = certified.check_link(name, signed, frames, syncs, sync_set, params, key=key,
                                     session=REPLAY_SESSION)
        released = releases(records, T)
        expected, deviating = None, []
        if any(not v.accepted for v in v_raw + v_com + v_cer):
            expected = expected_grids(frames, syncs, params)
            deviating = [(f.anchor // T, h, a) for f, h, a in zip(frames, logical, actual, strict=True) if a != h]
        for w, (a, b, c) in enumerate(zip(v_raw, v_com, v_cer, strict=True)):
            real, slots = padding(records[w])
            row = {"config": cfg.get("name"), "run": run, "link": name, "direction": "egress", "window": w,
                   "raw": a.violation, "committed": b.violation, "certified": c.violation,
                   "sync": [syncs[w].delta, str(syncs[w].rho)],
                   "real_frames": real, "padding_share": 1 - real / slots, "own_late_frames": own_late.get(w, 0)}
            if choices is not None:
                row["choices"] = choices[w]
            violation = a.violation or b.violation or c.violation
            if violation == "mismatch":
                row["cause"], row["detail"] = diagnose(w, T, own_late.get(w, 0), deviating, expected, records[w])
            elif violation is not None:
                row["cause"] = violation
            rows.append(row)
            tally["egress"]["real"] += real
            tally["egress"]["slots"] += slots
        del expected, deviating
        e = tally["egress"]
        e["late_windows"] += len(own_late)
        e["frames"] += len(frames)
        for k, f in enumerate(frames):
            hand, rd = actual[k], ready[k]
            e["shaping"][hand - rd] += 1
            rel = released.get(k)
            if rel is None:
                e["unreleased"] += 1
                continue
            e["queueing"][rel - hand] += 1
            e["end_to_end"][rel - rd] += 1
            if f.index == 0:
                e["first_frame"][rel - f.anchor] += 1
                e["first_frame_unshaped"][rd - f.anchor] += 1
        del records, published, signed, frames, released
    for link in cfg.get("ingress", []):
        name, params = link["link"], setup.ingress[link["link"]]
        windows = config_windows(cfg, setup.grid["window_buckets"])
        if "arrival_ms" not in link:
            raise ValueError(f"{cfg.get('name')}: ingress link {name} has counts, not arrival times")
        arrival_ms = sorted(link["arrival_ms"])
        hands = [math.ceil(t / params.bucket_ms) for t in arrival_ms]
        certifier, records, k = Certifier(name, params), [], 0
        for t in range(windows * params.buckets):
            while k < len(hands) and hands[k] <= t:
                certifier.enqueue(payload(k))
                k += 1
            certifier.tick()
            if (t + 1) % params.buckets == 0:
                records.append(certifier.close_window())
        published = [commit(r, params) for r in records]
        signed = certify(records, Direction.INGRESS, params, parameters_digest(params), signer)
        v_raw = raw.check_ingress_link(name, records, windows)
        v_com = committed.check_ingress_link(name, published, windows)
        v_cer = certified.check_ingress_link(name, signed, windows, key=key, params=params,
                                             session=REPLAY_SESSION)
        released = releases(records, params.buckets)
        i = tally["ingress"]
        for w, (a, b, c) in enumerate(zip(v_raw, v_com, v_cer, strict=True)):
            real, slots = padding(records[w])
            row = {"config": cfg.get("name"), "run": run, "link": name, "direction": "ingress", "window": w,
                   "raw": a.violation, "committed": b.violation, "certified": c.violation, "real_frames": real,
                   "padding_share": 1 - real / slots}
            if a.violation or b.violation or c.violation:
                row["cause"] = a.violation or b.violation or c.violation
            rows.append(row)
            i["real"] += real
            i["slots"] += slots
        i["frames"] += len(arrival_ms)
        for k, t in enumerate(arrival_ms):
            rel = released.get(k)
            if rel is None:
                i["unreleased"] += 1
            else:
                i["delivery"][round(rel * params.bucket_ms - t, 3)] += 1
    tally["configurations"] = 1
    tally["runs_accepted"] = {"raw": int(raw.accepted), "committed": int(committed.accepted),
                              "certified": int(certified.accepted)}
    return rows, tally, history


def new_tally() -> dict:
    hists = ("shaping", "queueing", "end_to_end", "first_frame", "first_frame_unshaped")
    return {"configurations": 0, "runs_accepted": dict.fromkeys(AUDITS, 0),
            "egress": {"real": 0, "slots": 0, "frames": 0, "unreleased": 0, "calibrator_misses": 0, "late_windows": 0}
            | {h: Counter() for h in hists},
            "ingress": {"real": 0, "slots": 0, "frames": 0, "unreleased": 0, "delivery": Counter()}}


def add_tally(into: dict, other: dict) -> None:
    into["configurations"] += other["configurations"]
    for k in AUDITS:
        into["runs_accepted"][k] += other["runs_accepted"][k]
    for side in ("egress", "ingress"):
        for k, v in other[side].items():
            into[side][k] += v


def hist_percentiles(hist: Counter, scale: float = 1.0) -> dict:
    """Nearest-rank percentiles and the maximum of a histogram's values, times `scale`."""
    total = sum(hist.values())
    if not total:
        return {}
    out, ranks = {}, {p: min(total, max(1, math.ceil(p / 100 * total))) for p in PERCENTILES}
    acc, keys = 0, sorted(hist)
    for x in keys:
        acc += hist[x]
        for p, rank in ranks.items():
            if f"p{p:g}" not in out and acc >= rank:
                out[f"p{p:g}"] = x * scale
    return out | {"max": keys[-1] * scale}


def summarize(rows: Sequence[dict], tally: dict, setup: Setup) -> dict:
    bucket_ms = setup.grid["bucket_ms"]
    summary: dict = {"parameters": setup.parameters(), "configurations": tally["configurations"],
                     "runs_accepted": tally["runs_accepted"]}
    for side in ("egress", "ingress"):
        mine = [r for r in rows if r["direction"] == side]
        t = tally[side]
        n = len(mine)
        acc = {k: sum(r[k] is None for r in mine) for k in AUDITS}
        rejected = [r for r in mine if any(r[k] for k in AUDITS)]
        s = {"link_windows": n, "accepted": acc,
             "acceptance_rate": {k: (v / n if n else None) for k, v in acc.items()},
             "violations_by_cause": dict(Counter(r["cause"] for r in rejected)),
             "frames": t["frames"], "unreleased_frames": t["unreleased"],
             "padding_share": (1 - t["real"] / t["slots"]) if t["slots"] else None}
        if side == "egress":
            s |= {"calibrator_misses": t["calibrator_misses"], "late_windows": t["late_windows"],
                  "rejected_without_own_late_frame": sum(r.get("own_late_frames", 0) == 0 for r in rejected),
                  "latency_ms": {k: hist_percentiles(t[k], bucket_ms)
                                 for k in ("shaping", "queueing", "end_to_end", "first_frame",
                                           "first_frame_unshaped")}}
        else:
            s["delivery_delay_ms"] = hist_percentiles(t["delivery"])
        summary[side] = s
    summary["committed_agrees"] = all(r["raw"] == r["committed"] for r in rows)
    summary["certified_agrees"] = all(certified_of(r["raw"]) == r["certified"] for r in rows)
    summary["disagreements"] = [r for r in rows if r["raw"] != r["committed"] or certified_of(r["raw"]) != r["certified"]]
    summary["violations"] = [r for r in rows if any(r[k] for k in AUDITS)]
    return summary


def read_trace(path: Path) -> dict:
    p = path / "trace.json" if path.is_dir() else path
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def replay_file(args: tuple[Path, dict, str, bool]) -> tuple[list[dict], dict]:
    """One trace file's configurations in order. An online rule carries its history from one configuration to the next
    of the same run (`labels.run`), unless `fresh`: they are one run's stretches between the windows its recording
    dropped, so a deployed prover would have seen the earlier ones."""
    path, report, syncs, fresh = args
    setup = Setup(report)
    doc = read_trace(path)
    grid = {k: doc[k] for k in ("bucket_ms", "window_buckets", "frame_bytes", "queue_capacity")}
    if grid != setup.grid:
        raise ValueError(f"{path}: grid {grid} differs from the calibration's {setup.grid}")
    configs = doc.pop("configurations")
    del doc
    rows, tally, history, run = [], new_tally(), [], None
    while configs:
        cfg = configs.pop(0)
        same = not fresh and run is not None and cfg.get("labels", {}).get("run") == run
        run = cfg.get("labels", {}).get("run")
        r, t, history = replay_configuration(cfg, setup, syncs, history if same else ())
        rows += r
        add_tally(tally, t)
    return rows, tally


def parse_sync_policy(name: str) -> str:
    """`hindsight` (`calibration.choose_syncs`), `envelope` (the calibration's envelope in every window), or an online
    rule (`online.rule`)."""
    if name not in ("hindsight", "envelope"):
        online.rule(name)
    return name


def replay(paths: Iterable[Path], report: dict, jobs: int = 1, syncs: str = "hindsight",
           fresh: bool = False) -> tuple[list[dict], dict]:
    """Every configuration of the traces at `paths`, at the report's parameters, with the prover's `syncs`
    (`parse_sync_policy`). With `jobs` > 1, one file per worker."""
    tasks = [(Path(p), report, parse_sync_policy(syncs), fresh) for p in paths]
    rows, tally = [], new_tally()
    if jobs > 1 and len(tasks) > 1:
        with multiprocessing.get_context("spawn").Pool(jobs, maxtasksperchild=1) as pool:
            results = pool.map(replay_file, tasks, chunksize=1)
    else:
        results = map(replay_file, tasks)
    for r, t in results:
        rows += r
        add_tally(tally, t)
    return rows, tally


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="replay.py", description=__doc__.splitlines()[0])
    ap.add_argument("traces", nargs="+", type=Path, help="trace files (JSON, or .json.gz) or job directories")
    ap.add_argument("--calibration", required=True, type=Path, help="network-certifier-calibrate's report (JSON)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--jobs", type=int, default=1, help="trace files replayed in parallel")
    ap.add_argument("--syncs", type=parse_sync_policy, default="hindsight",
                    help="the prover's syncs: hindsight, envelope, or an online rule "
                         "(last, cover-K-M[-rhoR][-fallback], sticky-K-M-N[-rhoR])")
    ap.add_argument("--fresh", action="store_true",
                    help="an online rule starts every configuration with no history, even within one run")
    args = ap.parse_args(argv)
    report = json.loads(args.calibration.read_text())
    rows, tally = replay(args.traces, report, args.jobs, args.syncs, args.fresh)
    setup = Setup(report)
    summary = summarize(rows, tally, setup) | {"calibration": str(args.calibration), "sync_policy": args.syncs,
                                               "fresh": args.fresh, "traces": [str(p) for p in args.traces]}
    if args.syncs != "hindsight":
        summary["egress"]["calibrator_misses"] = None
    cal = report["clock_sync"]
    summary["calibrator"] = {"honest_windows": sum(e["honest_windows"] for e in report["egress"]),
                             "honest_misses": sum(e["honest_misses"] for e in report["egress"]),
                             "honest_miss_rate": cal["honest_miss_rate"]}
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "verdicts.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    e, i = summary["egress"], summary["ingress"]
    print(f"{summary['configurations']} configurations; egress {e['accepted']['raw']}/{e['link_windows']} accepted raw, "
          f"{e['accepted']['committed']} committed, {e['accepted']['certified']} certified; "
          f"ingress {i['accepted']['raw']}/{i['link_windows']} raw, {i['accepted']['committed']} committed, "
          f"{i['accepted']['certified']} certified; committed agrees: {summary['committed_agrees']}; "
          f"certified agrees: {summary['certified_agrees']}; "
          f"violations {e['violations_by_cause'] | i['violations_by_cause']}")
    return 0 if summary["committed_agrees"] and summary["certified_agrees"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
