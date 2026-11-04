"""The slim replay bundle's planner (`--replay-deferred slim`): a CPU process the Commit starts before its engine runs.

The process loads the Programs of record, indexes them and enumerates their populations (`driver.populate`: minutes for a batch-8
deployment) while the GPU works.  Once the run is committed, the Commit sends the run's seed and facts (`Planner.plan`); the process takes
the replay's sample and walks the picks against a recording reader (`driver.planned`), and returns the committed reads they make.  The
Commit then makes those reads against its own store (`c2_replay.plan`), so its GPU hold pays only for reading them.  The sample is the
replay's own (`sampled_replay`), so the reads are the ones the CPU replay will make; a read the plan missed is not in the bundle and
fails closed there.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import subprocess
import sys
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from verity_vllm.config import option


@dataclass(frozen=True, kw_only=True)
class Options:
    fd: str = option("fd", help="the Commit's end of the planner's connection (an inherited socket)")


ARGS = ("workload", "programs_root", "program_dir", "program_digest", "required_manifest", "replay_k", "replay_draw", "replay_per_stratum",
        "replay_max", "replay_target", "window_slots", "window_mb")


class Planner:
    """The planner process of one deferred Commit (pair 0 of a config run)."""

    def __init__(self, args: Any) -> None:
        self._conn, child = mp.Pipe()
        self._proc = subprocess.Popen([sys.executable, "-m", "verity_vllm.pipeline.cli", "replay-plan", str(child.fileno())], pass_fds=(child.fileno(),))
        child.close()
        self._conn.send({k: getattr(args, k, None) for k in ARGS})
        self._ready: dict[str, Any] | None = None

    def ready(self) -> dict[str, Any]:
        """The Programs' facts the Commit's linkage reads (`linkage_facts`); blocks until the process has prepared."""
        if self._ready is None:
            self._ready = _recv(self._conn)
        return self._ready

    def plan(self, facts: dict[str, Any]) -> dict[str, Any]:
        """The sample's committed reads ({"reads": [(step, name, lo, hi)], "plan": the plan record}); the process exits after it."""
        self.ready()
        self._conn.send(facts)
        try:
            return _recv(self._conn)
        finally:
            self.close()

    def close(self) -> None:
        self._conn.close()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()


def _recv(conn: Connection) -> dict[str, Any]:
    msg = conn.recv()
    err = msg.get("error")
    if err is not None:
        raise RuntimeError(f"replay planner: {err}")
    return msg


def serve(conn: Connection) -> None:
    try:
        _serve(conn, SimpleNamespace(**conn.recv()))
    except Exception as e:  # noqa: BLE001  -- the Commit names it and replays in process
        conn.send({"error": f"{type(e).__name__}: {e}"[:400]})
    finally:
        conn.close()


def _serve(conn: Connection, args: Any) -> None:
    from verity_vllm.check.replay import driver as SR
    from verity_vllm.commit.partition import require_partition
    from verity_vllm.engine import vllm_adapter as va
    from verity_vllm.pipeline import c2_replay as C2

    t0 = time.perf_counter()
    wl = va.load_workload(args.workload)
    reqs, sampling = wl["requests"], wl["sampling"]
    manifest = require_partition(json.loads(Path(args.required_manifest).read_text()), args.programs_root, args.program_dir) if args.required_manifest else None
    rl = {str(r.get("request_id")): f"r{int(r.get('index', i))}" for i, r in enumerate(reqs)}
    loaded, _digs = C2.programs_of_record(SimpleNamespace(args=args, manifest=manifest, reqs=reqs, sampling=sampling), rl, definitions=False)
    progs = {k: v["instances"] for k, v in loaded.items()}
    dirs = {k: v["dir"] for k, v in loaded.items() if v.get("dir")}
    ring = int(args.window_slots) if args.window_slots and args.window_mb else None
    q = manifest if isinstance(manifest, dict) else None
    arrive = {rl[str(r.get("request_id"))]: int(r.get("arrive_step") or 0) for r in reqs}
    memo: dict = {}
    warm = C2.cached_prefixes(reqs, rl)
    SR.populate(progs, [0], arrive, {}, ring, attention_paths=None, tp=None, program_dirs=dirs, manifest=q, memo=memo, cached_prefixes=warm)
    k0 = next(iter(progs))
    conn.send(C2.linkage_facts(loaded, progs, memo[("index", id(progs[k0]))]) | {"prepare_s": round(time.perf_counter() - t0, 3)})
    f = conn.recv()
    reads: list[tuple[int, str, int, int]] = []

    def committed(step: int, name: str, lo: int, hi: int) -> bytes:
        reads.append((int(step), str(name), int(lo), int(hi)))
        return bytes(int(hi) - int(lo))

    sr = SR.sampled_replay(progs, json.loads(Path(f["binding_map"]).read_text()), committed, None, arrivals=f["arrivals"], per_stratum=int(args.replay_per_stratum),
                           seed=f["seed"], seed_source=f["seed_source"], seed_challenge=f["seed_challenge"], max_evaluate=int(args.replay_max),
                           targets=list(args.replay_target or []), replay_k=args.replay_k, replay_draw=C2.draw_of(args), ring_slots=ring, manifest=q, program_dirs=dirs, plan=True,
                           memo=memo, cached_prefixes=warm, **f["sr_kw"])
    conn.send({"reads": list(dict.fromkeys(reads)), "plan": sr | {"reads_made": len(reads)}})



def main(o: Options) -> int:
    serve(Connection(int(o.fd)))
    return 0
