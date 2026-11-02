"""The append-only ledger: one record per submission (every checkpoint of every run), plus the launch, promotion and
final-score records the tools write.

    python -m te_core.ledger ingest RUN_DIR --agent A --direction D --mechanism M [--parent ID] [--run-id R] [--reference]
    python -m te_core.ledger index        # rebuild the local index from every agent's records
    python -m te_core.ledger publish      # put this agent's unpublished records in the evidence store

Records live under `ROOT/records/<agent>/<id>.json`, ROOT being `$TE_LEDGER` (default `~/.research/te-ledger`). A
`Ledger` writes only its own agent's directory, each record once (created exclusively, never rewritten), so agents never
contend; `index()` reads every agent's. Each record goes to the evidence store as `research data put --kind evidence/v1
--file REC --preserve`, its art id kept beside it in `<id>.art`; `ROOT/index.jsonl` is derived and rebuilt from the
records. Only the scorer (agent `scorer`) writes promotions and final (test) scores, which the index joins onto their
submissions.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, UTC
from pathlib import Path

from . import VERSION
from .score import checkpoints

SCHEMA = "te-core/submission/v1"
SCORER = "scorer"
KINDS = ("submission", "launch", "promotion", "final")
AGENT = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def default_root() -> Path:
    return Path(os.environ.get("TE_LEDGER", Path.home() / ".research" / "te-ledger"))


def now_iso(clock=time.time) -> str:
    return datetime.fromtimestamp(clock(), UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class Ledger:
    def __init__(self, root: Path | None = None, agent: str = "", *, clock=time.time, publish: bool = False,
                 research: str = "research"):
        if agent and not AGENT.match(agent):
            raise ValueError(f"agent name {agent!r}: lowercase letters, digits, - and _")
        self.root, self.agent, self.clock = Path(root or default_root()), agent, clock
        self.publish_on_write, self.research = publish, research

    @property
    def mine(self) -> Path:
        if not self.agent:
            raise ValueError("a ledger without an agent is read-only")
        return self.root / "records" / self.agent

    def append(self, kind: str, body: dict) -> dict:
        """Write one record of this agent's, exclusively; return it with its id."""
        if kind not in KINDS:
            raise ValueError(f"kind {kind!r}")
        if kind in ("promotion", "final") and self.agent != SCORER:
            raise PermissionError(f"only the scorer writes {kind} records")
        d = self.mine
        d.mkdir(parents=True, exist_ok=True)
        rec = body | {"schema": SCHEMA, "kind": kind, "agent": self.agent, "te_core": VERSION,
                      "written_at": now_iso(self.clock), "written_ts": self.clock()}
        n = len(list(d.glob("*.json")))
        while True:
            rid = f"{self.agent}-{n:06d}"
            path = d / f"{rid}.json"
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
            except FileExistsError:
                n += 1
                continue
            rec["id"] = rid
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
            break
        if self.publish_on_write:
            self.publish_one(path)
        return rec

    def publish_one(self, path: Path) -> str:
        art_file = path.with_suffix(".art")
        if art_file.exists():
            return art_file.read_text().strip()
        rec = json.loads(path.read_text())
        meta = json.dumps({"schema": SCHEMA, "id": rec["id"], "kind": rec["kind"], "agent": rec["agent"]})
        out = subprocess.run([self.research, "data", "put", "--kind", "evidence/v1", "--meta", meta, "--file",
                              str(path), "--preserve"], capture_output=True, text=True)
        art = next((w for w in out.stdout.split() if w.startswith("art:")), "")
        if out.returncode or not art:
            raise RuntimeError(f"research data put {path.name}: rc {out.returncode}: {out.stderr.strip()[-300:]}")
        art_file.write_text(art + "\n")
        return art

    def publish(self) -> list[str]:
        return [self.publish_one(p) for p in sorted(self.mine.glob("*.json"))]

    # -- reading

    def records(self) -> list[dict]:
        out = []
        for p in sorted((self.root / "records").glob("*/*.json")):
            rec = json.loads(p.read_text())
            if rec.get("agent") != p.parent.name:
                continue  # a record in another agent's directory is not that agent's: skipped
            art = p.with_suffix(".art")
            if art.exists():
                rec["art"] = art.read_text().strip()
            out.append(rec)
        return out

    def index(self) -> list[dict]:
        """Every submission, joined with its promotions and final score; written to ROOT/index.jsonl."""
        recs = self.records()
        subs = {r["id"]: dict(r, promoted=0, test_bpb=None) for r in recs if r["kind"] == "submission"}
        for r in recs:
            if r["agent"] != SCORER or r.get("entry") not in subs:
                continue
            s = subs[r["entry"]]
            if r["kind"] == "promotion":
                s["promoted"] = max(s["promoted"], int(r["tier"]))
            elif r["kind"] == "final":
                s["test_bpb"] = r["test_bpb"]
        rows = sorted(subs.values(), key=lambda r: r["id"])
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root / "index.jsonl.tmp"
        tmp.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        tmp.rename(self.root / "index.jsonl")
        return rows

    def promoted(self, entry: str, tier: int) -> bool:
        return any(r["kind"] == "promotion" and r["agent"] == SCORER and r.get("entry") == entry and int(r["tier"]) >= tier
                   for r in self.records())

    def launches(self, agent: str, tier: int, since_ts: float) -> int:
        return sum(1 for r in self.records() if r["kind"] == "launch" and r["agent"] == agent and r["tier"] == tier
                   and r["written_ts"] >= since_ts)

    # -- the scorer's records

    def promote(self, entry: str, tier: int, reason: str) -> dict:
        return self.append("promotion", {"entry": entry, "tier": int(tier), "reason": reason})

    def final(self, entry: str, test_bpb: float) -> dict:
        return self.append("final", {"entry": entry, "test_bpb": float(test_bpb)})


def coverage_of(families: list[dict]) -> dict[int, float]:
    """Each covered step's least β_t, from a run record's verified families."""
    out: dict[int, float] = {}
    for f in families:
        beta = {int(k): v for k, v in f["beta"].items()}
        for t in f["covers"]:
            if t in beta:
                out[t] = min(beta[t], out.get(t, beta[t]))
    return out


def _macs_log(record: dict, through: int | None) -> dict:
    """MACs by node class and by operation and shape, summed over the prefix's steps (setup included)."""
    by, ops = {}, {}
    steps = [record["setup"], *(s for s in record["steps"] if through is None or s["step"] <= through)]
    if through is None:
        steps.append(record.get("tail", {"by": {}, "struct": None}))
    for s in steps:
        for k, v in s.get("by", {}).items():
            by[k] = by.get(k, 0) + v
        struct = record["structures"].get(s.get("struct"))
        for row in struct["log"] if struct else []:
            key = " ".join(str(x) for x in row[:4])
            n, m = ops.get(key, (0, 0))
            ops[key] = (n + row[4], m + row[5])
    return {"by": by, "ops": {k: list(v) for k, v in sorted(ops.items(), key=lambda kv: -kv[1][1])}}


def ingest(run_dir: Path, ledger: Ledger, *, direction: str, mechanism: str, parent: str = "", run_id: str = "",
           reference: bool = False, p: float | None = None) -> list[dict]:
    """One submission record per checkpoint of the run in `run_dir` (its `run.json`)."""
    record = json.loads((Path(run_dir) / "run.json").read_text())
    ev = record.get("evaluation", {})
    cov = coverage_of(record.get("families", []))
    last = record["steps"][-1]["step"] if record["steps"] else -1
    out = []
    for sub in checkpoints(record, cov, record["c"], p):
        cp = sub["checkpoint"]
        through = None if cp["step"] >= last else cp["step"]
        grid = sub["grid"]
        body = {
            "run_id": run_id, "commit": record.get("commit", ""), "program": record.get("program", ""),
            "program_sha256": record.get("program_sha256", ""), "direction": direction, "mechanism": mechanism,
            "parent": parent, "reference": reference, "tier": record["tier"], "start": record["start"].get("name", ""),
            "start_bpb": record["start"].get("val_bpb"), "seed": record["seed"], "hardware": record.get("hardware", ""),
            "node": record.get("node", ""), "run_te_core": record.get("te_core", ""),
            "measurement_commit": ev.get("measurement_commit", ""), "island": ev.get("island"), "tree": ev.get("tree"),
            "val_slice": ({k: cp["score"]["slice"][k] for k in ("coin", "weights_digest")}
                          if "slice" in (cp.get("score") or {}) else None),
            "theta_sha256": record["theta_sha256"], "checkpoint_step": cp["step"],
            "fraction": (cp["step"] + 1) / max(record.get("planned_steps") or len(record["steps"]), 1),
            "macs": sub["macs"], "macs_log": _macs_log(record, through), "annotation": sub["annotation"],
            "c": sub["c"], "p": sub["p"], "families": record.get("families", []),
            "phi": {k: v["phi"] for k, v in grid.items()}, "depth": {k: v["depth"] for k, v in grid.items()},
            "beta": {k: v["beta"] for k, v in grid.items()}, "rho": {k: v["rho"] for k, v in grid.items()},
            "option": {k: v["option"] for k, v in grid.items()}, "val_bpb": cp.get("val_bpb"),
            "final_bpb": cp.get("test_bpb"), "output": {k: cp[k] for k in ("source", "e_bytes", "e_sha256", "format",
                                                                          "submitted", "output_sha256") if k in cp},
            "flags": record.get("flags", []), "invalid": record.get("invalid", []),
            "seconds": {k: record["totals"][k] for k in ("cpu_s", "wall_s", "gpu_s")},
        }
        out.append(ledger.append("submission", body))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--agent", default=os.environ.get("TE_AGENT", ""))
    ap.add_argument("--publish", action="store_true", help="put each record in the evidence store as it is written")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest")
    i.add_argument("run_dir", type=Path)
    i.add_argument("--direction", required=True)
    i.add_argument("--mechanism", required=True)
    i.add_argument("--parent", default="")
    i.add_argument("--run-id", default="")
    i.add_argument("--reference", action="store_true")
    i.add_argument("--p", type=float, default=None, help="bits per updated value (default: the width multiplied)")
    sub.add_parser("index")
    sub.add_parser("publish")
    a = ap.parse_args(argv)
    led = Ledger(a.root, a.agent, publish=a.publish)
    if a.cmd == "ingest":
        recs = ingest(a.run_dir, led, direction=a.direction, mechanism=a.mechanism, parent=a.parent, run_id=a.run_id,
                      reference=a.reference, p=a.p)
        print(json.dumps([{"id": r["id"], "step": r["checkpoint_step"], "macs": r["macs"], "val_bpb": r["val_bpb"],
                           "rho": r["rho"]} for r in recs]))
    elif a.cmd == "index":
        print(f"{len(led.index())} submissions in {led.root / 'index.jsonl'}")
    else:
        print("\n".join(led.publish()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
