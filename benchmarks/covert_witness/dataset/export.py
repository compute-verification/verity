"""The training-efficiency search's dataset: one row per checkpoint of every valid record, with its start, method, seed,
harness, compute, held-out loss, formalism v2's ρ inputs (each marked measured, upper bound, estimated or unmeasured),
ρ over the dial and its validity; the tables derived from them (the frontier and each branch point's envelopes, which
`training/frontier-branches` plots); and the search's qualitative record (each run's hypothesis and outcome, the board's
lines attached to runs by id, the methods, each direction's ideas and the lessons). README.md is the schema in prose,
schema.json the same as JSON Schema, and every row is checked against it before anything is written.

    python -m dataset.export --out DIR --board BOARD.md [--methods METHODS.json] [--curation FILE] [--gates FILE]
                             [--pretraining DIR] [--checkout COMMIT] [--publish] [te_core.rescore's options]

Run it from benchmarks/covert_witness with the workspace on the path (`PYTHONPATH=../..`), as te_core's commands run.
It reads te_core's ledger and the scorer's directory as te_core.rescore does: the valid records at the pin
(te_core.frontier's rules) and each checkpoint's quantities by `--quantities` (default here the analysis,
analysis/rescore_v1.py, since `measured` has only the records ingested with a traced step), te_core.formalism's stubs
where a row has none, scored by te_core.formalism and plotted by te_core.panels, so this file holds no formula of its
own. It also reads the board as tools/merge_board.py writes it (a `### YYYY-MM-DD` heading over each
day's lines; a board without them is refused), the console's methods table (`--methods`, its panel body) and the
curated record (`--curation`, default curation.json here). DIR gets one JSON Lines file per table (`TABLES`),
`manifest.json` (what went in, the counts and what each rests on), `schema.json`, the curation it read, and the gate
factors the analysis priced counted MACs at (`analysis-gates.json`, which `--gates` seeds). `--publish` puts DIR in the
evidence store (`research data put --kind evidence/v1 --tree DIR --preserve`) and prints its art id.
"""
from __future__ import annotations

import argparse
import ast
import bisect
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import jsonschema

from te_core import VERSION, formalism, frontier, panels, plots, policy, rescore
from te_core.gates import of_log, per_mac
from te_core.ledger import default_root

HERE = Path(__file__).resolve().parent
NAME = "te-dataset/v0"
SCHEMA = HERE / "schema.json"
CURATION = HERE / "curation.json"
TABLES = ("checkpoints", "left_out", "earlier", "runs", "board", "methods", "directions", "lessons", "starts",
          "branch_points", "curves")
#: the second night's search begins here: a run launched at or after it is night 2's, one before it night 1's
NIGHT2 = datetime(2026, 11, 2, 19, 15, tzinfo=UTC)
#: the harness an earlier record's witness format implies, where neither its launch record nor pin.json names one
WITNESS_HARNESS = {"covert-witness/witness/v1": "2.0", "covert-witness/witness/v2": "2.1",
                   "covert-witness/witness/v3": "2.1", "covert-witness/witness/v4": "2.2"}
#: a final rerun row stands for the search checkpoint at the nearest MACs, within this share of either
MACS_MATCH = 0.02
#: the bases of a quantity, best first; a value derived from several takes the worst of theirs
BASES = ("measured", "upper-bound", "estimated", "unmeasured")
MEASURED = {rescore.LABELS["gates"], rescore.LABELS["token_gates"], formalism.V23["gates"], formalism.V23["token_gates"]}
W_FWD = "the start's forward MACs per token at the bf16 GEMM price (te_core.panels.w_fwd)"
UPDATE_CODE = ("Fit's update clause reads Delta, the update code; this run trains fp32 latents (straight-through, or "
               "fp32 masters coded at each checkpoint), which its update block must read, so Fit here is optimistic")
STUBBED = "every quantity from te_core.formalism's stubs on the row's own fields: no analysis or trace has this checkpoint"
POOLED = ("the sealed final row nearest this checkpoint pools the reruns of more than one checkpoint (its seeds "
          "repeat), so its min-max spread is not seed noise")
FIX_BOARD = "write a dated board: python3 benchmarks/covert_witness/tools/merge_board.py --out FILE"
DEVIATIONS = [
    "The schema's prose is README.md, not SCHEMA.md: the repository allows a fixed set of Markdown names "
    "(tests/test_repository.py), and README.md is the one a directory's own description takes.",
    "Tables are JSON Lines, not Parquet: pyarrow is not in the workspace, and each line is one row of its table "
    "with the fields schema.json gives.",
]
CAVEATS = [
    "Fit's update-application clause is checked as Delta + Delta_grad <= k with Delta the update code, as te_core "
    "prices it; the clause as ruled reads Delta_t, what the update block reads, which te_core does not price yet. A "
    "run of family update-code trains fp32 latents (straight-through QAT and coded factors, or fp32 masters coded at "
    "each checkpoint), which its update block reads, so its Fit is optimistic; those rows say so in `caveats`.",
    "One seed per record: the search's losses are single runs; the sealed final reruns (three seeds) check the "
    "entries they cover, and only night 1's finals exist.",
    "The v2.3 harness is the pilot of the next campaign's measurement, not its result: its counter predates "
    "te_core 4.0's update dependence, and every rho input it lacks is a labelled stub or assumption.",
    "C0 for every start is estimated: its training tokens at dense AdamW's counted gates per token (te_core.panels), "
    "since no start's shape carries counted training gates.",
    "For scale, the served demo block is c = 3.0e13 gates (a lower bound) at k = 145,568 bits; the dial's cells "
    "are relative to each start's forward.",
]
#: the caveat each `--quantities` mode adds
QUANTITIES = {
    rescore.ANALYSED: "Every scored checkpoint's rho inputs are the analysis's (analysis/rescore_v1.py), not measured "
                      "on its traced step: they spread a token's work evenly, where the trace finds most of it reads "
                      "no update value, so on full-training routes the analysis's rho is far above the one measured "
                      "under A1, which charges no block for earlier tokens' keys and values. `--quantities measured` "
                      "replaces them for every row ingested with --structures.",
    rescore.MEASURED: "Every scored checkpoint's rho inputs are measured on its record's traced step (te_core.capture); "
                      "A1 charges no block for earlier tokens' keys and values unless the export ran with "
                      "--charge-context.",
    rescore.STUBS: "Every checkpoint's rho inputs are te_core.formalism's stubs on the row's own fields.",
}

RUN_ID = re.compile(r"r(\d{8})-(\d{6})-([0-9a-f]{4})")
ENTRY = re.compile(r"^(\d{2}):(\d{2})Z (\S+)(?: \[([A-Z]+)\])? ?(.*)$")
DAY = re.compile(r"^### (\d{4})-(\d{2})-(\d{2})\s*$")
RUN_REF = re.compile(r"(?<![\w-])r(\d{8})-(\d{6})-([0-9a-f]{4})(?![\w-])(?:@(\d+))?")
SHORT_RUN = re.compile(r"(?<![\w-])-(\d{6})-([0-9a-f]{4})(?![\w-])(?:@(\d+))?")
ART_REF = re.compile(r"(?<![\w-])art:([0-9a-f]{6,64})(?![\w-])(?:@(\d+))?")
HEX_REF = re.compile(r"(?<![\w:/.-])([0-9a-f]{10,64})(?![\w-])(?:@(\d+))?")
LEDGER_REF = re.compile(r"(?<![\w])([a-z][a-z0-9_]*(?:-[a-z][a-z0-9_]*)*-(\d{6}))(?:\.\.(\d{1,6}))?(?:@(\d+))?")
SUFFIX_REF = re.compile(r"(?<![\w-])([0-9a-f]{4})(?![\w-])")


def parser() -> argparse.ArgumentParser:
    ap = rescore.arguments(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    ap.set_defaults(quantities=rescore.ANALYSED)
    ap.add_argument("--board", type=Path, required=True, help="the dated board (tools/merge_board.py --out)")
    ap.add_argument("--methods", type=Path, default=None, help="the console's training/methods panel body")
    ap.add_argument("--curation", type=Path, default=CURATION)
    ap.add_argument("--pretraining", type=Path, default=None,
                    help=f"te_core.panels' pretraining runs (default {panels.PRETRAINING}, fetched)")
    ap.add_argument("--gates", type=Path, default=None, help="the analysis's gate factors, copied into DIR first")
    ap.add_argument("--checkout", default=None, help="the commit this export ran from (default this checkout's HEAD)")
    ap.add_argument("--publish", action="store_true", help="put DIR in the evidence store and print its art id")
    ap.add_argument("--research", default="research", help="the research command --publish runs")
    return ap


# ---- small helpers -----------------------------------------------------------------------------------------------------

def parse_time(v: str | None) -> datetime | None:
    return datetime.fromisoformat(v.replace("Z", "+00:00")).astimezone(UTC) if v else None


def iso(t: datetime | None) -> str | None:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ") if t else None


def night_of(t: datetime | None) -> int | None:
    return None if t is None else 2 if t >= NIGHT2 else 1


def run_stamp(run: str | None) -> datetime | None:
    m = RUN_ID.fullmatch(run or "")
    return datetime.strptime(m[1] + m[2], "%Y%m%d%H%M%S").replace(tzinfo=UTC) if m else None


def literal(v):
    """A launch record's Python-literal field (`trainable`, `train_args`) as data; one that doesn't parse is kept as
    its text."""
    if not isinstance(v, str):
        return v
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError) as e:
        return {"unparsed": v, "error": type(e).__name__}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tag(text: str | None) -> str | None:
    """A direction or mechanism in launch.py's form (its `tag`), the ledger's; a launch record written before launch.py
    applied it keeps the raw text (`qlora+gains`)."""
    return re.sub(r"[^a-z0-9_-]", "", text.lower())[:64] if text else text


def method_key(direction: str | None, mechanism: str | None) -> str:
    return f"{tag(direction)}/{tag(mechanism)}"


def basis_of(label: str | None, value) -> str:
    """A quantity's basis from te_core's label for it (README, "Conventions")."""
    if value is None:
        return "unmeasured"
    if label and (label.startswith(("measured", "counted")) or label in MEASURED):
        return "measured"
    if label and label.startswith("upper bound"):
        return "upper-bound"
    return "estimated"


def worst(*bases: str) -> str:
    return max(bases, key=BASES.index)


def q(value, label: str | None, basis: str | None = None) -> dict:
    return {"value": value, "basis": basis or basis_of(label, value), "source": label}


def finite(x, count: list[int]):
    """`x` with every non-finite float null (counted in `count`): JSON has no infinity."""
    if isinstance(x, float) and not math.isfinite(x):
        count[0] += 1
        return None
    if isinstance(x, dict):
        return {k: finite(v, count) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite(v, count) for v in x]
    return x


# ---- the scorer's state ------------------------------------------------------------------------------------------------

def scorer_state(d: Path) -> dict:
    """The launch records (by art id and by run, the latest recorded), runs.json by run, the submissions, the pins
    pin.json names, the starts' tensors and when the scorer last collected."""
    launch: dict[str, dict] = {}
    for line in (d / "index.jsonl").open():
        r = json.loads(line)
        if r.get("source") == "launch.py":
            launch[r["id"]] = r
    by_run = {}
    for r in sorted(launch.values(), key=lambda r: r.get("recorded") or ""):
        if r.get("run"):
            by_run[r["run"]] = r
    pin = json.loads((d / "pin.json").read_text()) if (d / "pin.json").exists() else {}
    collect = json.loads((d / "collect.json").read_text()) if (d / "collect.json").exists() else {}
    return {"launch": launch, "by_run": by_run, "runs": {r["run"]: r for r in json.loads((d / "runs.json").read_text())},
            "subs": json.loads((d / "submissions.json").read_text()), "pins": pin.get("pins", {}),
            "starts": json.loads((d / "starts.json").read_text()), "collected_at": collect.get("at")}


def finals(d: Path, launch: dict[str, dict]) -> tuple[dict[str, dict], dict]:
    """Each search run with sealed final reruns (a `final*/` directory with final_submissions.json and table.json):
    its set, entry, the final starts' l0 and the rerun rows; and what was read."""
    out, read = {}, {"sets": [], "rows": 0, "entries": 0, "entries_without_launch": []}
    for f in sorted(p for p in d.glob("final*") if p.is_dir()):
        if not ((f / "final_submissions.json").exists() and (f / "table.json").exists()):
            continue
        tab = json.loads((f / "table.json").read_text())
        rows = json.loads((f / "final_submissions.json").read_text())
        read["sets"].append(f.name)
        read["rows"] += len(rows)
        for x in rows:
            run = (launch.get(x["id"]) or {}).get("run")
            if run is None:
                read["entries_without_launch"].append(x["id"])
                continue
            e = out.setdefault(run, {"set": f.name, "entry": x["id"], "final_l0": tab.get("final_starts", {}),
                                     "search_l0": tab.get("search_starts", {}), "rows": []})
            e["rows"].append(x)
    read["entries"] = len(out)
    read["entries_without_launch"] = sorted(set(read["entries_without_launch"]))
    return out, read


def final_of(r: dict, fin: dict[str, dict]) -> dict | None:
    """The sealed final reruns' value at row `r`'s checkpoint: the rerun row at the nearest MACs, within MACS_MATCH."""
    e = fin.get(r.get("run_id"))
    if not e or r.get("macs") is None:
        return None
    x = min(e["rows"], key=lambda x: abs(x["macs"] - r["macs"]))
    if abs(x["macs"] - r["macs"]) > MACS_MATCH * max(x["macs"], r["macs"]):
        return None
    l0 = e["final_l0"].get(r["start"])
    return {"set": e["set"], "entry": e["entry"], "rerun": x.get("run"), "mean_bpb": x["val_bpb"],
            "min_bpb": x.get("bpb_min"), "max_bpb": x.get("bpb_max"), "seeds": x.get("seeds"), "final_l0": l0,
            "drop": 1 - x["val_bpb"] / l0 if l0 else None, "macs": x["macs"]}


def parameters(trainable, tensors: dict) -> dict:
    """The trainable set's counts: full tensors, adapters (rank x (d_in + d_out) each) and their parameters."""
    if not isinstance(trainable, dict) or "unparsed" in trainable:
        return {"tensors": None, "adapters": None, "parameters": None}
    shapes = {n: s for n, (_, s) in tensors.items()}
    full = [n for n in trainable.get("full", []) if n in shapes]
    adapters = {n: int(r) for n, r in (trainable.get("adapters") or {}).items() if n in shapes}
    n = sum(math.prod(shapes[t]) for t in full) + sum(r * sum(shapes[t][:2]) for t, r in adapters.items())
    return {"tensors": len(full), "adapters": len(adapters), "parameters": n}


# ---- the board ---------------------------------------------------------------------------------------------------------

def board_entries(text: str) -> list[dict]:
    """The board's entries under `## Entries`, each dated by its day's heading; exact copies collapse into one."""
    lines = text.splitlines()
    start = next((i for i, x in enumerate(lines) if x.strip() == "## Entries"), None)
    if start is None:
        raise SystemExit(f"dataset: the board has no `## Entries` section: {FIX_BOARD}")
    day, out, cur = None, {}, None
    for line in lines[start + 1:]:
        if line.startswith("## "):
            break
        if m := DAY.match(line):
            day, cur = (int(m[1]), int(m[2]), int(m[3])), None
            continue
        if m := ENTRY.match(line):
            if day is None:
                raise SystemExit(f"dataset: the board's entries have no day heading (`### YYYY-MM-DD`): {FIX_BOARD}")
            cur = {"at": datetime(*day, int(m[1]), int(m[2]), tzinfo=UTC), "agent": m[3], "tag": m[4], "text": m[5]}
            key = (cur["at"], cur["agent"], cur["tag"], cur["text"])
            if key in out:
                out[key]["copies"] += 1
                cur = None
            else:
                cur["copies"] = 1
                out[key] = cur
        elif cur is not None and line.strip():
            cur["text"] += "\n" + line
    if not out:
        raise SystemExit(f"dataset: the board has no entries: {FIX_BOARD}")
    rows = sorted(out.values(), key=lambda e: (e["at"], e["agent"], e["text"]))
    for e in rows:
        digest = hashlib.sha256(f"{e['tag']} {e['text']}".encode()).hexdigest()[:6]
        e["id"] = f"{e['at']:%Y%m%dT%H%MZ}-{e['agent']}-{digest}"
    return rows


class Index:
    """What the board's ids resolve against: runs, arts and ledger ids, each with the run it belongs to."""

    def __init__(self, runs: dict[str, set[str]], arts: dict[str, str | None], ledger: dict[str, dict]):
        self.runs, self.ledger = runs, ledger
        self.arts = sorted(arts)
        self.art_run = arts
        self.by_suffix: dict[str, list[str]] = defaultdict(list)
        self.by_short: dict[str, list[str]] = defaultdict(list)
        for run in runs:
            if m := RUN_ID.fullmatch(run):
                self.by_suffix[m[3]].append(run)
                self.by_short[f"{m[2]}-{m[3]}"].append(run)
        self.by_number: dict[str, list[str]] = defaultdict(list)
        for lid in ledger:
            self.by_number[lid[-6:]].append(lid)
        self.steps: dict[tuple[str, int], list[str]] = defaultdict(list)
        for lid, r in ledger.items():
            if r.get("run_id") and r.get("checkpoint_step") is not None:
                self.steps[(r["run_id"], int(r["checkpoint_step"]))].append(lid)

    def art(self, prefix: str) -> str | None:
        """The one known art whose hex starts with `prefix`."""
        i = bisect.bisect_left(self.arts, prefix)
        if i < len(self.arts) and self.arts[i].startswith(prefix) and \
                not (i + 1 < len(self.arts) and self.arts[i + 1].startswith(prefix)):
            return self.arts[i]
        return None

    def ledger_id(self, token: str) -> str | None:
        """The one ledger id equal to `token` or ending with it (a route's last 12 characters)."""
        if token in self.ledger:
            return token
        hits = [x for x in self.by_number.get(token[-6:], []) if x.endswith(token)]
        return hits[0] if len(hits) == 1 else None

    def at_step(self, run: str | None, step: str | None) -> list[str]:
        return self.steps.get((run, int(step)), []) if run and step else []

    def owner(self, run: str, agent: str) -> bool:
        return agent in self.runs.get(run, set())


def links(e: dict, ix: Index) -> dict:
    """The runs, checkpoints (ledger ids) and arts an entry names: full run ids; `-HHMMSS-hhhh` with the day of the
    last full run id before it (else the entry's); a bare `hhhh` only when it is the suffix of exactly one of the
    entry's agent's own runs; `art:` prefixes and bare hex of 10 or more digits by a unique known art; ledger ids, or
    their last characters, and ranges `X-000101..120`; and a checkpoint `@step` after a run, art or ledger id. A full
    run id or an `art:` prefix that resolves to nothing is kept as `unresolved`."""
    text = e["text"]
    runs, cps, arts, unresolved = set(), set(), set(), set()
    taken = [False] * len(text)

    def free(m) -> bool:
        if any(taken[m.start():m.end()]):
            return False
        taken[m.start():m.end()] = [True] * (m.end() - m.start())
        return True

    day = f"{e['at']:%Y%m%d}"
    for m in RUN_REF.finditer(text):
        if free(m):
            run = f"r{m[1]}-{m[2]}-{m[3]}"
            day = m[1]
            if run in ix.runs:
                runs.add(run)
                cps.update(ix.at_step(run, m[4]))
            else:
                unresolved.add(run)
    for m in SHORT_RUN.finditer(text):
        if free(m):
            run = f"r{day}-{m[1]}-{m[2]}"
            hits = [run] if run in ix.runs else ix.by_short.get(f"{m[1]}-{m[2]}", [])
            if len(hits) == 1:
                runs.add(hits[0])
                cps.update(ix.at_step(hits[0], m[3]))
    for pattern in (ART_REF, HEX_REF):
        for m in pattern.finditer(text):
            if pattern is HEX_REF and not re.search(r"[a-f]", m[1]):
                continue
            if not free(m):
                continue
            if art := ix.art(m[1]):
                arts.add(f"art:{art}")
                if run := ix.art_run.get(art):
                    runs.add(run)
                    cps.update(ix.at_step(run, m[2]))
            elif pattern is ART_REF:
                unresolved.add(f"art:{m[1]}")
    for m in LEDGER_REF.finditer(text):
        if not free(m):
            continue
        first = ix.ledger_id(m[1])
        if first is None:
            continue
        ids = [first]
        if m[3]:
            n0, width = int(m[2]), len(m[2])
            end = int(m[2][:width - len(m[3])] + m[3])
            stem = first[:-width]
            ids = [f"{stem}{n:0{width}d}" for n in range(n0, end + 1) if f"{stem}{n:0{width}d}" in ix.ledger] or ids
        for lid in ids:
            cps.add(lid)
            if run := ix.ledger[lid].get("run_id"):
                runs.add(run)
    for m in SUFFIX_REF.finditer(text):
        s = m[1]
        if not (re.search(r"[a-f]", s) and re.search(r"\d", s)) or any(taken[m.start():m.end()]) or \
                re.fullmatch(r"\d+e\d+", s) or text[m.start() - 1:m.start()] == ".":
            continue
        mine = [r for r in ix.by_suffix.get(s, []) if ix.owner(r, e["agent"])]
        if len(mine) == 1:
            runs.add(mine[0])
    return {"runs": sorted(runs), "checkpoints": sorted(cps), "arts": sorted(arts), "unresolved": sorted(unresolved)}


# ---- quantities --------------------------------------------------------------------------------------------------------

def rho_inputs(m: formalism.Measured, shape: dict) -> dict:
    """The formalism's inputs of one checkpoint, each {value, basis, source} (README, "The tables")."""
    lab = m.labels
    affected = sum(p.affected_gates for p in m.parts)
    out = {
        "C_gates": q(m.gates, lab.get("gates")),
        "tokens": q(m.tokens, lab.get("tokens")),
        "token_gates": q(m.token_gates, lab.get("token_gates")),
        "affected_gates": q(affected, lab.get("affected_gates")),
        "tau_bits": q(m.tau_bits, lab.get("tau_bits")),
        "a_bits": q(m.a_bits, lab.get("a_bits")),
        "delta_bits": q(m.delta_bits, lab.get("delta_bits")),
        "u1_bits": q(m.u1_bits, lab.get("u1_bits")),
        "u_points": q([list(p) for p in m.u], lab.get("u")),
        "u_in_points": q([list(p) for p in m.u_in], lab.get("u_in")),
        "tail_share": q(m.tail_share, lab.get("u"), "estimated"),
        "d_gates": q(m.decoder_gates if m.decoder_gates is not None else 0.0, lab.get("decoder_gates"),
                     None if m.decoder_gates is not None else "estimated"),
        "delta_grad_bits": q(m.delta_grad_bits, lab.get("delta_grad_bits")),
        "w_fwd_gates": q(panels.w_fwd(shape), W_FWD, "estimated"),
    }
    b = {k: v["basis"] for k, v in out.items()}
    out["sigma"] = q(m.sigma, "token_gates / C_gates", worst(b["token_gates"], b["C_gates"]))
    out["w_gates"] = q(m.w, "token_gates / tokens", worst(b["token_gates"], b["tokens"]))
    out["alpha_max"] = q(m.alpha_max, "the largest per-token affected gates over the parts",
                         worst(b["affected_gates"], b["tokens"]))
    out["alpha_mean"] = q(m.alpha_mean, "affected_gates / tokens", worst(b["affected_gates"], b["tokens"]))
    out["parts"] = [{"gates": p.gates, "tokens": p.tokens, "token_gates": p.token_gates,
                     "affected_gates": p.affected_gates, "steps": p.steps, "delta_grad_bits": p.delta_grad_bits}
                    for p in m.parts]
    if m.context is not None or "context" in lab:
        out["context"] = q(m.context, lab.get("context"), "upper-bound" if m.context is not None else None)
    return out


def dial_cells(dial: dict) -> list[dict]:
    out = []
    for (n, kb), v in sorted(dial.items()):
        out.append({"c_tokens_per_block": n, "k_bytes": kb, "c_gates": v["c"], "k_bits": v["k"], "rho": v["rho"],
                    "basis": "fit" if v["fit"] else "target", "rho_fit": v["rho_fit"], "slowdown": v["slowdown"],
                    "x0": v["x0"], "s_star": v["s_star"], "x0_known": v["x0_known"],
                    "s_star_known": v["s_star_known"], "delta_counted": v["delta_counted"],
                    "fit_parts": [p["fit"] for p in v["parts"]]} | {k: v[k] for k in ("s_star_a1",) if k in v})
    return out


def harness_of(r: dict, launch: dict | None, pins: dict) -> str | None:
    """A record's harness version: its launch record's, else its measurement commit's in pin.json, else the commit it
    ran at, else its witness format's; None when none of them names one."""
    launch = launch or {}
    if launch.get("harness_version"):
        return launch["harness_version"]
    for commit in (r.get("measurement_commit"), launch.get("measurement_commit"), r.get("commit"),
                   launch.get("commit")):
        if commit and (v := (pins.get(commit) or {}).get("version")):
            return v
    return WITNESS_HARNESS.get(launch.get("witness"))


# ---- the export --------------------------------------------------------------------------------------------------------

class Export:
    """Every table, built from the ledger, the scorer's state, the board, the methods table and the curation."""

    def __init__(self, a, d: dict):
        self.a, self.d = a, d
        self.rows, self.pin, self.shapes = d["rows"], d["pin"], d["shapes"]
        self.keep, self.source, self.use = d["keep"], d["source"], d["use"]
        self.mod = rescore.load_analysis(a.analysis) if self.use else None
        self.s = scorer_state(a.scorer)
        self.fin, self.fin_read = finals(a.scorer, self.s["launch"])
        self.left = frontier.status(self.rows, self.pin)
        valid = [r for r in self.rows if r["id"] not in self.left]
        kept = [r for r in valid if self.keep is None or self.keep(r)]
        self.scored, self.skipped = frontier.formalism_rows(kept, self.shapes)
        rest = [r for r in valid if self.keep is not None and not self.keep(r)]
        self.stubbed, more = frontier.formalism_rows(rest, self.shapes)
        self.skipped |= more
        self.by_id = {r["id"]: r for r in self.rows}
        self.stamp = max((r.get("written_at") or "" for r in self.rows), default="") or None
        self.sched = {(x["id"], x["macs"]): x.get("schedule_end") for x in self.s["subs"]}

    # -- runs and nights

    def launch_of(self, run: str | None) -> dict | None:
        return self.s["by_run"].get(run) if run else None

    def when(self, run: str | None, fallback: str | None = None) -> datetime | None:
        launch, rec = self.launch_of(run) or {}, self.s["runs"].get(run) or {}
        return (parse_time(launch.get("launched")) or parse_time(rec.get("start_utc")) or run_stamp(run)
                or parse_time(fallback))

    def config(self, launch: dict | None, start: str | None) -> dict:
        """A run's plain configuration from its launch record: trainable set and its counts, arguments, update
        code, tokens, and (from the analysis) its optimizer, Delta_grad and family."""
        if not launch:
            return {}
        trainable, args = literal(launch.get("trainable")), literal(launch.get("train_args"))
        tensors = (self.s["starts"].get(start) or {}).get("tensors", {})
        out = {"trainable": trainable, "counts": parameters(trainable, tensors), "args": launch.get("args"),
               "train_args": args, "E_format": launch.get("E_format"), "E_bytes": launch.get("E_bytes"),
               "steps": launch.get("steps"), "tokens": launch.get("tokens"), "full_theta": launch.get("full_theta"),
               "lowest_updated_layer": launch.get("lowest_updated_layer"), "optimizer": None,
               "optimizer_state_bits": None, "family": None}
        if self.mod is not None and isinstance(trainable, dict) and "unparsed" not in trainable and \
                start in self.s["starts"] and launch.get("direction"):
            out["optimizer"] = self.mod.optimizer(launch)
            out["optimizer_state_bits"] = self.mod.delta_grad(launch, self.s["starts"][start], trainable)[0]
            out["family"] = self.mod.family({"mechanism": launch.get("mechanism"), "E_format": launch.get("E_format"),
                                             "optimizer": out["optimizer"], "direction": launch["direction"]})
        return out

    def complete(self, r: dict) -> bool:
        """Whether a checkpoint ends its schedule: the scorer's schedule_end where its submission joins, else the
        ledger's fraction of the planned steps."""
        launch = self.launch_of(r.get("run_id"))
        end = self.sched.get((launch["id"], r.get("macs"))) if launch else None
        return bool(end) if end is not None else (r.get("fraction") or 0) >= 1

    # -- the checkpoints

    def checkpoint(self, r: dict, quantities: str, in_panel: bool, info: dict, methods: dict) -> dict:
        shape = self.shapes[r["start"]]
        m: formalism.Measured = r["measured"]
        launch = self.launch_of(r.get("run_id"))
        rec = self.s["runs"].get(r.get("run_id")) or {}
        t = self.when(r.get("run_id"), r.get("written_at"))
        if "dial" not in r:
            r["dial"] = panels.dial(m, panels.w_fwd(shape))
        w = panels.w_fwd(shape)
        k_run, k_train = formalism.fit_k(m, panels.DEFAULT[0] * w, w)
        cfg = self.config(launch, r["start"])
        family = r.get("family") or cfg.get("family")
        key = method_key(r.get("direction"), r.get("mechanism"))
        args = cfg.get("train_args") if isinstance(cfg.get("train_args"), dict) else {}
        c0 = info["c0"].get(r["start"])
        final = final_of(r, self.fin)
        seeds = (final or {}).get("seeds") or []
        caveats = ([UPDATE_CODE] if family == "update-code" else []) + ([STUBBED] if quantities == "stubs" else []) + (
            [POOLED] if len(seeds) != len(set(seeds)) else [])
        start_bpb = r.get("start_bpb")
        return {
            "id": r["id"], "run": r.get("run_id") or None, "checkpoint": r.get("checkpoint_step"),
            "fraction": r.get("fraction"), "schedule_end": self.complete(r), "night": night_of(t),
            "launched": iso(t), "agent": r.get("agent"), "direction": r.get("direction"),
            "mechanism": tag(r.get("mechanism")), "family": family,
            "start": {"id": r["start"], "label": plots.start_label(shape), "parameters": shape.get("parameters"),
                      "tokens": shape.get("tokens"), "tokens_per_param": plots.saturation(shape),
                      "l0": info["l0"].get(r["start"]), "c0_gates": c0, "c0_basis": info["c0_basis"].get(r["start"])},
            "method": {"key": key, "label": methods.get(key, {}).get("label")},
            "config": {"optimizer": cfg.get("optimizer"), "optimizer_state_bits": cfg.get("optimizer_state_bits"),
                       "code": {"format": (r.get("output") or {}).get("format") or cfg.get("E_format"),
                                "bytes": (r.get("output") or {}).get("e_bytes")},
                       "trainable": cfg.get("counts"), "full_theta": cfg.get("full_theta"),
                       "lowest_updated_layer": cfg.get("lowest_updated_layer")},
            "seed": {"run": r.get("seed"), "data": args.get("data_seed")},
            "harness": {"version": harness_of(r, launch, self.s["pins"]),
                        "measurement_commit": r.get("measurement_commit"), "run_te_core": r.get("run_te_core"),
                        "te_core": r.get("te_core"), "witness": (launch or {}).get("witness"),
                        "commit": r.get("commit"), "program": r.get("program"),
                        "program_sha256": r.get("program_sha256"), "island": r.get("island"),
                        "island_tree": (launch or {}).get("island_tree") or (r.get("island") or {}).get("tree"),
                        "node": r.get("node"), "hardware": r.get("hardware")},
            "compute": {"gates": m.gates, "macs": r.get("macs"),
                        "flops": 2 * r["macs"] if r.get("macs") is not None else None,
                        "basis": basis_of(m.labels.get("gates"), m.gates)},
            "loss": {"search_bpb": r.get("val_bpb"), "start_bpb": start_bpb,
                     "drop": 1 - r["val_bpb"] / start_bpb if start_bpb else None, "final": final},
            "quantities": quantities, "rho_inputs": rho_inputs(m, shape), "dial": dial_cells(r["dial"]),
            "fit_k": {"c_gates": panels.DEFAULT[0] * w, "k_run_bits": k_run, "k_train_bits": k_train},
            "validity": {"status": "valid", "flags": r.get("flags") or [], "invalid": r.get("invalid") or [],
                         "failed_checks": (launch or {}).get("failed_checks"),
                         "launch_valid": (launch or {}).get("valid"), "cancelled": rec.get("cancelled"),
                         "process_tree": r.get("tree")},
            "in_panel": in_panel,
            "arts": {"record": r.get("art"), "launch": (launch or {}).get("id"), "run_record": rec.get("run_record")},
            "caveats": caveats,
        }

    def left_out(self) -> list[dict]:
        out = []
        reasons = self.left | self.skipped
        for r in self.rows:
            if r["id"] not in reasons:
                continue
            launch = self.launch_of(r.get("run_id"))
            t = self.when(r.get("run_id"), r.get("written_at"))
            out.append({"id": r["id"], "run": r.get("run_id") or None, "checkpoint": r.get("checkpoint_step"),
                        "night": night_of(t), "agent": r.get("agent"), "direction": r.get("direction"),
                        "mechanism": tag(r.get("mechanism")), "start": r.get("start"),
                        "measurement_commit": r.get("measurement_commit") or None,
                        "harness": harness_of(r, launch, self.s["pins"]), "macs": r.get("macs"),
                        "val_bpb": r.get("val_bpb"), "reason": reasons[r["id"]], "art": r.get("art")})
        return out

    def earlier(self, ledger_by_run_macs: dict) -> list[dict]:
        """Every scorer submission of an earlier regime (the harness before v2.3): its compute in gates (the ledger's
        counted ops where a row joins by run and MACs, else every MAC at the bf16 GEMM price), loss and credit."""
        out = []
        for x in self.s["subs"]:
            if x.get("regime") == "current":
                continue
            launch = self.s["launch"].get(x["id"]) or {}
            run = launch.get("run")
            joined = ledger_by_run_macs.get((run, x.get("macs")))
            if joined is not None:
                g = of_log(joined["macs_log"])["gates"]
                gates = q(g, "the ledger's counted ops in gates (te_core.gates), joined by run and MACs", "measured")
            else:
                gates = q(x["macs"] * per_mac("gemm/bf16") if x.get("macs") is not None else None,
                          "every MAC at the bf16 GEMM price (no ledger row)", "estimated")
            t = self.when(run)
            out.append({"id": f"{x['id']}@{x.get('cp')}", "entry": x["id"], "cp": x.get("cp"), "run": run,
                        "start": x.get("start"), "regime": x.get("regime"),
                        "harness": {"version": harness_of({}, launch, self.s["pins"]),
                                    "witness": launch.get("witness"), "te_core": launch.get("te_core"),
                                    "measurement_commit": launch.get("measurement_commit") or None,
                                    "commit": launch.get("commit") or None},
                        "direction": x.get("direction") or launch.get("direction"),
                        "mechanism": tag(launch.get("mechanism")), "agent": launch.get("agent"), "macs": x.get("macs"),
                        "gates": gates, "val_bpb": x.get("val_bpb"), "E_bytes": x.get("E_bytes"),
                        "counted": x.get("counted"), "schedule_end": x.get("schedule_end"),
                        "ledger_id": joined["id"] if joined else None,
                        "ledger_status": (self.left.get(joined["id"], "valid") if joined else None),
                        "night": night_of(t)})
        return out


def build(a) -> tuple[dict[str, list[dict]], dict]:
    """Every table and the manifest."""
    if a.gates:
        a.out.mkdir(parents=True, exist_ok=True)
        if a.gates.resolve() != (a.out / "analysis-gates.json").resolve():
            shutil.copyfile(a.gates, a.out / "analysis-gates.json")
    d = rescore.load(a)
    x = Export(a, d)
    pre_dir = a.pretraining or panels.fetch(panels.PRETRAINING)
    runs_pre = panels.pretraining(pre_dir)
    what = a.source or (f"python -m dataset.export ({NAME}; te_core {VERSION}; pin "
                        f"{' + '.join(c[:12] for c in x.pin)}; {len(x.scored)} analysed checkpoints)")
    p, info = panels.build(x.scored, x.shapes, runs_pre, what, x.stamp)
    branches = {b.name: b for b in info["branches"]}
    final_l0 = {}
    for e in x.fin.values():
        final_l0 |= {k: v for k, v in e["final_l0"].items() if v is not None}
    c0_basis = {}
    for s, shape in x.shapes.items():
        c0_basis[s] = "measured" if policy.c0(shape)[1] == "counted" else "estimated"
    ctx = {"l0": {s: b.l0 for s, b in branches.items()}, "c0": {s: b.c0 for s, b in branches.items()},
           "c0_basis": c0_basis}
    for s in x.shapes:
        if s not in branches:
            on = [r for r in x.stubbed if r["start"] == s]
            ctx["l0"][s] = frontier.start_bpb(on)
            ctx["c0"][s] = panels.c0(x.shapes[s], info["rate"])

    # board and the ids it resolves against
    text = a.board.read_text()
    entries = board_entries(text)
    owners: dict[str, set[str]] = defaultdict(set)
    arts: dict[str, str | None] = {}

    def own(run, *names):
        for n in names:
            if n:
                owners[run].add(n)
                if m := re.fullmatch(r"cursor/(?:te-)?(.+?)-[0-9a-f]{4}", n):
                    owners[run].add(m[1])

    for launch in x.s["launch"].values():
        if launch.get("run"):
            own(launch["run"], launch.get("direction"), launch.get("agent"))
            arts[launch["id"].removeprefix("art:")] = launch["run"]
    for run, rec in x.s["runs"].items():
        own(run, rec.get("direction"), rec.get("agent"))
        if rec.get("run_record"):
            arts[rec["run_record"].removeprefix("art:")] = run
    for r in x.rows:
        if r.get("run_id"):
            own(r["run_id"], r.get("agent"), r.get("direction"))
        if r.get("art"):
            arts[r["art"].removeprefix("art:")] = r.get("run_id") or None
    for e in x.fin.values():
        for row in e["rows"]:
            if row.get("run"):
                own(row["run"], "scorer")
    for f in sorted(p_ for p_ in a.scorer.glob("final*") if p_.is_dir()):
        if (f / "collected.json").exists():
            for run, art in json.loads((f / "collected.json").read_text()).items():
                own(run, "scorer")
                arts[art.removeprefix("art:")] = run
    ix = Index(owners, arts, x.by_id)
    board = []
    for e in entries:
        board.append({"id": e["id"], "at": iso(e["at"]), "night": night_of(e["at"]), "agent": e["agent"],
                      "tag": e["tag"], "text": e["text"], **links(e, ix), "copies": e["copies"]})
    by_run: dict[str, list[dict]] = defaultdict(list)
    for b in board:
        for run in b["runs"]:
            by_run[run].append(b)

    # methods
    panel = json.loads(a.methods.read_text()) if a.methods else {"rows": []}
    pcols = panel.get("columns", ["method", "label", "description"])
    prow = {"/".join(map(tag, r[pcols.index("method")].split("/", 1))): dict(zip(pcols, r))
            for r in panel.get("rows", [])}
    cur = json.loads(a.curation.read_text()) if a.curation and a.curation.exists() else {}
    keys = sorted({method_key(r.get("direction"), r.get("mechanism")) for r in x.rows if r.get("direction")}
                  | {method_key(v.get("direction"), v.get("mechanism")) for v in x.s["launch"].values()
                     if v.get("direction")})
    if stale := sorted(set(cur.get("methods") or {}) - set(keys)):
        raise SystemExit(f"dataset: curation.json describes methods no record has: {', '.join(stale[:5])} (a key is "
                         "direction/mechanism in launch.py's form, `tag`)")
    notes: dict[str, Counter] = defaultdict(Counter)
    for v in x.s["launch"].values():
        if v.get("note"):
            notes[method_key(v.get("direction"), v.get("mechanism"))][v["note"]] += 1
    methods = {}
    for key in keys:
        mech = key.split("/", 1)[1]
        over = (cur.get("methods") or {}).get(key) or {}
        if over.get("label"):
            label, desc, how = over["label"], over.get("description"), "curation"
        elif key in prow or mech in prow:
            hit = prow.get(key) or prow[mech]
            label, desc, how = hit.get("label"), hit.get("description"), "methods panel"
        elif notes.get(key):
            note = notes[key].most_common(1)[0][0]
            label, desc, how = mech, note, "launch note"
        else:
            label, desc, how = mech, None, None
        methods[key] = {"label": label, "description": desc, "description_source": how,
                        "learned": over.get("learned") or [], "open": over.get("open") or []}

    checkpoints = [x.checkpoint(r, x.source[r["id"]], True, ctx, methods) for r in x.scored]
    checkpoints += [x.checkpoint(r, rescore.STUBS, False, ctx, methods) for r in x.stubbed]
    checkpoints.sort(key=lambda c: c["id"])
    left_out = x.left_out()
    ledger_by_run_macs = {(r.get("run_id"), r.get("macs")): r for r in x.rows if r.get("run_id")}
    earlier = x.earlier(ledger_by_run_macs)

    # runs
    by_run_cp: dict[str, list[dict]] = defaultdict(list)
    for c in checkpoints:
        if c["run"]:
            by_run_cp[c["run"]].append(c)
    left_by_run: dict[str, Counter] = defaultdict(Counter)
    for r in left_out:
        if r["run"]:
            left_by_run[r["run"]][r["reason"]] += 1
    reruns = {row["run"]: run for run, e in x.fin.items() for row in e["rows"] if row.get("run")}
    first_row: dict[str, dict] = {}
    for r in x.rows:
        if r.get("run_id"):
            first_row.setdefault(r["run_id"], r)
    universe = set(x.s["by_run"]) | set(x.s["runs"]) | set(first_row) | set(reruns)
    runs = []
    for run in sorted(universe):
        launch, rec, row = x.launch_of(run) or {}, x.s["runs"].get(run) or {}, first_row.get(run) or {}
        start = launch.get("start") or row.get("start") or rec.get("start")
        t = x.when(run, row.get("written_at"))
        cps = by_run_cp.get(run, [])
        best = min(cps, key=lambda c: c["loss"]["search_bpb"], default=None)
        l0 = ctx["l0"].get(start)
        reached = []
        for f in panels.TARGETS:
            hit = [c for c in cps if l0 and c["loss"]["search_bpb"] <= l0 * (1 - f)]
            if hit:
                h = min(hit, key=lambda c: c["compute"]["gates"])
                reached.append({"drop": f, "id": h["id"], "checkpoint": h["checkpoint"], "gates": h["compute"]["gates"]})
        fin = [c["loss"]["final"] | {"id": c["id"], "checkpoint": c["checkpoint"]} for c in cps if c["loss"]["final"]]
        final_run = run in reruns or launch.get("final") is True
        kind = "final-rerun" if final_run else "search" if (launch or row) else "unrecorded"
        rerun_of = reruns.get(run)
        if final_run and rerun_of is None and (m := ART_REF.search(launch.get("note") or "")):
            rerun_of = ix.art_run.get(ix.art(m[1]) or "")
        cfg = x.config(launch or None, start)
        mine = by_run.get(run, [])
        runs.append({
            "run": run, "night": night_of(t), "launched": iso(t), "kind": kind,
            "agent": launch.get("agent") or rec.get("agent") or row.get("agent"),
            "direction": launch.get("direction") or row.get("direction") or rec.get("direction"),
            "mechanism": tag(launch.get("mechanism") or row.get("mechanism")),
            "class": launch.get("class") or rec.get("class"), "start": start, "island": launch.get("island"),
            "node": launch.get("node") or rec.get("machine") or row.get("node") or None, "state": rec.get("state"),
            "rc": rec.get("rc"), "cancelled": rec.get("cancelled"), "regime": launch.get("regime"),
            "harness": {"version": harness_of(row, launch, x.s["pins"]) if launch or row else None,
                        "measurement_commit": launch.get("measurement_commit") or row.get("measurement_commit")
                        or None,
                        "witness": launch.get("witness"),
                        "te_core": launch.get("te_core") or rec.get("te_core") or row.get("run_te_core") or None,
                        "commit": launch.get("commit") or rec.get("commit") or row.get("commit") or None,
                        "program": launch.get("program") or row.get("program") or None,
                        "island_tree": launch.get("island_tree") or (row.get("island") or {}).get("tree")},
            "hypothesis": launch.get("note") or None, "config": cfg or None,
            "seed": {"run": row.get("seed"),
                     "data": (cfg.get("train_args") or {}).get("data_seed")
                     if isinstance(cfg.get("train_args"), dict) else None},
            "outcome": {"checkpoints": len(cps), "left_out": dict(left_by_run.get(run, {})),
                        "in_panel": sum(c["in_panel"] for c in cps),
                        "best": best and {"id": best["id"], "checkpoint": best["checkpoint"],
                                          "search_bpb": best["loss"]["search_bpb"], "drop": best["loss"]["drop"],
                                          "gates": best["compute"]["gates"]},
                        "targets": reached, "launch_valid": launch.get("valid"),
                        "failed_checks": launch.get("failed_checks"), "final": fin},
            "rerun_of": rerun_of,
            "board": [b["id"] for b in mine], "findings": [b["id"] for b in mine if b["tag"] == "FINDING"],
            "negatives": [b["id"] for b in mine if b["tag"] == "NEGATIVE"],
            "arts": {"launch": launch.get("id"), "run_record": rec.get("run_record")},
        })
    run_of = {r["run"]: r for r in runs}

    # methods, with their runs' outcomes: a run is a method's when its launch record or one of its checkpoints says so
    cps_by: dict[str, dict[str, list[dict]]] = {"method": defaultdict(list), "direction": defaultdict(list)}
    runs_by: dict[str, dict[str, set[str]]] = {"method": defaultdict(set), "direction": defaultdict(set)}
    for c in checkpoints:
        cps_by["method"][c["method"]["key"]].append(c)
        cps_by["direction"][c["direction"]].append(c)
        if c["run"]:
            runs_by["method"][c["method"]["key"]].add(c["run"])
            runs_by["direction"][c["direction"]].add(c["run"])
    for r in runs:
        runs_by["method"][method_key(r["direction"], r["mechanism"])].add(r["run"])
        runs_by["direction"][r["direction"]].add(r["run"])

    def outcome_of(group: str, key: str) -> tuple[list[dict], list[dict]]:
        return [run_of[r] for r in sorted(runs_by[group].get(key, ()))], cps_by[group].get(key, [])

    method_rows = []
    for key, mv in methods.items():
        mine, cps = outcome_of("method", key)
        best = {}
        for c in cps:
            s = c["start"]["id"]
            if s not in best or c["loss"]["search_bpb"] < best[s]["search_bpb"]:
                best[s] = {"id": c["id"], "run": c["run"], "checkpoint": c["checkpoint"],
                           "search_bpb": c["loss"]["search_bpb"], "drop": c["loss"]["drop"],
                           "gates": c["compute"]["gates"]}
        fams = Counter(c["family"] for c in cps if c["family"])
        method_rows.append({
            "key": key, "direction": key.split("/", 1)[0], "mechanism": key.split("/", 1)[1], **mv,
            "family": fams.most_common(1)[0][0] if fams else None,
            "runs": len(mine), "checkpoints": len(cps), "left_out": sum(sum(r["outcome"]["left_out"].values())
                                                                       for r in mine),
            "starts": sorted({c["start"]["id"] for c in cps}),
            "nights": sorted({r["night"] for r in mine if r["night"]} | {c["night"] for c in cps if c["night"]}),
            "best": best, "findings": sorted({f for r in mine for f in r["findings"]}),
            "negatives": sorted({f for r in mine for f in r["negatives"]}),
        })

    # curation: directions and lessons, every cited id known
    known = {b["id"] for b in board} | set(run_of) | set(x.by_id)

    def check(ids: list[str], where: str) -> list[str]:
        bad = [i for i in ids if i not in known and not (i.startswith("art:") and len(i) >= 16 and ix.art(i[4:]))]
        if bad:
            raise SystemExit(f"dataset: {where} cites ids no table holds: {', '.join(bad[:5])} (cite a board id, a run "
                             "id, a ledger id or an art: prefix of 12 hex digits or more)")
        return ids

    directions = []
    for name, v in sorted((cur.get("directions") or {}).items()):
        mine, cps = outcome_of("direction", name)
        ideas = [{"idea": i["idea"], "verdict": i["verdict"], "outcome": i["outcome"],
                  "evidence": check(i.get("evidence", []), f"{name}'s idea")} for i in v.get("ideas", [])]
        directions.append({
            "direction": name, "what": v["what"], "ideas": ideas, "learned": v.get("learned", []),
            "open": v.get("open", []), "evidence": check(v.get("evidence", []), name),
            "runs": len(mine), "checkpoints": len(cps),
            "mechanisms": sorted({r["mechanism"] for r in mine if r["mechanism"]} | {c["mechanism"] for c in cps
                                                                                    if c["mechanism"]}),
            "starts": sorted({c["start"]["id"] for c in cps}),
            "nights": sorted({r["night"] for r in mine if r["night"]} | {c["night"] for c in cps if c["night"]}),
            "findings": len({f for r in mine for f in r["findings"]}),
            "negatives": len({f for r in mine for f in r["negatives"]}),
        })
    lessons = [{"id": v["id"], "kind": v["kind"], "lesson": v["lesson"], "fix": v.get("fix"),
                "status": v.get("status"), "rank": v.get("rank"), "evidence": check(v.get("evidence", []), v["id"])}
               for v in cur.get("lessons", [])]
    if len({v["id"] for v in lessons}) != len(lessons):
        raise SystemExit("dataset: curation.json repeats a lesson id")

    # derived: starts, branch points, curves
    starts = []
    for s, shape in sorted(x.shapes.items()):
        b = branches.get(s)
        cps = [c for c in checkpoints if c["start"]["id"] == s]
        how = policy.c0(shape)[1]
        starts.append({"id": s, "label": plots.start_label(shape), "shape": shape, "w_fwd_gates": panels.w_fwd(shape),
                       "l0": ctx["l0"].get(s), "l0_final": final_l0.get(s), "c0_gates": ctx["c0"].get(s),
                       "c0_basis": c0_basis[s],
                       "c0_source": how if how == "counted" else
                       f"the start's training tokens at dense AdamW's gates per token ({info['rate_how']}), "
                       f"{info['rate']:.4g} forward passes",
                       "on_frontier": b.on_frontier if b else None, "checkpoints": len(cps),
                       "in_panel": sum(c["in_panel"] for c in cps), "runs": len({c["run"] for c in cps if c["run"]})})
    complete = {r["id"]: x.complete(r) for r in x.scored}

    def setter(r: dict | None) -> dict | None:
        return r and {"id": r["id"], "run": r.get("run_id") or None, "checkpoint": r.get("checkpoint_step"),
                      "method": method_key(r.get("direction"), r.get("mechanism")),
                      "bound": "complete schedule" if complete.get(r["id"]) else "upper bound (checkpoint)"}

    branch_points = []
    for b in info["branches"]:
        for f in panels.TARGETS:
            l_star = b.l0 * (1 - f)
            hit = next(((xx, rr) for xx, yy, rr in b.honest if yy <= l_star), None)
            h = panels.least(b.honest, l_star)
            cells = []
            for key in ((n, kb) for n in panels.TOKENS_PER_BLOCK for kb in panels.K_BYTES):
                s_, r = b.s_at(key, l_star)
                v = r["dial"][key] if r else None
                if s_ is not None and s_ < 1 - 1e-12:
                    raise SystemExit(f"dataset: S = {s_} < 1 at {b.name} -{100 * f:g}% {key}: H and W* must read the "
                                     "same checkpoints (te_core.panels.Branch)")
                cells.append({"c_tokens_per_block": key[0], "k_bytes": key[1], "c_gates": v and v["c"],
                              "S": s_, "W_star_gates": s_ * h if s_ is not None and h else None,
                              "rho": v and v["rho"], "basis": v and ("fit" if v["fit"] else "target"),
                              "setter": setter(r), "x0": v and v["x0"], "s_star": v and v["s_star"],
                              "x0_known": v and v["x0_known"], "s_star_known": v and v["s_star_known"]})
            branch_points.append({"id": f"{b.name}@{f:g}", "start": b.name, "label": b.label, "c0_gates": b.c0,
                                  "l0": b.l0, "drop": f, "l_star": l_star, "reached": hit is not None,
                                  "H": {"gates": h, "total_gates": b.c0 + h if h is not None else None,
                                        "setter": setter(hit[1] if hit else None)},
                                  "cells": cells})
    label_to_start = {b.label: b.name for b in info["branches"]}
    curves = []
    for row in p["rows"]:
        v = dict(zip(panels.COLUMNS, row))
        ref, _, step = (v.get("run") or "").partition("@")
        v |= {"on_frontier": None if v["on_frontier"] is None else v["on_frontier"] == "yes",
              "start": v["start"] or None, "start_id": label_to_start.get(v["start"]) if v["start"] else None,
              "ledger_id": ref if ref in x.by_id else None,
              "checkpoint": int(step) if ref in x.by_id and step.isdigit() else None}
        curves.append(v)

    tables = {"checkpoints": checkpoints, "left_out": left_out, "earlier": earlier, "runs": runs, "board": board,
              "methods": method_rows, "directions": directions, "lessons": lessons, "starts": starts,
              "branch_points": branch_points, "curves": curves}
    manifest = summarize(a, x, tables, info, pre_dir, cur, entries)
    return tables, manifest


def summarize(a, x: Export, tables: dict, info: dict, pre_dir: Path, cur: dict, entries: list[dict]) -> dict:
    """What went in, the counts and what each rests on (README, "The tables")."""
    cps = tables["checkpoints"]
    basis: dict[str, Counter] = defaultdict(Counter)
    for c in cps:
        for k, v in c["rho_inputs"].items():
            if isinstance(v, dict):
                basis[k][v["basis"]] += 1
    labels: dict[str, Counter] = defaultdict(Counter)
    for r in x.scored + x.stubbed:
        for k, v in r["measured"].labels.items():
            labels[k][v] += 1
    dial = Counter(cell["basis"] for c in cps for cell in c["dial"])
    checkout = a.checkout or subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True,
                                            check=True).stdout.strip()
    days = Counter(e["at"].date().isoformat() for e in entries)

    def by(rows, key):
        return dict(sorted(Counter(key(r) for r in rows).items(), key=lambda kv: str(kv[0])))

    return {
        "dataset": NAME, "te_core": VERSION, "checkout": checkout, "pin": list(x.pin), "stamp": x.stamp,
        "tables": list(TABLES),
        "sources": {
            "ledger": {"path": str(Path(a.root or default_root())), "rows": len(x.rows)},
            "scorer": {"path": str(a.scorer), "collected_at": x.s["collected_at"], "launch_records": len(x.s["launch"]),
                       "runs": len(x.s["runs"]), "submissions": len(x.s["subs"]), "finals": x.fin_read},
            "quantities": {"mode": x.d["mode"], "how": x.d["how"], "word": x.d["quantities"]},
            "board": {"path": str(a.board), "sha256": sha256(a.board), "entries": len(entries), "days": dict(days)},
            "methods": {"path": str(a.methods), "sha256": sha256(a.methods)} if a.methods else None,
            "curation": {"path": str(a.curation), "sha256": sha256(a.curation)}
            if a.curation and a.curation.exists() else None,
            "pretraining": {"art": panels.PRETRAINING if a.pretraining is None or
                            a.pretraining.name == panels.PRETRAINING.removeprefix("art:") else None,
                            "path": str(pre_dir)},
        },
        "nights": [{"night": 1, "until": iso(NIGHT2)}, {"night": 2, "from": iso(NIGHT2)}],
        "counts": {k: len(v) for k, v in tables.items()},
        "checkpoints_by": {"night": by(cps, lambda c: c["night"]), "quantities": by(cps, lambda c: c["quantities"]),
                           "start": by(cps, lambda c: c["start"]["id"]), "direction": by(cps, lambda c: c["direction"]),
                           "harness": by(cps, lambda c: c["harness"]["version"]),
                           "family": by(cps, lambda c: c["family"]), "in_panel": by(cps, lambda c: c["in_panel"]),
                           "with_final": sum(c["loss"]["final"] is not None for c in cps)},
        "left_out_by_reason": by(tables["left_out"], lambda r: r["reason"]),
        "earlier_by_regime": by(tables["earlier"], lambda r: r["regime"]),
        "earlier_by_harness": by(tables["earlier"], lambda r: r["harness"]["version"]),
        "runs_by_kind": by(tables["runs"], lambda r: r["kind"]),
        "runs_by_night": by(tables["runs"], lambda r: r["night"]),
        "board_by_tag": by(tables["board"], lambda r: r["tag"] or "untagged"),
        "board_linked": sum(bool(r["runs"] or r["checkpoints"] or r["arts"]) for r in tables["board"]),
        "rho_input_basis": {k: dict(v) for k, v in sorted(basis.items())},
        "dial_basis": dict(dial),
        "labels": {k: dict(v) for k, v in sorted(labels.items())},
        "c0": {"rate_forward_passes_per_token": info["rate"], "rate_how": info["rate_how"],
               "from_scratch_loss_ratio": info["ratio"]},
        "uncurated_directions": sorted({c["direction"] for c in cps} - {d["direction"] for d in tables["directions"]}),
        "top_lessons": [v["id"] for v in sorted((v for v in tables["lessons"] if v.get("rank")),
                                               key=lambda v: v["rank"])],
        "deviations": DEVIATIONS, "caveats": CAVEATS + [QUANTITIES[x.d["mode"]]],
    }


def validate(tables: dict[str, list[dict]], schema: dict) -> None:
    for name, rows in tables.items():
        v = jsonschema.Draft202012Validator(schema["$defs"][name] | {"$defs": schema["$defs"]})
        for i, r in enumerate(rows):
            if err := next(iter(v.iter_errors(r)), None):
                where = "/".join(str(p) for p in err.absolute_path)
                raise SystemExit(f"dataset: {name} row {i} ({r.get('id') or r.get('run') or r.get('key')}) at "
                                 f"`{where}`: {err.message[:300]} (the exporter and schema.json change together)")


def write(out: Path, tables: dict[str, list[dict]], manifest: dict, curation: Path | None) -> None:
    schema = json.loads(SCHEMA.read_text())
    count = [0]
    clean = {k: [finite(r, count) for r in rows] for k, rows in tables.items()}
    validate(clean, schema)
    if set(schema["properties"]["tables"]["required"]) != set(TABLES):
        raise SystemExit("dataset: schema.json's tables are not TABLES")
    manifest = manifest | {"non_finite": count[0]}
    jsonschema.Draft202012Validator(schema["$defs"]["manifest"] | {"$defs": schema["$defs"]}).validate(manifest)
    out.mkdir(parents=True, exist_ok=True)
    for k, rows in clean.items():
        with (out / f"{k}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, separators=(",", ":"), allow_nan=False) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, allow_nan=False) + "\n")
    shutil.copyfile(SCHEMA, out / "schema.json")
    if curation and curation.exists():
        shutil.copyfile(curation, out / "curation.json")


def publish(out: Path, research: str, manifest: dict) -> str:
    meta = {"lane": "training", "what": f"{NAME}: the training-efficiency search's dataset (schema.json, manifest.json)",
            "schema": NAME, "te_core": manifest["te_core"], "checkout": manifest["checkout"], "pin": manifest["pin"],
            "stamp": manifest["stamp"], "counts": manifest["counts"]}
    cmd = [research, "data", "put", "--kind", "evidence/v1", "--tree", str(out), "--preserve", "--json",
           "--meta", json.dumps(meta)]
    if manifest["sources"]["pretraining"]["art"]:
        cmd += ["--ref", f"pretraining={manifest['sources']['pretraining']['art']}"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"dataset: research data put exited {r.returncode}: {r.stderr.strip()[-600:]}")
    return json.loads(r.stdout)["id"]


def report(manifest: dict, art: str | None) -> str:
    """The manifest in a few lines, for the console and the campaign's notes."""
    c = manifest["counts"]
    lines = [f"{manifest['dataset']} ({art or 'not published'}): te_core {manifest['te_core']}, pin "
             f"{' + '.join(p[:12] for p in manifest['pin'])}, ledger stamp {manifest['stamp']}",
             "counts: " + ", ".join(f"{k} {v}" for k, v in c.items()),
             f"checkpoints by night {manifest['checkpoints_by']['night']}, by quantities "
             f"{manifest['checkpoints_by']['quantities']}, with a final rerun {manifest['checkpoints_by']['with_final']}",
             f"left out {manifest['left_out_by_reason']}",
             f"earlier by regime {manifest['earlier_by_regime']}",
             f"runs by kind {manifest['runs_by_kind']}, board by tag {manifest['board_by_tag']} "
             f"({manifest['board_linked']} linked to a run, checkpoint or art)",
             "rho inputs by basis: " + "; ".join(f"{k} {v}" for k, v in manifest["rho_input_basis"].items()),
             f"dial cells by basis {manifest['dial_basis']}; non-finite values written null: {manifest['non_finite']}"]
    return "\n".join(lines)


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    tables, manifest = build(a)
    write(a.out, tables, manifest, a.curation)
    manifest = json.loads((a.out / "manifest.json").read_text())
    art = publish(a.out, a.research, manifest) if a.publish else None
    print(report(manifest, art))
    if art:
        print(art)
    return 0


if __name__ == "__main__":
    sys.exit(main())
