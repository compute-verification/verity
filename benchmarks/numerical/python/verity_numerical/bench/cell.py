"""One benchmark entry point: a cell is (backend, statement, input set) on a prover pod, measured under the TABLES protocol.

    python -m verity_numerical.bench.cell plan  --backend C-interactive \\
        --statement gemm-coordinate/k1536/sm90-mma-bf16+frame-v3/blake3-keyed --input-set bench-instances-bf16-hopper-v1 \\
        --prover vy-flock-h100 --verifier vy-flock-verifier --verifier-addr HOST:PORT --rtt-target HOST:22 \\
        --campaign C --lane L [--points "1024 2048 ..."] [--runs 5] [--per-proof 8192] [--out cell.json]
    python -m verity_numerical.bench.cell run      --cell cell.json            launch the plan's jobs (detached, --custody-r2)
    python -m verity_numerical.bench.cell check    --cell cell.json --prover-run RUN [--verifier-run RUN] [--result FILE]
    python -m verity_numerical.bench.cell register --cell cell.json --prover-run RUN [--verifier-run RUN] [--dry-run]

``--statement`` is ``<subcircuit id>+<scheme id>`` (the render's statement ids); ``--input-set`` a census input set id (a
backend's frozen set), an input-set directory, or an ``art:`` id of one (``input_sets``).  ``plan`` resolves the cell: the
backend's driver (``bench.drivers``, one module per backend), the family's lowering of the statement (``bench.lowerings``: why
not, or the relation name its scripts take), and whether the backend can run on the input set; it writes the cell and the
``research run`` of every job, which ``run`` launches in order.  The protocol is the backend's own scripts', wrapped, never
rewritten: the plateau sweep (doubling from 1,024 until two doublings add < 2 %), warm + >= 5 timed runs + the timing guard, the
verifier on a separate machine, an open-connection RTT probe and a loopback probe.

Separate means another physical machine on a routed link (``bench.placement``): ``plan`` reads both pods' RunPod machine id and
public IP (or ``--placement FILE``) and refuses a pair sharing either, or linked over a host-private address; each job first
records its container's identity (DMI product uuid, kernel boot id, CPU model, public IP, the route to the verifier) in the run's
``placement.json``, and ``check`` compares both runs' identities, never their hostnames.

``check`` reads the prover run's result and says what keeps it from counting: the result contract (``contract.validate``), the
protocol (``interaction.protocol_problem``), the interaction record (``interaction.interaction_problem``), a live verifier on another machine for an
interactive backend, the loopback probe, and that its statement and instance set are the cell's.  ``register`` runs the backend's
registration script, stamps the cell (backend, family, statement, subcircuit, scheme, input set, both machines) into the document, checks it,
and puts it in the evidence store (``research data put --kind bench-result/v1 --ref run_files=... --preserve``).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import census as CN
from . import contract, drivers, input_sets as IS, interaction, lowerings as LW, placement as PL, templates as TM

FAMILIES = {"A": "A-GKR", "B": "B-Ligero", "C": "C-Flock", "D": "D-SP1"}
BACKENDS = ("A-interactive", "A-fs", "A-route-a", "B-interactive", "B-fs", "C-interactive", "D-fs")
DEFAULT_POINTS = "1024 2048 4096 8192 16384 32768 65536"
RUN_ID = re.compile(r"\br\d{8}-\d{6}-[0-9a-f]{4}\b")


def family_of(backend: str) -> str:
    if backend not in BACKENDS:
        raise KeyError(f"unknown backend {backend!r} (one of {BACKENDS})")
    return FAMILIES[backend[0]]


def split_statement(statement: str) -> tuple[str, str]:
    sub, sep, scheme = statement.partition("+")
    if not sep or not sub or not scheme:
        raise ValueError(f"a statement is <subcircuit id>+<scheme id>, got {statement!r}")
    return sub, scheme


@dataclass
class Cell:
    backend: str
    family: str
    statement: str
    subcircuit: str
    scheme: str
    input_set: dict[str, Any]
    relation: str
    prover: str
    verifier: str | None = None
    verifier_addr: str | None = None
    rtt_target: str | None = None
    campaign: str | None = None
    lane: str | None = None
    points: str = DEFAULT_POINTS
    runs: int = 5
    per_proof: int | None = None
    backend_args: list[str] = field(default_factory=list)
    jobs: list[dict[str, Any]] = field(default_factory=list)
    #: an interactive cell's pods: {"prover": identity, "verifier": identity, "link": {"peer", "peer_ip"}} (``bench.placement``)
    placement: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, default=str) + "\n"

    @classmethod
    def load(cls, path: str | Path) -> "Cell":
        return cls(**json.loads(Path(path).read_text()))


class NotSupported(ValueError):
    """The cell cannot run: the reason is what a table prints (``n/s``)."""


def resolve_input_set(arg: str) -> dict[str, Any]:
    """A census frozen set (``kind = census``) or an input set (``kind = set``: a directory, or an ``art:`` id fetched)."""
    if arg in CN.INPUT_SETS:
        e = CN.INPUT_SETS[arg]
        return {"kind": "census", "id": e.id, "subcircuit": e.subcircuit, "dataset": e.dataset, "tier": e.tier, "n": e.size,
                "manifest_sha256": e.manifest_sha256, "source": e.provenance}
    art = None
    path = Path(arg)
    if arg.startswith("art:"):
        art = arg
        r = subprocess.run(["research", "data", "fetch", arg], capture_output=True, text=True)
        if r.returncode:
            raise ValueError(f"research data fetch {arg}: {r.stderr.strip()[-300:]}")
        path = Path(r.stdout.strip().splitlines()[-1])
    if not path.exists():
        raise ValueError(f"--input-set {arg!r}: not a census input set ({sorted(CN.INPUT_SETS)}), a directory or an art: id")
    s = IS.InputSet.open(path)
    return {"kind": "set", "id": s.name, "subcircuit": s.subcircuit.id, "path": str(s.path), "art": art, "n": s.n,
            "content_digest": s.content_digest, "manifest_sha256": s.manifest_sha256(), "source": s.source}


def plan(backend: str, statement: str, input_set: str, prover: str, relation: str | None = None, **kw: Any) -> Cell:
    """Resolve and check a cell and its jobs; raises :class:`NotSupported` with the reason when it cannot run. ``relation``
    names another of the family's statements of the same subcircuit and scheme than the template module's (the driver must
    know it: ``RELATIONS``)."""
    family = family_of(backend)
    sid, scheme = split_statement(statement)
    sub = TM.parse_subcircuit_id(sid)
    drv = drivers.driver(backend)
    if drv is None:
        raise NotSupported(f"{backend}: no driver module {drivers.module_path(backend)} (no sweep of this backend is wrapped yet)")
    # a driver judges the schemes of its own relations (None: supported, a reason, or False: the template module's to judge)
    judged = drv.supports_relation(relation, sub, scheme) if relation and hasattr(drv, "supports_relation") else False
    why = LW.registry(family).supports(sub, scheme) if judged is False else judged
    if why:
        raise NotSupported(why)
    ref = resolve_input_set(input_set)
    if ref["subcircuit"] != sub.id:
        raise NotSupported(f"input set {ref['id']} is of {ref['subcircuit']}, the statement's subcircuit is {sub.id}")
    if ref["kind"] == "set" and LW.registry(family).stage(sub) is None:
        raise NotSupported(f"{family}'s {sub.template.id} lowering has no stage(): its scripts regenerate their own frozen set, so "
                           f"it runs on a census set only, not on input set {ref['id']}")
    if relation is not None and not any(relation.startswith(r) for r in getattr(drv, "RELATIONS", ())):
        raise NotSupported(f"{backend}'s driver runs no relation {relation!r} (its RELATIONS: {getattr(drv, 'RELATIONS', ())})")
    cell = Cell(backend, family, statement, sub.id, scheme, ref, relation or LW.registry(family).relation(sub, scheme), prover, **kw)
    if drv.INTERACTIVE and not (cell.verifier and cell.verifier_addr):
        raise NotSupported(f"{backend} is interactive: the protocol puts its verifier on a separate pod (--verifier, --verifier-addr)")
    if drv.INTERACTIVE and cell.verifier == cell.prover:
        raise NotSupported("the verifier must run on a pod other than the prover's")
    if drv.INTERACTIVE:
        if not cell.placement:
            cell.placement = resolve_placement(cell)
        a = PL.assess(cell.placement.get("prover") or {}, cell.placement.get("verifier") or {}, cell.placement.get("link"))
        if a["shared_public_ip"] and not cell.placement.get("probed"):
            try:
                cell.placement = probe_pods(cell)
            except NotSupported as e:
                raise NotSupported("; ".join([*a["problems"], str(e)])) from e
            a = PL.assess(cell.placement["prover"], cell.placement["verifier"], cell.placement["link"])
        if a["problems"]:
            raise NotSupported("; ".join(a["problems"]))
        cell.placement["shared_public_ip"] = a["shared_public_ip"]
        if a["shared_public_ip"]:
            cell.placement["separation"] = a["evidence"]
    why = getattr(drv, "unmet", lambda _c: None)(cell)
    if why:
        raise NotSupported(why)
    cell.jobs = [research_run(cell, j) for j in drv.jobs(cell)]
    return cell


def resolve_placement(cell: Cell) -> dict[str, Any]:
    """Both pods' identity from the RunPod API (machine id, public IP) and the address the prover will reach the verifier at."""
    try:
        prover, verifier = PL.from_runpod(cell.prover), PL.from_runpod(cell.verifier or "")
    except Exception as e:  # noqa: BLE001  (no API, no key, an unknown pod: the separation cannot be shown)
        raise NotSupported(f"the pods' machine identity is unavailable ({type(e).__name__}: {e}); pass it with --placement FILE") from e
    return {"prover": prover, "verifier": verifier, "link": PL.link_to(cell.verifier_addr or "")}


#: Before a cell's job, each pod records its identity (and the prover the route to its verifier) in the run's placement.json.
PROBE_WRAP = (f'python3 {PL.PROBE} probe --peer "$1" --rtt "$2" --out "${{RESEARCH_RUN_DIR:-.}}/placement.json" >/dev/null '
              '|| echo "placement probe failed" >&2; shift 2; exec "$@"')


def probe_pods(cell: Cell) -> dict[str, Any]:
    """A shared-IP pair's full identity before any job: both pods probed over ssh (the prover with its link and TCP-connect RTT to
    the verifier), merged over the API's (red-team-flock-3 S2)."""
    try:
        p = PL.remote_probe(cell.prover, cell.verifier_addr)
        v = PL.remote_probe(cell.verifier or "")
    except Exception as e:  # noqa: BLE001  (no ssh, no pod: the exception cannot be shown)
        raise NotSupported(f"the pods share a public IP and could not be probed ({type(e).__name__}: {e})") from e
    plan = cell.placement
    return {"prover": PL.merge(plan.get("prover"), {k: x for k, x in p.items() if k != "link"}),
            "verifier": PL.merge(plan.get("verifier"), {k: x for k, x in v.items() if k != "link"}),
            "link": PL.merge(plan.get("link"), p.get("link")), "probed": True}


def research_run(cell: Cell, job: Mapping[str, Any]) -> dict[str, Any]:
    """A driver job as the command that launches it: a ``research run --on`` (detached, custody on R2), or, for a ``setup`` job,
    the given ``research pods`` command."""
    if job.get("setup"):
        return dict(job, argv=list(job["command"]))
    argv = ["research", "run", "--on", job["pod"], "--project", "verity", "--source", ".", "--cwd", "source", "--custody-r2",
            "--custody-ttl", job.get("ttl", "8h"), "--stage", job["stage"], "--exclusive"]
    if cell.campaign:
        argv += ["--campaign", cell.campaign]
    for f in job.get("send") or []:
        argv += ["--send", str(f)]
    for k, v in sorted((job.get("env") or {}).items()):
        argv += ["--env", f"{k}={v}"]
    for f in job.get("send") or []:
        argv += ["--send", f]
    command = list(job["command"])
    if cell.placement:
        prover = job.get("role") == "prover"
        command = ["bash", "-c", PROBE_WRAP, "cell-job", (cell.verifier_addr or "") if prover else "",
                   (PL.rtt_target(cell.verifier_addr) or "") if prover else "", *command]
    return dict(job, argv=argv + ["--", *command])


def launch(cell: Cell, dry_run: bool = False) -> list[dict[str, Any]]:
    """Run every job's command in order; each ``research run --on`` returns once the job is started on its pod."""
    out = []
    for j in cell.jobs:
        if dry_run:
            out.append({"role": j["role"], "argv": j["argv"]})
            continue
        r = subprocess.run(j["argv"], capture_output=True, text=True)
        rid = RUN_ID.search(r.stdout + r.stderr)
        out.append({"role": j["role"], "rc": r.returncode, "run": rid.group(0) if rid else None, "tail": (r.stdout + r.stderr)[-600:]})
        if r.returncode:
            break
    return out


# --- check -------------------------------------------------------------------------------------------------------------------

def _meas(doc: Mapping[str, Any]) -> dict[str, Any]:
    return {m["name"]: m.get("value") for m in doc.get("measurements") or [] if isinstance(m, Mapping) and "name" in m}


def run_placement(cell: Cell, prover_run: Path | None, verifier_run: Path | None) -> dict[str, Any]:
    """Each side's identity: the plan's (the RunPod API) merged with what its run recorded (``placement.json``, ``job.json``)."""
    plan = cell.placement or {}
    p = PL.merge(plan.get("prover"), PL.from_job(prover_run) if prover_run else None)
    v = PL.merge(plan.get("verifier"), PL.from_job(verifier_run) if verifier_run else None)
    return {"prover": p, "verifier": v, "link": PL.merge(plan.get("link"), p.pop("link", None)), "verifier_link": v.pop("link", None)}


def run_assessment(cell: Cell, prover_run: Path | None, verifier_run: Path | None) -> dict[str, Any]:
    """``placement.assess`` on both runs' identities (the plan's merged with the runs' records)."""
    pl = run_placement(cell, prover_run, verifier_run)
    return PL.assess(pl["prover"], pl["verifier"], pl["link"])


RUN_NAT_KEYS = ("pod_id", "boot_id", "hypervisor", "sys_vendor")


def placement_problems(cell: Cell, prover_run: Path | None, verifier_run: Path | None) -> list[str]:
    """Why the prover and verifier are not shown to be separate machines on a routed link, and whether each run ran on the pod
    the cell planned (a run whose pod id differs from the plan's).  Under the NAT exception every value is the runs' own
    (red-team-flock-3 S2): each run records all of :data:`RUN_NAT_KEYS` (the prover its link and RTT too), and the boot id
    the plan probed on that pod, since the machine id is the plan's."""
    if not (cell.placement or prover_run or verifier_run):
        return []
    a = run_assessment(cell, prover_run, verifier_run)
    out = list(a["problems"])
    if a["shared_public_ip"]:
        for role, run in (("prover", prover_run), ("verifier", verifier_run)):
            if run is None:
                continue
            rec, planned = PL.from_job(run), cell.placement.get(role) or {}
            keys = [k for k in RUN_NAT_KEYS if rec.get(k) is None or rec.get(k) == ""]
            if role == "prover":
                link = rec.get("link") or {}
                keys += [f"link.{k}" for k in ("peer", "peer_ip", "rtt") if not link.get(k)]
            if keys:
                out.append(f"the {role} run did not record {', '.join(keys)}: the NAT exception is re-checked on the runs' own records")
            if planned.get("boot_id") and rec.get("boot_id") and planned["boot_id"] != rec["boot_id"]:
                out.append(f"the {role} run's boot_id {rec['boot_id']} is not the {planned['boot_id']} the plan probed on pod "
                           f"{planned.get('pod_id')} (a reboot or another host since the plan)")
    for role, run in (("prover", prover_run), ("verifier", verifier_run)):
        planned, ran = (cell.placement.get(role) or {}).get("pod_id"), (PL.from_job(run) if run else {}).get("pod_id")
        if planned and ran and planned != ran:
            out.append(f"the {role} run ran on pod {ran}, the cell planned {planned}")
    return out


def check(cell: Cell, doc: Mapping[str, Any], prover_run: Path | None = None, verifier_run: Path | None = None) -> list[str]:
    """What keeps ``doc`` from counting for ``cell`` (empty: nothing)."""
    drv = drivers.driver(cell.backend)
    fp = doc.get("workload_fingerprint") if isinstance(doc.get("workload_fingerprint"), Mapping) else {}
    out = [f"contract: {p}" for p in contract.validate(doc)]
    why = interaction.protocol_problem(doc)
    if why:
        out.append(why)
    ix = interaction.interaction(doc)
    why = interaction.interaction_problem(ix)
    if why:
        out.append(why)
    m = _meas(doc)
    inter = doc.get("interaction") if isinstance(doc.get("interaction"), Mapping) else {}
    if drv is not None and drv.INTERACTIVE:
        if not ix["live"]:
            out.append("interactive backend without a live verifier session (no net.rtt_ms / net.wait_seconds / live verifier evidence)")
        placement = (fp.get("verifier") or {}).get("placement") if isinstance(fp.get("verifier"), Mapping) else None
        if placement in (None, "loopback") or str(placement).startswith("127."):
            out.append(f"verifier placement {placement!r}: the protocol's verifier runs on a separate pod")
        out += placement_problems(cell, prover_run, verifier_run)
        if not isinstance(m.get("live.loopback_round_seconds"), (int, float)) or not inter.get("loopback_method"):
            out.append("no loopback probe (live.loopback_round_seconds with interaction.loopback_method)")
        if not isinstance(m.get("net.rtt_ms"), (int, float)) or not inter.get("rtt_method"):
            out.append("no open-connection RTT probe (net.rtt_ms with interaction.rtt_method)")
    rel = fp.get("statement_relation", fp.get("relation"))      # B-Ligero: the statement's "<rel>+<leaf>" beside its relation name
    if rel is not None and rel != cell.relation:
        out.append(f"relation {rel!r}, the cell's is {cell.relation!r}")
    inst = fp.get("instances") if isinstance(fp.get("instances"), Mapping) else {}
    ref = cell.input_set
    if ref["kind"] == "census":
        if (inst.get("dataset"), inst.get("tier")) != (ref["dataset"], ref["tier"]):
            out.append(f"instances {inst.get('dataset')!r}/{inst.get('tier')!r}, the cell's set is {ref['dataset']}/{ref['tier']}")
    elif inst.get("content_digest") != ref["content_digest"]:
        out.append(f"instances content digest {inst.get('content_digest')!r}, the cell's set is {ref['content_digest']}")
    return out


# --- register ----------------------------------------------------------------------------------------------------------------

def runs_root() -> Path:
    return Path(os.environ.get("RESEARCH_RUNS", "~/.research/runs")).expanduser()


def run_dir(ref: str) -> Path:
    """A run directory, fetched (records only) when ``ref`` is a run id not present locally."""
    p = Path(ref)
    if p.is_dir():
        return p
    d = runs_root() / ref
    if not (d / "job.json").is_file():
        subprocess.run(["research", "fetch", ref], check=True, capture_output=True, text=True)
    if not (d / "placement.json").is_file() and (files := run_files(ref)):          # the probe's record is a run file, not a record
        r = subprocess.run(["research", "data", "fetch", files, "--path", "placement.json"], capture_output=True, text=True)
        got = Path(r.stdout.strip().splitlines()[-1]) / "placement.json" if r.returncode == 0 and r.stdout.strip() else None
        if got and got.is_file():
            (d / "placement.json").write_bytes(got.read_bytes())
    return d


#: how long ``register`` waits for the prover run's attempt: the machine's runner publishes it (--custody-r2) after the workload
#: exits, so a register the moment the run ends can find no run files yet
RUN_FILES_WAIT_S = 600.0


def run_files(run_id: str, wait_s: float = 0.0) -> str | None:
    end = time.monotonic() + wait_s
    while True:
        r = subprocess.run(["research", "data", "show", run_id, "--json"], capture_output=True, text=True)
        files = None if r.returncode else (json.loads(r.stdout).get("outputs") or {}).get("run_files")
        if files or time.monotonic() >= end:
            return files
        time.sleep(15)


def stamp(cell: Cell, doc: dict[str, Any], prover_run: Path | None = None, verifier_run: Path | None = None) -> dict[str, Any]:
    doc["cell"] = {"backend": cell.backend, "backend_family": cell.family, "statement": cell.statement, "subcircuit": cell.subcircuit,
                   "subcircuit_template": cell.subcircuit.split("/")[0], "scheme": cell.scheme, "relation": cell.relation,
                   "input_set": {k: cell.input_set.get(k) for k in ("kind", "id", "art", "n", "content_digest", "manifest_sha256", "source")},
                   "prover_pod": cell.prover, "verifier_pod": cell.verifier, "entry": "verity_numerical.bench.cell"}
    if cell.placement or prover_run or verifier_run:
        a = run_assessment(cell, prover_run, verifier_run)
        doc["cell"]["placement"] = dict(run_placement(cell, prover_run, verifier_run), plan=cell.placement,
                                        shared_public_ip=a["shared_public_ip"], **({"separation": a["evidence"]} if a["shared_public_ip"] else {}))
    if cell.lane:
        doc["cell"]["script_lane"] = {k: doc.get(k) for k in ("lane", "registered_by")}
        doc["lane"] = doc["registered_by"] = cell.lane      # the producer is the lane that ran the cell, not the script's author
    return doc


def register(cell: Cell, prover_ref: str, verifier_ref: str | None, work: Path, dry_run: bool = False, force: bool = False
             ) -> dict[str, Any]:
    drv = drivers.driver(cell.backend)
    pd = run_dir(prover_ref)
    vd = run_dir(verifier_ref) if verifier_ref else None
    work.mkdir(parents=True, exist_ok=True)
    meta = work / f"{pd.name}.meta.json"
    for cmd in drv.register(cell, pd, vd, meta):
        subprocess.run(cmd, check=True)
    doc = stamp(cell, json.loads(meta.read_text()), pd, vd)
    problems = check(cell, doc, pd, vd)
    files = run_files(pd.name, RUN_FILES_WAIT_S) if not dry_run else f"<run_files of {pd.name}>"
    if not files:
        problems.append(f"no run files: the prover run {pd.name}'s attempt names none after {RUN_FILES_WAIT_S:.0f} s (its proofs are "
                        f"what the cell counts; `research data show {pd.name}` once its custody lands, then register again)")
    doc.setdefault("validation", {})["cell_problems"] = problems
    meta.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    put = ["research", "data", "put", "--kind", "bench-result/v1", "--meta", f"@{meta}"]
    if files:
        put += ["--ref", f"run_files={files}"]
    if vd is not None:
        vf = run_files(vd.name) if not dry_run else f"<run_files of {vd.name}>"
        if vf:
            put += ["--ref", f"verifier_run_files={vf}"]
    if cell.input_set.get("art"):
        put += ["--ref", f"input_set={cell.input_set['art']}"]
    put.append("--preserve")
    res: dict[str, Any] = {"meta": str(meta), "problems": problems, "put": put}
    if problems and not force:
        res["refused"] = "the result does not meet the cell's protocol (problems above); --force registers it anyway"
        return res
    if not dry_run:
        r = subprocess.run(put, capture_output=True, text=True)
        res["rc"], res["art"] = r.returncode, next((w for w in r.stdout.split() if w.startswith("art:")), None)
    return res


# --- CLI ----------------------------------------------------------------------------------------------------------------------

def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="verity_numerical.bench.cell")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan")
    p.add_argument("--backend", required=True, choices=BACKENDS)
    p.add_argument("--statement", required=True)
    p.add_argument("--input-set", required=True)
    p.add_argument("--prover", required=True, help="the prover pod (vy-*), on the line's GPU")
    p.add_argument("--verifier", default=None, help="the verifier pod (vy-*), same datacenter, not the prover's")
    p.add_argument("--verifier-addr", default=None, help="HOST:PORT the prover reaches the verifier's session port at")
    p.add_argument("--rtt-target", default=None, help="HOST:PORT on the verifier pod the RTT probe connects to (its sshd)")
    p.add_argument("--campaign", default=os.environ.get("RESEARCH_CAMPAIGN"))
    p.add_argument("--lane", default=None)
    p.add_argument("--points", default=DEFAULT_POINTS, help="the sweep's batch sizes (both sides of a remote verifier use them)")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--per-proof", type=int, default=None)
    p.add_argument("--backend-arg", action="append", default=[], help="an extra argument for the backend's prover script")
    p.add_argument("--placement", default=None, help="JSON {prover, verifier, link}: the pods' identities (default: the RunPod API)")
    p.add_argument("--relation", default=None, help="another of the family's statements than the template module's (the driver's RELATIONS)")
    p.add_argument("--out", default="cell.json")
    for name in ("run", "check", "register"):
        q = sp.add_parser(name)
        q.add_argument("--cell", required=True)
        if name == "run":
            q.add_argument("--dry-run", action="store_true")
        if name in ("check", "register"):
            q.add_argument("--prover-run", required=True)
            q.add_argument("--verifier-run", default=None)
        if name == "check":
            q.add_argument("--result", default=None, help="the result document (default: the prover run's result.json)")
        if name == "register":
            q.add_argument("--work", default=".bench-cell")
            q.add_argument("--dry-run", action="store_true")
            q.add_argument("--force", action="store_true")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.cmd == "plan":
        try:
            cell = plan(a.backend, a.statement, a.input_set, a.prover, relation=a.relation, verifier=a.verifier, verifier_addr=a.verifier_addr,
                        rtt_target=a.rtt_target, campaign=a.campaign, lane=a.lane, points=a.points, runs=a.runs, per_proof=a.per_proof,
                        backend_args=a.backend_arg, placement=json.loads(Path(a.placement).read_text()) if a.placement else {})
        except NotSupported as e:
            print(json.dumps({"backend": a.backend, "statement": a.statement, "input_set": a.input_set, "n/s": str(e)}))
            return 3
        Path(a.out).write_text(cell.to_json())
        for j in cell.jobs:
            print(f"# {j['role']} on {j['pod']}\n{shlex.join(j['argv'])}")
        return 0
    cell = Cell.load(a.cell)
    if a.cmd == "run":
        res = launch(cell, a.dry_run)
        print(json.dumps(res, indent=1))
        runs = [r.get("run") for r in res if r.get("run")]
        if runs and cell.lane:
            print(f'research notes checkpoint {cell.lane} open "WAITING {" ".join(runs)} on {cell.prover}; next: '
                  f'python -m verity_numerical.bench.cell register --cell {a.cell} --prover-run <prover run> --verifier-run <verifier run>"')
        return 0 if all(r.get("rc", 0) == 0 for r in res) else 1
    if a.cmd == "check":
        pd = run_dir(a.prover_run)
        vd = run_dir(a.verifier_run) if a.verifier_run else None
        doc = json.loads(Path(a.result or pd / drivers.driver(cell.backend).RESULT).read_text())
        problems = check(cell, doc, pd, vd)
        print(json.dumps({"ok": not problems, "problems": problems}, indent=1))
        return 0 if not problems else 1
    res = register(cell, a.prover_run, a.verifier_run, Path(a.work), a.dry_run, a.force)
    print(json.dumps(res, indent=1, default=str))
    return 0 if not res.get("refused") and res.get("rc", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
