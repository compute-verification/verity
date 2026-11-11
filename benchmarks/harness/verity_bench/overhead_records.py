"""`verity bench overhead`'s store side: it launches the node side (`overhead_node.sh`, `overhead_node`) as a recorded
`research run` on one GPU of a node, stores a finished sweep's points as a versioned art, labels a timed run of record, and
renders the table of γ from labelled runs alone.

A record is named in the store, never in code (`served_table`'s rule): the current points of a card, model and precision
are the newest art of kind `POINTS_KIND` labelled `run_of_record=overhead-points` and not `superseded_by` a newer one; a run
of record is a run labelled `run_of_record=overhead`.

**The source check** (`of_record`, `gamma_of_record`): an overhead of record comes only from a labelled run of the door. A
γ stands when its run is labelled, ran the door's node job as the door launches it (`door_why`), passed with exit 0 from a
clean tree, wrote an `overhead.json` of `RUN_SCHEMA` the
node side found nothing against, was timed at the points of an art of record that nothing supersedes, ran its arm's cubin
from the `art:` it was sent (`run --cubin ART`; the art's bytes a binary the run's SASS gate passed, `cubin_why`; a cubin
built in the run is development only), and its own point passed the door's rules (`overhead.point_whys`: every linear timed, at least 20 paired reps, an
interval within ±5%, the cross-check, the arm's reference verifier accepting every row). `table` refuses every other
figure and says why; a reader of γ outside this package (the protocol table, the one-pager, a goal) takes it from
`verity bench overhead table --json`, whose rows carry `of_record` and `why`.
"""

from __future__ import annotations

import json
import math
import shlex
import shutil
import subprocess
import tempfile
import tomllib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from verity_bench import jobs, settings
from verity_bench import overhead as O
from verity_bench.served_table import RUN_OF_RECORD, SUPERSEDED_BY, records

POINTS_RECORD, RUN_RECORD = "overhead-points", "overhead"
POINTS_KIND = "verity-bench-overhead-points/v1"
BY = "benchmarks"
#: each card's GPU node, where `--on` names none
NODES = {"h200": "vy-dc-h200"}
NODE_JOB = "benchmarks/harness/overhead_node.sh"
#: the options the door gives each node job (`launch_points`, `launch_run`); the node job sets the rest itself
DOOR_OPTIONS = {"points": {"--card", "--model", "--precision"},
                "run": {"--card", "--scheme", "--points", "--points-art", "--point", "--cubin", "--cubin-art"}}
#: the one variable a run of the door sets with `--env` (`launch_argv`): any other could change what the node job runs
DOOR_ENV = frozenset({"VERITY_BENCH_VENV"})
#: the run's bound, which its GPU lease enforces too: the sweep's and a timed run's
MINUTES = {"points": 150, "run": 60}
CENSUS = Path("benchmarks/census")
DATACENTER, WORLD = "dc-12500-machines-600s", "world-inference-2027"
#: the one conversion from FLOP to Boolean gates, with its source and status
GATES = Path("verity/protocols/inference_only/rho/data/hardware.toml")

Executor = Callable[[Sequence[str], Path], int]


def _execute(argv: Sequence[str], root: Path) -> int:
    return subprocess.run(list(argv), cwd=root, check=False).returncode


def gpu_lease(minutes: float) -> list[str]:
    """The wait for one GPU of the node, held for the run's bound. The node job takes it itself, since `research run --gpus`
    is refused on a node whose GPUs gpu-lease doesn't hand out (vy-dc-h200)."""

    return ["gpu-lease", "1", "--wait", "--max-min", str(math.ceil(minutes)), "--"]


def launch_argv(root: Path, node: str, minutes: float, job: Sequence[str], sends: Sequence[Path] = ()) -> list[str]:
    """The node side as a `research run` on one GPU of `node`, from this checkout."""

    env = [f"VERITY_BENCH_VENV={v}"] if (v := settings.VERITY_BENCH_VENV) else []
    return ["research", "run", "--on", node, "--project", "verity", "--campaign", "overhead",
            "--source", str(root), "--cwd", "source", "--timeout", f"{minutes:g}m", "--require-result",
            *[x for f in sends for x in ("--send", str(f))], *[x for e in env for x in ("--env", e)],
            "--", *gpu_lease(minutes), "bash", NODE_JOB, *job]


def launch_points(root: Path, node: str, card: str, model: str, precision: str, minutes: float | None = None,
                  execute: Executor | None = None) -> int:
    job = ["points", "--card", card, "--model", model, "--precision", precision]
    return (execute or _execute)(launch_argv(root, node, minutes or MINUTES["points"], job), root)


def launch_run(store: Any, root: Path, node: str, card: str, scheme: str, point: str, points: str | None = None,
               minutes: float | None = None, execute: Executor | None = None, cubin: str | None = None) -> int:
    """A scheme's timed run at the card's points of record (or the art `points`), which it ships to the node with the art
    `cubin`, the one cubin the scheme's arm loads (without it the run is development only)."""

    if cubin is not None:
        if O.scheme(scheme).arm is None:
            raise ValueError(f"--cubin: {scheme} has no arm to load it")
        try:
            payload = store.get_manifest(cubin).payload
        except Exception as ex:
            raise ValueError(f"--cubin {cubin} isn't in the store ({type(ex).__name__}: {ex})") from ex
        if getattr(payload, "type", None) != "file":
            raise ValueError(f"--cubin {cubin} isn't one file: it names the art of the one cubin the arm loads")
    else:
        O.scheme(scheme)
    art = points or current_points(store, card)
    if art is None:
        raise ValueError(f"no points of record for the {card}: run `verity bench overhead points --card {card} --on NODE`, "
                         "then `verity bench overhead points --store RUN`")
    if store.get_manifest(art).meta.get("card") != card:
        raise ValueError(f"{art} holds the points of the {store.get_manifest(art).meta.get('card')}, not the {card}")
    with tempfile.TemporaryDirectory() as tmp:
        sent = [Path(tmp) / "points.json"]
        shutil.copyfile(store.fetch(art), sent[0])
        job = ["run", "--card", card, "--scheme", scheme, "--points", "points.json", "--points-art", art, "--point", point]
        if cubin is not None:
            (Path(tmp) / "cubin").mkdir()
            sent.append(Path(store.fetch(cubin, Path(tmp) / "cubin")))
            job += ["--cubin", sent[-1].name, "--cubin-art", cubin]
        return (execute or _execute)(launch_argv(root, node, minutes or MINUTES["run"], job, sent), root)


# ---- reading the store -------------------------------------------------------------------------------------------------------


def door_why(attempt: Mapping, run: str, sub: str) -> str | None:
    """Why `run` isn't a run of the door's `sub`: its command isn't the node job `bash NODE_JOB sub` with the door's options
    (`DOOR_OPTIONS`), under the gpu-lease that holds its GPU (the door's own, or `research run --gpus`'s) or alone, or it set
    a variable with `--env` the door doesn't (`DOOR_ENV`)."""

    argv = [str(x) for x in attempt.get("argv") or []]
    cmd = argv[argv.index("--") + 1:] if argv[:1] == ["gpu-lease"] and "--" in argv else argv
    options = {x.partition("=")[0] for x in cmd[3:] if x.startswith("-")}
    if cmd[:3] != ["bash", NODE_JOB, sub] or options - DOOR_OPTIONS[sub]:
        return (f"{run} isn't a run of the door: it ran `{shlex.join(argv)[:300]}`, not `bash {NODE_JOB} {sub}` with the "
                f"door's options (launch it with `verity bench overhead {sub}`)")
    if extra := sorted(set((attempt.get("params") or {}).get("env") or {}) - DOOR_ENV):
        return f"{run} set {', '.join(extra)} with --env, which a run of the door doesn't: it sets only {', '.join(sorted(DOOR_ENV))}"
    return None


def run_files(store: Any, run: str, sub: str) -> tuple[dict | None, str | None]:
    """A run's attempt and its run files' art, or (None, why) when it isn't a passing run of the door's `sub` from a clean
    tree."""

    attempt = store.get_attempt(run)
    if attempt is None:
        return None, f"{run} is not in the store or on its remote"
    if why := door_why(attempt, run, sub):
        return None, why
    status, source, out = attempt.get("status") or {}, attempt.get("source") or {}, attempt.get("outputs") or {}
    if status.get("validation") != "passed" or status.get("rc") != 0 or source.get("dirty"):
        return None, (f"{run} isn't a passing run from a clean tree (validation {status.get('validation')}, rc "
                      f"{status.get('rc')}, dirty {bool(source.get('dirty'))})")
    files = out.get("run_files") or out.get("run_record")
    if not files:
        return None, f"{run} kept no run files"
    return attempt, files


def _json(store: Any, files: str, path: str) -> dict:
    return json.loads((Path(store.fetch(files, paths=[path])) / path).read_text())


def _superseded(store: Any, target: str) -> list[str]:
    return [str(lab.value) for lab in store.labels(target, key=SUPERSEDED_BY)]


def current_points(store: Any, card: str, model: str = O.KERNEL_MODEL, precision: str = "fp8") -> str | None:
    """The points art of record of a card, model and precision: the newest such art labelled, and not superseded."""

    found = [art for art in records(store).get(POINTS_RECORD, []) if art.startswith("art:")
             and {k: store.get_manifest(art).meta.get(k) for k in ("card", "model", "precision")}
             == {"card": card, "model": model, "precision": precision} and not _superseded(store, art)]
    return max(found, key=lambda art: store.get_manifest(art).meta.get("version", 0), default=None)


def of_record(store: Any, run: str) -> tuple[dict | None, list[str]]:
    """A run's `overhead.json` and why it can't stand as a whole (each point's own whys are in it), or (None, whys)."""

    attempt, files = run_files(store, run, "run")
    if attempt is None:
        return None, [files]
    try:
        doc = _json(store, files, "out/overhead.json")
    except (OSError, ValueError, KeyError) as ex:
        return None, [f"{run} has no readable out/overhead.json ({type(ex).__name__}: {ex})"]
    whys = []
    if doc.get("schema") != O.RUN_SCHEMA:
        whys.append(f"{run}'s overhead.json is {doc.get('schema')!r}, not {O.RUN_SCHEMA}")
    if RUN_RECORD not in [lab.value for lab in store.labels(run, key=RUN_OF_RECORD)]:
        whys.append(f"{run} isn't labelled {RUN_OF_RECORD}={RUN_RECORD}")
    whys += doc.get("whys") or []
    art = doc.get("points_art")
    if not str(art or "").startswith("art:"):
        whys.append("its points come from no stored art")
    else:
        m = store.get_manifest(art)
        if POINTS_RECORD not in [lab.value for lab in store.labels(art, key=RUN_OF_RECORD)]:
            whys.append(f"its points {art} aren't of record")
        if later := _superseded(store, art):
            whys.append(f"its points {art} are superseded by {', '.join(later)}: time it at the current points")
        if m.payload is None or m.payload.sha256 != doc.get("points_sha256"):
            whys.append(f"the points it was timed at aren't {art}'s")
    if (arm := doc.get("arm")) and str(arm.get("cubin_art") or "").startswith("art:") and (why := cubin_why(store, arm)):
        whys.append(why)
    return doc, whys


def cubin_why(store: Any, arm: Mapping) -> str | None:
    """Why the arm's `cubin_art` isn't a binary the run's SASS gate passed (its bytes, by sha256), or None."""

    art = arm["cubin_art"]
    try:
        m = store.get_manifest(art)
    except Exception as ex:  # noqa: BLE001 - the store's NotFound, named in the why
        return f"its cubin {art} isn't in the store ({type(ex).__name__}: {ex})"
    blobs = {sha for sha, _ in m.payload.blobs()} if m.payload else set()
    if not blobs & {b.get("sha256") for b in arm.get("gated") or [] if b.get("ok")}:
        return f"its cubin {art} isn't a binary the run's SASS gate passed"
    return None


def gamma_rows(store: Any) -> list[dict]:
    """Every (scheme, card, point)'s γ from the newest labelled run that times it, of record or with why not."""

    best: dict[tuple, dict] = {}
    for run in records(store).get(RUN_RECORD, []):
        if run.startswith("art:"):
            continue
        doc, whys = of_record(store, run)
        if doc is None:
            best.setdefault(("?", "?", run), {"scheme": None, "card": None, "point": None, "run": run, "of_record": False,
                                              "why": "; ".join(whys)})
            continue
        for name, p in (doc.get("points") or {}).items():
            row = {"scheme": doc.get("scheme"), "card": doc.get("card"), "point": name, "m": p.get("m"), "run": run,
                   "points_art": doc.get("points_art"), "gamma": p.get("gamma"), "gamma_least_sm": p.get("gamma_least_sm"),
                   "gamma_weight_side_bound": p.get("gamma_weight_side_bound"), "aa": p.get("aa"),
                   "epsilon": p.get("epsilon"), "perplexity": p.get("perplexity"), "gate": p.get("gate"),
                   "linears": p.get("linears")}
            why = whys + list(p.get("whys") or [])
            row |= {"of_record": not why, "why": "; ".join(why) or None}
            best[(row["scheme"], row["card"], name)] = row
    return [best[k] for k in sorted(best, key=lambda k: tuple(map(str, k)))]


def gamma_of_record(store: Any, scheme: str, card: str, point: str) -> dict:
    """The γ of record of a scheme at a card's point, or ValueError naming why there is none: what a reader of γ calls."""

    row = next((r for r in gamma_rows(store) if (r["scheme"], r["card"], r["point"]) == (scheme, card, point)), None)
    if row is None:
        raise ValueError(f"no labelled run of `verity bench overhead run` times {scheme} at the {card}'s {point}")
    if not row["of_record"]:
        raise ValueError(f"{scheme} at the {card}'s {point} (run {row['run']}) isn't of record: {row['why']}")
    return row


# ---- writing the store -------------------------------------------------------------------------------------------------------


def store_points(store: Any, run: str, by: str = BY) -> str:
    """A finished sweep's points.json as the next version of its card's, model's and precision's points art, labelled of
    record; the version it replaces is labelled `superseded_by` it."""

    attempt, files = run_files(store, run, "points")
    if attempt is None:
        raise ValueError(files)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(store.fetch(files, paths=["out/points.json"])) / "out" / "points.json"
        doc = json.loads(path.read_text())
        if doc.get("schema") != O.POINTS_SCHEMA or doc.get("error") or None in (doc.get("c"), doc.get("l")):
            raise ValueError(f"{run}'s points.json isn't a finished sweep: {doc.get('error') or doc.get('schema')}")
        if doc.get("rule") != O.RULE:
            raise ValueError(f"{run} swept under the rule {doc.get('rule')}, not the door's {O.RULE}")
        prior = current_points(store, doc["card"], doc["model"], doc["precision"])
        version = 1 + (store.get_manifest(prior).meta.get("version", 0) if prior else 0)
        meta = {k: doc[k] for k in ("card", "model", "precision", "c", "l")} | {"run": run, "version": version,
                                                                              "supersedes": prior}
        copy = Path(tmp) / "points.json"
        shutil.copyfile(path, copy)
        art = store.put_artifact(POINTS_KIND, meta, file=copy)
    store.push(art)
    store.add_label(art, RUN_OF_RECORD, POINTS_RECORD, by=by)
    if prior:
        store.add_label(prior, SUPERSEDED_BY, art, by=by)
    store.push_labels([art, *([prior] if prior else [])])
    return art


def record_run(store: Any, run: str, by: str = BY) -> list[str]:
    """Labels a timed run of record when the run as a whole stands; returns its points' whys (a point that doesn't stand
    stays in the table, refused, with its why)."""

    doc, whys = of_record(store, run)
    whys = [w for w in whys if not w.endswith(f"isn't labelled {RUN_OF_RECORD}={RUN_RECORD}")]
    if doc is None or whys:
        raise ValueError(f"{run} can't be of record: {'; '.join(whys)}")
    store.add_label(run, RUN_OF_RECORD, RUN_RECORD, by=by)
    store.push_labels([run])
    return [f"{n}: {'; '.join(p['whys'])}" for n, p in (doc.get("points") or {}).items() if p.get("whys")]


# ---- the table ---------------------------------------------------------------------------------------------------------------


def census(root: Path) -> dict:
    """The stated parameters the table scales γ by: the census's datacenter and world, each card's datasheet rate, and the
    one FLOP-to-gates conversion."""

    def ids(name: str) -> dict:
        return {d["id"]: d for d in json.loads((root / CENSUS / name).read_text())}

    deployments, world, hardware = ids("deployments.json"), ids("adversary.json"), ids("hardware.json")
    dc = deployments[DATACENTER]
    gates = tomllib.loads((root / GATES).read_text())["units"]
    return {"datacenter": DATACENTER, "gpus": dc["machines"] * dc["gpus_per_machine"], "window_s": dc["window_s"],
            "world": WORLD, "world_flop_per_year": world[WORLD]["total_flop"],
            "flop_per_s": {c.gpu: hardware[c.datasheet]["flop_per_second"] for c in O.CARDS.values()},
            "gates_per_flop": gates["gates_per_flop"]["value"],
            "gates_basis": f"{GATES} units.gates_per_flop ({gates['gates_per_flop']['status']}, {gates['basis']})"}


def build(store: Any = None, *, root: Path | None = None, local_only: bool = False) -> dict:
    root = root or jobs.repository_root()
    completion = jobs.StoreCompletion(store, root=root, local_only=local_only)
    if not local_only:
        completion.pull_labels()
    rows = gamma_rows(completion.store)
    c = census(root)
    for row in rows:
        g = (row.get("gamma") or {}).get("gamma")
        sch = O.SCHEMES.get(row["scheme"] or "")
        if row["of_record"] and g is not None and sch is not None and sch.arm:
            row["periods"] = O.periods(g, c["gpus"], c["window_s"], c["world_flop_per_year"], c["flop_per_s"][row["card"]],
                                       c["gates_per_flop"])
    return {"schema": "verity-bench/overhead-table/v1", "census": c, "rows": rows}


def _g(x: Mapping | None) -> str:
    if not x or x.get("gamma") is None:
        return "—"
    ci = x.get("ci95")
    return f"{x['gamma']:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")


def _eps(e: Mapping | None) -> str:
    if not e:
        return "—"
    if "why" in e:
        return f"— ({e['why']})"
    worst = max((v.get("epsilon") for v in e.values() if isinstance(v, Mapping) and v.get("epsilon") is not None),
                default=None)
    return "—" if worst is None else f"≤ {worst:.3g} (worst linear)"


def render(table: Mapping) -> str:
    c = table["census"]
    lines = ["| Scheme | Card | Point | m | γ [95% CI] | over least-SM stock | weight side bound | A/A | ε | γ ≤ 10 | Run |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    refused = []
    for r in table["rows"]:
        if not r["of_record"]:
            refused.append(r)
            continue
        gate = r.get("gate") or {}
        lines.append(f"| {r['scheme']} | {r['card']} | {r['point']} | {r['m']} | {_g(r['gamma'])} | {_g(r['gamma_least_sm'])} "
                     f"| {_g(r['gamma_weight_side_bound'])} | {_g(r['aa'])} | {_eps(r['epsilon'])} | "
                     f"{'yes' if gate.get('passes') else 'no'} | {r['run']} |")
    lines += ["", f"Per monitoring period, over all compute: the {c['datacenter']} datacenter's {c['gpus']:,} GPUs in "
              f"{c['window_s']} s windows, spending the share γ/(1+γ) of its compute on proving, and the world's "
              f"{c['world_flop_per_year']:.3g} FLOP a year of useful work ({c['world']}), to which proving adds γ times it. "
              "γ is measured on one card at one operating point; every total is scaled from it, not proved. Gates at "
              f"{c['gates_per_flop']:g} a FLOP ({c['gates_basis']}); the datacenter's FLOP at its card's datasheet rate.", "",
              "| Scheme | Card | Point | Share | Period | DC GPU-hours | DC gates | World FLOP | World gates |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in table["rows"]:
        if p := r.get("periods"):
            for n in ("window", *O.PERIODS_H):
                q = p[n]
                lines.append(f"| {r['scheme']} | {r['card']} | {r['point']} | {p['share']:.2%} | {n} | "
                             f"{q['datacenter_gpu_h']:.3g} | {q['datacenter_gates']:.3g} | {q['world_flop']:.3g} | "
                             f"{q['world_gates']:.3g} |")
    if refused:
        lines += ["", "Refused (not of record):", ""]
        lines += [f"- {r['scheme']} at {r['card']} {r['point']} (run {r['run']}): {r['why']}" for r in refused]
    return "\n".join(lines) + "\n"
