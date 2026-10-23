"""The perplexity bench: `verity bench ppl COMMAND ...` (README.md says what it measures and how).

    run.py eval --model M --set c4|wikitext2 --ctx 2048|8192|32768 --out DIR [--gpus N] [--seeds 0,1,2]
                [--data DIR] [--hf-home DIR] [--plugin FILE.py]... SPEC...     on a GPU node: one results.json
    run.py submit (eval's flags but --out) [--max-min M] [--dry-run]        the same as a `research run` on vy-nebius-2
                                                                            (job.sh), the datasets sent with it
    run.py report RESULTS.json...                                           the tables
    run.py baseline [RESULTS.json...]                                       the locked baseline (against new results)
    run.py baseline --lock RESULTS.json...                                  lock those points in baseline.json
    run.py steps | specs                                                    the registered steps; the arms in specs/
    run.py tables [steps|arms|windows|models|baseline]...                   the README's tables, as Markdown
    run.py windows --model M [--data DIR] [--write]                         every window list's digest (and the lock)
    run.py setup --data DIR                                                 stage the datasets (datasets.json) there
    run.py selftest                                                         the engine on a tiny model, CPU or GPU

SPEC is a TOML file or the name of one in specs/. The bf16 reference and the plain int8 arm of each SPEC's scope run
with it. A run whose gates fail writes its gates and no perplexity, and exits 1."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import methodology  # noqa: E402
import models  # noqa: E402
import report  # noqa: E402
import spec  # noqa: E402
import tables  # noqa: E402
import windows  # noqa: E402

ROOT = HERE.parents[1]
DATA = Path(os.environ.get("PPL_DATA", "/workspace/ppl-data"))
LOCAL_DATA = Path.home() / ".cache" / "verity-ppl"
HF_HOME = Path(os.environ.get("HF_HOME", "/workspace/hf"))
NODE = "vy-nebius-2"


def load_plugins(paths) -> None:
    for p in paths or []:
        s = importlib.util.spec_from_file_location(Path(p).stem, p)
        mod = importlib.util.module_from_spec(s)
        s.loader.exec_module(mod)


def arms_for(refs: list[str]) -> list:
    """The SPECs, and the plain int8 arm of each one's scope, in that order (each once)."""
    import steps  # noqa: F401
    if any(Path(r).stem == spec.BF16.name for r in refs):
        raise ValueError("bf16 is the reference; every run has it")
    arms = [spec.load(r) for r in refs]
    names = [a.name for a in arms]
    for a in list(arms):
        if a.plain and a.plain not in names:
            arms.append(spec.load(a.plain))
            names.append(a.plain)
    if len(set(names)) != len(names):
        raise ValueError(f"two arms share a name: {names}")
    return arms


def parse_seeds(text: str) -> tuple:
    seeds = tuple(int(s) for s in text.split(","))
    if seeds != methodology.SEEDS[:len(seeds)] or not seeds:
        raise ValueError(f"seeds are a prefix of {methodology.SEEDS}")
    return seeds


def tokenizer(path: Path):
    import transformers
    return windows.encoder(transformers.AutoTokenizer.from_pretrained(str(path)))


def snapshot(model: str) -> Path:
    """The model's snapshot under HF_HOME; a streamed model's small files are put there first (`stream.ensure`)."""
    snap = models.snapshot(model, HF_HOME)
    pin = models.pin(model)
    if pin is not None:
        import stream
        stream.ensure(pin, snap)
    return snap


def check_ctx(model: str, ctx: int) -> None:
    top = models.get(model).get("max_ctx")
    if top is not None and ctx > top:
        raise SystemExit(f"{model} runs at contexts up to {top} (models.py)")


def build_windows(model: str, set_name: str, ctx: int, data: Path, calibration: bool):
    encode, bos = tokenizer(snapshot(model))
    fam = models.get(model)["tokenizer"]
    texts: dict = {}
    w, rec = windows.build(set_name, ctx, encode, bos, data, texts=texts)
    windows.check(fam, set_name, ctx, w)
    cal = None
    if calibration:
        c = methodology.CALIBRATION
        cal, _ = windows.build(c["set"], c["ctx"], encode, bos, data, calibration=True, texts=texts)
        windows.check(fam, c["set"], c["ctx"], cal, calibration=True)
    return w, cal, rec


def visible_gpus() -> list[str]:
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [x for x in env.split(",") if x]
    import torch
    return [str(i) for i in range(torch.cuda.device_count())]


def cmd_eval(a) -> int:
    import numpy as np
    load_plugins(a.plugin)
    seeds = parse_seeds(a.seeds)
    arms = arms_for(a.spec)
    methodology.windows(a.ctx)
    check_ctx(a.model, a.ctx)
    if a.set not in methodology.EVAL_SETS:
        raise SystemExit(f"--set is one of {methodology.EVAL_SETS}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    needs_cal = any(x.needs_calib for x in arms)
    ids, cal, wrec = build_windows(a.model, a.set, a.ctx, Path(a.data), needs_cal)
    snap = snapshot(a.model)
    gpus = a.gpus or models.get(a.model)["gpus"]
    devs = visible_gpus()
    if len(devs) < gpus:
        raise SystemExit(f"{gpus} GPUs asked, {len(devs)} visible")
    parts = __import__("engine").shards(len(ids), gpus)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        np.save(tmp / "ids.npy", ids)
        if cal is not None:
            np.save(tmp / "cal.npy", cal)
        procs = []
        for i, part in enumerate(parts):
            task = {"snapshot": str(snap), "ids": str(tmp / "ids.npy"), "cal": str(tmp / "cal.npy") if cal is not None else None,
                    "windows": part, "arms": a.spec, "plugins": a.plugin or [], "seeds": list(seeds),
                    "out": str(tmp / f"shard{i}"), "device": "cuda"}
            (tmp / f"task{i}.json").write_text(json.dumps(task))
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": devs[i]}
            log = open(out / f"shard{i}.log", "w")
            procs.append((subprocess.Popen([sys.executable, str(HERE / "run.py"), "_shard", str(tmp / f"task{i}.json")],
                                           env=env, stdout=log, stderr=subprocess.STDOUT), log))
        codes = [p.wait() for p, _ in procs]
        for _, log in procs:
            log.close()
        if any(codes):
            raise SystemExit(f"shards exited {codes} (see {out}/shard*.log)")
        shard_out = [json.loads((tmp / f"shard{i}.json").read_text()) for i in range(len(parts))]
        nll = {}
        for i in range(len(parts)):
            with np.load(tmp / f"shard{i}.npz") as z:
                for k in z.files:
                    nll.setdefault(k, []).append(z[k])
        nll = {k.replace("|", "/"): np.concatenate(v) for k, v in nll.items()}
    res = merge(a, arms, seeds, wrec, shard_out, nll, round(time.time() - t0, 1), len(parts))
    np.savez_compressed(out / "nll.npz", **{k.replace("/", "|"): v for k, v in nll.items()} if res["passed"] else {})
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(report.table(res))
    (out / "table.md").write_text(report.table(res) + "\n")
    return 0 if res["passed"] else 1


def merge(a, arms, seeds, wrec, shard_out, nll, seconds, n_gpus) -> dict:
    import numpy as np
    import torch
    from gates import Gates
    g = Gates()
    for s in shard_out:
        g.merge(s["gates"])
    n = methodology.CONTEXTS[a.ctx]
    g.record("windows", True, "lock", wrec["sha256"])
    g.record("tokens", all(v.shape == (n, a.ctx - 1) for v in nll.values()), "merged",
             {k: list(v.shape) for k, v in nll.items()})
    if g.log["hf_reference"]["checked"] == 0:
        g.record("hf_reference", False, "check window", g.log["hf_reference"].get("skipped", "never compared"))
    passed = g.passed
    res = {"schema": "ppl-bench/v1", "model": a.model, "set": a.set, "ctx": a.ctx, "n_windows": n,
           "windows": [0, n - 1], "scored_tokens": methodology.scored_tokens(a.ctx), "window_digest": wrec["sha256"],
           "seeds": list(seeds), "gates": g.log, "passed": passed, "gpus": n_gpus, "seconds": seconds,
           "env": {"torch": torch.__version__, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                   "run": os.environ.get("RESEARCH_RUN_ID", "")},
           "arms": {}}
    all_arms = [spec.BF16] + list(arms)
    for arm in all_arms:
        entry = {"spec": arm.to_toml(), "scope": arm.scope, "plain": arm.plain}
        keys = [f"{arm.name}/s{s}" for s in arm.seeds(seeds)]
        if passed:
            entry["seeds"] = {}
            for k in keys:
                v = nll[k]
                entry["seeds"][k.split("/s")[-1]] = {"ppl": float(np.exp(v.mean(dtype=np.float64))),
                                                     "nll": float(v.mean(dtype=np.float64)),
                                                     "window_nll": [float(x) for x in v.mean(axis=1, dtype=np.float64)]}
        entry["divergence_last"] = {k: s["divergence"][k][-1] for s in shard_out for k in keys if k in s["divergence"]}
        entry["attn_rel_err_w0"] = {k: s["attn_rel_err"][k] for s in shard_out for k in keys
                                    if s["attn_rel_err"].get(k) is not None}
        res["arms"][arm.name] = entry
    return res


def cmd_shard(task_path: str) -> int:
    import numpy as np
    import engine
    t = json.loads(Path(task_path).read_text())
    load_plugins(t["plugins"])
    arms = arms_for(t["arms"])
    ids = np.load(t["ids"])[t["windows"]]
    cal = np.load(t["cal"]) if t["cal"] else None
    r = engine.evaluate(Path(t["snapshot"]), ids, t["windows"], arms, tuple(t["seeds"]), cal, device=t["device"],
                        log=lambda s: print(s, flush=True))
    np.savez(t["out"] + ".npz", **{k.replace("/", "|"): v for k, v in r.pop("nll").items()})
    Path(t["out"] + ".json").write_text(json.dumps(r))
    return 0


def submit_command(a, data: Path) -> list[str]:
    """The `research run` that runs this eval on vy-nebius-2 under a gpu-lease of the model's GPUs, from the
    checkout's pushed commit, with the datasets (staged under `data`) sent along. A plugin or SPEC file must be in
    the commit (paths relative to the repository's root)."""
    gpus = a.gpus or models.get(a.model)["gpus"]
    if gpus > 2:
        raise SystemExit("a run leases at most 2 GPUs")
    flags = f"--model {a.model} --set {a.set} --ctx {a.ctx} --gpus {gpus} --seeds {a.seeds}"
    flags += "".join(f" --plugin {shlex.quote(p)}" for p in a.plugin or [])
    flags += " " + " ".join(shlex.quote(s) for s in a.spec)
    who = os.environ.get("GPU_LEASE_WHO", "ppl-bench")
    sends = [x for d in windows.DATASETS.values() for x in ("--send", str(Path(data) / d["file"]))]
    return ["research", "run", "--on", NODE, "--project", "verity", "--campaign", a.campaign, "--source", str(ROOT),
            "--cwd", "source", "--custody-ttl", "12h", *sends, "--env", f"GPU_LEASE_WHO={who}",
            "--env", f"PPL_GPUS={gpus}", "--env", f"PPL_MAX_MIN={a.max_min}", "--env", f"PPL_ARGS={flags}",
            "--", "bash", "benchmarks/ppl/job.sh"]


def cmd_submit(a) -> int:
    load_plugins(a.plugin)
    arms_for(a.spec)
    check_ctx(a.model, a.ctx)
    cmd = submit_command(a, Path(a.data))
    print(shlex.join(cmd))
    if a.dry_run:
        return 0
    stage(Path(a.data))
    return subprocess.run(cmd).returncode


def cmd_windows(a) -> int:
    fam = models.get(a.model)["tokenizer"]
    encode, bos = tokenizer(Path(a.tokenizer) if a.tokenizer else snapshot(a.model))
    lock = windows.lock()
    mine = lock.setdefault(fam, {})
    texts: dict = {}
    for s in methodology.EVAL_SETS:
        for ctx in methodology.CONTEXTS:
            _, rec = windows.build(s, ctx, encode, bos, Path(a.data), texts=texts)
            mine[windows.key(s, ctx)] = rec
            print(fam, windows.key(s, ctx), json.dumps(rec))
    c = methodology.CALIBRATION
    _, rec = windows.build(c["set"], c["ctx"], encode, bos, Path(a.data), calibration=True, texts=texts)
    mine[windows.key(c["set"], c["ctx"], True)] = rec
    print(fam, windows.key(c["set"], c["ctx"], True), json.dumps(rec))
    if a.write:
        windows.LOCK_PATH.write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n")
    elif a.out:
        Path(a.out).write_text(json.dumps({fam: mine}, indent=1, sort_keys=True) + "\n")
    return 0


def cmd_setup(a) -> int:
    return stage(Path(a.data))


def stage(data: Path) -> int:
    """Each dataset in datasets.json under `data`: kept if its sha256 is the pinned one, else fetched from the
    evidence store by its art id."""
    data.mkdir(parents=True, exist_ok=True)
    for name, d in windows.DATASETS.items():
        path = data / d["file"]
        if path.exists() and windows.file_sha256(path) == d["sha256"]:
            print(f"{name}: {path} ok")
            continue
        got = subprocess.run(["research", "data", "fetch", d["art"], "--to", str(data)], capture_output=True, text=True)
        if got.returncode:
            raise SystemExit(f"{name}: research data fetch {d['art']}: {got.stderr.strip()}")
        fetched = Path(got.stdout.strip().splitlines()[-1])
        if fetched != path:
            fetched.replace(path)
        windows.dataset_path(name, data)
        print(f"{name}: {path} fetched")
    return 0


def cmd_report(a) -> int:
    for r in report.load(a.results):
        print(report.table(r))
        print()
    return 0


BASELINE = tables.BASELINE
README = HERE / "README.md"


def cmd_baseline(a) -> int:
    lk = json.loads(BASELINE.read_text())
    if a.lock:
        lk = report.lock(lk, report.load(a.results))
        BASELINE.write_text(json.dumps(lk, indent=1) + "\n")
        print(report.baseline_table(lk))
        return 0
    if not a.results:
        print(report.baseline_table(lk))
        return 0
    text, ok = report.baseline_check(lk, report.load(a.results))
    print(text)
    return 0 if ok else 1


def cmd_steps(_a) -> int:
    for k, v in spec.catalogue().items():
        params = ", ".join(f"{p}={d!r}" if d is not None else p for p, d in v["params"].items())
        print(f"{k}({params})  [{v['phase']}{', on ' + '/'.join(v['on']) if k.startswith('attention') else ''}]")
        print("    " + v["doc"].split("\n\n")[0].replace("\n", "\n    "))
    return 0


def cmd_specs(_a) -> int:
    for p in sorted(spec.SPECS.glob("*.toml")):
        arm = spec.load(str(p))
        print(f"{arm.name:24} {arm.scope:9} {'seeded' if arm.seeded else 'deterministic':13} {arm.description}")
    return 0


def cmd_tables(a) -> int:
    print(tables.render(a.sections), end="")
    return 0


def cmd_selftest(_a) -> int:
    import pytest
    return pytest.main(["-q", str(HERE / "tests")])


def main(argv: list[str]) -> int:
    if argv[:1] == ["_shard"]:
        return cmd_shard(argv[1])
    p = argparse.ArgumentParser(prog="verity bench ppl", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def run_flags(q, out: bool):
        q.add_argument("--model", required=True, choices=sorted(models.MODELS))
        q.add_argument("--set", required=True, choices=methodology.EVAL_SETS)
        q.add_argument("--ctx", required=True, type=int, choices=sorted(methodology.CONTEXTS))
        q.add_argument("--gpus", type=int, default=0, help="GPUs (default: the model's in models.py)")
        q.add_argument("--seeds", default=",".join(map(str, methodology.SEEDS)))
        q.add_argument("--plugin", action="append", help="a .py file registering more steps")
        q.add_argument("spec", nargs="+")
        if out:
            q.add_argument("--out", required=True)
            q.add_argument("--data", default=str(DATA))

    run_flags(sub.add_parser("eval"), True)
    q = sub.add_parser("submit")
    run_flags(q, False)
    q.add_argument("--data", default=str(LOCAL_DATA), help="where the datasets are staged to be sent")
    q.add_argument("--max-min", type=int, default=120)
    q.add_argument("--campaign", default="cmp-13")
    q.add_argument("--dry-run", action="store_true")
    q = sub.add_parser("report")
    q.add_argument("results", nargs="+")
    q = sub.add_parser("baseline")
    q.add_argument("results", nargs="*")
    q.add_argument("--lock", action="store_true", help="lock these results' points in baseline.json")
    sub.add_parser("steps")
    sub.add_parser("specs")
    q = sub.add_parser("tables")
    q.add_argument("sections", nargs="*", metavar="SECTION", help=f"of {', '.join(tables.SECTIONS)} (default: all)")
    q = sub.add_parser("windows")
    q.add_argument("--model", required=True, choices=sorted(models.MODELS))
    q.add_argument("--data", default=str(DATA))
    q.add_argument("--tokenizer", help="a directory holding the model's tokenizer files (default: its snapshot)")
    q.add_argument("--write", action="store_true")
    q.add_argument("--out")
    q = sub.add_parser("setup")
    q.add_argument("--data", default=str(DATA))
    sub.add_parser("selftest")
    a = p.parse_args(argv)
    return {"eval": cmd_eval, "submit": cmd_submit, "report": cmd_report, "baseline": cmd_baseline, "steps": cmd_steps,
            "specs": cmd_specs, "tables": cmd_tables, "windows": cmd_windows, "setup": cmd_setup,
            "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
