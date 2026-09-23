"""`research` command line.

  research run [--project P] [--stage S] [--timeout T] [--expect K=V ...] [--require-result] [--runs-dir D] [harness flags] -- CMD...
  research run [--tool NAME|module:NAME] [--scratch NAME ...] [--campaign C] [...] -- CMD...   # evidence store: derivation before, attempt after
  research run --on MACHINE --project P [--source GITDIR] [--send FILE ...] [--env K=V ...] [--exclusive] [--mem-gb N [--mem-wait S]] [...] -- CMD...
  research fetch <run-id> [--all]                # copy the machine's records (or every file, verified) into the local run dir
  research status [--runs-dir D] [--refresh] [--balance] [--all] [--json]
  research inspect <run-id|prefix|path> [--runs-dir D] [--json]
  research compare <baseline> <candidate> [--json]   # differences and candidate/baseline ratios; the workload contract judges them
  research tele <telemetry subcommand...>        # the full harness surface (init, event, sample, cancel-intent, classify, ...)
  research data <store subcommand...>            # the evidence store: put show where fetch verify select snapshot labels label reindex sql
  research pods check-part <machine|pod-id>      # reference-part gate: name, VRAM vs [families.<fam>] vram_mb, host CPU, load, verdict (exit 3 = variant)
  research pods create <runpod flags> [--require-reference-part]   # research.pods.runpod create, then the same verdict once sshd answers
  research pods unpreserved <machine|pod-id>     # Attempts recorded on that machine that are NOT `preserved` (same predicate as `data preserved`); exit 1 if any
  research pods drain <machine|pod-id> [--force --reason TEXT]      # the report, then terminate ONLY when nothing is unpreserved (override logged)
  research env --source TREE [--rev REV] [--json]  # `export PYTHONPATH=...` from TREE's [tool.uv.workspace] members (research.pythonpath)
  research launch-request <request.json>         # machine side: claim the run dir, start the detached runner (idempotent per id)
  research mem [--ramlock DIR] [--json]           # machine side: cgroup limit, in use, live reservations, available (research.mem)

`run` allocates a run id, writes the v0.2 job record, gives the workload RESEARCH_RUN_ID / RESEARCH_RUN_DIR, and hands execution
to the telemetry harness (timeout, sampler, cancel evidence, classifier, tee'd logs).  The workload's cwd is its run dir, locally and
with --on; `--cwd source` (or `source/SUB`, a relative SUB) runs it in the source root (the invoking directory; with --on, the shipped
tree other runs of the commit share), an absolute --cwd as given; job.json run.cwd / run.cwd_basis record it.  It exits with the workload's status; with
--require-result, a zero exit and a missing or invalid result.json exits 65 instead.  Harness flags pass through unchanged
(--tree-sha, --workload-digest, --input name=path, --declared-output, --policy, --no-sampler, ...).
`--on <machine>` ships the tool (and, with --source, a clean git HEAD), stages the request, and has the machine claim the run id
and start the same runner detached; see remote.py.  A machine must be listed in ~/.research/machines.toml for the project.
`--tool` loads a store Tool declaration (research/store/tools_registry.py or dotted.module:NAME): before launch the runner records
conditions, the closure manifest, content-resolved key params, the derivation id and reuse_key into job.json (`store` section);
after the workload exits (any status) it publishes the attempt to the local store (research/store/README.md §4, §10).  `--scratch NAME`
gives the workload a cache dir keyed by reuse_key.  `research data ...` is the store's own surface (research.store.cli / cli_prov).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import compare as C
from . import record, remote, result
from . import runs as R

EX_DATAERR = 65


def _harness_parser(prog: str) -> argparse.ArgumentParser:
    from .telemetry.run import add_args

    ap = argparse.ArgumentParser(prog=prog, add_help=False, allow_abbrev=False)
    add_args(ap)
    return ap


def build_job_from(a: argparse.Namespace, command: list[str]) -> dict:
    """Mirror of the job the harness would build for this namespace (telemetry.run.run), so that pre-writing job.json changes
    nothing the harness records.  Kept in one place; if the harness's call changes upstream, change this."""
    from .telemetry.job import build_job, input_ref, sha256_file

    policy_sha = sha256_file(a.policy)[0] if a.policy and Path(a.policy).exists() else None
    policy_version = a.policy_version
    if a.policy and Path(a.policy).exists() and not policy_version:
        try:
            import tomllib

            policy_version = tomllib.loads(Path(a.policy).read_text()).get("policy", {}).get("version")
        except Exception:  # noqa: BLE001
            policy_version = None
    return build_job(project=a.project, task=a.task, job=a.job, attempt=a.attempt, lane=a.lane, generation=a.generation, tree_sha=a.tree_sha,
                     release_tag=a.release_tag, workload_digest=a.workload_digest, stage=a.stage, inputs=[input_ref(x) for x in a.input],
                     profile={k: os.environ.get(k) for k in a.profile_key if k in os.environ}, command=command, declared_outputs=a.declared_output,
                     timeout_s=a.timeout, reservation={"gpu": a.reservation_gpu, "host_memory_bytes": a.reservation_host_memory_bytes},
                     policy_version=policy_version, policy_sha256=policy_sha,
                     instruments=["cgroup", "process", "pss", "gpu", "storage"] if not a.no_sampler else [], cadence_s=a.cadence, pss_every_s=a.pss_every)


def _parse_expect(items: list[str]) -> dict | None:
    if not items:
        return None
    out = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep or not k:
            raise SystemExit(f"research run: --expect wants KEY=VALUE, got {it!r}")
        out[k] = v
    return out


def cmd_run(argv: list[str]) -> int:
    if "--" not in argv:
        print("research run: the workload command goes after `--`", file=sys.stderr)
        return 2
    i = argv.index("--")
    opts, command = argv[:i], argv[i + 1 :]
    if not command:
        print("research run: empty command after `--`", file=sys.stderr)
        return 2
    ap = argparse.ArgumentParser(prog="research run", add_help=True, allow_abbrev=False, description=__doc__.split("\n\n")[1])
    ap.add_argument("--runs-dir", default=None, help="registry root (default $RESEARCH_RUNS or ~/.research/runs)")
    ap.add_argument("--on", default=None, metavar="MACHINE", help="run on an existing authorized machine listed in the machines file")
    ap.add_argument("--project", default=None, help="ownership/resource boundary; required with --on (also recorded by the harness)")
    ap.add_argument("--source", default=None, metavar="GITDIR", help="with --on: ship `git archive HEAD` of this clean checkout; it becomes the source root "
                                                                       "and tree_sha (the workload still runs in its run dir unless --cwd source)")
    ap.add_argument("--send", action="append", default=[], metavar="FILE", help="with --on: ship this file to $RESEARCH_RUN_DIR/inputs/")
    ap.add_argument("--env", action="append", default=[], metavar="K=V", help="environment for the workload (local or, with --on, on the machine), recorded as "
                                                                                 "the attempt's params.env (values of secret-named keys redacted)")
    ap.add_argument("--exclusive", action="store_true", help="with --on: refuse if another research run is live on the machine")
    ap.add_argument("--mem-gb", type=float, default=None, metavar="N",
                    help="with --on: admit only if the machine's memory cgroup has N GB free after live reservations (/workspace/ramlock), "
                         "and reserve them for the run; else MemoryNotAvailable (research.mem)")
    ap.add_argument("--mem-wait", type=float, default=0.0, metavar="S", help="with --mem-gb: wait up to S seconds for the memory instead of refusing")
    ap.add_argument("--id", default=None, help="use this run id instead of allocating one")
    ap.add_argument("--claimed", action="store_true", help=argparse.SUPPRESS)          # machine side: the dir was created by launch-request
    ap.add_argument("--cost-json", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--remote-json", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--exclusive-lock", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--expect", action="append", default=[], metavar="K=V", help="expected workload fingerprint entry (recorded separately from what the workload observes)")
    ap.add_argument("--require-result", action="store_true", help="a zero exit without a valid result.json exits 65")
    ap.add_argument("--stage", default="run", help="stage name recorded in the job (default: run)")
    ap.add_argument("--tool", default=None, metavar="NAME|module:NAME",
                    help="store Tool declaration (research.store): closure manifest, key params, derivation and reuse_key are computed before launch, "
                         "the attempt is published after")
    ap.add_argument("--scratch", action="append", default=[], metavar="NAME",
                    help="tool cache dir under <store>/scratch/<reuse_key>/NAME (cold per commit, on purpose); `triton` exports TRITON_CACHE_DIR, "
                         "any other NAME exports RESEARCH_SCRATCH_<NAME>")
    ap.add_argument("--campaign", default=os.environ.get("RESEARCH_CAMPAIGN"), help="campaign recorded in the attempt's conditions (default $RESEARCH_CAMPAIGN)")
    ap.add_argument("--source-identity", default=None, help=argparse.SUPPRESS)   # machine side: JSON {commit, tree, dirty} of the shipped source
    ns, harness_opts = ap.parse_known_args(opts)
    mem_reservation = os.environ.pop("RESEARCH_MEM_RESERVATION", None) if ns.claimed else None    # research.mem: this runner's, not the workload's
    if ns.mem_gb is not None and not ns.on:
        print("research run: --mem-gb admits a run on a shared machine (--on MACHINE); a local run has no reservation to take", file=sys.stderr)
        return 2
    if ns.project:
        harness_opts = ["--project", ns.project, *harness_opts]
    root = R.runs_root(ns.runs_dir)
    run_id = ns.id or R.new_run_id()
    run_dir = root / run_id
    if ns.on:
        return _run_remote(ns, run_id, run_dir, harness_opts, command)
    env_map = _parse_env(ns.env)
    if env_map is None:
        return 2
    if ns.claimed and not env_map:
        env_map = _request_env(ns.remote_json)
    if run_dir.exists() and ((run_dir / "job.json").exists() or not ns.claimed):
        print(f"research run: {run_dir} already exists; a rerun gets a fresh id", file=sys.stderr)
        return 2
    hp = _harness_parser("research run")
    try:
        a = hp.parse_args(["--attempt-dir", str(run_dir), "--stage", ns.stage, "--job", run_id, *harness_opts])
    except SystemExit:
        return 2
    if ns.claimed and not os.environ.get("RESEARCH_STORE"):
        os.environ["RESEARCH_STORE"] = str(root.parent / "store")     # machine side: the store lives beside the runs, <root>/store (what `data pull` reads)
    cwd_basis = _resolve_cwd(a)
    try:
        prep = _prepare_store(ns, a, command, run_id, run_dir, env_map)
    except (ImportError, LookupError, TypeError) as e:
        print(f"research run: --cwd {a.cwd} / --tool {ns.tool}: {e}", file=sys.stderr)
        return 2
    if a.cwd is None:
        # the workload runs in its own run dir, locally and on a machine (a relative `> result.json` used to land in the checkout, and
        # on a machine in <root>/src/<sha>/, which every run of that commit shares: concurrent runs raced on it)
        a.cwd = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=ns.claimed)
    cost = json.loads(ns.cost_json) if ns.cost_json else None
    remote_info = json.loads(ns.remote_json) if ns.remote_json else None
    if ns.claimed and remote_info is not None:
        try:
            remote_info["tool_running"] = remote.check_tool(remote_info.get("tool_identity"))
        except remote.ResearchVersionMismatch as e:
            from .telemetry.events import utc_now_ms
            from .telemetry.job import write_json_atomic

            write_json_atomic(run_dir / "refused.json", {"run_id": run_id, "reason": str(e), "fault": type(e).__name__, "at_utc": utc_now_ms()})
            print(f"research run: refused: {e}", file=sys.stderr)
            if ns.exclusive_lock:
                remote.release_exclusive(Path(ns.exclusive_lock).parent, run_id)
            if mem_reservation:
                from .mem import release

                release(mem_reservation, run_id)
            return 3
    job = record.extend_job(build_job_from(a, command), run_id=run_id, runs_root=root, expected_fingerprint=_parse_expect(ns.expect),
                            cost=cost, remote=remote_info, store=prep["section"], cwd={"path": a.cwd, "basis": cwd_basis})
    from .telemetry.job import write_json_atomic
    from .telemetry.run import run as harness_run

    write_json_atomic(run_dir / "job.json", job)
    os.environ.update(result.env_for(run_id, run_dir, source_sha=getattr(a, "tree_sha", None)))
    os.environ.update(prep["env"])
    # --env is the workload's: set for the harness (which spawns it with this environment), then restored so the store publish below
    # runs with the caller's environment (a workload-only AWS_* must not become the push credentials)
    apply_env = {k: v for k, v in env_map.items() if not (ns.claimed and k == "PYTHONPATH")}   # launch_request composed it: tool, tree roots, then this
    saved_env = {k: os.environ.get(k) for k in apply_env}
    os.environ.update(apply_env)
    print(f"research: run {run_id} -> {run_dir}" + (f"  cwd={a.cwd}" if a.cwd else ""), file=sys.stderr)
    try:
        rc = harness_run(a, command)
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if ns.exclusive_lock:
            remote.release_exclusive(Path(ns.exclusive_lock).parent, run_id)
        if mem_reservation:
            from .mem import release

            release(mem_reservation, run_id)
    _publish_attempt(run_dir, prep)
    s = R.summarize(run_dir)
    line = f"research: run {run_id} {s['state']} rc={s['rc']} class={s['class'] or '-'} result={s['result']}"
    if s["result"] == "absent":
        line += f" (looked for {run_dir / 'result.json'}: write $RESEARCH_RUN_DIR/result.json)"
    if s["validation"]:
        line += f" validation={s['validation']}"
    print(line + f"  (research inspect {run_id})", file=sys.stderr)
    if ns.require_result and rc == 0 and not s["result_valid"]:
        print(f"research: --require-result: the workload exited 0 but result.json is {s['result']}; exiting {EX_DATAERR}", file=sys.stderr)
        return EX_DATAERR
    return rc


CWD_SOURCE = "source"


def _resolve_cwd(a: argparse.Namespace) -> str:
    """The workload's cwd, resolved in place on `a.cwd`; returns its basis for the record.

    The SOURCE ROOT is this process's cwd: the invoking directory locally, the shipped tree <root>/src/<sha>/ on a machine (launch_request
    starts the runner there; the store's closure and tool declarations are read from it).  `--cwd source` names it, `--cwd source/SUB` or
    a relative `--cwd SUB` a directory under it (an explicit opt-in to running inside a tree other runs may share); an absolute `--cwd`
    is taken as given.  No `--cwd`: basis `run_dir`, set by the caller once the store preparation has read the source root."""
    if a.cwd is None:
        return "run_dir"
    raw = a.cwd
    if os.path.isabs(raw):
        return "explicit"
    if raw == CWD_SOURCE or raw.startswith(CWD_SOURCE + "/"):
        raw = raw[len(CWD_SOURCE):].lstrip("/")
    a.cwd = os.path.abspath(os.path.join(os.getcwd(), raw))
    return "source"


def _parse_env(items: list[str]) -> dict[str, str] | None:
    """--env K=V ... -> {K: V}; None (after a message) on a malformed item."""
    env: dict[str, str] = {}
    for kv in items:
        k, sep, v = kv.partition("=")
        if not sep or not k:
            print(f"research run: --env wants K=V, got {kv!r}", file=sys.stderr)
            return None
        env[k] = v
    return env


def _request_env(remote_json: str | None) -> dict[str, str]:
    """Machine side: the `env` launch-request applied to this process, read back from the request.json that --remote-json names, so the
    attempt records it as params.env like a local --env (the values never travel on a command line); {} when unreadable."""
    try:
        req = json.loads(Path(json.loads(remote_json or "{}")["request"]).read_text())
        return {str(k): str(v) for k, v in (req.get("env") or {}).items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {}


def _prepare_store(ns: argparse.Namespace, a: argparse.Namespace, command: list[str], run_id: str, run_dir: Path, env_map: dict[str, str] | None = None) -> dict:
    """Before launch: load the Tool (errors propagate: the user asked for it), collect conditions, resolve params and inputs to
    content identity, compute the closure manifest, derivation id and reuse_key, create --scratch dirs.  Everything after the
    Tool load is best effort: a failure is recorded in the `store` section, never a reason not to run.  `env_map` (--env) is
    recorded as `params.env`, values redacted where the NAME matches SECRET_KEY; it is not part of the derivation (a Tool that
    depends on an environment variable declares it as a key param)."""
    from . import env as E
    from .store import attempt as A
    from .store import derivation as D
    from .store.tool import content_identity
    from .telemetry.job import SECRET_KEY, redact_argv

    source_root = Path(a.cwd or os.getcwd())
    tool = None
    if ns.tool:
        from .store import tools_registry

        tool = tools_registry.load(ns.tool, source_root)
    argv_red, _ = redact_argv(command)
    env_out: dict[str, str] = {}
    if ns.campaign:
        env_out["RESEARCH_CAMPAIGN"] = ns.campaign
    conds = None
    try:
        hint = json.loads(ns.source_identity) if ns.source_identity else None
        conds = E.conditions(run_dir=run_dir, project=ns.project, campaign=ns.campaign, exclusive=bool(ns.exclusive_lock), source_root=source_root, source_hint=hint)
        if tool is not None:
            conds.update(tool.tool_conditions(conds))
            params, key = tool.resolve_params(command, source_root)
            params = {k: ("<redacted>" if SECRET_KEY.search(k) and isinstance(v, str) else v) for k, v in params.items()}
            params["argv"] = argv_red
            closure = tool.closure_manifest(source_root)
            ident = {"name": tool.name, "version": tool.version, "spec": tool.spec}
            key_params, key_conditions = list(tool.key_params), list(tool.key_conditions)
        else:
            # undeclared: the whole (redacted) command line is the request, and nothing is reusable (reuse_key None below):
            # argv cannot name the files, environment or data the command reads
            params, key, closure = {"argv": argv_red}, {"argv": argv_red}, {}
            ident = {"name": Path(command[0]).name, "version": None, "spec": None}
            key_params, key_conditions = ["argv"], []
        if env_map:
            params["env"] = {k: ("<redacted>" if SECRET_KEY.search(k) else v) for k, v in env_map.items()}
        inputs = {}
        for spec in a.input:                         # harness --input name=path: a fetched artifact is an input; anything else enters the derivation
            name, _, path = spec.partition("=") if "=" in spec else (Path(spec).name, "=", spec)
            if path.startswith("art:"):              # a store id names the input directly (no fetched copy needed on this machine)
                inputs[name] = path
                continue
            cid = content_identity(path, source_root)   # by content, as the key param `input.<name>` (inputs hold art: ids only)
            if isinstance(cid, str) and cid.startswith("art:"):
                inputs[name] = cid
            else:
                key[f"input.{name}"] = cid
        drv = D.derivation_id(ident, inputs, key, conds, key_conditions)
        rk = D.reuse_key(drv, conds) if tool is not None else None
        sroot = A.store_root()
        scratch = {}
        for name in ns.scratch:
            p = D.scratch_dir(sroot, rk, run_id, name)
            scratch[name] = str(p)
            env_out.update(D.scratch_env(name, p))
        section = {"tool": ident, "derivation": drv, "reuse_key": rk, "key_params": key_params, "key_conditions": key_conditions, "params": params,
                   "key_params_resolved": key, "inputs": inputs, "closure": closure, "conditions": conds, "scratch": scratch, "campaign": ns.campaign,
                   "store_root": str(sroot)}
    except Exception as e:  # noqa: BLE001
        section = {"error": f"{type(e).__name__}: {e}", "tool": {"name": tool.name, "version": tool.version, "spec": tool.spec} if tool else None,
                   "campaign": ns.campaign}
        print(f"research run: store preparation failed ({section['error']}); running anyway, the attempt will be published from the records", file=sys.stderr)
    return {"tool": tool, "conditions": conds, "section": section, "env": env_out}


def _publish_attempt(run_dir: Path, prep: dict) -> None:
    from . import env as E
    from .store import attempt as A

    conds = prep.get("conditions")
    if conds is not None:
        conds["exec.load_end"] = E.load_now()
    out = A.publish(run_dir, tool=prep.get("tool"), conditions=conds)
    if out.get("published"):
        if out.get("already"):
            pres = "  (already in the store)"
        elif out.get("preserved"):
            pres = "  PRESERVED on the remote"
        elif out.get("preserve_error"):
            pres = f"  NOT preserved: {out['preserve_error']}  (`research data push {run_dir.name}` retries)"
        else:
            pres = f"  local only ({out.get('preserve_skipped', 'no remote')}; `research data push {run_dir.name}` with credentials preserves it)"
        print(f"research: attempt published: {out['n_outputs']} outputs drv={out['derivation']} reuse={out['reuse_key'] or '-'}{pres}", file=sys.stderr)
    else:
        print(f"research: attempt not published: {out.get('error')}  (store_publish_error.json in the run dir; `research data attempt publish` retries)",
              file=sys.stderr)


def _source_identity_hint(git_dir: Path) -> dict | None:
    from . import env as E

    src = E.source_conditions(git_dir)
    if not src.get("source.commit"):
        return None
    return {"commit": src["source.commit"], "tree": src["source.tree"], "dirty": src["source.dirty"], "dirty_digest": src["source.dirty_digest"]}


def _run_remote(ns: argparse.Namespace, run_id: str, run_dir: Path, harness_opts: list[str], command: list[str]) -> int:
    if not ns.project:
        print("research run --on: --project is required (the machine is bound to a project)", file=sys.stderr)
        return 2
    if run_dir.exists():
        print(f"research run: {run_dir} already exists; a rerun gets a fresh id", file=sys.stderr)
        return 2
    env = {}
    for kv in ns.env:
        k, sep, v = kv.partition("=")
        if not sep or not k:
            print(f"research run: --env wants K=V, got {kv!r}", file=sys.stderr)
            return 2
        env[k] = v
    asked_cwd = remote.flag_value(harness_opts, "--cwd")
    if asked_cwd is not None and not os.path.isabs(asked_cwd) and not ns.source:
        print(f"research run --on {ns.on}: --cwd {asked_cwd} is relative to the shipped source tree, and there is none; pass --source GITDIR "
              f"or an absolute --cwd (the default is the run dir)", file=sys.stderr)
        return 2
    harness_args = ["--stage", ns.stage, *harness_opts]
    for e in ns.expect:
        harness_args += ["--expect", e]
    if ns.require_result:
        harness_args.append("--require-result")
    # the store flags travel as `research run` flags: the machine-side runner computes derivation/reuse_key in the shipped source tree
    if ns.tool:
        # the machine runs the shipped tree's research package when --source carries one, else THIS launcher's: that one must know the tool
        from .store import tools_registry
        if remote.source_tool_package(Path(ns.source).expanduser() if ns.source else None) is None and ":" not in ns.tool \
                and ns.tool not in tools_registry.REGISTRY:
            print(f"research run --on {ns.on}: launcher research package does not know tool {ns.tool!r}; pass --source with tools/research "
                  f"(registered here: {', '.join(sorted(tools_registry.REGISTRY))})", file=sys.stderr)
            return 3
        harness_args += ["--tool", ns.tool]
    for s in ns.scratch:
        harness_args += ["--scratch", s]
    if ns.campaign:
        harness_args += ["--campaign", ns.campaign]
    if ns.source:
        hint = _source_identity_hint(Path(ns.source).expanduser())   # a shipped `git archive` has no .git: the machine gets what git knew here
        if hint:
            harness_args += ["--source-identity", json.dumps(hint, sort_keys=True)]
    try:
        rec = remote.launch(run_id=run_id, local_dir=run_dir, machine_name=ns.on, project=ns.project, command=command, harness_args=harness_args,
                            source=Path(ns.source).expanduser() if ns.source else None, sends=[Path(s).expanduser() for s in ns.send],
                            cwd=None, env=env, exclusive=ns.exclusive, mem_gb=ns.mem_gb, mem_wait_s=ns.mem_wait)
    except remote.Refused as e:
        print(f"research run --on {ns.on}: refused: {e}", file=sys.stderr)
        return 3
    launch = rec.get("launch") or {}
    if rec["phase"] == "refused":
        print(f"research: run {run_id} refused on {ns.on} ({launch.get('fault') or 'Refused'}): {launch.get('refused')}  (recorded in {run_dir})", file=sys.stderr)
        return 3
    print(f"research: run {run_id} launched on {ns.on} ({rec['pod_id']}) pid={launch.get('runner_pid')} -> {rec['remote_run_dir']}", file=sys.stderr)
    print(f"research: observe with `research fetch {run_id}`; the workload does not depend on this session staying alive", file=sys.stderr)
    return 0


def cmd_fetch(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research fetch")
    ap.add_argument("ref")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--all", action="store_true", help="every file, verified by sha256; writes preserved.json last")
    ns = ap.parse_args(argv)
    root = R.runs_root(ns.runs_dir)
    try:
        run_dir = R.resolve(root, ns.ref)
        rec = remote.fetch(run_dir, everything=ns.all)
    except (FileNotFoundError, LookupError, remote.Refused) as e:
        print(f"research fetch: {e}", file=sys.stderr)
        return 1
    if rec.get("last_fetch_error"):
        lo = rec.get("last_observed") or {}
        print(f"research fetch: machine unreachable ({rec['last_fetch_error']['stderr'].strip() or 'rc ' + str(rec['last_fetch_error']['rc'])}); "
              f"last observed {lo.get('state')} at {lo.get('at_utc')}; current status unknown", file=sys.stderr)
        return 4
    s = R.summarize(run_dir)
    print(f"research: {s['run_id']} observed {rec['observed_utc']}: {s['state']} rc={s['rc']} class={s['class'] or '-'} result={s['result']}"
          + (f"  preserved={'yes' if rec.get('preserved') else 'NO: ' + json.dumps(rec.get('preserve_error'))}" if ns.all else ""))
    return 0


def cmd_compare(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research compare")
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    root = R.runs_root(ns.runs_dir)
    try:
        b, c = R.resolve(root, ns.baseline), R.resolve(root, ns.candidate)
    except (FileNotFoundError, LookupError) as e:
        print(f"research compare: {e}", file=sys.stderr)
        return 1
    rep = C.compare(b, c)
    print(json.dumps(rep, indent=2, sort_keys=True, default=str) if ns.json else C.render(rep))
    return 0


def cmd_env(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research env", description="the PYTHONPATH a source tree's workspace declares (research.pythonpath)")
    ap.add_argument("--source", required=True, metavar="TREE", help="a checkout or shipped tree with a root pyproject.toml [tool.uv.workspace]")
    ap.add_argument("--rev", default=None, help="read this git revision of TREE instead of its files")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    from .pythonpath import import_roots_dir, import_roots_git, pythonpath

    tree = Path(ns.source).expanduser().resolve()
    try:
        derived = import_roots_git(tree, ns.rev) if ns.rev else import_roots_dir(tree)
    except LookupError as e:
        print(f"research env: {e}", file=sys.stderr)
        return 1
    if ns.json:
        print(json.dumps({**derived, "tree": str(tree), "pythonpath": pythonpath(str(tree), derived)}, indent=2))
    else:
        print(f"export PYTHONPATH={pythonpath(str(tree), derived)}")
    for m in derived["missing"]:
        print(f"research env: declared but absent: {m}", file=sys.stderr)
    if not derived["roots"]:
        print(f"research env: {tree}: no import roots ({derived['source']})", file=sys.stderr)
        return 1
    return 0


def cmd_launch_request(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research launch-request")
    ap.add_argument("--expect-tool", default=None, metavar="IDENTITY", help="the launcher's remote.tool_identity of the package it shipped")
    ap.add_argument("request")
    ns = ap.parse_args(argv)
    try:
        remote.check_tool(ns.expect_tool)                 # before the claim: a mismatched package must not interpret the request
        out = remote.launch_request(Path(ns.request))
    except remote.Refused as e:
        print(json.dumps({"claimed": False, "refused": str(e), "fault": type(e).__name__}))
        return 3
    print(json.dumps(out))
    return 0


def _fmt(s: dict) -> str:
    cmd = " ".join(s["command"]).replace("\n", " ") if s.get("command") else "?"
    if len(cmd) > 50:
        cmd = cmd[:47] + "..."
    rc = "" if s["rc"] is None else str(s["rc"])
    res = "-" if not s["result_present"] else ("ok" if s["result_valid"] else "BAD")
    val = s["validation"] or "-"
    where = (s.get("remote") or {}).get("machine") or "local"
    return f"{s['id']:<24} {str(s['state']):<10} {rc:>3}  {(s['class'] or ''):<22} {res:<4} {val:<8} {str(s['project']):<10} {where:<11} {str(s['stage']):<16} {cmd}"


def cmd_status(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research status")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--all", action="store_true", help="every run, not just the last 20")
    ap.add_argument("--refresh", action="store_true", help="fetch the records of every non-terminal remote run first")
    ap.add_argument("--balance", action="store_true", help="show the provider's account ledger (balance, spend limit, current $/h) with its observation time")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    root = R.runs_root(ns.runs_dir)
    ledger = None
    if ns.balance:
        try:
            from .pods import runpod
            from .telemetry.events import utc_now_ms

            data = runpod._graphql("{ myself { clientBalance spendLimit currentSpendPerHr } }")["data"]["myself"]
            ledger = {"provider": "runpod", "observed_utc": utc_now_ms(), **data}
        except Exception as e:  # noqa: BLE001
            ledger = {"provider": "runpod", "error": f"{type(e).__name__}: {e}"}
    if ns.refresh:
        for s in R.list_runs(root):
            if s.get("remote_rec") and s["state"] not in R.TERMINAL and (s["remote_rec"].get("remote_run_dir")):
                try:
                    remote.fetch(Path(s["dir"]))
                except remote.Refused as e:
                    print(f"research status: {s['id']}: {e}", file=sys.stderr)
    rows = R.list_runs(root)
    if not ns.all:
        rows = rows[-20:]
    if ns.json:
        print(json.dumps({"runs_root": str(root), "runs": rows, "ledger": ledger}, indent=2, sort_keys=True))
        return 0
    if ledger:
        if ledger.get("error"):
            print(f"ledger    unavailable: {ledger['error']}")
        else:
            print(f"ledger    runpod balance ${ledger['clientBalance']:.2f}  spend limit ${ledger['spendLimit']}  now ${ledger['currentSpendPerHr']}/h  (observed {ledger['observed_utc']}; the provider's numbers, not a research counter)")
        print()
    if not rows:
        print(f"no runs under {root}")
        return 0
    print(f"{'run':<24} {'state':<10} {'rc':>3}  {'class':<22} {'res':<4} {'valid':<8} {'project':<10} {'where':<11} {'stage':<16} command")
    for s in rows:
        print(_fmt(s))
    active = sum(1 for s in rows if s["state"] in R.ACTIVE)
    unfetched = sum(1 for s in rows if s.get("remote_rec") and not s["has_job"] and s["state"] != "refused")
    print(f"\n{len(rows)} shown ({active} active" + (f", {unfetched} remote not yet fetched: `research status --refresh`" if unfetched else "") + f") under {root}"
          f"   res: result.json ok/BAD/-absent   valid: the workload's own checker")
    return 0


def _tool_line(tool: dict | None) -> str:
    if tool is None:
        return "unknown (v0.1 record: the harness alone wrote it)"
    sha = tool.get("research_git_sha")
    state = "" if tool.get("research_dirty") is None else (" (dirty)" if tool.get("research_dirty") else " (clean)")
    ident = f"git {sha[:12]}{state}" if sha else (f"snapshot {tool['research_snapshot']}" if tool.get("research_snapshot") else "no git sha, no snapshot")
    return f"research {tool.get('research_version')} @ {ident}; classifier {tool.get('classifier_version')}"


def cmd_inspect(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research inspect")
    ap.add_argument("ref", help="run id, unique prefix, or directory")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    root = R.runs_root(ns.runs_dir)
    try:
        run_dir = R.resolve(root, ns.ref)
    except (FileNotFoundError, LookupError) as exc:
        print(f"research inspect: {exc}", file=sys.stderr)
        return 1
    s = R.summarize(run_dir)
    if ns.json:
        print(json.dumps(s, indent=2, sort_keys=True))
        return 0
    print(f"run       {s['run_id']}   ({s['schema'] or 'no job.json'})")
    print(f"dir       {s['dir']}")
    if not s["readable"]:
        print(f"WARNING   job.json schema {s['schema']!r} is not one this research can read; fields below are best effort")
    print(f"exited    {s['state']}  rc={s['rc']} signal={s['signal']} wall_s={s['wall_s']}")
    if s["class"]:
        print(f"class     {s['class']} ({s['certainty']})" + (f": {s['why']}" if s["why"] else ""))
    print(f"result    {s['result']}")
    print(f"validated {s['validation'] or 'no validation reported'}  (the workload's checker; research does not verify proofs)")
    pres = s["preserved"]
    print(f"preserved {'yes: ' + str(pres.get('destination')) if pres else 'no preservation manifest'}")
    print(f"project   {s['project']}  lane={s['lane']}  stage={s['stage']}")
    print(f"source    tree_sha={s['tree_sha']}  workload_digest={s['workload_digest']}")
    if s["fingerprint_expected"] is not None or s["fingerprint_observed"] is not None:
        print(f"expected  {json.dumps(s['fingerprint_expected'], sort_keys=True)}")
        print(f"observed  {json.dumps(s['fingerprint_observed'], sort_keys=True)}")
    if s["measurements"]:
        print("measured  " + "; ".join(f"{m['name']}={m['value']} {m['unit']}" for m in s["measurements"][:12]))
    tool = s["tool"]
    print(f"tool      {_tool_line(s['tool'])}")
    cost = s["cost"]
    print(f"cost      {('$' + str(cost['usd']) + ' est. (' + cost['basis'] + ')') if cost.get('usd') is not None else 'unknown: ' + str(cost.get('reason'))}")
    print(f"time      started={s['started_utc']}  ended={s['ended_utc']}")
    print(f"command   {' '.join(s['command']) if s['command'] else '?'}")
    if s["remote"]:
        r = s["remote"]
        print(f"remote    {r.get('machine')} ({r.get('pod_id')}) {r.get('root')}  tool={str(r.get('tool')).rsplit('/', 1)[-1]}  source={r.get('source_sha')}"
              + (f"  archive={str(r.get('source_archive_sha256'))[:16]} ({r.get('source_verified')})" if r.get("source_archive_sha256") else ""))
    if s.get("remote_rec"):
        rr = s["remote_rec"]
        lo = rr.get("last_observed") or {}
        print(f"observed  phase={rr.get('phase')} at={rr.get('observed_utc')} state={lo.get('state')}"
              + (f"  ERROR={rr.get('error')}" if rr.get("error") else "")
              + ("  (records shown are a cached copy; `research fetch` refreshes)" if rr.get("observed_utc") else "  (nothing fetched yet)"))
    files = sorted(p.name for p in Path(s["dir"]).iterdir() if p.is_file())
    print(f"files     {', '.join(files)}")
    missing = [n for n, ok in (("job.json", s["has_job"]), ("status.json", s["has_status"])) if not ok]
    if missing:
        print(f"missing   {', '.join(missing)}  (the run died before the harness wrote them)")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0
    if argv[0] == "--version":
        from . import __version__

        print(__version__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "run":
        return cmd_run(rest)
    if cmd == "status":
        return cmd_status(rest)
    if cmd == "inspect":
        return cmd_inspect(rest)
    if cmd == "fetch":
        return cmd_fetch(rest)
    if cmd == "compare":
        return cmd_compare(rest)
    if cmd == "launch-request":
        return cmd_launch_request(rest)
    if cmd == "env":
        return cmd_env(rest)
    if cmd == "mem":
        from .mem import main as mem_main

        return mem_main(rest)
    if cmd == "tele":
        from .telemetry.cli import main as tele_main

        return tele_main(rest)
    if cmd == "data":
        from .store.cli import main as data_main

        return data_main(rest)
    if cmd == "pods":
        from .pods.part import main as pods_main

        return pods_main(rest)
    print(f"research: unknown command {cmd!r}\n\n{__doc__.strip()}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
