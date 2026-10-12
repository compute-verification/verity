"""The pod registry every machine can update: one file per machine under `machines.d/` (cloud migration step g).

    research pods register NAME --pod-id ID --project P [--root DIR] [--python PY] [--guard N|true] [--replace] [--any-name]
    research pods register NAME --host ADDR --project P [--port N] [--user U] [--rate USD_PER_HOUR] [--root DIR] [--replace]
    research pods unregister NAME
    research pods registry [--json]                          # every machine `research run --on` accepts, and where it comes from
    research pods registry import [--apply] [--all]          # seed machines.d from machines.toml (running pods only unless --all)
    research pods create <runpod flags> --register --project P [--guard N]   # create, wait for sshd, register under --name

`~/.research/machines.toml` is one hand-edited file on the laptop; every lane that creates a pod had to merge an entry into it.
The registry is a directory with one TOML file per machine, `<dir>/<name>.toml`, holding that machine's table (`provider`,
`pod_id`, `project` or `projects`, and optionally `root`, `python`, `guard`, `ramlock`) plus who registered it and when.  Two
agents registering two pods write two files, so nothing is ever hand-merged; in the notes repo, `research notes sync` carries
them to every machine.  `<dir>/_families.toml` carries the `[families.<fam>]` VRAM references `pods check-part` reads.

Where it is read and written (`registry_dir`, the one directory `register`, `unregister`, `registry import` and `research run --on`
use): `$RESEARCH_MACHINES_D` when set, else `<notes>/machines.d`, whether or not machines.toml exists (with RESEARCH_NOTES_TOKEN
set, a missing notes clone is cloned first: `notes_sync.ensure_clone`).  `RESEARCH_MACHINES_D=none` turns the registry off: run
reads machines.toml alone, and the commands that write refuse, since nothing would read what they wrote.  With
`$RESEARCH_MACHINES_D` pointing elsewhere, `research run --on` still takes a machine only the notes clone's machines.d lists:
a hand-copied override never hides a machine registered since it was copied (vy-nebius-cpu-1 from the lander's VM, 8 Oct 03:43Z).
Both sources merge (`merged`); a name both define with different pod ids is dropped from the result, so it is refused
as unlisted and the refusal says why, never silently resolved to either pod.  A file that does not parse, names an invalid
machine, or lacks provider / pod_id (host for an ssh machine) / project is dropped the same way.

`register` refuses before writing anything when: the name is not a plain name; the pod id is unknown to the RunPod account (a
404, or any API failure: G2); the pod's name (campaign suffix stripped) is not NAME (`--any-name` allows it: a pod created before
the convention); machines.toml or the registry already binds NAME to another pod (`--replace` for the registry file only).  The
`register --host` (an ssh machine, such as a cloud VM with a static address) refuses unless `hostname` answers over ssh with the
research key.  The file is written atomically and synced to the notes remote at once, so other agents' `research run --on` sees the machine
(vy-nebius-cpu-1 was invisible to lanes for hours, 7 Oct); `RESEARCH_NOTES_SYNC=0` leaves the commit to the next `notes sync`.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import socket
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Callable

from . import strip_pod_suffix

REGISTRY_ENV = "RESEARCH_MACHINES_D"
FAMILIES_FILE = "_families.toml"
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
POD_ID_RE = re.compile(r"[a-z0-9]{8,32}")
KNOWN_KEYS = ("provider", "pod_id", "host", "port", "user", "rate_usd_per_hour", "project", "projects", "root", "python", "guard", "ramlock")
HOST_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.:-]{0,252}")
OFF = ("", "none", "off", "0")

problems: dict[str, str] = {}   # name -> why the last merge dropped it (read by remote.resolve_machine's refusal)


def notes_root(environ: dict | None = None) -> Path:
    env = os.environ if environ is None else environ
    return Path(env.get("RESEARCH_NOTES") or Path.home() / ".research" / "notes").expanduser()


def registry_dir(machines_toml: Path, environ: dict | None = None) -> Path | None:
    """The registry directory, or None when it is off (module docstring)."""
    env = os.environ if environ is None else environ
    if REGISTRY_ENV in env:
        v = env[REGISTRY_ENV].strip()
        return None if v.lower() in OFF else Path(v).expanduser()
    return notes_root(env) / "machines.d"


def write_dir(machines_toml: Path, environ: dict | None = None) -> Path:
    """Where `register`, `unregister` and `import` write: the directory `research run --on` reads.  Refused when the registry is
    off, since nothing would read the file."""
    from .. import remote

    d = registry_dir(machines_toml, environ)
    if d is None:
        raise remote.Refused(f"the registry is off (${REGISTRY_ENV}={(os.environ if environ is None else environ)[REGISTRY_ENV]!r}), "
                             f"so `research run --on` would not read a machines.d file; unset it or point it at a directory")
    return d


def validate(name: str, entry: Any) -> str | None:
    """Why `entry` cannot be machine `name`, or None."""
    if not NAME_RE.fullmatch(name) or name.startswith("_"):
        return f"{name!r} is not a machine name ({NAME_RE.pattern}, not starting with _)"
    if not isinstance(entry, dict):
        return "not a table"
    if entry.get("provider") == "ssh":
        if not isinstance(entry.get("host"), str) or not HOST_RE.fullmatch(entry["host"]):
            return f"host {entry.get('host')!r} is not a host name or address"
        if "port" in entry and not (isinstance(entry["port"], int) and 0 < entry["port"] < 65536):
            return f"port {entry['port']!r} is not a TCP port"
        if "user" in entry and not (isinstance(entry["user"], str) and NAME_RE.fullmatch(entry["user"])):
            return f"user {entry['user']!r} is not a user name"
    elif entry.get("provider") != "runpod":
        return f"provider {entry.get('provider')!r} (runpod or ssh)"
    elif not isinstance(entry.get("pod_id"), str) or not POD_ID_RE.fullmatch(entry["pod_id"]):
        return f"pod_id {entry.get('pod_id')!r} is not a pod id"
    projects = entry.get("projects") or ([entry["project"]] if entry.get("project") else [])
    if not projects or not all(isinstance(p, str) and p for p in projects):
        return "no project (project = \"...\" or projects = [...])"
    return None


def binding(entry: dict) -> str | None:
    """What a name is bound to: a runpod pod id, or an ssh machine's host."""
    return entry.get("pod_id") or entry.get("host")


def _bound(entry: dict) -> str:
    return f"pod {entry['pod_id']}" if entry.get("pod_id") else f"host {entry.get('host')}"


def load_dir(d: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """(name -> entry, name -> why dropped) for every `<name>.toml` in the directory."""
    entries: dict[str, dict] = {}
    bad: dict[str, str] = {}
    if not d.is_dir():
        return entries, bad
    for p in sorted(d.glob("*.toml")):
        if p.name.startswith("_"):
            continue
        try:
            doc = tomllib.loads(p.read_text())
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
            bad[p.stem] = f"{p.name} does not parse ({type(e).__name__})"
            continue
        if why := validate(p.stem, doc):
            bad[p.stem] = f"{p.name}: {why}"
            continue
        entries[p.stem] = doc
    return entries, bad


def merged(toml_machines: dict[str, dict], machines_toml: Path, environ: dict | None = None) -> dict[str, dict]:
    """machines.toml's machines plus the registry's; a name the two bind to different pods, or a broken registry file, is dropped
    (reason in `problems`).  Unchanged (and `problems` empty) when no registry directory is read.  Under a `$RESEARCH_MACHINES_D`
    override, a name neither it nor machines.toml defines comes from the notes clone's machines.d, if there is one."""
    problems.clear()
    d = registry_dir(machines_toml, environ)
    if d is None:
        return toml_machines
    if REGISTRY_ENV not in (os.environ if environ is None else environ):
        from ..notes_sync import ensure_clone  # noqa: PLC0415

        if line := ensure_clone(notes_root(environ)):
            print(line, file=sys.stderr)
    entries, bad = load_dir(d)
    out = dict(toml_machines)
    for name, why in bad.items():
        if name not in toml_machines:
            problems[name] = why
    for name, e in entries.items():
        t = toml_machines.get(name)
        if t is not None and binding(t) != binding(e):
            out.pop(name, None)
            problems[name] = (f"machines.toml binds it to {_bound(t)} and {d / (name + '.toml')} to {_bound(e)}; "
                              "refused until one of them is removed")
        elif t is None:
            out[name] = e
    if (notes := notes_root(environ) / "machines.d") != d and notes.is_dir():
        for name, e in load_dir(notes)[0].items():
            if name not in out and name not in problems:
                out[name] = e
    return out


def families(machines_toml: Path, environ: dict | None = None) -> dict[str, dict]:
    """`[families.*]` from machines.toml, then the registry's `_families.toml` for families machines.toml does not define."""
    out: dict[str, dict] = {}
    if machines_toml.exists():
        out.update(tomllib.loads(machines_toml.read_text()).get("families", {}))
    d = registry_dir(machines_toml, environ)
    if d is not None and (d / FAMILIES_FILE).is_file():
        try:
            extra = tomllib.loads((d / FAMILIES_FILE).read_text()).get("families", {})
        except (OSError, tomllib.TOMLDecodeError):
            extra = {}
        for k, v in extra.items():
            out.setdefault(k, v)
    return out


# ---- writing ------------------------------------------------------------------------------------------------------------------------------

def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    return json.dumps(str(v), ensure_ascii=True)       # a JSON string with ASCII escapes is a valid TOML basic string


def render_entry(name: str, entry: dict) -> str:
    head = f"# machine {name}: written by `research pods register`; one file per machine (research.pods.registry)\n"
    return head + "".join(f"{k} = {_toml_value(v)}\n" for k, v in entry.items() if v is not None)


def write_entry(d: Path, name: str, entry: dict) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.toml"
    tmp = d / f".{name}.toml.tmp-{os.getpid()}"
    tmp.write_text(render_entry(name, entry))
    os.replace(tmp, p)
    return p


def _pod(pod_id: str) -> dict:
    from . import runpod

    return runpod.get(pod_id)


def _guard(v: str | None) -> bool | float | None:
    if v is None:
        return None
    if v.lower() in ("true", "yes"):
        return True
    try:
        n = float(v)
    except ValueError:
        raise ValueError(f"--guard wants idle minutes or true, not {v!r}") from None
    if n <= 0:
        raise ValueError(f"--guard wants a positive number of idle minutes, not {v!r}")
    return int(n) if n == int(n) else n


def register(name: str, pod_id: str, project: str, *, root: str | None = None, python: str | None = None, guard: str | None = None,
             replace: bool = False, any_name: bool = False, machines_toml: Path | None = None, environ: dict | None = None,
             get_pod: Callable[[str], dict] = _pod, now: dt.datetime | None = None) -> tuple[Path, str]:
    """Check, then write `<dir>/<name>.toml` (module docstring).  -> (path, 'registered' | 'updated' | 'unchanged').
    Raises remote.Refused with the reason; nothing is written then."""
    from .. import remote
    from . import runpod

    mt = machines_toml or remote.machines_path()
    entry: dict[str, Any] = {"provider": "runpod", "pod_id": pod_id, "project": project, "root": root, "python": python,
                             "guard": _guard(guard)}
    if why := validate(name, entry):
        raise remote.Refused(f"register {name}: {why}")
    try:
        pod = get_pod(pod_id)
    except runpod.PodError as e:
        raise remote.Refused(f"register {name}: pod {pod_id} is not in the RunPod account ({e}); only existing pods may be registered") from None
    except Exception as e:  # noqa: BLE001  (no API key, no network: cannot vouch for the pod)
        raise remote.Refused(f"register {name}: cannot look pod {pod_id} up ({type(e).__name__}: {str(e)[:120]})") from None
    if not pod or pod.get("id", pod_id) != pod_id:
        raise remote.Refused(f"register {name}: pod {pod_id} is not in the RunPod account")
    pod_name = strip_pod_suffix(str(pod.get("name") or ""))
    if pod_name != name and not any_name:
        raise remote.Refused(f"register {name}: pod {pod_id} is named {pod.get('name')!r}, not {name!r} (plus a campaign suffix); "
                             "register it under its own name, or --any-name")
    return _write_checked(name, entry, mt, replace=replace, environ=environ, now=now)


def _write_checked(name: str, entry: dict[str, Any], mt: Path, *, replace: bool, environ: dict | None, now: dt.datetime | None) -> tuple[Path, str]:
    from .. import remote

    toml_machines = tomllib.loads(mt.read_text()).get("machines", {}) if mt.exists() else {}
    if (t := toml_machines.get(name)) is not None and binding(t) != binding(entry):
        raise remote.Refused(f"register {name}: {mt} binds {name} to {_bound(t)}; pick another name")
    d = write_dir(mt, environ)
    p = d / f"{name}.toml"
    old = load_dir(d)[0].get(name) if p.exists() else None
    if old is not None and binding(old) != binding(entry) and not replace:
        raise remote.Refused(f"register {name}: {p} binds it to {_bound(old)}; --replace rebinds the name")
    same = old is not None and all(old.get(k) == entry.get(k) for k in KNOWN_KEYS)
    if same:
        return p, "unchanged"
    now = now or dt.datetime.now(dt.timezone.utc)
    entry["registered_by"] = os.environ.get("RESEARCH_AGENT") or socket.gethostname()
    entry["registered_at"] = f"{now:%Y-%m-%dT%H:%MZ}"
    write_entry(d, name, entry)
    return p, "updated" if old is not None else "registered"


def _probe_ssh(entry: dict[str, Any]) -> str | None:
    """Why `entry`'s host cannot be reached over the research key (`hostname` over ssh), or None."""
    from .. import remote

    tgt = remote.ssh_host_target(entry)
    try:
        r = remote.ssh_run(remote.ssh_argv(tgt["host"], tgt["port"], user=tgt["user"]), ["hostname"], timeout=60)
    except subprocess.TimeoutExpired:
        return "ssh timed out"
    return None if r.returncode == 0 else f"ssh rc {r.returncode}: {r.stderr.decode(errors='replace').strip()[-200:]}"


def register_ssh(name: str, host: str, project: str, *, port: int | None = None, user: str | None = None, rate: float | None = None,
                 root: str | None = None, python: str | None = None, replace: bool = False, machines_toml: Path | None = None,
                 environ: dict | None = None, probe: Callable[[dict], str | None] = _probe_ssh,
                 now: dt.datetime | None = None) -> tuple[Path, str]:
    """`register` for a `provider = "ssh"` machine: refused unless `hostname` answers over ssh with the research key."""
    from .. import remote

    mt = machines_toml or remote.machines_path()
    entry: dict[str, Any] = {"provider": "ssh", "host": host, "port": port, "user": user, "rate_usd_per_hour": rate, "project": project,
                             "root": root, "python": python}
    entry = {k: v for k, v in entry.items() if v is not None}
    if why := validate(name, entry):
        raise remote.Refused(f"register {name}: {why}")
    if why := probe({**entry, "name": name}):
        raise remote.Refused(f"register {name}: {host} does not answer over ssh ({why}); only reachable machines may be registered")
    return _write_checked(name, entry, mt, replace=replace, environ=environ, now=now)


def unregister(name: str, machines_toml: Path | None = None, environ: dict | None = None) -> Path:
    from .. import remote

    mt = machines_toml or remote.machines_path()
    d = write_dir(mt, environ)
    p = d / f"{name}.toml"
    if not NAME_RE.fullmatch(name) or not p.is_file():
        raise remote.Refused(f"unregister {name}: no {p}")
    p.unlink()
    return p


def plan_import(machines_toml: Path, d: Path, running: set[str] | None) -> list[dict]:
    """machines.toml entries to copy into the registry: every valid one not there yet whose pod is running (`running` = pod ids;
    None = all).  Each row: name, action (`write` | `skip`), why."""
    rows = []
    toml_machines = tomllib.loads(machines_toml.read_text()).get("machines", {}) if machines_toml.exists() else {}
    have = load_dir(d)[0]
    for name, e in sorted(toml_machines.items()):
        why = validate(name, e)
        if why is None and name in have:
            why = "already registered" if have[name].get("pod_id") == e.get("pod_id") else f"registry binds it to {have[name].get('pod_id')}"
        if why is None and running is not None and e.get("pod_id") not in running:
            why = "pod not running"
        rows.append({"name": name, "pod_id": e.get("pod_id") if isinstance(e, dict) else None, "action": "skip" if why else "write",
                     "why": why or "", "entry": e})
    return rows


def _sync(paths: list[Path]) -> str | None:
    from .. import notes_sync

    if os.environ.get(notes_sync.SYNC_ENV, "").strip() in ("0", "false", "no", "off"):
        return None
    root = notes_root()
    rel = [str(p.relative_to(root)) for p in paths if root in p.parents]
    if not rel:
        return None
    return notes_sync.render_sync(notes_sync.sync(root, paths=rel, message=f"machines.d: {', '.join(Path(r).stem for r in rel)}"))


def rows(machines_toml: Path, environ: dict | None = None) -> list[dict]:
    toml_machines = tomllib.loads(machines_toml.read_text()).get("machines", {}) if machines_toml.exists() else {}
    all_ = merged(toml_machines, machines_toml, environ)
    d = registry_dir(machines_toml, environ)
    reg = load_dir(d)[0] if d else {}
    out = []
    for name in sorted(set(all_) | set(problems)):
        e = all_.get(name) or {}
        src = "refused" if name in problems else "both" if name in toml_machines and name in reg else \
            "machines.toml" if name in toml_machines else "machines.d"
        out.append({"name": name, "source": src, "pod_id": binding(e), "projects": e.get("projects") or [e.get("project")],
                    "guard": e.get("guard"), "problem": problems.get(name)})
    return out


def main(argv: list[str]) -> int:
    import argparse

    from .. import remote

    ap = argparse.ArgumentParser(prog="research pods", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rg = sub.add_parser("register", help="bind a machine name to an existing pod: one file in the registry, no shared file to merge")
    rg.add_argument("name")
    who = rg.add_mutually_exclusive_group(required=True)
    who.add_argument("--pod-id", help="a RunPod pod")
    who.add_argument("--host", help="an ssh machine (provider = \"ssh\"): its address; --port, --user, --rate go with it")
    rg.add_argument("--port", type=int, default=None, help="--host: ssh port (default 22)")
    rg.add_argument("--user", default=None, help="--host: ssh user (default root)")
    rg.add_argument("--rate", type=float, default=None, metavar="USD_PER_HOUR", help="--host: the machine's hourly rate, for the attempts' cost")
    rg.add_argument("--project", required=True)
    rg.add_argument("--root", default=None, help="the machine's research root (default /workspace/research)")
    rg.add_argument("--python", default=None, help="default python3")
    rg.add_argument("--guard", default=None, metavar="N|true", help="opt the pod into the idle guard (minutes; true = 90)")
    rg.add_argument("--replace", action="store_true", help="rebind a name the registry already binds to another pod")
    rg.add_argument("--any-name", action="store_true", help="allow a pod whose RunPod name is not NAME")
    ur = sub.add_parser("unregister", help="remove a machine's registry file (git history keeps it)")
    ur.add_argument("name")
    ls = sub.add_parser("registry", help="every machine run --on accepts, with its source; `import` seeds machines.d from machines.toml")
    ls.add_argument("action", nargs="?", choices=["list", "import"], default="list")
    ls.add_argument("--apply", action="store_true", help="import: write the files (default: show the plan)")
    ls.add_argument("--all", action="store_true", help="import: also entries whose pod is not running")
    ls.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    mt = remote.machines_path()
    try:
        if ns.cmd == "register" and ns.host:
            p, what = register_ssh(ns.name, ns.host, ns.project, port=ns.port, user=ns.user, rate=ns.rate, root=ns.root, python=ns.python,
                                   replace=ns.replace)
            print(f"{what.upper()} {ns.name} -> host {ns.host} ({p})")
            written = [p] if what != "unchanged" else []
        elif ns.cmd == "register":
            p, what = register(ns.name, ns.pod_id, ns.project, root=ns.root, python=ns.python, guard=ns.guard, replace=ns.replace,
                               any_name=ns.any_name)
            print(f"{what.upper()} {ns.name} -> pod {ns.pod_id} ({p})")
            written = [p] if what != "unchanged" else []
        elif ns.cmd == "unregister":
            p = unregister(ns.name)
            print(f"UNREGISTERED {ns.name} ({p})")
            written = [p]
        elif ns.action == "list":
            rs = rows(mt)
            if ns.json:
                print(json.dumps(rs, indent=1))
            else:
                d = registry_dir(mt)
                print(f"machines.toml: {mt if mt.exists() else '(absent)'}; registry: {d if d else '(off: ' + REGISTRY_ENV + ' is none)'}")
                for r in rs:
                    print(f"{r['name']:<36} {r['source']:<13} {r['pod_id'] or '-':<16} {','.join(p for p in r['projects'] if p):<12} "
                          f"{r['problem'] or ''}".rstrip())
            return 1 if any(r["problem"] for r in rs) else 0
        else:
            d = write_dir(mt)
            running = None
            if not ns.all:
                from .connect import _list_pods  # noqa: PLC0415

                running = {p.get("id") for p in _list_pods() if (p.get("desiredStatus") or "").upper() == "RUNNING"}
            plan = plan_import(mt, d, running)
            written = []
            for r in plan:
                if r["action"] == "write" and ns.apply:
                    e = {k: r["entry"][k] for k in KNOWN_KEYS if k in r["entry"]}
                    written.append(write_entry(d, r["name"], {**e, "registered_by": "import from machines.toml"}))
                print(f"{('WROTE' if ns.apply else 'WOULD WRITE') if r['action'] == 'write' else 'skip':<11} {r['name']:<36} "
                      f"{r['pod_id'] or '-':<16} {r['why']}".rstrip())
            if ns.apply and mt.exists() and (fams := tomllib.loads(mt.read_text()).get("families")) and not (d / FAMILIES_FILE).exists():
                (d / FAMILIES_FILE).write_text("# [families.*] VRAM references (pods check-part), imported from machines.toml\n" +
                                               "".join(f"\n[families.{k}]\n" + "".join(f"{kk} = {_toml_value(vv)}\n" for kk, vv in v.items())
                                                       for k, v in fams.items()))
                written.append(d / FAMILIES_FILE)
            print(f"{sum(r['action'] == 'write' for r in plan)} to write into {d}" + ("" if ns.apply else " (--apply writes them)"))
    except (remote.Refused, ValueError) as e:
        print(f"research pods {ns.cmd}: {e}", file=sys.stderr)
        return 2
    if written and (line := _sync(written)):
        print(line, file=sys.stderr)
    return 0
