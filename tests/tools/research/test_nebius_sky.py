"""The Nebius nodes' store eviction and retention collector (pods/nebius/sky/): which units run where, what each pass deletes and keeps."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
from research.pods import registry

SKY = Path(registry.__file__).resolve().parent / "nebius" / "sky"


def test_the_store_eviction_drops_only_preserved_copies_hourly_as_ubuntu_and_fails_on_a_tool_error(tmp_path):
    """vy-store-evict: `research deploy` installs it hourly as ubuntu (the job store's owner, store_evict.service); it runs `research data evict --runs`
    against the job store and runs dir toward its free-space mark, logs one line, treats a mark it can't reach as done (nothing
    more is preserved), and fails only when the tool's output isn't its JSON."""
    import subprocess
    assert "User=ubuntu" in (SKY / "store_evict.service").read_text() and "OnUnitActiveSec=1h" in (SKY / "store_evict.timer").read_text()
    research = (SKY / "store_evict_research.service").read_text()
    assert "User=research" in research and "VY_EVICT_STORE=/workspace/research/store VY_EVICT_RUNS=/workspace/research/runs" in research
    tool, bin_dir = tmp_path / "tool", tmp_path / "bin"
    (tool / "research").mkdir(parents=True)
    bin_dir.mkdir()
    (tool / "research" / "__init__.py").write_text("")
    (tool / "research" / "__main__.py").write_text(
        "import json, os, sys\n"
        "open(os.environ['FAKE_ARGV'], 'w').write(json.dumps({'argv': sys.argv[1:], 'store': os.environ.get('RESEARCH_STORE')}))\n"
        "mode = os.environ['FAKE_MODE']\n"
        "if mode == 'error':\n    print('research data evict: [Errno 1] Operation not permitted'); sys.exit(1)\n"
        "print(json.dumps({'store': {'evicted_bytes': 184.6e9, 'reached_target': mode == 'reached'}, 'runs': {'evicted_bytes': 3e9}}))\n"
        "sys.exit(0 if mode == 'reached' else 1)\n")
    (bin_dir / "df").write_text('#!/bin/sh\necho " Avail"\necho " 2161"\n')
    (bin_dir / "df").chmod(0o755)

    def run(mode: str, **extra):
        env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "VY_RESEARCH_TOOL": str(tool), "FAKE_MODE": mode, "FAKE_ARGV": str(tmp_path / "argv.json"),
               "VY_EVICT_GIT": str(tmp_path / "no-git"), **extra}
        return subprocess.run(["bash", str(SKY / "store_evict.sh")], env=env, capture_output=True, text=True)

    r = run("reached")
    assert r.returncode == 0 and "evicted 184.6 GB of store blobs, 3.0 GB of run files; target reached: True" in r.stdout
    seen = json.loads((tmp_path / "argv.json").read_text())
    assert seen["store"] == "/workspace/jobs/store"
    assert seen["argv"] == ["data", "evict", "--target-free-gb", "2500", "--runs", "--runs-dir", "/workspace/jobs/runs", "--min-mb", "1e12", "--json"]
    r = run("short")
    assert r.returncode == 0 and "target reached: False" in r.stdout
    r = run("error")
    assert r.returncode == 1 and "Operation not permitted" in r.stderr
    r = run("reached", VY_EVICT_STORE="/workspace/research/store", VY_EVICT_RUNS="/workspace/research/runs")   # research's own instance
    seen = json.loads((tmp_path / "argv.json").read_text())
    assert r.returncode == 0 and seen["store"] == "/workspace/research/store" and seen["argv"][6] == "/workspace/research/runs"


def test_researchs_store_eviction_also_deletes_unused_source_trees_and_says_what_it_kept(tmp_path):
    """Something must prune /workspace/research/src, or a node runs out of inodes: research's unit (its drop-in, from deploy.toml)
    passes `--src-dir` with the capabilities its /proc scan needs, and the journal line counts what went and why
    the rest stayed, the reason it deleted nothing included."""
    import subprocess
    conf = (SKY / "store_evict_src.conf").read_text()
    assert "Environment=VY_EVICT_SRC=/workspace/research/src VY_EVICT_ORIGIN=/workspace/research/git/verity.git\n" in conf and "AmbientCapabilities=CAP_SYS_PTRACE CAP_DAC_READ_SEARCH\n" in conf
    dest = "/etc/systemd/system/vy-store-evict-research.service.d/src.conf"
    import tomllib
    deploy = (SKY.parent / "deploy.toml").read_text()
    files = tomllib.loads(deploy)["file"]
    src_conf = [f for f in files if f["path"] == dest]
    assert [f["src"] for f in src_conf] == ["tools/research/src/research/pods/nebius/sky/store_evict_src.conf"]
    unit_nodes = {n for f in files if f["path"] == "/etc/systemd/system/vy-store-evict-research.service" for n in f["nodes"]}
    assert set(src_conf[0]["nodes"]) == unit_nodes and "vy-nebius-cpu-12" in unit_nodes
    for u in ("store_evict.service", "store_evict.timer", "store_evict_research.service", "store_evict_research.timer"):
        assert f'src = "tools/research/src/research/pods/nebius/sky/{u}"' in deploy
    assert 'ExecCondition=/bin/grep -qs "timed False" /workspace/pouw/fill/status.txt' in (SKY / "store_evict_timed.conf").read_text()
    assert deploy.count('path = "/usr/local/bin/vy-store-evict"\nsrc = "tools/research/src/research/pods/nebius/sky/store_evict.sh"\nmode = "755"\n') == 2, \
        "a fresh install gets the repository's mode (644), which systemd can't exec (203/EXEC)"
    unit = (SKY / "store_evict_research.service").read_text()
    assert "User=research\n" in unit and "VY_EVICT_STORE=/workspace/research/store VY_EVICT_RUNS=/workspace/research/runs" in unit
    assert "ExecStart=/usr/local/bin/vy-store-evict\n" in unit and "OnUnitActiveSec=30min\n" in (SKY / "store_evict_research.timer").read_text()
    tool, bin_dir = tmp_path / "tool", tmp_path / "bin"
    (tool / "research").mkdir(parents=True)
    bin_dir.mkdir()
    (tool / "research" / "__init__.py").write_text("")
    (tool / "research" / "store").mkdir()
    (tool / "research" / "store" / "cli.py").write_text('ap.add_argument("--src-dir")\n')
    (tool / "research" / "__main__.py").write_text(
        "import json, os, sys\n"
        "open(os.environ['FAKE_ARGV'], 'w').write(json.dumps(sys.argv[1:]))\n"
        "open(os.environ['FAKE_ARGV'] + '.cwd', 'w').write(os.getcwd())\n"
        "kept = json.loads(os.environ.get('FAKE_KEPT') or json.dumps([{'tree': 'c'}] * 3))\n"
        "src = {'deleted': [{'tree': 'a'}, {'tree': 'b'}], 'deleted_bytes': 31e9, 'deleted_inodes': 812345, 'kept': kept,\n"
        "       'kept_by': {'1': 1, '2': 2}, 'error': os.environ.get('FAKE_ERROR')}\n"
        "err = {'error': os.environ['FAKE_STORE_ERROR']} if os.environ.get('FAKE_STORE_ERROR') else {}\n"
        "print(json.dumps({'store': {'evicted_bytes': 0, 'reached_target': False}, 'runs': {'evicted_bytes': 0}, 'src': src, **err}))\n"
        "sys.exit(1)\n")
    (bin_dir / "df").write_text('#!/bin/sh\necho " Avail"\necho " 1443"\n')
    (bin_dir / "df").chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "VY_RESEARCH_TOOL": str(tool), "FAKE_ARGV": str(tmp_path / "argv.json"),
           "VY_EVICT_STORE": "/workspace/research/store", "VY_EVICT_RUNS": "/workspace/research/runs", "VY_EVICT_SRC": "/workspace/research/src",
           "VY_EVICT_GIT": str(tmp_path / "no-git")}
    r = subprocess.run(["bash", str(SKY / "store_evict.sh")], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads((tmp_path / "argv.json").read_text()) == ["data", "evict", "--target-free-gb", "2500", "--runs", "--runs-dir",
                                                                 "/workspace/research/runs", "--src-dir", "/workspace/research/src", "--min-mb", "1e12",
                                                                 "--json"]
    assert "target reached: False; src trees: deleted 2 (31.0 GB, 812345 inodes), kept 3 (1:1 2:2); no store blob" in r.stdout
    # a pass that keeps trees under condition 2 says which process named each
    kept = [{"tree": t, "why": ["2: process 7 (du) uses it", *(["2: process 9 (lake) uses it"] if t == "c" else [])]} for t in "cde"]
    r = subprocess.run(["bash", str(SKY / "store_evict.sh")], env={**env, "FAKE_KEPT": json.dumps(kept)}, capture_output=True, text=True)
    assert "kept 3 (1:1 2:2); trees a process names: 7 (du) 3, 9 (lake) 1; no store blob" in r.stdout
    r = subprocess.run(["bash", str(SKY / "store_evict.sh")], env={**env, "VY_EVICT_ORIGIN": "/workspace/research/git/verity.git",
                                                                     "FAKE_ERROR": "2: /proc/1 is unreadable"}, capture_output=True, text=True)
    assert json.loads((tmp_path / "argv.json").read_text())[-5:-2] == ["--src-origin", "/workspace/research/git/verity.git", "--min-mb"]
    assert "kept 3 (1:1 2:2), nothing more deleted: 2: /proc/1 is unreadable; no store blob" in r.stdout
    assert r.returncode == 1 and "could not run" in r.stderr, "a pass that can't establish its conditions fails the unit (cpu-2, 16 Oct)"
    r = subprocess.run(["bash", str(SKY / "store_evict.sh")], env={**env, "FAKE_ERROR": "another pass holds /workspace/research/src/.evict.lock"},
                       capture_output=True, text=True)
    assert r.returncode == 0 and "nothing more deleted: another pass holds" in r.stdout, "an overlapping pass is not a fault"
    r = subprocess.run(["bash", str(SKY / "store_evict.sh")], env={**env, "FAKE_STORE_ERROR": "store pass: OperationalError: readonly"},
                       capture_output=True, text=True)
    assert "src trees: deleted 2 (31.0 GB, 812345 inodes)" in r.stdout and "; FAILED store pass: OperationalError: readonly" in r.stdout
    assert r.returncode == 1 and "the source-tree pass ran without it" in r.stderr, "a store it can't open fails the unit, after the trees"
    assert (tmp_path / "argv.json.cwd").read_text() == "/", "python -m puts its cwd first on sys.path: never a writable one"
    (tool / "research" / "store" / "cli.py").unlink()
    r = subprocess.run(["bash", str(SKY / "store_evict.sh")], env=env, capture_output=True, text=True)
    assert "--src-dir" not in json.loads((tmp_path / "argv.json").read_text()), "a tool too old for it still evicts the rest"
    assert r.returncode == 0 and "; src trees: not evicted, tool has no --src-dir; no store blob" in r.stdout


def test_researchs_store_eviction_prunes_the_pycache_of_sources_that_are_gone(tmp_path):
    """Nothing else prunes research's PYTHONPYCACHEPREFIX, where the bytecode of evicted source trees and uv build envs takes a
    node's inodes by the million. The hourly pass removes each mirrored directory whose source no longer exists, and keeps the rest."""
    import subprocess
    assert "Environment=VY_EVICT_PYCACHE=/workspace/research/cache/pycache\n" in (SKY / "store_evict_src.conf").read_text()
    tool, bin_dir = tmp_path / "tool", tmp_path / "bin"
    (tool / "research" / "store").mkdir(parents=True)
    bin_dir.mkdir()
    (tool / "research" / "__init__.py").write_text("")
    (tool / "research" / "__main__.py").write_text(
        "import json\nprint(json.dumps({'store': {'evicted_bytes': 0, 'reached_target': True}, 'runs': {'evicted_bytes': 0}}))\n")
    (bin_dir / "df").write_text('#!/bin/sh\necho " Avail"\necho " 1443"\n')
    (bin_dir / "df").chmod(0o755)
    live, src = tmp_path / "src" / "live" / "pkg", tmp_path / "src"
    live.mkdir(parents=True)
    pyc = tmp_path / "pycache"
    def mirror(p):
        return pyc / str(p).lstrip("/")
    for d, n in ((live, 2), (src / "gone" / "pkg", 3), (live / "removed", 1)):
        mirror(d).mkdir(parents=True)
        for i in range(n):
            (mirror(d) / f"m{i}.cpython-312.pyc").write_bytes(b"x")
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "VY_RESEARCH_TOOL": str(tool), "VY_EVICT_GIT": str(tmp_path / "no-git"),
           "VY_EVICT_PYCACHE": str(pyc)}
    r = subprocess.run(["bash", str(SKY / "store_evict.sh")], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "; pycache: removed 2 dirs (4 files) whose source is gone (tool)" in r.stdout, r.stdout
    assert sorted(p.name for p in mirror(live).iterdir()) == ["m0.cpython-312.pyc", "m1.cpython-312.pyc"]
    assert not mirror(src / "gone").exists() and mirror(src / "live").is_dir()


def test_researchs_store_eviction_also_removes_what_nothing_else_does_and_says_what_stayed(tmp_path):
    """Request directories without a run dir, the clones of runs that are over, the store's lock files and uv's editable shards
    took a node's inodes by the million with nothing removing them. Research's unit points a pass at each (its drop-in, beside the
    capabilities their /proc scan needs), the journal line counts what each removed and why the rest stayed, naming the first path
    in use and the runs whose request or clone went (no lock file or shard: they hold nothing), and one that could not run fails
    the unit. A variable left empty switches its removal off."""
    import subprocess
    conf = (SKY / "store_evict_src.conf").read_text()
    assert "Environment=VY_EVICT_REQUESTS=/workspace/research/requests VY_EVICT_SCRATCH=/workspace/research/scratch\n" in conf
    assert "Environment=VY_EVICT_LOCKS=1 VY_EVICT_UV=/workspace/cache/uv\n" in conf and "AmbientCapabilities=" in conf
    assert " UV_CACHE_DIR=/workspace/cache/uv " in (SKY.parent / "vm_setup.sh").read_text(), "the cache the nodes' uv writes is the one the pass reads"
    tool, bin_dir = tmp_path / "tool", tmp_path / "bin"
    (tool / "research" / "store").mkdir(parents=True)
    bin_dir.mkdir()
    (tool / "research" / "__init__.py").write_text("")
    flags = ("--requests-dir", "--scratch-dir", "--locks", "--uv-cache", "--uv-keep-h")
    (tool / "research" / "store" / "cli.py").write_text("".join(f'ap.add_argument("{f}")\n' for f in flags))
    (tool / "research" / "__main__.py").write_text(
        "import json, os, sys\n"
        "open(os.environ['FAKE_ARGV'], 'w').write(json.dumps(sys.argv[1:]))\n"
        "left = json.loads(os.environ['FAKE_LEFT'])\n"
        "print(json.dumps({'store': {'evicted_bytes': 0, 'reached_target': False}, 'runs': {'evicted_bytes': 0},\n"
        "                  **{k: v for k, v in left.items() if '--' + {'requests': 'requests-dir', 'clones': 'scratch-dir', 'locks': 'locks',\n"
        "                                                           'uv': 'uv-cache'}[k] in sys.argv}}))\n"
        "sys.exit(1)\n")
    (bin_dir / "df").write_text('#!/bin/sh\necho " Avail"\necho " 1443"\n')
    (bin_dir / "df").chmod(0o755)
    shard = "/workspace/cache/uv/sdists-v9/editable/0123456789abcdef"
    a, b = "r20330518-033300-0a1b", "r20330518-033301-0a1b"
    left = {"requests": {"deleted": 2, "deleted_bytes": 3e6, "deleted_inodes": 12, "kept": 3, "kept_by": {"recent": 2, "run": 1}, "kept_first": [],
                         "deleted_first": [f"/workspace/research/requests/{a}", f"/workspace/research/requests/{b}"], "error": None},
            "clones": {"deleted": 21, "deleted_bytes": 4.5e9, "deleted_inodes": 189264, "kept": 0, "kept_by": {}, "kept_first": [],
                       "deleted_first": [f"/workspace/research/scratch/{a}"], "error": None},
            "locks": {"deleted": 316915, "deleted_bytes": 0, "deleted_inodes": 316915, "kept": 4, "kept_by": {"recent": 4}, "kept_first": [],
                      "deleted_first": ["/workspace/research/store/locks/blob-" + "0" * 64 + ".lock"], "error": None},
            "uv": {"deleted": 5, "deleted_bytes": 2e8, "deleted_inodes": 51, "kept": 9, "kept_by": {"2": 1, "recent": 6, "retired": 2},
                   "kept_first": [{"path": shard, "why": "2: its lock is held (.lock)"}], "error": None}}
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "VY_RESEARCH_TOOL": str(tool), "FAKE_ARGV": str(tmp_path / "argv.json"),
           "VY_EVICT_STORE": "/workspace/research/store", "VY_EVICT_RUNS": "/workspace/research/runs", "VY_EVICT_GIT": str(tmp_path / "no-git"),
           "VY_EVICT_REQUESTS": "/workspace/research/requests", "VY_EVICT_SCRATCH": "/workspace/research/scratch", "VY_EVICT_LOCKS": "1",
           "VY_EVICT_UV": "/workspace/cache/uv", "FAKE_LEFT": json.dumps(left)}

    def run(**extra):
        r = subprocess.run(["bash", str(SKY / "store_evict.sh")], env={**env, **extra}, capture_output=True, text=True)
        return r, json.loads((tmp_path / "argv.json").read_text())

    r, argv = run()
    assert r.returncode == 0, r.stderr
    assert argv == ["data", "evict", "--target-free-gb", "2500", "--runs", "--runs-dir", "/workspace/research/runs",
                    "--requests-dir", "/workspace/research/requests", "--scratch-dir", "/workspace/research/scratch", "--locks",
                    "--uv-cache", "/workspace/cache/uv", "--min-mb", "1e12", "--json"]
    assert (f"target reached: False; requests: deleted 2 (0.0 GB, 12 inodes), kept 3 (recent:2 run:1), gone: {a} {b}"
            f"; clones: deleted 21 (4.5 GB, 189264 inodes), kept 0 (none), gone: {a} ...; store locks: deleted 316915 (0.0 GB, 316915 inodes), kept 4 (recent:4)"
            f"; uv cache: deleted 5 (0.2 GB, 51 inodes), kept 9 (2:1 recent:6 retired:2), e.g. {shard}: 2: its lock is held (.lock)"
            "; no store blob") in r.stdout, r.stdout
    r, argv = run(VY_EVICT_UV="", VY_EVICT_LOCKS="0", VY_EVICT_UV_KEEP_H="72")
    assert r.returncode == 0 and "--uv-cache" not in argv and "--uv-keep-h" not in argv and "--locks" not in argv, "left empty: switched off"
    assert argv[7:11] == ["--requests-dir", "/workspace/research/requests", "--scratch-dir", "/workspace/research/scratch"]
    assert "uv cache" not in r.stdout and "store locks" not in r.stdout and "; clones: deleted 21 " in r.stdout
    r, argv = run(VY_EVICT_UV_KEEP_H="72")
    assert argv[12:16] == ["--uv-cache", "/workspace/cache/uv", "--uv-keep-h", "72"]
    failed = {**left, "uv": {**left["uv"], "deleted": 0, "error": "2: /proc/1 is unreadable"}}
    r, argv = run(FAKE_LEFT=json.dumps(failed))
    assert "kept 9 (2:1 recent:6 retired:2), e.g. " in r.stdout and ", nothing more deleted: 2: /proc/1 is unreadable; no store blob" in r.stdout
    assert r.returncode == 1 and "could not run" in r.stderr and "; clones: deleted 21 " in r.stdout, "the unit fails, after the line"
    (tool / "research" / "store" / "cli.py").write_text('ap.add_argument("--requests-dir")\n')
    r, argv = run()
    assert r.returncode == 0 and argv[7:9] == ["--requests-dir", "/workspace/research/requests"] and "--locks" not in argv
    assert (f"; requests: deleted 2 (0.0 GB, 12 inodes), kept 3 (recent:2 run:1), gone: {a} {b}; clones: not removed, tool has no --scratch-dir"
            "; store locks: not removed, tool has no --locks; uv cache: not removed, tool has no --uv-cache; no store blob") in r.stdout


def test_the_store_eviction_keeps_the_fixture_registry_of_the_newest_shipped_trees_and_else_evicts_no_blob(tmp_path):
    """Node 1's runs have no private remote and the public export can lack a fixture, so the evictor passes the union of
    `artifacts.json` at the newest `refs/research/src/*` commits as `--keep-registry`; with no registry to read, or a tool
    without the flag, it evicts no store blob or run file (`--min-mb 1e12`) and says why.  A commit from before the registry
    moved to the root holds it under `fixtures/` (the older commit here), and its fixtures are kept like a newer one's."""
    import subprocess
    git_dir = tmp_path / "verity.git"
    work = tmp_path / "work"
    (work / "fixtures").mkdir(parents=True)
    root, before = Path("artifacts.json"), Path("fixtures", "artifacts.json")

    def git(*a, date=0):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=work, check=True, capture_output=True,
                       env={"PATH": "/usr/bin:/bin", "GIT_COMMITTER_DATE": f"@{1_700_000_000 + date}", "GIT_AUTHOR_DATE": f"@{1_700_000_000 + date}"})

    subprocess.run(["git", "init", "-q", "--bare", str(git_dir)], check=True)
    git("init", "-q")
    for n, (place, reg) in enumerate(((before, {"art:old": "a"}), (root, {"art:new": "b", "previous": {"art:new": ["art:gone"]}}))):
        (work / before).unlink(missing_ok=True)
        (work / place).write_text(json.dumps(reg))
        git("add", "-A", ".")
        git("commit", "-q", "-m", str(n), date=n)
        git("push", "-q", str(git_dir), f"HEAD:refs/research/src/c{n}")
    tool, bin_dir = tmp_path / "tool", tmp_path / "bin"
    (tool / "research" / "store").mkdir(parents=True)
    bin_dir.mkdir()
    (tool / "research" / "__init__.py").write_text("")
    (tool / "research" / "store" / "cli.py").write_text('ap.add_argument("--keep-registry")\n')
    (tool / "research" / "__main__.py").write_text(
        "import json, os, sys\n"
        "a = sys.argv[1:]\n"
        "reg = json.load(open(a[a.index('--keep-registry') + 1])) if '--keep-registry' in a else None\n"
        "open(os.environ['FAKE_ARGV'], 'w').write(json.dumps({'argv': a, 'registry': reg}))\n"
        "print(json.dumps({'store': {'evicted_bytes': 0, 'reached_target': False}, 'runs': {'evicted_bytes': 0}}))\n")
    (bin_dir / "df").write_text('#!/bin/sh\necho " Avail"\necho " 1741"\n')
    (bin_dir / "df").chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "VY_RESEARCH_TOOL": str(tool), "FAKE_ARGV": str(tmp_path / "argv.json"),
           "VY_EVICT_GIT": str(git_dir)}

    def run(**extra):
        r = subprocess.run(["bash", str(SKY / "store_evict.sh")], env={**env, **extra}, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return r.stdout, json.loads((tmp_path / "argv.json").read_text())

    out, seen = run()
    assert seen["registry"] == {"art:old": "a", "art:new": "b"} and "--min-mb" not in seen["argv"] and "; kept 2 registry fixtures" in out
    out, seen = run(VY_EVICT_KEEP_REFS="1")
    assert seen["registry"] == {"art:new": "b"}, "only the newest commits' registries"
    out, seen = run(VY_EVICT_GIT=str(tmp_path / "no-git"))
    assert seen["argv"][-3:] == ["--min-mb", "1e12", "--json"] and "--keep-registry" not in seen["argv"]
    assert "; no store blob or run file evicted:" in out
    (tool / "research" / "store" / "cli.py").write_text("")
    out, seen = run()
    assert seen["argv"][-3:] == ["--min-mb", "1e12", "--json"] and "no store blob or run file evicted: tool has no --keep-registry (tool)" in out


def test_retention_gc_runs_hourly_on_the_nodes_outside_a_timed_window(tmp_path):
    """The retention policy is applied on a schedule, so a node never sits at 80% with terabytes collectable while the fill
    runner holds every queued run (the merge gates among them) for a `research retention gc` by hand. It reads the node's
    store, whose catalog records what custody verified on the remote: without it gc keeps every preserved path. It is installed
    wherever research runs are: the nodes whose runs the store eviction trims, and those `--on auto` places runs on."""
    import subprocess
    import tomllib

    from research import auto
    unit, timer = (SKY / "retention_gc.service").read_text(), (SKY / "retention_gc.timer").read_text()
    assert "User=research\n" in unit and "python3 -m research retention gc --apply $F'" in unit and "OnUnitActiveSec=1h\n" in timer
    assert "Environment=RESEARCH_STORE=/workspace/research/store\n" in unit
    deploy = (SKY.parent / "deploy.toml").read_text()
    doc = tomllib.loads(deploy)
    files = {f["path"]: f for f in doc["file"]}
    holds_runs = set(files["/etc/systemd/system/vy-store-evict-research.timer"]["nodes"]) | {m for m in doc["nodes"] if auto.CPU_NODE.fullmatch(m)}
    for u, path in (("retention_gc.service", "vy-retention-gc.service"), ("retention_gc.timer", "vy-retention-gc.timer")):
        f = files[f"/etc/systemd/system/{path}"]
        assert f["src"] == f"tools/research/src/research/pods/nebius/sky/{u}" and f["run"] == ["systemd:vy-retention-gc.timer"]
        assert set(f["nodes"]) == holds_runs, "the collector runs on every node of deploy.toml that holds research runs"
    cond = re.search(r"^ExecCondition=/bin/sh -c '(.*)'$", unit, re.M)[1]
    status = tmp_path / "status.txt"
    def gate():
        return subprocess.run(["sh", "-c", cond.replace("/workspace/pouw/fill/status.txt", str(status))]).returncode
    assert gate() == 0, "a node with no fill status runs it"
    status.write_text("2026-10-19T06:40:00Z free GPUs 0/8, timed True, window waiting False\n")
    assert gate() != 0, "node 2 inside a timed window: skipped"
    status.write_text("2026-10-19T06:40:00Z free GPUs 0/8, timed False, window waiting False\n")
    assert gate() == 0


CLEAN_UP_UNITS = ("store_evict.service", "store_evict_research.service", "retention_gc.service", "launcher_copies_gc.service")


def _copies(tmp_path: Path) -> dict[str, Path]:
    """Two commits' directories of main's copy with the link on the first (as `research deploy install` leaves a whole tree),
    a newer snapshot shipped from a branch, and an `ls` that offers that snapshot to whatever lists the snapshots.

    Each `research` writes $FAKE_OUT: where it ran from, its PYTHONPATH, its working directory, its arguments, and the
    version of a module it imports only at its end; with $FAKE_GATE it first says it has started and waits for its stdin
    to close."""
    main = ("import json, os, sys\n"
            "if os.environ.get('FAKE_GATE'):\n"
            "    open(os.environ['FAKE_GATE'], 'w').write('started')\n"
            "    sys.stdin.read()\n"
            "from research import late\n"
            "here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\n"
            "json.dump({'from': here, 'pythonpath': os.environ.get('PYTHONPATH'), 'cwd': os.getcwd(), 'argv': sys.argv[1:], 'late': late.V},\n"
            "          open(os.environ['FAKE_OUT'], 'w'))\n"
            "print(json.dumps({'store': {'evicted_bytes': 0, 'reached_target': True}, 'runs': {'evicted_bytes': 0}}))\n")
    dirs = {"v1": tmp_path / "copy" / ".deploy-tree" / "clean-up" / "v1", "v2": tmp_path / "copy" / ".deploy-tree" / "clean-up" / "v2",
            "branch": tmp_path / "snapshots" / "branch"}
    for name, d in dirs.items():
        (d / "research").mkdir(parents=True)
        (d / "research" / "__init__.py").write_text("")
        (d / "research" / "__main__.py").write_text(main)
        (d / "research" / "late.py").write_text(f"V = {name!r}\n")
    link = tmp_path / "copy" / "clean-up"
    link.symlink_to(dirs["v1"])
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, text in (("df", 'echo " Avail"\necho " 2161"\n'), ("ls", f'echo listed >> "{tmp_path}/ls.calls"\necho "{dirs["branch"]}/"\n'),
                       ("timeout", 'shift\nexec "$@"\n')):
        (bin_dir / name).write_text("#!/bin/sh\n" + text)
        (bin_dir / name).chmod(0o755)
    return {**dirs, "link": link, "bin": bin_dir, "out": tmp_path / "ran.json", "ls": tmp_path / "ls.calls"}


def _swap(link: Path, to: Path) -> None:
    """Point the link at `to` by one rename, as the install of a whole tree does."""
    link.with_name(link.name + ".new").symlink_to(to)
    os.replace(link.with_name(link.name + ".new"), link)


def test_a_clean_up_pass_runs_mains_copy_and_never_a_snapshot_a_run_shipped(tmp_path):
    """The eviction and the collector delete, so what they run is main's code, which only `research deploy install` places:
    each unit names the manifest's whole tree, a pass resolves that link and runs the directory it names, and no pass lists
    the snapshots `research run` ships, where the newest is whatever branch launched last. Without the copy a pass refuses
    and runs nothing, though a snapshot is there to be found."""
    import shlex
    import subprocess
    import tomllib
    doc = tomllib.loads((SKY.parent / "deploy.toml").read_text())
    trees = [t for t in doc["tree"] if t["src"] == "tools/research/src"]
    assert len(trees) == 1 and trees[0].get("whole") is True and trees[0].get("required") is True, "main's copy: one tree, whole"
    assert trees[0]["path"].startswith("/usr/local/lib/"), "root's: a run, which writes /workspace/research as research, can't replace it"
    for u in CLEAN_UP_UNITS:
        unit = (SKY / u).read_text()
        assert f"\nEnvironment=VY_RESEARCH_TOOL={trees[0]['path']}\n" in unit, f"{u} names the tree deploy.toml installs"
        on = {n for f in doc["file"] if f.get("src", "").endswith(f"/sky/{u}") for n in f["nodes"]}
        assert on and not on - set(trees[0]["nodes"]), f"{sorted(on - set(trees[0]['nodes']))} get {u} without the code it runs"
    picks = [p.name for p in SKY.parent.rglob("*") if p.is_file() and p.suffix in (".sh", ".service", ".conf", ".timer")
             and "research/tool/*" in p.read_text(errors="replace")]
    assert not picks, f"{picks} list the tool snapshots: what a unit runs is a tree deploy.toml installs"

    w = _copies(tmp_path)
    env = {"PATH": f"{w['bin']}:/usr/bin:/bin", "FAKE_OUT": str(w["out"]), "VY_EVICT_GIT": str(tmp_path / "no-git")}
    r = subprocess.run(["bash", str(SKY / "store_evict.sh")], env={**env, "VY_RESEARCH_TOOL": str(w["link"])}, capture_output=True, text=True)
    ran = json.loads(w["out"].read_text())
    assert r.returncode == 0 and r.stdout.rstrip().endswith("(v1)"), r.stderr
    assert (ran["from"], ran["pythonpath"], ran["late"], ran["cwd"]) == (str(w["v1"]), str(w["v1"]), "v1", "/")
    assert not w["ls"].exists(), "the snapshots were listed"
    w["out"].unlink()
    for missing in ({}, {"VY_RESEARCH_TOOL": ""}, {"VY_RESEARCH_TOOL": str(tmp_path / "copy" / "absent")}, {"VY_RESEARCH_TOOL": str(tmp_path / "copy")}):
        r = subprocess.run(["bash", str(SKY / "store_evict.sh")], env={**env, **missing}, capture_output=True, text=True)
        assert r.returncode == 1 and "no research package at VY_RESEARCH_TOOL" in r.stderr and "research deploy install" in r.stderr
        assert not w["out"].exists() and not w["ls"].exists(), "nothing ran, and no snapshot was looked for"

    line = re.search(r"^ExecStart=(.*)$", (SKY / "retention_gc.service").read_text(), re.M)[1]
    assert "${" not in line and "%" not in line, "systemd would expand it before the shell reads it"
    argv = shlex.split(line)
    assert argv[:2] == ["/bin/sh", "-c"] and len(argv) == 3

    def gc(**extra):
        return subprocess.run(["sh", "-c", argv[2]], env={**env, **extra}, capture_output=True, text=True)

    r = gc(VY_RESEARCH_TOOL=str(w["link"]))
    ran = json.loads(w["out"].read_text())
    assert r.returncode == 0 and f"vy-retention-gc: research from {w['v1']}\n" in r.stderr
    assert (ran["from"], ran["pythonpath"], ran["cwd"], ran["argv"]) == (str(w["v1"]), str(w["v1"]), str(w["v1"]), ["retention", "gc", "--apply"])
    assert not w["ls"].exists(), "the snapshots were listed"
    w["out"].unlink()
    for missing in ({}, {"VY_RESEARCH_TOOL": ""}, {"VY_RESEARCH_TOOL": str(tmp_path / "copy" / "absent")}, {"VY_RESEARCH_TOOL": str(tmp_path / "copy")}):
        r = gc(**missing)
        assert r.returncode == 1 and "no research package at VY_RESEARCH_TOOL" in r.stderr and "research deploy install" in r.stderr
        assert not w["out"].exists() and not w["ls"].exists(), "nothing ran, and no snapshot was looked for"


def test_the_collector_takes_unpreserved_runs_on_node_2_and_cpu_1_to_16_alone(tmp_path):
    """`--unpreserved-runs` deletes a run's only copy, so the unit passes it only where a tracked drop-in says so, which
    deploy.toml installs on vy-nebius-2 and vy-nebius-cpu-1 to cpu-16, each with the unit and the code it runs, and on no other
    machine: no other entry installs it or writes its path, and no other file of the nodes' sets the switch."""
    import shlex
    import subprocess
    import tomllib
    conf = "tools/research/src/research/pods/nebius/sky/retention_gc_unpreserved.conf"
    dropin = "/etc/systemd/system/vy-retention-gc.service.d/20-unpreserved-runs.conf"
    takers = ["vy-nebius-2"] + [f"vy-nebius-cpu-{i}" for i in range(1, 17)]
    doc = tomllib.loads((SKY.parent / "deploy.toml").read_text())
    entries = doc["file"] + doc["tree"]
    got = [(f["path"], f["nodes"]) for f in entries if f.get("src") == conf or f["path"] == dropin]
    assert got == [(dropin, takers)], "the drop-in, once, on vy-nebius-2 and cpu-1 to cpu-16 and on no other machine"
    assert not set(takers) - set(doc["nodes"]), f"{sorted(set(takers) - set(doc['nodes']))} have no [nodes] table"
    for needed in ("/etc/systemd/system/vy-retention-gc.service", "/etc/systemd/system/vy-retention-gc.timer", "/usr/local/lib/vy-monitoring/clean-up"):
        on = {n for f in entries if f["path"] == needed for n in f["nodes"]}
        assert not set(takers) - on, f"{sorted(set(takers) - on)} get the drop-in without {needed}"
    switch = [p.relative_to(SKY.parent).as_posix() for p in sorted(SKY.parent.rglob("*"))
              if p.is_file() and "VY_GC_UNPRESERVED_RUNS=1" in p.read_text(errors="replace")]
    assert switch == ["sky/retention_gc_unpreserved.conf"], f"{switch}: only the drop-in sets the switch"
    sets = [ln for ln in (SKY / "retention_gc_unpreserved.conf").read_text().splitlines() if ln and not ln.startswith("#")]
    assert sets == ["[Service]", "Environment=VY_GC_UNPRESERVED_RUNS=1"], "it sets the switch and replaces nothing of the unit"
    assert "unpreserved" not in "".join((SKY / u).read_text() for u in CLEAN_UP_UNITS if u != "retention_gc.service")
    w = _copies(tmp_path)
    script = shlex.split(re.search(r"^ExecStart=(.*)$", (SKY / "retention_gc.service").read_text(), re.M)[1])[2]
    env = {"PATH": f"{w['bin']}:/usr/bin:/bin", "FAKE_OUT": str(w["out"]), "VY_RESEARCH_TOOL": str(w["link"])}
    for extra, flags in (({}, []), ({"VY_GC_UNPRESERVED_RUNS": "1"}, ["--unpreserved-runs"]), ({"VY_GC_UNPRESERVED_RUNS": "0"}, []),
                         ({"VY_GC_UNPRESERVED_RUNS": ""}, []), ({"VY_GC_UNPRESERVED_RUNS": "--force"}, [])):
        r = subprocess.run(["sh", "-c", script], env={**env, **extra}, capture_output=True, text=True)
        assert r.returncode == 0 and json.loads(w["out"].read_text())["argv"] == ["retention", "gc", "--apply", *flags], (extra, r.stderr)


def test_the_checkers_copies_of_its_runs_are_collected_hourly_on_cpu_7_alone_and_only_while_it_says_which_it_needs(tmp_path):
    """Root's runs root on vy-nebius-cpu-7 holds the checker's copy of every run it launches, which the node's collector can't
    read and gives no record. Its own unit runs there alone, as root, with --launcher-copies over that directory: the flag that
    gives the copies their record is the one that holds the forced pass to them, so the unit deletes nothing else there. A pass deletes
    by what the remote says and keeps the copies the checker lists, so it refuses, running nothing, without a list written in
    the last 15 minutes (a checker that is down or doesn't write one would have every copy look spare), without the store
    key to ask the remote with, and without main's copy of the code."""
    import shlex
    import subprocess
    import tomllib
    unit, timer = (SKY / "launcher_copies_gc.service").read_text(), (SKY / "launcher_copies_gc.timer").read_text()
    files = {f["path"]: f for f in tomllib.loads((SKY.parent / "deploy.toml").read_text())["file"]}
    for u, path in (("launcher_copies_gc.service", "vy-launcher-copies-gc.service"), ("launcher_copies_gc.timer", "vy-launcher-copies-gc.timer")):
        f = files[f"/etc/systemd/system/{path}"]
        assert f["src"] == f"tools/research/src/research/pods/nebius/sky/{u}" and f["run"] == ["systemd:vy-launcher-copies-gc.timer"]
        assert f["nodes"] == ["vy-nebius-cpu-7"] and f["owner"] == "infra"
    assert "User=" not in unit, "root's: the directory is root's, and so is the store key"
    assert "OnUnitActiveSec=1h\n" in timer and "Environment=VY_LAUNCHER_RUNS=/workspace/research/root-runs\n" in unit
    line = re.search(r"^ExecStart=(.*)$", unit, re.M)[1]
    assert "${" not in line and "%" not in line, "systemd would expand it before the shell reads it"
    argv = shlex.split(line)
    assert argv[:2] == ["/bin/sh", "-c"] and len(argv) == 3

    w = _copies(tmp_path)
    runs, named, key = tmp_path / "root-runs", tmp_path / "named-runs", tmp_path / "r2.env"
    named.mkdir()
    (named / "runs").write_text("r20261001-000000-aaaa\n")
    key.write_text(f"FAKE_OUT={w['out']}\n")
    env = {"PATH": f"{w['bin']}:/usr/bin:/bin", "VY_RESEARCH_TOOL": str(w["link"]), "VY_LAUNCHER_RUNS": str(runs),
           "VY_LAUNCHER_NAMED": str(named), "VY_LAUNCHER_STORE_ENV": str(key)}

    def gc(**extra):
        return subprocess.run(["sh", "-c", argv[2]], env={**env, **extra}, capture_output=True, text=True)

    r = gc()
    assert r.returncode == 0 and f"vy-launcher-copies-gc: research from {w['v1']}\n" in r.stderr, r.stderr
    ran = json.loads(w["out"].read_text())                # written where the key file's variable says: the pass exported it
    assert (ran["from"], ran["pythonpath"], ran["cwd"]) == (str(w["v1"]), str(w["v1"]), str(w["v1"]))
    assert ran["argv"] == ["retention", "gc", "--root", str(runs), "--launcher-copies", "--force", "--low-pct", "0", "--max-paths", "100",
                           "--jobs-dir", str(named), "--apply"]
    assert "--launcher-copies" in ran["argv"] and "--unpreserved-runs" not in ran["argv"], (
        "forced over a whole runs root, the pass takes every expired record there unless --launcher-copies holds it to the copies")
    w["out"].unlink()

    def refused(why: str, **extra) -> None:
        r = gc(**extra)
        assert r.returncode == 1 and why in r.stderr and r.stderr.rstrip().endswith("nothing was deleted"), r.stderr
        assert not w["out"].exists(), "nothing ran"

    os.utime(named / "runs", (0, 0))
    refused("is missing or was written over 15 min ago")
    (named / "runs").unlink()
    refused("is missing or was written over 15 min ago")
    (named / "runs").mkdir()
    refused("is missing or was written over 15 min ago")
    (named / "runs").rmdir()
    (named / "runs").write_text("")
    refused("no store key at VY_LAUNCHER_STORE_ENV", VY_LAUNCHER_STORE_ENV=str(tmp_path / "absent.env"))
    refused("no research package at VY_RESEARCH_TOOL", VY_RESEARCH_TOOL=str(tmp_path / "copy" / "absent"))
    assert gc().returncode == 0 and w["out"].exists(), "an empty list written just now is a list: the checker needs none"


@pytest.mark.parametrize("unit", ["the eviction", "the collector"])
def test_a_pass_that_started_before_an_install_finishes_on_the_directory_it_started_on(tmp_path, unit):
    """An install of main's copy renames the link onto a new directory. A pass resolved the link once, when it started, so
    what it imports after the rename still comes from the directory it started on, and a pass that starts after the rename
    runs the new directory's code and nothing of the old."""
    import select
    import shlex
    import subprocess
    w = _copies(tmp_path)
    gate = tmp_path / "started"
    os.mkfifo(gate)
    reads = os.open(gate, os.O_RDWR)        # so the pass's open of it never waits, and this test waits on what the pass writes
    env = {"PATH": f"{w['bin']}:/usr/bin:/bin", "VY_RESEARCH_TOOL": str(w["link"]), "VY_EVICT_GIT": str(tmp_path / "no-git")}
    argv = (["bash", str(SKY / "store_evict.sh")] if unit == "the eviction" else
            ["sh", "-c", shlex.split(re.search(r"^ExecStart=(.*)$", (SKY / "retention_gc.service").read_text(), re.M)[1])[2]])
    first = subprocess.Popen(argv, env={**env, "FAKE_OUT": str(tmp_path / "first.json"), "FAKE_GATE": str(gate)}, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        ready, _, _ = select.select([reads, first.stdout], [], [])      # it started, or it ended without starting
        assert reads in ready and os.read(reads, 16) == b"started"      # it runs v1's code, and has not yet imported all of it
        assert os.readlink(f"/proc/{first.pid}/cwd") == str(w["v1"]), "the pass's process names its directory, so an install keeps it"
        _swap(w["link"], w["v2"])
        r = subprocess.run(argv, env={**env, "FAKE_OUT": str(tmp_path / "second.json")}, stdin=subprocess.DEVNULL, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    finally:
        out, err = first.communicate("")
        os.close(reads)
    assert first.returncode == 0, err
    ran = {n: json.loads((tmp_path / f"{n}.json").read_text()) for n in ("first", "second")}
    assert (ran["first"]["from"], ran["first"]["late"]) == (str(w["v1"]), "v1"), "the first pass read a file installed after it started"
    assert (ran["second"]["from"], ran["second"]["late"]) == (str(w["v2"]), "v2")
    if unit == "the eviction":
        assert out.rstrip().endswith("(v1)") and r.stdout.rstrip().endswith("(v2)"), "each pass's line names its directory"
    else:
        assert f"research from {w['v1']}\n" in err and f"research from {w['v2']}\n" in r.stderr, "each pass's line names its directory"


def test_every_cpu_node_the_retention_collector_runs_on_gets_researchs_store_eviction_whole():
    """A node that holds research runs gets a source tree for each commit `research run` ships there, and only research's
    store eviction with src.conf deletes them: under the retention collector alone the node fills. So every CPU node the
    collector is on gets the eviction, and none gets its service without the script it runs, its timer and src.conf."""
    import tomllib

    from research import auto
    files = tomllib.loads((SKY.parent / "deploy.toml").read_text())["file"]

    def on(path):
        return {n for f in files if f["path"] == path for n in f["nodes"]}

    unit = "/etc/systemd/system/vy-store-evict-research"
    service, script = on(f"{unit}.service"), on("/usr/local/bin/vy-store-evict")
    holds_runs = {n for n in on("/etc/systemd/system/vy-retention-gc.timer") if auto.CPU_NODE.fullmatch(n)}
    assert len(holds_runs) > 2, "the retention collector's CPU nodes"
    assert not holds_runs - service, (f"{sorted(holds_runs - service)} hold research runs without research's store eviction: add them "
                                      f"to the `nodes` of its script, {unit}.service, .timer and .service.d/src.conf")
    assert not service - script, f"{sorted(service - script)} get the service without the script it runs: add them to its `nodes`"
    for part in (f"{unit}.timer", f"{unit}.service.d/src.conf"):
        assert on(part) == service, f"{part} and the service it belongs to are on different nodes: {sorted(on(part) ^ service)}"
