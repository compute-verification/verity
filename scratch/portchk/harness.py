"""PORTCHK steps 1-2: each PR's head merged onto the port's head (its merge drivers), the codemod, then --check.

    python3 /tmp/portchk/harness.py --workers K [PR...]   (default: every PR in /tmp/portchk_prs.json)

Writes /tmp/portchk/out/<N>.json. The merged-and-converted commit is refs/portchk/merged/<N> when the merge is clean; when
it isn't, the tree with each conflict resolved mechanically (the PR's deletions kept, the PR's side of a lock, a lakefile and
a .lean hunk, the port's side of anything else) and then converted is refs/portchk/resolved/<N>, for the build step only.
Nothing here is pushed.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/agent/repos/verity")
PORT = "27ce82e6010eefb5064f2a497b8a9c4500a9a4c2"
OUT = Path("/tmp/portchk/out")
CODEMOD = "tools/verity/lean/modules.py"
ENV = {"GIT_AUTHOR_NAME": "portchk", "GIT_AUTHOR_EMAIL": "portchk@invalid", "GIT_COMMITTER_NAME": "portchk",
       "GIT_COMMITTER_EMAIL": "portchk@invalid", "PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(Path.home())}
GENERATED = re.compile(r"(_lean\.json|TRUSTED\.sha256)$")
LAKE = re.compile(r"(lakefile\.toml|lakefile\.lean|lake-manifest\.json|lean-toolchain)$")


def git(*a, cwd=REPO, check=True, env=None):
    p = subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True, env=env)
    if check and p.returncode:
        raise RuntimeError(f"git {' '.join(a)}: {p.stderr[-2000:]}")
    return p


def reset(wt: Path, rev: str):
    git("merge", "--abort", cwd=wt, check=False)
    git("reset", "-q", "--hard", cwd=wt, check=False)
    git("checkout", "-q", "-f", "--detach", rev, cwd=wt)
    git("clean", "-q", "-fd", cwd=wt)


def ensure(wt: Path, rev: str):
    if not wt.exists():
        git("worktree", "add", "-q", "--detach", str(wt), rev)


def conflicts(wt: Path) -> dict[str, str]:
    out = {}
    for line in git("status", "--porcelain=v1", "-uno", cwd=wt).stdout.splitlines():
        xy, path = line[:2], line[3:]
        if xy in ("DD", "AU", "UD", "UA", "DU", "AA", "UU"):
            out[path] = xy
    return out


def kind(path: str, xy: str) -> str:
    if path.endswith(".lean"):
        return "lean" if xy in ("UU", "AA") else "lean-deleted"
    if path.endswith("lean-audit.json"):
        return "lock"
    if GENERATED.search(path):
        return "generated"
    if LAKE.search(path):
        return "lake"
    return "other" if xy in ("UU", "AA") else "other-deleted"


def merge(wt: Path, base: str, ref: str) -> tuple[int, dict[str, str], str]:
    reset(wt, base)
    p = git("merge", "--no-ff", "--no-edit", "-m", f"portchk: {ref} onto {base[:10]}", ref, cwd=wt, check=False, env=ENV)
    return p.returncode, conflicts(wt), (p.stdout + p.stderr)[-6000:]


MARK = re.compile(r"^<<<<<<< [^\n]*\n(.*?)^(?:\|\|\|\|\|\|\| [^\n]*\n(.*?))?^=======\n(.*?)^>>>>>>> [^\n]*\n", re.S | re.M)


def theirs_hunks(text: str) -> tuple[str, int]:
    """``text`` with each conflict block replaced by its theirs side; the number of blocks."""
    n = 0

    def take(m):
        nonlocal n
        n += 1
        return m.group(3)
    return MARK.sub(take, text), n


def resolve(wt: Path, conf: dict[str, str]) -> dict[str, str]:
    """Each conflict resolved mechanically for the build step; {path: how}."""
    how = {}
    for path, xy in conf.items():
        k = kind(path, xy)
        if xy in ("UD", "DU", "DD"):
            git("rm", "-q", "--cached", "--ignore-unmatch", "--", path, cwd=wt, check=False)
            (wt / path).unlink(missing_ok=True)
            how[path] = "deleted (the deletion stands)"
        elif xy in ("AU", "UA"):
            git("add", "--", path, cwd=wt, check=False)
            how[path] = "the one side that has it"
        elif k == "lean":
            text, n = theirs_hunks((wt / path).read_text())
            if "<<<<<<<" in text or n == 0:
                git("checkout", "--theirs", "--", path, cwd=wt, check=False)
                how[path] = "theirs (whole file)"
            else:
                (wt / path).write_text(text)
                how[path] = f"theirs in {n} hunk(s)"
            git("add", "--", path, cwd=wt)
        elif k in ("lock", "lake"):
            git("checkout", "--theirs", "--", path, cwd=wt, check=False)
            git("add", "--", path, cwd=wt)
            how[path] = "theirs"
        else:
            git("checkout", "--ours", "--", path, cwd=wt, check=False)
            git("add", "--", path, cwd=wt)
            how[path] = "ours (the port's)"
    return how


def run_codemod(wt: Path, *args) -> tuple[int, str, str]:
    p = subprocess.run(["python3", CODEMOD, *args], cwd=wt, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def commit(wt: Path, msg: str) -> str:
    git("add", "-A", cwd=wt)
    git("commit", "-q", "--no-verify", "--allow-empty", "-m", msg, cwd=wt, env=ENV)
    return git("rev-parse", "HEAD", cwd=wt).stdout.strip()


def one(n: int, wt: Path, wt_main: Path) -> dict:
    ref = f"refs/portchk/pr/{n}"
    head = git("rev-parse", ref).stdout.strip()
    main = git("rev-parse", "origin/main").stdout.strip()
    mb = git("merge-base", head, main).stdout.strip()
    ns = git("diff", "--name-status", "--no-renames", mb, head).stdout.splitlines()
    files = {l.split("\t", 1)[1]: l.split("\t", 1)[0] for l in ns}
    lean = sorted(f for f, s in files.items() if f.endswith(".lean") and s != "D")
    rec = {"pr": n, "head": head, "main_base": mb, "main_base_date": git("log", "-1", "--format=%cI", mb).stdout.strip(),
           "port_base_contains_main_base": git("merge-base", "--is-ancestor", mb, PORT, check=False).returncode == 0,
           "files": len(files), "lean": lean,
           "lean_deleted": sorted(f for f, s in files.items() if f.endswith(".lean") and s == "D"),
           "lean_added": sorted(f for f in lean if files[f] == "A"),
           "locks": sorted(f for f in files if f.endswith("lean-audit.json")),
           "lakefiles": sorted(f for f in files if LAKE.search(f))}
    t0 = time.time()
    _, conf_m, _ = merge(wt_main, main, ref)
    rec["main_conflicts"] = conf_m
    reset(wt_main, main)
    rc, conf, text = merge(wt, PORT, ref)
    rec["merge_rc"], rec["conflicts"], rec["merge_log"] = rc, conf, text
    rec["conflict_kinds"] = {p: kind(p, xy) for p, xy in conf.items()}
    rec["refused_by_drivers"] = [l for l in text.splitlines() if l.startswith("merge.py:")]
    if conf:
        rec["resolution"] = resolve(wt, conf)
    pre_rc, pre_out, _ = run_codemod(wt, "--check")
    rec["unported_before"] = [l.split(":", 1)[0] for l in pre_out.splitlines() if l.strip()]
    c_rc, c_out, c_err = run_codemod(wt)
    rec["codemod_rc"], rec["codemod_out"], rec["codemod_err"] = c_rc, c_out.strip()[-3000:], c_err.strip()[-6000:]
    post_rc, post_out, post_err = run_codemod(wt, "--check")
    rec["check_rc"], rec["unported_after"] = post_rc, [l for l in (post_out + post_err).splitlines() if l.strip()]
    rec["codemod_changed"] = sorted(set(git("diff", "--name-only", cwd=wt).stdout.split())
                                    | set(git("ls-files", "--others", "--exclude-standard", cwd=wt).stdout.split()))
    sha = commit(wt, f"portchk: #{n} onto the port, {'resolved, ' if conf else ''}converted")
    git("update-ref", f"refs/portchk/{'resolved' if conf else 'merged'}/{n}", sha)
    rec["tree_commit"], rec["tree_kind"] = sha, ("resolved" if conf else "merged")
    rec["seconds"] = round(time.time() - t0, 1)
    reset(wt, PORT)
    return rec


def worker(i: int, prs: list[int]):
    wt, wt_main = Path(f"/tmp/portchk/wt{i}"), Path(f"/tmp/portchk/wt-main{i}")
    ensure(wt, PORT)
    ensure(wt_main, "origin/main")
    for n in prs:
        if (OUT / f"{n}.json").exists():
            continue
        try:
            rec = one(n, wt, wt_main)
        except Exception as e:  # recorded; the loop goes on
            rec = {"pr": n, "error": str(e)[-3000:]}
            try:
                reset(wt, PORT)
            except Exception:
                pass
        (OUT / f"{n}.json").write_text(json.dumps(rec, indent=1))
        print(n, rec.get("seconds"), "conf", len(rec.get("conflicts", {})), "main_conf", len(rec.get("main_conflicts", {})),
              "lean", len(rec.get("lean", [])), "after", len(rec.get("unported_after", [])), rec.get("error", "")[:200],
              flush=True)


def main(argv):
    import os
    os.environ.update({"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "gc.auto", "GIT_CONFIG_VALUE_0": "0"})
    ENV.update({"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "gc.auto", "GIT_CONFIG_VALUE_0": "0"})
    k = 1
    if argv[:1] == ["--workers"]:
        k, argv = int(argv[1]), argv[2:]
    OUT.mkdir(parents=True, exist_ok=True)
    prs = [int(a) for a in argv] or [p["number"] for p in json.load(open("/tmp/portchk_prs.json"))]
    prs = [n for n in prs if not (OUT / f"{n}.json").exists()]
    chunks = [prs[i::k] for i in range(k)]
    procs = [mp.Process(target=worker, args=(i, c)) for i, c in enumerate(chunks)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


if __name__ == "__main__":
    main(sys.argv[1:])
