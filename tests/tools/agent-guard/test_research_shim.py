"""`research.sh`, /usr/local/bin/research on an agent VM, runs origin/main's `at_main.py` as a script, which runs
origin/main's research for a checkout whose research code is only behind it."""
import os
import subprocess
from pathlib import Path

GUARD = (Path(__file__).resolve().parents[3] / "tools" / "agent-guard")
AT_MAIN = GUARD.parents[1] / "tools/research/src/research/at_main.py"
RESEARCH = "tools/research/src/research"
ENV = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
       "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
#: a research package's `__main__`, which says whose copy it is and what it was asked
WHOSE = "import sys\nprint({!r}, *sys.argv[1:])\n"


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          env={**os.environ, **ENV}).stdout.strip()


def commit(repo: Path, files: dict[str, str]) -> str:
    for path, text in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)
    git(repo, "add", *files)
    git(repo, "commit", "-q", "-m", "research")
    return git(repo, "rev-parse", "HEAD")


def test_the_shim_runs_origin_mains_research_for_a_checkout_only_behind_it(tmp_path):
    """The checkout's branch holds no `at_main.py`, as a checkout from before it doesn't: only origin/main's, which the
    shim runs with no package, decides; one that needs `research` as its package fails every command."""
    repo = tmp_path / "repo"
    git(tmp_path, "init", "-q", "-b", "main", str(repo))
    commit(repo, {f"{RESEARCH}/__init__.py": "", f"{RESEARCH}/cli.py": "", f"{RESEARCH}/__main__.py": WHOSE.format("old")})
    git(repo, "checkout", "-q", "-b", "lane")
    git(repo, "checkout", "-q", "main")
    main = commit(repo, {f"{RESEARCH}/at_main.py": AT_MAIN.read_text(), f"{RESEARCH}/__main__.py": WHOSE.format("main's")})
    git(repo, "update-ref", "refs/remotes/origin/main", main)
    git(repo, "checkout", "-q", "lane")
    env = {k: v for k, v in os.environ.items() if k not in ("RESEARCH_AT_CHECKOUT", "RESEARCH_CHECKOUT")}
    done = subprocess.run(["sh", str(GUARD / "research.sh"), "--version"], cwd=repo, capture_output=True, text=True,
                          env={**env, **ENV, "XDG_CACHE_HOME": str(tmp_path / "cache")})
    assert done.returncode == 0 and "Traceback" not in done.stderr, done.stderr
    assert done.stdout == "main's --version\n"
    assert done.stderr.startswith("research: this checkout's research code is 1 commit(s) behind origin/main")
