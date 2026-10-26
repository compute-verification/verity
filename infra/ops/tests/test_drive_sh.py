"""infra/control/drive.sh's checkout: `drive.sh merge` runs `research merge` on branch main, which it refuses on a detached HEAD
(CI-169, 25 Oct: every `drive.sh merge` was refused), while the tick and `drive.sh queue` keep a detached checkout of origin's main."""
import os
import re
import subprocess
from pathlib import Path

import pytest

DRIVE = Path(__file__).resolve().parents[3] / "infra" / "control" / "drive.sh"


def git(*args, cwd):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    origin, clone = tmp_path / "origin", tmp_path / "verity"
    origin.mkdir()
    git("init", "-q", "-b", "main", cwd=origin)
    git("commit", "-q", "--allow-empty", "-m", "first", cwd=origin)
    git("clone", "-q", str(origin), str(clone), cwd=tmp_path)
    git("checkout", "-q", "--detach", cwd=clone)                       # where the last tick left it
    git("commit", "-q", "--allow-empty", "-m", "second", cwd=origin)
    return clone, git("rev-parse", "HEAD", cwd=origin)


def function(name):
    fn = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", DRIVE.read_text(), re.M | re.S)
    assert fn, f"drive.sh defines {name}() {{ ... }}"
    return fn.group(0)


def checkout(repo, verb):
    """`checkout` with the clone already on the fast disk (`fast_clones` finds it and copies nothing)."""
    tmp = Path(repo).parent
    env = {**os.environ, "REPO": str(repo), "FAST": str(tmp / "fast"), "QUEUE": str(tmp / "queue"), "DIR": str(tmp / "volume"),
           "RESEARCH_STORE": str(tmp / "store")}
    fns = "".join(function(f) for f in ("fast_clones", "store_links", "checkout"))
    subprocess.run(["sh", "-c", f'{fns}checkout "$1"', "sh", verb], env=env, check=True)


def test_store_links_puts_the_stores_bulk_on_the_volume_and_keeps_what_it_held(tmp_path):
    """objects/ and attempts/ become links into the volume's copy (they took the pod's root to 94%, 26 Oct); a real directory
    already there is merged in first, so nothing it held is lost, and the catalog stays where it is."""
    store, vol = tmp_path / "store", tmp_path / "volume"
    (store / "objects" / "sha256" / "ab").mkdir(parents=True)
    (store / "objects" / "sha256" / "ab" / "abcd").write_text("blob")
    (store / "catalog.sqlite").write_text("catalog")
    env = {**os.environ, "DIR": str(vol), "RESEARCH_STORE": str(store)}
    for _ in range(2):
        subprocess.run(["sh", "-c", f'{function("store_links")}store_links'], env=env, check=True)
    for d in ("objects", "attempts"):
        assert (store / d).is_symlink() and (store / d).resolve() == (vol / "store" / d).resolve()
        assert not (store / f"{d}.old").exists()
    assert (store / "objects" / "sha256" / "ab" / "abcd").read_text() == "blob"
    assert (store / "catalog.sqlite").read_text() == "catalog" and not (store / "catalog.sqlite").is_symlink()


def test_fast_clones_seeds_the_container_disk_from_the_volume_once(tmp_path):
    """The volume's clones are copied to the container disk (`/root/fast`) once; the queue's path becomes a link to its copy
    and the volume's queue clone stays beside it (`.volume`)."""
    vol, fast, queue = tmp_path / "volume", tmp_path / "fast", tmp_path / "queue"
    for d in (vol / "verity", queue):
        d.mkdir(parents=True)
        git("init", "-q", "-b", "main", cwd=d)
    env = {**os.environ, "DIR": str(vol), "FAST": str(fast), "REPO": str(fast / "verity"), "QUEUE": str(queue)}
    for _ in range(2):
        subprocess.run(["sh", "-c", f'{function("fast_clones")}fast_clones'], env=env, check=True)
    assert (fast / "verity" / ".git").is_dir() and (fast / "queue-verity" / ".git").is_dir()
    assert queue.is_symlink() and queue.resolve() == (fast / "queue-verity").resolve()
    assert (tmp_path / "queue.volume" / ".git").is_dir()


def test_locked_calls_checkout_with_its_verb():
    body = re.search(r"^locked\(\) \{\n(.*?)^\}\n", DRIVE.read_text(), re.M | re.S).group(1)
    assert 'checkout "$1" || exit 1' in body and "--detach" not in body


def test_merge_checks_out_branch_main_at_origin(repo):
    clone, tip = repo
    checkout(clone, "merge")
    assert git("symbolic-ref", "--quiet", "--short", "HEAD", cwd=clone) == "main"  # research merge's _admissible reads this
    assert git("rev-parse", "HEAD", cwd=clone) == tip


def test_merge_resets_a_local_main_that_drifted(repo):
    clone, tip = repo
    git("checkout", "-q", "-B", "main", cwd=clone)
    git("commit", "-q", "--allow-empty", "-m", "an unpushed merge", cwd=clone)
    checkout(clone, "merge")
    assert git("symbolic-ref", "--quiet", "--short", "HEAD", cwd=clone) == "main"
    assert git("rev-parse", "HEAD", cwd=clone) == tip


def test_queue_and_the_tick_stay_detached(repo):
    clone, tip = repo
    checkout(clone, "queue")                                            # the tick is `locked queue drive ...`
    assert subprocess.run(["git", "symbolic-ref", "-q", "HEAD"], cwd=clone).returncode == 1
    assert git("rev-parse", "HEAD", cwd=clone) == tip
