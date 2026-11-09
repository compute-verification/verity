"""infra/control/drive.sh's checkout: `drive.sh merge` runs `research merge` in the queue's clone on branch main, which it refuses
on a detached HEAD (else every `drive.sh merge` is refused), while the tick and `drive.sh queue` keep a detached checkout of
origin's main."""
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
    """$REPO and the queue's clone, both behind origin's main: $REPO detached and the queue's clone on `checked`, where the last
    tick left them."""
    origin, clone, queue = tmp_path / "origin", tmp_path / "verity", tmp_path / "fast" / "queue-verity"
    origin.mkdir()
    git("init", "-q", "-b", "main", cwd=origin)
    git("commit", "-q", "--allow-empty", "-m", "first", cwd=origin)
    git("clone", "-q", str(origin), str(clone), cwd=tmp_path)
    git("checkout", "-q", "--detach", cwd=clone)
    git("clone", "-q", str(origin), str(queue), cwd=tmp_path)
    git("checkout", "-q", "-b", "checked", cwd=queue)
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
    """objects/ and attempts/ become links into the volume's copy (they can fill the pod's root); a real directory
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


def test_start_seeds_the_clones_before_its_push_check():
    """On a new container `$REPO` exists only once `fast_clones` has run, so a `start` that checks its dry-run push first is
    always refused there."""
    start = re.search(r"^  start\) (.*?);;\n", DRIVE.read_text(), re.M | re.S).group(1)
    assert 'flock 8; fast_clones' in start
    assert start.index("fast_clones") < start.index("gh_app_token.sh")


def test_research_writes_unbuffered_so_a_running_tick_shows_in_the_log():
    """drive's log is a pipe, where Python would hold a tick's lines until it ended."""
    export = re.search(r"^export PATH=.*?(?<!\\)\n", DRIVE.read_text(), re.M | re.S).group(0)
    assert "PYTHONUNBUFFERED=1" in export.split()


def test_locked_calls_checkout_with_its_verb_and_runs_merge_in_the_queues_clone():
    """`research merge` runs in the queue's clone, a full one: in $REPO, a blob:none partial clone, its sweep fetched each open
    PR's head it lacked, a pack each, until the pod's disk filled."""
    body = re.search(r"^locked\(\) \{\n(.*?)^\}\n", DRIVE.read_text(), re.M | re.S).group(1)
    assert 'checkout "$1" || exit 1' in body and "--detach" not in body
    assert 'if [ "$1" = merge ]; then cd "$FAST/queue-verity"; else cd "$REPO"; fi || exit 1' in body


@pytest.mark.parametrize("args, runs", [(("queue", "promote"), "unlocked"), (("queue", "baseline", "abc", "--by", "ci"), "unlocked"),
                                        (("queue", "eject", "5"), "locked"), (("merge", "--push"), "locked")])
def test_queue_baseline_and_promote_run_beside_a_tick(args, runs):
    """`queue baseline` and `queue promote` only fast-forward main and write nothing of drive's, so they wait for no tick (one
    holds its lock an hour and more, and main would wait with them) and check nothing out under one; every other verb does."""
    arms = "".join(re.search(rf"^  {verb}\) .*;;\n", DRIVE.read_text(), re.M).group(0) for verb in ("queue", "merge"))
    stubs = 'locked() { echo locked "$@"; }; unlocked() { echo unlocked "$@"; }; '
    out = subprocess.run(["sh", "-c", f"{stubs}case $1 in\n{arms}esac", "sh", *args], capture_output=True, text=True).stdout.strip()
    assert out == f"{runs} {' '.join(args)}"
    body = function("unlocked")
    assert "flock" not in body and "checkout" not in body and "RESEARCH_DRIVE=1" in body


def test_merge_checks_out_branch_main_at_origin_in_both_clones(repo):
    """The queue's clone, where `research merge` runs, and $REPO, whose code it runs, are both on branch main at origin's tip;
    `research merge`'s `_admissible` reads the branch."""
    clone, tip = repo
    checkout(clone, "merge")
    for c in (clone, Path(clone).parent / "fast" / "queue-verity"):
        assert git("symbolic-ref", "--quiet", "--short", "HEAD", cwd=c) == "main"
        assert git("rev-parse", "HEAD", cwd=c) == tip


def test_prune_trees_folds_the_code_clones_packs_into_one_once_they_pass_50(tmp_path):
    """Each checkout in $REPO, a partial clone, fetches the blobs it lacks as a pack of its own, and git's auto maintenance
    never repacks a partial clone's such packs (here auto gc is off): at 50 packs `prune_trees` leaves them, past 50 it folds
    them into one."""
    repo = tmp_path / "verity"
    repo.mkdir()
    (tmp_path / "trains").mkdir()
    git("init", "-q", "-b", "main", cwd=repo)
    git("config", "gc.auto", "0", cwd=repo)
    env = {**os.environ, "REPO": str(repo), "FAST": str(tmp_path / "fast"), "TRAIN_TREES": str(tmp_path / "trains")}

    def packs() -> int:
        return len(list((repo / ".git" / "objects" / "pack").glob("*.pack")))

    def prune() -> None:
        subprocess.run(["sh", "-c", f'{function("prune_trees")}prune_trees'], env=env, check=True)

    for n in range(1, 52):
        (repo / "f").write_text(f"{n}\n")
        git("add", "f", cwd=repo)
        git("commit", "-q", "-m", str(n), cwd=repo)
        git("repack", "-q", cwd=repo)
        if n == 50:
            prune()
            assert packs() == 50
    assert packs() == 51
    prune()
    assert packs() == 1 and git("rev-list", "--count", "HEAD", cwd=repo) == "51"


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


def deploy(repo, arg):
    tmp = Path(repo).parent
    env = {**os.environ, "REPO": str(repo), "DIR": str(tmp / "volume")}
    return subprocess.run(["sh", "-c", f'{function("deploy")}deploy "$1"', "sh", arg], env=env, capture_output=True, text=True)


def test_the_tick_runs_the_deployed_commit_until_deploy_main_and_merge_stays_on_main(repo):
    """A landing of drive's own code changes what drive runs only once it is deployed: the tick and `queue` check out the pin,
    `merge` still lands on main's tip, and `deploy main` follows main again."""
    clone, tip = repo
    origin = Path(clone).parent / "origin"
    first = git("rev-parse", "HEAD~1", cwd=origin)
    (Path(clone).parent / "volume").mkdir()
    assert deploy(clone, first).returncode == 0
    checkout(clone, "queue")
    assert git("rev-parse", "HEAD", cwd=clone) == first
    checkout(clone, "merge")
    assert git("rev-parse", "HEAD", cwd=clone) == tip
    assert deploy(clone, "main").returncode == 0
    checkout(clone, "queue")
    assert git("rev-parse", "HEAD", cwd=clone) == tip


def test_deploy_takes_only_landed_code_and_a_pin_that_names_no_commit_fails_the_tick(repo):
    """Drive pushes `next`, `checked` and `main`, so it runs only a commit `checked` or `main` holds; a pin file that names no
    commit fails the tick rather than run main's tip unannounced."""
    clone, _ = repo
    origin = Path(clone).parent / "origin"
    vol = Path(clone).parent / "volume"
    vol.mkdir()
    git("checkout", "-q", "-b", "unlanded", cwd=origin)
    git("commit", "-q", "--allow-empty", "-m", "on no branch drive lands", cwd=origin)
    other = git("rev-parse", "HEAD", cwd=origin)
    git("checkout", "-q", "main", cwd=origin)
    git("fetch", "-q", "origin", "unlanded", cwd=clone)
    out = deploy(clone, other)
    assert out.returncode == 1 and "on neither checked nor main" in out.stderr and not (vol / "deployed").exists()
    (vol / "deployed").write_text("not a commit\n")
    with pytest.raises(subprocess.CalledProcessError):
        checkout(clone, "queue")
