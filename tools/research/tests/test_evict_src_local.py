"""`_local_changes`: a tree goes only when deleting it loses nothing the node's repo doesn't hold.  The first seven tests are the
red team's recheck of 437c8cf32, each of which deleted a tree with work in it; the rest check that
what a ship and a test run leave behind (READY.json, build caches, fetched fixtures) doesn't pin a tree forever."""
import json

from research.store import evict as EV
from test_evict_src import NOW, _age, _git, fake_proc, kept, node, run  # noqa: F401  (node is a fixture)


def deleted(rep):
    return [d["tree"] for d in rep["deleted"]]


# ---- HIGH: ignored files are invisible to `git status` -------------------------------------------------------------------------

def test_HIGH_output_written_under_a_gitignored_dir_keeps_the_tree(node):
    """`_local_changes` runs `git status` without `--ignored`.  The repository ignores out/, runs/, **/results/, **/outputs/,
    **/reports/, *.png, *.pdf, *.tar.gz, results_*.json...: what a `--cwd source` workload writing relative paths produces.  On
    vy-nebius-1 today the dry run of 437c8cf32 deletes 4379188e78bb (out/config.json, out/stderr.txt, run r20261004-152806-b2d5) and
    827185029243 (out/compare.txt, out/sha256.txt, run r20261004-154231-ab77), whose run dirs have no out/: the only copy."""
    from test_evict_src import tree
    work, gd = node.tmp / "work", node.root / "git" / "verity.git"
    _git(work, "checkout", "-q", "main")
    (work / ".gitignore").write_text("out/\n")
    _git(work, "add", ".gitignore")
    _git(work, "commit", "-q", "-m", "ignore out/ as the repository does")
    d = _git(work, "rev-parse", "HEAD")
    _git(work, "push", "-q", str(gd), f"{d}:refs/research/src/{d}")
    t = tree(node.src, d, 5)
    (t / "out").mkdir()
    (t / "out" / "compare.txt").write_text("the only copy of a run's result")
    _age(t, 5)
    rep = run(node, origin=None)
    assert d not in deleted(rep), "an ignored out/ (a workload's output) was deleted with the tree"


# ---- MED: work held in the tree's own .git ------------------------------------------------------------------------------------

def test_MED_a_local_commit_held_only_by_the_trees_own_refs_keeps_it(node):
    """A shared clone writes new objects into its own .git/objects, not the node repo.  Commit an in-place fix on a branch, check the
    shipped commit out again: HEAD is the sha and status is clean, and the fix lives only in this tree."""
    t = node.src / node.a
    _git(t, "checkout", "-q", "-b", "fix")
    (t / "f").write_text("the fix")
    _git(t, "commit", "-q", "-am", "fix in place")
    _git(t, "-c", "advice.detachedHead=false", "checkout", "-q", "--detach", node.a)
    _age(t, 5)
    rep = run(node, origin=None)
    assert node.a not in deleted(rep), "a commit only the tree's own refs hold was deleted"


def test_MED_a_stash_in_the_tree_keeps_it(node):
    t = node.src / node.a
    (t / "f").write_text("stashed work")
    _git(t, "stash", "-q")
    _age(t, 5)
    assert node.a not in deleted(run(node, origin=None)), "a stash (refs/stash in the tree's .git) was deleted"


def test_MED_a_modification_hidden_by_the_index_s_assume_unchanged_or_skip_worktree_bit_keeps_the_tree(node):
    """`git update-index --assume-unchanged` / `--skip-worktree` (sparse checkouts set the latter) make status skip a file."""
    t = node.src / node.a
    _git(t, "update-index", "--skip-worktree", "f")
    (t / "f").write_text("edited; the index says not to look")
    _age(t, 5)
    assert node.a not in deleted(run(node, origin=None)), "a skip-worktree file's edit was deleted"


# ---- LOW: the tree's .git config is trusted --------------------------------------------------------------------------------------

def test_LOW_an_exclude_the_tree_s_own_git_adds_hides_nothing(node):
    """.git/info/exclude (or core.excludesFile in .git/config, or the user's ~/.config/git/ignore) is read by the status."""
    t = node.src / node.a
    (t / ".git" / "info").mkdir(exist_ok=True)
    with open(t / ".git" / "info" / "exclude", "a") as f:
        f.write("*\n")
    (t / "notes.md").write_text("work")
    _age(t, 5)
    assert node.a not in deleted(run(node, origin=None)), "an untracked file excluded by the tree's own .git was deleted"


def test_LOW_the_status_runs_no_command_from_the_tree_s_git_config(node):
    """core.fsmonitor (and filter.<x>.clean with .gitattributes) in a tree's .git/config is a command `git status` runs: here with
    the unit's CAP_SYS_PTRACE and CAP_DAC_READ_SEARCH.  A hook that answers can also tell status which files to skip."""
    t = node.src / node.a
    marker = node.tmp / "ran"
    _git(t, "config", "core.fsmonitor", f"touch {marker}; false")
    _age(t, 5)
    run(node, origin=None)
    assert not marker.exists(), "git status ran the tree's core.fsmonitor command"


# ---- LOW: /proc spellings the loose match still misses ----------------------------------------------------------------------------

def test_LOW_a_sibling_tree_named_relatively_from_another_tree_or_from_src_keeps_it(node):
    """_LOOSE needs the literal `src/` before the sha.  `diff -r . ../<sha>` from another tree, or `python3 <sha>/x.py` from <src>,
    names it without one."""
    fake_proc(node.proc, 800, cwd=f"{node.src}/{node.c}", cmdline=f"diff -r . ../{node.a}")
    fake_proc(node.proc, 801, cwd=str(node.src), cmdline=f"python3 {node.b}/tools/verity/check/check.py")
    rep = run(node, origin=None)
    assert node.a not in deleted(rep) and node.b not in deleted(rep), deleted(rep)


# ---- what a ship and a test run leave still lets a tree go -------------------------------------------------------------------

def _ignoring(node, ignore: str, artifacts: dict | None = None) -> str:
    from test_evict_src import tree
    work, gd = node.tmp / "work", node.root / "git" / "verity.git"
    _git(work, "checkout", "-q", "main")
    (work / ".gitignore").write_text(ignore)
    paths = [".gitignore"]
    if artifacts is not None:
        (work / "fixtures").mkdir(exist_ok=True)
        (work / "fixtures" / "artifacts.json").write_text(json.dumps(artifacts))
        paths.append("fixtures/artifacts.json")
    _git(work, "add", *paths)
    _git(work, "commit", "-q", "-m", "ignores")
    d = _git(work, "rev-parse", "HEAD")
    _git(work, "push", "-q", str(gd), f"{d}:refs/research/src/{d}")
    t = tree(node.src, d, 5)
    with open(t / ".git" / "info" / "exclude", "a") as f:
        f.write(f"/{EV.READY_FILE}\n")
    return d


def test_build_caches_and_fetched_fixtures_under_ignored_paths_let_the_tree_go(node):
    d = _ignoring(node, ".venv/\n.lake/\n__pycache__/\n.hypothesis/\nbig/\nfx/\n",
                  {"art:" + "1" * 64: "big/data.npz", "art:" + "2" * 64: "fx/tree", "previous": {"art:" + "1" * 64: ["art:" + "3" * 64]}})
    t = node.src / d
    for p in (".venv/bin/python", ".lake/build/x.olean", "pkg/__pycache__/m.cpython-312.pyc", "pkg/.hypothesis/.gitignore", "big/data.npz", "fx/tree/a/b.bin"):
        (t / p).parent.mkdir(parents=True, exist_ok=True)
        (t / p).write_text("regenerable")
    _age(t, 5)
    assert d in deleted(run(node, origin=None))


def test_an_ignored_directory_holding_a_fixture_and_something_else_keeps_the_tree(node):
    d = _ignoring(node, "big/\n", {"art:" + "1" * 64: "big/data.npz"})
    t = node.src / d
    (t / "big").mkdir()
    (t / "big" / "data.npz").write_text("fixture")
    (t / "big" / "result.json").write_text("a run's output")
    _age(t, 5)
    rep = run(node, origin=None)
    assert d not in deleted(rep) and any("ignored path" in w for w in kept(rep)[d]), kept(rep)[d]


def test_a_ship_s_own_exclude_and_config_let_the_tree_go(node):
    """A ship's clone: branch.<default>.* keys whatever the node repo's default branch, and /READY.json in .git/info/exclude."""
    t = node.src / node.a
    _git(t, "config", "branch.master.remote", "origin")
    _git(t, "config", "branch.master.merge", "refs/heads/master")
    (t / ".git" / "info").mkdir(exist_ok=True)
    (t / ".git" / "info" / "exclude").write_text("# git ls-files --others --exclude-from=.git/info/exclude\n/READY.json\n")
    _age(t, 5)
    assert node.a in deleted(run(node, origin=None))


def test_a_lock_another_user_made_stops_the_pass_with_a_reason(node, monkeypatch):
    import builtins
    real = builtins.open

    def refusing(f, *a, **kw):
        if str(f).endswith(EV.LOCK):
            raise PermissionError(13, "Permission denied", str(f))
        return real(f, *a, **kw)

    monkeypatch.setattr(builtins, "open", refusing)
    rep = run(node, origin=None)
    assert rep["deleted"] == [] and rep["error"].startswith(f"cannot open {node.src / EV.LOCK}: PermissionError")
