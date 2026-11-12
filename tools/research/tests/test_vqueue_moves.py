"""A `Move:` PR in the verified queue: `decide` builds it last in its batch, one a batch, with nothing above it and nothing
below or beside it that changes `tools/move/`'s scripts; the loop regenerates it on its batch's tip with `main`'s
`rename.py` and `move_check.py`, given its map's path rows as `--map`, and a certified one lands on its ready mark alone, its
certificate in the build line, while an uncertified one needs a land verdict.  The simulated queue holds `decide` to those
rules over every seed; the loop's tests run its build against a throwaway origin whose `main` holds stub scripts."""
from __future__ import annotations

import json
import random
import tomllib
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import research.queue as Q
from research.queue import path_rows
from research.vqueue import PASS, Advance, Build, Commit, Drop, Ready, RecordsRun, Snapshot, _moves_last, decide
from test_vqueue import MAIN, SEEDS, World, cand, lean, pr, snap, ticks, waiting
from research.executor import Executor
from test_vqueue_loop import Env, Fake, _acts, _git, _main_to, _move, env, policy  # noqa: F401 -- env is a fixture

MAP = "tools/move/x_map.toml"
#: the map `Moved` puts on `main`: `old/` moves to `new/`, and the word `oldword` becomes `newword`
MAP_ROWS = '[words]\n"old/" = "new/"\n"oldword" = "newword"\n'


def mv(n: int, **kw) -> Ready:
    return replace(pr(n, **kw), move=True)


def sc(n: int, **kw) -> Ready:
    return replace(pr(n, f"h{n}", "tools/move/rename.py", **kw), scripts=True)


def x(n: int, *moves: str, **kw) -> Ready:
    """PR ``n``, whose own change names nothing the maps of the move PRs at heads ``moves`` retire (`Ready.crosses`)."""
    return replace(pr(n, **kw), crosses=frozenset(moves))


# ---- decide -------------------------------------------------------------------------------------------------------------------


def test_a_move_goes_last_in_its_batch_and_a_second_move_waits_for_another_batch():
    ready = {r.number: r for r in (mv(1), pr(2), mv(3), pr(4))}
    assert _moves_last((1, 2, 3, 4), ready) == (2, 4, 1)
    assert _moves_last((2, 4), ready) == (2, 4)
    (b,) = [a for a in decide(snap(ready=ready.values(), batch=4)) if isinstance(a, Build)]
    assert b.members == (2, 4, 1), b


def test_a_move_is_left_out_of_a_batch_holding_a_pr_that_changes_the_move_scripts():
    ready = {r.number: r for r in (mv(1), sc(2), pr(3))}
    assert _moves_last((1, 2, 3), ready) == (2, 3)
    (b,) = [a for a in decide(snap(ready=ready.values(), batch=4)) if isinstance(a, Build)]
    assert b.members == (2, 3), b


def test_nothing_that_doesnt_cross_a_move_in_flight_is_built_above_it():
    s = snap(cand("q1", MAIN, 1), ready=[pr(2)], flying=(mv(1),), depth=3)
    assert not [a for a in decide(s) if isinstance(a, Build)]
    s = snap(cand("q1", MAIN, 1), ready=[pr(2)], flying=(pr(1),), depth=3)
    assert [a.members for a in decide(s) if isinstance(a, Build)] == [(2,)]


def test_a_move_isnt_built_above_a_pr_in_flight_that_changes_the_move_scripts_and_others_are():
    s = snap(cand("q1", MAIN, 1), ready=[mv(2), pr(3)], flying=(sc(1),), depth=3, batch=4)
    assert [a.members for a in decide(s) if isinstance(a, Build)] == [(3,)]
    s = snap(cand("q1", MAIN, 1), ready=[mv(2)], flying=(sc(1),), depth=3)
    assert not [a for a in decide(s) if isinstance(a, Build)]


def test_nothing_that_doesnt_cross_a_move_is_built_in_the_tick_that_advances_main_to_it():
    s = snap(cand("q1", MAIN, 1, state=PASS), ready=[mv(1), pr(2)], depth=3)
    acts = decide(s)
    assert Advance("q1") in acts and not [a for a in acts if isinstance(a, Build)], acts
    s = snap(cand("q1", MAIN, 1, state=PASS), ready=[pr(1), pr(2)], depth=3)
    assert [a.members for a in decide(s) if isinstance(a, Build)] == [(2,)]
    s = snap(cand("q1", MAIN, 1, state=PASS), cand("q2", "q1", 2, state=PASS), ready=[pr(1), mv(2), pr(3)], depth=3)
    acts = decide(s)
    assert Advance("q1") in acts and Advance("q2") in acts and not [a for a in acts if isinstance(a, Build)], acts


def test_a_move_dropped_for_a_leaver_below_it_is_built_again_last_and_nothing_that_doesnt_cross_it_on_it():
    left = ((1, "h1", "its ready mark was withdrawn by lane-1"),)
    for top, above in ((mv(3), []), (pr(3), [Build((4,), "q2", 1)])):
        s = snap(cand("q1", MAIN, 1, 2), cand("q2", "q1", 3), ready=[pr(2), top, pr(4)], left=left, pulled=frozenset({1}),
                 depth=3)
        assert [a for a in decide(s) if isinstance(a, Build)] == [Build((2,), MAIN, 3, replaces="q1"),
                                                                  Build((3,), "q1", 2, replaces="q2"), *above]


def test_a_staged_batch_rebuilt_on_the_chain_holding_a_move_has_nothing_that_doesnt_cross_it_built_on_it():
    """A passed staged batch whose Lean member isn't marked at the head it was staged at is built again without it on the
    chain's top; holding a move, nothing that doesn't cross it is built on that rebuild, as on any candidate holding one."""
    passed = waiting("s1", MAIN, 1, 2, records=RecordsRun(PASS))
    for two, above in ((mv(2), []), (pr(2), [Build((1,), "s1", 1, records=True), Build((4,), "s1", 1)])):
        s = snap(passed, cand("c2", MAIN, 3), ready=[pr(1, "h1b", lean=True), two, pr(3), pr(4)], records=True, depth=3)
        assert [a for a in decide(s) if isinstance(a, Build)] == [Build((2,), "c2", 2, replaces="s1"), *above]


def _builds_of(acts: list) -> list[tuple[tuple[int, ...], str]]:
    return [(a.members, a.base) for a in acts if isinstance(a, Build)]


def test_a_batch_above_a_move_in_flight_takes_only_the_prs_that_cross_it_at_its_head_as_built_and_no_second_move():
    """Above a candidate holding a move, ready or in flight, a batch takes the PRs whose own change names nothing the move's
    map retires at the head it was built at (`Ready.crosses`), and no move, whatever it crosses; the rest wait for it to
    land, higher up the chain too."""
    for flying in ((mv(1),), ()):
        ready = [*(() if flying else (mv(1),)), x(2, "h1"), pr(3), x(4, "h1x"), replace(mv(5), crosses=frozenset({"h1"})),
                 x(6, "h1", "h9")]
        s = snap(cand("q1", MAIN, 1), ready=ready, flying=flying, depth=3, batch=6)
        assert _builds_of(decide(s)) == [((2, 6), "q1")], flying
    s = snap(cand("q1", MAIN, 1), cand("q2", "q1", 2), ready=[mv(1), x(2, "h1"), x(3, "h1"), pr(4)], depth=3)
    assert _builds_of(decide(s)) == [((3,), "q2")]


def test_the_tick_that_advances_main_to_a_move_builds_on_it_only_the_prs_that_cross_it():
    """The ready PRs were read on the `main` below the move, so those its map can't clear wait a tick for `ready` to read them
    on the moved `main`; the ones it clears were cleared against it, and are built on it at once."""
    s = snap(cand("q1", MAIN, 1, state=PASS), ready=[mv(1), x(2, "h1"), pr(3)], depth=3)
    acts = decide(s)
    assert Advance("q1") in acts and _builds_of(acts) == [((2,), "q1")], acts


def test_a_batch_above_a_move_is_built_again_on_what_is_left_when_the_move_s_candidate_drops():
    """A leaver below the move drops its candidate and the batch above it: the same tick builds the move again on what is
    left and the batch above again on that, its members crossing the move, while a PR that doesn't cross it waits; with the
    move itself taken out, the batch above goes on `main` and that PR is built on it."""
    left = ((7, "h7", "its ready mark was withdrawn by lane-7"),)
    s = snap(cand("q1", MAIN, 7, 1), cand("q2", "q1", 2), ready=[mv(1), x(2, "h1"), pr(3)], left=left,
             pulled=frozenset({7}), depth=3)
    assert [a for a in decide(s) if isinstance(a, Build)] == [Build((1,), MAIN, 3, replaces="q1"),
                                                              Build((2,), "q1", 2, replaces="q2")]
    left = ((1, "h1", "held (by lane-9): not yet"),)
    s = snap(cand("q1", MAIN, 1), cand("q2", "q1", 2), ready=[x(2, "h1"), pr(3)], flying=(mv(1),), left=left,
             pulled=frozenset({1}), depth=3)
    assert [a for a in decide(s) if isinstance(a, Build)] == [Build((2,), MAIN, 3, replaces="q2"), Build((3,), "q2", 2)]


def test_a_staged_batch_goes_onto_a_move_on_the_chain_s_top_only_when_its_members_cross_it():
    passed = waiting("s1", MAIN, 2, records=RecordsRun(PASS))
    for two, commit in ((replace(lean(2), crosses=frozenset({"h1"})), [Commit("s1", "q1", 2)]), (lean(2), [])):
        s = snap(passed, cand("q1", MAIN, 1), ready=[mv(1), two], records=True, depth=3)
        assert [a for a in decide(s) if isinstance(a, Commit)] == commit, two


def test_a_staged_batch_dropped_in_the_tick_that_advances_main_to_a_move_is_not_built_again_in_it():
    left = ((5, "h5", "its ready mark was withdrawn by lane-1"),)
    for one, again in ((mv(1), []), (pr(1), [Build((2,), "q1", 3, records=True, replaces="s1")])):
        s = snap(cand("q1", MAIN, 1, state=PASS), waiting("s1", MAIN, 2, 5), ready=[one, lean(2)], left=left,
                 pulled=frozenset({5}), depth=3, records=True)
        acts = decide(s)
        assert Advance("q1") in acts and Drop("s1", "#5 isn't taken now: its ready mark was withdrawn by lane-1") in acts
        assert [a for a in acts if isinstance(a, Build)] == again, acts


# ---- the simulated queue ------------------------------------------------------------------------------------------------------


class Moves(World):
    """The simulated queue with one to three good PRs that are moves and up to two PRs (none erring) that change
    `tools/move/`'s scripts.  A move is certified (it lands on its mark), holds a land verdict (an uncertified one is merged
    by its head), is refused on its tip (taken on its mark alone, its build leaves it out and the loop never takes it again
    on that `main`'s refusal) or is withheld (no land verdict and its map isn't on `main`: never ready).  Its head never
    conflicts, being only the request, unless its land verdict merges it.  About half the PRs that are no move name nothing
    any move's map retires (`crossing`: `Ready.crosses` holds every move's head, ready or in flight).  `check` adds the move
    rules to every invariant."""

    def __init__(self, seed: int, **kw):
        super().__init__(seed, **kw)
        rng = random.Random(seed)
        good = sorted(self.good - set(self.fixes))
        movers = rng.sample(good, min(len(good), rng.randint(1, 3)))
        self.kind = {m: rng.choice(("certified", "certified", "verdict", "refused", "withheld")) for m in movers}
        for m in movers:
            self.marks[m] = replace(self.marks[m], move=True, conflict=self.marks[m].conflict and self.kind[m] == "verdict")
        rest = sorted(set(self.marks) - set(movers) - set(self.erring))
        self.scripts = set(rng.sample(rest, min(len(rest), rng.randint(0, 2))))
        for m in self.scripts:
            self.marks[m] = replace(self.marks[m], files=self.marks[m].files | {"tools/move/rename.py"}, scripts=True)
        self.left_out = {m for m, k in self.kind.items() if k == "withheld"}
        self.good -= {m for m, k in self.kind.items() if k in ("withheld", "refused")}
        others = random.Random(f"{seed} crossing")
        self.crossing = {m for m in sorted(set(self.marks) - set(movers)) if others.random() < 0.5}

    def ready(self) -> tuple[Ready, ...]:
        return tuple(r for r in super().ready() if r.number not in self.left_out)

    def snapshot(self) -> Snapshot:
        s = super().snapshot()
        moves = frozenset(r.head for r in (*s.ready, *s.flying) if r.move)
        return replace(s, ready=tuple(replace(r, crosses=moves) if r.number in self.crossing else r for r in s.ready))

    def crosses(self, r: Ready) -> bool:
        return not r.move and r.number in self.crossing

    def moving(self, s: Snapshot, acts: list, put: bool = True) -> bool:
        """Whether a move holds the tick's builds: one on main's chain once ``acts`` are applied (without the batches their
        `Commit`s put there, unless ``put``), or one ``acts`` advances to."""
        built = {r.number: r for r in (*s.ready, *s.flying)}
        advancing = [m for c in s.candidates if Advance(c.sha) in acts for m in c.members]
        commits = set() if put else {a.sha for a in acts if isinstance(a, Commit)}
        chain = [m for c in self.chain(s, acts) if c.sha not in commits for m in c.members]
        return any(built[m].move for m in (*advancing, *chain))

    def pool(self, s: Snapshot, acts: list, aside: bool = False) -> list[Ready]:
        """`World.pool` without the moves while a PR in flight below a build changes the move scripts, and only the PRs that
        cross the moves while one holds the builds."""
        built = {r.number: r for r in (*s.ready, *s.flying)}
        scripts = [m for c in self.chain(s, acts) for m in c.members if built[m].scripts]
        moving = self.moving(s, acts)
        return [r for r in super().pool(s, acts, aside) if not (r.move and scripts) and (not moving or self.crosses(r))]

    def admits(self, s: Snapshot, acts: list, where: str) -> None:
        """`World.admits` of the PRs `pool` leaves, so while a move holds the builds (the tick advances to one, or the chain,
        the batch its `Commit` puts there among it, holds one in flight) a PR that crosses it is built as any is while it
        waits, and only such a PR is built or committed."""
        ready = {r.number: r for r in s.ready}
        if self.moving(s, acts):
            built = [m for a in acts if isinstance(a, Build) and a.replaces is None for m in a.members]
            assert all(self.crosses(ready[m]) for m in built), f"{where}: built above a move what doesn't cross it"
            self.seen["a tick's builds held for a move"] += 1
            self.seen["a PR built above a move"] += bool(built)
        if self.moving(s, acts, put=False):
            commits = {a.sha for a in acts if isinstance(a, Commit)}
            put = [(m, h) for c in s.candidates if c.sha in commits for m, h in zip(c.members, c.heads)]
            assert all(m in ready and ready[m].head == h and self.crosses(ready[m]) for m, h in put), (
                f"{where}: committed above a move what doesn't cross it")
            self.seen["a staged batch put above a move"] += bool(put)
        super().admits(s, acts, where)

    def check(self, s: Snapshot, acts: list, moved: str | None) -> None:
        super().check(s, acts, moved)
        where = f"seed {self.seed} tick {self.tick}: {acts}"
        ready = {r.number: r for r in s.ready}
        built = {**ready, **{r.number: r for r in s.flying}}
        advancing = [m for c in s.candidates if Advance(c.sha) in acts for m in c.members]
        for a in acts:
            if not isinstance(a, Build):
                continue
            moves, below = [m for m in a.members if ready[m].move], self.below(s, acts, a.base) | set(
                next((c.members for c in s.candidates if c.sha == a.after), ()))
            assert not set(a.members) & self.left_out, f"{where}: a move with no land verdict whose map isn't on main built"
            assert len(moves) <= 1 and (not moves or a.members[-1] == moves[0]), f"{where}: a move not last and alone"
            assert not [m for m in below if built[m].move] or all(self.crosses(ready[m]) for m in a.members), (
                f"{where}: built above a move in flight what doesn't cross it")
            assert not moves or not [m for m in (*a.members, *below) if built[m].scripts], (
                f"{where}: a move built with what judges it changed below it")
            assert not [m for m in advancing if built[m].move] or all(self.crosses(ready[m]) for m in a.members), (
                f"{where}: built in the tick main advances to a move what doesn't cross it")
            self.seen["a move built"] += bool(moves)
            self.seen["a move built on a candidate"] += bool(moves and below)
            self.seen["a move built behind other members"] += bool(moves and len(a.members) > 1)
            self.seen["a PR built in the tick main advances to a move"] += bool([m for m in advancing if built[m].move])
        pool = [r for r in World.pool(self, s, acts) if r.move]
        flying = {m for c in s.candidates for m in c.members if c.sha not in {a.sha for a in acts if isinstance(a, Drop)}}
        self.seen["a move waited for the move scripts"] += bool(pool and [m for m in flying if built[m].scripts])
        self.seen["a PR waited above a move"] += bool([m for m in flying if built[m].move] and [
            r for r in World.pool(self, s, acts) if not self.crosses(r)])

    def apply(self, a) -> None:
        if isinstance(a, Build) and (refused := [m for m in a.members if self.kind.get(m) == "refused"]):
            self.left_out |= set(refused)
            self.seen["a move refused on its tip"] += 1
            if not (a := replace(a, members=tuple(m for m in a.members if m not in refused))).members:
                return
        if isinstance(a, Advance):
            c = self.cands[a.sha]
            assert all(self.kind.get(m, "certified") in ("certified", "verdict") for m in c.members), (
                f"seed {self.seed}: an uncertified move with no land verdict reaches main")
            for m in set(c.members) & set(self.kind):
                self.seen[f"a {self.kind[m]} move landed"] += 1
        super().apply(a)


def test_the_simulated_queue_with_moves_builds_each_last_alone_and_on_no_move_script_change_and_settles():
    """With moves and PRs that change the move scripts among the PRs: every invariant holds, and a move is last and the only
    move in its batch, only the PRs that cross it are built above it or in the tick `main` advances to it, and each as soon
    as a batch may take it, it isn't built on or beside a PR that changes `tools/move/`'s scripts, one with no land verdict
    lands only certified, and every scenario settles within `ticks`."""
    seen: Counter[str] = Counter()
    for seed in SEEDS:
        w = Moves(seed, records=seed % 3 == 0)
        while not w.done():
            assert w.tick < ticks(w), (f"seed {seed}: not settled after {ticks(w)} ticks: merged {sorted(w.merged)} of "
                                       f"{sorted(w.good)}, live {list(w.cands.values())}")
            w.step()
        seen.update(w.seen)
    for what in ("a move built", "a move built on a candidate", "a move built behind other members", "a move refused on its tip",
                 "a certified move landed", "a verdict move landed", "a move waited for the move scripts",
                 "a PR waited above a move", "a PR built above a move", "a PR built in the tick main advances to a move",
                 "a staged batch put above a move"):
        assert seen[what], f"no seed exercised: {what}"


MOVE_MISREADS = {
    "no PR as a move": lambda s: replace(s, ready=tuple(replace(r, move=False) for r in s.ready),
                                         flying=tuple(replace(r, move=False) for r in s.flying)),
    "every PR as crossing every move": lambda s: replace(s, ready=tuple(
        replace(r, crosses=frozenset(q.head for q in (*s.ready, *s.flying))) for r in s.ready)),
    "no PR as crossing a move": lambda s: replace(s, ready=tuple(replace(r, crosses=frozenset()) for r in s.ready)),
    "no PR as changing the move scripts": lambda s: replace(s, ready=tuple(replace(r, scripts=False) for r in s.ready),
                                                            flying=tuple(replace(r, scripts=False) for r in s.flying)),
    "no move in flight": lambda s: replace(s, flying=tuple(replace(r, move=False) for r in s.flying)),
    "no move script change in flight": lambda s: replace(s, flying=tuple(replace(r, scripts=False) for r in s.flying)),
}


@pytest.mark.parametrize("misread", MOVE_MISREADS)
def test_decide_misreading_a_move_field_breaks_an_invariant_of_the_simulated_queue_with_moves(misread):
    """The move invariants hold `decide` to `Ready.move` and `Ready.scripts`, of the ready PRs and of those in flight, and to
    `Ready.crosses`."""
    with pytest.raises(AssertionError):
        for seed in SEEDS:
            w = Moves(seed, policy=lambda s: decide(MOVE_MISREADS[misread](s)))
            while not w.done():
                assert w.tick < ticks(w), f"seed {seed}: not settled after {ticks(w)} ticks"
                w.step()


# ---- the map's path rows ------------------------------------------------------------------------------------------------------


def test_path_rows_are_the_map_rows_whose_old_and_new_words_are_paths_words_in_included_in_order_once():
    text = ('[words]\n"verity/old/" = "verity/new/"\n"OldName" = "NewName"\n"a/b.py" = "c/b.py"\n'
            '"x/" = "y"\n\n[[words_in]]\nglob = "**/*.md"\n[words_in.words]\n"docs/a.md" = "docs/b.md"\n'
            '"verity/old/" = "verity/new/"\n')
    assert path_rows(text) == [("verity/old/", "verity/new/"), ("a/b.py", "c/b.py"), ("docs/a.md", "docs/b.md")]
    assert path_rows('[words]\nx = 1\n"a/b" = 1\n"c/d" = ["e/f"]\n') == [] and path_rows('words = "a/b"\n') == [] and path_rows("[rewrite]\nexclude = []\n") == []
    with pytest.raises(tomllib.TOMLDecodeError):
        path_rows("[words\n")
    assert path_rows('words_in = ["x"]\n[words]\n"p/q" = "r/s"\n') == [("p/q", "r/s")]


# ---- the loop -----------------------------------------------------------------------------------------------------------------

#: `rename.py` as the loop runs it: each path row of the map's `[words]` moves the files under it, and every row's word is
#: rewritten in every file outside `tools/`
RENAME = '''import argparse, pathlib, tomllib
a = argparse.ArgumentParser()
a.add_argument("--map")
a.add_argument("--root")
x = a.parse_args()
root = pathlib.Path(x.root)
words = tomllib.loads((root / x.map).read_text()).get("words", {})
for f in sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts and p.parts[len(root.parts)] != "tools"):
    rel = new = f.relative_to(root).as_posix()
    for o, n in words.items():
        if "/" in o and new.startswith(o):
            new = n + new[len(o):]
    text = f.read_text()
    for o, n in words.items():
        text = text.replace(o, n)
    f.unlink()
    (root / new).parent.mkdir(parents=True, exist_ok=True)
    (root / new).write_text(text)
'''

#: `move_check.py --json`: each call's arguments appended to LOG; while REFUSE exists, a refusal, or as it says, no report
#: (`garbage`, or JSON that isn't one: `list`), a report certifying the move with a refusal's exit (`exit`) or one not certifying it with exit 0 (`zero`)
CHECK = """import json, pathlib, sys
log, refuse = pathlib.Path(LOG), pathlib.Path(REFUSE)
with log.open("a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\\n")
mode = refuse.read_text().strip() if refuse.exists() else None
if mode in ("garbage", "list"):
    print("Traceback: boom" if mode == "garbage" else "[]")
    sys.exit(2)
no = mode is not None and mode != "exit"
if mode == "zero":
    print(json.dumps({"certified": False, "refusals": []}))
    sys.exit(0)
print(json.dumps({"certified": not no, "base": sys.argv[1], "head": sys.argv[2], "renames": [], "edits": [], "words": {},
                  "rules": {}, "refusals": [{"path": "new/a.txt", "reason": "an edit no row explains"}] if no else []}))
print("move-check: refused" if no else "move-check: certified", file=sys.stderr)
sys.exit(0 if mode is None else 1)
"""


class Moved:
    """``env``'s `main` with the stub scripts and the map moving `old/` to `new/` (and the word `oldword`), and `old/a.txt`;
    `log` lists each `move_check.py` call's arguments, and `refuse` makes it refuse while it exists."""

    def __init__(self, env: Env, map_on_main: bool = True):
        self.env, self.log, self.refuse = env, env.tmp / "move_check.log", env.tmp / "move_check.refuse"
        check = CHECK.replace("LOG", repr(str(self.log))).replace("REFUSE", repr(str(self.refuse)))
        files = {Q.RENAME: RENAME, Q.MOVE_CHECK: check, "old/a.txt": "oldword in old/a.txt\n"}
        if map_on_main:
            files[MAP] = MAP_ROWS
        self.main = _main_to(env, files)
        self.closed: list[tuple[int, str]] = []
        env.o.gh.close = lambda n, comment: self.closed.append((n, comment)) or env.o.gh.prs.pop(n)

    def calls(self) -> list[list[str]]:
        return [json.loads(x) for x in self.log.read_text().splitlines()] if self.log.is_file() else []

    def move(self, n: int, verdict: bool) -> str:
        head = _move(self.env, n, {"old/a.txt": None, "new/a.txt": "newword in new/a.txt\n"})
        if not verdict:
            self.env.labels.verdicts[n, head] = []
        return head


def _cached(env: Env, cache: Path):
    """``env``'s loop with its world keeping its caches in ``cache``, as the loop's process keeps them beside its log."""
    loop = env.loop(policy)
    loop.world = w = Fake(loop.world.q, env.labels, Executor(env.graph2), post=lambda text, to=(): env.posts.append(text) or True,
                          now=lambda: env.clock, cache=cache)
    w.withdrawn, w.cancelled = env.withdrawn, env.cancelled
    return loop


def _passes(env: Env, sha: str) -> None:
    keys = env.listed(sha, {"pytest:a": ["new/a.txt"]})
    env.result(keys["pytest:a"], "pytest:a", "pass")


def _builds(recs: list[dict]) -> list[dict]:
    return [r for r in recs if r["action"] == "build" and r.get("sha")]


def _show(env: Env, rev: str) -> str:
    return _git(env.o.origin, "show", "-s", "--format=%B", rev)


def _files(env: Env, rev: str) -> list[str]:
    return _git(env.o.origin, "ls-tree", "-r", "--name-only", rev).split()


def test_a_certified_move_lands_on_its_mark_alone_regenerated_last_on_its_batch_s_tip_with_its_certificate(env):
    """A move PR with no land verdict whose map is on `main` is taken on its ready mark, built last in its batch as its move
    regenerated on the batch's tip by `main`'s `rename.py` and certified by `main`'s `move_check.py` (given the map's path
    rows as `--map`), its report in the build line; it lands as any candidate, and its PR, whose head `main` never holds, is
    closed."""
    m = Moved(env)
    head = m.move(1, verdict=False)
    two = env.put(2, {"b.txt": "2\n"})
    loop = env.loop(policy)
    (built,) = _builds(loop.tick())
    sha, tip = built["sha"], _git(env.o.origin, "rev-parse", f"{built['sha']}^1")
    assert (built["members"], built["heads"]) == ([2, 1], [two, head]), built
    assert _git(env.o.origin, "rev-list", "--parents", "-n1", tip).split() == [tip, m.main, two]
    assert _git(env.o.origin, "rev-list", "--parents", "-n1", sha).split() == [sha, tip]
    assert "new/a.txt" in _files(env, sha) and "old/a.txt" not in _files(env, sha) and "b.txt" in _files(env, sha)
    assert _git(env.o.origin, "show", f"{sha}:new/a.txt") == "newword in new/a.txt"
    assert Q.move_of(env.o.origin, sha) == MAP and f"PR: #1\nHead: {head}\n" in _show(env, sha)
    assert m.calls() == [[tip, sha, "--map=old/=new/", "--json"]]
    (move,) = built["moves"]
    assert {k: move[k] for k in ("pr", "map", "tip", "commit", "on_mark")} == {
        "pr": 1, "map": MAP, "tip": tip, "commit": sha, "on_mark": True}, move
    assert move["certificate"]["certified"] is True and move["certificate"]["head"] == sha, move
    _passes(env, sha)
    assert _acts(loop.tick()) == ["advance", "on_main"] and env.main() == sha
    assert [n for n, _ in m.closed] == [1] and sha[:12] in m.closed[0][1] and MAP in m.closed[0][1]


def test_a_move_regenerated_once_on_a_tip_is_kept_in_the_loop_s_cache_for_the_next_tick(env):
    """A process runs one tick: the regenerations the loop made on candidates of `main` are read back from its cache, so a
    refusal there keeps a move on its mark alone out in the next process, and an older `main`'s are dropped."""
    m = Moved(env)
    m.refuse.touch()
    head = m.move(1, verdict=False)
    recs = _cached(env, env.tmp / "cache").tick()
    assert not _builds(recs) and [r["why"][:41] for r in recs if r["action"] == "skip"] == [
        "it came on its mark alone, and its move r"], recs
    kept = json.loads((env.tmp / "cache" / "moves-made.json").read_text())
    assert [k.split(":")[1:] for k in kept] == [[m.main, MAP, "1", head]] and "isn't certified" in next(iter(kept.values()))["why"]
    again = _cached(env, env.tmp / "cache")
    again.world.ready(m.main)
    assert again.world.left[1][1].startswith(f"no land verdict, and its move isn't certified on a candidate on `main` {m.main[:12]}")
    later = _main_to(env, {"later.txt": "later\n"})
    env.put(2, {"d.txt": "d\n"})
    assert again.world.fetch() == later
    again.world.ready(later)
    assert 1 not in again.world.left and again.world.on_mark == {1}
    again.world.build([env.o.gh.prs[2]], later, later)
    assert json.loads((env.tmp / "cache" / "moves-made.json").read_text()) == {}


def test_a_move_refused_on_its_tip_is_left_out_when_it_came_on_its_mark_alone_and_told_why(env):
    """An uncertified move taken on its mark is left out of its build, its owner told why and what lands it, and the next tick
    leaves it out on the refusal; the batch's other members are built without it."""
    m = Moved(env)
    m.refuse.touch()
    head = m.move(1, verdict=False)
    two = env.put(2, {"b.txt": "2\n"})
    loop = env.loop(policy)
    recs = loop.tick()
    (skip,) = [r for r in recs if r["action"] == "skip"]
    (built,) = _builds(recs)
    assert (skip["number"], skip["heads"], built["members"], built["heads"]) == (1, [head], [2], [two]), recs
    assert "it came on its mark alone, and its move regenerated on " in skip["why"] and "new/a.txt: an edit no row" in skip["why"]
    assert "with 1 --map rows exit 1): move-check: refused: new/a.txt" in skip["why"], skip["why"]
    assert skip["text"].endswith("It lands with a land verdict, or on its mark once a candidate on a later `main` certifies "
                                 "its move.") and "merge `main`" not in skip["text"], skip
    assert "moves" not in built
    loop.tick()
    assert "its move isn't certified on a candidate on `main`" in loop.world.left[1][1], loop.world.left


def test_an_uncertified_move_with_a_land_verdict_is_merged_by_its_head(env):
    """A move with a land verdict whose regeneration isn't certified is merged by its head, as any PR is."""
    m = Moved(env)
    m.refuse.touch()
    head = m.move(1, verdict=True)
    loop = env.loop(policy)
    (built,) = _builds(loop.tick())
    assert (built["members"], built["heads"], "moves" in built) == ([1], [head], False), built
    assert _git(env.o.origin, "rev-parse", f"{built['sha']}^2") == head and len(m.calls()) == 1
    assert loop.world.certified == {}


def test_a_certified_move_with_a_land_verdict_is_regenerated_too(env):
    m = Moved(env)
    head = m.move(1, verdict=True)
    (built,) = _builds(env.loop(policy).tick())
    assert built["heads"] == [head] and built["moves"][0]["on_mark"] is False and Q.move_of(env.o.origin, built["sha"]) == MAP


def test_a_move_without_a_land_verdict_waits_for_its_map_on_main(env):
    Moved(env, map_on_main=False)
    head = _move(env, 1, {"old/a.txt": None, "new/a.txt": "x\n"})
    env.labels.verdicts[1, head] = []
    loop = env.loop(policy)
    assert not _builds(loop.tick())
    assert loop.world.left[1] == (head, f"no land verdict, and a move lands on its mark alone only once its map {MAP} is on `main`")


def test_a_move_waits_while_a_member_below_it_changes_the_move_scripts_its_owner_told_nothing(env):
    """A member below the move that changes `tools/move/`'s scripts holds it back: the build leaves it out, logged and told to
    nobody, and nothing is regenerated."""
    m = Moved(env)
    head = m.move(1, verdict=False)
    env.put(2, {Q.RENAME: RENAME + "# changed\n"})
    loop = env.loop(policy)
    recs = loop.tick()
    (skip,) = [r for r in recs if r["action"] == "skip"]
    assert (skip["number"], skip["heads"], skip["waits"], skip["done"]) == (1, [head], True, False), skip
    assert skip["why"].startswith(f"the members below it change {Q.RENAME}") and "text" not in skip
    assert [b["members"] for b in _builds(recs)] == [[2]] and m.calls() == [] and not env.posts


def test_a_ready_move_whose_regenerated_move_is_on_main_is_left_out(env):
    """A move PR left open after its regenerated move landed (its close failed) is never taken again."""
    m = Moved(env)
    head = m.move(1, verdict=False)
    env.o.gh.close = lambda n, comment: (_ for _ in ()).throw(Q.QueueError("no answer"))
    loop = env.loop(policy)
    (built,) = _builds(loop.tick())
    _passes(env, built["sha"])
    assert _acts(loop.tick()) == ["advance", "on_main"]
    assert any("landed as its regenerated move, but isn't closed" in x for x in env.out), env.out
    loop.world.ready(env.main())
    assert loop.world.left[1] == (head, "its move, regenerated from its head, is on `main`")


def test_a_move_s_conflicting_head_doesnt_hold_it_once_its_move_is_regenerated(env):
    """A move's head that doesn't merge onto `main` is moot once its move is regenerated there: it is built, not told to
    rebase."""
    m = Moved(env)
    base = m.main
    _main_to(env, {"new/a.txt": "main's own\n"})
    head = _move(env, 1, {"old/a.txt": None, "new/a.txt": "newword in new/a.txt\n"}, base=base)
    loop = env.loop(policy)
    recs = loop.tick()
    assert not [r for r in recs if r["action"] == "conflict"], recs
    ready = loop.world.ready(env.main())
    assert [(r.number, r.head, r.move, r.conflict) for r in ready] == [(1, head, True, False)]


@pytest.mark.parametrize("unmark", [False, True])
def test_only_a_pr_crossing_a_move_in_flight_is_built_above_it_marked_or_not(env, unmark):
    """`decide` reads a move in flight from the ready PRs (`World.ready`) or, once its mark is gone, from the PRs in flight
    (`Snapshot.flying`, its PR's `Move:` line), and the loop its map at its head as built: above it a PR whose own change
    names nothing that map retires is built, merged onto the moved tree, and a PR that names a path or a word it retires, in
    a file it changes or a line it adds, waits, as does a second move, in the tick that advances `main` to it too; on the
    moved `main` those two are left out for the move they lack (`queue.lacked_moves`)."""
    m = Moved(env)
    head = m.move(1, verdict=True)
    loop = env.loop(decide)
    (first,) = _builds(loop.tick())
    assert first["members"] == [1]
    if unmark:
        env.o.gh.ready.discard((1, head))
    two = env.put(2, {"b.txt": "2\n"})
    three, four = env.put(3, {"old/b.txt": "3\n"}), env.put(4, {"c.txt": "reads OldWord\n"})
    _move(env, 5, {"d.txt": "5\n"})
    (above,) = _builds(loop.tick())
    assert (above["members"], above["heads"], above["base"]) == ([2], [two], first["sha"]), above
    assert {"new/a.txt", "b.txt"} <= set(_files(env, above["sha"])) and "old/a.txt" not in _files(env, above["sha"])
    _passes(env, first["sha"])
    recs = loop.tick()
    assert "advance" in _acts(recs) and env.main() == first["sha"] and not _builds(recs), recs
    loop.world.ready(env.main())
    assert {n: loop.world.left[n][1].split(":")[0] for n in (3, 4)} == {
        n: f"it lacks the move {first['sha'][:12]} `main` landed ({MAP}), whose retired names its own change names (`{w}`)"
        for n, w in ((3, "old/"), (4, "oldword"))}, loop.world.left
    assert (loop.world.left[3][0], loop.world.left[4][0]) == (three, four)


@pytest.mark.parametrize("why", ["it doesn't parse", "it's missing", "the change can't be read"])
def test_a_pr_waits_above_a_move_whose_map_can_t_be_read_at_its_head_or_while_its_own_change_can_t_be_read(
        env, monkeypatch, why):
    """Fail closed: a move whose map doesn't parse at its head or is missing there is crossed by nothing, whatever `main`'s
    copy says, so nothing is built above it; and a PR whose own change can't be read crosses no move, while another is built
    above it."""
    Moved(env)
    files = {"old/a.txt": None, "new/a.txt": "newword in new/a.txt\n"}
    _move(env, 1, {**files, MAP: {"it doesn't parse": "[words\n", "it's missing": None}.get(why, MAP_ROWS)})
    loop = env.loop(decide)
    (first,) = _builds(loop.tick())
    assert first["members"] == [1] and Q.move_of(env.o.origin, first["sha"]) == MAP, first
    two, six = env.put(2, {"b.txt": "2\n"}), env.put(6, {"f.txt": "6\n"})
    if why == "the change can't be read":
        own = Q.own_text
        monkeypatch.setattr(Q, "own_text", lambda repo, main, head, files=None: (
            (_ for _ in ()).throw(Q.QueueError("git diff: bad object")) if head == two else own(repo, main, head, files)))
    builds = _builds(loop.tick())
    assert [(b["members"], b["heads"], b["base"]) for b in builds] == (
        [([6], [six], first["sha"])] if why == "the change can't be read" else []), builds


@pytest.mark.parametrize("unmark", [False, True])
def test_a_move_isnt_built_above_a_pr_in_flight_that_changes_the_move_scripts_marked_or_not(env, unmark):
    """A PR in flight changing `tools/move/`'s scripts, read from the ready PRs or from the PRs in flight, holds back a move:
    another PR is built above it, and the move isn't taken into that batch at all (the build's own guard, `World._move`, never
    sees it)."""
    m = Moved(env)
    two = env.put(2, {Q.RENAME: RENAME + "# changed\n"})
    loop = env.loop(decide)
    assert [b["members"] for b in _builds(loop.tick())] == [[2]]
    if unmark:
        env.o.gh.ready.discard((2, two))
    m.move(1, verdict=True)
    env.put(3, {"b.txt": "3\n"})
    recs = loop.tick()
    assert [b["members"] for b in _builds(recs)] == [[3]] and not [r for r in recs if r["action"] == "skip"], recs


@pytest.mark.parametrize("mode", ["garbage", "list", "exit", "zero"])
def test_a_move_check_that_gives_no_report_or_exits_as_refusing_doesnt_certify(env, mode):
    """`move_check.py` certifies only with its report saying so and its exit 0: no report, a report it exits 1 on or one not
    saying so, is a refusal, and the move on its mark alone is left out."""
    m = Moved(env)
    m.refuse.write_text(mode)
    m.move(1, verdict=False)
    recs = env.loop(policy).tick()
    (skip,) = [r for r in recs if r["action"] == "skip"]
    assert not _builds(recs) and ("couldn't judge" if mode in ("garbage", "list") else "isn't certified") in skip["why"], skip
    assert {"garbage": "no JSON report (", "list": "(exit 2): []", "exit": "exit 1)", "zero": "exit 0)"}[mode] in skip["why"], skip


def test_a_move_runs_main_s_move_check_whatever_a_member_below_it_does_to_that(env):
    """A member below the move that changes `move_check.py` (one that refuses every move) doesn't hold it back: the move is
    certified by `main`'s."""
    m = Moved(env)
    m.move(1, verdict=False)
    env.put(2, {Q.MOVE_CHECK: "import sys\nsys.exit(1)\n"})
    (built,) = _builds(env.loop(policy).tick())
    assert built["members"] == [2, 1] and built["moves"][0]["certificate"]["certified"] is True, built


def test_a_move_refused_on_its_mark_that_then_gets_a_land_verdict_is_merged_by_its_head(env):
    m = Moved(env)
    m.refuse.touch()
    head = m.move(1, verdict=False)
    loop = env.loop(policy)
    assert [r["number"] for r in loop.tick() if r["action"] == "skip"] == [1]
    del env.labels.verdicts[1, head]
    (built,) = _builds(loop.tick())
    assert (built["members"], built["heads"], "moves" in built) == ([1], [head], False), built


def test_a_build_carries_no_certificate_of_an_earlier_build(env):
    """A move certified in one build and merged by its head in a later one, on a newer `main` where its move is refused,
    carries no certificate there."""
    m = Moved(env)
    head = m.move(1, verdict=True)
    w = env.loop(policy).world
    w.ready(m.main)
    tip, merged, _ = w.build([env.o.gh.prs[1]], m.main, m.main)
    assert [(x.pr, x.head, x.commit) for x in merged] == [(1, head, tip)] and w.certified[1]["commit"] == tip
    assert _git(w.repo, "rev-parse", f"{tip}^") == m.main and Q.move_of(w.repo, tip) == MAP
    later = _main_to(env, {"later.txt": "later\n"})
    m.refuse.touch()
    assert w.fetch() == later
    w.ready(later)
    _, merged, _ = w.build([env.o.gh.prs[1]], later, later)
    assert [(x.pr, x.head) for x in merged] == [(1, head)] and w.certified == {}


def test_a_regeneration_on_a_tip_whose_members_change_the_move_scripts_is_refused(env):
    """`Queue._regenerated` itself refuses a tip whose members change `tools/move/`'s scripts, as drive's path calls it
    without the loop's guard."""
    m = Moved(env)
    m.move(1, verdict=True)
    tip = env.put(2, {Q.RENAME: RENAME + "# changed\n"})
    w = env.loop(policy).world
    w.ready(m.main)
    commit, why, report = w.q._regenerated(tip, env.o.gh.prs[1], m.main)
    assert (commit, report) == (None, None) and why.startswith(f"the members below it change {Q.RENAME} from `main`"), why
    assert m.calls() == []


def test_a_move_whose_map_doesnt_parse_on_its_tip_is_refused_before_anything_runs(env):
    m = Moved(env)
    m.move(1, verdict=False)
    env.put(2, {MAP: "[words\n"})
    recs = env.loop(policy).tick()
    (skip,) = [r for r in recs if r["action"] == "skip"]
    assert f"its map {MAP} at " in skip["why"] and " doesn't parse (" in skip["why"] and m.calls() == [], skip


def test_a_move_isnt_taken_on_its_mark_alone_with_the_low_risk_lane_off(env):
    """drive.toml's `lowrisk_lane = false` takes no PR without a land verdict: a certified move neither, while one with a land
    verdict is still built, regenerated."""
    m = Moved(env)
    _main_to(env, {Q.DRIVE_FLAGS: "lowrisk_lane = false\n"})
    alone = m.move(1, verdict=False)
    m.move(2, verdict=True)
    loop = env.loop(policy)
    (built,) = _builds(loop.tick())
    assert built["members"] == [2] and built["moves"][0]["on_mark"] is False, built
    assert loop.world.left[1] == (alone, "no land verdict, and drive.toml's `lowrisk_lane` is off, so a move lands only with one")


@pytest.mark.parametrize("guarded", [True, False])
def test_under_signed_verdicts_a_move_changing_a_signed_guard_needs_a_land_verdict(env, guarded):
    """Under drive.toml's `signed_verdicts`, a move whose change meets `SIGNED_GUARDS` (here, the map's word rewritten in `labelauth.py`) isn't
    taken on its mark alone, however certified; one that meets none is."""
    Moved(env)
    guard = "tools/research/src/research/labelauth.py"
    flags = 'signed_verdicts = true\nlowrisk_excludes = ["*.py", "*.json"]\n'
    _main_to(env, {Q.DRIVE_FLAGS: flags, guard: "old/ judges labels\n"})
    files = {"old/a.txt": None, "new/a.txt": "newword in new/a.txt\n", **({guard: "new/ judges labels\n"} if guarded else {})}
    head = _move(env, 1, files)
    env.labels.verdicts[1, head] = []
    loop = env.loop(policy)
    builds = _builds(loop.tick())
    if guarded:
        assert not builds and loop.world.left[1] == (head, f"no land verdict, and it changes {guard}, and under drive.toml's "
                                                            "`signed_verdicts` a move changing `SIGNED_GUARDS` lands only with "
                                                            "one"), loop.world.left
    else:
        assert [(b["members"], b["moves"][0]["on_mark"]) for b in builds] == [([1], True)], builds


def test_under_signed_verdicts_a_move_on_its_mark_whose_regeneration_rewrites_a_signed_guard_is_left_out(env):
    """`ready` takes a move on its mark by its head's files; the regenerated commit can change more (here, the map's word in
    the handle registry, which the head leaves alone), so the build judges that commit's files too and leaves the move out,
    told to get a land verdict, when one meets `SIGNED_GUARDS`."""
    Moved(env)
    guard = Q.REGISTRY.as_posix()
    flags = 'signed_verdicts = true\nlowrisk_excludes = ["*.py", "*.json"]\n'
    _main_to(env, {Q.DRIVE_FLAGS: flags, guard: json.dumps({"handles": [], "note": "old/ judges labels"}) + "\n"})
    head = _move(env, 1, {"old/a.txt": None, "new/a.txt": "newword in new/a.txt\n"})
    env.labels.verdicts[1, head] = []
    loop = env.loop(policy)
    recs = loop.tick()
    assert not _builds(recs), recs
    (skip,) = [r for r in recs if r["action"] == "skip" and r.get("number") == 1]
    assert skip["why"].startswith("it came on its mark alone, and its move regenerated on ") and (
        f"it changes {guard}, and under drive.toml's `signed_verdicts`" in skip["why"]), skip
    assert skip["text"].endswith("It lands with a land verdict."), skip
