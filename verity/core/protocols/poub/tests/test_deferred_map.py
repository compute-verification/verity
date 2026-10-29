"""memory-map/v1 under deferred residency against the map that writes each block at the step that completed it: vllm's
rules for moving one to the other (`deferred`, its compare.py in run r20261018-070848-81e2, the Store's map against
circuits' oracle), on hand histories and on seeded ones.  The moved map is
one `Replay` accepts, every index the oracle wrote is written once or withdrawn once, a block is withdrawn exactly
when its last structural reader is at or before the line its encode landed in, each written block is in the plans of
the passes after that line through its last structural reader, and a session the moved map drives is accepted."""

from __future__ import annotations

import random
from collections import defaultdict

import pytest

from test_appends import N, Run, payload, salts, store, units_mapped
from verity.core.protocols.poub import audit, codec
from verity.core.protocols.poub.memory_map import Header, Replay, Step, Written

POOL = list(range(2, N)) + list(range(100, 106))
"""A KV pool's slots: registered blocks 2 … 7, filler, and memory outside the store."""
SEEDS = range(16)
STEPS = 24


def deferred(osteps: list[Step], encodes: list[list], first: int, n: int) -> list[Step]:
    """`osteps`, a map that records each write at the step w that completed its block and each free at its values' last
    structural reader f, moved to deferred residency by the session's public `encodes` ([step made, step landed or
    None, first, count] per append, steps from `first`), as compare.py moves it except where noted:

    - a write at w moves to the line r(w) its append's encode landed in, its index unchanged, and is in no line if the
      encode never landed;
    - the filler a completed block's slot held (a slot below `n` not touched before) is freed at its reach, w;
    - a re-key, a write over a slot whose block is live, frees the old index at the re-key's step;
    - a written block's free at f stays at f if f > r(w); otherwise the block is withdrawn at r(w), with no write and
      no free (where compare.py moved the free to r(w) + 1);
    - a free is dropped where its line writes the slot: the write releases what the slot held.

    compare.py's last rule, moving a line's second write of a slot to the next line, is left nothing to move: a slot's
    next block completes no sooner than its last one's free, which is after that one's line unless it was withdrawn."""
    landed = {made - first: None if land is None else land - first for made, land, _, _ in encodes}
    writes: list[tuple[Written, int | None]] = []
    frees: list[tuple[int | None, int, int]] = []
    seen: set[int] = set()
    owner: dict[int, int] = {}
    for st in osteps:
        for x in st.freed:
            if x in owner:
                frees.append((owner.pop(x), x, st.step))
            else:
                frees.append((None, x, st.step))
                seen.add(x)
        for w in st.written:
            if w.slot in owner:
                frees.append((owner.pop(w.slot), w.slot, st.step))
            elif w.slot < n and w.slot not in seen:
                frees.append((None, w.slot, st.step))
            seen.add(w.slot)
            owner[w.slot] = len(writes)
            writes.append((w, landed.get(st.step)))
    gone = {i: f for i, _, f in frees if i is not None}
    dead = {i for i, (_, r) in enumerate(writes) if r is not None and i in gone and gone[i] <= r}
    lines: dict[int, list[int]] = defaultdict(list)
    for i, (_, r) in enumerate(writes):
        if r is not None:
            lines[r].append(i)
    freed: dict[int, set[int]] = defaultdict(set)
    for i, x, f in frees:
        if i is None or (writes[i][1] is not None and i not in dead):
            freed[f].add(x)
    out = []
    for s in sorted(set(freed) | set(lines)):
        ws = [writes[i][0] for i in lines.get(s, ()) if i not in dead]
        fs = freed.get(s, set()) - {w.slot for w in ws}
        wd = tuple(writes[i][0].index for i in lines.get(s, ()) if i in dead)
        if fs or ws or wd:
            out.append(Step(s, tuple(sorted(fs)), tuple(ws), wd))
    return out


def d(slot: int, step: int) -> bytes:
    return codec.digest(page(slot, step))


def page(slot: int, step: int) -> bytes:
    return payload(b"kv %d at step %d" % (slot, step))


def plans(dm: list[Step], steps: int) -> dict[int, list[int]]:
    """The steps whose pass holds each block, the moved map replayed with a pass at every step (`Replay.take`)."""
    replay, lines = Replay(Header(bytes(64), N)), {s.step: s for s in dm}
    plan, held = set(range(N)), defaultdict(list)
    for t in range(steps):
        added, removed = replay.take(t)
        plan = (plan - {j for f, c in removed for j in range(f, f + c)}) | {j for f, c in added for j in range(f, f + c)}
        for j in plan:
            held[j].append(t)
        if t in lines:
            replay.apply(lines[t])
    return dict(held)


def drive(run: Run, dm: list[Step], appends: dict[int, list[bytes]], steps: int, boundaries=()) -> audit.Verdict | None:
    """The moved map through a session: at each step t the pass at t (its change `Replay.take(t)`), then the append t
    made, answered with its coin, then line t's releases, its withdrawn blocks named (`Session.release(released,
    withdrawn)`); a window ends before each step in `boundaries` and after the last.  The first rejection, if any."""
    replay, lines = Replay(Header(run.store.registration.digest, N), run.store.registration), {s.step: s for s in dm}
    digests: dict[int, bytes] = {}
    for t in range(steps):
        if t in boundaries:
            run.boundary()
        if (v := run.run(change=replay.take(t), step=t)) is not None:
            return v
        if t in appends:
            a = run.store.append(appends[t], salts(b"append at step %d" % t, len(appends[t])))
            assert run.commit(a.root, a.count) == a.first
            digests.update(zip(range(a.first, a.first + a.count), a.digests))
        if t in lines:
            c = replay.apply(lines[t])
            assert c.digests == tuple(digests[j] for j in c.indices)
            if c.released and (v := run.s.release(c.released, c.withdrawn)) is not None:
                return v
    run.boundary()
    return None


def test_a_block_freed_before_its_encode_lands_is_withdrawn_in_the_line_its_write_would_be_in() -> None:
    """Step 0 completes blocks in slots 4, 5 (filler, freed at their reach) and 100, one append landing in step 2's
    line; slot 5's values' last structural reader is step 1, so step 2 writes 8 and 10 and withdraws 9, in no plan.
    Step 3's block lands at once, at 11; step 4 is 8's and 10's last structural reader."""
    oracle = [Step(0, (), (Written(4, N, d(4, 0)), Written(5, N + 1, d(5, 0)), Written(100, N + 2, d(100, 0)))),
              Step(1, (5,), ()), Step(3, (), (Written(5, N + 3, d(5, 3)),)), Step(4, (4, 100), ())]
    dm = deferred(oracle, [[0, 2, N, 3], [3, 3, N + 3, 1]], 0, N)
    assert dm == [Step(0, (4, 5), ()), Step(2, (), (Written(4, N, d(4, 0)), Written(100, N + 2, d(100, 0))), (N + 1,)),
                  Step(3, (), (Written(5, N + 3, d(5, 3)),)), Step(4, (4, 100), ())]
    held = plans(dm, 6)
    assert (held[4], held[5], held[N], held[N + 2], held[N + 3]) == ([0], [0], [3, 4], [3, 4], [4, 5])
    assert N + 1 not in held


@pytest.mark.parametrize("last_read, line, held", [
    (9, Step(7, (), (Written(4, N + 1, d(4, 0)),)), [8, 9]),
    (6, Step(7, (), (), (N + 1,)), []),
], ids=["read-after-landing", "read-before-landing"])
def test_a_re_key_frees_its_old_index_at_its_step_and_writes_the_new_one_when_its_encode_lands(last_read, line,
                                                                                              held) -> None:
    """Block 8 in slot 4 is re-keyed at step 5 (its payload appended again, at 9) and the re-key's encode
    lands in step 7's line.  Step 5 frees the slot, an exception to a free at the last structural reader, so 8 is in
    the passes through step 5, neither is in those at 6 and 7, and 9 is from step 8 through its last structural reader;
    with that reader before the re-key landed, 9 is withdrawn."""
    oracle = [Step(0, (), (Written(4, N, d(4, 0)),)), Step(5, (), (Written(4, N + 1, d(4, 0)),)),
              Step(last_read, (4,), ())]
    dm = deferred(oracle, [[0, 0, N, 1], [5, 7, N + 1, 1]], 0, N)
    assert dm == [Step(0, (), (Written(4, N, d(4, 0)),)), Step(5, (4,), ()), line] + (
        [Step(9, (4,), ())] if last_read == 9 else [])
    got = plans(dm, 11)
    assert (got[N], got.get(N + 1, [])) == ([1, 2, 3, 4, 5], held)


def test_deferred_a_re_key_falls_at_age_l_minus_one_so_a_lifetime_of_one_credits_nothing() -> None:
    """L = 2 over three windows of three passes: block 8, appended after step 1 (window 0, which it joins),
    lands in step 3's line, so it is resident from window 1, and is re-keyed at step 5, window 1's last, at age
    L − 1 = 1; the re-key (9) joins window 2 and is resident from step 7.  The session is accepted.  At L = 1 the block
    is stale in the window it becomes resident in (`lifetime`), so a session that defers residency is refused at
    L = 1 before it starts."""
    oracle = [Step(1, (), (Written(4, N, d(4, 1)),)), Step(5, (), (Written(4, N + 1, d(4, 1)),))]
    dm = deferred(oracle, [[1, 3, N, 1], [5, 6, N + 1, 1]], 0, N)
    assert dm == [Step(1, (4,), ()), Step(3, (), (Written(4, N, d(4, 1)),)), Step(5, (4,), ()),
                  Step(6, (), (Written(4, N + 1, d(4, 1)),))]
    assert (plans(dm, 9)[N], plans(dm, 9)[N + 1]) == ([4, 5], [7, 8])
    appends = {1: [page(4, 1)], 5: [page(4, 1)]}
    run = Run(store(), windows=3, samples=40, lifetime=2, deferred=True)
    assert drive(run, dm, appends, 9, boundaries=(3, 6)) is None
    assert [v.accepted for v in run.finish()] == [True] * 3
    assert [w.post_seed for w in run.s.record().windows] == [(0, 0, 0), (0, 1, 1), (0, 1, 1)]
    run = Run(store(), windows=3, samples=40, lifetime=1)
    v = drive(run, dm, appends, 9, boundaries=(3, 6))
    assert (v.failure, v.window) == ("lifetime", 1)
    with pytest.raises(ValueError, match="at least 2"):
        Run(store(), windows=3, lifetime=1, deferred=True)


def history(seed: int, steps: int = STEPS):
    """A serving history over `POOL`, as circuits' oracle records it: at each step some live blocks' values are read for
    the last time (freed), some resident ones re-keyed, and new blocks completed in free slots, all one append whose
    encode lands 0 to 3 steps later in append order, or after the session; none at the last step, which no root
    follows.  Returns the oracle's steps, the encodes, each append's payloads, and each index's (step completed, line
    landed, step freed or re-keyed, whether it is a re-key)."""
    rng = random.Random(seed)
    free, live = list(POOL), {}
    oracle, encodes, appends, life = [], [], {}, {}
    index, last = N, 0
    for s in range(steps):
        freed = sorted(x for x in live if rng.random() < 0.3)
        for x in freed:
            life[live.pop(x)[0]][2] = s
        free = sorted(set(free) | set(freed))
        rekey, new = [], []
        if s < steps - 1:
            rekey = sorted(x for x, (j, _) in live.items() if life[j][1] is not None and life[j][1] < s
                           and rng.random() < 0.15)
            new = rng.sample(free, min(len(free), rng.choice((0, 1, 1, 2, 3))))
        if not rekey and not new:
            if freed:
                oracle.append(Step(s, tuple(freed), ()))
            continue
        last = max(last, s + rng.choice((0, 0, 1, 2, 3)))
        land = last if last < steps else None
        slots = rekey + new
        payloads = [live[x][1] for x in rekey] + [page(x, s) for x in new]
        for x in rekey:
            life[live[x][0]][2] = s
        oracle.append(Step(s, tuple(freed), tuple(Written(x, index + u, codec.digest(m))
                                                  for u, (x, m) in enumerate(zip(slots, payloads)))))
        encodes.append([s, land, index, len(slots)])
        appends[s] = payloads
        for u, (x, m) in enumerate(zip(slots, payloads)):
            live[x] = (index + u, m)
            life[index + u] = [s, land, None, u < len(rekey)]
        free = [x for x in free if x not in new]
        index += len(slots)
    return oracle, encodes, appends, {j: tuple(v) for j, v in life.items()}


@pytest.mark.parametrize("seed", SEEDS)
def test_a_moved_map_writes_or_withdraws_each_landed_block_once_resident_from_its_line_to_its_last_reader(seed) -> None:
    """Withdrawn exactly when its free (last structural reader or re-key) is at or before its line; written, in the
    plans of the passes after its line through its free; a block whose encode never landed in none; a filler slot in
    the plans through the step that reached it.  `Replay` accepts every line."""
    oracle, encodes, _, life = history(seed)
    dm = deferred(oracle, encodes, 0, N)
    held = plans(dm, STEPS)
    written = [w.index for s in dm for w in s.written]
    withdrawn = [j for s in dm for j in s.withdrawn]
    assert sorted(written + withdrawn) == sorted(j for j, (_, r, _, _) in life.items() if r is not None)
    for j, (_, r, g, _) in life.items():
        if r is not None and g is not None and g <= r:
            assert j in withdrawn and j not in held
        elif r is not None:
            assert held.get(j, []) == list(range(r + 1, STEPS if g is None else g + 1))
        else:
            assert j not in held
    for x in range(2, N):
        reach = min((s.step for s in oracle for w in s.written if w.slot == x), default=STEPS)
        assert held[x] == list(range(min(reach + 1, STEPS)))


@pytest.mark.parametrize("seed", [0, 1])
def test_a_session_a_moved_map_drives_is_accepted(seed) -> None:
    """Both sides driven by a seeded history's moved map, withdrawals and re-keys among its lines (a pass a step, its
    appends after its root, its releases naming their withdrawn blocks): every window is accepted, the record counts
    the withdrawn blocks released, and each pass's head is the chain through the appends of the steps below its own,
    each append dated at the step that made it, not the line its encode landed in."""
    oracle, encodes, appends, life = history(seed)
    dm = deferred(oracle, encodes, 0, N)
    withdrawn = [j for s in dm for j in s.withdrawn]
    assert withdrawn and any(k for *_, k in life.values())
    run = Run(store(), samples=40, deferred=True)
    assert drive(run, dm, appends, STEPS, boundaries=(STEPS // 2,)) is None
    assert [v.accepted for v in run.finish()] == [True, True]
    rec = run.s.record()
    assert rec.appended == sum(len(m) for m in appends.values()) and rec.released >= len(withdrawn)
    made = [(s, a.first, a.count, a.root) for s, a in zip(sorted(appends), run.store.appends, strict=True)]
    assert units_mapped(rec, made)
    landed = [(s if land is None else land, *rest) for (_, land, _, _), (s, *rest) in zip(encodes, made, strict=True)]
    assert landed != made and not units_mapped(rec, landed)


def test_the_seeded_histories_reach_every_rule() -> None:
    """Over `SEEDS`: withdrawn blocks, among them a re-key's new index and one withdrawn beside a block its line writes;
    re-keys; a filler slot whose first block lands at the step that reached it, so its line's write absorbs the free;
    and an append whose encode never landed."""
    reached: dict[str, int] = defaultdict(int)
    for seed in SEEDS:
        oracle, encodes, _, life = history(seed)
        dm = deferred(oracle, encodes, 0, N)
        dead = {j for j, (_, r, g, _) in life.items() if r is not None and g is not None and g <= r}
        reached["withdrawn"] += len(dead)
        reached["re-key"] += sum(k for *_, k in life.values())
        reached["re-key withdrawn"] += sum(life[j][3] for j in dead)
        reached["withdrawn beside a write"] += sum(bool(s.withdrawn and s.written) for s in dm)
        reached["never landed"] += sum(r is None for _, r, _, _ in life.values())
        for x in range(2, N):
            first = next((w.index for s in oracle for w in s.written if w.slot == x), None)
            reached["filler free absorbed"] += first is not None and life[first][0] == life[first][1]
    assert all(reached[k] for k in ("withdrawn", "re-key", "re-key withdrawn", "withdrawn beside a write",
                                    "never landed", "filler free absorbed")), dict(reached)
