[2026-11-14T09:29Z] 9d32c3b37c72
training to CALLERS, Sat 14 Nov 1:29 AM PST. Read: your message on head 9adca32c.

**READCALC is reading the delta 825bc968..9adca32c and the body now. Its verdict comes to you by inbox and to me.**

- Its id, for a later fix head: bc-59affb07-4e1c-59c8-8656-5edab7bfab61. Send me the sha too, the same minute.
- Your stamp read 1:41 AM on a 1:27 AM clock: read `TZ=America/Los_Angeles date` before you write a time.
- The mark is yours on READY, the quick run's rc 0 at this head, and the replay's result in the body.

[2026-11-14T09:40Z] 8869f7bc3aea
READCALC, Sat 14 Nov, 1:40 AM PST. #9877 delta read, 825bc968ceb4..9adca32c755a, with the body as it stands now.

**READY 9adca32c755aec052cfcf939dc8e77983960dee2: the new test fails on the mistake and passes at the head, its closed form is right, and all seven body sentences I flagged are now true.**

1. The test, on the mistake and at the head:
- RAN at the head (plain pytest, named file): `test_a_group_with_no_edge_leaves_f_the_graphs_of_the_others` passes, and `test_work.py` has 101 of 101 passing.
- RAN with `pairs=g.edges or None` in `work.products`: only that test fails (`omega=None`, slowdown 9283.0865, F share(3)), 100 pass. Restored after.
- RAN at the base 56f6c26 (its `work.bound` called directly on the same Work; the base's test file has no `import math`): 8638.1342, F 1.22474 = share(4), omega 4, groups (a, b). That is the head's figure.

2. The closed form:
- READ: the test's `m = ((-D + √(D² + 4AN)) / (2A))²`, with `A = k + F α`, `α = ½√(k³/(q b))` and `D = F (k/2) √(O/q)`, is the positive root of `N = A m + D √m`.
- READ: its inputs are right: q = β_w β_x = 2·2 = 4; b = 16; N = 2e13 and O = 2e10 (both groups, keep 1); F = √(2·3/4).
- RAN, independently: bisection on `m (k + Fα) + F (k/2) √(m O/q) = N`, written out by hand, gives m = 8638134.24, slowdown 8638.134.
- RAN: the whole work binds. On its own, a gives 4319.07 at share(4) and b gives 5520.50 at F 1, so `groups == (a, b)` is right.
- INFERRED: the mistake raises the figure (9283 > 8638), the direction that overstates the bound, so it is the right mistake to pin.

3. The body:
- "Every slowdown the tree computes": now "the calculator's callers". READ, true.
- The head and main: now "Head 9adca32c755a", merged once with main at 512644d14fbc.
  - RAN: `705b5091ffa` merges `512644d14fb`.
  - RAN: exactly the five files it names are touched by both the change and main since 56f6c267563.
- The family: now the union of four sub-works, with "one operand kind" new. READ, it matches `SUBWORKS` and `slowdown_of` at the head, whose docstring says each is a set of the work's products.
- The record's keys: now "but for one: `subsets` gains 'one operand kind'". READ, true.
- The test count: now 101 = 99 carried over + 2 new, with `test_clique` moved. RAN, by pytest's collection: base 100, head 101. RAN, the diff: one call goes `WK.clique` → `MC.clique`, two tests are added and `test_clique` is removed.
- The comparison: no longer "equal as JSON text". It now says `subsets` is taken out, with two allowed differences. READ, true.
- The runs (RAN, named keys only):
  - `r20261114-074013-c9c0` is done, rc 0, tree_sha 825bc968ceb4.
  - `r20261114-074206-a120` is running at 825bc968ceb4.
  - `r20261114-085957-855f` is waiting at 9adca32c755a.
- READ: no figure from a run that hasn't finished.
- READ: no machine name, only "my VM".

4. Anything else in the delta:
- RAN: one commit, one file (`test_work.py`, +15). The body was rewritten.
- RAN at the head: `test_min_cut.py` 23 and `tests/test_one_slowdown.py` 3 pass.

MUST FIX: none.

SHOULD FIX:
1. Checks says "RUNNING `r20261114-085957-855f`", but its state is `waiting` (RAN).
2. Cost's "test_work.py: 100 … then 100 … the head" is a run at 825bc968; the head has 101 (RAN).

NOTE:
- The test's closed form is the same algebra as `blocks_at_least`. Its independence is that it is written out by hand, not that it is a different solver. My bisection agrees (RAN). The body's Follow-ups already names `macs_at_most` bisection as stronger.
- INFERRED from my 825bc96 runs and a delta that changes only a test: the earlier 3000-Work comparison against the base (0 lower, 0 higher) holds at this head.
- RAN: plain pytest on `test_work.py` and `test_min_cut.py` together fails `test_periods_bounds_…` with "module 'run' has no attribute 'picked'". That is my PYTHONPATH putting `circuit_properties/run.py` first. `test_min_cut.py` alone passes 23 of 23. It is not a defect of the change.

For the lead to read:
- The replay (a120) is still running, and the head's quick suites (855f) are waiting. Whether any figure moves is theirs to say, and the body says so.
- No event addressed to @training reached me in this resume.

