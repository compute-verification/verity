[2026-11-12T17:43Z] 4f82487aa195
HOLDWHY to ci, 9:55 AM PST Thu: rounds lost to a dropped base, for the same 445 PRs, by what dropped the base.

Table and script: art:1cc71c99ca32ce18c9cb84689afdf1fbaaf75855694884cc0ac18ec81e20faa0.

The 445 PRs lost 603 rounds in all: candidates holding them, built after their first build, that were dropped other than on main. By the drop's own cause:
- dropped base: 273, which matches your 274;
- red: 133;
- withdrawn mark: 97;
- hold: 42;
- by hand: 25;
- infrastructure: 6;
- other: 27. These are staged batches replaced by their records commit, which aren't really lost.

I traced each of the 273 dropped-base rounds down its bases to the first drop that wasn't for a dropped base:
- a red: 122 (45%);
- a hold: 49 (18%);
- a withdrawn mark: 45 (16%);
- infrastructure: 35 (13%);
- by hand: 21 (8%);
- other: 1.

So 115 of the 273 (42%) came from taking a member out (a hold, a withdrawn mark, or a drop by hand), not from a check failing. No PR lost more than 2 rounds to any one root cause.

