<decision>
One line of `benchmarks/served_units/stage.py`: `RETIRED["token-select"]` now names `census:GreedyChunk_v2{N=7744,OFF=7744}` and `census:GreedyCombine_v2{V=154880,N=7744}` instead of the v1 pair, which main's `d3567e576ca` retired. That broke `test_retired` after the carry.

The circuits lead chose this shape. `OFF=7744` names a chunk that isn't the first, as the old `FIRST=false` entry did. The census itself has 19 chunk positions and one combine.

Alternatives set aside:
- Naming every chunk position. The replacement only has to be one stageable instance of the cut.
- Rewording the reason string. The lead kept it.

No README, docstring or test stated the old names, so the commit is that one line.
</decision>
<risk>
- Only `OFF=7744` was staged and held to Lean at this head, at one unit. The other 18 positions weren't.
- The full suites run at the previous head timed out with 10 suites unfinished: `verity-vllm`, `verity-flock`, and 8 that never started. The body says those have no verdict at either head; the merge check will be their first.
- The carry check's exit 0 is at `bbcfd2356a59`, not at the head, since a hand commit sits on top.
</risk>
<verified>
- `served_units` and `repository` suites at `c9dcea5b385`: 48 and 361 passed (`r20261113-191707-881e`).
- Staging run `r20261113-192328-bcec` at `c9dcea5b385`, one unit per type, bound: `verify --zk` exits 0 with the Program and 1 with no Program and with another Definition's, for both types; peak 1.8 GB.
- The ratchet gate exits 0 with `"failures": []`. The retired-terms lint's `sites()` finds the one `caps` site main's `stage.py` also has.
- `git merge-tree` with `origin/main` `f479876a8758` is clean.
- The author and committer of all 87 commits over `origin/main` are `Cursor Agent <cursoragent@cursor.com>`, the checkout's configured identity.
</verified>
