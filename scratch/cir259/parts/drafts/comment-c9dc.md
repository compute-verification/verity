circuits/PARTSRETIRE, Fri 13 Nov 12:32 PM PST. #9569's head is `c9dcea5b385c`. `token-select` now stages `GreedyChunk_v2{N=7744,OFF=7744}` and `GreedyCombine_v2{V=154880,N=7744}`, and `test_retired` passes.

- Both types were staged bound and accepted (`r20261113-192328-bcec`).
- `served_units` and `repository` pass (`r20261113-191707-881e`).
- The full suites run (`712b`, at `bbcfd2356a59`) timed out: 119 passed, `served_units` failed (now fixed), and 10 were left unfinished.

The PR isn't marked.
