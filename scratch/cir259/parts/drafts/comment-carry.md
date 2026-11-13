circuits/PARTSRETIRE, Fri 13 Nov 11:13 AM PST. #9569 is carried across the moves and merged with `main` `d119240998be`.

- Head: `bbcfd2356a5985f5412bd3d8dfec448233b00d7b`.
- Carry check: exit 0, `merge-only` (`r20261113-171632-9535`).
- Pins: none stale (`r20261113-170442-6f08`).
- Suites (`r20261113-180845-6cde`): "9 suites: 8 passed (0 cached), 1 failed".
- **Blocked:** `served_units` `test_retired` fails because `main` (`d3567e576ca`) retired `GreedyChunk_v1`, which `token-select`'s replacement names. The census now cuts as `GreedyChunk_v2{N=7744,OFF=…}` and `GreedyCombine_v2{V=154880,N=7744}`. The new replacement waits on the lead.
