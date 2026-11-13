# CIR-274 piece 2: tp_binding takes a combine's output read inside a batch node

Thu 12 Nov 8:04 PM PST (04:04Z). PR #9627, branch `cursor/circuits-cir274-tp-batch-reader-6f04`, head
b90c374bd6e072fe1b3c0ed5635ef14202ca2451 (3266c6ee merged with `main` at 1e4d5c99). Open, unmarked.

## Collect runs

Source 3ec2a17bc8457aedc69927bc213078af906f6833 (b90c374b merged with this branch at c3001da7, pushed as
`cursor/circuits-cir274-c43m-6f04`). Class G, two outputs. Every run: `unwired` {}, `proof_blocks` {}, no backward wire.

| prompt ids | layers | driver | run | sites | wires |
|---|---|---|---|---|---|
| 1 | 4 | collect43.py | r20261113-033036-b53c | 20 | 768 |
| 2 | 4 | collect43.py | r20261113-033106-2ef2 | 26 | 1420 |
| 8 | 4 | collect43.py | r20261113-033134-59d5 | 62 | 1421 |
| 43 | 4 | collect43.py | r20261113-033148-defe | 20 | 724 |
| 2 | 78 | collect_par.py | r20261113-033217-fd7e | 470 | 2144 |
| 8 | 78 | collect_par.py | r20261113-033248-273e | 1394 | 2144 |
| 43 | 78 | collect_par.py | r20261113-033320-3992 | 316 | 937 |

Each run's result fields equal the same case's from the trees before the merge with `main` (3266c6ee plus this branch).

## collect_par.py against collect43.py

- `collect_par.py` equals `collect43.py` in every result field at 4 layers: LP2 (r20261113-023621-37c1 against
 -015817-75b2) and LP8 (r20261113-031735-fd9e against -015844-09ea).
- With `interface.layout` cached and without it, LP8 at 78 layers is equal in every field:
 - without the cache, r20261113-031018-5344 (3042 s, 2481 s of it in `sites_circuit`);
 - with it, r20261113-033248-273e.
- Profile at LP8, 4 layers (r20261113-031049-1a2b): `interface.layout` is 163 of 171 s in `main`'s `sites_circuit`.
- At 78 layers, the 8 rank compiles run serially (as `collect43.py` does) and the uncached `sites_circuit` together take
 longer than a 30-minute run. An earlier driver at LP8, r20261113-023719-bb08, timed out.

## Suites

r20261113-033011-70e8 at b90c374b:
- `repository`: 388 passed, 0 failed, 0 skipped.
- `verity-vllm`: 14603 passed, 0 failed, 479 skipped.
