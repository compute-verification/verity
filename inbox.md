[2026-11-14T04:40Z] 35b35dda3e0f
## Coordinator, Sat 14 Nov 04:40Z: RELAX: the move's place changed (not compose/)

RELAX — the move's place CHANGED (top on TOP-415, Fri 8:36 PM PST, after Daniel 8:34 PM PST). Read this before you push or note anything on the move.

1. NOT under `compose/`. Daniel: "I thought compose would have no notion of what it was composing." A list member inside `compose/` makes compose hold a thing that knows SPOT's record. Do not open, push or note a move to `verity/protocols/compose/verifier/spot_caller/`. If you already pushed a branch for it, leave it unmarked and say its name in your note; I mark nothing at that place.

2. The new place: a directory of its own in the proofs group, beside `spot/` and `recursion/` and above both (it requires both; neither requires it). Add its `[[tree]]` row to `tests/tree.toml` in the move's own PR: top, the file's owner, said yes to that row on TOP-415 and nothing waits on top.

3. The name: for what the package is on a list, not for who calls whom (`spot_caller` describes a call). On the lists it is the member `verity/sampled-proofs/v1` (and it carries `verity/sampled-proofs/recursive/v1` with it). Choose the directory's name from the architecture file's and the glossary's words; check it against the glossary's retired table and the retired-terms ratchet before you use it. Write the name and its one-line reason at the top of your note the moment you have chosen, then go on without waiting for me; I answer only to change it.

4. Redo the map, `move_check.py`, and the importers' runs at the new place. The 20 path lines elsewhere change to the new path. The PRs that cross it (#9873, #9817, #9738, #9574) are unchanged in what they need. New ready note for the move: by Sat 14 Nov 12:00Z, or say the hour you can hold.

5. #9849 is separate and unchanged: ready note at `318346c4d95b` when `r20261114-040300-4532` ends. ci warned that a branch behind #9751's move (`ac8b3b51e490`) needs `python3 tools/move/restack.py BRANCH --main ac8b3b51e490` before `main` is merged, when its own change names the retired names; say in the ready note that you checked (yours merged `main` after the move landed, so say what `restack.py` answers).

6. Monday's sizing note (by Mon 16 Nov 20:00Z) now answers three things beside the third change's size and day:
   (a) the recursive audit: split back out of the package to recursion, or the architecture file's row changes;
   (b) top's hinge question, PER FACT of `CallerHeld`: does the thin protocol take it, or does it go to the protocol whose fact it is. Top's reading: the declared day is registration's, the seed salt is PoUW's noise, the pool and its weekly reset are the sampling's own, the stream and the prover are the run's. "If it takes all, it is a second hub in place of the first." A table: fact, today's check (file:line), whose it is, where its check would live, what reads it;
   (c) A is done only when the checks themselves have moved (the third change on `main`).

Notes sync: use main's tool (`/workspace`'s is refused by GitHub): from a checkout of origin/main, `python -m research notes sync --path lanes/proofs`.

