# HONEST2: a two-stage registration and an accepted run at which every premise of the two-stage guarantee holds

Read `COMMON.md` beside this file first, then `## FRAME, Thu 12 Nov 21:10Z` (section a, its last two bullets, and b "L4").

**The gap.** A theorem with premises is worth what its premises allow. The lock lists both two-stage guarantees as unwitnessed ("missing: honest", "honest instance needs a built two-stage layout and frame"). An accepted run needs session statements inside Flock's zero-knowledge scope (each table's `m` from 25 to 35), and no held code in the tree has one. #8363 (`cursor/proofs-twoinstance-2a66`, open, `SpotFlock.Instance`) has a registration with the design and `Built`, "non-vacuity open". Worse, FRAME's entry cites two theorems that the address premise is false at the design served today (`not_addresses_of_input`, `SpotFlock/AddressesAt.lean:321`; `not_addresses_of_readsInterior`, `:293`), because a replay block's inputs are registered after the first stage's sample. So before anyone says "proved", someone must show the premises can all hold together at a design we can run.

**Yours.**
1. **By 23:30Z Thu, a map:** the list of every premise and domain condition of `SpotKSTwo` and of `ksTwo_connected` (on the base: `SpotFlock/WiredKS.lean`), each with: is there a registration in the tree at which it is proved to hold; if not, what the smallest one is. Read #8363 whole and say what it has and what stops it. Say what design makes the address premise true: by FRAME's reading, one where serving commits a replay block's inputs and its return as ports of round 1's record, before the first sample. Write that design down exactly (which ports, in which record, under which link), since the serving team is being asked for it tonight and needs the exact shape. Put that part under `To post`.
2. **Then the instance:** in new files, a registration of the smallest replay block that can be run (the token select, as Friday's run stages it: 19 chunks and a combine, or smaller if the theorem allows) with its inputs committed at serving, at which the design conditions and `Built` hold, and the witness theorems the lock's `honest` entries need. Whether a `deployed` entry in the lock requires the honest instance is not known: read `tools/verity/lean/check.py` and the lean-proofs skill and say.

**Not yours.** The proofs of the premises from the verifier's checks (ADDRWC, ADDRROW, RUNSAS, RUNCOINS, FRAME). FRAME's files (`SpotFlock/Code.lean`, `Pin.lean`, `AddressesWired.lean`, `RevealsWired.lean`, `Spot/SecondStage.lean`, `second_stage.py`, `TwoGuard.lean`, their tests). #8363's branch: read it, and build on a branch of your own.

**Base.** `origin/cursor/proofs-two-round-base-84b5` (today `bb527fd2e2d2`; merge it in when it moves, never rebase).

**Owed.** 20 min: first entry. 23:30Z Thu: the map, with the serving design under `To post`. 03:00Z Fri: the instance's signature elaborating and your size. Then the witnesses.
