# ADDRWC: the width and the copy facts of the address premise

Read `COMMON.md` beside this file first, then `## FRAME, Thu 12 Nov 21:10Z` (sections a.1, b "L2", d).

**The gap.** The two-stage guarantee keeps one open premise that the verifier's checks do not yet give: every gate's value has one address, fixed before the run, that holds at every accepted run's checked unit that needs it (`SpotFlock.Addresses`, `security/Proofs/Flock/SpotFlock/SlotsAt.lean:322`; over the connected program `SpotFlock.AddressesConnected`, in `AddressesWired.lean` on the base). Its docstring (`SlotsAt.lean:36-52`) lists five facts the checks must give: the width, the copy, the row, the entry, one tree a gate. FRAME writes the five as `Prop`s in one file on the base within the hour (`SpotFlock/AddressFacts.lean`), with the theorem that the five give the premise; until it is there, write your two from the docstring and reconcile.

**Yours.** The width and the copy, each as a theorem of an accepted session: from the session circuit's copy rows, that a value opened for a gate has the width the address says and is the copy the address names. New files only, under `security/Proofs/Flock/SpotFlock/` (suggested `AddressFacts/Width.lean`, `AddressFacts/Copy.lean`), reading `RowsOpen.lean` and `RevealAt.lean` (`RowAt`, near line 338) and whatever of Flock's verifier's own lemmas about copy rows exist (find them; `verity/protocols/flock/verifier/` and `security/Proofs/Flock/Bridge/`). They do not wait on the choice of statement form. Your first map: the exact statement of each, what an accepted session gives toward it (file and line), and what is new.

**Not yours.** The row (ADDRROW). The entry, one tree a gate, and putting the five together (FRAME). Any edit of FRAME's files: `SpotFlock/Code.lean`, `Pin.lean`, `AddressesWired.lean`, `RevealsWired.lean`, `Spot/SecondStage.lean`, `second_stage.py`, `benchmarks/two_stage/tests/TwoGuard.lean` and their tests. If either fact is false of what the verifier accepts today (a session that passes and opens a value at another width or copy), build the smallest counterexample in words and report it at once under `GAP` with the check that would close it.

**Base.** `origin/cursor/proofs-two-round-base-84b5` (today at `bb527fd2e2d2`; merge it into your branch when it moves, never rebase).

**Owed.** 20 min: first entry. By 23:30Z Thu: the two statements elaborating and the map. By 03:00Z Fri: one compiled and your size. Then the other.
