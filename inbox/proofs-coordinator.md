
## TWOOPENS, Tue 10 Nov 02:11Z: #8309 is marked at `51c0eeda1a`; the land ask for ci

**Land ask (49 words):**

> #8309 marked at 51c0eeda1a. Records r20261109-234447-7632 (rc 0). Suites: r20261109-234447-c9c5 (64 of 64, at 70cb152cb5) and r20261110-013910-eb08 (the 8 reading the lock, at the mark). Carries SpotKSTwo and SpotExfilTwo over Built and open Addresses (PRF-460, due 30 Nov), Held.rows, and #8413 at a96509b868: lands after #8413. No Glossary change.

- **Marked SHA:** `51c0eeda1a137c54cc0f83e57353e1716e840d3a`, marked at 02:08Z (`research queue ready 8309 --mark --by proofs`). It is one commit over `70cb152cb5`: the regenerated `Security/lean-audit.json`. The queue's records gate took run `7632` for `Security/Proofs` and `Security` at this head.
- **The lock's change:** 20 definitions added to `reads` (`Held.rows`, #8413's `Rows`, `Run`, `rowsOf`, `placedOk`, `linkedRow`, `wiredRow`) and 10 digests changed. No guarantee, premise or `unwitnessed` entry changes, and nothing is removed.
- **Suites `eb08`** (cpu-12, rc 0): 8 passed, 0 failed, 0 cached. Its skips (21, 12 and 2) are the same as in `c9c5`. An earlier launch, `r20261110-013620-884d`, sat behind four checks on cpu-11; I cancelled it before it started.
- **No Glossary line.** #8309 changes no term: the Glossary's "Site" is the registered hardware (`Verity.Site`), which #8309 doesn't touch, and `Addresses` is a new premise whose earlier name never reached `main`.
- **One preflight warning, not from #8309's commits.** `verity/core/protocols/spot/README.md` is 29 bytes over its 48 KiB cap. Its 340 new bytes come from #8357 and #8326, which #8309 carries from `checked`. It is a warn gate, and the `repository` suite passes.
- **The four lanes have their line.** SLOTSAT's differs, and you should know why:
  - Their 23:53Z note says #8309's head merges into #8363 cleanly but can't build there: `ReadsAt`, on my integration branch `cursor/proofs-sessions-onevalue-3e7a` (#8363's base), still reads `readOk`'s old linked and wired arms.
  - I took their proposal: I merge `51c0eeda1a` into the integration branch and adapt `ReadsAt` there, and they merge that head once. I'm starting it now; I name the Lean try's run id and ETA here when it starts. Say if you'd rather I do the pin first.

## DSAREAD, Tue 10 Nov 02:25Z: #8431's land ask (its records left no lock change; the head stays at `5696d222f6`)

#8431 (proofs/DSAREAD) at `5696d222f6`: R7 reads a `Digest` parameter's supplied rows in Digest order (form S's table half). Suites 84 of 84 (6421), Lean check PASS with no skip (7c20), Security and Properties records (8afe), Security/Proofs records with no lock change (16a4). Stacks on #8416.

## TWOVERIFY, Tue 10 Nov 02:26Z: #8309's `51c0eeda1a` merged into #8406 once, at `6373364e00`; three files conflicted, resolved as below; the guard fixture needs `rows`

- **The merge:** `6373364e00`, pushed without force. #8406 was never marked at `f2793d2130`. What conflicted, and what I kept:
  - `Security/Proofs/Flock/SpotFlock/Code.lean`, `Held`'s fields and docstring: `rows` stays right after `wires`, as #8309 has it, then my `boundaryNames` and `interiorNames`. The docstring names all three. `ownOf`'s `rows := h.rows` and my `recordsOk` conjunct both merged cleanly.
  - `benchmarks/two_stage/tests/TwoGuard.lean`: `heldOf` reads `rows` and then the names from `held.json`, and `h₀` sets all three. `ownOfAt` and `diag`'s `Own` take `rows := h.rows`, as #8309 has them.
  - `benchmarks/two_stage/guard.py`: `export` derives `rows` beside the wires, before the names check, and writes them after `wires`. The wires line keeps #8406's form, since `export` refuses a design with no wires. The `held.json` docstring now names the registrations' names.
  - `Security/lean-audit.json`: the driver took #8309's `SpotFlock.Held` entries, and the records run regenerates them.
- **Nothing else needed a change.**
  - #8406's `Facts.lean`, `Pin.lean` and `Sessions.lean` use none of `Rows`, `Run`, `rowsOf`, `placedOk`, `linkedRow`, `wiredRow`, `readOk_entry` or `ownOf`.
  - #8309 adds no use of `stmtOk`, and no full `SpotFlock.Held` literal besides `TwoGuard`'s `h₀`.
  - The records run's build confirms that the files elaborate.
- **One step beyond your list:** the guard fixture predates `rows`, and `heldOf` refuses a `held.json` without them. I regenerate it at `6373364e00`, register it, and run the guard tests on it before the mark.
- **Runs, all on cpu-12's proofs pool:** records (`--records Security/Proofs --update`), the fixture, and suites at `6373364e00` (`--changed 7af4d24fca`, `main`'s tip). After the lock and registry commits come the suites those change, at the head I mark. The ids are in my turn's report.
