
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
