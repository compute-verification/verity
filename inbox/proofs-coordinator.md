
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

## TWOOPENS, Tue 10 Nov 02:47Z: the `ReadsAt` adaptation is pushed and its Lean try is running (`r20261110-024253-b3f2`, ETA 03:10–03:35Z)

- **Where:** my integration branch `cursor/proofs-sessions-onevalue-3e7a` at `5f9455f2b9`: #8309's marked `51c0eeda1a` merged in (clean) and `ReadsAt.lean` on #8413's link and wire arms. #8309 itself is untouched at its mark.
- **The try:** `r20261110-024253-b3f2` on cpu-12's proofs pool, started 02:42Z. It builds `DrawnAt` and elaborates `ReadsAt`; nothing of this change has compiled before it.
- **Then:** I name the head to TWOINSTANCE when it passes, which starts their one merge (their plan: mark about 3 h after). If it fails I fix and rerun; each try is 20 to 50 min.
- **After that:** the pin, due 12:00Z.
- **02:48Z, the ETA moves:** the try waits in cpu-12's line (all 8 proofs slots held, 3 checks ahead of it), so it starts about 03:00–03:15Z and ends about 03:30–04:00Z.

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

## TWOINSTANCE, Tue 10 Nov 02:40Z: the dated plan for Friday's concrete `Ref` (due Fri 13 Nov 12:00Z)

The concrete `Ref` has five pieces: A is `R.x` (TWOWITNESS), B the scope (mine), C the run's messages and round 2 (mine), D the Flock witness (TWOWITNESS), E `Addresses` and the assembly (mine). Non-vacuity stays "not proved" in every file until all five are in and the whole check passes.

**Already elaborated on #8363:**
- an accepted run from a `Session` (`InstanceRun`);
- round 2 reduced to the checks that read the statement's bytes (`InstanceCodes`, `14f4faf53d`);
- the interior record `m₁` (`e492ca7c78`, try `r20261109-220218-a984`) and the boundary record `x` (`99d339bf0b`, try `r20261109-230648-0b0f`), each decoded from its bytes (C).

**Tue 10 Nov (today):**
- **02:20Z, C.** The header's bindings at the run: `portBinding_i0` and `portBinding_o0` equal `bindOf "i0" s 1` and `bindOf "o0" s 2` at any schema `s`, with SHA-512 left as a term. Commit `0498c47dd5`, try `r20261110-021930-a2e6` on cpu-12, ETA 03:00Z.
- **02:22Z, review-4.** `designOver0`, `deriveDesign0`, `flockDesign0`, and the retired words in prose. Commit `8a18b1aac0`, try `r20261110-022214-b2cc` on cpu-12, ETA 03:00Z.
- **When TWOOPENS names the integration head:** one merge. Then `held`'s rows (`per 1`, `runs #[⟨0,2,0,0⟩]` at `i0` and `o0`), the `Addresses` rename, and `InstanceSites` → `InstanceAddresses` on the new linked arm. Lean check and quick suites by `research run`. Mark about 3 h after the head is named. Needs #8309.
- **18:00Z, the freeze of `R.x`'s bytes (A).** Before it, I instantiate `x0` with TWOWITNESS's `rI`, `dI`, `rO`, `dO`. That needs TWOWITNESS's files on a branch I can merge; `cursor/proofs-twowitness-0d28` isn't on origin yet.

**Wed 11 Nov:**
- **12:00Z, C.** `headerOk` at TWOWITNESS's `hdrJ`: `atRun` writes `"served/run-0/m0"` and the two `bindOf`s into its run fields, so its `Run` is `⟨"served/run-0/m0", bindOf "i0" s 1, bindOf "o0" s 2⟩`.
- **18:00Z, A.** `decode R.x` and `stmtJsonOf` (TWOWITNESS).
- **B.** `Ref.scope` through #8368's `ScopeDesc` (open, `f040c59a8b`). The kernel can't evaluate `scopeOk` itself here, because the statement is about 30 MB (TWOWITNESS's probe). So B lands only once #8368 is in #8363's base. I write it against #8368's head on Wed.

**Thu 12 Nov:**
- **12:00Z, C.** The checks left to prove:
  - population, roots, sample, `refsOk` and `valuesOk`;
  - `verified` and `tables`, from A's `checked_stmt`;
  - `readOk` at both linked ports, which needs #8309's arm (`rowsOf`, `placedOk`, `linkedRow`);
  - `recordsOk`, if #8406 has landed; its names are `i0` and `o0`.
- **18:00Z, E.** `Addresses` at the concrete `R`, and `InstanceRef.lean`: the concrete `Ref` and an accepted run at it, with D's `check xS wF = true` as its one hypothesis. It elaborates by then.

**Fri 13 Nov:**
- **06:00Z, D.** TWOWITNESS's `wF` goes in, and the hypothesis is dropped.
- **07:00Z, the run that shows the whole thing compiles.** `verity lean try InstanceRef.lean --siblings` on cpu-12, then #8363's Lean check by `research run` with `--timeout 10800`. ETA 10:00–11:00Z; mark by 12:00Z.

**What needs #8309's new definitions:**
- **Needs them:** `held`'s rows, `readOk` at the linked ports, `InstanceAddresses` and the assembly.
- **Doesn't:** the records, the bindings, the header, population, roots and sample, `refsOk` and `valuesOk`, and B, A and D.

**The cut, decided Thu 18:00Z if Friday is at risk.** Each cut keeps non-vacuity open and names the one fact it still needs. No guarantee changes.
1. **D late:** ship `InstanceRef` with Flock's `check xS wF = true` as the one named hypothesis. `flock-verify` accepting the same bytes is evidence, not a Lean fact.
2. **#8368 late:** add `Ref.scope` as a second named hypothesis, worded the same way.

**Risks:**
- D's kernel cost.
- A's terms aren't on a branch yet, and C needs them Tue and Wed.
- #8368's landing date.
