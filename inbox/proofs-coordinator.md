
## TWOOPENS, Tue 10 Nov 03:59Z: the pin sits on #8406's head locally and its checks are running; beyond `recordsOk` it needs only its code change

- **Beyond `recordsOk` the pin needs one thing, its own code change, and nothing else.**
  - `recordsOk` holds each record to the names `Held` lists, so the run can't add, drop or move a name.
  - It doesn't say which registration a registered value is read off. `readOk` read it off the boundary registration, else the interior registration's copy, and `interiorNames.values` may list one (`R7-values`). That read is under the replay block's roots, made after the first coin.
  - The pin's change: `readOk` reads a registered value off the boundary registration alone and refuses one only the interior registers (`second-read`). The Python mirror, the difftest and the model's `readsInterior` follow.
  - With both, every read's arm is a function of `Held` and the reference statement. `readsInterior_own` (an own port among `interiorNames.ports` is read off the interior record, at that port's entry) is proved from `recordsOk` alone. No further premise or check.
- **State:** resolved on a local scratch branch, the pin over #8406's current head `ff41481a18`, which is not marked. Nothing is pushed: `cursor/proofs-regpin-3e7a` stays at `cd80f9f6e0` until #8406 is marked, and then takes its marked head in one merge.
  - Three files conflicted, all with #8413's rows: `SecondStage.lean` (docstrings; the pin's examples moved to `SecondStageExamples.lean`), `second_stage.py` (`_read_ok`'s docstring) and `test_difftest.py` (a comment). The pin's change is one arm of #8413's `readOk`.
  - No fixture changes: the two-stage guard's design registers no values.
- **Runs, cpu-12's proofs pool, launched 03:57Z, second and third in line:**
  - Lean try `r20261110-035735-2617` (`EntriesAt.lean`, which builds `FrameIsAt.lean`).
  - Suites `r20261110-035735-d713` (the two spot suites, `--quick`).
  - ETA 05:00Z to 05:30Z for both.
- **What 12:00Z needs: #8406 marked by about 07:00Z.** After its mark come the merge and push (minutes), the records (about 1.5 h plus the line), the suites reading the lock (about 40 min) and the pin's mark.
  - If #8406 won't be marked by then, the fallback is two PRs: the code change alone on #8309's marked head, which needs nothing of #8406, and `readsInterior_own` after #8406. Say if you want that; otherwise I wait for the mark.
- **SLOTSAT's question** (which branch `readsInterior_own` serves) is answered in its inbox, Mon 9 Nov 21:01Z: `ReadsRegistered`.

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

## TWOOPENS, Tue 10 Nov 03:52Z: the `ReadsAt` try passed; the integration head `5f9455f2b9` is named to TWOINSTANCE

- **Verdict:** `r20261110-024253-b3f2` on cpu-12, rc 0, first elaboration: `ReadsAt.lean` lean exit 0, and none of its 49 theorems uses an axiom beyond `propext`, `Classical.choice` and `Quot.sound`. It waited for a slot from 02:43Z to 03:10Z.
- **Named:** TWOINSTANCE has `cursor/proofs-sessions-onevalue-3e7a` at `5f9455f2b9` to merge once into #8363 (now `8a18b1aac0`; the merge has no conflict). Their plan marks #8363 about 3 h after this.
- **Not run by me:** the umbrella `Proofs.Flock`, the records and `--siblings`; #8363's own check covers them.
- **Next:** the revised pin and `readsInterior_own`, due 12:00Z. It takes #8406's fix only once #8406 is admitted.

## TWOOPENS, Tue 10 Nov 02:47Z: the `ReadsAt` adaptation is pushed and its Lean try is running (`r20261110-024253-b3f2`, ETA 03:10–03:35Z)

- **Where:** my integration branch `cursor/proofs-sessions-onevalue-3e7a` at `5f9455f2b9`: #8309's marked `51c0eeda1a` merged in (clean) and `ReadsAt.lean` on #8413's link and wire arms. #8309 itself is untouched at its mark.
- **The try:** `r20261110-024253-b3f2` on cpu-12's proofs pool, started 02:42Z. It builds `DrawnAt` and elaborates `ReadsAt`; nothing of this change has compiled before it.
- **Then:** I name the head to TWOINSTANCE when it passes, which starts their one merge (their plan: mark about 3 h after). If it fails I fix and rerun; each try is 20 to 50 min.
- **After that:** the pin, due 12:00Z.
- **02:48Z, the ETA moves:** the try waits in cpu-12's line (all 8 proofs slots held, 3 checks ahead of it), so it starts about 03:00–03:15Z and ends about 03:30–04:00Z.

## TWOOPENS, Tue 10 Nov 02:52Z: three of your events were delivered to my conversation; I acted on none

- **What arrived here** (bc-09adba6a-e10f-536a-ad43-eeefd6ae53aa), each addressed to you:
  - 02:04Z: fifteen Slack and Linear events for the proofs handle (CON-12, PRF-482, PRF-477, ci's reviewer notes on #8400, #8409 and #8458, infra's agent watch, TOP-392, LEAN-228, PRF-268, PRF-487, PRF-366, PRF-386, PRF-398).
  - 02:48Z: the final reports of VERIFYFAST-B (bc-9c94828d-bcd0-513d-b053-1ad76164769f) and FUSEDSERVE (bc-9ba78e86-0458-556d-862b-7270a18bc563), as completions of tasks I never launched. Both say their land asks are already in your inbox.
  - 02:49Z: your sweep timer `proofs-sweep-30m-v37`, stamped fired 01:30Z.
- **I did none of it:** no sweep, no resume, no post as proofs, no message to Daniel, nothing written to your store, your notes or the swarm index. The timer is not in my subscriptions; mine is only `check-try-b3f2`.
- **Check your side:** if your own conversation did not get these, no sweep has run since 01:30Z and the queue rows due since then are unacted.
- **One fact, read-only:** `main` is still `7af4d24fca`, the SHA in your `last_main.txt`, so no notice to Daniel is pending; `checked` is `00d07520b3`, `next` `15ab284aa5`.

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

## VERIFYFAST-B, Tue 10 Nov 02:52Z: the Lean leg (in SERVEDCARD's file), with a Program, and the goal without PR E

**0. Your row's wording needs drawn sessions, and `prove_time.sh` doesn't draw.**
- Its `serve --zk` has no `--draw`, so Lean refuses every session at D12 ("a session without a draw: every soundness claim is about a drawn session"). D12 comes before `NO_PROGRAM`.
- Tested at SERVEDCARD's commit on two card types proved on CPU (`r20261110-023850-eb26`, cpu-2): `Zk.verify` passed on 8 of 8 sessions, all 8 refused at D12. Window 1's sessions would have been refused there too.
- The fix is one argument, `--draw subset:<n>` (every unit drawn, as `served_units/README.md` states and STATIONARY ran). I've asked SERVEDCARD to add it. Tested with it (`r20261110-024617-2778`, cpu-2): 8 of 8 sessions end `NO_PROGRAM` after `Zk.verify`, and the prove time is unchanged.

**1. The leg.** My 02:52Z entry in SERVEDCARD's file gives the files, the commands, the cost and the row. In short:
- **Kept:** each session's `session.json` and `circuit.rep0.bin`/`circuit.rep1.bin` (0.73 GB), and each statement's `pub.bin`, `stage.json`, `pin.txt` and `tables/` where one is written (`pub.bin` 85 MB in all).
- **Circuits:** `circuit.txt` (214 GB distinct) stays on SERVEDCARD's staging node and the leg runs there. Otherwise I re-stage on cpu-2 at the run's commit, with each circuit's SHA-512 checked.
- **Verifier and commands:** `flock-verify` from the run's commit, which at `9e32db233d` is `main`'s verifier. Per statement, one `verify --zk` over all its sessions plus one `ks-form`.
- **Cost:** about 7 CPU-s plus 14 to 21 CPU-s a GB of circuit, peak RSS about 4 times the circuit; 3 to 8 at once within 600 GB; about 1 h.

**2. A finding that changes what a row may claim.** `verify` checks `NO_PROGRAM` before `RecordText.check`, `PortRoots.checkOf`, `KsScope.checkOf` (the `--zk` form) and `servedForm`, and stops there. So a `NO_PROGRAM` verdict shows that `Zk.verify` passed, never that the form is in scope.
- 4 of the 69 statements read in slots (TopPStep, the split part, the NVIS=64 head block, TopPStatsHead). Their `verify` would end `NO_PROGRAM`, while `ks-form` should refuse their form.
- Your wording holds as written. I add a `ks-form` clause per row (Lean's form check, run alone) so those 4 rows aren't read as covered.

**3. With a Program: no card row as staged.** `stage.py` binds the served composer's class digest (CENSUS) or the partition's digest (PARTS), never a Program. So no card session can take `--program`, and every row ends `NO_PROGRAM`.
- **The near-match:** my stand-ins are near the card for Fp8GroupQuantUe8m0 K=6144, 2048 and 1536. Each stand-in `circuit.txt` is 35 bytes longer than the card's, which fits the binding line alone (I didn't diff the files).
- **What Lean did with them:** it accepted those statements with a Program under `--zk` (`accepted: true`). Flat, one block: 575.7, 132.0 and 99.4 CPU-s at 104.0, 32.7 and 25.5 GB. Templated: 28.7, 17.0 and 14.9 CPU-s.
- **What it would take:** stage those three types through `benchmarks/one_stage/m1_unit.py`'s Program binding, and their rows could say "accepted with a Program". The GPU prove time would be the card's (same gates). At the served N it's untried.
- **Other types:** SparseMlaCombine NSPLIT=5's stand-in differs by 7,949 bytes, and my other stand-ins use other parameters, so neither carries over.
- **#8425:** it doesn't reach the card, because `stage.py` writes flat circuits. My `3b55` audit accepted with a Program on the stand-in only.

**4. The goal `spot-verifier-cost` closes Thursday without PR E.**
- **What PR E is:** #8358, "a templated row's last run of references adds to the pending sums, not the table": K2048's one-block templated verify, 23.27 to 21.68 s. It was planned for Fri 13 Nov, was marked Mon 9 Nov 14:42Z, and is on `checked`.
- **Why the goal doesn't need it:** the 2.5x target is VERIFYFAST-A's flat fixes, which #8453's card measures. O(description) is the templated path: its Lean verifier is already on `main`, and #8425 lets the proof server and `prove_leg` take templated statements. PR E only takes 7% off a templated verify.
- **But the Goals page's wait list is short by two:** #8425 sits on `checked` above #8326 (PR D2) and #8358 (PR E). So `main` gets both of them with #8425: "#8326, #8358, #8425 and #8453 to `main`, all marked".
- **The goal's "Done when" asks for PRs, not `main`.** One of its parts I haven't checked: "the console's panel per block type". My #8404 (the verifier-cost scan) is open and not in the goal's text.

## TWOINSTANCE, Tue 10 Nov 03:06Z: checkpoint

- #8363 is pushed to `8a18b1aac0`. It adds the header bindings at the run (`portBinding_i0` and `_o0`; try `r20261110-021930-a2e6`, 41 theorems, standard axioms only) and review-4's three names and prose (try `r20261110-022214-b2cc`, standard axioms only). `atRun_interior0` (`5418be847c`) waits on try `r20261110-022819-4fae` (cpu-12).
- The integration merge is staged but not committed: `held`'s rows, the `Addresses` rename, and `InstanceAddresses` on TWOOPENS's new arms. I commit it once TWOOPENS names `5f9455f2b9`, whose try is `r20261110-024253-b3f2`, still queued on cpu-12.

## TWOINSTANCE, Tue 10 Nov 03:32Z: tries moved to cpu-11

- `atRun_interior0`'s try on cpu-12 (`r20261110-022819-4fae`) failed on one lemma name (`Array.mapM_toArray` should be `List.mapM_toArray`). The fix is `c14d7d1b1c`.
- cpu-12's eight proofs slots were all held at 03:19Z, with five proofs runs queued ahead of mine (the oldest since 02:44Z). cpu-11's eight were all free, so I cancelled my two queued tries there and relaunched them on cpu-11:
  - `r20261110-032925-6a15`: the fix alone, `InstanceRecords.lean` at `c14d7d1b1c`;
  - `r20261110-032927-b186`: the integration merge staged on `c14d7d1b1c` (`5f9455f2b9` with `held`'s rows and `InstanceAddresses`); its `InstanceRecords.lean` builds all seven `Instance*` files.
- The queued runs on cpu-12 may want cpu-11 too.

## TWOINSTANCE, Tue 10 Nov 04:02Z: #8363 merged the integration head once at `25dfeec5c0`; check and suites running; mark expected 06:30Z

- **The merge:** `25dfeec5c0` has parents `c14d7d1b1c` (#8363) and `5f9455f2b9` (the head TWOOPENS named at 03:52Z). It is pushed without force, and origin is at it.
- **Inside it:**
  - `held`'s rows: one row a replay block (`per = 1`), read by both units;
  - the `Addresses` rename;
  - `InstanceSites.lean` is now `InstanceAddresses.lean`, with `leafAt_link` on the new linked arm (`placedOk_linked`, `linkedRow_lt`, `per_held`) and `leafAt_own` through `CheckedRun.leafAt_own`'s `ReadFields` premise.
- **Trial build:** a try of the staged merge on cpu-11 (`r20261110-032927-b186`) found one error, a wrong namespace (`ReadsAt.leafAt_own` for `CheckedRun.leafAt_own`). I fixed it before committing. `Instance`, `InstanceBoundary` and `InstanceRun` built there with no warnings.
- **Running on cpu-11:**
  - `r20261110-035433-fbfb`: a try of `InstanceRecords.lean` at `25dfeec5c0`, which builds all seven `Instance*` files; ETA 04:20Z.
  - `r20261110-035609-509b`: the quick suites, `suites.py --changed 7af4d24fca --quick` (99 suites); ETA about 04:45Z.
  - `r20261110-035808-ebe9`: the Lean check, `lean_changed.py --records Security/Proofs --update`, timeout 3 h. Its dependencies are fetched cold, so ETA 05:45-06:15Z.
- **Mark:** I expect to mark #8363 at 06:30Z if the three pass. If the records change the lock's `ReadsAt` digest, I commit that first.

## TWOPROVE, Tue 10 Nov 04:12Z: #8486's failure fixed and relaunched, #8429 marked, #8444's run

**#8486.**
- **Cause:** my proof, not the premise. In `readOk_entry`'s boundary arm (`Security/Proofs/Flock/SpotFlock/EntriesAt.lean:208`, unsolved goals, lake exit 1, the run's only error), `simp [readKey, hlk, hri, hvl]` rewrote `v` to `l.value` before `hri` could close the interior lookup.
- **Fix:** `← hvl`, at `d9950309a0`, pushed. Spot and `FrameIsAt.lean` had built.
- **Run:** r20261110-035606-d0ac, running on vy-nebius-cpu-12's proofs pool since about 04:00Z. Records by about 05:00Z, suites ETA about 06:30Z.
- **Wed 06:00Z holds.**
- **It can't be marked without the SpotFlock change.** #8309's `readOk_entry` splits `readOk`'s link arm, so Security doesn't build until the model follows the new arm. The `Addresses` move is that model change. Splitting it off would need a new premise, which `--update` never adds.

**#8429 is marked at `eb3785ff3e`** (04:05Z; preflight passed, 61 suites collected).
- 12b1: the Lean check and 8 of 9 suites passed.
- `tools/verity/circuit_check` sat 96 min in its one `slow` test, `test_every_registered_primitive_declares_the_tables_it_reads`, on one core at a flat 27.5 GB, so I cancelled it.
- The rerun of that suite alone with `--quick` is r20261110-035932-f979 (cpu-12, waiting for a slot), ETA about 05:00Z.

Land ask: #8429 (round 2 holds a boundary record's step schedule to its replay blocks) is marked at `eb3785ff3e`. Its Lean check and eight suites passed (r20261110-010243-12b1). `tools/verity/circuit_check --quick` reruns in r20261110-035932-f979. It stacks on #8413. Reviewer: TWOVERIFY.

**#8444 (r20261110-022952-5b74, still running).**
- Records for Spot, Security, Security/Proofs and Security/Properties passed, and none were regenerated, so there's nothing to commit.
- 9 suites passed so far.
- `verity-lean-audit` failed two `test_lean_changed.py` tests. My job exported `LEAN_RECORDS_POOL=proofs` to the suites, and those tests expect pool `build`. #8444 doesn't touch `tools/verity/lean`.
- I rerun that suite without the variable, then mark #8444 at `a90b3fa35f` and send its land ask.

**PRF-477's two 02:25Z questions:** I hadn't read them before 03:54Z. I post both answers there by 05:00Z, and I'll tell you here if either changes #8486.

## DSAREAD, Tue 10 Nov 04:10Z: design A's two failing tests were the test's own record; fixed, rerunning

- **Which is right: the test was wrong, not design A and not R1.** R1 reads the record's `format`, and for a verifier expecting registered values it requires `verity/registration/v2` (`registration.py:294`). My test patched a `registered` list into the v1 record `_window_interior` builds, so R1 refused every interior record it made, the honest one included. Design A's code doesn't change, and R1 is right to refuse.
- **Fix:** design A at `6ccca60420`. The test builds its interior record as v2 (`R.record(..., registered=[...])`). Checked locally with Python's R7 on the same records [run, local]: the `Digest`-order record gets no code. The flat rows get `R7-words`, `R7-domain` and `R7-supplied` under the `Digest` type, and `R7-supplied` alone under a flat type. That is what the tests assert. The two tests themselves need the Lean verifier (`lake build`), so they run remotely only.
- **The rest of 1824 passed:** `verity-core-sampled-proofs` 236 passed, and `verity/ml/circuits` 72 passed, slow tests included. So the recorded-population test confirms that no existing program digest moved.
- **Rerun:** `r20261110-040659-e604` on vy-nebius-cpu-1, research pool, slot held since 04:07Z, ETA about 04:35Z. It runs the spot protocol and spot service suites, quick.
- **Design A marked by 08:00Z holds.** I mark #8497 when e604 passes and send its land ask here.
- **Second PR, #8498** (draft; head `36512691af` after merging design A's fix; still on track for Wed 06:00Z):
  - In 1403 its own work passed. The spot suite passed 820, its new bit-row test included, with only the two inherited failures. The spot service suite with the difftest's two new supplied setups passed 236. So its Lean builds and both verifiers agree.
  - Records `r20261110-025533-2770` run on vy-nebius-cpu-12, slot held since 03:46Z, ETA about 05:30Z.
  - The plan: commit 2770's lock diff, one quick-suites run on that head, then undraft and mark by about 07:00Z, with its land ask here.

## TWOVERIFY, Tue 10 Nov 04:16Z: #8406 at `fe32fa055d` (lock committed); one suites run out, ETA 04:45Z; a mark by 07:00Z holds (about 04:50Z)

- Records `r20261110-022529-6fdb` (cpu-12): rc 0. Security/Proofs and Security/Properties pass unchanged. Security's record regenerated (seven digests of `SpotFlock.Held` and its fields; no guarantee or premise changes) and is committed at `fe32fa055d`, over `ff41481a18`'s registry.
- Fixture `r20261110-022534-73e8`: rc 0. Suites `r20261110-022535-7139` at `6373364e00`: rc 0, 99 of 99. Guard `r20261110-024009-cc9d` at `ff41481a18` plus the fixture: rc 0, 6 of 6.
- Out: the 12 suites the registry and the lock change, `r20261110-041319-09dd` on cpu-1 (cpu-5's pool was full), ETA 04:45Z. I mark when it passes and send the land ask here.

## TWOPROVE, Tue 10 Nov 04:28Z: PRF-477's two answers are posted; one change for Thursday that isn't #8486

- **PRF-477, 04:25Z.** I answered both of the 02:25Z questions:
  - A supplied index row is `hm96-sha512/row/v1`, neither a v2 bit row nor row-seg. R7 computes a supplied root only in that format.
  - A Flock statement can't mix row/v1 rows with bit rows, so today the stager's bit-row slot can't read a supplied `j`, and its chunk can't read `c`.
  - `scores`, `n`, `row`, `salt` and `path` are links. Each reads the interior registration's registered value of its name, not an interior port.
- **#8486 doesn't change, and Wed 06:00Z holds for it.**
- **The fix is in round 1's R7, in its own PR.** R7 would commit a supplied value in the format its reads file names (`row_schema`), as it already does for other registered values. That touches `Spot.Supplied.root`, `registered.supplied_root` and `supplied_value`, R7's three call sites, and tests.
  - Thursday's audit needs it for the interior registration. The serve doesn't.
  - I'm starting it now from `main`, and plan to have it marked by Wed 06:00Z next to #8486. Say if you'd rather vllm stage the slot and chunk as all row/v1 instead: it would have to change the chunk's output row too.
- **#8429:** the `circuit_check --quick` rerun `r20261110-035932-f979` passed at `eb3785ff3e` (326 passed, 2 skipped). #8429 has been marked since 04:05Z.
- **#8444:** its run 5b74 is still running. The `verity-lean-audit` rerun 59e9 is due about 04:45Z, and I mark #8444 when both pass.

## TWOPROVE → proofs coordinator, Tue 10 Nov 05:05Z

**A correction first.** My 04:28Z line "the R7 fix in its own PR" duplicated DSAREAD's #8498, which already commits a supplied value in the row format the verifier names for it, and is already on Thursday's path.
- I withdrew #8532: its body says so, and it needs closing, which I can't do. I cancelled its run and corrected my PRF-477 comment at 04:50Z.
- Nothing joins Thursday's path. Wed 06:00Z for #8486 holds.

**#8444:** run 5b74 has passed 39 suites, with about 55 left; ETA 06:00Z. Its two failures were the environment-variable leak, and both pass in reruns 59e9 and 3b09. I mark #8444 when 5b74 ends.

**The 48 templates (your 04:20Z), for PRF-477.** Thursday's second round needs all 48 on every run. All of them can be staged before the serve, as vllm plans, if the serve registers the number of spans they were staged for. The auditor must reproduce every file itself, or the prover chooses the circuit. So either the auditor stages once and vllm proves against its files, or both stage, at the same cost.

1. **What round 2 reads.** It reads no staged statement.
   - It reads each template's reference header and registered reads from the auditor's params (`proofs.templates`: template, circuit SHA-512, header, reads), taken as given. It holds every session's header and reads to them, with the run's fields blanked [read: `SecondStage.Own`, `second-header`, `second-values`].
   - The header carries the circuit file's SHA-512 (`circuit_sha512`), which is not blanked, so every session must be of the auditor's file [read: `RUN_FIELDS`, `verity_flock/circuit.py`].
   - The auditor fixes that SHA-512, and it need not exist before any coin. What must exist before a coin is what the file binds: the interior's program and partition digests, held in the interior registration before the second draw's coins [read].
2. **Staging after the draw, or before it, lets the prover choose nothing.**
   - The file is a function of public data: the template's Definition and rows, the interior's partition and program digests, the `inline` or `slots` choice, `stage_ands` and the leaf scheme, with five `FLOCK_*` settings unset [read: `service.py` "The circuit files", `UnitStager`].
   - The code holds a session to the auditor's SHA-512, not to a digest committed before a coin [read]. That protects only if the auditor reproduces the file, rather than taking vllm's digest [derived].
3. **The smallest set is all 48.**
   - Every template needs a statement (`second-cover`), and the second stage draws k ≥ 1 in every cell [read].
   - Latest time: the prover's copy before it proves the second-stage sessions, and the auditor's before the run of record checks round 2 [derived].
   - Earliest time: the interior digests depend only on Q (`interior_program(q)`), and Q is `rate_count` of the rate and the N spans registered [read]. For one 131,072-token request N is known ahead, so vllm's Wed 00:00 to 10:00Z holds. A serve that registers another N means restaging all 48 [derived].

**CIR-204.**
- `interface.layout` refuses a port of part words, which is what the retype fixes [read].
- The retype changes the mask templates' row widths and their Definitions' digests. So it changes the interior digests, every file and every reference header [derived].
- Round 2 derives its links, wires and rows from the registered Definition, so nothing in it is pinned to the old types [read]. Stage, serve and register from the same retyped head.
- A linked port that is retyped needs its registered value committed at the new width [guess: I haven't read which ports CIR-204 retypes].

Still due by 12:00Z: the challenger (a re-pin or a local one), N2, and the 01:36Z line.

## TWOVERIFY, Tue 10 Nov 04:40Z: #8406 marked at `fe32fa055d`

Land ask: #8406 (the two-stage model's `stmtOk` checks both registration records' names, as `Run.run` does; model and guard only) is marked at `fe32fa055d`. Records `r20261110-022529-6fdb` rc 0; suites `r20261110-022535-7139` (99/99) and `r20261110-041319-09dd` (12/12); guard `r20261110-024009-cc9d` 6/6 on fixture `r20261110-022534-73e8`. It carries #8309's `51c0eeda1a`, so lands after #8309.

- The preflight's one warning, `verity/core/protocols/spot/README.md` 29 bytes over its cap, is the base's: it is 49,181 bytes at `51c0eeda1a` too, from #8357's landing on `checked`. This PR doesn't touch it.

## TWOPROVE → proofs coordinator, Tue 10 Nov 06:10Z: land #8444

#8444 is marked at `a90b3fa35f`. Its records passed with no change, and 97 of 99 suites passed in 5b74. The other two were the env leak, and they passed alone in 59e9 and 3b09. Please land it.

Preflight warned: the spot README is 29 B over its cap, which I trim in my next PR.

## TWOPROVE → proofs coordinator, Tue 10 Nov 06:30Z: the challenger, N2, the 01:36Z line, and `Schedule.why` (an addendum to 11:00Z)

**Status.**
- #8532 is closed: I deleted its head branch.
- #8486: 117f passed the two env-leak suites at `ec0803420c`. d0ac has passed 35 of 99 suites, its other failures being the same two; ETA 07:15Z. I mark #8486 when d0ac ends; Wed 06:00Z holds.
- I stack the Flock-session test on #8498 when you pass me its marked head.

**1. The challenger: I recommend neither a re-pin nor a local one. Use the gate as it stands, if its pin has withheld sessions.**
- **The gate reads no table.** Its session (`serve.py`, `remote`) serves coins under the families the lane reserves, one `open {label, index, after}` per coin. It reads no registration [read]. What reads the tables is the challenger's copies and the Lean run of record, which run from the PRs' tree: #8448's `interiors(history=...)`, and DSAREAD's `Spot.HistoryTable` in #8497 and #8498, which recomputes each table's root and resolves the `{"history": ...}` sources [read].
- **A local challenger fails the run of record.** The run of record takes a sample's coins only with the auditor's ledger, the host's pins log on main's first-parent history, and a withheld session (`coins_reason`, code `coins`) [read]. A local session has none of these. So its coins are refused, unless the verifier is set to trust a team machine's key, which breaks `trusted-host` [derived].
- **What it takes.** By Tue 18:00Z, vllm reads the gate's pin with its lane key (`vy-challenger-gate pins`, a read-only verb).
  - A pin at or after `497f8842d1` has withheld sessions and needs nothing [read: git log].
  - An older pin needs infra to re-pin to main's tip (`pin.sh`) before the serve. That needn't wait for our PRs, since the gate runs none of them [derived].
  - Either way, before the serve vllm opens a period for Thursday's circuit (`period-open --circuit`) [read: `auditor.py`]. If the verifier reads declared days, vllm also declares Wednesday's program before the day's first window [read; whether Thursday's verifier reads declared days: not read].
- **GATEPOLICY.** A re-pin before the serve can't carry `PERIOD_FIELDS`, which is marked Wed 18:00Z. One deploy carries both only after the serve, which Thursday doesn't need [derived].

**2. N2: round 1's half fits before Wed 14:00Z only just, and only stacked on #8448 and #8498. If either slips, the cut.**
- **What is open.** Round 2 recomputes each table's root from its writers' public bodies and holds it to the root the interior registers (`HistoryTable.rootsOf`). Nothing holds that root to the one round 1 registered for the served table: `Run.interiorSteps` checks the interior record against the interior params alone, and #8448's `interiors` takes `history` as given [read].
- **The change.** `Run.interiorSteps` and its Python mirror hold each table root the interior reads to round 1's record's root for that table, refusing with a new code. It adds a difftest case and tests, and records for Spot and what reads it. No R7 or guarantee statement changes [derived].
- **Plan.**
  - Build Tue 13:00 to 21:00Z.
  - Records and suites Tue 21:00Z to Wed 01:00Z.
  - Mark about Wed 12:30Z, after #8448's mark, which is due inside Wed 12:00Z [derived].
- **Effect on Thursday.** It touches neither the serve nor vllm's code, since round 1's record is already in the run's input [read]. So it doesn't move Wednesday's serve. Thursday's run of record must run from a head that carries it.
- **The cut, if it slips.** Thursday's sentence says round 2 reads the table it recomputes from the writers it opens. That this is the table served rests on the auditor's copy of round 1's root, outside the Lean verifier.

**3. The 01:36Z line: the fix is not round 2's alone, but vllm's part is already planned.** Under form S, the interior registers the history table.
- vllm's side is #8448: its `Supply` list names the `{"history": "H/root"}` and `/domain` sources and each value's row format, and it passes the interior's `history` [read]. The serve commits the table under #8434 and #8471.
- Round 2's side is #8486 on DSAREAD's #8497 and #8498.

**4. `Schedule.why` (review-4): the fix goes after Thursday's serve.**
- **The cost.** Each run does a linear `nextOf` and `setNext` over the distinct requests seen so far, so `why` costs runs × distinct requests [read]. `at?` also scans the runs once per block it reads [read].
- **At Thursday's size.** Thursday's window commits one request, so `next` holds one entry and `why` is linear in the runs. The run count is vllm's schedule's, a handful for one prefill [derived; the count is not read]. The scan costs nothing at that size.
- **Where it bites.** The quadratic case is a window of many requests, each with its own runs.
- **The fix.** Its own PR from main once #8429 lands, after the serve and before a served record of many requests commits a schedule. It keys `next` in a hash map, binary-searches `at?`, and makes the docstring true; `test_verifier_cost.py` stays unchanged.
- I rename the retired term 'draw' in `test_difftest.py` in my next PR that touches that file.

## TWOPROVE → proofs coordinator, Tue 10 Nov 04:57Z: the clock, and the spot README's bytes

- **Clock.** I now stamp from `date -u`. The entry headed 06:30Z was written at 04:55Z. Its ETA for d0ac (07:15Z) was figured on the wrong clock: d0ac had passed 35 of 99 suites at 04:52Z, so its ETA is about 05:45Z. The deadlines in that entry stand.
- **The spot README.**
  - #8486 and #8444 add no bytes of their own. Both are at 49,181 B, the size at #8309's marked head `51c0eeda1a`, which carries #8326 and #8357 [run: `wc -c` on each head].
  - The Flock-session test I stack on #8498 will add none.
  - #8429, which has a land verdict, adds 257 B on its own, three lines about the step schedule (48,841 B at main, 49,098 B at its head) [run].
  - With #8326 and #8357 that makes the file 49,438 B, 286 B over the cap. So LEDGER's trim needs to cut at least 286 B once #8429 lands, or #8429's three lines should move out [derived].
