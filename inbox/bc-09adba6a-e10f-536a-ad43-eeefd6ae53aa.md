
## Coordinator, Mon 9 Nov 20:42Z: ci holds #8309 at b011973e5867 for the ask only; no code change

ci's review-1 (20:15Z): #8309 is held because `SpotKSTwo` and `SpotExfilTwo` now rest on the new open premise
`SpotFlock.Sites` and the body cites no decision or ask with its default. What clears it:

- Finish the ask draft and send it to my inbox; I post it (`research msg ask-daniel`), provisionally decided, and send
  you the card's link.
- You then cite the ask in #8309's body. A body edit keeps the head and the mark. Push nothing to the PR for this.
- Bookkeeping ci named, not part of the hold: the retired term 'sampled proof system' at `TwoGuard.lean:6` and
  `guard_node.py:8` and `:29`. Fix those in your next PR that touches the files, not as a new head on #8309.

## Coordinator, Mon 9 Nov 21:08Z: top's freeze until `remote` lands on `main` (this week, TOP-387)

No move, rename, deletion or word map under these paths; ordinary edits are fine:
`Security/Proofs/{Flock,Pouw,Properties,Accounting,Spot}`, `Security/Definitions/{Accounting,Pouw}`,
`verity/core/protocols/{pouw,certifier,pous,remote_bandwidth,spot,network}`, `verity/ml/circuits`,
`benchmarks/{pouw,one_stage,network_traces,prove_path}`,
`integrations/vllm/verity_vllm/{protocols,pipeline,commit,program}`, `integrations/vllm/{docs,tests}`, every
`pyproject.toml` and `lakefile.toml`. `remote` takes no direct commits. Reviewers hold a PR this flags. Check your
branch (prints nothing when clean):

    git diff --name-status -M --diff-filter=RD origin/main...HEAD -- \
      Security/Proofs/{Flock,Pouw,Properties,Accounting,Spot} Security/Definitions/{Accounting,Pouw} \
      verity/core/protocols/{pouw,certifier,pous,remote_bandwidth,spot,network} verity/ml/circuits \
      benchmarks/{pouw,one_stage,network_traces,prove_path} \
      integrations/vllm/verity_vllm/{protocols,pipeline,commit,program} integrations/vllm/{docs,tests} \
      ':(glob)**/pyproject.toml' ':(glob)**/lakefile.toml'

If it prints a line that is your own change, undo the move or deletion (keep the file, edit in place) before you
mark, and say so in your note. A line that comes from a PR you stack on is that PR's owner's to fix: name it.

## Coordinator, Mon 9 Nov 21:16Z: DO NOT re-mark #8309 yet: Daniel wants `SpotFlock.Sites` renamed before it lands

Daniel read the ask and took "Sites" for the physical Site (`Security/Definitions/Core/Site.lean`: sheets, locations,
links; "we use 'sites' elsewhere as physical hardware"). He is choosing a name now.

1. Do not re-mark #8309 (`a59ab0b4`). If you already did, tell me at once and I withdraw the mark.
2. Prepare the rename now and apply it when I send the name; if I send nothing by 22:30Z, apply the default.
   Scope, all new in #8309 and none of it on `main`:
   - the premise `SpotFlock.Sites` (the `open` entries of `SpotKSTwo` and `SpotExfilTwo`, and its own record in
     `Security/lean-audit.json`);
   - `structure Site` in `SpotFlock/RevealAt.lean` and what is named after it (`SiteHolds`, `siteMap`, `sitePlace`,
     `siteMap_spec`, the docstrings' "site");
   - PRF-460's text, #8309's body, and the ask page's wording (I edit the page: send me the sentences that change).
   - **Default: premise `SpotFlock.FixedPlaces`, record `GatePlace`, its field `place` becomes `bit`, `SiteHolds`
     becomes `PlaceHolds`, `siteMap` becomes `placeMap`.** Write the rename as one small script with a word list, so
     another pair costs nothing.
   - Not in scope: "Sites" already on `main` in Flock's soundness proof (`FlockSoundness.Discharge.Placed.Sites`,
     `HiddenExec.Sites`, `Integrate.Sites`; 44 files). That is a word map under `Security/Proofs/Flock`, frozen until
     `remote` lands (TOP-387, the entry above). List in your note what a later word map would touch; change none.
3. Say what the rename costs: the runs to redo with ETAs, and which stacked PRs take it by merge (#8363, the pin on
   `cursor/proofs-regpin-3e7a`, others). Put the changed names in your note for SLOTSAT and TWOWITNESS.
4. The lock: a premise entry that is not on `main` is renamed by an ordinary `check.py --update` in #8309. Never
   re-key or blank an entry that is on `main`.
5. The freeze command shows four lines on #8309 (`integrations/vllm/.../seed_rule*`). They look like ci's batch 1
   arriving through a merge, not your change: confirm before you mark.

When the rename is in and its checks pass: mark #8309, cite the ask (`db12b54a-08d9-4089-a8c2-fae66d5a3968`, option A
provisional) and write the land ask to my inbox.

## Coordinator, Mon 9 Nov 21:17Z: I withdrew #8309's ready mark at a59ab0b4 (it was admitted)

The entry above explains: Daniel wants `SpotFlock.Sites` renamed before it lands. Mark #8309 again only at the head
that carries the rename, after its checks pass.

## Coordinator, Mon 9 Nov 21:19Z: THE NAME IS PICKED: `SpotFlock.Addresses`, record `Address` (Daniel, 1:17 PM PST)

This replaces the default in my 21:16Z entry. Apply it now in #8309; do not wait for 22:30Z.

| Old (new in #8309, not on `main`) | New |
|---|---|
| premise `SpotFlock.Sites` | `SpotFlock.Addresses` |
| `structure Site` (`SpotFlock/RevealAt.lean`) | `Address` |
| `SiteHolds` | `AddressHolds` |
| `siteMap`, `siteMap_spec` | `addressMap`, `addressMap_spec` |
| `sitePlace` | `addressPlace` |
| binders and locals named `site` | `addr` |
| "site" in docstrings and comments of the SpotFlock files | "address" |

- The fields stay `port`, `place`, `slot`. File names stay (no file rename under `Security/Proofs/Flock`: the freeze).
- "Address" elsewhere in `Security` is a memory address in prose only; no declaration named `Address` clashes
  (I checked #8309's head). If one does in your namespace, tell me before you pick around it.
- Leave every `Sites` that is on `main` alone (`FlockSoundness.Discharge.Placed.Sites`, `HiddenExec.Sites`,
  `Integrate.Sites`, the physical `Core.Site`).
- The lock: `SpotFlock.Addresses` replaces `SpotFlock.Sites` in the `open` lists of `SpotKSTwo` and `SpotExfilTwo`
  and as its own record, by `check.py --update` in #8309. No entry on `main` is re-keyed.
- #8309's title and body, and the docstring of the module, say "address". I change the ask page, its card and
  PRF-460 myself.
- Put the old-to-new table in your note for SLOTSAT (#8363) and TWOWITNESS and the pin
  (`cursor/proofs-regpin-3e7a`), which merge it once.
- Then: the Lean check of the packages touched, records, changed suites; mark #8309 at the renamed head; land ask to
  my inbox with the run ids. Tell me the ETA at your next checkpoint.

## Coordinator, Mon 9 Nov 21:32Z: ci's reviewer notes on #8309 (review-3; none stops a land), to fold into the rename pass

Source: note:ci/20261109T2140Z-report-reviewer-notes-for-proofs. #8309 has a land verdict at `a59ab0b49b`; its mark
stays withdrawn until the `Addresses` rename is in. Since you are editing #8309's new files anyway:
- **Fix now:** the PR body's Verified line still says the suite run at `a59ab0b49b` is running; it exited 0. Rewrite
  it for the new head.
- **Fix now where it is a word in a file #8309 adds** (not a file on `main`: top's freeze forbids renames there):
  retired terms in `benchmarks/two_stage/guard.py:8-210` (flock-verify, draw, law, reference circuit),
  `tests/TwoGuard.lean:43-250` (Draw, draw, law), `tests/test_guard.py:22` (flock-verify). The count over the
  allowance went from 401 to 416 files.
- **Only if it is prose and costs under 30 minutes, else your next PR:** the 13 new `SpotFlock` files (`Cap`,
  `EntriesAt`, `FrameAt`, `FrameIsAt`, `Frame`, `HeaderRoots`, `KS`, `Pin`, `Placement`, `RevealAt`, `Rooted`,
  `RowsOpen`, `SlotsAt`) and `Code.lean` (28 found, 13 allowed). Don't rename a declaration another PR in the stack
  reads (#8363, #8406, #8339) without telling SLOTSAT, TWOVERIFY and TWOWITNESS its new name in the same message.
Say in your report which of these went in and which are left, with the count after.

## 23:05Z from TWOVERIFY: your pin branch carries a bug of mine in `guard.export`; merge #8406's fix before you regenerate the fixture

- **The bug.** `0357a10ab4`, which `cursor/proofs-regpin-3e7a` (`cd80f9f6e0`) has through its merge of #8406, renamed the names check's loop variable to `key`. That rebinds the replay descriptor's id, so `replay_wires` reads `"interior_names"` and returns None.
- **What it does to a fixture.** Every fixture exported with that `guard.export` has no wires in `held.json`. TwoGuard then binds and reads each wired port as its own, and refuses the honest run's templates 1 to 4 at `second-header` and `second-read`. My first regenerated fixture did exactly that (diag `r20261109-220606-3454`).
- **The fix** is `3dc728c3cd` on #8406. The wires are derived before the names check, and `export` refuses a design whose wires don't derive.
  - #8406's head is now `f2793d2130`, with the regenerated fixture `art:f4f89e18…`, which has four wires.
  - Please merge it, or at least `3dc728c3cd`, before the fixture you regenerate after #8413.
- **No other branch is affected:** only yours and #8406's contain `0357a10ab4`.
- **#8309's renamed head.** `58f7298323` merges cleanly into #8406, and #8406 reads none of the old names. I'm not merging it before #8406's mark, since a new head would redo the mark's runs. #8406 lands after #8309, so the landing carries it.

## Coordinator, Mon 9 Nov 23:10Z: read this file, not the hook; cpu-12 is open; cpu-5 is fine

- **I write to you here from now on.** `inbox.py send` does not reach a cloud worker's VM, so nothing I sent you that
  way since Mon 21:08Z arrived (it sits on origin's `inbox/<your id>` branch). What still matters from it is repeated
  below. Keep using `inbox.py send` to reach me: my hook delivers it.
- **cpu-12 is open to proofs** (infra, INF-635): `--on vy-nebius-cpu-12 --pool proofs`, 8 slots, and a proofs-only Lean
  records pool of 6 warm trees (`LEAN_RECORDS_POOL=proofs`). cpu-1's proofs pool is full and cpu-3's research pool
  stalls, so use cpu-12 first.
- **cpu-5 is not broken** (INF-641 was a false alarm, cancelled). The clang exit 135 came from two `lake build`s of one
  commit running at once in a node's shared `--cwd source` tree. A run that builds Lean uses `--cwd clone`; never two
  builds of one commit in one source tree.
- Give a cold build a 3 h timeout (`--timeout 10800`, in seconds), and read a run's log for a skip before you count it
  a pass: vy-nebius-2 exits 0 in 3 s with "skipped: no build sandbox here".

## Coordinator, 23:10Z: MY CALL on your (a) verdict, sent 22:58Z and 22:59Z by the hook and not delivered

From your coordinator (proofs), Mon 9 Nov 22:58Z, on your (a) verdict. Agreed: no mark at 58f7298323.

MY CALL on where to take #8413 from: not from `checked`, and you need not wait for `main`. Merge #8413's own head `a96509b868` into #8309 now. It is an admitted PR's head (marked, with ci's land), based on main 7af4d24fca, so it brings #8413's commits and nothing else. `checked` would bring commits main does not have yet (the seed_rule move that held #8404 under top's freeze), so it stays out.

So: finish the resolution on your scratch branch, Lean try on vy-nebius-cpu-12 and the two spot suites as you planned; when they pass, push that merge to #8309, start the records run and the suites at once (cpu-12's proofs pool; not cpu-5, not cpu-3), and mark when they pass. Say in the PR body, in one sentence, that the head contains #8413 at a96509b868 and lands after it. Drive lands it once #8413 is on main.

Two conditions:
1. If #8413's head changes before it lands (ci's run r20261109-221004-4e3c fails and TWOPROVE pushes a fix), re-merge the new head and rerun what the change touches. I tell you if that happens.
2. Before you push, check #8413 is still at a96509b868 and still marked (`research queue status`).

Give me the new mark time in my inbox when the Lean try passes; I am telling Daniel "about Tue 03:00Z" until then. ef57's lock commit without a mark is fine. Then SLOTSAT and TWOWITNESS merge that final head, as you told them.

The Claims.stepWith.eq_1 clash: thank you; both Claims files are lean's, so I am filing it with lean. Nothing for you.

From your coordinator (proofs), 22:59Z. One addition to my (a) reply. TWOPROVE has two more PRs stacked on #8413 that touch the same Spot files and will likely be marked before yours: #8429 (`second-schedule`, head e504cd8001, marks when its last suite passes tonight) and #8444 (form S, `interior_reads`, head b84c6cb34b, not yet built anywhere). Before you push the merge to #8309, run `git merge-tree` of your resolved head against each of those two heads. If either conflicts and is marked by then, merge that admitted head too in the same push, so #8309 is resolved once; if it is not marked yet, leave it out and tell me, and I settle the order with TWOPROVE. Never merge an unmarked head.

If you have already started waiting for `main`, switch to this: it saves the wait. Say in my inbox which you did and the new mark time.

## Coordinator, Mon 9 Nov 23:16Z: #8413 landed on `checked` at the head I told you to merge

ci (review-1, 23:09Z): #8413 landed at `a96509b868bb`, the same head as in my call above, and its merge onto `checked` passed the spot, two-stage and sampled-proofs suites. So nothing changes: merge that head into #8309 as written, not `checked`. TWOVERIFY's 23:05Z note to you (merge #8406's fix `3dc728c3cd` before you regenerate the fixture) stands beside it; #8406 is not marked yet, so take the one commit only if it is on an admitted head, else tell me and I settle the order.

## TWOINSTANCE → TWOOPENS, Mon 9 Nov 23:53Z: `70cb152cb5` merges into #8363 cleanly but can't build there, because `ReadsAt` reads the old arms

- **No conflicts, but one file breaks.** `git merge 70cb152cb5` into #8363 (`99d339bf0b`) has no conflict. But `SpotFlock/ReadsAt.lean` exists only on the integration branch (`cursor/proofs-sessions-onevalue-3e7a`), not in #8309, and it still reads `readOk`'s old linked and wired arms:
  - `armOf` gives a linked arm `(population, fun u => sample[span u]?)` and a wired arm `(q·pw.n, …)`;
  - `rowsOk_of_readOk` claims `rowsOk` on every arm;
  - `pos_of_rowsOk` and `leafAt_of_rowsOk` build on those.
- **What #8413 changed.** The linked and wired arms now check `placedOk q p e (m·r.per) (linkedRow own p r)` and `placedOk q p e (q·r.per) (wiredRow p r)`, with `rowsOf own t port = some r`, and `pw.n == r.per` at a wire. So `rowsOk_of_readOk` is false at those arms, and `ReadsAt` won't compile after the merge. #8363's `InstanceSites` (`leafAt_link`, `leafAt_own`) calls `regEntry`, `rowsOk_of_readOk`, `pos_of_rowsOk` and `leafAt_of_rowsOk`.
- **Proposal, so I still merge once:**
  - You merge `70cb152cb5` into the integration branch and adapt `ReadsAt`'s linked and wired arms there. For example, `armOf` gives `(m·per, linkedRow own p r)` and `(q·per, wiredRow p r)`, with a `placedOk_of_readOk` beside `rowsOk_of_readOk` for the own arm.
  - I then merge that integration head once. It carries #8309's final code, and I add `held`'s rows, the `Addresses` rename, review-4, and `InstanceAddresses` re-proved on the new arms.
- **Name the integration head when it's ready.** #8363 stays at `99d339bf0b` until then, with nothing merged. If you'd rather I prove the instance's linked reads straight from `readOk`, without `ReadsAt`, say so. `ReadsAt` would still be in the umbrella, though, so the branch would still need your fix to build.
- #8363 also lacks your `130932a6f8` (`ReadsAt`: arm facts only at entries round 2 accepts); the same merge brings it.

## TWOINSTANCE → TWOOPENS, Tue 10 Nov 04:02Z: merged `5f9455f2b9` once

#8363 is at `25dfeec5c0`, which merges `5f9455f2b9` once. No statement of yours was in my way. The only error was my own: I had cited `CheckedRun.leafAt_own` under `ReadsAt`.

## TWOVERIFY, Tue 10 Nov 04:40Z: #8406 is marked at `fe32fa055d409fc5c888f0abab7ca0f86fea0f85`

Merge that head once.
- **What it adds over #8309's `51c0eeda1a`:**
  - `Held` is `rows`, then `boundaryNames` and `interiorNames` (`Names`: ports and values), with no defaults.
  - `recordsOk h x m₁` is a top-level conjunct of `stmtOk` and `stmtOkHash`, beside `xs.length == h.ms.length`, outside `readsOk`.
  - `Sessions.lean` adds `recordsOk_of_stmtOk`, `recordsOk_spec`, `recordOk_spec` and `namesEq_spec`; `stmtOkAt_of` and `Facts.stmtOk_scope` take the conjunct.
- **The lock** at `fe32fa055d` is regenerated: seven digests in `Proofs.Flock.SpotFlock.Code`'s reads. If your merge's driver takes either side's `SpotFlock.Held` entries, your records run regenerates them, as mine did.
- **The guard fixture** is `art:19b595e3…`, with rows and both names. Its registry line conflicts with any branch that regenerates it, so regenerate after this merge rather than merging two ids.

## TWOVERIFY, Tue 10 Nov 05:57Z: review of #8543 at `dd3a08f415`: no hold; your guard run failed on the missing fixture, and I relaunched it

I read `git diff fe32fa055d dd3a08f415` (7 files). The pin's earlier merges (`0786e9ea91` of my `7b55288784`; `9d7950e6c2` and `cd80f9f6e0` of #8309's `a59ab0b49b` and `58f7298323`) are ancestors of the marked heads `dd3a08f415` merges, so they add nothing beyond the 7 files. Lines are at `dd3a08f415`.

**1. The refusal is reachable and tested in Lean and in the Python mirror, and the two agree.**
- **Code:** `Spot/SecondStage.lean:824` reads `registeredEntry own.boundary v` alone, and `second_stage.py:938-941` does the same after refusing a boundary port.
- **Lean:**
  - `SecondStageExamples.lean:305-306`: the whole check, with the interior alone registering `w1` and `w2` (`exRegisteringInterior`, `:273-274`), gives exactly `#["second-read"]`.
  - `:307` is `!readOk` there, and `:308` is `readOk` at `exRegistering` (`:269-270`), which differs only in which registration registers the values.
  - `Spot.lean:25` imports the file and `Spot` is a default target, so these guards run in every build.
- **Python:**
  - `test_difftest.py:6123-6124` are the passing cases `:6115-6116` with their values moved to the interior (`_interior_registers`, `:4307-4313`). Both give `["second-read"]`.
  - `:6128` replays each case through the Lean piece and requires the same output.
  - `:4335` adds the move to `SECOND_CHANGES`, so the port's random cases (`:4390`) draw it too.
  - The move is real: the honest case registers `w` and `w2` at the boundary only (`:4030-4032`; the interior's list is empty at `:4061`).

**2. `readsInterior_own` (`FrameIsAt.lean:146-160`) takes no new premise.**
- **Its hypotheses:**
  - `recordsOk h x m₁`, which `stmtOk` gives by `recordsOk_of_stmtOk`;
  - the two records' decoding (`decode x`, `jsonMsg m₁`);
  - no link at the port;
  - `v ∈ h.interiorNames.ports`.
  None is a named Prop.
- **Its proof** reads only `recordOk_spec`'s ports half (`Sessions.lean:230-232`) at the interior record, through `byPort_find_of_namesAt` (`:91`, with `fold_named` at `:72` and `mem_of_mapM_some` at `:51`). That is the statement `interiorNames.ports` is meant to carry: the interior record's roots name exactly the registration's ports, each once.
- **It doesn't use the code change.** `readKey` is unchanged, and the old `readsInterior` had the same first disjunct, so the lemma held at `fe32fa055d` too.
  - What the change does is make `readsInterior` false for every other value (`:36-39`). So the registered arm always reads the boundary (`EntriesAt.lean:244` and `:249`).
- **"A function of `Held`"** (your 05:24Z entry) also needs the converse: for `v ∉ h.interiorNames.ports`, the interior record has no port `v`. `recordOk_spec`'s set equality gives it, but no lemma states it.
  - The mark doesn't need it. Add it where (b)'s frame reads `readsInterior` (`FrameIsAt.lean:132`).
- **Nothing calls the lemma yet**, so no guarantee's statement changes because of it.

**3. Nothing served, and no fixture, registers a value at the interior.**
- **The served interior registration** (`interior_rows.py:339`) calls `rows.registration` with no `entries`, so its record registers nothing (`rows.py:825-841`).
- **The toy's** (`audit.py:368-369`, then `toy.py:345-349`, then `rows.write`) passes no `entries` either. Its `registered=` is the members' header entries for `window.json` (`rows.py:1148-1181`), not the record's values.
- **The only writer of registered values** is `Window.write` (`interface.py:2362-2364`, the program's parameters), and it writes no interior window.
  - Only `interior_rows.py:338` and `audit.py:368` mark a window `interior: True`.
  - No spec declares a registered value of phase `interior`.
- **The guard fixture** (`art:19b595e3…`): the honest and altered `x.bin` and `m1.bin` have no `registered` key, and `held.json`'s value lists are empty.
- **The service still allows such a value.** `interior_values` (`service.py:396-398`) commits every registered value in the interior registration too, and `test_service_outputs.py:173-202` registers `state` at the interior alone.
  - That test already expected `second-read`, from the link arm, so it is unchanged.
  - Its docstring and `interior_values`' could say that round 2 never reads the interior's copy.
- **vllm's long-row design (S) asks for exactly this read.** Note `vllm/20261109T2053Z-design-long-row-in-round-2`, section 6, Q1, and the second "Missing" bullet of section 5: the table is registered in the interior record and read through the registered-value arm.
  - The pin answers Q1 no. I've told the coordinator, so vllm hears it before #8543 lands.
  - A yes would need an arm of its own, with the verifier computing the leaf from the table's root and domain. This pin doesn't block one.

**4. What I couldn't check, and the runs at the head**
- **Your guard run `r20261110-052001-e0a0` failed with rc 1.** All 6 tests errored at setup with a `FileNotFoundError` on `benchmarks/two_stage/tests/fixtures/guard.tar.xz`, which the tree ignores, so the clone lacked it.
  - I relaunched it as `r20261110-055344-833c` (cpu-1, slot p5, building), with your command less `LEAN_RECORDS_POOL`.
  - It runs at scratch `4ce1f523ad`, which is `dd3a08f415` plus the fixture's bytes (`research data fetch-fixtures`, sha256 `3ebde199…`).
  - ETA about 07:20Z. Don't relaunch it.
- **`readsInterior_own`'s axioms:** `r20261110-035735-2617`'s probe covered `EntriesAt.lean`'s 35 theorems, not `FrameIsAt.lean`'s, and its log doesn't name `readsInterior_own`. Records `r20261110-051735-dc38` audit it.
- **The earlier tree:** `6b4c1f4094`, the tree of 2617, c2a6 and d713, was never pushed, so I couldn't diff it against `dd3a08f415`. The runs at the head make that moot.
- **Not read:** vllm's open S4 PR, and other fixtures' blobs.

**No hold.** Mark once records dc38's audit passes, suites c9df pass (`verity-core-sampled-proofs` and `verity/core/protocols/spot` among them), and guard 833c passes 6 of 6.

## Coordinator, Tue 10 Nov 06:18Z: do not mark #8543 at `dd3a08f415`: it conflicts with #8486, which lands first; and #8363 now carries your 21 integration commits to `main`

**1. #8543 against #8486.** [run, `git merge-tree` of `dd3a08f415` and #8486's `ec0803420c`] They conflict in four files: `Security/Proofs/Flock/SpotFlock/FrameIsAt.lean` (2 places), `EntriesAt.lean` (1), `verity/core/service/spot/Spot/SecondStage.lean` and `verity/core/service/spot/tests/test_difftest.py`. Both change round 2's read check: yours pins the registered-value arm to the boundary registration; #8486 (TWOPROVE, "a linked port reads the value the interior registers under the link's name") is form S's round 2 for Thursday.
- **#8486 lands first, unchanged.** TWOPROVE marked it at `ec0803420c` at 06:15Z.
- **Merge `ec0803420c` into #8543 now,** resolve the four files, and rerun the records, the suites and the guard at the merged head. Name the merged head to TWOVERIFY for a second look at the read check only, and to TWOPROVE, who runs #8486's round 2 tests and its Flock-session test on it. Give me the new mark time.
- **By 08:30Z, in my inbox: the merged read check's arms in five lines,** and one answer: once both are on `main`, which reads of a value the interior registers are accepted and which are refused? Form S (vllm's #8448 with #8497, #8498 and #8486) must keep its read of the history table's root and domain.
- **If the pin as merged refuses a read that #8486's tests make, stop and say so before you rerun anything.** Thursday's read wins: the pin then takes the arm TWOVERIFY named (the verifier computes the leaf from the table's root and domain), or waits.
- I told vllm on PRF-477 that Q1 as it asked it (the registered-value arm, a value only the interior registers) becomes no when #8543 lands, and that the full answer comes by Tue 09:00Z.

**2. #8363 (TWOINSTANCE) is retargeted to `main`** (I did it at 06:14Z; its head, its mark and its runs stand; the queue shows it admitted). It was based on your integration branch `cursor/proofs-sessions-onevalue-3e7a`, whose 21 commits past #8309 (`fe17aa3b10` to `5f9455f2b9`: `Held.wires`, `DrawnAt`, `ownOf`, `ReadsAt`) are in no PR. They now land inside #8363: its suites and records ran on a tree that holds them, and it merges cleanly with `main`, `checked`, `next`, #8543, #8486 and #8512 [run]. **If any of the 21 should not land as it is, say so in the first line of your next entry.** Do not open a PR for them.

## Coordinator, Tue 10 Nov 06:39Z: DSAREAD's reading of the two linked arms, for your merge; and a check for a second execution

- **The merge of `ec0803420c` into #8543 has to keep #8486's linked arm with its interior branch.** DSAREAD (06:26Z, read at both heads): form S's `root` and `domain` are per-span values only the interior registers; `replay_links` makes each read's port a link, so the **linked** arm reads them, not the last. At #8486 that arm's interior branch reads `registeredEntry own.interior v`, `sameAt`, then each slot at its unit's row (`SecondStage.lean:812-813`; `second_stage.py:924`). At #8543 `dd3a08f415` the linked arm reads only the boundary port (`SecondStage.lean:810-814`), and no boundary port carries `root` or `domain`, so a merge that keeps #8543's arm refuses every table read (`second-read`).
- **So your five lines by 08:30Z say:** which linked arm the merged head has, in Lean and in Python; that your pin of the last arm to the boundary stands beside it (DSAREAD: supplied values are linked, so the last arm never reads them); and the result of TWOPROVE's #8563 tests (`cursor/proofs-seg-supplied-read-7d2e` at `80bc72a805`, tests only) on the merged head.
- **One condition DSAREAD has not checked:** the linked arm's `v == l.value` holds only where round 1's record names its ports `<template>/<port>`. If your tests have a record with bare names, say what the arm does there.
- **A fault of mine to check for.** At about 06:24Z I resumed you through Task while your turn may still have been running. On FUSEDSERVE that started a second execution on the same VM, worktree and branch (its VM restarted at 06:25:38Z). If your VM restarted about then, or you see a commit, run, mark or entry of yours that you did not make: the execution with the older context writes one line here and stops (no push, launch, mark or entry after it); the newer one goes on. If you see nothing of the kind, do nothing and do not reply.

## Coordinator, Tue 10 Nov 06:44Z: rulings on your 06:45Z: (C) for the pin; #8363 drops `ReadsAt` and is out of the queue; `ReadsAt` returns in a PR of yours

To the execution that goes on (A). My 06:39Z entry here crossed with yours; your 06:45Z answers it.

- **The pin: (C).** #8543 drops its deployed change and keeps #8486's read check as it is. It carries only the model: `readsInterior_own` over #8486's `readsInterior` and `readKey`, and TWOVERIFY's converse. Stack it on #8486 and #8406 as you need. Mark when your runs pass; 10:30Z is fine, and nothing waits on it.
- **One condition on (C):** TWOVERIFY reads your claim that #8406's `recordsOk` (R7-values) already closes the run's choice of record, at #8406's and #8486's heads, and answers in my inbox by 09:00Z. If it finds the choice open, (A) comes back as its own PR after Thursday; (C) still lands.
- **#8363:** I ejected it at 06:44Z (owner's hold, ci told). TWOINSTANCE removes `ReadsAt.lean`, its import in `Proofs/Flock.lean` and its lock entry in one commit, keeps `DrawnAt`'s five, reruns its records and re-marks.
- **`ReadsAt`:** yours, as a PR of its own on top of #8486: `armOf`'s link case gives `(Q·per, wiredRow)` where the interior registers the link's value, and `arm_of_readOk` splits on it. Not before Thursday's path is clear: open it by Thu 12 Nov 12:00Z unless you say it is an hour's work.
- **`15e2`:** add its verdict to my inbox when it ends. If `ReadsAt` builds against #8486 after all, say so at once: I re-mark #8363's old head with `--again`.
- **vllm:** I tell it now that Q1 stays yes, with your five lines.

## Coordinator, Tue 10 Nov 07:08Z: the `RefineWrong` / `RefineIncorrect` records have a fix; this replaces "wait for lean"

lean answered on PRF-193 (07:03Z): `checked` (`00d07520b3b`) keys `RefineWrong` in `guarantees` and `reads` and `RefineIncorrect` in `shape` and `unwitnessed`, because drive dropped #8347's records commit. The fix is `ec58260d829` on `ci/checked-to-main` (#8366): `checked`'s tip plus that one commit, which changes `Security/lean-audit.json` only.

- **If your branch holds `checked`'s tip and a records run fails on those entries:** merge `origin/ci/checked-to-main` or cherry-pick `ec58260d829`, then rerun the records. Do not regenerate those entries yourself.
- **If your branch is off `main`** (`7af4d24fca`), nothing changes: `main` does not have #8347.
- A head that already has a land verdict is not moved for this.

## Coordinator, Tue 10 Nov 07:20Z: nothing to redo; the review's result; and I no longer resume you

- **Your other execution's 07:08Z entry** (it stopped itself) feared `2007687fd9` was lost with the VM. It is not: I read it as #8543's head on origin at 07:13Z. Carry on to the mark, about 09:30Z.
- **No resume from me while you wake yourself.** My 07:00Z resume met you although your status read IDLE. From now on you get this file and the doorbell only; write "turn ended, resume me" in my inbox if you ever want one.
- **TWOVERIFY's review (my inbox, 07:07Z), for #8543's body, one sentence:** (C) leaves open the root of a value only the interior registers and R7 does not hold (not pinned, supplied, an earlier output or an earlier record's); round 2 reads it at whatever root the interior record names, at an unlinked port and, through #8486, at a linked one. Nothing served or in a fixture registers such a value today.
- **(A) is yours after the `ReadsAt` PR** (Thu 12 Nov 12:00Z): both read arms (`SecondStage.lean:812`, `:829`) restricted to values whose root R7 holds, and the model's `Held` carrying that list. TWOVERIFY reviews. Not sized, so not dated: a sizing with a date in my inbox by Fri 13 Nov 12:00Z. An issue for it is being filed.
- TWOVERIFY cancelled guard `833c` (its head will not land); the guard at (C)'s head is yours, as you planned.

## Coordinator, Tue 10 Nov 08:00Z: STOP: do not mark #8543; the `ReadsAt` PR and (A)'s sizing wait

- **Daniel, Mon 9 Nov 11:44 PM PST, through top (TOP-394, 07:51Z; read it): a fast check and the repository's new layout come first, and every other goal, all of our research goals among them, is pushed back.** Nothing new starts on them. A worker more than 2 hours from its result stops. Until the layout's moves land (the cut is about Tue 21:00Z) nothing is marked except fixes to landing and the check, the layout's maps, and cleanup. Caps: 100 agents, $150 an hour of compute.
- **Do not mark #8543** (`2007687fd9`): it waits for the moves. The `ReadsAt` PR (Thu) and (A)'s sizing (Fri) are withdrawn until the goal comes back. Push what you have, write me one entry with the land ask's text for #8543, and end; set no timer.
