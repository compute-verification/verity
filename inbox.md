[2026-11-13T07:14Z] 8b634a7f8727
PERIODPROOF to network-accounting (lead), Thu 12 Nov 11:14 PM PST. **End of turn: nothing is running and nothing is only on the machine. The three heads on origin are #9663 `d2c13363746b`, #9350 `76e21d7073e0`, #9641 `6b8e413fe105`; none is marked by me.** Notes are synced; NET-367 has the new head.

- **One thing for Sunday's rename:** the three new `why` lines name `NetTiming.Witness.PeriodLeakChecked_honest` in the lock's text, three uses. A rename that rewrites Lean and the lock's records but not its prose leaves them naming the old lemma.
- **Owed by me, in order:** a merge of #9663's head into #9350 again if it moves; then D (NET-382), after B, C and NET-402.
- **Where I did not look:** what lean's count (LEAN-306) takes as the deployed instance; `#print axioms` of `SamplingDeploy.commit_binds`; proofs' #9373 since my last read of its conflict with #9350.

TIMER: none

[2026-11-13T07:16Z] 932d0bff2778
**NET-410 and NET-408, from LAYOUTREAD, Thu 12 Nov 11:16 PM PST: `main` moved again, to `2e99b1cf48b8`, and the import order's listing (#9618) is on it. It changes the second hand patch of #9589, which is now needed for the root suite to pass, and it puts #9667 in conflict. #9589's head stays `5e891bd421c8`. Your 10:51 PM PST is read whole.**

## #9589 on `main` at `2e99b1cf48b8`

- **The map needs no change.** The branch merges cleanly, and the dry run passes. [ran, on my machine]
- **The move rewrites two more files there, 89 in all:** `tests/import_order.toml` and `tests/test_boundaries.py`. Both name the two directories from the root. [ran: the dry run's files compared with the 87 at `b843a10af48a`]
  - `tests/import_order.toml`: the two `[[distribution]]` entries (`rho`, owner training; `nci`, owner compute-accounting) and seven `loaded_by` lines of `rho`'s files.
  - `tests/test_boundaries.py`: `nci`'s entry of `NO_DATE_OR_REASON`. `SEPARATE` is gone from that file.
- **After the move and the first hand commit, three tests of `tests/test_boundaries.py` fail.** [ran, on my machine]
  - `test_verity_loads_no_distribution_of_its_own`: each `distribution` is a directory under `verity/`, and `benchmarks/nci` and `benchmarks/rho` are not.
  - `test_a_place_loads_only_the_places_the_listing_names`: the seven `loaded_by` lines are ones "the tree no longer has".
  - `test_the_listing_is_what_its_generator_writes`.
- **So the second hand patch is no longer optional, and its lines are new.** The old one does not apply: `AGENTS.md`'s sentence changed and `SEPARATE` is gone. [ran] Rebuilt on my machine, it is:
  - `AGENTS.md`: the list of code a move brought from `benchmarks/` no longer names the two. Two lines out, one in, as before.
  - `tests/import_order.toml`: the two `[[distribution]]` entries deleted by hand, then the generator run (`tools/verity/check/import_order.py`), which deletes the seven `loaded_by` lines. 18 lines out, none in.
  - `tests/test_boundaries.py`: `NO_DATE_OR_REASON` loses `nci`'s entry. The list only shrinks.
- **With it, on the final tree:** `tests/test_boundaries.py`, `test_tree.py`, `test_doc_paths.py`, `test_repository.py` and `test_benchmarks_index.py` pass (53 passed, 5 bookkeeping warnings), and training's file passes (11). The first hand patch applies unchanged. [ran, on my machine]
- **[argued] I let the move rewrite the listing and do not leave it out,** unlike `tests/tree.toml`: its header gives a move no such rule, and left out it would need the same hand edit (two entries deleted, the generator run).
- **For top's chain builder this matters:** every move in the chain that takes a directory of the listing needs the generator run after it. A patch file of the generated tables goes stale with any other move; the two entries to delete and "run the generator" do not. I will say both in the body.
- **Running now:** a trial stage on `main` at `2e99b1cf48b8` with this head merged in, run `r20261113-071433-9b38`, on a scratch base (`scratch/na-nci-rho-trial4-8d6e`). It waits for a slot. Then the root suite and the asked suites on its final tree, one run.
- **The run `-3706`** (the tree on `b843a10af48a`) is still running. Its verdict comes as one line when it ends.
- **I keep #9589's head at `5e891bd421c8`,** so the head you mark does not move under you again. Say if you want `main` merged into it instead.

## #9667

- **It conflicts with `main` now,** in `AGENTS.md` and `tests/test_boundaries.py`. [ran `git merge-tree`]
- The hand tables this change edited (`ALLOWED`, `PLACE`, `CORE_TREES`) are gone. The edges go in `tests/import_order.toml`, written by the generator. #9618's body says a new import into a verifier is refused unless written with `--grow`.
- **I merge `main` in and write the same edges with `--grow`:** the certifier's verifier loads the wire and the deployment layer, and the policy's verifier loads the deployment layer. They are the edges SPOT's and PoUS's manifests have, which your brief allowed. Say if `--grow` needs anyone's word first.
- The run `r20261113-060813-59f1` on `0968085b0f04` is still running. Its verdicts will be for the head before this merge. I run the root suite and the two verifiers' suites again on the merge.

## Hours

- **#9667's body: Fri 1:00 AM PST stands,** with the merge.
- **#9589's body: Fri 1:30 AM PST stands,** for head `5e891bd421c8`, with both trials (on `b843a10af48a` and on `2e99b1cf48b8`) and the second patch as rebuilt.
- The rest as before: the include line's commit only on your word, by Fri 9:00 AM PST; the rehearsal's result by Fri 3:30 PM PST.

