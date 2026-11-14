# CARRIER, 05:01Z Sat 14 Nov: the ChargeOnce.lean change passes its try and is ready to cut; #9513 is not on `main`, so there is no pull request, and the entry goes from 5 to 2

Clock read with `date -u` at 05:01Z. **Step 1 is done as far as it can be before #9513 lands: the edit is made, tried and pushed on a branch over #9513's head, and a rehearsed script cuts it from `main` the moment #9513 is there. #9513 is not on `main` (`89605f38f901`), so I end my turn here (your step 4).**

**The branch, and why it is not the one you named**

- [run] The change is on `cursor/proofs-carrier-9513-terms-scratch-9257` at `a05073714cbf2b4d8e25555a0975e646189a0a79`, pushed, with no pull request. It is one commit over #9513's head `a2643447fa06`, so it holds #9513's commits and is never a pull request.
- [run] `cursor/proofs-carrier-chargeonce-terms-9257` does not exist yet. `main` has no `ChargeOnce.lean`: #9513 adds the file. A branch cut from `origin/main` today has nothing to edit.
- [run] The cut is rehearsed. On a local merge of `main` `89605f38f901` with #9513's head (never pushed, deleted since), my script (`cut-chargeonce.sh` on `cursor/proofs-carrier-scripts-9257`) made one commit over it: 2 files, 9 lines in and 9 out, and the lint's count of the file's sites 2. A plain patch of the list does not apply there, since `main`'s list has other lines beside the entry than #9513's; the script rewrites the entry's one line instead.

**The entry goes from 5 to 2, not to 1: your brief says 1, after my 02:52Z note, which was wrong**

- [read] That note said the file's body uses nothing of the namespace `FlockSoundness.Audit.Periods`. The statement of `ex_prod_add_le` reads two of its definitions, `Adapted` and `Predictable`.
- [derived] With the three theorems in `FlockSoundness.Audit.ChargeOnce`, the file must open the old namespace, and that line spells the retired word.
- [run] The lint of `main` `89605f38f901` counts 5 sites in the file at #9513's head and 2 at mine.

**The 2 sites that stay**

- Line 4, `public import Proofs.Flock.Soundness.Audit.Periods`: the module's name.
- Line 33, `open Claims FlockSoundness.Audit.Periods`: the namespace's name.
- [run] Both are names `main` declares in the audit theorems' module. Six other Lean files and two verifiers' Python files on `main` spell them. Renaming them is not in this change.

**No lock record moves**

- [run] The three theorems' full names change (`Ex_le_of_forall`, `Ex_fin_cons`, `ex_prod_add_le`). At my head no `.json` or `.toml` file names them, their module or either namespace, except the retired-terms list's entry for the file. No other Lean file reads them. One README names `ex_prod_add_le` by its short name.

**Tried over #9513's head** [run: `r20261114-043550-1be1`, vy-nebius-cpu-10, done, rc 0, read at 04:59Z]

- [run] **The file compiles under the new namespace**: `lean exit 0 in 50 s`, and the axiom probe says none of its 3 theorems uses an axiom beyond `propext`, `Classical.choice` and `Quot.sound`. `TRY_EXIT 0`.
- [run] **The sibling check finds no clash** among the 2,240 modules it imported. It left 1,829 unchecked, which have no build of their source as it is in that slot tree. [run] `git grep`: no other file spells `ChargeOnce` as a name, so none of those can declare one of the three.
- [run] **The module that imports it builds**: `lake build +Proofs.Flock.Claims.ChargeOnce +Proofs.Flock.Claims` exit 0 in 1,102 s. `BUILD_EXIT 0`.
- [run] **The retired-terms lint by name**: 17 passed, 1 warning. `LINT_EXIT 0`. The warning is the list's bookkeeping one, and it does not name `ChargeOnce.lean`.
- [run] **The ratchet gate** on `a2643447fa06..a05073714cbf`: passed, 0 failed. `GATE_EXIT 0`. The tree is unchanged at the run's end.
- [run] `r20261114-040330-d84b` is cancelled, by me at 04:35Z (rc 143). It had waited 32 minutes behind 2 for cpu-4's one Lean slot and had run nothing. `1be1` is the same job through `--on auto`, which started it at once.

**When #9513 is on `main`** (you resume me)

- I cut the branch from `main`, and run one job through `--on auto --cwd source`: the lint, the ratchet gate, `suites.py --changed <main> --quick`, the try, the build, and `lean_changed.py --records security/Proofs`. Then I open the pull request unmarked and write `ready-<PR>`.
- **One question: the `Issue:` line.** PRF-361 is the lane's issue for retired terms [read: its title]. Without your word I write `Issue: none:` with the reason.

**Not established**

- Nothing here is at `main`. The suites and the records' fast path have not run on this change; they run on the pull request's head.
- The run's own lint is #9513's tree's. `main`'s lint changed after `245da271367f`; the counts above are by `main`'s as it is now.
