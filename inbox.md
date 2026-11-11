[2026-11-11T13:50Z] d5b25ce39304
LEANMERGE to the network-accounting coordinator, Wed 11 Nov 5:50 AM PST. Job 4.

**Head `f1909ac4ab2793f3314f55b6a433e82d4ece7a10`, pushed to `cursor/net332-json-is-7dcf`, no PR. The network package's audit passes at it** [run `r20261111-132433-9341`, exit 0]: `AUDIT verity/core/protocols/network/lean: PASS 1528 declarations in 9 modules; axioms propext, Classical.choice, Quot.sound; 4 guarantees`, then `AUDIT: PASS`. The package was audited alone, with the kernel replay and its test, no `--update`. Its own record does not differ (`audit.json`: `regenerated: {}`), so there is no third commit. **The suites run `r20261111-132840-e0d3` is still running**: 4 of 53 suites passed, none failed, at 5:48 AM PST.

## The two commits [vm, `git show --stat`]

| commit | file | lines | what |
|---|---|---|---|
| `3b171993f551714e36961623c48d043d1652808d` | `verity/core/protocols/network/lean/lean-audit.json` | +2 −1 | `"Flock.JsonIs"` in `layers["NetworkCheck.Period"]["imports"]`, before `Flock.Program`; one clause appended to its `why`: "C-Flock's test that a JSON value is a given string (how the check compares a field with a string)" |
| `f1909ac4ab2793f3314f55b6a433e82d4ece7a10` | `Security/Properties/lean-audit.json` | +13 −7 | `9a2c`'s kept record, byte for byte |

- **The lock** [vm]: the file is `check.text` of its own JSON before and after, so no byte outside those lines moved. As parsed JSON only that layer's `imports` (one added, none removed) and `why` (its two clauses kept in order, one added after `; `) differ. `merge.py:436-457` merges a `why` clause by clause, so the clause is one item [read].
- **The record** [vm]: sha256 `abccf759aedafe3569866f35a8c257748482b181d22fea1ce7b6f4b4e70bde3b`, the file under `lean-audit/record/Security/Properties/` in `9a2c`'s files. Against the branch's file: 20 lines, and as parsed JSON 8 values, all under `reads`, the ones you named: `Flock.JsonIs` added (digest `9210fbcb…`, `Flock.J.strIs` `35c873bd…`), `NetworkCheck.Period`'s digest, and its `auditCheck`, `canonicalEntry`, `holds`, `rowsWindowCheck`, `slotEntryDigest`, `windowCheck`. Same keys in the same order, nothing else.

## Runs, all at `f1909ac4ab27`, through `research run --on auto`

1. **`r20261111-132433-9341`, the network package alone: done, exit 0**, 235 s of wall, 87 s of them check.py [run]. The verdict line is above. `audit.json`: `passed: true`, `failed: []`, `regenerated: {}`. Replay: 1518 declarations, the three axioms. The lock's test: `difftest: 2000 instances compared, 0 differences`.
   - It is `check.py --build` on the one package, not the fast path. `lean_changed.py --records` has no option to leave out the requirers [read, its argparse], and they are `Security` and `Security/Properties` (4325 s and 1157 s in `9a2c`).
   - It is not the very cheapest form: I left the replay and the test in (35 of the 87 s). So its line is a full audit's, where `9a2c`'s was the records audit's.
   - [vm] I replayed the layer rule (`check.py:1445-1476`) over this run's own import graph (`graph.json`). With the lock at `db845792d9ef`, `Flock.JsonIs` is outside the layer; with the head's, nothing is. So it passes for the reason the commit gives.
2. **`r20261111-131837-f562`, the suites, first launch: failed, exit 2**, 73 s [run]: `suites: … is not a git checkout, and suites are keyed from git's view of the tree`. No test ran. My launch's fault (below).
3. **`r20261111-132840-e0d3`, the suites again: still running** at 5:48 AM PST, 20 minutes after its launch, so I stop here [run, its `stdout.log` so far]. It runs `suites.py --changed db845792d9ef… --quick` and found the base in its tree: `suites: the change from db845792d9ef touches 53`. Passed so far: `benchmarks/pouw/pearlplus_lean`, `verity-core-periods`, `verity-core-sampled-proofs`, `research`. None failed. Resume me when it ends.

## Not asked

- **Two launches of one commit in the same second race on the ship** [vm, read]. Both push the commit to one ref of the machine's bare repository. One push fails, that launch ships the archive stream, and the tree is there without git (`remote._mark_ready`: "another launcher won the race: theirs stands"). That refused my first audit launch (`--cwd clone`, never a run) and failed `f562`. The same push alone succeeds. **This corrects m1**: "the git ship fails from my VM" was most likely this race [argued: that job's runs went out together too]. Friction note: `lanes/network-accounting/20261111T1333Z-friction-two-launches-race-the-ship.md`. Three other lanes' reports show the same pair of symptoms.
- **How `9341` ran** [vm]. The tree had no git by then, so the run is `--cwd source` and audits a scratch copy of the commit's 14,290 tracked files, each checked against my `sha256sum` of `git archive HEAD`. It writes nothing into the shared tree. check.py calls no git [read, by grep]. Script: `tools/net332-leanmerge/audit-alone.sh`.
- The `note:` line differs from `9a2c`'s: 33 definitions read by no guarantee here, 296 there. [argued] My copy is the whole tree, so the dependents' locks count as readers (`check.py:3775`). I did not check what the fast path's tree holds.
- The layer's `why` already held two near-duplicate clauses (348 and 404 characters, an earlier merge's two sides). I left them [vm].
- The clause says "C-Flock", the lock's own word (8 uses before, 9 now). The Glossary retires it in docs, and neither retired-terms lint reads a lock [read].
- Every requirer's inputs key moved with the layer commit, since the network lock is in their closures [vm, `lean_changed.py --inputs-key`]. So `9a2c`'s `Security/Properties` pass is not reusable at this head: the next walk reruns both requirers.
- The lean-proofs skill's ruling still says a string is compared by `Spot.jsonStrIs` (`SKILL.md:192`). The branch has a second, `Flock.J.strIs`, and does not touch the skill [read].

## Least sure, and where I did not look

- **That `Security/Properties` passes at this head with the committed record.** No run audited it here. [argued] The two commits change no `.lean` file, and the record is the one that passed after `--update` at `db845792d9ef`.
- `Security` at this head: not run. I expect `9a2c`'s three lines again.
- The suites: 49 of 53 not yet reported.
- I read check.py's uses of the repository root and of git, not the whole file, before auditing a copy rather than a clone.

