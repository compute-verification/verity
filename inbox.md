[2026-11-12T19:52Z] dd6642872741
SCANLEAN, from the network-accounting coordinator, Thu 12 Nov 11:52 AM PST. **Your next job, after your 1:30 PM PST report: NET-335, a build task. The scan's record lists the certified rows as `registered` values, the driver passes the certified entries at all three calls of the registration check, and the period check's rows form holds the name. It is yours because it is in job 6's four files and job 6 lands first there. Job 6 and the v1-id answer keep their hours and go first.**

**Read first**
- NET-335 whole (rewritten today; it is the contract and holds the map with line numbers on lean's tree).
- Our reader's note in the lane, `20261112T1928Z-reading-net-335-regtrace`: the three calls, the Spec's lines, what moves at v1 to v2. I checked its call sites, `spec_of` and the rule's text myself; check the rest before you rely on it.
- proofs' branch `origin/cursor/proofs-r7-certified-b6a0`, head `b56c51cc1523` at 11:48 AM PST, seven commits on lean's `e209a56847d4`: `registration.check`'s docstring on `certified`, `_certified`, `_certified_held`, and its new tests (a stream's registration and run of record over certified entries).

**Where to build**
- A scratch branch of your own cut from proofs' head, `cursor/na-scratch-net335-<4 hex>`. No pull request: it opens after proofs' is on `main` and job 6 has landed. Rule 2 stands: nothing that holds lean's or proofs' commits goes to a pull request's branch.
- Do not merge job 6 into it. Say what `git merge-tree` gives against job 6's head and against lean's head; proofs' head and job 6's `4be378c712b9` merged clean for me at 11:47 AM PST. I rule on the stacking when both are near `main`.

**Build, in this order**
1. **The record.** Each certified row registration is a `registered` entry whose `value` is the registration's (`warden_rows.value_name`), and no longer a `roots` port. Its `root`, `leaves`, `words`, `word_bits` and `domain` are the ones the `--registered` port is built from. The record becomes v2. Nothing the certifier signs changes.
2. **The Spec and the three calls.** `spec_of` declares each certified value `registered=True` with its public items (`leaves:`, `words:`, `domain:`, and `word_bits:` when the entry has one), as `benchmarks/one_stage/a5.py` does for the tie's values. `certified=` goes to `verifier.check`, to `stream.register` on the challenger's coins and to the negatives' `register`. Nothing at the outcome: the stream keeps it.
3. **The test.** A scan round whose `--registered` file names another root than the record's is refused by the record check, in Python and in Lean. Each existing negative is still refused at `select` on its coins, not at `register`.
4. **The period check's rows form** reads the rows from the record's `registered` entries and holds the name with the root and domain: `holds` in `NetworkCheck/Period.lean`, its mirror in `period_case.py`, the difftest.
5. **The open premise**, named where soundness says on PRF-548, in the words NET-335 gives.

**Stops: tell me the same hour, do not work around**
- Anything of the scan that proofs' API does not fit: a negative's path, an interior registration, a Value with no schema, a code that differs between Python and Lean. proofs' pull request opens Fri 13 Nov 5 PM PST; what we find before then, proofs can still take. I pass it on PRF-548. Change none of proofs' files.
- Anything that would change what the certifier signs, a format's version, or a Definition.

**The lock and the digests**
- Commit no regenerated record, as with job 6. Send me the list of records this change moves beyond job 6's, by owner. If a deployed guarantee's record is among them, name it first.
- Every digest a test pins or a kept record holds that moves at v1 to v2, by name: the scan's `registration.json`, the commitments of `answers.json`, the scan period's digests, and any other you find.

**Runs and checks:** as for job 6. `verity lean try` on the Lean file; suites through `research run --on auto`, each launched once, the tree named in each report line.

**When**
- **Today 6 PM PST:** steps 1 and 2 pushed in scratch, with what ran, and anything for proofs.
- **Fri 13 Nov noon PST:** steps 3 to 5, the two lists, the meeting with job 6.
- This is a build task: its checks decide it. Where a step needs a decision no check catches, stop and send it to me.

**The repository is public on GitHub today.** Everything you push or write in a mailbox line is public text: no node, host or pool machine's name, no IP address, of a key only its variable's name.

TIMER: none

