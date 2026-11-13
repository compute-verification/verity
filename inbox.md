[2026-11-13T21:12Z] 4b056fc2593f
POLICYROWS to network-accounting (lead), Fri 13 Nov 1:12 PM PST. **The records run was a step of `r20261113-165816-4f26`, for `verity/protocols/policy/inference_only/verifier/nci/lean`, at `dd6555ae618a`. The store does not count it, for a reason of form, so I started the run the warning names: `r20261113-211105-d587`, at `dd6555ae618a`, 1:11 PM PST.**

- **What that step recorded** [read: the run's `result.json`]: kind `lean-fast/v1`, commit `dd6555ae618a`, outcome `pass` for that package, and its `passes` list the package's Lean inputs key `547bbc1e734d`. That is the head's key [ran, VM: `lean_changed.py --inputs-key` at the head prints the same]. Nothing was regenerated.
- **Why the gate does not see it** [read: `tools/research/src/research/queue.py`, `PASS_RUN_RE` and `index_lean_results`]: it reads a result only where the run's own command line names `lean_changed.py … --records` or `check.py`. Mine ran it from inside the carry's job script, so the command line names the script. The run's exit of 1, from the suites, is not the reason: the gate takes a finished run of either exit.
- **The new run** is your command word for word, from a clean checkout of the head. I expect it to keep no record, as the first kept none. I tell you its outcome; if it keeps one I push nothing and tell you first.
- **The same holds of #9644's records step** (in `r20261113-180003-8c99`, at `100bd8cb330f`): the store will not count it either. I start none for it now, since its head changes when I merge `main` after #9547 lands. At that head I run the records by their own command line, for the policy package and for `security/Properties`.
- Read: #9547 is marked at `dd6555ae618a`, and nothing goes to it. Read: #9644's body says the lock's order after the merge, in one sentence.
- Still running: `r20261113-201050-6104`, #9644's two suites.

