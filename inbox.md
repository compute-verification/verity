[2026-11-14T02:13Z] a7eb79ae8dbd
CALLERS, Fri 13 Nov 6:14 PM PST. **TRN-67's draft is open as #9889 at `a6494c1f6a1c`, its changed suites passed, and main's lock holds HECharged only as a frozen, unstated shape entry.**

- `cursor/calculator-lean-names-56e2` is TRN-67's branch: the test that fails when the slowdown calculator cites a Lean name no lock lists, and the PR body's reading of the calculator's names beside the harness's. Draft https://github.com/compute-verification/verity/pull/9889, cut from main `6c52a8dd9ca`, `Goal training/one-slowdown-calculator`, `Refs TRN-67`.
- RAN `r20261114-015642-3bbb` at tree `a6494c1f6a1c` (`suites.py --fresh --changed 6c52a8dd9cae`): "5 suites: 5 passed (0 cached), 0 failed". `tests/test_public_boundary.py` passed in it.
- The lost run `r20261114-013334-e91d` was relaunched with no merge of main. I could not reread its argv under the run-JSON rule, so the relaunch is the testing skill's line with `--fresh --changed 6c52a8dd9cae`, which is what I recorded launching (INFERRED: the same). I had merged main locally before your message, never pushed; dropped it.
- RAN on my VM: `pytest -q benchmarks/rho/tests/test_lean_names.py`, 3 passed; ratchet gate 0 failed.
- RAN on my VM in the test: the calculator cites 35 Lean names, 34 listed. The one in `ALLOWED` is `min_cut.HE_CHARGED`.
- READ `security/lean-audit.json` on main: HECharged is no guarantee, signature, assumption, open, domain or reads entry. It sits only in `shape.frozen` as "unstated": proved from the count over a set of blocks taken as a hypothesis, "which no theorem proves yet". On `cursor/hecharged-from-count-a8e2` it is the guarantee `Nci.Cut.SecurityProofs.HECharged`. When that lands, #9889's `test_allowed_only_shrinks` fails until its entry goes.
- READ in the body's reading: `calculator.theorem()` always returns HE, while the record's `bound.theorem` is HEShared at F > 1, and its docstring says HEShared. Named as a follow-up, not done.
- `research queue open` warned that TRN-67's labels name the done goal `training/unread-bounds-deleted`, not the body's goal. I left the labels alone.
- #9877: understood. No merge of main before Monday, then one carry after the stack and the rename, then suites, the comparison again and one message.
- Handoff note `training-20261114T0213Z-handoff-CALLERS`, synced. Runs in flight: none.

