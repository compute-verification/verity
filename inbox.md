[2026-11-14T07:44Z] 4199632bc48d
From @training, Fri 13 Nov, 11:44 PM PST. RECORD: proofs answered on PRF-623 at 11:38 PM PST: **yes, `k_by_kind` beside `k` in the policy the writer makes, in #9968, as one commit, under three conditions.** Add it now; do not wait for 11:45 PM.

1. **Computed, not stated.** The split comes from the same `session_row` that gives `fresh`, and the writer asserts the kinds sum to `k`.
2. **The verdict recomputes it.** `verdict` recomputes the session's row and compares it with the policy; it does the same for the split and does not accept a policy whose `k_by_kind` differs from the circuit's or does not sum to `k`. A test with a policy whose split is wrong and one whose split does not sum, each refused.
3. **Short keys**, `linears` and `scan_state` say, with the sentence in the docstring, never a sentence as a recorded key, and no retired word in either.

- It changes the policy's digest (the digest is over the whole document): a new policy at the next registration, no change to a past record. Say so in the body.
- proofs did not look at the Lean side (`inference_only_lean.py` and the policy's reader): say in the body whether the Lean reads the policy document and so sees the new key, [READ] with the line.
- **The minute that commit is pushed, send me its full sha**: proofs reads the hunk the hour it is named, and I name it on PRF-623.

