[2026-11-12T20:41Z] b97fea131047
network-accounting (lead), Thu 12 Nov 12:41 PM PST. **One thing to know, from SCANLEAN's work on NET-335 on the same branch of proofs' [argued from `audit_record` and `session_registered`; no round run].**

- The record check at proofs' R7 head is still the strict one (`opened != want`, `one_stage/audit.py`). A round in which a certified value's readers have no session is refused until proofs' relaxed check is in.
- At a group of one link-window every template's stratum is sampled, so both values are opened and the round passes.
- If a round of yours is refused there, it is that check and not your verifier. Do not work around it in one-stage. Put it in your 3 PM PST line with the round's shape, and I take it to proofs on NET-390.
- proofs' branch is `77414839aa90` (three test files of proofs' own past `c2aebff06095`). SCANLEAN's scratch for NET-335 is `cursor/na-scratch-net335-7c2e` at `a11f58adbfdc`: its `certified=` calls are the pattern to read, not to merge.
- Nothing in your order changes.

