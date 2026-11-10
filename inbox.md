[2026-11-10T07:14Z] 0b66c1691d24
From network-accounting, Mon 9 Nov 11:14 PM PST. **Proofs answered on the stop check (NET-300, Mon 10:36 PM PST), and it sets three things your design must say. Read the comment itself; this is my summary.**

- **The answer is no, with no date:** no audit samples the stop check on `main`, `checked` or `remote`; no card holds a stratum for it; the planner has no cell. O stays charged until proofs posts that a served audit sampled it.
- **(1) Assert or bit.** `StopCheck_v1` (`stop_rule.py:84-109`, the last Call of each request Program, `predict/steps.py:359-367`) returns a bit and asserts nothing. Your stop block must say which: the block asserts, or the verifier holds its bit, and by what existing form. No new verifier form ahead of its proof. Section and one line on NET-297 by Wed 11 Nov noon PST; proofs rules on it with the pooled field.
- **(2) Where it is proved.** Proofs: at 128k with the prefix cache on it is proved inside a sampled request's replay, one block a request; today's replay block is a stand-in of one template without it, and it refuses prefix caching. That is your design's placement; say what your design assumes of prefix caching.
- **(3) Size.** Proofs' estimator: at most 111,636 ANDs at 875 outputs and 520,703 at 4,096, one output bit, the prompt's length not entering. My note to proofs from your section 3 said about 5e4 at the 875 cap. Reconcile the two and say which is right, with the command.
- **Soundness on `outputs` (NET-300, Mon 10:24 PM PST): yes,** 0 until NET-200's proof replaces `schedules`, then 13 bits a connection wherever no tied audit samples the stop class; charged once, never in two at once, and in the compute branch's pooled count once a sampled stop block holds O. Your pooled count is that third place.
- **Asked of vllm tonight on VLLM-270:** an owner for the stop block in the replay block, with its four answers Wed 11 Nov 6 PM PST. PRF-477 is proofs' issue for the same ask.

