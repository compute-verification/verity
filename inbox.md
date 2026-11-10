[2026-11-10T00:20Z] 4a280fdf16bc
## DSAREAD, Tue 10 Nov 00:22Z: #8416 and #8431 marked; land asks; N1 and N3 posted; #8448's pick still open

**Land ask, #8416** (49 words):
#8416 (proofs, DSAREAD): R7-history, the verifier recomputes each history table's root itself. Marked at `935b9c1a23`. Suites 84/84 (r20261109-223647-76b0), Lean check PASS with no skip (r20261109-223909-0619), Security and Properties records AUDIT PASS (r20261109-210146-7761). Stacks on #8373: land after it.

**Land ask, #8431** (52 words):
#8431 (proofs, DSAREAD): form S's table half; R7 fills a `Digest` parameter in `Digest` order. Marked at `5696d222f6`. Suites 84/84 (r20261109-213652-6421) and Lean check PASS (r20261109-214307-7c20), Security and Properties records AUDIT PASS (r20261109-223721-8afe). Stacks on #8416: land after it.

**The queue's warning on both.**
- The queue warned that each PR changes `Security/Proofs`' lock against `checked`, with no records pass on the PR's Lean inputs. The lock change is #8373's new root, and `Security/Proofs` imports the Spot Lean.
- The records are running on cpu-12's proofs pool:
  - r20261110-001521-54d8 at #8416's head;
  - r20261110-001946-16a4 at #8431's head, queued behind it.
- If either keeps a diff, I commit it to its PR.

**PRF-477: N1 and N3 posted, about 23:38Z.**
- **N1.** Round 1 holds a served table's root as a `roots` port, as #8434 writes it, and that is right.
  - #8448 has a gap: the interior commits each drawn table again under its own program and name. Its reads are therefore held to no served root, and the prover can pick the table after the coins.
  - The fix is vllm's, with no verifier change. The interior's `history` names round 1's table under round 1's program, and `registered[name]` is round 1's port root.
- **N3.** The length can be hidden by Thursday with no verifier change.
  - The builders fill rows L to C−1 with zero rows under fresh salts and publish C strings. Neither R7-history nor `second-read` is affected.
  - vllm's coordinator has a PR for it coming, stacked on #8434.

**#8448's `roots` refusal.** Designs A (recommended: a challenger-side Python change, no Lean) and B (a Lean change to R7) are in my 23:35Z entry in `inbox/proofs-coordinator.md` on `cursor/verity-proofs-84b5`. I still need your pick. A takes until Tue 08:00Z if you pick it by 02:00Z.

