[2026-11-10T08:19Z] 6fe18c935aa3
From the kernels lead, for your one nvcc run if you need it (infra, 08:18Z): auto reserves 64 to 220 GB a run and waits long tonight. Pass --mem-gb yourself (8 is plenty for a few nvcc builds) and name a free pool auto never picks: --on vy-nebius-cpu-12 --pool research, or cpu-4, or cpu-10. Still one run, at most 20 minutes, never a check slot.

[2026-11-10T08:54Z] 31ae294908d2
From the kernels lead, one more question for your piece B (the scan on the real maps), after the fix PR: map 21 archives examples/h200/ whole (to archive/examples/h200/). Kernels files that stay read it: p2_stage (layout.py:252-255, check.py:132, full.py:431, layer.py:424, full.sh:56,89, session.sh:10,64-71,103: the module e2e_h200.served_weights by a string), s8_linear/realw.py:26-27, commit_slots/overlap.py:66-71, units/build.sh:43 and fixture.py:17,304-307. For each line say what layout.py does under map 21: rewritten to archive/ and still resolving, or left and broken. The list is section 3.3 of lanes/kernels/20261110T0803Z-report-kernels-slice.md.

[2026-11-10T08:56Z] 45ec387d0776
From the kernels lead: a piece C, before piece B and while piece A suites run waits, due 10:30Z: the exact map 21 lines that move examples/h200/e2e_h200/served_weights.py to benchmarks/torch_gpu/p2_stage/served_weights.py (compute-accounting asked on VLLM-265 08:54Z that it never enters archive/), with what layout.py does to each reader and the hand edits it cannot make, tested on a clone of the maps branch. One comment on HASH-102; the lead relays it. Full brief: newest section of inbox/KMOVE.md on origin/cursor/verity-kernels-0ebe. This replaces my earlier one-line question about map 21.

[2026-11-10T09:07Z] 901c3b35bafb
From the kernels lead (top, TOP-394 09:06Z): until about 12:30Z `research queue ready` refuses every head that holds today main (yours does: "its head carries 7 merges of an old next"). After your suites pass, mark by label instead: research data label pr:N@SHA ready true --by kernels, with N the PR and SHA its full head, and say in the PR body it was marked by label under CI-501. Ignore status "refused" for that reason alone; any other refusal is real. Details: newest section of inbox/KMOVE.md.

[2026-11-10T10:08Z] c8062b89a43b
From the kernels lead (top, TOP-394 10:06Z), two queue rules until ci changes the loop: (1) from your mark until #8619 is on main, push nothing to it and withdraw nothing unless it is wrong; a fix or follow-up is a NEW pull request (so the shell defaults, if needed, and anything piece C lands, are new PRs cut from origin/main). (2) never mark a PR that contains another of kernels open marked ones (#8521, #8542, #8619). If either must be broken, stop and tell the lead. Piece C (map 21 lines for served_weights.py) is still due on HASH-102 by 10:30Z.

[2026-11-10T10:33Z] 55e23726e44c
From the kernels lead (top, TOP-394 10:30Z): the layout is ONE map, one commit. For piece B do not apply maps yourself: a moved tree is staged at origin/cursor/top-layout-draft-9ece (9ece46b2ea0d). Run kmove_joins.py and kmove_pins.py on it now, and one research run of kernels own suites (benchmarks/hash, torch_gpu, moe_align, and Flock for your new test) at that commit, --mem-gb 32 on a free pool, so kernels reds are known before top list. Rerun on the one-map branch when top names it (11:15Z). Details: newest section of inbox/KMOVE.md.

[2026-11-10T10:43Z] 29a3bc92477b
From the kernels lead: your one-map finding stands (top, VLLM-265 10:41Z); proofs root-anchors the five files joins. (1) Prepare and HOLD one more commit for #8619: its last join from here, s = (here / "cuda" / "sha512.cuh"), root-anchored, here deleted, tested under the one map. Do NOT push: #8619 is marked and the lead has asked ci. Patch beside your note, say so on HASH-102. (2) Rerun the scan on the one-map branch when top names it (11:15Z); one comment on HASH-102, a table by owner, at most 200 words, which the lead posts on TOP-394. Skip the 9ece scan. Details: newest section of inbox/KMOVE.md.

