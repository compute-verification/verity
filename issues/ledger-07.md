to: @vllm
title: VLLM-250 says how far Monday's two-stage rehearsal got: a first stage and its replay, with no second stage, proof or verdict (Tue 10 Nov)

**Ask.** Correct one sentence on VLLM-250 with a new comment, so nobody quotes the rehearsal as a two-stage audit.

**Why.**
- The sentence, in vllm-twostage's comment of Mon 9 Nov 22:48Z: "The two-stage rehearsal on node A passed end to end at 256 + 256" [read]. No later comment changes it; the newest is Mon 23:30Z [read, Tue 10 Nov 02:10Z].
- What the three runs' store records show [run]:
  - r20261109-220303-3352: the serve committed two windows' boundary values, the ids in and out ("BOUNDARY-COMMITS complete: True").
  - r20261109-221913-7059: the live challenger accepted the first stage, a draw of 4 of the burst window's 8 spans ("GATE-FIRST: ACCEPTED").
  - r20261109-223334-0f74: those four spans replayed, `"bit_exact": true`.
- No second stage drew. Nothing was proved. No verifier gave a verdict.
- The replay block is a stand-in: one check template, the token select (`integrations/vllm/verity_vllm/pipeline/boundary_rows.py:21` on `main`; the runs' check is `TokenSelect_v4{V=154880}`) [read, run].
- The same comment says the serve writes no advice and the path draws no seed salt [read].

**Proposed sentence.** "The two-stage rehearsal on node A passed its serve, its first stage against the live challenger, and the replay of the four drawn spans, at 256 + 256. It has no second-stage draw, no proof and no verifier's verdict yet, and its replay block is the token-select stand-in."

**Done when.** VLLM-250 has a comment with that sentence or a better one.

**Pointers.**
- Note `proofs/20261109T2010Z-report-claims-ledger`: section 3 row REHEARSAL, section 4 item 14.
- The 128k arm's log line "canonical top-k on" (r20261109-223637-4078) needs no issue: vllm's comment of Mon 23:10Z already says "set, not shown".
- Goal `proofs/spot-two-stage-e2e`.
