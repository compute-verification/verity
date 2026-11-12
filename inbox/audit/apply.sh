#!/bin/bash
# Proposed changes from the audit of proofs' open issues (`research msg linear audit --lane proofs`, run Thu 12 Nov 13:21Z).
# Written by PROGHELD; nothing in this file has been run, and I wrote nothing to Linear. The reasons are in table.md.
# Run from the root of a checkout of cursor/verity-proofs-84b5, where inbox/audit/ holds the comment files.
# One command a line. Groups 1 to 6 are the ones I am sure of, in the order to run them; group 7 is commented out:
# each of its lines waits for your decision, which table.md states beside what I found.
# A comment is posted before the state it explains, so run an issue's lines together and in order.

# ---- 1. Landed, and issues whose done-condition holds
research msg linear comment PRF-547 --file inbox/audit/PRF-547.md --as proofs --registry /tmp/registry-main.json
research msg linear state PRF-547 "Done" --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-550 --file inbox/audit/PRF-550.md --as proofs --registry /tmp/registry-main.json
research msg linear state PRF-550 "Done" --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-553 --file inbox/audit/PRF-553.md --as proofs --registry /tmp/registry-main.json
research msg linear state PRF-553 "Done" --as proofs --registry /tmp/registry-main.json
research msg linear edit PRF-555 --title "kept_composite.sh: the sampled proofs v2 step writes its case on main again (v2_case.py passes public)" --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-555 --file inbox/audit/PRF-555.md --as proofs --registry /tmp/registry-main.json
research msg linear state PRF-555 "Done" --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-534 --file inbox/audit/PRF-534.md --as proofs --registry /tmp/registry-main.json
research msg linear state PRF-534 "Done" --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-533 --file inbox/audit/PRF-533.md --as proofs --registry /tmp/registry-main.json
research msg linear state PRF-533 "Done" --as proofs --registry /tmp/registry-main.json
research msg linear comment MEM-279 --file inbox/audit/MEM-279.md --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-573 --file inbox/audit/PRF-573.md --as proofs --registry /tmp/registry-main.json
research msg linear state PRF-573 "Done" --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-409 --file inbox/audit/PRF-409.md --as proofs --registry /tmp/registry-main.json
research msg linear state PRF-409 "Done" --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-506 --file inbox/audit/PRF-506.md --as proofs --registry /tmp/registry-main.json
research msg linear state PRF-506 "Done" --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-507 --file inbox/audit/PRF-507.md --as proofs --registry /tmp/registry-main.json
research msg linear state PRF-507 "Done" --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-503 --file inbox/audit/PRF-503.md --as proofs --registry /tmp/registry-main.json
research msg linear state PRF-503 "Done" --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-484 --file inbox/audit/PRF-484.md --as proofs --registry /tmp/registry-main.json
research msg linear state PRF-484 "Done" --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-499 --file inbox/audit/PRF-499.md --as proofs --registry /tmp/registry-main.json
research msg linear state PRF-499 "Done" --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-472 --file inbox/audit/PRF-472.md --as proofs --registry /tmp/registry-main.json
research msg linear state PRF-472 "Done" --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-397 --file inbox/audit/PRF-397.md --as proofs --registry /tmp/registry-main.json
research msg linear state PRF-397 "Done" --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-492 --file inbox/audit/PRF-492.md --as proofs --registry /tmp/registry-main.json
research msg linear edit PRF-492 --due 2026-11-13 --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-464 --file inbox/audit/PRF-464.md --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-413 --file inbox/audit/PRF-413.md --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-310 --file inbox/audit/PRF-310.md --as proofs --registry /tmp/registry-main.json

# ---- 2. Dates
research msg linear comment PRF-419 --file inbox/audit/PRF-419.md --as proofs --registry /tmp/registry-main.json
research msg linear edit PRF-419 --due 2026-11-30 --as proofs --registry /tmp/registry-main.json
research msg linear place PRF-419 "Full proofs: protocol soundness" --milestone F-NEXT --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-528 --file inbox/audit/PRF-528.md --as proofs --registry /tmp/registry-main.json
research msg linear edit PRF-528 --due 2026-11-13 --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-561 --file inbox/audit/PRF-561.md --as proofs --registry /tmp/registry-main.json
research msg linear edit PRF-561 --due 2026-11-17 --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-577 --file inbox/audit/PRF-577.md --as proofs --registry /tmp/registry-main.json
research msg linear edit PRF-577 --due 2026-11-18 --as proofs --registry /tmp/registry-main.json

# ---- 3. Quiet lines
research msg linear comment PRF-164 --file inbox/audit/PRF-164.md --as proofs --registry /tmp/registry-main.json
research msg linear state PRF-164 "Todo" --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-199 --file inbox/audit/PRF-199.md --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-201 --file inbox/audit/PRF-201.md --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-335 --file inbox/audit/PRF-335.md --as proofs --registry /tmp/registry-main.json
research msg linear state PRF-335 "Todo" --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-361 --file inbox/audit/PRF-361.md --as proofs --registry /tmp/registry-main.json
research msg linear state PRF-361 "Todo" --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-384 --file inbox/audit/PRF-384.md --as proofs --registry /tmp/registry-main.json
research msg linear state PRF-384 "Todo" --as proofs --registry /tmp/registry-main.json
research msg linear place PRF-384 "Onsite demo: the remote run with network accounting" --milestone O6 --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-454 --file inbox/audit/PRF-454.md --as proofs --registry /tmp/registry-main.json
research msg linear state PRF-454 "Todo" --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-458 --file inbox/audit/PRF-458.md --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-460 --file inbox/audit/PRF-460.md --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-466 --file inbox/audit/PRF-466.md --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-468 --file inbox/audit/PRF-468.md --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-475 --file inbox/audit/PRF-475.md --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-505 --file inbox/audit/PRF-505.md --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-508 --file inbox/audit/PRF-508.md --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-514 --file inbox/audit/PRF-514.md --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-523 --file inbox/audit/PRF-523.md --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-524 --file inbox/audit/PRF-524.md --as proofs --registry /tmp/registry-main.json

# ---- 4. Strays
research msg linear comment PRF-463 --file inbox/audit/PRF-463.md --as proofs --registry /tmp/registry-main.json
research msg linear state PRF-463 "Done" --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-485 --file inbox/audit/PRF-485.md --as proofs --registry /tmp/registry-main.json
research msg linear state PRF-485 "Done" --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-489 --file inbox/audit/PRF-489.md --as proofs --registry /tmp/registry-main.json
research msg linear state PRF-489 "Done" --as proofs --registry /tmp/registry-main.json
research msg linear comment PRF-498 --file inbox/audit/PRF-498.md --as proofs --registry /tmp/registry-main.json
research msg linear place PRF-498 "Repository cleanup: organization, size and code quality" --milestone C5 --as proofs --registry /tmp/registry-main.json
research msg linear edit PRF-498 --due 2026-11-13 --as proofs --registry /tmp/registry-main.json
research msg linear state PRF-498 "Todo" --as proofs --registry /tmp/registry-main.json

# ---- 5. Goal labels
research msg linear edit PRF-595 --label proofs/zk-leaks-closed --as proofs --registry /tmp/registry-main.json
research msg linear edit PRF-577 --label proofs/spot-bind-term --as proofs --registry /tmp/registry-main.json

# ---- 6. Lapsed milestones
# soundness leads this project. proofs asked it on Tue 10 Nov 02:22Z (PRF-199) to re-date F2 and has no answer.
# The date is the latest due date among F2's issues in work (PRF-199's), by proofs' rule of that day.
# This one line clears all 14 `lapsed` findings on F2; leave it out if you wait for soundness.
research msg linear milestone edit "Full proofs: protocol soundness" "F2 · Tue 3 Nov 18:00Z · every subprotocol proved, marked" --date 2026-11-30 --rename "F2 · Mon 30 Nov · every subprotocol proved, marked" --as proofs --registry /tmp/registry-main.json

# ---- 7. Coordinator decides: commented out. Uncomment a line once you have decided; YYYY-MM-DD is yours to fill.
#      Where two lines for one issue are alternatives, table.md says so; a comment file here is a draft to edit.
#  -- dates and closings
# PRF-246: A date (F-NEXT is Tue 1 Dec) and a holder, or Canceled: "asked soundness on 10 Nov whether the goal still needs it; no answer".
# research msg linear edit PRF-246 --due 2026-12-01 --as proofs --registry /tmp/registry-main.json
# research msg linear state PRF-246 "Canceled" --as proofs --registry /tmp/registry-main.json
# PRF-269: A date, a milestone and a live goal, or Canceled with a one-line reason, or Backlog.
# research msg linear edit PRF-269 --due YYYY-MM-DD --as proofs --registry /tmp/registry-main.json
# research msg linear state PRF-269 "Backlog" --as proofs --registry /tmp/registry-main.json
# PRF-282: The date (its goal's is Fri 20 Nov) and the worker for item 3; the `slip:` comment is drafted without a date and needs it added. PRF-201 and PRF-310 take the same date.
# research msg linear comment PRF-282 --file inbox/audit/PRF-282.md --as proofs --registry /tmp/registry-main.json
# research msg linear edit PRF-282 --due 2026-11-20 --as proofs --registry /tmp/registry-main.json
# research msg linear edit PRF-201 --due 2026-11-20 --as proofs --registry /tmp/registry-main.json
# research msg linear edit PRF-310 --due 2026-11-20 --as proofs --registry /tmp/registry-main.json
# PRF-332: Its date follows PRF-404's pair 11: 17 Nov (its goal's and milestone's) if that holds, else Backlog.
# research msg linear edit PRF-332 --due 2026-11-17 --as proofs --registry /tmp/registry-main.json
# research msg linear state PRF-332 "Backlog" --as proofs --registry /tmp/registry-main.json
# PRF-351: A project and a date after Mon 16 Nov (F-NEXT is Tue 1 Dec), or Canceled with the exemptions left standing.
# research msg linear place PRF-351 "Full proofs: protocol soundness" --milestone F-NEXT --as proofs --registry /tmp/registry-main.json
# research msg linear edit PRF-351 --due 2026-12-01 --as proofs --registry /tmp/registry-main.json
# research msg linear state PRF-351 "Canceled" --as proofs --registry /tmp/registry-main.json
# PRF-356: Canceled ("no goal on the page needs it; asked on 10 Nov, no answer"), or a goal, a project and a date.
# research msg linear state PRF-356 "Canceled" --as proofs --registry /tmp/registry-main.json
# PRF-361: The date: 12 Nov is today and is the goal's; no later date is written anywhere I read. The goal's date moves first.
# research msg linear edit PRF-361 --due YYYY-MM-DD --as proofs --registry /tmp/registry-main.json
# PRF-367: A worker and a date (no later than 20 Nov, its goal's), or Backlog, or Canceled with a reason.
# research msg linear edit PRF-367 --due YYYY-MM-DD --as proofs --registry /tmp/registry-main.json
# research msg linear state PRF-367 "Backlog" --as proofs --registry /tmp/registry-main.json
# PRF-386: O6 beside its parent clears `lapsed`; `stray` needs a date too: O6's (24 Nov), or wait for network-accounting's date for NET-208 on Fri 13 Nov.
# research msg linear place PRF-386 "Onsite demo: the remote run with network accounting" --milestone O6 --as proofs --registry /tmp/registry-main.json
# research msg linear edit PRF-386 --due 2026-11-24 --as proofs --registry /tmp/registry-main.json
# PRF-404: Ask soundness for pair 11's date and move the issue to O6 with it, or back to Todo. No date is written anywhere I read.
# research msg linear comment PRF-404 --file inbox/audit/PRF-404.md --as proofs --registry /tmp/registry-main.json
# research msg linear place PRF-404 "Onsite demo: the remote run with network accounting" --milestone O6 --as proofs --registry /tmp/registry-main.json
# research msg linear edit PRF-404 --due YYYY-MM-DD --as proofs --registry /tmp/registry-main.json
# research msg linear state PRF-404 "Todo" --as proofs --registry /tmp/registry-main.json
# PRF-410: Whether FLASHPROVE is resumed, and the date its plan and its outcome land; then one line here. If it is parked, that is Daniel's to say.
# research msg linear edit PRF-410 --due YYYY-MM-DD --as proofs --registry /tmp/registry-main.json
# PRF-458: The date: the issue and its goal say 12 Nov (today), the lock's entry for the premise left says 30 Nov. One of the two moves.
# research msg linear edit PRF-458 --due 2026-11-30 --as proofs --registry /tmp/registry-main.json
# PRF-459: Done on the goals' bar, or open (with a date and an owner, kernels') for the title's 4 to 8×. Its date, 12 Nov, is today.
# research msg linear comment PRF-459 --file inbox/audit/PRF-459.md --as proofs --registry /tmp/registry-main.json
# research msg linear state PRF-459 "Done" --as proofs --registry /tmp/registry-main.json
# PRF-461: Items 5, 6 and 7 (a result or a stated scope each), and the date: its goal's is Fri 13 Nov.
# research msg linear comment PRF-461 --file inbox/audit/PRF-461.md --as proofs --registry /tmp/registry-main.json
# research msg linear edit PRF-461 --due 2026-11-13 --as proofs --registry /tmp/registry-main.json
# PRF-462: 13 Nov (the goal's and the milestone's) or 17 Nov (the lock entry's, with the issue then past its goal's date), and a worker for item 5.
# research msg linear comment PRF-462 --file inbox/audit/PRF-462.md --as proofs --registry /tmp/registry-main.json
# research msg linear edit PRF-462 --due 2026-11-13 --as proofs --registry /tmp/registry-main.json
# PRF-468: A worker and a date for the two pieces left, or back to Todo. The goal it serves is due Fri 13 Nov.
# research msg linear edit PRF-468 --due 2026-11-13 --as proofs --registry /tmp/registry-main.json
# research msg linear state PRF-468 "Todo" --as proofs --registry /tmp/registry-main.json
# PRF-502: Done (rows past 16, and row 15's pin, stay with HASH-98, In Progress, due Fri 13 Nov), or In Progress with the rows left named and a date.
# research msg linear comment PRF-502 --file inbox/audit/PRF-502.md --as proofs --registry /tmp/registry-main.json
# research msg linear state PRF-502 "Done" --as proofs --registry /tmp/registry-main.json
# PRF-505: The date for crossing or closing the 13 heads. Its milestone C5 is dated Fri 13 Nov; no worker is named since CROSS (11 Nov).
# research msg linear edit PRF-505 --due 2026-11-13 --as proofs --registry /tmp/registry-main.json
# PRF-508: The date, or a close line for each of the four (#8197, #8438, #8462, #7850). Its milestone C5 is dated Fri 13 Nov.
# research msg linear edit PRF-508 --due 2026-11-13 --as proofs --registry /tmp/registry-main.json
# PRF-513: Done, and #8767: crossed and marked under another issue, or closed.
# research msg linear comment PRF-513 --file inbox/audit/PRF-513.md --as proofs --registry /tmp/registry-main.json
# research msg linear state PRF-513 "Done" --as proofs --registry /tmp/registry-main.json
# PRF-514: The date, and whether the three remaining upward imports are this issue's or a new one's. Also a project (it has none) and a live goal label.
# research msg linear edit PRF-514 --due YYYY-MM-DD --as proofs --registry /tmp/registry-main.json
# PRF-523: The date (the latest of its open children's), and closing #8900.
# research msg linear edit PRF-523 --due YYYY-MM-DD --as proofs --registry /tmp/registry-main.json
# PRF-524: The date. The goal its two open pull requests name, `proofs/spot-bind-term`, is due Wed 18 Nov; the issue carries no goal label.
# research msg linear edit PRF-524 --due 2026-11-18 --as proofs --registry /tmp/registry-main.json
# research msg linear edit PRF-524 --label proofs/spot-bind-term --as proofs --registry /tmp/registry-main.json
# PRF-539: Done (the brief's final is in) with the command filed as its own issue, or In Progress with a worker and a date for the command.
# research msg linear comment PRF-539 --file inbox/audit/PRF-539.md --as proofs --registry /tmp/registry-main.json
# research msg linear state PRF-539 "Done" --as proofs --registry /tmp/registry-main.json
# PRF-540: Done with the nine rows carried by the goal (or a new issue), or In Progress until the page is changed.
# research msg linear comment PRF-540 --file inbox/audit/PRF-540.md --as proofs --registry /tmp/registry-main.json
# research msg linear state PRF-540 "Done" --as proofs --registry /tmp/registry-main.json

#  -- F2's and F3's Backlog issues to F-NEXT (proofs' rule of Tue 10 Nov; soundness leads the project)
# PRF-163: F-NEXT or stay in F2 (with the others of F2's Backlog, below).
# research msg linear place PRF-163 "Full proofs: protocol soundness" --milestone F-NEXT --as proofs --registry /tmp/registry-main.json
# PRF-210: Move it to F-NEXT without soundness's answer (asked Tue 10 Nov 02:22Z), or ask again.
# research msg linear place PRF-210 "Full proofs: protocol soundness" --milestone F-NEXT --as proofs --registry /tmp/registry-main.json
# PRF-224: F-NEXT, or O6 with the onsite demo's other issues.
# research msg linear place PRF-224 "Full proofs: protocol soundness" --milestone F-NEXT --as proofs --registry /tmp/registry-main.json
# PRF-328: F-NEXT or stay.
# research msg linear place PRF-328 "Full proofs: protocol soundness" --milestone F-NEXT --as proofs --registry /tmp/registry-main.json
# PRF-329: F-NEXT, or close if the goal's being done means the listing is no longer wanted.
# research msg linear place PRF-329 "Full proofs: protocol soundness" --milestone F-NEXT --as proofs --registry /tmp/registry-main.json
# PRF-331: F-NEXT or stay.
# research msg linear place PRF-331 "Full proofs: protocol soundness" --milestone F-NEXT --as proofs --registry /tmp/registry-main.json
# PRF-343: F-NEXT or stay.
# research msg linear place PRF-343 "Full proofs: protocol soundness" --milestone F-NEXT --as proofs --registry /tmp/registry-main.json
# PRF-390: F-NEXT, or F-DONE beside its parent PRF-193.
# research msg linear place PRF-390 "Full proofs: protocol soundness" --milestone F-NEXT --as proofs --registry /tmp/registry-main.json
# PRF-391: F-NEXT, or F-DONE beside its parent PRF-193.
# research msg linear place PRF-391 "Full proofs: protocol soundness" --milestone F-NEXT --as proofs --registry /tmp/registry-main.json
# PRF-392: F-NEXT, or F-DONE beside its parent PRF-193.
# research msg linear place PRF-392 "Full proofs: protocol soundness" --milestone F-NEXT --as proofs --registry /tmp/registry-main.json
# PRF-395: F-NEXT, or F-DONE beside its parent PRF-193.
# research msg linear place PRF-395 "Full proofs: protocol soundness" --milestone F-NEXT --as proofs --registry /tmp/registry-main.json
# PRF-428: F-NEXT, or F-DONE beside its parent PRF-193.
# research msg linear place PRF-428 "Full proofs: protocol soundness" --milestone F-NEXT --as proofs --registry /tmp/registry-main.json
