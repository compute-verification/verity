[2026-11-21T13:05Z] f52dc80e89ac
From proofs lead, 5:06 AM PST Sat: one-circuit now HOLDS main: ci fast-forwarded it at 5:01 AM PST to 02bdc610c98 (the merge of main 9e521ae2dd5, lean's lock for the merged tree, and the tip 986ba525fc3). Before you report a head as ready: git fetch origin one-circuit, MERGE origin/one-circuit into your branch (no rebase, no force), rerun what the merge touched, and give merge-tree against the NEW tip. Carry on.

