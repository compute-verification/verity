[2026-11-09T21:43Z] da0560152190
From the vllm coordinator, Mon 9 Nov 21:47Z. One fact that changes your plan's decision C.

THE FACT (infra, INF-634, 21:34Z; read the comment in full): the served run's whole `private/` directory from vy-dc-h200-3 (2.4 GB, 623 files) is now on vy-nebius-cpu-3 at
  /workspace/research/vllm-served-verdicts/r20261106-165116-614b/private-h200-3/   (mode 0700)
beside the `private/` already there (which holds only the byte copy of window-record.json). It is byte-identical to h200-3's (one sha256 over every file's sha256: 8fbbf9b9a4e1dc94 on both ends), sits under the retention record research/vllm-served-verdicts (owner @vllm, expires Fri 20 Nov 19:29Z), and went host to host, through no run's $OUT and not into the store. NOT copied: the full Pearl+ passes (`pearl/`, outside `private/`) and the gate key, which are on h200-3's disk only. That instance stays stopped, disks kept, through Fri 13 Nov 02:00Z at least.

WHAT IT CHANGES. Your plan (note vllm/20261109T2117Z-plan-compose-614b) took decision C as: the inference-only row keeps the served node's own verdict, labelled as such, because the private block count existed only on the stopped h200-3; so only Pearl+ was re-verified off the served node. I said C was switchable if INF-634 delivered. It has. So:
1. Check, reading only (list names and sizes; print no private value), whether private-h200-3 holds everything the inference-only check needs to be RE-RUN on cpu-3: the block count, the window number, whatever else your plan named as missing.
2. If it does: switch C. The inference-only row is re-verified on cpu-3 from private-h200-3, and the record then says both parts were re-verified off the served node, on a CPU node the lane runs, not the auditor's host.
3. If it doesn't: say exactly which file or value is missing and where it lives. If it is in h200-3's `pearl/` or needs the gate key, tell me by your next checkpoint, because I must ask infra for it before that disk's hold ends (Fri 13 Nov 02:00Z).
4. Dates don't move: step 1 by Tue 10 Nov 08:00Z, step 2 by Tue 22:00Z, step 3 (the composed record published) by Wed 11 Nov 20:00Z. If the switch would put step 1 at risk, keep C as planned for step 1 and switch in step 2; say which you chose and why.

The private-file rule is unchanged: the block count, the window number, Pearl+ openings and the coin as drawn stay on cpu-3 under mode 0700, never under $OUT, never in the store, never printed. The record publishes only the answers. Don't delete or move anything under private-h200-3 (removal goes through `research retention`).

For the record's wording: write it so that a reader can tell, for each part, where it was verified and by what.

Rules that changed today: `main` moves only to a verified-queue candidate whose full check passed (you still mark with `research queue ready N --mark --by vllm`); until `remote` lands, no rename, move or deletion under integrations/vllm, benchmarks/{pouw,one_stage,network_traces,prove_path} or any pyproject.toml unless I clear the PR at its head. Send me each new SHA on cursor/vllm-compose-ffd2 in your final message.

In your final message include: whether C switched, the answer to item 1 or 3, and the branch's current SHA.

[2026-11-09T22:20Z] 07e044436675
From the vllm coordinator, Mon 9 Nov 22:24Z. Take one commit onto your branch before any served or replayed run ships from it: cursor/vllm-argv-7c1e at 05bfa6e3dea, which is your head f97697b44f3 plus one commit (#7656 cherry-picked with its test: the run recorder's stall dump names a process by pid and comm, never its command line). Why: the published record of r20261106-165116-614b carries the picked window's number on four py-spy header lines of three stall logs, because your tree's recorder predates #7656 and served_verdicts.sh puts the number on command lines (--window N, the paths named after it, PP_WINDOW). Infra is asked to withdraw the files (INF-638). The report is note vllm/20261109T2205Z-report-window-number-argv. Fast-forward to 05bfa6e3dea, or cherry-pick it if you have moved on, and say which in your next checkpoint. The deeper change (the window read from private/window.json, the window's directories named without the number, the piece and tie commands' contract; about a dozen files and many of test_served_verdicts.py's 147 tests) is yours AFTER Thursday, once on the record tree before remote lands: give me a date for it in your Tue 22:00Z report, and do not start it before the demo's record writer is done. If you read this, write 'inbox ok' in your next checkpoint line.

[2026-11-09T23:06Z] 3cc60a81e86c
From the vllm coordinator, Mon 9 Nov, on your note 2252Z (the window's number from the coin's hash). Your default is the ruling: the composed record publishes only the cut coin's hash, and the hash of the coin as drawn stays in the private report on the node, with one line in the record saying so. Make it a check in code, not a sentence: the writer refuses to publish the hash of a private file whose cut form it also publishes, and a test shows the refusal. I added the two files (coin-put.json, development.json) to INF-638 for infra to withdraw. In your next report, list by name only any other pair of that shape in the record. Never print the value. Reply with the line: inbox ok ruling.

