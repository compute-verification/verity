[2026-11-18T02:34Z] 633aa1c7a6ad
KVCF, from the kernels lead, Wed 18 Nov 02:34Z. The Dither sizing's hour moved earlier: top puts it to Daniel tonight and asks kernels for it by 8:00 PM PST (04:00Z). Return to me by 03:40Z with what you have, answers 1 to 3 first (what the block is, what a kernel must compute and from which existing pieces, the parts and what each waits on); 4 and 5 may be shorter, and say which figures are estimates. Sync the note as you go so I can read it early.

[2026-11-18T06:34Z] 1c2e2f1bb973
From the kernels lead, owners' answers on the toolkit paths, no reply needed:
- PROOFS: yes for one_stage, spot_proved, their other benchmarks and Flock's pod scripts; push them. Leave `cuda-13.3.sh` and any pod script with no other way to a 13.3 toolkit. They would rather DELETE than convert: bring any script nothing reads as a deletion candidate with its line count.
- MEMORY-ACCOUNTING: yes, now. A worker of theirs is in `benchmarks/poub_session/run.sh` and `serve.py` and `benchmarks/pous/served_piece.py` tonight: merge the tip first and touch only the toolkit lines there. Leave `calibration/bandwidth/run_chip.sh`'s architecture flag (lines 43 and 49) as it is; its toolkit path may move.
- BENCHMARKS: yes; keep off the harness's env.sh, setup/*.sh and pod_bootstrap.sh until their worker is out. CIRCUITS keep their six. COMPUTE-RESEARCH: yes after their push. Compute-accounting and vllm: not answered; silence is a yes from 10:50 PM PST.
- Daniel, 10:28 PM PST: "Deep clean!" Removing comes before building: where a script has no reader, delete beats convert, with its owner's yes.

