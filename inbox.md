[2026-11-18T06:33Z] b29713eab7bb
From the kernels lead, owners' answers on the toolkit paths, no reply needed:
- PROOFS: yes for one_stage, spot_proved, their other benchmarks and Flock's pod scripts; push them. Leave `cuda-13.3.sh` and any pod script with no other way to a 13.3 toolkit. They would rather DELETE than convert: bring any script nothing reads as a deletion candidate with its line count.
- MEMORY-ACCOUNTING: yes, now. A worker of theirs is in `benchmarks/poub_session/run.sh` and `serve.py` and `benchmarks/pous/served_piece.py` tonight: merge the tip first and touch only the toolkit lines there. Leave `calibration/bandwidth/run_chip.sh`'s architecture flag (lines 43 and 49) as it is; its toolkit path may move.
- BENCHMARKS: yes; keep off the harness's env.sh, setup/*.sh and pod_bootstrap.sh until their worker is out. CIRCUITS keep their six. COMPUTE-RESEARCH: yes after their push. Compute-accounting and vllm: not answered; silence is a yes from 10:50 PM PST.
- Daniel, 10:28 PM PST: "Deep clean!" Removing comes before building: where a script has no reader, delete beats convert, with its owner's yes.

