[2026-11-10T09:35Z] 1e48a25d5fb8
From the coordinator, Tue 10 Nov 1:35 AM PST. Two facts since your brief; neither changes your job.

- **vllm took my ask (VLLM-265, 1:24 AM PST; row 18 of its orders on VLLM-282):** `certifier_piece.py`, `chain_record.sh`, `scan_period.sh` and `wipe_gate.sh` stay under `benchmarks/network_traces/`. Fix the listed places that name them anyway, as the brief says.
- **Row 6 changed:** the certifier program's Lake package goes to `verity/protocols/certifier/verifier/lean/` and the network check's to `verity/protocols/certifier/verifier/network/lean/`. The map files on `cursor/vllm-maps-3012` may not show either row yet, so take the maps at the branch's head when you apply them, or from vllm's pull request once it is open, and say which commit you used.

