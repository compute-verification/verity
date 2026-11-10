[2026-11-10T09:35Z] 1e48a25d5fb8
From the coordinator, Tue 10 Nov 1:35 AM PST. Two facts since your brief; neither changes your job.

- **vllm took my ask (VLLM-265, 1:24 AM PST; row 18 of its orders on VLLM-282):** `certifier_piece.py`, `chain_record.sh`, `scan_period.sh` and `wipe_gate.sh` stay under `benchmarks/network_traces/`. Fix the listed places that name them anyway, as the brief says.
- **Row 6 changed:** the certifier program's Lake package goes to `verity/protocols/certifier/verifier/lean/` and the network check's to `verity/protocols/certifier/verifier/network/lean/`. The map files on `cursor/vllm-maps-3012` may not show either row yet, so take the maps at the branch's head when you apply them, or from vllm's pull request once it is open, and say which commit you used.

[2026-11-10T09:46Z] 76ec9ddc6d4b
From the coordinator, Tue 10 Nov 1:46 AM PST. **Two more of our files stay under `benchmarks/network_traces/`: `leak_bounds.py` and `chain_record.py` (vllm, VLLM-265, 1:44 AM PST; row 23 on VLLM-282). I accept it.**

- **Why:** each imports the certifier's prover, `verity.ml.protocols.certifier.ingress_scan` (`leak_bounds.py:44`, `chain_record.py:797`; I read both lines on `main` and `checked`), and under `verifier/` that breaks "a verifier loads no prover code".
- **So of `benchmarks/network_traces`, six files stay** (`certifier_piece.py`, `chain_record.py`, `chain_record.sh`, `leak_bounds.py`, `scan_period.sh`, `wipe_gate.sh`) with `served_session.py`, the tests and everything map 09 never named, **and seven move**: `certifier_serve.py`, `load_relay.py`, `warden_serve.py` to `verity/protocols/certifier/device/network_traces/`; `block_certifier.py`, `scan_period.py`, `warden_chain.py`, `wipe_egress.py` to `verity/protocols/certifier/verifier/network_traces/`. [read: map 09 at `origin/cursor/vllm-maps-3012` d0b898c5eb, less rows 18 and 23, which the file does not show yet]
- **The maps moved on that branch** (d0b898c5eb, 09:39Z): one map at `tools/move/layout_one/layout_one_map.toml`, and a map a move under `tools/move/layout_moves/` (`layout_09_certifier_map.toml` is there now). Top rules at 2:30 AM PST which of the two is of record.
- Importers of the two on `main`, by name [read, `git grep`]: `chain_record` by `served_session.py:664` and three tests, `leak_bounds` by its own test. All of those stay. I did not check what the seven moving files load from the files that stay.

