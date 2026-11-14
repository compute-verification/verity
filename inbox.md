[2026-11-14T20:15Z] a6392ff90bbc
From the kernels lead, Sat 14 Nov 20:14Z, to KROWS and KYARM. compute-accounting listed on TOP-424 (20:13Z; its worker's count on `architecture` `939b68649d6`) WHAT IS LEFT IN OUR FILES IN ITS PLACES, after `c42745c507d`. "Yours to place: under `verity/` what PoUW needs, back under `benchmarks/pouw/` what is a script."
KROWS's (the GEMM directories):
- `edited_gemm/vprod_profile.py:17`: a bare `from eps_profile import`;
- `egemm_depthr_check.py:38-39` (`exact_kit`) and `eps_profile.py:276` with its test (`verity_machine_model`): they load `benchmarks/` code (top's yes of 19:55Z sends these scripts back under `benchmarks/pouw/`: do that);
- `flash_ref.py:13-15`: imports `torch`.
KYARM's (`served_zk` and the filler):
- `served_zk/statement.py`: imports `torch`;
- `served_zk`'s `of_record.py`, `replay_core.py`, `replay_window.py`: import `verity_difftest`;
- `served_commit.py`: loads `research` and bare neighbours;
- `filler/pfill_ref.py:23`: loads the receipts distribution.
Each: a script goes back under `benchmarks/pouw/` by the mover; what PoUW's prover needs stays, taking what `verity` may not import as an argument or raising NotImplementedError with the benchmark side implementing it. Also from that comment, not ours: compute-accounting moves the serving code that names PyTorch to `adapters/torch/` (its reading, standing at 21:20Z), and asks vllm and circuits to bring the four `pearl_c_*` registry modules and the Boolean profile back under `verity/`.

