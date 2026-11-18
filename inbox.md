[2026-11-18T06:13Z] 1cad3e7f372f
From the kernels lead, a change to your job 3 (moving the benchmark runners out of verity/), no reply needed:
- Do NOT move these eleven, they are being DELETED tonight at 10:35 PM PST (06:35Z) by another worker unless compute-accounting objects on TOP-441, since nothing reads them: edited_gemm/{cwrun,d2srun,egemm_dial_probe,egemm_fill_probe,run80}.sh, masked_gemm/{erun,lprun,pvrun,srun}.sh, served_zk/{bench,zk_window}.sh. If your move is already pushed, say so on return and leave them where they are; if not, leave these eleven out of the map.
- Job 2 matters more now: `kit.card_gencode` (kit.py ~280) has no caller on the branch. Wire `benchmarks/pouw/harness/native/build.sh` (~59), `benchmarks/kernels/ship.sh` (~57) and `ship_probe.sh` to it through a `gencode` verb of `machine.py`, tonight. Scripts under verity/ keep their flags until after the close.
- KAUDIT mended two of our tests on the kit's new refusal wording (223e8a4691a); merge origin/interfaces before you push.

