# Fri 13 Nov 03:15Z: the one lease of the first row (not a capture of record)

Read from `~/.research/runs/r20261113-031522-215e/` (stdout.log and out/*.log) on the machine that launched it.

- Run `r20261113-031522-215e` on `vy-spot-glm53`, `jobs/card_row.sh eh_norm eh_proj indexer_logits`, PR head `920597d5ca7f`.
  Launched 03:15:22Z after clock reads of 03:14:52Z and 03:15:16Z. Start check 03:15:52Z: card 0, 0 compute processes, 0 MiB.
  One lease on card 0 with `--cores 0-9`, 03:15:53Z to 03:17:57Z (124 s; asked for 27 minutes). Exit 1. No minutes lost to
  an overrun of the serve before it. Device: NVIDIA H200, driver 595.91.07, compute capability 9.0, 132 SMs.
- `eh_norm` (03:15:53Z to 03:16:23Z, exit 1): device gate ok; both launches deterministic, shape [1024, 12288] bf16; one PTX
  specialisation, `code_sha256 88978f9f23bf0bf743216e79cbc564e294a845fda819eeccf30c8ca355408b7d`, equal to the pinned one, PTX
  counts equal to the pinned ones, `.reqntid 128`; SASS `ffma_rcp_eps 2`, `MUFU.RSQ 2`, `MUFU.RCP 0`. Bindings true for
  `MtpEhNorm_v1{H=6144,BLOCK=8192,WARPS=4,EPS=1e-05}`, `TritonBlockRMSNorm_v1{N=6144,...}`, `MtpEhZero_v1{H=6144}`,
  `TritonBlockRMSNormSumSq_v1{N=6144,...}`, `DsaRMSNormApply_v1{G=64}`. Words: served/e, served/p, probe/e, probe/p, each
  1,024 rows and 6,291,456 words compared, 0 differ (25,165,824 in all); reference rows 23, 23, 28, 28, 0 differ from the
  twin. Then `seal` raised (`schema ... is not a registered property's`): no record, and the alternatives' refutation counts
  were not printed.
- `eh_proj` (03:16:23Z to 03:17:38Z, exit 1): scan ok. Routes at K 12288, N 6144 over M 1 to 8192, 13 runs:
  `[[1,104,null],[105,111,2],[112,128,null],[129,159,2],[160,160,null],[161,223,2],[224,271,null],[272,272,2],[273,287,null],[288,288,2],[289,335,null],[336,336,2],[337,8192,null]]`.
  14 launches at M 1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 96: all unsplit, no reduce kernel, `nvjet_tst_*` kernels,
  deterministic. Then, at the first split launch (M 105), the sibling capture's split-K probe raised: at K 12288, 2 splits,
  bf16 output its random rows do not tell the strided partition from the contiguous one. No comparison ran in the job.
  Kept: `out/eh_proj/scan/*/scan.json`, `words/w.npy`, `words/M*/{x,out,rows}.npy`.
- `indexer_logits` (03:17:38Z to 03:17:57Z, exit 1): device gate ok; selection `native_next_n_6 false`,
  `supports_native_decode_6 false`, `supports_varlen false`; launches `edges` (9 requests, 54 rows, 100,664 entries) and
  `batch` (64 requests, 384 rows, 190,269 entries), every launch check true (flatten_lens, flatten_bt, decode_lens_one, shape,
  deterministic); binding `DsaMqaLogit_v3{H=32,D=128}` true. Words: 100,664 and 190,269 compared (290,933), 0 differ, 0
  unresolved; reference 50 and 94, 0 differ from the twin. Then the same `seal` failure: no record.

After it, on a CPU pod and not in the job that ran the kernel: `r20261113-033120-282e` (`jobs/ehproj_words.sh`, at PR head
`bcc3d97e7d`): tonight's kept eh_proj words of the 14 unsplit launches against
`Gemm_v2{K=12288,N=6144,DOT=HopperBF16WgmmaDot16_v1}` (`bf16_gemm_check.linear_rows`): 321 rows, 1,972,224 words compared, 0
differ; 97 reference coordinates, 0 differ from the twin; 18 s on 8 workers. No word of a split-K 2 launch exists yet.

Why no second lease although the fixes were pushed by about 03:30Z and the row ran to 03:45Z: the row said one lease, at the
PR head `920597d5ca7f`.
