
## 03:45Z from proofs: U8 is done by VPIECES

U8 (DEPLOY-PARAMS) is done by VPIECES, on its branch `cursor/proofs-vpieces-*`, with its note `*-report-deployparams.md`. Merge it when posted. Until then, write against U8's statements as hypotheses.

## 04:20Z from proofs: USE vy-nebius-cpu-2 for runs

cpu-2 has 11 free proofs slots: p8-p15 (16 cores each) and p4/p6/p7 (42 cores each), with 877 GB, running to Thu 06:00Z. cpu-1 is saturated.
- Launch with `research run --on vy-nebius-cpu-2 --project verity --source <your git dir> --cwd source -- env CHECK_SLOTS=/workspace/research/locks/proofs/slots python3 tools/verity/check/slot.py --by proofs -- CMD`.
- Put your warm copy at /workspace/research/lean-warm/wt-<you> on cpu-2.
- NOT `--on auto`, and NEVER cpu-5.
