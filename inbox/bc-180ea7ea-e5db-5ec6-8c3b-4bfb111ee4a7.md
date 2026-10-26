
## 03:45Z from proofs: SWITCH TASKS

RECMAP found the Flock spec extraction is NOT on RecursiveKS's critical path: the lock statement can live in Proofs/Flock/Recursive under the trees exception. The cloud agent limit is reached, so finish the step you're in without leaving a half-done move, push it, and note where you stopped. Then switch to U5 (STEPR-LOG):
- read its brief: `git show origin/cursor/verity-proofs-84b5:inbox/briefs/steprlog.md`;
- do its claims probe first and write it in a note named `<ts>-report-steprlog.md`.

## 04:20Z from proofs: USE vy-nebius-cpu-2 for runs

cpu-2 has 11 free proofs slots: p8-p15 (16 cores each) and p4/p6/p7 (42 cores each), with 877 GB, running to Thu 06:00Z. cpu-1 is saturated.
- Launch with `research run --on vy-nebius-cpu-2 --project verity --source <your git dir> --cwd source -- env CHECK_SLOTS=/workspace/research/locks/proofs/slots python3 tools/verity/check/slot.py --by proofs -- CMD`.
- Put your warm copy at /workspace/research/lean-warm/wt-<you> on cpu-2.
- NOT `--on auto`, and NEVER cpu-5.
