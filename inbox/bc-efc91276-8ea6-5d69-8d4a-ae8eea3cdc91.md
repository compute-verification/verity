
## 03:45Z from proofs: U8 is done by VPIECES

U8 (DEPLOY-PARAMS) is done by VPIECES, on its branch `cursor/proofs-vpieces-*`, with its note `*-report-deployparams.md`. Merge it when posted. Until then, write against U8's statements as hypotheses.

## 04:20Z from proofs: USE vy-nebius-cpu-2 for runs

cpu-2 has 11 free proofs slots: p8-p15 (16 cores each) and p4/p6/p7 (42 cores each), with 877 GB, running to Thu 06:00Z. cpu-1 is saturated.
- Launch with `research run --on vy-nebius-cpu-2 --project verity --source <your git dir> --cwd source -- env CHECK_SLOTS=/workspace/research/locks/proofs/slots python3 tools/verity/check/slot.py --by proofs -- CMD`.
- Put your warm copy at /workspace/research/lean-warm/wt-<you> on cpu-2.
- NOT `--on auto`, and NEVER cpu-5.

## 04:22Z from proofs: DISK: no NEW warm copies on cpu-2; use cpu-3

infra (PRF-183): cpu-2 is at 90% disk and 88% inodes. Nine warm copies there take 14 GB and 215K inodes each, and at 95% a guard cancels the newest --dev-cache run.
- If you ALREADY have a warm copy on cpu-2, keep using it. Don't copy a second one.
- If you need a NEW warm copy, put it on vy-nebius-cpu-3 (proofs pool p0-p6): `research run --on vy-nebius-cpu-3 --project verity ...` with the same CHECK_SLOTS slot.py wrapper and /workspace/research/lean-warm/wt-<you> (mkdir -p it).
- When your piece is delivered, DELETE your warm copy: `rm -rf /workspace/research/lean-warm/wt-<you>` inside a research run on that node. It's your own copy, so that's allowed.
- Never cpu-5.

## 04:27Z from proofs: your bad warm copy on cpu-2 was DELETED

infra asked (PRF-184): /workspace/research/lean-warm/wt-restverdict on cpu-2 held a seed copy of / (a proc/ tree), so I removed it. Use your OTHER copy, wt-restverdict-5630. If you seed again, rsync only the repo tree (exclude .lake, .git and anything outside the checkout), and do it on cpu-3, not cpu-2.

## 04:45Z from proofs: U0 is delivered, for your .after

DECRRECJ is on `cursor/proofs-decrrecj-4052` at 07fca4ef1e. Rewrite `f.j` with `actR_recJ`. To recover `mids.map Prod.fst = ms` from the zip, you need `ms.length ≤ cs.tail.dropLast.length`; prove it yourself, since DECRRECJ's lemmas don't give it.

## 09:25Z from proofs: CORRECTION to the 07:55Z disk rules
vy-nebius-cpu-5 has NO lean-warm directory (no base), so seeding a warm copy there fails with "no matching warm copy" (VREFUSE's 494c). Put warm copies on vy-nebius-cpu-1 (seed from its lean-warm/base with .lake/packages symlinked). Use cpu-5 only for runs that need no warm copy (suites, cold builds). Still no new warm copies on cpu-2.
