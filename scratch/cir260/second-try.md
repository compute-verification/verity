# CIR-260: the capture's second try, at PR #9316's head ffaacd91c9dd0d6baf4494143fe5ec4ad2e40c44

Row: node A (`vy-spot-glm53`), Fri 13 Nov 03:15Z–04:25Z, card 2, cores 20-29 (provisional; cards 0-3 and cores 0-39 are vllm's 03:00Z–03:15Z for VLLM-339's serve, so no lease on card 2 before 03:15Z). Worker FP8WRITE, GPU_LEASE_WHO=circuits-fp8write.

1. `date -u` in its own call. `uv run -q --project tools/research research notes sync`, then read the circuits row of Fri 03:15Z
   in `~/.research/notes/calendars/node-a.md`. If card 2 is no longer circuits' in it, launch nothing and tell the coordinator;
   if its hours or cores changed, follow the row.
2. Check that PR #9316's head is still ffaacd91c9dd (`gh pr view 9316 --json headRefOid`) and check that commit out in a clean
   checkout: `git -C /agent/repos/verity fetch origin cursor/circuits-cir260-fp8-cache-write-6f04` then
   `git -C /agent/repos/verity checkout --detach ffaacd91c9dd0d6baf4494143fe5ec4ad2e40c44` (no `cd` in a git checkout command).
3. Launch once, from that checkout, with the launcher on this branch (`scratch/cir260/ds_mla_capture.sh`; it passes `--cores` to
   gpu-lease, pins both phases with taskset, exits 75 before NOT_BEFORE or with < 30 min left, and 76 if the card or its lease is busy):

       cp <scratch worktree>/scratch/cir260/ds_mla_capture.sh /home/ubuntu/ds_mla_capture.sh
       uv run -q --project tools/research research run --on vy-spot-glm53 --project verity --source /agent/repos/verity \
         --cwd source --timeout 50m --mem-gb 32 --send /home/ubuntu/ds_mla_capture.sh \
         --env CARD=2 --env CORES=20-29 --env NOT_BEFORE=2026-11-13T03:15:00Z --env END_UTC=2026-11-13T04:25:00Z \
         -- bash -c 'bash $RESEARCH_RUN_DIR/inputs/ds_mla_capture.sh'

   If the launcher exits 76 (card 2 or its lease busy), vllm's serve overran: wait (the tool that waits, never a shell sleep)
   and launch again; the last start is 03:50Z.

4. `research fetch RUN --all`; read `stdout.log` (the GO/NO-GO line, the lease's hold in stderr.log) and
   `out/glm53_ds_mla_write_capture.json` (`ok`, `summary.<case>.{tokens,slotted,differ,corners}`). Cache rows compared =
   slotted x 2 cases x 2 modes.
5. Comment on CIR-260 only: `research msg linear comment CIR-260 --file F.md --worker FP8WRITE`, leading in bold with the run id,
   rows compared and rows that differ; the first try was r20261112-172953-93dc at f912fe226f80 (12,512 rows, 0 differ). If the
   PR is marked by then, push nothing to it; the rerun is a comment only.
