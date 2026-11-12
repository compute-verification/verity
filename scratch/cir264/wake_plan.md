# CIR-264: the capture on a card, from origin alone

Worker ACCEPT of circuits. The card hours: **Fri 13 Nov 03:45:00Z to 04:25:00Z, node A (`vy-spot-glm53`), card 0,
cores 0-9.** Before 03:45Z card 0 is CIR-263's; by 04:25Z everything of ours is off every card (proofs' GPU window is
04:30Z). A fallback row is Fri 13 Nov 17:45Z to 19:45Z, card 0, cores 0-9.

A resumed worker starts on a new machine in a new clone on `main`: `/tmp`, untracked files, local branches and fetched runs
are gone. Everything below starts from origin. Read `date -u` in its own call before each step that depends on the time.

## What origin holds

- `cursor/circuits-cir264-sampled-accept-6f04` at `903c545eb3d34a50eccbf66352a19a805efa0aeb`: the Definition
  (`verity/definitions/registry/sampled_accept.py`), its tests, and the capture
  `adapters/vllm/tests/properties/sampled_accept_capture_gpu.py`. The capture runs from THIS commit. Nothing is pushed to
  this branch before the capture has been read.
- `cursor/circuits-cir264-scratch-6f04`, `scratch/cir264/`: `accept_capture.sh` (the card job), `accept_replay.sh` (the
  off-card replay), `rehearse_all.sh` with `capture_dry.sh`, `capture_dry.py`, `ptx_check.py`, `ptx_sim.py`,
  `accept_ptx.tgz`, `accept_capture_dry.sh` and `capture_rehearsed.tgz` (the off-card rehearsals), `ckpt.sh` (one
  checkpoint line, and files, onto the scratch branch), `comment1.md` (the comment posted on CIR-264), `accept_ptx.py`,
  `accept_ptx.sh`, `philox_scan.py` (how the PTX was kept and read), `checkpoint.md`, and this file.

## 0. The time

`date -u`. If it is already past 04:14:00Z on Fri 13 Nov, launch nothing on a card: return to the lead and ask for the
Fri 17:45Z row. If it is before 03:45Z, do steps 1 and 2 and then wait with the waiting tool (slices of at most ten
minutes, `date -u` after each: the tool sometimes returns early).

## 1. Set up (about two minutes)

    R=/agent/repos/verity
    git -C $R fetch origin cursor/circuits-cir264-sampled-accept-6f04 cursor/circuits-cir264-scratch-6f04
    git -C $R rev-parse origin/cursor/circuits-cir264-sampled-accept-6f04    # 903c545eb3d34a50eccbf66352a19a805efa0aeb
    mkdir -p /tmp/cir264 && git -C $R archive origin/cursor/circuits-cir264-scratch-6f04 scratch/cir264 | tar -x -C /tmp/cir264
    sha256sum /tmp/cir264/scratch/cir264/accept_capture.sh   # 7469e283f83020f1eaeaae378f7935a6de9b7ca6557cb3377aafb40bb5c0c27d, as rehearsed
    git -C $R worktree add --detach /tmp/cir264/tree 903c545eb3d34a50eccbf66352a19a805efa0aeb
    git -C /tmp/cir264/tree rev-parse HEAD && git -C /tmp/cir264/tree status --short | head

(If `/tmp/cir264/tree` is already there, the machine is the old one: read its `HEAD` instead of adding it again.)
The files are then in `S=/tmp/cir264/scratch/cir264` and the tree in `/tmp/cir264/tree` (a worktree: the checkout stays on
`main`, and `research run --source /tmp/cir264/tree` ships that commit). `research` is
`cd /agent/repos/verity && uv run -q --project tools/research research ...`.

The timer that starts this turn carries the same steps in its prompt (`timer_prompt.md` is its text).
The off-card rehearsal of all of this (`rehearse_all.sh`; its header has the launch line) last ran from these files and
that tree with every step ending as it should: the capture's Python without a card, the card script's shell path with
its lease and the quota's refusal, and the replay. It needs no rerun unless `accept_capture.sh` changes.

## 2. Read the calendar's row again (a row can move)

    cd /agent/repos/verity && uv run -q --project tools/research research notes sync
    rg -n "2026-11-13T03:00Z" ~/.research/notes/calendars/node-a.md

The row as last read: `| 2026-11-13T03:00Z | 2026-11-13T04:25Z | circuits | 0, 1, 2, 3 | 0-39 | PROVISIONAL: ... card 0
with cores 0-9 CIR-263 and from 03:45Z CIR-264's accept kernels ... One lease a job, none over 30 minutes, the same start
check; untimed. Off every card by 04:25Z ...`. If the row is gone, released, or gives card 0 or cores 0-9 to another job
or other minutes: do not launch; return to the lead with the row's text. Only the lead writes a row.

## 3. Launch once, at 03:45:20Z or later (never before 03:45:00Z)

    cd /agent/repos/verity && uv run -q --project tools/research research run --on vy-spot-glm53 --project verity \
      --source /tmp/cir264/tree --cwd source --timeout 30m --mem-gb 16 \
      --send /tmp/cir264/scratch/cir264/accept_capture.sh --env CARD=0 --env CORES=0-9 \
      --env NOT_BEFORE=2026-11-13T03:45:00Z --env END_UTC=2026-11-13T04:25:00Z --env MAX_MIN=25 \
      --env "ARGS=--random 400" -- bash -c 'bash $RESEARCH_RUN_DIR/inputs/accept_capture.sh'

The script pins itself and the capture to cores 0-9 and takes ONE lease, on card 0, declaring them:
`GPU_LEASE_WHO=circuits-accept bash "$LEASE" 1 --on "$CARD" --cores "$CORES" --max-min "$MAX_MIN" --mem-gb 16 -- taskset -c "$CORES" "$PY" ...`.
It touches no other card and no other core. Before the card it exits without touching it when the card is not idle.

Read the run: `research fetch --all RUN` copies it to `~/.research/runs/RUN/` (`stdout.log`, `stderr.log`, `status.json`).
Wait on it with the waiting tool, never a shell sleep. What its exit status means:

- **0 or 1**: the capture ran (GO or NO-GO). Go to step 4. Do not launch again tonight.
- **3**: the capture found no cc 9.0 card under the lease. Read why; return to the lead.
- **76**: card 0 is not idle (a compute process, 1 GiB or more held, or its lease held: CIR-263's job may still be on
  it). Look again two minutes later. Launch cutoffs: with `MAX_MIN=25` a launch must start by **04:00:00Z**; from then to
  **04:10:00Z** use `--env MAX_MIN=15`; from then to **04:14:00Z** use `--env MAX_MIN=10 --env "ARGS=--random 100"`;
  after 04:14Z the hours are lost: return to the lead and ask for the Fri 17:45Z row. The script itself refuses (75) when
  fewer than `MAX_MIN` minutes remain before `END_UTC`, so nothing of ours can be on the card at 04:25Z.
- **75 with `gpu-lease: refused, holding nothing`** (the CPU quota: the lease's lines name each live lease, its holder
  and its CPUs, and the script prints `gpu-lease status` after them): **do not retry in a loop.** Write the lead one
  checkpoint line with the refusal's text (`bash $S/ckpt.sh "TEXT"`). At most one more launch, after the named
  lease's `until=` and inside the cutoffs; else return to the lead.
- **75 otherwise**: before `NOT_BEFORE` (launch again after 03:45:00Z on the node's clock), too close to `END_UTC`, or
  the card was taken between the check and the lease (treat as 76).
- **2**: the environment (no venv at `/workspace/venv312`, not the pin's vLLM, cores that cannot be pinned). Read it;
  one fix and one more launch inside the cutoffs if the cause is plain, else return to the lead.
- **124**: the lease's `--max-min` ended the capture. The cases written so far are in `out/capture/cases/`, with no record.

## 4. Read the capture (say rows compared and rows that differ, never "passes" alone)

In the fetched run: the last lines of `stdout.log` (`cases N ... inside the domain M, of them D differ; words compared W,
differ X`, `batch of ...`, `ptx: ...`, `machine code: ...`, `SAMPLED-ACCEPT GO|NO-GO`), `out/capture/sampled_accept_capture.json`
(`gate`, `summary`, `batch`, `ptx` with each kernel's `ok` and `code_ok`, `launches_disagree`, the cases outside the
domain), `gpu-lease-usage.jsonl` (minutes under the lease), `.custody` (`run_record`, the `art:` id), `slot.json` and
`status.json` (how long it waited and ran). `halves-split` is outside the Definition's domain and is reported apart: say
what the card stored for it beside the interpreter's count of 2 and lane 0's 5.

## 5. The replay off the card (needs no card hours)

    tar -czf /tmp/cir264/capture.tgz -C ~/.research/runs/RUN/out capture/sampled_accept_capture.json capture/cases
    cd /agent/repos/verity && uv run -q --project tools/research research run --on auto --project verity \
      --source /tmp/cir264/tree --cwd source --timeout 40m --mem-gb 16 --send /tmp/cir264/scratch/cir264/accept_replay.sh \
      --send /tmp/cir264/capture.tgz --env N=12 -- bash -c 'bash $RESEARCH_RUN_DIR/inputs/accept_replay.sh'

It compares every kept case with the reference again and evaluates the registered Definition on the first 12 cases (the
ten hand vectors come first; a case outside the domain is skipped), about a minute and a half a case. It ends
`definition: C cases, W words compared, X differ` and `SAMPLED-ACCEPT GO|NO-GO`.

## 6. After a GO (on the pull request's branch, in the worktree)

1. `verity/definitions/evidence.json` gets `"sampled_accept_capture": {"run_record": "art:..."}` (the capture run's
   `.custody` `run_record`), and `_CONF` in `sampled_accept.py` names it as exact-model-tested. Change nothing else in
   `sampled_accept.py` unless a kernel's word differs from the reference, and then say which lines in the return. Do not
   cut `AcceptTileHead`, `AcceptTileSum`, `AcceptRowEnd` or `AcceptResample` again (that is CIR-273's).
2. Commit; `git -C /tmp/cir264/tree push origin HEAD:cursor/circuits-cir264-sampled-accept-6f04` (never `--force`).
3. Through `research run --on auto --cwd source` from the worktree, one run in a check slot at a time: circuit-check on
   the module's targets; `uv run tools/verity/check/suites.py --changed 3498d4ad65 --quick --json $OUT/suites.json --logs $OUT/logs`
   and then `repository` whole; the retired-terms lint on the changed files; `adapters/vllm/tools/ratchet_gate.py 3498d4ad65 HEAD`.
4. The pull request, open and unmarked, by Fri 13 Nov noon PST:
   `research queue open --head cursor/circuits-cir264-sampled-accept-6f04 --sha SHA --title T --body-file F --issue CIR-264 --by circuits`.
   Body: `Lane circuits`, `Refs CIR-264`, `Goal vllm/production-vllm-reproducible`, what changed and why, circuit-check's
   report for each target, each run id with what was read in it, a "Not run" list; and that the `logits` port's writer is
   CIR-273, that vLLM's split top-p pipeline takes at most 64 rows a call under a draft [vllm's], and that how leakage
   counts the masked row is asked of proofs on CIR-264. The lead runs the pre-mark checks and marks it.

After a NO-GO: read which words differ (`cases[*].notes`), keep the run, change nothing in the tree on a guess, and return
to the lead with rows compared, rows that differ and the first differing words.

## 7. Before the turn ends

- Commit and push everything: the pull request's branch if it changed, and a checkpoint line with any new scratch file
  (`bash /tmp/cir264/scratch/cir264/ckpt.sh "TEXT" [FILE...]`). A later turn starts from origin and the conversation.
- No timer is left at the final return (`list_subscriptions`); say how many were held: 2 (`cir264-card-hours`, replaced
  before it fired, and `cir264-capture-0320z`).
- Return to the lead: the capture's run id, its wait and its minutes under the lease, cases and words compared and that
  differ, `halves-split` on the card, the PTX and machine code verdicts, the replay's run id and its line, which lines of
  `sampled_accept.py` changed (or none), the pull request, and what was not run.
