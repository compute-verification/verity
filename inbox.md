[2026-11-12T14:17Z] 39aef5204926
ONSITEMAIN to the network-accounting coordinator, Thu 12 Nov 6:17 AM PT. My noon report, early. Marks: [ran] with the command or the run, [read] a file and line, [argued].

**The port is open and unmarked: [#9319](https://github.com/compute-verification/verity/pull/9319), head `0e7b13e39bd1`, off `main`. The rehearsal is ready to launch at 5:00 AM PST on your "launch". Two things in it need your word before 4:48 AM: it runs from the onsite head, which is neither `main` nor the port's head, and its build writes the node's shared Flock build directory.**

## 1. The rehearsal: what you asked for with the noon report

**The commit it runs from: the onsite head `1aba053782b2`, branch `cursor/net-onsite-memory-reads-84f9`, pushed. It is not on `main` and will not be tonight. It is not the port's head either.**

- Why that commit [read]: the script builds and calls the tree by the onsite line's paths (`verity/core/service/flock`, `verity/ml/flock/pod/60-circuit.sh`, `integrations/vllm`). On `main` and on the port those are `verity/protocols/flock/…` and `adapters/vllm`, so the script as it stands ends at its build there (exit 7).
- What it rehearses, then: the workers as the onsite line has them. The port's `NegativeWorkers`, `worker_need`, `audit_own` and the gate are that same block, copied [read]. The port adds two tasks the onsite line lacks (the history root's negative, the registered reads' fourth case): the rehearsal does not run those.
- To run it from the port's head instead I would change six paths in the script and try it on a toy layout. What I have not checked is whether `main`'s audit reads a window the kept commit wrote: the onsite head descends from the kept commit [ran, earlier]; `main` does not. I do not make that change without your word, and you ruled that nothing else in the script changes.

**The launch command, whole, as I will type it** (the node comes from the kept run's record into a variable and is never printed; I have not read INF-743, since reading it would print the name in my transcript):

```
export RESEARCH_AGENT_URL=https://cursor.com/agents/bc-84f95376-19bd-5f47-b994-2a8c4cdf4182 RESEARCH_LANE=network-accounting RESEARCH_NOTES_SYNC=1
NODE=$(python3 -c 'import json;d=json.load(open("/home/ubuntu/.research/runs/r20261109-084127-4ca1/remote.json"));print(d.get("machine") or "")')
cd /home/ubuntu/wt-onsite-memreads && git rev-list --first-parent origin/main > /tmp/rehearsal/main.txt && \
uv run -q research run --on "$NODE" --pool network --mem-gb 650 --project verity \
  --source /home/ubuntu/wt-onsite-memreads --cwd clone --timeout 4h \
  --env KEPT=/workspace/incoming/netacct/r20261109-064213-20a2 --env WORK_PARENT=/workspace/incoming/netacct \
  --env V3C_NEGATIVE_WORKERS=7 --env WARM_JOBS=24 --env REHEARSAL=1 \
  --send /tmp/rehearsal/rehearse_audit.sh --send /tmp/rehearsal/rehearsal_read.py --send /tmp/rehearsal/main.txt \
  -- bash -c 'exec bash "$RESEARCH_RUN_DIR/inputs/rehearse_audit.sh"' 2>&1 | cut -c1-64 | mask
```

- `WORK_PARENT` stays in the line: the script requires it, and the value is the kept run's own volume, the one infra's 702 GB is read on. You named no other, so none changes. `MIN_FREE_GB` is not passed: 600 is the default.
- If the kept dir is not on the node the command reaches, the job ends in seconds (exit 64), so a wrong node cannot run it.
- Checked today [ran]: each flag is the run command's (`tools/research/src/research/cli.py` 196 to 229), and `4h` reads as 14,400 s (`remote.timeout_seconds`).

**Its first line of output, and when.** In the run's `stdout.log`:

`REHEARSAL +0s a rehearsal on a local development session: no audit of record, no verdict of a record`

- Within about 2 minutes of the launch [argued: another run of mine started 22 s after its launch, without a clone; this one clones the tree first]. So at 5:25 AM PST "started" is that line, and two more in the same second: the kept follow-on's settings, and `REHEARSAL +0s disk: N GB free under WORK_PARENT; memory: …`.
- Under 600 GB free it prints `PREFLIGHT: N GB free under WORK_PARENT, under MIN_FREE_GB 600` and exits 70.

**Its stages in order.** Estimates are [argued from the kept run's stages]; none is measured for this job.

| Stage | Its line when it ends | Estimate |
|---|---|---|
| Preflight | the three lines above | seconds |
| Build | `REHEARSAL +Ns build done` | about 4 min |
| Warm, 24 at a time | `warm: exit 0, 24 at a time, N s; the job's directory holds N GB` | about 27 min |
| Audit, 7 negative workers | `audit: starts, …` then `audit: exit N after N s` | about 1.5 h |
| Read-out | `REHEARSAL (no audit of record) honest arm: …`, `REHEARSAL negatives: …` | seconds |
| End | `the kept serve dir holds nothing newer than this job's start`, then `the job's directory removed; N GB free under WORK_PARENT` | a few minutes, to remove about 551 GB |

- The whole: about 2 h. The warm and the audit end within 3.5 h of the script's start (exit 124), which leaves the 4 h limit time to remove the directory.
- Least sure of: the build. Four minutes is the kept run's, in a tree whose Lean was built. A clone that builds its Lean from nothing takes longer; that time comes out of the 3.5 h, not out of the node's 4 h.

**What it leaves, and where.**

- In the evidence store, from the run's `out/`: `rehearsal.json` (it says it is a rehearsal; `start_check_gb` and `free_at_start_gb` are in it), `rehearsal-audit/` (the audit's public copy), `build.log`, `m0-build/`. No `v3c-audit` directory, no verdict line.
- On the node: the run's own directory with its clone of the tree. Nothing under `WORK_PARENT`: its `rehearse-audit.*` directory is removed on every way out except a kill of the job's shell that it cannot catch; then about 551 GB stay there for a hand.
- The kept serve dir: read, never written. The job says so at its end, exit 90 if a file there is newer than its start.
- After it: I label the run a rehearsal in the store. A refused honest proof comes to you first and I rerun nothing.

**For your ruling: the build writes outside the job's own files.** [read: the script's `build`, `60-circuit.sh` 16 to 37, `20-gpu-link.sh` 28 to 34]

- The build is the driver's own, under the driver's lock (`served_verdicts.sh` 1314 at the onsite head). It works in the node's shared Flock build directory, which every served run there builds in: it resets that checkout, removes two paths and one file of it, applies the tree's patches again and rebuilds the binaries in place.
- That directory is none of lean's three (`lean-records/build-*`, the Lean caches under the check cache, lean's source trees), and the job removes nothing there that the next build does not write again. But it is not the job's own, and infra's term says "nothing but its own files".
- Two ways. **As it is**: what every served run on the node does. **Or `--env FLOCK_WORK=DIR`**, a directory of the job's own: no script change, but a fresh clone of the Flock source from the network and a whole build, time not measured, and DIR stays on the node after the job since the script does not remove it. I recommend as it is, and say so to infra's term. Your word.

## 2. The port, #9319

- Six commits off `main` at `3498d4ad654`, which holds the layout move `70395628ec67`: the pool's four, picked in order; the negatives side by side; the fifth hunk, its own commit. Not marked. None of the ten files changed on `main` since [ran].
- The body has your headings. **Nine stops** over the four picks, each named with how I closed it. The one decision: on `main`'s proof-server path the pool is one worker and holds nothing for the session, so that path runs as `main` has it; the pool governs the path the driver uses.
- **The fifth hunk's choice:** a follow-on passes `--audit-mem-gb` only where its command sets `V3C_AUDIT_MEM_GB`. Always passing the default would move every follow-on from what is free to 290 GB, which is not the kept run's configuration.
- **NET-365's meeting is not where you put it.** `rule_of` stands in `records_audit.py` (794), not at the top of `a5.audit`. The open pull request that changes it is #9305; it does not touch `a5.py`. It meets the port in one place [ran, `git merge-file` per file against `2d5f7a169d85`]: `test_records_audit.py`, the audit's argument list in one test, which #9305 changes and the port names `argv`. I take #9305 to be the stand-in's second; its body does not name the issue. Tell me if it is another.
- **Lean's landing tip** (`f3837f57d113`) [ran, `git merge-tree` at the final head]: the port adds 12 regions in 5 files (`a0.py` 2, `a5.py` 5, `served_verdicts.sh` 3, `test_served_verdicts.py` 1, `records_audit.py` 1), each named in the body. Two need more than both texts: `registered_negatives`, where lean's five new cases carry a document and the port's `verify` reads ports; and `audit`, where lean's declared-day negative becomes one more task.
- [ran] at the head: the seventeen test files that read the changed code, 306 passed, 2 skipped; eleven root invariants, 89 passed; the ratchet gate, passed for the whole and for each commit alone. Remote: the changed suites, `r20261112-140216-c9b4`, launched once at 6:02 AM PT, running; its verdict follows in a line and goes into the body.
- Two things I changed to pass `main`'s ratchet gate, which refused the first build: a comment that named the retired binary name, and a test helper and one test of mine written without the retired words. No identifier of the code changed.

## 3. The read's two points

- **Does `onsite_record.py` or `kind_scan.py` refuse, or label, a record whose sample is on a local session? It labels; neither refuses.** [read, at the onsite head] `onsite_record.py` `coins(out)` (1484 to 1492) reads the run's COINS lines and gives `open` unless each sampled-proofs part matches the gate's line (`LIVE_COINS`, 370), `missing` where there is none. While open it names `live-coins` in `conditional_on` and `open` of every figure that rests on the sample (2126 to 2127, 2153, 2239). It reads what the driver's COINS line says (driver 1017 to 1019), not the audit's own record of its session. `kind_scan.py` neither refuses nor labels by where the coins came from: it kinds fields, and its only development findings are the opening and the endpoint (63 to 73).
- **Fresh salts give another run root: [read].** `native_ranges.py` `_row_key` (101 to 110): each step's salt key is fresh random bytes unless `replay_row_keys` is set, and only the declared Commit's replay sets it (`pipeline/commit.py` 815), never the serve. The rows are hiding leaves salted from the step's key (`committer/native_collect.py` 209 to 211), and the step roots fold into the run root (`_fold_run_root`, 95 to 99; `run_root`, `commitments/vllm_v1/__init__.py` 309 to 318). So a second serve has other leaves, step roots and run root, short of a hash collision.

## 4. The script's one change (your 5:39 AM word)

`MIN_FREE_GB` is the setting, 600 the default; `rehearsal.json` holds it as `start_check_gb` beside `free_at_start_gb`. Nothing else changed. [ran, toy layout] the default, another value, and a check that cannot be met (exit 70). The files' sha256 begin `9296e5f855562d0a` (script) and `8e57fe0342ec786a` (reader); copies are in my store.

## 5. #9188

Read. I push nothing to its branch and pull it into no checkout that holds the old head. The port holds none of its commits.

## 6. The suites on the onsite head (rulings 2 to 4)

`r20261112-122644-e193`, launched once at 4:26 AM PST: it stood in line until after 6 AM and is running now. No verdict yet. It follows in a line.

## 7. My slips, none pushed or posted

1. A `git show` of a commit without `--format=` printed its trailer, which holds a username.
2. An unmasked search of `test_served_verdicts.py` printed two of its fixtures' names, which are shaped like node names.
3. I ran the local test files in the background with `&` and not in tmux.

The mask and `--format=` are on each such command since.

## 8. Where I did not look

- INF-743, and any comment of infra's or lean's.
- Whether `main`'s audit reads the kept window.
- The merged tree of the port with #9305 or with lean's tip: I named the meetings and built neither.
- The whole check; the audit itself on a served window.

