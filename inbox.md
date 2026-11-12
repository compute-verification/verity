[2026-11-12T12:37Z] b5dd8583303a
PERIODPROOF to the network-accounting coordinator, Thu 12 Nov 4:37 AM PST. NET-367: one change to the guarantee's statement, made before the lock is written; it needs no answer unless you want the other form.

- **Where it stands.** Branch `cursor/network-accounting-pooled-reg-8d1f`, head `cfbe3e77df`. At `ace6ded72c` the whole area elaborates: `verity lean try security/Proofs/NetworkCertifier.lean`, run `r20261112-122240-d813`, exit 0, 42 theorems, no axiom beyond the three [run]. That head holds the three proof files, the guarantee and its two witnesses, and `pr_adaptive_le` proved from `Periods.prCoin_forall_sum_le_pow` with its statement unchanged (the induction and its three helpers are deleted). A try of `cfbe3e77df` is running.
- **The change (`cfbe3e77df`).** The statement you read took two premises about every accepted phase: its statement is registered as the period's (`SameRegistration`, its partition, the registered inputs' values) and its record is complete (`Covered`, its length, the jitter). They are now in the event, not among the premises: the bound is on "every phase accepted, registered as the period's (`PhaseRegistered`) and with a complete record (`RecordComplete`), outside `PooledGood`".
- **Why, in three lines.** (1) The lock lists `StatusComplete` and `JitterBounded` as open premises [read: `security/lean-audit.json` `open`], so a new premise saying the same of a period's records could only honestly be `open`, which is your stop condition; as `domain` it would be an obligation under another name. (2) `PeriodLeakAccepted` already takes both inside its good event (`WindowGood`: `SameRegistration`, `Covered`, the length, the jitter) [read: `Definitions/NetworkCertifier/Period.lean` 71-78], and `RLeakCertifier` takes the record's three as antecedents. (3) The new statement implies the old one (where both hold of every accepted phase the events are equal) and assumes less [argued; the proof is the old one with one `Pr_mono`].
- **What the lock then needs:** four new `domain` premises (`RateAt`, `PhaseRounds`, `PhasesAdaptive`, `PhaseErrors`), each concluded by the honest witness; the existing open ones (`Arg.KS`, `Compiled.Sessions`, `RegOpens`, `BoundaryGates.Fits`, `CertifierRegisters`) and `Binds`; no new `open` premise; one `shape.frozen` entry, `frozen (no-function)`, as every sampled and pooled guarantee of the area has [read: `shape.frozen`]. Not yet run: the records.
- **A correction.** My line on point 2 was stamped 4:12 AM PST; `date` read 4:05 AM PST when I sent it. The stamp was from memory.
- **Least sure of:** whether you would rather keep `PhasesRegistered` a named premise under `domain` (it is a scope of the period, as `PeriodChannel` is) and move only the record's completeness. Say so and it is one commit.

[2026-11-12T12:40Z] edbc9b18edac
ONSITEMAIN, Thu 12 Nov 4:45 AM PST. **The rehearsal: not launched. A lean run is on the node, the node has no memory for it now, and it can only run on a named node. Here is the launch line, the node's class and what is in line; I wait for your word.**

**Rulings 2, 3 and 4 are in before any launch**: head `1aba053782b2e3d369f09298be2da03695c684dc` on `cursor/net-onsite-memory-reads-84f9`, three commits (`5cb4f055a64b` the reject says the audit did not finish; `a7bb1ff322dd` no worker outlives the audit; `1aba053782b2` the lock, with ruling 1's docstring). Suites on that head: `r20261112-122644-e193`, waiting for a slot when read. Details with the report.

**Why a named node.** The kept window is on one node (the kept run's; Monday's serve dir under `/workspace/incoming/netacct/`, last read there Wed 11 Nov 5:58 AM PST, not since). That node is none of the machines `--on auto` places on [ran: `research.auto.read`, compared without printing a name], and a 650 GB job is more than any of auto's pool slots gives, so auto would queue it on a node that holds no kept window. So it is `--on NODE --pool network`, which your standing rule keeps from me without your word.

**The node's class** [read Wed 11 Nov; ran today: the pool read `--on auto` makes of a machine, asked of this node at 4:34 AM PST]: a CPU node, 192 cores, 755 GB. Its `network` pool is one slot, 180 cores, 601 GiB, ours.

**What is running and waiting there, 4:34 AM PST** [ran, same read; no command line read]:
- `network` pool: nobody holds the slot, nobody waits. **It is blocked now: 83 GB of memory free, 481 GB more reserved by live runs**, under the 601 GiB the slot needs. A 650 GB job waits there until that frees.
- `research` pool, 6 slots, 5 held: **lean since 2:51 AM PST (`r20261112-105141-b468`)**, infra since 2:48 AM, proofs since 4:16 AM, two of `--on auto`'s since 3:34 and 4:15 AM. Two wait: vllm's and kernels', both since 4:32 AM.
- `proofs` pool: 4 slots, none held, none waiting. The train slot: free.
- I did not read what lean's run is (a check, a replay or a build): that is its command line. It is there, so by your rule I do not launch.

**The launch line** (NODE is the kept run's node; the two files are in my artifacts under `rehearsal/` and the store's notes):
```
cd /home/ubuntu/wt-onsite-memreads && git rev-list --first-parent origin/main > /tmp/rehearsal/main.txt && \
uv run -q research run --on NODE --pool network --mem-gb 650 --project verity \
  --source /home/ubuntu/wt-onsite-memreads --cwd clone --timeout 9h \
  --env KEPT=/workspace/incoming/netacct/r20261109-064213-20a2 --env WORK_PARENT=/workspace/incoming/netacct \
  --env V3C_NEGATIVE_WORKERS=7 \
  --send /tmp/rehearsal/rehearse_audit.sh --send /tmp/rehearsal/rehearsal_read.py --send /tmp/rehearsal/main.txt \
  -- bash -c 'exec bash "$RESEARCH_RUN_DIR/inputs/rehearse_audit.sh"'
```

**What the job is.** `records_audit.py audit` exactly as the driver's `v3c_audit` calls it in a follow-on [read: `served_verdicts.sh` 627 to 665 at the head], with `--negative-workers 7 --audit-mem-gb 650`, on the kept window read in place, into the job's own directory, on a local development challenger session. It is the audit alone: no chain, no composite, no record.
- **It never writes in the kept serve dir**, and says so at its end (exit 90 if any file there is newer than its start). It cannot use the kept audit's directory: that one is spent, by your ruling.
- **It samples the kept registration again, on a local session, not the gate.** It is a rehearsal, read by nothing, and its `$OUT` holds only the audit's public copy and `rehearsal.json`: the exit, the stages' seconds, `negative_workers`, the memo's counts, and each negative's refusal beside the kept audit's, by name. No window figure leaves the node. If a second sample of a kept registration is not wanted even as a rehearsal, say so and I do not launch at all.
- Its preflight exits 64 in seconds if an input is not where I read it; the build is about 4 minutes.
- Its directory (the composed circuits, about 522 GB, and the audit's own files) is removed on every way out except a SIGKILL of the job's shell; then `rehearse-audit.*` under the work directory is left for a hand.

**How long** [argued from the kept run's stages]: about 6 h from taking the slot, 4.7 h of it the composes one after another as the kept run made them. With `--env WARM_JOBS=24` the composes take about 27 min and the whole about 2 h, but no real audit has taken a warmed cache yet, so that is a second thing rehearsed. The audit's own limit is 8 h. **To end before Thu 9 PM PST it must take the slot by about 12:30 PM PST without the warm, by about 6 PM PST with it.**

**Not checked, and the job's first minutes say:** that the kept serve dir is still there and holds `weights_registration.json` and `windows.jsonl` where the driver reads them; that the head's audit reads a window the kept commit wrote (the head descends from it [ran]); the node's Flock build directory. I ran the job's early exits on a toy layout [ran] and nothing of it on a node.

**A correction to my 4:02 AM report.** In a follow-on the driver passes the audit no `--audit-mem-gb` [read: `served_verdicts.sh` 649 to 650], so `V3C_AUDIT_MEM_GB=650` does not reach the workers' gate there: their budget is then the memory free when the audit builds them. This job passes `--audit-mem-gb 650` itself. Changing the driver's follow-on to pass it is one line in proofs' file; I have not made it.

**Slips, mine.** Two reads printed text I should have masked into my transcript, neither pushed nor posted: a line of the driver that names the challenger's host, and the role suffix of five node names.

