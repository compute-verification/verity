[2026-11-13T18:35Z] 8cea4cc90750
GATERELAY to network-accounting (lead), Fri 13 Nov, the 1:00 PM line (NET-435). Nothing of the verb is built yet; base `fa3e619ee146`, branch `cursor/network-accounting-gaterelay-b51f`, nothing pushed on it yet. Stop me on any line below before I write it into the gate's docstring.

## 1. The verb and its contract, in ten lines

1. **Verb:** `relay-session --peer NAME --port PORT --client-port PORT --windows N --minutes M [--scan [--bucket-ms MS] [--scan-class CLASS]] [--watermark] [--height H]`, sent with a lane's own gate key. It takes none of a rehearsal's traffic flags (`--requests`, `--tokens`, `--prompts`, `--window-per-group`, the sampling): the gate sends no request in a session.
2. **Start:** the gate checks it with the rehearsal's own parser and bounds (peer, ports, the scan's schedule, class and tick, the height), writes `request.json` and starts the rehearsal's unit; one run at a time with the rehearsals, by the same refusal. The reply is a rehearsal's: the request, `run` and `unit`.
3. **Length:** `--windows` is the session's link-windows, fixed at the start and named by `deployment.json` before the first. The gate refuses a key that cannot sign them (`warden_key`, five signatures a link-window) and `--minutes` that do not hold the key's generation, the windows and the check.
4. **Where a client connects:** `127.0.0.1:CLIENT_PORT` on the peer. The gate's one ssh to the peer (the tunnel key, as today) carries the rehearsal's `-L` to the endpoint's port and a `-R` from that port to the relay's listener on the gate's loopback. The client speaks the relay's cells (`warden.cells`, `Shim`) at the schedule `deployment.json` names.
5. **Ready:** `status RUN` gains `relay`: null until the relay listens, then `{"client": "127.0.0.1:PORT", "deployment": {...}}`, the public `deployment.json` (key, session, schedule, `origin_unix`, `windows`, run id). A client reads the schedule and the origin there and sends from the origin on.
6. **End:** the relay stops after exactly `--windows` link-windows from its origin; the runner then checks the run and the gate publishes and writes `end.json`, as a rehearsal's end. `status` then has `end`, and `fetch RUN` gives the bundle. No verb ends a session early (why: section 4). The unit's time limit reaching it first is a failed run, which publishes its logs alone.
7. **What it publishes** (any key): `deployment.json`, `cells.jsonl` (the registrations), `certificates.jsonl` (certificates without heads or salts) and `end.json` (the check's verdict and counts, the registrations, each file's hash). **Witness** (the run's own lane and vy-admin): the three records, `certified.jsonl`, the cells' salts, the relay's log, `check.json`. The gate holds no client's log: its check reads the records, their certificates and registrations and the audit, and nothing of what a client received.
8. **A connection open at the end is cut.** Still queued: closed with no cell, in no record. Admitted and inside its first D buckets: the record has its ingress cells (its request may have been released to the endpoint) and no answer cell, and the check does not refuse. Past bucket s + D with no `END` or `FAIL`: the check refuses the link-window of its admission, so the run is not accepted and our piece answers nothing for it. So a client's last connection is admitted by bucket `windows·T − span − 1` (span = D + ⌊(egress_cells − 1)/r⌋, 28 buckets at w63k4, 43 at w63).
9. **What it establishes:** its records cover the connections that crossed the relay. That nothing else reaches the endpoint is the placement's and vllm's forward; the docstring and the pull request say so.
10. **The rehearsal** (`warden-rehearse`, `load-relay`) is unchanged: same flags, same request, same files.

I read line 8 off `warden.audit` and `Warden.run`; the test of item 4 of your brief asserts each of its three cases, so if one is wrong the test says so and I tell you.

## 2. How the gate on the auditor's machine takes a new verb

It is infra's step. The gate runs only the pinned tree: `vy-challenger-pin COMMIT`, as root on the auditor's machine, fed a `git archive COMMIT` of four paths (`verity`, the live challenger's directory, `benchmarks/network_traces`, the int8 checkpoints' manifest), which `pin.sh` takes only when COMMIT is on `origin/main`'s or `origin/next`'s first-parent history, and logs. A run uses the tree it started at.

So: **no run on the real gate can use the verb before a pin of a commit that holds it**, and that commit must be on `main` (after landing) or on `next`. Before the pin only `--gate local` runs it (this tree's gate in process on a scratch directory: the dry run, which never stands for the gate). To ask infra today: (a) will they pin from `next` ahead of landing, so that vllm can rehearse before Thu 19 Nov, and what puts a commit on `next`; (b) **do the serving nodes' sshd let the auditor's tunnel key listen with `-R` on a loopback port outside the challenge links' range 7811–7899?** The tree answers it only for the verifier's machine (a `PermitListen` list). The challenge links already use `-R` to the same peers inside that range; I refuse a `--client-port` inside it, since `free_port` hands those to challenger sessions and knows nothing of our run. If the answer is no, the fallback is a port range of our own in the nodes' sshd configuration, also infra's.

## 3. What the served runner needs to send another lane's load through the session

**Not `LOAD=front` as it is.** `LOAD=front` means the front's own group is the run's one load; the runner calls `CERTIFIER_FRONT` once, blocking, before the load, and sends every other load straight to `$PORT` with `serve_load.py`, which speaks HTTP, not cells. By file and owner (`research msg owners`):

| File | Owner | What it needs |
|---|---|---|
| `benchmarks/one_stage/served_verdicts.sh`, `tests/test_served_verdicts.py` | @proofs | A third `LOAD` value under which the load goes to the session's client port (STAMPS's 2, at the five tests of `front`), and the front's two halves around the load (STAMPS's 1): start-and-wait-ready before, wait-end-and-fetch after. The second half needs no line of the runner if our piece closes a session nobody closed, which I can build; the runner then only loses the outcome in its CERTIFIER-FRONT line. `front_ok` holds the new flags as it is if they stay flags of `certifier_piece.py front`. |
| `demos/vllm/serve_load.py` | @vllm | A sender that speaks cells to the client port and reads with `Shim` (STAMPS's 5). #9510's `cell_shim.py` already reads `deployment.json` for the address, the cell size and the origin: the `status` reply's `relay.deployment` is that file. |
| `demos/onsite_served/served_table.py`, `config.toml` | @memory-accounting | `LOADS` gains the name; `_certifier_piece` returns the session's calls; a binding for the client port (STAMPS's 12, 13). |
| `tools/verity/cli/tests/test_onsite_served.py` | @ci | Its cases for those. |
| `benchmarks/network_traces/certifier_piece.py`, `served_session.py`, `warden_serve.py` | ours | Mine: `front --session` (starts a session, sends no group, returns once `status` shows the relay ready, leaves `deployment.json` and the client's address in the serve dir's private directory), `close` (waits for the end, keeps the fetch where `kept_run` reads it, writes `session.json` through the leak guard), and `--gate local` for both. |
| `verity/protocols/compose/device/live_challenger/auditor.py` | **@proofs, by the registry** | The verb. Your brief gives it to me; the registry says the file is proofs'. I go on writing it unless you say stop, and name it in the pull request for proofs to read. |

I change none of the first four rows. Tell me which of the two shapes for the runner's second half you want proofs to see; I build both calls either way.

Our piece's own conditions stay as STAMPS read them (6 to 9): one kept run whose check passed, every request of the audited window a certified one by its `user`, each inside one link-window, N2 and the tie on every link-window only of a run of at most 8.

## 4. What in the relay's runner assumes its own client, and what I do

| Assumption | Where | In a session |
|---|---|---|
| The traffic is the gate's (`--requests` at the gate's rate and seed, or prompts on stdin in groups) | `warden_command`, `rehearse`, `traffic_of` | None is made. A new subcommand of the runner, sharing `rehearse`'s body: key, `serve`, `check`. No `client` process. |
| The run's length is the traffic's last arrival plus a tail | `rehearse` | `--windows`, given. |
| The minutes are bounded from the requests or groups (`warden_minutes`, `warden_group_minutes`) | the gate's parser | Bounded from `--windows` by the same `_minutes`: the windows' span, the origin's delay, the key and the check. |
| The key's height is held against `warden_windows(requests)` | `warden_key` | Held against `--windows` directly, same function. |
| The check reads `client.json` (each cell in its bucket at the client, each answer's hash) | `check` | No `client.json`, which `check` already takes: it then holds the records, certificates, registrations and the audit. **What a client received is not checked at the gate**; a client checks its own log against the fetch. I say so in the contract. |
| The relay's length is fixed and public before the first link-window | `Warden.run`, `deployment.json`, `check` | Kept. **An early end** would change `Warden.run` and what `check` counts, both the relay's trusted side, and the key is sized to the windows. I leave it out of this pull request: a session that outlives its load certifies empty link-windows, and costs their time and signatures. If the demo needs it, it is a later verb that changes no flag of this one. |
| The client is on the gate, beside the relay | `rehearse` (`--listen 127.0.0.1:0`) | The gate picks the listener's port and forwards the peer's client port to it (line 4 of the contract). |

## Also

- #9782 (the table's `SERVE_ARGS`): its node run `r20261113-182526-cb56` is not read yet; its result goes into the body next and I send you one line.
- Next: the contract into the gate's docstring and the runner's header, pushed, and its commit to you.

[2026-11-13T18:37Z] b2f980ff2b28
DEVICEKEYS to the lead, Fri 13 Nov 10:37 AM PST. **A fault of my own job script, which other workers' jobs may share: the suites step ran with `TMPDIR` exported to the shared scratch, and two suites fail there on 13 tests that read the temporary path.** No code is at fault and neither pull request is touched.

- **What failed (ran).** #9687 at `0a9c5578544c`, `r20261113-171257-8922`: pins' regen rewrote nothing; 63 suites, 61 passed, 2 failed. #9369 at `68b00f3bc1fb`, `r20261113-170558-1726` (its records step still running): pins' regen rewrote nothing; 86 suites, 84 passed, the same 2 failed on the same 13 tests.
  - `research` (tools/research): the ten `tests/test_nebius.py::test_n1_custody_*`. The custody round refuses a tree under a directory another uid can change, and the scratch root is one.
  - `verity-vllm` (adapters/vllm): `test_pouw_h1_handoff.py`, two tests, `AF_UNIX path too long` (a socket under the temporary path); `test_gate_plan.py::test_a_record_that_owns_no_call_under_a_values_module_is_said`, whose expected line is cut at 200 characters before the digest when the path is long.
- **The cause (ran, on my VM at `0a9c5578544c`).** All 13 pass with the default temporary directory. The ten fail with `TMPDIR` under a group-writable directory, by that refusal; the three fail with a 60-character `TMPDIR`, by those two errors. My script exported `TMPDIR` for the whole job; your order asked the scratch for the pins' regen only.
- **What I did.** Relaunched the two suites alone at each head with the machine's own temporary directory: #9687 `r20261113-183616-ad6b`, #9369 `r20261113-183647-2929`. I push nothing to #9369 and send you its list when its job and this rerun have ended.
- **For the others:** a job that takes the pins' scratch by `export TMPDIR=$(mktemp -d -p …/scratch)` should unset it before the suites.

Also ended, both passed (ran): the Lean check of the certifier's package at #9687's head, `r20261113-171401-79d6` (1613 declarations, 4 guarantees); the restated `CertifiedBy` and `RegisteredBy` elaborate at `6c641d2efd9a`, `r20261113-171614-f1b0`. 7 PM PST still looks set to hold.

[2026-11-13T18:39Z] 21c5025a42e9
ONSITEMAIN, Fri 13 Nov 10:38 AM PST. L and I1 carried and pushed; the finisher's thread built and its run launched; the rehearsal's commit pushed; #9319's run passed. Nothing marked.

**Carried heads, the last two** [rev-parse and ls-remote each; the tool's word `clean` on each]
- **#9678 (L)** `a3981a146b6d4e30ffe5eac229696539e358d244`, run `r20261113-181431-fabd`. The tool's result `a532f0063d02`, then **one hand merge** of A's carried head `de84698df480`: no conflict, no file resolved by hand. L's own change over A is the same six files, +367 −3, before and after [ran].
- **#9679 (I1)** `a5847b8f78ed0bccb167cf94280bca33228f9bde`, run `r20261113-183712-7c78`. The tool's result `fb227bbd455b`, then **one hand merge** of A's carried head: no conflict, no file resolved by hand. I1's own change over A is the same three files, +689 −5 [ran]; 19 passed in its test file [ran, local]. One thing for you: I1's `kind_scan.py` and its test name `served_integrity.public_copy`, which `main` does not have (the public copy's answer, below, is the same gap).
- On both, the inner merge of the two bases printed `merge_rows.py: …PROTOCOL.md: 1 conflict(s) left`; the file in the result holds no marker and equals A's carried head's and the tool's result's [ran].

**#9319's run** `r20261113-175646-88ac` at `214defbcafba`: exit 0. Pins regenerated 0 paths; six suites, six passed, none failed (one-stage 1109, network traces 716, repository 357, graph 220, prove path 170, the command 133) [ran, pool].

**The finisher's thread: `b82b94e92007539b86cf625b86ff11f76e963d83` on #9319** (one commit on top; pushed)
- **By reading, it can be alive at the first fork; nothing orders it.** The thread's last act is to publish the evidence; the audit reads the evidence before a negative is submitted (`coins_outcome`), and nothing joined the thread before the workers' fork.
- **By a local run of the real class** [ran, my VM]: `a5_coins.Draws` in its manager, a thread whose last act is `publish`, the caller reading `evidence()` and then asking: the thread was alive in 1438 of 2000 rounds. That is the instant after the read; the audit does more between the read and the fork, so the toy's run says how it stands at the fork itself.
- **The change:** `a5.audit` takes the session's thread (`session=`), and before its threads are read for the first fork it reads the evidence and joins that thread (`session_ended`). The refusal stays for every other thread, and for a caller that hands no thread in. The reading kept and printed before the fork gains Python's threads as they were before that wait (`before_session_ended`), so every run shows whether the thread was alive there.
- **Files: four, all #9319's own:** `benchmarks/one_stage/a5.py`, `benchmarks/prove_path/records_audit.py` (it hands its thread in: one argument) and their two test files. It reaches no other lane's file.
- **Tests wait on events** [ran, local]: the session's thread is held alive until the audit's read of the evidence, the first fork then sees the caller's thread alone and the workers run; a thread that is not the session's is refused as before. 31 passed in the two one-stage files, 50 in the records audit's; `tests/test_no_wall_clock.py` and the retired terms pass.
- **The settling run, launched once:** `r20261113-182751-cab5`, at `b82b94e92007`: the two-member toy window's audit (the test file's own: the real prover and Lean verifier, a local session), `--negative-workers 2`, eight times; each prints the threads before the wait and at the fork. I read "the shortest window the table allows" as the smallest window, two members; the session's minutes are the default, since `start` takes none. The suites at this head: `P9319b`, launched now.
- **Not changed, and yours to rule:** eight other callers hold the same thread and call the same audit without handing it in, so with more than one negatives' worker each can still be refused for it on the race: `prove_path.py:1237`, `a5_warden.py:463`, `a5_sparse_mla.py:110`, `a5_tp_gates.py:463`, `a5_row_tables.py:85`, `two_stage.py:902` and `:1033`, `two_stage/audit.py:210`, `network_traces/scan_zk.py:1294`. Each would take one argument, `session=finisher`, where it calls `a5.audit`. None is in #9319's ten files, so I touched none. The served run's path is `records_audit.py`, which is covered.

**The rehearsal branch: `b3c0d0c57b14f8fdf25d2eb309be00b32390f2db`** (toy tree only, no pool launch)
- The stop's message names which reading stopped the job, the scope's or the node's, with both readings as they were then: in the LOW MEMORY line, in the exit's own line and in `rehearsal.json`'s `stop_reason`.
- **The figure:** unset, the node's reading now fires at a fall of the scope's limit less `MEM_ROOM_GB`, in whole GB of the limit. For a limit of L GB and the default room of 32 it is floor(L) − 32: in the toy, 617 under a limit of 649.2. It is under the cap by the room, by construction. Before, it was a fixed 650 whatever the limit, which a job capped at or under 650 could not reach. What limit in bytes Friday's `--mem-gb 645` gave the scope I did not re-read; a run's start line says the figure and which of the three it is (derived, set, or the default where no level has a limit).
- A `MEM_STOP_GB` set at or over the scope's limit is refused at the preflight (exit 64) with the fix. This is a choice of mine: say if a warning is wanted instead.
- Six toy cases [ran]: no stop (exit 0, the start line says 617 and why); the scope's reading stops it (72); the node's stops it at a set 400 (72); the node's stops it at the default 617 (72); a set 650 refused (64); no limit (default 650, exit 0).

**A's suites, a thing to settle** (run `r20261113-170813-b455`, still running): two suites failed so far, neither under a file A changes [read]: `research`, ten tests of `tests/test_nebius.py` (`test_n1_custody_*`), and `verity-vllm`, three tests (`test_gate_plan.py` one, `test_pouw_h1_handoff.py` two). A's change touches the public boundary's table, which I take to be why those suites were chosen [argued]. Whether `main` at `f54194bbb196` fails the same thirteen I have not run: I will run those two suites' failing tests once at that commit when one of my runs ends, unless you say no.

Least sure of: that the eight other callers should be left as they are; the preflight's refusal of a set `MEM_STOP_GB`.

[2026-11-13T18:41Z] b24bcc5ebdfe
TERMGB to the lead, Fri 13 Nov 10:41 AM PST. #9305's base merge, tried in scratch: it comes down to `main`'s side whole in the three files and one flag. Still nothing pushed; your word. (My last line was headed 10:26; the clock read 10:24 when it went.)

**What the trial holds** [ran, in a scratch worktree at #9305's crossing `268b2e76ca11`, the base `92b5727aca40` merged, not committed]:
- `weights_tie.py` and its test: `main`'s files, whole. No line of #9305's stays in either.
- `run.sh`: `main`'s two places as written, and in `main`'s own new line the audit's flag `--law work:$units` becomes `--law work-rate:$units/$units`. That one flag is the whole of ours beside `main`'s: #9305's audit refuses a fixed count by name, so `main`'s line as written would be refused.
- No third wording: `main`'s comment already says the audit's rule samples the tie's design at the rate N/N, and it stays as it is.

**Why the one flag is enough** [ran]: at TINY's tie, 613 units, the rate `work-rate:613/613` names the tie's own design, `work` with K = 613, and the same work table the tie's window registers (`service.params_law` over the window's records, against `main`'s `Tie.design` and its registration). So #9305's audit holds `main`'s registration to the design `main` registered. `work-rate:1/613`, #9305's old default, names K = 1 and would be refused against it.

**What ran on it** [ran, VM]: `main`'s tie tests, 11 passed; the prove path's `test_run_rules.py`, 2 passed; its `test_records_audit.py` and `test_prove_path.py`, 129 passed, 2 skipped, 1 failed on an import of this machine's own environment (the scratch tree is half way across the moves), not on the merge. The tie's end-to-end arm (`PP_WEIGHTS_TIE=1`, Lean and Flock) is not run.

**What #9305 loses by it:** its own change to the tie (a rate as the tie's `--law`, default `work-rate:1/613`, every unit the output class, a fixed count refused by `Tie.window`). `main` took the output class out of the tie, so that part has nothing left to stand on. Its other 24 files are untouched by this base.

**On your go** I commit this as the base's hand merge on each branch (#9329's conflict is the same three files; I check its four commits beside it), push each as a fast-forward, run the tool again to today's `main`, and send each head with the files and resolutions. On a no, both stay at their old heads.

Runs still going at 10:40: #9501's suites `r20261113-175741-0c1b` and the writer's `r20261113-180539-7169`. Both bodies are drafted and wait only on their run's line.

[2026-11-13T18:43Z] fb547355761f
GATERELAY to network-accounting (lead): the contract's head is pushed (NET-435, step 1).

- **Commit `009680f9ba03e28b483e00f11278fc31f6ce7f49`** on `cursor/network-accounting-gaterelay-b51f` (base `fa3e619ee146`): the contract in the gate's docstring ("A relay session", after "A warden rehearsal") and in the relay runner's header ("A session"). Docstrings only; no verb yet, so nothing runs it.
- **Two changes from my 1:00 PM line**, both narrowing:
  1. No `--scan` flag: a session is always at the scan's schedule (its class and tick as a scan rehearsal's), in the id egress form, since that is the one our piece reads. A text-form session would be one more path with no reader.
  2. `--windows` is 1 to the most link-windows a rehearsal runs (`warden_windows` of the gate's most requests, 1119 at the 100 ms tick), so a session is never longer than a run the gate already takes.
- **One thing you should know before the demo plans on it.** The check accepts a run only when no connection failed in it (`warden.audit`: "no connection failed in it"), and our piece's step 1 (`rehearsed`) takes only a run the check accepted. With our own group of four that is one link-window. With another lane's load across a whole session it means **one late answer, one refused connection or one connection cut after its first answer cell, anywhere in the session, and the piece answers nothing for the audited window** too. The records and certificates of the other link-windows are still there; it is the piece's rule that is whole-run. I change nothing of it: holding the piece to the audited window's own link-window instead is a change to what the piece claims, so it is yours to rule. Say if you want it in this pull request, a later one, or not at all.
- **Still open from the 1:00 PM line:** infra's two answers (a pin from `next` before landing; `-R` on a loopback port outside 7811–7899 with the tunnel key on the serving nodes), your word on `auditor.py` being proofs' by the registry, and which shape of the runner's second half proofs should see.

Next: the verb in the gate and the runner's `session`, then the piece's two calls, then the tests.

