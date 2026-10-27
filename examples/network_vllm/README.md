# network_vllm

Network accounting end to end on a served model, as one command and one report. From a certifier-serve rehearsal (the
network certifier in front of a served endpoint, `benchmarks/network_traces/certifier_serve.py`), it takes:

- N2's verdict of record on every link-window, and the link's network verdict;
- head checks proved with C-Flock `prove --zk` behind the Lean firewall and held by Lean `verify --zk`
  (`benchmarks/network_traces/n2_zk.py`): every honest link-window at both ends, and the egress ends of the tampered
  and extra runs;
- the negatives, each refused: `late`, `extra` and a tampered head (the honest run's window 0 egress head, one count off).

It writes `report.json` and `report.md` under `--out` (`$OUT` under `research run`). `e2e.py` is the driver, and its
docstring has the details. It reuses the rehearsal and N2-ZK's driver and re-implements neither.

## The example

- **Protocol:** the network certifier (`verity/core/protocols/certifier`), its head checks proved by the proof service.
- **Workload:** `integrations/vllm`, a model vLLM serves behind `verity-vllm serve-openai`, with the certifier in front.
- **Placement** (`example.toml`), as in the runs of record below: the endpoint on a GPU node (vy-nebius-1), the
  certifier on the auditor's host (vy-challenger-1's gate), and the prover and the Lean verifier on a CPU node
  (vy-nebius-cpu-1's network slot).
- **Command:** `verity example network_vllm ARGS`, which runs `run.sh` (below).
- **Owner:** network-accounting.

## The one command

On a node, proving, from a rehearsal you already have (a directory, a tarball of one, or several, as below):

~~~text
research run --pool research --as LANE --pool-wait 3600 --timeout 5h --project verity --source . --cwd clone \
  --env E2E_MEM_GB=32 --send planned-40.tgz -- \
  bash -c 'bash examples/network_vllm/run.sh --from "$RESEARCH_RUN_DIR/inputs/planned-40.tgz"'
~~~

`run.sh` is what `verity example network_vllm` runs. It builds C-Flock's `flock-circuit`, the Lean `flock-verify`, the Lean
firewall and network-records (`benchmarks/pouw/hidden_zk/job.sh`'s `setup`, cached on the node), then runs `e2e.py`. On
vy-nebius-cpu-1 a run goes through a check slot (`--pool research`), so `--mem-gb` is dropped there and the slot's
memory applies. `E2E_MEM_GB` is the proving sessions' memory cap. Exit 0 means the report says OK.

N2 alone, without proofs, runs on any machine with `lake` on the PATH (the Lean audit's network-records is built on first
use):

~~~text
uv run --extra torch-cpu verity example network_vllm --from REHEARSAL --out DIR --no-prove
~~~

When `n2_zk.sh` has already proved these runs, don't prove them again: pass its run's `out/n2zk` (or a tarball of it) as
`--proofs DIR` instead of `--no-prove`. The report adopts those proofs only if every proved unit's public row carries the certificate digest of
the `--from` runs' record for that end and link-window (`e2e.adopt`).

To rehearse first, pass `--upstream URL` (an OpenAI streaming endpoint) or `--vllm MODEL` (vLLM started on this host's
GPU) instead of `--from`. Every option `e2e.py` doesn't take goes to `certifier_serve.py rehearse` (for example
`--requests 40 --profile served`). The rehearsal runs honest, late and extra into `DIR/rehearsal`.

## The chain: varied traffic, the advice, the prove path and the negatives

**G-A's live report of record: r20261024-201847-5246.** It is OK with nothing pending, and every row is from vllm's
live Llama-3.2-1B serving run r20261024-191722-ddad, driven while it served: the live run r20261024-192612-4460
(honest, late, extra, drop and token, `--probes 40`, on vy-nebius-cpu-1's network slot n3, against vy-nebius-1:18084
through the nodes' ssh forward). It re-makes 4460's report from 4460's outputs on 4460's tree with this PR merged
(42bd54226a), its prove path the `--serve-run` audits of two of the honest run's own Commit windows.
r20261024-205806-6408 rebuilt it from the same records at `next` (250fa54c2b), and every verdict is 5246's. N2's
advised verifier refuses this run's honest traffic: vLLM admitted two prefill pairs out of certified release order
(VLLM-89); a release-order rerun follows. 5246 lists N2's verdicts as not run, so its OK claims nothing about them.

- e2e: honest accepted (40 requests, 597 tokens, 38 ending at max_tokens and 2 at EOS); late, extra, drop and token
  refused by their own codes. The advice is ddad's Commit windows w37-w66: 30 windows, 40 requests, 528 steps, every
  window opened, bound and rebuilt; 23,800 v0 bits (3,953 derived, 2,406 shaped, 332 forced).
- I/O tie: honest accepted (597 outputs placed, frames held on all 17 windows at both ends); late, drop and token
  refused; extra not applicable, refused by its own code.
- prove path: r20261024-201458-07b3's `--serve-run` audits of ddad's w46 and w54, each on vllm's own registration of
  the window (its advice loop, `verity-vllm advice --window K`), adopted in place of the fixed workload. Each honest
  arm was accepted (2 of 4 units drawn) and each forged arm refused by Lean; each arm's root is the registered one,
  names the endpoint's Commit of its window and opens to the report's advice of it (`chain.held_to`). The
  registrations' file times, kept to the nanosecond by the lane's copy of ddad
  (art:17021c17419213ef07eb9de8ac3af3c71187f9a8fa2fccafbac47cf523b8baad), put w46 3.5 s after its window closed and
  2.3 s before the next closed, and w54 2.8 s after and 0.4 s before, both before their draws: at serve time. ddad's
  own record stamps them to the second (its `out/advice_windows.jsonl`, `record` below): w46 then still reads at
  serve time (no later than 3.9 s after its close, at least 1.9 s before the next), and w54 unknown, since its stamp
  19:38:16Z puts it up to 0.3 s after the next window's close at 19:38:16.746Z.

**The earlier report of record: r20261024-163242-7e60**, from vllm's sealed Llama-3.2-1B serving run
r20261024-141912-8eb1, rebuilt as r20261024-190804-7e4c, OK with nothing pending. 7e4c re-makes GA-CLOSER's closer
r20261024-150521-0551 (ded16ce22b, `--probes 40`, against vy-nebius-1:18084) from that run's outputs, on #3841's tree
(ac3067386b) with #3566's `io_tie` merged, as 7e60 did on #3778's tree before the report held its prove path's advice
by content. It reads 8eb1's records, whose sha256 equal 0551's copy, from a copy of the inputs on the node that holds
them (r20261024-190732-a08f, each file checked against its sha256 there), and drives no endpoint.

- e2e: honest accepted (40 requests, 597 tokens, 38 ending at max_tokens and 2 at EOS); late, extra and drop refused
  by their own codes. The advice is 8eb1's Commit windows w191-w220: 30 windows, 40 requests, 527 steps, every window
  opened, bound and rebuilt; 23,690 v0 bits (3,851 derived, 2,412 shaped, 340 forced).
- token: refused by the I/O tie, `IoTieWindow.content` at link-windows 1 and 2, the two its 23 changed ids fall in.
  The tie accepted honest: 597 outputs placed, frames held on all 17 windows at both ends. NEG's `tie-run` on each
  member, with the ids of 8eb1's requests that arrived in its link-windows, agrees (the run's `out/tie-run/`). Under
  the coverage that holds each request's ids to its served outputs (`chain.uncovered`, REDTEAM phase 14), those ids
  give every honest and token request an id for each output it served (r20261024-183733-92e3), so their verdicts
  stand. Extra's one request served past its max_tokens has an output with no sampled id, so the tie now gives extra
  no verdict; its own code refuses it.
- prove path: GA-AUDIT's `--serve-run` audits of 8eb1's w212 (r20261024-154338-7679), one of the honest run's own
  Commit windows, and w5 (r20261024-143655-ccc8), adopted in place of the fixed workload. Every honest arm was accepted
  and every forged arm refused by Lean. Each arm's committed root is the one registered for its window, the
  registration names the endpoint's Commit of that window, and its opening opens the root to the report's advice of
  the window (`chain.held_to`), so the report's OK holds both arms by content (7e60's held them by v0 bit count, with
  the content check apart in its `out/advice-check.json`). Read off the registrations' own times, both were
  registered after serving: w212's by the lane from the run's records (r20261024-153328-5776), 25.1 min after its
  Commit window closed and 14.2 min before the first draw, and w5's by vllm during the run, 11.3 s after its Commit
  window closed and 8.1 s after the next one (w6) closed, which 7e60's job stated as at serve time
  (r20261024-173746-1bb0, on the files in place; 7e4c's copies kept their times to the second, so it reads 25.0 min
  and 10.4 s). Both are after serving under the rule below as well, since each was registered after the next window
  closed.

The plan's target (G-A) is one command from serving to the Lean verdict: serve, certify, advice, compile, sampled ZKPs,
Lean, with the advice's bits per link-window. `chain.py` holds the stages after N2 and the proofs, each on what exists now:

~~~text
verity example network_vllm --endpoint-peer NAME:PORT --endpoint-run SERVE_DIR --traffic varied --out DIR \
  --strategies honest,late,extra,drop,token --prove-path PROVE_PATH_OUT
verity example network_vllm (--upstream URL | --endpoint-peer NAME:PORT) --traffic varied --out DIR \
  --strategies honest,late,extra,drop,token --advice ADVICE --serving-config CONFIG.json --prove-path PROVE_PATH_OUT
~~~

- `--traffic varied` is certifier-serve's (CERT-VARLEN's `rehearse`: Poisson arrivals, prompts as token ids, real EOS),
  passed on. Its plan's N2 profile is `advice`, on which N2 gives no verdict until N2-ADVICE derives the egress counts from
  the ingress head and the advice. So nothing is proved by `n2_zk`, and the rehearsal's own check (every record certified,
  the certified I/O, the audit, the salt) is the verdict, with late, extra and drop as its negatives (`CLAIM_ADVICE`).
- `--endpoint-peer NAME:PORT` reaches node NAME's endpoint on its 127.0.0.1:PORT through an ssh forward (`/etc/vy/peers`),
  so the certifier runs off the prover's node: better than the loopback, though not yet the auditor host.
- NEG's live tampers (`--strategies ...,drop,token`): drop is a negative, refused by N2's counts on the planned and
  served profiles and by the rehearsal's own check on the advice profile. There the certified I/O's finish rule refuses
  it when the response it cuts ran to its max_tokens (short of it without EOS); one that ended at EOS stays well formed
  one token shorter, which only N2-ADVICE's counts will refuse. Token is refused only by the I/O tie (IO-TIE). On the
  advice profile, where the tie is required (below), token is a negative it must refuse by content at exactly the
  link-windows its changed ids fall in, and the tie's refusal is its verdict whatever the tie gave the honest run. On
  the planned and served profiles, where the tie doesn't run, it's a row pending the tie, with the verdicts it got. A
  tamper whose shim received no request, or with no `server.json` to say it did, is reported as not run, never as
  refused.
- On the advice profile a negative is refused only when the rehearsal's own check names its own violation
  (`e2e.OWN_CODE`, as NEG's `judge` requires its code): late's window's coverage fault
  (`audit.status:coverage-never-established`), extra's response longer than its max_tokens, drop's short of it without
  EOS. A refusal for any other reason (a stalled forward's coverage fault on a drop run, say) is reported with the codes
  it did name, and isn't the negative's.
- `--advice FILE`: the advice, as I1 v0's JSON or `encode` bytes, or a prove-path stand-in's from before v0 (its first
  runs'), which is upgraded with each kept output released in its step's bucket. The report gives v0's bits per
  link-window (`advice.bits_by_window`, T the run's egress window). On a varied run the honest run's certified I/O
  (`io.json`) binds the advice to the run (`bits.bound`). With `--serving-config` (`derived.Config`'s fields, or a step
  trace's `config`) and the bound I/O, whose tokens the residual reads (canonical sources, derived hits), it also gives
  ENC-DERIVED's derived bits, unshaped and shaped, and the residual must rebuild the whole advice (`bits.rebuilt`). Where an advice is given, the OK needs both to be true. The serving config is the one the job
  was given, not one the endpoint registered, so the derived bits and the residual's own check under it
  (`public_fields`) stay out of the OK.
- `--endpoint-run SERVE_DIR`: the endpoint's own advice, in place of `--advice`. SERVE_DIR is a `verity-vllm
  serve-openai` run's `out/serve` on the `--endpoint-peer` node (or a local copy). The endpoint records each request's
  `user`, which the shim forwards on a varied plan. After the rehearsal its records (`chain.ENDPOINT`: the scheduler
  trace `advice_trace.jsonl` with each Commit window's open and close, `requests.jsonl`, `windows.jsonl`, `serve.json`)
  are copied into `DIR/endpoint`. `verity-vllm advice` (vllm's #3619; until it lands, its `verity_vllm` laid over the
  checkout, as its `research_tools` looks for the workspace above its own file) then writes one I1 v0 advice for each
  closed Commit window: a public registration naming the window's Commit, with the
  advice's hiding root, and the advice and its opening under `private/`. The honest run's Commit windows are those
  whose first step falls in its link-windows. The advice is bound to the run (`bits.bound`) when each of those windows'
  requests is a certified one, every certified request is in exactly one of them, and each window's opening opens its
  registered root, with its registration naming the window's Commit. Each window's advice must be rebuilt from its
  certified I/O (`bits.rebuilt`) under the config read off the endpoint's records. That config is outside the OK, as a
  given one is. The report lists each Commit window's bits and places them in the run's link-windows from its first
  step's bucket. The rehearsal opens the advice to the report, where only the registrations are public.
- `--prove-path DIR`: a `prove_path.py` run's output (PROVE-PATH's `run.sh` `$OUT`, or a tarball of it), adopted. The
  honest arm must pass every rank, and the forged arm must have Lean refuse its unit. The advice each arm committed is
  held to the `--advice` file (`committed_is_the_advice_file`): by the file's SHA-512 where it committed that, and by its
  bits only where it committed a hiding root (from 01ea29e10f on, whose opening is the prover's). That isn't these runs':
  the prove path is `bound` to them only once the file is their advice and the Commit it proves is held to theirs, which
  no input carries yet (`bound_why`), and only a bound prove path's verdict counts in the OK. Beside the endpoint's own
  advice there is no file to hold an arm to: a fixed workload's arms prove their own Commit, not this run's. An arm of
  `prove_path.py audit --serve-run` names the served run's Commit window it proves (`chain.served_commit`, the arm named
  `w<k> <arm>`), and `--prove-path` may be given once for each such audit. The prove path is then bound to these runs
  when every arm's Commit is the endpoint's (its `windows.jsonl` has that window's run id and root), its root the one
  sealed while serving (the audit's `serve` check), and an honest arm's Commit is one of the honest run's own windows.
  Every arm's committed advice must be the report's advice of its window by content (`chain.held_to`). Its root is the
  one registered under that window's Commit: in the report's own advice, or, registered apart under its own salt and
  compiler, in the `verity-vllm advice` directory `--prove-path-registered` names. That registration's opening must
  open the root to the report's advice of the window (`chain.opens`). The v0 bit count is never enough, and an arm not
  held so makes the report NOT OK, whether its prove path binds or not. `--prove-path-registered FILE` says, by window,
  who registered each audited window's advice, where when it isn't the report's own, the serving run's own record (its
  `research run` directory) and the arm's first draw (`{K: {by, registration, record, first_draw, ref}}`). When it was
  registered is never the job's to say. It is read off the serving run's record when `record` is given and its advice
  loop's `out/advice_windows.jsonl` stamps that window's registered root (UTC to the second, written once the
  registration is in place, so the registration was no later than a second past the stamp, `chain.serve_record`), and
  otherwise off the registration's file time (a registration has no time field), always against the endpoint's trace
  (each window's close is the first clocked record after its close event, `chain.closes`). A registration was at
  serve time when it was registered before its window's draw (every rank's `advice_before_draw`, the verifier's
  receipt of it before any coin) and no later than the next window's close (the lead's ruling, 24 Oct). The last
  window in the endpoint's records has no next close there. It is bounded by the session's end (the record's
  `status.json` `ended_utc`) when the record says the session ended with that window the last its loop logged.
  A registration reads "after serving" when its draw came first or when both its record's time and its file's are
  past its bound, and unknown otherwise: no receipt, no bound, or a stamp whose second straddles the bound while the
  file's time is before it. The report always prints the lag in seconds after its own window's close, and `stubbed`
  lists every window not read as at serve time. Each rank's time is reported per drawn
  unit, its prover's and Lean's seconds over
  the units its statement proves (PROVE-PATH's `per_unit`, 92e4e50ca5).
- NEG's registry (`benchmarks/network_traces/negatives.py`'s `suite`) runs on the runs unless `--no-negatives`. Each row
  that ran must be refused by the code it names; the rest say what they wait on. It doesn't run on the advice profile.
  With `--advice` and `--ids FILE` (each request's sampled ids by its advice id, ADVICE-SRC's `requests.jsonl`: `user`,
  `output_ids`), each negative the I/O tie reads also carries the tie's verdict (IO-TIE's reference in the clear, on a
  fake upstream's cells and Commit port of the advice and those ids), and its row passes only when the tie refuses it by the
  part it names and accepts the honest windows. Until IO-TIE lands, those rows say so.
- The I/O tie on the runs (`chain.tie`, advice profile only, where NEG's registry doesn't run): NEG's `tie_run` on every
  run, IO-TIE's reference in the clear on its stand-ins. Each window's certified frames are held to their head and
  committed under fake salts in place of the warden's cells. The sampled ids stand in for the Commit's port, one map
  for every run from `--ids`, or else from the `--endpoint-run` records (`chain.served_ids`). Every strategy sends the
  same users, so a run's ids are those of the requests whose `arrive` falls in its own link-windows, matched to
  `requests.jsonl` by engine id. The ids must cover each run's certified requests, an id for each output its `io.json`
  says a request served (`chain.uncovered`, REDTEAM phase 14), or the run gets no verdict. The tie must refuse token
  by content at exactly the link-windows its
  changed ids fall in (`chain.changed_windows`, from its `server.json`'s `tampered`). Drop's refusal by content is
  listed beside its own code. The tie is the only check of the served content on the advice profile, so it's a required
  gate there (`chain.tie_gate`, REDTEAM phase 12 target 2): the report isn't OK, and says why, when the tie didn't run
  (no IO-TIE in the tree, `io_tie.py` in #3566, or no ids), gave the honest run no verdict (its tie raised, or its ids
  don't cover its requests) or didn't accept it. A part the tie must establish beside its verdict goes in
  `chain.TIE_REQUIRED` and fails the report the same way: CERT-VARLEN's compact finish (REDTEAM phase 12 target 3), if
  the tie reports it as a key of its own.

The report's `stubbed` list says what each run still stood on that isn't the chain's own: the advice of another run, a
given serving config, a prove path of a declared workload, pending negatives, the certifier off the gate. Before the
committing endpoint, the chain ran on a PROVE-PATH run of a declared workload as its fixture: r20261024-060515-0e35
(its CLI from 01ea29e10f on), with `prove_path.py advice --workload out/standin/workload.json` as `--advice` and its
`out/` as `--prove-path`; before it, r20261024-052710-8932 (GLM-5.3 TP8 B1).

Before r20261024-163242-7e60, G-A's run of record on the endpoint's own advice was r20261024-111424-88c9 (d5e3b1055d, vy-nebius-2), the varied chain
against vllm's Llama-3.2-1B serving run r20261024-094558-6d6a on vy-nebius-1:18083 (40 requests at 0.5/s, max_tokens
up to 32, EOS 128001, 16 link-windows), report OK. honest accepted (597 tokens, 38 ending at max_tokens and 2 at EOS);
late, extra and drop refused by their own codes. Token, which the rehearsal's check accepted, is refused by the I/O tie
in r20261024-142953-359c: NEG's `tie-run` on each member, with the sampled ids of 6d6a's requests that arrived in that
member's link-windows, and #3566's `io_tie` merged into the tree for that run. The tie refused token by
`IoTieWindow.content` at link-windows 0 and 1, the two its 23 changed ids fall in, and accepted honest (597 outputs
placed, frames held on all 16 windows at both ends). The report re-made there with the tie folded in is OK, with token
a negative. 88c9's advice is 6d6a's Commit windows
w345-w375: 31 windows, 40 requests, 532 steps, every window opened, bound and rebuilt; 23,972 v0 bits, 4,021 derived
(2,496 shaped, 334 forced). Its prove path is PROVE-PATH's Llama TP1 fixed workload (r20261024-072909-b53f's honest arm,
r20261024-090358-218a's forged), labelled as such: 6d6a's Commit windows have no sealed replay bundle yet.

## Inputs: rehearsals and gate runs

Each `--from` source is one of these, or a gzipped tar of one:

- A rehearsal's directory (`certifier_serve.py rehearse --out`): `summary.json` and one run directory per strategy.
- One strategy's run directory: its `certified.jsonl`, `plan.json`, `calibration.json` and `deployment.json`.
- A gate run. On vy-challenger-1 the gate runs the rehearsal as the auditor's device:
  `vy-challenger-gate certifier-rehearse --peer NAME --port PORT --minutes M [--tamper late|extra] [--profile planned|served]`.
  `vy-challenger-gate fetch RUN` returns the run as a gzipped tar (`request.json`, `out/`, `end.json`). One gate run is one
  strategy, so fetch the honest run and one run per tamper, and pass them all:
  `--from honest.tgz late.tgz extra.tgz`. Reach the gate with your lane's gate key, as
  `benchmarks/remote_session/auditor.sh` does, or with the shared lane key, which reads only (`status`, `fetch`):
  `ssh -o IdentitiesOnly=yes -i KEY research@HOST fetch RUN > RUN.tgz`. `status RUN` gives its `end.json`, whose SHA-512
  of each file of `out/` the tarball must match. The report's source line names the gate's host, each gate run and the
  main commit the gate was pinned to.

The auditor-host runs of record: infra's gate rehearsals on vy-challenger-1 (INF-199), pinned to main 11ce5fe77c,
against vllm's Llama-3.1-8B INT8 on vy-nebius-1:18080 (10 minutes, 40 requests, 16 tokens, 11 link-windows), each
tarball's 21 files matching its `end.json`. Report run r20261024-081948-55d8 (a3e7433969, vy-nebius-cpu-1's network slot n1):

| gate run | strategy | gate | N2 | proved (Lean) | decided by proof (N2-ZK) |
|---|---|---|---|---|---|
| c20261024T080200Z-0c53 | honest, planned | accepted | accepted, 11/11 | both ends, every link-window accepted | accepted, 11/11 (5300) |
| c20261024T080333Z-9cf5 | late | refused | refused at windows 0-1 | not proved (late isn't an `n2_zk` member) | refused at windows 0-1 (5300) |
| c20261024T080506Z-7590 | extra | refused | refused at window 0 | egress, refused at window 0 | refused at window 0 (5300) |
| c20261024T080639Z-3c48 | honest, served | accepted | accepted, 11/11 | not proved (`NetworkServedCounts_v1`) | accepted, 11/11; its tampered head refused at window 0 (62bd) |

The first three are one report, OK, with the tampered head made from 0c53 refused at window 0, proved; 3c48 is its own
report (`out/served`), since a report takes one honest run, and is NOT OK only for having no late or extra run.

The same four against vllm's Llama-3.2-1B on vy-nebius-1:18082 (INF-204, pinned to main 11ce5fe77c, 40 requests, 16
tokens, 11 link-windows), report run r20261024-101818-8c94 (c46c32c2cb, vy-nebius-2), likewise OK, its served report
likewise NOT OK only for having no late or extra run:

| gate run | strategy | gate | N2 | proved (Lean) | decided by proof (N2-ZK) |
|---|---|---|---|---|---|
| c20261024T090216Z-e03e | honest, planned | accepted | accepted, 11/11 | both ends, every link-window accepted | accepted, 11/11 (9e19) |
| c20261024T090349Z-a90a | late | refused | refused at window 0 | not proved (late isn't an `n2_zk` member) | refused at window 0 (1a5b) |
| c20261024T090523Z-eb39 | extra | refused | refused at window 0 | egress, refused at window 0 | refused at window 0 (9e19) |
| c20261024T090656Z-0ce6 | honest, served | accepted | accepted, 11/11 | not proved (`NetworkServedCounts_v1`) | accepted, 11/11; its tampered head refused at window 0 (fc11) |

"Decided by proof" is N2-ZK's runs of #3597's registered units (`NetworkLinkWindow_v5` planned,
`NetworkServedLinkWindow_v4` served) on the same gate runs: each link-window's verdict is the one the verifier's own
Lean `verify --zk` accepted for the registered Program, with the certificates checked in the clear. 8B:
r20261024-091126-5300 (planned), r20261024-091913-62bd (served). 1B: r20261024-093930-1a5b (planned; its late), its
rerun r20261024-110735-9e19 (honest and extra), r20261024-093950-fc11 (served). The report reads them once #3597
lands, whose per-round summary `e2e` then adopts in place of the row adapter's.

An honest run is required, because the tampered head is made from it. Two runs of the same strategy are refused.

## The report

`report.json` (`verity-network-e2e/report/v3`) holds:

- per run: the rehearsal's own check, the gate's `end.json` verdict, N2's network verdict and its refused link-windows,
  and, per proved end, whether Lean held it and its proving session's cost (below);
- per link-window: N2's verdict, each end's proved verdict (accept or refuse, from the verdict row the verifier opens),
  and the head-check parts that fail (a diagnostic, evaluated in the clear on the rehearsal's witness);
- per negative: refused or not, and by what (N2, the rehearsal's check, the proved head checks);
- one link-window proved alone (`n2_zk`'s cost pass), and the claim (below);
- the chain's stages that ran (`chain`: the advice, its bits per link-window, the prove path, NEG's rows), and `stubbed`.

`report.md` renders the same thing as tables. OK means: the honest run was accepted on every link-window by N2 and, where
proofs ran, by every proved head check held by Lean; each negative was refused by everything that checked it; a given
advice is these runs' and rebuilt from their certified I/O; a bound prove path and NEG's registry, where they ran,
passed. On the advice profile the rehearsal's own check stands in for N2, and a negative counts only for its own
violation. `report.md`'s title says what the OK rests on (`ran`) and what didn't run (`not_run`), so an OK on the
self-check alone reads as that. When it isn't OK, `why_not` (and the line under the title) says which of these failed, a
missing negative's run among them.

## What it proves, and what it doesn't

The claim is in `e2e.py` (`CLAIM`) and in both files, word for word. **It covers counts only.** For the rehearsal's one
link, every honest link-window's per-bucket frame counts match the planned occupancy profile (on a served link, the counts
its ingress head derives). That fixes the link's timing to the profile. Late, extra and a tampered head are each refused.

It does not cover:

- **N3, the content.** The bytes and order of the frames inside the counts aren't checked.
- **The certificates' LMS check and the audit of each record.** Both stay on the host. The proofs cover each end's head
  check only.
- **A served link's count check, `NetworkServedCounts_v1`.** It has no prover, so a served run gets N2's verdict of
  record and no proof.
- **Late's head checks.** Late isn't one of `n2_zk`'s members, so it is refused by N2 alone.
- **A real link, or an auditor's host.** The rehearsal runs over loopback, its check plays the firewall and the
  challenger in one process, and it publishes witness (the records and the certifier's salts) so that its check can be
  replayed. Off the gate the certifier is a process on whatever host ran the rehearsal. Against an endpoint on that
  host's loopback, that host is the prover's node, not an auditor host; the report's `where` says which. `n2_zk` runs the
  prover, the firewall and the verifier on one node.
- **The model's size.** The report names the model as the plan does. The plan doesn't say whether the model was cut:
  GLM-M4's endpoint (r20261024-021333-0ca6) is GLM-5.3 INT8 cut to 5 of 78 layers, at 16 tokens a response.

Zero-knowledge gaps, from `verity/core/protocols/certifier/README.md` ("Zero knowledge: what N2's verdict reads, and
what's open"):

- N2's verdicts here are decided in the clear (`LinkWindows.unproved_outcome`, as for any run that proves nothing),
  which reads the heads and the certifier's salts. `n2_zk` now decides each link-window on a `--zk` proof of N2's
  registered unit, one member a round; this report doesn't read that summary yet, so it proves nothing itself and
  refuses to adopt such a run (``--no-prove`` takes N2's verdicts only).
- The proofs an adopted earlier `n2_zk` summary holds are of its row adapter, not of N2's registered unit, so
  `LinkWindows.proved` (which refuses a proof of any other Program, NET-43) doesn't take them. Lean holds each proof to
  the program and partition that `n2_zk` wrote.
- NET-42: the certificate's link to its head is checked on the host (`CertifiedRecord.verify`).

## Cost

Each end of a link-window is one unit of its member's statement. All of an end's link-windows are proved in one session,
one statement and one pair of proofs (`circuit.rep0.bin`, `circuit.rep1.bin`), and Lean verifies them at once, so a
link-window has no proving time or proof of its own. The report gives each session's prover seconds, Lean verify seconds,
proof bytes and peak memory, undivided. Peak memory is the largest process the driver had run by the end of the session
(`RUSAGE_CHILDREN`, which only grows). `n2_zk` proves the honest members first, so the tampered and extra sessions' peaks
include theirs. Beside the sessions is `n2_zk`'s cost pass: a session on the honest statement whose verifier draws one
link-window, per end, which is what one link-window costs proved alone. The driver's own peak (the unit's lowering)
is shown when it proved.
