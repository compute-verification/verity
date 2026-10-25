# network_e2e

Network accounting end to end on a served model, as one command and one report. From a certifier-serve rehearsal (the
network certifier in front of a served endpoint, `benchmarks/network_traces/certifier_serve.py`), it takes:

- N2's verdict of record on every link-window, and the link's network verdict;
- head checks proved with C-Flock `prove --zk` behind the Lean firewall and held by Lean `verify --zk`
  (`benchmarks/network_traces/n2_zk.py`): every honest link-window at both ends, and the egress ends of the tampered
  and extra runs;
- the negatives, each refused: `late`, `extra` and a tampered head (the honest run's window 0 egress head, one count off).

It writes `report.json` and `report.md` under `--out` (`$OUT` under `research run`). `e2e.py` is the driver, and its
docstring has the details. It reuses the rehearsal and N2-ZK's driver and re-implements neither.

## The one command

On a node, proving, from a rehearsal you already have (a directory, a tarball of one, or several, as below):

~~~text
research run --pool research --as LANE --pool-wait 3600 --timeout 5h --project verity --source . --cwd clone \
  --env E2E_MEM_GB=32 --send planned-40.tgz -- \
  bash -c 'bash benchmarks/network_e2e/run.sh --from "$RESEARCH_RUN_DIR/inputs/planned-40.tgz"'
~~~

`run.sh` is what `verity bench network_e2e` runs. It builds C-Flock's `flock-circuit`, the Lean `flock-verify`, the Lean
firewall and network-records (`benchmarks/pouw/hidden_zk/job.sh`'s `setup`, cached on the node), then runs `e2e.py`. On
vy-nebius-cpu-1 a run goes through a check slot (`--pool research`), so `--mem-gb` is dropped there and the slot's
memory applies. `E2E_MEM_GB` is the proving sessions' memory cap. Exit 0 means the report says OK.

N2 alone, without proofs, runs on any machine with `lake` on the PATH (the Lean audit's network-records is built on first
use):

~~~text
uv run --extra torch-cpu verity bench network_e2e --from REHEARSAL --out DIR --no-prove
~~~

When `n2_zk.sh` has already proved these runs, don't prove them again: pass its run's `out/n2zk` (or a tarball of it) as
`--proofs DIR` instead of `--no-prove`. The report adopts those proofs only if every proved unit's public row carries the certificate digest of
the `--from` runs' record for that end and link-window (`e2e.adopt`).

To rehearse first, pass `--upstream URL` (an OpenAI streaming endpoint) or `--vllm MODEL` (vLLM started on this host's
GPU) instead of `--from`. Every option `e2e.py` doesn't take goes to `certifier_serve.py rehearse` (for example
`--requests 40 --profile served`). The rehearsal runs honest, late and extra into `DIR/rehearsal`.

## The chain: varied traffic, the advice, the prove path and the negatives

The plan's target (G-A) is one command from serving to the Lean verdict: serve, certify, advice, compile, sampled ZKPs,
Lean, with the advice's bits per link-window. `chain.py` holds the stages after N2 and the proofs, each on what exists now:

~~~text
verity bench network_e2e --endpoint-peer NAME:PORT --endpoint-run SERVE_DIR --traffic varied --out DIR \
  --strategies honest,late,extra,drop,token --prove-path PROVE_PATH_OUT
verity bench network_e2e (--upstream URL | --endpoint-peer NAME:PORT) --traffic varied --out DIR \
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
  one token shorter, which only N2-ADVICE's counts will refuse. Token is refused only by the I/O tie (IO-TIE). Where
  the tie ran (below), token is a negative it must refuse by content at exactly the link-windows its changed ids fall
  in. Elsewhere it's a row pending the tie, with the verdicts it got and why the tie didn't run. A tamper whose shim
  received no request, or with no `server.json` to say it did, is reported as not run, never as refused.
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
  advice there is no file to hold an arm to: its arms prove their own fixed workload, not this run. Each rank's time
  is reported per drawn unit, its prover's and Lean's seconds over the units its statement proves (PROVE-PATH's
  `per_unit`, 92e4e50ca5).
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
  `requests.jsonl` by engine id. The tie must accept honest (else the report isn't OK) and refuse token by content at
  exactly the link-windows its changed ids fall in (`chain.changed_windows`, from its `server.json`'s `tampered`).
  Drop's refusal by content is listed beside its own code. Without IO-TIE in the tree (`io_tie.py`, #3566), or without
  the ids, the report says why and token stays pending.

The report's `stubbed` list says what each run still stood on that isn't the chain's own: the advice of another run, a
given serving config, a prove path of a declared workload, pending negatives, the certifier off the gate. Before the
committing endpoint, the chain ran on a PROVE-PATH run of a declared workload as its fixture: r20261024-060515-0e35
(its CLI from 01ea29e10f on), with `prove_path.py advice --workload out/standin/workload.json` as `--advice` and its
`out/` as `--prove-path`; before it, r20261024-052710-8932 (GLM-5.3 TP8 B1).

G-A's run of record on the endpoint's own advice: r20261024-111424-88c9 (d5e3b1055d, vy-nebius-2), the varied chain
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

| gate run | strategy | gate | N2 | proved (Lean) |
|---|---|---|---|---|
| c20261024T080200Z-0c53 | honest, planned | accepted | accepted, 11/11 | both ends, every link-window accepted |
| c20261024T080333Z-9cf5 | late | refused | refused at windows 0-1 | not proved (late isn't an `n2_zk` member) |
| c20261024T080506Z-7590 | extra | refused | refused at window 0 | egress, refused at window 0 |
| c20261024T080639Z-3c48 | honest, served | accepted | accepted, 11/11 | not proved (`NetworkServedCounts_v1`) |

The first three are one report, OK, with the tampered head made from 0c53 refused at window 0, proved; 3c48 is its own
report (`out/served`), since a report takes one honest run, and is NOT OK only for having no late or extra run.

The same four against vllm's Llama-3.2-1B on vy-nebius-1:18082 (INF-204, pinned to main 11ce5fe77c, 40 requests, 16
tokens, 11 link-windows), report run r20261024-101818-8c94 (c46c32c2cb, vy-nebius-2), likewise OK, its served report
likewise NOT OK only for having no late or extra run:

| gate run | strategy | gate | N2 | proved (Lean) |
|---|---|---|---|---|
| c20261024T090216Z-e03e | honest, planned | accepted | accepted, 11/11 | both ends, every link-window accepted |
| c20261024T090349Z-a90a | late | refused | refused at window 0 | not proved (late isn't an `n2_zk` member) |
| c20261024T090523Z-eb39 | extra | refused | refused at window 0 | egress, refused at window 0 |
| c20261024T090656Z-0ce6 | honest, served | accepted | accepted, 11/11 | not proved (`NetworkServedCounts_v1`) |

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

Zero-knowledge gaps, from `verity/core/protocols/certifier/README.md` ("Not yet zero-knowledge"):

- N2's verdict of record is still `LinkWindows.verify` on the host, which reads the heads and the certifier's salts. The
  proofs run beside it, on the roots it registered.
- The proofs are of `n2_zk`'s row adapter, not of N2's registered unit, so `LinkWindows.proved` (which refuses a proof
  of any other Program, NET-43) doesn't take them. Lean holds each proof to the program and partition that `n2_zk` wrote.
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
