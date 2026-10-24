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

## Inputs: rehearsals and gate runs

Each `--from` source is one of these, or a gzipped tar of one:

- A rehearsal's directory (`certifier_serve.py rehearse --out`): `summary.json` and one run directory per strategy.
- One strategy's run directory: its `certified.jsonl`, `plan.json`, `calibration.json` and `deployment.json`.
- A gate run. On vy-challenger-1 the gate runs the rehearsal as the auditor's device:
  `vy-challenger-gate certifier-rehearse --peer NAME --port PORT --minutes M [--tamper late|extra] [--profile planned|served]`.
  `vy-challenger-gate fetch RUN` returns the run as a gzipped tar (`request.json`, `out/`, `end.json`). One gate run is one
  strategy, so fetch the honest run and one run per tamper, and pass them all:
  `--from honest.tgz late.tgz extra.tgz`. Reach the gate with your lane's gate key, as
  `benchmarks/remote_session/auditor.sh` does: `ssh -o IdentitiesOnly=yes -i KEY research@HOST fetch RUN > RUN.tgz`.

An honest run is required, because the tampered head is made from it. Two runs of the same strategy are refused.

## The report

`report.json` (`verity-network-e2e/report/v1`) holds:

- per run: the rehearsal's own check, the gate's `end.json` verdict, N2's network verdict and its refused link-windows,
  and whether each proved end was held by Lean;
- per link-window: N2's verdict, each end's proved verdict (accept or refuse, from the verdict row the verifier opens),
  the head-check parts that fail (a diagnostic, evaluated in the clear on the rehearsal's witness), and the prove seconds,
  Lean verify seconds, proof bytes and peak memory;
- per negative: refused or not, and by what (N2, the rehearsal's check, the proved head checks);
- the cost per link-window, and the claim (below).

`report.md` renders the same thing as tables. OK means: the honest run was accepted on every link-window by N2 and, where
proofs ran, by every proved head check held by Lean; and each negative was refused by everything that checked it.

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
- NET-43: N2's unit binds no circuit digest. Here Lean holds each proof to the program and partition that `n2_zk` wrote.
- NET-42: the certificate's link to its head is checked on the host (`CertifiedRecord.verify`).

## Cost

Each end of a link-window is one unit of its member's statement. All of an end's link-windows are proved in one session,
so the report divides each session's prover seconds, Lean verify seconds and proof bytes by its units. Peak memory is
the largest process the driver had run by the end of the session (`RUSAGE_CHILDREN`, which only grows). `n2_zk` proves the
honest members first, so the cost, read from the honest run, is theirs. The driver's own peak (the unit's lowering) is
shown beside it.
