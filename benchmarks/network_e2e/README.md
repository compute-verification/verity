# network_e2e

Network accounting end to end on a served model, as one command and one report. From a certifier-serve rehearsal (the
network certifier in front of a served endpoint, `benchmarks/network_traces/certifier_serve.py`), it takes:

- N2's verdict of record on every sampling block, and the link's network verdict;
- head checks proved with C-Flock `prove --zk` behind the Lean firewall and held by Lean `verify --zk`
  (`benchmarks/network_traces/n2_zk.py`): every honest sampling block at both ends, and the egress ends of the tampered
  and extra runs;
- the negatives, each refused: `late`, `extra` and a tampered head (the honest run's sampling block 0 egress head, one
  count off);
- the chain's steps after N2 and the proofs (`chain.py`): the advice and its bits per sampling block, the prove path's
  audit, the registry of negatives and, on the advice profile, the I/O tie.

`e2e.py` is the driver and `chain.py` holds the chain's steps; their docstrings have each option and each check. It
reuses the rehearsal and `n2_zk.py`'s driver and re-implements neither. `encodings/` measures the advice's encodings, and
`traces/` records the vLLM scheduler traces they are measured on (`traces/README.md`).

## The one command

`verity bench network_e2e ARGS` runs `run.sh`, ARGS being `e2e.py`'s. Proving runs as a job on a CPU node, whose command
and resources are in `run.sh`'s header: it builds C-Flock's `flock-circuit`, the Lean `flock-verify`, the Lean firewall
and network-records (`benchmarks/pouw/hidden_zk/job.sh`'s `setup`, cached on the node), then runs `e2e.py` with the
report in `$OUT`. `E2E_MEM_GB` is the proving runs' memory cap. Exit 0 means the report says OK. The job publishes the
report with the run's outputs in the evidence store, where results are read.

N2 alone, without proofs, runs on any machine with `lake` on the PATH (the Lean audit's network-records is built on first
use):

~~~text
uv run --extra torch-cpu verity bench network_e2e --from REHEARSAL --out DIR --no-prove
~~~

When `n2_zk.sh` has already proved these runs, don't prove them again: pass its run's `out/n2zk` (or a tarball of it) as
`--proofs DIR` instead of `--no-prove`. The report adopts those proofs only if every proved proof block's public row
carries the certificate digest of the `--from` runs' record for that end and sampling block (`e2e.adopt`).

To rehearse first, pass `--upstream URL` (an OpenAI streaming endpoint), `--vllm MODEL` (vLLM started on this host's
GPU) or `--endpoint-peer NAME:PORT` (node NAME's endpoint on its loopback, through an ssh forward) instead of `--from`.
Every option `e2e.py` doesn't take goes to `certifier_serve.py rehearse` (for example `--requests 40 --profile served`).
The rehearsal runs honest, late and extra into `DIR/rehearsal`.

Over varied traffic, with the endpoint's own advice and the prove path:

~~~text
verity bench network_e2e --endpoint-peer NAME:PORT --endpoint-run SERVE_DIR --traffic varied --out DIR \
  --strategies honest,late,extra,drop,token --prove-path PROVE_PATH_OUT
verity bench network_e2e (--upstream URL | --endpoint-peer NAME:PORT) --traffic varied --out DIR \
  --strategies honest,late,extra,drop,token --advice ADVICE --serving-config CONFIG.json --prove-path PROVE_PATH_OUT
~~~

- `--traffic varied` is `certifier_serve.py rehearse`'s (Poisson arrivals, prompts as token ids, real EOS). Its plan's N2
  profile is `advice`, on which N2 gives no verdict, so nothing is proved and the rehearsal's own check is each run's
  verdict, with late, extra and drop as its negatives (`e2e.CLAIM_ADVICE`). There a negative is refused only when the
  check names its own violation (`e2e.OWN_CODE`).
- `--endpoint-run SERVE_DIR` is a `verity-vllm serve-openai` run's `out/serve`. Its records (`chain.ENDPOINT`) are
  copied into `DIR/endpoint`, and `verity-vllm advice` writes one I1 v0 advice for each closed Commit window, bound to
  the honest run and rebuilt from its certified I/O (`chain.served_advice`). `--advice FILE` (with `--serving-config`)
  gives an advice instead.
- `--prove-path DIR` adopts a `prove_path.py audit` run's output. Each arm's committed advice must be the report's
  advice of its Commit window by content (`chain.held_to`), and only a prove path bound to these runs counts in the OK.
  `--prove-path-registered FILE` says who registered each audited Commit window's advice and where its serving record
  is; when it was registered is read off the record and the endpoint's trace (`chain.serve_record`, `chain.closes`).
- The registry of negatives (`benchmarks/network_traces/negatives.py`'s `suite`) runs on the runs unless
  `--no-negatives`; each row that ran must be refused by the code it names. On the advice profile the I/O tie
  (`chain.tie`) runs instead and is required (`chain.tie_gate`).

The report's `stubbed` list says what each run still stands on that isn't the chain's own: the advice of another run, a
given serving config, a prove path of a declared workload, pending negatives, the certifier off the gate.

## Inputs: rehearsals and gate runs

Each `--from` source is one of these, or a gzipped tar of one:

- A rehearsal's directory (`certifier_serve.py rehearse --out`): `summary.json` and one run directory per strategy.
- One strategy's run directory: its `certified.jsonl`, `plan.json`, `calibration.json` and `deployment.json`.
- A gate run. On the auditor's machine the gate runs the rehearsal as the auditor's device:
  `vy-challenger-gate certifier-rehearse --peer NAME --port PORT --minutes M [--tamper late|extra] [--profile planned|served]`.
  `vy-challenger-gate fetch RUN` returns the run as a gzipped tar (`request.json`, `out/`, `end.json`). One gate run is one
  strategy, so fetch the honest run and one run per tamper, and pass them all:
  `--from honest.tgz late.tgz extra.tgz`. Reach the gate with your own gate key, as
  `benchmarks/remote_session/auditor.sh` does, or with the shared key, which reads only (`status`, `fetch`):
  `ssh -o IdentitiesOnly=yes -i KEY research@HOST fetch RUN > RUN.tgz`. `status RUN` gives its `end.json`, whose SHA-512
  of each file of `out/` the tarball must match. The report's source line names the gate's host, each gate run and the
  main commit the gate was pinned to.

An honest run is required, because the tampered head is made from it. Two runs of the same strategy are refused.

## The report

`report.json` (`verity-network-e2e/report/v3`) holds:

- per run: the rehearsal's own check, the gate's `end.json` verdict, N2's network verdict and its refused sampling
  blocks, and, per proved end, whether Lean held it and its proving run's cost (below);
- per sampling block: N2's verdict, each end's proved verdict (accept or refuse, from the verdict row the verifier
  opens), and the head-check parts that fail (a diagnostic, evaluated in the clear on the rehearsal's witness);
- per negative: refused or not, and by what (N2, the rehearsal's check, the proved head checks);
- one sampling block proved alone (`n2_zk`'s cost pass), and the claim (below);
- the chain's steps that ran (`chain`: the advice, its bits per sampling block, the prove path, the negatives' rows),
  and `stubbed`.

`report.md` renders the same thing as tables. OK means: the honest run was accepted on every sampling block by N2 and,
where proofs ran, by every proved head check held by Lean; each negative was refused by everything that checked it; a
given advice is these runs' and rebuilt from their certified I/O; a bound prove path and the registry of negatives,
where they ran, passed. On the advice profile the rehearsal's own check stands in for N2, and a negative counts only for
its own violation. `report.md`'s title says what the OK rests on (`ran`) and what didn't run (`not_run`), so an OK on
the self-check alone reads as that. When it isn't OK, `why_not` (and the line under the title) says which of these
failed, a missing negative's run among them.

## What it proves, and what it doesn't

The claim is in `e2e.py` (`CLAIM`) and in both files, word for word. **It covers counts only.** For the rehearsal's one
link, every honest sampling block's per-bucket frame counts match the planned occupancy profile (on a served link, the
counts its ingress head derives). That fixes the link's timing to the profile. Late, extra and a tampered head are each
refused.

It does not cover:

- **N3, the content.** The bytes and order of the frames inside the counts aren't checked. N3 is retired, and the
  network certifier's step-indexed scan that supersedes it doesn't run here.
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
- **The model's size.** The report names the model as the plan does. The plan doesn't say whether the model was cut.

Zero-knowledge gaps, from `verity/core/protocols/certifier/README.md` ("Zero knowledge: what N2's verdict reads, and
what's open"):

- N2's verdicts here are decided in the clear (`LinkWindows.unproved_outcome`, as for any run that proves nothing),
  which reads the heads and the certifier's salts. `n2_zk` now decides each sampling block on a `--zk` proof of N2's
  registered proof block, one member a round; this report doesn't read that summary yet, so it proves nothing itself
  and refuses to adopt such a run (``--no-prove`` takes N2's verdicts only).
- The proofs an adopted earlier `n2_zk` summary holds are of its row adapter, not of N2's registered proof block, so
  `LinkWindows.proved` (which refuses a proof of any other Program) doesn't take them. Lean holds each proof to the
  program and partition that `n2_zk` wrote.
- The certificate's link to its head is checked on the host (`CertifiedRecord.verify`).

## Cost

Each end of a sampling block is one proof block of its member's statement. All of an end's sampling blocks are proved in
one proving run, one statement and one pair of proofs (`circuit.rep0.bin`, `circuit.rep1.bin`), and Lean verifies them
at once, so a sampling block has no proving time or proof of its own. The report gives each proving run's prover
seconds, Lean verify seconds, proof bytes and peak memory, undivided. Peak memory is the largest process the driver had
run by the end of the proving run (`RUSAGE_CHILDREN`, which only grows). `n2_zk` proves the honest members first, so the
tampered and extra runs' peaks include theirs. Beside them is `n2_zk`'s cost pass: a proving run on the honest
statement whose verifier draws one sampling block, per end, which is what one sampling block costs proved alone. The
driver's own peak (the proof block's lowering) is shown when it proved.
