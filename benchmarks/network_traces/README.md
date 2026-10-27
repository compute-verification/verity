# network_traces

The network certifier (`verity/core/protocols/certifier/`) on real served traffic: per-token timing traces for its
calibration, the certifier in front of a served endpoint (as a proxy, `active.Proxy`, or as a relay that knows bytes
only, `warden`), the zero-knowledge proofs of their proof blocks, and the network chain's run of record end to end. Each
driver's docstring says what it runs, what it publishes and what its verifier receives.

## Running it

- `verity bench network_traces` runs `run.sh`: one chunk of per-token vLLM timing traces on a leased GPU
  (`record.py`), as a `research run` job whose command is in the script's header.
- Every other `.sh` here is a `research run` job, its command in its header. The proving jobs build C-Flock's
  `flock-circuit`, the Lean `flock-verify` and the Lean firewall with `benchmarks/pouw/hidden_zk/job.sh`.
- `research run` publishes every run, with its report and outputs, as an Attempt in the evidence store
  (`tools/research/src/research/store/README.md`). Results are read from there, never from this README.
- The tests: `uv run tools/verity/check/suites.py verity-network-traces` (`.agents/skills/testing/SKILL.md` says where).

## Entry points

- **Calibration and the honest path.** `record.py` records per-token timing of real vLLM serving; `replay.py` runs
  honest traces through the reference protocol at calibrated parameters; `active_replay.py` runs the active certifier
  on recorded traces in real time, and in front of a live vLLM (`active_live.sh`, `active_live_cells.sh`);
  `lean_replay.py` replays every recorded run's audit through the Lean verifier beside the Python. `covert.py`
  measures a colluding sender's channel through the declared clock sync, and `stall_probe.py` and `tick_load.py` find
  what makes a tick late.
- **The certifier in front of an endpoint.** `certifier_serve.py` runs the network certifier (`active.Proxy`) in front
  of a served OpenAI endpoint (`rehearse`) and checks what it published offline. The endpoints it rehearses against
  besides vLLM: `demo_endpoint.py`, `demo_stub.py` and `torch_loop_stub.py`.
  `compact_render.py` is the client's shim for compact egress framing.
- **The certifier as a relay.** `warden_serve.py` runs `verity.core.protocols.certifier.warden`, the network certifier
  as a relay that knows bytes only, in front of any TCP server and checks what it published offline; `warden_chain.py`
  is the chain's reads of a relay run, and `warden_zk.py` reads the relay's bucket rows in the proof system.
- **Proofs in zero knowledge.** Each driver proves its proof blocks with C-Flock `prove --zk` behind the Lean firewall
  and holds them by the Lean `flock-verify verify --zk`: `n2_zk.py` (N2's registered proof block, N2's verdict of
  record), `n2_advised_zk.py` (N2's advised proof block), `n2_tie_zk.py` (the ingress tie), `egress_zk.py` (N2's egress
  bucket), `scan_zk.py` (the step-indexed scan), `scan_period.py` (the scan's answers held by `NetworkCheck.Period`),
  `transfer_zk.py` (a server-time block's proof blocks read against the certifier's transfer rows) and
  `compile_slice_zk.py` (the compile slice). `leak_scan.py` is the scan of what the verifier received that every one of
  them runs, and `n2_lean_replay.py` replays recorded N2 runs' verdicts through the Lean beside the Python.
- **The chain's run of record.** `chain_record.py` (`chain_record.sh`, below), with `window_chain.py` (a served
  endpoint's Commit windows as one I1 v0 advice) and `negatives.py` (what network accounting must refuse, each by the
  check and code that refuse it). `block_certifier.py` decides a period by `network-period check` and writes its
  accepted answer as the block record's certifier entry, for `chain_record.py` and all_protocols' runner alike.
  `demo_chain.sh` runs the one-H200 demo by the certifier-only path.
- **The weights' load:** `load_relay.py` and `load_commit.py` (below).
- **A memory wipe's egress:** `wipe_gate.sh` and `wipe_egress.py` (below).
- `tool.py`: the harness's store declarations.

## The demo's load

The H200 demo serves plain INT8 weights generated from the public seed `verity/cmp92/h200/weights/v1`
(`e2e_h200.served_weights`). They are relayed to the H200 once per serving run and committed as they cross, through
two gate verbs on the auditor's machine (vy-challenger-1):

- `vy-challenger-gate weights-write [--shape h200|tiny]` writes the shape's weights stream onto the auditor's store disk
  (`/var/lib/vy-challenger/store`) once. It keeps the stream only when its tensors' SHA-256 fold to the pinned manifest
  digests (`load_relay.STORE_DIGESTS`).
- `vy-challenger-gate load-relay --peer HOST --port PORT --sink-port PORT --program SHA512 --minutes M` (with
  `warden-rehearse`'s flags) is `warden-rehearse` with the load first. The store goes over 4 TCP streams, an SSH
  tunnel each, to `load_relay.py sink` on the H200's loopback; each value is held to its digest and committed under the
  reading program's domain as it crosses. The certifier's one signer on that link then signs the load's heads before
  its first sampling block, so one LMS key and one attachment carry the load and the sampling blocks. `load.json` (the
  registrations the server-time block's record pins) and `load-relay.json` are public in the run's `out/`.
- The witness, every row's salts, is the sink's `--witness` file (0600) on the prover's host. The certifier's host
  never writes a salt.

## The wipe's egress

Memory accounting's streamed wipe signs its egress `UNCOUNTED` when its challenger is on another host. The count over
its period [t0, t1) comes after, from a relay run that spans it: `wipe_gate.sh` runs that relay through the gate
(`warden-rehearse --scan --prompts` against the toy, `demo_stub.py serve`, on the card's loopback), at a given UTC time
when asked, and fetches the run's bundle. `wipe_egress.py BUNDLE --t0-ns T0 --t1-ns T1` then writes the `wipe-egress/v1`
record from what the gate published. Each sampling block whose sampling period meets [t0 − 1 s, t1 + 1 s) counts at the
schedule's capacity, 2,621,440 bits a sampling block, with K that count and δ 0. The record is covered only when every
one of those sampling blocks is certified. The wipe's reader takes `wipe_period.certify(p, egress_mib)` from a covered
record and keeps the wipe uncounted otherwise.

## The chain's run of record

The run of record takes the network chain end to end on the gate's machines:

1. The network certifier signs every record with its LMS key, on an auditor machine (vy-challenger-1), never on the
   prover's node.
2. It certifies each sampling block's records and signs each sampling period's clock sync (a clock certificate).
3. The client receives the certificates and keeps them (`certificates.jsonl`, `clocks.jsonl`).
4. The consistency proof blocks are proved by sampled C-Flock `--zk` and decided by the Lean verifier of record: N2's
   (each sampling block's head checks and counts), and the scan's ingress and egress proof blocks chained by registered
   reads.
5. The verdict is the per-sampling-block acceptance that the certification period's leak bound reads.

**How it runs.** The gate runs on vy-challenger-1: `sudo vy-challenger-gate certifier-rehearse --peer HOST --port PORT
--minutes 30 --traffic varied --requests 100 --tokens 64 --seed 1 --eos-ids 128001 --height 20` (or `warden-rehearse
--scan`, the network certifier as a relay), then `status` and `fetch`. A prompt range shorter than the chat turn's tail
(`CHAT_TAIL`, as the toy's 8-byte prompts are) makes every chat turn one prompt, so such an endpoint takes its prompts
on stdin and a wider margin: `demo_stub.py prompts --count 100 | sudo vy-challenger-gate certifier-rehearse …
--traffic varied --requests 100 --rate 0.2 --tokens 24 --eos-ids 0 --prompts --margin 4` (`--prompts`: request j sends
the (j mod n)-th; `--margin`: the plan's over the probes, 1 to 8, 2 otherwise). The fetched tarballs go to one job,
`chain_record.sh`, with the endpoint's records (`benchmarks/network_e2e/chain.py`'s `ENDPOINT`: `advice_trace.jsonl`,
`requests.jsonl`, `windows.jsonl` and `serve.json` from the endpoint's serve directory, `CR_ENDPOINT`). The job runs
`chain_record.py`'s steps, network-e2e's binding and the two provers, and writes `out/report/report.json` and
`report.md`:

| step | what it checks | its ε at δ | computed in the clear |
|---|---|---|---|
| custody | each gate run's files against the SHA-512s in its `end.json` | none: a check of files | the files' digests |
| certificates | each sampling block's two certified records and its clock certificate, at the client, under the certifier's key and attachment; with `CR_DEPLOYED=1`, a key that isn't LMS at H = 20 is refused | none: signatures | every LMS signature, and the host audit of the client's own heads |
| endpoint | the endpoint's own advice bound to the run (each Commit window opened to its registered root, rebuilt from the certified I/O) and the I/O tie on the honest run (`e2e.py --no-prove`); required on the advice profile | none: in the clear | all of it |
| N2 | each sampling block's N2 proof block, proved `--zk` and decided by N2's verifier of record: the advised proof block on the bound advice and the certifier's clock certificates on the advice profile (`n2_advised_zk.py --e2e`), the plain one on the others (`n2_zk.py`) | Selection All: no sampling block undrawn; a sampled draw of k of n sampling blocks hides at most b faulty ones except with probability δ (the report's `wrong_units_at_delta`) | the registration, the draw, the certificates' public fields, the syncs, the pub row opened to the verifier's copy, the output word |
| scan | the scan's ingress and egress proof blocks of a sampling block, one registration, `verify --zk --registered` | the audit's `wrong_units` at the draw's sampling design | each proof block's output bit; the I/O tie stays outside the proofs |
| Lean | every proof of both steps is the verifier's own `flock-verify verify --zk` run | per proof, ε_ks + ε_bind | nothing |

The job's environment picks what it runs: N2's sampled draw on the gate's coins is `CR_N2_CMD` (`n2_advised_zk.py prove
--e2e --draw K --challenger gate:…`), the scan of the gate's own rows is `CR_WARDEN_RUN` with `CR_SCAN_ARGS="--draw K
--challenger gate:…"` (`scan_zk.py prove --warden-run`), and the client's `certificates.jsonl` and `clocks.jsonl` are
`CR_CLIENT`, by default the network certifier rehearsal's own `client/`. A step on a stand-in (the scan on its own
synthetic run, or N2 on `CR_N2_STANDIN=simulated`) never makes the report accepted, and a step that fails or doesn't run
says why.

**The relay path** (`CR_PATH=warden`, `chain_record.WARDEN`). Every step reads the relay's own run (the gate's
`warden-rehearse`, its witness under `witness/`; `warden_chain.py`), and a proxy rehearsal's run, whose records come
through `active.Proxy` in front of the endpoint, is refused by name wherever it is given. The certificates step reruns
the relay's own offline check on its witness (`warden_serve.check`) and holds the public `certificates.jsonl` to the
certified lines. The relay signs no clock certificate, so none is read. The endpoint step binds the endpoint's own
advice to the certified I/O read off the relay's cells (`chain_record.py endpoint`, with `CR_EOS_IDS` the model's EOS
ids). N2 is the plain proof block on the relay's link (`n2_zk.py prove --warden-run`: both heads at the relay's grid and
parameters digest, no sync declared or charged). The I/O tie is proved on the relay's ingress (`n2_tie_zk.py prove
--class warden`, the report's `tie` step), and the scan is `scan_zk.py prove --warden-run`'s. Each step is held to the
one relay run the certificates step read, and every one is required.

A Commit window that carries a request across its open or close (`verity_vllm.pipeline.window_advice`, schema
`verity-vllm/window-advice/v1`) is bound with the Commit window before it: `chain.served_advice` joins them into one I1
v0 advice (`window_chain.join`), and `n2_advised_zk --e2e` joins the run's Commit windows the same way. A carried
request's record and state are the earlier Commit window's, as its registered advice closed them
(`window_advice.follows`), never the later one's own string's, so the advice's bits, and N2's K, charge the carry once.
A Commit window that carries a request in from one the run doesn't hold, or that leaves a request with outputs still to
keep and no next Commit window, is refused. N2 proves the joined advice at its link's size class, whose advised counts
circuit-check pins (`advised.circuit-check.py`).

## The leak bound

`chain_record.py leak` computes K, the bits a covert sender's secrets can number over a monitoring period, from a run's
files: its parameters (`deployment.json`), its traffic and the scan's class. The terms are those of
`NetTiming.Guarantees.PeriodLeakAccepted` (`Security/Definitions/NetworkCertifier/Guarantees/Period.lean`, proved in
`Security/Proofs/NetworkCertifier.lean`), whose sampling blocks' failures `NetTiming.periodLeak_le_sum` sums by a union
bound (`Security/Proofs/NetworkCertifier/PeriodSum.lean`); the pooled form is
`NetTiming.Guarantees.HorizonLeakAccepted` (`Security/Definitions/NetworkCertifier/Guarantees/HorizonLeak.lean`). The
report's `d1` lists, per sampling block, whether its certificates, N2 and the scan accepted, and the report names what
it stands on beyond the theorem (`chain_record.STUBBED`, `WARDEN_STUBBED` and `HORIZON_CONDITION`).
