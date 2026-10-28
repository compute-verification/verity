# network_traces

The network certifier (`verity/core/protocols/certifier/`) on real served traffic: per-token timing traces for its
calibration, the certifier in front of a served endpoint (as a proxy, `active.Proxy`, or as a relay that knows bytes
only, `warden`), the zero-knowledge proofs of their proof blocks, and the network chain's run of record end to end. Each
driver's docstring says what it runs, what it publishes and what its verifier receives.

## Running it

- `verity bench network_traces` runs `run.sh`: one chunk of per-token vLLM timing traces on a leased GPU
  (`record.py`), as a job whose resources are in the script's header.
- Every other `.sh` here is a job on a node, its command and resources in its header. The proving jobs build C-Flock's
  `flock-circuit`, the Lean `flock-verify` and the Lean firewall with `benchmarks/pouw/hidden_zk/job.sh`.
- Every run is published, with its report and outputs, as an Attempt in the evidence store
  (`tools/research/src/research/store/README.md`). Results are read from there, never from this README.
- The tests: `uv run tools/verity/check/suites.py verity-network-traces` (`.agents/skills/testing/SKILL.md` says where).

## Entry points

- **Calibration and the honest path.** `record.py` records per-token timing of real vLLM serving; `replay.py` runs
  honest traces through the reference protocol at calibrated parameters; `active_replay.py` runs the active certifier
  on recorded traces in real time, and in front of a live vLLM (`active_live.sh`, `active_live_cells.sh`). `covert.py`
  measures a colluding sender's channel through the declared clock sync, and `stall_probe.py` and `tick_load.py` find
  what makes a tick late.
- **The certifier in front of an endpoint.** `certifier_serve.py` runs the network certifier (`active.Proxy`) in front
  of a served OpenAI endpoint (`rehearse`) and checks what it published offline. The endpoints it rehearses against
  besides vLLM: `demo_endpoint.py` and `demo_stub.py`. `compact_render.py` is the client's shim for compact egress
  framing.
- **The certifier as a relay.** `warden_serve.py` runs `verity.core.protocols.certifier.warden`, the network certifier
  as a relay that knows bytes only, in front of any TCP server and checks what it published offline; `warden_chain.py`
  is the chain's reads of a relay run, and `warden_zk.py` reads the relay's bucket rows in the proof system.
- **Proofs in zero knowledge.** Each driver proves its proof blocks with C-Flock `prove --zk` behind the Lean firewall
  and holds them by the Lean `flock-verify verify --zk`: `n2_zk.py` (N2's registered proof block, N2's verdict of
  record), `n2_advised_zk.py` (N2's advised proof block), `n2_tie_zk.py` (the ingress tie), `egress_zk.py` (N2's egress
  bucket), `scan_zk.py` (the step-indexed scan), `scan_period.py` (the scan's answers held by `NetworkCheck.Period`),
  `transfer_zk.py` (a server-time block's proof blocks read against the certifier's transfer rows) and
  `compile_slice_zk.py` (the compile slice). `leak_scan.py` is the scan of what the verifier received that every one of
  them runs.
- **The chain's run of record.** `chain_record.py` (`chain_record.sh`, below), with `window_chain.py` (a served
  endpoint's Commit windows as one I1 v0 advice), `negatives.py` (what network accounting must refuse, each by the
  check and code that refuse it) and `leak_guard.py` (the record's leak guard, below). `block_certifier.py` decides a
  period by `network-period check` and writes its accepted answer as the block record's certifier entry.
  `demo_chain.sh` runs the one-H200 demo by the certifier-only path, and `served_session.py` is the served session's
  runbook (below).
- **The weights' load:** `load_relay.py` and `load_commit.py` (below).
- **A memory wipe's egress:** `wipe_gate.sh` and `wipe_egress.py` (below).
- `tool.py`: the harness's store declarations.

## The demo's load

The H200 demo serves plain INT8 weights generated from the public seed `verity/cmp92/h200/weights/v1`
(`e2e_h200.served_weights`). They are relayed to the H200 once per serving run and committed as they cross, through
two gate verbs on the auditor's machine:

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
(`warden-rehearse --scan --prompts` against the toy, `demo_stub.py serve`, on the peer's loopback), at a given UTC time
when asked, and fetches the run's bundle: with `witness/` into a 0700 directory outside the run dir, and to the run's
`$OUT` only the bundle less `witness/`. `wipe_egress.py BUNDLE --t0-ns T0 --t1-ns T1` then writes the `wipe-egress/v1`
record from what the gate published. Each sampling block whose sampling period meets [t0 − 1 s, t1 + 1 s) counts at the
schedule's capacity, 2,621,440 bits a sampling block, with K that count and δ 0. The record is covered only when every
one of those sampling blocks is certified. The wipe's reader takes `wipe_period.certify(p, egress_mib)` from a covered
record and keeps the wipe uncounted otherwise.

## The served session

`served_session.py` is the runbook of the served session of record, the network certifier in front of a served model.
`plan` checks the request with this tree's gate and prints each run's command and the endpoint's flags. `session` runs
the gate's scan rehearsal (`warden-rehearse --scan --prompts`) against the endpoint on the peer's loopback, by default
the H200 peer's `serve-openai` on :18782, inside the scan's class: prompts of 256 ids from a fresh secret seed (prompts
of the class's 1,024 reach the relay late), each answered greedily with at most 64 ids, ending at the base model's EOS
id 128001. It runs at a given UTC time when asked. The endpoint is started by hand (serve-openai directly with
`examples/vllm/serve.sh`'s serve-openai line, not serve.sh), sealed without the attention interiors. Its records are
published as an artifact without `requests.jsonl`, whose prompt ids stay on the served node, and with
`request_users.jsonl`, each engine request's `engine_id` and its warden user in `requests.jsonl`'s order. `chain`
rebuilds `requests.jsonl` from that map and the kept warden run's certified I/O in a 0700 directory beside the kept
fetch, so the endpoint step checks the endpoint's trace against the certified I/O. The session's fetch, with
`witness/`, stays in a 0700 directory outside the run dir, and `$OUT` gets only the bundle less `witness/` and the
session's outcome. `chain` runs the chain's warden path on the kept fetch, and only from a tree with the record's leak
guard. `session --gate local` is the dry rehearsal: this tree's gate on a scratch host, with `demo_stub.py serve` as
the endpoint. `chain --coins gate` draws the scan's units on the gate's coins on the auditor's side, the node the chain
runs on, which the challenger's coin link reaches through that node's own line of
`/etc/vy/coin-ports` (ports 7821-7830). It refuses the served node as the draw's peer, since the gate key and the
warden run's witness never go on the prover's machine. Without the gate's coins the chain draws on the verifier's
(`--coins verifier`, the default).

`certifier_piece.py` is the network certifier's piece of `benchmarks/one_stage/served_verdicts.sh`. Its `front`
(`CERTIFIER_FRONT`) runs before that run's requests: the gate's scan rehearsal sends one group of the window circuit's
class (4 prompts of 24 ids, 4 ids out, `--window-per-group`) to serve-openai, all of it in link-window 0, so the engine
closes it into the run's earliest window, the one the run audits, and the gate's fetch is kept in the serve dir's
`private/certifier`. Its `piece` (`CERTIFIER_CMD`) runs the chain's warden path on that fetch with the serve dir's
endpoint records, run dir and quarantine private beside the fetch, and answers with `network-period check` on the rows
form of the scan's link-window (`scan_period.rows_case`). The answer is accept only when the chain accepted, every request
of the audited window is a certified one inside that link-window, and the Lean check accepts. Anything else is a reject,
or undecided when a step couldn't run. With no fetch kept the piece refuses to run (`refused.txt`), which leaves the
certifier missing.

## The chain's run of record

The run of record takes the network chain end to end on the gate's machines:

1. The network certifier signs every record with its LMS key, on an auditor machine, never on the prover's node.
2. It certifies each sampling block's records and signs each sampling period's clock sync (a clock certificate).
3. The client receives the certificates and keeps them (`certificates.jsonl`, `clocks.jsonl`).
4. The consistency proof blocks are proved by sampled C-Flock `--zk` and decided by the Lean verifier of record: N2's
   (each sampling block's head checks and counts), and the scan's ingress and egress proof blocks chained by registered
   reads.
5. The verdict is the per-sampling-block acceptance that the certification period's leak bound reads.

**How it runs.** The gate runs on the auditor's machine: `sudo vy-challenger-gate certifier-rehearse --peer HOST --port
PORT --minutes 30 --traffic varied --requests 100 --tokens 64 --seed 1 --eos-ids 128001 --height 20` (or
`warden-rehearse --scan`, the network certifier as a relay), then `status` and `fetch`. A prompt range shorter than the
chat turn's tail (`CHAT_TAIL`, as the toy's 8-byte prompts are) makes every chat turn one prompt, so such an endpoint
takes its prompts on stdin and a wider margin: `demo_stub.py prompts --count 100 | sudo vy-challenger-gate
certifier-rehearse … --traffic varied --requests 100 --rate 0.2 --tokens 24 --eos-ids 0 --prompts --margin 4`
(`--prompts`: request j sends the (j mod n)-th; `--margin`: the plan's over the probes, 1 to 8, 2 otherwise). The
fetched tarballs go to one job, `chain_record.sh`, with the endpoint's records (`benchmarks/network_e2e/chain.py`'s
`ENDPOINT`: `advice_trace.jsonl`, `requests.jsonl`, `windows.jsonl` and `serve.json` from the endpoint's serve
directory, `CR_ENDPOINT`). The job runs `chain_record.py`'s steps, network-e2e's binding and the two provers, and writes
`out/report/report.json` and `report.md`:

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

**The record's leak guard** (`leak_guard.py`). Custody publishes the whole run dir, `inputs/` with it, so the job moves
each gate sent under `inputs/` to the quarantine once it is unpacked, and on every exit runs the guard on the run dir.
The guard moves to the quarantine (`CR_QUARANTINE`, by default `quarantine/RUN` beside the runs root, 0700) every file
that holds a gate run's witness: one under a `witness/` directory or named as `leak_guard.json` declares, one holding an
aligned witness piece raw or as hex at any offset, an archive member that does, or an archive it can't read through.
The run then fails (exit 3), and `out/leak-guard.json` names each find by its path alone. A piece the gate's public
files hold, as they are or with their JSON written again, is no find. The false positives are declared data,
`leak_guard.json`'s `allow`, each entry written only after its find was read by counts and digests; the one there
excuses the stall probe's cgroup counters in the runner's own `resources.jsonl`. A signal before the guard runs, or a
guard that can't run, withholds `out/`, `inputs/` and the logs whole.

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

`leak_bounds.py` is the one calculator of K, the bits a covert sender's secrets can number over a monitoring period.
`per_link` gives K for one link: one network certifier on one uplink, certifying a request rate of one scan class (an
`ingress_scan.Class`, or a window class's `scan`). `datacenter` sums K over a datacenter's links, and `world` over the
world's. Each comes in two forms, beside its share of one copy of the weights and its ε. The terms are those of
`NetTiming.Guarantees.PeriodLeakAccepted` (`Security/Definitions/NetworkCertifier/Guarantees/Period.lean`, proved in
`Security/Proofs/NetworkCertifier.lean`) as `NetworkCheck.Period`'s rows form counts them, whose sampling blocks'
failures `NetTiming.periodLeak_le_sum` sums by a union bound (`Security/Proofs/NetworkCertifier/PeriodSum.lean`); the
pooled form is `NetTiming.Guarantees.HorizonLeakAccepted`
(`Security/Definitions/NetworkCertifier/Guarantees/HorizonLeak.lean`). Both are about one device on one uplink. The sum
over links is our composition, which no Lean statement covers. It holds except where some link's statement fails, so a
datacenter at δ takes each link at δ / links, and each link pays its own faulty proof blocks (`datacenter`'s
docstring says why pooling them by weight copy would be unsound).

**K per monitoring period on a run**: `uv run benchmarks/network_traces/chain_record.py leak --warden-run RUN --scan
SUMMARY --out FILE`, with RUN a fetched warden run and SUMMARY its scan's `summary.json` (its draw). It reads
`leak_bounds` on the run's parameters (`deployment.json`), its traffic (`end.json`'s requests and sampling blocks) and
the scan's class (`scan_zk.CLASS`), and refuses a scan that left a proof block undrawn. FILE holds the widths, each
term, the links and a row for each monitoring period, scope and reading; `FILE.md`, which it prints, holds the tables.
A run's figures are its published Attempt's. The report's `d1` lists, per sampling block, whether its certificates, N2
and the scan accepted.

The terms, in the generator reading of the scan's statement (each proof block's registered rows its boundary values):
- The schedules and the fail cells, ⌈log2(cap_c + 1)⌉ each a connection (`capsTerm`, charged twice, as
  `#S_s ≤ ∏_c (cap_c + 1)`; the record's `terms.cap_per_request`). Here cap_c bounds `WardenSchedule.cells c`, the
  egress cells of connection c's lane, which is egress_cells. The endpoint's slot count isn't a term of the network
  certifier's schedule: it enters the compute audit's terms, which go into JointPeriodLive.
- The free inputs (`terms.free`): as registered, the tied audit's slot entries, NR·T / EVERY entries of NE bits
  (`ingress_scan.entry_bits`) a sampling block; or an entry's 33 + KO·IDB bits a request if `prog` is an input only at
  its anchored entries.
- No faulty term with every proof block drawn, and D_j = 1, so the jitter term is 0: assumed (J), `NetTiming.JitterBounded`,
  not met at D_j = 1 until a run shows the warden's written buckets tell a receiver no more than an idle one. The
  warden hands each bucket to a pacer that releases it at a time fixed before the bucket is made
  (`warden.Warden(paced=…)`, `warden_serve.py --paced-ms`).

Each entry carries a read bit (`ingress_scan.entry_layout`: an occupied slot no Program reads is refused). Bucket t
reads slot j's entry at row j·nb + ⌊t / EVERY⌋ of the entries (`NetworkCheck.Period`'s `readsEntries`) and ignores it
outside due and occupied slots, so a sampling block's entries are its NR·nb boundaries'. Unless the scan's `prog` is a
compute audit's `SlotEntry_v1` template's entries (`<descriptor id>/c0/o0`, one member a slot), the read bit is the
prover's word (an entry of read 1 and no outputs passes the scan), so `NetworkCheck.Period` refuses the sampling block
by name (`wired`). `wired` pins the template's whole body by its closure digest (`slotEntryPins`,
`io_shell.SlotEntry_v1` at its circuit-check rows), so the block round's `prog`, the o0 of scan_zk's `SlotEntry_v1`
stand-in, whose entries, read bit included, are the prover's, is refused there (scan_period.py). A slot's Program reads
the requests from its own registered `SCAN_STATE`, and the check refuses, last, by name, a sampling block whose scan's
state isn't a wired audit's `SCAN_STATE` by root and domain (`stateTied`). scan_zk's state tree is the window state
(`ingress_scan.window_state`: leaf 0 the empty state, leaf t S_{t−1}), each leaf but the first the o1 of the proof block
that computes it and the sin of the next, both registered reads, so a state tree that skips or alters a transition is
refused at that proof block's o1; the block round's registration names it as `scan/state`.
Tied (`NetworkCheck.Period`'s `progTied`, scan_zk's block round), the entries are a compute audit's outputs
and the scan's own free term is 0: the bits are that audit's free inputs (the stand-in's `x`), which the rows form's
K charges to the sampling block's term (`tiedAudits`). The egress tie moves the free term from the scan's `prog` to the
tied audit's inputs; K is unchanged in what it bounds, and the registered reading's free term is those slot entries.

The two forms:
- The pooled form (`HorizonLeakAccepted`) adds Fmax = ⌈ln(1/δ)⌉ faulty proof blocks of b = NS + NV bits, and
  Fmax·log2(N + 1) (the record's `terms.fmax` and `terms.b`). The form is conditional: `hesc` and `hleak` await
  `SampledKSLive` and P2.
- The per-period form (`PeriodLeakAccepted`) has no δ term with every proof block drawn.
- Both are proved: `NetTiming.SecurityProofs.PeriodLeakAccepted` and `NetTiming.SecurityProofs.HorizonLeakAccepted`.
- Beside either, ε = m·δ_scan is the scans' δ summed over the period's m sampling blocks and the links
  (`NetworkCheck.Period`'s `eps`), each scan at its δ (`leak_bounds.SCAN_DELTA`, `scan_zk --delta`'s default).

The scale:
- A week, a month and a year of one link, of the census's reference datacenter (`benchmarks/census/deployments.json`,
  one link a machine, `leak_bounds.datacenter_links`) and of the world: the adversary resource model's compute
  (`benchmarks/census/adversary.json`) in machines of the reference datacenter's GPUs, one link each
  (`leak_bounds.world_links`).
- Every link has one network certifier in front of all its machine's compute, serving at the run's rate. A
  datacenter's and the world's links are each pooled at δ / links. The world's requests past its links' rate are
  uncertified, and no row counts them.
- The weights are one copy of Llama-3.2-1B in BF16 (`leak_bounds.WEIGHTS`), whatever the links. The served checkpoint
  is int8, so every share names the copy it is of.

FILE has both readings of the free inputs. `prog` as registered is the record's K. `prog` only at its anchored entries
is what the generator-form lowering would give if it registers `prog` everywhere but its anchors, and is not the
record's number.

K also rests on these conditions:
- The generator reading of the scan's statement, in which each proof block's registered rows are its boundary values.
  That reading is a claim, not yet a locked guarantee.
- Each connection's cap is its lane's egress_cells, which the network certifier holds a lane to and
  `NetworkCheck.Period` doesn't check.
- The scan proves sampling block 0. A monitoring period's figures scale its per-sampling-block and per-request terms at
  the run's rate.
- The jitter term is 0 at D_j = 1, assumed (J), not met until a run shows the paced warden's written buckets tell a
  receiver no more than an idle one.
- An untied compute audit's (the loop's) free inputs don't enter K: its δ goes into JointPeriodLive's sum. The tied
  audit's (the `SlotEntry_v1` stand-in's) are the sampling block's free inputs above.

The report names what it stands on beyond the theorem (`chain_record.STUBBED`, `WARDEN_STUBBED` and
`HORIZON_CONDITION`) and what the theorem doesn't cover (its `uncovered`).
