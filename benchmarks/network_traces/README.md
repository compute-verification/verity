# network_traces

The network certifier's measured runs and their zero-knowledge proofs: the certifier in front of a served endpoint
(`certifier_serve.py`), the network certifier functioning as an active warden (`warden_serve.py`), N2's units proved by C-Flock `--zk` (`n2_zk.py`,
`n2_advised_zk.py`), the scan's chained units (`scan_zk.py`), and the chain's run of record (`chain_record.sh`). Each
driver's docstring says what it runs and what its verifier receives. `n2_lean_replay.py` replays the recorded N2 proving
runs' link-window verdicts through the Lean verdict of record beside the Python, against what each run recorded.

## The demo's load

The H200 demo serves CMP-95 interface 1's plain INT8 weights, generated from the public seed
`verity/cmp92/h200/weights/v1`. They are relayed to the H200 once per serving session and committed as they cross
(CMP-97 option (a)), through two gate verbs on vy-challenger-1:

- `vy-challenger-gate weights-write [--shape h200|tiny]` writes the shape's weights stream onto the auditor's store disk
  (`/var/lib/vy-challenger/store`) once. It keeps the stream only when its tensors' SHA-256 fold to the pin of
  interface 1's manifest digests (`load_relay.STORE_DIGESTS`, from
  art:a6775aeef1f6d11badf5a7a38460cff7d4d59f27e70dbc3bf8dd995fbb4c82da).
- `vy-challenger-gate load-relay --peer vy-dc-h200-2 --port PORT --sink-port PORT --program SHA512 --minutes M` (with
  `warden-rehearse`'s flags) is the warden's rehearsal with the load first. The store goes over 4 TCP streams, an SSH
  tunnel each, to `load_relay.py sink` on the H200's loopback; each value is held to its digest and committed under the
  reading program's domain as it crosses. The serving session's one signer then signs the load's 733 heads before its
  first window, so one LMS key and one session carry the load and the windows. `load.json` (the registrations the
  block record pins) and `load-relay.json` are public in the run's `out/`.
- The witness, every row's salts (4.85 GB at H200), is the sink's `--witness` file (0600) on the prover's host,
  vy-dc-h200-2. The certifier's host never writes a salt.

## The wipe's egress

Memory accounting's streamed wipe signs its egress `UNCOUNTED` when its challenger is on another host. The count over its
session [t0, t1) comes after, from a warden session that spans it: `wipe_gate.sh` runs that session through the gate
(`warden-rehearse --scan --prompts` against the toy, `demo_stub.py serve`, on the card's loopback), at a given UTC time
when asked, and fetches the run's bundle. `wipe_egress.py BUNDLE --t0-ns T0 --t1-ns T1` then writes the
`wipe-egress/v1` record from what the gate published. Each link-window meeting [t0 − 1 s, t1 + 1 s) counts at the
schedule's capacity, 2,621,440 bits a link-window, with K that count and δ 0. The record is covered only when every one
of those windows is certified. The wipe's reader takes `wipe_period.certify(p, egress_mib)` from a covered record and
keeps the wipe uncounted otherwise.

## The chain's run of record

The run of record takes the network chain end to end on the gate's machines:

1. The network certifier signs every record with its LMS key, on an auditor machine (vy-challenger-1), never on the
   prover's node.
2. It certifies each link-window's records and signs each window's clock sync (a clock certificate).
3. The client receives the certificates and keeps them (`certificates.jsonl`, `clocks.jsonl`).
4. The consistency units are proved by sampled C-Flock `--zk` and decided by the Lean verifier of record: N2's units
   (each link-window's head checks and counts), and the scan's ingress and egress units chained by registered reads.
5. The verdict is the per-window acceptance that the period's leak bound reads.

**How it runs.** vy-admin runs the gate on vy-challenger-1: `sudo vy-challenger-gate certifier-rehearse --peer
vy-nebius-1 --port PORT --minutes 30 --traffic varied --requests 100 --tokens 64 --seed 1 --eos-ids 128001 --height 20`
(and `warden-rehearse --scan`, where it functions as an active warden), then `status` and `fetch`. A prompt range shorter
than the chat turn's tail (`CHAT_TAIL`, as the toy's 8-byte prompts are) makes every chat turn one prompt, so such an
endpoint takes its prompts on stdin and a wider margin: `demo_stub.py prompts --count 100 | sudo vy-challenger-gate
certifier-rehearse … --traffic varied --requests 100 --rate 0.2 --tokens 24 --eos-ids 0 --prompts --margin 4`
(`--prompts`: request j sends the (j mod n)-th; `--margin`: the plan's over the probes, 1 to 8, 2 otherwise). The fetched tarballs go to one job, `chain_record.sh`, with the
endpoint's records (`benchmarks/network_e2e/chain.py`'s `ENDPOINT`: `advice_trace.jsonl`, `requests.jsonl`, `windows.jsonl` and
`serve.json` from the endpoint's serve directory, `CR_ENDPOINT`). The job runs `chain_record.py`'s steps,
network-e2e's binding and the two provers, and writes `out/report/report.json` and `report.md`:

| step | what it checks | its ε at δ | computed in the clear |
|---|---|---|---|
| custody | each gate run's files against the SHA-512s in its `end.json` | none: a check of files | the files' digests |
| certificates | each window's two certified records and its clock certificate, at the client, under the certifier's key and session; with `CR_DEPLOYED=1`, a key that isn't LMS at H = 20 is refused | none: signatures | every LMS signature, and the host audit of the client's own heads |
| endpoint | the endpoint's own advice bound to the run (each Commit window opened to its registered root, rebuilt from the certified I/O) and the I/O tie on the honest run (`e2e.py --no-prove`); required on the advice profile | none: in the clear | all of it |
| N2 | each link-window's unit, proved `--zk` and decided by N2's verifier of record: the advised unit on the bound advice and the certifier's clock certificates on the advice profile (`n2_advised_zk.py --e2e`), the plain unit on the others (`n2_zk.py`) | Selection All: no window undrawn; a sampled draw of k of n windows hides at most b wrong windows except with probability δ (the report's `wrong_units_at_delta`) | the registration, the draw, the certificates' public fields, the syncs, the pub row opened to the verifier's copy, the output word |
| scan | the scan's ingress and egress units of a link-window, one registration, `verify --zk --registered` | the audit's `wrong_units` at the draw's law | each unit's output bit; the I/O tie stays outside the proofs |
| Lean | every proof of both steps is the verifier's own `flock-verify verify --zk` run | per proof, ε_ks + ε_bind | nothing |

The pieces not yet on `main` change only the commands the job runs. N2's sampled draw on the gate's coins is
`CR_N2_CMD` (`n2_advised_zk.py prove --e2e --draw K --challenger gate:…`). The scan of the gate's own rows is
`CR_WARDEN_RUN` with `CR_SCAN_ARGS="--draw K --challenger gate:…"` (`scan_zk.py prove --warden-run`). The client's
`certificates.jsonl` and `clocks.jsonl` are `CR_CLIENT`, by default the network certifier rehearsal's own `client/`.
A step on a stand-in (the scan on its own synthetic run, or N2 on `CR_N2_STANDIN=simulated`) never makes the report
accepted, and a step that fails or doesn't run says why.

**The warden path** (`CR_PATH=warden`, `chain_record.WARDEN`). Every step reads the active warden's own run (the
gate's `warden-rehearse`, its witness under `witness/`; `warden_chain.py`), and a network certifier rehearsal's run,
whose records come through `active.Proxy` in front of the endpoint, is refused by name wherever it is given. The
certificates step reruns the warden's own offline check on its witness (`warden_serve.check`) and holds the public
`certificates.jsonl` to the certified lines. The warden signs no clock certificate, so none is read. The endpoint step
binds the endpoint's own advice to the certified I/O read off the warden's cells (`chain_record.py endpoint`, with
`CR_EOS_IDS` the model's EOS ids). N2 is the plain unit on the warden link (`n2_zk.py prove --warden-run`: both heads
at the warden's grid and parameters digest, no sync declared or charged). The I/O tie is proved on the warden's ingress
(`n2_tie_zk.py prove --class warden`, the report's `tie` step), and the scan is `scan_zk.py prove --warden-run`'s.
Each step is held to the one warden run the certificates step read, and every one is required.

A Commit window that carries a request across its open or close (`verity_vllm.pipeline.window_advice`, schema
`verity-vllm/window-advice/v1`) is bound with the window before it: `chain.served_advice` joins them into one I1 v0
advice (`window_chain.join`), and `n2_advised_zk --e2e` joins the run's windows the same way. A carried request's
record and state are the earlier window's, as its registered advice closed them (`window_advice.follows`), never the
later window's own string's, so the advice's bits, and N2's K, charge the carry once. A window that carries a request
in from a window the run doesn't hold, or that leaves a request with outputs still to keep and no next window, is
refused. On the :18097 endpoint, 8 of the 56 Commit windows of c20261025T062158Z-9ff3 carry a request (114–115,
143–144, 147–148 and 152–153), and all 56 bind and rebuild from the certified I/O. N2 proves the joined advice at
that link's size class, T = 60 and R_in = 2, whose advised counts circuit-check pins (`advised.circuit-check.py`).

### The run of record (25–26 Oct 2026)

**Trees.**
- Every gate run is pinned to `main` aec7f4ef.
- The endpoints 6541 and 9995 run vllm's tree 1eb3995c56. r20261026-000932-b50b runs 9995's settings and stamps
  each request's arrival, submission, admission and first token in its `requests.jsonl`.
- The record's jobs run `cursor/chain-record-tree-061a`:
  - for N2, 9c6d972cf7: aec7f4ef with #4384 at 6edd24102a, #4407 at 1868e6ff6c (with #4342 at f83d75a1ff) and #4387 at
    35bb96cd76;
  - for the scan, a0fbcfe1b3, which adds `scan_zk`'s refusal of a void run by name.

**The runs.**

| what ran | run | endpoint | verdict |
|---|---|---|---|
| network certifier | c20261025T224504Z-2982 | r20261025-205018-6541 (:18097, `--window-requests 4`) | accepted |
| the network certifier functioning as an active warden | c20261025T230848Z-30ab | 6541 | void: lateness |
| the same | c20261025T234322Z-0420 | r20261025-233025-9995 (:18098, `--window-requests 8`) | void: lateness |
| the same, `--tokens 16` | c20261026T000242Z-89bf | 9995 | void: lateness |
| the same, light traffic (`--requests 40 --tokens 16`) | c20261026T010321Z-d951 | r20261026-000932-b50b (:18099) | accepted |
| the same, full traffic (`--requests 100 --tokens 16`) | c20261026T034045Z-6cd9 | r20261026-034055-bbee (:18100) | accepted |
| N2 on 2982 | r20261025-231253-ebf6, r20261025-231301-6b61, r20261025-231310-20da | | accepted |
| the scan on d951 | r20261026-011909-f489 | | accepted: honest 64 of 64, forged refused |
| the scan on 6cd9 | r20261026-035637-a739 | | accepted: honest 64 of 64, forged refused |

- The network certifier's run was `certifier-rehearse`: 100 varied requests of up to 64 tokens, 49 windows, the advice
  profile and an LMS key of H = 20. It is stored as art:fc8d4f61ebb0a45ecd47456d818bebcfeb0723ab1a8f00dd98f7983064eed76b.
- The active-warden runs were `warden-rehearse --scan --minutes 20 --requests 100`. Their public files are stored as
  art:1aa7089c7e11b282335c44f79d07ee6a9278cc49141539d8d559d3af3d074e84 (30ab),
  art:8da6ee13709bab38668a08750095b24f3aa3229584335ff8995921f14ba2da85 (0420) and
  art:af3344f7e9466408d8eb598e1db69625c6d27fbd93c01444929166abc885ac95 (89bf), each labelled `void lateness`.
- The network certifier's light-traffic active-warden run d951 ran `warden-rehearse --scan --minutes 20 --requests 40
  --tokens 16`: 40 of 40 requests ended, 129 of 129 link-window certificates, a peak tick lateness of 2.19 ms. Its
  public files are stored as art:763f549840c9882cfa3cadceb80187d974babefcd5570a4909e81d6aa0d269ab.
- The network certifier's full-traffic active-warden run 6cd9 sent 89bf's request (`--scan --minutes 20 --requests 100
  --tokens 16`) to bbee: 100 of 100 requests ended, 225 of 225 link-window certificates over 75 link-windows, a peak
  tick lateness of 2.18 ms and none past the slack. Its public files are stored as
  art:0afe9ca3ed8fa6e8941bdc07cc8f283700ef2e54545c142071b9a8f62beb8cc9.

**The steps on 2982.**

| step | verdict | what it found |
|---|---|---|
| custody | accepted | every fetched file against its SHA-512 in `end.json` |
| certificates | accepted | 49 windows: 98 certificates and 49 clock certificates verify under the network certifier's LMS key (H = 20, `CR_DEPLOYED=1`) and session, and they are exactly the client's kept copies |
| endpoint | accepted | 6541's advice is bound to 2982 in all 77 Commit windows (#4384's carry binding) and rebuilt from the certified I/O. The I/O tie, in the clear, accepted the honest run: 100 requests, 3,172 outputs placed |
| N2 | accepted | each member drawn 12 of 49 windows on the gate's coins (`subset:12`, one challenger session a run) |
| Lean | every N2 proof | each decided by the verifier's own `flock-verify verify --zk` |

**N2's members.**

| member | windows drawn | verdict |
|---|---|---|
| honest | 2, 5, 6, 9, 16, 18, 24, 28, 29, 32, 33, 48 | accepted: 12 of 12 proved and held by Lean |
| short | 1, 7, 12, 16, 18, 19, 23, 32, 39, 40, 44, 48 | refused: its faulty window 1 was drawn and refused |
| tampered_all | 2, 6, 10, 12, 18, 26, 27, 28, 31, 41, 43, 45 | refused: all 12 drawn windows refused |
| other_subset | 2, 4, 14, 17, 18, 21, 25, 26, 34, 38, 40, 46 | refused by name: the windows it registered and proved aren't the Lean verifier of record's draw on the period's coins |
| moved | 1, 3, 13, 16, 21, 23, 33, 35, 41, 42, 43, 44 | escaped the draw: its one faulty window (0) wasn't drawn |
| tampered | 1, 5, 12, 14, 19, 20, 33, 34, 36, 40, 42, 44 | escaped the draw: its one faulty window (0) wasn't drawn |

The two escapes are within the draw's bound. An accepting draw of 12 of 49 windows hides at most 2 wrong windows
except with probability 1/2, and at most 29 except with 2⁻²⁰ (`subset_bound`). A single faulty window escapes with
probability 37/49 ≈ 0.76.

- The unit is `NetworkAdvisedLinkWindow_v2{EGRESS=(6,60,0),INGRESS=(6,60,0),ADVISED=(60,2,1)}`: 3,453,596 ANDs,
  k_log 23.
- The cost per window, on one 16-core n1 slot:
  - a proving session of 64–80 s (the prover's own time 35–42 s, 1,672,612 proof bytes);
  - a Lean verify of 36–39 s.
- No witness piece leaked to the verifier. The finds the scan lists are structured: each straddles the proof format's
  fields.

**The scan on d951** (r20261026-011909-f489, one 8-core n1 slot, the gate's coins in challenger session
s20261026T012652Z-6165). Its files, without any gate run's witness, are stored as
art:b6a6013d2f37a00709e45261370f61b9d02ef5b62870d9781b830a970c40e323. The run's own record carries d951's witness
and is labelled `do-not-cite`.
- Its sampling block is link-window 0 (T = 32 buckets), 64 units of two subcircuits:
  - 32 of `IngressScanBucket_v1{B=512,r=4,lanes=4,C=47,IDB=17,EVERY=10,WITHIN=12,D=21,KO=64}`: 1,395,428 ANDs, k_log
    26;
  - 32 of `EgressBucket_v1{B=512,R=16,r=4,NR=4,K=64,IB=17,D=21}`: 77,099 ANDs, k_log 23.
- Honest: 64 of 64 units drawn on the gate's coins, every proof held by Lean, accepted, and the gate's outcome
  accepted. An accepting draw of all 64 hides no faulty proof block except with probability 2⁻²⁰.
- Forged (the 32 egress units, one of them forged): refused by Lean at the registration ("table row 22 does not open
  warden/egress/0's registered root at position 22").
- The cost a unit:
  - ingress: 10.6 s of proving (a 12.1 s share of the session), 2,036,356 proof bytes, a Lean verify of 1.20 s;
  - egress: 2.07 s of proving (2.56 s), 1,556,020 proof bytes, a Lean verify of 0.70 s.
- `chain_record.py report` over the certificates, the endpoint, N2 on 2982 and this scan: accepted. Every negative is
  refused, or escaped N2's draw within its bound.
- d951 is light traffic: 40 requests of at most 16 tokens over 43 link-windows. Full traffic is run 6cd9 on vllm's
  endpoint with `finalize` off the engine loop, r20261026-034055-bbee (:18100, 1a3ffe4a88). bbee replaced
  r20261026-021253-83ee, which stopped with n1's 03:30Z restart.

**The scan on 6cd9** (r20261026-035637-a739, one 8-core n1 slot, the gate's coins in challenger session
s20261026T040447Z-ffd7) is the record's scan at full traffic. Its run record has no gate run's witness, and is stored
as art:832cc424a984fc015958d5758350f03978d7e0c792d2b8462cceddaec9a988ff.
- Its sampling block is link-window 0, with the same 64 units of the same two subcircuits as on d951.
- Honest: 64 of 64 units drawn on the gate's coins, every proof held by Lean, accepted. An accepting draw of all 64
  hides no faulty proof block except with probability 2⁻²⁰.
- Forged (the 32 egress units, one of them forged): refused by Lean at the registration ("table row 22 does not open
  warden/egress/0's registered root at position 22").
- The cost a unit:
  - ingress: 10.6 s of proving (a 12.1 s share of the session), 2,036,356 proof bytes, a Lean verify of 1.14 s;
  - egress: 1.91 s of proving (2.39 s), 1,556,020 proof bytes, a Lean verify of 0.70 s.
- `chain_record.py report` over the certificates, the endpoint, N2 on 2982 and this scan: accepted, with every
  negative refused or escaped N2's draw within its bound. The report and `leak` on 6cd9 are stored as
  art:9f4adc70e0fb01ad3fbbee273269b4b0cb3392e5979360aed72893fe827059a6.
- The job's own rc is 1: its report runs without N2, which this part of the record doesn't run.

**Why the active-warden runs are void.** The gate's traffic is seeded, so all three runs sent the same schedule. On
every run the network certifier itself kept time, and the endpoint served every request, to its end or to the client's
abort after the network certifier had failed it. The lateness is between the endpoint's output and the egress
schedule.

The first two runs:
- Connections 81 and 82 arrive at t ≈ 151.5 s, in a burst of four requests within 1.2 s:
  - 81 has a prompt of 531 ids and 43 tokens;
  - 82 has a prompt of 839 ids and 20 tokens.
- On 9995, both were carried across the endpoint's close of window 74.
- The egress schedule sends a connection's first egress cell at admission + D (2.1 s), then r = 4 cells a bucket.
- About 2.5 s after admission (buckets 1541 and 1542), neither connection's next cell had arrived from the endpoint,
  so the network certifier failed both closed as late:
  - on 30ab, at egress cells 19 and 9;
  - on 0420, at egress cells 19 and 10.
- 98 of 100 requests ended.
- The network certifier itself kept time on both runs: 225 of 225 link-windows certified, a peak tick lateness of
  2.23 ms and 4.30 ms, and none past the 25 ms slack.
- `--window-requests 8` didn't change it: both runs fail the same two connections at the same buckets.

The third run, 89bf, capped every output at 16 tokens (`--tokens 16`, the least the gate takes) and failed more:
- 10 connections failed closed as late: 54, 55, 63, 65, 71, 72, 74, 81, 82 and 83. Each failed at egress cell 0 to 8,
  before any output token reached the network certifier. 90 of 100 requests ended.
- The network certifier kept time: 225 of 225 link-windows certified, a peak tick lateness of 2.10 ms, none past the
  slack.
- All ten are in the schedule's four bursts, at t ≈ 106, 120–122, 138–139 and 151–152 s. The endpoint's records put
  them in its four windows of several requests there (139, 144, 149 and 154). Those windows' sealed bundles are
  5.6–10.8 GB, against about 0.3 GB for a window of one request, and their seals were queued 47–72 s behind.
- On 0420 the same bursts' windows sealed 7–29 s behind, and only the window carried across a close failed. Over 89bf
  the endpoint's seal queue grew from 0 to 97 s (114.6 GB sealed for 100 requests); over 0420 it reached 53 s.
- So shorter outputs didn't help.

vllm traced the lateness to the endpoint's window close. While a window closes, its `finalize` (the committer
draining) runs on the engine loop, so no step runs, nothing new is submitted and in-flight streams pause. Around the
burst windows a close took 0.4–1.9 s, longer under seal load; the seal queue's depth alone doesn't predict lateness.
r20261026-021253-83ee runs `finalize` off the loop: a close of 0.012 s at p90. On its successor bbee, 89bf's request,
the same seeded schedule, is accepted (6cd9: 100 of 100 requests ended).

vllm's reading of bbee's records over 6cd9 (r20261026-040530-d82e and r20261026-040621-b1e4):
- The part of a window close still on the engine loop: 0.025 s at p50, 0.05 s at p90 and 0.076 s at most, over the
  90 closes made while seals ran. On 9995, under the void runs, it was 0.49 s at p90 and 1.95 s at most.
- Received to submitted: 0.001 s at p90, over 135 requests.
- Submitted to admitted: 0.018 s at p90. The three gaps over 0.1 s, at most 0.43 s, are `--window-requests 8`'s cap.
- Admitted to first token: 0.056 s at p50 and 0.53 s at p90.

**K per monitoring period** (`chain_record.py leak`). These are 6cd9's files at full traffic with its scan
(r20261026-035637-a739), every unit drawn and accepted. K uses:
- the run's parameters (`deployment.json`), the same on every active-warden run: B = 512, lanes = 4, r = 4, D = 21,
  T = 32 buckets of 100 ms, and egress_cells = 65;
- its traffic: 100 requests over 75 link-windows, 1.33 a link-window;
- the scan's class: B = 512, r = 4, lanes = 4, C = 47, IDB = 17 and KO = 64;
- every unit drawn.

The terms, in LEAK-LEAN's generator reading (note:network-accounting/20261025T2250Z-finding-warden-form-k):
- log2 #S_s = 0.
- log2(cap_c + 1) = log2 66 = 6.04 bits a connection. Here cap_c bounds `WardenSchedule.cells c`, the egress cells of
  connection c's lane, which is egress_cells. The endpoint's slot count isn't a term of the network certifier's
  schedule: it enters the compute audit's terms, which go into JointPeriod.
- The free inputs are the scan's `prog` rows: T·NP = 32 × 4,480 = 143,360 bits a link-window as a739 registered them,
  or 32 + KO·IDB = 1,120 bits a request if `prog` is an input only at its anchored entries. Each entry now carries a
  read bit as well (`ingress_scan.entry_layout`: a due slot no Program reads is refused), so a scan run today registers
  NP = 4,496: 143,872 bits a link-window, or 1,121 a request.
- No faulty term with every unit drawn, and D_j = 1.

The two forms:
- The pooled form (`HorizonLeakAccepted`, #4367) adds Fmax = ⌈ln(1/δ)⌉ units of b = NS + NV = 154,352 bits, and
  Fmax·log2(N + 1). Fmax is 5 at δ = 1% and 14 at δ = 10⁻⁶. The form is conditional: `hesc` and `hleak` await
  `SampledKSLive` and P2.
- The per-period form (`PeriodLeakAccepted`, #4326, #4349, #4373, #4370) has no δ term with every unit drawn.
- None of these theorems is on `main` yet.

The scale:
- A week, a month and a year of one machine and of 1,000 machines: memory-accounting's census convention, on the
  census's `rtx-pro-6000-server`.
- Each machine has one network certifier in front of all its compute, serving at the run's rate.
- The weights are Llama-3.2-1B's 1,235,814,400 BF16 parameters: 1.98·10^10 bits, 2.47 GB.

**`prog` as registered today (the record's K)**, bits, and the share of the weights:

| period | machines | K pooled, δ = 1% | K pooled, δ = 1e-6 | K per period | a link-window |
|---|---|---|---|---|---|
| week | 1 | 2.71·10^10 (1.37×) | 2.71·10^10 (1.37×) | 2.71·10^10 (1.37×) | 143,368 |
| week | 1,000 | 2.71·10^13 (1,370×) | 2.71·10^13 (1,370×) | 2.71·10^13 (1,370×) | 143,368 |
| month | 1 | 1.16·10^11 (5.87×) | 1.16·10^11 (5.87×) | 1.16·10^11 (5.87×) | 143,368 |
| month | 1,000 | 1.16·10^14 (5,873×) | 1.16·10^14 (5,873×) | 1.16·10^14 (5,873×) | 143,368 |
| year | 1 | 1.41·10^12 (71.5×) | 1.41·10^12 (71.5×) | 1.41·10^12 (71.5×) | 143,368 |
| year | 1,000 | 1.41·10^15 (71,456×) | 1.41·10^15 (71,456×) | 1.41·10^15 (71,456×) | 143,368 |

As registered today, K passes the size of the weights within one machine-week, so the record's K is vacuous for every
monitoring period.

**`prog` only at its anchored entries (conditional on the generator-form lowering, LEAN-49)**, bits, and the share of
the weights:

| period | machines | K pooled, δ = 1% | K pooled, δ = 1e-6 | K per period | a link-window |
|---|---|---|---|---|---|
| week | 1 | 2.85·10^8 (1.44%) | 2.86·10^8 (1.45%) | 2.84·10^8 (1.44%) | 1,501 |
| week | 1,000 | 2.84·10^11 (14.4×) | 2.84·10^11 (14.4×) | 2.84·10^11 (14.4×) | 1,501 |
| month | 1 | 1.22·10^9 (6.15%) | 1.22·10^9 (6.16%) | 1.22·10^9 (6.15%) | 1,501 |
| month | 1,000 | 1.22·10^12 (61.5×) | 1.22·10^12 (61.5×) | 1.22·10^12 (61.5×) | 1,501 |
| year | 1 | 1.48·10^10 (74.8%) | 1.48·10^10 (74.8%) | 1.48·10^10 (74.8%) | 1,501 |
| year | 1,000 | 1.48·10^13 (748×) | 1.48·10^13 (748×) | 1.48·10^13 (748×) | 1,501 |

This second table is what the generator-form lowering would give if it registers `prog` everywhere but its anchors
(ONE-TAP is on it). It is not the record's number.

K also rests on these conditions:
- The generator reading of the scan's statement, in which each unit's registered rows are its boundary values. That
  reading is a claim, with lean and Daniel.
- log2 #S_s = 0 only once the schedule's digest binds each connection's admission bucket and lane, and the server
  can't move an admission. Neither holds today.
- The scan proves link-window 0. A period's figures scale its per-link-window and per-request terms at the run's rate.
- At d951's light traffic (40 requests over 43 link-windows, 0.93 a link-window, its scan's every unit drawn) the
  anchored table's machine-week is 1.01% of the weights and its machine-year 52.2%; as registered today it is 1.37×
  a machine-week either way.
- The compute audit's free inputs don't enter K: its δ goes into JointPeriod's sum.

**`stubbed`.**
- NET-71: the units read the records through a stand-in until the rows form (#4452) and the generator-form lowering
  land. N2 is proved on 2982's records, not on an active-warden run's.
- StandInTie, on 2982: `IoTieRequest` doesn't read 2982's link format, so the endpoint step's tie runs in the clear on
  network-e2e's stand-ins.
  - r20261025-233707-ed7a (#4458) proved the tie at the served class on the active-warden link format: honest tied,
    forged refused, C-Flock `--zk` and Lean.
  - On d951 the tie is proved on the run's own ingress, which replaces the stand-in there: r20261026-013852-ecb2,
    art:046767802b0b4304494bab829d106543743a1fb006869fda367614aff91691a5. It ran `IoTieRequest_v1{B=512,L=4,R=4}`
    over all 43 link-windows and 40 of 40 connections, with C-Flock `--zk` on one GPU and Lean. Honest tied in 43 of
    43; a forged prompt digit was refused at link-window 0, at the registered read. 1,556,020 proof bytes a
    link-window. Its verifier still reads the tie's advice (prompt and output lengths, release) in the clear (NET-68).
- The clock certificate's opening is checked on the host: the certificates step's host audit opens the salted syncs in
  the clear (#4365's named gap).

**The theorem.** The run instantiates `NetTiming.Guarantees.PeriodLeakAccepted` (#4326, proved as
`NetTiming.SecurityProofs.PeriodLeakAccepted`), with `Claims.Sampling.KS`, `Claims.Flock.ListWithin` and
`trusted-host` as hypotheses. Over a period of n windows, each window's audit has `KSLinOn`'s bound,

Pr[window i accepted ∧ ¬RLeak δ] ≤ δ + ε_ks + ε_bind,

and the secrets of a covert sender whose every window meets `WindowGood` number at most 2^K, where

K = log2 #S_s + Σ_c log2(cap_c + 1) + Σ_i [#(freeInputs_i \ reg_i) + log2 #faultySets_i(δ) + β_i + k_i·log2 D_j].

`NetTiming.periodLeak_le_sum` (#4324) sums the windows' failures by a union bound:

Pr[every window accepted ∧ total > a + Σ_w b_w] ≤ Σ_w ε_w,

with a = log2 #S_s + Σ_c log2(cap_c + 1), charged once a period, b_w window w's term in the sum above, and
ε_w = δ + ε_ks + ε_bind. The report's `d1` lists, per window, whether its certificates, N2 and the scan accepted, and
Σ_w ε_w as n·(δ + ε_ks + ε_bind).

What it doesn't cover:

- The I/O tie. On 2982 IoTieRequest is checked in the clear, outside the proofs. On d951 it is proved (ecb2), but
  its verifier reads the tie's advice in the clear.
- `Claims.Sampling.KS` is a hypothesis until proofs' bridge restates it on `Flock.Guarantees.SampledKSLive`, which makes
  the error δ + ksAvgStrictZ + linkBoundZC.
- `hdata` (the active warden's arrivals are the framed egress of the network certifier's observations) and `Carries` are run conditions
  that no named assumption states yet.
- B_reg: the trace's values at reg are the registration's (proofs' #4253). No probability covers it yet.
- #(freeInputs \ reg) is not bounded by the advice bits until V3's padding, so b_w is vacuous (NET-68).
- No carry across windows: each window's terms stand alone, and a request that spans two windows is charged in each.
