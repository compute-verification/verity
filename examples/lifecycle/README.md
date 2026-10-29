# Lifecycle: the onsite and remote composites, stepped through

A minimal worked example of the two composite protocols, `verity/onsite/v0` and `verity/remote/v0`, run through the
whole lifecycle with the pieces that exist: from the hardware registration to the monitoring period's verdict. Its point
is the interfaces it exposes, listed in (b) and (c); where an interface doesn't exist yet, the example writes the smallest
version of it in `lifecycle.py`, and nothing under `verity/` or `Security/` changes.

```text
uv run python examples/lifecycle/run.py                          # one monitoring period of each composite, about 2 s on CPU
uv run python examples/lifecycle/run.py --only onsite --cheat off-node   # the developer deviating
verity example lifecycle                                         # the same, through the examples command
uv run --directory examples/lifecycle pytest tests/test_lifecycle.py     # honest, and every deviation refused
```

`lifecycle.py` holds the model, the site's `Layout`, the phases as functions and the two parties both composites share
(developer, verifier); `onsite.py` adds the network certifier, the memory challenger and the onsite verifier's subprotocols;
`remote.py` is the two-party deployment; `tests/test_lifecycle.py` runs both, honest and with every deviation.

**The proofs are a stand-in.** `prove(statement, witness) -> proof` returns the sampled proof block's openings, and
`verify(statement, proof) -> verdict` checks them against the commitments and re-evaluates the unit's circuit in the clear.
It is sound for the example and not zero-knowledge; Flock behind the ZK server replaces it (`verity/ml/flock/`,
`verity/core/service/flock/`). Everything else is the repository's own code, as (b) says.

## Terms

- **Registered circuit**: the circuit the developer claims, growing as it declares: C = B(a_1 ‖ … ‖ a_k), for a registered
  registered compiler B that is prefix-extending (the blocks before are unchanged).
- **Execution block**: what one declaration appends, B applied to a_k.
- **Execution phase**: the span of time one declaration covers. Monitoring period ⊃ execution phase ⊃ tick, and each
  execution phase has its one verification phase; a monitoring period is three of them.
- **Declaring**, and one act of it, **a declaration**: the developer reports the **advice** a_k, B appends the
  execution block, and the block's values are committed before any coins are flipped.

## (a) The lifecycle as run

Every party is a sans-IO party of the deployment layer (`verity/core/deploy/`), run by `runtime.run` in simulated mode:
in one process, on seeded coin sources and virtual clocks, with the manifest's channels and latencies. The model is two
4x4 int8 matmuls (`forward`: y = sat(W2 relu(W1 x) >> 7) >> 7) with a 16-bit running state; one request is one proof
block, the Boolean Definition `LifecycleRequest` (5,200 AND gates), and an execution block is `LifecycleBlock(N)`: N units
chained through the state wire. The monitoring period is three execution phases of 3, 4 and 2 requests.

1. **Hardware registration** (`Hardware`; `check_registration(layout, registration)`). The site is one `Layout`: cards
   by digest (`Card`, in `Accounting.Device`'s terms), a tree of locations with their owners (`Loc`: the site, hosts,
   devices naming a card, memory leaves with their tier and bytes, and the auditor's certifiers and challengers), and
   links with their ends, β and latency. A certifier sits on an uplink by splitting it: host to certifier, certifier to
   the outside. The layout is folded from one append-only log of `Registration` entries, each chained to the log's head
   and checked as a whole against its prefix's layout, refused by name: `previous`, `duplicate-id`, `unknown-ref`,
   `kind`, `card-digest` (a card that isn't the auditor's data sheet, `SHEETS`), `capacity`, `owner` (a certifier or
   challenger that isn't one of the auditor's own devices with its key), and onsite `uncertified` (a boundary link no
   certifier splits) and `challenger-reach`. After declaration 1 a later entry adds gpu2 with its HBM, and a disk.
2. **Registration** (`Register`; `register`, `check_baseline`). B's descriptor (format, advice cap of 32 bits a
   declaration, its static placement: gpu0 and gpu1, gpu2 too from declaration 2, theta in hbm0), and the registered information:
   theta's 8 rows and the initial state, each committed as hm96-sha512 leaves in a frame-v3-sha512 tree under B's
   digest. Only roots are public. Onsite, the verifier then starts the certifier.
3. **Operation** (`operate`). Remote, the developer serves its own load. Onsite, the certifier is the outside world's only
   way in: it carries each request in (`Ingress`) and its response out (`Egress`) and commits to each frame under its own
   salts. The developer's tape decides what only run time decides: the batching.
4. **Declaration k** (`Declare`; `declare`, `check_declaration`). The developer reports the advice a_k (a unit
   count, then each unit's batch), and B appends the execution block, padded to PoUW's grid (`pad(real + 1)`, one
   filler at least), and places it (`placement`, `Requirement.Placement`'s shape: each unit's device leaf, theta's memory
   leaf, and the layout's head they are placed on). The verifier refuses by name: `advice-cap`, `advice` (B doesn't
   parse it), `prefix` (it doesn't extend the circuit so far), `placement-off-layout` (another head, or a leaf no entry
   holds yet), `placement-auditor`, `placement-kind`, `placement-capacity`, `record`. The record binds the previous
   record, the placement, the block's Program digest and its values' root, all before any coin.
5. **Audit of declaration k.** After the `Declare`, the verifier flips its live coins and sampled proofs' `draw_first`
   picks subset:K′ of the block's units, at PoUW's `k_prime` for ε = 1/2, δ = 1/4 (`Draw`). The developer proves each
   sampled proof block (`Proofs`: the stand-in), whose statement reads three commitments: the block's row, the unit's input state
   (the row before, the previous block's last row, or the registered initial state), and theta. Onsite, the certifier's signed view
   (`View`: two LMS signatures over `lms.certificate_message`, under the key its registered location carries) gives the
   count and the frame commitments each sampled proof block's I/O gates are held to; the memory challenger's round (`Open`,
   `Challenge`, `Answer`, `Report`) picks 2 theta rows on its own coins and times the answer on its own clock. PoUS holds
   the openings to theta's root; the bandwidth audit holds the time to twice the challenger's link latency plus the rows'
   bits at the rate of the device theta is placed on, both read from the layout (`theta_reach`). Each subprotocol answers
   in `Core.Protocol`'s shape, and the composite is the composition's rule and binding over exactly its subprotocols.
6. **Accounting** (`account`). The chain from the registration through every record, each declaration's composite, the
   period's verdict by the same rule, the credited AND gates (PoUW's grid point at or below the real units, fillers
   credited nothing) against the compute the layout held in each execution phase, the advice bits charged as leakage,
   the pooled bound, each subprotocol's projection of the layout (`Layout.accounting`), and the same scaled to a week, a
   month and a year.

The first registration and declaration 1, onsite, and the period, remote (`run.py` prints every phase of both):

```text
1 hardware     registration: entry 72aec7485cfec4a3..., after e3d2030c65fc3ac4...
               cards      sim-int8-a, sim-int8-b
               locations  site0 (site), node0 (host in site0), gpu0 (device sim-int8-a in node0), hbm0 (memory hbm 64
                          GiB in gpu0), gpu1 (device sim-int8-b in node0), hbm1 (memory hbm 128 GiB in gpu1), dram0
                          (memory dram 512 GiB in node0), nvme0 (memory disk 1024 GiB in node0), cert0 (certifier
                          auditor in site0), chal0 (challenger auditor in site0)
               links      uplink0 node0-cert0, uplink0-out cert0-outside, chal0-link chal0-node0
declaration 1
  4 declare    advice 00000011000000010010 (20 bits charged); block of 4 units (1 filler) on gpu0, gpu1, gpu0
  5 audit      drew subset:3 of 4: units [0, 2, 3]
               verity/sampled-proofs/one-stage/v1       accept
               verity/pouw/v1                           accept
               verity/certifier/v0                      accept
               verity/pous/p2/v4/run                    accept
               verity/remote-bandwidth/v0               accept
               composite verity/onsite/v0               accept
6 accounting
  period       3 declarations: accept
  credited     46800 of 62400 appended AND gates; 288 multiply-adds of the 4.99e+13 the registered devices could do
  bound        except with probability 32/3125 = (2/5)^5, fewer than 1/2 of the period's 46800 credited AND gates
               (9 units) are in wrong units; a declaration alone: delta 1/4 at eps 1/2
  projections  a declaration period on the final layout: PoUW 3 devices, 1.92e+13 multiply-adds, PoUS hbm 320 GiB,
               dram 512 GiB, disk 2048 GiB, bandwidth 9.60e+14 device-memory bits, 6.00e+15 boundary bits,
               certifier: boundary ['uplink0'], certified [], 6.00e+15 bits unseen
  week         10080 declarations of 60 s like these: 1.57e+08 credited AND gates, share 5.00e-12 of the layout's
               1.94e+17 multiply-adds; delta 10^-6017 at eps 1/2
```

The deviations, one or more a phase, each refused or rejected (`tests/test_lifecycle.py`; `--cheat NAME`):

| cheat | phase | caught by |
| --- | --- | --- |
| `hardware`: the later entry names gpu0 again | hardware registration | refused `duplicate-id` |
| `builder`: B's descriptor is of a format the verifier doesn't run | registration | refused `builder` |
| `placement`: B's plan puts a unit on gpu2 before its entry | operation | refused `placement-off-layout` |
| `unreported`: one request served and left out of the advice | operation | onsite: the certifier's count; remote: accepted, uncredited |
| `advice-cap`: 33 bits of advice | declaration | refused `advice-cap` |
| `recommit`: a second commitment after the sample | declaration | refused `recommit` |
| `wrong-value`: every output of block 2 off by one | audit | sampled proofs, and PoUW, which reads it (onsite also the I/O check) |
| `io-mismatch`: the wire carries other outputs than the committed ones | audit | the certifier's frames, onsite |
| `lost-theta`: an answer with a row that isn't theta's | audit | PoUS, onsite |
| `off-node`: theta fetched from elsewhere, 3 ms a round trip | audit | the bandwidth audit, onsite |
| `fork`: record 2 names the registration, not record 1 | accounting | the chain; every subprotocol accepts |

`test_check_registration` and `test_check_placement` hold every refusal of phase 1 and of the placement to a case.

## (b) Interfaces

| name | caller | in | out | module |
| --- | --- | --- | --- | --- |
| `Card`, `Loc`, `Link`, `Registration` | developer | the site's hardware | one entry of the log, chained to its head | missing: written in the example |
| `Layout` (`genesis`, `add`, `device`, `boundary`, `accounting(ticks)`) | everyone | the log's entries | the site; each subprotocol's accounting | missing: written in the example (pieces: `Security/Definitions/Accounting/Physical.lean`'s `Hardware`, `Accounting/Device.lean`, the deploy manifest's machines and links) |
| `check_registration(layout, registration, auditor)` | verifier | the layout so far, an entry, the auditor's own devices and keys | the layout it folds to, or a refusal by name | missing: written in the example |
| `Builder` (`descriptor`, `of`, `digest`, `grow`) | developer, verifier | a_1 ‖ … ‖ a_k | the declaration blocks | missing: written in the example |
| `placement`, `check_placement` | developer, verifier | B, a block, the layout's head | `Requirement.Placement`'s data, or a refusal by name | missing: written in the example |
| `register`, `check_baseline` | developer, verifier | product, theta, B, state0 | `Baseline` (roots and B's descriptor) | missing: written in the example |
| `operate` | developer | requests, theta, state, tape | `Served` | the integration's (here the toy model) |
| `declare`, `check_declaration` | developer, verifier | advice, served values, layout, previous record | `Declaration`, `Block` | missing: written in the example |
| `commit`, `opens`, `committed` | developer, verifier | rows and salts; an opening | a root; whether it opens | `verity/core/protocols/sampled_proofs/one_stage/registered.py` over `verity.ml.commitments.hm96` |
| `UNIT`, `BLOCK`, `block_digest(n)` | B | — | Boolean Definitions; a Program digest | `verity.ml.circuits` (`boolean.trace`, `defs.CompositeDefinition`, `codec`) |
| `evaluate_unit` | the stand-in verifier | theta, state, x | y, state′ | `verity.ml.circuits.evaluation.evaluate_words` |
| `draw(coins, block)`, `draw_size` | verifier | live coins | the sampled proof blocks | `verity/core/protocols/sampled_proofs/coins.py` (`draw_first`), `verity/core/protocols/pouw/window.py` (`bucket`, `k_prime`) |
| `prove(statement, witness)` | ZK server | a `Statement`, the openings | a proof | STAND-IN for Flock |
| `verify(statement, proof)` | verifier | a `Statement`, a proof | each check's reason, or none | STAND-IN for Flock's verifier |
| `sampled_proofs`, `pouw_window`, `OnsiteVerifier.answers` | verifier | one declaration's checks, view, report | `Core.Protocol` outcomes | missing as per-declaration functions: written in the example |
| `compose(protocol, answers)` | verifier | the subprotocols' outcomes | the composite verdict and binding | `verity/core/protocols/composition/tests/reference.py` (`verdict`, `bind`, `ONSITE`, `REMOTE`) |
| `account(baseline, declarations, lines, layout)` | verifier | the period's records and verdicts | period verdict, credit, bound, projections, scaling | missing: written in the example |
| `Certifier`, `certified(I, view, direction)` | network certifier | frames seen | a signed view | `verity/ml/crypto/lms.py` (`certificate_message`, `sign_at`, `verify`) |
| `Challenger`, `theta_reach(layout, store)` | memory challenger, verifier | an open round; the layout | PoUS and bandwidth report; rate and latency | missing: written in the example |
| `Developer`, `Verifier`, manifests | runtime | params, capabilities, messages | actions | `verity/core/deploy/` (`party`, `manifest`, `runtime.run`) |

## (c) What this exposes

- **The registered circuit has no one digest, and Lean's `Declaration` holds one circuit.** Each execution block has its
  own Program digest, and the record chain binds them in order; `codec.program_digest` of B(a_1 ‖ … ‖ a_k) as one Program
  is not prefix-incremental. `Security/Definitions/Core/Declaration.lean`'s `Declaration` is the registered circuit so far,
  and its file says one Program is the circuit of record: d.D, one `Circuit`, is here every execution block composed
  along the state wire (block k's state input is block k − 1's state output, not an input gate), which the spec has no
  operation for; d.θ, the registered information, is theta's and the initial state's input gates. `Baseline` is d before any block,
  and each `Declaration` record extends d.D by one block and d.θ by nothing: its requests are novel input gates.
- **State crosses execution blocks as a committed value.** A sampled proof block at the start of block k opens block k − 1's
  last row (or the registered initial state), so one unit's statement reads three commitments. `Core.Protocol`'s answer carries
  one commitment: the cross-block read and the chain live outside every subprotocol, in `check_declaration` and
  `account`, and `fork` passes every subprotocol. Nothing composes declarations into a monitoring period; the example
  applies the composition's rule over them.
- **Where prefix-extending B breaks.** (i) PoUW's grid pads every block (`pad(real + 1)`), so the registered circuit is
  full of filler units, and their count depends on how the work was cut into declarations. (ii) Work that straddles an
  execution phase's boundary: a batch spanning two, or a decode whose KV cache carries into the next; the example cuts
  at request boundaries. (iii) A registered compiler that schedules globally isn't prefix-extending. (iv) B here is Python with a
  descriptor digest, not a registered circuit with a length cap, and its advice validation is its own code.
- **One `Layout`, folded from a log.** The site is `Accounting.Hardware`'s shape as registered data: locations are its
  `L`, links its `E` with `ends` and `β`, the certifier leaves its `cert`, the auditor-owned leaves its `trusted`, and
  `Layout.device` is `Accounting.Device`'s view of a device leaf. Each record names the head its block was placed on,
  so a later entry changes nothing already placed. Not here: the `tiers`, `processors` and `retroactive` refusals,
  removing or retiring hardware (a device that fails has no path: B's plan can't drop it without a new B), and a
  challenger's latency limit.
- **Placement is static, B's, and names the layout.** It isn't in the advice: B's plan and store name leaves, and each
  record's placement (units to device leaves, theta to a memory leaf) names the layout's head; a leaf no entry holds yet
  is refused. `Requirement.Placement`'s `ticks` and the onsite `placement-boundary` (the circuit's external I/O on a
  certified uplink of its unit, `NetTiming.Layout.Fits`) are missing: the certifier's view names its registered link,
  and nothing maps a unit's I/O gates to one.
- **Accounting is the layout's projections.** PoUW's compute, PoUS's memory by tier, the bandwidth audit's rates and the
  certifier's boundary are each `Layout.accounting`; the challenger's round reads its link's latency and theta's
  device's rate from it, and the verifier reads the certifier's key from its registered location, checked against the
  auditor's own (`owner`). A later entry raises capacity from the next execution phase on. The toy's share of it
  (5 × 10^-12) says only that the model is tiny; over all compute the scaling is a parameter the example doesn't take.
- **Remote can't see unreported work.** Without a view of the system's I/O, work left out of the advice is
  uncredited, not detected: remote's credit is a lower bound, and its projection says how many boundary bits no
  certifier sees. Onsite, the certifier's count catches it.
- **The period bound is computed, not proved.** A declaration alone gives δ = 1/4; pooled over the monitoring period's
  independent samples, a wrong share ε of the credited work escapes with probability at most q_max^⌈εN⌉ (q = 1 − K′/n, N
  credited units): 1.0 % over the three declarations, 10^-6017 over a week of declarations like these. The theorem from
  a declaration's audit to the period's is the layer the modular math still needs.
- **Per-declaration subprotocol checks don't exist yet.** PoUW's `core_check` and PoUS's and the bandwidth audit's verifiers take
  whole-run cases; the example writes one answer each a declaration. Only sampled proofs samples from the execution
  block; the memory challenger samples theta rows.
- **The advice is charged but the toy's circuit doesn't depend on it beyond the count.** Batching enters B only as
  validation; a batched matmul would make B's output depend on it.
- **Message kinds.** `Proofs` is private only because the stand-in's proof is an opening; Flock's proof would be
  public. The certifier's frame openings (`Mark`) and the memory challenger's answers are witness.

## (d) Proposed homes

In a noun-based layout, each part of the example goes where its noun is:

- `verity/registration`: `Layout`, its entries, `check_registration` and the projections; the registered information (`register`,
  `check_baseline`), B's registered descriptor and `grow`, and `placement`; `declare` and `check_declaration` too, as
  what is registered over time.
- `verity/circuits`: the units, the execution blocks and the registered circuit as blocks chained by state wires, with a
  digest of the chain (today's `verity.ml.circuits`).
- `verity/commitments`: `commit`, `opens`, the row and frame commitments (today split between `one_stage.registered`,
  `verity.ml.commitments` and the certifier's own).
- `verity/flock`: `prove` and `verify`, replacing the stand-in, plain or zero-knowledge behind the ZK server.
- `verity/protocols/onsite` and `verity/protocols/remote`: the parties and manifests, each composite's per-declaration
  subprotocol checks and `account`, over the composition's rule.
- The model, its load and `operate` stay with the integration that serves them (`integrations/`).
