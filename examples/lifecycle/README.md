# Lifecycle: the onsite and remote composites, stepped through

A minimal worked example of the two composite protocols, `verity/onsite/v0` and `verity/remote/v0`, run through the whole
lifecycle with the pieces that exist: from the hardware registration to the monitoring period's verdict. Its point is the
interfaces it exposes, listed in (b) and (c); where an interface doesn't exist yet, the example writes the smallest version
of it in `lifecycle.py`, and nothing under `verity/` or `Security/` changes.

```text
uv run python examples/lifecycle/run.py                         # one monitoring period of each composite, about 1 s on CPU
uv run python examples/lifecycle/run.py --only remote --cheat fork
verity example lifecycle
```

`lifecycle.py` holds the model, the phases as functions and the two parties both composites share (developer, verifier);
`onsite.py` adds the network certifier, the memory challenger and the onsite verifier's components; `remote.py` is the
two-party deployment; `tests/test_lifecycle.py` runs both, honest and with every deviation in `lifecycle.CHEATS`.

**The proofs are a stand-in.** `prove(statement, witness) -> proof` returns the drawn unit's openings, and
`verify(statement, proof) -> verdict` checks them against the commitments and re-evaluates the unit's circuit in the clear.
It is sound for the example and not zero-knowledge; C-Flock behind the firewall replaces it (`verity/ml/flock/`,
`verity/core/service/flock/`). Everything else is the repository's own code, as (b) says.

## (a) The lifecycle as run

Every party is a sans-IO party of the deployment layer (`verity/core/deploy/`), run by `runtime.run` in simulated mode:
in one process, on seeded coin sources and virtual clocks, with the manifest's channels and latencies. The model is two
4x4 int8 matmuls (`forward`: y = sat(W2 relu(W1 x) >> 7) >> 7) with a 16-bit running state; one request is one proof unit,
the Boolean Definition `LifecycleRequest` (5,200 AND gates), and a declaration block is `LifecycleBlock(N)`: N units chained
through the state wire. A monitoring period here is three declaration periods of 3, 4 and 2 requests.

1. **Hardware registration** (`Hardware`; `check_registration(layout, registration)`). The developer registers the site as
   one `Layout`: nodes, devices by SKU, stores, links with their rates and latencies, uplinks, and where the auditor's
   devices sit (onsite: the network certifier on the uplink, the memory challenger at the end of its link to the node).
   The verifier merges it into the layout so far or refuses by name: `duplicate`, `data-sheet` (a SKU the auditor has no
   sheet for), `node`, `link`, `auditor-device`, and onsite `uplink-certifier`. A device's limits are the auditor's data
   sheets (`SHEETS`, in `Accounting.Device`'s terms: `tc`, `mem`, `bw`, `chip`), never the developer's. After declaration
   1 a later registration adds gpu2 and a store.
2. **Registration** (`Register`; `register`, `check_baseline`). B's descriptor (format, advice cap of 32 bits a
   declaration, and its static placement plan: gpu0 and gpu1, gpu2 too from declaration 2), and the baseline: theta's 8
   rows and the initial state, each committed as hm96-sha512 leaves in a frame-v3-sha512 tree under B's digest. Only roots
   are public. Onsite, the verifier then starts the certifier.
3. **Operation** (`operate`). Remote, the developer serves its own load. Onsite, the certifier is the outside world's only
   way in: it carries each request in (`Ingress`) and its response out (`Egress`) and commits to each frame under its own
   salts. The developer's tape decides what only run time decides: the batching.
4. **Declaration k** (`Declare`; `declare`, `check_declaration`). The developer reports the declared advice a_k (a unit
   count, then each unit's batch), and B appends the declaration block to the declared circuit C = B(a_1 ‖ … ‖ a_k),
   padded to PoUW's grid (`pad(real + 1)`, one filler at least). The verifier refuses by name: `advice-cap`, `advice` (B
   doesn't parse it), `prefix` (it doesn't extend the circuit so far), `placement` (B puts a unit on a device no
   registration holds yet), `record`. The record binds the previous record, the layout's digest, the block's Program
   digest and its values' root, all before any coin.
5. **Audit of declaration k.** After the `Declare`, the verifier draws its live coins and sampled proofs'
   `draw_first` picks subset:K′ of the block's units, at PoUW's `k_prime` for ε = 1/2, δ = 1/4 (`Draw`). The developer
   proves each drawn unit (`Proofs`: the stand-in), whose statement reads three commitments: the block's row, the unit's
   input state (the row before, the previous block's last row, or the baseline's), and theta. Onsite, the certifier's
   signed view (`View`: two LMS signatures over `lms.certificate_message`) gives the count and the frame commitments each
   drawn unit's I/O gates are held to, and the memory challenger's round (`Open`, `Challenge`, `Answer`, `Report`), once the
   block is committed, picks 2 theta rows on its own coins and times the answer on its own clock: PoUS holds the openings
   to theta's root, and the bandwidth audit holds the time to twice the link's latency plus the rows' bits at the slowest
   device's `bw`. Each component answers in `Core.Protocol`'s shape, and the composite is the composition's rule and
   binding over exactly the product's components.
6. **Accounting** (`account`). The chain from the registration through every record, each declaration's composite, the
   period's verdict by the same rule, the credited AND gates (PoUW's grid point at or below the real units, fillers
   credited nothing) against the capacity of the devices registered in each declaration period, the advice bits charged
   as leakage, the pooled bound, and the same scaled to a week, a month and a year.

One declaration and the period, remote (onsite adds three components to each declaration's answers, all accepting):

```text
declaration 2
  3 operation  4 requests served, batches [0, 1, 2, 2]
  4 declare    advice 000001000000000100100010 (24 bits charged); block of 5 units (1 filler) on gpu0, gpu1, gpu2, gpu0
  5 audit      drew subset:3 of 5: units [1, 2, 4]
               verity/sampled-proofs/one-stage/v1       accept
               verity/pouw/v1                           accept
               composite verity/remote/v0               accept
6 accounting
  period       3 declarations: accept
  credited     46800 of 62400 appended AND gates; 288 multiply-adds of the 4.99e+13 the registered devices could do
  advice       60 bits charged as leakage
  bound        except with probability 32/3125 = (2/5)^5, fewer than 1/2 of the period's 46800 credited AND gates
               (9 units) are in wrong units; a declaration alone: delta 1/4 at eps 1/2
  week         10080 declarations of 60 s like these: 1.57e+08 credited AND gates, share 5.00e-12 of the layout's
               1.94e+17 multiply-adds; delta 10^-6017 at eps 1/2
```

The deviations, one or more a phase, each refused or rejected (`tests/test_lifecycle.py`):

| cheat | phase | caught by |
| --- | --- | --- |
| `hardware`: a later registration names gpu0 again | hardware registration | refused `duplicate` |
| `builder`: B's descriptor is of a format the verifier doesn't run | registration | refused `builder` |
| `placement`: B's plan puts a unit on gpu2 before its registration | operation | refused `placement` |
| `unreported`: one request served and left out of the advice | operation | onsite: the certifier's count; remote: accepted, uncredited |
| `advice-cap`: 33 bits of advice | declaration | refused `advice-cap` |
| `recommit`: a second commitment after the draw | declaration | refused `recommit` |
| `wrong-value`: every output of block 2 off by one | audit | sampled proofs (and onsite the I/O check) |
| `io-mismatch`: the wire carries other outputs than the committed ones | audit | the certifier's frames, onsite |
| `lost-theta`: an answer with a row that isn't theta's | audit | PoUS, onsite |
| `off-node`: theta fetched from elsewhere, 3 ms a round trip | audit | the bandwidth audit, onsite |
| `fork`: record 2 names the registration, not record 1 | accounting | the chain; every component accepts |

## (b) Interfaces

| name | caller | in | out | module |
| --- | --- | --- | --- | --- |
| `Layout`, `Layout.capacity(ticks)` | everyone | nodes, devices, stores, links, auditor devices | the site; what it can do in a period | missing: written in the example (pieces: `Security/Definitions/Accounting/Device.lean`, the deploy config's machines and links) |
| `check_registration(layout, registration, auditor)` | verifier | the layout so far, a `Layout` of additions, the auditor's devices | merged `Layout`, or a refusal by name | missing: written in the example |
| `Builder` (`descriptor`, `of`, `digest`, `grow`) | developer, verifier | a_1 ‖ … ‖ a_k | the declaration blocks, placed | missing: written in the example |
| `register`, `check_baseline` | developer, verifier | product, theta, B, state0 | `Baseline` (roots and B's descriptor) | missing: written in the example |
| `operate` | developer | requests, theta, state, tape | `Served` | the integration's (here the toy model) |
| `declare`, `check_declaration` | developer, verifier | advice, served values, layout, previous record | `Declaration`, `Block` | missing: written in the example |
| `commit`, `opens`, `committed` | developer, verifier | rows and salts; an opening | a root; whether it opens | `verity/core/protocols/sampled_proofs/one_stage/registered.py` over `verity.ml.commitments.hm96` |
| `UNIT`, `BLOCK`, `block_digest(n)` | B | — | Boolean Definitions; a Program digest | `verity.ml.circuits` (`boolean.trace`, `defs.CompositeDefinition`, `codec`) |
| `evaluate_unit` | the stand-in verifier | theta, state, x | y, state′ | `verity.ml.circuits.evaluation.evaluate_words` |
| `draw(coins, block)`, `draw_size` | verifier | live coins | the drawn units | `verity/core/protocols/sampled_proofs/coins.py` (`draw_first`), `verity/core/protocols/pouw/window.py` (`bucket`, `k_prime`) |
| `prove(statement, witness)` | firewall | a `Statement`, the openings | a proof | STAND-IN for C-Flock |
| `verify(statement, proof)` | verifier | a `Statement`, a proof | each check's reason, or none | STAND-IN for C-Flock's verifier of record |
| `sampled_proofs`, `pouw_window`, `OnsiteVerifier.answers` | verifier | one declaration's checks, view, report | `Core.Protocol` outcomes | missing as per-declaration functions: written in the example |
| `compose(protocol, answers)` | verifier | the components' outcomes | the composite verdict and binding | `verity/core/protocols/composition/tests/reference.py` (`verdict`, `bind`, `ONSITE`, `REMOTE`) |
| `account(baseline, declarations, lines, layout)` | verifier | the period's records and verdicts | period verdict, credit, share, bound, scaling | missing: written in the example |
| `Certifier`, `certified(I, view, direction)` | network certifier | frames seen | a signed view | `verity/ml/crypto/lms.py` (`certificate_message`, `sign_at`, `verify`) |
| `Challenger` | memory challenger | an open round | PoUS and bandwidth report | missing: written in the example |
| `Developer`, `Verifier`, manifests | runtime | params, capabilities, messages | actions | `verity/core/deploy/` (`party`, `manifest`, `runtime.run`) |

## (c) What this exposes

- **The declared circuit has no one digest.** Each block has its own Program digest, and the record chain binds them in
  order. `codec.program_digest` of B(a_1 ‖ … ‖ a_k) as one Program is not prefix-incremental, so "the circuit so far" is
  a list of block digests plus a chain, not one Program. Lean's `Declaration` (`Security/Definitions/Core/Declaration.lean`)
  holds one `Circuit` D; d.D here is the blocks composed along the state wire, which the spec has no operation for, and
  d.θ is the gates theta and the initial state are committed to (the Program's `registered` and `theta` inputs).
- **State crosses blocks as a committed value.** A drawn unit at the start of block k opens block k − 1's last row (or the
  baseline's state), so one unit's statement reads three commitments. `Core.Protocol`'s answer carries one commitment and
  one evidence: the cross-block read and the chain live outside every component, in `check_declaration` and `account`.
  `fork` passes every component and only accounting catches it. The composition composes one declaration's components;
  nothing composes declarations into a period. The example applies the same rule over them.
- **Where prefix-extending B breaks.** (i) PoUW's grid pads every block (`pad(real + 1)`), so the declared circuit is
  full of filler units, and their count depends on how the work was cut into declarations. (ii) Work that straddles a
  boundary: a batch spanning two declaration periods, or a decode whose KV cache carries into the next; the example cuts
  at request boundaries, and its carried state is one word. (iii) A builder that schedules globally (rebalancing load,
  placement by what came later) isn't prefix-extending. (iv) B here is Python with a descriptor digest, not itself a
  registered circuit with a length cap, and its advice validation (consecutive batches of at most 2) is B's own code.
- **Placement is static, and the layout grows.** B's plan names devices by declaration, checked at each declaration
  against the layout as registered then: B may name a device before its registration, and nothing lands on it until it is
  registered (`placement`). Removing a device, or one failing, has no path: B's plan can't drop it without a new B.
- **What registration must hold.** The layout (devices by SKU, stores, links with rates and latencies, uplinks, where each
  auditor device sits); B's descriptor (format, cap, plan, grid); theta's and the initial state's commitments under B's
  digest. Each record names the layout digest it was placed on. The auditor's own device keys (the certifier's public
  key) are verifier parameters here, not registered. There is no registration of what B's advice may say beyond its cap.
- **Remote can't see unreported work.** Without a view of the system's I/O, work left out of the advice is uncredited,
  not detected: remote's credit is a lower bound. Onsite, the certifier's count catches it.
- **The period bound is computed, not proved.** A declaration alone gives δ = 1/4, which looks weak; pooled over the
  period's independent draws, a wrong share ε of the credited work escapes with probability at most q_max^⌈εN⌉ (q = 1 −
  K′/n): 1.0 % over the three declarations, 10^-6017 over a week of declarations like these. The theorem from a
  declaration's audit to the period's is the layer the modular math still needs.
- **Capacity is the layout's.** Per-period capacity comes from the registered devices' data sheets, per declaration
  period, so a later registration raises it from then on. The toy's share of it (5 × 10^-12) says only that the model is
  tiny; over all compute (a datacenter's, the world's) the scaling is a parameter the example doesn't take.
- **Per-declaration components don't exist yet.** PoUW's `core_check` and PoUS's and the bandwidth audit's verifiers take
  whole-run cases; the example writes one answer each a declaration. Only sampled proofs draws from the declaration block;
  the memory challenger draws theta rows, and each subprotocol's sampling block is not yet its own draw.
- **The advice is charged but the toy's circuit doesn't depend on it beyond the count.** Batching enters B only as
  validation. A batched matmul would change the block's shape, and then B's output depends on the advice, as it should.
- **Message kinds.** `Proofs` is private only because the stand-in's proof is an opening; C-Flock's proof would be
  public. The certifier's frame openings (`Mark`) and the memory challenger's answers are witness, as they should be.

## (d) Proposed homes

In a noun-based layout, each part of the example goes where its noun is:

- `verity/registration`: `Layout` and its capacity, `check_registration`, the baseline (`register`, `check_baseline`),
  and B's registered descriptor and `grow`; `declare` and `check_declaration` too, as what is registered over time.
- `verity/circuits`: the units, the blocks and the declared circuit as blocks chained by state wires, with a digest of
  the chain (today's `verity.ml.circuits`).
- `verity/commitments`: `commit`, `opens`, the row and frame commitments (today split between `one_stage.registered`,
  `verity.ml.commitments` and the certifier's own).
- `verity/flock`: `prove` and `verify`, replacing the stand-in, plain or zero-knowledge behind the firewall.
- `verity/protocols/onsite` and `verity/protocols/remote`: the parties and manifests, each composite's per-declaration
  components and `account`, over the composition's rule.
- The model, its load and `operate` stay with the integration that serves them (`integrations/`).
