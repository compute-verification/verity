# Lifecycle: the remote composite, stepped through

A minimal worked example of the remote composite protocol, `verity/remote/v0` (sampled proofs and PoUW), run through the
whole lifecycle with the pieces that exist: from the hardware registration to the monitoring period's verdict. Its point
is the interfaces it exposes, listed in (b) and (c); where an interface doesn't exist yet, the example writes the smallest
version of it in `lifecycle.py`, and nothing under `verity/` or `Security/` changes.

```text
uv run python examples/lifecycle/run.py                  # one monitoring period, about 2 s on CPU
uv run python examples/lifecycle/run.py --cheat fork     # the developer deviating, one of lifecycle.CHEATS
verity example lifecycle                                 # the same, through the examples command
uv run --directory examples/lifecycle pytest tests/test_lifecycle.py   # honest, and every deviation refused
```

`lifecycle.py` holds the model, the phases as functions and the two parties (developer, verifier); `remote.py` is the
two-party deployment; `tests/test_lifecycle.py` runs it, honest and with every deviation in `lifecycle.CHEATS`.

**Missing: the onsite composite** (`verity/onsite/v0`): its network certifier on the uplink, its memory challenger and
their subprotocols (PoUS, the bandwidth audit, network certification). The shared parties have its hooks (the verifier's
`registered`, `declared`, `answers` and `frames`, and the onsite checks in `check_registration`), and nothing runs them yet.

**The proofs are a stand-in.** `prove(statement, witness) -> proof` returns the drawn unit's openings, and
`verify(statement, proof) -> verdict` checks them against the commitments and re-evaluates the unit's circuit in the clear.
It is sound for the example and not zero-knowledge; C-Flock behind the firewall replaces it (`verity/ml/flock/`,
`verity/core/service/flock/`). The verdict is the composition's Python mirror
(`verity/core/protocols/composition/tests/reference.py`), not the Lean `remote check` (`Core.Remote.protocol`), over the
example's per-declaration stand-ins for `SampledProofs.OneStage.protocol` and `PouwCheck.protocol`. So the example
exercises the lifecycle and the composition rule, not the verifier. Everything else is the repository's own code, as (b) says.

## Terms

- **Declared circuit**: the circuit the developer claims, growing as it declares: C = B(a_1 ‖ … ‖ a_k), for a registered
  builder B that is prefix-extending (the blocks before are unchanged).
- **Declaration block**: what one declaration appends, B applied to a_k.
- **Declaration period**: the span of time one declaration covers. Monitoring period ⊃ declaration period ⊇ sampling
  period ⊃ tick; here a declaration period is one sampling period, and a monitoring period is three of them.
- **Declaring**, and one act of it, **a declaration**: the developer reports the **declared advice** a_k, B appends the
  declaration block, and the block's values are committed before the period's audit coins.

## (a) The lifecycle as run

Every party is a sans-IO party of the deployment layer (`verity/core/deploy/`), run by `runtime.run` in simulated mode:
in one process, on seeded coin sources, with the manifest's channels. The model is two 4x4 int8 matmuls (`forward`:
y = sat(W2 relu(W1 x) >> 7) >> 7) with a 16-bit running state; one request is one proof block, the Boolean Definition
`LifecycleRequest` (5,200 AND gates), and a declaration block is `LifecycleBlock(N)`: N units chained through the state
wire. The monitoring period is three declaration periods of 3, 4 and 2 requests.

1. **Hardware registration** (`Hardware`; `check_registration(layout, registration)`). The developer registers the site as
   one `Layout`: nodes, devices by SKU, stores, links with their rates and latencies, uplinks, and where the auditor's
   devices sit (remote: none). The verifier merges it into the layout so far or refuses by name: `site`, `duplicate`,
   `data-sheet` (a SKU the auditor has no sheet for), `node`, `link`, `auditor-device`, and onsite `uplink-certifier`. A
   device's limits are the auditor's data sheets (`SHEETS`, in `Accounting.Device`'s terms: `tc`, `mem`, `bw`, `chip`),
   never the developer's. After declaration 1 a later registration adds gpu2 and a store.
2. **Registration** (`Register`; `register`, `check_baseline`). B's descriptor (format, advice cap of 32 bits a
   declaration, and its static placement plan: gpu0 and gpu1, gpu2 too from declaration 2), and the registered information: theta's 8
   rows and the initial state, each committed as hm96-sha512 leaves in a frame-v3-sha512 tree under B's digest. Only roots
   are public.
3. **Operation** (`operate`). The developer serves its own load. Its tape decides what only run time decides: the
   batching. Placement is not run time's: B decides it from its plan.
4. **Declaration k** (`Declare`; `declare`, `check_declaration`). The developer reports the declared advice a_k (a unit
   count, then each unit's batch), and B appends the declaration block to the declared circuit, padded to PoUW's grid
   (`pad(real + 1)`, one filler at least), with unit j on the plan's device j mod its length. The verifier refuses by
   name: `advice-cap`, `advice` (B doesn't parse it), `prefix` (it doesn't extend the circuit so far), `placement` (B puts
   a unit on a device no registration holds yet), `record`. The record binds the previous record, the layout's digest,
   the block's Program digest and its values' root, all before any coin.
5. **Audit of declaration k.** After the `Declare`, the verifier draws its live coins and sampled proofs' `draw_first`
   picks subset:K′ of the block's units, at PoUW's `k_prime` for ε = 1/2, δ = 1/4 (`Draw`). The developer proves each
   drawn unit (`Proofs`: the stand-in), whose statement reads three commitments: the block's row, the unit's input state
   (the row before, the previous block's last row, or the registered initial state), and theta. Each subprotocol answers in
   `Core.Protocol`'s shape, and the composite is the composition's rule and binding over exactly the product's subprotocols.
6. **Accounting** (`account`). The chain from the registration through every record, each declaration's composite, the
   period's verdict by the same rule, the credited AND gates (PoUW's grid point at or below the real units, fillers
   credited nothing) against the capacity of the devices registered in each declaration period, the advice bits charged
   as leakage, the pooled bound, and the same scaled to a week, a month and a year.

One declaration and the period (`run.py` prints every phase):

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
| `unreported`: one request served and left out of the advice | operation | accepted, uncredited (remote can't see it) |
| `advice-cap`: 33 bits of advice | declaration | refused `advice-cap` |
| `recommit`: a second commitment after the draw | declaration | refused `recommit` |
| `wrong-value`: every output of block 2 off by one | audit | sampled proofs rejects; so does PoUW, which reads it |
| `fork`: record 2 names the registration, not record 1 | accounting | the chain; every subprotocol accepts |

## (b) Interfaces

| name | caller | in | out | module |
| --- | --- | --- | --- | --- |
| `Layout`, `Layout.capacity(ticks)` | everyone | nodes, devices, stores, links, auditor devices | the site; what it can do in a period | missing: written in the example (pieces: `Security/Definitions/Accounting/Device.lean`, the deploy config's machines and links) |
| `check_registration(layout, registration, auditor)` | verifier | the layout so far, a `Layout` of additions, the auditor's devices | merged `Layout`, or a refusal by name | missing: written in the example |
| `Builder` (`descriptor`, `of`, `digest`, `grow`) | developer, verifier | a_1 ‖ … ‖ a_k | the declaration blocks, placed | missing: written in the example |
| `register`, `check_baseline` | developer, verifier | product, theta, B, state0 | `Baseline` (roots and B's descriptor) | missing: written in the example |
| `operate` | developer | requests, theta, state, tape | `Served` | the integration's (here the toy model) |
| `declare`, `check_declaration` | developer, verifier | declared advice, served values, layout, previous record | `Declaration`, `Block` | missing: written in the example |
| `commit`, `opens`, `committed` | developer, verifier | rows and salts; an opening | a root; whether it opens | `verity/core/protocols/sampled_proofs/one_stage/registered.py` over `verity.ml.commitments.hm96` |
| `UNIT`, `BLOCK`, `block_digest(n)` | B | — | Boolean Definitions; a Program digest | `verity.ml.circuits` (`boolean.trace`, `defs.CompositeDefinition`, `codec`) |
| `evaluate_unit` | the stand-in verifier | theta, state, x | y, state′ | `verity.ml.circuits.evaluation.evaluate_words` |
| `draw(coins, block)`, `draw_size` | verifier | live coins | the drawn proof blocks | `verity/core/protocols/sampled_proofs/coins.py` (`draw_first`), `verity/core/protocols/pouw/window.py` (`bucket`, `k_prime`) |
| `prove(statement, witness)` | firewall | a `Statement`, the openings | a proof | STAND-IN for C-Flock |
| `verify(statement, proof)` | verifier | a `Statement`, a proof | each check's reason, or none | STAND-IN for C-Flock's verifier |
| `sampled_proofs`, `pouw_window` | verifier | one declaration's checks | `Core.Protocol` outcomes | missing as per-declaration functions: written in the example |
| `compose(protocol, answers)` | verifier | the subprotocols' outcomes | the composite verdict and binding | `verity/core/protocols/composition/tests/reference.py` (`verdict`, `bind`, `REMOTE`) |
| `account(baseline, declarations, lines, layout)` | verifier | the period's records and verdicts | period verdict, credit, share, bound, scaling | missing: written in the example |
| `Developer`, `Verifier`, `remote.MANIFEST` | runtime | params, capabilities, messages | actions | `verity/core/deploy/` (`party`, `manifest`, `runtime.run`) |

## (c) What this exposes

- **The declared circuit has no one digest, and Lean's `Declaration` holds one circuit.** Each declaration block has its
  own Program digest, and the record chain binds them in order. `codec.program_digest` of B(a_1 ‖ … ‖ a_k) as one Program
  is not prefix-incremental, so "the declared circuit so far" is a list of block digests plus a chain, not one Program.
  `Security/Definitions/Core/Declaration.lean`'s `Declaration` is the declared circuit so far, and its file says one
  Program is the circuit of record: d.D, one `Circuit`, is here every declaration block composed along the state wire
  (block k's state input is block k − 1's state output, not an input gate), which the spec has no operation for; d.θ,
  the registered information, is theta's and the initial state's input gates (each block Program's `registered` and `theta` inputs).
  `Baseline` is d before any block, and each `Declaration` record extends d.D by one block and d.θ by nothing: its
  requests are novel input gates, outside θ.
- **State crosses declaration blocks as a committed value.** A drawn unit at the start of block k opens block k − 1's
  last row (or the registered initial state), so one unit's statement reads three commitments. `Core.Protocol`'s answer carries
  one commitment and one evidence: the cross-block read and the chain live outside every subprotocol, in
  `check_declaration` and `account`. `fork` passes every subprotocol and only accounting catches it. The composition
  composes one declaration's subprotocols; nothing composes declarations into a monitoring period. The example applies the
  same rule over them.
- **Where prefix-extending B breaks.** (i) PoUW's grid pads every block (`pad(real + 1)`), so the declared circuit is
  full of filler units, and their count depends on how the work was cut into declarations. (ii) Work that straddles a
  declaration period's boundary: a batch spanning two, or a decode whose KV cache carries into the next; the example cuts
  at request boundaries, and its carried state is one word. (iii) A builder that schedules globally (rebalancing load,
  placement by what came later) isn't prefix-extending. (iv) B here is Python with a descriptor digest, where the
  registered builder is to be a circuit with a public length cap, and its advice validation (consecutive batches of at most 2) is B's own code.
- **Placement is static, and the layout grows.** Placement isn't in the advice: B's plan names devices by declaration,
  and `check_declaration` holds each block's placement to the layout as registered then. B may name a device before its
  registration, and nothing lands on it until it is registered (`placement`). Removing a device, or one failing, has no
  path: B's plan can't drop it without a new B.
- **Hardware registration is its own phase, and it grows.** The first registration comes before B and the registered information, and
  a later one lands between two declarations; each record names the layout digest its block was placed on. What a later
  registration may not do (rename, remove, or change a device's SKU) is only `duplicate` here.
- **What registration must hold.** The layout (devices by SKU, stores, links with rates and latencies, uplinks, where each
  auditor device sits); B's descriptor (format, cap, plan, grid); theta's and the initial state's commitments under B's
  digest. There is no registration of what B's advice may say beyond its cap.
- **Remote can't see unreported work.** Without a view of the system's I/O, work left out of the declared advice is
  uncredited, not detected: remote's credit is a lower bound on the work done.
- **The period bound is computed, not proved.** A declaration alone gives δ = 1/4, which looks weak; pooled over the
  monitoring period's independent draws, a wrong share ε of the credited work escapes with probability at most
  q_max^⌈εN⌉ (q = 1 − K′/n, N credited units): 1.0 % over the three declarations, 10^-6017 over a week of declarations
  like these. The theorem from a declaration's audit to the period's is the layer the modular math still needs.
- **Capacity is the layout's.** Per-period capacity comes from the registered devices' data sheets, per declaration
  period, so a later registration raises it from then on. The toy's share of it (5 × 10^-12) says only that the model is
  tiny; over all compute (a datacenter's, the world's) the scaling is a parameter the example doesn't take.
- **Per-declaration subprotocol checks don't exist yet.** PoUW's `core_check` takes a whole run's case; the example writes one
  answer a declaration. Sampled proofs draws from the declaration block; each subprotocol's sampling block is not yet its
  own draw.
- **The advice is charged but the toy's circuit doesn't depend on it beyond the count.** Batching enters B only as
  validation. A batched matmul would change the block's shape, and then B's output depends on the advice, as it should.
- **Message kinds.** `Proofs` is private only because the stand-in's proof is an opening; C-Flock's proof would be
  public.

## (d) Proposed homes

In a noun-based layout, each part of the example goes where its noun is:

- `verity/registration`: `Layout` and its capacity, `check_registration`, the registered information (`register`, `check_baseline`),
  and B's registered descriptor and `grow`; `declare` and `check_declaration` too, as what is registered over time.
- `verity/circuits`: the units, the declaration blocks and the declared circuit as blocks chained by state wires, with a
  digest of the chain (today's `verity.ml.circuits`).
- `verity/commitments`: `commit`, `opens` and the row commitments (today split between `one_stage.registered` and
  `verity.ml.commitments`).
- `verity/flock`: `prove` and `verify`, replacing the stand-in, plain or zero-knowledge behind the firewall.
- `verity/protocols/remote` (and `verity/protocols/onsite`): the parties and manifests, each composite's per-declaration
  subprotocol checks and `account`, over the composition's rule.
- The model, its load and `operate` stay with the integration that serves them (`integrations/`).
