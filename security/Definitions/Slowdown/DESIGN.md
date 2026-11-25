# The hardware model and the security statement: design

`security/Definitions/Slowdown/Security.lean` holds the three sets of the security argument, and
`security/Definitions/Slowdown/Guarantees/Secure.lean` its one statement, `Slowdown.Guarantees.Secure`: at a frame
`F`, no computation is in `H F.hw`, in `A F x_s x_b x_w y_b` and in `B F tb sh Y`, for every training bound `tb` that
holds and every count `Y` at least its count on the frame (`Frame.bound`).
`security/Definitions/Slowdown/Guarantees/Run.lean` holds the two statements over a run's coins that stand on it, `RunSecure` and `RunComplete`. This file says why the hardware model `H` has the clauses it
has, what it takes from an H100's card, what the statements over a run say, and what they do not give.

## The computation and the hardware

A computation (`Computation`) is a run of the actual-circuit model (`ResourceAccounting.Actual.Run`): a circuit with
every gate at a leaf and a tick, each gate's value, and what each leaf holds at each tick. With it come what each gate
uses of its location's compute (a pricing), the link each wire between two leaves runs on, the registered inputs, and
the gates of the declared circuit.

The hardware (`ResourceAccounting.Hardware`) is leaves and links. `H` reads six of its fields: the latency between
two leaves, its locations (sets of leaves), each location's compute a tick and its memory, and each link's ends and
capacity. A machine is a choice of
leaves and links. `Hardware.ofLeaves` gives every set of leaves the sum of its leaves' limits, and `server n` is `n`
GPUs (a die and a device memory each), the host, the switch between the GPUs and the world outside. More GPUs, the
host's compute and a switch are more leaves of the one model.

## `H`: four clauses

| Clause | Statement | Lean | `Secure` reads it at | False without it |
| --- | --- | --- | --- | --- |
| Latency | a gate reads only gates placed no later than the latency between their leaves allows | `Placed.Causal` | every wire, of which it reads the order of the two gates alone | `order_needed`, of the order |
| Work | at the computation's pricing, the gates at a location over an interval of ticks cost at most its compute a tick | `Placed.ComputeRate` | the frame's servers, over missed windows | `work_needed` |
| Memory | at every tick a location's leaves hold, with what is in flight between them, at most its memory | `Actual.Run.Memory` | the frame's servers, at the cuts that open covered windows | `memory_needed` |
| Bandwidth | each link delivers at most its capacity a tick, each way | `Placed.Links` | the links across each server's boundary, over covered windows | `bandwidth_needed` |

Each theorem of the last column (`security/Proofs/Slowdown/SecureWitness.lean`) is a run that meets the other three
clauses, is in `A` and is in `B` at the fit bound's count, so the statement is false with any one clause dropped. `H` is a
set at the hardware alone: every audit's claim (`security/Proofs/Slowdown/SecureAudit.lean`) is about the same `H`,
whatever frame the audit covers.

`Secure` reads of the latency clause only that a gate reads no gate placed after it. The latency itself is what a
timed audit's deadline rests on: PoUS bounds what a location computes before a challenge's answer is due by the ticks
its ring's multiplications take to answer (`ResourceAccounting.RingResponder`), which is a latency between the
location and its oracle locations. No theorem yet reads that from `H`: `pous_slot` takes the audit's device model
at its deadline as a premise.

`H` leaves out sequential speed (`Placed.SequentialSpeed`), a latency ceiling (`Placed.LatencyAtMost`) and a bound on
depth a tick, which no claim over `H` reads. Dropping a clause enlarges `H`, so the statement holds of more computations. The rest of the `Hardware` bundle
(the ring, the trusted devices' leaves, buckets, jitter, record cells) is protocol structure, which `H` doesn't read.

## The training bound is a parameter

`Secure` and `RunSecure` take a training bound (`Training`): the premises it asks of a reference training algorithm
divided into windows, the read sizes it is defined at, and its count for `k` windows that each read at most `S` bits.
`Training.Holds` is its one obligation: under its premises, the counted multiplications outside a set of windows
number at most its count for the windows left. The statements read nothing else of it, so a route to a bound on
training is a value of `Training` with a proof of `Training.Holds`, and it enters at no other place. The fit bound is
`Training.fit` at a recipe (`fit_holds`, `security/Proofs/Slowdown/Secure.lean`), and the threaded route is
`Training.threaded` (`threaded_holds`, `security/Proofs/Slowdown/SecureThreaded.lean`). The missed windows' work
and what a reference training algorithm shares with the declared circuit (`Shared`) are not the training bound's:
`Secure` adds them for every bound.

## Choices the proofs forced

- **Memory is bits held**, read again or not (`Computation.holds`), not the bits a later gate reads. A storage audit
  certifies what a device holds, so the limit and the audited amount are in one unit.
- **A load is a gate at the destination** whose wire crosses a link. One rule covers the memory link, PCIe, NVLink
  and the network, and an input that arrives at a tick is a wire from a leaf outside.
- **Bandwidth counts a bit at each tick it is read** (`Computation.delivered`), the unit of a link's capacity and of
  the bandwidth audit's certificate.
- **One limit on a priced sum**, not a count of each kind of functional unit. The claims read what a gate costs,
  not what it computes, and a limit for each op class gives the one limit (`computeRate_of_classes`).
- **Limits are per leaf and add over a set of leaves.** A value in device memory that the die reads is in neither
  leaf's state alone; it is in the state of the two together.
- **`A` has an "at most" clause**, a count of what entered a server (`y_b`). Where the declared circuit does not
  exhaust a link, "at least `x_b`" alone leaves the link's whole capacity unaccounted for.
- **The run is cut into windows.** A window the memory and bandwidth audits cover is charged what it can read of a
  reference training algorithm, and a window they miss is charged its work.

## The statements over a run

A run is a `Verity.Game`, the verifier's coins and the developer's messages, and each of its outcomes has a
computation behind it (`world`). Its probability is `Verity.Game.prob` and soundness is `Verity.Sound`, Core's, so a
protocol's guarantee in that form composes with these with no translation.

- **`RunSecure`.** Every outcome's computation is in `H`. The verifier accepts only when the four audits do, and each
  audit is sound, at its own error, for one clause of `A` at each window it covers: bits held (`HeldAtLeast`), bits
  delivered to the declared circuit (`DeliveredAtLeast`), bits delivered in all (`DeliveredAtMost`) and work
  (`WorkAtLeast`). A clause's amount is what the audit certifies at the outcome, so it may read the outcome. Then
  against every developer the verifier accepts a run whose computation is in `B`, at a training bound that holds and a
  count at least its count at the outcome's amounts, with probability at most the errors added over the windows. `Secure` is the step from
  `A` to `B` inside it.
- **`RunComplete`.** A developer whose strategy each audit rejects with probability at most its own error is
  rejected with probability at most their sum.

The four soundness premises are the audits' slots: `RunSecure` takes them, and an audit fills one by proving that
premise of its own acceptance. A slot's clause is in bits of wire width, the unit of `A`. Three are filled
(`security/Proofs/Slowdown/SecureAudit.lean`), at the run of the audits' coins, one uniform coin, and at any frame
with a window that is the audit's window on the audit's device:

| Slot | Filled by | From | At the audit |
| --- | --- | --- | --- |
| Memory (`HeldAtLeast`) | `held_slot` | the capacity audit's per-run guarantee over one tick | `pous_slot`, PoUS's period |
| Work (`WorkAtLeast`) | `work_slot` | the compute audit's per-run guarantee on the priced sum | `w1_slot`, Pearl-C at W1 priced |
| Bandwidth (`DeliveredAtLeast`) | `delivered_slot` | the bandwidth audit's per-run guarantee | `poub_slot`, PoUB |
| The count (`DeliveredAtMost`) | no audit | the links' capacity `b ω`, which every feasible run meets | |

`secure_compute`, `secure_bandwidth`, `secure_capacity` and `secure_device` are `RunSecure` at these slots on a frame
of one device, read back as a bound on the coins' probability.

## The H100's parameters

One tick is one period of the SM clock. The figures are `calibration/hardware/cards/h100-sxm-80gb.toml`'s, each with
its source there. `calibration/tensor_duty/` and `calibration/bandwidth/` measure them on a card.

| Clause | Parameter | Card rows | Standing |
| --- | --- | --- | --- |
| Work | each pipe's rate | `rate` × `sm.count` × `clock.sm` | A ceiling for each pipe a SM-clock. The clause needs one ceiling on the priced sum at the certificate's prices, and the card states none: the sum of the pipes' ceilings is the only one it gives. |
| | the tick | `clock.sm` | The data sheet's boost clock. A ceiling a tick holds while the clock stays at or below it, and no row states that of the clock around tensor work. |
| Memory | the die | `sm.count` × (4 × (`store.reg` + `store.reg_extra`) + `store.smem`) + `store.l2` | Leaves out the instruction and constant caches, which the RTX PRO's card has rows for (`store.l15`, `store.l1i`, `store.l0i`, `store.const_l0`) and this card does not. |
| | device memory | `store.dram_installed` | What a program can address. |
| Bandwidth | die to device memory | `memory.bandwidth` | A ceiling for each direction; `memory.read` is a placeholder. |
| | die to host (PCIe), die to switch (NVLink) | none | No GPU's card has a row for either. |
| Side compute | the host | `calibration/hardware/cards/xeon-platinum-8468-2s.toml` | The host's own card. |

PCIe and NVLink end at the die, not at device memory: a kernel can read a peer GPU's memory directly (CUDA's
peer-to-peer memory access), so the die computes on bits that never sit in its own device memory. `server`'s `bus`
and `peer` links therefore join the host and the switch to the die.

Outside the model: the engines no row prices (the video and JPEG decoders, the copy engines, the on-board
processors), the sideband buses, and the limits the firmware sets (the clocks, the SMs and the cache it disables).

## What the statement does not give

- **Work.** `w1_slot` takes Pearl-C's gap property per coin, which is not proved, and the fit bound charges a
  missed window the die's compute a tick at the audit's prices, where the card gives no ceiling but the sum of its
  pipes'.
- **A GPU's ingress.** At a frame of one GPU the boundary is PCIe and NVLink, which no audit counts: `secure_device`
  takes the count as a premise (`counted`). At a frame of a whole server the network attester counts the boundary,
  and the host's memory and disks are inside it, which no storage audit covers.
- **Bandwidth.** PoUB's certificate is about the memory link, so `poub_slot`'s server is the die alone, and the fit
  bound there leaves PCIe's and NVLink's capacity unaccounted for.
- **Three audits, two frames.** Work and memory are at the device (the die and its device memory), bandwidth at the
  die. `secure_device` takes work, memory and a count; no theorem takes all three certificates.
- **The audits' premises.** Each slot's filler takes that the audit's schedule is the one the run
  induces on the device, and the premises of the audit's own guarantee, among them the memory audit's device model at
  its deadline, which is not derived from `H`'s latency.
- **Training.** That a run which trains is in `B` is `Trains`'s premises: the training bound's (`Training.Prem`;
  the fit bound's reach and fit are empirical), and the three `shared` premises, which bound what a reference
  training algorithm shares with the declared circuit.
- **Units.** `A` counts bits held, and a wipe's guarantee bounds the information that survives it.
- **The run's slots.** The count's slot has no audit: at a server the network attester counts ingress, and its
  guarantee is not restated as the slot's premise. The filled slots are at the run of one uniform coin; an audit
  whose guarantee is over an interaction with messages fills its slot at that interaction, which no filler does yet.
- **Honest acceptance.** `RunComplete` takes each audit's rejection probability at the honest strategy; that an
  honest developer's computation is in `H ∩ A` at nonzero amounts is shown at one small run (`served_mem`), not at a
  served workload.
