# The hardware model and the security statement: design

`security/Definitions/Slowdown/Security.lean` holds the three sets of the security argument, and
`security/Definitions/Slowdown/Guarantees/Secure.lean` its one statement, `Slowdown.Guarantees.Secure`: at a frame
`F`, no computation is in `H F.hw`, in `A F x_s x_b x_w y_b` and in `B F p Y`, for every count `Y` at least the fit
bound (`Frame.bound`). This file says why the hardware model `H` has the clauses it has, what it takes from an H100's
card, and what the statement does not give.

## The computation and the hardware

A computation (`Computation`) is a run of the actual-circuit model (`ResourceAccounting.Actual.Run`): a circuit with
every gate at a leaf and a tick, each gate's value, and what each leaf holds at each tick. With it come what each gate
uses of its location's compute (a pricing), the link each wire between two leaves runs on, the registered inputs, and
the gates of the declared circuit.

The hardware (`ResourceAccounting.Hardware`) is leaves and links. `H` reads five of its fields: its locations (sets
of leaves), each location's compute a tick and its memory, and each link's ends and capacity. A machine is a choice of
leaves and links. `Hardware.ofLeaves` gives every set of leaves the sum of its leaves' limits, and `server n` is `n`
GPUs (a die and a device memory each), the host, the switch between the GPUs and the world outside. More GPUs, the
host's compute and a switch are more leaves of the one model.

## `H`: four clauses

| Clause | Statement | Lean | `Secure` reads it at | False without it |
| --- | --- | --- | --- | --- |
| Order | a gate reads only gates placed no later than itself | `Placed.Causal` at no latency | every wire | `order_needed` |
| Work | at the computation's pricing, the gates at a location over an interval of ticks cost at most its compute a tick | `Placed.ComputeRate` | the frame's servers, over missed windows | `work_needed` |
| Memory | at every tick a location's leaves hold, with what is in flight between them, at most its memory | `Actual.Run.Memory` | the frame's servers, at the cuts that open covered windows | `memory_needed` |
| Bandwidth | each link delivers at most its capacity a tick, each way | `Placed.Links` | the links across each server's boundary, over covered windows | `bandwidth_needed` |

Each theorem of the last column (`security/Proofs/Slowdown/SecureWitness.lean`) is a run that meets the other three
clauses, is in `A` and is in `B` at the fit bound, so the statement is false with any one clause dropped. `H` is a
set at the hardware alone: every audit's claim (`security/Proofs/Slowdown/SecureAudit.lean`) is about the same `H`,
whatever frame the audit covers.

`H` leaves out what no claim reads: the latency between leaves (the proofs use the order of a wire's two gates, never
how far apart they are), `ResourceAccounting.Feasible`'s sequential speed, and a bound on depth a tick. Dropping a
clause enlarges `H`, so the statement holds of more computations. The rest of the `Hardware` bundle (the ring, the
trusted devices' leaves, buckets, jitter, record cells) is protocol structure, which `H` doesn't read.

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

- **Work.** `secure_w1` takes Pearl-C's gap property per coin, which is not proved, and its bound is the die's
  compute a tick at the audit's prices, where the card gives no ceiling but the sum of its pipes'.
- **A GPU's ingress.** At a frame of one GPU the boundary is PCIe and NVLink, which no audit counts: `secure_device`
  takes the count as a premise (`counted`). At a frame of a whole server the network attester counts the boundary,
  and the host's memory and disks are inside it, which no storage audit covers.
- **Bandwidth.** PoUB's certificate is about the memory link, so `secure_poub`'s frame is the die alone, and its
  bound leaves PCIe's and NVLink's capacity unaccounted for.
- **Three audits, two frames.** Work and memory are at the device (the die and its device memory), bandwidth at the
  die. `secure_device` takes work, memory and a count; no theorem takes all three certificates.
- **The audits' premises.** Each theorem over an audit's coins takes that the audit's schedule is the one the run
  induces on the device, and the premises of the audit's own guarantee.
- **Training.** That a run which trains is in `B` is `Trains`'s premises: `reach` and `fits` are empirical, and the
  three `shared` premises bound what a reference training algorithm shares with the declared circuit.
- **Units.** `A` counts bits held, and a wipe's guarantee bounds the information that survives it.
