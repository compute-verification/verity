# The ACCEL model of each measured card

`machine-model` prices PoUW's designs on the ACCEL model 𝓜 of each measured card. For a card it holds, as data, the rows
(the measured ceilings: a pipe, a port, or a joint frontier of several classes), the census that places every warp
instruction class on them, the attack families and the designs. From them it solves the price vectors and reports a
design's ε. Capacity is the peak rate per operand class, read at the card's clock: a row is a ceiling no program beats,
and the model has no power or energy term. It measures nothing: no command starts a process, reads the environment or
the network, or needs a GPU. The directory and the command keep the name "machine model"; everywhere else 𝓜 is the
ACCEL model (`GLOSSARY.md`).

## Try it

Re-solve the current price vector and check it:

~~~sh
uv run machine-model solve
~~~

It prints the vector's τ, U/F_low and margin beside the numbers it was published with, the cheapest attacks, the check
of every price, credit and attack, and each claim with what is under it. It exits 1 when the vector fails its check or
doesn't reproduce a published number. `uv run machine-model` alone prints every command's usage.

What it computes:

- **Price vectors** (`solve`): the LP that prices every warp instruction class as a convex combination of the rows. Each
  named vector re-solves to the τ, U/F_low and margin it was published with, and every attack of its epoch is checked
  against it.
- **ε** (`eps`): a design's core ε in the tensor–scalar model, every term from data.
- **The headline** (`headline`): each GPU family's design of record at the data-sheet and measured readings, the numbers
  the Lean headline guarantees are exported from ("The headline is per node").
- **The card table** (`theta`, `sheet`): each card's rates, capacities and clocks with their sources, and its data-sheet
  lines.
- **The opcode table** (`cover`): the evidence for the card assumption.

This file holds no table of the data. `machine-model render` prints every one as Markdown from a data directory alone:
the vectors, the epochs and their census, the rows, the floors, the designs, the honest kernels, the proof blocks, the
MMA forms, the table route, the headline's designs, the attack families, the devices and the card table.
`tests/benchmarks/machine_model/test_render.py` fails on a table here, so a new vector, design or census row touches only its own data file.

## The headline is per node

`machine-model headline`'s ε, and the Lean headline guarantees exported from this data
(`ResourceAccounting.Guarantees.TS.Sm120SheetHeadline` and `Sm120MeasuredHeadline`), bound work done by the node: an accepted
window's node, its GPUs, host CPU and peer GPUs together, performed at least (1 − ε) of the work the window credits. The
device that did it is left to the attribution layer. The floor argument places no work: every credited word is a fresh
function of the post-coin noise, so it was computed after the coin, but nothing ties a committed tile to the GPU that
computed it. The transcript is a Merkle root over tile digests, so a helper holding the salt ships one 64-byte digest a
tile. The model prices the work on one device's rows, which makes ε a share of the credited time `T_cred` on that
device, not of that device's own time.

A per-GPU reading of ε comes only through the attribution layer, under two conditions:

- **The host deduction**: what the host CPU can compute of the work the window credits is deducted exactly from the
  GPUs' credit. On int8 that is AMX for a tile's products and AVX-512 for the fold and credited fillers, each bit-exact.
- **The peer-GPU premise**: every billable GPU on the node is credited each window, so no uncredited peer computes a
  credited GPU's tiles over NVLink. Otherwise the node is the unit.

Counted per GPU without them, the host's AMX and AVX-512 (and NVLink peers) take every device over 1% on some real host.
