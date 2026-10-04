import FlockSoundness.Discharge.Layout.Scope

set_option Elab.async false

/-!
# The canonical I/O format, as one check on the parsed circuit

`PROTOCOL.md` §16.15 specifies one row format for every commitment a session registers: every row, input or output,
is a `sha512/row/v2` bit row of any length `ℓ < 2^64`, hashed as `SHA-512(P(ℓ) ‖ packBits(bits))` with SHA-512's own
padding (so a last block may be partial, and is the `hm96` slot's); output rows hold exactly the bits of their run of
output ports; and every unit input is a row bit or the zero. `formatOk` is that format as code that runs on the circuit
`HmOut.parse` returns, in the style of `Layout.scopeOk`, and `formatOk_spec` reads it back as the conditions the binding
lemmas take. It replaces `Layout.scopeOk`: no condition on the leaf groups (any number of input rows and groups, leaf
cuts included), none on the out net (a tail's last stage may return the outputs), and none on row widths (`16 · words`
retires with `sha512/row/v1`).

`unitLog ≤ 32` and `portReadsB` stay: the first narrows nothing (`Layout.unitLog_le`), and the second is what makes
`readsRows`'s keys the keys a computed row reads (an input column outside every port is a self row, which Δ copies
nothing into). `vusPow2` is new and does narrow: it is what `outNet = unitNet` gave the region pin for free.

**Proof work.** `formatOk_spec` is `Layout.scopeOk_spec`'s unfolding at the new conjuncts (`Bool.and_eq_true`,
`List.all_eq_true`): mechanical, no new lemma.
-/

namespace FlockSoundness.Discharge.Canonical

open Flock FlockLevel3

/-- **A canonical row**: a `sha512/row/v2` bit row (`v2`), unsegmented (no `sha512/row-seg/v1` segments). -/
def rowOk (p : Port) : Bool := p.v2 && p.segs.isNone

/-- **Every output row holds exactly its run's bits**: output row `j` (port `nIn + j`) is `16 · w` bits, `w` the words of
its run of output ports (`HmOut.outGroup`). No zero-padded output row. -/
def outRowsExact (c : Circuit) : Bool :=
  (List.range c.outRowPorts.size).all fun j =>
    (c.ports.getD (HmOut.nIn c + j) default).bits == 16 * (HmOut.outGroup c j).2.2.2

/-- **Every unit input is a row bit or the zero**: at every stacked unit input column, Δ's key (`Aliased.srcKey`) is the
zero, a row bit (`.msg`: a leaf's or a leaf cut's), or a column outside every port (`.pad`, which no computed row reads,
`portReadsB`); never another net's output (`.wire`) or a fixed column (`.lit`), which have no registered commit string. -/
def readsRows (c : Circuit) : Bool :=
  (List.range (c.upv * (c.unit.inWords * WORD))).all fun x =>
    match Aliased.srcKey c x with
    | .wire .. => false
    | .lit _ => false
    | _ => true

/-- **Every VU's rows are pinned**: the VUs per block `g` are a power of two. The statement's regions pin the `b ‖ c` of
VUs below `2^⌊log₂ g⌋` (`Hidden.bc_drawnH`'s `hg2`); `HmRow.parse` gives that for every VU only when the unit returns
the outputs (its unit range `g · upv` a power of two, `Layout.vu_lt_pow2`), and a tail's circuit has no such check. -/
def vusPow2 (c : Circuit) : Bool := 2 ^ c.g.log2 == c.g

/-- **The canonical I/O format** (`PROTOCOL.md` §16.15): every row canonical (`rowOk`), output rows exact
(`outRowsExact`), unit inputs row bits or the zero (`readsRows`), computed unit rows reading port bits only
(`Layout.portReadsB`), `unitLog ≤ 32`, and every VU's rows pinned (`vusPow2`). -/
def formatOk (c : Circuit) : Bool :=
  c.ports.all rowOk && outRowsExact c && readsRows c && Layout.portReadsB c.unit && decide (c.unit.unitLog ≤ 32) &&
    vusPow2 c

/-- **The canonical row widths**: row `q`'s own bits, `ℓ_q` (`sha512/row/v1`'s `16 · words`, `Layout.portRowW`, retires). -/
def rowW (c : Circuit) (q : Fin c.ports.size) : ℕ := c.ports[q].bits

/-- **`formatOk` gives the binding lemmas' circuit conditions**: every row a bit row without segments, output rows of
exactly their run's bits, every unit input a row bit, the zero or unread, computed unit rows reading port bits only,
and every VU below `2^⌊log₂ g⌋`.

*Route*: unfold `formatOk`, split the conjunction (`Bool.and_eq_true`), and read each `all` (`Array.all_eq_true'`,
`List.all_eq_true`, `List.mem_range`); `PortReads` is `Layout.portReads_of_portReadsB`; the VU bound rewrites `hg` by
`vusPow2`. -/
theorem formatOk_spec {c : Circuit} (hU : Built c.unit) (h : formatOk c = true) :
    (∀ p ∈ c.ports.toList, p.v2 = true ∧ p.segs = none) ∧
    (∀ j < c.outRowPorts.size,
      (c.ports.getD (HmOut.nIn c + j) default).bits = 16 * (HmOut.outGroup c j).2.2.2) ∧
    (∀ x < c.upv * (c.unit.inWords * WORD),
      (∀ net slot col, Aliased.srcKey c x ≠ .wire net slot col) ∧ ∀ col, Aliased.srcKey c x ≠ .lit col) ∧
    Layout.PortReads c hU ∧ (∀ {g : ℕ}, g < c.g → g < 2 ^ c.g.log2) := by
  sorry

end FlockSoundness.Discharge.Canonical
