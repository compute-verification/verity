import Definitions.Nci.CoverFanOut
import Mathlib.Basic.Real.Basic
import Mathlib.Data.Finset.Union

/-!
# The cover theorem's fan-in/fan-out tightening: the new premises

The fan-out bounds rest on two new premises, beside the cover theorem's `UpdateHardness`, each its own named `Prop`, so
that what each costs is visible. `ColumnsIncompressible` prices the embedding lookups a block holds, and
`ValueReadHardness` prices each value a block spends at `f` bits, from the same bits as the updated parameters. Both
are statements about the training circuit `C`, its products and a keyed token structure (`Keyed`), at block size `s`.
-/

namespace Nci.Cover.Assumptions

/-- NEW PREMISE. **Lookup columns are incompressible** ([COLUMNS-INCOMPRESSIBLE]) at block size `s`, `bcol` bits per
column parameter: a set `S` of at most `s` gates that holds lookups of one token (computes an exit of each without
being handed any of its gates) is handed `bcol` bits, besides the registered parameters, for each parameter in their
columns: `bcol · #(⋃ columns) ≤ #(inputs S \ θ)`.

Rationale: a static gather reads its whole column, so holding a trained embedding coordinate means holding bit `j` of
every one of the `V` entries in its column, handed (one input gate each) or computed from what the optimizer updated
them from. At `bcol = 1` this costs the same as being handed the column; it is a premise only because the column is
computed.

It holds only with `Keyed`'s lookup rule (L). A lookup declared narrower than its cone (the gather's output word alone)
is held from the last mux level's few bits, or a vocab-parallel gather's shard partials, and one declared wider (the id
decode inside, which the backward scatter-add reuses, so it is an exit) is held from the id's bits. A registered entry
(the first step after a checkpoint, a frozen row) is held for free, so a column lists only entries updated at that step.

It fails above 1 bit per entry, outright, if the gather is lowered as a bitwise mux: a block that holds one output bit
of it over a handed column pays only `V` bits (so not A3's 2.5). At `bcol ≤ 1` it holds even for rows a run rarely
updates: from the second step after a checkpoint on, each entry reads the previous step's whole backward pass through
the scatter-add, more than `s` gates, so a block pays a gate per entry. Those rows' values are nearly their registered
ones, but the premise counts gates, not information.

Kimi K2 (`V = 163840`, `k = 2·10⁷`): `bcol = 1` gives `E = 122` lookups per token per block. Both fan-out bounds need
far less, 0.41 bit per entry for `FanOutIntensity` and 0.27 for `FanOutReadIntensity`. -/
def ColumnsIncompressible {C : Circuit} {J L Tok Val : Type*} [DecidableEq L] [DecidableEq Val]
    {M : MulBlocks C J L} {d aOut c V f : ℕ} (K : Keyed M Tok Val d aOut c V f) (s : ℕ) (bcol : ℝ) : Prop :=
  ∀ S : Finset (Fin C.N), S.card ≤ s → ∀ X ⊆ K.lookups ∩ K.heldSet S, (∀ v ∈ X, ∀ v' ∈ X, K.tok v = K.tok v') →
    bcol * (X.biUnion K.col).card ≤ (C.inputs S \ C.θ).card

/-- NEW PREMISE. **A token value read costs `f` bits** ([VALUE-READ-HARDNESS]), A1's (H′) at block size `s`: a set `S`
of at most `s` gates that spends the values `X` (`Keyed.Spent`) and computes whole counted products over `w`
distinct updated parameters is handed `f` bits for each value and `b` for each parameter, besides the registered
parameters, the two added: `f · #X + b · w ≤ #(inputs S \ θ)`. `f` is the width at which `Keyed` declares the counted
products to read their activations.

It implies `UpdateHardness` for counted products, and it adds two things: the bits paying for parameters don't also pay
for values (its additivity is `ConditionalUpdateHardness`'s strengthening of `UpdateHardness`), and a value, a word of
`f` bits, costs `f` bits and not the one a gate of it costs on its own.

Rationale: a value is a word, and the products a block buys through one are computed from all of its `f` bits. A block
is handed them, or computes them from earlier values that it must be handed in turn.

It charges only spent values, which is how the fan-out proof charges. A1 charged every value a block reads, and on the
bit-level `Circuit` that is false: pad a block with gates that each read one bit of a value (a product's sign, the XOR
of the activation's sign bit and the weight's) and it reads values at 1 or 2 bits each. So `Spent` is the proof's own
charge, the first read on the key path back from a product the block computes whole or from a leg of a root it holds.
That is not assuming the conclusion: the premise prices reads, and the theorem counts products.

Scope:

* A2 (finding 3) shows that this premise and `UpdateHardness` together force `f` to be at most the number of
  activation bits that a product's output gate reads. `Keyed`'s (W) declares each product's output (`MulBlocks.o`) at
  a gate that reads `f` of its activation's exits (in Verity's NVFP4 lowering, `p[j]^s` with `j ≥ 1`, for `f = 4`).
* It fails where `UpdateHardness` fails.
* It fails when several spent values are cheap functions of one value: a block is handed each one's sign bit alone and
  computes the rest from the one value's bits, about 1 bit per value. Kimi K2's backward has 9 such codes per
  coordinate (the router-scaled MoE-output gradients the routed experts read, and the shared expert's), and its
  forward 8 dispatch slots per coordinate. `Keyed`'s (I) rules them out: each is a gate of its source value.
* A block may cut a value's word: a gate with a predecessor outside the block is one of its inputs, so the block reads
  the value at one bit and computes the rest, which come from what it is handed for other values. `Spent` charges only
  the first read on a key path, and with (I) the values beyond it are on no other spent value's key path. A path's start
  is the exception `Keyed`'s (F) rules out: a value keyed to an activation `κ` (a re-quantization of an `f`-bit `κ`)
  would be spent beside `κ` at one bit, `f + 1` bits for two values. With (I) and (F), no spent value is a key ancestor
  of another (`Keyed.spent_eq_of_follows`).

Kimi K2 (`f = 4`, `b = 2.5`): with `ColumnsIncompressible` at 1 bit per entry, about 3.6·10⁴ (`FanOutReadCoverBound`),
four times `FanOutCoverBound`'s 8,900. With Dgrad on the forward's tokens (`c = 2`), 7.3·10⁴ against 1.66·10⁴. -/
def ValueReadHardness {C : Circuit} {J L Tok Val : Type*} [DecidableEq J] [DecidableEq L] [DecidableEq Val]
    {M : MulBlocks C J L} {d aOut c V f : ℕ} (K : Keyed M Tok Val d aOut c V f) (s : ℕ) (b : ℝ) : Prop :=
  ∀ S : Finset (Fin C.N), S.card ≤ s → ∀ X : Finset Val, (∀ u ∈ X, K.Spent S u) →
    f * X.card + b * ((M.whole S).image Prod.snd).card ≤ (C.inputs S \ C.θ).card

end Nci.Cover.Assumptions
