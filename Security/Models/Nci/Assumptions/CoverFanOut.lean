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

It fails:

* above 1 bit per entry, outright, if the gather is lowered as a bitwise mux: a block that holds one output bit of it
  over a handed column pays only `V` bits (so not A3's 2.5);
* like `UpdateHardness`, in the first steps after a checkpoint, when the column is nearly its registered values, and on
  rows a run rarely updates;
* for a frozen embedding: a lookup over a registered column is held for free. Declare the token's input words as its
  lookups instead (`Keyed`): no block holds an input gate, so the premise says nothing about them.

Kimi K2 (`V = 163840`, `k = 2·10⁷`): `bcol = 1` gives `E = 122` lookups per token per block. Both fan-out bounds need
far less, 0.41 bit per entry for `FanOutIntensity` and 0.27 for `FanOutReadIntensity`. -/
def ColumnsIncompressible {C : Circuit} {J L Tok Val : Type*} [DecidableEq L] [DecidableEq Val]
    {M : MulBlocks C J L} {d aOut c V : ℕ} (K : Keyed M Tok Val d aOut c V) (s : ℕ) (bcol : ℝ) : Prop :=
  ∀ S : Finset (Fin C.N), S.card ≤ s → ∀ X ⊆ K.lookups ∩ K.heldSet S, (∀ v ∈ X, ∀ v' ∈ X, K.tok v = K.tok v') →
    bcol * (X.biUnion K.col).card ≤ (C.inputs S \ C.θ).card

/-- NEW PREMISE. **A token value read costs `f` bits** ([VALUE-READ-HARDNESS]), A1's (H′) at block size `s`: a set `S`
of at most `s` gates that spends the values `X` (`Keyed.Spent`) and computes whole counted products over `w`
distinct updated parameters is handed `f` bits for each value and `b` for each parameter, besides the registered
parameters, the two added: `f · #X + b · w ≤ #(inputs S \ θ)`.

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
  activation bits that a product's output gate reads. So declare each product's output (`MulBlocks.o`) at a gate that
  reads all `f` bits of its activation (in Verity's NVFP4 lowering, `p[j]^s` with `j ≥ 1`, for `f = 4`).
* It fails where `UpdateHardness` fails.
* It also fails when several spent values are cheap functions of one value (copies, or scalings by registered or
  already-paid scalars): a block can be handed each one's sign bit alone and compute the rest from the one value's
  bits, paying about 1 bit per value. So a copy belongs to the value it copies (`Keyed`: Kimi K2's dispatch copies are
  part of the routed token's MoE-input value), and Kimi K2's keyed values have no fan-out through registered weights.

Kimi K2 (`f = 4`, `b = 2.5`): with `ColumnsIncompressible` at 1 bit per entry, about 3.6·10⁴ (`FanOutReadCoverBound`),
four times `FanOutCoverBound`'s 8,900. -/
def ValueReadHardness {C : Circuit} {J L Tok Val : Type*} [DecidableEq J] [DecidableEq L] [DecidableEq Val]
    {M : MulBlocks C J L} {d aOut c V : ℕ} (K : Keyed M Tok Val d aOut c V) (s f : ℕ) (b : ℝ) : Prop :=
  ∀ S : Finset (Fin C.N), S.card ≤ s → ∀ X : Finset Val, (∀ u ∈ X, K.Spent S u) →
    f * X.card + b * ((M.whole S).image Prod.snd).card ≤ (C.inputs S \ C.θ).card

end Nci.Cover.Assumptions
