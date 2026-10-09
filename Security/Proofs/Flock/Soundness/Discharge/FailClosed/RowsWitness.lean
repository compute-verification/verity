import Proofs.Flock.Soundness.Discharge.FailClosed.Scope

/-!
# A circuit the hidden-output scope admits with `sha512/row/v2` rows

At hidden outputs the statement check's circuit rule is `Layout.scopeOkH` (`ProvedScope.scopeOkAt true`): `scopeOk`'s
unit conditions, and every row a `sha512/row/v1` row or every row whole (`Layout.wholeRows`). `witnessH` is a circuit as
C-Flock's composer writes a hidden-output statement over bit rows (`OUT_ROWS_V2_IDENTITY`): a `sha512/row/v1` input row
and one `sha512/row/v2` output row of whole SHA-512 blocks, one output port per output row. `witnessH_scope`: the
executable's rule at hidden outputs admits it, `Layout.scopeOkH` admits it and `Layout.scopeOk` refuses it, so the
hidden-output headlines (`FailClosed.ZkHidden`, `FailClosed.ZkRegistered`, `FailClosed.Exec`'s hidden cases) now hold of
circuits with bit rows that they did not reach before. `ZkHidden.drawHypsZHJ` and the `*_nonvacuous` theorems beside it
take `Layout.scopeOkH`, so their draw hypotheses hold together at such a circuit too. Checked by evaluation.

What none of this covers: that a session `verify` accepts exists over `witnessH` (it would need a proof file, which the
kernel does not evaluate), and a bit row of a partial last block or a segmented row, which `scopeOkH` refuses.
-/

namespace FlockSoundness.Discharge.FailClosed

open Flock

/-- One column per row: A's row 1024 is `[col]`, every other row empty. -/
def witnessRows (col : ℕ) : Sparse where
  rows := 2048
  width := 1026
  data := ⟨#[col.toUInt8, 0, 0, 0]⟩
  start := Array.replicate 1025 0
  len := (Array.replicate 1024 0).push 1

/-- A unit of one 1,024-bit input row (input words 0 … 7, columns 0 … 1023), one computed row (column 1024, the AND of
the row's bits 0 and 1), the constant (column 1025) and one output bit. -/
def witnessNet : Net where
  name := "witness"
  useful := 1026
  constPos := 1025
  inGroups := #[(⟨0, 8, #[1024]⟩ : Flock.Group)]
  outGroups := #[(⟨8, 1, #[1]⟩ : Flock.Group)]
  inWords := 8
  a := witnessRows 0
  b := witnessRows 1
  unitLog := 11

/-- A hidden-output circuit whose unit is `witnessNet`: its input row `w`, a `sha512/row/v1` row of 64 u16 words (1,024
bits) in one compression, then its output row `y`, a `sha512/row/v2` bit row of the output port's word zero-padded to a
whole SHA-512 block (1,024 bits), one output port per output row. -/
def witnessH : Circuit :=
  { (default : Circuit) with
    name := "witness", nets := #[("unit", witnessNet)], unitNet := 0, outNet := 0,
    ports := #[{ name := "w", words := 64, chunks := 1, nb := 1 },
      { name := "y", words := 64, chunks := 1, nb := 1, bits := 1024, v2 := true }],
    outPorts := #[("y", 1)], outRowPorts := #[1], g := 1, upv := 1 }

theorem witnessH_eval :
    Layout.scopeOkH witnessH = true ∧ Layout.scopeOk witnessH = false ∧ witnessH.ports[1]!.v2 = true := by
  decide +kernel

/-- **The hidden-output scope admits a circuit with a `sha512/row/v2` row**: the statement check's circuit rule at hidden
outputs passes `witnessH`, which is `Layout.scopeOkH` and not `Layout.scopeOk`, and its output row is a bit row. -/
theorem witnessH_scope :
    ProvedScope.scopeOkAt true witnessH = true ∧ Layout.scopeOkH witnessH = true ∧
      Layout.scopeOk witnessH = false ∧ witnessH.ports[1]!.v2 = true := by
  obtain ⟨h1, h2, h3⟩ := witnessH_eval
  refine ⟨?_, h1, h2, h3⟩
  show ProvedScope.scopeOkH witnessH = true
  rw [scopeOkH_eq]
  exact h1

end FlockSoundness.Discharge.FailClosed
