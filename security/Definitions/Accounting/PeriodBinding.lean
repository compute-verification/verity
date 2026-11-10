module

public import Definitions.Accounting.ComposeMonitoringPeriod
public import Definitions.Accounting.TS.Hashed

/-!
# A period's positions, and two one-tile periods (trusted)

What `Pouw.Guarantees.PeriodBindingPerProver` (`Definitions.Accounting.Guarantees.PeriodBinding`) reads beside
`Declared.periodLayer` and `TS.HashedOpenings`, and the instances its two witnesses are at.

* `Declared.periodPositions Cd n Np`: every position a period's tiles read, `(w, g)` for each verification phase `w`
  and each gate `g` among a tile's slots (`Declared.slots`), with the gate `periodLayer` reads past a tile's last slot
  (its first slot, or `dag.length` for a tile with none).
* `onePeriod`: a circuit of no gates and one round, whose period is one verification phase of one tile, over one coin
  bit before the sample (`coinBit`). Every position is cut before that bit.
* `oneOpenings v`: each slot's opening is the one-bit row `[v x]`, with a salt of 192 zero bytes and no siblings,
  against a tree of one leaf under the constant hash `zeroHash`. So every opening of the value `v x` is verified, and no
  leaf splits.
* `honestLayer`: every proof accepted, opening every slot to `false`, with `oneOpenings` of `false`.
* `acceptAllLayer`: every proof accepted, opening every slot to the coin bit, with `oneOpenings` of the coin bit. The
  value it opens moves with a coin sampled after the commitment, so its binding fails.
-/

@[expose] public section

namespace Accounting.Compose

open Accounting.TS Pouw.RelativeSecurity

/-- **The positions a period's tiles read**: `(w, g)` for each verification phase `w < n` and each gate `g` among tile
`t`'s slots, `t < Np`, with the gate `Declared.periodLayer` reads past a tile's last slot (its first slot, or
`dag.length` for a tile with no slot). -/
def Declared.periodPositions {ι : Type} (Cd : Declared ι) (n Np : ℕ) : Finset (Fin n × ℕ) :=
  Finset.univ ×ˢ Finset.univ.biUnion fun t : Fin Np =>
    insert ((Cd.slots t.val).headD Cd.dag.length) (Cd.slots t.val).toFinset

namespace PeriodBinding

/-- A circuit of no gates and one round: its period is one verification phase of one tile, over one coin
before the sample. -/
def onePeriod : Declared Unit where
  dag := []
  calls := 0
  call := id
  rounds := 1
  round _ := 0
  coinMode _ := .live
  commitRound _ := 0
  seedNames _ _ := false
  units := 1
  unit _ := 0

/-- `onePeriod`'s coins: one verification phase: its one coin bit before the sample, and the
sample of its one tile. -/
abbrev OneCoins : Type := Paths (Paths Bool 1 × (Equiv.Perm (Fin 1) × Unit)) 1

/-- The verification phase's coin bit. -/
def coinBit (x : OneCoins) : Bool := x.1.1.1

/-- The hash that sends every message to the zero digest. -/
def zeroHash : Hash.Msg → Hash.Digest := fun _ _ => 0

/-- The leaf verifier of a tree of one leaf, under the hm96 key `[]` and the format `row/v2`. -/
def oneVerifier : Hash.RowLeaf.Verifier := ⟨⟨fun _ => 0, 1, id⟩, [], .v2⟩

/-- **Each slot opened by the one-bit row of `v`**: on `x`, every position's leaf is the one leaf of a tree under
`oneVerifier`, at rank 0 and root the zero digest, a row decodes to its first bit, and each slot's opening is the row
`[v x]` with a salt of 192 zero bytes and no siblings. -/
def oneOpenings (v : OneCoins → Bool) :
    HashedOpenings Bool (Equiv.Perm (Fin 1) × Unit) 1 1 1 ℕ (Fin 1 × ℕ) Bool where
  leaf _ _ := oneVerifier
  rank _ _ := 0
  root _ _ := fun _ => 0
  decode row := row.headD false
  opening x _ _ _ := ⟨[v x], List.replicate Hash.Hm96.SALT_BYTES 0, []⟩

/-- Every proof accepted, opening every slot to `false`. -/
def honestOutcomes : Outcomes Bool (Equiv.Perm (Fin 1) × Unit) 1 1 1 1 Bool where
  accepted _ _ _ := True
  opened _ _ _ _ := false
  rel _ _ _ _ := True
  right _ _ _ _ := True

/-- Every proof accepted, opening every slot to the verification phase's coin bit. -/
def acceptAllOutcomes : Outcomes Bool (Equiv.Perm (Fin 1) × Unit) 1 1 1 1 Bool where
  accepted _ _ _ := True
  opened x _ _ _ := coinBit x
  rel _ _ _ _ := True
  right _ _ _ _ := True

/-- `onePeriod`'s period layer at `honestOutcomes`. -/
def honestLayer : SampledLayer Bool (Equiv.Perm (Fin 1) × Unit) 1 1 1 (Fin 1) ℕ (Fin 1 × ℕ) Bool :=
  onePeriod.periodLayer Nat.one_pos le_rfl honestOutcomes

/-- `onePeriod`'s period layer at `acceptAllOutcomes`. -/
def acceptAllLayer : SampledLayer Bool (Equiv.Perm (Fin 1) × Unit) 1 1 1 (Fin 1) ℕ (Fin 1 × ℕ) Bool :=
  onePeriod.periodLayer Nat.one_pos le_rfl acceptAllOutcomes

end PeriodBinding

end Accounting.Compose
