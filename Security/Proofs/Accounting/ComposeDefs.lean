import Definitions.Accounting.Compose

/-! The proof-only part of `Definitions.Accounting.Compose`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Accounting.Compose

open Accounting.TS Pouw.RelativeSecurity

open Classical in
/-- **`F` per kind** (decision 107): the most tiles one late cone of a tile of kind `k` holds, from the IR. It is `1`
for every kind served: decision 106's units hold whole K and `z` reads no noise, so no unit reaches another late. Only
a coordinate cut into K-chunks, which COMPILE doesn't make, would have `F > 1` (its running sums late within their
round). At a tile weight bound `w_max` per tile, `ρ = 1/(F·r)`, `F` the largest `F_k` and `r = (N_p/N)·(w_max/w̄)`,
meets `CircuitWork`'s `ρ·N_p·shareCone ≤ 1`. -/
noncomputable def kindF {ι : Type} (Cd : Declared ι) (Np : ℕ) (k : Kind) : ℕ :=
  (Finset.univ.filter fun t : Fin Np => tileKind Cd Np t = some k).sup fun t =>
    (Finset.univ.filter fun t' : Fin Np => Cd.Reaches t.val t'.val).card

/-- **Each position committed by its IR cut is bound there** (RT-I8's B1 for the window, decision 114, at the writing
gate's round, W1): each value `ℓ` the prover writes whose commit coin is no later than its IR cut (`LayerCommitted`'s
first case) is bound before coin `Declared.cutOf ℓ` of the window's `rounds` (the draw past the last). A value opened
one way for one drawn tile and another for a tile that reads it, after the reader's coin, is then a binding failure
(`DrawnLayer.MovedIn`). `Declared.layer` meets it, so at `Declared.atIR` for every value. A value of
`LayerCommitted`'s second case, committed past its IR cut, is bound at its commit coin, and only its own unit reads
it. -/
def StageCuts {ι C B Q V : Type} {s N : ℕ} (Cd : Declared ι) (pf : DrawnLayer C B 1 Cd.rounds s (Fin N) Q ℕ V) :
    Prop :=
  ∀ ℓ, Cd.Writes ℓ → Cd.commitRound ℓ ≤ Cd.cutOf ℓ → ((pf.cut ℓ).2 : ℕ) = min (Cd.cutOf ℓ) Cd.rounds

/-- **Each tile's noise reads its round's coin** (W2: one coin a round, decision 122): the layer's coin for tile `t` is
its unit's round, the coin `Declared.Late` reads. `Declared.layer` meets it under `Declared.Wf`. -/
def StageCoins {ι C B Q V : Type} {s N : ℕ} (Cd : Declared ι) (pf : DrawnLayer C B 1 Cd.rounds s (Fin N) Q ℕ V) :
    Prop :=
  ∀ t : Fin N, ((pf.coin 0 t : Fin Cd.rounds) : ℕ) = Cd.roundOf t.val

/-- **Each tile's slots and each position's writer lie inside the IR**: a position with a writer is a value the prover
writes, a gate of the writer's unit, and each slot of tile `t` is one of `t`'s own gates or an operand of one of them
(`Declared.slots`), or no position when `t` has no gates. With `StageCuts`, `StageCoins` and `LayerCommitted` it puts
the layer's late cones (`DrawnLayer.Late`) inside the IR's (`Declared.Reaches`), so a share charging each skipped tile
its `tileCone` covers every tile that isn't good (`ServiceCovers`). `Declared.layer` meets it by definition. -/
def StageSlots {ι C B V : Type} {s N : ℕ} (Cd : Declared ι) (pf : DrawnLayer C B 1 Cd.rounds s (Fin N) ℕ ℕ V) :
    Prop :=
  (∀ ℓ u, pf.src ℓ = some u → Cd.Writes ℓ ∧ Cd.unit ℓ = u.2.val) ∧
    ∀ (t : Fin N) (k : ℕ), pf.pos 0 t k ∈ Cd.slots t.val ∨ (Cd.slots t.val = [] ∧ pf.pos 0 t k = Cd.dag.length)

/-- **Every one of the IR's slots and writers is the layer's** (the completeness half `StageSlots` leaves out): every
value the prover writes has its unit as its writer, and every gate of tile `t` and every operand of one is one of `t`'s
slots. Without it the layer could miss a late edge, and `DrawnLayer.Good` would hold for a tile whose words read a late,
skipped writer, which `UnitFixed` can't meet. `Declared.layer` meets it under `Declared.Wf` and `units ≤ N`. -/
def StageComplete {ι C B V : Type} {s N : ℕ} (Cd : Declared ι) (pf : DrawnLayer C B 1 Cd.rounds s (Fin N) ℕ ℕ V) :
    Prop :=
  (∀ ℓ, Cd.Writes ℓ → ∃ u, pf.src ℓ = some u ∧ u.2.val = Cd.unit ℓ) ∧
    ∀ (t : Fin N) (ℓ : ℕ), ℓ ∈ Cd.slots t.val → ∃ k, pf.pos 0 t k = ℓ

end Accounting.Compose
