import Proofs.Accounting.TS.Session.Held.Layer

/-!
# A tile unit's proof layer with its Definition's checks, through its C-Flock sessions

`Accounting.TS.Checked.checkedLayer` reads its relation, `CheckedRel`, as `PouwH1TileZy_v1` with `ok = 1` only through
`Accounting.Served`'s prose: `CheckedRel` says the opened rows are one wire assignment the circuit's evaluation agrees
with, not what the tile's Definition computes. This layer states that, on words, at the same sessions:

* `TileView`: what unit `u` of a `PouwH1TileZy_v1` session reads, decoded from the session's registered reads: its `R`
  A rows and `C` B rows (each `K` codes and a scale) and its `R × C` z/y record. Words are integers as
  `Accounting.Served` reads them (codes s8, z s32, scales and y their encodings);
* `TileOk`: the Definition's checks at `ok = 1`, every cell with `z = Σ_k a_k b_k` and `y = dq [z, s_a, s_b]`, the
  tile counterpart of `Accounting.TS.Lanes.LaneOk`'s cells;
* `TileRel`: `CheckedRel`, and every drawn unit's decoded view meets `TileOk`, or the `ok` tie collides (`OkColl`);
* `tileLayer`: proof `i` of round `w` is accepted where flock-verify `--zk --registered` accepts its session at the
  registered file's known content (`Registered.checkKnownAll`) and the file names every port of the circuit
  (`Registered.checkComplete`), opens the committed rows at the registered reads (`openedReg`), and its relation is
  `TileRel`.

What ties the circuit to the Definition is the premise `Accounting.TS.Tile.TileComputes`, from which
`tileChecks_of_computes` gives `Accounting.Assumptions.TS.TileChecks`.

These definitions read `RowsSession`, in the proofs package, so they sit here under `reads_exempt`'s
`Proofs.Accounting.TS.Session`, as `Accounting.TS.Held`'s do.
-/

namespace Accounting.TS.Tile

open Accounting.TS Accounting.TS.Checked Accounting.TS.Held Flock.SoundAtBridge FlockSoundness.Discharge.ZkReg
open Pouw.RelativeSecurity (Paths)

/-- **What a tile unit reads**: its `R` A rows and `C` B rows, each `K` codes and a scale, and its z/y record,
row-major. -/
structure TileView (R C K : ℕ) where
  a : Fin R → Fin K → ℤ
  sa : Fin R → ℤ
  b : Fin C → Fin K → ℤ
  sb : Fin C → ℤ
  z : Fin R → Fin C → ℤ
  y : Fin R → Fin C → ℤ

/-- **The tile's checks at `ok = 1`**: every cell's `z` is the dot of its A row's and B row's codes, and its `y` is
`dq` of its `z` at the A row's and the B row's scales. -/
def TileOk {R C K : ℕ} (dq : List ℤ → ℤ) (v : TileView R C K) : Prop :=
  ∀ i j, v.z i j = ∑ k, v.a i k * v.b j k ∧ v.y i j = dq [v.z i j, v.sa i, v.sb j]

/-- **The tile unit's relation with its Definition's checks**: `CheckedRel`, and every drawn unit `u`'s view decoded
from the layer value `v` (`dec v u`) meets `TileOk`; or the `ok` tie collides (`OkColl`) at the `ok` port's known rows
in `known`. -/
def TileRel {R C K : ℕ} (s : RowsSession) (o : s.Out) (ok : String) (dq : List ℤ → ℤ)
    (dec : (ℕ → ℕ → Option Bool) → ℕ → TileView R C K) (known : Array (String × Array (Array ℕ × ByteArray)))
    (v : ℕ → ℕ → Option Bool) : Prop :=
  CheckedRel s o ok v ∧ ((∀ u ∈ o.2.1, TileOk dq (dec v u.val)) ∨ ∃ ks, (ok, ks) ∈ known ∧ OkColl s ok ks)

/-- **The tile layer through its C-Flock sessions**: proof `i` of round `w` is accepted where flock-verify
`--zk --registered` accepts its session at the registered file's known content `known x w i` (`LiveAcceptsZKR`, and
`Registered.checkKnownAll`) and under its complete check (`Registered.checkComplete`, run whenever `--registered` is
given: every port of the circuit is a port of `own`), opens the committed rows at the registered reads
(`openedReg`), and its relation is `TileRel` at the decode `dec x w i`. -/
noncomputable def tileLayer {Ω : Type} {S nπ R C K : ℕ} (sess : Paths Ω S → Fin S → Fin nπ → RowsSession)
    (out : ∀ x w i, (sess x w i).Out) (ok : String)
    (known : Paths Ω S → Fin S → Fin nπ → Array (String × Array (Array ℕ × ByteArray))) (dq : List ℤ → ℤ)
    (dec : Paths Ω S → Fin S → Fin nπ → (ℕ → ℕ → Option Bool) → ℕ → TileView R C K)
    (right : Paths Ω S → (Fin S → Fin nπ → ℕ → ℕ → Option Bool) → Prop) :
    ProofLayer Ω S nπ (ℕ → ℕ → Option Bool) where
  accepted x w i := LiveAcceptsZKR (sess x w i).I (sess x w i).own (out x w i) ∧
    Flock.Registered.checkKnownAll (sess x w i).own (known x w i) = .ok () ∧
    Flock.Registered.checkComplete (sess x w i).c (sess x w i).own = .ok ()
  opened x w i := openedReg (sess x w i)
  rel x w i v := TileRel (sess x w i) (out x w i) ok dq (dec x w i) (known x w i) v
  right := right

end Accounting.TS.Tile
