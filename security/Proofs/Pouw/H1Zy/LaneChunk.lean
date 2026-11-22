module

public import Definitions.Pouw.H1Zy.LaneChunk
public import Mathlib.Algebra.GroupWithZero.Defs
import Mathlib.Data.Rat.Cast.Order
import Mathlib.Tactic.Linarith.Frontend
import Mathlib.Tactic.NormNum.Abs
import Mathlib.Tactic.NormNum.DivMod
import Mathlib.Tactic.NormNum.OfScientific
import Mathlib.Tactic.Ring.RingNF

/-!
# A lane's chunks in sequence are the whole lane

`n` chunk statements of a lane (`Pouw.H1Zy.chunkOk`, `PouwH1LaneZyChunk_v1{TILE, n, c}`) read the lane's rows in
chunks (`joinIn`: each row's codes the chunks' concatenated, `joinRows`, its scales chunk 0's, and the last chunk's
record, partials, public inputs and index). The chain lemma (`run_join`, `zRun_join`): the whole `K`'s segments are the
chunks' in sequence, so the chain's state and each cell's z after chunk `c` are the whole lane's after its first
`(c + 1) K_c / 32` segments; from zero, `zRun` is `GemmAccumulatorS8`'s steps (`zRun_zero`).
-/

@[expose] public section

namespace Pouw.H1Zy

/-- **A lane's whole row from its chunks' rows**: entry `k` is entry `k mod K_c` of chunk `⌊k / K_c⌋` (`SPLITK`'s chunk
row `n i + c` holding entries `[c K_c, (c + 1) K_c)` of row `i`). -/
def joinRows {n Kc : ℕ} (x : Fin n → Fin Kc → BitVec 8) : Fin (n * Kc) → BitVec 8 := fun k =>
  have hK : 0 < Kc := Nat.pos_of_ne_zero fun h => by have := k.isLt; simp [h] at this
  x ⟨k.val / Kc, (Nat.div_lt_iff_lt_mul hK).2 k.isLt⟩ ⟨k.val % Kc, Nat.mod_lt _ hK⟩

/-- The last of `n > 0` chunks. -/
def lastFin (n : ℕ) (hn : 0 < n) : Fin n := ⟨n - 1, by omega⟩

/-- **The lane `n` chunks read**: each A and B row's codes the chunks' joined (`joinRows`), its scales chunk 0's, and the
last chunk's admit bits, record, partials, public inputs and index. -/
def joinIn {n Kc nw : ℕ} {P : Type} (xs : Fin n → ChunkIn Kc nw P) (hn : 0 < n) : LaneIn (n * Kc) nw P where
  a := fun s => joinRows fun c => (xs c).a s
  sa := (xs ⟨0, hn⟩).sa
  b := fun t => joinRows fun c => (xs c).b t
  sw := (xs ⟨0, hn⟩).sw
  admit := (xs (lastFin n hn)).admit
  zr := (xs (lastFin n hn)).zr
  yr := (xs (lastFin n hn)).yr
  partials := (xs (lastFin n hn)).partials
  pub := (xs (lastFin n hn)).pub
  lane := (xs (lastFin n hn)).lane

/-! ## The chain lemma -/

/-- Segment `c K_c / 32 + j` of a joined row is segment `j` of chunk `c`'s row. -/
theorem segOf_joinRows {n Kc : ℕ} (x : Fin n → Fin Kc → BitVec 8) (hKc : 32 ∣ Kc) (c : Fin n) {j : ℕ}
    (hj : j < Kc / 32) : segOf (joinRows x) (c.val * (Kc / 32) + j) = segOf (x c) j := by
  obtain ⟨q, rfl⟩ := hKc
  have hq : 32 * q / 32 = q := by omega
  rw [hq] at hj ⊢
  funext i
  have hi := i.isLt
  have hc := c.isLt
  have hlt' : 32 * j + i.val < 32 * q := by omega
  have key : 32 * (c.val * q + j) + i.val = (32 * j + i.val) + 32 * q * c.val := by ring
  have hlt : 32 * (c.val * q + j) + i.val < n * (32 * q) := by
    rw [key]
    have : c.val + 1 ≤ n := hc
    nlinarith
  have hpos : 0 < 32 * q := by omega
  simp only [segOf, codeAt, hlt, hlt', ↓reduceDIte, joinRows]
  congr 1
  · ext
    simp only [key, Nat.add_mul_div_left _ _ hpos, Nat.div_eq_of_lt hlt', zero_add]
  · ext
    simp only [key, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hlt']

/-- **The chain lemma**: the chain's state after `j` segments of chunk `c`, from the whole lane's state after its first
`c K_c / 32` segments, is the whole lane's after `c K_c / 32 + j`. -/
theorem run_join {n Kc nw : ℕ} {P : Type} (M : ChainSteps P nw) (p : P) (l : Fin 32) (hKc : 32 ∣ Kc)
    (a : Fin n → Fin 4 → Fin Kc → BitVec 8) (b : Fin n → Fin 8 → Fin Kc → BitVec 8) (σ : ChainSt) (c : Fin n) :
    ∀ j, j ≤ Kc / 32 →
      M.run p l (c.val * (Kc / 32)) (a c) (b c)
          (M.run p l 0 (fun s => joinRows fun d => a d s) (fun t => joinRows fun d => b d t) σ (c.val * (Kc / 32))) j =
        M.run p l 0 (fun s => joinRows fun d => a d s) (fun t => joinRows fun d => b d t) σ (c.val * (Kc / 32) + j)
  | 0, _ => rfl
  | j + 1, hj => by
    rw [← Nat.add_assoc]
    simp only [ChainSteps.run]
    rw [run_join M p l hKc a b σ c j (by omega), zero_add]
    simp only [segOf_joinRows _ hKc c (by omega : j < Kc / 32)]

/-- The same for a cell's z: `j` steps of chunk `c` from the whole lane's z after `c K_c / 32` steps. -/
theorem zRun_join {n Kc : ℕ} (x w : Fin n → Fin Kc → BitVec 8) (hKc : 32 ∣ Kc) (z : BitVec 32) (c : Fin n) :
    ∀ j, j ≤ Kc / 32 →
      zRun (x c) (w c) (zRun (joinRows x) (joinRows w) z (c.val * (Kc / 32))) j =
        zRun (joinRows x) (joinRows w) z (c.val * (Kc / 32) + j)
  | 0, _ => rfl
  | j + 1, hj => by
    rw [← Nat.add_assoc]
    simp only [zRun]
    rw [zRun_join x w hKc z c j (by omega), segOf_joinRows x hKc c (by omega), segOf_joinRows w hKc c (by omega)]

/-- From zero, `zRun` is `GemmAccumulatorS8`'s steps. -/
theorem zRun_zero {K : ℕ} (x w : Fin K → BitVec 8) : ∀ m, zRun x w 0 m = gemmAccSteps x w m
  | 0 => rfl
  | m + 1 => by simp only [zRun, gemmAccSteps, zRun_zero x w m]; rfl

/-! ## One chunk's checks -/

theorem mux_false_eq_true {b t : Bool} : mux b t false = true ↔ b = false ∧ t = true := by
  cases b <;> simp [mux]

theorem ChunkIn.l_eq {Kc nw : ℕ} {P : Type} {x y : ChunkIn Kc nw P} (h : x.lane = y.lane) : x.l = y.l :=
  Fin.ext (by simp only [ChunkIn.l, ChunkIn.sel, h])

/-! ## The induction over chunks -/

/-- A chunk binds where its lane does: at `n > 0` chunks of `K_c` entries, `n K_c < 2^17` gives `K_c < 2^17`, the lane
field's range at every chunk. -/
theorem chunk_lt {n Kc : ℕ} (hn : 0 < n) (hK : n * Kc < 2 ^ 17) : Kc < 2 ^ 17 :=
  lt_of_le_of_lt (Nat.le_mul_of_pos_left Kc hn) hK

end Pouw.H1Zy
