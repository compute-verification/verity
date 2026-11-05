import Definitions.Flock.Guarantees.Statement
import Mathlib.Data.Matrix.Basis

/-!
# C-Flock's statement model: what a satisfying witness holds

Facts about the statements of `Definitions/Flock/Statement.lean`, over the model alone. A position whose row of `A₀`
is zero holds 0 in every block of every satisfying witness (`block_eq_zero`), which proves
`Flock.Guarantees.ZeroRowForced`.

Its witnesses are at `pinStmt`, one block of 64 bits pinned at position 0 whose only constraint is the pin's self row
`[0]·[0]` (`Matrix.single 0 0 1`), so row 1 of its `A₀` is zero. The witness that is 1 at the pin only satisfies it
(`ZeroRowForced_honest`). The witness that is 1 at the pin and at position 1 meets every check but `Satisfies`, so the
verifier that accepts everything accepts it, and it holds 1 at position 1 (`ZeroRowForced_nontrivial`). `pinStmt` is a
notation, not a definition, so the witnesses' statements read only the definitions tree.
-/

namespace FlockSoundness.Model.Statement

/-- A position whose row of `A₀` is zero holds 0 in every block of a satisfying witness: its constraint reads
`(A₀ z)_c · (B₀ z)_c = z_c`, and `(A₀ z)_c = 0`. -/
theorem block_eq_zero {S : Statement} {z : Witness S.m} (hs : S.Satisfies z) {c : Fin (2 ^ S.kLog)}
    (hA : S.A₀ c = 0) (o : Fin (2 ^ S.nbl)) : S.block z o c = 0 := by
  have h := hs.1 o c
  have ha : (S.A₀.mulVec (S.block z o)) c = 0 := by
    simp [Matrix.mulVec, dotProduct, fun j => congrFun hA j]
  rw [ha, zero_mul] at h
  exact h.symm

/-- At a statement with no regions whose `A₀` and `B₀` are the pin's self row `[pin]·[pin]`, a witness that is 1 at the
pin and 0 elsewhere in every block satisfies it. -/
theorem satisfies_of_pinOnly {S : Statement} (hA : S.A₀ = Matrix.single S.pin S.pin 1) (hB : S.B₀ = S.A₀)
    (hR : S.regions = []) {z : Witness S.m} (hz : ∀ o i, S.block z o i = if i = S.pin then 1 else 0) :
    S.Satisfies z := by
  refine ⟨fun o i => ?_, fun o => by simp [hz], fun R h => by simp [hR] at h⟩
  rw [hB, hA, Matrix.single_mulVec, hz o S.pin, hz o i]
  by_cases h : i = S.pin
  · subst h
    simp
  · simp [h]

end FlockSoundness.Model.Statement

/-- **A zero row of `A₀` forces its position to 0** (`block_eq_zero`). -/
theorem Flock.SecurityProofs.ZeroRowForced : Flock.Guarantees.ZeroRowForced :=
  fun _ _ hs _ hA o => FlockSoundness.Model.Statement.block_eq_zero hs hA o

namespace FlockSoundness.Model.Statement

/-- One block of 64 bits (`m = k_log = 6`), pinned at position 0, with no regions, whose `A₀` and `B₀` are the pin's
self row `[0]·[0]`. -/
local notation "pinStmt" => Statement.mk 6 6 le_rfl le_rfl 0 (Matrix.single 0 0 1) (Matrix.single 0 0 1) List.nil

private theorem pinStmt_block (z : Witness (Statement.m pinStmt)) (o : Fin (2 ^ Statement.nbl pinStmt))
    (i : Fin (2 ^ Statement.kLog pinStmt)) : Statement.block pinStmt z o i = z ⟨i.val, i.isLt⟩ := by
  show z (Statement.addr pinStmt o i) = _
  congr 1
  exact Fin.ext (by simp [Statement.addr])

/-- **`ZeroRowForced` holds at a statement with a zero row**: at `pinStmt`, the witness that is 1 at the pin only
satisfies it, and row 1 of its `A₀` is zero. -/
theorem _root_.Flock.SecurityProofs.ZeroRowForced_honest :
    Statement.Satisfies pinStmt (fun a => if a.val = 0 then 1 else 0) ∧ Statement.A₀ pinStmt 1 = 0 := by
  refine ⟨satisfies_of_pinOnly rfl rfl rfl fun o i => ?_, ?_⟩
  · rw [pinStmt_block]
    show (if i.val = 0 then (1 : ZMod 2) else 0) = if i = 0 then 1 else 0
    by_cases h : i.val = 0
    · have hi : i = 0 := Fin.ext h
      rw [ite_eq_left h, ite_eq_left hi]
    · have hi : i ≠ 0 := fun e => h (congrArg Fin.val e)
      rw [ite_eq_right h, ite_eq_right hi]
  · funext j
    show Matrix.single (0 : Fin (2 ^ 6)) (0 : Fin (2 ^ 6)) (1 : ZMod 2) 1 j = 0
    exact Matrix.single_apply_of_row_ne (by decide) _ _ _

/-- **`ZeroRowForced` fails at the verifier that accepts everything**: at `pinStmt`, the witness that is 1 at the pin
and at position 1 holds 1 at position 1, whose row of `A₀` is zero; only `Satisfies` refuses it. -/
theorem _root_.Flock.SecurityProofs.ZeroRowForced_nontrivial :
    ¬ Statement.block pinStmt (fun a => if a.val ≤ 1 then 1 else 0) 0 1 = 0 := by
  rw [pinStmt_block]
  decide

end FlockSoundness.Model.Statement
