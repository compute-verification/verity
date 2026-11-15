module

public import Definitions.Accounting.Gates.Circuit
public import Proofs.Accounting.Gates
public import Proofs.Core.Circuit

/-!
# The circuit model is a core circuit

`toCircuit_eval`: on an input `x` in its window, `C.toCircuit` evaluated on `x`'s words (`IntDag.inputWords`) reads
back, at every gate, as the gate's value. By strong induction over the gates: an input gate holds its slot's word, and
a computed gate applies its operation to its operands' values, which the gates below it hold (`getD_values_take`,
`val_eq`; an operand at or past it reads `0` on both sides), and reads no input slot (`raw_of_not_input`). In the
window every value lies in its declared interval (`value_eq_raw`), whose words read back (`dec_enc`).
-/

@[expose] public section

namespace Accounting.SecurityProofs.Gates

open Accounting.Gates

variable {ι : Type}

/-- A value in a gate's declared interval reads back from its word. -/
lemma dec_enc (g : IntGate ι) {z : ℤ} (h1 : g.lo ≤ z) (h2 : z ≤ g.hi) : g.dec (g.enc z) = z := by
  have hsz : (z - g.lo).toNat < g.size := by
    unfold IntGate.size
    omega
  have hs : (z - g.lo).toNat < 2 ^ g.bits := lt_of_lt_of_le hsz (Nat.le_pow_clog (by norm_num) _)
  unfold IntGate.dec IntGate.enc
  rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt hs, Int.toNat_of_nonneg (by omega)]
  omega

/-- An operation that is not an input reads no input slot. -/
lemma raw_of_not_input (o : IntOp ι) (h : o.isInput = false) (x x' : ι → ℤ) (v : List ℤ) :
    o.raw x v = o.raw x' v := by
  cases o <;> simp_all [IntOp.isInput, IntOp.raw]

/-- **The model is a core circuit**: on an input `x` in its window, `C.toCircuit` evaluated on `x`'s words reads back
as `C`'s values, gate by gate. -/
theorem toCircuit_eval (C : IntDag ι) (x : ι → ℤ) (hw : InWindow C x) :
    ∀ i (h : i < C.length), C[i].dec (C.toCircuit.eval (C.inputWords x) ⟨i, h⟩) = val C x i := by
  intro i
  induction i using Nat.strong_induction_on with
  | _ i ih => ?_
  intro h
  have hok := hw i h
  have hval : val C x i = C[i].raw x (values (C.take i) x) := by
    rw [val_eq C x i h, value_eq_raw _ x _ hok]
  obtain ⟨hlo0, hhi0, -, -⟩ := id hok
  have hlo : C[i].lo ≤ val C x i := hval ▸ hlo0
  have hhi : val C x i ≤ C[i].hi := hval ▸ hhi0
  cases hi : C[i].op.isInput
  · rw [C.toCircuit.eval_op (C.inputWords x) (g := ⟨i, h⟩) hi]
    show C[i].dec (C[i].enc (C[i].op.compute
      (C[i].args.map (C.readBack (C.toCircuit.eval (C.inputWords x)) ⟨i, h⟩)))) = val C x i
    have hops : C[i].args.map (C.readBack (C.toCircuit.eval (C.inputWords x)) ⟨i, h⟩) =
        C[i].operands (values (C.take i) x) := by
      unfold IntGate.operands
      apply List.map_congr_left
      intro j _
      unfold IntDag.readBack
      by_cases hlt : j < i
      · simp only [hlt, ↓reduceDIte]
        rw [getD_values_take C x j (hlt.trans h) i hlt h.le, ← val_eq C x j (hlt.trans h)]
        exact ih j hlt (hlt.trans h)
      · simp only [hlt, ↓reduceDIte]
        have hlen : (values (C.take i) x).length ≤ j := by
          rw [values_take_length C x i h.le]
          omega
        simp [List.getD_eq_getElem?_getD, List.getElem?_eq_none hlen]
    have hc : C[i].op.compute (C[i].operands (values (C.take i) x)) = val C x i := by
      rw [hval]
      unfold IntOp.compute IntGate.raw
      rw [raw_of_not_input _ hi (fun _ => 0) x]
      exact value_eq_raw _ x _ hok
    rw [hops, hc]
    exact dec_enc _ hlo hhi
  · rw [C.toCircuit.eval_input (C.inputWords x) (g := ⟨i, h⟩) hi]
    show C[i].dec (C[i].enc (C[i].op.raw x [])) = val C x i
    have hv : C[i].op.raw x [] = val C x i := by
      rw [hval]
      unfold IntGate.raw
      cases hop : C[i].op <;> simp [hop, IntOp.isInput, IntOp.raw] at hi ⊢
    rw [hv]
    exact dec_enc _ hlo hhi

end Accounting.SecurityProofs.Gates
