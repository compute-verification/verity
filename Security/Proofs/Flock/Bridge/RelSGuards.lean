import Proofs.Flock.Bridge.CircuitOf
import Proofs.Flock.Bridge.FaultyRightC

/-!
# `card_wrong_lt_of_relS` is neither vacuous nor trivial

The two guards a bound gets beside it (one adversary form):

* `card_wrong_lt_of_relS_honest`: its hypothesis holds at the honest instance. The witness that opens the circuit's
  evaluation on the honest inputs (`witDep C Cm H roots Xr`) is in `RelS` at `δ = C(n − K₀, kd)/C(n, kd)`, whenever the
  commitment opens those values, the constants `κ` fix inputs to their honest values, `1 ≤ K₀` and `1 ≤ kd ≤ n`: no
  unit of it is faulty (`faultyRightC`), so the design accepts it with probability `1 > δ`.
* `card_wrong_lt_of_relS_nontrivial`: its conclusion fails for an accept-everything verifier. On `notCircuit` (an
  input and its negation, the output, one unit), the witness that opens nothing leaves its one unit wrong, so "fewer
  than `1` wrong unit" fails there, and `RelS` refuses it.
-/

namespace Flock.Bridge

open Claims.Sampling

theorem choose_sub_lt_choose {n K₀ kd : ℕ} (hK : 1 ≤ K₀) (hkd : 1 ≤ kd) (hkdn : kd ≤ n) :
    (n - K₀).choose kd < n.choose kd := by
  obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
  obtain ⟨j, rfl⟩ : ∃ j, kd = j + 1 := ⟨kd - 1, by omega⟩
  have h1 : (m + 1 - K₀).choose (j + 1) ≤ m.choose (j + 1) := Nat.choose_le_choose _ (by omega)
  have h2 : 0 < m.choose j := Nat.choose_pos (by omega)
  have h3 : (m + 1).choose (j + 1) = m.choose j + m.choose (j + 1) := Nat.choose_succ_succ m j
  omega

/-- **`card_wrong_lt_of_relS`'s hypothesis holds at the honest instance**: the witness opening the evaluation on the
honest inputs `Xr` is in `RelS` at `δ = C(n − K₀, kd)/C(n, kd)`, when the commitment opens every value the constants
don't fix, the constants fix only inputs and agree with `Xr`'s evaluation, `1 ≤ K₀` and `1 ≤ kd ≤ n`. -/
theorem card_wrong_lt_of_relS_honest {C : FlockSoundness.Audit.Circuit} {n : ℕ}
    (P : FlockSoundness.Audit.Partition C n) (κ : Fin C.N → Option Bool) (kd K₀ : ℕ) (desc : Claims.Msg) (q : Query)
    (roots : List Claims.Msg) (H : Claims.Msg → Claims.Msg) (Cm : Commitment Bool) (Xr : Fin C.N → Bool)
    (hκin : ∀ g b, κ g = some b → (C.op g).isInput = true) (hκ : ∀ g b, κ g = some b → C.eval Xr g = b)
    (hcom : ∀ g, κ g = none → ∃ d : Claims.Msg, Cm.Opens H roots ⟨posOf C g, C.eval Xr g, d⟩ = true)
    (hK : 1 ≤ K₀) (hkd : 1 ≤ kd) (hkdn : kd ≤ n) :
    RelS H Cm (((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)) (stmtOf P κ kd desc q roots) (partOf P)
      (witDep C Cm H roots Xr) := by
  have hnf : (faulty H Cm (stmtOf P κ kd desc q roots) (partOf P) (witDep C Cm H roots Xr)).filter (· < n) = ∅ := by
    ext v
    simp only [Finset.mem_filter, Finset.notMem_empty, iff_false, not_and]
    intro hv hlt
    refine faultyRightC P (fun g : Fin C.N => g) (fun (b : Bool) (_ : Fin C.N) => b) (id : Bool → Bool)
      (id : Bool → Bool) (C.eval Xr) (fun _ _ => True) H Cm roots Xr κ ⟨v, hlt⟩ hκin
      ⟨C.eval Xr, fun _ _ => ⟨rfl, rfl, trivial⟩⟩ (fun g _ hk o ho _ => ?_) (fun g _ b hb => hκ g b hb) kd desc q hv
    change o = C.eval Xr g at ho
    subst ho
    exact hcom g hk
  have hpa : ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) <
      (designOf kd).Pa (partOf P) (faulty H Cm (stmtOf P κ kd desc q roots) (partOf P) (witDep C Cm H roots Xr)) := by
    rw [pa_partOf_window, window_zero_filter, hnf, Finset.card_empty, Nat.sub_zero]
    have hpos : (0 : ℝ) < n.choose kd := by exact_mod_cast Nat.choose_pos hkdn
    rw [div_self hpos.ne', div_lt_one hpos]
    exact_mod_cast choose_sub_lt_choose hK hkd hkdn
  exact hpa

/-- An input and its negation, the output. -/
def notCircuit : FlockSoundness.Audit.Circuit where
  N := 2
  op g := if g = 0 then .input else .not 0
  topo := by decide
  outputs := {1}

/-- `notCircuit`'s negation, its one unit. -/
def notPartition : FlockSoundness.Audit.Partition notCircuit 1 where
  unit g := if g.val = 0 then none else some 0
  input_iff := by decide

/-- **`card_wrong_lt_of_relS`'s conclusion fails for an accept-everything verifier**: on `notCircuit`, the witness that
opens nothing leaves its one unit wrong, so "fewer than `K₀ = 1` wrong units" fails, and `RelS` refuses that witness. -/
theorem card_wrong_lt_of_relS_nontrivial (kd : ℕ) (desc : Claims.Msg) (q : Query) (roots : List Claims.Msg)
    (H : Claims.Msg → Claims.Msg) (Cm : Commitment Bool) :
    ¬ (notPartition.wrong (openedOf notCircuit (fun _ => none)
        ((⟨[]⟩ : Wit).val H Cm (stmtOf notPartition (fun _ => none) kd desc q roots)))).card < 1 ∧
      ¬ RelS H Cm (((1 - 1).choose kd : ℝ) / ((1 : ℕ).choose kd : ℝ)) (stmtOf notPartition (fun _ => none) kd desc q roots)
        (partOf notPartition) ⟨[]⟩ := by
  have hX : openedOf notCircuit (fun _ => none)
      ((⟨[]⟩ : Wit).val H Cm (stmtOf notPartition (fun _ => none) kd desc q roots)) = fun _ => false := by
    funext g
    simp [openedOf, Wit.val]
  have hw : (notPartition.wrong (openedOf notCircuit (fun _ => none)
      ((⟨[]⟩ : Wit).val H Cm (stmtOf notPartition (fun _ => none) kd desc q roots)))).card ≥ 1 := by
    rw [hX]
    refine Finset.card_pos.2 ⟨0, notPartition.mem_wrong.2 fun hc => ?_⟩
    have h1u : notPartition.unit ⟨1, by decide⟩ = some 0 := rfl
    have h0u : notPartition.unit ⟨0, by decide⟩ ≠ some 0 := by decide
    have hop : notCircuit.op ⟨1, by decide⟩ = .not ⟨0, by decide⟩ := rfl
    have e := hc ⟨1, by decide⟩ (notPartition.mem_committed_of_output (by decide)) h1u
    rw [notPartition.localEval_of_mem _ h1u, hop] at e
    simp only [FlockSoundness.Audit.Op.apply] at e
    rw [notPartition.localEval_of_not_mem _ h0u] at e
    simp at e
  exact ⟨by omega, fun h => by
    have := card_wrong_lt_of_relS notPartition (fun _ => none) kd 1 desc q roots H Cm ⟨[]⟩ h
    omega⟩

end Flock.Bridge
