import Proofs.Flock.Soundness.Refine.Basis
import Proofs.Flock.Soundness.Refine.Opening

/-!
# The inner constraints' helpers are the executable's

`innerOf`'s constraints read the model's `comb`, `foldAll`, `recombine`, `phi`, `eqAt` and `execArith`. Each is what
`Zk.verifyRep` computes:

* `comb`, `foldAll`: `folded_comb` below, on `Zk.replay`'s `st.fold.folded α β (skipWeights z) ρ ts`.
* `recombine`: `Refine.recombine_eq`, on `Flock.ringSwitch`'s `dot (eqTable rpp) t`, which `Zk.verifyRep` calls.
* `phi`: `Refine.contract_eq` and `Refine.bfc_get`, on `Flock.basis`, which `Zk.verifyRep` passes to `ZkLigerito.ligerito`.
* `eqAt`: `Refine.eqTable_vecOf` (`FlockLevel3.eqTable_eqAt`), on `Flock.eqTable`.
* the field arithmetic: `execArith_correct`, with `Refine.fOf_add`, `fOf_mul`, `fOf_inv` and `Refine.gOf_add`, `gOf_mul`,
  `gOf_smul`.
-/

namespace FlockSoundness.Discharge.Composed

open Flock FlockLevel3 Refine

local notation "𝔽" => FlockLevel3.GF128

/-- **The lincheck's circuit fold is the model's** (§10): when `Zk.replay`'s `st.fold.folded α β (skipWeights z) ρ ts`
returns `w`, on a fold realizing `S`, each of `w`'s 64 entries is the model's `foldAll` at the challenges `ts` of `comb` at
`(z, α, β, ρ)`. -/
theorem folded_comb (S : Model.Statement) (c : CircuitFold) (hc : FoldRealizes c S) (α β z : F128)
    (ρ ts w : Array F128) (hρ : ρ.size = S.kLog - 6) (hts : ts.size = S.kLog - 6)
    (h : c.folded α β (skipWeights z) ρ ts = .ok w) :
    ∀ i < 64, fOf w[i]! = Model.foldAll (ts.toList.map fOf) (2 ^ S.kLog)
      (Model.comb execArith S (fOf z) (fOf α) (fOf β) (vecOf (S.kLog - 6) ρ)) i := by
  have hk6 : 6 ≤ S.kLog := S.hk6
  obtain ⟨v, hv, hvsz, hwv⟩ := hc.folded _ _ _ _ _ w h
  rw [hc.pin] at hwv
  have hsz : (v.modify S.pin.val fun x => x + β).size = 2 ^ S.kLog := by simp [hvsz]
  have hlen : ts.toList.length = S.kLog - 6 := by simp [hts]
  have hdiv : 2 ^ S.kLog / 2 ^ (S.kLog - 6) = 64 := by
    rw [Nat.pow_div (by omega) (by norm_num), show S.kLog - (S.kLog - 6) = 6 by omega]
  intro i hi
  have hi' : i < (v.modify S.pin.val fun x => x + β).size / 2 ^ ts.toList.length := by
    rw [hsz, hlen, hdiv]; exact hi
  rw [hwv, ← Array.foldl_toList, foldl_halve ts.toList _ i hi', hsz]
  refine foldAll_congr _ (2 ^ S.kLog) _ _ (fun col hcol => ?_) i (by rw [List.length_map, hlen, hdiv]; exact hi)
  have hfr := hc.fold α (skipWeights z) ρ v (skipWeights_size _) hρ hv ⟨col, hcol⟩
  rw [modify_get! v S.pin.val col _ (by omega), Model.comb, dite_eq_left hcol]
  have hsum : fOf v[col]! = ∑ i : Fin (2 ^ S.kLog), execArith.lagS (fOf z) ⟨i.val % 64, Nat.mod_lt _ (by norm_num)⟩ *
      Model.eqAt (vecOf (S.kLog - 6) ρ) (i.val / 64) *
        (fOf α * ((S.A₀ i ⟨col, hcol⟩).val : 𝔽) + ((S.B₀ i ⟨col, hcol⟩).val : 𝔽)) := by
    rw [hfr]
    rfl
  by_cases hp : S.pin.val = col
  · rw [ite_eq_left hp, ite_eq_left hp.symm, fOf_add, hsum]
  · rw [ite_eq_right hp, ite_eq_right (Ne.symm hp), add_zero, hsum]

end FlockSoundness.Discharge.Composed
