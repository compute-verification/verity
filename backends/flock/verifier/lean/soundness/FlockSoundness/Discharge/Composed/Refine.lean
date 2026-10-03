import FlockSoundness.Refine.Basis
import FlockSoundness.Refine.Opening

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
  sorry

end FlockSoundness.Discharge.Composed
