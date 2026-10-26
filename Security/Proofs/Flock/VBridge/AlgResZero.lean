import Proofs.Flock.VBridge.AlgHolds

/-!
# `AlgebraHolds` from each part's residuals vanishing

`algebraHolds_of_resZero`: `algebraHolds_of_parts` with each part's residuals given as values, not as registered output
columns. When every part `p` of S's groups (`RecResiduals.part sh p`) has its residuals zero at elements `E` and
verifier values `vv` that read the rep's `w` at `wIdx` of the group's ports and the verifier's coefficients `v` at the
group's `V`, `AlgebraHolds` holds. `vv`'s agreement with `v` is the tie a caller takes from check 5 (`ResTie`) at the
verifier's values; it stays a hypothesis here.
-/

namespace FlockVBridge.OfVStar

open Flock.HmNets hiding and
open Flock FlockSoundness.Discharge.Hm
open Flock.RecResiduals (portSizes groups partS)
open Flock.RecAlgebra (structureOf verifierRows el)
open FlockVBridge.PartS (wIdx)
open FlockVBridge.Compiled (RepData AlgebraHolds)

/-- **The algebra holds when every part's residuals vanish at the rep's values.** At S (`structureOf`) with its wire
bits in range (`BitsOk`), the verifier's coefficients `v` at the rep's coins (`verifierRows`), the sizes and S's groups:
if for every group `p` the verifier's part `p` has every residual zero at elements `E` and verifier values `vv` that
are the rep's `w` at `wIdx` of the group's ports and `v` at its `V`, every residual of S is zero. -/
theorem algebraHolds_of_resZero {sh : Shape} {d : RepData} {S : RecAlgebra.GfS} (hS : structureOf sh = .ok S)
    (hbits : S.BitsOk) {v : Array F128} {info kappa}
    (hvr : verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa))
    (hw : d.w.size = S.nw) (hc : d.comb.size = 64) (hv : v.size = S.nv)
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hg : groups S (portSizes sh) = .ok gs)
    (hparts : ∀ p (hp : p < gs.size), ∃ S' ns nv, RecResiduals.part sh p = .ok (S', ns, nv) ∧
      ∃ E vv : ℕ → F128,
        (∀ k (hk : k < (wIdx (portSizes sh) gs[p].2.1).size), E k = el d.w (wIdx (portSizes sh) gs[p].2.1)[k]) ∧
        (∀ k (hk : k < gs[p].2.2.size), vv k = el v gs[p].2.2[k]) ∧
        ∀ q (hq : q < S'.res.length), resVal (gfsOf S') (termVal E vv) (resOf S'.res[q]) = 0) :
    AlgebraHolds sh d := by
  obtain ⟨rs, hrs, hz⟩ := FlockVBridge.Parts.residuals_zero_of_parts hbits hg hw hv fun j hj => by
    obtain ⟨S', ns, nv, hpart, E, vv, hE, hvv, h0⟩ := hparts j hj
    obtain ⟨hS', -, -⟩ := part_eq hS hg hj hpart
    refine ⟨S', hS', fun k hk => ?_⟩
    obtain ⟨_, e1⟩ := FlockVBridge.PartS.res_partS hS' (w := el d.w) (v := el v) hE hvv
    obtain ⟨_, e2⟩ := FlockVBridge.PartS.res_partS hS' (w := el d.w) (v := el v)
      (w' := fun k => el d.w ((wIdx (portSizes sh) gs[j].2.1).getD k 0)) (v' := fun k => el v (gs[j].2.2.getD k 0))
      (fun k hk => by simp only [Array.getD, hk, ↓reduceDIte]; rfl)
      (fun k hk => by simp only [Array.getD, hk, ↓reduceDIte]; rfl)
    exact (e2 k hk).trans ((e1 k hk).symm.trans (h0 k hk))
  exact ⟨v, info, kappa, S, rs, hvr, hS, hw, hc, hrs, hz⟩

end FlockVBridge.OfVStar
