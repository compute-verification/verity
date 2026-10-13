import Proofs.Flock.C3.Reps.ReadWhole
import Proofs.Flock.C3.Reps.WRead

/-!
# The residuals' check's whole reads at `wRead`'s parts: `ResReadCol`

`resReadCol`: `ResReadCol` holds where each part's session passed `RecResiduals.check` at the shape (`p₀ = p`), by
`ReadWhole`'s reads, which import no session.
-/

namespace FlockC3.Reps

open Flock
open FlockVBridge.Family (partsOf wIOf nsOf nvOf rowColOf partsOf_eq)
open FlockVBridge.OfVStar (rowCol partRep)
open FlockSoundness.Discharge.Recursion (VStmt)

/-- **An element of `w` a part reads is read whole at that part** (`ResReadCol`, at `p₀ = p` and `a₀ = a`): where each
part's session passed `RecResiduals.check` for that part at the shape, which refuses a unit reading a row-port element
in some but not all of its bits (`readCol_whole`). -/
theorem resReadCol {S : ℕ} {Vs : Fin S → VStmt} {sh : Flock.Shape} {sess : ℕ → Fin S}
    (hck : ∀ p < (partsOf sh).size, RecResiduals.check (Vs (sess p)).c = .ok () ∧
      (Vs (sess p)).c.unit.unitLog ≤ 32 ∧ ∃ m k cl sh₀,
        RecResiduals.params (Vs (sess p)).c.unit.name = some (m, k, cl, p) ∧ RecResiduals.shape? m k cl = some sh₀ ∧
        RecResiduals.holoOf (Vs (sess p)).c = .ok sh.holo ∧ sh = { sh₀ with holo := sh.holo }) :
    ResReadCol Vs sh sess := by
  intro p a b hp ha hb hr
  obtain ⟨hc, hu, m, k, cl, sh₀, hpar, hsh, hh, hsheq⟩ := hck p hp
  obtain ⟨S₁, ns, nv, hpart, hn, hw⟩ := ReadWhole.check_whole hc hpar hsh hh
  rw [← hsheq] at hpart
  obtain ⟨S0, gs, hS, hg⟩ := ReadWhole.part_ok hpart
  have hpo := partsOf_eq hS hg
  have hpg : p < gs.size := by rw [← hpo]; exact hp
  obtain ⟨-, hns, hnv⟩ := FlockVBridge.OfVStar.part_eq hS hg hpg hpart
  have hgp : (partsOf sh)[p]! = gs[p] := by rw [hpo]; exact getElem!_pos gs p hpg
  have hrc : ∀ x, rowColOf sh p x = rowCol (partRep ns nv).ports 0 x := fun x => by
    unfold rowColOf nsOf nvOf
    rw [hgp, hns, hnv]
  have hq : a < ns.sum := by
    have hsz : (wIOf sh p).size = ns.sum := by
      unfold wIOf
      rw [hgp, FlockVBridge.OfVStar.wIdx_size, hns]
    rw [← hsz]
    exact ha
  refine ⟨p, a, hp, ha, rfl, fun b₀ hb₀ => ?_⟩
  rw [hrc] at hr ⊢
  exact ReadWhole.readCol_whole (Vs (sess p)).hU hu hn hw hq hb hr hb₀

end FlockC3.Reps

#print axioms FlockC3.Reps.resReadCol
