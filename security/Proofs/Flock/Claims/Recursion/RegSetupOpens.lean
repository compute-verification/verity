module

public import Proofs.Flock.Claims.Recursion.RegSetup
public import Proofs.Flock.Claims.Recursion.OpensReads
public import Proofs.Flock.Soundness.Discharge.ZkReg.Checked

/-! The setup's registered openings open (`regOpenings_opens`), so `bc_eq_of_setup` assumes nothing about them. -/

@[expose] public section

namespace Flock.Registered

end Flock.Registered

namespace RecDeploy

theorem portOpenings_opens {st : _root_.Flock.Stmt} {own : Array _root_.Flock.Registered.Port}
    (h : _root_.Flock.checkRegistered st (some own) = .ok ()) {reg : _root_.Flock.Registered.Port}
    (hreg : reg ∈ own.toList) : ∀ o ∈ portOpenings st reg, _root_.Flock.Registered.opens reg o.1 o.2.1 o.2.2 = true := by
  obtain ⟨t, _, ht, -, hports⟩ :=
    FlockSoundness.Discharge.ZkReg.check_tables (FlockSoundness.Discharge.ZkReg.checkRegistered_ok h)
  obtain ⟨p, positions, paths, hp, he, hall⟩ :=
    FlockSoundness.Discharge.ZkReg.checkPort_at (hports reg (by simpa using hreg))
  intro o ho
  simp only [portOpenings, ht, hp, he, List.mem_map, List.mem_range] at ho
  obtain ⟨i, hi, rfl⟩ := ho
  exact (hall i hi).2.2

theorem regOpenings_opens {xO : Claims.Msg} {reg : _root_.Flock.Registered.Port} {bc path : ByteArray} {pos : ℕ}
    (ho : (reg, bc, pos, path) ∈ regOpenings xO) : _root_.Flock.Registered.opens reg bc pos path = true := by
  cases hS : setupO xO with
  | none => simp [regOpenings, hS] at ho
  | some v =>
    obtain ⟨s, spec, sts⟩ := v
    cases hown : s.base.registered with
    | none => simp [regOpenings, hS, hown] at ho
    | some own =>
      simp only [regOpenings, hS, hown, List.mem_flatMap, List.mem_map] at ho
      obtain ⟨st, hst, reg', hreg', o, hoo, he⟩ := ho
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj he
      have hcr := RecIns.setup_registered (setupO_some hS).2 st (by simpa using hst)
      rw [hown] at hcr
      exact portOpenings_opens hcr hreg' (bc, pos, path) hoo

theorem bc_eq_of_setup {Coin : Type} {D : Claims.Recursion.Deploy Coin} (hD : RegListed D)
    {l : List (Claims.Msg × Claims.Msg)} (hl : ∀ e ∈ l, FlockDeploy.sha512 e.1 = e.2) {xs : List Claims.Msg}
    (hc : (Claims.findColl (l ++ Claims.Recursion.setupCalls FlockDeploy.sha512 D xs)).isSome = false)
    {xO xO' : Claims.Msg} (hx : xO ∈ xs) (hx' : xO' ∈ xs) {reg reg' : _root_.Flock.Registered.Port}
    {bc bc' path path' : ByteArray} {pos : ℕ} (ho : (reg, bc, pos, path) ∈ regOpenings xO)
    (ho' : (reg', bc', pos, path') ∈ regOpenings xO') (hR : _root_.Flock.Registered.SameRoot reg reg') : bc = bc' :=
  bc_eq_of_regOpenings hD hl hc hx hx' ho ho' hR (regOpenings_opens ho) (regOpenings_opens ho')

end RecDeploy
