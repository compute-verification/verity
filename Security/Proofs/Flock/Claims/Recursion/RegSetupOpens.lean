import Proofs.Flock.Claims.Recursion.RegSetup
import Proofs.Flock.Claims.Recursion.OpensReads
import Proofs.Flock.Soundness.Discharge.ZkReg.Checked

/-!
# The setup's registered openings open (`regOpenings_opens`)

Wherever the recursive setup accepts a V\* statement (`setupO`), it ran `checkRegistered` on each of its statements
(`OpensReads`' `RecIns.setup_registered`), so `checkPort` accepted each of the verifier's registered ports there, and
the row each instance's ref names opens the registered root at its position through its path (`checkPort_at`). So every
opening `regOpenings` lists opens (`regOpenings_opens`), and `bc_eq_of_setup` is `bc_eq_of_regOpenings` with nothing
about the openings assumed.
-/

namespace Flock.Registered

/-- **`checkPort` accepting, row by row**: the port is the circuit's first port `p` of its name, the header's entry
gives `positions` and `paths`, and every row of its table opens the registered root at its position through its path
(`badRow`). -/
theorem checkPort_rows {c : Circuit} {pub : Public} {t : Tables} {indices : Array Nat} {reg : Port}
    (h : checkPort c pub t indices reg = .ok ()) :
    ∃ p positions paths, (List.range c.ports.size).find? (fun p => c.ports[p]!.name == reg.name) = some p ∧
      entry pub reg = .ok (positions, paths) ∧ ∀ r < t.counts.getD p 0,
        opens reg (tableRow pub t p r) (positions.getD r 0) (paths.getD r ByteArray.empty) = true := by
  unfold checkPort at h
  simp only at h
  split at h
  · rename_i p hp
    split at h
    · exact absurd (except_bind_ok h) fun ⟨_, h1, _⟩ => throw_ne_ok _ _ h1
    · split at h
      · exact absurd (except_bind_ok h) fun ⟨_, h1, _⟩ => throw_ne_ok _ _ h1
      · obtain ⟨⟨positions, paths⟩, hent, h⟩ := except_bind_ok h
        simp only at h
        split at h
        · exact absurd (except_bind_ok h) fun ⟨_, h1, _⟩ => throw_ne_ok _ _ h1
        · split at h
          · exact absurd h (throw_ne_ok _ _)
          · rename_i hrow
            refine ⟨p, positions, paths, hp, hent, fun r hr => ?_⟩
            have ho := List.find?_eq_none.mp hrow r (List.mem_range.mpr hr)
            simpa using ho
  · exact absurd h (throw_ne_ok _ _)

end Flock.Registered

namespace RecDeploy

/-- **A statement `checkRegistered` accepts at the verifier's `own` opens every row `portOpenings` lists**: each is the
row an instance's ref names, which `checkPort` opens (`checkPort_at`). -/
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

/-- **Every opening the recursive setup lists opens.** -/
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

/-- **`bc_eq_of_regOpenings` at the setup's own openings**: two openings the setup lists at V\* statements of `xs`, at
one position under one tree, have one commit string wherever the true calls `l` and the setup's evaluations at `xs`
have no collision. -/
theorem bc_eq_of_setup {Coin : Type} {D : Claims.Recursion.Deploy Coin} (hD : RegListed D)
    {l : List (Claims.Msg × Claims.Msg)} (hl : ∀ e ∈ l, FlockDeploy.sha512 e.1 = e.2) {xs : List Claims.Msg}
    (hc : (Claims.findColl (l ++ Claims.Recursion.setupCalls FlockDeploy.sha512 D xs)).isSome = false)
    {xO xO' : Claims.Msg} (hx : xO ∈ xs) (hx' : xO' ∈ xs) {reg reg' : _root_.Flock.Registered.Port}
    {bc bc' path path' : ByteArray} {pos : ℕ} (ho : (reg, bc, pos, path) ∈ regOpenings xO)
    (ho' : (reg', bc', pos, path') ∈ regOpenings xO') (hR : _root_.Flock.Registered.SameRoot reg reg') : bc = bc' :=
  bc_eq_of_regOpenings hD hl hc hx hx' ho ho' hR (regOpenings_opens ho) (regOpenings_opens ho')

end RecDeploy
