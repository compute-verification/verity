import Proofs.Flock.Soundness.Discharge.ZkRegV2.Scope
import Proofs.Flock.Verifier.Registered
import Proofs.Flock.Soundness.ExecSetup

set_option Elab.async false

/-!
# The registered check against `Layout.scopeOk` and `scopeOkV2`

`check_refuses_scopeOk`: at a circuit `Layout.scopeOk` admits, `Registered.check` refuses every non-empty list of
registered ports, because `scopeOk`'s rows are `sha512/row/v1` rows and `Registered.Port.fits` admits only
`sha512/row/v2` bit rows; so a theorem that takes both `scopeOk c` and the registered check's acceptance holds of no
session with a registered port. `fits_of_scopeOkV2`: each row of a circuit `scopeOkV2` admits is one `Port.fits` admits,
at a registered port of its bits, so `scopeOkV2` and `fits` don't exclude each other.
-/

namespace FlockSoundness.Discharge.ZkRegV2

open Flock

/-- **`Layout.scopeOk`'s rows are never rows the registered check admits.** -/
theorem not_fits_of_scopeOk {c : Circuit} (h : Layout.scopeOk c = true) (reg : Registered.Port) {q : Flock.Port}
    (hq : q ∈ c.ports.toList) : reg.fits q = false := by
  have hv := (Layout.scopeOk_ports h q hq).1
  simp [Registered.Port.fits, hv]

/-- **At a `Layout.scopeOk` circuit the registered check refuses every non-empty `own`.** -/
theorem check_refuses_scopeOk {c : Circuit} {pub : Public} {own : Array Registered.Port} {d : Bool}
    (hscope : Layout.scopeOk c = true) (hown : 0 < own.size) : Registered.check c pub (some own) d ≠ .ok () := by
  intro h
  obtain ⟨-, _, -, hall⟩ := Registered.check_ok h
  obtain ⟨p, hp, -, hfit, -⟩ := hall own[0] (Array.getElem_mem hown)
  rw [not_fits_of_scopeOk hscope _ (getElem_mem_toList' hp)] at hfit
  exact Bool.false_ne_true hfit

/-- **`scopeOkV2`'s rows are rows the registered check admits**: at a `scopeOkV2` circuit, a registered port whose rows
are `words * wordBits` bits fits each of the circuit's rows of that many bits. -/
theorem fits_of_scopeOkV2 {c : Circuit} (h : scopeOkV2 c = true) (reg : Registered.Port) {q : Flock.Port}
    (hq : q ∈ c.ports.toList) (hb : q.bits = reg.words * reg.wordBits) : reg.fits q = true := by
  obtain ⟨hv, hs, -, -⟩ := scopeOkV2_ports h q hq
  simp [Registered.Port.fits, hv, hs, hb]

end FlockSoundness.Discharge.ZkRegV2
