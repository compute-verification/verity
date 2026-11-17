module

public import Proofs.Flock.Soundness.Discharge.HiddenExec.SitesP
public import Proofs.Flock.Soundness.Discharge.HiddenExec.KeyHP
public import Proofs.Flock.Soundness.Discharge.HiddenExec.HeadlineExec
public import Proofs.Flock.Soundness.Discharge.Hidden.UnrolledP
public import Proofs.Flock.Soundness.Discharge.ZkHidden.BindStmt

@[expose] public section

set_option Elab.async false

/-! The rows at the hidden tables at the verifier's own public input words `own` (`layAtHP`, `rsAtHP`, `rsAtHP_computes`). -/

namespace FlockSoundness.Discharge.HiddenExec

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec (Inputs)

variable {n : ℕ} {I : Inputs} {own : Option ByteArray} {mPts : ℕ} (ht : I.tags.typed = false) {c : Flock.Circuit}
  (hcS : HmOut.parseStmt I.tags I.circuitFile I.tables = .ok c) (hscope : Layout.scopeOkH c = true)
  (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : SampleSetupHP I own mPts S₀ (dj S₀))
  (hU : Built c.unit) {pi : HmIn.Ports} (hpi : HmIn.parse c = .ok pi)
  {pop : Public} (hpub : HmIn.loadPublic I.tags c pi I.publicFile = .ok pop) {Reg : Type}

include hU hpi hpub in
theorem tableHP_bcSite {S : Finset (Fin n)} (x : SampleSetupHP I own mPts S (dj S)) {u : Fin n} (h : u ∈ S) :
    BcSiteHP (x.table.acc ht hcS (rowsFit_of_stmt ht hcS hscope)) pop (x.table.o u h) (x.table.vu u h) u.val := by
  have hcx : HmOut.unrolled x.st.c = c := x.table.unrolled_eq ht hcS
  have hC : Hidden.Committed I.tags I.circuitFile I.tables x.st.c := x.table.committed
  have hpx : (x.table.acc ht hcS (rowsFit_of_stmt ht hcS hscope)).pi = pi :=
    (x.table.acc ht hcS (rowsFit_of_stmt ht hcS hscope)).pi_eqU hcx hpi
  refine ⟨dj S, x.D, rfl, x.hD, Layout.proved_lt_population x.hD, ?_, ?_, ?_⟩
  · show HmIn.loadPublic I.tags x.st.c (x.table.acc ht hcS (rowsFit_of_stmt ht hcS hscope)).pi I.publicFile = .ok pop
    rw [hpx, ← Hidden.loadPublicIn_commit ht hC, hcx]
    exact hpub
  · show x.rank u % x.st.c.g < 2 ^ x.st.c.g.log2
    rw [← unrolled_g x.st.c, hcx]
    exact Layout.vu_lt_pow2_stmt ht hcS (Layout.scopeOkH_spec hU hscope).2.1
      (Nat.mod_lt _ (by rw [← hcx, unrolled_g]; exact x.hg))
  · show x.D.proved[x.rank u / x.st.c.g * x.st.c.g + x.rank u % x.st.c.g]? = some u.val
    rw [Nat.div_add_mod']
    exact x.proved_rank h

include hU hpi hpub in
theorem sitesAtHP_bcSite :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planAtHP dj x₀) (tabAtHP dj x₀) S R u), Binding.Sat (planAtHP dj x₀) (tabAtHP dj x₀) S R u M →
        BcSiteHP ((sitesAtHP ht hcS (rowsFit_of_stmt ht hcS hscope) dj x₀).acc S R u h) pop
          ((sitesAtHP ht hcS (rowsFit_of_stmt ht hcS hscope) dj x₀).o S R u h)
          ((sitesAtHP ht hcS (rowsFit_of_stmt ht hcS hscope) dj x₀).vu S R u h) u.val := by
  intro S R u h M hM
  show BcSiteHP ((tableAtHP dj x₀ S).acc ht hcS (rowsFit_of_stmt ht hcS hscope)) pop ((tableAtHP dj x₀ S).o u h)
    ((tableAtHP dj x₀ S).vu u h) u.val
  by_cases hS : Nonempty (SampleSetupHP I own mPts S (dj S))
  · have e : tableAtHP dj x₀ S = (Classical.choice hS).table := by
      unfold tableAtHP
      rw [dite_eq_left_of_eq_true (eq_true hS)]
    rw [e]
    exact tableHP_bcSite ht hcS hscope dj hU hpi hpub _ h
  · exact absurd hM (tableAtHP_unsat dj x₀ hS _)

include ht hcS hscope in
theorem keysRowP_of_scopeH :
    Layout.KeysRowP c hU pi (Layout.portRowW c) :=
  Layout.keysRowP_of_keysRow (Hidden.keysRow_stmt ht hcS (Layout.scopeOkH_spec hU hscope).1
    (Layout.scopeOkH_spec hU hscope).2.2)

/-- **`lay` at the hidden tables at `own`**: `keyLayoutHP` at `sitesAtHP`, the program's outputs `Layout.outsOfP`, the
registration the population's public file. -/
noncomputable abbrev layAtHP :=
  keyLayoutHP (Reg := Reg) (sitesAtHP_parse ht hcS (rowsFit_of_stmt ht hcS hscope) dj x₀) hpi
    (Layout.outsOfP c hU n pi) (fun _ => pop) (Layout.scopeOkH_spec hU hscope).2.1 (rowsFit_of_stmt ht hcS hscope)
    (Layout.outsOutP_outsOfP c hU n pi)

/-- **`rs` at the hidden tables at `own`**: `keyRowsHP` at `sitesAtHP`, `hsite` from `sitesAtHP_bcSite` and `KeysRowP`
from `keysRowP_of_scopeH`. -/
noncomputable abbrev rsAtHP : Binding.HmRowsX (layAtHP ht hcS hscope dj x₀ hU hpi (pop := pop) (Reg := Reg)) :=
  keyRowsHP (sitesAtHP_parse ht hcS (rowsFit_of_stmt ht hcS hscope) dj x₀) hpi (Layout.outsOfP c hU n pi)
    (fun _ => pop) (Layout.scopeOkH_spec hU hscope).2.1 (rowsFit_of_stmt ht hcS hscope)
    (Layout.outsOutP_outsOfP c hU n pi) (sitesAtHP_bcSite ht hcS hscope dj x₀ hU hpi hpub)
    (keysRowP_of_scopeH ht hcS hscope hU)

theorem rsAtHP_computes : (rsAtHP ht hcS hscope dj x₀ hU hpi hpub (Reg := Reg)).Computes :=
  keyRowsHP_computes (sitesAtHP_parse ht hcS (rowsFit_of_stmt ht hcS hscope) dj x₀) hpi (Layout.outsOfP c hU n pi)
    (fun _ => pop) (Layout.scopeOkH_spec hU hscope).2.1 (rowsFit_of_stmt ht hcS hscope)
    (Layout.outsOutP_outsOfP c hU n pi) (sitesAtHP_bcSite ht hcS hscope dj x₀ hU hpi hpub)
    (keysRowP_of_scopeH ht hcS hscope hU)

end FlockSoundness.Discharge.HiddenExec
