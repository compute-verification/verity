import Proofs.Flock.Soundness.Discharge.HiddenExec.SitesP
import Proofs.Flock.Soundness.Discharge.HiddenExec.KeyHP
import Proofs.Flock.Soundness.Discharge.HiddenExec.HeadlineExec

set_option Elab.async false

/-!
# The rows at the hidden tables at public inputs

`HeadlineExec`'s `layAtH`, `rsAtH` and `rsAtH_computes`, and `PlainReg`'s `sitesAtH_bcSiteS`, at the verifier's public
inputs `own`: the program is `Placed.keyProgP c pi hU n` (its unit inputs on the gates of their keys with public
inputs), its outputs `Layout.outsOfP`, the sites `sitesAtHP`, and the registration the population's public file `pop`
(`HmIn.loadPublic`, with or without shared rows):

* `sitesAtHP_bcSite`: `BcSiteHP` at every satisfiable site, from `tableHP_bcSite` at a draw flock-verify sets up and
  `tableAtHP_unsat` elsewhere;
* `layAtHP`, `rsAtHP`: `keyLayoutHP`, `keyRowsHP` at `sitesAtHP`, `KeysRowP` from the parse
  (`Layout.keysRowP_of_keysRow`, `Layout.keysRow_of_parse`);
* `rsAtHP_computes`: `keyRowsHP_computes`.
-/

namespace FlockSoundness.Discharge.HiddenExec

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec (Inputs)

variable {n : ℕ} {I : Inputs} {own : Option ByteArray} {mPts : ℕ} (ht : I.tags.typed = false) {c : Flock.Circuit}
  (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) (hscope : Layout.scopeOkH c = true)
  (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupHP I own mPts S₀ (dj S₀))
  (hU : Built c.unit) {pi : HmIn.Ports} (hpi : HmIn.parse c = .ok pi)
  {pop : Public} (hpub : HmIn.loadPublic I.tags c pi I.publicFile = .ok pop) {Reg : Type}

include hU hpi hpub in
/-- **The site fact at a drawn table at `own`** (`tableH_bcSiteS`): at a draw flock-verify sets up with
`Stmt.setupHidden … own`, each drawn unit's site is `BcSiteHP`: the draw `dj S` decodes to `D`, its proved units are in
its population, its public file loads (`HmIn.loadPublic`) to `pop`, and the unit's block and template instance index
its rank in `D.proved`, the template instance below `2^⌊log₂ c.g⌋` (`Layout.vu_lt_pow2`). -/
theorem tableHP_bcSite {S : Finset (Fin n)} (x : DrawSetupHP I own mPts S (dj S)) {u : Fin n} (h : u ∈ S) :
    BcSiteHP (x.table.acc ht hc (rowsFit_of_scopeH ht hc hscope)) pop (x.table.o u h) (x.table.vu u h) u.val := by
  have hcx : x.st.c = c := x.table.c_eq hc
  have hpx : (x.table.acc ht hc (rowsFit_of_scopeH ht hc hscope)).pi = pi :=
    (x.table.acc ht hc (rowsFit_of_scopeH ht hc hscope)).pi_eq (Hidden.parse_untyped ht hc).1 hpi
  refine ⟨dj S, x.D, rfl, x.hD, Layout.proved_lt_population x.hD, ?_, ?_, ?_⟩
  · show HmIn.loadPublic I.tags x.st.c (x.table.acc ht hc (rowsFit_of_scopeH ht hc hscope)).pi I.publicFile = .ok pop
    rw [hpx, hcx]
    exact hpub
  · show x.rank u % x.st.c.g < 2 ^ x.st.c.g.log2
    rw [hcx]
    exact Layout.vu_lt_pow2 (Hidden.parse_untyped ht hc).1 (Layout.scopeOkH_spec hU hscope).2.1
      (hcx ▸ Nat.mod_lt _ x.hg)
  · show x.D.proved[x.rank u / x.st.c.g * x.st.c.g + x.rank u % x.st.c.g]? = some u.val
    rw [Nat.div_add_mod']
    exact x.proved_rank h

include hU hpi hpub in
/-- **`BcSiteHP` at every satisfiable site** of the one-table hidden plan at `own`: at a draw flock-verify sets up by
`tableHP_bcSite`; at a refused draw no message satisfies the table (`tableAtHP_unsat`). -/
theorem sitesAtHP_bcSite :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planAtHP dj x₀) (tabAtHP dj x₀) S R u), Binding.Sat (planAtHP dj x₀) (tabAtHP dj x₀) S R u M →
        BcSiteHP ((sitesAtHP ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀).acc S R u h) pop
          ((sitesAtHP ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀).o S R u h)
          ((sitesAtHP ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀).vu S R u h) u.val := by
  intro S R u h M hM
  show BcSiteHP ((tableAtHP dj x₀ S).acc ht hc (rowsFit_of_scopeH ht hc hscope)) pop ((tableAtHP dj x₀ S).o u h)
    ((tableAtHP dj x₀ S).vu u h) u.val
  by_cases hS : Nonempty (DrawSetupHP I own mPts S (dj S))
  · have e : tableAtHP dj x₀ S = (Classical.choice hS).table := by
      unfold tableAtHP
      rw [dite_eq_left_of_eq_true (eq_true hS)]
    rw [e]
    exact tableHP_bcSite ht hc hscope dj hU hpi hpub _ h
  · exact absurd hM (tableAtHP_unsat dj x₀ hS _)

include ht hc hscope in
/-- **`KeysRowP` from `scopeOkH`**: `KeysRow` from the parse (`Layout.keysRow_of_parse`), at public inputs. -/
theorem keysRowP_of_scopeH :
    Layout.KeysRowP c hU pi (Layout.portRowW c) :=
  Layout.keysRowP_of_keysRow (Layout.keysRow_of_parse (Hidden.parse_untyped ht hc).1
    (Layout.scopeOkH_spec hU hscope).1 (Layout.scopeOkH_spec hU hscope).2.2)

/-- **`lay` at the hidden tables at `own`**: `keyLayoutHP` at `sitesAtHP`, the program's outputs `Layout.outsOfP`, the
registration the population's public file. -/
noncomputable abbrev layAtHP :=
  keyLayoutHP (Reg := Reg) (sitesAtHP_parse ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀) hpi
    (Layout.outsOfP c hU n pi) (fun _ => pop) (Layout.scopeOkH_spec hU hscope).2.1 (rowsFit_of_scopeH ht hc hscope)
    (Layout.outsOutP_outsOfP c hU n pi)

/-- **`rs` at the hidden tables at `own`**: `keyRowsHP` at `sitesAtHP`, `hsite` from `sitesAtHP_bcSite` and `KeysRowP`
from `keysRowP_of_scopeH`. -/
noncomputable abbrev rsAtHP : Binding.HmRowsX (layAtHP ht hc hscope dj x₀ hU hpi (pop := pop) (Reg := Reg)) :=
  keyRowsHP (sitesAtHP_parse ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀) hpi (Layout.outsOfP c hU n pi)
    (fun _ => pop) (Layout.scopeOkH_spec hU hscope).2.1 (rowsFit_of_scopeH ht hc hscope)
    (Layout.outsOutP_outsOfP c hU n pi) (sitesAtHP_bcSite ht hc hscope dj x₀ hU hpi hpub)
    (keysRowP_of_scopeH ht hc hscope hU)

/-- **`hHm` at the hidden tables at `own`** (`keyRowsHP_computes`). -/
theorem rsAtHP_computes : (rsAtHP ht hc hscope dj x₀ hU hpi hpub (Reg := Reg)).Computes :=
  keyRowsHP_computes (sitesAtHP_parse ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀) hpi (Layout.outsOfP c hU n pi)
    (fun _ => pop) (Layout.scopeOkH_spec hU hscope).2.1 (rowsFit_of_scopeH ht hc hscope)
    (Layout.outsOutP_outsOfP c hU n pi) (sitesAtHP_bcSite ht hc hscope dj x₀ hU hpi hpub)
    (keysRowP_of_scopeH ht hc hscope hU)

end FlockSoundness.Discharge.HiddenExec
