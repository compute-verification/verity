import FlockSoundness.Discharge.Layout.Scope
import FlockSoundness.Protocol.Discharge.HiddenExec.KeyH
import FlockSoundness.Protocol.Discharge.HiddenExec.Sites

set_option Elab.async false

/-!
# The headline at the executable over hidden-output sessions, with every open guarantee named

#828's `flock_headline_exec` (`Integrate/Headline.lean`) for the statement main's prover writes since #757
(`hiddenOutputs = true`, `Stmt.setupHidden`). `flock_headline_exec_hidden` instantiates `flock_headline_hidden` at the
production program `Placed.keyProg c hU n`, its returned outputs' bits for the program's outputs (`Layout.outsOf`), and
the one-table plan of the session's inputs (`planAtH`), with:
* `pp := ppH`, `hConst := hConstH`, `hZero := hZeroH` (`Sites.lean`);
* `lay := keyLayoutH` (`layAtH`) and `rs := keyRowsH` (`rsAtH`) at row widths `16 · words` (`KeyH.lean`): every row of
  the circuit, input or output, is a commit string whose `b ‖ c` the public file registers, and a unit's computed
  committed gate is a bit of an output row; `hsite` from `sitesAtH_bcSite`;
* `hHm := keyRowsH_computes` (`rsAtH_computes`).

So the hypotheses left are those of #828's headline: `hscope`, `hdec`, A3, `cr/sha-512`, `ecr/sha-512` and the numeric
conditions. flock-verify refuses a statement failing `hscope` or the statement's other form hypotheses (PROTOCOL.md
§16.15), and `FailClosed.flock_headline_exec_hidden` takes none of them.
-/

namespace FlockSoundness.Discharge.HiddenExec

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec
open scoped ENNReal

/-- `rs`'s scope gives `sitesAtH`'s rows: every port of a circuit in scope is a `sha512/row/v1` row. -/
theorem v2_of_scope {c : Flock.Circuit} (hscope : Layout.scopeOk c = true) : ∀ p ∈ c.ports.toList, p.v2 = false :=
  fun p hp => (Layout.scopeOk_ports hscope p hp).1

variable {n : ℕ} {I : Inputs} {mPts : ℕ} (ht : I.tags.typed = false) {c : Flock.Circuit}
  (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) (hscope : Layout.scopeOk c = true)
  (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) (hU : Built c.unit)
  {pub : Public} (hpub : HmOut.loadPublic I.tags c I.publicFile = .ok pub) (hpt : pub.tables = none) {Reg : Type}

include hU hpub in
/-- **The site fact at a drawn table under hidden outputs**: at a draw flock-verify sets up with `Stmt.setupHidden`,
each drawn unit's site is `BcSiteH`: the acceptance's draw is `dj S`, which decodes to `D`, its regions are the
executable's, its public file loads (`HmOut.loadPublic`) to `pub`, with or without shared rows, and the unit's block
and VU index its rank in `D.proved`, the VU below `2^⌊log₂ c.g⌋` (`Layout.vu_lt_pow2`). -/
theorem tableH_bcSiteS {S : Finset (Fin n)} (x : DrawSetupH I mPts S (dj S)) {u : Fin n} (h : u ∈ S) :
    BcSiteH (x.table.acc ht hc (v2_of_scope hscope)) pub (x.table.o u h) (x.table.vu u h) u.val := by
  have hcx : x.st.c = c := x.table.c_eq hc
  refine ⟨dj S, x.D, rfl, x.hD, Layout.proved_lt_population x.hD, fun r hr => List.mem_map_of_mem hr, ?_, ?_, ?_⟩
  · show HmOut.loadPublic I.tags x.st.c I.publicFile = .ok pub
    rw [hcx]
    exact hpub
  · show x.rank u % x.st.c.g < 2 ^ x.st.c.g.log2
    rw [hcx]
    exact Layout.vu_lt_pow2 (parse_untyped ht hc).1 (Layout.scopeOk_spec hU hscope).2.1 (hcx ▸ Nat.mod_lt _ x.hg)
  · show x.D.proved[x.rank u / x.st.c.g * x.st.c.g + x.rank u % x.st.c.g]? = some u.val
    rw [Nat.div_add_mod']
    exact x.proved_rank h

set_option linter.unusedSectionVars false in
include hU hpub hpt in
/-- `tableH_bcSiteS` at a public file without shared rows. -/
theorem tableH_bcSite {S : Finset (Fin n)} (x : DrawSetupH I mPts S (dj S)) {u : Fin n} (h : u ∈ S) :
    BcSiteH (x.table.acc ht hc (v2_of_scope hscope)) pub (x.table.o u h) (x.table.vu u h) u.val :=
  tableH_bcSiteS ht hc hscope dj hU hpub x h

include hU hpub hpt in
/-- **`BcSiteH` at every satisfiable site** of the one-table hidden plan: at a draw flock-verify sets up by
`tableH_bcSite`; at a refused draw no message satisfies the table (`tableAtH_unsat`). -/
theorem sitesAtH_bcSite :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planAtH dj x₀) (tabAtH dj x₀) S R u), Binding.Sat (planAtH dj x₀) (tabAtH dj x₀) S R u M →
        BcSiteH ((sitesAtH ht hc (v2_of_scope hscope) dj x₀).acc S R u h) pub
          ((sitesAtH ht hc (v2_of_scope hscope) dj x₀).o S R u h)
          ((sitesAtH ht hc (v2_of_scope hscope) dj x₀).vu S R u h) u.val := by
  intro S R u h M hM
  show BcSiteH ((tableAtH dj x₀ S).acc ht hc (v2_of_scope hscope)) pub ((tableAtH dj x₀ S).o u h)
    ((tableAtH dj x₀ S).vu u h) u.val
  by_cases hS : Nonempty (DrawSetupH I mPts S (dj S))
  · have e : tableAtH dj x₀ S = (Classical.choice hS).table := by
      unfold tableAtH
      rw [dite_eq_left_of_eq_true (eq_true hS)]
    rw [e]
    exact tableH_bcSite ht hc hscope dj hU hpub hpt _ h
  · exact absurd hM (tableAtH_unsat dj x₀ hS _)

/-- **`lay` at the hidden tables**: `keyLayoutH` at `sitesAtH`, the program's outputs the returned outputs' bits, the
registration the population's public file. -/
noncomputable abbrev layAtH :
    Binding.Layout ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtH (Reg := Reg) dj x₀)
      (tabAtH dj x₀)
      (((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)).placeDecoder (planAtH dj x₀) (tabAtH dj x₀)
        (keyPlaceH (sitesAtH_parse ht hc (v2_of_scope hscope) dj x₀) (Layout.outsOf c hU n))) :=
  keyLayoutH (sitesAtH_parse ht hc (v2_of_scope hscope) dj x₀) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOk_spec hU hscope).2.1 (Layout.scopeOk_ports hscope) (Layout.outsOut_outsOf c hU n)

/-- **`rs` at the hidden tables**: `keyRowsH` at `sitesAtH`, its circuit conditions from `scopeOk`
(`Layout.scopeOk_spec`, `Layout.scopeOk_ports`, `Layout.keysRow_of_parse`), with `hsite` from `sitesAtH_bcSite` at a
public file without shared-row tables (`hpt`). -/
noncomputable abbrev rsAtH : Binding.HmRows (layAtH ht hc hscope dj x₀ hU (pub := pub) (Reg := Reg)) :=
  keyRowsH (sitesAtH_parse ht hc (v2_of_scope hscope) dj x₀) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOk_spec hU hscope).2.1 (Layout.scopeOk_ports hscope) (Layout.outsOut_outsOf c hU n)
    (sitesAtH_bcSite ht hc hscope dj x₀ hU hpub hpt)
    (Layout.keysRow_of_parse (parse_untyped ht hc).1 (Layout.scopeOk_spec hU hscope).1
      (Layout.scopeOk_spec hU hscope).2.2)

end FlockSoundness.Discharge.HiddenExec
