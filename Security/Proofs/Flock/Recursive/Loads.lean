import Proofs.Flock.Recursive.VStarU2
import Proofs.Flock.Soundness.Discharge.FailClosed.Scope

/-!
# Recursion: where V*'s statement enters a session (gap 8)

A session of V*'s statement `Q` is a `flock-verify verify --zk --registered` session of the verifier's own inputs `I`,
and no check the verifier of record makes reads `Q`. What ties a session to `Q` is one definition and one open lemma,
the only place the recursive audit's proof takes the equation:

* `VStmt.Loads Q I`: the inputs load `Q`: their circuit is `Q`'s (`circZ I = Q.c`) and their registered population has
  `Q`'s `nv` units (`(pubZ I).n = Q.nv`). If V*'s statements fix only their unit, by name, and what the VBridge reads,
  this definition is what changes.
* `loads_of_verifies` (open): at an outcome the verifier of record accepts, every session loads V*'s statement. The
  verifier's check that a session's circuit and population are V*'s is to prove it.
* `VStmt.pop_eq_nv`: at a session that loads `Q`, with its draw file the verifier's draw at the law's own population
  (`DrawFileZK`, custody), the population is `Q`'s units: the draw's population is the registered population's count
  (`HmRow.drawn`'s U check, `VStmt.pop_of_session`), and `Loads` fixes that count. With the strata tiling the population
  (`ofJson`'s U2 check, `VStmt.tiles_of_session`) this gives `_hS`.
-/

namespace FlockSoundness.Discharge.Recursion

open Exec FailClosed

/-- **The inputs `I` load V*'s statement `Q`**: their circuit is `Q`'s and their registered population has `Q`'s
units. -/
def VStmt.Loads (Q : VStmt) (I : Inputs) : Prop :=
  circZ I = Q.c ∧ (pubZ I).n = Q.nv

/-- **Open (gap 8): at an outcome the verifier of record accepts, every session loads V*'s statement.** The one place
`VStmt.Loads` enters the recursive audit's proof. No check of the verifier reads V*'s statement yet, so this is a named
gap and not a theorem of the executable; the verifier's check that a session's circuit and population are V*'s
statement's is to prove it. -/
theorem loads_of_verifies {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
    (zs : Reg₂ → (j : Fin S) → (Vs j).Outer Reg') {dg : ByteArray → ByteArray} {N : ℕ}
    (o : Reg₂ × VRec Vs Reg' dg N) (_hacc : AcceptsV zs o) (_hck : ∀ j, ScopeZ (zs o.1 j).I) (j : Fin S) :
    (Vs j).Loads (zs o.1 j).I := by
  sorry

/-- **At a session that loads `Q`, the law's population is `Q`'s units**: the draw's population is the registered
population's count (`VStmt.pop_of_session`), the registered population is the inputs' own (`ScopeZ.load`, at `Q`'s
circuit), and `Loads` gives its count. -/
theorem VStmt.pop_eq_nv {Q : VStmt} {Reg' : Type} (z : Q.Outer Reg')
    (hdj : ZkLink.DrawFileZK z.dj (.stratified Q.law.Kd (Array.ofFn Q.law.St)) Q.law.pop)
    (hck : ScopeZ z.I) (hL : Q.Loads z.I) : Q.law.pop = Q.nv := by
  have hp : z.pub = pubZ z.I := by
    have h := hck.load z.hTags.1
    rw [hL.1] at h
    exact Except.ok.inj (z.hpub.symm.trans h)
  rw [VStmt.pop_of_session z hdj, hp]
  exact hL.2

/-- **The strata tile V*'s units at a session that loads `Q`**: `VStmt.tiles_of_session` at the law's own population,
which is `Q`'s units (`VStmt.pop_eq_nv`). -/
theorem VStmt.tiles_of_loads {Q : VStmt} {Reg' : Type} (z : Q.Outer Reg')
    (hdj : ZkLink.DrawFileZK z.dj (.stratified Q.law.Kd (Array.ofFn Q.law.St)) Q.law.pop)
    (hck : ScopeZ z.I) (hL : Q.Loads z.I) : Audit.Law.Tiles Q.law.St Q.nv :=
  VStmt.pop_eq_nv z hdj hck hL ▸ VStmt.tiles_of_session z hdj

end FlockSoundness.Discharge.Recursion
