import Proofs.Flock.Recursive.Raw
import Proofs.Flock.Recursive.VStarU2
import Proofs.Flock.Soundness.Discharge.FailClosed.One

/-!
# Recursion: where V*'s statement enters a session (gap 8)

A session of V*'s statement `Q` is a `flock-verify verify --zk --registered` session of the verifier's own inputs `I`,
as the prover supplies them (`ZkOuterR`), and no check the verifier of record makes reads `Q`. What ties a session to
`Q` is one definition and one open lemma, the only place the recursive audit's proof takes the equation:

* `VStmt.Loads Q I`: the inputs load `Q`: their circuit is `Q`'s (`circZ I = Q.c`) and their registered population has
  `Q`'s `nv` units (`(pubZ I).n = Q.nv`). If V*'s statements fix only their unit, by name, and what the VBridge reads,
  this definition is what changes.
* `loads_of_verifies` (open): a session the verifier of record accepts, at inputs that pass the statement check
  `--registered` runs (`ScopeZR`), loads V*'s statement. The verifier's check that a session's circuit and population
  are V*'s is to prove it.
* `typable_of_accepts`: such a session types (`ZkOuterR.Typable`): `Loads` gives its circuit and population, and the
  record's draw sets it up (`setupZ_of_verifiesZK`, at the record's custody).
* `prob_sessionsR_untypable`, `prob_recT_le`: where a session of some index never types, the raw sessions' event is
  empty; elsewhere the raw recursive audit's event is no likelier than the typed sessions' (`zsOf`) at the transported
  strategy (`recT`).
* `VStmt.pop_eq_nv`: at a session that loads `Q`, with its draw file the verifier's draw at the law's own population
  (`DrawFileZK`, custody), the population is `Q`'s units: the draw's population is the registered population's count
  (`HmRow.drawn`'s U check, `VStmt.pop_of_session`), and `Loads` fixes that count. With the strata tiling the population
  (`ofJson`'s U2 check, `VStmt.tiles_of_session`) this gives `_hS`.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg FailClosed
open scoped ENNReal

/-- **The inputs `I` load V*'s statement `Q`**: their circuit is `Q`'s and their registered population has `Q`'s
units. -/
def VStmt.Loads (Q : VStmt) (I : Inputs) : Prop :=
  circZ I = Q.c ∧ (pubZ I).n = Q.nv

/-- **Open (gap 8): a session the verifier of record accepts, at inputs that pass the statement check, loads V*'s
statement.** The one place `VStmt.Loads` enters the recursive audit's proof. No check of the verifier reads V*'s
statement yet, so this is a named gap and not a theorem of the executable; the verifier's check that a session's circuit
and population are V*'s statement's is to prove it. -/
theorem loads_of_verifies {Q : VStmt} {Reg' : Type} (x : Q.OuterR Reg') (t : Exec.Transcript)
    (_hacc : AcceptsZKR x.I x.own t) (_hck : ScopeZR x.I) : Q.Loads x.I := by
  sorry

/-- **A session the verifier of record accepts types**, where its inputs pass the statement check: they load V*'s
statement (`loads_of_verifies`), and the record carries the draw (custody), at which the session sets up
(`setupZ_of_verifiesZK`). -/
theorem typable_of_accepts {Q : VStmt} {Reg' : Type} (x : Q.OuterR Reg') (hRec : RecordsLiveZK x.I x.dj x.wr)
    (ω : Q.law.live.Ω) (R : Reg') (o : LiveOut Q.mPts)
    (hacc : AcceptsZKR x.I x.own (x.wr (Q.law.live.draw ω) R o)) (hck : ScopeZR x.I) : x.Typable := by
  have hL := loads_of_verifies x _ hacc hck
  have hacc' := acceptsZK_of_R hacc
  have hck' := hck.scopeZ x.hTags.1
  exact ⟨hck', hL.1, hL.2, _, setupZ_of_verifiesZK x.I x.hTags.1 x.hTags.2.2.2 x.dj ⟨hacc', hck'⟩
    (hRec _ R o hacc').1 (x.hDraw ω)⟩

/-- **A session that never types accepts nowhere** where its inputs pass the statement check: an acceptance would type
it (`typable_of_accepts`). -/
theorem prob_game_untypable {Q : VStmt} {Reg' : Type} (x : Q.OuterR Reg') (hRec : RecordsLiveZK x.I x.dj x.wr)
    (hx : ¬ x.Typable) (hck : ScopeZR x.I) (dg : ByteArray → ByteArray) (N : ℕ) (τ : Strategy (x.game dg N)) :
    prob (LiveAcceptsZKR x.I x.own) (x.game dg N) τ = 0 :=
  le_antisymm ((prob_auditLive_liveOf_mono dg N x.wr τ (p := LiveAcceptsZKR x.I x.own) (q := fun _ => False)
    fun ω o' h => hx (typable_of_accepts x hRec ω τ.1 o' h hck)).trans (prob_false _ _).le) (zero_le ..)

/-- **Where a session of index `j₀` never types, the raw sessions' event is empty**, for an event at which every
session accepts and passes the statement check: the event is `j₀`'s acceptance, which `prob_game_untypable` empties at
each of the prover's names for `j₀` (`prob_seqC_exists_le`). -/
theorem prob_sessionsR_untypable {S : ℕ} {Vs : Fin S → VStmt} {Reg' : Type} {dg : ByteArray → ByteArray} {N : ℕ}
    (zr : (j : Fin S) → (Vs j).OuterR Reg') (hRec : ∀ j, RecordsLiveZK (zr j).I (zr j).dj (zr j).wr) (j₀ : Fin S)
    (hx : ¬ (zr j₀).Typable) (p : VRec Vs Reg' dg N → Prop)
    (hp : ∀ b, p b → AcceptsR zr b ∧ ∀ j, ScopeZR (zr j).I) (s : Strategy (sessionsR zr dg N)) :
    prob p (sessionsR zr dg N) s = 0 := by
  by_cases hck : ScopeZR (zr j₀).I
  · refine le_antisymm (((prob_mono (q := fun b : VRec Vs Reg' dg N =>
        ∃ j, j = j₀ ∧ LiveAcceptsZKR (zr j).I (zr j).own (b j).2) (fun b hb => ⟨j₀, rfl, (hp b hb).1 j₀⟩) _ _).trans
      (prob_seqC_exists_le S _ _ (fun j r => j = j₀ ∧ LiveAcceptsZKR (zr j).I (zr j).own r.2)
        (fun _ _ => (0 : ℝ≥0∞)) ?_ s)).trans_eq ?_) (zero_le ..)
    · intro j τ
      by_cases hj : j = j₀
      · subst hj
        exact le_trans (prob_mono (fun a ha => ha.2) _ _) (prob_game_untypable _ (hRec _) hx hck dg N _).le
      · exact le_trans (prob_mono (q := fun _ => False) (fun a ha => hj ha.1) _ _) (prob_false _ _).le
    · simp only [Finset.sum_const_zero]
      exact exG_const 0 _ _
  · exact le_antisymm ((prob_mono (q := fun _ => False) (fun b hb => hck ((hp b hb).2 j₀)) _ _).trans
      (prob_false _ _).le) (zero_le ..)

/-- **The raw recursive audit's event is no likelier than the typed one's**, at the transported strategy (`recT`), when
the event has every session accept and pass the statement check (`hp`) and gives `q` where every session types (`hq`):
where every session at a registration types, the typed sessions are the raw ones (`sessions_zsOf`, `prob_sessT`); where
one doesn't, the raw event is empty there (`prob_sessionsR_untypable`). -/
theorem prob_recT_le {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type} {dg : ByteArray → ByteArray} {N : ℕ}
    {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type}
    (zr : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).OuterR Reg')
    (hRec : ∀ R ω t R₂ j, RecordsLiveZK (zr R ω t R₂ j).I (zr R ω t R₂ j).dj (zr R ω t R₂ j).wr) (h : Typed zr)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerVR (zr R ω t) dg N))
    (p q : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × VRec Vs Reg' dg N) → Prop)
    (hp : ∀ o, p o → AcceptsVR (zr o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ∀ j, ScopeZR (zr o.1 o.2.1 o.2.2.1 o.2.2.2.1 j).I)
    (hq : ∀ o, (∀ j, (zr o.1 o.2.1 o.2.2.1 o.2.2.2.1 j).Typable) → p o → q o) :
    prob p (recGame L Reg I Cm fun R ω t => outerVR (zr R ω t) dg N) σ ≤
      prob q (recGame L Reg I Cm fun R ω t => outerV (zsOf zr hRec h R ω t) dg N) (recT zr hRec h σ) := by
  refine prob_recMap_le (fun R ω t => outerT (zsOf zr hRec h R ω t) (zr R ω t)) p q σ fun ω t s => ?_
  rw [prob_outerVR, prob_outerT]
  by_cases hall : ∀ j, (zr σ.1 ω t s.1 j).Typable
  · rw [prob_sessT _ _ (sessions_zsOf zr hRec h σ.1 ω t s.1 hall)]
    exact prob_mono (fun b hb => hq (σ.1, ω, t, (s.1, b)) hall hb) _ _
  · push Not at hall
    obtain ⟨j₀, hj₀⟩ := hall
    rw [prob_sessionsR_untypable (zr σ.1 ω t s.1) (fun j => hRec σ.1 ω t s.1 j) j₀ hj₀ _
      (fun b hb => hp (σ.1, ω, t, (s.1, b)) hb) _]
    exact zero_le ..

/-- **Where every session at a history and registration types, the typed sessions accept and pass the statement check
where the raw ones do**: each is its build (`zsOf_pos`), with the raw session's inputs and own ports. -/
theorem verifies_zsOf {S : ℕ} {Vs : Fin S → VStmt} {Reg' A B C D : Type} {dg : ByteArray → ByteArray} {N : ℕ}
    (zr : A → B → C → D → (j : Fin S) → (Vs j).OuterR Reg')
    (hRec : ∀ R ω t R₂ j, RecordsLiveZK (zr R ω t R₂ j).I (zr R ω t R₂ j).dj (zr R ω t R₂ j).wr) (h : Typed zr)
    (R : A) (ω : B) (t : C) (o : D × VRec Vs Reg' dg N) (hall : ∀ j, (zr R ω t o.1 j).Typable)
    (hacc : AcceptsVR (zr R ω t) o) (hck : ∀ j, ScopeZR (zr R ω t o.1 j).I) :
    AcceptsV (zsOf zr hRec h R ω t) o ∧ ∀ j, ScopeZR (zsOf zr hRec h R ω t o.1 j).I := by
  have hz : zsOf zr hRec h R ω t o.1 = fun j => (zr R ω t o.1 j).build (hall j) (hRec R ω t o.1 j) :=
    funext fun j => zsOf_pos zr hRec h R ω t o.1 j (hall j)
  unfold AcceptsV Accepts
  rw [hz]
  exact ⟨hacc, hck⟩

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
    (hck : ScopeZ z.I) (hL : Q.Loads z.I) : Audit.Law.Tiles Q.law.St Q.nv := by
  have h := VStmt.tiles_of_session z hdj
  rwa [VStmt.pop_eq_nv z hdj hck hL] at h

end FlockSoundness.Discharge.Recursion
