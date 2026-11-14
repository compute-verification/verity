module

public import Proofs.Properties.Verity.SampledTwo.SessionHeld
public import Proofs.Properties.Verity.SampledTwo.DevAgrees

/-!
# The sessions' releases agree with the run's coins

`SessionsAgree` (`DevAgrees.lean`) at the sessions' reading `SessionCoins.withCoins`: on an accepted play whose log is
the device's, at coins the coin map's agreement holds of, the device's release for a read round past the statements' is
a session's round's, and it agrees with the run's coin there as the sessions' reading says: on the elements the
releases the session-round locator puts on the round carry, in the log the check read.

* `InteriorReads` (`SessionCoins.lean`): the form's acceptance reads the interior's log; `SessionHeld.lean` proves it.
* `DevElems`: the coin map's agreement at the session-round locator, in the play's log.
* `located_widths`: the elements a round's releases carry are the same in two logs of the same frames.
* `slot_exists`: a round past the statements' and before the sessions' end is in one session's rounds.
* `sessionsAgree`: `SessionsAgree` from `InteriorReads`, `InteriorLogged` and `DevElems`.
* `sessionsAgree_accepted`: the same at the form's acceptance, `InteriorReads` from `SessionHeld.interiorReads`.
-/

@[expose] public section

namespace Verity.SampledTwo.SessionCoins

/-- **The device's release at a session's round shares the run's coin's elements its releases carry** (the coin map's
agreement, at the session-round locator): on an accepted play whose log is the device's, at coins the coin map's
agreement `Ag` holds of, the release for a session's round the run's verifier reads and the run's coin there have the
same elements below the count the releases the locator puts on the round carry, in the play's log. -/
def DevElems (A : Live₂) (x : Claims.Msg) (loc : List Release → Verity.Frame → Option ℕ)
    (dev : List Release → ℕ → Option FlockDeploy.Coin) (Acc Good : Exchange → Prop)
    (Ag : Exchange → Coins A x → Prop) : Prop :=
  ∀ o, Acc o → Good o → ∀ r : Coins A x, Ag o r → ∀ i : Fin (A.L.V.k x), readsAt A.h i.val →
    groupOf A i.val = .sessions → ∀ c, dev o.log i.val = some c →
      ∀ e : Fin 65536, e.val < ((located loc i.val [] o.log).map fun y => widthOf y.1).sum → c e = r i e

/-- **The elements a round's releases carry are the same in two logs of the same frames.** -/
theorem located_widths {off K : ℕ → ℕ} {M : ℕ} {tab : ℕ → String} {n : ℕ} {l l' : List Release}
    (hl : List.Forall₂ (fun r r' => r.frame = r'.frame) l l') :
    ∀ {pre pre' : List Release}, Same pre pre' →
      ((located (sessLoc off K M tab) n pre' l').map fun y => widthOf y.1).sum =
        ((located (sessLoc off K M tab) n pre l).map fun y => widthOf y.1).sum := by
  induction hl with
  | nil => intro _ _ _; rfl
  | @cons r r' l l' hf _ ih =>
    intro pre pre' hS
    simp only [located, List.map_append, List.sum_append]
    rw [ih (hS.snoc hf), ← sessLoc_congr hS r'.frame, ← hf]
    by_cases hloc : sessLoc off K M tab pre r.frame = some n
    · rw [if_pos hloc, if_pos hloc]
      rfl
    · rw [if_neg hloc, if_neg hloc]

/-- A round from the sessions' first on and before their last's end is in one session's rounds. -/
theorem slot_exists {off K : ℕ → ℕ} (hs : ∀ t, off (t + 1) = off t + K t) {n : ℕ} :
    ∀ M, off 0 ≤ n → n < off M → ∃ t, t < M ∧ off t ≤ n ∧ n < off t + K t
  | 0, h0, hM => absurd hM (by omega)
  | M + 1, h0, hM => by
    by_cases h : n < off M
    · obtain ⟨t, ht, h1, h2⟩ := slot_exists hs M h0 h
      exact ⟨t, by omega, h1, h2⟩
    · exact ⟨M, by omega, by omega, by rw [← hs]; exact hM⟩

/-- **The sessions' releases agree with the run's coins** (piece: the sessions' coins, as `devAgrees_formReading`
takes it): `SessionsAgree` at the sessions' reading `withCoins`, from the coin map's agreement at the session-round
locator (`DevElems`), `InteriorLogged` and the read's existence on an accepted play (`InteriorReads`). A read
round past the statements' is in a session's rounds (`slot_exists`), and the releases the locator puts on it carry the
same elements in the log the check read as in the play's (`located_widths`). -/
theorem sessionsAgree (A : Live₂) (x : Claims.Msg) (C : CaseOf) (Ms : SessionReading) (tab : ℕ → String)
    (dev : List Release → ℕ → Option FlockDeploy.Coin) (Good : Exchange → Prop) (Ag : Exchange → Coins A x → Prop)
    (hread : InteriorReads C) (hlog : InteriorLogged C Good)
    (hE : DevElems A x (locOf A tab) dev (FormAccepts C) Good Ag) :
    SessionsAgree A x (withCoins A C tab dev Ms) dev (FormAccepts C) Good Ag := by
  intro o hA hG r hAg i hr hlt c hc
  have h0 : SpotFlock.off A.h 0 = A.h.stage + 2 := rfl
  have hM : i.val < SpotFlock.off A.h A.h.ms.length := hr.2
  obtain ⟨t, htM, h1, h2⟩ := slot_exists (off := SpotFlock.off A.h) (K := SpotFlock.k₀ A.h) (SpotFlock.off_succ A.h)
    A.h.ms.length (by omega) hM
  have hslot : SpotFlock.slotOf A.h (i.val + 1) = some t := SpotFlock.slotOf_eq A.h htM (by omega) (by omega)
  have hn : SpotFlock.off A.h t + (i.val - SpotFlock.off A.h t) = i.val := by omega
  have hg : groupOf A i.val = .sessions := by
    have e0 : ¬ i.val = 0 := by omega
    have e1 : ¬ i.val ≤ A.h.stage := by omega
    have e2 : ¬ i.val = A.h.stage + 1 := by omega
    simp only [groupOf, if_neg e0, if_neg e1, if_neg e2]
  obtain ⟨l, hl⟩ := hread o hA
  obtain ⟨q, l', hlog', hq, hR⟩ := hlog o hA hG l hl
  have hF : List.Forall₂ (fun r r' => r.frame = r'.frame) l l' :=
    (List.forall₂_map_left_iff.1 hR).imp fun _ _ h => (of_ofV2 h).1
  have hS : Same [] q := by simpa using same_append (pre := []) hq
  refine ⟨t, hslot, ?_⟩
  show ∃ l, interiorRead C o = some l ∧ ∀ e : Fin 65536,
    e.val < ((located (locOf A tab) (SpotFlock.off A.h t + (i.val - SpotFlock.off A.h t)) [] l).map
      fun y => widthOf y.1).sum → c e = r i e
  rw [hn]
  refine ⟨l, hl, fun e he => hE o hA hG r hAg i hr hg c hc e ?_⟩
  unfold locOf at he ⊢
  rw [hlog', located_append, located_none q (fun y hy => (hq y hy).1) []]
  simp only [List.nil_append]
  rw [located_widths hF hS]
  exact he

/-- **The sessions' releases agree with the run's coins** (`SessionsAgree`), at the form's acceptance: from the coin
map's agreement at the session-round locator (`DevElems`) and `InteriorLogged`. -/
theorem sessionsAgree_accepted (A : Live₂) (x : Claims.Msg) (C : CaseOf) (Ms : SessionReading) (tab : ℕ → String)
    (dev : List Release → ℕ → Option FlockDeploy.Coin) (Good : Exchange → Prop) (Ag : Exchange → Coins A x → Prop)
    (hsz : HeldSize) (hlog : InteriorLogged C Good) (hE : DevElems A x (locOf A tab) dev (FormAccepts C) Good Ag) :
    SessionsAgree A x (withCoins A C tab dev Ms) dev (FormAccepts C) Good Ag :=
  sessionsAgree A x C Ms tab dev Good Ag (interiorReads C hsz) hlog hE

end Verity.SampledTwo.SessionCoins
