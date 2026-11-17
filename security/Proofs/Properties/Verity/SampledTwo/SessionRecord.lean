module

public import Proofs.Properties.Verity.SampledTwo.SessionHeld

/-! The coin the check read for a session's round is the one the session's own record gives the round. -/

@[expose] public section

namespace Verity.SampledTwo.SessionCoins

def DevReads (A : Live₂) (loc : List Release → Verity.Frame → Option ℕ)
    (dev : List Release → ℕ → Option FlockDeploy.Coin) (elem : (ℕ → Bool) → Flock.F128) : Prop :=
  ∀ n, groupOf A n = .sessions → ∀ log, dev log n =
    if ((located loc n [] log).map fun y => widthOf y.1).sum = 0 then none
    else some fun e => elem fun b => ((located loc n [] log).flatMap (·.2)).getD (128 * e.val + b) false

theorem groupOf_off (A : Live₂) (t j : ℕ) : groupOf A (SpotFlock.off A.h t + j) = .sessions := by
  have hle := SpotFlock.stage_le_off A.h t
  have e0 : ¬ SpotFlock.off A.h t + j = 0 := by omega
  have e1 : ¬ SpotFlock.off A.h t + j ≤ A.h.stage := by omega
  have e2 : ¬ SpotFlock.off A.h t + j = A.h.stage + 1 := by omega
  simp only [groupOf, if_neg e0, if_neg e1, if_neg e2]

theorem coin_some {A : Live₂} {C : CaseOf} {Ms : SessionReading} {tab : ℕ → String}
    {dev : List Release → ℕ → Option FlockDeploy.Coin} {o : Exchange} {t j : ℕ} {c : FlockDeploy.Coin}
    (h : (withCoins A C tab dev Ms).coin o t j = some c) :
    ∃ l, interiorRead C o = some l ∧ dev l (SpotFlock.off A.h t + j) = some c := by
  have h' : ((interiorRead C o).bind fun l =>
      if (located (locOf A tab) (SpotFlock.off A.h t + j) [] l).all (fun x => !x.2.isEmpty) then
        dev l (SpotFlock.off A.h t + j)
      else none) = some c := h
  obtain ⟨l, hl, hc⟩ := Option.bind_eq_some_iff.1 h'
  refine ⟨l, hl, ?_⟩
  by_cases hall : (located (locOf A tab) (SpotFlock.off A.h t + j) [] l).all (fun x => !x.2.isEmpty) = true
  · rw [if_pos hall] at hc
    exact hc
  · rw [if_neg hall] at hc
    cases hc

theorem located_record (A : Live₂) (C : CaseOf) (tab : ℕ → String)
    (dev : List Release → ℕ → Option FlockDeploy.Coin) (elem : (ℕ → Bool) → Flock.F128)
    (hdev : DevReads A (locOf A tab) dev elem) (hsz : HeldSize) {o : Exchange} (hacc : FormAccepts C o)
    {l : List Release} (hl : interiorRead C o = some l) {t j : ℕ} (ht : t < A.h.ms.length)
    (hj : j < SpotFlock.k₀ A.h t) {c : FlockDeploy.Coin} (hc : dev l (SpotFlock.off A.h t + j) = some c) :
    ∃ recs rec i s n, sessionRecords C o = some recs ∧ recs[t]? = some rec ∧
      located (locOf A tab) (SpotFlock.off A.h t + j) [] l =
        [(.opening i s n, ((Verity.Sampled.recordRound rec (slotStream (tab t) (repsOf (SpotFlock.k₀ A.h t)) j).1
          (slotStream (tab t) (repsOf (SpotFlock.k₀ A.h t)) j).2).map
          Verity.bitsOfBytes).getD [])] := by
  rw [hdev _ (groupOf_off A t j) l] at hc
  have h1 := located_one_accepted A C tab hsz o hacc l hl (SpotFlock.off A.h t + j)
  cases hx : located (locOf A tab) (SpotFlock.off A.h t + j) [] l with
  | nil =>
    rw [hx] at hc
    simp only [List.map_nil, List.sum_nil, if_true] at hc
    cases hc
  | cons y rest =>
    cases rest with
    | cons z rest =>
      rw [hx] at h1
      simp only [List.length_cons] at h1
      omega
    | nil =>
      obtain ⟨recs, rec, hrecs, hrec, ⟨i, s, n, hfr⟩, hy2⟩ :=
        located_own_accepted A C tab hsz o hacc l hl t j ht hj y (by rw [hx]; exact List.mem_singleton.2 rfl)
      obtain ⟨fr, bits⟩ := y
      have hfr' : fr = .opening i s n := hfr
      have hbits : bits = ((Verity.Sampled.recordRound rec (slotStream (tab t) (repsOf (SpotFlock.k₀ A.h t)) j).1
        (slotStream (tab t) (repsOf (SpotFlock.k₀ A.h t)) j).2).map
          Verity.bitsOfBytes).getD [] := hy2
      exact ⟨recs, rec, i, s, n, hrecs, hrec, by rw [hfr', hbits]⟩

theorem coin_record (A : Live₂) (C : CaseOf) (Ms : SessionReading) (tab : ℕ → String)
    (dev : List Release → ℕ → Option FlockDeploy.Coin) (elem : (ℕ → Bool) → Flock.F128)
    (hdev : DevReads A (locOf A tab) dev elem) (hsz : HeldSize) {o : Exchange} (hacc : FormAccepts C o) {t j : ℕ}
    (ht : t < A.h.ms.length) (hj : j < SpotFlock.k₀ A.h t) {c : FlockDeploy.Coin}
    (hc : (withCoins A C tab dev Ms).coin o t j = some c) :
    ∃ recs rec, sessionRecords C o = some recs ∧ recs[t]? = some rec ∧
      c = fun e => elem fun b =>
        (((Verity.Sampled.recordRound rec (slotStream (tab t) (repsOf (SpotFlock.k₀ A.h t)) j).1
          (slotStream (tab t) (repsOf (SpotFlock.k₀ A.h t)) j).2).map
          Verity.bitsOfBytes).getD []).getD (128 * e.val + b) false := by
  obtain ⟨l, hl, hd⟩ := coin_some hc
  obtain ⟨recs, rec, i, s, n, hrecs, hrec, hx⟩ := located_record A C tab dev elem hdev hsz hacc hl ht hj hd
  refine ⟨recs, rec, hrecs, hrec, ?_⟩
  rw [hdev _ (groupOf_off A t j) l, hx] at hd
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, List.flatMap_cons, List.flatMap_nil,
    List.append_nil] at hd
  split_ifs at hd
  exact (Option.some.inj hd).symm

theorem agree_record (A : Live₂) (C : CaseOf) (Ms : SessionReading) (tab : ℕ → String)
    (dev : List Release → ℕ → Option FlockDeploy.Coin) (elem : (ℕ → Bool) → Flock.F128)
    (hdev : DevReads A (locOf A tab) dev elem) (hsz : HeldSize) {o : Exchange} (hacc : FormAccepts C o) {t j : ℕ}
    (ht : t < A.h.ms.length) (hj : j < SpotFlock.k₀ A.h t) {c c' : FlockDeploy.Coin}
    (hc : (withCoins A C tab dev Ms).coin o t j = some c) (hag : (withCoins A C tab dev Ms).agree o t j c c') :
    ∃ recs rec n, sessionRecords C o = some recs ∧ recs[t]? = some rec ∧ 0 < n ∧
      ∀ e : Fin 65536, e.val < n → c' e = elem fun b =>
        (((Verity.Sampled.recordRound rec (slotStream (tab t) (repsOf (SpotFlock.k₀ A.h t)) j).1
          (slotStream (tab t) (repsOf (SpotFlock.k₀ A.h t)) j).2).map
          Verity.bitsOfBytes).getD []).getD (128 * e.val + b) false := by
  obtain ⟨l, hl, hd⟩ := coin_some hc
  obtain ⟨recs, rec, i, s, n, hrecs, hrec, hx⟩ := located_record A C tab dev elem hdev hsz hacc hl ht hj hd
  obtain ⟨recs', rec', hrecs', hrec', hceq⟩ := coin_record A C Ms tab dev elem hdev hsz hacc ht hj hc
  obtain rfl : recs = recs' := Option.some.inj (hrecs.symm.trans hrecs')
  obtain rfl : rec = rec' := Option.some.inj (hrec.symm.trans hrec')
  have hag' : ∃ l, interiorRead C o = some l ∧ ∀ e : Fin 65536,
      e.val < ((located (locOf A tab) (SpotFlock.off A.h t + j) [] l).map fun x => widthOf x.1).sum →
        c e = c' e := hag
  obtain ⟨l', hl', hE⟩ := hag'
  obtain rfl : l = l' := Option.some.inj (hl.symm.trans hl')
  rw [hx] at hE
  have hw : (([(Verity.Frame.opening i s n,
      ((Verity.Sampled.recordRound rec (slotStream (tab t) (repsOf (SpotFlock.k₀ A.h t)) j).1
        (slotStream (tab t) (repsOf (SpotFlock.k₀ A.h t)) j).2).map
        Verity.bitsOfBytes).getD [])].map fun y => widthOf y.1).sum) = n := by
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, widthOf, Nat.add_zero]
  rw [hw] at hE
  have hn : 0 < n := by
    rw [hdev _ (groupOf_off A t j) l, hx, hw] at hd
    by_contra h0
    rw [if_pos (by omega)] at hd
    cases hd
  refine ⟨recs, rec, n, hrecs, hrec, hn, fun e he => ?_⟩
  rw [← hE e he, hceq]

end Verity.SampledTwo.SessionCoins
