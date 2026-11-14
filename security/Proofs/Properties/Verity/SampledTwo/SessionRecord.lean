module

public import Proofs.Properties.Verity.SampledTwo.SessionHeld

/-!
# The coin the check read for a session's round is its record's

`SessionCoins.withCoins` reads session `t`'s coin at its round `j` as the device's release for the run's round
`off t + j` in the log the check read. Where that release is read off the releases the session-round locator puts on
the round (`DevReads`, which the coin map's release satisfies), the coin is the one session `t`'s own record gives the
round's stream's round:

* `groupOf_off`: a session's round is of the sessions' group.
* `coin_record`: on a play the two-stage form accepts, the coin the check read for round `j` of session `t` has, as
  its elements' bits in order, the bits of the bytes session `t`'s record gives the round's stream's round
  (`SessionCoins.slotStream`, `Verity.Sampled.recordRound`), padded with zeros. One release is on the round
  (`located_one_accepted`) and it is session `t`'s (`located_own_accepted`).
* `agree_record`: a coin that agrees with it as the sessions' reading says (`withCoins`' `agree`) has those elements
  below the count of coins the round's opening carries.
-/

@[expose] public section

namespace Verity.SampledTwo.SessionCoins

/-- **`dev` at a session's round reads the releases the locator `loc` puts there**: none where they carry no element,
else the coin whose element `e` is `elem` of their bits from bit `128 e` on, in log order, padded with zeros. -/
def DevReads (A : Live₂) (loc : List Release → Verity.Frame → Option ℕ)
    (dev : List Release → ℕ → Option FlockDeploy.Coin) (elem : (ℕ → Bool) → Flock.F128) : Prop :=
  ∀ n, groupOf A n = .sessions → ∀ log, dev log n =
    if ((located loc n [] log).map fun y => widthOf y.1).sum = 0 then none
    else some fun e => elem fun b => ((located loc n [] log).flatMap (·.2)).getD (128 * e.val + b) false

/-- A session's round is of the sessions' group. -/
theorem groupOf_off (A : Live₂) (t j : ℕ) : groupOf A (SpotFlock.off A.h t + j) = .sessions := by
  have hle := SpotFlock.stage_le_off A.h t
  have e0 : ¬ SpotFlock.off A.h t + j = 0 := by omega
  have e1 : ¬ SpotFlock.off A.h t + j ≤ A.h.stage := by omega
  have e2 : ¬ SpotFlock.off A.h t + j = A.h.stage + 1 := by omega
  simp only [groupOf, if_neg e0, if_neg e1, if_neg e2]

/-- **What the check read for a session's round, unfolded**: the read log, every release on the round showing its
coins, and `dev`'s release there. -/
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

/-- **The one release on a session's round, and its record**: on a play the form accepts, where `dev` reads a coin
for round `j` of session `t` in the log the check read, the session-round locator puts exactly one release there, an
opening whose coins are the ones session `t`'s record gives the round's stream's round. -/
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

/-- **The coin the check read for a session's round is its record's**: on a play the form accepts, the coin the
sessions' reading reads for round `j` of session `t` has, as element `e`, `elem` of the bits from bit `128 e` on of
the bytes session `t`'s record gives the round's stream's round, padded with zeros. -/
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

/-- **A coin agreeing with the one the check read for a session's round has its record's elements**: on a play the
form accepts, where the sessions' reading reads a coin for round `j` of session `t`, the round's one release is an
opening of some `n > 0` coins, and a coin `c'` that agrees with the one read (`withCoins`' `agree`) has, below `n`,
the elements the record's round's bytes give. -/
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
