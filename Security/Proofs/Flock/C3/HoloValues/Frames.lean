import Proofs.Flock.VBridge.Family.Files
import Proofs.Flock.C3.HoloValues.ZcLc

/-!
# A holo rep's stream: its frames and coin counts

G's `frameAt` and `CoinsShaped` at a holo record (`rec_live.holo_rounds`: `rep_rounds` with `z_partial`'s round
replaced by two). `z_partial`'s round, at `3 + (m − 6) + 2 + (k_log − 6)`, issues no coins and retains `z_partial` and
a squeeze of none, the next round issues `r′` and retains `h`, and the rounds after it are G's moved up by one
(`holoFrameAt`, `HoloCoinsShaped`). `holoCounts_of_shaped` gives `UnitFrames`' counts and frames up to `r′`
(`HoloCounts`) on G's `streamRounds` at those frames, as `runs_filesAt` gives `run_of_frames`' at plain.

V* reads the record through `Coins.ofRounds`, which skips the empty round: `filter_getElem!_after` and
`filter_size_after` give its rounds past it as the record's one later.
-/

namespace FlockC3.HoloValues

open Flock FlockSoundness.Refine
open FlockSoundness.Discharge.Exec (chunks ligFrames)
open FlockVBridge.Family (frameAt streamRounds streamRounds_wf streamRounds_retains streamRounds_n frameAt_zero
  frameAt_one frameAt_two frameAt_zc frameAt_alpha frameAt_beta frameAt_lc)

/-! ## The frames -/

/-- **The bytes round `j` of a holo rep's stream retains**: G's `frameAt` below `z_partial`'s round, which retains
`z_partial` and a squeeze of none (`closeEmpty`), then `h` and `r′`'s squeeze, then G's rounds one later. -/
def holoFrameAt (st : Setup) (sch : Flock.Schedule) (dom : ByteArray) (n : ℕ) (p : Proof) (T : F128)
    (h : Array F128) (j : ℕ) : List UInt8 :=
  if j < 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) then frameAt st sch dom n p T j
  else if j = 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) then encs [.vec p.lincheck.zPartial.toList, .sq 2 0]
  else if j = 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 then encs [.vec h.toList, .sq 1 1]
  else frameAt st sch dom n p T (j - 1)

section frames

variable (st : Setup) (sch : Flock.Schedule) (dom : ByteArray) (n : ℕ) (p : Proof) (T : F128) (h : Array F128)

theorem holoFrameAt_lt {j : ℕ} (hj : j < 3 + (st.m - 6) + 2 + (st.fold.kLog - 6)) :
    holoFrameAt st sch dom n p T h j = frameAt st sch dom n p T j := by
  unfold holoFrameAt; rw [ite_eq_left hj]

theorem holoFrameAt_zp {j : ℕ} (hj : j = 3 + (st.m - 6) + 2 + (st.fold.kLog - 6)) :
    holoFrameAt st sch dom n p T h j = encs [.vec p.lincheck.zPartial.toList, .sq 2 0] := by
  unfold holoFrameAt; rw [ite_eq_right (by omega), ite_eq_left hj]

theorem holoFrameAt_h {j : ℕ} (hj : j = 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1) :
    holoFrameAt st sch dom n p T h j = encs [.vec h.toList, .sq 1 1] := by
  unfold holoFrameAt; rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left hj]

theorem holoFrameAt_gt {j : ℕ} (hj : 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 < j) :
    holoFrameAt st sch dom n p T h j = frameAt st sch dom n p T (j - 1) := by
  unfold holoFrameAt; rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega)]

end frames

/-! ## The coin counts -/

/-- **A holo rep's coin counts** (`holo_rounds`): G's `CoinsShaped` with `z_partial`'s round of no coins and `r′`'s
of one before ring switching, and every other round issuing some. -/
def HoloCoinsShaped (m kLog n L : ℕ) (coins : Array (Array F128)) : Prop :=
  3 + (m - 6) + 2 + (kLog - 6) + 2 + (2 + n) + 1 + L ≤ coins.size ∧
    (coins[0]!).size = 6 ∧ (coins[1]!).size = m - 13 ∧ (coins[2]!).size = 1 ∧
    (∀ i < m - 6, (coins[3 + i]!).size = 1) ∧
    (∀ j < 2 + (kLog - 6), (coins[3 + (m - 6) + j]!).size = 1) ∧
    (coins[3 + (m - 6) + 2 + (kLog - 6)]!).size = 0 ∧
    (coins[3 + (m - 6) + 2 + (kLog - 6) + 1]!).size = 1 ∧
    (∀ k < 2 + n, (coins[3 + (m - 6) + 2 + (kLog - 6) + 2 + k]!).size = 7) ∧
    (coins[3 + (m - 6) + 2 + (kLog - 6) + 2 + (2 + n)]!).size = 2 + n ∧
    ∀ j < coins.size, j ≠ 3 + (m - 6) + 2 + (kLog - 6) → (coins[j]!).size ≠ 0

/-- **A holo record's counts and frames up to `r′`**: `UnitFrames`' fields from round 0 through the lincheck with
hidden units `h`. -/
structure HoloCounts (st : Setup) (p : Proof) (h : Array F128) (t : Transcript) : Prop where
  k0 : t.k = 0
  wf : RoundsWF t.rounds
  n₀ : (t.rounds[t.k]!).n = 6
  n₁ : (t.rounds[t.k + 1]!).n = st.m - 13
  n₂ : (t.rounds[t.k + 2]!).n = 1
  nr : ∀ i < st.m - 6, (t.rounds[t.k + 3 + i]!).n = 1
  a₀ : Flock.RecAlgebra.AnswersL (t.rounds[t.k]!) (t.pending.data.toList ++ encs [.label "flock-r1cs-v0".toUTF8,
    .bytes st.digest, .bytes (p.cap.foldl (· ++ ·) ByteArray.empty), .label "flock-zerocheck-v0".toUTF8, .sq 2 6])
  a₁ : Flock.RecAlgebra.AnswersL (t.rounds[t.k + 1]!) (encs [.sq 2 (st.m - 13)])
  a₂ : Flock.RecAlgebra.AnswersL (t.rounds[t.k + 2]!)
    (encs [.vec p.zerocheck.round1ab.toList, .vec p.zerocheck.round1c.toList, .sq 1 1])
  ar : ∀ i < st.m - 6, Flock.RecAlgebra.AnswersL (t.rounds[t.k + 3 + i]!)
    (encs [.one (p.zerocheck.rounds[i]!).1, .one (p.zerocheck.rounds[i]!).2, .sq 1 1])
  lc : HoloLcAnswered ⟨t.rounds, t.k + 3 + (st.m - 6),
    ⟨(encs [.one p.zerocheck.finalA, .one p.zerocheck.finalB]).toArray⟩⟩ st.fold.kLog p.lincheck h

/-- **A holo record's counts and frames** (`HoloCounts`), when its rounds issue coins of `HoloCoinsShaped`'s counts and
retain `holoFrameAt`'s bytes from round 0, after the stream's domain. -/
theorem holoCounts_of_shaped {st : Setup} {sch : Flock.Schedule} {dom : ByteArray} {n L : ℕ} {p : Proof} {T : F128}
    {h : Array F128} {coins : Array (Array F128)} {t : Transcript}
    (hco : HoloCoinsShaped st.m st.fold.kLog n L coins)
    (htr : t.rounds = streamRounds coins (holoFrameAt st sch dom n p T h)) (htk : t.k = 0)
    (htp : t.pending = Transcript.header 1 0 dom.size ++ Transcript.padded dom) : HoloCounts st p h t := by
  obtain ⟨hcs, hc0, hc1, hc2, hcz, hcl, hzp, hh, -, -, -⟩ := hco
  have hR : ∀ j, j < coins.size → Retains t.rounds[j]! (holoFrameAt st sch dom n p T h j) :=
    fun j hj => by rw [htr]; exact streamRounds_retains hj rfl
  have hRl : ∀ j, j < 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) →
      Retains t.rounds[j]! (frameAt st sch dom n p T j) := fun j hj => by
    have e := hR j (by omega)
    rwa [holoFrameAt_lt _ _ _ _ _ _ _ hj] at e
  have hN : ∀ j c, j < coins.size → (coins[j]!).size = c → (t.rounds[j]!).n = c :=
    fun j c hj hc => by rw [htr]; exact streamRounds_n hj hc
  have hsz : t.rounds.size = coins.size := by rw [htr]; simp [streamRounds]
  have hOne : ∀ j B, 3 + (st.m - 6) ≤ j → j < 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) →
      frameAt st sch dom n p T j = B → OneCoin t.rounds[j]! B := fun j B h₁ h₂ hB => by
    have e := hcl (j - (3 + (st.m - 6))) (by omega)
    rw [show 3 + (st.m - 6) + (j - (3 + (st.m - 6))) = j by omega] at e
    exact ⟨hN j 1 (by omega) e, answers_of_retains (hB ▸ hRl j h₂)⟩
  have hzpR : Retains t.rounds[3 + (st.m - 6) + 2 + (st.fold.kLog - 6)]!
      (encs [.vec p.lincheck.zPartial.toList, .sq 2 0]) := by
    have e := hR (3 + (st.m - 6) + 2 + (st.fold.kLog - 6)) (by omega)
    rwa [holoFrameAt_zp _ _ _ _ _ _ _ rfl] at e
  have hhR : Retains t.rounds[3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1]! (encs [.vec h.toList, .sq 1 1]) := by
    have e := hR (3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1) (by omega)
    rwa [holoFrameAt_h _ _ _ _ _ _ _ rfl] at e
  have hlc : HoloLcAnswered ⟨t.rounds, 3 + (st.m - 6),
      ⟨(encs [.one p.zerocheck.finalA, .one p.zerocheck.finalB]).toArray⟩⟩ st.fold.kLog p.lincheck h :=
    ⟨hOne (3 + (st.m - 6)) _ le_rfl (by omega) (frameAt_alpha _ _ _ _ _ _ rfl),
      hOne (3 + (st.m - 6) + 1) _ (by omega) (by omega) (frameAt_beta _ _ _ _ _ _ rfl),
      fun j hj => hOne (3 + (st.m - 6) + 2 + j) _ (by omega) (by omega) (frameAt_lc _ _ _ _ _ _ rfl hj),
      by show 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) < t.rounds.size; omega,
      hN (3 + (st.m - 6) + 2 + (st.fold.kLog - 6)) 0 (by omega) hzp,
      answers_of_retains hzpR,
      ⟨hN (3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1) 1 (by omega) hh, answers_of_retains hhR⟩⟩
  refine ⟨htk, by rw [htr]; exact streamRounds_wf _ _, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [htk]; exact hN 0 6 (by omega) hc0
  · rw [htk, Nat.zero_add]; exact hN 1 _ (by omega) hc1
  · rw [htk, Nat.zero_add]; exact hN 2 _ (by omega) hc2
  · intro i hi
    rw [htk, Nat.zero_add]
    exact hN _ _ (by omega) (hcz i hi)
  · rw [htk, htp]
    have e := hRl 0 (by omega)
    rw [frameAt_zero] at e
    exact Flock.RecAlgebra.answersL_of_retains e
  · rw [htk, Nat.zero_add]
    have e := hRl 1 (by omega)
    rw [frameAt_one] at e
    exact Flock.RecAlgebra.answersL_of_retains e
  · rw [htk, Nat.zero_add]
    have e := hRl 2 (by omega)
    rw [frameAt_two] at e
    exact Flock.RecAlgebra.answersL_of_retains e
  · intro i hi
    rw [htk, Nat.zero_add]
    have e := hRl (3 + i) (by omega)
    rw [frameAt_zc _ _ _ _ _ _ rfl hi] at e
    exact Flock.RecAlgebra.answersL_of_retains e
  · rw [htk, Nat.zero_add]
    exact hlc

/-! ## V*'s view past the empty round -/

theorem list_filter_getElem!_after {α : Type} [Inhabited α] (p : α → Bool) :
    ∀ (l : List α) (z : ℕ), z < l.length → p l[z]! = false → (∀ j < l.length, j ≠ z → p l[j]! = true) →
      ∀ j, z ≤ j → (l.filter p)[j]! = l[j + 1]!
  | [], _, hz, _, _, _, _ => absurd hz (Nat.not_lt_zero _)
  | a :: l, 0, _, hp, h, j, _ => by
      rw [List.getElem!_cons_zero] at hp
      have hl : l.filter p = l := List.filter_eq_self.mpr fun x hx => by
        obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hx
        have e := h (i + 1) (by simp only [List.length_cons]; omega) (by omega)
        rwa [List.getElem!_cons_succ, getElem!_pos l i hi] at e
      rw [List.filter_cons_of_neg (by rw [hp]; decide), hl, List.getElem!_cons_succ]
  | a :: l, z + 1, hz, hp, h, j, hj => by
      have ha : p a = true := by
        have e := h 0 (by simp) (by omega)
        rwa [List.getElem!_cons_zero] at e
      rw [List.getElem!_cons_succ] at hp
      rw [List.filter_cons_of_pos ha]
      obtain ⟨j, rfl⟩ : ∃ j', j = j' + 1 := ⟨j - 1, by omega⟩
      rw [List.getElem!_cons_succ, List.getElem!_cons_succ]
      exact list_filter_getElem!_after p l z (by simp only [List.length_cons] at hz; omega) hp
        (fun i hi hiz => by
          have e := h (i + 1) (by simp only [List.length_cons]; omega) (by omega)
          rwa [List.getElem!_cons_succ] at e) j (by omega)

theorem list_filter_length_after {α : Type} [Inhabited α] (p : α → Bool) :
    ∀ (l : List α) (z : ℕ), z < l.length → p l[z]! = false → (∀ j < l.length, j ≠ z → p l[j]! = true) →
      (l.filter p).length + 1 = l.length
  | [], _, hz, _, _ => absurd hz (Nat.not_lt_zero _)
  | a :: l, 0, _, hp, h => by
      rw [List.getElem!_cons_zero] at hp
      have hl : l.filter p = l := List.filter_eq_self.mpr fun x hx => by
        obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hx
        have e := h (i + 1) (by simp only [List.length_cons]; omega) (by omega)
        rwa [List.getElem!_cons_succ, getElem!_pos l i hi] at e
      rw [List.filter_cons_of_neg (by rw [hp]; decide), hl, List.length_cons]
  | a :: l, z + 1, hz, hp, h => by
      have ha : p a = true := by
        have e := h 0 (by simp) (by omega)
        rwa [List.getElem!_cons_zero] at e
      rw [List.getElem!_cons_succ] at hp
      rw [List.filter_cons_of_pos ha, List.length_cons, List.length_cons]
      have := list_filter_length_after p l z (by simp only [List.length_cons] at hz; omega) hp
        (fun i hi hiz => by
          have e := h (i + 1) (by simp only [List.length_cons]; omega) (by omega)
          rwa [List.getElem!_cons_succ] at e)
      omega

theorem array_getElem!_toList {α : Type} [Inhabited α] (ys : Array α) (i : ℕ) : ys[i]! = ys.toList[i]! := by
  rcases ys with ⟨l⟩; exact List.getElem!_toArray

/-- **Filtering out the one element that fails** moves every later one down by one. -/
theorem filter_getElem!_after {α : Type} [Inhabited α] (p : α → Bool) (xs : Array α) (z : ℕ) (hz : z < xs.size)
    (hp : p xs[z]! = false) (h : ∀ j < xs.size, j ≠ z → p xs[j]! = true) :
    ∀ j, z ≤ j → (xs.filter p)[j]! = xs[j + 1]! := by
  intro j hj
  have e : xs.filter p = (xs.toList.filter p).toArray := by apply Array.ext'; simp
  rw [e, array_getElem!_toList (xs.toList.filter p).toArray, List.toList_toArray, array_getElem!_toList xs]
  exact list_filter_getElem!_after p xs.toList z (by simpa using hz) (by rw [← array_getElem!_toList]; exact hp)
    (fun i hi hiz => by rw [← array_getElem!_toList]; exact h i (by simpa using hi) hiz) j hj

/-- Filtering out the one element that fails drops one. -/
theorem filter_size_after {α : Type} [Inhabited α] (p : α → Bool) (xs : Array α) (z : ℕ) (hz : z < xs.size)
    (hp : p xs[z]! = false) (h : ∀ j < xs.size, j ≠ z → p xs[j]! = true) : (xs.filter p).size + 1 = xs.size := by
  have e : xs.filter p = (xs.toList.filter p).toArray := by apply Array.ext'; simp
  rw [e, List.size_toArray, ← Array.length_toList]
  exact list_filter_length_after p xs.toList z (by simpa using hz) (by rw [← array_getElem!_toList]; exact hp)
    (fun i hi hiz => by rw [← array_getElem!_toList]; exact h i (by simpa using hi) hiz)

end FlockC3.HoloValues
