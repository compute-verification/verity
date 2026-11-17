module

public import Proofs.Properties.Verity.SampledTwo.FirstCoins

/-! The second stage's coins agree with the device's on the bytes the second stage carries; the interior
audit's coin evidence carries a signature not bound to the current monitoring period, so an earlier period's passes. -/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open Lean (Json)

def secondDraw (k : Case) : Option (String × ByteArray) := do
  let p ← (Core.canonicalJson k.params >>= VStar.Recursive.SampledTwo.paramsOfJson).toOption
  let e ← (VStar.Recursive.SampledTwo.evidenceOf k.evidence).toOption
  let (_, label) ← p.audit.base.challenger
  let c₂ ← (do Spot.OneStage.commitmentOf (← Flock.canon e.record).toUTF8 : Flock.V _).toOption
  pure (label, c₂.digest)

def SecondShown (C : CaseOf) (Good : Verity.Exchange → Prop) : Prop :=
  ∀ o, FormAccepts C o → Good o → ∀ i it, inputOn C o = some i → i.interior = some it →
    ∃ label after, (C o).bind secondDraw = some (label, after) ∧
      Properties.Challenger.bitsOf it.source <+: (Verity.Sampled.freshCoins label after o.log).flatten

def DevSecond (A : Live₂) (C : CaseOf) (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) : Prop :=
  ∀ o, FormAccepts C o → ∀ label after (b : ByteArray), (C o).bind secondDraw = some (label, after) →
    Properties.Challenger.bitsOf b <+: (Verity.Sampled.freshCoins label after o.log).flatten →
    b.size ≤ A.h.stage * SpotFlock.L → ∀ j < A.h.stage, 0 < stageWidth b j →
    ∃ d, dev o.log (j + 1) = some d ∧
      (SpotFlock.src d).extract 0 (stageWidth b j) = b.extract (j * SpotFlock.L) (j * SpotFlock.L + stageWidth b j)

def StagesOfBytes : Prop :=
  ∀ (R : ℕ) (b : ByteArray), b.size ≤ R * SpotFlock.L →
    ∃ cs : Fin R → FlockDeploy.Coin, SpotFlock.stageSrc cs = b ++ ⟨Array.replicate (R * SpotFlock.L - b.size) 0⟩

theorem stageByte_at {R : ℕ} (cs : Fin R → FlockDeploy.Coin) (j : Fin R) (t : Fin SpotFlock.L)
    (x : Fin (R * SpotFlock.L)) (hx : x.val = j.val * SpotFlock.L + t.val) :
    SpotFlock.stageByte cs x = SpotFlock.coinByte (cs j) t := by
  have : finProdFinEquiv.symm x = (j, t) := by
    rw [Equiv.symm_apply_eq]
    apply Fin.ext
    rw [hx]
    show j.val * SpotFlock.L + t.val = t.val + SpotFlock.L * j.val
    rw [Nat.mul_comm, Nat.add_comm]
  unfold SpotFlock.stageByte
  rw [this]

theorem stageByte_mk {R : ℕ} (cs : Fin R → FlockDeploy.Coin) (j : Fin R) (t : Fin SpotFlock.L)
    (h : j.val * SpotFlock.L + t.val < R * SpotFlock.L) :
    SpotFlock.stageByte cs ⟨j.val * SpotFlock.L + t.val, h⟩ = SpotFlock.coinByte (cs j) t :=
  stageByte_at cs j t _ rfl

theorem stageSrc_extract {R : ℕ} (cs : Fin R → FlockDeploy.Coin) (j : Fin R) {w : ℕ} (hw : w ≤ SpotFlock.L) :
    (SpotFlock.stageSrc cs).extract (j * SpotFlock.L) (j * SpotFlock.L + w) =
      (SpotFlock.src (cs j)).extract 0 w := by
  have hjR : j.val * SpotFlock.L + SpotFlock.L ≤ R * SpotFlock.L := by
    have := Nat.mul_le_mul_right SpotFlock.L (Nat.succ_le_of_lt j.isLt)
    rwa [Nat.succ_mul] at this
  apply ByteArray.ext
  simp only [ByteArray.data_extract, SpotFlock.stageSrc, SpotFlock.src, Properties.Spot.bytesOf]
  apply Array.ext
  · simp only [Array.size_extract, Array.size_ofFn]
    generalize j.val * SpotFlock.L = m at *
    omega
  · intro t h1 h2
    have ht : t < SpotFlock.L := by
      simp only [Array.size_extract, Array.size_ofFn] at h2
      omega
    simp only [Array.getElem_extract, Array.getElem_ofFn, Nat.zero_add]
    rw [stageByte_mk cs j ⟨t, ht⟩]

theorem extract_append_within {b z : ByteArray} {s e : ℕ} (he : e ≤ b.size) : (b ++ z).extract s e = b.extract s e := by
  rw [ByteArray.extract_append]
  have : z.extract (s - b.size) (e - b.size) = ByteArray.empty := by
    rw [ByteArray.extract_eq_empty_iff]
    omega
  rw [this]
  ext1
  simp

theorem stageCoinsOf_src {h : SpotFlock.Held} {b : ByteArray} {cs : SpotFlock.Stage h} (hB : StagesOfBytes)
    (hc : stageCoinsOf h b = some cs) :
    b.size ≤ h.stage * SpotFlock.L ∧ ∀ j : Fin h.stage, 0 < stageWidth b j →
      (SpotFlock.src (cs j)).extract 0 (stageWidth b j) =
        b.extract (j * SpotFlock.L) (j * SpotFlock.L + stageWidth b j) := by
  unfold stageCoinsOf at hc
  by_cases hL : b.size ≤ h.stage * SpotFlock.L
  · rw [if_pos hL] at hc
    cases hc
    refine ⟨hL, fun j hw => ?_⟩
    rw [← stageSrc_extract _ j (w := stageWidth b j) (Nat.min_le_left _ _),
      Classical.epsilon_spec (hB h.stage b hL)]
    apply extract_append_within
    unfold stageWidth at hw ⊢
    generalize j.val * SpotFlock.L = m at *
    omega
  · rw [if_neg hL] at hc
    cases hc

theorem groupOf_second {A : Live₂} {n : ℕ} (h : groupOf A n = .second) : ∃ j, n = j + 1 ∧ j < A.h.stage := by
  unfold groupOf at h
  split_ifs at h with h0 h1
  all_goals first
    | exact ⟨n - 1, by omega, by omega⟩
    | cases h

theorem formReading_coin_second {A : Live₂} {C : CaseOf} {S : SessionReading}
    {dev : List Verity.Release → ℕ → Option FlockDeploy.Coin} {o : Verity.Exchange} {j : ℕ} (hj : j < A.h.stage) :
    (formReading A C S dev).coin o (j + 1) = (inputOn C o).bind fun i => i.interior.bind fun it =>
      (stageCoinsOf A.h it.source).map fun cs => cs ⟨j, hj⟩ := by
  have h0 : j + 1 ≠ 0 := Nat.succ_ne_zero j
  first
    | (simp only [formReading, if_neg h0, Nat.add_sub_cancel, dif_pos hj]; done)
    | (simp [formReading, hj]; done)

theorem formReading_agree_second {A : Live₂} {C : CaseOf} {S : SessionReading}
    {dev : List Verity.Release → ℕ → Option FlockDeploy.Coin} {o : Verity.Exchange} {j : ℕ} (hj : j < A.h.stage)
    (c c' : FlockDeploy.Coin) :
    (formReading A C S dev).agree o (j + 1) c c' ↔
      ∃ i it, inputOn C o = some i ∧ i.interior = some it ∧ prefixAgree (stageWidth it.source j) c c' := by
  have h0 : j + 1 ≠ 0 := Nat.succ_ne_zero j
  have h1 : j + 1 ≤ A.h.stage := hj
  first
    | (simp only [formReading, if_neg h0, if_pos h1, Nat.add_sub_cancel]; done)
    | (simp [formReading, h1]; done)

theorem coinsAgree_second (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (Good : Verity.Exchange → Prop)
    (hShow : SecondShown C Good) (hDev : DevSecond A C dev) (hB : StagesOfBytes) :
    CoinsAgree A (formReading A C S dev) (FormAccepts C) Good .second := by
  intro o ha hg n _ hgn c hc c' hd
  obtain ⟨j, rfl, hj⟩ := groupOf_second hgn
  rw [formReading_coin_second hj] at hc
  rw [formReading_agree_second hj]
  obtain ⟨i, hi, h2⟩ := Option.bind_eq_some_iff.1 hc
  obtain ⟨it, hit, h3⟩ := Option.bind_eq_some_iff.1 h2
  obtain ⟨cs, hcs, rfl⟩ := Option.map_eq_some_iff.1 h3
  refine ⟨i, it, hi, hit, ?_⟩
  obtain ⟨hL, hsrc⟩ := stageCoinsOf_src hB hcs
  unfold prefixAgree
  by_cases hw : stageWidth it.source j = 0
  · rw [hw]
    simp
  · have hcj := hsrc ⟨j, hj⟩ (Nat.pos_of_ne_zero hw)
    obtain ⟨label, after, hdraw, hbits⟩ := hShow o ha hg i it hi hit
    obtain ⟨d, hdd, hds⟩ := hDev o ha label after it.source hdraw hbits hL j hj (Nat.pos_of_ne_zero hw)
    obtain ⟨i', it', hi', hit', hpa⟩ := (formReading_agree_second (S := S) (dev := dev) hj d c').1 (hd d hdd)
    rw [hi] at hi'
    cases hi'
    rw [hit] at hit'
    cases hit'
    unfold prefixAgree at hpa
    rw [hcj, ← hds, hpa]

end Verity.SampledTwo
