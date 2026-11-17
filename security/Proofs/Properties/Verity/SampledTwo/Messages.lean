module

public import Proofs.Properties.Verity.SampledTwo.Reading
public import Proofs.Flock.Claims.Recursion.DeploySessions
public import Proofs.Flock.Claims.Prob
public import Proofs.Flock.Claims.Recursion.Inner
public import Proofs.Properties.Verity.Sampled.Messages

/-! The two-stage subprotocol's frame prover and the messages fixed by their digests (`CallsLe`,
`MessagesFixed`). -/

@[expose] public section

namespace Verity.SampledTwo.Messages

open Claims
open scoped ENNReal

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

def proverOf {Ω : Type} (pick : Ω → List (Msg × Msg) → Msg) : Claims.Prover Ω :=
  ⟨fun _ hist ω => .ret (pick ω hist)⟩

theorem stepWith_pLog (H : Msg → Msg) {Ω : Type} (pick : Ω → List (Msg × Msg) → Msg) (x : Msg) (ω : Ω)
    (vn : List Msg → OC Out) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Transcript),
      (stepWith H (proverOf pick) x ω vn n hist as t).pLog = t.pLog
  | 0, _, _, _ => rfl
  | n + 1, hist, as, t => by
    rcases hvb : (vn (as ++ [(((proverOf pick).next x hist ω).eval H).1])).eval H with ⟨o, lo⟩
    cases o with
    | msg m =>
      rw [RecDeploy.stepWith_succ_msg H _ x ω vn n hist as t hvb, stepWith_pLog]
      simp [proverOf, OC.eval]
    | acc =>
      rw [RecDeploy.stepWith_succ_acc H _ x ω vn n hist as t hvb]
      simp [proverOf, OC.eval]
    | rej =>
      rw [RecDeploy.stepWith_succ_rej H _ x ω vn n hist as t hvb]
      simp [proverOf, OC.eval]

theorem run_pLog (H : Msg → Msg) {Ω Coin : Type} [Inhabited Coin] (pick : Ω → List (Msg × Msg) → Msg)
    (Vr : Verifier Coin) (x : Msg) (ω : Ω) (r : Fin (Vr.k x) → Coin) :
    (Claims.run H (proverOf pick) Vr x ω r).pLog = [] :=
  stepWith_pLog H pick x ω _ _ [] [] ⟨[], [], [], .rej⟩

noncomputable def shownCoin (m : Msg) : Option FlockDeploy.Coin := by
  classical
  exact if h : ∃ c : FlockDeploy.Coin, c.bytes = m then some h.choose else none

noncomputable def shownAt (hist : List (Msg × Msg)) (j : ℕ) : Option FlockDeploy.Coin :=
  (hist[j]?).bind fun e => shownCoin e.2

def Shows {A : Live₂} {x : Msg} (hist : List (Msg × Msg)) (r : Coins A x) : Prop :=
  ∀ (j : Fin (A.L.V.k x)) c, shownAt hist j.val = some c → r j = c

noncomputable def merge {A : Live₂} {x : Msg} (hist : List (Msg × Msg)) (y : Coins A x) : Coins A x :=
  fun j => (shownAt hist j.val).getD (y j)

noncomputable def complete {A : Live₂} {x : Msg} (hist : List (Msg × Msg)) : Coins A x :=
  merge hist fun _ => default

noncomputable def opening {A : Live₂} {x : Msg} {Ω : Type} (out : Ω → Coins A x → Verity.Exchange) (Rd : Reading A)
    (Dg : Verity.Exchange → ℕ → Option Msg) (i : ℕ) (ω : Ω) (hist : List (Msg × Msg)) (r : Coins A x) :
    Option Msg := by
  classical
  exact (Rd.msg (out ω r) i).filter fun m => decide (Dg (out ω (complete hist)) i = some (Verity.sha512Msg m))

open Classical in
/-- **The most likely value of `f` on `S`**: a message `f` takes on at least as many members of `S` as any other;
`[]` where it takes none. -/
noncomputable def modeOn {β : Type} [Fintype β] (S : β → Prop) (f : β → Option Msg) : Msg :=
  if ∃ m b, S b ∧ f b = some m then
    Classical.epsilon fun m => (∃ b, S b ∧ f b = some m) ∧
      ∀ m', (Finset.univ.filter fun b => S b ∧ f b = some m').card ≤
        (Finset.univ.filter fun b => S b ∧ f b = some m).card
  else []

noncomputable def modePick {A : Live₂} {x : Msg} {Ω : Type} (out : Ω → Coins A x → Verity.Exchange) (Rd : Reading A)
    (Dg : Verity.Exchange → ℕ → Option Msg) (ω : Ω) (hist : List (Msg × Msg)) : Msg :=
  modeOn (Shows (A := A) (x := x) hist) (opening out Rd Dg hist.length ω hist)

noncomputable def outOf (V : Vocab d α) {A : Live₂} {x : Msg} (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ)
    (ω : TapeOf V θ F c σ) (r : Coins A x) : Verity.Exchange :=
  Verity.Game.Tape.out (V.game θ F c) σ (place ω r)

noncomputable def prover (V : Vocab d α) {A : Live₂} {x : Msg} (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ)
    (Rd : Reading A) (Dg : Verity.Exchange → ℕ → Option Msg) : Claims.Prover (TapeOf V θ F c σ) :=
  proverOf (modePick (outOf V θ F c σ place) Rd Dg)

noncomputable def frame (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ)
    (Rd : Reading A) (Dg : Verity.Exchange → ℕ → Option Msg) : Frame V A x θ F c σ :=
  ⟨place, prover V θ F c σ place Rd Dg⟩

theorem frame_out (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ)
    (Rd : Reading A) (Dg : Verity.Exchange → ℕ → Option Msg) (ω : TapeOf V θ F c σ) (r : Coins A x) :
    (frame V A x θ F c σ place Rd Dg).out ω r = outOf V θ F c σ place ω r := rfl

/-- `V.cost` is declared, so nothing else ties it to the verifier's SHA-512 calls: at a vocabulary whose
`cost` is `0` it fails as soon as the verifier hashes once. -/
def VerifierCallsLe (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (P : Claims.Prover (TapeOf V θ F c σ)) : Prop :=
  ∀ (ω : TapeOf V θ F c σ) (r : Coins A x),
    (Claims.run FlockDeploy.sha512 P A.L.V x ω r).vLog.length ≤ V.cost θ F c σ

theorem callsLe_of (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ)
    (pick : TapeOf V θ F c σ → List (Msg × Msg) → Msg) (hP : Fr.P = proverOf pick)
    (h : VerifierCallsLe V A x θ F c σ Fr.P) : CallsLe V A x θ F c σ Fr := by
  intro ω r
  have h0 := run_pLog FlockDeploy.sha512 pick A.L.V x ω r
  rw [← hP] at h0
  rw [h0, List.length_nil, Nat.zero_add]
  exact h ω r

theorem callsLe (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ)
    (Rd : Reading A) (Dg : Verity.Exchange → ℕ → Option Msg)
    (h : VerifierCallsLe V A x θ F c σ (prover V θ F c σ place Rd Dg)) :
    CallsLe V A x θ F c σ (frame V A x θ F c σ place Rd Dg) :=
  callsLe_of V A x θ F c σ _ _ rfl h

def MessagesReached (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) (miss : ℝ≥0∞) : Prop :=
  ENNReal.ofReal (Claims.Pr fun s : TapeOf V θ F c σ × Coins A x =>
      Acc (Fr.out s.1 s.2) ∧ Good (Fr.out s.1 s.2) ∧
        ∃ i < (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2).msgs.length, Rd.msg (Fr.out s.1 s.2) i ≠
          msgAt (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2) i) ≤ miss

def RunsThrough {V : Vocab d α} {A : Live₂} {x : Msg} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {σ : Verity.Game.Strategy (V.game θ F c)} (Fr : Frame V A x θ F c σ) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) : Prop :=
  ∀ (ω : TapeOf V θ F c σ) (r : Coins A x), Acc (Fr.out ω r) → Good (Fr.out ω r) →
    (∀ i < (Claims.run FlockDeploy.sha512 Fr.P A.L.V x ω r).msgs.length,
      Rd.msg (Fr.out ω r) i = msgAt (Claims.run FlockDeploy.sha512 Fr.P A.L.V x ω r) i) →
    A.L.V.k x ≤ (Claims.run FlockDeploy.sha512 Fr.P A.L.V x ω r).msgs.length

theorem messagesFixed_of_reached (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) (miss : ℝ≥0∞) (hR : RunsThrough Fr Rd Acc Good)
    (h : MessagesReached V A x θ F c σ Fr Rd Acc Good miss) : MessagesFixed V A x θ F c σ Fr Rd Acc Good miss := by
  refine le_trans (ENNReal.ofReal_le_ofReal (Claims.Pr_mono fun s hs => ?_)) h
  obtain ⟨hA, hG, i, hi, hne⟩ := hs
  refine ⟨hA, hG, ?_⟩
  by_contra hall
  push Not at hall
  exact hne (hall i (lt_of_lt_of_le hi (hR s.1 s.2 hA hG hall)))

def runHist (t : Claims.Transcript) (i : ℕ) : List (Msg × Msg) := (t.msgs.take i).map RecDeploy.histOf

noncomputable def playHist (A : Live₂) (x : Msg) (r : Coins A x) (ms : ℕ → Msg) (i : ℕ) : List (Msg × Msg) :=
  (List.range i).map fun j => (ms j, msgOf ((A.L.V.next x ((List.range (j + 1)).map ms)
    ((List.range (j + 1)).map (coinSeq r))).eval FlockDeploy.sha512).1)

def playMsgs {A : Live₂} (Rd : Reading A) (o : Verity.Exchange) (j : ℕ) : Msg := (Rd.msg o j).getD []

def MissAt {V : Vocab d α} {A : Live₂} {x : Msg} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {σ : Verity.Game.Strategy (V.game θ F c)} (Fr : Frame V A x θ F c σ) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) (i : ℕ) (s : TapeOf V θ F c σ × Coins A x) : Prop :=
  Acc (Fr.out s.1 s.2) ∧ Good (Fr.out s.1 s.2) ∧
    i < (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2).msgs.length ∧
    Rd.msg (Fr.out s.1 s.2) i ≠ msgAt (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2) i ∧
    ∀ j < i, Rd.msg (Fr.out s.1 s.2) j = msgAt (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2) j

def ReachedOfRounds (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) : Prop :=
  ∀ (Fr : Frame V A x θ F c σ) (Rd : Reading A) (Acc Good : Verity.Exchange → Prop) (miss : ℕ → ℝ≥0∞),
    (∀ i < A.L.V.k x, ENNReal.ofReal (Claims.Pr (MissAt Fr Rd Acc Good i)) ≤ miss i) →
      MessagesReached V A x θ F c σ Fr Rd Acc Good (∑ i ∈ Finset.range (A.L.V.k x), miss i)

theorem reachedOfRounds (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) : ReachedOfRounds V A x θ F c σ := by
  classical
  intro Fr Rd Acc Good miss h
  unfold MessagesReached
  refine le_trans (ENNReal.ofReal_le_ofReal ((Claims.Pr_mono fun s hs => ?_).trans
    (Verity.Game.pr_exists_le (fun i => MissAt Fr Rd Acc Good i) (Finset.range (A.L.V.k x))))) ?_
  · obtain ⟨hA, hG, hex⟩ := hs
    have hlen := Claims.Recursion.run_length_le FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2
    refine ⟨Nat.find hex, Finset.mem_range.2 (lt_of_lt_of_le (Nat.find_spec hex).1 hlen), hA, hG,
      (Nat.find_spec hex).1, (Nat.find_spec hex).2, fun j hj => ?_⟩
    by_contra hne
    exact Nat.find_min hex hj ⟨lt_trans hj (Nat.find_spec hex).1, hne⟩
  · rw [ENNReal.ofReal_sum_of_nonneg fun i _ => Claims.Pr_nonneg _]
    exact Finset.sum_le_sum fun i hi => h i (Finset.mem_range.1 hi)

section Fork

open Classical

variable {β γ κ : Type} [Fintype β] [Fintype γ]

noncomputable def classOf (cls : β → κ) (x : β) : Finset β := Finset.univ.filter fun x' => cls x' = cls x

theorem mem_classOf (cls : β → κ) {x x' : β} : x' ∈ classOf cls x ↔ cls x' = cls x := by
  simp [classOf]

theorem classOf_card_pos (cls : β → κ) (x : β) : (0 : ℝ) < (classOf cls x).card :=
  Nat.cast_pos.2 (Finset.card_pos.2 ⟨x, (mem_classOf cls).2 rfl⟩)

theorem classOf_eq (cls : β → κ) {x x' : β} (h : cls x' = cls x) : classOf cls x' = classOf cls x := by
  unfold classOf
  rw [h]

theorem sum_classAvg (cls : β → κ) (g : β → ℝ) :
    ∑ x, (∑ x' ∈ classOf cls x, g x') / (classOf cls x).card = ∑ x, g x := by
  have h : ∀ x, (∑ x' ∈ classOf cls x, g x') / ((classOf cls x).card : ℝ) =
      ∑ x', if cls x' = cls x then g x' / ((classOf cls x').card : ℝ) else 0 := by
    intro x
    rw [Finset.sum_div]
    unfold classOf
    rw [Finset.sum_filter]
    refine Finset.sum_congr rfl fun x' _ => ?_
    by_cases hx : cls x' = cls x
    · rw [if_pos hx, if_pos hx, hx]
    · rw [if_neg hx, if_neg hx]
  rw [Finset.sum_congr rfl fun x _ => h x, Finset.sum_comm]
  refine Finset.sum_congr rfl fun x' _ => ?_
  rw [← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul]
  have hfe : (Finset.univ.filter fun x => cls x' = cls x) = classOf cls x' := by
    unfold classOf
    exact Finset.filter_congr fun _ _ => eq_comm
  rw [hfe, mul_div_left_comm, div_self (classOf_card_pos cls x').ne', mul_one]

def bad (cls : β → κ) (F : κ → β → Option Msg) (v : κ → Msg) (x x' : β) : Prop :=
  ∃ m, F (cls x) x' = some m ∧ m ≠ v (cls x)

def fork (cls : β → κ) (F : κ → β → Option Msg) (x x' : β) : Prop :=
  ∃ u w, F (cls x) x = some u ∧ F (cls x) x' = some w ∧ u ≠ w

noncomputable def badCount (cls : β → κ) (F : κ → β → Option Msg) (v : κ → Msg) (x : β) : ℝ :=
  (((classOf cls x).filter (bad cls F v x)).card : ℝ)

noncomputable def forkCount (cls : β → κ) (F : κ → β → Option Msg) (x : β) : ℝ :=
  (((classOf cls x).filter (fork cls F x)).card : ℝ)

theorem card_ne_le (s : Finset β) (f : β → Option Msg) (u v : Msg)
    (h : (s.filter fun x => f x = some u).card ≤ (s.filter fun x => f x = some v).card) :
    (s.filter fun x => ∃ m, f x = some m ∧ m ≠ v).card ≤ (s.filter fun x => ∃ m, f x = some m ∧ m ≠ u).card := by
  have key : ∀ a, (s.filter fun x => f x = some a).card + (s.filter fun x => ∃ m, f x = some m ∧ m ≠ a).card =
      (s.filter fun x => (f x).isSome).card := by
    intro a
    rw [← Finset.card_filter_add_card_filter_not (s := s.filter fun x => (f x).isSome)
      (fun x => f x = some a), Finset.filter_filter, Finset.filter_filter]
    congr 2
    · exact Finset.filter_congr fun x _ => by cases f x <;> simp
    · exact Finset.filter_congr fun x _ => by cases f x <;> simp
  have hu := key u
  have hv := key v
  omega

theorem miss_le_sqrt [Nonempty β] [Nonempty γ] (cls : β → κ) (res : β → γ → β) (F : κ → β → Option Msg)
    (v : κ → Msg) (Miss : β → Prop)
    (hunif : ∀ x (φ : β → ℝ), (∑ y, φ (res x y)) * (classOf cls x).card =
      Fintype.card γ * ∑ x' ∈ classOf cls x, φ x')
    (hmode : ∀ x m, ((classOf cls x).filter fun x' => F (cls x) x' = some m).card ≤
      ((classOf cls x).filter fun x' => F (cls x) x' = some (v (cls x))).card)
    (hmiss : ∀ x, Miss x → bad cls F v x x) :
    Claims.Pr Miss ≤ Real.sqrt (Claims.Pr fun s : β × γ => fork cls F s.1 (res s.1 s.2)) := by
  have hn : (0 : ℝ) < Fintype.card β := Nat.cast_pos.2 Fintype.card_pos
  have hm : (0 : ℝ) < Fintype.card γ := Nat.cast_pos.2 Fintype.card_pos
  have hN : ∀ x, (0 : ℝ) < (classOf cls x).card := classOf_card_pos cls
  have hA : Claims.Pr Miss ≤ (∑ x, badCount cls F v x / (classOf cls x).card) / Fintype.card β := by
    rw [Claims.Pr_eq_sum]
    refine div_le_div_of_nonneg_right ?_ hn.le
    calc (∑ x, if Miss x then (1 : ℝ) else 0) ≤ ∑ x, if bad cls F v x x then (1 : ℝ) else 0 :=
          Finset.sum_le_sum fun x _ => by
            by_cases hx : Miss x
            · rw [if_pos hx, if_pos (hmiss x hx)]
            · rw [if_neg hx]
              split_ifs <;> norm_num
      _ = ∑ x, badCount cls F v x / (classOf cls x).card := by
          rw [← sum_classAvg cls (fun x => if bad cls F v x x then (1 : ℝ) else 0)]
          refine Finset.sum_congr rfl fun x _ => ?_
          rw [Finset.sum_boole]
          unfold badCount
          congr 3
          exact Finset.filter_congr fun x' hx' => by
            rw [mem_classOf] at hx'
            unfold bad
            rw [hx']
  have hrow : ∀ x, (∑ y, if fork cls F x (res x y) then (1 : ℝ) else 0) =
      Fintype.card γ * (forkCount cls F x / (classOf cls x).card) := by
    intro x
    have h1 := hunif x fun x' => if fork cls F x x' then 1 else 0
    have h2 : (∑ x' ∈ classOf cls x, if fork cls F x x' then (1 : ℝ) else 0) = forkCount cls F x := by
      rw [Finset.sum_boole]
      rfl
    rw [h2] at h1
    rw [mul_div_assoc', eq_div_iff (hN x).ne']
    exact h1
  have hge : ∑ x, (badCount cls F v x / (classOf cls x).card) ^ 2 ≤
      ∑ x, forkCount cls F x / (classOf cls x).card := by
    rw [← sum_classAvg cls (fun x => forkCount cls F x / (classOf cls x).card)]
    refine Finset.sum_le_sum fun x _ => ?_
    have hstep : ∀ x'' ∈ (classOf cls x).filter (bad cls F v x), badCount cls F v x ≤ forkCount cls F x'' := by
      intro x'' hx''
      obtain ⟨hcx, u, hu, -⟩ := Finset.mem_filter.1 hx''
      have hc : cls x'' = cls x := (mem_classOf cls).1 hcx
      unfold badCount forkCount
      rw [classOf_eq cls hc]
      refine Nat.cast_le.2 (le_trans (le_of_eq ?_)
        (le_of_le_of_eq (card_ne_le (classOf cls x) (F (cls x)) u (v (cls x)) (hmode x u)) ?_))
      · exact congrArg Finset.card (Finset.filter_congr fun x' _ => Iff.rfl)
      congr 1
      refine Finset.filter_congr fun x' _ => ?_
      unfold fork
      constructor
      · rintro ⟨w, hw, hne⟩
        exact ⟨u, w, by rw [hc, hu], by rw [hc]; exact hw, Ne.symm hne⟩
      · rintro ⟨u', w, hu', hw, hne⟩
        rw [hc, hu] at hu'
        rw [hc] at hw
        cases hu'
        exact ⟨w, hw, Ne.symm hne⟩
    have hsq : badCount cls F v x ^ 2 ≤ ∑ x'' ∈ classOf cls x, forkCount cls F x'' := by
      calc badCount cls F v x ^ 2
            = ∑ _x'' ∈ (classOf cls x).filter (bad cls F v x), badCount cls F v x := by
              rw [Finset.sum_const, nsmul_eq_mul, sq]
              rfl
        _ ≤ ∑ x'' ∈ (classOf cls x).filter (bad cls F v x), forkCount cls F x'' := Finset.sum_le_sum hstep
        _ ≤ ∑ x'' ∈ classOf cls x, forkCount cls F x'' :=
            Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _) fun _ _ _ => by
              unfold forkCount
              positivity
    have hcl : ∀ x'' ∈ classOf cls x, forkCount cls F x'' / ((classOf cls x'').card : ℝ) =
        forkCount cls F x'' / ((classOf cls x).card : ℝ) := fun x'' hx'' => by
      rw [classOf_eq cls ((mem_classOf cls).1 hx'')]
    rw [Finset.sum_congr rfl hcl, ← Finset.sum_div, div_div, div_pow, ← sq]
    exact div_le_div_of_nonneg_right hsq (by positivity)
  have hB : (∑ x, (badCount cls F v x / (classOf cls x).card) ^ 2) / Fintype.card β ≤
      Claims.Pr fun s : β × γ => fork cls F s.1 (res s.1 s.2) := by
    rw [Claims.Pr_eq_sum, Fintype.sum_prod_type, Fintype.card_prod, Nat.cast_mul]
    dsimp only
    rw [Finset.sum_congr rfl fun x _ => hrow x, ← Finset.mul_sum, mul_comm (Fintype.card β : ℝ), ← div_div,
      mul_div_cancel_left₀ _ hm.ne']
    exact div_le_div_of_nonneg_right hge hn.le
  have hCS : (∑ x, badCount cls F v x / (classOf cls x).card) ^ 2 ≤
      (Fintype.card β : ℝ) * ∑ x, (badCount cls F v x / (classOf cls x).card) ^ 2 := by
    have := Finset.sum_mul_sq_le_sq_mul_sq Finset.univ (fun _ => (1 : ℝ))
      (fun x => badCount cls F v x / (classOf cls x).card)
    simpa [Finset.card_univ] using this
  have hA0 : 0 ≤ (∑ x, badCount cls F v x / (classOf cls x).card) / Fintype.card β :=
    div_nonneg (Finset.sum_nonneg fun x _ => div_nonneg (by unfold badCount; positivity) (hN x).le) hn.le
  calc Claims.Pr Miss ≤ (∑ x, badCount cls F v x / (classOf cls x).card) / Fintype.card β := hA
    _ = Real.sqrt (((∑ x, badCount cls F v x / (classOf cls x).card) / Fintype.card β) ^ 2) :=
        (Real.sqrt_sq hA0).symm
    _ ≤ Real.sqrt (Claims.Pr fun s : β × γ => fork cls F s.1 (res s.1 s.2)) := by
        refine Real.sqrt_le_sqrt ?_
        rw [div_pow]
        calc (∑ x, badCount cls F v x / (classOf cls x).card) ^ 2 / (Fintype.card β : ℝ) ^ 2
            ≤ ((Fintype.card β : ℝ) * ∑ x, (badCount cls F v x / (classOf cls x).card) ^ 2) /
                (Fintype.card β : ℝ) ^ 2 := div_le_div_of_nonneg_right hCS (by positivity)
          _ = (∑ x, (badCount cls F v x / (classOf cls x).card) ^ 2) / Fintype.card β := by
              rw [sq (Fintype.card β : ℝ), mul_div_mul_left _ _ hn.ne']
          _ ≤ _ := hB

end Fork

section Mode

open Classical

theorem modeOn_spec {β : Type} [Fintype β] (S : β → Prop) (f : β → Option Msg) (m : Msg) :
    (Finset.univ.filter fun b => S b ∧ f b = some m).card ≤
      (Finset.univ.filter fun b => S b ∧ f b = some (modeOn S f)).card := by
  by_cases hex : ∃ m b, S b ∧ f b = some m
  · have hP : ∃ m₀, (∃ b, S b ∧ f b = some m₀) ∧ ∀ m', (Finset.univ.filter fun b => S b ∧ f b = some m').card ≤
        (Finset.univ.filter fun b => S b ∧ f b = some m₀).card := by
      obtain ⟨m₁, b₁, hb₁⟩ := hex
      have hne : ((Finset.univ.filter S).biUnion fun b => (f b).toFinset).Nonempty :=
        ⟨m₁, Finset.mem_biUnion.2 ⟨b₁, by simp [hb₁.1], by simp [hb₁.2]⟩⟩
      obtain ⟨m₀, hm₀, hmax⟩ := Finset.exists_max_image _
        (fun m => (Finset.univ.filter fun b => S b ∧ f b = some m).card) hne
      refine ⟨m₀, ?_, fun m' => ?_⟩
      · obtain ⟨b, hb, hfb⟩ := Finset.mem_biUnion.1 hm₀
        exact ⟨b, (Finset.mem_filter.1 hb).2, Option.mem_toFinset.1 hfb⟩
      · by_cases hm' : m' ∈ (Finset.univ.filter S).biUnion fun b => (f b).toFinset
        · exact hmax m' hm'
        · have h0 : (Finset.univ.filter fun b => S b ∧ f b = some m') = ∅ := by
            refine Finset.filter_eq_empty_iff.2 fun b _ hb => hm' ?_
            exact Finset.mem_biUnion.2 ⟨b, by simp [hb.1], by simp [hb.2]⟩
          rw [h0, Finset.card_empty]
          exact Nat.zero_le _
    unfold modeOn
    rw [if_pos hex]
    exact (Classical.epsilon_spec hP).2 m
  · have h0 : (Finset.univ.filter fun b => S b ∧ f b = some m) = ∅ :=
      Finset.filter_eq_empty_iff.2 fun b _ hb => hex ⟨m, b, hb⟩
    rw [h0, Finset.card_empty]
    exact Nat.zero_le _

theorem shows_merge {A : Live₂} {x : Msg} (hist : List (Msg × Msg)) (y : Coins A x) : Shows hist (merge hist y) := by
  intro j c h
  simp only [merge, h, Option.getD_some]

theorem merge_of_shows {A : Live₂} {x : Msg} {hist : List (Msg × Msg)} {r : Coins A x} (h : Shows hist r) :
    merge hist r = r := by
  funext j
  unfold merge
  rcases hs : shownAt hist j.val with _ | c
  · rfl
  · exact (h j c hs).symm

theorem fiber_card {A : Live₂} {x : Msg} (hist : List (Msg × Msg)) {r r' : Coins A x} (hr : Shows hist r)
    (hr' : Shows hist r') :
    (Finset.univ.filter fun y => merge hist y = r).card = (Finset.univ.filter fun y => merge hist y = r').card := by
  refine Finset.card_nbij' (fun y j => if (shownAt hist j.val).isSome then y j else r' j)
    (fun y j => if (shownAt hist j.val).isSome then y j else r j) ?_ ?_ ?_ ?_
  · intro y hy
    simp only [Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_setOf_eq, Finset.mem_filter] at hy ⊢
    funext j
    unfold merge
    rcases hs : shownAt hist j.val with _ | c
    · simp [hs]
    · exact (hr' j c hs).symm
  · intro y hy
    simp only [Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_setOf_eq, Finset.mem_filter] at hy ⊢
    funext j
    unfold merge
    rcases hs : shownAt hist j.val with _ | c
    · simp [hs]
    · exact (hr j c hs).symm
  · intro y hy
    simp only [Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_setOf_eq, Finset.mem_filter] at hy
    funext j
    rcases hs : shownAt hist j.val with _ | c
    · have := congrFun hy j
      simp only [merge, hs, Option.getD_none] at this
      simp [hs, this]
    · simp [hs]
  · intro y hy
    simp only [Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_setOf_eq, Finset.mem_filter] at hy
    funext j
    rcases hs : shownAt hist j.val with _ | c
    · have := congrFun hy j
      simp only [merge, hs, Option.getD_none] at this
      simp [hs, this]
    · simp [hs]

theorem sum_merge {A : Live₂} {x : Msg} (hist : List (Msg × Msg)) (φ : Coins A x → ℝ) :
    (∑ y, φ (merge hist y)) * ((Finset.univ.filter fun r : Coins A x => Shows hist r).card : ℝ) =
      (Fintype.card (Coins A x) : ℝ) * ∑ r ∈ Finset.univ.filter (fun r => Shows hist r), φ r := by
  have hS : ∀ y ∈ (Finset.univ : Finset (Coins A x)),
      merge hist y ∈ Finset.univ.filter fun r : Coins A x => Shows hist r :=
    fun y _ => Finset.mem_filter.2 ⟨Finset.mem_univ _, shows_merge hist y⟩
  have hfib : ∀ ψ : Coins A x → ℝ, ∑ y, ψ (merge hist y) =
      ((Finset.univ.filter fun y : Coins A x => merge hist y = complete hist).card : ℝ) *
        ∑ r ∈ Finset.univ.filter (fun r : Coins A x => Shows hist r), ψ r := by
    intro ψ
    rw [← Finset.sum_fiberwise_of_maps_to hS (fun y => ψ (merge hist y)), Finset.mul_sum]
    refine Finset.sum_congr rfl fun r hr => ?_
    rw [Finset.sum_congr rfl (fun y hy => by rw [(Finset.mem_filter.1 hy).2]), Finset.sum_const, nsmul_eq_mul,
      fiber_card hist (Finset.mem_filter.1 hr).2 (shows_merge hist _)]
    rfl
  have h1 := hfib fun _ => 1
  simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_one] at h1
  rw [hfib φ, h1]
  ring

end Mode

def MsgOpens (A : Live₂) (x : Msg) (Rd : Reading A) (Acc Good : Verity.Exchange → Prop)
    (Dg : Verity.Exchange → ℕ → Option Msg) : Prop :=
  ∀ o, Acc o → Good o → ∀ i < A.L.V.k x, ∃ m, Rd.msg o i = some m ∧ Dg o i = some (Verity.sha512Msg m)

def DigestBefore {V : Vocab d α} {A : Live₂} {x : Msg} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {σ : Verity.Game.Strategy (V.game θ F c)} (Fr : Frame V A x θ F c σ) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) (Dg : Verity.Exchange → ℕ → Option Msg) (i : ℕ) : Prop :=
  ∀ (ω : TapeOf V θ F c σ) (r r' : Coins A x), Acc (Fr.out ω r) → Good (Fr.out ω r) →
    Shows (playHist A x r (playMsgs Rd (Fr.out ω r)) i) r' → Dg (Fr.out ω r') i = Dg (Fr.out ω r) i

def RunShown (A : Live₂) (x : Msg) : Prop :=
  ∀ (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (r : Coins A x) (i : ℕ),
    Shows (runHist (Claims.run FlockDeploy.sha512 P A.L.V x ω r) i) r ∧
      ∀ r', Shows (runHist (Claims.run FlockDeploy.sha512 P A.L.V x ω r) i) r' →
        (Claims.run FlockDeploy.sha512 P A.L.V x ω r').msgs.take i =
          (Claims.run FlockDeploy.sha512 P A.L.V x ω r).msgs.take i

noncomputable def histAt (V : Vocab d α) {A : Live₂} {x : Msg} (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ)
    (Rd : Reading A) (Dg : Verity.Exchange → ℕ → Option Msg) (i : ℕ) (ω : TapeOf V θ F c σ) (r : Coins A x) :
    List (Msg × Msg) :=
  runHist (Claims.run FlockDeploy.sha512 (prover V θ F c σ place Rd Dg) A.L.V x ω r) i

noncomputable def msgFinder (V : Vocab d α) {A : Live₂} {x : Msg} (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c))
    (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ) (Rd : Reading A)
    (Dg : Verity.Exchange → ℕ → Option Msg) (i q : ℕ) : Claims.Finder where
  S := (TapeOf V θ F c σ × Coins A x) × Coins A x
  out s :=
    match opening (outOf V θ F c σ place) Rd Dg i s.1.1 (histAt V θ F c σ place Rd Dg i s.1.1 s.1.2) s.1.2,
      opening (outOf V θ F c σ place) Rd Dg i s.1.1 (histAt V θ F c σ place Rd Dg i s.1.1 s.1.2)
        (merge (histAt V θ F c σ place Rd Dg i s.1.1 s.1.2) s.2) with
    | some m, some m' => if m = m' then none else some (m, m')
    | _, _ => none
  cost _ := q

def MissBound (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) : Prop :=
  ∀ (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ) (Rd : Reading A) (Acc Good : Verity.Exchange → Prop)
    (Dg : Verity.Exchange → ℕ → Option Msg) (i q : ℕ),
    MsgOpens A x Rd Acc Good Dg → DigestBefore (frame V A x θ F c σ place Rd Dg) Rd Acc Good Dg i → RunShown A x →
      ENNReal.ofReal (Claims.Pr (MissAt (frame V A x θ F c σ place Rd Dg) Rd Acc Good i)) ≤
        ENNReal.ofReal (Real.sqrt ((msgFinder V θ F c σ place Rd Dg i q).Pr Verity.sha512Msg))

theorem opening_some {A : Live₂} {x : Msg} {Ω : Type} {out : Ω → Coins A x → Verity.Exchange} {Rd : Reading A}
    {Dg : Verity.Exchange → ℕ → Option Msg} {i : ℕ} {ω : Ω} {hist : List (Msg × Msg)} {r : Coins A x} {u : Msg}
    (h : opening out Rd Dg i ω hist r = some u) :
    Rd.msg (out ω r) i = some u ∧ Dg (out ω (complete hist)) i = some (Verity.sha512Msg u) := by
  unfold opening at h
  rw [Option.filter_eq_some_iff] at h
  exact ⟨h.1, of_decide_eq_true h.2⟩

theorem opening_of {A : Live₂} {x : Msg} {Ω : Type} {out : Ω → Coins A x → Verity.Exchange} {Rd : Reading A}
    {Dg : Verity.Exchange → ℕ → Option Msg} {i : ℕ} {ω : Ω} {hist : List (Msg × Msg)} {r : Coins A x} {u : Msg}
    (h1 : Rd.msg (out ω r) i = some u) (h2 : Dg (out ω (complete hist)) i = some (Verity.sha512Msg u)) :
    opening out Rd Dg i ω hist r = some u := by
  unfold opening
  rw [h1, Option.filter_some_eq_some]
  exact decide_eq_true h2

theorem runHist_length (t : Claims.Transcript) {i : ℕ} (h : i ≤ t.msgs.length) : (runHist t i).length = i := by
  simp [runHist, h]

theorem outMsg_eq (o : Out) : Claims.Recursion.outMsg o = msgOf o := by
  cases o <;> rfl

theorem run_msgAt {Ω : Type} (pick : Ω → List (Msg × Msg) → Msg) {Coin : Type} [Inhabited Coin]
    (Vr : Verifier Coin) (x : Msg) (ω : Ω) (r : Fin (Vr.k x) → Coin) (i : ℕ)
    (h : i < (Claims.run FlockDeploy.sha512 (proverOf pick) Vr x ω r).msgs.length) :
    msgAt (Claims.run FlockDeploy.sha512 (proverOf pick) Vr x ω r) i =
      some (pick ω (runHist (Claims.run FlockDeploy.sha512 (proverOf pick) Vr x ω r) i)) := by
  obtain ⟨new, hnew, hget⟩ := RecDeploy.run_get FlockDeploy.sha512 (proverOf pick) Vr x ω r i h
  unfold msgAt runHist
  rw [hget, ← hnew]
  rfl

theorem map_fst_take_eq {L : List (Msg × Out)} {ms : ℕ → Msg} {n : ℕ} (hn : n ≤ L.length)
    (h : ∀ l < n, (L[l]?).map Prod.fst = some (ms l)) : (L.take n).map Prod.fst = (List.range n).map ms := by
  apply List.ext_getElem?
  intro l
  rw [List.getElem?_map, List.getElem?_take, List.getElem?_map]
  by_cases hl : l < n
  · rw [if_pos hl, h l hl, List.getElem?_range hl]
    rfl
  · rw [if_neg hl, List.getElem?_eq_none (l := List.range n) (by simpa using hl)]
    rfl

theorem playHist_eq {Ω : Type} (A : Live₂) (x : Msg) (P : Claims.Prover Ω) (ω : Ω) (r : Coins A x) (ms : ℕ → Msg)
    (i : ℕ) (hi : i ≤ (Claims.run FlockDeploy.sha512 P A.L.V x ω r).msgs.length)
    (h : ∀ l < i, msgAt (Claims.run FlockDeploy.sha512 P A.L.V x ω r) l = some (ms l)) :
    playHist A x r ms i = runHist (Claims.run FlockDeploy.sha512 P A.L.V x ω r) i := by
  apply List.ext_getElem?
  intro n
  unfold playHist runHist
  rw [List.getElem?_map, List.getElem?_map, List.getElem?_take]
  by_cases hn : n < i
  · rw [List.getElem?_range hn, if_pos hn]
    obtain ⟨new, hnew, hget⟩ := RecDeploy.run_get FlockDeploy.sha512 P A.L.V x ω r n (by omega)
    rw [hget]
    have hp : ((P.next x (new.map RecDeploy.histOf) ω).eval FlockDeploy.sha512).1 = ms n := by
      have := h n hn
      unfold msgAt at this
      rw [hget] at this
      exact Option.some.inj this
    have hfst : new.map Prod.fst = (List.range n).map ms := by
      rw [hnew]
      exact map_fst_take_eq (by omega) fun l hl => h l (by omega)
    simp only [Option.map_some]
    rw [hp, hfst, List.range_succ, List.map_append, List.map_singleton]
    simp only [RecDeploy.histOf, outMsg_eq]
  · rw [if_neg hn, List.getElem?_eq_none (l := List.range i) (by simpa using hn)]
    rfl

theorem hist_eq_iff {A : Live₂} {x : Msg} (hS : RunShown A x) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω)
    (r r' : Coins A x) (i : ℕ) :
    runHist (Claims.run FlockDeploy.sha512 P A.L.V x ω r') i = runHist (Claims.run FlockDeploy.sha512 P A.L.V x ω r) i ↔
      Shows (runHist (Claims.run FlockDeploy.sha512 P A.L.V x ω r) i) r' := by
  constructor
  · intro h
    rw [← h]
    exact (hS _ P ω r' i).1
  · intro h
    unfold runHist
    rw [(hS _ P ω r i).2 r' h]

section Round

variable (V : Vocab d α) {A : Live₂} {x : Msg} (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
  (σ : Verity.Game.Strategy (V.game θ F c)) (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ)
  (Rd : Reading A) (Dg : Verity.Exchange → ℕ → Option Msg) (i : ℕ)

noncomputable def clsOf (s : TapeOf V θ F c σ × Coins A x) : TapeOf V θ F c σ × List (Msg × Msg) :=
  (s.1, histAt V θ F c σ place Rd Dg i s.1 s.2)

noncomputable def resOf (s : TapeOf V θ F c σ × Coins A x) (y : Coins A x) : TapeOf V θ F c σ × Coins A x :=
  (s.1, merge (histAt V θ F c σ place Rd Dg i s.1 s.2) y)

noncomputable def openAt (K : TapeOf V θ F c σ × List (Msg × Msg)) (s : TapeOf V θ F c σ × Coins A x) :
    Option Msg :=
  opening (outOf V θ F c σ place) Rd Dg i K.1 K.2 s.2

noncomputable def modeAt (K : TapeOf V θ F c σ × List (Msg × Msg)) : Msg :=
  modeOn (Shows (A := A) (x := x) K.2) (opening (outOf V θ F c σ place) Rd Dg i K.1 K.2)

open Classical in
theorem classOf_clsOf (hS : RunShown A x) (s : TapeOf V θ F c σ × Coins A x) :
    classOf (clsOf V θ F c σ place Rd Dg i) s =
      (Finset.univ.filter fun r' => Shows (histAt V θ F c σ place Rd Dg i s.1 s.2) r').map
        ⟨fun r' => (s.1, r'), fun a b h => (Prod.mk.inj h).2⟩ := by
  ext ⟨ω', r'⟩
  rw [Finset.mem_map]
  unfold classOf
  simp only [Finset.mem_filter]
  constructor
  · rintro ⟨-, h⟩
    have h1 : ω' = s.1 := congrArg Prod.fst h
    have h2 : histAt V θ F c σ place Rd Dg i ω' r' = histAt V θ F c σ place Rd Dg i s.1 s.2 := congrArg Prod.snd h
    subst h1
    exact ⟨r', ⟨Finset.mem_univ _, (hist_eq_iff hS (prover V θ F c σ place Rd Dg) s.1 s.2 r' i).1 h2⟩, rfl⟩
  · rintro ⟨a, ha, he⟩
    have he' : (s.1, a) = (ω', r') := he
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj he'
    exact ⟨Finset.mem_univ _, Prod.ext rfl
      ((hist_eq_iff hS (prover V θ F c σ place Rd Dg) s.1 s.2 _ i).2 ha.2)⟩

end Round

theorem missBound (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) : MissBound V A x θ F c σ := by
  classical
  intro place Rd Acc Good Dg i q hO hD hS
  have hunif : ∀ (s : TapeOf V θ F c σ × Coins A x) (φ : TapeOf V θ F c σ × Coins A x → ℝ),
      (∑ y, φ (resOf V θ F c σ place Rd Dg i s y)) * (classOf (clsOf V θ F c σ place Rd Dg i) s).card =
        Fintype.card (Coins A x) * ∑ x' ∈ classOf (clsOf V θ F c σ place Rd Dg i) s, φ x' := by
    intro s φ
    rw [classOf_clsOf V θ F c σ place Rd Dg i hS, Finset.sum_map, Finset.card_map]
    exact sum_merge (histAt V θ F c σ place Rd Dg i s.1 s.2) fun r' => φ (s.1, r')
  have hmode : ∀ (s : TapeOf V θ F c σ × Coins A x) m,
      ((classOf (clsOf V θ F c σ place Rd Dg i) s).filter fun x' =>
          openAt V θ F c σ place Rd Dg i (clsOf V θ F c σ place Rd Dg i s) x' = some m).card ≤
        ((classOf (clsOf V θ F c σ place Rd Dg i) s).filter fun x' =>
          openAt V θ F c σ place Rd Dg i (clsOf V θ F c σ place Rd Dg i s) x' =
            some (modeAt V θ F c σ place Rd Dg i (clsOf V θ F c σ place Rd Dg i s))).card := by
    intro s m
    rw [classOf_clsOf V θ F c σ place Rd Dg i hS, Finset.filter_map, Finset.filter_map, Finset.card_map,
      Finset.card_map, Finset.filter_filter, Finset.filter_filter]
    exact modeOn_spec _ _ m
  have hmiss : ∀ s, MissAt (frame V A x θ F c σ place Rd Dg) Rd Acc Good i s →
      bad (clsOf V θ F c σ place Rd Dg i) (openAt V θ F c σ place Rd Dg i) (modeAt V θ F c σ place Rd Dg i) s s := by
    rintro ⟨ω, r⟩ hs
    dsimp only [MissAt] at hs
    obtain ⟨hA, hG, hlen, hne, hagree⟩ := hs
    have hk : i < A.L.V.k x := lt_of_lt_of_le hlen (Claims.Recursion.run_length_le _ _ _ _ _ _)
    obtain ⟨m, hm, hdg⟩ := hO _ hA hG i hk
    have hKlen : (histAt V θ F c σ place Rd Dg i ω r).length = i := runHist_length _ hlen.le
    have hmsg : msgAt (Claims.run FlockDeploy.sha512 (frame V A x θ F c σ place Rd Dg).P A.L.V x ω r) i =
        some (modePick (outOf V θ F c σ place) Rd Dg ω (histAt V θ F c σ place Rd Dg i ω r)) :=
      run_msgAt (modePick (outOf V θ F c σ place) Rd Dg) A.L.V x ω r i hlen
    have hplay : playHist A x r (playMsgs Rd ((frame V A x θ F c σ place Rd Dg).out ω r)) i =
        histAt V θ F c σ place Rd Dg i ω r := by
      refine playHist_eq A x (frame V A x θ F c σ place Rd Dg).P ω r _ i hlen.le fun l hl => ?_
      obtain ⟨ml, hml, -⟩ := hO _ hA hG l (lt_trans hl hk)
      rw [← hagree l hl, hml]
      simp [playMsgs, hml]
    have hdig : Dg ((frame V A x θ F c σ place Rd Dg).out ω (complete (histAt V θ F c σ place Rd Dg i ω r))) i =
        some (Verity.sha512Msg m) := by
      rw [← hdg]
      refine hD ω r _ hA hG ?_
      rw [hplay]
      exact shows_merge _ _
    have hv : modePick (outOf V θ F c σ place) Rd Dg ω (histAt V θ F c σ place Rd Dg i ω r) =
        modeAt V θ F c σ place Rd Dg i (clsOf V θ F c σ place Rd Dg i (ω, r)) := by
      unfold modePick
      rw [hKlen]
      rfl
    unfold bad
    refine Exists.intro m (And.intro (opening_of hm hdig) fun hmv => hne ?_)
    rw [hm, hmsg, hmv, hv]
  have key := miss_le_sqrt (clsOf V θ F c σ place Rd Dg i) (resOf V θ F c σ place Rd Dg i)
    (openAt V θ F c σ place Rd Dg i) (modeAt V θ F c σ place Rd Dg i)
    (MissAt (frame V A x θ F c σ place Rd Dg) Rd Acc Good i) hunif hmode hmiss
  have hfin : (Claims.Pr fun s : (TapeOf V θ F c σ × Coins A x) × Coins A x =>
      fork (clsOf V θ F c σ place Rd Dg i) (openAt V θ F c σ place Rd Dg i) s.1
        (resOf V θ F c σ place Rd Dg i s.1 s.2)) ≤
      (msgFinder V θ F c σ place Rd Dg i q).Pr Verity.sha512Msg := by
    unfold Claims.Finder.Pr
    refine Claims.Pr_mono fun s hs => ?_
    obtain ⟨u, w, hu, hw, huw⟩ := hs
    have hu' : opening (outOf V θ F c σ place) Rd Dg i s.1.1 (histAt V θ F c σ place Rd Dg i s.1.1 s.1.2) s.1.2 =
        some u := hu
    have hw' : opening (outOf V θ F c σ place) Rd Dg i s.1.1 (histAt V θ F c σ place Rd Dg i s.1.1 s.1.2)
        (merge (histAt V θ F c σ place Rd Dg i s.1.1 s.1.2) s.2) = some w := hw
    refine Exists.intro u (Exists.intro w (And.intro ?_ (And.intro huw ?_)))
    · show (msgFinder V θ F c σ place Rd Dg i q).out s = some (u, w)
      simp [msgFinder, hu', hw', huw]
    · have h1 := (opening_some hu').2
      have h2 := (opening_some hw').2
      rw [h1] at h2
      exact Option.some.inj h2
  exact le_trans (ENNReal.ofReal_le_ofReal key) (ENNReal.ofReal_le_ofReal (Real.sqrt_le_sqrt hfin))

def MessagesBound (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) : Prop :=
  ∀ (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ) (Rd : Reading A) (Acc Good : Verity.Exchange → Prop)
    (Dg : Verity.Exchange → ℕ → Option Msg) (q : ℕ),
    MsgOpens A x Rd Acc Good Dg → (∀ i < A.L.V.k x, DigestBefore (frame V A x θ F c σ place Rd Dg) Rd Acc Good Dg i) →
      RunShown A x → RunsThrough (frame V A x θ F c σ place Rd Dg) Rd Acc Good →
      MessagesFixed V A x θ F c σ (frame V A x θ F c σ place Rd Dg) Rd Acc Good
        (∑ i ∈ Finset.range (A.L.V.k x),
          ENNReal.ofReal (Real.sqrt ((msgFinder V θ F c σ place Rd Dg i q).Pr Verity.sha512Msg)))

theorem messagesBound_of (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (hM : MissBound V A x θ F c σ) :
    MessagesBound V A x θ F c σ := by
  intro place Rd Acc Good Dg q hO hB hS hT
  exact messagesFixed_of_reached V A x θ F c σ _ Rd Acc Good _ hT
    (reachedOfRounds V A x θ F c σ _ Rd Acc Good _ fun i hi => hM place Rd Acc Good Dg i q hO (hB i hi) hS)

theorem messagesBound (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) : MessagesBound V A x θ F c σ :=
  messagesBound_of V A x θ F c σ (missBound V A x θ F c σ)

end Verity.SampledTwo.Messages
