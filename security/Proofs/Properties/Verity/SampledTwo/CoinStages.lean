module

public import Proofs.Properties.Verity.SampledTwo.CoinMap
public import Proofs.Properties.Verity.SampledTwo.CoinMapLaw
public import Proofs.Properties.Verity.SampledDraw
public import Proofs.Flock.Soundness.Discharge.ZkView.Decode

/-!
# The two-stage audit's coins from their families

The coin map's locator for the two-stage audit (`stageLocate`): the first stage's family's fresh coins continue the
first stage's coin, and the second stage's family's continue the second stage's coins in order, 16384 fresh coins
(65536 elements) a coin. At it, the coin `dev` reads for each stage starts with the bytes the log shows released in
the stage's family after its root:

* `devBits_first`, `devBits_second`: the bits the locator collects for a stage are the family's released coins
  (`Verity.Sampled.freshCoins`), all of them for the first stage and the stage's slice of them for the second, where
  the log shows the family's coins after one root only (`OneRoot`);
* `coinByte_testBit`, `src_testBit`: bit `q` of a coin's byte `k` (`SpotFlock.src`) is bit `8 (k % 16) + q` of its
  element `k / 16` (`F128.toBytes`' order), which is bit `8 k + q` of the bits `dev` collected;
* `bitsOf_getD`: bit `8 k + q` of a byte string's bits (`Properties.Challenger.bitsOf`) is bit `q` of its byte `k`;
* `devFirst_stage`, `devSecond_stage`: where a stage's sample names the locator's family and the log shows that family's
  coins after the sample's root only (`StageDraws`), bytes whose bits start the family's released coins start the coin
  `dev` reads for the stage, coin by coin for the second.

`StageDraws` is a hypothesis: it is what the run's coin check gives at the key change, where the family is the
verifier's (`Spot.Run.coinsCode`'s `family`) and a family whose log shows a root other than the registration's digest is
refused (`Challenger.FAMILY_ROOT`, `Properties.Challenger.FamilyOneCommitmentRead`'s `∀ r ∈ s.roots label, r = after`).
-/

@[expose] public section

namespace Verity.SampledTwo.CoinMap

open Verity.Game
open Claims

/-! ## The locator -/

/-- Whether a release is a fresh coin of family `l`. -/
def isFresh (l : String) (rl : Verity.Release) : Bool :=
  match rl.frame with
  | .fresh l' => l' == l
  | _ => false

/-- **The two-stage audit's locator** at the first stage's family `ℓ₁` and the second stage's `ℓ₂`: a fresh coin of
`ℓ₁` continues stage 0, and a fresh coin of `ℓ₂` stage `1 + n / 16384`, where `n` of `ℓ₂`'s fresh coins came before
it; nothing else is located. -/
def stageLocate (ℓ₁ ℓ₂ : String) : Locate := fun pre fr _ =>
  match fr with
  | .fresh l => if l = ℓ₁ then some 0 else if l = ℓ₂ then some (1 + pre.countP (isFresh ℓ₂) / 16384) else none
  | _ => none

/-- **The log shows family `ℓ`'s coins after one root**: every fresh coin of `ℓ` the log holds answers `after`. -/
def OneRoot (ℓ : String) (after : ByteArray) (log : List Verity.Release) : Prop :=
  ∀ rl ∈ log, rl.frame = .fresh ℓ → rl.answers = after

/-- **Every fresh coin the log holds carries its 512 bits**, as every play of the device's game releases them
(`Verity.releases`). -/
def FreshWide (log : List Verity.Release) : Prop :=
  ∀ rl ∈ log, ∀ l, rl.frame = .fresh l → rl.coins.length = 512

/-- **A stage's sample is the locator's family after one root** (owner TWOLEG, from
`Properties.Challenger.FamilyOneCommitmentRead` once the base holds the key change): on a play `Acc` holds of, where
the stage's sample (`draw`, the case's family and digest) is `(label, after)`, `label` is the locator's `ℓ` and the log
shows `ℓ`'s coins after `after` only. -/
def StageDraws (Acc : Verity.Exchange → Prop) (draw : Verity.Exchange → Option (String × ByteArray)) (ℓ : String) :
    Prop :=
  ∀ o, Acc o → ∀ label after, draw o = some (label, after) → label = ℓ ∧ OneRoot ℓ after o.log

theorem locAt_fresh {ℓ₁ ℓ₂ : String} {pre : List Verity.Release} {fr : Verity.Frame} {b : ByteArray} {i : ℕ}
    (h : locAt (stageLocate ℓ₁ ℓ₂) pre fr b = some i) : ∃ l, fr = .fresh l := by
  cases fr with
  | fresh l => exact ⟨l, rfl⟩
  | hello _ => exact absurd h (by simp [locAt])
  | opening _ _ _ => exact absurd h (by simp [locAt, stageLocate])

/-- **Where the locator puts no element on a stage, it collects no bit for it.** -/
theorem devBits_nil_of_used (ℓ₁ ℓ₂ : String) (i : ℕ) :
    ∀ (log pre : List Verity.Release), usedOf (stageLocate ℓ₁ ℓ₂) i pre log = 0 →
      devBits (stageLocate ℓ₁ ℓ₂) i pre log = []
  | [], _, _ => rfl
  | rl :: log, pre, h => by
    simp only [usedOf] at h
    simp only [devBits]
    rw [devBits_nil_of_used ℓ₁ ℓ₂ i log (pre ++ [rl]) (by omega), List.append_nil]
    by_cases hloc : locAt (stageLocate ℓ₁ ℓ₂) pre rl.frame rl.answers = some i
    · rw [if_pos hloc] at h
      obtain ⟨l, hl⟩ := locAt_fresh hloc
      rw [hl] at h
      simp [widthOf] at h
    · rw [if_neg hloc]

/-! ## Every play's fresh coins carry their 512 bits -/

/-- **Every leaf of `g` satisfies `P`**, whatever the prover sends and the coins (`FlockSoundness.Game.Leaves`'s
shape, which this module doesn't import). -/
def Leaves {β : Type} (P : β → Prop) : Game β → Prop
  | .ret a => P a
  | .send _ k => ∀ m, Leaves P (k m)
  | @Game.coin _ _ _ _ k => ∀ c, Leaves P (k c)

theorem Leaves.reach {β : Type} {P : β → Prop} :
    ∀ (g : Game β) (s : Strategy g) {o : β}, Leaves P g → PlayReaches g s o → P o
  | .ret a, _, o, h, ho => by
    have e : o = a := ho
    rw [e]
    exact h
  | .send _ k, s, _, h, ho => Leaves.reach (k s.1) s.2 (h s.1) ho
  | @Game.coin _ _ _ _ k, s, _, h, ho => by
    obtain ⟨c, hc⟩ := ho
    exact Leaves.reach (k c) (s c) (h c) hc

theorem FreshWide.cons {rl : Verity.Release} {log : List Verity.Release}
    (hr : ∀ l, rl.frame = .fresh l → rl.coins.length = 512) (h : FreshWide log) : FreshWide (rl :: log) := by
  intro r hr' l hl
  rcases List.mem_cons.1 hr' with rfl | hm
  · exact hr l hl
  · exact h r hm l hl

theorem leaves_optGame {β γ : Type} {P : β → Prop} {n : Game β} {y : γ → Game β} (hn : Leaves P n)
    (hy : ∀ x, Leaves P (y x)) : ∀ o, Leaves P (optGame o n y)
  | none => hn
  | some x => hy x

/-- **The device's releases release each fresh coin's 512 bits**: where `f` at every such log has `P` at every leaf,
so do the releases then `f`. -/
theorem leaves_relK {β : Type} (P : β → Prop) :
    ∀ (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree) (f : List Verity.Release → Game β),
      (∀ log, FreshWide log → Leaves P (f log)) → Leaves P (relK k fs ts f)
  | 0, _, _, f, hf => hf [] (fun _ h => by cases h)
  | k + 1, fs, ts, f, hf => by
    have h0 : Leaves P (f []) := hf [] (fun _ h => by cases h)
    intro m
    rcases m with _ | ⟨fr, b⟩
    · exact h0
    · cases fr with
      | fresh l =>
        refine leaves_optGame h0 (fun fs' => ?_) _
        intro x
        exact leaves_relK P k fs' ts _ fun log hl =>
          hf _ (FreshWide.cons (fun _ _ => by dsimp only; exact List.length_ofFn) hl)
      | hello spec =>
        show Leaves P (if _ then _ else _)
        split
        · intro t
          exact leaves_relK P k fs _ _ fun log hl => hf _ (FreshWide.cons (fun _ h => by cases h) hl)
        · exact h0
      | opening i sn n =>
        show Leaves P (if _ then _ else _)
        split
        · refine leaves_optGame h0 (fun T => ?_) _
          refine leaves_optGame h0 (fun p => ?_) _
          intro d
          exact leaves_relK P k fs _ _ fun log hl => hf _ (FreshWide.cons (fun _ h => by cases h) hl)
        · exact h0

/-- **Every play of the period's game releases each fresh coin's 512 bits.** -/
theorem freshWide_play {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α] (V : Vocab d α) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Strategy (V.game θ F c)) {o : Verity.Exchange}
    (h : PlayReaches (V.game θ F c) σ o) : FreshWide o.log := by
  have hl : Leaves (fun o : Verity.Exchange => FreshWide o.log) (V.game θ F c) := by
    rw [game_eq]
    intro a
    refine leaves_relK _ _ _ _ _ fun log hl => ?_
    intro _ _ _
    exact hl
  exact Leaves.reach _ σ hl h

/-! ## The bits collected for each stage -/

/-- **The first stage's bits are its family's released coins.** -/
theorem devBits_first (ℓ₁ ℓ₂ : String) (after : ByteArray) :
    ∀ (log pre : List Verity.Release), OneRoot ℓ₁ after log →
      devBits (stageLocate ℓ₁ ℓ₂) 0 pre log = (Verity.Sampled.freshCoins ℓ₁ after log).flatten
  | [], _, _ => by simp [devBits, Verity.Sampled.freshCoins]
  | ⟨fr, ans, cs⟩ :: log, pre, h => by
    have ih := devBits_first ℓ₁ ℓ₂ after log (pre ++ [⟨fr, ans, cs⟩])
      (fun r hr => h r (List.mem_cons_of_mem _ hr))
    simp only [devBits]
    rw [ih]
    unfold Verity.Sampled.freshCoins
    rw [List.filterMap_cons]
    cases fr with
    | fresh l =>
      by_cases hl : l = ℓ₁
      · subst hl
        have ha : ans = after := h _ (List.mem_cons.2 (Or.inl rfl)) rfl
        subst ha
        simp [locAt, stageLocate]
      · by_cases h2 : l = ℓ₂
        · subst h2
          simp [locAt, stageLocate, hl]
        · simp [locAt, stageLocate, hl, h2]
    | hello _ => simp [locAt]
    | opening _ _ _ => simp [locAt, stageLocate]

/-- **Stage `j + 1`'s bits are its slice of the second stage's family's released coins**: those numbered `16384 j`
to `16384 (j + 1)` among the family's, counting the `n₀` the releases before had. -/
theorem devBits_second (ℓ₁ ℓ₂ : String) (hne : ℓ₁ ≠ ℓ₂) (after : ByteArray) (j : ℕ) :
    ∀ (log pre : List Verity.Release), OneRoot ℓ₂ after log →
      devBits (stageLocate ℓ₁ ℓ₂) (j + 1) pre log =
        (((Verity.Sampled.freshCoins ℓ₂ after log).take (16384 * (j + 1) - pre.countP (isFresh ℓ₂))).drop
          (16384 * j - pre.countP (isFresh ℓ₂))).flatten
  | [], _, _ => by simp [devBits, Verity.Sampled.freshCoins]
  | ⟨fr, ans, cs⟩ :: log, pre, h => by
    have ih := devBits_second ℓ₁ ℓ₂ hne after j log (pre ++ [⟨fr, ans, cs⟩])
      (fun r hr => h r (List.mem_cons_of_mem _ hr))
    simp only [devBits]
    rw [ih, List.countP_append]
    unfold Verity.Sampled.freshCoins
    rw [List.filterMap_cons]
    cases fr with
    | fresh l =>
      by_cases h2 : l = ℓ₂
      · subst h2
        have ha : ans = after := h _ (List.mem_cons.2 (Or.inl rfl)) rfl
        subst ha
        have hc : List.countP (isFresh l) [⟨.fresh l, ans, cs⟩] = 1 := by simp [isFresh]
        rw [hc]
        simp only [locAt, stageLocate, Ne.symm hne, and_self, ↓reduceIte]
        generalize pre.countP (isFresh l) = n
        by_cases hA : n < 16384 * j
        · have hj : ¬ (1 + n / 16384 = j + 1) := by omega
          rw [if_neg (by simpa using hj), List.nil_append,
            show 16384 * (j + 1) - n = (16384 * (j + 1) - (n + 1)) + 1 by omega, List.take_succ_cons,
            show 16384 * j - n = (16384 * j - (n + 1)) + 1 by omega, List.drop_succ_cons]
        · by_cases hB : n < 16384 * (j + 1)
          · have hj : 1 + n / 16384 = j + 1 := by omega
            rw [if_pos (by simp [hj]), show 16384 * (j + 1) - n = (16384 * (j + 1) - (n + 1)) + 1 by omega,
              List.take_succ_cons, show 16384 * j - n = 0 by omega, show 16384 * j - (n + 1) = 0 by omega,
              List.drop_zero, List.drop_zero, List.flatten_cons]
          · have hj : ¬ (1 + n / 16384 = j + 1) := by omega
            rw [if_neg (by simpa using hj), List.nil_append, show 16384 * (j + 1) - n = 0 by omega,
              show 16384 * (j + 1) - (n + 1) = 0 by omega]
            simp
      · have hc : List.countP (isFresh ℓ₂) [⟨.fresh l, ans, cs⟩] = 0 := by simp [isFresh, h2]
        rw [hc, Nat.add_zero]
        by_cases h1 : l = ℓ₁
        · subst h1
          simp [locAt, stageLocate, h2]
        · simp [locAt, stageLocate, h1, h2]
    | hello sp =>
      have hc : List.countP (isFresh ℓ₂) [⟨.hello sp, ans, cs⟩] = 0 := by simp [isFresh]
      rw [hc, Nat.add_zero]
      simp [locAt]
    | opening a s n =>
      have hc : List.countP (isFresh ℓ₂) [⟨.opening a s n, ans, cs⟩] = 0 := by simp [isFresh]
      rw [hc, Nat.add_zero]
      simp [locAt, stageLocate]

/-! ## Lists of chunks of one length -/

theorem length_flatten_const {β : Type} {m : ℕ} : ∀ (cs : List (List β)), (∀ c ∈ cs, c.length = m) →
    cs.flatten.length = m * cs.length
  | [], _ => by simp
  | c :: cs, h => by
    rw [List.flatten_cons, List.length_append, h c (by simp),
      length_flatten_const cs (fun d hd => h d (List.mem_cons_of_mem _ hd)), List.length_cons, Nat.mul_succ]
    omega

theorem flatten_drop_const {β : Type} {m : ℕ} : ∀ (cs : List (List β)), (∀ c ∈ cs, c.length = m) → ∀ A,
    (cs.drop A).flatten = cs.flatten.drop (m * A)
  | [], _, A => by simp
  | _, _, 0 => by simp
  | c :: cs, h, A + 1 => by
    rw [List.drop_succ_cons, flatten_drop_const cs (fun d hd => h d (List.mem_cons_of_mem _ hd)) A,
      List.flatten_cons, show m * (A + 1) = c.length + m * A by rw [h c (by simp)]; ring, List.drop_append,
      List.drop_of_length_le (Nat.le_add_right _ _), List.nil_append, Nat.add_sub_cancel_left]

theorem getD_prefix {β : Type} {l₁ l₂ : List β} (h : l₁ <+: l₂) {t : ℕ} (ht : t < l₁.length) (d : β) :
    l₂.getD t d = l₁.getD t d := by
  obtain ⟨r, rfl⟩ := h
  exact List.getD_append _ _ _ _ ht

/-- **The bits of a slice of chunks of one length, from the bits of all.** -/
theorem getD_slice {β : Type} {m : ℕ} (cs : List (List β)) (h : ∀ c ∈ cs, c.length = m) (A n : ℕ) {t : ℕ}
    (ht : t < m * n) (d : β) : (((cs.take (A + n)).drop A).flatten).getD t d = cs.flatten.getD (m * A + t) d := by
  have hd : ∀ c ∈ cs.drop A, c.length = m := fun c hc => h c (List.mem_of_mem_drop hc)
  have e : cs.flatten.getD (m * A + t) d = (cs.drop A).flatten.getD t d := by
    rw [flatten_drop_const cs h A, List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_drop]
  rw [List.drop_take, Nat.add_sub_cancel_left, e]
  by_cases hn : n ≤ (cs.drop A).length
  · have hp : ((cs.drop A).take n).flatten <+: (cs.drop A).flatten :=
      ⟨((cs.drop A).drop n).flatten, by rw [← List.flatten_append, List.take_append_drop]⟩
    refine (getD_prefix hp (t := t) ?_ d).symm
    rw [length_flatten_const _ (fun c hc => hd c (List.mem_of_mem_take hc)), List.length_take, Nat.min_eq_left hn]
    exact ht
  · rw [List.take_of_length_le (by omega)]

/-! ## Bytes and bits -/

theorem toList_loop' (b : ByteArray) (i : Nat) (r : List UInt8) :
    ByteArray.toList.loop b i r = r.reverse ++ b.data.toList.drop i := by
  fun_induction ByteArray.toList.loop b i r with
  | case1 i r h ih =>
    rw [ih, List.reverse_cons, List.append_assoc, List.singleton_append]
    congr 1
    conv_rhs => rw [List.drop_eq_getElem_cons (by simpa using h)]
    simp [ByteArray.get!, h]
  | case2 i r h =>
    rw [List.drop_eq_nil_of_le (by simp; simpa using h), List.append_nil]

theorem toList_eq' (b : ByteArray) : b.toList = b.data.toList := by
  unfold ByteArray.toList
  rw [toList_loop']
  simp

theorem getD_flatMap_const {α β : Type} (f : α → List β) {m : ℕ} (d : β) :
    ∀ (l : List α), (∀ a ∈ l, (f a).length = m) → ∀ {g : ℕ} {a : α}, l[g]? = some a → ∀ {i : ℕ}, i < m →
      (l.flatMap f).getD (m * g + i) d = (f a).getD i d
  | [], _, _, _, h, _, _ => by simp at h
  | b :: l, hf, 0, a, h, i, hi => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h
    rw [List.flatMap_cons, List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, Nat.mul_zero, Nat.zero_add,
      List.getElem?_append_left (by rw [hf b (by simp)]; exact hi)]
  | b :: l, hf, g + 1, a, h, i, hi => by
    simp only [List.getElem?_cons_succ] at h
    have hb := hf b (by simp)
    have hm : m * (g + 1) + i - m = m * g + i := by rw [Nat.mul_succ]; omega
    rw [List.flatMap_cons, List.getD_eq_getElem?_getD,
      List.getElem?_append_right (by rw [hb, Nat.mul_succ]; omega), hb, hm, ← List.getD_eq_getElem?_getD]
    exact getD_flatMap_const f d l (fun c hc => hf c (List.mem_cons_of_mem b hc)) h hi

theorem length_flatMap_const {α β : Type} (f : α → List β) {m : ℕ} (hf : ∀ a, (f a).length = m) :
    ∀ l : List α, (l.flatMap f).length = m * l.length
  | [] => by simp
  | a :: l => by
    rw [List.flatMap_cons, List.length_append, hf a, length_flatMap_const f hf l, List.length_cons, Nat.mul_succ]
    omega

theorem length_bitsOf (b : ByteArray) : (Properties.Challenger.bitsOf b).length = 8 * b.size := by
  unfold Properties.Challenger.bitsOf
  rw [toList_eq', length_flatMap_const _ (m := 8) (fun a => by simp)]
  simp

/-- **Bit `8 k + q` of a byte string's bits is bit `q` of its byte `k`.** -/
theorem bitsOf_getD (b : ByteArray) {k q : ℕ} (hk : k < b.size) (hq : q < 8) :
    (Properties.Challenger.bitsOf b).getD (8 * k + q) false = b[k].toNat.testBit q := by
  unfold Properties.Challenger.bitsOf
  rw [toList_eq', getD_flatMap_const (fun x : UInt8 => (List.range 8).map fun i => x.toNat.testBit i) (m := 8) false
    b.data.toList (fun a _ => by simp) (g := k) (a := b[k])
    (List.getElem?_eq_some_iff.2 ⟨by simpa using hk, by simp [ByteArray.getElem_eq_getElem_data]⟩) hq,
    List.getD_eq_getElem ((List.range 8).map fun i => b[k].toNat.testBit i) false (by simpa using hq),
    List.getElem_map, List.getElem_range]

theorem testBit_toUInt8 (x : UInt64) {i : ℕ} (hi : i < 8) : x.toUInt8.toNat.testBit i = x.toNat.testBit i := by
  have h : x.toUInt8.toNat = x.toNat % 2 ^ 8 := by simp [UInt64.toNat_toUInt8]
  rw [h, Nat.testBit_mod_two_pow]
  simp [hi]

theorem testBit_shift (x : UInt64) {j : ℕ} (hj : j < 8) (q : ℕ) :
    (x >>> (8 * j).toUInt64).toNat.testBit q = x.toNat.testBit (8 * j + q) := by
  rw [UInt64.toNat_shiftRight, Nat.toUInt64, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega),
    Nat.mod_eq_of_lt (by omega), Nat.testBit_shiftRight]

/-- **Bit `q` of an element's byte `m` is its bit `8 m + q`** (`F128.toBytes`: `le64 lo ‖ le64 hi`). -/
theorem toBytes_testBit (a : Flock.F128) {m q : ℕ} (hm : m < 16) (hq : q < 8) :
    (a.toBytes.data.toList.getD m 0).toNat.testBit q = a.bit (8 * m + q) := by
  have hd : a.toBytes.data.toList.getD m 0 =
      if m < 8 then (a.lo >>> (8 * m).toUInt64).toUInt8 else (a.hi >>> (8 * (m - 8)).toUInt64).toUInt8 := by
    rw [List.getD_eq_getElem a.toBytes.data.toList 0 (by simpa [Flock.F128.toBytes] using hm), Array.getElem_toList]
    exact Array.getElem_ofFn _
  rw [hd]
  unfold Flock.F128.bit
  by_cases h8 : m < 8
  · rw [if_pos h8, if_pos (show 8 * m + q < 64 by omega), Flock.F128.bitU64 _ _ (by omega), testBit_toUInt8 _ hq,
      testBit_shift _ h8]
  · rw [if_neg h8, if_neg (show ¬ 8 * m + q < 64 by omega), Flock.F128.bitU64 _ _ (by omega), testBit_toUInt8 _ hq,
      testBit_shift _ (show m - 8 < 8 by omega), show 8 * m + q - 64 = 8 * (m - 8) + q by omega]

/-- **Bit `q` of a coin's byte `k` is bit `8 (k % 16) + q` of its element `k / 16`.** -/
theorem coinByte_testBit (d : FlockDeploy.Coin) (k : Fin SpotFlock.L) {q : ℕ} (hq : q < 8) :
    (SpotFlock.coinByte d k).val.testBit q =
      (d ⟨k.val / 16, by have := k.isLt; simp only [SpotFlock.L] at this; omega⟩).bit (8 * (k.val % 16) + q) := by
  have hk := k.isLt
  simp only [SpotFlock.L] at hk
  have e : d.bytes.getD k.val 0 =
      (d ⟨k.val / 16, by omega⟩).toBytes.data.toList.getD (k.val % 16) 0 := by
    unfold FlockDeploy.Coin.bytes
    have := getD_flatMap_const (fun i => (d i).toBytes.data.toList) (0 : UInt8) (List.finRange 65536)
      (fun a _ => by simpa using FlockSoundness.Discharge.ZkView.toBytes_size (d a)) (g := k.val / 16)
      (a := ⟨k.val / 16, by omega⟩)
      (List.getElem?_eq_some_iff.2 ⟨by simpa using (by omega : k.val / 16 < 65536), Fin.ext (by simp)⟩)
      (i := k.val % 16) (Nat.mod_lt _ (by norm_num))
    rwa [show 16 * (k.val / 16) + k.val % 16 = k.val by omega] at this
  unfold SpotFlock.coinByte
  dsimp only
  rw [e, toBytes_testBit _ (Nat.mod_lt _ (by norm_num)) hq]

/-- **Bit `q` of byte `k` of the coin `dev` reads is bit `8 k + q` of the bits it collected.** -/
theorem src_testBit {L : Locate} {log : List Verity.Release} {i : ℕ} {d : FlockDeploy.Coin}
    (h : dev L log i = some d) (k : Fin SpotFlock.L) {q : ℕ} (hq : q < 8) :
    (SpotFlock.coinByte d k).val.testBit q = (devBits L i [] log).getD (8 * k.val + q) false := by
  rw [coinByte_testBit d k hq]
  unfold dev at h
  by_cases h0 : usedOf L i [] log = 0
  · rw [if_pos h0] at h
    cases h
  · rw [if_neg h0] at h
    obtain rfl := Option.some.inj h
    dsimp only
    rw [bit_elemOfBits _ (by omega)]
    congr 1
    omega

theorem byte_eq_of_testBit {x y : UInt8} (h : ∀ q < 8, x.toNat.testBit q = y.toNat.testBit q) : x = y := by
  apply UInt8.toNat_inj.mp
  apply Nat.eq_of_testBit_eq
  intro q
  by_cases hq : q < 8
  · exact h q hq
  · have hp : 2 ^ 8 ≤ 2 ^ q := Nat.pow_le_pow_right (by norm_num) (by omega)
    rw [Nat.testBit_eq_false_of_lt (lt_of_lt_of_le (UInt8.toNat_lt x) hp),
      Nat.testBit_eq_false_of_lt (lt_of_lt_of_le (UInt8.toNat_lt y) hp)]

theorem size_bytesOf {n : ℕ} (f : Fin n → Fin 256) : (Properties.Spot.bytesOf f).size = n := by
  simp [Properties.Spot.bytesOf, ByteArray.size]

/-- Byte `k` of a source's bytes is the source's `k`. -/
-- Stated at every length `n`: at `SpotFlock.L` the kernel would unroll `Array.ofFn`'s 1048576 steps to check the
-- index's bound.
theorem bytesOf_getElem {n : ℕ} (f : Fin n → Fin 256) {k : ℕ} (hn : k < n) (hk : k < (Properties.Spot.bytesOf f).size) :
    ((Properties.Spot.bytesOf f)[k]).toNat = (f ⟨k, hn⟩).val := by
  simp [Properties.Spot.bytesOf, ByteArray.getElem_eq_getElem_data]

theorem size_src (d : FlockDeploy.Coin) : (SpotFlock.src d).size = SpotFlock.L :=
  size_bytesOf _

theorem src_getElem (d : FlockDeploy.Coin) {k : ℕ} (hk : k < (SpotFlock.src d).size) :
    ((SpotFlock.src d)[k]).toNat = (SpotFlock.coinByte d ⟨k, by rwa [size_src] at hk⟩).val :=
  bytesOf_getElem (SpotFlock.coinByte d) _ hk

/-- **Bytes whose bits are the collected bits from `8 off` on, `w` of them, start the coin `dev` reads.** -/
theorem src_extract {L : Locate} {log : List Verity.Release} {i : ℕ} {d : FlockDeploy.Coin}
    (h : dev L log i = some d) (b : ByteArray) (off w : ℕ) (hw : w ≤ SpotFlock.L) (hb : off + w ≤ b.size)
    (hbits : ∀ k < w, ∀ q < 8, (devBits L i [] log).getD (8 * k + q) false =
      (Properties.Challenger.bitsOf b).getD (8 * (off + k) + q) false) :
    (SpotFlock.src d).extract 0 w = b.extract off (off + w) := by
  apply ByteArray.ext_getElem
  · rw [ByteArray.size_extract, ByteArray.size_extract, size_src]
    omega
  · intro k hk hk'
    rw [ByteArray.size_extract, size_src] at hk
    rw [ByteArray.getElem_extract, ByteArray.getElem_extract]
    apply byte_eq_of_testBit
    intro q hq
    simp only [Nat.zero_add]
    rw [src_getElem, src_testBit h _ hq, hbits k (by omega) q hq, bitsOf_getD b (by omega) hq]

/-! ## The stages' coins -/

/-- **The first stage's coin starts with the bytes its family's released coins start with**, where the log shows the
family's coins after one root only. -/
theorem devFirst_of (ℓ₁ ℓ₂ : String) {log : List Verity.Release} {after : ByteArray} (hone : OneRoot ℓ₁ after log)
    {b : ByteArray} (hb : Properties.Challenger.bitsOf b <+: (Verity.Sampled.freshCoins ℓ₁ after log).flatten)
    (h0 : 0 < b.size) (hL : b.size ≤ SpotFlock.L) :
    ∃ d, dev (stageLocate ℓ₁ ℓ₂) log 0 = some d ∧ (SpotFlock.src d).extract 0 b.size = b := by
  rw [← devBits_first ℓ₁ ℓ₂ after log [] hone] at hb
  have hu : usedOf (stageLocate ℓ₁ ℓ₂) 0 [] log ≠ 0 := fun hu => by
    rw [devBits_nil_of_used ℓ₁ ℓ₂ 0 log [] hu, List.prefix_nil] at hb
    have := length_bitsOf b
    rw [hb, List.length_nil] at this
    omega
  cases hd : dev (stageLocate ℓ₁ ℓ₂) log 0 with
  | none =>
    unfold dev at hd
    rw [if_neg hu] at hd
    cases hd
  | some d =>
    refine ⟨d, rfl, ?_⟩
    have e := src_extract hd b 0 b.size hL (by omega) fun k hk q hq => by
      rw [Nat.zero_add, getD_prefix hb (t := 8 * k + q) (by rw [length_bitsOf]; omega)]
    rwa [Nat.zero_add, ByteArray.extract_zero_size] at e

theorem freshCoins_length {ℓ : String} {after : ByteArray} {log : List Verity.Release} (hw : FreshWide log) :
    ∀ c ∈ Verity.Sampled.freshCoins ℓ after log, c.length = 512 := by
  intro c hc
  unfold Verity.Sampled.freshCoins at hc
  obtain ⟨rl, hrl, he⟩ := List.mem_filterMap.1 hc
  revert he
  cases hf : rl.frame with
  | fresh l =>
    intro he
    by_cases hc' : l = ℓ ∧ rl.answers = after
    · simp only [hc', and_self, if_true, Option.some.injEq] at he
      rw [← he]
      exact hw rl hrl l hf
    · simp [hc'] at he
  | hello _ => intro he; cases he
  | opening _ _ _ => intro he; cases he

/-- **Stage `j + 1`'s coin starts with the bytes of its slice of the second stage's bytes**, where the second stage's
family's released coins start with the second stage's bytes, the log shows them after one root only and each carries
its 512 bits. -/
theorem devSecond_of (ℓ₁ ℓ₂ : String) (hne : ℓ₁ ≠ ℓ₂) {log : List Verity.Release} {after : ByteArray}
    (hone : OneRoot ℓ₂ after log) (hw : FreshWide log) {b : ByteArray}
    (hb : Properties.Challenger.bitsOf b <+: (Verity.Sampled.freshCoins ℓ₂ after log).flatten) (j : ℕ)
    (hpos : 0 < stageWidth b j) :
    ∃ d, dev (stageLocate ℓ₁ ℓ₂) log (j + 1) = some d ∧
      (SpotFlock.src d).extract 0 (stageWidth b j) =
        b.extract (j * SpotFlock.L) (j * SpotFlock.L + stageWidth b j) := by
  have hcs := freshCoins_length (ℓ := ℓ₂) (after := after) hw
  have hdev := devBits_second ℓ₁ ℓ₂ hne after j log [] hone
  simp only [List.countP_nil, Nat.sub_zero] at hdev
  rw [show 16384 * (j + 1) = 16384 * j + 16384 by ring] at hdev
  generalize Verity.Sampled.freshCoins ℓ₂ after log = cs at hb hcs hdev
  have hwid : stageWidth b j ≤ SpotFlock.L ∧ j * SpotFlock.L + stageWidth b j ≤ b.size := by
    unfold stageWidth at hpos ⊢
    simp only [SpotFlock.L] at hpos ⊢
    omega
  have hlen := length_bitsOf b
  have hflat := length_flatten_const cs hcs
  have hpre := hb.length_le
  have hbits : ∀ k < stageWidth b j, ∀ q < 8, (devBits (stageLocate ℓ₁ ℓ₂) (j + 1) [] log).getD (8 * k + q) false =
      (Properties.Challenger.bitsOf b).getD (8 * (j * SpotFlock.L + k) + q) false := by
    intro k hk q hq
    rw [hdev, getD_slice cs hcs (16384 * j) 16384 (t := 8 * k + q) (by simp only [SpotFlock.L] at hwid; omega),
      getD_prefix hb (t := 512 * (16384 * j) + (8 * k + q)) (by simp only [SpotFlock.L] at hwid; omega)]
    congr 1
    simp only [SpotFlock.L]
    ring
  have hu : usedOf (stageLocate ℓ₁ ℓ₂) (j + 1) [] log ≠ 0 := fun hu => by
    have hlt : 16384 * j < cs.length := by simp only [SpotFlock.L] at hwid; omega
    have hne' : (((cs.take (16384 * j + 16384)).drop (16384 * j)).flatten) ≠ [] := by
      intro he
      have := length_flatten_const ((cs.take (16384 * j + 16384)).drop (16384 * j))
        (fun c hc => hcs c (List.mem_of_mem_take (List.mem_of_mem_drop hc)))
      rw [he, List.length_nil, List.length_drop, List.length_take] at this
      omega
    rw [← hdev, devBits_nil_of_used ℓ₁ ℓ₂ (j + 1) log [] hu] at hne'
    exact hne' rfl
  cases hd : dev (stageLocate ℓ₁ ℓ₂) log (j + 1) with
  | none =>
    unfold dev at hd
    rw [if_neg hu] at hd
    cases hd
  | some d => exact ⟨d, rfl, src_extract hd b _ _ hwid.1 hwid.2 hbits⟩

/-! ## How many elements each stage's coin reads -/

/-- **Where every release a round takes has 4 elements, the elements that fit are all, up to the coin's 65536.** -/
theorem fitOf_ge (L : Locate) (i : ℕ) (hW : ∀ pre fr b, locAt L pre fr b = some i → widthOf fr = 4) :
    ∀ (log pre : List Verity.Release), 4 ∣ usedOf L i [] pre →
      min (usedOf L i pre log) (65536 - usedOf L i [] pre) ≤ fitOf L i pre log
  | [], _, _ => by simp [usedOf, fitOf]
  | rl :: log, pre, hd => by
    have hs := usedOf_snoc L i pre rl
    simp only [usedOf, fitOf]
    by_cases hl : locAt L pre rl.frame rl.answers = some i
    · have hw := hW _ _ _ hl
      rw [if_pos hl, hw] at hs
      have ih := fitOf_ge L i hW log (pre ++ [rl]) (by rw [hs]; omega)
      rw [hs] at ih
      rw [if_pos hl, hw]
      by_cases hf : usedOf L i [] pre + 4 ≤ 65536
      · rw [if_pos ⟨hl, hf⟩]
        omega
      · rw [if_neg (fun h => hf h.2)]
        omega
    · rw [if_neg hl, Nat.add_zero] at hs
      have ih := fitOf_ge L i hW log (pre ++ [rl]) (by rw [hs]; exact hd)
      rw [hs] at ih
      rw [if_neg hl, if_neg (fun h => hl h.1)]
      omega

theorem widthOf_stage (ℓ₁ ℓ₂ : String) (i : ℕ) :
    ∀ pre fr b, locAt (stageLocate ℓ₁ ℓ₂) pre fr b = some i → widthOf fr = 4 := by
  intro pre fr b h
  obtain ⟨l, rfl⟩ := locAt_fresh h
  rfl

/-- **The bits a stage collects are 128 a element**, where each fresh coin carries its 512 bits. -/
theorem length_devBits (ℓ₁ ℓ₂ : String) (i : ℕ) :
    ∀ (log : List Verity.Release), FreshWide log → ∀ pre,
      (devBits (stageLocate ℓ₁ ℓ₂) i pre log).length = 128 * usedOf (stageLocate ℓ₁ ℓ₂) i pre log
  | [], _, _ => by simp [devBits, usedOf]
  | rl :: log, hw, pre => by
    simp only [devBits, usedOf, List.length_append]
    rw [length_devBits ℓ₁ ℓ₂ i log (fun r hr => hw r (List.mem_cons_of_mem _ hr)) (pre ++ [rl])]
    by_cases hl : locAt (stageLocate ℓ₁ ℓ₂) pre rl.frame rl.answers = some i
    · obtain ⟨l, hf⟩ := locAt_fresh hl
      rw [if_pos hl, if_pos hl, hw rl List.mem_cons_self l hf, hf]
      simp only [widthOf]
      omega
    · rw [if_neg hl, if_neg hl]
      simp

/-- **The first stage's bytes fit in what its coin reads**: bytes whose bits start the first stage's family's
released coins, at most a coin's, are at most 16 bytes an element that fits. -/
theorem size_le_fit_first (ℓ₁ ℓ₂ : String) {log : List Verity.Release} {after : ByteArray}
    (hone : OneRoot ℓ₁ after log) (hw : FreshWide log) {b : ByteArray}
    (hb : Properties.Challenger.bitsOf b <+: (Verity.Sampled.freshCoins ℓ₁ after log).flatten)
    (hL : b.size ≤ SpotFlock.L) : b.size ≤ 16 * fitOf (stageLocate ℓ₁ ℓ₂) 0 [] log := by
  have hf := fitOf_ge (stageLocate ℓ₁ ℓ₂) 0 (widthOf_stage ℓ₁ ℓ₂ 0) log [] (by simp [usedOf])
  have hl := length_devBits ℓ₁ ℓ₂ 0 log hw []
  rw [devBits_first ℓ₁ ℓ₂ after log [] hone] at hl
  have hpre := hb.length_le
  have hlen := length_bitsOf b
  simp only [usedOf, Nat.sub_zero] at hf
  simp only [SpotFlock.L] at hL
  omega

/-- **Each second stage's slice fits in what its coin reads**: stage `j + 1`'s bytes (`stageWidth`) are at most 16
bytes an element that fits. -/
theorem stageWidth_le_fit_second (ℓ₁ ℓ₂ : String) (hne : ℓ₁ ≠ ℓ₂) {log : List Verity.Release} {after : ByteArray}
    (hone : OneRoot ℓ₂ after log) (hw : FreshWide log) {b : ByteArray}
    (hb : Properties.Challenger.bitsOf b <+: (Verity.Sampled.freshCoins ℓ₂ after log).flatten) (j : ℕ) :
    stageWidth b j ≤ 16 * fitOf (stageLocate ℓ₁ ℓ₂) (j + 1) [] log := by
  have hf := fitOf_ge (stageLocate ℓ₁ ℓ₂) (j + 1) (widthOf_stage ℓ₁ ℓ₂ (j + 1)) log [] (by simp [usedOf])
  have hl := length_devBits ℓ₁ ℓ₂ (j + 1) log hw []
  rw [devBits_second ℓ₁ ℓ₂ hne after j log [] hone] at hl
  simp only [List.countP_nil, Nat.sub_zero] at hl
  have hcs := freshCoins_length (ℓ := ℓ₂) (after := after) hw
  generalize Verity.Sampled.freshCoins ℓ₂ after log = cs at hl hb hcs
  rw [length_flatten_const _ (fun c hc => hcs c (List.mem_of_mem_take (List.mem_of_mem_drop hc))),
    List.length_drop, List.length_take] at hl
  have hflat := length_flatten_const cs hcs
  have hpre := hb.length_le
  have hlen := length_bitsOf b
  unfold stageWidth
  simp only [usedOf, Nat.sub_zero] at hf
  simp only [SpotFlock.L]
  omega

/-- **`FirstElems`' elements at the locator**: where the coins agree below what fits for stage 0 (`ElemAgrees`), they
agree below some `u` whose `16 u` bytes hold the first stage's. -/
theorem elems_first (ℓ₁ ℓ₂ : String) {log : List Verity.Release} {after : ByteArray} (hone : OneRoot ℓ₁ after log)
    (hw : FreshWide log) {b : ByteArray}
    (hb : Properties.Challenger.bitsOf b <+: (Verity.Sampled.freshCoins ℓ₁ after log).flatten)
    (hL : b.size ≤ SpotFlock.L) {c c' : FlockDeploy.Coin}
    (he : ∀ j : Fin 65536, j.val < fitOf (stageLocate ℓ₁ ℓ₂) 0 [] log → c j = c' j) :
    ∃ u, b.size ≤ 16 * u ∧ ∀ j : Fin 65536, j.val < u → c j = c' j :=
  ⟨_, size_le_fit_first ℓ₁ ℓ₂ hone hw hb hL, he⟩

/-- **`SecondElems`' elements at the locator**: as `elems_first`, at stage `j + 1` and its slice (`stageWidth`). -/
theorem elems_second (ℓ₁ ℓ₂ : String) (hne : ℓ₁ ≠ ℓ₂) {log : List Verity.Release} {after : ByteArray}
    (hone : OneRoot ℓ₂ after log) (hw : FreshWide log) {b : ByteArray}
    (hb : Properties.Challenger.bitsOf b <+: (Verity.Sampled.freshCoins ℓ₂ after log).flatten) (j : ℕ)
    {c c' : FlockDeploy.Coin} (he : ∀ k : Fin 65536, k.val < fitOf (stageLocate ℓ₁ ℓ₂) (j + 1) [] log → c k = c' k) :
    ∃ u, stageWidth b j ≤ 16 * u ∧ ∀ k : Fin 65536, k.val < u → c k = c' k :=
  ⟨_, stageWidth_le_fit_second ℓ₁ ℓ₂ hne hone hw hb j, he⟩

/-- **`DevFirst` at the locator** (STAGECOINS's, at `draw` the first stage's sample): where the first stage's sample is the
locator's family after one root (`StageDraws`). -/
theorem devFirst_stage {Acc : Verity.Exchange → Prop} {draw : Verity.Exchange → Option (String × ByteArray)}
    (ℓ₁ ℓ₂ : String) (hS : StageDraws Acc draw ℓ₁) :
    ∀ o, Acc o → ∀ label after (b : ByteArray), draw o = some (label, after) →
      Properties.Challenger.bitsOf b <+: (Verity.Sampled.freshCoins label after o.log).flatten →
      0 < b.size → b.size ≤ SpotFlock.L →
      ∃ d, dev (stageLocate ℓ₁ ℓ₂) o.log 0 = some d ∧ (SpotFlock.src d).extract 0 b.size = b := by
  intro o hA label after b hd hb h0 hL
  obtain ⟨rfl, hone⟩ := hS o hA label after hd
  exact devFirst_of _ ℓ₂ hone hb h0 hL

/-- **`DevSecond` at the locator** (STAGECOINS's, at `draw` the second stage's sample): where the second stage's sample is
the locator's second family after one root (`StageDraws`), the two families differ and each fresh coin the log holds
carries its 512 bits (`FreshWide`). -/
theorem devSecond_stage {Acc : Verity.Exchange → Prop} {draw : Verity.Exchange → Option (String × ByteArray)}
    (ℓ₁ ℓ₂ : String) (hne : ℓ₁ ≠ ℓ₂) (hS : StageDraws Acc draw ℓ₂) (stage : ℕ) :
    ∀ o, Acc o → FreshWide o.log → ∀ label after (b : ByteArray), draw o = some (label, after) →
      Properties.Challenger.bitsOf b <+: (Verity.Sampled.freshCoins label after o.log).flatten →
      b.size ≤ stage * SpotFlock.L → ∀ j < stage, 0 < stageWidth b j →
      ∃ d, dev (stageLocate ℓ₁ ℓ₂) o.log (j + 1) = some d ∧
        (SpotFlock.src d).extract 0 (stageWidth b j) =
          b.extract (j * SpotFlock.L) (j * SpotFlock.L + stageWidth b j) := by
  intro o hA hw label after b hd hb _ j _ hpos
  obtain ⟨rfl, hone⟩ := hS o hA label after hd
  exact devSecond_of ℓ₁ _ hne hone hw hb j hpos

end Verity.SampledTwo.CoinMap
