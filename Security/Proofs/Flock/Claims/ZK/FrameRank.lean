import Flock.Firewall.Drive
import Proofs.Flock.Claims.Complete.Zk.MaskRank

/-!
# The frame's rank check is the prover's (`FrameAdmits`)

At the opening's label the deployed frame refuses coins at which the mask slot's weights have GF(2)-rank short of
`128 k` over the `k` claim points so far (`Frame.Masker.label`, `Frame.maskRank`): it reduces each word's weight row,
`2 k` limbs of 64 bits, into an echelon basis whose pivot is the row's lowest set bit, and counts the rows it inserts.
`FrameAdmits` needs that a rank the frame finds full is the session's (`MaskRank.maskRankOk`, the weights' GF(2)-span
all of `GF(2^128)^k`):

* `lowBit_spec`: `Frame.lowBit v` is `v`'s lowest set bit (`v &&& -v` isolates it).
* `span_of_pivots`: rows with a pivot at every one of the `128 k` columns, each in an additive subgroup `W`, put every
  vector in `W`.
* `maskRank_inv`: every row the frame inserts has its pivot at its column and is a GF(2)-sum of weight rows, and the
  rank counts the columns filled.
* **`maskRankOk_of_maskRank`**: so `maskRank points m kLog = 128 · points.size` gives `maskRankOk` at the points and
  the slot's words, which at the statement's slot (`Drive.maskSlot`) are its mask words (`words_maskSlot`,
  `Composed.maskWordsOf`).
-/

namespace Claims.ZKServer.Rank

open _root_.Flock (F128)
open _root_.Flock.Firewall.Frame (MaskSlot maskRank lowBit)
open FlockLevel3 (GF128)
open FlockComplete.MaskRank (maskRankOk weights maskRankOk_iff)

/-! ## Loops in `Id` -/

theorem forIn_inv {α β : Type} (P : β → Prop) (f : α → β → Id (ForInStep β)) :
    ∀ (l : List α), (∀ a ∈ l, ∀ b, P b → P (f a b).run.value) → ∀ b, P b → P (forIn l b f).run
  | [], _, b, hb => hb
  | a :: l, hf, b, hb => by
    rw [List.forIn_cons]
    have h := hf a (List.mem_cons_self ..) b hb
    obtain ⟨s, hs⟩ : ∃ s : ForInStep β, f a b = s := ⟨_, rfl⟩
    rw [hs] at h ⊢
    cases s with
    | done b' => exact h
    | yield b' => exact forIn_inv P f l (fun a' ha' => hf a' (List.mem_cons_of_mem _ ha')) b' h

/-- **A loop over `[:N]` in `Id` keeps an invariant** each step keeps, and so whatever the invariant gives. -/
theorem forIn_range_elim {β : Type} (P : β → Prop) (Q : β → Prop) {N : ℕ} {f : ℕ → β → Id (ForInStep β)} {b : β}
    (hb : P b) (hf : ∀ a < N, ∀ b, P b → P (f a b).run.value) (hQ : ∀ r, P r → Q r) :
    Q (forIn [:N] b f).run := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  refine hQ _ (forIn_inv P f _ (fun a ha => hf a ?_) b hb)
  simp only [List.mem_range'_1, Std.Legacy.Range.size] at ha
  omega

/-! ## Bits -/

theorem testBit_ge_64 (u : UInt64) {j : ℕ} (hj : 64 ≤ j) : u.toNat.testBit j = false :=
  Nat.testBit_lt_two_pow (lt_of_lt_of_le u.toNat_lt (Nat.pow_le_pow_right (by decide) hj))

theorem limb_eq_zero (u : UInt64) (h : ∀ j < 64, u.toNat.testBit j = false) : u = 0 := by
  apply UInt64.toNat_inj.mp
  apply Nat.eq_of_testBit_eq
  intro j
  rw [UInt64.toNat_zero, Nat.zero_testBit]
  by_cases hj : j < 64
  · exact h j hj
  · exact testBit_ge_64 u (by omega)

theorem toNat_testBit (u : UInt64) (j : ℕ) : u.toNat.testBit j = u.toBitVec.getLsbD j :=
  BitVec.testBit_toNat u.toBitVec

/-- **`lowBit v` is `v`'s lowest set bit**, for `v ≠ 0`: `v &&& (0 - v)` is `2^t` at `v`'s lowest set bit `t`. -/
theorem lowBit_spec (v : UInt64) (hv : v ≠ 0) :
    lowBit v < 64 ∧ v.toNat.testBit (lowBit v) = true ∧ ∀ j < lowBit v, v.toNat.testBit j = false := by
  classical
  have hex : ∃ j, v.toNat.testBit j = true := by
    by_contra hc
    apply hv
    apply UInt64.toNat_inj.mp
    apply Nat.eq_of_testBit_eq
    intro j
    rw [UInt64.toNat_zero, Nat.zero_testBit]
    cases h : v.toNat.testBit j
    · rfl
    · exact absurd ⟨j, h⟩ hc
  let t := Nat.find hex
  have ht : v.toNat.testBit t = true := Nat.find_spec hex
  have hmin : ∀ j < t, v.toNat.testBit j = false := fun j hj =>
    Bool.eq_false_iff.mpr (Nat.find_min hex hj)
  have ht64 : t < 64 := by
    by_contra h
    rw [testBit_ge_64 v (by omega)] at ht
    exact Bool.false_ne_true ht
  have hand : (v &&& (0 - v)).toNat = 2 ^ t := by
    apply Nat.eq_of_testBit_eq
    intro i
    rw [Nat.testBit_two_pow, toNat_testBit, UInt64.toBitVec_and, UInt64.toBitVec_sub, UInt64.toBitVec_zero,
      BitVec.zero_sub, BitVec.getLsbD_and, BitVec.getLsbD_neg]
    have hv' : ∀ j, v.toBitVec.getLsbD j = v.toNat.testBit j := fun j => (toNat_testBit v j).symm
    rw [hv']
    by_cases hi : 64 ≤ i
    · rw [testBit_ge_64 v hi, Bool.false_and]
      exact (decide_eq_false (by omega)).symm
    rcases Nat.lt_trichotomy i t with h | h | h
    · rw [hmin i h, Bool.false_and]
      exact (decide_eq_false (by omega)).symm
    · rw [h]
      have : ¬ ∃ j < t, v.toBitVec.getLsbD j = true := by
        rintro ⟨j, hj, hj'⟩
        rw [hv', hmin j hj] at hj'
        exact Bool.false_ne_true hj'
      simp [ht, this, ht64]
    · have : ∃ j < i, v.toBitVec.getLsbD j = true := ⟨t, h, by rw [hv']; exact ht⟩
      have hne : t ≠ i := by omega
      have hi' : i < 64 := by omega
      cases hb : v.toNat.testBit i <;> simp [this, hne, hi']
  have hlow : lowBit v = t := by
    unfold lowBit
    rw [hand, Nat.log2_two_pow]
  rw [hlow]
  exact ⟨ht64, ht, hmin⟩

/-! ## Rows of limbs -/

/-- The point vector a row of limbs carries: point `i`'s word is limbs `2 i` (low) and `2 i + 1` (high). -/
def vecOf (n : ℕ) (row : Array UInt64) : Fin n → GF128 := fun i => GF128.of ⟨row[2 * i.val]!, row[2 * i.val + 1]!⟩

/-- Bit `col` of a row of limbs. -/
def bitAt (row : Array UInt64) (col : ℕ) : Bool := (row[col / 64]!).toNat.testBit (col % 64)

/-- A row whose lowest set bit is `col`. -/
def Piv (row : Array UInt64) (col : ℕ) : Prop := bitAt row col = true ∧ ∀ c < col, bitAt row c = false

theorem getElem!_mapRange (N : ℕ) (g : ℕ → UInt64) {t : ℕ} (ht : t < N) : ((Array.range N).map g)[t]! = g t := by
  have h : t < ((Array.range N).map g).size := by simp [ht]
  rw [getElem!_pos _ t h, Array.getElem_map, Array.getElem_range]

theorem vecOf_xor (n : ℕ) (row b r : Array UInt64) (h : ∀ q < 2 * n, r[q]! = row[q]! ^^^ b[q]!) :
    vecOf n r = vecOf n row + vecOf n b := by
  funext i
  simp only [vecOf, Pi.add_apply]
  rw [h _ (by omega), h _ (by omega), FlockLevel3.GF128.add_eq]
  rfl

/-- The limbs below a pivot's are zero. -/
theorem limb_zero_of_piv {b : Array UInt64} {col q : ℕ} (hp : Piv b col) (hq : 64 * (q + 1) ≤ col) : b[q]! = 0 := by
  apply limb_eq_zero
  intro j hj
  have := hp.2 (64 * q + j) (by omega)
  unfold bitAt at this
  rwa [show (64 * q + j) / 64 = q by omega, show (64 * q + j) % 64 = j by omega] at this

/-- A row with no set bit below `128 n` carries the zero vector. -/
theorem vecOf_zero (n : ℕ) (v : Array UInt64) (h : ∀ c < 64 * (2 * n), bitAt v c = false) : vecOf n v = 0 := by
  have hl : ∀ q < 2 * n, v[q]! = 0 := fun q hq => limb_eq_zero _ fun j hj => by
    have := h (64 * q + j) (by omega)
    unfold bitAt at this
    rwa [show (64 * q + j) / 64 = q by omega, show (64 * q + j) % 64 = j by omega] at this
  funext i
  simp only [vecOf, Pi.zero_apply]
  rw [hl _ (by omega), hl _ (by omega), FlockLevel3.GF128.zero_eq]
  rfl

theorem add_self_vec {n : ℕ} (a : Fin n → GF128) : a + a = 0 := FlockComplete.MaskRank.add_self_pi a

/-- **Pivots at every column span**: rows in `W` with their pivot at each column below `128 n` put every row of
limbs' vector in `W` (by downward induction on the lowest column a row may have set). -/
theorem span_of_pivots (n : ℕ) (W : AddSubgroup (Fin n → GF128)) (piv : ℕ → Array UInt64)
    (hp : ∀ c < 64 * (2 * n), Piv (piv c) c ∧ vecOf n (piv c) ∈ W) :
    ∀ d (v : Array UInt64), (∀ c < 64 * (2 * n) - d, bitAt v c = false) → vecOf n v ∈ W := by
  intro d
  induction d with
  | zero =>
    intro v hv
    rw [vecOf_zero n v (by simpa using hv)]
    exact W.zero_mem
  | succ d ih =>
    intro v hv
    by_cases hd : 64 * (2 * n) ≤ d
    · exact ih v (fun c hc => hv c (by omega))
    let c₀ := 64 * (2 * n) - (d + 1)
    have hc₀ : c₀ < 64 * (2 * n) := by omega
    by_cases hb : bitAt v c₀ = false
    · refine ih v (fun c hc => ?_)
      by_cases hcc : c < c₀
      · exact hv c hcc
      · rw [show c = c₀ by omega]; exact hb
    have hb' : bitAt v c₀ = true := by simpa using hb
    obtain ⟨⟨hpc, hpl⟩, hW⟩ := hp c₀ hc₀
    let v' := (Array.range (2 * n)).map fun q => v[q]! ^^^ (piv c₀)[q]!
    have hv' : ∀ q < 2 * n, v'[q]! = v[q]! ^^^ (piv c₀)[q]! := fun q hq => getElem!_mapRange _ _ hq
    have hbit : ∀ c < 64 * (2 * n), bitAt v' c = (bitAt v c ^^ bitAt (piv c₀) c) := by
      intro c hc
      unfold bitAt
      rw [hv' _ (by omega), UInt64.toNat_xor, Nat.testBit_xor]
    have hin := ih v' (fun c hc => by
      rw [hbit c (by omega)]
      by_cases hcc : c < c₀
      · rw [hv c hcc, hpl c hcc]; rfl
      · rw [show c = c₀ by omega, hb', hpc]; rfl)
    have heq : vecOf n v = vecOf n v' + vecOf n (piv c₀) := by
      rw [vecOf_xor n v (piv c₀) v' hv', add_assoc, add_self_vec, add_zero]
    rw [heq]
    exact W.add_mem hin hW

/-! ## The frame's reduction -/

/-- Rows from position `w` of a range, found or not. -/
theorem find?_range'_some {p : ℕ → Bool} :
    ∀ (len w k : ℕ), List.find? p (List.range' w len) = some k →
      w ≤ k ∧ k < w + len ∧ p k = true ∧ ∀ t, w ≤ t → t < k → p t = false
  | 0, _, _, h => by simp at h
  | len + 1, w, k, h => by
    rw [List.range'_succ, List.find?_cons] at h
    cases hw : p w with
    | true =>
      rw [hw] at h
      cases h
      exact ⟨le_refl _, by omega, hw, fun t h1 h2 => by omega⟩
    | false =>
      rw [hw] at h
      obtain ⟨h1, h2, h3, h4⟩ := find?_range'_some len (w + 1) k h
      refine ⟨by omega, by omega, h3, fun t ht1 ht2 => ?_⟩
      by_cases htw : t = w
      · rw [htw]; exact hw
      · exact h4 t (by omega) ht2

theorem getElem!_set! {α : Type} [Inhabited α] (B : Array α) {i : ℕ} (j : ℕ) (v : α) (hi : i < B.size) :
    (B.set! i v)[j]! = if j = i then v else B[j]! := by
  by_cases hj : j < B.size
  · have hj' : j < (B.set! i v).size := by simpa using hj
    rw [getElem!_pos _ j hj', getElem!_pos _ j hj]
    simp only [Array.set!, Array.getElem_setIfInBounds hj]
    by_cases h : i = j
    · simp [h]
    · simp [h, Ne.symm h]
  · have hj' : ¬ j < (B.set! i v).size := by simpa using hj
    have hne : j ≠ i := by omega
    rw [getElem!_neg _ j hj', getElem!_neg _ j hj, if_neg hne]

theorem countP_range_flip (p q : ℕ → Bool) {c : ℕ} (hp : p c = false) (hq : q c = true)
    (h : ∀ c', c' ≠ c → q c' = p c') : ∀ N, c < N → (List.range N).countP q = (List.range N).countP p + 1
  | 0, hc => by omega
  | N + 1, hc => by
    rw [List.range_succ, List.countP_append, List.countP_append]
    by_cases hN : N = c
    · subst hN
      have : (List.range N).countP q = (List.range N).countP p :=
        List.countP_congr fun x hx => by rw [h x (by simp at hx; omega)]
      simp [this, hp, hq]
    · rw [countP_range_flip p q hp hq h N (by omega)]
      simp [h N hN]
      omega

theorem replicate_none (F c : ℕ) : (Array.replicate F (none : Option (Array UInt64)))[c]! = none := by
  by_cases h : c < F
  · rw [getElem!_pos _ c (by simpa using h)]
    simp
  · rw [getElem!_neg _ c (by simpa using h)]
    rfl

/-- A loop in `Id` followed by `k`: whatever `k` makes of a state the loop's invariant holds at. -/
theorem forIn_range_elim_bind {β γ : Type} (P : β → Prop) (Q : γ → Prop) {N : ℕ} {f : ℕ → β → Id (ForInStep β)}
    {b : β} (k : β → Id γ) (hb : P b) (hf : ∀ a < N, ∀ b, P b → P (f a b).run.value)
    (hQ : ∀ r, P r → Q (k r).run) : Q (forIn [:N] b f >>= k).run :=
  forIn_range_elim P (fun r => Q (k r).run) hb hf hQ

variable (n : ℕ) (W : AddSubgroup (Fin n → GF128))

/-- **The basis so far**: `full` slots, the rank the count of those filled, and each filled slot `c` a row in `W` with
its pivot at `c`. -/
def OutInv (full : ℕ) (B : Array (Option (Array UInt64))) (rank : ℕ) : Prop :=
  B.size = full ∧ rank = (List.range full).countP (fun c => (B[c]!).isSome) ∧
    ∀ c b, B[c]! = some b → Piv b c ∧ vecOf n b ∈ W

/-- **A row mid-reduction**: the basis's invariant, the row's limbs below `w` zero, and the row in `W`. -/
def InInv (full : ℕ) (s : Array (Option (Array UInt64)) × ℕ × Array UInt64 × ℕ) : Prop :=
  OutInv n W full s.1 s.2.1 ∧ (∀ t < s.2.2.2, s.2.2.1[t]! = 0) ∧ vecOf n s.2.2.1 ∈ W

theorem outInv_init (F : ℕ) : OutInv n W F (Array.replicate F none) 0 := by
  refine ⟨by simp, ?_, fun c b h => ?_⟩
  · symm
    rw [List.countP_eq_zero]
    intro c _
    simp [replicate_none]
  · rw [replicate_none] at h
    cases h

section Step

variable {n W} {L : ℕ} {B : Array (Option (Array UInt64))} {rank : ℕ} {row : Array UInt64}
  {w k : ℕ} (hs : InInv n W (64 * L) (B, rank, row, w))
  (hfind : List.find? (fun k => row[k]! != 0) (List.range' w (L - w)) = some k)
include hs hfind

theorem step_facts : k < L ∧ row[k]! ≠ 0 ∧ (∀ t < k, row[t]! = 0) ∧ Piv row (64 * k + lowBit row[k]!) ∧
    64 * k + lowBit row[k]! < 64 * L := by
  obtain ⟨-, hz, -⟩ := hs
  obtain ⟨hwk, hkl, hpk, hbefore⟩ := find?_range'_some _ _ _ hfind
  have hk0 : row[k]! ≠ 0 := by simpa using hpk
  have hzk : ∀ t < k, row[t]! = 0 := fun t ht => by
    by_cases htw : t < w
    · exact hz t htw
    · simpa using hbefore t (by omega) ht
  obtain ⟨hlb, hlbt, hlbl⟩ := lowBit_spec _ hk0
  refine ⟨by omega, hk0, hzk, ⟨?_, fun c hc => ?_⟩, by omega⟩
  · unfold bitAt
    rw [show (64 * k + lowBit row[k]!) / 64 = k by omega, show (64 * k + lowBit row[k]!) % 64 = lowBit row[k]! by omega]
    exact hlbt
  · unfold bitAt
    by_cases hck : c / 64 < k
    · rw [hzk _ hck, UInt64.toNat_zero, Nat.zero_testBit]
    · rw [show c / 64 = k by omega]
      exact hlbl _ (by omega)

/-- **A reduction step**: the row's pivot column holds a basis row `b`, and the row less `b` from limb `k` on keeps
the invariant at `k`. -/
theorem step_xor (hL : L = 2 * n) {b : Array UInt64} (hb : B[64 * k + lowBit row[k]!]! = some b) :
    InInv n W (64 * L) (B, rank, (Array.range L).map (fun t => if t ≥ k then row[t]! ^^^ b[t]! else row[t]!), k) := by
  obtain ⟨hkL, -, hzk, -, -⟩ := step_facts hs hfind
  obtain ⟨⟨hBsz, hcnt, hBrow⟩, -, hrowW⟩ := hs
  dsimp only at hBsz hcnt hBrow hrowW
  obtain ⟨hbp, hbW⟩ := hBrow _ b hb
  have hbz : ∀ q < k, b[q]! = 0 := fun q hq => limb_zero_of_piv hbp (by omega)
  refine ⟨⟨hBsz, hcnt, hBrow⟩, fun t ht => ?_, ?_⟩
  · change t < k at ht
    show ((Array.range L).map fun t => if t ≥ k then row[t]! ^^^ b[t]! else row[t]!)[t]! = 0
    rw [getElem!_mapRange _ _ (by omega), if_neg (by omega)]
    exact hzk t ht
  · show vecOf n ((Array.range L).map fun t => if t ≥ k then row[t]! ^^^ b[t]! else row[t]!) ∈ W
    rw [vecOf_xor n row b _ (fun q hq => by
      rw [getElem!_mapRange _ _ (by omega)]
      split
      · rfl
      · rw [hbz q (by omega), UInt64.xor_zero])]
    exact W.add_mem hrowW hbW

/-- **An insertion**: the row's pivot column is free, and the row fills it. -/
theorem step_insert (hL : L = 2 * n) (hb : B[64 * k + lowBit row[k]!]! = none) :
    InInv n W (64 * L) (B.set! (64 * k + lowBit row[k]!) (some row), rank + 1, row, k) := by
  obtain ⟨-, -, hzk, hpiv, hcol⟩ := step_facts hs hfind
  obtain ⟨⟨hBsz, hcnt, hBrow⟩, -, hrowW⟩ := hs
  dsimp only at hBsz hcnt hBrow hrowW
  have hcB : 64 * k + lowBit row[k]! < B.size := by omega
  refine ⟨⟨?_, ?_, fun c b' h => ?_⟩, hzk, hrowW⟩
  · show (B.set! (64 * k + lowBit row[k]!) (some row)).size = 64 * L
    simp [hBsz]
  · show rank + 1 = (List.range (64 * L)).countP fun c => ((B.set! (64 * k + lowBit row[k]!) (some row))[c]!).isSome
    rw [hcnt]
    symm
    refine countP_range_flip _ _ (c := 64 * k + lowBit row[k]!) (by simp [hb])
      (by rw [getElem!_set! B _ _ hcB, if_pos rfl]; rfl)
      (fun c' hc' => by rw [getElem!_set! B _ _ hcB, if_neg hc']) _ hcol
  · change (B.set! (64 * k + lowBit row[k]!) (some row))[c]! = some b' at h
    rw [getElem!_set! B _ _ hcB] at h
    split_ifs at h with hc
    · cases h
      rw [hc]
      exact ⟨hpiv, hrowW⟩
    · exact hBrow c b' h

end Step

/-! ## The weight rows -/

theorem row0_toList (E : Array F128 → F128) :
    ∀ (l : List (Array F128)) (acc : Array UInt64),
      (l.foldl (fun acc p => (acc.push (E p).lo).push (E p).hi) acc).toList =
        acc.toList ++ l.flatMap fun p => [(E p).lo, (E p).hi]
  | [], acc => by simp
  | p :: l, acc => by
    rw [List.foldl_cons, row0_toList E l, List.flatMap_cons]
    simp

theorem flatMap_pair (g h : Array F128 → UInt64) :
    ∀ (l : List (Array F128)) (i : ℕ), i < l.length →
      (l.flatMap fun p => [g p, h p])[2 * i]! = g l[i]! ∧ (l.flatMap fun p => [g p, h p])[2 * i + 1]! = h l[i]!
  | [], _, hi => by simp at hi
  | p :: l, 0, _ => by simp
  | p :: l, i + 1, hi => by
    have := flatMap_pair g h l i (by simp at hi; omega)
    rw [List.flatMap_cons, show 2 * (i + 1) = 2 * i + 2 by omega, show 2 * i + 2 + 1 = 2 * i + 1 + 2 by omega]
    simpa using this

theorem eqWord_frame (p : Array F128) (x : ℕ) :
    _root_.Flock.Firewall.Frame.eqWord p x = FlockSoundness.Discharge.Composed.eqWord p.toList x := by
  unfold _root_.Flock.Firewall.Frame.eqWord FlockSoundness.Discharge.Composed.eqWord
  simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, bind_pure,
    Array.length_toList, Array.getElem!_toList]
  have hs : [:p.size].size = p.size := by simp [Std.Legacy.Range.size]
  rw [hs, List.range_eq_range']
  rfl

/-- **The frame's row for the word `x` carries the weight vector at `x`.** -/
theorem vecOf_row0 (points : Array (Array F128)) (x : ℕ) :
    vecOf (points.map Array.toList).size
        (points.foldl (fun acc p => (acc.push (_root_.Flock.Firewall.Frame.eqWord p x).lo).push
          (_root_.Flock.Firewall.Frame.eqWord p x).hi) #[]) =
      weights (points.map Array.toList) x := by
  funext i
  have hi : i.val < points.size := by have := i.isLt; simpa using this
  have hrow := row0_toList (fun p => _root_.Flock.Firewall.Frame.eqWord p x) points.toList #[]
  rw [Array.foldl_toList] at hrow
  obtain ⟨h0, h1⟩ := flatMap_pair (fun p => (_root_.Flock.Firewall.Frame.eqWord p x).lo)
    (fun p => (_root_.Flock.Firewall.Frame.eqWord p x).hi) points.toList i.val (by simpa using hi)
  simp only [vecOf, weights]
  rw [← Array.getElem!_toList, ← Array.getElem!_toList (i := 2 * i.val + 1), hrow]
  simp only [List.nil_append]
  rw [h0, h1, Array.getElem!_toList, getElem!_pos points i.val hi, Fin.getElem_fin, Array.getElem_map, eqWord_frame]

/-! ## The rank check -/

def toF (a : GF128) : F128 := a

/-- **A rank the frame finds full is the prover's** (`zk_veil::mask_rank_ok`): where the frame's reduction over the slot's
words fills all `128 k` columns, every vector of `GF(2^128)^k` is a GF(2)-sum of the weight rows. -/
theorem maskRankOk_of_maskRank (points : Array (Array F128)) (m : MaskSlot) (kLog : ℕ)
    (h : maskRank points m kLog = 64 * (2 * points.size)) :
    maskRankOk (points.map Array.toList) ((List.range (m.blocks * m.n)).map (m.word kLog)) = true := by
  rw [maskRankOk_iff]
  have hw : ∀ i (h : i < ((List.range (m.blocks * m.n)).map (m.word kLog)).length),
      ((List.range (m.blocks * m.n)).map (m.word kLog))[i] = m.word kLog i := fun i h => by simp
  have hwl : ((List.range (m.blocks * m.n)).map (m.word kLog)).length = m.blocks * m.n := by simp
  generalize hpts : points.map Array.toList = pts
  generalize hwords : (List.range (m.blocks * m.n)).map (m.word kLog) = words at hw hwl ⊢
  have hn : points.size = pts.size := by rw [← hpts, Array.size_map]
  let W : AddSubgroup (Fin pts.size → GF128) :=
    AddSubgroup.closure (Set.range fun j : Fin words.length => weights pts words[j])
  show W = ⊤
  have hrow0 : ∀ i < m.blocks * m.n, vecOf pts.size
      (points.foldl (fun acc p => (acc.push (_root_.Flock.Firewall.Frame.eqWord p (m.word kLog i)).lo).push
        (_root_.Flock.Firewall.Frame.eqWord p (m.word kLog i)).hi) #[]) ∈ W := by
    intro i hi
    have hv := vecOf_row0 points (m.word kLog i)
    rw [hpts] at hv
    rw [hv]
    refine AddSubgroup.subset_closure ⟨⟨i, by omega⟩, ?_⟩
    show weights pts words[i] = _
    rw [hw]
  have hL : 2 * points.size = 2 * pts.size := by rw [hn]
  revert h
  unfold maskRank
  refine forIn_range_elim_bind (fun r => OutInv pts.size W (64 * (2 * points.size)) r.1 r.2)
    (fun g => g = 64 * (2 * points.size) → W = ⊤) _ (outInv_init pts.size W _) ?_ ?_
  · rintro i hi ⟨B, rank⟩ hs
    dsimp only
    split
    · exact hs
    · refine forIn_range_elim_bind (InInv pts.size W (64 * (2 * points.size)))
        (fun g : ForInStep (Array (Option (Array UInt64)) × ℕ) =>
          OutInv pts.size W (64 * (2 * points.size)) g.value.1 g.value.2) _
        ⟨hs, fun t ht => absurd ht (Nat.not_lt_zero _), hrow0 i hi⟩ ?_ (fun R hR => hR.1)
      rintro - - ⟨B', rank', row, w⟩ hs'
      dsimp only
      split
      · exact hs'
      · rename_i k hfind
        split
        · rename_i b hb
          exact step_xor hs' hfind hL hb
        · rename_i hb
          exact step_insert hs' hfind hL hb
  · rintro ⟨B, rank⟩ ⟨-, hcnt, hrows⟩ hfull
    have hall : ∀ c < 64 * (2 * points.size), (B[c]!).isSome = true := by
      have hlen : (List.range (64 * (2 * points.size))).countP (fun c => (B[c]!).isSome) =
          (List.range (64 * (2 * points.size))).length := by
        rw [List.length_range, ← hcnt]
        exact hfull
      rw [List.countP_eq_length] at hlen
      exact fun c hc => hlen c (List.mem_range.mpr hc)
    have hp : ∀ c < 64 * (2 * pts.size), Piv ((B[c]!).getD #[]) c ∧ vecOf pts.size ((B[c]!).getD #[]) ∈ W := by
      intro c hc
      obtain ⟨b, hb⟩ := Option.isSome_iff_exists.mp (hall c (by omega))
      rw [hb]
      exact hrows c b hb
    rw [eq_top_iff]
    intro a _
    let v : Array UInt64 := (Array.range (2 * pts.size)).map fun t =>
      if h : t / 2 < pts.size then (if t % 2 = 0 then (toF (a ⟨t / 2, h⟩)).lo else (toF (a ⟨t / 2, h⟩)).hi) else 0
    have hva : vecOf pts.size v = a := by
      funext i
      have e0 : v[2 * i.val]! = (toF (a i)).lo := by
        rw [getElem!_mapRange _ _ (by omega), dif_pos (by omega), if_pos (by omega)]
        exact congrArg (fun j => (toF (a j)).lo) (Fin.ext (by simp))
      have e1 : v[2 * i.val + 1]! = (toF (a i)).hi := by
        rw [getElem!_mapRange _ _ (by omega), dif_pos (by omega), if_neg (by omega)]
        exact congrArg (fun j => (toF (a j)).hi) (Fin.ext (by simp; omega))
      simp only [vecOf]
      rw [e0, e1]
      rfl
    rw [← hva]
    exact span_of_pivots pts.size W (fun c => (B[c]!).getD #[]) hp (64 * (2 * pts.size)) v
      (fun c hc => by omega)

/-- The slot's words block by block. -/
theorem words_split (B N K w0 : ℕ) (hN : 0 < N) :
    (List.range (B * N)).map (fun i => i / N * K + w0 + i % N) =
      (List.range B).flatMap fun b => (List.range N).map fun j => b * K + w0 + j := by
  induction B with
  | zero => simp
  | succ B ih =>
    rw [Nat.succ_mul, List.range_add, List.map_append, ih, List.range_succ, List.flatMap_append, List.map_map]
    congr 1
    simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil]
    refine List.map_congr_left fun j hj => ?_
    have hj' : j < N := List.mem_range.mp hj
    have h1 : (B * N + j) / N = B := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ hN, Nat.div_eq_of_lt hj', Nat.zero_add]
    have h2 : (B * N + j) % N = j := by rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hj']
    simp only [Function.comp, h1, h2]

/-- **The statement's mask slot's words are its mask words**: the slot `Drive.maskSlot` takes, word by word, is
`Composed.maskWordsOf`. -/
theorem words_maskSlot {st : _root_.Flock.Stmt} {m : MaskSlot} (h : _root_.Flock.Firewall.Drive.maskSlot st = .ok m) :
    (List.range (m.blocks * m.n)).map (m.word st.c.kLog) = FlockSoundness.Discharge.Composed.maskWordsOf st := by
  unfold _root_.Flock.Firewall.Drive.maskSlot at h
  unfold FlockSoundness.Discharge.Composed.maskWordsOf
  rw [Array.find?_toList]
  split at h
  · rename_i r hr
    rw [hr]
    by_cases hs : r.slotLog < 7
    · simp only [hs, if_true] at h
      cases h
    · simp only [hs, if_false] at h
      cases h
      exact words_split _ _ _ _ (Nat.two_pow_pos _)
  · cases h

end Claims.ZKServer.Rank
