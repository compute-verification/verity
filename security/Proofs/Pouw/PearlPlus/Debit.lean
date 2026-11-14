module

public import Proofs.Pouw.PearlPlus.Rules
public import Proofs.Pouw.PearlPlus.Keys
public import Proofs.Pouw.PearlPlus.Semantics
public import Proofs.Pouw.PearlC.Freeze
public import Proofs.Pouw.PearlC.AccumProofs

/-!
# Pearl+'s bridge, part (c), the inert-work replay: `word_debit` counts the game's flags and later +0 promotions

`Rule.wordDebit` replays a word's inert work on its two rows of codes: each promotion group's `2^G` masked words
(`Rule.masked`), the running totals, from the last group back the band of keys a total may take before each group and
still promote to the word's total (`least` and `carried`), and per group the nonzero masks whose promoted word lands in
the next group's band. `debitIsTheReplay` proves its two counts are the game's: a mask is kept exactly when `Skip.keeps`
holds, since a key in the band is a total the later promotions carry to the word's total (`KeyBand`), so the flagged
atoms are `Skip.unitFlags`'; and the later promotions into a +0 total are `laterZeroPromos`' at the H100 record. It reads
the keys' three statements (`Keys`); a group of finite atoms has a word below 2⁴⁶ in magnitude (`Freeze.tame_inWord`),
the range `carried` is the rounded add on, and an atom the replay steps finitely is finite.

`replayReadsTheCodes` puts the replay on the semantics' codes. Where a word's rows are whole promotion groups and pass the
rules, the check's A′ and B̃ rows are `pc8Sem`'s codes at the H100 atom bit for bit (`formRow_quant8`: the forming reads
a −0 word as +0 and its atom never steps to −0), and these are `unitChain`'s at `semH3`'s noise. Elsewhere `word_debit`
refuses both words alike.
-/

@[expose] public section

namespace Pouw.PearlPlus.Bridge

open Pouw.PearlC Pouw.Fp8Atom PouwCheck.PearlPlus

namespace DebitSim

/-! ## Loops in `Except` -/

section Loops

variable {α β : Type}

theorem forIn_list_from (P : List α → β → Prop) (f : α → β → Except String (ForInStep β))
    (hstep : ∀ pre x s r, P pre s → f x s = .ok r → ∃ s', r = .yield s' ∧ P (pre ++ [x]) s') :
    ∀ (rest pre : List α) (b r : β), P pre b → forIn rest b f = .ok r → P (pre ++ rest) r
  | [], pre, b, r, hb, h => by
    rw [List.forIn_nil] at h
    obtain rfl := WordsSim.pure_ok h
    rwa [List.append_nil]
  | x :: xs, pre, b, r, hb, h => by
    rw [List.forIn_cons] at h
    obtain ⟨st, hst, h⟩ := WordsSim.bind_ok h
    obtain ⟨s', rfl, hs'⟩ := hstep pre x b st hb hst
    have := forIn_list_from P f hstep xs (pre ++ [x]) s' r hs' h
    rwa [List.append_assoc, List.singleton_append] at this

theorem forIn_list_inv {P : List α → β → Prop} {f : α → β → Except String (ForInStep β)} {l : List α} {b r : β}
    (h : forIn l b f = .ok r) (h0 : P [] b)
    (hstep : ∀ pre x s r, P pre s → f x s = .ok r → ∃ s', r = .yield s' ∧ P (pre ++ [x]) s') : P l r := by
  have := forIn_list_from P f hstep l [] b r h0 h
  rwa [List.nil_append] at this

theorem forIn_range_from (P : ℕ → β → Prop) (f : ℕ → β → Except String (ForInStep β)) (N : ℕ)
    (hstep : ∀ p s r, p < N → P p s → f p s = .ok r → ∃ s', r = .yield s' ∧ P (p + 1) s') :
    ∀ (n c : ℕ) (b r : β), c + n ≤ N → P c b → forIn (List.range' c n) b f = .ok r → P (c + n) r
  | 0, c, b, r, _, hb, h => by
    rw [show List.range' c 0 = [] from rfl, List.forIn_nil] at h
    obtain rfl := WordsSim.pure_ok h
    exact hb
  | n + 1, c, b, r, hn, hb, h => by
    rw [List.range'_succ, List.forIn_cons] at h
    obtain ⟨st, hst, h⟩ := WordsSim.bind_ok h
    obtain ⟨s', rfl, hs'⟩ := hstep c b st (by omega) hb hst
    have := forIn_range_from P f N hstep n (c + 1) s' r (by omega) hs' h
    rwa [show c + 1 + n = c + (n + 1) by omega] at this

theorem forIn_range_inv {P : ℕ → β → Prop} {f : ℕ → β → Except String (ForInStep β)} {n : ℕ} {b r : β}
    (h : forIn (List.range' 0 n) b f = .ok r) (h0 : P 0 b)
    (hstep : ∀ p s r, p < n → P p s → f p s = .ok r → ∃ s', r = .yield s' ∧ P (p + 1) s') : P n r := by
  have := forIn_range_from P f n hstep n 0 b r (by omega) h0 h
  rwa [Nat.zero_add] at this

end Loops

theorem range_size (n : ℕ) : [:n].size = n := by simp [Std.Legacy.Range.size]

/-! ## Arrays -/

section Arrays

variable {α : Type} [Inhabited α]

theorem singleton_get (x : α) : (#[x])[0]! = x := by
  simp

theorem push_get (s : Array α) (o : α) (q : ℕ) (hq : q ≤ s.size) :
    (s.push o)[q]! = if q = s.size then o else s[q]! := by
  by_cases h : q = s.size
  · subst h
    rw [ite_eq_left rfl, getElem!_pos (s.push o) s.size (by simp)]
    exact Array.getElem_push_eq
  · rw [ite_eq_right h, getElem!_pos (s.push o) q (by simp; omega), getElem!_pos s q (by omega)]
    exact Array.getElem_push_lt (by omega)

theorem rev_get (xs : Array α) (g : ℕ) (hg : g < xs.size) : xs.reverse[g]! = xs[xs.size - 1 - g]! := by
  rw [getElem!_pos xs.reverse g (by simpa using hg), getElem!_pos xs (xs.size - 1 - g) (by omega),
    Array.getElem_reverse]

end Arrays

/-! ## Atoms: the check's atom on a word's slices is the game's step -/

/-- The check's two slices `p` are the game's atom `x`: stepping them is `Skip.step` from every word. -/
def AtomRel (p : List ℕ × List ℕ) (x : AtomIn) : Prop := ∀ w, Words.atom w p.1 p.2 = Skip.step w x

theorem step_ok {w : ℕ} {a b : List ℕ} {d : ℕ} (h : Rule.step w a b = .ok d) :
    d = Words.atom w a b ∧ (Words.atom w a b >>> 23) % 256 ≠ 255 := by
  unfold Rule.step at h
  dsimp only at h
  split at h
  · obtain ⟨_, ha, -⟩ := WordsSim.bind_ok h
    exact (WordsSim.throw_ok ha).elim
  · rename_i hc
    exact ⟨(WordsSim.pure_ok h).symm, by simpa only [beq_iff_eq, word_and_ff] using hc⟩

theorem groupStepTotal_nan (wd : ℕ) (ze : ℤ) (acc : Value) (ps : List Value)
    (h : (acc :: ps).any Value.isNaN = true) : groupStepTotal wd ze acc ps = .nan := by
  unfold groupStepTotal
  simp only [h, ↓reduceIte]

/-- An atom with a NaN code steps every word to NaN, whose exponent field is 255. -/
theorem step_nan (w : ℕ) (x : AtomIn) (hx : ¬ Freeze.FiniteAtom x) : Skip.step w x = 0x7FFFFFFF := by
  have hv : (fp32Value w :: ((((List.ofFn x.1).zip (List.ofFn x.2)).take 32).map
      fun c => productTotal c.1 c.2)).any Value.isNaN = true := by
    obtain ⟨l, hl⟩ : ∃ l, (x.1 l).isNaN = true ∨ (x.2 l).isNaN = true := by
      by_contra hne
      push Not at hne
      exact hx ⟨fun l => by simpa using (hne l).1, fun l => by simpa using (hne l).2⟩
    have hlen : ((List.ofFn x.1).zip (List.ofFn x.2)).length ≤ 32 := by simp
    rw [List.any_eq_true]
    refine ⟨productTotal (x.1 l) (x.2 l), List.mem_cons_of_mem _ ?_, ?_⟩
    · rw [List.take_of_length_le hlen, H1T.Proofs.zip_ofFn, List.mem_map]
      exact ⟨(x.1 l, x.2 l), List.mem_ofFn.2 ⟨l, rfl⟩, rfl⟩
    · rcases hl with h | h <;> simp [productTotal, h, Value.isNaN]
  show packValue (groupStepTotal 14 (-139) (fp32Value w)
    ((((List.ofFn x.1).zip (List.ofFn x.2)).take 32).map fun c => productTotal c.1 c.2)) = 0x7FFFFFFF
  rw [groupStepTotal_nan _ _ _ _ hv]
  rfl

theorem rowAtoms_length {k : ℕ} (a b : Fin k → Code) : (rowAtoms a b).length = k / 32 := List.length_ofFn

/-- Atom `j` of group `g` of the word's chunks is the check's slice pair at `32·(4g + j)`. -/
theorem atomRel_chunk {k : ℕ} (a b : Fin k → Code) (g j : ℕ) (hg : g < ((rowAtoms a b).toChunks 4).length)
    (hj : j < (((rowAtoms a b).toChunks 4)[g]'hg).length) :
    AtomRel (Words.slice (Array.ofFn fun l => (a l).val) (32 * (4 * g + j)) 32,
      Words.slice (Array.ofFn fun l => (b l).val) (32 * (4 * g + j)) 32) ((((rowAtoms a b).toChunks 4)[g]'hg)[j]'hj) := by
  obtain ⟨hn, hx⟩ := WordsSim.rowAtoms_chunk a b g j hg hj
  have hA : ∀ t (ht : t < k), (Array.ofFn fun l => (a l).val)[t]! = (a ⟨t, ht⟩).val := fun t ht => by
    rw [getElem!_pos (Array.ofFn fun l => (a l).val) t (by simpa using ht), Array.getElem_ofFn]
  have hB : ∀ t (ht : t < k), (Array.ofFn fun l => (b l).val)[t]! = (b ⟨t, ht⟩).val := fun t ht => by
    rw [getElem!_pos (Array.ofFn fun l => (b l).val) t (by simpa using ht), Array.getElem_ofFn]
  unfold AtomRel
  intro w
  dsimp only
  rw [hx, Words.atom, atomIsTheStep w _ _, WordsSim.slice_code Array.size_ofFn hA rfl hn,
    WordsSim.slice_code Array.size_ofFn hB rfl hn]
  rfl

theorem forall₂_range {γ δ : Type} {R : γ → δ → Prop} (f : ℕ → γ) :
    ∀ l : List δ, (∀ j (hj : j < l.length), R (f j) l[j]) → List.Forall₂ R ((List.range l.length).map f) l
  | [], _ => List.Forall₂.nil
  | y :: l, h => by
    rw [List.length_cons, List.range_succ_eq_map, List.map_cons, List.map_map]
    exact .cons (h 0 (by simp)) (forall₂_range (f ∘ Nat.succ) l fun j hj =>
      h (j + 1) (by simp only [List.length_cons]; omega))

theorem chunks_len {k : ℕ} (a b : Fin k → Code) (hk : k % 128 = 0) :
    ((rowAtoms a b).toChunks 4).length = k / 128 := by
  rw [Freeze.length_toChunks_of_dvd (G := 4) (l := rowAtoms a b) (by norm_num) (by rw [rowAtoms_length]; omega),
    rowAtoms_length]
  omega

theorem chunk_len {k : ℕ} (a b : Fin k → Code) (hk : k % 128 = 0) (g : ℕ) (hg : g < k / 128) :
    (((rowAtoms a b).toChunks 4).getD g []).length = 4 := by
  have hg' : g < ((rowAtoms a b).toChunks 4).length := by rw [chunks_len a b hk]; exact hg
  rw [List.getD_eq_getElem _ _ hg']
  exact Freeze.length_getElem_toChunks_of_dvd (G := 4) (l := rowAtoms a b) (by norm_num)
    (by rw [rowAtoms_length]; omega) g hg'

/-- The check's group `g`, its four slice pairs, is the game's group `g`. -/
theorem group_rel {k : ℕ} (a b : Fin k → Code) (hk : k % 128 = 0) (g : ℕ) (hg : g < k / 128) :
    List.Forall₂ AtomRel ((List.range 4).map fun j =>
      (Words.slice (Array.ofFn fun l => (a l).val) (32 * (4 * g + j)) 32,
       Words.slice (Array.ofFn fun l => (b l).val) (32 * (4 * g + j)) 32))
      (((rowAtoms a b).toChunks 4).getD g []) := by
  have hg' : g < ((rowAtoms a b).toChunks 4).length := by rw [chunks_len a b hk]; exact hg
  have hl := chunk_len a b hk g hg
  rw [List.getD_eq_getElem _ _ hg'] at hl ⊢
  have := forall₂_range (R := AtomRel) (fun j =>
      (Words.slice (Array.ofFn fun l => (a l).val) (32 * (4 * g + j)) 32,
       Words.slice (Array.ofFn fun l => (b l).val) (32 * (4 * g + j)) 32))
    (((rowAtoms a b).toChunks 4)[g]'hg') (fun j hj => atomRel_chunk a b g j hg' hj)
  rwa [hl] at this

/-! ## Promotions -/

theorem stepWord_zero : AccumProofs.StepWord 0 := ⟨by norm_num, by decide, WordsSim.finite?_of (by decide)⟩

theorem promote_some {w : ℕ} {t y : ℚ} (hy : wordQ w = some y) :
    Pouw.PearlC.promote (some t) w = some (rnAdd t y) := by
  simp only [Pouw.PearlC.promote, hy]

/-- The check's FP32 add is the semantics' on any two finite words. -/
theorem addAgrees : ∀ a b : ℕ, (wordQ a).isSome → (wordQ b).isSome →
    Evaluate.Fp32.f32AddV2 a b = TileCircuit.f32Add a b := by
  intro a b ha hb
  obtain ⟨x, hx⟩ := Option.isSome_iff_exists.1 ha
  obtain ⟨y, hy⟩ := Option.isSome_iff_exists.1 hb
  rw [show Evaluate.Fp32.f32AddV2 = Verity.Fp32.f32AddV2 from Properties.Evaluator.Fp32.f32AddV2]
  obtain ⟨Ma, hMa, rfl⟩ := CoreOps.decode_fin hx
  obtain ⟨Mb, hMb, rfl⟩ := CoreOps.decode_fin hy
  unfold Verity.Fp32.f32AddV2 TileCircuit.f32Add
  rw [hMa, hMb, hx, hy]
  simp only [Verity.Fp32.Decoded.addRn]
  rw [CoreOps.roundSum_eq _ _ _ _ Nat.one_pos]
  simp only [Nat.cast_one, div_one, CoreOps.val_eq_zero]

/-- A promotion the check takes is of two finite words, and its word is the rounded add of their values. -/
theorem promote_ok {T w o : ℕ} (h : Rule.promote T w = .ok o) :
    ∃ t y, wordQ T = some t ∧ wordQ w = some y ∧ AccumProofs.StepWord o ∧ wordQ o = some (rnAdd t y) := by
  unfold Rule.promote at h
  dsimp only at h
  split at h
  · obtain ⟨_, ha, -⟩ := WordsSim.bind_ok h
    exact (WordsSim.throw_ok ha).elim
  · rename_i hc
    have he : (Evaluate.Fp32.f32AddV2 T w >>> 23) % 256 ≠ 255 := by
      simpa only [beq_iff_eq, word_and_ff] using hc
    obtain ⟨v, hv⟩ := WordsSim.wordQ_of he
    obtain ⟨hTs, hws, -⟩ := fp32AddIsTheSpec T w (by simp [hv])
    obtain ⟨t, ht⟩ := Option.isSome_iff_exists.1 hTs
    obtain ⟨y, hy⟩ := Option.isSome_iff_exists.1 hws
    have hadd := WordsSim.addWord_of fp32AddIsTheSpec he
    have hlt := WholeCircuit.addWord?_lt hadd
    have ho := WordsSim.pure_ok h
    have hp : PeelExact.promoteWord? T w = some o := by
      rw [← ho]
      unfold PeelExact.promoteWord?
      rw [hadd]
      by_cases h0 : Evaluate.Fp32.f32AddV2 T w = 0x80000000 <;> simp [h0]
    refine ⟨t, y, ht, hy, ?_, TileCircuit.promoteWord_wordQ ht hy hp⟩
    rw [← ho]
    by_cases h0 : Evaluate.Fp32.f32AddV2 T w = 0x80000000
    · rw [ite_eq_left (by simp [h0])]
      exact stepWord_zero
    · rw [ite_eq_right (by simpa using h0)]
      exact ⟨hlt, h0, WordsSim.finite?_of he⟩

/-! ## Masks: `Rule.masked` is every mask's `Skip.wordIn` -/

/-- A mask as the check's bits: bit `j` set skips atom `j`. -/
def encMask : List Bool → ℕ
  | [] => 0
  | b :: m => b.toNat + 2 * encMask m

theorem encMask_append (m : List Bool) (b : Bool) : encMask (m ++ [b]) = encMask m + b.toNat * 2 ^ m.length := by
  induction m with
  | nil => simp [encMask]
  | cons c m ih => simp only [List.cons_append, encMask, ih, List.length_cons, pow_succ]; ring

theorem encMask_lt (m : List Bool) : encMask m < 2 ^ m.length := by
  induction m with
  | nil => simp [encMask]
  | cons b m ih =>
    simp only [encMask, List.length_cons, pow_succ]
    have hb : b.toNat ≤ 1 := by cases b <;> decide
    generalize 2 ^ m.length = N at ih ⊢
    omega

theorem encMask_bit : ∀ (m : List Bool) (j : ℕ), (encMask m >>> j) &&& 1 = (m.getD j false).toNat
  | [], j => by simp [encMask]
  | b :: m, 0 => by
    simp only [encMask, Nat.shiftRight_zero, List.getD_cons_zero, Nat.and_one_is_mod]
    cases b <;> simp only [Bool.toNat_false, Bool.toNat_true] <;> omega
  | b :: m, j + 1 => by
    have e : encMask (b :: m) / 2 = encMask m := by
      simp only [encMask]; cases b <;> simp only [Bool.toNat_false, Bool.toNat_true] <;> omega
    rw [Nat.shiftRight_succ_inside, e, List.getD_cons_succ]
    exact encMask_bit m j

theorem encMask_zero : ∀ m : List Bool, encMask m = 0 → m = List.replicate m.length false
  | [], _ => rfl
  | b :: m, h => by
    simp only [encMask] at h
    have hm : encMask m = 0 := by omega
    cases b
    · rw [List.length_cons, List.replicate_succ, ← encMask_zero m hm]
    · simp only [Bool.toNat_true] at h; omega

theorem encMask_replicate : ∀ n, encMask (List.replicate n false) = 0
  | 0 => rfl
  | n + 1 => by simp [List.replicate_succ, encMask, encMask_replicate n]

theorem or_bit {e j : ℕ} (h : e < 2 ^ j) : e ||| 1 <<< j = e + 2 ^ j := by
  have := Nat.shiftLeft_add_eq_or_of_lt h 1
  rw [Nat.one_shiftLeft] at this
  rw [Nat.one_shiftLeft, Nat.or_comm, ← this]
  exact Nat.add_comm _ _

theorem length_of_mem_allMasks : ∀ (n : ℕ) (m : List Bool), m ∈ Skip.allMasks n → m.length = n
  | 0, m, h => by
    simp [Skip.allMasks] at h
    simp [h]
  | n + 1, m, h => by
    simp only [Skip.allMasks, List.mem_flatMap] at h
    obtain ⟨m', hm', h⟩ := h
    rcases List.mem_cons.1 h with rfl | h
    · simp [length_of_mem_allMasks n m' hm']
    · rw [List.mem_singleton.1 h]
      simp [length_of_mem_allMasks n m' hm']

theorem foldMask_append {γ δ : Type} (f : δ → γ → δ) (x : γ) (b : Bool) :
    ∀ (xs : List γ) (ms : List Bool) (w : δ), xs.length = ms.length →
      Skip.foldMask f w (xs ++ [x]) (ms ++ [b]) =
        if b then Skip.foldMask f w xs ms else f (Skip.foldMask f w xs ms) x
  | [], [], w, _ => by cases b <;> rfl
  | [], _ :: _, _, h => by simp at h
  | _ :: _, [], _, h => by simp at h
  | y :: xs, c :: ms, w, h => by
    show Skip.foldMask f (if c then w else f w y) (xs ++ [x]) (ms ++ [b]) =
      if b then Skip.foldMask f (if c then w else f w y) xs ms else f (Skip.foldMask f (if c then w else f w y) xs ms) x
    exact foldMask_append f x b xs ms _ (by simpa using h)

theorem wordIn_append (g : List AtomIn) (x : AtomIn) (m : List Bool) (b : Bool) (h : g.length = m.length) :
    Skip.wordIn (g ++ [x]) (m ++ [b]) = if b then Skip.wordIn g m else Skip.step (Skip.wordIn g m) x :=
  foldMask_append Skip.step x b g m 0 h

/-- `L` is the masked words of group prefix `g`: one entry per mask of its length, the mask's bits and its word. -/
def MaskSet (g : List AtomIn) (L : List (ℕ × ℕ)) : Prop :=
  (∀ e ∈ L, ∃ m : List Bool, m.length = g.length ∧ e = (encMask m, Skip.wordIn g m)) ∧
    ∀ m : List Bool, m.length = g.length → (encMask m, Skip.wordIn g m) ∈ L

theorem maskSet_nil : MaskSet [] [(0, 0)] := by
  refine ⟨fun e he => ⟨[], rfl, by rw [List.mem_singleton.1 he]; rfl⟩, fun m hm => ?_⟩
  rw [List.length_eq_zero_iff.1 hm]
  exact List.mem_singleton.2 rfl

/-- One atom's step of `Rule.masked`: each entry runs the atom and passes it through, with bit `n` set. -/
def maskNext (x : AtomIn) (n : ℕ) (W : List (ℕ × ℕ)) : List (ℕ × ℕ) :=
  W.flatMap fun e => [(e.1, Skip.step e.2 x), (e.1 ||| 1 <<< n, e.2)]

theorem maskNext_append (x : AtomIn) (n : ℕ) (W : List (ℕ × ℕ)) (e : ℕ × ℕ) :
    maskNext x n (W ++ [e]) = maskNext x n W ++ [(e.1, Skip.step e.2 x), (e.1 ||| 1 <<< n, e.2)] := by
  simp [maskNext, List.flatMap_append]

theorem maskSet_next (pre : List AtomIn) (x : AtomIn) (W : List (ℕ × ℕ)) (hW : MaskSet pre W) :
    MaskSet (pre ++ [x]) (maskNext x pre.length W) := by
  refine ⟨fun e he => ?_, fun m' hm' => ?_⟩
  · obtain ⟨e0, he0, he⟩ := List.mem_flatMap.1 he
    obtain ⟨m, hm, rfl⟩ := hW.1 e0 he0
    rcases List.mem_cons.1 he with rfl | he
    · refine ⟨m ++ [false], by simp [hm], ?_⟩
      rw [encMask_append, wordIn_append pre x m false hm.symm]
      simp
    · obtain rfl := List.mem_singleton.1 he
      refine ⟨m ++ [true], by simp [hm], ?_⟩
      rw [encMask_append, wordIn_append pre x m true hm.symm]
      show (encMask m ||| 1 <<< pre.length, Skip.wordIn pre m) = _
      rw [← hm, or_bit (encMask_lt m)]
      simp
  · rcases List.eq_nil_or_concat m' with rfl | ⟨m, b, rfl⟩
    · simp at hm'
    · rw [List.concat_eq_append] at hm' ⊢
      have hm : m.length = pre.length := by simpa using hm'
      have h0 := hW.2 m hm
      rw [encMask_append, wordIn_append pre x m b hm.symm]
      refine List.mem_flatMap.2 ⟨_, h0, ?_⟩
      cases b
      · exact List.mem_cons.2 (Or.inl (by simp))
      · refine List.mem_cons.2 (Or.inr (List.mem_singleton.2 ?_))
        show _ = (encMask m ||| 1 <<< pre.length, Skip.wordIn pre m)
        rw [← hm, or_bit (encMask_lt m)]
        simp

theorem inner_loop {p : List ℕ × List ℕ} {x : AtomIn} (hp : AtomRel p x) (n : ℕ) (W : List (ℕ × ℕ))
    {body : ℕ × ℕ → List (ℕ × ℕ) → Except String (ForInStep (List (ℕ × ℕ)))} {s : List (ℕ × ℕ)}
    (hbody : ∀ e acc r, body e acc = .ok r → ∃ d, Rule.step e.2 p.1 p.2 = .ok d ∧
      r = .yield (acc ++ [(e.1, d), (e.1 ||| 1 <<< n, e.2)]))
    (h : forIn W [] body = .ok s) :
    s = maskNext x n W ∧ ∀ e ∈ W, (Skip.step e.2 x >>> 23) % 256 ≠ 255 := by
  refine forIn_list_inv (P := fun pre acc => acc = maskNext x n pre ∧
    ∀ e ∈ pre, (Skip.step e.2 x >>> 23) % 256 ≠ 255) h ⟨rfl, fun e he => by simp at he⟩ ?_
  rintro pre e acc r ⟨hacc, hfin⟩ hr
  obtain ⟨d, hd, rfl⟩ := hbody e acc r hr
  obtain ⟨rfl, hdf⟩ := step_ok hd
  refine ⟨_, rfl, ?_, ?_⟩
  · rw [hacc, maskNext_append, hp e.2]
  · intro e' he'
    rcases List.mem_append.1 he' with he' | he'
    · exact hfin e' he'
    · rw [List.mem_singleton.1 he', ← hp e.2]
      exact hdf

theorem masked_aux {F : List (ℕ × ℕ) → (List ℕ × List ℕ) × ℕ → Except String (List (ℕ × ℕ))}
    (hF : ∀ W p j W' x, AtomRel p x → F W (p, j) = .ok W' →
      W' = maskNext x j W ∧ ∀ e ∈ W, (Skip.step e.2 x >>> 23) % 256 ≠ 255) :
    ∀ {ps : List (List ℕ × List ℕ)} {gr : List AtomIn}, List.Forall₂ AtomRel ps gr →
      ∀ (pre : List AtomIn) (W L : List (ℕ × ℕ)), MaskSet pre W →
        List.foldlM F W (ps.zipIdx pre.length) = .ok L → MaskSet (pre ++ gr) L ∧ ∀ x ∈ gr, Freeze.FiniteAtom x := by
  intro ps gr hR
  induction hR with
  | nil =>
    intro pre W L hW h
    rw [List.zipIdx_nil, List.foldlM_nil] at h
    obtain rfl := WordsSim.pure_ok h
    exact ⟨by rwa [List.append_nil], fun x hx => by simp at hx⟩
  | @cons p x ps gr hpx _ ih =>
    intro pre W L hW h
    rw [List.zipIdx_cons, List.foldlM_cons] at h
    obtain ⟨W', hW', h⟩ := WordsSim.bind_ok h
    obtain ⟨rfl, hfin⟩ := hF W p pre.length W' x hpx hW'
    obtain ⟨h1, h2⟩ := ih (pre ++ [x]) _ L (maskSet_next pre x W hW)
      (by rwa [List.length_append, List.length_singleton])
    refine ⟨by rwa [List.append_assoc, List.singleton_append] at h1, fun y hy => ?_⟩
    rcases List.mem_cons.1 hy with rfl | hy
    · by_contra hnf
      have hmem := hW.2 (List.replicate pre.length false) (by simp)
      have := hfin _ hmem
      rw [step_nan _ _ hnf] at this
      exact this (by decide)
    · exact h2 y hy

/-- **`Rule.masked` on a group's slices** is every mask's word of the game's group, and the group's atoms are finite. -/
theorem masked_spec {ps : List (List ℕ × List ℕ)} {g : List AtomIn} {L : List (ℕ × ℕ)}
    (hR : List.Forall₂ AtomRel ps g) (h : Rule.masked ps = .ok L) :
    MaskSet g L ∧ ∀ x ∈ g, Freeze.FiniteAtom x := by
  unfold Rule.masked at h
  try simp only at h
  obtain ⟨h1, h2⟩ := masked_aux (fun W p j W' x hpx hW' => by
    obtain ⟨s, hs, hW'⟩ := WordsSim.bind_ok hW'
    obtain rfl := WordsSim.pure_ok hW'
    exact inner_loop hpx j W (fun e acc r hr => by
      obtain ⟨d, hd, hr⟩ := WordsSim.bind_ok hr
      exact ⟨d, hd, (WordsSim.pure_ok hr).symm⟩) hs) hR [] [(0, 0)] L maskSet_nil h
  exact ⟨h1, h2⟩

theorem lookup_some (k0 v : ℕ) : ∀ L : List (ℕ × ℕ), (∀ e ∈ L, e.1 = k0 → e.2 = v) → (∃ e ∈ L, e.1 = k0) →
    L.lookup k0 = some v
  | [], _, ⟨_, he, _⟩ => absurd he (by simp)
  | (a, b) :: L, h1, h2 => by
    rw [List.lookup_cons]
    by_cases hk : k0 = a
    · rw [show (k0 == a) = true by simpa using hk]
      exact congrArg some (h1 (a, b) List.mem_cons_self hk.symm)
    · rw [show (k0 == a) = false by simpa using hk]
      refine lookup_some k0 v L (fun e he => h1 e (List.mem_cons_of_mem _ he)) ?_
      obtain ⟨e, he, he1⟩ := h2
      rcases List.mem_cons.1 he with rfl | he
      · exact absurd he1.symm hk
      · exact ⟨e, he, he1⟩

/-- The check's `base`, the mask-0 word, is the group's word. -/
theorem base_eq {g : List AtomIn} {L : List (ℕ × ℕ)} (hL : MaskSet g L) : (L.lookup 0).getD 0 = groupWord g := by
  have h0 : L.lookup 0 = some (groupWord g) := by
    refine lookup_some 0 _ L (fun e he he0 => ?_) ⟨_, hL.2 (List.replicate g.length false) (by simp), ?_⟩
    · obtain ⟨m, hm, rfl⟩ := hL.1 e he
      have := encMask_zero m he0
      rw [this, hm]
      exact Freeze.foldMask_replicate Skip.step 0 g
    · exact encMask_replicate g.length
  rw [h0, Option.getD_some]

/-! ## Groups and totals -/

theorem term_abs_lt (t : Term) (hm : t.mag < 2 ^ 24) (he : t.exp ≤ 44) : |t.value| < 2 ^ 46 := by
  have h1 : (2 : ℚ) ^ (t.exp - 23) ≤ 2 ^ (21 : ℤ) := zpow_le_zpow_right₀ (by norm_num) (by omega)
  have h2 : (t.mag : ℚ) < 2 ^ 24 := by exact_mod_cast hm
  have h3 : (0 : ℚ) < 2 ^ (t.exp - 23) := zpow_pos (by norm_num) _
  have h4 : (t.mag : ℚ) * 2 ^ (t.exp - 23) < 2 ^ 46 := by
    calc (t.mag : ℚ) * 2 ^ (t.exp - 23) ≤ (t.mag : ℚ) * 2 ^ (21 : ℤ) :=
          mul_le_mul_of_nonneg_left h1 (Nat.cast_nonneg _)
      _ < 2 ^ 24 * 2 ^ (21 : ℤ) := mul_lt_mul_of_pos_right h2 (zpow_pos (by norm_num) _)
      _ ≤ 2 ^ 46 := by norm_num
  have h5 : (0 : ℚ) ≤ (t.mag : ℚ) * 2 ^ (t.exp - 23) := mul_nonneg (Nat.cast_nonneg _) h3.le
  unfold Term.value
  rw [abs_lt]
  constructor <;> split_ifs <;> linarith

/-- A group of four finite atoms has a finite word below 2⁴⁶ in magnitude. -/
theorem groupWord_bound (g : List AtomIn) (hfin : ∀ x ∈ g, Freeze.FiniteAtom x) (hl : g.length = 4) :
    ∃ y, wordQ (groupWord g) = some y ∧ |y| < 2 ^ 46 := by
  have ht := Freeze.tame_inWord g (by omega) hfin 4 (by omega)
  have he : Freeze.inWord g 4 = groupWord g := by
    unfold Freeze.inWord
    rw [List.take_of_length_le (by omega)]
    rfl
  rw [he] at ht
  rcases ht with h0 | ⟨tc, htc, -, -, hexp⟩
  · exact ⟨0, by rw [h0]; exact Freeze.wordQ_zero, by norm_num⟩
  · refine ⟨tc.value, ?_, term_abs_lt tc (Freeze.fp32Value_fin htc).1 (by omega)⟩
    simp [wordQ, htc, Value.finite?]

theorem inTotal_succ (gs : List (List AtomIn)) (p : ℕ) (hp : p < gs.length) :
    Skip.inTotal gs (p + 1) = Pouw.PearlC.promote (Skip.inTotal gs p) (groupWord (gs.getD p [])) := by
  unfold Skip.inTotal
  rw [List.take_add_one, List.getElem?_eq_getElem hp, Option.toList_some, List.foldl_append,
    List.getD_eq_getElem gs [] hp]
  rfl

theorem total_eq_inTotal (gs : List (List AtomIn)) : Pouw.PearlC.total gs = Skip.inTotal gs gs.length := by
  rw [Freeze.total_eq_foldl]
  unfold Skip.inTotal
  rw [List.take_length]

/-! ## Bands: the totals before a group that promote to the word's total -/

/-- The groups from `g` on promoted into a total of value `v`. -/
def laterFrom (gs : List (List AtomIn)) (g : ℕ) (v : ℚ) : Option ℚ :=
  ((gs.drop g).map groupWord).foldl Pouw.PearlC.promote (some v)

theorem laterFrom_step (gs : List (List AtomIn)) (g : ℕ) (hg : g < gs.length) (v y : ℚ)
    (hy : wordQ (groupWord (gs.getD g [])) = some y) : laterFrom gs g v = laterFrom gs (g + 1) (rnAdd v y) := by
  unfold laterFrom
  rw [List.drop_eq_getElem_cons hg, List.map_cons, List.foldl_cons, ← List.getD_eq_getElem gs [] hg, promote_some hy]

/-- `b` is the band of keys before group `g`: a key in [−TOP, TOP] is in it exactly when the groups from `g` on promote
its value to the word's total. -/
def KeyBand (gs : List (List AtomIn)) (g : ℕ) (b : ℤ × ℤ) : Prop :=
  ∀ x, -Rule.TOP ≤ x → x ≤ Rule.TOP → (b.1 ≤ x ∧ x ≤ b.2 ↔ laterFrom gs g (keyVal x) = Pouw.PearlC.total gs)

theorem band_last (hK : KeyIsTheValue) (gs : List (List AtomIn)) {T : ℕ} (hT : AccumProofs.StepWord T)
    (hTv : wordQ T = Pouw.PearlC.total gs) : KeyBand gs gs.length (Rule.key T, Rule.key T) := by
  intro x _ _
  have hfe := AccumProofs.finite?_exp hT.2.2
  obtain ⟨hq, -, -⟩ := hK.2.1 T hT.1 hfe
  have e : laterFrom gs gs.length (keyVal x) = some (keyVal x) := by simp [laterFrom]
  rw [e, ← hTv, hq]
  constructor
  · rintro ⟨h1, h2⟩
    rw [show x = Rule.key T by dsimp only at h1 h2; omega]
  · intro h
    have := hK.1.injective (Option.some.inj h)
    exact ⟨le_of_eq this.symm, le_of_eq this⟩

/-- **One step back**: `least` over `carried` through group `g`'s word turns the band after it into the band before. -/
theorem band_step (hK : KeyIsTheValue) (hL : LeastIsTheThreshold) (hC : CarriedIsTheAdd)
    (gs : List (List AtomIn)) (g : ℕ) (hg : g < gs.length) {w : ℕ} {y : ℚ}
    (hw : w = groupWord (gs.getD g [])) (hy : wordQ (groupWord (gs.getD g [])) = some y) (hy46 : |y| < 2 ^ 46)
    {lo0 hi0 : ℤ} (h : KeyBand gs (g + 1) (lo0, hi0)) :
    KeyBand gs g (Rule.least (fun x => decide (Rule.carried x w ≥ lo0)),
      Rule.least (fun x => decide (Rule.carried x w > hi0)) - 1) := by
  have hyw : wordQ w = some y := by rw [hw]; exact hy
  have hcar := hC addAgrees
  have hmono : ∀ x x', -Rule.TOP ≤ x → x ≤ x' → x' ≤ Rule.TOP → Rule.carried x w ≤ Rule.carried x' w := by
    intro x x' h1 h2 h3
    obtain ⟨-, -, e1⟩ := hcar x w y h1 (by omega) hyw hy46
    obtain ⟨-, -, e2⟩ := hcar x' w y (by omega) h3 hyw hy46
    rw [← hK.1.le_iff_le, e1, e2]
    exact Freeze.rnAdd_mono_left y (hK.1.monotone h2)
  have hl := hL (fun x => decide (Rule.carried x w ≥ lo0)) (fun x x' h1 h2 h3 hp => by
    simp only [decide_eq_true_eq, ge_iff_le] at hp ⊢
    exact le_trans hp (hmono x x' h1 h2 h3))
  have hu := hL (fun x => decide (Rule.carried x w > hi0)) (fun x x' h1 h2 h3 hp => by
    simp only [decide_eq_true_eq, gt_iff_lt] at hp ⊢
    exact lt_of_lt_of_le hp (hmono x x' h1 h2 h3))
  intro x hx1 hx2
  obtain ⟨hc1, hc2, hcv⟩ := hcar x w y hx1 hx2 hyw hy46
  have e : laterFrom gs g (keyVal x) = laterFrom gs (g + 1) (keyVal (Rule.carried x w)) := by
    rw [hcv]; exact laterFrom_step gs g hg (keyVal x) y hy
  rw [e, ← h (Rule.carried x w) hc1 hc2]
  dsimp only
  rw [hl.2.2 x hx1 hx2, Int.le_sub_one_iff, ← not_le, hu.2.2 x hx1 hx2]
  simp only [decide_eq_true_eq, not_lt, ge_iff_le, gt_iff_lt]

/-! ## Counting -/

theorem length_filter_range (q : ℕ → Bool) : ∀ n : ℕ,
    ((List.range n).filter q).length = ((Finset.range n).filter fun g => q g = true).card
  | 0 => by simp
  | n + 1 => by
    rw [List.range_succ, List.filter_append, List.length_append, length_filter_range q n, Finset.range_add_one,
      Finset.filter_insert]
    by_cases hq : q n = true
    · rw [ite_eq_left hq, Finset.card_insert_of_notMem]
      · simp [hq]
      · simp
    · rw [ite_eq_right hq]
      simp [hq]

theorem count_shift (q : ℕ → Bool) (P : ℕ → Prop) {_ : DecidablePred P} (M : ℕ)
    (hqP : ∀ g, g < M - 1 → (q g = true ↔ P (g + 1))) :
    ((List.range (M - 1)).filter q).length = ((Finset.Ico 1 M).filter P).card := by
  rw [length_filter_range]
  refine Finset.card_nbij' (· + 1) (· - 1) ?_ ?_ ?_ ?_
  · intro g hg
    simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_range, Finset.mem_Ico] at hg ⊢
    exact ⟨⟨by omega, by omega⟩, (hqP g hg.1).1 hg.2⟩
  · intro g hg
    simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_range, Finset.mem_Ico] at hg ⊢
    have h1 : g - 1 + 1 = g := by omega
    exact ⟨by omega, (hqP (g - 1) (by omega)).2 (by rw [h1]; exact hg.2)⟩
  · intro g _
    show g + 1 - 1 = g
    omega
  · intro g hg
    simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_Ico] at hg
    show g - 1 + 1 = g
    omega

theorem toChunks_take {γ : Type} (l : List γ) (g : ℕ) : (l.take (4 * g)).toChunks 4 = (l.toChunks 4).take g := by
  apply List.ext_getElem
  · rw [Freeze.length_toChunks (G := 4) (by norm_num), List.length_take, List.length_take,
      Freeze.length_toChunks (G := 4) (by norm_num)]
    omega
  · intro t h1 h2
    have htg : t < g := by
      rw [List.length_take] at h2
      omega
    rw [Freeze.getElem_toChunks (G := 4) (by norm_num), List.getElem_take,
      Freeze.getElem_toChunks (G := 4) (by norm_num), List.drop_take, List.take_take]
    congr 1
    omega

theorem rowAtoms_prefix {m n k : ℕ} (A : Matrix (Fin m) (Fin k) Code) (B : Matrix (Fin n) (Fin k) Code)
    (i : Fin m) (j : Fin n) {k' : ℕ} (h : k' ≤ k) :
    rowAtoms (prefixCols A k' h i) (prefixCols B k' h j) = (rowAtoms (A i) (B j)).take (k' / 32) := by
  apply List.ext_getElem
  · simp only [rowAtoms, List.length_ofFn, List.length_take]
    omega
  · intro t h1 h2
    simp only [rowAtoms, List.getElem_ofFn, List.getElem_take]
    rfl

/-- The H100 record's ticket on the first `128·g` columns is the total after `g` groups. -/
theorem ticket_prefix {m n k : ℕ} (A : Matrix (Fin m) (Fin k) Code) (B : Matrix (Fin n) (Fin k) Code)
    (i : Fin m) (j : Fin n) (g : ℕ) (h : 128 * g ≤ k) :
    devH100.ticket (prefixCols A (128 * g) h) (prefixCols B (128 * g) h) i j =
      Skip.inTotal ((rowAtoms (A i) (B j)).toChunks 4) g := by
  show Pouw.PearlC.total ((rowAtoms (prefixCols A (128 * g) h i) (prefixCols B (128 * g) h j)).toChunks 4) = _
  rw [rowAtoms_prefix A B i j h, show 128 * g / 32 = 4 * g by omega, toChunks_take]
  exact Freeze.total_eq_foldl _

theorem unitFlags_eq {m n k : ℕ} (A : Matrix (Fin m) (Fin k) Code) (B : Matrix (Fin n) (Fin k) Code)
    (i : Fin m) (j : Fin n) (g jj : ℕ) (hjj : jj < 4) :
    Skip.unitFlags A B i j (4 * g + jj) = (Skip.jointFlags ((rowAtoms (A i) (B j)).toChunks 4) g).getD jj false := by
  show (Skip.jointFlags ((rowAtoms (A i) (B j)).toChunks 4) ((4 * g + jj) / 4)).getD ((4 * g + jj) % 4) false = _
  rw [show (4 * g + jj) / 4 = g by omega, show (4 * g + jj) % 4 = jj by omega]

theorem getD_map_range {γ : Type} (F : ℕ → γ) (d : γ) {jj n : ℕ} (h : jj < n) :
    ((List.range n).map F).getD jj d = F jj := by
  rw [List.getD_eq_getElem _ _ (by simpa using h), List.getElem_map, List.getElem_range]

/-! ## Flags: a kept mask is `Skip.keeps` -/

theorem keepsLoop {T : ℕ} {lo hi : ℤ} {f : ℕ × ℕ → List ℕ → Except String (ForInStep (List ℕ))}
    (L : List (ℕ × ℕ)) {keeps : List ℕ} (h : forIn L [] f = .ok keeps)
    (hf : ∀ x acc r, f x acc = .ok r → (x.1 = 0 ∧ r = .yield acc) ∨ (x.1 ≠ 0 ∧ ∃ o, Rule.promote T x.2 = .ok o ∧
      ((¬ (lo ≤ Rule.key o ∧ Rule.key o ≤ hi) ∧ r = .yield acc) ∨
        (lo ≤ Rule.key o ∧ Rule.key o ≤ hi ∧ r = .yield (x.1 :: acc))))) :
    (∀ mask, mask ∈ keeps ↔ ∃ w o, (mask, w) ∈ L ∧ mask ≠ 0 ∧ Rule.promote T w = .ok o ∧
      lo ≤ Rule.key o ∧ Rule.key o ≤ hi) ∧ ∀ e ∈ L, e.1 ≠ 0 → ∃ o, Rule.promote T e.2 = .ok o := by
  refine forIn_list_inv (P := fun pre acc => (∀ mask, mask ∈ acc ↔ ∃ w o, (mask, w) ∈ pre ∧ mask ≠ 0 ∧
    Rule.promote T w = .ok o ∧ lo ≤ Rule.key o ∧ Rule.key o ≤ hi) ∧
      ∀ e ∈ pre, e.1 ≠ 0 → ∃ o, Rule.promote T e.2 = .ok o) h ⟨fun mask => by simp, fun e he => by simp at he⟩ ?_
  rintro pre x acc r ⟨hacc, hok⟩ hr
  rcases hf x acc r hr with ⟨h0, rfl⟩ | ⟨h0, o, ho, ⟨hout, rfl⟩ | ⟨h1, h2, rfl⟩⟩
  · refine ⟨_, rfl, fun mask => ?_, fun e he hne => ?_⟩
    · rw [hacc]
      constructor
      · rintro ⟨w, o', hm, hne, hp⟩
        exact ⟨w, o', List.mem_append_left _ hm, hne, hp⟩
      · rintro ⟨w, o', hm, hne, hp⟩
        rcases List.mem_append.1 hm with hm | hm
        · exact ⟨w, o', hm, hne, hp⟩
        · obtain rfl := List.mem_singleton.1 hm
          exact absurd h0 hne
    · rcases List.mem_append.1 he with he | he
      · exact hok e he hne
      · obtain rfl := List.mem_singleton.1 he
        exact absurd h0 hne
  · refine ⟨_, rfl, fun mask => ?_, fun e he hne => ?_⟩
    · rw [hacc]
      constructor
      · rintro ⟨w, o', hm, hne, hp⟩
        exact ⟨w, o', List.mem_append_left _ hm, hne, hp⟩
      · rintro ⟨w, o', hm, hne, hp, hb1, hb2⟩
        rcases List.mem_append.1 hm with hm | hm
        · exact ⟨w, o', hm, hne, hp, hb1, hb2⟩
        · obtain rfl := List.mem_singleton.1 hm
          obtain rfl := Except.ok.inj (ho.symm.trans hp)
          exact (hout ⟨hb1, hb2⟩).elim
    · rcases List.mem_append.1 he with he | he
      · exact hok e he hne
      · obtain rfl := List.mem_singleton.1 he
        exact ⟨o, ho⟩
  · refine ⟨_, rfl, fun mask => ?_, fun e he hne => ?_⟩
    · rw [List.mem_cons, hacc]
      constructor
      · rintro (rfl | ⟨w, o', hm, hne, hp⟩)
        · exact ⟨x.2, o, List.mem_append_right _ (List.mem_singleton.2 rfl), h0, ho, h1, h2⟩
        · exact ⟨w, o', List.mem_append_left _ hm, hne, hp⟩
      · rintro ⟨w, o', hm, hne, hp⟩
        rcases List.mem_append.1 hm with hm | hm
        · exact Or.inr ⟨w, o', hm, hne, hp⟩
        · obtain rfl := List.mem_singleton.1 hm
          exact Or.inl rfl
    · rcases List.mem_append.1 he with he | he
      · exact hok e he hne
      · obtain rfl := List.mem_singleton.1 he
        exact ⟨o, ho⟩

/-- A masked word promoted into the total before group `g` lands in the band after it exactly when `Skip.keeps`. -/
theorem keep_iff (hK : KeyIsTheValue) (gs : List (List AtomIn)) (g : ℕ) (hg : g < gs.length) {T : ℕ}
    (hT : wordQ T = Skip.inTotal gs g) {b : ℤ × ℤ} (hband : KeyBand gs (g + 1) b) (m : List Bool) {w o : ℕ}
    (hw : w = Skip.wordIn (gs.getD g []) m) (ho : Rule.promote T w = .ok o) :
    (b.1 ≤ Rule.key o ∧ Rule.key o ≤ b.2) ↔
      Skip.keeps (gs.getD g []) (Skip.inTotal gs g) (Skip.laterWords gs g) m = true := by
  obtain ⟨t, y, ht, hy, hso, hov⟩ := promote_ok ho
  have hfe := AccumProofs.finite?_exp hso.2.2
  obtain ⟨hv, hk1, hk2⟩ := hK.2.1 o hso.1 hfe
  have e2 : (Skip.laterWords gs g).foldl Pouw.PearlC.promote
      (Pouw.PearlC.promote (Skip.inTotal gs g) (groupWord (gs.getD g []))) = Pouw.PearlC.total gs := by
    rw [Freeze.total_split gs g hg, List.getD_eq_getElem gs [] hg]
  have e3 : (Skip.laterWords gs g).foldl Pouw.PearlC.promote
      (Pouw.PearlC.promote (Skip.inTotal gs g) (Skip.wordIn (gs.getD g []) m)) =
        laterFrom gs (g + 1) (keyVal (Rule.key o)) := by
    rw [← hT, ht, ← hw, promote_some hy, ← Option.some.inj (hov.symm.trans hv)]
    rfl
  unfold Skip.keeps
  rw [decide_eq_true_iff, e3, e2]
  exact hband (Rule.key o) hk1 hk2

/-- **A group's flag** `jj` in the replay (some kept mask has bit `jj`) is its joint flag. -/
theorem flag_iff (hK : KeyIsTheValue) (gs : List (List AtomIn)) (g : ℕ) (hg : g < gs.length)
    (hl : (gs.getD g []).length = 4) {L : List (ℕ × ℕ)} (hMS : MaskSet (gs.getD g []) L)
    {T : ℕ} (hT : wordQ T = Skip.inTotal gs g) {b : ℤ × ℤ} (hband : KeyBand gs (g + 1) b) {keeps : List ℕ}
    (hkeeps : ∀ mask, mask ∈ keeps ↔ ∃ w o, (mask, w) ∈ L ∧ mask ≠ 0 ∧ Rule.promote T w = .ok o ∧
      b.1 ≤ Rule.key o ∧ Rule.key o ≤ b.2)
    (hok : ∀ e ∈ L, e.1 ≠ 0 → ∃ o, Rule.promote T e.2 = .ok o) (jj : ℕ) (hjj : jj < 4) :
    (keeps.any fun mask => mask >>> jj &&& 1 == 1) = true ↔ (Skip.jointFlags gs g).getD jj false = true := by
  unfold Skip.jointFlags Skip.jointAt
  rw [hl, getD_map_range _ _ hjj, List.any_eq_true, List.any_eq_true]
  constructor
  · rintro ⟨mask, hmem, hbit⟩
    obtain ⟨w, o, hmw, hne, ho, hb1, hb2⟩ := (hkeeps mask).1 hmem
    obtain ⟨m, hm, he⟩ := hMS.1 _ hmw
    simp only [Prod.mk.injEq] at he
    obtain ⟨rfl, rfl⟩ := he
    have hbit' : m.getD jj false = true := by
      have := encMask_bit m jj
      rw [beq_iff_eq.1 hbit] at this
      revert this
      cases m.getD jj false <;> simp
    refine ⟨m, ?_, ?_⟩
    · rw [hl] at hm
      rw [← hm]
      exact Freeze.mem_allMasks m
    · rw [hbit', Bool.true_and]
      exact (keep_iff hK gs g hg hT hband m rfl ho).1 ⟨hb1, hb2⟩
  · rintro ⟨m, hmA, hm⟩
    rw [Bool.and_eq_true] at hm
    have hlen : m.length = (gs.getD g []).length := by rw [hl]; exact length_of_mem_allMasks 4 m hmA
    have hmem := hMS.2 m hlen
    have hne : encMask m ≠ 0 := by
      intro h0
      have := encMask_bit m jj
      rw [h0, Nat.zero_shiftRight, hm.1] at this
      simp at this
    obtain ⟨o, ho⟩ := hok _ hmem hne
    refine ⟨encMask m, (hkeeps _).2 ⟨_, o, hmem, hne, ho, (keep_iff hK gs g hg hT hband m rfl ho).2 hm.2⟩, ?_⟩
    rw [beq_iff_eq, encMask_bit, hm.1, Bool.toNat_true]

theorem flags_push (c : Array Bool) (b : Bool) (N : ℕ) (q : ℕ → Bool) (hs : c.size = N)
    (hg : ∀ t < N, c[t]! = q t) (hb : b = q N) : (c.push b).size = N + 1 ∧ ∀ t < N + 1, (c.push b)[t]! = q t := by
  refine ⟨by rw [Array.size_push, hs], fun t ht => ?_⟩
  rw [push_get c b t (by omega)]
  by_cases htp : t = c.size
  · rw [ite_eq_left htp, htp, hs, hb]
  · rw [ite_eq_right htp]
    exact hg t (by omega)

theorem count_push (c : Array Bool) (b : Bool) (N : ℕ) (q : ℕ → Bool)
    (hn : (c.filter id).size = ((Finset.range N).filter fun t => q t = true).card) (hb : b = q N) :
    ((c.push b).filter id).size = ((Finset.range (N + 1)).filter fun t => q t = true).card := by
  subst hb
  have hp : ((c.push (q N)).filter id).size = (c.filter id).size + (if q N = true then 1 else 0) := by
    cases h : q N <;> simp [Array.filter_push, h]
  rw [hp, hn, Finset.range_add_one, Finset.filter_insert]
  cases h : q N
  · rw [ite_eq_right (by simp), ite_eq_right (by simp), Nat.add_zero]
  · rw [ite_eq_left rfl, ite_eq_left rfl, Finset.card_insert_of_notMem]
    simp

end DebitSim

open DebitSim in
/-- **The atom flags the check replays are the game's**, from the keys' three statements: where `word_flags` is a value
on a word's two rows of codes, its flags are `Skip.unitFlags`' at each of the row's `k/32` atom blocks, so many of them
are set as the game flags, and its later promotions into a +0 total are `laterZeroPromos`' at the H100 record. -/
theorem wordFlagsIsTheReplayOf (hK : KeyIsTheValue) (hL : LeastIsTheThreshold) (hC : CarriedIsTheAdd) {m n k : ℕ}
    (A : Matrix (Fin m) (Fin k) Code) (B : Matrix (Fin n) (Fin k) Code) (i : Fin m) (j : Fin n) {fl : Array Bool}
    {z : ℕ} (h : Rule.wordFlags (Array.ofFn fun l => (A i l).val) (Array.ofFn fun l => (B j l).val) = .ok (fl, z)) :
    fl.size = k / 32 ∧ (∀ t < k / 32, fl[t]! = Skip.unitFlags A B i j t) ∧
      (fl.filter id).size = ((Finset.range (k / 32)).filter fun t => Skip.unitFlags A B i j t = true).card ∧
      z = laterZeroPromos devH100 A B i j := by
  have hG : PouwCheck.PearlPlus.G = 4 := rfl
  unfold Rule.wordFlags at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', range_size, Array.size_ofFn, hG, Nat.reduceMul] at h
  split at h
  · obtain ⟨_, ha, -⟩ := WordsSim.bind_ok h
    exact (WordsSim.throw_ok ha).elim
  rename_i hcond
  have hk : k % 128 = 0 := by
    by_contra hne
    exact hcond (by simp [hne])
  obtain ⟨gs, hgs⟩ : ∃ gs, gs = (rowAtoms (A i) (B j)).toChunks 4 := ⟨_, rfl⟩
  have hgl : gs.length = k / 128 := by rw [hgs]; exact chunks_len (A i) (B j) hk
  have hlen4 : ∀ g < k / 128, (gs.getD g []).length = 4 := fun g hg => by
    rw [hgs]; exact chunk_len (A i) (B j) hk g hg
  -- the groups' masked words
  obtain ⟨groups, hgr, h⟩ := WordsSim.bind_ok h
  obtain ⟨-, hgall⟩ := WordsSim.mapM_ok hgr
  have hgrp : ∀ g < k / 128, MaskSet (gs.getD g []) groups[g]! ∧ ∀ x ∈ gs.getD g [], Freeze.FiniteAtom x := by
    intro g hg
    have h1 := hgall g (by simpa using hg)
    have hidx : (List.range (k / 128)).toArray[g]! = g := by
      rw [getElem!_pos (List.range (k / 128)).toArray g (by simpa using hg), List.getElem_toArray,
        List.getElem_range]
    rw [hidx] at h1
    rw [hgs]
    exact masked_spec (group_rel (A i) (B j) hk g hg) h1
  have hbase : ∀ g < k / 128, (List.lookup 0 groups[g]!).getD 0 = groupWord (gs.getD g []) :=
    fun g hg => base_eq (hgrp g hg).1
  have hgw : ∀ g < k / 128, ∃ y, wordQ (groupWord (gs.getD g [])) = some y ∧ |y| < 2 ^ 46 :=
    fun g hg => groupWord_bound _ (hgrp g hg).2 (hlen4 g hg)
  -- the running totals
  obtain ⟨totals, htot, h⟩ := WordsSim.bind_ok h
  have hT : totals.size = k / 128 + 1 ∧ ∀ q ≤ k / 128, AccumProofs.StepWord totals[q]! ∧
      wordQ totals[q]! = Skip.inTotal gs q := by
    refine forIn_range_inv (P := fun p (s : Array ℕ) => s.size = p + 1 ∧ ∀ q ≤ p, AccumProofs.StepWord s[q]! ∧
      wordQ s[q]! = Skip.inTotal gs q) htot ⟨rfl, fun q hq => ?_⟩ ?_
    · obtain rfl : q = 0 := by omega
      rw [singleton_get]
      exact ⟨stepWord_zero, Freeze.wordQ_zero⟩
    · rintro p s r hp ⟨hs, hq⟩ hr
      obtain ⟨o, ho, hr⟩ := WordsSim.bind_ok hr
      obtain rfl := WordsSim.pure_ok hr
      rw [hbase p hp] at ho
      obtain ⟨t, y, ht, hy, hso, hov⟩ := promote_ok ho
      refine ⟨_, rfl, by rw [Array.size_push, hs], fun q hq' => ?_⟩
      rw [push_get s o q (by omega), hs]
      by_cases hqp : q = p + 1
      · rw [ite_eq_left hqp, hqp, inTotal_succ gs p (by omega), ← (hq p le_rfl).2, ht, promote_some hy]
        exact ⟨hso, hov⟩
      · rw [ite_eq_right hqp]
        exact hq q (by omega)
  have hTs := (hT.2 _ le_rfl).1
  have hTv : wordQ totals[k / 128]! = Pouw.PearlC.total gs := by
    rw [(hT.2 _ le_rfl).2, total_eq_inTotal, hgl]
  -- the bands, from the last group back
  obtain ⟨bnd, hbnd, h⟩ := WordsSim.bind_ok h
  have hB : bnd.2.2.size = k / 128 - 1 + 1 ∧ (∀ q ≤ k / 128 - 1, KeyBand gs (k / 128 - q) bnd.2.2[q]!) ∧
      (bnd.1, bnd.2.1) = bnd.2.2[k / 128 - 1]! := by
    refine forIn_range_inv (P := fun p (s : ℤ × ℤ × Array (ℤ × ℤ)) => s.2.2.size = p + 1 ∧
      (∀ q ≤ p, KeyBand gs (k / 128 - q) s.2.2[q]!) ∧ (s.1, s.2.1) = s.2.2[p]!) hbnd
      ⟨rfl, fun q hq => ?_, (singleton_get _).symm⟩ ?_
    · obtain rfl : q = 0 := by omega
      show KeyBand gs (k / 128 - 0) (#[(Rule.key totals[k / 128]!, Rule.key totals[k / 128]!)])[0]!
      rw [singleton_get, Nat.sub_zero]
      have := band_last hK gs hTs hTv
      rwa [hgl] at this
    · rintro p s r hp ⟨hs, hb, hcur⟩ hr
      obtain rfl := WordsSim.pure_ok hr
      obtain ⟨y, hy, hy46⟩ := hgw (k / 128 - 1 - p) (by omega)
      have hb' := hb p le_rfl
      rw [← hcur, show k / 128 - p = k / 128 - 1 - p + 1 by omega] at hb'
      have hnew := band_step hK hL hC gs (k / 128 - 1 - p) (by omega) (hbase _ (by omega)) hy hy46 hb'
      refine ⟨_, rfl, ?_⟩
      dsimp only
      refine ⟨?_, fun q hq => ?_, ?_⟩
      · rw [Array.size_push]
        omega
      · rw [push_get s.2.2 _ q (by omega), hs]
        by_cases hqp : q = p + 1
        · rw [ite_eq_left hqp, hqp, show k / 128 - (p + 1) = k / 128 - 1 - p by omega]
          exact hnew
        · rw [ite_eq_right hqp]
          exact hb q (by omega)
      · rw [push_get s.2.2 _ (p + 1) (by omega), ite_eq_left hs.symm]
  have hrev : ∀ g < k / 128, KeyBand gs (g + 1) bnd.2.2.reverse[g]! := by
    intro g hg
    rw [rev_get _ g (by rw [hB.1]; omega), hB.1]
    have := hB.2.1 (k / 128 - 1 + 1 - 1 - g) (by omega)
    rwa [show k / 128 - (k / 128 - 1 + 1 - 1 - g) = g + 1 by omega] at this
  -- the flagged atoms
  obtain ⟨fl0, hfl, h⟩ := WordsSim.bind_ok h
  have hfz := WordsSim.pure_ok h
  simp only [Prod.mk.injEq] at hfz
  obtain ⟨hf, hz⟩ := hfz
  have hF : fl0.size = 4 * (k / 128) ∧ (∀ t < 4 * (k / 128), fl0[t]! = Skip.unitFlags A B i j t) ∧
      (fl0.filter id).size = ((Finset.range (4 * (k / 128))).filter fun t => Skip.unitFlags A B i j t = true).card := by
    refine forIn_range_inv (P := fun p (c : Array Bool) => c.size = 4 * p ∧
      (∀ t < 4 * p, c[t]! = Skip.unitFlags A B i j t) ∧
      (c.filter id).size = ((Finset.range (4 * p)).filter fun t => Skip.unitFlags A B i j t = true).card) hfl
      ⟨rfl, fun t ht => by omega, by simp⟩ ?_
    rintro p c r hp hc hr
    obtain ⟨keeps, hkeeps, hr⟩ := WordsSim.bind_ok hr
    have hgp : p < gs.length := by omega
    obtain ⟨hMS, -⟩ := hgrp p hp
    have hTp : wordQ totals[p]! = Skip.inTotal gs p := (hT.2 p (by omega)).2
    obtain ⟨hkeep, hok⟩ := keepsLoop (T := totals[p]!) (lo := bnd.2.2.reverse[p]!.1)
      (hi := bnd.2.2.reverse[p]!.2) groups[p]! hkeeps (fun x acc r hr => by
        split at hr
        · rename_i hx
          obtain ⟨o, ho, hr⟩ := WordsSim.bind_ok hr
          split at hr
          · rename_i hb
            simp only [Bool.and_eq_true, decide_eq_true_eq] at hb
            exact Or.inr ⟨by simpa using hx, o, ho, Or.inr ⟨hb.1, hb.2, (WordsSim.pure_ok hr).symm⟩⟩
          · rename_i hb
            simp only [Bool.and_eq_true, decide_eq_true_eq] at hb
            exact Or.inr ⟨by simpa using hx, o, ho, Or.inl ⟨hb, (WordsSim.pure_ok hr).symm⟩⟩
        · rename_i hx
          exact Or.inl ⟨by simpa using hx, (WordsSim.pure_ok hr).symm⟩)
    obtain ⟨c', hc', hr⟩ := WordsSim.bind_ok hr
    obtain rfl := WordsSim.pure_ok hr
    have hinner := forIn_range_inv (P := fun jj (c2 : Array Bool) => c2.size = 4 * p + jj ∧
      (∀ t < 4 * p + jj, c2[t]! = Skip.unitFlags A B i j t) ∧
      (c2.filter id).size = ((Finset.range (4 * p + jj)).filter fun t => Skip.unitFlags A B i j t = true).card) hc'
      (by rw [Nat.add_zero]; exact hc) (fun jj c2 r2 hjj hc2 hr2 => by
        have hu : Skip.unitFlags A B i j (4 * p + jj) = true ↔
            (keeps.any fun mask => mask >>> jj &&& 1 == 1) = true := by
          rw [unitFlags_eq A B i j p jj hjj, ← hgs]
          exact (flag_iff hK gs p hgp (hlen4 p hp) hMS hTp (hrev p hp) hkeep hok jj hjj).symm
        have hb : (keeps.any fun mask => mask >>> jj &&& 1 == 1) = Skip.unitFlags A B i j (4 * p + jj) :=
          Bool.eq_iff_iff.2 hu.symm
        obtain ⟨hs2, hg2, hn2⟩ := hc2
        obtain rfl := WordsSim.pure_ok hr2
        refine ⟨_, rfl, ?_⟩
        rw [show 4 * p + (jj + 1) = 4 * p + jj + 1 by omega]
        exact ⟨(flags_push c2 _ _ _ hs2 hg2 hb).1, (flags_push c2 _ _ _ hs2 hg2 hb).2,
          count_push c2 _ _ _ hn2 hb⟩)
    refine ⟨_, rfl, ?_⟩
    rwa [show 4 * p + 4 = 4 * (p + 1) by ring] at hinner
  obtain ⟨hFs, hFg, hFn⟩ := hF
  have hk32 : k / 32 = 4 * (k / 128) := by omega
  refine ⟨by rw [← hf, hFs, hk32], fun t ht => by rw [← hf]; exact hFg t (by omega), by rw [← hf, hFn, hk32], ?_⟩
  -- the later promotions into a +0 total
  · rw [← hz]
    unfold laterZeroPromos
    rw [ite_eq_right (show ¬ devH100.G = 0 by decide), show devH100.G = 4 from rfl, show (32 : ℕ) * 4 = 128 from rfl]
    refine count_shift _ _ (k / 128) ?_
    intro g hg
    have hq := hT.2 (g + 1) (by omega)
    have hle : 128 * (g + 1) ≤ k := by omega
    simp only [beq_iff_eq]
    constructor
    · intro h0
      refine ⟨hle, ?_⟩
      rw [ticket_prefix A B i j (g + 1) hle, ← hgs, ← hq.2, h0]
      exact Freeze.wordQ_zero
    · rintro ⟨h', h0⟩
      rw [ticket_prefix A B i j (g + 1) h', ← hgs, ← hq.2] at h0
      exact AccumProofs.wordQ_zero_word hq.1 h0

/-- **The inert-work replay is the game's**, from the keys' three statements: `word_debit` counts `word_flags`' flags. -/
theorem debitIsTheReplayOf (hK : KeyIsTheValue) (hL : LeastIsTheThreshold) (hC : CarriedIsTheAdd) :
    DebitIsTheReplay := by
  intro m n k A B i j f z h
  unfold Rule.wordDebit at h
  obtain ⟨⟨fl, z'⟩, hw, h⟩ := WordsSim.bind_ok h
  have hfz := WordsSim.pure_ok h
  simp only [Prod.mk.injEq] at hfz
  obtain ⟨hf, hz⟩ := hfz
  obtain ⟨-, -, hn, hzz⟩ := wordFlagsIsTheReplayOf hK hL hC A B i j hw
  exact ⟨by rw [← hf, hn], by rw [← hz]; exact hzz⟩

/-- **(c) The atom flags the check replays are the game's.** -/
theorem wordFlagsIsTheReplay {m n k : ℕ} (A : Matrix (Fin m) (Fin k) Code) (B : Matrix (Fin n) (Fin k) Code)
    (i : Fin m) (j : Fin n) {fl : Array Bool} {z : ℕ}
    (h : Rule.wordFlags (Array.ofFn fun l => (A i l).val) (Array.ofFn fun l => (B j l).val) = .ok (fl, z)) :
    fl.size = k / 32 ∧ (∀ t < k / 32, fl[t]! = Skip.unitFlags A B i j t) ∧
      (fl.filter id).size = ((Finset.range (k / 32)).filter fun t => Skip.unitFlags A B i j t = true).card ∧
      z = laterZeroPromos devH100 A B i j :=
  wordFlagsIsTheReplayOf keyIsTheValue leastIsTheThreshold carriedIsTheAdd A B i j h

/-- **(c) The inert-work replay is the game's.** -/
theorem debitIsTheReplay : DebitIsTheReplay :=
  debitIsTheReplayOf keyIsTheValue leastIsTheThreshold carriedIsTheAdd

namespace DebitSim

/-! ## The check's codes -/

/-- `word_flags` refuses a word whose first row is not whole promotion groups. -/
theorem wordFlags_ungrouped {a b : Array ℕ} (h : a.size % 128 ≠ 0) :
    Rule.wordFlags a b = .error "a word's rows are whole promotion groups" := by
  have hG : PouwCheck.PearlPlus.G = 4 := rfl
  unfold Rule.wordFlags
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', range_size, hG, Nat.reduceMul]
  split
  · rfl
  · rename_i hcond
    exact (hcond (by simp [h])).elim

/-- `word_debit` refuses a word whose first row is not whole promotion groups. -/
theorem wordDebit_ungrouped {a b : Array ℕ} (h : a.size % 128 ≠ 0) :
    Rule.wordDebit a b = .error "a word's rows are whole promotion groups" := by
  unfold Rule.wordDebit
  rw [wordFlags_ungrouped h]
  rfl

/-- `semH3`'s noisy codes are `pc8Sem`'s `quant8A` and `quant8B` at its noise. -/
theorem unitChain_h3 {Q R S : Type} (saltOf : S → List ℕ) (rootOf : ℕ → List ℕ) (H : Q → R) (s : S)
    (U : Pouw.PearlC.Workload) (u : ℕ) (act : Codes) :
    ((semH3 saltOf rootOf : PearlCSem Q R S).unitChain H s U u act).1 =
        WholeCircuit.quant8A RowCircuit.coreOps hopperE4m3K32
          ((semH3 saltOf rootOf : PearlCSem Q R S).noise H s U u act) act ∧
      ((semH3 saltOf rootOf : PearlCSem Q R S).unitChain H s U u act).2 =
        WholeCircuit.quant8B RowCircuit.coreOps hopperE4m3K32
          ((semH3 saltOf rootOf : PearlCSem Q R S).noise H s U u act) (U.fixed (U.layout.fixedId u)) :=
  ⟨rfl, rfl⟩

/-- **A word's codes**: over a noise `z` whose lines and bases are the check's (`hEA`, `hEB`, `hFA`, `hFB`), where the
check's fixed noise and operands are values, entry `x`'s A′ row and entry `y`'s B̃ row have `k` codes, and where `8 ∣ k`
on rows that are `Row8OK` they are `pc8Sem`'s codes at the H100 atom. -/
theorem encode_codes {m n k u : ℕ} {act wt : Codes} {z : Noise m n k Params.pi.r} {salt rootB : ByteArray}
    {wn : Words.FixedNoise} {rows cols : List ℕ} {ops : Words.Ops}
    (hkK : k ≤ 2 ^ 16) (hact : ∀ i ∈ rows, ∀ l < k, act i l < 2 ^ 32) (hwt : ∀ j ∈ cols, ∀ l < k, wt j l < 2 ^ 32)
    (hFA : ∀ l (hl : l < k) (t : Fin Params.pi.r), (wn.fa[l]!)[t.val]! = (z.FA ⟨l, hl⟩ t).val)
    (hFB : ∀ l (hl : l < k) (t : Fin Params.pi.r), (wn.fb[l]!)[t.val]! = (z.FB ⟨l, hl⟩ t).val)
    (hEA : ∀ (i : Fin m) (e : Array ℕ),
      Words.sampleLine (Words.lineKey (Words.rowSeed salt rootB u i.val (rowBytesOf act k i.val))) 0 0 0 Words.NORM_E =
        .ok e → e.size = 32 ∧ ∀ t : Fin Params.pi.r, e[t.val]! = (z.EA i t).val)
    (hEB : ∀ (j : Fin n) (e : Array ℕ), Words.sampleLine (Words.lineKey wn.seedB) 1 0 j.val Words.NORM_E = .ok e →
        e.size = 32 ∧ ∀ t : Fin Params.pi.r, e[t.val]! = (z.EB j t).val)
    (hwn : Words.fixedNoise salt rootB k = .ok wn)
    (henc : Words.encode salt rootB wn u k rows (rows.map (rowBytesOf act k)) cols (cols.map (rowBytesOf wt k)) = .ok ops)
    {x y : ℕ} (i : Fin m) (j : Fin n) (hx : x < rows.length) (hxi : rows[x]! = i.val) (hy : y < cols.length)
    (hyj : cols[y]! = j.val) :
    (ops.a.codes[x]!).size = k ∧ (ops.b.codes[y]!).size = k ∧
      (8 ∣ k → WholeCircuit.Row8OK RowCircuit.coreOps k (act i) → WholeCircuit.Row8OK RowCircuit.coreOps k (wt j) →
        ops.a.codes[x]! = Array.ofFn (fun l => (WholeCircuit.quant8A RowCircuit.coreOps hopperE4m3K32 z act i l).val) ∧
        ops.b.codes[y]! = Array.ofFn (fun l => (WholeCircuit.quant8B RowCircuit.coreOps hopperE4m3K32 z wt j l).val)) := by
  obtain ⟨hfR, -⟩ := SemSim.fixedNoise_ok hwn
  unfold Words.encode at henc
  obtain ⟨ea, hea, henc⟩ := WordsSim.bind_ok henc
  obtain ⟨va, hva, henc⟩ := WordsSim.bind_ok henc
  obtain ⟨a, ha, henc⟩ := WordsSim.bind_ok henc
  try dsimp only at henc
  obtain ⟨eb, heb, henc⟩ := WordsSim.bind_ok henc
  obtain ⟨vb, hvb, henc⟩ := WordsSim.bind_ok henc
  obtain ⟨b, hb, henc⟩ := WordsSim.bind_ok henc
  obtain ⟨pas, -, henc⟩ := WordsSim.bind_ok henc
  obtain ⟨pbs, -, henc⟩ := WordsSim.bind_ok henc
  obtain rfl := WordsSim.pure_ok henc
  try dsimp only
  have hmx : i.val ∈ rows := by rw [← hxi]; exact SemSim.get!_mem hx
  have hmy : j.val ∈ cols := by rw [← hyj]; exact SemSim.get!_mem hy
  obtain ⟨hvaS, hvaA⟩ := WordsSim.mapM_ok hva
  obtain ⟨heaS, heaA⟩ := WordsSim.mapM_ok hea
  obtain ⟨hvbS, hvbA⟩ := WordsSim.mapM_ok hvb
  obtain ⟨hebS, hebA⟩ := WordsSim.mapM_ok heb
  simp only [List.size_toArray, List.length_zip, List.length_map, Nat.min_self] at hvaS heaS hvbS hebS
  obtain ⟨-, -, haR⟩ := SemSim.formV1_ok ha (by rw [hvaS, heaS])
  obtain ⟨-, -, hbR⟩ := SemSim.formV1_ok hb (by rw [hvbS, hebS])
  obtain ⟨s, ρw, αw, βw, hs, hep, hcodes⟩ := haR x (by omega)
  obtain ⟨s', ρw', αw', βw', hs', hep', hcodes'⟩ := hbR y (by omega)
  have hvx := hvaA x (by simpa using hx)
  rw [List.getElem!_toArray, SemSim.lmap_get! hx, hxi] at hvx
  obtain ⟨hvxS, hvxW⟩ := WordsSim.parseRow_ok (hact _ hmx) hvx
  have hvy := hvbA y (by simpa using hy)
  rw [List.getElem!_toArray, SemSim.lmap_get! hy, hyj] at hvy
  obtain ⟨hvyS, hvyW⟩ := WordsSim.parseRow_ok (hwt _ hmy) hvy
  have hkx : (a.codes[x]!).size = k := by rw [(WordsSim.mapM_ok hcodes).1, Array.size_range, hvxS]
  have hky : (b.codes[y]!).size = k := by rw [(WordsSim.mapM_ok hcodes').1, Array.size_range, hvyS]
  refine ⟨hkx, hky, fun hk8 hRA hRB => ?_⟩
  have hex := heaA x (by simpa using hx)
  rw [SemSim.zipL_get! hx] at hex
  try dsimp only at hex
  rw [hxi] at hex
  obtain ⟨heS, heA⟩ := hEA i ea[x]! hex
  have hey := hebA y (by simpa using hy)
  rw [List.getElem!_toArray, hyj] at hey
  obtain ⟨heS', heA'⟩ := hEB j eb[y]! hey
  have hk0 : 0 < k := by have := SemSim.s5Stats_pos hs; omega
  have hk27 : k ≤ 2 ^ 27 := le_trans hkK (by norm_num)
  obtain ⟨-, -, -, hcodA⟩ := formRow_quant8 (E := z.EA i) (F := z.FA) hk0 hk8 hk27 hvxS hvxW (hact _ hmx) hRA heS heA
    (fun t => (hfR t.val t.isLt).1) (fun t q => hFA t.val t.isLt q) hs hep hcodes
  obtain ⟨-, -, -, hcodB⟩ := formRow_quant8 (E := z.EB j) (F := z.FB) hk0 hk8 hk27 hvyS hvyW (hwt _ hmy) hRB heS' heA'
    (fun t => (hfR t.val t.isLt).2) (fun t q => hFB t.val t.isLt q) hs' hep' hcodes'
  refine ⟨Array.ext (by rw [Array.size_ofFn]; exact hkx) fun t h1 _ => ?_,
    Array.ext (by rw [Array.size_ofFn]; exact hky) fun t h1 _ => ?_⟩
  · rw [Array.getElem_ofFn]
    exact (getElem!_pos (a.codes[x]!) t h1).symm.trans (hcodA ⟨t, h1.trans_eq hkx⟩)
  · rw [Array.getElem_ofFn]
    exact (getElem!_pos (b.codes[y]!) t h1).symm.trans (hcodB ⟨t, h1.trans_eq hky⟩)

end DebitSim

open DebitSim in
/-- **(c) The replay reads the semantics' codes.** The check's A′ and B̃ rows of a word have `k` codes; where `k` is
whole promotion groups, on rows that pass the rules (`row8OK_of_floor`), they are `pc8Sem`'s codes at `semH3`'s noise,
whose lines and bases are the check's (`seedsAreTheSpec`, `lineIsTheSpec`, `fixedNoiseIsTheSpec`), and so
`unitChain`'s; where it isn't, `word_debit` refuses both words. -/
theorem replayReadsTheCodes : ReplayReadsTheCodes := by
  intro Q R S saltOf rootOf H s U u act salt rootB wn rows cols ops hsalt hroot hk _ _ hact hwt hW hwn henc x y i j
    hxi hyj hR
  obtain ⟨hx, hxi'⟩ := List.getElem?_eq_some_iff.1 hxi
  obtain ⟨hy, hyj'⟩ := List.getElem?_eq_some_iff.1 hyj
  have hWj := hW _ (List.mem_of_getElem? hyj)
  obtain ⟨hseed, -, -, hF⟩ := fixedNoiseIsTheSpec salt rootB _ wn hwn
  have hsd : bytesNat wn.seedB = NoiseSha512.seedB (saltOf s) (rootOf (U.layout.fixedId u)) := by
    rw [hseed, hsalt, hroot]
  obtain ⟨hAS, -, hAB⟩ := encode_codes (z := (semH3 saltOf rootOf : PearlCSem Q R S).noise H s U u act)
    (wt := U.fixed (U.layout.fixedId u)) hk hact hwt
    (fun l hl t => by rw [(hF l hl t).1, hsd]; rfl) (fun l hl t => by rw [(hF l hl t).2, hsd]; rfl)
    (fun i e he => by
      obtain ⟨h1, h2⟩ := lineIsTheSpec _ 0 0 0 Words.NORM_E 256 e (by norm_num) (by norm_num) (Or.inl ⟨rfl, rfl⟩) he
      refine ⟨h1, fun t => ?_⟩
      rw [h2 t, (seedsAreTheSpec).2.2, hsalt, hroot, WordsSim.bytesNat_rowBytesOf]
      rfl)
    (fun j e he => by
      obtain ⟨h1, h2⟩ := lineIsTheSpec _ 1 0 j.val Words.NORM_E 256 e (by norm_num) (by norm_num) (Or.inl ⟨rfl, rfl⟩) he
      refine ⟨h1, fun t => ?_⟩
      rw [h2 t, hsd]
      rfl)
    hwn henc i j hx (by rw [getElem!_pos rows x hx]; exact hxi') hy (by rw [getElem!_pos cols y hy]; exact hyj')
  by_cases hk128 : (U.layout.shape u).k % 128 = 0
  · obtain ⟨hA, hB⟩ := hAB (Nat.dvd_of_mod_eq_zero (by omega))
      (WholeCircuit.row8OK_of_floor CoreOps.pc8OpsAgree_coreOps hopperE4m3K32
        (semH3 saltOf rootOf : PearlCSem Q R S).noise i.isLt hR.1 (hR.2 i.isLt).2)
      (WholeCircuit.row8OK_of_floor CoreOps.pc8OpsAgree_coreOps hopperE4m3K32
        (semH3 saltOf rootOf : PearlCSem Q R S).noise j.isLt hWj.1 (hWj.2 j.isLt).2)
    obtain ⟨hU1, hU2⟩ := unitChain_h3 saltOf rootOf H s U u act
    rw [hA, hB, hU1, hU2]
  · have h1 : (ops.a.codes[x]!).size % 128 ≠ 0 := by rw [hAS]; exact hk128
    have h2 : (Array.ofFn fun l => (((semH3 saltOf rootOf : PearlCSem Q R S).unitChain H s U u act).1 i l).val).size %
        128 ≠ 0 := by
      rw [Array.size_ofFn]; exact hk128
    rw [wordDebit_ungrouped h1, wordDebit_ungrouped h2]

end Pouw.PearlPlus.Bridge
