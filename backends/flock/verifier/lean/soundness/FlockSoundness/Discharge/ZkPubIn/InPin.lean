import FlockSoundness.Discharge.ZkPubIn.SiteP
import FlockSoundness.Discharge.Layout.OutSite
import FlockSoundness.ExecDraw

set_option Elab.async false

/-!
# The `in` regions pin a public leaf's bits to the verifier's own words

`HmIn.regions` reads every public leaf's 16 bits from an `in` region: each word of a unit slot that holds some unit's
public leaf is in `HmIn.words` (`mem_words`), so in one of its aligned blocks (`blocks_cover`) and in one of the
`in` regions over it (`inRegions_cover`). The region's bytes at block `o` are, slot by slot and leaf by leaf, the
leaf's word u16 LE (`inBytes_bit`): at a public leaf of a proved unit's slot, the drawn statement's public word of the
slot's instance (`Circuit.instOf`), which is the population's word of the proved unit (`HmIn.drawn`), which is the
verifier's own (`OwnWords`). So at a drawn unit's site (`BcSiteHP`), a satisfying witness holds at a public leaf's
bit `(q, t)` of input `x` bit `t` of the verifier's own word `q` of the unit (`pub_siteHP`).
-/

namespace FlockSoundness.Discharge.ZkPubIn

open Flock FlockLevel3 Model HiddenExec

/-! ## Coverage -/

section cover

theorem foldl_mem_mono {α β : Type} (f : Array α → β → Array α) (hf : ∀ acc b, ∀ x ∈ acc, x ∈ f acc b) :
    ∀ (L : List β) (acc : Array α) {x : α}, x ∈ acc → x ∈ L.foldl f acc
  | [], _, _, h => h
  | b :: L, acc, _, h => foldl_mem_mono f hf L (f acc b) (hf acc b _ h)

theorem foldl_mem_of {α β : Type} (f : Array α → β → Array α) (hf : ∀ acc b, ∀ x ∈ acc, x ∈ f acc b)
    {L : List β} {b : β} (hb : b ∈ L) {x : α} (hx : ∀ acc, x ∈ f acc b) : ∀ acc : Array α, x ∈ L.foldl f acc := by
  induction L with
  | nil => cases hb
  | cons b' L ih =>
    intro acc
    rcases List.mem_cons.1 hb with rfl | hb
    · exact foldl_mem_mono f hf L _ (hx acc)
    · exact ih hb _

/-- The dedupe step of `HmIn.words`. -/
def dedupStep (acc : Array ℕ) (w : ℕ) : Array ℕ := if acc.back? == some w then acc else acc.push w

theorem dedupStep_mono (acc : Array ℕ) (w : ℕ) : ∀ x ∈ acc, x ∈ dedupStep acc w := by
  intro x hx
  unfold dedupStep
  split
  · exact hx
  · exact Array.mem_push_of_mem _ hx

theorem dedupStep_self (acc : Array ℕ) (w : ℕ) : w ∈ dedupStep acc w := by
  unfold dedupStep
  split
  · rename_i h
    obtain ⟨ys, rfl⟩ := Array.back?_eq_some_iff.1 (by simpa using h)
    exact Array.mem_push_self
  · exact Array.mem_push_self

/-- The public words' push step of `HmIn.words`. -/
def wordsStep (col : ℕ) (l : Array ℤ) (acc : Array ℕ) (j : ℕ) : Array ℕ :=
  if l[j]! ≥ 0 then acc.push (col + j / 8) else acc

theorem wordsStep_mono (col : ℕ) (l : Array ℤ) (acc : Array ℕ) (j : ℕ) : ∀ x ∈ acc, x ∈ wordsStep col l acc j := by
  intro x hx
  unfold wordsStep
  split
  · exact Array.mem_push_of_mem _ hx
  · exact hx

theorem wordsLeaf_mono (col : ℕ) (acc : Array ℕ) (l : Array ℤ) :
    ∀ x ∈ acc, x ∈ (Array.range l.size).foldl (wordsStep col l) acc := by
  intro x hx
  rw [← Array.foldl_toList]
  exact foldl_mem_mono _ (wordsStep_mono col l) _ _ hx

/-- **A public leaf's word is one of `HmIn.words`**: leaf `j` of unit `u`'s public leaf map, a public word, is in
word `col + j / 8` of the leaf group, and that word is in `HmIn.words`. -/
theorem mem_words {c : Circuit} {pi : HmIn.Ports} {u j : ℕ} (hu : u < pi.leaves.size)
    (hj : j < (pi.leaves.getD u #[]).size) (hq : 0 ≤ (pi.leaves.getD u #[]).getD j (-1)) :
    (c.unit.inGroups.getD 0 default).col + j / 8 ∈ HmIn.words c pi := by
  unfold HmIn.words
  rw [← Array.foldl_toList]
  refine foldl_mem_of dedupStep dedupStep_mono ?_ (fun acc => dedupStep_self acc _) #[]
  rw [Array.mem_toList_iff, FlockSoundness.Audit.DrawExec.mem_qsort, ← Array.foldl_toList]
  have hl : pi.leaves.getD u #[] = pi.leaves[u] := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hu, Option.getD_some]
  rw [hl] at hj hq
  refine foldl_mem_of (fun acc l => (Array.range l.size).foldl (wordsStep _ l) acc) (fun acc l => wordsLeaf_mono _ acc l)
    (Array.getElem_mem_toList hu) (fun acc => ?_) #[]
  rw [← Array.foldl_toList]
  refine foldl_mem_of (wordsStep _ pi.leaves[u]) (wordsStep_mono _ _) (by simpa using hj) (fun acc => ?_) acc
  unfold wordsStep
  have : (pi.leaves[u])[j]! ≥ 0 := by
    rw [getElem!_pos pi.leaves[u] j hj]
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hj, Option.getD_some] at hq
    exact hq
  simp only [this, ↓reduceIte]
  exact Array.mem_push_self

/-- The runs step of `HmIn.blocks`. -/
def runsStep (acc : Array (ℕ × ℕ)) (w : ℕ) : Array (ℕ × ℕ) :=
  match acc.back? with
  | some (a, b) => if (w == b) = true then acc.pop.push (a, b + 1) else acc.push (w, w + 1)
  | none => #[(w, w + 1)]

/-- A run (or block) covers `y`. -/
def Covers (R : Array (ℕ × ℕ)) (y : ℕ) : Prop := ∃ r ∈ R.toList, r.1 ≤ y ∧ y < r.2

def RunsWF (R : Array (ℕ × ℕ)) : Prop := ∀ r ∈ R.toList, r.1 ≤ r.2

theorem runsStep_wf {acc : Array (ℕ × ℕ)} (h : RunsWF acc) (w : ℕ) : RunsWF (runsStep acc w) := by
  unfold runsStep
  split
  · rename_i a b hb
    obtain ⟨ys, rfl⟩ := Array.back?_eq_some_iff.1 hb
    have hab := h (a, b) (by simp)
    split
    · rw [Array.pop_push]
      intro r hr
      simp only [Array.toList_push, List.mem_append, Array.mem_toList_iff, List.mem_singleton] at hr
      rcases hr with hr | rfl
      · exact h r (by simp [hr])
      · simp only at hab ⊢
        omega
    · intro r hr
      simp only [Array.toList_push, List.mem_append, Array.mem_toList_iff, List.mem_singleton] at hr
      rcases hr with (hr | rfl) | rfl
      · exact h r (by simp [hr])
      · exact hab
      · simp
  · intro r hr
    simp at hr
    subst hr
    simp

theorem runsStep_mono {acc : Array (ℕ × ℕ)} (w : ℕ) {y : ℕ} (hy : Covers acc y) :
    Covers (runsStep acc w) y := by
  obtain ⟨r, hr, h1, h2⟩ := hy
  unfold runsStep
  split
  · rename_i a b hb
    obtain ⟨ys, rfl⟩ := Array.back?_eq_some_iff.1 hb
    simp only [Array.toList_push, List.mem_append, Array.mem_toList_iff, List.mem_singleton] at hr
    rcases hr with hr' | rfl
    · split
      · rw [Array.pop_push]
        exact ⟨r, by simp [hr'], h1, h2⟩
      · exact ⟨r, by simp [hr'], h1, h2⟩
    · split
      · rw [Array.pop_push]
        exact ⟨(a, b + 1), by simp, h1, by simp only at h2 ⊢; omega⟩
      · exact ⟨(a, b), by simp, h1, h2⟩
  · rename_i hb
    rw [Array.back?_eq_none_iff] at hb
    subst hb
    simp at hr

theorem runsStep_self {acc : Array (ℕ × ℕ)} (h : RunsWF acc) (w : ℕ) : Covers (runsStep acc w) w := by
  unfold runsStep
  split
  · rename_i a b hb
    obtain ⟨ys, rfl⟩ := Array.back?_eq_some_iff.1 hb
    split
    · rename_i hw
      have hw' : w = b := by simpa using hw
      subst hw'
      rw [Array.pop_push]
      have := h (a, w) (by simp)
      exact ⟨(a, w + 1), by simp, this, by simp⟩
    · exact ⟨(w, w + 1), by simp, le_rfl, by simp⟩
  · exact ⟨(w, w + 1), by simp, le_rfl, by simp⟩

theorem foldl_runs : ∀ (L : List ℕ) (acc : Array (ℕ × ℕ)), RunsWF acc →
    RunsWF (L.foldl runsStep acc) ∧ (∀ y, Covers acc y → Covers (L.foldl runsStep acc) y) ∧
      ∀ y ∈ L, Covers (L.foldl runsStep acc) y
  | [], acc, h => ⟨h, fun _ hy => hy, fun _ hy => by cases hy⟩
  | w :: L, acc, h => by
      obtain ⟨h1, h2, h3⟩ := foldl_runs L (runsStep acc w) (runsStep_wf h w)
      refine ⟨h1, fun y hy => h2 y (runsStep_mono w hy), fun y hy => ?_⟩
      rcases List.mem_cons.1 hy with rfl | hy
      · exact h2 y (runsStep_self h y)
      · exact h3 y hy

/-- A block covers `y`: `y` is one of its `2^e` words from `w`. -/
def BCovers (B : Array (ℕ × ℕ)) (y : ℕ) : Prop := ∃ b ∈ B.toList, b.1 ≤ y ∧ y < b.1 + 2 ^ b.2

theorem bcovers_push {B : Array (ℕ × ℕ)} {y : ℕ} (h : BCovers B y) (x : ℕ × ℕ) : BCovers (B.push x) y := by
  obtain ⟨b, hb, h1, h2⟩ := h
  exact ⟨b, by simp [hb], h1, h2⟩

/-- **Every word of `ws` is in one of `HmIn.blocks ws`'s blocks.** -/
theorem blocks_cover (ws : Array ℕ) {y : ℕ} (hy : y ∈ ws) : BCovers (HmIn.blocks ws) y := by
  have hruns := foldl_runs ws.toList #[] (fun r hr => by simp at hr)
  have hcov := hruns.2.2 y (Array.mem_toList_iff.2 hy)
  rw [Array.foldl_toList] at hcov
  unfold HmIn.blocks
  simp only [Id.run, bind_pure_comp]
  change BCovers (forIn (m := Id) (ws.foldl runsStep #[]) (#[] : Array (ℕ × ℕ)) _) y
  revert hcov
  generalize ws.foldl runsStep #[] = runs
  intro hcov
  rw [← Array.forIn_toList]
  obtain ⟨r, hr, hr1, hr2⟩ := hcov
  refine forIn_id_inv_tail (fun l out => BCovers out y ∨ ∃ r ∈ l, r.1 ≤ y ∧ y < r.2) ?_ runs.toList #[]
    (.inr ⟨r, hr, hr1, hr2⟩) |>.elim id (fun ⟨_, h, _⟩ => by cases h)
  rintro ⟨a0, b⟩ l out hP
  refine ⟨_, rfl, ?_⟩
  dsimp only
  have key : ∀ (L : List ℕ) (s : Array (ℕ × ℕ) × ℕ), L.length ≤ b - a0 → a0 ≤ s.2 →
      min b (a0 + (b - a0 - L.length)) ≤ s.2 → (∀ x, BCovers out x → BCovers s.1 x) →
      (∀ x, a0 ≤ x → x < s.2 → x < b → BCovers s.1 x) →
      (∀ x, BCovers out x → BCovers (forIn (m := Id) L s fun _ (s : Array (ℕ × ℕ) × ℕ) =>
          if s.2 < b then pure (ForInStep.yield (s.1.push (s.2,
            ((List.range ((b - s.2).log2 + 1)).filter (fun x => s.2 % 2 ^ x == 0)).getLast?.getD 0),
            s.2 + 2 ^ ((List.range ((b - s.2).log2 + 1)).filter (fun x => s.2 % 2 ^ x == 0)).getLast?.getD 0))
          else pure (ForInStep.yield (s.1, s.2))).1 x) ∧
      (∀ x, a0 ≤ x → x < min b (a0 + (b - a0)) → BCovers (forIn (m := Id) L s fun _ (s : Array (ℕ × ℕ) × ℕ) =>
          if s.2 < b then pure (ForInStep.yield (s.1.push (s.2,
            ((List.range ((b - s.2).log2 + 1)).filter (fun x => s.2 % 2 ^ x == 0)).getLast?.getD 0),
            s.2 + 2 ^ ((List.range ((b - s.2).log2 + 1)).filter (fun x => s.2 % 2 ^ x == 0)).getLast?.getD 0))
          else pure (ForInStep.yield (s.1, s.2))).1 x) := by
    intro L
    induction L with
    | nil =>
      intro s _ _ hmin hmono hcov
      refine ⟨hmono, fun x hx1 hx2 => hcov x hx1 (by simp at hmin; omega) (by omega)⟩
    | cons i L ih =>
      intro s hlen ha0 hmin hmono hcov
      rw [List.forIn_cons]
      obtain ⟨o, a⟩ := s
      simp only [List.length_cons] at hlen hmin ha0 hmono hcov ⊢
      have hlen' : L.length ≤ b - a0 := by omega
      split
      · rename_i hab
        generalize ((List.range ((b - a).log2 + 1)).filter (fun x => a % 2 ^ x == 0)).getLast?.getD 0 = e
        have he := Nat.one_le_two_pow (n := e)
        refine ih (o.push (a, e), a + 2 ^ e) hlen' (by omega) (by omega)
          (fun x hx => bcovers_push (hmono x hx) _) ?_
        intro x hx1 hx2 hx3
        by_cases hxa : x < a
        · exact bcovers_push (hcov x hx1 hxa hx3) _
        · exact ⟨(a, e), by simp, by simp only; omega, by simp only; omega⟩
      · rename_i hab
        exact ih (o, a) hlen' ha0 (by omega) hmono hcov
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  have hlenR : (List.range' [:b - a0].start [:b - a0].size [:b - a0].step).length = b - a0 := by
    simp [Std.Legacy.Range.size]
  have hk := key _ (out, a0) hlenR.le le_rfl (by rw [hlenR]; omega)
    (fun _ h => h) (fun x h1 h2 _ => absurd h2 (by omega))
  rcases hP with hP | ⟨r, hr, hr1, hr2⟩
  · exact .inl (hk.1 y hP)
  · rcases List.mem_cons.1 hr with rfl | hr
    · exact .inl (hk.2 y hr1 (by simp only at hr1 hr2 ⊢; omega))
    · exact .inr ⟨r, hr, hr1, hr2⟩

theorem pushes_mono {α : Type} (f : ℕ → α) : ∀ (L : List ℕ) (acc : Array α) {x : α}, x ∈ acc →
    x ∈ L.foldl (fun acc r => acc.push (f r)) acc :=
  foldl_mem_mono _ fun _ _ _ h => Array.mem_push_of_mem _ h

/-- **Every word of a block is in one of the `in` regions over it** (`HmIn.inRegions`): region `(w', s)`, `2^s` words
from `w'`, cut from a block `(w, e)` of `HmIn.blocks`, `w ≤ w'`. -/
theorem inRegions_cover (c : Circuit) (pi : HmIn.Ports) {y : ℕ} (hy : BCovers (HmIn.blocks (HmIn.words c pi)) y) :
    ∃ R ∈ (HmIn.inRegions c pi).toList, ∃ b ∈ (HmIn.blocks (HmIn.words c pi)).toList,
      b.1 ≤ R.1 ∧ R.1 ≤ y ∧ y < R.1 + 2 ^ R.2 := by
  obtain ⟨⟨w, e⟩, hb, h1, h2⟩ := hy
  simp only at h1 h2
  set s := min e (PT_LOCAL - (7 + log2ceil c.unitRange.count)) with hs
  have hse : s ≤ e := min_le_left _ _
  have hP : 0 < 2 ^ s := Nat.two_pow_pos _
  have hes : 2 ^ e = 2 ^ (e - s) * 2 ^ s := by rw [← pow_add, Nat.sub_add_cancel hse]
  set r := (y - w) / 2 ^ s with hr
  have hr1 : r * 2 ^ s ≤ y - w := Nat.div_mul_le_self _ _
  have hr2 : y - w < r * 2 ^ s + 2 ^ s := by
    have := Nat.lt_div_mul_add (a := y - w) hP
    rw [hr]; linarith
  have hrlt : r < 2 ^ (e - s) := by
    rw [hr, Nat.div_lt_iff_lt_mul hP, ← hes]; omega
  refine ⟨(w + r * 2 ^ s, s), ?_, (w, e), hb, by simp, by simp only; omega, by simp only; omega⟩
  unfold HmIn.inRegions
  rw [Array.mem_toList_iff, ← Array.foldl_toList]
  refine foldl_mem_of _ (fun acc b => ?_) hb (fun acc => ?_) #[]
  · obtain ⟨w', e'⟩ := b
    intro x hx
    dsimp only
    rw [← Array.foldl_toList]
    exact pushes_mono _ _ _ hx
  · dsimp only
    rw [← Array.foldl_toList]
    exact foldl_mem_of (fun (acc : Array (ℕ × ℕ)) r => acc.push (w + r * 2 ^ s, s))
      (fun _ _ _ h => Array.mem_push_of_mem _ h) (L := (Array.range (2 ^ (e - s))).toList) (b := r) (by simpa using hrlt)
      (fun _ => Array.mem_push_self) acc

end cover

/-! ## The `in` regions' bytes -/

section bytes

/-- **Leaf `j` of unit slot `k` at block `b`, as region `in` over words from `w` packs it** (`HmIn.regions`): a public
leaf of a slot's instance its public word, else 0. -/
def inVal (c : Circuit) (pi : HmIn.Ports) (pub : Public) (w b k j : ℕ) : ℕ :=
  match (if k < c.unitRange.count then c.instOf pub.n b (k / c.upv) else none) with
  | some i =>
    if (pi.leaves.getD (k % c.upv) #[]).getD (8 * (w - (c.unit.inGroups.getD 0 default).col) + j) (-1) ≥ 0 then
      pub.outputs.getD (i * pi.words +
        ((pi.leaves.getD (k % c.upv) #[]).getD (8 * (w - (c.unit.inGroups.getD 0 default).col) + j) (-1)).toNat) 0
    else 0
  | none => 0

/-- Region `in`'s bytes at block `b`: `K` slots of `J` 16-bit values `V k j`, u16 LE. -/
def u16Bytes (K J : ℕ) (V : ℕ → ℕ → ℕ) : ByteArray :=
  (Array.range K).foldl (fun acc k => (Array.range J).foldl (fun acc j =>
    (acc.push (V k j % 256).toUInt8).push (V k j / 256).toUInt8) acc) ByteArray.empty

theorem u16_data (V : ℕ → ℕ) : ∀ (L : List ℕ) (acc : ByteArray),
    (L.foldl (fun acc j => (acc.push (V j % 256).toUInt8).push (V j / 256).toUInt8) acc).data.toList =
      acc.data.toList ++ (L.map fun j => [(V j % 256).toUInt8, (V j / 256).toUInt8]).flatten
  | [], acc => by simp
  | j :: L, acc => by
      rw [List.foldl_cons, u16_data V L]
      simp [ByteArray.push]

theorem u16Slots_data (J : ℕ) (V : ℕ → ℕ → ℕ) : ∀ (L : List ℕ) (acc : ByteArray),
    (L.foldl (fun acc k => (Array.range J).foldl (fun acc j =>
      (acc.push (V k j % 256).toUInt8).push (V k j / 256).toUInt8) acc) acc).data.toList =
      acc.data.toList ++ (L.map fun k =>
        ((List.range J).map fun j => [(V k j % 256).toUInt8, (V k j / 256).toUInt8]).flatten).flatten
  | [], acc => by simp
  | k :: L, acc => by
      rw [List.foldl_cons, u16Slots_data J V L, ← Array.foldl_toList, u16_data, Array.toList_range]
      simp

/-- **A u16 LE byte's bit**: bit `t % 8` of byte `t / 8` of `v`'s two bytes is bit `t` of `v`. -/
theorem u16_bit (v : ℕ) {t : ℕ} (ht : t < 16) :
    (([(v % 256).toUInt8, (v / 256).toUInt8].getD (t / 8) default).toNat).testBit (t % 8) = v.testBit t := by
  by_cases h8 : t < 8
  · have e1 : t / 8 = 0 := by omega
    have e2 : t % 8 = t := by omega
    rw [e1, e2]
    simp only [List.getD_cons_zero, Nat.toUInt8, UInt8.toNat_ofNat']
    rw [show (256 : ℕ) = 2 ^ 8 from rfl, Nat.mod_mod, Nat.testBit_mod_two_pow]
    simp [h8]
  · have e1 : t / 8 = 1 := by omega
    rw [e1]
    simp only [List.getD_cons_succ, List.getD_cons_zero, Nat.toUInt8, UInt8.toNat_ofNat']
    rw [show (256 : ℕ) = 2 ^ 8 from rfl, Nat.testBit_mod_two_pow, Nat.testBit_div_two_pow]
    have : t % 8 < 8 := Nat.mod_lt _ (by norm_num)
    simp only [this, decide_true, Bool.true_and]
    congr 1
    omega

theorem length_flatten_two {α : Type} (f g : ℕ → α) : ∀ L : List ℕ, (L.map fun j => [f j, g j]).flatten.length = 2 * L.length
  | [] => rfl
  | j :: L => by
      rw [List.map_cons, List.flatten_cons, List.length_append, length_flatten_two f g L, List.length_cons]
      simp only [List.length_cons, List.length_nil]
      ring

/-- **Region `in`'s bit at slot `k`'s leaf `j`, bit `t`**: bit `t % 8` of byte `k · 2J + 2j + t / 8` is bit `t` of
`V k j`. -/
theorem u16Bytes_bit {K J : ℕ} (V : ℕ → ℕ → ℕ) {k j t : ℕ} (hk : k < K) (hj : j < J) (ht : t < 16) :
    ((u16Bytes K J V).get! (k * (2 * J) + (2 * j + t / 8))).toNat.testBit (t % 8) = (V k j).testBit t := by
  rw [Layout.get!_eq_getD, u16Bytes, ← Array.foldl_toList, Array.toList_range, u16Slots_data]
  rw [show ByteArray.empty.data.toList = [] from rfl, List.nil_append]
  have ht8 : t / 8 < 2 := by omega
  rw [Layout.getD_flatten_const default k _ (fun l hl => by
      obtain ⟨k', -, rfl⟩ := List.mem_map.1 hl
      rw [length_flatten_two, List.length_range]) (by simpa using hk) (by omega)]
  have hslot : ((List.range K).map fun k =>
      ((List.range J).map fun j => [(V k j % 256).toUInt8, (V k j / 256).toUInt8]).flatten).getD k [] =
      ((List.range J).map fun j => [(V k j % 256).toUInt8, (V k j / 256).toUInt8]).flatten := by
    rw [List.getD_eq_getElem _ _ (by simpa using hk)]
    simp
  rw [hslot, show 2 * j + t / 8 = j * 2 + t / 8 by ring, Layout.getD_flatten_const default j _ (fun l hl => by
      obtain ⟨j', -, rfl⟩ := List.mem_map.1 hl
      rfl) (by simpa using hj) ht8]
  have hw : ((List.range J).map fun j => [(V k j % 256).toUInt8, (V k j / 256).toUInt8]).getD j [] =
      [(V k j % 256).toUInt8, (V k j / 256).toUInt8] := by
    rw [List.getD_eq_getElem _ _ (by simpa using hj)]
    simp
  rw [hw]
  exact u16_bit _ ht

/-- The free bits of region `in` over `2^s` words: a slot's `7 + s` column bits, and the slot bits. -/
def inFree (c : Circuit) (s : ℕ) : Array ℕ :=
  (Array.range (7 + s)).map (0 + ·) ++ (Array.range (log2ceil c.unitRange.count)).map (c.unitRange.slotLog + ·)

/-- **Each `in` region is a region of `HmIn.regions`**: region `(w, s)` of `HmIn.inRegions` is `mkRegion`'s, at the
words from `w` of every unit slot, with bytes `u16Bytes` of `inVal`. -/
theorem in_region {c : Circuit} {pi : HmIn.Ports} {pub : Public} {ri : Array Region}
    (h : HmIn.regions c pi pub = .ok ri) (hp : pi.ports.isEmpty = false) {w s : ℕ}
    (hR : (w, s) ∈ (HmIn.inRegions c pi).toList) :
    ∃ R ∈ ri.toList, ∃ name, mkRegion c.kLog name (inFree c s) (c.unitRange.start + w * WORD)
      (fun b => u16Bytes (2 ^ log2ceil c.unitRange.count) (8 * 2 ^ s) (inVal c pi pub w b)) = .ok R := by
  obtain ⟨r, hr, hrw⟩ := List.getElem_of_mem hR
  unfold HmIn.regions at h
  simp only [hp, Bool.false_eq_true, ↓reduceIte] at h
  obtain ⟨hsz, hall⟩ := Exec.range_mapM_ok h
  have hr' : r < ri.size := by rw [hsz]; simpa using hr
  refine ⟨ri[r], Array.getElem_mem_toList hr', ?_⟩
  have := hall r hr'
  have hget : (HmIn.inRegions c pi)[r]! = (w, s) := by
    rw [getElem!_pos _ _ (by simpa using hr)]
    simpa using hrw
  rw [hget] at this
  exact ⟨_, this⟩

end bytes

end FlockSoundness.Discharge.ZkPubIn
