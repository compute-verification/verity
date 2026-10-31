import Proofs.Flock.Claims.Sampling.KS
import Proofs.Flock.Claims.Sampling.Session
import Proofs.Flock.Claims.Sampling.Select.Misses

/-!
# The draw law from the deployed draw (`Claims.Sampling.DrawLaw`)

* `Design.Pa_empty`: a well-formed design has `P_a(∅) = 1`.
* `allSlots`, `drawChecked`: every slot of a layout, and the deployed draw's checked slots, `Slots.select`'s or, where
  it is refused (the run rejects), every slot.
* `Partition.shape`: each class's strata sizes, in proof blocks; `P_a`, the slots and the design's well-formedness read
  only it (`Design.Pa_of_shape`, `Partition.blocks_length_of_shape`, `Design.wellFormed_of_shape`).
* `DrawSlots`: the deployed draw is `Slots.select`'s on a layout of the statement's partition's shape and on its design,
  from uniform bytes of the coins.
* `drawLaw_of_slots`: `DrawLaw` there, from `drawMisses`.
-/

open SampledProofs
open Lean (Json)

namespace Claims.Sampling

open Claims Finset Properties.SampledProofs Properties.SampledProofs.SecurityProofs

/-! ## `P_a(∅)` on a well-formed design -/

theorem window_empty (o n : ℕ) : window o n ∅ = ∅ := by simp [window]

theorem strataPa_empty : ∀ (l : List (ℕ × ℕ)) (o : ℕ), (∀ p ∈ l, p.2 ≤ p.1) → strataPa l o ∅ = 1
  | [], _, _ => rfl
  | (n, k) :: l, o, h => by
    have hk : k ≤ n := h _ List.mem_cons_self
    rw [strataPa, window_empty, Finset.card_empty, Nat.sub_zero,
      div_self (Nat.cast_ne_zero.2 (Nat.choose_pos hk).ne'), one_mul]
    exact strataPa_empty l _ fun p hp => h p (List.mem_cons_of_mem _ hp)

theorem OneStage.Pa_empty (ns : List ℕ) (st : OneStage) (h : st.WellFormed ns) : st.Pa ns ∅ = 1 := by
  cases st with
  | all => simp [OneStage.Pa]
  | subset K =>
    simp only [OneStage.Pa, Finset.card_empty, Nat.sub_zero]
    exact div_self (Nat.cast_ne_zero.2 (Nat.choose_pos h.2).ne')
  | bernoulli num den => simp [OneStage.Pa]
  | stratified K =>
    obtain ⟨hK1, hKN, hn⟩ := h
    refine strataPa_empty _ 0 fun p hp => ?_
    obtain ⟨n, hnm, rfl⟩ := List.mem_map.1 hp
    have h1 := hn n hnm
    refine max_le h1 (Nat.lt_succ_iff.1 ((Nat.div_lt_iff_lt_mul (by omega)).2 ?_))
    have : K * n ≤ ns.sum * n := Nat.mul_le_mul_right n hKN
    nlinarith
  | work K w f =>
    refine strataPa_empty _ 0 fun p hp => ?_
    obtain ⟨q, -, rfl⟩ := List.mem_map.1 hp
    exact min_le_left _ _

theorem classesPa_empty : ∀ (rs : List OneStage) (cs : List (List (List (List ℕ)))) (o : ℕ), rs.length = cs.length →
    (∀ p ∈ rs.zip cs, p.1.WellFormed (strataSizes p.2)) → classesPa rs cs o ∅ = 1
  | [], [], _, _, _ => rfl
  | r :: rs, c :: cs, o, hl, h => by
    rw [classesPa, window_empty, OneStage.Pa_empty _ r (h _ List.mem_cons_self), one_mul]
    exact classesPa_empty rs cs _ (by simpa using hl) fun p hp => h p (List.mem_cons_of_mem _ hp)
  | [], _ :: _, _, hl, _ => absurd hl (by simp)
  | _ :: _, [], _, hl, _ => absurd hl (by simp)

/-- **A well-formed design accepts the empty set surely**: `P_a(∅) = 1`. -/
theorem Design.Pa_empty (d : Design) (P : Partition) (h : d.WellFormed P) : d.Pa P ∅ = 1 :=
  classesPa_empty _ _ 0 h.1 h.2

/-! ## A partition's shape -/

/-- **A partition's shape**: each class's strata sizes, in proof blocks (`strataSizes`). The design and the draw read a
partition only through it: its slots, `P_a` and the design's well-formedness. -/
def Partition.shape (P : Partition) : List (List ℕ) := P.classes.map strataSizes

theorem flatten_length_eq_sizes (c : List (List (List ℕ))) : c.flatten.length = (strataSizes c).sum := by
  simp [strataSizes, List.length_flatten]

theorem Partition.blocks_length_eq_shape (P : Partition) : P.blocks.length = (P.shape.map List.sum).sum := by
  simp [Partition.blocks, Partition.shape, strataSizes, List.length_flatten, List.map_flatten, List.sum_flatten,
    Function.comp_def]

theorem Partition.blocks_length_of_shape {P Q : Partition} (h : P.shape = Q.shape) :
    P.blocks.length = Q.blocks.length := by
  rw [P.blocks_length_eq_shape, Q.blocks_length_eq_shape, h]

theorem classesPa_of_shape : ∀ (rs : List OneStage) (cs cs' : List (List (List (List ℕ)))) (o : ℕ) (B : Finset ℕ),
    cs.map strataSizes = cs'.map strataSizes → classesPa rs cs o B = classesPa rs cs' o B
  | r :: rs, c :: cs, c' :: cs', o, B, h => by
    simp only [List.map_cons, List.cons.injEq] at h
    rw [classesPa, classesPa, flatten_length_eq_sizes c, flatten_length_eq_sizes c', h.1,
      classesPa_of_shape rs cs cs' _ B h.2]
  | [], _, _, _, _, _ => by simp [classesPa]
  | _ :: _, [], [], _, _, _ => rfl
  | _ :: _, [], _ :: _, _, _, h => absurd h (by simp)
  | _ :: _, _ :: _, [], _, _, h => absurd h (by simp)

/-- **`P_a` reads only the shape.** -/
theorem Design.Pa_of_shape (d : Design) {P Q : Partition} (h : P.shape = Q.shape) (B : Finset ℕ) :
    d.Pa P B = d.Pa Q B :=
  classesPa_of_shape _ _ _ 0 B h

theorem forall_zip_of_shape : ∀ (rs : List OneStage) (cs cs' : List (List (List (List ℕ)))),
    cs.map strataSizes = cs'.map strataSizes →
      ((∀ p ∈ rs.zip cs, p.1.WellFormed (strataSizes p.2)) ↔ ∀ p ∈ rs.zip cs', p.1.WellFormed (strataSizes p.2))
  | [], _, _, _ => by simp
  | _ :: _, [], [], _ => Iff.rfl
  | r :: rs, c :: cs, c' :: cs', h => by
    simp only [List.map_cons, List.cons.injEq] at h
    simp only [List.zip_cons_cons, List.forall_mem_cons]
    rw [h.1, forall_zip_of_shape rs cs cs' h.2]
  | _ :: _, [], _ :: _, h => absurd h (by simp)
  | _ :: _, _ :: _, [], h => absurd h (by simp)

/-- **The design's well-formedness reads only the shape.** -/
theorem Design.wellFormed_of_shape (d : Design) {P Q : Partition} (h : P.shape = Q.shape) :
    d.WellFormed P ↔ d.WellFormed Q := by
  have hl : P.classes.length = Q.classes.length := by
    simpa [Partition.shape] using congrArg List.length h
  unfold Design.WellFormed
  rw [hl, forall_zip_of_shape _ _ _ h]

/-! ## The deployed draw's checked slots -/

/-- Every slot of a layout, in order. -/
def allSlots (l : Slots.Layout) : Array ℕ := Array.range l.blocks.size

theorem mem_allSlots (l : Slots.Layout) (v : ℕ) : v ∈ allSlots l ↔ v < l.blocks.size := by
  simp [allSlots]

theorem slotPartition_blocks (l : Slots.Layout) : (slotPartition l).blocks = l.blocks.toList.map puGates := by
  simp [slotPartition, Partition.blocks, Slots.Layout.blocks, List.map_flatten, Function.comp_def]

theorem size_allSlots (l : Slots.Layout) : (allSlots l).size = (slotPartition l).blocks.length := by
  simp [allSlots, slotPartition_blocks]

theorem size_checked_le (l : Slots.Layout) (proved : Array ℕ) :
    (Slots.checked l proved).size ≤ (slotPartition l).blocks.length := by
  rw [slotPartition_blocks, List.length_map, Array.length_toList, ← Array.length_toList]
  simp only [Slots.checked, Array.toList_map, Array.toList_filter, List.length_map]
  exact List.filter_sublist.length_le.trans (by simp)

/-- **The slots the deployed draw checks** on the bytes `src`: `Slots.select`'s, and every slot of the layout `l` where
it is refused, where the run rejects. -/
def drawChecked (rule : Select.Rule) (d : Program) (query : Json) (table : Option Json)
    (cm : Option (Array (Int × Array Int))) (l : Slots.Layout) (src : ByteArray) : Array ℕ :=
  match Slots.select rule d query table cm src with
  | .ok (_, c) => c
  | .error _ => allSlots l

/-- A slot of `c` is at an index below `n` when `c` has at most `n`. -/
theorem index_of_mem {c : Array ℕ} {n : ℕ} (hc : c.size ≤ n) {u : ℕ} (hu : u ∈ c) :
    ∃ p : Fin n, c.toList[p.val]? = some u := by
  obtain ⟨p, hp⟩ := List.mem_iff_getElem?.1 (Array.mem_toList_iff.2 hu)
  have := (List.getElem?_eq_some_iff.1 hp).1
  exact ⟨⟨p, by rw [Array.length_toList] at this; omega⟩, hp⟩

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin]

/-- **The deployed draw is `Slots.select`'s**: on a statement that parses and whose query evaluates, the verifier's own
program `d` (its parse of the statement's descriptor), query, selection rule, work table and closure map, whose layout
`l` has the statement's partition's shape
(`Partition.shape`: its strata's sizes in proof blocks, in their classes; the blocks' gates are the partition's) and
whose design is the statement's; `L` bytes of the coins, uniform when the coins are; and the checked slots the draw's
on them (`drawChecked`). -/
def DrawSlots (D : Deploy Coin Col T) : Prop :=
  ∀ x s Pt, D.parse x = some s → D.partition s = some Pt →
  ∃ (rule : Select.Rule) (d : Program) (query : Json) (table : Option Json)
    (cm : Option (Array (Int × Array Int))) (l : Slots.Layout) (L : ℕ) (bytes : (Fin (D.V.k x) → Coin) → Fin L → Fin 256),
    Program.parse ⟨s.desc.toArray⟩ = .ok d ∧ Slots.layout d query = .ok l ∧ Pt.shape = (slotPartition l).shape ∧
      slotDesign rule d query table = .ok s.design ∧
      (∀ E : (Fin L → Fin 256) → Prop, Pr (fun c => E (bytes c)) = Pr E) ∧
      ∀ c, D.checked x c = (drawChecked rule d query table cm l (bytesOf (bytes c))).toList

/-- **The draw law from the deployed draw**: where the draw is `Slots.select`'s on uniform bytes (`DrawSlots`), its
miss probability at fixed first-stage coins averages at most `P_a(B)` (`drawMisses`); a refused draw checks every slot,
so it misses no nonempty `B`, and a well-formed design has `P_a(∅) = 1`. The partition is read through its shape. -/
theorem drawLaw_of_slots (D : Deploy Coin Col T) (h : DrawSlots D) : DrawLaw D := by
  intro x s Pt hx hP hin B hB
  obtain ⟨rule, d, query, table, cm, l, L, bytes, -, hl, hshape, hds, hunif, hch⟩ := h x s Pt hx hP
  have hnPU : nPU D x = (slotPartition l).blocks.length := by
    simp [nPU, hx, Deploy.part, hP, Partition.blocks_length_of_shape hshape]
  rw [Partition.blocks_length_of_shape hshape] at hB
  rw [Design.Pa_of_shape _ hshape]
  have hwf := (Design.wellFormed_of_shape _ hshape).1 hin.2.2
  by_cases hB0 : B = ∅
  · subst hB0
    refine ⟨fun _ => 1, fun _ => zero_le_one, fun r => Pr_le_one _, ?_⟩
    rw [Ex_const, Design.Pa_empty _ _ hwf]
  obtain ⟨v0, hv0⟩ := Finset.nonempty_iff_ne_empty.2 hB0
  have hlen : (slotPartition l).blocks.length = l.blocks.size := by
    rw [slotPartition_blocks, List.length_map, Array.length_toList]
  let E : (Fin L → Fin 256) → Prop := fun src =>
    (Slots.select rule d query table cm (bytesOf src)).isOk = true ∧
      Misses B (checkedOf (Slots.select rule d query table cm (bytesOf src)))
  refine ⟨fun r => Pr fun dd => E (bytes (forkOn (D.stage2 x) r dd)), fun r => Pr_nonneg _,
    fun r => Pr_mono fun dd hm => ?_, ?_⟩
  · rw [hch] at hm
    unfold drawChecked at hm
    split at hm
    · rename_i l' c hs
      obtain ⟨hl', draw, proved, read, -, rfl⟩ := slots_select_ok _ _ _ _ _ _ _ _ hs
      rw [hl] at hl'
      obtain rfl := Except.ok.inj hl'
      refine ⟨by rw [hs]; rfl, ?_⟩
      simp only [hs, checkedOf]
      intro v hv hi
      obtain ⟨p, hp⟩ := index_of_mem (n := nPU D x) (hnPU ▸ size_checked_le l proved) hi
      exact hm v hv p hp
    · exfalso
      have hv0l : v0 < l.blocks.size := hlen ▸ Finset.mem_range.1 (hB hv0)
      obtain ⟨p, hp⟩ := index_of_mem (n := nPU D x) (hnPU ▸ (size_allSlots l).le) ((mem_allSlots _ _).2 hv0l)
      exact hm v0 hv0 p hp
  · have hav : Ex (fun r => Pr fun dd => E (bytes (forkOn (D.stage2 x) r dd))) = Pr fun c => E (bytes c) := by
      classical
      rw [← Ex_ind fun c => E (bytes c)]
      rw [← Ex_resample (fun _ => D.stage2 x) (fun _ _ => rfl) fun c => if E (bytes c) then 1 else 0]
      exact congrArg Ex (funext fun r => (Ex_ind _).symm)
    rw [hav, hunif]
    have hb := drawMisses rule d query table cm L B
    simp only [missBound, hl, hds] at hb
    exact hb

end Claims.Sampling
