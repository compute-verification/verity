import Proofs.Flock.Claims.Sampling.KS
import Proofs.Flock.Claims.Sampling.Session
import Proofs.Flock.Claims.Sampling.Select.Misses

/-!
# The draw law from the deployed draw (`Claims.Sampling.DrawLaw`)

* `Design.Pa_empty`: a well-formed design has `P_a(∅) = 1`.
* `allSlots`, `drawChecked`: every proof unit of a layout, and the deployed draw's checked units, `Slots.select`'s or,
  where it is refused (the run rejects), every slot.
* `Partition.shape`: each replay unit's count of proof units, in its strata and classes; `P_a`, the slots, their sizes
  and the design's well-formedness read only it (`Design.Pa_of_shape`, `Partition.sizes_of_shape`,
  `Design.wellFormed_of_shape`).
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

theorem Rule.Pa_empty (r : Rule) (c : List (List RU)) (h : r.WellFormed c) :
    r.Pa (strataSizes c) (ruSizes c) ∅ = 1 := by
  obtain ⟨first, check⟩ := r
  unfold Rule.Pa
  split
  · exact Finset.prod_empty
  · exact OneStage.Pa_empty _ _ h.1

theorem classesPa_empty : ∀ (rs : List Rule) (cs : List (List (List RU))) (o : ℕ), rs.length = cs.length →
    (∀ p ∈ rs.zip cs, p.1.WellFormed p.2) → classesPa rs cs o ∅ = 1
  | [], [], _, _, _ => rfl
  | r :: rs, c :: cs, o, hl, h => by
    rw [classesPa, window_empty, Rule.Pa_empty r c (h _ List.mem_cons_self), one_mul]
    exact classesPa_empty rs cs _ (by simpa using hl) fun p hp => h p (List.mem_cons_of_mem _ hp)
  | [], _ :: _, _, hl, _ => absurd hl (by simp)
  | _ :: _, [], _, hl, _ => absurd hl (by simp)

/-- **A well-formed design accepts the empty set surely**: `P_a(∅) = 1`. -/
theorem Design.Pa_empty (d : Design) (P : Partition) (h : d.WellFormed P) : d.Pa P ∅ = 1 :=
  classesPa_empty _ _ 0 h.1 h.2

/-! ## A partition's shape -/

/-- A class's shape: each stratum's replay units' counts of proof units. -/
def classShape (c : List (List RU)) : List (List ℕ) := c.map fun s => s.map List.length

/-- **A partition's shape**: each class's (`classShape`). The design and the draw read a partition only through it: its
slots, their sizes `n_v`, `P_a` and the design's well-formedness. -/
def Partition.shape (P : Partition) : List (List (List ℕ)) := P.classes.map classShape

theorem strataSizes_eq_shape (c : List (List RU)) : strataSizes c = (classShape c).map List.length := by
  simp [strataSizes, classShape, Function.comp_def]

theorem ruSizes_eq_shape (c : List (List RU)) : ruSizes c = (classShape c).flatten := by
  simp [ruSizes, classShape, List.map_flatten]

theorem flatten_length_eq_shape (c : List (List RU)) : c.flatten.length = (classShape c).flatten.length := by
  simp [classShape, List.length_flatten, Function.comp_def]

theorem Partition.sizes_eq_shape (P : Partition) : P.sizes = P.shape.flatten.flatten := by
  simp only [Partition.sizes, Partition.rus, Partition.shape, List.map_flatten]
  rfl

theorem Partition.sizes_of_shape {P Q : Partition} (h : P.shape = Q.shape) : P.sizes = Q.sizes := by
  rw [P.sizes_eq_shape, Q.sizes_eq_shape, h]

theorem Partition.rus_length_of_shape {P Q : Partition} (h : P.shape = Q.shape) : P.rus.length = Q.rus.length := by
  simpa [Partition.sizes] using congrArg List.length (Partition.sizes_of_shape h)

/-- Partitions of one shape whose replay units are one proof unit each: either both or neither. -/
theorem Partition.single_of_shape {P Q : Partition} (h : P.shape = Q.shape) (hQ : ∀ r ∈ Q.rus, r.length = 1) :
    ∀ r ∈ P.rus, r.length = 1 := by
  intro r hr
  have hm : r.length ∈ Q.sizes := by
    rw [← Partition.sizes_of_shape h]
    exact List.mem_map_of_mem hr
  obtain ⟨r', hr', he⟩ := List.mem_map.1 hm
  rw [← he, hQ r' hr']

theorem classesPa_of_shape : ∀ (rs : List Rule) (cs cs' : List (List (List RU))) (o : ℕ) (B : Finset ℕ),
    cs.map classShape = cs'.map classShape → classesPa rs cs o B = classesPa rs cs' o B
  | r :: rs, c :: cs, c' :: cs', o, B, h => by
    simp only [List.map_cons, List.cons.injEq] at h
    rw [classesPa, classesPa, strataSizes_eq_shape c, strataSizes_eq_shape c', ruSizes_eq_shape c,
      ruSizes_eq_shape c', flatten_length_eq_shape c, flatten_length_eq_shape c', h.1,
      classesPa_of_shape rs cs cs' _ B h.2]
  | [], _, _, _, _, _ => by simp [classesPa]
  | _ :: _, [], [], _, _, _ => rfl
  | _ :: _, [], _ :: _, _, _, h => absurd h (by simp)
  | _ :: _, _ :: _, [], _, _, h => absurd h (by simp)

/-- **`P_a` reads only the shape.** -/
theorem Design.Pa_of_shape (d : Design) {P Q : Partition} (h : P.shape = Q.shape) (B : Finset ℕ) :
    d.Pa P B = d.Pa Q B :=
  classesPa_of_shape _ _ _ 0 B h

theorem Rule.wellFormed_of_shape (r : Rule) {c c' : List (List RU)} (h : classShape c = classShape c') :
    r.WellFormed c ↔ r.WellFormed c' := by
  simp only [Rule.WellFormed, strataSizes_eq_shape, h]

theorem forall_zip_of_shape : ∀ (rs : List Rule) (cs cs' : List (List (List RU))),
    cs.map classShape = cs'.map classShape →
      ((∀ p ∈ rs.zip cs, p.1.WellFormed p.2) ↔ ∀ p ∈ rs.zip cs', p.1.WellFormed p.2)
  | [], _, _, _ => by simp
  | _ :: _, [], [], _ => Iff.rfl
  | r :: rs, c :: cs, c' :: cs', h => by
    simp only [List.map_cons, List.cons.injEq] at h
    simp only [List.zip_cons_cons, List.forall_mem_cons]
    rw [Rule.wellFormed_of_shape r h.1, forall_zip_of_shape rs cs cs' h.2]
  | _ :: _, [], _ :: _, h => absurd h (by simp)
  | _ :: _, _ :: _, [], h => absurd h (by simp)

/-- **The design's well-formedness reads only the shape.** -/
theorem Design.wellFormed_of_shape (d : Design) {P Q : Partition} (h : P.shape = Q.shape) :
    d.WellFormed P ↔ d.WellFormed Q := by
  have hl : P.classes.length = Q.classes.length := by
    simpa [Partition.shape] using congrArg List.length h
  unfold Design.WellFormed
  rw [hl, forall_zip_of_shape _ _ _ h]

/-! ## The deployed draw's checked units -/

/-- Every proof unit of a layout, as `(slot, index)`, in slot order. -/
def allSlots (l : Slots.Layout) : Array (ℕ × ℕ) :=
  l.rus.zipIdx.flatMap fun (r, v) => (Array.range r.pus.size).map (v, ·)

theorem mem_allSlots (l : Slots.Layout) (v i : ℕ) :
    (v, i) ∈ allSlots l ↔ ∃ h : v < l.rus.size, i < l.rus[v].pus.size := by
  unfold allSlots
  simp only [Array.mem_flatMap, Array.mem_map, Array.mem_range, Prod.mk.injEq]
  constructor
  · rintro ⟨⟨r, w⟩, hz, i', hi', rfl, rfl⟩
    dsimp only at hi' ⊢
    rw [Array.mem_zipIdx_iff_getElem?] at hz
    dsimp only at hz
    obtain ⟨hv, rfl⟩ := Array.getElem?_eq_some_iff.1 hz
    exact ⟨hv, hi'⟩
  · rintro ⟨hv, hi⟩
    exact ⟨(l.rus[v], v), Array.mem_zipIdx_iff_getElem?.2 (Array.getElem?_eq_getElem hv), i, hi, rfl, rfl⟩

theorem slotPartition_rus (l : Slots.Layout) :
    (slotPartition l).rus = l.rus.toList.map fun r => r.pus.toList.map puGates := by
  simp [slotPartition, Partition.rus, Slots.Layout.rus, List.map_flatten, Function.comp_def]

theorem slotPartition_sizes (l : Slots.Layout) : (slotPartition l).sizes = l.rus.toList.map fun r => r.pus.size := by
  simp [Partition.sizes, slotPartition_rus, Function.comp_def]

theorem size_allSlots (l : Slots.Layout) : (allSlots l).size = (slotPartition l).sizes.sum := by
  rw [slotPartition_sizes, ← Array.length_toList]
  simp only [allSlots, Array.toList_flatMap, List.length_flatMap, Array.toList_map, List.length_map,
    Array.toList_range, List.length_range, Array.toList_zipIdx]
  rw [show (fun a : Slots.RU × ℕ => a.1.pus.size) = (fun r => r.pus.size) ∘ Prod.fst from rfl, ← List.map_map,
    List.zipIdx_map_fst]

theorem size_checked_le (l : Slots.Layout) (proved : Array ℕ) :
    (Slots.checked l proved).size ≤ (slotPartition l).sizes.sum := by
  rw [← size_allSlots, ← Array.length_toList, ← Array.length_toList]
  simp only [Slots.checked, allSlots, Array.toList_flatMap, Array.toList_filter]
  exact (List.filter_sublist.flatMap _).length_le

/-- **The proof units the deployed draw checks** on the bytes `src`: `Slots.select`'s, and every proof unit of the
layout `l` where it is refused, where the run rejects. -/
def drawChecked (rule : Select.Rule) (d : Program) (query : Json) (table : Option Json)
    (cm : Option (Array (Int × Array Int))) (l : Slots.Layout) (src : ByteArray) : Array (ℕ × ℕ) :=
  match Slots.select rule d query table cm src with
  | .ok (_, c) => c
  | .error _ => allSlots l

/-- A proof unit of `c` is at an index below `n` when `c` has at most `n`. -/
theorem index_of_mem {c : Array (ℕ × ℕ)} {n : ℕ} (hc : c.size ≤ n) {u : ℕ × ℕ} (hu : u ∈ c) :
    ∃ p : Fin n, c.toList[p.val]? = some u := by
  obtain ⟨p, hp⟩ := List.mem_iff_getElem?.1 (Array.mem_toList_iff.2 hu)
  have := (List.getElem?_eq_some_iff.1 hp).1
  exact ⟨⟨p, by rw [Array.length_toList] at this; omega⟩, hp⟩

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin]

/-- **The deployed draw is `Slots.select`'s**: on a statement that parses and whose query evaluates, the verifier's own
program `d` (its parse of the statement's descriptor), query, selection rule, work table and closure map, whose layout
`l` has the statement's partition's shape
(`Partition.shape`: its slots, each with its count of proof units, in their strata and classes; the proof units' gates
are the partition's) and whose design is the statement's; `L` bytes of the coins, uniform when the coins are; and the
checked proof units the draw's on them (`drawChecked`). -/
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
  have hnPU : nPU D x = (slotPartition l).sizes.sum := by
    simp [nPU, hx, Deploy.part, hP, Partition.sizes_of_shape hshape]
  rw [Partition.rus_length_of_shape hshape] at hB
  rw [Partition.sizes_of_shape hshape, Design.Pa_of_shape _ hshape]
  have hwf := (Design.wellFormed_of_shape _ hshape).1 hin.2.2
  by_cases hB0 : B = ∅
  · subst hB0
    refine ⟨fun _ => 1, fun _ => zero_le_one, fun r f _ => Pr_le_one _, ?_⟩
    rw [Ex_const, Design.Pa_empty _ _ hwf]
  obtain ⟨v0, hv0⟩ := Finset.nonempty_iff_ne_empty.2 hB0
  have hlen : (slotPartition l).rus.length = l.rus.size := by rw [slotPartition_rus, List.length_map, Array.length_toList]
  have hsz : ∀ v (hv : v < l.rus.size), (slotPartition l).sizes.getD v 0 = l.rus[v].pus.size := by
    intro v hv
    rw [slotPartition_sizes, List.getD_eq_getElem _ _ (by rw [List.length_map, Array.length_toList]; exact hv),
      List.getElem_map, Array.getElem_toList]
  let E : (Fin L → Fin 256) → Prop := fun src =>
    (Slots.select rule d query table cm (bytesOf src)).isOk = true ∧
      Misses B (checkedOf (Slots.select rule d query table cm (bytesOf src)))
  refine ⟨fun r => Pr fun dd => E (bytes (forkOn (D.stage2 x) r dd)), fun r => Pr_nonneg _,
    fun r f hf => Pr_mono fun dd hm => ?_, ?_⟩
  · rw [hch] at hm
    unfold drawChecked at hm
    split at hm
    · rename_i l' c hs
      obtain ⟨hl', draw, proved, read, -, rfl⟩ := slots_select_ok _ _ _ _ _ _ _ _ hs
      rw [hl] at hl'
      obtain rfl := Except.ok.inj hl'
      refine ⟨by rw [hs]; rfl, ?_⟩
      simp only [hs, checkedOf]
      intro v hv i hi
      obtain ⟨hvl, hu, -⟩ := (mem_checked _ _ _ _).1 hi
      have hf' := hf v hv
      rw [hsz v hvl] at hf'
      obtain ⟨p, hp⟩ := index_of_mem (hnPU ▸ size_checked_le l proved) ((mem_checked _ _ _ _).2 ⟨hvl, hu, hf'⟩)
      exact hm v hv p hp
    · exfalso
      have hv0l : v0 < l.rus.size := hlen ▸ Finset.mem_range.1 (hB hv0)
      have hf' := hf v0 hv0
      rw [hsz v0 hv0l] at hf'
      obtain ⟨p, hp⟩ := index_of_mem (hnPU ▸ (size_allSlots l).le) ((mem_allSlots _ _ _).2 ⟨hv0l, hf'⟩)
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
