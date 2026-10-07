/-
Δ's copies into the rows' circuits' inputs. Each of Δ's later writes is a copy `(dst, dst), (dst, src)` (`pair2`), and at
each `cv`, `pad` and chain bit of a row there is exactly one: the rows' wires (`rowWires_eq`) go into distinct inputs,
since port `p`'s compressions sit at the distinct places `at_p + k` (`rowSlots_eq`).
-/
import Proofs.Flock.Soundness.Discharge.Hm.Delta

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hm

/-- Port `p`'s chain wire into compression `k ≥ 1`, and its `hm96` wire. -/
def chainWire (ports : Array Port) (sha p k : ℕ) : Wire :=
  ⟨(sha, ((HmRow.rowSlots ports).2[p]!)[k]!.1, 2 * ((HmRow.rowSlots ports).2[p]!)[k]!.2),
    (sha, ((HmRow.rowSlots ports).2[p]!)[k - 1]!.1, ((HmRow.rowSlots ports).2[p]!)[k - 1]!.2)⟩

def hmWire (ports : Array Port) (sha hm p : ℕ) : Wire :=
  ⟨(hm, p, 0), (sha, ((HmRow.rowSlots ports).2[p]!)[((HmRow.rowSlots ports).2[p]!).size - 1]!.1,
    ((HmRow.rowSlots ports).2[p]!)[((HmRow.rowSlots ports).2[p]!).size - 1]!.2)⟩

theorem forIn_range1_append {γ : Type} (G : ℕ → List γ) (n : ℕ) {f : ℕ → Array γ → Id (ForInStep (Array γ))}
    (hf : ∀ a, 1 ≤ a → ∀ d, ∃ d', f a d = ForInStep.yield d' ∧ d'.toList = d.toList ++ G a) (d : Array γ) :
    (forIn [1:n] d f).toList = d.toList ++ (List.range' 1 (n - 1)).flatMap G := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  have := forIn_id_append G (List.range' 1 [1:n].size) (fun a ha d => hf a
    (by simp only [Std.Legacy.Range.size, List.mem_range'_1] at ha; omega) d) d
  simpa [Std.Legacy.Range.size] using this

/-- An unsegmented row's only chain start is its first compression. -/
theorem chainStart_unseg {p : Port} (hs : p.segs = none) (k : ℕ) :
    HmRow.chainStart p k = if k == 0 then some 0 else none := by
  simp only [HmRow.chainStart, hs]

/-- **The rows' wires**, port by port, when every row has a compression and none is segmented: compression `k`'s `cv` from compression
`k − 1`'s output for `k ≥ 1`, then the `hm96` slot's `cv` from the last compression's output. -/
theorem rowWires_eq (ports : Array Port) (sha hm : ℕ) (hpos : ∀ p < ports.size, 0 < ports[p]!.chunks)
    (hs : ∀ p < ports.size, ports[p]!.segs = none) :
    (HmRow.rowWires ports sha hm).toList = (List.range (HmRow.rowSlots ports).2.size).flatMap fun p =>
      (List.range' 1 (((HmRow.rowSlots ports).2[p]!).size - 1)).map (chainWire ports sha p) ++
        [hmWire ports sha hm p] := by
  unfold HmRow.rowWires
  simp only
  have hS : (HmRow.rowSlots ports).2.size = ports.size := by
    rw [← Array.length_toList, rowSlots_eq, List.length_map, List.length_range]
  show (forIn (m := Id) [:(HmRow.rowSlots ports).2.size] (#[] : Array Wire) _).toList = _
  refine Eq.trans (forIn_range_append (fun p =>
      List.map (chainWire ports sha p) (List.range' 1 (((HmRow.rowSlots ports).2[p]!).size - 1)) ++
        [hmWire ports sha hm p]) _ (fun p hp d => ?_) #[]) ?_
  · have h0 : 0 < ((HmRow.rowSlots ports).2[p]!).size := by
      rw [rowSlots_size (hS ▸ hp)]; exact hpos _ (hS ▸ hp)
    have hcs := chainStart_unseg (hs _ (hS ▸ hp))
    have h1 : (0 < ((HmRow.rowSlots ports).2[p]!).size &&
        (HmRow.chainStart ports[p]! ((HmRow.rowSlots ports).2[p]!).size).isNone) = true := by
      rw [hcs]; simp [h0, Nat.pos_iff_ne_zero.1 h0]
    simp only [h1, ↓reduceIte]
    refine ⟨_, rfl, ?_⟩
    rw [Array.toList_push, ← List.append_assoc]
    refine (List.append_left_inj _).mpr ?_
    refine Eq.trans (forIn_range1_append (fun k => [chainWire ports sha p k]) _ (fun k hk d => ?_) d) ?_
    · have hk' : (HmRow.chainStart ports[p]! k).isNone = true := by
        rw [hcs]; simp [show k ≠ 0 by omega]
      simp only [hk', ↓reduceIte]
      refine ⟨_, rfl, ?_⟩
      rw [Array.toList_push]
      rfl
    · rw [List.map_eq_flatMap]
  · simp

/-! ## Copies, counted -/

/-- The two pairs Δ writes for one copy `(dst, src)`. -/
def pair2 (e : ℕ × ℕ) : List (ℕ × ℕ) := [(e.1, e.1), (e.1, e.2)]

theorem count_pair2 (L : List (ℕ × ℕ)) (x j : ℕ) :
    (L.flatMap pair2).count (x, j) =
      ((L.filter fun e => e.1 = x).map fun e => (if j = x then 1 else 0) + (if j = e.2 then 1 else 0)).sum := by
  induction L with
  | nil => rfl
  | cons e L ih =>
      rw [List.flatMap_cons, List.count_append, ih, List.filter_cons]
      by_cases he : e.1 = x
      · subst he
        simp only [pair2, decide_true, ite_true, List.map_cons, List.sum_cons, List.count_cons, List.count_nil,
          beq_iff_eq, Prod.mk.injEq, true_and, zero_add]
        by_cases h1 : e.1 = j <;> by_cases h2 : e.2 = j <;> simp [h1, h2, eq_comm] <;> omega
      · have h0 : (pair2 e).count (x, j) = 0 := List.count_eq_zero.2 (by
          simp only [pair2, List.mem_cons, List.not_mem_nil, or_false, Prod.mk.injEq, not_or, not_and]
          exact ⟨fun h => absurd h.symm he, fun h => absurd h.symm he⟩)
        simp [h0, he]

theorem filter_flatMap_none {α β : Type} (l : List α) (F : α → List β) (p : β → Bool)
    (h : ∀ a ∈ l, (F a).filter p = []) : (l.flatMap F).filter p = [] := by
  induction l with
  | nil => rfl
  | cons a l ih =>
      rw [List.flatMap_cons, List.filter_append, h a List.mem_cons_self,
        ih fun b hb => h b (List.mem_cons_of_mem _ hb), List.nil_append]

theorem filter_flatMap_single {α β : Type} (l : List α) (hl : l.Nodup) (F : α → List β) (p : β → Bool) (a₀ : α)
    (h₀ : a₀ ∈ l) (hne : ∀ a ∈ l, a ≠ a₀ → (F a).filter p = []) : (l.flatMap F).filter p = (F a₀).filter p := by
  induction l with
  | nil => cases h₀
  | cons a l ih =>
      rw [List.nodup_cons] at hl
      rw [List.flatMap_cons, List.filter_append]
      by_cases ha : a = a₀
      · subst ha
        rw [filter_flatMap_none l F p fun b hb => hne b (List.mem_cons_of_mem _ hb) (fun e => hl.1 (e ▸ hb)),
          List.append_nil]
      · rw [hne a List.mem_cons_self ha, List.nil_append]
        exact ih hl.2 ((List.mem_cons.1 h₀).resolve_left (Ne.symm ha)) fun b hb => hne b (List.mem_cons_of_mem _ hb)

theorem filter_map_single {β : Type} (n : ℕ) (D : ℕ) (S : ℕ → β) (t₀ : ℕ) (ht₀ : t₀ < n) :
    ((List.range n).map fun t => (D + t, S t)).filter (fun e => e.1 = D + t₀) = [(D + t₀, S t₀)] := by
  rw [List.map_eq_flatMap, filter_flatMap_single _ List.nodup_range _ _ t₀ (List.mem_range.2 ht₀)
    (fun t _ ht => by simp [ht])]
  simp

theorem filter_map_none {β : Type} (l : List ℕ) (D : ℕ) (S : ℕ → β) (x : ℕ) (h : ∀ t ∈ l, D + t ≠ x) :
    (l.map fun t => (D + t, S t)).filter (fun e => e.1 = x) = [] := by
  rw [List.map_eq_flatMap]
  exact filter_flatMap_none _ _ _ fun t ht => by simp [h t ht]

/-- Δ's copies into the rows' constant bits. -/
def rowCopies (c : Circuit) (pin : ℕ) : List (ℕ × ℕ) :=
  (List.range c.g).flatMap fun g => (List.range (HmRow.rowSlots c.ports).2.size).flatMap fun p =>
    (List.range 512).map (fun t => (cvCol c g p + t, constSrc c pin (HmRow.midstate c.ports[p]!.words) t)) ++
      (List.range 1024).map fun t => (padCol c g p + t, constSrc c pin (HmRow.padWords c.ports[p]!.words) t)

/-- Δ's copies along the wires. -/
def wireCopies (c : Circuit) : List (ℕ × ℕ) :=
  (List.range c.g).flatMap fun g => c.wires.toList.flatMap fun w =>
    (List.range (Flat.wireWidth c w)).map fun t => (Flat.wireDst c g w + t, Flat.wireSrc c g w + t)

theorem rowPairs_eq (c : Circuit) (pin : ℕ) : rowPairs c pin = (rowCopies c pin).flatMap pair2 := by
  simp only [rowPairs, rowCopies, List.flatMap_assoc, List.flatMap_append, List.flatMap_map, pair2]

theorem wirePairs_eq (c : Circuit) : Flat.wirePairs c = (wireCopies c).flatMap pair2 := by
  simp only [Flat.wirePairs, wireCopies, List.flatMap_assoc, List.flatMap_map, pair2]

theorem leafPairs_nil {c : Circuit} (h1 : c.leavesIn = #[#[]]) (h2 : c.upv = 1) : Flat.leafPairs c = [] := by
  simp [Flat.leafPairs, h1, h2]

theorem cutPairs_nil {c : Circuit} (h : c.leafCuts = #[]) : Flat.cutPairs c = [] := by
  simp [Flat.cutPairs, h]

end FlockSoundness.Discharge.Hm
