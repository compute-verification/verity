/-
A row's `b ‖ c` in a satisfying witness of the accepted template statement. `HmRow.rowSlots` puts port `p`'s compression
`k` at `sha512x3` slot `(at_p + k) / 3`, place `(at_p + k) % 3`, with `at_p` the earlier ports' chunks (`rowSlots_eq`).
So the satisfying witness holds the `sha512x3` and `hm96` rows at every compression of the row (`shaSlot_of_sat`,
`hmSlot_of_sat`), and `port_commit_rows` gives the row's commit string at its `hm96` slot's outputs (`row_bc_of_sat`),
given the port's words fill its chunks and the block holds Δ's copies into the slots' `cv` and `pad` inputs.
-/
import FlockSoundness.Protocol.Discharge.Hm.Place

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hm

open Compose Flock.DeriveAll

/-- The compressions before port `p`'s. -/
def atOf (ports : Array Port) (p : ℕ) : ℕ := ((ports.toList.take p).map (·.chunks)).sum

/-- Port `p`'s compressions, each its `sha512x3` slot and its place in the slot. -/
def slotsOf (ports : Array Port) (p : ℕ) : Array (ℕ × ℕ) :=
  (Array.range ports[p]!.chunks).map fun k => ((atOf ports p + k) / 3, (atOf ports p + k) % 3)

theorem rowSlots_eq (ports : Array Port) :
    (HmRow.rowSlots ports).2.toList = (List.range ports.size).map (slotsOf ports) := by
  unfold HmRow.rowSlots
  simp only [HmRow.PER_SLOT]
  rw [← Array.forIn_toList]
  have := forIn_id_inv_tail (fun (L : List Port) (s : ℕ × Array (Array (ℕ × ℕ))) =>
      ∃ pre, pre ++ L = ports.toList ∧ s.1 = (pre.map (·.chunks)).sum ∧
        s.2.toList = (List.range pre.length).map (slotsOf ports))
    (f := fun p s => pure (ForInStep.yield (s.1 + p.chunks,
      s.2.push (Array.map (fun k => ((s.1 + k) / 3, (s.1 + k) % 3)) (Array.range p.chunks)))))
    (fun p L s hs => ⟨_, rfl, ?_⟩) ports.toList (0, #[]) ⟨[], by simp, by simp, by simp⟩
  · obtain ⟨pre, hpre, -, h2⟩ := this
    rw [List.append_nil] at hpre
    subst hpre
    rw [Array.length_toList] at h2
    exact h2
  · obtain ⟨pre, hpre, h1, h2⟩ := hs
    have hlen : pre.length < ports.size := by
      rw [← Array.length_toList, ← hpre]; simp
    have hget : ports[pre.length]! = p := by
      rw [getElem!_pos ports pre.length hlen, ← Array.getElem_toList]
      simp only [← hpre, List.getElem_append_right (Nat.le_refl _), Nat.sub_self, List.getElem_cons_zero]
    have hat : atOf ports pre.length = s.1 := by
      rw [h1, atOf, ← hpre, List.take_left' rfl]
    refine ⟨pre ++ [p], by simp [hpre], by simp [h1], ?_⟩
    rw [Array.toList_push, h2, List.length_append, List.length_singleton, List.range_succ, List.map_append,
      List.map_singleton]
    congr 2
    rw [slotsOf, hget, hat]

theorem rowSlots_get {ports : Array Port} {p : ℕ} (hp : p < ports.size) :
    (HmRow.rowSlots ports).2[p]! = slotsOf ports p := by
  have h := rowSlots_eq ports
  have hsz : (HmRow.rowSlots ports).2.size = ports.size := by
    rw [← Array.length_toList, h, List.length_map, List.length_range]
  rw [getElem!_pos _ p (by rw [hsz]; exact hp), ← Array.getElem_toList]
  simp only [h, List.getElem_map, List.getElem_range]

theorem rowSlots_size {ports : Array Port} {p : ℕ} (hp : p < ports.size) :
    ((HmRow.rowSlots ports).2[p]!).size = ports[p]!.chunks := by
  rw [rowSlots_get hp, slotsOf, Array.size_map, Array.size_range]

theorem rowSlots_entry {ports : Array Port} {p k : ℕ} (hp : p < ports.size) (hk : k < ports[p]!.chunks) :
    ((HmRow.rowSlots ports).2[p]!)[k]! = ((atOf ports p + k) / 3, (atOf ports p + k) % 3) := by
  rw [rowSlots_get hp, slotsOf, getElem!_pos _ k (by simp [hk])]
  simp

/-- Compression `k` of port `p`'s row: its slot is one of the VU's `sha512x3` slots, and its place in the slot is below 3. -/
theorem rowSlot_lt {ports : Array Port} {p k : ℕ} (hp : p < ports.size) (hk : k < ports[p]!.chunks) :
    ((HmRow.rowSlots ports).2[p]!)[k]!.1 < (HmRow.rowSlots ports).1 ∧ ((HmRow.rowSlots ports).2[p]!)[k]!.2 < 3 := by
  have hsz : (HmRow.rowSlots ports).2.size = ports.size := by
    rw [← Array.length_toList, rowSlots_eq, List.length_map, List.length_range]
  exact rowSlots_all ports _ (getElem_mem_toList' (by rw [hsz]; exact hp)) _
    (getElem_mem_toList' (by rw [rowSlots_size hp]; exact hk))

section Slots

variable {types : String → Option CircuitType} {ls : Array Flock.Layout} {info : String → Option TableInfo}
  {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D} {st : Stmt} {tb : TypedBlock}

end Slots

end FlockSoundness.Discharge.Hm
