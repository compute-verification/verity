/-
A row's `b ‖ c` in a satisfying witness of the accepted template statement. `HmRow.rowSlots` puts port `p`'s compression
`k` at `sha512x3` slot `(at_p + k) / 3`, place `(at_p + k) % 3`, with `at_p` the earlier ports' chunks (`rowSlots_eq`).
So the satisfying witness holds the `sha512x3` and `hm96` rows at every compression of the row (`shaSlot_of_sat`,
`hmSlot_of_sat`), and `port_commit_rows` gives the row's commit string at its `hm96` slot's outputs (`row_bc_of_sat`),
given the port's words fill its chunks and the block holds Δ's copies into the slots' `cv` and `pad` inputs.
-/
import Proofs.Flock.Soundness.Discharge.Hm.Place
import Proofs.Flock.Soundness.ExecTemplateSetup

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hm

open Compose Flock.DeriveAll Types.Dag

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

/-- Compression `k` of port `p`'s row: its slot is one of the template instance's `sha512x3` slots, and its place in the slot is below 3. -/
theorem rowSlot_lt {ports : Array Port} {p k : ℕ} (hp : p < ports.size) (hk : k < ports[p]!.chunks) :
    ((HmRow.rowSlots ports).2[p]!)[k]!.1 < (HmRow.rowSlots ports).1 ∧ ((HmRow.rowSlots ports).2[p]!)[k]!.2 < 3 := by
  have hsz : (HmRow.rowSlots ports).2.size = ports.size := by
    rw [← Array.length_toList, rowSlots_eq, List.length_map, List.length_range]
  exact rowSlots_all ports _ (getElem_mem_toList' (by rw [hsz]; exact hp)) _
    (getElem_mem_toList' (by rw [rowSlots_size hp]; exact hk))

section Slots

variable {types : String → Option CircuitType} {ls : Array Flock.Layout} {info : String → Option TableInfo}
  {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D} {st : Stmt} {tb : TypedBlock}

set_option hygiene false in
local notation "𝒖" => done.getD unit default

/-- **A row's `b ‖ c` in a satisfying witness of the accepted template statement.** At template instance `g`, port `p`'s
row: if it is not segmented (`hs`) and its words fill its chunks, at least one (`hn`, `hw`) and the block holds Δ's
copies into the slots' inputs (the first compression's `cv` the row prefix's midstate, each later `cv` the previous
compression's output, the `hm96` slot's `cv` the last output and its `pad` the padding block), then the 128 bytes the
block holds at `bcCol` are the commit string of the `2 · words` row bytes at `Typed.msgCol` and the 192 salt bytes at
`saltCol`. -/
theorem row_bc_of_sat {tags : Tags} {t : Typed.Template} (hd : deriveChecked types ls info words unit = .ok done)
    (hul : unit < ls.size) (T : TemplateLayout st tb words done 𝒖) (U : UnitShape done 𝒖)
    (P : ParseFactsT tags t st.c) (C : CheckFactsT st.c tb)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (T.model regs).m}
    (hz : (T.model regs).Satisfies z) (o : Fin (2 ^ (T.model regs).nbl)) {g p : ℕ} (hg : g < st.c.g)
    (hp : p < st.c.ports.size) (hs : st.c.ports[p]!.segs = none) (hn : 0 < st.c.ports[p]!.chunks)
    (hw : 2 * st.c.ports[p]!.words = 128 * st.c.ports[p]!.chunks)
    (hcv0 : ∀ t < 512, blockBit (T.model regs) z o (shaBase st.c g p 0 + shaIn st.c (2 * shaSub st.c p 0) + t) =
      wordsBit (HmRow.midstate st.c.ports[p]!.words) t)
    (hchain : ∀ k, k + 1 < st.c.ports[p]!.chunks → ∀ t < 512,
      blockBit (T.model regs) z o (shaBase st.c g p (k + 1) + shaIn st.c (2 * shaSub st.c p (k + 1)) + t) =
        blockBit (T.model regs) z o (shaBase st.c g p k + shaOut st.c (shaSub st.c p k) + t))
    (hcvh : ∀ t < 512, blockBit (T.model regs) z o (st.c.slotCol st.c.hmNet g p + hmIn st.c 0 + t) =
      blockBit (T.model regs) z o
        (shaBase st.c g p (st.c.ports[p]!.chunks - 1) + shaOut st.c (shaSub st.c p (st.c.ports[p]!.chunks - 1)) + t))
    (hpad : ∀ t < 1024, blockBit (T.model regs) z o (st.c.slotCol st.c.hmNet g p + hmIn st.c 1 + t) =
      wordsBit (HmRow.padWords st.c.ports[p]!.words) t) :
    bytesOf (blockBit (T.model regs) z o) 128 (bcCol st.c g p) = Binding.commitString
      ((HmRow.rowPrefix st.c.ports[p]!.words).data.toList ++
        bytesOf (blockBit (T.model regs) z o) (2 * st.c.ports[p]!.words) (Typed.msgCol st.c g p))
      (bytesOf (blockBit (T.model regs) z o) 192 (saltCol st.c g p)) := by
  refine port_commit_rows st.c _ g p hs hn hw (rowSlots_size hp).ge P.rowNets (fun k hk => (rowSlot_lt hp hk).2)
    (fun k hk => ?_) ?_ hcv0 hchain hcvh hpad
  · exact shaSlot_of_sat hd hul T U hz o P.rowNets hg (by rw [C.perVu_sha]; exact (rowSlot_lt hp hk).1)
  · exact hmSlot_of_sat hd hul T U hz o P.rowNets hg (by rw [C.perVu_hm]; exact lt_of_lt_of_le hp (le_nextPow2 _))

end Slots

end FlockSoundness.Discharge.Hm
