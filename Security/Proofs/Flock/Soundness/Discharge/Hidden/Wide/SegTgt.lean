import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegPort
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegRowBc

/-!
# A segmented row's `b ‖ c` in a satisfying witness

The copies `port_commit_seg` takes, from Δ: at a segmented row's own columns (`segPos`, compression `k`'s inputs, the
`hm96` slot's at `k = chunks`), Δ's copies are the row's (`filter_rowsAny`): a chain start's `cv` bit its midstate's
(`segCv_filter_start`), any other `cv` bit the chain's or the `hm96` wire's (`segCv_filter_chain`, `segCv_filter_hm`),
and a message bit the constant or digest bit `HmRow.msgBit` names (`segMsg_filter`). So in every block of a satisfying
witness of the untyped statement those bits hold what `port_commit_seg` needs (`seg_copies_tgt`), and a segmented row's
`b ‖ c` is the commit string of `RowSeg.enc` of its bytes (`seg_bc_tgt`).
-/

set_option Elab.async false

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

open Hm Compose Flock.DeriveAll Types.Dag
open Binding.RowSeg (count segBits enc)
open SegShape

namespace Any

/-! ## A segmented row's columns -/

/-- Offset `off` of compression `k`'s inputs at template instance `g`'s row `p`, its `hm96` slot's at `k = chunks`. -/
def segPos (c : Circuit) (g p k off : ℕ) : ℕ :=
  if k < c.ports[p]!.chunks then shaPos c g p k off else c.slotCol c.hmNet g p + off

section Pos

variable {c : Circuit} (R : RowRanges c)
include R

theorem segCv_pos {g p k : ℕ} (hp : p < c.ports.size) (t : ℕ) : segCvCol c g p k + t = segPos c g p k t := by
  unfold segPos
  split
  · exact (segCols_sha R hp (by assumption) t).1
  · exact (segCols_hm R hp (by assumption) t).1

theorem segMsg_pos {g p k : ℕ} (hp : p < c.ports.size) (t : ℕ) :
    segMsgCol c g p k + t = segPos c g p k (512 + t) := by
  unfold segPos
  split
  · exact (segCols_sha R hp (by assumption) t).2
  · exact (segCols_hm R hp (by assumption) t).2

theorem segPos_inj {g p k off k' off' : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hk : k ≤ c.ports[p]!.chunks)
    (ho : off < 1536) (hk' : k' ≤ c.ports[p]!.chunks) (ho' : off' < 1536)
    (h : segPos c g p k off = segPos c g p k' off') : k = k' ∧ off = off' := by
  unfold segPos at h
  split at h <;> split at h
  · obtain ⟨-, -, h1, h2⟩ := shaPos_inj R hg hp (by assumption) ho hg hp (by assumption) ho' h
    exact ⟨h1, h2⟩
  · exact absurd h (shaPos_ne_hm R hg hp (by assumption) ho hg hp (by omega))
  · exact absurd h.symm (shaPos_ne_hm R hg hp (by assumption) ho' hg hp (by omega))
  · exact ⟨by omega, by omega⟩

omit R in
theorem segPos_dst {g p k off : ℕ} (ho : off < 1536) :
    Dst c g p (segPos c g p k off) := by
  unfold segPos
  split
  · exact Or.inl ⟨k, off, by assumption, ho, rfl⟩
  · exact Or.inr ⟨off, ho, rfl⟩

omit R in
theorem segCvCopies_mem {pin g p : ℕ} {segs : Array (ℕ × ℕ)} {k : ℕ} {e : ℕ × ℕ} (he : e ∈ segCvCopies c pin g p segs k) :
    ∃ t < 512, e.1 = segCvCol c g p k + t := by
  unfold segCvCopies at he
  split at he
  · simp only [List.mem_map, List.mem_range] at he
    obtain ⟨t, ht, rfl⟩ := he
    exact ⟨t, ht, rfl⟩
  · cases he

omit R in
theorem segMsgCopies_mem {pin g p k : ℕ} {e : ℕ × ℕ} (he : e ∈ segMsgCopies c pin g p k) :
    ∃ t < 1024, e.1 = segMsgCol c g p k + t := by
  simp only [segMsgCopies, List.mem_flatMap, List.mem_range] at he
  obtain ⟨t, ht, he⟩ := he
  exact ⟨t, ht, segBitCopy_fst he⟩

/-- Δ's copies into a segmented row at its compression `k₀`'s inputs are compression `k₀`'s. -/
theorem segCopies_filter {pin g p k₀ off₀ : ℕ} {segs : Array (ℕ × ℕ)} (hg : g < c.g) (hp : p < c.ports.size)
    (hk₀ : k₀ ≤ c.ports[p]!.chunks) (ho₀ : off₀ < 1536) :
    (segCopiesOf c pin g p segs).filter (fun e => e.1 = segPos c g p k₀ off₀) =
      (segCvCopies c pin g p segs k₀ ++ segMsgCopies c pin g p k₀).filter (fun e => e.1 = segPos c g p k₀ off₀) := by
  unfold segCopiesOf
  refine filter_flatMap_single _ List.nodup_range _ _ k₀ (by rw [List.mem_range, rowSlots_size hp]; omega)
    fun k hk hne => filter_nil_of _ _ fun e he => ?_
  rw [List.mem_range, rowSlots_size hp] at hk
  rcases List.mem_append.1 he with he | he
  · obtain ⟨t, ht, he1⟩ := segCvCopies_mem he
    rw [he1, segCv_pos R hp]
    exact fun h => hne (segPos_inj R hg hp (by omega) (by omega) hk₀ ho₀ h).1
  · obtain ⟨t, ht, he1⟩ := segMsgCopies_mem he
    rw [he1, segMsg_pos R hp]
    exact fun h => hne (segPos_inj R hg hp (by omega) (by omega) hk₀ ho₀ h).1

/-- The rows' wires' copies at a segmented row: a chain's into a compression's `cv` that is no chain start, and the
`hm96` wire's into its slot's `cv` when that is none. -/
theorem wireFAny_mem' {g p : ℕ} {e : ℕ × ℕ} (hp : p < c.ports.size) (he : e ∈ wireFAny c g p) :
    (∃ k t, 1 ≤ k ∧ k < c.ports[p]!.chunks ∧ HmRow.chainStart c.ports[p]! k = none ∧ t < 512 ∧
        e = (segPos c g p k t, Flat.wireSrc c g (chainWire c.ports c.shaNet p k) + t)) ∨
      (0 < c.ports[p]!.chunks ∧ HmRow.chainStart c.ports[p]! c.ports[p]!.chunks = none ∧ ∃ t < 512,
        e = (segPos c g p c.ports[p]!.chunks t, Flat.wireSrc c g (hmWire c.ports c.shaNet c.hmNet p) + t)) := by
  unfold wireFAny at he
  obtain ⟨w, hw, he⟩ := List.mem_flatMap.1 he
  obtain ⟨t, ht, rfl⟩ := List.mem_map.1 he
  rw [List.mem_range] at ht
  unfold rowWiresOf at hw
  rcases List.mem_append.1 hw with hw | hw
  · obtain ⟨k, hk, hw⟩ := List.mem_flatMap.1 hw
    rw [List.mem_range'_1, rowSlots_size hp] at hk
    split at hw
    · rename_i hn
      rw [List.mem_singleton] at hw
      subst hw
      have hs := (rowSlot_lt hp (show k < c.ports[p]!.chunks by omega)).2
      rw [chainWidth_eq R.rowNets hs] at ht
      refine Or.inl ⟨k, t, by omega, by omega, Option.isNone_iff_eq_none.1 hn, ht, ?_⟩
      rw [← segCv_pos R hp, (segCols_sha R hp (show k < c.ports[p]!.chunks by omega) t).1,
        chainDst_eq R.rowNets hs, (shaIn_eq R.rowNets (shaSub c p k) hs).1, shaPos]
    · cases hw
  · split at hw
    · rename_i hn
      rw [List.mem_singleton] at hw
      subst hw
      rw [hmWidth_eq R.rowNets] at ht
      simp only [Bool.and_eq_true, decide_eq_true_eq, rowSlots_size hp] at hn
      refine Or.inr ⟨hn.1, Option.isNone_iff_eq_none.1 hn.2, t, ht, ?_⟩
      rw [← segCv_pos R hp, (segCols_hm R hp (lt_irrefl _) t).1]
      show (hmWireDst c g p + t, _) = _
      rw [hmWireDst_eq R.rowNets]
    · cases hw

/-- **Δ's one copy into a chain start's `cv` bit**: its midstate's (a segment's, or the outer message's). -/
theorem segCv_filter_start {pin g p k t₀ j : ℕ} {segs : Array (ℕ × ℕ)} (hs : c.ports[p]!.segs = some segs)
    (hg : g < c.g) (hp : p < c.ports.size) (hk : k ≤ c.ports[p]!.chunks) (ht₀ : t₀ < 512)
    (hj : HmRow.chainStart c.ports[p]! k = some j) :
    (rowCopiesAny c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
        (fun e => e.1 = segCvCol c g p k + t₀) =
      [(segCvCol c g p k + t₀, constSrc c pin
        (if j < segs.size then HmRow.segMidstate c.ports[p]! j else HmRow.midstateOf c.ports[p]!) t₀)] := by
  rw [segCv_pos R hp, filter_rowsAny R hg hp (segPos_dst (by omega))]
  have hw : (wireFAny c g p).filter (fun e => e.1 = segPos c g p k t₀) = [] := filter_nil_of _ _ fun e he => by
    rcases wireFAny_mem' R hp he with ⟨k', t, -, hk', hn, ht, rfl⟩ | ⟨-, hn, t, ht, rfl⟩
    · intro h
      have := (segPos_inj R hg hp (by omega) (by omega) hk (by omega) h).1
      subst this
      rw [hn] at hj; cases hj
    · intro h
      have := (segPos_inj R hg hp le_rfl (by omega) hk (by omega) h).1
      rw [← this, hn] at hj; cases hj
  rw [hw, List.append_nil]
  unfold rowCopiesAnyOf
  rw [hs]
  dsimp only
  rw [segCopies_filter R hg hp hk (by omega), List.filter_append]
  have hm : (segMsgCopies c pin g p k).filter (fun e => e.1 = segPos c g p k t₀) = [] :=
    filter_nil_of _ _ fun e he => by
      obtain ⟨t, ht, he1⟩ := segMsgCopies_mem he
      rw [he1, segMsg_pos R hp]
      intro h
      have := (segPos_inj R hg hp hk (by omega) hk (by omega) h).2
      omega
  rw [hm, List.append_nil]
  unfold segCvCopies
  rw [hj]
  dsimp only
  rw [← segCv_pos R hp]
  exact filter_map_single _ _ _ _ ht₀

/-- **Δ's one copy into a compression's `cv` bit that is no chain start**: the chain's from the previous compression. -/
theorem segCv_filter_chain {pin g p k t₀ : ℕ} {segs : Array (ℕ × ℕ)} (hs : c.ports[p]!.segs = some segs)
    (hg : g < c.g) (hp : p < c.ports.size) (hk1 : 1 ≤ k) (hk : k < c.ports[p]!.chunks) (ht₀ : t₀ < 512)
    (hn : HmRow.chainStart c.ports[p]! k = none) :
    (rowCopiesAny c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
        (fun e => e.1 = segCvCol c g p k + t₀) =
      [(segCvCol c g p k + t₀, Flat.wireSrc c g (chainWire c.ports c.shaNet p k) + t₀)] := by
  rw [segCv_pos R hp, filter_rowsAny R hg hp (segPos_dst (by omega))]
  have hr : (rowCopiesAnyOf c pin g p).filter (fun e => e.1 = segPos c g p k t₀) = [] := by
    unfold rowCopiesAnyOf
    rw [hs]
    dsimp only
    rw [segCopies_filter R hg hp (by omega) (by omega), List.filter_append]
    refine (congrArg₂ (· ++ ·) (filter_nil_of _ _ fun e he => ?_) (filter_nil_of _ _ fun e he => ?_)).trans rfl
    · unfold segCvCopies at he
      rw [hn] at he
      cases he
    · obtain ⟨t, ht, he1⟩ := segMsgCopies_mem he
      rw [he1, segMsg_pos R hp]
      intro h
      have := (segPos_inj R hg hp (by omega) (by omega) (by omega) (by omega) h).2
      omega
  rw [hr, List.nil_append]
  unfold wireFAny rowWiresOf
  rw [List.flatMap_append, List.filter_append]
  have hkm : k ∈ List.range' 1 (((HmRow.rowSlots c.ports).2[p]!).size - 1) := by
    rw [List.mem_range'_1, rowSlots_size hp]; omega
  have hs' := (rowSlot_lt hp hk).2
  have hx : segPos c g p k t₀ = Flat.wireDst c g (chainWire c.ports c.shaNet p k) + t₀ := by
    rw [← segCv_pos R hp, (segCols_sha R hp hk t₀).1, chainDst_eq R.rowNets hs', (shaIn_eq R.rowNets (shaSub c p k) hs').1,
      shaPos]
  rw [List.flatMap_assoc, filter_flatMap_single _ List.nodup_range' _ _ k hkm fun k' hk' hne => filter_nil_of _ _
    fun e he => by
      rw [List.mem_range'_1, rowSlots_size hp] at hk'
      obtain ⟨w, hw, he⟩ := List.mem_flatMap.1 he
      split at hw
      · rw [List.mem_singleton] at hw
        subst hw
        have hs'' := (rowSlot_lt hp (show k' < c.ports[p]!.chunks by omega)).2
        obtain ⟨t, ht, rfl⟩ := List.mem_map.1 he
        rw [List.mem_range, chainWidth_eq R.rowNets hs''] at ht
        show Flat.wireDst c g (chainWire c.ports c.shaNet p k') + t ≠ _
        rw [chainDst_eq R.rowNets hs'', (shaIn_eq R.rowNets (shaSub c p k') hs'').1, ← shaPos,
          ← (segCols_sha R hp (show k' < c.ports[p]!.chunks by omega) t).1, segCv_pos R hp]
        exact fun h => hne (segPos_inj R hg hp (by omega) (by omega) (by omega) (by omega) h).1
      · cases hw]
  have hb : ((if 0 < ((HmRow.rowSlots c.ports).2[p]!).size &&
        (HmRow.chainStart c.ports[p]! ((HmRow.rowSlots c.ports).2[p]!).size).isNone then
          [hmWire c.ports c.shaNet c.hmNet p] else []).flatMap fun w =>
        (List.range (Flat.wireWidth c w)).map fun t => (Flat.wireDst c g w + t, Flat.wireSrc c g w + t)).filter
        (fun e => e.1 = segPos c g p k t₀) = [] := filter_nil_of _ _ fun e he => by
    obtain ⟨w, hw, he⟩ := List.mem_flatMap.1 he
    split at hw
    · rw [List.mem_singleton] at hw
      subst hw
      obtain ⟨t, ht, rfl⟩ := List.mem_map.1 he
      rw [List.mem_range, hmWidth_eq R.rowNets] at ht
      show hmWireDst c g p + t ≠ _
      rw [hmWireDst_eq R.rowNets, ← (segCols_hm R hp (lt_irrefl _) t).1, segCv_pos R hp]
      exact fun h => absurd (segPos_inj R hg hp le_rfl (by omega) (by omega) (by omega) h).1 (by omega)
    · cases hw
  rw [hb, List.append_nil]
  simp only [hn, Option.isNone_none, ↓reduceIte, List.flatMap_cons, List.flatMap_nil, List.append_nil]
  rw [hx, chainWidth_eq R.rowNets hs']
  exact filter_map_single _ _ _ _ ht₀

/-- **Δ's one copy into the `hm96` slot's `cv` bit when that is no chain start**: the `hm96` wire's. -/
theorem segCv_filter_hm {pin g p t₀ : ℕ} {segs : Array (ℕ × ℕ)} (hs : c.ports[p]!.segs = some segs)
    (hg : g < c.g) (hp : p < c.ports.size) (h0 : 0 < c.ports[p]!.chunks) (ht₀ : t₀ < 512)
    (hn : HmRow.chainStart c.ports[p]! c.ports[p]!.chunks = none) :
    (rowCopiesAny c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
        (fun e => e.1 = segCvCol c g p c.ports[p]!.chunks + t₀) =
      [(segCvCol c g p c.ports[p]!.chunks + t₀, Flat.wireSrc c g (hmWire c.ports c.shaNet c.hmNet p) + t₀)] := by
  rw [segCv_pos R hp, filter_rowsAny R hg hp (segPos_dst (by omega))]
  have hr : (rowCopiesAnyOf c pin g p).filter (fun e => e.1 = segPos c g p c.ports[p]!.chunks t₀) = [] := by
    unfold rowCopiesAnyOf
    rw [hs]
    dsimp only
    rw [segCopies_filter R hg hp le_rfl (by omega), List.filter_append]
    refine (congrArg₂ (· ++ ·) (filter_nil_of _ _ fun e he => ?_) (filter_nil_of _ _ fun e he => ?_)).trans rfl
    · unfold segCvCopies at he
      rw [hn] at he
      cases he
    · obtain ⟨t, ht, he1⟩ := segMsgCopies_mem he
      rw [he1, segMsg_pos R hp]
      intro h
      have := (segPos_inj R hg hp le_rfl (by omega) le_rfl (by omega) h).2
      omega
  rw [hr, List.nil_append]
  unfold wireFAny rowWiresOf
  rw [List.flatMap_append, List.filter_append]
  have ha : (((List.range' 1 (((HmRow.rowSlots c.ports).2[p]!).size - 1)).flatMap fun k =>
        if (HmRow.chainStart c.ports[p]! k).isNone then [chainWire c.ports c.shaNet p k] else []).flatMap fun w =>
        (List.range (Flat.wireWidth c w)).map fun t => (Flat.wireDst c g w + t, Flat.wireSrc c g w + t)).filter
        (fun e => e.1 = segPos c g p c.ports[p]!.chunks t₀) = [] := filter_nil_of _ _ fun e he => by
    obtain ⟨w, hw, he⟩ := List.mem_flatMap.1 he
    obtain ⟨k', hk', hw⟩ := List.mem_flatMap.1 hw
    rw [List.mem_range'_1, rowSlots_size hp] at hk'
    split at hw
    · rw [List.mem_singleton] at hw
      subst hw
      have hs'' := (rowSlot_lt hp (show k' < c.ports[p]!.chunks by omega)).2
      obtain ⟨t, ht, rfl⟩ := List.mem_map.1 he
      rw [List.mem_range, chainWidth_eq R.rowNets hs''] at ht
      show Flat.wireDst c g (chainWire c.ports c.shaNet p k') + t ≠ _
      rw [chainDst_eq R.rowNets hs'', (shaIn_eq R.rowNets (shaSub c p k') hs'').1, ← shaPos,
        ← (segCols_sha R hp (show k' < c.ports[p]!.chunks by omega) t).1, segCv_pos R hp]
      exact fun h => absurd (segPos_inj R hg hp (by omega) (by omega) le_rfl (by omega) h).1 (by omega)
    · cases hw
  rw [ha, List.nil_append]
  have hc : (0 < ((HmRow.rowSlots c.ports).2[p]!).size &&
      (HmRow.chainStart c.ports[p]! ((HmRow.rowSlots c.ports).2[p]!).size).isNone) = true := by
    simp only [rowSlots_size hp, hn, Option.isNone_none, Bool.and_true, decide_eq_true_eq]
    exact h0
  rw [if_pos hc, List.flatMap_cons, List.flatMap_nil, List.append_nil, hmWidth_eq R.rowNets]
  have hx : segPos c g p c.ports[p]!.chunks t₀ = Flat.wireDst c g (hmWire c.ports c.shaNet c.hmNet p) + t₀ := by
    rw [← segCv_pos R hp, (segCols_hm R hp (lt_irrefl _) t₀).1]
    show _ = hmWireDst c g p + t₀
    rw [hmWireDst_eq R.rowNets]
  rw [hx]
  exact filter_map_single _ _ _ _ ht₀

/-- **Δ's copies into a segmented row's message bit**: the constant or digest bit `HmRow.msgBit` names, none at a row
bit. -/
theorem segMsg_filter {pin g p k t₀ : ℕ} {segs : Array (ℕ × ℕ)} (hs : c.ports[p]!.segs = some segs)
    (hg : g < c.g) (hp : p < c.ports.size) (hk : k ≤ c.ports[p]!.chunks) (ht₀ : t₀ < 1024) :
    (rowCopiesAny c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
        (fun e => e.1 = segMsgCol c g p k + t₀) =
      segBitCopy c pin g p (segMsgCol c g p k + t₀) (HmRow.msgBit c.ports[p]! k t₀) := by
  rw [segMsg_pos R hp, filter_rowsAny R hg hp (segPos_dst (by omega))]
  have hw : (wireFAny c g p).filter (fun e => e.1 = segPos c g p k (512 + t₀)) = [] := filter_nil_of _ _ fun e he => by
    rcases wireFAny_mem' R hp he with ⟨k', t, -, hk', -, ht, rfl⟩ | ⟨-, -, t, ht, rfl⟩
    · intro h
      have := (segPos_inj R hg hp (by omega) (by omega) hk (by omega) h).2
      omega
    · intro h
      have := (segPos_inj R hg hp le_rfl (by omega) hk (by omega) h).2
      omega
  rw [hw, List.append_nil]
  unfold rowCopiesAnyOf
  rw [hs]
  dsimp only
  rw [segCopies_filter R hg hp hk (by omega), List.filter_append]
  have hc : (segCvCopies c pin g p segs k).filter (fun e => e.1 = segPos c g p k (512 + t₀)) = [] :=
    filter_nil_of _ _ fun e he => by
      obtain ⟨t, ht, he1⟩ := segCvCopies_mem he
      rw [he1, segCv_pos R hp]
      intro h
      have := (segPos_inj R hg hp hk (by omega) hk (by omega) h).2
      omega
  rw [hc, List.nil_append]
  unfold segMsgCopies
  rw [filter_flatMap_single _ List.nodup_range _ _ t₀ (List.mem_range.2 ht₀) fun t ht hne => filter_nil_of _ _
    fun e he => by
      rw [segBitCopy_fst he, segMsg_pos R hp]
      intro h
      have := (segPos_inj R hg hp hk (by have := List.mem_range.1 ht; omega) hk (by omega) h).2
      exact hne (by omega)]
  rw [← segMsg_pos R hp]
  refine List.filter_eq_self.2 fun e he => ?_
  simp [segBitCopy_fst he]

end Pos

/-! ## In a satisfying witness -/

section Bc

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags st)
include L F

/-- **The copies into a segmented row**, in every block of a satisfying witness of the untyped statement: each chain
start's `cv` its midstate, every other `cv` the previous compression's output, and each message bit the constant or
the digest bit `HmRow.msgBit` names. -/
theorem seg_copies_tgt (hda : st.da = (HmRow.delta st.c st.pin).1) (hupv : 0 < st.c.upv)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g p : ℕ} (hg : g < st.c.g)
    (hp : p < st.c.ports.size) {segs : Array (ℕ × ℕ)} (hs : st.c.ports[p]!.segs = some segs) :
    (∀ k ≤ st.c.ports[p]!.chunks, ∀ j, HmRow.chainStart st.c.ports[p]! k = some j → ∀ t < 512,
      blockBit (L.model regs) z o (segCvCol st.c g p k + t) =
        wordsBit (if j < segs.size then HmRow.segMidstate st.c.ports[p]! j else HmRow.midstateOf st.c.ports[p]!) t) ∧
    (∀ k ≤ st.c.ports[p]!.chunks, 0 < k → HmRow.chainStart st.c.ports[p]! k = none → ∀ t < 512,
      blockBit (L.model regs) z o (segCvCol st.c g p k + t) =
        blockBit (L.model regs) z o (shaBase st.c g p (k - 1) + shaOut st.c (shaSub st.c p (k - 1)) + t)) ∧
    (∀ k ≤ st.c.ports[p]!.chunks, ∀ t < 1024, ∀ v, HmRow.msgBit st.c.ports[p]! k t = .const v →
      blockBit (L.model regs) z o (segMsgCol st.c g p k + t) = v) ∧
    (∀ k ≤ st.c.ports[p]!.chunks, ∀ t < 1024, ∀ kk col, HmRow.msgBit st.c.ports[p]! k t = .digest kk col →
      blockBit (L.model regs) z o (segMsgCol st.c g p k + t) = blockBit (L.model regs) z o (digestSrc st.c g p kk col)) := by
  have R := RowRanges.ofL L F.parse F.check
  have hrw : RowWires st.c := F.parse.rowWires
  have hpin := pin_bit L hz o
  have hzero := zero_bit L F hupv (by omega) hz o
  -- every column of the row is a self input bit of its slot
  have hat : ∀ k ≤ st.c.ports[p]!.chunks, ∀ off < 1536, ∀ src,
      (rowCopiesAny st.c st.pin ++
          wireCopiesOf st.c (HmRow.rowWires st.c.ports st.c.shaNet st.c.hmNet).toList).filter
        (fun e => e.1 = segPos st.c g p k off) = [(segPos st.c g p k off, src)] →
      blockBit (L.model regs) z o (segPos st.c g p k off) = blockBit (L.model regs) z o src := by
    intro k hk off ho src hf
    by_cases hks : k < st.c.ports[p]!.chunks
    · have hsub : shaSub st.c p k < 3 := (rowSlot_lt hp hks).2
      have e : segPos st.c g p k off = shaPos st.c g p k off := by unfold segPos; rw [if_pos hks]
      rw [e] at hf ⊢
      exact inBit_copyAll L F hda hrw hz o (Or.inl rfl) (shaPos_at R hg hp hks ho) (shaNet_ofRows R.rowNets)
        (by rw [shaNet_inRows R.rowNets]; omega) (shaLaid_self (by omega)) hf
    · have e : segPos st.c g p k off = st.c.slotCol st.c.hmNet g p + off := by unfold segPos; rw [if_neg hks]
      rw [e] at hf ⊢
      exact inBit_copyAll L F hda hrw hz o (Or.inr rfl) (R.hmIn_at hg hp (off := off) (by omega))
        (hmNet_ofRows R.rowNets) (by rw [hmNet_inRows R.rowNets]; omega) (hmLaid_self (by omega)) hf
  refine ⟨fun k hk j hj t ht => ?_, fun k hk h0 hn t ht => ?_, fun k hk t ht v hv => ?_, fun k hk t ht kk col hv => ?_⟩
  · have hf := segCv_filter_start R (pin := st.pin) hs hg hp hk ht hj
    rw [segCv_pos R hp] at hf ⊢
    rw [hat k hk t (by omega) _ hf]
    exact blockBit_constSrc _ _ hpin hzero
  · by_cases hks : k < st.c.ports[p]!.chunks
    · have hf := segCv_filter_chain R (pin := st.pin) hs hg hp h0 hks ht hn
      rw [segCv_pos R hp] at hf ⊢
      rw [hat k hk t (by omega) _ hf]
      unfold Flat.wireSrc chainWire shaBase shaOut shaSub
      rfl
    · have hkN : k = st.c.ports[p]!.chunks := by omega
      subst hkN
      have hf := segCv_filter_hm R (pin := st.pin) hs hg hp h0 ht hn
      rw [segCv_pos R hp] at hf ⊢
      rw [hat _ hk t (by omega) _ hf]
      unfold Flat.wireSrc hmWire shaBase shaOut shaSub
      simp only [rowSlots_size hp]
  · have hf := segMsg_filter R (pin := st.pin) hs hg hp hk ht
    rw [hv] at hf
    simp only [segBitCopy] at hf
    rw [segMsg_pos R hp] at hf ⊢
    rw [hat k hk _ (by omega) _ hf]
    cases v
    · exact hzero
    · exact hpin
  · have hf := segMsg_filter R (pin := st.pin) hs hg hp hk ht
    rw [hv] at hf
    simp only [segBitCopy] at hf
    rw [segMsg_pos R hp] at hf ⊢
    exact hat k hk _ (by omega) _ hf

/-- **A segmented row's `b ‖ c`**, in every block of a satisfying witness of the untyped statement: at a
`sha512/row-seg/v1` row of `l` bits shaped `HmRow.segShape l` whose segments' bytes reach their last blocks
(`SegFits`), the commit string of `RowSeg.enc l` of the row's `(l + 7) / 8` bytes at `Typed.msgCol`, under the salt its
`hm96` slot reads. -/
theorem seg_bc_tgt (hda : st.da = (HmRow.delta st.c st.pin).1) (hupv : 0 < st.c.upv)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g p l : ℕ} (hg : g < st.c.g)
    (hp : p < st.c.ports.size) (hs : st.c.ports[p]!.segs = some (HmRow.segShape l).1) (hbits : st.c.ports[p]!.bits = l)
    (hn : st.c.ports[p]!.chunks = firstOf (HmRow.segShape l).1 (count l) + outerComps l) (hfit : SegFits l) :
    bytesOf (blockBit (L.model regs) z o) 128 (bcCol st.c g p) = Binding.commitString
      (enc l (bytesOf (blockBit (L.model regs) z o) ((l + 7) / 8) (Typed.msgCol st.c g p)))
      (bytesOf (blockBit (L.model regs) z o) 192 (saltCol st.c g p)) := by
  have R := RowRanges.ofL L F.parse F.check
  have hhm := hmSlot_flat L F hz o hg (by rw [R.perVu_hm]; exact lt_of_lt_of_le hp (le_nextPow2 _))
  obtain ⟨hcvS, hcvC, hmC, hmD⟩ := seg_copies_tgt L F hda hupv hz o hg hp hs
  have hpos := segShape_pos l
  refine port_commit_seg R _ hp hs hbits hn hfit
    (fun k hk => shaSlot_flat L F hz o hg (by rw [R.perVu_sha]; exact (rowSlot_lt hp hk).1)) hhm
    (fun j hj t ht => ?_) hcvC hmC hmD
  have hsz : (HmRow.segShape l).1.size = count l := Binding.RowSeg.size_segShape l
  have hle : firstOf (HmRow.segShape l).1 j ≤ st.c.ports[p]!.chunks := by
    rw [hn]
    have := firstOf_mono hpos hj (by rw [hsz])
    omega
  have := hcvS _ hle j (chainStart_first hpos hs (by rw [hsz]; exact hj)) t ht
  rw [hsz] at this
  exact this

end Bc

end Any

end FlockSoundness.Discharge.Hidden
