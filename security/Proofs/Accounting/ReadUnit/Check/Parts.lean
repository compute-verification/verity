module

public import Proofs.Accounting.ReadUnit.Check.Compose

/-!
# Each call's part, read at the root

For each call of the root, its part's statement on the values the root's columns hold: the segments and the outer hash
(`root_seg`, `root_outer`, together `root_rowSeg`), the hm96 leaf (`root_leaf`), the frame's leaf hash (`root_leafHash`)
and nodes (`root_node`, up the path in `root_climb`), the comparison with the root (`root_eq`) and the select
(`root_select`).
-/

@[expose] public section

namespace Accounting.ReadUnit.Gates

open FlockSoundness FlockSoundness.TmplSpec FlockSoundness.Discharge.Hm Flock.ReadNet
open Flock.HmNets hiding and

variable {V H : ℕ} {z : ℕ → Bool}

/-- Every call's copy holds its part's run (`root_facts`). -/
abbrev Copies (V H : ℕ) (z : ℕ → Bool) : Prop :=
  ∀ i < nCalls V H, ∀ p ∈ (parts V H)[partOf V H i]?, CopyHolds V H z i p

theorem evs_drop (f : Var → Bool) (l : List Lin) (n : ℕ) : evs f (l.drop n) = (evs f l).drop n := by
  simp [evs]

theorem flatten_drop {k : ℕ} {ws : List Word} (h : Shaped k ws) {j : ℕ} (hj : j ≤ k) :
    (ws.drop j).flatten = ws.flatten.drop (64 * j) := by
  conv_rhs => rw [← List.take_append_drop j ws, List.flatten_append]
  rw [List.drop_left' (shaped_flatten_length (k := j)
    ⟨by rw [List.length_take, h.1]; omega, fun w hw => h.2 w (List.mem_of_mem_take hw)⟩)]

theorem drop_outV (i a n : ℕ) : (outV V H z i (a + n)).drop a =
    (List.range n).map fun k => rootVal V H z ((lay V H).outAt i (a + k)) := by
  apply List.ext_getElem (by simp [outV])
  intro t _ _
  simp [outV]

theorem drop_colV (k a n : ℕ) : (colV V H z k (a + n)).drop a = colV V H z (k + a) n := by
  apply List.ext_getElem (by simp [colV])
  intro t _ _
  simp [colV, Nat.add_assoc]

/-- **Segment `j`**: its copy's words hold the digest of the segment's message. -/
theorem root_seg (hf : rootVal V H z (lay V H).zero = false) (hc : rootVal V H z ((lay V H).const + 1) = true)
    (hcp : Copies V H z) {j : ℕ} (hj : j < segs H) :
    (Verity.sha512 ((Flock.HmRow.segmentPrefix j (segs H) (segM H j)).data.toList ++
      Hash.RowSeg.packBits (colV V H z (ROW_AT + Flock.HmRow.ROW_SEG_BITS * j) (segM H j)))).bytes =
      ReadUnit.bytesOf (outV V H z j 512) := by
  have hi : j < nCalls V H := by unfold nCalls; omega
  have hp := seg_part V H hj
  have hP := hcp j hi _ hp
  have hcs := inputs_seg V H hj
  have hlen : (inputs V H (lay V H) j).length = (segPart (segM H j)).ins := by
    rw [hcs]
    simp [length_cvRefs, segPart]
  obtain ⟨o, ho, hspec⟩ := copy_run hP (sound_flat (segment_spec (segM H j)
    (Flock.HmRow.segmentPrefix j (segs H) (segM H j)) (wordsAt 0 8) (bitsAt 512 (segM H j)) (size_segmentPrefix _ _ _)))
  have hcv : (wordsAt 0 8).map (bv (copyV V H z j (segPart (segM H j)))) =
      chainV (Flock.HmRow.segmentPrefix j (segs H) (segM H j)) 1 := by
    rw [← map_bv_cvWords (copyV V H z j (segPart (segM H j))) (zero := Lin.const false) (one := Lin.const true) rfl
      rfl]
    refine map_bv_of_flatten (shaped_wordsAt 0 8).2 (words64_cvWords _ _ _) (by simp [wordsAt, length_cvWords]) ?_
    rw [evs_copy_words (k := 0) (m := 8) hlen (P := []) (Q := cvRefs (lay V H) H j)
      (R := (List.range (segM H j)).map (ROW_AT + Flock.HmRow.ROW_SEG_BITS * j + ·)) (by simp only [hcs, List.nil_append]) rfl
      (length_cvRefs _ _ _)]
    exact map_cvRefs _ _ _ hf hc _
  obtain ⟨o1, o2, o3⟩ := hspec (by simp [wordsAt]) (shaped_wordsAt 0 8).2 hcv (length_bitsAt _ _)
  have hbits : evs (copyV V H z j (segPart (segM H j))) (bitsAt 512 (segM H j)) =
      colV V H z (ROW_AT + Flock.HmRow.ROW_SEG_BITS * j) (segM H j) := by
    rw [evs_copy hlen (k := 512) (n := segM H j) (P := cvRefs (lay V H) H j)
      (Q := (List.range (segM H j)).map (ROW_AT + Flock.HmRow.ROW_SEG_BITS * j + ·)) (R := [])
      (by simp only [hcs, List.append_nil]) (length_cvRefs _ _ _) (by simp), map_at]
  have hout := copy_words_out hi hp hP ho ⟨o1, o2⟩ (nOut_mid V H (by omega) (by omega) (by omega))
  rw [hbits, ← wordsBytes_eq _ ⟨o1, o2⟩, wordsBytes, hout] at o3
  exact o3

/-- **The outer hash**: its copy's words hold SHA-512 of `Q(16 H)` and the segments' digests. -/
theorem root_outer (hcp : Copies V H z) :
    (Verity.sha512 (Hash.RowSeg.outerPrefix (16 * H) ++
      (List.range (segs H)).flatMap fun j => ReadUnit.bytesOf (outV V H z j 512))).bytes =
      ReadUnit.bytesOf (outV V H z (segs H) 512) := by
  have hi : segs H < nCalls V H := by unfold nCalls; omega
  have hp := outer_part V H
  have hP := hcp _ hi _ hp
  have hcs := inputs_outer V H
  have hlen : (inputs V H (lay V H) (segs H)).length = (outerPart (16 * H) (segs H)).ins := by
    rw [hcs]
    simp [outerPart, List.length_flatMap, List.map_const', List.sum_replicate, smul_eq_mul, Nat.mul_comm]
  obtain ⟨o, ho, hspec⟩ := copy_run hP (sound_flat (outer_spec (16 * H) (wordsAt 0 (8 * segs H))))
  obtain ⟨o1, o2, o3⟩ := hspec (shaped_wordsAt 0 _).2
  have hd : evs (copyV V H z (segs H) (outerPart (16 * H) (segs H))) (wordsAt 0 (8 * segs H)).flatten =
      (List.range (segs H)).flatMap fun j => outV V H z j 512 := by
    rw [evs_copy_words (k := 0) (m := 8 * segs H) hlen (P := []) (Q := inputs V H (lay V H) (segs H)) (R := [])
      (by simp) rfl (by rw [hlen]; show 512 * segs H = 64 * (8 * segs H); ring), hcs, List.flatMap_def, List.flatMap_def,
      List.map_flatten, List.map_map]
    refine congrArg List.flatten (List.map_congr_left fun j _ => ?_)
    simp only [Function.comp_apply, map_outs]
  have hout := copy_words_out hi hp hP ho ⟨o1, o2⟩ (nOut_mid V H (by omega) (by omega) (by omega))
  rw [← wordsBytes_eq _ (shaped_wordsAt 0 _), wordsBytes, hd, bytesOf_flatMap _ (fun _ => length_outV _ _),
    ← wordsBytes_eq _ ⟨o1, o2⟩, wordsBytes, hout] at o3
  exact o3

theorem ins_drop_take (k n a b : ℕ) (h : a + b ≤ n) : ((ins k n).drop a).take b = ins (k + a) b := by
  apply List.ext_getElem (by simp [ins]; omega)
  intro t _ _
  simp [ins, Nat.add_assoc]

/-- **Part A at the root**: the outer hash's copy holds the row's `sha512/row-seg/v1` digest. -/
theorem root_rowSeg (hf : rootVal V H z (lay V H).zero = false) (hc : rootVal V H z ((lay V H).const + 1) = true)
    (hcp : Copies V H z) :
    Hash.RowSeg.digest Verity.sha512 (colV V H z ROW_AT (16 * H)) = ReadUnit.digestOf (outV V H z (segs H) 512) := by
  have hrow : evs (rootVal V H z) (ins ROW_AT (16 * H)) = colV V H z ROW_AT (16 * H) := by simp [evs, ins, colV]
  have hseg : ∀ j ∈ List.range (segs H),
      (Verity.sha512 (Hash.RowSeg.segmentMsg (colV V H z ROW_AT (16 * H)) j)).bytes =
        ReadUnit.bytesOf (outV V H z j 512) := by
    intro j hj
    have hj' := List.mem_range.1 hj
    have hb : Flock.HmRow.ROW_SEG_BITS * j + segM H j ≤ 16 * H := by
      simp only [segM, segs, rsb] at *
      omega
    have he : evs (rootVal V H z) (ins (ROW_AT + Flock.HmRow.ROW_SEG_BITS * j) (segM H j)) =
        colV V H z (ROW_AT + Flock.HmRow.ROW_SEG_BITS * j) (segM H j) := by simp [evs, ins, colV]
    rw [← hrow, segmentMsg_eq _ (length_ins _ _) j,
      show min (16 * H - Flock.HmRow.ROW_SEG_BITS * j) Flock.HmRow.ROW_SEG_BITS = segM H j from rfl,
      ins_drop_take _ _ _ _ hb, he]
    exact root_seg hf hc hcp hj'
  have hfl : (Hash.RowSeg.segmentMsgs (colV V H z ROW_AT (16 * H))).flatMap (fun m => (Verity.sha512 m).bytes) =
      (List.range (segs H)).flatMap fun j => ReadUnit.bytesOf (outV V H z j 512) := by
    rw [Hash.RowSeg.segmentMsgs, length_colV, List.flatMap_def, List.flatMap_def, List.map_map]
    congr 1
    exact List.map_congr_left fun j hj => hseg j hj
  rw [← Hash.BindingProofs.bytes_inj, digestOf_bytes (length_outV _ _), ← root_outer hcp, Hash.RowSeg.digest,
    Hash.RowSeg.outerMsg, length_colV, hfl]

/-- **Part B at the root**: the leaf's copy holds the hm96 leaf of the row's digest and the salt. -/
theorem root_leaf (hcp : Copies V H z) :
    ReadUnit.digestOf ((outV V H z (segs H + 1) 1536).drop 1024) =
      Hash.Hm96.leaf Verity.sha512 pinnedKey (ReadUnit.digestOf (outV V H z (segs H) 512))
        (ReadUnit.bytesOf (colV V H z (saltAt H) 1536)) := by
  have hi : segs H + 1 < nCalls V H := by unfold nCalls; omega
  have hp := leaf_part V H
  have hP := hcp _ hi _ hp
  have hcs := inputs_leaf V H
  have hlen : (inputs V H (lay V H) (segs H + 1)).length = leafPart.ins := by
    rw [hcs]
    simp [leafPart]
  obtain ⟨o, ho, hspec⟩ := copy_run hP (sound_flat (leaf_spec (wordsAt 0 8) (wordsAt 512 24)))
  obtain ⟨o1, o2⟩ := hspec (shaped_wordsAt 0 8) (shaped_wordsAt 512 24)
  have hx : evs (copyV V H z (segs H + 1) leafPart) (wordsAt 0 8).flatten = outV V H z (segs H) 512 := by
    rw [evs_copy_words (k := 0) (m := 8) hlen (P := []) (Q := (List.range 512).map ((lay V H).outAt (segs H)))
      (R := (List.range 1536).map (saltAt H + ·)) (by simp only [hcs, List.nil_append]) rfl (by simp), map_outs]
  have hy : evs (copyV V H z (segs H + 1) leafPart) (wordsAt 512 24).flatten = colV V H z (saltAt H) 1536 := by
    rw [evs_copy_words (k := 512) (m := 24) hlen (P := (List.range 512).map ((lay V H).outAt (segs H)))
      (Q := (List.range 1536).map (saltAt H + ·)) (R := []) (by simp only [hcs, List.append_nil]) (by simp) (by simp),
      map_at]
  have hout := copy_words_out hi hp hP ho o1 (by simp [nOut_leaf])
  rw [wordsDigest, wordsDigest, wordsBytes, hx, hy, flatten_drop (j := 16) o1 (by norm_num), evs_drop, hout] at o2
  exact o2

theorem indexBytes_le {b : ℕ} (hb : b ≤ 32) : Frame.indexBytes b ≤ 4 := by
  unfold Frame.indexBytes
  omega

theorem le_indexBytes (b : ℕ) : b ≤ 8 * Frame.indexBytes b := by
  unfold Frame.indexBytes
  omega

theorem utf8_seg : Hash.utf8 Flock.HmRow.ROW_SCHEMA_SEG = Hash.utf8 Hash.RowLeaf.Format.seg.schema := rfl

/-- **The frame's leaf hash at the root**: its copy holds the frame leaf of the hm96 leaf at the token's index, in the
dense domain under the domain id. -/
theorem root_leafHash (hf : rootVal V H z (lay V H).zero = false) (hcp : Copies V H z) (hD : depth V ≤ 32) :
    ReadUnit.digestOf (outV V H z (segs H + 2) 512) = Verity.sha512 (Hash.Tree.leafMsg
      (denseDomain (depth V) (ReadUnit.digestOf (colV V H z DOMAIN_AT 512))) (ReadUnit.num (colV V H z 0 (depth V)))
      Hash.RowLeaf.Format.seg.schema (ReadUnit.digestOf ((outV V H z (segs H + 1) 1536).drop 1024)).bytes) := by
  have hn1 : 1 ≤ Frame.indexBytes (depth V) := le_max_left _ _
  have hn4 := indexBytes_le hD
  have hnD := le_indexBytes (depth V)
  have hi : segs H + 2 < nCalls V H := by unfold nCalls; omega
  have hp := leafHash_part V H
  have hP := hcp _ hi _ hp
  have hcs := inputs_leafHash V H
  have hlen : (inputs V H (lay V H) (segs H + 2)).length = (leafHashPart (Frame.indexBytes (depth V))).ins := by
    rw [hcs]
    simp [leafHashPart]
    omega
  obtain ⟨o, ho, hspec⟩ := copy_run hP (sound_flat (FrameRead.sound_leafHash Flock.HmRow.ROW_SCHEMA_SEG
    (Frame.indexBytes (depth V)) (wordsAt 0 8) (wordsAt 512 8) (bitsAt 1024 (8 * Frame.indexBytes (depth V)))))
  obtain ⟨o1, o2⟩ := hspec hn1 (length_bitsAt _ _) (by rw [utf8_seg, FrameRead.length_utf8_seg]; omega)
    (shaped_wordsAt 0 8) (shaped_wordsAt 512 8)
  have hd : evs (copyV V H z (segs H + 2) (leafHashPart (Frame.indexBytes (depth V)))) (wordsAt 0 8).flatten =
      colV V H z DOMAIN_AT 512 := by
    rw [evs_copy_words (k := 0) (m := 8) hlen (P := []) (Q := (List.range 512).map (DOMAIN_AT + ·))
      (R := (List.range 512).map (fun k => (lay V H).outAt (segs H + 1) (1024 + k)) ++
        ((List.range (depth V)).map (0 + ·) ++
          List.replicate (8 * Frame.indexBytes (depth V) - depth V) (lay V H).zero))
      (by simp only [hcs, List.append_assoc, List.nil_append]) rfl (by simp), map_at]
  have hv : evs (copyV V H z (segs H + 2) (leafHashPart (Frame.indexBytes (depth V)))) (wordsAt 512 8).flatten =
      (outV V H z (segs H + 1) 1536).drop 1024 := by
    rw [evs_copy_words (k := 512) (m := 8) hlen (P := (List.range 512).map (DOMAIN_AT + ·))
      (Q := (List.range 512).map fun k => (lay V H).outAt (segs H + 1) (1024 + k))
      (R := (List.range (depth V)).map (0 + ·) ++
        List.replicate (8 * Frame.indexBytes (depth V) - depth V) (lay V H).zero)
      (by simp only [hcs, List.append_assoc]) (by simp) (by simp),
      show (1536 : ℕ) = 1024 + 512 from rfl, drop_outV]
    simp [Function.comp_def]
  have hx : evs (copyV V H z (segs H + 2) (leafHashPart (Frame.indexBytes (depth V))))
      (bitsAt 1024 (8 * Frame.indexBytes (depth V))) =
      colV V H z 0 (depth V) ++ List.replicate (8 * Frame.indexBytes (depth V) - depth V) false := by
    rw [evs_copy hlen (k := 1024) (n := 8 * Frame.indexBytes (depth V)) (P := (List.range 512).map (DOMAIN_AT + ·) ++
        (List.range 512).map fun k => (lay V H).outAt (segs H + 1) (1024 + k))
      (Q := (List.range (depth V)).map (0 + ·) ++
        List.replicate (8 * Frame.indexBytes (depth V) - depth V) (lay V H).zero) (R := [])
      (by simp only [hcs, List.append_assoc, List.append_nil]) (by simp) (by simp; omega),
      List.map_append, map_at, List.map_replicate, hf]
  have hout := copy_words_out hi hp hP ho o1 (nOut_mid V H (by omega) (by omega) (by omega))
  simp only [wordsDigest] at o2
  rw [hd, hv, hx, num_append_false, hout] at o2
  exact o2

theorem eq_part : eqPart ∈ (parts V H)[partOf V H (segs H + 3 + depth V)]? := by
  have h1 : ¬segs H + 3 + depth V + 1 < segs H := by omega
  have h2 : ¬segs H + 3 + depth V < segs H := by omega
  have h3 : segs H + 3 + depth V ≠ segs H := by omega
  have h4 : segs H + 3 + depth V ≠ segs H + 1 := by omega
  have h5 : segs H + 3 + depth V ≠ segs H + 2 := by omega
  have h6 : ¬segs H + 3 + depth V < segs H + 3 + depth V := by omega
  simp only [partOf, h1, h2, h3, h4, h5, h6, ↓reduceIte]
  rfl

theorem select_part : selectPart V H ∈ (parts V H)[partOf V H (segs H + 4 + depth V)]? := by
  have h1 : ¬segs H + 4 + depth V + 1 < segs H := by omega
  have h2 : ¬segs H + 4 + depth V < segs H := by omega
  have h3 : segs H + 4 + depth V ≠ segs H := by omega
  have h4 : segs H + 4 + depth V ≠ segs H + 1 := by omega
  have h5 : segs H + 4 + depth V ≠ segs H + 2 := by omega
  have h6 : ¬segs H + 4 + depth V < segs H + 3 + depth V := by omega
  have h7 : segs H + 4 + depth V ≠ segs H + 3 + depth V := by omega
  simp only [partOf, h1, h2, h3, h4, h5, h6, h7, ↓reduceIte]
  rfl

/-! ## Up the path -/

/-- The digest after level `j` of the climb: the frame leaf's (`j = 0`), then each node's. -/
def curV (V H : ℕ) (z : ℕ → Bool) (j : ℕ) : Verity.Digest := ReadUnit.digestOf (outV V H z (segs H + 2 + j) 512)

/-- The path's `j`-th sibling. -/
def sibV (V H : ℕ) (z : ℕ → Bool) (j : ℕ) : Verity.Digest := ReadUnit.digestOf (colV V H z (pathAt H + 512 * j) 512)

/-- The dense domain of `2 ^ D` ranks under the domain id the root's input holds. -/
def domV (V H : ℕ) (z : ℕ → Bool) : Hash.Tree.Domain :=
  denseDomain (depth V) (ReadUnit.digestOf (colV V H z DOMAIN_AT 512))

/-- The index the climb reads: the token's low `D` bits. -/
def posV (V H : ℕ) (z : ℕ → Bool) : ℕ := ReadUnit.num (colV V H z 0 (depth V))

/-- **Node `k` at the root**: its copy holds the frame node at level `k` over the digest below and the path's `k`-th
sibling, swapped by the index's bit `k`. -/
theorem root_node (hf : rootVal V H z (lay V H).zero = false) (hc : rootVal V H z ((lay V H).const + 1) = true)
    (hcp : Copies V H z) (hD : depth V ≤ 32) {k : ℕ} (hk : k < depth V) :
    curV V H z (k + 1) = Verity.sha512 (if posV V H z / 2 ^ k % 2 = 0 then
        Hash.Tree.nodeMsg (domV V H z) k (posV V H z / 2 ^ k / 2) (curV V H z k) (sibV V H z k)
      else Hash.Tree.nodeMsg (domV V H z) k (posV V H z / 2 ^ k / 2) (sibV V H z k) (curV V H z k)) := by
  have hi : segs H + 3 + k < nCalls V H := by unfold nCalls; omega
  have hp := node_part V H hk hD
  have hP := hcp _ hi _ hp
  have hcs := inputs_node V H hk
  have hbn := le_indexBytes (depth V - 1 - k)
  have hn1 : 1 ≤ Frame.indexBytes (depth V - 1 - k) := le_max_left _ _
  have hn4 := indexBytes_le (show depth V - 1 - k ≤ 32 by omega)
  generalize hb : depth V - 1 - k = b at hp hP hcs hbn hn1 hn4
  generalize hn : Frame.indexBytes b = n at hp hP hcs hbn hn1 hn4
  have hlen : (inputs V H (lay V H) (segs H + 3 + k)).length = (nodePart n).ins := by
    rw [hcs]
    simp only [List.length_append, List.length_map, List.length_range, List.length_replicate, length_constRefs,
      List.length_singleton, List.length_cons, List.length_nil, nodePart]
    omega
  obtain ⟨o, ho, hspec⟩ := copy_run hP (sound_flat (FrameRead.sound_nodeHash n (wordsAt 0 8) (bitsAt 512 8)
    (bitsAt 520 (8 * n)) (Lin.var (520 + 8 * n)) (wordsAt (521 + 8 * n) 8) (wordsAt (1033 + 8 * n) 8)))
  obtain ⟨o1, o2⟩ := hspec hn1 (length_bitsAt _ _) (length_bitsAt _ _) (by omega) (shaped_wordsAt 0 8)
    (shaped_wordsAt _ 8) (shaped_wordsAt _ 8)
  have hd : evs (copyV V H z (segs H + 3 + k) (nodePart n)) (wordsAt 0 8).flatten = colV V H z DOMAIN_AT 512 := by
    rw [evs_copy_words (k := 0) (m := 8) hlen (P := []) (Q := (List.range 512).map (DOMAIN_AT + ·))
      (R := constRefs (lay V H) k 8 ++ ((List.range b).map (k + 1 + ·) ++
        (List.replicate (8 * n - b) (lay V H).zero ++ ([k] ++ ((List.range 512).map ((lay V H).outAt (segs H + 2 + k)) ++
          (List.range 512).map (pathAt H + 512 * k + ·))))))
      (by simp only [hcs, List.append_assoc, List.nil_append]) rfl (by simp), map_at]
  have hlv : evs (copyV V H z (segs H + 3 + k) (nodePart n)) (bitsAt 512 8) = (List.range 8).map k.testBit := by
    rw [evs_copy hlen (k := 512) (n := 8) (P := (List.range 512).map (DOMAIN_AT + ·)) (Q := constRefs (lay V H) k 8)
      (R := (List.range b).map (k + 1 + ·) ++
        (List.replicate (8 * n - b) (lay V H).zero ++ ([k] ++ ((List.range 512).map ((lay V H).outAt (segs H + 2 + k)) ++
          (List.range 512).map (pathAt H + 512 * k + ·)))))
      (by simp only [hcs, List.append_assoc]) (by simp) (length_constRefs _ _ _), map_constRefs _ hf hc]
  have hix : evs (copyV V H z (segs H + 3 + k) (nodePart n)) (bitsAt 520 (8 * n)) =
      colV V H z (k + 1) b ++ List.replicate (8 * n - b) false := by
    rw [evs_copy hlen (k := 520) (n := 8 * n) (P := (List.range 512).map (DOMAIN_AT + ·) ++ constRefs (lay V H) k 8)
      (Q := (List.range b).map (k + 1 + ·) ++ List.replicate (8 * n - b) (lay V H).zero)
      (R := [k] ++ ((List.range 512).map ((lay V H).outAt (segs H + 2 + k)) ++
          (List.range 512).map (pathAt H + 512 * k + ·)))
      (by simp only [hcs, List.append_assoc]) (by simp [length_constRefs])
      (by simp only [List.length_append, List.length_map, List.length_range, List.length_replicate]; omega),
      List.map_append, map_at, List.map_replicate, hf]
  have hdv : ev (copyV V H z (segs H + 3 + k) (nodePart n)) (Lin.var (520 + 8 * n)) = rootVal V H z k := by
    have h : evs (copyV V H z (segs H + 3 + k) (nodePart n)) (bitsAt (520 + 8 * n) 1) = [k].map (rootVal V H z) :=
      evs_copy hlen (P := (List.range 512).map (DOMAIN_AT + ·) ++ constRefs (lay V H) k 8 ++
        (List.range b).map (k + 1 + ·) ++ List.replicate (8 * n - b) (lay V H).zero) (Q := [k])
      (R := (List.range 512).map ((lay V H).outAt (segs H + 2 + k)) ++ (List.range 512).map (pathAt H + 512 * k + ·))
      (by simp only [hcs, List.append_assoc]) (k := 520 + 8 * n) (n := 1)
      (by simp only [List.length_append, List.length_map, List.length_range, List.length_replicate,
        length_constRefs]; omega) rfl
    simpa [evs, bitsAt] using h
  have hcur : evs (copyV V H z (segs H + 3 + k) (nodePart n)) (wordsAt (521 + 8 * n) 8).flatten =
      outV V H z (segs H + 2 + k) 512 := by
    rw [evs_copy_words (k := 521 + 8 * n) (m := 8) hlen (P := (List.range 512).map (DOMAIN_AT + ·) ++
        constRefs (lay V H) k 8 ++ (List.range b).map (k + 1 + ·) ++ List.replicate (8 * n - b) (lay V H).zero ++ [k])
      (Q := (List.range 512).map ((lay V H).outAt (segs H + 2 + k))) (R := (List.range 512).map (pathAt H + 512 * k + ·))
      (by simp only [hcs, List.append_assoc])
      (by simp only [List.length_append, List.length_map, List.length_range, List.length_replicate,
        length_constRefs, List.length_singleton]; omega) (by simp), map_outs]
  have hsib : evs (copyV V H z (segs H + 3 + k) (nodePart n)) (wordsAt (1033 + 8 * n) 8).flatten =
      colV V H z (pathAt H + 512 * k) 512 := by
    rw [evs_copy_words (k := 1033 + 8 * n) (m := 8) hlen (P := (List.range 512).map (DOMAIN_AT + ·) ++
        constRefs (lay V H) k 8 ++ (List.range b).map (k + 1 + ·) ++ List.replicate (8 * n - b) (lay V H).zero ++ [k] ++
        (List.range 512).map ((lay V H).outAt (segs H + 2 + k)))
      (Q := (List.range 512).map (pathAt H + 512 * k + ·)) (R := [])
      (by simp only [hcs, List.append_nil])
      (by simp only [List.length_append, List.length_map, List.length_range, List.length_replicate,
        length_constRefs, List.length_singleton]; omega) (by simp), map_at]
  have hout := copy_words_out hi hp hP ho o1 (nOut_mid V H (by omega) (by omega) (by omega))
  have hidx : ReadUnit.num (colV V H z (k + 1) b) = posV V H z / 2 ^ k / 2 := by
    have h := drop_colV (V := V) (H := H) (z := z) 0 (k + 1) b
    rw [show k + 1 + b = depth V by omega, Nat.zero_add] at h
    rw [← h, FrameRead.num_drop, posV, Nat.div_div_eq_div_mul, ← Nat.pow_succ]
  have hbit : rootVal V H z k = decide (posV V H z / 2 ^ k % 2 = 1) := by
    rw [← Nat.testBit_eq_decide_div_mod_eq, posV, FrameRead.testBit_num]
    simp [colV, hk]
  have hlvu : Hash.uint k = [UInt8.ofNat k] := FrameRead.uint_small (by omega)
  simp only [wordsDigest, FrameRead.byteV] at o2
  rw [hd, hlv, hix, hdv, hcur, hsib, hout, num_append_false, FrameRead.num_map_testBit,
    Nat.mod_eq_of_lt (show k < 2 ^ 8 by norm_num; omega), ← hlvu, hidx, hbit] at o2
  have e : segs H + 2 + (k + 1) = segs H + 3 + k := by omega
  simp only [curV, e]
  refine o2.trans ?_
  by_cases h2 : posV V H z / 2 ^ k % 2 = 0
  · rw [if_pos h2, decide_eq_false (show ¬(posV V H z / 2 ^ k % 2 = 1) by omega)]
    simp only [Bool.false_eq_true, ↓reduceIte]
    rfl
  · rw [if_neg h2, decide_eq_true (show posV V H z / 2 ^ k % 2 = 1 by omega)]
    simp only [↓reduceIte]
    rfl

/-- **The climb at the root**: from level `k`, the frame's climb over the path's siblings from `k` reaches the last
node's digest. -/
theorem root_climb (hf : rootVal V H z (lay V H).zero = false) (hc : rootVal V H z ((lay V H).const + 1) = true)
    (hcp : Copies V H z) (hD : depth V ≤ 32) : ∀ m k, k + m = depth V →
      (Hash.Tree.climb Verity.sha512 (domV V H z) k (posV V H z / 2 ^ k) (curV V H z k)
        ((List.range m).map fun t => sibV V H z (k + t))).2 = curV V H z (depth V)
  | 0, k, h => by
    have hk : k = depth V := by omega
    subst hk
    rfl
  | m + 1, k, h => by
    have hk : k < depth V := by omega
    have hl : ((List.range (m + 1)).map fun t => sibV V H z (k + t)) =
        sibV V H z k :: (List.range m).map fun t => sibV V H z (k + 1 + t) := by
      simp only [List.range_succ_eq_map, List.map_cons, List.map_map, Nat.add_zero, List.cons.injEq, true_and]
      refine List.map_congr_left fun t _ => ?_
      simp only [Function.comp_apply]
      exact congrArg (sibV V H z) (by omega)
    rw [hl, FrameRead.climb_cons_snd, ← root_node hf hc hcp hD hk,
      show posV V H z / 2 ^ k / 2 = posV V H z / 2 ^ (k + 1) by rw [Nat.div_div_eq_div_mul, ← Nat.pow_succ]]
    exact root_climb hf hc hcp hD m (k + 1) (by omega)

/-! ## The comparison and the select -/

theorem sound_eqPart : Sound eqPart.prog fun z out => ∃ b, out = [b] ∧
    (ev z b = true → wordsDigest z (wordsAt 0 8) = wordsDigest z (wordsAt 512 8)) :=
  Sound.bind' (FrameRead.sound_eqDigest (wordsAt 0 8) (wordsAt 512 8)) fun b =>
    Sound.pure' fun _ h => ⟨b, rfl, h (shaped_wordsAt 0 8) (shaped_wordsAt 512 8)⟩

/-- **The comparison at the root**: where its bit is set, the last node's digest is the root's input. -/
theorem root_eq (hcp : Copies V H z) (hok : rootVal V H z ((lay V H).outAt (segs H + 3 + depth V) 0) = true) :
    curV V H z (depth V) = ReadUnit.digestOf (colV V H z ROOT_AT 512) := by
  have hi : segs H + 3 + depth V < nCalls V H := by unfold nCalls; omega
  have hp := eq_part (V := V) (H := H)
  have hP := hcp _ hi _ hp
  have hcs := inputs_eq (V := V) (H := H)
  have hlen : (inputs V H (lay V H) (segs H + 3 + depth V)).length = eqPart.ins := by
    rw [hcs]
    simp [eqPart]
  obtain ⟨b, hb, himp⟩ := copy_run hP sound_eqPart
  have h := copy_outs hi hp hP (by rw [hb, nOut_eq, List.length_singleton])
  rw [hb, nOut_eq] at h
  simp only [evs, outV, List.range_one, List.map_cons, List.map_nil, List.cons.injEq, and_true] at h
  have hx : evs (copyV V H z (segs H + 3 + depth V) eqPart) (wordsAt 0 8).flatten =
      outV V H z (segs H + 2 + depth V) 512 := by
    rw [evs_copy_words (k := 0) (m := 8) hlen (P := [])
      (Q := (List.range 512).map ((lay V H).outAt (segs H + 2 + depth V))) (R := (List.range 512).map (ROOT_AT + ·))
      (by simp only [hcs, List.nil_append]) rfl (by simp), map_outs]
  have hy : evs (copyV V H z (segs H + 3 + depth V) eqPart) (wordsAt 512 8).flatten = colV V H z ROOT_AT 512 := by
    rw [evs_copy_words (k := 512) (m := 8) hlen (P := (List.range 512).map ((lay V H).outAt (segs H + 2 + depth V)))
      (Q := (List.range 512).map (ROOT_AT + ·)) (R := []) (by simp only [hcs, List.append_nil]) (by simp) (by simp),
      map_at]
  have he := himp (h.trans hok)
  simp only [wordsDigest] at he
  rw [hx, hy] at he
  exact he

theorem sound_selectPart (hV : V < 2 ^ 32) : Sound (selectPart V H).prog fun z out => ∃ r : List Lin × Lin,
    out = r.1 ++ [r.2] ∧ evs z r.1 = (if ReadUnit.num (evs z (bitsAt 0 32)) < V then evs z (bitsAt 33 (16 * H))
      else Hash.EmbeddingRead.fill H) ∧
      ev z r.2 = (decide (V ≤ ReadUnit.num (evs z (bitsAt 0 32))) || ev z (Lin.var 32)) :=
  Sound.bind' (select_spec hV _ _ _) fun r => Sound.pure' fun z h => by
    obtain ⟨h1, h2⟩ := h (length_bitsAt _ _) (by rw [length_bitsAt]; exact Nat.mul_mod_right 16 H)
    refine ⟨r, rfl, ?_, h2⟩
    rw [h1, length_bitsAt, Nat.mul_div_cancel_left H (by norm_num : 0 < 16)]

theorem outV_succ (i n : ℕ) :
    outV V H z i (n + 1) = outV V H z i n ++ [rootVal V H z ((lay V H).outAt i n)] := by
  simp [outV, List.range_succ]

/-- **The select at the root**: its row is the row input inside the vocabulary and the fill row outside, and its
verdict is the comparison's bit inside and set outside. -/
theorem root_select (hV : V < 2 ^ 32) (hcp : Copies V H z) :
    outV V H z (segs H + 4 + depth V) (16 * H) =
        (if ReadUnit.num (colV V H z 0 32) < V then colV V H z ROW_AT (16 * H) else Hash.EmbeddingRead.fill H) ∧
      rootVal V H z ((lay V H).outAt (segs H + 4 + depth V) (16 * H)) =
        (decide (V ≤ ReadUnit.num (colV V H z 0 32)) ||
          rootVal V H z ((lay V H).outAt (segs H + 3 + depth V) 0)) := by
  have hi : segs H + 4 + depth V < nCalls V H := by unfold nCalls; omega
  have hp := select_part (V := V) (H := H)
  have hP := hcp _ hi _ hp
  have hcs := inputs_select (V := V) (H := H)
  have hlen : (inputs V H (lay V H) (segs H + 4 + depth V)).length = (selectPart V H).ins := by
    rw [hcs]
    simp only [List.length_append, List.length_map, List.length_range, List.length_singleton, selectPart]
  obtain ⟨r, hr, h1, h2⟩ := copy_run hP (sound_selectPart (H := H) hV)
  have hrl : r.1.length = 16 * H := by
    have := congrArg List.length h1
    rw [length_evs] at this
    rw [this]
    split
    · exact length_evs _ _ |>.trans (length_bitsAt _ _)
    · exact length_fill H
  have ho := copy_outs hi hp hP (by rw [hr, List.length_append, hrl, List.length_singleton, nOut_select])
  have hsplit : evs (copyV V H z (segs H + 4 + depth V) (selectPart V H)) r.1 ++
      [ev (copyV V H z (segs H + 4 + depth V) (selectPart V H)) r.2] =
      outV V H z (segs H + 4 + depth V) (16 * H) ++
        [rootVal V H z ((lay V H).outAt (segs H + 4 + depth V) (16 * H))] := by
    rw [← outV_succ, ← nOut_select, ← ho, hr]
    simp [evs]
  obtain ⟨ha, hb⟩ := List.append_inj hsplit (by rw [length_evs, hrl, length_outV])
  have hb' := List.singleton_inj.1 hb
  have ht : evs (copyV V H z (segs H + 4 + depth V) (selectPart V H)) (bitsAt 0 32) = colV V H z 0 32 := by
    rw [evs_copy hlen (k := 0) (n := 32) (P := []) (Q := (List.range 32).map (0 + ·))
      (R := [(lay V H).outAt (segs H + 3 + depth V) 0] ++ (List.range (16 * H)).map (ROW_AT + ·))
      (by simp only [hcs, List.append_assoc, List.nil_append]) rfl (by simp), map_at]
  have hop : ev (copyV V H z (segs H + 4 + depth V) (selectPart V H)) (Lin.var 32) =
      rootVal V H z ((lay V H).outAt (segs H + 3 + depth V) 0) := by
    have h : evs (copyV V H z (segs H + 4 + depth V) (selectPart V H)) (bitsAt 32 1) =
        [(lay V H).outAt (segs H + 3 + depth V) 0].map (rootVal V H z) :=
      evs_copy hlen (P := (List.range 32).map (0 + ·)) (Q := [(lay V H).outAt (segs H + 3 + depth V) 0])
      (R := (List.range (16 * H)).map (ROW_AT + ·)) (by simp only [hcs, List.append_assoc]) (k := 32) (n := 1)
      (by simp) rfl
    simpa [evs, bitsAt] using h
  have hrow : evs (copyV V H z (segs H + 4 + depth V) (selectPart V H)) (bitsAt 33 (16 * H)) =
      colV V H z ROW_AT (16 * H) := by
    rw [evs_copy hlen (k := 33) (n := 16 * H) (P := (List.range 32).map (0 + ·) ++
        [(lay V H).outAt (segs H + 3 + depth V) 0])
      (Q := (List.range (16 * H)).map (ROW_AT + ·)) (R := []) (by simp only [hcs, List.append_nil]) (by simp) (by simp),
      map_at]
  rw [ht, hrow] at h1
  rw [ht, hop] at h2
  exact ⟨ha.symm.trans h1, hb'.symm.trans h2⟩

/-! ## The read -/

theorem depth_le (hV : V < 2 ^ 32) : depth V ≤ 32 := by
  unfold depth
  split
  · omega
  · have := (Nat.log2_lt (n := V - 1) (k := 32) (by omega)).2 (by omega)
    omega

theorem le_two_pow_depth (V : ℕ) : V ≤ 2 ^ depth V := by
  unfold depth
  split
  · simp only [pow_zero]; omega
  · have := Nat.lt_log2_self (n := V - 1)
    omega

theorem take_colV {k n m : ℕ} (h : m ≤ n) : (colV V H z k n).take m = colV V H z k m := by
  simp only [colV]
  rw [← List.map_take, List.take_range, Nat.min_eq_left h]

/-- The opening the root's inputs hold: the row, the salt's bytes and the path's siblings. -/
def openV (V H : ℕ) (z : ℕ → Bool) : Hash.RowLeaf.Opening :=
  ⟨colV V H z ROW_AT (16 * H), ReadUnit.bytesOf (colV V H z (saltAt H) 1536), (List.range (depth V)).map (sibV V H z)⟩

/-- **The read at the root**: the select's row is the read's at the token, root, domain id and opening the root's
inputs hold, and where its verdict is set the read accepts them at SHA-512 under the pinned hm96 key. -/
theorem root_read (hV : V < 2 ^ 32) (hH : 16 * H < 2 ^ 64) (hf : rootVal V H z (lay V H).zero = false)
    (hc : rootVal V H z ((lay V H).const + 1) = true) (hcp : Copies V H z) :
    outV V H z (segs H + 4 + depth V) (16 * H) =
        Hash.EmbeddingRead.h V H (ReadUnit.num (colV V H z 0 32)) (openV V H z) ∧
      (rootVal V H z ((lay V H).outAt (segs H + 4 + depth V) (16 * H)) = true →
        Hash.EmbeddingRead.ok Verity.sha512 Verity.sha512 ⟨ReadUnit.digestOf (colV V H z DOMAIN_AT 512), V, fun r => r⟩
          pinnedKey (ReadUnit.digestOf (colV V H z ROOT_AT 512)) V (ReadUnit.num (colV V H z 0 32)) (openV V H z)) := by
  have hD := depth_le hV
  have hVD := le_two_pow_depth V
  obtain ⟨hsel, hok⟩ := root_select hV hcp
  refine ⟨hsel.trans rfl, fun hout => ?_⟩
  rw [hok] at hout
  by_cases hlt : ReadUnit.num (colV V H z 0 32) < V
  · right
    have hop : rootVal V H z ((lay V H).outAt (segs H + 3 + depth V) 0) = true := by
      simpa [Nat.not_le.mpr hlt] using hout
    have heq := root_eq hcp hop
    have hpos : posV V H z = ReadUnit.num (colV V H z 0 32) := by
      rw [posV, ← take_colV hD]
      exact num_take_of_lt _ _ (lt_of_lt_of_le hlt hVD)
    have hcur0 : curV V H z 0 = Verity.sha512 (Hash.Tree.leafMsg (domV V H z) (ReadUnit.num (colV V H z 0 32))
        Hash.RowLeaf.Format.seg.schema (Hash.Hm96.leaf Verity.sha512 pinnedKey
          (Hash.RowSeg.digest Verity.sha512 (colV V H z ROW_AT (16 * H)))
          (ReadUnit.bytesOf (colV V H z (saltAt H) 1536))).bytes) := by
      rw [← hpos, root_rowSeg hf hc hcp, ← root_leaf hcp]
      exact root_leafHash hf hcp hD
    refine ⟨?_, ?_, hlt, ?_, ?_⟩
    · show (colV V H z ROW_AT (16 * H)).length < 2 ^ 64
      rw [length_colV]
      exact hH
    · show (ReadUnit.bytesOf (colV V H z (saltAt H) 1536)).length = Hash.Hm96.SALT_BYTES
      simp only [length_readBytes, length_colV]
      rfl
    · show ((List.range (depth V)).map (sibV V H z)).length = Hash.Tree.depth V
      rw [List.length_map, List.length_range, depth_eq]
    · show (Hash.Tree.climb Verity.sha512 ⟨ReadUnit.digestOf (colV V H z DOMAIN_AT 512), V, fun r => r⟩ 0
          (ReadUnit.num (colV V H z 0 32))
          (Verity.sha512 (Hash.Tree.leafMsg (domV V H z) (ReadUnit.num (colV V H z 0 32))
            Hash.RowLeaf.Format.seg.schema (Hash.Hm96.leaf Verity.sha512 pinnedKey
              (Hash.RowSeg.digest Verity.sha512 (colV V H z ROW_AT (16 * H)))
              (ReadUnit.bytesOf (colV V H z (saltAt H) 1536))).bytes))
          ((List.range (depth V)).map (sibV V H z))).2 = ReadUnit.digestOf (colV V H z ROOT_AT 512)
      have hpath : (List.range (depth V)).map (sibV V H z) = (List.range (depth V)).map fun t => sibV V H z (0 + t) := by
        simp
      rw [climb_count Verity.sha512 _ V (2 ^ depth V) (fun r => r), hpath, ← hcur0, ← hpos]
      have hcl := root_climb hf hc hcp hD (depth V) 0 (by omega)
      rw [Nat.pow_zero, Nat.div_one] at hcl
      exact hcl.trans heq
  · left
    omega

end Accounting.ReadUnit.Gates
