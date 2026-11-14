module

public import Proofs.Accounting.ReadUnit.Check.Copies

/-!
# The read's root composes its calls into the read

At the root's references' values `g`, each call's copy holds its part's run on the values its inputs bind (`root_facts`).
Each part's statement then reads as a fact about lists of bit values: a copy's input forms hold the values of the root
columns its call binds (`evs_copy`), and its returned forms hold the root's values at the call's output columns
(`copy_outs`). The facts chain the parts in order: the segments and the outer hash give the row's
`sha512/row-seg/v1` digest (`root_rowSeg`), then the hm96 leaf, the frame's leaf hash, its nodes up the path and the
comparison with the root (`root_climb`), then the select (`root_read`).
-/

@[expose] public section

namespace Accounting.ReadUnit.Gates

open FlockSoundness FlockSoundness.TmplSpec FlockSoundness.Discharge.Hm Flock.ReadNet
open Flock.HmNets hiding and

/-! ## Bit-value lists -/

theorem bytesOf_append (B : List Bool) : ∀ (n : ℕ) {A : List Bool}, A.length = 64 * n →
    ReadUnit.bytesOf (A ++ B) = ReadUnit.bytesOf A ++ ReadUnit.bytesOf B
  | 0, A, hA => by
    obtain rfl : A = [] := List.eq_nil_of_length_eq_zero (by omega)
    simp [ReadUnit.bytesOf]
  | n + 1, A, hA => by
    have h64 : (A.take 64).length = 64 := by rw [List.length_take]; omega
    have hd : (A.drop 64).length = 64 * n := by rw [List.length_drop]; omega
    rw [← List.take_append_drop 64 A, List.append_assoc, bytesOf_append64 _ h64, bytesOf_append64 _ h64,
      bytesOf_append B n hd, List.append_assoc]

theorem bytesOf_flatMap {α : Type} (f : α → List Bool) (hf : ∀ a, (f a).length = 512) : ∀ l : List α,
    ReadUnit.bytesOf (l.flatMap f) = l.flatMap fun a => ReadUnit.bytesOf (f a)
  | [] => by simp [ReadUnit.bytesOf]
  | a :: l => by
    rw [List.flatMap_cons, List.flatMap_cons, bytesOf_append _ 8 (by rw [hf]), bytesOf_flatMap f hf l]

theorem num_replicate_false : ∀ m, ReadUnit.num (List.replicate m false) = 0
  | 0 => rfl
  | m + 1 => by simp only [List.replicate_succ, ReadUnit.num, num_replicate_false m]; rfl

theorem num_append_false (A : List Bool) (m : ℕ) : ReadUnit.num (A ++ List.replicate m false) = ReadUnit.num A := by
  rw [num_append, num_replicate_false, Nat.mul_zero, Nat.add_zero]

/-! ## A part's forms -/

theorem bitsAt_add (k a b : ℕ) : bitsAt k (a + b) = bitsAt k a ++ bitsAt (k + a) b := by
  simp only [bitsAt, List.range_add, List.map_append, List.map_map]
  congr 1
  refine List.map_congr_left fun i _ => ?_
  simp [Nat.add_assoc]

theorem wordsAt_succ (k m : ℕ) : wordsAt k (m + 1) = wordsAt k m ++ [bitsAt (k + 64 * m) 64] := by
  simp [wordsAt, List.range_succ]

theorem flatten_wordsAt (k : ℕ) : ∀ m, (wordsAt k m).flatten = bitsAt k (64 * m)
  | 0 => by simp [wordsAt, bitsAt]
  | m + 1 => by
    rw [wordsAt_succ, List.flatten_append, flatten_wordsAt k m, Nat.mul_succ, bitsAt_add]
    simp

theorem shaped_wordsAt (k m : ℕ) : Shaped m (wordsAt k m) :=
  ⟨by simp [wordsAt], fun w hw => by
    obtain ⟨j, -, rfl⟩ := List.mem_map.1 hw
    simp [bitsAt]⟩

theorem length_bitsAt (k n : ℕ) : (bitsAt k n).length = n := by simp [bitsAt]

theorem evs_bitsAt (f : Var → Bool) (k n : ℕ) : evs f (bitsAt k n) = (List.range n).map fun t => f (k + t) := by
  simp [evs, bitsAt, Function.comp_def]

/-- A digest's bytes are its 512 bits' bytes. -/
theorem digestOf_bytes {bs : List Bool} (h : bs.length = 512) :
    (ReadUnit.digestOf bs).bytes = ReadUnit.bytesOf bs := by
  have hs : Shaped 8 (wordsAt 0 8) := shaped_wordsAt 0 8
  have he : evs (fun x => bs.getD x false) (wordsAt 0 8).flatten = bs := by
    rw [flatten_wordsAt, evs_bitsAt]
    apply List.ext_getElem (by simp [h])
    intro t h1 h2
    simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h2]
  have := (wordsDigest_bytes (fun x => bs.getD x false) hs).trans (wordsBytes_eq _ hs).symm
  rwa [wordsDigest, wordsBytes, he] at this

/-- Two lists of words whose bits hold the same values hold the same words. -/
theorem valN_congr {z z' : Var → Bool} : ∀ {w w' : Word}, evs z w = evs z' w' → valN z w = valN z' w'
  | [], [], _ => rfl
  | f :: w, f' :: w', h => by
    simp only [evs, List.map_cons, List.cons.injEq] at h
    simp only [valN, h.1, valN_congr (w := w) (w' := w') h.2]
  | [], _ :: _, h => by simp [evs] at h
  | _ :: _, [], h => by simp [evs] at h

theorem map_bv_of_flatten {z z' : Var → Bool} : ∀ {ws ws' : List Word}, Words64 ws → Words64 ws' →
    ws.length = ws'.length → evs z ws.flatten = evs z' ws'.flatten → ws.map (bv z) = ws'.map (bv z')
  | [], [], _, _, _, _ => rfl
  | w :: ws, w' :: ws', h1, h2, hl, h => by
    have a := h1 w (by simp)
    have b := h2 w' (by simp)
    simp only [List.flatten_cons, evs, List.map_append] at h
    have hs := List.append_inj h (by simp [a, b])
    simp only [List.map_cons, bv, valN_congr (w := w) (w' := w') hs.1,
      map_bv_of_flatten (fun x hx => h1 x (by simp [hx])) (fun x hx => h2 x (by simp [hx]))
        (by simpa using hl) hs.2]
  | [], _ :: _, _, _, hl, _ => by simp at hl
  | _ :: _, [], _, _, hl, _ => by simp at hl

theorem sound_flat {m : M (List Word)} {P : (Var → Bool) → List Word → Prop} (hm : Sound m P) :
    Sound (do let o ← m; pure o.flatten) fun z out => ∃ o, out = o.flatten ∧ P z o :=
  Sound.bind' hm fun o => Sound.pure' fun _ h => ⟨o, rfl, h⟩

theorem shaped_flatten_length {k : ℕ} {ws : List Word} (h : Shaped k ws) : ws.flatten.length = 64 * k := by
  rw [length_flatten_of_shaped h, Nat.mul_comm]

/-! ## A copy, read at the root -/

section copy

variable {ins : ℕ} {r : List Lin × St} {v : ℕ → Bool} {g : ℕ → Bool} {cs : List ℕ}

theorem partVal_lt (hlen : cs.length = ins) {x : ℕ} (hx : x < ins) :
    partVal ins r v (cs.map g) x = g (cs.getD x 0) := by
  have h1 : ¬((defOf ins r).cols + x < (defOf ins r).cols) := by omega
  have h2 : (defOf ins r).cols + x < (defOf ins r).cols + (defOf ins r).ins := by rw [defOf_ins]; omega
  rw [partVal, colOf, if_pos hx, Flock.Tmpl.Def.refV, if_neg h1, if_pos h2, Nat.add_sub_cancel_left,
    List.getD_eq_getElem?_getD, List.getElem?_map, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (by omega)]
  rfl

/-- **A copy's input forms** at `k` hold the values of the root columns its call binds there. -/
theorem evs_copy (hlen : cs.length = ins) {P Q R : List ℕ} (hcs : cs = P ++ (Q ++ R)) {k n : ℕ}
    (hk : P.length = k) (hn : Q.length = n) : evs (partVal ins r v (cs.map g)) (bitsAt k n) = Q.map g := by
  subst hk hn
  apply List.ext_getElem (by simp [evs, bitsAt])
  intro t h1 _
  have ht : t < Q.length := by simpa [evs, bitsAt] using h1
  have hx : P.length + t < ins := by rw [← hlen, hcs]; simp; omega
  simp only [evs, bitsAt, List.getElem_map, List.getElem_range, ev_var]
  rw [partVal_lt hlen hx, hcs, List.getD_eq_getElem?_getD, List.getElem?_append_right (by omega),
    Nat.add_sub_cancel_left, List.getElem?_append_left ht, List.getElem?_eq_getElem ht]
  rfl

theorem evs_copy_words (hlen : cs.length = ins) {P Q R : List ℕ} (hcs : cs = P ++ (Q ++ R)) {k m : ℕ}
    (hk : P.length = k) (hm : Q.length = 64 * m) :
    evs (partVal ins r v (cs.map g)) (wordsAt k m).flatten = Q.map g := by
  rw [flatten_wordsAt, evs_copy hlen hcs hk hm]

/-- **A copy's returned forms** hold its own columns that commit them. -/
theorem evs_copy_outs (hp : PartHolds ins r v (cs.map g)) :
    evs (partVal ins r v (cs.map g)) r.1 = (List.range r.1.length).map fun j => v (r.2.next - ins + j) := by
  apply List.ext_getElem (by simp [evs])
  intro j h1 _
  simp only [evs, List.getElem_map, List.getElem_range]
  exact (hp.2 j (by simpa [evs] using h1)).symm

end copy

/-! ## The root's calls -/

section root

variable {V H : ℕ} {z : ℕ → Bool}

/-- The values the root's columns hold at call `i`'s first `n` outputs. -/
def outV (V H : ℕ) (z : ℕ → Bool) (i n : ℕ) : List Bool :=
  (List.range n).map fun k => rootVal V H z ((lay V H).outAt i k)

/-- The values `n` root columns from `k` hold. -/
def colV (V H : ℕ) (z : ℕ → Bool) (k n : ℕ) : List Bool := (List.range n).map fun t => rootVal V H z (k + t)

theorem map_at (k n : ℕ) : ((List.range n).map (k + ·)).map (rootVal V H z) = colV V H z k n := by
  simp [colV, Function.comp_def]

theorem map_outs (i n : ℕ) : ((List.range n).map ((lay V H).outAt i)).map (rootVal V H z) = outV V H z i n := by
  simp [outV, Function.comp_def]

theorem length_outV (i n : ℕ) : (outV V H z i n).length = n := by simp [outV]

theorem length_colV (k n : ℕ) : (colV V H z k n).length = n := by simp [colV]

theorem rootVal_le {x : ℕ} (hx : x ≤ (lay V H).const) : rootVal V H z x = z x := by
  simp [Flock.Tmpl.Def.refV, rootDef_cols, show x < (lay V H).const + 1 by omega]

theorem partDefs_getD {i : ℕ} {p : Part} (hp : p ∈ (parts V H)[i]?) :
    (partDefs V H).getD i default = defOf p.ins p.run := by
  rw [Array.getD_eq_getD_getElem?, partDefs, Array.getElem?_map, Option.mem_def.1 hp]
  rfl

theorem base_cols_le {i : ℕ} (hi : i < nCalls V H) :
    (lay V H).base i + ((partDefs V H).getD (partOf V H i) default).cols ≤ (lay V H).const := by
  rw [← lay_base_succ]
  have h1 := lay_base_mono V H (show i + 1 ≤ nCalls V H by omega)
  have h2 := lay_oStart_ge V H
  rw [lay_const]
  omega

/-- The value a copy of call `i`'s part gives its run's variables. -/
abbrev copyV (V H : ℕ) (z : ℕ → Bool) (i : ℕ) (p : Part) : Var → Bool :=
  partVal p.ins p.run (fun y => z ((lay V H).base i + y)) ((inputs V H (lay V H) i).map (rootVal V H z))

/-- What `root_facts` gives call `i`'s copy of part `p`. -/
abbrev CopyHolds (V H : ℕ) (z : ℕ → Bool) (i : ℕ) (p : Part) : Prop :=
  PartHolds p.ins p.run (fun y => z ((lay V H).base i + y)) ((inputs V H (lay V H) i).map (rootVal V H z))

/-- **A copy's returned forms** hold the root's values at its call's output columns. -/
theorem copy_outs {i : ℕ} (hi : i < nCalls V H) {p : Part} (hp : p ∈ (parts V H)[partOf V H i]?)
    (hP : CopyHolds V H z i p) (hn : p.run.1.length = nOut V H i) :
    evs (copyV V H z i p) p.run.1 = outV V H z i (nOut V H i) := by
  rw [evs_copy_outs hP, hn, outV]
  refine List.map_congr_left fun k hk => ?_
  have hk' := List.mem_range.1 hk
  have hc := partDefs_getD hp
  have hle := base_cols_le (V := V) (H := H) hi
  rw [hc, defOf_cols] at hle
  have hout : (lay V H).outAt i k = (lay V H).base i + (p.run.2.next - p.ins + k) := by
    show (lay V H).base i + ((partDefs V H).getD (partOf V H i) default).cols - nOut V H i + k = _
    rw [hc, defOf_cols]
    omega
  rw [hout, rootVal_le (by omega)]

/-- A copy's statement, from its part's generator's. -/
theorem copy_run {i : ℕ} {p : Part} (hP : CopyHolds V H z i p) {P : (Var → Bool) → List Lin → Prop}
    (hm : Sound p.prog P) : P (copyV V H z i p) p.run.1 :=
  (hm ⟨p.ins, []⟩).2 _ hP.1

/-- The words a copy returns hold the root's values at its call's output columns. -/
theorem copy_words_out {i : ℕ} (hi : i < nCalls V H) {p : Part} (hp : p ∈ (parts V H)[partOf V H i]?)
    (hP : CopyHolds V H z i p) {o : List Word} {k : ℕ} (ho : p.run.1 = o.flatten) (hs : Shaped k o)
    (hn : nOut V H i = 64 * k) : evs (copyV V H z i p) o.flatten = outV V H z i (64 * k) := by
  rw [← ho, copy_outs hi hp hP (by rw [ho, shaped_flatten_length hs, hn]), hn]

end root

/-! ## Each call's part, inputs and outputs -/

theorem rsb : Flock.HmRow.ROW_SEG_BITS = 8192 := rfl

/-- Segment `j`'s bits. -/
def segM (H j : ℕ) : ℕ := min (16 * H - Flock.HmRow.ROW_SEG_BITS * j) Flock.HmRow.ROW_SEG_BITS

/-- Segment `j`'s chaining value's columns: the constant where its bit is set, else the zero column. -/
def cvRefs (L : Layout) (H j : ℕ) : List ℕ :=
  (List.range 8).flatMap fun w => (List.range 64).map fun b =>
    if (Flock.Sha512.compress Flock.Sha512.iv (Flock.HmRow.segmentPrefix j (segs H) (segM H j)) 0)[w]!.toBitVec.getLsbD b
    then L.const + 1 else L.zero

theorem length_cvRefs (L : Layout) (H j : ℕ) : (cvRefs L H j).length = 512 := by
  simp [cvRefs, List.length_flatMap, List.map_const', List.sum_replicate, smul_eq_mul]

theorem map_cvRefs (L : Layout) (H j : ℕ) {g : ℕ → Bool} (hf : g L.zero = false) (hc : g (L.const + 1) = true)
    (zc : Var → Bool) :
    (cvRefs L H j).map g = evs zc (Flock.ReadNet.cvWords (Lin.const false) (Lin.const true)
      (Flock.HmRow.segmentPrefix j (segs H) (segM H j))).flatten := by
  rw [cvRefs, List.flatMap_def, Flock.ReadNet.cvWords, evs, List.map_flatten, List.map_flatten, List.map_map,
    List.map_map]
  congr 1
  refine List.map_congr_left fun w _ => ?_
  simp only [Function.comp_apply, List.map_map]
  refine List.map_congr_left fun b _ => ?_
  simp only [Function.comp_apply]
  split <;> simp [hf, hc]

theorem length_constRefs (L : Layout) (v w : ℕ) : (constRefs L v w).length = w := by simp [constRefs]

theorem map_constRefs (L : Layout) {g : ℕ → Bool} (hf : g L.zero = false) (hc : g (L.const + 1) = true) (v w : ℕ) :
    (constRefs L v w).map g = (List.range w).map v.testBit := by
  simp only [constRefs, List.map_map]
  refine List.map_congr_left fun j _ => ?_
  simp only [Function.comp_apply]
  split <;> simp_all

section calls

variable (V H : ℕ)

theorem inputs_seg {j : ℕ} (hj : j < segs H) : inputs V H (lay V H) j =
    cvRefs (lay V H) H j ++ (List.range (segM H j)).map (ROW_AT + Flock.HmRow.ROW_SEG_BITS * j + ·) := by
  simp only [inputs, hj, ↓reduceIte]
  rfl

theorem inputs_outer : inputs V H (lay V H) (segs H) =
    (List.range (segs H)).flatMap fun j => (List.range 512).map ((lay V H).outAt j) := by
  simp only [inputs, lt_irrefl, ↓reduceIte]

theorem inputs_leaf : inputs V H (lay V H) (segs H + 1) =
    (List.range 512).map ((lay V H).outAt (segs H)) ++ (List.range 1536).map (saltAt H + ·) := by
  have h1 : ¬segs H + 1 < segs H := by omega
  have h2 : segs H + 1 ≠ segs H := by omega
  simp only [inputs, h1, h2, ↓reduceIte]

theorem inputs_leafHash : inputs V H (lay V H) (segs H + 2) =
    (List.range 512).map (DOMAIN_AT + ·) ++ (List.range 512).map (fun k => (lay V H).outAt (segs H + 1) (1024 + k)) ++
      (List.range (depth V)).map (0 + ·) ++
      List.replicate (8 * Frame.indexBytes (depth V) - depth V) (lay V H).zero := by
  have h1 : ¬segs H + 2 < segs H := by omega
  have h2 : segs H + 2 ≠ segs H := by omega
  have h3 : segs H + 2 ≠ segs H + 1 := by omega
  simp only [inputs, h1, h2, h3, ↓reduceIte]

theorem inputs_node {k : ℕ} (hk : k < depth V) : inputs V H (lay V H) (segs H + 3 + k) =
    (List.range 512).map (DOMAIN_AT + ·) ++ constRefs (lay V H) k 8 ++
      (List.range (depth V - 1 - k)).map (k + 1 + ·) ++
      List.replicate (8 * Frame.indexBytes (depth V - 1 - k) - (depth V - 1 - k)) (lay V H).zero ++ [k] ++
      (List.range 512).map ((lay V H).outAt (segs H + 2 + k)) ++ (List.range 512).map (pathAt H + 512 * k + ·) := by
  have h1 : ¬segs H + 3 + k < segs H := by omega
  have h2 : segs H + 3 + k ≠ segs H := by omega
  have h3 : segs H + 3 + k ≠ segs H + 1 := by omega
  have h4 : segs H + 3 + k ≠ segs H + 2 := by omega
  have h5 : segs H + 3 + k < segs H + 3 + depth V := by omega
  have h6 : segs H + 3 + k - 1 = segs H + 2 + k := by omega
  simp only [inputs, h1, h2, h3, h4, h5, h6, ↓reduceIte, Nat.add_sub_cancel_left]

theorem inputs_eq : inputs V H (lay V H) (segs H + 3 + depth V) =
    (List.range 512).map ((lay V H).outAt (segs H + 2 + depth V)) ++ (List.range 512).map (ROOT_AT + ·) := by
  have h1 : ¬segs H + 3 + depth V < segs H := by omega
  have h2 : segs H + 3 + depth V ≠ segs H := by omega
  have h3 : segs H + 3 + depth V ≠ segs H + 1 := by omega
  have h4 : segs H + 3 + depth V ≠ segs H + 2 := by omega
  have h5 : ¬segs H + 3 + depth V < segs H + 3 + depth V := by omega
  simp only [inputs, h1, h2, h3, h4, h5, ↓reduceIte]

theorem inputs_select : inputs V H (lay V H) (segs H + 4 + depth V) =
    (List.range 32).map (0 + ·) ++ [(lay V H).outAt (segs H + 3 + depth V) 0] ++
      (List.range (16 * H)).map (ROW_AT + ·) := by
  have h1 : ¬segs H + 4 + depth V < segs H := by omega
  have h2 : segs H + 4 + depth V ≠ segs H := by omega
  have h3 : segs H + 4 + depth V ≠ segs H + 1 := by omega
  have h4 : segs H + 4 + depth V ≠ segs H + 2 := by omega
  have h5 : ¬segs H + 4 + depth V < segs H + 3 + depth V := by omega
  have h6 : segs H + 4 + depth V ≠ segs H + 3 + depth V := by omega
  simp only [inputs, h1, h2, h3, h4, h5, h6, ↓reduceIte]

theorem nOut_mid {i : ℕ} (h1 : i ≠ segs H + 1) (h2 : i ≠ segs H + 3 + depth V) (h3 : i ≠ segs H + 4 + depth V) :
    nOut V H i = 512 := by
  simp only [nOut, h1, h2, h3, ↓reduceIte]

theorem nOut_leaf : nOut V H (segs H + 1) = 1536 := by simp [nOut]

theorem nOut_eq : nOut V H (segs H + 3 + depth V) = 1 := by
  have h1 : segs H + 3 + depth V ≠ segs H + 1 := by omega
  simp only [nOut, h1, ↓reduceIte]

theorem nOut_select : nOut V H (segs H + 4 + depth V) = 16 * H + 1 := by
  have h1 : segs H + 4 + depth V ≠ segs H + 1 := by omega
  have h2 : segs H + 4 + depth V ≠ segs H + 3 + depth V := by omega
  simp only [nOut, h1, h2, ↓reduceIte]

theorem seg_part {j : ℕ} (hj : j < segs H) : segPart (segM H j) ∈ (parts V H)[partOf V H j]? := by
  by_cases h : j + 1 < segs H
  · have hm : segM H j = Flock.HmRow.ROW_SEG_BITS := by
      simp only [segM, segs, rsb] at *
      omega
    simp only [partOf, h, ↓reduceIte, hm]
    rfl
  · have hm : segM H j = 16 * H - Flock.HmRow.ROW_SEG_BITS * (segs H - 1) := by
      simp only [segM, segs, rsb] at *
      omega
    simp only [partOf, h, hj, ↓reduceIte, hm]
    rfl

theorem outer_part : outerPart (16 * H) (segs H) ∈ (parts V H)[partOf V H (segs H)]? := by
  have h1 : ¬segs H + 1 < segs H := by omega
  simp only [partOf, h1, lt_irrefl, ↓reduceIte]
  rfl

theorem leaf_part : leafPart ∈ (parts V H)[partOf V H (segs H + 1)]? := by
  have h1 : ¬segs H + 1 + 1 < segs H := by omega
  have h2 : ¬segs H + 1 < segs H := by omega
  have h3 : segs H + 1 ≠ segs H := by omega
  simp only [partOf, h1, h2, h3, ↓reduceIte]
  rfl

theorem leafHash_part : leafHashPart (Frame.indexBytes (depth V)) ∈ (parts V H)[partOf V H (segs H + 2)]? := by
  have h1 : ¬segs H + 2 + 1 < segs H := by omega
  have h2 : ¬segs H + 2 < segs H := by omega
  have h3 : segs H + 2 ≠ segs H := by omega
  have h4 : segs H + 2 ≠ segs H + 1 := by omega
  simp only [partOf, h1, h2, h3, h4, ↓reduceIte]
  rfl

theorem node_part {k : ℕ} (hk : k < depth V) (hD : depth V ≤ 32) :
    nodePart (Frame.indexBytes (depth V - 1 - k)) ∈ (parts V H)[partOf V H (segs H + 3 + k)]? := by
  have h1 : ¬segs H + 3 + k + 1 < segs H := by omega
  have h2 : ¬segs H + 3 + k < segs H := by omega
  have h3 : segs H + 3 + k ≠ segs H := by omega
  have h4 : segs H + 3 + k ≠ segs H + 1 := by omega
  have h5 : segs H + 3 + k ≠ segs H + 2 := by omega
  have h6 : segs H + 3 + k < segs H + 3 + depth V := by omega
  simp only [partOf, h1, h2, h3, h4, h5, h6, ↓reduceIte, Nat.add_sub_cancel_left]
  have hn1 : 1 ≤ Frame.indexBytes (depth V - 1 - k) := le_max_left _ _
  have hn4 : Frame.indexBytes (depth V - 1 - k) ≤ 4 := by unfold Frame.indexBytes; omega
  generalize Frame.indexBytes (depth V - 1 - k) = n at hn1 hn4 ⊢
  interval_cases n <;> rfl

end calls

end Accounting.ReadUnit.Gates
