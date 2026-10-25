import Properties.Proofs.Nci.Novel

/-!
# Toward `Novel.ReadsExact`: what `Desc.analyze`'s pieces and segments name

`Novel.ReadsExact` says that the segments `Desc.analyze` gives a root node name exactly the gates outside the node that
its gates read. Its proof reads each piece, segment and sequence of segments as the list of positions it names in order
(`pos`, `posAll`, `segPos`, `psPos`), a position being a target (a parameter's or a node's leaves, or the activation's
gates of one width) and an index, and holds each of `Desc`'s operations on them to the positions it means:

* a slice of pieces, of a segment or of a sequence of segments is its positions `[a, b)` (`slice_pos`,
  `slicePieces_pos`, `segSlice_pos`, `psSlice_pos`, `pieces_pos`), and `shiftGates` moves gate positions
  (`shiftGates_pos`);
* `PS.of` lays its segments end to end (`of_ok`), `ofPart` and `ofSeq` name a decoded sequence's references in order
  (`ofSeq_pos`, `partsPos_get`), where no strided part names a negative position, which the decoder ensures
  (`parsePartsIn_nonneg`), and `memberSpans` are a batch member's indices into its argument (`memberSpans_idx`);
* a context's arguments name their references (`ctxOf_arg`), and a loop appending segments names its elements'
  positions in order, each segment it adds from its element (`forIn_append_pos`, `forIn_pm`);
* `nodeReads` gives a primitive's node its arguments' positions read whole (`nodeReads_prim`), and a composite's those
  its callee's `used` names, as the rows its members read (`Desc.memberRows`) or member by member, then a scan's
  carries (`Desc.carriesRead`), kept by `keep` (`nodeReads_comp`), each read piece by piece through `down` (`readPos`,
  `downPos`, `argPos`, `carryPos`), every segment one `down` gives a piece of an argument (`FromDown`) or of the rows
  (`FromRows`), or `carriesRead` gives (`FromCarry`); `Seg.alike` reads a copy of leaves once, naming the same
  positions (`mem_segPos_alike`, `alike_count`, `segGood_alike`) and kept as its segment is (`keep_alike`);
* `argSpans` is a member's used leaves' indices into its argument (`argSpans_batch`, `argSpans_xs`, `argSpans_other`);
* `Array.qsort` by a key sorts by it (`qsort_sorted`), so `Desc.Ivs.norm` keeps exactly its intervals' leaves
  (`norm_mem`), and `Desc.Ivs.covers` checks every leaf is in an interval (`covers_mem`);
* a segment's gates (`Width.segGates`) are its gate positions (`segGates_pos`);
* on the circuit's side, a gate's operands are its node's references resolved (`operands_mem`), and a gate of a
  primitive root node reads its member's operands in the root activation (`reads_rootPrim`);
* toward the core of the open statements, that `down` keeps the gate each position names in an activation (`den`): its
  base cases, a gate's or parameter's piece (`down_gate`, `down_param`), a primitive node's (`down_prim`, `den_prim`)
  and a batch of a primitive's (`down_batchPrim`, `den_batchPrim`), and resolution through a step (`resolve_inl`,
  `resolve_inr`);
* `down` reads a node's piece as its flat form, consecutive rows as one (`down_flat`, `pos_flat`, `flat_cases`,
  `flat_tgt`, `flat_count`, `flat_flat`), and what follows a walk's bound is what it gives (`walkAtMost_bind`);
* `down` unfolded in every other case, as Lean gives no equations for it: the loop a Call's or a batch of a
  composite's piece runs (`down_call_one`, `down_call_pts`, `down_rows`, `down_batch_affine`), and the cases it refuses
  (`down_zero`, `down_node`, `down_prim_ret`, `down_scan_ret`, `down_scan0`, `down_batch0`); with the step resolution
  takes in each, down through a composite's output into the
  member's activation (`den_call`, `den_batch`, `den_scan`) and up through a parameter to the node's argument
  (`den_up`, `den_up_batch`, `den_up_scan`, `den_up_scan0`);
* the core's induction on fuel, case by case: what `down` gives a piece naming its nodes' output leaves (`Fits`)
  agrees position by position with the gates it names (`Agree`, `DownSpec`) wherever `fromCallee` one step down
  agrees with its callee's return leaves (`CalleeSpec`): a gate's, a parameter's, a primitive's and a batch of a
  primitive's piece, a piece of rows read row by row, and a Call's piece of one row (`downSpec_gate`,
  `downSpec_param`, `downSpec_count0`, `downSpec_prim`, `downSpec_batchPrim`, `byRows_agree`, `downSpec_callOne`,
  `downSpec_callPts`), with `Resolve.DownOk` keeping each range in its callee's return; so `down`'s step holds in every
  case but a batch of a composite's (`BatchSpec`) and a scan's piece of one row (`ScanSpec`), in a body
  `Resolve.BodyOk` holds (`downSpec_succ`);
* `fromCallee`'s step: a slice of a summary's segments comes from slices of its segments (`psSlice_mem`), each of a
  copy at least and more than one only of gates or of copies all alike (`segSlice_reps`, `retOk_slice`), the latter
  read once through `throughArg` and repeated (`segPos_alike`, `forIn_src`, `posAll_flatMap_ps`); a member's gate
  positions moved by
  its offset name in the parent what they name in the member (`agree_shift`), and agreements compose across the
  activations (`agree_trans`); so `fromCallee` one step up keeps what its callee's return leaves resolve to wherever
  the callees' summaries hold `RetOk`, `throughArg` one step down keeps what its parameters' leaves name (`ThroughSpec`)
  and its segments are of a copy at least (`ThroughReps`) (`calleeSpec_succ`);
* every segment `down`, `fromCallee`, `throughArg` and `carry` give is of a copy at least, and of more than one only of
  gates or of copies all alike (`RepsOk`, `reps_fuel`), followed through `Desc`'s code (`OkIs`, `reps_step`); so
  `ThroughReps` holds (`throughReps`) and so does `RetOk`'s third part of every summary `memoOf` makes (`retOk_reps`);
* `throughArg`'s step: a member's parameter leaves, read through its node's argument (whole, a batch member's spans, or
  a scan iteration's slice, or its carry through `carry`), keep the gates they resolve to in the member wherever
  `down` and `carry` do (`CarrySpec`, `one_agree`, `throughSpec_succ`);
* a scan's piece of one row, wherever its output is its final carry's leaves and then its ys (`ScanOut`, which the
  decoder's `buildNode` gives): stride 1 reads the carry's leaves through `carry` and the ys iteration by iteration
  through `fromCallee` (`down_scan_head`, `scanOne`, `ys_agree`, `scanYs_agree`), any other stride position by
  position (`scanPts`); `scanSpec_of`;
* a batch of a composite's piece in its one-run case, the node's leaf `m·L + t` being the run's gate for member `m`'s
  return leaf `t` (`batchAffine`, `batchSpec_of`), with helpers toward the rest (`members_agree`, member by member
  through `fromCallee`; `copies_seg_agree`, whole members as copies of one member's return); so `down`, `fromCallee`,
  `throughArg` and `carry` keep the gates they read at every fuel (`specs`, `specs_of`) wherever the callees' summaries
  hold `RetOk`, a composite's return has as many leaves as its type, a primitive returns one leaf, a scan's output is
  `ScanOut`'s, the decoder's facts on the body hold (`Resolve.BodyOk`, `ArgOk`, a batch's dims), and so do the steps
  open here (`OpenSteps`): a batch of a composite's piece past its one-run case (`BatchRest`) and `carry` one step up
  (`CarrySpec`); in a Definition's body, wherever
  its references (`Resolve.BodyOk`), its callees' summaries (`RetOk`), the decoder's facts its private `composite`
  sets (`Decoded`, `ScanOut`) and `OpenSteps` do (`specs_decoded`).

So `ReadsExact` holds wherever one statement about the segments `Desc.nodeReads` gives each root node does
(`readsExact_of_rootReads`): `RootReads`. That holds wherever two smaller ones do (`rootReads_of`), named `Prop`s here,
open: `RootSegs`, that every segment `down` or `carriesRead` gives a root node names gates of one root node with their
widths, and `RootSet`, that the positions the node reads (`rootRead`) name exactly the gates outside the node that its
gates read; a composite's segment `keep` keeps is then one outside the node (`keep_iff`).
-/

namespace NciCheck.Properties.SecurityProofs.ReadsExact

open NciCheck.Desc (Piece Seg PS Tgt)

theorem flatMap_range_length {α : Type} (f : Nat → List α) (L : Nat) (hf : ∀ r, (f r).length = L) :
    ∀ n, ((List.range n).flatMap f).length = n * L
  | 0 => by simp
  | n + 1 => by
    rw [List.range_succ, List.flatMap_append, List.flatMap_singleton, List.length_append,
      flatMap_range_length f L hf n, hf, Nat.succ_mul]

/-- A list of blocks of one length `L`: element `t` is element `t % L` of block `t / L`. -/
theorem flatMap_range_getElem? {α : Type} (f : Nat → List α) (L : Nat) (hf : ∀ r, (f r).length = L) :
    ∀ (n t : Nat), ((List.range n).flatMap f)[t]? = if t < n * L then (f (t / L))[t % L]? else none
  | 0, t => by simp
  | n + 1, t => by
    rw [List.range_succ, List.flatMap_append, List.flatMap_singleton]
    have hlen := flatMap_range_length f L hf n
    have hs : (n + 1) * L = n * L + L := Nat.succ_mul n L
    by_cases ht : t < n * L
    · rw [List.getElem?_append_left (by omega), flatMap_range_getElem? f L hf n t, if_pos ht,
        if_pos (by omega)]
    · rw [List.getElem?_append_right (by omega), hlen]
      have hL : 0 < L ∨ L = 0 := by omega
      rcases hL with hL | hL
      · by_cases ht' : t < (n + 1) * L
        · rw [if_pos ht']
          have hd := Nat.div_add_mod t L
          have hm := Nat.mod_lt t hL
          have h1 : t / L = n := by
            have hlo : n * L ≤ t := by omega
            have hhi : t < (n + 1) * L := ht'
            have e1 : n ≤ t / L := (Nat.le_div_iff_mul_le hL).2 hlo
            have e2 : t / L < n + 1 := (Nat.div_lt_iff_lt_mul hL).2 hhi
            omega
          rw [h1]
          rw [h1] at hd
          have : t - n * L = t % L := by
            have : L * n = n * L := Nat.mul_comm L n
            omega
          rw [this]
        · rw [if_neg ht']
          apply List.getElem?_eq_none
          rw [hf]; omega
      · subst hL
        simp only [Nat.mul_zero]
        apply List.getElem?_eq_none
        rw [hf]; omega

/-- A piece's positions in order: row `r`, column `t` is `(tgt, base + r·rowStride + t·stride)`. -/
def pos (q : Piece) : List (Tgt × Int) :=
  (List.range q.rows).flatMap fun (r : Nat) => (List.range q.len).map fun (t : Nat) =>
    (q.tgt, q.base + (r : Int) * q.rowStride + (t : Int) * q.stride)

/-- Position `t` of a piece: row `t / len`, column `t % len`. -/
def ppt (q : Piece) (t : Nat) : Int :=
  q.base + ((t / q.len : Nat) : Int) * q.rowStride + ((t % q.len : Nat) : Int) * q.stride

theorem pos_length (q : Piece) : (pos q).length = q.count := by
  unfold pos Piece.count
  rw [flatMap_range_length _ q.len (by simp)]

theorem pos_getElem? (q : Piece) (t : Nat) :
    (pos q)[t]? = if t < q.count then some (q.tgt, ppt q t) else none := by
  unfold pos Piece.count
  rw [flatMap_range_getElem? _ q.len (by simp)]
  split
  · rename_i h
    have hL : 0 < q.len := by
      rcases Nat.eq_zero_or_pos q.len with h0 | h0
      · rw [h0, Nat.mul_zero] at h; omega
      · exact h0
    rw [List.getElem?_map, List.getElem?_range (Nat.mod_lt t hL)]
    rfl
  · rfl


theorem pos_eq_map (q : Piece) : pos q = (List.range q.count).map fun t => (q.tgt, ppt q t) := by
  apply List.ext_getElem?
  intro t
  rw [pos_getElem?, List.getElem?_map]
  by_cases h : t < q.count
  · rw [if_pos h, List.getElem?_range h]; rfl
  · rw [if_neg h, List.getElem?_eq_none (by simp; omega)]; rfl

def posAll (ps : Array Piece) : List (Tgt × Int) := ps.toList.flatMap pos

/-- A one-row piece of `q`'s target, `len'` positions of `q` from `s` (all in row `s / q.len`): `q`'s positions
`[s, s + len')`. -/
theorem range'_getElem? (s n t : Nat) : (List.range' s n)[t]? = if t < n then some (s + t) else none := by
  by_cases h : t < n
  · rw [if_pos h, List.getElem?_eq_getElem (by simpa using h), List.getElem_range']; simp
  · rw [if_neg h, List.getElem?_eq_none (by simp; omega)]

theorem toNat_div_cast (a b : Nat) : (((a / b : Nat)) : Int) = ((a / b : Nat) : Int) := rfl

/-- A one-row piece of `q`'s target, `len'` positions of `q` from `s` (all in row `s / q.len`): `q`'s positions
`[s, s + len')`. -/
theorem pos_run (q : Piece) (r s e : Nat) (B : Int) (hB : B = q.base + (r : Int) * q.rowStride + (s : Int) * q.stride)
    (hse : s ≤ e) (he : e ≤ q.len) :
    pos ⟨q.tgt, B, q.stride, e - s, 1, 0⟩ =
      (List.range' (r * q.len + s) (e - s)).map fun t => (q.tgt, ppt q t) := by
  rw [pos_eq_map]
  apply List.ext_getElem?
  intro t
  simp only [Desc.Piece.count, List.getElem?_map, Nat.one_mul]
  rw [range'_getElem?]
  by_cases h : t < e - s
  · rw [List.getElem?_range h, if_pos h]
    simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq, true_and]
    have hL : 0 < q.len := by omega
    have h1 : (r * q.len + s + t) / q.len = r := by
      rw [Nat.add_assoc, Nat.mul_comm, Nat.mul_add_div hL, Nat.div_eq_of_lt (by omega), Nat.add_zero]
    have h2 : (r * q.len + s + t) % q.len = s + t := by
      rw [Nat.add_assoc, Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt (by omega)]
    unfold ppt
    simp only [Nat.mod_eq_of_lt (show t < e - s by omega), Nat.div_eq_of_lt (show t < e - s by omega)]
    rw [h1, h2, hB]
    push_cast
    rw [Int.add_mul]; omega
  · rw [List.getElem?_eq_none (by simp; omega), if_neg h]; rfl


/-- Whole rows `[lo, lo + n)` of `q`: its positions `[lo · len, (lo + n) · len)`. -/
theorem pos_rows (q : Piece) (lo n : Nat) :
    pos ⟨q.tgt, q.base + (lo : Int) * q.rowStride, q.stride, q.len, n, q.rowStride⟩ =
      (List.range' (lo * q.len) (n * q.len)).map fun t => (q.tgt, ppt q t) := by
  rw [pos_eq_map]
  apply List.ext_getElem?
  intro t
  simp only [Desc.Piece.count, List.getElem?_map]
  rw [range'_getElem?]
  by_cases h : t < n * q.len
  · rw [List.getElem?_range h, if_pos h]
    simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq, true_and]
    have hL : 0 < q.len := by
      rcases Nat.eq_zero_or_pos q.len with h0 | h0
      · rw [h0, Nat.mul_zero] at h; omega
      · exact h0
    have h1 : (lo * q.len + t) / q.len = lo + t / q.len := by
      rw [Nat.mul_comm, Nat.mul_add_div hL]
    have h2 : (lo * q.len + t) % q.len = t % q.len := by
      rw [Nat.mul_comm, Nat.mul_add_mod]
    unfold ppt
    simp only
    rw [h1, h2]
    push_cast
    rw [Int.add_mul]; omega
  · rw [List.getElem?_eq_none (by simp; omega), if_neg h]; rfl

theorem posAll_append (a b : Array Piece) : posAll (a ++ b) = posAll a ++ posAll b := by
  simp [posAll, List.flatMap_append]

theorem posAll_singleton (q : Piece) : posAll #[q] = pos q := by simp [posAll]

theorem posAll_empty : posAll #[] = [] := rfl

theorem range'_split (a m b : Nat) (h1 : a ≤ m) (h2 : m ≤ b) :
    List.range' a (b - a) = List.range' a (m - a) ++ List.range' m (b - m) := by
  have := @List.range'_append a (m - a) (b - m) 1
  rw [show a + 1 * (m - a) = m by omega] at this
  rw [show b - a = (m - a) + (b - m) by omega, ← this]

theorem range'_join (x m n : Nat) (h : x ≤ m) :
    List.range' x (m - x) ++ List.range' m n = List.range' x (m + n - x) := by
  rw [range'_split x m (m + n) h (by omega), show m + n - m = n by omega]

/-- **A slice of a piece** is its positions `[a, b)`. -/
theorem slice_pos (q : Piece) (a b : Nat) (hb : b ≤ q.count) :
    posAll (q.slice a b) = (List.range' a (b - a)).map fun t => (q.tgt, ppt q t) := by
  unfold Desc.Piece.slice
  by_cases hab : b ≤ a
  · rw [if_pos hab, posAll_empty, show b - a = 0 by omega]; rfl
  rw [if_neg hab]
  have hcnt : q.count = q.rows * q.len := rfl
  have hL : 0 < q.len := by
    rcases Nat.eq_zero_or_pos q.len with h0 | h0
    · rw [hcnt, h0, Nat.mul_zero] at hb; omega
    · exact h0
  simp only [Desc.Piece.row, Desc.Piece.at]
  by_cases hr : q.rows ≤ 1
  · rw [if_pos hr, posAll_singleton]
    have hb' : b ≤ q.len := by
      have : q.rows * q.len ≤ 1 * q.len := Nat.mul_le_mul_right _ hr
      omega
    have := pos_run q 0 a b _ rfl (by omega) hb'
    simp only [Int.natCast_zero, Int.zero_mul, Int.add_zero, Nat.zero_mul, Nat.zero_add] at this
    exact this
  rw [if_neg hr]
  have hpart : ∀ r s e : Nat, s ≤ e → e ≤ q.len →
      posAll (if s < e then #[(⟨q.tgt, q.base + (r : Int) * q.rowStride + (s : Int) * q.stride, q.stride, e - s, 1, 0⟩ :
        Piece)] else #[]) = (List.range' (r * q.len + s) (e - s)).map fun t => (q.tgt, ppt q t) := by
    intro r s e h1 h2
    split
    · rw [posAll_singleton]
      exact pos_run q r s e _ rfl h1 h2
    · rw [posAll_empty, show e - s = 0 by omega]; rfl
  have ha := Nat.div_add_mod a q.len
  have hb' := Nat.div_add_mod b q.len
  have ha2 := Nat.mod_lt a hL
  have hb2 := Nat.mod_lt b hL
  have hdiv : a / q.len ≤ b / q.len := Nat.div_le_div_right (by omega)
  rw [Nat.mul_comm] at ha hb'
  generalize a / q.len = r0 at *
  generalize a % q.len = t0 at *
  generalize b / q.len = r1 at *
  generalize b % q.len = t1 at *
  split
  · rename_i heq
    have heq' : r0 = r1 := by simpa using heq
    subst heq'
    rw [hpart _ _ _ (by omega) (by omega)]
    congr 2; omega
  · rename_i hne
    have hne' : r0 < r1 := by
      have : r0 ≠ r1 := by simpa using hne
      omega
    have hlast := hpart r1 0 t1 (by omega) (by omega)
    rw [posAll_append, posAll_append, hlast, Nat.add_zero, Nat.sub_zero]
    have hmid : ∀ lo, lo ≤ r1 → posAll (if lo < r1 then #[(⟨q.tgt, q.base + (lo : Int) * q.rowStride, q.stride, q.len,
        r1 - lo, q.rowStride⟩ : Piece)] else #[]) =
        (List.range' (lo * q.len) (r1 * q.len - lo * q.len)).map fun t => (q.tgt, ppt q t) := by
      intro lo hlo2
      split
      · rw [posAll_singleton, pos_rows, Nat.sub_mul]
      · rw [posAll_empty, show r1 * q.len - lo * q.len = 0 by
          have : r1 = lo := by omega
          subst this; omega]
        rfl
    have hend : r1 * q.len + (b - r1 * q.len) = b := by omega
    by_cases h0 : t0 = 0
    · subst h0
      simp only [show ((0 : Nat) == 0) = true from rfl, if_true, posAll_empty, List.nil_append]
      rw [hmid r0 (by omega), ← List.map_append, Nat.add_zero,
        range'_join _ _ _ (Nat.mul_le_mul_right _ (by omega))]
      congr 2; omega
    · have h0' : (t0 == 0) = false := by simpa using h0
      simp only [h0', Bool.false_eq_true, if_false]
      rw [posAll_singleton, pos_run q r0 t0 q.len _ rfl (by omega) (Nat.le_refl _), hmid (r0 + 1) (by omega),
        ← List.map_append, ← List.map_append, Nat.add_zero]
      have e1 : r0 * q.len + t0 + (q.len - t0) = (r0 + 1) * q.len := by rw [Nat.succ_mul]; omega
      have e2 : (r0 + 1) * q.len ≤ r1 * q.len := Nat.mul_le_mul_right _ (by omega)
      rw [show q.len - t0 = (r0 + 1) * q.len - (r0 * q.len + t0) by omega,
        range'_join _ _ _ (by omega), show (r0 + 1) * q.len + (r1 * q.len - (r0 + 1) * q.len) - (r0 * q.len + t0) =
          r1 * q.len - (r0 * q.len + t0) by omega]
      simp only [Nat.add_zero]
      rw [range'_join _ _ _ (by omega)]
      congr 2; omega

/-! ## Slices of pieces laid end to end -/

/-- Positions `[a, b)` of a list. -/
def sl {α : Type} (l : List α) (a b : Nat) : List α := (l.drop a).take (b - a)

theorem sl_append {α : Type} (l₁ l₂ : List α) (a b : Nat) :
    sl (l₁ ++ l₂) a b = sl l₁ a (min b l₁.length) ++ sl l₂ (a - l₁.length) (b - l₁.length) := by
  unfold sl
  rw [List.drop_append, List.take_append, List.length_drop]
  congr 1
  · rw [List.take_eq_take_iff, List.length_drop]; omega
  · congr 1; omega

theorem sl_map_range {α : Type} (f : Nat → α) (n a b : Nat) (hb : b ≤ n) :
    sl ((List.range n).map f) a b = (List.range' a (b - a)).map f := by
  unfold sl
  rw [← List.map_drop, ← List.map_take, List.range_eq_range', List.drop_range',
    List.take_range'_of_length_ge (by omega)]
  congr 2; omega

theorem slice_sl (q : Piece) (a b : Nat) (hb : b ≤ q.count) : posAll (q.slice a b) = sl (pos q) a b := by
  rw [slice_pos q a b hb, pos_eq_map, sl_map_range _ _ _ _ hb]

def slGo (a b : Nat) : List Piece → Nat → Array Piece → Array Piece
  | [], _, out => out
  | q :: qs, st, out =>
    if st ≥ b then out
    else slGo a b qs (st + q.count)
      (if st + q.count > a then out ++ q.slice (max a st - st) (min b (st + q.count) - st) else out)

theorem slicePieces_go (ps : Array Piece) (a b : Nat) : Desc.slicePieces ps a b = slGo a b ps.toList 0 #[] := by
  unfold Desc.slicePieces
  simp only [Id.run]
  rw [← Array.forIn_toList]
  generalize ps.toList = l
  generalize (#[] : Array Piece) = out
  generalize (0 : Nat) = st
  induction l generalizing st out with
  | nil => rfl
  | cons q qs ih =>
    simp only [List.forIn_cons, slGo]
    split
    · rfl
    · rw [← ih]
      split <;> rfl

theorem sl_nil {α : Type} (a b : Nat) : sl ([] : List α) a b = [] := by simp [sl]

theorem sl_empty {α : Type} (l : List α) (a b : Nat) (h : b ≤ a ∨ l.length ≤ a) : sl l a b = [] := by
  unfold sl
  rcases h with h | h
  · rw [show b - a = 0 by omega, List.take_zero]
  · rw [List.drop_eq_nil_of_le h, List.take_nil]

theorem slGo_pos (a b : Nat) : ∀ (l : List Piece) (st : Nat) (out : Array Piece),
    posAll (slGo a b l st out) = posAll out ++ sl (l.flatMap pos) (a - st) (b - st)
  | [], st, out => by simp only [slGo, List.flatMap_nil, sl_nil, List.append_nil]
  | q :: qs, st, out => by
    unfold slGo
    split
    · rw [sl_empty _ _ _ (Or.inl (by omega)), List.append_nil]
    · rw [slGo_pos a b qs, List.flatMap_cons, sl_append, pos_length, ← List.append_assoc,
        show a - st - q.count = a - (st + q.count) by omega, show b - st - q.count = b - (st + q.count) by omega]
      congr 1
      split
      · rw [posAll_append, slice_sl _ _ _ (by omega)]
        congr 2 <;> omega
      · rw [sl_empty (pos q) _ _ (Or.inr (by rw [pos_length]; omega)), List.append_nil]

/-- **Positions `[a, b)` of pieces laid end to end.** -/
theorem slicePieces_pos (ps : Array Piece) (a b : Nat) :
    posAll (Desc.slicePieces ps a b) = sl (posAll ps) a b := by
  rw [slicePieces_go, slGo_pos, posAll_empty, List.nil_append, Nat.sub_zero, Nat.sub_zero]; rfl

/-! ## Copies and segments -/

/-- A position moved by `δ` when it is a gate's. -/
def shiftPos (δ : Int) (x : Tgt × Int) : Tgt × Int :=
  match x.1 with
  | .gate _ => (x.1, x.2 + δ)
  | .ref _ _ => x

theorem shiftPos_zero : shiftPos 0 = id := by
  funext x
  unfold shiftPos
  split <;> simp

theorem shiftPos_shiftPos (δ ε : Int) : shiftPos ε ∘ shiftPos δ = shiftPos (δ + ε) := by
  funext x
  rcases x with ⟨t, v⟩
  cases t <;> simp [shiftPos, Int.add_assoc]

theorem pos_shiftGate (δ : Int) (q : Piece) :
    pos (match q.tgt with
      | .gate _ => { q with base := q.base + δ }
      | _ => q) = (pos q).map (shiftPos δ) := by
  rcases q with ⟨t, base, stride, len, rows, rowStride⟩
  cases t with
  | gate w =>
    simp only [pos, List.map_flatMap, List.map_map]
    congr 1; funext r; congr 1; funext t
    simp only [Function.comp, shiftPos]
    congr 1; omega
  | ref p i =>
    simp only [pos, List.map_flatMap, List.map_map]
    congr 1

/-- **Moving the gates by `δ`** moves each gate's position by `δ`. -/
theorem shiftGates_pos (δ : Int) (ps : Array Piece) :
    posAll (Desc.shiftGates δ ps) = (posAll ps).map (shiftPos δ) := by
  unfold Desc.shiftGates posAll
  rw [Array.toList_map, List.flatMap_map, List.map_flatMap]
  congr 1; funext q
  exact pos_shiftGate δ q

theorem shiftGates_length (δ : Int) (ps : Array Piece) :
    (posAll (Desc.shiftGates δ ps)).length = (posAll ps).length := by
  rw [shiftGates_pos, List.length_map]

/-- A segment's positions: its copies' in order. -/
def segPos (s : Seg) : List (Tgt × Int) := (List.range s.reps).flatMap fun m => posAll (s.copy m)

def segsPos (ss : Array Seg) : List (Tgt × Int) := ss.toList.flatMap segPos

theorem segsPos_append (a b : Array Seg) : segsPos (a ++ b) = segsPos a ++ segsPos b := by
  simp [segsPos, List.flatMap_append]

theorem segsPos_empty : segsPos #[] = [] := rfl

theorem segsPos_singleton (s : Seg) : segsPos #[s] = segPos s := by simp [segsPos]

theorem copy_pos (s : Seg) (m : Nat) : posAll (s.copy m) = (posAll s.ps).map (shiftPos ((m : Int) * s.shift)) :=
  shiftGates_pos _ _

theorem segPos_one (q : Piece) : segPos (Seg.one q) = pos q := by
  simp only [segPos, Seg.one, Desc.Seg.copy, List.range_one, List.flatMap_singleton, shiftGates_pos,
    Int.mul_zero, shiftPos_zero, List.map_id, posAll_singleton]

theorem segsPos_one (ps : Array Piece) : segsPos (ps.map Seg.one) = posAll ps := by
  simp only [segsPos, posAll, Array.toList_map, List.flatMap_map, segPos_one]

/-- A list of blocks of one length `L`, as the map of its positions. -/
theorem blocks_eq_map {α : Type} (F : Nat → List α) (L : Nat) (hF : ∀ m, (F m).length = L) (d : α) (n : Nat) :
    (List.range n).flatMap F = (List.range (n * L)).map fun t => (F (t / L)).getD (t % L) d := by
  apply List.ext_getElem?
  intro t
  rw [flatMap_range_getElem? F L hF, List.getElem?_map]
  by_cases h : t < n * L
  · have hL : 0 < L := by
      rcases Nat.eq_zero_or_pos L with h0 | h0
      · rw [h0, Nat.mul_zero] at h; omega
      · exact h0
    have hm : t % L < (F (t / L)).length := by rw [hF]; exact Nat.mod_lt t hL
    rw [if_pos h, List.getElem?_range h]
    simp only [Option.map_some, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hm, Option.getD_some]
  · rw [if_neg h, List.getElem?_eq_none (by simp; omega)]; rfl

theorem list_eq_map {α : Type} (l : List α) (d : α) : l = (List.range l.length).map fun t => l.getD t d := by
  apply List.ext_getElem?
  intro t
  rw [List.getElem?_map]
  by_cases h : t < l.length
  · rw [List.getElem?_range h, List.getElem?_eq_getElem h]
    simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h]
  · rw [List.getElem?_eq_none (by omega), List.getElem?_eq_none (by simp; omega)]; rfl

theorem copy_length (s : Seg) (m : Nat) : (posAll (s.copy m)).length = (posAll s.ps).length := by
  rw [copy_pos, List.length_map]

theorem copy_zero (s : Seg) : posAll (s.copy 0) = posAll s.ps := by
  rw [copy_pos, show ((0 : Nat) : Int) * s.shift = 0 by simp, shiftPos_zero, List.map_id]

theorem copy_copy (s : Seg) (lo m : Nat) :
    posAll (Desc.shiftGates ((m : Int) * s.shift) (s.copy lo)) = posAll (s.copy (lo + m)) := by
  rw [shiftGates_pos, copy_pos, copy_pos, List.map_map, shiftPos_shiftPos, Int.natCast_add, Int.add_mul]

/-- Position `t` of a segment whose pieces name `s.len` positions: position `t % len` of copy `t / len`. -/
def spt (s : Seg) (t : Nat) : Tgt × Int := (posAll (s.copy (t / s.len))).getD (t % s.len) default

theorem segPos_eq_map (s : Seg) (hlen : (posAll s.ps).length = s.len) :
    segPos s = (List.range s.count).map (spt s) :=
  blocks_eq_map (fun m => posAll (s.copy m)) s.len (fun m => by rw [copy_length, hlen]) default s.reps

theorem spt_add (s : Seg) (m t : Nat) (ht : t < s.len) :
    spt s (m * s.len + t) = (posAll (s.copy m)).getD t default := by
  have hL : 0 < s.len := by omega
  unfold spt
  rw [show m * s.len + t = t + s.len * m by rw [Nat.mul_comm]; omega, Nat.add_mul_div_left _ _ hL,
    Nat.add_mul_mod_self_left, Nat.div_eq_of_lt ht, Nat.mod_eq_of_lt ht, Nat.zero_add]

/-- **A slice of a segment** is its positions `[a, b)`. -/
theorem segSlice_pos (s : Seg) (hlen : (posAll s.ps).length = s.len) (a b : Nat) (hb : b ≤ s.count) :
    segsPos (s.slice a b) = sl (segPos s) a b := by
  rw [segPos_eq_map s hlen, sl_map_range _ _ _ _ hb]
  unfold Desc.Seg.slice
  by_cases hab : b ≤ a
  · rw [if_pos hab, segsPos_empty, show b - a = 0 by omega]; rfl
  rw [if_neg hab]
  have hcnt : s.count = s.reps * s.len := rfl
  have hL : 0 < s.len := by
    rcases Nat.eq_zero_or_pos s.len with h0 | h0
    · rw [hcnt, h0, Nat.mul_zero] at hb; omega
    · exact h0
  by_cases hr : s.reps ≤ 1
  · rw [if_pos hr, segsPos_one, slicePieces_pos]
    have hr1 : s.reps = 1 := by
      rcases Nat.lt_or_ge s.reps 1 with h | h
      · rw [hcnt, show s.reps = 0 by omega, Nat.zero_mul] at hb; omega
      · omega
    have e : segPos s = posAll s.ps := by
      simp only [segPos, hr1, List.range_one, List.flatMap_singleton, copy_zero]
    rw [← e, segPos_eq_map s hlen, sl_map_range _ _ _ _ hb]
  rw [if_neg hr]
  dsimp only
  have hpart : ∀ m x y : Nat, x ≤ y → y ≤ s.len →
      segsPos ((Desc.slicePieces (s.copy m) x y).map Seg.one) =
        (List.range' (m * s.len + x) (y - x)).map (spt s) := by
    intro m x y hxy hy
    have hl : (posAll (s.copy m)).length = s.len := by rw [copy_length, hlen]
    rw [segsPos_one, slicePieces_pos]
    conv => lhs; rw [list_eq_map (posAll (s.copy m)) default]
    rw [hl, sl_map_range _ _ _ _ hy, ← List.map_add_range', List.map_map]
    apply List.map_congr_left
    intro t ht
    obtain ⟨i, hi, rfl⟩ := List.mem_range'.1 ht
    simp only [Function.comp, Nat.one_mul]
    rw [spt_add s m _ (by omega)]
  have hmid : ∀ lo m1 : Nat, lo ≤ m1 →
      segsPos (if lo < m1 then #[{ s with ps := s.copy lo, reps := m1 - lo }] else #[]) =
        (List.range' (lo * s.len) (m1 * s.len - lo * s.len)).map (spt s) := by
    intro lo m1 hlo
    split
    · rw [segsPos_singleton]
      have e : segPos { s with ps := s.copy lo, reps := m1 - lo } =
          (List.range (m1 - lo)).flatMap fun m => posAll (s.copy (lo + m)) := by
        unfold segPos
        congr 1; funext m
        exact copy_copy s lo m
      rw [e, blocks_eq_map (fun m => posAll (s.copy (lo + m))) s.len (fun m => by rw [copy_length, hlen]) default,
        List.range'_eq_map_range, List.map_map, Nat.sub_mul]
      apply List.map_congr_left
      intro t _
      simp only [Function.comp, spt]
      rw [show lo * s.len + t = t + s.len * lo by rw [Nat.mul_comm]; omega, Nat.add_mul_div_left _ _ hL,
        Nat.add_mul_mod_self_left, Nat.add_comm (t / s.len) lo]
    · rw [segsPos_empty, show m1 * s.len - lo * s.len = 0 by
        have : m1 = lo := by omega
        subst this; omega]
      rfl
  have ha := Nat.div_add_mod a s.len
  have hb' := Nat.div_add_mod b s.len
  have ha2 := Nat.mod_lt a hL
  have hb2 := Nat.mod_lt b hL
  have hdiv : a / s.len ≤ b / s.len := Nat.div_le_div_right (by omega)
  rw [Nat.mul_comm] at ha hb'
  generalize a / s.len = r0 at *
  generalize a % s.len = t0 at *
  generalize b / s.len = r1 at *
  generalize b % s.len = t1 at *
  split
  · rename_i heq
    have heq' : r0 = r1 := by simpa using heq
    subst heq'
    rw [hpart _ _ _ (by omega) (by omega)]
    congr 2; omega
  · rename_i hne
    have hne' : r0 < r1 := by
      have : r0 ≠ r1 := by simpa using hne
      omega
    have hlast := hpart r1 0 t1 (by omega) (by omega)
    rw [segsPos_append, segsPos_append, hlast, Nat.add_zero, Nat.sub_zero]
    have hend : r1 * s.len + (b - r1 * s.len) = b := by omega
    by_cases h0 : t0 = 0
    · subst h0
      simp only [show ((0 : Nat) == 0) = true from rfl, if_true, segsPos_empty, List.nil_append]
      rw [hmid r0 r1 (by omega), ← List.map_append, Nat.add_zero,
        range'_join _ _ _ (Nat.mul_le_mul_right _ (by omega))]
      congr 2; omega
    · have h0' : (t0 == 0) = false := by simpa using h0
      simp only [h0', Bool.false_eq_true, if_false]
      rw [hpart r0 t0 s.len (by omega) (Nat.le_refl _), hmid (r0 + 1) r1 (by omega),
        ← List.map_append, ← List.map_append]
      have e1 : r0 * s.len + t0 + (s.len - t0) = (r0 + 1) * s.len := by rw [Nat.succ_mul]; omega
      have e2 : (r0 + 1) * s.len ≤ r1 * s.len := Nat.mul_le_mul_right _ (by omega)
      rw [show s.len - t0 = (r0 + 1) * s.len - (r0 * s.len + t0) by omega,
        range'_join _ _ _ (by omega), show (r0 + 1) * s.len + (r1 * s.len - (r0 + 1) * s.len) - (r0 * s.len + t0) =
          r1 * s.len - (r0 * s.len + t0) by omega]
      simp only [Nat.add_zero]
      rw [range'_join _ _ _ (by omega)]
      congr 2; omega

/-! ## Sequences of segments -/

/-- A segment whose pieces name `len` positions. -/
def SegOk (s : Seg) : Prop := (posAll s.ps).length = s.len

theorem segPos_length (s : Seg) (h : SegOk s) : (segPos s).length = s.count := by
  unfold segPos Seg.count
  exact flatMap_range_length _ s.len (fun m => by rw [copy_length]; exact h) s.reps

theorem segsPos_length (ss : List Seg) (h : ∀ g ∈ ss, SegOk g) :
    (ss.flatMap segPos).length = (ss.map Seg.count).sum := by
  induction ss with
  | nil => rfl
  | cons g gs ih =>
    rw [List.flatMap_cons, List.length_append, segPos_length g (h g (by simp)),
      ih (fun g' hg' => h g' (by simp [hg'])), List.map_cons, List.sum_cons]

/-- A sequence's positions: its segments' in order. -/
def psPos (s : PS) : List (Tgt × Int) := segsPos s.segs

/-- What `PS.of` makes: segments that name their `len` positions, of positive count, each starting where the ones before
it end. -/
def PSOk (s : PS) : Prop :=
  (∀ i < s.segs.size, SegOk s.segs[i]! ∧ 0 < s.segs[i]!.count) ∧ s.starts.size = s.segs.size ∧
    (∀ i < s.segs.size, s.starts[i]! = ((s.segs.toList.take i).map Seg.count).sum) ∧
    s.count = (s.segs.toList.map Seg.count).sum

theorem forIn_inv {α σ : Type} (P : σ → Prop) (f : α → σ → Id (ForInStep σ))
    (hstep : ∀ x st, P st → P (f x st).value) : ∀ (l : List α) (init : σ), P init → P (forIn l init f)
  | [], init, h => h
  | x :: l, init, h => by
    rw [List.forIn_cons]
    have hx := hstep x init h
    revert hx
    cases f x init with
    | done b => intro hx; exact hx
    | yield b => intro hx; exact forIn_inv P f hstep l b hx

/-- `PS.slice`'s loop over the segments from the one the bisection found. -/
def psGo (s : PS) (lo hi : Nat) : List Nat → Array Seg → Array Seg
  | [], out => out
  | i :: is, out =>
    if s.starts[i]! ≥ hi then out
    else psGo s lo hi is (if s.starts[i]! + s.segs[i]!.count > lo then out ++ s.segs[i]!.slice
      (max lo s.starts[i]! - s.starts[i]!) (min hi (s.starts[i]! + s.segs[i]!.count) - s.starts[i]!) else out)

theorem psGo_eq (s : PS) (lo hi : Nat) : ∀ (l : List Nat) (out : Array Seg),
    (forIn l out fun i (r : Array Seg) =>
      if s.starts[i]! ≥ hi then (pure (ForInStep.done r) : Id _)
      else if s.starts[i]! + s.segs[i]!.count > lo then
        pure (ForInStep.yield (r ++ s.segs[i]!.slice (max lo s.starts[i]! - s.starts[i]!)
          (min hi (s.starts[i]! + s.segs[i]!.count) - s.starts[i]!)))
      else pure (ForInStep.yield r)) = psGo s lo hi l out
  | [], _ => rfl
  | i :: is, out => by
    rw [List.forIn_cons]
    simp only [psGo]
    split
    · rfl
    · split
      · exact psGo_eq s lo hi is _
      · exact psGo_eq s lo hi is _

theorem psGo_pos (s : PS) (hs : PSOk s) (lo hi : Nat) : ∀ (n a : Nat) (out : Array Seg), a + n = s.segs.size →
    segsPos (psGo s lo hi (List.range' a n) out) = segsPos out ++
      sl ((s.segs.toList.drop a).flatMap segPos) (lo - ((s.segs.toList.take a).map Seg.count).sum)
        (hi - ((s.segs.toList.take a).map Seg.count).sum)
  | 0, a, out, h => by
    rw [List.drop_eq_nil_of_le (by simp; omega)]
    simp [psGo, sl]
  | n + 1, a, out, h => by
    have hseg := hs.1
    have hst := hs.2.2.1
    have ha : a < s.segs.size := by omega
    have hga : s.segs[a]! = s.segs.toList[a]'(by simpa using ha) := by
      rw [getElem!_pos s.segs a ha, Array.getElem_toList]
    have hpre : ((s.segs.toList.take (a + 1)).map Seg.count).sum =
        ((s.segs.toList.take a).map Seg.count).sum + s.segs[a]!.count := by
      rw [List.take_add_one, List.getElem?_eq_getElem (by simpa using ha), hga]
      simp
    rw [List.range'_succ]
    simp only [psGo]
    rw [hst a ha]
    generalize ((s.segs.toList.take a).map Seg.count).sum = st at hpre
    split
    · rw [sl_empty _ _ _ (Or.inl (by omega)), List.append_nil]
    · rw [psGo_pos s hs lo hi n (a + 1) _ (by omega), hpre,
        List.drop_eq_getElem_cons (i := a) (by simpa using ha), ← hga, List.flatMap_cons, sl_append,
        segPos_length _ (hseg a ha).1, ← List.append_assoc,
        show lo - st - s.segs[a]!.count = lo - (st + s.segs[a]!.count) by omega,
        show hi - st - s.segs[a]!.count = hi - (st + s.segs[a]!.count) by omega]
      congr 1
      split
      · rw [segsPos_append, segSlice_pos _ (hseg a ha).1 _ _ (by omega)]
        congr 2 <;> omega
      · rw [sl_empty (segPos s.segs[a]!) _ _ (Or.inr (by rw [segPos_length _ (hseg a ha).1]; omega)),
          List.append_nil]

theorem segOk_of_mem (s : PS) (hs : PSOk s) (g : Seg) (hg : g ∈ s.segs.toList) : SegOk g := by
  obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hg
  have hi' : i < s.segs.size := by simpa using hi
  have := (hs.1 i hi').1
  rwa [getElem!_pos s.segs i hi', ← Array.getElem_toList] at this

theorem bind_id {α β : Type} (x : Id α) (f : α → Id β) : (x >>= f) = f x := rfl

/-- **A slice of a sequence** is its positions `[lo, hi)`. -/
theorem psSlice_pos (s : PS) (hs : PSOk s) (lo hi : Nat) : segsPos (s.slice lo hi) = sl (psPos s) lo hi := by
  unfold Desc.PS.slice
  simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size]
  split
  · rename_i h
    simp only [Bool.or_eq_true, decide_eq_true_eq] at h
    rcases h with h | h
    · exact (sl_empty _ _ _ (Or.inl h)).symm
    · have h0 : s.segs.toList = [] := by simpa using h
      show [] = (((s.segs.toList.flatMap segPos).drop lo).take (hi - lo))
      rw [h0]; simp
  · rename_i h
    simp only [Bool.or_eq_true, decide_eq_true_eq, not_or] at h
    have hne : 0 < s.segs.size := by
      rcases Nat.eq_zero_or_pos s.segs.size with h0 | h0
      · exfalso; apply h.2; simp [Array.isEmpty, h0]
      · exact h0
    generalize hB : (forIn (m := Id) (List.range' 0 _) (0, s.segs.size) _ : Id (Nat × Nat)) = B
    have hinv : B.1 < s.segs.size ∧ B.2 ≤ s.segs.size ∧ (B.1 = 0 ∨ s.starts[B.1]! ≤ lo) := by
      rw [← hB]
      refine forIn_inv (fun x : Nat × Nat => x.1 < s.segs.size ∧ x.2 ≤ s.segs.size ∧ (x.1 = 0 ∨ s.starts[x.1]! ≤ lo))
        _ ?_ _ _ ⟨hne, Nat.le_refl _, Or.inl rfl⟩
      intro x st ⟨h1, h2, h3⟩
      split
      · exact ⟨h1, h2, h3⟩
      · rename_i hgt
        split
        · rename_i hle
          exact ⟨by show (st.1 + st.2) / 2 < s.segs.size; omega, h2, Or.inr hle⟩
        · exact ⟨h1, by show (st.1 + st.2) / 2 ≤ s.segs.size; omega, h3⟩
    rw [bind_id, psGo_eq, show (s.segs.size - B.1 + 1 - 1) / 1 = s.segs.size - B.1 by omega, bind_id]
    change segsPos (psGo s lo hi (List.range' B.1 (s.segs.size - B.1)) #[]) = _
    rw [psGo_pos s hs lo hi (s.segs.size - B.1) B.1 #[] (by omega), segsPos_empty, List.nil_append]
    obtain ⟨h1, -, h3⟩ := hinv
    generalize B.1 = a at h1 h3
    have hpre : ((s.segs.toList.take a).map Seg.count).sum ≤ lo := by
      rcases h3 with h3 | h3
      · subst h3; simp
      · rw [← hs.2.2.1 a h1]; exact h3
    show _ = sl (s.segs.toList.flatMap segPos) lo hi
    conv => rhs; rw [← List.take_append_drop a s.segs.toList]
    have hlen : ((s.segs.toList.take a).flatMap segPos).length = ((s.segs.toList.take a).map Seg.count).sum :=
      segsPos_length _ fun g hg => segOk_of_mem s hs g (List.mem_of_mem_take hg)
    rw [List.flatMap_append, sl_append, hlen, sl_empty ((s.segs.toList.take a).flatMap segPos) lo _
      (Or.inr (by rw [hlen]; exact hpre)), List.nil_append]

theorem posAll_flatMap {β : Type} (xs : Array β) (f : β → Array Piece) :
    posAll (xs.flatMap f) = xs.toList.flatMap fun x => posAll (f x) := by
  simp [posAll, Array.toList_flatMap, List.flatMap_assoc]

/-- **A sequence's pieces `[lo, hi)`** are its positions `[lo, hi)`. -/
theorem pieces_pos (s : PS) (hs : PSOk s) (lo hi : Nat) : posAll (s.pieces lo hi) = sl (psPos s) lo hi := by
  rw [← psSlice_pos s hs lo hi]
  unfold Desc.PS.pieces
  rw [posAll_flatMap]
  unfold segsPos
  congr 1; funext g
  rw [posAll_flatMap, Array.toList_range]
  rfl

theorem of_fold (l : List Seg) : ∀ (A : Array Nat) (c : Nat),
    l.foldl (fun (acc : Array Nat × Nat) s => (acc.1.push acc.2, acc.2 + s.count)) (A, c) =
      (A ++ ((List.range l.length).map fun i => c + ((l.take i).map Seg.count).sum).toArray,
        c + (l.map Seg.count).sum) := by
  induction l with
  | nil => intro A c; simp
  | cons g gs ih =>
    intro A c
    rw [List.foldl_cons, ih]
    simp only [List.length_cons, List.map_cons, List.sum_cons, Prod.mk.injEq]
    refine ⟨?_, by omega⟩
    apply Array.ext'
    simp only [Array.toList_append, Array.toList_push, List.append_assoc]
    congr 1
    rw [List.range_succ_eq_map, List.map_cons, List.map_map]
    simp only [List.take_zero, List.map_nil, List.sum_nil, Nat.add_zero, List.singleton_append]
    congr 1
    apply List.map_congr_left
    intro i _
    simp only [Function.comp, List.take_succ_cons, List.map_cons, List.sum_cons]
    omega

theorem flatMap_filter_count (l : List Seg) (h : ∀ g ∈ l, SegOk g) :
    (l.filter (fun g => decide (g.count > 0))).flatMap segPos = l.flatMap segPos := by
  induction l with
  | nil => rfl
  | cons g gs ih =>
    have ih' := ih (fun g' hg' => h g' (by simp [hg']))
    rw [List.filter_cons]
    split
    · rw [List.flatMap_cons, List.flatMap_cons, ih']
    · rename_i hc
      have h0 : g.count = 0 := by simpa using hc
      have : segPos g = [] := List.eq_nil_of_length_eq_zero (by rw [segPos_length g (h g (by simp)), h0])
      rw [List.flatMap_cons, this, List.nil_append, ih']

theorem of_segs (segs : Array Seg) : (PS.of segs).segs = segs.filter fun g => decide (g.count > 0) := rfl

theorem of_starts_count (segs : Array Seg) :
    ((PS.of segs).starts, (PS.of segs).count) = (segs.filter fun g => decide (g.count > 0)).foldl
      (fun (acc : Array Nat × Nat) s => (acc.1.push acc.2, acc.2 + s.count)) (#[], 0) := rfl

/-- **`PS.of`** keeps its segments' positions, and makes a sequence `PS.slice` reads. -/
theorem of_ok (segs : Array Seg) (h : ∀ g ∈ segs.toList, SegOk g) :
    PSOk (PS.of segs) ∧ psPos (PS.of segs) = segsPos segs ∧ (PS.of segs).count = (segsPos segs).length := by
  have hf : ∀ g ∈ (PS.of segs).segs.toList, SegOk g ∧ 0 < g.count := by
    intro g hg
    rw [of_segs, Array.toList_filter, List.mem_filter] at hg
    exact ⟨h g hg.1, by simpa using hg.2⟩
  have hsc := of_starts_count segs
  rw [← Array.foldl_toList, of_fold, ← of_segs] at hsc
  simp only [Prod.mk.injEq] at hsc
  obtain ⟨hst, hcnt⟩ := hsc
  have hpos : psPos (PS.of segs) = segsPos segs := by
    unfold psPos segsPos
    rw [of_segs, Array.toList_filter, flatMap_filter_count _ h]
  have hsz : (PS.of segs).segs.size = (PS.of segs).segs.toList.length := by simp
  refine ⟨⟨?_, ?_, ?_, ?_⟩, hpos, ?_⟩
  · intro i hi
    have hi' : i < (PS.of segs).segs.toList.length := by rw [← hsz]; exact hi
    have := hf _ (List.getElem_mem hi')
    rwa [getElem!_pos _ i hi, ← Array.getElem_toList]
  · rw [hst, hsz]; simp
  · intro i hi
    have hi' : i < (PS.of segs).segs.toList.length := by rw [← hsz]; exact hi
    rw [hst, getElem!_pos _ i (by simpa using hi')]
    simp
  · rw [hcnt, Nat.zero_add]
  · rw [hcnt, Nat.zero_add, ← hpos]
    unfold psPos segsPos
    rw [segsPos_length _ (fun g hg => (hf g hg).1)]

/-! ## Reference sequences as pieces -/

open Flock.Program (Ref Part Seq)

/-- A reference as a position: its collection's leaf. -/
def refPos (r : Ref) : Tgt × Int := (.ref r.p r.i, (r.j : Int))

/-- A part's references never step below leaf 0: a strided part's pattern (`parsePartsIn` checks its corners). -/
def PartNonneg : Part → Prop
  | .strided _ _ b inner os is c => ∀ k < c, 0 ≤ b + ((k / inner : Nat) : Int) * os + ((k % inner : Nat) : Int) * is
  | _ => True

theorem pos_one (t : Tgt) (base stride : Int) (len : Nat) (rs : Int) :
    pos ⟨t, base, stride, len, 1, rs⟩ = (List.range len).map fun (k : Nat) => (t, base + (k : Int) * stride) := by
  simp [pos]

theorem posAll_filter_count (ps : Array Piece) :
    posAll (ps.filter fun q => decide (q.count > 0)) = posAll ps := by
  unfold posAll
  rw [Array.toList_filter]
  generalize ps.toList = l
  induction l with
  | nil => rfl
  | cons q qs ih =>
    rw [List.filter_cons]
    split
    · rw [List.flatMap_cons, List.flatMap_cons, ih]
    · rename_i hc
      have h0 : q.count = 0 := by simpa using hc
      have : pos q = [] := List.eq_nil_of_length_eq_zero (by rw [pos_length, h0])
      rw [List.flatMap_cons, this, List.nil_append, ih]

theorem posAll_pair (x y : Piece) : posAll #[x, y] = pos x ++ pos y := by simp [posAll]

theorem range_split (a n : Nat) : List.range (a + n) = List.range a ++ List.range' a n := by
  rw [List.range_eq_range', List.range_eq_range', show List.range' a n = List.range' (0 + a) n by rw [Nat.zero_add],
    List.range'_append_1]

theorem tgt_eq_of_beq {t : Tgt} {p : Bool} {i : Nat} (h : (t == Tgt.ref p i) = true) : t = Tgt.ref p i := by
  cases t with
  | gate w =>
    have e : (Tgt.gate w == Tgt.ref p i) = false := rfl
    rw [e] at h; cases h
  | ref p' i' =>
    have e : (Tgt.ref p' i' == Tgt.ref p i) = (p' == p && i' == i) := rfl
    rw [e, Bool.and_eq_true, beq_iff_eq, beq_iff_eq] at h
    rw [h.1, h.2]

theorem ofPart_affine (p : Bool) (i b c : Nat) :
    posAll (Desc.ofPart (.affine p i b c)) = (List.range c).map fun k => refPos ((Part.affine p i b c).get k) := by
  simp only [Desc.ofPart, posAll_singleton, pos_one, Flock.Program.Part.get, refPos]
  apply List.map_congr_left
  intro k _
  simp

theorem ofPart_strided (p : Bool) (i : Nat) (b : Int) (inner : Nat) (os is : Int) (c : Nat)
    (hq : PartNonneg (.strided p i b inner os is c)) :
    posAll (Desc.ofPart (.strided p i b inner os is c)) =
      (List.range c).map fun k => refPos ((Part.strided p i b inner os is c).get k) := by
  have hget : ∀ k < c, refPos ((Part.strided p i b inner os is c).get k) =
      (.ref p i, b + ((k / inner : Nat) : Int) * os + ((k % inner : Nat) : Int) * is) := by
    intro k hk
    simp only [Flock.Program.Part.get, refPos]
    rw [Int.toNat_of_nonneg (hq k hk)]
  have hmap : ∀ (f : Nat → Tgt × Int), (∀ k < c, f k = (.ref p i,
      b + ((k / inner : Nat) : Int) * os + ((k % inner : Nat) : Int) * is)) →
      (List.range c).map f = (List.range c).map fun k => refPos ((Part.strided p i b inner os is c).get k) := by
    intro f hf
    apply List.map_congr_left
    intro k hk
    have hk' := List.mem_range.1 hk
    rw [hf k hk', hget k hk']
  simp only [Desc.ofPart]
  split
  · rename_i h0
    have : c = 0 := by simpa using h0
    subst this; rfl
  split
  · rename_i h1
    have : inner = 1 := by simpa using h1
    subst this
    rw [posAll_singleton, pos_one]
    apply hmap
    intro k _
    simp
  split
  · rename_i h2
    have h2' : os = (inner : Int) * is := by simpa using h2
    rw [posAll_singleton, pos_one]
    apply hmap
    intro k _
    have hk : (k : Int) = ((k / inner : Nat) : Int) * inner + ((k % inner : Nat) : Int) := by
      have := Nat.div_add_mod k inner
      rw [Nat.mul_comm] at this
      exact_mod_cast this.symm
    rw [h2', hk, Int.add_mul, ← Int.mul_assoc, Int.add_assoc]
  · rw [posAll_filter_count, posAll_pair, pos_one, pos_eq_map]
    simp only [Piece.count]
    have hc : c = c / inner * inner + c % inner := by
      have := Nat.div_add_mod c inner
      rw [Nat.mul_comm] at this; omega
    conv => rhs; rw [hc]
    rw [range_split, List.map_append]
    rw [← hc]
    congr 1
    · apply List.map_congr_left
      intro k hk
      have hk' := List.mem_range.1 hk
      have hkc : k < c := by
        have : c / inner * inner ≤ c := Nat.div_mul_le_self c inner
        omega
      rw [hget k hkc]
      simp [ppt]
    · rw [List.range'_eq_map_range, List.map_map]
      apply List.map_congr_left
      intro t ht
      have ht' := List.mem_range.1 ht
      have hkc : c / inner * inner + t < c := by omega
      simp only [Function.comp]
      rw [hget _ hkc]
      by_cases hin : inner = 0
      · subst hin
        simp
      · have hin' : 0 < inner := by omega
        have hlt : t < inner := by
          have := Nat.mod_lt c hin'
          omega
        rw [show c / inner * inner + t = t + inner * (c / inner) by rw [Nat.mul_comm]; omega,
          Nat.add_mul_div_left _ _ hin', Nat.add_mul_mod_self_left, Nat.div_eq_of_lt hlt, Nat.mod_eq_of_lt hlt,
          Nat.zero_add]

/-- `ofPart`'s step over an explicit part's reference: extend the last piece when the reference is its next element. -/
def exStep (acc : Array Piece) (r : Ref) : Array Piece :=
  let one : Piece := { tgt := .ref r.p r.i, base := r.j, stride := 1, len := 1 }
  match acc.back? with
  | some q => if q.tgt == .ref r.p r.i && q.stride == 1 && q.at q.len == r.j then acc.pop.push { q with len := q.len + 1 }
    else acc.push one
  | none => acc.push one

theorem posAll_push (acc : Array Piece) (q : Piece) : posAll (acc.push q) = posAll acc ++ pos q := by
  simp [posAll]

theorem exStep_spec (acc : Array Piece) (r : Ref) (h : ∀ q ∈ acc.toList, q.rows = 1) :
    posAll (exStep acc r) = posAll acc ++ [refPos r] ∧ ∀ q ∈ (exStep acc r).toList, q.rows = 1 := by
  have hone : pos { tgt := .ref r.p r.i, base := r.j, stride := 1, len := 1 } = [refPos r] := by
    rw [pos_one]; simp [refPos]
  unfold exStep
  rcases hb : acc.back? with _ | q
  · refine ⟨by rw [posAll_push, hone], ?_⟩
    intro x hx
    rw [Array.toList_push, List.mem_append] at hx
    rcases hx with hx | hx
    · exact h x hx
    · simp at hx; rw [hx]
  · obtain ⟨ys, rfl⟩ := Array.back?_eq_some_iff.1 hb
    have hq : q.rows = 1 := h q (by simp)
    have hys : ∀ x ∈ ys.toList, x.rows = 1 := fun x hx => h x (by simp [hx])
    dsimp only
    split
    · rename_i hc
      simp only [Bool.and_eq_true, beq_iff_eq] at hc
      obtain ⟨⟨htgt, hst⟩, hat⟩ := hc
      rw [Array.pop_push]
      refine ⟨?_, ?_⟩
      · rw [posAll_push, posAll_push, List.append_assoc]
        congr 1
        rcases q with ⟨t, base, stride, len, rows, rs⟩
        simp only at hq hst hat htgt
        subst hq hst
        rw [pos_one, pos_one, List.range_succ, List.map_append]
        simp only [Desc.Piece.at, Int.mul_one] at hat
        rw [tgt_eq_of_beq htgt]
        simp [refPos, hat]
      · intro x hx
        rw [Array.toList_push, List.mem_append] at hx
        rcases hx with hx | hx
        · exact hys x hx
        · simp at hx; rw [hx]; exact hq
    · refine ⟨by rw [posAll_push, hone], ?_⟩
      intro x hx
      rw [Array.toList_push, List.mem_append] at hx
      rcases hx with hx | hx
      · exact h x hx
      · simp at hx; rw [hx]

theorem exFold_spec : ∀ (l : List Ref) (acc : Array Piece), (∀ q ∈ acc.toList, q.rows = 1) →
    posAll (l.foldl exStep acc) = posAll acc ++ l.map refPos
  | [], acc, _ => by simp
  | r :: l, acc, h => by
    obtain ⟨h1, h2⟩ := exStep_spec acc r h
    rw [List.foldl_cons, exFold_spec l _ h2, h1, List.map_cons, List.append_assoc, List.singleton_append]

theorem ofPart_explicit (it : Array Ref) :
    posAll (Desc.ofPart (.explicit it)) = (List.range it.size).map fun k => refPos ((Part.explicit it).get k) := by
  have e : Desc.ofPart (.explicit it) = it.toList.foldl exStep #[] := by
    simp only [Desc.ofPart]
    rw [← Array.foldl_toList]
    rfl
  rw [e, exFold_spec _ _ (by simp), posAll_empty, List.nil_append]
  apply List.ext_getElem?
  intro k
  rw [List.getElem?_map, List.getElem?_map]
  by_cases hk : k < it.size
  · rw [List.getElem?_range hk, Array.getElem?_toList, Array.getElem?_eq_getElem hk]
    simp [Flock.Program.Part.get, getElem!_pos it k hk]
  · rw [List.getElem?_eq_none (by simp; omega), List.getElem?_eq_none (by simp; omega)]; rfl

/-- **A part as pieces** names its references, in order. -/
theorem ofPart_pos (q : Part) (hq : PartNonneg q) :
    posAll (Desc.ofPart q) = (List.range q.count).map fun k => refPos (q.get k) := by
  cases q with
  | affine p i b c => exact ofPart_affine p i b c
  | strided p i b inner os is c => exact ofPart_strided p i b inner os is c hq
  | explicit it => exact ofPart_explicit it

/-- A sequence's references as positions, part by part. -/
def partsPos (ps : Array Part) : List (Tgt × Int) :=
  ps.toList.flatMap fun q => (List.range q.count).map fun k => refPos (q.get k)

theorem partsFold_pos : ∀ (l : List Part) (acc : Array Piece), (∀ q ∈ l, PartNonneg q) →
    posAll (l.foldl (fun acc q => acc ++ Desc.ofPart q) acc) =
      posAll acc ++ l.flatMap fun q => (List.range q.count).map fun k => refPos (q.get k)
  | [], acc, _ => by simp
  | q :: l, acc, h => by
    rw [List.foldl_cons, partsFold_pos l _ (fun q' hq' => h q' (by simp [hq'])), posAll_append,
      ofPart_pos q (h q (by simp)), List.flatMap_cons, List.append_assoc]

theorem segOk_one (q : Piece) : SegOk (Seg.one q) := by
  show (posAll #[q]).length = q.count
  rw [posAll_singleton, pos_length]

/-- **A sequence as segments** (`Desc.ofSeq`) names its references, in order. -/
theorem ofSeq_pos (s : Seq) (hq : ∀ q ∈ s.parts.toList, PartNonneg q) :
    PSOk (Desc.ofSeq s) ∧ psPos (Desc.ofSeq s) = partsPos s.parts ∧
      (Desc.ofSeq s).count = (partsPos s.parts).length := by
  have hok : ∀ g ∈ ((s.parts.foldl (fun acc q => acc ++ Desc.ofPart q) #[]).map Seg.one).toList, SegOk g := by
    intro g hg
    rw [Array.toList_map, List.mem_map] at hg
    obtain ⟨q, -, rfl⟩ := hg
    exact segOk_one q
  have hpos : segsPos ((s.parts.foldl (fun acc q => acc ++ Desc.ofPart q) #[]).map Seg.one) = partsPos s.parts := by
    rw [segsPos_one, ← Array.foldl_toList, partsFold_pos _ _ hq, posAll_empty, List.nil_append]; rfl
  obtain ⟨h1, h2, h3⟩ := of_ok _ hok
  exact ⟨h1, h2.trans hpos, h3.trans (by rw [hpos])⟩

/-- Block `i` of a list of blocks starts after the blocks before it. -/
theorem flatMap_getElem?_block {α β : Type} (F : α → List β) :
    ∀ (l : List α) (i t : Nat) (hi : i < l.length), t < (F l[i]).length →
      (l.flatMap F)[((l.take i).map fun x => (F x).length).sum + t]? = (F l[i])[t]?
  | [], _, _, hi, _ => absurd hi (by simp)
  | x :: l, 0, t, _, ht => by
    simp only [List.take_zero, List.map_nil, List.sum_nil, Nat.zero_add, List.flatMap_cons,
      List.getElem_cons_zero] at ht ⊢
    rw [List.getElem?_append_left ht]
  | x :: l, i + 1, t, hi, ht => by
    simp only [List.take_succ_cons, List.map_cons, List.sum_cons, List.flatMap_cons, List.getElem_cons_succ]
    rw [List.getElem?_append_right (by omega), Nat.add_assoc, Nat.add_sub_cancel_left]
    exact flatMap_getElem?_block F l i t (by simpa using hi) ht

/-! ## A member's runs of a mapped argument -/

/-- Runs `[x, y)` laid end to end. -/
def spansPos (xs : Array (Nat × Nat)) : List Nat := xs.toList.flatMap fun p => List.range' p.1 (p.2 - p.1)

theorem range'_shift (X s n : Nat) : List.range' X n = (List.range' s n).map fun t => X + (t - s) := by
  rw [List.range'_eq_map_range, List.range'_eq_map_range, List.map_map]
  apply List.map_congr_left
  intro i _
  simp only [Function.comp]
  omega

/-- The runs of `[lo, hi)` in blocks of `L`, from block `q0`, for `n` blocks. -/
theorem blocks_cover (L lo hi q0 : Nat) (h0 : q0 * L ≤ lo) : ∀ n : Nat,
    (List.range n).flatMap (fun q' => List.range' (max lo ((q0 + q') * L))
      (min hi ((q0 + q' + 1) * L) - max lo ((q0 + q') * L))) =
      List.range' lo (min hi ((q0 + n) * L) - lo)
  | 0 => by
    rw [List.range_zero, List.flatMap_nil, Nat.add_zero, show min hi (q0 * L) - lo = 0 by omega]; rfl
  | n + 1 => by
    rw [List.range_succ, List.flatMap_append, blocks_cover L lo hi q0 h0 n, List.flatMap_singleton]
    have hY : (q0 + n + 1) * L = (q0 + n) * L + L := Nat.succ_mul _ _
    have hX : q0 * L ≤ (q0 + n) * L := Nat.mul_le_mul_right _ (by omega)
    rw [show q0 + (n + 1) = q0 + n + 1 by omega]
    generalize (q0 + n) * L = X at hY hX
    rw [hY]
    by_cases hx : hi ≤ X
    · rw [show min hi (X + L) - max lo X = 0 by omega, show min hi X = hi by omega,
        show min hi (X + L) = hi by omega]
      simp
    · by_cases hl : X ≤ lo
      · rw [show min hi X - lo = 0 by omega, show max lo X = lo by omega]
        simp
      · rw [show max lo X = X by omega, show min hi X = X by omega, range'_join _ _ _ (by omega)]
        congr 1; omega

theorem flatMap_filterMap_spans {β : Type} (g : β → Option (Nat × Nat)) (l : List β) :
    (l.filterMap g).flatMap (fun p => List.range' p.1 (p.2 - p.1)) =
      l.flatMap fun b => match g b with
        | some p => List.range' p.1 (p.2 - p.1)
        | none => [] := by
  induction l with
  | nil => rfl
  | cons b l ih =>
    rw [List.filterMap_cons, List.flatMap_cons]
    cases g b with
    | none => simp [ih]
    | some p => simp [ih]

/-- **A member's runs** (`Desc.memberSpans`) are the argument's indices of the member's parameter leaves `[lo, hi)`,
in order: leaf `t` of member `m` of an argument mapped with dims `(d, L)` is index `(t / L · d + m) · L + t % L`. -/
theorem memberSpans_pos (a : Flock.Program.Arg) (m lo hi : Nat) (d L : Nat) (hd : a.dims = some (d, L))
    (hL : L = 0 → hi ≤ lo) :
    spansPos (Desc.memberSpans a m lo hi) =
      (List.range' lo (hi - lo)).map fun t => (t / L * d + m) * L + t % L := by
  unfold Desc.memberSpans
  rw [hd]
  dsimp only
  split
  · rename_i h
    have : L = 0 := by simpa using h
    rw [show hi - lo = 0 by have := hL this; omega]; rfl
  rename_i h
  have hL0 : 0 < L := by
    have : L ≠ 0 := by simpa using h
    omega
  unfold spansPos
  rw [Array.toList_filterMap, Array.toList_range, flatMap_filterMap_spans]
  have hq0 : lo / L * L ≤ lo := Nat.div_mul_le_self lo L
  have hblock : ∀ q' : Nat,
      (match (if max lo ((lo / L + q') * L) < min hi ((lo / L + q' + 1) * L) then
          some (((lo / L + q') * d + m) * L + (max lo ((lo / L + q') * L) - (lo / L + q') * L),
            ((lo / L + q') * d + m) * L + (min hi ((lo / L + q' + 1) * L) - (lo / L + q') * L)) else none) with
        | some p => List.range' p.1 (p.2 - p.1)
        | none => []) =
      (List.range' (max lo ((lo / L + q') * L)) (min hi ((lo / L + q' + 1) * L) - max lo ((lo / L + q') * L))).map
        fun t => (t / L * d + m) * L + t % L := by
    intro q'
    generalize lo / L + q' = q
    have hY : (q + 1) * L = q * L + L := Nat.succ_mul _ _
    by_cases hse : max lo (q * L) < min hi ((q + 1) * L)
    · rw [if_pos hse]
      dsimp only
      rw [range'_shift _ (max lo (q * L)), show ((q * d + m) * L + (min hi ((q + 1) * L) - q * L) -
          ((q * d + m) * L + (max lo (q * L) - q * L))) = min hi ((q + 1) * L) - max lo (q * L) by omega]
      apply List.map_congr_left
      intro t ht
      obtain ⟨i, hi', rfl⟩ := List.mem_range'.1 ht
      have hM1 : q * L ≤ max lo (q * L) := Nat.le_max_right _ _
      have hlt : max lo (q * L) + 1 * i - q * L < L := by omega
      have hsplit : max lo (q * L) + 1 * i = (max lo (q * L) + 1 * i - q * L) + L * q := by
        rw [Nat.mul_comm L q]; omega
      have e1 : (max lo (q * L) + 1 * i) / L = q := by
        conv => lhs; rw [hsplit]
        rw [Nat.add_mul_div_left _ _ hL0, Nat.div_eq_of_lt hlt, Nat.zero_add]
      have e2 : (max lo (q * L) + 1 * i) % L = max lo (q * L) + 1 * i - q * L := by
        conv => lhs; rw [hsplit]
        rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hlt]
      rw [e1, e2]
      generalize max lo (q * L) = M at *
      generalize q * L = Q at *
      omega
    · rw [if_neg hse]
      dsimp only
      rw [show min hi ((q + 1) * L) - max lo (q * L) = 0 by omega]; rfl
  simp only [hblock]
  rw [← List.map_flatMap, blocks_cover L lo hi (lo / L) hq0]
  congr 1
  by_cases hlh : hi ≤ lo
  · rw [show hi - lo = 0 by omega]
    rw [show min hi ((lo / L + ((hi + L - 1) / L - lo / L)) * L) - lo = 0 by omega]
  · have hce : hi ≤ (hi + L - 1) / L * L := by
      have h1 := Nat.div_add_mod (hi + L - 1) L
      have h2 := Nat.mod_lt (hi + L - 1) hL0
      have h3 : (hi + L - 1) / L * L = L * ((hi + L - 1) / L) := Nat.mul_comm _ _
      omega
    have hle : lo / L ≤ (hi + L - 1) / L := Nat.div_le_div_right (by omega)
    rw [show lo / L + ((hi + L - 1) / L - lo / L) = (hi + L - 1) / L by omega, show min hi ((hi + L - 1) / L * L) = hi
      by omega]

theorem memberSpans_none (a : Flock.Program.Arg) (m lo hi : Nat) (hd : a.dims = none) :
    spansPos (Desc.memberSpans a m lo hi) = List.range' lo (hi - lo) := by
  unfold Desc.memberSpans
  rw [hd]
  simp [spansPos]

/-- **Reference `k` of a decoded sequence** is position `k` of its parts' (`Walk.seq_get`). -/
theorem partsPos_get (ps : Array Part) :
    (List.range (Seq.ofParts ps).count).map (fun k => refPos ((Seq.ofParts ps).get k)) = partsPos ps := by
  obtain ⟨-, -, hst, hcnt⟩ := Walk.ofParts_spec ps
  have hlen : (partsPos ps).length = (Seq.ofParts ps).count := by
    unfold partsPos
    rw [hcnt, List.length_flatMap]
    simp
  apply List.ext_getElem?
  intro k
  rw [List.getElem?_map]
  by_cases hk : k < (Seq.ofParts ps).count
  · obtain ⟨i, hi, h1, h2, hget⟩ := Walk.seq_get ps hk
    have hi' : i < ps.toList.length := by simpa using hi
    have hpi : ps.toList[i] = ps[i]! := by rw [Array.getElem_toList, getElem!_pos ps i hi]
    have hblock := flatMap_getElem?_block (fun q : Part => (List.range q.count).map fun k => refPos (q.get k))
      ps.toList i (k - (Seq.ofParts ps).starts[i]!) hi' (by simp only [List.length_map, List.length_range, hpi]; omega)
    have hsum : ((ps.toList.take i).map fun x => ((List.range x.count).map fun k => refPos (x.get k)).length).sum =
        (Seq.ofParts ps).starts[i]! := by
      rw [hst i hi]; simp
    rw [hsum, Nat.add_sub_cancel' h1] at hblock
    unfold partsPos
    rw [hblock, List.getElem?_range hk, Option.map_some, hget, hpi, List.getElem?_map,
      List.getElem?_range (by omega)]
    rfl
  · rw [List.getElem?_eq_none (by simp; omega)]
    symm
    apply List.getElem?_eq_none
    omega

/-- **A member's runs** are its parameter leaves' indices into the argument (`Resolve.memberIdx`). -/
theorem memberSpans_idx (a : Flock.Program.Arg) (m lo hi : Nat)
    (hL : ∀ d L, a.dims = some (d, L) → L = 0 → hi ≤ lo) :
    spansPos (Desc.memberSpans a m lo hi) = (List.range' lo (hi - lo)).map (Resolve.memberIdx a m) := by
  rcases hd : a.dims with _ | ⟨d, L⟩
  · rw [memberSpans_none a m lo hi hd]
    conv => lhs; rw [← List.map_id (List.range' lo (hi - lo))]
    apply List.map_congr_left
    intro t _
    simp [Resolve.memberIdx, hd]
  · rw [memberSpans_pos a m lo hi d L hd (hL d L hd)]
    apply List.map_congr_left
    intro t _
    simp [Resolve.memberIdx, hd]


/-! ## Gates, as `Width.segGates` names them -/

/-- A position as `Width.segGates` names it: a gate with its width; a parameter's or a node's leaf names none. -/
def gatePt (x : Tgt × Int) : Option (Int × Nat) :=
  match x.1 with
  | .gate w => some (x.2, w)
  | .ref _ _ => none

theorem flatMap_congr {α β : Type} {l : List α} {f g : α → List β} (h : ∀ a ∈ l, f a = g a) :
    l.flatMap f = l.flatMap g := by
  simp only [List.flatMap, List.map_congr_left h]

/-- **A piece's gates are its gate positions**, where it has a row or no columns (`pieceGates` reads a piece of no rows
as one row). -/
theorem pieceGates_pos (q : Piece) (h : q.rows ≠ 0 ∨ q.len = 0) :
    Width.pieceGates q = (pos q).filterMap gatePt := by
  unfold Width.pieceGates pos
  rw [List.filterMap_flatMap]
  rcases ht : q.tgt with ⟨p, i⟩ | w
  · dsimp only
    symm
    apply List.flatMap_eq_nil_iff.2
    intro r _
    rw [List.filterMap_map]
    simp [gatePt, Function.comp_def]
  · dsimp only
    rcases h with h | h
    · rw [show max q.rows 1 = q.rows by omega]
      apply flatMap_congr
      intro r _
      rw [List.filterMap_map]
      simp [gatePt, Function.comp_def]
    · rw [List.flatMap_eq_nil_iff.2 (fun _ _ => by rw [h]; rfl), List.flatMap_eq_nil_iff.2 (fun _ _ => by rw [h]; rfl)]

theorem piecesGates_pos (ps : Array Piece) (h : ∀ q ∈ ps.toList, q.rows ≠ 0 ∨ q.len = 0) :
    Width.piecesGates ps = (posAll ps).filterMap gatePt := by
  unfold Width.piecesGates posAll
  rw [List.filterMap_flatMap]
  exact flatMap_congr fun q hq => pieceGates_pos q (h q hq)

theorem shiftGates_rows (δ : Int) (ps : Array Piece) :
    ∀ q ∈ (Desc.shiftGates δ ps).toList, ∃ q' ∈ ps.toList, q.rows = q'.rows ∧ q.len = q'.len := by
  intro q hq
  simp only [Desc.shiftGates, Array.toList_map, List.mem_map] at hq
  obtain ⟨q', hq', rfl⟩ := hq
  refine ⟨q', hq', ?_⟩
  split <;> exact ⟨rfl, rfl⟩

/-- **A segment's gates are its gate positions**, where it has a copy and each piece a row or no columns. -/
theorem segGates_pos (s : Seg) (hr : s.reps ≠ 0) (h : ∀ q ∈ s.ps.toList, q.rows ≠ 0 ∨ q.len = 0) :
    Width.segGates s = (segPos s).filterMap gatePt := by
  unfold Width.segGates segPos
  rw [List.filterMap_flatMap]
  by_cases h1 : s.reps ≤ 1
  · rw [if_pos h1, show s.reps = 1 by omega, List.range_one, List.flatMap_singleton, copy_zero]
    exact piecesGates_pos s.ps h
  · rw [if_neg h1]
    apply flatMap_congr
    intro m _
    apply piecesGates_pos
    intro q hq
    obtain ⟨q', hq', e1, e2⟩ := shiftGates_rows _ _ q hq
    rw [e1, e2]; exact h q' hq'


/-! ## Parsed strided parts name no negative leaf -/

/-- One unfolding of `parsePartsIn`, as `Walk.parsePartsIn_step` takes it, for `PartNonneg`: a strided part's corners
are within its collection, so every position it names is (`Walk.strided_corners`). -/
theorem parsePartsIn_nonneg_step (leaves : Bool → Nat → Option Nat) (args : Nat → Nat → Option Seq)
    (hargs : ∀ k x s, args k x = some s → ∀ q ∈ s.parts, PartNonneg q) (n : Nat)
    (b : Nat.below (motive := fun _ => Lean.Json → Flock.V (Array Part)) (n + 1))
    (hb : ∀ x ps, b.1 x = .ok ps → ∀ q ∈ ps, PartNonneg q) (j : Lean.Json) (ps : Array Part)
    (h : Flock.Program.parsePartsIn._f leaves args (n + 1) b j = .ok ps) : ∀ q ∈ ps, PartNonneg q := by
  delta Flock.Program.parsePartsIn._f at h
  dsimp only at h
  cases j with
  | null => exact nomatch h
  | bool _ => exact nomatch h
  | num _ => exact nomatch h
  | str _ => exact nomatch h
  | arr _ => exact nomatch h
  | obj kvs =>
    dsimp only at h
    split at h
    · rename_i a _
      rcases a with _ | _ | _ | _ | ⟨⟨l⟩⟩ | _
      all_goals try (injection h; done)
      rcases l with _ | ⟨k, _ | ⟨x, _ | ⟨y, l⟩⟩⟩
      all_goals try (injection h; done)
      conv at h => lhs; whnf
      split at h
      all_goals try (injection h; done)
      split at h
      all_goals try (injection h; done)
      rename_i s hs
      injection h with h
      subst h
      exact hargs _ _ s hs
    · split at h
      · rename_i c _
        rcases c with _ | _ | _ | _ | cs | _
        all_goals try (injection h; done)
        dsimp only at h
        rcases hm : cs.mapM (fun x => b.1 x) with e | r <;> simp only [hm, bind, Except.bind] at h
        · exact absurd h (by intro h; injection h)
        injection h with h; subst h
        intro q hq
        rw [← Array.foldl_toList] at hq
        rcases Walk.mem_foldl_append r.toList #[] q hq with hq | ⟨y, hy, hq⟩
        · simp at hq
        · obtain ⟨x, -, hx⟩ := Walk.array_mapM_ok _ cs r hm y (by simpa using hy)
          exact hb x y hx q hq
      · split at h
        · rename_i a _
          rcases a with _ | _ | _ | _ | ⟨⟨l⟩⟩ | _
          all_goals try (injection h; done)
          rcases l with _ | ⟨s0, _ | ⟨i0, _ | ⟨b0, _ | ⟨c0, _ | ⟨y, l⟩⟩⟩⟩⟩
          all_goals try (injection h; done)
          conv at h => lhs; whnf
          split at h
          all_goals try (injection h; done)
          split at h
          · injection h with h; subst h
            intro q hq
            simp only [Array.mem_singleton] at hq
            subst hq
            trivial
          · split at h
            all_goals try (injection h; done)
            injection h with h; subst h
            intro q hq
            simp only [Array.mem_singleton] at hq
            subst hq
            trivial
        · split at h
          · rename_i a _
            rcases a with _ | _ | _ | _ | ⟨⟨l⟩⟩ | _
            all_goals try (injection h; done)
            rcases l with _ | ⟨s0, _ | ⟨i0, _ | ⟨b0, _ | ⟨n0, _ | ⟨o0, _ | ⟨j0, _ | ⟨c0, _ | ⟨y, l⟩⟩⟩⟩⟩⟩⟩⟩
            all_goals try (injection h; done)
            conv at h => lhs; whnf
            split at h
            all_goals try (injection h; done)
            rename_i _ cc inn ost ist c _
            split at h
            all_goals try (injection h; done)
            rename_i hinn
            try dsimp only at h
            split at h
            · injection h with h; subst h
              intro q hq
              simp only [Array.mem_singleton] at hq
              subst hq
              intro k hk
              exact absurd hk (Nat.not_lt_zero k)
            try dsimp only at h
            split at h
            all_goals try (injection h; done)
            rename_i p i hw
            injection h with h; subst h
            intro q hq
            simp only [Array.mem_singleton] at hq
            subst hq
            intro k hk
            have hin : 0 < inn := by simp at hinn; omega
            obtain ⟨⟨⟨x1, y1⟩, hp1, h1⟩, -⟩ := Walk.strided_corners cc ost ist hin hk
            split at hw
            all_goals try (injection hw; done)
            split at hw
            all_goals try (injection hw; done)
            split at hw
            all_goals rename_i hq0
            · rw [if_pos hq0] at hp1
              split at hw
              all_goals try (injection hw; done)
              rename_i hc
              simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
              have e1 := le_trans hc.1 (Walk.foldl_min_le _ _ _ (Or.inl (List.mem_map_of_mem hp1)))
              dsimp only at h1 e1
              exact le_trans e1 h1
            · rw [if_neg hq0] at hp1
              split at hw
              all_goals try (injection hw; done)
              rename_i hc
              simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
              have e1 := le_trans hc.1 (Walk.foldl_min_le _ _ _ (Or.inl (List.mem_map_of_mem hp1)))
              dsimp only at h1 e1
              exact le_trans e1 h1
          · split at h
            · rename_i e _
              rcases e with _ | _ | _ | _ | items | _
              all_goals try (injection h; done)
              dsimp only at h
              obtain ⟨rs, -, h⟩ := Walk.except_bind_ok h
              injection h with h; subst h
              intro q hq
              simp only [Array.mem_singleton] at hq
              subst hq
              trivial
            · exact absurd h (by intro h; injection h)

/-- **No position of a parsed strided part is negative**, given none of an argument an alias may name is: its corners'
least is within its collection (`parsePartsIn`'s `within`), so `Desc.ofPart`'s piece names its references
(`ofPart_pos`). -/
theorem parsePartsIn_nonneg (leaves : Bool → Nat → Option Nat) (args : Nat → Nat → Option Seq)
    (hargs : ∀ k x s, args k x = some s → ∀ q ∈ s.parts, PartNonneg q) :
    ∀ fuel j ps, Flock.Program.parsePartsIn fuel j leaves args = .ok ps → ∀ q ∈ ps, PartNonneg q := by
  intro fuel
  induction fuel with
  | zero =>
    intro j ps h
    delta Flock.Program.parsePartsIn at h
    exact nomatch h
  | succ n ih =>
    intro j ps h
    delta Flock.Program.parsePartsIn at h
    exact parsePartsIn_nonneg_step leaves args hargs n _
      (fun x ps hx => ih x ps (by delta Flock.Program.parsePartsIn; exact hx)) j ps h


/-! ## `ReadsExact` from what `nodeReads` gives each root node

`Desc.analyze` summarizes the reached Definitions (`memoOf`), then reads each root node with those summaries
(`readsOf`, `analyze_eq`), refusing a read that names no gate. So `ReadsExact` holds wherever `RootReads` does, which
speaks of `Desc.nodeReads`' segments as the positions they name (`segPos`): their gates are `Width.segGates`'
(`segGates_pos`), and the root node of a segment's first piece is `Desc.nodeOf`'s (`nodeOf_root`). -/

open Flock.Program (Prog)
open DescAgree (nodeCount rootOffs)

/-- `Desc.analyze`'s summaries of the reached Definitions, callees first. -/
def memoOf (P : Prog) : Except String (Array Desc.Summary) := do
  let seen := Desc.reached P
  let mut memo : Array Desc.Summary := #[]
  for d in [0:P.defs.size] do
    memo := memo.push (← if seen[d]! && d != P.root then Desc.summarize P memo d else pure default)
  return memo

/-- The context `Desc.analyze` reads the root's nodes in. -/
def rootCtx (P : Prog) (memo : Array Desc.Summary) : Desc.Ctx :=
  { P, memo, d := P.root, args := (P.defs[P.root]!).body.nodes.map fun n => n.args.map (Desc.ofSeq ·.seq) }

/-- `Desc.analyze`'s reads of the root's nodes, given the summaries. -/
def readsOf (P : Prog) (memo : Array Desc.Summary) : Except String (Array (Array Seg)) := do
  let body := (P.defs[P.root]!).body
  let c := rootCtx P memo
  let mut reads := #[]
  for k in [0:body.nodes.size] do
    let rs ← Desc.nodeReads c k
    if rs.any (·.ps.any fun q => match q.tgt with | .gate _ => false | _ => true) then throw "program"
    reads := reads.push rs
  return reads

theorem analyze_eq (P : Prog) : Desc.analyze P = (memoOf P >>= readsOf P) := by
  unfold Desc.analyze memoOf readsOf
  simp only [bind_assoc, pure_bind]
  rfl

/-- A loop that pushes one value per element, each one `G` holds of. -/
theorem forIn_push {ε α : Type} [Inhabited α] (f : Nat → Array α → Except ε (ForInStep (Array α)))
    (G : Nat → α → Prop) (hf : ∀ k r y, f k r = .ok y → ∃ x, G k x ∧ y = .yield (r.push x)) :
    ∀ (l : List Nat) (acc out : Array α), forIn l acc f = .ok out →
      out.size = acc.size + l.length ∧ (∀ i < acc.size, out[i]! = acc[i]!) ∧
        ∀ i (hi : i < l.length), G l[i] out[acc.size + i]!
  | [], acc, out, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨by simp, fun _ _ => rfl, fun i hi => absurd hi (by simp)⟩
  | k :: l, acc, out, h => by
    rw [List.forIn_cons] at h
    rcases hk : f k acc with e | y <;> rw [hk] at h
    · exact absurd h (by intro h; cases h)
    obtain ⟨x, hx, rfl⟩ := hf k acc _ hk
    obtain ⟨h1, h2, h3⟩ := forIn_push f G hf l (acc.push x) out h
    simp only [Array.size_push] at h1 h2 h3
    refine ⟨by simp only [List.length_cons]; omega, fun i hi => ?_, fun i hi => ?_⟩
    · rw [h2 i (by omega), getElem!_pos (acc.push x) i (by simp; omega), getElem!_pos acc i hi,
        Array.getElem_push_lt]
    · rcases i with _ | i
      · rw [Nat.add_zero, h2 acc.size (by omega), getElem!_pos (acc.push x) acc.size (by simp),
          Array.getElem_push_eq]
        exact hx
      · have := h3 i (by simp at hi; omega)
        simp only [List.getElem_cons_succ]
        rw [show acc.size + (i + 1) = acc.size + 1 + i by omega]
        exact this


theorem pieces_gates (rs : Array Seg)
    (h : (rs.any fun x => x.ps.any fun q => match q.tgt with | .gate _ => false | _ => true) = false) :
    ∀ s ∈ rs, ∀ q ∈ s.ps, ∃ w, q.tgt = .gate w := by
  intro s hs q hq
  rw [Array.any_eq_false] at h
  obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hs
  have h2 := h i hi
  rw [Bool.not_eq_true, Array.any_eq_false] at h2
  obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 hq
  have h3 := h2 j hj
  revert h3
  rcases (rs[i].ps[j]).tgt with ⟨p, i'⟩ | w
  · simp
  · exact fun _ => ⟨w, rfl⟩

/-- What `readsOf` gives: per root node, `nodeReads`' segments, every piece of which names gates. -/
theorem readsOf_spec (P : Prog) (memo : Array Desc.Summary) (reads : Array (Array Seg))
    (h : readsOf P memo = .ok reads) :
    reads.size = nodeCount P ∧ ∀ k < nodeCount P, Desc.nodeReads (rootCtx P memo) k = .ok reads[k]! ∧
      ∀ s ∈ reads[k]!, ∀ q ∈ s.ps, ∃ w, q.tgt = .gate w := by
  unfold readsOf at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at h
  obtain ⟨r, hr, h⟩ := Walk.except_bind_ok h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  obtain ⟨h1, -, h3⟩ := forIn_push _ (fun k rs => Desc.nodeReads (rootCtx P memo) k = .ok rs ∧
      (rs.any fun x => x.ps.any fun q => match q.tgt with | .gate _ => false | _ => true) = false)
    (fun k acc y hy => by
      obtain ⟨rs, hrs, hy⟩ := Walk.except_bind_ok hy
      by_cases hb : (rs.any fun x => x.ps.any fun q => match q.tgt with | .gate _ => false | _ => true) = true
      · rw [if_pos hb] at hy; exact absurd hy (by intro h; cases h)
      · rw [if_neg hb] at hy
        simp only [pure, Except.pure, Except.ok.injEq] at hy
        exact ⟨rs, ⟨hrs, by simpa using hb⟩, hy.symm⟩) _ #[] r hr
  have hn : ((P.defs[P.root]!).body.nodes.size - 0 + 1 - 1) / 1 = nodeCount P := by
    simp [nodeCount, DescAgree.rootBody]
  rw [List.length_range', hn] at h1
  refine ⟨by simpa using h1, fun k hk => ?_⟩
  have := h3 k (by rw [List.length_range', hn]; exact hk)
  simp only [List.getElem_range', Array.size_empty, Nat.zero_add, Nat.one_mul] at this
  exact ⟨this.1, pieces_gates _ this.2⟩


/-- **`Desc.nodeOf` finds the root node holding a gate**, over offsets laid out as `rootLaidOut` checks
(`Walk.bisect_spec`). -/
theorem nodeOf_eq (offs : Array Nat) (K g k : Nat) (hsz : offs.size = K + 1) (h0 : offs[0]! = 0)
    (hmono : ∀ i < K, offs[i]! ≤ offs[i + 1]!) (hk : k < K) (hg : offs[k]! ≤ g ∧ g < offs[k + 1]!) :
    Desc.nodeOf offs g = k := by
  let f : Nat → Nat × Nat → Id (ForInStep (Nat × Nat)) := fun _ (st : Nat × Nat) =>
    have lo := st.1; have hi := st.2
    if hi - lo ≤ 1 then pure (ForInStep.done (lo, hi))
    else
      have mid := (lo + hi) / 2
      if offs[mid]! ≤ g then pure (ForInStep.yield (mid, hi)) else pure (ForInStep.yield (lo, mid))
  obtain ⟨b1, b2, b3, b4, b5⟩ := Walk.bisect_spec offs g (offs.size - 1) f (fun _ _ _ => rfl)
    (List.range' 0 offs.size 1) 0 (offs.size - 1) (by omega) (le_refl _) (by rw [h0]; omega)
    (fun h => absurd h (lt_irrefl _)) (by simp; omega)
  have hr : Desc.nodeOf offs g = (forIn (List.range' 0 offs.size 1) (0, offs.size - 1) f : Id (Nat × Nat)).1 := by
    unfold Desc.nodeOf
    have hfor : ∀ (g : Nat → Nat × Nat → Id (ForInStep (Nat × Nat))) (init : Nat × Nat),
        (forIn [:offs.size] init g : Id (Nat × Nat)) = forIn (List.range' 0 offs.size) init g := by
      intro g init
      rw [Std.Legacy.Range.forIn_eq_forIn_range']
      simp [Std.Legacy.Range.size]
    simp only [hfor]
    rfl
  generalize (forIn (List.range' 0 offs.size 1) (0, offs.size - 1) f : Id (Nat × Nat)) = r at hr b1 b2 b3 b4 b5
  rw [hsz, Nat.add_sub_cancel] at b2 b4
  have hr2 : r.2 = r.1 + 1 := by omega
  have hlt : g < offs[r.1 + 1]! := by
    rcases Nat.lt_or_ge r.2 K with h | h
    · rw [← hr2]; exact b4 h
    · have : r.1 + 1 = K := by omega
      rw [this]; exact Nat.lt_of_lt_of_le hg.2 (DescAgree.offs_mono offs K hmono (k + 1) K (by omega) (le_refl _))
  rw [hr]
  exact DescAgree.node_unique offs K hmono (by omega) hk ⟨b3, hlt⟩ hg

/-- The root node `Desc.nodeOf` finds for a gate of root node `k` is `k`. -/
theorem nodeOf_root (P : Prog) (hl : Desc.rootLaidOut P = true) {k a : Nat} (hk : k < nodeCount P)
    (ha : a ∈ nodeGates P k) : Desc.nodeOf (rootOffs P) a = k := by
  obtain ⟨hsz, h0, -, hmono⟩ := DescAgree.rootLaidOut_spec P hl
  exact nodeOf_eq (rootOffs P) (nodeCount P) a k hsz h0 hmono hk ((DescAgree.mem_nodeGates P k a).1 ha)


/-- **Open: `nodeReads`' segments at the root, as the positions they name.** On a Program `Desc.evaluate` reads, with
its summaries `memo` and `circuitOf`'s circuit `c`: for root node `k`, whose segments `rs` (`Desc.nodeReads`) have
pieces naming gates only, every segment of positive count has pieces of positive count and names gates `a` of width
`c.w[a]` of one root node; and the gates the segments of positive count name are exactly the gates outside node `k`
that its gates read. `readsExact_of_rootReads` gives `Novel.ReadsExact` from it. -/
def RootReads : Prop :=
  ∀ (bytes : ByteArray) (P : Prog) (memo : Array Desc.Summary) (c : Circuit),
    Flock.Program.decode bytes = .ok P → P.orderViolation = none → Desc.rootLaidOut P = true →
    Desc.inputsAtRoot P = true → Desc.walkable P = true → memoOf P = .ok memo → circuitOf P = .ok c →
    ∀ k < nodeCount P, ∀ rs, Desc.nodeReads (rootCtx P memo) k = .ok rs →
      (∀ s ∈ rs, ∀ q ∈ s.ps, ∃ w, q.tgt = .gate w) →
      (∀ s ∈ rs, s.count ≠ 0 → (∀ q ∈ s.ps.toList, 0 < q.count) ∧
        ∃ k' < nodeCount P, ∀ x ∈ segPos s, ∃ a : Nat, x = (.gate c.w[a]!, (a : Int)) ∧ a ∈ nodeGates P k') ∧
      ∀ a : Nat, (∃ s ∈ rs, s.count ≠ 0 ∧ ∃ w, (Tgt.gate w, (a : Int)) ∈ segPos s) ↔
        ∃ g ∈ nodeGates P k, a ∈ c.reads[g]! ∧ a ∉ nodeGates P k

theorem reps_ne_zero {s : Seg} (h : s.count ≠ 0) : s.reps ≠ 0 := by
  intro h0; apply h; simp [Desc.Seg.count, h0]

theorem rows_ok {s : Seg} (h : ∀ q ∈ s.ps.toList, 0 < q.count) : ∀ q ∈ s.ps.toList, q.rows ≠ 0 ∨ q.len = 0 := by
  intro q hq
  have := h q hq
  left; intro h0; simp [Desc.Piece.count, h0] at this

/-- A segment of a copy names its first piece's first position. -/
theorem first_mem_segPos {s : Seg} {q : Piece} (hr : s.reps ≠ 0) (hq : s.ps[0]? = some q) (hc : 0 < q.count) :
    (q.tgt, q.base) ∈ segPos s := by
  unfold segPos
  refine List.mem_flatMap.2 ⟨0, List.mem_range.2 (by omega), ?_⟩
  rw [copy_zero]
  refine List.mem_flatMap.2 ⟨q, ?_, ?_⟩
  · obtain ⟨h0, rfl⟩ := Array.getElem?_eq_some_iff.1 hq
    exact Array.getElem_mem_toList h0
  · rw [pos_eq_map]
    refine List.mem_map.2 ⟨0, List.mem_range.2 hc, ?_⟩
    simp [ppt]

theorem gatePt_eq {y : Tgt × Int} {a : Int} {w : Nat} (h : gatePt y = some (a, w)) : y = (Tgt.gate w, a) := by
  rcases y with ⟨t, i⟩
  rcases t with ⟨p, i'⟩ | w'
  · simp [gatePt] at h
  · simp only [gatePt, Option.some.injEq, Prod.mk.injEq] at h
    rw [h.1, h.2]

/-- **`ReadsExact` from `RootReads`**: `analyze`'s reads are `nodeReads`' (`readsOf_spec`), their gates the gate
positions they name (`segGates_pos`), and a segment's first piece names a gate of the one root node its gates are in,
which `Desc.nodeOf` finds (`nodeOf_root`). -/
theorem readsExact_of_rootReads (h : RootReads) : Novel.ReadsExact := by
  intro bytes P reads c hd ho hl hi hw ha hc k hk
  rw [analyze_eq] at ha
  obtain ⟨memo, hm, hr⟩ := Walk.except_bind_ok ha
  obtain ⟨-, hks⟩ := readsOf_spec P memo reads hr
  obtain ⟨hnr, hg⟩ := hks k hk
  obtain ⟨h1, h2⟩ := h bytes P memo c hd ho hl hi hw hm hc k hk _ hnr hg
  refine ⟨fun s hs hcnt q hq => ?_, fun a => ?_⟩
  · obtain ⟨hpos, k', hk', hx⟩ := h1 s hs hcnt
    have hq0 := first_mem_segPos (reps_ne_zero hcnt) hq
      (hpos q (by obtain ⟨h0, rfl⟩ := Array.getElem?_eq_some_iff.1 hq; exact Array.getElem_mem_toList h0))
    obtain ⟨a, ha, hak⟩ := hx _ hq0
    have hb : q.base.toNat = a := by
      simp only [Prod.mk.injEq] at ha
      rw [ha.2]; simp
    rw [hb, nodeOf_root P hl hk' hak]
    refine ⟨hk', fun x hxs => ?_⟩
    rw [segGates_pos s (reps_ne_zero hcnt) (rows_ok hpos)] at hxs
    obtain ⟨y, hy, hgy⟩ := List.mem_filterMap.1 hxs
    obtain ⟨b, rfl, hb'⟩ := hx y hy
    simp only [gatePt, Option.some.injEq] at hgy
    exact ⟨b, hgy.symm, hb'⟩
  · rw [← h2 a]
    constructor
    · rintro ⟨s, hs, hcnt, w, hw⟩
      rw [segGates_pos s (reps_ne_zero hcnt) (rows_ok (h1 s hs hcnt).1)] at hw
      obtain ⟨y, hy, hgy⟩ := List.mem_filterMap.1 hw
      exact ⟨s, hs, hcnt, w, by rw [← gatePt_eq hgy]; exact hy⟩
    · rintro ⟨s, hs, hcnt, w, hw⟩
      refine ⟨s, hs, hcnt, w, ?_⟩
      rw [segGates_pos s (reps_ne_zero hcnt) (rows_ok (h1 s hs hcnt).1)]
      exact List.mem_filterMap.2 ⟨_, hw, rfl⟩


/-! ## Building blocks for `RootReads` -/

/-- The context `Desc.summarize` and `Desc.analyze` read Definition `d`'s body in. -/
def ctxOf (P : Prog) (memo : Array Desc.Summary) (d : Nat) : Desc.Ctx :=
  { P, memo, d, args := (P.defs[d]!).body.nodes.map fun n => n.args.map (Desc.ofSeq ·.seq) }

theorem rootCtx_eq (P : Prog) (memo : Array Desc.Summary) : rootCtx P memo = ctxOf P memo P.root := rfl

/-- **Argument `i` of node `k`, as the context holds it**, names the argument's references in order, when its sequence
is its parts' and none of them names a negative position (`ofSeq_pos`, `partsPos_get`). -/
theorem ctxOf_arg (P : Prog) (memo : Array Desc.Summary) (d k i : Nat)
    (hk : k < (P.defs[d]!).body.nodes.size) (hi : i < ((P.defs[d]!).body.nodes[k]!).args.size)
    (hs : (((P.defs[d]!).body.nodes[k]!).args[i]!).seq =
      Seq.ofParts (((P.defs[d]!).body.nodes[k]!).args[i]!).seq.parts)
    (hn : ∀ q ∈ (((P.defs[d]!).body.nodes[k]!).args[i]!).seq.parts.toList, PartNonneg q) :
    PSOk ((ctxOf P memo d).arg k i) ∧
      psPos ((ctxOf P memo d).arg k i) =
        (List.range (((P.defs[d]!).body.nodes[k]!).args[i]!).seq.count).map
          fun t => refPos ((((P.defs[d]!).body.nodes[k]!).args[i]!).seq.get t) := by
  have harg : (ctxOf P memo d).arg k i = Desc.ofSeq (((P.defs[d]!).body.nodes[k]!).args[i]!).seq := by
    unfold Desc.Ctx.arg ctxOf
    dsimp only
    generalize (P.defs[d]!).body.nodes = ns at hk hi ⊢
    rw [getElem!_pos ns k hk] at hi ⊢
    rw [Array.getElem?_map, Array.getElem?_eq_getElem hk, Option.map_some, Option.bind_some, Array.getElem?_map,
      Array.getElem?_eq_getElem hi, Option.map_some, Option.getD_some, getElem!_pos _ i hi]
  obtain ⟨h1, h2, -⟩ := ofSeq_pos _ hn
  rw [harg]
  refine ⟨h1, ?_⟩
  rw [h2]
  generalize (((P.defs[d]!).body.nodes[k]!).args[i]!).seq = s at hs ⊢
  rw [hs, partsPos_get, (Walk.ofParts_spec _).1]

/-- **A loop that appends each element's segments** names, in order, each element's positions, when each element's
segments name `D` of it. -/
theorem forIn_append_pos {β : Type} (F : β → Except String (Array Seg)) (D : β → List (Tgt × Int))
    (f : β → Array Seg → Except String (ForInStep (Array Seg)))
    (hf : ∀ x r y, f x r = .ok y → ∃ segs, F x = .ok segs ∧ y = .yield (r ++ segs))
    (hF : ∀ x segs, F x = .ok segs → segsPos segs = D x) :
    ∀ (l : List β) (acc out : Array Seg), forIn l acc f = .ok out → segsPos out = segsPos acc ++ l.flatMap D
  | [], acc, out, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp
  | x :: l, acc, out, h => by
    rw [List.forIn_cons] at h
    rcases hx : f x acc with e | y <;> rw [hx] at h
    · exact absurd h (by intro h; cases h)
    obtain ⟨segs, hs, rfl⟩ := hf x acc _ hx
    rw [forIn_append_pos F D f hf hF l (acc ++ segs) out h, segsPos_append, hF x segs hs, List.flatMap_cons,
      List.append_assoc]


/-! ## What `Desc.nodeReads` gives, block by block

`nodeReads` reads each argument of node `k` over spans of its positions, piece by piece through `down`, and a scan's later
iterations' carries through `fromCallee`. Its segments' positions are those reads' positions laid end to end
(`nodeReads_prim`, `nodeReads_comp`), a composite's kept where `keep` keeps their segments. -/

/-- The positions of `down`'s segments for a piece, where it gives them. -/
def downPos (c : Desc.Ctx) (q : Piece) : List (Tgt × Int) :=
  match Desc.down c Flock.Steps.FUEL q with
  | .ok segs => segsPos segs
  | .error _ => []

/-- The positions of `fromCallee`'s segments, where it gives them. -/
def calleePos (c : Desc.Ctx) (k m a b : Nat) : List (Tgt × Int) :=
  match Desc.fromCallee c Flock.Steps.FUEL k m a b with
  | .ok segs => segsPos segs
  | .error _ => []

/-- The positions `nodeReads` reads argument `i` of node `k` at over `spans`. -/
def readPos (c : Desc.Ctx) (k i : Nat) (spans : Array (Nat × Nat)) : List (Tgt × Int) :=
  spans.toList.flatMap fun x => ((c.arg k i).pieces x.1 x.2).toList.flatMap (downPos c)

/-- A loop that adds `D x`'s positions to its segments' for each element `x`, each segment it adds one `Q x` holds of. -/
theorem forIn_pm {β : Type} (D : β → List (Tgt × Int)) (Q : β → Seg → Prop)
    (f : β → Array Seg → Except String (ForInStep (Array Seg)))
    (hf : ∀ x r y, f x r = .ok y → ∃ r', y = .yield r' ∧ segsPos r' = segsPos r ++ D x ∧ ∀ s ∈ r', s ∈ r ∨ Q x s) :
    ∀ (l : List β) (acc out : Array Seg), forIn l acc f = .ok out →
      segsPos out = segsPos acc ++ l.flatMap D ∧ ∀ s ∈ out, s ∈ acc ∨ ∃ x ∈ l, Q x s
  | [], acc, out, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨by simp, fun s hs => Or.inl hs⟩
  | x :: l, acc, out, h => by
    rw [List.forIn_cons] at h
    rcases hx : f x acc with e | y <;> rw [hx] at h
    · exact absurd h (by intro h; cases h)
    obtain ⟨r', rfl, hr', hm⟩ := hf x acc _ hx
    obtain ⟨h1, h2⟩ := forIn_pm D Q f hf l r' out h
    refine ⟨by rw [h1, hr', List.flatMap_cons, List.append_assoc], fun s hs => ?_⟩
    rcases h2 s hs with h | ⟨x', hx', hq⟩
    · rcases hm s h with h | h
      · exact Or.inl h
      · exact Or.inr ⟨x, List.mem_cons_self, h⟩
    · exact Or.inr ⟨x', List.mem_cons_of_mem _ hx', hq⟩

/-- A segment `down` gives a piece of argument `i` of node `k`. -/
def FromDown (c : Desc.Ctx) (k i : Nat) (s : Seg) : Prop :=
  ∃ lo hi, ∃ q ∈ (c.arg k i).pieces lo hi, ∃ segs, Desc.down c Flock.Steps.FUEL q = .ok segs ∧ s ∈ segs

/-- A segment `down` gives a piece of argument `i` of node `k` that `Desc.memberRows` gives, all members at once. -/
def FromRows (c : Desc.Ctx) (k : Nat) (node : Flock.Program.Node) (s : Seg) : Prop :=
  ∃ i u ps, Desc.memberRows c node k i u = .ok (some ps) ∧ ∃ p ∈ ps, ∃ segs,
    Desc.down c Flock.Steps.FUEL p = .ok segs ∧ s ∈ segs

/-- A segment `Desc.carriesRead` gives for a scan's carries into its iterations after the first. -/
def FromCarry (c : Desc.Ctx) (k : Nat) (node : Flock.Program.Node) (s : Seg) : Prop :=
  ∃ segs, Desc.carriesRead c k ((c.memo[node.fn]!).used[0]?.getD #[]) = .ok segs ∧ s ∈ segs

theorem mem_append_or {s : Seg} {r lift : Array Seg} (h : s ∈ r ++ lift) : s ∈ r ∨ s ∈ lift := Array.mem_append.1 h

theorem walkAtMost_bind {α : Type} (what : String) (n : Nat) (f : Unit → Except String α) {x : α}
    (h : (Desc.walkAtMost what n >>= f) = .ok x) : f () = .ok x := by
  unfold Desc.walkAtMost at h
  by_cases hn : n > Desc.WALK
  · rw [if_pos hn] at h; cases h
  · rw [if_neg hn] at h; exact h

theorem ok_bind_pure {α : Type} {x : Except String α} {y : α} (h : (x >>= fun r => pure r) = Except.ok y) :
    x = .ok y := by
  obtain ⟨r, hr, h⟩ := Walk.except_bind_ok h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  rw [hr, h]

/-- `nodeReads`' read of argument `i` over `spans`, from the segments `acc`. -/
theorem readLoop_pos (c : Desc.Ctx) (k i : Nat) (spans : Array (Nat × Nat)) (acc out : Array Seg)
    (h : forIn spans acc (fun (x : Nat × Nat) __s => do
        let __s ← forIn ((c.arg k i).pieces x.1 x.2) __s fun q __s => do
          let __do_lift ← Desc.down c Flock.Steps.FUEL q
          pure (ForInStep.yield (__s ++ __do_lift))
        pure (ForInStep.yield __s)) = .ok out) :
    segsPos out = segsPos acc ++ readPos c k i spans ∧ ∀ s ∈ out, s ∈ acc ∨ FromDown c k i s := by
  rw [← Array.forIn_toList] at h
  obtain ⟨h1, h2⟩ := forIn_pm (fun x => ((c.arg k i).pieces x.1 x.2).toList.flatMap (downPos c))
    (fun x s => ∃ q ∈ (c.arg k i).pieces x.1 x.2, ∃ segs, Desc.down c Flock.Steps.FUEL q = .ok segs ∧ s ∈ segs) _
    (fun x r y hy => by
      obtain ⟨r', hr', hy⟩ := Walk.except_bind_ok hy
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      refine ⟨r', hy.symm, ?_⟩
      rw [← Array.forIn_toList] at hr'
      obtain ⟨g1, g2⟩ := forIn_pm (downPos c)
        (fun q s => ∃ segs, Desc.down c Flock.Steps.FUEL q = .ok segs ∧ s ∈ segs) _ (fun q r y hy => by
          obtain ⟨segs, hs, hy⟩ := Walk.except_bind_ok hy
          simp only [pure, Except.pure, Except.ok.injEq] at hy
          refine ⟨r ++ segs, hy.symm, by rw [segsPos_append, downPos, hs], fun s h => ?_⟩
          rcases mem_append_or h with h | h
          · exact Or.inl h
          · exact Or.inr ⟨segs, hs, h⟩) _ r r' hr'
      refine ⟨g1, fun s h => ?_⟩
      rcases g2 s h with h | ⟨q, hq, h⟩
      · exact Or.inl h
      · exact Or.inr ⟨q, Array.mem_toList_iff.1 hq, h⟩) _ acc out h
  refine ⟨h1, fun s hs => ?_⟩
  rcases h2 s hs with h | ⟨x, -, hx⟩
  · exact Or.inl h
  · exact Or.inr ⟨x.1, x.2, hx⟩

/-- A read from no segments. -/
theorem readBlock_pos (c : Desc.Ctx) (k i : Nat) (spans : Array (Nat × Nat)) (out : Array Seg)
    (h : ((forIn spans #[] fun (x : Nat × Nat) __s => do
          let __s ← forIn ((c.arg k i).pieces x.1 x.2) __s fun q __s => do
            let __do_lift ← Desc.down c Flock.Steps.FUEL q
            pure (ForInStep.yield (__s ++ __do_lift))
          pure (ForInStep.yield __s)) >>= fun __s => pure __s) = .ok out) :
    segsPos out = readPos c k i spans ∧ ∀ s ∈ out, FromDown c k i s := by
  obtain ⟨h1, h2⟩ := readLoop_pos c k i spans #[] out (ok_bind_pure h)
  refine ⟨by rw [h1, segsPos_empty, List.nil_append], fun s hs => ?_⟩
  rcases h2 s hs with h | h
  · simp at h
  · exact h

theorem range'_size (n : Nat) : List.range' 0 ((n - 0 + 1 - 1) / 1) = List.range n := by
  rw [List.range_eq_range']; congr 1; omega

/-- **A primitive's node reads its arguments whole**, unless it is a batch of no members, each segment from `down`;
copies alike are one (`Seg.alike`). -/
theorem nodeReads_prim (c : Desc.Ctx) (k : Nat) (node : Flock.Program.Node) (hn : c.node k = .ok node)
    (hp : (c.P.defs[node.fn]!).prim = true) (rs : Array Seg) (h : Desc.nodeReads c k = .ok rs) :
    ∃ out : Array Seg, rs = out.map Seg.alike ∧
      segsPos out = (if (node.form == .batch && node.n == 0) = true then [] else
        (List.range node.args.size).flatMap fun i => readPos c k i #[(0, (c.arg k i).count)]) ∧
      ∀ s ∈ out, ∃ i, FromDown c k i s := by
  unfold Desc.nodeReads at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at h
  obtain ⟨node', hn', h⟩ := Walk.except_bind_ok h
  rw [hn] at hn'
  injection hn' with e
  subst e
  rw [if_pos hp] at h
  split at h
  · rename_i hb
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨#[], by simp, by rw [if_pos hb, segsPos_empty], fun s hs => by simp at hs⟩
  · rename_i hb
    obtain ⟨out, hout, h⟩ := Walk.except_bind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    obtain ⟨h1, h2⟩ := forIn_pm _ (fun i s => FromDown c k i s) _ (fun i r y hy => by
      obtain ⟨lift, hl, hy⟩ := Walk.except_bind_ok hy
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      obtain ⟨b1, b2⟩ := readBlock_pos c k i _ lift hl
      refine ⟨r ++ lift, hy.symm, by rw [segsPos_append, b1], fun s h => ?_⟩
      rcases mem_append_or h with h | h
      · exact Or.inl h
      · exact Or.inr (b2 s h)) _ #[] out hout
    refine ⟨out, h.symm, by rw [if_neg hb, ← range'_size, h1, segsPos_empty, List.nil_append], fun s hs => ?_⟩
    rcases h2 s hs with h | ⟨i, -, h⟩
    · simp at h
    · exact ⟨i, h⟩

/-- The segments `nodeReads` keeps of a composite's node `k`: those not starting at a gate of the node. -/
def keep (c : Desc.Ctx) (k : Nat) (s : Seg) : Bool :=
  match s.ps[0]? with
  | some q => match q.tgt with
    | .gate _ => !(decide ((c.body.offsets[k]! : Int) ≤ q.base) && decide (q.base < (c.body.offsets[k + 1]! : Int)))
    | _ => true
  | none => false

/-- The positions `nodeReads` reads argument `i` of a composite's node `k` at: those its callee's `used` names, all
members at once (`Desc.memberRows`) or member by member where the argument is mapped and not read whole, and a scan's
initial carry. -/
def argPos (c : Desc.Ctx) (k : Nat) (node : Flock.Program.Node) (i : Nat) : List (Tgt × Int) :=
  if (node.form == .scan && i == 0) = true then readPos c k 0 ((c.memo[node.fn]!).used[i]?.getD #[])
  else if (!match node.form with
      | .batch => node.args[i]!.dims.isSome
      | .scan => decide (1 ≤ i) && decide (i ≤ node.nxs)
      | _ => false) = true then readPos c k i ((c.memo[node.fn]!).used[i]?.getD #[])
  else if ((c.memo[node.fn]!).used[i]?.getD #[]).covers
      (((c.P.defs[node.fn]!).params[i]?.map (·.2.leaves)).getD 0) = true then
    readPos c k i #[(0, (c.arg k i).count)]
  else match Desc.memberRows c node k i ((c.memo[node.fn]!).used[i]?.getD #[]) with
    | .ok (some ps) => ps.toList.flatMap (downPos c)
    | _ => (List.range node.members).flatMap fun m =>
      readPos c k i (Desc.argSpans node i m ((c.memo[node.fn]!).used[i]?.getD #[]))

/-- The positions a scan's iterations after the first read their carries at (`Desc.carriesRead`). -/
def carryPos (c : Desc.Ctx) (k : Nat) (node : Flock.Program.Node) : List (Tgt × Int) :=
  if (node.form == .scan && node.n ≥ 2) = true then
    match Desc.carriesRead c k ((c.memo[node.fn]!).used[0]?.getD #[]) with
    | .ok segs => segsPos segs
    | .error _ => []
  else []

/-- **A composite's node reads its arguments where its callee reads its parameters**, and a scan's carries, keeping the
segments that don't start in the node, copies alike as one (`Seg.alike`); each segment from `down` or `carriesRead`. -/
theorem nodeReads_comp (c : Desc.Ctx) (k : Nat) (node : Flock.Program.Node) (hn : c.node k = .ok node)
    (hp : (c.P.defs[node.fn]!).prim = false) (rs : Array Seg) (h : Desc.nodeReads c k = .ok rs) :
    if node.members = 0 then rs = #[] else
      ∃ out : Array Seg, rs = (out.filter (keep c k)).map Seg.alike ∧
        segsPos out = (List.range node.args.size).flatMap (argPos c k node) ++ carryPos c k node ∧
        ∀ s ∈ out, (∃ i, FromDown c k i s) ∨ FromRows c k node s ∨ FromCarry c k node s := by
  unfold Desc.nodeReads at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at h
  obtain ⟨node', hn', h⟩ := Walk.except_bind_ok h
  rw [hn] at hn'
  injection hn' with e
  subst e
  rw [if_neg (by rw [hp]; decide)] at h
  split at h
  · rename_i hm
    simp only [pure, Except.pure, Except.ok.injEq] at h
    rw [if_pos (by simpa using hm)]
    exact h.symm
  · rename_i hm
    rw [if_neg (by simpa using hm)]
    obtain ⟨r1, h1, h⟩ := Walk.except_bind_ok h
    obtain ⟨hr1, hm1⟩ := forIn_pm (argPos c k node) (fun _ s => (∃ i, FromDown c k i s) ∨ FromRows c k node s) _
      (fun i r y hy => by
      rw [argPos]
      rcases hform : node.form with _ | _ | _ | _ <;> simp only [hform] at hy ⊢
      all_goals
        split at hy
        · rename_i hs
          obtain ⟨lift, hl, hy⟩ := Walk.except_bind_ok hy
          simp only [pure, Except.pure, Except.ok.injEq] at hy
          obtain ⟨b1, b2⟩ := readBlock_pos c k 0 _ lift hl
          refine ⟨r ++ lift, hy.symm, by rw [segsPos_append, b1, if_pos hs], fun s h => ?_⟩
          rcases mem_append_or h with h | h
          · exact Or.inl h
          · exact Or.inr (Or.inl ⟨0, b2 s h⟩)
        · rename_i hs
          split at hy
          · rename_i hpm
            obtain ⟨lift, hl, hy⟩ := Walk.except_bind_ok hy
            simp only [pure, Except.pure, Except.ok.injEq] at hy
            obtain ⟨b1, b2⟩ := readBlock_pos c k i _ lift hl
            refine ⟨r ++ lift, hy.symm, by rw [segsPos_append, b1, if_neg hs, if_pos hpm], fun s h => ?_⟩
            rcases mem_append_or h with h | h
            · exact Or.inl h
            · exact Or.inr (Or.inl ⟨i, b2 s h⟩)
          · rename_i hpm
            split at hy
            · rename_i hcv
              obtain ⟨lift, hl, hy⟩ := Walk.except_bind_ok hy
              simp only [pure, Except.pure, Except.ok.injEq] at hy
              obtain ⟨b1, b2⟩ := readBlock_pos c k i _ lift hl
              refine ⟨r ++ lift, hy.symm, by rw [segsPos_append, b1, if_neg hs, if_neg hpm, if_pos hcv],
                fun s h => ?_⟩
              rcases mem_append_or h with h | h
              · exact Or.inl h
              · exact Or.inr (Or.inl ⟨i, b2 s h⟩)
            · rename_i hcv
              obtain ⟨mr, hmr, hy⟩ := Walk.except_bind_ok hy
              rcases mr with _ | ps
              · have hy := walkAtMost_bind _ _ _ hy
                obtain ⟨r', hr', hy⟩ := Walk.except_bind_ok hy
                simp only [pure, Except.pure, Except.ok.injEq] at hy
                obtain ⟨g1, g2⟩ := forIn_pm _ (fun _ s => FromDown c k i s) _ (fun m r y hy => by
                  obtain ⟨lift, hl, hy⟩ := Walk.except_bind_ok hy
                  simp only [pure, Except.pure, Except.ok.injEq] at hy
                  obtain ⟨b1, b2⟩ := readBlock_pos c k i _ lift hl
                  refine ⟨r ++ lift, hy.symm, by rw [segsPos_append, b1], fun s h => ?_⟩
                  rcases mem_append_or h with h | h
                  · exact Or.inl h
                  · exact Or.inr (b2 s h)) _ r r' hr'
                refine ⟨r', hy.symm, by rw [g1, if_neg hs, if_neg hpm, if_neg hcv, hmr, range'_size],
                  fun s h => ?_⟩
                rcases g2 s h with h | ⟨m, -, h⟩
                · exact Or.inl h
                · exact Or.inr (Or.inl ⟨i, h⟩)
              · obtain ⟨r', hr', hy⟩ := Walk.except_bind_ok hy
                simp only [pure, Except.pure, Except.ok.injEq] at hy
                rw [← Array.forIn_toList] at hr'
                obtain ⟨g1, g2⟩ := forIn_pm (downPos c)
                  (fun p s => ∃ segs, Desc.down c Flock.Steps.FUEL p = .ok segs ∧ s ∈ segs) _ (fun p r y hy => by
                    obtain ⟨segs, hs, hy⟩ := Walk.except_bind_ok hy
                    simp only [pure, Except.pure, Except.ok.injEq] at hy
                    refine ⟨r ++ segs, hy.symm, by rw [segsPos_append, downPos, hs], fun s h => ?_⟩
                    rcases mem_append_or h with h | h
                    · exact Or.inl h
                    · exact Or.inr ⟨segs, hs, h⟩) _ r r' hr'
                refine ⟨r', hy.symm, by rw [g1, if_neg hs, if_neg hpm, if_neg hcv, hmr], fun s h => ?_⟩
                rcases g2 s h with h | ⟨p, hp', segs, hs', h⟩
                · exact Or.inl h
                · exact Or.inr (Or.inr ⟨i, _, ps, hmr, p, Array.mem_toList_iff.1 hp', segs, hs', h⟩)) _ #[] r1 h1
    rw [segsPos_empty, List.nil_append, range'_size] at hr1
    have hm1' : ∀ s ∈ r1, (∃ i, FromDown c k i s) ∨ FromRows c k node s := fun s hs => by
      rcases hm1 s hs with h | ⟨i, -, h⟩
      · simp at h
      · exact h
    split at h
    · rename_i hsc
      obtain ⟨cr, hcr, h⟩ := Walk.except_bind_ok h
      simp only [pure, Except.pure, Except.ok.injEq] at h
      refine ⟨r1 ++ cr, h.symm, by rw [segsPos_append, hr1, carryPos, if_pos hsc, hcr], fun s hs => ?_⟩
      rcases mem_append_or hs with hs | hs
      · rcases hm1' s hs with h | h
        · exact Or.inl h
        · exact Or.inr (Or.inl h)
      · exact Or.inr (Or.inr ⟨cr, hcr, hs⟩)
    · rename_i hsc
      simp only [pure, Except.pure, Except.ok.injEq] at h
      refine ⟨r1, h.symm, by rw [hr1, carryPos, if_neg hsc, List.append_nil], fun s hs => ?_⟩
      rcases hm1' s hs with h | h
      · exact Or.inl h
      · exact Or.inr (Or.inl h)


/-! ## `RootReads` from the segments `down` gives at the root and the positions read

`nodeReads` gives each root node's segments from `down` and a scan's carries (`nodeReads_prim`, `nodeReads_comp`), so
`RootReads` holds wherever two smaller statements do (`rootReads_of`): `RootSegs`, that each such segment names gates of
one root node with their widths, and `RootSet`, that the positions read name exactly the gates outside the node that
its gates read. A composite's segment `keep` keeps is then one outside the node (`keep_iff`). -/

/-- A segment as `RootReads` speaks of it: of no count, it names no positions; else its pieces have positive count and
it names gates `a` of one root node, each of width `c.w[a]`. -/
def SegGood (P : Prog) (c : Circuit) (s : Seg) : Prop :=
  (s.count = 0 → segPos s = []) ∧
    (s.count ≠ 0 → (∀ q ∈ s.ps.toList, 0 < q.count) ∧
      ∃ k' < nodeCount P, ∀ x ∈ segPos s, ∃ a : Nat, x = (.gate c.w[a]!, (a : Int)) ∧ a ∈ nodeGates P k')

/-- The positions root node `k` reads (`nodeReads_prim`, `nodeReads_comp`), a composite's before `keep`. -/
def rootRead (P : Prog) (memo : Array Desc.Summary) (node : Flock.Program.Node) (k : Nat) : List (Tgt × Int) :=
  if (P.defs[node.fn]!).prim = true then
    (if (node.form == .batch && node.n == 0) = true then [] else
      (List.range node.args.size).flatMap fun i =>
        readPos (rootCtx P memo) k i #[(0, ((rootCtx P memo).arg k i).count)])
  else if node.members = 0 then []
  else (List.range node.args.size).flatMap (argPos (rootCtx P memo) k node) ++ carryPos (rootCtx P memo) k node

/-- **Open: the segments `down` gives at the root.** On a Program `Desc.evaluate` reads, with its summaries `memo` and
`circuitOf`'s circuit `c`: for root node `k`, whose segments (`Desc.nodeReads`) have pieces naming gates only, every
segment `down` gives a piece of one of its arguments (or one `Desc.memberRows` gives), or `Desc.carriesRead` gives of
its scan's carries, is `SegGood`. -/
def RootSegs : Prop :=
  ∀ (bytes : ByteArray) (P : Prog) (memo : Array Desc.Summary) (c : Circuit),
    Flock.Program.decode bytes = .ok P → P.orderViolation = none → Desc.rootLaidOut P = true →
    Desc.inputsAtRoot P = true → Desc.walkable P = true → memoOf P = .ok memo → circuitOf P = .ok c →
    ∀ k < nodeCount P, ∀ node rs, (rootCtx P memo).node k = .ok node → Desc.nodeReads (rootCtx P memo) k = .ok rs →
      (∀ s ∈ rs, ∀ q ∈ s.ps, ∃ w, q.tgt = .gate w) →
      ∀ s, ((∃ i, FromDown (rootCtx P memo) k i s) ∨ FromRows (rootCtx P memo) k node s ∨
        FromCarry (rootCtx P memo) k node s) → SegGood P c s

/-- **Open: the positions a root node reads.** With `RootSegs`' premises: the gates the positions root node `k` reads
name (`rootRead`), outside the node where its callee is a composite, are exactly the gates outside the node that its
gates read. -/
def RootSet : Prop :=
  ∀ (bytes : ByteArray) (P : Prog) (memo : Array Desc.Summary) (c : Circuit),
    Flock.Program.decode bytes = .ok P → P.orderViolation = none → Desc.rootLaidOut P = true →
    Desc.inputsAtRoot P = true → Desc.walkable P = true → memoOf P = .ok memo → circuitOf P = .ok c →
    ∀ k < nodeCount P, ∀ node rs, (rootCtx P memo).node k = .ok node → Desc.nodeReads (rootCtx P memo) k = .ok rs →
      (∀ s ∈ rs, ∀ q ∈ s.ps, ∃ w, q.tgt = .gate w) →
      ∀ a : Nat, ((∃ w, (Tgt.gate w, (a : Int)) ∈ rootRead P memo node k) ∧
          ((P.defs[node.fn]!).prim = false → a ∉ nodeGates P k)) ↔
        ∃ g ∈ nodeGates P k, a ∈ c.reads[g]! ∧ a ∉ nodeGates P k

theorem node_of_reads (c : Desc.Ctx) (k : Nat) (rs : Array Seg) (h : Desc.nodeReads c k = .ok rs) :
    ∃ node, c.node k = .ok node := by
  unfold Desc.nodeReads at h
  obtain ⟨node, hn, -⟩ := Walk.except_bind_ok h
  exact ⟨node, hn⟩

/-- A segment naming a position has a first piece. -/
theorem first_piece {s : Seg} {x : Tgt × Int} (h : x ∈ segPos s) : ∃ q, s.ps[0]? = some q := by
  rcases hq : s.ps[0]? with _ | q
  · exfalso
    have he : s.ps = #[] := Array.eq_empty_of_size_eq_zero (by have := Array.getElem?_eq_none_iff.1 hq; omega)
    unfold segPos at h
    obtain ⟨m, -, hm⟩ := List.mem_flatMap.1 h
    rw [copy_pos, he] at hm
    simp [posAll] at hm
  · exact ⟨q, rfl⟩

/-- `keep` keeps a segment whose first piece names gate `a0` exactly when `a0` is outside the node. -/
theorem keep_gate (P : Prog) (memo : Array Desc.Summary) (k : Nat) (s : Seg) (q : Piece) (hq : s.ps[0]? = some q)
    (w a0 : Nat) (ht : q.tgt = .gate w) (hb : q.base = (a0 : Int)) :
    keep (rootCtx P memo) k s = true ↔ a0 ∉ nodeGates P k := by
  have ho : (rootCtx P memo).body.offsets = rootOffs P := rfl
  unfold keep
  rw [hq]
  simp only [ht, hb, ho, DescAgree.mem_nodeGates, Bool.not_eq_true', Bool.and_eq_false_iff, decide_eq_false_iff_not,
    Int.ofNat_le, Int.ofNat_lt, not_and_or]

/-- **A `SegGood` segment of positive count is kept exactly when a gate it names is outside the node**: its gates are
one root node's, its first piece's among them. -/
theorem keep_iff (P : Prog) (memo : Array Desc.Summary) (c : Circuit) (hl : Desc.rootLaidOut P = true) {k : Nat}
    (hk : k < nodeCount P) {s : Seg} (hg : SegGood P c s) (hc0 : s.count ≠ 0) {w a : Nat}
    (hw : (Tgt.gate w, (a : Int)) ∈ segPos s) : keep (rootCtx P memo) k s = true ↔ a ∉ nodeGates P k := by
  obtain ⟨-, -, -, hmono⟩ := DescAgree.rootLaidOut_spec P hl
  obtain ⟨hpos, k', hk', hx⟩ := hg.2 hc0
  obtain ⟨a', ha', hak'⟩ := hx _ hw
  simp only [Prod.mk.injEq, Nat.cast_inj] at ha'
  rw [← ha'.2] at hak'
  obtain ⟨q, hq⟩ := first_piece hw
  have hq0 := first_mem_segPos (reps_ne_zero hc0) hq
    (hpos q (by obtain ⟨h0, rfl⟩ := Array.getElem?_eq_some_iff.1 hq; exact Array.getElem_mem_toList h0))
  obtain ⟨a0, ha0, ha0k⟩ := hx _ hq0
  simp only [Prod.mk.injEq] at ha0
  rw [keep_gate P memo k s q hq _ a0 ha0.1 ha0.2]
  have hu : ∀ b b', b ∈ nodeGates P k' → b' ∈ nodeGates P k' → b ∈ nodeGates P k → b' ∈ nodeGates P k := by
    intro b b' hb hb' hbk
    have e := DescAgree.node_unique (rootOffs P) (nodeCount P) hmono hk hk' ((DescAgree.mem_nodeGates P k b).1 hbk)
      ((DescAgree.mem_nodeGates P k' b).1 hb)
    subst e
    exact hb'
  exact ⟨fun h ha => h (hu a a0 hak' ha0k ha), fun h ha => h (hu a0 a ha0k hak' ha)⟩

theorem alike_ps (s : Seg) : (Seg.alike s).ps = s.ps := by
  unfold Desc.Seg.alike; split <;> rfl

theorem keep_alike (c : Desc.Ctx) (k : Nat) (s : Seg) : keep c k (Seg.alike s) = keep c k s := by
  unfold keep; rw [alike_ps]

/-- Copies alike as one name what the copies name. -/
theorem mem_segPos_alike (s : Seg) (x : Tgt × Int) : x ∈ segPos (Seg.alike s) ↔ x ∈ segPos s := by
  unfold Desc.Seg.alike
  split
  · rename_i h
    simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at h
    unfold segPos
    simp only [List.mem_flatMap, List.mem_range, copy_pos, h.2, Int.mul_zero, shiftPos_zero, List.map_id]
    exact ⟨fun ⟨_, _, hx⟩ => ⟨0, by omega, hx⟩, fun ⟨_, _, hx⟩ => ⟨0, by omega, hx⟩⟩
  · rfl

theorem alike_count (s : Seg) : (Seg.alike s).count = 0 ↔ s.count = 0 := by
  unfold Desc.Seg.alike Desc.Seg.count
  split
  · rename_i h
    simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at h
    simp only [Nat.mul_eq_zero]
    omega
  · rfl

theorem segGood_alike {P : Prog} {c : Circuit} {s : Seg} (h : SegGood P c s) : SegGood P c (Seg.alike s) := by
  refine ⟨fun h0 => ?_, fun h0 => ?_⟩
  · have e := h.1 ((alike_count s).1 h0)
    refine List.eq_nil_iff_forall_not_mem.2 fun x hx => ?_
    rw [mem_segPos_alike, e] at hx
    exact List.not_mem_nil hx
  · obtain ⟨hp, k', hk', hx⟩ := h.2 fun h' => h0 ((alike_count s).2 h')
    exact ⟨by rw [alike_ps]; exact hp, k', hk', fun x hx' => hx x ((mem_segPos_alike s x).1 hx')⟩

/-- **`RootReads` from `RootSegs` and `RootSet`.** -/
theorem rootReads_of (hS : RootSegs) (hE : RootSet) : RootReads := by
  intro bytes P memo c hd ho hl hi hw hm hc k hk rs hrs hg
  obtain ⟨node, hn⟩ := node_of_reads _ k rs hrs
  have hS' := hS bytes P memo c hd ho hl hi hw hm hc k hk node rs hn hrs hg
  have hE' := hE bytes P memo c hd ho hl hi hw hm hc k hk node rs hn hrs hg
  suffices H : (∀ s ∈ rs, SegGood P c s) ∧ ∀ a : Nat,
      (∃ s ∈ rs, s.count ≠ 0 ∧ ∃ w, (Tgt.gate w, (a : Int)) ∈ segPos s) ↔
        ((∃ w, (Tgt.gate w, (a : Int)) ∈ rootRead P memo node k) ∧
          ((P.defs[node.fn]!).prim = false → a ∉ nodeGates P k)) from
    ⟨fun s hs hc0 => (H.1 s hs).2 hc0, fun a => (H.2 a).trans (hE' a)⟩
  rcases hp : (P.defs[node.fn]!).prim with _ | _
  · have hcomp := nodeReads_comp (rootCtx P memo) k node hn hp rs hrs
    by_cases hm0 : node.members = 0
    · rw [if_pos hm0] at hcomp
      subst hcomp
      refine ⟨fun s hs => by simp at hs, fun a => ?_⟩
      rw [rootRead, if_neg (by rw [hp]; decide), if_pos hm0]
      simp
    · rw [if_neg hm0] at hcomp
      obtain ⟨out, rfl, hpos, hsrc⟩ := hcomp
      have hgood : ∀ s ∈ out, SegGood P c s := fun s hs => hS' s (hsrc s hs)
      have hR : rootRead P memo node k = segsPos out := by
        rw [rootRead, if_neg (by rw [hp]; decide), if_neg hm0, hpos]
      refine ⟨fun s hs => ?_, fun a => ?_⟩
      · obtain ⟨s', hs', rfl⟩ := Array.mem_map.1 hs
        exact segGood_alike (hgood s' (Array.mem_filter.1 hs').1)
      rw [hR]
      constructor
      · rintro ⟨s, hs, hc0, w, hw⟩
        obtain ⟨s', hs', rfl⟩ := Array.mem_map.1 hs
        obtain ⟨hso, hkeep⟩ := Array.mem_filter.1 hs'
        have hc0' : s'.count ≠ 0 := fun h0 => hc0 ((alike_count s').2 h0)
        have hw' := (mem_segPos_alike s' _).1 hw
        exact ⟨⟨w, List.mem_flatMap.2 ⟨s', Array.mem_toList_iff.2 hso, hw'⟩⟩,
          fun _ => (keep_iff P memo c hl hk (hgood s' hso) hc0' hw').1 hkeep⟩
      · rintro ⟨⟨w, hw⟩, hout⟩
        obtain ⟨s, hs, hw⟩ := List.mem_flatMap.1 hw
        have hso := Array.mem_toList_iff.1 hs
        have hc0 : s.count ≠ 0 := fun h0 => by rw [(hgood s hso).1 h0] at hw; simp at hw
        refine ⟨Seg.alike s, Array.mem_map.2 ⟨s, Array.mem_filter.2 ⟨hso,
          (keep_iff P memo c hl hk (hgood s hso) hc0 hw).2 (hout rfl)⟩, rfl⟩,
          fun h0 => hc0 ((alike_count s).1 h0), w, (mem_segPos_alike s _).2 hw⟩
  · obtain ⟨out, rfl, hpos, hsrc⟩ := nodeReads_prim (rootCtx P memo) k node hn hp rs hrs
    have hgood : ∀ s ∈ out, SegGood P c s := fun s hs => hS' s (Or.inl (hsrc s hs))
    have hR : rootRead P memo node k = segsPos out := by rw [rootRead, if_pos hp, hpos]
    refine ⟨fun s hs => ?_, fun a => ?_⟩
    · obtain ⟨s', hs', rfl⟩ := Array.mem_map.1 hs
      exact segGood_alike (hgood s' hs')
    rw [hR]
    constructor
    · rintro ⟨s, hs, -, w, hw⟩
      obtain ⟨s', hs', rfl⟩ := Array.mem_map.1 hs
      exact ⟨⟨w, List.mem_flatMap.2 ⟨s', Array.mem_toList_iff.2 hs', (mem_segPos_alike s' _).1 hw⟩⟩,
        fun h => absurd h (by simp)⟩
    · rintro ⟨⟨w, hw⟩, -⟩
      obtain ⟨s, hs, hw⟩ := List.mem_flatMap.1 hw
      have hs' := Array.mem_toList_iff.1 hs
      refine ⟨Seg.alike s, Array.mem_map.2 ⟨s, hs', rfl⟩, fun h0 => ?_, w, (mem_segPos_alike s _).2 hw⟩
      rw [(hgood s hs').1 ((alike_count s).1 h0)] at hw
      simp at hw


/-! ## A gate's operands

`Flock.Program.operands` resolves each reference of each of a node's arguments (a batch member's slice of a mapped one), so
a gate's operands are exactly those resolutions (`operands_mem`). -/

/-- A loop pushing `f x`'s value for each element `x`: what it ends with is what it started with and those values. -/
theorem forIn_push_mem {α β ε : Type} (f : β → Except ε α) :
    ∀ (l : List β) (acc out : Array α),
      forIn l acc (fun x (r : Array α) => do let v ← f x; pure (ForInStep.yield (r.push v))) = .ok out →
      ∀ a, a ∈ out ↔ a ∈ acc ∨ ∃ x ∈ l, f x = .ok a
  | [], acc, out, h, a => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp
  | x :: l, acc, out, h, a => by
    rw [List.forIn_cons] at h
    obtain ⟨st, hst, h⟩ := Walk.except_bind_ok h
    obtain ⟨v, hv, hst⟩ := Walk.except_bind_ok hst
    simp only [pure, Except.pure, Except.ok.injEq] at hst
    subst hst
    rw [forIn_push_mem f l (acc.push v) out h a, Array.mem_push]
    constructor
    · rintro ((h1 | rfl) | ⟨y, hy, hfy⟩)
      · exact Or.inl h1
      · exact Or.inr ⟨x, List.mem_cons_self, hv⟩
      · exact Or.inr ⟨y, List.mem_cons_of_mem _ hy, hfy⟩
    · rintro (h1 | ⟨y, hy, hfy⟩)
      · exact Or.inl (Or.inl h1)
      · rcases List.mem_cons.1 hy with rfl | hy
        · rw [hv] at hfy
          cases hfy
          exact Or.inl (Or.inr rfl)
        · exact Or.inr ⟨y, hy, hfy⟩

/-- Two nested loops pushing `f i t`'s value for each `t` of `L i`, for each `i`. -/
theorem forIn_nest_mem {α β γ ε : Type} (L : β → List γ) (f : β → γ → Except ε α) :
    ∀ (l : List β) (acc out : Array α),
      forIn l acc (fun i (r : Array α) => do
          let r' ← forIn (L i) r (fun t (s : Array α) => do let v ← f i t; pure (ForInStep.yield (s.push v)))
          pure (ForInStep.yield r')) = .ok out →
      ∀ a, a ∈ out ↔ a ∈ acc ∨ ∃ i ∈ l, ∃ t ∈ L i, f i t = .ok a
  | [], acc, out, h, a => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp
  | i :: l, acc, out, h, a => by
    rw [List.forIn_cons] at h
    obtain ⟨st, hst, h⟩ := Walk.except_bind_ok h
    obtain ⟨r', hx, hst⟩ := Walk.except_bind_ok hst
    simp only [pure, Except.pure, Except.ok.injEq] at hst
    subst hst
    rw [forIn_nest_mem L f l r' out h a, forIn_push_mem (f i) (L i) acc r' hx a]
    constructor
    · rintro ((h1 | ⟨t, ht, hf⟩) | ⟨j, hj, t, ht, hf⟩)
      · exact Or.inl h1
      · exact Or.inr ⟨i, List.mem_cons_self, t, ht, hf⟩
      · exact Or.inr ⟨j, List.mem_cons_of_mem _ hj, t, ht, hf⟩
    · rintro (h1 | ⟨j, hj, t, ht, hf⟩)
      · exact Or.inl (Or.inl h1)
      · rcases List.mem_cons.1 hj with rfl | hj
        · exact Or.inl (Or.inr ⟨t, ht, hf⟩)
        · exact Or.inr ⟨j, hj, t, ht, hf⟩

/-- **A gate's operands are exactly its node's references resolved**: for each argument `i`, each of its references
(a batch member's `memberRef` of a mapped one), resolved in the activation. -/
theorem operands_mem (P : Prog) (sc : Flock.Program.Scope) (node : Flock.Program.Node) (m : Nat) (ops : Array Nat)
    (h : Flock.Program.operands P sc node m = .ok ops) (a : Nat) :
    a ∈ ops ↔ ∃ i < node.args.size,
      ∃ t < (if (node.form == .batch && (node.args[i]!).dims.isSome) = true then
        ((P.defs[node.fn]!).params[i]!).2.leaves else (node.args[i]!).seq.count),
      Flock.Program.resolve P sc (if (node.form == .batch) = true then (node.args[i]!).memberRef m t
        else (node.args[i]!).seq.get t) = .ok a := by
  unfold Flock.Program.operands at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  have h' := ok_bind_pure h
  rw [forIn_nest_mem _ _ _ _ _ h' a]
  simp only [List.mem_range'_1, Std.Legacy.Range.size, Nat.add_sub_cancel, Nat.div_one, Nat.sub_zero, Nat.zero_add,
    Array.not_mem_empty, false_or, zero_le, true_and]


/-- `gateStep` from activation `sc`, the node holding `g` found: a primitive node's member's operands, else that member's
activation. -/
theorem gateStep_at (P : Prog) (g : Nat) (sc : Flock.Program.Scope) (k : Nat)
    (hf : (Array.range (P.defs[sc.d]!).body.nodes.size).find? (fun k => (P.defs[sc.d]!).body.offsets[k]! ≤ g - sc.off &&
      g - sc.off < (P.defs[sc.d]!).body.offsets[k + 1]!) = some k) :
    Flock.Program.gateStep P g sc =
      let node := (P.defs[sc.d]!).body.nodes[k]!
      let m := if (node.form == .prim || node.form == .call) = true then 0
        else (g - sc.off - (P.defs[sc.d]!).body.offsets[k]!) / (P.defs[node.fn]!).gates
      if (P.defs[node.fn]!).prim = true then
        (Flock.Program.operands P sc node m >>= fun ops => pure (.inl (node.fn, ops, sc)))
      else pure (.inr (P.child sc k m)) := by
  unfold Flock.Program.gateStep
  simp only [bind, Except.bind, hf]

/-- Root node `k` is the one `gateStep`'s search finds for a gate of it. -/
theorem find_root (P : Prog) (hl : Desc.rootLaidOut P = true) {k g : Nat} (hk : k < nodeCount P)
    (hg : g ∈ nodeGates P k) :
    (Array.range (P.defs[P.rootScope.d]!).body.nodes.size).find? (fun k =>
      (P.defs[P.rootScope.d]!).body.offsets[k]! ≤ g - P.rootScope.off &&
        g - P.rootScope.off < (P.defs[P.rootScope.d]!).body.offsets[k + 1]!) = some k := by
  obtain ⟨-, -, -, hmono⟩ := DescAgree.rootLaidOut_spec P hl
  have hgk := (DescAgree.mem_nodeGates P k g).1 hg
  have hd : P.rootScope.d = P.root := rfl
  have ho : P.rootScope.off = 0 := rfl
  simp only [hd, ho, Nat.sub_zero]
  rcases h : (Array.range (P.defs[P.root]!).body.nodes.size).find? (fun k =>
      (P.defs[P.root]!).body.offsets[k]! ≤ g && g < (P.defs[P.root]!).body.offsets[k + 1]!) with _ | k'
  · exfalso
    have := Array.find?_eq_none.1 h k (Array.mem_range.2 hk)
    simp only [Bool.and_eq_true, decide_eq_true_eq, not_and, not_lt] at this
    have := this hgk.1
    exact absurd hgk.2 (by unfold rootOffs DescAgree.rootBody at *; omega)
  · obtain ⟨hk', hp⟩ := DescAgree.find?_range_some h
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hp
    rw [DescAgree.node_unique (rootOffs P) (nodeCount P) hmono hk' hk hp hgk]

/-- **A gate of a primitive root node reads its member's operands in the root activation**: member `0` of a primitive
or Call node, else the one `(g - offset) / gates` holds. -/
theorem reads_rootPrim (P : Prog) (c : Circuit) (hc : circuitOf P = .ok c) (hl : Desc.rootLaidOut P = true)
    {k : Nat} (hk : k < nodeCount P) (hp : (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).prim = true) {g : Nat}
    (hg : g ∈ nodeGates P k) :
    Flock.Program.operands P P.rootScope ((P.defs[P.root]!).body.nodes[k]!)
      (if (((P.defs[P.root]!).body.nodes[k]!).form == .prim ||
          ((P.defs[P.root]!).body.nodes[k]!).form == .call) = true then 0
        else (g - (rootOffs P)[k]!) / (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates) = .ok c.reads[g]! := by
  obtain ⟨-, -, hK, hmono⟩ := DescAgree.rootLaidOut_spec P hl
  have hgk := (DescAgree.mem_nodeGates P k g).1 hg
  have hgG : g < gates P := by
    have := DescAgree.offs_mono (rootOffs P) (nodeCount P) hmono (k + 1) (nodeCount P) (by omega) le_rfl
    omega
  obtain ⟨p, sc, hgate, -⟩ := circuitOf_gate hc hgG
  unfold Flock.Program.Prog.gate at hgate
  rw [if_neg (by unfold gates at hgG; omega)] at hgate
  have hF : Flock.Steps.FUEL = (Flock.Steps.FUEL - 1) + 1 := by unfold Flock.Steps.FUEL; omega
  rw [hF, Flock.Steps.run.eq_2, gateStep_at P g P.rootScope k (find_root P hl hk hg)] at hgate
  have hd : P.rootScope.d = P.root := rfl
  have ho : P.rootScope.off = 0 := rfl
  simp only [hd, ho, Nat.sub_zero, hp, ite_true] at hgate
  rcases hops : Flock.Program.operands P P.rootScope ((P.defs[P.root]!).body.nodes[k]!)
      (if (((P.defs[P.root]!).body.nodes[k]!).form == .prim ||
          ((P.defs[P.root]!).body.nodes[k]!).form == .call) = true then 0
        else (g - (P.defs[P.root]!).body.offsets[k]!) / (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates)
    with e | ops
  · rw [show rootOffs P = (P.defs[P.root]!).body.offsets from rfl, hops]
    rw [hops] at hgate
    exact absurd hgate (by simp [bind, Except.bind])
  · rw [show rootOffs P = (P.defs[P.root]!).body.offsets from rfl, hops]
    rw [hops] at hgate
    simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at hgate
    rw [hgate.2.1]


/-! ## Toward the core: `down` keeps the gate every position names

What a position names in an activation `sc` of the body `down` reads (`den`): a gate position's gate, `sc.off` past it,
else its reference resolved there. The core of `RootSegs` and `RootSet` is that `down` keeps every position's gate,
`(segsPos segs).map (den P sc) = (pos q).map (den P sc)`, by induction on its fuel with `fromCallee` and `throughArg`,
from the callees' summaries'. Its base cases hold here: a gate's or a parameter's piece comes back as it is
(`down_gate`, `down_param`), a primitive node's names its gate (`down_prim`, `den_prim`), and a batch of a primitive's
its members' gates (`down_batchPrim`, `den_batchPrim`). Resolution ends with a step's gate and goes on from a step's
state (`resolve_inl`, `resolve_inr`). Lean generates no equations for `down` (its match on `.tuple #[carry, y]`), so
each case unfolds it by `delta` and `whnf`. -/

/-- The gate position `x` names in activation `sc`. -/
def den (P : Prog) (sc : Flock.Program.Scope) : Tgt × Int → Except String Nat
  | (.gate _, p) => if p < 0 then .error "negative" else .ok (sc.off + p.toNat)
  | (.ref b i, j) => if j < 0 then .error "negative" else Flock.Program.resolve P sc ⟨b, i, j.toNat⟩

theorem fuel_succ : ∃ n, Flock.Steps.FUEL = n + 1 := ⟨Flock.Steps.FUEL - 1, by unfold Flock.Steps.FUEL; omega⟩

theorem run_inr {α β : Type} {what : String} {step : α → Flock.V (β ⊕ α)} {n : Nat} {a a' : α} {b : β}
    (hs : step a = .ok (.inr a')) (h : Flock.Steps.run what step (n + 1) a = .ok b) :
    Flock.Steps.run what step n a' = .ok b := by
  rw [Flock.Steps.run.eq_2, hs] at h
  exact h

/-- Resolution ending at a step's gate. -/
theorem resolve_inl (P : Prog) (sc : Flock.Program.Scope) (r : Ref) (g : Nat)
    (h : Flock.Program.resolveStep P (sc, r) = .ok (.inl g)) : Flock.Program.resolve P sc r = .ok g := by
  obtain ⟨n, hn⟩ := fuel_succ
  unfold Flock.Program.resolve
  rw [hn, Flock.Steps.run.eq_2, h]
  rfl

/-- Resolution going on from a step's state, to the same gate. -/
theorem resolve_inr (P : Prog) (sc : Flock.Program.Scope) (r : Ref) (s : Flock.Program.Scope × Ref)
    (h : Flock.Program.resolveStep P (sc, r) = .ok (.inr s)) {g : Nat} (hr : Flock.Program.resolve P sc r = .ok g) :
    Flock.Program.resolve P s.1 s.2 = .ok g := by
  obtain ⟨n, hn⟩ := fuel_succ
  unfold Flock.Program.resolve at hr ⊢
  rw [hn] at hr ⊢
  exact Flock.Steps.run_mono (Nat.le_succ n) (run_inr h hr)

theorem down_gate (c : Desc.Ctx) (fuel : Nat) (q : Piece) {w : Nat} (ht : q.tgt = .gate w) :
    Desc.down c (fuel + 1) q = .ok #[Seg.one q] := by
  delta Desc.down
  conv => lhs; whnf
  rw [ht]
  rfl

theorem down_param (c : Desc.Ctx) (fuel : Nat) (q : Piece) {i : Nat} (ht : q.tgt = .ref true i) :
    Desc.down c (fuel + 1) q = .ok #[Seg.one q] := by
  delta Desc.down
  conv => lhs; whnf
  rw [ht]
  rfl

theorem segsPos_single (q : Piece) : segsPos #[Seg.one q] = pos q := by
  simp [segsPos, segPos_one]

theorem node_get {c : Desc.Ctx} {k : Nat} {node : Flock.Program.Node} (hn : c.node k = .ok node) :
    c.body.nodes[k]! = node := by
  unfold Desc.Ctx.node at hn
  rcases h : c.body.nodes[k]? with _ | n
  · rw [h] at hn; cases hn
  · rw [h] at hn
    simp only [pure, Except.pure, Except.ok.injEq] at hn
    subst hn
    obtain ⟨hk, rfl⟩ := Array.getElem?_eq_some_iff.1 h
    exact getElem!_pos c.body.nodes k hk

/-- A primitive node's piece: none of no count, else one piece naming the node's gate at each position. -/
theorem down_prim (c : Desc.Ctx) (fuel : Nat) (q : Piece) {k : Nat} (ht : q.tgt = .ref false k) (hff : q.flat = q)
    {node : Flock.Program.Node} (hn : c.node k = .ok node) (hform : node.form = .prim) {w : Nat}
    (hret : (c.P.defs[node.fn]!).ret = .value w) :
    Desc.down c (fuel + 1) q = .ok (if q.count = 0 then #[] else
      #[Seg.one { tgt := .gate w, base := (c.body.offsets[k]! : Int), stride := 0, len := q.count }]) := by
  delta Desc.down
  conv => lhs; whnf
  rw [ht, hff]
  dsimp only
  by_cases h0 : q.count = 0
  · rw [if_pos (show (q.count == 0) = true by simpa using h0), if_pos h0]
    rfl
  · rw [if_neg (show ¬ ((q.count == 0) = true) by simpa using h0), if_neg h0, hn]
    dsimp only [bind, Except.bind]
    rw [hform]
    dsimp only
    rw [hret]
    rfl

/-- What a primitive node's piece names: its gate at each position, as resolution finds it. -/
theorem den_prim (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (q : Piece) {k : Nat}
    (ht : q.tgt = .ref false k) {node : Flock.Program.Node} (hn : c.node k = .ok node) (hform : node.form = .prim)
    (hq : ∀ x ∈ pos q, 0 ≤ x.2) (w : Nat) :
    (segsPos (if q.count = 0 then #[] else
      #[Seg.one { tgt := .gate w, base := (c.body.offsets[k]! : Int), stride := 0, len := q.count }])).map
        (den c.P sc) = (pos q).map (den c.P sc) := by
  have hnode : ((c.P.defs[sc.d]!).body.nodes[k]!) = node := by rw [hd]; exact node_get hn
  have hres : ∀ j : Nat, Flock.Program.resolve c.P sc ⟨false, k, j⟩ = .ok (sc.off + c.body.offsets[k]!) := by
    intro j
    apply resolve_inl
    rw [Resolve.rs_prim c.P sc k j (by rw [hnode]; exact hform), hd]
    rfl
  by_cases h0 : q.count = 0
  · rw [if_pos h0]
    have : pos q = [] := List.eq_nil_of_length_eq_zero (by rw [pos_length, h0])
    simp [this, segsPos]
  · rw [if_neg h0, segsPos_single, pos_eq_map, pos_eq_map, List.map_map, List.map_map]
    have hc : ({ tgt := .gate w, base := (c.body.offsets[k]! : Int), stride := 0, len := q.count } : Piece).count =
      q.count := by simp [Desc.Piece.count]
    rw [hc]
    apply List.map_congr_left
    intro t ht'
    have hpt : 0 ≤ ppt q t := by
      have := hq (q.tgt, ppt q t) (by rw [pos_eq_map]; exact List.mem_map.2 ⟨t, ht', rfl⟩)
      exact this
    simp only [Function.comp, den, ppt, ht]
    simp only [Int.mul_zero, Int.add_zero, Int.natCast_nonneg, not_lt.2, ite_false, Int.toNat_natCast]
    rw [if_neg (by unfold ppt at hpt; omega), hres]

/-- A batch of a primitive's piece: none of no count, else the piece moved onto the members' gates. -/
theorem down_batchPrim (c : Desc.Ctx) (fuel : Nat) (q : Piece) {k : Nat} (ht : q.tgt = .ref false k) (hff : q.flat = q)
    {node : Flock.Program.Node} (hn : c.node k = .ok node) (hform : node.form = .batch)
    (hp : (c.P.defs[node.fn]!).prim = true) {w : Nat} (hret : (c.P.defs[node.fn]!).ret = .value w) :
    Desc.down c (fuel + 1) q = .ok (if q.count = 0 then #[] else
      #[Seg.one { q with tgt := .gate w, base := (c.body.offsets[k]! : Int) + q.base }]) := by
  delta Desc.down
  conv => lhs; whnf
  rw [ht, hff]
  dsimp only
  by_cases h0 : q.count = 0
  · rw [if_pos (show (q.count == 0) = true by simpa using h0), if_pos h0]
    rfl
  · rw [if_neg (show ¬ ((q.count == 0) = true) by simpa using h0), if_neg h0, hn]
    dsimp only [bind, Except.bind]
    rw [hform]
    dsimp only
    rw [if_pos hp]
    rw [hret]
    rfl

/-- What a batch of a primitive's piece names: member `j`'s gate at position `j`, as resolution finds it, the
primitive returning one leaf. -/
theorem den_batchPrim (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (q : Piece) {k : Nat}
    (ht : q.tgt = .ref false k) {node : Flock.Program.Node} (hn : c.node k = .ok node) (hform : node.form = .batch)
    (hp : (c.P.defs[node.fn]!).prim = true) (hL : (c.P.defs[node.fn]!).outLeaves = 1)
    (hq : ∀ x ∈ pos q, 0 ≤ x.2) (w : Nat) :
    (segsPos (if q.count = 0 then #[] else
      #[Seg.one { q with tgt := .gate w, base := (c.body.offsets[k]! : Int) + q.base }])).map (den c.P sc) =
      (pos q).map (den c.P sc) := by
  have hnode : ((c.P.defs[c.d]!).body.nodes[k]!) = node := node_get hn
  have hres : ∀ j : Nat, Flock.Program.resolve c.P sc ⟨false, k, j⟩ = .ok (sc.off + c.body.offsets[k]! + j) := by
    intro j
    apply resolve_inl
    rw [Resolve.rs_batch c.P sc k j (by rw [hd, hnode]; exact hform)]
    simp only [hd, hnode, hp, ite_true, hL, Nat.div_one]
    rfl
  by_cases h0 : q.count = 0
  · rw [if_pos h0]
    have : pos q = [] := List.eq_nil_of_length_eq_zero (by rw [pos_length, h0])
    simp [this, segsPos]
  · rw [if_neg h0, segsPos_single, pos_eq_map, pos_eq_map, List.map_map, List.map_map]
    have hc : ({ q with tgt := .gate w, base := (c.body.offsets[k]! : Int) + q.base } : Piece).count = q.count := rfl
    rw [hc]
    apply List.map_congr_left
    intro t ht'
    have hpt : 0 ≤ ppt q t := by
      have := hq (q.tgt, ppt q t) (by rw [pos_eq_map]; exact List.mem_map.2 ⟨t, ht', rfl⟩)
      exact this
    simp only [Function.comp, den, ht]
    unfold ppt at hpt ⊢
    simp only
    rw [if_neg (by omega), if_neg (by omega), hres]
    congr 1
    omega


/-! ## `down`'s composite cases, and the resolution steps they follow

A composite node's piece goes on through `fromCallee` (a Call's, a scan's, row by row through `down`), so each case
unfolds to the loop it runs (`down_call_one`, `down_call_pts`, `down_rows`, `down_scan`); `delta` on both sides, since
the recursive calls unfold to the same recursor. Resolution takes the same step: down through a composite node's output
into the member's activation (`den_call`, `den_batch`, `den_scan`), and up through a parameter to the node's argument
in the parent (`den_up`, `den_up_batch`, `den_up_scan`, `den_up_scan0`). -/

/-- A node's piece of no count names nothing. -/
theorem down_count0 (c : Desc.Ctx) (fuel : Nat) (q : Piece) {k : Nat} (ht : q.tgt = .ref false k) (h0 : q.count = 0) :
    Desc.down c (fuel + 1) q = .ok #[] := by
  delta Desc.down
  conv => lhs; whnf
  rw [ht]
  dsimp only
  rw [if_pos (show (q.count == 0) = true by simpa using h0)]
  rfl

/-! ## `Piece.flat` and `walkAtMost`

`down` reads a piece of consecutive rows as one row (`Piece.flat`), which names the same positions in order
(`pos_flat`), and refuses a walk past `Desc.WALK`, which is no step where it goes on (`walkAtMost_bind`). -/

theorem flat_cases (q : Piece) :
    q.flat = q ∨
      (q.len = 1 ∧ q.flat = { q with stride := q.rowStride, len := q.rows, rows := 1, rowStride := 0 }) ∨
      (q.rowStride = (q.len : Int) * q.stride ∧ q.flat = { q with len := q.rows * q.len, rows := 1, rowStride := 0 }) := by
  unfold Desc.Piece.flat
  by_cases h1 : q.rows ≤ 1
  · left; rw [if_pos h1]
  rw [if_neg h1]
  by_cases h2 : q.len = 1
  · right; left; rw [if_pos (by simp [h2])]; exact ⟨h2, rfl⟩
  rw [if_neg (by simpa using h2)]
  by_cases h3 : q.rowStride = (q.len : Int) * q.stride
  · right; right; rw [if_pos (by simp [h3])]; exact ⟨h3, rfl⟩
  · left; rw [if_neg (by simpa using h3)]

theorem flat_tgt (q : Piece) : q.flat.tgt = q.tgt := by
  rcases flat_cases q with h | ⟨-, h⟩ | ⟨-, h⟩ <;> rw [h]

theorem flat_count (q : Piece) : q.flat.count = q.count := by
  rcases flat_cases q with h | ⟨h1, h⟩ | ⟨-, h⟩ <;> rw [h]
  · simp [Desc.Piece.count, h1]
  · simp [Desc.Piece.count]

theorem flat_flat (q : Piece) : q.flat.flat = q.flat := by
  rcases flat_cases q with h | ⟨-, h⟩ | ⟨-, h⟩
  · rw [h, h]
  · rw [h]; unfold Desc.Piece.flat; rw [if_pos (le_refl 1)]
  · rw [h]; unfold Desc.Piece.flat; rw [if_pos (le_refl 1)]

theorem pos_flat (q : Piece) : pos q.flat = pos q := by
  rw [pos_eq_map, pos_eq_map, flat_count, flat_tgt]
  apply List.map_congr_left
  intro t ht
  rw [List.mem_range, ← flat_count] at ht
  congr 1
  rcases flat_cases q with h | ⟨h1, h⟩ | ⟨h1, h⟩
  · rw [h]
  · rw [h] at ht ⊢
    simp only [Desc.Piece.count] at ht
    unfold ppt
    simp only [h1, Nat.div_one, Nat.mod_one]
    rw [Nat.div_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)]
    push_cast; ring
  · rw [h] at ht ⊢
    simp only [Desc.Piece.count] at ht
    unfold ppt
    dsimp only at ht ⊢
    rw [Nat.div_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega), h1]
    have e := Nat.div_add_mod t q.len
    have e' : ((t : Nat) : Int) = ((q.len * (t / q.len) + t % q.len : Nat) : Int) := by rw [e]
    simp only [Nat.cast_zero, zero_mul, add_zero]
    rw [e']
    push_cast; ring

/-- A node's piece: `down` reads it as one row where its rows are consecutive. -/
theorem down_flat (c : Desc.Ctx) (fuel : Nat) (q : Piece) {k : Nat} (ht : q.tgt = .ref false k) :
    Desc.down c (fuel + 1) q = Desc.down c (fuel + 1) q.flat := by
  have hft := flat_tgt q
  have hfc := flat_count q
  have hff := flat_flat q
  delta Desc.down
  conv => lhs; whnf
  rw [← hfc]
  generalize q.flat = q' at hft hfc hff ⊢
  rw [ht] at hft ⊢
  conv => rhs; whnf
  rw [hft, hff]

/-- `down` of each row of a piece in turn. -/
def byRows (c : Desc.Ctx) (fuel : Nat) (q : Piece) : Except String (Array Seg) := do
  Desc.walkAtMost "rows of a read resolved row by row" q.rows
  let mut out := #[]
  for x in q.rowsOf do out := out ++ (← Desc.down c fuel x)
  return out

/-- `fromCallee` of member 0 at each of a piece's positions in turn. -/
def callPts (c : Desc.Ctx) (fuel k : Nat) (q : Piece) : Except String (Array Seg) := do
  let mut out := #[]
  for t in [0:q.len] do
    out := out ++ (← Desc.fromCallee c fuel k 0 (q.at t).toNat ((q.at t).toNat + 1))
  return out

/-- A Call's piece of one row and stride 1: its callee's return leaves over the piece's span. -/
theorem down_call_one (c : Desc.Ctx) (fuel : Nat) (q : Piece) {k : Nat} (ht : q.tgt = .ref false k) (hff : q.flat = q)
    {node : Flock.Program.Node} (hn : c.node k = .ok node) (hform : node.form = .call) (h0 : q.count ≠ 0)
    (hr : ¬ q.rows > 1) (hs : q.stride = 1) :
    Desc.down c (fuel + 1) q = Desc.fromCallee c fuel k 0 q.base.toNat (q.base.toNat + q.len) := by
  delta Desc.down
  conv => lhs; whnf
  rw [ht, hff]
  dsimp only
  rw [if_neg (show ¬ ((q.count == 0) = true) by simpa using h0), hn]
  dsimp only [bind, Except.bind]
  rw [hform]
  dsimp only
  rw [if_neg hr, if_pos (show (q.stride == 1) = true by simp [hs])]
  delta Desc.fromCallee
  rfl

/-- A Call's piece of one row and another stride: its callee's return leaf at each position. -/
theorem down_call_pts (c : Desc.Ctx) (fuel : Nat) (q : Piece) {k : Nat} (ht : q.tgt = .ref false k) (hff : q.flat = q)
    {node : Flock.Program.Node} (hn : c.node k = .ok node) (hform : node.form = .call) (h0 : q.count ≠ 0)
    (hr : ¬ q.rows > 1) (hs : q.stride ≠ 1) :
    Desc.down c (fuel + 1) q = callPts c fuel k q := by
  delta Desc.down
  conv => lhs; whnf
  rw [ht, hff]
  dsimp only
  rw [if_neg (show ¬ ((q.count == 0) = true) by simpa using h0), hn]
  dsimp only [bind, Except.bind]
  rw [hform]
  dsimp only
  rw [if_neg hr, if_neg (show ¬ ((q.stride == 1) = true) by simpa using hs)]
  delta callPts Desc.fromCallee
  rfl

/-- A Call's or a scan's piece of more than one row: row by row. -/
theorem down_rows (c : Desc.Ctx) (fuel : Nat) (q : Piece) {k : Nat} (ht : q.tgt = .ref false k) (hff : q.flat = q)
    {node : Flock.Program.Node} (hn : c.node k = .ok node)
    (hform : node.form = .call ∨ (node.form = .scan ∧ node.n ≠ 0))
    (h0 : q.count ≠ 0) (hr : q.rows > 1) :
    Desc.down c (fuel + 1) q = byRows c fuel q := by
  delta Desc.down
  conv => lhs; whnf
  rw [ht, hff]
  dsimp only
  rw [if_neg (show ¬ ((q.count == 0) = true) by simpa using h0), hn]
  dsimp only [bind, Except.bind]
  rcases hform with hform | ⟨hform, hn0⟩
  · rw [hform]
    dsimp only
    rw [if_pos hr]
    delta byRows Desc.down
    rfl
  · rw [hform]
    dsimp only
    rw [if_neg (show ¬ ((node.n == 0) = true) by simpa using hn0), if_pos hr]
    delta byRows Desc.down
    rfl

/-- A scan with no iterations has no output for `down` to read. -/
theorem down_scan0 (c : Desc.Ctx) (fuel : Nat) (q : Piece) {k : Nat} (ht : q.tgt = .ref false k) (hff : q.flat = q)
    {node : Flock.Program.Node} (hn : c.node k = .ok node) (hform : node.form = .scan) (hn0 : node.n = 0)
    (h0 : q.count ≠ 0) (x : Array Seg) : Desc.down c (fuel + 1) q ≠ .ok x := by
  delta Desc.down
  conv => lhs; whnf
  rw [ht, hff]
  dsimp only
  rw [if_neg (show ¬ ((q.count == 0) = true) by simpa using h0), hn]
  dsimp only [bind, Except.bind]
  rw [hform]
  dsimp only
  rw [if_pos (show (node.n == 0) = true by simpa using hn0)]
  intro h
  cases h

/-- Down through a Call's output: leaf `j` resolves as the callee's return leaf `j` in the Call's activation. -/
theorem den_call (P : Prog) (sc : Flock.Program.Scope) {k : Nat} {node : Flock.Program.Node}
    (hnode : (P.defs[sc.d]!).body.nodes[k]! = node) (hform : node.form = .call) {j a : Nat}
    (h : Flock.Program.resolve P sc ⟨false, k, j⟩ = .ok a) :
    Flock.Program.resolve P (.child sc node.fn (sc.off + (P.defs[sc.d]!).body.offsets[k]!) k 0)
      ((P.defs[node.fn]!).body.ret.get j) = .ok a := by
  have e := resolve_inr P sc _ _ (Resolve.rs_call P sc k j (by rw [hnode]; exact hform)) h
  rw [hnode] at e
  exact e

/-- Down through a batch of a composite's output: leaf `j` resolves as member `j / L`'s return leaf `j % L`. -/
theorem den_batch (P : Prog) (sc : Flock.Program.Scope) {k : Nat} {node : Flock.Program.Node}
    (hnode : (P.defs[sc.d]!).body.nodes[k]! = node) (hform : node.form = .batch)
    (hp : (P.defs[node.fn]!).prim = false) {j a : Nat}
    (h : Flock.Program.resolve P sc ⟨false, k, j⟩ = .ok a) :
    Flock.Program.resolve P (.child sc node.fn (sc.off + (P.defs[sc.d]!).body.offsets[k]! +
        j / (P.defs[node.fn]!).outLeaves * (P.defs[node.fn]!).gates) k (j / (P.defs[node.fn]!).outLeaves))
      ((P.defs[node.fn]!).body.ret.get (j % (P.defs[node.fn]!).outLeaves)) = .ok a := by
  have hs := Resolve.rs_batch P sc k j (by rw [hnode]; exact hform)
  rw [hnode] at hs
  simp only [hp, Bool.false_eq_true, ite_false] at hs
  exact resolve_inr P sc _ _ hs h

/-- Down through a scan's output: a carry leaf resolves as the last iteration's, else leaf `j` as iteration
`(j - cl) / yl`'s `y` leaf. -/
theorem den_scan (P : Prog) (sc : Flock.Program.Scope) {k : Nat} {node : Flock.Program.Node}
    (hnode : (P.defs[sc.d]!).body.nodes[k]! = node) (hform : node.form = .scan) (hn0 : node.n ≠ 0)
    {carry y : Flock.Program.Ty} (hret : (P.defs[node.fn]!).ret = .tuple #[carry, y]) {j a : Nat}
    (h : Flock.Program.resolve P sc ⟨false, k, j⟩ = .ok a) :
    let mj := if j < carry.leaves then (node.n - 1, j)
      else ((j - carry.leaves) / y.leaves, carry.leaves + (j - carry.leaves) % y.leaves)
    Flock.Program.resolve P (.child sc node.fn (sc.off + (P.defs[sc.d]!).body.offsets[k]! +
        mj.1 * (P.defs[node.fn]!).gates) k mj.1) ((P.defs[node.fn]!).body.ret.get mj.2) = .ok a := by
  have hs := Resolve.rs_scan P sc k j carry y (by rw [hnode]; exact hform) (by rw [hnode]; exact hn0)
    (by rw [hnode]; exact hret)
  have e := resolve_inr P sc _ _ hs h
  rw [hnode] at e
  exact e

/-- Up through a parameter of a primitive's or a Call's activation: the node's argument in the parent. -/
theorem den_up (P : Prog) (parent : Flock.Program.Scope) {d off k m i j : Nat} {node : Flock.Program.Node}
    (hnode : (P.defs[parent.d]!).body.nodes[k]! = node) (hform : node.form = .prim ∨ node.form = .call) {a : Nat}
    (h : Flock.Program.resolve P (.child parent d off k m) ⟨true, i, j⟩ = .ok a) :
    Flock.Program.resolve P parent ((node.args[i]!).seq.get j) = .ok a := by
  have e := resolve_inr P _ _ _ (Resolve.rs_up P parent d off k m i j (by rw [hnode]; exact hform)) h
  rw [hnode] at e
  exact e

/-- Up through a parameter of a batch member's activation: the member's reference of the node's argument. -/
theorem den_up_batch (P : Prog) (parent : Flock.Program.Scope) {d off k m i j : Nat} {node : Flock.Program.Node}
    (hnode : (P.defs[parent.d]!).body.nodes[k]! = node) (hform : node.form = .batch) {a : Nat}
    (h : Flock.Program.resolve P (.child parent d off k m) ⟨true, i, j⟩ = .ok a) :
    Flock.Program.resolve P parent ((node.args[i]!).memberRef m j) = .ok a := by
  have e := resolve_inr P _ _ _ (Resolve.rs_up_batch P parent d off k m i j (by rw [hnode]; exact hform)) h
  rw [hnode] at e
  exact e

/-- Up through a parameter other than the carry of a scan iteration's activation: its `xs` slice of the node's
argument, or the whole argument. -/
theorem den_up_scan (P : Prog) (parent : Flock.Program.Scope) {d off k m i j : Nat} {node : Flock.Program.Node}
    (hnode : (P.defs[parent.d]!).body.nodes[k]! = node) (hform : node.form = .scan) (hi : i ≠ 0) {a : Nat}
    (h : Flock.Program.resolve P (.child parent d off k m) ⟨true, i, j⟩ = .ok a) :
    Flock.Program.resolve P parent (if i ≤ node.nxs
      then (node.args[i]!).seq.get (m * ((node.args[i]!).dims.map (·.2)).getD 0 + j)
      else (node.args[i]!).seq.get j) = .ok a := by
  have e := resolve_inr P _ _ _ (Resolve.rs_up_scan P parent d off k m i j hi (by rw [hnode]; exact hform)) h
  rw [hnode] at e
  exact e

/-- Up through the carry of a scan iteration's activation: the node's initial carry for the first iteration, else the
iteration before's return. -/
theorem den_up_scan0 (P : Prog) (parent : Flock.Program.Scope) {d off k m j : Nat} {node : Flock.Program.Node}
    (hnode : (P.defs[parent.d]!).body.nodes[k]! = node) (hform : node.form = .scan) {a : Nat}
    (h : Flock.Program.resolve P (.child parent d off k m) ⟨true, 0, j⟩ = .ok a) :
    if m = 0 then Flock.Program.resolve P parent ((node.args[0]!).seq.get j) = .ok a
    else Flock.Program.resolve P (.child parent d (off - (P.defs[d]!).gates) k (m - 1))
      ((P.defs[d]!).body.ret.get j) = .ok a := by
  have hs := Resolve.rs_up_scan0 P parent d off k m j (by rw [hnode]; exact hform)
  rw [hnode] at hs
  by_cases hm : m = 0
  · rw [if_pos hm] at hs ⊢
    exact resolve_inr P _ _ _ hs h
  · rw [if_neg hm] at hs ⊢
    exact resolve_inr P _ _ _ hs h


/-- A batch of a composite's piece when every member returns one run of its gates and the members' runs make one,
leaf `e` gate `A + e * F` (`down`'s first case for a batch of a composite). -/
def affineOf (c : Desc.Ctx) (k : Nat) (node : Flock.Program.Node) (q : Piece) : Option Piece :=
  let callee := c.P.defs[node.fn]!
  let L := callee.outLeaves
  let G : Int := callee.gates
  let r := (c.memo[node.fn]!).ret
  let o : Int := c.body.offsets[k]!
  if r.segs.size == 1 && r.segs[0]!.reps ≤ 1 && r.segs[0]!.ps.size == 1 then
    let g := r.segs[0]!.ps[0]!
    match g.tgt with
    | .gate w =>
      if g.rows ≤ 1 && g.len == L && (L == 1 || G == L * g.stride) then
        some (q.affine w (o + g.base) (if L == 1 then G else g.stride))
      else none
    | _ => none
  else none

/-- A batch of a composite's piece in its one-run case: the piece moved onto that run. -/
theorem down_batch_affine (c : Desc.Ctx) (fuel : Nat) (q : Piece) {k : Nat} (ht : q.tgt = .ref false k)
    (hff : q.flat = q) {node : Flock.Program.Node} (hn : c.node k = .ok node) (hform : node.form = .batch)
    (hp : (c.P.defs[node.fn]!).prim = false) (hL : (c.P.defs[node.fn]!).outLeaves ≠ 0) (h0 : q.count ≠ 0)
    {p : Piece} (ha : affineOf c k node q = some p) :
    Desc.down c (fuel + 1) q = .ok #[Seg.one p] := by
  delta Desc.down
  conv => lhs; whnf
  rw [ht, hff]
  dsimp only
  rw [if_neg (show ¬ ((q.count == 0) = true) by simpa using h0), hn]
  dsimp only [bind, Except.bind]
  rw [hform]
  dsimp only
  rw [if_neg (show ¬ ((c.P.defs[node.fn]!).prim = true) by simp [hp])]
  rw [if_neg (show ¬ (((c.P.defs[node.fn]!).outLeaves == 0) = true) by simpa using hL)]
  unfold affineOf at ha
  dsimp only at ha
  rcases hC : ((c.memo[node.fn]!).ret.segs.size == 1 && (c.memo[node.fn]!).ret.segs[0]!.reps ≤ 1 &&
      (c.memo[node.fn]!).ret.segs[0]!.ps.size == 1) with _ | _ <;>
    simp only [hC, Bool.false_eq_true, ite_false, ite_true] at ha
  · cases ha
  rw [if_pos rfl]
  cases hT : ((c.memo[node.fn]!).ret.segs[0]!.ps[0]!).tgt with
  | ref b i => simp only [hT] at ha; cases ha
  | gate w =>
    simp only [hT] at ha
    rcases hC2 : (((c.memo[node.fn]!).ret.segs[0]!.ps[0]!).rows ≤ 1 &&
        ((c.memo[node.fn]!).ret.segs[0]!.ps[0]!).len == (c.P.defs[node.fn]!).outLeaves &&
        ((c.P.defs[node.fn]!).outLeaves == 1 || ((c.P.defs[node.fn]!).gates : Int) ==
          ((c.P.defs[node.fn]!).outLeaves : Int) * ((c.memo[node.fn]!).ret.segs[0]!.ps[0]!).stride)) with _ | _ <;>
      simp only [hC2, Bool.false_eq_true, ite_false, ite_true, Option.some.injEq] at ha
    · cases ha
    subst ha
    rfl

/-- A batch of a composite returning nothing has no output for `down` to read. -/
theorem down_batch0 (c : Desc.Ctx) (fuel : Nat) (q : Piece) {k : Nat} (ht : q.tgt = .ref false k) (hff : q.flat = q)
    {node : Flock.Program.Node} (hn : c.node k = .ok node) (hform : node.form = .batch)
    (hp : (c.P.defs[node.fn]!).prim = false) (hL : (c.P.defs[node.fn]!).outLeaves = 0) (h0 : q.count ≠ 0)
    (x : Array Seg) : Desc.down c (fuel + 1) q ≠ .ok x := by
  delta Desc.down
  conv => lhs; whnf
  rw [ht, hff]
  dsimp only
  rw [if_neg (show ¬ ((q.count == 0) = true) by simpa using h0), hn]
  dsimp only [bind, Except.bind]
  rw [hform]
  dsimp only
  rw [if_neg (show ¬ ((c.P.defs[node.fn]!).prim = true) by simp [hp])]
  rw [if_pos (show (((c.P.defs[node.fn]!).outLeaves == 0) = true) by simpa using hL)]
  intro h
  cases h

/-- `down` with no fuel left refuses. -/
theorem down_zero (c : Desc.Ctx) (q : Piece) (x : Array Seg) : Desc.down c 0 q ≠ .ok x := by
  delta Desc.down
  intro h
  cases h

/-- A primitive node's piece of positive count, or a batch of a primitive's, needs the primitive to return one
value. -/
theorem down_prim_ret (c : Desc.Ctx) (fuel : Nat) (q : Piece) {k : Nat} (ht : q.tgt = .ref false k) (hff : q.flat = q)
    {node : Flock.Program.Node} (hn : c.node k = .ok node)
    (hform : node.form = .prim ∨ (node.form = .batch ∧ (c.P.defs[node.fn]!).prim = true)) (h0 : q.count ≠ 0)
    (hret : ∀ w, (c.P.defs[node.fn]!).ret ≠ .value w) (x : Array Seg) : Desc.down c (fuel + 1) q ≠ .ok x := by
  delta Desc.down
  conv => lhs; whnf
  rw [ht, hff]
  dsimp only
  rw [if_neg (show ¬ ((q.count == 0) = true) by simpa using h0), hn]
  dsimp only [bind, Except.bind]
  rcases hform with hform | ⟨hform, hp⟩
  · rw [hform]
    dsimp only
    cases hr : (c.P.defs[node.fn]!).ret with
    | value w => exact absurd hr (hret w)
    | array n e => intro h; cases h
    | tuple fs => intro h; cases h
    | record fs => intro h; cases h
  · rw [hform]
    dsimp only
    rw [if_pos hp]
    cases hr : (c.P.defs[node.fn]!).ret with
    | value w => exact absurd hr (hret w)
    | array n e => intro h; cases h
    | tuple fs => intro h; cases h
    | record fs => intro h; cases h

/-- A scan's piece of one row needs its step to return `(carry, y)`. -/
theorem down_scan_ret (c : Desc.Ctx) (fuel : Nat) (q : Piece) {k : Nat} (ht : q.tgt = .ref false k) (hff : q.flat = q)
    {node : Flock.Program.Node} (hn : c.node k = .ok node) (hform : node.form = .scan) (hn0 : node.n ≠ 0)
    (h0 : q.count ≠ 0) (hr : ¬ q.rows > 1)
    (hret : ∀ carry y, (c.P.defs[node.fn]!).ret ≠ .tuple #[carry, y]) (x : Array Seg) :
    Desc.down c (fuel + 1) q ≠ .ok x := by
  delta Desc.down
  conv => lhs; whnf
  rw [ht, hff]
  dsimp only
  rw [if_neg (show ¬ ((q.count == 0) = true) by simpa using h0), hn]
  dsimp only [bind, Except.bind]
  rw [hform]
  dsimp only
  rw [if_neg (show ¬ ((node.n == 0) = true) by simpa using hn0), if_neg hr]
  cases hre : (c.P.defs[node.fn]!).ret with
  | value w => intro h; cases h
  | array n e => intro h; cases h
  | record fs => intro h; cases h
  | tuple fs =>
    rcases fs with ⟨_ | ⟨a, _ | ⟨b, _ | ⟨d, l⟩⟩⟩⟩
    · intro h; cases h
    · intro h; cases h
    · exact absurd hre (hret a b)
    · intro h; cases h

/-- A scan's piece of one row and stride 1, `cl` carry leaves and `yl` leaves of `y`: the final carry's leaves together
(`carry`), then the ys iteration by iteration (`fromCallee`). -/
def scanOne (c : Desc.Ctx) (fuel k : Nat) (node : Flock.Program.Node) (cl yl : Nat) (q : Piece) :
    Except String (Array Seg) := do
  let mut out := #[]
  let (a, b) := (q.base.toNat, q.base.toNat + q.len)
  if a < cl then out := out ++ (← Desc.carry c fuel k (node.n - 1) a (min b cl))
  if cl < b && yl > 0 then
    let (lo, hi) := (max a cl - cl, b - cl)
    Desc.walkAtMost "iterations of a scan whose ys are read one by one" ((hi - 1) / yl + 1 - lo / yl)
    for m in [lo / yl:(hi - 1) / yl + 1] do
      out := out ++ (← Desc.fromCallee c fuel k m (cl + max lo (m * yl) - m * yl) (cl + min hi ((m + 1) * yl) - m * yl))
  return out

/-- A scan's piece of one row and another stride: position by position, a carry leaf through `carry`, a `y` leaf
through `fromCallee`. -/
def scanPts (c : Desc.Ctx) (fuel k : Nat) (node : Flock.Program.Node) (cl yl : Nat) (q : Piece) :
    Except String (Array Seg) := do
  let mut out := #[]
  Desc.walkAtMost "leaves of a scan's output read one by one" q.len
  for t in [0:q.len] do
    let j := (q.at t).toNat
    if j < cl then out := out ++ (← Desc.carry c fuel k (node.n - 1) j (j + 1))
    else
      let (m, jj) := ((j - cl) / yl, cl + (j - cl) % yl)
      out := out ++ (← Desc.fromCallee c fuel k m jj (jj + 1))
  return out

theorem down_scan_head (c : Desc.Ctx) (fuel : Nat) (q : Piece) {k : Nat} (ht : q.tgt = .ref false k)
    (hff : q.flat = q) {node : Flock.Program.Node} (hn : c.node k = .ok node) (hform : node.form = .scan)
    (hn0 : node.n ≠ 0) (h0 : q.count ≠ 0) (hr : ¬ q.rows > 1) {carry y : Flock.Program.Ty}
    (hret : (c.P.defs[node.fn]!).ret = .tuple #[carry, y]) :
    Desc.down c (fuel + 1) q = (if q.stride = 1 then scanOne c fuel k node carry.leaves y.leaves q
      else scanPts c fuel k node carry.leaves y.leaves q) := by
  delta Desc.down
  conv => lhs; whnf
  rw [ht, hff]
  dsimp only
  rw [if_neg (show ¬ ((q.count == 0) = true) by simpa using h0), hn]
  dsimp only [bind, Except.bind]
  rw [hform]
  dsimp only
  rw [if_neg (show ¬ ((node.n == 0) = true) by simpa using hn0), if_neg hr]
  conv => lhs; whnf
  rw [hret]
  cases hsb : (q.stride == 1) with
  | true =>
    rw [if_pos (by simpa using hsb)]
    delta scanOne Desc.fromCallee Desc.carry
    rfl
  | false =>
    rw [if_neg (by simpa using hsb)]
    delta scanPts Desc.fromCallee Desc.carry
    rfl

/-- A node's piece of positive count needs the node. -/
theorem down_node (c : Desc.Ctx) (fuel : Nat) (q : Piece) {k : Nat} (ht : q.tgt = .ref false k) (h0 : q.count ≠ 0)
    {e : String} (hn : c.node k = .error e) (x : Array Seg) : Desc.down c (fuel + 1) q ≠ .ok x := by
  delta Desc.down
  conv => lhs; whnf
  rw [ht]
  dsimp only
  rw [if_neg (show ¬ ((q.count == 0) = true) by simpa using h0), hn]
  intro h
  cases h


/-! ## The core, case by case: `down` keeps each position's gate

`Agree P sc xs vs`: position by position, wherever `vs` resolves to a gate, `xs` names that gate in activation `sc`.
What `down` and `fromCallee` give agrees so with what they read (`DownSpec`, `CalleeSpec`), by induction on their fuel,
each step from the one before. Here are `down`'s steps that read no callee's summary: a gate's or a parameter's piece,
one of no count, a primitive node's or a batch of a primitive's (`downSpec_gate`, `downSpec_param`, `downSpec_count0`,
`downSpec_prim`, `downSpec_batchPrim`), a piece read row by row (`byRows_agree`), and a Call's or a scan's piece of
one row, read through what `fromCallee` keeps (`downSpec_callOne`, `downSpec_callPts`, `downSpec_scanPts`). -/

/-- Position by position, `xs` names in `sc` the gate `vs` resolves to, wherever it resolves. -/
def Agree (P : Prog) (sc : Flock.Program.Scope) (xs : List (Tgt × Int)) (vs : List (Except String Nat)) : Prop :=
  List.Forall₂ (fun x v => ∀ a, v = .ok a → den P sc x = .ok a) xs vs

theorem agree_refl (P : Prog) (sc : Flock.Program.Scope) (xs : List (Tgt × Int)) :
    Agree P sc xs (xs.map (den P sc)) := by
  induction xs with
  | nil => exact List.Forall₂.nil
  | cons x xs ih => exact List.Forall₂.cons (fun a h => h) ih

theorem agree_append {P : Prog} {sc : Flock.Program.Scope} {xs ys : List (Tgt × Int)}
    {vs ws : List (Except String Nat)} (h1 : Agree P sc xs vs) (h2 : Agree P sc ys ws) :
    Agree P sc (xs ++ ys) (vs ++ ws) := by
  induction h1 with
  | nil => exact h2
  | cons h _ ih => exact List.Forall₂.cons h ih

theorem agree_flatMap {β : Type} {P : Prog} {sc : Flock.Program.Scope} (l : List β) (F : β → List (Tgt × Int))
    (G : β → List (Except String Nat)) (h : ∀ x ∈ l, Agree P sc (F x) (G x)) :
    Agree P sc (l.flatMap F) (l.flatMap G) := by
  induction l with
  | nil => exact List.Forall₂.nil
  | cons x l ih =>
    rw [List.flatMap_cons, List.flatMap_cons]
    exact agree_append (h x List.mem_cons_self) (ih fun y hy => h y (List.mem_cons_of_mem _ hy))

/-- Agreement with what resolves no less. -/
theorem agree_weaken {P : Prog} {sc : Flock.Program.Scope} {xs : List (Tgt × Int)} {vs ws : List (Except String Nat)}
    (h : Agree P sc xs vs) (hw : List.Forall₂ (fun w v => ∀ a, w = .ok a → v = .ok a) ws vs) : Agree P sc xs ws := by
  induction h generalizing ws with
  | nil => cases hw; exact List.Forall₂.nil
  | cons hx _ ih =>
    cases hw with
    | cons hv hw' => exact List.Forall₂.cons (fun a ha => hx a (hv a ha)) (ih hw')

theorem agree_of_map {P : Prog} {sc : Flock.Program.Scope} {xs ys : List (Tgt × Int)}
    (h : xs.map (den P sc) = ys.map (den P sc)) : Agree P sc xs (ys.map (den P sc)) :=
  h ▸ agree_refl P sc xs

theorem forall₂_range_map {α β : Type} (R : α → β → Prop) (n : Nat) (f : Nat → α) (g : Nat → β)
    (h : ∀ t < n, R (f t) (g t)) : List.Forall₂ R ((List.range n).map f) ((List.range n).map g) := by
  induction n with
  | zero => exact List.Forall₂.nil
  | succ n ih =>
    rw [List.range_succ, List.map_append, List.map_append]
    exact List.rel_append (ih fun t ht => h t (by omega)) (List.Forall₂.cons (h n (by omega)) List.Forall₂.nil)

/-- A loop appending each element's segments agrees with its elements' reads laid end to end. -/
theorem forIn_agree {β : Type} {P : Prog} {sc : Flock.Program.Scope} (F : β → Except String (Array Seg))
    (T : β → List (Except String Nat)) (f : β → Array Seg → Except String (ForInStep (Array Seg)))
    (hf : ∀ x r y, f x r = .ok y → ∃ segs, F x = .ok segs ∧ y = .yield (r ++ segs)) :
    ∀ (l : List β), (∀ x ∈ l, ∀ segs, F x = .ok segs → Agree P sc (segsPos segs) (T x)) →
      ∀ (acc out : Array Seg) (vs : List (Except String Nat)), Agree P sc (segsPos acc) vs →
      forIn l acc f = .ok out → Agree P sc (segsPos out) (vs ++ l.flatMap T)
  | [], _, acc, out, vs, ha, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    simpa using ha
  | x :: l, hF, acc, out, vs, ha, h => by
    rw [List.forIn_cons] at h
    rcases hx : f x acc with e | y <;> rw [hx] at h
    · exact absurd h (by intro h; cases h)
    obtain ⟨segs, hs, rfl⟩ := hf x acc _ hx
    have := forIn_agree F T f hf l (fun y hy => hF y (List.mem_cons_of_mem _ hy)) (acc ++ segs) out (vs ++ T x)
      (by rw [segsPos_append]; exact agree_append ha (hF x List.mem_cons_self segs hs)) h
    rwa [List.flatMap_cons, ← List.append_assoc]

/-- `F`'s loop from no segments, its result returned. -/
theorem loop_agree {β : Type} {P : Prog} {sc : Flock.Program.Scope} (F : β → Except String (Array Seg))
    (T : β → List (Except String Nat)) (l : List β)
    (hF : ∀ x ∈ l, ∀ segs, F x = .ok segs → Agree P sc (segsPos segs) (T x)) (out : Array Seg)
    (h : (forIn l #[] (fun x (r : Array Seg) => do
      let v ← F x
      pure (ForInStep.yield (r ++ v))) >>= fun r => pure r) = .ok out) :
    Agree P sc (segsPos out) (l.flatMap T) := by
  have := forIn_agree F T _ (fun x r y hy => by
    obtain ⟨segs, hs, hy⟩ := Walk.except_bind_ok hy
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    exact ⟨segs, hs, hy.symm⟩) l hF #[] out [] List.Forall₂.nil (ok_bind_pure h)
  simpa using this

/-- A position of the body's node `k` names one of its output leaves. -/
def Fits (c : Desc.Ctx) (x : Tgt × Int) : Prop :=
  ∀ k, x.1 = .ref false k → x.2 < ((c.body.nodes[k]!).out.leaves : Int)

/-- What `down` keeps of a piece whose positions are all at or past 0 and name its node's output leaves: position by
position, the gate each names. -/
def DownSpec (c : Desc.Ctx) (sc : Flock.Program.Scope) (fuel : Nat) : Prop :=
  ∀ q segs, (∀ x ∈ pos q, 0 ≤ x.2) → (∀ x ∈ pos q, Fits c x) → Desc.down c fuel q = .ok segs →
    Agree c.P sc (segsPos segs) ((pos q).map (den c.P sc))

/-- What `fromCallee` keeps of member `m`'s return leaves `[a, b)`, within its callee's return when not empty: the gate
each resolves to in the member's activation. -/
def CalleeSpec (c : Desc.Ctx) (sc : Flock.Program.Scope) (fuel : Nat) : Prop :=
  ∀ k m a b segs, m < (c.body.nodes[k]!).members → (a < b → b ≤ (c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.count) →
    Desc.fromCallee c fuel k m a b = .ok segs →
    Agree c.P sc (segsPos segs) ((List.range' a (b - a)).map fun j =>
      Flock.Program.resolve c.P (c.P.child sc k m) ((c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.get j))

/-- What `carry` keeps of scan iteration `m`'s return leaves `[a, b)`, within its step's return when not empty: as
`fromCallee`'s (`CalleeSpec`), the gate each resolves to in the iteration's activation. -/
def CarrySpec (c : Desc.Ctx) (sc : Flock.Program.Scope) (fuel : Nat) : Prop :=
  ∀ k m a b segs, (c.body.nodes[k]!).form = .scan → m < (c.body.nodes[k]!).members →
    (a < b → b ≤ (c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.count) →
    Desc.carry c fuel k m a b = .ok segs →
    Agree c.P sc (segsPos segs) ((List.range' a (b - a)).map fun j =>
      Flock.Program.resolve c.P (c.P.child sc k m) ((c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.get j))

theorem downSpec_gate (c : Desc.Ctx) (sc : Flock.Program.Scope) (fuel : Nat) (q : Piece) {w : Nat}
    (ht : q.tgt = .gate w) {segs : Array Seg} (h : Desc.down c (fuel + 1) q = .ok segs) :
    Agree c.P sc (segsPos segs) ((pos q).map (den c.P sc)) := by
  rw [down_gate c fuel q ht] at h
  injection h with h
  subst h
  rw [segsPos_single]
  exact agree_refl _ _ _

theorem downSpec_param (c : Desc.Ctx) (sc : Flock.Program.Scope) (fuel : Nat) (q : Piece) {i : Nat}
    (ht : q.tgt = .ref true i) {segs : Array Seg} (h : Desc.down c (fuel + 1) q = .ok segs) :
    Agree c.P sc (segsPos segs) ((pos q).map (den c.P sc)) := by
  rw [down_param c fuel q ht] at h
  injection h with h
  subst h
  rw [segsPos_single]
  exact agree_refl _ _ _

theorem downSpec_count0 (c : Desc.Ctx) (sc : Flock.Program.Scope) (fuel : Nat) (q : Piece) {k : Nat}
    (ht : q.tgt = .ref false k) (h0 : q.count = 0) {segs : Array Seg} (h : Desc.down c (fuel + 1) q = .ok segs) :
    Agree c.P sc (segsPos segs) ((pos q).map (den c.P sc)) := by
  rw [down_count0 c fuel q ht h0] at h
  injection h with h
  subst h
  have : pos q = [] := List.eq_nil_of_length_eq_zero (by rw [pos_length, h0])
  rw [this]
  exact List.Forall₂.nil

theorem downSpec_prim (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (fuel : Nat) (q : Piece) {k : Nat}
    (ht : q.tgt = .ref false k) (hff : q.flat = q) {node : Flock.Program.Node} (hn : c.node k = .ok node)
    (hform : node.form = .prim) (hq : ∀ x ∈ pos q, 0 ≤ x.2) {segs : Array Seg} (h : Desc.down c (fuel + 1) q = .ok segs) :
    Agree c.P sc (segsPos segs) ((pos q).map (den c.P sc)) := by
  by_cases h0 : q.count = 0
  · exact downSpec_count0 c sc fuel q ht h0 h
  cases hr : (c.P.defs[node.fn]!).ret with
  | value w =>
    rw [down_prim c fuel q ht hff hn hform hr] at h
    injection h with h
    subst h
    exact agree_of_map (den_prim c sc hd q ht hn hform hq w)
  | _ => exact absurd h (down_prim_ret c fuel q ht hff hn (Or.inl hform) h0 (fun w e => by rw [hr] at e; cases e) segs)

theorem downSpec_batchPrim (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (fuel : Nat) (q : Piece)
    {k : Nat} (ht : q.tgt = .ref false k) (hff : q.flat = q) {node : Flock.Program.Node} (hn : c.node k = .ok node)
    (hform : node.form = .batch) (hp : (c.P.defs[node.fn]!).prim = true) (hL : (c.P.defs[node.fn]!).outLeaves = 1)
    (hq : ∀ x ∈ pos q, 0 ≤ x.2) {segs : Array Seg} (h : Desc.down c (fuel + 1) q = .ok segs) :
    Agree c.P sc (segsPos segs) ((pos q).map (den c.P sc)) := by
  by_cases h0 : q.count = 0
  · exact downSpec_count0 c sc fuel q ht h0 h
  cases hr : (c.P.defs[node.fn]!).ret with
  | value w =>
    rw [down_batchPrim c fuel q ht hff hn hform hp hr] at h
    injection h with h
    subst h
    exact agree_of_map (den_batchPrim c sc hd q ht hn hform hp hL hq w)
  | _ =>
    exact absurd h
      (down_prim_ret c fuel q ht hff hn (Or.inr ⟨hform, hp⟩) h0 (fun w e => by rw [hr] at e; cases e) segs)

/-- A piece's rows lay its positions end to end. -/
theorem posAll_rowsOf (q : Piece) : posAll q.rowsOf = pos q := by
  unfold posAll Desc.Piece.rowsOf pos
  rw [Array.toList_map, Array.toList_range, List.flatMap_map]
  apply flatMap_congr
  intro r _
  simp [Desc.Piece.row]

theorem byRows_agree (c : Desc.Ctx) (sc : Flock.Program.Scope) (fuel : Nat) (hD : DownSpec c sc fuel) (q : Piece)
    (hq : ∀ x ∈ pos q, 0 ≤ x.2) (hf : ∀ x ∈ pos q, Fits c x) {segs : Array Seg} (h : byRows c fuel q = .ok segs) :
    Agree c.P sc (segsPos segs) ((pos q).map (den c.P sc)) := by
  unfold byRows at h
  have h := walkAtMost_bind _ _ _ h
  dsimp only at h
  rw [← Array.forIn_toList] at h
  have hm : ∀ x ∈ q.rowsOf.toList, ∀ y ∈ pos x, y ∈ pos q := fun x hx y hy => by
    rw [← posAll_rowsOf q]
    exact List.mem_flatMap.2 ⟨x, hx, hy⟩
  have := loop_agree (P := c.P) (sc := sc) (Desc.down c fuel) (fun x => (pos x).map (den c.P sc)) q.rowsOf.toList
    (fun x hx segs hs => hD x segs (fun y hy => hq y (hm x hx y hy)) (fun y hy => hf y (hm x hx y hy)) hs) segs h
  rw [← posAll_rowsOf q, posAll, List.map_flatMap]
  exact this

theorem rows_one {q : Piece} (h0 : q.count ≠ 0) (hr : ¬ q.rows > 1) : q.rows = 1 := by
  unfold Desc.Piece.count at h0
  rcases Nat.lt_or_ge q.rows 1 with h | h
  · rw [show q.rows = 0 by omega, Nat.zero_mul] at h0; exact absurd rfl h0
  · omega

theorem ppt_one {q : Piece} (hr : q.rows = 1) {t : Nat} (ht : t < q.count) : ppt q t = q.at t := by
  have hl : t < q.len := by unfold Desc.Piece.count at ht; rw [hr, Nat.one_mul] at ht; exact ht
  unfold ppt Desc.Piece.at
  rw [Nat.div_eq_of_lt hl, Nat.mod_eq_of_lt hl]
  simp

theorem count_one {q : Piece} (hr : q.rows = 1) : q.count = q.len := by
  unfold Desc.Piece.count; rw [hr, Nat.one_mul]

theorem at_mem_pos {q : Piece} (hr : q.rows = 1) {t : Nat} (ht : t < q.len) : (q.tgt, q.at t) ∈ pos q := by
  rw [pos_eq_map]
  have ht' : t < q.count := by rw [count_one hr]; exact ht
  exact List.mem_map.2 ⟨t, List.mem_range.2 ht', by rw [ppt_one hr ht']⟩

/-- The member activation `Prog.child` names, the body's node `k` being `node`. -/
theorem child_eq (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) {k : Nat} {node : Flock.Program.Node}
    (hn : c.node k = .ok node) (m : Nat) :
    c.P.child sc k m = .child sc node.fn (sc.off + (c.P.defs[sc.d]!).body.offsets[k]! + m * (c.P.defs[node.fn]!).gates)
      k m := by
  have hnode : (c.P.defs[sc.d]!).body.nodes[k]! = node := by rw [hd]; exact node_get hn
  unfold Flock.Program.Prog.child
  dsimp only
  rw [hnode]

theorem members_call {node : Flock.Program.Node} (h : node.form = .call) : node.members = 1 := by
  unfold Flock.Program.Node.members; rw [h]; rfl

theorem members_scan {node : Flock.Program.Node} (h : node.form = .scan) : node.members = node.n := by
  unfold Flock.Program.Node.members; rw [h]; rfl

theorem callee_eq (c : Desc.Ctx) {k : Nat} {node : Flock.Program.Node} (hn : c.node k = .ok node) :
    c.P.defs[(c.body.nodes[k]!).fn]! = c.P.defs[node.fn]! := by
  rw [node_get hn]

/-- A Call's piece of one row and stride 1, from what `fromCallee` keeps. -/
theorem downSpec_callOne (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (fuel : Nat)
    (hC : CalleeSpec c sc fuel) (q : Piece) {k : Nat} (ht : q.tgt = .ref false k) (hff : q.flat = q)
    {node : Flock.Program.Node} (hn : c.node k = .ok node) (hform : node.form = .call) (h0 : q.count ≠ 0) (hr : ¬ q.rows > 1)
    (hs : q.stride = 1) (hq : ∀ x ∈ pos q, 0 ≤ x.2) (hf : ∀ x ∈ pos q, Fits c x)
    (hdo : Resolve.DownOk node (c.P.defs[node.fn]!)) {segs : Array Seg} (h : Desc.down c (fuel + 1) q = .ok segs) :
    Agree c.P sc (segsPos segs) ((pos q).map (den c.P sc)) := by
  rw [down_call_one c fuel q ht hff hn hform h0 hr hs] at h
  have h1 := rows_one h0 hr
  have hlen : 0 < q.len := by rw [← count_one h1]; omega
  have hb : 0 ≤ q.base := by
    have := hq _ (at_mem_pos h1 hlen)
    simpa [Desc.Piece.at] using this
  have hfl := hf _ (at_mem_pos h1 (t := q.len - 1) (by omega)) k ht
  simp only [Desc.Piece.at, hs, node_get hn] at hfl
  have hdk := hdo ((q.base + ((q.len - 1 : Nat) : Int) * 1).toNat) (by omega)
  simp only [hform] at hdk
  have hA := hC k 0 q.base.toNat (q.base.toNat + q.len) segs
    (by rw [node_get hn, members_call hform]; omega) (fun _ => by rw [callee_eq c hn]; omega) h
  apply agree_weaken hA
  rw [pos_eq_map, List.map_map, count_one h1, Nat.add_sub_cancel_left, List.range'_eq_map_range, List.map_map]
  apply forall₂_range_map
  intro t ht' a ha
  have hp := ppt_one h1 (t := t) (by rw [count_one h1]; exact ht')
  simp only [Function.comp, den, ht, Desc.Piece.at, hs] at ha hp
  rw [hp, if_neg (by push_cast; omega)] at ha
  have hnode : (c.P.defs[sc.d]!).body.nodes[k]! = node := by rw [hd]; exact node_get hn
  have e := den_call c.P sc hnode hform ha
  simp only [Function.comp]
  rw [child_eq c sc hd hn, callee_eq c hn, Nat.zero_mul, Nat.add_zero,
    show q.base.toNat + t = (q.base + (t : Int) * 1).toNat by omega]
  exact e

/-- A Call's piece of one row and another stride, from what `fromCallee` keeps at each position. -/
theorem downSpec_callPts (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (fuel : Nat)
    (hC : CalleeSpec c sc fuel) (q : Piece) {k : Nat} (ht : q.tgt = .ref false k) (hff : q.flat = q)
    {node : Flock.Program.Node} (hn : c.node k = .ok node) (hform : node.form = .call) (h0 : q.count ≠ 0) (hr : ¬ q.rows > 1)
    (hs : q.stride ≠ 1) (hq : ∀ x ∈ pos q, 0 ≤ x.2) (hf : ∀ x ∈ pos q, Fits c x)
    (hdo : Resolve.DownOk node (c.P.defs[node.fn]!)) {segs : Array Seg} (h : Desc.down c (fuel + 1) q = .ok segs) :
    Agree c.P sc (segsPos segs) ((pos q).map (den c.P sc)) := by
  rw [down_call_pts c fuel q ht hff hn hform h0 hr hs] at h
  unfold callPts at h
  dsimp only at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at h
  rw [range'_size] at h
  have h1 := rows_one h0 hr
  have hA := loop_agree (P := c.P) (sc := sc)
    (fun t => Desc.fromCallee c fuel k 0 (q.at t).toNat ((q.at t).toNat + 1))
    (fun t => [Flock.Program.resolve c.P (c.P.child sc k 0)
      ((c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.get (q.at t).toNat)])
    (List.range q.len) (fun t htm segs hs => by
      have hm := at_mem_pos h1 (List.mem_range.1 htm)
      have hb := hq _ hm
      have hfl := hf _ hm k ht
      simp only [node_get hn] at hb hfl
      have hdk := hdo (q.at t).toNat (by omega)
      simp only [hform] at hdk
      have := hC k 0 (q.at t).toNat ((q.at t).toNat + 1) segs
        (by rw [node_get hn, members_call hform]; omega) (fun _ => by rw [callee_eq c hn]; omega) hs
      simpa using this) segs h
  apply agree_weaken hA
  rw [← List.map_eq_flatMap, pos_eq_map, List.map_map, count_one h1]
  apply forall₂_range_map
  intro t ht' a ha
  have hp := ppt_one h1 (t := t) (by rw [count_one h1]; exact ht')
  have hnn := hq (q.tgt, ppt q t)
    (by rw [pos_eq_map]; exact List.mem_map.2 ⟨t, List.mem_range.2 (by rw [count_one h1]; exact ht'), rfl⟩)
  simp only [Function.comp, den, ht] at ha
  simp only at hnn
  rw [if_neg (by omega), hp] at ha
  have hnode : (c.P.defs[sc.d]!).body.nodes[k]! = node := by rw [hd]; exact node_get hn
  have e := den_call c.P sc hnode hform ha
  rw [child_eq c sc hd hn, callee_eq c hn, Nat.zero_mul, Nat.add_zero]
  exact e

/-! ## `down`'s step, every case

`down` reads a piece as one row where its rows are consecutive (`down_flat`), so its step is taken on such a piece.
A batch of a composite's piece and a scan's piece of one row are left to their own statements (`BatchSpec`,
`ScanSpec`). Past them, `down` one step up keeps the gates its piece names wherever `down` and `fromCallee` do
(`downSpec_succ`), in a body whose node's references `Resolve.BodyOk` keeps in range, every primitive returning one
leaf. -/

/-- What `down` keeps of a batch of a composite's piece of positive count read as one row where it can be, one step
up. -/
def BatchSpec (c : Desc.Ctx) (sc : Flock.Program.Scope) (fuel : Nat) : Prop :=
  ∀ q segs k node, q.tgt = .ref false k → c.node k = .ok node → node.form = .batch →
    (c.P.defs[node.fn]!).prim = false → q.count ≠ 0 → q.flat = q → (∀ x ∈ pos q, 0 ≤ x.2) →
    (∀ x ∈ pos q, Fits c x) → Desc.down c (fuel + 1) q = .ok segs →
    Agree c.P sc (segsPos segs) ((pos q).map (den c.P sc))

/-- **What `down` keeps of a scan's piece of one row**, one step up: the final carry's leaves through `carry`, the
`ys` through `fromCallee` iteration by iteration, together or position by position (`scanSpec_of`). -/
def ScanSpec (c : Desc.Ctx) (sc : Flock.Program.Scope) (fuel : Nat) : Prop :=
  ∀ q segs k node (carry y : Flock.Program.Ty), q.tgt = .ref false k → c.node k = .ok node → node.form = .scan →
    node.n ≠ 0 → q.count ≠ 0 → q.flat = q → ¬ q.rows > 1 → (c.P.defs[node.fn]!).ret = .tuple #[carry, y] →
    (∀ x ∈ pos q, 0 ≤ x.2) → (∀ x ∈ pos q, Fits c x) → Desc.down c (fuel + 1) q = .ok segs →
    Agree c.P sc (segsPos segs) ((pos q).map (den c.P sc))

theorem node_lt {c : Desc.Ctx} {k : Nat} {node : Flock.Program.Node} (hn : c.node k = .ok node) :
    k < c.body.nodes.size := by
  unfold Desc.Ctx.node at hn
  rcases h : c.body.nodes[k]? with _ | n
  · rw [h] at hn; cases hn
  · exact (Array.getElem?_eq_some_iff.1 h).1

/-- A node of a body `Resolve.BodyOk` holds reads its output leaves in range. -/
theorem downOk_of {c : Desc.Ctx} (hB : Resolve.BodyOk c.P c.d) {k : Nat} {node : Flock.Program.Node}
    (hn : c.node k = .ok node) : Resolve.DownOk node (c.P.defs[node.fn]!) := by
  have := (hB.1 k (node_lt hn)).2.2.2.1
  have e : (c.P.defs[c.d]!).body.nodes[k]! = node := node_get hn
  rwa [e] at this

/-- **`down`'s step**: one step up, `down` keeps the gates its piece names wherever `down`, `fromCallee`, a batch of
a composite's case and a scan's case of one row do. -/
theorem downSpec_succ (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (hB : Resolve.BodyOk c.P c.d)
    (hP1 : ∀ f : Nat, (c.P.defs[f]!).prim = true → (c.P.defs[f]!).outLeaves = 1) (fuel : Nat)
    (hD : DownSpec c sc fuel) (hC : CalleeSpec c sc fuel) (hBatch : BatchSpec c sc fuel)
    (hScan : ScanSpec c sc fuel) : DownSpec c sc (fuel + 1) := by
  intro q segs hq hf h
  cases ht : q.tgt with
  | gate w => exact downSpec_gate c sc fuel q ht h
  | ref b k =>
  cases b with
  | true => exact downSpec_param c sc fuel q ht h
  | false =>
  by_cases h0 : q.count = 0
  · exact downSpec_count0 c sc fuel q ht h0 h
  rw [down_flat c fuel q ht] at h
  rw [← pos_flat q] at hq hf ⊢
  have ht' : q.flat.tgt = .ref false k := by rw [flat_tgt, ht]
  have h0' : q.flat.count ≠ 0 := by rw [flat_count]; exact h0
  have hff := flat_flat q
  generalize q.flat = p at ht' h0' hff hq hf h ⊢
  rcases hn : c.node k with e | node
  · exact absurd h (down_node c fuel p ht' h0' hn segs)
  have hdo := downOk_of hB hn
  cases hform : node.form with
  | prim => exact downSpec_prim c sc hd fuel p ht' hff hn hform hq h
  | call =>
    by_cases hr : p.rows > 1
    · rw [down_rows c fuel p ht' hff hn (Or.inl hform) h0' hr] at h
      exact byRows_agree c sc fuel hD p hq hf h
    by_cases hs : p.stride = 1
    · exact downSpec_callOne c sc hd fuel hC p ht' hff hn hform h0' hr hs hq hf hdo h
    · exact downSpec_callPts c sc hd fuel hC p ht' hff hn hform h0' hr hs hq hf hdo h
  | scan =>
    by_cases hn0 : node.n = 0
    · exact absurd h (down_scan0 c fuel p ht' hff hn hform hn0 h0' segs)
    by_cases hr : p.rows > 1
    · rw [down_rows c fuel p ht' hff hn (Or.inr ⟨hform, hn0⟩) h0' hr] at h
      exact byRows_agree c sc fuel hD p hq hf h
    by_cases hex : ∃ carry y, (c.P.defs[node.fn]!).ret = .tuple #[carry, y]
    · obtain ⟨carry, y, hret⟩ := hex
      exact hScan p segs k node carry y ht' hn hform hn0 h0' hff hr hret hq hf h
    · exact absurd h
        (down_scan_ret c fuel p ht' hff hn hform hn0 h0' hr (fun carry y e => hex ⟨carry, y, e⟩) segs)
  | batch =>
    by_cases hp : (c.P.defs[node.fn]!).prim = true
    · exact downSpec_batchPrim c sc hd fuel p ht' hff hn hform hp (hP1 _ hp) hq h
    · exact hBatch p segs k node ht' hn hform (by simpa using hp) h0' hff hq hf h


/-! ## `fromCallee`'s step

A member's activation's positions read in its parent: a gate of the member is its parent's gate past the member's
offset, so a gate position moved by that offset names in the parent what it names in the member (`agree_shift`); and
what names in the parent what a member's positions name, where those name its return leaves, names those leaves
(`agree_trans`). -/

theorem agree_trans {P : Prog} {sc sc' : Flock.Program.Scope} {ys xs : List (Tgt × Int)}
    {vs : List (Except String Nat)} (h1 : Agree P sc ys (xs.map (den P sc'))) (h2 : Agree P sc' xs vs) :
    Agree P sc ys vs := by
  induction h2 generalizing ys with
  | nil => cases h1; exact List.Forall₂.nil
  | cons hx _ ih =>
    rw [List.map_cons] at h1
    cases h1 with
    | cons hy h1' => exact List.Forall₂.cons (fun a ha => hy a (hx a ha)) (ih h1')

theorem sl_map_range' {α : Type} (f : Nat → α) (n a b : Nat) (hb : a < b → b ≤ n) :
    sl ((List.range n).map f) a b = (List.range' a (b - a)).map f := by
  by_cases h : a < b
  · exact sl_map_range f n a b (hb h)
  · rw [sl_empty _ _ _ (Or.inl (by omega)), show b - a = 0 by omega]; rfl

theorem agree_sl {P : Prog} {sc : Flock.Program.Scope} {xs : List (Tgt × Int)} {vs : List (Except String Nat)}
    (h : Agree P sc xs vs) (a b : Nat) : Agree P sc (sl xs a b) (sl vs a b) :=
  List.forall₂_take _ (List.forall₂_drop _ h)

/-- A loop each of whose steps appends segments agreeing with its element's reads appends their concatenation. -/
theorem forIn_ext {β : Type} {P : Prog} {sc : Flock.Program.Scope} (T : β → List (Except String Nat))
    (f : β → Array Seg → Except String (ForInStep (Array Seg))) :
    ∀ (l : List β), (∀ x ∈ l, ∀ r y, f x r = .ok y →
      ∃ segs, y = .yield (r ++ segs) ∧ Agree P sc (segsPos segs) (T x)) →
      ∀ (acc out : Array Seg), forIn l acc f = .ok out →
        ∃ segs, out = acc ++ segs ∧ Agree P sc (segsPos segs) (l.flatMap T)
  | [], _, acc, out, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    exact ⟨#[], by simp [h], List.Forall₂.nil⟩
  | x :: l, hF, acc, out, h => by
    rw [List.forIn_cons] at h
    rcases hx : f x acc with e | y <;> rw [hx] at h
    · exact absurd h (by intro h; cases h)
    obtain ⟨segs, rfl, hs⟩ := hF x List.mem_cons_self acc _ hx
    obtain ⟨segs', h1, h2⟩ := forIn_ext T f l (fun y hy => hF y (List.mem_cons_of_mem _ hy)) (acc ++ segs) out h
    refine ⟨segs ++ segs', by rw [h1, Array.append_assoc], ?_⟩
    rw [segsPos_append, List.flatMap_cons]
    exact agree_append hs h2

/-- A loop each of whose steps appends segments `R` holds of appends only such segments. -/
theorem forIn_src {β : Type} (R : Seg → Prop) (f : β → Array Seg → Except String (ForInStep (Array Seg))) :
    ∀ (l : List β), (∀ x ∈ l, ∀ r y, f x r = .ok y → ∃ segs, y = .yield (r ++ segs) ∧ ∀ s ∈ segs.toList, R s) →
      ∀ (acc out : Array Seg), forIn l acc f = .ok out → ∀ s ∈ out.toList, s ∈ acc.toList ∨ R s
  | [], _, acc, out, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact fun s hs => Or.inl hs
  | x :: l, hF, acc, out, h => by
    rw [List.forIn_cons] at h
    rcases hx : f x acc with e | y <;> rw [hx] at h
    · exact absurd h (by intro h; cases h)
    obtain ⟨segs, rfl, hs⟩ := hF x List.mem_cons_self acc _ hx
    intro s hso
    rcases forIn_src R f l (fun y hy => hF y (List.mem_cons_of_mem _ hy)) (acc ++ segs) out h s hso with h' | h'
    · rw [Array.toList_append, List.mem_append] at h'
      exact h'.imp id (hs s)
    · exact Or.inr h'

theorem segPos_shift (s : Seg) (o : Int) :
    segPos { ps := Desc.shiftGates o s.ps, len := s.len, reps := s.reps, shift := s.shift } =
      (segPos s).map (shiftPos o) := by
  unfold segPos
  rw [List.map_flatMap]
  apply flatMap_congr
  intro m _
  rw [copy_pos, copy_pos]
  simp only [shiftGates_pos, List.map_map, shiftPos_shiftPos]
  rw [Int.add_comm]

/-- Gate positions at or past 0, moved by a member's offset `o`, name in the parent what they name in the member. -/
theorem agree_shift (P : Prog) {sc sc' : Flock.Program.Scope} {o : Int} (ho : 0 ≤ o)
    (hoff : sc'.off = sc.off + o.toNat) (xs : List (Tgt × Int))
    (hx : ∀ x ∈ xs, (∃ w, x.1 = .gate w) ∧ 0 ≤ x.2) :
    Agree P sc (xs.map (shiftPos o)) (xs.map (den P sc')) := by
  induction xs with
  | nil => exact List.Forall₂.nil
  | cons x xs ih =>
    rw [List.map_cons, List.map_cons]
    refine List.Forall₂.cons ?_ (ih fun y hy => hx y (List.mem_cons_of_mem _ hy))
    obtain ⟨⟨w, hw⟩, h0⟩ := hx x List.mem_cons_self
    obtain ⟨t, p⟩ := x
    simp only at hw h0
    subst hw
    intro a ha
    simp only [shiftPos, den] at ha ⊢
    rw [if_neg (by omega)] at ha ⊢
    rw [← ha, hoff]
    congr 1
    omega


/-- A segment of `Seg.slice` has a copy at least, and more than one only as copies of `s`'s pieces moved, `s` of more
than one copy. -/
theorem segSlice_reps (s : Seg) (a b : Nat) (g : Seg) (hg : g ∈ (s.slice a b).toList) :
    1 ≤ g.reps ∧ (g.reps > 1 → s.reps > 1 ∧ g.shift = s.shift ∧ ∃ δ, g.ps = Desc.shiftGates δ s.ps) := by
  have hone : ∀ (ps : Array Piece), g ∈ (ps.map Seg.one).toList →
      1 ≤ g.reps ∧ (g.reps > 1 → s.reps > 1 ∧ g.shift = s.shift ∧ ∃ δ, g.ps = Desc.shiftGates δ s.ps) := by
    intro ps h
    rw [Array.toList_map, List.mem_map] at h
    obtain ⟨q, -, rfl⟩ := h
    exact ⟨Nat.le_refl _, fun h => absurd h (by simp [Seg.one])⟩
  unfold Desc.Seg.slice at hg
  split at hg
  · simp at hg
  split at hg
  · exact hone _ hg
  rename_i h2
  dsimp only at hg
  split at hg
  · exact hone _ hg
  simp only [Array.toList_append, List.mem_append] at hg
  rcases hg with (hg | hg) | hg
  · split at hg
    · simp at hg
    · exact hone _ hg
  · generalize (if (a % s.len == 0) = true then a / s.len else a / s.len + 1) = lo at hg
    split at hg
    · rename_i hlt
      simp only [List.mem_singleton] at hg
      subst hg
      exact ⟨by dsimp only; omega, fun _ => ⟨by omega, rfl, _, rfl⟩⟩
    · simp at hg
  · exact hone _ hg

/-- A segment `psGo` adds is one `Seg.slice` gives of a segment of `s`. -/
theorem psGo_mem (s : PS) (lo hi : Nat) : ∀ (l : List Nat) (out : Array Seg) (g : Seg),
    g ∈ (psGo s lo hi l out).toList → g ∈ out.toList ∨ ∃ i a b : Nat, g ∈ (Desc.Seg.slice s.segs[i]! a b).toList
  | [], out, g, h => Or.inl h
  | i :: is, out, g, h => by
    unfold psGo at h
    split at h
    · exact Or.inl h
    rcases psGo_mem s lo hi is _ g h with h | h
    · split at h
      · simp only [Array.toList_append, List.mem_append] at h
        rcases h with h | h
        · exact Or.inl h
        · exact Or.inr ⟨i, _, _, h⟩
      · exact Or.inl h
    · exact Or.inr h

/-- A segment of `PS.slice` is one `Seg.slice` gives of a segment of `s`. -/
theorem psSlice_mem (s : PS) (lo hi : Nat) (g : Seg) (hg : g ∈ (s.slice lo hi).toList) :
    ∃ i a b : Nat, g ∈ (Desc.Seg.slice s.segs[i]! a b).toList := by
  unfold Desc.PS.slice at hg
  simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at hg
  split at hg
  · change g ∈ ([] : List Seg) at hg
    cases hg
  generalize (forIn (m := Id) (List.range' 0 _) (0, s.segs.size) _ : Id (Nat × Nat)) = B at hg
  rw [bind_id, psGo_eq, bind_id] at hg
  change g ∈ (psGo s lo hi _ #[]).toList at hg
  rcases psGo_mem s lo hi _ #[] g hg with h | h
  · simp at h
  · exact h


theorem mem_pos_tgt {q : Piece} {x : Tgt × Int} (h : x ∈ pos q) : x.1 = q.tgt := by
  unfold pos at h
  simp only [List.mem_flatMap, List.mem_map, List.mem_range] at h
  obtain ⟨_, _, _, _, rfl⟩ := h
  rfl

theorem shiftGates_tgt {δ : Int} {ps : Array Piece} {q : Piece} (h : q ∈ (Desc.shiftGates δ ps).toList) :
    ∃ g ∈ ps.toList, q.tgt = g.tgt := by
  unfold Desc.shiftGates at h
  rw [Array.toList_map, List.mem_map] at h
  obtain ⟨g, hg, rfl⟩ := h
  refine ⟨g, hg, ?_⟩
  split <;> rfl

/-- A segment's positions name its pieces' targets. -/
theorem segPos_tgt {s : Seg} {x : Tgt × Int} (h : x ∈ segPos s) : ∃ g ∈ s.ps.toList, x.1 = g.tgt := by
  unfold segPos at h
  obtain ⟨m, -, h⟩ := List.mem_flatMap.1 h
  unfold posAll at h
  obtain ⟨q, hq, h⟩ := List.mem_flatMap.1 h
  obtain ⟨g, hg, e⟩ := shiftGates_tgt hq
  exact ⟨g, hg, (mem_pos_tgt h).trans e⟩

theorem segPos_one_rep {s : Seg} (h : s.reps = 1) : segPos s = posAll s.ps := by
  unfold segPos
  rw [h, List.range_one, List.flatMap_singleton, copy_pos]
  simp [shiftPos_zero]

/-- Definition `f`'s summarized return (`memo`'s entry), read in any activation of `f`: segments as `PS.of` makes them,
positions at or past 0 naming `f`'s gates or its parameters' leaves in range, a segment of more than one copy gates
only or copies all alike (`shift = 0`, as `down` gives a batch's members returning what they share), and position by
position the gate `f`'s return leaf resolves to. -/
def RetOk (P : Prog) (memo : Array Desc.Summary) (f : Nat) : Prop :=
  PSOk (memo[f]!).ret ∧
  (∀ x ∈ psPos (memo[f]!).ret, 0 ≤ x.2 ∧ (∀ k, x.1 ≠ .ref false k) ∧
    ∀ i, x.1 = .ref true i → i < (P.defs[f]!).body.params.size ∧
      x.2 < (((P.defs[f]!).body.params[i]!).2.leaves : Int)) ∧
  (∀ s ∈ (memo[f]!).ret.segs.toList, s.reps > 1 → (∀ g ∈ s.ps.toList, ∃ w, g.tgt = .gate w) ∨ s.shift = 0) ∧
  ∀ sc : Flock.Program.Scope, sc.d = f →
    Agree P sc (psPos (memo[f]!).ret) ((List.range (P.defs[f]!).body.ret.count).map fun j =>
      Flock.Program.resolve P sc ((P.defs[f]!).body.ret.get j))

/-- What `throughArg` keeps of member `m`'s parameter `i`'s leaves `g`, each in range: the gate each names in the
member's activation. -/
def ThroughSpec (c : Desc.Ctx) (sc : Flock.Program.Scope) (fuel : Nat) : Prop :=
  ∀ k m i g segs, m < (c.body.nodes[k]!).members → g.tgt = .ref true i →
    (∀ x ∈ pos g, 0 ≤ x.2 ∧ i < (c.P.defs[(c.body.nodes[k]!).fn]!).body.params.size ∧
      x.2 < ((((c.P.defs[(c.body.nodes[k]!).fn]!).body.params[i]!).2.leaves : Int))) →
    Desc.throughArg c fuel k m i g = .ok segs →
    Agree c.P sc (segsPos segs) ((pos g).map (den c.P (c.P.child sc k m)))

/-- A segment of a slice of a summary `RetOk` holds: of one copy at least, more than one only of gates or all alike. -/
theorem retOk_slice {memo : Array Desc.Summary} {f : Nat} (hgo : ∀ s ∈ (memo[f]!).ret.segs.toList,
    s.reps > 1 → (∀ g ∈ s.ps.toList, ∃ w, g.tgt = .gate w) ∨ s.shift = 0) {a b : Nat} {s : Seg}
    (hs : s ∈ ((memo[f]!).ret.slice a b).toList) :
    1 ≤ s.reps ∧ (s.reps > 1 → (∀ g ∈ s.ps.toList, ∃ w, g.tgt = .gate w) ∨ s.shift = 0) := by
  obtain ⟨i, a', b', hi⟩ := psSlice_mem _ a b s hs
  obtain ⟨h1, h2⟩ := segSlice_reps _ a' b' s hi
  refine ⟨h1, fun hr => ?_⟩
  obtain ⟨hr', hsh, δ, e⟩ := h2 hr
  have hlt : i < (memo[f]!).ret.segs.size := by
    refine (Nat.lt_or_ge i _).resolve_right fun hge => ?_
    rw [getElem!_neg _ i (by omega)] at hr'
    exact absurd hr' (by decide)
  rw [getElem!_pos _ i hlt] at hr' hsh e
  rcases hgo _ (Array.getElem_mem_toList hlt) hr' with hg | hg
  · refine Or.inl fun g hgm => ?_
    rw [e] at hgm
    obtain ⟨g0, hg0, et⟩ := shiftGates_tgt hgm
    obtain ⟨w, hw⟩ := hg g0 hg0
    exact ⟨w, et.trans hw⟩
  · exact Or.inr (hsh.trans hg)


/-- Pieces `fromCallee` tests for naming gates only do. -/
theorem gates_of_all {ps : Array Piece} (h : (ps.all fun g => match g.tgt with
      | .gate _ => true
      | _ => false) = true) : ∀ g ∈ ps.toList, ∃ w, g.tgt = .gate w := by
  rw [Array.all_eq_true_iff_forall_mem] at h
  intro g hg
  have := h g (Array.mem_toList_iff.1 hg)
  revert this
  cases g.tgt with
  | gate w => intro _; exact ⟨w, rfl⟩
  | ref b i => intro h; cases h

/-- A segment of copies all alike names its pieces' positions once per copy. -/
theorem segPos_alike {s : Seg} (h : s.shift = 0) : segPos s = (List.range s.reps).flatMap fun _ => posAll s.ps := by
  unfold segPos
  apply flatMap_congr
  intro m _
  rw [copy_pos, h, Int.mul_zero, shiftPos_zero, List.map_id]

/-- Segments of one copy each, their pieces laid end to end, name their positions. -/
theorem posAll_flatMap_ps (ss : Array Seg) (h : ∀ x ∈ ss.toList, x.reps = 1) :
    posAll (ss.flatMap (·.ps)) = segsPos ss := by
  unfold posAll segsPos
  rw [Array.toList_flatMap, List.flatMap_assoc]
  apply flatMap_congr
  intro x hx
  rw [segPos_one_rep (h x hx), posAll]

/-- Every segment `throughArg` gives has a copy at least, as each one `down`, `fromCallee` and `carry` make does
(`Seg.one`, a slice of a summary's, or copies of more than one): `throughReps`. -/
def ThroughReps (c : Desc.Ctx) : Prop :=
  ∀ fuel k m i g segs, Desc.throughArg c fuel k m i g = .ok segs → ∀ x ∈ segs.toList, x.reps ≠ 0

/-- **`fromCallee`'s step**: one step up, `fromCallee` keeps the gates its callee's return leaves resolve to wherever
its callees' summaries hold `RetOk` and `throughArg` one step down keeps what its parameters' leaves name, each of its
segments of a copy at least (`ThroughReps`). -/
theorem calleeSpec_succ (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (fuel : Nat)
    (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn) (hT : ThroughSpec c sc fuel)
    (hTR : ThroughReps c) : CalleeSpec c sc (fuel + 1) := by
  intro k m a b segs hm hb h
  rw [Desc.fromCallee.eq_2] at h
  obtain ⟨node, hn, h⟩ := Walk.except_bind_ok h
  dsimp only at h
  have h' := ok_bind_pure h
  rw [← Array.forIn_toList] at h'
  obtain ⟨hps, hpos, hgo, hag⟩ := hR k node hn
  have hnode : c.body.nodes[k]! = node := node_get hn
  rw [hnode] at hm hb
  rw [hnode]
  have hsc := child_eq c sc hd hn m
  have hd' : (c.P.child sc k m).d = node.fn := by rw [hsc]; rfl
  have ho : (0 : Int) ≤ (c.body.offsets[k]! : Int) + (m : Int) * ((c.P.defs[node.fn]!).gates : Int) := by positivity
  have hoff : (c.P.child sc k m).off =
      sc.off + ((c.body.offsets[k]! : Int) + (m : Int) * ((c.P.defs[node.fn]!).gates : Int)).toNat := by
    rw [hsc, show ((c.body.offsets[k]! : Int) + (m : Int) * ((c.P.defs[node.fn]!).gates : Int)) =
      ((c.body.offsets[k]! + m * (c.P.defs[node.fn]!).gates : Nat) : Int) by push_cast; rfl, Int.toNat_natCast,
      Nat.add_assoc, hd]
    rfl
  have hslice := psSlice_pos _ hps a b
  have hmemS : ∀ s ∈ ((c.memo[node.fn]!).ret.slice a b).toList, ∀ x ∈ segPos s,
      x ∈ psPos (c.memo[node.fn]!).ret := by
    intro s hs x hx
    have : x ∈ segsPos ((c.memo[node.fn]!).ret.slice a b) := List.mem_flatMap.2 ⟨s, hs, hx⟩
    rw [hslice] at this
    exact List.mem_of_mem_drop (List.mem_of_mem_take this)
  obtain ⟨out, hout, hA⟩ : ∃ out, segs = #[] ++ out ∧ Agree c.P sc (segsPos out)
      (((c.memo[node.fn]!).ret.slice a b).toList.flatMap fun s => (segPos s).map (den c.P (c.P.child sc k m))) := by
    refine forIn_ext (P := c.P) (sc := sc) (fun s => (segPos s).map (den c.P (c.P.child sc k m))) _ _ ?_ #[] segs h'
    intro s hs r y hy
    obtain ⟨hrep1, hgates⟩ := retOk_slice hgo hs
    have hx := hmemS s hs
    split at hy
    · rename_i hr
      split at hy
      · rename_i hall
        have hgs := gates_of_all hall
        simp only [pure, Except.pure, Except.ok.injEq] at hy
        subst hy
        refine ⟨#[_], congrArg _ Array.push_eq_append, ?_⟩
        rw [segsPos_singleton, segPos_shift]
        exact agree_shift c.P ho hoff _ fun x hxm => by
          obtain ⟨g, hg, e⟩ := segPos_tgt hxm
          obtain ⟨w, hw⟩ := hgs g hg
          exact ⟨⟨w, e.trans hw⟩, (hpos x (hx x hxm)).1⟩
      rename_i hnot
      have hsh : s.shift = 0 := by
        rcases hgates hr with hg | hg
        · refine absurd ?_ hnot
          rw [Array.all_eq_true]
          intro i hi
          obtain ⟨w, hw⟩ := hg _ (Array.getElem_mem_toList hi)
          simp only [hw]
        · exact hg
      rw [if_neg (by simp [hsh])] at hy
      obtain ⟨one, hone, hy⟩ := Walk.except_bind_ok hy
      rw [← Array.forIn_toList] at hone
      have hxg : ∀ g ∈ s.ps.toList, ∀ x ∈ pos g, x ∈ psPos (c.memo[node.fn]!).ret := fun g hg x hxm =>
        hx x (by
          rw [segPos_alike hsh]
          exact List.mem_flatMap.2 ⟨0, List.mem_range.2 (by omega), List.mem_flatMap.2 ⟨g, hg, hxm⟩⟩)
      have hsrc := forIn_src (fun x : Seg => x.reps ≠ 0) _ s.ps.toList (fun g _ r' y' hy' => by
          split at hy'
          · simp only [pure, Except.pure, Except.ok.injEq] at hy'
            subst hy'
            refine ⟨#[_], congrArg _ Array.push_eq_append, ?_⟩
            intro x hx'
            rcases List.mem_singleton.1 hx' with rfl
            exact Nat.one_ne_zero
          · exact absurd hy' (by intro h; cases h)
          · rename_i i hi
            obtain ⟨v', hv', hy'⟩ := Walk.except_bind_ok hy'
            simp only [pure, Except.pure, Except.ok.injEq] at hy'
            subst hy'
            exact ⟨v', rfl, hTR fuel k m i g v' hv'⟩) #[] one hone
      obtain ⟨one', hone', hA'⟩ := forIn_ext (P := c.P) (sc := sc)
        (fun g => (pos g).map (den c.P (c.P.child sc k m))) _ s.ps.toList (fun g hg r' y' hy' => by
          split at hy'
          · rename_i w hw
            simp only [pure, Except.pure, Except.ok.injEq] at hy'
            subst hy'
            refine ⟨#[_], congrArg _ Array.push_eq_append, ?_⟩
            rw [segsPos_single]
            have e := pos_shiftGate ((c.body.offsets[k]! : Int) + (m : Int) * ((c.P.defs[node.fn]!).gates : Int)) g
            rw [hw] at e
            dsimp only at e
            rw [show ({ tgt := g.tgt, base := (c.body.offsets[k]! : Int) + (m : Int) * ((c.P.defs[node.fn]!).gates : Int)
                + g.base, stride := g.stride, len := g.len, rows := g.rows, rowStride := g.rowStride } : Piece) =
                { g with base := g.base + ((c.body.offsets[k]! : Int) + (m : Int) * ((c.P.defs[node.fn]!).gates : Int)) }
                by rw [Int.add_comm], hw, e]
            exact agree_shift c.P ho hoff _ fun x hxm =>
              ⟨⟨w, (mem_pos_tgt hxm).trans hw⟩, (hpos x (hxg g hg x hxm)).1⟩
          · exact absurd hy' (by intro h; cases h)
          · rename_i i hi
            obtain ⟨v', hv', hy'⟩ := Walk.except_bind_ok hy'
            simp only [pure, Except.pure, Except.ok.injEq] at hy'
            subst hy'
            refine ⟨v', rfl, ?_⟩
            refine hT k m i g v' (by rw [hnode]; exact hm) hi (fun x hxm => ?_) hv'
            obtain ⟨h0, -, hp⟩ := hpos x (hxg g hg x hxm)
            obtain ⟨h1, h2⟩ := hp i ((mem_pos_tgt hxm).trans hi)
            rw [hnode]
            exact ⟨h0, h1, h2⟩) #[] one hone
      rw [Array.empty_append] at hone'
      subst hone'
      have hA1 : Agree c.P sc (segsPos one) ((posAll s.ps).map (den c.P (c.P.child sc k m))) := by
        rw [posAll, List.map_flatMap]
        exact hA'
      split at hy
      · rename_i hall
        simp only [pure, Except.pure, Except.ok.injEq] at hy
        subst hy
        refine ⟨#[_], congrArg _ Array.push_eq_append, ?_⟩
        have h1 : ∀ x ∈ one.toList, x.reps = 1 := by
          intro x hx'
          have hle := (Array.all_eq_true_iff_forall_mem.1 hall) x (Array.mem_toList_iff.1 hx')
          simp only [decide_eq_true_eq] at hle
          rcases hsrc x hx' with h0 | h0
          · simp at h0
          · omega
        rw [segsPos_singleton, segPos_alike rfl, segPos_alike hsh, List.map_flatMap, posAll_flatMap_ps one h1]
        exact agree_flatMap _ _ _ fun _ _ => hA1
      · have hy := walkAtMost_bind _ _ _ hy
        obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
        simp only [pure, Except.pure, Except.ok.injEq] at hy
        subst hy
        simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at hv
        obtain ⟨segs', hsegs', hA2⟩ := forIn_ext (P := c.P) (sc := sc)
          (fun _ => (posAll s.ps).map (den c.P (c.P.child sc k m))) _ _ (fun _ _ r' y' hy' => by
            simp only [pure, Except.pure, Except.ok.injEq] at hy'
            exact ⟨one, hy'.symm, hA1⟩) r v hv
        refine ⟨segs', by rw [hsegs'], ?_⟩
        rw [segPos_alike hsh, List.map_flatMap, List.range_eq_range']
        rwa [show (s.reps - 0 + 1 - 1) / 1 = s.reps by omega] at hA2
    · rename_i hr
      have h1 : s.reps = 1 := by omega
      obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      subst hy
      rw [← Array.forIn_toList] at hv
      have hxg : ∀ g ∈ s.ps.toList, ∀ x ∈ pos g, x ∈ psPos (c.memo[node.fn]!).ret := fun g hg x hxm =>
        hx x (by rw [segPos_one_rep h1]; exact List.mem_flatMap.2 ⟨g, hg, hxm⟩)
      obtain ⟨segs', rfl, hA'⟩ := forIn_ext (P := c.P) (sc := sc) (fun g => (pos g).map (den c.P (c.P.child sc k m)))
        _ s.ps.toList (fun g hg r' y' hy' => by
          split at hy'
          · rename_i w hw
            simp only [pure, Except.pure, Except.ok.injEq] at hy'
            subst hy'
            refine ⟨#[_], congrArg _ Array.push_eq_append, ?_⟩
            rw [segsPos_single]
            have e := pos_shiftGate ((c.body.offsets[k]! : Int) + (m : Int) * ((c.P.defs[node.fn]!).gates : Int)) g
            rw [hw] at e
            dsimp only at e
            rw [show ({ tgt := g.tgt, base := (c.body.offsets[k]! : Int) + (m : Int) * ((c.P.defs[node.fn]!).gates : Int)
                + g.base, stride := g.stride, len := g.len, rows := g.rows, rowStride := g.rowStride } : Piece) =
                { g with base := g.base + ((c.body.offsets[k]! : Int) + (m : Int) * ((c.P.defs[node.fn]!).gates : Int)) }
                by rw [Int.add_comm], hw, e]
            exact agree_shift c.P ho hoff _ fun x hxm =>
              ⟨⟨w, (mem_pos_tgt hxm).trans hw⟩, (hpos x (hxg g hg x hxm)).1⟩
          · exact absurd hy' (by intro h; cases h)
          · rename_i i hi
            obtain ⟨v', hv', hy'⟩ := Walk.except_bind_ok hy'
            simp only [pure, Except.pure, Except.ok.injEq] at hy'
            subst hy'
            refine ⟨v', rfl, ?_⟩
            refine hT k m i g v' (by rw [hnode]; exact hm) hi (fun x hxm => ?_) hv'
            obtain ⟨h0, -, hp⟩ := hpos x (hxg g hg x hxm)
            obtain ⟨h1, h2⟩ := hp i ((mem_pos_tgt hxm).trans hi)
            rw [hnode]
            exact ⟨h0, h1, h2⟩) r v hv
      refine ⟨segs', rfl, ?_⟩
      rw [segPos_one_rep h1, posAll, List.map_flatMap]
      exact hA'
  rw [hout, Array.empty_append]
  have h2 := agree_sl (hag _ hd') a b
  rw [sl_map_range' _ _ _ _ hb, ← hslice] at h2
  rw [← List.map_flatMap] at hA
  exact agree_trans hA h2


/-! ## `throughArg`'s step

Up through member `m`'s parameter `i`: its leaves are the node's argument `i`'s references the member reads (§7), each
read in the parent through `down` (`one_agree`). -/

/-- `throughArg`'s read of node `k`'s argument `i` over `[lo, hi)`: `down` of each of its pieces. -/
def oneOf (c : Desc.Ctx) (fuel k i lo hi : Nat) : Except String (Array Seg) := do
  let mut out := #[]
  for q in (c.arg k i).pieces lo hi do out := out ++ (← Desc.down c fuel q)
  return out

/-- Node `k`'s argument `i` as the context holds it names the argument's references in order (`ctxOf_arg`). -/
def ArgOk (c : Desc.Ctx) (k i : Nat) : Prop :=
  PSOk (c.arg k i) ∧ psPos (c.arg k i) = (List.range ((c.body.nodes[k]!).args[i]!).seq.count).map
    fun t => refPos (((c.body.nodes[k]!).args[i]!).seq.get t)

theorem den_refPos (P : Prog) (sc : Flock.Program.Scope) (r : Ref) :
    den P sc (refPos r) = Flock.Program.resolve P sc r := by
  simp only [refPos, den]
  rw [if_neg (by omega), Int.toNat_natCast]

/-- **An argument's references `[lo, hi)`, read through `down`**, keep the gates they resolve to in the parent. -/
theorem one_agree (c : Desc.Ctx) (sc : Flock.Program.Scope) (fuel : Nat) (hD : DownSpec c sc fuel) {k i : Nat}
    (hA : ArgOk c k i)
    (hR : ∀ t < ((c.body.nodes[k]!).args[i]!).seq.count,
      Resolve.RefOk c.P c.d k (((c.body.nodes[k]!).args[i]!).seq.get t))
    {lo hi : Nat} (hhi : lo < hi → hi ≤ ((c.body.nodes[k]!).args[i]!).seq.count) {out : Array Seg}
    (h : oneOf c fuel k i lo hi = .ok out) :
    Agree c.P sc (segsPos out) ((List.range' lo (hi - lo)).map fun t =>
      Flock.Program.resolve c.P sc (((c.body.nodes[k]!).args[i]!).seq.get t)) := by
  unfold oneOf at h
  dsimp only at h
  rw [← Array.forIn_toList] at h
  have hp := pieces_pos _ hA.1 lo hi
  rw [hA.2, sl_map_range' _ _ _ _ hhi] at hp
  have hmem : ∀ q ∈ ((c.arg k i).pieces lo hi).toList, ∀ x ∈ pos q,
      ∃ t < ((c.body.nodes[k]!).args[i]!).seq.count, x = refPos (((c.body.nodes[k]!).args[i]!).seq.get t) := by
    intro q hq x hx
    have : x ∈ posAll ((c.arg k i).pieces lo hi) := List.mem_flatMap.2 ⟨q, hq, hx⟩
    rw [hp] at this
    obtain ⟨t, ht, rfl⟩ := List.mem_map.1 this
    exact ⟨t, by rw [List.mem_range'_1] at ht; omega, rfl⟩
  have := loop_agree (P := c.P) (sc := sc) (Desc.down c fuel) (fun q => (pos q).map (den c.P sc)) _
    (fun q hq segs hs => hD q segs
      (fun x hx => by obtain ⟨t, -, rfl⟩ := hmem q hq x hx; simp [refPos])
      (fun x hx k' hk' => by
        obtain ⟨t, ht, rfl⟩ := hmem q hq x hx
        simp only [refPos, Tgt.ref.injEq] at hk'
        obtain ⟨hp', rfl⟩ := hk'
        have := ((hR t ht).2 hp').2
        simp only [refPos]
        exact_mod_cast this) hs) out h
  rw [← List.map_flatMap] at this
  change Agree c.P sc (segsPos out) ((posAll _).map (den c.P sc)) at this
  rw [hp, List.map_map] at this
  convert this using 2
  funext t
  simp only [Function.comp, den_refPos]


theorem agree_mono {P : Prog} {sc : Flock.Program.Scope} {ys : List (Tgt × Int)} {α : Type} {xs : List α}
    {f g : α → Except String Nat} (h : Agree P sc ys (xs.map f)) (hfg : ∀ x ∈ xs, ∀ a, g x = .ok a → f x = .ok a) :
    Agree P sc ys (xs.map g) := by
  induction xs generalizing ys with
  | nil => cases h; exact List.Forall₂.nil
  | cons x xs ih =>
    rw [List.map_cons] at h ⊢
    cases h with
    | cons hy h' =>
      exact List.Forall₂.cons (fun a ha => hy a (hfg x (by simp) a ha))
        (ih h' (fun x hx => hfg x (by simp [hx])))

/-- A node of a body `Resolve.BodyOk` holds reads its parameters' leaves in range. -/
theorem upOk_of {c : Desc.Ctx} (hB : Resolve.BodyOk c.P c.d) {k : Nat} {node : Flock.Program.Node}
    (hn : c.node k = .ok node) : Resolve.UpOk node (c.P.defs[node.fn]!) := by
  have := (hB.1 k (node_lt hn)).2.2.1
  have e : (c.P.defs[c.d]!).body.nodes[k]! = node := node_get hn
  rwa [e] at this

theorem pos_row (g : Piece) (r : Nat) :
    pos (g.row r) = (List.range g.len).map fun (t : Nat) => (g.tgt, g.base + (r : Int) * g.rowStride + (t : Int) * g.stride) := by
  simp [pos, Desc.Piece.row]

/-- `throughArg`'s elements: `g`'s positions, a row's as one run where its stride is 1, else one by one. -/
theorem elems_pos (g : Piece) (hg : ∀ x ∈ pos g, 0 ≤ x.2) :
    spansPos (g.rowsOf.flatMap fun g =>
      if (g.stride == 1) = true then #[(g.base.toNat, g.base.toNat + g.len)]
      else (Array.range g.len).map fun t => ((g.at t).toNat, (g.at t).toNat + 1)) =
    (pos g).map fun x => x.2.toNat := by
  rw [← posAll_rowsOf] at hg ⊢
  unfold spansPos posAll at *
  rw [Array.toList_flatMap, List.flatMap_assoc, List.map_flatMap]
  apply flatMap_congr
  intro r hr
  have hr' : ∀ x ∈ pos r, 0 ≤ x.2 := fun x hx => hg x (List.mem_flatMap.2 ⟨r, hr, hx⟩)
  obtain ⟨j, -, rfl⟩ : ∃ j < g.rows, g.row j = r := by
    simpa [Desc.Piece.rowsOf] using hr
  rw [pos_row] at hr' ⊢
  simp only [Desc.Piece.row, Desc.Piece.at] at hr' ⊢
  by_cases hs : g.stride = 1
  · simp only [hs, beq_self_eq_true, if_true, mul_one] at hr' ⊢
    simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, Nat.add_sub_cancel_left]
    rw [List.range'_eq_map_range, List.map_map]
    apply List.map_congr_left
    intro t ht
    rw [List.mem_range] at ht
    have h0 := hr' _ (List.mem_map.2 ⟨0, List.mem_range.2 (by omega), rfl⟩)
    simp only [Function.comp, Nat.cast_zero, add_zero] at h0 ⊢
    omega
  · simp only [hs, if_false, beq_iff_eq, Array.toList_map, Array.toList_range] at hr' ⊢
    rw [List.flatMap_map, List.map_map]
    simp only [Nat.add_sub_cancel_left, List.range'_one]
    rw [List.map_eq_flatMap]
    rfl

/-- **`throughArg`'s step**: one step up, member `m`'s parameter leaves read through node `k`'s argument `i` keep the
gates they resolve to in the member, wherever `down` and `fromCallee` do. -/
theorem throughSpec_succ (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (hB : Resolve.BodyOk c.P c.d)
    (hA : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size → ArgOk c k i)
    (hL0 : ∀ k i d L : Nat, ((c.body.nodes[k]!).args[i]!).dims = some (d, L) → L = 0 →
      i < (c.P.defs[(c.body.nodes[k]!).fn]!).body.params.size →
      ((c.P.defs[(c.body.nodes[k]!).fn]!).body.params[i]!).2.leaves = 0)
    (fuel : Nat) (hD : DownSpec c sc fuel) (hK : CarrySpec c sc fuel) : ThroughSpec c sc (fuel + 1) := by
  intro k m i g segs hm hg hx h
  rw [Desc.throughArg.eq_2] at h
  obtain ⟨node, hn, h⟩ := Walk.except_bind_ok h
  rcases ha : node.args[i]? with _ | a
  · rw [ha] at h; exact absurd h (by intro h; cases h)
  rw [ha] at h
  dsimp only at h
  have h' := ok_bind_pure h
  rw [← Array.forIn_toList] at h'
  have hnode : c.body.nodes[k]! = node := node_get hn
  have hk : k < c.body.nodes.size := node_lt hn
  obtain ⟨hia, hai⟩ := Array.getElem?_eq_some_iff.1 ha
  have hai' : node.args[i]! = a := by rw [getElem!_pos node.args i hia]; exact hai
  have hnode' : (c.P.defs[sc.d]!).body.nodes[k]! = node := by rw [hd]; exact hnode
  have hUp := upOk_of hB hn
  have hAk : ArgOk c k i := hA k i hk (by rw [hnode]; exact hia)
  have hRk : ∀ t < ((c.body.nodes[k]!).args[i]!).seq.count,
      Resolve.RefOk c.P c.d k (((c.body.nodes[k]!).args[i]!).seq.get t) := (hB.1 k hk).2.1 i
  have hone : ∀ lo hi v, (lo < hi → hi ≤ a.seq.count) → oneOf c fuel k i lo hi = .ok v →
      Agree c.P sc (segsPos v) ((List.range' lo (hi - lo)).map fun t => Flock.Program.resolve c.P sc (a.seq.get t)) := by
    intro lo hi v hb hv
    have := one_agree c sc fuel hD hAk hRk (by rw [hnode, hai']; exact hb) hv
    rwa [hnode, hai'] at this
  rw [hnode] at hm hx
  have hsp := elems_pos g (fun x hx' => (hx x hx').1)
  have hj : ∀ x ∈ (g.rowsOf.flatMap fun g =>
      if (g.stride == 1) = true then #[(g.base.toNat, g.base.toNat + g.len)]
      else (Array.range g.len).map fun t => ((g.at t).toNat, (g.at t).toNat + 1)).toList, ∀ j, x.1 ≤ j → j < x.2 →
      i < (c.P.defs[node.fn]!).body.params.size ∧ j < ((c.P.defs[node.fn]!).body.params[i]!).2.leaves := by
    intro x hxm j h1 h2
    have : j ∈ spansPos _ := List.mem_flatMap.2 ⟨x, hxm, List.mem_range'_1.2 ⟨h1, by omega⟩⟩
    rw [hsp] at this
    obtain ⟨y, hy, rfl⟩ := List.mem_map.1 this
    have := hx y hy
    exact ⟨this.2.1, by omega⟩
  have hfin := forIn_ext (P := c.P) (sc := sc)
    (fun x : Nat × Nat => (List.range' x.1 (x.2 - x.1)).map fun j =>
      Flock.Program.resolve c.P (c.P.child sc k m) ⟨true, i, j⟩) _ _ ?_ #[] segs h'
  · obtain ⟨segs', rfl, hag⟩ := hfin
    rw [Array.empty_append]
    have e1 : ∀ E : Array (Nat × Nat), E.toList.flatMap (fun x : Nat × Nat => (List.range' x.1 (x.2 - x.1)).map fun j =>
        Flock.Program.resolve c.P (c.P.child sc k m) ⟨true, i, j⟩) =
        (spansPos E).map fun j => Flock.Program.resolve c.P (c.P.child sc k m) ⟨true, i, j⟩ := by
      intro E; unfold spansPos; rw [List.map_flatMap]
    rw [e1, hsp, List.map_map] at hag
    convert hag using 1
    apply List.map_congr_left
    intro x hxp
    obtain ⟨t, j⟩ := x
    have ht : t = .ref true i := by rw [← hg]; exact mem_pos_tgt hxp
    subst ht
    have := (hx _ hxp).1
    simp only [den, Function.comp]
    rw [if_neg (by omega)]
  · intro x hxm r y hy
    obtain ⟨lo, hi⟩ := x
    have hjx := hj (lo, hi) hxm
    have hchild := child_eq c sc hd hn m
    rcases hf : node.form with _ | _ | _ | _
    case prim | call =>
      simp only [hf] at hy
      obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      refine ⟨v, hy.symm, agree_mono (hone lo hi v (fun hlt => ?_) hv) ?_⟩
      · obtain ⟨hi0, hjl⟩ := hjx (hi - 1) (by omega) (by omega)
        have := hUp i hi0 (hi - 1) hjl m hm
        simp only [hf, hai'] at this
        omega
      · intro t _ b hb
        rw [hchild] at hb
        have := den_up c.P sc hnode' (by simp [hf]) hb
        rwa [hai'] at this
    case batch =>
      simp only [hf] at hy
      obtain ⟨w, hw, hy⟩ := Walk.except_bind_ok hy
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      rw [← Array.forIn_toList] at hw
      have hLz : ∀ d L, a.dims = some (d, L) → L = 0 → hi ≤ lo := by
        intro d L hdl hL
        by_contra hlt
        have h1 := (hjx lo le_rfl (by omega)).2
        have h2 := hL0 k i d L (by rw [hnode, hai']; exact hdl) hL
          (by rw [hnode]; exact (hjx lo le_rfl (by omega)).1)
        rw [hnode] at h2
        omega
      have hms := memberSpans_idx a m lo hi hLz
      have hfin2 := forIn_ext (P := c.P) (sc := sc)
        (fun x : Nat × Nat => (List.range' x.1 (x.2 - x.1)).map fun t => Flock.Program.resolve c.P sc (a.seq.get t))
        _ _ ?_ r w hw
      · obtain ⟨segs2, rfl, hag2⟩ := hfin2
        refine ⟨segs2, hy.symm, ?_⟩
        have e2 : (Desc.memberSpans a m lo hi).toList.flatMap (fun x : Nat × Nat => (List.range' x.1 (x.2 - x.1)).map
            fun t => Flock.Program.resolve c.P sc (a.seq.get t)) =
            (spansPos (Desc.memberSpans a m lo hi)).map fun t => Flock.Program.resolve c.P sc (a.seq.get t) := by
          unfold spansPos; rw [List.map_flatMap]
        rw [e2, hms, List.map_map] at hag2
        refine agree_mono hag2 ?_
        intro t _ b hb
        rw [hchild] at hb
        have := den_up_batch c.P sc hnode' hf hb
        rw [hai', Resolve.memberRef_eq] at this
        exact this
      · intro x hxm2 r2 y2 hy2
        obtain ⟨s, e⟩ := x
        obtain ⟨v, hv, hy2⟩ := Walk.except_bind_ok hy2
        simp only [pure, Except.pure, Except.ok.injEq] at hy2
        refine ⟨v, hy2.symm, hone s e v (fun hlt => ?_) hv⟩
        have hmem : e - 1 ∈ spansPos (Desc.memberSpans a m lo hi) :=
          List.mem_flatMap.2 ⟨(s, e), hxm2, List.mem_range'_1.2 ⟨by omega, by omega⟩⟩
        rw [hms] at hmem
        obtain ⟨j, hjm, hje⟩ := List.mem_map.1 hmem
        rw [List.mem_range'_1] at hjm
        obtain ⟨hi0, hjl⟩ := hjx j hjm.1 (by omega)
        have := hUp i hi0 j hjl m hm
        simp only [hf, hai', hje] at this
        omega
    case scan =>
      simp only [hf] at hy
      by_cases hi0 : i = 0
      · subst hi0
        rw [if_pos (show ((0 : Nat) == 0) = true from rfl)] at hy
        by_cases hm0 : m = 0
        · subst hm0
          rw [if_pos (show ((0 : Nat) == 0) = true from rfl)] at hy
          obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
          simp only [pure, Except.pure, Except.ok.injEq] at hy
          refine ⟨v, hy.symm, agree_mono (hone lo hi v (fun hlt => ?_) hv) ?_⟩
          · obtain ⟨hi0, hjl⟩ := hjx (hi - 1) (by omega) (by omega)
            have := hUp 0 hi0 (hi - 1) hjl 0 hm
            simp only [hf, hai', ↓reduceIte] at this
            omega
          · intro t _ b hb
            rw [hchild] at hb
            have := den_up_scan0 c.P sc hnode' hf hb
            rw [if_pos rfl, hai'] at this
            exact this
        · rw [if_neg (show ¬ ((m == 0) = true) by simpa using hm0)] at hy
          obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
          simp only [pure, Except.pure, Except.ok.injEq] at hy
          have hm1 : m - 1 < (c.body.nodes[k]!).members := by rw [hnode]; omega
          have hc := hK k (m - 1) lo hi v (by rw [hnode]; exact hf) hm1 (fun hlt => ?_) hv
          · rw [hnode] at hc
            refine ⟨v, hy.symm, agree_mono hc ?_⟩
            intro t _ b hb
            rw [hchild] at hb
            have := den_up_scan0 c.P sc hnode' hf hb
            rw [if_neg hm0] at this
            rw [child_eq c sc hd hn (m - 1)]
            obtain ⟨m', rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by omega⟩
            have eo : ∀ X G : Nat, X + (m' + 1) * G - G = X + (m' + 1 - 1) * G := by
              intro X G; rw [Nat.add_sub_cancel, Nat.add_mul, one_mul]; omega
            rwa [eo] at this
          · obtain ⟨hi0, hjl⟩ := hjx (hi - 1) (by omega) (by omega)
            have := hUp 0 hi0 (hi - 1) hjl m hm
            simp only [hf, hm0, ↓reduceIte] at this
            rw [hnode]
            omega
      · rw [if_neg (show ¬ ((i == 0) = true) by simpa using hi0)] at hy
        by_cases hx0 : i ≤ node.nxs
        · rw [if_pos hx0] at hy
          obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
          simp only [pure, Except.pure, Except.ok.injEq] at hy
          generalize hL : (Option.map (fun x => x.2) a.dims).getD 0 = L at hv
          have hc := hone (m * L + lo) (m * L + hi) v (fun hlt => ?_) hv
          · refine ⟨v, hy.symm, ?_⟩
            rw [show m * L + hi - (m * L + lo) = hi - lo by omega, ← List.map_add_range', List.map_map] at hc
            refine agree_mono hc ?_
            intro t _ b hb
            rw [hchild] at hb
            have := den_up_scan c.P sc hnode' hf hi0 hb
            rw [if_pos hx0, hai', hL] at this
            exact this
          · obtain ⟨hi0', hjl⟩ := hjx (hi - 1) (by omega) (by omega)
            have := hUp i hi0' (hi - 1) hjl m hm
            simp only [hf, hi0, hx0, hai', hL, ↓reduceIte] at this
            omega
        · rw [if_neg hx0] at hy
          obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
          simp only [pure, Except.pure, Except.ok.injEq] at hy
          refine ⟨v, hy.symm, agree_mono (hone lo hi v (fun hlt => ?_) hv) ?_⟩
          · obtain ⟨hi0', hjl⟩ := hjx (hi - 1) (by omega) (by omega)
            have := hUp i hi0' (hi - 1) hjl m hm
            simp only [hf, hi0, hx0, hai', ↓reduceIte] at this
            omega
          · intro t _ b hb
            rw [hchild] at hb
            have := den_up_scan c.P sc hnode' hf hi0 hb
            rw [if_neg hx0, hai'] at this
            exact this


/-! ## A batch of a composite's piece -/

/-- Member `m`'s block of output leaves `[a, b)`, `L` leaves to a member: its leaves `e` are leaves `m * L + e`. -/
theorem block_shift {α : Type} (V : Nat → Nat → α) {L : Nat} (hL : L ≠ 0) (a b m : Nat) :
    (List.range' (max a (m * L) - m * L) ((min b ((m + 1) * L) - m * L) - (max a (m * L) - m * L))).map (V m) =
      (List.range' (max a (m * L)) (min b ((m + 1) * L) - max a (m * L))).map fun j => V (j / L) (j % L) := by
  have hmL : (m + 1) * L = m * L + L := by rw [Nat.add_mul, Nat.one_mul]
  rw [show (min b ((m + 1) * L) - m * L) - (max a (m * L) - m * L) = min b ((m + 1) * L) - max a (m * L) by omega]
  rw [List.range'_eq_map_range, List.range'_eq_map_range, List.map_map, List.map_map]
  apply List.map_congr_left
  intro t ht
  rw [List.mem_range] at ht
  have he' : max a (m * L) - m * L + t < L := by omega
  simp only [Function.comp]
  rw [show max a (m * L) + t = (max a (m * L) - m * L + t) + m * L by omega,
    Nat.add_mul_div_right _ _ (Nat.pos_of_ne_zero hL), Nat.div_eq_of_lt he', Nat.zero_add,
    Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt he']

/-- `fromCallee` member by member over a batch's output leaves `[a, b)`: leaf `j` is member `j / L`'s return leaf
`j % L`. -/
theorem members_agree (c : Desc.Ctx) (sc : Flock.Program.Scope) (fuel : Nat) (hC : CalleeSpec c sc fuel)
    {k L a b : Nat} (hL : L ≠ 0) (hab : a < b) (hm : (b - 1) / L < (c.body.nodes[k]!).members)
    (hj : ∀ j, a ≤ j → j < b → j % L < (c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.count) {segs : Array Seg}
    (h : (do
      let __s ← forIn (List.range' (a / L) ((b - 1) / L + 1 - a / L)) #[] fun m __s => do
        let __do_lift ← Desc.fromCallee c fuel k m (max a (m * L) - m * L) (min b ((m + 1) * L) - m * L)
        pure (ForInStep.yield (__s ++ __do_lift))
      pure __s) = .ok segs) :
    Agree c.P sc (segsPos segs) ((List.range' a (b - a)).map fun j =>
      Flock.Program.resolve c.P (c.P.child sc k (j / L)) ((c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.get (j % L))) := by
  have hA := loop_agree (P := c.P) (sc := sc)
    (fun m => Desc.fromCallee c fuel k m (max a (m * L) - m * L) (min b ((m + 1) * L) - m * L))
    (fun m => (List.range' (max a (m * L) - m * L) ((min b ((m + 1) * L) - m * L) - (max a (m * L) - m * L))).map
      fun e => Flock.Program.resolve c.P (c.P.child sc k m) ((c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.get e))
    (List.range' (a / L) ((b - 1) / L + 1 - a / L)) (fun m hmm segs hs => by
      rw [List.mem_range'_1] at hmm
      refine hC k m (max a (m * L) - m * L) (min b ((m + 1) * L) - m * L) segs (by omega) (fun hlt => ?_) hs
      have hmL : (m + 1) * L = m * L + L := by rw [Nat.add_mul, Nat.one_mul]
      have := hj (m * L + (min b ((m + 1) * L) - m * L - 1)) (by omega) (by omega)
      rw [show m * L + (min b ((m + 1) * L) - m * L - 1) = (min b ((m + 1) * L) - m * L - 1) + m * L by omega,
        Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt (by omega)] at this
      omega) segs (by rw [h]; rfl)
  rw [flatMap_congr (fun m _ => block_shift (fun m e => Flock.Program.resolve c.P (c.P.child sc k m)
    ((c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.get e)) hL a b m), ← List.map_flatMap,
    List.range'_eq_map_range, List.flatMap_map, blocks_cover L a b (a / L) (Nat.div_mul_le_self a L)] at hA
  have h1 : a / L ≤ (b - 1) / L := Nat.div_le_div_right (by omega)
  have h2 : b ≤ ((b - 1) / L + 1) * L := by
    have e := Nat.div_add_mod' (b - 1) L
    have := Nat.mod_lt (b - 1) (Nat.pos_of_ne_zero hL)
    rw [Nat.add_mul, Nat.one_mul]
    omega
  rwa [show a / L + ((b - 1) / L + 1 - a / L) = (b - 1) / L + 1 by omega, Nat.min_eq_left h2] at hA

theorem members_batch {node : Flock.Program.Node} (h : node.form = .batch) : node.members = node.n := by
  unfold Flock.Program.Node.members; rw [h]; rfl

/-- A batch of a composite's piece of one row and stride 1: leaf `j` of the node is member `j / L`'s return leaf
`j % L`. -/
theorem batch_den (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (q : Piece) {k : Nat}
    (ht : q.tgt = .ref false k) {node : Flock.Program.Node} (hn : c.node k = .ok node) (hform : node.form = .batch)
    (hp : (c.P.defs[node.fn]!).prim = false) (h0 : q.count ≠ 0) (hr : ¬ q.rows > 1) (hs : q.stride = 1)
    (hq : ∀ x ∈ pos q, 0 ≤ x.2) {ys : List (Tgt × Int)}
    (hA : Agree c.P sc ys ((List.range' q.base.toNat q.len).map fun j =>
      Flock.Program.resolve c.P (c.P.child sc k (j / (c.P.defs[node.fn]!).outLeaves))
        ((c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.get (j % (c.P.defs[node.fn]!).outLeaves)))) :
    Agree c.P sc ys ((pos q).map (den c.P sc)) := by
  have h1 := rows_one h0 hr
  have hlen : 0 < q.len := by rw [← count_one h1]; omega
  have hb : 0 ≤ q.base := by
    have := hq _ (at_mem_pos h1 hlen)
    simpa [Desc.Piece.at] using this
  rw [List.range'_eq_map_range, List.map_map] at hA
  rw [pos_eq_map, List.map_map, count_one h1]
  refine agree_mono hA ?_
  intro t ht' x hx
  rw [List.mem_range] at ht'
  have hp' := ppt_one h1 (t := t) (by rw [count_one h1]; exact ht')
  simp only [Function.comp, den, ht, Desc.Piece.at, hs] at hx hp'
  rw [hp', if_neg (by omega)] at hx
  have hnode : (c.P.defs[sc.d]!).body.nodes[k]! = node := by rw [hd]; exact node_get hn
  have e := den_batch c.P sc hnode hform hp hx
  simp only [Function.comp]
  rw [child_eq c sc hd hn, callee_eq c hn, show q.base.toNat + t = (q.base + (t : Int) * 1).toNat by omega]
  exact e

/-- `fromCallee` iteration by iteration over a scan's `y` leaves `[lo, hi)` past its `cl` carry leaves, `yl` leaves to
an iteration: `y` leaf `j` is iteration `j / yl`'s return leaf `cl + j % yl`. -/
theorem ys_agree (c : Desc.Ctx) (sc : Flock.Program.Scope) (fuel : Nat) (hC : CalleeSpec c sc fuel)
    {k cl yl lo hi : Nat} (hyl : yl ≠ 0) (hlh : lo < hi) (hm : (hi - 1) / yl < (c.body.nodes[k]!).members)
    (hj : ∀ j, lo ≤ j → j < hi → cl + j % yl < (c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.count)
    {out0 segs : Array Seg}
    (h : forIn (List.range' (lo / yl) ((hi - 1) / yl + 1 - lo / yl)) out0 (fun m (s : Array Seg) => do
        let v ← Desc.fromCallee c fuel k m (cl + max lo (m * yl) - m * yl) (cl + min hi ((m + 1) * yl) - m * yl)
        pure (ForInStep.yield (s ++ v))) = .ok segs) :
    ∃ segs2, segs = out0 ++ segs2 ∧ Agree c.P sc (segsPos segs2) ((List.range' lo (hi - lo)).map fun j =>
      Flock.Program.resolve c.P (c.P.child sc k (j / yl))
        ((c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.get (cl + j % yl))) := by
  let V : Nat → Nat → Except String Nat := fun m e =>
    Flock.Program.resolve c.P (c.P.child sc k m) ((c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.get e)
  obtain ⟨segs2, hs2, hA⟩ := forIn_ext (P := c.P) (sc := sc)
    (fun m => (List.range' (cl + max lo (m * yl) - m * yl)
      ((cl + min hi ((m + 1) * yl) - m * yl) - (cl + max lo (m * yl) - m * yl))).map (V m))
    _ _ (fun m hmm r y hy => by
      rw [List.mem_range'_1] at hmm
      obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      refine ⟨v, hy.symm, hC k m _ _ v (by omega) (fun hlt => ?_) hv⟩
      have hmL : (m + 1) * yl = m * yl + yl := by rw [Nat.add_mul, Nat.one_mul]
      have := hj (m * yl + (min hi ((m + 1) * yl) - m * yl - 1)) (by omega) (by omega)
      rw [show m * yl + (min hi ((m + 1) * yl) - m * yl - 1) = (min hi ((m + 1) * yl) - m * yl - 1) + m * yl by omega,
        Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt (by omega)] at this
      omega) out0 segs h
  refine ⟨segs2, hs2, ?_⟩
  have e1 : ∀ m ∈ List.range' (lo / yl) ((hi - 1) / yl + 1 - lo / yl),
      (List.range' (cl + max lo (m * yl) - m * yl)
        ((cl + min hi ((m + 1) * yl) - m * yl) - (cl + max lo (m * yl) - m * yl))).map (V m) =
      (List.range' (max lo (m * yl)) (min hi ((m + 1) * yl) - max lo (m * yl))).map
        fun j => V (j / yl) (cl + j % yl) := by
    intro m hmm
    rw [List.mem_range'_1] at hmm
    have hmL : (m + 1) * yl = m * yl + yl := by rw [Nat.add_mul, Nat.one_mul]
    have hle : m * yl ≤ (hi - 1) / yl * yl := Nat.mul_le_mul_right _ (by omega)
    have hdl := Nat.div_mul_le_self (hi - 1) yl
    rw [show cl + max lo (m * yl) - m * yl = cl + (max lo (m * yl) - m * yl) by omega,
      show (cl + min hi ((m + 1) * yl) - m * yl) - (cl + (max lo (m * yl) - m * yl)) =
        (min hi ((m + 1) * yl) - m * yl) - (max lo (m * yl) - m * yl) by omega,
      ← List.map_add_range', List.map_map]
    exact block_shift (fun m e => V m (cl + e)) hyl lo hi m
  rw [flatMap_congr e1, ← List.map_flatMap, List.range'_eq_map_range, List.flatMap_map,
    blocks_cover yl lo hi (lo / yl) (Nat.div_mul_le_self lo yl)] at hA
  have h1 : lo / yl ≤ (hi - 1) / yl := Nat.div_le_div_right (by omega)
  have h2 : hi ≤ ((hi - 1) / yl + 1) * yl := by
    have e := Nat.div_add_mod' (hi - 1) yl
    have := Nat.mod_lt (hi - 1) (Nat.pos_of_ne_zero hyl)
    rw [Nat.add_mul, Nat.one_mul]
    omega
  rwa [show lo / yl + ((hi - 1) / yl + 1 - lo / yl) = (hi - 1) / yl + 1 by omega, Nat.min_eq_left h2] at hA

/-- A scan's `y` leaves `[max a cl, b)` of a piece `[a, b)` of its output, read past its `cl` carry leaves iteration by
iteration onto `out0`: none where `b` is within the carry. -/
theorem scanYs_agree (c : Desc.Ctx) (sc : Flock.Program.Scope) (fuel : Nat) (hC : CalleeSpec c sc fuel)
    {k cl yl a b : Nat} (hab : a < b) (hy0 : yl = 0 → b ≤ cl)
    (hm : cl < b → (b - 1 - cl) / yl < (c.body.nodes[k]!).members)
    (hj : ∀ j, max a cl ≤ j → j < b → cl + (j - cl) % yl < (c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.count)
    {out0 segs : Array Seg}
    (h : (if (decide (cl < b) && decide (yl > 0)) = true then do
        Desc.walkAtMost "iterations of a scan whose ys are read one by one" ((b - cl - 1) / yl + 1 - (max a cl - cl) / yl)
        (forIn [(max a cl - cl) / yl:(b - cl - 1) / yl + 1] out0 fun m s => do
          let v ← Desc.fromCallee c fuel k m (cl + max (max a cl - cl) (m * yl) - m * yl)
            (cl + min (b - cl) ((m + 1) * yl) - m * yl)
          pure (ForInStep.yield (s ++ v))) >>= fun s => pure s
      else pure out0) = .ok segs) :
    ∃ s2, segs = out0 ++ s2 ∧ Agree c.P sc (segsPos s2) ((List.range' (max a cl) (b - max a cl)).map fun j =>
      Flock.Program.resolve c.P (c.P.child sc k ((j - cl) / yl))
        ((c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.get (cl + (j - cl) % yl))) := by
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
    have h := walkAtMost_bind _ _ _ h
    obtain ⟨v, hv, h⟩ := Walk.except_bind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, Nat.add_sub_cancel,
      Nat.div_one] at hv
    obtain ⟨s2, hs2, hA⟩ := ys_agree c sc fuel hC (k := k) (cl := cl) (yl := yl) (by omega)
      (show max a cl - cl < b - cl by omega)
      (by rw [show b - cl - 1 = b - 1 - cl by omega]; exact hm hc.1) (fun j h1 h2 => by
        have := hj (cl + j) (by omega) (by omega)
        rwa [show cl + j - cl = j by omega] at this) hv
    refine ⟨s2, hs2, ?_⟩
    rw [show max a cl = cl + (max a cl - cl) by omega,
      show b - (cl + (max a cl - cl)) = b - cl - (max a cl - cl) by omega, ← List.map_add_range', List.map_map]
    exact agree_mono hA fun x _ a h => by simpa only [Function.comp, Nat.add_sub_cancel_left] using h
  · rename_i hc
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    refine ⟨#[], by simp, ?_⟩
    have : b - max a cl = 0 := by
      simp only [Bool.and_eq_true, decide_eq_true_eq, not_and] at hc
      by_cases hyl : yl = 0
      · have := hy0 hyl; omega
      · have := mt (hc · ) (by omega : ¬ ¬ (yl > 0)); omega
    rw [this]
    exact List.Forall₂.nil

/-- **Open: a scan node's output**, as the decoder computes it (`buildNode`, `(carry, array n y)`): the final carry's
leaves, then `n` iterations' `y` leaves. -/
def ScanOut (c : Desc.Ctx) : Prop :=
  ∀ k node (carry y : Flock.Program.Ty), c.node k = .ok node → node.form = .scan →
    (c.P.defs[node.fn]!).ret = .tuple #[carry, y] → node.out.leaves = carry.leaves + node.n * y.leaves

/-- **A scan's piece of one row**, one step up: its final carry's leaves through `carry`, its ys through `fromCallee`,
leaf `j` of the node being the last iteration's return leaf `j` below the carry's `cl` leaves, else iteration
`(j - cl) / yl`'s return leaf `cl + (j - cl) % yl`. -/
theorem scanSpec_of (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (hB : Resolve.BodyOk c.P c.d)
    (hSO : ScanOut c) (fuel : Nat) (hC : CalleeSpec c sc fuel) (hK : CarrySpec c sc fuel) : ScanSpec c sc fuel := by
  intro q segs k node carry y ht hn hform hn0 h0 hff hr hret hq hf h
  have hdo := downOk_of hB hn
  have h1 := rows_one h0 hr
  have hnode : c.body.nodes[k]! = node := node_get hn
  let M : Nat → Nat × Nat := fun j => if j < carry.leaves then (node.n - 1, j)
    else ((j - carry.leaves) / y.leaves, carry.leaves + (j - carry.leaves) % y.leaves)
  suffices hA : Agree c.P sc (segsPos segs) ((List.range q.len).map fun t =>
      Flock.Program.resolve c.P (c.P.child sc k (M (q.at t).toNat).1)
        ((c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.get (M (q.at t).toNat).2)) by
    apply agree_weaken hA
    rw [pos_eq_map, List.map_map, count_one h1]
    apply forall₂_range_map
    intro t ht' a ha
    have hp := ppt_one h1 (t := t) (by rw [count_one h1]; exact ht')
    have hnn := hq (q.tgt, ppt q t)
      (by rw [pos_eq_map]; exact List.mem_map.2 ⟨t, List.mem_range.2 (by rw [count_one h1]; exact ht'), rfl⟩)
    simp only [Function.comp, den, ht] at ha
    simp only at hnn
    rw [if_neg (by omega), hp] at ha
    have hnode' : (c.P.defs[sc.d]!).body.nodes[k]! = node := by rw [hd]; exact hnode
    have e := den_scan c.P sc hnode' hform hn0 hret ha
    rw [child_eq c sc hd hn, callee_eq c hn]
    exact e
  have hbd : ∀ j, j < node.out.leaves →
      (M j).1 < node.n ∧ (M j).2 < (c.P.defs[node.fn]!).body.ret.count := by
    intro j hj
    have hdk := hdo j hj
    simp only [hform] at hdk
    obtain ⟨carry', y', hr', hmn, hj'⟩ := hdk.2 hn0
    rw [hret] at hr'
    injection hr' with e
    obtain ⟨rfl, rfl⟩ : carry = carry' ∧ y = y' := by simpa using e
    by_cases hlt : j < carry.leaves
    · simp only [M, if_pos hlt]; rw [if_pos hlt] at hmn hj'; exact ⟨hmn, hj'⟩
    · simp only [M, if_neg hlt]; rw [if_neg hlt] at hmn hj'; exact ⟨hmn, hj'⟩
  have hpt : ∀ t < q.len, 0 ≤ q.at t ∧ (q.at t).toNat < node.out.leaves := by
    intro t ht'
    have hm := at_mem_pos h1 ht'
    have hb := hq _ hm
    have hfl := hf _ hm k ht
    simp only [hnode] at hb hfl
    exact ⟨hb, by omega⟩
  rw [down_scan_head c fuel q ht hff hn hform hn0 h0 hr hret] at h
  split at h
  · rename_i hs
    unfold scanOne at h
    dsimp only at h
    have hlen : 0 < q.len := by rw [← count_one h1]; omega
    have hat : ∀ t, (q.at t).toNat = q.base.toNat + t := fun t => by
      have := (hpt 0 hlen).1
      simp only [Desc.Piece.at, hs] at this ⊢
      omega
    have hbl : q.base.toNat + q.len ≤ node.out.leaves := by
      have := (hpt (q.len - 1) (by omega)).2
      rw [hat] at this
      omega
    have hout := hSO k node carry y hn hform hret
    have hy0 : y.leaves = 0 → q.base.toNat + q.len ≤ carry.leaves := fun h0 => by
      rw [h0, Nat.mul_zero, Nat.add_zero] at hout; omega
    have hmB : carry.leaves < q.base.toNat + q.len →
        (q.base.toNat + q.len - 1 - carry.leaves) / y.leaves < (c.body.nodes[k]!).members := by
      intro hcb
      have := (hbd (q.base.toNat + q.len - 1) (by omega)).1
      simp only [M, if_neg (show ¬ (q.base.toNat + q.len - 1 < carry.leaves) by omega)] at this
      rw [hnode, members_scan hform]
      exact this
    have hjB : ∀ j, max q.base.toNat carry.leaves ≤ j → j < q.base.toNat + q.len →
        carry.leaves + (j - carry.leaves) % y.leaves < (c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.count := by
      intro j h1 h2
      have := (hbd j (by omega)).2
      simp only [M, if_neg (show ¬ (j < carry.leaves) by omega)] at this
      rw [hnode]
      exact this
    rw [show (List.range q.len).map (fun t => Flock.Program.resolve c.P (c.P.child sc k (M (q.at t).toNat).1)
        ((c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.get (M (q.at t).toNat).2)) =
        (List.range' q.base.toNat (q.base.toNat + q.len - q.base.toNat)).map (fun j =>
          Flock.Program.resolve c.P (c.P.child sc k (M j).1) ((c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.get (M j).2))
      by
        rw [show q.base.toNat + q.len - q.base.toNat = q.len by omega, List.range'_eq_map_range, List.map_map]
        apply List.map_congr_left
        intro t _
        simp only [Function.comp, hat]]
    split at h
    · rename_i hac
      obtain ⟨v1, hv1, h⟩ := Walk.except_bind_ok h
      obtain ⟨s2, rfl, hA2⟩ := scanYs_agree c sc fuel hC (by omega) hy0 hmB hjB h
      have hA1 := hK k (node.n - 1) _ _ v1 (by rw [hnode]; exact hform)
        (by rw [hnode, members_scan hform]; omega) (fun hlt => by
          have := (hbd (min (q.base.toNat + q.len) carry.leaves - 1) (by omega)).2
          simp only [M, if_pos (show min (q.base.toNat + q.len) carry.leaves - 1 < carry.leaves by omega)] at this
          rw [hnode]
          omega) hv1
      rw [Array.empty_append, segsPos_append,
        range'_split _ (min (q.base.toNat + q.len) carry.leaves) _ (by omega) (by omega), List.map_append]
      refine agree_append (agree_mono hA1 fun j hj a' ha' => ?_) ?_
      · rw [List.mem_range'_1] at hj
        simpa only [M, if_pos (show j < carry.leaves by omega)] using ha'
      · rw [show List.range' (min (q.base.toNat + q.len) carry.leaves)
            (q.base.toNat + q.len - min (q.base.toNat + q.len) carry.leaves) =
            List.range' (max q.base.toNat carry.leaves) (q.base.toNat + q.len - max q.base.toNat carry.leaves) by
          by_cases hb : q.base.toNat + q.len ≤ carry.leaves
          · rw [show q.base.toNat + q.len - min (q.base.toNat + q.len) carry.leaves = 0 by omega,
              show q.base.toNat + q.len - max q.base.toNat carry.leaves = 0 by omega]
            rfl
          · rw [show min (q.base.toNat + q.len) carry.leaves = max q.base.toNat carry.leaves by omega]]
        refine agree_mono hA2 fun j hj a' ha' => ?_
        rw [List.mem_range'_1] at hj
        simpa only [M, if_neg (show ¬ j < carry.leaves by omega)] using ha'
    · rename_i hac
      obtain ⟨s2, rfl, hA2⟩ := scanYs_agree c sc fuel hC (by omega) hy0 hmB hjB h
      rw [Array.empty_append]
      rw [show max q.base.toNat carry.leaves = q.base.toNat by omega] at hA2
      refine agree_mono hA2 fun j hj a' ha' => ?_
      rw [List.mem_range'_1] at hj
      simpa only [M, if_neg (show ¬ j < carry.leaves by omega)] using ha'
  · rename_i hs
    unfold scanPts at h
    dsimp only at h
    have h := walkAtMost_bind _ _ _ h
    obtain ⟨v, hv, h⟩ := Walk.except_bind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at hv
    rw [range'_size] at hv
    obtain ⟨segs', hsegs, hA⟩ := forIn_ext (P := c.P) (sc := sc) (fun t =>
      [Flock.Program.resolve c.P (c.P.child sc k (M (q.at t).toNat).1)
        ((c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.get (M (q.at t).toNat).2)]) _ (List.range q.len)
      (fun t htm r y' hy => by
        obtain ⟨hb0, hlt⟩ := hpt t (List.mem_range.1 htm)
        obtain ⟨hmn, hjr⟩ := hbd _ hlt
        split at hy
        · rename_i hc
          obtain ⟨v', hv', hy⟩ := Walk.except_bind_ok hy
          simp only [pure, Except.pure, Except.ok.injEq] at hy
          refine ⟨v', hy.symm, ?_⟩
          simp only [M, if_pos hc] at hmn hjr ⊢
          have := hK k (node.n - 1) (q.at t).toNat ((q.at t).toNat + 1) v' (by rw [hnode]; exact hform)
            (by rw [hnode, members_scan hform]; exact hmn) (fun _ => by rw [hnode]; omega) hv'
          simpa using this
        · rename_i hc
          obtain ⟨v', hv', hy⟩ := Walk.except_bind_ok hy
          simp only [pure, Except.pure, Except.ok.injEq] at hy
          refine ⟨v', hy.symm, ?_⟩
          simp only [M, if_neg hc] at hmn hjr ⊢
          have := hC k _ _ _ v' (by rw [hnode, members_scan hform]; exact hmn) (fun _ => by rw [hnode]; omega) hv'
          simpa using this) #[] v hv
    rw [hsegs, Array.empty_append]
    rwa [← List.map_eq_flatMap] at hA

/-- **A batch of a composite's piece in its one-run case** (`affineOf`, every member returning one run of its gates and
the members' runs making one), which `batchAffine` proves wherever the callees' summaries hold `RetOk` and a
composite's return has as many leaves as its type. -/
def BatchAffine (c : Desc.Ctx) (sc : Flock.Program.Scope) : Prop :=
  ∀ q k node p, q.tgt = .ref false k → c.node k = .ok node → node.form = .batch →
    (c.P.defs[node.fn]!).prim = false → (c.P.defs[node.fn]!).outLeaves ≠ 0 → q.count ≠ 0 →
    (∀ x ∈ pos q, 0 ≤ x.2) → (∀ x ∈ pos q, Fits c x) →
    affineOf c k node q = some p → Agree c.P sc (segsPos #[Seg.one p]) ((pos q).map (den c.P sc))

/-! ## Segments of a copy at least

Every segment `down`, `fromCallee`, `throughArg` and `carry` return has a copy at least, and more than one only of
gates or of copies all alike (`RepsOk`; `reps_fuel`, `throughReps`): `Seg.one`, a member's return segments of more than
one copy (gates moved by `shiftGates`, `gates_shiftGates`), a summary's slice copies of one resolved (`shift = 0`), or
what the calls one step down return, followed through `Desc`'s code (`OkIs`, `reps_step`), its scan case and `down`'s
body copied out (`scanBranch`, `downBody`, `down_succ`). So every summary `memoOf` makes has return segments of that
shape (`summarize_reps`, `memoOf_reps`), `RetOk`'s third part (`retOk_reps`). -/

/-- `scanBranch` where the step returns `(ct, y)`. -/
def scanTuple (c : Desc.Ctx) (fuel k : Nat) (node : Flock.Program.Node) (ct y : Flock.Program.Ty) (q : Piece) :
    Except String (Array Seg) := do
        let cl := ct.leaves
        let yl := y.leaves
        let mut out := #[]
        if q.stride == 1 then
          let (a, b) := (q.base.toNat, q.base.toNat + q.len)
          if a < cl then out := out ++ (← Desc.carry c fuel k (node.n - 1) a (min b cl))
          if cl < b && yl > 0 then
            let (lo, hi) := (max a cl - cl, b - cl)
            Desc.walkAtMost "iterations of a scan whose ys are read one by one" ((hi - 1) / yl + 1 - lo / yl)
            for m in [lo / yl:(hi - 1) / yl + 1] do
              out := out ++ (← Desc.fromCallee c fuel k m (cl + max lo (m * yl) - m * yl) (cl + min hi ((m + 1) * yl) - m * yl))
          return out
        Desc.walkAtMost "leaves of a scan's output read one by one" q.len
        for t in [0:q.len] do
          let j := (q.at t).toNat
          if j < cl then out := out ++ (← Desc.carry c fuel k (node.n - 1) j (j + 1))
          else
            let (m, jj) := ((j - cl) / yl, cl + (j - cl) % yl)
            out := out ++ (← Desc.fromCallee c fuel k m jj (jj + 1))
        return out

/-- `down`'s scan case past its row-by-row one: the final carry's leaves and the ys. -/
def scanBranch (c : Desc.Ctx) (fuel k : Nat) (node : Flock.Program.Node) (q : Piece) : Except String (Array Seg) := do
        let .tuple #[ct, y] := (c.P.defs[node.fn]!).ret | throw "a scan step returns (carry, y)"
        let cl := ct.leaves
        let yl := y.leaves
        let mut out := #[]
        if q.stride == 1 then
          let (a, b) := (q.base.toNat, q.base.toNat + q.len)
          if a < cl then out := out ++ (← Desc.carry c fuel k (node.n - 1) a (min b cl))
          if cl < b && yl > 0 then
            let (lo, hi) := (max a cl - cl, b - cl)
            Desc.walkAtMost "iterations of a scan whose ys are read one by one" ((hi - 1) / yl + 1 - lo / yl)
            for m in [lo / yl:(hi - 1) / yl + 1] do
              out := out ++ (← Desc.fromCallee c fuel k m (cl + max lo (m * yl) - m * yl) (cl + min hi ((m + 1) * yl) - m * yl))
          return out
        Desc.walkAtMost "leaves of a scan's output read one by one" q.len
        for t in [0:q.len] do
          let j := (q.at t).toNat
          if j < cl then out := out ++ (← Desc.carry c fuel k (node.n - 1) j (j + 1))
          else
            let (m, jj) := ((j - cl) / yl, cl + (j - cl) % yl)
            out := out ++ (← Desc.fromCallee c fuel k m jj (jj + 1))
        return out

/-- `down` one step up, its recursive calls at `fuel`: `Desc.down`'s body. -/
def downBody (c : Desc.Ctx) (fuel : Nat) (q : Piece) : Except String (Array Seg) := do
    match q.tgt with
    | .gate _ | .ref true _ => return #[.one q]
    | .ref false k =>
      if q.count == 0 then return #[]
      let q := q.flat
      let node ← c.node k
      let o : Int := c.body.offsets[k]!
      let callee := c.P.defs[node.fn]!
      let byRow (_ : Unit) : Except String (Array Seg) := do
        Desc.walkAtMost "rows of a read resolved row by row" q.rows
        let mut out := #[]
        for x in q.rowsOf do out := out ++ (← Desc.down c fuel x)
        return out
      match node.form with
      | .prim =>
        let .value w := callee.ret | throw "program"
        return #[.one { tgt := .gate w, base := o, stride := 0, len := q.count }]
      | .batch =>
        if callee.prim then
          let .value w := callee.ret | throw "program"
          return #[.one { q with tgt := .gate w, base := o + q.base }]
        let L := callee.outLeaves
        let G : Int := callee.gates
        let r := (c.memo[node.fn]!).ret
        if L == 0 then throw "program"
        if r.segs.size == 1 && r.segs[0]!.reps ≤ 1 && r.segs[0]!.ps.size == 1 then
          let g := r.segs[0]!.ps[0]!
          if let .gate w := g.tgt then
            if g.rows ≤ 1 && g.len == L && (L == 1 || G == L * g.stride) then
              return #[.one (q.affine w (o + g.base) (if L == 1 then G else g.stride))]
        if q.rows > 1 then byRow ()
        else if q.stride == 1 then
          let a := q.base.toNat
          let b := a + q.len
          let (m0, m1) := ((a + L - 1) / L, b / L)
          let ends (mid : Array Seg) : Except String (Array Seg) := do
            let mut out := #[]
            if a < m0 * L then out := out ++ (← Desc.fromCallee c fuel k (a / L) (a % L) L)
            out := out ++ mid
            if m1 * L < b then out := out ++ (← Desc.fromCallee c fuel k m1 0 (b - m1 * L))
            return out
          let gatesOnly := r.segs.all fun s => s.reps ≤ 1 && s.ps.all fun g => match g.tgt with
            | .gate _ => true
            | _ => false
          if gatesOnly && m0 + 1 < m1 then
            return ← ends #[{ ps := Desc.shiftGates (o + m0 * G) (r.segs.flatMap (·.ps)), len := L, reps := m1 - m0, shift := G }]
          let shared := r.segs.all fun s => s.reps ≤ 1 && s.ps.all fun g => match g.tgt with
            | .ref true i => (node.args[i]?.map (·.dims.isNone)).getD false
            | _ => false
          if shared && m0 + 1 < m1 then
            let one ← Desc.fromCallee c fuel k m0 0 L
            if one.all (·.reps ≤ 1) then
              return ← ends #[{ ps := one.flatMap (·.ps), len := L, reps := m1 - m0, shift := 0 }]
          if m0 + 1 < m1 && r.segs.size == 1 && r.segs[0]!.reps ≤ 1 && r.segs[0]!.ps.size == 1 then
            let g := r.segs[0]!.ps[0]!
            if let .ref true i := g.tgt then
              if let some (d, Li) := (node.args[i]?.bind (·.dims)) then
                let lo := g.base.toNat
                if Li > 0 && g.rows ≤ 1 && (g.stride == 1 || g.len ≤ 1) && g.len == L && lo / Li == (lo + L - 1) / Li then
                  let qb := lo / Li
                  let p : Piece := { tgt := .gate 0, base := (((qb * d + m0) * Li + (lo - qb * Li) : Nat) : Int),
                                     stride := 1, len := L, rows := m1 - m0, rowStride := (Li : Int) }
                  if let some bp := Desc.throughPS (c.arg k i) p then
                    return ← ends (← Desc.down c fuel bp)
          Desc.walkAtMost "members of a batch read one by one" ((b - 1) / L + 1 - a / L)
          let mut out := #[]
          for m in [a / L:(b - 1) / L + 1] do
            out := out ++ (← Desc.fromCallee c fuel k m (max a (m * L) - m * L) (min b ((m + 1) * L) - m * L))
          return out
        else
          if q.len > 1 && q.stride % (L : Int) == 0 && q.base ≥ 0 then
            let j := q.base.toNat % L
            let m0 := q.base.toNat / L
            let leaf := r.slice j (j + 1)
            if leaf.all (·.reps ≤ 1) then
              let ps := leaf.flatMap (·.ps)
              if ps.all fun g => match g.tgt with | .gate _ => true | _ => false then
                return #[{ ps := Desc.shiftGates (o + m0 * G) ps, len := 1, reps := q.len, shift := q.stride / L * G }]
              if ps.all fun g => match g.tgt with
                  | .ref true i => (node.args[i]?.map (·.dims.isNone)).getD false
                  | _ => false then
                let one ← Desc.fromCallee c fuel k m0 j (j + 1)
                if one.all (·.reps ≤ 1) then return #[{ ps := one.flatMap (·.ps), len := 1, reps := q.len, shift := 0 }]
          Desc.walkAtMost "leaves of a batch's output read one by one" q.len
          let mut out := #[]
          for t in [0:q.len] do
            out := out ++ (← Desc.down c fuel { q with base := q.at t, stride := 1, len := 1 })
          return out
      | .call =>
        if q.rows > 1 then byRow ()
        else if q.stride == 1 then Desc.fromCallee c fuel k 0 q.base.toNat (q.base.toNat + q.len)
        else
          let mut out := #[]
          for t in [0:q.len] do
            out := out ++ (← Desc.fromCallee c fuel k 0 (q.at t).toNat ((q.at t).toNat + 1))
          return out
      | .scan =>
        if node.n == 0 then throw "a scan with no iterations has no output to read"
        if q.rows > 1 then return ← byRow ()
        scanBranch c fuel k node q

theorem down_succ (c : Desc.Ctx) (fuel : Nat) (q : Piece) : Desc.down c (fuel + 1) q = downBody c fuel q := by
  delta Desc.down downBody scanBranch Desc.fromCallee Desc.carry
  rfl


/-- Wherever `x` succeeds, `I` holds of its value. -/
def OkIs {α : Type} (I : α → Prop) (x : Except String α) : Prop := ∀ v, x = .ok v → I v

theorem okIs_pure {α : Type} {I : α → Prop} {v : α} (h : I v) : OkIs I (pure v) := by
  intro w hw
  simp only [pure, Except.pure, Except.ok.injEq] at hw
  subst hw
  exact h

theorem okIs_throw {α : Type} {I : α → Prop} {e : String} : OkIs I (throw e) := by
  intro w hw
  cases hw

theorem okIs_bind {α β : Type} {I : β → Prop} {x : Except String α} {f : α → Except String β}
    (h : ∀ a, x = .ok a → OkIs I (f a)) : OkIs I (x >>= f) := by
  intro w hw
  obtain ⟨a, ha, hw⟩ := Walk.except_bind_ok hw
  exact h a ha w hw

theorem okIs_forIn {β σ : Type} {I : σ → Prop} (f : β → σ → Except String (ForInStep σ)) :
    ∀ (l : List β) (acc : σ), (∀ x ∈ l, ∀ r, I r → OkIs (fun y => I y.value) (f x r)) → I acc →
      OkIs I (forIn l acc f)
  | [], acc, _, hI => by
    intro w hw
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at hw
    subst hw
    exact hI
  | x :: l, acc, hF, hI => by
    intro w hw
    rw [List.forIn_cons] at hw
    rcases hx : f x acc with e | y <;> rw [hx] at hw
    · cases hw
    have hy := hF x List.mem_cons_self acc hI y hx
    cases y with
    | done b =>
      have hw' : (Except.ok b : Except String σ) = .ok w := hw
      injection hw' with hw'
      subst hw'
      exact hy
    | yield b => exact okIs_forIn f l b (fun x hx => hF x (List.mem_cons_of_mem _ hx)) hy w hw

theorem okIs_ite {α : Type} {I : α → Prop} {p : Prop} [Decidable p] {a b : Except String α}
    (ha : p → OkIs I a) (hb : ¬ p → OkIs I b) : OkIs I (if p then a else b) := by
  by_cases h : p
  · rw [if_pos h]; exact ha h
  · rw [if_neg h]; exact hb h

theorem okIs_bind_of {α β : Type} {J : α → Prop} {I : β → Prop} {x : Except String α} {f : α → Except String β}
    (hx : OkIs J x) (h : ∀ a, J a → OkIs I (f a)) : OkIs I (x >>= f) :=
  okIs_bind fun a ha => h a (hx a ha)

theorem okIs_throw_bind {α β : Type} {I : β → Prop} {e : String} {f : α → Except String β} :
    OkIs I (throw e >>= f) := by
  intro w hw
  cases hw

/-- A segment of a copy at least, and of more than one only of gates or of copies all alike (`shift = 0`). -/
def RepsOk (x : Seg) : Prop := x.reps ≠ 0 ∧ (x.reps > 1 → (∀ g ∈ x.ps.toList, ∃ w, g.tgt = .gate w) ∨ x.shift = 0)

/-- Every segment of `segs` is `RepsOk`. -/
def AllReps (segs : Array Seg) : Prop := ∀ x ∈ segs.toList, RepsOk x

theorem okIs_forIn_bind {β γ : Type} {J : γ → Prop} (f : β → Array Seg → Except String (ForInStep (Array Seg)))
    (l : List β) (acc : Array Seg) {g : Array Seg → Except String γ}
    (hF : ∀ x ∈ l, ∀ r, AllReps r → OkIs (fun y => AllReps y.value) (f x r)) (hI : AllReps acc)
    (hg : ∀ s, AllReps s → OkIs J (g s)) : OkIs J (forIn l acc f >>= g) :=
  okIs_bind_of (okIs_forIn f l acc hF hI) hg


theorem okIs_forIn_bind_arr {β γ : Type} {J : γ → Prop} (f : β → Array Seg → Except String (ForInStep (Array Seg)))
    (xs : Array β) (acc : Array Seg) {g : Array Seg → Except String γ}
    (hF : ∀ x ∈ xs.toList, ∀ r, AllReps r → OkIs (fun y => AllReps y.value) (f x r)) (hI : AllReps acc)
    (hg : ∀ s, AllReps s → OkIs J (g s)) : OkIs J (forIn xs acc f >>= g) := by
  rw [← Array.forIn_toList]
  exact okIs_forIn_bind f xs.toList acc hF hI hg

theorem okIs_forIn_arr {β σ : Type} {I : σ → Prop} (f : β → σ → Except String (ForInStep σ)) (xs : Array β) (acc : σ)
    (hF : ∀ x ∈ xs.toList, ∀ r, I r → OkIs (fun y => I y.value) (f x r)) (hI : I acc) : OkIs I (forIn xs acc f) := by
  rw [← Array.forIn_toList]
  exact okIs_forIn f xs.toList acc hF hI

theorem allReps_empty : AllReps #[] := by
  intro x hx
  simp at hx

theorem allReps_range {n : Std.Legacy.Range} {f : Nat → Array Seg → Except String (ForInStep (Array Seg))}
    {a : Array Seg} (hF : ∀ t r, AllReps r → OkIs (fun y => AllReps y.value) (f t r)) (h : forIn n #[] f = .ok a) :
    AllReps a := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  exact okIs_forIn f _ _ (fun x _ r hr => hF x r hr) allReps_empty a h

theorem okIs_forIn_range {σ : Type} {I : σ → Prop} (f : Nat → σ → Except String (ForInStep σ))
    (n : Std.Legacy.Range) (acc : σ) (hF : ∀ t r, I r → OkIs (fun y => I y.value) (f t r)) (hI : I acc) :
    OkIs I (forIn n acc f) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  exact okIs_forIn f _ acc (fun x _ r hr => hF x r hr) hI

theorem allReps_append {a b : Array Seg} (ha : AllReps a) (hb : AllReps b) : AllReps (a ++ b) := by
  intro x hx
  rw [Array.toList_append, List.mem_append] at hx
  exact hx.elim (ha x) (hb x)

theorem allReps_single {s : Seg} (hs : RepsOk s) : AllReps #[s] := by
  intro x hx
  simp at hx
  subst hx
  exact hs

theorem allReps_push {a : Array Seg} {s : Seg} (ha : AllReps a) (hs : RepsOk s) : AllReps (a.push s) := by
  rw [Array.push_eq_append]
  exact allReps_append ha (allReps_single hs)

theorem repsOk_one {s : Seg} (h : s.reps = 1) : RepsOk s := ⟨by omega, fun h' => by omega⟩

theorem repsOk_alike {s : Seg} (hr : s.reps ≠ 0) (h0 : s.shift = 0) : RepsOk s := ⟨hr, fun _ => .inr h0⟩

theorem repsOk_gates {s : Seg} (hr : s.reps ≠ 0) (hg : ∀ g ∈ s.ps.toList, ∃ w, g.tgt = .gate w) : RepsOk s :=
  ⟨hr, fun _ => .inl hg⟩

/-- `shiftGates` of pieces `p` finds gates names gates. -/
theorem gates_shiftGates {p : Piece → Bool} {ps : Array Piece} (h : ps.all p = true)
    (hp : ∀ g, p g = true → ∃ w, g.tgt = .gate w) (δ : Int) : ∀ g ∈ (Desc.shiftGates δ ps).toList, ∃ w, g.tgt = .gate w := by
  intro g hg
  unfold Desc.shiftGates at hg
  rw [Array.toList_map, List.mem_map] at hg
  obtain ⟨g0, hg0, rfl⟩ := hg
  obtain ⟨w, hw⟩ := hp g0 (Array.all_eq_true_iff_forall_mem.1 h g0 (Array.mem_toList_iff.1 hg0))
  exact ⟨w, by simp only [hw]⟩

/-- `down`'s test that a batch's members return gates only, one copy to a segment, names its pieces' gates. -/
theorem gates_flatMap {segs : Array Seg} {p : Piece → Bool} {b : Bool}
    (h : ((segs.all fun s => decide (s.reps ≤ 1) && s.ps.all p) && b) = true) : (segs.flatMap (·.ps)).all p = true := by
  rw [Bool.and_eq_true, Array.all_eq_true_iff_forall_mem] at h
  rw [Array.all_eq_true_iff_forall_mem]
  intro g hg
  obtain ⟨s, hs, hgs⟩ := Array.mem_flatMap.1 hg
  have := h.1 s hs
  rw [Bool.and_eq_true, Array.all_eq_true_iff_forall_mem] at this
  exact this.2 g hgs

set_option hygiene false in
/-- One step of `OkIs AllReps` through `Desc`'s code, its calls' segments by the hypotheses in scope named `hD`
(`down`), `hC` (`fromCallee`), `hK` (`carry`), `hT` (`throughArg`) and `hS` (`scanBranch`). -/
macro "reps_step" : tactic => `(tactic| first
  | with_reducible exact okIs_throw
  | with_reducible exact okIs_throw_bind
  | (with_reducible apply okIs_bind_of (hD _)) <;> intro _ _
  | (with_reducible apply okIs_bind_of (hC _ _ _ _)) <;> intro _ _
  | (with_reducible apply okIs_bind_of (hK _ _ _ _)) <;> intro _ _
  | (with_reducible apply okIs_bind_of (hT _ _ _ _)) <;> intro _ _
  | (with_reducible apply okIs_bind_of (hS _ _ _)) <;> intro _ _
  | with_reducible exact hD _
  | with_reducible exact hC _ _ _ _
  | with_reducible exact hK _ _ _ _
  | with_reducible exact hT _ _ _ _
  | with_reducible exact hS _ _ _
  | rw [Std.Legacy.Range.forIn_eq_forIn_range']
  | rw [← Array.forIn_toList]
  | with_reducible refine okIs_forIn_bind _ _ _ (fun _ _ _ _ => ?_) ?_ (fun _ _ => ?_)
  | with_reducible refine okIs_forIn_bind_arr _ _ _ (fun _ _ _ _ => ?_) ?_ (fun _ _ => ?_)
  | with_reducible refine okIs_forIn_arr _ _ _ (fun _ _ _ _ => ?_) ?_
  | with_reducible refine okIs_forIn_range _ _ _ (fun _ _ _ => ?_) ?_
  | with_reducible refine okIs_bind_of (J := AllReps) ?_ (fun _ _ => ?_)
  | (with_reducible apply okIs_bind) <;> intro _ _
  | (with_reducible apply okIs_ite) <;> intro _
  | (with_reducible apply okIs_forIn) <;> try intro _ _ _ _
  | with_reducible apply okIs_pure
  | with_reducible exact allReps_empty
  | with_reducible apply allReps_single
  | exact repsOk_one rfl
  | with_reducible apply allReps_append
  | with_reducible apply allReps_push
  | assumption
  | split
  | dsimp only [ForInStep.value]
  | (with_reducible refine allReps_range (fun _ _ _ => ?_) ‹_›)
  | (refine repsOk_alike ?_ rfl; (try dsimp only);
      (try simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, gt_iff_lt] at *); omega)
  | (refine repsOk_gates ?_ (gates_shiftGates (by first | exact ‹_› | exact gates_flatMap ‹_›)
        (fun _ hg => by split at hg <;> first | exact ⟨_, ‹_›⟩ | cases hg) _);
      (try dsimp only); (try simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, gt_iff_lt] at *); omega))

-- one `repeat' reps_step` walks every case of `down`'s body, about 4x the default budget
set_option maxHeartbeats 1000000 in
theorem down_reps (c : Desc.Ctx) (fuel : Nat) (hD : ∀ q, OkIs AllReps (Desc.down c fuel q))
    (hC : ∀ k m a b, OkIs AllReps (Desc.fromCallee c fuel k m a b))
    (hS : ∀ k node q, OkIs AllReps (scanBranch c fuel k node q)) (q : Piece) :
    OkIs AllReps (Desc.down c (fuel + 1) q) := by
  rw [down_succ]
  unfold downBody
  repeat' reps_step

theorem scanBranch_reps (c : Desc.Ctx) (fuel : Nat)
    (hC : ∀ k m a b, OkIs AllReps (Desc.fromCallee c fuel k m a b))
    (hK : ∀ k m a b, OkIs AllReps (Desc.carry c fuel k m a b)) (k : Nat) (node : Flock.Program.Node) (q : Piece) :
    OkIs AllReps (scanBranch c fuel k node q) := by
  unfold scanBranch
  rcases hr : (c.P.defs[node.fn]!).ret with w | ⟨n, e⟩ | ⟨⟨_ | ⟨ct, _ | ⟨y, _ | ⟨z, l⟩⟩⟩⟩⟩ | fs
  all_goals (try exact okIs_throw)
  show OkIs AllReps (scanTuple c fuel k node ct y q)
  unfold scanTuple
  repeat' reps_step


theorem fromCallee_reps (c : Desc.Ctx) (fuel : Nat) (hT : ∀ k m i g, OkIs AllReps (Desc.throughArg c fuel k m i g))
    (k m a b : Nat) : OkIs AllReps (Desc.fromCallee c (fuel + 1) k m a b) := by
  rw [Desc.fromCallee.eq_2]
  repeat' reps_step

theorem throughArg_reps (c : Desc.Ctx) (fuel : Nat) (hD : ∀ q, OkIs AllReps (Desc.down c fuel q))
    (hK : ∀ k m a b, OkIs AllReps (Desc.carry c fuel k m a b)) (k m i : Nat) (g : Piece) :
    OkIs AllReps (Desc.throughArg c (fuel + 1) k m i g) := by
  rw [Desc.throughArg.eq_2]
  repeat' reps_step

theorem carry_reps (c : Desc.Ctx) (fuel : Nat) (hC : ∀ k m a b, OkIs AllReps (Desc.fromCallee c fuel k m a b))
    (hT : ∀ k m i g, OkIs AllReps (Desc.throughArg c fuel k m i g)) (k m a b : Nat) :
    OkIs AllReps (Desc.carry c (fuel + 1) k m a b) := by
  rw [Desc.carry.eq_2]
  repeat' reps_step

/-- **Every segment `down`, `fromCallee`, `throughArg` and `carry` return is of a copy at least**, at every fuel. -/
theorem reps_fuel (c : Desc.Ctx) : ∀ fuel, (∀ q, OkIs AllReps (Desc.down c fuel q)) ∧
    (∀ k m a b, OkIs AllReps (Desc.fromCallee c fuel k m a b)) ∧
    (∀ k m i g, OkIs AllReps (Desc.throughArg c fuel k m i g)) ∧
    (∀ k m a b, OkIs AllReps (Desc.carry c fuel k m a b))
  | 0 => by
    refine ⟨fun q v h => absurd h (down_zero c q v), ?_, ?_, ?_⟩
    · intro _ _ _ _ _ h; rw [Desc.fromCallee.eq_1] at h; cases h
    · intro _ _ _ _ _ h; rw [Desc.throughArg.eq_1] at h; cases h
    · intro _ _ _ _ _ h; rw [Desc.carry.eq_1] at h; cases h
  | fuel + 1 =>
    have ⟨hD, hC, hT, hK⟩ := reps_fuel c fuel
    ⟨down_reps c fuel hD hC (scanBranch_reps c fuel hC hK), fromCallee_reps c fuel hT, throughArg_reps c fuel hD hK,
      carry_reps c fuel hC hT⟩

/-- **`throughArg`'s segments are each of a copy at least.** -/
theorem throughReps (c : Desc.Ctx) : ThroughReps c :=
  fun fuel k m i g segs h x hx => ((reps_fuel c fuel).2.2.1 k m i g segs h x hx).1

/-- `PS.of` keeps its segments of positive count. -/
theorem allReps_of {segs : Array Seg} (h : AllReps segs) : AllReps (PS.of segs).segs := by
  rw [of_segs]
  intro x hx
  simp only [Array.toList_filter, List.mem_filter] at hx
  exact h x hx.1

/-- **A summary's return segments are each `RepsOk`**: `Desc.summarize` lays `down`'s segments of its return end to
end. -/
theorem summarize_reps (P : Prog) (memo : Array Desc.Summary) (d : Nat) (s : Desc.Summary)
    (h : Desc.summarize P memo d = .ok s) : AllReps s.ret.segs := by
  unfold Desc.summarize at h
  extract_lets defn body c ret0 acc0 at h
  split at h
  · simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact allReps_empty
  · obtain ⟨ret, hret, h⟩ := Walk.except_bind_ok h
    obtain ⟨acc, -, h⟩ := Walk.except_bind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    have hD : ∀ q, OkIs AllReps (Desc.down c Flock.Steps.FUEL q) := (reps_fuel c _).1
    have hret' : AllReps ret := by
      refine (?_ : OkIs AllReps _) ret hret
      repeat' reps_step
    exact allReps_of hret'

theorem memo_push_reps {memo : Array Desc.Summary} {x : Desc.Summary} (hm : ∀ f : Nat, AllReps (memo[f]!).ret.segs)
    (hx : AllReps x.ret.segs) (f : Nat) : AllReps ((memo.push x)[f]!).ret.segs := by
  by_cases hf : f < memo.size
  · rw [getElem!_pos (memo.push x) f (by simp; omega), Array.getElem_push_lt hf, ← getElem!_pos memo f hf]
    exact hm f
  by_cases he : f = memo.size
  · subst he
    rw [getElem!_pos (memo.push x) memo.size (by simp), Array.getElem_push_eq]
    exact hx
  · rw [getElem!_neg (memo.push x) f (by simp; omega)]
    exact allReps_empty

/-- **Every summary `memoOf` makes has return segments each `RepsOk`**, so the third part of `RetOk` holds of every
Definition: a segment of more than one copy names gates only or is copies all alike. -/
theorem memoOf_reps (P : Prog) (memo : Array Desc.Summary) (h : memoOf P = .ok memo) (f : Nat) :
    AllReps (memo[f]!).ret.segs := by
  unfold memoOf at h
  extract_lets seen memo0 at h
  have h' := ok_bind_pure h
  refine (okIs_forIn_range (I := fun m : Array Desc.Summary => ∀ f : Nat, AllReps (m[f]!).ret.segs) _ _ _
    (fun t r hr => ?_) (fun f => by rw [getElem!_neg _ f (by simp [memo0])]; exact allReps_empty)) memo h' f
  exact okIs_ite
    (fun _ => okIs_bind_of (J := fun x : Desc.Summary => AllReps x.ret.segs) (fun v hv => summarize_reps P r t v hv)
      (fun x hx => okIs_pure (memo_push_reps hr hx)))
    (fun _ => okIs_bind_of (J := fun x : Desc.Summary => AllReps x.ret.segs) (okIs_pure allReps_empty)
      (fun x hx => okIs_pure (memo_push_reps hr hx)))

theorem retOk_reps (P : Prog) (memo : Array Desc.Summary) (h : memoOf P = .ok memo) (f : Nat) :
    ∀ s ∈ (memo[f]!).ret.segs.toList, s.reps > 1 → (∀ g ∈ s.ps.toList, ∃ w, g.tgt = .gate w) ∨ s.shift = 0 :=
  fun s hs => (memoOf_reps P memo h f s hs).2

/-! ## The core's induction on fuel -/

/-- **`down`, `fromCallee`, `throughArg` and `carry` keep the gates they read, at every fuel**, wherever a batch of a
composite's piece, a scan's piece of one row and `carry` do one step up from them. -/
theorem specs (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (hB : Resolve.BodyOk c.P c.d)
    (hP1 : ∀ f : Nat, (c.P.defs[f]!).prim = true → (c.P.defs[f]!).outLeaves = 1)
    (hA : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size → ArgOk c k i)
    (hL0 : ∀ k i d L : Nat, ((c.body.nodes[k]!).args[i]!).dims = some (d, L) → L = 0 →
      i < (c.P.defs[(c.body.nodes[k]!).fn]!).body.params.size →
      ((c.P.defs[(c.body.nodes[k]!).fn]!).body.params[i]!).2.leaves = 0)
    (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn)
    (hBatch : ∀ fuel, DownSpec c sc fuel → CalleeSpec c sc fuel → BatchSpec c sc fuel)
    (hScan : ∀ fuel, CalleeSpec c sc fuel → CarrySpec c sc fuel → ScanSpec c sc fuel)
    (hCarry : ∀ fuel, CalleeSpec c sc fuel → ThroughSpec c sc fuel → CarrySpec c sc (fuel + 1)) :
    ∀ fuel, DownSpec c sc fuel ∧ CalleeSpec c sc fuel ∧ ThroughSpec c sc fuel ∧ CarrySpec c sc fuel
  | 0 => by
    refine ⟨fun q segs _ _ h => absurd h (down_zero c q segs), ?_, ?_, ?_⟩
    · intro _ _ _ _ _ _ _ h; rw [Desc.fromCallee.eq_1] at h; cases h
    · intro _ _ _ _ _ _ _ _ h; rw [Desc.throughArg.eq_1] at h; cases h
    · intro _ _ _ _ _ _ _ _ h; rw [Desc.carry.eq_1] at h; cases h
  | fuel + 1 =>
    have ⟨hD, hC, hT, hK⟩ := specs c sc hd hB hP1 hA hL0 hR hBatch hScan hCarry fuel
    ⟨downSpec_succ c sc hd hB hP1 fuel hD hC (hBatch fuel hD hC) (hScan fuel hC hK),
      calleeSpec_succ c sc hd fuel hR hT (throughReps c), throughSpec_succ c sc hd hB hA hL0 fuel hD hK, hCarry fuel hC hT⟩


/-! ## A batch of a composite's whole members, as copies of one member's return -/

/-- Leaves `[e, e + n)` of member `m`, `L` leaves to a member, are the node's leaves `m * L + e` on. -/
theorem block_at {α : Type} (V : Nat → Nat → α) {L : Nat} (hL : L ≠ 0) (m e n : Nat) (he : e + n ≤ L) :
    (List.range' e n).map (V m) = (List.range' (m * L + e) n).map fun j => V (j / L) (j % L) := by
  rw [List.range'_eq_map_range, List.range'_eq_map_range, List.map_map, List.map_map]
  apply List.map_congr_left
  intro t ht
  rw [List.mem_range] at ht
  have he' : e + t < L := by omega
  simp only [Function.comp]
  rw [show m * L + e + t = (e + t) + m * L by omega, Nat.add_mul_div_right _ _ (Nat.pos_of_ne_zero hL),
    Nat.div_eq_of_lt he', Nat.zero_add, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt he']

theorem range'_blocks (s L : Nat) : ∀ n : Nat,
    List.range' s (n * L) = (List.range n).flatMap fun u => List.range' (s + u * L) L
  | 0 => by simp
  | n + 1 => by
    rw [List.range_succ, List.flatMap_append, ← range'_blocks s L n, List.flatMap_singleton,
      List.range'_append_1, Nat.add_mul, Nat.one_mul]

/-- `batchRest`'s test that a return names gates only, one copy to a segment. -/
theorem gatesOnly_spec {segs : Array Seg}
    (h : (segs.all fun s => decide (s.reps ≤ 1) && s.ps.all fun g => match g.tgt with
      | .gate _ => true
      | _ => false) = true) :
    ∀ x ∈ segs.toList, x.reps ≤ 1 ∧ ∀ g ∈ x.ps.toList, ∃ w, g.tgt = .gate w := by
  rw [Array.all_eq_true_iff_forall_mem] at h
  intro x hx
  have := h x (Array.mem_toList_iff.1 hx)
  simp only [Bool.and_eq_true, decide_eq_true_eq] at this
  refine ⟨this.1, fun g hg => ?_⟩
  have h2 := this.2
  rw [Array.all_eq_true_iff_forall_mem] at h2
  have := h2 g (Array.mem_toList_iff.1 hg)
  revert this
  cases g.tgt with
  | gate w => intro _; exact ⟨w, rfl⟩
  | ref b i => intro h; cases h

/-- A return of one copy to a segment names its pieces' positions in order. -/
theorem psPos_flat (s : PS) (hok : PSOk s) (h1 : ∀ x ∈ s.segs.toList, x.reps ≤ 1) :
    psPos s = posAll (s.segs.flatMap (·.ps)) := by
  unfold psPos segsPos posAll
  rw [Array.toList_flatMap, List.flatMap_assoc]
  apply flatMap_congr
  intro x hx
  have hle := h1 x hx
  obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hx
  have hi' : i < s.segs.size := by simpa using hi
  have hc := (hok.1 i hi').2
  rw [getElem!_pos s.segs i hi'] at hc
  have h0 := reps_ne_zero (s := s.segs.toList[i]) (by simp only [Array.getElem_toList]; omega)
  rw [segPos_one_rep (by omega)]
  rfl

/-- Whole members `[m0, m1)` of a batch of a composite whose return names gates only: copies of one member's return,
each moved by its member's offset. -/
theorem copies_seg_agree (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) {k : Nat}
    {node : Flock.Program.Node} (hn : c.node k = .ok node) (hR : RetOk c.P c.memo node.fn)
    (hL : (c.P.defs[node.fn]!).outLeaves ≠ 0)
    (hOL : (c.P.defs[node.fn]!).outLeaves = (c.P.defs[node.fn]!).body.ret.count)
    (hgo : ∀ x ∈ (c.memo[node.fn]!).ret.segs.toList, x.reps ≤ 1 ∧ ∀ g ∈ x.ps.toList, ∃ w, g.tgt = .gate w)
    (m0 m1 : Nat) :
    Agree c.P sc (segPos (Desc.Seg.mk
        (Desc.shiftGates ((c.body.offsets[k]! : Int) + (m0 : Int) * ((c.P.defs[node.fn]!).gates : Int))
          ((c.memo[node.fn]!).ret.segs.flatMap (·.ps)))
        (c.P.defs[node.fn]!).outLeaves (m1 - m0) ((c.P.defs[node.fn]!).gates : Int)))
      ((List.range' (m0 * (c.P.defs[node.fn]!).outLeaves) ((m1 - m0) * (c.P.defs[node.fn]!).outLeaves)).map fun j =>
        Flock.Program.resolve c.P (c.P.child sc k (j / (c.P.defs[node.fn]!).outLeaves))
          ((c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.get (j % (c.P.defs[node.fn]!).outLeaves))) := by
  obtain ⟨hok, hnn, -, hag⟩ := hR
  have hflat := psPos_flat _ hok (fun x hx => (hgo x hx).1)
  unfold segPos
  simp only [copy_pos, shiftGates_pos, List.map_map, shiftPos_shiftPos]
  rw [range'_blocks, List.map_flatMap]
  apply agree_flatMap
  intro u hu
  rw [List.mem_range] at hu
  have hb := block_at (fun m e => Flock.Program.resolve c.P (c.P.child sc k m)
    ((c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.get e)) hL (m0 + u) 0 (c.P.defs[node.fn]!).outLeaves (by omega)
  rw [show m0 * (c.P.defs[node.fn]!).outLeaves + u * (c.P.defs[node.fn]!).outLeaves =
    (m0 + u) * (c.P.defs[node.fn]!).outLeaves + 0 by ring, ← hb, ← hflat, ← List.range_eq_range', callee_eq c hn,
    hOL]
  have hsc : c.P.child sc k (m0 + u) = .child sc node.fn
      (sc.off + c.body.offsets[k]! + (m0 + u) * (c.P.defs[node.fn]!).gates) k (m0 + u) := by
    rw [child_eq c sc hd hn, hd]; rfl
  have ho : (0 : Int) ≤ (c.body.offsets[k]! : Int) + (m0 : Int) * ((c.P.defs[node.fn]!).gates : Int) +
      (u : Int) * ((c.P.defs[node.fn]!).gates : Int) := by positivity
  have hoff : (c.P.child sc k (m0 + u)).off = sc.off + ((c.body.offsets[k]! : Int) +
      (m0 : Int) * ((c.P.defs[node.fn]!).gates : Int) + (u : Int) * ((c.P.defs[node.fn]!).gates : Int)).toNat := by
    rw [hsc, show ((c.body.offsets[k]! : Int) + (m0 : Int) * ((c.P.defs[node.fn]!).gates : Int) +
      (u : Int) * ((c.P.defs[node.fn]!).gates : Int)) =
      ((c.body.offsets[k]! + (m0 + u) * (c.P.defs[node.fn]!).gates : Nat) : Int) by push_cast; ring,
      Int.toNat_natCast, Nat.add_assoc]
    rfl
  refine agree_trans (agree_shift c.P ho hoff _ fun x hx => ?_) (hag _ (by rw [hsc]; rfl))
  refine ⟨?_, (hnn x hx).1⟩
  rw [hflat] at hx
  obtain ⟨g, hg, hxg⟩ := List.mem_flatMap.1 hx
  rw [Array.toList_flatMap] at hg
  obtain ⟨s, hs, hgs⟩ := List.mem_flatMap.1 hg
  obtain ⟨w, hw⟩ := (hgo s hs).2 g hgs
  exact ⟨w, (mem_pos_tgt hxg).trans hw⟩

theorem segsPos_push_append (X Y : Array Seg) (C : Seg) :
    segsPos (X.push C ++ Y) = segsPos X ++ segPos C ++ segsPos Y := by
  rw [segsPos_append, Array.push_eq_append, segsPos_append, segsPos_singleton]

theorem segsPos_push (X : Array Seg) (C : Seg) : segsPos (X.push C) = segsPos X ++ segPos C := by
  rw [Array.push_eq_append, segsPos_append, segsPos_singleton]

/-! ## A batch of a composite's piece in its one-run case -/

theorem ppt_affine (q : Piece) (w : Nat) (A F : Int) (t : Nat) :
    ppt (q.affine w A F) t = A + ppt q t * F := by
  unfold ppt Desc.Piece.affine
  simp only
  ring

/-- **A batch of a composite's piece in its one-run case** keeps the gates it reads: the node's leaf `e = m·L + t` is
member `m`'s return leaf `t`, the gate `g.base + t·g.stride` of member `m`'s activation, which is the run's gate
`o + g.base + e·F`. -/
theorem batchAffine (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d)
    (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn)
    (hOL : ∀ k node, c.node k = .ok node → (c.P.defs[node.fn]!).prim = false →
      (c.P.defs[node.fn]!).outLeaves = (c.P.defs[node.fn]!).body.ret.count) : BatchAffine c sc := by
  intro q k node p ht hn hform hp hL h0 hq hf ha
  obtain ⟨hok, -, -, hag⟩ := hR k node hn
  have hol := hOL k node hn hp
  unfold affineOf at ha
  dsimp only at ha
  split at ha
  swap
  · cases ha
  rename_i hc
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hc
  obtain ⟨⟨hsz, hreps⟩, hps⟩ := hc
  obtain ⟨s0, hs0⟩ := Array.size_eq_one_iff.1 hsz
  obtain ⟨g0, hg0⟩ := Array.size_eq_one_iff.1 (by rw [hs0] at hps; exact hps)
  rw [hs0] at ha hreps
  simp only [hg0] at ha
  have e1 : (#[s0] : Array Seg)[0]! = s0 := rfl
  have e2 : (#[g0] : Array Piece)[0]! = g0 := rfl
  rw [e1] at hreps
  simp only [e2] at ha
  rcases hgt : g0.tgt with _ | w
  · rw [hgt] at ha; cases ha
  rw [hgt] at ha
  dsimp only at ha
  split at ha
  swap
  · cases ha
  rename_i hc
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, Bool.or_eq_true] at hc
  obtain ⟨⟨hrows, hlen⟩, hFG⟩ := hc
  simp only [Option.some.injEq] at ha
  subst ha
  rw [e1] at hg0
  have hflat : psPos (c.memo[node.fn]!).ret = pos g0 := by
    rw [psPos_flat _ hok (fun x hx => by
      rw [hs0] at hx
      simp at hx
      subst hx; exact hreps), hs0]
    simp [hg0, posAll_singleton]
  have hob : (c.P.defs[sc.d]!).body.offsets[k]! = c.body.offsets[k]! := by rw [hd]; rfl
  have hnode : (c.P.defs[sc.d]!).body.nodes[k]! = node := by rw [hd]; exact node_get hn
  set L := (c.P.defs[node.fn]!).outLeaves with hLdef
  set G := (c.P.defs[node.fn]!).gates with hGdef
  have hL0 : 0 < L := Nat.pos_of_ne_zero hL
  rw [segsPos_singleton, segPos_one, pos_eq_map, pos_eq_map, List.map_map]
  unfold Agree
  apply forall₂_range_map
  intro t htc a hda
  simp only [Function.comp] at hda ⊢
  have hx0 : 0 ≤ ppt q t := hq _ (by rw [pos_eq_map]; exact List.mem_map.2 ⟨t, List.mem_range.2 htc, rfl⟩)
  obtain ⟨e, he⟩ : ∃ e : Nat, ppt q t = e := ⟨_, (Int.toNat_of_nonneg hx0).symm⟩
  rw [ht, he] at hda
  simp only [den] at hda
  rw [if_neg (by omega), Int.toNat_natCast] at hda
  have hb := den_batch c.P sc hnode hform hp hda
  rw [hob] at hb
  have hA := hag (.child sc node.fn (sc.off + c.body.offsets[k]! + e / L * G) k (e / L)) rfl
  rw [hflat, pos_eq_map] at hA
  have hl := hA.length_eq
  simp only [List.length_map, List.length_range] at hl
  have hr1 : g0.rows = 1 := by
    unfold Desc.Piece.count at hl
    rw [hlen, ← hol] at hl
    rcases Nat.lt_or_ge g0.rows 1 with h | h
    · rw [show g0.rows = 0 by omega, Nat.zero_mul] at hl; omega
    · omega
  have htl : e % L < L := Nat.mod_lt _ hL0
  have hcg : g0.count = L := by rw [count_one hr1, hlen]
  have hget := hA.get (i := e % L) (by simp only [List.length_map, List.length_range]; omega)
    (by simp only [List.length_map, List.length_range]; omega)
  simp only [List.get_eq_getElem, List.getElem_map, List.getElem_range] at hget
  have hv := hget a hb
  rw [hgt, ppt_one hr1 (by omega)] at hv
  simp only [den, Desc.Piece.at] at hv
  rw [ppt_affine, he]
  simp only [den, Desc.Piece.affine]
  have key : (c.body.offsets[k]! : Int) + g0.base + (e : Int) *
      (if (L == 1) = true then (G : Int) else g0.stride) =
      ((c.body.offsets[k]! + e / L * G : Nat) : Int) + (g0.base + ((e % L : Nat) : Int) * g0.stride) := by
    have hdm : ((e / L * L + e % L : Nat) : Int) = (e : Int) := congrArg (fun n : Nat => (n : Int)) (Nat.div_add_mod' e L)
    by_cases hL1 : L = 1
    · rw [if_pos (by simp [hL1])]
      rw [hL1, Nat.div_one, Nat.mod_one]
      push_cast
      ring
    · rw [if_neg (by simpa using hL1)]
      have hG : (G : Int) = (L : Int) * g0.stride := hFG.resolve_left hL1
      push_cast at hdm ⊢
      rw [hG]
      linear_combination (-g0.stride) * hdm
  by_cases hneg : g0.base + ((e % L : Nat) : Int) * g0.stride < 0
  · simp only [hneg, ↓reduceIte] at hv; cases hv
  simp only [hneg, ↓reduceIte, Except.ok.injEq] at hv
  rw [key]
  have hnn : ¬ (((c.body.offsets[k]! + e / L * G : Nat) : Int) +
      (g0.base + ((e % L : Nat) : Int) * g0.stride) < 0) := by omega
  simp only [hnn, ↓reduceIte, Except.ok.injEq]
  rw [← hv]
  show sc.off + _ = sc.off + c.body.offsets[k]! + e / L * G + _
  omega


/-- **Open: a batch of a composite's piece past its one-run case**, one step up: row by row, whole members as copies
of one member's return (of gates, or of leaves of arguments the members share, resolved once), one run of a mapped
argument through `throughPS`, a stride of whole members as copies of one member's leaf, or member by member and
position by position. -/
def BatchRest (c : Desc.Ctx) (sc : Flock.Program.Scope) (fuel : Nat) : Prop :=
  ∀ q segs k node, q.tgt = .ref false k → c.node k = .ok node → node.form = .batch →
    (c.P.defs[node.fn]!).prim = false → (c.P.defs[node.fn]!).outLeaves ≠ 0 → q.count ≠ 0 → q.flat = q →
    affineOf c k node q = none → (∀ x ∈ pos q, 0 ≤ x.2) → (∀ x ∈ pos q, Fits c x) →
    Desc.down c (fuel + 1) q = .ok segs → Agree c.P sc (segsPos segs) ((pos q).map (den c.P sc))

/-- **A batch of a composite's piece**, in its one-run case or past it. -/
theorem batchSpec_of (c : Desc.Ctx) (sc : Flock.Program.Scope) (fuel : Nat) (hAff : BatchAffine c sc)
    (hRest : BatchRest c sc fuel) : BatchSpec c sc fuel := by
  intro q segs k node ht hn hform hp h0 hff hq hf h
  by_cases hL : (c.P.defs[node.fn]!).outLeaves = 0
  · exact absurd h (down_batch0 c fuel q ht hff hn hform hp hL h0 segs)
  rcases ha : affineOf c k node q with _ | p
  · exact hRest q segs k node ht hn hform hp hL h0 hff ha hq hf h
  · rw [down_batch_affine c fuel q ht hff hn hform hp hL h0 ha] at h
    simp only [Except.ok.injEq] at h
    subst h
    exact hAff q k node p ht hn hform hp hL h0 hq hf ha

/-- **Open: `down`'s steps past what is proved here**, each one step up from the specs at its fuel: a batch of a
composite's piece past its one-run case (`BatchRest`) and `carry` (`CarrySpec`), which follows a scan's carry back
through its iterations. -/
def OpenSteps (c : Desc.Ctx) (sc : Flock.Program.Scope) : Prop :=
  (∀ fuel, DownSpec c sc fuel → CalleeSpec c sc fuel → BatchRest c sc fuel) ∧
    (∀ fuel, CalleeSpec c sc fuel → ThroughSpec c sc fuel → CarrySpec c sc (fuel + 1))

/-- **`down`, `fromCallee`, `throughArg` and `carry` keep the gates they read, at every fuel**, wherever the callees'
summaries hold (`RetOk`), a composite's return has as many leaves as its type, a scan's output is its final carry's
leaves and then its ys (`ScanOut`), the decoder's facts on the body do, and so do `down`'s open steps (`OpenSteps`). -/
theorem specs_of (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (hB : Resolve.BodyOk c.P c.d)
    (hP1 : ∀ f : Nat, (c.P.defs[f]!).prim = true → (c.P.defs[f]!).outLeaves = 1)
    (hA : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size → ArgOk c k i)
    (hL0 : ∀ k i d L : Nat, ((c.body.nodes[k]!).args[i]!).dims = some (d, L) → L = 0 →
      i < (c.P.defs[(c.body.nodes[k]!).fn]!).body.params.size →
      ((c.P.defs[(c.body.nodes[k]!).fn]!).body.params[i]!).2.leaves = 0)
    (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn)
    (hOL : ∀ k node, c.node k = .ok node → (c.P.defs[node.fn]!).prim = false →
      (c.P.defs[node.fn]!).outLeaves = (c.P.defs[node.fn]!).body.ret.count) (hSO : ScanOut c)
    (hO : OpenSteps c sc) :
    ∀ fuel, DownSpec c sc fuel ∧ CalleeSpec c sc fuel ∧ ThroughSpec c sc fuel ∧ CarrySpec c sc fuel :=
  specs c sc hd hB hP1 hA hL0 hR
    (fun fuel hD hC => batchSpec_of c sc fuel (batchAffine c sc hd hR hOL) (hO.1 fuel hD hC))
    (fun fuel hC hK => scanSpec_of c sc hd hB hSO fuel hC hK) hO.2


/-! ## The decoder's facts the core reads -/

/-- **Open: the decoder's facts the core reads**, as `Flock.Program.getDef` gives every Definition (its private
`composite` sets them): a primitive returns one leaf, a composite's return has as many leaves as its type, and each
node's argument is the sequence of its parts, none naming a negative position, a batch's or scan's argument of no leaves
to a member going to a parameter of none, where its callee's body has that parameter. -/
def Decoded (P : Prog) : Prop :=
  ∀ d < P.defs.size, ((P.defs[d]!).prim = true → (P.defs[d]!).outLeaves = 1) ∧
    ((P.defs[d]!).prim = false → (P.defs[d]!).outLeaves = (P.defs[d]!).body.ret.count) ∧
    ∀ k < (P.defs[d]!).body.nodes.size, ∀ i < ((P.defs[d]!).body.nodes[k]!).args.size,
      (((P.defs[d]!).body.nodes[k]!).args[i]!).seq =
        Seq.ofParts (((P.defs[d]!).body.nodes[k]!).args[i]!).seq.parts ∧
      (∀ q ∈ (((P.defs[d]!).body.nodes[k]!).args[i]!).seq.parts.toList, PartNonneg q) ∧
      ∀ e L : Nat, (((P.defs[d]!).body.nodes[k]!).args[i]!).dims = some (e, L) → L = 0 →
        i < (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).body.params.size →
        ((P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).body.params[i]!).2.leaves = 0

/-- **`down`, `fromCallee` and `throughArg` keep the gates they read in Definition `d`'s body, at every fuel**, wherever
the decoder's facts (`Decoded`, `ScanOut`), the body's references (`Resolve.BodyOk`) and its callees' summaries
(`RetOk`) hold. -/
theorem specs_decoded (P : Prog) (memo : Array Desc.Summary) {d : Nat} (hdP : d < P.defs.size) (hDec : Decoded P)
    (hB : Resolve.BodyOk P d) (hR : ∀ k node, (ctxOf P memo d).node k = .ok node → RetOk P memo node.fn)
    (hSO : ScanOut (ctxOf P memo d)) (sc : Flock.Program.Scope) (hd : sc.d = d)
    (hO : OpenSteps (ctxOf P memo d) sc) :
    ∀ fuel, DownSpec (ctxOf P memo d) sc fuel ∧ CalleeSpec (ctxOf P memo d) sc fuel ∧
      ThroughSpec (ctxOf P memo d) sc fuel ∧ CarrySpec (ctxOf P memo d) sc fuel := by
  have hfn : ∀ k node, (ctxOf P memo d).node k = .ok node → node.fn < P.defs.size := by
    intro k node hn
    have hk := node_lt hn
    have := (hB.1 k hk).1
    have hg : (P.defs[d]!).body.nodes[k]! = node := node_get hn
    rwa [hg] at this
  refine specs_of (ctxOf P memo d) sc hd hB ?_ ?_ ?_ hR ?_ hSO hO
  · intro f hf
    by_cases hlt : f < P.defs.size
    · exact (hDec f hlt).1 hf
    · change (P.defs[f]!).prim = true at hf
      rw [getElem!_neg P.defs f hlt] at hf; cases hf
  · intro k i hk hi
    have := (hDec d hdP).2.2 k hk i hi
    exact ctxOf_arg P memo d k i hk hi this.1 this.2.1
  · intro k i e L hdims hL hip
    change ((((P.defs[d]!).body.nodes[k]!).args[i]!).dims = some (e, L)) at hdims
    by_cases hk : k < (P.defs[d]!).body.nodes.size
    · by_cases hi : i < ((P.defs[d]!).body.nodes[k]!).args.size
      · exact ((hDec d hdP).2.2 k hk i hi).2.2 e L hdims hL hip
      · rw [getElem!_neg _ i hi] at hdims; cases hdims
    · rw [getElem!_neg _ k hk, getElem!_neg _ i (Nat.not_lt_zero i)] at hdims; cases hdims
  · intro k node hn hp
    exact (hDec node.fn (hfn k node hn)).2.1 hp


/-- **A batch member's spans** (`Desc.argSpans`) are its used parameter leaves' indices into the argument. -/
theorem argSpans_batch (node : Flock.Program.Node) (i m : Nat) (used : Desc.Ivs) (hf : node.form = .batch)
    (hL : ∀ d L, (node.args[i]!).dims = some (d, L) → L = 0 → ∀ p ∈ used.toList, p.2 ≤ p.1) :
    spansPos (Desc.argSpans node i m used) = (spansPos used).map (Resolve.memberIdx node.args[i]! m) := by
  unfold Desc.argSpans
  rw [hf]
  dsimp only
  rw [← Array.foldl_toList]
  have key : ∀ (l : List (Nat × Nat)) (acc : Array (Nat × Nat)), (∀ p ∈ l, ∀ d L,
      (node.args[i]!).dims = some (d, L) → L = 0 → p.2 ≤ p.1) →
      spansPos (l.foldl (fun acc (p : Nat × Nat) => acc ++ Desc.memberSpans node.args[i]! m p.1 p.2) acc) =
        spansPos acc ++ (l.flatMap fun p => List.range' p.1 (p.2 - p.1)).map (Resolve.memberIdx node.args[i]! m) := by
    intro l
    induction l with
    | nil => intro acc _; simp
    | cons p l ih =>
      intro acc hl
      rw [List.foldl_cons, ih _ (fun q hq => hl q (List.mem_cons_of_mem _ hq))]
      unfold spansPos
      rw [Array.toList_append, List.flatMap_append, List.flatMap_cons, List.map_append, List.append_assoc]
      congr 1
      have := memberSpans_idx node.args[i]! m p.1 p.2 (fun d L hd h0 => hl p List.mem_cons_self d L hd h0)
      unfold spansPos at this
      rw [this]
  have := key used.toList #[] (fun p hp d L hd h0 => hL d L hd h0 p hp)
  simp only [spansPos, List.flatMap] at this ⊢
  simpa using this

/-- An iteration's spans of a scan's `xs` are its used leaves `m · L` on. -/
theorem argSpans_xs (node : Flock.Program.Node) (i m : Nat) (used : Desc.Ivs) (hf : node.form = .scan)
    (hi : 1 ≤ i ∧ i ≤ node.nxs) :
    spansPos (Desc.argSpans node i m used) =
      (spansPos used).map fun t => m * (((node.args[i]!).dims.map (·.2)).getD 0) + t := by
  unfold Desc.argSpans
  rw [hf]
  simp only [show (1 ≤ i && i ≤ node.nxs) = true by simp [hi.1, hi.2], ite_true]
  unfold spansPos
  rw [Array.toList_map, List.flatMap_map, List.map_flatMap]
  apply flatMap_congr
  intro p _
  rw [show m * ((node.args[i]!).dims.map (·.2)).getD 0 + p.2 - (m * ((node.args[i]!).dims.map (·.2)).getD 0 + p.1) =
    p.2 - p.1 by omega, List.map_add_range']

/-- Any other argument's spans are its used leaves. -/
theorem argSpans_other (node : Flock.Program.Node) (i m : Nat) (used : Desc.Ivs) (hb : node.form ≠ .batch)
    (hs : node.form = .scan → ¬ (1 ≤ i ∧ i ≤ node.nxs)) : Desc.argSpans node i m used = used := by
  unfold Desc.argSpans
  rcases hf : node.form with _ | _ | _ | _
  all_goals first | exact absurd hf hb | skip
  all_goals dsimp only
  have := hs hf
  simp only [Bool.and_eq_true, decide_eq_true_eq]
  rw [if_neg this]


open private Array.qpartition.loop Array.qsort.sort from Init.Data.Array.QSort.Basic

/-! ## `Array.qsort` sorts by a key -/

section QSort

variable {α : Type} (f : α → Nat)

/-- `bs` agrees with `as` outside `[lo, hi]`, and each of its elements in `[lo, hi]` is one of `as`'s there. -/
def Within {n : Nat} (as bs : Vector α n) (lo hi : Nat) : Prop :=
  (∀ t (h : t < n), (t < lo ∨ hi < t) → bs[t] = as[t]) ∧
  (∀ t (h : t < n), lo ≤ t → t ≤ hi → ∃ s, ∃ hs : s < n, lo ≤ s ∧ s ≤ hi ∧ bs[t] = as[s])

theorem within_refl {n : Nat} (as : Vector α n) (lo hi : Nat) : Within as as lo hi :=
  ⟨fun _ _ _ => rfl, fun t h h1 h2 => ⟨t, h, h1, h2, rfl⟩⟩

theorem within_trans {n : Nat} {as bs cs : Vector α n} {lo hi : Nat} (h1 : Within as bs lo hi)
    (h2 : Within bs cs lo hi) : Within as cs lo hi := by
  refine ⟨fun t h ht => (h2.1 t h ht).trans (h1.1 t h ht), fun t h a b => ?_⟩
  obtain ⟨s, hs, s1, s2, e⟩ := h2.2 t h a b
  obtain ⟨u, hu, u1, u2, e'⟩ := h1.2 s hs s1 s2
  exact ⟨u, hu, u1, u2, e.trans e'⟩

theorem within_mono {n : Nat} {as bs : Vector α n} {lo hi lo' hi' : Nat} (h : Within as bs lo hi) (h1 : lo' ≤ lo)
    (h2 : hi ≤ hi') : Within as bs lo' hi' := by
  refine ⟨fun t ht h' => h.1 t ht (by omega), fun t ht a b => ?_⟩
  by_cases hin : lo ≤ t ∧ t ≤ hi
  · obtain ⟨s, hs, s1, s2, e⟩ := h.2 t ht hin.1 hin.2
    exact ⟨s, hs, by omega, by omega, e⟩
  · exact ⟨t, ht, a, b, h.1 t ht (by omega)⟩

theorem within_swap {n : Nat} (as : Vector α n) {lo hi i j : Nat} (hi' : i < n) (hj : j < n) (h1 : lo ≤ i)
    (h2 : i ≤ hi) (h3 : lo ≤ j) (h4 : j ≤ hi) : Within as (as.swap i j hi' hj) lo hi := by
  refine ⟨fun t h ht => ?_, fun t h a b => ?_⟩
  · rw [Vector.getElem_swap_of_ne (by omega) (by omega)]
  · rw [Vector.getElem_swap]
    split
    · exact ⟨j, hj, h3, h4, rfl⟩
    · split
      · exact ⟨i, hi', h1, h2, rfl⟩
      · exact ⟨t, h, a, b, rfl⟩

theorem within_ite_swap {n : Nat} (as : Vector α n) (b : Bool) {lo hi i j : Nat} (hi' : i < n) (hj : j < n)
    (h1 : lo ≤ i) (h2 : i ≤ hi) (h3 : lo ≤ j) (h4 : j ≤ hi) :
    Within as (if b then as.swap i j hi' hj else as) lo hi := by
  split
  · exact within_swap as hi' hj h1 h2 h3 h4
  · exact within_refl as lo hi

/-- `qpartition`'s loop: below `i` the elements are below the pivot, from `i` to `k` not, the pivot at `hi`; it ends
with the pivot at its result, below `hi` when some element from `i` is not below it. -/
theorem qpartition_loop_spec {n : Nat} {lo hi : Nat} (hhi : hi < n) (pivot : α) :
    ∀ (d : Nat) (as : Vector α n) (i k : Nat) (ilo : lo ≤ i) (ik : i ≤ k) (w : k ≤ hi), hi - k = d →
      (∀ t (h : t < n), lo ≤ t → t < i → f as[t] < f pivot) →
      (∀ t (h : t < n), i ≤ t → t < k → f pivot ≤ f as[t]) →
      as[hi] = pivot →
      (∃ j, ∃ hj : j < n, i ≤ j ∧ j < hi ∧ f pivot ≤ f as[j]) →
      let r := Array.qpartition.loop (fun x y => decide (f x < f y)) lo hi hhi pivot as i k ilo ik w
      ∃ hm : r.1.1 < hi, Within as r.2 lo hi ∧
        (∀ t (h : t < n), lo ≤ t → t < r.1.1 → f r.2[t] < f pivot) ∧
        r.2[r.1.1] = pivot ∧
        (∀ t (h : t < n), r.1.1 < t → t ≤ hi → f pivot ≤ f r.2[t]) := by
  intro d
  induction d using Nat.strongRecOn with
  | _ d ih =>
    intro as i k ilo ik w hd hlt hge hp hex
    dsimp only
    unfold Array.qpartition.loop
    split
    · rename_i hk
      split
      · rename_i hc
        have hc' : f as[k] < f pivot := of_decide_eq_true hc
        obtain ⟨hm, hw, h1, h2, h3⟩ := ih (hi - (k + 1)) (by omega) (as.swap i k (by omega) (by omega)) (i + 1)
          (k + 1) (by omega) (by omega) (by omega) rfl
          (fun t ht a b => by
            rw [Vector.getElem_swap]
            split
            · rename_i e; subst e; exact hc'
            · split
              · omega
              · exact hlt t ht a (by omega))
          (fun t ht a b => by
            rw [Vector.getElem_swap]
            split
            · omega
            · split
              · exact hge i (by omega) (Nat.le_refl _) (by omega)
              · exact hge t ht (by omega) (by omega))
          (by rw [Vector.getElem_swap_of_ne (by omega) (by omega)]; exact hp)
          (by
            obtain ⟨j, hj, j1, j2, hj'⟩ := hex
            by_cases hji : j = i
            · subst hji
              have hik : j ≠ k := fun e => by subst e; omega
              refine ⟨k, by omega, by omega, hk, ?_⟩
              rw [Vector.getElem_swap_right]; exact hj'
            · have hjk : j ≠ k := fun e => by subst e; omega
              refine ⟨j, hj, by omega, j2, ?_⟩
              rw [Vector.getElem_swap_of_ne hji hjk]; exact hj')
        exact ⟨hm, within_trans (within_swap as _ _ ilo (by omega) (by omega) (by omega)) hw, h1, h2, h3⟩
      · rename_i hc
        have hc' : ¬ f as[k] < f pivot := fun h => hc (decide_eq_true h)
        exact ih (hi - (k + 1)) (by omega) as i (k + 1) ilo (by omega) (by omega) rfl hlt
          (fun t ht a b => by
            by_cases e : t = k
            · subst e; omega
            · exact hge t ht a (by omega)) hp hex
    · rename_i hk
      have hkh : k = hi := by omega
      subst hkh
      obtain ⟨j, hj, j1, j2, hj'⟩ := hex
      refine ⟨by dsimp only; omega, within_swap as (by omega) hhi ilo (by omega) (by omega) (Nat.le_refl _), ?_, ?_, ?_⟩
      · intro t ht a b
        dsimp only at b ⊢
        rw [Vector.getElem_swap_of_ne (by omega) (by omega)]
        exact hlt t ht a b
      · dsimp only
        rw [Vector.getElem_swap_left]; exact hp
      · intro t ht a b
        dsimp only at a ⊢
        rw [Vector.getElem_swap]
        split
        · omega
        · split
          · exact hge i (by omega) (Nat.le_refl _) (by omega)
          · exact hge t ht (by omega) (by omega)

/-- `qpartition`'s third swap leaves its pivot at most the middle element. -/
theorem median_le {n : Nat} (as : Vector α n) (i j : Nat) (hi : i < n) (hj : j < n) :
    f (if decide (f as[i] < f as[j]) = true then as.swap i j hi hj else as)[j] ≤
      f (if decide (f as[i] < f as[j]) = true then as.swap i j hi hj else as)[i] := by
  split
  · rename_i hc
    rw [Vector.getElem_swap_left, Vector.getElem_swap_right]
    exact Nat.le_of_lt (of_decide_eq_true hc)
  · rename_i hc
    exact Nat.le_of_not_lt fun h => hc (decide_eq_true h)

theorem qpartition_core {n : Nat} (as as3 : Vector α n) (lo hi : Nat) (w : lo ≤ hi) (hhi : hi < n) (hlh : lo < hi)
    (hmid : (lo + hi) / 2 < n) (hW : Within as as3 lo hi) (hmed : f as3[hi] ≤ f as3[(lo + hi) / 2]) :
    let r := Array.qpartition.loop (fun x y => decide (f x < f y)) lo hi hhi as3[hi] as3 lo lo (Nat.le_refl _)
      (Nat.le_refl _) w
    r.1.1 < hi ∧ Within as r.2 lo hi ∧ ∃ v : Nat,
      (∀ t (h : t < n), lo ≤ t → t ≤ r.1.1 → f r.2[t] ≤ v) ∧ (∀ t (h : t < n), r.1.1 < t → t ≤ hi → v ≤ f r.2[t]) := by
  obtain ⟨hm, hw, h1, h2, h3⟩ := qpartition_loop_spec f hhi as3[hi] (hi - lo) as3 lo lo (Nat.le_refl _)
    (Nat.le_refl _) w rfl (fun t _ a b => by omega) (fun t _ a b => by omega) rfl
    ⟨(lo + hi) / 2, hmid, by omega, by omega, hmed⟩
  refine ⟨hm, within_trans hW hw, f as3[hi], fun t ht a b => ?_, h3⟩
  rcases Nat.lt_or_ge t (Array.qpartition.loop (fun x y => decide (f x < f y)) lo hi hhi as3[hi] as3 lo lo
    (Nat.le_refl _) (Nat.le_refl _) w).1.1 with b' | b'
  · exact Nat.le_of_lt (h1 t ht a b')
  · have e : t = (Array.qpartition.loop (fun x y => decide (f x < f y)) lo hi hhi as3[hi] as3 lo lo
      (Nat.le_refl _) (Nat.le_refl _) w).1.1 := by omega
    simp only [e, h2, Nat.le_refl]

/-- `qpartition` of `[lo, hi]`, `lo < hi`: its result `m` is below `hi`, the elements to `m` are at most `m`'s, after it
at least, and only `[lo, hi]` changes, to elements it held. -/
theorem qpartition_spec {n : Nat} (as : Vector α n) (lo hi : Nat) (w : lo ≤ hi) (hlo : lo < n) (hhi : hi < n)
    (hlh : lo < hi) :
    let r := Array.qpartition as (fun x y => decide (f x < f y)) lo hi w hlo hhi
    r.1.1 < hi ∧ Within as r.2 lo hi ∧ ∃ v : Nat,
      (∀ t (h : t < n), lo ≤ t → t ≤ r.1.1 → f r.2[t] ≤ v) ∧ (∀ t (h : t < n), r.1.1 < t → t ≤ hi → v ≤ f r.2[t]) := by
  dsimp only
  unfold Array.qpartition
  dsimp only
  refine qpartition_core f as _ lo hi w hhi hlh (by omega) ?_ ?_
  · refine within_trans (within_trans (within_ite_swap as _ _ _ ?_ ?_ ?_ ?_) (within_ite_swap _ _ _ _ ?_ ?_ ?_ ?_))
      (within_ite_swap _ _ _ _ ?_ ?_ ?_ ?_) <;> omega
  · exact median_le f _ _ _ _ _

/-- `qsort`'s `sort` of `[lo, hi]` sorts it by `f`, changing only it, to elements it held. -/
theorem sort_spec {n : Nat} :
    ∀ (d : Nat) (as : Vector α n) (lo hi : Nat) (w : lo ≤ hi) (hlo : lo < n) (hhi : hi < n), hi - lo = d →
      Within as (Array.qsort.sort (fun x y => decide (f x < f y)) as lo hi w hlo hhi) lo hi ∧
      ∀ i j (hi' : i < n) (hj : j < n), lo ≤ i → i ≤ j → j ≤ hi →
        f (Array.qsort.sort (fun x y => decide (f x < f y)) as lo hi w hlo hhi)[i] ≤
          f (Array.qsort.sort (fun x y => decide (f x < f y)) as lo hi w hlo hhi)[j] := by
  intro d
  induction d using Nat.strongRecOn with
  | _ d ih =>
    intro as lo hi w hlo hhi hd
    rw [Array.qsort.sort]
    split
    · rename_i h₁
      have hp := qpartition_spec f as lo hi w hlo hhi h₁
      revert hp
      match Array.qpartition as (fun x y => decide (f x < f y)) lo hi w hlo hhi with
      | (⟨mid, hmid⟩, as') =>
        intro hp
        dsimp only at hp ⊢
        obtain ⟨hm, hW, v, hle, hge⟩ := hp
        split
        · omega
        · obtain ⟨hW1, hS1⟩ := ih (mid - lo) (by omega) as' lo mid hmid.1 hlo (by omega) rfl
          obtain ⟨hW2, hS2⟩ := ih (hi - (mid + 1)) (by omega)
            (Array.qsort.sort (fun x y => decide (f x < f y)) as' lo mid hmid.1 hlo (by omega)) (mid + 1) hi
            (by omega) (by omega) hhi rfl
          refine ⟨within_trans hW (within_trans (within_mono hW1 (Nat.le_refl _) (by omega))
            (within_mono hW2 (by omega) (Nat.le_refl _))), fun i j hi' hj a b c => ?_⟩
          by_cases hjm : j ≤ mid
          · rw [hW2.1 i hi' (by omega), hW2.1 j hj (by omega)]
            exact hS1 i j hi' hj a b hjm
          · by_cases him : mid + 1 ≤ i
            · exact hS2 i j hi' hj him b c
            · rw [hW2.1 i hi' (by omega)]
              obtain ⟨s, hs, s1, s2, es⟩ := hW1.2 i hi' a (by omega)
              obtain ⟨u, hu, u1, u2, eu⟩ := hW2.2 j hj (by omega) c
              rw [es, eu, hW1.1 u hu (by omega)]
              exact Nat.le_trans (hle s hs s1 s2) (hge u hu (by omega) u2)
    · refine ⟨within_refl as lo hi, fun i j hi' hj a b c => ?_⟩
      have e : i = j := by omega
      subst e; exact Nat.le_refl _

/-- **`Array.qsort` by a key sorts by it.** -/
theorem qsort_sorted (as : Array α) :
    (as.qsort (fun x y => decide (f x < f y))).toList.Pairwise (fun x y => f x ≤ f y) := by
  unfold Array.qsort
  split
  · rename_i h
    rw [Array.size_eq_zero_iff.1 h]
    exact List.Pairwise.nil
  · rename_i h
    dsimp only
    obtain ⟨-, hS⟩ := sort_spec f _ as.toVector (min 0 (as.size - 1)) (max (min 0 (as.size - 1))
      (min (as.size - 1) (as.size - 1))) (by omega) (by omega) (by omega) rfl
    rw [List.pairwise_iff_getElem]
    intro i j hi hj hij
    simp only [Array.length_toList, Vector.size_toArray] at hi hj
    simp only [Array.getElem_toList, Vector.getElem_toArray]
    exact hS i j hi hj (by omega) (by omega) (by omega)

end QSort

/-! ## The used leaves' intervals (`Desc.Ivs.norm`, `Desc.Ivs.covers`) -/

/-- `Desc.Ivs.norm`'s step: an interval starting within the last one's extends it, else it follows. -/
def normStep (acc : Array (Nat × Nat)) (p : Nat × Nat) : Array (Nat × Nat) :=
  match acc.back? with
  | some (c, e) => if p.1 ≤ e then acc.pop.push (c, max p.2 e) else acc.push p
  | none => acc.push p

theorem norm_eq (xs : Array (Nat × Nat)) :
    Desc.Ivs.norm xs = ((xs.filter fun p => p.1 < p.2).qsort (fun x y => x.1 < y.1)).toList.foldl normStep #[] := by
  unfold Desc.Ivs.norm
  rw [← Array.foldl_toList]
  rfl

theorem inIvs_push (acc : Array (Nat × Nat)) (p : Nat × Nat) (a : Nat) :
    Width.inIvs (acc.push p).toList a ↔ Width.inIvs acc.toList a ∨ (p.1 ≤ a ∧ a < p.2) := by
  rw [Array.toList_push, Width.inIvs_append]
  simp [Width.inIvs]

/-- Every leaf `Desc.Ivs.norm`'s steps keep is in an interval they were given. -/
theorem normFold_sub : ∀ (l : List (Nat × Nat)) (acc : Array (Nat × Nat)) (a : Nat),
    Width.inIvs (l.foldl normStep acc).toList a → Width.inIvs acc.toList a ∨ Width.inIvs l a
  | [], acc, a, h => Or.inl h
  | p :: l, acc, a, h => by
    rw [List.foldl_cons] at h
    have hp : Width.inIvs (p :: l) a ↔ (p.1 ≤ a ∧ a < p.2) ∨ Width.inIvs l a := by
      rw [show p :: l = [p] ++ l from rfl, Width.inIvs_append]; simp [Width.inIvs]
    rw [hp]
    rcases normFold_sub l (normStep acc p) a h with h | h
    · unfold normStep at h
      rcases hb : acc.back? with _ | ⟨c, e⟩ <;> rw [hb] at h <;> dsimp only at h
      · rw [inIvs_push] at h; tauto
      · obtain ⟨ys, rfl⟩ := Array.back?_eq_some_iff.1 hb
        split at h
        · rename_i hpe
          rw [Array.pop_push, inIvs_push] at h
          rw [inIvs_push]
          rcases h with h | ⟨h1, h2⟩
          · exact Or.inl (Or.inl h)
          · by_cases hae : a < e
            · exact Or.inl (Or.inr ⟨h1, hae⟩)
            · exact Or.inr (Or.inl ⟨by omega, by omega⟩)
        · rw [inIvs_push] at h; tauto
    · exact Or.inr (Or.inr h)

/-- Over intervals sorted by their starts, `Desc.Ivs.norm`'s steps keep every leaf they are given. -/
theorem normFold_sup : ∀ (l : List (Nat × Nat)) (acc : Array (Nat × Nat)), l.Pairwise (fun x y => x.1 ≤ y.1) →
    (∀ p ∈ l, ∀ x, acc.back? = some x → x.1 ≤ p.1) → ∀ a,
      Width.inIvs acc.toList a ∨ Width.inIvs l a → Width.inIvs (l.foldl normStep acc).toList a
  | [], acc, _, _, a, h => by
    rcases h with h | ⟨iv, hiv, -⟩
    · exact h
    · simp at hiv
  | p :: l, acc, hs, hb, a, h => by
    rw [List.foldl_cons]
    have hs' := List.pairwise_cons.1 hs
    apply normFold_sup l (normStep acc p) hs'.2
    · intro q hq x hx
      unfold normStep at hx
      rcases hb' : acc.back? with _ | ⟨c, e⟩ <;> rw [hb'] at hx <;> dsimp only at hx
      · rw [Array.back?_push] at hx; cases hx; exact hs'.1 q hq
      · split at hx
        · rw [Array.back?_push] at hx; cases hx
          exact Nat.le_trans (hb p List.mem_cons_self (c, e) hb') (hs'.1 q hq)
        · rw [Array.back?_push] at hx; cases hx; exact hs'.1 q hq
    · have hp : Width.inIvs (p :: l) a ↔ (p.1 ≤ a ∧ a < p.2) ∨ Width.inIvs l a := by
        rw [show p :: l = [p] ++ l from rfl, Width.inIvs_append]; simp [Width.inIvs]
      rw [hp] at h
      unfold normStep
      rcases hb' : acc.back? with _ | ⟨c, e⟩ <;> dsimp only
      · rw [inIvs_push]; tauto
      · obtain ⟨ys, rfl⟩ := Array.back?_eq_some_iff.1 hb'
        have hc : c ≤ p.1 := hb p List.mem_cons_self _ hb'
        split
        · rename_i hpe
          rw [Array.pop_push, inIvs_push]
          rw [inIvs_push] at h
          rcases h with (h | ⟨h1, h2⟩) | (⟨h1, h2⟩ | h)
          · exact Or.inl (Or.inl h)
          · exact Or.inl (Or.inr ⟨h1, by omega⟩)
          · exact Or.inl (Or.inr ⟨by omega, by omega⟩)
          · exact Or.inr h
        · rw [inIvs_push]; tauto


theorem inIvs_nonempty (xs : Array (Nat × Nat)) (a : Nat) :
    Width.inIvs (xs.filter fun p => p.1 < p.2).toList a ↔ Width.inIvs xs.toList a := by
  simp only [Width.inIvs, Array.toList_filter, List.mem_filter, decide_eq_true_eq]
  constructor
  · rintro ⟨iv, ⟨hiv, -⟩, h⟩; exact ⟨iv, hiv, h⟩
  · rintro ⟨iv, hiv, h⟩; exact ⟨iv, ⟨hiv, by omega⟩, h⟩

/-- **Every leaf of `Desc.Ivs.norm`'s intervals is in one it was given.** -/
theorem norm_sub (xs : Array (Nat × Nat)) (a : Nat) (h : Width.inIvs (Desc.Ivs.norm xs).toList a) :
    Width.inIvs xs.toList a := by
  rw [norm_eq] at h
  rcases normFold_sub _ #[] a h with h | ⟨iv, hiv, h⟩
  · obtain ⟨iv, hiv, -⟩ := h; simp at hiv
  · rw [← inIvs_nonempty]
    exact ⟨iv, (Properties.RemoteAudit.qsort_toList_perm _ _).mem_iff.1 hiv, h⟩

/-- **`Desc.Ivs.norm` keeps exactly its intervals' leaves.** -/
theorem norm_mem (xs : Array (Nat × Nat)) (a : Nat) :
    Width.inIvs (Desc.Ivs.norm xs).toList a ↔ Width.inIvs xs.toList a := by
  refine ⟨norm_sub xs a, fun h => ?_⟩
  rw [norm_eq]
  apply normFold_sup _ #[] (qsort_sorted Prod.fst _) (fun p _ x hx => by simp at hx)
  rw [← inIvs_nonempty] at h
  obtain ⟨iv, hiv, h⟩ := h
  exact Or.inr ⟨iv, (Properties.RemoteAudit.qsort_toList_perm _ _).mem_iff.2 hiv, h⟩

/-- **What `Desc.Ivs.covers` checks**: every leaf below `n` is in an interval. -/
theorem covers_mem (s : Desc.Ivs) (n : Nat) (h : s.covers n = true) (a : Nat) (ha : a < n) :
    Width.inIvs s.toList a := by
  unfold Desc.Ivs.covers at h
  simp only [Bool.or_eq_true, beq_iff_eq, Bool.and_eq_true, decide_eq_true_eq] at h
  rcases h with h | ⟨⟨h1, h2⟩, h3⟩
  · omega
  · refine ⟨s[0]!, ?_, by omega, by omega⟩
    rw [getElem!_pos s 0 (by omega)]
    exact Array.getElem_mem_toList _

end NciCheck.Properties.SecurityProofs.ReadsExact
