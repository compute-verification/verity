import Proofs.Properties.Nci.Novel

/-!
# Toward `Novel.ReadsExact`: what `Desc.analyze`'s pieces and segments name

`Novel.ReadsExact` says that the segments `Desc.analyze` gives a root node name exactly the gates outside the node that
its gates read. Its proof reads each piece, segment and sequence of segments as the list of positions it names in order
(`pos`, `posAll`, `segPos`, `psPos`), a position being a target (a parameter's or a node's leaves, or the activation's
gates of one width) and an index, and holds each of `Desc`'s operations on them to the positions it means:

* a slice of pieces, of a segment or of a sequence of segments is its positions `[a, b)` (`slice_pos`,
  `slicePieces_pos`, `segSlice_pos`, `psSlice_pos`, `pieces_pos`), and `shiftGates` moves gate positions
  (`shiftGates_pos`);
* `Desc.coalesce` names a return's positions in the same order, each segment it gives one it was given or one piece
  of positive count (`coalesce_pos`, `coalesce_mem`);
* `PS.of` lays its segments end to end (`of_ok`), `ofPart` and `ofSeq` name a decoded sequence's references in order
  (`ofSeq_pos`, `partsPos_get`), where no strided part names a negative position, which the decoder ensures
  (`parsePartsIn_nonneg`), and `memberSpans` are a batch member's indices into its argument (`memberSpans_idx`);
* a context's arguments name their references (`ctxOf_arg`), and a loop appending segments names its elements'
  positions in order, each segment it adds from its element (`forIn_append_pos`, `forIn_pm`);
* `nodeReads` gives a primitive's node its arguments' positions read whole (`nodeReads_prim`), and a composite's those
  its callee's `used` names, as the rows its members read (`Desc.memberRows`) or member by member, then a scan's
  carries (`Desc.carriesRead`), kept by `keep` (`nodeReads_comp`), each read piece by piece through `down` (`readPos`,
  `downPos`, `argPos`, `carryPos`), every segment one `down` gives a piece of an argument (`FromDown`) or of the rows
  (`FromRows`), or `carriesRead` gives (`FromCarry`); `Seg.split` reads copies all alike once and one copy piece by
  piece, naming the same positions (`mem_segPos_alike`, `mem_segPos_split`);
* `argSpans` is a member's used leaves' indices into its argument (`argSpans_batch`, `argSpans_xs`, `argSpans_other`);
* `Array.qsort` by a key sorts by it (`qsort_sorted`), and `Desc.Ivs.norm`, which sorts by `List.mergeSort`, keeps
  exactly its intervals' leaves (`norm_mem`), and `Desc.Ivs.covers` checks every leaf is in an interval (`covers_mem`);
* a segment's gates (`Width.segGates`) are its gate positions (`segGates_pos`);
* on the circuit's side, a gate's operands are its node's references resolved (`operands_mem`), and a gate of a
  primitive root node reads its member's operands in the root activation (`reads_rootPrim`);
* toward the core of the statements below, that `down` keeps the gate each position names in an activation (`den`): its
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
  gates or of copies all alike (`RepsOk`, `reps_fuel`), followed through `Desc`'s code (`OkIs`, `reps_succ`); so
  `ThroughReps` holds (`throughReps`) and so does `RetOk`'s third part of every summary `memoOf` makes (`retOk_reps`);
* `throughArg`'s step: a member's parameter leaves, read through its node's argument (whole, a batch member's spans, or
  a scan iteration's slice, or its carry through `carry`), keep the gates they resolve to in the member wherever
  `down` and `carry` do (`CarrySpec`, `one_agree`, `throughSpec_succ`);
* a scan's piece of one row, wherever its output is its final carry's leaves and then its ys (`ScanOut`, which the
  decoder's `buildNode` gives): stride 1 reads the carry's leaves through `carry` and the ys iteration by iteration
  through `fromCallee` (`down_scan_head`, `scanOne`, `ys_agree`, `scanYs_agree`), any other stride position by
  position (`scanPts`); `scanSpec_of`;
* `carry`'s step: a scan's return leaves of iteration `m` that the step's summary returns in place (`Desc.carryRuns`,
  `carryRuns_cover`) are the initial carry's leaves through `throughArg` (`init_agree`, `carry_init`), and any other
  follows carry leaf by carry leaf (`Desc.chainOf`, `chainOf_ok`) to the leaf a callee returns through `fromCallee`
  (`carry_chain`) or around a cycle back to the initial carry (`chainAt_cyc`); so `carry` one step up keeps what it
  reads wherever the step's summary holds `RetOk` and `fromCallee` and `throughArg` one step down do
  (`carrySpec_succ`);
* a batch of a composite's piece in its one-run case, the node's leaf `m·L + t` being the run's gate for member `m`'s
  return leaf `t` (`batchAffine`, `batchSpec_of`), and past it as below; so `down`, `fromCallee`,
  `throughArg` and `carry` keep the gates they read at every fuel (`specs`, `specs_of`) wherever the callees' summaries
  hold `RetOk`, a composite's return has as many leaves as its type, a primitive returns one leaf, a scan's output is
  `ScanOut`'s and the decoder's facts on the body hold (`Resolve.BodyOk`, `ArgOk`, a batch's dims). A batch past its
  one-run case is proved: more than one row row by row (`down_batch_rows`); one row and stride 1 as copies of a
  member's return of gates (`copies_seg_agree`), of leaves of arguments the members share (`batchSharedMid`), or of one run of a mapped
  argument's leaves within one row of its axis through `throughPS` (`throughPS_some`, `batchThroughMid`), or member
  by member, the cut members at either end each through its return (`down_batch_one`, `ends_agree`,
  `batchOne_agree`); a stride of whole members as copies of a member's leaf of gates (`gateStride_agree`,
  `batchStride_agree`) or of leaves of arguments the members share (`batchShared`), which `fromCallee` reads the same
  for every member (`fromCallee_shared`, a slice's pieces having its segments' targets, `psSlice_tgt`); any other
  stride position by position (`down_batch_pts`, `batchPts_agree`). In a Definition's body, wherever its references
  (`Resolve.BodyOk`), its callees' summaries (`RetOk`) and the decoder's facts its private `composite` sets (`Decoded`,
  `ScanOut`) do (`specs_decoded`);
* so every summary `memoOf` makes of a reached Definition other than the root holds `RetOk` (`summarize_retOk`,
  `memoOf_retOk`), callees first (`reached` marks a composite's callees, `reached_closed`), the segments `down` gives
  naming their `len` positions, at or past 0 and none a node's (`DownShape`, `downShape_decoded`; a batch's piece of
  one row and stride 1 by `batchOne_shape`, a stride of whole members by `batchStride_shape`), wherever besides those
  a composite's return is the sequence of its parts and a primitive's none (`RetParts`, the decoder's).

So `ReadsExact` holds wherever one statement about the segments `Desc.nodeReads` gives each root node does
(`readsExact_of_rootReads`): `RootReads`. That holds wherever two smaller ones do (`rootReads_of`), named `Prop`s here:
`RootSegs`, that every read `Seg.split` makes of a segment `down` or `carriesRead` gives a root node names gates
of one root node with their widths, and `RootSet` (`BodiesOk.rootSet`), that the positions the node reads (`rootRead`) name exactly the
gates outside the node that its gates read; a composite's segment `keep` keeps is then one outside the node
(`keep_iff`). `RootSegs` holds wherever the decoder's facts do (`RootFacts`) and those reads name gates of one
root node with their widths (`RootGates`) (`rootSegs_of`): every segment is `ShapeOk` (`rootShape`; a scan's carries'
by `carriesRead_shape`), so each read is `SegGood` (`segGood_of`). `RootGates` holds wherever every such
segment is `SegNodes`, its gates of their widths, each piece's in one root node and a shifting segment's copies' in
one (`RootNodes`, `rootGates_of`).
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

theorem segPos_one_rep {s : Seg} (h : s.reps = 1) : segPos s = posAll s.ps := by
  unfold segPos
  rw [h, List.range_one, List.flatMap_singleton, copy_pos]
  simp [shiftPos_zero]

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

theorem form_eq_scan {f : Flock.Program.Form} (h : (f == .scan) = true) : f = .scan := by
  cases f <;> first | rfl | cases h

/-! ## A summary's return coalesced

`Desc.coalesce` joins a return's one-piece segments where they continue one progression (`Piece.join`) and stacks equal
runs as rows (`Piece.stack`): it names the same positions in the same order (`coalesce_pos`), and each segment it gives
is one it was given or one piece of positive count (`coalesce_mem`). -/

theorem segsPos_push (X : Array Seg) (C : Seg) : segsPos (X.push C) = segsPos X ++ segPos C := by
  rw [Array.push_eq_append, segsPos_append, segsPos_singleton]

theorem gate_beq_eq {w : Nat} {t : Tgt} (h : (Tgt.gate w == t) = true) : t = .gate w := by
  cases t with
  | ref p i => cases h
  | gate w' =>
    have e : (Tgt.gate w == Tgt.gate w') = (w == w') := rfl
    rw [e, beq_iff_eq] at h
    rw [h]

/-- **Joining two runs of a progression** names `p`'s positions then `q`'s. -/
theorem join_pos (p q r : Piece) (h : Desc.Piece.join p q = some r) : pos r = pos p ++ pos q ∧ 0 < r.count := by
  rcases p with ⟨tp, bp, sp, lp, rp, rsp⟩
  rcases q with ⟨tq, bq, sq, lq, rq, rsq⟩
  unfold Desc.Piece.join at h
  cases tp with
  | ref _ _ => cases h
  | gate w =>
    dsimp only at h
    generalize hst0 : (if lp == 1 then bq - bp else sp) = st at h
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, Bool.or_eq_true] at hc
      obtain ⟨⟨⟨⟨⟨⟨⟨htq, hrp⟩, hrq⟩, hlp⟩, hlq⟩, -⟩, hst⟩, hb⟩ := hc
      cases h
      cases gate_beq_eq htq
      subst hrp hrq
      refine ⟨?_, by simp [Desc.Piece.count]; omega⟩
      rw [pos_one, pos_one, pos_one, List.range_add, List.map_append, List.map_map]
      have hfirst : ((List.range lp).map fun (k : Nat) => (Tgt.gate w, bp + (k : Int) * st)) =
          (List.range lp).map fun (k : Nat) => (Tgt.gate w, bp + (k : Int) * sp) := by
        by_cases h1 : lp = 1
        · subst h1; simp
        · have h1' : (lp == 1) = false := by simpa using h1
          simp only [h1', Bool.false_eq_true, ite_false] at hst0
          rw [hst0]
      rw [hfirst]
      congr 1
      apply List.map_congr_left
      intro k hk
      simp only [Function.comp, Prod.mk.injEq, true_and]
      rcases hst with hst | hst
      · subst hst
        have : k = 0 := by simpa using hk
        subst this
        simp only [Nat.add_zero, Int.natCast_zero, Int.zero_mul, Int.add_zero]
        omega
      · subst hst
        rw [Int.natCast_add, Int.add_mul]
        omega
    · cases h

/-- **Stacking a row under rows** names `p`'s positions then `q`'s. -/
theorem stack_pos (p q r : Piece) (h : Desc.Piece.stack p q = some r) : pos r = pos p ++ pos q ∧ 0 < r.count := by
  rcases p with ⟨tp, bp, sp, lp, rp, rsp⟩
  rcases q with ⟨tq, bq, sq, lq, rq, rsq⟩
  unfold Desc.Piece.stack at h
  cases tp with
  | ref _ _ => cases h
  | gate w =>
    dsimp only at h
    generalize hrs0 : (if rp == 1 then bq - bp else rsp) = rs at h
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, Bool.or_eq_true] at hc
      obtain ⟨⟨⟨⟨⟨⟨⟨htq, hrp⟩, hrq⟩, hlp⟩, hlq⟩, hst⟩, -⟩, hb⟩ := hc
      cases h
      cases gate_beq_eq htq
      subst hrq hlq
      refine ⟨?_, by simp only [Desc.Piece.count]; exact Nat.mul_pos (Nat.succ_pos _) hlp⟩
      rw [pos_one]
      unfold pos
      dsimp only
      rw [List.range_succ, List.flatMap_append, List.flatMap_singleton]
      have hfirst : ((List.range rp).flatMap fun (r' : Nat) => (List.range lq).map fun (t : Nat) =>
            (Tgt.gate w, bp + (r' : Int) * rs + (t : Int) * sp)) =
          (List.range rp).flatMap fun (r' : Nat) => (List.range lq).map fun (t : Nat) =>
            (Tgt.gate w, bp + (r' : Int) * rsp + (t : Int) * sp) := by
        by_cases h1 : rp = 1
        · subst h1; simp
        · have h1' : (rp == 1) = false := by simpa using h1
          simp only [h1', Bool.false_eq_true, ite_false] at hrs0
          rw [hrs0]
      rw [hfirst]
      congr 1
      apply List.map_congr_left
      intro k hk
      simp only [Prod.mk.injEq, true_and]
      rcases hst with hst | hst
      · subst hst
        have : k = 0 := by simpa using hk
        subst this
        simp only [Int.natCast_zero, Int.zero_mul, Int.add_zero]
        omega
      · subst hst
        omega
    · cases h

/-- The piece of a segment of one copy of one piece names the segment's positions. -/
theorem single_pos (s : Seg) (p : Piece) (h : s.single = some p) : segPos s = pos p := by
  rcases s with ⟨⟨l⟩, len, reps, shift⟩
  unfold Desc.Seg.single at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, beq_iff_eq] at hc
    obtain ⟨hr, hsz⟩ := hc
    subst hr
    obtain ⟨a, rfl⟩ := List.length_eq_one_iff.1 (by simpa using hsz)
    have : a = p := by simpa using h
    subst this
    simp [segPos, Desc.Seg.copy, shiftGates_pos, shiftPos_zero, posAll_singleton]
  · cases h

/-- **`joinWith`** names `out`'s positions then `s`'s, wherever `f`'s joins name their two pieces'. -/
theorem joinWith_pos (f : Piece → Piece → Option Piece) (hf : ∀ p q r, f p q = some r → pos r = pos p ++ pos q)
    (out : Array Seg) (s : Seg) : segsPos (Desc.Seg.joinWith f out s) = segsPos out ++ segPos s := by
  unfold Desc.Seg.joinWith
  split
  · rename_i p q hp hq
    split
    · rename_i r hr
      obtain ⟨b, hb, hbp⟩ := Option.bind_eq_some_iff.1 hp
      obtain ⟨ys, rfl⟩ := Array.back?_eq_some_iff.1 hb
      rw [Array.pop_push, segsPos_push, segsPos_push, segPos_one, hf p q r hr, single_pos b p hbp,
        single_pos s q hq, List.append_assoc]
    · exact segsPos_push out s
  · exact segsPos_push out s

/-- A property each of `out`'s segments and `s` have, and every one-piece segment of positive count, each segment
`joinWith` gives has, wherever `f`'s joins are of positive count. -/
theorem joinWith_mem (f : Piece → Piece → Option Piece) (hf : ∀ p q r, f p q = some r → 0 < r.count)
    (Q : Seg → Prop) (hQ : ∀ r : Piece, 0 < r.count → Q (Seg.one r)) (out : Array Seg) (s : Seg)
    (hout : ∀ g ∈ out.toList, Q g) (hs : Q s) : ∀ g ∈ (Desc.Seg.joinWith f out s).toList, Q g := by
  have hpush : ∀ g ∈ (out.push s).toList, Q g := by
    intro g hg
    rw [Array.toList_push, List.mem_append, List.mem_singleton] at hg
    rcases hg with hg | rfl
    · exact hout g hg
    · exact hs
  unfold Desc.Seg.joinWith
  split
  · rename_i p q hp hq
    split
    · rename_i r hr
      obtain ⟨b, hb, -⟩ := Option.bind_eq_some_iff.1 hp
      obtain ⟨ys, rfl⟩ := Array.back?_eq_some_iff.1 hb
      intro g hg
      rw [Array.pop_push, Array.toList_push, List.mem_append, List.mem_singleton] at hg
      rcases hg with hg | rfl
      · exact hout g (by rw [Array.toList_push]; exact List.mem_append_left _ hg)
      · exact hQ r (hf p q r hr)
    · exact hpush
  · exact hpush

theorem foldl_joinWith_pos (f : Piece → Piece → Option Piece)
    (hf : ∀ p q r, f p q = some r → pos r = pos p ++ pos q) :
    ∀ (l : List Seg) (out : Array Seg), segsPos (l.foldl (Desc.Seg.joinWith f) out) = segsPos out ++ l.flatMap segPos
  | [], out => by simp
  | s :: l, out => by
    rw [List.foldl_cons, foldl_joinWith_pos f hf l, joinWith_pos f hf, List.flatMap_cons, List.append_assoc]

theorem foldl_joinWith_mem (f : Piece → Piece → Option Piece) (hf : ∀ p q r, f p q = some r → 0 < r.count)
    (Q : Seg → Prop) (hQ : ∀ r : Piece, 0 < r.count → Q (Seg.one r)) :
    ∀ (l : List Seg) (out : Array Seg), (∀ g ∈ out.toList, Q g) → (∀ g ∈ l, Q g) →
      ∀ g ∈ (l.foldl (Desc.Seg.joinWith f) out).toList, Q g
  | [], out, ho, _ => by simpa using ho
  | s :: l, out, ho, hl => by
    rw [List.foldl_cons]
    exact foldl_joinWith_mem f hf Q hQ l _ (joinWith_mem f hf Q hQ out s ho (hl s (by simp)))
      (fun g hg => hl g (by simp [hg]))

/-- **`coalesce`** names its segments' positions in order. -/
theorem coalesce_pos (segs : Array Seg) : segsPos (Desc.coalesce segs) = segsPos segs := by
  have e : ∀ f : Piece → Piece → Option Piece, (∀ p q r, f p q = some r → pos r = pos p ++ pos q) →
      ∀ x : Array Seg, segsPos (x.foldl (Desc.Seg.joinWith f) #[]) = segsPos x := by
    intro f hf x
    rw [← Array.foldl_toList, foldl_joinWith_pos f hf]
    rfl
  unfold Desc.coalesce
  rw [e _ (fun p q r h => (stack_pos p q r h).1), e _ (fun p q r h => (join_pos p q r h).1)]

/-- **Each segment `coalesce` gives is one of its segments, or one piece of positive count**: a property both have,
it has. -/
theorem coalesce_mem (Q : Seg → Prop) (hQ : ∀ r : Piece, 0 < r.count → Q (Seg.one r)) (segs : Array Seg)
    (h : ∀ g ∈ segs.toList, Q g) : ∀ g ∈ (Desc.coalesce segs).toList, Q g := by
  have e : ∀ f : Piece → Piece → Option Piece, (∀ p q r, f p q = some r → 0 < r.count) →
      ∀ x : Array Seg, (∀ g ∈ x.toList, Q g) → ∀ g ∈ (x.foldl (Desc.Seg.joinWith f) #[]).toList, Q g := by
    intro f hf x hx
    rw [← Array.foldl_toList]
    exact foldl_joinWith_mem f hf Q hQ _ _ (by simp) hx
  exact e _ (fun p q r h => (stack_pos p q r h).2) _ (e _ (fun p q r h => (join_pos p q r h).2) _ h)

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


/-- **`nodeReads`' segments at the root, as the positions they name.** On a Program `Desc.evaluate` reads, with
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

/-- A sequence as segments (`Desc.ofSeq`) has pieces each of positive count: one to a segment, empty ones dropped. -/
theorem ofSeq_count (s : Seq) : ∀ g ∈ (Desc.ofSeq s).segs.toList, ∀ q ∈ g.ps.toList, 0 < q.count := by
  intro g hg q hq
  unfold Desc.ofSeq at hg
  rw [of_segs, Array.toList_filter, List.mem_filter, Array.toList_map, List.mem_map] at hg
  obtain ⟨⟨p, -, rfl⟩, hc⟩ := hg
  have hq' : q = p := by simpa [Desc.Seg.one] using hq
  subst hq'
  have h1 : (Desc.Seg.one q).count > 0 := of_decide_eq_true hc
  have h2 : (Desc.Seg.one q).count = q.count := Nat.one_mul _
  omega

/-- **The context's arguments' pieces are of positive count** (`ofSeq_count`). -/
theorem ctxOf_argCount (P : Prog) (memo : Array Desc.Summary) (d k i : Nat)
    (hk : k < (P.defs[d]!).body.nodes.size) (hi : i < ((P.defs[d]!).body.nodes[k]!).args.size) :
    ∀ g ∈ ((ctxOf P memo d).arg k i).segs.toList, ∀ q ∈ g.ps.toList, 0 < q.count := by
  have harg : (ctxOf P memo d).arg k i = Desc.ofSeq (((P.defs[d]!).body.nodes[k]!).args[i]!).seq := by
    unfold Desc.Ctx.arg ctxOf
    dsimp only
    generalize (P.defs[d]!).body.nodes = ns at hk hi ⊢
    rw [getElem!_pos ns k hk] at hi ⊢
    rw [Array.getElem?_map, Array.getElem?_eq_getElem hk, Option.map_some, Option.bind_some, Array.getElem?_map,
      Array.getElem?_eq_getElem hi, Option.map_some, Option.getD_some, getElem!_pos _ i hi]
  rw [harg]
  exact ofSeq_count _

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
  node.form = .scan ∧ 2 ≤ node.n ∧
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

/-- **A primitive's node reads its arguments whole**, unless it is a batch of no members, each segment from `down`, as
reads (`Seg.split`: copies alike as one, one copy piece by piece). -/
theorem nodeReads_prim (c : Desc.Ctx) (k : Nat) (node : Flock.Program.Node) (hn : c.node k = .ok node)
    (hp : (c.P.defs[node.fn]!).prim = true) (rs : Array Seg) (h : Desc.nodeReads c k = .ok rs) :
    ∃ out : Array Seg, rs = out.flatMap Seg.split ∧
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

/-- **A composite's node reads its arguments where its callee reads its parameters**, and a scan's carries, as reads
(`Seg.split`: copies alike as one, one copy piece by piece), keeping those that don't start in the node; each segment
from `down` or `carriesRead`. -/
theorem nodeReads_comp (c : Desc.Ctx) (k : Nat) (node : Flock.Program.Node) (hn : c.node k = .ok node)
    (hp : (c.P.defs[node.fn]!).prim = false) (rs : Array Seg) (h : Desc.nodeReads c k = .ok rs) :
    if node.members = 0 then rs = #[] else
      ∃ out : Array Seg, rs = (out.flatMap Seg.split).filter (keep c k) ∧
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
      · simp only [Bool.and_eq_true, decide_eq_true_eq] at hsc
        exact Or.inr (Or.inr ⟨form_eq_scan hsc.1, hsc.2, cr, hcr, hs⟩)
    · rename_i hsc
      simp only [pure, Except.pure, Except.ok.injEq] at h
      refine ⟨r1, h.symm, by rw [hr1, carryPos, if_neg hsc, List.append_nil], fun s hs => ?_⟩
      rcases hm1' s hs with h | h
      · exact Or.inl h
      · exact Or.inr (Or.inl h)


/-! ## `RootReads` from the segments `down` gives at the root and the positions read

`nodeReads` gives each root node's segments from `down` and a scan's carries (`nodeReads_prim`, `nodeReads_comp`), so
`RootReads` holds wherever two smaller statements do (`rootReads_of`): `RootSegs`, that each read `Seg.split` makes of
such a segment names gates of one root node with their widths, and `RootSet`, that the positions read name exactly the
gates outside the node that its gates read. A composite's read `keep` keeps is then one outside the node (`keep_iff`).
-/

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

/-- **The segments `down` gives at the root, as reads.** On a Program `Desc.evaluate` reads, with its summaries
`memo` and `circuitOf`'s circuit `c`: for root node `k`, whose segments (`Desc.nodeReads`) have pieces naming gates only,
every read `Seg.split` makes of a segment `down` gives a piece of one of its arguments (or one `Desc.memberRows` gives),
or `Desc.carriesRead` gives of its scan's carries, is `SegGood`. Open past `rootSegs_of` and `rootGates_of`:
`RootNodes`. -/
def RootSegs : Prop :=
  ∀ (bytes : ByteArray) (P : Prog) (memo : Array Desc.Summary) (c : Circuit),
    Flock.Program.decode bytes = .ok P → P.orderViolation = none → Desc.rootLaidOut P = true →
    Desc.inputsAtRoot P = true → Desc.walkable P = true → memoOf P = .ok memo → circuitOf P = .ok c →
    ∀ k < nodeCount P, ∀ node rs, (rootCtx P memo).node k = .ok node → Desc.nodeReads (rootCtx P memo) k = .ok rs →
      (∀ s ∈ rs, ∀ q ∈ s.ps, ∃ w, q.tgt = .gate w) →
      ∀ s, ((∃ i, FromDown (rootCtx P memo) k i s) ∨ FromRows (rootCtx P memo) k node s ∨
        FromCarry (rootCtx P memo) k node s) → ∀ g ∈ (Seg.split s).toList, SegGood P c g

/-- **The positions a root node reads.** With `RootSegs`' premises: the gates the positions root node `k` reads
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

/-- The reads `Seg.split` makes of a segment name what it names. -/
theorem mem_segPos_split (s : Seg) (x : Tgt × Int) :
    (∃ g ∈ (Seg.split s).toList, x ∈ segPos g) ↔ x ∈ segPos s := by
  rw [← mem_segPos_alike s x]
  unfold Desc.Seg.split
  dsimp only
  split
  · rename_i h
    rw [segPos_one_rep (beq_iff_eq.1 h)]
    simp only [Array.toList_map, List.mem_map, posAll, List.mem_flatMap]
    constructor
    · rintro ⟨g, ⟨q, hq, rfl⟩, hx⟩
      exact ⟨q, hq, by rwa [segPos_one] at hx⟩
    · rintro ⟨q, hq, hx⟩
      exact ⟨Seg.one q, ⟨q, hq, rfl⟩, by rwa [segPos_one]⟩
  · simp

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
      have hgood : ∀ s ∈ out, ∀ g ∈ (Seg.split s).toList, SegGood P c g := fun s hs => hS' s (hsrc s hs)
      have hR : rootRead P memo node k = segsPos out := by
        rw [rootRead, if_neg (by rw [hp]; decide), if_neg hm0, hpos]
      refine ⟨fun g hg => ?_, fun a => ?_⟩
      · obtain ⟨s, hs, hgs⟩ := Array.mem_flatMap.1 (Array.mem_filter.1 hg).1
        exact hgood s hs g (Array.mem_toList_iff.2 hgs)
      rw [hR]
      constructor
      · rintro ⟨g, hg, hc0, w, hw⟩
        obtain ⟨hg1, hkeep⟩ := Array.mem_filter.1 hg
        obtain ⟨s, hs, hgs⟩ := Array.mem_flatMap.1 hg1
        have hgd := hgood s hs g (Array.mem_toList_iff.2 hgs)
        exact ⟨⟨w, List.mem_flatMap.2 ⟨s, Array.mem_toList_iff.2 hs,
            (mem_segPos_split s _).1 ⟨g, Array.mem_toList_iff.2 hgs, hw⟩⟩⟩,
          fun _ => (keep_iff P memo c hl hk hgd hc0 hw).1 hkeep⟩
      · rintro ⟨⟨w, hw⟩, hout⟩
        obtain ⟨s, hs, hw⟩ := List.mem_flatMap.1 hw
        have hso := Array.mem_toList_iff.1 hs
        obtain ⟨g, hgs, hwg⟩ := (mem_segPos_split s _).2 hw
        have hgd := hgood s hso g hgs
        have hc0 : g.count ≠ 0 := fun h0 => by rw [hgd.1 h0] at hwg; simp at hwg
        exact ⟨g, Array.mem_filter.2 ⟨Array.mem_flatMap.2 ⟨s, hso, Array.mem_toList_iff.1 hgs⟩,
          (keep_iff P memo c hl hk hgd hc0 hwg).2 (hout rfl)⟩, hc0, w, hwg⟩
  · obtain ⟨out, rfl, hpos, hsrc⟩ := nodeReads_prim (rootCtx P memo) k node hn hp rs hrs
    have hgood : ∀ s ∈ out, ∀ g ∈ (Seg.split s).toList, SegGood P c g := fun s hs => hS' s (Or.inl (hsrc s hs))
    have hR : rootRead P memo node k = segsPos out := by rw [rootRead, if_pos hp, hpos]
    refine ⟨fun g hg => ?_, fun a => ?_⟩
    · obtain ⟨s, hs, hgs⟩ := Array.mem_flatMap.1 hg
      exact hgood s hs g (Array.mem_toList_iff.2 hgs)
    rw [hR]
    constructor
    · rintro ⟨g, hg, -, w, hw⟩
      obtain ⟨s, hs, hgs⟩ := Array.mem_flatMap.1 hg
      exact ⟨⟨w, List.mem_flatMap.2 ⟨s, Array.mem_toList_iff.2 hs,
          (mem_segPos_split s _).1 ⟨g, Array.mem_toList_iff.2 hgs, hw⟩⟩⟩, fun h => absurd h (by simp)⟩
    · rintro ⟨⟨w, hw⟩, -⟩
      obtain ⟨s, hs, hw⟩ := List.mem_flatMap.1 hw
      have hso := Array.mem_toList_iff.1 hs
      obtain ⟨g, hgs, hwg⟩ := (mem_segPos_split s _).2 hw
      refine ⟨g, Array.mem_flatMap.2 ⟨s, hso, Array.mem_toList_iff.1 hgs⟩, fun h0 => ?_, w, hwg⟩
      rw [(hgood s hso g hgs).1 h0] at hwg
      simp at hwg


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

/-- A batch of a composite's piece of more than one row past its one-run case, read row by row. -/
theorem down_batch_rows (c : Desc.Ctx) (fuel : Nat) (q : Piece) {k : Nat} (ht : q.tgt = .ref false k)
    (hff : q.flat = q) {node : Flock.Program.Node} (hn : c.node k = .ok node) (hform : node.form = .batch)
    (hp : (c.P.defs[node.fn]!).prim = false) (hL : (c.P.defs[node.fn]!).outLeaves ≠ 0) (h0 : q.count ≠ 0)
    (ha : affineOf c k node q = none) (hr : q.rows > 1) :
    Desc.down c (fuel + 1) q = byRows c fuel q := by
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
  · rw [if_neg (by simp), if_pos hr]
    delta byRows Desc.down
    rfl
  · rw [if_pos rfl]
    cases hT : ((c.memo[node.fn]!).ret.segs[0]!.ps[0]!).tgt with
    | ref b i =>
      dsimp only
      rw [if_pos hr]
      delta byRows Desc.down
      rfl
    | gate w =>
      simp only [hT] at ha
      dsimp only
      rcases hC2 : (((c.memo[node.fn]!).ret.segs[0]!.ps[0]!).rows ≤ 1 &&
          ((c.memo[node.fn]!).ret.segs[0]!.ps[0]!).len == (c.P.defs[node.fn]!).outLeaves &&
          ((c.P.defs[node.fn]!).outLeaves == 1 || ((c.P.defs[node.fn]!).gates : Int) ==
            ((c.P.defs[node.fn]!).outLeaves : Int) * ((c.memo[node.fn]!).ret.segs[0]!.ps[0]!).stride)) with _ | _ <;>
        simp only [hC2, Bool.false_eq_true, ite_false, ite_true] at ha
      · rw [if_neg (by simp), if_pos hr]
        delta byRows Desc.down
        rfl
      · cases ha

/-- `down` at each of a piece's positions in turn. -/
def batchPts (c : Desc.Ctx) (fuel : Nat) (q : Piece) : Except String (Array Seg) := do
  Desc.walkAtMost "leaves of a batch's output read one by one" q.len
  let mut out := #[]
  for t in [0:q.len] do
    out := out ++ (← Desc.down c fuel { q with base := q.at t, stride := 1, len := 1 })
  return out

/-- A batch of a composite's piece of one row past its one-run case, of a stride neither 1 nor of whole members,
read position by position. -/
theorem down_batch_pts (c : Desc.Ctx) (fuel : Nat) (q : Piece) {k : Nat} (ht : q.tgt = .ref false k)
    (hff : q.flat = q) {node : Flock.Program.Node} (hn : c.node k = .ok node) (hform : node.form = .batch)
    (hp : (c.P.defs[node.fn]!).prim = false) (hL : (c.P.defs[node.fn]!).outLeaves ≠ 0) (h0 : q.count ≠ 0)
    (ha : affineOf c k node q = none) (hr : ¬ q.rows > 1) (hs : q.stride ≠ 1)
    (hcp : ¬ (q.len > 1 ∧ q.stride % ((c.P.defs[node.fn]!).outLeaves : Int) = 0 ∧ q.base ≥ 0)) :
    Desc.down c (fuel + 1) q = batchPts c fuel q := by
  have hs' : ¬ ((q.stride == 1) = true) := by simpa using hs
  have hcp' : ¬ ((decide (q.len > 1) && q.stride % ((c.P.defs[node.fn]!).outLeaves : Int) == 0 &&
      decide (q.base ≥ 0)) = true) := by
    simpa [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] using hcp
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
  · rw [if_neg (by simp), if_neg hr, if_neg hs', if_neg hcp']
    delta batchPts Desc.down
    rfl
  · rw [if_pos rfl]
    cases hT : ((c.memo[node.fn]!).ret.segs[0]!.ps[0]!).tgt with
    | ref b i =>
      dsimp only
      rw [if_neg hr, if_neg hs', if_neg hcp']
      delta batchPts Desc.down
      rfl
    | gate w =>
      simp only [hT] at ha
      dsimp only
      rcases hC2 : (((c.memo[node.fn]!).ret.segs[0]!.ps[0]!).rows ≤ 1 &&
          ((c.memo[node.fn]!).ret.segs[0]!.ps[0]!).len == (c.P.defs[node.fn]!).outLeaves &&
          ((c.P.defs[node.fn]!).outLeaves == 1 || ((c.P.defs[node.fn]!).gates : Int) ==
            ((c.P.defs[node.fn]!).outLeaves : Int) * ((c.memo[node.fn]!).ret.segs[0]!.ps[0]!).stride)) with _ | _ <;>
        simp only [hC2, Bool.false_eq_true, ite_false, ite_true] at ha
      · rw [if_neg (by simp), if_neg hr, if_neg hs', if_neg hcp']
        delta batchPts Desc.down
        rfl
      · cases ha

/-- `down`'s batch of a composite's piece of one row and a stride of whole members: copies of one member's leaf `j`
(of gates, or of leaves of arguments the members share, resolved once), else position by position. -/
def batchStride (c : Desc.Ctx) (fuel k : Nat) (node : Flock.Program.Node) (q : Piece) :
    Except String (Array Seg) := do
  let L := (c.P.defs[node.fn]!).outLeaves
  let G : Int := (c.P.defs[node.fn]!).gates
  let o : Int := c.body.offsets[k]!
  let j := q.base.toNat % L
  let m0 := q.base.toNat / L
  let leaf := (c.memo[node.fn]!).ret.slice j (j + 1)
  if leaf.all (·.reps ≤ 1) then
    let ps := leaf.flatMap (·.ps)
    if ps.all fun g => match g.tgt with | .gate _ => true | _ => false then
      return #[{ ps := Desc.shiftGates (o + m0 * G) ps, len := 1, reps := q.len, shift := q.stride / L * G }]
    if ps.all fun g => match g.tgt with
        | .ref true i => (node.args[i]?.map (·.dims.isNone)).getD false
        | _ => false then
      let one ← Desc.fromCallee c fuel k m0 j (j + 1)
      if one.all (·.reps ≤ 1) then return #[{ ps := one.flatMap (·.ps), len := 1, reps := q.len, shift := 0 }]
  batchPts c fuel q

/-- A batch of a composite's piece of one row past its one-run case, of a stride of whole members, as `batchStride`
reads it. -/
theorem down_batch_stride (c : Desc.Ctx) (fuel : Nat) (q : Piece) {k : Nat} (ht : q.tgt = .ref false k)
    (hff : q.flat = q) {node : Flock.Program.Node} (hn : c.node k = .ok node) (hform : node.form = .batch)
    (hp : (c.P.defs[node.fn]!).prim = false) (hL : (c.P.defs[node.fn]!).outLeaves ≠ 0) (h0 : q.count ≠ 0)
    (ha : affineOf c k node q = none) (hr : ¬ q.rows > 1) (hs : q.stride ≠ 1)
    (hcp : q.len > 1 ∧ q.stride % ((c.P.defs[node.fn]!).outLeaves : Int) = 0 ∧ q.base ≥ 0) :
    Desc.down c (fuel + 1) q = batchStride c fuel k node q := by
  have hs' : ¬ ((q.stride == 1) = true) := by simpa using hs
  have hcp' : (decide (q.len > 1) && q.stride % ((c.P.defs[node.fn]!).outLeaves : Int) == 0 &&
      decide (q.base ≥ 0)) = true := by
    simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq]
    exact ⟨⟨hcp.1, hcp.2.1⟩, hcp.2.2⟩
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
  · rw [if_neg (by simp), if_neg hr, if_neg hs', if_pos hcp']
    delta batchStride batchPts Desc.down Desc.fromCallee
    rfl
  · rw [if_pos rfl]
    cases hT : ((c.memo[node.fn]!).ret.segs[0]!.ps[0]!).tgt with
    | ref b i =>
      dsimp only
      rw [if_neg hr, if_neg hs', if_pos hcp']
      delta batchStride batchPts Desc.down Desc.fromCallee
      rfl
    | gate w =>
      simp only [hT] at ha
      dsimp only
      rcases hC2 : (((c.memo[node.fn]!).ret.segs[0]!.ps[0]!).rows ≤ 1 &&
          ((c.memo[node.fn]!).ret.segs[0]!.ps[0]!).len == (c.P.defs[node.fn]!).outLeaves &&
          ((c.P.defs[node.fn]!).outLeaves == 1 || ((c.P.defs[node.fn]!).gates : Int) ==
            ((c.P.defs[node.fn]!).outLeaves : Int) * ((c.memo[node.fn]!).ret.segs[0]!.ps[0]!).stride)) with _ | _ <;>
        simp only [hC2, Bool.false_eq_true, ite_false, ite_true] at ha
      · rw [if_neg (by simp), if_neg hr, if_neg hs', if_pos hcp']
        delta batchStride batchPts Desc.down Desc.fromCallee
        rfl
      · cases ha

/-- A batch's output leaves `[a, b)` read as whole members `[m0, m1)` (`mid`) and the cut members at either end each
through its return. -/
def batchEnds (c : Desc.Ctx) (fuel k L a b : Nat) (mid : Array Seg) : Except String (Array Seg) := do
  let (m0, m1) := ((a + L - 1) / L, b / L)
  let mut out := #[]
  if a < m0 * L then out := out ++ (← Desc.fromCallee c fuel k (a / L) (a % L) L)
  out := out ++ mid
  if m1 * L < b then out := out ++ (← Desc.fromCallee c fuel k m1 0 (b - m1 * L))
  return out

/-- A batch's output leaves `[a, b)` read member by member, each through its return. -/
def batchMembers (c : Desc.Ctx) (fuel k L a b : Nat) : Except String (Array Seg) := do
  Desc.walkAtMost "members of a batch read one by one" ((b - 1) / L + 1 - a / L)
  let mut out := #[]
  for m in [a / L:(b - 1) / L + 1] do
    out := out ++ (← Desc.fromCallee c fuel k m (max a (m * L) - m * L) (min b ((m + 1) * L) - m * L))
  return out

/-- A batch's output leaves `[a, b)` whose members each return one run of a mapped argument's leaves within one row
of its axis: the whole members' are that run in each member's row, one piece of rows through `throughPS`, else
member by member. -/
def batchThrough (c : Desc.Ctx) (fuel k : Nat) (node : Flock.Program.Node) (a b : Nat) :
    Except String (Array Seg) := do
  let L := (c.P.defs[node.fn]!).outLeaves
  let r := (c.memo[node.fn]!).ret
  let (m0, m1) := ((a + L - 1) / L, b / L)
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
            return ← batchEnds c fuel k L a b (← Desc.down c fuel bp)
  batchMembers c fuel k L a b

/-- `down`'s batch of a composite's piece of one row and stride 1 past its one-run case. -/
def batchOne (c : Desc.Ctx) (fuel k : Nat) (node : Flock.Program.Node) (q : Piece) : Except String (Array Seg) := do
  let L := (c.P.defs[node.fn]!).outLeaves
  let G : Int := (c.P.defs[node.fn]!).gates
  let o : Int := c.body.offsets[k]!
  let r := (c.memo[node.fn]!).ret
  let a := q.base.toNat
  let b := a + q.len
  let (m0, m1) := ((a + L - 1) / L, b / L)
  let gatesOnly := r.segs.all fun s => s.reps ≤ 1 && s.ps.all fun g => match g.tgt with
    | .gate _ => true
    | _ => false
  if gatesOnly && m0 + 1 < m1 then
    return ← batchEnds c fuel k L a b
      #[{ ps := Desc.shiftGates (o + m0 * G) (r.segs.flatMap (·.ps)), len := L, reps := m1 - m0, shift := G }]
  let shared := r.segs.all fun s => s.reps ≤ 1 && s.ps.all fun g => match g.tgt with
    | .ref true i => (node.args[i]?.map (·.dims.isNone)).getD false
    | _ => false
  if shared && m0 + 1 < m1 then
    let one ← Desc.fromCallee c fuel k m0 0 L
    if one.all (·.reps ≤ 1) then
      return ← batchEnds c fuel k L a b #[{ ps := one.flatMap (·.ps), len := L, reps := m1 - m0, shift := 0 }]
  batchThrough c fuel k node a b

/-- A batch of a composite's piece of one row and stride 1 past its one-run case, as `batchOne` reads it. -/
theorem down_batch_one (c : Desc.Ctx) (fuel : Nat) (q : Piece) {k : Nat} (ht : q.tgt = .ref false k)
    (hff : q.flat = q) {node : Flock.Program.Node} (hn : c.node k = .ok node) (hform : node.form = .batch)
    (hp : (c.P.defs[node.fn]!).prim = false) (hL : (c.P.defs[node.fn]!).outLeaves ≠ 0) (h0 : q.count ≠ 0)
    (ha : affineOf c k node q = none) (hr : ¬ q.rows > 1) (hs : q.stride = 1) :
    Desc.down c (fuel + 1) q = batchOne c fuel k node q := by
  have hs' : (q.stride == 1) = true := by simpa using hs
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
  · rw [if_neg (by simp), if_neg hr, if_pos hs']
    delta batchOne batchThrough batchEnds batchMembers Desc.down Desc.fromCallee
    rfl
  · rw [if_pos rfl]
    cases hT : ((c.memo[node.fn]!).ret.segs[0]!.ps[0]!).tgt with
    | ref b i =>
      dsimp only
      rw [if_neg hr, if_pos hs']
      delta batchOne batchThrough batchEnds batchMembers Desc.down Desc.fromCallee
      dsimp only
      rw [hT]
      rfl
    | gate w =>
      simp only [hT] at ha
      dsimp only
      rcases hC2 : (((c.memo[node.fn]!).ret.segs[0]!.ps[0]!).rows ≤ 1 &&
          ((c.memo[node.fn]!).ret.segs[0]!.ps[0]!).len == (c.P.defs[node.fn]!).outLeaves &&
          ((c.P.defs[node.fn]!).outLeaves == 1 || ((c.P.defs[node.fn]!).gates : Int) ==
            ((c.P.defs[node.fn]!).outLeaves : Int) * ((c.memo[node.fn]!).ret.segs[0]!.ps[0]!).stride)) with _ | _ <;>
        simp only [hC2, Bool.false_eq_true, ite_false, ite_true] at ha
      · rw [if_neg (by simp), if_neg hr, if_pos hs']
        delta batchOne batchThrough batchEnds batchMembers Desc.down Desc.fromCallee
        dsimp only
        rw [hT]
        rfl
      · cases ha

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

/-- A piece of one row read position by position through `down`, from what `down` keeps at each position. -/
theorem batchPts_agree (c : Desc.Ctx) (sc : Flock.Program.Scope) (fuel : Nat) (hD : DownSpec c sc fuel) (q : Piece)
    (h1 : q.rows = 1) (hq : ∀ x ∈ pos q, 0 ≤ x.2) (hf : ∀ x ∈ pos q, Fits c x) {segs : Array Seg}
    (h : batchPts c fuel q = .ok segs) : Agree c.P sc (segsPos segs) ((pos q).map (den c.P sc)) := by
  unfold batchPts at h
  have h := walkAtMost_bind _ _ _ h
  dsimp only at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at h
  rw [range'_size] at h
  have hpt : ∀ t, pos { q with base := q.at t, stride := 1, len := 1 } = [(q.tgt, q.at t)] := by
    intro t
    simp [pos, h1]
  have hA := loop_agree (P := c.P) (sc := sc)
    (fun t => Desc.down c fuel { q with base := q.at t, stride := 1, len := 1 })
    (fun t => [den c.P sc (q.tgt, q.at t)])
    (List.range q.len) (fun t htm segs hs => by
      have hm := at_mem_pos h1 (List.mem_range.1 htm)
      have := hD _ segs (by rw [hpt t]; intro x hx; simp only [List.mem_singleton] at hx; subst hx; exact hq _ hm)
        (by rw [hpt t]; intro x hx; simp only [List.mem_singleton] at hx; subst hx; exact hf _ hm) hs
      rwa [hpt t] at this) segs h
  apply agree_weaken hA
  rw [← List.map_eq_flatMap, pos_eq_map, List.map_map, count_one h1]
  apply forall₂_range_map
  intro t ht' a ha
  have hp := ppt_one h1 (t := t) (by rw [count_one h1]; exact ht')
  simp only [Function.comp] at ha ⊢
  rw [hp] at ha
  exact ha

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

theorem pieceSlice_tgt {q : Piece} {a b : Nat} {p : Piece} (h : p ∈ (q.slice a b).toList) : p.tgt = q.tgt := by
  unfold Desc.Piece.slice at h
  dsimp only at h
  split_ifs at h <;> simp [Desc.Piece.row] at h <;> aesop

theorem slGo_tgt (a b : Nat) : ∀ (l : List Piece) (st : Nat) (out : Array Piece) (p : Piece),
    p ∈ (slGo a b l st out).toList → p ∈ out.toList ∨ ∃ q ∈ l, p.tgt = q.tgt
  | [], _, _, _, h => Or.inl h
  | q :: qs, st, out, p, h => by
    unfold slGo at h
    split at h
    · exact Or.inl h
    rcases slGo_tgt a b qs _ _ p h with h | h
    · split at h
      · rw [Array.toList_append, List.mem_append] at h
        rcases h with h | h
        · exact Or.inl h
        · exact Or.inr ⟨q, List.mem_cons_self, pieceSlice_tgt h⟩
      · exact Or.inl h
    · obtain ⟨q', hq', e⟩ := h
      exact Or.inr ⟨q', List.mem_cons_of_mem _ hq', e⟩

theorem slicePieces_tgt {ps : Array Piece} {a b : Nat} {p : Piece} (h : p ∈ (Desc.slicePieces ps a b).toList) :
    ∃ q ∈ ps.toList, p.tgt = q.tgt := by
  rw [slicePieces_go] at h
  rcases slGo_tgt a b _ 0 #[] p h with h | h
  · simp at h
  · exact h

/-- A piece of a segment of `Seg.slice` has the target of one of `s`'s. -/
theorem segSlice_tgt (s : Seg) (a b : Nat) (g : Seg) (hg : g ∈ (s.slice a b).toList) :
    ∀ p ∈ g.ps.toList, ∃ q ∈ s.ps.toList, p.tgt = q.tgt := by
  have hone : ∀ (ps : Array Piece), (∀ p ∈ ps.toList, ∃ q ∈ s.ps.toList, p.tgt = q.tgt) → ∀ x y,
      g ∈ ((Desc.slicePieces ps x y).map Seg.one).toList → ∀ p ∈ g.ps.toList, ∃ q ∈ s.ps.toList, p.tgt = q.tgt := by
    intro ps hps x y h p hp
    rw [Array.toList_map, List.mem_map] at h
    obtain ⟨p', hp', rfl⟩ := h
    simp only [Desc.Seg.one, List.mem_singleton] at hp
    subst hp
    obtain ⟨q, hq, e⟩ := slicePieces_tgt hp'
    obtain ⟨q0, hq0, e0⟩ := hps q hq
    exact ⟨q0, hq0, e.trans e0⟩
  have hcopy : ∀ m, ∀ p ∈ (s.copy m).toList, ∃ q ∈ s.ps.toList, p.tgt = q.tgt := fun m p hp => shiftGates_tgt hp
  have hself : ∀ p ∈ s.ps.toList, ∃ q ∈ s.ps.toList, p.tgt = q.tgt := fun p hp => ⟨p, hp, rfl⟩
  unfold Desc.Seg.slice at hg
  split at hg
  · simp at hg
  split at hg
  · exact hone _ hself _ _ hg
  dsimp only at hg
  split at hg
  · exact hone _ (hcopy _) _ _ hg
  simp only [Array.toList_append, List.mem_append] at hg
  rcases hg with (hg | hg) | hg
  · split at hg
    · simp at hg
    · exact hone _ (hcopy _) _ _ hg
  · generalize (if (a % s.len == 0) = true then a / s.len else a / s.len + 1) = lo at hg
    split at hg
    · simp only [List.mem_singleton] at hg
      subst hg
      exact hcopy lo
    · simp at hg
  · exact hone _ (hcopy _) _ _ hg

/-- A piece of a segment of `PS.slice` has the target of a piece of one of `s`'s segments. -/
theorem psSlice_tgt (s : PS) (lo hi : Nat) (g : Seg) (hg : g ∈ (s.slice lo hi).toList) :
    ∀ p ∈ g.ps.toList, ∃ s0 ∈ s.segs.toList, ∃ q ∈ s0.ps.toList, p.tgt = q.tgt := by
  intro p hp
  obtain ⟨i, a, b, hi⟩ := psSlice_mem s lo hi g hg
  obtain ⟨q, hq, e⟩ := segSlice_tgt _ a b g hi p hp
  have hlt : i < s.segs.size := by
    refine (Nat.lt_or_ge i _).resolve_right fun hge => ?_
    rw [getElem!_neg _ i (by omega)] at hq
    cases hq
  rw [getElem!_pos _ i hlt] at hq
  exact ⟨_, Array.getElem_mem_toList hlt, q, hq, e⟩

/-- Definition `f`'s summarized return (`memo`'s entry), read in any activation of `f`: segments as `PS.of` makes them,
positions at or past 0 naming `f`'s gates or its parameters' leaves in range, a segment of more than one copy gates
only or copies all alike (`shift = 0`, as `down` gives a batch's members returning what they share), position by
position the gate `f`'s return leaf resolves to, and pieces each of positive count. -/
def RetOk (P : Prog) (memo : Array Desc.Summary) (f : Nat) : Prop :=
  PSOk (memo[f]!).ret ∧
  (∀ x ∈ psPos (memo[f]!).ret, 0 ≤ x.2 ∧ (∀ k, x.1 ≠ .ref false k) ∧
    ∀ i, x.1 = .ref true i → i < (P.defs[f]!).body.params.size ∧
      x.2 < (((P.defs[f]!).body.params[i]!).2.leaves : Int)) ∧
  (∀ s ∈ (memo[f]!).ret.segs.toList, s.reps > 1 → (∀ g ∈ s.ps.toList, ∃ w, g.tgt = .gate w) ∨ s.shift = 0) ∧
  (∀ sc : Flock.Program.Scope, sc.d = f →
    Agree P sc (psPos (memo[f]!).ret) ((List.range (P.defs[f]!).body.ret.count).map fun j =>
      Flock.Program.resolve P sc ((P.defs[f]!).body.ret.get j))) ∧
  ∀ s ∈ (memo[f]!).ret.segs.toList, ∀ g ∈ s.ps.toList, 0 < g.count

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
  obtain ⟨hps, hpos, hgo, hag, -⟩ := hR k node hn
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

/-- A batch of a composite's piece of one row: the node's leaf at position `t` is member `q.at t / L`'s return leaf
`q.at t % L`. -/
theorem batch_den_at (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (q : Piece) {k : Nat}
    (ht : q.tgt = .ref false k) {node : Flock.Program.Node} (hn : c.node k = .ok node) (hform : node.form = .batch)
    (hp : (c.P.defs[node.fn]!).prim = false) (h1 : q.rows = 1) (hq : ∀ x ∈ pos q, 0 ≤ x.2) {ys : List (Tgt × Int)}
    (hA : Agree c.P sc ys ((List.range q.len).map fun t =>
      Flock.Program.resolve c.P (c.P.child sc k ((q.at t).toNat / (c.P.defs[node.fn]!).outLeaves))
        ((c.P.defs[node.fn]!).body.ret.get ((q.at t).toNat % (c.P.defs[node.fn]!).outLeaves)))) :
    Agree c.P sc ys ((pos q).map (den c.P sc)) := by
  rw [pos_eq_map, List.map_map, count_one h1]
  refine agree_mono hA ?_
  intro t ht' a hx
  rw [List.mem_range] at ht'
  have hp' := ppt_one h1 (t := t) (by rw [count_one h1]; exact ht')
  have h0 : 0 ≤ q.at t := by simpa using hq _ (at_mem_pos h1 ht')
  simp only [Function.comp, ht] at hx ⊢
  rw [hp'] at hx
  simp only [den, if_neg (show ¬ q.at t < 0 by omega)] at hx
  have hnode : (c.P.defs[sc.d]!).body.nodes[k]! = node := by rw [hd]; exact node_get hn
  have e := den_batch c.P sc hnode hform hp hx
  rw [child_eq c sc hd hn]
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

/-- **A scan node's output**, as the decoder computes it (`buildNode`, `(carry, array n y)`): the final carry's
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
what the calls one step down return, followed through `Desc`'s code (`OkIs`, `reps_succ`), its scan case and `down`'s
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

-- one `repeat'` walks every case of `down`'s, `scanBranch`'s, `fromCallee`'s, `throughArg`'s and `carry`'s bodies,
-- about 8x the default budget
set_option maxHeartbeats 1600000 in
/-- **One step of fuel**: every segment `down`, `fromCallee`, `throughArg` and `carry` return one step up is of a copy
at least wherever those one step down are, followed through `Desc`'s code (`OkIs`), its calls' segments by the
hypotheses `hD` (`down`), `hC` (`fromCallee`), `hK` (`carry`), `hT` (`throughArg`) and `hS` (`scanBranch`). -/
theorem reps_succ (c : Desc.Ctx) (fuel : Nat) (hD : ∀ q, OkIs AllReps (Desc.down c fuel q))
    (hC : ∀ k m a b, OkIs AllReps (Desc.fromCallee c fuel k m a b))
    (hT : ∀ k m i g, OkIs AllReps (Desc.throughArg c fuel k m i g))
    (hK : ∀ k m a b, OkIs AllReps (Desc.carry c fuel k m a b)) :
    (∀ q, OkIs AllReps (Desc.down c (fuel + 1) q)) ∧
    (∀ k m a b, OkIs AllReps (Desc.fromCallee c (fuel + 1) k m a b)) ∧
    (∀ k m i g, OkIs AllReps (Desc.throughArg c (fuel + 1) k m i g)) ∧
    (∀ k m a b, OkIs AllReps (Desc.carry c (fuel + 1) k m a b)) := by
  refine (fun (hS : ∀ k node q, OkIs AllReps (scanBranch c fuel k node q)) =>
    ⟨fun q => ?down, fun k m a b => ?callee, fun k m i g => ?through, fun k m a b => ?carry⟩) (fun k node q => ?scan)
  case' down => rw [down_succ]; unfold downBody
  case' callee => rw [Desc.fromCallee.eq_2]
  case' through => rw [Desc.throughArg.eq_2]
  case' carry => rw [Desc.carry.eq_2]
  case' scan =>
    unfold scanBranch
    rcases hr : (c.P.defs[node.fn]!).ret with w | ⟨n, e⟩ | ⟨⟨_ | ⟨ct, _ | ⟨y, _ | ⟨z, l⟩⟩⟩⟩⟩ | fs
    all_goals (try exact okIs_throw)
    show OkIs AllReps (scanTuple c fuel k node ct y q)
    unfold scanTuple
  all_goals repeat' (first
    | with_reducible exact okIs_throw
    | with_reducible exact okIs_throw_bind
    | (with_reducible apply okIs_bind_of (hD _)) <;> intro _ _
    | (with_reducible apply okIs_bind_of (hC _ _ _ _)) <;> intro _ _
    | (with_reducible apply okIs_bind_of (hK _ _ _ _)) <;> intro _ _
    | (with_reducible apply okIs_bind_of (hT _ _ _ _)) <;> intro _ _
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
    reps_succ c fuel hD hC hT hK

/-- **`throughArg`'s segments are each of a copy at least.** -/
theorem throughReps (c : Desc.Ctx) : ThroughReps c :=
  fun fuel k m i g segs h x hx => ((reps_fuel c fuel).2.2.1 k m i g segs h x hx).1

/-- `PS.of` keeps its segments of positive count. -/
theorem allReps_of {segs : Array Seg} (h : AllReps segs) : AllReps (PS.of segs).segs := by
  rw [of_segs]
  intro x hx
  simp only [Array.toList_filter, List.mem_filter] at hx
  exact h x hx.1

/-- `coalesce` keeps its segments `RepsOk`: each it gives is one of them or of one copy. -/
theorem allReps_coalesce {segs : Array Seg} (h : AllReps segs) : AllReps (Desc.coalesce segs) :=
  coalesce_mem RepsOk (fun _ _ => repsOk_one rfl) segs h

/-- **A summary's return segments are each `RepsOk`**: `Desc.summarize` lays `down`'s segments of its return end to
end, coalesced. -/
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
      repeat' (first
        | with_reducible exact okIs_throw
        | with_reducible exact okIs_throw_bind
        | (with_reducible apply okIs_bind_of (hD _)) <;> intro _ _
        | with_reducible exact hD _
        | rw [Std.Legacy.Range.forIn_eq_forIn_range']
        | rw [← Array.forIn_toList]
        | with_reducible refine okIs_forIn_bind _ _ _ (fun _ _ _ _ => ?_) ?_ (fun _ _ => ?_)
        | with_reducible refine okIs_forIn_bind_arr _ _ _ (fun _ _ _ _ => ?_) ?_ (fun _ _ => ?_)
        | with_reducible refine okIs_forIn_arr _ _ _ (fun _ _ _ _ => ?_) ?_
        | with_reducible refine okIs_forIn_range _ _ _ (fun _ _ _ => ?_) ?_
        | with_reducible refine okIs_bind_of (J := AllReps) ?_ (fun _ _ => ?_)
        | (with_reducible apply okIs_forIn) <;> try intro _ _ _ _
        | with_reducible apply okIs_pure
        | with_reducible exact allReps_empty
        | with_reducible apply allReps_single
        | exact repsOk_one rfl
        | with_reducible apply allReps_append
        | with_reducible apply allReps_push
        | assumption
        | split
        | dsimp only [ForInStep.value])
    exact allReps_of (allReps_coalesce hret')

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
  obtain ⟨hok, hnn, -, hag, -⟩ := hR
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
  obtain ⟨hok, -, -, hag, -⟩ := hR k node hn
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


/-- A stride of whole members of a batch of a composite whose member's leaf `j` names gates only: copies of member
`m0`'s leaf, copy `u` moved by `u · stride / L` members. -/
theorem gateStride_agree (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (q : Piece) {k : Nat}
    (ht : q.tgt = .ref false k) {node : Flock.Program.Node} (hn : c.node k = .ok node) (hform : node.form = .batch)
    (hp : (c.P.defs[node.fn]!).prim = false) (hL : (c.P.defs[node.fn]!).outLeaves ≠ 0)
    (hR : RetOk c.P c.memo node.fn) (hOL : (c.P.defs[node.fn]!).outLeaves = (c.P.defs[node.fn]!).body.ret.count)
    (h1 : q.rows = 1) (hdv : q.stride % ((c.P.defs[node.fn]!).outLeaves : Int) = 0) (hb : q.base ≥ 0)
    (hq : ∀ x ∈ pos q, 0 ≤ x.2)
    (hrep : (((c.memo[node.fn]!).ret.slice (q.base.toNat % (c.P.defs[node.fn]!).outLeaves)
      (q.base.toNat % (c.P.defs[node.fn]!).outLeaves + 1)).all (·.reps ≤ 1)) = true)
    (hall : ((((c.memo[node.fn]!).ret.slice (q.base.toNat % (c.P.defs[node.fn]!).outLeaves)
      (q.base.toNat % (c.P.defs[node.fn]!).outLeaves + 1)).flatMap (·.ps)).all fun g => match g.tgt with
        | .gate _ => true
        | _ => false) = true) :
    Agree c.P sc (segsPos #[Desc.Seg.mk (Desc.shiftGates ((c.body.offsets[k]! : Int) +
        ((q.base.toNat / (c.P.defs[node.fn]!).outLeaves : Nat) : Int) * ((c.P.defs[node.fn]!).gates : Int))
        (((c.memo[node.fn]!).ret.slice (q.base.toNat % (c.P.defs[node.fn]!).outLeaves)
          (q.base.toNat % (c.P.defs[node.fn]!).outLeaves + 1)).flatMap (·.ps)))
      1 q.len (q.stride / ((c.P.defs[node.fn]!).outLeaves : Int) * ((c.P.defs[node.fn]!).gates : Int))])
      ((pos q).map (den c.P sc)) := by
  obtain ⟨hok, hnn, hgo, hag, -⟩ := hR
  rw [segsPos_singleton]
  apply batch_den_at c sc hd q ht hn hform hp h1 hq
  generalize hLd : (c.P.defs[node.fn]!).outLeaves = L at *
  generalize hGd : (c.P.defs[node.fn]!).gates = G at *
  have hleaf1 : ∀ x ∈ ((c.memo[node.fn]!).ret.slice (q.base.toNat % L) (q.base.toNat % L + 1)).toList,
      x.reps = 1 := fun x hx => by
    have h1' := (retOk_slice hgo hx).1
    have h2' := (Array.all_eq_true_iff_forall_mem.1 hrep) x (Array.mem_toList_iff.1 hx)
    simp only [decide_eq_true_eq] at h2'
    omega
  generalize hrd : (c.memo[node.fn]!).ret = r at *
  have hLp : 0 < L := Nat.pos_of_ne_zero hL
  have hjL : q.base.toNat % L < L := Nat.mod_lt _ hLp
  generalize hjd : q.base.toNat % L = j at *
  generalize hmd : q.base.toNat / L = m0 at *
  have hpsPos : posAll ((r.slice j (j + 1)).flatMap (·.ps)) = sl (psPos r) j (j + 1) := by
    rw [posAll_flatMap_ps _ hleaf1, psSlice_pos _ hok]
  have hgs := gates_of_all hall
  have hxs : ∀ x ∈ posAll ((r.slice j (j + 1)).flatMap (·.ps)), (∃ w, x.1 = .gate w) ∧ 0 ≤ x.2 := by
    intro x hx
    refine ⟨?_, ?_⟩
    · unfold posAll at hx
      obtain ⟨g, hg, hxg⟩ := List.mem_flatMap.1 hx
      obtain ⟨w, hw⟩ := hgs g hg
      exact ⟨w, (mem_pos_tgt hxg).trans hw⟩
    · rw [hpsPos] at hx
      exact (hnn x (List.mem_of_mem_drop (List.mem_of_mem_take hx))).1
  obtain ⟨d, hdd⟩ : ∃ d, q.stride = (L : Int) * d :=
    ⟨q.stride / L, (Int.mul_ediv_cancel' (Int.dvd_of_emod_eq_zero hdv)).symm⟩
  have hdiv : q.stride / (L : Int) = d := by
    rw [hdd]; exact Int.mul_ediv_cancel_left _ (by exact_mod_cast hL)
  rw [hdiv]
  unfold segPos
  rw [List.map_eq_flatMap]
  apply agree_flatMap
  intro u hu
  rw [List.mem_range] at hu
  simp only [copy_pos, shiftGates_pos, List.map_map, shiftPos_shiftPos]
  have hB : (q.base.toNat : Int) = q.base := Int.toNat_of_nonneg hb
  have hBdec : q.base.toNat = m0 * L + j := by rw [← hmd, ← hjd]; exact (Nat.div_add_mod' _ _).symm
  have hat : q.at u = ((m0 : Int) + u * d) * L + j := by
    unfold Desc.Piece.at; rw [hdd, ← hB, hBdec]; push_cast; ring
  have hat0 : 0 ≤ q.at u := by simpa using hq _ (at_mem_pos h1 hu)
  have he0 : 0 ≤ (m0 : Int) + u * d := by
    by_contra hneg
    push_neg at hneg
    have hle : ((m0 : Int) + u * d) * L ≤ -L := by nlinarith
    have : (j : Int) < L := by exact_mod_cast hjL
    omega
  obtain ⟨mu, hmu⟩ : ∃ mu : Nat, (m0 : Int) + u * d = mu := ⟨_, (Int.toNat_of_nonneg he0).symm⟩
  have hatN : (q.at u).toNat = mu * L + j := by
    rw [show q.at u = ((mu * L + j : Nat) : Int) by rw [hat, hmu]; push_cast; ring, Int.toNat_natCast]
  have hdivN : (q.at u).toNat / L = mu := by
    rw [hatN, Nat.add_comm, Nat.add_mul_div_right _ _ hLp, Nat.div_eq_of_lt hjL, Nat.zero_add]
  have hmodN : (q.at u).toNat % L = j := by
    rw [hatN, Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hjL]
  rw [hdivN, hmodN]
  have hsh : (c.body.offsets[k]! : Int) + (m0 : Int) * (G : Int) + (u : Int) * (d * (G : Int)) =
      ((c.body.offsets[k]! + mu * G : Nat) : Int) := by
    push_cast; rw [← hmu]; ring
  rw [hsh]
  have hsc : c.P.child sc k mu = .child sc node.fn (sc.off + c.body.offsets[k]! + mu * G) k mu := by
    rw [child_eq c sc hd hn, hd, hGd]; rfl
  have hoff : (c.P.child sc k mu).off = sc.off + ((c.body.offsets[k]! + mu * G : Nat) : Int).toNat := by
    rw [hsc, Int.toNat_natCast, Nat.add_assoc]; rfl
  refine agree_trans (agree_shift c.P (by positivity) hoff _ hxs) ?_
  have hA := agree_sl (hag (c.P.child sc k mu) (by rw [hsc]; rfl)) j (j + 1)
  rw [← hpsPos, sl_map_range' _ _ j (j + 1) (fun _ => by omega)] at hA
  simpa using hA

/-- A batch of a composite's piece of one row and a stride of whole members, as copies of one member's leaf naming
leaves of arguments the members share, one step up: what `fromCallee` keeps of member `m0`'s leaf `j`, once for each
position, every member's leaf `j` resolving as member `m0`'s (`batchShared`). -/
def BatchShared (c : Desc.Ctx) (sc : Flock.Program.Scope) (fuel : Nat) : Prop :=
  ∀ q one k node, q.tgt = .ref false k → c.node k = .ok node → node.form = .batch →
    (c.P.defs[node.fn]!).prim = false → (c.P.defs[node.fn]!).outLeaves ≠ 0 → q.rows = 1 → q.len > 1 →
    q.stride % ((c.P.defs[node.fn]!).outLeaves : Int) = 0 → q.base ≥ 0 →
    (∀ x ∈ pos q, 0 ≤ x.2) → (∀ x ∈ pos q, Fits c x) →
    (((c.memo[node.fn]!).ret.slice (q.base.toNat % (c.P.defs[node.fn]!).outLeaves)
      (q.base.toNat % (c.P.defs[node.fn]!).outLeaves + 1)).flatMap (·.ps)).all (fun g => match g.tgt with
        | .ref true i => (node.args[i]?.map (·.dims.isNone)).getD false
        | _ => false) = true →
    Desc.fromCallee c fuel k (q.base.toNat / (c.P.defs[node.fn]!).outLeaves)
      (q.base.toNat % (c.P.defs[node.fn]!).outLeaves) (q.base.toNat % (c.P.defs[node.fn]!).outLeaves + 1) =
      .ok one →
    one.all (·.reps ≤ 1) = true →
    Agree c.P sc (segPos (Desc.Seg.mk (one.flatMap (·.ps)) 1 q.len 0)) ((pos q).map (den c.P sc))

theorem memberSpans_of_none {a : Flock.Program.Arg} (h : a.dims = none) (m lo hi : Nat) :
    Desc.memberSpans a m lo hi = #[(lo, hi)] := by
  unfold Desc.memberSpans; rw [h]

theorem shared_arg {node : Flock.Program.Node} {i : Nat}
    (hi : (node.args[i]?.map (·.dims.isNone)).getD false = true) :
    ∃ a, node.args[i]? = some a ∧ a.dims = none := by
  rcases h : node.args[i]? with _ | a
  · rw [h] at hi; cases hi
  · rw [h] at hi
    simp only [Option.map_some, Option.getD_some, Option.isNone_iff_eq_none] at hi
    exact ⟨a, rfl, hi⟩

/-- `throughArg` reads an argument the members share the same for every member. -/
theorem throughArg_shared (c : Desc.Ctx) (fuel : Nat) {k i : Nat} {node : Flock.Program.Node}
    (hn : c.node k = .ok node) (hf : node.form = .batch)
    (hi : (node.args[i]?.map (·.dims.isNone)).getD false = true) (m m' : Nat) (g : Piece) :
    Desc.throughArg c fuel k m i g = Desc.throughArg c fuel k m' i g := by
  cases fuel with
  | zero => rfl
  | succ fuel =>
    obtain ⟨a, ha, hda⟩ := shared_arg hi
    rw [Desc.throughArg.eq_2, Desc.throughArg.eq_2, hn]
    simp only [bind, Except.bind, ha, hf, memberSpans_of_none hda]

theorem forIn_congr_mem {β σ : Type} (l : List β) {f g : β → σ → Except String (ForInStep σ)}
    (h : ∀ x ∈ l, f x = g x) (init : σ) : forIn l init f = forIn l init g := by
  induction l generalizing init with
  | nil => rfl
  | cons x l ih =>
    rw [List.forIn_cons, List.forIn_cons, h x List.mem_cons_self]
    congr 1
    funext r
    cases r with
    | done => rfl
    | yield b => exact ih (fun y hy => h y (List.mem_cons_of_mem _ hy)) b

theorem forIn_congr_arr {β σ : Type} (l : Array β) {f g : β → σ → Except String (ForInStep σ)}
    (h : ∀ x ∈ l.toList, f x = g x) (init : σ) : forIn l init f = forIn l init g := by
  rw [← Array.forIn_toList, ← Array.forIn_toList]
  exact forIn_congr_mem _ h init

theorem shiftGates_refs {δ : Int} {ps : Array Piece} (h : ∀ g ∈ ps.toList, ∃ i, g.tgt = .ref true i) :
    Desc.shiftGates δ ps = ps := by
  unfold Desc.shiftGates
  conv => rhs; rw [← Array.map_id ps]
  apply Array.map_congr_left
  intro g hg
  obtain ⟨i, hi⟩ := h g (Array.mem_toList_iff.2 hg)
  simp [hi]

/-- Member `m`'s return leaves `[a, b)`, where they name leaves of arguments the members share: what every member's
are. -/
theorem fromCallee_shared (c : Desc.Ctx) (fuel : Nat) {k : Nat} {node : Flock.Program.Node}
    (hn : c.node k = .ok node) (hf : node.form = .batch) {a b : Nat}
    (hsh : ∀ s ∈ ((c.memo[node.fn]!).ret.slice a b).toList, ∀ g ∈ s.ps.toList, ∃ i, g.tgt = .ref true i ∧
      (node.args[i]?.map (·.dims.isNone)).getD false = true) (m m' : Nat) :
    Desc.fromCallee c fuel k m a b = Desc.fromCallee c fuel k m' a b := by
  cases fuel with
  | zero => rfl
  | succ fuel =>
  rw [Desc.fromCallee.eq_2, Desc.fromCallee.eq_2, hn]
  simp only [bind, Except.bind]
  rw [← Array.forIn_toList, ← Array.forIn_toList]
  congr 1
  apply forIn_congr_mem
  intro s hs
  funext r
  have hsr : ∀ g ∈ s.ps.toList, ∃ i, g.tgt = .ref true i := fun g hg => (hsh s hs g hg).imp fun _ h => h.1
  have hin : ∀ {init : Array Seg} {F G : Piece → Array Seg → Except String (ForInStep (Array Seg))},
      (∀ g ∈ s.ps.toList, F g = G g) → forIn s.ps init F = forIn s.ps init G :=
    fun h => forIn_congr_arr _ h _
  simp only [shiftGates_refs hsr]
  by_cases hr : s.reps > 1
  · simp only [hr, ↓reduceIte]
    split
    · rfl
    by_cases hz : (s.shift != 0) = true
    · simp only [hz, ↓reduceIte]
      rfl
    · simp only [hz, ↓reduceIte, Bool.false_eq_true]
      congr 1
      apply hin
      intro g hg
      funext r'
      obtain ⟨i, hgt, hi⟩ := hsh s hs g hg
      simp only [hgt]
      rw [throughArg_shared c fuel hn hf hi m m' g]
  · simp only [hr, ↓reduceIte]
    congr 1
    apply hin
    intro g hg
    funext r'
    obtain ⟨i, hgt, hi⟩ := hsh s hs g hg
    simp only [hgt]
    rw [throughArg_shared c fuel hn hf hi m m' g]

/-- Pieces `down` tests for naming leaves of arguments the members share do. -/
theorem shared_of_all {node : Flock.Program.Node} {ps : Array Piece} (h : (ps.all fun g => match g.tgt with
      | .ref true i => (node.args[i]?.map (·.dims.isNone)).getD false
      | _ => false) = true) :
    ∀ g ∈ ps.toList, ∃ i, g.tgt = .ref true i ∧ (node.args[i]?.map (·.dims.isNone)).getD false = true := by
  rw [Array.all_eq_true_iff_forall_mem] at h
  intro g hg
  have := h g (Array.mem_toList_iff.1 hg)
  cases hgt : g.tgt with
  | gate w => simp only [hgt] at this; cases this
  | ref b i =>
    cases b with
    | false => simp only [hgt] at this; cases this
    | true => simp only [hgt] at this; exact ⟨i, rfl, this⟩

/-- **A batch of a composite's piece of one row and a stride of whole members, as copies of one member's leaf naming
leaves of arguments the members share**: each position's member's leaf `j` is what `fromCallee` keeps of member
`m0`'s, the same for every member (`fromCallee_shared`). -/
theorem batchShared (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (hB : Resolve.BodyOk c.P c.d)
    (hOL : ∀ k node, c.node k = .ok node → (c.P.defs[node.fn]!).prim = false →
      (c.P.defs[node.fn]!).outLeaves = (c.P.defs[node.fn]!).body.ret.count)
    (fuel : Nat) (hC : CalleeSpec c sc fuel) : BatchShared c sc fuel := by
  intro q one k node ht hn hform hp hL h1 hlen hdv hb hq hf hsh hone hone1
  apply batch_den_at c sc hd q ht hn hform hp h1 hq
  have hol := hOL k node hn hp
  have hdo := downOk_of hB hn
  have hnode := node_get hn
  have hrep1 : ∀ x ∈ one.toList, x.reps = 1 := by
    intro x hx
    have h0 := ((reps_fuel c fuel).2.1 k _ _ _ one hone x hx).1
    have hle := (Array.all_eq_true_iff_forall_mem.1 hone1) x (Array.mem_toList_iff.1 hx)
    simp only [decide_eq_true_eq] at hle
    omega
  rw [segPos_alike rfl, posAll_flatMap_ps one hrep1, List.map_eq_flatMap]
  apply agree_flatMap
  intro u hu
  rw [List.mem_range] at hu
  have hsh' := fun s (hs : s ∈ ((c.memo[node.fn]!).ret.slice (q.base.toNat % (c.P.defs[node.fn]!).outLeaves)
      (q.base.toNat % (c.P.defs[node.fn]!).outLeaves + 1)).toList) g (hg : g ∈ s.ps.toList) =>
    shared_of_all hsh g (by rw [Array.toList_flatMap]; exact List.mem_flatMap.2 ⟨s, hs, hg⟩)
  have hfu := hf _ (at_mem_pos h1 hu) k ht
  rw [hnode] at hfu
  generalize hLd : (c.P.defs[node.fn]!).outLeaves = L at *
  have hLp : 0 < L := Nat.pos_of_ne_zero hL
  have hjL : q.base.toNat % L < L := Nat.mod_lt _ hLp
  generalize hjd : q.base.toNat % L = j at *
  generalize hmd : q.base.toNat / L = m0 at *
  obtain ⟨d, hdd⟩ : ∃ d, q.stride = (L : Int) * d :=
    ⟨q.stride / L, (Int.mul_ediv_cancel' (Int.dvd_of_emod_eq_zero hdv)).symm⟩
  have hB' : (q.base.toNat : Int) = q.base := Int.toNat_of_nonneg hb
  have hBdec : q.base.toNat = m0 * L + j := by rw [← hmd, ← hjd]; exact (Nat.div_add_mod' _ _).symm
  have hat : q.at u = ((m0 : Int) + u * d) * L + j := by
    unfold Desc.Piece.at; rw [hdd, ← hB', hBdec]; push_cast; ring
  have hat0 : 0 ≤ q.at u := by simpa using hq _ (at_mem_pos h1 hu)
  have he0 : 0 ≤ (m0 : Int) + u * d := by
    by_contra hneg
    push Not at hneg
    have hle : ((m0 : Int) + u * d) * L ≤ -L := by nlinarith
    have : (j : Int) < L := by exact_mod_cast hjL
    omega
  obtain ⟨mu, hmu⟩ : ∃ mu : Nat, (m0 : Int) + u * d = mu := ⟨_, (Int.toNat_of_nonneg he0).symm⟩
  have hatN : (q.at u).toNat = mu * L + j := by
    rw [show q.at u = ((mu * L + j : Nat) : Int) by rw [hat, hmu]; push_cast; ring, Int.toNat_natCast]
  have hdivN : (q.at u).toNat / L = mu := by
    rw [hatN, Nat.add_comm, Nat.add_mul_div_right _ _ hLp, Nat.div_eq_of_lt hjL, Nat.zero_add]
  have hmodN : (q.at u).toNat % L = j := by
    rw [hatN, Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hjL]
  have hmem : mu < (c.body.nodes[k]!).members := by
    have := hdo (q.at u).toNat (by simp only at hfu; omega)
    simp only [hform] at this
    rw [hnode, members_batch hform, ← hdivN, ← hLd]
    exact this.1
  rw [hdivN, hmodN]
  have hA := hC k mu j (j + 1) one hmem (fun _ => by rw [hnode, ← hol]; omega)
    (by rw [← fromCallee_shared c fuel hn hform hsh' m0 mu]; exact hone)
  rw [hnode] at hA
  simpa using hA

/-- A batch of a composite's piece of one row and a stride of whole members: copies of a member's leaf of gates
(`gateStride_agree`), of leaves of shared arguments (`BatchShared`), or position by position (`batchPts_agree`). -/
theorem batchStride_agree (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (fuel : Nat)
    (hD : DownSpec c sc fuel) (hSh : BatchShared c sc fuel) (q : Piece) {k : Nat} (ht : q.tgt = .ref false k)
    {node : Flock.Program.Node} (hn : c.node k = .ok node) (hform : node.form = .batch)
    (hp : (c.P.defs[node.fn]!).prim = false) (hL : (c.P.defs[node.fn]!).outLeaves ≠ 0)
    (hR : RetOk c.P c.memo node.fn) (hOL : (c.P.defs[node.fn]!).outLeaves = (c.P.defs[node.fn]!).body.ret.count)
    (h1 : q.rows = 1) (hcp : q.len > 1 ∧ q.stride % ((c.P.defs[node.fn]!).outLeaves : Int) = 0 ∧ q.base ≥ 0)
    (hq : ∀ x ∈ pos q, 0 ≤ x.2) (hf : ∀ x ∈ pos q, Fits c x) {segs : Array Seg}
    (h : batchStride c fuel k node q = .ok segs) : Agree c.P sc (segsPos segs) ((pos q).map (den c.P sc)) := by
  unfold batchStride at h
  dsimp only at h
  split at h
  · rename_i hrep
    split at h
    · rename_i hall
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      exact gateStride_agree c sc hd q ht hn hform hp hL hR hOL h1 hcp.2.1 hcp.2.2 hq hrep hall
    · split at h
      · rename_i hsh
        obtain ⟨one, hone, h⟩ := Walk.except_bind_ok h
        split at h
        · rename_i hone1
          simp only [pure, Except.pure, Except.ok.injEq] at h
          subst h
          rw [segsPos_singleton]
          exact hSh q one k node ht hn hform hp hL h1 hcp.1 hcp.2.1 hcp.2.2 hq hf hsh hone hone1
        · exact batchPts_agree c sc fuel hD q h1 hq hf h
      · exact batchPts_agree c sc fuel hD q h1 hq hf h
  · exact batchPts_agree c sc fuel hD q h1 hq hf h

/-- A batch's output leaves `[a, b)` from what `mid` keeps of whole members `[m0, m1)` and `fromCallee` of the cut
members at either end. -/
theorem ends_agree (c : Desc.Ctx) (sc : Flock.Program.Scope) (fuel : Nat) (hC : CalleeSpec c sc fuel)
    {k L a b : Nat} (hL : L ≠ 0) (hab : (a + L - 1) / L ≤ b / L)
    (hm : ∀ j, a ≤ j → j < b → j / L < (c.body.nodes[k]!).members ∧
      j % L < (c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.count)
    {mid segs : Array Seg}
    (hmid : Agree c.P sc (segsPos mid) ((List.range' ((a + L - 1) / L * L) ((b / L - (a + L - 1) / L) * L)).map
      fun j => Flock.Program.resolve c.P (c.P.child sc k (j / L)) ((c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.get (j % L))))
    (h : batchEnds c fuel k L a b mid = .ok segs) :
    Agree c.P sc (segsPos segs) ((List.range' a (b - a)).map fun j =>
      Flock.Program.resolve c.P (c.P.child sc k (j / L)) ((c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.get (j % L))) := by
  unfold batchEnds at h
  dsimp only at h
  generalize hm0 : (a + L - 1) / L = m0 at *
  generalize hm1 : b / L = m1 at *
  have hLp : 0 < L := Nat.pos_of_ne_zero hL
  have ha0 : a ≤ m0 * L := by
    rw [← hm0]
    have := Nat.div_add_mod (a + L - 1) L
    have := Nat.mod_lt (a + L - 1) hLp
    have : L * ((a + L - 1) / L) = (a + L - 1) / L * L := Nat.mul_comm _ _
    omega
  have ha1 : m0 * L < a + L := by
    rw [← hm0]
    exact lt_of_le_of_lt (Nat.div_mul_le_self _ _) (by omega)
  have hb0 : m1 * L ≤ b := by rw [← hm1]; exact Nat.div_mul_le_self _ _
  have hb1 : b < m1 * L + L := by
    rw [← hm1]
    have := Nat.div_add_mod b L
    have := Nat.mod_lt b hLp
    have : L * (b / L) = b / L * L := Nat.mul_comm _ _
    omega
  have hmm : m0 * L ≤ m1 * L := Nat.mul_le_mul_right _ hab
  let V : Nat → Nat → Except String Nat := fun m e =>
    Flock.Program.resolve c.P (c.P.child sc k m) ((c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.get e)
  have hfirst : ∀ x, a < m0 * L → Desc.fromCallee c fuel k (a / L) (a % L) L = .ok x →
      Agree c.P sc (segsPos x) ((List.range' a (m0 * L - a)).map fun j => V (j / L) (j % L)) := by
    intro x hlt hx
    have hdm : a / L * L + a % L = a := by
      have := Nat.div_add_mod a L
      have : L * (a / L) = a / L * L := Nat.mul_comm _ _
      omega
    have hmod : a % L ≠ 0 := by
      intro h0
      have : a / L * L = a := by omega
      have : m0 * L = a := by
        rw [← hm0]
        rw [show a + L - 1 = (L - 1) + a / L * L by omega, Nat.add_mul_div_right _ _ hLp,
          Nat.div_eq_of_lt (by omega), Nat.zero_add]
        omega
      omega
    have hm0' : m0 * L - a = L - a % L := by
      have : m0 = a / L + 1 := by
        rw [← hm0, show a + L - 1 = (a % L - 1) + (a / L + 1) * L by
          rw [Nat.add_mul, Nat.one_mul]; omega, Nat.add_mul_div_right _ _ hLp,
          Nat.div_eq_of_lt (by have := Nat.mod_lt a hLp; omega), Nat.zero_add]
      rw [this, Nat.add_mul, Nat.one_mul]
      omega
    obtain ⟨hmA, hjA⟩ := hm a le_rfl (by omega)
    have hA := hC k (a / L) (a % L) L x hmA (fun _ => by
      have := hm (a / L * L + (L - 1)) (by omega) (by omega)
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt (by omega)] at this
      omega) hx
    rw [block_at V hL (a / L) (a % L) (L - a % L) (by omega), hdm] at hA
    rwa [hm0']
  have hlast : ∀ y, m1 * L < b → Desc.fromCallee c fuel k m1 0 (b - m1 * L) = .ok y →
      Agree c.P sc (segsPos y) ((List.range' (m1 * L) (b - m1 * L)).map fun j => V (j / L) (j % L)) := by
    intro y hlt hy
    obtain ⟨hmB, -⟩ := hm (m1 * L) (by omega) hlt
    rw [Nat.mul_div_cancel _ hLp] at hmB
    have hA := hC k m1 0 (b - m1 * L) y hmB (fun _ => by
      have := hm (b - 1) (by omega) (by omega)
      rw [show b - 1 = (b - 1 - m1 * L) + m1 * L by omega, Nat.add_mul_mod_self_right,
        Nat.mod_eq_of_lt (by omega)] at this
      omega) hy
    rw [Nat.sub_zero, block_at V hL m1 0 (b - m1 * L) (by omega), Nat.add_zero] at hA
    exact hA
  have hsplit : List.range' a (b - a) =
      List.range' a (m0 * L - a) ++ List.range' (m0 * L) ((m1 - m0) * L) ++ List.range' (m1 * L) (b - m1 * L) := by
    have e1 := List.range'_append_1 (s := a) (m := m0 * L - a) (n := (m1 - m0) * L)
    have e2 := List.range'_append_1 (s := a) (m := m0 * L - a + (m1 - m0) * L) (n := b - m1 * L)
    rw [show a + (m0 * L - a) = m0 * L by omega] at e1
    rw [show a + (m0 * L - a + (m1 - m0) * L) = m1 * L by rw [Nat.sub_mul]; omega] at e2
    rw [e1, e2, show m0 * L - a + (m1 - m0) * L + (b - m1 * L) = b - a by rw [Nat.sub_mul]; omega]
  obtain ⟨A, B, rfl, hA, hB⟩ : ∃ A B, segs = A ++ mid ++ B ∧
      Agree c.P sc (segsPos A) ((List.range' a (m0 * L - a)).map fun j => V (j / L) (j % L)) ∧
      Agree c.P sc (segsPos B) ((List.range' (m1 * L) (b - m1 * L)).map fun j => V (j / L) (j % L)) := by
    split at h
    · rename_i hlt
      obtain ⟨x, hx, h⟩ := Walk.except_bind_ok h
      split at h
      · rename_i hlt'
        obtain ⟨y, hy, h⟩ := Walk.except_bind_ok h
        simp only [pure, Except.pure, Except.ok.injEq] at h
        exact ⟨x, y, by rw [← h]; try simp, hfirst x hlt hx, hlast y hlt' hy⟩
      · rename_i hnlt
        simp only [pure, Except.pure, Except.ok.injEq] at h
        refine ⟨x, #[], by rw [← h]; try simp, hfirst x hlt hx, ?_⟩
        rw [show b - m1 * L = 0 by omega]
        exact List.Forall₂.nil
    · rename_i hnlt0
      split at h
      · rename_i hlt'
        obtain ⟨y, hy, h⟩ := Walk.except_bind_ok h
        simp only [pure, Except.pure, Except.ok.injEq] at h
        refine ⟨#[], y, by rw [← h]; try simp, ?_, hlast y hlt' hy⟩
        rw [show m0 * L - a = 0 by omega]
        try exact List.Forall₂.nil
      · rename_i hnlt
        simp only [pure, Except.pure, Except.ok.injEq] at h
        refine ⟨#[], #[], by rw [← h]; try simp, ?_, ?_⟩
        · rw [show m0 * L - a = 0 by omega]
          exact List.Forall₂.nil
        · rw [show b - m1 * L = 0 by omega]
          exact List.Forall₂.nil
  rw [segsPos_append, segsPos_append, hsplit, List.map_append, List.map_append]
  exact agree_append (agree_append hA hmid) hB

/-- A batch's output leaves `[a, b)` member by member, from what `fromCallee` keeps (`members_agree`). -/
theorem batchMembers_agree (c : Desc.Ctx) (sc : Flock.Program.Scope) (fuel : Nat) (hC : CalleeSpec c sc fuel)
    {k L a b : Nat} (hL : L ≠ 0) (hab : a < b) (hm : (b - 1) / L < (c.body.nodes[k]!).members)
    (hj : ∀ j, a ≤ j → j < b → j % L < (c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.count) {segs : Array Seg}
    (h : batchMembers c fuel k L a b = .ok segs) :
    Agree c.P sc (segsPos segs) ((List.range' a (b - a)).map fun j =>
      Flock.Program.resolve c.P (c.P.child sc k (j / L)) ((c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.get (j % L))) := by
  unfold batchMembers at h
  have h := walkAtMost_bind _ _ _ h
  dsimp only at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, Nat.add_sub_cancel, Nat.div_one] at h
  exact members_agree c sc fuel hC hL hab hm hj (ok_bind_pure h)

/-- Whole members of a batch of a composite's piece of one row and stride 1 as copies of one member's return naming
leaves of arguments the members share, one step up: what `fromCallee` keeps of member `m0`'s return, once for each
member of `[m0, m1)`, every member's return resolving as member `m0`'s (`batchSharedMid`). -/
def BatchSharedMid (c : Desc.Ctx) (sc : Flock.Program.Scope) (fuel : Nat) : Prop :=
  ∀ k node m0 m1 one,
    let L := (c.P.defs[node.fn]!).outLeaves
    c.node k = .ok node → node.form = .batch → (c.P.defs[node.fn]!).prim = false → L ≠ 0 →
    m0 + 1 < m1 → m1 ≤ (c.body.nodes[k]!).members →
    ((c.memo[node.fn]!).ret.segs.all fun s => s.reps ≤ 1 && s.ps.all fun g => match g.tgt with
      | .ref true i => (node.args[i]?.map (·.dims.isNone)).getD false
      | _ => false) = true →
    Desc.fromCallee c fuel k m0 0 L = .ok one → one.all (·.reps ≤ 1) = true →
    Agree c.P sc (segPos (Desc.Seg.mk (one.flatMap (·.ps)) L (m1 - m0) 0))
      ((List.range' (m0 * L) ((m1 - m0) * L)).map fun j =>
        Flock.Program.resolve c.P (c.P.child sc k (j / L)) ((c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.get (j % L)))

/-- **Whole members `[m0, m1)` of a batch of a composite's piece of one row and stride 1, as copies of one member's
return naming leaves of arguments the members share**: each member's return is what `fromCallee` keeps of member
`m0`'s, the same for every member (`fromCallee_shared`). -/
theorem batchSharedMid (c : Desc.Ctx) (sc : Flock.Program.Scope)
    (hOL : ∀ k node, c.node k = .ok node → (c.P.defs[node.fn]!).prim = false →
      (c.P.defs[node.fn]!).outLeaves = (c.P.defs[node.fn]!).body.ret.count)
    (fuel : Nat) (hC : CalleeSpec c sc fuel) : BatchSharedMid c sc fuel := by
  intro k node m0 m1 one
  dsimp only
  intro hn hform hp hL hm01 hm1 hsh hone hone1
  have hol := hOL k node hn hp
  have hnode := node_get hn
  have hrep1 : ∀ x ∈ one.toList, x.reps = 1 := by
    intro x hx
    have h0 := ((reps_fuel c fuel).2.1 k _ _ _ one hone x hx).1
    have hle := (Array.all_eq_true_iff_forall_mem.1 hone1) x (Array.mem_toList_iff.1 hx)
    simp only [decide_eq_true_eq] at hle
    omega
  have hshR : ∀ s ∈ (c.memo[node.fn]!).ret.segs.toList, ∀ g ∈ s.ps.toList, ∃ i, g.tgt = .ref true i ∧
      (node.args[i]?.map (·.dims.isNone)).getD false = true := by
    intro s hs
    have := (Array.all_eq_true_iff_forall_mem.1 hsh) s (Array.mem_toList_iff.1 hs)
    rw [Bool.and_eq_true] at this
    exact shared_of_all this.2
  have hsh' : ∀ s ∈ ((c.memo[node.fn]!).ret.slice 0 (c.P.defs[node.fn]!).outLeaves).toList, ∀ g ∈ s.ps.toList,
      ∃ i, g.tgt = .ref true i ∧ (node.args[i]?.map (·.dims.isNone)).getD false = true := by
    intro s hs g hg
    obtain ⟨s0, hs0, q, hq, e⟩ := psSlice_tgt _ _ _ s hs g hg
    obtain ⟨i, hi, h⟩ := hshR s0 hs0 q hq
    exact ⟨i, e.trans hi, h⟩
  rw [segPos_alike rfl, posAll_flatMap_ps one hrep1, range'_blocks, List.map_flatMap]
  apply agree_flatMap
  intro u hu
  have hu' : u < m1 - m0 := List.mem_range.1 hu
  have hb := block_at (fun m e => Flock.Program.resolve c.P (c.P.child sc k m)
    ((c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.get e)) hL (m0 + u) 0 (c.P.defs[node.fn]!).outLeaves (by omega)
  rw [show m0 * (c.P.defs[node.fn]!).outLeaves + u * (c.P.defs[node.fn]!).outLeaves =
    (m0 + u) * (c.P.defs[node.fn]!).outLeaves + 0 by ring, ← hb]
  have hA := hC k (m0 + u) 0 (c.P.defs[node.fn]!).outLeaves one (by omega)
    (fun _ => le_of_eq (by rw [hnode, ← hol]))
    (by rw [← fromCallee_shared c fuel hn hform hsh' m0 (m0 + u)]; exact hone)
  simpa using hA

/-- Whole members of a batch of a composite's piece of one row and stride 1, each returning one run of a mapped
argument's leaves within one row of its axis, one step up: what `down` keeps of the piece of rows `throughPS` gives, the
run in each member's row (`batchThroughMid`). -/
def BatchThroughMid (c : Desc.Ctx) (sc : Flock.Program.Scope) (fuel : Nat) : Prop :=
  ∀ k node m0 m1 i d Li bp segs,
    let L := (c.P.defs[node.fn]!).outLeaves
    let g := (c.memo[node.fn]!).ret.segs[0]!.ps[0]!
    let lo := g.base.toNat
    c.node k = .ok node → node.form = .batch → (c.P.defs[node.fn]!).prim = false → L ≠ 0 →
    m0 + 1 < m1 → m1 ≤ (c.body.nodes[k]!).members → (c.memo[node.fn]!).ret.segs.size = 1 →
    (c.memo[node.fn]!).ret.segs[0]!.reps ≤ 1 → (c.memo[node.fn]!).ret.segs[0]!.ps.size = 1 →
    g.tgt = .ref true i → (node.args[i]?.bind (·.dims)) = some (d, Li) →
    (Li > 0 && g.rows ≤ 1 && (g.stride == 1 || g.len ≤ 1) && g.len == L && lo / Li == (lo + L - 1) / Li) = true →
    Desc.throughPS (c.arg k i) (Desc.Piece.mk (.gate 0) (((lo / Li * d + m0) * Li + (lo - lo / Li * Li) : Nat) : Int)
      1 L (m1 - m0) (Li : Int)) = some bp →
    Desc.down c fuel bp = .ok segs →
    Agree c.P sc (segsPos segs) ((List.range' (m0 * L) ((m1 - m0) * L)).map fun j =>
      Flock.Program.resolve c.P (c.P.child sc k (j / L)) ((c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.get (j % L)))

/-- An array of one element is that element. -/
theorem arr_size_one {α : Type} [Inhabited α] {A : Array α} (h : A.size = 1) : A = #[A[0]!] := by
  apply Array.ext (by simp [h])
  intro j h1 h2
  have hj : j = 0 := by simp at h2; omega
  subst hj
  simp [getElem!_pos A 0 (by omega)]

/-- `throughPS` on an argument of one segment of one piece. -/
theorem throughPS_eq (pa : Piece) (len reps : Nat) (shift : Int) (st : Array Nat) (n : Nat) (q : Piece) :
    Desc.throughPS ⟨#[⟨#[pa], len, reps, shift⟩], st, n⟩ q =
      if reps > 1 || pa.rows > 1 || q.count == 0 then none else
      if (q.base + (if q.stride < 0 then ((q.len - 1 : Nat) : Int) * q.stride else 0) +
          (if q.rowStride < 0 then ((q.rows - 1 : Nat) : Int) * q.rowStride else 0)) < 0 ||
        (q.base + (if q.stride > 0 then ((q.len - 1 : Nat) : Int) * q.stride else 0) +
          (if q.rowStride > 0 then ((q.rows - 1 : Nat) : Int) * q.rowStride else 0)) ≥ (pa.len : Int) then none
      else some { tgt := pa.tgt, base := pa.base + q.base * pa.stride, stride := q.stride * pa.stride, len := q.len,
                  rows := q.rows, rowStride := q.rowStride * pa.stride } := rfl

/-- What `throughPS` gives: the argument's one piece of one row, read at the piece's positions. -/
theorem throughPS_some {ap : PS} {q bp : Piece} (h : Desc.throughPS ap q = some bp) :
    ∃ s pa, ap.segs = #[s] ∧ s.ps = #[pa] ∧ s.reps ≤ 1 ∧ pa.rows ≤ 1 ∧
      bp = { tgt := pa.tgt, base := pa.base + q.base * pa.stride, stride := q.stride * pa.stride, len := q.len,
             rows := q.rows, rowStride := q.rowStride * pa.stride } := by
  obtain ⟨segs, st, n⟩ := ap
  rcases segs with ⟨_ | ⟨s, _ | ⟨s', l⟩⟩⟩
  · unfold Desc.throughPS at h; dsimp only at h; cases h
  rotate_left
  · unfold Desc.throughPS at h; dsimp only at h; cases h
  obtain ⟨ps, len, reps, shift⟩ := s
  rcases ps with ⟨_ | ⟨pa, _ | ⟨pa', l'⟩⟩⟩
  · unfold Desc.throughPS at h; dsimp only at h; cases h
  rotate_left
  · unfold Desc.throughPS at h; dsimp only at h; cases h
  rw [throughPS_eq] at h
  split at h
  · cases h
  rename_i hc
  rw [Option.ite_none_left_eq_some] at h
  simp only [Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq, not_or, not_lt] at hc
  exact ⟨_, pa, rfl, rfl, hc.1.1, hc.1.2, (Option.some.inj h.2).symm⟩

/-- Agreement at one position of lists indexed alike. -/
theorem agree_range_get {P : Prog} {sc : Flock.Program.Scope} {n : Nat} {F : Nat → Tgt × Int}
    {G : Nat → Except String Nat} (h : Agree P sc ((List.range n).map F) ((List.range n).map G)) {t : Nat}
    (ht : t < n) {a : Nat} (hG : G t = .ok a) : den P sc (F t) = .ok a := by
  unfold Agree at h
  rw [List.forall₂_map_left_iff, List.forall₂_map_right_iff, List.forall₂_same] at h
  exact h t (List.mem_range.2 ht) a hG

/-- Lists indexed alike are of one length and equal position by position. -/
theorem map_range_eq {α : Type} {n1 n2 : Nat} {F G : Nat → α} (h : (List.range n1).map F = (List.range n2).map G) :
    n1 = n2 ∧ ∀ t < n1, F t = G t := by
  have hl : n1 = n2 := by simpa using congrArg List.length h
  subst hl
  refine ⟨rfl, fun t ht => ?_⟩
  have := congrArg (·[t]?) h
  simpa [List.getElem?_map, List.getElem?_range ht] using this

/-- `agree_mono` over blocks of `L`. -/
theorem agree_mono2 {P : Prog} {sc : Flock.Program.Scope} {ys : List (Tgt × Int)} {n L : Nat}
    {f g : Nat → Nat → Except String Nat}
    (h : Agree P sc ys ((List.range n).flatMap fun u => (List.range L).map (f u)))
    (hfg : ∀ u < n, ∀ t < L, ∀ a, g u t = .ok a → f u t = .ok a) :
    Agree P sc ys ((List.range n).flatMap fun u => (List.range L).map (g u)) := by
  have e : ∀ F : Nat → Nat → Except String Nat, ((List.range n).flatMap fun u => (List.range L).map (F u)) =
      ((List.range n).flatMap fun u => (List.range L).map fun t => (u, t)).map fun p => F p.1 p.2 := by
    intro F; rw [List.map_flatMap]; simp only [List.map_map]; rfl
  rw [e] at h ⊢
  refine agree_mono h ?_
  intro x hx a ha
  obtain ⟨u, hu, hx⟩ := List.mem_flatMap.1 hx
  obtain ⟨t, ht, rfl⟩ := List.mem_map.1 hx
  exact hfg u (List.mem_range.1 hu) t (List.mem_range.1 ht) a ha

/-- **Whole members `[m0, m1)` of a batch of a composite's piece of one row and stride 1, each returning one run of a
mapped argument's leaves within one row of its axis**: member `m`'s return leaf `t` is its parameter's leaf `lo + t`,
the argument's reference `memberIdx` names (row `lo / Li`, member `m`), so the rows `throughPS` gives name, position by
position, the references each member's return resolves through, and `down` keeps their gates. -/
theorem batchThroughMid (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (hB : Resolve.BodyOk c.P c.d)
    (hA : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size → ArgOk c k i)
    (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn)
    (hOL : ∀ k node, c.node k = .ok node → (c.P.defs[node.fn]!).prim = false →
      (c.P.defs[node.fn]!).outLeaves = (c.P.defs[node.fn]!).body.ret.count)
    (fuel : Nat) (hD : DownSpec c sc fuel) : BatchThroughMid c sc fuel := by
  intro k node m0 m1 i d Li bp segs
  dsimp only
  intro hn hform hp hL hm01 hm1 hs1 hr1 hp1 hti hdl hc hbp hx
  simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, Bool.or_eq_true] at hc
  obtain ⟨⟨⟨⟨hLi, hgr⟩, hgs⟩, hgl⟩, hq⟩ := hc
  obtain ⟨a, ha, hdims⟩ := Option.bind_eq_some_iff.1 hdl
  obtain ⟨hia, hai⟩ := Array.getElem?_eq_some_iff.1 ha
  have hai' : node.args[i]! = a := by rw [getElem!_pos node.args i hia]; exact hai
  have hnode : c.body.nodes[k]! = node := node_get hn
  have hk : k < c.body.nodes.size := node_lt hn
  have hnode' : (c.P.defs[sc.d]!).body.nodes[k]! = node := by rw [hd]; exact hnode
  have hUp := upOk_of hB hn
  have hRk : ∀ t < ((c.body.nodes[k]!).args[i]!).seq.count,
      Resolve.RefOk c.P c.d k (((c.body.nodes[k]!).args[i]!).seq.get t) := (hB.1 k hk).2.1 i
  rw [hnode, hai'] at hRk
  obtain ⟨hokA, hposA⟩ := hA k i hk (by rw [hnode]; exact hia)
  rw [hnode, hai'] at hposA
  rw [hnode] at hm1
  obtain ⟨s, pa, hsegs, hps, hsr, hpr, hbpe⟩ := throughPS_some hbp
  have hpa : pos pa = (List.range a.seq.count).map fun t => refPos (a.seq.get t) := by
    rw [← hposA, psPos_flat _ hokA (by rw [hsegs]; simpa using hsr)]
    simp [hsegs, hps, posAll]
  rw [pos_eq_map] at hpa
  obtain ⟨hcnt, hpt⟩ := map_range_eq hpa
  have hpaJ : ∀ J < a.seq.count, (pa.tgt, pa.base + (J : Int) * pa.stride) = refPos (a.seq.get J) := by
    intro J hJ
    unfold Desc.Piece.count at hcnt
    have hr1 : pa.rows = 1 := by
      rcases Nat.lt_or_ge pa.rows 1 with h | h
      · rw [show pa.rows = 0 by omega, Nat.zero_mul] at hcnt; omega
      · omega
    have hJl : J < pa.len := by rw [hr1, Nat.one_mul] at hcnt; omega
    rw [← hpt J (by unfold Desc.Piece.count; omega)]
    simp only [ppt, Prod.mk.injEq, true_and, Nat.div_eq_of_lt hJl, Nat.mod_eq_of_lt hJl]
    simp
  obtain ⟨hokR, hnn, -, hag, -⟩ := hR k node hn
  have hol := hOL k node hn hp
  set L := (c.P.defs[node.fn]!).outLeaves with hLdef
  set R := (c.memo[node.fn]!).ret with hRdef
  set g := R.segs[0]!.ps[0]! with hgdef
  set lo := g.base.toNat with hlodef
  have hR1 : R.segs = #[R.segs[0]!] := arr_size_one hs1
  have hP1 : R.segs[0]!.ps = #[g] := arr_size_one hp1
  have hposR : psPos R = pos g := by
    rw [psPos_flat _ hokR (by rw [hR1]; simpa using hr1), hR1]
    simp [hP1, posAll]
  have hgc : g.count = L := by
    have := (hag (c.P.child sc k m0) (by rw [child_eq c sc hd hn]; rfl)).length_eq
    rw [hposR, pos_length] at this
    simpa [hol] using this
  have hppt : ∀ t < L, ppt g t = g.base + (t : Int) := by
    intro t ht
    unfold ppt
    rw [hgl, Nat.div_eq_of_lt ht, Nat.mod_eq_of_lt ht]
    rcases hgs with h | h
    · rw [h]; simp
    · have : t = 0 := by omega
      subst this; simp
  have hmemg : ∀ t < L, (g.tgt, ppt g t) ∈ psPos R := by
    intro t ht; rw [hposR, pos_eq_map, hgc]; exact List.mem_map.2 ⟨t, List.mem_range.2 ht, rfl⟩
  have hL0 : 0 < L := Nat.pos_of_ne_zero hL
  have hb0 : 0 ≤ g.base := by
    have := (hnn _ (hmemg 0 hL0)).1
    rw [hppt 0 hL0] at this; simpa using this
  have hglo : g.base = (lo : Int) := (Int.toNat_of_nonneg hb0).symm
  have hpar : ∀ t < L, i < (c.P.defs[node.fn]!).body.params.size ∧
      lo + t < ((c.P.defs[node.fn]!).body.params[i]!).2.leaves := by
    intro t ht
    have := (hnn _ (hmemg t ht)).2.2 i hti
    rw [hppt t ht, hglo] at this
    have h2 := this.2
    simp only at h2
    exact ⟨this.1, by exact_mod_cast h2⟩
  have hdiv : ∀ t < L, (lo + t) / Li = lo / Li := by
    intro t ht
    apply le_antisymm
    · rw [hq]; exact Nat.div_le_div_right (by omega)
    · exact Nat.div_le_div_right (by omega)
  have hJ : ∀ u t, t < L → Resolve.memberIdx a (m0 + u) (lo + t) =
      (lo / Li * d + m0) * Li + (lo - lo / Li * Li) + u * Li + t := by
    intro u t ht
    unfold Resolve.memberIdx; rw [hdims]; dsimp only
    have h1 := Nat.div_add_mod (lo + t) Li
    have h2 := Nat.div_mul_le_self lo Li
    rw [hdiv t ht] at h1 ⊢
    rw [Nat.mul_comm Li] at h1
    rw [Nat.add_mul, Nat.add_mul, Nat.add_mul]
    omega
  have hJc : ∀ u < m1 - m0, ∀ t < L, Resolve.memberIdx a (m0 + u) (lo + t) < a.seq.count := by
    intro u hu t ht
    have := hUp i (hpar t ht).1 (lo + t) (hpar t ht).2 (m0 + u) (by omega)
    simp only [hform, hai'] at this
    exact this
  have hxp : ∀ u < m1 - m0, ∀ t < L, pa.base + (((lo / Li * d + m0) * Li + (lo - lo / Li * Li) : Nat) : Int) *
      pa.stride + (u : Int) * ((Li : Int) * pa.stride) + (t : Int) * (1 * pa.stride) =
      pa.base + ((Resolve.memberIdx a (m0 + u) (lo + t) : Nat) : Int) * pa.stride := by
    intro u _ t ht
    rw [hJ u t ht]
    push_cast
    ring
  have hxs : ∀ x ∈ pos bp, ∃ J < a.seq.count, x = refPos (a.seq.get J) := by
    intro x hx'
    rw [hbpe] at hx'
    simp only [pos] at hx'
    obtain ⟨u, hu, hx'⟩ := List.mem_flatMap.1 hx'
    obtain ⟨t, ht, rfl⟩ := List.mem_map.1 hx'
    rw [List.mem_range] at hu ht
    refine ⟨_, hJc u hu t ht, ?_⟩
    rw [hxp u hu t ht]
    exact hpaJ _ (hJc u hu t ht)
  have hDS := hD _ segs (fun x hx' => by obtain ⟨J, -, rfl⟩ := hxs x hx'; simp [refPos])
    (fun x hx' k' hk' => by
      obtain ⟨J, hJ', rfl⟩ := hxs x hx'
      simp only [refPos, Tgt.ref.injEq] at hk'
      obtain ⟨hp', rfl⟩ := hk'
      have := ((hRk J hJ').2 hp').2
      simp only [refPos]
      exact_mod_cast this) hx
  have e2 : ((List.range' (m0 * L) ((m1 - m0) * L)).map fun j =>
      Flock.Program.resolve c.P (c.P.child sc k (j / L)) ((c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.get (j % L))) =
      (List.range (m1 - m0)).flatMap fun u => (List.range L).map fun t =>
        Flock.Program.resolve c.P (c.P.child sc k (m0 + u)) ((c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.get t) := by
    rw [range'_blocks, List.map_flatMap]
    apply flatMap_congr
    intro u hu
    have hb := block_at (fun m e => Flock.Program.resolve c.P (c.P.child sc k m)
      ((c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.get e)) hL (m0 + u) 0 L (by omega)
    rw [show m0 * L + u * L = (m0 + u) * L + 0 by ring, ← hb, List.range_eq_range']
  rw [e2]
  rw [hbpe] at hDS
  simp only [pos, List.map_flatMap, List.map_map] at hDS
  refine agree_mono2 hDS ?_
  intro u hu t ht a' ha'
  simp only [Function.comp]
  rw [hxp u hu t ht, hpaJ _ (hJc u hu t ht), den_refPos]
  have hsc' := child_eq c sc hd hn (m0 + u)
  have hag' := hag (c.P.child sc k (m0 + u)) (by rw [hsc']; rfl)
  rw [hposR, pos_eq_map, hgc, ← hol] at hag'
  rw [callee_eq c hn] at ha'
  have h1 := agree_range_get hag' ht ha'
  rw [hti, hppt t ht, hglo] at h1
  simp only [den] at h1
  rw [if_neg (by omega), show ((lo : Int) + (t : Int)).toNat = lo + t by omega, hsc'] at h1
  have h2 := den_up_batch c.P sc hnode' hform h1
  rwa [hai', Resolve.memberRef_eq] at h2

/-- A batch's output leaves `[a, b)` through `throughPS` (`BatchThroughMid`), else member by member. -/
theorem batchThrough_agree (c : Desc.Ctx) (sc : Flock.Program.Scope) (fuel : Nat) (hC : CalleeSpec c sc fuel)
    (hTM : BatchThroughMid c sc fuel) {k : Nat} {node : Flock.Program.Node} (hn : c.node k = .ok node)
    (hform : node.form = .batch) (hp : (c.P.defs[node.fn]!).prim = false) (hL : (c.P.defs[node.fn]!).outLeaves ≠ 0)
    {a b : Nat} (hab : a < b)
    (hm : ∀ j, a ≤ j → j < b → j / (c.P.defs[node.fn]!).outLeaves < (c.body.nodes[k]!).members ∧
      j % (c.P.defs[node.fn]!).outLeaves < (c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.count)
    (hm1 : b / (c.P.defs[node.fn]!).outLeaves ≤ (c.body.nodes[k]!).members)
    {segs : Array Seg} (h : batchThrough c fuel k node a b = .ok segs) :
    Agree c.P sc (segsPos segs) ((List.range' a (b - a)).map fun j =>
      Flock.Program.resolve c.P (c.P.child sc k (j / (c.P.defs[node.fn]!).outLeaves))
        ((c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.get (j % (c.P.defs[node.fn]!).outLeaves))) := by
  have hmem := fun (hh : batchMembers c fuel k (c.P.defs[node.fn]!).outLeaves a b = .ok segs) =>
    batchMembers_agree c sc fuel hC hL hab (hm (b - 1) (by omega) (by omega)).1 (fun j h1 h2 => (hm j h1 h2).2) hh
  unfold batchThrough at h
  dsimp only at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hc
    split at h
    · rename_i i hti
      split at h
      · rename_i d Li hdl
        split at h
        · rename_i hg
          split at h
          · rename_i bp hbp
            obtain ⟨x, hx, h⟩ := Walk.except_bind_ok h
            exact ends_agree c sc fuel hC hL (by omega) hm
              (hTM k node _ _ i d Li bp x hn hform hp hL hc.1.1.1 hm1 hc.1.1.2 hc.1.2 hc.2 hti hdl hg hbp hx) h
          · exact hmem h
        · exact hmem h
      · exact hmem h
    · exact hmem h
  · exact hmem h

/-- **A batch of a composite's piece of one row and stride 1 past its one-run case**: whole members as copies of
one member's return of gates (`copies_seg_agree`), of leaves of shared arguments (`BatchSharedMid`), through
`throughPS` (`BatchThroughMid`), or member by member, the cut members at either end each through its return
(`ends_agree`). -/
theorem batchOne_agree (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (hB : Resolve.BodyOk c.P c.d)
    (fuel : Nat) (hC : CalleeSpec c sc fuel) (hSM : BatchSharedMid c sc fuel) (hTM : BatchThroughMid c sc fuel)
    (q : Piece) {k : Nat} (ht : q.tgt = .ref false k)
    {node : Flock.Program.Node} (hn : c.node k = .ok node) (hform : node.form = .batch)
    (hp : (c.P.defs[node.fn]!).prim = false) (hL : (c.P.defs[node.fn]!).outLeaves ≠ 0)
    (hR : RetOk c.P c.memo node.fn) (hOL : (c.P.defs[node.fn]!).outLeaves = (c.P.defs[node.fn]!).body.ret.count)
    (h0 : q.count ≠ 0) (hr : ¬ q.rows > 1) (hs : q.stride = 1)
    (hq : ∀ x ∈ pos q, 0 ≤ x.2) (hf : ∀ x ∈ pos q, Fits c x) {segs : Array Seg}
    (h : batchOne c fuel k node q = .ok segs) : Agree c.P sc (segsPos segs) ((pos q).map (den c.P sc)) := by
  apply batch_den c sc hd q ht hn hform hp h0 hr hs hq
  have h1 := rows_one h0 hr
  have hlen : 0 < q.len := by rw [← count_one h1]; omega
  have hdo := downOk_of hB hn
  have hnode := node_get hn
  have hbl : q.base.toNat + q.len ≤ node.out.leaves := by
    have hm := at_mem_pos h1 (show q.len - 1 < q.len by omega)
    have hb := hq _ (at_mem_pos h1 hlen)
    have := hf _ hm k ht
    simp only [hnode, Desc.Piece.at, hs] at this hb
    omega
  have hm : ∀ j, q.base.toNat ≤ j → j < q.base.toNat + q.len →
      j / (c.P.defs[node.fn]!).outLeaves < (c.body.nodes[k]!).members ∧
        j % (c.P.defs[node.fn]!).outLeaves < (c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.count := by
    intro j _ hj
    have := hdo j (by omega)
    simp only [hform] at this
    rw [hnode, members_batch hform]
    exact ⟨this.1, this.2.2 hp⟩
  have hlen' : q.base.toNat < q.base.toNat + q.len := by omega
  have hm1 : (q.base.toNat + q.len) / (c.P.defs[node.fn]!).outLeaves ≤ (c.body.nodes[k]!).members := by
    have hLp := Nat.pos_of_ne_zero hL
    have h1' := (hm (q.base.toNat + q.len - 1) (by omega) (by omega)).1
    have e1 := Nat.div_mul_le_self (q.base.toNat + q.len) (c.P.defs[node.fn]!).outLeaves
    have e5 := Nat.lt_div_mul_add (a := q.base.toNat + q.len - 1) hLp
    have e6 : (q.base.toNat + q.len) / (c.P.defs[node.fn]!).outLeaves * (c.P.defs[node.fn]!).outLeaves ≤
        ((q.base.toNat + q.len - 1) / (c.P.defs[node.fn]!).outLeaves + 1) * (c.P.defs[node.fn]!).outLeaves := by
      rw [Nat.add_mul, Nat.one_mul]; omega
    have := Nat.le_of_mul_le_mul_right e6 hLp
    omega
  rw [show q.len = q.base.toNat + q.len - q.base.toNat by omega]
  unfold batchOne at h
  dsimp only at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
    have hgo : ∀ x ∈ (c.memo[node.fn]!).ret.segs.toList, x.reps ≤ 1 ∧ ∀ g ∈ x.ps.toList, ∃ w, g.tgt = .gate w := by
      intro x hx
      have := (Array.all_eq_true_iff_forall_mem.1 hc.1) x (Array.mem_toList_iff.1 hx)
      simp only [Bool.and_eq_true, decide_eq_true_eq] at this
      exact ⟨this.1, gates_of_all this.2⟩
    refine ends_agree c sc fuel hC hL (by omega) hm ?_ h
    rw [segsPos_singleton]
    exact copies_seg_agree c sc hd hn hR hL hOL hgo _ _
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
    obtain ⟨one, hone, h⟩ := Walk.except_bind_ok h
    split at h
    · rename_i hone1
      refine ends_agree c sc fuel hC hL (by omega) hm ?_ h
      rw [segsPos_singleton]
      exact hSM k node _ _ one hn hform hp hL hc.2 hm1 hc.1 hone hone1
    · exact batchThrough_agree c sc fuel hC hTM hn hform hp hL hlen' hm hm1 h
  · exact batchThrough_agree c sc fuel hC hTM hn hform hp hL hlen' hm hm1 h

/-- **A batch of a composite's piece**, in its one-run case, of more than one row (row by row), or of one row: of
stride 1 (`batchOne_agree`), of a stride of whole members (`batchStride_agree`), or of any other stride (position by
position). -/
theorem batchSpec_of (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (hB : Resolve.BodyOk c.P c.d)
    (fuel : Nat) (hD : DownSpec c sc fuel) (hC : CalleeSpec c sc fuel)
    (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn)
    (hOL : ∀ k node, c.node k = .ok node → (c.P.defs[node.fn]!).prim = false →
      (c.P.defs[node.fn]!).outLeaves = (c.P.defs[node.fn]!).body.ret.count)
    (hAff : BatchAffine c sc) (hTM : BatchThroughMid c sc fuel) : BatchSpec c sc fuel := by
  intro q segs k node ht hn hform hp h0 hff hq hf h
  by_cases hL : (c.P.defs[node.fn]!).outLeaves = 0
  · exact absurd h (down_batch0 c fuel q ht hff hn hform hp hL h0 segs)
  rcases ha : affineOf c k node q with _ | p
  · by_cases hr : q.rows > 1
    · rw [down_batch_rows c fuel q ht hff hn hform hp hL h0 ha hr] at h
      exact byRows_agree c sc fuel hD q hq hf h
    by_cases hs : q.stride = 1
    · rw [down_batch_one c fuel q ht hff hn hform hp hL h0 ha hr hs] at h
      exact batchOne_agree c sc hd hB fuel hC (batchSharedMid c sc hOL fuel hC) hTM q ht hn hform hp hL (hR k node hn)
        (hOL k node hn hp) h0 hr hs
        hq hf h
    by_cases hcp : q.len > 1 ∧ q.stride % ((c.P.defs[node.fn]!).outLeaves : Int) = 0 ∧ q.base ≥ 0
    · rw [down_batch_stride c fuel q ht hff hn hform hp hL h0 ha hr hs hcp] at h
      exact batchStride_agree c sc hd fuel hD (batchShared c sc hd hB hOL fuel hC) q ht hn hform hp hL (hR k node hn)
        (hOL k node hn hp) (rows_one h0 hr) hcp hq hf h
    rw [down_batch_pts c fuel q ht hff hn hform hp hL h0 ha hr hs hcp] at h
    exact batchPts_agree c sc fuel hD q (rows_one h0 hr) hq hf h
  · rw [down_batch_affine c fuel q ht hff hn hform hp hL h0 ha] at h
    simp only [Except.ok.injEq] at h
    subst h
    exact hAff q k node p ht hn hform hp hL h0 hq hf ha

/-! ## `carry`'s step

`carry` follows a scan's return leaves `[a, b)` of iteration `m` back through the step's summary (`Desc.carryRuns`,
`Desc.chainOf`): a run the step returns in place is the initial carry's leaves through `throughArg` (`init_agree`,
`carry_init`), and any other return leaf follows carry leaf by carry leaf (`chainOf_ok`) to the leaf a callee returns
(`fromCallee`, `carry_chain`) or around a cycle back to the initial carry (`chainAt_cyc`). -/

/-- A loop whose every step continues with `g`: the fold of `g`. -/
theorem forIn_id_yield {α β : Type} (l : List α) (b : β) (g : α → β → β) (f : α → β → Id (ForInStep β))
    (hf : ∀ a b, f a b = pure (.yield (g a b))) : forIn l b f = pure (l.foldl (fun b a => g a b) b) := by
  induction l generalizing b with
  | nil => rfl
  | cons a l ih => rw [List.forIn_cons, hf, List.foldl_cons]; exact ih (g a b)

/-- A piece `Piece.slice` gives of positions within `q` has some. -/
theorem pieceSlice_count (q : Piece) (x y : Nat) (hy : y ≤ q.count) : ∀ p ∈ (q.slice x y).toList, 0 < p.count := by
  intro p hp
  unfold Desc.Piece.slice at hp
  split at hp
  · simp at hp
  rename_i hxy
  split at hp
  · simp only [List.mem_singleton] at hp
    subst hp
    simp only [Desc.Piece.count, Desc.Piece.row]; omega
  rename_i hr
  have hl : 0 < q.len := by
    rcases Nat.eq_zero_or_pos q.len with h | h
    · simp [Desc.Piece.count, h] at hy; omega
    · exact h
  have hpart : ∀ r s e, ∀ p ∈ (if s < e then
      #[{ q.row r with base := (q.row r).at s, len := e - s }] else (#[] : Array Piece)).toList, 0 < p.count := by
    intro r s e p hp
    split at hp
    · simp only [List.mem_singleton] at hp
      subst hp
      simp only [Desc.Piece.count, Desc.Piece.row]; omega
    · simp at hp
  dsimp only at hp
  split at hp
  · exact hpart _ _ _ p hp
  simp only [Array.toList_append, List.mem_append] at hp
  rcases hp with (hp | hp) | hp
  · by_cases ht : (x % q.len == 0) = true
    · simp [ht] at hp
    · rw [ite_eq_right ht] at hp
      exact hpart _ _ _ p hp
  · generalize (if (x % q.len == 0) = true then x / q.len else x / q.len + 1) = lo at hp
    by_cases hlo : lo < y / q.len
    · rw [ite_eq_left hlo] at hp
      simp only [List.mem_singleton] at hp
      subst hp
      simp only [Desc.Piece.count]
      exact Nat.mul_pos (by omega) hl
    · rw [ite_eq_right hlo] at hp
      simp at hp
  · exact hpart _ _ _ p hp

theorem slGo_count (a b : Nat) : ∀ (l : List Piece) (st : Nat) (out : Array Piece),
    (∀ p ∈ out.toList, 0 < p.count) → ∀ p ∈ (slGo a b l st out).toList, 0 < p.count
  | [], _, out, h => h
  | q :: qs, st, out, h => by
    unfold slGo
    split
    · exact h
    · refine slGo_count a b qs _ _ ?_
      split
      · intro p hp
        rw [Array.toList_append, List.mem_append] at hp
        rcases hp with hp | hp
        · exact h p hp
        · exact pieceSlice_count q _ _ (by omega) p hp
      · exact h

/-- Every piece of `slicePieces` has some positions. -/
theorem slicePieces_count (ps : Array Piece) (a b : Nat) : ∀ p ∈ (Desc.slicePieces ps a b).toList, 0 < p.count := by
  rw [slicePieces_go]
  exact slGo_count a b _ 0 #[] (fun p hp => by simp at hp)

theorem shiftGates_count (δ : Int) (ps : Array Piece) (h : ∀ q ∈ ps.toList, 0 < q.count) :
    ∀ p ∈ (Desc.shiftGates δ ps).toList, 0 < p.count := by
  intro p hp
  unfold Desc.shiftGates at hp
  rw [Array.toList_map, List.mem_map] at hp
  obtain ⟨q, hq, rfl⟩ := hp
  have := h q hq
  split <;> exact this

theorem segOk_mid (s : Seg) (hs : SegOk s) (lo n : Nat) : SegOk { s with ps := s.copy lo, reps := n } := by
  unfold SegOk at hs ⊢
  rw [copy_length]; exact hs

/-- A segment `Seg.slice` gives of one naming its `len` positions, its pieces of positive count, does too. -/
theorem segSlice_ok (s : Seg) (hs : SegOk s) (hp : ∀ q ∈ s.ps.toList, 0 < q.count) (a b : Nat) :
    ∀ g ∈ (s.slice a b).toList, SegOk g ∧ ∀ q ∈ g.ps.toList, 0 < q.count := by
  intro g hg
  have hone : ∀ (ps : Array Piece), (∀ q ∈ ps.toList, 0 < q.count) → g ∈ (ps.map Seg.one).toList →
      SegOk g ∧ ∀ q ∈ g.ps.toList, 0 < q.count := by
    intro ps hps h
    rw [Array.toList_map, List.mem_map] at h
    obtain ⟨q, hq, rfl⟩ := h
    refine ⟨segOk_one q, fun q' hq' => ?_⟩
    have e : q' = q := by simpa [Seg.one] using hq'
    rw [e]; exact hps q hq
  unfold Desc.Seg.slice at hg
  split at hg
  · simp at hg
  split at hg
  · exact hone _ (slicePieces_count _ _ _) hg
  dsimp only at hg
  split at hg
  · exact hone _ (slicePieces_count _ _ _) hg
  simp only [Array.toList_append, List.mem_append] at hg
  rcases hg with (hg | hg) | hg
  · split at hg
    · simp at hg
    · exact hone _ (slicePieces_count _ _ _) hg
  · generalize (if (a % s.len == 0) = true then a / s.len else a / s.len + 1) = lo at hg
    split at hg
    · simp only [List.mem_singleton] at hg
      subst hg
      exact ⟨segOk_mid s hs lo _, shiftGates_count _ _ hp⟩
    · simp at hg
  · exact hone _ (slicePieces_count _ _ _) hg

/-- A segment of a slice of a sequence `PS.of` makes, its pieces of positive count: naming its `len` positions, of a
copy at least, its pieces of positive count. -/
theorem psSlice_ok (r : PS) (hr : PSOk r) (hp : ∀ s ∈ r.segs.toList, ∀ q ∈ s.ps.toList, 0 < q.count) (a b : Nat) :
    ∀ g ∈ (r.slice a b).toList, SegOk g ∧ 1 ≤ g.reps ∧ ∀ q ∈ g.ps.toList, 0 < q.count := by
  intro g hg
  obtain ⟨i, a', b', hi⟩ := psSlice_mem r a b g hg
  have hsi : SegOk r.segs[i]! ∧ ∀ q ∈ r.segs[i]!.ps.toList, 0 < q.count := by
    by_cases h : i < r.segs.size
    · refine ⟨(hr.1 i h).1, hp _ ?_⟩
      rw [getElem!_pos r.segs i h]; exact Array.getElem_mem_toList h
    · rw [getElem!_neg r.segs i h]
      exact ⟨rfl, fun q hq => by rw [show (default : Seg).ps = #[] from rfl] at hq; simp at hq⟩
  obtain ⟨h1, h2⟩ := segSlice_ok _ hsi.1 hsi.2 a' b' g hi
  exact ⟨h1, (segSlice_reps _ a' b' g hi).1, h2⟩

/-- **`retAt`'s piece** names return position `j`. -/
theorem retAt_pos (r : PS) (hr : PSOk r) (hp : ∀ s ∈ r.segs.toList, ∀ q ∈ s.ps.toList, 0 < q.count) {j : Nat}
    {q : Piece} (h : Desc.retAt r j = some q) : (psPos r)[j]? = some (q.tgt, q.base) := by
  unfold Desc.retAt at h
  rcases hs : (r.slice j (j + 1))[0]? with _ | s
  · rw [hs] at h; cases h
  rw [hs] at h
  change s.ps[0]? = some q at h
  have hsm : s ∈ (r.slice j (j + 1)).toList := by
    obtain ⟨h0, e⟩ := Array.getElem?_eq_some_iff.1 hs
    rw [← e]; exact Array.getElem_mem_toList h0
  obtain ⟨-, hrep, hpos⟩ := psSlice_ok r hr hp j (j + 1) s hsm
  have hq := first_mem_segPos (by omega) h (hpos q (by
    obtain ⟨h0, e⟩ := Array.getElem?_eq_some_iff.1 h
    rw [← e]; exact Array.getElem_mem_toList h0))
  have hmem : (q.tgt, q.base) ∈ sl (psPos r) j (j + 1) := by
    rw [← psSlice_pos r hr]
    exact List.mem_flatMap.2 ⟨s, hsm, hq⟩
  unfold sl at hmem
  rw [show j + 1 - j = 1 by omega] at hmem
  rw [show j = j + 0 by rfl, ← List.getElem?_drop]
  rcases hd : (psPos r).drop j with _ | ⟨x, l⟩
  · rw [hd] at hmem; simp at hmem
  · rw [hd] at hmem
    simp only [List.take_succ_cons, List.take_zero, List.mem_singleton] at hmem
    rw [hmem]; rfl

/-- A loop that continues keeping `I` or stops with `Q` ends with one of them. -/
theorem forIn_done {α σ : Type} (I Q : σ → Prop) (f : α → σ → Except String (ForInStep σ))
    (hf : ∀ x s y, I s → f x s = .ok y → (∃ s', y = .yield s' ∧ I s') ∨ (∃ s', y = .done s' ∧ Q s')) :
    ∀ (l : List α) (s out : σ), I s → forIn l s f = .ok out → I out ∨ Q out
  | [], s, out, hs, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h; exact .inl hs
  | x :: l, s, out, hs, h => by
    rw [List.forIn_cons] at h
    rcases hx : f x s with e | y <;> rw [hx] at h
    · exact absurd h (by intro h; cases h)
    rcases hf x s y hs hx with ⟨s', rfl, hs'⟩ | ⟨s', rfl, hq⟩
    · exact forIn_done I Q f hf l s' out hs' h
    · change Except.ok s' = Except.ok out at h
      injection h with h
      subst h; exact .inr hq

/-- What `chainOf` follows from return position `j`: each next position the carry leaf the one before returns, then
the piece the last one returns when no carry leaf, else (`exit = none`) the carry leaf it returns is position `cyc`'s. -/
def ChainOk (r : PS) (j : Nat) (ch : Desc.Chain) : Prop :=
  0 < ch.hist.size ∧ ch.hist[0]! = j ∧
  (∀ s, s + 1 < ch.hist.size → ∃ q, Desc.retAt r ch.hist[s]! = some q ∧ q.tgt = .ref true 0 ∧
    q.base.toNat = ch.hist[s + 1]!) ∧
  (∀ q, ch.exit = some q → Desc.retAt r ch.hist[ch.hist.size - 1]! = some q ∧ q.tgt ≠ .ref true 0) ∧
  (ch.exit = none → ch.cyc < ch.hist.size ∧ ∃ q, Desc.retAt r ch.hist[ch.hist.size - 1]! = some q ∧
    q.tgt = .ref true 0 ∧ q.base.toNat = ch.hist[ch.cyc]!)

theorem push_get! (a : Array Nat) (x i : Nat) (hi : i < a.size) : (a.push x)[i]! = a[i]! := by
  rw [getElem!_pos (a.push x) i (by simp; omega), Array.getElem_push_lt hi, getElem!_pos a i hi]

theorem push_get!_size (a : Array Nat) (x : Nat) : (a.push x)[a.size]! = x := by
  rw [getElem!_pos (a.push x) a.size (by simp), Array.getElem_push_eq]

/-- **`chainOf`** follows return position `j` as `ChainOk` says. -/
theorem chainOf_ok (r : PS) (j : Nat) (ch : Desc.Chain) (h : Desc.chainOf r j = .ok ch) : ChainOk r j ch := by
  unfold Desc.chainOf at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at h
  obtain ⟨st, hst, h⟩ := Walk.except_bind_ok h
  let I : Option Desc.Chain × Array Nat × Std.HashMap Nat Nat → Prop := fun st =>
    st.1 = none ∧ 0 < st.2.1.size ∧ st.2.1[0]! = j ∧
    (∀ s, s + 1 < st.2.1.size → ∃ q, Desc.retAt r st.2.1[s]! = some q ∧ q.tgt = .ref true 0 ∧
      q.base.toNat = st.2.1[s + 1]!) ∧
    ∀ x v, st.2.2[x]? = some v → v < st.2.1.size ∧ st.2.1[v]! = x
  let Q : Option Desc.Chain × Array Nat × Std.HashMap Nat Nat → Prop := fun st =>
    ∃ ch, st.1 = some ch ∧ ChainOk r j ch
  have hI0 : I (none, #[j], (∅ : Std.HashMap Nat Nat).insert j 0) := by
    refine ⟨rfl, by simp, rfl, fun s hs => absurd hs (by simp), fun x v hv => ?_⟩
    rw [Std.HashMap.getElem?_insert] at hv
    split at hv
    · rename_i e
      simp only [beq_iff_eq] at e
      simp only [Option.some.injEq] at hv
      subst hv; subst e
      exact ⟨by simp, rfl⟩
    · simp at hv
  refine (forIn_done I Q _ ?_ _ _ st hI0 hst).elim (fun hI => ?_) (fun ⟨ch', hch, hok⟩ => ?_)
  rotate_left
  · rw [hI.1] at h; exact absurd h (by intro h; cases h)
  · rw [hch] at h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h; exact hok
  intro _ s y ⟨hs1, hsz, hs0, hsc, hseen⟩ hy
  rcases hq : Desc.retAt r s.2.1.back! with _ | q <;> rw [hq] at hy <;> dsimp only at hy
  · exact absurd hy (by intro h; cases h)
  have hback : s.2.1.back! = s.2.1[s.2.1.size - 1]! := rfl
  rw [hback] at hq
  split at hy
  · rename_i ht
    split at hy
    · rename_i cy hcy
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      subst hy
      obtain ⟨hcl, hce⟩ := hseen _ _ hcy
      refine .inr ⟨_, rfl, _, rfl, hsz, hs0, hsc, fun q' hq' => (by cases hq'), fun _ => ⟨hcl, q, hq, ht, hce.symm⟩⟩
    · rename_i hnone
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      subst hy
      refine .inl ⟨_, rfl, rfl, by simp, ?_, fun t ht' => ?_, fun x v hv => ?_⟩
      · rw [push_get! _ _ _ hsz]; exact hs0
      · simp only [Array.size_push] at ht'
        by_cases hlt : t + 1 < s.2.1.size
        · rw [push_get! _ _ _ (by omega), push_get! _ _ _ hlt]
          exact hsc t hlt
        · have e : t = s.2.1.size - 1 := by omega
          subst e
          rw [push_get! _ _ _ (by omega), show s.2.1.size - 1 + 1 = s.2.1.size by omega, push_get!_size]
          exact ⟨q, hq, ht, rfl⟩
      · simp only [Array.size_push]
        rw [Std.HashMap.getElem?_insert] at hv
        split at hv
        · rename_i e
          simp only [beq_iff_eq] at e
          simp only [Option.some.injEq] at hv
          subst hv; subst e
          exact ⟨by omega, push_get!_size _ _⟩
        · obtain ⟨h1, h2⟩ := hseen x v hv
          exact ⟨by omega, by rw [push_get! _ _ _ h1]; exact h2⟩
  · rename_i ht
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    subst hy
    refine .inr ⟨_, rfl, _, rfl, hsz, hs0, hsc, fun q' hq' => ?_, fun h => (by cases h)⟩
    simp only [Option.some.injEq] at hq'
    subst hq'
    exact ⟨hq, fun e => ht e⟩

/-- Index `t` of a chain of `n` positions whose cycle starts at `c`: past its positions, around the cycle. -/
def cycIdx (n c t : Nat) : Nat := if t < n then t else c + (t - c) % (n - c)

theorem chainAt_idx (ch : Desc.Chain) (t : Nat) : ch.at t = ch.hist[cycIdx ch.hist.size ch.cyc t]! := by
  unfold Desc.Chain.at cycIdx
  split <;> rfl

theorem chainAt_lt (ch : Desc.Chain) {t : Nat} (ht : t < ch.hist.size) : ch.at t = ch.hist[t]! := by
  rw [chainAt_idx, cycIdx, ite_eq_left ht]

/-- Past its positions a chain goes around its cycle `[c, n)`: each index the one after the one before, after the last
`c`. -/
theorem cyc_idx (n c t : Nat) (hc : c < n) : cycIdx n c t < n ∧
    ((cycIdx n c t + 1 < n ∧ cycIdx n c (t + 1) = cycIdx n c t + 1) ∨
      (cycIdx n c t = n - 1 ∧ cycIdx n c (t + 1) = c)) := by
  have hL : 0 < n - c := by omega
  unfold cycIdx
  by_cases h1 : t + 1 < n
  · rw [ite_eq_left (show t < n by omega), ite_eq_left h1]; omega
  by_cases h2 : t + 1 = n
  · rw [ite_eq_left (show t < n by omega), ite_eq_right h1, show t + 1 - c = n - c by omega, Nat.mod_self]; omega
  rw [ite_eq_right (show ¬t < n by omega), ite_eq_right h1]
  have hm := Nat.mod_lt (t - c) hL
  have hdm := Nat.div_add_mod (t - c) (n - c)
  by_cases h3 : (t - c) % (n - c) + 1 < n - c
  · rw [show t + 1 - c = ((t - c) % (n - c) + 1) + (n - c) * ((t - c) / (n - c)) by omega,
      Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt h3]
    omega
  · rw [show t + 1 - c = 0 + (n - c) * ((t - c) / (n - c) + 1) by rw [Nat.mul_add, Nat.mul_one]; omega,
      Nat.add_mul_mod_self_left, Nat.zero_mod]
    omega

/-- **In a cycle** (`exit = none`) every position of the chain returns the carry leaf the next one is. -/
theorem chainAt_cyc {r : PS} {j : Nat} {ch : Desc.Chain} (h : ChainOk r j ch) (hx : ch.exit = none) (t : Nat) :
    ∃ q, Desc.retAt r (ch.at t) = some q ∧ q.tgt = .ref true 0 ∧ q.base.toNat = ch.at (t + 1) := by
  obtain ⟨hsz, -, hsc, -, hcy⟩ := h
  obtain ⟨hc, q, hq, ht, hb⟩ := hcy hx
  obtain ⟨-, hstep⟩ := cyc_idx ch.hist.size ch.cyc t hc
  rw [chainAt_idx, chainAt_idx ch (t + 1)]
  rcases hstep with ⟨h1, h2⟩ | ⟨h1, h2⟩
  · rw [h2]; exact hsc _ h1
  · rw [h1, h2]; exact ⟨q, hq, ht, hb⟩

theorem agree_get {P : Prog} {sc : Flock.Program.Scope} : ∀ {xs : List (Tgt × Int)} {vs : List (Except String Nat)},
    Agree P sc xs vs → ∀ {i : Nat} {x : Tgt × Int} {v : Except String Nat}, xs[i]? = some x → vs[i]? = some v →
      ∀ a, v = .ok a → den P sc x = .ok a
  | _, _, .nil, _, _, _, hx, _ => by simp at hx
  | _, _, .cons h _, 0, x, v, hx, hv => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at hx hv
    subst hx; subst hv; exact h
  | _, _, .cons _ hrest, i + 1, _, _, hx, hv => by
    simp only [List.getElem?_cons_succ] at hx hv
    exact agree_get hrest hx hv

/-- **Up through a step's return**: a return leaf of an iteration that the step's summary names as carry leaf `h'`
resolves as that leaf. -/
theorem carry_up (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) {k : Nat} {node : Flock.Program.Node}
    (hn : c.node k = .ok node) (hRet : RetOk c.P c.memo node.fn) {m h h' a : Nat}
    (hx : (psPos (c.memo[node.fn]!).ret)[h]? = some (.ref true 0, (h' : Int)))
    (hv : Flock.Program.resolve c.P (c.P.child sc k m) ((c.P.defs[node.fn]!).body.ret.get h) = .ok a) :
    Flock.Program.resolve c.P (c.P.child sc k m) ⟨true, 0, h'⟩ = .ok a := by
  obtain ⟨-, -, -, hag, -⟩ := hRet
  have hd' : (c.P.child sc k m).d = node.fn := by rw [child_eq c sc hd hn m]; rfl
  have hA := hag _ hd'
  have hlt : h < (c.P.defs[node.fn]!).body.ret.count := by
    have := (List.getElem?_eq_some_iff.1 hx).1
    rw [hA.length_eq] at this; simpa using this
  have := agree_get hA hx (by rw [List.getElem?_map, List.getElem?_range hlt]; rfl) a hv
  simp only [den, Int.toNat_natCast] at this
  rwa [ite_eq_right (by omega)] at this

/-- **Back one iteration**: the carry leaf `h` of an iteration past the first resolves as the iteration before's return
leaf `h`. -/
theorem carry_back (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) {k : Nat} {node : Flock.Program.Node}
    (hn : c.node k = .ok node) (hform : node.form = .scan) {m h a : Nat} (hm : m ≠ 0)
    (hv : Flock.Program.resolve c.P (c.P.child sc k m) ⟨true, 0, h⟩ = .ok a) :
    Flock.Program.resolve c.P (c.P.child sc k (m - 1)) ((c.P.defs[node.fn]!).body.ret.get h) = .ok a := by
  have hnode : (c.P.defs[sc.d]!).body.nodes[k]! = node := by rw [hd]; exact node_get hn
  rw [child_eq c sc hd hn m] at hv
  have := den_up_scan0 c.P sc hnode hform hv
  rw [ite_eq_right hm] at this
  rw [child_eq c sc hd hn (m - 1)]
  obtain ⟨n, rfl⟩ : ∃ n, m = n + 1 := ⟨m - 1, by omega⟩
  rw [Nat.add_sub_cancel] at this ⊢
  rwa [show sc.off + (c.P.defs[sc.d]!).body.offsets[k]! + (n + 1) * (c.P.defs[node.fn]!).gates -
      (c.P.defs[node.fn]!).gates = sc.off + (c.P.defs[sc.d]!).body.offsets[k]! + n * (c.P.defs[node.fn]!).gates by
    rw [Nat.add_mul, Nat.one_mul]; omega] at this

/-- **Back through the carry**: a return leaf of iteration `m` the step's summary names carry leaf by carry leaf `t`
times (`f`) resolves as iteration `m - t`'s return leaf `f t`. -/
theorem carry_chain (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) {k : Nat} {node : Flock.Program.Node}
    (hn : c.node k = .ok node) (hform : node.form = .scan) (hRet : RetOk c.P c.memo node.fn) (f : Nat → Nat)
    {m a : Nat} : ∀ t ≤ m, (∀ s < t, (psPos (c.memo[node.fn]!).ret)[f s]? = some (.ref true 0, (f (s + 1) : Int))) →
      Flock.Program.resolve c.P (c.P.child sc k m) ((c.P.defs[node.fn]!).body.ret.get (f 0)) = .ok a →
      Flock.Program.resolve c.P (c.P.child sc k (m - t)) ((c.P.defs[node.fn]!).body.ret.get (f t)) = .ok a
  | 0, _, _, hv => hv
  | t + 1, ht, hs, hv => by
    have ih := carry_chain c sc hd hn hform hRet f t (by omega) (fun s hs' => hs s (by omega)) hv
    have hup := carry_up c sc hd hn hRet (hs t (by omega)) ih
    have := carry_back c sc hd hn hform (m := m - t) (by omega) hup
    rwa [show m - t - 1 = m - (t + 1) by omega] at this

/-- **Back to the initial carry**: a return leaf of iteration `m` the step's summary names carry leaf by carry leaf
`m + 1` times resolves as the first iteration's carry leaf `f (m + 1)`. -/
theorem carry_init (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) {k : Nat} {node : Flock.Program.Node}
    (hn : c.node k = .ok node) (hform : node.form = .scan) (hRet : RetOk c.P c.memo node.fn) (f : Nat → Nat)
    {m a : Nat} (hs : ∀ s ≤ m, (psPos (c.memo[node.fn]!).ret)[f s]? = some (.ref true 0, (f (s + 1) : Int)))
    (hv : Flock.Program.resolve c.P (c.P.child sc k m) ((c.P.defs[node.fn]!).body.ret.get (f 0)) = .ok a) :
    Flock.Program.resolve c.P (c.P.child sc k 0) ⟨true, 0, f (m + 1)⟩ = .ok a := by
  have := carry_chain c sc hd hn hform hRet f m le_rfl (fun s hs' => hs s (by omega)) hv
  rw [Nat.sub_self] at this
  exact carry_up c sc hd hn hRet (hs m le_rfl) this

theorem pos_carry (lo n : Nat) :
    pos { tgt := .ref true 0, base := (lo : Int), stride := 1, len := n } =
      (List.range' lo n).map fun (j : Nat) => (Tgt.ref true 0, (j : Int)) := by
  rw [pos_one, List.range'_eq_map_range, List.map_map]
  apply List.map_congr_left
  intro t _
  simp

/-- **The initial carry's leaves `[lo, hi)`**, through `throughArg`, keep the gates the first iteration's carry leaves
resolve to. -/
theorem init_agree (c : Desc.Ctx) (sc : Flock.Program.Scope) {fuel k : Nat} {node : Flock.Program.Node}
    (hn : c.node k = .ok node) (hT : ThroughSpec c sc fuel) (hm0 : 0 < node.members) {lo hi : Nat}
    (hlv : ∀ j, lo ≤ j → j < hi → 0 < (c.P.defs[node.fn]!).body.params.size ∧
      j < ((c.P.defs[node.fn]!).body.params[0]!).2.leaves) {segs : Array Seg}
    (h : Desc.throughArg c fuel k 0 0 { tgt := .ref true 0, base := lo, stride := 1, len := hi - lo } = .ok segs) :
    Agree c.P sc (segsPos segs) ((List.range' lo (hi - lo)).map fun j =>
      Flock.Program.resolve c.P (c.P.child sc k 0) ⟨true, 0, j⟩) := by
  have hnode := node_get hn
  have := hT k 0 0 _ segs (by rw [hnode]; exact hm0) rfl (fun x hx => by
    rw [pos_carry, List.mem_map] at hx
    obtain ⟨j, hj, rfl⟩ := hx
    rw [List.mem_range'_1] at hj
    rw [hnode]
    have := hlv j hj.1 (by omega)
    exact ⟨by simp, this.1, by show (j : Int) < _; exact_mod_cast this.2⟩) h
  rw [pos_carry, List.map_map] at this
  have e : ((List.range' lo (hi - lo)).map fun j =>
      Flock.Program.resolve c.P (c.P.child sc k 0) ⟨true, 0, j⟩) =
      (List.range' lo (hi - lo)).map (den c.P (c.P.child sc k 0) ∘ fun (j : Nat) => (Tgt.ref true 0, (j : Int))) := by
    apply List.map_congr_left
    intro j _
    simp [den]
  rw [e]; exact this

/-- `carryRuns`' step over a piece of a segment of one copy. -/
def runStep (st : Array (Nat × Nat × Bool) × Nat) (q : Piece) : Array (Nat × Nat × Bool) × Nat :=
  (st.1.push (st.2, st.2 + q.count, q.tgt == Tgt.ref true 0 && decide (q.rows ≤ 1) && q.base == (st.2 : Int) &&
    (q.stride == 1 || decide (q.len ≤ 1))), st.2 + q.count)

/-- `carryRuns`' step over a segment. -/
def segStep (st : Array (Nat × Nat × Bool) × Nat) (s : Seg) : Array (Nat × Nat × Bool) × Nat :=
  if s.reps > 1 then (st.1.push (st.2, st.2 + s.count, false), st.2 + s.count) else s.ps.toList.foldl runStep st

theorem carryRuns_eq (r : PS) (a b : Nat) :
    Desc.carryRuns r a b = ((r.slice a b).toList.foldl segStep (#[], a)).1 := by
  unfold Desc.carryRuns
  simp only [Id.run]
  rw [← Array.forIn_toList, forIn_id_yield _ _ (fun s st => segStep st s)]
  · rfl
  intro s st
  unfold segStep
  split
  · rfl
  · rw [← Array.forIn_toList, forIn_id_yield _ _ (fun q st => runStep st q)]
    · rfl
    · intro q st; rfl

/-- Runs `out` lie end to end over `[a, p)`, a run the step returns in place (`id`) naming carry leaves where `L` (the
positions from `a`) does. -/
def RunsOk (L : List (Tgt × Int)) (a : Nat) (out : List (Nat × Nat × Bool)) (p : Nat) : Prop :=
  a ≤ p ∧ out.flatMap (fun x => List.range' x.1 (x.2.1 - x.1)) = List.range' a (p - a) ∧
  ∀ x ∈ out, x.2.2 = true → ∀ j, x.1 ≤ j → j < x.2.1 → a ≤ j ∧ L[j - a]? = some (.ref true 0, (j : Int))

theorem runsOk_push {L : List (Tgt × Int)} {a p : Nat} {out : List (Nat × Nat × Bool)} (h : RunsOk L a out p)
    (n : Nat) (id : Bool)
    (hid : id = true → ∀ j, p ≤ j → j < p + n → L[j - a]? = some (.ref true 0, (j : Int))) :
    RunsOk L a (out ++ [(p, p + n, id)]) (p + n) := by
  obtain ⟨hap, hfl, hids⟩ := h
  refine ⟨by omega, ?_, fun x hx hxid j h1 h2 => ?_⟩
  · rw [List.flatMap_append, hfl, List.flatMap_singleton, range'_split a p (p + n) hap (by omega)]
  · rw [List.mem_append, List.mem_singleton] at hx
    rcases hx with hx | rfl
    · exact hids x hx hxid j h1 h2
    · exact ⟨by dsimp only at h1; omega, hid hxid j h1 h2⟩

theorem runs_pieces (L : List (Tgt × Int)) (a : Nat) : ∀ (qs : List Piece) (st : Array (Nat × Nat × Bool) × Nat),
    RunsOk L a st.1.toList st.2 → (∀ i x, (qs.flatMap pos)[i]? = some x → L[st.2 - a + i]? = some x) →
    RunsOk L a (qs.foldl runStep st).1.toList (qs.foldl runStep st).2 ∧
      (qs.foldl runStep st).2 = st.2 + (qs.flatMap pos).length
  | [], st, h, _ => ⟨h, by simp⟩
  | q :: qs, st, h, hL => by
    rw [List.foldl_cons]
    have h1 : RunsOk L a (runStep st q).1.toList (runStep st q).2 := by
      unfold runStep
      dsimp only
      rw [Array.toList_push]
      refine runsOk_push h q.count _ fun hid j hj1 hj2 => ?_
      simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, Bool.or_eq_true] at hid
      obtain ⟨⟨⟨ht, hrows⟩, hbase⟩, hst⟩ := hid
      have ht := tgt_eq_of_beq ht
      have hx := hL (j - st.2) (q.tgt, ppt q (j - st.2)) (by
        rw [List.flatMap_cons, List.getElem?_append_left (by rw [pos_length]; omega), pos_getElem?,
          ite_eq_left (by omega)])
      have hr1 : q.rows = 1 := by
        rcases Nat.lt_or_ge q.rows 1 with h0 | h0
        · have : q.rows = 0 := by omega
          simp [Desc.Piece.count, this] at hj2; omega
        · omega
      have hlt : j - st.2 < q.len := by
        have : q.count = q.len := by simp [Desc.Piece.count, hr1]
        omega
      rw [show st.2 - a + (j - st.2) = j - a by have := h.1; omega] at hx
      rw [hx, ht]
      simp only [Option.some.injEq, Prod.mk.injEq, true_and]
      unfold ppt
      rw [Nat.div_eq_of_lt hlt, Nat.mod_eq_of_lt hlt, hbase]
      rcases hst with hs | hs
      · rw [hs]; push_cast; omega
      · have : j - st.2 = 0 := by omega
        rw [this]; push_cast; omega
    have hL1 : ∀ i x, (qs.flatMap pos)[i]? = some x → L[(runStep st q).2 - a + i]? = some x := by
      intro i x hx
      have := hL (q.count + i) x (by
        rw [List.flatMap_cons, List.getElem?_append_right (by rw [pos_length]; omega), pos_length,
          Nat.add_sub_cancel_left]
        exact hx)
      unfold runStep; dsimp only
      rw [show st.2 + q.count - a + i = st.2 - a + (q.count + i) by have := h.1; omega]
      exact this
    obtain ⟨h2, e2⟩ := runs_pieces L a qs (runStep st q) h1 hL1
    refine ⟨h2, ?_⟩
    rw [e2, List.flatMap_cons, List.length_append, pos_length]
    unfold runStep; dsimp only; omega

theorem runs_segs (L : List (Tgt × Int)) (a : Nat) : ∀ (ss : List Seg) (st : Array (Nat × Nat × Bool) × Nat),
    (∀ s ∈ ss, SegOk s ∧ 1 ≤ s.reps) → RunsOk L a st.1.toList st.2 →
    (∀ i x, (ss.flatMap segPos)[i]? = some x → L[st.2 - a + i]? = some x) →
    RunsOk L a (ss.foldl segStep st).1.toList (ss.foldl segStep st).2 ∧
      (ss.foldl segStep st).2 = st.2 + (ss.flatMap segPos).length
  | [], st, _, h, _ => ⟨h, by simp⟩
  | s :: ss, st, hs, h, hL => by
    rw [List.foldl_cons]
    obtain ⟨hok, hrep⟩ := hs s List.mem_cons_self
    have hlen : (segPos s).length = s.count := segPos_length s hok
    have h1 : RunsOk L a (segStep st s).1.toList (segStep st s).2 ∧ (segStep st s).2 = st.2 + s.count := by
      unfold segStep
      split
      · dsimp only
        rw [Array.toList_push]
        exact ⟨runsOk_push h s.count false (fun h => by cases h), rfl⟩
      · rename_i hr
        have h1 : s.reps = 1 := by omega
        have hpa : segPos s = s.ps.toList.flatMap pos := segPos_one_rep h1
        obtain ⟨h2, e2⟩ := runs_pieces L a s.ps.toList st h (fun i x hx => hL i x (by
          rw [List.flatMap_cons, List.getElem?_append_left (by rw [hpa]; exact (List.getElem?_eq_some_iff.1 hx).1),
            hpa]
          exact hx))
        refine ⟨h2, ?_⟩
        rw [e2, ← hpa, hlen]
    have hL1 : ∀ i x, (ss.flatMap segPos)[i]? = some x → L[(segStep st s).2 - a + i]? = some x := by
      intro i x hx
      have := hL (s.count + i) x (by
        rw [List.flatMap_cons, List.getElem?_append_right (by omega), hlen, Nat.add_sub_cancel_left]
        exact hx)
      rw [h1.2, show st.2 + s.count - a + i = st.2 - a + (s.count + i) by have := h.1; omega]
      exact this
    obtain ⟨h2, e2⟩ := runs_segs L a ss (segStep st s) (fun g hg => hs g (List.mem_cons_of_mem _ hg)) h1.1 hL1
    refine ⟨h2, ?_⟩
    rw [e2, h1.2, List.flatMap_cons, List.length_append, hlen]
    omega

/-- **`carryRuns`' runs** lie end to end over `[a, b)`, each one the step returns in place (`id`) naming carry leaf `j`
at every return position `j` of it. -/
theorem carryRuns_cover (r : PS) (hr : PSOk r) (hp : ∀ s ∈ r.segs.toList, ∀ q ∈ s.ps.toList, 0 < q.count)
    (a b : Nat) (hb : a < b → b ≤ (psPos r).length) :
    (Desc.carryRuns r a b).toList.flatMap (fun x => List.range' x.1 (x.2.1 - x.1)) = List.range' a (b - a) ∧
    ∀ x ∈ (Desc.carryRuns r a b).toList, x.2.2 = true → ∀ j, x.1 ≤ j → j < x.2.1 →
      (psPos r)[j]? = some (.ref true 0, (j : Int)) := by
  rw [carryRuns_eq]
  have hsl := psSlice_pos r hr a b
  obtain ⟨⟨-, hfl, hids⟩, e⟩ := runs_segs (sl (psPos r) a b) a (r.slice a b).toList (#[], a)
    (fun s hs => by obtain ⟨h1, h2, -⟩ := psSlice_ok r hr hp a b s hs; exact ⟨h1, h2⟩)
    ⟨le_rfl, by simp, fun x hx => by simp at hx⟩
    (fun i x hx => by
      show (sl (psPos r) a b)[a - a + i]? = some x
      rw [Nat.sub_self, Nat.zero_add, ← hsl]; exact hx)
  have hlen : ((r.slice a b).toList.flatMap segPos).length = b - a := by
    change (segsPos (r.slice a b)).length = b - a
    rw [hsl]
    unfold sl
    rw [List.length_take, List.length_drop]
    by_cases hab : a < b
    · have := hb hab; omega
    · omega
  dsimp only at e
  rw [hlen] at e
  refine ⟨by rw [hfl, e]; congr 1; omega, fun x hx hid j h1 h2 => ?_⟩
  obtain ⟨haj, hj⟩ := hids x hx hid j h1 h2
  unfold sl at hj
  rw [List.getElem?_take] at hj
  split at hj
  · rw [List.getElem?_drop, show a + (j - a) = j by omega] at hj
    exact hj
  · cases hj

/-- `carryRuns`' runs the step returns in place name carry leaf `j` at every return position `j` of them, whatever
`[a, b)`. -/
theorem carryRuns_id (r : PS) (hr : PSOk r) (hp : ∀ s ∈ r.segs.toList, ∀ q ∈ s.ps.toList, 0 < q.count)
    (a b : Nat) : ∀ x ∈ (Desc.carryRuns r a b).toList, x.2.2 = true → ∀ j, x.1 ≤ j → j < x.2.1 →
      (psPos r)[j]? = some (.ref true 0, (j : Int)) := by
  rw [carryRuns_eq]
  have hsl := psSlice_pos r hr a b
  obtain ⟨⟨-, -, hids⟩, -⟩ := runs_segs (sl (psPos r) a b) a (r.slice a b).toList (#[], a)
    (fun s hs => by obtain ⟨h1, h2, -⟩ := psSlice_ok r hr hp a b s hs; exact ⟨h1, h2⟩)
    ⟨le_rfl, by simp, fun x hx => by simp at hx⟩
    (fun i x hx => by
      show (sl (psPos r) a b)[a - a + i]? = some x
      rw [Nat.sub_self, Nat.zero_add, ← hsl]; exact hx)
  intro x hx hid j h1 h2
  obtain ⟨haj, hj⟩ := hids x hx hid j h1 h2
  unfold sl at hj
  rw [List.getElem?_take] at hj
  split at hj
  · rw [List.getElem?_drop, show a + (j - a) = j by omega] at hj
    exact hj
  · cases hj


theorem forall2_map_same {α β γ : Type} (R : β → γ → Prop) (f : α → β) (g : α → γ) :
    ∀ (l : List α), (∀ x ∈ l, R (f x) (g x)) → List.Forall₂ R (l.map f) (l.map g)
  | [], _ => .nil
  | x :: l, h => .cons (h x List.mem_cons_self) (forall2_map_same R f g l fun y hy => h y (List.mem_cons_of_mem _ hy))

theorem flatMap_single {α β : Type} (l : List α) (f : α → β) : (l.flatMap fun x => [f x]) = l.map f := by
  induction l with
  | nil => rfl
  | cons x l ih => rw [List.flatMap_cons, List.map_cons, ih]; rfl

/-- A return position the step's summary names as a carry leaf, as `retAt` gives it. -/
theorem carryPos_of_retAt {P : Prog} {memo : Array Desc.Summary} {f : Nat} (hR : RetOk P memo f) {h h' : Nat}
    {q : Piece} (hq : Desc.retAt (memo[f]!).ret h = some q) (ht : q.tgt = .ref true 0) (hb : q.base.toNat = h') :
    (psPos (memo[f]!).ret)[h]? = some (.ref true 0, (h' : Int)) := by
  obtain ⟨hps, hpos, -, -, hpp⟩ := hR
  have hx := retAt_pos _ hps hpp hq
  have h0 := (hpos _ (List.mem_of_getElem? hx)).1
  rw [hx, ht, ← hb]
  dsimp only at h0
  congr 2; omega

/-- **`carry`'s step**: one step up, `carry` keeps the gates scan iteration `m`'s return leaves resolve to, wherever
the step's summary holds `RetOk` and `fromCallee` and `throughArg` one step down keep what they read. -/
theorem carrySpec_succ (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (fuel : Nat)
    (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn) (hC : CalleeSpec c sc fuel)
    (hT : ThroughSpec c sc fuel) : CarrySpec c sc (fuel + 1) := by
  intro k m a b segs hform hm hb h
  rw [Desc.carry.eq_2] at h
  obtain ⟨node, hn, h⟩ := Walk.except_bind_ok h
  dsimp only at h
  have h' := ok_bind_pure h
  rw [← Array.forIn_toList] at h'
  have hnode : c.body.nodes[k]! = node := node_get hn
  rw [hnode] at hform hm hb ⊢
  have hRk := hR k node hn
  obtain ⟨hps, hpos, -, hag, hpp⟩ := hRk
  have hlenR : (psPos (c.memo[node.fn]!).ret).length = (c.P.defs[node.fn]!).body.ret.count := by
    have := (hag (c.P.child sc k 0) (by rw [child_eq c sc hd hn 0]; rfl)).length_eq
    simpa using this
  obtain ⟨hcov, hid⟩ := carryRuns_cover _ hps hpp a b (fun hab => by rw [hlenR]; exact hb hab)
  have hm0 : 0 < node.members := by omega
  have hcarry : ∀ h j' : Nat, (psPos (c.memo[node.fn]!).ret)[h]? = some (.ref true 0, (j' : Int)) →
      0 < (c.P.defs[node.fn]!).body.params.size ∧ j' < ((c.P.defs[node.fn]!).body.params[0]!).2.leaves := by
    intro h j' hx
    have := (hpos _ (List.mem_of_getElem? hx)).2.2 0 rfl
    exact ⟨this.1, by have h2 : ((j' : Nat) : Int) < _ := this.2; exact_mod_cast h2⟩
  obtain ⟨out, hout, hA⟩ : ∃ out, segs = #[] ++ out ∧ Agree c.P sc (segsPos out)
      ((Desc.carryRuns (c.memo[node.fn]!).ret a b).toList.flatMap fun x =>
        (List.range' x.1 (x.2.1 - x.1)).map fun j =>
          Flock.Program.resolve c.P (c.P.child sc k m) ((c.P.defs[node.fn]!).body.ret.get j)) := by
    refine forIn_ext (P := c.P) (sc := sc) _ _ _ ?_ #[] segs h'
    intro x hx r' y hy
    split at hy
    · rename_i hxid
      obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      subst hy
      refine ⟨v, rfl, ?_⟩
      have hin := init_agree c sc hn hT hm0 (fun j h1 h2 => hcarry j j (hid x hx hxid j h1 h2)) hv
      refine agree_weaken hin (forall2_map_same _ _ _ _ fun j hj => ?_)
      intro a ha
      rw [List.mem_range'_1] at hj
      exact carry_init c sc hd hn hform (hR k node hn) (fun _ => j) (fun s _ => hid x hx hxid j hj.1 (by omega)) ha
    · have hy := walkAtMost_bind _ _ _ hy
      obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      subst hy
      simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at hv
      obtain ⟨segs', hsegs', hA2⟩ : ∃ segs', v = r' ++ segs' ∧ Agree c.P sc (segsPos segs')
          ((List.range' x.1 ((x.2.1 - x.1 + 1 - 1) / 1)).flatMap fun j =>
            [Flock.Program.resolve c.P (c.P.child sc k m) ((c.P.defs[node.fn]!).body.ret.get j)]) := by
        refine forIn_ext (P := c.P) (sc := sc) _ _ _ ?_ r' v hv
        intro j _ r'' y' hy'
        obtain ⟨ch, hch, hy'⟩ := Walk.except_bind_ok hy'
        have hok := chainOf_ok _ j ch hch
        split at hy'
        · rename_i hcond
          simp only [Bool.and_eq_true, decide_eq_true_eq, Option.isSome_iff_exists] at hcond
          obtain ⟨⟨q0, hq0⟩, he⟩ := hcond
          obtain ⟨v', hv', hy'⟩ := Walk.except_bind_ok hy'
          simp only [pure, Except.pure, Except.ok.injEq] at hy'
          subst hy'
          refine ⟨v', rfl, ?_⟩
          obtain ⟨hsz, h0, hsc, hex, -⟩ := hok
          obtain ⟨hq, -⟩ := hex q0 hq0
          have hlt : ch.hist[ch.hist.size - 1]! < (c.P.defs[node.fn]!).body.ret.count := by
            have := retAt_pos _ hps hpp hq
            rw [← hlenR]; exact (List.getElem?_eq_some_iff.1 this).1
          have hcal := hC k (m - (ch.hist.size - 1)) _ _ v' (by rw [hnode]; omega) (by rw [hnode]; intro; omega) hv'
          rw [hnode, show ch.hist[ch.hist.size - 1]! + 1 - ch.hist[ch.hist.size - 1]! = 1 by omega] at hcal
          simp only [List.range'_one, List.map_cons, List.map_nil] at hcal
          refine agree_weaken hcal (List.Forall₂.cons (fun a ha => ?_) List.Forall₂.nil)
          have := carry_chain c sc hd hn hform (hR k node hn) (fun s => ch.hist[s]!) (ch.hist.size - 1) he
            (fun s hs => by
              obtain ⟨q, hq, ht, hb⟩ := hsc s (by omega)
              exact carryPos_of_retAt (hR k node hn) hq ht hb) (by rw [h0]; exact ha)
          exact this
        · rename_i hcond
          obtain ⟨v', hv', hy'⟩ := Walk.except_bind_ok hy'
          simp only [pure, Except.pure, Except.ok.injEq] at hy'
          subst hy'
          refine ⟨v', rfl, ?_⟩
          have hs : ∀ s ≤ m, (psPos (c.memo[node.fn]!).ret)[ch.at s]? = some (.ref true 0, (ch.at (s + 1) : Int)) := by
            intro s hs
            rcases hex : ch.exit with _ | q0
            · obtain ⟨q, hq, ht, hb⟩ := chainAt_cyc hok hex s
              exact carryPos_of_retAt (hR k node hn) hq ht hb
            · have he : ¬ ch.hist.size - 1 ≤ m := fun he => hcond (by simp [hex, he])
              obtain ⟨-, -, hsc, -, -⟩ := hok
              obtain ⟨q, hq, ht, hb⟩ := hsc s (by omega)
              rw [chainAt_lt ch (show s < ch.hist.size by omega), chainAt_lt ch (show s + 1 < ch.hist.size by omega)]
              exact carryPos_of_retAt (hR k node hn) hq ht hb
          have hin := init_agree c sc hn hT hm0 (lo := ch.at (m + 1)) (hi := ch.at (m + 1) + 1)
            (fun j' h1 h2 => by rw [show j' = ch.at (m + 1) by omega]; exact hcarry _ _ (hs m le_rfl)) hv'
          rw [show ch.at (m + 1) + 1 - ch.at (m + 1) = 1 by omega] at hin
          simp only [List.range'_one, List.map_cons, List.map_nil] at hin
          refine agree_weaken hin (List.Forall₂.cons (fun a ha => ?_) List.Forall₂.nil)
          have h0 : ch.at 0 = j := by rw [chainAt_lt ch hok.1]; exact hok.2.1
          exact carry_init c sc hd hn hform (hR k node hn) ch.at hs (by rw [h0]; exact ha)
      refine ⟨segs', by rw [hsegs'], ?_⟩
      rw [show (x.2.1 - x.1 + 1 - 1) / 1 = x.2.1 - x.1 by omega, flatMap_single] at hA2
      exact hA2
  rw [hout, Array.empty_append, ← hcov, List.map_flatMap]
  exact hA

/-- **`down`, `fromCallee`, `throughArg` and `carry` keep the gates they read, at every fuel**, wherever the callees'
summaries hold (`RetOk`), a composite's return has as many leaves as its type, a scan's output is its final carry's
leaves and then its ys (`ScanOut`) and the decoder's facts on the body do. -/
theorem specs_of (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (hB : Resolve.BodyOk c.P c.d)
    (hP1 : ∀ f : Nat, (c.P.defs[f]!).prim = true → (c.P.defs[f]!).outLeaves = 1)
    (hA : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size → ArgOk c k i)
    (hL0 : ∀ k i d L : Nat, ((c.body.nodes[k]!).args[i]!).dims = some (d, L) → L = 0 →
      i < (c.P.defs[(c.body.nodes[k]!).fn]!).body.params.size →
      ((c.P.defs[(c.body.nodes[k]!).fn]!).body.params[i]!).2.leaves = 0)
    (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn)
    (hOL : ∀ k node, c.node k = .ok node → (c.P.defs[node.fn]!).prim = false →
      (c.P.defs[node.fn]!).outLeaves = (c.P.defs[node.fn]!).body.ret.count) (hSO : ScanOut c) :
    ∀ fuel, DownSpec c sc fuel ∧ CalleeSpec c sc fuel ∧ ThroughSpec c sc fuel ∧ CarrySpec c sc fuel :=
  specs c sc hd hB hP1 hA hL0 hR
    (fun fuel hD hC => batchSpec_of c sc hd hB fuel hD hC hR hOL (batchAffine c sc hd hR hOL)
      (batchThroughMid c sc hd hB hA hR hOL fuel hD))
    (fun fuel hC hK => scanSpec_of c sc hd hB hSO fuel hC hK) (fun fuel hC hT => carrySpec_succ c sc hd fuel hR hC hT)


/-! ## The decoder's facts the core reads -/

/-- **The decoder's facts the core reads**, as `Flock.Program.getDef` gives every Definition (its private
`composite` sets them; every decoded Program has them, `BodiesOk.decoded_reads`): a primitive returns one leaf, a
composite's return has as many leaves as its type, and each node's argument is the sequence of its parts, none naming a
negative position, a batch's or scan's argument of no leaves to a member going to a parameter of none, where its
callee's body has that parameter. -/
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
    (hSO : ScanOut (ctxOf P memo d)) (sc : Flock.Program.Scope) (hd : sc.d = d) :
    ∀ fuel, DownSpec (ctxOf P memo d) sc fuel ∧ CalleeSpec (ctxOf P memo d) sc fuel ∧
      ThroughSpec (ctxOf P memo d) sc fuel ∧ CarrySpec (ctxOf P memo d) sc fuel := by
  have hfn : ∀ k node, (ctxOf P memo d).node k = .ok node → node.fn < P.defs.size := by
    intro k node hn
    have hk := node_lt hn
    have := (hB.1 k hk).1
    have hg : (P.defs[d]!).body.nodes[k]! = node := node_get hn
    rwa [hg] at this
  refine specs_of (ctxOf P memo d) sc hd hB ?_ ?_ ?_ hR ?_ hSO
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


/-! ## The summaries `memoOf` makes

`Desc.summarize` lays the segments `down` gives its return's pieces end to end (`forIn_trace2`), so a summary holds
`RetOk` wherever those segments name their `len` positions, at or past 0 and none a node's (`DownShape`), keep the gates
they read (`specs_decoded`) and the return is the sequence of its parts (`RetParts`) (`summarize_retOk`); and so every
summary `memoOf` makes of a reached Definition other than the root does, callees first (`reached_closed`,
`memoOf_retOk`). -/

/-- A loop appending what `F` gives each element: the outputs, element by element, after the start. -/
theorem forIn_trace {β : Type} (F : β → Except String (Array Seg)) :
    ∀ (l : List β) (acc out : Array Seg), forIn l acc (fun q (r : Array Seg) => do
        let v ← F q
        pure (ForInStep.yield (r ++ v))) = .ok out →
      ∃ outs : List (Array Seg), List.Forall₂ (fun q v => F q = .ok v) l outs ∧ out = outs.foldl (· ++ ·) acc
  | [], acc, out, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    exact ⟨[], .nil, h.symm⟩
  | x :: l, acc, out, h => by
    rw [List.forIn_cons] at h
    obtain ⟨y, hy, h⟩ := Walk.except_bind_ok h
    obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    subst hy
    obtain ⟨outs, h1, h2⟩ := forIn_trace F l (acc ++ v) out h
    exact ⟨v :: outs, .cons hv h1, h2⟩

theorem forall2_append {α β : Type} {R : α → β → Prop} : ∀ {l1 : List α} {m1 : List β} {l2 : List α}
    {m2 : List β}, List.Forall₂ R l1 m1 → List.Forall₂ R l2 m2 → List.Forall₂ R (l1 ++ l2) (m1 ++ m2)
  | _, _, _, _, .nil, h2 => h2
  | _, _, _, _, .cons h hrest, h2 => .cons h (forall2_append hrest h2)

/-- Two such loops nested, over a sequence's segments and each one's pieces. -/
theorem forIn_trace2 (F : Piece → Except String (Array Seg)) :
    ∀ (l : List Seg) (acc out : Array Seg), forIn l acc (fun (s : Seg) (r : Array Seg) => do
        let r' ← forIn s.ps r (fun q (r : Array Seg) => do
          let v ← F q
          pure (ForInStep.yield (r ++ v)))
        pure (ForInStep.yield r')) = .ok out →
      ∃ outs : List (Array Seg), List.Forall₂ (fun q v => F q = .ok v) (l.flatMap fun s => s.ps.toList) outs ∧
        out = outs.foldl (· ++ ·) acc
  | [], acc, out, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    exact ⟨[], .nil, h.symm⟩
  | s :: l, acc, out, h => by
    rw [List.forIn_cons] at h
    obtain ⟨y, hy, h⟩ := Walk.except_bind_ok h
    obtain ⟨r', hr', hy⟩ := Walk.except_bind_ok hy
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    subst hy
    rw [← Array.forIn_toList] at hr'
    obtain ⟨o1, h1, e1⟩ := forIn_trace F _ acc r' hr'
    obtain ⟨o2, h2, e2⟩ := forIn_trace2 F l r' out h
    refine ⟨o1 ++ o2, ?_, ?_⟩
    · rw [List.flatMap_cons]; exact forall2_append h1 h2
    · rw [e2, e1, List.foldl_append]

theorem segsPos_foldl : ∀ (outs : List (Array Seg)) (acc : Array Seg),
    segsPos (outs.foldl (· ++ ·) acc) = segsPos acc ++ outs.flatMap segsPos
  | [], acc => by simp
  | v :: outs, acc => by
    rw [List.foldl_cons, segsPos_foldl outs, segsPos_append, List.flatMap_cons, List.append_assoc]

theorem mem_foldl : ∀ (outs : List (Array Seg)) (acc : Array Seg) (g : Seg),
    g ∈ (outs.foldl (· ++ ·) acc).toList → g ∈ acc.toList ∨ ∃ v ∈ outs, g ∈ v.toList
  | [], acc, g, h => .inl h
  | v :: outs, acc, g, h => by
    rw [List.foldl_cons] at h
    rcases mem_foldl outs _ g h with h | ⟨w, hw, hg⟩
    · rw [Array.toList_append, List.mem_append] at h
      rcases h with h | h
      · exact .inl h
      · exact .inr ⟨v, List.mem_cons_self, h⟩
    · exact .inr ⟨w, List.mem_cons_of_mem _ hw, hg⟩

theorem agree_trace {P : Prog} {sc : Flock.Program.Scope} {F : Piece → Except String (Array Seg)}
    {T : Piece → List (Except String Nat)} : ∀ {qs : List Piece} {outs : List (Array Seg)},
    List.Forall₂ (fun q v => F q = .ok v) qs outs → (∀ q ∈ qs, ∀ v, F q = .ok v → Agree P sc (segsPos v) (T q)) →
    Agree P sc (outs.flatMap segsPos) (qs.flatMap T)
  | _, _, .nil, _ => List.Forall₂.nil
  | _, _, .cons (a := q) (b := v) hv hrest, hF => by
    rw [List.flatMap_cons, List.flatMap_cons]
    exact agree_append (hF q List.mem_cons_self v hv)
      (agree_trace hrest fun q' hq' => hF q' (List.mem_cons_of_mem _ hq'))

theorem forall2_mem {α β : Type} {R : α → β → Prop} : ∀ {l : List α} {m : List β}, List.Forall₂ R l m →
    ∀ b ∈ m, ∃ a ∈ l, R a b
  | _, _, .nil, _, hb => absurd hb (by simp)
  | _, _, .cons (a := a) (b := b') h hrest, b, hb => by
    rcases List.mem_cons.1 hb with rfl | hb
    · exact ⟨a, List.mem_cons_self, h⟩
    · obtain ⟨a', ha', h'⟩ := forall2_mem hrest b hb
      exact ⟨a', List.mem_cons_of_mem _ ha', h'⟩

/-- A segment of Definition `d`'s body as `down` gives it: naming its `len` positions, each at or past 0 and naming a
gate or a leaf of one of `d`'s parameters in range, its pieces each of positive count, of one copy at least. -/
def ShapeOk (P : Prog) (d : Nat) (s : Seg) : Prop :=
  SegOk s ∧ (∀ x ∈ segPos s, 0 ≤ x.2 ∧ (∀ k, x.1 ≠ .ref false k) ∧
    ∀ i, x.1 = .ref true i → i < (P.defs[d]!).body.params.size ∧
      x.2 < (((P.defs[d]!).body.params[i]!).2.leaves : Int)) ∧
  (∀ g ∈ s.ps.toList, 0 < g.count) ∧ 1 ≤ s.reps

/-- **The shape of `down`'s segments**, at every fuel: of a piece of positive count whose positions are at or past
0 and name its nodes' output leaves and its parameters' leaves in range, each segment `down` gives is `ShapeOk`
(`downShape_decoded`). -/
def DownShape (c : Desc.Ctx) : Prop :=
  ∀ fuel q segs, 0 < q.count → (∀ x ∈ pos q, 0 ≤ x.2) → (∀ x ∈ pos q, Fits c x) →
    (∀ x ∈ pos q, ∀ i, x.1 = .ref true i → i < c.body.params.size ∧ x.2 < ((c.body.params[i]!).2.leaves : Int)) →
    Desc.down c fuel q = .ok segs → ∀ s ∈ segs.toList, ShapeOk c.P c.d s

/-- One piece of positive count, its positions at or past 0, naming no node's output and its parameters' leaves in
range, as a segment. -/
theorem shapeOk_one (P : Prog) (d : Nat) (q : Piece) (h0 : 0 < q.count) (hq : ∀ x ∈ pos q, 0 ≤ x.2)
    (hk : ∀ k, q.tgt ≠ .ref false k)
    (hp : ∀ x ∈ pos q, ∀ i, x.1 = .ref true i → i < (P.defs[d]!).body.params.size ∧
      x.2 < (((P.defs[d]!).body.params[i]!).2.leaves : Int)) :
    ShapeOk P d (Seg.one q) := by
  refine ⟨segOk_one q, fun x hx => ?_, fun g hg => ?_, Nat.le_refl 1⟩
  · rw [segPos_one] at hx
    exact ⟨hq x hx, fun k e => hk k (by rw [← mem_pos_tgt hx]; exact e), hp x hx⟩
  · have e : g = q := by simpa [Seg.one] using hg
    subst e
    exact h0

/-- `down` keeps a gate's piece as it is. -/
theorem downShape_gate (c : Desc.Ctx) (fuel : Nat) (q : Piece) {w : Nat} (ht : q.tgt = .gate w)
    (h0 : 0 < q.count) (hq : ∀ x ∈ pos q, 0 ≤ x.2) {segs : Array Seg}
    (h : Desc.down c (fuel + 1) q = .ok segs) : ∀ s ∈ segs.toList, ShapeOk c.P c.d s := by
  rw [down_gate c fuel q ht] at h
  injection h with h
  subst h
  intro s hs
  have e : s = Seg.one q := by simpa using hs
  subst e
  exact shapeOk_one c.P c.d q h0 hq (fun k e => by rw [ht] at e; cases e)
    (fun x hx i e => by rw [mem_pos_tgt hx, ht] at e; cases e)

/-- `down` keeps a parameter's piece as it is. -/
theorem downShape_param (c : Desc.Ctx) (fuel : Nat) (q : Piece) {i : Nat} (ht : q.tgt = .ref true i)
    (h0 : 0 < q.count) (hq : ∀ x ∈ pos q, 0 ≤ x.2)
    (hp : ∀ x ∈ pos q, ∀ i, x.1 = .ref true i → i < c.body.params.size ∧ x.2 < ((c.body.params[i]!).2.leaves : Int))
    {segs : Array Seg} (h : Desc.down c (fuel + 1) q = .ok segs) : ∀ s ∈ segs.toList, ShapeOk c.P c.d s := by
  rw [down_param c fuel q ht] at h
  injection h with h
  subst h
  intro s hs
  have e : s = Seg.one q := by simpa using hs
  subst e
  exact shapeOk_one c.P c.d q h0 hq (fun k e => by rw [ht] at e; cases e) hp

/-- A primitive node's piece is the node's gate, once per position. -/
theorem downShape_prim (c : Desc.Ctx) (fuel : Nat) (q : Piece) {k : Nat} (ht : q.tgt = .ref false k)
    (hff : q.flat = q) {node : Flock.Program.Node} (hn : c.node k = .ok node) (hform : node.form = .prim) {w : Nat}
    (hret : (c.P.defs[node.fn]!).ret = .value w) {segs : Array Seg} (h : Desc.down c (fuel + 1) q = .ok segs) :
    ∀ s ∈ segs.toList, ShapeOk c.P c.d s := by
  rw [down_prim c fuel q ht hff hn hform hret] at h
  injection h with h
  subst h
  split
  · intro s hs
    simp at hs
  · rename_i h0
    intro s hs
    have e : s = Seg.one { tgt := .gate w, base := (c.body.offsets[k]! : Int), stride := 0, len := q.count } := by
      simpa using hs
    subst e
    refine shapeOk_one _ _ _ ?_ ?_ (fun k' e => by cases e) (fun x hx i e => by rw [mem_pos_tgt hx] at e; cases e)
    · show 0 < 1 * q.count
      omega
    · intro x hx
      unfold pos at hx
      simp only [List.mem_flatMap, List.mem_map, List.mem_range] at hx
      obtain ⟨r, -, t, -, rfl⟩ := hx
      simp

/-- A batch of a primitive's piece is its members' gates, at the piece's positions past the node's first gate. -/
theorem downShape_batchPrim (c : Desc.Ctx) (fuel : Nat) (q : Piece) {k : Nat} (ht : q.tgt = .ref false k)
    (hff : q.flat = q) {node : Flock.Program.Node} (hn : c.node k = .ok node) (hform : node.form = .batch)
    (hp : (c.P.defs[node.fn]!).prim = true) {w : Nat} (hret : (c.P.defs[node.fn]!).ret = .value w)
    (hq : ∀ x ∈ pos q, 0 ≤ x.2) {segs : Array Seg} (h : Desc.down c (fuel + 1) q = .ok segs) :
    ∀ s ∈ segs.toList, ShapeOk c.P c.d s := by
  rw [down_batchPrim c fuel q ht hff hn hform hp hret] at h
  injection h with h
  subst h
  split
  · intro s hs
    simp at hs
  · rename_i h0
    intro s hs
    have e : s = Seg.one { q with tgt := .gate w, base := (c.body.offsets[k]! : Int) + q.base } := by
      simpa using hs
    subst e
    refine shapeOk_one _ _ _ (Nat.pos_of_ne_zero h0) ?_ (fun k' e => by cases e)
      (fun x hx i e => by rw [mem_pos_tgt hx] at e; cases e)
    intro x hx
    unfold pos at hx
    simp only [List.mem_flatMap, List.mem_map, List.mem_range] at hx
    obtain ⟨r, hr, t, htl, rfl⟩ := hx
    have hy := hq (q.tgt, q.base + (r : Int) * q.rowStride + (t : Int) * q.stride) (by
      unfold pos
      simp only [List.mem_flatMap, List.mem_map, List.mem_range]
      exact ⟨r, hr, t, htl, rfl⟩)
    have ho : (0 : Int) ≤ (c.body.offsets[k]! : Int) := Int.natCast_nonneg _
    simp only at hy ⊢
    linarith

/-- Each segment a loop appending `F x`'s segments ends with is one it started with or one some `F x` gave. -/
theorem forIn_mem {β : Type} (Q : Seg → Prop) (F : β → Except String (Array Seg))
    (f : β → Array Seg → Except String (ForInStep (Array Seg)))
    (hf : ∀ x r y, f x r = .ok y → ∃ segs, F x = .ok segs ∧ y = .yield (r ++ segs)) :
    ∀ (l : List β), (∀ x ∈ l, ∀ segs, F x = .ok segs → ∀ s ∈ segs.toList, Q s) →
      ∀ (acc out : Array Seg), (∀ s ∈ acc.toList, Q s) → forIn l acc f = .ok out → ∀ s ∈ out.toList, Q s
  | [], _, acc, out, ha, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ha
  | x :: l, hF, acc, out, ha, h => by
    rw [List.forIn_cons] at h
    rcases hx : f x acc with e | y <;> rw [hx] at h
    · exact absurd h (by intro h; cases h)
    obtain ⟨segs, hs, rfl⟩ := hf x acc _ hx
    refine forIn_mem Q F f hf l (fun y hy => hF y (List.mem_cons_of_mem _ hy)) (acc ++ segs) out
      (fun s hs' => ?_) h
    rw [Array.toList_append, List.mem_append] at hs'
    rcases hs' with hs' | hs'
    · exact ha s hs'
    · exact hF x List.mem_cons_self segs hs s hs'

/-- `F`'s loop from no segments, its result returned: each segment is one some `F x` gave. -/
theorem loop_mem {β : Type} (Q : Seg → Prop) (F : β → Except String (Array Seg)) (l : List β)
    (hF : ∀ x ∈ l, ∀ segs, F x = .ok segs → ∀ s ∈ segs.toList, Q s) (out : Array Seg)
    (h : (forIn l #[] (fun x (r : Array Seg) => do
      let v ← F x
      pure (ForInStep.yield (r ++ v))) >>= fun r => pure r) = .ok out) :
    ∀ s ∈ out.toList, Q s :=
  forIn_mem Q F _ (fun x r y hy => by
    obtain ⟨segs, hs, hy⟩ := Walk.except_bind_ok hy
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    exact ⟨segs, hs, hy.symm⟩) l hF #[] out (fun s hs => by simp at hs) (ok_bind_pure h)

/-- `DownShape` at one fuel. -/
def DownShapeAt (c : Desc.Ctx) (fuel : Nat) : Prop :=
  ∀ q segs, 0 < q.count → (∀ x ∈ pos q, 0 ≤ x.2) → (∀ x ∈ pos q, Fits c x) →
    (∀ x ∈ pos q, ∀ i, x.1 = .ref true i → i < c.body.params.size ∧ x.2 < ((c.body.params[i]!).2.leaves : Int)) →
    Desc.down c fuel q = .ok segs → ∀ s ∈ segs.toList, ShapeOk c.P c.d s

/-- What `fromCallee` gives of member `m`'s return leaves `[a, b)`, within its callee's return when not empty, is
`ShapeOk`. -/
def CalleeShape (c : Desc.Ctx) (fuel : Nat) : Prop :=
  ∀ k m a b segs, m < (c.body.nodes[k]!).members → (a < b → b ≤ (c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.count) →
    Desc.fromCallee c fuel k m a b = .ok segs → ∀ s ∈ segs.toList, ShapeOk c.P c.d s

/-- What `down` gives a batch of a composite's piece of positive count, read as one row where it can be, one step up,
is `ShapeOk`. -/
def BatchShape (c : Desc.Ctx) (fuel : Nat) : Prop :=
  ∀ q segs k node, q.tgt = .ref false k → c.node k = .ok node → node.form = .batch →
    (c.P.defs[node.fn]!).prim = false → q.count ≠ 0 → q.flat = q → (∀ x ∈ pos q, 0 ≤ x.2) →
    (∀ x ∈ pos q, Fits c x) → Desc.down c (fuel + 1) q = .ok segs → ∀ s ∈ segs.toList, ShapeOk c.P c.d s

/-- What `down` gives a scan's piece of one row, one step up, is `ShapeOk`. -/
def ScanShape (c : Desc.Ctx) (fuel : Nat) : Prop :=
  ∀ q segs k node (carry y : Flock.Program.Ty), q.tgt = .ref false k → c.node k = .ok node → node.form = .scan →
    node.n ≠ 0 → q.count ≠ 0 → q.flat = q → ¬ q.rows > 1 → (c.P.defs[node.fn]!).ret = .tuple #[carry, y] →
    (∀ x ∈ pos q, 0 ≤ x.2) → (∀ x ∈ pos q, Fits c x) → Desc.down c (fuel + 1) q = .ok segs →
    ∀ s ∈ segs.toList, ShapeOk c.P c.d s

/-- A piece's rows, each through `down`. -/
theorem byRows_shape (c : Desc.Ctx) (fuel : Nat) (hD : DownShapeAt c fuel) (q : Piece) (h0 : 0 < q.count)
    (hq : ∀ x ∈ pos q, 0 ≤ x.2) (hf : ∀ x ∈ pos q, Fits c x)
    (hp : ∀ x ∈ pos q, ∀ i, x.1 = .ref true i → i < c.body.params.size ∧ x.2 < ((c.body.params[i]!).2.leaves : Int))
    {segs : Array Seg} (h : byRows c fuel q = .ok segs) : ∀ s ∈ segs.toList, ShapeOk c.P c.d s := by
  unfold byRows at h
  have h := walkAtMost_bind _ _ _ h
  dsimp only at h
  rw [← Array.forIn_toList] at h
  have hm : ∀ x ∈ q.rowsOf.toList, ∀ y ∈ pos x, y ∈ pos q := fun x hx y hy => by
    rw [← posAll_rowsOf q]
    exact List.mem_flatMap.2 ⟨x, hx, hy⟩
  have hlen : q.len ≠ 0 := fun hl => by
    unfold Desc.Piece.count at h0
    rw [hl, Nat.mul_zero] at h0
    exact absurd h0 (Nat.lt_irrefl 0)
  have hc : ∀ x ∈ q.rowsOf.toList, 0 < x.count := by
    intro x hx
    unfold Desc.Piece.rowsOf at hx
    simp only [Array.toList_map, List.mem_map] at hx
    obtain ⟨r, -, rfl⟩ := hx
    show 0 < 1 * q.len
    omega
  exact loop_mem _ (Desc.down c fuel) q.rowsOf.toList (fun x hx segs hs => hD x segs (hc x hx)
    (fun y hy => hq y (hm x hx y hy)) (fun y hy => hf y (hm x hx y hy)) (fun y hy => hp y (hm x hx y hy)) hs) segs h

/-- A Call's piece of one row and stride 1: what `fromCallee` gives. -/
theorem downShape_callOne (c : Desc.Ctx) (fuel : Nat) (hC : CalleeShape c fuel) (q : Piece) {k : Nat}
    (ht : q.tgt = .ref false k) (hff : q.flat = q) {node : Flock.Program.Node} (hn : c.node k = .ok node)
    (hform : node.form = .call) (h0 : q.count ≠ 0) (hr : ¬ q.rows > 1) (hs : q.stride = 1)
    (hq : ∀ x ∈ pos q, 0 ≤ x.2) (hf : ∀ x ∈ pos q, Fits c x) (hdo : Resolve.DownOk node (c.P.defs[node.fn]!))
    {segs : Array Seg} (h : Desc.down c (fuel + 1) q = .ok segs) : ∀ s ∈ segs.toList, ShapeOk c.P c.d s := by
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
  exact hC k 0 q.base.toNat (q.base.toNat + q.len) segs
    (by rw [node_get hn, members_call hform]; omega) (fun _ => by rw [callee_eq c hn]; omega) h

/-- A Call's piece of one row and another stride: what `fromCallee` gives at each position. -/
theorem downShape_callPts (c : Desc.Ctx) (fuel : Nat) (hC : CalleeShape c fuel) (q : Piece) {k : Nat}
    (ht : q.tgt = .ref false k) (hff : q.flat = q) {node : Flock.Program.Node} (hn : c.node k = .ok node)
    (hform : node.form = .call) (h0 : q.count ≠ 0) (hr : ¬ q.rows > 1) (hs : q.stride ≠ 1)
    (hq : ∀ x ∈ pos q, 0 ≤ x.2) (hf : ∀ x ∈ pos q, Fits c x) (hdo : Resolve.DownOk node (c.P.defs[node.fn]!))
    {segs : Array Seg} (h : Desc.down c (fuel + 1) q = .ok segs) : ∀ s ∈ segs.toList, ShapeOk c.P c.d s := by
  rw [down_call_pts c fuel q ht hff hn hform h0 hr hs] at h
  unfold callPts at h
  dsimp only at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at h
  rw [range'_size] at h
  have h1 := rows_one h0 hr
  exact loop_mem _ (fun t => Desc.fromCallee c fuel k 0 (q.at t).toNat ((q.at t).toNat + 1)) (List.range q.len)
    (fun t htm segs hs => by
      have hm := at_mem_pos h1 (List.mem_range.1 htm)
      have hb := hq _ hm
      have hfl := hf _ hm k ht
      simp only [node_get hn] at hb hfl
      have hdk := hdo (q.at t).toNat (by omega)
      simp only [hform] at hdk
      exact hC k 0 (q.at t).toNat ((q.at t).toNat + 1) segs
        (by rw [node_get hn, members_call hform]; omega) (fun _ => by rw [callee_eq c hn]; omega) hs) segs h

/-- **`down`'s shape step**: one step up, `down`'s segments are `ShapeOk` wherever `down`'s and `fromCallee`'s are and a
batch of a composite's and a scan's piece of one row give `ShapeOk` ones, in a body whose node's references
`Resolve.BodyOk` keeps in range. -/
theorem downShape_succ (c : Desc.Ctx) (hB : Resolve.BodyOk c.P c.d) (fuel : Nat) (hD : DownShapeAt c fuel)
    (hC : CalleeShape c fuel) (hBatch : BatchShape c fuel) (hScan : ScanShape c fuel) :
    DownShapeAt c (fuel + 1) := by
  intro q segs h0 hq hf hp h
  cases ht : q.tgt with
  | gate w => exact downShape_gate c fuel q ht h0 hq h
  | ref b k =>
  cases b with
  | true => exact downShape_param c fuel q ht h0 hq hp h
  | false =>
  rw [down_flat c fuel q ht] at h
  rw [← pos_flat q] at hq hf hp
  have ht' : q.flat.tgt = .ref false k := by rw [flat_tgt, ht]
  have h0' : q.flat.count ≠ 0 := by rw [flat_count]; omega
  have hff := flat_flat q
  generalize q.flat = p at ht' h0' hff hq hf hp h
  rcases hn : c.node k with e | node
  · exact absurd h (down_node c fuel p ht' h0' hn segs)
  have hdo := downOk_of hB hn
  cases hform : node.form with
  | prim =>
    cases hr : (c.P.defs[node.fn]!).ret with
    | value w => exact downShape_prim c fuel p ht' hff hn hform hr h
    | _ => exact absurd h (down_prim_ret c fuel p ht' hff hn (Or.inl hform) h0' (fun w e => by rw [hr] at e; cases e) segs)
  | call =>
    by_cases hr : p.rows > 1
    · rw [down_rows c fuel p ht' hff hn (Or.inl hform) h0' hr] at h
      exact byRows_shape c fuel hD p (Nat.pos_of_ne_zero h0') hq hf hp h
    by_cases hs : p.stride = 1
    · exact downShape_callOne c fuel hC p ht' hff hn hform h0' hr hs hq hf hdo h
    · exact downShape_callPts c fuel hC p ht' hff hn hform h0' hr hs hq hf hdo h
  | scan =>
    by_cases hn0 : node.n = 0
    · exact absurd h (down_scan0 c fuel p ht' hff hn hform hn0 h0' segs)
    by_cases hr : p.rows > 1
    · rw [down_rows c fuel p ht' hff hn (Or.inr ⟨hform, hn0⟩) h0' hr] at h
      exact byRows_shape c fuel hD p (Nat.pos_of_ne_zero h0') hq hf hp h
    by_cases hex : ∃ carry y, (c.P.defs[node.fn]!).ret = .tuple #[carry, y]
    · obtain ⟨carry, y, hret⟩ := hex
      exact hScan p segs k node carry y ht' hn hform hn0 h0' hff hr hret hq hf h
    · exact absurd h
        (down_scan_ret c fuel p ht' hff hn hform hn0 h0' hr (fun carry y e => hex ⟨carry, y, e⟩) segs)
  | batch =>
    by_cases hpr : (c.P.defs[node.fn]!).prim = true
    · cases hr : (c.P.defs[node.fn]!).ret with
      | value w => exact downShape_batchPrim c fuel p ht' hff hn hform hpr hr hq h
      | _ =>
        exact absurd h
          (down_prim_ret c fuel p ht' hff hn (Or.inr ⟨hform, hpr⟩) h0' (fun w e => by rw [hr] at e; cases e) segs)
    · exact hBatch p segs k node ht' hn hform (by simpa using hpr) h0' hff hq hf h

/-- A loop each of whose steps appends segments `Q` holds of ends with what it began with and such segments. -/
theorem forIn_app {β : Type} (Q : Seg → Prop) (f : β → Array Seg → Except String (ForInStep (Array Seg))) :
    ∀ (l : List β), (∀ x ∈ l, ∀ r y, f x r = .ok y → ∃ segs, y = .yield (r ++ segs) ∧ ∀ s ∈ segs.toList, Q s) →
      ∀ (acc out : Array Seg), forIn l acc f = .ok out → ∃ segs, out = acc ++ segs ∧ ∀ s ∈ segs.toList, Q s
  | [], _, acc, out, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    exact ⟨#[], by simp [h], fun s hs => by simp at hs⟩
  | x :: l, hF, acc, out, h => by
    rw [List.forIn_cons] at h
    rcases hx : f x acc with e | y <;> rw [hx] at h
    · exact absurd h (by intro h; cases h)
    obtain ⟨segs, rfl, hs⟩ := hF x List.mem_cons_self acc _ hx
    obtain ⟨segs', h1, h2⟩ := forIn_app Q f l (fun y hy => hF y (List.mem_cons_of_mem _ hy)) (acc ++ segs) out h
    refine ⟨segs ++ segs', by rw [h1, Array.append_assoc], fun s hm => ?_⟩
    rw [Array.toList_append, List.mem_append] at hm
    exact hm.elim (hs s) (h2 s)

/-- **An argument's references `[lo, hi)`, read through `down`**, give `ShapeOk` segments, of an argument whose pieces
are of positive count. -/
theorem one_shape (c : Desc.Ctx) (fuel : Nat) (hD : DownShapeAt c fuel) {k i : Nat} (hA : ArgOk c k i)
    (hP : ∀ s ∈ (c.arg k i).segs.toList, ∀ q ∈ s.ps.toList, 0 < q.count)
    (hR : ∀ t < ((c.body.nodes[k]!).args[i]!).seq.count,
      Resolve.RefOk c.P c.d k (((c.body.nodes[k]!).args[i]!).seq.get t))
    {lo hi : Nat} (hhi : lo < hi → hi ≤ ((c.body.nodes[k]!).args[i]!).seq.count) {out : Array Seg}
    (h : oneOf c fuel k i lo hi = .ok out) : ∀ s ∈ out.toList, ShapeOk c.P c.d s := by
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
  have hcnt : ∀ q ∈ ((c.arg k i).pieces lo hi).toList, 0 < q.count := by
    intro q hq
    unfold Desc.PS.pieces at hq
    rw [Array.toList_flatMap, List.mem_flatMap] at hq
    obtain ⟨g, hg, hq⟩ := hq
    rw [Array.toList_flatMap, List.mem_flatMap] at hq
    obtain ⟨m, -, hq⟩ := hq
    exact shiftGates_count _ _ (psSlice_ok _ hA.1 hP lo hi g hg).2.2 q hq
  exact loop_mem _ (Desc.down c fuel) _ (fun q hq segs hs => hD q segs (hcnt q hq)
    (fun x hx => by obtain ⟨t, -, rfl⟩ := hmem q hq x hx; simp [refPos])
    (fun x hx k' hk' => by
      obtain ⟨t, ht, rfl⟩ := hmem q hq x hx
      simp only [refPos, Tgt.ref.injEq] at hk'
      obtain ⟨hp', rfl⟩ := hk'
      have := ((hR t ht).2 hp').2
      simp only [refPos]
      exact_mod_cast this)
    (fun x hx i' hi' => by
      obtain ⟨t, ht, rfl⟩ := hmem q hq x hx
      simp only [refPos, Tgt.ref.injEq] at hi'
      obtain ⟨hp', rfl⟩ := hi'
      have := (hR t ht).1 hp'
      simp only [refPos]
      exact ⟨this.1, by exact_mod_cast this.2⟩) hs) out h

/-- A gate piece of positive count, its positions at or past 0, offset by `o ≥ 0`, as a segment. -/
theorem shapeOk_gate (P : Prog) (d : Nat) {o : Int} (ho : 0 ≤ o) (g : Piece) {w : Nat} (hw : g.tgt = .gate w)
    (h0 : 0 < g.count) (hq : ∀ x ∈ pos g, 0 ≤ x.2) : ShapeOk P d (Seg.one { g with base := o + g.base }) := by
  rcases g with ⟨t, base, stride, len, rows, rowStride⟩
  dsimp only at hw
  subst hw
  have e := pos_shiftGate o ⟨.gate w, base, stride, len, rows, rowStride⟩
  dsimp only at e
  have e' : (⟨.gate w, o + base, stride, len, rows, rowStride⟩ : Piece) =
      ⟨.gate w, base + o, stride, len, rows, rowStride⟩ := by rw [Int.add_comm]
  refine shapeOk_one P d _ h0 (fun x hx => ?_) (fun k hk => by cases hk) (fun x hx i hi => ?_)
  · dsimp only at hx
    rw [e', e] at hx
    obtain ⟨y, hy, rfl⟩ := List.mem_map.1 hx
    have hy0 := hq y hy
    obtain ⟨t, j⟩ := y
    have hyt : t = .gate w := mem_pos_tgt hy
    subst hyt
    show 0 ≤ j + o
    dsimp only at hy0
    linarith
  · have := (mem_pos_tgt hx).symm.trans hi
    cases this

/-- What `throughArg` gives of member `m`'s parameter `i`'s leaves `g`, each in range, is `ShapeOk`. -/
def ThroughShape (c : Desc.Ctx) (fuel : Nat) : Prop :=
  ∀ k m i g segs, m < (c.body.nodes[k]!).members → g.tgt = .ref true i →
    (∀ x ∈ pos g, 0 ≤ x.2 ∧ i < (c.P.defs[(c.body.nodes[k]!).fn]!).body.params.size ∧
      x.2 < ((((c.P.defs[(c.body.nodes[k]!).fn]!).body.params[i]!).2.leaves : Int))) →
    Desc.throughArg c fuel k m i g = .ok segs → ∀ s ∈ segs.toList, ShapeOk c.P c.d s

/-- What `carry` gives of scan iteration `m`'s return leaves `[a, b)`, within its step's return when not empty, is
`ShapeOk`. -/
def CarryShape (c : Desc.Ctx) (fuel : Nat) : Prop :=
  ∀ k m a b segs, (c.body.nodes[k]!).form = .scan → m < (c.body.nodes[k]!).members →
    (a < b → b ≤ (c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.count) →
    Desc.carry c fuel k m a b = .ok segs → ∀ s ∈ segs.toList, ShapeOk c.P c.d s

/-- **`throughArg`'s shape step**: one step up, what member `m`'s parameter leaves read through node `k`'s argument `i`
give is `ShapeOk` wherever `down`'s and `carry`'s segments are, of arguments whose pieces are of positive count. -/
theorem throughShape_succ (c : Desc.Ctx) (hB : Resolve.BodyOk c.P c.d)
    (hA : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size → ArgOk c k i)
    (hAP : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size →
      ∀ s ∈ (c.arg k i).segs.toList, ∀ q ∈ s.ps.toList, 0 < q.count)
    (hL0 : ∀ k i d L : Nat, ((c.body.nodes[k]!).args[i]!).dims = some (d, L) → L = 0 →
      i < (c.P.defs[(c.body.nodes[k]!).fn]!).body.params.size →
      ((c.P.defs[(c.body.nodes[k]!).fn]!).body.params[i]!).2.leaves = 0)
    (fuel : Nat) (hD : DownShapeAt c fuel) (hK : CarryShape c fuel) : ThroughShape c (fuel + 1) := by
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
  have hUp := upOk_of hB hn
  have hAk : ArgOk c k i := hA k i hk (by rw [hnode]; exact hia)
  have hPk := hAP k i hk (by rw [hnode]; exact hia)
  have hRk : ∀ t < ((c.body.nodes[k]!).args[i]!).seq.count,
      Resolve.RefOk c.P c.d k (((c.body.nodes[k]!).args[i]!).seq.get t) := (hB.1 k hk).2.1 i
  have hone : ∀ lo hi v, (lo < hi → hi ≤ a.seq.count) → oneOf c fuel k i lo hi = .ok v →
      ∀ s ∈ v.toList, ShapeOk c.P c.d s := fun lo hi v hb hv =>
    one_shape c fuel hD hAk hPk hRk (by rw [hnode, hai']; exact hb) hv
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
  have hfa := forIn_app (ShapeOk c.P c.d) _ _ ?_ #[] segs h'
  · obtain ⟨segs', rfl, hfin⟩ := hfa
    intro s hs
    rw [Array.toList_append, List.mem_append] at hs
    exact hs.elim (fun hs => by simp at hs) (hfin s)
  intro x hxm r y hy
  obtain ⟨lo, hi⟩ := x
  have hjx := hj (lo, hi) hxm
  rcases hf : node.form with _ | _ | _ | _
  case prim | call =>
    simp only [hf] at hy
    obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    refine ⟨v, hy.symm, hone lo hi v (fun hlt => ?_) hv⟩
    obtain ⟨hi0, hjl⟩ := hjx (hi - 1) (by omega) (by omega)
    have := hUp i hi0 (hi - 1) hjl m hm
    simp only [hf, hai'] at this
    omega
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
    have hfa2 := forIn_app (ShapeOk c.P c.d) _ _ ?_ r w hw
    · obtain ⟨segs2, rfl, hag2⟩ := hfa2
      exact ⟨segs2, hy.symm, hag2⟩
    intro x hxm2 r2 y2 hy2
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
        refine ⟨v, hy.symm, hone lo hi v (fun hlt => ?_) hv⟩
        obtain ⟨hi0, hjl⟩ := hjx (hi - 1) (by omega) (by omega)
        have := hUp 0 hi0 (hi - 1) hjl 0 hm
        simp only [hf, hai', ↓reduceIte] at this
        omega
      · rw [if_neg (show ¬ ((m == 0) = true) by simpa using hm0)] at hy
        obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
        simp only [pure, Except.pure, Except.ok.injEq] at hy
        have hm1 : m - 1 < (c.body.nodes[k]!).members := by rw [hnode]; omega
        refine ⟨v, hy.symm, hK k (m - 1) lo hi v (by rw [hnode]; exact hf) hm1 (fun hlt => ?_) hv⟩
        obtain ⟨hi0, hjl⟩ := hjx (hi - 1) (by omega) (by omega)
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
        refine ⟨v, hy.symm, hone (m * L + lo) (m * L + hi) v (fun hlt => ?_) hv⟩
        obtain ⟨hi0', hjl⟩ := hjx (hi - 1) (by omega) (by omega)
        have := hUp i hi0' (hi - 1) hjl m hm
        simp only [hf, hi0, hx0, hai', hL, ↓reduceIte] at this
        omega
      · rw [if_neg hx0] at hy
        obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
        simp only [pure, Except.pure, Except.ok.injEq] at hy
        refine ⟨v, hy.symm, hone lo hi v (fun hlt => ?_) hv⟩
        obtain ⟨hi0', hjl⟩ := hjx (hi - 1) (by omega) (by omega)
        have := hUp i hi0' (hi - 1) hjl m hm
        simp only [hf, hi0, hx0, hai', ↓reduceIte] at this
        omega

/-- **`fromCallee`'s shape step**: one step up, `fromCallee`'s segments are `ShapeOk` wherever its callees' summaries
hold `RetOk` and `throughArg`'s segments one step down are `ShapeOk`, keep what they read (`ThroughSpec`, which gives
copies all alike their length) and are each of a copy at least (`ThroughReps`). -/
theorem calleeShape_succ (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (fuel : Nat)
    (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn) (hTS : ThroughShape c fuel)
    (hT : ThroughSpec c sc fuel) (hTR : ThroughReps c) : CalleeShape c (fuel + 1) := by
  intro k m a b segs hm hb h
  rw [Desc.fromCallee.eq_2] at h
  obtain ⟨node, hn, h⟩ := Walk.except_bind_ok h
  dsimp only at h
  have h' := ok_bind_pure h
  rw [← Array.forIn_toList] at h'
  obtain ⟨hps, hpos, hgo, -, hpp⟩ := hR k node hn
  have hnode : c.body.nodes[k]! = node := node_get hn
  rw [hnode] at hm hb
  have hsc := child_eq c sc hd hn m
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
  have hsok := psSlice_ok _ hps hpp a b
  have hfa := forIn_app (ShapeOk c.P c.d) _ _ ?_ #[] segs h'
  · obtain ⟨out, rfl, hout⟩ := hfa
    intro s hs
    rw [Array.toList_append, List.mem_append] at hs
    exact hs.elim (fun hs => by simp at hs) (hout s)
  intro s hs r y hy
  obtain ⟨hrep1, hgates⟩ := retOk_slice hgo hs
  have hx := hmemS s hs
  have hsg : ∀ g ∈ s.ps.toList, 0 < g.count := (hsok s hs).2.2
  split at hy
  · rename_i hr
    split at hy
    · rename_i hall
      have hgs := gates_of_all hall
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      subst hy
      refine ⟨#[_], congrArg _ Array.push_eq_append, fun s' hs' => ?_⟩
      simp only [List.mem_singleton] at hs'
      subst hs'
      refine ⟨?_, fun x hxm => ?_, fun g hg => shiftGates_count _ _ hsg g hg, hrep1⟩
      · show (posAll (Desc.shiftGates _ s.ps)).length = s.len
        rw [shiftGates_pos, List.length_map]
        exact (hsok s hs).1
      · rw [segPos_shift] at hxm
        obtain ⟨y, hy, rfl⟩ := List.mem_map.1 hxm
        obtain ⟨g, hg, e⟩ := segPos_tgt hy
        obtain ⟨w, hw⟩ := hgs g hg
        have hy0 := (hpos y (hx y hy)).1
        obtain ⟨t, j⟩ := y
        have hyt : t = .gate w := e.trans hw
        subst hyt
        dsimp only at hy0
        refine ⟨show 0 ≤ j + _ by linarith, ⟨fun k' hk' => ?_, fun i hi => ?_⟩⟩
        · simp [shiftPos] at hk'
        · simp [shiftPos] at hi
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
    obtain ⟨one'', h1'', h2''⟩ := forIn_app (ShapeOk c.P c.d) _ s.ps.toList (fun g hg r' y' hy' => by
        split at hy'
        · rename_i w hw
          simp only [pure, Except.pure, Except.ok.injEq] at hy'
          subst hy'
          refine ⟨#[_], congrArg _ Array.push_eq_append, fun s' hs' => ?_⟩
          simp only [List.mem_singleton] at hs'
          subst hs'
          exact shapeOk_gate c.P c.d ho g hw (hsg g hg) (fun x hxm => (hpos x (hxg g hg x hxm)).1)
        · exact absurd hy' (by intro h; cases h)
        · rename_i i hi
          obtain ⟨v', hv', hy'⟩ := Walk.except_bind_ok hy'
          simp only [pure, Except.pure, Except.ok.injEq] at hy'
          subst hy'
          refine ⟨v', rfl, ?_⟩
          refine hTS k m i g v' (by rw [hnode]; exact hm) hi (fun x hxm => ?_) hv'
          obtain ⟨h0, -, hp⟩ := hpos x (hxg g hg x hxm)
          obtain ⟨h1, h2⟩ := hp i ((mem_pos_tgt hxm).trans hi)
          rw [hnode]
          exact ⟨h0, h1, h2⟩) #[] one hone
    have hone_shape : ∀ s' ∈ one.toList, ShapeOk c.P c.d s' := by
      rw [h1'', Array.empty_append]; exact h2''
    split at hy
    · rename_i hall
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
      have hA1 : Agree c.P sc (segsPos one) ((posAll s.ps).map (den c.P (c.P.child sc k m))) := by
        rw [hone', posAll, List.map_flatMap]
        exact hA'
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      subst hy
      refine ⟨#[_], congrArg _ Array.push_eq_append, fun s' hs' => ?_⟩
      simp only [List.mem_singleton] at hs'
      subst hs'
      have h1 : ∀ x ∈ one.toList, x.reps = 1 := by
        intro x hx'
        have hle := (Array.all_eq_true_iff_forall_mem.1 hall) x (Array.mem_toList_iff.1 hx')
        simp only [decide_eq_true_eq] at hle
        rcases hsrc x hx' with h0 | h0
        · simp at h0
        · omega
      have hpa := posAll_flatMap_ps one h1
      refine ⟨?_, fun x hx' => ?_, fun g hg => ?_, hrep1⟩
      · show (posAll (one.flatMap (·.ps))).length = s.len
        rw [hpa, hA1.length_eq, List.length_map]
        exact (hsok s hs).1
      · rw [segPos_alike rfl] at hx'
        obtain ⟨_, -, hx'⟩ := List.mem_flatMap.1 hx'
        rw [hpa] at hx'
        obtain ⟨s', hs', hx''⟩ := List.mem_flatMap.1 hx'
        exact (hone_shape s' hs').2.1 x hx''
      · rw [Array.toList_flatMap, List.mem_flatMap] at hg
        obtain ⟨s', hs', hg⟩ := hg
        exact (hone_shape s' hs').2.2.1 g hg
    · have hy := walkAtMost_bind _ _ _ hy
      obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      subst hy
      simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at hv
      obtain ⟨segs', hsegs', hA2⟩ := forIn_app (ShapeOk c.P c.d) _ _ (fun _ _ r' y' hy' => by
          simp only [pure, Except.pure, Except.ok.injEq] at hy'
          exact ⟨one, hy'.symm, hone_shape⟩) r v hv
      exact ⟨segs', by rw [hsegs'], hA2⟩
  · rename_i hr
    have h1 : s.reps = 1 := by omega
    obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    subst hy
    rw [← Array.forIn_toList] at hv
    have hxg : ∀ g ∈ s.ps.toList, ∀ x ∈ pos g, x ∈ psPos (c.memo[node.fn]!).ret := fun g hg x hxm =>
      hx x (by rw [segPos_one_rep h1]; exact List.mem_flatMap.2 ⟨g, hg, hxm⟩)
    obtain ⟨segs', rfl, hA'⟩ := forIn_app (ShapeOk c.P c.d) _ s.ps.toList (fun g hg r' y' hy' => by
        split at hy'
        · rename_i w hw
          simp only [pure, Except.pure, Except.ok.injEq] at hy'
          subst hy'
          refine ⟨#[_], congrArg _ Array.push_eq_append, fun s' hs' => ?_⟩
          simp only [List.mem_singleton] at hs'
          subst hs'
          exact shapeOk_gate c.P c.d ho g hw (hsg g hg) (fun x hxm => (hpos x (hxg g hg x hxm)).1)
        · exact absurd hy' (by intro h; cases h)
        · rename_i i hi
          obtain ⟨v', hv', hy'⟩ := Walk.except_bind_ok hy'
          simp only [pure, Except.pure, Except.ok.injEq] at hy'
          subst hy'
          refine ⟨v', rfl, ?_⟩
          refine hTS k m i g v' (by rw [hnode]; exact hm) hi (fun x hxm => ?_) hv'
          obtain ⟨h0, -, hp⟩ := hpos x (hxg g hg x hxm)
          obtain ⟨h1, h2⟩ := hp i ((mem_pos_tgt hxm).trans hi)
          rw [hnode]
          exact ⟨h0, h1, h2⟩) r v hv
    exact ⟨segs', rfl, hA'⟩

/-- A scan's initial carry's leaves `[lo, hi)`, each in range, through `throughArg`: `ShapeOk` segments. -/
theorem init_shape (c : Desc.Ctx) {fuel k : Nat} {node : Flock.Program.Node} (hn : c.node k = .ok node)
    (hTS : ThroughShape c fuel) (hm0 : 0 < node.members) {lo hi : Nat}
    (hlv : ∀ j, lo ≤ j → j < hi → 0 < (c.P.defs[node.fn]!).body.params.size ∧
      j < ((c.P.defs[node.fn]!).body.params[0]!).2.leaves) {segs : Array Seg}
    (h : Desc.throughArg c fuel k 0 0 { tgt := .ref true 0, base := lo, stride := 1, len := hi - lo } = .ok segs) :
    ∀ s ∈ segs.toList, ShapeOk c.P c.d s := by
  have hnode := node_get hn
  exact hTS k 0 0 _ segs (by rw [hnode]; exact hm0) rfl (fun x hx => by
    rw [pos_carry, List.mem_map] at hx
    obtain ⟨j, hj, rfl⟩ := hx
    rw [List.mem_range'_1] at hj
    rw [hnode]
    have := hlv j hj.1 (by omega)
    exact ⟨by simp, this.1, by show (j : Int) < _; exact_mod_cast this.2⟩) h

/-- **`carry`'s shape step**: one step up, `carry`'s segments are `ShapeOk` wherever the step's summary holds `RetOk`
and `fromCallee`'s and `throughArg`'s segments one step down are `ShapeOk`. -/
theorem carryShape_succ (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (fuel : Nat)
    (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn) (hCS : CalleeShape c fuel)
    (hTS : ThroughShape c fuel) : CarryShape c (fuel + 1) := by
  intro k m a b segs hform hm hb h
  rw [Desc.carry.eq_2] at h
  obtain ⟨node, hn, h⟩ := Walk.except_bind_ok h
  dsimp only at h
  have h' := ok_bind_pure h
  rw [← Array.forIn_toList] at h'
  have hnode : c.body.nodes[k]! = node := node_get hn
  rw [hnode] at hform hm hb
  have hRk := hR k node hn
  obtain ⟨hps, hpos, -, hag, hpp⟩ := hRk
  have hlenR : (psPos (c.memo[node.fn]!).ret).length = (c.P.defs[node.fn]!).body.ret.count := by
    have := (hag (c.P.child sc k 0) (by rw [child_eq c sc hd hn 0]; rfl)).length_eq
    simpa using this
  obtain ⟨hcov, hid⟩ := carryRuns_cover _ hps hpp a b (fun hab => by rw [hlenR]; exact hb hab)
  have hm0 : 0 < node.members := by omega
  have hcarry : ∀ h j' : Nat, (psPos (c.memo[node.fn]!).ret)[h]? = some (.ref true 0, (j' : Int)) →
      0 < (c.P.defs[node.fn]!).body.params.size ∧ j' < ((c.P.defs[node.fn]!).body.params[0]!).2.leaves := by
    intro h j' hx
    have := (hpos _ (List.mem_of_getElem? hx)).2.2 0 rfl
    exact ⟨this.1, by have h2 : ((j' : Nat) : Int) < _ := this.2; exact_mod_cast h2⟩
  have hfa := forIn_app (ShapeOk c.P c.d) _ _ ?_ #[] segs h'
  · obtain ⟨out, rfl, hout⟩ := hfa
    intro s hs
    rw [Array.toList_append, List.mem_append] at hs
    exact hs.elim (fun hs => by simp at hs) (hout s)
  intro x hx r' y hy
  split at hy
  · rename_i hxid
    obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    subst hy
    exact ⟨v, rfl, init_shape c hn hTS hm0 (fun j h1 h2 => hcarry j j (hid x hx hxid j h1 h2)) hv⟩
  · have hy := walkAtMost_bind _ _ _ hy
    obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    subst hy
    simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at hv
    obtain ⟨segs', hsegs', hA2⟩ := forIn_app (ShapeOk c.P c.d) _ _ (fun j _ r'' y' hy' => by
        obtain ⟨ch, hch, hy'⟩ := Walk.except_bind_ok hy'
        have hok := chainOf_ok _ j ch hch
        split at hy'
        · rename_i hcond
          simp only [Bool.and_eq_true, decide_eq_true_eq, Option.isSome_iff_exists] at hcond
          obtain ⟨⟨q0, hq0⟩, he⟩ := hcond
          obtain ⟨v', hv', hy'⟩ := Walk.except_bind_ok hy'
          simp only [pure, Except.pure, Except.ok.injEq] at hy'
          subst hy'
          refine ⟨v', rfl, ?_⟩
          obtain ⟨hsz, h0, hsc, hex, -⟩ := hok
          obtain ⟨hq, -⟩ := hex q0 hq0
          have hlt : ch.hist[ch.hist.size - 1]! < (c.P.defs[node.fn]!).body.ret.count := by
            have := retAt_pos _ hps hpp hq
            rw [← hlenR]; exact (List.getElem?_eq_some_iff.1 this).1
          exact hCS k (m - (ch.hist.size - 1)) _ _ v' (by rw [hnode]; omega) (by rw [hnode]; intro; omega) hv'
        · rename_i hcond
          obtain ⟨v', hv', hy'⟩ := Walk.except_bind_ok hy'
          simp only [pure, Except.pure, Except.ok.injEq] at hy'
          subst hy'
          refine ⟨v', rfl, ?_⟩
          have hs : ∀ s ≤ m, (psPos (c.memo[node.fn]!).ret)[ch.at s]? = some (.ref true 0, (ch.at (s + 1) : Int)) := by
            intro s hs
            rcases hex : ch.exit with _ | q0
            · obtain ⟨q, hq, ht, hb⟩ := chainAt_cyc hok hex s
              exact carryPos_of_retAt (hR k node hn) hq ht hb
            · have he : ¬ ch.hist.size - 1 ≤ m := fun he => hcond (by simp [hex, he])
              obtain ⟨-, -, hsc, -, -⟩ := hok
              obtain ⟨q, hq, ht, hb⟩ := hsc s (by omega)
              rw [chainAt_lt ch (show s < ch.hist.size by omega), chainAt_lt ch (show s + 1 < ch.hist.size by omega)]
              exact carryPos_of_retAt (hR k node hn) hq ht hb
          exact init_shape c hn hTS hm0 (lo := ch.at (m + 1)) (hi := ch.at (m + 1) + 1)
            (fun j' h1 h2 => by rw [show j' = ch.at (m + 1) by omega]; exact hcarry _ _ (hs m le_rfl)) hv') r' v hv
    exact ⟨segs', by rw [hsegs'], hA2⟩

/-- **`down`, `fromCallee`, `throughArg` and `carry` give `ShapeOk` segments, at every fuel**, wherever `throughArg`
keeps what it reads (`specs`), a batch of a composite's piece and a scan's piece of one row give `ShapeOk` segments one
step up from them, and the arguments' pieces are of positive count. -/
theorem shapes (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (hB : Resolve.BodyOk c.P c.d)
    (hA : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size → ArgOk c k i)
    (hAP : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size →
      ∀ s ∈ (c.arg k i).segs.toList, ∀ q ∈ s.ps.toList, 0 < q.count)
    (hL0 : ∀ k i d L : Nat, ((c.body.nodes[k]!).args[i]!).dims = some (d, L) → L = 0 →
      i < (c.P.defs[(c.body.nodes[k]!).fn]!).body.params.size →
      ((c.P.defs[(c.body.nodes[k]!).fn]!).body.params[i]!).2.leaves = 0)
    (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn) (hT : ∀ fuel, ThroughSpec c sc fuel)
    (hBatch : ∀ fuel, DownShapeAt c fuel → CalleeShape c fuel → BatchShape c fuel)
    (hScan : ∀ fuel, CalleeShape c fuel → CarryShape c fuel → ScanShape c fuel) :
    ∀ fuel, DownShapeAt c fuel ∧ CalleeShape c fuel ∧ ThroughShape c fuel ∧ CarryShape c fuel
  | 0 => by
    refine ⟨fun q segs _ _ _ _ h => absurd h (down_zero c q segs), ?_, ?_, ?_⟩
    · intro _ _ _ _ _ _ _ h; rw [Desc.fromCallee.eq_1] at h; cases h
    · intro _ _ _ _ _ _ _ _ h; rw [Desc.throughArg.eq_1] at h; cases h
    · intro _ _ _ _ _ _ _ _ h; rw [Desc.carry.eq_1] at h; cases h
  | fuel + 1 =>
    have ⟨hD, hC, hTS, hK⟩ := shapes c sc hd hB hA hAP hL0 hR hT hBatch hScan fuel
    ⟨downShape_succ c hB fuel hD hC (hBatch fuel hD hC) (hScan fuel hC hK),
      calleeShape_succ c sc hd fuel hR hTS (hT fuel) (throughReps c), throughShape_succ c hB hA hAP hL0 fuel hD hK,
      carryShape_succ c sc hd fuel hR hC hTS⟩

/-- **`DownShape`** from `shapes`. -/
theorem downShape_of (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (hB : Resolve.BodyOk c.P c.d)
    (hA : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size → ArgOk c k i)
    (hAP : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size →
      ∀ s ∈ (c.arg k i).segs.toList, ∀ q ∈ s.ps.toList, 0 < q.count)
    (hL0 : ∀ k i d L : Nat, ((c.body.nodes[k]!).args[i]!).dims = some (d, L) → L = 0 →
      i < (c.P.defs[(c.body.nodes[k]!).fn]!).body.params.size →
      ((c.P.defs[(c.body.nodes[k]!).fn]!).body.params[i]!).2.leaves = 0)
    (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn) (hT : ∀ fuel, ThroughSpec c sc fuel)
    (hBatch : ∀ fuel, DownShapeAt c fuel → CalleeShape c fuel → BatchShape c fuel)
    (hScan : ∀ fuel, CalleeShape c fuel → CarryShape c fuel → ScanShape c fuel) : DownShape c :=
  fun fuel => (shapes c sc hd hB hA hAP hL0 hR hT hBatch hScan fuel).1

/-- A batch's piece of one row and any other stride, position by position through `down`: `ShapeOk` segments. -/
theorem batchPts_shape (c : Desc.Ctx) (fuel : Nat) (hD : DownShapeAt c fuel) (q : Piece) {k : Nat}
    (ht : q.tgt = .ref false k) (h1 : q.rows = 1) (hq : ∀ x ∈ pos q, 0 ≤ x.2) (hf : ∀ x ∈ pos q, Fits c x)
    {segs : Array Seg} (h : batchPts c fuel q = .ok segs) : ∀ s ∈ segs.toList, ShapeOk c.P c.d s := by
  unfold batchPts at h
  have h := walkAtMost_bind _ _ _ h
  dsimp only at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at h
  rw [range'_size] at h
  have hpt : ∀ t, pos { q with base := q.at t, stride := 1, len := 1 } = [(q.tgt, q.at t)] := by
    intro t
    simp [pos, h1]
  exact loop_mem _ (fun t => Desc.down c fuel { q with base := q.at t, stride := 1, len := 1 }) (List.range q.len)
    (fun t htm segs hs => by
      have hm := at_mem_pos h1 (List.mem_range.1 htm)
      exact hD _ segs (by simp [Desc.Piece.count, h1])
        (by rw [hpt t]; intro x hx; simp only [List.mem_singleton] at hx; subst hx; exact hq _ hm)
        (by rw [hpt t]; intro x hx; simp only [List.mem_singleton] at hx; subst hx; exact hf _ hm)
        (by rw [hpt t]; intro x hx i hi; simp only [List.mem_singleton] at hx; subst hx; rw [ht] at hi; cases hi)
        hs) segs h

/-- What `batchOne` gives a batch of a composite's piece of one row and stride 1, past its one-run case, is
`ShapeOk`. -/
def BatchOneShape (c : Desc.Ctx) (fuel : Nat) : Prop :=
  ∀ q k node segs, q.tgt = .ref false k → c.node k = .ok node → node.form = .batch →
    (c.P.defs[node.fn]!).prim = false → (c.P.defs[node.fn]!).outLeaves ≠ 0 → q.count ≠ 0 → q.rows = 1 →
    q.stride = 1 → (∀ x ∈ pos q, 0 ≤ x.2) → (∀ x ∈ pos q, Fits c x) → batchOne c fuel k node q = .ok segs →
    ∀ s ∈ segs.toList, ShapeOk c.P c.d s

/-- **`BatchShape` from its cases**: a batch of a composite's piece row by row and position by position through
`down` (`byRows_shape`, `batchPts_shape`), and in its one-run, stride-1 and whole-member-stride cases wherever those
give `ShapeOk` segments. -/
theorem batchShape_of (c : Desc.Ctx) (fuel : Nat) (hD : DownShapeAt c fuel)
    (hAff : ∀ q k node p, q.tgt = .ref false k → c.node k = .ok node → node.form = .batch →
      (c.P.defs[node.fn]!).prim = false → (c.P.defs[node.fn]!).outLeaves ≠ 0 → q.count ≠ 0 →
      (∀ x ∈ pos q, 0 ≤ x.2) → (∀ x ∈ pos q, Fits c x) → affineOf c k node q = some p → ShapeOk c.P c.d (Seg.one p))
    (hOne : BatchOneShape c fuel)
    (hStride : ∀ q k node segs, q.tgt = .ref false k → c.node k = .ok node → node.form = .batch →
      (c.P.defs[node.fn]!).prim = false → (c.P.defs[node.fn]!).outLeaves ≠ 0 → q.rows = 1 →
      (q.len > 1 ∧ q.stride % ((c.P.defs[node.fn]!).outLeaves : Int) = 0 ∧ q.base ≥ 0) →
      (∀ x ∈ pos q, 0 ≤ x.2) → (∀ x ∈ pos q, Fits c x) → batchStride c fuel k node q = .ok segs →
      ∀ s ∈ segs.toList, ShapeOk c.P c.d s) :
    BatchShape c fuel := by
  intro q segs k node ht hn hform hp h0 hff hq hf h
  by_cases hL : (c.P.defs[node.fn]!).outLeaves = 0
  · exact absurd h (down_batch0 c fuel q ht hff hn hform hp hL h0 segs)
  rcases ha : affineOf c k node q with _ | p
  · by_cases hr : q.rows > 1
    · rw [down_batch_rows c fuel q ht hff hn hform hp hL h0 ha hr] at h
      exact byRows_shape c fuel hD q (Nat.pos_of_ne_zero h0) hq hf
        (fun x hx i hi => by rw [mem_pos_tgt hx, ht] at hi; cases hi) h
    by_cases hs : q.stride = 1
    · rw [down_batch_one c fuel q ht hff hn hform hp hL h0 ha hr hs] at h
      exact hOne q k node segs ht hn hform hp hL h0 (rows_one h0 hr) hs hq hf h
    by_cases hcp : q.len > 1 ∧ q.stride % ((c.P.defs[node.fn]!).outLeaves : Int) = 0 ∧ q.base ≥ 0
    · rw [down_batch_stride c fuel q ht hff hn hform hp hL h0 ha hr hs hcp] at h
      exact hStride q k node segs ht hn hform hp hL (rows_one h0 hr) hcp hq hf h
    rw [down_batch_pts c fuel q ht hff hn hform hp hL h0 ha hr hs hcp] at h
    exact batchPts_shape c fuel hD q ht (rows_one h0 hr) hq hf h
  · rw [down_batch_affine c fuel q ht hff hn hform hp hL h0 ha] at h
    simp only [Except.ok.injEq] at h
    subst h
    intro s hs
    simp only [List.mem_singleton] at hs
    subst hs
    exact hAff q k node p ht hn hform hp hL h0 hq hf ha

/-- A piece of positive count, its positions at or past 0, moved onto gates `A + e * F` with `A, F ≥ 0`, as a
segment. -/
theorem shapeOk_affine (P : Prog) (d : Nat) (q : Piece) (w : Nat) {A F : Int} (hA : 0 ≤ A) (hF : 0 ≤ F)
    (h0 : q.count ≠ 0) (hq : ∀ x ∈ pos q, 0 ≤ x.2) : ShapeOk P d (Seg.one (q.affine w A F)) := by
  refine shapeOk_one P d _ (Nat.pos_of_ne_zero h0) (fun x hx => ?_) (fun k hk => by cases hk) (fun x hx i hi => ?_)
  · rw [pos_eq_map, List.mem_map] at hx
    obtain ⟨t, ht, rfl⟩ := hx
    rw [List.mem_range] at ht
    have ht' : t < q.count := ht
    have hx0 : 0 ≤ ppt q t := hq _ (by rw [pos_eq_map]; exact List.mem_map.2 ⟨t, List.mem_range.2 ht', rfl⟩)
    show 0 ≤ ppt (q.affine w A F) t
    rw [ppt_affine]
    have := mul_nonneg hx0 hF
    linarith
  · have := (mem_pos_tgt hx).symm.trans hi
    cases this

/-- **A batch of a composite's piece in its one-run case** is `ShapeOk`: the run's first gate and its step are at or
past 0. `batchShape_of`'s `hAff`. -/
theorem batchAffine_shape (c : Desc.Ctx) (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn) :
    ∀ q k node p, q.tgt = .ref false k → c.node k = .ok node → node.form = .batch →
      (c.P.defs[node.fn]!).prim = false → (c.P.defs[node.fn]!).outLeaves ≠ 0 → q.count ≠ 0 →
      (∀ x ∈ pos q, 0 ≤ x.2) → (∀ x ∈ pos q, Fits c x) → affineOf c k node q = some p →
      ShapeOk c.P c.d (Seg.one p) := by
  intro q k node p ht hn hform hp hL h0 hq hf ha
  obtain ⟨hok, hpos, -, -, hpp⟩ := hR k node hn
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
  have hg0c : 0 < g0.count := hpp s0 (by rw [hs0]; simp) g0 (by rw [hg0]; simp)
  have hr1 : g0.rows = 1 := by
    unfold Desc.Piece.count at hg0c
    rcases Nat.lt_or_ge g0.rows 1 with h | h
    · rw [show g0.rows = 0 by omega, Nat.zero_mul] at hg0c; omega
    · omega
  have hl0 : 0 < g0.len := by rw [count_one hr1] at hg0c; exact hg0c
  have hb0 : 0 ≤ g0.base := by
    have hm := at_mem_pos hr1 hl0
    have := (hpos _ (by rw [hflat]; exact hm)).1
    simpa [Desc.Piece.at] using this
  refine shapeOk_affine c.P c.d q w (add_nonneg (Int.natCast_nonneg _) hb0) ?_ h0 hq
  split
  · exact Int.natCast_nonneg _
  · rename_i hL1
    have hL1' : (c.P.defs[node.fn]!).outLeaves ≠ 1 := by simpa using hL1
    have hG := hFG.resolve_left hL1'
    by_contra hneg
    push Not at hneg
    have hL0 : (0 : Int) < ((c.P.defs[node.fn]!).outLeaves : Int) := by exact_mod_cast Nat.pos_of_ne_zero hL
    nlinarith [Int.natCast_nonneg ((c.P.defs[node.fn]!).gates)]

/-- **A batch of a composite's piece of one row and a stride of whole members**, as `batchStride` reads it, gives
`ShapeOk` segments: copies of a member's leaf of gates, each copy a member at or past member 0; copies of the one
position `fromCallee` keeps of a leaf of arguments the members share (`CalleeShape`, `CalleeSpec`); or position by
position (`batchPts_shape`). -/
theorem batchStride_shape (c : Desc.Ctx) (sc : Flock.Program.Scope) (hB : Resolve.BodyOk c.P c.d) (fuel : Nat)
    (hD : DownShapeAt c fuel) (hCS : CalleeShape c fuel) (hC : CalleeSpec c sc fuel)
    (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn) :
    ∀ q k node segs, q.tgt = .ref false k → c.node k = .ok node → node.form = .batch →
      (c.P.defs[node.fn]!).prim = false → (c.P.defs[node.fn]!).outLeaves ≠ 0 → q.rows = 1 →
      (q.len > 1 ∧ q.stride % ((c.P.defs[node.fn]!).outLeaves : Int) = 0 ∧ q.base ≥ 0) →
      (∀ x ∈ pos q, 0 ≤ x.2) → (∀ x ∈ pos q, Fits c x) → batchStride c fuel k node q = .ok segs →
      ∀ s ∈ segs.toList, ShapeOk c.P c.d s := by
  intro q k node segs ht hn hform hp hL h1 hcp hq hf h
  obtain ⟨hok, hnn, hgo, hag, hpp⟩ := hR k node hn
  have hnode := node_get hn
  have hl0 : 0 < q.len := by omega
  have hfu : q.at 0 < ((c.body.nodes[k]!).out.leaves : Int) := hf _ (at_mem_pos h1 hl0) k ht
  rw [hnode] at hfu
  have hbl : q.base.toNat < node.out.leaves := by
    have : q.at 0 = q.base := by simp [Desc.Piece.at]
    rw [this] at hfu
    omega
  have hdo0 := downOk_of hB hn q.base.toNat hbl
  simp only [hform] at hdo0
  obtain ⟨hmn, -, hjr⟩ := hdo0
  have hjr := hjr hp
  unfold batchStride at h
  dsimp only at h
  split at h
  · rename_i hrep
    split at h
    · rename_i hall
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      intro s hs
      simp only [List.mem_singleton] at hs
      subst hs
      have hgs := gates_of_all hall
      generalize hLd : (c.P.defs[node.fn]!).outLeaves = L at *
      generalize hGd : (c.P.defs[node.fn]!).gates = G at *
      have hleaf1 : ∀ x ∈ ((c.memo[node.fn]!).ret.slice (q.base.toNat % L) (q.base.toNat % L + 1)).toList,
          x.reps = 1 := fun x hx => by
        have h1' := (retOk_slice hgo hx).1
        have h2' := (Array.all_eq_true_iff_forall_mem.1 hrep) x (Array.mem_toList_iff.1 hx)
        simp only [decide_eq_true_eq] at h2'
        omega
      have hsok := psSlice_ok _ hok hpp (q.base.toNat % L) (q.base.toNat % L + 1)
      have hlenr := (hag (.top node.fn 0 none) rfl).length_eq
      simp only [List.length_map, List.length_range] at hlenr
      generalize hrd : (c.memo[node.fn]!).ret = r at *
      have hLp : 0 < L := Nat.pos_of_ne_zero hL
      have hjL : q.base.toNat % L < L := Nat.mod_lt _ hLp
      generalize hjd : q.base.toNat % L = j at *
      generalize hmd : q.base.toNat / L = m0 at *
      have hpsPos : posAll ((r.slice j (j + 1)).flatMap (·.ps)) = sl (psPos r) j (j + 1) := by
        rw [posAll_flatMap_ps _ hleaf1, psSlice_pos _ hok]
      obtain ⟨d, hdd⟩ : ∃ d, q.stride = (L : Int) * d :=
        ⟨q.stride / L, (Int.mul_ediv_cancel' (Int.dvd_of_emod_eq_zero hcp.2.1)).symm⟩
      have hdiv : q.stride / (L : Int) = d := by
        rw [hdd]; exact Int.mul_ediv_cancel_left _ (by exact_mod_cast hL)
      have hB' : (q.base.toNat : Int) = q.base := Int.toNat_of_nonneg hcp.2.2
      have hBdec : q.base.toNat = m0 * L + j := by rw [← hmd, ← hjd]; exact (Nat.div_add_mod' _ _).symm
      refine ⟨?_, fun x hx => ?_, ?_, by show 1 ≤ q.len; omega⟩
      · show (posAll (Desc.shiftGates _ _)).length = 1
        rw [shiftGates_pos, List.length_map, hpsPos]
        unfold sl
        simp only [List.length_take, List.length_drop]
        omega
      · unfold segPos at hx
        obtain ⟨u, hu, hx⟩ := List.mem_flatMap.1 hx
        rw [List.mem_range] at hu
        simp only [copy_pos, shiftGates_pos, List.map_map, shiftPos_shiftPos, List.mem_map] at hx
        obtain ⟨y, hy, rfl⟩ := hx
        have hy0 : 0 ≤ y.2 := by
          rw [hpsPos] at hy
          exact (hnn y (List.mem_of_mem_drop (List.mem_of_mem_take hy))).1
        unfold posAll at hy
        obtain ⟨g, hg, hyg⟩ := List.mem_flatMap.1 hy
        obtain ⟨w, hw⟩ := hgs g hg
        have hyt : y.1 = .gate w := (mem_pos_tgt hyg).trans hw
        have he0 : 0 ≤ (m0 : Int) + u * d := by
          have hat : q.at u = ((m0 : Int) + u * d) * L + j := by
            unfold Desc.Piece.at; rw [hdd, ← hB', hBdec]; push_cast; ring
          have hat0 : 0 ≤ q.at u := by simpa using hq _ (at_mem_pos h1 hu)
          by_contra hneg
          push Not at hneg
          have hle : ((m0 : Int) + u * d) * L ≤ -L := by nlinarith
          have : (j : Int) < L := by exact_mod_cast hjL
          omega
        obtain ⟨t, v⟩ := y
        dsimp only at hyt hy0
        subst hyt
        refine ⟨?_, fun k' hk' => ?_, fun i hi => ?_⟩
        · simp only [shiftPos]
          rw [hdiv]
          nlinarith [Int.natCast_nonneg G, Int.natCast_nonneg (c.body.offsets[k]!)]
        · simp [shiftPos] at hk'
        · simp [shiftPos] at hi
      · exact shiftGates_count _ _ (fun g hg => by
          rw [Array.toList_flatMap, List.mem_flatMap] at hg
          obtain ⟨s, hs, hg⟩ := hg
          exact (hsok s hs).2.2 g hg)
    · split at h
      · rename_i hsh
        obtain ⟨one, hone, h⟩ := Walk.except_bind_ok h
        split at h
        · rename_i hone1
          simp only [pure, Except.pure, Except.ok.injEq] at h
          subst h
          intro s hs
          simp only [List.mem_singleton] at hs
          subst hs
          have hrep1 : ∀ x ∈ one.toList, x.reps = 1 := by
            intro x hx
            have h0 := ((reps_fuel c fuel).2.1 k _ _ _ one hone x hx).1
            have hle := (Array.all_eq_true_iff_forall_mem.1 hone1) x (Array.mem_toList_iff.1 hx)
            simp only [decide_eq_true_eq] at hle
            omega
          have hmem : q.base.toNat / (c.P.defs[node.fn]!).outLeaves < (c.body.nodes[k]!).members := by
            rw [hnode, members_batch hform]; exact hmn
          have hbr : q.base.toNat % (c.P.defs[node.fn]!).outLeaves <
              q.base.toNat % (c.P.defs[node.fn]!).outLeaves + 1 →
              q.base.toNat % (c.P.defs[node.fn]!).outLeaves + 1 ≤
                (c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.count :=
            fun _ => by rw [hnode]; omega
          have hA := (hC k _ _ _ one hmem hbr hone).length_eq
          have hS := hCS k _ _ _ one hmem hbr hone
          have hposeq := posAll_flatMap_ps one hrep1
          refine ⟨?_, fun x hx => ?_, fun g hg => ?_, by show 1 ≤ q.len; omega⟩
          · show (posAll (one.flatMap (·.ps))).length = 1
            rw [hposeq, hA]
            simp
          · rw [segPos_alike rfl] at hx
            obtain ⟨_, -, hx⟩ := List.mem_flatMap.1 hx
            rw [hposeq] at hx
            unfold segsPos at hx
            obtain ⟨s, hs, hxs⟩ := List.mem_flatMap.1 hx
            exact (hS s hs).2.1 x hxs
          · rw [Array.toList_flatMap, List.mem_flatMap] at hg
            obtain ⟨s, hs, hg⟩ := hg
            exact (hS s hs).2.2.1 g hg
        · exact batchPts_shape c fuel hD q ht h1 hq hf h
      · exact batchPts_shape c fuel hD q ht h1 hq hf h
  · exact batchPts_shape c fuel hD q ht h1 hq hf h

/-- A batch's output leaves `[a, b)` from `mid` and what `fromCallee` keeps of the cut members at either end:
`ShapeOk` wherever `mid`'s segments are (`CalleeShape`). -/
theorem ends_shape (c : Desc.Ctx) (fuel : Nat) (hCS : CalleeShape c fuel) {k L a b : Nat} (hL : L ≠ 0)
    (hab : (a + L - 1) / L ≤ b / L) (hLr : L ≤ (c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.count)
    (hm : ∀ j, a ≤ j → j < b → j / L < (c.body.nodes[k]!).members)
    {mid segs : Array Seg} (hmid : ∀ s ∈ mid.toList, ShapeOk c.P c.d s)
    (h : batchEnds c fuel k L a b mid = .ok segs) : ∀ s ∈ segs.toList, ShapeOk c.P c.d s := by
  unfold batchEnds at h
  dsimp only at h
  generalize hm0 : (a + L - 1) / L = m0 at *
  generalize hm1 : b / L = m1 at *
  have hLp : 0 < L := Nat.pos_of_ne_zero hL
  have ha0 : a ≤ m0 * L := by
    rw [← hm0]
    have := Nat.div_add_mod (a + L - 1) L
    have := Nat.mod_lt (a + L - 1) hLp
    have : L * ((a + L - 1) / L) = (a + L - 1) / L * L := Nat.mul_comm _ _
    omega
  have hb0 : m1 * L ≤ b := by rw [← hm1]; exact Nat.div_mul_le_self _ _
  have hb1 : b < m1 * L + L := by
    rw [← hm1]
    have := Nat.div_add_mod b L
    have := Nat.mod_lt b hLp
    have : L * (b / L) = b / L * L := Nat.mul_comm _ _
    omega
  have hmm : m0 * L ≤ m1 * L := Nat.mul_le_mul_right _ hab
  have hfirst : ∀ x, a < m0 * L → Desc.fromCallee c fuel k (a / L) (a % L) L = .ok x →
      ∀ s ∈ x.toList, ShapeOk c.P c.d s := fun x hlt hx =>
    hCS k (a / L) (a % L) L x (hm a le_rfl (by omega)) (fun _ => hLr) hx
  have hlast : ∀ y, m1 * L < b → Desc.fromCallee c fuel k m1 0 (b - m1 * L) = .ok y →
      ∀ s ∈ y.toList, ShapeOk c.P c.d s := by
    intro y hlt hy
    have hmB := hm (m1 * L) (by omega) hlt
    rw [Nat.mul_div_cancel _ hLp] at hmB
    exact hCS k m1 0 (b - m1 * L) y hmB (fun _ => by omega) hy
  obtain ⟨A, B, rfl, hA, hB⟩ : ∃ A B, segs = A ++ mid ++ B ∧ (∀ s ∈ A.toList, ShapeOk c.P c.d s) ∧
      ∀ s ∈ B.toList, ShapeOk c.P c.d s := by
    split at h
    · rename_i hlt
      obtain ⟨x, hx, h⟩ := Walk.except_bind_ok h
      split at h
      · rename_i hlt'
        obtain ⟨y, hy, h⟩ := Walk.except_bind_ok h
        simp only [pure, Except.pure, Except.ok.injEq] at h
        exact ⟨x, y, by rw [← h]; try simp, hfirst x hlt hx, hlast y hlt' hy⟩
      · simp only [pure, Except.pure, Except.ok.injEq] at h
        exact ⟨x, #[], by rw [← h]; try simp, hfirst x hlt hx, by simp⟩
    · split at h
      · rename_i hlt'
        obtain ⟨y, hy, h⟩ := Walk.except_bind_ok h
        simp only [pure, Except.pure, Except.ok.injEq] at h
        exact ⟨#[], y, by rw [← h]; try simp, by simp, hlast y hlt' hy⟩
      · simp only [pure, Except.pure, Except.ok.injEq] at h
        exact ⟨#[], #[], by rw [← h]; try simp, by simp, by simp⟩
  intro s hs
  simp only [Array.toList_append, List.mem_append] at hs
  rcases hs with (hs | hs) | hs
  · exact hA s hs
  · exact hmid s hs
  · exact hB s hs

/-- A batch's output leaves `[a, b)` member by member: `ShapeOk` (`CalleeShape`). -/
theorem batchMembers_shape (c : Desc.Ctx) (fuel : Nat) (hCS : CalleeShape c fuel) {k L a b : Nat}
    (hLr : L ≤ (c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.count) (hm : (b - 1) / L < (c.body.nodes[k]!).members)
    {segs : Array Seg} (h : batchMembers c fuel k L a b = .ok segs) : ∀ s ∈ segs.toList, ShapeOk c.P c.d s := by
  unfold batchMembers at h
  have h := walkAtMost_bind _ _ _ h
  dsimp only at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, Nat.add_sub_cancel, Nat.div_one] at h
  exact loop_mem (ShapeOk c.P c.d)
    (fun m => Desc.fromCallee c fuel k m (max a (m * L) - m * L) (min b ((m + 1) * L) - m * L)) _
    (fun m hmm segs hs => by
      rw [List.mem_range'_1] at hmm
      refine hCS k m _ _ segs (by omega) (fun _ => ?_) hs
      have hmL : (m + 1) * L = m * L + L := by rw [Nat.add_mul, Nat.one_mul]
      omega) segs h

/-- What `throughPS` gives, of an argument whose segments and pieces are of positive count: of positive count, at
positions of the argument's one piece (its bounds check). -/
theorem throughPS_mem {ap : PS} {q bp : Piece} (h : Desc.throughPS ap q = some bp) (hok : PSOk ap)
    (hP : ∀ s ∈ ap.segs.toList, ∀ g ∈ s.ps.toList, 0 < g.count) :
    0 < bp.count ∧ ∀ x ∈ pos bp, x ∈ psPos ap := by
  obtain ⟨segs, st, n⟩ := ap
  rcases segs with ⟨_ | ⟨s, _ | ⟨s', l⟩⟩⟩
  · unfold Desc.throughPS at h; dsimp only at h; cases h
  rotate_left
  · unfold Desc.throughPS at h; dsimp only at h; cases h
  obtain ⟨ps, len, reps, shift⟩ := s
  rcases ps with ⟨_ | ⟨pa, _ | ⟨pa', l'⟩⟩⟩
  · unfold Desc.throughPS at h; dsimp only at h; cases h
  rotate_left
  · unfold Desc.throughPS at h; dsimp only at h; cases h
  rw [throughPS_eq] at h
  split at h
  · cases h
  rename_i hc
  rw [Option.ite_none_left_eq_some] at h
  obtain ⟨hb, hbp⟩ := h
  simp only [Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq, not_or, not_lt, not_le] at hc hb
  have e := Option.some.inj hbp
  subst e
  obtain ⟨⟨hr1, hpr⟩, hq0⟩ := hc
  have hsc : 0 < reps * len := (hok.1 0 (by simp)).2
  have hpc : 0 < pa.rows * pa.len := hP ⟨#[pa], len, reps, shift⟩ (by simp) pa (by simp)
  have hrep : reps = 1 := by
    rcases Nat.eq_zero_or_pos reps with h0 | h0
    · rw [h0, Nat.zero_mul] at hsc; omega
    · omega
  have hpr1 : pa.rows = 1 := by
    rcases Nat.eq_zero_or_pos pa.rows with h0 | h0
    · rw [h0, Nat.zero_mul] at hpc; omega
    · omega
  have hpos : psPos ⟨#[⟨#[pa], len, reps, shift⟩], st, n⟩ = pos pa := by
    show segsPos #[_] = _
    rw [segsPos_singleton, segPos_one_rep (show (⟨#[pa], len, reps, shift⟩ : Seg).reps = 1 from hrep)]
    exact posAll_singleton pa
  refine ⟨by show 0 < q.rows * q.len; unfold Desc.Piece.count at hq0; omega, fun x hx => ?_⟩
  rw [hpos]
  unfold pos at hx
  obtain ⟨r, hr, hx⟩ := List.mem_flatMap.1 hx
  obtain ⟨t, ht, rfl⟩ := List.mem_map.1 hx
  rw [List.mem_range] at hr ht
  dsimp only at hr ht ⊢
  have lo1 : (if q.stride < 0 then ((q.len - 1 : Nat) : Int) * q.stride else 0) ≤ (t : Int) * q.stride := by
    split_ifs with hs
    · have : (t : Int) ≤ ((q.len - 1 : Nat) : Int) := by omega
      nlinarith
    · push Not at hs
      exact mul_nonneg (Int.natCast_nonneg t) hs
  have lo2 : (if q.rowStride < 0 then ((q.rows - 1 : Nat) : Int) * q.rowStride else 0) ≤
      (r : Int) * q.rowStride := by
    split_ifs with hs
    · have : (r : Int) ≤ ((q.rows - 1 : Nat) : Int) := by omega
      nlinarith
    · push Not at hs
      exact mul_nonneg (Int.natCast_nonneg r) hs
  have hi1 : (t : Int) * q.stride ≤ (if q.stride > 0 then ((q.len - 1 : Nat) : Int) * q.stride else 0) := by
    split_ifs with hs
    · have : (t : Int) ≤ ((q.len - 1 : Nat) : Int) := by omega
      nlinarith
    · push Not at hs
      exact mul_nonpos_of_nonneg_of_nonpos (Int.natCast_nonneg t) hs
  have hi2 : (r : Int) * q.rowStride ≤
      (if q.rowStride > 0 then ((q.rows - 1 : Nat) : Int) * q.rowStride else 0) := by
    split_ifs with hs
    · have : (r : Int) ≤ ((q.rows - 1 : Nat) : Int) := by omega
      nlinarith
    · push Not at hs
      exact mul_nonpos_of_nonneg_of_nonpos (Int.natCast_nonneg r) hs
  have he0 : 0 ≤ q.base + (r : Int) * q.rowStride + (t : Int) * q.stride := by linarith [hb.1]
  have heL : q.base + (r : Int) * q.rowStride + (t : Int) * q.stride < pa.len := by linarith [hb.2]
  convert at_mem_pos hpr1 (t := (q.base + (r : Int) * q.rowStride + (t : Int) * q.stride).toNat) (by omega) using 2
  unfold Desc.Piece.at
  rw [Int.toNat_of_nonneg he0]
  ring

/-- `down` of the piece `throughPS` gives of an argument whose pieces are of positive count: `ShapeOk` segments, its
positions being the argument's references. -/
theorem throughPS_shape (c : Desc.Ctx) (fuel : Nat) (hD : DownShapeAt c fuel) {k i : Nat} (hA : ArgOk c k i)
    (hP : ∀ s ∈ (c.arg k i).segs.toList, ∀ q ∈ s.ps.toList, 0 < q.count)
    (hR : ∀ t < ((c.body.nodes[k]!).args[i]!).seq.count,
      Resolve.RefOk c.P c.d k (((c.body.nodes[k]!).args[i]!).seq.get t))
    {p bp : Piece} (h : Desc.throughPS (c.arg k i) p = some bp) {segs : Array Seg}
    (hs : Desc.down c fuel bp = .ok segs) : ∀ s ∈ segs.toList, ShapeOk c.P c.d s := by
  obtain ⟨hcnt, hmem⟩ := throughPS_mem h hA.1 hP
  have hmem' : ∀ x ∈ pos bp, ∃ t < ((c.body.nodes[k]!).args[i]!).seq.count,
      x = refPos (((c.body.nodes[k]!).args[i]!).seq.get t) := by
    intro x hx
    have := hmem x hx
    rw [hA.2] at this
    obtain ⟨t, ht, rfl⟩ := List.mem_map.1 this
    exact ⟨t, List.mem_range.1 ht, rfl⟩
  exact hD bp segs hcnt
    (fun x hx => by obtain ⟨t, -, rfl⟩ := hmem' x hx; simp [refPos])
    (fun x hx k' hk' => by
      obtain ⟨t, ht, rfl⟩ := hmem' x hx
      simp only [refPos, Tgt.ref.injEq] at hk'
      obtain ⟨hp', rfl⟩ := hk'
      have := ((hR t ht).2 hp').2
      simp only [refPos]
      exact_mod_cast this)
    (fun x hx i' hi' => by
      obtain ⟨t, ht, rfl⟩ := hmem' x hx
      simp only [refPos, Tgt.ref.injEq] at hi'
      obtain ⟨hp', rfl⟩ := hi'
      have := (hR t ht).1 hp'
      simp only [refPos]
      exact ⟨this.1, by exact_mod_cast this.2⟩) hs

/-- A batch's output leaves `[a, b)` through `throughPS`, else member by member: `ShapeOk`. -/
theorem batchThrough_shape (c : Desc.Ctx) (hB : Resolve.BodyOk c.P c.d) (fuel : Nat) (hD : DownShapeAt c fuel)
    (hCS : CalleeShape c fuel)
    (hA : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size → ArgOk c k i)
    (hAP : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size →
      ∀ s ∈ (c.arg k i).segs.toList, ∀ q ∈ s.ps.toList, 0 < q.count)
    {k : Nat} {node : Flock.Program.Node} (hn : c.node k = .ok node) (hL : (c.P.defs[node.fn]!).outLeaves ≠ 0)
    (hLr : (c.P.defs[node.fn]!).outLeaves ≤ (c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.count) {a b : Nat}
    (hm : ∀ j, a ≤ j → j < b → j / (c.P.defs[node.fn]!).outLeaves < (c.body.nodes[k]!).members)
    (hmB : (b - 1) / (c.P.defs[node.fn]!).outLeaves < (c.body.nodes[k]!).members)
    {segs : Array Seg} (h : batchThrough c fuel k node a b = .ok segs) : ∀ s ∈ segs.toList, ShapeOk c.P c.d s := by
  have hmem := fun (hh : batchMembers c fuel k (c.P.defs[node.fn]!).outLeaves a b = .ok segs) =>
    batchMembers_shape c fuel hCS hLr hmB hh
  unfold batchThrough at h
  dsimp only at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hc
    split at h
    · rename_i i hti
      split at h
      · rename_i d Li hdl
        split at h
        · rename_i hg
          split at h
          · rename_i bp hbp
            obtain ⟨x, hx, h⟩ := Walk.except_bind_ok h
            obtain ⟨a', ha, -⟩ := Option.bind_eq_some_iff.1 hdl
            obtain ⟨hia, -⟩ := Array.getElem?_eq_some_iff.1 ha
            have hnode := node_get hn
            have hk : k < c.body.nodes.size := node_lt hn
            have hia' : i < (c.body.nodes[k]!).args.size := by rw [hnode]; exact hia
            have hRk : ∀ t < ((c.body.nodes[k]!).args[i]!).seq.count,
                Resolve.RefOk c.P c.d k (((c.body.nodes[k]!).args[i]!).seq.get t) := (hB.1 k hk).2.1 i
            exact ends_shape c fuel hCS hL (by omega) hLr hm
              (throughPS_shape c fuel hD (hA k i hk hia') (hAP k i hk hia') hRk hbp hx) h
          · exact hmem h
        · exact hmem h
      · exact hmem h
    · exact hmem h
  · exact hmem h

/-- Copies of a return of gates only, each moved by a whole member's gates past an offset at or past 0: `ShapeOk`. -/
theorem copiesSeg_shape (P : Prog) (d : Nat) (memo : Array Desc.Summary) (f : Nat) (hR : RetOk P memo f)
    (hgo : ∀ x ∈ (memo[f]!).ret.segs.toList, x.reps ≤ 1 ∧ ∀ g ∈ x.ps.toList, ∃ w, g.tgt = .gate w)
    {o G : Int} (ho : 0 ≤ o) (hG : 0 ≤ G) {len n : Nat} (hl : len = (P.defs[f]!).body.ret.count) (hn : 1 ≤ n) :
    ShapeOk P d (Desc.Seg.mk (Desc.shiftGates o ((memo[f]!).ret.segs.flatMap (·.ps))) len n G) := by
  obtain ⟨hok, hnn, -, hag, hpp⟩ := hR
  have hflat := psPos_flat _ hok (fun x hx => (hgo x hx).1)
  have hlenr := (hag (.top f 0 none) rfl).length_eq
  simp only [List.length_map, List.length_range] at hlenr
  refine ⟨?_, fun x hx => ?_, ?_, hn⟩
  · show (posAll (Desc.shiftGates _ _)).length = len
    rw [shiftGates_pos, List.length_map, ← hflat, hlenr, hl]
  · unfold segPos at hx
    obtain ⟨u, -, hx⟩ := List.mem_flatMap.1 hx
    simp only [copy_pos, shiftGates_pos, List.map_map, shiftPos_shiftPos, List.mem_map] at hx
    obtain ⟨y, hy, rfl⟩ := hx
    have hy' := hy
    unfold posAll at hy'
    obtain ⟨g, hg, hyg⟩ := List.mem_flatMap.1 hy'
    rw [Array.toList_flatMap, List.mem_flatMap] at hg
    obtain ⟨s, hs, hg⟩ := hg
    obtain ⟨w, hw⟩ := (hgo s hs).2 g hg
    have hyt : y.1 = .gate w := (mem_pos_tgt hyg).trans hw
    rw [← hflat] at hy
    have hy0 := (hnn y hy).1
    obtain ⟨t, v⟩ := y
    dsimp only at hyt hy0
    subst hyt
    refine ⟨?_, fun k' hk' => ?_, fun i hi => ?_⟩
    · simp only [shiftPos]
      nlinarith [mul_nonneg (Int.natCast_nonneg u) hG]
    · simp [shiftPos] at hk'
    · simp [shiftPos] at hi
  · exact shiftGates_count _ _ (fun g hg => by
      rw [Array.toList_flatMap, List.mem_flatMap] at hg
      obtain ⟨s, hs, hg⟩ := hg
      exact hpp s hs g hg)

/-- Copies of what `fromCallee` keeps of member `m`'s return leaves `[a, b)`, one copy each: `ShapeOk`
(`CalleeShape`, of length `b - a` by `CalleeSpec`). -/
theorem sharedSeg_shape (c : Desc.Ctx) (sc : Flock.Program.Scope) (fuel : Nat) (hCS : CalleeShape c fuel)
    (hC : CalleeSpec c sc fuel) {k m a b len n : Nat} {one : Array Seg} (hm : m < (c.body.nodes[k]!).members)
    (hb : a < b → b ≤ (c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.count)
    (hone : Desc.fromCallee c fuel k m a b = .ok one) (hone1 : one.all (·.reps ≤ 1) = true)
    (hl : len = b - a) (hn : 1 ≤ n) :
    ShapeOk c.P c.d (Desc.Seg.mk (one.flatMap (·.ps)) len n 0) := by
  have hrep1 : ∀ x ∈ one.toList, x.reps = 1 := by
    intro x hx
    have h0 := ((reps_fuel c fuel).2.1 k _ _ _ one hone x hx).1
    have hle := (Array.all_eq_true_iff_forall_mem.1 hone1) x (Array.mem_toList_iff.1 hx)
    simp only [decide_eq_true_eq] at hle
    omega
  have hA := (hC k m a b one hm hb hone).length_eq
  have hS := hCS k m a b one hm hb hone
  have hposeq := posAll_flatMap_ps one hrep1
  refine ⟨?_, fun x hx => ?_, fun g hg => ?_, hn⟩
  · show (posAll (one.flatMap (·.ps))).length = len
    rw [hposeq, hA, hl]
    simp
  · rw [segPos_alike rfl] at hx
    obtain ⟨_, -, hx⟩ := List.mem_flatMap.1 hx
    rw [hposeq] at hx
    unfold segsPos at hx
    obtain ⟨s, hs, hxs⟩ := List.mem_flatMap.1 hx
    exact (hS s hs).2.1 x hxs
  · rw [Array.toList_flatMap, List.mem_flatMap] at hg
    obtain ⟨s, hs, hg⟩ := hg
    exact (hS s hs).2.2.1 g hg

/-- **A batch of a composite's piece of one row and stride 1 past its one-run case**, as `batchOne` reads it, gives
`ShapeOk` segments: whole members as copies of a return of gates (`copiesSeg_shape`) or of what `fromCallee` keeps of
a member's return of shared leaves (`sharedSeg_shape`), through `throughPS`, or member by member, the cut members at
either end each through its return (`ends_shape`, `batchThrough_shape`). -/
theorem batchOne_shape (c : Desc.Ctx) (sc : Flock.Program.Scope) (hB : Resolve.BodyOk c.P c.d) (fuel : Nat)
    (hD : DownShapeAt c fuel) (hCS : CalleeShape c fuel) (hC : CalleeSpec c sc fuel)
    (hA : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size → ArgOk c k i)
    (hAP : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size →
      ∀ s ∈ (c.arg k i).segs.toList, ∀ q ∈ s.ps.toList, 0 < q.count)
    (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn)
    (hOL : ∀ k node, c.node k = .ok node → (c.P.defs[node.fn]!).prim = false →
      (c.P.defs[node.fn]!).outLeaves = (c.P.defs[node.fn]!).body.ret.count) : BatchOneShape c fuel := by
  intro q k node segs ht hn hform hp hL h0 h1 hs hq hf h
  have hlen : 0 < q.len := by rw [← count_one h1]; omega
  have hdo := downOk_of hB hn
  have hnode := node_get hn
  have hol := hOL k node hn hp
  have hLr : (c.P.defs[node.fn]!).outLeaves ≤ (c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.count := by
    rw [hnode, hol]
  have hbl : q.base.toNat + q.len ≤ node.out.leaves := by
    have hm := at_mem_pos h1 (show q.len - 1 < q.len by omega)
    have hb := hq _ (at_mem_pos h1 hlen)
    have := hf _ hm k ht
    simp only [hnode, Desc.Piece.at, hs] at this hb
    omega
  have hm : ∀ j, q.base.toNat ≤ j → j < q.base.toNat + q.len →
      j / (c.P.defs[node.fn]!).outLeaves < (c.body.nodes[k]!).members := by
    intro j _ hj
    have := hdo j (by omega)
    simp only [hform] at this
    rw [hnode, members_batch hform]
    exact this.1
  have hmB := hm (q.base.toNat + q.len - 1) (by omega) (by omega)
  have hm1 : (q.base.toNat + q.len) / (c.P.defs[node.fn]!).outLeaves ≤ (c.body.nodes[k]!).members := by
    have hLp := Nat.pos_of_ne_zero hL
    have e1 := Nat.div_mul_le_self (q.base.toNat + q.len) (c.P.defs[node.fn]!).outLeaves
    have e5 := Nat.lt_div_mul_add (a := q.base.toNat + q.len - 1) hLp
    have e6 : (q.base.toNat + q.len) / (c.P.defs[node.fn]!).outLeaves * (c.P.defs[node.fn]!).outLeaves ≤
        ((q.base.toNat + q.len - 1) / (c.P.defs[node.fn]!).outLeaves + 1) * (c.P.defs[node.fn]!).outLeaves := by
      rw [Nat.add_mul, Nat.one_mul]; omega
    have := Nat.le_of_mul_le_mul_right e6 hLp
    omega
  unfold batchOne at h
  dsimp only at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
    have hgo : ∀ x ∈ (c.memo[node.fn]!).ret.segs.toList, x.reps ≤ 1 ∧ ∀ g ∈ x.ps.toList, ∃ w, g.tgt = .gate w := by
      intro x hx
      have := (Array.all_eq_true_iff_forall_mem.1 hc.1) x (Array.mem_toList_iff.1 hx)
      simp only [Bool.and_eq_true, decide_eq_true_eq] at this
      exact ⟨this.1, gates_of_all this.2⟩
    refine ends_shape c fuel hCS hL (by omega) hLr hm ?_ h
    intro s hs
    simp only [List.mem_singleton] at hs
    subst hs
    exact copiesSeg_shape c.P c.d c.memo node.fn (hR k node hn) hgo (by positivity) (by positivity) hol (by omega)
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
    obtain ⟨one, hone, h⟩ := Walk.except_bind_ok h
    split at h
    · rename_i hone1
      refine ends_shape c fuel hCS hL (by omega) hLr hm ?_ h
      intro s hs
      simp only [List.mem_singleton] at hs
      subst hs
      exact sharedSeg_shape c sc fuel hCS hC (by omega) (fun _ => hLr) hone hone1 (by omega) (by omega)
    · exact batchThrough_shape c hB fuel hD hCS hA hAP hn hL hLr hm hmB h
  · exact batchThrough_shape c hB fuel hD hCS hA hAP hn hL hLr hm hmB h

/-- A scan's ys `[lo, hi)`, `yl` leaves to an iteration, through `fromCallee` iteration by iteration: `ShapeOk`
segments. -/
theorem ys_shape (c : Desc.Ctx) (fuel : Nat) (hCS : CalleeShape c fuel)
    {k cl yl lo hi : Nat} (hyl : yl ≠ 0) (hlh : lo < hi) (hm : (hi - 1) / yl < (c.body.nodes[k]!).members)
    (hj : ∀ j, lo ≤ j → j < hi → cl + j % yl < (c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.count)
    {out0 segs : Array Seg}
    (h : forIn (List.range' (lo / yl) ((hi - 1) / yl + 1 - lo / yl)) out0 (fun m (s : Array Seg) => do
        let v ← Desc.fromCallee c fuel k m (cl + max lo (m * yl) - m * yl) (cl + min hi ((m + 1) * yl) - m * yl)
        pure (ForInStep.yield (s ++ v))) = .ok segs) :
    ∃ segs2, segs = out0 ++ segs2 ∧ ∀ s ∈ segs2.toList, ShapeOk c.P c.d s :=
  forIn_app (ShapeOk c.P c.d) _ _ (fun m hmm r y hy => by
      rw [List.mem_range'_1] at hmm
      obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      refine ⟨v, hy.symm, hCS k m _ _ v (by omega) (fun hlt => ?_) hv⟩
      have hmL : (m + 1) * yl = m * yl + yl := by rw [Nat.add_mul, Nat.one_mul]
      have := hj (m * yl + (min hi ((m + 1) * yl) - m * yl - 1)) (by omega) (by omega)
      rw [show m * yl + (min hi ((m + 1) * yl) - m * yl - 1) = (min hi ((m + 1) * yl) - m * yl - 1) + m * yl by omega,
        Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt (by omega)] at this
      omega) out0 segs h

/-- A scan's piece of stride 1: its ys past the carry's `cl` leaves, as `scanOne` reads them, give `ShapeOk`
segments. -/
theorem scanYs_shape (c : Desc.Ctx) (fuel : Nat) (hCS : CalleeShape c fuel)
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
    ∃ s2, segs = out0 ++ s2 ∧ ∀ s ∈ s2.toList, ShapeOk c.P c.d s := by
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
    have h := walkAtMost_bind _ _ _ h
    obtain ⟨v, hv, h⟩ := Walk.except_bind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, Nat.add_sub_cancel,
      Nat.div_one] at hv
    exact ys_shape c fuel hCS (k := k) (cl := cl) (yl := yl) (by omega)
      (show max a cl - cl < b - cl by omega)
      (by rw [show b - cl - 1 = b - 1 - cl by omega]; exact hm hc.1) (fun j h1 h2 => by
        have := hj (cl + j) (by omega) (by omega)
        rwa [show cl + j - cl = j by omega] at this) hv
  · simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨#[], by simp, fun s hs => by simp at hs⟩

/-- **A scan's piece of one row**, one step up: what `carry` and `fromCallee` give of its final carry's leaves and its
ys is `ShapeOk` wherever their segments one step down are. -/
theorem scanShape_of (c : Desc.Ctx) (hB : Resolve.BodyOk c.P c.d) (hSO : ScanOut c) (fuel : Nat)
    (hCS : CalleeShape c fuel) (hK : CarryShape c fuel) : ScanShape c fuel := by
  intro q segs k node carry y ht hn hform hn0 h0 hff hr hret hq hf h
  have hdo := downOk_of hB hn
  have h1 := rows_one h0 hr
  have hnode : c.body.nodes[k]! = node := node_get hn
  let M : Nat → Nat × Nat := fun j => if j < carry.leaves then (node.n - 1, j)
    else ((j - carry.leaves) / y.leaves, carry.leaves + (j - carry.leaves) % y.leaves)
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
    split at h
    · rename_i hac
      obtain ⟨v1, hv1, h⟩ := Walk.except_bind_ok h
      obtain ⟨s2, rfl, hA2⟩ := scanYs_shape c fuel hCS (by omega) hy0 hmB hjB h
      have hA1 := hK k (node.n - 1) _ _ v1 (by rw [hnode]; exact hform)
        (by rw [hnode, members_scan hform]; omega) (fun hlt => by
          have := (hbd (min (q.base.toNat + q.len) carry.leaves - 1) (by omega)).2
          simp only [M, if_pos (show min (q.base.toNat + q.len) carry.leaves - 1 < carry.leaves by omega)] at this
          rw [hnode]
          omega) hv1
      intro s hs
      rw [Array.empty_append, Array.toList_append, List.mem_append] at hs
      exact hs.elim (hA1 s) (hA2 s)
    · rename_i hac
      obtain ⟨s2, rfl, hA2⟩ := scanYs_shape c fuel hCS (by omega) hy0 hmB hjB h
      rw [Array.empty_append]
      exact hA2
  · rename_i hs
    unfold scanPts at h
    dsimp only at h
    have h := walkAtMost_bind _ _ _ h
    obtain ⟨v, hv, h⟩ := Walk.except_bind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at hv
    rw [range'_size] at hv
    obtain ⟨segs', hsegs, hA⟩ := forIn_app (ShapeOk c.P c.d) _ (List.range q.len)
      (fun t htm r y' hy => by
        obtain ⟨hb0, hlt⟩ := hpt t (List.mem_range.1 htm)
        obtain ⟨hmn, hjr⟩ := hbd _ hlt
        split at hy
        · rename_i hc
          obtain ⟨v', hv', hy⟩ := Walk.except_bind_ok hy
          simp only [pure, Except.pure, Except.ok.injEq] at hy
          refine ⟨v', hy.symm, ?_⟩
          simp only [M, if_pos hc] at hmn hjr
          have := hK k (node.n - 1) (q.at t).toNat ((q.at t).toNat + 1) v' (by rw [hnode]; exact hform)
            (by rw [hnode, members_scan hform]; exact hmn) (fun _ => by rw [hnode]; omega) hv'
          simpa using this
        · rename_i hc
          obtain ⟨v', hv', hy⟩ := Walk.except_bind_ok hy
          simp only [pure, Except.pure, Except.ok.injEq] at hy
          refine ⟨v', hy.symm, ?_⟩
          simp only [M, if_neg hc] at hmn hjr
          have := hCS k _ _ _ v' (by rw [hnode, members_scan hform]; exact hmn) (fun _ => by rw [hnode]; omega) hv'
          simpa using this) #[] v hv
    rw [hsegs, Array.empty_append]
    exact hA

/-- A decoded Definition's arguments as the context holds them name their references in order (`ctxOf_arg`). -/
theorem argOk_decoded (P : Prog) (memo : Array Desc.Summary) {d : Nat} (hdP : d < P.defs.size) (hDec : Decoded P) :
    ∀ k i, k < (ctxOf P memo d).body.nodes.size → i < ((ctxOf P memo d).body.nodes[k]!).args.size →
      ArgOk (ctxOf P memo d) k i := by
  intro k i hk hi
  have := (hDec d hdP).2.2 k hk i hi
  exact ctxOf_arg P memo d k i hk hi this.1 this.2.1

/-- **`down`, `fromCallee`, `throughArg` and `carry` give `ShapeOk` segments in a decoded Definition's body**, from
`shapes` with the decoder's facts and `specs_decoded`, a batch's cases by `batchShape_of`. -/
theorem shapes_decoded (P : Prog) (memo : Array Desc.Summary) {d : Nat} (hdP : d < P.defs.size)
    (hDec : Decoded P) (hB : Resolve.BodyOk P d)
    (hR : ∀ k node, (ctxOf P memo d).node k = .ok node → RetOk P memo node.fn) (hSO : ScanOut (ctxOf P memo d)) :
    ∀ fuel, DownShapeAt (ctxOf P memo d) fuel ∧ CalleeShape (ctxOf P memo d) fuel ∧
      ThroughShape (ctxOf P memo d) fuel ∧ CarryShape (ctxOf P memo d) fuel := by
  have hS := specs_decoded P memo hdP hDec hB hR hSO (.top d 0 none) rfl
  have hfn : ∀ k node, (ctxOf P memo d).node k = .ok node → node.fn < P.defs.size := by
    intro k node hn
    have hk := node_lt hn
    have := (hB.1 k hk).1
    have hg : (P.defs[d]!).body.nodes[k]! = node := node_get hn
    rwa [hg] at this
  have hA := argOk_decoded P memo hdP hDec
  have hAP : ∀ k i, k < (ctxOf P memo d).body.nodes.size → i < ((ctxOf P memo d).body.nodes[k]!).args.size →
      ∀ s ∈ ((ctxOf P memo d).arg k i).segs.toList, ∀ q ∈ s.ps.toList, 0 < q.count :=
    fun k i hk hi => ctxOf_argCount P memo d k i hk hi
  have hOL : ∀ k node, (ctxOf P memo d).node k = .ok node → (P.defs[node.fn]!).prim = false →
      (P.defs[node.fn]!).outLeaves = (P.defs[node.fn]!).body.ret.count :=
    fun k node hn hp => (hDec node.fn (hfn k node hn)).2.1 hp
  refine shapes (ctxOf P memo d) (.top d 0 none) rfl hB hA hAP ?_ hR (fun fuel => (hS fuel).2.2.1)
    (fun fuel hD hC => batchShape_of _ fuel hD (batchAffine_shape _ hR)
      (batchOne_shape _ (.top d 0 none) hB fuel hD hC (hS fuel).2.1 hA hAP hR hOL)
      (batchStride_shape _ (.top d 0 none) hB fuel hD hC (hS fuel).2.1 hR))
    (fun fuel hC hK => scanShape_of _ hB hSO fuel hC hK)
  intro k i e L hdims hL hip
  change ((((P.defs[d]!).body.nodes[k]!).args[i]!).dims = some (e, L)) at hdims
  by_cases hk : k < (P.defs[d]!).body.nodes.size
  · by_cases hi : i < ((P.defs[d]!).body.nodes[k]!).args.size
    · exact ((hDec d hdP).2.2 k hk i hi).2.2 e L hdims hL hip
    · rw [getElem!_neg _ i hi] at hdims; cases hdims
  · rw [getElem!_neg _ k hk, getElem!_neg _ i (Nat.not_lt_zero i)] at hdims; cases hdims

/-- **`DownShape` in a decoded Definition's body** (`shapes_decoded`). -/
theorem downShape_decoded (P : Prog) (memo : Array Desc.Summary) {d : Nat} (hdP : d < P.defs.size)
    (hDec : Decoded P) (hB : Resolve.BodyOk P d)
    (hR : ∀ k node, (ctxOf P memo d).node k = .ok node → RetOk P memo node.fn) (hSO : ScanOut (ctxOf P memo d)) :
    DownShape (ctxOf P memo d) :=
  fun fuel => (shapes_decoded P memo hdP hDec hB hR hSO fuel).1

/-- **The decoder's facts on a Definition's return**, as its private `composite` sets them: a primitive's body (the
default) returns none, and a composite's return is the sequence of its parts, none naming a negative position. Every
decoded Program has them (`BodiesOk.retParts_reads`). -/
def RetParts (P : Prog) : Prop :=
  ∀ d < P.defs.size, ((P.defs[d]!).prim = true → (P.defs[d]!).body.ret.count = 0) ∧
    ((P.defs[d]!).prim = false → (P.defs[d]!).body.ret = Seq.ofParts (P.defs[d]!).body.ret.parts ∧
      ∀ q ∈ (P.defs[d]!).body.ret.parts.toList, PartNonneg q)

theorem ofSeq_reps (s : Seq) : ∀ g ∈ (Desc.ofSeq s).segs.toList, g.reps = 1 := by
  intro g hg
  unfold Desc.ofSeq at hg
  rw [of_segs, Array.toList_filter, List.mem_filter, Array.toList_map, List.mem_map] at hg
  obtain ⟨⟨q, -, rfl⟩, -⟩ := hg
  rfl

theorem segsPos_pieces (segs : List Seg) (h : ∀ g ∈ segs, g.reps = 1) :
    segs.flatMap segPos = (segs.flatMap fun s => s.ps.toList).flatMap pos := by
  rw [List.flatMap_assoc]
  apply flatMap_congr
  intro g hg
  rw [segPos_one_rep (h g hg)]
  rfl

theorem psPos_default : psPos (default : PS) = [] := rfl

theorem psOk_default : PSOk (default : PS) :=
  ⟨fun i hi => absurd hi (Nat.not_lt_zero _), rfl, fun i hi => absurd hi (Nat.not_lt_zero _), rfl⟩

/-- `coalesce` keeps its segments `ShapeOk`: a piece it joins names their positions, so the positions' facts hold, and
is one copy of one piece of positive count. -/
theorem shapeOk_coalesce {P : Prog} {d : Nat} {segs : Array Seg} (h : ∀ g ∈ segs.toList, ShapeOk P d g) :
    ∀ g ∈ (Desc.coalesce segs).toList, ShapeOk P d g := by
  intro g hg
  rcases coalesce_mem (fun g => g ∈ segs.toList ∨ ∃ r, g = Seg.one r ∧ 0 < r.count)
      (fun r hr => .inr ⟨r, rfl, hr⟩) segs (fun g hg => .inl hg) g hg with hg' | ⟨r, rfl, hr⟩
  · exact h g hg'
  · refine ⟨?_, fun x hx => ?_, fun q hq => ?_, Nat.le_refl 1⟩
    · show (posAll #[r]).length = r.count
      rw [posAll_singleton, pos_length]
    · have : x ∈ segsPos segs := by
        rw [← coalesce_pos]; exact List.mem_flatMap.2 ⟨_, hg, hx⟩
      obtain ⟨g', hg', hx'⟩ := List.mem_flatMap.1 this
      exact (h g' hg').2.1 x hx'
    · have : q = r := by simpa [Seg.one] using hq
      subst this
      exact hr

/-- **What `Desc.summarize` gives Definition `d` holds `RetOk`**, wherever `down`'s segments are `ShapeOk`
(`DownShape`) and keep the gates they read (`DownSpec`) in its body, and the decoder's facts on its return hold. -/
theorem summarize_retOk (P : Prog) (memo : Array Desc.Summary) (d : Nat) (s : Desc.Summary)
    (hdP : d < P.defs.size) (hRP : RetParts P) (hB : Resolve.BodyOk P d)
    (hS : (P.defs[d]!).prim = false → DownShape (ctxOf P memo d))
    (hD : (P.defs[d]!).prim = false → ∀ sc : Flock.Program.Scope, sc.d = d →
      DownSpec (ctxOf P memo d) sc Flock.Steps.FUEL)
    (h : Desc.summarize P memo d = .ok s) (m : Array Desc.Summary) (hm : m[d]! = s) : RetOk P m d := by
  have h3 := summarize_reps P memo d s h
  unfold RetOk
  rw [hm]
  unfold Desc.summarize at h
  extract_lets defn body c ret0 acc0 at h
  split at h
  · rename_i hp
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    have h0 : (P.defs[d]!).body.ret.count = 0 := (hRP d hdP).1 hp
    refine ⟨psOk_default, fun x hx => absurd hx (by rw [psPos_default]; simp), fun g hg => (by cases hg),
      fun sc _ => ?_, fun g hg => (by cases hg)⟩
    rw [psPos_default, h0]
    exact List.Forall₂.nil
  · rename_i hp
    have hp' : (P.defs[d]!).prim = false := by simpa using hp
    obtain ⟨ret, hret, h⟩ := Walk.except_bind_ok h
    obtain ⟨acc, -, h⟩ := Walk.except_bind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    rw [← Array.forIn_toList] at hret
    obtain ⟨outs, hF, hout⟩ := forIn_trace2 _ _ ret0 ret hret
    obtain ⟨hpe, hnn⟩ := (hRP d hdP).2 hp'
    have hpe' : body.ret = Seq.ofParts body.ret.parts := hpe
    have hnn' : ∀ q ∈ body.ret.parts.toList, PartNonneg q := hnn
    have hpos : psPos (Desc.ofSeq body.ret) = (List.range body.ret.count).map fun t => refPos (body.ret.get t) := by
      rw [(ofSeq_pos body.ret hnn').2.1]
      conv => rhs; rw [hpe']
      rw [partsPos_get]
    have hqs : ((Desc.ofSeq body.ret).segs.toList.flatMap fun s => s.ps.toList).flatMap pos =
        psPos (Desc.ofSeq body.ret) := by
      rw [← segsPos_pieces _ (ofSeq_reps body.ret)]; rfl
    have hx : ∀ q ∈ ((Desc.ofSeq body.ret).segs.toList.flatMap fun s => s.ps.toList), ∀ x ∈ pos q,
        ∃ t < body.ret.count, x = refPos (body.ret.get t) := by
      intro q hq x hx
      have : x ∈ psPos (Desc.ofSeq body.ret) := by
        rw [← hqs]; exact List.mem_flatMap.2 ⟨q, hq, hx⟩
      rw [hpos, List.mem_map] at this
      obtain ⟨t, ht, e⟩ := this
      exact ⟨t, List.mem_range.1 ht, e.symm⟩
    have hrefOk : ∀ t < body.ret.count, Resolve.RefOk P d body.nodes.size (body.ret.get t) := hB.2
    have hq1 : ∀ q ∈ ((Desc.ofSeq body.ret).segs.toList.flatMap fun s => s.ps.toList), ∀ x ∈ pos q, 0 ≤ x.2 := by
      intro q hq x hxq
      obtain ⟨t, -, rfl⟩ := hx q hq x hxq
      simp [refPos]
    have hq2 : ∀ q ∈ ((Desc.ofSeq body.ret).segs.toList.flatMap fun s => s.ps.toList), ∀ x ∈ pos q,
        Fits (ctxOf P memo d) x := by
      intro q hq x hxq k hk
      obtain ⟨t, ht, rfl⟩ := hx q hq x hxq
      simp only [refPos, Tgt.ref.injEq] at hk
      have := (hrefOk t ht).2 hk.1
      rw [← hk.2]
      show ((body.ret.get t).j : Int) < _
      exact_mod_cast this.2
    have hq3 : ∀ q ∈ ((Desc.ofSeq body.ret).segs.toList.flatMap fun s => s.ps.toList), ∀ x ∈ pos q,
        ∀ i, x.1 = .ref true i → i < (ctxOf P memo d).body.params.size ∧
          x.2 < (((ctxOf P memo d).body.params[i]!).2.leaves : Int) := by
      intro q hq x hxq i hi
      obtain ⟨t, ht, rfl⟩ := hx q hq x hxq
      simp only [refPos, Tgt.ref.injEq] at hi
      have := (hrefOk t ht).1 hi.1
      rw [← hi.2]
      exact ⟨this.1, by show ((body.ret.get t).j : Int) < _; exact_mod_cast this.2⟩
    have hq0 : ∀ q ∈ ((Desc.ofSeq body.ret).segs.toList.flatMap fun s => s.ps.toList), 0 < q.count := by
      intro q hq
      obtain ⟨g, hg, hqg⟩ := List.mem_flatMap.1 hq
      unfold Desc.ofSeq at hg
      rw [of_segs, Array.toList_filter, List.mem_filter, Array.toList_map, List.mem_map] at hg
      obtain ⟨⟨q', -, rfl⟩, hc⟩ := hg
      have e : q = q' := by simpa [Seg.one] using hqg
      subst e
      simpa [Desc.Seg.count, Seg.one] using hc
    have hshape : ∀ g ∈ ret.toList, ShapeOk P d g := by
      intro g hg
      rw [hout] at hg
      rcases mem_foldl outs ret0 g hg with hg | ⟨v, hv, hg⟩
      · exact absurd hg (by simp [ret0])
      · obtain ⟨q, hq, hqv⟩ := forall2_mem hF v hv
        exact hS hp' Flock.Steps.FUEL q v (hq0 q hq) (hq1 q hq) (hq2 q hq) (hq3 q hq) hqv g hg
    have hshape' := shapeOk_coalesce hshape
    obtain ⟨hok, hpsPos, -⟩ := of_ok (Desc.coalesce ret) fun g hg => (hshape' g hg).1
    rw [coalesce_pos] at hpsPos
    refine ⟨hok, fun x hx => ?_, fun g hg => (h3 g hg).2, fun sc hsc => ?_, fun g hg => ?_⟩
    · rw [hpsPos] at hx
      obtain ⟨g, hg, hxg⟩ := List.mem_flatMap.1 hx
      exact (hshape g hg).2.1 x hxg
    rotate_left
    · change g ∈ (PS.of (Desc.coalesce ret)).segs.toList at hg
      rw [of_segs, Array.toList_filter, List.mem_filter] at hg
      exact (hshape' g hg.1).2.2.1
    · show Agree P sc (psPos (PS.of (Desc.coalesce ret))) ((List.range body.ret.count).map fun j =>
        Flock.Program.resolve P sc (body.ret.get j))
      rw [hpsPos, hout, segsPos_foldl, show segsPos ret0 = [] from rfl, List.nil_append]
      have ha := agree_trace (P := P) (sc := sc) (T := fun q => (pos q).map (den P sc)) hF
        (fun q hq v hv => hD hp' sc hsc q v (hq1 q hq) (hq2 q hq) hv)
      have e1 : (((Desc.ofSeq body.ret).segs.toList.flatMap fun s => s.ps.toList).flatMap fun q =>
          (pos q).map (den P sc)) = (psPos (Desc.ofSeq body.ret)).map (den P sc) := by
        rw [← hqs, List.map_flatMap]
      have e2 : ((List.range body.ret.count).map fun j => Flock.Program.resolve P sc (body.ret.get j)) =
          (psPos (Desc.ofSeq body.ret)).map (den P sc) := by
        rw [hpos, List.map_map]
        apply List.map_congr_left
        intro j _
        exact (den_refPos P sc _).symm
      rw [e2, ← e1]
      exact ha

/-- A loop over `[s, s + n)` that always continues, keeping an invariant of the index reached. -/
theorem forIn_range_idx {σ : Type} (I : Nat → σ → Prop) (f : Nat → σ → Except String (ForInStep σ)) (N : Nat)
    (hF : ∀ t r y, t < N → I t r → f t r = .ok y → ∃ v, y = .yield v ∧ I (t + 1) v) :
    ∀ (n s : Nat) (acc out : σ), s + n ≤ N → I s acc → forIn (List.range' s n) acc f = .ok out → I (s + n) out
  | 0, s, acc, out, _, hI, h => by
    simp only [List.range'_zero, List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h; simpa using hI
  | n + 1, s, acc, out, hN, hI, h => by
    rw [List.range'_succ, List.forIn_cons] at h
    obtain ⟨y, hy, h⟩ := Walk.except_bind_ok h
    obtain ⟨v, rfl, hv⟩ := hF s acc y (by omega) hI hy
    have := forIn_range_idx I f N hF n (s + 1) v out (by omega) hv h
    rwa [Nat.add_assoc, Nat.add_comm 1 n] at this

theorem retOk_push {P : Prog} {r : Array Desc.Summary} {f : Nat} (hf : f < r.size) (v : Desc.Summary)
    (h : RetOk P r f) : RetOk P (r.push v) f := by
  unfold RetOk at h ⊢
  rwa [getElem!_pos (r.push v) f (by simp; omega), Array.getElem_push_lt hf, ← getElem!_pos r f hf]

theorem set!_size (a : Array Bool) (i : Nat) (v : Bool) : (a.set! i v).size = a.size := by
  rw [Array.set!_eq_setIfInBounds, Array.size_setIfInBounds]

theorem set!_get (a : Array Bool) (i x : Nat) :
    (a.set! i true)[x]! = true ↔ (x = i ∧ i < a.size) ∨ a[x]! = true := by
  rw [Array.set!_eq_setIfInBounds]
  by_cases hx : x < a.size
  · rw [getElem!_pos _ x (by rw [Array.size_setIfInBounds]; exact hx), Array.getElem_setIfInBounds hx,
      getElem!_pos a x hx]
    split
    · rename_i e; subst e; simp [hx]
    · rename_i e; simp [Ne.symm e]
  · rw [getElem!_neg _ x (by rw [Array.size_setIfInBounds]; exact hx), getElem!_neg a x hx]
    constructor
    · intro h; cases h
    · rintro (⟨rfl, h⟩ | h)
      · exact absurd h hx
      · cases h

/-- Marking the callees of `ns`: each one marked, nothing else newly. -/
theorem marks_spec : ∀ (ns : List Flock.Program.Node) (b : Array Bool),
    (ns.foldl (fun r n => r.set! n.fn true) b).size = b.size ∧
    (∀ x, (ns.foldl (fun r n => r.set! n.fn true) b)[x]! = true ↔
      (∃ n ∈ ns, n.fn = x ∧ x < b.size) ∨ b[x]! = true)
  | [], b => ⟨rfl, fun x => by simp⟩
  | n :: ns, b => by
    obtain ⟨h1, h2⟩ := marks_spec ns (b.set! n.fn true)
    rw [set!_size] at h1
    refine ⟨by rw [List.foldl_cons]; exact h1, fun x => ?_⟩
    rw [List.foldl_cons, h2, set!_get, set!_size]
    constructor
    · rintro (⟨m, hm, rfl, hx⟩ | ⟨rfl, hx⟩ | hx)
      · exact .inl ⟨m, List.mem_cons_of_mem _ hm, rfl, hx⟩
      · exact .inl ⟨n, List.mem_cons_self, rfl, hx⟩
      · exact .inr hx
    · rintro (⟨m, hm, rfl, hx⟩ | hx)
      · rcases List.mem_cons.1 hm with rfl | hm
        · exact .inr (.inl ⟨rfl, hx⟩)
        · exact .inl ⟨m, hm, rfl, hx⟩
      · exact .inr (.inr hx)

/-- **`Desc.reached` is closed under calls**: a composite it marks has its callees marked, its Definitions' callees
decoded before them. -/
theorem reached_closed (P : Prog)
    (hcf : ∀ d < P.defs.size, ∀ k < (P.defs[d]!).body.nodes.size, ((P.defs[d]!).body.nodes[k]!).fn < d)
    (d : Nat) (hd : (Desc.reached P)[d]! = true) (hp : (P.defs[d]!).prim = false) :
    ∀ k < (P.defs[d]!).body.nodes.size, (Desc.reached P)[((P.defs[d]!).body.nodes[k]!).fn]! = true := by
  let N := P.defs.size
  let G : Nat → Array Bool → Array Bool := fun i s =>
    if (!s[N - 1 - i]!) = true then s
    else if (P.defs[N - 1 - i]!).prim = true then s
    else (P.defs[N - 1 - i]!).body.nodes.foldl (fun b a => b.set! a.fn true) s
  let Inv : Nat → Array Bool → Prop := fun s b => b.size = N ∧ ∀ d', N - s ≤ d' → d' < N → b[d']! = true →
    (P.defs[d']!).prim = false → ∀ k < (P.defs[d']!).body.nodes.size, b[((P.defs[d']!).body.nodes[k]!).fn]! = true
  have step : ∀ s b, s < N → Inv s b → Inv (s + 1) (G s b) := by
    intro s b hs ⟨hsz, hI⟩
    have hd0 : N - (s + 1) = N - 1 - s := by omega
    by_cases hA : (!b[N - 1 - s]!) = true
    · have e : G s b = b := ite_eq_left hA
      rw [e]
      refine ⟨hsz, fun d' h1 h2 h3 h4 k hk => ?_⟩
      rcases Nat.eq_or_lt_of_le (hd0 ▸ h1) with e' | e'
      · subst e'; simp [h3] at hA
      · exact hI d' (by omega) h2 h3 h4 k hk
    by_cases hB : (P.defs[N - 1 - s]!).prim = true
    · have e : G s b = b := by simp only [G]; rw [ite_eq_right hA, ite_eq_left hB]
      rw [e]
      refine ⟨hsz, fun d' h1 h2 h3 h4 k hk => ?_⟩
      rcases Nat.eq_or_lt_of_le (hd0 ▸ h1) with e' | e'
      · subst e'; rw [hB] at h4; cases h4
      · exact hI d' (by omega) h2 h3 h4 k hk
    have e : G s b = (P.defs[N - 1 - s]!).body.nodes.foldl (fun b a => b.set! a.fn true) b := by
      simp only [G]; rw [ite_eq_right hA, ite_eq_right hB]
    rw [e, ← Array.foldl_toList]
    obtain ⟨m1, m2⟩ := marks_spec (P.defs[N - 1 - s]!).body.nodes.toList b
    have hlt : ∀ n ∈ (P.defs[N - 1 - s]!).body.nodes.toList, n.fn < N - 1 - s := by
      intro n hn
      obtain ⟨k, hk, rfl⟩ := List.mem_iff_getElem.1 hn
      have hk' : k < (P.defs[N - 1 - s]!).body.nodes.size := by simpa using hk
      have := hcf (N - 1 - s) (by omega) k hk'
      rwa [getElem!_pos _ k hk', ← Array.getElem_toList] at this
    refine ⟨by rw [m1, hsz], fun d' h1 h2 h3 h4 k hk => ?_⟩
    rcases Nat.eq_or_lt_of_le (hd0 ▸ h1) with e' | e'
    · subst e'
      refine (m2 _).2 (.inl ⟨_, ?_, rfl, ?_⟩)
      · rw [getElem!_pos _ k hk, ← Array.getElem_toList]; exact List.getElem_mem _
      · have := hcf _ (by omega) k hk; omega
    · rcases (m2 d').1 h3 with ⟨n, hn, e2, -⟩ | h3
      · have := hlt n hn; omega
      · exact (m2 _).2 (.inr (hI d' (by omega) h2 h3 h4 k hk))
  have run : ∀ n s b, s + n ≤ N → Inv s b → Inv (s + n) ((List.range' s n).foldl (fun b i => G i b) b) := by
    intro n
    induction n with
    | zero => intro s b _ h; simpa using h
    | succ n ih =>
      intro s b hn h
      rw [List.range'_succ, List.foldl_cons]
      have := ih (s + 1) (G s b) (by omega) (step s b (by omega) h)
      rwa [Nat.add_assoc, Nat.add_comm 1 n] at this
  have hfin : Inv N (Desc.reached P) := by
    have h0 : Inv 0 ((Array.replicate N false).set! P.root true) :=
      ⟨by rw [set!_size, Array.size_replicate], fun d' h1 h2 => absurd h2 (by omega)⟩
    have := run N 0 _ (by omega) h0
    rw [Nat.zero_add] at this
    unfold Desc.reached
    dsimp only
    rw [Std.Legacy.Range.forIn_eq_forIn_range']
    rw [forIn_id_yield _ _ G _ (fun a b => ?_)]
    · simpa using this
    · dsimp only [G]
      split
      · rfl
      · split
        · rfl
        · rw [Array.forIn_pure_yield_eq_foldl]; rfl
  have hdN : d < N := by
    by_contra hge
    rw [getElem!_neg _ d (by rw [hfin.1]; exact hge)] at hd
    cases hd
  exact hfin.2 d (by omega) hdN hd hp

/-- **Every summary `memoOf` makes of a reached Definition other than the root holds `RetOk`**, wherever every
Definition's callees are decoded before it and none is the root, and the reached Definitions' references
(`Resolve.BodyOk`) and the decoder's facts (`Decoded`, `ScanOut`, `RetParts`) hold. -/
theorem memoOf_retOk (P : Prog) (memo : Array Desc.Summary) (h : memoOf P = .ok memo)
    (hRP : RetParts P) (hDec : Decoded P)
    (hcf : ∀ d < P.defs.size, ∀ k < (P.defs[d]!).body.nodes.size, ((P.defs[d]!).body.nodes[k]!).fn < d)
    (hnr : ∀ d < P.defs.size, (Desc.reached P)[d]! = true →
      ∀ k < (P.defs[d]!).body.nodes.size, ((P.defs[d]!).body.nodes[k]!).fn ≠ P.root)
    (hB : ∀ d < P.defs.size, (Desc.reached P)[d]! = true → Resolve.BodyOk P d)
    (hSO : ∀ m d, ScanOut (ctxOf P m d)) :
    ∀ f < P.defs.size, (Desc.reached P)[f]! = true → f ≠ P.root → RetOk P memo f := by
  unfold memoOf at h
  extract_lets seen memo0 at h
  have h' := ok_bind_pure h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h'
  have := forIn_range_idx (fun t (r : Array Desc.Summary) => r.size = t ∧
      ∀ f < t, (Desc.reached P)[f]! = true → f ≠ P.root → RetOk P r f) _ P.defs.size ?_ P.defs.size 0 memo0 memo
    (by simp) ⟨rfl, fun f hf => absurd hf (Nat.not_lt_zero _)⟩ (by simpa using h')
  · exact fun f hf => (this.2 f (by simpa using hf))
  intro t r y htP ⟨hsz, hI⟩ hy
  obtain ⟨v, rfl, hv⟩ : ∃ v, y = .yield (r.push v) ∧
      ((seen[t]! = true ∧ ¬t = P.root) → Desc.summarize P r t = .ok v) := by
    split at hy
    · rcases hs : Desc.summarize P r t with e | v
      · rw [hs] at hy; cases hy
      · rw [hs] at hy
        change Except.ok (ForInStep.yield (r.push v)) = Except.ok y at hy
        injection hy with hy
        exact ⟨v, hy.symm, fun _ => rfl⟩
    · rename_i hc
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      exact ⟨default, hy.symm, fun h => absurd h hc⟩
  refine ⟨_, rfl, by simp [hsz], fun f hf hrf hfr => ?_⟩
  rcases Nat.lt_succ_iff_lt_or_eq.1 hf with hf | rfl
  · exact retOk_push (by omega) v (hI f hf hrf hfr)
  · replace hv := hv ⟨hrf, hfr⟩
    have hR : (P.defs[f]!).prim = false → ∀ k node, (ctxOf P r f).node k = .ok node → RetOk P r node.fn := by
      intro hp k node hn
      have hk := node_lt hn
      have hg := node_get hn
      change k < (P.defs[f]!).body.nodes.size at hk
      change (P.defs[f]!).body.nodes[k]! = node at hg
      rw [← hg]
      exact hI _ (hcf f htP k hk) (reached_closed P hcf f hrf hp k hk) (hnr f htP hrf k hk)
    refine summarize_retOk P r f v htP hRP (hB f htP hrf)
      (fun hp => downShape_decoded P r htP hDec (hB f htP hrf) (hR hp) (hSO r f))
      (fun hp sc hsc => (specs_decoded P r htP hDec (hB f htP hrf) (hR hp) (hSO r f) sc hsc
        Flock.Steps.FUEL).1) hv (r.push v) ?_
    subst hsz
    rw [getElem!_pos (r.push v) r.size (by simp), Array.getElem_push_eq]

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

/-- `Array.qsort` by a key keeps only elements it was given. -/
theorem qsort_mem (as : Array α) {x : α} (hx : x ∈ (as.qsort (fun x y => decide (f x < f y))).toList) :
    x ∈ as.toList := by
  unfold Array.qsort at hx
  split at hx
  · exact hx
  · rename_i h
    dsimp only at hx
    obtain ⟨hW, -⟩ := sort_spec f _ as.toVector (min 0 (as.size - 1)) (max (min 0 (as.size - 1))
      (min (as.size - 1) (as.size - 1))) (by omega) (by omega) (by omega) rfl
    obtain ⟨t, ht, e⟩ := List.getElem_of_mem hx
    simp only [Array.length_toList, Vector.size_toArray] at ht
    obtain ⟨s, hs, -, -, es⟩ := hW.2 t ht (by omega) (by omega)
    rw [← e]
    simp only [Array.getElem_toList, Vector.getElem_toArray]
    rw [es]
    exact Array.mem_toList_iff.2 (Array.getElem_mem hs)

end QSort

/-! ## The used leaves' intervals (`Desc.Ivs.norm`, `Desc.Ivs.covers`) -/

/-- `Desc.Ivs.norm`'s step: an interval starting within the last one's extends it, else it follows. -/
def normStep (acc : Array (Nat × Nat)) (p : Nat × Nat) : Array (Nat × Nat) :=
  match acc.back? with
  | some (c, e) => if p.1 ≤ e then acc.pop.push (c, max p.2 e) else acc.push p
  | none => acc.push p

theorem norm_eq (xs : Array (Nat × Nat)) :
    Desc.Ivs.norm xs =
      ((xs.filter fun p => p.1 < p.2).toList.mergeSort fun x y => decide (x.1 ≤ y.1)).foldl normStep #[] := by
  unfold Desc.Ivs.norm
  rfl

/-- `List.mergeSort` by starts sorts by them. -/
theorem sorted_starts (L : List (Nat × Nat)) :
    (L.mergeSort fun x y => decide (x.1 ≤ y.1)).Pairwise (fun x y => x.1 ≤ y.1) :=
  (List.pairwise_mergeSort (le := fun x y : Nat × Nat => decide (x.1 ≤ y.1))
    (fun a b c hab hbc => by simp only [decide_eq_true_eq] at *; omega)
    (fun a b => by simp only [Bool.or_eq_true, decide_eq_true_eq]; omega) L).imp fun h => by simpa using h

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
    exact ⟨iv, (List.mergeSort_perm _ _).mem_iff.1 hiv, h⟩

/-- **`Desc.Ivs.norm` keeps exactly its intervals' leaves.** -/
theorem norm_mem (xs : Array (Nat × Nat)) (a : Nat) :
    Width.inIvs (Desc.Ivs.norm xs).toList a ↔ Width.inIvs xs.toList a := by
  refine ⟨norm_sub xs a, fun h => ?_⟩
  rw [norm_eq]
  apply normFold_sup _ #[] (sorted_starts _) (fun p _ x hx => by simp at hx)
  rw [← inIvs_nonempty] at h
  obtain ⟨iv, hiv, h⟩ := h
  exact Or.inr ⟨iv, (List.mergeSort_perm _ _).mem_iff.2 hiv, h⟩

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

/-! ## The shape of a root node's segments

Every segment `down` gives a piece of a root node's argument (`FromDown`), or one `Desc.memberRows` gives (`FromRows`),
is `ShapeOk` in the root's body (`shapes_decoded`), wherever the decoder's facts hold (`RootFacts`): the root's callees'
summaries hold `RetOk` (`memoOf_retOk`, `reached_root`). -/

theorem mem_sl {α : Type} {l : List α} {a b : Nat} {x : α} (h : x ∈ sl l a b) : x ∈ l :=
  List.mem_of_mem_drop (List.mem_of_mem_take h)

/-- An argument a node doesn't have is, as the context holds it, of no segments. -/
theorem ctxOf_arg_ge (P : Prog) (memo : Array Desc.Summary) (d k i : Nat)
    (hi : ¬ i < ((P.defs[d]!).body.nodes[k]!).args.size) : ((ctxOf P memo d).arg k i).segs = #[] := by
  unfold Desc.Ctx.arg ctxOf
  dsimp only
  by_cases hk : k < (P.defs[d]!).body.nodes.size
  · rw [getElem!_pos _ k hk] at hi
    rw [Array.getElem?_map, Array.getElem?_eq_getElem hk, Option.map_some, Option.bind_some, Array.getElem?_map,
      Array.getElem?_eq_none (by omega), Option.map_none, Option.getD_none]
    rfl
  · rw [Array.getElem?_map, Array.getElem?_eq_none (by omega), Option.map_none, Option.bind_none, Option.getD_none]
    rfl

theorem pieces_empty {s : Desc.PS} (h : s.segs = #[]) (lo hi : Nat) : s.pieces lo hi = #[] := by
  unfold Desc.PS.pieces Desc.PS.slice
  simp [h]

theorem throughPS_empty {s : Desc.PS} (h : s.segs = #[]) (p : Piece) : Desc.throughPS s p = none := by
  unfold Desc.throughPS
  rw [h]
  rfl

/-- A piece argument `i` of node `k` gives (`PS.pieces`), through `down`: `ShapeOk` segments. -/
theorem piece_shape (c : Desc.Ctx) (fuel : Nat) (hD : DownShapeAt c fuel) {k i : Nat} (hA : ArgOk c k i)
    (hP : ∀ s ∈ (c.arg k i).segs.toList, ∀ q ∈ s.ps.toList, 0 < q.count)
    (hR : ∀ t < ((c.body.nodes[k]!).args[i]!).seq.count,
      Resolve.RefOk c.P c.d k (((c.body.nodes[k]!).args[i]!).seq.get t))
    {lo hi : Nat} {q : Piece} (hq : q ∈ (c.arg k i).pieces lo hi) {segs : Array Seg}
    (h : Desc.down c fuel q = .ok segs) : ∀ s ∈ segs.toList, ShapeOk c.P c.d s := by
  have hq' := Array.mem_toList_iff.2 hq
  have hmem : ∀ x ∈ pos q, ∃ t < ((c.body.nodes[k]!).args[i]!).seq.count,
      x = refPos (((c.body.nodes[k]!).args[i]!).seq.get t) := by
    intro x hx
    have : x ∈ posAll ((c.arg k i).pieces lo hi) := List.mem_flatMap.2 ⟨q, hq', hx⟩
    rw [pieces_pos _ hA.1 lo hi] at this
    have := mem_sl this
    rw [hA.2] at this
    obtain ⟨t, ht, rfl⟩ := List.mem_map.1 this
    exact ⟨t, List.mem_range.1 ht, rfl⟩
  have hcnt : 0 < q.count := by
    unfold Desc.PS.pieces at hq'
    rw [Array.toList_flatMap, List.mem_flatMap] at hq'
    obtain ⟨g, hg, hq'⟩ := hq'
    rw [Array.toList_flatMap, List.mem_flatMap] at hq'
    obtain ⟨m, -, hq'⟩ := hq'
    exact shiftGates_count _ _ (psSlice_ok _ hA.1 hP lo hi g hg).2.2 q hq'
  exact hD q segs hcnt
    (fun x hx => by obtain ⟨t, -, rfl⟩ := hmem x hx; simp [refPos])
    (fun x hx k' hk' => by
      obtain ⟨t, ht, rfl⟩ := hmem x hx
      simp only [refPos, Tgt.ref.injEq] at hk'
      obtain ⟨hp', rfl⟩ := hk'
      have := ((hR t ht).2 hp').2
      simp only [refPos]
      exact_mod_cast this)
    (fun x hx i' hi' => by
      obtain ⟨t, ht, rfl⟩ := hmem x hx
      simp only [refPos, Tgt.ref.injEq] at hi'
      obtain ⟨hp', rfl⟩ := hi'
      have := (hR t ht).1 hp'
      simp only [refPos]
      exact ⟨this.1, by exact_mod_cast this.2⟩) h

/-- **`Desc.reached` marks the root.** -/
theorem reached_root (P : Prog) (hr : P.root < P.defs.size) : (Desc.reached P)[P.root]! = true := by
  let N := P.defs.size
  let G : Nat → Array Bool → Array Bool := fun i s =>
    if (!s[N - 1 - i]!) = true then s
    else if (P.defs[N - 1 - i]!).prim = true then s
    else (P.defs[N - 1 - i]!).body.nodes.foldl (fun b a => b.set! a.fn true) s
  have step : ∀ s b, b.size = N ∧ b[P.root]! = true → (G s b).size = N ∧ (G s b)[P.root]! = true := by
    intro s b ⟨hsz, hb⟩
    simp only [G]
    split
    · exact ⟨hsz, hb⟩
    split
    · exact ⟨hsz, hb⟩
    rw [← Array.foldl_toList]
    obtain ⟨m1, m2⟩ := marks_spec (P.defs[N - 1 - s]!).body.nodes.toList b
    exact ⟨m1.trans hsz, (m2 _).2 (.inr hb)⟩
  have run : ∀ (l : List Nat) b, b.size = N ∧ b[P.root]! = true →
      (l.foldl (fun b i => G i b) b).size = N ∧ (l.foldl (fun b i => G i b) b)[P.root]! = true := by
    intro l
    induction l with
    | nil => intro b h; exact h
    | cons i l ih => intro b h; exact ih _ (step i b h)
  have h0 : ((Array.replicate N false).set! P.root true).size = N ∧
      ((Array.replicate N false).set! P.root true)[P.root]! = true := by
    refine ⟨by rw [set!_size, Array.size_replicate], ?_⟩
    rw [set!_get]
    exact .inl ⟨rfl, by rw [Array.size_replicate]; exact hr⟩
  have := run (List.range' 0 N) _ h0
  unfold Desc.reached
  dsimp only
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  rw [forIn_id_yield _ _ G _ (fun a b => ?_)]
  · simpa using this.2
  · dsimp only [G]
    split
    · rfl
    · split
      · rfl
      · rw [Array.forIn_pure_yield_eq_foldl]; rfl

/-- What `Seg.split` makes of a `ShapeOk` segment: segments naming their `len` positions, each a position of the
segment, of pieces of positive count. -/
theorem split_shape {P : Prog} {d : Nat} {s g : Seg} (hs : ShapeOk P d s) (hg : g ∈ (Seg.split s).toList) :
    SegOk g ∧ (∀ x ∈ segPos g, x ∈ segPos s) ∧ ∀ q ∈ g.ps.toList, 0 < q.count := by
  have hx : ∀ x, x ∈ segPos g → x ∈ segPos s := fun x hx =>
    (mem_segPos_split s x).1 ⟨g, hg, hx⟩
  refine ⟨?_, hx, ?_⟩
  all_goals unfold Desc.Seg.split at hg; dsimp only at hg
  all_goals split at hg
  · rw [Array.toList_map, List.mem_map] at hg
    obtain ⟨q, -, rfl⟩ := hg
    show (posAll #[q]).length = q.count
    rw [posAll_singleton, pos_length]
  · simp only [List.mem_singleton] at hg
    subst hg
    unfold Desc.Seg.alike
    split <;> exact hs.1
  · rw [Array.toList_map, List.mem_map] at hg
    obtain ⟨q, hq, rfl⟩ := hg
    intro q' hq'
    simp only [Desc.Seg.one, List.mem_singleton] at hq'
    subst hq'
    unfold Desc.Seg.alike at hq
    split at hq <;> exact hs.2.2.1 _ hq
  · simp only [List.mem_singleton] at hg
    subst hg
    unfold Desc.Seg.alike
    split <;> exact hs.2.2.1

/-- What `Desc.memberRows` gives are pieces `throughPS` gives of the argument. -/
theorem memberRows_mem (c : Desc.Ctx) (node : Flock.Program.Node) (k i : Nat) (u : Desc.Ivs) {ps : Array Piece}
    (h : Desc.memberRows c node k i u = .ok (some ps)) :
    ∀ bp ∈ ps.toList, ∃ p, Desc.throughPS (c.arg k i) p = some bp := by
  let T : Array Piece → Prop := fun out => ∀ bp ∈ out.toList, ∃ p, Desc.throughPS (c.arg k i) p = some bp
  let I : Option (Option (Array Piece)) × Array Piece → Prop := fun st => (st.1 = none ∨ st.1 = some none) ∧ T st.2
  have hpush : ∀ out bp p, T out → Desc.throughPS (c.arg k i) p = some bp → T (out.push bp) := by
    intro out bp p ht hp x hx
    rw [Array.toList_push, List.mem_append, List.mem_singleton] at hx
    rcases hx with hx | rfl
    · exact ht x hx
    · exact ⟨p, hp⟩
  unfold Desc.memberRows at h
  extract_lets a out0 at h
  obtain ⟨st, hst, h⟩ := Walk.except_bind_ok h
  rw [← Array.forIn_toList] at hst
  have hfin := forIn_done I I _ ?_ u.toList _ st ⟨.inl rfl, fun x hx => by simp [out0] at hx⟩ hst
  · have hI : I st := hfin.elim id id
    obtain ⟨r, out⟩ := st
    obtain ⟨hr | hr, ht⟩ := hI
    · dsimp only at hr ht h
      subst hr
      simp only [pure, Except.pure, Except.ok.injEq, Option.some.injEq] at h
      subst h
      exact ht
    · dsimp only at hr ht h
      subst hr
      simp [pure, Except.pure] at h
  intro x s y hs hf
  obtain ⟨lo, hi⟩ := x
  extract_lets r out jp at hf
  dsimp (config := { zeta := false }) only at hf
  have hjp : ∀ rows y, jp rows = .ok y → (∃ s', y = .yield s' ∧ I s') ∨ (∃ s', y = .done s' ∧ I s') := by
    intro rows y hy
    simp only [jp] at hy
    obtain ⟨st2, hst2, hy⟩ := Walk.except_bind_ok hy
    rw [← Array.forIn_toList] at hst2
    have h2 := forIn_done I I _ ?_ rows.toList (none, out) st2 ⟨.inl rfl, hs.2⟩ hst2
    · have hI2 : I st2 := h2.elim id id
      obtain ⟨r2, out2⟩ := st2
      obtain ⟨hr | hr, ht⟩ := hI2
      · dsimp only at hr ht hy; subst hr
        simp only [pure, Except.pure, Except.ok.injEq] at hy
        exact .inl ⟨_, hy.symm, .inl rfl, ht⟩
      · dsimp only at hr ht hy; subst hr
        simp only [pure, Except.pure, Except.ok.injEq] at hy
        exact .inr ⟨_, hy.symm, .inr rfl, ht⟩
    intro p s2 y2 hs2 hf2
    split at hf2
    · rename_i bp hbp
      simp only [pure, Except.pure, Except.ok.injEq] at hf2
      exact .inl ⟨_, hf2.symm, .inl rfl, hpush _ bp p hs2.2 hbp⟩
    · simp only [pure, Except.pure, Except.ok.injEq] at hf2
      exact .inr ⟨_, hf2.symm, .inr rfl, hs2.2⟩
  split at hf
  · obtain ⟨rows, -, hf⟩ := Walk.except_bind_ok hf; exact hjp rows y hf
  · split at hf
    · obtain ⟨rows, -, hf⟩ := Walk.except_bind_ok hf; exact hjp rows y hf
    · have hf := walkAtMost_bind _ _ _ hf
      obtain ⟨rows, -, hf⟩ := Walk.except_bind_ok hf; exact hjp rows y hf
  · obtain ⟨rows, -, hf⟩ := Walk.except_bind_ok hf; exact hjp rows y hf

/-- The decoder's facts `RootSegs` reads, each one of every Program `Flock.Program.decode` gives whose order
`Prog.orderViolation` passes (`BodiesOk`): the root a composite without parameters, `Decoded`, `RetParts` and
`ScanOut`, every callee decoded before its caller, no reached Definition calling the root, and every reached
Definition's references `Resolve.BodyOk`. -/
def RootFacts (P : Prog) : Prop :=
  P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false ∧ (P.defs[P.root]!).body.params.size = 0 ∧ Decoded P ∧
  RetParts P ∧ (∀ m d, ScanOut (ctxOf P m d)) ∧
  (∀ d < P.defs.size, ∀ k < (P.defs[d]!).body.nodes.size, ((P.defs[d]!).body.nodes[k]!).fn < d) ∧
  (∀ d < P.defs.size, (Desc.reached P)[d]! = true →
    ∀ k < (P.defs[d]!).body.nodes.size, ((P.defs[d]!).body.nodes[k]!).fn ≠ P.root) ∧
  (∀ d < P.defs.size, (Desc.reached P)[d]! = true → Resolve.BodyOk P d)

/-- The root's callees' summaries hold `RetOk` (`memoOf_retOk`): each is reached (`reached_root`, `reached_closed`)
and decoded before the root. -/
theorem root_retOk (P : Prog) (memo : Array Desc.Summary) (hm : memoOf P = .ok memo) (hF : RootFacts P) :
    ∀ k node, (rootCtx P memo).node k = .ok node → RetOk P memo node.fn := by
  obtain ⟨hr, hp, -, hDec, hRP, hSO, hcf, hnr, hB⟩ := hF
  intro k node hn
  have hk := node_lt hn
  have hg := node_get hn
  change k < (P.defs[P.root]!).body.nodes.size at hk
  change (P.defs[P.root]!).body.nodes[k]! = node at hg
  have hlt := hcf P.root hr k hk
  rw [← hg]
  exact memoOf_retOk P memo hm hRP hDec hcf hnr hB hSO _ (by omega)
    (reached_closed P hcf P.root (reached_root P hr) hp k hk) (by omega)

/-- A loop inserting a position into a set at each step keeps a property every inserted position has. -/
theorem insertLoop_mem {α : Type} (G : Nat → Prop) (g : α → Nat) :
    ∀ (l : List α) (s0 s : Std.HashSet Nat), (∀ x ∈ s0, G x) → (∀ t ∈ l, G (g t)) →
      forIn (m := Except String) l s0 (fun t s => pure (ForInStep.yield (s.insert (g t)))) = .ok s →
      ∀ x ∈ s, G x
  | [], s0, s, h0, _, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h; exact h0
  | t :: l, s0, s, h0, hl, h => by
    rw [List.forIn_cons] at h
    exact insertLoop_mem G g l _ s (fun x hx => by
      rcases Std.HashSet.mem_insert.1 hx with e | hx
      · rw [← beq_iff_eq.1 e]; exact hl t List.mem_cons_self
      · exact h0 x hx) (fun t' ht' => hl t' (List.mem_cons_of_mem _ ht')) h

/-- `Desc.carriesRead`'s segments for scan node `k`'s carries are `ShapeOk`. -/
def CarriesShape (c : Desc.Ctx) : Prop :=
  ∀ k node segs, c.node k = .ok node → node.form = .scan → 2 ≤ node.n →
    Desc.carriesRead c k ((c.memo[node.fn]!).used[0]?.getD #[]) = .ok segs → ∀ s ∈ segs.toList, ShapeOk c.P c.d s

/-- **`Desc.carriesRead`'s segments hold `Q`** wherever the scans' steps' summaries hold `RetOk`, and the segments
a scan's `fromCallee` gives of its steps' returns, `throughArg` of its initial carry's leaves in range, and `down` of a
piece `throughPS` gives hold `Q`. -/
theorem carriesRead_of (Q : Seg → Prop) (c : Desc.Ctx)
    (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn)
    (hCS0 : ∀ k node, c.node k = .ok node → node.form = .scan → ∀ m a b segs, m < (c.body.nodes[k]!).members →
      (a < b → b ≤ (c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.count) →
      Desc.fromCallee c Flock.Steps.FUEL k m a b = .ok segs → ∀ s ∈ segs.toList, Q s)
    (hI : ∀ k node, c.node k = .ok node → 0 < node.members → ∀ lo hi segs,
      (∀ j, lo ≤ j → j < hi → 0 < (c.P.defs[node.fn]!).body.params.size ∧
        j < ((c.P.defs[node.fn]!).body.params[0]!).2.leaves) →
      Desc.throughArg c Flock.Steps.FUEL k 0 0 { tgt := .ref true 0, base := lo, stride := 1, len := hi - lo } =
        .ok segs → ∀ s ∈ segs.toList, Q s)
    (hPS : ∀ k node i p bp segs, c.node k = .ok node → Desc.throughPS (c.arg k i) p = some bp →
      Desc.down c Flock.Steps.FUEL bp = .ok segs → ∀ s ∈ segs.toList, Q s) :
    ∀ k node segs, c.node k = .ok node → node.form = .scan → 2 ≤ node.n →
      Desc.carriesRead c k ((c.memo[node.fn]!).used[0]?.getD #[]) = .ok segs → ∀ s ∈ segs.toList, Q s := by
  intro k node segs hn hform hn2 h
  have hCS := hCS0 k node hn hform
  unfold Desc.carriesRead at h
  obtain ⟨node', hn', h⟩ := Walk.except_bind_ok h
  rw [hn] at hn'
  cases hn'
  extract_lets fuel r T init out0 seen0 at h
  have hnode := node_get hn
  obtain ⟨hps, hpos, -, hag, hpp⟩ := hR k node hn
  have hlenR : (psPos r).length = (c.P.defs[node.fn]!).body.ret.count := by
    have := (hag (.top node.fn 0 none) rfl).length_eq
    simpa using this
  have hmem : node.members = node.n := by unfold Flock.Program.Node.members; rw [hform]; rfl
  have hm0 : 0 < node.members := by omega
  let G : Nat → Prop := fun l => 0 < (c.P.defs[node.fn]!).body.params.size ∧
    l < ((c.P.defs[node.fn]!).body.params[0]!).2.leaves
  have hcarry : ∀ h j' : Nat, (psPos r)[h]? = some (.ref true 0, (j' : Int)) → G j' := by
    intro h j' hx
    have := (hpos _ (List.mem_of_getElem? hx)).2.2 0 rfl
    exact ⟨this.1, by have h2 : ((j' : Nat) : Int) < _ := this.2; exact_mod_cast h2⟩
  have hinit : ∀ lo hi v, (∀ j, lo ≤ j → j < hi → G j) → init lo hi = .ok v → ∀ s ∈ v.toList, Q s :=
    fun lo hi v hlv hv => hI k node hn hm0 lo hi v hlv hv
  have h' := ok_bind_pure h
  rw [← Array.forIn_toList] at h'
  have hfa := forIn_app Q _ _ ?_ out0 segs h'
  · obtain ⟨out, hout, hQ⟩ := hfa
    rw [hout]
    intro s hs
    rw [Array.toList_append, List.mem_append] at hs
    exact hs.elim (fun hs => by simp [out0] at hs) (hQ s)
  intro x _ r' y hy
  obtain ⟨a, b⟩ := x
  dsimp (config := { zeta := false }) only at hy
  obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
  simp only [pure, Except.pure, Except.ok.injEq] at hy
  subst hy
  rw [← Array.forIn_toList] at hv
  have hfa := forIn_app Q _ _ ?_ r' v hv
  · obtain ⟨segs', hsegs', hA2⟩ := hfa
    exact ⟨segs', by rw [hsegs'], hA2⟩
  intro x hx r'' y hy
  split at hy
  · rename_i hxid
    obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    subst hy
    exact ⟨v, rfl, hinit _ _ v (fun j h1 h2 => hcarry j j (carryRuns_id r hps hpp a b x hx hxid j h1 h2)) hv⟩
  have hy := walkAtMost_bind _ _ _ hy
  obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
  simp only [pure, Except.pure, Except.ok.injEq] at hy
  subst hy
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hv
  have hfa := forIn_app Q _ _ ?_ r'' v hv
  · obtain ⟨segs', hsegs', hA2⟩ := hfa
    exact ⟨segs', by rw [hsegs'], hA2⟩
  intro j _ o y hy
  obtain ⟨ch, hch, hy⟩ := Walk.except_bind_ok hy
  have hok := chainOf_ok _ j ch hch
  extract_lets e upto jp at hy
  have he : e = ch.hist.size - 1 := rfl
  have hupto : upto = if ch.exit.isSome = true then min T e else T := rfl
  have hsz := hok.1
  obtain ⟨s1, hs1, hy⟩ := Walk.except_bind_ok hy
  dsimp only at hy
  have hbound : ∀ q, ch.exit = some q → ch.hist[e]! + 1 ≤ (c.P.defs[node.fn]!).body.ret.count := by
    intro q hq
    obtain ⟨hq', -⟩ := hok.2.2.2.1 q hq
    have := retAt_pos _ hps hpp hq'
    rw [← hlenR]
    exact (List.getElem?_eq_some_iff.1 this).1
  have hjp : ∀ sn y, (∀ l ∈ sn, G l) → jp () sn = .ok y →
      ∃ segs, y = .yield (o ++ segs) ∧ ∀ s ∈ segs.toList, Q s := by
    intro sn y hG hy
    simp only [jp] at hy
    obtain ⟨v1, hv1, hy⟩ := Walk.except_bind_ok hy
    rw [← Array.forIn_toList] at hv1
    obtain ⟨sA, hsA, hQA⟩ := forIn_app Q _ _ (fun l hl o' y' hy' => by
        obtain ⟨v', hv', hy'⟩ := Walk.except_bind_ok hy'
        simp only [pure, Except.pure, Except.ok.injEq] at hy'
        subst hy'
        refine ⟨v', rfl, hinit l (l + 1) v' (fun j' h1 h2 => ?_) hv'⟩
        rw [show j' = l by omega]
        exact hG l (Std.HashSet.mem_toArray.1 (Array.mem_toList_iff.1 (qsort_mem (fun x => x) _ hl)))) o v1 hv1
    subst hsA
    have hfin : ∀ extra, y = .yield (o ++ sA ++ extra) → (∀ s ∈ extra.toList, Q s) →
        ∃ segs, y = .yield (o ++ segs) ∧ ∀ s ∈ segs.toList, Q s :=
      fun extra hy hE => ⟨sA ++ extra, by rw [hy, Array.append_assoc], fun s hs => by
        rw [Array.toList_append, List.mem_append] at hs
        exact hs.elim (hQA s) (hE s)⟩
    split at hy
    · rename_i q hq0
      split at hy
      · rename_i heT
        split at hy
        · simp only [pure, Except.pure, Except.ok.injEq] at hy
          exact hfin #[] (by rw [← hy, Array.append_empty]) (by simp)
        · cases hy
        · rename_i i hqi
          split at hy
          · split at hy
            · rename_i bp hbp
              obtain ⟨v', hv', hy⟩ := Walk.except_bind_ok hy
              simp only [pure, Except.pure, Except.ok.injEq] at hy
              exact hfin v' hy.symm (hPS k node i _ bp v' hn hbp hv')
            · have hy := walkAtMost_bind _ _ _ hy
              obtain ⟨v', hv', hy⟩ := Walk.except_bind_ok hy
              simp only [pure, Except.pure, Except.ok.injEq] at hy
              simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at hv'
              obtain ⟨sB, hsB, hQB⟩ := forIn_app Q _ _ (fun it hit o' y' hy' => by
                  obtain ⟨v'', hv'', hy'⟩ := Walk.except_bind_ok hy'
                  simp only [pure, Except.pure, Except.ok.injEq] at hy'
                  subst hy'
                  refine ⟨v'', rfl, hCS it _ _ v'' ?_ (fun _ => by rw [hnode]; exact hbound q hq0) hv''⟩
                  rw [List.mem_range'_1] at hit
                  rw [hnode, hmem]
                  omega) _ v' hv'
              subst hsB
              exact hfin sB hy.symm hQB
          · obtain ⟨v', hv', hy⟩ := Walk.except_bind_ok hy
            simp only [pure, Except.pure, Except.ok.injEq] at hy
            exact hfin v' hy.symm
              (hCS 0 _ _ v' (by rw [hnode]; exact hm0) (fun _ => by rw [hnode]; exact hbound q hq0) hv')
      · simp only [pure, Except.pure, Except.ok.injEq] at hy
        exact hfin #[] (by rw [← hy, Array.append_empty]) (by simp)
    · simp only [pure, Except.pure, Except.ok.injEq] at hy
      exact hfin #[] (by rw [← hy, Array.append_empty]) (by simp)
  have hstep : ∀ t, 1 ≤ t → t < ch.hist.size → G ch.hist[t]! := by
    intro t h1 h2
    obtain ⟨q, hq, ht', hb⟩ := hok.2.2.1 (t - 1) (by omega)
    have := carryPos_of_retAt (hR k node hn) hq ht' hb
    rw [show t - 1 + 1 = t by omega] at this
    exact hcarry _ _ this
  have hG1 : ∀ l ∈ s1, G l := by
    simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at hs1
    refine insertLoop_mem G (fun t => ch.hist[t]!) _ _ _ (fun x hx => by simp [seen0] at hx) (fun t ht => ?_) hs1
    rw [List.mem_range'_1] at ht
    exact hstep t ht.1 (by omega)
  split at hy
  · rename_i hue
    have hx : ch.exit = none := by
      rcases hx : ch.exit with _ | q
      · rfl
      · exfalso
        rw [hupto, hx] at hue
        simp only [Option.isSome_some, ↓reduceIte] at hue
        omega
    split at hy
    · obtain ⟨s2, hs2, hy⟩ := Walk.except_bind_ok hy
      simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at hs2
      refine hjp s2 y (insertLoop_mem G (fun t => ch.hist[t]!) _ _ _ hG1 (fun t ht => ?_) hs2) hy
      rw [List.mem_range'_1] at ht
      obtain ⟨hc, q, hq, ht', hb⟩ := hok.2.2.2.2 hx
      by_cases htc : t = ch.cyc
      · rw [htc]; exact hcarry _ _ (carryPos_of_retAt (hR k node hn) hq ht' hb)
      · exact hstep t (by omega) (by omega)
    · obtain ⟨s2, hs2, hy⟩ := Walk.except_bind_ok hy
      simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at hs2
      refine hjp s2 y (insertLoop_mem G (fun t => ch.at t) _ _ _ hG1 (fun t ht => ?_) hs2) hy
      rw [List.mem_range'_1] at ht
      obtain ⟨q, hq, ht', hb⟩ := chainAt_cyc hok hx (t - 1)
      have := carryPos_of_retAt (hR k node hn) hq ht' hb
      rw [show t - 1 + 1 = t by omega] at this
      exact hcarry _ _ this
  · exact hjp s1 y hG1 hy

/-- **`Desc.carriesRead`'s segments are `ShapeOk`** wherever the scans' steps' summaries hold `RetOk`, and
`fromCallee`'s, `throughArg`'s and `down`'s of a piece `throughPS` gives are `ShapeOk` (`carriesRead_of`). -/
theorem carriesRead_shape (c : Desc.Ctx) (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn)
    (hCS : CalleeShape c Flock.Steps.FUEL) (hTS : ThroughShape c Flock.Steps.FUEL)
    (hPS : ∀ k node i p bp segs, c.node k = .ok node → Desc.throughPS (c.arg k i) p = some bp →
      Desc.down c Flock.Steps.FUEL bp = .ok segs → ∀ s ∈ segs.toList, ShapeOk c.P c.d s) : CarriesShape c :=
  carriesRead_of _ c hR (fun k _ _ _ => hCS k) (fun _ _ hn hm0 _ _ _ hlv h => init_shape c hn hTS hm0 hlv h) hPS

/-- **Every segment `down` gives a piece of a root node's argument, one `Desc.memberRows` gives, or one
`Desc.carriesRead` gives, is `ShapeOk`** (`piece_shape`, `throughPS_shape`, `carriesRead_shape`). -/
theorem rootShape (P : Prog) (memo : Array Desc.Summary) (hm : memoOf P = .ok memo) (hF : RootFacts P)
    {k : Nat} {node : Flock.Program.Node} (hn : (rootCtx P memo).node k = .ok node) {s : Seg}
    (hs : (∃ i, FromDown (rootCtx P memo) k i s) ∨ FromRows (rootCtx P memo) k node s ∨
      FromCarry (rootCtx P memo) k node s) : ShapeOk P P.root s := by
  have hR := root_retOk P memo hm hF
  obtain ⟨hr, -, -, hDec, -, hSO, -, -, hB⟩ := hF
  have hBr := hB P.root hr (reached_root P hr)
  rw [rootCtx_eq] at hn hs hR
  have hSh := shapes_decoded P memo hr hDec hBr hR (hSO memo P.root) Flock.Steps.FUEL
  have hA := argOk_decoded P memo hr hDec
  have hPS : ∀ k node i p bp segs, (ctxOf P memo P.root).node k = .ok node →
      Desc.throughPS ((ctxOf P memo P.root).arg k i) p = some bp →
      Desc.down (ctxOf P memo P.root) Flock.Steps.FUEL bp = .ok segs → ∀ s ∈ segs.toList, ShapeOk P P.root s := by
    intro k node i p bp segs hn hp hd
    have hk := node_lt hn
    by_cases hi' : i < ((ctxOf P memo P.root).body.nodes[k]!).args.size
    · exact throughPS_shape _ _ hSh.1 (hA k i hk hi') (ctxOf_argCount P memo P.root k i hk hi') ((hBr.1 k hk).2.1 i)
        hp hd
    · rw [throughPS_empty (ctxOf_arg_ge P memo P.root k i hi')] at hp
      cases hp
  have hk := node_lt hn
  rcases hs with ⟨i, lo, hi, q, hq, segs, hd, hs⟩ | ⟨i, u, ps, hmr, p, hp, segs, hd, hs⟩ | ⟨hfs, hn2, segs, hc, hs⟩
  · by_cases hi' : i < ((ctxOf P memo P.root).body.nodes[k]!).args.size
    · exact piece_shape _ _ hSh.1 (hA k i hk hi') (ctxOf_argCount P memo P.root k i hk hi') ((hBr.1 k hk).2.1 i) hq hd
        s (Array.mem_toList_iff.2 hs)
    · rw [pieces_empty (ctxOf_arg_ge P memo P.root k i hi')] at hq
      simp at hq
  · obtain ⟨p', hp'⟩ := memberRows_mem _ node k i u hmr p (Array.mem_toList_iff.2 hp)
    exact hPS k node i p' p segs hn hp' hd s (Array.mem_toList_iff.2 hs)
  · exact carriesRead_shape _ hR hSh.2.1 hSh.2.2.1 hPS k node segs hn hfs hn2 hc s (Array.mem_toList_iff.2 hs)

/-- A read `Seg.split` makes of a `ShapeOk` segment of the root's body, the root without parameters, is `SegGood`
wherever, of positive count, it names gates of one root node with their widths. -/
theorem segGood_of {P : Prog} {c : Circuit} {s g : Seg} (hp0 : (P.defs[P.root]!).body.params.size = 0)
    (hs : ShapeOk P P.root s) (hg : g ∈ (Seg.split s).toList)
    (hG : g.count ≠ 0 → ∃ k' < nodeCount P, ∀ w p, (Tgt.gate w, p) ∈ segPos g →
      w = c.w[p.toNat]! ∧ p.toNat ∈ nodeGates P k') :
    SegGood P c g := by
  obtain ⟨hok, hx, hpc⟩ := split_shape hs hg
  refine ⟨fun h0 => ?_, fun h0 => ⟨hpc, ?_⟩⟩
  · have := segPos_length g hok
    rw [h0] at this
    exact List.eq_nil_of_length_eq_zero this
  · obtain ⟨k', hk', hw⟩ := hG h0
    refine ⟨k', hk', fun x hxg => ?_⟩
    obtain ⟨t, p⟩ := x
    obtain ⟨hp0', hnk, hpar⟩ := hs.2.1 _ (hx _ hxg)
    rcases t with ⟨b, i⟩ | w
    · cases b
      · exact absurd rfl (hnk i)
      · have := (hpar i rfl).1
        rw [hp0] at this
        omega
    · obtain ⟨e1, e2⟩ := hw w p hxg
      refine ⟨p.toNat, ?_, e2⟩
      dsimp only at hp0'
      rw [← e1, Int.toNat_of_nonneg hp0']

/-- **The gates a root node's reads name are one root node's, with their widths.** On a Program `Desc.evaluate`
reads, with its summaries `memo` and `circuitOf`'s circuit `c`: every read `Seg.split` makes of a segment `down` gives
a piece of one of root node `k`'s arguments (or one `Desc.memberRows` gives), or `Desc.carriesRead` gives of its scan's
carries, of positive count, names gates `p` of width `c.w[p]` of one root node. -/
def RootGates : Prop :=
  ∀ (bytes : ByteArray) (P : Prog) (memo : Array Desc.Summary) (c : Circuit),
    Flock.Program.decode bytes = .ok P → P.orderViolation = none → Desc.rootLaidOut P = true →
    Desc.inputsAtRoot P = true → Desc.walkable P = true → memoOf P = .ok memo → circuitOf P = .ok c →
    ∀ k < nodeCount P, ∀ node, (rootCtx P memo).node k = .ok node →
      ∀ s, ((∃ i, FromDown (rootCtx P memo) k i s) ∨ FromRows (rootCtx P memo) k node s ∨
        FromCarry (rootCtx P memo) k node s) → ∀ g ∈ (Seg.split s).toList, g.count ≠ 0 →
        ∃ k' < nodeCount P, ∀ w p, (Tgt.gate w, p) ∈ segPos g → w = c.w[p.toNat]! ∧ p.toNat ∈ nodeGates P k'

/-- A segment of the root's body whose gates have their widths in `c` and lie in root nodes as `Seg.split` reads them:
each piece's in one root node, and, when its copies shift (`reps > 1`, `shift ≠ 0`), all its copies' in one. -/
def SegNodes (P : Prog) (c : Circuit) (s : Seg) : Prop :=
  (∀ w p, (Tgt.gate w, p) ∈ segPos s → w = c.w[p.toNat]!) ∧
  (∀ q ∈ s.ps.toList, q.count ≠ 0 →
    ∃ k' < nodeCount P, ∀ w p, (Tgt.gate w, p) ∈ pos q → p.toNat ∈ nodeGates P k') ∧
  (1 < s.reps → s.shift ≠ 0 → s.count ≠ 0 →
    ∃ k' < nodeCount P, ∀ w p, (Tgt.gate w, p) ∈ segPos s → p.toNat ∈ nodeGates P k')

/-- **`RootGates` segment by segment.** With `RootGates`' premises, every segment `down` gives a piece of one of root
node `k`'s arguments (or one `Desc.memberRows` gives), or `Desc.carriesRead` gives of its scan's carries, is
`SegNodes` (`rootNodes_of`, from `RootOneNode`). -/
def RootNodes : Prop :=
  ∀ (bytes : ByteArray) (P : Prog) (memo : Array Desc.Summary) (c : Circuit),
    Flock.Program.decode bytes = .ok P → P.orderViolation = none → Desc.rootLaidOut P = true →
    Desc.inputsAtRoot P = true → Desc.walkable P = true → memoOf P = .ok memo → circuitOf P = .ok c →
    ∀ k < nodeCount P, ∀ node, (rootCtx P memo).node k = .ok node →
      ∀ s, ((∃ i, FromDown (rootCtx P memo) k i s) ∨ FromRows (rootCtx P memo) k node s ∨
        FromCarry (rootCtx P memo) k node s) → SegNodes P c s

/-- A read `Seg.split` makes of a `SegNodes` segment, of positive count, names gates of one root node with their
widths. -/
theorem split_nodes {P : Prog} {c : Circuit} {s g : Seg} (hs : SegNodes P c s) (hg : g ∈ (Seg.split s).toList)
    (h0 : g.count ≠ 0) :
    ∃ k' < nodeCount P, ∀ w p, (Tgt.gate w, p) ∈ segPos g → w = c.w[p.toNat]! ∧ p.toNat ∈ nodeGates P k' := by
  have hx : ∀ x, x ∈ segPos g → x ∈ segPos s := fun x hx => (mem_segPos_split s x).1 ⟨g, hg, hx⟩
  suffices H : ∃ k' < nodeCount P, ∀ w p, (Tgt.gate w, p) ∈ segPos g → p.toNat ∈ nodeGates P k' by
    obtain ⟨k', hk', h⟩ := H
    exact ⟨k', hk', fun w p hp => ⟨hs.1 w p (hx _ hp), h w p hp⟩⟩
  unfold Desc.Seg.split at hg
  dsimp only at hg
  split at hg
  · rw [Array.toList_map, List.mem_map] at hg
    obtain ⟨q, hq, rfl⟩ := hg
    have hq' : q ∈ s.ps.toList := by
      unfold Desc.Seg.alike at hq
      split at hq <;> exact hq
    have hc : q.count ≠ 0 := by simpa [Desc.Seg.count, Desc.Seg.one] using h0
    obtain ⟨k', hk', h⟩ := hs.2.1 q hq' hc
    exact ⟨k', hk', fun w p hp => h w p (by rwa [segPos_one] at hp)⟩
  · rename_i hr
    simp only [List.mem_singleton] at hg
    subst hg
    unfold Desc.Seg.alike at hr h0 ⊢
    by_cases ha : (s.reps > 1 && s.shift == 0) = true
    · simp only [ha, ↓reduceIte] at hr
      simp at hr
    · simp only [ha, Bool.false_eq_true, ↓reduceIte] at hr h0 ⊢
      simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, not_and] at ha hr
      have h1 : 1 < s.reps := by
        rcases Nat.lt_or_ge 1 s.reps with h | h
        · exact h
        · exfalso
          apply h0
          unfold Desc.Seg.count
          rw [show s.reps = 0 by omega, Nat.zero_mul]
      exact hs.2.2 h1 (ha h1) h0

/-- **`RootGates` from `RootNodes`** (`split_nodes`). -/
theorem rootGates_of (hN : RootNodes) : RootGates := by
  intro bytes P memo c hd ho hl hi hw hm hc k hk node hn s hs g hg h0
  exact split_nodes (hN bytes P memo c hd ho hl hi hw hm hc k hk node hn s hs) hg h0

/-- **`RootSegs` from the decoder's facts and `RootGates`**: every segment is `ShapeOk` (`rootShape`), so each read
`Seg.split` makes of it is `SegGood` (`segGood_of`). -/
theorem rootSegs_of (hF : ∀ bytes P, Flock.Program.decode bytes = .ok P → P.orderViolation = none → RootFacts P)
    (hG : RootGates) : RootSegs := by
  intro bytes P memo c hd ho hl hi hw hm hc k hk node rs hn hrs hg s hs g hgs
  have hF' := hF bytes P hd ho
  exact segGood_of hF'.2.2.1 (rootShape P memo hm hF' hn hs) hgs
    (hG bytes P memo c hd ho hl hi hw hm hc k hk node hn s hs g hgs)

/-! ## A primitive root node's gates -/

/-- **A gate of a root node calling a primitive is that primitive's**: `Prog.gate` stops at the root's scope. -/
theorem gate_root_prim (P : Prog) (hl : Desc.rootLaidOut P = true) {g p : Nat} {ops : Array Nat}
    {sc : Flock.Program.Scope} (h : P.gate g = .ok (p, ops, sc)) {k : Nat} (hk : k < nodeCount P)
    (hg : g ∈ nodeGates P k) (hprim : (P.defs[((DescAgree.rootBody P).nodes[k]!).fn]!).prim = true) :
    p = ((DescAgree.rootBody P).nodes[k]!).fn ∧ sc = P.rootScope := by
  rw [DescAgree.mem_nodeGates] at hg
  obtain ⟨-, -, -, hmono⟩ := DescAgree.rootLaidOut_spec P hl
  unfold Flock.Program.Prog.gate at h
  by_cases hge : g ≥ (P.defs[P.root]!).gates
  · simp [hge, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  simp only [hge, ite_false, bind, Except.bind] at h
  have hF : Flock.Steps.FUEL = (Flock.Steps.FUEL - 1) + 1 := by unfold Flock.Steps.FUEL; omega
  rw [hF, Flock.Steps.run.eq_2] at h
  generalize Flock.Steps.FUEL - 1 = fuel at h
  rcases hs : Flock.Program.gateStep P g P.rootScope with e | x
  · simp [hs, bind, Except.bind] at h
  obtain ⟨k0, hf, hx⟩ := DescAgree.gateStep_eq P g P.rootScope x hs
  obtain ⟨hk0, hp0⟩ := DescAgree.find?_range_some hf
  have hd : P.rootScope.d = P.root := rfl
  have ho : P.rootScope.off = 0 := rfl
  simp only [hd, ho, Nat.sub_zero, Bool.and_eq_true, decide_eq_true_eq] at hk0 hp0
  have hk0K : k0 < nodeCount P := hk0
  have hkk : k0 = k := DescAgree.node_unique (rootOffs P) (nodeCount P) hmono hk0K hk hp0 hg
  subst hkk
  simp only [hd] at hx
  rcases hx with ⟨-, ops', rfl⟩ | ⟨hprim', -, -, -⟩
  · simp only [hs, bind, Except.bind, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    exact ⟨h.1.symm, h.2.2.symm⟩
  · simp only [DescAgree.rootBody] at hprim
    rw [hprim] at hprim'
    cases hprim'

/-- **A gate of a root node calling a primitive has that primitive's width** in `circuitOf`'s circuit. -/
theorem w_root_prim (P : Prog) {c : Circuit} (hc : circuitOf P = .ok c) (hl : Desc.rootLaidOut P = true) {k g : Nat}
    (hk : k < nodeCount P) (hg : g ∈ nodeGates P k)
    (hprim : (P.defs[((DescAgree.rootBody P).nodes[k]!).fn]!).prim = true) :
    (P.defs[((DescAgree.rootBody P).nodes[k]!).fn]!).ret = .value c.w[g]! := by
  have hgl : g < gates P := by
    obtain ⟨-, -, hlast, hmono⟩ := DescAgree.rootLaidOut_spec P hl
    rw [DescAgree.mem_nodeGates] at hg
    have := DescAgree.offs_mono (rootOffs P) (nodeCount P) hmono (k + 1) (nodeCount P) (by omega) le_rfl
    omega
  obtain ⟨p, sc, hp, hw, -⟩ := circuitOf_gate hc hgl
  obtain ⟨rfl, -⟩ := gate_root_prim P hl hp hk hg hprim
  exact hw

/-! ## Gate widths -/

/-- Gate `j` of an activation of Definition `d` has width `w`, located as `Prog.gate` locates it: the node holding it
calls a primitive returning `.value w`, or a composite whose member's activation holding it has it of width `w`. -/
inductive LocW (P : Prog) : Nat → Nat → Nat → Prop
  | prim {d j k w : Nat} :
      (Array.range (P.defs[d]!).body.nodes.size).find? (fun k =>
        (P.defs[d]!).body.offsets[k]! ≤ j && j < (P.defs[d]!).body.offsets[k + 1]!) = some k →
      (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).prim = true →
      (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).ret = .value w → LocW P d j w
  | comp {d j k w : Nat} (m : Nat) :
      (Array.range (P.defs[d]!).body.nodes.size).find? (fun k =>
        (P.defs[d]!).body.offsets[k]! ≤ j && j < (P.defs[d]!).body.offsets[k + 1]!) = some k →
      (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).prim = false →
      m = (if ((P.defs[d]!).body.nodes[k]!).form == .prim || ((P.defs[d]!).body.nodes[k]!).form == .call then 0
        else (j - (P.defs[d]!).body.offsets[k]!) / (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates) →
      LocW P ((P.defs[d]!).body.nodes[k]!).fn
        (j - (P.defs[d]!).body.offsets[k]! - m * (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates) w →
      LocW P d j w

/-- A gate has one width. -/
theorem locW_unique {P : Prog} {d j w w' : Nat} (h : LocW P d j w) (h' : LocW P d j w') : w = w' := by
  induction h generalizing w' with
  | prim hf hp hr =>
    cases h' with
    | prim hf' _ hr' =>
      rw [hf] at hf'
      cases hf'
      rw [hr] at hr'
      cases hr'
      rfl
    | comp _ hf' hp' _ _ =>
      rw [hf] at hf'
      cases hf'
      rw [hp] at hp'
      cases hp'
  | comp m hf hp hm _ ih =>
    cases h' with
    | prim hf' hp' _ =>
      rw [hf] at hf'
      cases hf'
      rw [hp] at hp'
      cases hp'
    | comp m' hf' _ hm' hrec' =>
      rw [hf] at hf'
      cases hf'
      have e : m = m' := by rw [hm, hm']
      subst e
      exact ih hrec'

/-- `gateStep`'s step into a composite node's member's activation. -/
theorem gateStep_inr (P : Prog) (g : Nat) (sc sc' : Flock.Program.Scope)
    (h : Flock.Program.gateStep P g sc = .ok (.inr sc')) :
    ∃ k, (Array.range (P.defs[sc.d]!).body.nodes.size).find? (fun k =>
        (P.defs[sc.d]!).body.offsets[k]! ≤ g - sc.off && g - sc.off < (P.defs[sc.d]!).body.offsets[k + 1]!) = some k ∧
      (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = false ∧
      sc' = P.child sc k (if ((P.defs[sc.d]!).body.nodes[k]!).form == .prim ||
          ((P.defs[sc.d]!).body.nodes[k]!).form == .call then 0
        else (g - sc.off - (P.defs[sc.d]!).body.offsets[k]!) / (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates) := by
  unfold Flock.Program.gateStep at h
  simp only [bind, Except.bind] at h
  split at h
  · rename_i k hk
    refine ⟨k, hk, ?_⟩
    by_cases hp : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true
    · simp only [hp, ite_true] at h
      split at h
      · cases h
      · simp [pure, Except.pure] at h
    · simp only [hp, Bool.false_eq_true, ite_false, pure, Except.pure, Except.ok.injEq, Sum.inr.injEq] at h
      exact ⟨by simpa using hp, h.symm⟩
  · simp [throw, throwThe, MonadExceptOf.throw] at h

/-- **`Prog.gate`'s walk locates a gate of its width**: from an activation, the primitive it ends at returns the width
`LocW` gives the gate there. -/
theorem run_locW (P : Prog) (g : Nat) : ∀ (n : Nat) (sc : Flock.Program.Scope) {p : Nat} {ops : Array Nat}
    {sc' : Flock.Program.Scope}, Flock.Steps.run "gate location" (Flock.Program.gateStep P g) n sc = .ok (p, ops, sc') →
    ∀ w, (P.defs[p]!).ret = .value w → LocW P sc.d (g - sc.off) w
  | 0, _, _, _, _, h => by simp [Flock.Steps.run, throw, throwThe, MonadExceptOf.throw] at h
  | n + 1, sc, p, ops, sc', h => by
    intro w hw
    rw [Flock.Steps.run.eq_2] at h
    rcases hs : Flock.Program.gateStep P g sc with e | (b | a')
    · simp [hs, bind, Except.bind] at h
    · simp only [hs, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
      subst h
      obtain ⟨k, hk, hx⟩ := DescAgree.gateStep_eq P g sc _ hs
      rcases hx with ⟨hp, ops', he⟩ | ⟨-, sc'', he, -⟩
      · simp only [Sum.inl.injEq, Prod.mk.injEq] at he
        obtain ⟨rfl, -, -⟩ := he
        exact LocW.prim hk hp hw
      · cases he
    · simp only [hs, bind, Except.bind] at h
      obtain ⟨k, hk, hp, rfl⟩ := gateStep_inr P g sc a' hs
      have ih := run_locW P g n _ h w hw
      refine LocW.comp _ hk hp rfl ?_
      have e : ∀ a b c d : Nat, a - (b + c + d) = a - b - c - d := by intros; omega
      rw [← e]
      exact ih

/-- **A root gate's width in `circuitOf`'s circuit is the one `LocW` gives it.** -/
theorem locW_root {P : Prog} {c : Circuit} (hc : circuitOf P = .ok c) {g : Nat} (hg : g < gates P) :
    LocW P P.root g c.w[g]! := by
  obtain ⟨p, sc, hp, hw, -⟩ := circuitOf_gate hc hg
  unfold Flock.Program.Prog.gate at hp
  by_cases hge : g ≥ (P.defs[P.root]!).gates
  · simp [hge, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at hp
  simp only [hge, ite_false, bind, Except.bind] at hp
  have h := run_locW P g _ _ hp _ hw
  have h0 : g - P.rootScope.off = g := Nat.sub_zero g
  rw [h0] at h
  exact h

/-- Node `k` is the one `gateStep`'s search finds for a gate of it, offsets nondecreasing. -/
theorem find_node {offs : Array Nat} {K k x : Nat} (hmono : ∀ k' < K, offs[k']! ≤ offs[k' + 1]!) (hk : k < K)
    (hx : offs[k]! ≤ x ∧ x < offs[k + 1]!) :
    (Array.range K).find? (fun k' => offs[k']! ≤ x && x < offs[k' + 1]!) = some k := by
  rcases hs : (Array.range K).find? (fun k' => offs[k']! ≤ x && x < offs[k' + 1]!) with _ | k'
  · exfalso
    have := Array.find?_eq_none.1 hs k (Array.mem_range.2 hk)
    simp only [Bool.and_eq_true, decide_eq_true_eq, not_and, not_lt] at this
    exact absurd hx.2 (Nat.not_lt.2 (this hx.1))
  · obtain ⟨hk', hpk⟩ := DescAgree.find?_range_some hs
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hpk
    rw [DescAgree.node_unique _ _ hmono hk' hk hpk hx]

/-- **A primitive node's gate**: in a body of Definition `d` whose offsets are nondecreasing, a gate of node `k`
calling a primitive returning `.value w` has width `w`. -/
theorem locW_prim (P : Prog) {d k x w : Nat}
    (hmono : ∀ k' < (P.defs[d]!).body.nodes.size,
      (P.defs[d]!).body.offsets[k']! ≤ (P.defs[d]!).body.offsets[k' + 1]!)
    (hk : k < (P.defs[d]!).body.nodes.size) (hp : (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).prim = true)
    (hw : (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).ret = .value w)
    (hx : (P.defs[d]!).body.offsets[k]! ≤ x ∧ x < (P.defs[d]!).body.offsets[k + 1]!) : LocW P d x w :=
  LocW.prim (find_node hmono hk hx) hp hw

/-- **A member's gate in its caller**: in a laid-out body of Definition `d`, gate `j` of member `m` of composite node
`k`'s activation is gate `offsets[k] + m * gates + j` of `d`'s, of its width. -/
theorem locW_child (P : Prog) {d k m j w : Nat}
    (hoff : ∀ k' < (P.defs[d]!).body.nodes.size, (P.defs[d]!).body.offsets[k' + 1]! =
      (P.defs[d]!).body.offsets[k']! + ((P.defs[d]!).body.nodes[k']!).members *
        (P.defs[((P.defs[d]!).body.nodes[k']!).fn]!).gates)
    (hk : k < (P.defs[d]!).body.nodes.size) (hp : (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).prim = false)
    (hm : m < ((P.defs[d]!).body.nodes[k]!).members) (hj : j < (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates)
    (h : LocW P ((P.defs[d]!).body.nodes[k]!).fn j w) :
    LocW P d ((P.defs[d]!).body.offsets[k]! + m * (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates + j) w := by
  have hmono : ∀ k' < (P.defs[d]!).body.nodes.size,
      (P.defs[d]!).body.offsets[k']! ≤ (P.defs[d]!).body.offsets[k' + 1]! := fun k' hk' => by
    rw [hoff k' hk']; exact Nat.le_add_right _ _
  have hxk : (P.defs[d]!).body.offsets[k]! ≤
        (P.defs[d]!).body.offsets[k]! + m * (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates + j ∧
      (P.defs[d]!).body.offsets[k]! + m * (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates + j <
        (P.defs[d]!).body.offsets[k + 1]! := by
    rw [hoff k hk]
    have hle := Nat.mul_le_mul_right (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates hm
    rw [Nat.succ_mul] at hle
    omega
  have hf := find_node hmono hk hxk
  have hG : 0 < (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates := by omega
  have hx : ((P.defs[d]!).body.offsets[k]! + m * (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates + j -
      (P.defs[d]!).body.offsets[k]!) / (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates = m := by
    rw [show (P.defs[d]!).body.offsets[k]! + m * (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates + j -
        (P.defs[d]!).body.offsets[k]! = j + m * (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates by omega,
      Nat.add_mul_div_right _ _ hG, Nat.div_eq_of_lt hj, Nat.zero_add]
  refine LocW.comp m hf hp ?_ ?_
  · rw [hx]
    unfold Flock.Program.Node.members at hm
    revert hm
    cases ((P.defs[d]!).body.nodes[k]!).form <;> intro hm
    · exact (Nat.lt_one_iff.1 hm : m = 0)
    · exact (Nat.lt_one_iff.1 hm : m = 0)
    · rfl
    · rfl
  · rw [show (P.defs[d]!).body.offsets[k]! + m * (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates + j -
        (P.defs[d]!).body.offsets[k]! - m * (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates = j by omega]
    exact h

/-- **A gate of a width is in a node**: one of the body's nodes holds it. -/
theorem locW_node {P : Prog} {d j w : Nat} (h : LocW P d j w) :
    ∃ k < (P.defs[d]!).body.nodes.size, (P.defs[d]!).body.offsets[k]! ≤ j ∧ j < (P.defs[d]!).body.offsets[k + 1]! := by
  cases h with
  | prim hf _ _ =>
    obtain ⟨hk, hp⟩ := DescAgree.find?_range_some hf
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hp
    exact ⟨_, hk, hp⟩
  | comp _ hf _ _ _ =>
    obtain ⟨hk, hp⟩ := DescAgree.find?_range_some hf
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hp
    exact ⟨_, hk, hp⟩

/-- A gate of a width is one of the Definition's, its body laid out. -/
theorem locW_lt {P : Prog} {d j w : Nat} (h : LocW P d j w)
    (hoff : ∀ k' < (P.defs[d]!).body.nodes.size, (P.defs[d]!).body.offsets[k' + 1]! =
      (P.defs[d]!).body.offsets[k']! + ((P.defs[d]!).body.nodes[k']!).members *
        (P.defs[((P.defs[d]!).body.nodes[k']!).fn]!).gates)
    (hlast : (P.defs[d]!).body.offsets[(P.defs[d]!).body.nodes.size]! = (P.defs[d]!).gates) :
    j < (P.defs[d]!).gates := by
  obtain ⟨k, hk, -, h2⟩ := locW_node h
  have hmono : ∀ k' < (P.defs[d]!).body.nodes.size,
      (P.defs[d]!).body.offsets[k']! ≤ (P.defs[d]!).body.offsets[k' + 1]! := fun k' hk' => by
    have := hoff k' hk'
    exact this ▸ Nat.le_add_right _ _
  have := DescAgree.offs_mono _ _ hmono (k + 1) (P.defs[d]!).body.nodes.size (by omega) le_rfl
  omega

/-- A segment of Definition `d`'s body naming its gates with their widths (`LocW`). -/
def GateW (P : Prog) (d : Nat) (s : Seg) : Prop := ∀ w p, (Tgt.gate w, p) ∈ segPos s → LocW P d p.toNat w

/-- `down` keeps a gate's piece as it is, with its widths. -/
theorem downW_gate (c : Desc.Ctx) (fuel : Nat) (q : Piece) {w : Nat} (ht : q.tgt = .gate w)
    (hq : ∀ w' p, (Tgt.gate w', p) ∈ pos q → LocW c.P c.d p.toNat w') {segs : Array Seg}
    (h : Desc.down c (fuel + 1) q = .ok segs) : ∀ s ∈ segs.toList, GateW c.P c.d s := by
  rw [down_gate c fuel q ht] at h
  injection h with h
  subst h
  intro s hs
  have e : s = Seg.one q := by simpa using hs
  subst e
  intro w' p hx
  rw [segPos_one] at hx
  exact hq w' p hx

/-- `down` keeps a parameter's piece as it is, naming no gate. -/
theorem downW_param (c : Desc.Ctx) (fuel : Nat) (q : Piece) {i : Nat} (ht : q.tgt = .ref true i)
    {segs : Array Seg} (h : Desc.down c (fuel + 1) q = .ok segs) : ∀ s ∈ segs.toList, GateW c.P c.d s := by
  rw [down_param c fuel q ht] at h
  injection h with h
  subst h
  intro s hs
  have e : s = Seg.one q := by simpa using hs
  subst e
  intro w' p hx
  rw [segPos_one] at hx
  have := mem_pos_tgt hx
  rw [ht] at this
  cases this

/-- A primitive node's piece names the node's gate, of the primitive's width, in a laid-out body. -/
theorem downW_prim (c : Desc.Ctx) (fuel : Nat) (q : Piece) {k : Nat} (ht : q.tgt = .ref false k)
    (hff : q.flat = q) {node : Flock.Program.Node} (hn : c.node k = .ok node) (hform : node.form = .prim) {w : Nat}
    (hret : (c.P.defs[node.fn]!).ret = .value w) (hp : (c.P.defs[node.fn]!).prim = true)
    (hg1 : (c.P.defs[node.fn]!).gates = 1)
    (hoff : ∀ k' < c.body.nodes.size, c.body.offsets[k' + 1]! =
      c.body.offsets[k']! + (c.body.nodes[k']!).members * (c.P.defs[(c.body.nodes[k']!).fn]!).gates)
    {segs : Array Seg} (h : Desc.down c (fuel + 1) q = .ok segs) : ∀ s ∈ segs.toList, GateW c.P c.d s := by
  have hk := node_lt hn
  have hg := node_get hn
  rw [down_prim c fuel q ht hff hn hform hret] at h
  injection h with h
  subst h
  split
  · intro s hs
    simp at hs
  · intro s hs
    have e : s = Seg.one { tgt := .gate w, base := (c.body.offsets[k]! : Int), stride := 0, len := q.count } := by
      simpa using hs
    subst e
    intro w' p hx
    rw [segPos_one] at hx
    unfold pos at hx
    simp only [List.mem_flatMap, List.mem_map, List.mem_range, Prod.mk.injEq, Tgt.gate.injEq] at hx
    obtain ⟨r, -, t, -, rfl, rfl⟩ := hx
    simp only [Int.mul_zero, Int.add_zero, Int.toNat_natCast]
    have hmono : ∀ k' < (c.P.defs[c.d]!).body.nodes.size,
        (c.P.defs[c.d]!).body.offsets[k']! ≤ (c.P.defs[c.d]!).body.offsets[k' + 1]! := fun k' hk' => by
      have := hoff k' hk'
      exact this ▸ Nat.le_add_right _ _
    have hm1 : node.members = 1 := by
      unfold Flock.Program.Node.members
      rw [hform]
      rfl
    have hx : c.body.offsets[k]! ≤ c.body.offsets[k]! ∧ c.body.offsets[k]! < c.body.offsets[k + 1]! := by
      rw [hoff k hk, hg, hm1, hg1]
      omega
    rw [← hg] at hp hret
    exact locW_prim c.P hmono hk hp hret hx

/-- A batch of a primitive's piece names its members' gates, of the primitive's width, in a laid-out body: its
positions at or past 0 and within the node's output, of no more leaves than members. -/
theorem downW_batchPrim (c : Desc.Ctx) (fuel : Nat) (q : Piece) {k : Nat} (ht : q.tgt = .ref false k)
    (hff : q.flat = q) {node : Flock.Program.Node} (hn : c.node k = .ok node) (hform : node.form = .batch)
    (hp : (c.P.defs[node.fn]!).prim = true) {w : Nat} (hret : (c.P.defs[node.fn]!).ret = .value w)
    (hg1 : (c.P.defs[node.fn]!).gates = 1)
    (hoff : ∀ k' < c.body.nodes.size, c.body.offsets[k' + 1]! =
      c.body.offsets[k']! + (c.body.nodes[k']!).members * (c.P.defs[(c.body.nodes[k']!).fn]!).gates)
    (hq : ∀ x ∈ pos q, 0 ≤ x.2) (hF : ∀ x ∈ pos q, Fits c x) (hout : node.out.leaves ≤ node.members)
    {segs : Array Seg} (h : Desc.down c (fuel + 1) q = .ok segs) : ∀ s ∈ segs.toList, GateW c.P c.d s := by
  have hk := node_lt hn
  have hg := node_get hn
  rw [down_batchPrim c fuel q ht hff hn hform hp hret] at h
  injection h with h
  subst h
  split
  · intro s hs
    simp at hs
  · intro s hs
    have e : s = Seg.one { q with tgt := .gate w, base := (c.body.offsets[k]! : Int) + q.base } := by
      simpa using hs
    subst e
    intro w' p hx
    rw [segPos_one] at hx
    unfold pos at hx
    simp only [List.mem_flatMap, List.mem_map, List.mem_range, Prod.mk.injEq, Tgt.gate.injEq] at hx
    obtain ⟨r, hr, t, htl, rfl, rfl⟩ := hx
    have hmem : (q.tgt, q.base + (r : Int) * q.rowStride + (t : Int) * q.stride) ∈ pos q := by
      unfold pos
      simp only [List.mem_flatMap, List.mem_map, List.mem_range]
      exact ⟨r, hr, t, htl, rfl⟩
    have h0 := hq _ hmem
    have h1 := hF _ hmem k ht
    rw [hg] at h1
    obtain ⟨n, hn'⟩ := Int.eq_ofNat_of_zero_le h0
    have hpn : (c.body.offsets[k]! : Int) + q.base + (r : Int) * q.rowStride + (t : Int) * q.stride =
        ((c.body.offsets[k]! + n : Nat) : Int) := by
      push_cast
      simp only at hn'
      linarith
    rw [hpn, Int.toNat_natCast]
    have hlt : n < node.members := by
      have : (n : Int) < node.out.leaves := by simp only at hn' h1; rw [← hn']; exact h1
      omega
    have hmono : ∀ k' < (c.P.defs[c.d]!).body.nodes.size,
        (c.P.defs[c.d]!).body.offsets[k']! ≤ (c.P.defs[c.d]!).body.offsets[k' + 1]! := fun k' hk' => by
      have := hoff k' hk'
      exact this ▸ Nat.le_add_right _ _
    have hx : c.body.offsets[k]! ≤ c.body.offsets[k]! + n ∧ c.body.offsets[k]! + n < c.body.offsets[k + 1]! := by
      rw [hoff k hk, hg, hg1]
      omega
    rw [← hg] at hp hret
    exact locW_prim c.P hmono hk hp hret hx

/-- **A callee's segment as member `m`'s** (`fromCallee`'s `shiftGates`): naming the callee's gates with their widths,
shifted to the member's first gate it names the caller's with theirs, in a laid-out body. -/
theorem gateW_shift (P : Prog) {d k m : Nat} {o : Int}
    (hoff : ∀ k' < (P.defs[d]!).body.nodes.size, (P.defs[d]!).body.offsets[k' + 1]! =
      (P.defs[d]!).body.offsets[k']! + ((P.defs[d]!).body.nodes[k']!).members *
        (P.defs[((P.defs[d]!).body.nodes[k']!).fn]!).gates)
    (hk : k < (P.defs[d]!).body.nodes.size) (hp : (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).prim = false)
    (hm : m < ((P.defs[d]!).body.nodes[k]!).members)
    (ho : o = (((P.defs[d]!).body.offsets[k]! + m * (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates : Nat) : Int))
    {s : Seg} (hW : GateW P ((P.defs[d]!).body.nodes[k]!).fn s)
    (hin : ∀ w p, (Tgt.gate w, p) ∈ segPos s → 0 ≤ p ∧ p < ((P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates : Int)) :
    GateW P d { ps := Desc.shiftGates o s.ps, len := s.len, reps := s.reps, shift := s.shift } := by
  intro w p hx
  rw [segPos_shift] at hx
  obtain ⟨⟨t, v⟩, hxs, he⟩ := List.mem_map.1 hx
  cases t with
  | ref b i => cases he
  | gate w0 =>
    simp only [shiftPos, Prod.mk.injEq, Tgt.gate.injEq] at he
    obtain ⟨rfl, rfl⟩ := he
    have hl := hW w0 v hxs
    obtain ⟨h0, h1⟩ := hin w0 v hxs
    obtain ⟨n, rfl⟩ := Int.eq_ofNat_of_zero_le h0
    rw [Int.toNat_natCast] at hl
    rw [ho, ← Int.natCast_add, Int.toNat_natCast, Nat.add_comm n]
    exact locW_child P hoff hk hp hm (by omega) hl

/-- **A callee's piece of gates as member `m`'s** (`fromCallee`'s `{ g with base := o + g.base }`): of the callee's
gates with their widths, it names the caller's with theirs, in a laid-out body. -/
theorem gateW_oneShift (P : Prog) {d k m : Nat} {o : Int}
    (hoff : ∀ k' < (P.defs[d]!).body.nodes.size, (P.defs[d]!).body.offsets[k' + 1]! =
      (P.defs[d]!).body.offsets[k']! + ((P.defs[d]!).body.nodes[k']!).members *
        (P.defs[((P.defs[d]!).body.nodes[k']!).fn]!).gates)
    (hk : k < (P.defs[d]!).body.nodes.size) (hp : (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).prim = false)
    (hm : m < ((P.defs[d]!).body.nodes[k]!).members)
    (ho : o = (((P.defs[d]!).body.offsets[k]! + m * (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates : Nat) : Int))
    {g : Piece} (hW : GateW P ((P.defs[d]!).body.nodes[k]!).fn (Seg.one g))
    (hin : ∀ w p, (Tgt.gate w, p) ∈ pos g → 0 ≤ p ∧ p < ((P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates : Int)) :
    GateW P d (Seg.one { g with base := o + g.base }) := by
  intro w p hx
  rw [segPos_one] at hx
  unfold pos at hx
  simp only [List.mem_flatMap, List.mem_map, List.mem_range, Prod.mk.injEq] at hx
  obtain ⟨r, hr, t, htl, hgt, rfl⟩ := hx
  have hmem : (Tgt.gate w, g.base + (r : Int) * g.rowStride + (t : Int) * g.stride) ∈ pos g := by
    unfold pos
    simp only [List.mem_flatMap, List.mem_map, List.mem_range]
    exact ⟨r, hr, t, htl, by rw [hgt]⟩
  have hl := hW w _ (by rw [segPos_one]; exact hmem)
  obtain ⟨h0, h1⟩ := hin w _ hmem
  obtain ⟨n, hn⟩ := Int.eq_ofNat_of_zero_le h0
  rw [hn, Int.toNat_natCast] at hl
  have e : o + g.base + (r : Int) * g.rowStride + (t : Int) * g.stride =
      (((P.defs[d]!).body.offsets[k]! + m * (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates + n : Nat) : Int) := by
    rw [ho]
    push_cast
    linarith
  rw [e, Int.toNat_natCast]
  exact locW_child P hoff hk hp hm (by omega) hl

/-- Definition `d`'s body laid out: each node's gates its members' activations', end to end. -/
def Laid (P : Prog) (d : Nat) : Prop :=
  ∀ k' < (P.defs[d]!).body.nodes.size, (P.defs[d]!).body.offsets[k' + 1]! =
    (P.defs[d]!).body.offsets[k']! + ((P.defs[d]!).body.nodes[k']!).members *
      (P.defs[((P.defs[d]!).body.nodes[k']!).fn]!).gates

/-- Definition `f`'s summarized return names its gates with their widths (`LocW`). -/
def RetW (P : Prog) (memo : Array Desc.Summary) (f : Nat) : Prop :=
  ∀ w p, (Tgt.gate w, p) ∈ psPos (memo[f]!).ret → LocW P f p.toNat w

/-- What `throughArg` gives of member `m`'s parameter `i`'s leaves `g`, each in range, names the body's gates with
their widths. -/
def ThroughW (c : Desc.Ctx) (fuel : Nat) : Prop :=
  ∀ k m i g segs, m < (c.body.nodes[k]!).members → g.tgt = .ref true i →
    (∀ x ∈ pos g, 0 ≤ x.2 ∧ i < (c.P.defs[(c.body.nodes[k]!).fn]!).body.params.size ∧
      x.2 < ((((c.P.defs[(c.body.nodes[k]!).fn]!).body.params[i]!).2.leaves : Int))) →
    Desc.throughArg c fuel k m i g = .ok segs → ∀ s ∈ segs.toList, GateW c.P c.d s

/-- What `fromCallee` gives of member `m` of a composite node names the body's gates with their widths. -/
def CalleeW (c : Desc.Ctx) (fuel : Nat) : Prop :=
  ∀ k m a b segs, m < (c.body.nodes[k]!).members → (c.P.defs[(c.body.nodes[k]!).fn]!).prim = false →
    Desc.fromCallee c fuel k m a b = .ok segs → ∀ s ∈ segs.toList, GateW c.P c.d s

/-- **`fromCallee`'s width step**: one step up, `fromCallee`'s segments name the body's gates with their widths
wherever its callees' summaries hold `RetOk` and name theirs (`RetW`), the bodies are laid out, and `throughArg`'s
segments one step down name the body's gates with their widths, each of a copy at least (`ThroughReps`). -/
theorem calleeW_succ (c : Desc.Ctx) (fuel : Nat)
    (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn)
    (hRW : ∀ k node, c.node k = .ok node → RetW c.P c.memo node.fn) (hL : Laid c.P c.d)
    (hLF : ∀ k node, c.node k = .ok node → (c.P.defs[node.fn]!).prim = false → Laid c.P node.fn ∧
      (c.P.defs[node.fn]!).body.offsets[(c.P.defs[node.fn]!).body.nodes.size]! = (c.P.defs[node.fn]!).gates)
    (hTW : ThroughW c fuel) (hTR : ThroughReps c) : CalleeW c (fuel + 1) := by
  intro k m a b segs hm hp h
  rw [Desc.fromCallee.eq_2] at h
  obtain ⟨node, hn, h⟩ := Walk.except_bind_ok h
  have hnode : c.body.nodes[k]! = node := node_get hn
  have hk : k < c.body.nodes.size := node_lt hn
  subst hnode
  dsimp only at h
  have h' := ok_bind_pure h
  rw [← Array.forIn_toList] at h'
  obtain ⟨hps, hpos, hgo, -, hpp⟩ := hR k _ hn
  have hrw := hRW k _ hn
  obtain ⟨hLf, hlast⟩ := hLF k _ hn hp
  have hslice := psSlice_pos _ hps a b
  have hmemS : ∀ s ∈ ((c.memo[(c.body.nodes[k]!).fn]!).ret.slice a b).toList, ∀ x ∈ segPos s,
      x ∈ psPos (c.memo[(c.body.nodes[k]!).fn]!).ret := by
    intro s hs x hx
    have : x ∈ segsPos ((c.memo[(c.body.nodes[k]!).fn]!).ret.slice a b) := List.mem_flatMap.2 ⟨s, hs, hx⟩
    rw [hslice] at this
    exact List.mem_of_mem_drop (List.mem_of_mem_take this)
  have ho : ((c.body.offsets[k]! : Int) + (m : Int) * ((c.P.defs[(c.body.nodes[k]!).fn]!).gates : Int)) =
      ((c.body.offsets[k]! + m * (c.P.defs[(c.body.nodes[k]!).fn]!).gates : Nat) : Int) := by push_cast; rfl
  have hin : ∀ w p, (Tgt.gate w, p) ∈ psPos (c.memo[(c.body.nodes[k]!).fn]!).ret →
      0 ≤ p ∧ p < ((c.P.defs[(c.body.nodes[k]!).fn]!).gates : Int) := by
    intro w p hx
    have h0 := (hpos _ hx).1
    refine ⟨h0, ?_⟩
    have := locW_lt (hrw w p hx) hLf hlast
    omega
  have hfa := forIn_app (GateW c.P c.d) _ _ ?_ #[] segs h'
  · obtain ⟨out, rfl, hout⟩ := hfa
    intro s hs
    rw [Array.toList_append, List.mem_append] at hs
    exact hs.elim (fun hs => by simp at hs) (hout s)
  intro s hs r y hy
  obtain ⟨hrep1, hgates⟩ := retOk_slice hgo hs
  have hx := hmemS s hs
  have hone1 : ∀ g, (∀ x ∈ pos g, x ∈ psPos (c.memo[(c.body.nodes[k]!).fn]!).ret) →
      GateW c.P c.d (Seg.one { g with base := ((c.body.offsets[k]! : Int) +
        (m : Int) * ((c.P.defs[(c.body.nodes[k]!).fn]!).gates : Int)) + g.base }) := fun g hg =>
    gateW_oneShift c.P hL hk hp hm ho (fun w p hxm => hrw w p (hg _ (by rwa [segPos_one] at hxm)))
      (fun w p hxm => hin w p (hg _ hxm))
  split at hy
  · rename_i hr
    split at hy
    · rename_i hall
      have hgs := gates_of_all hall
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      subst hy
      refine ⟨#[_], congrArg _ Array.push_eq_append, fun s' hs' => ?_⟩
      simp only [List.mem_singleton] at hs'
      subst hs'
      exact gateW_shift c.P hL hk hp hm ho (fun w p hxm => hrw w p (hx _ hxm)) (fun w p hxm => hin w p (hx _ hxm))
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
    have hxg : ∀ g ∈ s.ps.toList, ∀ x ∈ pos g, x ∈ psPos (c.memo[(c.body.nodes[k]!).fn]!).ret := fun g hg x hxm =>
      hx x (by
        rw [segPos_alike hsh]
        exact List.mem_flatMap.2 ⟨0, List.mem_range.2 (by omega), List.mem_flatMap.2 ⟨g, hg, hxm⟩⟩)
    obtain ⟨one'', h1'', h2''⟩ := forIn_app (GateW c.P c.d) _ s.ps.toList (fun g hg r' y' hy' => by
        split at hy'
        · simp only [pure, Except.pure, Except.ok.injEq] at hy'
          subst hy'
          refine ⟨#[_], congrArg _ Array.push_eq_append, fun s' hs' => ?_⟩
          simp only [List.mem_singleton] at hs'
          subst hs'
          exact hone1 g (hxg g hg)
        · exact absurd hy' (by intro h; cases h)
        · rename_i i hi
          obtain ⟨v', hv', hy'⟩ := Walk.except_bind_ok hy'
          simp only [pure, Except.pure, Except.ok.injEq] at hy'
          subst hy'
          refine ⟨v', rfl, ?_⟩
          refine hTW k m i g v' hm hi (fun x hxm => ?_) hv'
          obtain ⟨h0, -, hp⟩ := hpos x (hxg g hg x hxm)
          obtain ⟨h1, h2⟩ := hp i ((mem_pos_tgt hxm).trans hi)
          exact ⟨h0, h1, h2⟩) #[] one hone
    have hone_w : ∀ s' ∈ one.toList, GateW c.P c.d s' := by
      rw [h1'', Array.empty_append]; exact h2''
    split at hy
    · rename_i hall
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
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      subst hy
      refine ⟨#[_], congrArg _ Array.push_eq_append, fun s' hs' => ?_⟩
      simp only [List.mem_singleton] at hs'
      subst hs'
      have h1 : ∀ x ∈ one.toList, x.reps = 1 := by
        intro x hx'
        have hle := (Array.all_eq_true_iff_forall_mem.1 hall) x (Array.mem_toList_iff.1 hx')
        simp only [decide_eq_true_eq] at hle
        rcases hsrc x hx' with h0 | h0
        · simp at h0
        · omega
      have hpa := posAll_flatMap_ps one h1
      intro w p hx'
      rw [segPos_alike rfl] at hx'
      obtain ⟨_, -, hx'⟩ := List.mem_flatMap.1 hx'
      rw [hpa] at hx'
      obtain ⟨s', hs', hx''⟩ := List.mem_flatMap.1 hx'
      exact hone_w s' hs' w p hx''
    · have hy := walkAtMost_bind _ _ _ hy
      obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      subst hy
      simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at hv
      obtain ⟨segs', hsegs', hA2⟩ := forIn_app (GateW c.P c.d) _ _ (fun _ _ r' y' hy' => by
          simp only [pure, Except.pure, Except.ok.injEq] at hy'
          exact ⟨one, hy'.symm, hone_w⟩) r v hv
      exact ⟨segs', by rw [hsegs'], hA2⟩
  · rename_i hr
    have h1 : s.reps = 1 := by omega
    obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    subst hy
    rw [← Array.forIn_toList] at hv
    have hxg : ∀ g ∈ s.ps.toList, ∀ x ∈ pos g, x ∈ psPos (c.memo[(c.body.nodes[k]!).fn]!).ret := fun g hg x hxm =>
      hx x (by rw [segPos_one_rep h1]; exact List.mem_flatMap.2 ⟨g, hg, hxm⟩)
    obtain ⟨segs', rfl, hA'⟩ := forIn_app (GateW c.P c.d) _ s.ps.toList (fun g hg r' y' hy' => by
        split at hy'
        · simp only [pure, Except.pure, Except.ok.injEq] at hy'
          subst hy'
          refine ⟨#[_], congrArg _ Array.push_eq_append, fun s' hs' => ?_⟩
          simp only [List.mem_singleton] at hs'
          subst hs'
          exact hone1 g (hxg g hg)
        · exact absurd hy' (by intro h; cases h)
        · rename_i i hi
          obtain ⟨v', hv', hy'⟩ := Walk.except_bind_ok hy'
          simp only [pure, Except.pure, Except.ok.injEq] at hy'
          subst hy'
          refine ⟨v', rfl, ?_⟩
          refine hTW k m i g v' hm hi (fun x hxm => ?_) hv'
          obtain ⟨h0, -, hp⟩ := hpos x (hxg g hg x hxm)
          obtain ⟨h1, h2⟩ := hp i ((mem_pos_tgt hxm).trans hi)
          exact ⟨h0, h1, h2⟩) r v hv
    exact ⟨segs', rfl, hA'⟩

/-- `down`'s widths at one fuel: of a piece of positive count, its positions at or past 0, naming its nodes' output
leaves and its parameters' leaves in range and its gates with their widths, each segment `down` gives names the body's
gates with their widths. -/
def DownWAt (c : Desc.Ctx) (fuel : Nat) : Prop :=
  ∀ q segs, 0 < q.count → (∀ x ∈ pos q, 0 ≤ x.2) → (∀ x ∈ pos q, Fits c x) →
    (∀ x ∈ pos q, ∀ i, x.1 = .ref true i → i < c.body.params.size ∧ x.2 < ((c.body.params[i]!).2.leaves : Int)) →
    (∀ w p, (Tgt.gate w, p) ∈ pos q → LocW c.P c.d p.toNat w) →
    Desc.down c fuel q = .ok segs → ∀ s ∈ segs.toList, GateW c.P c.d s

/-- What `carry` gives of scan iteration `m`'s return leaves `[a, b)`, within its step's return when not empty, names
the body's gates with their widths. -/
def CarryW (c : Desc.Ctx) (fuel : Nat) : Prop :=
  ∀ k m a b segs, (c.body.nodes[k]!).form = .scan → m < (c.body.nodes[k]!).members →
    (a < b → b ≤ (c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.count) →
    Desc.carry c fuel k m a b = .ok segs → ∀ s ∈ segs.toList, GateW c.P c.d s

/-- Node `k`'s argument `i`'s positions `[lo, hi)` through `down`: segments naming the body's gates with their widths,
the argument naming references. -/
theorem one_w (c : Desc.Ctx) (fuel : Nat) (hD : DownWAt c fuel) {k i : Nat} (hA : ArgOk c k i)
    (hP : ∀ s ∈ (c.arg k i).segs.toList, ∀ q ∈ s.ps.toList, 0 < q.count)
    (hR : ∀ t < ((c.body.nodes[k]!).args[i]!).seq.count,
      Resolve.RefOk c.P c.d k (((c.body.nodes[k]!).args[i]!).seq.get t))
    {lo hi : Nat} (hhi : lo < hi → hi ≤ ((c.body.nodes[k]!).args[i]!).seq.count) {out : Array Seg}
    (h : oneOf c fuel k i lo hi = .ok out) : ∀ s ∈ out.toList, GateW c.P c.d s := by
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
  have hcnt : ∀ q ∈ ((c.arg k i).pieces lo hi).toList, 0 < q.count := by
    intro q hq
    unfold Desc.PS.pieces at hq
    rw [Array.toList_flatMap, List.mem_flatMap] at hq
    obtain ⟨g, hg, hq⟩ := hq
    rw [Array.toList_flatMap, List.mem_flatMap] at hq
    obtain ⟨m, -, hq⟩ := hq
    exact shiftGates_count _ _ (psSlice_ok _ hA.1 hP lo hi g hg).2.2 q hq
  exact loop_mem _ (Desc.down c fuel) _ (fun q hq segs hs => hD q segs (hcnt q hq)
    (fun x hx => by obtain ⟨t, -, rfl⟩ := hmem q hq x hx; simp [refPos])
    (fun x hx k' hk' => by
      obtain ⟨t, ht, rfl⟩ := hmem q hq x hx
      simp only [refPos, Tgt.ref.injEq] at hk'
      obtain ⟨hp', rfl⟩ := hk'
      have := ((hR t ht).2 hp').2
      simp only [refPos]
      exact_mod_cast this)
    (fun x hx i' hi' => by
      obtain ⟨t, ht, rfl⟩ := hmem q hq x hx
      simp only [refPos, Tgt.ref.injEq] at hi'
      obtain ⟨hp', rfl⟩ := hi'
      have := (hR t ht).1 hp'
      simp only [refPos]
      exact ⟨this.1, by exact_mod_cast this.2⟩)
    (fun w p hx => by
      obtain ⟨t, -, he⟩ := hmem q hq _ hx
      simp [refPos] at he) hs) out h

/-- **`throughArg`'s width step**: one step up, what member `m`'s parameter leaves read through node `k`'s argument `i`
give names the body's gates with their widths wherever `down`'s and `carry`'s segments do (`throughShape_succ`'s
argument). -/
theorem throughW_succ (c : Desc.Ctx) (hB : Resolve.BodyOk c.P c.d)
    (hA : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size → ArgOk c k i)
    (hAP : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size →
      ∀ s ∈ (c.arg k i).segs.toList, ∀ q ∈ s.ps.toList, 0 < q.count)
    (hL0 : ∀ k i d L : Nat, ((c.body.nodes[k]!).args[i]!).dims = some (d, L) → L = 0 →
      i < (c.P.defs[(c.body.nodes[k]!).fn]!).body.params.size →
      ((c.P.defs[(c.body.nodes[k]!).fn]!).body.params[i]!).2.leaves = 0)
    (fuel : Nat) (hD : DownWAt c fuel) (hK : CarryW c fuel) : ThroughW c (fuel + 1) := by
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
  have hUp := upOk_of hB hn
  have hAk : ArgOk c k i := hA k i hk (by rw [hnode]; exact hia)
  have hPk := hAP k i hk (by rw [hnode]; exact hia)
  have hRk : ∀ t < ((c.body.nodes[k]!).args[i]!).seq.count,
      Resolve.RefOk c.P c.d k (((c.body.nodes[k]!).args[i]!).seq.get t) := (hB.1 k hk).2.1 i
  have hone : ∀ lo hi v, (lo < hi → hi ≤ a.seq.count) → oneOf c fuel k i lo hi = .ok v →
      ∀ s ∈ v.toList, GateW c.P c.d s := fun lo hi v hb hv =>
    one_w c fuel hD hAk hPk hRk (by rw [hnode, hai']; exact hb) hv
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
  have hfa := forIn_app (GateW c.P c.d) _ _ ?_ #[] segs h'
  · obtain ⟨segs', rfl, hfin⟩ := hfa
    intro s hs
    rw [Array.toList_append, List.mem_append] at hs
    exact hs.elim (fun hs => by simp at hs) (hfin s)
  intro x hxm r y hy
  obtain ⟨lo, hi⟩ := x
  have hjx := hj (lo, hi) hxm
  rcases hf : node.form with _ | _ | _ | _
  case prim | call =>
    simp only [hf] at hy
    obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    refine ⟨v, hy.symm, hone lo hi v (fun hlt => ?_) hv⟩
    obtain ⟨hi0, hjl⟩ := hjx (hi - 1) (by omega) (by omega)
    have := hUp i hi0 (hi - 1) hjl m hm
    simp only [hf, hai'] at this
    omega
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
    have hfa2 := forIn_app (GateW c.P c.d) _ _ ?_ r w hw
    · obtain ⟨segs2, rfl, hag2⟩ := hfa2
      exact ⟨segs2, hy.symm, hag2⟩
    intro x hxm2 r2 y2 hy2
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
        refine ⟨v, hy.symm, hone lo hi v (fun hlt => ?_) hv⟩
        obtain ⟨hi0, hjl⟩ := hjx (hi - 1) (by omega) (by omega)
        have := hUp 0 hi0 (hi - 1) hjl 0 hm
        simp only [hf, hai', ↓reduceIte] at this
        omega
      · rw [if_neg (show ¬ ((m == 0) = true) by simpa using hm0)] at hy
        obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
        simp only [pure, Except.pure, Except.ok.injEq] at hy
        have hm1 : m - 1 < (c.body.nodes[k]!).members := by rw [hnode]; omega
        refine ⟨v, hy.symm, hK k (m - 1) lo hi v (by rw [hnode]; exact hf) hm1 (fun hlt => ?_) hv⟩
        obtain ⟨hi0, hjl⟩ := hjx (hi - 1) (by omega) (by omega)
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
        refine ⟨v, hy.symm, hone (m * L + lo) (m * L + hi) v (fun hlt => ?_) hv⟩
        obtain ⟨hi0', hjl⟩ := hjx (hi - 1) (by omega) (by omega)
        have := hUp i hi0' (hi - 1) hjl m hm
        simp only [hf, hi0, hx0, hai', hL, ↓reduceIte] at this
        omega
      · rw [if_neg hx0] at hy
        obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
        simp only [pure, Except.pure, Except.ok.injEq] at hy
        refine ⟨v, hy.symm, hone lo hi v (fun hlt => ?_) hv⟩
        obtain ⟨hi0', hjl⟩ := hjx (hi - 1) (by omega) (by omega)
        have := hUp i hi0' (hi - 1) hjl m hm
        simp only [hf, hi0, hx0, hai', ↓reduceIte] at this
        omega

/-- What `down` gives a batch of a composite's piece of positive count, one step up, names the body's gates with their
widths. -/
def BatchW (c : Desc.Ctx) (fuel : Nat) : Prop :=
  ∀ q segs k node, q.tgt = .ref false k → c.node k = .ok node → node.form = .batch →
    (c.P.defs[node.fn]!).prim = false → q.count ≠ 0 → q.flat = q → (∀ x ∈ pos q, 0 ≤ x.2) →
    (∀ x ∈ pos q, Fits c x) → Desc.down c (fuel + 1) q = .ok segs → ∀ s ∈ segs.toList, GateW c.P c.d s

/-- What `down` gives a scan's piece of one row, one step up, names the body's gates with their widths. -/
def ScanW (c : Desc.Ctx) (fuel : Nat) : Prop :=
  ∀ q segs k node (carry y : Flock.Program.Ty), q.tgt = .ref false k → c.node k = .ok node → node.form = .scan →
    node.n ≠ 0 → q.count ≠ 0 → q.flat = q → ¬ q.rows > 1 → (c.P.defs[node.fn]!).ret = .tuple #[carry, y] →
    (∀ x ∈ pos q, 0 ≤ x.2) → (∀ x ∈ pos q, Fits c x) → Desc.down c (fuel + 1) q = .ok segs →
    ∀ s ∈ segs.toList, GateW c.P c.d s

/-- A piece's rows, each through `down`, name the body's gates with their widths. -/
theorem byRows_w (c : Desc.Ctx) (fuel : Nat) (hD : DownWAt c fuel) (q : Piece) (h0 : 0 < q.count)
    (hq : ∀ x ∈ pos q, 0 ≤ x.2) (hf : ∀ x ∈ pos q, Fits c x)
    (hp : ∀ x ∈ pos q, ∀ i, x.1 = .ref true i → i < c.body.params.size ∧ x.2 < ((c.body.params[i]!).2.leaves : Int))
    (hg : ∀ w p, (Tgt.gate w, p) ∈ pos q → LocW c.P c.d p.toNat w)
    {segs : Array Seg} (h : byRows c fuel q = .ok segs) : ∀ s ∈ segs.toList, GateW c.P c.d s := by
  unfold byRows at h
  have h := walkAtMost_bind _ _ _ h
  dsimp only at h
  rw [← Array.forIn_toList] at h
  have hm : ∀ x ∈ q.rowsOf.toList, ∀ y ∈ pos x, y ∈ pos q := fun x hx y hy => by
    rw [← posAll_rowsOf q]
    exact List.mem_flatMap.2 ⟨x, hx, hy⟩
  have hlen : q.len ≠ 0 := fun hl => by
    unfold Desc.Piece.count at h0
    rw [hl, Nat.mul_zero] at h0
    exact absurd h0 (Nat.lt_irrefl 0)
  have hc : ∀ x ∈ q.rowsOf.toList, 0 < x.count := by
    intro x hx
    unfold Desc.Piece.rowsOf at hx
    simp only [Array.toList_map, List.mem_map] at hx
    obtain ⟨r, -, rfl⟩ := hx
    show 0 < 1 * q.len
    omega
  exact loop_mem _ (Desc.down c fuel) q.rowsOf.toList (fun x hx segs hs => hD x segs (hc x hx)
    (fun y hy => hq y (hm x hx y hy)) (fun y hy => hf y (hm x hx y hy)) (fun y hy => hp y (hm x hx y hy))
    (fun w p hy => hg w p (hm x hx _ hy)) hs) segs h

/-- A Call's piece of one row and stride 1, of a composite: what `fromCallee` gives, with its widths. -/
theorem downW_callOne (c : Desc.Ctx) (fuel : Nat) (hC : CalleeW c fuel) (q : Piece) {k : Nat}
    (ht : q.tgt = .ref false k) (hff : q.flat = q) {node : Flock.Program.Node} (hn : c.node k = .ok node)
    (hform : node.form = .call) (h0 : q.count ≠ 0) (hr : ¬ q.rows > 1) (hs : q.stride = 1)
    (hpr : (c.P.defs[node.fn]!).prim = false)
    {segs : Array Seg} (h : Desc.down c (fuel + 1) q = .ok segs) : ∀ s ∈ segs.toList, GateW c.P c.d s := by
  rw [down_call_one c fuel q ht hff hn hform h0 hr hs] at h
  exact hC k 0 q.base.toNat (q.base.toNat + q.len) segs
    (by rw [node_get hn, members_call hform]; omega) (by rw [node_get hn]; exact hpr) h

/-- A Call's piece of one row and another stride, of a composite: what `fromCallee` gives at each position, with its
widths. -/
theorem downW_callPts (c : Desc.Ctx) (fuel : Nat) (hC : CalleeW c fuel) (q : Piece) {k : Nat}
    (ht : q.tgt = .ref false k) (hff : q.flat = q) {node : Flock.Program.Node} (hn : c.node k = .ok node)
    (hform : node.form = .call) (h0 : q.count ≠ 0) (hr : ¬ q.rows > 1) (hs : q.stride ≠ 1)
    (hpr : (c.P.defs[node.fn]!).prim = false)
    {segs : Array Seg} (h : Desc.down c (fuel + 1) q = .ok segs) : ∀ s ∈ segs.toList, GateW c.P c.d s := by
  rw [down_call_pts c fuel q ht hff hn hform h0 hr hs] at h
  unfold callPts at h
  dsimp only at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at h
  rw [range'_size] at h
  exact loop_mem _ (fun t => Desc.fromCallee c fuel k 0 (q.at t).toNat ((q.at t).toNat + 1)) (List.range q.len)
    (fun _ _ segs hs => hC k 0 _ _ segs (by rw [node_get hn, members_call hform]; omega)
      (by rw [node_get hn]; exact hpr) hs) segs h

/-- **`down`'s width step**: one step up, `down`'s segments name the body's gates with their widths wherever `down`'s
and `fromCallee`'s do and a batch of a composite's and a scan's piece of one row give such segments, in a laid-out body
whose primitives are one gate of one leaf and whose nodes call what the decoder lets them (`downShape_succ`'s
argument). -/
theorem downW_succ (c : Desc.Ctx) (hB : Resolve.BodyOk c.P c.d) (hL : Laid c.P c.d)
    (hG1 : ∀ f : Nat, (c.P.defs[f]!).prim = true → (c.P.defs[f]!).gates = 1)
    (hP1 : ∀ f : Nat, (c.P.defs[f]!).prim = true → (c.P.defs[f]!).outLeaves = 1)
    (hCP : ∀ k node, c.node k = .ok node → (node.form = .prim → (c.P.defs[node.fn]!).prim = true) ∧
      (node.form = .call ∨ node.form = .scan → (c.P.defs[node.fn]!).prim = false))
    (fuel : Nat) (hD : DownWAt c fuel) (hC : CalleeW c fuel) (hBatch : BatchW c fuel) (hScan : ScanW c fuel) :
    DownWAt c (fuel + 1) := by
  intro q segs h0 hq hf hp hg h
  cases ht : q.tgt with
  | gate w => exact downW_gate c fuel q ht hg h
  | ref b k =>
  cases b with
  | true => exact downW_param c fuel q ht h
  | false =>
  rw [down_flat c fuel q ht] at h
  rw [← pos_flat q] at hq hf hp hg
  have ht' : q.flat.tgt = .ref false k := by rw [flat_tgt, ht]
  have h0' : q.flat.count ≠ 0 := by rw [flat_count]; omega
  have hff := flat_flat q
  generalize q.flat = p at ht' h0' hff hq hf hp hg h
  rcases hn : c.node k with e | node
  · exact absurd h (down_node c fuel p ht' h0' hn segs)
  have hdo := downOk_of hB hn
  cases hform : node.form with
  | prim =>
    have hpr := (hCP k node hn).1 hform
    cases hr : (c.P.defs[node.fn]!).ret with
    | value w => exact downW_prim c fuel p ht' hff hn hform hr hpr (hG1 _ hpr) hL h
    | _ => exact absurd h (down_prim_ret c fuel p ht' hff hn (Or.inl hform) h0' (fun w e => by rw [hr] at e; cases e) segs)
  | call =>
    have hpr := (hCP k node hn).2 (Or.inl hform)
    by_cases hr : p.rows > 1
    · rw [down_rows c fuel p ht' hff hn (Or.inl hform) h0' hr] at h
      exact byRows_w c fuel hD p (Nat.pos_of_ne_zero h0') hq hf hp hg h
    by_cases hs : p.stride = 1
    · exact downW_callOne c fuel hC p ht' hff hn hform h0' hr hs hpr h
    · exact downW_callPts c fuel hC p ht' hff hn hform h0' hr hs hpr h
  | scan =>
    by_cases hn0 : node.n = 0
    · exact absurd h (down_scan0 c fuel p ht' hff hn hform hn0 h0' segs)
    by_cases hr : p.rows > 1
    · rw [down_rows c fuel p ht' hff hn (Or.inr ⟨hform, hn0⟩) h0' hr] at h
      exact byRows_w c fuel hD p (Nat.pos_of_ne_zero h0') hq hf hp hg h
    by_cases hex : ∃ carry y, (c.P.defs[node.fn]!).ret = .tuple #[carry, y]
    · obtain ⟨carry, y, hret⟩ := hex
      exact hScan p segs k node carry y ht' hn hform hn0 h0' hff hr hret hq hf h
    · exact absurd h
        (down_scan_ret c fuel p ht' hff hn hform hn0 h0' hr (fun carry y e => hex ⟨carry, y, e⟩) segs)
  | batch =>
    by_cases hpr : (c.P.defs[node.fn]!).prim = true
    · cases hr : (c.P.defs[node.fn]!).ret with
      | value w =>
        have hout : node.out.leaves ≤ node.members := by
          rw [members_batch hform]
          by_contra hlt
          have := hdo node.n (by omega)
          simp only [hform, hP1 _ hpr, Nat.div_one] at this
          exact absurd this.1 (Nat.lt_irrefl _)
        exact downW_batchPrim c fuel p ht' hff hn hform hpr hr (hG1 _ hpr) hL hq hf hout h
      | _ =>
        exact absurd h
          (down_prim_ret c fuel p ht' hff hn (Or.inr ⟨hform, hpr⟩) h0' (fun w e => by rw [hr] at e; cases e) segs)
    · exact hBatch p segs k node ht' hn hform (by simpa using hpr) h0' hff hq hf h

/-- A scan's initial carry's leaves `[lo, hi)`, each in range, through `throughArg`: segments naming the body's gates
with their widths. -/
theorem init_w (c : Desc.Ctx) {fuel k : Nat} {node : Flock.Program.Node} (hn : c.node k = .ok node)
    (hTW : ThroughW c fuel) (hm0 : 0 < node.members) {lo hi : Nat}
    (hlv : ∀ j, lo ≤ j → j < hi → 0 < (c.P.defs[node.fn]!).body.params.size ∧
      j < ((c.P.defs[node.fn]!).body.params[0]!).2.leaves) {segs : Array Seg}
    (h : Desc.throughArg c fuel k 0 0 { tgt := .ref true 0, base := lo, stride := 1, len := hi - lo } = .ok segs) :
    ∀ s ∈ segs.toList, GateW c.P c.d s := by
  have hnode := node_get hn
  exact hTW k 0 0 _ segs (by rw [hnode]; exact hm0) rfl (fun x hx => by
    rw [pos_carry, List.mem_map] at hx
    obtain ⟨j, hj, rfl⟩ := hx
    rw [List.mem_range'_1] at hj
    rw [hnode]
    have := hlv j hj.1 (by omega)
    exact ⟨by simp, this.1, by show (j : Int) < _; exact_mod_cast this.2⟩) h

/-- **`carry`'s width step**: one step up, `carry`'s segments name the body's gates with their widths wherever the
step's summary holds `RetOk`, the step is a composite, and `fromCallee`'s and `throughArg`'s segments one step down name
them (`carryShape_succ`'s argument). -/
theorem carryW_succ (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (fuel : Nat)
    (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn)
    (hCP : ∀ k node, c.node k = .ok node → node.form = .scan → (c.P.defs[node.fn]!).prim = false)
    (hCW : CalleeW c fuel) (hTW : ThroughW c fuel) : CarryW c (fuel + 1) := by
  intro k m a b segs hform hm hb h
  rw [Desc.carry.eq_2] at h
  obtain ⟨node, hn, h⟩ := Walk.except_bind_ok h
  dsimp only at h
  have h' := ok_bind_pure h
  rw [← Array.forIn_toList] at h'
  have hnode : c.body.nodes[k]! = node := node_get hn
  rw [hnode] at hform hm hb
  have hsp := hCP k node hn hform
  have hRk := hR k node hn
  obtain ⟨hps, hpos, -, hag, hpp⟩ := hRk
  have hlenR : (psPos (c.memo[node.fn]!).ret).length = (c.P.defs[node.fn]!).body.ret.count := by
    have := (hag (c.P.child sc k 0) (by rw [child_eq c sc hd hn 0]; rfl)).length_eq
    simpa using this
  obtain ⟨-, hid⟩ := carryRuns_cover _ hps hpp a b (fun hab => by rw [hlenR]; exact hb hab)
  have hm0 : 0 < node.members := by omega
  have hcarry : ∀ h j' : Nat, (psPos (c.memo[node.fn]!).ret)[h]? = some (.ref true 0, (j' : Int)) →
      0 < (c.P.defs[node.fn]!).body.params.size ∧ j' < ((c.P.defs[node.fn]!).body.params[0]!).2.leaves := by
    intro h j' hx
    have := (hpos _ (List.mem_of_getElem? hx)).2.2 0 rfl
    exact ⟨this.1, by have h2 : ((j' : Nat) : Int) < _ := this.2; exact_mod_cast h2⟩
  have hfa := forIn_app (GateW c.P c.d) _ _ ?_ #[] segs h'
  · obtain ⟨out, rfl, hout⟩ := hfa
    intro s hs
    rw [Array.toList_append, List.mem_append] at hs
    exact hs.elim (fun hs => by simp at hs) (hout s)
  intro x hx r' y hy
  split at hy
  · rename_i hxid
    obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    subst hy
    exact ⟨v, rfl, init_w c hn hTW hm0 (fun j h1 h2 => hcarry j j (hid x hx hxid j h1 h2)) hv⟩
  · have hy := walkAtMost_bind _ _ _ hy
    obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    subst hy
    simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at hv
    obtain ⟨segs', hsegs', hA2⟩ := forIn_app (GateW c.P c.d) _ _ (fun j _ r'' y' hy' => by
        obtain ⟨ch, hch, hy'⟩ := Walk.except_bind_ok hy'
        have hok := chainOf_ok _ j ch hch
        split at hy'
        · obtain ⟨v', hv', hy'⟩ := Walk.except_bind_ok hy'
          simp only [pure, Except.pure, Except.ok.injEq] at hy'
          subst hy'
          exact ⟨v', rfl, hCW k (m - (ch.hist.size - 1)) _ _ v' (by rw [hnode]; omega) (by rw [hnode]; exact hsp) hv'⟩
        · rename_i hcond
          obtain ⟨v', hv', hy'⟩ := Walk.except_bind_ok hy'
          simp only [pure, Except.pure, Except.ok.injEq] at hy'
          subst hy'
          refine ⟨v', rfl, ?_⟩
          have hs : ∀ s ≤ m, (psPos (c.memo[node.fn]!).ret)[ch.at s]? = some (.ref true 0, (ch.at (s + 1) : Int)) := by
            intro s hs
            rcases hex : ch.exit with _ | q0
            · obtain ⟨q, hq, ht, hb⟩ := chainAt_cyc hok hex s
              exact carryPos_of_retAt (hR k node hn) hq ht hb
            · have he : ¬ ch.hist.size - 1 ≤ m := fun he => hcond (by simp [hex, he])
              obtain ⟨-, -, hsc, -, -⟩ := hok
              obtain ⟨q, hq, ht, hb⟩ := hsc s (by omega)
              rw [chainAt_lt ch (show s < ch.hist.size by omega), chainAt_lt ch (show s + 1 < ch.hist.size by omega)]
              exact carryPos_of_retAt (hR k node hn) hq ht hb
          exact init_w c hn hTW hm0 (lo := ch.at (m + 1)) (hi := ch.at (m + 1) + 1)
            (fun j' h1 h2 => by rw [show j' = ch.at (m + 1) by omega]; exact hcarry _ _ (hs m le_rfl)) hv') r' v hv
    exact ⟨segs', by rw [hsegs'], hA2⟩

/-- **`down`, `fromCallee`, `throughArg` and `carry` name the body's gates with their widths, at every fuel**, in a
laid-out body whose callees are laid out and name theirs (`RetW`), whose primitives are one gate of one leaf, whose nodes
call what the decoder lets them, wherever a batch of a composite's piece and a scan's piece of one row give such
segments one step up (`shapes`'s argument). -/
theorem widths (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (hB : Resolve.BodyOk c.P c.d)
    (hL : Laid c.P c.d)
    (hG1 : ∀ f : Nat, (c.P.defs[f]!).prim = true → (c.P.defs[f]!).gates = 1)
    (hP1 : ∀ f : Nat, (c.P.defs[f]!).prim = true → (c.P.defs[f]!).outLeaves = 1)
    (hCP : ∀ k node, c.node k = .ok node → (node.form = .prim → (c.P.defs[node.fn]!).prim = true) ∧
      (node.form = .call ∨ node.form = .scan → (c.P.defs[node.fn]!).prim = false))
    (hA : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size → ArgOk c k i)
    (hAP : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size →
      ∀ s ∈ (c.arg k i).segs.toList, ∀ q ∈ s.ps.toList, 0 < q.count)
    (hL0 : ∀ k i d L : Nat, ((c.body.nodes[k]!).args[i]!).dims = some (d, L) → L = 0 →
      i < (c.P.defs[(c.body.nodes[k]!).fn]!).body.params.size →
      ((c.P.defs[(c.body.nodes[k]!).fn]!).body.params[i]!).2.leaves = 0)
    (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn)
    (hRW : ∀ k node, c.node k = .ok node → RetW c.P c.memo node.fn)
    (hLF : ∀ k node, c.node k = .ok node → (c.P.defs[node.fn]!).prim = false → Laid c.P node.fn ∧
      (c.P.defs[node.fn]!).body.offsets[(c.P.defs[node.fn]!).body.nodes.size]! = (c.P.defs[node.fn]!).gates)
    (hBatch : ∀ fuel, DownWAt c fuel → CalleeW c fuel → BatchW c fuel)
    (hScan : ∀ fuel, CalleeW c fuel → CarryW c fuel → ScanW c fuel) :
    ∀ fuel, DownWAt c fuel ∧ CalleeW c fuel ∧ ThroughW c fuel ∧ CarryW c fuel
  | 0 => by
    refine ⟨fun q segs _ _ _ _ _ h => absurd h (down_zero c q segs), ?_, ?_, ?_⟩
    · intro _ _ _ _ _ _ _ h; rw [Desc.fromCallee.eq_1] at h; cases h
    · intro _ _ _ _ _ _ _ _ h; rw [Desc.throughArg.eq_1] at h; cases h
    · intro _ _ _ _ _ _ _ _ h; rw [Desc.carry.eq_1] at h; cases h
  | fuel + 1 =>
    have ⟨hD, hC, hTW, hK⟩ := widths c sc hd hB hL hG1 hP1 hCP hA hAP hL0 hR hRW hLF hBatch hScan fuel
    ⟨downW_succ c hB hL hG1 hP1 hCP fuel hD hC (hBatch fuel hD hC) (hScan fuel hC hK),
      calleeW_succ c fuel hR hRW hL hLF hTW (throughReps c), throughW_succ c hB hA hAP hL0 fuel hD hK,
      carryW_succ c sc hd fuel hR (fun k node hn hf => (hCP k node hn).2 (Or.inr hf)) hC hTW⟩

/-- A scan's ys iterations `[lo / yl, (hi - 1) / yl]`, each through `fromCallee`, of a composite step: segments naming
the body's gates with their widths. -/
theorem ys_w (c : Desc.Ctx) (fuel : Nat) (hCW : CalleeW c fuel)
    {k cl yl lo hi : Nat} (hm : (hi - 1) / yl < (c.body.nodes[k]!).members)
    (hsp : (c.P.defs[(c.body.nodes[k]!).fn]!).prim = false)
    {out0 segs : Array Seg}
    (h : forIn (List.range' (lo / yl) ((hi - 1) / yl + 1 - lo / yl)) out0 (fun m (s : Array Seg) => do
        let v ← Desc.fromCallee c fuel k m (cl + max lo (m * yl) - m * yl) (cl + min hi ((m + 1) * yl) - m * yl)
        pure (ForInStep.yield (s ++ v))) = .ok segs) :
    ∃ segs2, segs = out0 ++ segs2 ∧ ∀ s ∈ segs2.toList, GateW c.P c.d s :=
  forIn_app (GateW c.P c.d) _ _ (fun m hmm r y hy => by
      rw [List.mem_range'_1] at hmm
      obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      exact ⟨v, hy.symm, hCW k m _ _ v (by omega) hsp hv⟩) out0 segs h

/-- A scan's piece of stride 1: its ys past the carry's `cl` leaves, as `scanOne` reads them, of a composite step, give
segments naming the body's gates with their widths. -/
theorem scanYs_w (c : Desc.Ctx) (fuel : Nat) (hCW : CalleeW c fuel)
    {k cl yl a b : Nat} (hm : cl < b → (b - 1 - cl) / yl < (c.body.nodes[k]!).members)
    (hsp : (c.P.defs[(c.body.nodes[k]!).fn]!).prim = false)
    {out0 segs : Array Seg}
    (h : (if (decide (cl < b) && decide (yl > 0)) = true then do
        Desc.walkAtMost "iterations of a scan whose ys are read one by one" ((b - cl - 1) / yl + 1 - (max a cl - cl) / yl)
        (forIn [(max a cl - cl) / yl:(b - cl - 1) / yl + 1] out0 fun m s => do
          let v ← Desc.fromCallee c fuel k m (cl + max (max a cl - cl) (m * yl) - m * yl)
            (cl + min (b - cl) ((m + 1) * yl) - m * yl)
          pure (ForInStep.yield (s ++ v))) >>= fun s => pure s
      else pure out0) = .ok segs) :
    ∃ s2, segs = out0 ++ s2 ∧ ∀ s ∈ s2.toList, GateW c.P c.d s := by
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
    have h := walkAtMost_bind _ _ _ h
    obtain ⟨v, hv, h⟩ := Walk.except_bind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, Nat.add_sub_cancel,
      Nat.div_one] at hv
    exact ys_w c fuel hCW (k := k) (cl := cl) (yl := yl) (lo := max a cl - cl) (hi := b - cl)
      (by rw [show b - cl - 1 = b - 1 - cl by omega]; exact hm hc.1) hsp hv
  · simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨#[], by simp, fun s hs => by simp at hs⟩

/-- **A scan's piece of one row**, one step up: what `carry` and `fromCallee` give of its final carry's leaves and its
ys names the body's gates with their widths wherever their segments one step down do (`scanShape_of`'s argument). -/
theorem scanW_of (c : Desc.Ctx) (hB : Resolve.BodyOk c.P c.d) (fuel : Nat)
    (hCW : CalleeW c fuel) (hK : CarryW c fuel) : ScanW c fuel := by
  intro q segs k node carry y ht hn hform hn0 h0 hff hr hret hq hf h
  have hdo := downOk_of hB hn
  have h1 := rows_one h0 hr
  have hnode : c.body.nodes[k]! = node := node_get hn
  let M : Nat → Nat × Nat := fun j => if j < carry.leaves then (node.n - 1, j)
    else ((j - carry.leaves) / y.leaves, carry.leaves + (j - carry.leaves) % y.leaves)
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
    · simp only [M, hlt, ↓reduceIte]; simp only [hlt, ↓reduceIte] at hmn hj'; exact ⟨hmn, hj'⟩
    · simp only [M, hlt, ↓reduceIte]; simp only [hlt, ↓reduceIte] at hmn hj'; exact ⟨hmn, hj'⟩
  have hpt : ∀ t < q.len, 0 ≤ q.at t ∧ (q.at t).toNat < node.out.leaves := by
    intro t ht'
    have hm := at_mem_pos h1 ht'
    have hb := hq _ hm
    have hfl := hf _ hm k ht
    simp only [hnode] at hb hfl
    exact ⟨hb, by omega⟩
  have hlen : 0 < q.len := by rw [← count_one h1]; omega
  have hsp : (c.P.defs[(c.body.nodes[k]!).fn]!).prim = false := by
    have := hdo _ (hpt 0 hlen).2
    simp only [hform] at this
    rw [hnode]
    exact this.1
  rw [down_scan_head c fuel q ht hff hn hform hn0 h0 hr hret] at h
  split at h
  · rename_i hs
    unfold scanOne at h
    dsimp only at h
    have hat : ∀ t, (q.at t).toNat = q.base.toNat + t := fun t => by
      have := (hpt 0 hlen).1
      simp only [Desc.Piece.at, hs] at this ⊢
      omega
    have hbl : q.base.toNat + q.len ≤ node.out.leaves := by
      have := (hpt (q.len - 1) (by omega)).2
      rw [hat] at this
      omega
    have hmB : carry.leaves < q.base.toNat + q.len →
        (q.base.toNat + q.len - 1 - carry.leaves) / y.leaves < (c.body.nodes[k]!).members := by
      intro hcb
      have := (hbd (q.base.toNat + q.len - 1) (by omega)).1
      simp only [M, show ¬ (q.base.toNat + q.len - 1 < carry.leaves) by omega, ↓reduceIte] at this
      rw [hnode, members_scan hform]
      exact this
    split at h
    · obtain ⟨v1, hv1, h⟩ := Walk.except_bind_ok h
      obtain ⟨s2, rfl, hA2⟩ := scanYs_w c fuel hCW hmB hsp h
      have hA1 := hK k (node.n - 1) _ _ v1 (by rw [hnode]; exact hform)
        (by rw [hnode, members_scan hform]; omega) (fun hlt => by
          have := (hbd (min (q.base.toNat + q.len) carry.leaves - 1) (by omega)).2
          simp only [M, show min (q.base.toNat + q.len) carry.leaves - 1 < carry.leaves by omega,
            ↓reduceIte] at this
          rw [hnode]
          omega) hv1
      intro s hs
      rw [Array.empty_append, Array.toList_append, List.mem_append] at hs
      exact hs.elim (hA1 s) (hA2 s)
    · obtain ⟨s2, rfl, hA2⟩ := scanYs_w c fuel hCW hmB hsp h
      rw [Array.empty_append]
      exact hA2
  · unfold scanPts at h
    dsimp only at h
    have h := walkAtMost_bind _ _ _ h
    obtain ⟨v, hv, h⟩ := Walk.except_bind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at hv
    rw [range'_size] at hv
    obtain ⟨segs', hsegs, hA⟩ := forIn_app (GateW c.P c.d) _ (List.range q.len)
      (fun t htm r y' hy => by
        obtain ⟨-, hlt⟩ := hpt t (List.mem_range.1 htm)
        obtain ⟨hmn, hjr⟩ := hbd _ hlt
        split at hy
        · rename_i hc
          obtain ⟨v', hv', hy⟩ := Walk.except_bind_ok hy
          simp only [pure, Except.pure, Except.ok.injEq] at hy
          refine ⟨v', hy.symm, ?_⟩
          simp only [M, hc, ↓reduceIte] at hmn hjr
          have := hK k (node.n - 1) (q.at t).toNat ((q.at t).toNat + 1) v' (by rw [hnode]; exact hform)
            (by rw [hnode, members_scan hform]; exact hmn) (fun _ => by rw [hnode]; omega) hv'
          simpa using this
        · rename_i hc
          obtain ⟨v', hv', hy⟩ := Walk.except_bind_ok hy
          simp only [pure, Except.pure, Except.ok.injEq] at hy
          refine ⟨v', hy.symm, ?_⟩
          simp only [M, hc, ↓reduceIte] at hmn
          have := hCW k _ _ _ v' (by rw [hnode, members_scan hform]; exact hmn) hsp hv'
          simpa using this) #[] v hv
    rw [hsegs, Array.empty_append]
    exact hA

/-- Leaf `e` of a batch whose members each return `L` leaves at gates `B + j * s`, `G` apart, at gate
`B + e * F` (`affineOf`'s step `F`): member `e / L`'s leaf `e % L`. -/
theorem affine_pos {e L : Nat} {B s G : Int} (hLG : L = 1 ∨ G = (L : Int) * s) :
    B + (e : Int) * (if (L == 1) = true then G else s) =
      ((e / L : Nat) : Int) * G + (B + ((e % L : Nat) : Int) * s) := by
  split
  · rename_i h1
    have h1' : L = 1 := by simpa using h1
    subst h1'
    rw [Nat.div_one, Nat.mod_one]
    push_cast
    ring
  · rename_i h1
    have h1' : L ≠ 1 := by simpa using h1
    rw [hLG.resolve_left h1']
    have hdm : (e : Int) = ((e / L : Nat) : Int) * (L : Int) + ((e % L : Nat) : Int) := by
      exact_mod_cast (Nat.div_add_mod' e L).symm
    rw [hdm]
    ring

/-- **A batch of a composite's piece in its one-run case** names the body's gates with their widths: leaf `e` of the
node is member `e / L`'s return leaf `e % L`, a gate of the run its callee names with its width (`RetW`), in laid-out
bodies. `batchW_of`'s `hAff`. -/
theorem batchAffine_w (c : Desc.Ctx) (hB : Resolve.BodyOk c.P c.d) (hL : Laid c.P c.d)
    (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn)
    (hRW : ∀ k node, c.node k = .ok node → RetW c.P c.memo node.fn)
    (hLF : ∀ k node, c.node k = .ok node → (c.P.defs[node.fn]!).prim = false → Laid c.P node.fn ∧
      (c.P.defs[node.fn]!).body.offsets[(c.P.defs[node.fn]!).body.nodes.size]! = (c.P.defs[node.fn]!).gates) :
    ∀ q k node p, q.tgt = .ref false k → c.node k = .ok node → node.form = .batch →
      (c.P.defs[node.fn]!).prim = false → (c.P.defs[node.fn]!).outLeaves ≠ 0 → q.count ≠ 0 →
      (∀ x ∈ pos q, 0 ≤ x.2) → (∀ x ∈ pos q, Fits c x) → affineOf c k node q = some p →
      GateW c.P c.d (Seg.one p) := by
  intro q k node p ht hn hform hp hL0 h0 hq hf ha
  obtain ⟨hok, hpos, -, -, hpp⟩ := hR k node hn
  have hRWk := hRW k node hn
  obtain ⟨hLc, hlast⟩ := hLF k node hn hp
  have hdo := downOk_of hB hn
  have hk := node_lt hn
  have hnode := node_get hn
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
  have hg0c : 0 < g0.count := hpp s0 (by rw [hs0]; simp) g0 (by rw [hg0]; simp)
  have hr1 : g0.rows = 1 := by
    unfold Desc.Piece.count at hg0c
    rcases Nat.lt_or_ge g0.rows 1 with h | h
    · rw [show g0.rows = 0 by omega, Nat.zero_mul] at hg0c; omega
    · omega
  intro w' x hx
  rw [segPos_one, pos_eq_map, List.mem_map] at hx
  obtain ⟨t, ht', he⟩ := hx
  simp only [Prod.mk.injEq] at he
  obtain ⟨hw, rfl⟩ := he
  have hw' : w = w' := by simpa [Desc.Piece.affine] using hw
  rw [← hw', ppt_affine]
  have ht'' : t < q.count := List.mem_range.1 ht'
  have hmem : (q.tgt, ppt q t) ∈ pos q := by
    rw [pos_eq_map]; exact List.mem_map.2 ⟨t, List.mem_range.2 ht'', rfl⟩
  have he0 : 0 ≤ ppt q t := hq _ hmem
  have hfe : ppt q t < ((c.body.nodes[k]!).out.leaves : Int) := hf _ hmem k ht
  rw [hnode] at hfe
  obtain ⟨e, he⟩ := Int.eq_ofNat_of_zero_le he0
  rw [he] at hfe ⊢
  have hel : e < node.out.leaves := by exact_mod_cast hfe
  have hdk := hdo e hel
  simp only [hform] at hdk
  have hmn := hdk.1
  have hLpos : 0 < (c.P.defs[node.fn]!).outLeaves := Nat.pos_of_ne_zero hL0
  have hjL : e % (c.P.defs[node.fn]!).outLeaves < g0.len := by rw [hlen]; exact Nat.mod_lt _ hLpos
  have hgm := at_mem_pos hr1 hjL
  rw [hgt] at hgm
  have hRj := hRWk w _ (by rw [hflat]; exact hgm)
  have hj0 : 0 ≤ g0.at (e % (c.P.defs[node.fn]!).outLeaves) := (hpos _ (by rw [hflat]; exact hgm)).1
  obtain ⟨n', hn'⟩ := Int.eq_ofNat_of_zero_le hj0
  rw [hn', Int.toNat_natCast] at hRj
  have hnl := locW_lt hRj hLc hlast
  subst hnode
  have hm' : e / (c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves < (c.body.nodes[k]!).members := by
    rw [members_batch hform]; exact hmn
  have key := locW_child c.P hL hk hp hm' hnl hRj
  have hpe := affine_pos (e := e) (B := (c.body.offsets[k]! : Int) + g0.base) (s := g0.stride)
    (G := ((c.P.defs[(c.body.nodes[k]!).fn]!).gates : Int)) hFG
  have hat : g0.at (e % (c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves) =
      g0.base + ((e % (c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves : Nat) : Int) * g0.stride := rfl
  have hfin : (c.body.offsets[k]! : Int) + g0.base + (e : Int) *
      (if ((c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves == 1) = true then
        ((c.P.defs[(c.body.nodes[k]!).fn]!).gates : Int) else g0.stride) =
      ((c.body.offsets[k]! + e / (c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves *
        (c.P.defs[(c.body.nodes[k]!).fn]!).gates + n' : Nat) : Int) := by
    rw [hpe, Int.add_assoc, ← hat, hn']
    push_cast
    ring
  rw [hfin, Int.toNat_natCast]
  exact key

/-- A batch's piece of one row and any other stride, position by position through `down`: segments naming the body's
gates with their widths. -/
theorem batchPts_w (c : Desc.Ctx) (fuel : Nat) (hD : DownWAt c fuel) (q : Piece) {k : Nat}
    (ht : q.tgt = .ref false k) (h1 : q.rows = 1) (hq : ∀ x ∈ pos q, 0 ≤ x.2) (hf : ∀ x ∈ pos q, Fits c x)
    {segs : Array Seg} (h : batchPts c fuel q = .ok segs) : ∀ s ∈ segs.toList, GateW c.P c.d s := by
  unfold batchPts at h
  have h := walkAtMost_bind _ _ _ h
  dsimp only at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at h
  rw [range'_size] at h
  have hpt : ∀ t, pos { q with base := q.at t, stride := 1, len := 1 } = [(q.tgt, q.at t)] := by
    intro t
    simp [pos, h1]
  exact loop_mem _ (fun t => Desc.down c fuel { q with base := q.at t, stride := 1, len := 1 }) (List.range q.len)
    (fun t htm segs hs => by
      have hm := at_mem_pos h1 (List.mem_range.1 htm)
      exact hD _ segs (by simp [Desc.Piece.count, h1])
        (by rw [hpt t]; intro x hx; simp only [List.mem_singleton] at hx; subst hx; exact hq _ hm)
        (by rw [hpt t]; intro x hx; simp only [List.mem_singleton] at hx; subst hx; exact hf _ hm)
        (by rw [hpt t]; intro x hx i hi; simp only [List.mem_singleton] at hx; subst hx; rw [ht] at hi; cases hi)
        (by rw [hpt t]; intro w p hx; simp only [List.mem_singleton, Prod.mk.injEq] at hx; rw [ht] at hx
            exact absurd hx.1 (by intro e; cases e))
        hs) segs h

/-- What `batchOne` gives a batch of a composite's piece of one row and stride 1, past its one-run case, names the
body's gates with their widths. -/
def BatchOneW (c : Desc.Ctx) (fuel : Nat) : Prop :=
  ∀ q k node segs, q.tgt = .ref false k → c.node k = .ok node → node.form = .batch →
    (c.P.defs[node.fn]!).prim = false → (c.P.defs[node.fn]!).outLeaves ≠ 0 → q.count ≠ 0 → q.rows = 1 →
    q.stride = 1 → (∀ x ∈ pos q, 0 ≤ x.2) → (∀ x ∈ pos q, Fits c x) → batchOne c fuel k node q = .ok segs →
    ∀ s ∈ segs.toList, GateW c.P c.d s

/-- **`BatchW` from its cases**: a batch of a composite's piece row by row and position by position through `down`
(`byRows_w`, `batchPts_w`), and in its one-run, stride-1 and whole-member-stride cases wherever those name the body's
gates with their widths (`batchShape_of`'s argument). -/
theorem batchW_of (c : Desc.Ctx) (fuel : Nat) (hD : DownWAt c fuel)
    (hAff : ∀ q k node p, q.tgt = .ref false k → c.node k = .ok node → node.form = .batch →
      (c.P.defs[node.fn]!).prim = false → (c.P.defs[node.fn]!).outLeaves ≠ 0 → q.count ≠ 0 →
      (∀ x ∈ pos q, 0 ≤ x.2) → (∀ x ∈ pos q, Fits c x) → affineOf c k node q = some p → GateW c.P c.d (Seg.one p))
    (hOne : BatchOneW c fuel)
    (hStride : ∀ q k node segs, q.tgt = .ref false k → c.node k = .ok node → node.form = .batch →
      (c.P.defs[node.fn]!).prim = false → (c.P.defs[node.fn]!).outLeaves ≠ 0 → q.rows = 1 →
      (q.len > 1 ∧ q.stride % ((c.P.defs[node.fn]!).outLeaves : Int) = 0 ∧ q.base ≥ 0) →
      (∀ x ∈ pos q, 0 ≤ x.2) → (∀ x ∈ pos q, Fits c x) → batchStride c fuel k node q = .ok segs →
      ∀ s ∈ segs.toList, GateW c.P c.d s) :
    BatchW c fuel := by
  intro q segs k node ht hn hform hp h0 hff hq hf h
  by_cases hL : (c.P.defs[node.fn]!).outLeaves = 0
  · exact absurd h (down_batch0 c fuel q ht hff hn hform hp hL h0 segs)
  rcases ha : affineOf c k node q with _ | p
  · by_cases hr : q.rows > 1
    · rw [down_batch_rows c fuel q ht hff hn hform hp hL h0 ha hr] at h
      exact byRows_w c fuel hD q (Nat.pos_of_ne_zero h0) hq hf
        (fun x hx i hi => by rw [mem_pos_tgt hx, ht] at hi; cases hi)
        (fun w p hx => by have e := mem_pos_tgt hx; rw [ht] at e; cases e) h
    by_cases hs : q.stride = 1
    · rw [down_batch_one c fuel q ht hff hn hform hp hL h0 ha hr hs] at h
      exact hOne q k node segs ht hn hform hp hL h0 (rows_one h0 hr) hs hq hf h
    by_cases hcp : q.len > 1 ∧ q.stride % ((c.P.defs[node.fn]!).outLeaves : Int) = 0 ∧ q.base ≥ 0
    · rw [down_batch_stride c fuel q ht hff hn hform hp hL h0 ha hr hs hcp] at h
      exact hStride q k node segs ht hn hform hp hL (rows_one h0 hr) hcp hq hf h
    rw [down_batch_pts c fuel q ht hff hn hform hp hL h0 ha hr hs hcp] at h
    exact batchPts_w c fuel hD q ht (rows_one h0 hr) hq hf h
  · rw [down_batch_affine c fuel q ht hff hn hform hp hL h0 ha] at h
    simp only [Except.ok.injEq] at h
    subst h
    intro s hs
    simp only [List.mem_singleton] at hs
    subst hs
    exact hAff q k node p ht hn hform hp hL h0 hq hf ha

/-- **Copies of a callee's gates, a copy a member** (`shiftGates` to member `m0`'s first gate, each next copy `dm`
members on): of the callee's gates at or past 0 with their widths, each copy a member of node `k`, they name the
caller's gates with theirs, in laid-out bodies. -/
theorem gateW_copies (P : Prog) {d k : Nat}
    (hoff : Laid P d) (hk : k < (P.defs[d]!).body.nodes.size)
    (hp : (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).prim = false)
    (hLc : Laid P ((P.defs[d]!).body.nodes[k]!).fn)
    (hlast : (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).body.offsets[
        (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).body.nodes.size]! =
      (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates)
    {ps : Array Piece} {m0 : Nat} {δ dm S : Int} {len reps : Nat}
    (hδ : δ = ((P.defs[d]!).body.offsets[k]! : Int) +
      (m0 : Int) * ((P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates : Int))
    (hS : S = dm * ((P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates : Int))
    (hW : ∀ w v, (Tgt.gate w, v) ∈ posAll ps → 0 ≤ v ∧ LocW P ((P.defs[d]!).body.nodes[k]!).fn v.toNat w)
    (hin : ∀ u < reps, 0 ≤ (m0 : Int) + (u : Int) * dm ∧
      (m0 : Int) + (u : Int) * dm < (((P.defs[d]!).body.nodes[k]!).members : Int)) :
    GateW P d { ps := Desc.shiftGates δ ps, len := len, reps := reps, shift := S } := by
  intro w p hx
  unfold segPos at hx
  obtain ⟨u, hu, hx⟩ := List.mem_flatMap.1 hx
  rw [List.mem_range] at hu
  simp only [copy_pos, shiftGates_pos, List.map_map, shiftPos_shiftPos, List.mem_map] at hx
  obtain ⟨⟨t, v⟩, hy, he⟩ := hx
  cases t with
  | ref b i => simp [shiftPos] at he
  | gate w0 =>
    simp only [shiftPos, Prod.mk.injEq, Tgt.gate.injEq] at he
    obtain ⟨rfl, rfl⟩ := he
    obtain ⟨hv0, hl⟩ := hW w0 v hy
    obtain ⟨n, rfl⟩ := Int.eq_ofNat_of_zero_le hv0
    rw [Int.toNat_natCast] at hl
    have hnl := locW_lt hl hLc hlast
    obtain ⟨h0, h1⟩ := hin u hu
    obtain ⟨mm, hmm⟩ := Int.eq_ofNat_of_zero_le h0
    have hmlt : mm < ((P.defs[d]!).body.nodes[k]!).members := by omega
    have key := locW_child P hoff hk hp hmlt hnl hl
    have e : (n : Int) + (δ + (u : Int) * S) = (((P.defs[d]!).body.offsets[k]! +
        mm * (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates + n : Nat) : Int) := by
      rw [hδ, hS]
      push_cast
      rw [← hmm]
      ring
    rw [e, Int.toNat_natCast]
    exact key

/-- Copies of one member's segments, each of a copy, that name the body's gates with their widths, alike: a segment
naming them with theirs. -/
theorem gateW_alike (P : Prog) (d : Nat) {one : Array Seg} (h1 : ∀ x ∈ one.toList, x.reps = 1)
    (hW : ∀ s ∈ one.toList, GateW P d s) (len reps : Nat) :
    GateW P d { ps := one.flatMap (·.ps), len := len, reps := reps, shift := 0 } := by
  intro w p hx
  rw [segPos_alike rfl] at hx
  obtain ⟨_, -, hx⟩ := List.mem_flatMap.1 hx
  rw [posAll_flatMap_ps one h1] at hx
  unfold segsPos at hx
  obtain ⟨s, hs, hxs⟩ := List.mem_flatMap.1 hx
  exact hW s hs w p hxs

/-- **A batch of a composite's piece of one row and a stride of whole members**, as `batchStride` reads it, names the
body's gates with their widths: copies of a member's leaf of gates, a copy a member (`gateW_copies`); copies of what
`fromCallee` gives of a leaf of arguments the members share (`gateW_alike`); or position by position (`batchPts_w`).
`batchW_of`'s `hStride`. -/
theorem batchStride_w (c : Desc.Ctx) (hB : Resolve.BodyOk c.P c.d) (hL : Laid c.P c.d) (fuel : Nat)
    (hD : DownWAt c fuel) (hCW : CalleeW c fuel)
    (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn)
    (hRW : ∀ k node, c.node k = .ok node → RetW c.P c.memo node.fn)
    (hLF : ∀ k node, c.node k = .ok node → (c.P.defs[node.fn]!).prim = false → Laid c.P node.fn ∧
      (c.P.defs[node.fn]!).body.offsets[(c.P.defs[node.fn]!).body.nodes.size]! = (c.P.defs[node.fn]!).gates) :
    ∀ q k node segs, q.tgt = .ref false k → c.node k = .ok node → node.form = .batch →
      (c.P.defs[node.fn]!).prim = false → (c.P.defs[node.fn]!).outLeaves ≠ 0 → q.rows = 1 →
      (q.len > 1 ∧ q.stride % ((c.P.defs[node.fn]!).outLeaves : Int) = 0 ∧ q.base ≥ 0) →
      (∀ x ∈ pos q, 0 ≤ x.2) → (∀ x ∈ pos q, Fits c x) → batchStride c fuel k node q = .ok segs →
      ∀ s ∈ segs.toList, GateW c.P c.d s := by
  intro q k node segs ht hn hform hp hL0 h1 hcp hq hf h
  obtain ⟨hok, hnn, hgo, -, hpp⟩ := hR k node hn
  have hRWk := hRW k node hn
  obtain ⟨hLc, hlast⟩ := hLF k node hn hp
  have hdo := downOk_of hB hn
  have hk := node_lt hn
  have hnode := node_get hn
  have hl0 : 0 < q.len := by omega
  have hfu : q.at 0 < ((c.body.nodes[k]!).out.leaves : Int) := hf _ (at_mem_pos h1 hl0) k ht
  rw [hnode] at hfu
  have hbl : q.base.toNat < node.out.leaves := by
    have : q.at 0 = q.base := by simp [Desc.Piece.at]
    rw [this] at hfu
    omega
  have hdo0 := hdo q.base.toNat hbl
  simp only [hform] at hdo0
  have hmn := hdo0.1
  unfold batchStride at h
  dsimp only at h
  split at h
  · rename_i hrep
    split at h
    · simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      intro s hs
      simp only [List.mem_singleton] at hs
      subst hs
      have hleaf1 : ∀ x ∈ ((c.memo[node.fn]!).ret.slice (q.base.toNat % (c.P.defs[node.fn]!).outLeaves)
          (q.base.toNat % (c.P.defs[node.fn]!).outLeaves + 1)).toList, x.reps = 1 := fun x hx => by
        have h1' := (retOk_slice hgo hx).1
        have h2' := (Array.all_eq_true_iff_forall_mem.1 hrep) x (Array.mem_toList_iff.1 hx)
        simp only [decide_eq_true_eq] at h2'
        omega
      have hpsPos := posAll_flatMap_ps _ hleaf1
      rw [psSlice_pos _ hok] at hpsPos
      have hLp : 0 < (c.P.defs[node.fn]!).outLeaves := Nat.pos_of_ne_zero hL0
      have hLI : ((c.P.defs[node.fn]!).outLeaves : Int) ≠ 0 := by exact_mod_cast hL0
      obtain ⟨dd, hdd⟩ : ∃ dd, q.stride = ((c.P.defs[node.fn]!).outLeaves : Int) * dd :=
        ⟨q.stride / (c.P.defs[node.fn]!).outLeaves, (Int.mul_ediv_cancel' (Int.dvd_of_emod_eq_zero hcp.2.1)).symm⟩
      have hdiv : q.stride / ((c.P.defs[node.fn]!).outLeaves : Int) = dd := by
        rw [hdd]; exact Int.mul_ediv_cancel_left _ hLI
      have hB' : (q.base.toNat : Int) = q.base := Int.toNat_of_nonneg hcp.2.2
      have h2 : (q.base : Int) = ((q.base.toNat / (c.P.defs[node.fn]!).outLeaves : Nat) : Int) *
          ((c.P.defs[node.fn]!).outLeaves : Int) + ((q.base.toNat % (c.P.defs[node.fn]!).outLeaves : Nat) : Int) := by
        rw [← hB']
        exact_mod_cast (Nat.div_add_mod' q.base.toNat (c.P.defs[node.fn]!).outLeaves).symm
      have hjL : ((q.base.toNat % (c.P.defs[node.fn]!).outLeaves : Nat) : Int) <
          ((c.P.defs[node.fn]!).outLeaves : Int) := by exact_mod_cast Nat.mod_lt _ hLp
      subst hnode
      refine gateW_copies c.P hL hk hp hLc hlast (m0 := q.base.toNat / (c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves)
        (dm := q.stride / ((c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves : Int)) rfl rfl ?_ ?_
      · intro w v hx
        rw [hpsPos] at hx
        have hx' := List.mem_of_mem_drop (List.mem_of_mem_take hx)
        exact ⟨(hnn _ hx').1, hRWk w v hx'⟩
      · intro u hu
        rw [hdiv]
        have hat : q.at u = (((q.base.toNat / (c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves : Nat) : Int) +
            (u : Int) * dd) * ((c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves : Int) +
            ((q.base.toNat % (c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves : Nat) : Int) := by
          unfold Desc.Piece.at
          rw [hdd]
          linear_combination h2
        have hEq : q.at u / ((c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves : Int) =
            ((q.base.toNat / (c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves : Nat) : Int) + (u : Int) * dd := by
          rw [hat, Int.add_comm, Int.add_mul_ediv_right _ _ hLI,
            Int.ediv_eq_zero_of_lt (Int.natCast_nonneg _) hjL, Int.zero_add]
        have hm := at_mem_pos h1 hu
        have hE0 : 0 ≤ q.at u := hq _ hm
        have hEl : q.at u < ((c.body.nodes[k]!).out.leaves : Int) := hf _ hm k ht
        obtain ⟨E, hE⟩ := Int.eq_ofNat_of_zero_le hE0
        rw [hE] at hEq hEl
        have hEl' : E < (c.body.nodes[k]!).out.leaves := by exact_mod_cast hEl
        have hdE := (hdo E hEl')
        simp only [hform] at hdE
        have hcast : ((E / (c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves : Nat) : Int) =
            (E : Int) / ((c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves : Int) := by norm_cast
        rw [← hEq, ← hcast]
        refine ⟨Int.natCast_nonneg _, ?_⟩
        have hmb : ((c.P.defs[c.d]!).body.nodes[k]!).members = (c.body.nodes[k]!).n := members_batch hform
        rw [hmb]
        exact_mod_cast hdE.1
    · split at h
      · obtain ⟨one, hone, h⟩ := Walk.except_bind_ok h
        split at h
        · rename_i hone1
          simp only [pure, Except.pure, Except.ok.injEq] at h
          subst h
          intro s hs
          simp only [List.mem_singleton] at hs
          subst hs
          have hrep1 : ∀ x ∈ one.toList, x.reps = 1 := by
            intro x hx
            have h0 := ((reps_fuel c fuel).2.1 k _ _ _ one hone x hx).1
            have hle := (Array.all_eq_true_iff_forall_mem.1 hone1) x (Array.mem_toList_iff.1 hx)
            simp only [decide_eq_true_eq] at hle
            omega
          have hmem : q.base.toNat / (c.P.defs[node.fn]!).outLeaves < (c.body.nodes[k]!).members := by
            rw [hnode, members_batch hform]; exact hmn
          exact gateW_alike c.P c.d hrep1 (hCW k _ _ _ one hmem (by rw [hnode]; exact hp) hone) 1 q.len
        · exact batchPts_w c fuel hD q ht h1 hq hf h
      · exact batchPts_w c fuel hD q ht h1 hq hf h
  · exact batchPts_w c fuel hD q ht h1 hq hf h

/-- A batch's output leaves `[a, b)` from `mid` and what `fromCallee` keeps of the cut members at either end, of a
composite: naming the body's gates with their widths wherever `mid`'s segments do (`ends_shape`'s argument). -/
theorem ends_w (c : Desc.Ctx) (fuel : Nat) (hCW : CalleeW c fuel) {k L a b : Nat} (hL : L ≠ 0)
    (hab : (a + L - 1) / L ≤ b / L) (hp : (c.P.defs[(c.body.nodes[k]!).fn]!).prim = false)
    (hm : ∀ j, a ≤ j → j < b → j / L < (c.body.nodes[k]!).members)
    {mid segs : Array Seg} (hmid : ∀ s ∈ mid.toList, GateW c.P c.d s)
    (h : batchEnds c fuel k L a b mid = .ok segs) : ∀ s ∈ segs.toList, GateW c.P c.d s := by
  unfold batchEnds at h
  dsimp only at h
  generalize hm0 : (a + L - 1) / L = m0 at *
  generalize hm1 : b / L = m1 at *
  have hLp : 0 < L := Nat.pos_of_ne_zero hL
  have ha0 : a ≤ m0 * L := by
    rw [← hm0]
    have := Nat.div_add_mod (a + L - 1) L
    have := Nat.mod_lt (a + L - 1) hLp
    have : L * ((a + L - 1) / L) = (a + L - 1) / L * L := Nat.mul_comm _ _
    omega
  have hb0 : m1 * L ≤ b := by rw [← hm1]; exact Nat.div_mul_le_self _ _
  have hmm : m0 * L ≤ m1 * L := Nat.mul_le_mul_right _ hab
  have hfirst : ∀ x, a < m0 * L → Desc.fromCallee c fuel k (a / L) (a % L) L = .ok x →
      ∀ s ∈ x.toList, GateW c.P c.d s := fun x hlt hx =>
    hCW k (a / L) (a % L) L x (hm a le_rfl (by omega)) hp hx
  have hlast : ∀ y, m1 * L < b → Desc.fromCallee c fuel k m1 0 (b - m1 * L) = .ok y →
      ∀ s ∈ y.toList, GateW c.P c.d s := by
    intro y hlt hy
    have hmB := hm (m1 * L) (by omega) hlt
    rw [Nat.mul_div_cancel _ hLp] at hmB
    exact hCW k m1 0 (b - m1 * L) y hmB hp hy
  obtain ⟨A, B, rfl, hA, hB⟩ : ∃ A B, segs = A ++ mid ++ B ∧ (∀ s ∈ A.toList, GateW c.P c.d s) ∧
      ∀ s ∈ B.toList, GateW c.P c.d s := by
    split at h
    · rename_i hlt
      obtain ⟨x, hx, h⟩ := Walk.except_bind_ok h
      split at h
      · rename_i hlt'
        obtain ⟨y, hy, h⟩ := Walk.except_bind_ok h
        simp only [pure, Except.pure, Except.ok.injEq] at h
        exact ⟨x, y, by rw [← h]; try simp, hfirst x hlt hx, hlast y hlt' hy⟩
      · simp only [pure, Except.pure, Except.ok.injEq] at h
        exact ⟨x, #[], by rw [← h]; try simp, hfirst x hlt hx, by simp⟩
    · split at h
      · rename_i hlt'
        obtain ⟨y, hy, h⟩ := Walk.except_bind_ok h
        simp only [pure, Except.pure, Except.ok.injEq] at h
        exact ⟨#[], y, by rw [← h]; try simp, by simp, hlast y hlt' hy⟩
      · simp only [pure, Except.pure, Except.ok.injEq] at h
        exact ⟨#[], #[], by rw [← h]; try simp, by simp, by simp⟩
  intro s hs
  simp only [Array.toList_append, List.mem_append] at hs
  rcases hs with (hs | hs) | hs
  · exact hA s hs
  · exact hmid s hs
  · exact hB s hs

/-- A batch's output leaves `[a, b)` member by member, of a composite: naming the body's gates with their widths. -/
theorem batchMembers_w (c : Desc.Ctx) (fuel : Nat) (hCW : CalleeW c fuel) {k L a b : Nat}
    (hp : (c.P.defs[(c.body.nodes[k]!).fn]!).prim = false) (hm : (b - 1) / L < (c.body.nodes[k]!).members)
    {segs : Array Seg} (h : batchMembers c fuel k L a b = .ok segs) : ∀ s ∈ segs.toList, GateW c.P c.d s := by
  unfold batchMembers at h
  have h := walkAtMost_bind _ _ _ h
  dsimp only at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, Nat.add_sub_cancel, Nat.div_one] at h
  exact loop_mem (GateW c.P c.d)
    (fun m => Desc.fromCallee c fuel k m (max a (m * L) - m * L) (min b ((m + 1) * L) - m * L)) _
    (fun m hmm segs hs => by
      rw [List.mem_range'_1] at hmm
      exact hCW k m _ _ segs (by omega) hp hs) segs h

/-- `down` of the piece `throughPS` gives of an argument whose pieces are of positive count: segments naming the body's
gates with their widths, its positions being the argument's references. -/
theorem throughPS_w (c : Desc.Ctx) (fuel : Nat) (hD : DownWAt c fuel) {k i : Nat} (hA : ArgOk c k i)
    (hP : ∀ s ∈ (c.arg k i).segs.toList, ∀ q ∈ s.ps.toList, 0 < q.count)
    (hR : ∀ t < ((c.body.nodes[k]!).args[i]!).seq.count,
      Resolve.RefOk c.P c.d k (((c.body.nodes[k]!).args[i]!).seq.get t))
    {p bp : Piece} (h : Desc.throughPS (c.arg k i) p = some bp) {segs : Array Seg}
    (hs : Desc.down c fuel bp = .ok segs) : ∀ s ∈ segs.toList, GateW c.P c.d s := by
  obtain ⟨hcnt, hmem⟩ := throughPS_mem h hA.1 hP
  have hmem' : ∀ x ∈ pos bp, ∃ t < ((c.body.nodes[k]!).args[i]!).seq.count,
      x = refPos (((c.body.nodes[k]!).args[i]!).seq.get t) := by
    intro x hx
    have := hmem x hx
    rw [hA.2] at this
    obtain ⟨t, ht, rfl⟩ := List.mem_map.1 this
    exact ⟨t, List.mem_range.1 ht, rfl⟩
  exact hD bp segs hcnt
    (fun x hx => by obtain ⟨t, -, rfl⟩ := hmem' x hx; simp [refPos])
    (fun x hx k' hk' => by
      obtain ⟨t, ht, rfl⟩ := hmem' x hx
      simp only [refPos, Tgt.ref.injEq] at hk'
      obtain ⟨hp', rfl⟩ := hk'
      have := ((hR t ht).2 hp').2
      simp only [refPos]
      exact_mod_cast this)
    (fun x hx i' hi' => by
      obtain ⟨t, ht, rfl⟩ := hmem' x hx
      simp only [refPos, Tgt.ref.injEq] at hi'
      obtain ⟨hp', rfl⟩ := hi'
      have := (hR t ht).1 hp'
      simp only [refPos]
      exact ⟨this.1, by exact_mod_cast this.2⟩)
    (fun w v hx => by
      obtain ⟨t, -, he⟩ := hmem' _ hx
      simp [refPos] at he) hs

/-- A batch's output leaves `[a, b)` through `throughPS`, else member by member, of a composite: naming the body's
gates with their widths. -/
theorem batchThrough_w (c : Desc.Ctx) (hB : Resolve.BodyOk c.P c.d) (fuel : Nat) (hD : DownWAt c fuel)
    (hCW : CalleeW c fuel)
    (hA : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size → ArgOk c k i)
    (hAP : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size →
      ∀ s ∈ (c.arg k i).segs.toList, ∀ q ∈ s.ps.toList, 0 < q.count)
    {k : Nat} {node : Flock.Program.Node} (hn : c.node k = .ok node) (hL : (c.P.defs[node.fn]!).outLeaves ≠ 0)
    (hp : (c.P.defs[(c.body.nodes[k]!).fn]!).prim = false) {a b : Nat}
    (hm : ∀ j, a ≤ j → j < b → j / (c.P.defs[node.fn]!).outLeaves < (c.body.nodes[k]!).members)
    (hmB : (b - 1) / (c.P.defs[node.fn]!).outLeaves < (c.body.nodes[k]!).members)
    {segs : Array Seg} (h : batchThrough c fuel k node a b = .ok segs) : ∀ s ∈ segs.toList, GateW c.P c.d s := by
  have hmem := fun (hh : batchMembers c fuel k (c.P.defs[node.fn]!).outLeaves a b = .ok segs) =>
    batchMembers_w c fuel hCW hp hmB hh
  unfold batchThrough at h
  dsimp only at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hc
    split at h
    · rename_i i hti
      split at h
      · rename_i d Li hdl
        split at h
        · split at h
          · rename_i bp hbp
            obtain ⟨x, hx, h⟩ := Walk.except_bind_ok h
            obtain ⟨a', ha, -⟩ := Option.bind_eq_some_iff.1 hdl
            obtain ⟨hia, -⟩ := Array.getElem?_eq_some_iff.1 ha
            have hnode := node_get hn
            have hk : k < c.body.nodes.size := node_lt hn
            have hia' : i < (c.body.nodes[k]!).args.size := by rw [hnode]; exact hia
            have hRk : ∀ t < ((c.body.nodes[k]!).args[i]!).seq.count,
                Resolve.RefOk c.P c.d k (((c.body.nodes[k]!).args[i]!).seq.get t) := (hB.1 k hk).2.1 i
            exact ends_w c fuel hCW hL (by omega) hp hm
              (throughPS_w c fuel hD (hA k i hk hia') (hAP k i hk hia') hRk hbp hx) h
          · exact hmem h
        · exact hmem h
      · exact hmem h
    · exact hmem h
  · exact hmem h

/-- **A batch of a composite's piece of one row and stride 1 past its one-run case**, as `batchOne` reads it, names the
body's gates with their widths: whole members as copies of a return of gates, a copy a member (`gateW_copies`), or of
what `fromCallee` keeps of a member's return of shared leaves (`gateW_alike`), through `throughPS`, or member by
member, the cut members at either end each through its return (`ends_w`, `batchThrough_w`). `batchW_of`'s `hOne`. -/
theorem batchOne_w (c : Desc.Ctx) (hB : Resolve.BodyOk c.P c.d) (hLd : Laid c.P c.d) (fuel : Nat)
    (hD : DownWAt c fuel) (hCW : CalleeW c fuel)
    (hA : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size → ArgOk c k i)
    (hAP : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size →
      ∀ s ∈ (c.arg k i).segs.toList, ∀ q ∈ s.ps.toList, 0 < q.count)
    (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn)
    (hRW : ∀ k node, c.node k = .ok node → RetW c.P c.memo node.fn)
    (hLF : ∀ k node, c.node k = .ok node → (c.P.defs[node.fn]!).prim = false → Laid c.P node.fn ∧
      (c.P.defs[node.fn]!).body.offsets[(c.P.defs[node.fn]!).body.nodes.size]! = (c.P.defs[node.fn]!).gates) :
    BatchOneW c fuel := by
  intro q k node segs ht hn hform hp hL h0 h1 hs hq hf h
  have hlen : 0 < q.len := by rw [← count_one h1]; omega
  have hdo := downOk_of hB hn
  have hnode := node_get hn
  have hp' : (c.P.defs[(c.body.nodes[k]!).fn]!).prim = false := by rw [hnode]; exact hp
  have hbl : q.base.toNat + q.len ≤ node.out.leaves := by
    have hm := at_mem_pos h1 (show q.len - 1 < q.len by omega)
    have hb := hq _ (at_mem_pos h1 hlen)
    have := hf _ hm k ht
    simp only [hnode, Desc.Piece.at, hs] at this hb
    omega
  have hm : ∀ j, q.base.toNat ≤ j → j < q.base.toNat + q.len →
      j / (c.P.defs[node.fn]!).outLeaves < (c.body.nodes[k]!).members := by
    intro j _ hj
    have := hdo j (by omega)
    simp only [hform] at this
    rw [hnode, members_batch hform]
    exact this.1
  have hmB := hm (q.base.toNat + q.len - 1) (by omega) (by omega)
  have hm1 : (q.base.toNat + q.len) / (c.P.defs[node.fn]!).outLeaves ≤ (c.body.nodes[k]!).members := by
    have hLp := Nat.pos_of_ne_zero hL
    have e1 := Nat.div_mul_le_self (q.base.toNat + q.len) (c.P.defs[node.fn]!).outLeaves
    have e5 := Nat.lt_div_mul_add (a := q.base.toNat + q.len - 1) hLp
    have e6 : (q.base.toNat + q.len) / (c.P.defs[node.fn]!).outLeaves * (c.P.defs[node.fn]!).outLeaves ≤
        ((q.base.toNat + q.len - 1) / (c.P.defs[node.fn]!).outLeaves + 1) * (c.P.defs[node.fn]!).outLeaves := by
      rw [Nat.add_mul, Nat.one_mul]; omega
    have := Nat.le_of_mul_le_mul_right e6 hLp
    omega
  unfold batchOne at h
  dsimp only at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
    have hgo : ∀ x ∈ (c.memo[node.fn]!).ret.segs.toList, x.reps ≤ 1 := by
      intro x hx
      have := (Array.all_eq_true_iff_forall_mem.1 hc.1) x (Array.mem_toList_iff.1 hx)
      simp only [Bool.and_eq_true, decide_eq_true_eq] at this
      exact this.1
    refine ends_w c fuel hCW hL (by omega) hp' hm ?_ h
    intro s hs
    simp only [List.mem_singleton] at hs
    subst hs
    obtain ⟨hok, hnn, -, -, -⟩ := hR k node hn
    have hRWk := hRW k node hn
    obtain ⟨hLc, hlast⟩ := hLF k node hn hp
    have hflat := psPos_flat _ hok hgo
    have hk := node_lt hn
    subst hnode
    have hm1' : (q.base.toNat + q.len) / (c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves ≤
        ((c.P.defs[c.d]!).body.nodes[k]!).members := hm1
    refine gateW_copies c.P hLd hk hp hLc hlast
      (m0 := (q.base.toNat + (c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves - 1) /
        (c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves) (dm := 1) rfl (Int.one_mul _).symm ?_ ?_
    · intro w v hx
      rw [← hflat] at hx
      exact ⟨(hnn _ hx).1, hRWk w v hx⟩
    · intro u hu
      generalize (q.base.toNat + (c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves - 1) /
        (c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves = m0 at hu ⊢
      generalize (q.base.toNat + q.len) / (c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves = m1 at hu hm1'
      constructor <;> omega
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
    obtain ⟨one, hone, h⟩ := Walk.except_bind_ok h
    split at h
    · rename_i hone1
      refine ends_w c fuel hCW hL (by omega) hp' hm ?_ h
      intro s hs
      simp only [List.mem_singleton] at hs
      subst hs
      have hrep1 : ∀ x ∈ one.toList, x.reps = 1 := by
        intro x hx
        have h0 := ((reps_fuel c fuel).2.1 k _ _ _ one hone x hx).1
        have hle := (Array.all_eq_true_iff_forall_mem.1 hone1) x (Array.mem_toList_iff.1 hx)
        simp only [decide_eq_true_eq] at hle
        omega
      exact gateW_alike c.P c.d hrep1 (hCW k _ _ _ one (by omega) hp' hone) _ _
    · exact batchThrough_w c hB fuel hD hCW hA hAP hn hL hp' hm hmB h
  · exact batchThrough_w c hB fuel hD hCW hA hAP hn hL hp' hm hmB h

/-- **`down`, `fromCallee`, `throughArg` and `carry` name the body's gates with their widths, at every fuel**: `widths`
with a batch's cases (`batchW_of`) and a scan's (`scanW_of`). -/
theorem widths_of (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (hB : Resolve.BodyOk c.P c.d)
    (hL : Laid c.P c.d)
    (hG1 : ∀ f : Nat, (c.P.defs[f]!).prim = true → (c.P.defs[f]!).gates = 1)
    (hP1 : ∀ f : Nat, (c.P.defs[f]!).prim = true → (c.P.defs[f]!).outLeaves = 1)
    (hCP : ∀ k node, c.node k = .ok node → (node.form = .prim → (c.P.defs[node.fn]!).prim = true) ∧
      (node.form = .call ∨ node.form = .scan → (c.P.defs[node.fn]!).prim = false))
    (hA : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size → ArgOk c k i)
    (hAP : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size →
      ∀ s ∈ (c.arg k i).segs.toList, ∀ q ∈ s.ps.toList, 0 < q.count)
    (hL0 : ∀ k i d L : Nat, ((c.body.nodes[k]!).args[i]!).dims = some (d, L) → L = 0 →
      i < (c.P.defs[(c.body.nodes[k]!).fn]!).body.params.size →
      ((c.P.defs[(c.body.nodes[k]!).fn]!).body.params[i]!).2.leaves = 0)
    (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn)
    (hRW : ∀ k node, c.node k = .ok node → RetW c.P c.memo node.fn)
    (hLF : ∀ k node, c.node k = .ok node → (c.P.defs[node.fn]!).prim = false → Laid c.P node.fn ∧
      (c.P.defs[node.fn]!).body.offsets[(c.P.defs[node.fn]!).body.nodes.size]! = (c.P.defs[node.fn]!).gates) :
    ∀ fuel, DownWAt c fuel ∧ CalleeW c fuel ∧ ThroughW c fuel ∧ CarryW c fuel :=
  widths c sc hd hB hL hG1 hP1 hCP hA hAP hL0 hR hRW hLF
    (fun fuel hD hC => batchW_of c fuel hD (batchAffine_w c hB hL hR hRW hLF)
      (batchOne_w c hB hL fuel hD hC hA hAP hR hRW hLF) (batchStride_w c hB hL fuel hD hC hR hRW hLF))
    (fun fuel hC hK => scanW_of c hB fuel hC hK)

/-- **The decoder's facts the widths read**, as `Flock.Program.getDef` gives every Definition and `bodiesLaidOut`
checks: a primitive is one gate, a composite's body is laid out from its first gate to its last (`Walk.LaidOut`), and
its nodes call what the decoder lets them: a primitive node a primitive, a Call or a scan a composite. -/
def WidthFacts (P : Prog) : Prop :=
  ∀ d < P.defs.size, ((P.defs[d]!).prim = true → (P.defs[d]!).gates = 1) ∧
    ((P.defs[d]!).prim = false → Laid P d ∧
      (P.defs[d]!).body.offsets[(P.defs[d]!).body.nodes.size]! = (P.defs[d]!).gates ∧
      ∀ k < (P.defs[d]!).body.nodes.size,
        (((P.defs[d]!).body.nodes[k]!).form = .prim → (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).prim = true) ∧
        (((P.defs[d]!).body.nodes[k]!).form = .call ∨ ((P.defs[d]!).body.nodes[k]!).form = .scan →
          (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).prim = false))

/-- **`down`, `fromCallee`, `throughArg` and `carry` name a decoded composite's gates with their widths**, from
`widths_of` with the decoder's facts (`Decoded`, `WidthFacts`), wherever its callees' summaries hold `RetOk` and name
theirs (`RetW`). -/
theorem widths_decoded (P : Prog) (memo : Array Desc.Summary) {d : Nat} (hdP : d < P.defs.size)
    (hp : (P.defs[d]!).prim = false) (hDec : Decoded P) (hWF : WidthFacts P) (hB : Resolve.BodyOk P d)
    (hR : ∀ k node, (ctxOf P memo d).node k = .ok node → RetOk P memo node.fn)
    (hRW : ∀ k node, (ctxOf P memo d).node k = .ok node → RetW P memo node.fn) :
    ∀ fuel, DownWAt (ctxOf P memo d) fuel ∧ CalleeW (ctxOf P memo d) fuel ∧
      ThroughW (ctxOf P memo d) fuel ∧ CarryW (ctxOf P memo d) fuel := by
  have hfn : ∀ k node, (ctxOf P memo d).node k = .ok node → node.fn < P.defs.size := by
    intro k node hn
    have hk := node_lt hn
    have := (hB.1 k hk).1
    have hg : (P.defs[d]!).body.nodes[k]! = node := node_get hn
    rwa [hg] at this
  have hpr : ∀ f : Nat, (P.defs[f]!).prim = true → f < P.defs.size := by
    intro f hf
    by_contra hlt
    rw [getElem!_neg P.defs f hlt] at hf
    cases hf
  obtain ⟨hLd, -, hnodes⟩ := (hWF d hdP).2 hp
  refine widths_of (ctxOf P memo d) (.top d 0 none) rfl hB hLd
    (fun f hf => (hWF f (hpr f hf)).1 hf) (fun f hf => (hDec f (hpr f hf)).1 hf)
    (fun k node hn => by
      have hk := node_lt hn
      have hg : (P.defs[d]!).body.nodes[k]! = node := node_get hn
      rw [← hg]
      exact hnodes k hk)
    (argOk_decoded P memo hdP hDec) (fun k i hk hi => ctxOf_argCount P memo d k i hk hi) ?_ hR hRW
    (fun k node hn hp' => ⟨((hWF node.fn (hfn k node hn)).2 hp').1, ((hWF node.fn (hfn k node hn)).2 hp').2.1⟩)
  intro k i e L hdims hL hip
  change ((((P.defs[d]!).body.nodes[k]!).args[i]!).dims = some (e, L)) at hdims
  by_cases hk : k < (P.defs[d]!).body.nodes.size
  · by_cases hi : i < ((P.defs[d]!).body.nodes[k]!).args.size
    · exact ((hDec d hdP).2.2 k hk i hi).2.2 e L hdims hL hip
    · rw [getElem!_neg _ i hi] at hdims; cases hdims
  · rw [getElem!_neg _ k hk, getElem!_neg _ i (Nat.not_lt_zero i)] at hdims; cases hdims

/-- **What `Desc.summarize` gives Definition `d` names its gates with their widths** (`RetW`), wherever `down`'s
segments are `ShapeOk` (`DownShape`) and name the body's gates with their widths (`DownWAt`) in its body, and the
decoder's facts on its return hold (`summarize_retOk`'s argument). -/
theorem summarize_retW (P : Prog) (memo : Array Desc.Summary) (d : Nat) (s : Desc.Summary)
    (hdP : d < P.defs.size) (hRP : RetParts P) (hB : Resolve.BodyOk P d)
    (hS : (P.defs[d]!).prim = false → DownShape (ctxOf P memo d))
    (hW : (P.defs[d]!).prim = false → DownWAt (ctxOf P memo d) Flock.Steps.FUEL)
    (h : Desc.summarize P memo d = .ok s) (m : Array Desc.Summary) (hm : m[d]! = s) : RetW P m d := by
  unfold RetW
  rw [hm]
  unfold Desc.summarize at h
  extract_lets defn body c ret0 acc0 at h
  split at h
  · simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    intro w p hx
    rw [psPos_default] at hx
    cases hx
  · rename_i hp
    have hp' : (P.defs[d]!).prim = false := by simpa using hp
    obtain ⟨ret, hret, h⟩ := Walk.except_bind_ok h
    obtain ⟨acc, -, h⟩ := Walk.except_bind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    rw [← Array.forIn_toList] at hret
    obtain ⟨outs, hF, hout⟩ := forIn_trace2 _ _ ret0 ret hret
    obtain ⟨hpe, hnn⟩ := (hRP d hdP).2 hp'
    have hpe' : body.ret = Seq.ofParts body.ret.parts := hpe
    have hnn' : ∀ q ∈ body.ret.parts.toList, PartNonneg q := hnn
    have hpos : psPos (Desc.ofSeq body.ret) = (List.range body.ret.count).map fun t => refPos (body.ret.get t) := by
      rw [(ofSeq_pos body.ret hnn').2.1]
      conv => rhs; rw [hpe']
      rw [partsPos_get]
    have hqs : ((Desc.ofSeq body.ret).segs.toList.flatMap fun s => s.ps.toList).flatMap pos =
        psPos (Desc.ofSeq body.ret) := by
      rw [← segsPos_pieces _ (ofSeq_reps body.ret)]; rfl
    have hx : ∀ q ∈ ((Desc.ofSeq body.ret).segs.toList.flatMap fun s => s.ps.toList), ∀ x ∈ pos q,
        ∃ t < body.ret.count, x = refPos (body.ret.get t) := by
      intro q hq x hx
      have : x ∈ psPos (Desc.ofSeq body.ret) := by
        rw [← hqs]; exact List.mem_flatMap.2 ⟨q, hq, hx⟩
      rw [hpos, List.mem_map] at this
      obtain ⟨t, ht, e⟩ := this
      exact ⟨t, List.mem_range.1 ht, e.symm⟩
    have hrefOk : ∀ t < body.ret.count, Resolve.RefOk P d body.nodes.size (body.ret.get t) := hB.2
    have hq1 : ∀ q ∈ ((Desc.ofSeq body.ret).segs.toList.flatMap fun s => s.ps.toList), ∀ x ∈ pos q, 0 ≤ x.2 := by
      intro q hq x hxq
      obtain ⟨t, -, rfl⟩ := hx q hq x hxq
      simp [refPos]
    have hq2 : ∀ q ∈ ((Desc.ofSeq body.ret).segs.toList.flatMap fun s => s.ps.toList), ∀ x ∈ pos q,
        Fits (ctxOf P memo d) x := by
      intro q hq x hxq k hk
      obtain ⟨t, ht, rfl⟩ := hx q hq x hxq
      simp only [refPos, Tgt.ref.injEq] at hk
      have := (hrefOk t ht).2 hk.1
      rw [← hk.2]
      show ((body.ret.get t).j : Int) < _
      exact_mod_cast this.2
    have hq3 : ∀ q ∈ ((Desc.ofSeq body.ret).segs.toList.flatMap fun s => s.ps.toList), ∀ x ∈ pos q,
        ∀ i, x.1 = .ref true i → i < (ctxOf P memo d).body.params.size ∧
          x.2 < (((ctxOf P memo d).body.params[i]!).2.leaves : Int) := by
      intro q hq x hxq i hi
      obtain ⟨t, ht, rfl⟩ := hx q hq x hxq
      simp only [refPos, Tgt.ref.injEq] at hi
      have := (hrefOk t ht).1 hi.1
      rw [← hi.2]
      exact ⟨this.1, by show ((body.ret.get t).j : Int) < _; exact_mod_cast this.2⟩
    have hq4 : ∀ q ∈ ((Desc.ofSeq body.ret).segs.toList.flatMap fun s => s.ps.toList), ∀ w p,
        (Tgt.gate w, p) ∈ pos q → LocW P d p.toNat w := by
      intro q hq w p hxq
      obtain ⟨t, -, he⟩ := hx q hq _ hxq
      simp [refPos] at he
    have hq0 : ∀ q ∈ ((Desc.ofSeq body.ret).segs.toList.flatMap fun s => s.ps.toList), 0 < q.count := by
      intro q hq
      obtain ⟨g, hg, hqg⟩ := List.mem_flatMap.1 hq
      unfold Desc.ofSeq at hg
      rw [of_segs, Array.toList_filter, List.mem_filter, Array.toList_map, List.mem_map] at hg
      obtain ⟨⟨q', -, rfl⟩, hc⟩ := hg
      have e : q = q' := by simpa [Seg.one] using hqg
      subst e
      simpa [Desc.Seg.count, Seg.one] using hc
    have hgo : ∀ g ∈ ret.toList, ShapeOk P d g ∧ GateW P d g := by
      intro g hg
      rw [hout] at hg
      rcases mem_foldl outs ret0 g hg with hg | ⟨v, hv, hg⟩
      · exact absurd hg (by simp [ret0])
      · obtain ⟨q, hq, hqv⟩ := forall2_mem hF v hv
        exact ⟨hS hp' Flock.Steps.FUEL q v (hq0 q hq) (hq1 q hq) (hq2 q hq) (hq3 q hq) hqv g hg,
          hW hp' q v (hq0 q hq) (hq1 q hq) (hq2 q hq) (hq3 q hq) (hq4 q hq) hqv g hg⟩
    obtain ⟨-, hpsPos, -⟩ := of_ok (Desc.coalesce ret) fun g hg => (shapeOk_coalesce (fun g hg => (hgo g hg).1) g hg).1
    rw [coalesce_pos] at hpsPos
    intro w p hx'
    rw [hpsPos] at hx'
    obtain ⟨g, hg, hxg⟩ := List.mem_flatMap.1 hx'
    exact (hgo g hg).2 w p hxg

theorem retW_push {P : Prog} {r : Array Desc.Summary} {f : Nat} (hf : f < r.size) (v : Desc.Summary)
    (h : RetW P r f) : RetW P (r.push v) f := by
  unfold RetW at h ⊢
  rwa [getElem!_pos (r.push v) f (by simp; omega), Array.getElem_push_lt hf, ← getElem!_pos r f hf]

/-- **Every summary `memoOf` makes of a reached Definition other than the root holds `RetOk` and names its gates with
their widths** (`RetW`), wherever `memoOf_retOk`'s facts and the widths' (`WidthFacts`) hold. -/
theorem memoOf_retW (P : Prog) (memo : Array Desc.Summary) (h : memoOf P = .ok memo)
    (hRP : RetParts P) (hDec : Decoded P) (hWF : WidthFacts P)
    (hcf : ∀ d < P.defs.size, ∀ k < (P.defs[d]!).body.nodes.size, ((P.defs[d]!).body.nodes[k]!).fn < d)
    (hnr : ∀ d < P.defs.size, (Desc.reached P)[d]! = true →
      ∀ k < (P.defs[d]!).body.nodes.size, ((P.defs[d]!).body.nodes[k]!).fn ≠ P.root)
    (hB : ∀ d < P.defs.size, (Desc.reached P)[d]! = true → Resolve.BodyOk P d)
    (hSO : ∀ m d, ScanOut (ctxOf P m d)) :
    ∀ f < P.defs.size, (Desc.reached P)[f]! = true → f ≠ P.root → RetOk P memo f ∧ RetW P memo f := by
  unfold memoOf at h
  extract_lets seen memo0 at h
  have h' := ok_bind_pure h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h'
  have := forIn_range_idx (fun t (r : Array Desc.Summary) => r.size = t ∧
      ∀ f < t, (Desc.reached P)[f]! = true → f ≠ P.root → RetOk P r f ∧ RetW P r f) _ P.defs.size ?_
    P.defs.size 0 memo0 memo (by simp) ⟨rfl, fun f hf => absurd hf (Nat.not_lt_zero _)⟩ (by simpa using h')
  · exact fun f hf => (this.2 f (by simpa using hf))
  intro t r y htP ⟨hsz, hI⟩ hy
  obtain ⟨v, rfl, hv⟩ : ∃ v, y = .yield (r.push v) ∧
      ((seen[t]! = true ∧ ¬t = P.root) → Desc.summarize P r t = .ok v) := by
    split at hy
    · rcases hs : Desc.summarize P r t with e | v
      · rw [hs] at hy; cases hy
      · rw [hs] at hy
        change Except.ok (ForInStep.yield (r.push v)) = Except.ok y at hy
        injection hy with hy
        exact ⟨v, hy.symm, fun _ => rfl⟩
    · rename_i hc
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      exact ⟨default, hy.symm, fun h => absurd h hc⟩
  refine ⟨_, rfl, by simp [hsz], fun f hf hrf hfr => ?_⟩
  rcases Nat.lt_succ_iff_lt_or_eq.1 hf with hf | rfl
  · exact ⟨retOk_push (by omega) v (hI f hf hrf hfr).1, retW_push (by omega) v (hI f hf hrf hfr).2⟩
  · replace hv := hv ⟨hrf, hfr⟩
    have hRR : (P.defs[f]!).prim = false → ∀ k node, (ctxOf P r f).node k = .ok node →
        RetOk P r node.fn ∧ RetW P r node.fn := by
      intro hp k node hn
      have hk := node_lt hn
      have hg := node_get hn
      change k < (P.defs[f]!).body.nodes.size at hk
      change (P.defs[f]!).body.nodes[k]! = node at hg
      rw [← hg]
      exact hI _ (hcf f htP k hk) (reached_closed P hcf f hrf hp k hk) (hnr f htP hrf k hk)
    have hlast : (r.push v)[f]! = v := by
      subst hsz
      rw [getElem!_pos (r.push v) r.size (by simp), Array.getElem_push_eq]
    have hS : (P.defs[f]!).prim = false → DownShape (ctxOf P r f) := fun hp =>
      downShape_decoded P r htP hDec (hB f htP hrf) (fun k n hn => (hRR hp k n hn).1) (hSO r f)
    refine ⟨summarize_retOk P r f v htP hRP (hB f htP hrf) hS
      (fun hp sc hsc => (specs_decoded P r htP hDec (hB f htP hrf) (fun k n hn => (hRR hp k n hn).1) (hSO r f) sc
        hsc Flock.Steps.FUEL).1) hv (r.push v) hlast, ?_⟩
    exact summarize_retW P r f v htP hRP (hB f htP hrf) hS
      (fun hp => (widths_decoded P r htP hp hDec hWF (hB f htP hrf) (fun k n hn => (hRR hp k n hn).1)
        (fun k n hn => (hRR hp k n hn).2) Flock.Steps.FUEL).1) hv (r.push v) hlast

/-- A root gate of width `w` (`LocW`) has width `w` in `circuitOf`'s circuit. -/
theorem w_of_locW {P : Prog} {c : Circuit} (hc : circuitOf P = .ok c) {g w : Nat} (hg : g < gates P)
    (h : LocW P P.root g w) : w = c.w[g]! :=
  locW_unique h (locW_root hc hg)

/-- A piece argument `i` of node `k` gives (`PS.pieces`), through `down`: segments naming the body's gates with their
widths, the argument naming references. -/
theorem piece_w (c : Desc.Ctx) (fuel : Nat) (hD : DownWAt c fuel) {k i : Nat} (hA : ArgOk c k i)
    (hP : ∀ s ∈ (c.arg k i).segs.toList, ∀ q ∈ s.ps.toList, 0 < q.count)
    (hR : ∀ t < ((c.body.nodes[k]!).args[i]!).seq.count,
      Resolve.RefOk c.P c.d k (((c.body.nodes[k]!).args[i]!).seq.get t))
    {lo hi : Nat} {q : Piece} (hq : q ∈ (c.arg k i).pieces lo hi) {segs : Array Seg}
    (h : Desc.down c fuel q = .ok segs) : ∀ s ∈ segs.toList, GateW c.P c.d s := by
  have hq' := Array.mem_toList_iff.2 hq
  have hmem : ∀ x ∈ pos q, ∃ t < ((c.body.nodes[k]!).args[i]!).seq.count,
      x = refPos (((c.body.nodes[k]!).args[i]!).seq.get t) := by
    intro x hx
    have : x ∈ posAll ((c.arg k i).pieces lo hi) := List.mem_flatMap.2 ⟨q, hq', hx⟩
    rw [pieces_pos _ hA.1 lo hi] at this
    have := mem_sl this
    rw [hA.2] at this
    obtain ⟨t, ht, rfl⟩ := List.mem_map.1 this
    exact ⟨t, List.mem_range.1 ht, rfl⟩
  have hcnt : 0 < q.count := by
    unfold Desc.PS.pieces at hq'
    rw [Array.toList_flatMap, List.mem_flatMap] at hq'
    obtain ⟨g, hg, hq'⟩ := hq'
    rw [Array.toList_flatMap, List.mem_flatMap] at hq'
    obtain ⟨m, -, hq'⟩ := hq'
    exact shiftGates_count _ _ (psSlice_ok _ hA.1 hP lo hi g hg).2.2 q hq'
  exact hD q segs hcnt
    (fun x hx => by obtain ⟨t, -, rfl⟩ := hmem x hx; simp [refPos])
    (fun x hx k' hk' => by
      obtain ⟨t, ht, rfl⟩ := hmem x hx
      simp only [refPos, Tgt.ref.injEq] at hk'
      obtain ⟨hp', rfl⟩ := hk'
      have := ((hR t ht).2 hp').2
      simp only [refPos]
      exact_mod_cast this)
    (fun x hx i' hi' => by
      obtain ⟨t, ht, rfl⟩ := hmem x hx
      simp only [refPos, Tgt.ref.injEq] at hi'
      obtain ⟨hp', rfl⟩ := hi'
      have := (hR t ht).1 hp'
      simp only [refPos]
      exact ⟨this.1, by exact_mod_cast this.2⟩)
    (fun w p hx => by
      obtain ⟨t, -, he⟩ := hmem _ hx
      simp [refPos] at he) h

/-- The root's callees' summaries name their gates with their widths (`memoOf_retW`). -/
theorem root_retW (P : Prog) (memo : Array Desc.Summary) (hm : memoOf P = .ok memo) (hF : RootFacts P)
    (hWF : WidthFacts P) : ∀ k node, (rootCtx P memo).node k = .ok node → RetW P memo node.fn := by
  obtain ⟨hr, hp, -, hDec, hRP, hSO, hcf, hnr, hB⟩ := hF
  intro k node hn
  have hk := node_lt hn
  have hg := node_get hn
  change k < (P.defs[P.root]!).body.nodes.size at hk
  change (P.defs[P.root]!).body.nodes[k]! = node at hg
  have hlt := hcf P.root hr k hk
  rw [← hg]
  exact (memoOf_retW P memo hm hRP hDec hWF hcf hnr hB hSO _ (by omega)
    (reached_closed P hcf P.root (reached_root P hr) hp k hk) (by omega)).2

/-- **Every segment `down` gives a piece of a root node's argument, one `Desc.memberRows` gives, or one
`Desc.carriesRead` gives, names the root's gates with their widths** (`piece_w`, `throughPS_w`, `carriesRead_of`). -/
theorem rootW (P : Prog) (memo : Array Desc.Summary) (hm : memoOf P = .ok memo) (hF : RootFacts P)
    (hWF : WidthFacts P) {k : Nat} {node : Flock.Program.Node} (hn : (rootCtx P memo).node k = .ok node) {s : Seg}
    (hs : (∃ i, FromDown (rootCtx P memo) k i s) ∨ FromRows (rootCtx P memo) k node s ∨
      FromCarry (rootCtx P memo) k node s) : GateW P P.root s := by
  have hR := root_retOk P memo hm hF
  have hRW := root_retW P memo hm hF hWF
  obtain ⟨hr, hp, -, hDec, -, -, -, -, hB⟩ := hF
  have hBr := hB P.root hr (reached_root P hr)
  rw [rootCtx_eq] at hn hs hR hRW
  have hW := widths_decoded P memo hr hp hDec hWF hBr hR hRW Flock.Steps.FUEL
  have hA := argOk_decoded P memo hr hDec
  have hPS : ∀ k node i p bp segs, (ctxOf P memo P.root).node k = .ok node →
      Desc.throughPS ((ctxOf P memo P.root).arg k i) p = some bp →
      Desc.down (ctxOf P memo P.root) Flock.Steps.FUEL bp = .ok segs → ∀ s ∈ segs.toList, GateW P P.root s := by
    intro k node i p bp segs hn hp hd
    have hk := node_lt hn
    by_cases hi' : i < ((ctxOf P memo P.root).body.nodes[k]!).args.size
    · exact throughPS_w _ _ hW.1 (hA k i hk hi') (ctxOf_argCount P memo P.root k i hk hi') ((hBr.1 k hk).2.1 i)
        hp hd
    · rw [throughPS_empty (ctxOf_arg_ge P memo P.root k i hi')] at hp
      cases hp
  have hk := node_lt hn
  rcases hs with ⟨i, lo, hi, q, hq, segs, hd, hs⟩ | ⟨i, u, ps, hmr, p, hp, segs, hd, hs⟩ | ⟨hfs, hn2, segs, hc, hs⟩
  · by_cases hi' : i < ((ctxOf P memo P.root).body.nodes[k]!).args.size
    · exact piece_w _ _ hW.1 (hA k i hk hi') (ctxOf_argCount P memo P.root k i hk hi') ((hBr.1 k hk).2.1 i) hq hd
        s (Array.mem_toList_iff.2 hs)
    · rw [pieces_empty (ctxOf_arg_ge P memo P.root k i hi')] at hq
      simp at hq
  · obtain ⟨p', hp'⟩ := memberRows_mem _ node k i u hmr p (Array.mem_toList_iff.2 hp)
    exact hPS k node i p' p segs hn hp' hd s (Array.mem_toList_iff.2 hs)
  · refine carriesRead_of (GateW P P.root) _ hR ?_ ?_ hPS k node segs hn hfs hn2 hc s (Array.mem_toList_iff.2 hs)
    · intro k' node' hn' hf' m a b segs' hm' _ h'
      refine hW.2.1 k' m a b segs' hm' ?_ h'
      have hk' := node_lt hn'
      have hg : (P.defs[P.root]!).body.nodes[k']! = node' := node_get hn'
      exact (((hWF P.root hr).2 hp).2.2 k' hk').2 (Or.inr (by rw [hg]; exact hf'))
    · intro k' node' hn' hm0 lo hi' segs' hlv h'
      exact init_w _ hn' hW.2.2.1 hm0 hlv h'

/-- **The gates a root node's reads name have their widths in `circuitOf`'s circuit** (`SegNodes`' first part):
`rootW`, each gate in the root's range (`locW_lt`), and `w_of_locW`. -/
theorem rootW_circuit (P : Prog) (memo : Array Desc.Summary) (c : Circuit) (hm : memoOf P = .ok memo)
    (hc : circuitOf P = .ok c) (hF : RootFacts P) (hWF : WidthFacts P) {k : Nat} {node : Flock.Program.Node}
    (hn : (rootCtx P memo).node k = .ok node) {s : Seg}
    (hs : (∃ i, FromDown (rootCtx P memo) k i s) ∨ FromRows (rootCtx P memo) k node s ∨
      FromCarry (rootCtx P memo) k node s) : ∀ w p, (Tgt.gate w, p) ∈ segPos s → w = c.w[p.toNat]! := by
  intro w p hp
  have hL := rootW P memo hm hF hWF hn hs w p hp
  obtain ⟨hLd, hlast, -⟩ := (hWF P.root hF.1).2 hF.2.1
  exact w_of_locW hc (locW_lt hL hLd hlast) hL

/-- **`SegNodes`' node parts.** With `RootNodes`' premises, of every segment `down` gives a piece of one of root
node `k`'s arguments (or one `Desc.memberRows` gives), or `Desc.carriesRead` gives of its scan's carries: each piece
of positive count names gates of one root node, and, when its copies shift, all its copies' gates are one root
node's. Proved in `ReadsOneNode` (`rootOneNode_of`). -/
def RootOneNode : Prop :=
  ∀ (bytes : ByteArray) (P : Prog) (memo : Array Desc.Summary) (c : Circuit),
    Flock.Program.decode bytes = .ok P → P.orderViolation = none → Desc.rootLaidOut P = true →
    Desc.inputsAtRoot P = true → Desc.walkable P = true → memoOf P = .ok memo → circuitOf P = .ok c →
    ∀ k < nodeCount P, ∀ node, (rootCtx P memo).node k = .ok node →
      ∀ s, ((∃ i, FromDown (rootCtx P memo) k i s) ∨ FromRows (rootCtx P memo) k node s ∨
        FromCarry (rootCtx P memo) k node s) →
        (∀ q ∈ s.ps.toList, q.count ≠ 0 →
          ∃ k' < nodeCount P, ∀ w p, (Tgt.gate w, p) ∈ pos q → p.toNat ∈ nodeGates P k') ∧
        (1 < s.reps → s.shift ≠ 0 → s.count ≠ 0 →
          ∃ k' < nodeCount P, ∀ w p, (Tgt.gate w, p) ∈ segPos s → p.toNat ∈ nodeGates P k')

/-- **`RootNodes` from `RootOneNode`** and the decoder's facts (`RootFacts`, `WidthFacts`): the widths are
`rootW_circuit`'s. -/
theorem rootNodes_of
    (hF : ∀ bytes P, Flock.Program.decode bytes = .ok P → P.orderViolation = none → Desc.walkable P = true →
      RootFacts P ∧ WidthFacts P)
    (hO : RootOneNode) : RootNodes := by
  intro bytes P memo c hd ho hl hi hw hm hc k hk node hn s hs
  obtain ⟨hRF, hWF⟩ := hF bytes P hd ho hw
  exact ⟨rootW_circuit P memo c hm hc hRF hWF hn hs, hO bytes P memo c hd ho hl hi hw hm hc k hk node hn s hs⟩

end NciCheck.Properties.SecurityProofs.ReadsExact
