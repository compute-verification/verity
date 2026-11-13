module

public import Proofs.Accounting.ReadUnit.Accepts.Cost
public import Proofs.Accounting.ReadUnit.Frame.Parts

/-!
# How many columns each of the read's parts has

Each part's program adds at most a fixed number of variables and returns at most a fixed number of bits
(`cost_segPart`, …, `cost_selectPart`), so its Definition's columns are at most `partCap H` (`partDef_cols_le`).
-/

@[expose] public section

namespace Accounting.ReadUnit.Accepts

open Flock.HmNets hiding and
open Flock.ReadNet

/-! ## Words -/

theorem length_padTail_le (total : ℕ) : (padTail total).length ≤ 144 := by
  simp only [padTail, List.length_append, List.length_cons, List.length_nil, List.length_replicate,
    List.length_map, List.length_range']
  omega

theorem length_wordsAt (k m : ℕ) : (wordsAt k m).length = m := by simp [wordsAt]

theorem w64_wordsAt (k m : ℕ) : W64 (wordsAt k m) := by
  intro x hx
  simp only [wordsAt, List.mem_map] at hx
  obtain ⟨j, -, rfl⟩ := hx
  simp [bitsAt]

theorem w8_wordsAt (k : ℕ) : W8 (wordsAt k 8) := ⟨(length_wordsAt k 8).le, w64_wordsAt k 8⟩

theorem w64_constWords (l : List UInt8) : W64 (constWords l) := by
  intro x hx
  simp only [constWords, List.mem_map] at hx
  obtain ⟨i, -, rfl⟩ := hx
  simp [constW]

theorem length_constWords_le (l : List UInt8) : (constWords l).length ≤ l.length / 8 := by simp [constWords]

theorem w8_midstate (pre : ByteArray) : W8 (midstate pre) := by
  refine ⟨by simp [midstate], ?_⟩
  intro x hx
  simp only [midstate, List.mem_map] at hx
  obtain ⟨i, -, rfl⟩ := hx
  simp [constW]

theorem w64_msgWords (fs : List Lin) : W64 (msgWords fs) := by
  intro x hx
  simp only [msgWords, List.mem_map] at hx
  obtain ⟨i, -, rfl⟩ := hx
  simp

theorem w64_append {a b : List Word} (ha : W64 a) (hb : W64 b) : W64 (a ++ b) :=
  fun x hx => (List.mem_append.1 hx).elim (ha x) (hb x)

theorem w8_ivW : W8 Frame.ivW :=
  ⟨FrameRead.length_ivW.le, fun x hx => (FrameRead.words64_ivW x hx).le⟩

theorem cost_and_pure (a b : Lin) (g : Lin → Lin) :
    Cost 1 (do let r ← Flock.HmNets.and a b; pure (g r) : M Lin) fun _ => True :=
  Cost.bind' (cost_and a b) fun _ _ => Cost.pure' 0 trivial

theorem cost_flat {m : M (List Word)} {k : ℕ} (h : Cost k m W8) :
    Cost k (do let o ← m; pure o.flatten : M (List Lin)) fun o => o.length ≤ 512 :=
  Cost.mono (Cost.bind' h fun o ho => Cost.pure' 0 (Q := fun o : List Lin => o.length ≤ 512) (flatten_le_of_W8 ho))
    (by omega) fun _ h => h

/-! ## The row segment, its outer hash and the leaf -/

theorem cost_segment (m : ℕ) (cv : List Word) (bits : List Lin) (hcv : W8 cv) (hb : bits.length = m) :
    Cost ((m + 1159) / 1024 * COMPRESS) (Flock.ReadNet.segment m cv bits) W8 := by
  unfold Flock.ReadNet.segment
  dsimp only
  refine Cost.mono (cost_shaW _ cv _ hcv (w64_msgWords _)) (Nat.mul_le_mul_right _ ?_) fun _ h => h
  have hp := length_padTail_le (128 + (m + 7) / 8)
  simp only [msgWords, List.length_map, List.length_range, List.length_append, List.length_replicate, constBits, hb]
  omega

theorem cost_outer (l : ℕ) (d : List Word) (hd : W64 d) :
    Cost ((d.length + 18) / 16 * COMPRESS) (outer l d) W8 := by
  unfold outer
  dsimp only
  refine Cost.mono (cost_shaW _ _ _ (w8_midstate _) (w64_append hd (w64_constWords _)))
    (Nat.mul_le_mul_right _ ?_) fun _ h => h
  have hp := length_padTail_le (128 + 8 * d.length)
  have hc := length_constWords_le (padTail (128 + 8 * d.length))
  rw [List.length_append]
  omega

theorem cost_leaf (x y : List Word) (hy : W64 y) : Cost (512 + (2 * COMPRESS + (2 * COMPRESS + 0))) (leaf x y)
    fun r => r.length ≤ 24 ∧ W64 r := by
  unfold leaf
  dsimp only
  refine Cost.bind' (Cost.mono (Cost.mapM' _ (P := fun _ => True) (k := 64)
      (fun _ _ => Cost.mono (cost_commitW _) (by simp) (Q := fun w : Word => w.length ≤ 64)
        fun r hr => (by rw [hr]; simp : r.length ≤ 64)) _ fun _ _ => trivial)
    (by simp) (Q := fun b => b.length = 8 ∧ W64 b) fun r hr => ⟨by simpa using hr.1, hr.2⟩) fun b hb => ?_
  refine Cost.bind' (cost_shaW 2 _ _ (w8_midstate _) (w64_append hy (w64_constWords _))) fun c hc => ?_
  refine Cost.bind' (cost_shaW 2 _ _ (w8_midstate _) (w64_append (w64_append hb.2 hc.2) (w64_constWords _)))
    fun lf hlf => Cost.pure' 0 ⟨?_, w64_append (w64_append hb.2 hc.2) hlf.2⟩
  have h1 := hb.1
  have h2 := hc.1
  have h3 := hlf.1
  simp only [List.length_append]
  omega

/-! ## The frame's gadgets -/

theorem cost_orL (a b : Lin) : Cost 1 (Frame.orL a b) fun _ => True := cost_and_pure _ _ _

theorem length_ite_cons_le {c : Prop} [Decidable c] (a : Lin) (l : List Lin) :
    (if c then a :: l else l).length ≤ l.length + 1 := by
  split <;> simp

theorem cost_flagsGo (bits : List Lin) : ∀ (t : ℕ) (acc : Option Lin) (out : List Lin),
    Cost t (Frame.flagsGo bits t acc out) fun r => r.length ≤ out.length + t
  | 0, _, out => by
      unfold Frame.flagsGo
      exact Cost.pure' _ (by simp)
  | t + 1, none, out => by
      unfold Frame.flagsGo
      exact Cost.mono (Cost.bind' (Cost.pure' 0 (Q := fun _ => True) trivial) fun a _ =>
          Cost.mono (cost_flagsGo bits t (some a) _) le_rfl (Q := fun r => r.length ≤ out.length + 1 + t)
            fun r hr => (by have := length_ite_cons_le (c := (t + 8) % 8 = 0) a out; omega :
              r.length ≤ out.length + 1 + t))
        (by omega) fun r hr => by omega
  | t + 1, some a₀, out => by
      unfold Frame.flagsGo
      exact Cost.mono (Cost.bind' (cost_orL a₀ _) fun a _ =>
          Cost.mono (cost_flagsGo bits t (some a) _) le_rfl (Q := fun r => r.length ≤ out.length + 1 + t)
            fun r hr => (by have := length_ite_cons_le (c := (t + 8) % 8 = 0) a out; omega :
              r.length ≤ out.length + 1 + t))
        (by omega) fun r hr => by omega

theorem cost_flags (bits : List Lin) (n : ℕ) :
    Cost (8 * n - 8) (Frame.flags bits n) fun r => r.length ≤ 8 * n - 8 :=
  Cost.mono (cost_flagsGo bits _ none []) le_rfl fun r h => by simpa using h

theorem cost_selectGo : ∀ (m : Lin) (zs : List Lin) (ps : List (Lin × Lin)),
    Cost zs.length (Frame.selectGo m zs ps) fun _ => True
  | m, z :: zs, (a, b) :: ps => by
      unfold Frame.selectGo
      dsimp only
      split
      · exact Cost.mono (Cost.bind' (Cost.pure' 0 (Q := fun _ => True) trivial) fun m' _ => cost_selectGo m' zs ps)
          (by simp only [List.length_cons]; omega) fun _ h => h
      · exact Cost.mono (Cost.bind' (cost_and _ _) fun _ _ =>
            Cost.bind' (Cost.pure' 0 (Q := fun _ => True) trivial) fun m' _ => cost_selectGo m' zs ps)
          (by simp only [List.length_cons]; omega) fun _ h => h
  | m, [], _ => by
      unfold Frame.selectGo
      exact Cost.pure' _ trivial
  | m, _ :: _, [] => by
      unfold Frame.selectGo
      exact Cost.pure' _ trivial

theorem cost_selectBit (fl col : List Lin) : Cost fl.length (Frame.selectBit fl col) fun _ => True :=
  cost_selectGo _ fl _

theorem cost_frameSelect (fl : List Lin) (layouts : List (List Word)) :
    Cost ((layouts.headD []).length * (64 * fl.length)) (Frame.select fl layouts)
      fun r => r.length = (layouts.headD []).length ∧ W64 r :=
  Cost.mono (Cost.mapM' _ (P := fun _ => True) (k := 64 * fl.length)
      (fun _ _ => Cost.mono (Cost.mapM' _ (P := fun _ => True) (k := fl.length)
        (fun _ _ => cost_selectBit _ _) _ fun _ _ => trivial) (by simp)
        (Q := fun w : Word => w.length ≤ 64) fun r h => (by simp [h.1] : r.length ≤ 64)) _ fun _ _ => trivial)
    (by simp) fun r h => ⟨by simpa using h.1, h.2⟩

theorem headD_range_map {β : Type} (f : ℕ → List β) {n : ℕ} (hn : 1 ≤ n) :
    ((List.range n).map f).headD [] = f 0 := by
  obtain ⟨k, rfl⟩ : ∃ k, n = k + 1 := ⟨n - 1, by omega⟩
  simp [List.range_succ_eq_map]

theorem cost_hashUint (tag : List UInt8) (head : List (List Frame.Byte)) (x : List Lin) (n : ℕ)
    (tail : List Frame.Byte → List (List Frame.Byte)) {Lm : ℕ}
    (hL : (((List.range n).map fun L => Frame.message tag (head ++ tail (Frame.uintBytes x (L + 1)))).headD []).length
      ≤ Lm) :
    Cost ((8 * n - 8) + (Lm * (64 * (8 * n - 8)) + (Lm + 15) / 16 * COMPRESS)) (Frame.hashUint tag head x n tail)
      W8 := by
  unfold Frame.hashUint
  dsimp only
  refine Cost.bind' (cost_flags x n) fun fl hfl => ?_
  refine Cost.bind' (Cost.mono (cost_frameSelect fl _) (Nat.mul_le_mul hL (Nat.mul_le_mul_left 64 hfl))
    (Q := fun ws : List Word => ws.length ≤ Lm ∧ W64 ws) fun r hr => ⟨by rw [hr.1]; exact hL, hr.2⟩)
    fun ws hws => ?_
  exact Cost.mono (cost_shaW _ _ ws w8_ivW hws.2)
    (Nat.mul_le_mul_right _ (Nat.div_le_div_right (Nat.add_le_add_right hws.1 15))) fun _ h => h

theorem length_schema_seg : (Flock.HmRow.ROW_SCHEMA_SEG.toUTF8.data.toList).length = 22 := by
  show ("hm96-sha512/row-seg/v1".toUTF8.data.toList).length = 22
  rw [FrameRead.length_utf8]
  simp [Char.utf8Size]

theorem cost_leafHash (n : ℕ) (hn : 1 ≤ n) (domain value : List Word) (x : List Lin) (hd : domain.length = 8)
    (hv : value.length = 8) :
    Cost ((8 * n - 8) + (32 * (64 * (8 * n - 8)) + 2 * COMPRESS))
      (Frame.leafHash Flock.HmRow.ROW_SCHEMA_SEG n domain value x) W8 :=
  Cost.mono (cost_hashUint (Lm := 32) _ _ _ _ _ (by
    rw [headD_range_map _ hn]
    exact (FrameRead.length_message_leaf hd hv x (0 + 1) _ (by rw [length_schema_seg]; omega)).le))
    (by norm_num) fun _ h => h

theorem cost_nodeHash (n : ℕ) (hn : 1 ≤ n) (domain : List Word) (level : Frame.Byte) (index : List Lin) (d : Lin)
    (cur sib : List Word) (hd : domain.length = 8) (hc : cur.length = 8) (hs : sib.length = 8) (hcw : W64 cur) :
    Cost (512 + ((8 * n - 8) + (48 * (64 * (8 * n - 8)) + 3 * COMPRESS)))
      (Frame.nodeHash n domain level index d cur sib) W8 := by
  unfold Frame.nodeHash
  refine Cost.bind' (k₁ := 512) (P := fun t : List Word => t.length = 8) ?_ fun t ht => ?_
  · refine Cost.mono (Cost.mapM' _ (P := fun cs : Word × Word => cs.1.length ≤ 64) (Q := fun _ : Word => True)
      (k := 64) ?_ (cur.zip sib) ?_) ?_ ?_
    · intro cs hcs
      have h : cs.1.length ≤ 64 := hcs
      refine Cost.mono (Cost.mapM' _ (P := fun _ => True) (k := 1) (fun _ _ => cost_and _ _) _ fun _ _ => trivial)
        ?_ fun _ _ => trivial
      simp only [List.length_zip, Nat.mul_one]
      omega
    · intro cs hcs
      exact hcw _ (List.of_mem_zip (show (cs.1, cs.2) ∈ cur.zip sib from hcs)).1
    · simp [List.length_zip, hc, hs]
    · intro t ht
      show t.length = 8
      rw [ht.1]
      simp [hc, hs]
  have ht' : t.length = 8 := ht
  dsimp only
  exact Cost.mono (cost_hashUint (Lm := 48) _ _ _ _ _ (by
    rw [headD_range_map _ hn]
    exact (FrameRead.length_message_node hd (by simp [hc, ht']) (by simp [hs, ht']) level index (0 + 1)
      (by omega)).le)) (by norm_num) fun _ h => h

theorem cost_andPairs : ∀ l : List Lin, Cost (l.length / 2) (Frame.andPairs l) fun r => r.length = (l.length + 1) / 2
  | a :: b :: rest => by
      unfold Frame.andPairs
      exact Cost.mono (Cost.bind' (cost_and a b) fun r _ => Cost.bind' (cost_andPairs rest) fun rs hrs =>
          Cost.pure' 0 (a := r :: rs) (Q := fun x => x.length = ((a :: b :: rest).length + 1) / 2)
            (by simp only [List.length_cons]; omega))
        (by simp only [List.length_cons]; omega) fun _ h => h
  | [] => by
      unfold Frame.andPairs
      exact Cost.pure' _ (by simp)
  | [_] => by
      unfold Frame.andPairs
      exact Cost.pure' _ (by simp)

theorem cost_andTree : ∀ (k : ℕ) (l : List Lin), Cost l.length (Frame.andTree k l) fun _ => True
  | _, [] => by
      unfold Frame.andTree
      exact Cost.pure' _ trivial
  | _, [_] => by
      unfold Frame.andTree
      exact Cost.pure' _ trivial
  | 0, _ :: _ :: _ => by
      unfold Frame.andTree
      exact Cost.pure' _ trivial
  | k + 1, a :: b :: l => by
      unfold Frame.andTree
      exact Cost.mono (Cost.bind' (cost_andPairs _) fun l' hl' =>
          Cost.mono (cost_andTree k l') (k' := (l.length + 3) / 2)
            (by rw [hl']; simp only [List.length_cons]; omega) fun _ h => h)
        (by simp only [List.length_cons]; omega) fun _ h => h

theorem cost_eqDigest (x y : List Word) (hx : W8 x) : Cost 512 (Frame.eqDigest x y) fun _ => True := by
  unfold Frame.eqDigest Frame.andReduce
  refine Cost.mono (cost_andTree _ _) ?_ fun _ h => h
  rw [List.length_flatMap]
  refine (sum_map_le _ _ 64 fun ab hab => ?_).trans ?_
  · have := hx.2 _ (List.of_mem_zip (show (ab.1, ab.2) ∈ x.zip y from hab)).1
    simp only [List.length_map, List.length_zip]
    omega
  · have := hx.1
    simp only [List.length_zip]
    omega

/-! ## The select -/

theorem cost_belowGo (bound : ℕ) : ∀ (bs : List Lin) (k : ℕ) (lt : Lin),
    Cost bs.length (belowGo bound bs k lt) fun _ => True
  | [], _, _ => by
      unfold belowGo
      exact Cost.pure' _ trivial
  | _ :: bs, _, _ => by
      unfold belowGo
      dsimp only
      split
      · exact Cost.mono (Cost.bind' (cost_and _ _) fun _ _ =>
            Cost.bind' (Cost.pure' 0 (Q := fun _ => True) trivial) fun _ _ => cost_belowGo bound bs _ _)
          (by simp only [List.length_cons]; omega) fun _ h => h
      · exact Cost.mono (Cost.bind' (cost_and _ _) fun _ _ => cost_belowGo bound bs _ _)
          (by simp only [List.length_cons]; omega) fun _ h => h

theorem cost_selectRow (V : ℕ) (tok : List Lin) (opened : Lin) (row : List Lin) :
    Cost (tok.length + (row.length + (1 + 0))) (select V tok opened row) fun r => r.1.length ≤ row.length := by
  unfold select below
  exact Cost.bind' (cost_belowGo _ _ _ _) fun _ _ =>
    Cost.bind' (Cost.mono (Cost.mapM' _ (P := fun _ => True) (k := 1) (fun _ _ => cost_and_pure _ _ _) _
        fun _ _ => trivial) (by simp only [List.length_zip, Nat.mul_one]; omega)
        (Q := fun h : List Lin => h.length ≤ row.length)
        fun r hr => (by rw [hr.1]; simp only [List.length_zip]; omega : r.length ≤ row.length))
      fun h hh => Cost.bind' (cost_and _ _) fun r _ => Cost.pure' 0 (a := (h, Lin.one ^^^ r)) hh

/-! ## The parts -/

theorem cost_segPart (m : ℕ) (hm : m ≤ 8192) : Cost (9 * COMPRESS) (segPart m).prog fun o => o.length ≤ 512 :=
  cost_flat (Cost.mono (cost_segment m _ _ (w8_wordsAt 0) (by simp [bitsAt]))
    (Nat.mul_le_mul_right _ (by omega)) fun _ h => h)

theorem cost_outerPart (l s : ℕ) : Cost ((s + 2) * COMPRESS) (outerPart l s).prog fun o => o.length ≤ 512 :=
  cost_flat (Cost.mono (cost_outer l _ (w64_wordsAt 0 _))
    (Nat.mul_le_mul_right _ (by rw [length_wordsAt]; omega)) fun _ h => h)

theorem cost_leafPart : Cost (512 + 4 * COMPRESS) leafPart.prog fun o => o.length ≤ 1536 :=
  Cost.mono (Cost.bind' (cost_leaf _ _ (w64_wordsAt 512 24)) fun _ ho =>
      Cost.pure' 0 (Q := fun o : List Lin => o.length ≤ 1536) (length_flatten_le (n := 24) ho.1 ho.2))
    (by omega) fun _ h => h

theorem cost_leafHashPart (n : ℕ) (hn : 1 ≤ n) (hn4 : n ≤ 4) :
    Cost (3 * COMPRESS + 49176) (leafHashPart n).prog fun o => o.length ≤ 512 :=
  cost_flat (Cost.mono (cost_leafHash n hn _ _ _ (length_wordsAt 0 8) (length_wordsAt 512 8)) (by omega)
    fun _ h => h)

theorem cost_nodePart (n : ℕ) (hn : 1 ≤ n) (hn4 : n ≤ 4) :
    Cost (4 * COMPRESS + 74264) (nodePart n).prog fun o => o.length ≤ 512 :=
  cost_flat (Cost.mono (cost_nodeHash n hn _ _ _ _ _ _ (length_wordsAt 0 8) (length_wordsAt _ 8)
    (length_wordsAt _ 8) (w64_wordsAt _ 8)) (by omega) fun _ h => h)

theorem cost_eqPart : Cost 512 eqPart.prog fun o => o.length ≤ 1 :=
  Cost.mono (Cost.bind' (cost_eqDigest _ _ (w8_wordsAt 0)) fun b _ =>
      Cost.pure' 0 (a := [b]) (Q := fun o : List Lin => o.length ≤ 1) (by simp))
    (by omega) fun _ h => h

theorem cost_selectPart (V H : ℕ) : Cost (33 + 16 * H) (selectPart V H).prog fun o => o.length ≤ 16 * H + 1 :=
  Cost.mono (Cost.bind' (cost_selectRow V _ _ _) fun r hr =>
      Cost.pure' 0 (a := r.1 ++ [r.2]) (Q := fun o : List Lin => o.length ≤ 16 * H + 1) (by
        simp only [bitsAt, List.length_map, List.length_range] at hr
        simp only [List.length_append, List.length_cons, List.length_nil]
        omega))
    (by simp only [bitsAt, List.length_map, List.length_range]; omega) fun _ h => h

/-! ## Each part's columns -/

theorem cols_le {p : Part} {k L : ℕ} (h : Cost k p.prog fun o => o.length ≤ L) :
    (defOf p.ins p.run).cols ≤ k + L := by
  obtain ⟨h1, h2⟩ := h ⟨p.ins, []⟩
  have h1' : p.run.2.next ≤ p.ins + k := h1
  have h2' : p.run.1.length ≤ L := h2
  show p.run.2.next - p.ins + p.run.1.length ≤ k + L
  omega

/-- The columns of each part's Definition are at most this. -/
def partCap (H : ℕ) : ℕ := (segs H + 9) * COMPRESS + 32 * H + 2048

theorem depth_le {V : ℕ} (hV : V < 2 ^ 32) : depth V ≤ 32 := by
  unfold depth
  split
  · omega
  · have := (Nat.log2_lt (n := V - 1) (k := 32) (by omega)).2 (lt_of_le_of_lt (Nat.sub_le V 1) hV)
    omega

theorem partDef_cols_le {V H : ℕ} (hV : V < 2 ^ 32) :
    ∀ p ∈ parts V H, (defOf p.ins p.run).cols ≤ partCap H := by
  have hC : COMPRESS = 121856 := rfl
  have hr : Flock.HmRow.ROW_SEG_BITS = 8192 := rfl
  have hD := depth_le hV
  have hs : 16 * H ≤ 8192 * segs H := by unfold segs; rw [hr]; omega
  have hn1 : 1 ≤ Frame.indexBytes (depth V) := by unfold Frame.indexBytes; omega
  have hn4 : Frame.indexBytes (depth V) ≤ 4 := by unfold Frame.indexBytes; omega
  have cap : ∀ {k L : ℕ}, k + L ≤ partCap H → ∀ {p : Part}, Cost k p.prog (fun o => o.length ≤ L) →
      (defOf p.ins p.run).cols ≤ partCap H := fun hkL _ h => (cols_le h).trans hkL
  intro p hp
  simp only [parts, List.mem_toArray, List.mem_cons, List.not_mem_nil, or_false] at hp
  rcases hp with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact cap (by unfold partCap; rw [Nat.add_mul, hC]; omega) (cost_segPart _ (le_of_eq hr))
  · exact cap (by unfold partCap; rw [Nat.add_mul, hC]; omega) (cost_segPart _ (by rw [hr]; omega))
  · exact cap (by unfold partCap; rw [Nat.add_mul, Nat.add_mul, hC]; omega) (cost_outerPart _ _)
  · exact cap (by unfold partCap; rw [Nat.add_mul, hC]; omega) cost_leafPart
  · exact cap (by unfold partCap; rw [Nat.add_mul, hC]; omega) (cost_leafHashPart _ hn1 hn4)
  · exact cap (by unfold partCap; rw [Nat.add_mul, hC]; omega) (cost_nodePart 1 le_rfl (by omega))
  · exact cap (by unfold partCap; rw [Nat.add_mul, hC]; omega) (cost_nodePart 2 (by omega) (by omega))
  · exact cap (by unfold partCap; rw [Nat.add_mul, hC]; omega) (cost_nodePart 3 (by omega) (by omega))
  · exact cap (by unfold partCap; rw [Nat.add_mul, hC]; omega) (cost_nodePart 4 (by omega) le_rfl)
  · exact cap (by unfold partCap; rw [Nat.add_mul, hC]; omega) cost_eqPart
  · exact cap (by unfold partCap; rw [Nat.add_mul, hC]; omega) (cost_selectPart V H)

end Accounting.ReadUnit.Accepts
