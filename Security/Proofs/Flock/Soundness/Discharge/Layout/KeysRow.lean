import Proofs.Flock.Soundness.Discharge.Layout.OutShape
import Proofs.Flock.Soundness.Discharge.Layout.KeyProg

set_option Elab.async false

/-!
# `KeysRow` from the parse

`keyProg_valPinned` takes `KeysRow`: every input a computed row reads has the zero or a row bit as its key. At the row
widths `16 · words` (`portRowW`), in an untyped circuit whose unit reads any number of input groups, it is what the
verifier checks, once every input a unit row reads is a port bit (`PortReads`) and no wire reaches the unit
(`CutsFromRows`; `keysRow_of_cuts`). A unit with only its leaf group has no wire into it (`cutsFromRows_of_single`), so
`keysRow_of_parse` is that case:

* [e2e-aliased]'s `srcKey_leaf`, restated with the key: a stacked leaf port bit's key is the zero or message bit `t % 8`
  of byte `2 (x − start q) + t / 8` of the port `q` holding leaf word `x`, for the port bit `t`;
* a cut port bit's key is the same at its one leaf cut's row word (`check_facts`' `wired`: with no wire into the unit,
  each cut port of each unit copy has exactly one leaf cut), or the zero past the cut word's 16 bits;
* `check_in`, `check_cuts`: `HmRow.check` has each leaf input word and each leaf cut's row word below `inWords`, and
  `check_facts` the leaf ports 16 bits;
* `starts_find`: `Typed.starts` holds the ports' first words, so the last port starting at or before a word below
  `inWords` holds it.

A unit whose cut input a wire feeds (another net's output, as every multi-group unit the vLLM lowerings emit has: an
RMSNorm unit's scalar from the tail's `lookup:rsq` stage, a GEMM coordinate's chained accumulator) has the key
`Src.wire` there, which no gate of `keyProg` computes: its template instance is the unit's rows stacked `upv` times. Such
a unit needs a key program over the whole template instance (the unit copies, the stages and the lookup slots, joined by
Δ's wires in an order the verifier checks acyclic), not this condition (`Inst.instKeysRow_of_parse`).

`PortReads` is a circuit condition. `Net.ofRows` refuses a row that reads an input column whose row is empty
(`checkOrder`), and has each port bit's row a self row (`inputsOk`), but an input column outside every port may also be a
self row, and then a row may read it: Δ copies nothing into it, so its bit is the prover's.
-/

namespace FlockSoundness.Discharge.Layout

open Flock FlockLevel3

/-! ## The ports' first words -/

theorem foldl_starts (l : List Port) (acc : Array ℕ) (s : ℕ) :
    (l.foldl (fun (a : Array ℕ × ℕ) p => (a.1.push a.2, a.2 + p.words)) (acc, s)).1.toList =
        acc.toList ++ (List.range l.length).map (fun q => s + ((l.take q).map (·.words)).sum) ∧
      (l.foldl (fun (a : Array ℕ × ℕ) p => (a.1.push a.2, a.2 + p.words)) (acc, s)).2 =
        s + (l.map (·.words)).sum := by
  induction l generalizing acc s with
  | nil => simp
  | cons p l ih =>
    simp only [List.foldl_cons]
    obtain ⟨h1, h2⟩ := ih (acc.push s) (s + p.words)
    refine ⟨?_, ?_⟩
    · rw [h1, List.length_cons, List.range_succ_eq_map, List.map_cons, List.map_map]
      simp [Function.comp_def, List.take_succ_cons, Nat.add_assoc]
    · rw [h2]
      simp [Nat.add_assoc]

theorem starts_toList (c : Circuit) :
    (Typed.starts c).toList = (List.range c.ports.size).map fun q => ((c.ports.toList.take q).map (·.words)).sum := by
  unfold Typed.starts
  rw [← Array.foldl_toList]
  rw [(foldl_starts c.ports.toList #[] 0).1]
  simp

theorem starts_size (c : Circuit) : (Typed.starts c).size = c.ports.size := by
  have := congrArg List.length (starts_toList c)
  simpa using this

theorem starts_get (c : Circuit) {q : ℕ} (hq : q < c.ports.size) :
    (Typed.starts c)[q]! = ((c.ports.toList.take q).map (·.words)).sum := by
  rw [getElem!_pos (Typed.starts c) q (by rw [starts_size]; exact hq), ← Array.getElem_toList]
  simp only [starts_toList, List.getElem_map, List.getElem_range]

theorem inWords_eq (c : Circuit) : c.inWords = (c.ports.toList.map (·.words)).sum := by
  unfold Circuit.inWords
  rw [← Array.foldl_toList]
  have h : ∀ (l : List Port) (a : ℕ), l.foldl (fun a p => a + p.words) a = a + (l.map (·.words)).sum := by
    intro l
    induction l with
    | nil => simp
    | cons p l ih => intro a; rw [List.foldl_cons, ih]; simp [Nat.add_assoc]
  rw [h]
  simp

theorem find_rev_range {n : ℕ} {p : ℕ → Bool} {q : ℕ} (h : (List.range n).reverse.find? p = some q) :
    q < n ∧ p q = true ∧ ∀ q', q < q' → q' < n → p q' = false := by
  induction n with
  | zero => simp at h
  | succ n ih =>
    rw [List.range_succ, List.reverse_append, List.reverse_singleton, List.singleton_append, List.find?_cons] at h
    cases hp : p n with
    | true =>
      rw [hp] at h
      cases h
      exact ⟨by omega, hp, fun q' h1 h2 => by omega⟩
    | false =>
      rw [hp] at h
      obtain ⟨h1, h2, h3⟩ := ih h
      refine ⟨by omega, h2, fun q' hq1 hq2 => ?_⟩
      by_cases e : q' = n
      · rw [e]; exact hp
      · exact h3 q' hq1 (by omega)

/-- **The port holding a word**: the last port starting at or before a word below `inWords` holds it. -/
theorem starts_find (c : Circuit) {x q : ℕ} (hx : x < c.inWords)
    (hq : ((List.range (Typed.starts c).size).reverse.find? (fun q => (Typed.starts c)[q]! ≤ x)).getD 0 = q) :
    ∃ hq : q < c.ports.size, (Typed.starts c)[q]! ≤ x ∧ x < (Typed.starts c)[q]! + c.ports[q].words := by
  have hpos : 0 < c.ports.size := by
    rw [inWords_eq] at hx
    rcases hl : c.ports.toList with _ | ⟨p, l⟩
    · rw [hl] at hx; simp at hx
    · have := congrArg List.length hl
      simp only [Array.length_toList, List.length_cons] at this
      omega
  rcases hf : (List.range (Typed.starts c).size).reverse.find? (fun q => (Typed.starts c)[q]! ≤ x) with _ | q'
  · rw [List.find?_eq_none] at hf
    refine absurd (hf 0 (by simp [starts_size, hpos])) ?_
    simp [starts_get c hpos]
  · rw [hf, Option.getD_some] at hq
    subst hq
    obtain ⟨hlt, hp, hmax⟩ := find_rev_range hf
    rw [starts_size] at hlt
    refine ⟨hlt, by simpa using hp, ?_⟩
    have htake : ((c.ports.toList.take (q' + 1)).map (·.words)).sum =
        ((c.ports.toList.take q').map (·.words)).sum + c.ports[q'].words := by
      rw [List.take_add_one, List.map_append, List.sum_append, List.getElem?_eq_getElem (by simpa using hlt)]
      simp
    by_cases hn : q' + 1 < c.ports.size
    · have := hmax (q' + 1) (by omega) (by rw [starts_size]; exact hn)
      simp only [decide_eq_false_iff_not, not_le] at this
      rw [starts_get c hn, htake] at this
      rw [starts_get c hlt]
      exact this
    · have he : q' + 1 = c.ports.toList.length := by simp; omega
      have hall : ((c.ports.toList.take (q' + 1)).map (·.words)).sum = c.inWords := by
        rw [he, List.take_length, inWords_eq]
      rw [starts_get c hlt, ← htake, hall]
      exact hx

/-! ## The leaf inputs' words -/

/-- **What `HmRow.check` checks of the leaf inputs' words**, in an untyped circuit: each is below `inWords`. -/
theorem check_in {c : Circuit} {tags : Tags} {ls : Lean.Json} (ht : c.typed = none)
    (h : HmRow.check c tags ls = .ok ()) :
    ∀ u < c.upv, ∀ x ∈ (c.leavesIn[u]!).toList, x < (c.inWords : Int) := by
  unfold HmRow.check at h
  obtain ⟨spans, -, h⟩ := except_bind_ok h
  obtain ⟨u, -, h⟩ := except_bind_ok h
  obtain ⟨-, h⟩ := ite_throw_cond h
  obtain ⟨-, h⟩ := ite_throw_cond h
  obtain ⟨-, h⟩ := ite_throw_cond h
  simp only [ht] at h
  obtain ⟨-, h⟩ := ite_throw_cond h
  obtain ⟨-, h⟩ := ite_throw_cond h
  replace h := ite_join h fun ha => (ite_throw_cond ha).2
  obtain ⟨u1, -, h⟩ := except_bind_ok h
  obtain ⟨u2, -, h⟩ := except_bind_ok h
  obtain ⟨wired, -, h⟩ := except_bind_ok h
  obtain ⟨wired2, -, h⟩ := except_bind_ok h
  obtain ⟨u3, -, h⟩ := except_bind_ok h
  obtain ⟨-, h⟩ := ite_throw_cond h
  obtain ⟨seen, hseen, -⟩ := except_bind_ok h
  have leaves := forIn_range_forall
    (P := fun u => ∀ x ∈ (c.leavesIn[u]!).toList, x < (c.inWords : Int))
    (fun u _ _ s t hs => by
      obtain ⟨hc, hs⟩ := ite_throw_cond hs
      obtain ⟨s', -, hs⟩ := except_bind_ok hs
      cases hs
      refine ⟨⟨_, rfl⟩, fun x hx => ?_⟩
      simp only [Bool.or_eq_true, not_or, Array.any_eq_true', not_exists, not_and, decide_eq_true_eq, not_le,
        not_lt] at hc
      exact (hc.1.2 x (Array.mem_def.2 hx)).2) hseen
  exact fun u hu => leaves u (Nat.zero_le _) hu

/-- **What `HmRow.check` checks of the leaf cuts' row words and the block**, in an untyped circuit: each leaf cut's row
word is below `inWords`, and a block has a template instance. -/
theorem check_cuts {c : Circuit} {tags : Tags} {ls : Lean.Json} (ht : c.typed = none)
    (h : HmRow.check c tags ls = .ok ()) : (∀ e ∈ c.leafCuts.toList, e.2.2 < c.inWords) ∧ 0 < c.g := by
  unfold HmRow.check at h
  obtain ⟨spans, -, h⟩ := except_bind_ok h
  obtain ⟨u, -, h⟩ := except_bind_ok h
  obtain ⟨-, h⟩ := ite_throw_cond h
  obtain ⟨-, h⟩ := ite_throw_cond h
  obtain ⟨-, h⟩ := ite_throw_cond h
  simp only [ht] at h
  obtain ⟨-, h⟩ := ite_throw_cond h
  obtain ⟨-, h⟩ := ite_throw_cond h
  replace h := ite_join h fun ha => (ite_throw_cond ha).2
  obtain ⟨u1, -, h⟩ := except_bind_ok h
  obtain ⟨u2, -, h⟩ := except_bind_ok h
  obtain ⟨wired, -, h⟩ := except_bind_ok h
  obtain ⟨wired2, hcuts, h⟩ := except_bind_ok h
  obtain ⟨u3, -, h⟩ := except_bind_ok h
  obtain ⟨-, h⟩ := ite_throw_cond h
  obtain ⟨seen, -, h⟩ := except_bind_ok h
  obtain ⟨-, h⟩ := ite_throw_cond h
  obtain ⟨-, h⟩ := ite_throw_cond h
  obtain ⟨-, h⟩ := ite_throw_cond h
  obtain ⟨hg, -⟩ := ite_throw_cond h
  simp only [beq_iff_eq] at hg
  refine ⟨forIn_array_forall (P := fun e : ℕ × ℕ × ℕ => e.2.2 < c.inWords) (fun e _ s t hs => ?_) hcuts,
    Nat.pos_of_ne_zero hg⟩
  obtain ⟨hc, hs⟩ := ite_throw_cond hs
  cases hs
  refine ⟨⟨_, rfl⟩, ?_⟩
  simp only [Bool.or_eq_true, decide_eq_true_eq, not_or, not_le, not_lt] at hc
  exact hc.2

/-! ## A stacked row's inputs -/

/-- **A stacked row's input** is copy `i / (nComp + 1)`'s input, one its row reads. -/
theorem stackRow_input {R : Rows} {k : ℕ} {f : Fin R.nComp → List (Fin R.w)} {i : Fin (k * (R.nComp + 1))}
    {x : Fin (R.stackW k)} (hx : x ∈ R.stackRow k f i) (hxn : x.val < k * R.nIn) :
    ∃ (j : Fin R.nComp) (y : Fin R.w), y ∈ f j ∧ y.val < R.nIn ∧ i.val / (R.nComp + 1) < k ∧
      x.val = i.val / (R.nComp + 1) * R.nIn + y.val := by
  unfold Rows.stackRow at hx
  split at hx
  · simp only [List.mem_singleton] at hx
    rw [hx] at hxn
    simp only at hxn
    omega
  · obtain ⟨y, hy, rfl⟩ := List.mem_map.1 hx
    unfold Rows.stackCol at hxn ⊢
    split
    · rename_i h
      exact ⟨_, y, hy, h, R.stack_copy_lt k i.isLt, rfl⟩
    · rename_i h
      rw [dite_eq_right h] at hxn
      split at hxn <;> simp only at hxn <;> omega

/-! ## `KeysRow` -/

variable (c : Flock.Circuit) (hU : Built c.unit)

/-- **The row widths**: port `q`'s `16 · words` bits. -/
def portRowW (q : Fin c.ports.size) : ℕ := 16 * c.ports[q].words

/-- **Every input a unit row reads is a port bit**: a circuit condition (an input column outside every port may be a self
row, which `Net.ofRows` lets a row read; Δ copies nothing into it). -/
def PortReads : Prop :=
  ∀ (j : Fin (Rows.ofNet hU).nComp) (x : Fin (Rows.ofNet hU).w), x ∈ (Rows.ofNet hU).a j ++ (Rows.ofNet hU).b j →
    x.val < (Rows.ofNet hU).nIn → x.val ∈ portBits c.unit.inGroups

variable {c hU}

theorem ofNet_nIn_eq : (Rows.ofNet hU).nIn = c.unit.inWords * WORD := by
  rw [Rows.ofNet_nIn]
  obtain ⟨name, useful, constPos, ig, og, ra, rb, h⟩ := hU
  have hk := Net.ofRows_ok h
  rw [hk.2.2.2.2.1]
  exact Nat.min_eq_left hk.2.2.2.2.2.2.2.1

variable (c) in
/-- **No wire reaches the unit**: its cut inputs, in every input group past the leaf group, are all leaf cuts, each a
word of the rows. A circuit condition. -/
def CutsFromRows : Prop := ∀ w ∈ c.wires.toList, w.dst.1 ≠ c.unitNet

/-- **A unit with only its leaf group has no wire into it**, in a circuit `HmRow.parse` accepted with nets `pre` first:
`HmRow.check` refuses a wire into a leaf port, and that group's ports are all the unit's. -/
theorem cutsFromRows_of_singlePre {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    {pre : Array (String × Net)} (h : HmRow.parse tags bytes tables (pre := pre) = .ok c)
    (h1 : c.unit.inGroups.size = 1) : CutsFromRows c := by
  have P := (parse_factsPre_nets h).1
  obtain ⟨ls, hls⟩ := P.check
  have C := check_facts P.typed hls
  intro w hw hd
  have h2 := (C.wires w hw).2.2
  have h3 := C.wire_leaf w hw hd
  rw [hd] at h2
  change w.dst.2.2 < (portCols c.unit.inGroups).size at h2
  obtain ⟨g, hg⟩ := List.length_eq_one_iff.1 (show c.unit.inGroups.toList.length = 1 by simpa using h1)
  have hgs : c.unit.inGroups = #[g] := Array.toList_inj.1 (by simpa using hg)
  have hsz : (portCols #[g]).size = g.bits.size := by
    simpa using congrArg List.length (portCols_single g)
  have hg0 : c.unit.inGroups.getD 0 default = g := by rw [hgs]; rfl
  rw [hg0] at h3
  rw [hgs, hsz] at h2
  omega

/-- **A unit with only its leaf group has no wire into it** (`cutsFromRows_of_singlePre` with no net ahead of the
file). -/
theorem cutsFromRows_of_single {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    (h : HmRow.parse tags bytes tables = .ok c) (h1 : c.unit.inGroups.size = 1) : CutsFromRows c :=
  cutsFromRows_of_singlePre h h1

/-- **Every port-bit input of a unit slot has a row key**, in an untyped circuit `HmRow.parse` accepted with nets `pre`
first, with no wire into its unit (`CutsFromRows`), at the row widths `16 · words`. -/
theorem rowKey_of_cutsPre {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    {pre : Array (String × Net)} (h : HmRow.parse tags bytes tables (pre := pre) = .ok c) (hcr : CutsFromRows c)
    (hnd : (portBits c.unit.inGroups).Nodup) {x : ℕ} (hm : x % (c.unit.inWords * WORD) ∈ portBits c.unit.inGroups)
    (hu : x / (c.unit.inWords * WORD) < c.upv) : RowKey c (portRowW c) x := by
  obtain ⟨P, -, hunit, hsn, hhn⟩ := parse_factsPre_nets h
  obtain ⟨ls, hls⟩ := P.check
  have C := check_facts P.typed hls
  have hin := check_in P.typed hls
  obtain ⟨hcin, hg⟩ := check_cuts P.typed hls
  have hname : (c.nets.getD c.unitNet default).1 = "unit" := by
    obtain ⟨hlt, hp', -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hunit
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hlt, Option.getD_some]
    simpa using hp'
  have hsha : c.unitNet ≠ c.shaNet := fun he => by
    rw [← he, hname] at hsn
    exact absurd hsn (by unfold HmRow.SHA_NET; decide)
  have hhm : c.unitNet ≠ c.hmNet := fun he => by
    rw [← he, hname] at hhn
    exact absurd hhn (by unfold HmRow.HM_NET; decide)
  obtain ⟨p, t, hp, ht, hpt⟩ := Aliased.mem_portBits hm
  have hkey : Aliased.srcKey c x = Aliased.srcAt c (x / (c.unit.inWords * WORD)) p t := by
    unfold Aliased.srcKey
    rw [ite_eq_left hm, hpt, portOf_eq hnd hp ht, Nat.add_sub_cancel_left]
  unfold RowKey
  rw [hkey]
  generalize x / (c.unit.inWords * WORD) = u at hu ⊢
  unfold Aliased.srcAt
  by_cases hp' : p < (c.unit.inGroups.getD 0 default).bits.size
  swap
  · rw [ite_eq_right hp']
    have hpv : c.perVu.getD c.unitNet 0 = c.upv := by
      obtain ⟨hur, hk, hcount⟩ := unit_range_facts_of P.unitNet C
      exact perVu_unit C hur hk hcount hg
    have hone := C.wired c.unitNet P.unitNet hsha hhm u (by rw [hpv]; exact hu) p
      (by rw [ite_eq_left rfl]; omega) hp
    have hw0 : (c.wires.toList.filter fun w => decide (w.dst = (c.unitNet, u, p))) = [] :=
      List.filter_eq_nil_iff.2 fun w hw => by
        simp only [decide_eq_true_eq]
        intro hd
        exact hcr w hw (by rw [hd])
    have hcf : (c.leafCuts.toList.filter fun e => decide ((c.unitNet, e.1, e.2.1) = (c.unitNet, u, p))) =
        c.leafCuts.toList.filter fun e => decide (e.1 = u ∧ e.2.1 = p) :=
      List.filter_congr fun e _ => by simp
    rw [hw0, List.length_nil, Nat.zero_add, hcf] at hone
    obtain ⟨e, he⟩ := List.length_eq_one_iff.1 hone
    have hem : e ∈ c.leafCuts.toList :=
      List.mem_of_mem_filter (p := fun e => decide (e.1 = u ∧ e.2.1 = p)) (by rw [he]; exact List.mem_singleton_self e)
    simp only [he]
    split
    · rename_i hcond
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hcond
      obtain ⟨ht16, -⟩ := hcond
      refine Or.inr ⟨_, _, _, rfl, ?_⟩
      obtain ⟨hq, hs1, hs2⟩ := starts_find c (hcin e hem) rfl
      refine ⟨hq, ?_, Nat.mod_lt _ (by decide)⟩
      unfold portRowW
      simp only [Fin.getElem_fin]
      omega
    · exact Or.inl rfl
  have ht16 : t < 16 := by
    obtain ⟨-, -, hw⟩ := firstGroup_port C.unit_groups.1 hp'
    rw [hw, C.unit_groups.2 _ (getElem_mem_toList' hp')] at ht
    exact ht
  rw [ite_eq_left hp']
  split
  · exact Or.inl rfl
  · rename_i hneg
    dsimp only
    split
    swap
    · exact Or.inl rfl
    refine Or.inr ⟨_, _, _, rfl, ?_⟩
    have hget : c.leavesIn.getD u #[] = c.leavesIn[u]! := by
      rw [Array.getD_eq_getD_getElem?, getElem!_pos c.leavesIn u (by rw [C.leaves.1]; exact hu),
        Array.getElem?_eq_getElem (by rw [C.leaves.1]; exact hu), Option.getD_some]
    rw [hget] at hneg ⊢
    have hpl : p < (c.leavesIn[u]!).size := by rw [C.leaves.2 u hu]; exact hp'
    have hlt := hin u hu ((c.leavesIn[u]!)[p]!) (by
      rw [getElem!_pos (c.leavesIn[u]!) p hpl]; exact Array.getElem_mem_toList _)
    have hxl : ((c.leavesIn[u]!)[p]!).toNat < c.inWords := by omega
    obtain ⟨hq, hs1, hs2⟩ := starts_find c hxl rfl
    refine ⟨hq, ?_, Nat.mod_lt _ (by decide)⟩
    unfold portRowW
    simp only [Fin.getElem_fin]
    omega

/-- **`KeysRow` from each port-bit input's row key**: every input a computed row reads is, when every input a unit row
reads is a port bit (`PortReads`), a port bit of a unit slot. -/
theorem keysRow_of_rowKey (hpr : PortReads c hU)
    (hk : ∀ x, x % (c.unit.inWords * WORD) ∈ portBits c.unit.inGroups → x / (c.unit.inWords * WORD) < c.upv →
      RowKey c (portRowW c) x) : KeysRow c hU (portRowW c) := by
  intro i x hx hxn
  set R := Rows.ofNet hU with hR
  have hxn' : x.val < c.upv * R.nIn := hxn
  have hx' : x ∈ R.stackRow c.upv R.a i ++ R.stackRow c.upv R.b i := hx
  obtain ⟨j, y, hy, hyn, hu, hxy⟩ : ∃ (j : Fin R.nComp) (y : Fin R.w), y ∈ R.a j ++ R.b j ∧ y.val < R.nIn ∧
      i.val / (R.nComp + 1) < c.upv ∧ x.val = i.val / (R.nComp + 1) * R.nIn + y.val := by
    rcases List.mem_append.1 hx' with hx' | hx'
    · obtain ⟨j, y, hy, h2, h3, h4⟩ := stackRow_input hx' hxn'
      exact ⟨j, y, List.mem_append_left _ hy, h2, h3, h4⟩
    · obtain ⟨j, y, hy, h2, h3, h4⟩ := stackRow_input hx' hxn'
      exact ⟨j, y, List.mem_append_right _ hy, h2, h3, h4⟩
  have hm := hpr j y hy hyn
  set u := i.val / (R.nComp + 1)
  have hW : R.nIn = c.unit.inWords * WORD := ofNet_nIn_eq
  rw [hW] at hyn hxy
  have hmod : x.val % (c.unit.inWords * WORD) = y.val := by
    rw [hxy, Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hyn]
  have hdiv : x.val / (c.unit.inWords * WORD) = u := by
    rw [hxy, Nat.mul_comm, Nat.mul_add_div (by omega), Nat.div_eq_of_lt hyn, Nat.add_zero]
  exact hk x.val (by rw [hmod]; exact hm) (by rw [hdiv]; exact hu)

/-- **`KeysRow` from the parse, for a unit reading any number of input groups**: in an untyped circuit `HmRow.parse`
accepted with nets `pre` first, with no wire into its unit (`CutsFromRows`), at the row widths `16 · words`, every
input a computed row reads has the zero or a row bit as its key, when every input a unit row reads is a port bit
(`PortReads`). -/
theorem keysRow_of_cutsPre {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    {pre : Array (String × Net)} (h : HmRow.parse tags bytes tables (pre := pre) = .ok c) (hcr : CutsFromRows c)
    (hpr : PortReads c hU) : KeysRow c hU (portRowW c) :=
  keysRow_of_rowKey hpr fun _ hm hu => rowKey_of_cutsPre h hcr (Aliased.portBits_nodup hU) hm hu

/-- **`KeysRow` from the parse, for a unit reading any number of input groups** (`keysRow_of_cutsPre` with no net ahead
of the file). -/
theorem keysRow_of_cuts {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    (h : HmRow.parse tags bytes tables = .ok c) (hcr : CutsFromRows c) (hpr : PortReads c hU) :
    KeysRow c hU (portRowW c) :=
  keysRow_of_cutsPre h hcr hpr

/-- **`KeysRow` from the parse, with nets `pre` first** (a unit given ahead of the file): in an untyped circuit whose
unit has only its leaf group, at the row widths `16 · words`, every input a computed row reads has the zero or a row bit
as its key, when every input a unit row reads is a port bit (`PortReads`). -/
theorem keysRow_of_parsePre {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    {pre : Array (String × Net)} (h : HmRow.parse tags bytes tables (pre := pre) = .ok c)
    (h1 : c.unit.inGroups.size = 1) (hpr : PortReads c hU) : KeysRow c hU (portRowW c) :=
  keysRow_of_cutsPre h (cutsFromRows_of_singlePre h h1) hpr

/-- **`KeysRow` from the parse**: in an untyped circuit whose unit has only its leaf group, at the row widths
`16 · words`, every input a computed row reads has the zero or a row bit as its key, when every input a unit row reads is
a port bit (`PortReads`). -/
theorem keysRow_of_parse {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    (h : HmRow.parse tags bytes tables = .ok c) (h1 : c.unit.inGroups.size = 1) (hpr : PortReads c hU) :
    KeysRow c hU (portRowW c) :=
  keysRow_of_parsePre h h1 hpr

end FlockSoundness.Discharge.Layout
