module

public import Proofs.Flock.SpotFlock.NeededBitsOf
public import Proofs.Flock.SpotFlock.WireMapRewires
public import Definitions.Certifier.SampledAudit
import Mathlib.Logic.Relation

/-!
# Where the members' programs and proof blocks sit, with no hypothesis on the members

`Properties.Spot.bitsProgs ms` puts the members' programs side by side, member `m` from `Spot.Bits.base ms m`. The
lemmas here take no `htop` (each member's program topologically ordered), which the audit's statement does not give; each
holds of the key program's own layout:

* `bitsProgs_gate_base`: member `m`'s gate `p` is `Bits.base ms m + p`, its reads shifted (`NeededBitsOf.gates_bitsProgs`),
  and `bitsProgs_base_const` is its gate 0, the constant.
* `bitsProgs_outBit`, `bitsProgs_inBit`: the position of a returned output's bit is a returned AND gate; the position of
  an input bit some proof block reads is an input gate.
* `owner_inGate_of_unitReads`: the source gate on an input bit the unit net reads is owned by its unit.
* `membersPart_reads_own`: a proof block's gates read only its own gates and constants
  (`NeededBitsOf.blocksClosed_bitsProgs`); `bitsProgs_depends_member`: a gate of member `m` depends on gates of member `m`
  alone.
-/

@[expose] public section

namespace Flock.Bridge.Members

open Claims.Sampling Properties.Spot

/-! ## The wrappers over `NeededBitsOf` -/

/-- **Member `m`'s gate `p` sits at `Bits.base ms m + p`**, its reads `Bits.base ms m` later. -/
theorem bitsProgs_gate_base (ms : Spot.Bits.Members) {m p : ℕ} (hm : m < ms.length)
    (hp : p < (memberProg ms[m].1 ms[m].2.size).gates.length) :
    (bitsProgs ms).gates[Spot.Bits.base ms m + p]? =
      (memberProg ms[m].1 ms[m].2.size).gates[p]?.map (shiftGate (Spot.Bits.base ms m)) :=
  NeededBitsOf.gates_bitsProgs ms hm hp

/-- **A proof block of the members reads its own gates or constants.** -/
theorem membersPart_reads_own (ms : Spot.Bits.Members) {v : List ℕ} (hv : v ∈ (membersPart ms).blocks)
    {g i : ℕ} (hg : g ∈ v) (hi : i ∈ ((bitsProgs ms).gates[g]?.map BGate.reads).getD []) :
    i ∈ v ∨ ∃ b, (bitsProgs ms).gates[i]? = some (.const b) := by
  obtain ⟨hc, -⟩ := NeededBitsOf.blocksClosed_bitsProgs ms
  cases hgate : (bitsProgs ms).gates[g]? with
  | none =>
    rw [hgate] at hi
    simp at hi
  | some x =>
    rw [hgate] at hi
    exact hc v hv g hg x hgate i (by simpa using hi)

/-! ## A member's gates read its own gates -/

theorem keyWire_lt_size (c : _root_.Flock.Circuit) (n : ℕ) {u : ℕ} (col : ℕ) (hu : u < n) :
    keyWire c n u col < (Spot.Bits.keyProg c n).size := by
  rw [keyProg_size]
  rcases NeededBitsOf.keyWire_cases c n u col with h | h | ⟨b', hb', h⟩ | ⟨j, hj, h⟩
  · rw [h]
    omega
  · rw [h]
    omega
  · rw [h]
    have := NeededBitsOf.src_lt hu (keyIdx_lt hb')
    omega
  · rw [h]
    have := NeededBitsOf.mul_add_lt hu hj
    omega

/-- **A row of the key program reads gates of the program.** -/
theorem row_args_lt_size (c : _root_.Flock.Circuit) (n : ℕ) {h : ℕ} {a b : List ℕ}
    (hr : (Spot.Bits.keyProg c n).row? h = some (a, b)) {w : ℕ} (hw : w ∈ a ++ b) :
    w < (Spot.Bits.keyProg c n).size := by
  have hs : h < (Spot.Bits.keyProg c n).size := lt_size_of_isSome_row? (by simp [hr])
  have hi : (Spot.Bits.keyProg c n).inputs ≤ h := by
    refine Nat.le_of_not_lt fun hlt => ?_
    have := NeededBitsOf.row?_of_lt hlt
    rw [hr] at this
    cases this
  rw [keyProg_inputs] at hi
  rw [keyProg_size] at hs
  obtain ⟨u, i, hu, hi', rfl⟩ := NeededBitsOf.split_units hi hs
  rw [keyProg_row?_row c n hu hi'] at hr
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hr)
  rcases List.mem_append.1 hw with hw | hw
  · obtain ⟨col, -, rfl⟩ := List.mem_map.1 hw
    exact keyWire_lt_size c n col hu
  · obtain ⟨col, -, rfl⟩ := List.mem_map.1 hw
    exact keyWire_lt_size c n col hu

/-- **A gate of a member's program reads gates of the program.** -/
theorem memberProg_reads_lt (c : _root_.Flock.Circuit) (n : ℕ) {p : ℕ} {x : BGate}
    (hx : (memberProg c n).gates[p]? = some x) {w : ℕ} (hw : w ∈ x.reads) :
    w < (memberProg c n).gates.length := by
  have hmem : x ∈ (memberProg c n).gates := List.mem_iff_getElem?.2 ⟨p, hx⟩
  rw [NeededBitsOf.memberProg_eq c n] at hmem ⊢
  rw [SecurityProofs.bitsProg_length]
  have hmem' : x ∈ ((List.range (Spot.Bits.keyProg c n).size).flatMap
      (Spot.Bits.lower (Spot.Bits.keyProg c n) (Spot.Bits.consts c n))).map bitsGate := hmem
  obtain ⟨L, hL, rfl⟩ := List.mem_map.1 hmem'
  obtain ⟨h, hh, hLh⟩ := List.mem_flatMap.1 hL
  have hhs := List.mem_range.1 hh
  rw [SecurityProofs.bitsGate_reads] at hw
  cases hr : (Spot.Bits.keyProg c n).row? h with
  | none =>
    cases hk : Spot.Bits.consts c n h with
    | none =>
      simp [Spot.Bits.lower, hr, hk] at hLh
      subst hLh
      simp [Spot.Bits.LGate.reads] at hw
    | some b =>
      simp [Spot.Bits.lower, hr, hk] at hLh
      subst hLh
      simp [Spot.Bits.LGate.reads] at hw
  | some ab =>
    obtain ⟨a, b⟩ := ab
    simp only [Spot.Bits.lower, hr, List.mem_cons, List.not_mem_nil, or_false] at hLh
    rcases hLh with rfl | rfl | rfl
    · simp only [Spot.Bits.LGate.reads] at hw
      obtain ⟨y, hy, rfl⟩ := List.mem_map.1 hw
      exact NeededBitsOf.posOf_lt_off (row_args_lt_size c n hr (List.mem_append_left _ hy))
    · simp only [Spot.Bits.LGate.reads] at hw
      obtain ⟨y, hy, rfl⟩ := List.mem_map.1 hw
      exact NeededBitsOf.posOf_lt_off (row_args_lt_size c n hr (List.mem_append_right _ hy))
    · simp only [Spot.Bits.LGate.reads, List.mem_cons, List.not_mem_nil, or_false] at hw
      have h3 := NeededBitsOf.gateSize_of_some hr
      rcases hw with rfl | rfl
      · have := NeededBitsOf.off_add_lt hhs (j := 0) (by omega)
        omega
      · have := NeededBitsOf.off_add_lt hhs (j := 1) (by omega)
        omega

/-- **A gate depends only on its own member's**: a gate of member `m` reads gates of member `m`. -/
theorem bitsProgs_depends_member (ms : Spot.Bits.Members) {m i g : ℕ} (hm : m < ms.length)
    (hg₁ : Spot.Bits.base ms m ≤ g) (hg₂ : g < Spot.Bits.base ms (m + 1))
    (hd : NetTiming.Depends (bitsProgs ms) i g) :
    Spot.Bits.base ms m ≤ i ∧ i < Spot.Bits.base ms (m + 1) := by
  have hb := base_succ ms hm
  have step : ∀ y, Spot.Bits.base ms m ≤ y → y < Spot.Bits.base ms (m + 1) →
      ∀ x ∈ ((bitsProgs ms).gates[y]?.map BGate.reads).getD [],
        Spot.Bits.base ms m ≤ x ∧ x < Spot.Bits.base ms (m + 1) := by
    intro y hy1 hy2 x hx
    obtain ⟨q, rfl⟩ : ∃ q, y = Spot.Bits.base ms m + q := ⟨y - Spot.Bits.base ms m, by omega⟩
    have hq : q < (memberProg ms[m].1 ms[m].2.size).gates.length := by omega
    rw [bitsProgs_gate_base ms hm hq] at hx
    cases hgq : (memberProg ms[m].1 ms[m].2.size).gates[q]? with
    | none =>
      rw [hgq] at hx
      simp at hx
    | some z =>
      rw [hgq] at hx
      simp only [Option.map_some, Option.getD_some, reads_shiftGate, List.mem_map] at hx
      obtain ⟨w, hw, rfl⟩ := hx
      have := memberProg_reads_lt _ _ hgq hw
      omega
  unfold NetTiming.Depends at hd
  induction hd using Relation.ReflTransGen.head_induction_on with
  | refl => exact ⟨hg₁, hg₂⟩
  | @head a c' h' h ih => exact step c' ih.1 ih.2 a h'

/-! ## The constant at a member's gate 0 -/

theorem memberProg_gate_zero (c : _root_.Flock.Circuit) (n : ℕ) :
    (memberProg c n).gates[0]? = some (.const true) := by
  have hin : 0 < (Spot.Bits.keyProg c n).inputs := by
    rw [keyProg_inputs]
    omega
  have h := NeededBitsOf.gate_fixed (kp := Spot.Bits.keyProg c n) (Spot.Bits.consts c n) (g := 0) (b := true)
    (NeededBitsOf.lt_size_of_lt hin) (NeededBitsOf.row?_of_lt hin) (NeededBitsOf.consts_zero c n)
  rw [NeededBitsOf.posOf_of_none (NeededBitsOf.row?_of_lt hin),
    NeededBitsOf.off_eq_of_le_inputs (Nat.zero_le _)] at h
  exact h

/-- **Member `m`'s gate 0 is the constant true**, at `Bits.base ms m`. -/
theorem bitsProgs_base_const (ms : Spot.Bits.Members) {m : ℕ} (hm : m < ms.length) :
    (bitsProgs ms).gates[Spot.Bits.base ms m]? = some (.const true) := by
  have h0 := memberProg_gate_zero ms[m].1 ms[m].2.size
  have hp : 0 < (memberProg ms[m].1 ms[m].2.size).gates.length := (List.getElem?_eq_some_iff.1 h0).1
  have h := bitsProgs_gate_base ms hm hp
  rw [Nat.add_zero, h0] at h
  exact h.trans rfl

/-! ## Returned output bits -/

theorem mem_outs_sideBySide : ∀ (ps : List Claims.Sampling.Prog) (m p : ℕ) (hm : m < ps.length),
    p ∈ ps[m].outs → offsetOf ps m + p ∈ (sideBySide ps).outs
  | [], _, _, hm, _ => absurd hm (Nat.not_lt_zero _)
  | q :: ps, 0, p, _, hp => by
    rw [offsetOf_zero, sideBySide_outs, List.mem_append]
    exact Or.inl (by simpa using hp)
  | q :: ps, m + 1, p, hm, hp => by
    have hm' : m < ps.length := by simpa using hm
    rw [offsetOf_succ, sideBySide_outs, List.mem_append]
    exact Or.inr (List.mem_map.2 ⟨offsetOf ps m + p, mem_outs_sideBySide ps m p hm' (by simpa using hp),
      (Nat.add_assoc _ _ _).symm⟩)

theorem mem_outs_bitsProgs (ms : Spot.Bits.Members) {m p : ℕ} (hm : m < ms.length)
    (hp : p ∈ (memberProg ms[m].1 ms[m].2.size).outs) : Spot.Bits.base ms m + p ∈ (bitsProgs ms).outs := by
  have h := mem_outs_sideBySide (ms.map fun ci => memberProg ci.1 ci.2.size) m p (by simpa using hm)
    (by simpa using hp)
  rw [offsetOf_bitsProgs] at h
  exact h

/-- **A returned output's bit is an AND gate the member's program returns.** -/
theorem memberProg_outBit (c : _root_.Flock.Circuit) (n : ℕ) {u k j t x : ℕ}
    (hx : Spot.Bits.outBit c n u k j t = some x) :
    x ∈ (memberProg c n).outs ∧ x < (memberProg c n).gates.length ∧
      ∃ ws, (memberProg c n).gates[x]? = some (.and ws) := by
  unfold Spot.Bits.outBit at hx
  obtain ⟨a, ha, rfl⟩ := Option.map_eq_some_iff.1 hx
  obtain ⟨hu, hk, hj, ht, -⟩ := NeededBitsOf.outGate_some c n ha
  have hao := outGate_mem_outs hu hk hj ht ha
  have hand := bitsProg_row (Spot.Bits.keyProg c n) (Spot.Bits.consts c n) (Flock.Bridge.isSome_row?_of_mem_outs hao)
  have hand' : (memberProg c n).gates[Spot.Bits.posOf (Spot.Bits.keyProg c n) a]? =
      some (.and [Spot.Bits.off (Spot.Bits.keyProg c n) a, Spot.Bits.off (Spot.Bits.keyProg c n) a + 1]) := hand
  exact ⟨by rw [NeededBitsOf.memberProg_outs]; exact List.mem_map.2 ⟨a, hao, rfl⟩,
    (List.getElem?_eq_some_iff.1 hand').1, _, hand'⟩

/-- **A returned output's bit is an output, an AND gate**, at its member's positions. -/
theorem bitsProgs_outBit (ms : Spot.Bits.Members) {m u k j t x : ℕ} (hm : m < ms.length)
    (hx : Spot.Bits.outBit ms[m].1 ms[m].2.size u k j t = some x) :
    Spot.Bits.base ms m + x ∈ (bitsProgs ms).outs ∧
      ∃ ws, (bitsProgs ms).gates[Spot.Bits.base ms m + x]? = some (.and ws) := by
  obtain ⟨ho, hlt, ws, hws⟩ := memberProg_outBit _ _ hx
  refine ⟨mem_outs_bitsProgs ms hm ho, ws.map (Spot.Bits.base ms m + ·), ?_⟩
  rw [bitsProgs_gate_base ms hm hlt, hws]
  rfl

/-! ## Input bits -/

/-- **An input bit's source gate** is a source gate of the key program. -/
theorem inGate_spec {c : _root_.Flock.Circuit} {n u k p t g : ℕ}
    (h : Spot.Bits.inGate c n u k p t = some g) : 2 ≤ g ∧ g < (Spot.Bits.keyProg c n).inputs := by
  unfold Spot.Bits.inGate at h
  dsimp only at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
    obtain ⟨⟨⟨hu, -⟩, -⟩, -⟩ := hc
    obtain ⟨b, hb, hgb⟩ := Option.map_eq_some_iff.1 h
    subst hgb
    refine ⟨show 2 ≤ 2 + u * (bitsRows c).nIn + b by omega, ?_⟩
    rw [keyProg_inputs]
    exact NeededBitsOf.src_lt hu (keyIdx_lt hb)
  · cases h

/-- **An input bit a proof block reads is an input gate of the member's program.** -/
theorem memberProg_inBit (c : _root_.Flock.Circuit) (n : ℕ) {u k p t x : ℕ}
    (hx : Spot.Bits.inBit c n u k p t = some x) (hr : Spot.Bits.owner (Spot.Bits.keyProg c n) x ≠ none) :
    x < (memberProg c n).gates.length ∧ (memberProg c n).gates[x]? = some .input := by
  unfold Spot.Bits.inBit at hx
  obtain ⟨g, hg, rfl⟩ := Option.map_eq_some_iff.1 hx
  obtain ⟨h2, hin⟩ := inGate_spec hg
  rw [posOf_of_lt_inputs _ hin] at hr ⊢
  have hg0 : ¬ g = 0 := by omega
  have hg1 : ¬ g = 1 := by omega
  have hκ : Spot.Bits.consts c n g = none := by
    simp [Spot.Bits.consts, hg0, hg1, hr]
  exact ⟨NeededBitsOf.lt_len_of_input (Spot.Bits.consts c n) hin,
    NeededBitsOf.gate_input_free (Spot.Bits.consts c n) hin hκ⟩

/-- **An input bit a proof block reads is an input gate**, at its member's positions. -/
theorem bitsProgs_inBit (ms : Spot.Bits.Members) {m u k p t x : ℕ} (hm : m < ms.length)
    (hx : Spot.Bits.inBit ms[m].1 ms[m].2.size u k p t = some x)
    (hr : Spot.Bits.owner (Spot.Bits.keyProg ms[m].1 ms[m].2.size) x ≠ none) :
    (bitsProgs ms).gates[Spot.Bits.base ms m + x]? = some .input := by
  obtain ⟨hlt, hin⟩ := memberProg_inBit _ _ hx hr
  rw [bitsProgs_gate_base ms hm hlt, hin]
  rfl

/-! ## The unit net's read columns -/

theorem getD_set_true {m : Array Bool} {y x : ℕ} (h : (m.set! y true).getD x false = true) :
    m.getD x false = true ∨ x = y := by
  by_cases hxy : x = y
  · exact Or.inr hxy
  · refine Or.inl ?_
    have hyx : y ≠ x := fun e => hxy e.symm
    have : (m.set! y true).getD x false = m.getD x false := by
      simp [Array.set!, Array.getD_eq_getD_getElem?, hyx]
    rwa [this] at h

theorem foldl_inner_reads (nIn x : ℕ) : ∀ (r : List ℕ) (m : Array Bool),
    (r.foldl (fun m y => if y < nIn then m.set! y true else m) m).getD x false = true →
      m.getD x false = true ∨ (x ∈ r ∧ x < nIn)
  | [], m, h => Or.inl h
  | y :: r, m, h => by
    rw [List.foldl_cons] at h
    rcases foldl_inner_reads nIn x r _ h with h1 | ⟨h1, h2⟩
    · by_cases hy : y < nIn
      · rw [ite_eq_left hy] at h1
        rcases getD_set_true h1 with h3 | h3
        · exact Or.inl h3
        · subst h3
          exact Or.inr ⟨List.mem_cons_self, hy⟩
      · rw [ite_eq_right hy] at h1
        exact Or.inl h1
    · exact Or.inr ⟨List.mem_cons_of_mem _ h1, h2⟩

theorem foldl_outer_reads (nIn x : ℕ) : ∀ (L : List (List ℕ)) (m : Array Bool),
    (L.foldl (fun m r => r.foldl (fun m y => if y < nIn then m.set! y true else m) m) m).getD x false = true →
      m.getD x false = true ∨ ∃ r ∈ L, x ∈ r ∧ x < nIn
  | [], m, h => Or.inl h
  | r :: L, m, h => by
    rw [List.foldl_cons] at h
    rcases foldl_outer_reads nIn x L _ h with h1 | ⟨r', hr', h2⟩
    · rcases foldl_inner_reads nIn x r m h1 with h3 | ⟨h3, h4⟩
      · exact Or.inl h3
      · exact Or.inr ⟨r, List.mem_cons_self, h3, h4⟩
    · exact Or.inr ⟨r', List.mem_cons_of_mem _ hr', h2⟩

theorem mem_getD_of_mem {A : Array (List ℕ)} {r : List ℕ} (h : r ∈ A) : ∃ j < A.size, A.getD j [] = r := by
  obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 h
  exact ⟨j, hj, by simp [Array.getD_eq_getD_getElem?, hj]⟩

/-- **A column the unit net reads** is below its input columns, and one of its computed rows' A or B reads it. -/
theorem exists_row_of_unitReads (c : _root_.Flock.Circuit) {x : ℕ}
    (h : (Spot.Bits.unitReads c).getD x false = true) :
    x < (Spot.Bits.Rows.ofNet c.unit).nIn ∧ ∃ j < (Spot.Bits.Rows.ofNet c.unit).nComp,
      x ∈ (Spot.Bits.Rows.ofNet c.unit).a.getD j [] ∨ x ∈ (Spot.Bits.Rows.ofNet c.unit).b.getD j [] := by
  unfold Spot.Bits.unitReads at h
  dsimp only at h
  rw [← Array.foldl_toList] at h
  rcases foldl_outer_reads _ x _ _ h with h1 | ⟨r, hr, hxr, hxn⟩
  · exfalso
    by_cases hx : x < (Spot.Bits.Rows.ofNet c.unit).nIn <;> simp [Array.getD_eq_getD_getElem?, hx] at h1
  · refine ⟨hxn, ?_⟩
    rw [Array.toList_append, List.mem_append] at hr
    rcases hr with hr | hr
    · have hra : r ∈ (Spot.Bits.Rows.ofNet c.unit).a := by simpa using hr
      obtain ⟨j, hj, hjr⟩ := mem_getD_of_mem hra
      refine ⟨j, ?_, Or.inl (by rw [hjr]; exact hxr)⟩
      simpa [Spot.Bits.Rows.ofNet] using hj
    · have hrb : r ∈ (Spot.Bits.Rows.ofNet c.unit).b := by simpa using hr
      obtain ⟨j, hj, hjr⟩ := mem_getD_of_mem hrb
      refine ⟨j, ?_, Or.inr (by rw [hjr]; exact hxr)⟩
      simpa [Spot.Bits.Rows.ofNet] using hj

theorem stack_a_getD (R : Spot.Bits.Rows) (k I : ℕ) (hI : I < k * (R.nComp + 1)) :
    (R.stack k).a.getD I [] = R.stackRow k R.a I := by
  simp [Spot.Bits.Rows.stack, Array.getD_eq_getD_getElem?, hI]

theorem stack_b_getD (R : Spot.Bits.Rows) (k I : ℕ) (hI : I < k * (R.nComp + 1)) :
    (R.stack k).b.getD I [] = R.stackRow k R.b I := by
  simp [Spot.Bits.Rows.stack, Array.getD_eq_getD_getElem?, hI]

/-- **An input column of instance `k` that the unit net reads is read by a stacked row.** -/
theorem inCol_mem_stack_row (c : _root_.Flock.Circuit) {k j0 x : ℕ} (hk : k < c.upv)
    (hxn : x < (Spot.Bits.Rows.ofNet c.unit).nIn) (hj0 : j0 < (Spot.Bits.Rows.ofNet c.unit).nComp)
    (hrow : x ∈ (Spot.Bits.Rows.ofNet c.unit).a.getD j0 [] ∨ x ∈ (Spot.Bits.Rows.ofNet c.unit).b.getD j0 []) :
    ∃ I < (bitsRows c).nComp,
      k * (Spot.Bits.Rows.ofNet c.unit).nIn + x ∈ (bitsRows c).a.getD I [] ∨
        k * (Spot.Bits.Rows.ofNet c.unit).nIn + x ∈ (bitsRows c).b.getD I [] := by
  have hlt : (1 + j0) + k * ((Spot.Bits.Rows.ofNet c.unit).nComp + 1) <
      c.upv * ((Spot.Bits.Rows.ofNet c.unit).nComp + 1) := by
    have := NeededBitsOf.mul_add_lt (N := (Spot.Bits.Rows.ofNet c.unit).nComp + 1) hk
      (show 1 + j0 < (Spot.Bits.Rows.ofNet c.unit).nComp + 1 by omega)
    omega
  have hmod : ((1 + j0) + k * ((Spot.Bits.Rows.ofNet c.unit).nComp + 1)) %
      ((Spot.Bits.Rows.ofNet c.unit).nComp + 1) = 1 + j0 := by
    rw [Nat.add_mul_mod_self_right]
    exact Nat.mod_eq_of_lt (by omega)
  have hdiv : ((1 + j0) + k * ((Spot.Bits.Rows.ofNet c.unit).nComp + 1)) /
      ((Spot.Bits.Rows.ofNet c.unit).nComp + 1) = k := by
    have hpos : 0 < (Spot.Bits.Rows.ofNet c.unit).nComp + 1 := Nat.succ_pos _
    rw [Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt (show 1 + j0 < (Spot.Bits.Rows.ofNet c.unit).nComp + 1 by omega),
      Nat.zero_add]
  refine ⟨(1 + j0) + k * ((Spot.Bits.Rows.ofNet c.unit).nComp + 1), hlt, ?_⟩
  unfold bitsRows
  rcases hrow with hx | hx
  · left
    rw [stack_a_getD _ _ _ hlt]
    unfold Spot.Bits.Rows.stackRow
    rw [hmod, hdiv, ite_eq_right (show ¬ (1 + j0 = 0) by omega), Nat.add_sub_cancel_left]
    exact List.mem_map.2 ⟨x, hx, by simp [Spot.Bits.Rows.stackCol, hxn]⟩
  · right
    rw [stack_b_getD _ _ _ hlt]
    unfold Spot.Bits.Rows.stackRow
    rw [hmod, hdiv, ite_eq_right (show ¬ (1 + j0 = 0) by omega), Nat.add_sub_cancel_left]
    exact List.mem_map.2 ⟨x, hx, by simp [Spot.Bits.Rows.stackCol, hxn]⟩

theorem keyWire_of_keyIdx (c : _root_.Flock.Circuit) (n u col b : ℕ) (hcol : col < (bitsRows c).nIn)
    (hb : Spot.Bits.keyIdx c (bitsRows c).nIn col = some b) :
    keyWire c n u col = 2 + u * (bitsRows c).nIn + b := by
  unfold keyWire
  rw [ite_eq_left hcol]
  have hk : ((Array.range (bitsRows c).nIn).map (Spot.Bits.keyIdx c (bitsRows c).nIn)).getD col none =
      Spot.Bits.keyIdx c (bitsRows c).nIn col := by
    simp [Array.getD_eq_getD_getElem?, hcol]
  rw [hk, hb]

/-- **The source gate on an input bit the unit net reads is owned by its unit.** -/
theorem owner_inGate_of_unitReads {c : _root_.Flock.Circuit} {n u k p t g : ℕ}
    (hg : Spot.Bits.inGate c n u k p t = some g)
    (hr : (Spot.Bits.unitReads c).getD (((_root_.Flock.portCols c.unit.inGroups).getD p default).1 + t) false = true) :
    Spot.Bits.owner (Spot.Bits.keyProg c n) g = some u := by
  unfold Spot.Bits.inGate at hg
  dsimp only at hg
  split at hg
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
    obtain ⟨⟨⟨hu, hk⟩, -⟩, -⟩ := hc
    obtain ⟨b, hb, rfl⟩ := Option.map_eq_some_iff.1 hg
    obtain ⟨hxn, j0, hj0, hrow⟩ := exists_row_of_unitReads c hr
    obtain ⟨I, hI, hcol⟩ := inCol_mem_stack_row c hk hxn hj0 hrow
    have hcolB : Spot.Bits.inCol c k p t < (bitsRows c).nIn := NeededBitsOf.mul_add_lt hk hxn
    have hbB : b < (bitsRows c).nIn := keyIdx_lt hb
    have hkw := keyWire_of_keyIdx c n u (Spot.Bits.inCol c k p t) b hcolB hb
    have hrowg := keyProg_row?_row c n hu hI
    have hunit := NeededBitsOf.unitOf_row c n hu hI
    have hw : 2 + u * (bitsRows c).nIn + b ∈ Spot.Bits.args (Spot.Bits.keyProg c n)
        (2 + n * (bitsRows c).nIn + (u * (bitsRows c).nComp + I)) := by
      simp only [Spot.Bits.args, hrowg, List.mem_append, List.mem_map]
      rcases hcol with h | h
      · exact Or.inl ⟨_, h, hkw⟩
      · exact Or.inr ⟨_, h, hkw⟩
    exact NeededBitsOf.owner_src c n hunit hbB hw
  · cases hg

end Flock.Bridge.Members
