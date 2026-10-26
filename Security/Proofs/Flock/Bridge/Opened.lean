import Proofs.Core.Circuit
import Proofs.Flock.Bridge.CircuitOf

/-!
# A unit that is not faulty is bound by its openings

R4's event reads the commitment: every transcript the commitment opens to (each committed wire at its registered
position, a wire with no valid opening counting as wrong) has few wrong units, unless two openings of one position
break binding. This file gives the step from the sampling relation to that form, at a game circuit `C`, its partition
`P`, the constants `κ` and the roots the statement carries:

* `OpensAt C κ H Cm roots g v`: wire `g` opens to `v`: a constant input its constant, every other wire an opening
  under `roots` at its position (`posOf`) that passes.
* `bound_of_not_faulty`: a unit that is not faulty in the sampling statement opens every wire of its `io`
  (its committed inputs and outputs) to the value `openedOf` reads.
* `unbound`: the units wrong in `openedOf`'s transcript, or with a wire of their `io` that doesn't open to it;
  `card_unbound_lt_of_relS`: `RelS` at `δ = C(n − K₀, kd)/C(n, kd)` leaves fewer than `K₀` of them.
* `correct_of_not_unbound`: where openings are unique (`Unique`, the negation of a break on these roots), a unit
  outside `unbound` is correct in every transcript its `io` opens to, so the bound holds of each such transcript.
-/

namespace Flock.Bridge

open Claims.Sampling

variable {C : FlockSoundness.Audit.Circuit}

/-- **Wire `g` opens to `v`** under `roots`: an input the constants `κ` fix, to its constant; any other wire, by an
opening at its position that passes. -/
def OpensAt (C : FlockSoundness.Audit.Circuit) (κ : Fin C.N → Option Bool) (H : Claims.Msg → Claims.Msg)
    (Cm : Commitment Bool) (roots : List Claims.Msg) (g : Fin C.N) (v : Bool) : Prop :=
  if (C.op g).isInput = true ∧ (κ g).isSome = true then κ g = some v
  else ∃ o : Opening Bool, o.pos = posOf C g ∧ Cm.Opens H roots o = true ∧ o.val = v

/-- **Openings are unique** under `roots`: two openings of one position that pass carry one value. -/
def Unique (H : Claims.Msg → Claims.Msg) (Cm : Commitment Bool) (roots : List Claims.Msg) : Prop :=
  ∀ o o' : Opening Bool, o.pos = o'.pos → Cm.Opens H roots o = true → Cm.Opens H roots o' = true → o.val = o'.val

theorem opening_of_val {H : Claims.Msg → Claims.Msg} {Cm : Commitment Bool} {s : Claims.Sampling.Stmt} {w : Wit}
    {i : ℕ} {b : Bool}
    (h : w.val H Cm s i = some b) : ∃ o : Opening Bool, o.pos = i ∧ Cm.Opens H s.roots o = true ∧ o.val = b := by
  unfold Wit.val at h
  cases hf : w.vals.find? (fun o => o.pos == i && Cm.Opens H s.roots o) with
  | none => rw [hf] at h; exact absurd h (by simp)
  | some o =>
    rw [hf] at h
    have hp := List.find?_some hf
    simp only [Bool.and_eq_true, beq_iff_eq] at hp
    exact ⟨o, hp.1, hp.2, by simpa using h⟩

theorem opensAt_of_val {κ : Fin C.N → Option Bool} {H : Claims.Msg → Claims.Msg} {Cm : Commitment Bool}
    {s : Claims.Sampling.Stmt} {w : Wit} {g : Fin C.N} {b : Bool} (hk : ¬((C.op g).isInput = true ∧ (κ g).isSome = true))
    (h : w.val H Cm s (posOf C g) = some b) : OpensAt C κ H Cm s.roots g b := by
  rw [OpensAt, if_neg hk]
  exact opening_of_val h

theorem opensAt_unique {κ : Fin C.N → Option Bool} {H : Claims.Msg → Claims.Msg} {Cm : Commitment Bool}
    {roots : List Claims.Msg} (hU : Unique H Cm roots) {g : Fin C.N} {v v' : Bool}
    (h : OpensAt C κ H Cm roots g v) (h' : OpensAt C κ H Cm roots g v') : v = v' := by
  unfold OpensAt at h h'
  split_ifs at h h'
  · exact Option.some.inj (h.symm.trans h')
  · obtain ⟨o, ho, hop, rfl⟩ := h
    obtain ⟨o', ho', hop', rfl⟩ := h'
    exact hU o o' (ho.trans ho'.symm) hop hop'

theorem correct_of_not_faulty {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (κ : Fin C.N → Option Bool) (kd : ℕ)
    (desc : Claims.Msg) (q : Query) (roots : List Claims.Msg) (H : Claims.Msg → Claims.Msg) (Cm : Commitment Bool)
    (w : Wit) {u : Fin n} (h : u.val ∉ faulty H Cm (stmtOf P κ kd desc q roots) (partOf P) w) :
    ∃ ι : ℕ → Bool, (progOf C κ).Correct
      (RU.ref (progOf C κ) [unitGates P u] (w.val H Cm (stmtOf P κ kd desc q roots)) ι) (unitGates P u) := by
  classical
  have hlt := u.isLt
  have hF : ¬ Faulty H Cm (stmtOf P κ kd desc q roots) (partOf P) w u.val := fun hf =>
    h (Finset.mem_filter.2 ⟨Finset.mem_range.2 (by rw [rus_length_partOf]; omega), hf⟩)
  obtain ⟨r, hr, ι, hι⟩ := not_not.1 hF
  rw [rus_partOf_getElem? P u] at hr
  cases Option.some.inj hr
  exact ⟨ι, hι _ (List.mem_singleton_self _)⟩

/-- **A unit that is not faulty is bound by its openings**: each wire of its `io` (its committed inputs and outputs)
opens, under the statement's roots, to the value `openedOf` reads there. -/
theorem bound_of_not_faulty {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (κ : Fin C.N → Option Bool) (kd : ℕ)
    (desc : Claims.Msg) (q : Query) (roots : List Claims.Msg) (H : Claims.Msg → Claims.Msg) (Cm : Commitment Bool)
    (w : Wit) {u : Fin n} (h : u.val ∉ faulty H Cm (stmtOf P κ kd desc q roots) (partOf P) w) :
    ∀ g : Fin C.N, g ∈ (Verity.Partition.ofAudit P).io u →
      OpensAt C κ H Cm roots g (openedOf C κ (w.val H Cm (stmtOf P κ kd desc q roots)) g) := by
  classical
  obtain ⟨ι, hc⟩ := correct_of_not_faulty P κ kd desc q roots H Cm w h
  have hrefB : ∀ p, (progOf C κ).Boundary (unitGates P u) p = true →
      RU.ref (progOf C κ) [unitGates P u] (w.val H Cm (stmtOf P κ kd desc q roots)) ι p =
        w.val H Cm (stmtOf P κ kd desc q roots) p := by
    intro p hp
    simp [RU.ref, RU.gates, hp]
  have hrefO : ∀ p, p ∉ unitGates P u →
      RU.ref (progOf C κ) [unitGates P u] (w.val H Cm (stmtOf P κ kd desc q roots)) ι p =
        w.val H Cm (stmtOf P κ kd desc q roots) p := by
    intro p hp
    simp [RU.ref, RU.gates, hp]
  intro g hg
  obtain ⟨hgc, hrd⟩ := Finset.mem_filter.1 hg
  by_cases hgu : P.unit g = some u
  · have hB := boundary_posOf (κ := κ) ((Verity.Partition.mem_committed_ofAudit P g).1 hgc) hgu
    obtain ⟨b, hb, -⟩ := (hc (posOf C g) (posOf_mem_unitGates.2 hgu)).2 hB
    rw [hrefB _ hB] at hb
    have hin := isInput_of_unit hgu
    have hnin : ¬((C.op g).isInput = true) := by rw [hin]; exact Bool.false_ne_true
    rw [show openedOf C κ (w.val H Cm (stmtOf P κ kd desc q roots)) g = b by
      unfold openedOf; rw [if_neg hnin, hb]; rfl]
    exact opensAt_of_val (fun hk => hnin hk.1) hb
  · obtain ⟨h', hu', hr'⟩ := hrd.resolve_left hgu
    rw [Verity.Circuit.ofAudit_reads, List.mem_toFinset] at hr'
    have hu' : P.unit h' = some u := hu'
    obtain ⟨j, hj, x, hx, hrx⟩ := exists_lower_reads (κ := κ) hr'
    have hgx : (progOf C κ).gates[off C h'.val + j]? = some x := by rw [gates_off h' hj]; exact hx
    have hp : off C h'.val + j ∈ unitGates P u := mem_unitGates.2 ⟨h', hu', j, hj, rfl⟩
    have hna : posOf C g ∉ unitGates P u := fun hm => hgu (posOf_mem_unitGates.1 hm)
    by_cases hcst : ∃ b, (progOf C κ).gates[posOf C g]? = some (.const b)
    · obtain ⟨b, hb⟩ := hcst
      obtain ⟨a, ha, hin, hk⟩ := const_pos hb
      obtain rfl := posOf_injective ha
      rw [openedOf_const _ hin hk]
      simp [OpensAt, hin, hk, FlockSoundness.Audit.Op.isInput]
    · have hc' : ∀ b, (progOf C κ).gates[posOf C g]? ≠ some (.const b) := fun b hb => hcst ⟨b, hb⟩
      have hne := (hc _ hp).1 x hgx (posOf C g) hrx hna
      rw [Claims.Sampling.Boundary.read_eq _ hc', hrefO _ hna] at hne
      rw [openedOf_of_not_const _ hc']
      obtain ⟨b, hb⟩ := Option.ne_none_iff_exists'.1 hne
      rw [hb, Option.getD_some]
      refine opensAt_of_val ?_ hb
      rintro ⟨hin, hk⟩
      obtain ⟨b', hk'⟩ := Option.isSome_iff_exists.1 hk
      exact hc' b' (gates_posOf_const (input_of_isInput hin) hk')

open Classical in
/-- **The unbound units** of the opened values `val`: wrong in `openedOf`'s transcript, or with a wire of their `io`
that doesn't open to it under `roots`. -/
noncomputable def unbound {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (κ : Fin C.N → Option Bool)
    (H : Claims.Msg → Claims.Msg) (Cm : Commitment Bool) (roots : List Claims.Msg) (val : ℕ → Option Bool) :
    Finset (Fin n) :=
  Finset.univ.filter fun u =>
    ¬ (Verity.Partition.ofAudit P).Correct (Verity.Circuit.ofBits (openedOf C κ val)) u ∨
      ∃ g : Fin C.N, g ∈ (Verity.Partition.ofAudit P).io u ∧ ¬ OpensAt C κ H Cm roots g (openedOf C κ val g)

/-- An unbound unit is faulty. -/
theorem unbound_subset_faulty {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (κ : Fin C.N → Option Bool) (kd : ℕ)
    (desc : Claims.Msg) (q : Query) (roots : List Claims.Msg) (H : Claims.Msg → Claims.Msg) (Cm : Commitment Bool)
    (w : Wit) {u : Fin n} (hu : u ∈ unbound P κ H Cm roots (w.val H Cm (stmtOf P κ kd desc q roots))) :
    u.val ∈ faulty H Cm (stmtOf P κ kd desc q roots) (partOf P) w := by
  classical
  rcases (Finset.mem_filter.1 hu).2 with hw | ⟨g, hg, hn⟩
  · have hm : u ∈ (Verity.Partition.ofAudit P).wrong
        (Verity.Circuit.ofBits (openedOf C κ (w.val H Cm (stmtOf P κ kd desc q roots)))) :=
      (Verity.Partition.ofAudit P).mem_wrong.2 hw
    rw [Verity.Partition.wrong_ofAudit] at hm
    exact wrong_subset_faulty P κ kd desc q roots H Cm w hm
  · by_contra hf
    exact hn (bound_of_not_faulty P κ kd desc q roots H Cm w hf g hg)

/-- **The sampling relation bounds the unbound units**: a witness in `RelS` at `δ = C(n − K₀, kd)/C(n, kd)` leaves
fewer than `K₀` units wrong in the wire values it opens or with a wire of their `io` that doesn't open there. -/
theorem card_unbound_lt_of_relS {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (κ : Fin C.N → Option Bool)
    (kd K₀ : ℕ) (desc : Claims.Msg) (q : Query) (roots : List Claims.Msg) (H : Claims.Msg → Claims.Msg)
    (Cm : Commitment Bool) (w : Wit)
    (h : RelS H Cm (((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)) (stmtOf P κ kd desc q roots) (partOf P) w) :
    (unbound P κ H Cm roots (w.val H Cm (stmtOf P κ kd desc q roots))).card < K₀ := by
  have hle : (unbound P κ H Cm roots (w.val H Cm (stmtOf P κ kd desc q roots))).card ≤
      ((faulty H Cm (stmtOf P κ kd desc q roots) (partOf P) w).filter (· < n)).card :=
    Finset.card_le_card_of_injOn Fin.val
      (by intro u hu; exact Finset.mem_filter.2 ⟨unbound_subset_faulty P κ kd desc q roots H Cm w hu, u.isLt⟩)
      (by intro a _ b _ h; exact Fin.ext h)
  have hpa : ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) <
      (designOf kd).Pa (partOf P) (faulty H Cm (stmtOf P κ kd desc q roots) (partOf P) w) := h
  rw [pa_partOf_window, window_zero_filter] at hpa
  generalize (faulty H Cm (stmtOf P κ kd desc q roots) (partOf P) w).filter (· < n) = S at hpa hle
  by_contra hK
  have hc : (n - S.card).choose kd ≤ (n - K₀).choose kd := Nat.choose_le_choose _ (by omega)
  have hdiv : ((n - S.card).choose kd : ℝ) / (n.choose kd : ℝ) ≤ ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) := by
    gcongr
  linarith

theorem ofBool_getLsbD_zero (x : BitVec 1) : BitVec.ofBool (x.getLsbD 0) = x := by
  refine BitVec.eq_of_getLsbD_eq fun i hi => ?_
  obtain rfl : i = 0 := by omega
  simp

/-- **A bound unit is correct in every transcript its `io` opens to**: where openings under `roots` are unique, a unit
outside `unbound` is correct in any transcript `T` each wire of whose `io` opens to `T`'s bit there. -/
theorem correct_of_not_unbound {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (κ : Fin C.N → Option Bool)
    (H : Claims.Msg → Claims.Msg) (Cm : Commitment Bool) (roots : List Claims.Msg) (val : ℕ → Option Bool)
    (hU : Unique H Cm roots) {u : Fin n} (hu : u ∉ unbound P κ H Cm roots val)
    {T : (Verity.Circuit.ofAudit C).Transcript}
    (hT : ∀ g : Fin C.N, g ∈ (Verity.Partition.ofAudit P).io u → OpensAt C κ H Cm roots g ((T g).getLsbD 0)) :
    (Verity.Partition.ofAudit P).Correct T u := by
  classical
  have hu' : ¬(¬ (Verity.Partition.ofAudit P).Correct (Verity.Circuit.ofBits (openedOf C κ val)) u ∨
      ∃ g : Fin C.N, g ∈ (Verity.Partition.ofAudit P).io u ∧ ¬ OpensAt C κ H Cm roots g (openedOf C κ val g)) :=
    fun h => hu (Finset.mem_filter.2 ⟨Finset.mem_univ _, h⟩)
  have hc := not_not.1 fun h => hu' (Or.inl h)
  have hb : ∀ g : Fin C.N, g ∈ (Verity.Partition.ofAudit P).io u → OpensAt C κ H Cm roots g (openedOf C κ val g) :=
    fun g hg => not_not.1 fun h => hu' (Or.inr ⟨g, hg, h⟩)
  refine (Verity.Partition.ofAudit P).correct_congr (fun g hg => ?_) hc
  calc T g = BitVec.ofBool ((T g).getLsbD 0) := (ofBool_getLsbD_zero _).symm
    _ = Verity.Circuit.ofBits (openedOf C κ val) g := by
      rw [opensAt_unique hU (hT g hg) (hb g hg)]
      rfl

end Flock.Bridge
