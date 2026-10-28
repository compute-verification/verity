import Definitions.Core.Boolean
import Proofs.Core.Circuit
import Proofs.Flock.Bridge.Count
import Proofs.Flock.Soundness.Discharge.Layout.Outs

/-!
# The parsed circuit file as core's circuit, and the sampling relation as its wrong units

The circuit a C-Flock session checks is defined from its circuit file: the descriptor `D` the verifier parses
(`Flock.Desc.parse`), the unit's rows built from it (`hU : Built D.unroll.unit`, which `parse_built` gives) and the
population `n` (the public file's `pub.n`) give the key program `keyProg D.unroll hU n`, whose circuit and partition
(`keyCircuit`, `keyPartition`) the session's game and V3 are stated over. Here that circuit is core's:

* `Verity.Circuit.ofAudit C`: a game circuit (`FlockSoundness.Audit.Circuit`) as core's (`Verity.Circuit`), through the
  Boolean profile (`FlockSoundness.Audit.Circuit.toCircuit`): the same gates, one bit each. `Verity.Partition.ofAudit P` is
  the same partition, and `Verity.Circuit.ofBits xs` a transcript of bits. `Verity.Partition.wrong_ofAudit`: the wrong
  units are the same, so every statement about the game's wrong units is one about core's.
* `circuitOf D hU n := .ofAudit (keyCircuit D hU n)` and `partsOf D hU n := .ofAudit (keyPartition D hU n)`: the
  parsed file as core's circuit and its `n` units. `circuitOf_eq` and `partsOf_eq` restate the key program's circuit
  and partition as the ones `circuitOf` and `partsOf` embed (by definition), and `partsOf_wrong` its wrong units as
  theirs.
* The wrong-units bridge, the converse of `Flock.Claims.WrongUnits` (faulty ⇒ wrong): a unit wrong in the wire values a
  witness opens (`openedOf`) is faulty in the sampling statement (`wrong_subset_faulty`), because a unit that is not faulty
  is correct on its openings (`correct_of_progCorrect`). So `RelS` at `δ = C(n − K₀, kd)/C(n, kd)` leaves fewer than
  `K₀` wrong units (`card_wrong_lt_of_relS`), at every circuit and every witness, with no side condition, and at the
  parsed file's circuit (`card_wrong_partsOf_lt`).
-/

namespace Flock.Bridge

open Claims.Sampling

/-! ## A game circuit as core's -/

/-- A game gate as the Boolean profile's: the same gate. -/
def coreOp {N : ℕ} : FlockSoundness.Audit.Op N → FlockSoundness.Audit.Op N
  | .input => .input
  | .and a b => .and a b
  | .xor a b => .xor a b
  | .not a => .not a
  | .row a b => .row a b

theorem args_coreOp {N : ℕ} (o : FlockSoundness.Audit.Op N) : (coreOp o).args = o.args := by
  cases o <;> rfl

theorem isInput_coreOp {N : ℕ} (o : FlockSoundness.Audit.Op N) : (coreOp o).isInput = o.isInput := by
  cases o <;> rfl

theorem apply_coreOp {N : ℕ} (o : FlockSoundness.Audit.Op N) (v : Fin N → Bool) :
    (coreOp o).apply v = o.apply v := by
  cases o <;> rfl

/-- A game circuit as a Boolean circuit: the same gates and outputs. -/
def coreBool (C : FlockSoundness.Audit.Circuit) : FlockSoundness.Audit.Circuit where
  N := C.N
  op g := coreOp (C.op g)
  topo g a ha := C.topo g a (by rwa [args_coreOp] at ha)
  outputs := C.outputs

end Flock.Bridge

namespace Verity

open Flock.Bridge (coreOp coreBool args_coreOp isInput_coreOp apply_coreOp)

/-- **A game circuit as core's**: its Boolean circuit (`Circuit.ofAudit_eq_toCircuit`), one bit a gate, written out so
that its gates are the game circuit's. -/
abbrev Circuit.ofAudit (C : FlockSoundness.Audit.Circuit) : Circuit where
  N := C.N
  w _ := 1
  isInput g := (coreOp (C.op g)).isInput
  reads g := (coreOp (C.op g)).args.toFinset
  reads_lt g a ha := C.topo g a (by rwa [List.mem_toFinset, args_coreOp] at ha)
  input_reads g h := by simp [FlockSoundness.Audit.Op.args_of_isInput h]
  op g v := BitVec.ofBool ((coreOp (C.op g)).apply fun a => (v a).getLsbD 0)
  op_congr g v v' h := by
    rw [FlockSoundness.Audit.Op.apply_congr (coreOp (C.op g)) fun x hx => by rw [h x (List.mem_toFinset.2 hx)]]
  outputs := C.outputs

section OfAudit

variable {C : FlockSoundness.Audit.Circuit}

/-- `Circuit.ofAudit` is the Boolean profile's circuit of the game circuit. -/
theorem Circuit.ofAudit_eq_toCircuit : Circuit.ofAudit C = (coreBool C).toCircuit := rfl

theorem Circuit.ofAudit_isInput (g : Fin C.N) : (Circuit.ofAudit C).isInput g = (C.op g).isInput :=
  isInput_coreOp _

theorem Circuit.ofAudit_reads (g : Fin C.N) : (Circuit.ofAudit C).reads g = (C.op g).args.toFinset :=
  congrArg List.toFinset (args_coreOp _)

theorem Circuit.ofAudit_op (g : Fin C.N) (v : (Circuit.ofAudit C).Transcript) :
    (Circuit.ofAudit C).op g v = BitVec.ofBool ((C.op g).apply fun a => (v a).getLsbD 0) :=
  congrArg BitVec.ofBool (apply_coreOp _ _)

/-- A game partition as core's: the same units. -/
def Partition.ofAudit {n : ℕ} (P : FlockSoundness.Audit.Partition C n) : Partition (Circuit.ofAudit C) n where
  unit := P.unit
  input_iff g := (P.input_iff g).trans (by rw [Circuit.ofAudit_isInput])

/-- A transcript of bits as core's. -/
def Circuit.ofBits (xs : Fin C.N → Bool) : (Circuit.ofAudit C).Transcript := fun g => BitVec.ofBool (xs g)

variable {n : ℕ} (P : FlockSoundness.Audit.Partition C n)

theorem Partition.ofAudit_unit : (Partition.ofAudit P).unit = P.unit := rfl

theorem Partition.mem_committed_ofAudit (g : Fin C.N) :
    g ∈ (Partition.ofAudit P).committed ↔ g ∈ P.committed := by
  classical
  simp only [Partition.committed, FlockSoundness.Audit.Partition.committed, Finset.mem_filter,
    Finset.mem_univ, true_and, isInput_coreOp, args_coreOp, List.mem_toFinset, Partition.ofAudit_unit]

theorem Partition.localEval_ofAudit (xs : Fin C.N → Bool) (u : Fin n) (g : Fin C.N) :
    (Partition.ofAudit P).localEval (Circuit.ofBits xs) u g = BitVec.ofBool (P.localEval xs u g) := by
  induction g using WellFoundedLT.induction with
  | _ g ih =>
  by_cases hu : P.unit g = some u
  · rw [(Partition.ofAudit P).localEval_of_mem (Circuit.ofBits xs)
      (show (Partition.ofAudit P).unit g = some u from hu), Circuit.ofAudit_op, P.localEval_of_mem xs hu]
    congr 1
    exact FlockSoundness.Audit.Op.apply_congr _ fun a ha => by simp [ih a (C.topo g a ha)]
  · rw [(Partition.ofAudit P).localEval_of_not_mem (Circuit.ofBits xs)
      (show (Partition.ofAudit P).unit g ≠ some u from hu), P.localEval_of_not_mem xs hu]
    rfl

/-- **The wrong units are the game's**: core's partition of a game circuit has the game partition's wrong units, on
the same bits. -/
theorem Partition.wrong_ofAudit (xs : Fin C.N → Bool) :
    (Partition.ofAudit P).wrong (Circuit.ofBits xs) = P.wrong xs := by
  classical
  ext u
  rw [Partition.mem_wrong, FlockSoundness.Audit.Partition.mem_wrong]
  refine not_congr ⟨fun h g hg hu => ?_, fun h g hg hu => ?_⟩
  · have e := h g ((Partition.mem_committed_ofAudit P g).2 hg) hu
    rw [Partition.localEval_ofAudit] at e
    have e' := congrArg (fun v : BitVec 1 => v.getLsbD 0) e
    simpa [Circuit.ofBits] using e'
  · rw [Partition.localEval_ofAudit]
    exact congrArg BitVec.ofBool (h g ((Partition.mem_committed_ofAudit P g).1 hg) hu)

end OfAudit

end Verity

namespace Flock.Bridge

open Claims.Sampling

/-! ## A unit that is not faulty is correct on its openings -/

/-- **The wire values a witness opens**: an input the constants `κ` fix, its constant; every other wire, the value
opened at its position (`posOf`), `false` where none is. -/
def openedOf (C : FlockSoundness.Audit.Circuit) (κ : Fin C.N → Option Bool) (val : ℕ → Option Bool) (g : Fin C.N) :
    Bool :=
  if (C.op g).isInput then (κ g).getD ((val (posOf C g)).getD false) else (val (posOf C g)).getD false

section Opened

variable {C : FlockSoundness.Audit.Circuit}

theorem openedOf_of_not_const {κ : Fin C.N → Option Bool} (val : ℕ → Option Bool) {a : Fin C.N}
    (hc : ∀ b, (progOf C κ).gates[posOf C a]? ≠ some (.const b)) :
    openedOf C κ val a = (val (posOf C a)).getD false := by
  unfold openedOf
  split_ifs with hin
  · cases hk : κ a with
    | none => rfl
    | some b => exact absurd (gates_posOf_const (input_of_isInput hin) hk) (hc b)
  · rfl

theorem openedOf_const {κ : Fin C.N → Option Bool} (val : ℕ → Option Bool) {a : Fin C.N} {b : Bool}
    (hin : C.op a = .input) (hk : κ a = some b) :
    openedOf C κ val a = b := by
  simp [openedOf, hin, hk, FlockSoundness.Audit.Op.isInput]

/-- A gate's lowered gates read the positions of its arguments: some lowered gate reads each. -/
theorem exists_lower_reads {κ : Fin C.N → Option Bool} {g h : Fin C.N} (hg : g ∈ (C.op h).args) :
    ∃ j < gateSize (C.op h), ∃ x, (lower C κ h)[j]? = some x ∧ posOf C g ∈ x.reads := by
  unfold lower
  cases hop : C.op h with
  | input => rw [hop] at hg; simp [FlockSoundness.Audit.Op.args] at hg
  | and a b =>
    rw [hop] at hg
    refine ⟨0, by simp [gateSize], .and [posOf C a, posOf C b], by simp, ?_⟩
    simp only [FlockSoundness.Audit.Op.args, List.mem_cons, List.not_mem_nil, or_false] at hg
    rcases hg with rfl | rfl <;> simp [BGate.reads]
  | xor a b =>
    rw [hop] at hg
    refine ⟨0, by simp [gateSize], .xor [posOf C a, posOf C b], by simp, ?_⟩
    simp only [FlockSoundness.Audit.Op.args, List.mem_cons, List.not_mem_nil, or_false] at hg
    rcases hg with rfl | rfl <;> simp [BGate.reads]
  | not a =>
    rw [hop] at hg
    refine ⟨0, by simp [gateSize], .not (posOf C a), by simp, ?_⟩
    simp only [FlockSoundness.Audit.Op.args, List.mem_cons, List.not_mem_nil, or_false] at hg
    subst hg
    simp [BGate.reads]
  | row a b =>
    rw [hop] at hg
    simp only [FlockSoundness.Audit.Op.args, List.mem_append] at hg
    rcases hg with hg | hg
    · exact ⟨0, by simp [gateSize], .xor (a.map (posOf C)), by simp, List.mem_map.2 ⟨g, hg, rfl⟩⟩
    · exact ⟨1, by simp [gateSize], .xor (b.map (posOf C)), by simp, List.mem_map.2 ⟨g, hg, rfl⟩⟩

/-- **A committed wire of a unit is on its lowered unit's boundary**: an output's position is returned, and a wire
another unit reads is read at its position by one of that unit's lowered gates. -/
theorem boundary_posOf {κ : Fin C.N → Option Bool} {n : ℕ} {P : FlockSoundness.Audit.Partition C n} {u : Fin n}
    {g : Fin C.N} (hg : g ∈ P.committed) (hu : P.unit g = some u) :
    (progOf C κ).Boundary (unitGates P u) (posOf C g) = true := by
  classical
  simp only [FlockSoundness.Audit.Partition.committed, Finset.mem_filter, Finset.mem_univ, true_and] at hg
  simp only [Prog.Boundary, Bool.or_eq_true, List.contains_iff_mem, decide_eq_true_eq, List.any_eq_true,
    List.mem_range, Bool.and_eq_true, decide_not, Bool.not_eq_true', decide_eq_false_iff_not]
  rcases hg with hin | hout | ⟨h, hgh, hne⟩
  · rw [isInput_of_unit hu] at hin
    exact absurd hin (by simp)
  · exact Or.inl (Or.inl (List.mem_map.2 ⟨g, by simp [hout], rfl⟩))
  · obtain ⟨j, hj, x, hx, hr⟩ := exists_lower_reads (κ := κ) hgh
    have hgx : (progOf C κ).gates[off C h.val + j]? = some x := by rw [gates_off h hj]; exact hx
    refine Or.inr ⟨off C h.val + j, (List.getElem?_eq_some_iff.1 hgx).1, fun hm => ?_, ?_⟩
    · obtain ⟨g₂, hu₂, j₂, hj₂, he⟩ := mem_unitGates.1 hm
      obtain ⟨rfl, -⟩ := block_inj hj hj₂ he
      exact hne (by rw [hu₂, hu])
    · rw [hgx]
      simpa using hr

/-- **A unit that is not faulty is correct on its openings**: when its lowered proof block is correct in
`progOf C κ` on its committed values `val` (and any interior `ι`), each committed wire of the unit holds, in the wire
values `val` opens, what the unit computes from them. -/
theorem correct_of_progCorrect {κ : Fin C.N → Option Bool} {n : ℕ} (P : FlockSoundness.Audit.Partition C n)
    (u : Fin n) (val : ℕ → Option Bool) (ι : ℕ → Bool)
    (h : (progOf C κ).Correct ((progOf C κ).refIn (unitGates P u) val ι) (unitGates P u)) :
    P.Correct (openedOf C κ val) u := by
  have hrefB : ∀ p, (progOf C κ).Boundary (unitGates P u) p = true →
      (progOf C κ).refIn (unitGates P u) val ι p = val p := by
    intro p hp
    simp [Claims.Sampling.Prog.refIn, hp]
  have hrefO : ∀ p, p ∉ unitGates P u → (progOf C κ).refIn (unitGates P u) val ι p = val p := by
    intro p hp
    simp [Claims.Sampling.Prog.refIn, hp]
  have hread : ∀ p ∈ unitGates P u, ∀ x, (progOf C κ).gates[p]? = some x → ∀ a : Fin C.N, posOf C a ∈ x.reads →
      posOf C a ∉ unitGates P u →
      (progOf C κ).read ((progOf C κ).refIn (unitGates P u) val ι) (posOf C a) =
        some (P.localEval (openedOf C κ val) u a) := by
    intro p hp x hx a ha hna
    have hua : P.unit a ≠ some u := fun h' => hna (posOf_mem_unitGates.2 h')
    rw [P.localEval_of_not_mem _ hua]
    by_cases hc : ∃ b, (progOf C κ).gates[posOf C a]? = some (.const b)
    · obtain ⟨b, hb⟩ := hc
      rw [read_const hb]
      obtain ⟨a', ha', hin, hk⟩ := const_pos hb
      cases posOf_injective ha'
      rw [openedOf_const val hin hk]
    · have hc' : ∀ b, (progOf C κ).gates[posOf C a]? ≠ some (.const b) := fun b hb => hc ⟨b, hb⟩
      have hne := (h p hp).1 x hx (posOf C a) ha hna
      rw [Claims.Sampling.Boundary.read_eq _ hc', hrefO _ hna] at hne ⊢
      rw [openedOf_of_not_const val hc']
      cases hv : val (posOf C a) with
      | none => exact absurd hv hne
      | some b => rfl
  intro g hg hu
  have hB := boundary_posOf (κ := κ) hg hu
  obtain ⟨b, hb, hrun⟩ := (h (posOf C g) (posOf_mem_unitGates.2 hu)).2 hB
  have hrun' := runIn_posOf_of (κ := κ) P u (P.localEval (openedOf C κ val) u)
    (fun g hg => P.localEval_of_mem _ hg) _ hread g.val g rfl hu
  rw [hrefB _ hB] at hb
  have hX : openedOf C κ val g = b := by simp [openedOf, isInput_of_unit hu, hb]
  rw [hX]
  rw [hrun] at hrun'
  exact Option.some_injective _ hrun'

/-- **A unit wrong in the wire values a witness opens is faulty** in the sampling statement of `progOf C κ`. -/
theorem wrong_subset_faulty {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (κ : Fin C.N → Option Bool) (kd : ℕ)
    (desc : Claims.Msg) (q : Query) (roots : List Claims.Msg) (H : Claims.Msg → Claims.Msg) (Cm : Commitment Bool)
    (w : Wit) {u : Fin n} (hu : u ∈ P.wrong (openedOf C κ (w.val H Cm (stmtOf P κ kd desc q roots)))) :
    u.val ∈ faulty H Cm (stmtOf P κ kd desc q roots) (partOf P κ) w := by
  classical
  rw [FlockSoundness.Audit.Partition.mem_wrong] at hu
  have hlt := u.isLt
  refine Finset.mem_filter.2 ⟨Finset.mem_range.2 (by rw [blocks_length_partOf]; omega), ?_⟩
  rintro ⟨b, hb, ι, hι⟩
  rw [blocks_partOf_getElem? P u] at hb
  cases Option.some.inj hb
  change (progOf C κ).Correct ((progOf C κ).refIn (blockOf P κ u) _ ι) (blockOf P κ u) at hι
  rw [ref_blockOf, correct_blockOf] at hι
  exact hu (correct_of_progCorrect P u _ ι hι.1)

theorem card_wrong_le_faulty {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (κ : Fin C.N → Option Bool) (kd : ℕ)
    (desc : Claims.Msg) (q : Query) (roots : List Claims.Msg) (H : Claims.Msg → Claims.Msg) (Cm : Commitment Bool)
    (w : Wit) :
    (P.wrong (openedOf C κ (w.val H Cm (stmtOf P κ kd desc q roots)))).card ≤
      ((faulty H Cm (stmtOf P κ kd desc q roots) (partOf P κ) w).filter (· < n)).card :=
  Finset.card_le_card_of_injOn Fin.val
    (by intro u hu; exact Finset.mem_filter.2 ⟨wrong_subset_faulty P κ kd desc q roots H Cm w hu, u.isLt⟩)
    (by intro a _ b _ h; exact Fin.ext h)

/-- **The sampling relation bounds the wrong units**: a witness in `RelS` at `δ = C(n − K₀, kd)/C(n, kd)` leaves fewer
than `K₀` of the circuit's units wrong in the wire values it opens. -/
theorem card_wrong_lt_of_relS {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (κ : Fin C.N → Option Bool)
    (kd K₀ : ℕ) (desc : Claims.Msg) (q : Query) (roots : List Claims.Msg) (H : Claims.Msg → Claims.Msg)
    (Cm : Commitment Bool) (w : Wit)
    (h : RelS H Cm (((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)) (stmtOf P κ kd desc q roots) (partOf P κ) w) :
    (P.wrong (openedOf C κ (w.val H Cm (stmtOf P κ kd desc q roots)))).card < K₀ := by
  have hle := card_wrong_le_faulty P κ kd desc q roots H Cm w
  have hpa : ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) <
      (designOf kd).Pa (partOf P κ) (faulty H Cm (stmtOf P κ kd desc q roots) (partOf P κ) w) := h
  rw [pa_partOf_window, window_zero_filter] at hpa
  generalize (faulty H Cm (stmtOf P κ kd desc q roots) (partOf P κ) w).filter (· < n) = S at hpa hle
  by_contra hK
  have hc : (n - S.card).choose kd ≤ (n - K₀).choose kd := Nat.choose_le_choose _ (by omega)
  have hdiv : ((n - S.card).choose kd : ℝ) / (n.choose kd : ℝ) ≤ ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) := by
    gcongr
  linarith

end Opened

/-! ## The parsed file's circuit -/

/-- The key program's circuit at the descriptor `D`, its unit's rows `hU` and the population `n`. -/
noncomputable abbrev keyCircuit (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (n : ℕ) :
    FlockSoundness.Audit.Circuit :=
  (FlockSoundness.Discharge.Placed.keyProg D.unroll hU n).circuit (FlockSoundness.Discharge.Layout.outsOf D.unroll hU n)

/-- The key program's partition into its `n` units. -/
noncomputable abbrev keyPartition (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (n : ℕ) :
    FlockSoundness.Audit.Partition (keyCircuit D hU n) n :=
  (FlockSoundness.Discharge.Placed.keyProg D.unroll hU n).partition
    (FlockSoundness.Discharge.Layout.outsOf D.unroll hU n)

/-- **The parsed circuit file as core's circuit**: the key program of the descriptor `D`, its unit's rows `hU` and the
population `n`, through the Boolean profile. -/
noncomputable def circuitOf (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (n : ℕ) : Verity.Circuit :=
  .ofAudit (keyCircuit D hU n)

/-- **The parsed circuit file's units**, as core's partition of `circuitOf`. -/
noncomputable def partsOf (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (n : ℕ) :
    Verity.Partition (circuitOf D hU n) n :=
  .ofAudit (keyPartition D hU n)

/-- **The transcript the witness opens**, at `circuitOf`'s wires: `openedOf`'s bits. -/
noncomputable def transcriptOf (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (n : ℕ)
    (κ : Fin (keyCircuit D hU n).N → Option Bool) (val : ℕ → Option Bool) : (circuitOf D hU n).Transcript :=
  Verity.Circuit.ofBits (openedOf (keyCircuit D hU n) κ val)

section Parsed

variable (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (n : ℕ)

/-- **The key program's circuit restated from `circuitOf`**: `circuitOf` is it, through the Boolean profile. -/
theorem circuitOf_eq : circuitOf D hU n = .ofAudit ((FlockSoundness.Discharge.Placed.keyProg D.unroll hU n).circuit
    (FlockSoundness.Discharge.Layout.outsOf D.unroll hU n)) := rfl

/-- **The key program's partition restated from `partsOf`.** -/
theorem partsOf_eq : partsOf D hU n =
    .ofAudit ((FlockSoundness.Discharge.Placed.keyProg D.unroll hU n).partition
      (FlockSoundness.Discharge.Layout.outsOf D.unroll hU n)) := rfl

/-- **`partsOf`'s wrong units are the key program's**, on the same bits. -/
theorem partsOf_wrong (xs : Fin (keyCircuit D hU n).N → Bool) :
    (partsOf D hU n).wrong (Verity.Circuit.ofBits xs) = (keyPartition D hU n).wrong xs :=
  Verity.Partition.wrong_ofAudit _ xs

/-- **The wrong-units bridge at the parsed file's circuit**: a witness in `RelS` at `δ = C(n − K₀, kd)/C(n, kd)`, in
the sampling statement of the key program at constants `κ`, leaves fewer than `K₀` of `partsOf`'s units wrong in the
transcript it opens. -/
theorem card_wrong_partsOf_lt (κ : Fin (keyCircuit D hU n).N → Option Bool) (kd K₀ : ℕ) (desc : Claims.Msg)
    (q : Query) (roots : List Claims.Msg) (H : Claims.Msg → Claims.Msg) (Cm : Commitment Bool) (w : Wit)
    (h : RelS H Cm (((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)) (stmtOf (keyPartition D hU n) κ kd desc q roots)
      (partOf (keyPartition D hU n) κ) w) :
    ((partsOf D hU n).wrong
      (transcriptOf D hU n κ (w.val H Cm (stmtOf (keyPartition D hU n) κ kd desc q roots)))).card < K₀ := by
  rw [transcriptOf, partsOf_wrong]
  exact card_wrong_lt_of_relS _ κ kd K₀ desc q roots H Cm w h

end Parsed

end Flock.Bridge
