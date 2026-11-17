module

public import Definitions.Compose.JointLive
public import Proofs.Flock.Bridge.CircuitOf
public import Proofs.Flock.Bridge.FaultyRightC

/-! A unit unbound in the transcript a witness opens is faulty. -/

@[expose] public section

namespace Flock.Bridge

open Claims.Sampling

theorem val_roots {H : Claims.Msg → Claims.Msg} {Cm : CommitmentScheme Bool} {s s' : AuditStatement} (w : Wit)
    (h : s.roots = s'.roots) : w.val H Cm s = w.val H Cm s' := by
  unfold Wit.val
  rw [h]

theorem opens_of_val {H : Claims.Msg → Claims.Msg} {Cm : CommitmentScheme Bool} {s : AuditStatement} {w : Wit} {p : ℕ}
    {b : Bool} (h : w.val H Cm s p = some b) : ∃ d : Claims.Msg, Cm.Opens H s.roots ⟨p, b, d⟩ = true := by
  unfold Wit.val at h
  obtain ⟨o, ho, hb⟩ := Option.map_eq_some_iff.1 h
  have hpo := List.find?_some ho
  simp only [Bool.and_eq_true, beq_iff_eq] at hpo
  obtain ⟨hp, hO⟩ := hpo
  subst hp hb
  exact ⟨o.data, hO⟩

section Unbound

variable {C : Verity.Boolean.Circuit}

theorem not_correct_of_unbound {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (κ : Fin C.N → Option Bool)
    (val : ℕ → Option Bool)
    (opens : (g : Fin (Verity.Circuit.ofAudit C).N) → BitVec ((Verity.Circuit.ofAudit C).w g) → Prop)
    (hκ : ∀ g b, κ g = some b → (C.op g).isInput = true)
    (hK : ∀ g b, κ g = some b → opens g (BitVec.ofBool b))
    (hO : ∀ g b, κ g = none → val (posOf C g) = some b → opens g (BitVec.ofBool b))
    {u : Fin n}
    (hu : u ∈ ResourceAccounting.unbound (Verity.Partition.ofAudit P) opens (Verity.Circuit.ofBits (openedOf C κ val)))
    (ι : ℕ → Bool) :
    ¬ (progOf C κ).Correct ((progOf C κ).refIn (unitGates P u) val ι) (unitGates P u) := by
  classical
  intro hc
  simp only [ResourceAccounting.unbound, Finset.mem_filter, Finset.mem_univ, true_and] at hu
  rcases hu with hw | ⟨g, hg, hop⟩
  · have h1 : u ∉ P.incorrect (openedOf C κ val) := fun hm => P.mem_incorrect.1 hm (correct_of_progCorrect P u val ι hc)
    rw [← Verity.Partition.incorrect_ofAudit P] at h1
    exact h1 ((Verity.Partition.ofAudit P).mem_incorrect.2 hw)
  · simp only [Verity.Partition.io, Finset.mem_filter] at hg
    obtain ⟨hgc, hgu⟩ := hg
    have hgc' : g ∈ P.committed := (Verity.Partition.mem_committed_ofAudit P g).1 hgc
    have hX : ∀ b, openedOf C κ val g = b → opens g (BitVec.ofBool b) →
        opens g (Verity.Circuit.ofBits (openedOf C κ val) g) := by
      rintro b rfl h
      exact h
    cases hk : κ g with
    | some b => exact hop (hX b (openedOf_const val (input_of_isInput (hκ g b hk)) hk) (hK g b hk))
    | none =>
      cases hv : val (posOf C g) with
      | some b =>
        refine hop (hX b ?_ (hO g b hk hv))
        unfold openedOf
        split_ifs <;> simp [hk, hv]
      | none =>
        by_cases hgU : P.unit g = some u
        · have hB := boundary_posOf (κ := κ) hgc' hgU
          obtain ⟨b, hb, -⟩ := (hc (posOf C g) (posOf_mem_unitGates.2 hgU)).2 hB
          simp [Claims.Sampling.Prog.refIn, hB, hv] at hb
        · rcases hgu with hgu | ⟨h, hhu, hgh⟩
          · exact hgU hgu
          · change g ∈ (Verity.Circuit.ofAudit C).reads h at hgh
            rw [Verity.Circuit.ofAudit_reads, List.mem_toFinset] at hgh
            obtain ⟨j, hj, x, hx, hr⟩ := exists_lower_reads (κ := κ) hgh
            have hgx : (progOf C κ).gates[off C h.val + j]? = some x := by rw [gates_off h hj]; exact hx
            have hp : off C h.val + j ∈ unitGates P u := mem_unitGates.2 ⟨h, hhu, j, hj, rfl⟩
            have hna : posOf C g ∉ unitGates P u := fun hm => hgU (posOf_mem_unitGates.1 hm)
            have hnc : ∀ b, (progOf C κ).gates[posOf C g]? ≠ some (.const b) := by
              intro b hb
              obtain ⟨a, ha, -, hka⟩ := const_pos hb
              have hga := posOf_injective ha
              subst hga
              rw [hk] at hka
              cases hka
            have hne := (hc _ hp).1 x hgx (posOf C g) hr hna
            rw [Claims.Sampling.Boundary.read_eq _ hnc] at hne
            apply hne
            simp [Claims.Sampling.Prog.refIn, hna, hv]

theorem unbound_subset_faulty_slots {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (κ : Fin C.N → Option Bool)
    (H : Claims.Msg → Claims.Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (hs : s.prog = progOf C κ)
    (Ps : Claims.Sampling.Partition) (hslot : ∀ u : Fin n, Ps.blocks[u.val]? = some (unitGates P u)) (w : Wit)
    (opens : (g : Fin (Verity.Circuit.ofAudit C).N) → BitVec ((Verity.Circuit.ofAudit C).w g) → Prop)
    (hκ : ∀ g b, κ g = some b → (C.op g).isInput = true)
    (hK : ∀ g b, κ g = some b → opens g (BitVec.ofBool b))
    (hO : ∀ g b, κ g = none → w.val H Cm s (posOf C g) = some b → opens g (BitVec.ofBool b)) :
    (ResourceAccounting.unbound (Verity.Partition.ofAudit P) opens
        (Verity.Circuit.ofBits (openedOf C κ (w.val H Cm s)))).image Fin.val ⊆ faulty H Cm s Ps w := by
  classical
  intro v hv
  obtain ⟨u, hu, rfl⟩ := Finset.mem_image.1 hv
  refine Finset.mem_filter.2 ⟨Finset.mem_range.2 (List.getElem?_eq_some_iff.1 (hslot u)).1, ?_⟩
  rintro ⟨b, hb, ι, hι⟩
  rw [hslot u] at hb
  cases Option.some.inj hb
  rw [hs] at hι
  exact not_correct_of_unbound P κ _ opens hκ hK hO hu ι hι

theorem unbound_subset_faulty_partOf {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (κ : Fin C.N → Option Bool)
    {ns : List ℕ} (hns : ns.sum = n) (H : Claims.Msg → Claims.Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement)
    (hs : s.prog = progOf C κ) (w : Wit)
    (opens : (g : Fin (Verity.Circuit.ofAudit C).N) → BitVec ((Verity.Circuit.ofAudit C).w g) → Prop)
    (hκ : ∀ g b, κ g = some b → (C.op g).isInput = true)
    (hK : ∀ g b, κ g = some b → opens g (BitVec.ofBool b))
    (hO : ∀ g b, κ g = none → w.val H Cm s (posOf C g) = some b → opens g (BitVec.ofBool b)) :
    (ResourceAccounting.unbound (Verity.Partition.ofAudit P) opens
        (Verity.Circuit.ofBits (openedOf C κ (w.val H Cm s)))).image Fin.val ⊆ faulty H Cm s (partOf P κ ns) w := by
  classical
  intro v hv
  obtain ⟨u, hu, rfl⟩ := Finset.mem_image.1 hv
  refine Finset.mem_filter.2 ⟨Finset.mem_range.2 (by rw [blocks_length_partOf P hns]; exact u.isLt), ?_⟩
  rintro ⟨b, hb, ι, hι⟩
  rw [blocks_partOf_getElem? P hns u] at hb
  cases Option.some.inj hb
  rw [hs, ref_blockOf, correct_blockOf] at hι
  exact not_correct_of_unbound P κ _ opens hκ hK hO hu ι hι.1

/-- The premises hold at the honest instance, where no unit is unbound: the witness opening the evaluation on
the honest inputs `Xr`. -/
theorem unbound_subset_faulty_honest {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (κ : Fin C.N → Option Bool)
    (roots : List Claims.Msg) (H : Claims.Msg → Claims.Msg) (Cm : CommitmentScheme Bool) (Xr : Fin C.N → Bool)
    (hκin : ∀ g b, κ g = some b → (C.op g).isInput = true) (hκ : ∀ g b, κ g = some b → C.eval Xr g = b)
    (hcom : ∀ g, κ g = none → ∃ d : Claims.Msg, Cm.Opens H roots ⟨posOf C g, C.eval Xr g, d⟩ = true)
    (s : AuditStatement) (hr : s.roots = roots)
    (opens : (g : Fin (Verity.Circuit.ofAudit C).N) → BitVec ((Verity.Circuit.ofAudit C).w g) → Prop)
    (hop : ∀ g v, opens g v ↔
      κ g = some (v.getLsbD 0) ∨ ∃ d : Claims.Msg, Cm.Opens H roots ⟨posOf C g, v.getLsbD 0, d⟩ = true) :
    (∀ g b, κ g = some b → opens g (BitVec.ofBool b)) ∧
      (∀ g b, κ g = none → (witDep C Cm H roots Xr).val H Cm s (posOf C g) = some b → opens g (BitVec.ofBool b)) ∧
      ResourceAccounting.unbound (Verity.Partition.ofAudit P) opens
        (Verity.Circuit.ofBits (openedOf C κ ((witDep C Cm H roots Xr).val H Cm s))) = ∅ := by
  classical
  have hK : ∀ g b, κ g = some b → opens g (BitVec.ofBool b) := fun g b hk =>
    (hop g _).2 (Or.inl (by simpa [BitVec.getLsbD_ofBool] using hk))
  have hO : ∀ g b, κ g = none → (witDep C Cm H roots Xr).val H Cm s (posOf C g) = some b →
      opens g (BitVec.ofBool b) := fun g b _ hv => by
    obtain ⟨d, hd⟩ := opens_of_val hv
    exact (hop g _).2 (Or.inr ⟨d, by simpa [BitVec.getLsbD_ofBool, hr] using hd⟩)
  refine ⟨hK, hO, Finset.eq_empty_of_forall_notMem fun u hu => ?_⟩
  have hs' : s.roots = (stmtOf P κ .all [] ⟨"", 0, []⟩ roots).roots := hr
  rw [val_roots _ hs'] at hu
  have hsub := unbound_subset_faulty_partOf P κ (ns := [n]) (by simp) H Cm (stmtOf P κ .all [] ⟨"", 0, []⟩ roots) rfl
    (witDep C Cm H roots Xr) opens hκin hK (fun g b hk hv => hO g b hk (by rwa [val_roots _ hs']))
    (Finset.mem_image_of_mem Fin.val hu)
  exact faultyRightC P (fun g : Fin C.N => g) (fun (b : Bool) (_ : Fin C.N) => b) (id : Bool → Bool)
    (id : Bool → Bool) (C.eval Xr) (fun _ _ => True) H Cm roots Xr κ u hκin
    ⟨C.eval Xr, fun _ _ => ⟨rfl, rfl, trivial⟩⟩
    (fun g _ hk o ho _ => by change o = C.eval Xr g at ho; subst ho; exact hcom g hk) (fun g _ b hb => hκ g b hb)
    (ns := [n]) (by simp) .all [] ⟨"", 0, []⟩ hsub

end Unbound

namespace Unbound

def notCircuit : Verity.Boolean.Circuit where
  N := 2
  op g := if g = 0 then .input else .not 0
  topo := by decide
  outputs := {1}

def notPartition : FlockSoundness.Audit.Partition notCircuit 1 where
  unit g := if g.val = 0 then none else some 0
  input_iff := by decide

end Unbound

open Unbound in
theorem unbound_subset_faulty_nontrivial (H : Claims.Msg → Claims.Msg) (roots : List Claims.Msg) {ns : List ℕ}
    (hns : ns.sum = 1) (r : OneStage) (desc : Claims.Msg) (q : Query) :
    let Cm : CommitmentScheme Bool := ⟨fun _ _ => .ret true, fun _ _ _ => []⟩
    let s := stmtOf notPartition (fun _ => none) r desc q roots
    let w := witDep notCircuit Cm H roots (fun _ => false)
    ¬ (ResourceAccounting.unbound (Verity.Partition.ofAudit notPartition) (fun _ _ => False)
        (Verity.Circuit.ofBits (openedOf notCircuit (fun _ => none) (w.val H Cm s)))).image Fin.val ⊆
      faulty H Cm s (partOf notPartition (fun _ => none) ns) w := by
  intro Cm s w hsub
  have hun : (0 : Fin 1) ∈ ResourceAccounting.unbound (Verity.Partition.ofAudit notPartition) (fun _ _ => False)
      (Verity.Circuit.ofBits (openedOf notCircuit (fun _ => none) (w.val H Cm s))) := by
    classical
    simp only [ResourceAccounting.unbound, Finset.mem_filter, Finset.mem_univ, true_and]
    refine Or.inr ⟨⟨1, by decide⟩, ?_, by simp⟩
    simp only [Verity.Partition.io, Finset.mem_filter]
    exact ⟨(Verity.Partition.mem_committed_ofAudit notPartition _).2
      (notPartition.mem_committed_of_output (by decide)), Or.inl rfl⟩
  exact faultyRightC notPartition (fun g : Fin notCircuit.N => g) (fun (b : Bool) (_ : Fin notCircuit.N) => b)
    (id : Bool → Bool) (id : Bool → Bool) (notCircuit.eval fun _ => false) (fun _ _ => True) H Cm roots
    (fun _ => false) (fun _ => none) 0 (fun _ _ h => by simp at h)
    ⟨notCircuit.eval fun _ => false, fun _ _ => ⟨rfl, rfl, trivial⟩⟩
    (fun _ _ _ _ _ _ => ⟨[], rfl⟩) (fun _ _ _ h => by simp at h) hns r desc q
    (hsub (Finset.mem_image_of_mem Fin.val hun))

end Flock.Bridge
