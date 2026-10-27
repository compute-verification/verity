import Definitions.Accounting.JointLive
import Proofs.Flock.Bridge.CircuitOf

/-!
# A unit with a committed value unopened is faulty

`card_wrong_le_faulty` bounds the units wrong in the values a witness opens by the faulty ones. Sampled
proofs' claim (`Accounting.unbound`) also counts a unit one of whose committed values (an input it reads from outside,
or a value it computes and another unit reads or the circuit returns) the commitment doesn't open; these are faulty
too, since a proof unit that is correct on its committed values reads and returns them all.

* `opened_of_progCorrect`: where a unit's lowered proof unit is correct on the values `val` opens, every committed wire
  the unit computes or reads that isn't a constant is opened.
* `unopened_mem_faulty`: so a unit with such a wire unopened in a witness is faulty in the sampling statement.
-/

namespace Flock.Bridge

open Claims.Sampling

variable {C : FlockSoundness.Audit.Circuit}

/-- **A correct unit opens its committed wires**: where the unit `u`'s one lowered proof unit is correct in
`progOf C κ` on the values `val` (and any interior `ι`), every committed wire `g` the unit computes or one of its gates
reads, other than a constant, is opened at its position. -/
theorem opened_of_progCorrect {κ : Fin C.N → Option Bool} {n : ℕ} (P : FlockSoundness.Audit.Partition C n)
    (u : Fin n) (val : ℕ → Option Bool) (ι : ℕ → Bool)
    (h : (progOf C κ).Correct (RU.ref (progOf C κ) [unitGates P u] val ι) (unitGates P u))
    {g : Fin C.N} (hg : g ∈ P.committed)
    (hio : P.unit g = some u ∨ ∃ h' : Fin C.N, P.unit h' = some u ∧ g ∈ (C.op h').args)
    (hc : ∀ b, (progOf C κ).gates[posOf C g]? ≠ some (.const b)) :
    val (posOf C g) ≠ none := by
  have hgates : RU.gates [unitGates P u] = unitGates P u := by simp [RU.gates]
  by_cases hgu : P.unit g = some u
  · have hB := boundary_posOf (κ := κ) hg hgu
    obtain ⟨b, hb, -⟩ := (h (posOf C g) (posOf_mem_unitGates.2 hgu)).2 hB
    have href : RU.ref (progOf C κ) [unitGates P u] val ι (posOf C g) = val (posOf C g) := by
      simp [RU.ref, hgates, hB]
    rw [href] at hb
    rw [hb]
    exact Option.some_ne_none b
  · obtain ⟨h', hu', hgh⟩ := hio.resolve_left hgu
    obtain ⟨j, hj, x, hx, hr⟩ := exists_lower_reads (κ := κ) hgh
    have hgx : (progOf C κ).gates[off C h'.val + j]? = some x := by rw [gates_off h' hj]; exact hx
    have hp : off C h'.val + j ∈ unitGates P u := mem_unitGates.2 ⟨h', hu', j, hj, rfl⟩
    have hna : posOf C g ∉ unitGates P u := fun hm => hgu (posOf_mem_unitGates.1 hm)
    have hne := (h _ hp).1 x hgx (posOf C g) hr hna
    have href : RU.ref (progOf C κ) [unitGates P u] val ι (posOf C g) = val (posOf C g) := by
      simp [RU.ref, hgates, hna]
    rw [Claims.Sampling.Boundary.read_eq _ hc, href] at hne
    exact hne

/-- **A unit with a committed wire unopened is faulty** in the sampling statement of `progOf C κ`: the wire, other
than a constant, is one the unit computes or one of its gates reads, and the witness opens no value at its position. -/
theorem unopened_mem_faulty {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (κ : Fin C.N → Option Bool) (kd : ℕ)
    (desc : Claims.Msg) (q : Query) (roots : List Claims.Msg) (H : Claims.Msg → Claims.Msg) (Cm : Commitment Bool)
    (w : Wit) {u : Fin n} {g : Fin C.N} (hg : g ∈ P.committed)
    (hio : P.unit g = some u ∨ ∃ h' : Fin C.N, P.unit h' = some u ∧ g ∈ (C.op h').args)
    (hc : ∀ b, (progOf C κ).gates[posOf C g]? ≠ some (.const b))
    (hv : w.val H Cm (stmtOf P κ kd desc q roots) (posOf C g) = none) :
    u.val ∈ faulty H Cm (stmtOf P κ kd desc q roots) (partOf P) w := by
  classical
  have hlt := u.isLt
  refine Finset.mem_filter.2 ⟨Finset.mem_range.2 (by rw [rus_length_partOf]; omega), ?_⟩
  rintro ⟨r, hr, ι, hι⟩
  rw [rus_partOf_getElem? P u] at hr
  cases Option.some.inj hr
  exact opened_of_progCorrect P u _ ι (hι _ (List.mem_singleton_self _)) hg hio hc hv

/-- **The unbound units are at most the faulty ones**: in the values a witness opens (`openedOf`), under any openings
`opens` of core's circuit that hold wherever the witness opens a value other than a constant (`hopen`) and at the
constants (`hconst`), every unit unbound (`Accounting.unbound`: wrong, or with a committed value `opens` doesn't
open) is faulty in the sampling statement. -/
theorem card_unbound_le_faulty {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (κ : Fin C.N → Option Bool) (kd : ℕ)
    (desc : Claims.Msg) (q : Query) (roots : List Claims.Msg) (H : Claims.Msg → Claims.Msg) (Cm : Commitment Bool)
    (w : Wit) (opens : (g : Fin (Verity.Circuit.ofAudit C).N) → BitVec ((Verity.Circuit.ofAudit C).w g) → Prop)
    (hopen : ∀ (g : Fin C.N) (b : Bool), (∀ b', (progOf C κ).gates[posOf C g]? ≠ some (.const b')) →
      w.val H Cm (stmtOf P κ kd desc q roots) (posOf C g) = some b → opens g (BitVec.ofBool b))
    (hconst : ∀ (g : Fin C.N) (b : Bool), (progOf C κ).gates[posOf C g]? = some (.const b) →
      opens g (Verity.Circuit.ofBits (openedOf C κ (w.val H Cm (stmtOf P κ kd desc q roots))) g)) :
    (Accounting.unbound (Verity.Partition.ofAudit P) opens
      (Verity.Circuit.ofBits (openedOf C κ (w.val H Cm (stmtOf P κ kd desc q roots))))).card ≤
      ((faulty H Cm (stmtOf P κ kd desc q roots) (partOf P) w).filter (· < n)).card := by
  classical
  refine Finset.card_le_card_of_injOn (fun u : Fin n => u.val) (fun u hu => ?_) (fun a _ b _ h => Fin.ext h)
  rw [Finset.mem_coe, Accounting.unbound, Finset.mem_filter] at hu
  have hu' := hu.2
  refine Finset.mem_coe.2 (Finset.mem_filter.2 ⟨?_, u.isLt⟩)
  rcases hu' with hw | ⟨g, hg, hno⟩
  · have hmem : u ∈ (Verity.Partition.ofAudit P).wrong
        (Verity.Circuit.ofBits (openedOf C κ (w.val H Cm (stmtOf P κ kd desc q roots)))) := by
      rw [Verity.Partition.mem_wrong]
      exact hw
    rw [Verity.Partition.wrong_ofAudit] at hmem
    exact wrong_subset_faulty P κ kd desc q roots H Cm w hmem
  · obtain ⟨hgc, hio⟩ := Finset.mem_filter.1 hg
    have hgc' : g ∈ P.committed := (Verity.Partition.mem_committed_ofAudit P g).1 hgc
    have hio' : P.unit g = some u ∨ ∃ h' : Fin C.N, P.unit h' = some u ∧ g ∈ (C.op h').args := by
      rcases hio with h1 | ⟨h', hu1, hr⟩
      · exact Or.inl h1
      · rw [Verity.Circuit.ofAudit_reads, List.mem_toFinset] at hr
        exact Or.inr ⟨h', hu1, hr⟩
    by_cases hc : ∃ b, (progOf C κ).gates[posOf C g]? = some (.const b)
    · obtain ⟨b, hb⟩ := hc
      exact absurd (hconst g b hb) hno
    · have hc' : ∀ b, (progOf C κ).gates[posOf C g]? ≠ some (.const b) := fun b hb => hc ⟨b, hb⟩
      cases hv : w.val H Cm (stmtOf P κ kd desc q roots) (posOf C g) with
      | none => exact unopened_mem_faulty P κ kd desc q roots H Cm w hgc' hio' hc' hv
      | some b =>
        refine absurd ?_ hno
        have hX : Verity.Circuit.ofBits (openedOf C κ (w.val H Cm (stmtOf P κ kd desc q roots))) g =
            BitVec.ofBool b := by
          show BitVec.ofBool (openedOf C κ _ g) = _
          simp only [openedOf_of_not_const _ hc', hv, Option.getD_some]
        rw [hX]
        exact hopen g b hc' hv

end Flock.Bridge
