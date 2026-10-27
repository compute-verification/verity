import Proofs.Flock.Bridge.CircuitOf

/-!
# A unit with a committed value unopened is faulty

`card_wrong_le_faulty` (ONE-7) bounds the units wrong in the values a witness opens by the faulty ones. Sampled
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

end Flock.Bridge
