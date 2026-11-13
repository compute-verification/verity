module

public import Proofs.Flock.SpotFlock.Honest.Roots
public import Proofs.Flock.SpotFlock.WiringOk

/-!
# The registered wiring's decided conditions at the one-template instance

At the instance every port of `R` is linked, there is no wire and the interior registration registers no value, so
each condition the second stage's check decides of the registered wiring holds where a link names the bit's port:

* no written bit the boundary record holds is at an unlinked port (`returnedWhole_held`), from `RefPorts.written`;
* no wire names a written bit's port (`writtenUnwired_held`): there is no wire;
* each reference read is at a linked port or reads a port of the interior registration (`refReadsOwn_held`), from
  `RefReadsNamed`;
* the interior registration registers no value (`values_held`).

With `wiresFirst_held`, `sourcesNamed_held` and `wiresMove_held` (`Instance`), these are the conditions the base
decides; the statement is true of the auditor's copies `held R B desc q c₀`.
-/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy

variable (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code)

/-- **Each of `R`'s reference registered reads is at one of its ports, or reads a port of the interior
registration.** -/
def RefReadsNamed : Prop :=
  ∀ pe ∈ Spot.entries (Spot.getD R.reads "ports"), R.ports.contains pe.1 = true ∨
    ∃ v, Spot.getD pe.2 "value" = Lean.Json.str v ∧ B.interiorPorts.contains v = true

/-- **Every template's name is `R`'s**: there is one template. -/
theorem name_held {t : ℕ} (ht : t < (held R B desc q c₀).ms.length) :
    (held R B desc q c₀).names.getD t "" = R.name := by
  have h0 : t = 0 := by
    rw [ms_held] at ht
    simpa [ms2] using ht
  subst h0
  rfl

/-- **No written bit the boundary record holds is at a port no link names**: every written bit is at one of `R`'s
ports, each linked. -/
theorem returnedWhole_held (hR : RefPorts R) : returnedWhole (held R B desc q c₀) = true := by
  unfold returnedWhole
  rw [List.all_eq_true]
  rintro ⟨y, t, r⟩ hy
  obtain ⟨ht, -⟩ := mem_writtenBits hy
  have hrow : R.ports.contains r.row = true := hR.written _ hy
  dsimp only
  rw [name_held R B desc q c₀ ht, heldLink_held R B desc q c₀ _ hrow, Bool.true_or]

/-- **No wire names a written bit's port**: there is no wire. -/
theorem writtenUnwired_held : writtenUnwired (held R B desc q c₀) = true := by
  unfold writtenUnwired
  rw [List.all_eq_true]
  intro y _
  rw [heldWire_held, Option.isNone_none, Bool.or_true]

/-- **Each reference read at a port no link or wire names reads a port of the interior registration.** -/
theorem refReadsOwn_held (hO : RefReadsNamed R B) : refReadsOwn (held R B desc q c₀) = true := by
  unfold refReadsOwn
  rw [List.all_eq_true]
  intro t ht
  rw [Array.all_eq_true_iff_forall_mem]
  intro pe hpe
  rw [name_held R B desc q c₀ (List.mem_range.1 ht)]
  rcases hO pe hpe with hp | ⟨v, hv, hc⟩
  · simp only [heldLink_held R B desc q c₀ _ hp, Bool.true_or]
  · simp only [heldWire_held, Option.isSome_none, Bool.or_false, hv]
    exact Bool.or_eq_true_iff.2 (Or.inr hc)

/-- **The interior registration registers no value.** -/
theorem values_held : (held R B desc q c₀).interiorNames.values = [] := rfl

end SpotFlock.Honest
