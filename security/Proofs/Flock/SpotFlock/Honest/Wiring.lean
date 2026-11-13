module

public import Proofs.Flock.SpotFlock.Honest.Instance
public import Proofs.Flock.SpotFlock.WiringOk

/-!
# The registered wiring's decided conditions at the one-template instance

At the instance every port of `R` is linked, there is no wire and the interior registration registers no value, so
each clause of the registered wiring's check (`wiringOk`) holds where a link names the bit's port:

* each name is one template's (`namesOnce_held`): there is one template;
* each wire's reader port is like its writer's (`wiresAlike_held`), and no wire names a written bit's port
  (`writtenUnwired_held`): there is no wire;
* no written bit the boundary record holds is at an unlinked port (`returnedWhole_held`), and none needs a wire to
  read it (`writtenRead_held`), where every bit `R` writes is at one of its ports (`RefWritten`);
* each reference read is at a linked port or reads a port of the interior registration (`refReadsOwn_held`), from
  `RefReadsNamed`;
* the interior registration registers no value (`values_held`).

With `wiresFirst_held`, `sourcesNamed_held` and `wiresMove_held` (`Instance`), the check passes (`wiringOk_held`).
Where `R`'s circuit is also in the scope Flock's hidden-output proof covers (`membersScoped_held`), the design
evaluates (`two_of_ref`). The statements are true of the auditor's copies `held R B desc q c₀`.
-/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy

variable (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code)

/-- **Every bit `R` writes is at one of its ports.** -/
def RefWritten : Prop := ∀ y ∈ writtenBits (ms2 R.x), R.ports.contains y.2.2.row = true

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
theorem returnedWhole_held (hR : RefWritten R) : returnedWhole (held R B desc q c₀) = true := by
  unfold returnedWhole
  rw [List.all_eq_true]
  rintro ⟨y, t, r⟩ hy
  obtain ⟨ht, -⟩ := mem_writtenBits hy
  have hrow : R.ports.contains r.row = true := hR _ hy
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

/-- **Each name is one template's**: there is one template. -/
theorem namesOnce_held : namesOnce (held R B desc q c₀) = true := by
  unfold namesOnce
  rw [ms_held]
  show decide ([R.name].Nodup) = true
  simp

/-- **Each wire's reader port is like its writer's output port**: there is no wire. -/
theorem wiresAlike_held : wiresAlike (held R B desc q c₀) = true := by
  simp [wiresAlike, held, held₀]

/-- **Every port a member writes is linked**, so none needs a wire to read it. -/
theorem writtenRead_held (hR : RefWritten R) : writtenRead (held R B desc q c₀) = true := by
  unfold writtenRead
  rw [List.all_eq_true]
  rintro ⟨y, t, r⟩ hy
  obtain ⟨ht, -⟩ := mem_writtenBits hy
  have hrow : R.ports.contains r.row = true := hR _ hy
  dsimp only
  rw [name_held R B desc q c₀ ht, heldLink_held R B desc q c₀ _ hrow, Bool.true_or]

/-- **The registered wiring passes its check at the instance** (`wiringOk`), each clause by its own lemma. -/
theorem wiringOk_held (hR : RefWritten R) (hO : RefReadsNamed R B) : wiringOk (held R B desc q c₀) = true :=
  (wiringOk_iff _).2 ⟨namesOnce_held R B desc q c₀, wiresFirst_held R B desc q c₀, sourcesNamed_held R B desc q c₀,
    wiresMove_held R B desc q c₀, wiresAlike_held R B desc q c₀, returnedWhole_held R B desc q c₀ hR,
    writtenUnwired_held R B desc q c₀, writtenRead_held R B desc q c₀ hR, refReadsOwn_held R B desc q c₀ hO,
    values_held R B desc q c₀⟩

/-- **The template's circuit is in the scope Flock's hidden-output proof covers** (`membersScoped`), where `R`'s
statement's circuit is. -/
theorem membersScoped_held (hS : _root_.Flock.ProvedScope.scopeOkH (stmtOf R.x) = true) :
    membersScoped (held R B desc q c₀) = true := by
  unfold membersScoped
  rw [List.all_eq_true]
  intro t _
  exact hS

/-- **The design evaluates at the instance**, where `R`'s circuit is in the scope Flock's hidden-output proof
covers, each bit it writes is at one of its ports and each of its reference reads is at one of its ports or reads a
port of the interior registration. -/
theorem two_of_ref (hS : _root_.Flock.ProvedScope.scopeOkH (stmtOf R.x) = true) (hR : RefWritten R)
    (hO : RefReadsNamed R B) (s : AuditStatement) :
    two (held R B desc q c₀) s =
      some ⟨Properties.Spot.filesReplay (ms2 R.x) #[2] (oneLayout B.gs), design1, [1]⟩ :=
  two_held R B desc q c₀ (membersScoped_held R B desc q c₀ hS) (wiringOk_held R B desc q c₀ hR hO) s

end SpotFlock.Honest
