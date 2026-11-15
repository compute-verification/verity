module

public import Proofs.Flock.SpotFlock.Honest.Instance

/-!
# The registered wiring's decided conditions at the one-template instance

At the instance every port of `R` is linked, there is no wire and the interior registration registers no value, so
each clause of the registered wiring's check (`wiringOk`) holds where a link names the row or is a fact of `R`'s
circuit:

* each name is one template's (`namesOnce_held`): there is one template;
* every row the template's units read source bits at is linked (`sourceRowsNamed_held`, from `Ref.named`);
* every wired source bit has a gate to go to (`wiresMoveRows_held`), each wire's reader port is like its writer's
  (`wiresAlike_held`), and no wire names a row the template's units write (`writtenRowsUnwired_held`): there is no
  wire;
* every row the template's units write is linked, so none needs a wire to read it (`writtenRowsRead_held`), where
  each is one of `R`'s ports (`RefWritten`);
* each reference read is at a linked port or reads a port of the interior registration (`refReadsOwn_held`), from
  `RefReadsNamed`;
* the interior registration registers no value (`values_held`);
* the circuit names its ports once (`portsOnce_held`) and every returned output bit of its unit sits on a computed
  row (`outsOk_held`), where `R`'s statement's circuit does.

With `wiresFirst_held` (`Instance`), the check passes (`wiringOk_held`). Where `R`'s circuit is also in the scope
Flock's hidden-output proof covers (`membersScoped_held`), the design evaluates (`two_of_ref`). The statements are
true of the auditor's copies `held R B desc q c₀`.
-/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy

variable (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code)

/-- **Every row `R`'s units write is one of its ports.** -/
def RefWritten : Prop := ∀ row ∈ writtenRows (stmtOf R.x), R.ports.contains row = true

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

/-- **Every row the template's units read source bits at is a port a link names**: each is one of `R`'s ports
(`Ref.named`). -/
theorem sourceRowsNamed_held : sourceRowsNamed (held R B desc q c₀) = true := by
  unfold sourceRowsNamed
  rw [List.all_eq_true]
  intro t ht
  rw [List.all_eq_true]
  intro row hrow
  have hn : R.ports.contains row = true := List.all_eq_true.1 R.named row hrow
  rw [name_held R B desc q c₀ (List.mem_range.1 ht), heldLink_held R B desc q c₀ _ hn, Bool.true_or]

theorem rowGoes_held (t : ℕ) (row : String) : rowGoes (held R B desc q c₀) t row = true := by
  unfold rowGoes
  split <;> simp_all [heldWire_held]

theorem bitGoes_held (t : ℕ) (row : String) (bit : ℕ) : bitGoes (held R B desc q c₀) t row bit = true := by
  unfold bitGoes
  split <;> simp_all [heldWire_held]

/-- **Every wired source bit has a gate to go to**: there is no wire. -/
theorem wiresMoveRows_held : wiresMoveRows (held R B desc q c₀) = true := by
  unfold wiresMoveRows
  rw [List.all_eq_true]
  intro t _
  exact Bool.and_eq_true_iff.2 ⟨List.all_eq_true.2 fun row _ => rowGoes_held R B desc q c₀ t row,
    List.all_eq_true.2 fun rb _ => bitGoes_held R B desc q c₀ t rb.1 rb.2⟩

/-- **No wire names a row the template's units write**: there is no wire. -/
theorem writtenRowsUnwired_held : writtenRowsUnwired (held R B desc q c₀) = true := by
  unfold writtenRowsUnwired
  rw [List.all_eq_true]
  intro t _
  rw [List.all_eq_true]
  intro row _
  rw [heldWire_held, Option.isNone_none, Bool.or_true]

/-- **Every row the template's units write is linked**, so none needs a wire to read it. -/
theorem writtenRowsRead_held (hR : RefWritten R) : writtenRowsRead (held R B desc q c₀) = true := by
  unfold writtenRowsRead
  rw [List.all_eq_true]
  intro t ht
  rw [List.all_eq_true]
  intro row hrow
  have hn : R.ports.contains row = true := hR row hrow
  rw [name_held R B desc q c₀ (List.mem_range.1 ht), heldLink_held R B desc q c₀ _ hn, Bool.true_or]

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

/-- **The template's circuit names its ports once**, where `R`'s statement's circuit does. -/
theorem portsOnce_held (hP : ((stmtOf R.x).ports.toList.map (·.name)).Nodup) :
    portsOnce (held R B desc q c₀) = true := by
  unfold portsOnce
  rw [List.all_eq_true]
  intro t _
  exact decide_eq_true hP

/-- **Every returned output bit of the template's unit sits on a computed row**, where it does in `R`'s statement's
circuit. -/
theorem outsOk_held (hC : outsComputed (stmtOf R.x) = true) : outsOk (held R B desc q c₀) = true := by
  unfold outsOk
  rw [List.all_eq_true]
  intro t _
  exact hC

/-- **The registered wiring passes its check at the instance** (`wiringOk`), each clause by its own lemma. -/
theorem wiringOk_held (hR : RefWritten R) (hO : RefReadsNamed R B)
    (hP : ((stmtOf R.x).ports.toList.map (·.name)).Nodup) (hC : outsComputed (stmtOf R.x) = true) :
    wiringOk (held R B desc q c₀) = true :=
  (wiringOk_iff _).2 ⟨namesOnce_held R B desc q c₀, wiresFirst_held R B desc q c₀,
    sourceRowsNamed_held R B desc q c₀, wiresMoveRows_held R B desc q c₀, wiresAlike_held R B desc q c₀,
    writtenRowsUnwired_held R B desc q c₀, writtenRowsRead_held R B desc q c₀ hR,
    refReadsOwn_held R B desc q c₀ hO, values_held R B desc q c₀, portsOnce_held R B desc q c₀ hP,
    outsOk_held R B desc q c₀ hC⟩

/-- **The template's circuit is in the scope Flock's hidden-output proof covers** (`membersScoped`), where `R`'s
statement's circuit is. -/
theorem membersScoped_held (hS : _root_.Flock.ProvedScope.scopeOkH (stmtOf R.x) = true) :
    membersScoped (held R B desc q c₀) = true := by
  unfold membersScoped
  rw [List.all_eq_true]
  intro t _
  exact hS

/-- **The design evaluates at the instance**, where `R`'s circuit is in the scope Flock's hidden-output proof
covers, each row its units write is one of its ports, each of its reference reads is at one of its ports or reads a
port of the interior registration, it names its ports once and every returned output bit of its unit sits on a
computed row. -/
theorem two_of_ref (hS : _root_.Flock.ProvedScope.scopeOkH (stmtOf R.x) = true) (hR : RefWritten R)
    (hO : RefReadsNamed R B) (hP : ((stmtOf R.x).ports.toList.map (·.name)).Nodup)
    (hC : outsComputed (stmtOf R.x) = true) (s : AuditStatement) :
    two (held R B desc q c₀) s =
      some ⟨Properties.Spot.filesReplay (ms2 R.x) #[2] (oneLayout B.gs), design1, [1]⟩ :=
  two_held R B desc q c₀ (membersScoped_held R B desc q c₀ hS) (wiringOk_held R B desc q c₀ hR hO hP hC) s

end SpotFlock.Honest
