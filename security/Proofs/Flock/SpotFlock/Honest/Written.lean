module

public import Proofs.Flock.SpotFlock.Honest.Wiring

/-!
# What the wiring check gives of the instance's ports

`RefPorts.written` asks that every written bit's row is one of `R`'s ports. The registered wiring's check decides it:
`writtenRead` asks a link or a wire at each written bit's port, the instance has no wire, and each of its links is at
one of `R`'s ports (`port_of_heldLink`). So where the check passes the rows are `R`'s ports
(`written_of_writtenRead`), and so where the design evaluates (`written_of_two`). There `RefPorts` asks only that no
port of `R` is named like an opening row (`refPorts_of_two`). The statements are true of the auditor's copies
`held R B desc q c₀`.
-/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy

variable (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code)

/-- **A link the instance holds is at one of `R`'s ports**, whichever template it is asked for. -/
theorem port_of_heldLink {n p : String} (hl : (heldLink (held R B desc q c₀) n p).isSome = true) :
    R.ports.contains p = true := by
  have hl' : ((R.ports.map fun p => (⟨R.name, p, p⟩ : Spot.SecondStage.Link)).find?
      fun l => l.template == n && l.port == p).isSome = true := hl
  obtain ⟨l, hmem, hlp⟩ := Array.find?_isSome.1 hl'
  obtain ⟨p', hp', rfl⟩ := Array.mem_map.1 hmem
  have hlp' : (R.name == n && p' == p) = true := hlp
  have e : p' = p := eq_of_beq (Bool.and_eq_true_iff.1 hlp').2
  rw [← e]
  exact Array.contains_eq_true_of_mem hp'

/-- **Every written bit's row is one of `R`'s ports where the wiring's check reads each written port**
(`writtenRead`): the check asks a link or a wire there, and the instance has no wire. -/
theorem written_of_writtenRead (hw : writtenRead (held R B desc q c₀) = true) :
    ∀ y ∈ writtenBits (ms2 R.x), R.ports.contains y.2.2.row = true := by
  intro y hy
  unfold writtenRead at hw
  have h1 := List.all_eq_true.1 hw y hy
  rcases Bool.or_eq_true_iff.1 h1 with hl | hany
  · exact port_of_heldLink R B desc q c₀ hl
  · obtain ⟨i, hi, -⟩ := Array.any_eq_true.1 hany
    have hi' : i < 0 := hi
    exact absurd hi' (Nat.not_lt_zero i)

/-- **Every written bit's row is one of `R`'s ports where the design evaluates**: the registered wiring passed its
check there. -/
theorem written_of_two {s : AuditStatement} {d : Design.Two} (h2 : two (held R B desc q c₀) s = some d) :
    ∀ y ∈ writtenBits (ms2 R.x), R.ports.contains y.2.2.row = true := by
  obtain ⟨-, -, -, -, -, -, -, hw, -, -⟩ := wiring_of_two _ h2
  exact written_of_writtenRead R B desc q c₀ hw

/-- **What the roots ask of `R`'s ports, where the design evaluates**: that none is named like an opening row. -/
theorem refPorts_of_two (hP : ∀ p ∈ R.ports, Spot.SecondStage.OPENING_ROW.contains p = false)
    {s : AuditStatement} {d : Design.Two} (h2 : two (held R B desc q c₀) s = some d) : RefPorts R :=
  ⟨written_of_two R B desc q c₀ h2, hP⟩

end SpotFlock.Honest
