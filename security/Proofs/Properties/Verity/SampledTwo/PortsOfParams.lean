module

public import Proofs.Properties.Verity.SampledTwo.AcceptsTop
public import Proofs.Flock.SpotFlock.ParamsWf

/-!
# A play's registrations list each port once, where its parameters' tree is well formed

`PortsDistinctAt` is a premise of the form's accept joined to the run's (`AcceptsTop`): at each accepted play, the
run of record's input lists no port twice, at its boundary and at its interior. The input's parameters are the form's
boundary's and its interior's the form's interior's (`inputOf_first`, `inputOf_interior`), and each lists no port
twice where the tree it was read from is well formed (`SpotFlock.ParamsWf.ports_nodup_form`). So it holds at every
play whose case has an input, accepted or not.

## What this does not show

That the tree `Core.canonicalJson` returns is well formed: the hypothesis `hcj`, as in `SpotFlock.ParamsWf`.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims.Json (Wf)
open Properties.Spot.SecurityProofs (bind_eq_ok)

/-- **A case's input lists each port once**, at its boundary and at its interior, where `Core.canonicalJson`
returns well-formed trees. -/
theorem ports_of_inputOfCase (hcj : ∀ b j, Core.canonicalJson b = .ok j → Wf j) {k : Case} {i : Spot.Run.Input}
    {it : Spot.Run.Interior} (hi : inputOfCase k = some i) (hit : i.interior = some it) :
    (i.params.verifier.ports.map (·.1)).toList.Nodup ∧ (it.params.verifier.ports.map (·.1)).toList.Nodup := by
  unfold inputOfCase at hi
  split at hi
  · rename_i i' h
    have h0 : i' = i := Option.some.inj hi
    obtain ⟨p, hp, h⟩ := bind_eq_ok.1 h
    obtain ⟨c, -, h⟩ := bind_eq_ok.1 h
    obtain ⟨e, -, h⟩ := bind_eq_ok.1 h
    obtain ⟨⟨c₂, ξ₂, r, i''⟩, hin, h⟩ := bind_eq_ok.1 h
    have h1 : i'' = i' := Except.ok.inj h
    obtain ⟨-, -, -, -, vs, rc, j₁, j₂, -, hieq, -, -⟩ := VStar.Recursive.SampledTwo.inputAt_ok hin
    have hI := h0.symm.trans (h1.symm.trans hieq)
    obtain ⟨j, -, hj⟩ := bind_eq_ok.1 hp
    obtain ⟨hb, hn⟩ := SpotFlock.ParamsWf.ports_nodup_form hcj hj
    obtain ⟨it', hit', hpar, -⟩ := _root_.VStar.Recursive.SampledTwo.inputOf_interior p { c with record := j₁ }
      e.source ξ₂ { e with record := j₂, stmts := VStar.Recursive.SampledTwo.stmtsOwn p e k.blobs } rc vs
    subst hI
    have hii : it = it' := Option.some.inj (hit.symm.trans hit')
    rw [hii, hpar, (_root_.VStar.Recursive.SampledTwo.inputOf_first _ _ _ _ _ _ _).2.1]
    exact ⟨hb, hn⟩
  · cases hi

/-- **`PortsDistinctAt` holds** where `Core.canonicalJson` returns well-formed trees. -/
theorem portsDistinctAt_of (hcj : ∀ b j, Core.canonicalJson b = .ok j → Wf j) (C : CaseOf) : PortsDistinctAt C := by
  intro o i it _ hi hit
  unfold inputOn at hi
  cases hk : C o with
  | none =>
    rw [hk] at hi
    cases hi
  | some k =>
    rw [hk, Option.bind_some] at hi
    exact ports_of_inputOfCase hcj hi hit

end Verity.SampledTwo
