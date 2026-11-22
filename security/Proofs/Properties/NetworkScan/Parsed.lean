module

public import Properties.NetworkScan.Flattens
public import Properties.Spot.Members
public import Proofs.Properties.Verity.Sampled.Record
public import Proofs.Properties.Verity.Sampled.Unbound

/-!
# The deployed sampled verifiers' statements are no Program flattened whose gate 0 is an input

`Properties.NetworkScan.StmtFlattens` asks of a compiled sampled verifier that each statement it parses is its
descriptor flattened (`Flattens`). The deployed parsers build a statement's gates from its circuit files, gate 0 the
constant:

* SPOT[Flock]'s (`SpotFlock.parse`, `prog := bitsProgs h.ms`): over at least one member, gate 0 is `.const true`
  (`bitsProgs_gate_zero`), the first member's key program's constant (`Spot.Bits.consts`).
* One-stage v2's record (`Verity.Sampled.Record.code`, whose statement is `stmtA`, `prog := progOf A.Cc A.κ`): gate 0 is
  `.const true` (`stmtA_gate_zero`), the key program's constant wire, an input gate the constants fix.

A constant gate is no input gate, so neither statement is the flattening of a Program whose gate 0 is an input gate
(`not_flattens_of_const_zero`, and so `not_flattens_bitsProgs` and `not_flattens_stmtA`). Nothing here says which
Programs have an input at gate 0: the network check's `scanOf` checks no property of the root's node 0.
-/

@[expose] public section

namespace Properties.NetworkScan

/-- **A circuit whose gate 0 is a constant is no flattening of a Program whose gate 0 is an input gate**: `Flattens`
makes gate 0 an input gate exactly where its primitive is an input node. -/
theorem not_flattens_of_const_zero {P : Flock.Program.Prog} {C : Claims.Sampling.Prog} {b : Bool}
    (hC : C.gates[0]? = some (.const b))
    (hP : ∀ p ops sc, P.gate 0 = .ok (p, ops, sc) → Flock.Program.isInputId (P.defs[p]!).key = true) :
    ¬ Flattens P C := by
  rintro ⟨-, hg, -⟩
  have h0 : 0 < C.gates.length := by
    rcases h : C.gates with _ | ⟨x, xs⟩
    · rw [h] at hC; simp at hC
    · simp
  obtain ⟨p, ops, sc, hgate, hin, -⟩ := hg 0 h0
  have := hin.2 (hP p ops sc hgate)
  rw [hC] at this
  cases this

theorem keyProg_inputs_pos (c : Flock.Circuit) (n : ℕ) : 0 < (Spot.Bits.keyProg c n).inputs := by
  unfold Spot.Bits.keyProg
  exact Nat.lt_of_lt_of_le (by decide) (Nat.le_add_right 2 _)

theorem lower_zero (c : Flock.Circuit) (n : ℕ) :
    Spot.Bits.lower (Spot.Bits.keyProg c n) (Spot.Bits.consts c n) 0 = [.const true] := by
  have hr : (Spot.Bits.keyProg c n).row? 0 = none := by
    simp only [Spot.Bits.KeyProg.row?, keyProg_inputs_pos c n, ite_true]
  simp only [Spot.Bits.lower, hr, Spot.Bits.consts, beq_self_eq_true, ite_true]

/-- **A member's gate 0 is the constant**: the key program's gate 0 is an input gate its constants set. -/
theorem memberProg_gate_zero (c : Flock.Circuit) (n : ℕ) :
    (Properties.Spot.memberProg c n).gates[0]? = some (.const true) := by
  obtain ⟨k, hk⟩ : ∃ k, (Spot.Bits.keyProg c n).size = k + 1 :=
    ⟨(Spot.Bits.keyProg c n).size - 1, by
      have := keyProg_inputs_pos c n
      unfold Spot.Bits.KeyProg.size; omega⟩
  simp only [Properties.Spot.memberProg, Properties.Spot.bitsProg, Spot.Bits.prog, hk, List.range_succ_eq_map,
    List.flatMap_cons, lower_zero, List.map_cons, List.cons_append, List.nil_append, List.getElem?_cons_zero,
    Properties.Spot.bitsGate]

/-- **SPOT[Flock]'s statement's gate 0 is the constant**: over at least one member, the members' programs side by side
(`bitsProgs`, `SpotFlock.parse`'s `prog`) start with the first member's, whose gate 0 is the constant. -/
theorem bitsProgs_gate_zero (ci : Flock.Circuit × Array ℕ) (ms : Spot.Bits.Members) :
    (Properties.Spot.bitsProgs (ci :: ms)).gates[0]? = some (.const true) := by
  have h := memberProg_gate_zero ci.1 ci.2.size
  simp only [Properties.Spot.bitsProgs, List.map_cons, Properties.Spot.sideBySide]
  cases hg : (Properties.Spot.memberProg ci.1 ci.2.size).gates with
  | nil => rw [hg] at h; simp at h
  | cons x xs =>
    rw [hg] at h
    simp only [List.cons_append, List.getElem?_cons_zero] at h ⊢
    exact h

/-- **SPOT[Flock]'s statement is no flattening of a Program whose gate 0 is an input gate**: at its parse
(`SpotFlock.parse`, `prog := bitsProgs h.ms`), over at least one member. -/
theorem not_flattens_bitsProgs {P : Flock.Program.Prog} (ci : Flock.Circuit × Array ℕ) (ms : Spot.Bits.Members)
    (hP : ∀ p ops sc, P.gate 0 = .ok (p, ops, sc) → Flock.Program.isInputId (P.defs[p]!).key = true) :
    ¬ Flattens P (Properties.Spot.bitsProgs (ci :: ms)) :=
  not_flattens_of_const_zero (bitsProgs_gate_zero ci ms) hP

/-- **Bridge's lowering puts a fixed input gate 0 at gate 0 as its constant**: gate 0 an input gate the constants `κ`
fix to `b` lowers to `.const b`, the program's first gate. -/
theorem progOf_gate_zero {C : Verity.Boolean.Circuit} {κ : Fin C.N → Option Bool} {b : Bool} (g : Fin C.N)
    (hg : g.val = 0) (hin : C.op g = .input) (hκ : κ g = some b) :
    (Flock.Bridge.progOf C κ).gates[0]? = some (.const b) := by
  have hpos : 0 < C.N := g.pos
  cases hl : List.finRange C.N with
  | nil =>
    have := congrArg List.length hl
    simp only [List.length_finRange, List.length_nil] at this
    omega
  | cons a t =>
    have ha : a = g := by
      have e : (List.finRange C.N)[0]? = some a := by rw [hl]; rfl
      rw [List.getElem?_eq_getElem (by simp only [List.length_finRange]; exact hpos)] at e
      apply Fin.ext
      rw [← Option.some.inj e, hg]
      simp
    subst ha
    simp only [Flock.Bridge.progOf, hl, List.flatMap_cons, Flock.Bridge.lower, hin, hκ, List.cons_append,
      List.getElem?_cons_zero]

/-- **One-stage v2's statement's gate 0 is the constant**: its record's statement (`Verity.Sampled.Record.stmtA`,
`prog := progOf A.Cc A.κ`) lowers the key program's constant wire, gate 0, an input gate its constants set. -/
theorem stmtA_gate_zero (A : Verity.Sampled.Live) :
    (Verity.Sampled.Record.stmtA A).prog.gates[0]? = some (.const true) := by
  let g : Fin A.Cc.N := FlockSoundness.Discharge.Layout.oneG (c := A.D.unroll) (hU := A.hU) (n := A.n)
  have hg : g.val = 0 := rfl
  have hc : Flock.Bridge.constsOfP A.D.unroll A.hU A.n {} A.pub g = some true := by
    rw [Flock.Bridge.constsOfP_empty, Flock.Bridge.constWiresOf_eq_ite]
    simp
    exact hg
  have hin : A.Cc.op g = .input := by
    have h1 : (A.Cc.op g).isInput = true :=
      Flock.Bridge.constsOfP_isInput (FlockSoundness.Discharge.Layout.outsOfP A.D.unroll A.hU A.n {}) hc
    revert h1
    generalize A.Cc.op g = o
    cases o <;> simp [Verity.Boolean.Op.isInput]
  exact progOf_gate_zero g hg hin hc

/-- **One-stage v2's statement is no flattening of a Program whose gate 0 is an input gate**: at its record's parse
(`Verity.Sampled.Record.code`, whose statement is `stmtA`). -/
theorem not_flattens_stmtA {P : Flock.Program.Prog} (A : Verity.Sampled.Live)
    (hP : ∀ p ops sc, P.gate 0 = .ok (p, ops, sc) → Flock.Program.isInputId (P.defs[p]!).key = true) :
    ¬ Flattens P (Verity.Sampled.Record.stmtA A).prog :=
  not_flattens_of_const_zero (stmtA_gate_zero A) hP

end Properties.NetworkScan
