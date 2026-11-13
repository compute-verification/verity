module

public import Properties.NetworkScan.Assumptions
public import Proofs.Flock.SpotFlock.Code
public import Proofs.Flock.Bridge.BitsKeyProg

/-!
# `StmtFlattens` at SPOT[Flock]'s two-stage parser

A scratch module: what `Properties.NetworkScan.StmtFlattens` asks of the registered descriptor at the two-stage parser
`SpotFlock.parse`, whose statements' gates are Bits' layout of the members' circuit files
(`Properties.Spot.bitsProgs`), not a flattening of the descriptor.

* `bitsProgs_gate_zero`: with a member, gate 0 of Bits' layout is the constant.
* `first_gate_of_flattens`: a Program that flattens to a circuit whose gate 0 is the constant has a first gate that is
  no input gate and reads nothing.
* `parse_of_json`, `parse_desc_prog`: the parser takes every message that reads as JSON, to a statement holding the
  auditor's descriptor and Bits' layout.
* `stmtFlattens_first_gate`: so at a registration with a template and a message that reads as JSON, `StmtFlattens`
  gives a Program the auditor's descriptor decodes to, whose first gate is no input gate and reads nothing.
* `not_stmtFlattens_of_input_first`: where the descriptor's Program's first gate is an input gate, `StmtFlattens` fails.
-/

@[expose] public section

namespace ScanFlattensTry

/-- A list whose entry 0 is `a` starts with `a`. -/
theorem cons_of_getElem?_zero {α : Type} {l : List α} {a : α} (h : l[0]? = some a) : ∃ t, l = a :: t := by
  cases l with
  | nil => simp at h
  | cons b t =>
    have hb : b = a := by simpa using h
    exact ⟨t, by rw [hb]⟩

/-- Gate 0 of a circuit file's key program lowers to the constant: it is an input gate, which the constants set. -/
theorem lower_zero (c : Flock.Circuit) (n : ℕ) :
    Spot.Bits.lower (Spot.Bits.keyProg c n) (Spot.Bits.consts c n) 0 = [Spot.Bits.LGate.const true] := by
  have h2 : 2 ≤ (Spot.Bits.keyProg c n).inputs := by
    rw [Flock.Bridge.keyProg_inputs]
    exact Nat.le_add_right 2 _
  have hrow : (Spot.Bits.keyProg c n).row? 0 = none := by
    have h0 : 0 < (Spot.Bits.keyProg c n).inputs := by omega
    simp [Spot.Bits.KeyProg.row?, h0]
  have hk : Spot.Bits.consts c n 0 = some true := rfl
  simp only [Spot.Bits.lower, hrow, hk]

/-- **Gate 0 of a member's program is the constant.** -/
theorem memberProg_gate_zero (c : Flock.Circuit) (n : ℕ) :
    (Properties.Spot.memberProg c n).gates[0]? = some (.const true) := by
  have h2 : 2 ≤ (Spot.Bits.keyProg c n).inputs := by
    rw [Flock.Bridge.keyProg_inputs]
    exact Nat.le_add_right 2 _
  obtain ⟨k, hk⟩ : ∃ k, (Spot.Bits.keyProg c n).size = k + 1 :=
    ⟨(Spot.Bits.keyProg c n).size - 1, by unfold Spot.Bits.KeyProg.size; omega⟩
  have e : (Properties.Spot.memberProg c n).gates =
      ((List.range (Spot.Bits.keyProg c n).size).flatMap
        (Spot.Bits.lower (Spot.Bits.keyProg c n) (Spot.Bits.consts c n))).map Properties.Spot.bitsGate := rfl
  rw [e, hk, List.range_succ_eq_map, List.flatMap_cons, lower_zero]
  rfl

/-- **Gate 0 of Bits' layout is the constant**, where the audit has a member. -/
theorem bitsProgs_gate_zero {ms : Spot.Bits.Members} (h : ms ≠ []) :
    (Properties.Spot.bitsProgs ms).gates[0]? = some (.const true) := by
  obtain ⟨ci, rest, rfl⟩ := List.exists_cons_of_ne_nil h
  obtain ⟨t, ht⟩ := cons_of_getElem?_zero (memberProg_gate_zero ci.1 ci.2.size)
  have e : (Properties.Spot.bitsProgs (ci :: rest)).gates =
      (Properties.Spot.memberProg ci.1 ci.2.size).gates ++
        (Properties.Spot.sideBySide (rest.map fun ci => Properties.Spot.memberProg ci.1 ci.2.size)).gates.map
          (Properties.Spot.shiftGate (Properties.Spot.memberProg ci.1 ci.2.size).gates.length) := rfl
  rw [e, ht]
  rfl

/-- **A Program that flattens to a circuit whose gate 0 is the constant**: its first gate is no input gate, and it
reads nothing. -/
theorem first_gate_of_flattens {P : Flock.Program.Prog} {C : Claims.Sampling.Prog}
    (hC : C.gates[0]? = some (.const true)) (hF : Properties.NetworkScan.Flattens P C) :
    ∃ (p : ℕ) (ops : Array ℕ) (sc : Flock.Program.Scope), P.gate 0 = .ok (p, ops, sc) ∧
      Flock.Program.isInputId (P.defs[p]!).key = false ∧ ∀ a, a ∉ ops := by
  obtain ⟨-, hg, -⟩ := hF
  have hlen : 0 < C.gates.length := by
    refine Nat.pos_of_ne_zero fun h0 => ?_
    rw [List.eq_nil_of_length_eq_zero h0] at hC
    simp at hC
  obtain ⟨p, ops, sc, hgate, hiff, hreads⟩ := hg 0 hlen
  refine ⟨p, ops, sc, hgate, ?_, fun a ha => ?_⟩
  · cases hk : Flock.Program.isInputId (P.defs[p]!).key with
    | false => rfl
    | true =>
      have hi := hiff.2 hk
      rw [hC] at hi
      cases hi
  · have hr := (hreads a).2 ha
    rw [hC] at hr
    simp [Claims.Sampling.BGate.reads] at hr

/-- **The two-stage parser takes every message that reads as JSON.** -/
theorem parse_of_json (h : SpotFlock.Held) {x : Claims.Msg} {rec : Lean.Json}
    (hx : SpotFlock.jsonMsg x = some rec) : ∃ s, SpotFlock.parse h x = some s := by
  unfold SpotFlock.parse SpotFlock.decode
  rw [hx]
  exact ⟨_, rfl⟩

/-- **A statement the two-stage parser gives** holds the auditor's descriptor and Bits' layout of its members. -/
theorem parse_desc_prog {h : SpotFlock.Held} {x : Claims.Msg} {s : Claims.Sampling.AuditStatement}
    (hx : SpotFlock.parse h x = some s) : s.desc = h.desc ∧ s.prog = Properties.Spot.bitsProgs h.ms := by
  unfold SpotFlock.parse at hx
  obtain ⟨st, -, rfl⟩ := Option.map_eq_some_iff.1 hx
  exact ⟨rfl, rfl⟩

/-- A registration with a template has a member. -/
theorem ms_ne_nil {h : SpotFlock.Held} (hv : h.vus ≠ []) : h.ms ≠ [] := by
  intro hnil
  have hl := congrArg List.length hnil
  exact hv (by simpa [SpotFlock.Held.ms] using hl)

/-- **What `StmtFlattens` asks at the two-stage parser.** At a compiled verifier whose parser is `SpotFlock.parse h`,
a registration with a template and a message that reads as JSON, `StmtFlattens` gives a Program the auditor's
descriptor decodes to, whose first gate is no input gate and reads nothing: it sits where Bits' layout holds its
constant. -/
theorem stmtFlattens_first_gate {Coin : Type} {A : Claims.SamplingGeneric.Arg Coin}
    (D : Claims.SamplingGeneric.Compiled Coin A) (h : SpotFlock.Held) (hD : D.parse = SpotFlock.parse h)
    (hv : h.vus ≠ []) {x : Claims.Msg} {rec : Lean.Json} (hx : SpotFlock.jsonMsg x = some rec)
    (hF : Properties.NetworkScan.StmtFlattens D) :
    ∃ P, Flock.Program.decode ⟨h.desc.toArray⟩ = .ok P ∧
      ∃ (p : ℕ) (ops : Array ℕ) (sc : Flock.Program.Scope), P.gate 0 = .ok (p, ops, sc) ∧
        Flock.Program.isInputId (P.defs[p]!).key = false ∧ ∀ a, a ∉ ops := by
  obtain ⟨s, hs⟩ := parse_of_json h hx
  obtain ⟨hdesc, hprog⟩ := parse_desc_prog hs
  obtain ⟨P, hP, hFl⟩ := hF x s (by rw [hD]; exact hs)
  rw [hprog] at hFl
  rw [hdesc] at hP
  exact ⟨P, hP, first_gate_of_flattens (bitsProgs_gate_zero (ms_ne_nil hv)) hFl⟩

/-- **`StmtFlattens` fails at the two-stage parser** at a registration with a template, where a message reads as JSON
and the first gate of every Program the auditor's descriptor decodes to is an input gate. -/
theorem not_stmtFlattens_of_input_first {Coin : Type} {A : Claims.SamplingGeneric.Arg Coin}
    (D : Claims.SamplingGeneric.Compiled Coin A) (h : SpotFlock.Held) (hD : D.parse = SpotFlock.parse h)
    (hv : h.vus ≠ []) {x : Claims.Msg} {rec : Lean.Json} (hx : SpotFlock.jsonMsg x = some rec)
    (hin : ∀ P, Flock.Program.decode ⟨h.desc.toArray⟩ = .ok P →
      ∀ (p : ℕ) (ops : Array ℕ) (sc : Flock.Program.Scope), P.gate 0 = .ok (p, ops, sc) →
        Flock.Program.isInputId (P.defs[p]!).key = true) :
    ¬ Properties.NetworkScan.StmtFlattens D := by
  intro hF
  obtain ⟨P, hP, p, ops, sc, hgate, hno, -⟩ := stmtFlattens_first_gate D h hD hv hx hF
  have hyes := hin P hP p ops sc hgate
  rw [hno] at hyes
  cases hyes

/-- The two-stage code's parser is `SpotFlock.parse`. -/
theorem code_parse (h : SpotFlock.Held) : (SpotFlock.code h).parse = SpotFlock.parse h := rfl

end ScanFlattensTry
