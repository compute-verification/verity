module

public import Proofs.Properties.Flock.Circuit.LowerEval.Widths

/-! A callee that is one of §9's gates or constant bits, by the decoder's predicate (`gateKind?`): no input, a gate of
`Flock.Lower` (`gateOf?`), of return `Value<1>` and parameters on bits, its value its gate's on its operands. -/

@[expose] public section

namespace Flock.Chain

open Flock.Program
open private widthAfter? from Circuits.Descriptor

/-- **A gate callee**: Definition `f` is a primitive whose key §9 names a gate or a constant bit (`gateKind?`). -/
def GateCallee (P : Prog) (f : ℕ) : Prop :=
  (P.defs[f]!).prim = true ∧ (gateKind? (P.defs[f]!).key).isSome = true

theorem widthAfter_family {pre id : String} {n : ℕ} (h : widthAfter? pre id = some n) :
    Flock.Lower.family? pre id = true ∧ id.startsWith pre = true := by
  unfold widthAfter? at h
  unfold Flock.Lower.family?
  split at h
  · cases h
  · dsimp only at h
    split at h
    · simp_all
    · cases h

theorem widthAfter_none {pre id : String} (h : id.startsWith pre = false) : widthAfter? pre id = none := by
  unfold widthAfter?
  simp [h]

theorem startsWith_false {s p : String} (h : ¬ p.toList <+: s.toList) : s.startsWith p = false := by
  cases hs : s.startsWith p
  · rfl
  · exact absurd (String.startsWith_string_iff.1 hs) h

theorem startsWith_apart {s p q : String} (hq : s.startsWith q = true) (h1 : ¬ p.toList <+: q.toList)
    (h2 : ¬ q.toList <+: p.toList) : s.startsWith p = false := by
  cases hs : s.startsWith p
  · rfl
  · rcases List.prefix_or_prefix_of_prefix (String.startsWith_string_iff.1 hs) (String.startsWith_string_iff.1 hq)
      with h | h
    · exact absurd h h1
    · exact absurd h h2

theorem gateKind_cases {k : String} (h : (gateKind? k).isSome = true) :
    k = "Not_v1" ∨ k = "Const1[0x0]_v1" ∨ k = "Const1[0x1]_v1" ∨ (∃ n, widthAfter? "And" k = some n) ∨
      ∃ n, widthAfter? "Xor" k = some n := by
  by_cases h1 : k = "Not_v1"
  · exact Or.inl h1
  by_cases h2 : k = "Const1[0x0]_v1"
  · exact Or.inr (Or.inl h2)
  by_cases h3 : k = "Const1[0x1]_v1"
  · exact Or.inr (Or.inr (Or.inl h3))
  cases hA : widthAfter? "And" k with
  | some n => exact Or.inr (Or.inr (Or.inr (Or.inl ⟨n, rfl⟩)))
  | none =>
    cases hX : widthAfter? "Xor" k with
    | some n => exact Or.inr (Or.inr (Or.inr (Or.inr ⟨n, rfl⟩)))
    | none =>
      unfold gateKind? at h
      simp [h1, h2, h3, hA, hX] at h

/-- A key `gateKind?` names is no input's. -/
theorem gateKind_input {k : String} (h : (gateKind? k).isSome = true) : isInputId k = false := by
  have hI : ∀ p ∈ ["Input", "RegisteredInput", "SuppliedInput"], k.startsWith p = false := by
    intro p hp
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases gateKind_cases h with rfl | rfl | rfl | ⟨n, hn⟩ | ⟨n, hn⟩
    · exact startsWith_false (by rcases hp with rfl | rfl | rfl <;> decide)
    · exact startsWith_false (by rcases hp with rfl | rfl | rfl <;> decide)
    · exact startsWith_false (by rcases hp with rfl | rfl | rfl <;> decide)
    · exact startsWith_apart (widthAfter_family hn).2 (by rcases hp with rfl | rfl | rfl <;> decide)
        (by rcases hp with rfl | rfl | rfl <;> decide)
    · exact startsWith_apart (widthAfter_family hn).2 (by rcases hp with rfl | rfl | rfl <;> decide)
        (by rcases hp with rfl | rfl | rfl <;> decide)
  unfold isInputId inputId?
  rw [widthAfter_none (hI "Input" (by simp)), widthAfter_none (hI "RegisteredInput" (by simp)),
    widthAfter_none (hI "SuppliedInput" (by simp))]
  rfl

/-- A key `gateKind?` names is a gate of `Flock.Lower`. -/
theorem gateKind_gateOf {k : String} (h : (gateKind? k).isSome = true) : (Flock.Lower.gateOf? k).isSome = true := by
  rcases gateKind_cases h with rfl | rfl | rfl | ⟨n, hn⟩ | ⟨n, hn⟩
  · decide
  · decide
  · decide
  · have hf := (widthAfter_family hn).1
    unfold Flock.Lower.gateOf?
    repeat' split
    all_goals rfl
  · have hf := (widthAfter_family hn).1
    unfold Flock.Lower.gateOf?
    repeat' split
    all_goals rfl

/-- **What `decode` gives a gate callee**: no input and no word primitive, of return `Value<1>`, its parameters on bits,
and a gate of `Flock.Lower` by its id. -/
theorem gateCallee_facts {bytes : ByteArray} {P : Prog} (h : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[]) {f : ℕ} (hf : f < P.defs.size) (hg : GateCallee P f) :
    isInputId (P.defs[f]!).key = false ∧ isWordPrim (P.defs[f]!).prim (P.defs[f]!).key = false ∧
      (P.defs[f]!).ret = .value 1 ∧ (P.defs[f]!).params.map (·.2) = (P.defs[f]!).params.map (·.2.bitType) ∧
      ∃ g, Flock.Lower.gateOf? (P.defs[f]!).id = some g := by
  obtain ⟨-, hfacts⟩ := FlockSoundness.DecodeBoolean.decode_facts bytes P h hr
  obtain ⟨hid, -, hsig⟩ := hfacts f hf hg.1
  obtain ⟨gk, hgk⟩ := Option.isSome_iff_exists.1 hg.2
  obtain ⟨hpar, hret⟩ := hsig gk hgk
  refine ⟨gateKind_input hg.2, by simp [isWordPrim, isBooleanPrim, hg.2], ?_, ?_, ?_⟩
  · rw [hret]; cases gk <;> rfl
  · rw [hpar]; cases gk <;> simp [GateKind.signature, Ty.bitType]
  · rw [hid]; exact Option.isSome_iff_exists.1 (gateKind_gateOf hg.2)

/-- A primitive with a gate is its gate on its operands, at any fuel past none. -/
theorem evalDef_gate {Q : Prog} {f fuel : ℕ} {defn : Defn} (hd : Q.defs[f]? = some defn) (hp : defn.prim = true)
    {g : Flock.Lower.Gate} (hg : Flock.Lower.gateOf? defn.id = some g) (y : List Bool) :
    Flock.Program.evalDef? Q (fuel + 1) f y = some [Flock.Program.gateVal g y] := by
  rw [FlockSoundness.EvalBool.evalDef?_succ]
  simp [FlockSoundness.EvalBool.evalStep, hd, hp, hg]

/-- A gate callee's `Prog.evalBool` is its gate on its operands. -/
theorem evalBool_gate {P : Prog} {f : ℕ} (hf : f < P.defs.size) (hp : (P.defs[f]!).prim = true)
    {g : Flock.Lower.Gate} (hg : Flock.Lower.gateOf? (P.defs[f]!).id = some g) (y : List Bool) :
    Prog.evalBool P f y = [Flock.Program.gateVal g y] := by
  have hd : P.defs[f]? = some (P.defs[f]!) := by rw [getElem!_pos P.defs f hf]; exact getElem?_pos P.defs f hf
  unfold Prog.evalBool
  rw [show 2 * P.defs.size + 2 = (2 * P.defs.size + 1) + 1 from rfl, evalDef_gate hd hp hg]
  rfl

end Flock.Chain
