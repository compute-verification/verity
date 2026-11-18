module

public import Proofs.Properties.InferenceOnly.RecordWireRead

/-!
# The wires a Program's root states

`Spot.Edges.edgesOf` gives one wire per reader row, each of one digit of radix at least 1 and one step, and only from
an affine or strided part of an argument of a root node that reads an earlier non-input root node (`edgesOf_wire`).
So the record's wiring is of wires SPOT's digits rule takes (`edgeDigits`, the hypothesis `EdgeDigits` proved), and a
root none of whose parts reads such a node states no wire (`edgesOf_nil`).
-/

@[expose] public section

namespace InferenceOnlyVerifier.Record.SecurityProofs

open Flock.Program (Prog Part)
open Spot.Edges (edgesOf partWires leafOf isInputNode)
open InferenceOnlyVerifier.Record.SecurityProofs.Json (bind_ok)

private theorem bind_ok_elim {α β : Type} {x : Except String α} {f : α → Except String β} {b : β} {G : Prop}
    (h : x >>= f = .ok b) (k : ∀ a, x = .ok a → f a = .ok b → G) : G := by
  obtain ⟨a, hx, hf⟩ := bind_ok h
  exact k a hx hf

private theorem ite_ok {α : Type} {c : Prop} [Decidable c] {x y z : α} {G : Prop} (h : (if c then x else y) = z)
    (hx : c → x = z → G) (hy : ¬ c → y = z → G) : G := by
  by_cases hc : c
  · exact hx hc (by simpa [hc] using h)
  · exact hy hc (by simpa [hc] using h)

private theorem forIn_inv {β : Type} (I : β → Prop) (l : List Nat) (f : Nat → β → Except String (ForInStep β))
    (hf : ∀ j ∈ l, ∀ a s, I a → f j a = .ok s → I s.value) :
    ∀ (b b' : β), I b → forIn l b f = .ok b' → I b' := by
  induction l with
  | nil => intro b b' hb h; simp only [List.forIn_nil] at h; cases h; exact hb
  | cons a l ih =>
    intro b b' hb h
    simp only [List.forIn_cons] at h
    cases hs : f a b with
    | error e => rw [hs] at h; cases h
    | ok s =>
      rw [hs] at h
      have := hf a (by simp) b s hb hs
      cases s with
      | done c => cases h; exact this
      | yield c => exact ih (fun j hj => hf j (by simp [hj])) c b' this h

/-- An invariant of a loop over `[0:n]` that each iteration keeps holds of its result. -/
private theorem forIn_upto {β : Type} (I : β → Prop) (n : Nat) (f : Nat → β → Except String (ForInStep β))
    (hf : ∀ j a s, j < n → I a → f j a = .ok s → I s.value) (b b' : β) (hb : I b)
    (h : forIn [:n] b f = .ok b') : I b' := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  refine forIn_inv I _ f (fun j hj a s ha hs => hf j a s ?_ ha hs) b b' hb h
  simp [List.mem_range', Std.Legacy.Range.size] at hj
  omega

/-- A part that reads a root node's outputs: affine or strided, of a node and not a parameter, the node no input. -/
def ReadsNode (P : Prog) : Part → Prop
  | .affine p i _ _ => p = false ∧ isInputNode P i = false
  | .strided p i _ _ _ _ _ => p = false ∧ isInputNode P i = false
  | .explicit _ => False

/-- A wire of one digit, of radix at least 1, with one step. -/
def OneDigit (w : Spot.Wires.Wire) : Prop := (∃ c, w.radix = [c] ∧ 1 ≤ c) ∧ w.step.length = 1

/-- Some part of an argument of a root node reads a root node's outputs (`ReadsNode`). -/
def EdgeAt (P : Prog) : Prop :=
  ∃ k i qi, k < (P.defs[P.root]!).body.nodes.size ∧ i < ((P.defs[P.root]!).body.nodes[k]!).args.size ∧
    qi < (((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts.size ∧
    ReadsNode P ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts[qi]!)

private theorem leafOf_node {q : Part} {k0 L len n w l0 : Nat} {S : Int}
    (h : leafOf q k0 L len n = .ok (w, false, l0, S)) :
    match q with
    | .affine p i _ _ => p = false ∧ i = w
    | .strided p i _ _ _ _ _ => p = false ∧ i = w
    | .explicit _ => False := by
  cases q with
  | affine p i b c =>
    simp only [leafOf] at h
    split at h
    · cases h
    · cases h; simp_all
  | strided p i b inner os is c =>
    simp only [leafOf] at h
    split at h
    · cases h
    · split at h
      · cases h
      · cases h; simp_all
  | explicit it => simp [leafOf] at h

private theorem appended {P : Prog} {q : Part} {a : List Spot.Wires.Wire} {c : Nat} {wire : Spot.Wires.Wire}
    {x : Nat × Bool × Nat × Int} {k0 L len n : Nat} (ha : ∀ w ∈ a, OneDigit w ∧ ReadsNode P q)
    (hx : leafOf q k0 L len n = .ok x) (he : ¬ x.2.1 = true) (hi : ¬ isInputNode P x.1 = true)
    (hc : ¬ (c == 0) = true) (hr : wire.radix = [c]) (hs : wire.step.length = 1) :
    ∀ w ∈ a ++ [wire], OneDigit w ∧ ReadsNode P q := by
  intro w hw
  rcases List.mem_append.1 hw with hw | hw
  · exact ha w hw
  · rw [List.mem_singleton] at hw
    subst hw
    obtain ⟨v, e, l0, S⟩ := x
    simp only [Bool.not_eq_true] at he hi
    subst he
    have hq := leafOf_node hx
    refine ⟨⟨⟨c, hr, by simp at hc; omega⟩, hs⟩, ?_⟩
    cases q with
    | affine p i _ _ => exact ⟨hq.1, hq.2 ▸ hi⟩
    | strided p i _ _ _ _ _ => exact ⟨hq.1, hq.2 ▸ hi⟩
    | explicit _ => exact hq.elim

/-- **A part's wires are of one digit, and only from a part that reads a node.** -/
theorem partWires_wire {P : Prog} {fu W k i q s0 out} (h : partWires P fu W k i q s0 = .ok out) :
    ∀ w ∈ out, OneDigit w ∧ ReadsNode P q := by
  unfold partWires at h
  split at h
  · rename_i rt rb hfu
    dsimp only at h
    generalize (P.defs[(P.defs[P.root]!.body.nodes[k]!).fn]!).params[i]! = pt at h
    obtain ⟨port, ty⟩ := pt
    dsimp only at h
    by_cases hc : (W.cutsOf rt port == 0 || ty.leaves % W.cutsOf rt port != 0 ||
        Spot.Bits.bitsOf ty % W.cutsOf rt port != 0) = true
    · rw [ite_eq_left_iff.2 (fun h' => absurd hc h')] at h
      refine bind_ok_elim h (fun _ hx _ => ?_)
      simp [throw, throwThe, MonadExceptOf.throw] at hx
    · rw [ite_eq_right_iff.2 (fun h' => absurd h' hc)] at h
      refine bind_ok_elim h (fun o hs h => ?_)
      cases h
      refine forIn_upto (fun o => ∀ w ∈ o, OneDigit w ∧ ReadsNode P q) _ _ ?_ _ _ (by simp) hs
      intro j a st _ ha hst
      repeat' (first
        | (cases hst; exact ha)
        | (cases hst; exact appended ha ‹_› ‹_› ‹_› ‹_› rfl rfl)
        | (refine bind_ok_elim hst (fun _ hx hst => ?_)
           first
             | (simp [throw, throwThe, MonadExceptOf.throw] at hx; done)
             | skip)
        | (refine ite_ok hst (fun _ hst => ?_) (fun _ hst => ?_))
        | (split at hst))
  · cases h

/-- **Each wire `edgesOf` gives is of one digit, and comes from a part of the root that reads a node.** -/
theorem edgesOf_wire {P : Prog} {rs W ws} (h : edgesOf P rs W = .ok ws) : ∀ w ∈ ws, OneDigit w ∧ EdgeAt P := by
  unfold edgesOf at h
  dsimp only at h
  refine ite_ok h (fun _ h => ?_) (fun _ h => ?_)
  · refine bind_ok_elim h (fun _ hx _ => ?_)
    simp [throw, throwThe, MonadExceptOf.throw] at hx
  refine bind_ok_elim h (fun o hs h => ?_)
  cases h
  refine forIn_upto (fun o => ∀ w ∈ o, OneDigit w ∧ EdgeAt P) _ _ ?_ _ _ (by simp) hs
  intro k a st hk ha hst
  refine ite_ok hst (fun _ hst => ?_) (fun _ hst => ?_)
  · cases hst; exact ha
  split at hst
  · refine ite_ok hst (fun _ hst => ?_) (fun _ hst => ?_)
    · refine bind_ok_elim hst (fun _ hx _ => ?_)
      simp [throw, throwThe, MonadExceptOf.throw] at hx
    refine ite_ok hst (fun _ hst => ?_) (fun _ hst => ?_)
    · refine bind_ok_elim hst (fun _ hx _ => ?_)
      simp [throw, throwThe, MonadExceptOf.throw] at hx
    refine bind_ok_elim hst (fun a' hi hst => ?_)
    cases hst
    refine forIn_upto (fun o => ∀ w ∈ o, OneDigit w ∧ EdgeAt P) _ _ ?_ _ _ ha hi
    intro i b st' hi' hb hst'
    refine bind_ok_elim hst' (fun b' hq hst' => ?_)
    cases hst'
    refine forIn_upto (fun o => ∀ w ∈ o, OneDigit w ∧ EdgeAt P) _ _ ?_ _ _ hb hq
    intro qi c st'' hqi hc hst''
    refine bind_ok_elim hst'' (fun l hl hst'' => ?_)
    cases hst''
    intro w hw
    rcases List.mem_append.1 hw with hw | hw
    · exact hc w hw
    · obtain ⟨hd, hr⟩ := partWires_wire hl w hw
      exact ⟨hd, k, i, qi, hk, hi', hqi, hr⟩
  · refine bind_ok_elim hst (fun _ hx _ => ?_)
    simp [throw, throwThe, MonadExceptOf.throw] at hx

/-- **A root none of whose parts reads a node states no wire.** -/
theorem edgesOf_nil {P : Prog} {rs W ws} (h : edgesOf P rs W = .ok ws) (hn : ¬ EdgeAt P) : ws = [] := by
  cases ws with
  | nil => rfl
  | cons w ws => exact absurd (edgesOf_wire h w (List.mem_cons_self ..)).2 hn

/-- **Every wire of the record's wiring has a step for each digit, and no digit of radix zero** (`EdgeDigits`). -/
theorem edgeDigits : EdgeDigits := by
  intro P doc text h w hw
  unfold Spot.RecordText.wiring at h
  refine bind_ok_elim h (fun ts _ h => ?_)
  dsimp only at h
  split at h
  · refine bind_ok_elim h (fun rs' hr h => ?_)
    cases hr
    refine bind_ok_elim h (fun cuts _ h => ?_)
    refine bind_ok_elim h (fun ws hws h => ?_)
    cases h
    obtain ⟨⟨c, hc, h1⟩, hs⟩ := (edgesOf_wire hws w hw).1
    rw [hc, hs]
    simpa using h1
  · refine bind_ok_elim h (fun _ hx _ => ?_)
    simp [throw, throwThe, MonadExceptOf.throw] at hx

end InferenceOnlyVerifier.Record.SecurityProofs
