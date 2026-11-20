module

public import Proofs.Properties.InferenceOnly.RecordWireRead

/-!
# The wires a Program's root states

Each wire `Spot.Edges.edgesOf` gives has a step for each digit and no digit of radix zero (`edgesOf_wire`), so the
record's wiring is of wires SPOT's digits rule takes (`edgeDigits`, the hypothesis `EdgeDigits` proved).
-/

@[expose] public section

namespace InferenceOnlyVerifier.Record.SecurityProofs

open Flock.Program (Prog Part)
open Spot.Edges (edgesOf partWires)
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

/-- A wire with a step for each digit and no digit of radix zero. -/
def Digits (w : Spot.Wires.Wire) : Prop := w.step.length = w.radix.length ∧ ∀ r ∈ w.radix, 1 ≤ r

private theorem appended {a : List Spot.Wires.Wire} {wire : Spot.Wires.Wire} {radix : List Nat} {steps : List Int}
    {f : Int → Int} (ha : ∀ w ∈ a, Digits w)
    (hg : ¬ (radix.length != steps.length || radix.any (· == 0)) = true) (hr : wire.radix = radix)
    (hs : wire.step = steps.map f) : ∀ w ∈ a ++ [wire], Digits w := by
  intro w hw
  rcases List.mem_append.1 hw with hw | hw
  · exact ha w hw
  · rw [List.mem_singleton] at hw
    subst hw
    have h1 : radix.length = steps.length := by
      refine Classical.byContradiction fun hne => hg ?_
      simp only [Bool.or_eq_true]
      exact Or.inl (bne_iff_ne.2 hne)
    refine ⟨by rw [hs, hr, List.length_map, h1], fun r hm => ?_⟩
    rw [hr] at hm
    refine Classical.byContradiction fun h0 => hg ?_
    simp only [Bool.or_eq_true, List.any_eq_true]
    exact Or.inr ⟨r, hm, beq_iff_eq.2 (by omega)⟩

/-- **A part's wires have a step for each digit and no digit of radix zero.** -/
theorem partWires_wire {P : Prog} {fu W k i q s0 out} (h : partWires P fu W k i q s0 = .ok out) :
    ∀ w ∈ out, Digits w := by
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
      refine forIn_upto (fun o => ∀ w ∈ o, Digits w) _ _ ?_ _ _ (by simp) hs
      intro j a st _ ha hst
      repeat' (first
        | (cases hst; exact ha)
        | (cases hst; exact appended ha ‹_› rfl rfl)
        | (refine bind_ok_elim hst (fun _ hx hst => ?_)
           first
             | (simp [throw, throwThe, MonadExceptOf.throw] at hx; done)
             | skip)
        | (refine ite_ok hst (fun _ hst => ?_) (fun _ hst => ?_))
        | (split at hst))
  · cases h

/-- **Each wire `edgesOf` gives has a step for each digit and no digit of radix zero.** -/
theorem edgesOf_wire {P : Prog} {rs W ws} (h : edgesOf P rs W = .ok ws) : ∀ w ∈ ws, Digits w := by
  unfold edgesOf at h
  dsimp only at h
  refine ite_ok h (fun _ h => ?_) (fun _ h => ?_)
  · refine bind_ok_elim h (fun _ hx _ => ?_)
    simp [throw, throwThe, MonadExceptOf.throw] at hx
  refine bind_ok_elim h (fun o hs h => ?_)
  cases h
  suffices hI : ∀ w ∈ o, Digits w from fun w hw => hI w (List.mem_reverse.1 hw)
  refine forIn_upto (fun o => ∀ w ∈ o, Digits w) _ _ ?_ _ _ (by simp) hs
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
    refine forIn_upto (fun o => ∀ w ∈ o, Digits w) _ _ ?_ _ _ ha hi
    intro i b st' hi' hb hst'
    refine bind_ok_elim hst' (fun b' hq hst' => ?_)
    cases hst'
    refine forIn_upto (fun o => ∀ w ∈ o, Digits w) _ _ ?_ _ _ hb hq
    intro qi c st'' hqi hc hst''
    refine bind_ok_elim hst'' (fun l hl hst'' => ?_)
    cases hst''
    intro w hw
    rcases List.mem_append.1 hw with hw | hw
    · exact partWires_wire hl w (List.mem_reverse.1 hw)
    · exact hc w hw
  · refine bind_ok_elim hst (fun _ hx _ => ?_)
    simp [throw, throwThe, MonadExceptOf.throw] at hx

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
    exact edgesOf_wire hws w hw
  · refine bind_ok_elim h (fun _ hx _ => ?_)
    simp [throw, throwThe, MonadExceptOf.throw] at hx

end InferenceOnlyVerifier.Record.SecurityProofs
