module

public import Proofs.Flock.RecGeneric.ZKProof

/-!
# Z1-cost's proof: the compiled simulator's calls

A run of `simRounds` makes at most one question to `V*` and its message program's calls a round (`simRounds_cost`).
The compiled simulator is such a run of `n_h` rounds of frames, then the refusal's rounds or the outer simulator under
the first part (`spTag`, which renames only its questions), so its calls are the frames', one question a first-part
round, and the refusal's rounds or the outer simulator's (`ZkRelease.sim_cost`).

`Flock.SecurityProofs.GenericRecursiveZKCost` is Z1-cost.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

/-- **`simRounds`' calls**: at most its message program's `B` and one question to `V*` a round. -/
theorem simRounds_cost (H : Msg → Msg) (ask : List Msg → Out) (f : List (Msg × Out) → SP Msg) (B : ℕ)
    (hf : ∀ ms, (SP.eval H ask (f ms)).2 ≤ B) :
    ∀ (n : ℕ) (ms : List (Msg × Out)), (SP.eval H ask (simRounds f n ms)).2 ≤ n * (B + 1)
  | 0, _ => by simp [simRounds, SP.eval]
  | n + 1, ms => by
    unfold simRounds
    split_ifs
    · rw [ZKMal.recSpBind_eq, ZKMal.spBind_eval]
      simp only [SP.eval]
      exact (Nat.add_le_add (hf ms) (Nat.add_le_add_right (simRounds_cost H ask f B hf n _) 1)).trans
        (le_of_eq (by ring))
    · simp [SP.eval]

namespace ZkRelease

variable {W : Type} {Lv : Leaves} (Zs : ZkRelease W Lv)

/-- **The compiled simulator's calls** given the outcome `s`: the first part's `n_h·(c_F + 1)`, then `k_O` at the
refusal or the outer simulator's `c_O`. -/
theorem sim_cost (H : Msg → Msg) (cF cO : Msg → ℕ) (hF : ∀ x pre ls, ((Zs.frame x pre ls).eval H).2.length ≤ cF x)
    (hO : ∀ x xO o (c : Zs.CO x) ask, ((Zs.progO x xO o c).eval H ask).2 ≤ cO x) (x : Msg) (s : Option Zs.St)
    (c : Zs.sim.C x) (ask : List Msg → Out) :
    ((Zs.sim.prog x s c).eval H ask).2 ≤ Zs.nh x * (cF x + 1) + s.elim (Zs.ko x) fun _ => cO x := by
  have h1 : (SP.eval H ask (simRounds (fun ms => if h : ms.length < Zs.nh x then
        spOf (Zs.frame x ms (List.ofFn fun j => Lv.ideal (c.1 (Zs.leafAt x h j))))
      else .ret []) (Zs.nh x) [])).2 ≤ Zs.nh x * (cF x + 1) := by
    refine simRounds_cost H ask _ (cF x) (fun ms => ?_) (Zs.nh x) []
    by_cases h : ms.length < Zs.nh x
    · rw [dite_eq_left h, ZKMal.recSpOf_eq, ZKMal.ocSP_eval]
      exact hF _ _ _
    · rw [dite_eq_right h]
      simp [SP.eval]
  cases s with
  | none =>
    show (SP.eval H ask (spBind (simRounds _ (Zs.nh x) []) fun ms =>
      if goesOn ms then simRounds (fun _ => .ret refusal) (Zs.ko x) ms else .ret ms)).2 ≤ _
    rw [ZKMal.recSpBind_eq, ZKMal.spBind_eval]
    refine Nat.add_le_add h1 ?_
    split_ifs
    · simpa using simRounds_cost H ask (fun _ => .ret refusal) 0 (fun _ => by simp [SP.eval]) (Zs.ko x) _
    · simp [SP.eval]
  | some o =>
    show (SP.eval H ask (spBind (simRounds _ (Zs.nh x) []) fun ms =>
      if goesOn ms then spBind (spTag (ms.map Prod.fst) (Zs.progO x (Zs.xo x ms) o c.2)) fun ts =>
          .ret (ms ++ ts.map fun p => (tagOuter p.1, p.2))
      else .ret ms)).2 ≤ _
    rw [ZKMal.recSpBind_eq, ZKMal.spBind_eval]
    refine Nat.add_le_add h1 ?_
    split_ifs
    · rw [ZKMal.recSpBind_eq, ZKMal.spBind_eval, spTag_eval]
      simp only [SP.eval, add_zero, Option.elim]
      exact hO _ _ _ _ _
    · simp [SP.eval]

end ZkRelease

end Claims.RecGeneric

open Claims Claims.RecGeneric in
/-- **Z1-cost** (`Flock.Guarantees.GenericRecursiveZKCost`): `ZkRelease.sim_cost` at each outcome. -/
theorem Flock.SecurityProofs.GenericRecursiveZKCost : Flock.Guarantees.GenericRecursiveZKCost := by
  intro H W Lv Zs cF cO hF hO
  exact ⟨fun x c ask => Zs.sim_cost H cF cO hF hO x none c ask,
    fun o x c ask => Zs.sim_cost H cF cO hF hO x (some o) c ask⟩
