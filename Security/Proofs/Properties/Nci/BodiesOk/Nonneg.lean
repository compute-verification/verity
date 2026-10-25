import Proofs.Properties.Nci.BodiesOk.Basic

/-!
# Parsed strided parts name no negative leaf

`parsePartsIn` checks a strided part's corners against its collection, so no reference of a parsed strided part steps
below leaf 0 (`parsePartsIn_nonneg`), given none of an argument an alias may name does. `ReadsExact.PartNonneg` is the
same predicate (`partNonneg_iff`).
-/

namespace NciCheck.Properties.SecurityProofs.BodiesOk
open Flock.Program

/-- A part's references never step below leaf 0: a strided part's pattern (`parsePartsIn` checks its corners). -/
def PartNonneg : Part → Prop
  | .strided _ _ b inner os is c => ∀ k < c, 0 ≤ b + ((k / inner : Nat) : Int) * os + ((k % inner : Nat) : Int) * is
  | _ => True

/-- One unfolding of `parsePartsIn`, as `Walk.parsePartsIn_step` takes it, for `PartNonneg`: a strided part's corners
are within its collection, so every position it names is (`Walk.strided_corners`). -/
theorem parsePartsIn_nonneg_step (leaves : Bool → Nat → Option Nat) (args : Nat → Nat → Option Seq)
    (hargs : ∀ k x s, args k x = some s → ∀ q ∈ s.parts, PartNonneg q) (n : Nat)
    (b : Nat.below (motive := fun _ => Lean.Json → Flock.V (Array Part)) (n + 1))
    (hb : ∀ x ps, b.1 x = .ok ps → ∀ q ∈ ps, PartNonneg q) (j : Lean.Json) (ps : Array Part)
    (h : Flock.Program.parsePartsIn._f leaves args (n + 1) b j = .ok ps) : ∀ q ∈ ps, PartNonneg q := by
  delta Flock.Program.parsePartsIn._f at h
  dsimp only at h
  cases j with
  | null => exact nomatch h
  | bool _ => exact nomatch h
  | num _ => exact nomatch h
  | str _ => exact nomatch h
  | arr _ => exact nomatch h
  | obj kvs =>
    dsimp only at h
    split at h
    · rename_i a _
      rcases a with _ | _ | _ | _ | ⟨⟨l⟩⟩ | _
      all_goals try (injection h; done)
      rcases l with _ | ⟨k, _ | ⟨x, _ | ⟨y, l⟩⟩⟩
      all_goals try (injection h; done)
      conv at h => lhs; whnf
      split at h
      all_goals try (injection h; done)
      split at h
      all_goals try (injection h; done)
      rename_i s hs
      injection h with h
      subst h
      exact hargs _ _ s hs
    · split at h
      · rename_i c _
        rcases c with _ | _ | _ | _ | cs | _
        all_goals try (injection h; done)
        dsimp only at h
        rcases hm : cs.mapM (fun x => b.1 x) with e | r <;> simp only [hm, bind, Except.bind] at h
        · exact absurd h (by intro h; injection h)
        injection h with h; subst h
        intro q hq
        rw [← Array.foldl_toList] at hq
        rcases Walk.mem_foldl_append r.toList #[] q hq with hq | ⟨y, hy, hq⟩
        · simp at hq
        · obtain ⟨x, -, hx⟩ := Walk.array_mapM_ok _ cs r hm y (by simpa using hy)
          exact hb x y hx q hq
      · split at h
        · rename_i a _
          rcases a with _ | _ | _ | _ | ⟨⟨l⟩⟩ | _
          all_goals try (injection h; done)
          rcases l with _ | ⟨s0, _ | ⟨i0, _ | ⟨b0, _ | ⟨c0, _ | ⟨y, l⟩⟩⟩⟩⟩
          all_goals try (injection h; done)
          conv at h => lhs; whnf
          split at h
          all_goals try (injection h; done)
          split at h
          · injection h with h; subst h
            intro q hq
            simp only [Array.mem_singleton] at hq
            subst hq
            trivial
          · split at h
            all_goals try (injection h; done)
            injection h with h; subst h
            intro q hq
            simp only [Array.mem_singleton] at hq
            subst hq
            trivial
        · split at h
          · rename_i a _
            rcases a with _ | _ | _ | _ | ⟨⟨l⟩⟩ | _
            all_goals try (injection h; done)
            rcases l with _ | ⟨s0, _ | ⟨i0, _ | ⟨b0, _ | ⟨n0, _ | ⟨o0, _ | ⟨j0, _ | ⟨c0, _ | ⟨y, l⟩⟩⟩⟩⟩⟩⟩⟩
            all_goals try (injection h; done)
            conv at h => lhs; whnf
            split at h
            all_goals try (injection h; done)
            rename_i _ cc inn ost ist c _
            split at h
            all_goals try (injection h; done)
            rename_i hinn
            try dsimp only at h
            split at h
            · injection h with h; subst h
              intro q hq
              simp only [Array.mem_singleton] at hq
              subst hq
              intro k hk
              exact absurd hk (Nat.not_lt_zero k)
            try dsimp only at h
            split at h
            all_goals try (injection h; done)
            rename_i p i hw
            injection h with h; subst h
            intro q hq
            simp only [Array.mem_singleton] at hq
            subst hq
            intro k hk
            have hin : 0 < inn := by simp at hinn; omega
            obtain ⟨⟨⟨x1, y1⟩, hp1, h1⟩, -⟩ := Walk.strided_corners cc ost ist hin hk
            split at hw
            all_goals try (injection hw; done)
            split at hw
            all_goals try (injection hw; done)
            split at hw
            all_goals rename_i hq0
            · rw [if_pos hq0] at hp1
              split at hw
              all_goals try (injection hw; done)
              rename_i hc
              simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
              have e1 := le_trans hc.1 (Walk.foldl_min_le _ _ _ (Or.inl (List.mem_map_of_mem hp1)))
              dsimp only at h1 e1
              exact le_trans e1 h1
            · rw [if_neg hq0] at hp1
              split at hw
              all_goals try (injection hw; done)
              rename_i hc
              simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
              have e1 := le_trans hc.1 (Walk.foldl_min_le _ _ _ (Or.inl (List.mem_map_of_mem hp1)))
              dsimp only at h1 e1
              exact le_trans e1 h1
          · split at h
            · rename_i e _
              rcases e with _ | _ | _ | _ | items | _
              all_goals try (injection h; done)
              dsimp only at h
              obtain ⟨rs, -, h⟩ := Walk.except_bind_ok h
              injection h with h; subst h
              intro q hq
              simp only [Array.mem_singleton] at hq
              subst hq
              trivial
            · exact absurd h (by intro h; injection h)

/-- **No position of a parsed strided part is negative**, given none of an argument an alias may name is. -/
theorem parsePartsIn_nonneg (leaves : Bool → Nat → Option Nat) (args : Nat → Nat → Option Seq)
    (hargs : ∀ k x s, args k x = some s → ∀ q ∈ s.parts, PartNonneg q) :
    ∀ fuel j ps, Flock.Program.parsePartsIn fuel j leaves args = .ok ps → ∀ q ∈ ps, PartNonneg q := by
  intro fuel
  induction fuel with
  | zero =>
    intro j ps h
    delta Flock.Program.parsePartsIn at h
    exact nomatch h
  | succ n ih =>
    intro j ps h
    delta Flock.Program.parsePartsIn at h
    exact parsePartsIn_nonneg_step leaves args hargs n _
      (fun x ps hx => ih x ps (by delta Flock.Program.parsePartsIn; exact hx)) j ps h

end NciCheck.Properties.SecurityProofs.BodiesOk
