import Proofs.Flock.Claims.Recursion.DeployedR
import Proofs.Flock.Claims.Recursion.PartsCompose

/-!
# `finalR`'s cut over openings laid row then salt, and commit strings laid end to end

The layout-free half of `OpensSlots`: where the openings `finalR` reads are each a row then its 192-byte salt, laid
round after round, `cutRounds` at their rows' lengths gives back each round's rows and salts (`cutRounds_groups`); and
where a round's message is 128-byte commit strings end to end, `commitAt` reads the `i`-th (`commitAt_flatten`).
-/

namespace RecDeploy.CutOpens

open Claims (Msg)

theorem saltBytes_eq : Flock.Hm96.sha512.saltBytes = 192 := rfl

/-- **One round's cut gives back its rows and salts**, each row then its 192-byte salt, laid end to end. -/
theorem cutRound_pairs (rest : Msg) : ∀ l : List (Msg × Msg), (∀ p ∈ l, p.2.length = 192) →
    DeployedR.cutRound (l.map (·.1.length)) (l.flatMap (fun p => p.1 ++ p.2) ++ rest) =
      some (l.map fun p => (p.1, DeployedR.saltOf p.2), rest)
  | [], _ => rfl
  | p :: l, h => by
    have h2 : p.2.length = 192 := h p List.mem_cons_self
    have ih := cutRound_pairs rest l fun q hq => h q (List.mem_cons_of_mem _ hq)
    simp only [List.map_cons, List.flatMap_cons, DeployedR.cutRound, List.append_assoc, saltBytes_eq]
    rw [List.drop_left, List.take_left, List.drop_left' h2, List.take_left' h2, ih]
    have hc : p.1.length + 192 ≤ (p.1 ++ (p.2 ++ (l.flatMap (fun p => p.1 ++ p.2) ++ rest))).length := by
      simp only [List.length_append, h2]
      omega
    simp only [hc, ↓reduceIte]
    rfl

/-- **Every round's cut gives back each round's rows and salts**, the rounds laid one after another. -/
theorem cutRounds_groups (rest : Msg) : ∀ L : List (List (Msg × Msg)), (∀ l ∈ L, ∀ p ∈ l, p.2.length = 192) →
    DeployedR.cutRounds (L.map fun l => l.map (·.1.length)) (L.flatMap (·.flatMap fun p => p.1 ++ p.2) ++ rest) =
      some (L.map fun l => l.map fun p => (p.1, DeployedR.saltOf p.2), rest)
  | [], _ => rfl
  | l :: L, h => by
    simp only [List.map_cons, List.flatMap_cons, DeployedR.cutRounds, List.append_assoc,
      cutRound_pairs (L.flatMap (·.flatMap fun p => p.1 ++ p.2) ++ rest) l (h l List.mem_cons_self), Option.bind_some,
      cutRounds_groups rest L fun l' hl' => h l' (List.mem_cons_of_mem _ hl'), Option.map_some]

/-- **The `i`-th of 128-byte commit strings laid end to end** is what `commitAt` reads at `i`. -/
theorem commitAt_flatten : ∀ (l : List Msg), (∀ m ∈ l, m.length = 128) → ∀ i (hi : i < l.length),
    Claims.Recursion.VStarB.commitAt l.flatten i = l[i]
  | [], _, _, hi => absurd hi (by simp)
  | m :: l, h, 0, _ => by
    have hm : m.length = 128 := h m List.mem_cons_self
    simp only [Claims.Recursion.VStarB.commitAt, List.flatten_cons, Nat.mul_zero, List.drop_zero,
      List.getElem_cons_zero]
    exact List.take_left' hm
  | m :: l, h, i + 1, hi => by
    have hm : m.length = 128 := h m List.mem_cons_self
    have ih := commitAt_flatten l (fun m' hm' => h m' (List.mem_cons_of_mem _ hm')) i
      (by simpa using hi)
    simp only [Claims.Recursion.VStarB.commitAt, List.flatten_cons, List.getElem_cons_succ] at ih ⊢
    rw [show 128 * (i + 1) = m.length + 128 * i by omega, ← List.drop_drop, List.drop_left]
    exact ih

end RecDeploy.CutOpens
