module

public import Definitions.Core.Rblk
public import Proofs.Core.Circuit

/-!
# `Rblk`'s lemmas

`rblk_iff`: `Rblk` is the openings at in(S) and out(S) and the block's correctness (`Partition.Correct`).
`rblk_congr`: it reads the trace only at in(S) and out(S).
-/

@[expose] public section

namespace Verity

namespace Partition

variable {C : Circuit} {n : ℕ} (P : Partition C n)

theorem rblk_iff {α : Type} (Sc : Hash.StringCommitment α) (H : Hash.Msg → Digest)
    (enc : (g : Fin C.N) → BitVec (C.w g) → α) (L : Layout C) (c : Root → Sc.Com) (u : Fin n) (X : C.Trace)
    (π : Fin C.N → Sc.Op) :
    P.Rblk Sc H enc L c u X π ↔
      (∀ g ∈ C.into (P.gates u) ∪ C.exits (P.gates u),
          ∃ p, P.pos L g = some p ∧ Sc.Opens H (c p.1) p.2 (enc g (X g)) (π g)) ∧
        P.Correct X u :=
  and_congr Iff.rfl (P.restrict_run_iff_correct X u)

/-- **The free inputs are free**: `Rblk` reads the trace only at in(S) and out(S), so every other gate, a gate outside
`S` that `S` never reads among them, takes any value without changing it. -/
theorem rblk_congr {α : Type} (Sc : Hash.StringCommitment α) (H : Hash.Msg → Digest)
    (enc : (g : Fin C.N) → BitVec (C.w g) → α) (L : Layout C) (c : Root → Sc.Com) (u : Fin n) {X X' : C.Trace}
    (π : Fin C.N → Sc.Op) (hX : ∀ g ∈ C.into (P.gates u) ∪ C.exits (P.gates u), X g = X' g) :
    P.Rblk Sc H enc L c u X π ↔ P.Rblk Sc H enc L c u X' π := by
  have hio : ∀ g ∈ P.io u, X g = X' g := fun g hg => hX g (P.into_union_exits_eq_io u ▸ hg)
  rw [P.rblk_iff, P.rblk_iff]
  refine and_congr ?_ ⟨fun h => P.correct_congr (fun g hg => (hio g hg).symm) h, P.correct_congr hio⟩
  refine forall₂_congr fun g hg => ?_
  rw [hX g hg]

end Partition

end Verity
