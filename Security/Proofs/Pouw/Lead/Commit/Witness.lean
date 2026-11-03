import Proofs.Pouw.Lead.Commit.RowBound
import Proofs.Pouw.Lead.Commit.Rounds

/-!
# row-bound and g-private's order say something

The honest and nontrivial witnesses (the lock's `witnesses`) of `leafFirstBlock`, `rowBound` and
`roundsOpenAfterRoots`, properties of `PouwCommit.Domain.mk?` and of `PouwWindow.roundsReason` accepting.

* `domain_honest`, the honest witness of both row-bound properties: `Domain.mk?` builds the domain of a 64-byte binding,
  the weights' owner `-2` and one position.
* `leafFirstBlock_nontrivial`: a SHA-512 domain whose id is 128 bytes, which `Domain.mk?` never builds (its ids are
  SHA-512 digests), has its leaves' first block end inside the id, so its leaves at positions 0 and 1 share their first
  block: with `Domain.mk?`'s acceptance dropped, `LeafFirstBlock`'s conclusion fails.
* `rowBound_nontrivial`: at a domain with an empty id, the leaves at positions 0 and 1 are of two framed messages whose
  SHA-512 digests differ, so neither case of `RowBound`'s conclusion holds there.
* `roundsOpenAfterRoots_honest`: the round check accepts a live epoch's coin commitment, with its salt and round
  families, before any round root is recorded.
* `roundsOpenAfterRoots_nontrivial`: a coin commitment of no scheme and no families is no live epoch's, so with the
  round check's acceptance dropped `RoundsOpenAfterRoots`'s conclusion fails at it.
-/

namespace Pouw.SecurityProofs.Lead.Commit

open Properties.Pouw.Lead.Commit

/-- A SHA-512 domain whose id is 128 bytes, which `Domain.mk?` never builds. -/
local notation "D₁₂₈" => PouwCommit.Domain.mk PouwCommit.Hash.sha512 2 (ByteArray.mk (Array.replicate 128 0))

/-- A SHA-512 domain with an empty id, which `Domain.mk?` never builds. -/
local notation "D₀" => PouwCommit.Domain.mk PouwCommit.Hash.sha512 1 ByteArray.empty

/-- **`LeafFirstBlock`'s and `RowBound`'s honest witness**: `Domain.mk?` builds the SHA-512 domain of a 64-byte
binding, the weights' owner and one position. -/
theorem domain_honest :
    ∃ d : PouwCommit.Domain, PouwCommit.Domain.mk? .sha512 (ByteArray.mk (Array.replicate 64 0)) (-2) 1 = .ok d :=
  ⟨_, rfl⟩

/-- **`LeafFirstBlock`'s nontrivial witness**: at a domain whose id is 128 bytes, the leaves at positions 0 and 1, both
below `2^56`, share their first block, at two positions. -/
theorem leafFirstBlock_nontrivial :
    ∃ r : {r : ℕ × ℕ // firstBlock (leafMessage D₁₂₈ r.1 "" .empty) = firstBlock (leafMessage D₁₂₈ r.2 "" .empty) ∧
        r.1 < 2 ^ 56 ∧ r.2 < 2 ^ 56},
      ¬ r.1.1 = r.1.2 :=
  ⟨⟨(0, 1), by decide +kernel, by decide, by decide⟩, by decide⟩

/-- **`RowBound`'s nontrivial witness**: at a domain with an empty id, the leaves at positions 0 and 1 are at two
positions, and their framed messages, different, have different SHA-512 digests. -/
theorem rowBound_nontrivial :
    ¬ ((PouwCommit.Domain.id D₀ = PouwCommit.Domain.id D₀ ∧ (0 : ℕ) = 1) ∨
      (leafMessage D₀ 0 "" .empty ≠ leafMessage D₀ 1 "" .empty ∧
        Flock.Sha512.hash (leafMessage D₀ 0 "" .empty) = Flock.Sha512.hash (leafMessage D₀ 1 "" .empty))) := by
  rintro (⟨-, h⟩ | ⟨-, h⟩)
  · exact absurd h (by decide)
  · exact absurd h (by decide +kernel)

/-- **`RoundsOpenAfterRoots`'s honest witness**: the round check accepts a live epoch's coin commitment, its salt and
round families (`liveFamilies 1`; the draw coin is a session of its own), before any round root is recorded. -/
theorem roundsOpenAfterRoots_honest :
    PouwWindow.roundsReason ⟨PouwWindow.LIVE_SCHEME, 0, [(PouwWindow.SALT, 1), (PouwWindow.ROUND, 1)],
      ByteArray.mk (Array.replicate 64 0)⟩ [] [] = .ok none :=
  rfl

/-- **`RoundsOpenAfterRoots`'s nontrivial witness**: a coin commitment of no scheme and no families is no live
epoch's. -/
theorem roundsOpenAfterRoots_nontrivial : ¬ PouwWindow.liveEpochReason ⟨"", 0, [], .empty⟩ = none := by
  decide

end Pouw.SecurityProofs.Lead.Commit
