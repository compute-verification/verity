module

public import Proofs.Properties.Verity.SampledTwo.FirstCoins
public import Proofs.Properties.Verity.Sampled.Assembled

/-! What a stage's read of the device's log shows of the coins `sampleCoins` read. -/

@[expose] public section

namespace Verity.SampledTwo

open Lean (Json)
open Verity.Sampled (freshCoins readLog)

theorem bitsOf_extract_prefix (b : ByteArray) (n : ℕ) :
    Properties.Challenger.bitsOf (b.extract 0 n) <+: Properties.Challenger.bitsOf b := by
  by_cases hn : n ≤ b.size
  · have hb : b = b.extract 0 n ++ b.extract n b.size := by
      rw [ByteArray.extract_append_extract, Nat.zero_min, Nat.max_eq_right hn, ByteArray.extract_zero_size]
    conv_rhs => rw [hb]
    rw [Verity.Sampled.bitsOf_append]
    exact List.prefix_append _ _
  · rw [extract_of_size_le (by omega)]
    all_goals exact List.prefix_refl _

theorem sampled_fresh {key : Spot.Challenger.ChallengerKey} {w label : String} {after : ByteArray} {coins : Json}
    {ξ : ByteArray} {records : List Json} {ls : Array Json} {lines : List String} {l : List Verity.Release}
    (hd : Spot.Challenger.sampleCoins key w label after coins = .ok ξ)
    (hlog : Spot.getD (Spot.getD coins "session") "log" = .arr ls) (hls : ls.toList = lines.map .str)
    (hr : readLog label after records ξ lines = some l) :
    (freshCoins label after l).flatten = Properties.Challenger.bitsOf ξ ∧ ∀ cs ∈ freshCoins label after l, cs ≠ [] := by
  obtain ⟨cm, roots, hts, hstream⟩ := Verity.Sampled.drawCoins_ok hd
  obtain ⟨_, hsn⟩ := Verity.Sampled.trustedChallengePhaseOf_ok hts
  obtain ⟨_, hso⟩ := Verity.Sampled.challengePhaseNamed_ok hsn
  obtain ⟨ls', lines', hlog', hls', hrr⟩ := Verity.Sampled.challengePhaseOf_ok hso
  rw [hlog] at hlog'
  cases hlog'
  rw [hls] at hls'
  have hlines : lines' = lines :=
    ((List.map_injective_iff.2 fun _ _ h => Json.str.inj h) hls').symm
  subst hlines
  have hsize := Verity.Sampled.challengerStream_ok hstream
  have hfresh := Verity.Sampled.fresh_readLog _ _ _ _ hr
  rw [Verity.Sampled.countFresh_eq, ← Verity.Sampled.releasedRootsOf_ok hrr] at hfresh
  rw [hfresh]
  exact ⟨Verity.Sampled.chunks_flatten _ _ hsize, Verity.Sampled.chunks_ne_nil_of_size _ _ hsize⟩

theorem freshCoins_append (label : String) (after : ByteArray) (a b : List Verity.Release) :
    freshCoins label after (a ++ b) = freshCoins label after a ++ freshCoins label after b := by
  unfold freshCoins
  rw [List.filterMap_append]

theorem sampled_shown {key : Spot.Challenger.ChallengerKey} {w label : String} {after : ByteArray} {coins : Json}
    {ξ : ByteArray} {records : List Json} {ls : Array Json} {lines : List String} {l olog : List Verity.Release}
    (hd : Spot.Challenger.sampleCoins key w label after coins = .ok ξ)
    (hlog : Spot.getD (Spot.getD coins "session") "log" = .arr ls) (hls : ls.toList = lines.map .str)
    (hr : readLog label after records ξ lines = some l)
    (hO : List.Forall₂ Verity.Sampled.Binding.Read.Of (l.map Verity.Sampled.Binding.Read.ofV2) olog) :
    (freshCoins label after olog).flatten = Properties.Challenger.bitsOf ξ := by
  obtain ⟨hf, hne⟩ := sampled_fresh hd hlog hls hr
  rw [Verity.Sampled.freshCoins_ofV2 label after hO hne, hf]

theorem sampled_shown_after {key : Spot.Challenger.ChallengerKey} {w label : String} {after : ByteArray}
    {coins : Json} {ξ : ByteArray} {records : List Json} {ls : Array Json} {lines : List String}
    {l₀ l olog : List Verity.Release}
    (hd : Spot.Challenger.sampleCoins key w label after coins = .ok ξ)
    (hlog : Spot.getD (Spot.getD coins "session") "log" = .arr ls) (hls : ls.toList = lines.map .str)
    (hr : readLog label after records ξ lines = some l) (h₀ : freshCoins label after l₀ = [])
    (hO : List.Forall₂ Verity.Sampled.Binding.Read.Of ((l₀ ++ l).map Verity.Sampled.Binding.Read.ofV2) olog) :
    (freshCoins label after olog).flatten = Properties.Challenger.bitsOf ξ := by
  obtain ⟨hf, hne⟩ := sampled_fresh hd hlog hls hr
  have happ : freshCoins label after (l₀ ++ l) = freshCoins label after l := by
    rw [freshCoins_append, h₀, List.nil_append]
  have hne' : ∀ cs ∈ freshCoins label after (l₀ ++ l), cs ≠ [] := by
    rw [happ]
    exact hne
  rw [Verity.Sampled.freshCoins_ofV2 label after hO hne', happ, hf]

theorem sampled_shown_append {key : Spot.Challenger.ChallengerKey} {w label : String} {after : ByteArray}
    {coins : Json} {ξ : ByteArray} {records : List Json} {ls : Array Json} {lines : List String}
    {l l' olog : List Verity.Release}
    (hd : Spot.Challenger.sampleCoins key w label after coins = .ok ξ)
    (hlog : Spot.getD (Spot.getD coins "session") "log" = .arr ls) (hls : ls.toList = lines.map .str)
    (hr : readLog label after records ξ lines = some l)
    (hO : List.Forall₂ Verity.Sampled.Binding.Read.Of ((l ++ l').map Verity.Sampled.Binding.Read.ofV2) olog) :
    Properties.Challenger.bitsOf ξ <+: (freshCoins label after olog).flatten := by
  rw [List.map_append] at hO
  have h₁ := List.forall₂_take (l.map Verity.Sampled.Binding.Read.ofV2).length hO
  rw [List.take_left] at h₁
  rw [← List.take_append_drop (l.map Verity.Sampled.Binding.Read.ofV2).length olog, freshCoins_append,
    List.flatten_append,
    sampled_shown hd hlog hls hr h₁]
  exact List.prefix_append _ _

end Verity.SampledTwo
