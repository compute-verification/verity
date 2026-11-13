module

public import Proofs.Properties.Verity.SampledTwo.FirstCoins
public import Proofs.Properties.Verity.Sampled.Assembled

/-!
# What a stage's read of the device's log shows of its coins

Where `drawCoins` reads coins `ξ` from a stage's coin evidence, in family `label` after `after`, and the read of its
session's log lines places them (`readLog`, as the cut's `readsLogs` reads each session), every log that read matches
release by release (`Binding.Read.Of`, as the cut's `LogIsDevice` asks of the play's log) has exactly `ξ`'s bits as its
coins of that family after `after` (`drawn_shown`, from the read's own coins, `drawn_fresh`); a log whose read is
followed by other releases has them first (`drawn_shown_append`); and one whose read follows releases holding none of
that family's coins after `after` has exactly them (`drawn_shown_after`). With the bytes a stage read a prefix of `ξ` (`bitsOf_extract_prefix`), these give the stages'
premises `FirstShown` and `SecondShown` from the cut's `LogIsDevice`.
-/

@[expose] public section

namespace Verity.SampledTwo

open Lean (Json)
open Verity.Sampled (freshCoins readLog)

/-- The bits of a byte string's first bytes start its bits. -/
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

/-- **A sample's read holds its coins**: where `drawCoins` reads coins `ξ` from coin evidence `coins` under a key, a
window, family `label` after `after`, and the read of its session's log lines places them (`readLog`), the read's
coins of the family released after `after` are `ξ`'s bits, none of them empty. -/
theorem drawn_fresh {key : Spot.Challenger.ChallengerKey} {w label : String} {after : ByteArray} {coins : Json}
    {ξ : ByteArray} {records : List Json} {ls : Array Json} {lines : List String} {l : List Verity.Release}
    (hd : Spot.Challenger.drawCoins key w label after coins = .ok ξ)
    (hlog : Spot.getD (Spot.getD coins "session") "log" = .arr ls) (hls : ls.toList = lines.map .str)
    (hr : readLog label after records ξ lines = some l) :
    (freshCoins label after l).flatten = Properties.Challenger.bitsOf ξ ∧ ∀ cs ∈ freshCoins label after l, cs ≠ [] := by
  obtain ⟨cm, roots, hts, hstream⟩ := Verity.Sampled.drawCoins_ok hd
  obtain ⟨_, hsn⟩ := Verity.Sampled.trustedSessionOf_ok hts
  obtain ⟨_, hso⟩ := Verity.Sampled.sessionNamed_ok hsn
  obtain ⟨ls', lines', hlog', hls', hrr⟩ := Verity.Sampled.sessionOf_ok hso
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

/-- A log's coins of a family are its parts' in order. -/
theorem freshCoins_append (label : String) (after : ByteArray) (a b : List Verity.Release) :
    freshCoins label after (a ++ b) = freshCoins label after a ++ freshCoins label after b := by
  unfold freshCoins
  rw [List.filterMap_append]

/-- **A sample's read shows its coins**: every log the read of `drawn_fresh` matches release by release has `ξ`'s bits as
its coins of the family released after `after`. -/
theorem drawn_shown {key : Spot.Challenger.ChallengerKey} {w label : String} {after : ByteArray} {coins : Json}
    {ξ : ByteArray} {records : List Json} {ls : Array Json} {lines : List String} {l olog : List Verity.Release}
    (hd : Spot.Challenger.drawCoins key w label after coins = .ok ξ)
    (hlog : Spot.getD (Spot.getD coins "session") "log" = .arr ls) (hls : ls.toList = lines.map .str)
    (hr : readLog label after records ξ lines = some l)
    (hO : List.Forall₂ Verity.Sampled.Binding.Read.Of (l.map Verity.Sampled.Binding.Read.ofV2) olog) :
    (freshCoins label after olog).flatten = Properties.Challenger.bitsOf ξ := by
  obtain ⟨hf, hne⟩ := drawn_fresh hd hlog hls hr
  rw [Verity.Sampled.freshCoins_ofV2 label after hO hne, hf]

/-- **A sample's read shows its coins after releases of other families**: where releases `l₀` that hold no coins of the
family after `after` come before the read, a log the two match release by release has `ξ`'s bits as those coins. -/
theorem drawn_shown_after {key : Spot.Challenger.ChallengerKey} {w label : String} {after : ByteArray}
    {coins : Json} {ξ : ByteArray} {records : List Json} {ls : Array Json} {lines : List String}
    {l₀ l olog : List Verity.Release}
    (hd : Spot.Challenger.drawCoins key w label after coins = .ok ξ)
    (hlog : Spot.getD (Spot.getD coins "session") "log" = .arr ls) (hls : ls.toList = lines.map .str)
    (hr : readLog label after records ξ lines = some l) (h₀ : freshCoins label after l₀ = [])
    (hO : List.Forall₂ Verity.Sampled.Binding.Read.Of ((l₀ ++ l).map Verity.Sampled.Binding.Read.ofV2) olog) :
    (freshCoins label after olog).flatten = Properties.Challenger.bitsOf ξ := by
  obtain ⟨hf, hne⟩ := drawn_fresh hd hlog hls hr
  have happ : freshCoins label after (l₀ ++ l) = freshCoins label after l := by
    rw [freshCoins_append, h₀, List.nil_append]
  have hne' : ∀ cs ∈ freshCoins label after (l₀ ++ l), cs ≠ [] := by
    rw [happ]
    exact hne
  rw [Verity.Sampled.freshCoins_ofV2 label after hO hne', happ, hf]

/-- **A sample's read shows its coins first** in a log it matches the start of: there, `ξ`'s bits start the log's coins of
the family released after `after`. -/
theorem drawn_shown_append {key : Spot.Challenger.ChallengerKey} {w label : String} {after : ByteArray}
    {coins : Json} {ξ : ByteArray} {records : List Json} {ls : Array Json} {lines : List String}
    {l l' olog : List Verity.Release}
    (hd : Spot.Challenger.drawCoins key w label after coins = .ok ξ)
    (hlog : Spot.getD (Spot.getD coins "session") "log" = .arr ls) (hls : ls.toList = lines.map .str)
    (hr : readLog label after records ξ lines = some l)
    (hO : List.Forall₂ Verity.Sampled.Binding.Read.Of ((l ++ l').map Verity.Sampled.Binding.Read.ofV2) olog) :
    Properties.Challenger.bitsOf ξ <+: (freshCoins label after olog).flatten := by
  rw [List.map_append] at hO
  have h₁ := List.forall₂_take (l.map Verity.Sampled.Binding.Read.ofV2).length hO
  rw [List.take_left] at h₁
  rw [← List.take_append_drop (l.map Verity.Sampled.Binding.Read.ofV2).length olog, freshCoins_append,
    List.flatten_append,
    drawn_shown hd hlog hls hr h₁]
  exact List.prefix_append _ _

end Verity.SampledTwo
