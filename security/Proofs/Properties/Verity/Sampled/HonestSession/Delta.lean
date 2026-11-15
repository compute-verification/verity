module

public import CoreProtocol.Ratio
public import Flock.Scan
public import Proofs.Flock.VBridge.Family.Names
import all Init.Data.String.Defs
import all Init.Data.String.Basic
import all Init.Data.String.Slice
import all Init.Data.String.Search

/-!
# The parameters' `delta`, `"1/2"`, parses to `⟨1, 2⟩`

`Core.Ratio.ofString "1/2" "delta" = .ok ⟨1, 2⟩` (`ratio_half`) and the one split it reads (`split_half`), which the
parameters' walk rewrites by. Kept apart from `Params`: there its wider `import all` set makes `whnf` on these one-character
literals' `decide`s run without bound, while under these imports they reduce at once.
-/

set_option maxHeartbeats 2000000

public section

namespace Verity.Sampled.Instance.HonestSession.Params

/-- A one-character numeral's `toNat?`, via `Flock.Scan.toNat?_digits` on its single digit (the kernel cannot reduce
`String.toNat?` on a byte-backed literal). -/
private theorem toNat_one : ("1" : String).toNat? = some 1 := by
  rw [show ("1" : String) = String.ofList ['1'] from by
        rw [← show ("1" : String).toList = ['1'] from by decide, String.ofList_toList],
      Flock.Scan.toNat?_digits ['1'] (by decide) (by decide)]
  decide

private theorem toNat_two : ("2" : String).toNat? = some 2 := by
  rw [show ("2" : String) = String.ofList ['2'] from by
        rw [← show ("2" : String).toList = ['2'] from by decide, String.ofList_toList],
      Flock.Scan.toNat?_digits ['2'] (by decide) (by decide)]
  decide

/-- `toNat!` from `toNat?`: on a valid numeral the `isNat` guard holds, so the panicking branch never runs. -/
private theorem toNat!_one : ("1" : String).toNat! = 1 := by
  have h := toNat_one
  unfold String.toNat? String.Slice.toNat? at h
  unfold String.toNat! String.Slice.toNat!
  split at h
  · rename_i hi; rw [if_pos hi]; exact Option.some.inj h
  · exact absurd h (by simp)

private theorem toNat!_two : ("2" : String).toNat! = 2 := by
  have h := toNat_two
  unfold String.toNat? String.Slice.toNat? at h
  unfold String.toNat! String.Slice.toNat!
  split at h
  · rename_i hi; rw [if_pos hi]; exact Option.some.inj h
  · exact absurd h (by simp)

/-- `"1/2".splitOn "/" = ["1", "2"]`, via `FlockVBridge.Names.splitOn_sep` (the one-byte separator reduces `String.splitOn`
to `List.splitOn` on the characters, which `String.splitOn` is not kernel-reducible on). The one split `delta` needs. -/
theorem split_half : "1/2".splitOn "/" = ["1", "2"] := by
  have hsplit : (['1', '/', '2'] : List Char).splitOn '/' = [['1'], ['2']] := by
    rw [show (['1', '/', '2'] : List Char) = ['1'] ++ '/' :: ['2'] from rfl]
    rw [show (['1'] ++ '/' :: ['2'] : List Char).splitOn '/' = ['1'] :: (['2'].splitOn '/') from by
          simp only [List.splitOn_eq_splitOnP]
          exact List.splitOnP_append_cons_of_forall_mem (by decide) '/' (by decide) ['2']]
    rw [show (['2'] : List Char).splitOn '/' = [['2']] from by
          rw [List.splitOn_eq_splitOnP, List.splitOnP_eq_singleton]
          all_goals decide]
  rw [FlockVBridge.Names.splitOn_sep "/" '/' rfl rfl rfl rfl (by decide) "1/2",
      show "1/2".toList = ['1', '/', '2'] from by decide, hsplit]
  decide

/-- `Core.natOf` on the one-character numerals `delta` holds. The guard (`isEmpty`/`all isDigit`/`length`) is discharged by
`String.all_bool_eq` on the single character, and `toNat!` by `toNat!_one`/`toNat!_two`. -/
private theorem natOf_one : Core.natOf "1" "delta" = .ok 1 := by
  unfold Core.natOf
  split
  · rw [toNat!_one]
  · rename_i h
    exact absurd (by rw [String.all_bool_eq, show ("1" : String).toList = ['1'] from by decide]; decide) h

private theorem natOf_two : Core.natOf "2" "delta" = .ok 2 := by
  unfold Core.natOf
  split
  · rw [toNat!_two]
  · rename_i h
    exact absurd (by rw [String.all_bool_eq, show ("2" : String).toList = ['2'] from by decide]; decide) h

/-- `Core.Ratio.ofString "1/2" "delta" = .ok ⟨1, 2⟩`: the parameters' `delta`. `split_half` clears the one `splitOn`;
`natOf_one`/`natOf_two` clear the two `natOf` reads, so the remaining `rfl` sees only the two concrete naturals, the `≤`
and the `gcd`. -/
theorem ratio_half : Core.Ratio.ofString "1/2" "delta" = .ok ⟨1, 2⟩ := by
  unfold Core.Ratio.ofString
  rw [split_half]
  simp only [natOf_one, natOf_two, bind, Except.bind]
  rfl

end Verity.Sampled.Instance.HonestSession.Params
