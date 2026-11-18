module

public import Proofs.Flock.SpotFlock.InstanceStmt.TraceRegions
public import Proofs.Flock.SpotFlock.InstanceStmt.TraceDa
public import Proofs.Flock.SpotFlock.InstanceStmt.TraceFacts
public import Proofs.Flock.SpotFlock.InstanceStmt.ScopeZk
public import Proofs.Flock.SessionLoop.Concrete.Live
import Flock.HmRow
import Flock.HmNets
import Flock.Canon
import Flock.JsonParse
import Flock.Net
import Flock.ZkServer
import Flock.Norm
import Flock.Public
import Flock.Statement
import Flock.Comp
import Flock.Bytes
import Flock.FV3
import Flock.Hm96
import Flock.Hash
import Flock.RowLeaf
import Flock.Circuit
import Flock.Tags
import Flock.Zk
import Flock.CoinTree
import Flock.Merkle
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic
import all Std.Data.TreeMap.Raw.Basic
import all Std.Data.DTreeMap.Raw.Basic
import all Std.Data.DTreeMap.Internal.Operations
import all Std.Data.DTreeMap.Internal.Queries
import all Std.Data.DTreeMap.Internal.Balancing
import all Std.Data.DTreeMap.Internal.Def
import all Init.Data.Ord.String
import all Init.Data.Ord.Basic
import all Init.Data.String.Defs
import all Init.Data.String.Basic
import all Init.Data.String.Extra
import all Init.Data.Array.QSort.Basic
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control
import all Init.Data.Range.Basic
import all Init.Data.Int.Repr
import all Init.Data.Repr
import all Init.Data.Nat.Fold
import all Init.Data.String.TakeDrop
import all Init.Data.String.Slice
import all Init.Data.String.Pattern.Basic
import all Init.Data.String.Pattern.String
import all Init.Data.String.Pattern.Char
import all Init.Data.String.Search
import all Init.Data.String.Iterate
import all Init.Data.String.Decode
import all Init.Data.String.Termination
import all Init.Data.Array.Attach
import all Init.WF



/-!
# The table's numbers at the fixed nets

`facts_stX` (the rows' proof's facts, given Δ), and `regionFacts_stF`: the circuit's and the sampled public file's numbers the regions read, by the kernel at the closed circuit `cF`.
-/
@[expose] public section

open Flock Flock.ZkServer RecDeploy.VStarNets FlockSoundness FlockSoundness.Discharge
open Properties.Flock.Nets (sparseOf FixedNetsBuild)
open FlockDeploy.Concrete (shaB hmB)
open SpotFlock.InstanceStmt.TraceHolds (Facts)
open SpotFlock.InstanceStmt.TraceDa (stF da_stF)
open SpotFlock.InstanceStmt.TraceRegions (RegionFacts bcP commitString_zero_I commitString_zero_O)

set_option maxHeartbeats 200000000
set_option maxRecDepth 200000

namespace SpotFlock.InstanceStmt.Rel

/-! ## The table's numbers, at any nets and hashes -/

section facts

variable (r : Run) (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 s5 s6 : String)
  (hello : ByteArray → Nat → String)

theorem facts_stX (hda : (stX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello).da.toList = TraceDelta.EX ++ TraceDelta.OX) :
    Facts (stX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello) := by
  have hc := TraceFacts.c_stX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello
  have hnets := TraceFacts.nets_cO ls lh H C A B A' B' s1
  have hiw := TraceFacts.inWords_cO ls lh H C A B A' B' s1
  have hur := TraceFacts.unitRange_cO ls lh H C A B A' B' s1
  have hsha := TraceFacts.shaRange_cO ls lh H C A B A' B' s1
  have hhm := TraceFacts.hmRange_cO ls lh H C A B A' B' s1
  have hpin := TraceFacts.pin_stX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello
  refine ⟨?_, hpin, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, hda, ?_⟩
  · rw [hc]; exact TraceFacts.kLog_cO ls lh H C A B A' B' s1
  · rw [hc]; exact TraceFacts.tmpl_cO ls lh H C A B A' B' s1
  · rw [hc]; exact TraceFacts.unit_cO ls lh H C A B A' B' s1
  · rw [hc]; exact hur.1
  · rw [hc]; exact hur.2
  · rw [hc]; exact ⟨lt_of_lt_of_eq Nat.one_pos hsha.1.symm, hsha.2.1, hsha.2.2⟩
  · rw [hc]; exact ⟨by rw [hhm.1]; norm_num, hhm.2.1, hhm.2.2.1⟩
  · rw [hc]; exact ⟨by rw [hhm.1]; norm_num, hhm.2.1, hhm.2.2.2⟩
  · rw [hc, hnets.2.1]; exact hiw.2.1
  · rw [hc, hnets.2.2]; exact hiw.2.2
  · rw [hc, hpin]; exact TraceFacts.constPairs_cO ls lh H C A B A' B' s1
  · rw [hc]; exact TraceFacts.ranges_cO ls lh H C A B A' B' s1

end facts

/-! ## What the regions read, at the fixed nets' circuit -/

theorem ports_cF : cF.ports.size = 2 := by kernel_rfl'
theorem words0_cF : cF.ports[0]!.words = 64 := by kernel_rfl'
theorem words1_cF : cF.ports[1]!.words = 1 := by kernel_rfl'
theorem perVu0_cF : cF.perVu.getD cF.hmNet 0 = 2 := by kernel_rfl'
theorem perVu1_cF : cF.perVu.getD cF.hmNet 1 = 2 := by kernel_rfl'

theorem rm_cF : ∀ p < 2, Hidden.Any.rowMsg cF.ports[p]! (List.replicate (2 * cF.ports[p]!.words) 0) =
    (HmRow.rowPrefixV2 (16 * cF.ports[p]!.words)).data.toList ++ List.replicate (2 * cF.ports[p]!.words) 0 := by
  intro p hp
  interval_cases p <;> kernel_rfl'

theorem commit_cF : ∀ p < 2, Binding.commitString
    ((HmRow.rowPrefixV2 (16 * cF.ports[p]!.words)).data.toList ++ List.replicate (2 * cF.ports[p]!.words) 0)
    (List.replicate 192 0) = (bcP p).data.toList := by
  intro p hp
  interval_cases p
  · rw [words0_cF]
    exact commitString_zero_I
  · rw [words1_cF]
    exact commitString_zero_O

theorem msgOK_cF : ∀ p < 2, ∀ j < 2 * cF.ports[p]!.words, ∀ t < 8,
    (524800 ≤ Typed.msgCol cF 0 p j t ∧ Typed.msgCol cF 0 p j t < 528896) ∨
      (262704 ≤ Typed.msgCol cF 0 p j t ∧ Typed.msgCol cF 0 p j t < 262720) := by
  decide +kernel

theorem saltOK_cF : ∀ p < 2, ∀ j < 192, ∀ t < 8,
    (1536 ≤ Discharge.Hm.saltCol cF 0 p j t ∧ Discharge.Hm.saltCol cF 0 p j t < 4096) ∨
      (263680 ≤ Discharge.Hm.saltCol cF 0 p j t ∧ Discharge.Hm.saltCol cF 0 p j t < 266240) := by
  decide +kernel

theorem dummy_cF : ∀ p < 2, cF.dummyDigests.getD p ByteArray.empty = bcP p := by
  intro p hp
  interval_cases p <;> kernel_rfl'

theorem n_stF (r : Run) : (stF r).pub.n = 1 := by kernel_rfl'

theorem dig_stF (r : Run) : ∀ p < 2, (stF r).pub.digest 0 p 2 = bcP p := by
  intro p hp
  interval_cases p <;> kernel_rfl'

theorem regionFacts_stF (r : Run) : RegionFacts (stF r).c (stF r).pub := by
  have hc : (stF r).c = cF := TraceFacts.c_stX r _ _ _ _ _ _ _ _ _ _ _ _ _ _
  rw [hc]
  exact
    { kLog := TraceFacts.kLog_cO _ _ _ _ _ _ _ _ _
      g := (TraceFacts.g_cO _ _ _ _ _ _ _ _ _).1
      upv := (TraceFacts.g_cO _ _ _ _ _ _ _ _ _).2
      nports := ports_cF
      perVu0 := perVu0_cF
      perVu1 := perVu1_cF
      rm := rm_cF
      commit := commit_cF
      msgOK := msgOK_cF
      saltOK := saltOK_cF
      n := n_stF r
      dig := dig_stF r
      dummy := dummy_cF }



end SpotFlock.InstanceStmt.Rel
