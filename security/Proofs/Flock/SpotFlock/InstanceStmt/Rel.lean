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
# The trace is a witness of the deployed relation at the two-instance statement

`rel_x`: the deployed masked relation holds of `x r` at the trace `wX r`. The verifier parses `x r` to its one table
(`parse_x`, `Scope.parse_rX`); `relS_relation` reads the relation as that table's block R1CS being satisfied by the
witness's share, which is `TraceHolds.satisfies_rows` at regions `[]` (rows and pin) and `TraceRegions.satisfies_regions_x`
at the table's regions. The table's layout is the one its hidden-output setup gives (`SpecC.layout`), its row ranges those
of its flat twin (`RowRanges.ofL`), and `facts_stX` and `regionFacts_stX` read its numbers by the kernel.
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



/-! ## The statement at the fixed nets -/

/-- The statement, parsed: its inputs' readable fields and its one table. -/
noncomputable def pF (r : Run) : _root_.Flock.Readable.Parsed :=
  { rX r (HmNets.shaLaid ()) (HmNets.hmLaid ()) circuitHash classHash with sts := #[stF r] }

/-- **The witness**: the trace `TraceVal.zIX`, one hidden value per bit. -/
noncomputable def wX (r : Run) : _root_.Claims.Flock.Wit :=
  ⟨List.ofFn fun a : Fin (2 ^ (stF r).m) => TraceVal.zIX (a.val % 2 ^ 22)⟩

theorem zOf_wX (r : Run) :
    _root_.Flock.Readable.zOf 1 (wX r) 0 (stF r).m = TraceRows.zOf TraceVal.zIX (stF r).m := by
  funext a
  simp only [_root_.Flock.Readable.zOf, wX, Nat.mul_one, Nat.add_zero, List.getD_eq_getElem?_getD,
    List.getElem?_ofFn, a.isLt, dite_true, Option.getD_some, _root_.Flock.Readable.toZ, TraceRows.zOf]

theorem parse_x (hn : FixedNetsBuild) {r : Run} (hr : r.Plain) :
    _root_.Flock.Readable.parseBytes (FlockDeploy.readOf true) (x r) = some (pF r) := by
  have hp := FlockDeploy.fixedNets_parse hn
  have hCe : classHash.isEmpty = false := classOfB_isEmpty (HmNets.shaLaid ()) (HmNets.hmLaid ())
  have hpar := parse_rX r (HmNets.shaLaid ()) (HmNets.hmLaid ()) circuitHash classHash AF BF AF' BF' s1X
    (s3X (hexOf classHash)) (s4X r (hexOf circuitHash) (hexOf classHash))
    (s5X r cF (hexOf circuitHash) (hexOf classHash)) s6X helloX hr rfl (classOf_eq _ _)
    (FlockDeploy.sha_lines hn) (FlockDeploy.hm_lines hn) hp.1.2 hp.2.2 hc1_of hc2_of
    (fun c h1 h2 => FlockDeploy.Concrete.check_of hp.1.1 hp.2.1 c h1 h2) hCe (hu1_of _)
    (hu2_of _ _ _ _ _ _ _ _ _ hCe) (hd_of _ _ _) (hd2_of _ _ _ _ _ _ _ _ _ _ _ _) hid_of hhello_of
  unfold _root_.Flock.Readable.parseBytes
  rewrite [readOf_x hn hr, Option.bind_some, show _root_.Flock.Readable.Read.parse (ins r).read = _ from hpar]
  rfl

/-- **The statement is in the deployed masked relation** with the trace as its witness. -/
theorem rel_x (hn : FixedNetsBuild) {r : Run} (hr : r.Plain) :
    _root_.Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse (x r) (wX r) := by
  have hp := FlockDeploy.fixedNets_parse hn
  have hCe : classHash.isEmpty = false := classOfB_isEmpty (HmNets.shaLaid ()) (HmNets.hmLaid ())
  have hd := setup_stX r (HmNets.shaLaid ()) (HmNets.hmLaid ()) circuitHash classHash AF BF AF' BF' s1X
    (s3X (hexOf classHash)) (s4X r (hexOf circuitHash) (hexOf classHash))
    (s5X r cF (hexOf circuitHash) (hexOf classHash)) s6X helloX hr rfl (classOf_eq _ _)
    (FlockDeploy.sha_lines hn) (FlockDeploy.hm_lines hn) hp.1.2 hp.2.2 hc1_of hc2_of
    (fun c h1 h2 => FlockDeploy.Concrete.check_of hp.1.1 hp.2.1 c h1 h2) hCe (hu1_of _)
    (hu2_of _ _ _ _ _ _ _ _ _ hCe) (hd_of _ _ _) (hd2_of _ _ _ _ _ _ _ _ _ _ _ _) hid_of hhello_of
  have hc := TraceFacts.c_stX r (HmNets.shaLaid ()) (HmNets.hmLaid ()) circuitHash classHash AF BF AF' BF' s1X
    (s3X (hexOf classHash)) (s4X r (hexOf circuitHash) (hexOf classHash))
    (s5X r cF (hexOf circuitHash) (hexOf classHash)) s6X helloX
  have ht : tz.typed = false := typed_tzX
  have hpin0 : HmRow.pinnedCount (stF r).c.metaJ = 0 := by
    rw [hc]
    exact TraceFacts.pinned_cO (HmNets.shaLaid ()) (HmNets.hmLaid ()) circuitHash classHash AF BF AF' BF' s1X
  have hC := (Hidden.setupHidden_specCC hd).committed
  have hrows : (Discharge.Layout.v1Ports (stF r).c || Discharge.Layout.bitRows (stF r).c) = true := by
    rw [hc]
    kernel_rfl'
  have hrf : Hidden.Any.RowsFitS (stF r).c := HiddenExec.rowsFit_of_committed ht hC hrows
  have L : FlockSoundness.Layout (stF r) := (Hidden.setupHidden_specCC hd).layout ht hpin0
  have R : Hm.RowRanges (stF r).c :=
    Hm.RowRanges.ofL (st := Hidden.flatTwin (stF r)) ((Hidden.setupHidden_specCC hd).flatLayout ht)
      ((Hidden.setupHidden_specCC hd).unitFacts ht).parse ((Hidden.setupHidden_specCC hd).unitFacts ht).check
  have hS := TraceRegions.satisfies_regions_x hd ht hrf hpin0 hn L R
    (facts_stX r _ _ _ _ _ _ _ _ _ _ _ _ _ _ (da_stF r)) (regionFacts_stF r)
  refine ⟨_root_.Flock.Readable.toReadable _root_.Flock.Readable.relation (pF r), ?_, ?_⟩
  · change (_root_.Flock.Readable.parseBytes (FlockDeploy.readOf true) (x r)).map
      (_root_.Flock.Readable.toReadable _root_.Flock.Readable.relation) = _
    rw [parse_x hn hr]
    rfl
  refine (_root_.Flock.Readable.relS_relation (pF r) (wX r)).2 fun c hc' => ?_
  have h1 : (pF r).sts.size = 1 := rfl
  obtain rfl : c = 0 := by omega
  have hz : _root_.Flock.Readable.zOf (pF r).sts.size (wX r) 0 (pF r).sts[0]!.m =
      TraceRows.zOf TraceVal.zIX (stF r).m := zOf_wX r
  refine ⟨⟨rfl, rfl, (L.model []).hk6, (L.model []).hk, (L.model []).pin.isLt⟩, ?_⟩
  rw [hz]
  exact (hS : (L.model (Refine.regionsOf (stF r))).Satisfies (TraceRows.zOf TraceVal.zIX (stF r).m))

end SpotFlock.InstanceStmt.Rel
