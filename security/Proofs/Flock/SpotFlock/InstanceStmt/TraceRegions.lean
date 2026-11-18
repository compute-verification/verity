module

public import Proofs.Flock.SpotFlock.InstanceStmt.TraceHolds
public import Proofs.Flock.SessionLoop.Concrete.Regions
public import Proofs.Flock.Soundness.Discharge.HiddenExec.SiteH
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Flat
public import Proofs.Flock.Soundness.Discharge.Hidden.Public
public import Proofs.Flock.Recursive.InnerRows
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
# The trace satisfies the two-instance table's public regions

`HmOut.regions` makes the table's two regions, "digest 0" and "digest 1": row `p`'s `b ‖ c` at its `hm96` slot's output.
The public file's digest and the circuit's pinned dummy are both the zero row's commit string, `bcI` and `bcO`
(`bcChunk_bcP`). At the trace both rows' message bits and salts are 0 (`msgOK`, `saltOK`), so the row's `b ‖ c` in a witness
that satisfies the rows (`AcceptedH.row_bc`, at an accepted statement at no regions) is the commit string of the zero
row under the zero salt (`commitString_zero_I`, `commitString_zero_O`), whose bytes are the public ones; so the trace holds
each region's bytes (`zIX_bcCol`, `satisfies_regions_x`).
-/

@[expose] public section

open Flock FlockLevel3 FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Hm
open FlockDeploy.Concrete.Trace (bytesOf_false mask_zeros xorBytes_zeros bit_ite lt22 bit_inj')
open SpotFlock.InstanceStmt.TraceRows (zOf block_zOf chunkBit_zOf)
open SpotFlock.InstanceStmt.TraceVal (zIX zIX_msg zIX_sha_in zIX_hm0_zero zIX_hm1_zero)
open SpotFlock.InstanceStmt.TraceHolds (Facts satisfies_rows)

set_option maxRecDepth 200000

namespace SpotFlock.InstanceStmt.TraceRegions

/-- Row `p`'s public `b ‖ c`: the zero row's commit string. -/
def bcP (p : ℕ) : ByteArray := if p = 0 then bcI else bcO

theorem commitString_zero_I :
    Binding.commitString ((HmRow.rowPrefixV2 1024).data.toList ++ List.replicate 128 0) (List.replicate 192 0) =
      bcI.data.toList := by
  unfold Binding.commitString
  rw [mask_zeros]
  have h64 : (Binding.digBytes (Refine.H512 ((HmRow.rowPrefixV2 1024).data.toList ++ List.replicate 128 0))).length = 64 :=
    Refine.hL512 _
  have hx := xorBytes_zeros (Binding.digBytes (Refine.H512 ((HmRow.rowPrefixV2 1024).data.toList ++ List.replicate 128 0)))
  rw [h64] at hx
  rw [hx]
  decide +kernel

theorem commitString_zero_O :
    Binding.commitString ((HmRow.rowPrefixV2 16).data.toList ++ List.replicate 2 0) (List.replicate 192 0) =
      bcO.data.toList := by
  unfold Binding.commitString
  rw [mask_zeros]
  have h64 : (Binding.digBytes (Refine.H512 ((HmRow.rowPrefixV2 16).data.toList ++ List.replicate 2 0))).length = 64 :=
    Refine.hL512 _
  have hx := xorBytes_zeros (Binding.digBytes (Refine.H512 ((HmRow.rowPrefixV2 16).data.toList ++ List.replicate 2 0)))
  rw [h64] at hx
  rw [hx]
  decide +kernel

theorem bcI_size : bcI.size = 128 := by decide +kernel
theorem bcO_size : bcO.size = 128 := by decide +kernel

theorem bcP_size (p : ℕ) : (bcP p).size = 128 := by
  unfold bcP
  split
  · exact bcI_size
  · exact bcO_size

/-- What the regions' proof reads of the circuit and the sampled public file. -/
structure RegionFacts (c : Circuit) (pub : Public) : Prop where
  kLog : c.kLog = 22
  g : c.g = 1
  upv : c.upv = 1
  nports : c.ports.size = 2
  perVu0 : c.perVu.getD c.hmNet 0 = 2
  perVu1 : c.perVu.getD c.hmNet 1 = 2
  rm : ∀ p < 2, Hidden.Any.rowMsg c.ports[p]! (List.replicate (2 * c.ports[p]!.words) 0) =
    (HmRow.rowPrefixV2 (16 * c.ports[p]!.words)).data.toList ++ List.replicate (2 * c.ports[p]!.words) 0
  commit : ∀ p < 2, Binding.commitString
    ((HmRow.rowPrefixV2 (16 * c.ports[p]!.words)).data.toList ++ List.replicate (2 * c.ports[p]!.words) 0)
    (List.replicate 192 0) = (bcP p).data.toList
  msgOK : ∀ p < 2, ∀ j < 2 * c.ports[p]!.words, ∀ t < 8,
    (524800 ≤ Typed.msgCol c 0 p j t ∧ Typed.msgCol c 0 p j t < 528896) ∨
      (262704 ≤ Typed.msgCol c 0 p j t ∧ Typed.msgCol c 0 p j t < 262720)
  saltOK : ∀ p < 2, ∀ j < 192, ∀ t < 8,
    (1536 ≤ saltCol c 0 p j t ∧ saltCol c 0 p j t < 4096) ∨
      (263680 ≤ saltCol c 0 p j t ∧ saltCol c 0 p j t < 266240)
  n : pub.n = 1
  dig : ∀ p < 2, pub.digest 0 p 2 = bcP p
  dummy : ∀ p < 2, c.dummyDigests.getD p ByteArray.empty = bcP p

section facts

variable {c : Circuit} {pub : Public} (F : RegionFacts c pub)
include F

/-- **Region "digest p"'s bytes at every block**: the public digest or the pinned dummy, both row `p`'s zero-row
commit string. -/
theorem bcChunk_bcP {p : ℕ} (hp : p < 2) (b : ℕ) : Discharge.Layout.bcChunk c pub b 0 p = bcP p := by
  unfold Discharge.Layout.bcChunk
  cases hi : c.instOf pub.n b 0 with
  | none => exact F.dummy p hp
  | some i =>
    have hi0 : i = 0 := by
      unfold Circuit.instOf at hi
      simp only [F.g, F.n] at hi
      simp at hi
      omega
    subst hi0
    show pub.digest 0 p c.ports.size = bcP p
    rw [F.nports]
    exact F.dig p hp

/-- **The trace's `b ‖ c`** at row `p`'s `hm96` slot's output, given the row's commit string there: the row's message and
salt are 0 in the trace, so it is the zero row's. -/
theorem zIX_bcCol {w : ℕ → Bool} (hw : ∀ y < 2 ^ 22, w y = zIX y) {p : ℕ} (hp : p < 2)
    (hbc : Discharge.Hm.bytesOf w 128 (bcCol c 0 p) = Binding.commitString
      (Hidden.Any.rowMsg c.ports[p]! (Discharge.Hm.bytesOf w (2 * c.ports[p]!.words) (Typed.msgCol c 0 p)))
      (Discharge.Hm.bytesOf w 192 (saltCol c 0 p)))
    {q : ℕ} (hq : q < 1024) (h22 : bcCol c 0 p (q / 8) (q % 8) < 2 ^ 22) :
    zIX (bcCol c 0 p (q / 8) (q % 8)) = Discharge.Hm.bitL (bcP p).data.toList q := by
  have hmsg : Discharge.Hm.bytesOf w (2 * c.ports[p]!.words) (Typed.msgCol c 0 p) = List.replicate (2 * c.ports[p]!.words) 0 := by
    refine FlockDeploy.Concrete.Trace.bytesOf_false fun j hj t ht => ?_
    have h := F.msgOK p hp j hj t ht
    rw [hw _ (lt22 (by omega))]
    rcases h with ⟨h1, h2⟩ | ⟨h1, h2⟩
    · exact zIX_sha_in h1 h2
    · exact zIX_msg h1 h2
  have hsalt : Discharge.Hm.bytesOf w 192 (saltCol c 0 p) = List.replicate 192 0 := by
    refine FlockDeploy.Concrete.Trace.bytesOf_false fun j hj t ht => ?_
    have h := F.saltOK p hp j hj t ht
    rw [hw _ (lt22 (by omega))]
    rcases h with ⟨h1, h2⟩ | ⟨h1, h2⟩
    · exact zIX_hm0_zero h1 h2
    · exact zIX_hm1_zero h1 h2
  rw [hmsg, hsalt, F.rm p hp, F.commit p hp] at hbc
  have hb := Discharge.Hm.bitL_bytesOf w (bcCol c 0 p) (N := 128) (show q < 8 * 128 by omega)
  rw [hbc, hw _ h22] at hb
  exact hb.symm

end facts

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}

/-- **The trace satisfies the table at its regions**, where it satisfies its rows and pin and the statement is one
`Stmt.setupHidden` accepts: at each region "digest p" the trace holds the public bytes. -/
theorem satisfies_regions_x
    (hd : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) (ht : tags.typed = false)
    (hrf : Hidden.Any.RowsFitS st.c) (hpin0 : HmRow.pinnedCount st.c.metaJ = 0)
    (hn : Properties.Flock.Nets.FixedNetsBuild) (L : FlockSoundness.Layout st) (R : RowRanges st.c) (F : Facts st)
    (RF : RegionFacts st.c st.pub) :
    (L.model (Refine.regionsOf st)).Satisfies (zOf zIX (L.model (Refine.regionsOf st)).m) := by
  have hz := satisfies_rows L R F hn
  have s := Hidden.setupHidden_spec' hd
  have hu : Hidden.unpin st = st := Flock.InnerRows.unpin_self s.da s.db hpin0
  have hregs : HmOut.regions st.c st.pub = .ok st.regions := (Hidden.setupHidden_specCC hd).self.regions
  obtain ⟨hsz, hall⟩ := Hidden.regions_get hregs
  obtain ⟨T, hTdef⟩ : ∃ T : Model.Statement, T = (Hidden.unpin_layout ht hd).model [] := ⟨_, rfl⟩
  have hTe : T = L.model [] := by
    rw [hTdef]
    exact Flock.InnerRows.model_congr hu L _ [] [] HEq.rfl
  have hzT : T.Satisfies (zOf zIX T.m) := by
    rw [hTe]
    exact hz
  have hkT : T.kLog = 22 := by rw [hTe]; exact RF.kLog
  have hbcVal : ∀ p, p < 2 → ∀ q < 1024, bcCol st.c 0 p (q / 8) (q % 8) < 2 ^ 22 →
      zIX (bcCol st.c 0 p (q / 8) (q % 8)) = Discharge.Hm.bitL (bcP p).data.toList q := by
    intro p hp q hq h22
    have o : Fin (2 ^ T.nbl) := ⟨0, Nat.two_pow_pos _⟩
    have hbc : Discharge.Hm.bytesOf (chunkBit T (zOf zIX T.m) o) 128 (bcCol st.c 0 p) = Binding.commitString
        (Hidden.Any.rowMsg st.c.ports[p]!
          (Discharge.Hm.bytesOf (chunkBit T (zOf zIX T.m) o) (2 * st.c.ports[p]!.words) (Typed.msgCol st.c 0 p)))
        (Discharge.Hm.bytesOf (chunkBit T (zOf zIX T.m) o) 192 (saltCol st.c 0 p)) :=
      HiddenExec.AcceptedH.row_bc (T := T)
        ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, Hidden.setupHidden_specCC hd,
          Hidden.unpin_layout ht hd, HiddenExec.unitRowsFlat_unpin ht hd hrf.1.toP, [], hTdef.symm⟩
        (show 0 < st.c.upv by rw [RF.upv]; omega) hrf.toP hzT o (g := 0) (p := p)
        (show 0 < st.c.g by rw [RF.g]; omega) (show p < st.c.ports.size by rw [RF.nports]; exact hp)
    exact zIX_bcCol RF (fun y hy => TraceRows.chunkBit_zOf (S := T) hkT o hy) hp hbc hq h22
  refine ⟨hz.1, hz.2.1, ?_⟩
  intro R' hR' o i hi h
  have hR'' : R' ∈ st.regions.toList.map (Refine.regionOf st.c.kLog (st.m - st.c.kLog)) := hR'
  obtain ⟨reg, hreg, rfl⟩ := List.mem_map.1 hR''
  obtain ⟨p, hp, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 hreg)
  have hp2 : p < 2 := by rw [hsz] at hp; rw [← RF.nports]; exact hp
  have hmk := hall p hp
  suffices key : zIX ((Refine.regionOf st.c.kLog (st.m - st.c.kLog) st.regions[p]).inBlock i) =
      ((st.regions[p].bytes o.val).get! (i / 8)).toNat.testBit (i % 8) by
    apply bit_inj'
    exact (congrArg FlockSoundness.bit (TraceRows.block_zOf (S := L.model (Refine.regionsOf st)) RF.kLog o ⟨_, h⟩)).trans
      ((bit_ite _).trans (key.trans (Discharge.Layout.regionOf_bit st.c.kLog (st.m - st.c.kLog) st.regions[p] o i).symm))
  obtain ⟨hrf', -, -, -, -, hlt⟩ := Discharge.Layout.mkRegion_spec hmk
  have hfree : st.regions[p].free = Discharge.Layout.hmFree st.c := by rw [hrf']
  have hbytes : st.regions[p].bytes = Discharge.Layout.hmBytes st.c st.pub p := by rw [hrf']
  have hlen : (Refine.regionOf st.c.kLog (st.m - st.c.kLog) st.regions[p]).free.length = 10 := by
    rw [← List.length_map (f := Fin.val), Discharge.Layout.regionOf_free_map _ _ _ (fun b hb => hlt b (by
      rw [hfree] at hb; exact Array.mem_toList_iff.1 hb)), hfree, Discharge.Layout.hmFree_toList, RF.g,
      FlockDeploy.Concrete.Trace.log2_one]
    simp
  have hi0 : i < 2 ^ (Refine.regionOf st.c.kLog (st.m - st.c.kLog) st.regions[p]).free.length := hi
  rw [hlen] at hi0
  have hi' : i < 1024 := lt_of_lt_of_eq hi0 (by norm_num)
  have hpv : st.c.perVu.getD st.c.hmNet 0 = 2 ^ (st.c.perVu.getD st.c.hmNet 1).log2 := by
    rw [RF.perVu0, RF.perVu1, show Nat.log2 2 = 1 from Nat.log2_two_pow (n := 1)]
    norm_num
  have hib := Discharge.Layout.inBlock_digest (st.m - st.c.kLog) hmk (g := 0) (j := i / 8) (t := i % 8)
    (Nat.two_pow_pos _) (by omega) (Nat.mod_lt _ (by norm_num))
  rw [show 0 * 1024 + (8 * (i / 8) + i % 8) = i by omega] at hib
  have hcol : (Refine.regionOf st.c.kLog (st.m - st.c.kLog) st.regions[p]).inBlock i =
      bcCol st.c 0 p (i / 8) (i % 8) := hib.trans (Discharge.Layout.bcAddr_eq_slotCol _ _ _ _ _ hpv)
  have h22 : bcCol st.c 0 p (i / 8) (i % 8) < 2 ^ 22 := by
    rw [← hcol]
    exact lt_of_lt_of_eq h (by show 2 ^ st.c.kLog = 2 ^ 22; rw [RF.kLog])
  have hsz' : ∀ g' ≤ 0, (Discharge.Layout.bcChunk st.c st.pub o.val g' p).size = 128 := by
    intro g' hg'
    obtain rfl : g' = 0 := by omega
    exact (congrArg ByteArray.size (bcChunk_bcP RF hp2 o.val)).trans (bcP_size p)
  have hbyte : (st.regions[p].bytes o.val).get! (i / 8) = (bcP p).data.toList.getD (i / 8) default := by
    rw [hbytes]
    have := Discharge.Layout.hmBytes_get (c := st.c) (pub := st.pub) (p := p) (b := o.val) (g := 0) (j := i / 8)
      (by rw [RF.g]; omega) (by omega) hsz'
    rw [Nat.zero_mul, Nat.zero_add, bcChunk_bcP RF hp2, Discharge.Layout.get!_eq_getD (bcP p)] at this
    exact this
  rw [hcol, hbcVal p hp2 i hi' h22, hbyte]
  rfl

end SpotFlock.InstanceStmt.TraceRegions
