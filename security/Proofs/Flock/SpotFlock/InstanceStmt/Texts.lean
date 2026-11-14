module

public import Proofs.Flock.SpotFlock.InstanceStmt.File
import Proofs.Flock.Verifier.Hash
import Flock.HmRow
import Flock.Canon
import Flock.Statement
import Flock.Tags
import Flock.Zk
import Flock.CoinTree
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
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control
import all Init.Data.Range.Basic
import all Init.Data.Int.Repr
import all Init.Data.Repr
import all Init.Data.Nat.Fold
import all Init.Data.Array.Attach

/-! # The two-instance statement's canonical texts and hello

`setup_stX` takes as hypotheses the texts `canon` and `canonUtf8` write for META's leaf scheme and the pinned one
(`hc1`, `hc2`), the header's units (`hu1`, `hu2`), the public file's header (`hd`), the sample's header `hdrD` (`hd2`)
and the identity (`hid`), and the hello (`hhello`). Each exists because its value holds only integers
(`FlockDeploy.Concrete.IntJ`), and each is named (`s1X`, `s3X`, `s4X`, `s5X`, `s6X`, `helloX`) so that no walk
evaluates it. Two objects holding the same entries have one text: META's leaf scheme and the pinned one (`ls_arr`), and
the circuit's units entry and the header's units (`units_arr`), the classes among them because the class hash is a
SHA-512 and so not empty (`hCe_of`, `classOfB_isEmpty`).
-/

@[expose] public section
open Flock
open FlockDeploy.Concrete (IntJ canon_eq canonUtf8_eq canon_ok ok_of_isOk canon_obj_congr objM)
open Lean (Json)

namespace SpotFlock.InstanceStmt

theorem hCe_of (m : ByteArray) : (Sha512.hash m).isEmpty = false := by
  simp [ByteArray.isEmpty, Sha512.hash_size]

theorem classOfB_isEmpty (ls lh : HmNets.Laid) : (classOfB ls lh).isEmpty = false := by
  obtain ⟨m, hm⟩ := classOfB_hash ls lh
  rw [hm]
  exact hCe_of m

/-! ## The units -/

def s3X (cls : String) : String := (canon (unitsJ cls)).toOption.getD ""

theorem intJ_units (cls : String) : IntJ (unitsJ cls) = true := by kernel_rfl

theorem hu1_of (cls : String) : canon (unitsJ cls) = .ok (s3X cls) := canon_eq (intJ_units cls)

theorem units_arr (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 : String) (hCe : C.isEmpty = false) :
    (objM (HmRow.unitsEntry { cO ls lh H C A B A' B' s1 with outPorts := #[] } #[0, 1])).toArray =
      (objM (unitsJ (hexOf C))).toArray := by
  simp only [HmRow.unitsEntry]
  split
  · next hc => exact absurd (hc.symm.trans hCe) (by decide)
  · kernel_rfl'

theorem hu2_of (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 : String) (hCe : C.isEmpty = false) :
    canon (HmRow.unitsEntry { cO ls lh H C A B A' B' s1 with outPorts := #[] } #[0, 1]) = .ok (s3X (hexOf C)) := by
  rw [show HmRow.unitsEntry { cO ls lh H C A B A' B' s1 with outPorts := #[] } #[0, 1] =
      .obj (objM (HmRow.unitsEntry { cO ls lh H C A B A' B' s1 with outPorts := #[] } #[0, 1])) from rfl,
    canon_obj_congr (units_arr ls lh H C A B A' B' s1 hCe),
    show Json.obj (objM (unitsJ (hexOf C))) = unitsJ (hexOf C) from rfl]
  exact hu1_of _

/-! ## The leaf scheme: META's and the pinned one -/

def s1X : String := (canon pinJ).toOption.getD ""

theorem hc2_of : canon pinJ = .ok s1X := canon_eq (j := pinJ) (by decide +kernel)

theorem ls_objM : lsJ = .obj (objM lsJ) := by kernel_rfl
theorem pin_objM : pinJ = .obj (objM pinJ) := by kernel_rfl
theorem ls_arr : (objM lsJ).toArray = (objM pinJ).toArray := by kernel_rfl

theorem hc1_of : canon lsJ = .ok s1X := by
  rw [ls_objM, canon_obj_congr ls_arr, ← pin_objM]
  exact hc2_of

/-! ## The headers and the identity -/

def s4X (r : Run) (circ cls : String) : String := (canonUtf8 (hdrJ r circ cls)).toOption.getD ""

theorem intJ_hdr (r : Run) (circ cls : String) : IntJ (hdrJ r circ cls) = true := by kernel_rfl

theorem hd_of (r : Run) (circ cls : String) : canonUtf8 (hdrJ r circ cls) = .ok (s4X r circ cls) :=
  canonUtf8_eq (intJ_hdr r circ cls)

def s5X (r : Run) (c : Circuit) (circ cls : String) : String := (canonUtf8 (hdrD r c circ cls)).toOption.getD ""

theorem intJ_hdrD (r : Run) (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 circ cls : String) :
    IntJ (hdrD r (cO ls lh H C A B A' B' s1) circ cls) = true := by
  simp only [hdrD, HmRow.unitsEntry]
  split <;> kernel_rfl'

theorem hd2_of (r : Run) (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 circ cls : String) :
    canonUtf8 (hdrD r (cO ls lh H C A B A' B' s1) circ cls) = .ok (s5X r (cO ls lh H C A B A' B' s1) circ cls) :=
  canonUtf8_eq (intJ_hdrD r ls lh H C A B A' B' s1 circ cls)

def s6X : String := (canonUtf8 (Tags.rowV2Identity tz.identity)).toOption.getD ""

theorem intJ_id : IntJ (Tags.rowV2Identity tz.identity) = true := by decide +kernel

theorem hid_of : canonUtf8 (Tags.rowV2Identity tz.identity) = .ok s6X := canonUtf8_eq intJ_id

/-! ## The hello -/

def helloX (σ : ByteArray) (m : Nat) : String := (helloOf tz σ m coinsX).toOption.getD ""

theorem hello_isOk (σ : ByteArray) (m : Nat) : (helloOf tz σ m coinsX).isOk = true :=
  canon_ok _ _ (Nat.lt_succ_self _) (by kernel_rfl)

theorem hhello_of (σ : ByteArray) (m : Nat) : helloOf tz σ m coinsX = .ok (helloX σ m) := ok_of_isOk _ (hello_isOk σ m)

end SpotFlock.InstanceStmt
