module

public import Proofs.Flock.SpotFlock.Honest.Instance
public import Proofs.Flock.SpotFlock.InstanceStmt.Scope
public import Proofs.Flock.SpotFlock.ScopeDesc
public import Proofs.Flock.SpotFlock.ScopeOf
public import Proofs.Properties.Flock.Nets
public import Proofs.Flock.Soundness.Discharge.Hidden.Built
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
import Flock.Draw
import Flock.Desc
import Flock.ProvedScope
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
# The two-instance statement as a template's reference statement

`refAt reads headers` is a `Ref` (`Honest.Instance`) at the two-instance statement `InstanceStmt.x run₀`: template
`T_v1` over the ports `i0` and `o0`, with reference registered reads `reads` and reference header `headers`, which none
of its facts reads. The facts:

* `zk`, `tmpl`: the statement is in Flock's masked scope and reads as a member's (`InstanceStmt.zkOk_x`,
  `InstanceStmt.templateOk_x`) where the verifier's two fixed nets build, which `Properties.Flock.Nets.fixedNetsBuild`
  proves;
* `scope`: the two-stage scope holds at the members, their partition and the design `subset:1` over one request of two
  units (`scopeOk_members`): the member fits (`memberOk_cO`, its unit net built by `built_of_parseStmt`), the replay
  layout fits and fills its file (`filesFit_two`, `fillOk`), and the design is well formed on its strata (`designOk`);
* `named`: every row a unit reads its source bits at is one of the circuit's ports (`mem_sourceRows`), which are `i0`
  and `o0` (`names_cO`).

`run₀` is a run whose strings are plain (`run₀_plain`).
-/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy Properties.Spot Flock.Bridge.Members
open Flock Flock.ZkServer
open SpotFlock.InstanceStmt
open Properties.Flock.Nets (FixedNetsBuild fixedNetsBuild)
open Lean (Json)

/-- **A run whose strings are plain**: its input set and its two ports' bindings. -/
def run₀ : InstanceStmt.Run := ⟨"honest", "i0", "o0"⟩

theorem run₀_plain : run₀.Plain where
  set := by simp [run₀, plainStr, String.all_bool_eq]
  bindI := by simp [run₀, plainStr, String.all_bool_eq]
  bindO := by simp [run₀, plainStr, String.all_bool_eq]
  size := by decide

/-- **The statement's circuit fits a member**: one unit of at most `2^32` gates, a positive count of units, and a unit's
inputs and outputs within the largest size class; at any nets. -/
theorem memberOk_cO (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 : String) :
    memberOk (cO ls lh H C A B A' B' s1, #[0, 1]) = true := by
  kernel_rfl'

/-! ## The rows a unit reads its source bits at are its circuit's ports

`sourceRows` reaches a row's name only through a message bit (`Spot.Bits.msgAt`), which is one only below its port's
bits, so at a port of the circuit; the binary search that picks the port (`lastLe`) is never evaluated. -/

theorem lt_of_bits_ne {ps : Array _root_.Flock.Port} {q : ℕ} (h : (ps.getD q default).bits ≠ 0) : q < ps.size := by
  refine Nat.lt_of_not_le fun hq => h ?_
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none hq]
  rfl

theorem msgAt_port {c : _root_.Flock.Circuit} {ix : Spot.Bits.Index} {x t : ℕ} {wide : Bool} {p byte bit : ℕ}
    (h : Spot.Bits.msgAt c ix x t wide = .msg p byte bit) : p < c.ports.size := by
  dsimp only [Spot.Bits.msgAt] at h
  split at h
  · rename_i hc
    injection h with h1
    subst h1
    refine lt_of_bits_ne fun h0 => ?_
    rw [h0] at hc
    simp at hc
  · cases h

theorem srcAt_port {c : _root_.Flock.Circuit} {ix : Spot.Bits.Index} {k p' t p byte bit : ℕ}
    (h : Spot.Bits.srcAt c ix k p' t = .msg p byte bit) : p < c.ports.size := by
  unfold Spot.Bits.srcAt at h
  repeat' (first | exact msgAt_port h | (cases h; done) | split at h | dsimp only at h)

theorem srcKey_port {c : _root_.Flock.Circuit} {ix : Spot.Bits.Index} {x p byte bit : ℕ}
    (h : Spot.Bits.srcKey c ix x = .msg p byte bit) : p < c.ports.size := by
  unfold Spot.Bits.srcKey at h
  repeat' (first | exact srcAt_port h | (cases h; done) | split at h | dsimp only at h)

/-- **A row a unit reads its source bits at is one of the circuit's ports.** -/
theorem mem_sourceRows {c : _root_.Flock.Circuit} {row : String} (h : row ∈ sourceRows c) :
    ∃ p, ∃ hp : p < c.ports.size, c.ports[p].name = row := by
  dsimp only [sourceRows] at h
  obtain ⟨b, -, hb⟩ := List.mem_filterMap.1 h
  have hb : (Spot.Bits.srcPos c 0 (Spot.Bits.srcKey c (Spot.Bits.Index.of c) b)).map (fun x => x.row) = some row := hb
  generalize hs : Spot.Bits.srcKey c (Spot.Bits.Index.of c) b = s at hb
  cases s with
  | msg p byte bit =>
    have hp := srcKey_port hs
    refine ⟨p, hp, ?_⟩
    have hn : (c.ports.getD p default).name = row := Option.some.inj hb
    rw [← hn, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hp]
    rfl
  | _ => cases hb

theorem sourceRows_named {c : _root_.Flock.Circuit} {ns : List String} (hc : c.ports.toList.map (·.name) = ns) :
    (sourceRows c).all (fun row => ns.toArray.contains row) = true := by
  rw [List.all_eq_true]
  intro row hrow
  obtain ⟨p, hp, rfl⟩ := mem_sourceRows hrow
  rw [Array.contains_iff_mem, List.mem_toArray, ← hc]
  exact List.mem_map.2 ⟨_, Array.getElem_mem_toList hp, rfl⟩

/-- **The statement's circuit's ports are `i0` and `o0`**, at any nets. -/
theorem names_cO (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 : String) :
    (cO ls lh H C A B A' B' s1).ports.toList.map (·.name) = ["i0", "o0"] := by
  kernel_rfl'

theorem built_cF : FlockSoundness.Built cF.unit :=
  FlockSoundness.Discharge.Hidden.built_of_parseStmt (parseStmt_x fixedNetsBuild run₀) typed_tzX

theorem stmtOf_x₀ : stmtOf (InstanceStmt.x run₀) = cF := stmtOf_x fixedNetsBuild run₀_plain

theorem blocks_oneLayout (gs : ℕ × ℕ) : (oneLayout gs).blocks = #[⟨some 0, gs⟩] := rfl

/-- **The replay layout of one request of two units fits its file**: its one block is unit `0`'s, and two units fit
the request's two. -/
theorem filesFit_two (x : Msg) : FilesFit (ms2 x) #[2] (oneLayout (0, 0)) := by
  intro ρ b hb u hu
  rw [blocks_oneLayout] at hb
  match ρ, hb with
  | 0, hb =>
    obtain rfl : b = ⟨some 0, (0, 0)⟩ := (Option.some.inj hb).symm
    obtain rfl : u = 0 := (Option.some.inj hu).symm
    refine ⟨rfl, fun t ht => ?_⟩
    obtain rfl : t = 0 := by simpa [ms2] using ht
    exact Nat.le_refl 2
  | _ + 1, hb => simp at hb

theorem scope₀ :
    scopeOk (bitsProgs (ms2 (InstanceStmt.x run₀))) (membersPart (ms2 (InstanceStmt.x run₀)))
      ⟨part2 (InstanceStmt.x run₀), design1, [1]⟩ = true := by
  refine scopeOk_members (fun ci hci => ?_) (filesFit_two _) (fillOk_iff.1 rfl)
    (wellFormed_filesReplay_iff.2 rfl) rfl (by simp)
  obtain rfl : ci = (stmtOf (InstanceStmt.x run₀), #[0, 1]) := by simpa [ms2] using hci
  rw [stmtOf_x₀]
  exact memberFits_of_memberOk built_cF (memberOk_cO _ _ _ _ _ _ _ _ _)

/-- **The two-instance statement as template `T_v1`'s reference statement**, over the ports `i0` and `o0`, with
reference registered reads `reads` and reference header `headers`. -/
def refAt (reads headers : Json) : Ref where
  x := InstanceStmt.x run₀
  name := "T_v1"
  ports := #["i0", "o0"]
  reads := reads
  headers := headers
  zk := zkOk_x fixedNetsBuild run₀_plain
  tmpl := fun h h0 => templateOk_x fixedNetsBuild run₀_plain h h0
  scope := scope₀
  named := by
    rw [stmtOf_x₀]
    exact sourceRows_named (names_cO _ _ _ _ _ _ _ _ _)

end SpotFlock.Honest
