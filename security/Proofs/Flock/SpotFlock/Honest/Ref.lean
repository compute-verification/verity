module

public import Proofs.Flock.SpotFlock.Honest.Instance
public import Proofs.Flock.SpotFlock.InstanceStmt.Scope
public import Proofs.Flock.SpotFlock.ScopeDesc
public import Proofs.Flock.SpotFlock.ScopeOf
public import Proofs.Properties.Flock.Nets
public import Proofs.Flock.Soundness.Discharge.Hidden.Built

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
* `named`: every row a unit reads its source bits at is `i0` or `o0` (`sourceRows_cO`).

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
  set := by first | decide | simp [run₀, plainStr, String.all_bool_eq]
  bindI := by first | decide | simp [run₀, plainStr, String.all_bool_eq]
  bindO := by first | decide | simp [run₀, plainStr, String.all_bool_eq]
  size := by first | decide | simp [run₀]

/-- **The statement's circuit fits a member**: one unit of at most `2^32` gates, a positive count of units, and a unit's
inputs and outputs within the largest size class; at any nets. -/
theorem memberOk_cO (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 : String) :
    memberOk (cO ls lh H C A B A' B' s1, #[0, 1]) = true := by
  kernel_rfl'

/-- **Every row a unit of the statement's circuit reads its source bits at is `i0` or `o0`**, at any nets. -/
theorem sourceRows_cO (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 : String) :
    (sourceRows (cO ls lh H C A B A' B' s1)).all (fun row => #["i0", "o0"].contains row) = true := by
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
    exact sourceRows_cO _ _ _ _ _ _ _ _ _

end SpotFlock.Honest
