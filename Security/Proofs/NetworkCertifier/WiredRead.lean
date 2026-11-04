import Proofs.NetworkCertifier.WiredNet
import Proofs.Flock.Claims.Witnesses.BindingMeets
import Proofs.Flock.Claims.KS.ZkExtend

/-!
# The revealed wires at the deployed sampling record

At the deployed sampling record (`SamplingDeploy.deploy rel c`) and SHA-512 (`FlockDeploy.sha512`), an opening passes
exactly when the verifier's own registered check accepts its row's commit string (`ReadOpens`), the check a session's
statement runs where it reads a committed leaf by path. So the wires are revealed at the record (`WiresRevealed`)
exactly where they are revealed at that check (`WiredReads`), and sampling at the incorrect set follows from the
record's facts.

* `ReadOpens c roots o`: under `roots`, the verifier's own port whose root `c.at_` names for the opening's position
  opens the commit string `b ‖ c` of the opening's row under its salt (`Flock.Registered.opens`), at the leaf index
  `c.at_` names, through the opening's path.
* `opens_iff_read`: at SHA-512 the record's opening check is `ReadOpens` (`Binding_meets`, the hm96 leaf being the
  executable's tree leaf of the commit string, `hm96Leaf_bytes`); `opens_of_read` is the half the wires use.
* `WiredReads rel c wires`: `RevealsWires` at the record and SHA-512, at the check `ReadOpens c`.
* `wiresRevealed_deployed`: at the record and SHA-512, `WiresRevealed` is `WiredReads`.
* `ksWired_deployed`: **sampling at the incorrect set at the deployed record** (`KSWired`), at SHA-512 at the deployed
  relation, where Flock's KS holds with no premise (`FlockDeploy.KS.zk_KS_untrim`). Its hypotheses are the facts of the
  record's run that `sampling_deployed_sha512`'s KS reads, with `WiredReads` beside `SessionOpens`; binding, the draw
  law and refusal are discharged as `sampling_assembled_of` discharges them.
-/

namespace NetTiming

open Claims Claims.Sampling
open FlockSoundness.Refine FlockSoundness.Binding

local notation "relD" => _root_.Flock.Readable.relation
local notation "shaD" => FlockDeploy.sha512

/-- **The hm96 leaf's bytes are the executable's tree leaf of the commit string** of the row and the salt
(`hm96_treeLeaf`). -/
theorem hm96Leaf_bytes (r y : Msg) :
    bytes (digBytes (H512 (hm96Input r y))) =
      _root_.Flock.Hm96.Default512.treeLeaf
        (_root_.Flock.Hm96.Default512.commitString (_root_.Flock.Sha512.hash (bytes r)) (bytes y)) := by
  have h := hm96_treeLeaf (Val := List UInt8) id Function.injective_id r (bytes y)
  rw [show (bytes y).data.toList = y by simp [bytes]] at h
  exact (bytes_of_list (H512 (hm96Input r y)).1).trans h

/-- **The verifier's own check accepts the opening `o` under `roots`**: at the root and leaf index `c.at_` names for
its position, the verifier's own port whose root that is opens the commit string `b ‖ c` of the opening's row under its
salt (`Flock.Registered.opens`), through the opening's path. -/
def ReadOpens (c : SamplingDeploy.Code) (roots : List Msg) (o : Opening Bool) : Prop :=
  ∃ root i reg, c.at_ roots o.pos = some (root, i) ∧ c.own.find? (fun p => decide (p.root = bytes root)) = some reg ∧
    _root_.Flock.Registered.opens reg
      (_root_.Flock.Hm96.Default512.commitString
        (_root_.Flock.Sha512.hash (bytes (SamplingDeploy.row c.place c.bits o))) (bytes (c.salt o)))
      i (bytes (c.π o)) = true

/-- **At SHA-512 the deployed record's opening check is the verifier's own** (`Binding_meets`, whose hash
`sha512Msg` is `FlockDeploy.sha512`). -/
theorem opens_iff_read (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (c : SamplingDeploy.Code) (roots : List Msg) (o : Opening Bool) :
    (SamplingDeploy.deploy rel c).commit.Opens shaD roots o = true ↔ ReadOpens c roots o := by
  show (SamplingDeploy.deploy rel c).commit.Opens sha512Msg roots o = true ↔ _
  rw [Binding_meets]
  simp only [ReadOpens, _root_.Flock.Registered.opens, hm96Leaf_bytes]

/-- **Where the verifier's own check accepts, the opening passes** at the deployed record and SHA-512. -/
theorem opens_of_read (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (c : SamplingDeploy.Code) {roots : List Msg} {o : Opening Bool} (h : ReadOpens c roots o) :
    (SamplingDeploy.deploy rel c).commit.Opens shaD roots o = true :=
  (opens_iff_read rel c roots o).2 h

/-- **The wires are revealed at the verifier's own check**: `RevealsWires` at the deployed record and SHA-512, at
`ReadOpens c`. -/
def WiredReads (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (c : SamplingDeploy.Code)
    (wires : AuditStatement → ℕ → Option ℕ) : Prop :=
  RevealsWires shaD (SamplingDeploy.deploy rel c) (ReadOpens c) wires

/-- **At the deployed record and SHA-512 the wires are revealed exactly where they are at the verifier's own check**
(`opens_iff_read`). -/
theorem wiresRevealed_deployed (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (c : SamplingDeploy.Code) (wires : AuditStatement → ℕ → Option ℕ) :
    WiresRevealed shaD (SamplingDeploy.deploy rel c) wires ↔ WiredReads rel c wires :=
  ⟨fun h => RevealsWires.mono (fun r o ho => (opens_iff_read rel c r o).1 ho) h,
    fun h => RevealsWires.mono (fun r o ho => (opens_iff_read rel c r o).2 ho) h⟩

set_option synthInstance.maxSize 1024 in
/-- **Sampling at the incorrect set at the deployed record**, at SHA-512 at the deployed relation:
`ksWired_of_revealed` at `deploy relD c`, its binding from the hm96 leaves under the registered path check
(`Binding_of_hm96`), the sessions' KS from Flock's (`SessionKS_of` at `FlockDeploy.KS.zk_KS_untrim`), the draw law from
the slots and refusal from the scope checks, as `sampling_assembled_of` has them, and the wires revealed from
`WiredReads` (`wiresRevealed_deployed`). -/
theorem ksWired_deployed (c : SamplingDeploy.Code) (wires : AuditStatement → ℕ → Option ℕ)
    (hfixed : DrawFixed (SamplingDeploy.deploy relD c)) (hsessions : SessionRuns shaD (SamplingDeploy.deploy relD c))
    (hnum : SamplingDeploy.SessionsIn c) (hslots : DrawSlots (SamplingDeploy.deploy relD c))
    (iroots : Msg → Seen → ℕ → List Msg) (hopens : SessionOpens shaD (SamplingDeploy.deploy relD c) iroots)
    (hfix : InteriorFixed shaD (SamplingDeploy.deploy relD c) iroots)
    (hprog : ProgramScope (SamplingDeploy.deploy relD c)) (hchecks : ChecksScope shaD (SamplingDeploy.deploy relD c))
    (hw : WiredReads relD c wires) : KSWired shaD (SamplingDeploy.deploy relD c) wires :=
  ksWired_of_revealed wires iroots
    (Binding_of_hm96 _ c.own c.at_ (SamplingDeploy.row c.place c.bits) c.salt c.π c.layer
      (SamplingDeploy.deploy_commit relD c) (SamplingDeploy.row_places c.place c.bits) shaD)
    (SessionKS_of (H := shaD) (D := SamplingDeploy.deploy relD c) FlockDeploy.KS.zk_KS_untrim hfixed hsessions
      (SamplingDeploy.innerNumbers_deploy relD c hnum) (SamplingDeploy.deploy_εtable_nonneg relD c))
    hopens ((wiresRevealed_deployed relD c wires).2 hw) hfix (drawLaw_of_slots _ hslots)
    (refuses_of shaD _ (scopeMatches_of _ hslots hprog) hchecks) (SamplingDeploy.deploy_εtable_nonneg relD c)

end NetTiming
