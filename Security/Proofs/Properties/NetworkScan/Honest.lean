import Properties.NetworkScan
import Properties.NetworkScan.Honest
import Proofs.Properties.NetworkScan.Check
import Proofs.Properties.NetworkScan.Passes
import Proofs.Properties.NetworkScan.Honest.Loop
import Proofs.Properties.NetworkScan.Honest.ReadsIngress
import Proofs.Properties.NetworkScan.Honest.ReadsEgress

/-!
# An accepted scan whose premises hold

`ScanFits`' honest witness (`scanFits_honest`), at the Program `Honest.prog` and its two templates' circuit files
(`Properties.NetworkScan.Honest`, in the properties tree, since the witness's statement names them):

* `Honest.scan`: `scanProg` accepts the Program at the schedule `Honest.sched`: its keys are distinct
  (`keysUnique_of_nodup`) and its composites' nodes read only parameters and earlier nodes (`orderViolation_none`).
* `Honest.filesOf`: its template nodes are its root's nodes 3 and 4 (`Honest.tnodes`), and each template's circuit file
  passes Bits' check at its template (`Honest.checkIngress`, `Honest.checkEgress`).
* `Honest.rowsRead`: each file's unit reads its template's `row` parameter's bits and no input padding
  (`Honest.readsIngress`, `Honest.readsEgress`).
* `Honest.evaluate`, `Honest.membersOf`: the template query over the two templates finds one instance of each, and the
  two files' members are those instances.

Each is computed by `cbv`, or by the kernel, with `lastLe`'s loop read one step at a time (`Honest.loopId`) and a
type's bits summed by `Flock.Sigma.bits` (`Check.bits_bitsOf`).
-/

namespace Properties.NetworkScan.Honest

section
attribute [local cbv_eval] loopId Check.bits_bitsOf

set_option maxRecDepth 8192 in
set_option cbv.maxSteps 5000000 in
theorem scan : NetworkCheck.Period.scanProg prog (Flock.hexOf .empty) sched = .ok ⟨1, 0, 50, 50, 16⟩ := by
  have hku : NetworkCheck.Period.keysUnique prog = true := keysUnique_of_nodup (by decide_cbv)
  have hov : prog.orderViolation = none := orderViolation_none prog (by decide +kernel) (by decide +kernel)
  unfold NetworkCheck.Period.scanProg
  simp only [hku, hov]
  cbv

set_option maxRecDepth 8192 in
set_option cbv.maxSteps 10000000 in
theorem checkIngress : SampledProofs.Bits.check fIngress prog "IngressScanFirst_v1" = none := by
  cbv

set_option maxRecDepth 8192 in
set_option cbv.maxSteps 10000000 in
theorem checkEgress : SampledProofs.Bits.check fEgress prog "EgressBucket_v1" = none := by
  cbv

set_option cbv.maxSteps 2000000 in
theorem evaluate : Flock.TemplateQuery.evaluate (.ofProg prog) #["IngressScanFirst_v1", "EgressBucket_v1"] =
    .ok #[⟨3, "IngressScanFirst_v1", 0, 1⟩, ⟨4, "EgressBucket_v1", 1, 1⟩] := by
  cbv

theorem tnodes : NetworkCheck.Period.templateNodes prog = [3, 4] := by
  cbv

end

theorem filesOf : FilesOf prog [fIngress, fEgress] := by
  unfold FilesOf
  refine ⟨by rw [tnodes]; rfl, fun i hi => ?_⟩
  match i, hi with
  | 0, _ => rw [tnodes]; exact checkIngress
  | 1, _ => rw [tnodes]; exact checkEgress
  | n + 2, h => exact absurd h (by simp)

theorem rowsRead : RowsRead prog [fIngress, fEgress] := by
  unfold RowsRead
  intro i hi
  match i, hi with
  | 0, _ => rw [tnodes]; exact readsIngress
  | 1, _ => rw [tnodes]; exact readsEgress
  | n + 2, h => exact absurd h (by simp)

theorem membersOf : MembersOf prog [fIngress, fEgress]
    #[⟨3, "IngressScanFirst_v1", 0, 1⟩, ⟨4, "EgressBucket_v1", 1, 1⟩] [(fIngress, #[0]), (fEgress, #[1])] := by
  unfold MembersOf
  refine ⟨by decide +kernel, fun m hm => ?_⟩
  match m, hm with
  | 0, _ => exact ⟨0, by decide, rfl, fun v hv => by simp at hv; subst hv; decide +kernel⟩
  | 1, _ => exact ⟨1, by decide, rfl, fun v hv => by simp at hv; subst hv; decide +kernel⟩
  | n + 2, h => exact absurd h (by simp)

end Properties.NetworkScan.Honest

namespace Properties.NetworkScan

/-- **ScanFits' honest witness**: at a Program of one ingress and one egress template (`Honest.prog`), `scanProg`
accepts, the templates' two circuit files pass Bits' check and read their row parameters, the template query finds one
instance of each template, and the files' members are those instances. -/
theorem scanFits_honest :
    NetworkCheck.Period.scanProg Honest.prog (Flock.hexOf .empty) Honest.sched = .ok ⟨1, 0, 50, 50, 16⟩ ∧
      FilesOf Honest.prog [Honest.fIngress, Honest.fEgress] ∧ RowsRead Honest.prog [Honest.fIngress, Honest.fEgress] ∧
      Flock.TemplateQuery.evaluate (.ofProg Honest.prog) #["IngressScanFirst_v1", "EgressBucket_v1"] =
        .ok #[⟨3, "IngressScanFirst_v1", 0, 1⟩, ⟨4, "EgressBucket_v1", 1, 1⟩] ∧
      MembersOf Honest.prog [Honest.fIngress, Honest.fEgress]
        #[⟨3, "IngressScanFirst_v1", 0, 1⟩, ⟨4, "EgressBucket_v1", 1, 1⟩]
        [(Honest.fIngress, #[0]), (Honest.fEgress, #[1])] :=
  ⟨Honest.scan, Honest.filesOf, Honest.rowsRead, Honest.evaluate, Honest.membersOf⟩

end Properties.NetworkScan
