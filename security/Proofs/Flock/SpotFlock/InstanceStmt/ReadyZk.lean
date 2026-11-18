module

public import Proofs.Flock.SpotFlock.InstanceStmt.Scope
public import Proofs.Flock.Claims.Complete.ZkTables.Accepts
import Proofs.Flock.SessionLoop.Concrete.Fwd
import Proofs.Flock.SessionLoop.FinPoints
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.Soundness.Discharge.HiddenExec.Setup
import Proofs.Flock.Claims.Recursion.HonestVerdictFinal

/-! # The two-instance statement before the masked verifier

`zkIn_x` is `FlockDeploy.ZkIn` at the statement `x r`: its setup (`setupOf_x`) and its table's sizes (`zkBody_tabX`).
`readyZk_x` is `FlockComplete.readyZk sha512` there, given only that its honest masked record fits its length word
(`RecordFitsZkAt`): its one table (`tabX_size`), its sample, its length (`x_small`), its regions (`regionsFit_tabAt`:
the hidden setup's `checkRegionBytes`, `setupHidden_regionBytes`) and its coin tree (`tabX_coinTree`). So `accepts_x`:
the honest masked prover is accepted at some coins, at a witness in scope (`ZkTables.accepts_masked`).
-/

@[expose] public section
open Flock RecDeploy.VStarNets
open Properties.Flock.Nets (FixedNetsBuild)
open FlockDeploy (ZkIn)
open FlockComplete (RegionsFit RecordFitsZkAt readyZk)

set_option maxHeartbeats 200000000 -- the table's count and coin tree are read off its setup by the kernel
set_option maxRecDepth 200000

namespace SpotFlock.InstanceStmt

theorem tabX_size (r : Run) (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 s5 s6 : String)
    (hello : ByteArray → Nat → String) : (tabX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello).2.size = 1 := by
  kernel_rfl'

theorem tabX_coinTree (r : Run) (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse)
    (s1 s3 s4 s5 s6 : String) (hello : ByteArray → Nat → String) :
    (tabX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello).1.coinTree.isSome = true := by
  kernel_rfl'

/-- **The statement's session in the masked scope's numbers**: its setup, and its table's sizes. -/
theorem zkIn_x (hn : FixedNetsBuild) {r : Run} (hr : r.Plain) : ZkIn (x r) (ins r) (tabAt r).1 (tabAt r).2 := by
  have hb : zkBody (tabAt r).1 (tabAt r).2 = true := zkBody_tabX r _ _ _ _ _ _ _ _ _ _ _ _ _ _
  simp only [zkBody, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at hb
  obtain ⟨hb3, hall⟩ := hb
  refine ⟨setupOf_x hn hr, by omega, by omega, by omega, fun st hst => ?_⟩
  have h := hall st hst
  exact ⟨by omega, by omega, show _ ∧ _ ∧ _ from ⟨(by omega : st.c.kLog ≤ 27),
    (by omega : st.regions.size ≤ 1024), (by omega : (tabAt r).1.mPts ≤ 64)⟩⟩

/-- **The statement's table has each region's bytes its free bits**: the hidden setup refuses any other
(`setupHidden_regionBytes`). -/
theorem regionsFit_tabAt (hn : FixedNetsBuild) {r : Run} (hr : r.Plain) : ∀ st ∈ (tabAt r).2, RegionsFit st := by
  have hp := FlockDeploy.fixedNets_parse hn
  have hCe : classHash.isEmpty = false := classOfB_isEmpty (HmNets.shaLaid ()) (HmNets.hmLaid ())
  have hset := setupTables_X r (HmNets.shaLaid ()) (HmNets.hmLaid ()) circuitHash classHash AF BF AF' BF' s1X
    (s3X (hexOf classHash)) (s4X r (hexOf circuitHash) (hexOf classHash))
    (s5X r cF (hexOf circuitHash) (hexOf classHash)) s6X helloX hr rfl (classOf_eq _ _)
    (FlockDeploy.sha_lines hn) (FlockDeploy.hm_lines hn) hp.1.2 hp.2.2 hc1_of hc2_of
    (fun c h1 h2 => FlockDeploy.Concrete.check_of hp.1.1 hp.2.1 c h1 h2) hCe (hu1_of _)
    (hu2_of _ _ _ _ _ _ _ _ _ hCe) (hd_of _ _ _) (hd2_of _ _ _ _ _ _ _ _ _ _ _ _) hid_of hhello_of
  obtain ⟨-, hall⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_setupOne
    (spec := (tabAt r).1) (sts := (tabAt r).2) hset
  intro st hst
  obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 hst
  obtain ⟨d, hd, -⟩ := hall j hj
  have hh : (FlockSoundness.Discharge.Exec.tagsAt tz 1 j).hiddenOutputs = true := by
    unfold FlockSoundness.Discharge.Exec.tagsAt
    rw [if_pos (Nat.le_refl 1)]
    exact hidden_tzX
  rw [FlockSoundness.Discharge.HiddenExec.setupOne_hidden hh] at hd
  have hrb := Claims.ZKMal.Masked.setupHidden_regionBytes hd
  intro reg hreg b hb
  exact hrb reg (Array.mem_toList_iff.2 hreg) b hb

/-- **The statement is ready for the masked verifier at SHA-512** where its honest masked record fits its length
word: every other conjunct of `readyZk` holds of the statement. -/
theorem readyZk_x (hn : FixedNetsBuild) {r : Run} (hr : r.Plain)
    (hrec : RecordFitsZkAt FlockDeploy.sha512 (x r) (tabAt r).1 (tabAt r).2) :
    readyZk FlockDeploy.sha512 (x r) = true := by
  have h1 : (tabAt r).2.size = 1 := tabX_size r _ _ _ _ _ _ _ _ _ _ _ _ _ _
  have hct : (tabAt r).1.coinTree.isSome = true := tabX_coinTree r _ _ _ _ _ _ _ _ _ _ _ _ _ _
  have hreg := regionsFit_tabAt hn hr
  have hl := x_small hn hr
  rw [FlockComplete.readyZk_sha512, setupOf_x hn hr]
  revert h1 hct hreg hrec
  generalize tabAt r = T
  obtain ⟨spec, sts⟩ := T
  intros
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, and_assoc]
  exact ⟨‹_›, rfl, hl, ‹_›, ‹_›, ‹_›⟩

/-- **The honest masked prover is accepted at some coins** at the statement, at a witness in scope, where its honest
masked record fits its length word (`ZkTables.accepts_masked` at `zkIn_x` and `readyZk_x`). -/
theorem accepts_x (hn : FixedNetsBuild) {r : Run} (hr : r.Plain)
    (hrec : RecordFitsZkAt FlockDeploy.sha512 (x r) (tabAt r).1 (tabAt r).2) {w : Claims.Flock.Wit}
    (hS : Claims.Flock.Scope (FlockDeploy.deployZk Readable.relation) (x r))
    (hR : Claims.Flock.R (FlockDeploy.deployZk Readable.relation).parse (x r) w)
    (ω : (FlockDeploy.deployZk Readable.relation).Tape) :
    ∃ c, (Claims.run FlockDeploy.sha512 ((FlockDeploy.deployZk Readable.relation).P w)
      (FlockDeploy.deployZk Readable.relation).V (x r) ω c).out = .acc :=
  FlockComplete.ZkTables.accepts_masked FlockDeploy.sha512 FlockDeploy.out64_sha512 (zkIn_x hn hr) (readyZk_x hn hr hrec) hS hR ω

end SpotFlock.InstanceStmt
