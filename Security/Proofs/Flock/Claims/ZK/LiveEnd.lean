import Proofs.Flock.Claims.ZK.PrefixLive
import Proofs.Flock.Claims.ZK.FinExe

/-!
# The live session ends where the session stopped after the last round does (`Masked.liveEnd`)

`LiveEnd`: after `refK x` openings `ops`, the session answered by them returns what the session answered by the padded
openings and stopped after round `refK x` returns.

* **The coupling** (`PrefixAgree.sessionPZk_agree` at `n = refK x`): the two runs end the same, or the stopped run threw
  at send `refK x + 1` and the live run's sends extend it.
* **The sends** (`sessionPZk_sendsLe`): every run of the masked session, whether it returns or stops, sends at most
  `repStartZk` of its last rep, `1 + rounds = refK x` (`repStart_last`, `refK_eq`): the link message, and each rep at
  most its `repRounds true` (`repPZk_sends`, which is that count at a `--zk` schedule, with `k0 ≥ 1` and at least two
  levels). So the second case can't happen, and `LiveEnd`'s `Admits` hypothesis isn't used.
-/

namespace FlockComplete.PrefixAgree

open Flock Transcript FlockComplete FlockDeploy With JsonRT ZkTables

/-- **A loop each of whose steps at `a` sends at most `c a`** sends at most their sum. -/
theorem sendsC_forIn_array {α β : Type} (xs : Array α) (b : β) {f : α → β → PS (ForInStep β)} (c : α → ℕ)
    (h : ∀ a ∈ xs, ∀ b, sendsC.P (f a b) (c a)) : sendsC.P (forIn xs b f) (xs.toList.map c).sum := by
  rw [← Array.forIn_toList]
  have key : ∀ l : List α, (∀ a ∈ l, ∀ b, sendsC.P (f a b) (c a)) → ∀ b, sendsC.P (forIn l b f) (l.map c).sum := by
    intro l
    induction l with
    | nil => exact fun _ b => sendsC.mono (sendsC.pure b) (by simp)
    | cons a l ih =>
      intro hl b
      rw [List.forIn_cons]
      refine sendsC.mono (sendsC.bind (hl a List.mem_cons_self b) fun r => ?_) (n := c a + (l.map c).sum)
        (by simp)
      cases r with
      | done b => exact sendsC.mono (sendsC.pure b) (by omega)
      | yield b => exact ih (fun a ha => hl a (List.mem_cons_of_mem _ ha)) b
  exact key _ (fun a ha => h a (Array.mem_toList_iff.1 ha)) b

/-- **The masked session after the link sends at most its reps' rounds**, at tables of `14 ≤ m`, one spec table per
statement and each table's `--zk` schedule. -/
theorem afterLinkPZk_sendsLe (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (spec : SessionSpec)
    (sts : Array Stmt) (words : Array (List ℕ)) (schs : Array (Schedule × ℕ)) (trs : Array Trace) (tape : TapeZk)
    (tbls : Array (Array F128)) (cms : Array Commit) (c1 : ℕ → F128) (hspec : spec.tables.size = sts.size)
    (hsts : ∀ j (hj : j < sts.size), 14 ≤ sts[j].m)
    (hschs : ∀ j (hj : j < sts.size), Zk.schedule sts[j].m = .ok schs[j]!) :
    sendsC.P (afterLinkPZk (asksS resp) (Hashes.ofSha512 h) spec (tabsOf sts) words trs tape schs tbls cms c1)
      ((List.range (2 * sts.size)).map fun q => repRounds true spec sts[q / 2]!).sum := by
  unfold afterLinkPZk
  extract_lets rootB points streams proofs
  have hpts : points.size = spec.points := by simp [points]
  refine sendsC.mono (sendsC.bind (m := 0) (sendsC_forIn_array _ _ (fun a => repRounds true spec sts[a.1]!)
    fun a ha b => ?_) fun s => ?_) ?_
  · obtain ⟨i, r, nm⟩ := a
    have hi : i < sts.size := by
      rw [← Array.mem_toList_iff, FlockDeploy.repStreams_toList, List.mem_map] at ha
      obtain ⟨k, hk, he⟩ := ha
      rw [List.mem_range, hspec] at hk
      have := congrArg Prod.fst he
      simp only at this
      omega
    have ht : (tabsOf sts)[i]! = Setup.ofCircuit sts[i] := by
      rw [getElem!_pos (tabsOf sts) i (by simpa [tabsOf] using hi)]
      simp [tabsOf]
    have hst : sts[i]! = sts[i] := getElem!_pos sts i hi
    have hv := hschs i hi
    obtain ⟨-, -, -, hk0, -, hdim, -, -, -⟩ :=
      FlockSoundness.Discharge.ZkExec.zk_schedule_seams sts[i].m schs[i]!.1 schs[i]!.2 hv
    obtain ⟨-, h2, -, -, -, -, -⟩ := hdim
    have hm := hsts i hi
    extract_lets
    dsimp only
    simp only [ht, hst]
    apply sendsC_lift_bind
    intro extras he
    have hx : extras.size = spec.points * sts[i].regions.size := by
      rw [FlockDeploy.extraClaims_size (st := sts[i]) he, hpts, Nat.mul_comm]
    refine sendsC_quiet_bind (innerCommitP_quiet h _ _) fun ic _ => ?_
    apply sendsC.mono
    · cost_walk [repPZk_sends resp h]
    · rw [← repRoundsT_eq hv hx]
      have e6 : (Setup.ofCircuit sts[i]).fold.kLog = sts[i].c.kLog := rfl
      have e7 : (Setup.ofCircuit sts[i]).m = sts[i].m := rfl
      unfold repRoundsT
      rw [e6, e7, hx]
      omega
  · cost_walk [recordJson_sends h]
  · rw [FlockDeploy.repStreams_toList, List.map_map, hspec]
    exact le_of_eq (by rfl)

/-- **Every run of the masked session sends at most `repStartZk` of its last rep**, whether it returns or stops, at
tables of `14 ≤ m` and one spec table per statement. -/
theorem sessionPZk_sendsLe (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (spec : SessionSpec)
    (sts : Array Stmt) (words : Array (List ℕ)) (trs : Array Trace) (tape : TapeZk)
    (hspec : spec.tables.size = sts.size) (hsts : ∀ j (hj : j < sts.size), 14 ≤ sts[j].m) :
    sendsC.P (sessionPZk (asksS resp) (Hashes.ofSha512 h) spec (tabsOf sts) words trs tape)
      (repStartZk spec sts (2 * sts.size)) := by
  rw [sessionPZk_eq]
  refine sendsC_quiet_bind (schedsPZk_quiet _) fun schs hs => ?_
  have htabs : (tabsOf sts).size = sts.size := ZkTables.tabsOf_size sts
  obtain ⟨hssz, hsf⟩ : schs.size = (tabsOf sts).size ∧ ∀ i (h₁ : i < (tabsOf sts).size) (h₂ : i < schs.size),
      Zk.schedule (tabsOf sts)[i].m = .ok schs[i] := by
    have hA : (schedsPZk (tabsOf sts) : PS _).go #[] = (.ok schs, #[]) := by rw [schedsPZk_quiet, hs]
    exact Ret.go (Ret.mapM_array _ (fun st' v => Zk.schedule st'.m = .ok v) _ fun a _ =>
      Ret.monadLift (Zk.schedule a.m)) hA
  have hschs : ∀ i (hi : i < sts.size), Zk.schedule sts[i].m = .ok schs[i]! := fun i hi => by
    have := hsf i (by omega) (by omega)
    rw [← getElem!_pos schs i (by omega)] at this
    simp only [tabsOf, Array.getElem_map] at this
    exact this
  refine sendsC_quiet_bind (commitsPZk_quiet h _ _ _ _) fun cms _ => ?_
  exact sendsC.bind (sendsC_link resp _ _) fun c1 =>
    afterLinkPZk_sendsLe resp h spec sts words schs trs tape _ cms c1 hspec hsts hschs

end FlockComplete.PrefixAgree

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

/-- **`LiveEnd`**: after `refK x` openings, the session answered by them returns what the session answered by the
padded openings and stopped after round `refK x` returns (`PrefixAgree.sessionPZk_agree`; its second case sends more
than `sessionPZk_sendsLe` allows). -/
theorem liveEnd : LiveEnd := by
  intro x hx w _ ω ops hl _ s spec hsx
  obtain ⟨-, -, hspec, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  obtain ⟨hp, hmp, hreg⟩ := live_bounds hx hsx
  have hK := refK_eq hsx
  have hpos : 0 < ops.length := by rw [hl, hK]; omega
  have hc : ∀ k < ops.length, ∀ i < 65536, FlockComplete.coinOf ((ops.map fun p => coinOfD p.2).getD k []) i =
      slotAt (padOps (x := x) ops) k i := fun k hk i hi => (slotAt_padOps_lt ops hk (by omega) hi).symm
  have hcpl : FlockComplete.PrefixAgree.Couple ops.length (liveRunAt x w ω ops spec)
      (runExe x w ω (padOps (x := x) ops) ops.length spec) :=
    FlockComplete.PrefixAgree.sessionPZk_agree _ _ hc _root_.Flock.Sha512.hash spec (stsOf x) _ _ ω hpos hspec hp
      hmp (fun j hj => ⟨(hall j hj).1, (hall j hj).2.1⟩) hreg
  have hle : (liveRunAt x w ω ops spec).2.size ≤ refK x := by
    have := FlockComplete.sendsC_ran (FlockComplete.PrefixAgree.sessionPZk_sendsLe _ _root_.Flock.Sha512.hash spec
      (stsOf x) _ _ ω hspec (fun j hj => by have := (hall j hj).1; omega))
      (r := (liveRunAt x w ω ops spec).1) (l := (liveRunAt x w ω ops spec).2) rfl
    rw [FlockComplete.ZkTables.repStart_last, ← hK] at this
    exact this
  rw [hl] at hcpl
  rcases hcpl with ⟨e, -⟩ | ⟨-, hs, bs, hbs⟩
  · exact congrArg Prod.fst e
  · have := congrArg Array.size hbs
    rw [Array.size_append] at this
    omega

end Masked

end Claims.ZKMal
