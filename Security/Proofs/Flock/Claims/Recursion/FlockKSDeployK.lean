import Proofs.Flock.Claims.Recursion.ConflictTiedE
import Proofs.Flock.Claims.DeployVK

/-!
# The code facts and the accepted run's decision at `FlockDeploy.VK keep`

`FlockKSDeployE`'s code facts and `ConflictTiedE`'s accepted-run decision, at every `keep`, by the same proofs at
`nextK keep` (`nextK_decide`): `VK keep` shows each round's coin (`coinsShown_VK`), accepts only at its last round
(`accLast_VK`, `streams_size_plainK` holding at every `keep`), decides without the last coin (`decideFree_VK`) and never
accepts the empty message (`refusable_VK`); an accepted run of it decides within its own verifier log (`accK_decide`),
so its verdict accepts with SHA-512 answered only within that log (`decisionK_runs`, `verdictK_within`).

At `keep := fun _ => true` these are the facts at `VE` (`VK_true`), and at `keep := id` the ones at `V false`
(`VK_id`), so a proof that reads them at `VK keep` holds at both.
-/

namespace FlockDeploy

open Flock
open Claims.Recursion

/-- **Where `VK keep` answers, the deployed plain verifier answers alike.** -/
theorem next_msg_of_nextK (keep : Bool → Bool) (H : Claims.Msg → Claims.Msg) {x : Claims.Msg}
    {as : List Claims.Msg} {cs : List Coin} {m : Claims.Msg} (h : ((nextK keep x as cs).eval H).1 = .msg m) :
    ((next false x as cs).eval H).1 = .msg m := by
  cases ha : act false x as cs with
  | rej => rw [nextK_rej keep H ha] at h; cases h
  | coin c => rw [nextK_coin keep H ha] at h; rw [next_coin H ha]; exact h
  | decide s spec sts link c₁ mids f =>
    simp only [nextK, ha, outOf, Claims.OC.eval_bind, Claims.OC.eval] at h
    split at h <;> simp at h

theorem refusable_VK (keep : Bool → Bool) (H : Claims.Msg → Claims.Msg) : Refusable H (VK keep) := fun x =>
  ⟨[], fun as cs => by
    show ((nextK keep x (as ++ [[]]) cs).eval H).1 ≠ .acc
    rw [nextK_rej keep H (act_snoc_nil false x as cs)]
    exact fun h => Claims.Out.noConfusion h⟩

theorem coinsShown_VK (keep : Bool → Bool) (H : Claims.Msg → Claims.Msg) : CoinsShown H (VK keep) decCoin :=
  fun x as cs c m hl h => coinsShown_V H false x as cs c m hl (next_msg_of_nextK (cs := cs ++ [c]) keep H h)

theorem decideFree_VK (keep : Bool → Bool) (H : Claims.Msg → Claims.Msg) : DecideFree H (VK keep) := by
  intro x as cs c c' hl hacc
  show (nextK keep x as (cs ++ [c'])).eval H = (nextK keep x as (cs ++ [c])).eval H
  change ((nextK keep x as (cs ++ [c])).eval H).1 = .acc at hacc
  cases hact : act false x as (cs ++ [c]) with
  | rej =>
    rw [nextK_rej keep H hact] at hacc
    cases hacc
  | coin c₂ =>
    rw [nextK_coin keep H hact] at hacc
    cases hacc
  | decide s spec sts link c₁ mids f =>
    have hlen := act_decide_length (by simp [hl]) hact
    obtain ⟨c₀, cs', rfl⟩ : ∃ c₀ cs', cs = c₀ :: cs' := by
      rcases cs with _ | ⟨c₀, cs'⟩
      · simp only [List.length_nil, zero_add] at hl
        omega
      · exact ⟨c₀, cs', rfl⟩
    have ha : ∀ t, as.getLast? ≠ some (0 :: t) := fun t h => by
      rcases act_snoc_zero (zk := false) (x := x) (cs := c₀ :: cs') (c := c) h with h' | h' <;>
        rw [hact] at h' <;> cases h'
    unfold nextK
    rw [act_cons_snoc false x as c₀ cs' c c' ha]

/-- **Wherever `VK keep` accepts, the prover sent the rounds of the session the statement sets up against `H`**
(`nextE_acc_length_setup`'s proof, `streams_size_plainK` at `keep`). -/
theorem nextK_acc_length_setup (keep : Bool → Bool) (H : Claims.Msg → Claims.Msg) {x : Claims.Msg}
    {as : List Claims.Msg} {cs : List Coin} (hcs : cs.length = as.length) (h : ((nextK keep x as cs).eval H).1 = .acc) :
    ∃ (s : Ins) (spec : SessionSpec) (sts : Array Flock.Stmt) (spec' : SessionSpec) (sts' : Array Flock.Stmt),
      setupOf false x = some (s, spec, sts) ∧ setupWith (Claims.hOf H) s.read = .ok (spec', sts') ∧
        as.length = 2 + rounds false spec' sts' := by
  cases ha : act false x as cs with
  | rej => rw [nextK_rej keep H ha] at h; cases h
  | coin c => rw [nextK_coin keep H ha] at h; cases h
  | decide s spec sts link c₁ mids f =>
  obtain ⟨spec', sts', hsu, hv, hc⟩ := (nextK_decide keep H ha).1 h
  refine ⟨s, spec, sts, spec', sts', act_decide ha, hsu, ?_⟩
  obtain ⟨lv, hlv, hcus⟩ := checks_custody hc
  obtain ⟨hsz, hk⟩ := streams_size_plainK keep (Hashes.ofSha512 (Claims.hOf H)) hv hlv
  rw [act_decide_length hcs ha, ← custody_length hcus, rounds_sum, List.length_flatMap,
    Nat.add_comm _ 2, ← sum_pairs (fun i => repRounds false spec' sts'[i]!)]
  congr 2
  apply List.ext_getElem
  · simp [hsz]
  · intro k h₁ h₂
    simp only [List.getElem_map, List.getElem_range, Array.getElem_toList, Array.length_toList]
    exact hk k _

/-- **At SHA-512, `VK keep` accepts only at its last round.** -/
theorem accLast_VK (keep : Bool → Bool) : AccLast sha512 (VK keep) := fun x as cs hcs h => by
  obtain ⟨s, spec, sts, spec', sts', hso, hsu, hl⟩ := nextK_acc_length_setup keep sha512 hcs h
  rw [hOf_sha512, setupWith_sha512, Ins.setup_read (setupOf_some hso).2] at hsu
  cases hsu
  show as.length = (match setupOf false x with | some (_, spec, sts) => 2 + rounds false spec sts | none => 1)
  rw [hso]
  dsimp only
  exact hl

end FlockDeploy

namespace FlockDeploy.KS

open Flock
open Claims (OC OCV)

/-- **An accepted run of `VK keep` decides, within its own verifier log** (`accE_decide`'s proof, at `nextK keep`). -/
theorem accK_decide (keep : Bool → Bool) {H : Claims.Msg → Claims.Msg} {Ω : Type} {P : Claims.Prover Ω}
    {x : Claims.Msg} {ω : Ω} {r : Fin ((VK keep).k x) → Coin} (h : (Claims.run H P (VK keep) x ω r).out = .acc) :
    ∃ s spec sts link c₁ mids f,
      act false x ((Claims.run H P (VK keep) x ω r).msgs.map Prod.fst)
          ((List.range ((Claims.run H P (VK keep) x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) =
        .decide s spec sts link c₁ mids f ∧
      Runs H (decisionK keep s x link c₁ mids f) () ((Claims.run H P (VK keep) x ω r).vLog.map Prod.fst) := by
  have ha := FlockDeploy.KS.run_acc_next H P (VK keep) x ω r h
  have hl := run_acc_log H P (VK keep) x ω r h
  generalize (List.range _).map (Claims.coinSeq r) = cs at ha hl
  replace ha : ((nextK keep x ((Claims.run H P (VK keep) x ω r).msgs.map Prod.fst) cs).eval H).1 = .acc := ha
  replace hl : ∀ p ∈ ((nextK keep x ((Claims.run H P (VK keep) x ω r).msgs.map Prod.fst) cs).eval H).2,
      p ∈ (Claims.run H P (VK keep) x ω r).vLog := hl
  cases hact : act false x ((Claims.run H P (VK keep) x ω r).msgs.map Prod.fst) cs with
  | rej => rw [nextK_rej keep H hact] at ha; cases ha
  | coin c => rw [nextK_coin keep H hact] at ha; cases ha
  | decide s spec sts link c₁ mids f =>
    refine ⟨s, spec, sts, link, c₁, mids, f, rfl, ?_, fun p hp => List.mem_map_of_mem (hl p ?_)⟩
    · simp only [nextK, hact] at ha
      exact outOf_acc H _ ha
    · simp only [nextK, hact, outOf_log]
      exact hp

/-- **A decision read through `keep`, within `L`**, at SHA-512 on a statement that sets up at `(spec, sts)`: the
verdict read through `keep` within `L`, and the checks (`decisionE_runs`'s proof). -/
theorem decisionK_runs (keep : Bool → Bool) {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    {L : List Claims.Msg} (hs : s.setup = .ok (spec, sts))
    (h : Runs sha512 (decisionK keep s x link c₁ mids f) () L) :
    Runs sha512 (verifyWithMK keep Claims.Hashes.oc spec (tabsOf sts) f.record f.proofs) () L ∧
      checks false s spec x link c₁ mids f = .ok () := by
  unfold decisionK at h
  obtain ⟨⟨spec', sts'⟩, h0, h⟩ := Runs.bind h
  have e : setupWith (Claims.hOf sha512) s.read = .ok (spec', sts') := by
    rw [← (setupM_eval sha512 s.read).1]
    exact h0.1
  rw [hOf_sha512, setupWith_sha512, Ins.setup_read hs] at e
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj e)
  unfold decisionAtK at h
  obtain ⟨⟨⟩, h1, h2⟩ := Runs.bind h
  exact ⟨h1, Runs.lift h2⟩

/-- **The verdict read through `keep` within `L` is that verdict with SHA-512 answered only within `L`.** -/
theorem verdictK_within (keep : Bool → Bool) {spec : SessionSpec} {tabs : Array Setup} {record : String}
    {proofs : Array ByteArray} {L : List Claims.Msg}
    (h : Runs sha512 (verifyWithMK keep Claims.Hashes.oc spec tabs record proofs) () L) :
    verifyWithMK keep (hashesIn L) spec tabs record proofs = .ok () :=
  (With.verifyWithMK_sim (within L) (within_hashes L) keep spec tabs record proofs) () h

end FlockDeploy.KS
