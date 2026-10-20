import Proofs.Flock.Claims.ZK.Stopped

/-!
# The masked session's rounds where no refusal is seen

At a round after the first where the openings show no refusal, `flock-circuit --zk` sends the masked session's next
body. No answered outer coin is `1`, so `honestPR` is `honestP` at the tape's mask words; fed fewer answers than the
session sends, it sends the session's body at their count, tagged `0` (`FlockComplete.ZkTables.honestPR_sent`).

* `setupOf_of_refK`: a statement with a pre-final round sets up a session.
* `refused_of_outerHit`: an outer hit at the openings' answers is a refusal they show (`Refused`'s first kind), the
  converse of `outerHit_of_lt`.
* `liveRun`: the masked session on the mask witness at the tape, answered by the openings.
* `honestPR_live`: where the openings show no refusal and `liveRun` sends past them, `honestPR`'s message is
  `liveRun`'s next body; `honestPR_stopped`: where `liveRun` stops within them, it is `[]`.
* `Flock.zk_deployZk_masked_exe_sent_gap`: `Flock.zk_deployZk_masked_exe_live_gap` with the encoding asked against the
  session's run: where no refusal is seen, either `liveRun` sends past the openings and the encoding is its next body,
  tagged `0`, or it stops within them and the encoding is `[]` (`hstep`). The session can stop with no refusal the
  coins show: table `0`'s extra claims throw at a statement whose regions do not fit, right after the link message.
-/

namespace Claims.ZKMal

open Claims FlockSoundness.Discharge
open CoinLayer

namespace Masked

/-- **A statement with a pre-final round sets up a session**: where none sets up, the verifier has one round. -/
theorem setupOf_of_refK {x : Msg} (h : 0 < refK x) :
    ∃ s spec, FlockDeploy.setupOf true x = some (s, spec, stsOf x) := by
  unfold stsOf
  unfold refK at h
  cases hs : FlockDeploy.setupOf true x with
  | none => simp [FlockDeploy.V, hs] at h
  | some v =>
    obtain ⟨s, spec, sts⟩ := v
    exact ⟨s, spec, rfl⟩

/-- **An outer hit at the openings' answers is a refusal they show**: some table's rep's outer coin at an answered
slot is `1` at the opened coins (`outerAt`). -/
theorem refused_of_outerHit {x : Msg} (hx : Live x) (ops : List (Msg × Msg)) (hl : ops.length < refK x)
    (h : FlockComplete.ZkTables.outerHit x ((ops.map fun p => (p.1, coinOfD p.2)).map Prod.snd) = true) :
    Refused x (padOps ops) ops.length := by
  obtain ⟨s, spec, hsx⟩ := setupOf_of_refK (lt_of_le_of_lt (Nat.zero_le _) hl)
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hh : (ops.map fun p => (p.1, coinOfD p.2)).map Prod.snd = ops.map fun p => coinOfD p.2 := by
    rw [List.map_map]
    rfl
  rw [hh] at h
  unfold FlockComplete.ZkTables.outerHit at h
  rw [List.any_eq_true] at h
  obtain ⟨p, hp, hp'⟩ := h
  rw [Bool.and_eq_true, decide_eq_true_iff, decide_eq_true_iff, List.length_map] at hp'
  obtain ⟨hpl, hp1⟩ := hp'
  unfold FlockComplete.outerSlotsZk at hp
  rw [hsx] at hp
  simp only [List.mem_flatMap, List.mem_range, List.mem_map] at hp
  obtain ⟨q, hq, i, hi, rfl⟩ := hp
  have hj : q / 2 < (stsOf x).size := by omega
  have hs : outerSlot x (q / 2) (q % 2) = FlockComplete.repStartZk spec (stsOf x) q + 1 := by
    rw [outerSlot_eq hsx, repBase_eq hsx, show 2 * (q / 2) + q % 2 = q by omega]
  have hm35 : (stsOf x)[q / 2]!.m ≤ 35 := by
    rw [getElem!_pos (stsOf x) (q / 2) hj]
    exact (hall _ hj).2.1
  refine .inl ⟨q / 2, hj, q % 2, Nat.mod_lt _ (by norm_num), by rw [hs]; exact hpl, i, hi, ?_⟩
  change slotAt (padOps ops) (outerSlot x (q / 2) (q % 2)) i = 1
  rw [hs, slotAt_padOps_lt ops hpl (by omega) (by omega)]
  exact hp1

/-- **The masked session where the openings `ops` are answered**: on the mask witness (`maskWOf`) at the tape `ω`'s
mask words, with `hOf H` SHA-512, each reply the opening's coin. -/
def liveRun (x : Msg) (w : Claims.Flock.Wit) (ω : FlockComplete.TapeZk) (ops : List (Msg × Msg))
    (spec : _root_.Flock.SessionSpec) :
    Except String (String × Array ByteArray) × Array Msg :=
  (FlockComplete.sessionPZk (FlockComplete.asksS fun k => (ops.map fun p => coinOfD p.2).getD k [])
    (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) spec (FlockDeploy.tabsOf (stsOf x))
    (FlockComplete.MaskRank.masksOf (stsOf x))
    (FlockComplete.ZkTables.traces (stsOf x) (FlockComplete.ZkTables.maskWOf x w ω.mask)) ω).go #[]

/-- **Where the openings show no refusal, `honestPR` sends the session's next body**: after `n < refK x` openings `ops`
at which no refusal is seen, where `liveRun` sends more than `n` bodies, `honestPR`'s message is body `n`, tagged `0`
(`honestPR_sent`: no outer coin `1` is answered, `refused_of_outerHit`). -/
theorem honestPR_live (H : Msg → Msg) (hH : Claims.hOf H = _root_.Flock.Sha512.hash) {x : Msg} (hx : Live x)
    (w : Claims.Flock.Wit) (ω : FlockComplete.TapeZk) (ops : List (Msg × Msg)) (hl : ops.length < refK x)
    (hs : ¬ Refused x (padOps ops) ops.length) {s spec} (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x))
    (hsend : ops.length < (liveRun x w ω ops spec).2.size) :
    (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOfD p.2)) ω).eval H).1 =
      0 :: (liveRun x w ω ops spec).2[ops.length]! := by
  have hh : (ops.map fun p => (p.1, coinOfD p.2)).map Prod.snd = ops.map fun p => coinOfD p.2 := by
    rw [List.map_map]
    rfl
  have ho : FlockComplete.ZkTables.outerHit x (ops.map fun p => coinOfD p.2) = false := by
    rw [← hh]
    cases h : FlockComplete.ZkTables.outerHit x ((ops.map fun p => (p.1, coinOfD p.2)).map Prod.snd)
    · rfl
    · exact absurd (refused_of_outerHit hx ops hl h) hs
  have hrun : (FlockComplete.sessionPZk (FlockComplete.asksS fun k => (ops.map fun p => coinOfD p.2).getD k [])
      (_root_.Flock.Hashes.ofSha512 (Claims.hOf H)) spec (FlockDeploy.tabsOf (stsOf x))
      (FlockComplete.MaskRank.masksOf (stsOf x))
      (FlockComplete.ZkTables.traces (stsOf x) (FlockComplete.ZkTables.maskWOf x w ω.mask)) ω).go #[] =
        liveRun x w ω ops spec := by
    rw [hH]
    rfl
  have e := FlockComplete.ZkTables.honestPR_sent H x w ω hsx _ _ hh ho hrun
    (by rw [List.length_map]; exact hsend)
  rw [List.length_map] at e
  exact e

/-- **A session that stops within the openings sends `[]`**: where `liveRun` throws having sent at most the openings'
count, `honestPR`'s message is `[]` (`honestPR_stops`). -/
theorem honestPR_stopped (H : Msg → Msg) (hH : Claims.hOf H = _root_.Flock.Sha512.hash) {x : Msg}
    (w : Claims.Flock.Wit) (ω : FlockComplete.TapeZk) (ops : List (Msg × Msg)) {s spec}
    (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (he : ∃ e, (liveRun x w ω ops spec).1 = .error e)
    (hn : (liveRun x w ω ops spec).2.size ≤ ops.length) :
    (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOfD p.2)) ω).eval H).1 = [] := by
  have hh : (ops.map fun p => (p.1, coinOfD p.2)).map Prod.snd = ops.map fun p => coinOfD p.2 := by
    rw [List.map_map]
    rfl
  refine FlockComplete.ZkTables.honestPR_stops H x w ω hsx _ _ hh fun R L hY => ?_
  have hrun : liveRun x w ω ops spec = (R, L) := by
    rw [← hY, hH]
    rfl
  rw [hrun] at he hn
  exact ⟨he, by rw [List.length_map]; exact hn⟩

/-- **`hlive` from the session's run**: where no refusal is seen after the first round, either `liveRun` sends past the
openings and the encoding of the cut view is its next body, tagged `0`, or it stops within them and the encoding is
`[]` (`hstep`). -/
theorem hlive_of_sent {P : Composed.LigProver}
    {cz : (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (tabC P x j).Rest}
    {enc : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → ℕ → (∀ j, ((tabC P x j).T (cz x c j)).tab.Cut) → Msg}
    {fin : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → (∀ j, ((tabC P x j).T (cz x c j)).tab.View) → Msg}
    (U : TapePlan (tabC P)) (nr : Msg → ℕ → ℕ) (H : Msg → Msg) (hH : Claims.hOf H = _root_.Flock.Sha512.hash)
    (hstep : ∀ x, Live x → ∀ w, RZk x w → ∀ (τ : U.Tape) (ops : List (Msg × Msg)), 0 < ops.length →
      ops.length < refK x → ¬ Refused x (padOps ops) ops.length →
      ∀ s spec, FlockDeploy.setupOf true x = some (s, spec, stsOf x) →
        (ops.length < (liveRun x w (U.tapeZk τ) ops spec).2.size ∧
          enc x (padOps ops) ops.length
              ((session nr (tableData P (RoundPlan.stop cz enc fin) U)).cutOf x (padOps ops) (wtOf P x w) τ) =
            0 :: (liveRun x w (U.tapeZk τ) ops spec).2[ops.length]!) ∨
        ((∃ e, (liveRun x w (U.tapeZk τ) ops spec).1 = .error e) ∧
          (liveRun x w (U.tapeZk τ) ops spec).2.size ≤ ops.length ∧
          enc x (padOps ops) ops.length
              ((session nr (tableData P (RoundPlan.stop cz enc fin) U)).cutOf x (padOps ops) (wtOf P x w) τ) = [])) :
    ∀ x, Live x → ∀ w, RZk x w → ∀ (τ : U.Tape) (ops : List (Msg × Msg)), 0 < ops.length →
      ops.length < refK x → ¬ Refused x (padOps ops) ops.length →
      enc x (padOps ops) ops.length
          ((session nr (tableData P (RoundPlan.stop cz enc fin) U)).cutOf x (padOps ops) (wtOf P x w) τ) =
        (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOfD p.2)) (U.tapeZk τ)).eval H).1 := by
  intro x hx w hR τ ops h0 hl hs
  obtain ⟨s, spec, hsx⟩ := setupOf_of_refK (lt_of_le_of_lt (Nat.zero_le _) hl)
  rcases hstep x hx w hR τ ops h0 hl hs s spec hsx with ⟨hn, hb⟩ | ⟨he, hn, hb⟩
  · rw [honestPR_live H hH hx w (U.tapeZk τ) ops hl hs hsx hn]
    exact hb
  · rw [honestPR_stopped H hH w (U.tapeZk τ) ops hsx he hn]
    exact hb

end Masked

open Masked

open Classical in
/-- **Flock's zero knowledge at the deployed masked session, at the executable's layout and a plan that stops where a
refusal is seen, against the session's run**: `Flock.zk_deployZk_masked_exe_live_gap` with its encoding asked at the
masked session's run where no refusal is seen (`hlive_of_sent`). Left: that there the encoding is the session's next
body past the openings, or `[]` where the session stops within them (`hstep`), the plan's `Causal`, its round `0` and
the last round where every table admits. -/
theorem Flock.zk_deployZk_masked_exe_sent_gap (H : Msg → Msg) (hH : Claims.hOf H = _root_.Flock.Sha512.hash)
    (nr : Msg → ℕ → ℕ) (P : Composed.LigProver)
    {cz : (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (tabC P x j).Rest}
    {enc : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → ℕ → (∀ j, ((tabC P x j).T (cz x c j)).tab.Cut) → Msg}
    {fin : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → (∀ j, ((tabC P x j).T (cz x c j)).tab.View) → Msg}
    (h0 : ∀ x c cut, enc x c 0 cut = enc0 P x (cz x c) cut) (t : ℕ) (ht : 0 < t)
    (hc : (RoundPlan.stop cz enc fin).Causal) (ho : OuterReads cz) (hr : CoinReads cz) (hp : PointReads cz)
    (hstep : ∀ x, Live x → ∀ w, RZk x w → ∀ τ (ops : List (Msg × Msg)), 0 < ops.length →
      ops.length < refK x → ¬ Refused x (padOps ops) ops.length →
      ∀ s spec, FlockDeploy.setupOf true x = some (s, spec, stsOf x) →
        (ops.length < (liveRun x w ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk τ) ops spec).2.size ∧
          enc x (padOps ops) ops.length ((session nr (tableData P (RoundPlan.stop cz enc fin)
              (TapePlan.ofLayout (TapeLayout.exe P) exeTape))).cutOf x (padOps ops) (wtOf P x w) τ) =
            0 :: (liveRun x w ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk τ) ops spec).2[ops.length]!) ∨
        ((∃ e, (liveRun x w ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk τ) ops spec).1 = .error e) ∧
          (liveRun x w ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk τ) ops spec).2.size ≤ ops.length ∧
          enc x (padOps ops) ops.length ((session nr (tableData P (RoundPlan.stop cz enc fin)
              (TapePlan.ofLayout (TapeLayout.exe P) exeTape))).cutOf x (padOps ops) (wtOf P x w) τ) = []))
    (hfin : (session nr (tableData P (RoundPlan.stop cz (encStop enc) fin)
      (TapePlan.ofLayout (TapeLayout.exe P) exeTape))).HonestFinOn Live RZk H coinOfD
        fun ω => (TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk ω.2) :
    Claims.Flock.ZK H (zkDeployed _root_.Flock.Readable.relation nr
        (tableData P (RoundPlan.stop cz (encStop enc) fin) (TapePlan.ofLayout (TapeLayout.exe P) exeTape)) t)
      (fun x => if Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x then
        8 * nL P x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash +
          8 * t * nL P x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash + 2 * (1 / 3) ^ t else 1)
      fun _ => 4 * t * 3 ^ t :=
  Flock.zk_deployZk_masked_exe_live_gap H hH nr P h0 t ht hc ho hr hp
    (hlive_of_sent _ nr H hH hstep) hfin

end Claims.ZKMal
