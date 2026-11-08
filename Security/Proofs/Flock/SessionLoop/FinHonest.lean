module

public import Proofs.Flock.SessionLoop.LiveInnerOwn
public import Proofs.Flock.SessionLoop.InnerSlot

/-!
# The finish's rounds at the honest values

`withInner` takes the rounds the finish hands a rep's frame for its inner proof as a parameter `fin`, of which
`Frame.innerSched` reads round 0's `final_c'` and round 2's `τ` commitment. The loop's finish computes them from the
history's messages (`Loop.finOf`); at the honest values they are the words the worker that sends the scheduled rounds
with them writes there (`Loop.valsDev`), which read the history only through its coins:

* **`Loop.finHonest`**: round 0 the inner proof's label and the honest `final_c` under the rep's pad at `final_c`, round 2
  the honest `τ` commitment (`Loop.vsAt`, `Loop.vsTauAt`), at a release's configuration and rep.
  `Loop.innerSched_finHonest0`, `Loop.innerSched_finHonest2`: the frame's own inner rounds 0 and 2 on them are those
  rounds. `withInner` keeps the configuration and rep of the release it wraps (`Loop.cfg_ofLoopLink_withInner`,
  `Loop.vsAt_ofLoopLink_withInner`, `Loop.vsTauAt_ofLoopLink_withInner`), so `finHonest` is taken at the release
  without the finish's rounds.
* **The transfer, on the honest worker's own runs**: at every history the deferred server meets (`ZkServer.OwnHist`),
  the finish's rounds 0 and 2 are `finHonest`'s at their own inner rounds (`LeanRelease.finOf_finHonest0_own`,
  `LeanRelease.finOf_finHonest2_own`: `finOf_finalC_own` and `finOf_tau_own` at the honest values), so `withInner`
  schedules the same round at the loop's finish and at `finHonest` (`LeanRelease.scheduled_finOf_own`): the frame's own
  inner round reads the finish's rounds only at their own index (`Loop.innerSched_congr`, `Loop.innerOf_congr`).
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)
open _root_.Flock.Firewall.Frame (Cfg Masker INNER_ROUNDS innerSched ownRound)
open _root_.Flock.Firewall.Finish (Tab)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoopLink`'s session types, as in `Statement`

namespace Loop

/-- **The finish's rounds at the honest values** after the history `h` at `x`, the witness `w` and the tape `ρ`, at the
release `L`'s configuration and rep: round 0 the inner proof's label and the honest `final_c` under the rep's pad at
`final_c` (`vsAt`), round 2 the honest `τ` commitment (`vsTauAt`), the words `valsDev` writes there. They read the
history only through its coins. -/
noncomputable def finHonest (L : LeanRelease Flock.Wit Tp) (x : Msg) (w : Flock.Wit) (ρ : Tp) (h : List (Msg × Msg)) :
    Array ByteArray :=
  #[inner0 (vsAt finS.honestVals L x w ρ h (L.cfg x ρ h).lay.fc +
      (L.cfg x ρ h).h.getD (L.cfg x ρ h).lay.fc 0).toBytes, .empty,
    inner2 (vsTauAt finS.honestVals L x w ρ h), .empty]

theorem finHonest_getD0 (L : LeanRelease Flock.Wit Tp) (x : Msg) (w : Flock.Wit) (ρ : Tp) (h : List (Msg × Msg)) :
    (finHonest L x w ρ h).getD 0 .empty = inner0 (vsAt finS.honestVals L x w ρ h (L.cfg x ρ h).lay.fc +
      (L.cfg x ρ h).h.getD (L.cfg x ρ h).lay.fc 0).toBytes :=
  rfl

theorem finHonest_getD2 (L : LeanRelease Flock.Wit Tp) (x : Msg) (w : Flock.Wit) (ρ : Tp) (h : List (Msg × Msg)) :
    (finHonest L x w ρ h).getD 2 .empty = inner2 (vsTauAt finS.honestVals L x w ρ h) :=
  rfl

theorem ownRound_zero_congr (c : Cfg) (s : Masker) (w t t' : ByteArray) (n : ℕ) :
    ownRound c s w t n 0 = ownRound c s w t' n 0 :=
  rfl

theorem ownRound_two_congr (c : Cfg) (s : Masker) (w w' t : ByteArray) (n : ℕ) :
    ownRound c s w t n 2 = ownRound c s w' t n 2 :=
  rfl

/-- **The frame's own inner round reads the finish's rounds only at its own index**: round 0 reads round 0's, round 2
round 2's, and rounds 1 and 3 neither. -/
theorem innerSched_congr (c : Cfg) (s : Masker) (n : ℕ) {fin fin' : Array ByteArray} :
    ∀ k, (k = 0 → fin.getD 0 .empty = fin'.getD 0 .empty) → (k = 2 → fin.getD 2 .empty = fin'.getD 2 .empty) →
      innerSched c s fin n k = innerSched c s fin' n k
  | 0, h0, _ => by
    unfold innerSched
    rw [h0 rfl]
    exact ownRound_zero_congr _ _ _ _ _ _
  | 1, _, _ => rfl
  | 2, _, h2 => by
    unfold innerSched
    rw [h2 rfl]
    exact ownRound_two_congr _ _ _ _ _ _
  | _ + 3, _, _ => rfl

/-- **The frame's own inner round 0 on `finHonest`** is the inner proof's label and the honest `final_c` under the
pad. -/
theorem innerSched_finHonest0 (L : LeanRelease Flock.Wit Tp) (x : Msg) (w : Flock.Wit) (ρ : Tp)
    (h : List (Msg × Msg)) (c : Cfg) (s : Masker) (n : ℕ) :
    innerSched c s (finHonest L x w ρ h) n 0 = inner0 (vsAt finS.honestVals L x w ρ h (L.cfg x ρ h).lay.fc +
      (L.cfg x ρ h).h.getD (L.cfg x ρ h).lay.fc 0).toBytes := by
  dsimp only [innerSched, ownRound]
  rw [finHonest_getD0, wordBytes_words_inner0]
  rfl

/-- **The frame's own inner round 2 on `finHonest`** is the honest `τ` commitment's, at a rep of the tables. -/
theorem innerSched_finHonest2 (L : LeanRelease Flock.Wit Tp) (x : Msg) (w : Flock.Wit) (ρ : Tp)
    (h : List (Msg × Msg)) (c : Cfg) (s : Masker) (n : ℕ) {j : Fin (Masked.jOf x)} {r : Fin 2}
    (hrep : L.rep x h.length = 2 * j.val + r.val) :
    innerSched c s (finHonest L x w ρ h) n 2 = inner2 (vsTauAt finS.honestVals L x w ρ h) := by
  dsimp only [innerSched, ownRound]
  rw [finHonest_getD2, wordBytes_words_inner2 (vsTauAt_size _ L x w ρ h hrep)]
  rfl

/-- `withInner` keeps the configuration of the release it wraps. -/
theorem cfg_ofLoopLink_withInner (L₁ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Tp → ℕ → V (Tab × Array ByteArray))
    (fin : Msg → Flock.Wit → Tp → List (Msg × Msg) → Array ByteArray) (x : Msg) (ρ : Tp) (h : List (Msg × Msg)) :
    (LeanRelease.ofLoopLink (LeanRelease.withInner Masked.dataE L₁ real fin) src real).cfg x ρ h =
      (LeanRelease.ofLoopLink L₁ src real).cfg x ρ h := by
  unfold LeanRelease.cfg LeanRelease.drawsAt
  rfl

/-- `withInner` keeps the values `vsAt` reads at the release it wraps: they read its rep. -/
theorem vsAt_ofLoopLink_withInner (vs : finS.SwVals) (L₁ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Tp → ℕ → V (Tab × Array ByteArray))
    (fin : Msg → Flock.Wit → Tp → List (Msg × Msg) → Array ByteArray) (x : Msg) (w : Flock.Wit) (ρ : Tp)
    (h : List (Msg × Msg)) (i : ℕ) :
    vsAt vs (LeanRelease.ofLoopLink (LeanRelease.withInner Masked.dataE L₁ real fin) src real) x w ρ h i =
      vsAt vs (LeanRelease.ofLoopLink L₁ src real) x w ρ h i :=
  rfl

theorem vsTauAt_ofLoopLink_withInner (vs : finS.SwVals) (L₁ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Tp → ℕ → V (Tab × Array ByteArray))
    (fin : Msg → Flock.Wit → Tp → List (Msg × Msg) → Array ByteArray) (x : Msg) (w : Flock.Wit) (ρ : Tp)
    (h : List (Msg × Msg)) :
    vsTauAt vs (LeanRelease.ofLoopLink (LeanRelease.withInner Masked.dataE L₁ real fin) src real) x w ρ h =
      vsTauAt vs (LeanRelease.ofLoopLink L₁ src real) x w ρ h :=
  rfl

/-- **The loop's own inner round at two finishes' rounds**: where they give the frame's own inner round alike at every
located rep past its schedule, at its index there, `innerOf` is the same at both. -/
theorem innerOf_congr (real : ℕ → V (Tab × Array ByteArray)) (c : Cfg) (x : Msg) (cs : List FlockDeploy.Coin)
    {fin fin' : Array ByteArray}
    (h : ∀ (s0 : FlockDeploy.Ins) (spec : SessionSpec) (sts : Array Stmt) (q k i : ℕ) (t : Tab)
      (hi : Array ByteArray) (sched : Array (ByteArray × ℕ)), FlockDeploy.setupOf true x = some (s0, spec, sts) →
      cs.length = q + 1 → CoinLayer.locate (roundsOf spec sts) 0 q = (k, i) → real k = .ok (t, hi) →
      schedSoFar t (k % 2) (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8
        (_root_.Flock.Firewall.Drive.extras sts[k / 2]!
          (_root_.Flock.Firewall.Drive.linkPoints sts[k / 2]! (wordsAt x cs 0)))
        hi ((Array.range i).map fun j => wordsAt x cs (q - i + j + 1)) = .ok sched →
      sched.size ≤ i → i - sched.size < INNER_ROUNDS →
      ∀ (s : Masker) (n : ℕ), innerSched c s fin n (i - sched.size) = innerSched c s fin' n (i - sched.size)) :
    innerOf real c fin x cs = innerOf real c fin' x cs := by
  rcases hsx : FlockDeploy.setupOf true x with _ | ⟨s0, spec, sts⟩
  · unfold innerOf
    simp only [hsx]
  rcases hq : cs.length with _ | q
  · unfold innerOf
    simp only [hsx, hq]
  rcases hl : CoinLayer.locate (roundsOf spec sts) 0 q with ⟨k, i⟩
  rcases hr : real k with e | ⟨t, hi⟩
  · unfold innerOf
    rw [hsx, hq]
    simp only [hl, hr]
  rcases hs : schedSoFar t (k % 2) (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8
      (_root_.Flock.Firewall.Drive.extras sts[k / 2]!
        (_root_.Flock.Firewall.Drive.linkPoints sts[k / 2]! (wordsAt x cs 0)))
      hi ((Array.range i).map fun j => wordsAt x cs (q - i + j + 1)) with e | sched
  · unfold innerOf
    rw [hsx, hq]
    simp only [hl, hr, hs]
  unfold innerOf
  rw [hsx, hq]
  simp only [hl, hr, hs, Array.size_map, Array.size_range]
  by_cases hc : sched.size ≤ i ∧ i - sched.size < INNER_ROUNDS
  · rw [if_pos hc, if_pos hc, h s0 spec sts q k i t hi sched hsx hq hl hr hs hc.1 hc.2]
  · rw [if_neg hc, if_neg hc]

end Loop

namespace LeanRelease

/-- **`withInner`'s round at two finishes' rounds**: where the loop's own inner round is the same at both, so is the
round in the device's place. -/
theorem withInner_scheduled_congr (T : CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    {fin fin' : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand →
      List (Msg × Msg) → Array ByteArray}
    (x : Msg) (w : Flock.Wit)
    (ρ : (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand)
    (h : List (Msg × Msg)) (vals : String → Option ByteArray)
    (e : Loop.innerOf (real x w ρ) ((withFrames T L₀ real).cfg x ρ h) (fin x w ρ h) x
        (h.tail.map fun p => CoinLayer.rd p.2) =
      Loop.innerOf (real x w ρ) ((withFrames T L₀ real).cfg x ρ h) (fin' x w ρ h) x
        (h.tail.map fun p => CoinLayer.rd p.2)) :
    (withInner T L₀ real fin).scheduled x w ρ h vals = (withInner T L₀ real fin').scheduled x w ρ h vals := by
  show (match Loop.innerOf (real x w ρ) ((withFrames T L₀ real).cfg x ρ h) (fin x w ρ h) x
      (h.tail.map fun p => CoinLayer.rd p.2) with
    | some r => r.data.toList
    | none => L₀.scheduled x w ρ h vals) =
    (match Loop.innerOf (real x w ρ) ((withFrames T L₀ real).cfg x ρ h) (fin' x w ρ h) x
      (h.tail.map fun p => CoinLayer.rd p.2) with
    | some r => r.data.toList
    | none => L₀.scheduled x w ρ h vals)
  rw [e]

/-- **The finish's round 0 at the honest worker's own run is `finHonest`'s**: at a history the deferred server meets
(`ZkServer.OwnHist`), live, whose openings pass and whose coins the frame takes, whose next round is rep `2j + r`'s inner
round 0, the loop's finish hands the frame the inner proof's label and the honest `final_c` under the configuration's
pad (`finOf_finalC_own` at the honest values), which is `finHonest`'s at the release without the finish's rounds. -/
theorem finOf_finHonest0_own (L₁ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray))
    (hreal : real = Loop.realOf src) (x : Msg) (w : Flock.Wit) (hTab : Loop.finS.Tables)
    (hTape : Loop.finS.TapeOK) (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w)
    (hSv : Loop.Serves x) (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hSrc : Loop.Src (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src x w) (hc : Loop.finS.Causal)
    (ν : Loop.finS.Nonce) (t : Loop.finS.Tape)
    (κ : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).toSrv.Chk) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) (h0 : 0 < ops.length) (hlt : ops.length < refK x)
    (hl : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).toSrv.live
      (Loop.valsWk Loop.finS.honestVals
        (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real))
      x w (ν, t) κ ((m₀, ρ₀) :: ops) = true)
    (hok : ((allOk (Loop.finS.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval CoinOpen.shaH).1 = true)
    (hr : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).refusesCoins x (ν, t)
      ((m₀, ρ₀) :: ops) = false)
    (hown : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).toSrv.OwnHist
      (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).fillOf
      (Loop.valsWk Loop.finS.honestVals
        (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real))
      x w (ν, t) κ ((m₀, ρ₀) :: ops))
    (hI : Loop.InnerAt (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2))
    (j : Fin (Masked.jOf x)) (r : Fin 2)
    (hrep : (ofLoopLink L₁ src real).rep x (ops.length + 1) = 2 * j.val + r.val)
    (hslot : ops.length = Masked.repSlot x (2 * j.val + r.val) (Masked.innerOff 0)) :
    (Loop.finOf real L₁.draws x w (ν, t) ((m₀, ρ₀) :: ops)).getD 0 .empty =
      (Loop.finHonest (ofLoopLink L₁ src real) x w (ν, t) ((m₀, ρ₀) :: ops)).getD 0 .empty := by
  have hlast : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).last x =
      refK x + 1 :=
    loop_last Loop.loopNr Masked.dataE _ x
  have hkd : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).kindAt x
      ((m₀, ρ₀) :: ops).length = .masked :=
    kindAt_masked_of _ (by rw [List.length_cons]; omega) (by rw [hlast, List.length_cons]; omega)
  rw [finOf_finalC_own Loop.finS.honestVals L₁ src real hreal x w hR hSv hsc hSrc
      (Loop.swCausalOn_swOk Masked.dataE _ (RefSession.swCausal_vals _ hc)) _
      (liveSchedOwn_honestVals _ src real hreal hTab hTape hR hSv hsc hSrc) (Loop.finalCFixed_honestVals x w)
      ν t κ m₀ ρ₀ ops _ h0 hlt hl hok hr hown (Loop.valsWk_masked _ _ hkd) hI j r hrep hslot,
    Loop.finHonest_getD0, Loop.vsAt_eq _ _ x w (ν, t) ((m₀, ρ₀) :: ops) _ hrep, Loop.cfg_ofLoopLink_withInner]
  all_goals rfl

/-- **The finish's round 2 at the honest worker's own run is `finHonest`'s**: at a history the deferred server meets,
as in `finOf_finHonest0_own`, whose next round is rep `2j + r`'s inner round 2, the loop's finish hands the frame the
honest `τ` commitment (`finOf_tau_own` at the honest values, whose rows hold: `Loop.rowsOK_honestVals`), which is
`finHonest`'s. -/
theorem finOf_finHonest2_own (L₁ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray))
    (hreal : real = Loop.realOf src) (x : Msg) (w : Flock.Wit) (hTab : Loop.finS.Tables)
    (hTape : Loop.finS.TapeOK) (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w)
    (hSv : Loop.Serves x) (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hSrc : Loop.Src (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src x w) (hc : Loop.finS.Causal)
    (ν : Loop.finS.Nonce) (t : Loop.finS.Tape)
    (κ : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).toSrv.Chk) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) (h0 : 0 < ops.length) (hlt : ops.length < refK x)
    (hl : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).toSrv.live
      (Loop.valsWk Loop.finS.honestVals
        (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real))
      x w (ν, t) κ ((m₀, ρ₀) :: ops) = true)
    (hok : ((allOk (Loop.finS.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval CoinOpen.shaH).1 = true)
    (hr : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).refusesCoins x (ν, t)
      ((m₀, ρ₀) :: ops) = false)
    (hown : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).toSrv.OwnHist
      (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).fillOf
      (Loop.valsWk Loop.finS.honestVals
        (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real))
      x w (ν, t) κ ((m₀, ρ₀) :: ops))
    (hI : Loop.InnerAt (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2))
    (j : Fin (Masked.jOf x)) (r : Fin 2)
    (hrep : (ofLoopLink L₁ src real).rep x (ops.length + 1) = 2 * j.val + r.val)
    (hslot : ops.length = Masked.repSlot x (2 * j.val + r.val) (Masked.innerOff 2)) :
    (Loop.finOf real L₁.draws x w (ν, t) ((m₀, ρ₀) :: ops)).getD 2 .empty =
      (Loop.finHonest (ofLoopLink L₁ src real) x w (ν, t) ((m₀, ρ₀) :: ops)).getD 2 .empty := by
  have hlast : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).last x =
      refK x + 1 :=
    loop_last Loop.loopNr Masked.dataE _ x
  have hkd : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).kindAt x
      ((m₀, ρ₀) :: ops).length = .masked :=
    kindAt_masked_of _ (by rw [List.length_cons]; omega) (by rw [hlast, List.length_cons]; omega)
  rw [finOf_tau_own Loop.finS.honestVals L₁ src real hreal x w hR hSv hsc hSrc
      (Loop.swCausalOn_swOk Masked.dataE _ (RefSession.swCausal_vals _ hc)) _
      (liveSchedOwn_honestVals _ src real hreal hTab hTape hR hSv hsc hSrc) (Loop.finalCFixed_honestVals x w)
      (Loop.rowsOK_honestVals hR) ν t κ m₀ ρ₀ ops _ h0 hlt hl hok hr hown (Loop.valsWk_masked _ _ hkd) hI j r hrep
      hslot,
    Loop.finHonest_getD2, Loop.vsTauAt_eq _ _ x w (ν, t) ((m₀, ρ₀) :: ops) hrep]
  all_goals rfl

/-- **The transfer: `withInner` schedules the same round at the loop's finish and at `finHonest` on the honest worker's
own runs**. At a history the deferred server meets (`ZkServer.OwnHist`), live, whose openings pass and whose coins the
frame takes, the round in the device's place at `withInner` on the loop's finish's rounds (`Loop.finOf`) is the one at
`finHonest`'s: past a rep's schedule the frame's own inner round reads the finish's round 0 at its round 0 and its round
2 at its round 2 (`Loop.innerOf_congr`, `Loop.innerSched_congr`), which are `finHonest`'s there
(`finOf_finHonest0_own`, `finOf_finHonest2_own`), and elsewhere `withInner` sends `L₁`'s round at both. -/
theorem scheduled_finOf_own (L₁ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray))
    (hreal : real = Loop.realOf src) (x : Msg) (w : Flock.Wit) (hTab : Loop.finS.Tables)
    (hTape : Loop.finS.TapeOK) (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w)
    (hSv : Loop.Serves x) (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hSrc : Loop.Src (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src x w) (hc : Loop.finS.Causal)
    (ν : Loop.finS.Nonce) (t : Loop.finS.Tape)
    (κ : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).toSrv.Chk) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) (h0 : 0 < ops.length) (hlt : ops.length < refK x)
    (hl : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).toSrv.live
      (Loop.valsWk Loop.finS.honestVals
        (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real))
      x w (ν, t) κ ((m₀, ρ₀) :: ops) = true)
    (hok : ((allOk (Loop.finS.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval CoinOpen.shaH).1 = true)
    (hr : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).refusesCoins x (ν, t)
      ((m₀, ρ₀) :: ops) = false)
    (hown : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).toSrv.OwnHist
      (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).fillOf
      (Loop.valsWk Loop.finS.honestVals
        (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real))
      x w (ν, t) κ ((m₀, ρ₀) :: ops))
    (vals : String → Option ByteArray) :
    (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)).scheduled x w (ν, t) ((m₀, ρ₀) :: ops) vals =
      (withInner Masked.dataE L₁ real (Loop.finHonest (ofLoopLink L₁ src real))).scheduled x w (ν, t)
        ((m₀, ρ₀) :: ops) vals := by
  have hx := Masked.live_of hsc hR
  refine withInner_scheduled_congr Masked.dataE L₁ real x w (ν, t) _ vals
    (Loop.innerOf_congr _ _ x _ fun s0 spec sts q k i tr hi sched hsx hq hloc hre hs hle _ s n => ?_)
  have hq1 : ops.length = q + 1 := by
    rw [List.tail_cons, List.length_map] at hq
    exact hq
  have hRT : Count.RealTabs sts (real x w (ν, t)) := by
    subst hreal
    exact Loop.realTabs_of src x w (ν, t) hsx
  obtain ⟨jf, rf, hk, -, hslot, -⟩ := Loop.innerAt_slot hx hSv hsx hq (by simpa using hlt) hloc hRT hre hs hle
  have hI : Loop.InnerAt (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2) :=
    ⟨s0, spec, sts, q, k, i, tr, hi, sched, hsx, hq, hloc, hre, hs, hle⟩
  have hrep : (ofLoopLink L₁ src real).rep x (ops.length + 1) = 2 * jf.val + rf.val := by
    show Loop.repOf x (ops.length + 1) = _
    rw [show ops.length + 1 = q + 2 by omega, repOf_add_two hsx, hloc]
    exact hk
  have hsl : ops.length = Masked.repSlot x (2 * jf.val + rf.val) (Masked.innerOff (i - sched.size)) :=
    hq1.trans hslot
  generalize hE : i - sched.size = e at hsl ⊢
  refine Loop.innerSched_congr _ s n e (fun he => ?_) (fun he => ?_)
  · subst he
    exact finOf_finHonest0_own L₁ src real hreal x w hTab hTape hR hSv hsc hSrc hc ν t κ m₀ ρ₀ ops h0 hlt hl hok hr
      hown hI jf rf hrep hsl
  · subst he
    exact finOf_finHonest2_own L₁ src real hreal x w hTab hTape hR hSv hsc hSrc hc ν t κ m₀ ρ₀ ops h0 hlt hl hok hr
      hown hI jf rf hrep hsl

end LeanRelease

end Claims.ZKServer
