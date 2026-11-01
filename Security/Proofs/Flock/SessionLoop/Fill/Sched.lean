import Proofs.Flock.SessionLoop.Statement

/-!
# `FillRoundEq` at the loop's own frames: the round in a failed rep's place

`FillRoundEq` at `SessionLoopZKDeployed`'s release (`LeanRelease.ofLoopLink`) reads the round the release sends in a
failed rep's place (`scheduled`). This file says which round that is and splits `FillRoundEq` there:

* `LeanRelease.scheduled_ofLoopLink`: past `Hello`, the loop's own round (`Loop.schedOf`) on the rep's table and salts
  (`real`), and `L₀`'s where the loop schedules none.
* `Loop.InnerAt`: where it schedules none at a session's history: the round is rep stream `k`'s round `i`
  (`CoinLayer.locate`) at or past the end of the rounds `Loop.schedSoFar` schedules on the stream's table at the coins
  it answered, the inner proof's (`Frame.INNER_ROUNDS` of them, round `i - sched.size`: `Loop.InnerRound`,
  `Loop.innerAt_iff`). `Loop.schedOf_eq_none_iff`: at a statement that sets up a session (`0 < refK x`) and a history
  past the link, `schedOf` is none exactly there.
  `LeanRelease.scheduled_inner0`: at its first round `LoopInnerRound0` gives `L₀`'s round.
* `LeanRelease.FillSched`, `LeanRelease.FillInner`: `FillRoundEq`'s equation at the rounds the loop schedules
  (`schedOf`'s round, released as a stopped rep's) and at the inner proof's (`L₀`'s round, where `InnerAt` holds).
  `LeanRelease.fillRoundEq_ofLoopLink`: the two give `FillRoundEq`.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)
open _root_.Flock.Transcript (header)
open _root_.Flock.Firewall.Frame (labelFrame)
open _root_.Flock.Firewall.Finish (Tab)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoopLink`'s session types, as in `Statement`

namespace Loop

/-- Where `x` sets up no session there is no round past the link. -/
theorem refK_of_none {x : Msg} (hs : FlockDeploy.setupOf true x = none) : refK x = 0 := by
  unfold refK
  simp only [FlockDeploy.V, hs]

/-- **The inner proof's rounds** after the decoded coins `cs` at `x`, on the rep streams' tables and salts `real`: `x`
sets up a session, the round is rep stream `k`'s round `i` (`CoinLayer.locate` of rep round `cs.length - 1`), the
stream's table and salts are `real k`'s, and `i` is at or past the end of the rounds `schedSoFar` schedules there at the
coins the stream answered (`schedOf`'s): the stream's rounds past its scheduled ones, the inner proof's round
`i - sched.size`. -/
def InnerAt (real : ℕ → V (Tab × Array ByteArray)) (x : Msg) (cs : List FlockDeploy.Coin) : Prop :=
  ∃ (s0 : FlockDeploy.Ins) (spec : SessionSpec) (sts : Array Stmt) (q k i : ℕ) (t : Tab) (hi : Array ByteArray)
    (sched : Array (ByteArray × ℕ)),
    FlockDeploy.setupOf true x = some (s0, spec, sts) ∧ cs.length = q + 1 ∧
    CoinLayer.locate (roundsOf spec sts) 0 q = (k, i) ∧ real k = .ok (t, hi) ∧
    schedSoFar t (k % 2) (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8
      (_root_.Flock.Firewall.Drive.extras sts[k / 2]!
        (_root_.Flock.Firewall.Drive.linkPoints sts[k / 2]! (wordsAt x cs 0)))
      hi ((Array.range i).map fun j => wordsAt x cs (q - i + j + 1)) = .ok sched ∧
    sched.size ≤ i

/-- **The inner proof's round `n`** after the decoded coins `cs` at `x`: `InnerAt` at rep stream `k`'s round `i`, `n`
rounds past the `sched.size` rounds `schedSoFar` schedules there. -/
def InnerRound (real : ℕ → V (Tab × Array ByteArray)) (x : Msg) (cs : List FlockDeploy.Coin) (n : ℕ) : Prop :=
  ∃ (s0 : FlockDeploy.Ins) (spec : SessionSpec) (sts : Array Stmt) (q k i : ℕ) (t : Tab) (hi : Array ByteArray)
    (sched : Array (ByteArray × ℕ)),
    FlockDeploy.setupOf true x = some (s0, spec, sts) ∧ cs.length = q + 1 ∧
    CoinLayer.locate (roundsOf spec sts) 0 q = (k, i) ∧ real k = .ok (t, hi) ∧
    schedSoFar t (k % 2) (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8
      (_root_.Flock.Firewall.Drive.extras sts[k / 2]!
        (_root_.Flock.Firewall.Drive.linkPoints sts[k / 2]! (wordsAt x cs 0)))
      hi ((Array.range i).map fun j => wordsAt x cs (q - i + j + 1)) = .ok sched ∧
    i = sched.size + n

/-- `InnerAt` is the inner proof's round `n` for some `n`. -/
theorem innerAt_iff {real : ℕ → V (Tab × Array ByteArray)} {x : Msg} {cs : List FlockDeploy.Coin} :
    InnerAt real x cs ↔ ∃ n, InnerRound real x cs n := by
  constructor
  · rintro ⟨s0, spec, sts, q, k, i, t, hi, sched, hs, hq, hl, hr, hss, hle⟩
    exact ⟨i - sched.size, s0, spec, sts, q, k, i, t, hi, sched, hs, hq, hl, hr, hss, by omega⟩
  · rintro ⟨n, s0, spec, sts, q, k, i, t, hi, sched, hs, hq, hl, hr, hss, hn⟩
    exact ⟨s0, spec, sts, q, k, i, t, hi, sched, hs, hq, hl, hr, hss, by omega⟩

/-- `schedOf` at a session's history past the link, rep stream `k`'s round `i`: the stream's scheduled round `i` on its
table, the empty round where its table or schedule is refused. -/
theorem schedOf_at {real : ℕ → V (Tab × Array ByteArray)} {x : Msg} {cs : List FlockDeploy.Coin}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} {q k i : ℕ}
    (hs : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : cs.length = q + 1)
    (hl : CoinLayer.locate (roundsOf spec sts) 0 q = (k, i)) :
    schedOf real x cs =
      match real k with
      | .ok (t, hi) =>
        match schedSoFar t (k % 2) (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8
            (_root_.Flock.Firewall.Drive.extras sts[k / 2]!
              (_root_.Flock.Firewall.Drive.linkPoints sts[k / 2]! (wordsAt x cs 0)))
            hi ((Array.range i).map fun j => wordsAt x cs (q - i + j + 1)) with
        | .ok sched => sched[i]?.map (·.1)
        | .error _ => some .empty
      | .error _ => some .empty := by
  simp only [schedOf, hs, hq, hl]
  rfl

/-- **Exactly the inner proof's rounds have no scheduled round**: at a statement that sets up a session and a history
past the link, `schedOf` is none exactly at `InnerAt`. -/
theorem schedOf_eq_none_iff {real : ℕ → V (Tab × Array ByteArray)} {x : Msg} {cs : List FlockDeploy.Coin}
    (hk : 0 < refK x) (hc : 0 < cs.length) : schedOf real x cs = none ↔ InnerAt real x cs := by
  obtain ⟨q, hq⟩ : ∃ q, cs.length = q + 1 := ⟨cs.length - 1, by omega⟩
  rcases hs : FlockDeploy.setupOf true x with _ | ⟨s0, spec, sts⟩
  · exact absurd (refK_of_none hs) (by omega)
  rcases hl : CoinLayer.locate (roundsOf spec sts) 0 q with ⟨k, i⟩
  rw [schedOf_at hs hq hl]
  constructor
  · intro hn
    rcases hr : real k with e | ⟨t, hi⟩
    · rw [hr] at hn
      simp at hn
    · rw [hr] at hn
      dsimp only at hn
      rcases hss : schedSoFar t (k % 2) (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8
          (_root_.Flock.Firewall.Drive.extras sts[k / 2]!
            (_root_.Flock.Firewall.Drive.linkPoints sts[k / 2]! (wordsAt x cs 0)))
          hi ((Array.range i).map fun j => wordsAt x cs (q - i + j + 1)) with e | sched
      · rw [hss] at hn
        simp at hn
      · rw [hss] at hn
        exact ⟨s0, spec, sts, q, k, i, t, hi, sched, hs, hq, hl, hr, hss, by simpa using hn⟩
  · rintro ⟨s0', spec', sts', q', k', i', t, hi, sched, hs', hq', hl', hr, hss, hle⟩
    obtain ⟨rfl, rfl, rfl⟩ : s0' = s0 ∧ spec' = spec ∧ sts' = sts := by
      simpa only [Option.some.injEq, Prod.mk.injEq] using hs'.symm.trans hs
    obtain rfl : q' = q := by omega
    obtain ⟨rfl, rfl⟩ : k' = k ∧ i' = i := by
      simpa only [Prod.mk.injEq] using hl'.symm.trans hl
    rw [hr]
    dsimp only
    rw [hss]
    simp [hle]

end Loop

namespace LeanRelease

/-- **The round `ofLoopLink` sends in a failed rep's place**: past `Hello`, the loop's own (`Loop.schedOf`) on the rep's
table and salts `real`, and `L₀`'s where the loop schedules none (`Loop.InnerAt`: the inner proof's rounds). -/
theorem scheduled_ofLoopLink
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (x : Msg) (w : Flock.Wit)
    (ρ : (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)
    (p : Msg × Msg) (ops : List (Msg × Msg)) (vals : String → Option ByteArray) :
    (ofLoopLink L₀ src real).scheduled x w ρ (p :: ops) vals =
      match Loop.schedOf (real x w ρ) x (ops.map fun p => CoinLayer.rd p.2) with
      | some r => r.data.toList
      | none => L₀.scheduled x w ρ (p :: ops) vals :=
  rfl

/-- **`LoopInnerRound0` is `L₀`'s round at the inner proof's first round**: where the history `p :: ops` stands at rep
stream `k`'s round `i`, just past the `i` rounds `schedSoFar` schedules on its table (`InnerAt` with `sched.size = i`),
`L₀`'s round there is the inner proof's label frame and some 16-byte `final_c'` between its headers. -/
theorem scheduled_inner0
    {L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)}
    {real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray)}
    {x : Msg} {w : Flock.Wit} (h0 : _root_.Flock.Guarantees.LoopInnerRound0 Masked.dataE L₀ real x w)
    {ρ : (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand}
    {p : Msg × Msg} {ops : List (Msg × Msg)} (vals : String → Option ByteArray) {s0 : FlockDeploy.Ins}
    {spec : SessionSpec} {sts : Array Stmt} {q k i : ℕ} {t : Tab} {hi : Array ByteArray}
    {sched : Array (ByteArray × ℕ)} (hs : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : ops.length = q + 1)
    (hl : CoinLayer.locate (Loop.roundsOf spec sts) 0 q = (k, i)) (hr : real x w ρ k = .ok (t, hi))
    (hss : Loop.schedSoFar t (k % 2) (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8
      (_root_.Flock.Firewall.Drive.extras sts[k / 2]! (_root_.Flock.Firewall.Drive.linkPoints sts[k / 2]!
        (Loop.wordsAt x (ops.map fun p => CoinLayer.rd p.2) 0)))
      hi ((Array.range i).map fun j => Loop.wordsAt x (ops.map fun p => CoinLayer.rd p.2) (q - i + j + 1)) = .ok sched)
    (hsz : sched.size = i) (hpos : 0 < i) :
    ∃ w' : ByteArray, w'.size = 16 ∧ L₀.scheduled x w ρ (p :: ops) vals =
      (labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1 ++ w' ++ header 4 1 1).data.toList := by
  obtain ⟨j, rfl⟩ : ∃ j, i = j + 1 := ⟨i - 1, by omega⟩
  exact h0 ρ (p :: ops) vals s0 spec sts q k j t hi sched hs (by simp only [List.length_cons]; omega) hl hr hss hsz

/-- **`FillRoundEq` at the rounds the loop schedules**: its equation after every history it reads at which the loop
schedules its own round `r` (`Loop.schedOf`), `r` released as a stopped rep's. -/
def FillSched (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (κ : (ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (r : ByteArray),
    0 < ops.length → ops.length < refK x →
    (ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = false →
    ((allOk ((CoinLayer.session Loop.loopNr Masked.dataE).refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval
      CoinOpen.shaH).1 = true →
    (ofLoopLink L₀ src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    Loop.schedOf (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2) = some r →
    (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) true r.data.toList =
      dropDom ((CoinLayer.session Loop.loopNr Masked.dataE).swMsgAt x
        ((CoinLayer.session Loop.loopNr Masked.dataE).wt x w) (vs x w ν) t
        (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2))

/-- **`FillRoundEq` at the inner proof's rounds** (`Loop.InnerAt`): its equation after every history it reads at which the
loop schedules no round of its own, `L₀`'s round released as a stopped rep's. -/
def FillInner (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (κ : (ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)),
    0 < ops.length → ops.length < refK x →
    (ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = false →
    ((allOk ((CoinLayer.session Loop.loopNr Masked.dataE).refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval
      CoinOpen.shaH).1 = true →
    (ofLoopLink L₀ src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    Loop.InnerAt (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2) →
    (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) true
        (L₀.scheduled x w (ν, t) ((m₀, ρ₀) :: ops)
          (computedOf ((ofLoopLink L₀ src real).computed x w (ν, t) ((m₀, ρ₀) :: ops)))) =
      dropDom ((CoinLayer.session Loop.loopNr Masked.dataE).swMsgAt x
        ((CoinLayer.session Loop.loopNr Masked.dataE).wt x w) (vs x w ν) t
        (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2))

/-- **`FillRoundEq` from the scheduled rounds and the inner proof's**: at `SessionLoopZKDeployed`'s release, where the
loop schedules its own round (`FillSched`) and at the inner proof's rounds (`FillInner`). -/
theorem fillRoundEq_ofLoopLink (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit)
    (hS : FillSched vs L₀ src real wk x w) (hI : FillInner vs L₀ src real wk x w) :
    FillRoundEq (CoinLayer.session Loop.loopNr Masked.dataE) vs (ofLoopLink L₀ src real) CoinOpen.shaH wk x w := by
  intro ν t κ m₀ ρ₀ ops hpos hlt hlive hok href
  rw [scheduled_ofLoopLink]
  split
  · rename_i r hsc
    exact hS ν t κ m₀ ρ₀ ops r hpos hlt hlive hok href hsc
  · rename_i hsc
    exact hI ν t κ m₀ ρ₀ ops hpos hlt hlive hok href
      ((Loop.schedOf_eq_none_iff (by omega) (by simpa using hpos)).mp hsc)

end LeanRelease

end Claims.ZKServer
