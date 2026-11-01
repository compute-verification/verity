import Proofs.Flock.SessionLoop.Statement
import Proofs.Flock.Claims.ZK.FinishSwMkAny
import Proofs.Flock.Claims.ZK.FinishSwFrame
import Proofs.Flock.Claims.ZK.FrameFree
import Proofs.Flock.Claims.ZK.L0PinLive
import Proofs.Flock.Claims.ZK.PadsRoot

/-!
# The values the frame took, as the switched session's values

The session loop's frame takes each value of a rep's algebraic phase as the free word the worker sent there, and
releases it masked, `v + h_i` under the rep's pad `h_i` (`Frame.Masker.absorb`); the loop reads the value back as the
released word plus the pad (`Loop.valsOf`). The switched session at the deployed tables (`CoinLayer.session Loop.loopNr
Masked.dataE`) masks the values `vs` gives at the opened coins the same way (`Masked.swSrcE_masked_pad_any`). This file
states the switched session's side by pad index, as the firewall's words:

* **`Loop.vsWord`**: table `j`'s rep `r` value `k` of `vs` at the opened coins, at the table's base on the tape (its
  mask slot and extra lanes, which a table's values may read), as the firewall's word; `0` from `n_E` on.
* **`Loop.RowsOK`**: each table's rows hold at `vs` at its base, at every nonce, opened coins and tape. Where they fail
  the switched session commits its pads tree over the moved pads (`Table.swPads`), while the frame commits it over the
  drawn ones, so a rep's first round, which carries the pads root, is the switched message only where they hold.
* **`Loop.TakesVals`**: at every live round, each word the frame frees in the worker's round is `vs`'s value there.
* **`Loop.TakesFinalC`**: at the inner proof's round 0, which the frame frees nothing of, the worker's `final_c'` plus
  its pad is `vs`'s `final_c`.
* **`Loop.swSrcE_masked_vsWord`**: below `n_E`, the switched source's masked value is `vsWord` plus the session's pad
  (`padsOf`), with no premise on the rows.

`LiveRoundEq` at `SessionLoopZKDeployed`'s release (`LeanRelease.ofLoopLink`) splits where the loop schedules a round of
its own and where it doesn't:

* `LeanRelease.scheduled_ofLoopLink`: past `Hello`, the round in a device round's place is the loop's own
  (`Loop.schedOf`) on the rep's table and salts `real`, and `L₀`'s where the loop schedules none.
* `Loop.InnerAt`: where it schedules none at a session's history: the round is rep stream `k`'s round `i`
  (`CoinLayer.locate`) at or past the end of the rounds `Loop.schedSoFar` schedules on the stream's table at the coins
  it answered, the inner proof's (`Frame.INNER_ROUNDS` of them, round `i - sched.size`). `Loop.schedOf_eq_none_iff`: at
  a statement that sets up a session (`0 < refK x`) and a history past the link, `schedOf` is none exactly there.
* `LeanRelease.LiveSched`, `LeanRelease.LiveInner`: `LiveRoundEq`'s equation at the rounds the loop schedules and at
  the inner proof's. `LeanRelease.liveRoundEq_ofLoopLink`: the two give `LiveRoundEq`.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal

section Vals

open Claims.ZKMal.CoinLayer Claims.ZKMal.Masked

namespace Loop

/-- **`vs`'s value at pad `k`** of table `j`'s rep `r`, at the opened coins `c` and at table `j`'s base on the tape `τ`
(its mask slot and extra lanes), as the firewall's word; `0` from `n_E` on. -/
noncomputable def vsWord (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals) (x : Msg) (w : Flock.Wit)
    (ν : CoinLayer.Nonce) (c : Fin (refK x) → FlockDeploy.Coin) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) (k : ℕ) :
    _root_.Flock.F128 :=
  if hk : k < (tabC PE x j).z.nE then
    FlockLevel3.GF128.val (vs x w ν c j r (wtOf PE x w j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1 ⟨k, hk⟩)
  else 0

/-- **The rows hold at `vs`**: at every nonce, opened coins and tape, each table's rows hold at `vs`'s values at the
table's base (`Table.RowsAt`). -/
def RowsOK (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : CoinLayer.Nonce) (c : Fin (refK x) → FlockDeploy.Coin) (τ : UE.Tape) (j : Fin (jOf x)),
    ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, vs x w ν c j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1

/-- **The worker's free words are `vs`'s values**: after every history `LiveRoundEq` reads (`0 < ops.length < refK x`
openings, live, whose openings pass under the tape's nonce), where the frame takes the stream's coins (`answered`) and
the worker's round `v` of rep stream `2j + r`, each word the frame frees in `v` (`Frame.roundFree`: the algebraic phase's
values, `v`'s `k`-th at the absorbed order's place `m.pos + k`) is `vs`'s value at that pad index (`vsWord`), at the coins
the openings open and the tape. -/
def TakesVals (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (v : Msg) (m : _root_.Flock.Firewall.Frame.Masker)
    (out : ByteArray) (s : _root_.Flock.Firewall.Frame.Masker) (fr : Array ℕ) (j : Fin (jOf x)) (r : Fin 2),
    0 < ops.length → ops.length < refK x → L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk ((CoinLayer.session Loop.loopNr Masked.dataE).refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval
      CoinOpen.shaH).1 = true →
    wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false) →
    L.answered x (ν, t) ((m₀, ρ₀) :: ops) = .ok m →
    _root_.Flock.Firewall.Frame.roundFree (L.cfg x (ν, t) ((m₀, ρ₀) :: ops)) { m with stopped := false } ⟨v.toArray⟩
      (L.coins x (ops.length + 1)) = .ok (out, s, fr) →
    L.rep x (ops.length + 1) = 2 * j.val + r.val →
    ∀ k < fr.size,
      (_root_.Flock.Firewall.Frame.words ⟨v.toArray⟩ fr[k]! 1)[0]! =
        vsWord vs x w ν ((CoinLayer.session Loop.loopNr Masked.dataE).pad
          (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2)) t j r
          (L.cfg x (ν, t) ((m₀, ρ₀) :: ops)).order[m.pos + k]!

/-- **The worker's `final_c'` is `vs`'s `final_c` under its pad**: after every history `LiveRoundEq` reads, where the
worker's round `v` of rep stream `2j + r` is the inner proof's round 0 (its label frame, then the 16-byte `final_c'`
between its headers, the only shape `Frame.innerOwn` takes there), `final_c'` plus the rep's pad at `final_c`'s index
(the value the frame took there, `Loop.valsOf`) is `vs`'s value at that index (`vsWord`). -/
def TakesFinalC (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (v : Msg) (w' : ByteArray) (j : Fin (jOf x)) (r : Fin 2),
    0 < ops.length → ops.length < refK x → L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk ((CoinLayer.session Loop.loopNr Masked.dataE).refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval
      CoinOpen.shaH).1 = true →
    wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false) →
    w'.size = 16 →
    v = (_root_.Flock.Firewall.Frame.labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ _root_.Flock.Transcript.header 3 1 1 ++
      w' ++ _root_.Flock.Transcript.header 4 1 1).data.toList →
    L.rep x (ops.length + 1) = 2 * j.val + r.val →
    finalCOf ⟨v.toArray⟩ + (L.cfg x (ν, t) ((m₀, ρ₀) :: ops)).h.getD (L.cfg x (ν, t) ((m₀, ρ₀) :: ops)).lay.fc 0 =
      vsWord vs x w ν ((CoinLayer.session Loop.loopNr Masked.dataE).pad
        (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2)) t j r
        (L.cfg x (ν, t) ((m₀, ρ₀) :: ops)).lay.fc

/-- **The switched source's masked value below `n_E` is `vsWord` plus the session's pad**, whatever the rows. -/
theorem swSrcE_masked_vsWord (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals) (x : Msg) (w : Flock.Wit)
    (ν : CoinLayer.Nonce) (c : Fin (refK x) → FlockDeploy.Coin) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) {k : ℕ}
    (hk : k < (tabC PE x j).z.nE) :
    ((swSrcE x c w (vs x w ν c) τ j.val).rep r.val).masked k =
      vsWord vs x w ν c τ j r k + (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j)).h[k]! := by
  rw [swSrcE_masked_pad_any x c w (vs x w ν c) τ j r hk]
  unfold vsWord
  split
  · rfl
  · contradiction

end Loop

end Vals

section Inner

open _root_.Flock (V F128 Stmt SessionSpec)
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
    (hk : 0 < refK x) (hc : 0 < cs.length) : schedOf real x cs = none ↔ Loop.InnerAt real x cs := by
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

/-- **The round in a device round's place at `ofLoopLink`**: past `Hello`, the loop's own (`Loop.schedOf`) on the rep's
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

/-- **`LiveRoundEq` at the rounds the loop schedules**: its equation after every history it reads at which the loop
schedules its own round `r` (`Loop.schedOf`), against which the frame passed the worker's round. -/
def LiveSched (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
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
    (κ : (ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (v : Msg) (r : ByteArray),
    0 < ops.length → ops.length < refK x →
    (ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk ((CoinLayer.session Loop.loopNr Masked.dataE).refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval
      CoinOpen.shaH).1 = true →
    (ofLoopLink L₀ src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false) →
    Loop.schedOf (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2) = some r →
    (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) false v =
      dropDom ((CoinLayer.session Loop.loopNr Masked.dataE).swMsgAt x
        ((CoinLayer.session Loop.loopNr Masked.dataE).wt x w) (vs x w ν) t
        (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2))

/-- **`LiveRoundEq` at the inner proof's rounds** (`Loop.InnerAt`): its equation after every history it reads at which
the loop schedules no round of its own, `L₀`'s round in the device round's place. -/
def LiveInner (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
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
    (κ : (ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (v : Msg),
    0 < ops.length → ops.length < refK x →
    (ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk ((CoinLayer.session Loop.loopNr Masked.dataE).refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval
      CoinOpen.shaH).1 = true →
    (ofLoopLink L₀ src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false) →
    Loop.InnerAt (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2) →
    (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) false v =
      dropDom ((CoinLayer.session Loop.loopNr Masked.dataE).swMsgAt x
        ((CoinLayer.session Loop.loopNr Masked.dataE).wt x w) (vs x w ν) t
        (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2))

/-- **`LiveRoundEq` from the scheduled rounds and the inner proof's**: at `SessionLoopZKDeployed`'s release, where the
loop schedules its own round (`LiveSched`) and at the inner proof's rounds (`LiveInner`). -/
theorem liveRoundEq_ofLoopLink (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit)
    (hS : LiveSched vs L₀ src real wk x w) (hI : LiveInner vs L₀ src real wk x w) :
    LiveRoundEq (CoinLayer.session Loop.loopNr Masked.dataE) vs (ofLoopLink L₀ src real) CoinOpen.shaH wk x w := by
  intro ν t κ m₀ ρ₀ ops v hpos hlt hl hok href hv
  rcases hsc : Loop.schedOf (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2) with _ | r
  · exact hI ν t κ m₀ ρ₀ ops v hpos hlt hl hok href hv
      ((Loop.schedOf_eq_none_iff (by omega) (by simpa using hpos)).mp hsc)
  · exact hS ν t κ m₀ ρ₀ ops v r hpos hlt hl hok href hv hsc

end LeanRelease

end Inner

end Claims.ZKServer
