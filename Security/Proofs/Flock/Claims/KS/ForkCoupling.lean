import Proofs.Flock.Claims.KS.ForkSlack
import Proofs.Flock.Claims.KS.FnFork
import Proofs.Flock.Claims.KS.LiveMix

/-!
# The fork's coin coupling at a simulator of uniform functions

`ForkSlack` asks for the coupling of the fork's coin law (`ForkCoinCoupling`) at a simulator's coins; `FnFork` proves
it at `fnAt`, a tape of uniform functions of the live coins' history read through a prefix key
(`FnFork.fnAt_coupling`). Here the two meet, in `ForkSlack`'s closure:

* `forkCoinCoupling_fnAt`: at any verifier whose coins are `M` symbols of a finite `X`, the simulated forks
  `simForkPr` at `fnAt` and the live fork `forkN` satisfy `ForkCoinCoupling` at the chances `FnFork.δN X M n`
  (`δ_t = |X|^{-min (n t) M}`, `FnFork.δAt_eq`), the self-pair's event being the one-run event
  (`tiedColl_of_append_self`).
* `slack_fnAt`: so the simulated fork at round `j` is at most the deployed fork there and the slack at those chances
  (`slack_of_coupling`).
* `forkCoinCoupling_plain`: the coupling at the deployed plain verifier `V false`, at `Claims.Bridge.Live.fnAt` with
  the live coins `LCoin 65536` (`live_fnAt_eq`).
-/

namespace Claims.Flock

open FnFork

/-- **The coupling at `fnAt`**: with the verifier's coins `M` symbols of `X`, the live coins indexed by `Fin L`,
`L = K`, and a prefix key `key` of the live coins' history (`PrefixKey`), the simulated forks at `fnAt` and the live
fork `forkN` couple with the deployed fork at the chances `δN X M n`. -/
theorem forkCoinCoupling_fnAt {M : ℕ} {«X» D : Type} [Fintype «X»] [DecidableEq «X»] [Nonempty «X»]
    [hI : Inhabited (Fin M → «X»)] [Fintype D] [DecidableEq D] {Ω : Type} (H : Msg → Msg)
    (V : Verifier (Fin M → «X»)) (P : Prover Ω) (x : Msg) (ω : Ω) {L : ℕ} (h : L = V.k x) (n : Fin (V.k x) → ℕ)
    {key : Fin (V.k x) → (Fin (V.k x) → Fin M → «X») → D} (hk : PrefixKey n key) :
    ForkCoinCoupling H V P x (δN «X» M n)
      (simForkPr H V P x (fun q => fnAt h n key q.1 q.2) (fun t => forkN t) ω) ω := by
  obtain ⟨hstep, hlast⟩ := fnAt_coupling h hk (pairEv H V P x ω) (oneEv H V P x ω)
    fun s hs => ⟨hs.1, FlockDeploy.KS.tiedColl_of_append_self hs.2.2⟩
  refine ⟨fun t ht => ?_, ?_⟩
  · unfold forkPrAt
    rw [dite_eq_left_of_eq_true (eq_true ht)]
    exact hstep t ht
  · unfold forkPrAt
    rw [dite_eq_right_of_eq_false (eq_false (lt_irrefl _))]
    exact hlast

/-- **The slack at `fnAt`**: the simulated fork at `fnAt` at round `j` is at most the deployed fork there and the
slack at the chances `δN X M n` (`slack_of_coupling` at `forkCoinCoupling_fnAt`). -/
theorem slack_fnAt {M : ℕ} {«X» D : Type} [Fintype «X»] [DecidableEq «X»] [Nonempty «X»]
    [Inhabited (Fin M → «X»)] [Fintype D] [DecidableEq D] {Ω : Type} (H : Msg → Msg) (V : Verifier (Fin M → «X»))
    (P : Prover Ω) (x : Msg) (ω : Ω) {L : ℕ} (h : L = V.k x) (n : Fin (V.k x) → ℕ)
    {key : Fin (V.k x) → (Fin (V.k x) → Fin M → «X») → D}
    (hk : PrefixKey n key) (j : Fin (V.k x)) :
    simForkPr H V P x (fun q => fnAt h n key q.1 q.2) (fun t => forkN t) ω j ≤
      (Pr fun s : (Fin (V.k x) → Fin M → «X») × (Fin (V.k x) → Fin M → «X») => forkEv H V P x j ω s.1 s.2) +
        forkSlack H V P x (δN «X» M n) j ω :=
  slack_of_coupling H V P x (fun t => δN_nonneg («X» := «X») (M := M) n t) (forkCoinCoupling_fnAt H V P x ω h n hk) j

end Claims.Flock

namespace FlockDeploy.KS

/-- **`Claims.Bridge.Live.fnAt` is `FnFork.fnAt`**: the two are one definition. -/
theorem live_fnAt_eq {K L M : ℕ} {«X» D : Type} (h : L = K) (n : Fin K → ℕ)
    (key : Fin K → (Fin K → Fin M → «X») → D) (f : Fin K → D → Fin M → «X») (r : Fin L → Fin M → «X») :
    Claims.Bridge.Live.fnAt h n key f r = Claims.Flock.FnFork.fnAt h n key f r := rfl

/-- **The coupling at the deployed plain verifier**: at `V false`, with a tape of uniform functions of the live coins
`LCoin 65536` read through a prefix key (`Claims.Bridge.Live.fnAt`), `ForkCoinCoupling` holds at the chances
`FnFork.δN GF128 65536 n`, `δ_t = 2^{-128·min (n t) 65536}` (`FnFork.δAt_eq`). -/
theorem forkCoinCoupling_plain {D : Type} [Fintype D] [DecidableEq D] {Ω : Type} (P : Claims.Prover Ω)
    (x : Claims.Msg) (ω : Ω) {L : ℕ} (h : L = (V false).k x) (n : Fin ((V false).k x) → ℕ)
    {key : Fin ((V false).k x) → (Fin ((V false).k x) → Claims.Bridge.Live.LCoin 65536) → D}
    (hk : Claims.Flock.FnFork.PrefixKey n key) :
    Claims.Flock.ForkCoinCoupling sha512 (V false) P x (Claims.Flock.FnFork.δN FlockLevel3.GF128 65536 n)
      (Claims.Flock.simForkPr sha512 (V false) P x
        (fun q : (Fin ((V false).k x) → D → Claims.Bridge.Live.LCoin 65536) ×
            (Fin L → Claims.Bridge.Live.LCoin 65536) => Claims.Bridge.Live.fnAt h n key q.1 q.2)
        (fun t => Claims.Flock.FnFork.forkN t) ω) ω :=
  Claims.Flock.forkCoinCoupling_fnAt («X» := FlockLevel3.GF128) (M := 65536) (hI := instInhabitedCoin) sha512 (V false)
    P x ω h n hk

end FlockDeploy.KS
