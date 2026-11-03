import Proofs.Flock.Claims.ZK.FrameOuter
import Proofs.Flock.Claims.ZK.FrameTaken

/-!
# The switched prover's messages at a live history's prefixes, at one-tape causality

`RefSession.swMsgAt_take` gives the switched prover's message after a prefix of a coin vector from the full
`SwCausal`. At `vsOf` only the one-tape form at `Loop.SwOk`'s coin vectors holds (`RefSession.SwCausalOn`), so this
file gives the message there from it:

* **`RefSession.swMsgAt_take_on`**: before the last coin, the message after a prefix is round `j`'s switched encoding
  at the coin vector itself, where `Ok` admits it and the padded prefix (`enc_swCutOf_on`, `swMsgAt_take_pre`).
* **`LeanRelease.swOk_of_live`**, **`LeanRelease.swOk_take_of_live`**: a live, unrefused history's opened coins and
  every prefix of them are `Loop.SwOk`'s, at a statement the loop serves.

`tapePairs_self`, `enc_swCutOf_on` and `swMsgAt_take_pre` belong in `Blind`, and `outerAt_pad_ge`, `swOk_of_live` and
`swOk_take_of_live` in `FrameOuter`: they are stated here, by those names, until those files state them.
-/

namespace Claims.ZKMal

open Claims
open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace RefSession

/-- **At the other leaves' pairs of the tape, `tapePairs` is every leaf's pair under the tape's salts.** -/
theorem tapePairs_self (S : RefSession) (x : Msg) (t : S.Tape) :
    S.tapePairs x t (fun l1 => (ZkView.My ((S.eP x t).2 ((leaves01 (fun j => (S.C x j).z)).symm (.inr l1))),
      ZkView.saltHash ((S.eP x t).2 ((leaves01 (fun j => (S.C x j).z)).symm (.inr l1))))) =
      fun l => (ZkView.My ((S.eP x t).2 l), ZkView.saltHash ((S.eP x t).2 l)) := by
  funext l
  rcases l with ⟨j, l | l⟩ <;> rfl

/-- **Round `i`'s switched encoding at two coin vectors `Ok` admits that agree before `i`** (`SwCausalOn` at the
executable's leaves). -/
theorem enc_swCutOf_on (S : RefSession) (vs : S.SwVals) (Ok : (x : Msg) → (Fin (refK x) → S.Coin) → Prop)
    (x : Msg) (w : Flock.Wit) (ν : S.Nonce) (hc : S.SwCausalOn Ok vs) (t : S.Tape) (c c' : Fin (refK x) → S.Coin)
    (i : ℕ) (h : ∀ k : Fin (refK x), k.val < i → c k = c' k) (hO : Ok x c) (hO' : Ok x c') :
    S.enc x c i (S.swCutOf x c (S.wt x w) (vs x w ν c) t) = S.enc x c' i (S.swCutOf x c' (S.wt x w) (vs x w ν c') t) := by
  have e := hc x w ν t (fun l1 => (ZkView.My ((S.eP x t).2 ((leaves01 (fun j => (S.C x j).z)).symm (.inr l1))),
    ZkView.saltHash ((S.eP x t).2 ((leaves01 (fun j => (S.C x j).z)).symm (.inr l1))))) c c' i h hO hO'
  rw [S.tapePairs_self x t] at e
  unfold swCutOf
  rw [swExeOf_eq, swExeOf_eq]
  convert e using 4

open Classical in
/-- **The switched prover's message after a prefix of a coin vector, before the last coin**: round `j`'s encoding at the
prefix, padded. -/
theorem swMsgAt_take_pre (S : RefSession) (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (v : (c : Fin (refK x) → S.Coin) → (∀ j, ZkView.Seen (S.C x j).z) → ∀ j, ((S.C x j).T (S.cz x c j)).tab.Vals)
    (t : S.Tape) (c : Fin (refK x) → S.Coin) (j : ℕ) (hj : j < refK x) :
    S.swMsgAt x wt v t ((List.ofFn c).take j) = S.enc x (S.pad ((List.ofFn c).take j)) j
      (S.swCutOf x (S.pad ((List.ofFn c).take j)) wt (v (S.pad ((List.ofFn c).take j))) t) := by
  have hl : ((List.ofFn c).take j).length = j := by simp only [List.length_take, List.length_ofFn]; omega
  unfold swMsgAt
  rw [hl, ite_eq_left hj]

/-- **The switched prover's message after a prefix of a coin vector, before the last coin, at one-tape causality**:
round `j`'s switched encoding at the coin vector itself, where `Ok` admits it and the padded prefix (`swMsgAt_take` at
`SwCausalOn`). -/
theorem swMsgAt_take_on (S : RefSession) (vs : S.SwVals) (Ok : (x : Msg) → (Fin (refK x) → S.Coin) → Prop)
    (x : Msg) (w : Flock.Wit) (ν : S.Nonce) (hc : S.SwCausalOn Ok vs) (t : S.Tape) (c : Fin (refK x) → S.Coin)
    (j : ℕ) (hj : j < refK x) (hO : Ok x (S.pad ((List.ofFn c).take j))) (hO' : Ok x c) :
    S.swMsgAt x (S.wt x w) (vs x w ν) t ((List.ofFn c).take j) =
      S.enc x c j (S.swCutOf x c (S.wt x w) (vs x w ν c) t) := by
  rw [S.swMsgAt_take_pre x (S.wt x w) (vs x w ν) t c j hj]
  exact S.enc_swCutOf_on vs Ok x w ν hc t _ c j (fun k hk => S.pad_take c j k hk) hO hO'

end RefSession

end Claims.ZKMal

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace LeanRelease

open Loop

/-- **Past the opened coins no outer coin is `1`**: the padding's coins are `0`. -/
theorem outerAt_pad_ge (T : ZKMal.CoinLayer.TableData) {x : Msg} (cs : List (ZKMal.CoinLayer.session loopNr T).Coin)
    {j r : ℕ} (hs : cs.length ≤ Masked.outerSlot x j r) (e : ℕ) :
    Masked.outerAt (x := x) ((ZKMal.CoinLayer.session loopNr T).pad (K := refK x) cs) j r e ≠ 1 := by
  unfold Masked.outerAt
  split
  · rename_i h
    have hd : (ZKMal.CoinLayer.session loopNr T).pad (K := refK x) cs ⟨Masked.outerSlot x j r, h⟩ = default := by
      show cs.getD (Masked.outerSlot x j r) default = default
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none hs]
      rfl
    rw [hd]
    unfold FlockDeploy.Coin.at
    split
    · exact (by decide : (0 : _root_.Flock.F128) ≠ 1)
    · decide
  · decide

/-- **A live history's opened coins are `SwOk`'s**: at a statement the loop serves, the coins of a live history of at
most `refK x` openings the release doesn't refuse have no outer coin `1`, where the frames took them
(`outer_unrefused`) and past them (`outerAt_pad_ge`). -/
theorem swOk_of_live (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (wk : (ofLoop T L₀ real).toSrv.Worker) (x : Msg) (w : Flock.Wit)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (κ : (ofLoop T L₀ real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hl : ops.length ≤ refK x)
    (hlive : (ofLoop T L₀ real).toSrv.live wk x w ρ κ ((m₀, ρ₀) :: ops) = true)
    (hr : (ofLoop T L₀ real).refusesCoins x ρ ((m₀, ρ₀) :: ops) = false) :
    SwOk T x ((ZKMal.CoinLayer.session loopNr T).pad (K := refK x)
      (ops.map fun p => (ZKMal.CoinLayer.session loopNr T).decOf p.2)) := by
  refine .inl fun hx hSv j hj r hr2 e he => ?_
  by_cases hs : Masked.outerSlot x j r < ops.length
  · exact outer_unrefused T L₀ real wk x w hx hSv ρ κ m₀ ρ₀ ops hl hlive hr hj hr2 hs he
  · exact outerAt_pad_ge T _ (by rw [List.length_map]; exact Nat.le_of_not_lt hs) e

/-- **The same at every prefix of a live history**: its prefixes are live (`live_take`) and, a refusal being for good
(`refusesCoins_of_take`), not refused. -/
theorem swOk_take_of_live (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (wk : (ofLoop T L₀ real).toSrv.Worker) (x : Msg) (w : Flock.Wit)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (κ : (ofLoop T L₀ real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hl : ops.length ≤ refK x)
    (hlive : (ofLoop T L₀ real).toSrv.live wk x w ρ κ ((m₀, ρ₀) :: ops) = true)
    (hr : (ofLoop T L₀ real).refusesCoins x ρ ((m₀, ρ₀) :: ops) = false) (k : ℕ) :
    SwOk T x ((ZKMal.CoinLayer.session loopNr T).pad (K := refK x)
      ((ops.take k).map fun p => (ZKMal.CoinLayer.session loopNr T).decOf p.2)) := by
  have ht : ((m₀, ρ₀) :: ops).take (k + 1) = (m₀, ρ₀) :: ops.take k := List.take_succ_cons
  have hlive' := (ofLoop T L₀ real).live_take wk x w ρ κ _ hlive (k + 1)
  rw [ht] at hlive'
  have hr' : (ofLoop T L₀ real).refusesCoins x ρ ((m₀, ρ₀) :: ops.take k) = false := by
    cases h : (ofLoop T L₀ real).refusesCoins x ρ ((m₀, ρ₀) :: ops.take k) with
    | false => rfl
    | true =>
      have h' := (ofLoop T L₀ real).refusesCoins_of_take x ρ ((m₀, ρ₀) :: ops) (k + 1) (by rw [ht]; exact h)
      rw [hr] at h'
      cases h'
  exact swOk_of_live T L₀ real wk x w ρ κ m₀ ρ₀ (ops.take k) (by rw [List.length_take]; omega) hlive' hr'

end LeanRelease

end Claims.ZKServer
