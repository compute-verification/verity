import Proofs.Flock.Claims.ZK.Refine

/-!
# The deferred release's own histories

The deferred server meets only its own histories (`ZkServer.OwnHist`: each message is the deferred frame on the
history before it), and `LeanRelease.refD` hands each item of `ItemsRefine` that history's `hown`. The live equations
hold for any worker only there: `live` reads the worker's values at each prefix, never the history's messages, so off
`OwnHist` the released rounds a finish reads are arbitrary bytes. This file holds what the any-worker forms of the live
equations share:

* **The own histories**: `ZkServer.ownHist_nil`, `OwnHist.take` (closed under prefixes), `ownHist_concat_iff` (one
  more round is own exactly when its message is the deferred frame), `ownHist_cons_iff` (the live equations' histories
  `(m₀, ρ₀) :: ops`), and `ZkServer.ownOf` (the own history on given answers: `ownHist_ownOf`, `map_snd_ownOf`, and
  `OwnHist.eq_ownOf`, every own history is the one on its answers).
* **What an own history's messages are** at the Lean frame's release: at a frame round where the worker's run is live,
  the frame's release of the worker's round (`LeanRelease.ownHist_msg_live`); where it is not, the release of the
  stand-in, the firewall's scheduled round as a stopped rep's (`LeanRelease.ownHist_msg_fill`).
* **`LeanRelease.LiveRoundEqOwn`**: `LiveRoundEq` after the histories that are the deferred release's own, `OwnHist`
  placed as `FinalRoundEq` places it. It is weaker than `LiveRoundEq` (`LiveRoundEq.own`).
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal

namespace ZkServer

variable {W R : Type} (S : ZkServer W R) (fill : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → S.Val)
  (wk : S.Worker) (x : Msg) (w : W) (ρ : R) (κ : S.Chk)

/-- The empty history is own. -/
theorem ownHist_nil : S.OwnHist fill wk x w ρ κ [] := fun _ hj => absurd hj (Nat.not_lt_zero _)

/-- **Every prefix of an own history is own.** -/
theorem OwnHist.take {S : ZkServer W R} {fill : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → S.Val} {wk : S.Worker}
    {x : Msg} {w : W} {ρ : R} {κ : S.Chk} {h : List (Msg × Msg)} (hown : S.OwnHist fill wk x w ρ κ h) (n : ℕ) :
    S.OwnHist fill wk x w ρ κ (h.take n) := by
  sorry

/-- **One more round is own exactly when its message is the deferred frame** on the history before it. -/
theorem ownHist_concat_iff (h : List (Msg × Msg)) (p : Msg × Msg) :
    S.OwnHist fill wk x w ρ κ (h ++ [p]) ↔ S.OwnHist fill wk x w ρ κ h ∧ p.1 = S.frameD fill wk x w ρ κ h := by
  sorry

/-- **The live equations' histories**: `(m₀, ρ₀) :: ops` is own exactly when `m₀` is the deferred frame on the empty
history and each opening's message the deferred frame on the history before it. -/
theorem ownHist_cons_iff (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) :
    S.OwnHist fill wk x w ρ κ ((m₀, ρ₀) :: ops) ↔
      m₀ = S.frameD fill wk x w ρ κ [] ∧
        ∀ j (hj : j < ops.length), (ops[j]'hj).1 = S.frameD fill wk x w ρ κ ((m₀, ρ₀) :: ops.take j) := by
  sorry

/-- **The own history on the answers `as`**: each message the deferred frame on the history before it. -/
def ownOf (as : List Msg) : List (Msg × Msg) :=
  as.foldl (fun h a => h ++ [(S.frameD fill wk x w ρ κ h, a)]) []

theorem ownHist_ownOf (as : List Msg) : S.OwnHist fill wk x w ρ κ (S.ownOf fill wk x w ρ κ as) := by
  sorry

theorem map_snd_ownOf (as : List Msg) : (S.ownOf fill wk x w ρ κ as).map Prod.snd = as := by
  sorry

/-- **An own history is the own history on its answers.** -/
theorem OwnHist.eq_ownOf {S : ZkServer W R} {fill : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → S.Val}
    {wk : S.Worker} {x : Msg} {w : W} {ρ : R} {κ : S.Chk} {h : List (Msg × Msg)}
    (hown : S.OwnHist fill wk x w ρ κ h) : h = S.ownOf fill wk x w ρ κ (h.map Prod.snd) := by
  sorry

end ZkServer

namespace LeanRelease

section Msgs

variable {W R : Type} (L : LeanRelease W R) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R) (κ : L.toSrv.Chk)

/-- **An own history's message at a frame round where the worker's run is live** is the frame's release of the
worker's round there. -/
theorem ownHist_msg_live {h : List (Msg × Msg)} (hown : L.toSrv.OwnHist L.fillOf wk x w ρ κ h) {j : ℕ}
    (hj : j < h.length) (hk : L.kindAt x j = .masked) (hl : L.toSrv.live wk x w ρ κ (h.take j) = true) {v : Msg}
    (hv : wk x w ρ (h.take j) 0 = some (v, false)) : (h[j]'hj).1 = L.release x ρ (h.take j) false v := by
  sorry

/-- **An own history's message at a frame round where the worker's run is not live** is the frame's release of the
stand-in there: the firewall's scheduled round, at the values its release log reports, as a stopped rep's. -/
theorem ownHist_msg_fill {h : List (Msg × Msg)} (hown : L.toSrv.OwnHist L.fillOf wk x w ρ κ h) {j : ℕ}
    (hj : j < h.length) (hk : L.kindAt x j = .masked) (hl : L.toSrv.live wk x w ρ κ (h.take j) = false) :
    (h[j]'hj).1 =
      L.release x ρ (h.take j) true (L.scheduled x w ρ (h.take j) (computedOf (L.computed x w ρ (h.take j)))) := by
  sorry

end Msgs

/-- **The live frame rounds' bytes at the worker's own histories**: `LiveRoundEq`'s equation after every history it
reads whose messages are the deferred release's own (`ZkServer.OwnHist`, placed as `FinalRoundEq` places it), the only
histories the deferred release meets. -/
def LiveRoundEqOwn (S : RefSession) (vs : S.SwVals) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : S.Nonce) (t : S.Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (v : Msg),
    0 < ops.length → ops.length < refK x → L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = true →
    L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    L.toSrv.OwnHist L.fillOf wk x w (ν, t) κ ((m₀, ρ₀) :: ops) →
    wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false) →
    L.release x (ν, t) ((m₀, ρ₀) :: ops) false v =
      dropDom (S.swMsgAt x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2))

/-- `LiveRoundEq` gives its equation at the own histories. -/
theorem LiveRoundEq.own {S : RefSession} {vs : S.SwVals} {L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)}
    {H : Msg → Msg} {wk : L.toSrv.Worker} {x : Msg} {w : Flock.Wit} (hE : LiveRoundEq S vs L H wk x w) :
    LiveRoundEqOwn S vs L H wk x w :=
  fun ν t κ m₀ ρ₀ ops v h0 hlt hl hok hr _ hv => hE ν t κ m₀ ρ₀ ops v h0 hlt hl hok hr hv

end LeanRelease

end Claims.ZKServer
