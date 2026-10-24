import Proofs.Flock.SessionLoop.FinishParts
import Proofs.Flock.Claims.ZK.FrameTaken

/-!
# (d) `FinRepRounds`: the reps' rounds

* **`LeanRelease.msg_of_liveRoundEq`**: on a live history of `refK x` openings whose openings pass and whose messages are
  the deferred release's own (`ZkServer.OwnHist`), each frame round's message (rounds `2` to `refK x`) is the session's
  switched message at the coins the openings before it open, without its leading domain frame: `LiveRoundEq` at that
  prefix, which is live (`live_take`), takes its coins (`refusesCoins_of_live`) and passes its openings (`allOk_eval`).
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal

namespace LeanRelease

/-- **Each frame round of a `Finish` `FinishIs` reads is the session's switched message** at the coins the openings
before it open, without its leading domain frame. -/
theorem msg_of_liveRoundEq (S : RefSession) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg)
    (w : Flock.Wit) (hE : LiveRoundEq S vs L H wk x w) (hlast : L.last x = refK x + 1) (ν : S.Nonce) (t : S.Tape)
    (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hlen : ops.length = refK x)
    (hl : L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true)
    (hok : ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = true)
    (hown : L.toSrv.OwnHist L.fillOf wk x w (ν, t) κ ((m₀, ρ₀) :: ops)) (j : ℕ) (h2 : 2 ≤ j) (hj : j ≤ refK x) :
    (((m₀, ρ₀) :: ops)[j]'(by simp only [List.length_cons]; omega)).1 =
      dropDom (S.swMsgAt x (S.wt x w) (vs x w ν) t ((ops.take (j - 1)).map fun p => S.decOf p.2)) := by
  have hj' : j < ((m₀, ρ₀) :: ops).length := by simp only [List.length_cons]; omega
  have htake : ((m₀, ρ₀) :: ops).take j = (m₀, ρ₀) :: ops.take (j - 1) := by
    obtain ⟨i, rfl⟩ : ∃ i, j = i + 1 := ⟨j - 1, by omega⟩
    simp
  have hlj : (((m₀, ρ₀) :: ops).take j).length = j := by
    simp only [List.length_take, List.length_cons]
    omega
  have hk : L.kindAt x (((m₀, ρ₀) :: ops).take j).length = .masked := by
    rw [hlj]
    unfold kindAt
    split_ifs with h0 h1
    · omega
    · rw [hlast] at h1
      omega
    · rfl
  have hl' := L.live_take wk x w (ν, t) κ _ hl j
  obtain ⟨v, hv, -⟩ := L.live_some wk x w (ν, t) κ _ hk hl'
  have hu : L.toSrv.wkD L.fillOf wk κ x w (ν, t) (((m₀, ρ₀) :: ops).take j) 0 = some (v, false) := by
    simp only [ZkServer.wkD, hl', ↓reduceIte, hv]
  have hr := L.refusesCoins_of_live wk x w (ν, t) κ _ hk hl'
  refine (hown j hj').trans ?_
  rw [L.frameD_toSrv L.fillOf wk x w (ν, t) κ _ hk hu]
  rw [htake] at hl' hr hv
  rw [htake]
  have hlen' : (ops.take (j - 1)).length = j - 1 := by
    simp only [List.length_take]
    omega
  refine hE ν t κ m₀ ρ₀ (ops.take (j - 1)) v (by rw [hlen']; omega) (by rw [hlen']; omega) hl' ?_ hr hv
  rw [allOk_eval] at hok ⊢
  intro i hi
  simp only [List.length_map] at hi
  have := hok i (by simp only [List.length_map]; omega)
  simpa [List.getElem_map, List.getElem_take] using this

end LeanRelease

end Claims.ZKServer
