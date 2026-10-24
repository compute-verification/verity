import Proofs.Flock.Claims.ZK.FrameRounds

/-!
# The masked zerocheck's and lincheck's proofs, answered by `asksCut`

`zerocheckCut` and `lincheckCut` send the masked values `mk` at their pad indices and return them as the proof. With
room for their asks before round `n`, the run returns, and the proof it returns is `mk` at the layout's indices
(`FrameRounds`' `zerocheckCut_bodies`, `lincheckCut_bodies`, which state the sends):

* **`zerocheckCut_proof`**: `round1_ab`, `round1_c`, the `m − 6` residual rounds and the three finals.
* **`lincheckCut_proof`**: the `k_log − 6` rounds and `z_partial`.
-/

namespace Claims.ZKServer

open _root_.Flock (F128 Transcript TMW)
open FlockComplete (PS asksCut zerocheckCut lincheckCut sampleVecP sampleP_cut_ok)
open FlockComplete.Replay (pushes bind_go liftTM_ok pure_go)

/-- **The cut's zerocheck proof is its masked values**: answered by `asksCut` with room for its `3 + (m − 6)` asks,
the run returns `mk` at `round1_ab`'s, `round1_c`'s, the residual rounds' and the finals' indices. -/
theorem zerocheckCut_proof (coin : ℕ → ℕ → F128) (n : ℕ) {m : ℕ} (hm : 14 ≤ m) (digest cap root : ByteArray)
    (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128) (t : Transcript) (l : Array Claims.Msg)
    (hl : l.size + 3 + (m - 6) ≤ n) :
    ∃ zc t' L', ((zerocheckCut (asksCut coin n) m digest cap root lay mk).run t).go l =
      (.ok (((⟨(Array.range 64).map fun i => mk (lay.r1ab i), (Array.range 64).map fun i => mk (lay.r1c i),
          pushes (fun j => (mk (lay.ml j 0), mk (lay.ml j 1))) (m - 6), mk lay.fa, mk lay.fb, mk lay.fc⟩ :
          _root_.Flock.ZerocheckProof), zc), t'), L') := by
  unfold zerocheckCut
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.label_eq _ _) _)]
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbBytes_eq _ _) _)]
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbBytes_eq _ _) _)]
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.label_eq _ _) _)]
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.label_eq _ _) _)]
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbBytes_eq _ _) _)]
  rw [bind_go (sampleVecP_cut_ok coin (by decide) _ ?h1)]
  rw [bind_go (sampleVecP_cut_ok coin (show m - 13 ≠ 0 by omega) _ ?h2)]
  dsimp only
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbSlice_eq _ _) _)]
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbSlice_eq _ _) _)]
  rw [bind_go (sampleP_cut_ok coin _ ?h3)]
  case h1 => omega
  case h2 => simp only [Array.size_push]; omega
  case h3 => simp only [Array.size_push]; omega
  erw [bind_go (forIn_go_from _
      (fun i => (pushes (fun j => (mk (lay.ml j 0), mk (lay.ml j 1))) i, pushes (fun j => coin (l.size + 3 + j) 0) i))
      (pairLoopT (fun i t => mk (lay.ml i t)) coin (l.size + 3))
      (pairLoopB (fun i t => mk (lay.ml i t)) coin (l.size + 3)) (l.size + 3) (m - 6)
      (fun _ => rfl) ?hbody _ _ ?hl0)]
  case hbody =>
    intro t₀ i hi l' hl'
    dsimp only
    rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbF_eq _ _) _)]
    rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbF_eq _ _) _)]
    rw [bind_go (sampleP_cut_ok coin _ (by omega))]
    rw [hl']
    rfl
  case hl0 => simp only [Array.size_push]
  dsimp only
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbF_eq _ _) _)]
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbF_eq _ _) _)]
  rw [pure_go]
  exact ⟨_, _, _, rfl⟩

/-- **The cut's lincheck proof is its masked values**: answered by `asksCut` with room for its `3 + (k_log − 6)` asks
and the circuit's fold defined at `α`, the run returns `mk` at the rounds' and `z_partial`'s indices. -/
theorem lincheckCut_proof (coin : ℕ → ℕ → F128) (n : ℕ) (c : _root_.Flock.CircuitFold) (m : ℕ)
    (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128) (zc : _root_.Flock.ZcOut) (t : Transcript)
    (l : Array Claims.Msg) (hl : l.size + 3 + (c.kLog - 6) ≤ n)
    (hf : ∃ v, c.fold (coin l.size 0) (_root_.Flock.skipWeights zc.z) (zc.rho.extract 0 (c.kLog - 6)) = .ok v) :
    ∃ lc t' L', ((lincheckCut (asksCut coin n) c m lay mk zc).run t).go l =
      (.ok (((⟨pushes (fun j => (mk (lay.lc j 0), mk (lay.lc j 1))) (c.kLog - 6),
          (Array.range 64).map fun i => mk (lay.zp i)⟩ : _root_.Flock.LincheckProof), lc), t'), L') := by
  obtain ⟨v, hv⟩ := hf
  unfold lincheckCut
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.label_eq _ _) _)]
  rw [bind_go (sampleP_cut_ok coin _ ?h1)]
  rw [bind_go (sampleP_cut_ok coin _ ?h2)]
  case h1 => omega
  case h2 => simp only [Array.size_push]; omega
  dsimp only
  rw [hv]
  dsimp only
  rw [bind_go (pure_go _ _ _)]
  erw [bind_go (forIn_go_from _
      (fun j => (pushes (fun j => (mk (lay.lc j 0), mk (lay.lc j 1))) j, pushes (fun j => coin (l.size + 2 + j) 0) j))
      (pairLoopT (fun j t => mk (lay.lc j t)) coin (l.size + 2))
      (pairLoopB (fun j t => mk (lay.lc j t)) coin (l.size + 2)) (l.size + 2) (c.kLog - 6)
      (fun _ => rfl) ?hbody _ _ ?hl0)]
  case hbody =>
    intro t₀ j hj l' hl'
    dsimp only
    rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbF_eq _ _) _)]
    rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbF_eq _ _) _)]
    rw [bind_go (sampleP_cut_ok coin _ (by omega))]
    rw [hl']
    rfl
  case hl0 => simp only [Array.size_push]
  dsimp only
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbSlice_eq _ _) _)]
  rw [bind_go (sampleP_cut_ok coin _ ?h3)]
  case h3 =>
    simp only [Array.size_append, Array.size_push, List.size_toArray, List.length_map, List.length_range]
    omega
  rw [pure_go]
  exact ⟨_, _, _, rfl⟩

end Claims.ZKServer
