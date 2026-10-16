import Proofs.Flock.Claims.ZK.GKProof

/-!
# The Goldreich–Kahan simulator's cost

`Proto.gk_cost`: under `CostLe` (at most `cChk` calls per opening check, `cMsg` per dummy-witness message, `cSim` per
SHVZK simulation), the simulator with `N` rewinds makes at most `simBound` calls (hash and ask) against every verifier:
`(N + 1)·(cSim + K·cChk + k·(1 + k·cChk + cMsg))`, one dummy-witness run and `N` rewinds, each of `k` rounds of one
question to `V*`, the prover's checks of every opening so far and its message, then a classification of `K` checks.
-/

namespace Claims.ZKMal

open Claims

section Cost

variable (H : Msg → Msg)

theorem allOk_log_le (acc : ℕ → Msg → OC Bool) (B : ℕ) (hacc : ∀ j m, ((acc j m).eval H).2.length ≤ B) :
    ∀ (ms : List Msg) (j : ℕ), ((allOk acc j ms).eval H).2.length ≤ ms.length * B
  | [], _ => by simp [allOk, OC.eval]
  | m :: ms, j => by
      simp only [allOk, ocBind_eval, List.length_append, List.length_cons, Nat.succ_mul]
      have h1 := hacc j m
      split
      · have := allOk_log_le acc B hacc ms (j + 1)
        omega
      · simp only [OC.eval, List.length_nil]
        omega

theorem strat_log_le {Coin : Type} (hello abort : Msg) (dec : Msg → Coin) (acc : Msg → ℕ → Msg → OC Bool)
    (gen : List Coin → OC Msg) (B Bg : ℕ) (hacc : ∀ ρ j m, ((acc ρ j m).eval H).2.length ≤ B)
    (hgen : ∀ cs, ((gen cs).eval H).2.length ≤ Bg) :
    ∀ h : List (Msg × Msg), ((strat hello abort dec acc gen h).eval H).2.length ≤ h.length * B + Bg
  | [] => by simp [strat, OC.eval]
  | (_, ρ) :: ops => by
      simp only [strat, ocBind_eval, List.length_append, List.length_cons, Nat.succ_mul]
      have h1 := allOk_log_le H (acc ρ) B (hacc ρ) (ops.map Prod.snd) 0
      simp only [List.length_map] at h1
      split
      · have := hgen (ops.map fun p => dec p.2)
        omega
      · simp only [OC.eval, List.length_nil]
        omega

theorem spRun_cost (ask : List Msg → Out) (nxt : List (Msg × Msg) → OC Msg) (L B : ℕ)
    (hnxt : ∀ h : List (Msg × Msg), h.length < L → ((nxt h).eval H).2.length ≤ B) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)), hist.length + n ≤ L → (SP.eval H ask (spRun nxt n hist)).2 ≤ n * (1 + B)
  | 0, _, _ => by simp [spRun, SP.eval]
  | n + 1, hist, hl => by
      simp only [spRun, spBind_eval, ocSP_eval, SP.eval]
      have h1 := hnxt hist (by omega)
      cases ask (hist.map Prod.fst ++ [((nxt hist).eval H).1]) with
      | msg m =>
          simp only [spBind_eval, SP.eval]
          have := spRun_cost ask nxt L B hnxt n (hist ++ [(((nxt hist).eval H).1, m)])
            (by simp only [List.length_append, List.length_cons, List.length_nil]; omega)
          rw [Nat.succ_mul]
          omega
      | acc => simp only [SP.eval]; rw [Nat.succ_mul]; omega
      | rej => simp only [SP.eval]; rw [Nat.succ_mul]; omega

theorem clsGo_log_le (chk : Msg → ℕ → Msg → OC Bool) (B : ℕ) (hchk : ∀ ρ j m, ((chk ρ j m).eval H).2.length ≤ B)
    (τ : List (Msg × Out)) (ρ : Msg) : ∀ (n j : ℕ), ((clsGo chk τ ρ j n).eval H).2.length ≤ n * B
  | 0, _ => by simp [clsGo, OC.eval]
  | n + 1, j => by
      simp only [clsGo]
      cases resp τ (j + 1) with
      | none => simp [OC.eval]
      | some m =>
          simp only [ocBind_eval, List.length_append, Nat.succ_mul]
          have h1 := hchk ρ j m
          split
          · have := clsGo_log_le chk B hchk τ ρ n (j + 1)
            omega
          · simp only [OC.eval, List.length_nil]
            omega

theorem fclsGo_log_le {Coin : Type} [DecidableEq Coin] (chk : Msg → ℕ → Msg → OC Bool) (dec : Msg → Coin)
    (cc : ℕ → Option Coin) (B : ℕ) (hchk : ∀ ρ j m, ((chk ρ j m).eval H).2.length ≤ B) (τ : List (Msg × Out))
    (ρ : Msg) : ∀ (n j : ℕ), ((fclsGo chk dec cc τ ρ j n).eval H).2.length ≤ n * B
  | 0, _ => by simp [fclsGo, OC.eval]
  | n + 1, j => by
      simp only [fclsGo]
      cases resp τ (j + 1) with
      | none => simp [OC.eval]
      | some m =>
          simp only [ocBind_eval, List.length_append, Nat.succ_mul]
          have h1 := hchk ρ j m
          split
          · split
            · have := fclsGo_log_le chk dec cc B hchk τ ρ n (j + 1)
              omega
            · simp only [OC.eval, List.length_nil]
              omega
          · simp only [OC.eval, List.length_nil]
            omega

end Cost

namespace Proto

variable {W : Type}

theorem clsOC_log_le (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (B : ℕ)
    (hchk : ∀ ρ j m, ((G.chk x ν ρ j m).eval H).2.length ≤ B) (τ : List (Msg × Out)) :
    ((clsOC (G.K x) (G.chk x ν) G.dec τ).eval H).2.length ≤ G.K x * B := by
  unfold clsOC
  cases resp τ 0 with
  | none => simp [OC.eval]
  | some ρ =>
      simp only [ocBind_eval, OC.eval, List.length_append, List.length_nil, Nat.add_zero]
      exact clsGo_log_le H _ B hchk τ ρ _ 0

theorem fclsOC_log_le (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin) (B : ℕ)
    (hchk : ∀ ρ j m, ((G.chk x ν ρ j m).eval H).2.length ≤ B) (τ : List (Msg × Out)) :
    ((fclsOC (G.K x) (G.chk x ν) G.dec c τ).eval H).2.length ≤ G.K x * B := by
  unfold fclsOC
  cases resp τ 0 with
  | none => simp [OC.eval]
  | some ρ => exact fclsGo_log_le H _ _ _ B hchk τ ρ _ 0

/-- A rewind: the simulation, `k` rounds of the forced prover, and the classification. -/
theorem rewindSP_cost (G : Proto W) (H : Msg → Msg) (ask : List Msg → Out) (x : Msg) (ν : G.Nonce)
    (c : Fin (G.K x) → G.Coin) (s : G.SimRand x) (cChk cSim : ℕ)
    (hchk : ∀ ρ j m, ((G.chk x ν ρ j m).eval H).2.length ≤ cChk) (hsim : ((G.sim x ν c s).eval H).2.length ≤ cSim) :
    (SP.eval H ask (G.rewindSP x ν c s)).2 ≤ cSim + G.k x * (1 + G.k x * cChk) + G.K x * cChk := by
  simp only [rewindSP, spBind_eval, ocSP_eval, SP.eval, Nat.add_zero]
  have hrun := spRun_cost H ask (G.forced x ν c ((G.sim x ν c s).eval H).1) (G.k x) (G.k x * cChk)
    (fun h hl => by
      have := strat_log_le H (G.hello x ν) G.abort G.dec (facc (G.chk x ν) G.dec c)
        (fun cs => OC.ret ((((G.sim x ν c s).eval H).1).getD cs.length G.abort)) cChk 0
        (fun ρ j m => by simp only [facc, ocBind_eval, OC.eval, List.append_nil]; exact hchk ρ j m)
        (fun _ => by simp [OC.eval]) h
      have := Nat.mul_le_mul_right cChk hl.le
      unfold forced
      omega)
    (G.k x) [] (by simp)
  have hcls := G.fclsOC_log_le H x ν c cChk hchk
    (interact ask (fun h => ((G.forced x ν c ((G.sim x ν c s).eval H).1 h).eval H).1) (G.k x) [])
  rw [spRun_eval] at *
  omega

theorem rewindsSP_cost (G : Proto W) (H : Msg → Msg) (ask : List Msg → Out) (x : Msg) (ν : G.Nonce)
    (c : Fin (G.K x) → G.Coin) (cChk cSim : ℕ) (hchk : ∀ ρ j m, ((G.chk x ν ρ j m).eval H).2.length ≤ cChk)
    (hsim : ∀ s, ((G.sim x ν c s).eval H).2.length ≤ cSim) :
    ∀ (N : ℕ) (s : Fin N → G.SimRand x),
      (SP.eval H ask (G.rewindsSP x ν c N s)).2 ≤ N * (cSim + G.k x * (1 + G.k x * cChk) + G.K x * cChk)
  | 0, _ => by simp [rewindsSP, SP.eval]
  | N + 1, s => by
      simp only [rewindsSP, spBind_eval]
      have h1 := G.rewindSP_cost H ask x ν c (s 0) cChk cSim hchk (hsim _)
      have h2 := rewindsSP_cost G H ask x ν c cChk cSim hchk hsim N fun i => s i.succ
      rw [Nat.succ_mul]
      split
      · simp only [SP.eval]; omega
      · omega

/-- **The simulator's cost** (`SimCost`): at most `simBound` calls, hash and ask, against every verifier. -/
theorem gk_cost (G : Proto W) (H : Msg → Msg) (N : ℕ) (cChk cMsg cSim : Msg → ℕ)
    (hc : G.CostLe H cChk cMsg cSim) : SimCost H (G.simulator N) (G.simBound N cChk cMsg cSim) := by
  obtain ⟨hchk, hmsg, hsim⟩ := hc
  intro x c ask
  obtain ⟨ν, r, s⟩ := c
  simp only [simulator, spBind_eval, ocSP_eval]
  have hrun := spRun_cost H ask (fun hist => G.dprover.next x hist (ν, r)) (G.k x) (G.k x * cChk x + cMsg x)
    (fun h hl => by
      have := strat_log_le H (G.hello x ν) G.abort G.dec (G.chk x ν) (G.dmsg x ν r) (cChk x) (cMsg x)
        (fun ρ j m => hchk x ν ρ j m) (fun cs => hmsg x ν r cs) h
      have := Nat.mul_le_mul_right (cChk x) hl.le
      show ((strat (G.hello x ν) G.abort G.dec (G.chk x ν) (G.dmsg x ν r) h).eval H).2.length ≤ _
      omega)
    (G.k x) [] (by simp)
  set τ₀ := (SP.eval H ask (spRun (fun hist => G.dprover.next x hist (ν, r)) (G.k x) [])).1
  have hcls := G.clsOC_log_le H x ν (cChk x) (hchk x ν) τ₀
  have hrw : ∀ cs, (SP.eval H ask (G.rewindsSP x ν cs N s)).2 ≤
      N * (cSim x + G.k x * (1 + G.k x * cChk x) + G.K x * cChk x) := fun cs =>
    G.rewindsSP_cost H ask x ν cs (cChk x) (cSim x) (hchk x ν) (hsim x ν cs) N s
  have hle : G.k x * (1 + G.k x * cChk x) ≤ G.k x * (1 + (G.k x * cChk x + cMsg x)) :=
    Nat.mul_le_mul_left _ (by omega)
  have hN := Nat.mul_le_mul_left N (Nat.add_le_add_right (Nat.add_le_add_left hle (cSim x)) (G.K x * cChk x))
  unfold simBound
  rw [Nat.succ_mul]
  split
  · simp only [SP.eval]
    have : 0 ≤ N * (cSim x + G.K x * cChk x + G.k x * (1 + G.k x * cChk x + cMsg x)) := Nat.zero_le _
    simp only [Nat.add_assoc] at *
    omega
  · next cs _ =>
    have := hrw cs
    have e : cSim x + G.K x * cChk x + G.k x * (1 + G.k x * cChk x + cMsg x) =
        cSim x + G.k x * (1 + (G.k x * cChk x + cMsg x)) + G.K x * cChk x := by
      rw [Nat.add_assoc 1]; omega
    rw [e]
    simp only [Nat.add_assoc] at *
    omega

end Proto

end Claims.ZKMal
