module

public import Definitions.Claims.ZKOut
public import Proofs.Flock.Claims.ZK.Server

/-!
# The deferred server, zero knowledge per outcome

`ZkServer.zk_deferred` bounds the server that defers every stop to its last round by twice the simulator that guesses
whether it stopped (`leaky1`): the stop is one bit a session. Told that bit, the simulator needs no guess, so per outcome
the bound has no factor `2`: the deferred run is the unswitched prover's cut at its own stop bit (`split_deferred`,
`cutT_stopBit`), and the unswitched prover is zero knowledge (`ZKmalR`).

* `stopBit`: the outcome, read off the transcript: `1` where its message at the last round `k x - 1` is the stop.
* `leakyOut`: `leaky1` told its bit, a simulator given the outcome.
* `cutT_stopBit`: a transcript cut at `k x - c` is the cut at its own stop bit, for any transcript and `c : Fin 2`.
* **`ZkServer.zkOut_deferred`**: where `P` meets `ZKmalR` and the deferred server's frame is `P`'s message wherever it
  releases one (`RefDOwn`), the deferred server is `ZKOut` at the stop bit, by `leakyOut`, at `P`'s error
  `ε_zk + γ·Fork`.
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal

/-- **The stop bit**: `1` where the transcript's message at the last round `k x - 1` is the stop, else `0`. -/
def stopBit (stop : Msg) (k : Msg → ℕ) (x : Msg) (τ : List (Msg × Out)) : Fin 2 :=
  if (τ[k x - 1]?).map Prod.fst = some stop then 1 else 0

/-- **`leaky1` told its bit**: given the outcome `o`, `Sim`'s transcript replayed against the verifier, its last message
replaced by the stop where `o = 1`. -/
def leakyOut (Sim : Simulator) (k : Msg → ℕ) (stop : Msg) : SimOut (Fin 2) where
  C := Sim.C
  prog x o c := (leaky1 Sim k stop).prog x (o, c)

theorem leakyOut_out {Θ : Type} (H : Msg → Msg) (Sim : Simulator) (k : Msg → ℕ) (stop : Msg) (Vs : VStar Θ) (θ : Θ)
    (x : Msg) (o : Fin 2) (c : Sim.C x) :
    ((leakyOut Sim k stop).given o).out H Vs θ x c =
      cutT (fun as => ((Vs.next x as θ).eval H).1) stop (k x - (o : ℕ)) (k x) (Sim.out H Vs θ x c) :=
  leaky1_out H Sim k stop Vs θ x (o, c)

/-- Where the run cut at the last round shows the stop there, the run uncut is that cut. -/
theorem cutT_last_of_stop (ask : List Msg → Out) (stop : Msg) (N : ℕ) (τ : List (Msg × Out))
    (h : ((cutT ask stop N N τ)[N - 1]?).map Prod.fst = some stop) :
    cutT ask stop (N - 1) N τ = cutT ask stop N N τ := by
  unfold cutT at h ⊢
  refine interact_congr ask (cutMsg stop N τ) (cutMsg stop (N - 1) τ) N [] fun i hi => ?_
  simp only [List.nil_append]
  have hN := interact_length_le ask (cutMsg stop N τ) N []
  have hl : (histOf ((interact ask (cutMsg stop N τ) N []).take i)).length = i := by
    rw [length_histOf_take]; omega
  by_cases hi1 : i < N - 1
  · rw [cutMsg_of_lt (t := N - 1) (by rw [hl]; exact hi1), cutMsg_of_lt (t := N) (by rw [hl]; omega)]
  · have hiN : i = N - 1 := by omega
    subst hiN
    rw [cutMsg_of_ge (t := N - 1) (by rw [hl]; exact hi1)]
    have hg := interact_getElem_fst ask (cutMsg stop N τ) N [] (N - 1) hi
    rw [List.nil_append, h] at hg
    exact Option.some.inj hg

/-- Where the run cut before the last round shows no stop there, it is the run uncut. -/
theorem cutT_last_of_ne (ask : List Msg → Out) (stop : Msg) (N : ℕ) (τ : List (Msg × Out))
    (h : ¬ ((cutT ask stop (N - 1) N τ)[N - 1]?).map Prod.fst = some stop) :
    cutT ask stop N N τ = cutT ask stop (N - 1) N τ := by
  unfold cutT at h ⊢
  refine interact_congr ask (cutMsg stop (N - 1) τ) (cutMsg stop N τ) N [] fun i hi => ?_
  simp only [List.nil_append]
  have hN := interact_length_le ask (cutMsg stop (N - 1) τ) N []
  have hl : (histOf ((interact ask (cutMsg stop (N - 1) τ) N []).take i)).length = i := by
    rw [length_histOf_take]; omega
  by_cases hi1 : i < N - 1
  · rw [cutMsg_of_lt (t := N) (by rw [hl]; omega), cutMsg_of_lt (t := N - 1) (by rw [hl]; exact hi1)]
  · exfalso
    have hiN : i = N - 1 := by omega
    subst hiN
    have hg := interact_getElem_fst ask (cutMsg stop (N - 1) τ) N [] (N - 1) hi
    rw [List.nil_append, cutMsg_of_ge (t := N - 1) (by rw [hl]; exact hi1)] at hg
    exact h hg

/-- **A cut transcript is the cut at its own stop bit**: for any transcript `τ` and `c : Fin 2`, `τ` cut at `k x - c`
is `τ` cut at `k x` less the stop bit of that cut. -/
theorem cutT_stopBit (ask : List Msg → Out) (stop : Msg) (k : Msg → ℕ) (x : Msg) (τ : List (Msg × Out))
    (c : Fin 2) :
    cutT ask stop (k x - (c : ℕ)) (k x) τ =
      cutT ask stop (k x - (stopBit stop k x (cutT ask stop (k x - (c : ℕ)) (k x) τ) : ℕ)) (k x) τ := by
  rcases (show c = 0 ∨ c = 1 by fin_cases c <;> simp) with rfl | rfl
  · by_cases h : ((cutT ask stop (k x) (k x) τ)[k x - 1]?).map Prod.fst = some stop
    · have hb : stopBit stop k x (cutT ask stop (k x - ((0 : Fin 2) : ℕ)) (k x) τ) = 1 := by
        rw [Fin.val_zero, Nat.sub_zero]
        exact ite_eq_left h
      rw [hb, Fin.val_zero, Fin.val_one, Nat.sub_zero]
      exact (cutT_last_of_stop ask stop (k x) τ h).symm
    · have hb : stopBit stop k x (cutT ask stop (k x - ((0 : Fin 2) : ℕ)) (k x) τ) = 0 := by
        rw [Fin.val_zero, Nat.sub_zero]
        exact ite_eq_right h
      rw [hb]
  · by_cases h : ((cutT ask stop (k x - 1) (k x) τ)[k x - 1]?).map Prod.fst = some stop
    · have hb : stopBit stop k x (cutT ask stop (k x - ((1 : Fin 2) : ℕ)) (k x) τ) = 1 := by
        rw [Fin.val_one]
        exact ite_eq_left h
      rw [hb]
    · have hb : stopBit stop k x (cutT ask stop (k x - ((1 : Fin 2) : ℕ)) (k x) τ) = 0 := by
        rw [Fin.val_one]
        exact ite_eq_right h
      rw [hb, Fin.val_zero, Fin.val_one, Nat.sub_zero]
      exact (cutT_last_of_ne ask stop (k x) τ h).symm

namespace ZkServer

/-- **The deferred run is the unswitched run cut at its own stop bit**: where the deferred server's frame is `P`'s
message wherever it releases one (`RefDOwn`), its run against `ask` is `P`'s cut at `n x` less the stop bit of its run. -/
theorem split_deferred_stopBit {W R : Type} (S : ZkServer W R) (H : Msg → Msg) (P : W → Prover R)
    (fill : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → S.Val) (n : Msg → ℕ) (x : Msg) (w : W) (wk : S.Worker)
    (hRef : S.RefDOwn H P fill n wk x w) (ask : List Msg → Out) (ρ : R) (κ : S.Chk) :
    interact ask (S.nextD fill n wk x w ρ κ) (n x) [] =
      cutT ask S.stop (n x - (stopBit S.stop n x (interact ask (S.nextD fill n wk x w ρ κ) (n x) []) : ℕ)) (n x)
        (interact ask (fun h => (((P w).next x h ρ).eval H).1) (n x) []) := by
  obtain ⟨c, hc⟩ := S.split_deferred H P fill n x w wk hRef ask ρ κ
  rw [hc]
  exact cutT_stopBit ask S.stop n x _ c

/-- **Zero knowledge of the deferred server, per outcome**: where `P` meets `ZKmalR` (with simulator `Sim`) and the
deferred server's frame is `P`'s message wherever it releases one (`RefDOwn`), the deferred server with the worker
`wk` is zero knowledge with its stop bit public (`ZKOut`): per stop bit `o`, an event of its transcript and the stop bit
`o` is at most the event at `Sim`'s transcript cut at `o` (`leakyOut`), plus `P`'s `ε_zk + γ·Fork`. -/
theorem zkOut_deferred {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (S : ZkServer W (N × Ω))
    (fill : Msg → ℕ → ℕ → W → N × Ω → List (Msg × Msg) → S.Val) (H : Msg → Msg) (k : Msg → ℕ)
    (P : W → Prover (N × Ω)) (Rl : Rel W) (Sim : Simulator) (εzk γ : Msg → ℝ) (hZK : ZKmalR H k P Rl Sim εzk γ)
    (wk : S.Worker) :
    ZKOut (fun _ => N × Ω) (fun _ => S.Chk) H k (fun w _ => S.proverD fill k wk w)
      (fun x w => Rl x w ∧ S.RefDOwn H P fill k wk x w) (stopBit S.stop k) (leakyOut Sim k S.stop)
      (fun x w _ Vs θ _ => εzk x + γ x * (Fork H (P w) Vs θ x).Pr H) := by
  intro x w hR Θ Vs θ hk o Ev
  set ask : List Msg → Out := fun as => ((Vs.next x as θ).eval H).1
  have hP : ∀ ρ : N × Ω, (runStar H (P w) Vs x ρ θ).msgs =
      interact ask (fun h => (((P w).next x h ρ).eval H).1) (k x) [] := fun ρ => by
    rw [runStar_msgs, hk]
  have hcut : ∀ ω : (N × Ω) × S.Chk, (runStar H (S.proverD fill k wk w) Vs x ω θ).msgs =
      cutT ask S.stop (k x - (stopBit S.stop k x (runStar H (S.proverD fill k wk w) Vs x ω θ).msgs : ℕ)) (k x)
        (runStar H (P w) Vs x ω.1 θ).msgs := fun ω => by
    have hsrv : (runStar H (S.proverD fill k wk w) Vs x ω θ).msgs =
        interact ask (S.nextD fill k wk x w ω.1 ω.2) (k x) [] := by
      rw [runStar_msgs, hk]; rfl
    rw [hsrv, hP]
    exact S.split_deferred_stopBit H P fill k x w wk hR.2 ask ω.1 ω.2
  have hZ := hZK x w hR.1 Θ Vs θ hk (fun τ => Ev (cutT ask S.stop (k x - (o : ℕ)) (k x) τ))
  calc Pr (fun ω : (N × Ω) × S.Chk => stopBit S.stop k x (runStar H (S.proverD fill k wk w) Vs x ω θ).msgs = o ∧
          Ev (runStar H (S.proverD fill k wk w) Vs x ω θ).msgs)
        ≤ Pr (fun ω : (N × Ω) × S.Chk =>
          Ev (cutT ask S.stop (k x - (o : ℕ)) (k x) (runStar H (P w) Vs x ω.1 θ).msgs)) := by
        refine Pr_mono fun ω hω => ?_
        have e := hcut ω
        rw [hω.1] at e
        rw [← e]
        exact hω.2
    _ = Pr (fun ρ : N × Ω => Ev (cutT ask S.stop (k x - (o : ℕ)) (k x) (runStar H (P w) Vs x ρ θ).msgs)) :=
        Pr_fst (fun ρ : N × Ω => Ev (cutT ask S.stop (k x - (o : ℕ)) (k x) (runStar H (P w) Vs x ρ θ).msgs))
    _ ≤ Pr (fun c : Sim.C x => Ev (cutT ask S.stop (k x - (o : ℕ)) (k x) (Sim.out H Vs θ x c))) +
          (εzk x + γ x * (Fork H (P w) Vs θ x).Pr H) := by
        linarith [le_abs_self (Pr (fun ρ : N × Ω => Ev (cutT ask S.stop (k x - (o : ℕ)) (k x)
          (runStar H (P w) Vs x ρ θ).msgs)) -
            Pr (fun c : Sim.C x => Ev (cutT ask S.stop (k x - (o : ℕ)) (k x) (Sim.out H Vs θ x c))))]
    _ = Pr (fun c : (leakyOut Sim k S.stop).C x => Ev (((leakyOut Sim k S.stop).given o).out H Vs θ x c)) +
          (εzk x + γ x * (Fork H (P w) Vs θ x).Pr H) := by
        congr 1
        exact Pr_congr fun c => Iff.of_eq (congrArg Ev (leakyOut_out H Sim k S.stop Vs θ x o c).symm)

end ZkServer

end Claims.ZKServer
