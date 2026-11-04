module

public import Definitions.Claims.ZKOut
public import Proofs.Flock.ZkAny.Unswitched

/-!
# The deferred release per outcome: the `Finish` verdict

The server that defers every stop (`ZkServer.proverD`) releases, at its last round `n x - 1`, its frame or the stop.
The verdict read off the release (`stoppedAt`: its message at round `n x - 1` is the stop) is its public outcome, and
on each outcome the release is one prover's run:

* `release_stopped`: where the release shows the stop at its last round, it is the stop prover's run cut there, for any
  prover whose messages are the deferred server's frames below its last round (`RefDBelow`).
* `release_go`: where it doesn't, it is the go prover's run, for any prover whose messages are the deferred server's
  frames wherever it releases one (`RefDOwn`).
* `zk_deferred_out`: so the deferred release is zero knowledge per outcome (`ZKOut`) at the verdict, from the stop
  prover's zero knowledge (`ZKmalR`) read through the cut (`cutSim`), and the go prover's at its own verdict: no guess
  of the outcome, each outcome's simulator its own (`verdictSim`), fixed before the witness.
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal

/-- **The `Finish` verdict read off a transcript**: whether its message at round `t` is the stop. -/
def stoppedAt (stop : Msg) (t : ℕ) (τ : List (Msg × Out)) : Bool := decide ((τ[t]?).map Prod.fst = some stop)

/-- **The simulator cut at round `t`**: `Sim`'s transcript, its messages from round `t` on the stop, replayed against
the verifier. -/
def cutSim (Sim : Simulator) (t k : Msg → ℕ) (stop : Msg) : Simulator where
  C := Sim.C
  prog x c := spBind (Sim.prog x c) fun τ => spRun (fun h => .ret (cutMsg stop (t x) τ h)) (k x) []

theorem cutSim_out {Θ : Type} (H : Msg → Msg) (Sim : Simulator) (t k : Msg → ℕ) (stop : Msg) (Vs : VStar Θ) (θ : Θ)
    (x : Msg) (c : (cutSim Sim t k stop).C x) :
    (cutSim Sim t k stop).out H Vs θ x c =
      cutT (fun as => ((Vs.next x as θ).eval H).1) stop (t x) (k x) (Sim.out H Vs θ x c) := by
  unfold Simulator.out cutSim cutT
  rw [spBind_eval, spRun_eval]
  rfl

/-- **The deferred release's simulator given its verdict**: on the stop, the stop prover's simulator cut at the last
round `n x - 1`; otherwise the go prover's. Its coins are both simulators'. -/
def verdictSim (Sims Simg : Simulator) (n : Msg → ℕ) (stop : Msg) : SimOut Bool where
  C x := Sims.C x × Simg.C x
  prog x o c := if o then (cutSim Sims (fun x => n x - 1) n stop).prog x c.1 else Simg.prog x c.2

private theorem pr_snd {α β : Type} [Fintype α] [Fintype β] [Nonempty α] (p : β → Prop) :
    Pr (fun s : α × β => p s.2) = Pr p := by
  classical
  rw [Pr_eq_sum (fun s : α × β => p s.2), Pr_eq_sum p, Fintype.sum_prod_type, Fintype.card_prod]
  have : (0 : ℝ) < Fintype.card α := by exact_mod_cast Fintype.card_pos
  dsimp only
  rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, Nat.cast_mul]
  exact mul_div_mul_left _ _ this.ne'

namespace ZkServer

/-- **The deferred release's prefixes below its last round are its own runs** (`OwnHist`): below the last round it
releases its frame. -/
theorem ownHist_release {W R : Type} (S : ZkServer W R) (fill : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → S.Val)
    (n : Msg → ℕ) (wk : S.Worker) (x : Msg) (w : W) (ask : List Msg → Out) (ρ : R) (κ : S.Chk) (i : ℕ)
    (hi : i + 1 ≤ n x) :
    S.OwnHist fill wk x w ρ κ (histOf ((interact ask (S.nextD fill n wk x w ρ κ) (n x) []).take i)) := by
  set τ := interact ask (S.nextD fill n wk x w ρ κ) (n x) [] with hτ
  intro j hj
  have hjl : j < τ.length := by rw [length_histOf_take] at hj; omega
  have hji : j < i := by rw [length_histOf_take] at hj; omega
  have hg := interact_getElem_fst ask (S.nextD fill n wk x w ρ κ) (n x) [] j hjl
  rw [← hτ, List.nil_append, List.getElem?_eq_getElem hjl] at hg
  have hg' : (τ[j]'hjl).1 = S.nextD fill n wk x w ρ κ (histOf (τ.take j)) := Option.some.inj hg
  have hm : ((histOf (τ.take i))[j]'hj).1 = (τ[j]'hjl).1 := by simp [histOf]
  rw [hm, histOf_take_take τ hji.le, hg']
  unfold nextD
  rw [ite_eq_left (Or.inl (by rw [length_histOf_take]; omega))]

/-- **Where the release shows the stop at its last round, it is the stop prover's run cut there**: for every worker,
check coins and tape, if `P`'s messages are the deferred server's frames below its last round (`RefDBelow`), a release
whose message at round `n x - 1` is the stop is `P`'s run with that message the stop. -/
theorem release_stopped {W R : Type} (S : ZkServer W R) (H : Msg → Msg) (P : W → Prover R)
    (fill : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → S.Val) (n : Msg → ℕ) (x : Msg) (w : W) (wk : S.Worker)
    (hRef : S.RefDBelow H P fill n wk x w) (ask : List Msg → Out) (ρ : R) (κ : S.Chk)
    (hs : stoppedAt S.stop (n x - 1) (interact ask (S.nextD fill n wk x w ρ κ) (n x) []) = true) :
    interact ask (S.nextD fill n wk x w ρ κ) (n x) [] =
      cutT ask S.stop (n x - 1) (n x) (interact ask (fun h => (((P w).next x h ρ).eval H).1) (n x) []) := by
  set pm : List (Msg × Msg) → Msg := fun h => (((P w).next x h ρ).eval H).1 with hpm
  set τ := interact ask (S.nextD fill n wk x w ρ κ) (n x) [] with hτ
  rw [← interact_stopAt]
  symm
  refine interact_congr ask (S.nextD fill n wk x w ρ κ) (stopAt pm (n x - 1) S.stop) (n x) [] fun i hi => ?_
  simp only [List.nil_append]
  rw [← hτ] at hi ⊢
  have hτn : τ.length ≤ n x := interact_length_le ask _ (n x) []
  have hgl : (histOf (τ.take i)).length = i := by rw [length_histOf_take]; omega
  by_cases hlast : i + 1 < n x
  · have hown := S.ownHist_release fill n wk x w ask ρ κ i (by omega)
    rw [← hτ] at hown
    rw [stopAt_of_lt (by rw [hgl]; omega)]
    unfold nextD
    rw [ite_eq_left (Or.inl (by rw [hgl]; exact hlast))]
    exact hRef ρ κ _ hown (by rw [hgl]; exact hlast)
  · have hiN : i = n x - 1 := by omega
    rw [stopAt_of_ge (by rw [hgl]; omega)]
    have hg := interact_getElem_fst ask (S.nextD fill n wk x w ρ κ) (n x) [] i (by rw [← hτ]; exact hi)
    rw [← hτ, List.nil_append] at hg
    have hs' : (τ[n x - 1]?).map Prod.fst = some S.stop := by simpa [stoppedAt] using hs
    rw [← hiN, hg] at hs'
    exact (Option.some.inj hs').symm

/-- **Where the release doesn't show the stop at its last round, it is the go prover's run**: for every worker, check
coins and tape, if `P`'s messages are the deferred server's frames wherever it releases one (`RefDOwn`), a release
whose message at round `n x - 1` is not the stop (or that ended before it) is `P`'s run. -/
theorem release_go {W R : Type} (S : ZkServer W R) (H : Msg → Msg) (P : W → Prover R)
    (fill : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → S.Val) (n : Msg → ℕ) (x : Msg) (w : W) (wk : S.Worker)
    (hRef : S.RefDOwn H P fill n wk x w) (ask : List Msg → Out) (ρ : R) (κ : S.Chk)
    (hg : stoppedAt S.stop (n x - 1) (interact ask (S.nextD fill n wk x w ρ κ) (n x) []) = false) :
    interact ask (S.nextD fill n wk x w ρ κ) (n x) [] =
      interact ask (fun h => (((P w).next x h ρ).eval H).1) (n x) [] := by
  set τ := interact ask (S.nextD fill n wk x w ρ κ) (n x) [] with hτ
  symm
  refine interact_congr ask (S.nextD fill n wk x w ρ κ) (fun h => (((P w).next x h ρ).eval H).1) (n x) []
    fun i hi => ?_
  simp only [List.nil_append]
  rw [← hτ] at hi ⊢
  have hτn : τ.length ≤ n x := interact_length_le ask _ (n x) []
  have hgl : (histOf (τ.take i)).length = i := by rw [length_histOf_take]; omega
  have hown := S.ownHist_release fill n wk x w ask ρ κ i (by omega)
  rw [← hτ] at hown
  have hcond : (histOf (τ.take i)).length + 1 < n x ∨ S.live wk x w ρ κ (histOf (τ.take i)) = true := by
    by_cases hlast : i + 1 < n x
    · exact Or.inl (by rw [hgl]; exact hlast)
    · right
      have hiN : i = n x - 1 := by omega
      by_contra hne
      have hnc : ¬ ((histOf (τ.take i)).length + 1 < n x ∨ S.live wk x w ρ κ (histOf (τ.take i)) = true) := by
        rintro (h1 | h1)
        · rw [hgl] at h1; omega
        · exact hne h1
      have hstop : S.nextD fill n wk x w ρ κ (histOf (τ.take i)) = S.stop := by
        unfold nextD
        rw [ite_eq_right hnc]
      have hgi := interact_getElem_fst ask (S.nextD fill n wk x w ρ κ) (n x) [] i (by rw [← hτ]; exact hi)
      rw [← hτ, List.nil_append, hstop] at hgi
      rw [← hiN] at hg
      simp [stoppedAt, hgi] at hg
  unfold nextD
  rw [ite_eq_left hcond]
  exact hRef ρ κ _ hown hcond

/-- **The deferred release is zero knowledge per outcome at its `Finish` verdict**: for every worker, if the stop
prover `Ps`'s messages are the deferred server's frames below its last round (`RefDBelow`) and `Ps` meets `ZKmalR`,
and the go prover `Pg`'s are its frames wherever it releases one (`RefDOwn`) and `Pg` is simulated where its own run
doesn't show the stop, the release meets `ZKOut` at the verdict: on the stop at `Ps`'s simulator cut at the last round
with `Ps`'s error, otherwise at `Pg`'s simulator with `Pg`'s error. The verdict is read off the release, so no outcome
is guessed. -/
theorem zk_deferred_out {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (S : ZkServer W (N × Ω))
    [Fintype S.Chk] [Nonempty S.Chk] (fill : Msg → ℕ → ℕ → W → N × Ω → List (Msg × Msg) → S.Val) (H : Msg → Msg)
    (n : Msg → ℕ) (wk : S.Worker) (Ps Pg : W → Prover (N × Ω)) (Rl : Rel W)
    (hs : ∀ x w, Rl x w → S.RefDBelow H Ps fill n wk x w) (hg : ∀ x w, Rl x w → S.RefDOwn H Pg fill n wk x w)
    (Sims : Simulator) (εzk γ : Msg → ℝ) (hZK : ZKmalR H n Ps Rl Sims εzk γ) (Simg : Simulator)
    (εg : (x : Msg) → W → (Θ : Type) → VStar Θ → Θ → ℝ)
    (hgo : ∀ x w, Rl x w → ∀ (Θ : Type) (Vs : VStar Θ) (θ : Θ), Vs.k x = n x → ∀ Ev : List (Msg × Out) → Prop,
      Pr (fun ρ : N × Ω => stoppedAt S.stop (n x - 1) (runStar H (Pg w) Vs x ρ θ).msgs = false ∧
        Ev (runStar H (Pg w) Vs x ρ θ).msgs) ≤ Pr (fun c : Simg.C x => Ev (Simg.out H Vs θ x c)) + εg x w Θ Vs θ) :
    ZKOut (fun _ => N × Ω) (fun _ => S.Chk) H n (fun w _ => S.proverD fill n wk w) Rl
      (fun x τ => stoppedAt S.stop (n x - 1) τ) (verdictSim Sims Simg n S.stop)
      (fun x w Θ Vs θ o => if o then εzk x + γ x * (Fork H (Ps w) Vs θ x).Pr H else εg x w Θ Vs θ) := by
  intro x w hR Θ Vs θ hk o Ev
  beta_reduce
  set ask : List Msg → Out := fun as => ((Vs.next x as θ).eval H).1 with hask
  have hsrv : ∀ ω : (N × Ω) × S.Chk, (runStar H (S.proverD fill n wk w) Vs x ω θ).msgs =
      interact ask (S.nextD fill n wk x w ω.1 ω.2) (n x) [] := fun ω => by
    rw [runStar_msgs, hk]; rfl
  have hP : ∀ (P : W → Prover (N × Ω)) (ρ : N × Ω), (runStar H (P w) Vs x ρ θ).msgs =
      interact ask (fun h => (((P w).next x h ρ).eval H).1) (n x) [] := fun P ρ => by
    rw [runStar_msgs, hk]
  cases o with
  | true =>
    have hsim : Pr (fun c : (verdictSim Sims Simg n S.stop).C x =>
        Ev (((verdictSim Sims Simg n S.stop).given true).out H Vs θ x c)) =
        Pr (fun c : Sims.C x => Ev (cutT ask S.stop (n x - 1) (n x) (Sims.out H Vs θ x c))) := by
      refine Eq.trans ?_ (Pr_fst (β := Simg.C x)
        fun c : Sims.C x => Ev (cutT ask S.stop (n x - 1) (n x) (Sims.out H Vs θ x c)))
      exact Pr_congr fun c => iff_of_eq (congrArg Ev (cutSim_out H Sims (fun x => n x - 1) n S.stop Vs θ x c.1))
    have hA := hZK x w hR Θ Vs θ hk (fun τ => Ev (cutT ask S.stop (n x - 1) (n x) τ))
    calc Pr (fun ω : (N × Ω) × S.Chk =>
          stoppedAt S.stop (n x - 1) (runStar H (S.proverD fill n wk w) Vs x ω θ).msgs = true ∧
            Ev (runStar H (S.proverD fill n wk w) Vs x ω θ).msgs)
        ≤ Pr (fun ω : (N × Ω) × S.Chk =>
            Ev (cutT ask S.stop (n x - 1) (n x) (runStar H (Ps w) Vs x ω.1 θ).msgs)) := by
          refine Pr_mono fun ω hω => ?_
          rw [hsrv] at hω
          rw [hP, ← S.release_stopped H Ps fill n x w wk (hs x w hR) ask ω.1 ω.2 hω.1]
          exact hω.2
      _ = Pr (fun ρ : N × Ω => Ev (cutT ask S.stop (n x - 1) (n x) (runStar H (Ps w) Vs x ρ θ).msgs)) :=
          Pr_fst (fun ρ : N × Ω => Ev (cutT ask S.stop (n x - 1) (n x) (runStar H (Ps w) Vs x ρ θ).msgs))
      _ ≤ Pr (fun c : Sims.C x => Ev (cutT ask S.stop (n x - 1) (n x) (Sims.out H Vs θ x c))) +
            (εzk x + γ x * (Fork H (Ps w) Vs θ x).Pr H) := by
          linarith [le_abs_self
            (Pr (fun ρ : N × Ω => Ev (cutT ask S.stop (n x - 1) (n x) (runStar H (Ps w) Vs x ρ θ).msgs)) -
              Pr (fun c : Sims.C x => Ev (cutT ask S.stop (n x - 1) (n x) (Sims.out H Vs θ x c))))]
      _ = _ := by
          rw [hsim]
          try rfl
  | false =>
    have hsim : Pr (fun c : (verdictSim Sims Simg n S.stop).C x =>
        Ev (((verdictSim Sims Simg n S.stop).given false).out H Vs θ x c)) =
        Pr (fun c : Simg.C x => Ev (Simg.out H Vs θ x c)) :=
      pr_snd (α := Sims.C x) fun c : Simg.C x => Ev (Simg.out H Vs θ x c)
    calc Pr (fun ω : (N × Ω) × S.Chk =>
          stoppedAt S.stop (n x - 1) (runStar H (S.proverD fill n wk w) Vs x ω θ).msgs = false ∧
            Ev (runStar H (S.proverD fill n wk w) Vs x ω θ).msgs)
        ≤ Pr (fun ω : (N × Ω) × S.Chk => stoppedAt S.stop (n x - 1) (runStar H (Pg w) Vs x ω.1 θ).msgs = false ∧
            Ev (runStar H (Pg w) Vs x ω.1 θ).msgs) := by
          refine Pr_mono fun ω hω => ?_
          rw [hsrv] at hω
          rw [hP, ← S.release_go H Pg fill n x w wk (hg x w hR) ask ω.1 ω.2 hω.1]
          exact hω
      _ = Pr (fun ρ : N × Ω => stoppedAt S.stop (n x - 1) (runStar H (Pg w) Vs x ρ θ).msgs = false ∧
            Ev (runStar H (Pg w) Vs x ρ θ).msgs) :=
          Pr_fst (fun ρ : N × Ω => stoppedAt S.stop (n x - 1) (runStar H (Pg w) Vs x ρ θ).msgs = false ∧
            Ev (runStar H (Pg w) Vs x ρ θ).msgs)
      _ ≤ Pr (fun c : Simg.C x => Ev (Simg.out H Vs θ x c)) + εg x w Θ Vs θ := hgo x w hR Θ Vs θ hk Ev
      _ = _ := by
          rw [hsim]
          try rfl

end ZkServer

end Claims.ZKServer
