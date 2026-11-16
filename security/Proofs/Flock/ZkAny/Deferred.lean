module

public import Definitions.Claims.ZKOut
public import Proofs.Flock.Claims.ZK.GKOneSided
public import Proofs.Flock.ZkAny.Unswitched

/-!
# The deferred release per outcome: the `Finish` verdict

The server that defers every stop (`ZkServer.proverD`) releases, at its last round `n x - 1`, its frame or the stop.
The verdict read off the release (`goesOnAt`: it reaches its last round and its message there is not the stop) is its
public outcome, and on each outcome the release is one prover's run:

* `release_stopped`: where the release doesn't go on at its last round (it shows the stop there, or ended before it),
  it is the stop prover's run cut there, for any prover whose messages are the deferred server's frames below its last
  round (`RefDBelow`).
* `zk_deferred_out`: so the deferred release is zero knowledge per outcome (`ZKOut`) at the verdict, from the stop
  prover's zero knowledge (`ZKmalR`) read through the cut (`cutSim`), and, where the release goes on, its being a go
  prover's run and that prover simulated where its own run goes on: no guess of the outcome, each outcome's simulator
  its own (`verdictSim`), fixed before the witness.
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal

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

/-- **The deferred release's simulator given its verdict**: where it goes on, the go prover's simulator; otherwise the
stop prover's cut at the last round `n x - 1`. Its coins are both simulators'. -/
def verdictSim (Simg Sims : Simulator) (n : Msg → ℕ) (stop : Msg) : SimOut Bool where
  C x := Simg.C x × Sims.C x
  prog x o c := if o then Simg.prog x c.1 else (cutSim Sims (fun x => n x - 1) n stop).prog x c.2

private theorem pr_snd {α β : Type} [Fintype α] [Fintype β] [Nonempty α] (p : β → Prop) :
    Pr (fun s : α × β => p s.2) = Pr p := by
  classical
  rw [Pr_eq_sum (fun s : α × β => p s.2), Pr_eq_sum p, Fintype.sum_prod_type, Fintype.card_prod]
  have : (0 : ℝ) < Fintype.card α := by exact_mod_cast Fintype.card_pos
  have e : ∀ a : α, (∑ b : β, if p (a, b).2 then (1 : ℝ) else 0) = ∑ b : β, if p b then (1 : ℝ) else 0 :=
    fun _ => rfl
  rw [Finset.sum_congr rfl fun a _ => e a, Finset.sum_const, Finset.card_univ, nsmul_eq_mul, Nat.cast_mul]
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

/-- **Where the release doesn't go on at its last round, it is the stop prover's run cut there**: for every worker,
check coins and tape, if `P`'s messages are the deferred server's frames below its last round (`RefDBelow`), a release
whose message at round `n x - 1` is the stop, or that ended before it, is `P`'s run with that message the stop. -/
theorem release_stopped {W R : Type} (S : ZkServer W R) (H : Msg → Msg) (P : W → Prover R)
    (fill : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → S.Val) (n : Msg → ℕ) (x : Msg) (w : W) (wk : S.Worker)
    (hRef : S.RefDBelow H P fill n wk x w) (ask : List Msg → Out) (ρ : R) (κ : S.Chk)
    (hs : goesOnAt S.stop (n x - 1) (interact ask (S.nextD fill n wk x w ρ κ) (n x) []) = false) :
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
    rw [← hτ, List.nil_append, List.getElem?_eq_getElem hi] at hg
    have hg' : (τ[i]'hi).1 = S.nextD fill n wk x w ρ κ (histOf (τ.take i)) := Option.some.inj hg
    rw [← hiN] at hs
    unfold goesOnAt at hs
    rw [List.getElem?_eq_getElem hi] at hs
    have hs' : (τ[i]'hi).1 = S.stop := by simpa using hs
    rw [← hg', hs']

/-- **The deferred release is zero knowledge per outcome at its `Finish` verdict**: for every worker, if the release
that goes on at its last round is the go prover `Pg`'s run and `Pg` is simulated where its own run goes on there, and
the stop prover `Ps`'s messages are the deferred server's frames below its last round (`RefDBelow`) and `Ps` meets
`ZKmalR`, the release meets `ZKOut` at the verdict: where it goes on at `Pg`'s simulator with `Pg`'s error, otherwise
at `Ps`'s simulator cut at the last round with `Ps`'s error. The verdict is read off the release, so no outcome is
guessed. -/
theorem zk_deferred_out {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (S : ZkServer W (N × Ω))
    (fill : Msg → ℕ → ℕ → W → N × Ω → List (Msg × Msg) → S.Val) (H : Msg → Msg)
    (n : Msg → ℕ) (wk : S.Worker) (Pg Ps : W → Prover (N × Ω)) (Rl : Rel W)
    (hg : ∀ x w, Rl x w → ∀ (ask : List Msg → Out) (ρ : N × Ω) (κ : S.Chk),
      goesOnAt S.stop (n x - 1) (interact ask (S.nextD fill n wk x w ρ κ) (n x) []) = true →
      interact ask (S.nextD fill n wk x w ρ κ) (n x) [] = interact ask (fun h => (((Pg w).next x h ρ).eval H).1) (n x) [])
    (hs : ∀ x w, Rl x w → S.RefDBelow H Ps fill n wk x w) (Simg Sims : Simulator)
    (εg : (x : Msg) → W → (Θ : Type) → VStar Θ → Θ → ℝ) (εs γs : Msg → ℝ)
    (hgo : ∀ x w, Rl x w → ∀ (Θ : Type) (Vs : VStar Θ) (θ : Θ), Vs.k x = n x → ∀ Ev : List (Msg × Out) → Prop,
      Pr (fun ρ : N × Ω => goesOnAt S.stop (n x - 1) (runStar H (Pg w) Vs x ρ θ).msgs = true ∧
        Ev (runStar H (Pg w) Vs x ρ θ).msgs) ≤ Pr (fun c : Simg.C x => Ev (Simg.out H Vs θ x c)) + εg x w Θ Vs θ)
    (hZKs : ZKmalR H n Ps Rl Sims εs γs) :
    ZKOut (fun _ => N × Ω) (fun _ => S.Chk) H n (fun w _ => S.proverD fill n wk w) Rl
      (fun x τ => goesOnAt S.stop (n x - 1) τ) (verdictSim Simg Sims n S.stop)
      (fun x w Θ Vs θ o => if o then εg x w Θ Vs θ else εs x + γs x * (Fork H (Ps w) Vs θ x).Pr H) := by
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
    have hsim : Pr (fun c : (verdictSim Simg Sims n S.stop).C x =>
        Ev (((verdictSim Simg Sims n S.stop).given true).out H Vs θ x c)) =
        Pr (fun c : Simg.C x => Ev (Simg.out H Vs θ x c)) :=
      Pr_fst (β := Sims.C x) fun c : Simg.C x => Ev (Simg.out H Vs θ x c)
    calc Pr (fun ω : (N × Ω) × S.Chk =>
          goesOnAt S.stop (n x - 1) (runStar H (S.proverD fill n wk w) Vs x ω θ).msgs = true ∧
            Ev (runStar H (S.proverD fill n wk w) Vs x ω θ).msgs)
        ≤ Pr (fun ω : (N × Ω) × S.Chk => goesOnAt S.stop (n x - 1) (runStar H (Pg w) Vs x ω.1 θ).msgs = true ∧
            Ev (runStar H (Pg w) Vs x ω.1 θ).msgs) := by
          refine Pr_mono fun ω hω => ?_
          rw [hsrv] at hω
          rw [hP, ← hg x w hR ask ω.1 ω.2 hω.1]
          exact hω
      _ = Pr (fun ρ : N × Ω => goesOnAt S.stop (n x - 1) (runStar H (Pg w) Vs x ρ θ).msgs = true ∧
            Ev (runStar H (Pg w) Vs x ρ θ).msgs) :=
          Pr_fst (fun ρ : N × Ω => goesOnAt S.stop (n x - 1) (runStar H (Pg w) Vs x ρ θ).msgs = true ∧
            Ev (runStar H (Pg w) Vs x ρ θ).msgs)
      _ ≤ Pr (fun c : Simg.C x => Ev (Simg.out H Vs θ x c)) + εg x w Θ Vs θ := hgo x w hR Θ Vs θ hk Ev
      _ = _ := by
          rw [hsim]
          try rfl
  | false =>
    have hsim : Pr (fun c : (verdictSim Simg Sims n S.stop).C x =>
        Ev (((verdictSim Simg Sims n S.stop).given false).out H Vs θ x c)) =
        Pr (fun c : Sims.C x => Ev (cutT ask S.stop (n x - 1) (n x) (Sims.out H Vs θ x c))) := by
      refine Eq.trans ?_ (pr_snd (α := Simg.C x)
        fun c : Sims.C x => Ev (cutT ask S.stop (n x - 1) (n x) (Sims.out H Vs θ x c)))
      exact Pr_congr fun c => iff_of_eq (congrArg Ev (cutSim_out H Sims (fun x => n x - 1) n S.stop Vs θ x c.2))
    have hA := hZKs x w hR _ Vs θ hk (fun τ => Ev (cutT ask S.stop (n x - 1) (n x) τ))
    calc Pr (fun ω : (N × Ω) × S.Chk =>
          goesOnAt S.stop (n x - 1) (runStar H (S.proverD fill n wk w) Vs x ω θ).msgs = false ∧
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
            (εs x + γs x * (Fork H (Ps w) Vs θ x).Pr H) := by
          linarith [le_abs_self
            (Pr (fun ρ : N × Ω => Ev (cutT ask S.stop (n x - 1) (n x) (runStar H (Ps w) Vs x ρ θ).msgs)) -
              Pr (fun c : Sims.C x => Ev (cutT ask S.stop (n x - 1) (n x) (Sims.out H Vs θ x c))))]
      _ = _ := by
          rw [hsim]
          try rfl

end ZkServer

end Claims.ZKServer
