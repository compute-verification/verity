import Proofs.Flock.Claims.ZK.Interact
import Proofs.Flock.Claims.DeployCode

/-!
# Flock's `ZK` at the masked record's reference prover

`Claims.Flock.ZK` reads a masked record (`Claims.Flock.ZkDeploy`): its reference prover in the masked interaction
(`ZkDeploy.prover`, the record's `P` behind the `Hello` and the opening checks) and its simulator (`ZkDeploy.S`). The
proofs prove `Claims.Flock.ZKAt`, the same form at a committed-coin protocol's prover (`Proto.prover`). This module
carries one to the other:

* `zkmalR_congr`: `ZKmalR` at one prover gives it at another whose messages are the same at every history and whose
  calls include its own, at `γ ≥ 0`: the transcripts are the same (`runStar_msgs_congr`), and `Fork` finds at least as
  much (`fork_pr_mono`).
* `zkOf`: the masked record at a protocol `G`, `FlockDeploy.deployZk` with `G`'s nonce, tape, `Hello`, abort and opening
  check, a reading of each opening as its coin (`coinOf`), the reference prover's tape at `G`'s (`tapeOf`), and a
  simulator.
* `Proto.Honest`: `G`'s message after the openings is the record's reference prover's (`honestPZk`) at those openings'
  coins and the tape `tapeOf`, and `G` makes no call that it doesn't. Where it holds, the record's prover sends
  `G.prover`'s messages and makes its calls (`zkOf_prover`), so `ZKAt` at `G.prover` is `Claims.Flock.ZK` at `zkOf`
  (`zk_zkOf`).
-/

namespace Claims.ZKMal

open Claims
open Claims.Flock (allPass)

local notation "𝔽" => FlockLevel3.GF128

/-! ## One check, two binds -/

theorem ocBind_eq_bind {α β : Type} (f : α → OC β) : ∀ o : OC α, ocBind o f = o.bind f
  | .ret _ => rfl
  | .query q k => by
      simp only [ocBind, OC.bind]
      exact congrArg (OC.query q) (funext fun r => ocBind_eq_bind f (k r))

theorem allOk_eq_allPass (acc : ℕ → Msg → OC Bool) : ∀ (j : ℕ) (ms : List Msg), allOk acc j ms = allPass acc j ms
  | _, [] => rfl
  | j, m :: ms => by
      simp only [allOk, allPass, ocBind_eq_bind, allOk_eq_allPass acc (j + 1) ms]

/-! ## Two provers with the same messages within the rounds -/

/-- Two provers whose messages agree at every history shorter than `K` give the same transcript of at most `K` rounds. -/
theorem interact_congr_lt (ask : List Msg → Out) (pm pm' : List (Msg × Msg) → Msg) (K : ℕ)
    (hv : ∀ hist : List (Msg × Msg), hist.length < K → pm hist = pm' hist) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)), hist.length + n ≤ K → interact ask pm n hist = interact ask pm' n hist
  | 0, _, _ => rfl
  | n + 1, hist, hn => by
      have e := hv hist (by omega)
      simp only [interact, e]
      cases ask (hist.map Prod.fst ++ [pm' hist]) with
      | msg m =>
          dsimp only
          rw [interact_congr_lt ask pm pm' K hv n _ (by simp only [List.length_append, List.length_singleton]; omega)]
      | acc => rfl
      | rej => rfl

theorem runStar_msgs_congr {Ω Θ : Type} (H : Msg → Msg) (P Q : Prover Ω) (Vs : VStar Θ) (x : Msg) (ω : Ω) (θ : Θ)
    (hv : ∀ hist : List (Msg × Msg), hist.length < Vs.k x →
      ((P.next x hist ω).eval H).1 = ((Q.next x hist ω).eval H).1) :
    (runStar H P Vs x ω θ).msgs = (runStar H Q Vs x ω θ).msgs := by
  simp only [runStar_msgs]
  exact interact_congr_lt _ _ _ (Vs.k x) hv _ [] (by simp)

theorem stepWith_pLog_congr {Ω : Type} (H : Msg → Msg) (P Q : Prover Ω) (x : Msg) (ω : Ω)
    (vnext : List Msg → OC Out) (K : ℕ)
    (hv : ∀ hist : List (Msg × Msg), hist.length < K → ((P.next x hist ω).eval H).1 = ((Q.next x hist ω).eval H).1)
    (hl : ∀ hist : List (Msg × Msg), hist.length < K → ((P.next x hist ω).eval H).2 ⊆ ((Q.next x hist ω).eval H).2) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t t' : Trace), hist.length + n ≤ K → t.pLog ⊆ t'.pLog →
      (stepWith H P x ω vnext n hist as t).pLog ⊆ (stepWith H Q x ω vnext n hist as t').pLog
  | 0, _, _, _, _, _, h => by simpa [stepWith] using h
  | n + 1, hist, as, t, t', hn, h => by
      have hs : t.pLog ++ ((P.next x hist ω).eval H).2 ⊆ t'.pLog ++ ((Q.next x hist ω).eval H).2 :=
        List.append_subset.mpr
          ⟨List.Subset.trans h (List.subset_append_left _ _),
            List.Subset.trans (hl hist (by omega)) (List.subset_append_right _ _)⟩
      simp only [stepWith, hv hist (by omega)]
      cases ((vnext (as ++ [((Q.next x hist ω).eval H).1])).eval H).1 with
      | msg m =>
          exact stepWith_pLog_congr H P Q x ω vnext K hv hl n _ _ _ _
            (by simp only [List.length_append, List.length_singleton]; omega) hs
      | acc => exact hs
      | rej => exact hs

theorem runStar_pLog_congr {Ω Θ : Type} (H : Msg → Msg) (P Q : Prover Ω) (Vs : VStar Θ) (x : Msg) (ω : Ω) (θ : Θ)
    (hv : ∀ hist : List (Msg × Msg), hist.length < Vs.k x →
      ((P.next x hist ω).eval H).1 = ((Q.next x hist ω).eval H).1)
    (hl : ∀ hist : List (Msg × Msg), hist.length < Vs.k x →
      ((P.next x hist ω).eval H).2 ⊆ ((Q.next x hist ω).eval H).2) :
    (runStar H P Vs x ω θ).pLog ⊆ (runStar H Q Vs x ω θ).pLog :=
  stepWith_pLog_congr H P Q x ω _ (Vs.k x) hv hl (Vs.k x) [] [] _ _ (by simp) (List.Subset.refl _)

/-- **`Fork` finds at least as much at a prover with the same messages and more calls** within `V*`'s rounds. -/
theorem fork_pr_mono {N Ω Θ : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg)
    (P Q : Prover (N × Ω)) (Vs : VStar Θ) (θ : Θ) (x : Msg)
    (hv : ∀ (hist : List (Msg × Msg)) ω, hist.length < Vs.k x →
      ((P.next x hist ω).eval H).1 = ((Q.next x hist ω).eval H).1)
    (hl : ∀ (hist : List (Msg × Msg)) ω, hist.length < Vs.k x →
      ((P.next x hist ω).eval H).2 ⊆ ((Q.next x hist ω).eval H).2) :
    (Fork H P Vs θ x).Pr H ≤ (Fork H Q Vs θ x).Pr H := by
  refine le_trans (Pr_mono fun s hs => ?_) (Fork_pr_ge H Q Vs θ x)
  obtain ⟨u, v, hs, -⟩ := hs
  have hne : (Fork H P Vs θ x).out s ≠ none := by
    rw [hs]
    exact Option.some_ne_none _
  obtain ⟨p, hp, q, hq, h₁, h₂⟩ := (findColl_ne_none_iff _).mp hne
  have hsub : ∀ e ∈ (runStar H P Vs x (s.1, s.2.1) θ).pLog ++ (runStar H P Vs x (s.1, s.2.2) θ).pLog,
      e ∈ (runStar H Q Vs x (s.1, s.2.1) θ).pLog ++ (runStar H Q Vs x (s.1, s.2.2) θ).pLog := fun e he => by
    rcases List.mem_append.mp he with he | he
    · exact List.mem_append_left _
        (runStar_pLog_congr H P Q Vs x _ θ (fun h hh => hv h _ hh) (fun h hh => hl h _ hh) he)
    · exact List.mem_append_right _
        (runStar_pLog_congr H P Q Vs x _ θ (fun h hh => hv h _ hh) (fun h hh => hl h _ hh) he)
  exact (findColl_ne_none_iff _).mpr ⟨p, hsub p hp, q, hsub q hq, h₁, h₂⟩

/-- **`ZKmalR` carries over to a prover with the same messages and more calls** within the rounds, at `γ ≥ 0`: the real
transcripts are the same, and its `Fork` finds at least as much. -/
theorem zkmalR_congr {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] {H : Msg → Msg} {k : Msg → ℕ}
    {P Q : W → Prover (N × Ω)} {R : Rel W} {S : Simulator} {εzk γ : Msg → ℝ} (hγ : ∀ x, 0 ≤ γ x)
    (hv : ∀ w x (hist : List (Msg × Msg)) ω, hist.length < k x →
      (((P w).next x hist ω).eval H).1 = (((Q w).next x hist ω).eval H).1)
    (hl : ∀ w x (hist : List (Msg × Msg)) ω, hist.length < k x →
      (((P w).next x hist ω).eval H).2 ⊆ (((Q w).next x hist ω).eval H).2)
    (h : ZKmalR H k P R S εzk γ) : ZKmalR H k Q R S εzk γ := by
  intro x w hR Θ Vs θ hk Ev
  have e : ∀ ω, (runStar H (Q w) Vs x ω θ).msgs = (runStar H (P w) Vs x ω θ).msgs := fun ω =>
    (runStar_msgs_congr H (P w) (Q w) Vs x ω θ fun hist hh => hv w x hist ω (hk ▸ hh)).symm
  simp only [e]
  exact (h x w hR Θ Vs θ hk Ev).trans (add_le_add le_rfl (mul_le_mul_of_nonneg_left
    (fork_pr_mono H (P w) (Q w) Vs θ x (fun hist ω hh => hv w x hist ω (hk ▸ hh))
      (fun hist ω hh => hl w x hist ω (hk ▸ hh))) (hγ x)))

/-- **`ZKmalR` carries over to a prover with the same messages and the same calls** within the rounds, at every `γ`. -/
theorem zkmalR_congr_eq {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] {H : Msg → Msg}
    {k : Msg → ℕ} {P Q : W → Prover (N × Ω)} {R : Rel W} {S : Simulator} {εzk γ : Msg → ℝ}
    (hv : ∀ w x (hist : List (Msg × Msg)) ω, hist.length < k x →
      ((P w).next x hist ω).eval H = ((Q w).next x hist ω).eval H)
    (h : ZKmalR H k P R S εzk γ) : ZKmalR H k Q R S εzk γ := by
  intro x w hR Θ Vs θ hk Ev
  have e : ∀ ω, (runStar H (Q w) Vs x ω θ).msgs = (runStar H (P w) Vs x ω θ).msgs := fun ω =>
    (runStar_msgs_congr H (P w) (Q w) Vs x ω θ fun hist hh => by rw [hv w x hist ω (hk ▸ hh)]).symm
  have f : (Fork H (Q w) Vs θ x).Pr H = (Fork H (P w) Vs θ x).Pr H :=
    le_antisymm
      (fork_pr_mono H _ _ Vs θ x (fun hist ω hh => by rw [hv w x hist ω (hk ▸ hh)])
        (fun hist ω hh => by rw [hv w x hist ω (hk ▸ hh)]))
      (fork_pr_mono H _ _ Vs θ x (fun hist ω hh => by rw [hv w x hist ω (hk ▸ hh)])
        (fun hist ω hh => by rw [hv w x hist ω (hk ▸ hh)]))
  simp only [e, f]
  exact h x w hR Θ Vs θ hk Ev

/-! ## The masked record at a committed-coin protocol -/

/-- **The masked record at a committed-coin protocol** `G`: `FlockDeploy.deployZk rel` with `G`'s nonce and tape, its
`Hello`, abort message and opening check, the coin each opening reads as (`coinOf`), the reference prover's tape at
`G`'s (`tapeOf`), and the simulator `S`. -/
noncomputable def zkOf (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (G : Proto Flock.Wit)
    (coinOf : Msg → Msg) (tapeOf : G.Nonce × G.Rand → FlockComplete.TapeZk) (S : Simulator) :
    Claims.Flock.ZkDeploy FlockDeploy.Coin (List 𝔽) Claims.Flock.ZkTrace :=
  { FlockDeploy.deployZk rel with
    Nonce := G.Nonce, Rest := G.Rand, tapeOf := tapeOf, hello := G.hello, abort := G.abort, chk := G.chk,
    coinOf := coinOf, S := S }

/-- **`G`'s messages are the reference prover's** within `G`'s rounds, at `H`: after the openings `ops` (the history
after `Hello`'s answer, at most `G.K x` of them), `G`'s message at the coins they open (`dec`) is `P w`'s at the history
with each opening read as its coin (`coinOf`), at the tape `tapeOf (ν, r)`, and its calls are among `P w`'s. -/
def Proto.Honest {Ω : Type} (G : Proto Flock.Wit) (H : Msg → Msg) (P : Flock.Wit → Prover Ω) (coinOf : Msg → Msg)
    (tapeOf : G.Nonce × G.Rand → Ω) : Prop :=
  ∀ x w ν r (ops : List (Msg × Msg)), ops.length ≤ G.K x →
    ((G.msg x w ν r (ops.map fun p => G.dec p.2)).eval H).1 =
        (((P w).next x (ops.map fun p => (p.1, coinOf p.2)) (tapeOf (ν, r))).eval H).1 ∧
      ((G.msg x w ν r (ops.map fun p => G.dec p.2)).eval H).2 ⊆
        (((P w).next x (ops.map fun p => (p.1, coinOf p.2)) (tapeOf (ν, r))).eval H).2

/-- **The masked record's reference prover sends `G.prover`'s messages and makes its calls** within `G`'s rounds, where
`G`'s messages are the reference prover's (`Proto.Honest` at `honestPZk`): both send `Hello`, check the openings with
`G.chk`, and abort alike; past the checks, `G`'s message is `honestPZk`'s. -/
theorem zkOf_prover (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (G : Proto Flock.Wit)
    (coinOf : Msg → Msg) (tapeOf : G.Nonce × G.Rand → FlockComplete.TapeZk) (S : Simulator) (H : Msg → Msg)
    (hG : G.Honest H FlockComplete.honestPZk coinOf tapeOf) (w : Flock.Wit) (x : Msg) (hist : List (Msg × Msg))
    (ω : G.Nonce × G.Rand) (hlen : hist.length < G.k x) :
    (((G.prover w).next x hist ω).eval H).1 = ((((zkOf rel G coinOf tapeOf S).prover w).next x hist ω).eval H).1 ∧
      (((G.prover w).next x hist ω).eval H).2 ⊆ ((((zkOf rel G coinOf tapeOf S).prover w).next x hist ω).eval H).2 := by
  cases hist with
  | nil => exact ⟨rfl, List.Subset.refl _⟩
  | cons p ops =>
    have hops : ops.length ≤ G.K x := by
      simp only [List.length_cons, Proto.k] at hlen
      omega
    obtain ⟨a, ρ⟩ := p
    have e₁ : (G.prover w).next x ((a, ρ) :: ops) ω = (allPass (G.chk x ω.1 ρ) 0 (ops.map Prod.snd)).bind
        fun ok => if ok then G.msg x w ω.1 ω.2 (ops.map fun p => G.dec p.2) else .ret G.abort := by
      show ocBind (allOk (G.chk x ω.1 ρ) 0 (ops.map Prod.snd)) _ = _
      rw [ocBind_eq_bind, allOk_eq_allPass]
    have e₂ : ((zkOf rel G coinOf tapeOf S).prover w).next x ((a, ρ) :: ops) ω =
        (allPass (G.chk x ω.1 ρ) 0 (ops.map Prod.snd)).bind fun ok =>
          if ok then (FlockComplete.honestPZk w).next x (ops.map fun p => (p.1, coinOf p.2)) (tapeOf ω)
          else .ret G.abort := rfl
    rw [e₁, e₂, OC.eval_bind, OC.eval_bind]
    obtain ⟨h₁, h₂⟩ := hG x w ω.1 ω.2 ops hops
    cases ((allPass (G.chk x ω.1 ρ) 0 (ops.map Prod.snd)).eval H).1 with
    | true =>
        simp only [↓reduceIte]
        exact ⟨h₁, List.append_subset.mpr
          ⟨List.subset_append_left _ _, List.Subset.trans h₂ (List.subset_append_right _ _)⟩⟩
    | false => exact ⟨rfl, List.Subset.refl _⟩

/-- **`Claims.Flock.ZK` at the masked record of a committed-coin protocol**, from `ZKAt` at its prover: where `G`'s
rounds are `Hello` and the deployed verifier's (`deployZk_proto_k`) and its messages are the reference prover's
(`Proto.Honest`), at `γ ≥ 0`. -/
theorem zk_zkOf (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (G : Proto Flock.Wit)
    (coinOf : Msg → Msg) (tapeOf : G.Nonce × G.Rand → FlockComplete.TapeZk) (S : Simulator) {H : Msg → Msg}
    {εzk γ : Msg → ℝ} (hγ : ∀ x, 0 ≤ γ x) (hk : ∀ x, G.k x = (FlockDeploy.deployZk rel).V.k x + 1)
    (hG : G.Honest H FlockComplete.honestPZk coinOf tapeOf)
    (h : Claims.Flock.ZKAt H (FlockDeploy.deployZk rel) G.prover S εzk γ) :
    Claims.Flock.ZK H (zkOf rel G coinOf tapeOf S) εzk γ :=
  ⟨h.1, zkmalR_congr hγ
    (fun w x hist ω hh => (zkOf_prover rel G coinOf tapeOf S H hG w x hist ω (lt_of_lt_of_eq hh (hk x).symm)).1)
    (fun w x hist ω hh => (zkOf_prover rel G coinOf tapeOf S H hG w x hist ω (lt_of_lt_of_eq hh (hk x).symm)).2)
    h.2⟩

end Claims.ZKMal
