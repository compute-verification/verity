module

public import Definitions.Claims.SamplingGeneric
public import Proofs.Flock.Claims.Recursion.SeqPrefix
public import Proofs.Flock.Claims.ZK.Congr
public import Proofs.Flock.Claims.KS.FnFork
public import Proofs.Flock.Claims.KS.Wait

/-!
# The sampled compiler is zero knowledge (`CompilerZK`)

The masked compiled verifier's transcript is the framing's rounds, then `M` sessions in turn, each `A`'s prover against
`V*` after the run's messages before it (`ZkCompiled.residual`). The proof is a hybrid over the sessions, last first:
hybrid `j` runs the honest prover for the first `j` sessions and simulates the rest (`hyb`). Going from hybrid `j + 1`
to hybrid `j` resamples slot `j`'s tape and simulator coins (`Pr_resample`): with the earlier sessions' tapes fixed,
session `j` is `A`'s run at the statement its history routes to, so `A`'s zero knowledge there, at the outcome the
audit's outcome gives it and the event that runs the later sessions' simulators after it, costs `A`'s error at
`residual` where the honest run reaches the session (`step_le`). Hybrid `M` is the honest run (`hyb_M`), and hybrid `0`
the composed simulator (`hyb_zero`), since the framing's messages are functions of the statement and the history.

* `Ex_resample`, `Pr_resample`: resampling one coordinate keeps the law.
* `pre`, `pre_succ`: the honest run's first sessions; where it goes on after session `j`'s start, the next session is
  `A`'s run against `residual`. `pre_congr`: the first `j` sessions read only the first `j` slots.
* `hyb`, `step_le`, `hyb_M`, `hyb_zero`: the hybrids.
* **`compilerZK : CompilerZK`**.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims

/-! ## Transcripts and simulator programs -/

theorem histOf_eq (ms : List (Msg × Out)) : histOf ms = ZKMal.histOf ms := by
  unfold histOf ZKMal.histOf
  congr 1
  funext p
  obtain ⟨a, o⟩ := p
  cases o <;> rfl

theorem recGoesOn_cons_msg (a m : Msg) (τ : List (Msg × Out)) :
    Recursion.goesOn ((a, .msg m) :: τ) = Recursion.goesOn τ := by
  induction τ using List.reverseRecOn with
  | nil => rfl
  | append_singleton l p _ =>
    obtain ⟨b, o⟩ := p
    rw [← List.cons_append, ZKMal.recGoesOn_snoc, ZKMal.recGoesOn_snoc]

/-- An honest run's transcript goes on, in the claims' sense, exactly where it has all its rounds, each a message. -/
theorem recGoesOn_interact (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)),
      Recursion.goesOn (ZKMal.interact ask pm n hist) = ZKMal.goesOn n (ZKMal.interact ask pm n hist)
  | 0, _ => rfl
  | n + 1, hist => by
    simp only [ZKMal.interact]
    cases ask (hist.map Prod.fst ++ [pm hist]) with
    | msg m => rw [recGoesOn_cons_msg, ZKMal.goesOn_cons]; exact recGoesOn_interact ask pm n _
    | acc => simp [Recursion.goesOn, ZKMal.goesOn, ZKMal.isMsg]
    | rej => simp [Recursion.goesOn, ZKMal.goesOn, ZKMal.isMsg]

theorem oc_bind_eval_fst (H : Msg → Msg) {α β : Type} (o : OC α) (f : α → OC β) :
    ((o.bind f).eval H).1 = ((f (o.eval H).1).eval H).1 := by
  induction o with
  | ret a => rfl
  | query q k ih => exact ih (H q)

theorem spUnder_eval_fst {α : Type} (H : Msg → Msg) (ask : List Msg → Out) (pre : List Msg) (p : SP α) :
    (SP.eval H ask (Recursion.spUnder pre p)).1 = (SP.eval H (fun as => ask (pre ++ as)) p).1 := by
  rw [ZKMal.recSpUnder_eq, ZKMal.spUnder_eval]

theorem simAt_of {St : Type} {S : SimOut St} {C : Type} {xS : Msg} (h : Nonempty (C ≃ S.C xS)) (o : St) (c : C) :
    simAt S C xS o c = S.prog xS o (h.some c) := by
  unfold simAt
  split
  · rfl
  · contradiction

/-! ## Resampling one coordinate -/

/-- **Resampling keeps the law**: where `(a, b) ↦ (u a b, v a)` is an involution, drawing `a`, then `b` fresh in place
of `v a`, draws `u a b` uniformly. -/
theorem Ex_resample {α β : Type} [Fintype α] [Fintype β] [Nonempty β] (u : α → β → α) (v : α → β)
    (h1 : ∀ a b, u (u a b) (v a) = a) (h2 : ∀ a b, v (u a b) = b) (f : α → ℝ) :
    (Ex fun a => Ex fun b => f (u a b)) = Ex f := by
  let Ψ : α × β ≃ α × β :=
    { toFun := fun p => (u p.1 p.2, v p.1)
      invFun := fun p => (u p.1 p.2, v p.1)
      left_inv := fun p => by simp only [h1, h2]
      right_inv := fun p => by simp only [h1, h2] }
  have e1 := Flock.FnFork.Ex_prod (fun p : α × β => f (u p.1 p.2))
  have e2 := Flock.FnFork.Ex_equiv Ψ (fun q : α × β => f q.1)
  have e3 := Flock.FnFork.Ex_prod (fun q : α × β => f q.1)
  simp only at e1 e3
  rw [← e1]
  refine e2.trans ?_
  rw [e3]
  congr 1
  funext a
  exact Ex_const _

theorem Pr_resample {α β : Type} [Fintype α] [Fintype β] [Nonempty β] (u : α → β → α) (v : α → β)
    (h1 : ∀ a b, u (u a b) (v a) = a) (h2 : ∀ a b, v (u a b) = b) (p : α → Prop) :
    Pr p = Ex fun a => Pr fun b => p (u a b) := by
  rw [Flock.FnFork.Pr_eq_Ex, ← Ex_resample u v h1 h2]
  simp only [Flock.FnFork.Pr_eq_Ex]
  rfl

theorem Pr_snd {α β : Type} [Fintype α] [Nonempty α] [Fintype β] (q : β → Prop) :
    Pr (fun p : α × β => q p.2) = Pr q := by
  rw [← Pr_fst (β := α) q, ← Flock.FnFork.Pr_equiv (Equiv.prodComm α β) fun p : β × α => q p.1]
  rfl

theorem Ex_fst {α β : Type} [Fintype α] [Fintype β] [Nonempty β] (f : α → ℝ) :
    Ex (fun p : α × β => f p.1) = Ex f := by
  rw [Flock.FnFork.Ex_prod]
  congr 1
  funext a
  exact Ex_const _

/-! ## The masked compiled verifier's honest run -/

namespace ZkCompiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : ZkArg Coin} (L : ZkCompiled Coin A)

theorem start_zero (x : Msg) : L.start x 0 = L.k₀ x := by simp [start]

theorem start_succ (x : Msg) (j : ℕ) : L.start x (j + 1) = L.start x j + L.n x j := by
  simp [start, Finset.sum_range_succ, add_assoc]

theorem start_mono (x : Msg) {i j : ℕ} (h : i ≤ j) : L.start x i ≤ L.start x j := by
  induction h with
  | refl => exact le_rfl
  | step _ ih => rw [L.start_succ]; omega

/-- A round after the first part and before session `j` is in some earlier session. -/
theorem locate (x : Msg) (len : ℕ) :
    ∀ j, L.k₀ x ≤ len → len < L.start x j → ∃ i < j, L.start x i ≤ len ∧ len < L.start x (i + 1)
  | 0, h0, h => by rw [start_zero] at h; omega
  | j + 1, h0, h => by
    by_cases hj : len < L.start x j
    · obtain ⟨i, hi, h1, h2⟩ := locate x len j h0 hj
      exact ⟨i, by omega, h1, h2⟩
    · exact ⟨j, by omega, by omega, h⟩

/-- The tape at `x`: a nonce and a rest per session. -/
abbrev Tp (x : Msg) := (Fin (L.M x) → A.Nonce) × (Fin (L.M x) → A.Rest)

/-- The composed prover's message at a history. -/
def pm (H : Msg → Msg) (w : Sampling.Wit) (x : Msg) (t : L.Tp x) (hist : List (Msg × Msg)) : Msg :=
  (((L.composed w x).next x hist t).eval H).1

/-- Session `j`'s witness at its statement. -/
def wS (H : Msg → Msg) (w : Sampling.Wit) (x xS : Msg) : A.W := ((L.wit w x xS).eval H).1

/-- Slot `j`'s nonce and rest. -/
def slot {x : Msg} (t : L.Tp x) (j : Fin (L.M x)) : A.Nonce × A.Rest := (t.1 j, t.2 j)

variable {L} {H : Msg → Msg}

theorem pm_say {w : Sampling.Wit} {x : Msg} (t : L.Tp x) {hist : List (Msg × Msg)} {m : Msg}
    (h : L.step x hist = .say m) : L.pm H w x t hist = m := by
  simp [pm, composed, h, OC.eval]

theorem pm_session {w : Sampling.Wit} {x : Msg} (t : L.Tp x) {hist : List (Msg × Msg)} {j : ℕ} {xS : Msg}
    {hS : List (Msg × Msg)} (h : L.step x hist = .session j xS hS) (hj : j < L.M x) :
    L.pm H w x t hist = (((A.P (L.wS H w x xS)).next xS hS (L.slot t ⟨j, hj⟩)).eval H).1 := by
  simp only [pm, composed, h, hj, ↓reduceDIte]
  exact oc_bind_eval_fst H _ _

variable (hS : L.ZkSessions H) {x : Msg} {s : AuditStatement} (hs : L.parse x = some s)
include hS hs

theorem pm_framing (w : Sampling.Wit) (t : L.Tp x) {hist : List (Msg × Msg)} (hl : hist.length < L.k₀ x) :
    L.pm H w x t hist = (L.step x hist).said := by
  obtain ⟨m, hm⟩ := (hS.routed x s hs).1 hist hl
  rw [pm_say t hm, hm]
  rfl

theorem pm_at (w : Sampling.Wit) (t : L.Tp x) {i : ℕ} (hi : i < L.M x) {hist : List (Msg × Msg)}
    (h1 : L.start x i ≤ hist.length) (h2 : hist.length < L.start x (i + 1)) :
    L.pm H w x t hist =
      (((A.P (L.wS H w x (L.step x (hist.take (L.start x i))).stmt)).next (L.step x (hist.take (L.start x i))).stmt
        (hist.drop (L.start x i)) (L.slot t ⟨i, hi⟩)).eval H).1 :=
  pm_session t ((hS.routed x s hs).2 i hi hist h1 h2) hi

/-- **The first `j` sessions read only the first `j` slots.** -/
theorem pm_congr (w : Sampling.Wit) {t t' : L.Tp x} {j : ℕ} (hj : j ≤ L.M x)
    (ht : ∀ i : Fin (L.M x), i.val < j → L.slot t i = L.slot t' i) {hist : List (Msg × Msg)}
    (hl : hist.length < L.start x j) : L.pm H w x t hist = L.pm H w x t' hist := by
  by_cases h0 : hist.length < L.k₀ x
  · rw [pm_framing hS hs w t h0, pm_framing hS hs w t' h0]
  · obtain ⟨i, hi, h1, h2⟩ := L.locate x hist.length j (by omega) hl
    rw [pm_at hS hs w t (by omega) h1 h2, pm_at hS hs w t' (by omega) h1 h2, ht ⟨i, by omega⟩ hi]

end ZkCompiled

end Claims.SamplingGeneric
