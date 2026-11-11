module

public import Definitions.Claims.SamplingGeneric
public import Proofs.Flock.Claims.Recursion.SeqPrefix
public import Proofs.Flock.Claims.ZK.Congr
public import Proofs.Flock.Claims.KS.FnFork
public import Proofs.Flock.Claims.KS.Wait
public import Proofs.Flock.Claims.Sampling.Wait

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

* `Ex_resample`, `Pr_resample`: resampling one coordinate keeps the sampling design; `Pr_le_resample`: a hybrid step from two
  resamplings.
* `pre`, `pre_succ`: the honest run's first sessions; where it goes on after session `j`'s start, the next session is
  `A`'s run against `residual`. `pre_congr`: the first `j` sessions read only the first `j` slots.
* `hyb`, `step_le`, `hyb_M`, `hyb_zero`: the hybrids, hybrid `j` with each of the first `j` sessions showing its
  outcome; `errAt`: `err`'s term at a session.
* **`compilerZK : CompilerZK`**.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims

/-! ## Transcripts and simulator programs -/

theorem histOf_eq (ms : List (Msg × Out)) : histOf ms = Claims.histOf ms := by
  unfold histOf Claims.histOf
  congr 1

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
      Recursion.goesOn (Claims.interact ask pm n hist) = ZKMal.goesOn n (Claims.interact ask pm n hist)
  | 0, _ => rfl
  | n + 1, hist => by
    simp only [Claims.interact]
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

/-- **Resampling keeps the sampling design**: where `(a, b) ↦ (u a b, v a)` is an involution, sampling `a`, then `b` fresh in place
of `v a`, samples `u a b` uniformly. -/
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
  exact Ex_const (α := β) (f a)

/-! ## The masked compiled verifier's honest run -/

namespace ZkCompiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : ZkArg Coin} (L : ZkCompiled Coin A)

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem start_zero (x : Msg) : L.start x 0 = L.k₀ x := by simp [start]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem start_succ (x : Msg) (j : ℕ) : L.start x (j + 1) = L.start x j + L.n x j := by
  simp [start, Finset.sum_range_succ, add_assoc]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem start_mono (x : Msg) {i j : ℕ} (h : i ≤ j) : L.start x i ≤ L.start x j := by
  induction h with
  | refl => exact le_rfl
  | step _ ih => rw [L.start_succ]; omega

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
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

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem pm_say {w : Sampling.Wit} {x : Msg} (t : L.Tp x) {hist : List (Msg × Msg)} {m : Msg}
    (h : L.step x hist = .say m) : L.pm H w x t hist = m := by
  simp [pm, composed, h, OC.eval]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem pm_session {w : Sampling.Wit} {x : Msg} (t : L.Tp x) {hist : List (Msg × Msg)} {j : ℕ} {xS : Msg}
    {hS : List (Msg × Msg)} (h : L.step x hist = .session j xS hS) (hj : j < L.M x) :
    L.pm H w x t hist = (((A.P (L.wS H w x xS)).next xS hS (L.slot t ⟨j, hj⟩)).eval H).1 := by
  simp only [pm, composed, h, hj, ↓reduceDIte]
  exact oc_bind_eval_fst H _ _

variable (hS : L.ZkSessions H) {x : Msg} {s : Sampling.AuditStatement} (hs : L.parse x = some s)
include hS hs

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem pm_framing (w : Sampling.Wit) (t : L.Tp x) {hist : List (Msg × Msg)} (hl : hist.length < L.k₀ x) :
    L.pm H w x t hist = (L.step x hist).said := by
  obtain ⟨m, hm⟩ := (hS.routed x s hs).1 hist hl
  rw [pm_say t hm, hm]
  rfl

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem pm_at (w : Sampling.Wit) (t : L.Tp x) {i : ℕ} (hi : i < L.M x) {hist : List (Msg × Msg)}
    (h1 : L.start x i ≤ hist.length) (h2 : hist.length < L.start x (i + 1)) :
    L.pm H w x t hist =
      (((A.P (L.wS H w x (L.step x (hist.take (L.start x i))).stmt)).next (L.step x (hist.take (L.start x i))).stmt
        (hist.drop (L.start x i)) (L.slot t ⟨i, hi⟩)).eval H).1 :=
  pm_session t ((hS.routed x s hs).2 i hi hist h1 h2) hi

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **The first `j` sessions read only the first `j` slots.** -/
theorem pm_congr (w : Sampling.Wit) {t t' : L.Tp x} {j : ℕ} (hj : j ≤ L.M x)
    (ht : ∀ i : Fin (L.M x), i.val < j → L.slot t i = L.slot t' i) {hist : List (Msg × Msg)}
    (hl : hist.length < L.start x j) : L.pm H w x t hist = L.pm H w x t' hist := by
  by_cases h0 : hist.length < L.k₀ x
  · rw [pm_framing hS hs w t h0, pm_framing hS hs w t' h0]
  · obtain ⟨i, hi, h1, h2⟩ := L.locate x hist.length j (by omega) hl
    rw [pm_at hS hs w t (by omega) h1 h2, pm_at hS hs w t' (by omega) h1 h2, ht ⟨i, by omega⟩ hi]

end ZkCompiled

/-! ## The hybrid argument -/

/-- `V*`'s answer at `x` and `θ` to the prover messages `as`. -/
abbrev askOf {Θ : Type} (H : Msg → Msg) (Vs : VStar Θ) (x : Msg) (θ : Θ) (as : List Msg) : Out :=
  ((Vs.next x as θ).eval H).1

/-- **One hybrid step**: where resampling one coordinate keeps the sampling design on each side, a bound at each fixed rest of the
coordinates is a bound on average. -/
theorem Pr_le_resample {α β γ : Type} [Fintype α] [Fintype β] [Nonempty β] [Fintype γ] [Nonempty γ]
    (u : α → β → α) (v : α → β) (h1 : ∀ a b, u (u a b) (v a) = a) (h2 : ∀ a b, v (u a b) = b)
    (u' : α → γ → α) (v' : α → γ) (h1' : ∀ a c, u' (u' a c) (v' a) = a) (h2' : ∀ a c, v' (u' a c) = c)
    (P Q : α → Prop) (e : α → ℝ) (h : ∀ a, Pr (fun b => P (u a b)) ≤ Pr (fun c => Q (u' a c)) + e a) :
    Pr P ≤ Pr Q + Ex e := by
  rw [Pr_resample u v h1 h2 P, Pr_resample u' v' h1' h2' Q, ← Ex_add]
  exact Ex_mono h

theorem Pr_le_of_imp {α β : Type} [Fintype α] [Nonempty α] [Fintype β] [Nonempty β] {P : α → Prop} {Q : β → Prop}
    {p : Prop} (h1 : ∀ a, P a → p) (h2 : p → ∀ b, Q b) : Pr P ≤ Pr Q := by
  by_cases hp : p
  · calc Pr P ≤ 1 := Pr_le_one P
      _ = Pr Q := by rw [Pr_congr fun b => iff_true_intro (h2 hp b)]; exact Sampling.Pr_true.symm
  · calc Pr P ≤ Pr (fun _ : α => False) := Pr_mono fun a h => hp (h1 a h)
      _ = 0 := Pr_false
      _ ≤ Pr Q := Pr_nonneg Q

/-- `simRounds` from nothing, where each message is a function of the history, is the run of that function. -/
theorem simRounds_said (H : Msg → Msg) (ask : List Msg → Out) (f : List (Msg × Out) → SP Msg)
    (g : List (Msg × Msg) → Msg) (hf : ∀ ms, f ms = .ret (g (Claims.histOf ms))) (n : ℕ) :
    (SP.eval H ask (Recursion.simRounds f n [])).1 = Claims.interact ask g n [] := by
  have hh : ∀ h : List (Msg × Msg), Claims.histOf (RecDeploy.asRounds h) = h := by
    intro h
    induction h with
    | nil => rfl
    | cons p h ih => exact congrArg (p :: ·) ih
  have hg : (fun h => (SP.eval H ask (f (RecDeploy.asRounds h))).1) = g := by
    funext h
    rw [hf, hh]
    rfl
  rw [RecDeploy.simRounds_interact H ask f n [] (by simp), hg, List.nil_append]
  rfl

theorem finRange_drop {M i : ℕ} (hi : i < M) :
    (List.finRange M).drop i = ⟨i, hi⟩ :: (List.finRange M).drop (i + 1) := by
  rw [List.drop_eq_getElem_cons (l := List.finRange M) (i := i) (by simpa using hi)]
  congr 1
  exact Fin.ext (by simp)

theorem le_of_mem_drop {M k : ℕ} {j : Fin M} (h : j ∈ (List.finRange M).drop k) : k ≤ j.val := by
  obtain ⟨m, _, he⟩ := List.mem_drop_iff_getElem.1 h
  rw [← he]
  simp

namespace ZkCompiled

variable {Coin : Type} {A : ZkArg Coin}

/-- The honest run's messages before session `j`. -/
def pre (L : ZkCompiled Coin A) (H : Msg → Msg) (w : Sampling.Wit) (x : Msg) {Θ : Type} (Vs : VStar Θ) (θ : Θ)
    (t : L.Tp x) (j : ℕ) : List (Msg × Out) :=
  (runStar H (L.prover w x) Vs x t θ).msgs.take (L.start x j)

/-- Session `j` of the honest run shows `os`'s `j`-th outcome. -/
def okAt (L : ZkCompiled Coin A) (H : Msg → Msg) (w : Sampling.Wit) (x : Msg) {Θ : Type} (Vs : VStar Θ) (θ : Θ)
    (os : List A.St) (t : L.Tp x) (j : ℕ) : Prop :=
  os[j]? = some (A.out (L.step x (histOf (L.pre H w x Vs θ t j))).stmt
    ((L.pre H w x Vs θ t (j + 1)).drop (L.start x j)))

/-- The tape `t` with slot `i` replaced by `a`. -/
def upd (L : ZkCompiled Coin A) {x : Msg} (t : L.Tp x) (i : Fin (L.M x)) (a : A.Nonce × A.Rest) : L.Tp x :=
  (Function.update t.1 i a.1, Function.update t.2 i a.2)

/-- The sessions `js` simulated after the transcript `ms` against `V*`, given the outcomes `os`, on the coins `c`. -/
noncomputable def simEval (L : ZkCompiled Coin A) (H : Msg → Msg) (x : Msg) {Θ : Type} (Vs : VStar Θ) (θ : Θ)
    (os : List A.St) (c : (j : Fin (L.M x)) → L.CB x j) (js : List (Fin (L.M x))) (ms : List (Msg × Out)) :
    List (Msg × Out) :=
  (SP.eval H (askOf H Vs x θ) (L.simSessions x os c js ms)).1

/-- A tape and the simulator's coins. -/
abbrev HC (L : ZkCompiled Coin A) (x : Msg) := L.Tp x × ((j : Fin (L.M x)) → L.CB x j)

/-- **Hybrid `i`**: the honest run through its first `i` sessions, each showing its outcome in `os`, then the sessions
from `i` on simulated given their outcomes, on the event `Ev`. -/
noncomputable def hyb (L : ZkCompiled Coin A) (H : Msg → Msg) (w : Sampling.Wit) (x : Msg) {Θ : Type} (Vs : VStar Θ)
    (θ : Θ) (os : List A.St) (Ev : List (Msg × Out) → Prop) (i : ℕ) : ℝ :=
  Pr fun p : L.HC x => (∀ j < i, L.okAt H w x Vs θ os p.1 j) ∧
    Ev (L.simEval H x Vs θ os p.2 ((List.finRange (L.M x)).drop i) (L.pre H w x Vs θ p.1 i))

/-- `err`'s term at session `j` on the tape `t`. -/
noncomputable def errAt (L : ZkCompiled Coin A) (H : Msg → Msg) (w : Sampling.Wit) (x : Msg) (Θ : Type) (Vs : VStar Θ)
    (θ : Θ) (os : List A.St) (t : L.Tp x) (j : ℕ) : ℝ :=
  match os[j]?, decide ((L.pre H w x Vs θ t j).length = L.start x j) && Recursion.goesOn (L.pre H w x Vs θ t j) with
  | some oj, true =>
    A.εzk (L.step x (histOf (L.pre H w x Vs θ t j))).stmt
      ((L.wit w x (L.step x (histOf (L.pre H w x Vs θ t j))).stmt).eval H).1 Θ
      (residual Vs x ((L.pre H w x Vs θ t j).map Prod.fst) (L.n x j)) θ oj
  | _, _ => 0

theorem slot_upd (L : ZkCompiled Coin A) {x : Msg} (t : L.Tp x) (i : Fin (L.M x)) (a : A.Nonce × A.Rest) :
    L.slot (L.upd t i a) i = a := by
  simp [slot, upd]

theorem okAt_of_outOf (L : ZkCompiled Coin A) (H : Msg → Msg) (w : Sampling.Wit) (x : Msg) {Θ : Type} (Vs : VStar Θ)
    (θ : Θ) (t : L.Tp x) {os : List A.St} (hout : L.outOf x (runStar H (L.prover w x) Vs x t θ).msgs = os) {j : ℕ}
    (hj : j < L.M x) : L.okAt H w x Vs θ os t j := by
  unfold okAt pre
  rw [← hout, outOf, List.getElem?_map, List.getElem?_range hj, Option.map_some, List.drop_take, L.start_succ,
    Nat.add_sub_cancel_left]

theorem simSessions_stop (L : ZkCompiled Coin A) (x : Msg) (os : List A.St) (c : (j : Fin (L.M x)) → L.CB x j)
    {ms : List (Msg × Out)} (hg : Recursion.goesOn ms = false) : ∀ js, L.simSessions x os c js ms = .ret ms
  | [] => rfl
  | _ :: _ => by simp only [simSessions, hg]

theorem simSessions_go (L : ZkCompiled Coin A) (x : Msg) (os : List A.St) (c : (j : Fin (L.M x)) → L.CB x j)
    (j : Fin (L.M x)) (js : List (Fin (L.M x))) {ms : List (Msg × Out)} {oj : A.St} (hg : Recursion.goesOn ms = true)
    (ho : os[j.val]? = some oj) :
    L.simSessions x os c (j :: js) ms =
      Recursion.spBind (Recursion.spUnder (ms.map Prod.fst) (simAt A.S (L.CB x j) (L.step x (histOf ms)).stmt oj (c j)))
        fun τ => L.simSessions x os c js (ms ++ τ) := by
  simp only [simSessions, hg, ho]

theorem simSessions_congr (L : ZkCompiled Coin A) (x : Msg) (os : List A.St) (c c' : (j : Fin (L.M x)) → L.CB x j) :
    ∀ (js : List (Fin (L.M x))) (ms : List (Msg × Out)), (∀ j ∈ js, c j = c' j) →
      L.simSessions x os c js ms = L.simSessions x os c' js ms
  | [], _, _ => rfl
  | j :: js, ms, h => by
    cases hg : Recursion.goesOn ms with
    | false => rw [simSessions_stop L x os c hg, simSessions_stop L x os c' hg]
    | true =>
      cases ho : os[j.val]? with
      | none => simp only [simSessions, hg, ho]
      | some oj =>
        rw [simSessions_go L x os c j js hg ho, simSessions_go L x os c' j js hg ho, h j List.mem_cons_self]
        congr 1
        funext τ
        exact simSessions_congr L x os c c' js _ fun i hi => h i (List.mem_cons_of_mem _ hi)

theorem simEval_nil (L : ZkCompiled Coin A) (H : Msg → Msg) (x : Msg) {Θ : Type} (Vs : VStar Θ) (θ : Θ)
    (os : List A.St) (c : (j : Fin (L.M x)) → L.CB x j) (ms : List (Msg × Out)) :
    L.simEval H x Vs θ os c [] ms = ms := rfl

theorem simEval_stop (L : ZkCompiled Coin A) (H : Msg → Msg) (x : Msg) {Θ : Type} (Vs : VStar Θ) (θ : Θ)
    (os : List A.St) (c : (j : Fin (L.M x)) → L.CB x j) (js : List (Fin (L.M x))) {ms : List (Msg × Out)}
    (hg : Recursion.goesOn ms = false) : L.simEval H x Vs θ os c js ms = ms := by
  rw [simEval, simSessions_stop L x os c hg]
  rfl

theorem simEval_go (L : ZkCompiled Coin A) (H : Msg → Msg) (x : Msg) {Θ : Type} (Vs : VStar Θ) (θ : Θ)
    (os : List A.St) (c : (j : Fin (L.M x)) → L.CB x j) (j : Fin (L.M x)) (js : List (Fin (L.M x)))
    {ms : List (Msg × Out)} {oj : A.St} (hg : Recursion.goesOn ms = true) (ho : os[j.val]? = some oj) :
    L.simEval H x Vs θ os c (j :: js) ms =
      L.simEval H x Vs θ os c js (ms ++ (SP.eval H (fun as => askOf H Vs x θ (ms.map Prod.fst ++ as))
        (simAt A.S (L.CB x j) (L.step x (histOf ms)).stmt oj (c j))).1) := by
  rw [simEval, simSessions_go L x os c j js hg ho, ZKMal.recSpBind_eq, ZKMal.spBind_eval, spUnder_eval_fst]
  rfl

theorem simEval_congr (L : ZkCompiled Coin A) (H : Msg → Msg) (x : Msg) {Θ : Type} (Vs : VStar Θ) (θ : Θ)
    (os : List A.St) (c c' : (j : Fin (L.M x)) → L.CB x j) (js : List (Fin (L.M x))) (ms : List (Msg × Out))
    (h : ∀ j ∈ js, c j = c' j) : L.simEval H x Vs θ os c js ms = L.simEval H x Vs θ os c' js ms := by
  unfold simEval
  rw [simSessions_congr L x os c c' js ms h]

/-- The composed simulator's transcript: the framing's rounds, then the sessions simulated after them. -/
theorem sim_out (L : ZkCompiled Coin A) (H : Msg → Msg) (x : Msg) {Θ : Type} (Vs : VStar Θ) (θ : Θ)
    (o : List Msg × List A.St) (c : L.sim.C x) :
    (L.sim.given o).out H Vs θ x c =
      L.simEval H x Vs θ o.2 c (List.finRange (L.M x))
        (Claims.interact (askOf H Vs x θ) (fun h => (L.step x h).said) (L.k₀ x) []) := by
  show (SP.eval H (askOf H Vs x θ) (Recursion.spBind (Recursion.simRounds
    (fun ms => (.ret (L.step x (histOf ms)).said : SP Msg)) (L.k₀ x) [])
      fun ms => L.simSessions x o.2 c (List.finRange (L.M x)) ms)).1 = _
  rw [ZKMal.recSpBind_eq, ZKMal.spBind_eval, simRounds_said H (askOf H Vs x θ)
    (fun ms => (.ret (L.step x (histOf ms)).said : SP Msg)) (fun h => (L.step x h).said)
    (fun ms => by simp only [histOf_eq]) (L.k₀ x)]
  rfl

theorem err_eq (L : ZkCompiled Coin A) (H : Msg → Msg) (x : Msg) (w : Sampling.Wit) (Θ : Type) (Vs : VStar Θ) (θ : Θ)
    (o : List Msg × List A.St) :
    L.err H x w Θ Vs θ o = ∑ j ∈ Finset.range (L.M x), Ex fun t : L.Tp x => L.errAt H w x Θ Vs θ o.2 t j := rfl

theorem errAt_eq (L : ZkCompiled Coin A) (H : Msg → Msg) (w : Sampling.Wit) (x : Msg) {Θ : Type} (Vs : VStar Θ)
    (θ : Θ) (os : List A.St) (t : L.Tp x) (j : ℕ) {oj : A.St} (ho : os[j]? = some oj)
    (hlen : (L.pre H w x Vs θ t j).length = L.start x j) (hg : Recursion.goesOn (L.pre H w x Vs θ t j) = true) :
    L.errAt H w x Θ Vs θ os t j =
      A.εzk (L.step x (histOf (L.pre H w x Vs θ t j))).stmt
        ((L.wit w x (L.step x (histOf (L.pre H w x Vs θ t j))).stmt).eval H).1 Θ
        (residual Vs x ((L.pre H w x Vs θ t j).map Prod.fst) (L.n x j)) θ oj := by
  simp only [errAt, ho, decide_eq_true hlen, hg, Bool.and_self]

section Hybrid

variable {L : ZkCompiled Coin A} {H : Msg → Msg} (hS : L.ZkSessions H) {x : Msg} {s : Sampling.AuditStatement}
  (hs : L.parse x = some s)
include hS hs

/-- Within session `i`, the composed prover is `A`'s at the session's own history. -/
theorem pm_in (w : Sampling.Wit) (t : L.Tp x) {i : ℕ} (hi : i < L.M x) {hist h' : List (Msg × Msg)}
    (hl : hist.length = L.start x i) (hh : h'.length < L.n x i) :
    L.pm H w x t (hist ++ h') =
      (((A.P (L.wS H w x (L.step x hist).stmt)).next (L.step x hist).stmt h' (L.slot t ⟨i, hi⟩)).eval H).1 := by
  rw [pm_at hS hs w t hi (hist := hist ++ h') (by rw [List.length_append, hl]; omega)
    (by rw [List.length_append, hl, L.start_succ]; omega), List.take_left' hl, List.drop_left' hl]

/-- The honest run before session `j` is the composed prover's. -/
theorem pre_eq (w : Sampling.Wit) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = L.V.k x) (t : L.Tp x) {j : ℕ}
    (hj : j ≤ L.M x) :
    L.pre H w x Vs θ t j = Claims.interact (askOf H Vs x θ) (L.pm H w x t) (L.start x j) [] := by
  have hm : (runStar H (L.prover w x) Vs x t θ).msgs =
      Claims.interact (askOf H Vs x θ) (L.pm H w x t) (L.start x (L.M x)) [] := by
    rw [ZKMal.runStar_msgs, hk, hS.rounds x s hs]
    refine ZKMal.interact_congr _ _ _ _ _ fun i hi => hS.honest x s hs w _ t ?_
    rw [List.nil_append, ZKMal.histOf_length, List.length_take]
    exact lt_of_le_of_lt (min_le_left _ _) (lt_of_lt_of_le hi (ZKMal.interact_length_le _ _ _ _))
  unfold pre
  rw [hm]
  exact ZKMal.interact_take _ _ _ (L.start_mono x hj)

/-- Before the sessions, the honest run says what the framing fixes. -/
theorem pre_zero (w : Sampling.Wit) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = L.V.k x) (t : L.Tp x) :
    L.pre H w x Vs θ t 0 = Claims.interact (askOf H Vs x θ) (fun h => (L.step x h).said) (L.k₀ x) [] := by
  rw [pre_eq hS hs w Vs θ hk t (Nat.zero_le _), L.start_zero]
  refine ZKMal.interact_congr _ _ _ _ _ fun i hi => pm_framing hS hs w t ?_
  rw [List.nil_append, ZKMal.histOf_length, List.length_take]
  exact lt_of_le_of_lt (min_le_left _ _) (lt_of_lt_of_le hi (ZKMal.interact_length_le _ _ _ _))

/-- **Session `i` of the honest run**: where the run goes on at the session's start, the session is `A`'s run at the
statement its history routes to, against `residual`, on slot `i`. -/
theorem pre_succ (w : Sampling.Wit) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = L.V.k x) (t : L.Tp x) {i : ℕ}
    (hi : i < L.M x) (hg : Recursion.goesOn (L.pre H w x Vs θ t i) = true) :
    (L.pre H w x Vs θ t i).length = L.start x i ∧
      L.pre H w x Vs θ t (i + 1) = L.pre H w x Vs θ t i ++
        (runStar H (A.P (L.wS H w x (L.step x (histOf (L.pre H w x Vs θ t i))).stmt))
          (residual Vs x ((L.pre H w x Vs θ t i).map Prod.fst) (L.n x i))
          (L.step x (histOf (L.pre H w x Vs θ t i))).stmt (L.slot t ⟨i, hi⟩) θ).msgs := by
  have e := pre_eq hS hs w Vs θ hk t hi.le
  rw [e, recGoesOn_interact] at hg
  have hlen := ZKMal.goesOn_length hg
  rw [e]
  refine ⟨hlen, ?_⟩
  rw [pre_eq hS hs w Vs θ hk t (show i + 1 ≤ L.M x from hi), L.start_succ, ZKMal.interact_add, ZKMal.cont,
    ite_eq_left hg, List.nil_append]
  congr 1
  rw [ZKMal.runStar_msgs, histOf_eq]
  have e3 := ZKMal.interact_shift (askOf H Vs x θ) (L.pm H w x t)
    (Claims.histOf (Claims.interact (askOf H Vs x θ) (L.pm H w x t) (L.start x i) [])) (L.n x i) []
  rw [List.append_nil, ZKMal.histOf_map_fst] at e3
  rw [e3]
  refine ZKMal.interact_congr _ _ _ _ _ fun m hm => ?_
  have hl : m < L.n x i := lt_of_lt_of_le hm (ZKMal.interact_length_le _ _ _ _)
  simp only [List.nil_append]
  exact pm_in hS hs w t hi (by rw [ZKMal.histOf_length]; exact hlen)
    (by rw [ZKMal.histOf_length, List.length_take]; exact lt_of_le_of_lt (min_le_left _ _) hl)

/-- Where the honest run stops before session `i`, it shows nothing more. -/
theorem pre_stop (w : Sampling.Wit) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = L.V.k x) (t : L.Tp x) {i : ℕ}
    (hi : i < L.M x) (hg : Recursion.goesOn (L.pre H w x Vs θ t i) = false) :
    L.pre H w x Vs θ t (i + 1) = L.pre H w x Vs θ t i := by
  have e := pre_eq hS hs w Vs θ hk t hi.le
  rw [e, recGoesOn_interact] at hg
  rw [e, pre_eq hS hs w Vs θ hk t (show i + 1 ≤ L.M x from hi), L.start_succ, ZKMal.interact_add, ZKMal.cont]
  exact ite_eq_right fun hz => by simp [hz] at hg

/-- **The first `j` sessions read only the first `j` slots**: of the honest run. -/
theorem pre_congr (w : Sampling.Wit) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = L.V.k x) (t : L.Tp x)
    (i : Fin (L.M x)) (a : A.Nonce × A.Rest) {j : ℕ} (hj : j ≤ i.val) :
    L.pre H w x Vs θ (L.upd t i a) j = L.pre H w x Vs θ t j := by
  have hjM : j ≤ L.M x := le_trans hj i.isLt.le
  rw [pre_eq hS hs w Vs θ hk (L.upd t i a) hjM, pre_eq hS hs w Vs θ hk t hjM]
  refine ZKMal.interact_congr _ _ _ _ _ fun m hm => ?_
  refine pm_congr hS hs w (j := i.val) i.isLt.le (fun l hl' => ?_) ?_
  · simp only [slot, upd, Function.update_of_ne (Fin.ne_of_val_ne (Nat.ne_of_lt hl'))]
  · rw [List.nil_append, ZKMal.histOf_length, List.length_take]
    exact lt_of_le_of_lt (min_le_left _ _)
      (lt_of_lt_of_le hm (le_trans (ZKMal.interact_length_le _ _ _ _) (L.start_mono x hj)))

theorem pre_all (w : Sampling.Wit) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = L.V.k x) (t : L.Tp x) :
    L.pre H w x Vs θ t (L.M x) = (runStar H (L.prover w x) Vs x t θ).msgs := by
  unfold pre
  refine List.take_of_length_le ?_
  have := ZKMal.runStar_msgs_length_le H (L.prover w x) Vs x t θ
  rw [hk, hS.rounds x s hs] at this
  exact this

/-- `A`'s error is nonnegative wherever `err` reads it: its zero knowledge at the empty event. -/
theorem errAt_nonneg (hA : A.ZK H) (w : Sampling.Wit) {Θ : Type} (Vs : VStar Θ) (θ : Θ)
    (hR : Sampling.Rhon H L.commit L.parse L.partition x w) (os : List A.St) (t : L.Tp x) {j : ℕ} (hj : j < L.M x) :
    0 ≤ L.errAt H w x Θ Vs θ os t j := by
  unfold errAt
  cases ho : os[j]? with
  | none => exact le_rfl
  | some oj =>
    cases hb : (decide ((L.pre H w x Vs θ t j).length = L.start x j) && Recursion.goesOn (L.pre H w x Vs θ t j)) with
    | false => exact le_rfl
    | true =>
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hb
      have hlh : (histOf (L.pre H w x Vs θ t j)).length = L.start x j := by
        rw [histOf_eq, ZKMal.histOf_length, hb.1]
      obtain ⟨hkz, -⟩ := hS.shaped x s hs j hj _ hlh
      have h := hA _ _ (hS.witness x w hR j hj _ hlh) Θ
        (residual Vs x ((L.pre H w x Vs θ t j).map Prod.fst) (L.n x j)) θ hkz.symm oj fun _ => False
      simp only [and_false, Pr_false, zero_add] at h
      exact h

/-- **One hybrid step at fixed other slots**: replacing session `i`'s honest run, on slot `i`'s tape, by `A`'s simulator
given the session's outcome, on slot `i`'s coins, costs `A`'s error where the run reaches the session. -/
theorem step_le (hA : A.ZK H) (w : Sampling.Wit) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = L.V.k x)
    (hR : Sampling.Rhon H L.commit L.parse L.partition x w) (os : List A.St) (Ev : List (Msg × Out) → Prop) {i : ℕ}
    (hi : i < L.M x) (t : L.Tp x) (c : (j : Fin (L.M x)) → L.CB x j) :
    Pr (fun a : A.Nonce × A.Rest => (∀ j < i + 1, L.okAt H w x Vs θ os (L.upd t ⟨i, hi⟩ a) j) ∧
      Ev (L.simEval H x Vs θ os c ((List.finRange (L.M x)).drop (i + 1))
        (L.pre H w x Vs θ (L.upd t ⟨i, hi⟩ a) (i + 1)))) ≤
    Pr (fun b : L.CB x i => (∀ j < i, L.okAt H w x Vs θ os t j) ∧
      Ev (L.simEval H x Vs θ os (Function.update c ⟨i, hi⟩ b) ((List.finRange (L.M x)).drop i)
        (L.pre H w x Vs θ t i))) +
    L.errAt H w x Θ Vs θ os t i := by
  have hpre : ∀ a, L.pre H w x Vs θ (L.upd t ⟨i, hi⟩ a) i = L.pre H w x Vs θ t i := fun a =>
    pre_congr hS hs w Vs θ hk t ⟨i, hi⟩ a (j := i) le_rfl
  have hok : ∀ a, ∀ j < i, (L.okAt H w x Vs θ os (L.upd t ⟨i, hi⟩ a) j ↔ L.okAt H w x Vs θ os t j) := by
    intro a j hj
    unfold okAt
    rw [pre_congr hS hs w Vs θ hk t ⟨i, hi⟩ a (j := j) (by show j ≤ i; omega),
      pre_congr hS hs w Vs θ hk t ⟨i, hi⟩ a (j := j + 1) (by show j + 1 ≤ i; omega)]
  have hnn := errAt_nonneg hS hs hA w Vs θ hR os t hi
  by_cases hprev : ∀ j < i, L.okAt H w x Vs θ os t j
  swap
  · calc _ ≤ Pr (fun _ : A.Nonce × A.Rest => False) :=
          Pr_mono fun a h => hprev fun j hj => (hok a j hj).1 (h.1 j (by omega))
      _ = 0 := Pr_false
      _ ≤ _ := add_nonneg (Pr_nonneg _) hnn
  cases hg : Recursion.goesOn (L.pre H w x Vs θ t i) with
  | false =>
    refine le_trans ?_ (le_add_of_nonneg_right hnn)
    refine Pr_le_of_imp (p := Ev (L.pre H w x Vs θ t i)) (fun a h => ?_) fun hE b => ⟨hprev, ?_⟩
    · have e := pre_stop hS hs w Vs θ hk (L.upd t ⟨i, hi⟩ a) hi (by rw [hpre a]; exact hg)
      rw [e, hpre a, simEval_stop L H x Vs θ os c _ hg] at h
      exact h.2
    · rw [simEval_stop L H x Vs θ os _ _ hg]
      exact hE
  | true =>
    cases ho : os[i]? with
    | none =>
      calc _ ≤ Pr (fun _ : A.Nonce × A.Rest => False) := Pr_mono fun a h => by
              have := h.1 i (Nat.lt_add_one i)
              unfold okAt at this
              rw [ho] at this
              cases this
          _ = 0 := Pr_false
          _ ≤ _ := add_nonneg (Pr_nonneg _) hnn
    | some oi =>
      have hlen := (pre_succ hS hs w Vs θ hk t hi hg).1
      have hlh : (histOf (L.pre H w x Vs θ t i)).length = L.start x i := by
        rw [histOf_eq, ZKMal.histOf_length, hlen]
      obtain ⟨hkz, hCB⟩ := hS.shaped x s hs i hi _ hlh
      rw [errAt_eq L H w x Vs θ os t i ho hlen hg]
      refine le_trans (le_of_eq (Pr_congr fun a => ?_)) (le_trans (hA _ _ (hS.witness x w hR i hi _ hlh) Θ
        (residual Vs x ((L.pre H w x Vs θ t i).map Prod.fst) (L.n x i)) θ hkz.symm oi
        fun τ => Ev (L.simEval H x Vs θ os c ((List.finRange (L.M x)).drop (i + 1)) (L.pre H w x Vs θ t i ++ τ)))
        (le_of_eq ?_))
      · have hga : Recursion.goesOn (L.pre H w x Vs θ (L.upd t ⟨i, hi⟩ a) i) = true := by rw [hpre a]; exact hg
        have hsa := (pre_succ hS hs w Vs θ hk (L.upd t ⟨i, hi⟩ a) hi hga).2
        rw [hpre a, slot_upd] at hsa
        have hoka : L.okAt H w x Vs θ os (L.upd t ⟨i, hi⟩ a) i ↔
            A.out (L.step x (histOf (L.pre H w x Vs θ t i))).stmt
              (runStar H (A.P (L.wS H w x (L.step x (histOf (L.pre H w x Vs θ t i))).stmt))
                (residual Vs x ((L.pre H w x Vs θ t i).map Prod.fst) (L.n x i))
                (L.step x (histOf (L.pre H w x Vs θ t i))).stmt a θ).msgs = oi := by
          unfold okAt
          rw [hsa, hpre a, ho, List.drop_left' hlen]
          exact ⟨fun h => (Option.some.inj h).symm, fun h => by rw [h]⟩
        rw [Nat.forall_lt_succ_right, hoka, hsa]
        exact ⟨fun h => ⟨h.1.2, h.2⟩, fun h => ⟨⟨fun j hj => (hok a j hj).2 (hprev j hj), h.1⟩, h.2⟩⟩
      · congr 1
        rw [← Flock.FnFork.Pr_equiv hCB.some]
        refine Pr_congr fun b => ?_
        rw [finRange_drop hi, simEval_go L H x Vs θ os _ ⟨i, hi⟩ _ hg ho, Function.update_self, simAt_of hCB,
          simEval_congr L H x Vs θ os (Function.update c ⟨i, hi⟩ b) c ((List.finRange (L.M x)).drop (i + 1)) _
            fun j hj => Function.update_of_ne (fun he : j = ⟨i, hi⟩ => by
              have h1 := le_of_mem_drop hj
              have h2 : j.val = i := congrArg Fin.val he
              omega) b c]
        exact ⟨fun h => ⟨hprev, h⟩, fun h => h.2⟩

/-- **Hybrid `i + 1` is within session `i`'s error of hybrid `i`.** -/
theorem hyb_succ (hA : A.ZK H) (w : Sampling.Wit) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = L.V.k x)
    (hR : Sampling.Rhon H L.commit L.parse L.partition x w) (os : List A.St) (Ev : List (Msg × Out) → Prop) {i : ℕ}
    (hi : i < L.M x) :
    L.hyb H w x Vs θ os Ev (i + 1) ≤
      L.hyb H w x Vs θ os Ev i + Ex fun t : L.Tp x => L.errAt H w x Θ Vs θ os t i := by
  unfold hyb
  rw [← SamplingGeneric.Ex_fst (β := (j : Fin (L.M x)) → L.CB x j) fun t : L.Tp x => L.errAt H w x Θ Vs θ os t i]
  refine Pr_le_resample (fun (p : L.HC x) (a : A.Nonce × A.Rest) => (L.upd p.1 ⟨i, hi⟩ a, p.2))
    (fun p => L.slot p.1 ⟨i, hi⟩) (fun p a => by simp [upd, slot]) (fun p a => L.slot_upd p.1 ⟨i, hi⟩ a)
    (fun (p : L.HC x) (b : L.CB x i) => (p.1, Function.update p.2 ⟨i, hi⟩ b)) (fun p => p.2 ⟨i, hi⟩)
    (fun p b => by simp) (fun p b => by simp) _ _ _ fun p => ?_
  exact step_le hS hs hA w Vs θ hk hR os Ev hi p.1 p.2

theorem hyb_le (hA : A.ZK H) (w : Sampling.Wit) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = L.V.k x)
    (hR : Sampling.Rhon H L.commit L.parse L.partition x w) (os : List A.St) (Ev : List (Msg × Out) → Prop) :
    ∀ i ≤ L.M x, L.hyb H w x Vs θ os Ev i ≤
      L.hyb H w x Vs θ os Ev 0 + ∑ j ∈ Finset.range i, Ex fun t : L.Tp x => L.errAt H w x Θ Vs θ os t j := by
  intro i
  induction i with
  | zero => intro _; simp
  | succ i ih =>
    intro hi
    have h1 := hyb_succ hS hs hA w Vs θ hk hR os Ev (Nat.lt_of_succ_le hi)
    have h2 := ih (Nat.le_of_succ_le hi)
    rw [Finset.sum_range_succ]
    linarith

/-- **Hybrid `M` is the honest run**, at the outcome. -/
theorem hyb_M (w : Sampling.Wit) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = L.V.k x)
    (o : List Msg × List A.St) (Ev : List (Msg × Out) → Prop) :
    Pr (fun t : L.Tp x => L.outOf x (runStar H (L.prover w x) Vs x t θ).msgs = o.2 ∧
      Ev (runStar H (L.prover w x) Vs x t θ).msgs) ≤ L.hyb H w x Vs θ o.2 Ev (L.M x) := by
  refine le_trans (le_of_eq (Pr_fst (β := (j : Fin (L.M x)) → L.CB x j) _).symm) ?_
  unfold hyb
  refine Pr_mono fun p h => ⟨fun j hj => okAt_of_outOf L H w x Vs θ p.1 h.1 hj, ?_⟩
  rw [List.drop_of_length_le List.length_finRange.le, pre_all hS hs w Vs θ hk p.1, simEval_nil]
  exact h.2

/-- **Hybrid `0` is the composed simulator.** -/
theorem hyb_zero (w : Sampling.Wit) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = L.V.k x)
    (o : List Msg × List A.St) (Ev : List (Msg × Out) → Prop) :
    L.hyb H w x Vs θ o.2 Ev 0 = Pr (fun c : L.sim.C x => Ev ((L.sim.given o).out H Vs θ x c)) := by
  unfold hyb
  rw [← SamplingGeneric.Pr_snd (α := L.Tp x) (fun c : L.sim.C x => Ev ((L.sim.given o).out H Vs θ x c))]
  refine Pr_congr fun p => ?_
  rw [List.drop_zero, pre_zero hS hs w Vs θ hk p.1, sim_out]
  exact ⟨fun h => h.2, fun h => ⟨fun j hj => absurd hj (by omega), h⟩⟩

end Hybrid

theorem zk_of {L : ZkCompiled Coin A} {H : Msg → Msg} (hA : A.ZK H) (hS : L.ZkSessions H) : L.ZK H := by
  intro l x w hRl Θ Vs θ hk o Ev
  obtain ⟨s, _, hs, -⟩ := hRl.1
  calc _ ≤ Pr (fun t : L.Tp x => L.outOf x (runStar H (L.prover w x) Vs x t θ).msgs = o.2 ∧
          Ev (runStar H (L.prover w x) Vs x t θ).msgs) := Pr_mono fun t h => ⟨congrArg Prod.snd h.1, h.2⟩
    _ ≤ L.hyb H w x Vs θ o.2 Ev (L.M x) := hyb_M hS hs w Vs θ hk o Ev
    _ ≤ L.hyb H w x Vs θ o.2 Ev 0 +
          ∑ j ∈ Finset.range (L.M x), Ex fun t : L.Tp x => L.errAt H w x Θ Vs θ o.2 t j :=
        hyb_le hS hs hA w Vs θ hk hRl.1 o.2 Ev (L.M x) le_rfl
    _ = _ := by rw [hyb_zero hS hs w Vs θ hk o Ev, err_eq]

end ZkCompiled

/-- **The sampled compiler is zero knowledge** (`CompilerZK`). -/
theorem compilerZK : CompilerZK := fun _ _ _ _ _ _ _ hA hS => ZkCompiled.zk_of hA hS

end Claims.SamplingGeneric
