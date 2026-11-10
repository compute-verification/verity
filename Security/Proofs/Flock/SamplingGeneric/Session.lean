module

public import Definitions.Claims.SamplingGeneric
public import Proofs.Flock.Claims.Sampling.Session
public import Proofs.Flock.Claims.Recursion.Cost
public import Proofs.Flock.Claims.Recursion.InnerAbort
public import Proofs.Flock.SamplingGeneric.Coins
public import Proofs.Flock.SamplingGeneric.Fails
public import Proofs.Flock.SamplingGeneric.Split

/-!
# The sampled compiler: a sampled proof block's session

At an accepted run, a session's extraction (`Compiled.sessWit`, at the session's first sampled block) is `A`'s extractor
at the block's session, each of its reruns the compiled verifier's run with the session's rounds replaced (`place`),
seen as the session's view
(`serve`, `viewing`). Here its success (`Compiled.Found`) is bounded from `A`'s knowledge soundness (`Arg.KS`), its
rounds (`Arg.wf`) and the contract's `runs` and `fixed` (`RunsFixed`, which `Compiled.Sessions` and
`CompiledTwo.Sessions` both give).

* `served_eval`, `found_of`: under `runs`, the served extraction is `A`'s extractor against the session's prover `P'`
  at the session's coins, since the session's coins of the placed coins are the placed ones (`coinsOf_place`, `A`
  having as many rounds at the session's statement as the session) and the coins off the session are the run's
  (`place_of_not_mem`).
* `session_transfer`: **the session transfer**. At coins `R0` that sample `u`, over fresh coins on the session's rounds
  and `A`'s extractor's coins, the run accepts while the block's extraction fails (not `Found`) with probability at
  most `κ_A + √(γ_A·Pr[B_sess])` at the session's statement, with `B_sess` the session's prover's `B` against `A`'s
  verifier: `A.KS` at `P'`.
* `session_fork`, `session_fork_div`: **the session's fork**. `k_A·Pr[B_sess]` is at most the sum over the compiled
  verifier's rounds of its fork's collision probability with the coins off the session fixed: a session round is a
  compiled round through `emb`, and `P'`'s SHA-512 calls are among the compiled verifier's. So
  `Pr[B_sess] ≤ (k_V/k_A)·Pr[B]` at the coins off the session, `Pr[B]` there the compiled verifier's `B` at its round
  guess (`forkCollAt`), which averages to `PrB` (`Ex_forkCollAt`).
* `fiber_bound`: the two at the largest of the proof blocks' error bounds, `κ_s + √(k_V·γ_s·Pr[B])`, since `A`'s
  errors at the session are within the block's bounds (`runs`), so `κ_A ≤ κ_s` and `γ_A/k_A ≤ γ_s`, `k_A` the rounds
  of the block's shape (`Arg.wf`).
* `sessionKS_of`, `sessionKSOf`: **`SessionKS`**, at the slot a selection picks among the sample's, over the second
  stage, `κ_s + √(k_V·γ_s·β(ω, r))` with `β` averaging at most `Pr[B]`: the coins the picked slot gives its block are
  one coordinate of the slots' and then of the blocks' (`Pr_pi_eval_dep`), and resampling the block's session's rounds
  keeps the pick (`fixed`, `Ex_resample`).
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims.Sampling (forkOn LogColl Ex_comp_inj Ex_resample forkOn_empty forkOn_sub B_Pr_logColl PrB_nonneg Pr_unit
  Pr_restrict_pair Pr_equiv Ex_forkOn Ex_finsum Pr_prod_eq_Ex)
open Claims.Recursion (serve viewing)
open Finset

/-! ## Coins -/

/-- **One coordinate of uniform dependent coins is uniform.** -/
theorem Pr_pi_eval_dep {ι : Type} [Fintype ι] [DecidableEq ι] {E : ι → Type} [∀ i, Fintype (E i)]
    [∀ i, Nonempty (E i)] (i : ι) (q : E i → Prop) : Pr (fun c : (j : ι) → E j => q (c i)) = Pr q := by
  rw [← Pr_fst (β := (j : {j // j ≠ i}) → E j) q]
  exact Pr_equiv (Equiv.piSplitAt i E) fun p => q p.1

/-- **Uniform coins restricted along an injective map, beside `A`'s extractor's coins cast along an equation of
shapes, are uniform.** -/
theorem Pr_restrict_shape {Coin : Type} [Fintype Coin] [Nonempty Coin] {A : Arg Coin} {m n : ℕ}
    (emb : Fin m → Fin n) (hemb : Function.Injective emb) {s s' : A.Sh} (hs : s = s')
    (Q : (Fin m → Coin) → A.EC s → Prop) :
    Pr (fun p : (Fin n → Coin) × A.EC s' => Q (fun i => p.1 (emb i)) (cast (congrArg A.EC hs.symm) p.2)) =
      Pr (fun p : (Fin m → Coin) × A.EC s => Q p.1 p.2) := by
  subst hs
  rw [Pr_prod_eq_Ex, Pr_prod_eq_Ex]
  exact Ex_comp_inj emb hemb fun a => Pr (Q a)

/-! ## A verifier's `B`, at given coins -/

section Fork

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- A verifier's `B` collision at the tape `ω` and the judged run's coins `R`, forking on round `j` to `R2`: the SHA-512
calls of the two runs collide. -/
def forkColl (H : Msg → Msg) (V : Verifier Coin) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (j : Fin (V.k x))
    (R R2 : Fin (V.k x) → Coin) : Prop :=
  LogColl ((run H P V x ω R).vLog ++ (run H P V x ω (forkCoins j R R2)).vLog)

/-- **`B`'s collision probability at `(ω, R)`**: over its round guess and the fresh coins after it. -/
noncomputable def forkCollAt (H : Msg → Msg) (V : Verifier Coin) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (R : Fin (V.k x) → Coin) : ℝ :=
  (∑ j, Pr fun R2 => forkColl H V P x ω j R R2) / V.k x

omit [Nonempty Coin] in
theorem forkCollAt_nonneg (H : Msg → Msg) (V : Verifier Coin) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (R : Fin (V.k x) → Coin) : 0 ≤ forkCollAt H V P x ω R :=
  div_nonneg (sum_nonneg fun _ _ => Pr_nonneg _) (Nat.cast_nonneg _)

/-- `B`'s collision probability at `(ω, R)` averages to `Pr[B]`. -/
theorem Ex_forkCollAt (H : Msg → Msg) (V : Verifier Coin) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (x : Msg) : Ex (fun p : Ω × (Fin (V.k x) → Coin) => forkCollAt H V P x p.1 p.2) = PrB H V P x := by
  unfold PrB
  split_ifs with hk
  · rw [B_Pr_logColl, Pr_guess, Ex_prod]
    simp only [forkCollAt, Ex_div, Ex_finsum]
    congr 1
    refine Finset.sum_congr rfl fun j _ => ?_
    symm
    rw [Pr_prod_eq_Ex]
    congr 1
    funext ω
    rw [Pr_prod_eq_Ex]
    rfl
  · have h0 : (V.k x : ℝ) = 0 := by exact_mod_cast Nat.eq_zero_of_not_pos hk
    simp [forkCollAt, h0, Ex]

end Fork

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- The contract's `fixed` and `runs`, which are all the sessions' bound reads of it, so a contract with another sampling
design (`CompiledTwo.Sessions`) gives the bound too. -/
structure RunsFixed (H : Msg → Msg) (L : Compiled Coin A) : Prop where
  fixed : ∀ x r u, u ∈ L.checked x r →
    L.rounds x u ⊆ L.stage2 x ∧ ∀ d, L.checked x (forkOn (L.rounds x u) r d) = L.checked x r
  runs : ∀ x (Ω : Type) (P : Prover Ω) (ω : Ω) r u, u ∈ L.checked x r →
    ∃ (xS : Msg) (P' : Prover Unit), A.shape xS = L.sh x u ∧ A.κ xS ≤ L.κb x u ∧ A.γ xS ≤ L.γb x u ∧
      A.V.k xS = (L.rounds x u).card ∧
      (∀ r' : Fin (L.V.k x) → Coin, (∀ t ∉ L.rounds x u, r' t = r t) →
        (run H P L.V x ω r').out = .acc → (run H P' A.V xS () (L.coinsOf x u xS r')).out = .acc) ∧
      ∀ r' : Fin (L.V.k x) → Coin, (∀ t ∉ L.rounds x u, (∀ t' ∈ L.rounds x u, t < t') → r' t = r t) →
        L.session x u (run H P L.V x ω r').seen = some (xS, (run H P' A.V xS () (L.coinsOf x u xS r')).seen) ∧
          ∀ q ∈ (run H P' A.V xS () (L.coinsOf x u xS r')).vLog, q ∈ (run H P L.V x ω r').vLog

omit [Nonempty Coin] in
theorem Sessions.toRunsFixed {H : Msg → Msg} {L : Compiled Coin A} (hS : L.Sessions H) : L.RunsFixed H :=
  ⟨hS.fixed, hS.runs⟩

/-! ## A session's served reruns -/

omit [Fintype Coin] [Nonempty Coin] in
/-- **The served program is the session's prover's**: a program over `A`'s coins at the session's statement, its reruns
served at coins `R` agreeing with `R0` off the session's rounds and seen as the session's view, returns what it returns
against the session's prover `P'` (`runs`). -/
theorem served_eval (H : Msg → Msg) (L : Compiled Coin A) {x : Msg} {Ω : Type} (P : Prover Ω) (ω : Ω) {u : ℕ}
    {xS : Msg} (P' : Prover Unit) (R0 R : Fin (L.V.k x) → Coin) (hcard : A.V.k xS = (L.rounds x u).card)
    (hview : ∀ r' : Fin (L.V.k x) → Coin, (∀ t ∉ L.rounds x u, (∀ t' ∈ L.rounds x u, t < t') → r' t = R0 t) →
      L.session x u (run H P L.V x ω r').seen = some (xS, (run H P' A.V xS () (L.coinsOf x u xS r')).seen))
    (hR : ∀ t ∉ L.rounds x u, R t = R0 t) {α : Type} (p : RP (Fin (A.V.k xS) → Coin) α) :
    ((serve (fun r' => viewing (L.view x u) (L.place x u xS R r')) p).eval H (run H P L.V x ω)).1 =
      (p.eval H (run H P' A.V xS ())).1 := by
  rw [Claims.Recursion.serve_eq, (RP.serve_eval H _ _ p).1]
  refine Claims.Recursion.RP.eval_congr H _ _ p fun r' _ => ?_
  rw [servedRun_seen]
  show L.view x u (run H P L.V x ω (L.place x u xS R r')).seen = (run H P' A.V xS () r').seen
  unfold view
  rw [hview _ fun t ht _ => (L.place_of_not_mem R r' ht).trans (hR t ht), L.coinsOf_place hcard]
  rfl

/-- **The block is found where `A`'s extractor finds a witness** against the session's prover, at the session's coins
of `R` and the block's coins cast to the session's shape. -/
theorem found_of (H : Msg → Msg) (L : Compiled Coin A) {x : Msg} {Ω : Type} (P : Prover Ω) (ω : Ω)
    {u : Fin (L.nPU x)} {xS : Msg} (P' : Prover Unit) (R0 : Fin (L.V.k x) → Coin) (hshape : A.shape xS = L.sh x u)
    (hcard : A.V.k xS = (L.rounds x u).card)
    (hview : ∀ r' : Fin (L.V.k x) → Coin, (∀ t ∉ L.rounds x u, (∀ t' ∈ L.rounds x u, t < t') → r' t = R0 t) →
      L.session x u (run H P L.V x ω r').seen = some (xS, (run H P' A.V xS () (L.coinsOf x u xS r')).seen))
    {R : Fin (L.V.k x) → Coin} (hR : ∀ t ∉ L.rounds x u, R t = R0 t) (c : A.EC (L.sh x u))
    (hext : A.R.ext xS (A.E.eval H P' xS () (L.coinsOf x u xS R) (cast (congrArg A.EC hshape.symm) c)).1) :
    L.Found H x (run H P L.V x ω) u R c := by
  refine ⟨xS, _, hview R fun t ht _ => hR t ht, hshape, ?_⟩
  rw [L.served_eval H P ω P' R0 R hcard hview hR]
  change A.R.ext xS ((A.ext xS (L.coinsOf x u xS R) (cast (congrArg A.EC hshape.symm) c)).eval H
    (run H P' A.V xS ())).1 at hext
  revert hext
  cases ((A.ext xS (L.coinsOf x u xS R) (cast (congrArg A.EC hshape.symm) c)).eval H (run H P' A.V xS ())).1 with
  | none => exact False.elim
  | some w => exact fun h => ⟨w, rfl, h⟩

omit [Fintype Coin] [Nonempty Coin] in
/-- **The session's coins of coins fresh on the session's rounds are the fresh ones**, read through `emb`. -/
theorem coinsOf_forkOn (L : Compiled Coin A) {x : Msg} {u : ℕ} {xS : Msg} (hcard : A.V.k xS = (L.rounds x u).card)
    (R0 σ : Fin (L.V.k x) → Coin) :
    L.coinsOf x u xS (forkOn (L.rounds x u) R0 σ) = fun i => σ (L.emb x u xS hcard i) := by
  funext i
  have hm : L.emb x u xS hcard i ∈ L.rounds x u := Finset.orderEmbOfFin_mem _ _ _
  simp only [Compiled.coinsOf, dite_eq_left hcard, forkOn, ite_eq_left hm]

/-! ## The session transfer and the session's fork -/

/-- **The session transfer**: at coins `R0` that sample `u`, with `runs`'s session statement `xS` and prover `P'`, over
fresh coins on the session's rounds and `A`'s extractor's coins, the run accepts while the block's extraction fails
(not `Found`) with probability at most `κ_A + √(γ_A·Pr[B_sess])` at the session's statement: the run accepts exactly when
`P'`'s does on the session's coins, its extraction succeeds where `A`'s extractor against `P'` does (`found_of`), and
the session's coins and the cast extractor's coins are uniform (`Pr_restrict_shape`), so this is `A.KS` at `P'`. -/
theorem session_transfer {H : Msg → Msg} (hKS : A.KS H) (L : Compiled Coin A) (x : Msg) {Ω : Type} (P : Prover Ω)
    (ω : Ω) (R0 : Fin (L.V.k x) → Coin) (u : Fin (L.nPU x)) (xS : Msg) (P' : Prover Unit)
    (hshape : A.shape xS = L.sh x u)
    (hcard : A.V.k xS = (L.rounds x u).card)
    (hacc : ∀ r' : Fin (L.V.k x) → Coin, (∀ t ∉ L.rounds x u, r' t = R0 t) →
      (run H P L.V x ω r').out = .acc → (run H P' A.V xS () (L.coinsOf x u xS r')).out = .acc)
    (hview : ∀ r' : Fin (L.V.k x) → Coin, (∀ t ∉ L.rounds x u, (∀ t' ∈ L.rounds x u, t < t') → r' t = R0 t) →
      L.session x u (run H P L.V x ω r').seen = some (xS, (run H P' A.V xS () (L.coinsOf x u xS r')).seen)) :
    Pr (fun s : (Fin (L.V.k x) → Coin) × A.EC (L.sh x u) =>
        (run H P L.V x ω (forkOn (L.rounds x u) R0 s.1)).out = .acc ∧
          ¬ L.Found H x (run H P L.V x ω) u (forkOn (L.rounds x u) R0 s.1) s.2) ≤
      A.κ xS + Real.sqrt (A.γ xS * PrB H A.V P' xS) := by
  have hag0 : ∀ σ : Fin (L.V.k x) → Coin, ∀ t ∉ L.rounds x u, forkOn (L.rounds x u) R0 σ t = R0 t :=
    fun σ t ht => by simp [forkOn, ht]
  let Q : (Fin (A.V.k xS) → Coin) → A.EC (A.shape xS) → Prop := fun ρ c =>
    (run H P' A.V xS () ρ).out = .acc ∧ ¬ A.R.ext xS (A.E.eval H P' xS () ρ c).1
  have himp : ∀ s : (Fin (L.V.k x) → Coin) × A.EC (L.sh x u),
      (run H P L.V x ω (forkOn (L.rounds x u) R0 s.1)).out = .acc ∧
          ¬ L.Found H x (run H P L.V x ω) u (forkOn (L.rounds x u) R0 s.1) s.2 →
        Q (fun i => s.1 (L.emb x u xS hcard i)) (cast (congrArg A.EC hshape.symm) s.2) := by
    rintro s ⟨h1, h2⟩
    have ha := hacc _ (hag0 s.1) h1
    rw [L.coinsOf_forkOn hcard] at ha
    refine ⟨ha, fun hR => h2 (L.found_of H P ω P' R0 hshape hcard hview (hag0 s.1) s.2 ?_)⟩
    rw [L.coinsOf_forkOn hcard]
    exact hR
  calc _ ≤ Pr (fun s : (Fin (L.V.k x) → Coin) × A.EC (L.sh x u) =>
          Q (fun i => s.1 (L.emb x u xS hcard i)) (cast (congrArg A.EC hshape.symm) s.2)) := Pr_mono himp
    _ = Pr (fun p : (Fin (A.V.k xS) → Coin) × A.EC (A.shape xS) => Q p.1 p.2) :=
        Pr_restrict_shape (L.emb x u xS hcard) (L.emb x u xS hcard).injective hshape Q
    _ = PrBad H A.V A.R A.E xS P' := by
        unfold PrBad
        exact (Pr_unit _).symm
    _ ≤ A.κ xS + Real.sqrt (A.γ xS * PrB H A.V P' xS) := hKS xS Unit P'

/-- **The session's fork**: `k_A·Pr[B_sess]`, the session's prover's `B` against `A`'s verifier, is at most the sum over
the compiled verifier's rounds of its fork's collision probability, the coins off the session's rounds those of `R0`: a
fork of `P'` at round `j` is the compiled verifier's fork at round `emb j` (`emb` strictly monotone), and `P'`'s SHA-512
calls are among the compiled verifier's (`runs`). -/
theorem session_fork (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω)
    (R0 : Fin (L.V.k x) → Coin) (u : ℕ) (xS : Msg) (P' : Prover Unit) (hcard : A.V.k xS = (L.rounds x u).card)
    (hlog : ∀ r' : Fin (L.V.k x) → Coin, (∀ t ∉ L.rounds x u, (∀ t' ∈ L.rounds x u, t < t') → r' t = R0 t) →
      ∀ q ∈ (run H P' A.V xS () (L.coinsOf x u xS r')).vLog, q ∈ (run H P L.V x ω r').vLog) :
    (A.V.k xS : ℝ) * PrB H A.V P' xS ≤
      Ex fun σ => ∑ j, Pr fun R2 => forkColl H L.V P x ω j (forkOn (L.rounds x u) R0 σ) R2 := by
  have hinj := (L.emb x u xS hcard).injective
  have hS : ∀ i, L.emb x u xS hcard i ∈ L.rounds x u := fun i => Finset.orderEmbOfFin_mem _ _ _
  have hag0 : ∀ σ : Fin (L.V.k x) → Coin, ∀ t ∉ L.rounds x u, forkOn (L.rounds x u) R0 σ t = R0 t :=
    fun σ t ht => by simp [forkOn, ht]
  by_cases hkS : 0 < A.V.k xS
  · rw [show PrB H A.V P' xS = (B H A.V P' xS hkS).Pr H from dite_eq_left hkS, B_Pr_logColl, Pr_guess,
      mul_div_cancel₀ _ (by positivity : (A.V.k xS : ℝ) ≠ 0), Ex_finsum]
    have hterm : ∀ jS : Fin (A.V.k xS),
        Pr (fun b : Unit × (Fin (A.V.k xS) → Coin) × (Fin (A.V.k xS) → Coin) =>
          LogColl ((run H P' A.V xS b.1 b.2.1).vLog ++ (run H P' A.V xS b.1 (forkCoins jS b.2.1 b.2.2)).vLog)) ≤
        Ex fun σ => Pr fun R2 => forkColl H L.V P x ω (L.emb x u xS hcard jS) (forkOn (L.rounds x u) R0 σ) R2 := by
      intro jS
      rw [Pr_unit (fun b : (Fin (A.V.k xS) → Coin) × (Fin (A.V.k xS) → Coin) =>
          LogColl ((run H P' A.V xS () b.1).vLog ++ (run H P' A.V xS () (forkCoins jS b.1 b.2)).vLog)),
        ← Pr_restrict_pair (L.emb x u xS hcard) hinj (fun ρ1 ρ2 =>
          LogColl ((run H P' A.V xS () ρ1).vLog ++ (run H P' A.V xS () (forkCoins jS ρ1 ρ2)).vLog)),
        ← Pr_prod_eq_Ex (fun s : (Fin (L.V.k x) → Coin) × (Fin (L.V.k x) → Coin) =>
          forkColl H L.V P x ω (L.emb x u xS hcard jS) (forkOn (L.rounds x u) R0 s.1) s.2)]
      refine Pr_mono fun s hcol => ?_
      have hlt : ∀ i, (L.emb x u xS hcard i).val < (L.emb x u xS hcard jS).val ↔ i.val < jS.val := fun i =>
        (L.emb x u xS hcard).strictMono.lt_iff_lt (a := i) (b := jS)
      have e1 : (fun i => s.1 (L.emb x u xS hcard i)) = L.coinsOf x u xS (forkOn (L.rounds x u) R0 s.1) :=
        (L.coinsOf_forkOn hcard R0 s.1).symm
      have e2 : forkCoins jS (fun i => s.1 (L.emb x u xS hcard i)) (fun i => s.2 (L.emb x u xS hcard i)) =
          L.coinsOf x u xS (forkCoins (L.emb x u xS hcard jS) (forkOn (L.rounds x u) R0 s.1) s.2) := by
        funext i
        simp only [Compiled.coinsOf, dite_eq_left hcard]
        by_cases h : i.val < jS.val <;> simp [forkCoins, forkOn, hS i, h, hlt]
      rw [e2, e1] at hcol
      have v1 := hlog (forkOn (L.rounds x u) R0 s.1) fun t ht _ => hag0 s.1 t ht
      have v2 := hlog (forkCoins (L.emb x u xS hcard jS) (forkOn (L.rounds x u) R0 s.1) s.2) fun t ht hb => by
        have : t.val < (L.emb x u xS hcard jS).val := hb _ (hS jS)
        simp [forkCoins, this, hag0 s.1 t ht]
      exact hcol.mono fun q hq => (List.mem_append.1 hq).elim (fun h => List.mem_append_left _ (v1 q h))
        fun h => List.mem_append_right _ (v2 q h)
    refine (Finset.sum_le_sum fun jS _ => hterm jS).trans ?_
    rw [← Finset.sum_image (s := Finset.univ) (g := ⇑(L.emb x u xS hcard))
      (f := fun j => Ex fun σ => Pr fun R2 => forkColl H L.V P x ω j (forkOn (L.rounds x u) R0 σ) R2)
      (fun a _ b _ h => hinj h)]
    exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _) fun j _ _ => Ex_nonneg fun _ => Pr_nonneg _
  · rw [show PrB H A.V P' xS = 0 from dite_eq_right hkS, mul_zero]
    exact Ex_nonneg fun _ => sum_nonneg fun _ _ => Pr_nonneg _

/-- **The session's fork, as a bound on `Pr[B_sess]`**: `Pr[B_sess] ≤ (k_V/k_A)·Pr[B]`, with `Pr[B]` the compiled
verifier's `B` at its round guess, averaged over the session's coins with the coins off them those of `R0`. -/
theorem session_fork_div (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω)
    (R0 : Fin (L.V.k x) → Coin) (u : ℕ) (xS : Msg) (P' : Prover Unit) (hcard : A.V.k xS = (L.rounds x u).card)
    (hlog : ∀ r' : Fin (L.V.k x) → Coin, (∀ t ∉ L.rounds x u, (∀ t' ∈ L.rounds x u, t < t') → r' t = R0 t) →
      ∀ q ∈ (run H P' A.V xS () (L.coinsOf x u xS r')).vLog, q ∈ (run H P L.V x ω r').vLog) :
    PrB H A.V P' xS ≤
      (L.V.k x / A.V.k xS : ℝ) * Ex fun σ => forkCollAt H L.V P x ω (forkOn (L.rounds x u) R0 σ) := by
  have hF := L.session_fork H x P ω R0 u xS P' hcard hlog
  have hEx : (Ex fun σ => forkCollAt H L.V P x ω (forkOn (L.rounds x u) R0 σ)) =
      (Ex fun σ => ∑ j, Pr fun R2 => forkColl H L.V P x ω j (forkOn (L.rounds x u) R0 σ) R2) / L.V.k x := by
    simp only [forkCollAt, Ex_div]
  rw [hEx]
  by_cases hk : 0 < A.V.k xS
  · have hkle : A.V.k xS ≤ L.V.k x := by
      simpa using Fintype.card_le_of_injective _ (L.emb x u xS hcard).injective
    have hkS : (0 : ℝ) < A.V.k xS := by exact_mod_cast hk
    have hkV0 : (L.V.k x : ℝ) ≠ 0 := by
      have := lt_of_lt_of_le hk hkle
      positivity
    calc PrB H A.V P' xS = A.V.k xS * PrB H A.V P' xS / A.V.k xS := by
          rw [mul_comm, mul_div_assoc, div_self hkS.ne', mul_one]
      _ ≤ (Ex fun σ => ∑ j, Pr fun R2 => forkColl H L.V P x ω j (forkOn (L.rounds x u) R0 σ) R2) / A.V.k xS :=
          div_le_div_of_nonneg_right hF hkS.le
      _ = (L.V.k x / A.V.k xS : ℝ) * ((Ex fun σ => ∑ j, Pr fun R2 =>
            forkColl H L.V P x ω j (forkOn (L.rounds x u) R0 σ) R2) / L.V.k x) := by
          rw [div_mul_div_comm, mul_comm (L.V.k x : ℝ), mul_div_mul_right _ _ hkV0]
  · have h0 : (A.V.k xS : ℝ) = 0 := by exact_mod_cast Nat.eq_zero_of_not_pos hk
    exact (show PrB H A.V P' xS = 0 from dite_eq_right hk).trans_le
      (le_of_eq (by rw [h0, div_zero, zero_mul]))

/-- **At one sampled proof block's session, the others' coins fixed**: the session transfer and the session's fork at
`runs`'s session, with `A`'s errors there within the block's bounds (`runs`), so at most the largest of the proof
blocks' bounds: `κ_A ≤ κb ≤ κ_s`, and `γ_A·Pr[B_sess] ≤ γ_s·k_A·Pr[B_sess] ≤ k_V·γ_s·Pr[B]`, `k_A` the rounds of the
block's shape (`Arg.wf`). -/
theorem fiber_bound {H : Msg → Msg} (hKS : A.KS H) {L : Compiled Coin A} (hS : L.RunsFixed H) (x : Msg)
    {Ω : Type} (P : Prover Ω) (ω : Ω) (R0 : Fin (L.V.k x) → Coin) (u : Fin (L.nPU x))
    (hu : u.val ∈ L.checked x R0) :
    Pr (fun s : (Fin (L.V.k x) → Coin) × A.EC (L.sh x u) =>
        (run H P L.V x ω (forkOn (L.rounds x u) R0 s.1)).out = .acc ∧
          ¬ L.Found H x (run H P L.V x ω) u (forkOn (L.rounds x u) R0 s.1) s.2) ≤
      L.κs x + Real.sqrt ((L.V.k x : ℝ) * L.γs x *
        Ex fun σ => forkCollAt H L.V P x ω (forkOn (L.rounds x u) R0 σ)) := by
  obtain ⟨xS, P', hshape, hκb, hγb, hcard, hacc, hview⟩ := hS.runs x Ω P ω R0 u hu
  have hT := L.session_transfer hKS x P ω R0 u xS P' hshape hcard hacc fun r' h => (hview r' h).1
  have hF := L.session_fork H x P ω R0 u xS P' hcard fun r' h => (hview r' h).2
  have hκ := hκb.trans (L.κ_le_κs x u.isLt)
  have hkS : A.V.k xS = A.rounds (L.sh x u) := (A.wf xS).trans (congrArg A.rounds hshape)
  have hkle : A.V.k xS ≤ L.V.k x := by
    simpa using Fintype.card_le_of_injective _ (L.emb x u xS hcard).injective
  have hγs0 := L.γs_nonneg x
  have hY : 0 ≤ Ex fun σ => ∑ j, Pr fun R2 => forkColl H L.V P x ω j (forkOn (L.rounds x u) R0 σ) R2 :=
    Ex_nonneg fun _ => sum_nonneg fun _ _ => Pr_nonneg _
  have hEx : (Ex fun σ => forkCollAt H L.V P x ω (forkOn (L.rounds x u) R0 σ)) =
      (Ex fun σ => ∑ j, Pr fun R2 => forkColl H L.V P x ω j (forkOn (L.rounds x u) R0 σ) R2) / L.V.k x := by
    simp only [forkCollAt, Ex_div]
  have hγ : A.γ xS * PrB H A.V P' xS ≤
      (L.V.k x : ℝ) * L.γs x * Ex fun σ => forkCollAt H L.V P x ω (forkOn (L.rounds x u) R0 σ) := by
    rw [hEx]
    by_cases hk : 0 < A.V.k xS
    · have hkS0 : (A.V.k xS : ℝ) ≠ 0 := by positivity
      have hkV0 : (L.V.k x : ℝ) ≠ 0 := by
        have := lt_of_lt_of_le hk hkle
        positivity
      have hg : A.γ xS / A.V.k xS ≤ L.γs x := by
        refine (div_le_div_of_nonneg_right hγb (Nat.cast_nonneg _)).trans ?_
        rw [hkS]
        exact L.γ_le_γs x u.isLt
      have hP := PrB_nonneg H A.V P' xS
      calc A.γ xS * PrB H A.V P' xS
          = A.γ xS / A.V.k xS * (A.V.k xS * PrB H A.V P' xS) := by
            rw [← mul_assoc, div_mul_cancel₀ _ hkS0]
        _ ≤ L.γs x * (A.V.k xS * PrB H A.V P' xS) :=
            mul_le_mul_of_nonneg_right hg (mul_nonneg (Nat.cast_nonneg _) hP)
        _ ≤ L.γs x * Ex fun σ => ∑ j, Pr fun R2 => forkColl H L.V P x ω j (forkOn (L.rounds x u) R0 σ) R2 :=
            mul_le_mul_of_nonneg_left hF hγs0
        _ = (L.V.k x : ℝ) * L.γs x * ((Ex fun σ => ∑ j, Pr fun R2 =>
              forkColl H L.V P x ω j (forkOn (L.rounds x u) R0 σ) R2) / L.V.k x) := by
            rw [mul_comm (L.V.k x : ℝ), mul_assoc, mul_div_cancel₀ _ hkV0]
    · rw [show PrB H A.V P' xS = 0 from dite_eq_right hk, mul_zero]
      exact mul_nonneg (mul_nonneg (Nat.cast_nonneg _) hγs0) (div_nonneg hY (Nat.cast_nonneg _))
  exact hT.trans (add_le_add hκ (Real.sqrt_le_sqrt hγ))

/-! ## The selected slot's session, resampled -/

/-- The slot a selection picks among the sample's at `R`, with the proof block there, when it holds one below
`n_PU`. -/
def selSlot (L : Compiled Coin A) (x : Msg) (sel : List ℕ → Option (Fin (L.nPU x))) (R : Fin (L.V.k x) → Coin) :
    Option (Fin (L.nPU x) × Fin (L.nPU x)) :=
  (sel (L.checked x R)).bind fun i =>
    ((L.checked x R)[i.val]?).bind fun u => if hu : u < L.nPU x then some (i, ⟨u, hu⟩) else none

/-- Its block's session's rounds, none when no slot is picked. -/
def selRounds (L : Compiled Coin A) (x : Msg) (sel : List ℕ → Option (Fin (L.nPU x))) (R : Fin (L.V.k x) → Coin) :
    Finset (Fin (L.V.k x)) :=
  match L.selSlot x sel R with
  | some iu => L.rounds x iu.2
  | none => ∅

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem selSlot_eq_some {L : Compiled Coin A} {x : Msg} {sel : List ℕ → Option (Fin (L.nPU x))}
    {R : Fin (L.V.k x) → Coin} {i u : Fin (L.nPU x)} :
    L.selSlot x sel R = some (i, u) ↔ sel (L.checked x R) = some i ∧ (L.checked x R)[i.val]? = some u.val := by
  unfold selSlot
  rw [Option.bind_eq_some_iff]
  constructor
  · rintro ⟨j, hj, h⟩
    rw [Option.bind_eq_some_iff] at h
    obtain ⟨v, hv, h⟩ := h
    by_cases hlt : v < L.nPU x
    · rw [dite_eq_left hlt, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨hj, hv⟩
    · rw [dite_eq_right hlt] at h
      cases h
  · rintro ⟨hs, hm⟩
    refine ⟨i, hs, ?_⟩
    rw [hm, Option.bind_some, dite_eq_left u.isLt]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem mem_of_selSlot {L : Compiled Coin A} {x : Msg} {sel : List ℕ → Option (Fin (L.nPU x))}
    {R : Fin (L.V.k x) → Coin} {i u : Fin (L.nPU x)} (h : L.selSlot x sel R = some (i, u)) :
    u.val ∈ L.checked x R :=
  List.mem_of_getElem? (selSlot_eq_some.1 h).2

omit [Fintype Coin] [Nonempty Coin] in
theorem selSlot_forkOn {H : Msg → Msg} {L : Compiled Coin A} (hS : L.RunsFixed H) {x : Msg}
    {sel : List ℕ → Option (Fin (L.nPU x))} {R : Fin (L.V.k x) → Coin} {i u : Fin (L.nPU x)}
    (h : L.selSlot x sel R = some (i, u)) (σ : Fin (L.V.k x) → Coin) :
    L.selSlot x sel (forkOn (L.rounds x u) R σ) = some (i, u) := by
  obtain ⟨hs, hm⟩ := selSlot_eq_some.1 h
  rw [selSlot_eq_some, (hS.fixed x R u (mem_of_selSlot h)).2 σ]
  exact ⟨hs, hm⟩

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem selRounds_some {L : Compiled Coin A} {x : Msg} {sel : List ℕ → Option (Fin (L.nPU x))}
    {R : Fin (L.V.k x) → Coin} {i u : Fin (L.nPU x)} (h : L.selSlot x sel R = some (i, u)) :
    L.selRounds x sel R = L.rounds x u := by
  unfold selRounds
  rw [h]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem selRounds_none {L : Compiled Coin A} {x : Msg} {sel : List ℕ → Option (Fin (L.nPU x))}
    {R : Fin (L.V.k x) → Coin} (h : L.selSlot x sel R = none) : L.selRounds x sel R = ∅ := by
  unfold selRounds
  rw [h]

omit [Fintype Coin] [Nonempty Coin] in
/-- Resampling the picked slot's block's session's rounds keeps them (`fixed`). -/
theorem selRounds_resample {H : Msg → Msg} {L : Compiled Coin A} (hS : L.RunsFixed H) {x : Msg}
    (sel : List ℕ → Option (Fin (L.nPU x))) (r d σ : Fin (L.V.k x) → Coin) :
    L.selRounds x sel (forkOn (L.stage2 x) r (forkOn (L.selRounds x sel (forkOn (L.stage2 x) r d)) d σ)) =
      L.selRounds x sel (forkOn (L.stage2 x) r d) := by
  cases h : L.selSlot x sel (forkOn (L.stage2 x) r d) with
  | none => rw [selRounds_none h, forkOn_empty, selRounds_none h]
  | some iu =>
    obtain ⟨i, u⟩ := iu
    rw [selRounds_some h, forkOn_sub (hS.fixed x _ u (mem_of_selSlot h)).1]
    exact selRounds_some (selSlot_forkOn hS h σ)

/-- The probability over the slots' coins that the run at `R` accepts and the block at the slot a selection picks fails
at the slot's coins. -/
noncomputable def selMiss (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (sel : List ℕ → Option (Fin (L.nPU x))) (R : Fin (L.V.k x) → Coin) : ℝ :=
  Pr fun c : L.SlotEC x => (run H P L.V x ω R).out = .acc ∧
    ∃ i u, L.selSlot x sel R = some (i, u) ∧ ¬ L.Found H x (run H P L.V x ω) u R (c i u)

theorem selMiss_none (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (sel : List ℕ → Option (Fin (L.nPU x))) {R : Fin (L.V.k x) → Coin} (h : L.selSlot x sel R = none) :
    L.selMiss H P x ω sel R = 0 := by
  unfold selMiss
  rw [h]
  exact (Pr_congr fun _ => ⟨fun ⟨_, _, _, hu, _⟩ => Option.some_ne_none _ hu.symm, False.elim⟩).trans Pr_false

/-- **At the picked slot's block's session's coins**, the others fixed: the coins the slot gives its block are one
coordinate of the slots' and then of the blocks' (`Pr_pi_eval_dep`), and `fiber_bound`. -/
theorem selMiss_fiber {H : Msg → Msg} (hKS : A.KS H) {L : Compiled Coin A} (hS : L.RunsFixed H)
    (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω) (sel : List ℕ → Option (Fin (L.nPU x)))
    {R0 : Fin (L.V.k x) → Coin} {i u : Fin (L.nPU x)} (h : L.selSlot x sel R0 = some (i, u)) :
    (Ex fun σ => L.selMiss H P x ω sel (forkOn (L.rounds x u) R0 σ)) ≤
      L.κs x + Real.sqrt ((L.V.k x : ℝ) * L.γs x *
        Ex fun σ => forkCollAt H L.V P x ω (forkOn (L.rounds x u) R0 σ)) := by
  have e : ∀ σ, L.selMiss H P x ω sel (forkOn (L.rounds x u) R0 σ) =
      Pr fun c : A.EC (L.sh x u) => (run H P L.V x ω (forkOn (L.rounds x u) R0 σ)).out = .acc ∧
        ¬ L.Found H x (run H P L.V x ω) u (forkOn (L.rounds x u) R0 σ) c := by
    intro σ
    let Q : A.EC (L.sh x u) → Prop := fun c => (run H P L.V x ω (forkOn (L.rounds x u) R0 σ)).out = .acc ∧
      ¬ L.Found H x (run H P L.V x ω) u (forkOn (L.rounds x u) R0 σ) c
    unfold selMiss
    rw [selSlot_forkOn hS h σ]
    refine (Pr_congr fun c => ⟨fun ⟨ha, i', u', hiu, hw⟩ => ?_, fun ⟨ha, hw⟩ => ⟨ha, i, u, rfl, hw⟩⟩).trans
      ((Pr_pi_eval_dep (E := fun _ : Fin (L.nPU x) => L.BlockEC x) i fun b : L.BlockEC x => Q (b u)).trans
        (Pr_pi_eval_dep (E := fun v : Fin (L.nPU x) => A.EC (L.sh x v)) u Q))
    rw [Option.some.injEq, Prod.mk.injEq] at hiu
    obtain ⟨rfl, rfl⟩ := hiu
    exact ⟨ha, hw⟩
  simp only [e]
  rw [← Pr_prod_eq_Ex (fun s : (Fin (L.V.k x) → Coin) × A.EC (L.sh x u) =>
    (run H P L.V x ω (forkOn (L.rounds x u) R0 s.1)).out = .acc ∧
      ¬ L.Found H x (run H P L.V x ω) u (forkOn (L.rounds x u) R0 s.1) s.2)]
  exact fiber_bound hKS hS x P ω R0 u (mem_of_selSlot h)

/-- At second-stage coins `d`, resampling the picked block's session's rounds: `selMiss_fiber`, or `0` when none is
picked. -/
theorem resample_bound {H : Msg → Msg} (hKS : A.KS H) {L : Compiled Coin A} (hS : L.RunsFixed H)
    (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω) (sel : List ℕ → Option (Fin (L.nPU x)))
    (r d : Fin (L.V.k x) → Coin) :
    (Ex fun σ => L.selMiss H P x ω sel
        (forkOn (L.stage2 x) r (forkOn (L.selRounds x sel (forkOn (L.stage2 x) r d)) d σ))) ≤
      L.κs x + Real.sqrt ((L.V.k x : ℝ) * L.γs x * Ex fun σ => forkCollAt H L.V P x ω
        (forkOn (L.stage2 x) r (forkOn (L.selRounds x sel (forkOn (L.stage2 x) r d)) d σ))) := by
  cases h : L.selSlot x sel (forkOn (L.stage2 x) r d) with
  | none =>
    rw [selRounds_none h]
    simp only [forkOn_empty]
    rw [selMiss_none H L P x ω sel h, Ex_const (α := Fin (L.V.k x) → Coin) (0 : ℝ)]
    exact add_nonneg (L.κs_nonneg x) (Real.sqrt_nonneg _)
  | some iu =>
    obtain ⟨i, u⟩ := iu
    rw [selRounds_some h]
    simp only [forkOn_sub (hS.fixed x _ u (mem_of_selSlot h)).1]
    exact selMiss_fiber hKS hS x P ω sel h

/-- **`SessionKS` from `A`'s knowledge soundness**, its rounds and the contract: `β(ω, r)` is `B`'s collision
probability averaged over the second stage, which averages to `Pr[B]` (`Ex_forkCollAt`); at `(ω, r)`, resampling the
picked slot's block's session's rounds (`Ex_resample`, `selRounds_resample`) gives `resample_bound` at each
second-stage `d`, averaged by Jensen's inequality (`Ex_sqrt_le`). -/
theorem sessionKS_of {H : Msg → Msg} (hKS : A.KS H) {L : Compiled Coin A} (hS : L.RunsFixed H) :
    L.SessionKS H := by
  intro x Ω _ _ P
  refine ⟨fun ω r => Ex fun d => forkCollAt H L.V P x ω (forkOn (L.stage2 x) r d),
    fun ω r => Ex_nonneg fun d => forkCollAt_nonneg H L.V P x ω _, le_of_eq ?_, fun ω r sel => ?_⟩
  · rw [Ex_prod, ← Ex_forkCollAt H L.V P x, Ex_prod]
    congr 1
    funext ω
    exact (Ex_prod (fun p : (Fin (L.V.k x) → Coin) × (Fin (L.V.k x) → Coin) =>
      forkCollAt H L.V P x ω (forkOn (L.stage2 x) p.1 p.2))).symm.trans (Ex_forkOn _ _)
  have hres := Ex_resample (fun d => L.selRounds x sel (forkOn (L.stage2 x) r d)) (selRounds_resample hS sel r)
    fun d => forkCollAt H L.V P x ω (forkOn (L.stage2 x) r d)
  have hres' := Ex_resample (fun d => L.selRounds x sel (forkOn (L.stage2 x) r d)) (selRounds_resample hS sel r)
    fun d => L.selMiss H P x ω sel (forkOn (L.stage2 x) r d)
  beta_reduce at hres hres'
  have step1 : Pr (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
      (run H P L.V x ω (forkOn (L.stage2 x) r p.1)).out = .acc ∧
        ∃ i, sel (L.checked x (forkOn (L.stage2 x) r p.1)) = some i ∧
          ∃ u : Fin (L.nPU x), (L.checked x (forkOn (L.stage2 x) r p.1))[i.val]? = some u.val ∧
            ¬ L.Found H x (run H P L.V x ω) u (forkOn (L.stage2 x) r p.1) (p.2 i u)) ≤
      Ex fun d => L.selMiss H P x ω sel (forkOn (L.stage2 x) r d) := by
    rw [Pr_prod_eq_Ex]
    exact Ex_mono fun d => Pr_mono fun c ⟨ha, i, hs, u, hm, hn⟩ => ⟨ha, i, u, selSlot_eq_some.2 ⟨hs, hm⟩, hn⟩
  calc _ ≤ Ex fun d => L.selMiss H P x ω sel (forkOn (L.stage2 x) r d) := step1
    _ = Ex fun d => Ex fun σ => L.selMiss H P x ω sel
          (forkOn (L.stage2 x) r (forkOn (L.selRounds x sel (forkOn (L.stage2 x) r d)) d σ)) := hres'.symm
    _ ≤ Ex fun d => L.κs x + Real.sqrt ((L.V.k x : ℝ) * L.γs x * Ex fun σ => forkCollAt H L.V P x ω
          (forkOn (L.stage2 x) r (forkOn (L.selRounds x sel (forkOn (L.stage2 x) r d)) d σ))) :=
        Ex_mono fun d => resample_bound hKS hS x P ω sel r d
    _ = L.κs x + Ex fun d => Real.sqrt ((L.V.k x : ℝ) * L.γs x * Ex fun σ => forkCollAt H L.V P x ω
          (forkOn (L.stage2 x) r (forkOn (L.selRounds x sel (forkOn (L.stage2 x) r d)) d σ))) := by
        rw [Ex_add, Ex_const]
    _ ≤ L.κs x + Real.sqrt (Ex fun d => (L.V.k x : ℝ) * L.γs x * Ex fun σ => forkCollAt H L.V P x ω
          (forkOn (L.stage2 x) r (forkOn (L.selRounds x sel (forkOn (L.stage2 x) r d)) d σ))) :=
        add_le_add le_rfl (Ex_sqrt_le _ fun d => mul_nonneg (mul_nonneg (Nat.cast_nonneg _) (L.γs_nonneg x))
          (Ex_nonneg fun σ => forkCollAt_nonneg H L.V P x ω _))
    _ = L.κs x + Real.sqrt ((L.V.k x : ℝ) * L.γs x *
          Ex fun d => forkCollAt H L.V P x ω (forkOn (L.stage2 x) r d)) := by
        rw [Ex_mul_left, hres]

end Compiled

/-- **The sessions' KS from the proof system's**: `Compiled.sessionKS_of` at every compiled sampled verifier. -/
theorem sessionKSOf : SessionKSOf := fun _ _ _ _ _ _ _ hKS hS => Compiled.sessionKS_of hKS hS.toRunsFixed

end Claims.SamplingGeneric
