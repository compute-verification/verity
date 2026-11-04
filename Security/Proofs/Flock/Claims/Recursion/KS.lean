import Proofs.Flock.Claims.Recursion.Finders
import Proofs.Flock.Claims.Recursion.Hybrid

/-!
# The recursion: KS

The recursion's `KS` at a general inner protocol `I`, proved in the claims' format from the outer `--zk` Flock's KS at
each session and the inner protocol's KS at the served runs, at the errors that derivation gives: `KSA`, the draft's
`KS` (`Claims.Recursion.KS`) with its `κ` and the weights `γ_A`, `γ_eA`. `E*` is the composed extractor of `Compose`
(`served_eq`), so `servedERep_bad_le_good` splits its failure on an accepted run in two:

* **`E_O` fails at one of the inner stage's draws**, at most `N_I = 1 + m′_I·k_I` times as often as `E_O` fails
  (Wald). `E_O` gives up only when one session's `E^rep` does (`EO_none_session`), so `E_O`'s failure is at most the sum
  of the sessions' (`PrBad_EO_le`), each bounded by the outer Flock's KS at its session (`OuterKS`):
  `κ_O,s + √(γ_O,s·k(x)·Pr[B])`. That is `N_I·Σ_s κ_O,s` and, summed under one root, `√(γ_A·Pr[B])` with
  `γ_A = (N_I·Σ_s √(γ_O,s·k(x)))²`.
* **The inner stage fails though every outer extraction succeeds**: what the inner protocol's KS bounds once it is
  carried to the served runs (`InnerKS`): `κ_I + √(γ_I·Pr[B_I]) + N_I·√(k_I·k(x)·Pr[Be])`. `B_I` is `Be` at the inner
  rounds' forks only (`BI`), so `Pr[B_I] ≤ k(x)·Pr[Be]` (`PrBI_le`, since the inner rounds are the compiled verifier's
  first, `InnerRounds`).

`Be`'s two roots are one, `√(γ_eA·Pr[Be])` with `γ_eA = (√(γ_I·k(x)) + N_I·√(k_I·k(x)))²`: `KS_of`. `KSFlock_of` is
the instance with Flock as the inner protocol. The draft's `γ` is `γ_A` with a `+1` to spare (`γA_le_γ`); its `γ_e`,
`(√(γ_I·k(x)) + N_I + 1)²`, is below `γ_eA` wherever `N_I·√(k_I·k(x)) > N_I + 1`, so the draft's `KS` isn't proved.

**Hypotheses, each a named `Prop`:**

* `OuterKS`: the outer `--zk` Flock's KS at each V* session, the extraction `E_O` makes there (`EOs`) at the compiled
  verifier, in `Claims.KS`'s inequality at that session's numbers. `outerKS_of_flockKS` (`OuterKS`'s module) proves
  it from `Claims.Flock.KS` at the outer deployment and code facts about how the compiled verifier runs each session.
* `InnerKS`: the inner protocol's KS carried to the served runs, with the term the carriage costs (an inner view two
  extractions open differently, at any of the inner `E^rep`'s reruns) in `γ_eA`'s budget. `innerKS_of_KSAbort`
  (`Recursion/InnerAbort.lean`) proves it from the inner protocol's KS against provers that may abort
  (`Inner.KSAbort`) and code facts about the inner session, and `KS_of_flockKS` there is `KS_of` with both. It doesn't
  follow from `Inner.KS`: `E*` sees each inner rerun through `E_O` at fresh outer coins, rejected wherever the compiled
  run rejects there, so the outer sessions' coins thin its inner reruns, and `Inner.KS` bounds `E^rep` only against
  runs whose acceptance is the inner verifier's. `innerKS_gap` (`Recursion/Gap.lean`) is a deployment with every code
  fact and `Inner.KS` but not `Inner.KSAbort`, whose `KS` fails.
* Code facts, which bind `D`'s readers to the compiled verifier: `Refuses` (the compiled verifier accepts only a
  statement that parses with no circuit commitment) and `InnerRounds` (its first rounds are the inner session's).
-/

namespace Claims.Recursion

open Claims

section KS

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-! ## The outer extraction, session by session -/

/-- `E_O`'s output, as the composition reads it: `E_O` gave one. Its failure at a draw is its giving up there. -/
def outR : Rel (List Flock.Wit) := fun _ _ => True

/-- **Outer session `s`'s extraction, as `E_O` makes it**: replay the judged run; if it accepted, Flock's `E^rep` at
the session's V* statement, over forks of the judged run at the session's first round, each rerun seen as the session's
view; the statement, with the witness it recovers. -/
noncomputable def EOs (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) (s : ℕ) : Extractor Coin (Msg × Flock.Wit) D.V where
  C x := Fin (D.nums x s).waits → Fin (D.nums x s).cap → Fin (D.V.k x) → Coin
  ext x r₀ c := .rerun r₀ fun v₀ =>
    if v₀.out = .acc then
      let xs := D.vstar x (v₀.msgs.take (I.V.k (xI D x) + D.kS x))
      (serve (viewing (D.session x s))
          (firstWait (D.nums x s).k (fun v => [DO.cols v])
            (Flock.assemble DO.parse DO.table decO DO.read (xs.getD s [])) (Flock.check DO.parse (xs.getD s []))
            ((List.finRange (D.nums x s).waits).map fun i => (List.finRange (D.nums x s).cap).map fun j =>
              fork (I.V.k (xI D x) + D.kS x + ((xs.take s).map DO.V.k).sum) r₀ (c i j)))).bind
        fun o => .ret (o.map fun w => (xs.getD s [], w))
    else .ret none

/-- Flock's relation at the session's V* statement. -/
def sessR (DO : Flock.Deploy Coin Col T) : Rel (Msg × Flock.Wit) :=
  fun _ p => Flock.R DO.parse p.1 p.2

/-- **The outer `--zk` Flock's KS at each session**: at every outer decoder that meets the list-decoding spec, outer
session `s`'s extraction fails on an accepted run of the compiled verifier with probability at most
`κ_O,s + √(γ_O,s·k(x)·Pr[B])`, the session's numbers at its V* statements (`Deploy.nums`) and `B` the compiled
verifier's (a fork at the session's first round is one of `B`'s `k(x)` guesses). `Claims.KS`'s inequality at each
session, as `Claims.Flock.KS` states it of Flock's verifier; the session's embedding in the compiled verifier's run
carries one to the other. -/
def OuterKS [Zero Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) : Prop :=
  ∀ decO, decOK DO decO → ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (s : Fin (D.nS x)),
    PrBad H D.V (sessR DO) (EOs D I DO decO s) x P ≤
      (D.nums x s).κ + Real.sqrt ((D.nums x s).γ * D.V.k x * PrB H D.V P x)

/-- `collect` gives up only where one of its programs does. -/
theorem collect_none {R α : Type} (H : Msg → Msg) (run : R → Trace) :
    ∀ ps : List (RP R (Option α)), ((collect ps).eval H run).1 = none → ∃ p ∈ ps, (p.eval H run).1 = none
  | [], h => by simp [collect, RP.eval] at h
  | p :: ps, h => by
    simp only [collect] at h
    rw [(RP.eval_bind H run p _).1] at h
    rcases hp : (p.eval H run).1 with _ | a
    · exact ⟨p, List.mem_cons_self, hp⟩
    · rw [hp] at h
      simp only at h
      rw [(RP.eval_bind H run _ _).1] at h
      simp only [RP.eval, Option.map_eq_none_iff] at h
      obtain ⟨q, hq, hq'⟩ := collect_none H run ps h
      exact ⟨q, List.mem_cons_of_mem _ hq, hq'⟩

omit [DecidableEq Col] in
/-- **`E_O` gives up on an accepted run only when one session's extraction does**, at that session's coins. -/
theorem EO_none_session (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω) (r₀ : Fin (D.V.k x) → Coin)
    (c : (EO D I DO decO).C x) (hacc : (run H P D.V x ω r₀).out = .acc)
    (h : ((EO D I DO decO).eval H P x ω r₀ c).1 = none) :
    ∃ s : Fin (D.nS x), ((EOs D I DO decO s).eval H P x ω r₀ (c s)).1 = none := by
  simp only [Extractor.eval, EO, RP.eval, Trace.seen, hacc, ↓reduceIte] at h
  obtain ⟨p, hp, hpn⟩ := collect_none H _ _ h
  obtain ⟨s, _, rfl⟩ := List.mem_map.1 hp
  refine ⟨s, ?_⟩
  simp only [Extractor.eval, EOs, RP.eval, Trace.seen, hacc, ↓reduceIte]
  rw [(RP.eval_bind H _ _ _).1, hpn]
  rfl

omit [DecidableEq Col] in
/-- **`E_O`'s failure is at most the sessions'**, summed. -/
theorem PrBad_EO_le (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] (P : Prover Ω) :
    PrBad H D.V outR (EO D I DO decO) x P ≤
      ∑ s : Fin (D.nS x), PrBad H D.V (sessR DO) (EOs D I DO decO s) x P := by
  classical
  calc PrBad H D.V outR (EO D I DO decO) x P
      ≤ Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × ((s : Fin (D.nS x)) → (EOs D I DO decO s).C x) =>
          ∃ s : Fin (D.nS x), (run H P D.V x t.1 t.2.1).out = .acc ∧
            ((EOs D I DO decO s).eval H P x t.1 t.2.1 (t.2.2 s)).1 = none) := by
        refine Pr_mono fun t h => ?_
        have hn : ((EO D I DO decO).eval H P x t.1 t.2.1 t.2.2).1 = none := by
          rcases ho : ((EO D I DO decO).eval H P x t.1 t.2.1 t.2.2).1 with _ | ws
          · rfl
          · exact absurd (by rw [ho]; trivial) h.2
        obtain ⟨s, hs⟩ := EO_none_session H D I DO decO x P t.1 t.2.1 t.2.2 h.1 hn
        exact ⟨s, h.1, hs⟩
    _ ≤ ∑ s : Fin (D.nS x), Pr (fun t : Ω × (Fin (D.V.k x) → Coin) ×
          ((s : Fin (D.nS x)) → (EOs D I DO decO s).C x) => (run H P D.V x t.1 t.2.1).out = .acc ∧
            ((EOs D I DO decO s).eval H P x t.1 t.2.1 (t.2.2 s)).1 = none) := Pr_exists_le _
    _ ≤ _ := by
        refine Finset.sum_le_sum fun s _ => ?_
        rw [Pr_pi_coord (C := fun s : Fin (D.nS x) => (EOs D I DO decO s).C x) s
          (fun ω r c => (run H P D.V x ω r).out = .acc ∧ ((EOs D I DO decO s).eval H P x ω r c).1 = none)]
        refine Pr_mono fun t h => ⟨h.1, ?_⟩
        rw [h.2]
        exact id

/-! ## The inner rounds' finder -/

/-- **`B_I`, the inner protocol's `B` read through `E_O`**: guess an inner round `j`, fork the compiled run there, run
`E_O` on both runs with fresh coins, and output the first collision among the SHA-512 evaluations the two runs' outer
witnesses record (`Deploy.calls`), as `Be` does at a guess among all the compiled verifier's rounds. -/
noncomputable def BI (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg)
    (hk : 0 < I.V.k (xI D x)) : Finder where
  S := Fin (I.V.k (xI D x)) × Ω × (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x ×
    (EO D I DO decO).C x
  ne := ⟨(⟨0, hk⟩, Classical.arbitrary _, fun _ => default, fun _ => default, Classical.arbitrary _,
    Classical.arbitrary _)⟩
  out s :=
    findColl (D.calls x (((EO D I DO decO).eval H P x s.2.1 s.2.2.1 s.2.2.2.2.1).1.getD []) ++
      D.calls x (((EO D I DO decO).eval H P x s.2.1 (fork s.1 s.2.2.1 s.2.2.2.1) s.2.2.2.2.2).1.getD []))
  cost s :=
    (run H P D.V x s.2.1 s.2.2.1).pLog.length + (run H P D.V x s.2.1 s.2.2.1).vLog.length +
      (run H P D.V x s.2.1 (fork s.1 s.2.2.1 s.2.2.2.1)).pLog.length +
      (run H P D.V x s.2.1 (fork s.1 s.2.2.1 s.2.2.2.1)).vLog.length +
      ((EO D I DO decO).eval H P x s.2.1 s.2.2.1 s.2.2.2.2.1).2.1 +
      ((EO D I DO decO).eval H P x s.2.1 (fork s.1 s.2.2.1 s.2.2.2.1) s.2.2.2.2.2).2.1

/-- `Pr[B_I^{P*}(x) outputs a collision]`, `0` for an inner protocol that takes no rounds at `x_I`. -/
noncomputable def PrBI (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) : ℝ :=
  if hk : 0 < I.V.k (xI D x) then (BI H D I DO decO P x hk).Pr H else 0

/-- **The compiled verifier's first rounds are the inner session's** (a code fact): it takes at least the inner
verifier's rounds at the inner statement. -/
def InnerRounds (D : Deploy Coin) (I : Inner Coin W) : Prop := ∀ x, I.V.k (xI D x) ≤ D.V.k x

/-- The collision `Be` and `B_I` look for, at the fork at round `j`. -/
def forkColl (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} (P : Prover Ω) (x : Msg) (j : ℕ)
    (t : Ω × (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x × (EO D I DO decO).C x) :
    Prop :=
  ∃ u v, findColl (D.calls x (((EO D I DO decO).eval H P x t.1 t.2.1 t.2.2.2.1).1.getD []) ++
      D.calls x (((EO D I DO decO).eval H P x t.1 (fork j t.2.1 t.2.2.1) t.2.2.2.2).1.getD [])) = some (u, v) ∧
    Collision H u v

omit [DecidableEq Col] in
theorem PrBe_nonneg (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) :
    0 ≤ PrBe H D I DO decO P x := by
  unfold PrBe; split
  · exact Pr_nonneg _
  · exact le_rfl

omit [DecidableEq Col] in
theorem PrBe_le_one (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) :
    PrBe H D I DO decO P x ≤ 1 := by
  unfold PrBe; split
  · exact Pr_le_one _
  · exact zero_le_one

omit [DecidableEq Col] in
/-- **`Be` is the average of its forks' collisions over all the compiled verifier's rounds.** -/
theorem PrBe_eq_avg (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg)
    (hk : 0 < D.V.k x) :
    PrBe H D I DO decO P x = (∑ j : Fin (D.V.k x), Pr (forkColl H D I DO decO P x j)) / D.V.k x := by
  simp only [PrBe, hk, ↓reduceDIte, Finder.Pr]
  let e : Fin (D.V.k x) × Ω × (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x ×
      (EO D I DO decO).C x ≃ (Be H D I DO decO P x hk).S :=
    { toFun := fun t => ((t.1, t.2.1, t.2.2.1, t.2.2.2.1), t.2.2.2.2.1, t.2.2.2.2.2)
      invFun := fun s => (s.1.1, s.1.2.1, s.1.2.2.1, s.1.2.2.2, s.2.1, s.2.2)
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  rw [← Pr_equiv e, Pr_guess]
  rfl

omit [DecidableEq Col] in
/-- **`B_I` at most `k(x)` times `Be`**: `B_I` guesses among the inner rounds, which are among the compiled verifier's
`k(x)` (`InnerRounds`), the same forks `Be` makes. -/
theorem PrBI_le (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg)
    (hR : I.V.k (xI D x) ≤ D.V.k x) : PrBI H D I DO decO P x ≤ D.V.k x * PrBe H D I DO decO P x := by
  unfold PrBI
  split
  · rename_i hkI
    have hk : 0 < D.V.k x := lt_of_lt_of_le hkI hR
    have hq : ∀ j, 0 ≤ Pr (forkColl H D I DO decO P x j) := fun j => Pr_nonneg _
    have hBI : (BI H D I DO decO P x hkI).Pr H =
        (∑ j : Fin (I.V.k (xI D x)), Pr (forkColl H D I DO decO P x j)) / I.V.k (xI D x) := by
      unfold Finder.Pr
      exact Pr_guess (fun s : (BI H D I DO decO P x hkI).S => ∃ u v, (BI H D I DO decO P x hkI).out s = some (u, v) ∧
        Collision H u v)
    have hsum : (∑ j : Fin (I.V.k (xI D x)), Pr (forkColl H D I DO decO P x j)) ≤
        ∑ j : Fin (D.V.k x), Pr (forkColl H D I DO decO P x j) := by
      rw [Fin.sum_univ_eq_sum_range (fun j => Pr (forkColl H D I DO decO P x j)),
        Fin.sum_univ_eq_sum_range (fun j => Pr (forkColl H D I DO decO P x j))]
      exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_mono hR) fun j _ _ => hq j
    have hk' : (0 : ℝ) < D.V.k x := by exact_mod_cast hk
    have hkI' : (1 : ℝ) ≤ I.V.k (xI D x) := by exact_mod_cast hkI
    have hS : 0 ≤ ∑ j : Fin (D.V.k x), Pr (forkColl H D I DO decO P x j) := Finset.sum_nonneg fun j _ => hq j
    rw [hBI, PrBe_eq_avg H D I DO decO P x hk, mul_div_cancel₀ _ hk'.ne']
    calc (∑ j : Fin (I.V.k (xI D x)), Pr (forkColl H D I DO decO P x j)) / I.V.k (xI D x)
        ≤ (∑ j : Fin (D.V.k x), Pr (forkColl H D I DO decO P x j)) / I.V.k (xI D x) :=
          div_le_div_of_nonneg_right hsum (by positivity)
      _ ≤ ∑ j : Fin (D.V.k x), Pr (forkColl H D I DO decO P x j) := div_le_self hS hkI'
  · exact mul_nonneg (Nat.cast_nonneg _) (PrBe_nonneg H D I DO decO P x)

/-! ## The inner stage -/

/-- The inner relation at the inner statement. -/
def innerR (D : Deploy Coin) (I : Inner Coin W) : Rel W := fun x w => I.R (xI D x) w

/-- **The compiled verifier refuses what it can't parse, and the private-circuit form** (a code fact): every accepted
run is of a statement that parses with no circuit commitment. -/
def Refuses (H : Msg → Msg) (D : Deploy Coin) : Prop :=
  ∀ (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin), (run H P D.V x ω r).out = .acc →
    ∃ s, D.parse x = some s ∧ s.circ = none

/-- **The inner protocol's KS, carried to the served runs**: at every outer decoder that meets the list-decoding spec,
the inner stage of `E*` (the inner `E^rep` at the inner statement, each rerun served by `E_O`) fails on an accepted run,
when `E_O` succeeds at every draw it reruns at, with probability at most
`κ_I + √(γ_I·Pr[B_I]) + N_I·√(k_I·k(x)·Pr[Be])`: the inner protocol's bound with `B_I` for the inner `B` (the inner
verifier's calls are recorded in the outer witnesses only), and, at each of the inner `E^rep`'s `N_I` expected reruns,
the inner rounds' differing openings, each at most `√(Pr[a differing opening at a fork at that round])`. -/
def InnerKS [Zero Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) : Prop :=
  ∀ decO, decOK DO decO → ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
    PrBadGood H D.V (innerR D I) outR (EO D I DO decO) (dec H D) (fun x => I.waits (xI D x))
        (fun x => I.k (xI D x)) (fun x => I.cap (xI D x)) I.opens (fun x => I.assemble (xI D x))
        (fun x => I.check (xI D x)) x P ≤
      I.κ (xI D x) + Real.sqrt (I.γ (xI D x) * PrBI H D I DO decO P x) +
        NI D I x * Real.sqrt (I.V.k (xI D x) * D.V.k x * PrBe H D I DO decO P x)

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
/-- On an accepted run of a statement the compiled verifier parses with no circuit commitment, the inherited relation
is the inner one at the inner statement. -/
theorem R_of_innerR (D : Deploy Coin) (I : Inner Coin W) {x : Msg} {s : CompiledStatement} (hs : D.parse x = some s)
    (hc : s.circ = none) (o : Option W) (h : (innerR D I).ext x o) : (R D I).ext x o := by
  rcases o with _ | w
  · exact h
  · refine ⟨s, hs, hc, ?_⟩
    have : xI D x = s.inner := by simp [xI, hs]
    rw [← this]
    exact h

/-! ## The recursion's KS -/

/-- **`B`'s weight, as `KS_of` proves it**: `γ_A = (N_I · Σ_s √(γ_O,s · k(x)))²`. -/
noncomputable def γA (D : Deploy Coin) (I : Inner Coin W) (x : Msg) : ℝ :=
  (NI D I x * ∑ s : Fin (D.nS x), Real.sqrt ((D.nums x s).γ * D.V.k x)) ^ 2

/-- **`Be`'s weight, as `KS_of` proves it**: `γ_eA = (√(γ_I(x_I) · k(x)) + N_I · √(k_I(x_I) · k(x)))²`: the inner `B`'s
term, whose collisions are among calls only the outer witnesses record, and the inner rounds' differing openings at
each of the inner `E^rep`'s `N_I` expected reruns. -/
noncomputable def γeA (D : Deploy Coin) (I : Inner Coin W) (x : Msg) : ℝ :=
  (Real.sqrt (I.γ (xI D x) * D.V.k x) + NI D I x * Real.sqrt (I.V.k (xI D x) * D.V.k x)) ^ 2

/-- **The recursion's KS at the errors its proof gives**: the draft's `KS` with `γ_A` and `γ_eA` for `γ` and `γ_e`. -/
def KSA [Zero Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) : Prop :=
  ∀ decO, decOK DO decO → ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
    PrBad H D.V (R D I) (E H D I DO decO) x P ≤
      κ D I x + Real.sqrt (γA D I x * PrB H D.V P x) + Real.sqrt (γeA D I x * PrBe H D I DO decO P x) +
        Real.sqrt (γeA D I x * PrBr H D I DO decO P x)

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
/-- The draft's `γ` is `γ_A`'s with a `+1` to spare. -/
theorem γA_le_γ (D : Deploy Coin) (I : Inner Coin W) (x : Msg) : γA D I x ≤ γ D I x := by
  have h : 0 ≤ (NI D I x : ℝ) * ∑ s : Fin (D.nS x), Real.sqrt ((D.nums x s).γ * D.V.k x) :=
    mul_nonneg (Nat.cast_nonneg _) (Finset.sum_nonneg fun s _ => Real.sqrt_nonneg _)
  unfold γA γ
  nlinarith

theorem Nums.γ_nonneg (n : Nums) : 0 ≤ n.γ := by unfold Nums.γ; positivity

/-- **The recursion's KS**, at a general inner protocol: from the outer `--zk` Flock's KS at each session (`OuterKS`),
the inner protocol's KS at the served runs (`InnerKS`), and the code facts `Refuses` and `InnerRounds`, `E*` fails on
an accepted run with probability at most `κ(x) + √(γ_A(x)·Pr[B]) + √(γ_eA(x)·Pr[Be])`, under `KSA`'s bound with no
`Br` term. -/
theorem KS_of [Zero Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (hRef : Refuses H D) (hRounds : InnerRounds D I) (hO : OuterKS H D I DO) (hI : InnerKS H D I DO) :
    KSA H D I DO := by
  intro decO hdec x Ω _ _ P
  set A : ℝ := (NI D I x : ℝ)
  set p := PrB H D.V P x
  set q := PrBe H D I DO decO P x
  set G := ∑ s : Fin (D.nS x), Real.sqrt ((D.nums x s).γ * D.V.k x)
  have hA : (1 + (I.waits (xI D x) : ℝ) * (I.k (xI D x) : ℝ)) = A := by simp only [A, NI]; push_cast; ring
  have hA0 : 0 ≤ A := Nat.cast_nonneg _
  have hp0 : 0 ≤ p := PrB_nonneg H D.V P x
  have hq0 : 0 ≤ q := PrBe_nonneg H D I DO decO P x
  have hG0 : 0 ≤ G := Finset.sum_nonneg fun s _ => Real.sqrt_nonneg _
  -- the split: `E_O` fails at one of the inner stage's draws, or the inner stage fails
  have hsplit := servedERep_bad_le_good H D.V (R D I) outR (EO D I DO decO) (dec H D) (fun x => I.waits (xI D x))
    (fun x => I.k (xI D x)) (fun x => I.cap (xI D x)) I.opens (fun x => I.assemble (xI D x))
    (fun x => I.check (xI D x)) x P
  rw [hA] at hsplit
  have hgood : PrBadGood H D.V (R D I) outR (EO D I DO decO) (dec H D) (fun x => I.waits (xI D x))
      (fun x => I.k (xI D x)) (fun x => I.cap (xI D x)) I.opens (fun x => I.assemble (xI D x))
      (fun x => I.check (xI D x)) x P ≤
      PrBadGood H D.V (innerR D I) outR (EO D I DO decO) (dec H D) (fun x => I.waits (xI D x))
      (fun x => I.k (xI D x)) (fun x => I.cap (xI D x)) I.opens (fun x => I.assemble (xI D x))
      (fun x => I.check (xI D x)) x P := by
    refine Pr_mono fun t h => ⟨h.1, h.2.1, fun hi => h.2.2 ?_⟩
    obtain ⟨s, hs, hc⟩ := hRef x Ω P t.1 t.2.1 h.1
    exact R_of_innerR D I hs hc _ hi
  have hin := hI decO hdec x Ω P
  -- the outer sessions
  have hEO : PrBad H D.V outR (EO D I DO decO) x P ≤
      ∑ s : Fin (D.nS x), (D.nums x s).κ + G * Real.sqrt p := by
    refine (PrBad_EO_le H D I DO decO x P).trans ?_
    rw [Finset.sum_mul, ← Finset.sum_add_distrib]
    refine Finset.sum_le_sum fun s _ => (hO decO hdec x Ω P s).trans (le_of_eq ?_)
    rw [Real.sqrt_mul (mul_nonneg (Nums.γ_nonneg _) (Nat.cast_nonneg _))]
  -- `B`'s terms under one root
  have hB : A * (G * Real.sqrt p) = Real.sqrt (γA D I x * p) := by
    have hγ : γA D I x = (A * G) ^ 2 := rfl
    rw [hγ, Real.sqrt_mul' _ hp0, Real.sqrt_sq (mul_nonneg hA0 hG0), mul_assoc]
  -- `Be`'s terms under one root
  have hBI : Real.sqrt (I.γ (xI D x) * PrBI H D I DO decO P x) ≤ Real.sqrt (I.γ (xI D x) * D.V.k x * q) := by
    by_cases hγI : 0 ≤ I.γ (xI D x)
    · refine Real.sqrt_le_sqrt ?_
      rw [mul_assoc]
      exact mul_le_mul_of_nonneg_left (PrBI_le H D I DO decO P x (hRounds x)) hγI
    · rw [not_le] at hγI
      have h1 : I.γ (xI D x) * PrBI H D I DO decO P x ≤ 0 := by
        refine mul_nonpos_of_nonpos_of_nonneg hγI.le ?_
        unfold PrBI; split
        · exact Pr_nonneg _
        · exact le_rfl
      rw [Real.sqrt_eq_zero'.2 h1]
      exact Real.sqrt_nonneg _
  have hBe : Real.sqrt (I.γ (xI D x) * D.V.k x * q) + A * Real.sqrt (I.V.k (xI D x) * D.V.k x * q) =
      Real.sqrt (γeA D I x * q) := by
    have hγe : γeA D I x = (Real.sqrt (I.γ (xI D x) * D.V.k x) + A * Real.sqrt (I.V.k (xI D x) * D.V.k x)) ^ 2 :=
      rfl
    rw [hγe, Real.sqrt_mul' (I.γ (xI D x) * D.V.k x) hq0, Real.sqrt_mul' ((I.V.k (xI D x) : ℝ) * D.V.k x) hq0,
      Real.sqrt_mul' (_ ^ 2) hq0, Real.sqrt_sq (add_nonneg (Real.sqrt_nonneg _) (mul_nonneg hA0 (Real.sqrt_nonneg _)))]
    ring
  have hκ : κ D I x = I.κ (xI D x) + A * ∑ s : Fin (D.nS x), (D.nums x s).κ := rfl
  show PrBad H D.V (R D I) (served D.V (EO D I DO decO) (dec H D) (fun x => I.waits (xI D x))
    (fun x => I.k (xI D x)) (fun x => I.cap (xI D x)) I.opens (fun x => I.assemble (xI D x))
    (fun x => I.check (xI D x))) x P ≤ _
  rw [served_eq, hκ]
  have hmul := mul_le_mul_of_nonneg_left hEO hA0
  nlinarith [hsplit, hgood, hin, hmul, hB, hBI, hBe, Real.sqrt_nonneg (γeA D I x * PrBr H D I DO decO P x)]

end KS

/-! ## At Flock -/

section AtFlock

variable {Coin Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
  [DecidableEq Col']

/-- **The recursion's KS over Flock at the errors its proof gives**: `KSA` at every inner decoder that meets the
spec, as `KSFlock` is `KS`, the inner and outer Flock each on its own columns and traces. -/
def KSFlockA [Zero Col] [Zero Col'] (H : Msg → Msg) (D : Deploy Coin) (DI : Flock.Deploy Coin Col T)
    (DO : Flock.Deploy Coin Col' T') : Prop :=
  ∀ decI, decOK DI decI → KSA H D (Inner.ofFlock DI decI) DO

/-- **KS of the recursion over Flock**: `KS_of` with Flock as the inner protocol, at every inner decoder that meets the
spec. -/
theorem KSFlock_of [Zero Col] [Zero Col'] (H : Msg → Msg) (D : Deploy Coin) (DI : Flock.Deploy Coin Col T)
    (DO : Flock.Deploy Coin Col' T') (hRef : Refuses H D)
    (hRounds : ∀ x, DI.V.k (xI D x) ≤ D.V.k x)
    (hO : ∀ decI, decOK DI decI → OuterKS H D (Inner.ofFlock DI decI) DO)
    (hI : ∀ decI, decOK DI decI → InnerKS H D (Inner.ofFlock DI decI) DO) : KSFlockA H D DI DO :=
  fun decI h => KS_of H D _ DO hRef hRounds (hO decI h) (hI decI h)

end AtFlock

end Claims.Recursion
