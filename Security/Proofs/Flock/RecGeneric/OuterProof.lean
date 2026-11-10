module

public import Proofs.Flock.RecGeneric.StatementProof
public import Proofs.Flock.RecGeneric.Outer
public import Proofs.Flock.Claims.Recursion.KS
public import Proofs.Flock.Claims.Recursion.InnerAccepts
public import Proofs.Flock.Claims.Recursion.InnerAbort
public import Proofs.Flock.Claims.Recursion.Gap
public import Proofs.Flock.Claims.Recursion.Hybrid
public import Proofs.Flock.Claims.Recursion.CostExp
public import Proofs.Flock.Claims.Recursion.Cost
public import Proofs.Flock.Claims.Aggregate

/-!
# O1 and L2, proved

`outerSessionsKS` proves `OuterSessionsKS` (O1): `E_O` meets `Claims.KS` at `κ_O = Σ_s κb_s`,
`γ_O = (Σ_s √(γb_s·k(x)))²` wherever `A`'s counted extractor is knowledge sound and the compiler meets
`Compiled.Sessions`.

* `E_O` fails only where some session's extraction fails on the accepted run (`Compiled.eo_bad_session`: `collect`
  gives none at a session that gives none, and witnesses outside `R_O` come from a session whose witness is outside
  `A`'s relation), so its failure is at most the sum over sessions (`Compiled.PrBad_le_sum`).
* A session's extraction, on the forks of a sample at the session's first round, is `A`'s extractor against the session
  prover after the sample's earlier messages (`Compiled.sessionProg_eval`, from `Runs`), at the statement the sample's
  staging rounds fix, whose shape is the session's (`Shapes`); so it fails at most as `A`'s does there, within the
  session's bounds (`Bounded`; `Compiled.prefix_bad_le`, the fork's coins and `E_O`'s coins read uniformly).
* Averaged over prefixes, Jensen moves the average under the root; the session's round guesses are among the compiled
  verifier's `k(x)` (`Compiled.Ex_sPrB_le`), so session `s` costs `κb_s + √(γb_s·k(x)·Pr[B])`
  (`Compiled.sessKS`), and the sum of the roots is `√(γ_O·Pr[B])`.

`compilerKS` is L2, `Definitions.Claims.RecursionGeneric`'s `CompilerKS`: L1 (`compositionKS`) at
`compile(Π_in, A, S)` with O1 for its `E_O`.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion
open Claims.SamplingGeneric (Arg)

theorem forall₂_exists_left {α β : Type} {R : α → β → Prop} {l₁ : List α} {l₂ : List β}
    (h : List.Forall₂ R l₁ l₂) : ∀ b ∈ l₂, ∃ a ∈ l₁, R a b := by
  induction h with
  | nil => intro b hb; simp at hb
  | cons hab _ ih =>
    intro b hb
    rcases List.mem_cons.1 hb with rfl | hb
    · exact ⟨_, List.mem_cons_self, hab⟩
    · obtain ⟨a', ha', h'⟩ := ih b hb
      exact ⟨a', List.mem_cons_of_mem _ ha', h'⟩

theorem rel_ext_some {W : Type} {R : Rel W} {x : Msg} {o : Option W} (h : R.ext x o) : ∃ w, o = some w ∧ R x w := by
  cases o with
  | none => exact False.elim h
  | some w => exact ⟨w, rfl, h⟩

namespace Compiled

variable {Coin : Type} [Inhabited Coin] {A : Arg Coin}

/-! ## One session at a sample -/

/-- Session `s`'s staged statement in the run at coins `r`. -/
def stmtAt (L : Compiled Coin A) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (s : ℕ)
    (r : Fin (L.V.k x) → Coin) : Msg :=
  (L.stmts x (run H P L.V x ω r).msgs).getD s []

/-- Session `s`'s prover at the coins `r`: `P` after the run's messages before the session's first round. -/
def proverAt (L : Compiled Coin A) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (s : ℕ)
    (r : Fin (L.V.k x) → Coin) : Prover Unit :=
  sessProver P x ω ((run H P L.V x ω r).msgs.take (L.off x s))

/-! ## Offsets and coins -/

omit [Inhabited Coin] in
theorem off_succ (L : Compiled Coin A) (x : Msg) (s : ℕ) : L.off x (s + 1) = L.off x s + A.rounds (L.sh x s) := by
  simp only [off, List.range_succ, List.map_append, List.sum_append, List.map_cons, List.map_nil, List.sum_cons,
    List.sum_nil]
  omega

omit [Inhabited Coin] in
theorem off_mono (L : Compiled Coin A) (x : Msg) {s t : ℕ} (h : s ≤ t) : L.off x s ≤ L.off x t := by
  induction h with
  | refl => exact le_rfl
  | step _ ih => rw [L.off_succ x]; omega

omit [Inhabited Coin] in
theorem off_add_le (L : Compiled Coin A) (hF : L.Fits) (x : Msg) (s : Fin (L.nS x)) :
    L.off x s + A.rounds (L.sh x s) ≤ L.V.k x := by
  rw [← L.off_succ x s]
  exact (L.off_mono x (Nat.succ_le_of_lt s.isLt)).trans (hF x)

omit [Inhabited Coin] in
theorem pre_le_off (L : Compiled Coin A) (x : Msg) (s : ℕ) : L.pre x ≤ L.off x s := Nat.le_add_right _ _

theorem sessCoins_placeAt {k n : ℕ} (o : ℕ) (h : o + n ≤ k) (r : Fin n → Coin) :
    sessCoins o (placeAt (Coin := Coin) (k := k) o r) n = r := by
  funext i
  have h1 : o + i.val < k := by omega
  simp [sessCoins, placeAt, coinSeq, h1]

/-! ## A fork at a session's first round keeps its statement and prover -/

theorem stmtAt_fork (L : Compiled Coin A) (H : Msg → Msg) (hSt : L.Stages) {Ω : Type} (P : Prover Ω) (x : Msg)
    (ω : Ω) (s : ℕ) (r d : Fin (L.V.k x) → Coin) :
    L.stmtAt H P x ω s (fork (L.off x s) r d) = L.stmtAt H P x ω s r := by
  unfold stmtAt
  rw [hSt x (run H P L.V x ω (fork (L.off x s) r d)).msgs, hSt x (run H P L.V x ω r).msgs,
    Claims.Recursion.run_msgs_take H P L.V x ω (L.pre x) (r := fork (L.off x s) r d) (r' := r)
      fun i hi => (fork_agree (L.off x s) r d i (lt_of_lt_of_le hi (L.pre_le_off x s))).symm]

theorem proverAt_fork (L : Compiled Coin A) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (s : ℕ)
    (r d : Fin (L.V.k x) → Coin) : L.proverAt H P x ω s (fork (L.off x s) r d) = L.proverAt H P x ω s r := by
  unfold proverAt
  rw [Claims.Recursion.run_msgs_take H P L.V x ω (L.off x s) (r := fork (L.off x s) r d) (r' := r)
    fun i hi => (fork_agree (L.off x s) r d i hi).symm]

/-- At a prefix some continuation of which accepts, the session's staged statement has the session's shape. -/
theorem shape_stmtAt (L : Compiled Coin A) (H : Msg → Msg) (hS : L.Sessions H) {Ω : Type} (P : Prover Ω) (x : Msg)
    (ω : Ω) (s : Fin (L.nS x)) (r d₁ : Fin (L.V.k x) → Coin)
    (hacc : (run H P L.V x ω (fork (L.off x s) r d₁)).out = .acc) :
    A.shape (L.stmtAt H P x ω s r) = L.sh x s := by
  have h : A.shape (L.stmtAt H P x ω s (fork (L.off x s) r d₁)) = L.sh x s := hS.shapes x Ω P ω _ hacc s
  rwa [L.stmtAt_fork H hS.stages] at h

/-- `Runs` at a prefix some continuation of which accepts: every fork at the session's first round shows the
session prover's run, and `A`'s verifier's calls in it are the compiled verifier's. -/
theorem runs_fork (L : Compiled Coin A) (H : Msg → Msg) (hS : L.Sessions H) {Ω : Type} (P : Prover Ω) (x : Msg)
    (ω : Ω) (s : Fin (L.nS x)) (r d₁ d : Fin (L.V.k x) → Coin)
    (hacc : (run H P L.V x ω (fork (L.off x s) r d₁)).out = .acc) :
    (run H (L.proverAt H P x ω s r) A.V (L.stmtAt H P x ω s r) () (sessCoins (L.off x s) d _)).seen =
        L.session x s (run H P L.V x ω (fork (L.off x s) r d)).seen ∧
      ∀ e ∈ (run H (L.proverAt H P x ω s r) A.V (L.stmtAt H P x ω s r) () (sessCoins (L.off x s) d _)).vLog,
        e ∈ (run H P L.V x ω (fork (L.off x s) r d)).vLog := by
  have h : (run H (L.proverAt H P x ω s (fork (L.off x s) r d₁)) A.V
        (L.stmtAt H P x ω s (fork (L.off x s) r d₁)) () (sessCoins (L.off x s) d _)).seen =
        L.session x s (run H P L.V x ω (fork (L.off x s) (fork (L.off x s) r d₁) d)).seen ∧
      ∀ e ∈ (run H (L.proverAt H P x ω s (fork (L.off x s) r d₁)) A.V
          (L.stmtAt H P x ω s (fork (L.off x s) r d₁)) () (sessCoins (L.off x s) d _)).vLog,
        e ∈ (run H P L.V x ω (fork (L.off x s) (fork (L.off x s) r d₁) d)).vLog :=
    hS.runs x s Ω P ω _ d hacc
  rwa [L.proverAt_fork H, L.stmtAt_fork H hS.stages, fork_fork] at h

/-! ## Session `s`'s extraction is `A`'s at the session prover -/

variable [Fintype Coin] [Nonempty Coin]

/-- **Session `s`'s extraction fails** on the judged run `r₀`: the run accepted, and the session gives no witness in
`A`'s relation at its staged statement. -/
abbrev SessBad (L : Compiled Coin A) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (s : Fin (L.nS x)) (ω : Ω)
    (r₀ : Fin (L.V.k x) → Coin) (c : A.EC (L.sh x s)) : Prop :=
  (run H P L.V x ω r₀).out = .acc ∧
    ¬ ∃ p, ((L.sessionProg x r₀ s (L.stmtAt H P x ω s r₀) c).eval H (run H P L.V x ω)).1 = some p ∧ A.R p.1 p.2

/-- **On an accepted fork at the session's first round, session `s`'s extraction is `A`'s extractor** at the
session prover and statement, on the fork's coins from that round on, its witness paired with the statement. -/
theorem sessionProg_eval (L : Compiled Coin A) (H : Msg → Msg) (hS : L.Sessions H) {Ω : Type}
    (P : Prover Ω) (x : Msg) (ω : Ω) (s : Fin (L.nS x)) (r d : Fin (L.V.k x) → Coin) (c : A.EC (L.sh x s))
    (hacc : (run H P L.V x ω (fork (L.off x s) r d)).out = .acc)
    (hsh : A.shape (L.stmtAt H P x ω s r) = L.sh x s) :
    ((L.sessionProg x (fork (L.off x s) r d) s (L.stmtAt H P x ω s (fork (L.off x s) r d)) c).eval H
        (run H P L.V x ω)).1 =
      ((A.ext (L.stmtAt H P x ω s r) (sessCoins (L.off x s) d _) (cast (congrArg A.EC hsh.symm) c)).eval H
        (run H (L.proverAt H P x ω s r) A.V (L.stmtAt H P x ω s r) ())).1.map
          fun w => (L.stmtAt H P x ω s r, w) := by
  have hle : L.off x s + A.V.k (L.stmtAt H P x ω s r) ≤ L.V.k x := by
    rw [A.wf, hsh]
    exact L.off_add_le hS.fits x s
  rw [L.stmtAt_fork H hS.stages]
  unfold sessionProg
  rw [dite_eq_left hsh, RP.eval_bind_ret, serve_eq, (RP.serve_eval H _ _ _).1, sessCoins_fork]
  dsimp only
  refine congrArg (Option.map _) ?_
  refine RP.eval_congr H _ _ _ fun r' _ => ?_
  have h := (L.runs_fork H hS P x ω s r d (placeAt (L.off x s) r') hacc).1
  rw [sessCoins_placeAt _ hle] at h
  show L.session x s (run H P L.V x ω (fork (L.off x s) (fork (L.off x s) r d) (placeAt (L.off x s) r'))).seen = _
  rw [fork_fork]
  exact h.symm

/-- **Session `s`'s failure at a fork is `A`'s at the session prover**: on an accepted fork at the session's first
round where session `s`'s extraction fails, the session prover's run at the fork's coins from that round on accepts,
and `A`'s extractor at the session's statement fails on `E_O`'s coins for the session. -/
theorem bad_session (L : Compiled Coin A) (H : Msg → Msg) (hS : L.Sessions H) {Ω : Type}
    (P : Prover Ω) (x : Msg) (ω : Ω) (s : Fin (L.nS x)) (r d : Fin (L.V.k x) → Coin) (c : A.EC (L.sh x s))
    (hsh : A.shape (L.stmtAt H P x ω s r) = L.sh x s) (h : L.SessBad H P x s ω (fork (L.off x s) r d) c) :
    (run H (L.proverAt H P x ω s r) A.V (L.stmtAt H P x ω s r) () (sessCoins (L.off x s) d _)).out = .acc ∧
      ¬ A.R.ext (L.stmtAt H P x ω s r)
        (A.E.eval H (L.proverAt H P x ω s r) (L.stmtAt H P x ω s r) () (sessCoins (L.off x s) d _)
          (cast (congrArg A.EC hsh.symm) c)).1 := by
  obtain ⟨hacc, hbad⟩ := h
  have hacc' : (run H (L.proverAt H P x ω s r) A.V (L.stmtAt H P x ω s r) ()
      (sessCoins (L.off x s) d _)).out = .acc :=
    (congrArg Seen.out (L.runs_fork H hS P x ω s r d d hacc).1).trans
      (hS.accepts x s (run H P L.V x ω (fork (L.off x s) r d)).seen s.isLt hacc)
  refine ⟨hacc', fun hF => hbad ?_⟩
  obtain ⟨w, hw, hR⟩ := rel_ext_some hF
  rw [L.sessionProg_eval H hS P x ω s r d c hacc hsh]
  have hw' : ((A.ext (L.stmtAt H P x ω s r) (sessCoins (L.off x s) d _) (cast (congrArg A.EC hsh.symm) c)).eval H
      (run H (L.proverAt H P x ω s r) A.V (L.stmtAt H P x ω s r) ())).1 = some w := hw
  exact ⟨(L.stmtAt H P x ω s r, w), Option.map_eq_some_iff.2 ⟨w, hw', rfl⟩, hR⟩

/-! ## `E_O` fails only where a session does -/

/-- **`E_O` fails only where some session's extraction fails**: `collect` gives none only at a session that gives
none, and its witnesses fall outside `R_O` only at a session whose witness is outside `A`'s relation. -/
theorem eo_bad_session (L : Compiled Coin A) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r₀ : Fin (L.V.k x) → Coin) (c : L.EO.C x)
    (h : (run H P L.V x ω r₀).out = .acc ∧ ¬ L.toComp.RO.ext x (L.EO.eval H P x ω r₀ c).1) :
    ∃ s, L.SessBad H P x s ω r₀ (c s) := by
  obtain ⟨hacc, hbad⟩ := h
  have he : (L.EO.eval H P x ω r₀ c).1 = ((collect ((List.finRange (L.nS x)).map fun s =>
      L.sessionProg x r₀ s (L.stmtAt H P x ω s r₀) (c s))).eval H (run H P L.V x ω)).1 := by
    simp only [Extractor.eval, EO, RP.eval, Transcript.seen, hacc, ↓reduceIte]
    rfl
  rcases ho : (L.EO.eval H P x ω r₀ c).1 with _ | ws
  · rw [he] at ho
    obtain ⟨p, hp, hn⟩ := collect_none H _ _ ho
    obtain ⟨s, -, rfl⟩ := List.mem_map.1 hp
    exact ⟨s, hacc, fun ⟨q, hq, _⟩ => absurd (hq.symm.trans hn) (Option.some_ne_none q)⟩
  · rw [ho] at hbad
    rw [he] at ho
    have hb : ¬ ∀ p ∈ ws, A.R p.1 p.2 := hbad
    push Not at hb
    obtain ⟨q, hq, hR⟩ := hb
    obtain ⟨p, hp, hpq⟩ := forall₂_exists_left (collect_some H _ _ ws ho) q hq
    obtain ⟨s, -, rfl⟩ := List.mem_map.1 hp
    refine ⟨s, hacc, fun ⟨q', hq', hR'⟩ => hR ?_⟩
    obtain rfl : q = q' := Option.some.inj (hpq.symm.trans hq')
    exact hR'

/-- **`E_O`'s failure is at most the sum of its sessions'**, each over its own coins (`E_O`'s coins at the
session, read uniformly). -/
theorem PrBad_le_sum (L : Compiled Coin A) (H : Msg → Msg) {Ω : Type} [Fintype Ω] (P : Prover Ω) (x : Msg) :
    PrBad H L.V L.toComp.RO L.toComp.EO x P ≤ ∑ s : Fin (L.nS x),
      Pr (fun t : Ω × (Fin (L.V.k x) → Coin) × A.EC (L.sh x s) => L.SessBad H P x s t.1 t.2.1 t.2.2) := by
  calc PrBad H L.V L.toComp.RO L.toComp.EO x P
      ≤ Pr (fun t : Ω × (Fin (L.V.k x) → Coin) × L.EO.C x => ∃ s, L.SessBad H P x s t.1 t.2.1 (t.2.2 s)) :=
        Pr_mono fun t ht => L.eo_bad_session H P x t.1 t.2.1 t.2.2 ht
    _ ≤ ∑ s, Pr (fun t : Ω × (Fin (L.V.k x) → Coin) × L.EO.C x => L.SessBad H P x s t.1 t.2.1 (t.2.2 s)) :=
        Claims.Pr_exists_le fun s (t : Ω × (Fin (L.V.k x) → Coin) × L.EO.C x) =>
          L.SessBad H P x s t.1 t.2.1 (t.2.2 s)
    _ = _ := by
        refine Finset.sum_congr rfl fun s _ => ?_
        have hev : UnifMap (fun c : L.EO.C x => c s) :=
          (UnifMap.equiv (Equiv.piSplitAt s fun j => A.EC (L.sh x j))).comp UnifMap.fst
        exact (UnifMap.prod (UnifMap.self (A := Ω))
          (UnifMap.prod (UnifMap.self (A := Fin (L.V.k x) → Coin)) hev)).Pr
            fun t : Ω × (Fin (L.V.k x) → Coin) × A.EC (L.sh x s) => L.SessBad H P x s t.1 t.2.1 t.2.2

/-! ## Each prefix: `A`'s KS at the session prover -/

open Classical in
/-- **Session `s`'s collision probability at the prefix of `r`**: `A`'s `Pr[B]` at the session prover and statement,
where some continuation of the prefix accepts; `0` where none does. -/
noncomputable def sPrB (L : Compiled Coin A) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (s : Fin (L.nS x))
    (ω : Ω) (r : Fin (L.V.k x) → Coin) : ℝ :=
  if ∃ d, (run H P L.V x ω (fork (L.off x s) r d)).out = .acc then
    PrB H A.V (L.proverAt H P x ω s r) (L.stmtAt H P x ω s r)
  else 0

theorem sPrB_nonneg (L : Compiled Coin A) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (s : Fin (L.nS x))
    (ω : Ω) (r : Fin (L.V.k x) → Coin) : 0 ≤ L.sPrB H P x s ω r := by
  unfold sPrB
  split_ifs
  · exact PrB_nonneg _ _ _ _
  · exact le_rfl

theorem sPrB_eq_zero (L : Compiled Coin A) (H : Msg → Msg) (hS : L.Sessions H) {Ω : Type}
    (P : Prover Ω) (x : Msg) (s : Fin (L.nS x)) (ω : Ω) (r : Fin (L.V.k x) → Coin) (h0 : A.rounds (L.sh x s) = 0) :
    L.sPrB H P x s ω r = 0 := by
  unfold sPrB
  split_ifs with hA
  · obtain ⟨d₁, hacc⟩ := hA
    have hk : A.V.k (L.stmtAt H P x ω s r) = 0 := by rw [A.wf, L.shape_stmtAt H hS P x ω s r d₁ hacc, h0]
    simp [PrB, hk]
  · rfl

/-- **One prefix**: over the forks of `r` at session `s`'s first round and the session's coins, its extraction fails
at most as often as `A`'s extractor at the session prover (`bad_session`, the coins read uniformly), which `A`'s KS
bounds at the session's staged statement, within the session's bounds (`Bounded`, at an accepted fork); it never
fails where no fork accepts. -/
theorem prefix_bad_le (L : Compiled Coin A) (H : Msg → Msg) (hS : L.Sessions H) (hKS : A.KS H) {Ω : Type}
    (P : Prover Ω) (x : Msg) (s : Fin (L.nS x)) (ω : Ω) (r : Fin (L.V.k x) → Coin) :
    Pr (fun q : (Fin (L.V.k x) → Coin) × A.EC (L.sh x s) => L.SessBad H P x s ω (fork (L.off x s) r q.1) q.2) ≤
      L.κb x s + Real.sqrt (L.γb x s * L.sPrB H P x s ω r) := by
  unfold sPrB
  split_ifs with hA
  · obtain ⟨d₁, hacc₁⟩ := hA
    have hsh := L.shape_stmtAt H hS P x ω s r d₁ hacc₁
    have hbd : A.κ (L.stmtAt H P x ω s r) ≤ L.κb x s ∧ A.γ (L.stmtAt H P x ω s r) ≤ L.γb x s := by
      rw [← L.stmtAt_fork H hS.stages P x ω s r d₁]
      exact (hS.bounded x s).2.2 Ω P ω _ hacc₁
    have hle : L.off x s + A.V.k (L.stmtAt H P x ω s r) ≤ L.V.k x := by
      rw [A.wf, hsh]
      exact L.off_add_le hS.fits x s
    have hφ : UnifMap (fun q : (Fin (L.V.k x) → Coin) × A.EC (L.sh x s) =>
        ((), sessCoins (L.off x s) q.1 (A.V.k (L.stmtAt H P x ω s r)), cast (congrArg A.EC hsh.symm) q.2)) :=
      UnifMap.comp ((unifMap_sessCoins _ _ hle).prod (UnifMap.equiv (Equiv.cast (congrArg A.EC hsh.symm))))
        UnifMap.unit
    calc _ ≤ Pr (fun q : (Fin (L.V.k x) → Coin) × A.EC (L.sh x s) =>
            (run H (L.proverAt H P x ω s r) A.V (L.stmtAt H P x ω s r) ()
              (sessCoins (L.off x s) q.1 (A.V.k (L.stmtAt H P x ω s r)))).out = .acc ∧
            ¬ A.R.ext (L.stmtAt H P x ω s r)
              (A.E.eval H (L.proverAt H P x ω s r) (L.stmtAt H P x ω s r) ()
                (sessCoins (L.off x s) q.1 (A.V.k (L.stmtAt H P x ω s r)))
                (cast (congrArg A.EC hsh.symm) q.2)).1) :=
          Pr_mono fun q hq => L.bad_session H hS P x ω s r q.1 q.2 hsh hq
      _ = PrBad H A.V A.R A.E (L.stmtAt H P x ω s r) (L.proverAt H P x ω s r) :=
          hφ.Pr fun t : Unit × (Fin (A.V.k (L.stmtAt H P x ω s r)) → Coin) × A.E.C (L.stmtAt H P x ω s r) =>
            (run H (L.proverAt H P x ω s r) A.V (L.stmtAt H P x ω s r) t.1 t.2.1).out = .acc ∧
              ¬ A.R.ext (L.stmtAt H P x ω s r)
                (A.E.eval H (L.proverAt H P x ω s r) (L.stmtAt H P x ω s r) t.1 t.2.1 t.2.2).1
      _ ≤ A.κ (L.stmtAt H P x ω s r) + Real.sqrt (A.γ (L.stmtAt H P x ω s r) *
            PrB H A.V (L.proverAt H P x ω s r) (L.stmtAt H P x ω s r)) :=
          hKS (L.stmtAt H P x ω s r) Unit (L.proverAt H P x ω s r)
      _ ≤ _ := add_le_add hbd.1 (Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_right hbd.2 (PrB_nonneg _ _ _ _)))
  · rw [mul_zero, Real.sqrt_zero, add_zero]
    calc _ ≤ Pr (fun _ : (Fin (L.V.k x) → Coin) × A.EC (L.sh x s) => False) :=
          Pr_mono fun q hq => hA ⟨q.1, hq.1⟩
      _ = 0 := Pr_false
      _ ≤ _ := (hS.bounded x s).1

/-! ## The collision term: the session's forks are the compiled verifier's -/

/-- **The session's forks are the compiled verifier's**: at a prefix some continuation of which accepts, `A`'s fork at
round `j` of the session prover collides only where the compiled verifier's fork at round `off + j` does, on the same
coins from the session's first round on. -/
theorem sPrB_mul_le (L : Compiled Coin A) (H : Msg → Msg) (hS : L.Sessions H) {Ω : Type}
    (P : Prover Ω) (x : Msg) (s : Fin (L.nS x)) (ω : Ω) (r : Fin (L.V.k x) → Coin) :
    L.sPrB H P x s ω r * A.rounds (L.sh x s) ≤
      ∑ j ∈ Finset.range (A.rounds (L.sh x s)), Pr fun p : (Fin (L.V.k x) → Coin) × (Fin (L.V.k x) → Coin) =>
        vForkColl H L.V P x ω (L.off x s + j) (fork (L.off x s) r p.1) p.2 := by
  unfold sPrB
  split_ifs with hA
  · obtain ⟨d₁, hacc⟩ := hA
    have hk' : A.V.k (L.stmtAt H P x ω s r) = A.rounds (L.sh x s) := by
      rw [A.wf, L.shape_stmtAt H hS P x ω s r d₁ hacc]
    have hle : L.off x s + A.V.k (L.stmtAt H P x ω s r) ≤ L.V.k x := by
      rw [hk']
      exact L.off_add_le hS.fits x s
    rw [← hk', PrB_mul_eq]
    refine Finset.sum_le_sum fun j _ => ?_
    have hφ : UnifMap (fun p : (Fin (L.V.k x) → Coin) × (Fin (L.V.k x) → Coin) =>
        ((), sessCoins (L.off x s) p.1 (A.V.k (L.stmtAt H P x ω s r)),
          sessCoins (L.off x s) p.2 (A.V.k (L.stmtAt H P x ω s r)))) :=
      UnifMap.comp ((unifMap_sessCoins _ _ hle).prod (unifMap_sessCoins _ _ hle)) UnifMap.unit
    unfold forkPrN
    rw [← hφ.Pr fun t => vForkColl H A.V (L.proverAt H P x ω s r) (L.stmtAt H P x ω s r) t.1 j t.2.1 t.2.2]
    refine Pr_mono fun p hp => ?_
    dsimp only [vForkColl] at hp ⊢
    rw [fork_sessCoins] at hp
    rw [← fork_add_fork]
    refine findColl_isSome_of_sub H (fun e he => ?_) (fun e he => ?_) hp
    · rcases List.mem_append.1 he with he | he
      · exact (run_logs H P L.V x ω _).2 e he
      · exact (run_logs H P L.V x ω _).2 e he
    · rcases List.mem_append.1 he with he | he
      · exact List.mem_append_left _ ((L.runs_fork H hS P x ω s r d₁ p.1 hacc).2 e he)
      · exact List.mem_append_right _ ((L.runs_fork H hS P x ω s r d₁ _ hacc).2 e he)
  · rw [zero_mul]
    exact Finset.sum_nonneg fun j _ => Pr_nonneg _

/-- **The collision term, averaged**: `𝔼[Pr[B] at the session prover]·k_s ≤ k(x)·Pr[B]`. A sample's prefix is a fork of
a sample (`Ex_fork`), so the session's forks at rounds `off + j`, `j < k_s`, are among the compiled verifier's `k(x)`
round guesses (`Fits`). -/
theorem Ex_sPrB_le (L : Compiled Coin A) (H : Msg → Msg) (hS : L.Sessions H) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (x : Msg) (s : Fin (L.nS x)) :
    Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => L.sPrB H P x s p.1 p.2) * A.rounds (L.sh x s) ≤
      PrB H L.V P x * L.V.k x := by
  have hoff : PrefixDet (k := L.V.k x) (Coin := Coin) fun _ => L.off x s := fun _ _ _ => rfl
  rw [← Ex_mul_right, PrB_mul_eq, Ex_prod]
  have hω : ∀ ω : Ω, Ex (fun r : Fin (L.V.k x) → Coin => L.sPrB H P x s ω r * A.rounds (L.sh x s)) ≤
      Ex fun R : Fin (L.V.k x) → Coin => ∑ j ∈ Finset.range (L.V.k x),
        Pr fun d' : Fin (L.V.k x) → Coin => vForkColl H L.V P x ω j R d' := by
    intro ω
    let G : (Fin (L.V.k x) → Coin) → ℝ := fun R => ∑ j ∈ Finset.range (A.rounds (L.sh x s)),
      Pr fun d' : Fin (L.V.k x) → Coin => vForkColl H L.V P x ω (L.off x s + j) R d'
    calc Ex (fun r : Fin (L.V.k x) → Coin => L.sPrB H P x s ω r * A.rounds (L.sh x s))
        ≤ Ex fun r : Fin (L.V.k x) → Coin => ∑ j ∈ Finset.range (A.rounds (L.sh x s)),
            Pr fun p : (Fin (L.V.k x) → Coin) × (Fin (L.V.k x) → Coin) =>
              vForkColl H L.V P x ω (L.off x s + j) (fork (L.off x s) r p.1) p.2 :=
          Ex_mono fun r => L.sPrB_mul_le H hS P x s ω r
      _ = Ex fun p : (Fin (L.V.k x) → Coin) × (Fin (L.V.k x) → Coin) => G (fork (L.off x s) p.1 p.2) := by
          rw [Ex_prod]
          congr 1
          funext r
          simp only [G]
          rw [Ex_sum]
          exact Finset.sum_congr rfl fun j _ => Claims.Recursion.Pr_prod _
      _ = Ex G := Ex_fork hoff G
      _ ≤ _ := Ex_mono fun R =>
          sum_range_shift_le (fun j => Pr fun d' : Fin (L.V.k x) → Coin => vForkColl H L.V P x ω j R d')
            (fun j => Pr_nonneg _) (L.off_add_le hS.fits x s)
  calc _ ≤ Ex fun ω : Ω => Ex fun R : Fin (L.V.k x) → Coin => ∑ j ∈ Finset.range (L.V.k x),
          Pr fun d' : Fin (L.V.k x) → Coin => vForkColl H L.V P x ω j R d' := Ex_mono fun ω => hω ω
    _ = ∑ j ∈ Finset.range (L.V.k x), forkPrN H L.V P x j := by
      simp only [Ex_sum]
      refine Finset.sum_congr rfl fun j _ => ?_
      unfold forkPrN
      rw [Claims.Recursion.Pr_prod]
      congr 1
      funext ω
      rw [Claims.Recursion.Pr_prod]

/-! ## Each session, from `A`'s KS -/

/-- **Session `s` fails at most at `κb_s + √(γb_s·k(x)·Pr[B])`**: each prefix's failure is `A`'s at the session
prover, within the session's bounds (`prefix_bad_le`); Jensen moves the average under the root, and the session's round
guesses are among the compiled verifier's `k(x)` (`Ex_sPrB_le`). -/
theorem sessKS (L : Compiled Coin A) (H : Msg → Msg) (hS : L.Sessions H) (hKS : A.KS H) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (x : Msg) (s : Fin (L.nS x)) :
    Pr (fun t : Ω × (Fin (L.V.k x) → Coin) × A.EC (L.sh x s) => L.SessBad H P x s t.1 t.2.1 t.2.2) ≤
      L.κb x s + Real.sqrt (L.γb x s * L.V.k x * PrB H L.V P x) := by
  have hoff : PrefixDet (k := L.V.k x) (Coin := Coin) fun _ => L.off x s := fun _ _ _ => rfl
  have hγ : 0 ≤ L.γb x s := (hS.bounded x s).2.1
  have hZ0 : ∀ p : Ω × (Fin (L.V.k x) → Coin), 0 ≤ L.sPrB H P x s p.1 p.2 := fun p =>
    L.sPrB_nonneg H P x s p.1 p.2
  have hZ : Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => L.sPrB H P x s p.1 p.2) ≤ L.V.k x * PrB H L.V P x := by
    rcases Nat.eq_zero_or_pos (A.rounds (L.sh x s)) with h0 | hpos
    · have hz : ∀ p : Ω × (Fin (L.V.k x) → Coin), L.sPrB H P x s p.1 p.2 = 0 := fun p =>
        L.sPrB_eq_zero H hS P x s p.1 p.2 h0
      calc Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => L.sPrB H P x s p.1 p.2)
          = Ex (fun _ : Ω × (Fin (L.V.k x) → Coin) => (0 : ℝ)) := congrArg Ex (funext hz)
        _ = 0 := Ex_const 0
        _ ≤ _ := mul_nonneg (Nat.cast_nonneg _) (PrB_nonneg H L.V P x)
    · have h1 : (1 : ℝ) ≤ A.rounds (L.sh x s) := by exact_mod_cast hpos
      calc Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => L.sPrB H P x s p.1 p.2)
          ≤ Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => L.sPrB H P x s p.1 p.2) * A.rounds (L.sh x s) :=
            le_mul_of_one_le_right (Ex_nonneg hZ0) h1
        _ ≤ PrB H L.V P x * L.V.k x := L.Ex_sPrB_le H hS P x s
        _ = L.V.k x * PrB H L.V P x := mul_comm _ _
  have hL : Pr (fun t : Ω × (Fin (L.V.k x) → Coin) × A.EC (L.sh x s) => L.SessBad H P x s t.1 t.2.1 t.2.2) =
      Ex fun ω : Ω => Ex fun r : Fin (L.V.k x) → Coin => Pr fun q : (Fin (L.V.k x) → Coin) × A.EC (L.sh x s) =>
        L.SessBad H P x s ω (fork (L.off x s) r q.1) q.2 := by
    rw [Claims.Recursion.Pr_prod]
    refine congrArg Ex (funext fun ω => ?_)
    rw [Claims.Recursion.Pr_prod, ← Ex_fork hoff (fun r => Pr fun c : A.EC (L.sh x s) => L.SessBad H P x s ω r c),
      Ex_prod]
    refine congrArg Ex (funext fun r => ?_)
    rw [Claims.Recursion.Pr_prod (fun q : (Fin (L.V.k x) → Coin) × A.EC (L.sh x s) =>
      L.SessBad H P x s ω (fork (L.off x s) r q.1) q.2)]
  have hb : ∀ (ω : Ω) (r : Fin (L.V.k x) → Coin),
      Pr (fun q : (Fin (L.V.k x) → Coin) × A.EC (L.sh x s) => L.SessBad H P x s ω (fork (L.off x s) r q.1) q.2) ≤
        L.κb x s + Real.sqrt (L.γb x s * L.sPrB H P x s ω r) := fun ω r =>
    L.prefix_bad_le H hS hKS P x s ω r
  rw [hL]
  calc (Ex fun ω : Ω => Ex fun r : Fin (L.V.k x) → Coin =>
          Pr fun q : (Fin (L.V.k x) → Coin) × A.EC (L.sh x s) => L.SessBad H P x s ω (fork (L.off x s) r q.1) q.2)
      ≤ Ex fun ω : Ω => Ex fun r : Fin (L.V.k x) → Coin =>
          L.κb x s + Real.sqrt (L.γb x s * L.sPrB H P x s ω r) :=
        Ex_mono fun ω => Ex_mono fun r => hb ω r
    _ = Ex fun p : Ω × (Fin (L.V.k x) → Coin) =>
          L.κb x s + Real.sqrt (L.γb x s * L.sPrB H P x s p.1 p.2) :=
        (Ex_prod fun p : Ω × (Fin (L.V.k x) → Coin) =>
          L.κb x s + Real.sqrt (L.γb x s * L.sPrB H P x s p.1 p.2)).symm
    _ = L.κb x s + Ex fun p : Ω × (Fin (L.V.k x) → Coin) =>
          Real.sqrt (L.γb x s * L.sPrB H P x s p.1 p.2) := by
        rw [Ex_add, Ex_const]
    _ ≤ L.κb x s + Real.sqrt (Ex fun p : Ω × (Fin (L.V.k x) → Coin) =>
          L.γb x s * L.sPrB H P x s p.1 p.2) :=
        add_le_add le_rfl (Ex_sqrt_le _ fun p => mul_nonneg hγ (hZ0 p))
    _ ≤ L.κb x s + Real.sqrt (L.γb x s * L.V.k x * PrB H L.V P x) := by
        rw [Ex_mul_left, mul_assoc]
        exact add_le_add le_rfl (Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left hZ hγ))

end Compiled

/-- **O1** (`OuterSessionsKS`): `E_O` fails at most at the sum of its sessions' failures (`Compiled.PrBad_le_sum`),
session `s` at most at `κb_s + √(γb_s·k(x))·√Pr[B]` (`Compiled.sessKS`), and the sum of the roots is
`√(γ_O·Pr[B])`. -/
theorem outerSessionsKS : OuterSessionsKS := by
  intro Coin _ _ _ H A L hKS hS x Ω _ _ P
  have hk : ∀ s : Fin (L.nS x), Real.sqrt (L.γb x s * L.V.k x * PrB H L.V P x) =
      Real.sqrt (L.γb x s * L.V.k x) * Real.sqrt (PrB H L.V P x) := fun s =>
    Real.sqrt_mul (mul_nonneg (hS.bounded x s).2.1 (Nat.cast_nonneg _)) _
  have hO : Real.sqrt (L.toComp.γO x * PrB H L.V P x) =
      (∑ s : Fin (L.nS x), Real.sqrt (L.γb x s * L.V.k x)) * Real.sqrt (PrB H L.V P x) := by
    show Real.sqrt ((∑ s : Fin (L.nS x), Real.sqrt (L.γb x s * L.V.k x)) ^ 2 * PrB H L.V P x) = _
    rw [Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq (Finset.sum_nonneg fun s _ => Real.sqrt_nonneg _)]
  calc PrBad H L.V L.toComp.RO L.toComp.EO x P
      ≤ ∑ s : Fin (L.nS x), Pr (fun t : Ω × (Fin (L.V.k x) → Coin) × A.EC (L.sh x s) =>
          L.SessBad H P x s t.1 t.2.1 t.2.2) := L.PrBad_le_sum H P x
    _ ≤ ∑ s : Fin (L.nS x), (L.κb x s + Real.sqrt (L.γb x s * L.V.k x * PrB H L.V P x)) :=
        Finset.sum_le_sum fun s _ => L.sessKS H hS hKS P x s
    _ = L.toComp.κO x + Real.sqrt (L.toComp.γO x * PrB H L.V P x) := by
        rw [hO, Finset.sum_add_distrib, Finset.sum_mul]
        simp only [hk]
        rfl

/-- `κ_O` is nonnegative where the sessions' bounds are (`Compiled.Bounded`). -/
theorem Compiled.κO_nonneg {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}
    {L : Compiled Coin A} {H : Msg → Msg} (hS : L.Sessions H) (x : Msg) : 0 ≤ L.toComp.κO x :=
  show (0 : ℝ) ≤ ∑ s : Fin (L.nS x), L.κb x s from Finset.sum_nonneg fun s _ => (hS.bounded x s).1

/-- `γ_O` is a square. -/
theorem Compiled.γO_nonneg {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}
    (L : Compiled Coin A) (x : Msg) : 0 ≤ L.toComp.γO x :=
  show (0 : ℝ) ≤ (∑ s : Fin (L.nS x), Real.sqrt (L.γb x s * L.V.k x)) ^ 2 from sq_nonneg _

/-- **L2** (`CompilerKS`): L1 (`compositionKS`) at `compile(Π_in, A, S)`, whose `E_O` is knowledge sound by O1
(`outerSessionsKS`) at `κ_O`, `γ_O`, nonnegative by the sessions' bounds. -/
theorem compilerKS : CompilerKS := by
  intro Coin W _ _ _ H I S A C hWF hKS hCF hB hAKS hSess x hK Ω _ _ P m M
  exact compositionKS Coin W H I S (compile C I S).toComp hWF hKS hCF hB x
    (fun Ω _ _ P => outerSessionsKS Coin H A (compile C I S) hAKS hSess x Ω P)
    (Compiled.κO_nonneg hSess x) (Compiled.γO_nonneg _ x) hK Ω P m M

end Claims.RecGeneric
