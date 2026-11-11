module

public import Proofs.Flock.RecGeneric.OuterProof
public import Proofs.Flock.RecGeneric.AtFlockCompiledSessions

/-!
# `Compiled.EO`'s expected reruns

`E_O` at a compiled verifier (`Compiled.EO`) replays the judged run and then, at each session, runs `A`'s extractor
`E^rep` on forks of the judged run at the session's first round: one replay at the session's coins of the judged run,
then `m′` waits over fresh session coins, each stopping at `k` accepted reruns. So it reruns the prover at most
`1 + Σ_s (1 + m′_s·k_s)` times in expectation (`Compiled.EO_runs_exp_of`), one more per session than
`Claims.Recursion.EO`'s `1 + Σ_s m′_s·k_s`, wherever the staged statements read only the rounds before the sessions
(`Stages`), each has its session's shape and `A`'s `k` there is at most `k_s`.

* A session's cost at the judged coins `r` is its replay and, where the replay accepts, its waits
  (`Compiled.sessionProg_runs_le`); over `E_O`'s coins for the session the waits average `m′·N`, `N` one wait's expected
  reruns over fresh session coins (`Compiled.Ex_sessionProg_le`).
* The replay and the waits' reruns are one function of `r`'s coins before the session's first round and of the session's
  coins (`Compiled.sessRunAt`), and the session's coins of `r` are uniform where the run reads them, after that round and
  before the compiled verifier's last (`Ex_sessCoins`), so the replay accepts, at each prefix, with the probability `ε`
  that a wait's rerun does (`Ex_acc_fork`). One wait's `ε·N ≤ k` (`waitRuns_exp`) then gives `m′·k` per session.

No fact about where the sessions end is read: a session's coins past the compiled verifier's last round are coins no run
reads. At today's compiled record (`Compiled.ofRec`), each session's V\* statement has the session's numbers
(`SessionNums`), which fix its shape and its `k`, so the bound holds there (`ofRec_EO_runs_exp`) and with it
`CompiledEORunsExp` (`compiledEORunsExp`).
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion
open Claims.SamplingGeneric (Arg)

section Coins

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- The coins of `p.1` from round `o` on swapped with the session's coins `p.2`, where both exist. -/
def swapSess {k n : ℕ} (o : ℕ) (p : (Fin k → Coin) × (Fin n → Coin)) : (Fin k → Coin) × (Fin n → Coin) :=
  (fun i => if h : o ≤ i.val ∧ i.val - o < n then p.2 ⟨i.val - o, h.2⟩ else p.1 i,
    fun j => if h : o + j.val < k then p.1 ⟨o + j.val, h⟩ else p.2 j)

theorem swapSess_swapSess {k n : ℕ} (o : ℕ) (p : (Fin k → Coin) × (Fin n → Coin)) :
    swapSess o (swapSess o p) = p := by
  obtain ⟨d, e⟩ := p
  refine Prod.ext (funext fun i => ?_) (funext fun j => ?_)
  · have hi := i.isLt
    simp only [swapSess]
    split_ifs with h1 h2 <;>
      first | rfl | (exfalso; omega) | exact congrArg d (Fin.ext (Nat.add_sub_of_le (by omega)))
  · have hj := j.isLt
    simp only [swapSess]
    split_ifs with h1 h2 <;>
      first | rfl | (exfalso; omega) | exact congrArg e (Fin.ext (Nat.add_sub_cancel_left _ _))

/-- **A session's coins read off uniform coins are uniform where `F` reads them**: `F` reads the session's coins only
at the rounds `o + j` the run has. -/
theorem Ex_sessCoins {k n : ℕ} (o : ℕ) (F : (Fin n → Coin) → ℝ)
    (hF : ∀ d e : Fin n → Coin, (∀ j : Fin n, o + j.val < k → d j = e j) → F d = F e) :
    Ex (fun d : Fin k → Coin => F (sessCoins o d n)) = Ex F := by
  have hsw : ∀ p : (Fin k → Coin) × (Fin n → Coin), F (sessCoins o p.1 n) = F (swapSess o p).2 :=
    fun p => hF _ _ fun j hj => by simp [sessCoins, coinSeq, swapSess, hj]
  have h1 : Ex (fun p : (Fin k → Coin) × (Fin n → Coin) => F (sessCoins o p.1 n)) =
      Ex (fun p : (Fin k → Coin) × (Fin n → Coin) => F p.2) := by
    rw [congrArg Ex (funext hsw)]
    exact Claims.Recursion.Ex_equiv
      (⟨swapSess o, swapSess o, swapSess_swapSess o, swapSess_swapSess o⟩ :
        ((Fin k → Coin) × (Fin n → Coin)) ≃ ((Fin k → Coin) × (Fin n → Coin)))
      fun p => F p.2
  rw [Ex_prod, Ex_prod] at h1
  simpa only [Ex_const] using h1

/-- A fork at round `o` onto coins placed from `o` on reads them only at the rounds the run has. -/
theorem fork_placeAt_congr {k n : ℕ} (o : ℕ) (r : Fin k → Coin) {d e : Fin n → Coin}
    (h : ∀ j : Fin n, o + j.val < k → d j = e j) :
    fork o r (Compiled.placeAt o d) = fork o r (Compiled.placeAt o e) := by
  funext i
  have hi := i.isLt
  simp only [fork, Compiled.placeAt, coinSeq]
  split_ifs with h1 h2
  all_goals first | rfl | exact h _ (by first | omega | (dsimp only; omega))

/-- `A`'s extractor's coins at a session's shape, as its rerun program reads them at a statement of that shape. -/
def ecEquiv (A : Arg Coin) (xO : Msg) :
    A.EC (A.shape xO) ≃ (Fin (A.waits (A.shape xO)) → Fin (A.cap (A.shape xO)) → Fin (A.V.k xO) → Coin) where
  toFun c i j t := c i j (Fin.cast (A.wf xO) t)
  invFun ρ i j t := ρ i j (Fin.cast (A.wf xO).symm t)
  left_inv c := by first | rfl | (funext i j t; simp)
  right_inv ρ := by first | rfl | (funext i j t; simp)

/-- **`A`'s extractor's coins at a session's shape are uniform** as its rerun program reads them at a statement of
that shape. -/
theorem Ex_EC_cast (A : Arg Coin) (xO : Msg) {sh : A.Sh} (h : A.shape xO = sh)
    (F : (Fin (A.waits (A.shape xO)) → Fin (A.cap (A.shape xO)) → Fin (A.V.k xO) → Coin) → ℝ) :
    Ex (fun c : A.EC sh => F fun i j t => (cast (congrArg A.EC h.symm) c) i j (Fin.cast (A.wf xO) t)) = Ex F := by
  subst h
  exact Claims.Recursion.Ex_equiv (ecEquiv A xO) F

/-- **`E^rep`'s reruns at a judged coin vector `d₀`**, against any runs: the replay and, where it accepted, the waits. -/
theorem ERep_ext_runs_le {W Op : Type} (H : Msg → Msg) (V : Verifier Coin) (m k K : Msg → ℕ)
    (opens : Seen → List Op) (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) (x : Msg)
    (R : (Fin (V.k x) → Coin) → Transcript) (d₀ : Fin (V.k x) → Coin) (ρ : Fin (m x) → Fin (K x) → Fin (V.k x) → Coin) :
    ((((ERep V m k K opens assemble check).ext x d₀ ρ).eval H R).2.2 : ℝ) ≤
      (if (R d₀).out = .acc then (1 : ℝ) else 0) *
        ∑ i, (waitRuns R (k x) ((List.finRange (K x)).map (ρ i)) 0 : ℝ) + 1 := by
  simp only [ERep, RP.eval]
  by_cases h : (R d₀).out = .acc
  · have hs : (R d₀).seen.out = .acc := h
    have hw := firstWait_runs_le_sum H R (k x) opens (assemble x) (check x)
      ((List.finRange (m x)).map fun i => (List.finRange (K x)).map (ρ i))
    simp only [List.map_map, Function.comp_def] at hw
    rw [← Fin.sum_univ_def] at hw
    simp only [hs, h, ↓reduceIte, one_mul]
    exact_mod_cast Nat.add_le_add_right hw 1
  · have hs : ¬ (R d₀).seen.out = .acc := h
    simp [hs, h, RP.eval]

/-- A rerun program served by reruns at `g r`, each seen through `f`, reruns the prover once per rerun of its own. -/
theorem serve_viewingAt_runs {R₁ R₂ α : Type} (H : Msg → Msg) (run : R₂ → Transcript) (f : Seen → Seen) (g : R₁ → R₂)
    (p : RP R₁ α) :
    ((serve (fun r => viewing f (g r)) p).eval H run).2.2 =
      (p.eval H (servedRun H (fun r => viewing f (g r)) run)).2.2 := by
  rw [serve_eq, (RP.serve_eval H _ run p).2.2, RP.eval_runs_eq]
  simp only [viewing_runs, List.map_const', List.sum_replicate, smul_eq_mul, mul_one]

end Coins

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-! ## One session's reruns -/

/-- **Session `s`'s view of a rerun at the session's coins `d`**, as `E_O`'s session program serves it at the judged
coins `r`: the run at the fork of `r` at the session's first round onto `d`, placed from that round on, seen through the
session's view. -/
def sessRunAt (L : Compiled Coin A) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (s : ℕ)
    (r : Fin (L.V.k x) → Coin) (n : ℕ) (d : Fin n → Coin) : Transcript :=
  servedRun H (fun r' : Fin n → Coin => viewing (L.session x s) (fork (L.off x s) r (placeAt (L.off x s) r')))
    (run H P L.V x ω) d

theorem sessRunAt_fork (L : Compiled Coin A) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (s : ℕ)
    (r d : Fin (L.V.k x) → Coin) (n : ℕ) :
    L.sessRunAt H P x ω s (fork (L.off x s) r d) n = L.sessRunAt H P x ω s r n := by
  funext e
  simp only [sessRunAt, fork_fork]

theorem sessRunAt_congr (L : Compiled Coin A) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (s : ℕ)
    (r : Fin (L.V.k x) → Coin) {n : ℕ} {d e : Fin n → Coin}
    (h : ∀ j : Fin n, L.off x s + j.val < L.V.k x → d j = e j) :
    L.sessRunAt H P x ω s r n d = L.sessRunAt H P x ω s r n e := by
  simp only [sessRunAt, servedRun, fork_placeAt_congr (L.off x s) r h]

/-- **Session `s`'s replay accepts** at the judged coins `r`: `E_O`'s session program's first rerun, at the session's
coins of `r`, at the statement `r`'s staging rounds fix. -/
abbrev replayAt (L : Compiled Coin A) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (s : ℕ)
    (r : Fin (L.V.k x) → Coin) : Prop :=
  (L.sessRunAt H P x ω s r (A.V.k (L.stmtAt H P x ω s r))
    (sessCoins (L.off x s) r (A.V.k (L.stmtAt H P x ω s r)))).out = .acc

/-- **Session `s`'s waits' expected reruns** at the judged coins `r`: `m′` waits, each over fresh session coins, at the
statement `r`'s staging rounds fix. -/
noncomputable def waitsAt (L : Compiled Coin A) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (s : ℕ)
    (r : Fin (L.V.k x) → Coin) : ℝ :=
  (A.waits (A.shape (L.stmtAt H P x ω s r)) : ℝ) *
    waitExp (L.sessRunAt H P x ω s r (A.V.k (L.stmtAt H P x ω s r))) (A.k (A.shape (L.stmtAt H P x ω s r)))
      (A.cap (A.shape (L.stmtAt H P x ω s r))) 0

theorem replayAt_fork (L : Compiled Coin A) (H : Msg → Msg) (hSt : L.Stages) {Ω : Type} (P : Prover Ω) (x : Msg)
    (ω : Ω) (s : ℕ) (r d : Fin (L.V.k x) → Coin) :
    L.replayAt H P x ω s (fork (L.off x s) r d) =
      ((L.sessRunAt H P x ω s r (A.V.k (L.stmtAt H P x ω s r))
        (sessCoins (L.off x s) d (A.V.k (L.stmtAt H P x ω s r)))).out = .acc) := by
  unfold replayAt
  rw [L.stmtAt_fork H hSt, L.sessRunAt_fork, sessCoins_fork]

theorem waitsAt_fork (L : Compiled Coin A) (H : Msg → Msg) (hSt : L.Stages) {Ω : Type} (P : Prover Ω) (x : Msg)
    (ω : Ω) (s : ℕ) (r d : Fin (L.V.k x) → Coin) :
    L.waitsAt H P x ω s (fork (L.off x s) r d) = L.waitsAt H P x ω s r := by
  unfold waitsAt
  rw [L.stmtAt_fork H hSt, L.sessRunAt_fork]

/-- **Session `s`'s reruns at the judged coins `r`**, at a statement `xO` of the session's shape: the replay, at the
session's coins of `r`, and, where it accepted, `E^rep`'s waits over the session coins `E_O`'s coins give. -/
theorem sessionProg_runs_le (L : Compiled Coin A) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (s : Fin (L.nS x)) (r : Fin (L.V.k x) → Coin) (xO : Msg) (h : A.shape xO = L.sh x s) (c : A.EC (L.sh x s)) :
    (((L.sessionProg x r s xO c).eval H (run H P L.V x ω)).2.2 : ℝ) ≤
      (if (L.sessRunAt H P x ω s r (A.V.k xO) (sessCoins (L.off x s) r (A.V.k xO))).out = .acc then (1 : ℝ)
        else 0) *
        ∑ i, (waitRuns (L.sessRunAt H P x ω s r (A.V.k xO)) (A.k (A.shape xO))
          ((List.finRange (A.cap (A.shape xO))).map
            fun j t => (cast (congrArg A.EC h.symm) c) i j (Fin.cast (A.wf xO) t)) 0 : ℝ) + 1 := by
  unfold sessionProg
  rw [dif_pos h, RP.eval_bind_ret]
  dsimp only
  rw [serve_viewingAt_runs]
  exact ERep_ext_runs_le H A.V (fun x => A.waits (A.shape x)) (fun x => A.k (A.shape x))
    (fun x => A.cap (A.shape x)) A.opens
    (fun x os => (A.assemble x os).1) A.check xO (L.sessRunAt H P x ω s r (A.V.k xO))
    (sessCoins (L.off x s) r (A.V.k xO)) fun i j t => (cast (congrArg A.EC h.symm) c) i j (Fin.cast (A.wf xO) t)

/-- **Over `E_O`'s coins for session `s`**, at the judged coins `r` and a statement `xO` of the session's shape, the
session reruns at most its replay and, where the replay accepts, `m′` waits' expected reruns over fresh session
coins. -/
theorem Ex_sessionProg_le (L : Compiled Coin A) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (s : Fin (L.nS x)) (r : Fin (L.V.k x) → Coin) (xO : Msg) (h : A.shape xO = L.sh x s) :
    Ex (fun c : A.EC (L.sh x s) => (((L.sessionProg x r s xO c).eval H (run H P L.V x ω)).2.2 : ℝ)) ≤
      (if (L.sessRunAt H P x ω s r (A.V.k xO) (sessCoins (L.off x s) r (A.V.k xO))).out = .acc then (1 : ℝ)
        else 0) *
        ((A.waits (A.shape xO) : ℝ) *
          waitExp (L.sessRunAt H P x ω s r (A.V.k xO)) (A.k (A.shape xO)) (A.cap (A.shape xO)) 0) + 1 := by
  refine (Ex_mono fun c => L.sessionProg_runs_le H P x ω s r xO h c).trans (le_of_eq ?_)
  rw [Ex_add, Ex_const, Ex_mul_left]
  congr 2
  exact (Ex_EC_cast A xO h fun ρ => ∑ i, (waitRuns (L.sessRunAt H P x ω s r (A.V.k xO)) (A.k (A.shape xO))
      ((List.finRange (A.cap (A.shape xO))).map (ρ i)) 0 : ℝ)).trans
    (Ex_sum_pi (fun σ : Fin (A.cap (A.shape xO)) → Fin (A.V.k xO) → Coin =>
      (waitRuns (L.sessRunAt H P x ω s r (A.V.k xO)) (A.k (A.shape xO))
        ((List.finRange (A.cap (A.shape xO))).map σ) 0 : ℝ))
      (A.waits (A.shape xO)))

/-- Session `s`'s waits read the judged coins only before the session's first round. -/
theorem waitsAt_prefix (L : Compiled Coin A) (H : Msg → Msg) (hSt : L.Stages) {Ω : Type} (P : Prover Ω) (x : Msg)
    (ω : Ω) (s : ℕ) (r r' : Fin (L.V.k x) → Coin) (h : ∀ i : Fin (L.V.k x), i.val < L.off x s → r i = r' i) :
    L.waitsAt H P x ω s r' = L.waitsAt H P x ω s r := by
  have hr : fork (L.off x s) r r' = r' := funext fun i => by
    simp only [fork]
    split_ifs with hi <;> first | exact h i hi | rfl
  rw [← hr, L.waitsAt_fork H hSt]

/-- **The replay's acceptance weighs the waits by its probability at each prefix** (`Ex_acc_fork`, at the session's
first round, a constant). -/
theorem Ex_replayAt_waits (L : Compiled Coin A) (H : Msg → Msg) (hSt : L.Stages) {Ω : Type} (P : Prover Ω)
    (x : Msg) (ω : Ω) (s : ℕ) :
    Ex (fun r : Fin (L.V.k x) → Coin => (if L.replayAt H P x ω s r then (1 : ℝ) else 0) * L.waitsAt H P x ω s r) =
      Ex (fun r => Pr (fun d : Fin (L.V.k x) → Coin => L.replayAt H P x ω s (fork (L.off x s) r d)) *
        L.waitsAt H P x ω s r) :=
  Ex_acc_fork (off := fun _ => L.off x s) (fun _ _ _ => rfl) (L.replayAt H P x ω s) (L.waitsAt H P x ω s)
    fun r r' h => L.waitsAt_prefix H hSt P x ω s r r' h

/-- **At each prefix the replay accepts with a wait's rerun's probability**: the session's coins of uniform coins
are uniform where the run reads them (`Ex_sessCoins`). -/
theorem Pr_replayAt_fork (L : Compiled Coin A) (H : Msg → Msg) (hSt : L.Stages) {Ω : Type} (P : Prover Ω)
    (x : Msg) (ω : Ω) (s : ℕ) (r : Fin (L.V.k x) → Coin) :
    Pr (fun d : Fin (L.V.k x) → Coin => L.replayAt H P x ω s (fork (L.off x s) r d)) =
      Pr (fun d => (L.sessRunAt H P x ω s r (A.V.k (L.stmtAt H P x ω s r)) d).out = .acc) := by
  have hrep : (fun d : Fin (L.V.k x) → Coin => L.replayAt H P x ω s (fork (L.off x s) r d)) =
      fun d => (L.sessRunAt H P x ω s r (A.V.k (L.stmtAt H P x ω s r))
        (sessCoins (L.off x s) d (A.V.k (L.stmtAt H P x ω s r)))).out = .acc :=
    funext fun d => L.replayAt_fork H hSt P x ω s r d
  rw [hrep, ← Ex_ind, ← Ex_ind]
  exact Ex_sessCoins (L.off x s)
    (fun d => if (L.sessRunAt H P x ω s r (A.V.k (L.stmtAt H P x ω s r)) d).out = .acc then (1 : ℝ) else 0)
    fun d e h => by simp only [L.sessRunAt_congr H P x ω s r h]

/-- **At each prefix the replay's probability times the waits' expected reruns is at most `m′·k`**: one wait's
`ε·N ≤ k` (`waitRuns_exp`). -/
theorem replay_waits_le (L : Compiled Coin A) (H : Msg → Msg) (hSt : L.Stages) {Ω : Type} (P : Prover Ω)
    (x : Msg) (ω : Ω) (s : Fin (L.nS x)) (kk : ℕ) (r : Fin (L.V.k x) → Coin)
    (hSh : A.shape (L.stmtAt H P x ω s r) = L.sh x s) (hk : A.k (A.shape (L.stmtAt H P x ω s r)) ≤ kk) :
    Pr (fun d : Fin (L.V.k x) → Coin => L.replayAt H P x ω s (fork (L.off x s) r d)) *
      L.waitsAt H P x ω s r ≤ (A.waits (L.sh x s) : ℝ) * kk := by
  rw [L.Pr_replayAt_fork H hSt P x ω s r]
  have hw := waitRuns_exp (L.sessRunAt H P x ω s r (A.V.k (L.stmtAt H P x ω s r)))
    (A.k (A.shape (L.stmtAt H P x ω s r))) (A.cap (A.shape (L.stmtAt H P x ω s r))) 0 (Nat.zero_le _)
  simp only [Nat.cast_zero, sub_zero] at hw
  have hkr : (A.k (A.shape (L.stmtAt H P x ω s r)) : ℝ) ≤ kk := by exact_mod_cast hk
  unfold waitsAt
  calc _ = (A.waits (A.shape (L.stmtAt H P x ω s r)) : ℝ) *
        (Pr (fun d => (L.sessRunAt H P x ω s r (A.V.k (L.stmtAt H P x ω s r)) d).out = .acc) *
          waitExp (L.sessRunAt H P x ω s r (A.V.k (L.stmtAt H P x ω s r))) (A.k (A.shape (L.stmtAt H P x ω s r)))
            (A.cap (A.shape (L.stmtAt H P x ω s r))) 0) := by ring
    _ ≤ (A.waits (A.shape (L.stmtAt H P x ω s r)) : ℝ) * kk :=
        mul_le_mul_of_nonneg_left (hw.trans hkr) (Nat.cast_nonneg _)
    _ = (A.waits (L.sh x s) : ℝ) * kk := by rw [hSh]

/-- **Session `s` reruns the prover at most `1 + m′·k` times in expectation**, over the judged coins and `E_O`'s
coins for the session, where the staged statements read only the rounds before the sessions, session `s`'s has its
shape and `A`'s `k` there is at most `k`: the replay accepts at each prefix with the probability `ε` a wait's rerun does
(`Ex_replayAt_waits`, `Pr_replayAt_fork`), and one wait's `ε·N ≤ k` (`replay_waits_le`). -/
theorem sess_runs_exp (L : Compiled Coin A) (H : Msg → Msg) (hSt : L.Stages) (x : Msg) {Ω : Type} (P : Prover Ω)
    (ω : Ω) (s : Fin (L.nS x)) (kk : ℕ) (hSh : ∀ r, A.shape (L.stmtAt H P x ω s r) = L.sh x s)
    (hk : ∀ r, A.k (A.shape (L.stmtAt H P x ω s r)) ≤ kk) :
    Ex (fun p : (Fin (L.V.k x) → Coin) × A.EC (L.sh x s) =>
      (((L.sessionProg x p.1 s (L.stmtAt H P x ω s p.1) p.2).eval H (run H P L.V x ω)).2.2 : ℝ)) ≤
      ((1 + A.waits (L.sh x s) * kk : ℕ) : ℝ) := by
  have hc : ∀ r, Ex (fun c : A.EC (L.sh x s) =>
      (((L.sessionProg x r s (L.stmtAt H P x ω s r) c).eval H (run H P L.V x ω)).2.2 : ℝ)) ≤
      (if L.replayAt H P x ω s r then (1 : ℝ) else 0) * L.waitsAt H P x ω s r + 1 :=
    fun r => L.Ex_sessionProg_le H P x ω s r _ (hSh r)
  rw [Ex_prod]
  refine (Ex_mono hc).trans ?_
  rw [Ex_add, Ex_const, L.Ex_replayAt_waits H hSt P x ω s]
  refine (add_le_add ((Ex_mono fun r => L.replay_waits_le H hSt P x ω s kk r (hSh r) (hk r)).trans
    (Ex_const _).le) le_rfl).trans (le_of_eq ?_)
  push_cast
  ring

/-! ## `E_O`'s expected reruns -/

/-- **`E_O` reruns the prover at most `1 + Σ_s (1 + m′_s·k_s)` times in expectation** at a compiled verifier, wherever
the staged statements read only the rounds before the sessions (`Stages`), each session's has the session's shape and
`A`'s `k` there is at most `k_s`: the judged run, then at each session its replay and its waits (`sess_runs_exp`). -/
theorem EO_runs_exp_of (L : Compiled Coin A) (H : Msg → Msg) (hSt : L.Stages) (kk : Msg → ℕ → ℕ)
    (hSh : ∀ x (s : Fin (L.nS x)) ms, A.shape ((L.stmts x ms).getD s []) = L.sh x s)
    (hk : ∀ x (s : Fin (L.nS x)) ms, A.k (A.shape ((L.stmts x ms).getD s [])) ≤ kk x s) :
    ExtractRunsExpLe H L.EO fun x => 1 + ∑ s : Fin (L.nS x), ((1 + A.waits (L.sh x s) * kk x s : ℕ) : ℝ) := by
  intro x Ω P ω
  have hpt : ∀ p : (Fin (L.V.k x) → Coin) × L.EO.C x, ((L.EO.eval H P x ω p.1 p.2).2.2 : ℝ) ≤
      1 + ∑ s : Fin (L.nS x),
        (((L.sessionProg x p.1 s (L.stmtAt H P x ω s p.1) (p.2 s)).eval H (run H P L.V x ω)).2.2 : ℝ) := by
    rintro ⟨r₀, c⟩
    simp only [Extractor.eval, EO, RP.eval]
    split_ifs with hacc
    · rw [Nat.cast_add, Nat.cast_one, add_comm (1 : ℝ)]
      refine add_le_add ((Nat.cast_le.2 (collect_runs_le H _ _)).trans ?_) le_rfl
      rw [List.map_map, Nat.cast_list_sum, List.map_map, ← Fin.sum_univ_def]
      exact le_rfl
    · simp only [RP.eval, Nat.zero_add, Nat.cast_one]
      exact le_add_of_nonneg_right (Finset.sum_nonneg fun s _ => Nat.cast_nonneg _)
  refine (Ex_mono hpt).trans ?_
  rw [Ex_add, Ex_const, Ex_sum]
  refine add_le_add le_rfl (Finset.sum_le_sum fun s _ => ?_)
  refine (le_of_eq (Ex_pair_coord (C := fun s : Fin (L.nS x) => A.EC (L.sh x s)) s
    (fun r c => (((L.sessionProg x r s (L.stmtAt H P x ω s r) c).eval H (run H P L.V x ω)).2.2 : ℝ)))).trans ?_
  exact L.sess_runs_exp H hSt x P ω s (kk x s) (fun r => hSh x s _) (fun r => hk x s _)

end Compiled

/-! ## At today's compiled record -/

section Instance

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {Col' T' : Type} [DecidableEq Col']

/-- **`Compiled.ofRec`'s `E_O` reruns the prover at most `1 + Σ_s (1 + m′_s·k_s)` times in expectation** wherever
each session's V\* statement has the session's numbers (`SessionNums`), so its public shape and its `k`, and the staged
statements read only the inner and staging rounds (`ofRec_stages`). -/
theorem ofRec_EO_runs_exp (H : Msg → Msg) (D : RecAt Coin) (DO : Claims.Flock.Deploy Coin Col' T')
    (IV : Verifier Coin) (hN : SessionNums (D.record IV) DO) :
    ExtractRunsExpLe H (Compiled.ofRec D DO IV).EO fun x =>
      1 + ∑ s : Fin ((D.record IV).nS x),
        ((1 + ((D.record IV).nums x s).waits * ((D.record IV).nums x s).k : ℕ) : ℝ) := by
  refine (Compiled.ofRec D DO IV).EO_runs_exp_of H (ofRec_stages D DO IV) (fun x s => ((D.record IV).nums x s).k)
    (fun x s ms => ?_) (fun x s ms => le_of_eq ?_)
  · obtain ⟨hk, hW, hK, hN₀, -, -⟩ :=
      hN x s (ms.take (IV.k (Claims.Recursion.xI (D.record IV) x) + (D.record IV).kS x))
    exact congrArg₂ Prod.mk hW (congrArg₂ Prod.mk hK (congrArg₂ Prod.mk hk hN₀))
  · obtain ⟨-, -, -, hN₀, -, -⟩ :=
      hN x s (ms.take (IV.k (Claims.Recursion.xI (D.record IV) x) + (D.record IV).kS x))
    show DO.k (((D.record IV).vstar x (ms.take (IV.k (Claims.Recursion.xI (D.record IV) x) +
      (D.record IV).kS x))).getD s []) = ((D.record IV).nums x s).k
    unfold Claims.Flock.Deploy.k Nums.k
    rw [hN₀]

/-- **`CompiledEORunsExp` holds at every compiled record** (`ofRec_EO_runs_exp`). `SessionAccepts` is not read: each
session's replay is counted whether or not it accepts. -/
theorem compiledEORunsExp (H : Msg → Msg) (D : RecAt Coin) (DO : Claims.Flock.Deploy Coin Col' T')
    (IV : Verifier Coin) : CompiledEORunsExp H D DO IV :=
  fun hN _ => ofRec_EO_runs_exp H D DO IV hN

end Instance

end Claims.RecGeneric
