import Proofs.Flock.Claims.ERepCost

/-!
# Serving one rerun program's reruns by another's

A recursion's extractor runs an inner stage whose reruns are not the prover's: each is served by an outer stage, a
rerun program over the prover's own runs that returns what the inner stage sees of its rerun (`RP.serve`). Against the
prover's runs, the composed program

* returns what the inner stage returns against the served runs (`servedRun`: each rerun seen as the outer stage's
  output, with no SHA-512 calls of its own);
* makes the inner stage's own SHA-512 calls and, at each rerun the inner stage makes, the outer stage's (both parties'
  in every rerun it makes);
* reruns the prover as often as the outer stage does at each of the inner stage's reruns, summed (`RP.serve_eval`).

So the worst cases multiply (`RP.serve_runs_le`, `RP.serve_hashes_le`), and the expected cost is the inner stage's
reruns weighted by the outer stage's cost at each (`RP.serve_runs_cast`, which `Recursion/Wald.lean` averages).
-/

namespace Claims

/-- **Serve each rerun of `p` by the program `q r`**, which reruns at `R₂` and returns what `p` sees of its rerun at
`r`. -/
def RP.serve {R₁ R₂ α : Type} (q : R₁ → RP R₂ Seen) : RP R₁ α → RP R₂ α
  | .ret a => .ret a
  | .hash m k => .hash m fun a => RP.serve q (k a)
  | .rerun r k => (q r).bind fun v => RP.serve q (k v)

/-- **The served runs**: the rerun at `r` is seen as what `q r` returns against `run`, and logs no SHA-512 call of its
own (the outer stage's are counted where it runs). -/
def servedRun {R₁ R₂ : Type} (H : Msg → Msg) (q : R₁ → RP R₂ Seen) (run : R₂ → Trace) (r : R₁) : Trace :=
  ⟨((q r).eval H run).1.msgs, [], [], ((q r).eval H run).1.out⟩

/-- A program reruns once per draw it reruns at. -/
theorem RP.eval_runs_eq {R α : Type} (H : Msg → Msg) (run : R → Trace) (p : RP R α) :
    (p.eval H run).2.2 = (p.rerunsAt H run).length := by
  induction p with
  | ret a => rfl
  | hash q k ih => simp only [RP.eval, RP.rerunsAt]; exact ih _
  | rerun r k ih => simp only [RP.eval, RP.rerunsAt, List.length_cons]; rw [ih]

/-- A bound program's SHA-512 calls are the first part's and then the continuation's. -/
theorem RP.eval_bind_hashes {R α β : Type} (H : Msg → Msg) (run : R → Trace) (p : RP R α) (f : α → RP R β) :
    ((p.bind f).eval H run).2.1 = (p.eval H run).2.1 + ((f (p.eval H run).1).eval H run).2.1 := by
  induction p with
  | ret a => simp [RP.bind, RP.eval]
  | hash q k ih => simp only [RP.bind, RP.eval]; rw [ih]; ring
  | rerun r k ih => simp only [RP.bind, RP.eval]; rw [ih]; ring

/-- A bound program reruns at the first part's draws and then the continuation's. -/
theorem RP.rerunsAt_bind {R α β : Type} (H : Msg → Msg) (run : R → Trace) (p : RP R α) (f : α → RP R β) :
    (p.bind f).rerunsAt H run = p.rerunsAt H run ++ (f (p.eval H run).1).rerunsAt H run := by
  induction p with
  | ret a => simp [RP.bind, RP.eval, RP.rerunsAt]
  | hash q k ih => simp only [RP.bind, RP.eval, RP.rerunsAt]; exact ih _
  | rerun r k ih => simp only [RP.bind, RP.eval, RP.rerunsAt, List.cons_append]; rw [ih]

/-- What the inner stage sees of a served rerun is what the outer stage returns. -/
theorem servedRun_seen {R₁ R₂ : Type} (H : Msg → Msg) (q : R₁ → RP R₂ Seen) (run : R₂ → Trace) (r : R₁) :
    (servedRun H q run r).seen = ((q r).eval H run).1 := rfl

/-- **Serving, exactly**: against the prover's runs `run`, the served program returns the inner stage's result against
the served runs; its SHA-512 calls are the inner stage's against them plus the outer stage's at each draw the inner
stage reruns at; and its reruns are the outer stage's at each of those draws, summed. -/
theorem RP.serve_eval {R₁ R₂ α : Type} (H : Msg → Msg) (q : R₁ → RP R₂ Seen) (run : R₂ → Trace) (p : RP R₁ α) :
    ((p.serve q).eval H run).1 = (p.eval H (servedRun H q run)).1 ∧
      ((p.serve q).eval H run).2.1 = (p.eval H (servedRun H q run)).2.1 +
        ((p.rerunsAt H (servedRun H q run)).map fun r => ((q r).eval H run).2.1).sum ∧
      ((p.serve q).eval H run).2.2 =
        ((p.rerunsAt H (servedRun H q run)).map fun r => ((q r).eval H run).2.2).sum := by
  induction p with
  | ret a => simp [RP.serve, RP.eval, RP.rerunsAt]
  | hash m k ih =>
    obtain ⟨h1, h2, h3⟩ := ih (H m)
    simp only [RP.serve, RP.eval, RP.rerunsAt]
    refine ⟨h1, ?_, h3⟩
    rw [h2]; ring
  | rerun r k ih =>
    obtain ⟨h1, h2, h3⟩ := ih ((q r).eval H run).1
    simp only [RP.serve, RP.eval, RP.rerunsAt, servedRun_seen, List.map_cons, List.sum_cons]
    refine ⟨(RP.eval_bind H run _ _).1.trans h1, ?_, ?_⟩
    · rw [RP.eval_bind_hashes, h2]
      simp only [servedRun, List.length_nil]
      ring
    · rw [(RP.eval_bind H run _ _).2, h3]

/-- **The served reruns, as reals**: the outer stage's reruns at each draw the inner stage reruns at, summed. -/
theorem RP.serve_runs_cast {R₁ R₂ α : Type} (H : Msg → Msg) (q : R₁ → RP R₂ Seen) (run : R₂ → Trace)
    (p : RP R₁ α) :
    (((p.serve q).eval H run).2.2 : ℝ) =
      ((p.rerunsAt H (servedRun H q run)).map fun r => (((q r).eval H run).2.2 : ℝ)).sum := by
  rw [(RP.serve_eval H q run p).2.2, Nat.cast_list_sum, List.map_map]
  rfl

/-- **The served SHA-512 calls, as reals.** -/
theorem RP.serve_hashes_cast {R₁ R₂ α : Type} (H : Msg → Msg) (q : R₁ → RP R₂ Seen) (run : R₂ → Trace)
    (p : RP R₁ α) :
    (((p.serve q).eval H run).2.1 : ℝ) = ((p.eval H (servedRun H q run)).2.1 : ℝ) +
      ((p.rerunsAt H (servedRun H q run)).map fun r => (((q r).eval H run).2.1 : ℝ)).sum := by
  rw [(RP.serve_eval H q run p).2.1, Nat.cast_add, Nat.cast_list_sum, List.map_map]
  rfl

/-- A sum over a list of terms each at most `N` is at most its length times `N`. -/
theorem list_sum_le_mul {R : Type} (l : List R) (f : R → ℕ) (N : ℕ) (h : ∀ r ∈ l, f r ≤ N) :
    (l.map f).sum ≤ l.length * N := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [List.map_cons, List.sum_cons, List.length_cons]
    have := ih fun r hr => h r (List.mem_cons_of_mem a hr)
    have := h a List.mem_cons_self
    nlinarith

/-- **The worst cases multiply**: an inner stage of at most `N₁` reruns against the served runs, each served by at most
`N₂` reruns, makes at most `N₁ · N₂`. -/
theorem RP.serve_runs_le {R₁ R₂ α : Type} (H : Msg → Msg) (q : R₁ → RP R₂ Seen) (run : R₂ → Trace)
    (p : RP R₁ α) {N₁ N₂ : ℕ} (hp : (p.eval H (servedRun H q run)).2.2 ≤ N₁)
    (hq : ∀ r, ((q r).eval H run).2.2 ≤ N₂) : ((p.serve q).eval H run).2.2 ≤ N₁ * N₂ := by
  rw [(RP.serve_eval H q run p).2.2]
  refine (list_sum_le_mul _ _ N₂ fun r _ => hq r).trans ?_
  rw [← RP.eval_runs_eq]
  exact Nat.mul_le_mul_right N₂ hp

/-- **The SHA-512 calls' worst case**: the inner stage's own, and at most `N_h` for each of its at most `N₁` reruns. -/
theorem RP.serve_hashes_le {R₁ R₂ α : Type} (H : Msg → Msg) (q : R₁ → RP R₂ Seen) (run : R₂ → Trace)
    (p : RP R₁ α) {N₁ Nh : ℕ} (hp : (p.eval H (servedRun H q run)).2.2 ≤ N₁)
    (hq : ∀ r, ((q r).eval H run).2.1 ≤ Nh) :
    ((p.serve q).eval H run).2.1 ≤ (p.eval H (servedRun H q run)).2.1 + N₁ * Nh := by
  rw [(RP.serve_eval H q run p).2.1]
  refine Nat.add_le_add_left ((list_sum_le_mul _ _ Nh fun r _ => hq r).trans ?_) _
  rw [← RP.eval_runs_eq]
  exact Nat.mul_le_mul_right Nh hp

/-- Running a program followed by a return adds no rerun and no SHA-512 call. -/
theorem RP.eval_bind_ret {R α β : Type} (H : Msg → Msg) (run : R → Trace) (p : RP R α) (f : α → β) :
    ((p.bind fun a => .ret (f a)).eval H run) = (f (p.eval H run).1, (p.eval H run).2.1, (p.eval H run).2.2) := by
  induction p with
  | ret a => rfl
  | hash q k ih => simp only [RP.bind, RP.eval]; rw [ih]
  | rerun r k ih => simp only [RP.bind, RP.eval]; rw [ih]

end Claims
