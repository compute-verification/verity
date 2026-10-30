import Proofs.Flock.Recursive.KSNontrivial

/-!
# What `E_O` returns at a draw passes each session's check

The link clause's binding reads the link part's outer witness at `E*`'s judged draw. `EO_checks`: where `E_O` returns
outer witnesses `ws` at a draw, the draw's run accepted, `ws` holds one witness per V\* session, and witness `s` passes
the outer Flock's `check` at session `s`'s statement, read off the run's inner and staging rounds. `firstWait` returns
only a candidate its `check` accepts (`eval_serve_firstWait_check`), and `collect` returns each program's result in
order (`eval_collect_map`).
-/

namespace Claims.Recursion

open Claims

/-- **Waits return only a checked candidate**, however their reruns are served. -/
theorem eval_serve_firstWait_check {R₁ R₂ W' Op : Type} (q : R₁ → RP R₂ Seen) (k : ℕ) (opens : Seen → List Op)
    (assemble : List Op → Option W') (check : W' → Bool) (H : Msg → Msg) (run : R₂ → Trace) :
    ∀ (rss : List (List R₁)) {w : W'}, ((serve q (firstWait k opens assemble check rss)).eval H run).1 = some w →
      check w = true
  | [], _, h => by simp [firstWait, serve, RP.eval] at h
  | rs :: rss, w, h => by
    simp only [firstWait] at h
    rw [Witness.Recursion.serve_bind, eval_bind_fst] at h
    generalize ((serve q (waitAcc k rs [])).eval H run).1 = vs at h
    dsimp only at h
    split at h
    · next w' _ =>
      by_cases hc : check w' = true
      · rw [if_pos hc] at h
        have e : some w' = some w := h
        exact Option.some.inj e ▸ hc
      · rw [if_neg hc] at h
        exact eval_serve_firstWait_check q k opens assemble check H run rss h
    · exact eval_serve_firstWait_check q k opens assemble check H run rss h

/-- **`collect` returns each program's result**, in order: where it returns `os` over `l.map g`, `os` has `l`'s length
and its `i`-th entry is `g l[i]`'s result. -/
theorem eval_collect_map {R α β : Type} (H : Msg → Msg) (run : R → Trace) (g : β → RP R (Option α)) :
    ∀ (l : List β) {os : List α}, ((collect (l.map g)).eval H run).1 = some os →
      os.length = l.length ∧ ∀ i (h1 : i < l.length) (h2 : i < os.length), ((g l[i]).eval H run).1 = some os[i]
  | [], os, h => by
    have e : some ([] : List α) = some os := h
    cases Option.some.inj e
    exact ⟨rfl, fun i h1 => absurd h1 (Nat.not_lt_zero i)⟩
  | b :: l, os, h => by
    simp only [List.map_cons, collect] at h
    rw [eval_bind_fst] at h
    cases hb : ((g b).eval H run).1 with
    | none =>
      rw [hb] at h
      simp [RP.eval] at h
    | some a =>
      rw [hb] at h
      dsimp only at h
      rw [eval_bind_fst] at h
      cases hc : ((collect (l.map g)).eval H run).1 with
      | none =>
        rw [hc] at h
        simp [RP.eval] at h
      | some os' =>
        rw [hc] at h
        have e : some (a :: os') = some os := h
        cases Option.some.inj e
        obtain ⟨hl, he⟩ := eval_collect_map H run g l hc
        refine ⟨by rw [List.length_cons, List.length_cons, hl], fun i h1 h2 => ?_⟩
        cases i with
        | zero => exact hb
        | succ i => exact he i (by simpa using h1) (by simpa using h2)

section EO

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- **`E_O`'s witnesses pass their sessions' checks**: where `E_O` returns `ws` at the draw `r₀`, the run there
accepted, `ws` has one witness per V\* session, and witness `s` passes the outer Flock's `check` at session `s`'s
statement (the deployment's `vstar` at the run's inner and staging rounds). -/
theorem EO_checks (H : Msg → Msg) (A : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (r₀ : Fin (A.V.k x) → Coin)
    (c : (EO A I DO decO).C x) {ws : List Flock.Wit} (h : ((EO A I DO decO).eval H P x ω r₀ c).1 = some ws) :
    (run H P A.V x ω r₀).out = .acc ∧ ws.length = A.nS x ∧
      ∀ s (hs : s < ws.length), Flock.check DO.parse
        ((A.vstar x ((run H P A.V x ω r₀).msgs.take (I.V.k (xI A x) + A.kS x))).getD s []) ws[s] = true := by
  unfold Extractor.eval EO at h
  simp only [RP.eval, Trace.seen] at h
  split at h
  · next hacc =>
    obtain ⟨hl, he⟩ := eval_collect_map H (run H P A.V x ω) _ (List.finRange (A.nS x)) h
    rw [List.length_finRange] at hl
    refine ⟨hacc, hl, fun s hs => ?_⟩
    have h1 : s < (List.finRange (A.nS x)).length := by rw [List.length_finRange, ← hl]; exact hs
    have hv : ((List.finRange (A.nS x))[s]'h1).val = s := by simp
    have := he s h1 hs
    beta_reduce at this
    rw [hv] at this
    exact eval_serve_firstWait_check _ _ _ _ _ H _ _ this
  · simp [RP.eval] at h

end EO

end Claims.Recursion
