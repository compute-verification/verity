import Proofs.Flock.Claims.Recursion.FlockKSAbort
import Proofs.Flock.Claims.Recursion.Hybrid
import Proofs.Flock.Claims.Bridge.Coins

/-!
# The abort-law table bound from the table bound with escapes on accepted runs

`TableBoundθAbort` from `TableBoundθ` with escapes on accepted runs (`TableBoundθOnAcc`) and four code facts of the
verifier: its messages show each round's coin (`CoinsShown`, `PublicCoin` at the verifier), it accepts only at its last
round (`AccLast`), it decides without the coin of the round it decides at (`DecideFree`), and it never accepts some
message (`Refusable`). So Flock meets `Inner.KSAbort` at `Inner.ofFlock` from the KS chain's obligations at Flock's own
`κ` and `γ` (`KSAbort_of_tableBoundOnAcc`), and the recursion's KS over Flock rests on them (`KSFlock_of_flockKS`).

* **The refuser** (`refuser`): at a tape `ω` and a function `g` from the coins before the last round to a last coin and
  abort coins, it sends `P`'s messages, except at the last round, where it sends the message the verifier never accepts
  unless `keep` keeps the coins its history shows with `g`'s last coin and abort coins. Its run at coins `r` accepts
  exactly where the run at the draw `(r with g's last coin, g's abort coins)` accepts as it aborts, and is that run
  there (`refuser_run`): kept, it is `P`'s run (`stepWith_refuser_keep`), which accepts at `r` exactly where it does at
  `r` with any last coin, and is the same run there (`run_eq_of_acc`); refused, the verifier rejects its last message
  (`stepWith_refuser_rej`).
* **The coupling** (`Pr_refDraw`, `Pr_fork_refDraw_le`): over a uniform `g`, that draw is uniform, and two forked runs'
  draws are `forkDraw`'s, except where the two runs share their coins before the last round: there they share `g`'s
  value, and a collision among one run's calls twice over is one among that run's.
* **The bound** (`tableBoundθAbort_of_onAcc`): the refuser's bound, averaged over `g`, with its escapes on accepted runs
  read at the runs as they abort (each fork event's root by Jensen).
-/

namespace Claims.Recursion

open Claims Claims.Flock

/-! ## Code facts of the verifier -/

section CodeFacts

variable {Coin : Type}

/-- **The verifier shows each round's coin** (a code fact; `PublicCoin` at the verifier): its message at a round shows
that round's coin, which `decC` reads off it. -/
def CoinsShown (H : Msg → Msg) (V : Verifier Coin) (decC : Msg → Msg → Coin) : Prop :=
  ∀ (x : Msg) (as : List Msg) (cs : List Coin) (c : Coin) (m : Msg), as.length = cs.length + 1 →
    ((V.next x as (cs ++ [c])).eval H).1 = .msg m → decC x m = c

/-- **The verifier accepts only at its last round** (a code fact): wherever it accepts, with one coin per message, the
prover sent `k(x)` messages. -/
def AccLast (H : Msg → Msg) (V : Verifier Coin) : Prop :=
  ∀ (x : Msg) (as : List Msg) (cs : List Coin), cs.length = as.length → ((V.next x as cs).eval H).1 = .acc →
    as.length = V.k x

/-- **The verifier decides without the round's coin** (a code fact): where it accepts at a round, it does the same,
calls and all, at any other coin of that round. -/
def DecideFree (H : Msg → Msg) (V : Verifier Coin) : Prop :=
  ∀ (x : Msg) (as : List Msg) (cs : List Coin) (c c' : Coin), as.length = cs.length + 1 →
    ((V.next x as (cs ++ [c])).eval H).1 = .acc →
      (V.next x as (cs ++ [c'])).eval H = (V.next x as (cs ++ [c])).eval H

/-- **A message the verifier never accepts** (a code fact), at each statement. -/
def Refusable (H : Msg → Msg) (V : Verifier Coin) : Prop :=
  ∀ x : Msg, ∃ m₀ : Msg, ∀ (as : List Msg) (cs : List Coin), ((V.next x (as ++ [m₀]) cs).eval H).1 ≠ .acc

theorem coinsShown_of_publicCoin [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {Col T : Type} [DecidableEq Col]
    {H : Msg → Msg} {D : Flock.Deploy Coin Col T} {decode : List (ℕ × Col) → List T} {decC : Msg → Msg → Coin}
    (h : PublicCoin H (Inner.ofFlock D decode) decC) : CoinsShown H D.V decC :=
  h

end CodeFacts

/-! ## The last round's coin -/

section LastCoin

variable {Coin : Type} {n : ℕ}

/-- The coins `r` with `c` as the last round's coin. -/
def setLast (r : Fin n → Coin) (c : Coin) : Fin n → Coin := fun i => if i.val + 1 < n then r i else c

/-- The draw at coins `r` with the last coin and the abort coins `y`. -/
def drawOf {A : Type} (r : Fin n → Coin) (y : Coin × A) : (Fin n → Coin) × A := (setLast r y.1, y.2)

theorem setLast_congr {r r' : Fin n → Coin} {c₀ : Coin} (h : setLast r c₀ = setLast r' c₀) (c : Coin) :
    setLast r c = setLast r' c := by
  funext i
  have hi := congrFun h i
  unfold setLast at hi ⊢
  by_cases h1 : i.val + 1 < n
  · rw [ite_eq_left h1, ite_eq_left h1] at hi ⊢
    exact hi
  · rw [ite_eq_right h1, ite_eq_right h1]

theorem forkCoins_setLast (j : Fin n) (r r' : Fin n → Coin) (c c' : Coin) :
    forkCoins j (setLast r c) (setLast r' c') = setLast (forkCoins j r r') c' := by
  funext i
  unfold forkCoins setLast
  have hj := j.isLt
  by_cases h1 : i.val + 1 < n
  · simp only [h1, ↓reduceIte]
  · have h2 : ¬ i.val < j.val := by omega
    simp only [h1, h2, ↓reduceIte]

/-- The coins with the last coin `c`, and the coins' last coin. -/
def swapLast (hn : 0 < n) (p : (Fin n → Coin) × Coin) : (Fin n → Coin) × Coin :=
  (setLast p.1 p.2, p.1 ⟨n - 1, by omega⟩)

theorem swapLast_involutive (hn : 0 < n) : Function.Involutive (swapLast (Coin := Coin) hn) := by
  rintro ⟨r, c⟩
  unfold swapLast setLast
  refine Prod.ext (funext fun i => ?_) ?_
  · by_cases h1 : i.val + 1 < n
    · simp only [h1, ↓reduceIte]
    · simp only [h1, ↓reduceIte]
      exact congrArg r (Fin.ext (by simp only; omega))
  · simp only [show ¬ n - 1 + 1 < n by omega, ↓reduceIte]

/-- **The last coin redrawn is uniform**: averaging over the coins and a fresh last coin is averaging over the coins. -/
theorem Ex_setLast [Fintype Coin] [Nonempty Coin] (hn : 0 < n) (h : (Fin n → Coin) → ℝ) :
    (Ex fun r : Fin n → Coin => Ex fun c : Coin => h (setLast r c)) = Ex h := by
  rw [← Ex_prod (fun p : (Fin n → Coin) × Coin => h (setLast p.1 p.2))]
  have he := Ex_equiv (Function.Involutive.toPerm _ (swapLast_involutive (Coin := Coin) hn))
    (fun p : (Fin n → Coin) × Coin => h p.1)
  refine he.trans ?_
  rw [Ex_prod]
  exact congrArg Ex (funext fun r => Ex_const (h r))

/-- **The draw at coins with a fresh last coin and abort coins is uniform.** -/
theorem Ex_drawOf {A : Type} [Fintype Coin] [Nonempty Coin] [Fintype A] (hn : 0 < n)
    (k : (Fin n → Coin) × A → ℝ) :
    (Ex fun r : Fin n → Coin => Ex fun y : Coin × A => k (drawOf r y)) = Ex k :=
  calc (Ex fun r : Fin n → Coin => Ex fun y : Coin × A => k (drawOf r y))
      = Ex fun r : Fin n → Coin => Ex fun c : Coin => Ex fun a : A => k (setLast r c, a) :=
        congrArg Ex (funext fun r => Ex_prod fun y : Coin × A => k (drawOf r y))
    _ = Ex fun r' : Fin n → Coin => Ex fun a : A => k (r', a) :=
        Ex_setLast hn fun r' : Fin n → Coin => Ex fun a : A => k (r', a)
    _ = Ex k := (Ex_prod k).symm

end LastCoin

/-! ## Random functions -/

section RandomFun

variable {α β : Type} [Fintype α] [DecidableEq α] [Fintype β] [Nonempty β]

/-- **A random function's value at a point is uniform.** -/
theorem Ex_fun_at (p : α) (f : β → ℝ) : (Ex fun g : α → β => f (g p)) = Ex f :=
  Ex_pi_coord (C := fun _ => β) p f

/-- **A random function's values at two points are independent and uniform.** -/
theorem Ex_fun_at₂ {p q : α} (hpq : p ≠ q) (f : β → β → ℝ) :
    (Ex fun g : α → β => f (g p) (g q)) = Ex fun y => Ex fun y' => f y y' := by
  let e : Fin 2 ↪ α := ⟨![p, q], fun i j h => by
    fin_cases i <;> fin_cases j <;> simp_all [Ne.symm hpq]⟩
  have h := UnifMap.comp_emb (γ := β) e (fun ρ : Fin 2 → β => f (ρ 0) (ρ 1))
  refine h.trans ?_
  have hc := Ex_cons 1 fun (y : β) (σ : Fin 1 → β) => f y (σ 0)
  refine hc.trans (congrArg Ex (funext fun y => ?_))
  exact Ex_fun_at (α := Fin 1) 0 (f y)

end RandomFun

/-! ## The refuser -/

section Refuser

variable {Ω Coin A : Type} [Inhabited Coin]

/-- The coins the history `hist` shows at its rounds, and `c` at the rest. -/
def histCoins (decC : Msg → Msg → Coin) (x : Msg) (n : ℕ) (hist : List (Msg × Msg)) (c : Coin) : Fin n → Coin :=
  fun i => if i.val < hist.length then decC x (hist.getD i.val ([], [])).2 else c

/-- **The refuser**: at the tape `ω` and `g`, `P`'s messages, except at the last round (of `n`), the message `m₀` unless
`keep` keeps the coins the history shows with `g`'s last coin and `g`'s abort coins, `g` read at the coins the history
shows. -/
def refuser (P : Prover Ω) (x : Msg) (n : ℕ) (decC : Msg → Msg → Coin) (keep : Ω → (Fin n → Coin) → A → Bool)
    (m₀ : Msg) : Prover (Ω × ((Fin n → Coin) → Coin × A)) where
  next x' hist ω :=
    if hist.length + 1 = n ∧ keep ω.1 (histCoins decC x n hist (ω.2 (histCoins decC x n hist default)).1)
        (ω.2 (histCoins decC x n hist default)).2 = false then .ret m₀
    else P.next x' hist ω.1

/-- The refuser's draw at coins `r`: `r` with `g`'s last coin, and `g`'s abort coins, `g` read at `r` before its last
coin. -/
def refDraw {n : ℕ} (g : (Fin n → Coin) → Coin × A) (r : Fin n → Coin) : (Fin n → Coin) × A :=
  drawOf r (g (setLast r default))

/-- What the rounds keep at coins `r`: `M` of `n` rounds left, a message per round, and the coins the history shows are
`r`'s. -/
def RefInv (decC : Msg → Msg → Coin) (x : Msg) {n : ℕ} (r : Fin n → Coin) (M : ℕ) (hist : List (Msg × Msg))
    (as : List Msg) : Prop :=
  hist.length + M = n ∧ as.length = hist.length ∧
    ∀ i < hist.length, decC x (hist.getD i ([], [])).2 = coinSeq r i

theorem refInv_start (decC : Msg → Msg → Coin) (x : Msg) {n : ℕ} (r : Fin n → Coin) : RefInv decC x r n [] [] :=
  ⟨by simp, rfl, fun i hi => absurd hi (Nat.not_lt_zero i)⟩

theorem histCoins_eq {decC : Msg → Msg → Coin} {x : Msg} {n : ℕ} {r : Fin n → Coin} {M : ℕ}
    {hist : List (Msg × Msg)} {as : List Msg} (h : RefInv decC x r M hist as) (hl : hist.length + 1 = n) (c : Coin) :
    histCoins decC x n hist c = setLast r c := by
  funext i
  unfold histCoins setLast
  have hi : i.val < hist.length ↔ i.val + 1 < n := by omega
  by_cases h1 : i.val + 1 < n
  · rw [ite_eq_left (hi.2 h1), ite_eq_left h1, h.2.2 i.val (hi.2 h1)]
    unfold coinSeq
    rw [dite_eq_left i.isLt]
  · rw [ite_eq_right (fun h' => h1 (hi.1 h')), ite_eq_right h1]

theorem refInv_step {decC : Msg → Msg → Coin} {x : Msg} {n : ℕ} {r : Fin n → Coin} {M : ℕ}
    {hist : List (Msg × Msg)} {as : List Msg} (h : RefInv decC x r (M + 1) hist as) (a m : Msg)
    (hm : decC x m = coinSeq r as.length) : RefInv decC x r M (hist ++ [(a, m)]) (as ++ [a]) := by
  obtain ⟨h1, h2, h3⟩ := h
  refine ⟨by simp only [List.length_append, List.length_singleton]; omega,
    by simp only [List.length_append, List.length_singleton, h2], fun i hi => ?_⟩
  simp only [List.length_append, List.length_singleton] at hi
  rcases Nat.lt_succ_iff_lt_or_eq.1 hi with hi | hi
  · rw [List.getD_append _ _ _ _ hi]
    exact h3 i hi
  · subst hi
    rw [List.getD_append_right _ _ _ _ le_rfl, Nat.sub_self]
    rw [← h2]
    exact hm

/-- **A kept refuser is `P`**: where `keep` keeps the refuser's draw at `r`, the refuser's rounds at `r` are `P`'s. -/
theorem stepWith_refuser_keep (H : Msg → Msg) (P : Prover Ω) (x : Msg) {n : ℕ} (decC : Msg → Msg → Coin)
    (keep : Ω → (Fin n → Coin) → A → Bool) (m₀ : Msg) (ω : Ω) (g : (Fin n → Coin) → Coin × A) (r : Fin n → Coin)
    (vn : List Msg → OC Out)
    (hshow : ∀ (as : List Msg) (a m : Msg), ((vn (as ++ [a])).eval H).1 = .msg m → decC x m = coinSeq r as.length)
    (hk : keep ω (refDraw g r).1 (refDraw g r).2 = true) :
    ∀ M hist as t, RefInv decC x r M hist as →
      stepWith H (refuser P x n decC keep m₀) x (ω, g) vn M hist as t = stepWith H P x ω vn M hist as t :=
  stepWith_congr H (refuser P x n decC keep m₀) P x (ω, g) ω vn (fun M hist as => RefInv decC x r M hist as)
    (fun M hist as h => by
      have hn : (refuser P x n decC keep m₀).next x hist (ω, g) = P.next x hist ω := by
        simp only [refuser]
        refine ite_eq_right fun ⟨hl, hk'⟩ => ?_
        simp only [histCoins_eq h hl] at hk'
        unfold refDraw drawOf at hk
        rw [hk] at hk'
        exact Bool.noConfusion hk'
      rw [hn])
    (fun M hist as m h hv => refInv_step h _ m (hshow as _ m hv))

/-- **A refused refuser is rejected**: where `keep` refuses the refuser's draw at `r`, the refuser's run at `r` doesn't
accept: the verifier accepts no earlier round, and not the message `m₀` it sends at the last. -/
theorem stepWith_refuser_rej (H : Msg → Msg) (P : Prover Ω) (x : Msg) {n : ℕ} (decC : Msg → Msg → Coin)
    (keep : Ω → (Fin n → Coin) → A → Bool) (m₀ : Msg) (ω : Ω) (g : (Fin n → Coin) → Coin × A) (r : Fin n → Coin)
    (vn : List Msg → OC Out)
    (hshow : ∀ (as : List Msg) (a m : Msg), ((vn (as ++ [a])).eval H).1 = .msg m → decC x m = coinSeq r as.length)
    (hlast : ∀ as : List Msg, ((vn as).eval H).1 = .acc → as.length = n)
    (href : ∀ as : List Msg, ((vn (as ++ [m₀])).eval H).1 ≠ .acc)
    (hk : keep ω (refDraw g r).1 (refDraw g r).2 = false) :
    ∀ M hist as t, RefInv decC x r M hist as →
      (stepWith H (refuser P x n decC keep m₀) x (ω, g) vn M hist as t).out ≠ .acc
  | 0, _, _, _, _ => by simp [stepWith]
  | M + 1, hist, as, t, h => by
    simp only [stepWith]
    by_cases hl : hist.length + 1 = n
    · have hnext : (refuser P x n decC keep m₀).next x hist (ω, g) = .ret m₀ := by
        simp only [refuser]
        refine ite_eq_left ⟨hl, ?_⟩
        simp only [histCoins_eq h hl]
        exact hk
      have hM : M = 0 := by have := h.1; omega
      subst hM
      rw [hnext]
      simp only [OC.eval]
      rcases hv : ((vn (as ++ [m₀])).eval H).1 with m | _ | _
      · simp [stepWith]
      · exact absurd hv (href as)
      · simp
    · generalize hpa : (refuser P x n decC keep m₀).next x hist (ω, g) = pa
      rcases hv : ((vn (as ++ [(pa.eval H).1])).eval H).1 with m | _ | _
      · exact stepWith_refuser_rej H P x decC keep m₀ ω g r vn hshow hlast href hk M _ _ _
          (refInv_step h _ m (hshow as _ m hv))
      · exfalso
        have h1 := hlast _ hv
        simp only [List.length_append, List.length_singleton] at h1
        have := h.1
        have := h.2.1
        omega
      · simp

/-- **Rounds against two verifiers that agree before the last round, and at it where the first accepts**, are the same
where the first accepts. -/
theorem stepWith_vn_congr (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (n : ℕ) (vn₁ vn₂ : List Msg → OC Out)
    (hlt : ∀ as : List Msg, as.length < n → (vn₂ as).eval H = (vn₁ as).eval H)
    (hn : ∀ as : List Msg, as.length = n → ((vn₁ as).eval H).1 = .acc → (vn₂ as).eval H = (vn₁ as).eval H) :
    ∀ M hist (as : List Msg) t, as.length + M = n → (stepWith H P x ω vn₁ M hist as t).out = .acc →
      stepWith H P x ω vn₂ M hist as t = stepWith H P x ω vn₁ M hist as t
  | 0, _, _, _, _ => by simp [stepWith]
  | M + 1, hist, as, t, hl => by
    simp only [stepWith]
    generalize (P.next x hist ω).eval H = pa
    by_cases hlt' : as.length + 1 < n
    · rw [hlt (as ++ [pa.1]) (by simp only [List.length_append, List.length_singleton]; exact hlt')]
      rcases hv : ((vn₁ (as ++ [pa.1])).eval H).1 with m | _ | _
      · exact stepWith_vn_congr H P x ω n vn₁ vn₂ hlt hn M _ _ _
          (by simp only [List.length_append, List.length_singleton]; omega)
      · exact fun _ => rfl
      · exact fun _ => rfl
    · have hlen : (as ++ [pa.1]).length = n := by
        simp only [List.length_append, List.length_singleton]; omega
      by_cases hacc : ((vn₁ (as ++ [pa.1])).eval H).1 = .acc
      · rw [hn _ hlen hacc]
        intro _
        rw [hacc]
      · have hM : M = 0 := by omega
        subst hM
        rcases hv : ((vn₁ (as ++ [pa.1])).eval H).1 with m | _ | _
        · intro h
          simp [stepWith] at h
        · exact absurd hv hacc
        · intro h
          simp at h

/-- **An accepted run is the run at its coins with any last coin**, where the verifier decides without the round's
coin. -/
theorem run_eq_of_acc (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin) (x : Msg) (hfree : DecideFree H V) (ω : Ω)
    {r r' : Fin (V.k x) → Coin} (hr : ∀ i : Fin (V.k x), i.val + 1 < V.k x → r' i = r i)
    (h : (run H P V x ω r).out = .acc) : run H P V x ω r' = run H P V x ω r := by
  have hcs : ∀ L, L + 1 ≤ V.k x → (List.range L).map (coinSeq r') = (List.range L).map (coinSeq r) := by
    intro L hL
    refine List.map_congr_left fun i hi => ?_
    have hi' := List.mem_range.1 hi
    unfold coinSeq
    rw [dite_eq_left (by omega), dite_eq_left (by omega)]
    exact hr ⟨i, by omega⟩ (by simp only; omega)
  refine stepWith_vn_congr H P x ω (V.k x) _ _ (fun as hl => ?_) (fun as hl ha => ?_) (V.k x) [] [] _
    (by simp) h
  · rw [hcs as.length (by omega)]
  · rcases Nat.eq_zero_or_pos (V.k x) with h0 | hpos
    · have : as.length = 0 := by omega
      rw [this, List.range_zero, List.map_nil, List.map_nil]
    · obtain ⟨L, hL⟩ : ∃ L, as.length = L + 1 := ⟨as.length - 1, by omega⟩
      have e1 : (List.range as.length).map (coinSeq r) = (List.range L).map (coinSeq r) ++ [coinSeq r L] := by
        rw [hL, List.range_succ, List.map_append, List.map_singleton]
      have e2 : (List.range as.length).map (coinSeq r') = (List.range L).map (coinSeq r) ++ [coinSeq r' L] := by
        rw [hL, List.range_succ, List.map_append, List.map_singleton, hcs L (by omega)]
      have key := hfree x as ((List.range L).map (coinSeq r)) (coinSeq r L) (coinSeq r' L) (by simp [hL])
        (by rw [← e1]; exact ha)
      rw [← e1, ← e2] at key
      exact key

/-- **The refuser's run is the run as it aborts at the refuser's draw** where either accepts, and they accept
together. -/
theorem refuser_run (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin) (x : Msg) (decC : Msg → Msg → Coin)
    (hpc : CoinsShown H V decC) (hlast : AccLast H V) (hfree : DecideFree H V)
    (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) (m₀ : Msg)
    (hm₀ : ∀ (as : List Msg) (cs : List Coin), ((V.next x (as ++ [m₀]) cs).eval H).1 ≠ .acc)
    (ω : Ω) (g : (Fin (V.k x) → Coin) → Coin × A) (r : Fin (V.k x) → Coin) :
    ((run H (refuser P x (V.k x) decC keep m₀) V x (ω, g) r).out = .acc ↔
      (abortRun H P V x keep ω (refDraw g r)).out = .acc) ∧
    ((run H (refuser P x (V.k x) decC keep m₀) V x (ω, g) r).out = .acc →
      run H (refuser P x (V.k x) decC keep m₀) V x (ω, g) r = abortRun H P V x keep ω (refDraw g r)) := by
  have hshow : ∀ (as : List Msg) (a m : Msg),
      ((V.next x (as ++ [a]) ((List.range (as ++ [a]).length).map (coinSeq r))).eval H).1 = .msg m →
        decC x m = coinSeq r as.length := by
    intro as a m hv
    rw [List.length_append, List.length_singleton, List.range_succ, List.map_append,
      List.map_singleton] at hv
    exact hpc x _ _ _ m (by simp) hv
  have hagree : ∀ i : Fin (V.k x), i.val + 1 < V.k x → (refDraw g r).1 i = r i := fun i hi => by
    simp only [refDraw, drawOf, setLast, hi, ↓reduceIte]
  by_cases hk : keep ω (refDraw g r).1 (refDraw g r).2 = true
  · have hrun : run H (refuser P x (V.k x) decC keep m₀) V x (ω, g) r = run H P V x ω r :=
      stepWith_refuser_keep H P x decC keep m₀ ω g r
        (fun as => V.next x as ((List.range as.length).map (coinSeq r))) hshow hk (V.k x) [] []
        ⟨[], [], [], .rej⟩ (refInv_start decC x r)
    rw [hrun, abortRun_of_keep hk]
    refine ⟨⟨fun h => ?_, fun h => ?_⟩, fun h => (run_eq_of_acc H P V x hfree ω hagree h).symm⟩
    · rw [run_eq_of_acc H P V x hfree ω hagree h]
      exact h
    · rw [run_eq_of_acc H P V x hfree ω (fun i hi => (hagree i hi).symm) h]
      exact h
  · have hk' : keep ω (refDraw g r).1 (refDraw g r).2 = false := by simpa using hk
    have hrej : (run H (refuser P x (V.k x) decC keep m₀) V x (ω, g) r).out ≠ .acc :=
      stepWith_refuser_rej H P x decC keep m₀ ω g r
        (fun as => V.next x as ((List.range as.length).map (coinSeq r))) hshow
        (fun as ha => hlast x as ((List.range as.length).map (coinSeq r)) (by simp) ha)
        (fun as => hm₀ as ((List.range (as ++ [m₀]).length).map (coinSeq r))) hk' (V.k x) [] []
        ⟨[], [], [], .rej⟩ (refInv_start decC x r)
    have hab : (abortRun H P V x keep ω (refDraw g r)).out = .rej := by
      unfold abortRun
      rw [ite_eq_right hk]
      rfl
    refine ⟨⟨fun h => absurd h hrej, fun h => ?_⟩, fun h => absurd h hrej⟩
    rw [hab] at h
    cases h

end Refuser

/-! ## The coupling -/

section Coupling

variable {Coin A : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [Fintype A] [Nonempty A] {n : ℕ}

/-- **The refuser's draw is uniform** over a uniform `g`. -/
theorem Ex_refDraw [DecidableEq (Fin n → Coin)] (hn : 0 < n) (k : (Fin n → Coin) × A → ℝ) :
    (Ex fun g : (Fin n → Coin) → Coin × A => Ex fun r : Fin n → Coin => k (refDraw g r)) = Ex k :=
  calc (Ex fun g : (Fin n → Coin) → Coin × A => Ex fun r : Fin n → Coin => k (refDraw g r))
      = Ex fun r : Fin n → Coin => Ex fun g : (Fin n → Coin) → Coin × A => k (refDraw g r) :=
        Claims.Bridge.Ex_comm fun (g : (Fin n → Coin) → Coin × A) (r : Fin n → Coin) => k (refDraw g r)
    _ = Ex fun r : Fin n → Coin => Ex fun y : Coin × A => k (drawOf r y) :=
        congrArg Ex (funext fun r => Ex_fun_at (setLast r default) fun y => k (drawOf r y))
    _ = Ex k := Ex_drawOf hn k

theorem Pr_refDraw [DecidableEq (Fin n → Coin)] (hn : 0 < n) (p : (Fin n → Coin) × A → Prop) :
    (Ex fun g : (Fin n → Coin) → Coin × A => Pr fun r : Fin n → Coin => p (refDraw g r)) = Pr p := by
  classical
  calc (Ex fun g : (Fin n → Coin) → Coin × A => Pr fun r : Fin n → Coin => p (refDraw g r))
      = Ex fun g : (Fin n → Coin) → Coin × A => Ex fun r : Fin n → Coin =>
          if p (refDraw g r) then (1 : ℝ) else 0 :=
        congrArg Ex (funext fun g => (Ex_ind fun r : Fin n → Coin => p (refDraw g r)).symm)
    _ = Ex fun d : (Fin n → Coin) × A => if p d then (1 : ℝ) else 0 :=
        Ex_refDraw hn fun d => if p d then (1 : ℝ) else 0
    _ = Pr p := Ex_ind p

/-- A collision among one run's calls twice over is one among its calls and any others'. -/
theorem TiedColl.of_dup {H : Msg → Msg} {l : List (Msg × Msg)} (h : TiedColl H (l ++ l)) (l' : List (Msg × Msg)) :
    TiedColl H (l ++ l') := by
  obtain ⟨u, v, hc, hu, hv⟩ := h
  simp only [List.map_append, List.mem_append, or_self] at hu hv
  exact TiedColl.append_right ⟨u, v, hc, hu, hv⟩ l'

/-- **Two forked refuser draws are at most `forkDraw`'s** over a uniform `g`, for a value at least as large at any
second draw as at the first draw twice: where the two runs' coins before the last round differ, `g`'s values there are
independent, and where they agree, the two draws are one. -/
theorem Ex_fork_refDraw_le [DecidableEq (Fin n → Coin)] (j : Fin n)
    (F : (Fin n → Coin) × A → (Fin n → Coin) × A → ℝ) (hF : ∀ d d', F d d ≤ F d d') :
    (Ex fun g : (Fin n → Coin) → Coin × A => Ex fun s : (Fin n → Coin) × (Fin n → Coin) =>
        F (refDraw g s.1) (refDraw g (forkCoins j s.1 s.2))) ≤
      Ex fun s : ((Fin n → Coin) × A) × ((Fin n → Coin) × A) => F s.1 (forkDraw j s.1 s.2) := by
  have hn : 0 < n := j.pos
  have hpt : ∀ s : (Fin n → Coin) × (Fin n → Coin),
      (Ex fun g : (Fin n → Coin) → Coin × A => F (refDraw g s.1) (refDraw g (forkCoins j s.1 s.2))) ≤
        Ex fun y : Coin × A => Ex fun y' : Coin × A => F (drawOf s.1 y) (drawOf (forkCoins j s.1 s.2) y') := by
    intro s
    by_cases hp : setLast s.1 default = setLast (forkCoins j s.1 s.2) default
    · have hd : ∀ g : (Fin n → Coin) → Coin × A, refDraw g (forkCoins j s.1 s.2) = refDraw g s.1 := fun g => by
        unfold refDraw drawOf
        rw [← hp]
        exact Prod.ext (setLast_congr hp _).symm rfl
      calc (Ex fun g : (Fin n → Coin) → Coin × A => F (refDraw g s.1) (refDraw g (forkCoins j s.1 s.2)))
          = Ex fun g : (Fin n → Coin) → Coin × A => F (refDraw g s.1) (refDraw g s.1) :=
            congrArg Ex (funext fun g => congrArg (F (refDraw g s.1)) (hd g))
        _ = Ex fun y : Coin × A => F (drawOf s.1 y) (drawOf s.1 y) :=
            Ex_fun_at (setLast s.1 default) fun y => F (drawOf s.1 y) (drawOf s.1 y)
        _ ≤ Ex fun y : Coin × A => Ex fun y' : Coin × A => F (drawOf s.1 y) (drawOf (forkCoins j s.1 s.2) y') :=
            Ex_mono fun y => (Ex_const (α := Coin × A) (F (drawOf s.1 y) (drawOf s.1 y))).symm.le.trans
              (Ex_mono fun y' => hF (drawOf s.1 y) (drawOf (forkCoins j s.1 s.2) y'))
    · exact le_of_eq (Ex_fun_at₂ hp fun y y' => F (drawOf s.1 y) (drawOf (forkCoins j s.1 s.2) y'))
  have hfork : ∀ (r r' : Fin n → Coin) (y y' : Coin × A),
      drawOf (forkCoins j r r') y' = forkDraw j (drawOf r y) (drawOf r' y') := fun r r' y y' =>
    Prod.ext (forkCoins_setLast j r r' y.1 y'.1).symm rfl
  calc (Ex fun g : (Fin n → Coin) → Coin × A => Ex fun s : (Fin n → Coin) × (Fin n → Coin) =>
        F (refDraw g s.1) (refDraw g (forkCoins j s.1 s.2)))
      = Ex fun s : (Fin n → Coin) × (Fin n → Coin) => Ex fun g : (Fin n → Coin) → Coin × A =>
          F (refDraw g s.1) (refDraw g (forkCoins j s.1 s.2)) :=
        Claims.Bridge.Ex_comm fun (g : (Fin n → Coin) → Coin × A) (s : (Fin n → Coin) × (Fin n → Coin)) =>
          F (refDraw g s.1) (refDraw g (forkCoins j s.1 s.2))
    _ ≤ Ex fun s : (Fin n → Coin) × (Fin n → Coin) => Ex fun y : Coin × A => Ex fun y' : Coin × A =>
          F (drawOf s.1 y) (drawOf (forkCoins j s.1 s.2) y') := Ex_mono hpt
    _ = Ex fun r : Fin n → Coin => Ex fun r' : Fin n → Coin => Ex fun y : Coin × A => Ex fun y' : Coin × A =>
          F (drawOf r y) (forkDraw j (drawOf r y) (drawOf r' y')) :=
        (Ex_prod fun s : (Fin n → Coin) × (Fin n → Coin) => Ex fun y : Coin × A => Ex fun y' : Coin × A =>
          F (drawOf s.1 y) (drawOf (forkCoins j s.1 s.2) y')).trans
          (congrArg Ex (funext fun r => congrArg Ex (funext fun r' => congrArg Ex (funext fun y =>
            congrArg Ex (funext fun y' => congrArg (F (drawOf r y)) (hfork r r' y y'))))))
    _ = Ex fun r : Fin n → Coin => Ex fun y : Coin × A => Ex fun r' : Fin n → Coin => Ex fun y' : Coin × A =>
          F (drawOf r y) (forkDraw j (drawOf r y) (drawOf r' y')) :=
        congrArg Ex (funext fun r => Claims.Bridge.Ex_comm fun (r' : Fin n → Coin) (y : Coin × A) =>
          Ex fun y' : Coin × A => F (drawOf r y) (forkDraw j (drawOf r y) (drawOf r' y')))
    _ = Ex fun r : Fin n → Coin => Ex fun y : Coin × A => Ex fun d' : (Fin n → Coin) × A =>
          F (drawOf r y) (forkDraw j (drawOf r y) d') :=
        congrArg Ex (funext fun r => congrArg Ex (funext fun y =>
          Ex_drawOf hn fun d' => F (drawOf r y) (forkDraw j (drawOf r y) d')))
    _ = Ex fun d : (Fin n → Coin) × A => Ex fun d' : (Fin n → Coin) × A => F d (forkDraw j d d') :=
        Ex_drawOf hn fun d => Ex fun d' : (Fin n → Coin) × A => F d (forkDraw j d d')
    _ = Ex fun s : ((Fin n → Coin) × A) × ((Fin n → Coin) × A) => F s.1 (forkDraw j s.1 s.2) :=
        (Ex_prod fun s : ((Fin n → Coin) × A) × ((Fin n → Coin) × A) => F s.1 (forkDraw j s.1 s.2)).symm

/-- **Two forked refuser draws are at most `forkDraw`'s** over a uniform `g`, for an event that holds at any second
draw where it holds at the first draw twice. -/
theorem Pr_fork_refDraw_le [DecidableEq (Fin n → Coin)] (j : Fin n)
    (Q : (Fin n → Coin) × A → (Fin n → Coin) × A → Prop) (hQ : ∀ d d', Q d d → Q d d') :
    (Ex fun g : (Fin n → Coin) → Coin × A => Pr fun s : (Fin n → Coin) × (Fin n → Coin) =>
        Q (refDraw g s.1) (refDraw g (forkCoins j s.1 s.2))) ≤
      Pr fun s : ((Fin n → Coin) × A) × ((Fin n → Coin) × A) => Q s.1 (forkDraw j s.1 s.2) := by
  classical
  have hF : ∀ d d' : (Fin n → Coin) × A, (if Q d d then (1 : ℝ) else 0) ≤ if Q d d' then (1 : ℝ) else 0 :=
    fun d d' => by
      by_cases h : Q d d
      · simp only [h, hQ d d' h, ↓reduceIte, le_refl]
      · simp only [h, ↓reduceIte]
        split_ifs <;> norm_num
  calc (Ex fun g : (Fin n → Coin) → Coin × A => Pr fun s : (Fin n → Coin) × (Fin n → Coin) =>
        Q (refDraw g s.1) (refDraw g (forkCoins j s.1 s.2)))
      = Ex fun g : (Fin n → Coin) → Coin × A => Ex fun s : (Fin n → Coin) × (Fin n → Coin) =>
          if Q (refDraw g s.1) (refDraw g (forkCoins j s.1 s.2)) then (1 : ℝ) else 0 :=
        congrArg Ex (funext fun g => (Ex_ind fun s : (Fin n → Coin) × (Fin n → Coin) =>
          Q (refDraw g s.1) (refDraw g (forkCoins j s.1 s.2))).symm)
    _ ≤ Ex fun s : ((Fin n → Coin) × A) × ((Fin n → Coin) × A) =>
          if Q s.1 (forkDraw j s.1 s.2) then (1 : ℝ) else 0 :=
        Ex_fork_refDraw_le j (fun d d' => if Q d d' then (1 : ℝ) else 0) hF
    _ = Pr fun s : ((Fin n → Coin) × A) × ((Fin n → Coin) × A) => Q s.1 (forkDraw j s.1 s.2) :=
        Ex_ind fun s : ((Fin n → Coin) × A) × ((Fin n → Coin) × A) => Q s.1 (forkDraw j s.1 s.2)

end Coupling

/-! ## The abort-law table bound -/

section TableBound

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **`TableBoundθ` with escapes on accepted runs**: `TableBoundθAt` at every statement, with escapes whose fork events
hold only where both forked runs accept and whose one-run events hold only where their run accepts. -/
def TableBoundθOnAcc (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (N : Msg → ℕ) (Ok : Msg → Col → Prop) : Prop :=
  ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω), ∃ e : Escapes H D.V P x,
    ((∀ i ω r r', e.ev i ω r r' →
        (run H P D.V x ω r).out = .acc ∧ (run H P D.V x ω (forkCoins (e.j i) r r')).out = .acc) ∧
      ∀ l ω r, e.ev₁ l ω r → (run H P D.V x ω r).out = .acc) ∧
    e.FitsFork (D.weight x) (D.k x) ∧ ∀ (ω : Ω) (C : Fin (N x) → Col), (∀ p, Ok x (C p)) →
      ¬ GoodTable D N x C →
      accPr H D.V P x ω ≤ D.εtable x + e.term ω +
        Pr fun r : Fin (D.V.k x) → Coin => (run H P D.V x ω r).out = .acc ∧
          ∃ (p : Fin (N x)) (u : Col), lookCol (D.cols (run H P D.V x ω r).seen 0) p.val = some u ∧ u ≠ C p

omit [Nonempty Coin] in
theorem tableBoundθ_of_onAcc {H : Msg → Msg} {D : Flock.Deploy Coin Col T} {N : Msg → ℕ} {Ok : Msg → Col → Prop}
    (h : TableBoundθOnAcc H D N Ok) : TableBoundθ H D N Ok := by
  intro x Ω _ _ P
  obtain ⟨e, -, hf, hb⟩ := h x Ω P
  exact ⟨e, hf, hb⟩

/-- **The refuser's escapes read at the runs as they abort**: the same rounds and weights, each event a collision among
the aborted runs' verifier calls. -/
abbrev Escapes.toAbortD {Ω A : Type} {H : Msg → Msg} {P : Prover Ω} {V : Verifier Coin} {x : Msg}
    {decC : Msg → Msg → Coin} {keep : Ω → (Fin (V.k x) → Coin) → A → Bool} {m₀ : Msg}
    (e : Escapes H V (refuser P x (V.k x) decC keep m₀) x) : EscapesD H (V.k x) (abortRun H P V x keep) :=
  { I := e.I
    c := e.c
    j := e.j
    ev := fun i ω d d' => TiedColl H ((abortRun H P V x keep ω d).vLog ++
      (abortRun H P V x keep ω (forkDraw (e.j i) d d')).vLog)
    L := e.L
    β := e.β
    ev₁ := fun _ ω d => TiedColl H (abortRun H P V x keep ω d).vLog
    c_nonneg := e.c_nonneg
    β_nonneg := e.β_nonneg
    tied := fun _ _ _ _ h => h
    tied₁ := fun _ _ _ h => h }

/-- **The refuser's bound averaged over its `g` is the bound at the abort law**, for any event `Q` on a run's view:
where the verifier takes a round, shows its coins, accepts only at its last round, decides without that round's coin
and refuses `m₀`, a bound `accPr ≤ ε + e.term + Pr[acc ∧ Q]` of the refuser at every `g`, with escapes on accepted
first runs, gives `accPrD ≤ ε + e.toAbortD.term + Pr[acc ∧ Q]` at the runs as they abort. Each table bound at the
abort law is this at its own bad-table event. -/
theorem accPrD_le_of_refuser {Ω A : Type} [Fintype Ω] [Nonempty Ω] [Fintype A] [Nonempty A] (H : Msg → Msg)
    (P : Prover Ω) (V : Verifier Coin) (x : Msg) (decC : Msg → Msg → Coin) (hVk : 0 < V.k x)
    (hpc : CoinsShown H V decC) (hlast : AccLast H V) (hfree : DecideFree H V)
    (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) (m₀ : Msg)
    (hm₀ : ∀ (as : List Msg) (cs : List Coin), ((V.next x (as ++ [m₀]) cs).eval H).1 ≠ .acc)
    (e : Escapes H V (refuser P x (V.k x) decC keep m₀) x)
    (hon : ∀ i ω r r', e.ev i ω r r' → (run H (refuser P x (V.k x) decC keep m₀) V x ω r).out = .acc ∧
      ((run H (refuser P x (V.k x) decC keep m₀) V x ω (forkCoins (e.j i) r r')).out = .acc ∨
        TiedColl H (run H (refuser P x (V.k x) decC keep m₀) V x ω r).vLog))
    (hon₁ : ∀ l ω r, e.ev₁ l ω r → (run H (refuser P x (V.k x) decC keep m₀) V x ω r).out = .acc)
    (ε : ℝ) (Q : Seen → Prop) (ω : Ω)
    (hb : ∀ g : (Fin (V.k x) → Coin) → Coin × A, accPr H V (refuser P x (V.k x) decC keep m₀) x (ω, g) ≤
      ε + e.term (ω, g) + Pr fun r : Fin (V.k x) → Coin =>
        (run H (refuser P x (V.k x) decC keep m₀) V x (ω, g) r).out = .acc ∧
          Q (run H (refuser P x (V.k x) decC keep m₀) V x (ω, g) r).seen) :
    accPrD (abortRun H P V x keep ω) ≤ ε + (Escapes.toAbortD e).term ω +
      Pr fun d : (Fin (V.k x) → Coin) × A => (abortRun H P V x keep ω d).out = .acc ∧
        Q (abortRun H P V x keep ω d).seen := by
  classical
  have hrun := refuser_run H P V x decC hpc hlast hfree keep m₀ hm₀
  have hEx := Ex_mono fun g : (Fin (V.k x) → Coin) → Coin × A => hb g
  rw [Ex_add, Ex_add, Ex_const] at hEx
  have hL : (Ex fun g : (Fin (V.k x) → Coin) → Coin × A =>
      accPr H V (refuser P x (V.k x) decC keep m₀) x (ω, g)) = accPrD (abortRun H P V x keep ω) := by
    unfold accPr accPrD
    rw [← Pr_refDraw hVk]
    exact congrArg Ex (funext fun g => Pr_congr fun r => (hrun ω g r).1)
  have hO : (Ex fun g : (Fin (V.k x) → Coin) → Coin × A =>
      Pr fun r : Fin (V.k x) → Coin => (run H (refuser P x (V.k x) decC keep m₀) V x (ω, g) r).out = .acc ∧
        Q (run H (refuser P x (V.k x) decC keep m₀) V x (ω, g) r).seen) =
      Pr fun d : (Fin (V.k x) → Coin) × A => (abortRun H P V x keep ω d).out = .acc ∧
        Q (abortRun H P V x keep ω d).seen := by
    rw [← Pr_refDraw hVk]
    refine congrArg Ex (funext fun g => Pr_congr fun r => ?_)
    constructor
    · rintro ⟨ha, rest⟩
      rw [(hrun ω g r).2 ha] at rest
      exact ⟨(hrun ω g r).1.1 ha, rest⟩
    · rintro ⟨ha, rest⟩
      have ha' := (hrun ω g r).1.2 ha
      rw [← (hrun ω g r).2 ha'] at rest
      exact ⟨ha', rest⟩
  have hT : (Ex fun g : (Fin (V.k x) → Coin) → Coin × A => e.term (ω, g)) ≤ (Escapes.toAbortD e).term ω := by
    unfold Escapes.term EscapesD.term
    rw [Ex_add, Ex_sum, Ex_sum]
    refine add_le_add (Finset.sum_le_sum fun i _ => ?_) (Finset.sum_le_sum fun l _ => ?_)
    · refine (Ex_sqrt_le _ fun g => mul_nonneg (e.c_nonneg i) (Pr_nonneg _)).trans ?_
      rw [Ex_mul_left]
      refine Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left ?_ (e.c_nonneg i))
      refine (Ex_mono fun g => Pr_mono fun s hs => ?_).trans
        (Pr_fork_refDraw_le (e.j i) (fun d d' => TiedColl H ((abortRun H P V x keep ω d).vLog ++
          (abortRun H P V x keep ω d').vLog)) fun d d' h => TiedColl.of_dup h _)
      obtain ⟨h1, h2 | h2⟩ := hon i (ω, g) s.1 s.2 hs
      · have ht := e.tied i (ω, g) s.1 s.2 hs
        rw [(hrun ω g _).2 h1, (hrun ω g _).2 h2] at ht
        exact ht
      · rw [(hrun ω g _).2 h1] at h2
        exact h2.append_right _
    · rw [Ex_mul_left]
      refine mul_le_mul_of_nonneg_left ?_ (e.β_nonneg l)
      refine (Ex_mono fun g => Pr_mono fun r hr => ?_).trans
        (le_of_eq (Pr_refDraw hVk fun d => TiedColl H (abortRun H P V x keep ω d).vLog))
      have ht := e.tied₁ l (ω, g) r hr
      rw [(hrun ω g r).2 (hon₁ l (ω, g) r hr)] at ht
      exact ht
  rw [hL, hO] at hEx
  linarith

/-- **The abort-law table bound from the table bound with escapes on accepted runs**, where the verifier takes a round,
shows its coins, accepts only at its last round, decides without that round's coin and refuses some message: the
refuser's bound averaged over its `g`, its escapes read at the runs as they abort. -/
theorem tableBoundθAbort_of_onAcc (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (N : Msg → ℕ)
    (Ok : Msg → Col → Prop) (decC : Msg → Msg → Coin) (htab : TableBoundθOnAcc H D N Ok) (hVk : ∀ x, 0 < D.V.k x)
    (hpc : CoinsShown H D.V decC) (hlast : AccLast H D.V) (hfree : DecideFree H D.V) (href : Refusable H D.V) :
    TableBoundθAbort H D N Ok := by
  classical
  intro x Ω _ _ P A _ _ keep
  obtain ⟨m₀, hm₀⟩ := href x
  obtain ⟨e, ⟨hon, hon₁⟩, hfit, hb⟩ :=
    htab x (Ω × ((Fin (D.V.k x) → Coin) → Coin × A)) (refuser P x (D.V.k x) decC keep m₀)
  refine ⟨Escapes.toAbortD e, hfit, fun ω C hC hg => ?_⟩
  exact accPrD_le_of_refuser H P D.V x decC (hVk x) hpc hlast hfree keep m₀ hm₀ e
    (fun i ω r r' h => ⟨(hon i ω r r' h).1, Or.inl (hon i ω r r' h).2⟩) hon₁ (D.εtable x)
    (fun s => ∃ (p : Fin (N x)) (u : Col), lookCol (D.cols s 0) p.val = some u ∧ u ≠ C p) ω
    (fun g => hb (ω, g) C hC hg)

end TableBound

/-! ## KS -/

section KS

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **Flock as the inner protocol meets `Inner.KSAbort`, from the KS chain's obligations with escapes on accepted
runs**: at every decoder within the radius, from `TableBoundθOnAcc`, `ColsOk` and `ConflictTied`, and the verifier's
code facts (`CoinsShown`, `AccLast`, `DecideFree`, `Refusable`), when the word is no longer than `N₀`, `K > 0`,
`ε_table ≥ 0`, the verifier takes a round, `Ok` admits `0`, and `k ≥ 8`, the relation splits by table and the
statement has one table wherever the verifier accepts some run. -/
theorem KSAbort_of_tableBoundOnAcc (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (N : Msg → ℕ)
    (Ok : Msg → Col → Prop) (decC : Msg → Msg → Coin) (hN : ∀ x, (N x : ℝ) ≤ D.N₀ x) (hεt : ∀ x, 0 ≤ D.εtable x)
    (hVk : ∀ x, 0 < D.V.k x) (hK : ∀ x, 0 < D.cap x) (hk : ∀ x, (8 ≤ D.k x ∧ Splits D x) ∨ NoAcc H D.V x)
    (hone : ∀ x, (∃ s, D.parse x = some s ∧ s.tables = 1) ∨ NoAcc H D.V x) (hz : ∀ x, Ok x 0)
    (hok : ColsOk H D N Ok) (hconf : ConflictTied H D N) (htab : TableBoundθOnAcc H D N Ok)
    (hpc : CoinsShown H D.V decC) (hlast : AccLast H D.V) (hfree : DecideFree H D.V) (href : Refusable H D.V)
    (decode : List (ℕ × Col) → List T) (hdec : decOK D decode) : (Inner.ofFlock D decode).KSAbort H :=
  KSAbort_of_oneWaitAbort H D hK hk hεt
    (oneWaitAbort_of_tableBound H D N hN hεt hVk (fun x => (hk x).imp (fun h => by omega) id) hone Ok hz hok hconf
      (tableBoundθAbort_of_onAcc H D N Ok decC htab hVk hpc hlast hfree href)) decode hdec

end KS

/-! ## The recursion's KS over Flock -/

section RecursionKS

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- **KS of the recursion over Flock, from Flock's KS at the outer deployment and the KS chain's obligations at the
inner one**: `KS_of_flockKS` at each inner decoder that meets the spec, with `Inner.KSAbort` at Flock from
`TableBoundθOnAcc`, `ColsOk` and `ConflictTied` at `DI` and the inner verifier's code facts
(`KSAbort_of_tableBoundOnAcc`; its coins shown are `PublicCoin`'s). It doesn't follow from `Claims.Flock.KS` at `DI`
(`flockKS_not_KSAbort`). -/
theorem KSFlock_of_flockKS [Zero Col] (H : Msg → Msg) (D : Deploy Coin) (DI DO : Flock.Deploy Coin Col T)
    (opening : Opening) (decC : Msg → Msg → Coin) (hF : Flock.KS H DO) (NI : Msg → ℕ) (Ok : Msg → Col → Prop)
    (hNI : ∀ x, (NI x : ℝ) ≤ DI.N₀ x) (hεI : ∀ x, 0 ≤ DI.εtable x) (hVkI : ∀ x, 0 < DI.V.k x)
    (hKI : ∀ x, 0 < DI.cap x) (hkI : ∀ x, (8 ≤ DI.k x ∧ Splits DI x) ∨ NoAcc H DI.V x)
    (hone : ∀ x, (∃ s, DI.parse x = some s ∧ s.tables = 1) ∨ NoAcc H DI.V x) (hz : ∀ x, Ok x 0)
    (hok : ColsOk H DI NI Ok) (hconf : ConflictTied H DI NI) (htab : TableBoundθOnAcc H DI NI Ok)
    (hlastI : AccLast H DI.V) (hfreeI : DecideFree H DI.V) (hrefI : Refusable H DI.V)
    (hRef : Refuses H D) (hRounds : ∀ x, DI.V.k (xI D x) ≤ D.V.k x) (hN : SessionNums D DO)
    (hε : TableErrNonneg D) (hRd : ∀ decI, SessionRounds D (Inner.ofFlock DI decI) DO) (hS : SessionAccepts D)
    (hR : ∀ decI, SessionRuns H D (Inner.ofFlock DI decI) DO)
    (hV : ∀ decI, InnerViewRun H D (Inner.ofFlock DI decI) DO)
    (hT : ∀ decI, CallsTrue H D (Inner.ofFlock DI decI) DO)
    (hL : ∀ decI, CallsLogColl H D (Inner.ofFlock DI decI) DO)
    (hO : ∀ decI, CallsOpenPlayed H D (Inner.ofFlock DI decI) DO opening)
    (hAcc : ∀ decI, InnerAccepts H D (Inner.ofFlock DI decI) DO)
    (hpc : ∀ decI, PublicCoin H (Inner.ofFlock DI decI) decC) (hN0 : CallsNil H D) : KSFlockA H D DI DO :=
  fun decI h => KS_of_flockKS H D _ DO opening decC hF
    (KSAbort_of_tableBoundOnAcc H DI NI Ok decC hNI hεI hVkI hKI hkI hone hz hok hconf htab
      (coinsShown_of_publicCoin (hpc decI)) hlastI hfreeI hrefI decI h) hRef hRounds hN hε
    (hRd decI) hS (hR decI) (hV decI) (hT decI) (hL decI) (hO decI) (hAcc decI) (hpc decI) hN0

end RecursionKS

end Claims.Recursion
