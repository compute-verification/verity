import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Fintype.Option
import Mathlib.Data.Fintype.Sum
import Mathlib.Tactic.GCongr
import Definitions.Pous
import Models.Pous.Assumptions
import Proofs.Pous.Statements
import Proofs.Pous.Sanity
import Proofs.Pous.Accounting.Numbers

/-!
# B5, Pietrzak's ex post facto reduction: a proof of `Pous.Guarantees.B5ExPostFacto(Bounded)`

Pietrzak 2018/194, Theorem 10 with §8's data-carrying labels, for one-way labels (`Pous.Labelling`), as pinned.
The proof follows the draft's plan, with the encoding done as a union bound over prediction patterns.

* **Replay.** For a fixed state `σ`, the threads `A₂(σ, c)`, `c ∈ VC`, run in lockstep. `trace` lists each thread's
  queries with their rounds. A query of round `r` depends only on the answers of earlier rounds (`trace_stable`),
  an output only on the rounds before the thread stops (`run_stable`), and a label only on the real queries of its
  ancestors (`label_congr`).
* **Positions.** There are `|VC|·(Q + 1) + N` of them (`Replay.Pos`): slot `k < Q` of thread `c` is its `k`-th
  query, slot `Q` its guess, and `inr i` the sequential labeller's query of node `i`, after every round.
* **Predictions.** A pair `(o, r)` of positions *predicts* (`Replay.cond`) when the point asked at `r` is asked in
  no earlier round, and its answer XOR `d` is the value `o` showed no later than `r`'s round. That value is either
  a query input's entry, or a guess of a thread that stopped by then. A pattern of predictions for different
  nodes in round order (`Replay.Ev`) holds with probability at most `2^(-w·|π|)` (`Replay.pr_Ev`). Resampling the
  oracle at the last predicted point keeps the rest of the pattern and is injective (`Replay.Ev_update`), while
  only one of the `2^w` answers makes the last prediction true (`Replay.card_Ev_snoc`).
* **Fresh labels and pebbling.** Label `i` is fresh (`Replay.fresh`) if a query input shows it no later than the
  first round asking `i`'s real query, or thread `i` guesses it before that round. Pebbling the fresh labels, then
  each node when its real query is asked, is legal (`Replay.mem_reach_of_asked`). It pebbles every correctly
  answered challenge within `D` rounds (`Replay.correct_mem_reach`), so winning needs more than `s` fresh labels
  (`PebblingHard`). Each fresh label gives a prediction (`Replay.exists_pair`), and sorting them by round gives a
  pattern that holds (`Replay.exists_Ev`), which truncates to exactly `s + 1` predictions.
* **Count.** A union over `σ ∈ Bits m` and the `(|VC|·(Q + 1) + N)^(2(s+1))` patterns gives
  `2^m·((|VC|·(Q + 1) + N)²/2^w)^(s+1)`. That is at most the pinned bound, since `N ≥ 1` whenever the event can
  happen. The parallel-query form follows from `work ≤ depth·width ≤ D·q`.
-/

namespace PousB5

open Pous Pous.Labelling
open scoped ENNReal

/-! ### Transcripts of a program, and what they depend on -/

section Trace

variable {M : IdealModel} {α : Type}

/-- The queries on the path taken when queries are answered by `f`, in order, each with its round (from 0). -/
def trace (f : M.Qry → M.Ans) : Prog M α → List (ℕ × M.Qry)
  | .ret _ => []
  | .round qs k => qs.map (fun x => (0, x)) ++ (trace f (k (qs.map f))).map fun e => (e.1 + 1, e.2)

theorem length_trace (f : M.Qry → M.Ans) (P : Prog M α) : (trace f P).length = P.work f := by
  induction P with
  | ret a => rfl
  | round qs k ih => simp [trace, Prog.work, ih]

theorem round_lt_depth (f : M.Qry → M.Ans) (P : Prog M α) : ∀ e ∈ trace f P, e.1 < P.depth f := by
  induction P with
  | ret a => simp [trace]
  | round qs k ih =>
    intro e he
    simp only [trace, List.mem_append, List.mem_map] at he
    simp only [Prog.depth]
    rcases he with ⟨x, _, rfl⟩ | ⟨e', he', rfl⟩
    · simp
    · have := ih _ e' he'
      simp only
      omega

/-- `f` and `g` agree on every point that `P` asks, under `f`, before round `r`. -/
def AgreeBefore (f g : M.Qry → M.Ans) (P : Prog M α) (r : ℕ) : Prop :=
  ∀ e ∈ trace f P, e.1 < r → f e.2 = g e.2

theorem mem_trace_round (f : M.Qry → M.Ans) (qs : List M.Qry) (k : List M.Ans → Prog M α) (x : M.Qry)
    (hx : x ∈ qs) : (0, x) ∈ trace f (.round qs k) := by
  simp only [trace, List.mem_append, List.mem_map]
  exact Or.inl ⟨x, hx, rfl⟩

theorem mem_trace_round_succ (f : M.Qry → M.Ans) (qs : List M.Qry) (k : List M.Ans → Prog M α)
    (e : ℕ × M.Qry) (he : e ∈ trace f (k (qs.map f))) : (e.1 + 1, e.2) ∈ trace f (.round qs k) := by
  simp only [trace, List.mem_append, List.mem_map]
  exact Or.inr ⟨e, he, rfl⟩

theorem agree_round {f g : M.Qry → M.Ans} {qs : List M.Qry} {k : List M.Ans → Prog M α} {r : ℕ}
    (h : AgreeBefore f g (.round qs k) (r + 1)) :
    qs.map f = qs.map g ∧ AgreeBefore f g (k (qs.map f)) r :=
  ⟨List.map_congr_left fun x hx => h (0, x) (mem_trace_round f qs k x hx) (Nat.succ_pos r),
    fun e he hlt => h (e.1 + 1, e.2) (mem_trace_round_succ f qs k e he) (by omega)⟩

/-- **Transcript locality.** If `f` and `g` agree on the points asked before round `r`, then the transcripts agree,
entry by entry, up to round `r`; and every entry of `g`'s transcript up to round `r` is one of `f`'s. -/
theorem trace_stable (f g : M.Qry → M.Ans) (P : Prog M α) : ∀ r : ℕ, AgreeBefore f g P r →
    (∀ (n : ℕ) (e : ℕ × M.Qry), (trace f P)[n]? = some e → e.1 ≤ r → (trace g P)[n]? = some e) ∧
    (∀ e ∈ trace g P, e.1 ≤ r → e ∈ trace f P) := by
  induction P with
  | ret a => intro r _; simp [trace]
  | round qs k ih =>
    intro r hag
    cases r with
    | zero =>
      constructor
      · intro n e hn he
        simp only [trace] at hn ⊢
        by_cases hlt : n < qs.length
        · rw [List.getElem?_append_left (by simpa using hlt)] at hn ⊢
          exact hn
        · rw [List.getElem?_append_right (by simpa using not_lt.1 hlt), List.getElem?_map,
            List.length_map] at hn
          cases hk : (trace f (k (qs.map f)))[n - qs.length]? with
          | none => rw [hk] at hn; simp at hn
          | some e' =>
            simp only [hk, Option.map_some, Option.some.injEq] at hn
            subst hn
            simp at he
      · intro e he he0
        simp only [trace, List.mem_append, List.mem_map] at he ⊢
        rcases he with ⟨x, hx, rfl⟩ | ⟨e', _, rfl⟩
        · exact Or.inl ⟨x, hx, rfl⟩
        · simp at he0
    | succ r =>
      obtain ⟨hqs, hag'⟩ := agree_round hag
      obtain ⟨ih1, ih2⟩ := ih (qs.map f) r hag'
      constructor
      · intro n e hn he
        simp only [trace] at hn ⊢
        rw [← hqs]
        by_cases hlt : n < qs.length
        · rw [List.getElem?_append_left (by simpa using hlt)] at hn ⊢
          exact hn
        · rw [List.getElem?_append_right (by simpa using not_lt.1 hlt), List.getElem?_map,
            List.length_map] at hn ⊢
          cases hk : (trace f (k (qs.map f)))[n - qs.length]? with
          | none => rw [hk] at hn; simp at hn
          | some e' =>
            simp only [hk, Option.map_some, Option.some.injEq] at hn
            subst hn
            rw [ih1 _ e' hk (by simp only at he; omega)]
            rfl
      · intro e he her
        simp only [trace, List.mem_append, List.mem_map] at he ⊢
        rw [← hqs] at he
        rcases he with ⟨x, hx, rfl⟩ | ⟨e', he', rfl⟩
        · exact Or.inl ⟨x, hx, rfl⟩
        · exact Or.inr ⟨e', ih2 e' he' (by simp only at her; omega), rfl⟩

/-- **Output locality.** A program that stops by round `r` returns the same output, at the same depth, under any
oracle that agrees with `f` on the points asked before round `r`. -/
theorem run_stable (f g : M.Qry → M.Ans) (P : Prog M α) : ∀ r : ℕ, AgreeBefore f g P r → P.depth f ≤ r →
    P.run g = P.run f ∧ P.depth g = P.depth f := by
  induction P with
  | ret a => intro _ _ _; exact ⟨rfl, rfl⟩
  | round qs k ih =>
    intro r hag hd
    cases r with
    | zero => simp [Prog.depth] at hd
    | succ r =>
      obtain ⟨hqs, hag'⟩ := agree_round hag
      have hd' : (k (qs.map f)).depth f ≤ r := by simp only [Prog.depth] at hd; omega
      obtain ⟨h1, h2⟩ := ih (qs.map f) r hag' hd'
      simp only [Prog.run, Prog.depth, ← hqs]
      exact ⟨h1, by rw [h2]⟩

theorem work_le_depth_mul_width (f : M.Qry → M.Ans) (P : Prog M α) : P.work f ≤ P.depth f * P.width f := by
  induction P with
  | ret a => simp [Prog.work]
  | round qs k ih =>
    simp only [Prog.work, Prog.depth, Prog.width]
    have h1 := ih (qs.map f)
    have h2 : (k (qs.map f)).depth f * (k (qs.map f)).width f ≤
        (k (qs.map f)).depth f * max qs.length ((k (qs.map f)).width f) :=
      Nat.mul_le_mul_left _ (le_max_right _ _)
    have h3 : qs.length ≤ max qs.length ((k (qs.map f)).width f) := le_max_left _ _
    rw [Nat.succ_mul]
    omega

end Trace

/-! ### Labels -/

section Labels

variable {N w : ℕ}

/-- The real query of node `i`: its index and its parents' labels. -/
def realQ (G : TopoDAG N) (d : Fin N → Bits w) (H : LabelQry N w → Bits w) (i : Fin N) : LabelQry N w :=
  (i, fun j => if j ∈ G.par i then some (label G d H j) else none)

theorem label_eq (G : TopoDAG N) (d : Fin N → Bits w) (H : LabelQry N w → Bits w) (i : Fin N) :
    label G d H i = bxor (H (realQ G d H i)) (d i) := by
  rw [label]
  simp only [dite_eq_ite]
  rfl

theorem bxor_cancel (a b : Bits w) : bxor (bxor a b) b = a :=
  funext fun j => by simp [bxor]

theorem eq_bxor_of_bxor_eq {a b v : Bits w} (h : bxor a b = v) : a = bxor v b := by
  rw [← h, bxor_cancel]

/-- **Label locality.** A label depends only on the oracle at the real queries of its ancestors and itself. -/
theorem label_congr (G : TopoDAG N) (d : Fin N → Bits w) {H H' : LabelQry N w → Bits w} (i : Fin N)
    (h : ∀ j ≤ i, H' (realQ G d H j) = H (realQ G d H j)) : label G d H' i = label G d H i := by
  induction i using WellFoundedLT.induction with
  | _ i ih =>
    have hq : realQ G d H' i = realQ G d H i := by
      unfold realQ
      congr 1
      funext j
      split_ifs with hj
      · rw [ih j (G.par_lt i j hj) fun j' hj' => h j' (hj'.trans (G.par_lt i j hj).le)]
      · rfl
    rw [label_eq, label_eq, hq, h i le_rfl]

theorem realQ_congr (G : TopoDAG N) (d : Fin N → Bits w) {H H' : LabelQry N w → Bits w} (i : Fin N)
    (h : ∀ j < i, H' (realQ G d H j) = H (realQ G d H j)) : realQ G d H' i = realQ G d H i := by
  unfold realQ
  congr 1
  funext j
  split_ifs with hj
  · rw [label_congr G d j fun j' hj' => h j' (lt_of_le_of_lt hj' (G.par_lt i j hj))]
  · rfl

end Labels

end PousB5

namespace PousB5

open Pous Pous.Labelling
open scoped ENNReal

variable {N w : ℕ}

/-- B5's oracle model. -/
abbrev RO (N w : ℕ) : IdealModel := IdealModel.randomOracle (LabelQry N w) (Bits w)

/-- One replay, for a fixed preprocessed state: the graph, the data, the challenges, the budget and the online
programs `P c = A₂(σ, c)`. -/
structure Replay (N w : ℕ) where
  G : TopoDAG N
  d : Fin N → Bits w
  VC : Finset (Fin N)
  D : ℕ
  Q : ℕ
  P : Fin N → Prog (RO N w) (Bits w)

namespace Replay

variable (R : Replay N w)

/-- Positions of the replay: slot `k < Q` of a challenge's thread is its `k`-th query and slot `Q` its guess;
`inr i` is the sequential labeller's query of node `i`. There are `|VC|·(Q + 1) + N` of them. -/
abbrev Pos : Type := ({c // c ∈ R.VC} × Fin (R.Q + 1)) ⊕ Fin N

/-- Thread `c`'s transcript, at the query type `LabelQry N w`. -/
def tr (H : LabelQry N w → Bits w) (c : Fin N) : List (ℕ × LabelQry N w) := trace H (R.P c)

/-- Some thread asks `z` in round `ρ`. -/
def askedAt (H : LabelQry N w → Bits w) (z : LabelQry N w) (ρ : ℕ) : Prop :=
  ∃ c ∈ R.VC, (ρ, z) ∈ R.tr H c

theorem tr_stable {H H' : LabelQry N w → Bits w} {c : Fin N} {κ : ℕ} (h : AgreeBefore H H' (R.P c) κ) :
    (∀ (n : ℕ) (e : ℕ × LabelQry N w), (R.tr H c)[n]? = some e → e.1 ≤ κ → (R.tr H' c)[n]? = some e) ∧
    (∀ e ∈ R.tr H' c, e.1 ≤ κ → e ∈ R.tr H c) :=
  trace_stable (M := RO N w) H H' (R.P c) κ h

theorem tr_round_lt (hP : ∀ c, (R.P c).Bounded R.D R.Q) (H : LabelQry N w → Bits w) (c : Fin N) :
    ∀ e ∈ R.tr H c, e.1 < R.D :=
  fun e he => lt_of_lt_of_le (round_lt_depth (M := RO N w) H (R.P c) e he) (hP c H).1

theorem tr_length (hP : ∀ c, (R.P c).Bounded R.D R.Q) (H : LabelQry N w → Bits w) (c : Fin N) :
    (R.tr H c).length ≤ R.Q :=
  (length_trace (M := RO N w) H (R.P c)).le.trans (hP c H).2

/-- The query at a position, as `(node, point, round)`; the labeller's queries come after every round. -/
def realAt (H : LabelQry N w → Bits w) : R.Pos → Option (Fin N × LabelQry N w × ℕ)
  | .inl (c, k) =>
      if k.val < R.Q then ((R.tr H c.1)[k.val]?).map fun e => (e.2.1, e.2, e.1) else none
  | .inr i => some (i, realQ R.G R.d H i, R.D + i.val)

/-- What a position shows about node `i`'s label, with the time it is shown: the `i`-th entry of a query's input
(at its round), or thread `i`'s guess (at its depth). -/
def obsAt (H : LabelQry N w → Bits w) (i : Fin N) : R.Pos → Option (Bits w × ℕ)
  | .inl (c, k) =>
      if k.val < R.Q then
        ((R.tr H c.1)[k.val]?).bind fun e => (e.2.2 i).map fun v => (v, e.1)
      else if c.1 = i then some ((R.P c.1).run H, (R.P c.1).depth H) else none
  | .inr _ => none

/-- A pair `(o, r)` of positions *predicts* the answer at `r`: the point asked at `r` is asked in no earlier
round, and its answer, XOR the data, is what `o` showed no later than `r`'s round. -/
def cond (H : LabelQry N w → Bits w) (p : R.Pos × R.Pos) : Prop :=
  ∃ i z κ v τ, R.realAt H p.2 = some (i, z, κ) ∧ R.obsAt H i p.1 = some (v, τ) ∧ τ ≤ κ ∧
    (∀ ρ < κ, ¬ R.askedAt H z ρ) ∧ bxor (H z) (R.d i) = v

/-- Two predicted positions concern different nodes and come in round order. -/
def rel (H : LabelQry N w → Bits w) (t t' : R.Pos × R.Pos) : Prop :=
  ∀ i z κ i' z' κ', R.realAt H t.2 = some (i, z, κ) → R.realAt H t'.2 = some (i', z', κ') → i ≠ i' ∧ κ ≤ κ'

/-- The pattern `π` of predictions holds. -/
def Ev (π : List (R.Pos × R.Pos)) (H : LabelQry N w → Bits w) : Prop :=
  (∀ t ∈ π, R.cond H t) ∧ π.Pairwise (R.rel H)

theorem Ev_take {π : List (R.Pos × R.Pos)} {H : LabelQry N w → Bits w} (n : ℕ) (h : R.Ev π H) :
    R.Ev (π.take n) H :=
  ⟨fun t ht => h.1 t (List.mem_of_mem_take ht), h.2.sublist (List.take_sublist _ _)⟩

theorem realAt_fst {H : LabelQry N w → Bits w} {p : R.Pos} {i : Fin N} {z : LabelQry N w} {κ : ℕ}
    (h : R.realAt H p = some (i, z, κ)) : z.1 = i := by
  cases p with
  | inl ck =>
    obtain ⟨c, k⟩ := ck
    simp only [realAt] at h
    split_ifs at h
    cases he : (R.tr H c.1)[k.val]? with
    | none => simp [he] at h
    | some e =>
      simp only [he, Option.map_some, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, rfl⟩ := h
      rfl
  | inr j =>
    simp only [realAt, Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl, rfl⟩ := h
    rfl

theorem realAt_inl_lt (hP : ∀ c, (R.P c).Bounded R.D R.Q) {H : LabelQry N w → Bits w} {c k} {i : Fin N}
    {z : LabelQry N w} {κ : ℕ} (h : R.realAt H (.inl (c, k)) = some (i, z, κ)) : κ < R.D := by
  simp only [realAt] at h
  split_ifs at h
  cases he : (R.tr H c.1)[k.val]? with
  | none => simp [he] at h
  | some e =>
    simp only [he, Option.map_some, Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl, rfl⟩ := h
    exact R.tr_round_lt hP H c.1 e (List.mem_of_getElem? he)

section Stable

variable {H H' : LabelQry N w → Bits w} {κ : ℕ} (hag : ∀ c ∈ R.VC, AgreeBefore H H' (R.P c) κ)
include hag

theorem realAt_inl_stable {c k} {x : Fin N × LabelQry N w × ℕ} (h : R.realAt H (.inl (c, k)) = some x)
    (hx : x.2.2 ≤ κ) : R.realAt H' (.inl (c, k)) = some x := by
  simp only [realAt] at h ⊢
  split_ifs at h ⊢ with hk
  cases he : (R.tr H c.1)[k.val]? with
  | none => simp [he] at h
  | some e =>
    rw [he, Option.map_some, Option.some.injEq] at h
    subst h
    rw [(R.tr_stable (hag c.1 c.2)).1 _ e he hx, Option.map_some]

theorem obsAt_stable {i : Fin N} {o : R.Pos} {v : Bits w} {τ : ℕ} (h : R.obsAt H i o = some (v, τ))
    (hτ : τ ≤ κ) : R.obsAt H' i o = some (v, τ) := by
  cases o with
  | inl ck =>
    obtain ⟨c, k⟩ := ck
    simp only [obsAt] at h ⊢
    split_ifs at h ⊢ with hk hci
    · cases he : (R.tr H c.1)[k.val]? with
      | none => simp [he] at h
      | some e =>
        rw [he, Option.bind_some] at h
        have he1 : e.1 = τ := by
          cases hv : (e.2 : LabelQry N w).2 i with
          | none => simp [hv] at h
          | some v' =>
            simp only [hv, Option.map_some, Option.some.injEq, Prod.mk.injEq] at h
            exact h.2
        rw [(R.tr_stable (hag c.1 c.2)).1 _ e he (he1 ▸ hτ), Option.bind_some]
        exact h
    · simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      obtain ⟨h1, h2⟩ := run_stable (M := RO N w) H H' (R.P c.1) κ (hag c.1 c.2) hτ
      rw [h1, h2]
  | inr j => simp [obsAt] at h

theorem askedAt_stable {z : LabelQry N w} {ρ : ℕ} (hρ : ρ < κ) (h : R.askedAt H' z ρ) : R.askedAt H z ρ := by
  obtain ⟨c, hc, he⟩ := h
  exact ⟨c, hc, (R.tr_stable (hag c hc)).2 _ he hρ.le⟩

end Stable

/-- **Resampling at the last prediction.** Changing the oracle at the point `z` asked at the last predicted
position keeps the other predictions, the last position's query, and what its observation showed. -/
theorem Ev_update (hP : ∀ c, (R.P c).Bounded R.D R.Q) {π : List (R.Pos × R.Pos)} {t : R.Pos × R.Pos}
    {H : LabelQry N w → Bits w} {i : Fin N} {z : LabelQry N w} {κ : ℕ} (hE : R.Ev (π ++ [t]) H)
    (hr : R.realAt H t.2 = some (i, z, κ)) (y : Bits w) :
    R.Ev π (Function.update H z y) ∧ R.realAt (Function.update H z y) t.2 = R.realAt H t.2 ∧
      ∀ v τ, R.obsAt H i t.1 = some (v, τ) → τ ≤ κ → R.obsAt (Function.update H z y) i t.1 = some (v, τ) := by
  obtain ⟨hcond, hpw⟩ := hE
  rw [List.pairwise_append] at hpw
  obtain ⟨hpw, -, hlast⟩ := hpw
  obtain ⟨i₀, z₀, κ₀, v, τ, hr₀, ho, hτ, hfresh, heq⟩ := hcond t (by simp)
  rw [hr] at hr₀
  simp only [Option.some.injEq, Prod.mk.injEq] at hr₀
  obtain ⟨rfl, rfl, rfl⟩ := hr₀
  have hag : ∀ c ∈ R.VC, AgreeBefore H (Function.update H z y : LabelQry N w → Bits w) (R.P c) κ := by
    intro c hc e he hlt
    have hne : (e.2 : LabelQry N w) ≠ z := fun h => hfresh e.1 hlt ⟨c, hc, by rw [← h]; exact he⟩
    exact (Function.update_of_ne (α := LabelQry N w) hne y H).symm
  have hzi : z.1 = i := R.realAt_fst hr
  -- the labeller's queries before the last one keep their labels
  have hinr : ∀ j : Fin N, R.D + j.val ≤ κ → ∀ j' < j,
      Function.update H z y (realQ R.G R.d H j') = H (realQ R.G R.d H j') := by
    intro j hj j' hj'
    apply Function.update_of_ne
    intro h
    cases ht : t.2 with
    | inl ck =>
      rw [ht] at hr
      have := R.realAt_inl_lt hP hr
      omega
    | inr j₀ =>
      rw [ht] at hr
      simp only [realAt, Option.some.injEq, Prod.mk.injEq] at hr
      obtain ⟨-, hz, hκ⟩ := hr
      have h1 : j' = j₀ := by
        have := congrArg Prod.fst (h.trans hz.symm)
        simpa [realQ] using this
      have h2 : (j' : ℕ) < j := hj'
      subst h1
      omega
  have hstab : ∀ (p : R.Pos) (i' : Fin N) (z' : LabelQry N w) (κ' : ℕ), R.realAt H p = some (i', z', κ') →
      κ' ≤ κ → R.realAt (Function.update H z y) p = R.realAt H p := by
    intro p i' z' κ' hp hκ'
    cases p with
    | inl ck =>
      obtain ⟨c, k⟩ := ck
      rw [hp]
      exact R.realAt_inl_stable hag hp hκ'
    | inr j =>
      have hκj : R.D + j.val = κ' := by
        simp only [realAt, Option.some.injEq, Prod.mk.injEq] at hp
        exact hp.2.2
      simp only [realAt]
      rw [realQ_congr R.G R.d j (hinr j (hκj ▸ hκ'))]
  have hRA : ∀ a ∈ π, R.realAt (Function.update H z y) a.2 = R.realAt H a.2 := by
    intro a ha
    obtain ⟨i', z', κ', -, -, hr', -⟩ := hcond a (List.mem_append_left _ ha)
    exact hstab _ _ _ _ hr' (hlast a ha t (by simp) i' z' κ' i z κ hr' hr).2
  refine ⟨⟨fun t' ht' => ?_, ?_⟩, hstab t.2 i z κ hr le_rfl, fun v' τ' ho' hτ' => R.obsAt_stable hag ho' hτ'⟩
  · obtain ⟨i', z', κ', v', τ', hr', ho', hτ', hfr', heq'⟩ := hcond t' (List.mem_append_left _ ht')
    obtain ⟨hne, hκ'⟩ := hlast t' ht' t (by simp) i' z' κ' i z κ hr' hr
    have hz' : z' ≠ z := fun h => hne (by rw [← R.realAt_fst hr', ← hzi, h])
    refine ⟨i', z', κ', v', τ', (hRA t' ht').trans hr', R.obsAt_stable hag ho' (hτ'.trans hκ'), hτ',
      fun ρ hρ ha => hfr' ρ hρ (R.askedAt_stable hag (lt_of_lt_of_le hρ hκ') ha), ?_⟩
    rw [Function.update_of_ne hz']
    exact heq'
  · refine hpw.imp_of_mem fun {a b} ha hb hab => ?_
    intro i₁ z₁ κ₁ i₂ z₂ κ₂ h₁ h₂
    rw [hRA a ha] at h₁
    rw [hRA b hb] at h₂
    exact hab i₁ z₁ κ₁ i₂ z₂ κ₂ h₁ h₂

end Replay

end PousB5

namespace PousB5

open Pous Pous.Labelling
open scoped ENNReal

variable {N w : ℕ}

/-- **Union bound** (from `thm1-family`). -/
theorem pr_exists_le {α ι : Type} [Fintype α] (s : Finset ι) (P : ι → α → Prop) :
    pr (fun a => ∃ x ∈ s, P x a) ≤ ∑ x ∈ s, pr (P x) := by
  classical
  simp only [pr_def, div_eq_mul_inv]
  rw [← Finset.sum_mul]
  gcongr
  have : (Finset.univ.filter fun a => ∃ x ∈ s, P x a) = s.biUnion fun x => Finset.univ.filter (P x) := by
    ext a; simp
  rw [this]
  exact_mod_cast Finset.card_biUnion_le

/-! ### Pebbling -/

theorem subset_reach (G : TopoDAG N) (P₀ : Finset (Fin N)) : ∀ r, P₀ ⊆ reach G P₀ r
  | 0 => le_rfl
  | r + 1 => (subset_reach G P₀ r).trans Finset.subset_union_left

theorem mem_reach_succ (G : TopoDAG N) (P₀ : Finset (Fin N)) (r : ℕ) (v : Fin N)
    (h : G.par v ⊆ reach G P₀ r) : v ∈ reach G P₀ (r + 1) :=
  Finset.mem_union_right _ (Finset.mem_filter.2 ⟨Finset.mem_univ _, h⟩)

namespace Replay

variable (R : Replay N w)

/-! ### A prediction costs `w` bits -/

open Classical in
/-- **One prediction costs `w` bits.** Resampling the last predicted point is injective and stays in the event of
the shorter pattern, while exactly one of the `2^w` answers makes the prediction come true. -/
theorem card_Ev_snoc (hP : ∀ c, (R.P c).Bounded R.D R.Q) (π : List (R.Pos × R.Pos)) (t : R.Pos × R.Pos) :
    (Finset.univ.filter (R.Ev (π ++ [t]))).card * 2 ^ w ≤ (Finset.univ.filter (R.Ev π)).card := by
  let f : (LabelQry N w → Bits w) × Bits w → (LabelQry N w → Bits w) := fun p =>
    match R.realAt p.1 t.2 with
    | some x => Function.update p.1 x.2.1 p.2
    | none => p.1
  have hcard : ((Finset.univ.filter (R.Ev (π ++ [t]))) ×ˢ (Finset.univ : Finset (Bits w))).card =
      (Finset.univ.filter (R.Ev (π ++ [t]))).card * 2 ^ w := by
    rw [Finset.card_product, Finset.card_univ, Fintype.card_fun, Fintype.card_bool, Fintype.card_fin]
  rw [← hcard]
  refine Finset.card_le_card_of_injOn f ?_ ?_
  · intro p hp
    rw [Finset.mem_coe, Finset.mem_product, Finset.mem_filter] at hp
    obtain ⟨⟨-, hE⟩, -⟩ := hp
    obtain ⟨i, z, κ, v, τ, hr, -⟩ := hE.1 t (by simp)
    rw [Finset.mem_coe, Finset.mem_filter]
    refine ⟨Finset.mem_univ _, ?_⟩
    simp only [f, hr]
    exact (R.Ev_update hP hE hr p.2).1
  · intro p₁ hp₁ p₂ hp₂ heq
    rw [Finset.mem_coe, Finset.mem_product, Finset.mem_filter] at hp₁ hp₂
    obtain ⟨⟨-, hE₁⟩, -⟩ := hp₁
    obtain ⟨⟨-, hE₂⟩, -⟩ := hp₂
    obtain ⟨i₁, z₁, κ₁, v₁, τ₁, hr₁, ho₁, hτ₁, -, heq₁⟩ := hE₁.1 t (by simp)
    obtain ⟨i₂, z₂, κ₂, v₂, τ₂, hr₂, ho₂, hτ₂, -, heq₂⟩ := hE₂.1 t (by simp)
    simp only [f, hr₁, hr₂] at heq
    obtain ⟨-, hra₁, hob₁⟩ := R.Ev_update hP hE₁ hr₁ p₁.2
    obtain ⟨-, hra₂, hob₂⟩ := R.Ev_update hP hE₂ hr₂ p₂.2
    rw [heq, hr₁] at hra₁
    rw [hra₁, hr₂, Option.some.injEq, Prod.mk.injEq, Prod.mk.injEq] at hra₂
    obtain ⟨rfl, rfl, rfl⟩ := hra₂
    have hv := (hob₁ v₁ τ₁ ho₁ hτ₁).symm.trans (heq ▸ hob₂ v₂ τ₂ ho₂ hτ₂)
    rw [Option.some.injEq, Prod.mk.injEq] at hv
    obtain ⟨rfl, -⟩ := hv
    have hz : p₁.1 z₁ = p₂.1 z₁ := by
      rw [eq_bxor_of_bxor_eq heq₁, eq_bxor_of_bxor_eq heq₂]
    have h1 : p₁.1 = p₂.1 := by
      funext x
      by_cases hx : x = z₁
      · rw [hx, hz]
      · have := congrFun heq x
        rwa [Function.update_of_ne hx, Function.update_of_ne hx] at this
    have h2 : p₁.2 = p₂.2 := by
      have := congrFun heq z₁
      rwa [Function.update_self, Function.update_self] at this
    exact Prod.ext h1 h2

open Classical in
theorem card_Ev (hP : ∀ c, (R.P c).Bounded R.D R.Q) (π : List (R.Pos × R.Pos)) :
    (Finset.univ.filter (R.Ev π)).card * (2 ^ w) ^ π.length ≤ Fintype.card (LabelQry N w → Bits w) := by
  induction π using List.reverseRecOn with
  | nil =>
    rw [List.length_nil, pow_zero, mul_one]
    exact (Finset.card_filter_le _ _).trans Finset.card_univ.le
  | append_singleton π t ih =>
    rw [List.length_append, List.length_singleton, pow_succ]
    calc (Finset.univ.filter (R.Ev (π ++ [t]))).card * ((2 ^ w) ^ π.length * 2 ^ w)
        = (Finset.univ.filter (R.Ev (π ++ [t]))).card * 2 ^ w * (2 ^ w) ^ π.length := by ring
      _ ≤ (Finset.univ.filter (R.Ev π)).card * (2 ^ w) ^ π.length :=
          Nat.mul_le_mul_right _ (R.card_Ev_snoc hP π t)
      _ ≤ _ := ih

/-- A pattern of `n` predictions holds with probability at most `2^(-w·n)`. -/
theorem pr_Ev (hP : ∀ c, (R.P c).Bounded R.D R.Q) (π : List (R.Pos × R.Pos)) :
    pr (R.Ev π) ≤ ((2 ^ w : ℝ≥0∞)⁻¹) ^ π.length := by
  classical
  have hΩ0 : (Fintype.card (LabelQry N w → Bits w) : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have hΩt : (Fintype.card (LabelQry N w → Bits w) : ℝ≥0∞) ≠ ⊤ := ENNReal.natCast_ne_top _
  have h2 : ((2 : ℝ≥0∞) ^ w) ^ π.length ≠ 0 := pow_ne_zero _ (pow_ne_zero _ two_ne_zero)
  have h2t : ((2 : ℝ≥0∞) ^ w) ^ π.length ≠ ⊤ := ENNReal.pow_ne_top (ENNReal.pow_ne_top ENNReal.ofNat_ne_top)
  rw [pr_def, ENNReal.div_le_iff hΩ0 hΩt]
  have hc : ((Finset.univ.filter (R.Ev π)).card : ℝ≥0∞) * ((2 : ℝ≥0∞) ^ w) ^ π.length ≤
      Fintype.card (LabelQry N w → Bits w) := by exact_mod_cast R.card_Ev hP π
  calc ((Finset.univ.filter (R.Ev π)).card : ℝ≥0∞)
      = (Finset.univ.filter (R.Ev π)).card * ((2 : ℝ≥0∞) ^ w) ^ π.length * (((2 : ℝ≥0∞) ^ w) ^ π.length)⁻¹ := by
        rw [mul_assoc, ENNReal.mul_inv_cancel h2 h2t, mul_one]
    _ ≤ Fintype.card (LabelQry N w → Bits w) * (((2 : ℝ≥0∞) ^ w) ^ π.length)⁻¹ := by gcongr
    _ = ((2 ^ w : ℝ≥0∞)⁻¹) ^ π.length * Fintype.card (LabelQry N w → Bits w) := by
        rw [ENNReal.inv_pow, mul_comm]

/-! ### Fresh labels and the ex post facto pebbling -/

/-- Node `i`'s label is *fresh*: some query's input shows it no later than the first round that asks `i`'s real
query, or thread `i` guesses it correctly before that round. -/
def fresh (H : LabelQry N w → Bits w) (i : Fin N) : Prop :=
  (∃ c ∈ R.VC, ∃ e ∈ R.tr H c, e.2.2 i = some (label R.G R.d H i) ∧
      ∀ ρ < e.1, ¬ R.askedAt H (realQ R.G R.d H i) ρ) ∨
    (i ∈ R.VC ∧ (R.P i).run H = label R.G R.d H i ∧
      ∀ ρ < (R.P i).depth H, ¬ R.askedAt H (realQ R.G R.d H i) ρ)

open Classical in
/-- **The ex post facto pebbling.** Pebbling the fresh labels first and then every node whose real query is asked
is legal: a node whose real query is asked before round `r` is pebbled within `r` rounds. -/
theorem mem_reach_of_asked (H : LabelQry N w → Bits w) :
    ∀ (r : ℕ) (i : Fin N), (∃ ρ < r, R.askedAt H (realQ R.G R.d H i) ρ) →
      i ∈ reach R.G (Finset.univ.filter (R.fresh H)) r
  | 0, _, ⟨_, hρ, _⟩ => absurd hρ (Nat.not_lt_zero _)
  | r + 1, i, ⟨ρ, hρ, c, hc, he⟩ => by
    refine mem_reach_succ _ _ r i fun j hj => ?_
    by_cases hf : R.fresh H j
    · exact subset_reach _ _ r (Finset.mem_filter.2 ⟨Finset.mem_univ _, hf⟩)
    · have hshow : (realQ R.G R.d H i).2 j = some (label R.G R.d H j) := by simp [realQ, hj]
      have : ∃ ρ' < ρ, R.askedAt H (realQ R.G R.d H j) ρ' := by
        by_contra hno
        push Not at hno
        exact hf (Or.inl ⟨c, hc, (ρ, realQ R.G R.d H i), he, hshow, fun ρ' hρ' ha => hno ρ' hρ' ha⟩)
      obtain ⟨ρ', hρ', ha⟩ := this
      exact mem_reach_of_asked H r j ⟨ρ', by omega, ha⟩

open Classical in
/-- Every correctly answered challenge is pebbled within `D` rounds from the fresh labels. -/
theorem correct_mem_reach (hP : ∀ c, (R.P c).Bounded R.D R.Q) (H : LabelQry N w → Bits w) (c : Fin N)
    (hc : c ∈ R.VC) (hcorr : (R.P c).run H = label R.G R.d H c) :
    c ∈ reach R.G (Finset.univ.filter (R.fresh H)) R.D := by
  by_cases ha : ∃ ρ < R.D, R.askedAt H (realQ R.G R.d H c) ρ
  · exact R.mem_reach_of_asked H R.D c ha
  · push Not at ha
    refine subset_reach _ _ R.D (Finset.mem_filter.2 ⟨Finset.mem_univ _, Or.inr ⟨hc, hcorr, fun ρ hρ => ?_⟩⟩)
    exact ha ρ (lt_of_lt_of_le hρ (hP c H).1)

/-! ### The prediction of a fresh label -/

open Classical in
/-- The first position asking node `i`'s real query (the labeller's, if no thread does), and its round. -/
theorem exists_real (hP : ∀ c, (R.P c).Bounded R.D R.Q) (H : LabelQry N w → Bits w) (i : Fin N) :
    ∃ (r : R.Pos) (κ : ℕ), R.realAt H r = some (i, realQ R.G R.d H i, κ) ∧
      (∀ ρ < κ, ¬ R.askedAt H (realQ R.G R.d H i) ρ) ∧
      ∀ τ ≤ R.D, (∀ ρ < τ, ¬ R.askedAt H (realQ R.G R.d H i) ρ) → τ ≤ κ := by
  by_cases h : ∃ ρ, R.askedAt H (realQ R.G R.d H i) ρ
  · obtain ⟨c, hc, he⟩ := Nat.find_spec h
    obtain ⟨n, hn⟩ := List.mem_iff_getElem?.1 he
    have hnQ : n < R.Q := lt_of_lt_of_le (List.getElem?_eq_some_iff.1 hn).1 (R.tr_length hP H c)
    refine ⟨.inl (⟨c, hc⟩, ⟨n, by omega⟩), Nat.find h, ?_, fun ρ hρ => Nat.find_min h hρ, fun τ _ hτ => ?_⟩
    · simp only [realAt, hnQ, ↓reduceIte, hn, Option.map_some]
      rfl
    · by_contra hlt
      exact hτ _ (not_le.1 hlt) (Nat.find_spec h)
  · push Not at h
    refine ⟨.inr i, R.D + i.val, rfl, fun ρ _ => h ρ, fun τ hτ _ => by omega⟩

open Classical in
/-- A fresh label gives a prediction whose real position asks that node's real query. -/
theorem exists_pair (hP : ∀ c, (R.P c).Bounded R.D R.Q) (H : LabelQry N w → Bits w) (i : Fin N)
    (hi : R.fresh H i) :
    ∃ x : (R.Pos × R.Pos) × ℕ, R.realAt H x.1.2 = some (i, realQ R.G R.d H i, x.2) ∧ R.cond H x.1 := by
  obtain ⟨r, κ, hr, hfr, hle⟩ := R.exists_real hP H i
  have hobs : ∃ (o : R.Pos) (τ : ℕ), R.obsAt H i o = some (label R.G R.d H i, τ) ∧ τ ≤ κ := by
    rcases hi with ⟨c, hc, e, he, hv, hno⟩ | ⟨hiVC, hrun, hno⟩
    · obtain ⟨n, hn⟩ := List.mem_iff_getElem?.1 he
      have hnQ : n < R.Q := lt_of_lt_of_le (List.getElem?_eq_some_iff.1 hn).1 (R.tr_length hP H c)
      refine ⟨.inl (⟨c, hc⟩, ⟨n, by omega⟩), e.1, ?_, hle _ (R.tr_round_lt hP H c e he).le hno⟩
      simp only [obsAt, hnQ, ↓reduceIte, hn, Option.bind_some, hv, Option.map_some]
    · refine ⟨.inl (⟨i, hiVC⟩, ⟨R.Q, Nat.lt_succ_self _⟩), (R.P i).depth H, ?_, hle _ (hP i H).1 hno⟩
      simp only [obsAt, lt_irrefl, ↓reduceIte, hrun]
  obtain ⟨o, τ, ho, hτ⟩ := hobs
  exact ⟨((o, r), κ), hr, i, _, κ, _, τ, hr, ho, hτ, hfr, (label_eq R.G R.d H i).symm⟩

open Classical in
/-- **The fresh pattern.** Predictions for all the fresh labels, in round order, hold together. -/
theorem exists_Ev (hP : ∀ c, (R.P c).Bounded R.D R.Q) (H : LabelQry N w → Bits w) (F : Finset (Fin N))
    (hF : ∀ i ∈ F, R.fresh H i) : ∃ π : List (R.Pos × R.Pos), π.length = F.card ∧ R.Ev π H := by
  choose x hx using fun i : F => R.exists_pair hP H i.1 (hF i.1 i.2)
  let l := F.attach.toList.mergeSort fun a b => decide ((x a).2 ≤ (x b).2)
  refine ⟨l.map fun a => (x a).1, ?_, ?_, ?_⟩
  · rw [List.length_map, List.length_mergeSort, Finset.length_toList, Finset.card_attach]
  · intro t ht
    obtain ⟨a, -, rfl⟩ := List.mem_map.1 ht
    exact (hx a).2
  · rw [List.pairwise_map]
    have hs : l.Pairwise fun a b => decide ((x a).2 ≤ (x b).2) :=
      List.pairwise_mergeSort (fun a b c hab hbc => by simp only [decide_eq_true_eq] at *; omega)
        (fun a b => by simp only [Bool.or_eq_true, decide_eq_true_eq]; omega) _
    have hn : l.Nodup := (List.mergeSort_perm _ _).nodup_iff.2 (Finset.nodup_toList _)
    refine (hs.and hn).imp fun {a b} hab => ?_
    intro i₁ z₁ κ₁ i₂ z₂ κ₂ h₁ h₂
    rw [(hx a).1, Option.some.injEq, Prod.mk.injEq] at h₁
    rw [(hx b).1, Option.some.injEq, Prod.mk.injEq] at h₂
    obtain ⟨rfl, -, rfl⟩ := h₁
    obtain ⟨rfl, -, rfl⟩ := h₂
    exact ⟨fun h => hab.2 (Subtype.ext h), by simpa using hab.1⟩

end Replay

end PousB5

namespace PousB5

open Pous Pous.Labelling
open scoped ENNReal

/-- **B5, `(D, Q)` form** (`Pous.Guarantees.B5ExPostFactoBounded`). -/
theorem b5_ex_post_facto_bounded : Pous.Guarantees.B5ExPostFactoBounded := by
  intro N w m s D Q k G VC d hard A₁ A₂ hA₂
  classical
  -- `k ≥ 1`: from no pebbles, fewer than `k` challenges are pebbled
  have hk : 0 < k := lt_of_le_of_lt (Nat.zero_le _) (hard ∅ (Nat.zero_le _))
  rcases Nat.eq_zero_or_pos N with rfl | hN
  · -- no nodes, so no challenge is answered
    have hno : ∀ H : LabelQry 0 w → Bits w,
        ¬ k ≤ (VC.filter fun c => (A₂ (A₁ H) c).run H = label G d H c).card := by
      intro H h
      have := (Finset.card_le_univ (VC.filter fun c => (A₂ (A₁ H) c).run H = label G d H c)).trans_eq
        (Fintype.card_fin 0)
      omega
    calc pr (fun H : LabelQry 0 w → Bits w =>
          k ≤ (VC.filter fun c => (A₂ (A₁ H) c).run H = label G d H c).card)
        ≤ pr (fun _ : LabelQry 0 w → Bits w => False) := pr_mono fun H h => hno H h
      _ = 0 := by rw [pr_def]; simp
      _ ≤ _ := by simp
  -- one replay per preprocessed state
  let R : Bits m → Replay N w := fun σ => ⟨G, d, VC, D, Q, A₂ σ⟩
  have hP : ∀ σ c, ((R σ).P c).Bounded (R σ).D (R σ).Q := fun σ c => hA₂ σ c
  let Pos₀ : Type := ({c // c ∈ VC} × Fin (Q + 1)) ⊕ Fin N
  -- winning needs more than `s` fresh labels, hence `s + 1` predictions that come true
  have hwin : ∀ H : LabelQry N w → Bits w,
      k ≤ (VC.filter fun c => (A₂ (A₁ H) c).run H = label G d H c).card →
      ∃ σ ∈ (Finset.univ : Finset (Bits m)), ∃ φ ∈ (Finset.univ : Finset (Fin (s + 1) → Pos₀ × Pos₀)),
        (R σ).Ev (List.ofFn φ) H := by
    intro H hwin
    refine ⟨A₁ H, Finset.mem_univ _, ?_⟩
    have hFs : s + 1 ≤ (Finset.univ.filter ((R (A₁ H)).fresh H)).card := by
      by_contra hlt
      have h1 := hard (Finset.univ.filter ((R (A₁ H)).fresh H)) (by omega)
      have h2 : (VC.filter fun c => (A₂ (A₁ H) c).run H = label G d H c) ⊆
          VC.filter fun c => c ∈ reach G (Finset.univ.filter ((R (A₁ H)).fresh H)) D := by
        intro c hc
        rw [Finset.mem_filter] at hc ⊢
        exact ⟨hc.1, (R (A₁ H)).correct_mem_reach (hP _) H c hc.1 hc.2⟩
      have := Finset.card_le_card h2
      omega
    obtain ⟨π, hlen, hE⟩ := (R (A₁ H)).exists_Ev (hP _) H (Finset.univ.filter ((R (A₁ H)).fresh H))
      fun i hi => (Finset.mem_filter.1 hi).2
    have hlen' : (π.take (s + 1)).length = s + 1 := by rw [List.length_take]; omega
    refine ⟨fun j => (π.take (s + 1))[j.val]'(by rw [hlen']; exact j.2), Finset.mem_univ _, ?_⟩
    have hofFn : List.ofFn (fun j : Fin (s + 1) => (π.take (s + 1))[j.val]'(by rw [hlen']; exact j.2)) =
        π.take (s + 1) := by
      apply List.ext_getElem (by rw [List.length_ofFn, hlen'])
      intro n h1 h2
      rw [List.getElem_ofFn]
    rw [hofFn]
    exact (R (A₁ H)).Ev_take (s + 1) hE
  have hcardPos : Fintype.card Pos₀ = VC.card * (Q + 1) + N := by
    simp only [Pos₀, Fintype.card_sum, Fintype.card_prod, Fintype.card_coe, Fintype.card_fin]
  calc pr (fun H : LabelQry N w → Bits w =>
        k ≤ (VC.filter fun c => (A₂ (A₁ H) c).run H = label G d H c).card)
      ≤ pr (fun H => ∃ σ ∈ (Finset.univ : Finset (Bits m)),
          ∃ φ ∈ (Finset.univ : Finset (Fin (s + 1) → Pos₀ × Pos₀)), (R σ).Ev (List.ofFn φ) H) := pr_mono hwin
    _ ≤ ∑ σ : Bits m, pr (fun H => ∃ φ ∈ (Finset.univ : Finset (Fin (s + 1) → Pos₀ × Pos₀)),
          (R σ).Ev (List.ofFn φ) H) := pr_exists_le _ _
    _ ≤ ∑ σ : Bits m, ∑ φ : Fin (s + 1) → Pos₀ × Pos₀, pr ((R σ).Ev (List.ofFn φ)) :=
        Finset.sum_le_sum fun σ _ => pr_exists_le _ _
    _ ≤ ∑ _σ : Bits m, ∑ _φ : Fin (s + 1) → Pos₀ × Pos₀, ((2 ^ w : ℝ≥0∞)⁻¹) ^ (s + 1) := by
        refine Finset.sum_le_sum fun σ _ => Finset.sum_le_sum fun φ _ => ?_
        have := (R σ).pr_Ev (hP σ) (List.ofFn φ)
        rwa [List.length_ofFn] at this
    _ = 2 ^ m * ((((VC.card * (Q + 1) + N) ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ w) ^ (s + 1) := by
        simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fun, Fintype.card_prod, hcardPos,
          Fintype.card_fin, Fintype.card_bool, nsmul_eq_mul]
        generalize VC.card * (Q + 1) + N = T
        rw [div_eq_mul_inv, mul_pow]
        push_cast
        ring
    _ ≤ N * 2 ^ m * ((((VC.card * (Q + 1) + N) ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ w) ^ (s + 1) := by
        rw [mul_assoc]
        exact le_mul_of_one_le_left' (by exact_mod_cast hN)

/-- **B5, parallel-query form** (`Pous.Guarantees.B5ExPostFacto`): `RoundBounded D q` gives work `≤ D·q`. -/
theorem b5_ex_post_facto : Pous.Guarantees.B5ExPostFacto := by
  intro N w m s D q k G VC d hard A₁ A₂ hA₂
  refine b5_ex_post_facto_bounded N w m s D (D * q) k G VC d hard A₁ A₂ fun σ c ω => ⟨(hA₂ σ c ω).1, ?_⟩
  exact (work_le_depth_mul_width _ _).trans (Nat.mul_le_mul (hA₂ σ c ω).1 (hA₂ σ c ω).2)

end PousB5
