import Definitions.Accounting.Window
import Models.Accounting.Guarantees.Responder
import Proofs.Accounting.Responder

/-!
# B2's oracle-location form, proved: a window as a ring responder

`B2Window` (`Security/Models/Accounting/Guarantees/Responder.lean`): a family of `RingWindow`s whose multiplications
take at least `ℓ > 0` ticks to answer and whose spans are at most `D·ℓ` ticks is the step of an `(N, D, Q, L)`
sequential responder. With `ℓ = ⌈(Δ + RTT)/D_p⌉` (the latency form of sequential speed) and one challenge every
`Δ + RTT` ticks (fixed pacing), that is `D = D_p`.

`sim` is the responder's program for one window. It runs the window tick by tick, asks the additions, `one` and the
keys at once, holds the multiplications, and asks every held one at the first tick the earliest-asked of them could be
due (`ℓ` ticks after it was asked), in that tick's round; the multiplications still held at the end are asked in a last
round. No held multiplication is due before it is asked, so every batch the window sees is the oracle's (`sim_spec`);
rounds with a multiplication are at least `ℓ` ticks apart, so there are at most `⌈span/ℓ⌉` of them (`sim_mulDepth`); and
it asks every query the window asks once (`sim_spec`'s counts). `cap` stops a program with a default once it would
exceed `(D, Q, L)`, so that it is `RingBounded` on every primitive, and leaves its run alone where it stays within them
(`cap_bounded`, `cap_run`). `window_responder` puts them together for one window, and `B2Window` picks one for each.

`B2WindowAudit` applies PoUS's guarantee to that responder: where every window stays within `(Q, L)`, its answers are
the windows' results (`seqAnswers_congr`), so an outcome the windows answer correctly is one it does.
`B2WindowP2Ring` is that at P2's deployed point.
-/

namespace Accounting.SecurityProofs

namespace Latency

open Pous

variable {B w p : ℕ} [NeZero p] {α : Type}

open Classical in
/-- `g`, updated by the answers `r` to the queries `qs`. -/
noncomputable def upd (g : RingQry B w → Bits w) : List (RingQry B w) → List (Bits w) → RingQry B w → Bits w
  | [], _ => g
  | _ :: _, [] => g
  | q :: qs, a :: r => fun x => if x = q then a else upd g qs r x

theorem upd_map (g f : RingQry B w → Bits w) (qs : List (RingQry B w)) (q : RingQry B w) :
    (q ∈ qs ∨ g q = f q) → upd g qs (qs.map f) q = f q := by
  induction qs with
  | nil =>
    intro h
    rcases h with h | h
    · simp at h
    · simpa [upd] using h
  | cons q' qs ih =>
    intro h
    simp only [List.map_cons, upd]
    by_cases hq : q = q'
    · subst hq; simp
    · rw [ite_eq_right hq]
      apply ih
      rcases h with h | h
      · exact Or.inl ((List.mem_cons.1 h).resolve_left hq)
      · exact Or.inr h

theorem isKey_of_isMul {q : RingQry B w} (h : q.isMul = true) : q.isKey = false := by
  cases q <;> simp_all [RingQry.isMul, RingQry.isKey]

theorem any_filter_not_mul (A : List (RingQry B w)) : (A.filter fun q => !q.isMul).any RingQry.isMul = false :=
  List.any_eq_false.2 fun q hq => by simpa using (List.mem_filter.1 hq).2

/-- A round of a ring program, its queries and answers at the ring's own types. -/
def rnd (qs : List (RingQry B w)) (k : List (Bits w) → Prog (ringRO B w p) α) : Prog (ringRO B w p) α :=
  .round qs k

section rnd
variable (f : RingQry B w → Bits w) (qs : List (RingQry B w)) (k : List (Bits w) → Prog (ringRO B w p) α)
theorem run_rnd : (rnd qs k).run f = (k (qs.map f)).run f := rfl
theorem keyWork_rnd : (rnd qs k).keyWork f = qs.countP RingQry.isKey + (k (qs.map f)).keyWork f := rfl
theorem ringWork_rnd : (rnd qs k).ringWork f = qs.countP (fun q => !q.isKey) + (k (qs.map f)).ringWork f := rfl
theorem mulDepth_rnd :
    (rnd qs k).mulDepth f = (if qs.any RingQry.isMul then 1 else 0) + (k (qs.map f)).mulDepth f := rfl
end rnd

open Classical in
/-- The held multiplications asked at tick `n`: all of them, once the earliest-asked could be due. -/
noncomputable def fired (held : List (ℕ × RingQry B w)) (h0 ℓ n : ℕ) : List (RingQry B w) :=
  if held ≠ [] ∧ h0 + ℓ ≤ n then held.map Prod.snd else []

open Classical in
/-- The multiplications still held after tick `n`'s round, before that tick's own. -/
noncomputable def kept (held : List (ℕ × RingQry B w)) (h0 ℓ n : ℕ) : List (ℕ × RingQry B w) :=
  if held ≠ [] ∧ h0 + ℓ ≤ n then [] else held

open Classical in
/-- The responder's program for the window `W` from tick `n`, with `k` ticks left: `as` the queries asked so far with
their ticks, `ds` the batches delivered so far, `g` the answers it knows, `held` the multiplications it holds, with
their ticks, and `h0` the tick the earliest-asked of them was asked at. -/
noncomputable def sim (W : RingWindow B w α) (ℓ : ℕ) :
    ℕ → ℕ → List (ℕ × RingQry B w) → List (List (Bits w)) → (RingQry B w → Bits w) → List (ℕ × RingQry B w) → ℕ →
      Prog (ringRO B w p) α
  | 0, _, as, ds, g, held, _ =>
    rnd (held.map Prod.snd) fun r => .ret (W.out ds (W.inFlight (upd g (held.map Prod.snd) r) as))
  | k + 1, n, as, ds, g, held, h0 =>
    let qs := fired held h0 ℓ n ++ (W.ask ds).filter fun q => !q.isMul
    let as' := as ++ (W.ask ds).map fun q => (n, q)
    rnd qs fun r =>
      sim W ℓ k (n + 1) as' (ds ++ [W.dueAt (upd g qs r) as' n]) (upd g qs r)
        (kept held h0 ℓ n ++ ((W.ask ds).filter RingQry.isMul).map fun q => (n, q))
        (if kept held h0 ℓ n = [] then n else h0)

open Classical in
theorem sim_succ (W : RingWindow B w α) (ℓ k n : ℕ) (as : List (ℕ × RingQry B w)) (ds : List (List (Bits w)))
    (g : RingQry B w → Bits w) (held : List (ℕ × RingQry B w)) (h0 : ℕ) :
    sim (p := p) W ℓ (k + 1) n as ds g held h0 =
      rnd (fired held h0 ℓ n ++ (W.ask ds).filter fun q => !q.isMul) fun r =>
        sim W ℓ k (n + 1) (as ++ (W.ask ds).map fun q => (n, q))
          (ds ++ [W.dueAt (upd g (fired held h0 ℓ n ++ (W.ask ds).filter fun q => !q.isMul) r)
            (as ++ (W.ask ds).map fun q => (n, q)) n])
          (upd g (fired held h0 ℓ n ++ (W.ask ds).filter fun q => !q.isMul) r)
          (kept held h0 ℓ n ++ ((W.ask ds).filter RingQry.isMul).map fun q => (n, q))
          (if kept held h0 ℓ n = [] then n else h0) :=
  rfl

/-! ## The responder answers as the window does, and asks what it asks -/

open Classical in
theorem sim_spec (W : RingWindow B w α) {ℓ : ℕ} (hℓ : 0 < ℓ) (hlat : ∀ q : RingQry B w, q.isMul = true → ℓ ≤ W.lat q)
    (f : RingQry B w → Bits w) :
    ∀ k n (g : RingQry B w → Bits w) (held : List (ℕ × RingQry B w)) h0, n + k = W.span →
      (∀ e ∈ (W.trace f n).1, g e.2 = f e.2 ∨ e ∈ held) → (∀ e ∈ held, e.2.isMul = true ∧ h0 ≤ e.1 ∧ e.1 < n) →
      (sim (p := p) W ℓ k n (W.trace f n).1 (W.trace f n).2 g held h0).run f = W.result f ∧
      (sim (p := p) W ℓ k n (W.trace f n).1 (W.trace f n).2 g held h0).keyWork f +
          (W.trace f n).1.countP (fun e => e.2.isKey) = (W.trace f W.span).1.countP (fun e => e.2.isKey) ∧
      (sim (p := p) W ℓ k n (W.trace f n).1 (W.trace f n).2 g held h0).ringWork f +
          (W.trace f n).1.countP (fun e => !e.2.isKey) =
        (W.trace f W.span).1.countP (fun e => !e.2.isKey) + held.length := by
  intro k
  induction k with
  | zero =>
    intro n g held h0 hk hg hheld
    have hn : n = W.span := by omega
    subst hn
    simp only [sim, run_rnd, keyWork_rnd, ringWork_rnd, Prog.run, Prog.keyWork, Prog.ringWork]
    have hm : ∀ q ∈ held.map Prod.snd, q.isMul = true := fun q hq => by
      obtain ⟨e, he, rfl⟩ := List.mem_map.1 hq
      exact (hheld e he).1
    have k0 : (held.map Prod.snd).countP RingQry.isKey = 0 :=
      List.countP_eq_zero.2 fun q hq => by simp [isKey_of_isMul (hm q hq)]
    have r0 : (held.map Prod.snd).countP (fun q => !q.isKey) = held.length := by
      rw [List.countP_eq_length.2 fun q hq => by simp [isKey_of_isMul (hm q hq)], List.length_map]
    refine ⟨?_, ?_, ?_⟩
    · unfold RingWindow.result
      congr 1
      unfold RingWindow.inFlight
      refine List.map_congr_left fun e he => upd_map _ _ _ _ ?_
      rcases hg e (List.mem_filter.1 he).1 with h | h
      · exact Or.inr h
      · exact Or.inl (List.mem_map.2 ⟨e, h, rfl⟩)
    · omega
    · omega
  | succ k ih =>
    intro n g held h0 hk hg hheld
    rw [sim_succ]
    simp only [run_rnd, keyWork_rnd, ringWork_rnd]
    have hup : ∀ q, (q ∈ fired held h0 ℓ n ++ (W.ask (W.trace f n).2).filter (fun q => !q.isMul) ∨ g q = f q) →
        upd g (fired held h0 ℓ n ++ (W.ask (W.trace f n).2).filter (fun q => !q.isMul))
          ((fired held h0 ℓ n ++ (W.ask (W.trace f n).2).filter (fun q => !q.isMul)).map f) q = f q :=
      fun q => upd_map g f _ q
    generalize upd g (fired held h0 ℓ n ++ (W.ask (W.trace f n).2).filter (fun q => !q.isMul))
      ((fired held h0 ℓ n ++ (W.ask (W.trace f n).2).filter (fun q => !q.isMul)).map f) = g' at hup ⊢
    have htr : W.trace f (n + 1) = ((W.trace f n).1 ++ (W.ask (W.trace f n).2).map (fun q => (n, q)),
        (W.trace f n).2 ++ [W.dueAt f ((W.trace f n).1 ++ (W.ask (W.trace f n).2).map (fun q => (n, q))) n]) := rfl
    generalize W.ask (W.trace f n).2 = A at hup htr ⊢
    have hkept : ∀ e ∈ kept held h0 ℓ n, e ∈ held ∧ n < e.1 + W.lat e.2 := by
      intro e he
      by_cases hF : held ≠ [] ∧ h0 + ℓ ≤ n
      · rw [kept, ite_eq_left hF] at he
        simp at he
      · rw [kept, ite_eq_right hF] at he
        refine ⟨he, ?_⟩
        obtain ⟨hm, h0e, _⟩ := hheld e he
        have := hlat _ hm
        have : ¬ h0 + ℓ ≤ n := fun h => hF ⟨List.ne_nil_of_mem he, h⟩
        omega
    have hfm : ∀ q ∈ fired held h0 ℓ n, q.isMul = true := by
      intro q hq
      by_cases hF : held ≠ [] ∧ h0 + ℓ ≤ n
      · rw [fired, ite_eq_left hF] at hq
        obtain ⟨e, he, rfl⟩ := List.mem_map.1 hq
        exact (hheld e he).1
      · rw [fired, ite_eq_right hF] at hq
        simp at hq
    have hfired : ∀ e ∈ held, e ∉ kept held h0 ℓ n → e.2 ∈ fired held h0 ℓ n := by
      intro e he hn
      by_cases hF : held ≠ [] ∧ h0 + ℓ ≤ n
      · rw [fired, ite_eq_left hF]
        exact List.mem_map.2 ⟨e, he, rfl⟩
      · exact absurd (by rw [kept, ite_eq_right hF]; exact he) hn
    have hlen : (fired held h0 ℓ n).length + (kept held h0 ℓ n).length = held.length := by
      by_cases hF : held ≠ [] ∧ h0 + ℓ ≤ n
      · rw [fired, kept, ite_eq_left hF, ite_eq_left hF]
        simp
      · rw [fired, kept, ite_eq_right hF, ite_eq_right hF]
        simp
    have hnm : ∀ q ∈ A, q.isMul = false → g' q = f q := fun q hq hm =>
      hup q (Or.inl (List.mem_append_right _ (List.mem_filter.2 ⟨hq, by simp [hm]⟩)))
    have hold : ∀ e ∈ (W.trace f n).1, g' e.2 = f e.2 ∨ e ∈ kept held h0 ℓ n := by
      intro e he
      rcases hg e he with h | h
      · exact Or.inl (hup _ (Or.inr h))
      · by_cases hk' : e ∈ kept held h0 ℓ n
        · exact Or.inr hk'
        · exact Or.inl (hup _ (Or.inl (List.mem_append_left _ (hfired e h hk'))))
    have hdue : W.dueAt g' ((W.trace f n).1 ++ A.map fun q => (n, q)) n =
        W.dueAt f ((W.trace f n).1 ++ A.map fun q => (n, q)) n := by
      unfold RingWindow.dueAt
      refine List.map_congr_left fun e he => ?_
      obtain ⟨he, hn⟩ := List.mem_filter.1 he
      have hn : e.1 + W.lat e.2 = n := by simpa using hn
      rcases List.mem_append.1 he with he | he
      · rcases hold e he with h | h
        · exact h
        · have := (hkept e h).2
          omega
      · obtain ⟨q, hq, rfl⟩ := List.mem_map.1 he
        cases hm : q.isMul
        · exact hnm q hq hm
        · have := hlat q hm
          dsimp only at hn
          omega
    have hg' : ∀ e ∈ (W.trace f (n + 1)).1, g' e.2 = f e.2 ∨
        e ∈ kept held h0 ℓ n ++ (A.filter RingQry.isMul).map fun q => (n, q) := by
      rw [htr]
      intro e he
      dsimp only at he
      rcases List.mem_append.1 he with he | he
      · rcases hold e he with h | h
        · exact Or.inl h
        · exact Or.inr (List.mem_append_left _ h)
      · obtain ⟨q, hq, rfl⟩ := List.mem_map.1 he
        cases hm : q.isMul
        · exact Or.inl (hnm q hq hm)
        · exact Or.inr (List.mem_append_right _ (List.mem_map.2 ⟨q, List.mem_filter.2 ⟨hq, hm⟩, rfl⟩))
    have hheld' : ∀ e ∈ kept held h0 ℓ n ++ (A.filter RingQry.isMul).map (fun q => (n, q)),
        e.2.isMul = true ∧ (if kept held h0 ℓ n = [] then n else h0) ≤ e.1 ∧ e.1 < n + 1 := by
      intro e he
      rcases List.mem_append.1 he with he | he
      · have hne : kept held h0 ℓ n ≠ [] := List.ne_nil_of_mem he
        rw [ite_eq_right hne]
        obtain ⟨hm, h1, h2⟩ := hheld e (hkept e he).1
        exact ⟨hm, h1, by omega⟩
      · obtain ⟨q, hq, rfl⟩ := List.mem_map.1 he
        refine ⟨(List.mem_filter.1 hq).2, ?_, by simp⟩
        by_cases hne : kept held h0 ℓ n = []
        · simp [hne]
        · rw [ite_eq_right hne]
          obtain ⟨e', he'⟩ := List.exists_mem_of_ne_nil _ hne
          have := hheld e' (hkept e' he').1
          dsimp only
          omega
    obtain ⟨h1, h2, h3⟩ := ih (n + 1) g' _ _ (by omega) hg' hheld'
    rw [htr] at h1 h2 h3
    dsimp only at h1 h2 h3
    rw [← hdue] at h1 h2 h3
    refine ⟨h1, ?_, ?_⟩
    · have km : (List.map (fun q => (n, q)) A).countP (fun e => e.2.isKey) = A.countP RingQry.isKey := by
        rw [List.countP_map]
        rfl
      rw [List.countP_append, km] at h2
      rw [List.countP_append]
      have kf : (fired held h0 ℓ n).countP RingQry.isKey = 0 :=
        List.countP_eq_zero.2 fun q hq => by simp [isKey_of_isMul (hfm q hq)]
      have kA := List.countP_eq_countP_filter_add A RingQry.isKey RingQry.isMul
      have k0 : (A.filter RingQry.isMul).countP RingQry.isKey = 0 :=
        List.countP_eq_zero.2 fun q hq => by simp [isKey_of_isMul (List.mem_filter.1 hq).2]
      omega
    · have km : (List.map (fun q => (n, q)) A).countP (fun e => !e.2.isKey) = A.countP (fun q => !q.isKey) := by
        rw [List.countP_map]
        rfl
      rw [List.countP_append, km, List.length_append, List.length_map] at h3
      rw [List.countP_append]
      have rf : (fired held h0 ℓ n).countP (fun q => !q.isKey) = (fired held h0 ℓ n).length :=
        List.countP_eq_length.2 fun q hq => by simp [isKey_of_isMul (hfm q hq)]
      have rA := List.countP_eq_countP_filter_add A (fun q => !q.isKey) RingQry.isMul
      have r0 : (A.filter RingQry.isMul).countP (fun q => !q.isKey) = (A.filter RingQry.isMul).length :=
        List.countP_eq_length.2 fun q hq => by simp [isKey_of_isMul (List.mem_filter.1 hq).2]
      omega

/-! ## Rounds with a multiplication are `ℓ` ticks apart -/

theorem step_bound {ℓ : ℕ} (hℓ : 0 < ℓ) {S c c' m r : ℕ} (hr : r ≤ (S - c') / ℓ)
    (h : m = 0 ∧ c ≤ c' ∨ m ≤ 1 ∧ c + ℓ ≤ c' ∧ c' ≤ S) : m + r ≤ (S - c) / ℓ := by
  rcases h with ⟨hm, hc⟩ | ⟨hm, hc, hS⟩
  · subst hm
    rw [Nat.zero_add]
    exact hr.trans (Nat.div_le_div_right (by omega))
  · calc m + r ≤ 1 + (S - c') / ℓ := by omega
      _ = (S - c' + ℓ) / ℓ := by rw [Nat.add_div_right _ hℓ]; omega
      _ ≤ (S - c) / ℓ := Nat.div_le_div_right (by omega)

open Classical in
theorem sim_mulDepth (W : RingWindow B w α) {ℓ : ℕ} (hℓ : 0 < ℓ) (f : RingQry B w → Bits w) :
    ∀ k n (as : List (ℕ × RingQry B w)) (ds : List (List (Bits w))) (g : RingQry B w → Bits w)
      (held : List (ℕ × RingQry B w)) h0, n + k = W.span → (held ≠ [] → h0 < n) →
      (sim (p := p) W ℓ k n as ds g held h0).mulDepth f ≤ (W.span + ℓ - 1 - if held = [] then n else h0) / ℓ := by
  intro k
  induction k with
  | zero =>
    intro n as ds g held h0 hk hh
    simp only [sim, mulDepth_rnd, Prog.mulDepth, Nat.add_zero]
    by_cases hne : held = []
    · subst hne
      simp
    · rw [ite_eq_right hne]
      calc _ ≤ 1 := by split_ifs <;> omega
        _ ≤ _ := (Nat.le_div_iff_mul_le hℓ).2 (by have := hh hne; omega)
  | succ k ih =>
    intro n as ds g held h0 hk hh
    rw [sim_succ]
    simp only [mulDepth_rnd]
    refine step_bound hℓ (ih _ _ _ _ _ _ (by omega) ?_) ?_
    · intro _
      by_cases hk' : kept held h0 ℓ n = []
      · rw [ite_eq_left hk']
        omega
      · rw [ite_eq_right hk']
        have hne : held ≠ [] := fun h => hk' (by rw [kept]; split_ifs <;> simp [h])
        have := hh hne
        omega
    · by_cases hF : held ≠ [] ∧ h0 + ℓ ≤ n
      · right
        have hk' : kept held h0 ℓ n = [] := by rw [kept, ite_eq_left hF]
        rw [hk', ite_eq_right hF.1]
        refine ⟨by split_ifs <;> omega, ?_, ?_⟩
        · simp only [List.nil_append]
          split_ifs <;> omega
        · simp only [List.nil_append]
          split_ifs <;> omega
      · left
        have hf : fired held h0 ℓ n = [] := by rw [fired, ite_eq_right hF]
        have hk' : kept held h0 ℓ n = held := by rw [kept, ite_eq_right hF]
        rw [hf, hk', List.nil_append, any_filter_not_mul]
        refine ⟨ite_eq_right (by simp), ?_⟩
        by_cases hh0 : held = []
        · subst hh0
          simp only [List.nil_append]
          split_ifs <;> omega
        · simp [hh0]

/-! ## Capping a program at `(D, Q, L)` -/

/-- The program `P`, stopped with `d` at the first round that would take it past `D` rounds with a multiplication, `Q`
key queries or `L` ring queries. -/
def cap (d : α) : ℕ → ℕ → ℕ → Prog (ringRO B w p) α → Prog (ringRO B w p) α
  | _, _, _, .ret a => .ret a
  | D, Q, L, .round qs k =>
    if (if qs.any RingQry.isMul then 1 else 0) ≤ D ∧ qs.countP RingQry.isKey ≤ Q ∧
        qs.countP (fun q => !q.isKey) ≤ L then
      .round qs fun r => cap d (D - if qs.any RingQry.isMul then 1 else 0) (Q - qs.countP RingQry.isKey)
        (L - qs.countP fun q => !q.isKey) (k r)
    else .ret d

theorem cap_bounded (d : α) (f : (ringRO B w p).Qry → (ringRO B w p).Ans) (P : Prog (ringRO B w p) α) :
    ∀ D Q L, (cap d D Q L P).mulDepth f ≤ D ∧ (cap d D Q L P).keyWork f ≤ Q ∧ (cap d D Q L P).ringWork f ≤ L := by
  induction P with
  | ret a =>
    intro D Q L
    simp [cap, Prog.mulDepth, Prog.keyWork, Prog.ringWork]
  | round qs k ih =>
    intro D Q L
    simp only [cap]
    by_cases h : (if qs.any RingQry.isMul then 1 else 0) ≤ D ∧ qs.countP RingQry.isKey ≤ Q ∧
        qs.countP (fun q => !q.isKey) ≤ L
    · rw [ite_eq_left h]
      obtain ⟨h1, h2, h3⟩ := ih (qs.map f) (D - if qs.any RingQry.isMul then 1 else 0)
        (Q - qs.countP RingQry.isKey) (L - qs.countP fun q => !q.isKey)
      simp only [Prog.mulDepth, Prog.keyWork, Prog.ringWork]
      generalize (if qs.any RingQry.isMul then 1 else 0) = m at h h1 h2 h3 ⊢
      omega
    · rw [ite_eq_right h]
      simp [Prog.mulDepth, Prog.keyWork, Prog.ringWork]

theorem cap_run (d : α) (f : (ringRO B w p).Qry → (ringRO B w p).Ans) (P : Prog (ringRO B w p) α) :
    ∀ D Q L, P.mulDepth f ≤ D → P.keyWork f ≤ Q → P.ringWork f ≤ L → (cap d D Q L P).run f = P.run f := by
  induction P with
  | ret a =>
    intro D Q L _ _ _
    simp [cap]
  | round qs k ih =>
    intro D Q L hD hQ hL
    simp only [Prog.mulDepth, Prog.keyWork, Prog.ringWork] at hD hQ hL
    simp only [cap]
    rw [ite_eq_left ⟨by omega, by omega, by omega⟩]
    simp only [Prog.run]
    exact ih (qs.map f) _ _ _ (by omega) (by omega) (by omega)

/-! ## A window as a `RingBounded` program -/

/-- **A window as a ring responder.** When every multiplication takes at least `ℓ > 0` ticks to answer and the window
spans at most `D·ℓ` ticks, some `RingBounded D Q L` program yields what the window yields, whatever the oracle answers,
wherever the window asks at most `Q` keys and `L` ring operations. -/
theorem window_responder (W : RingWindow B w α) (ℓ D Q L : ℕ) (d : α) (hℓ : 0 < ℓ)
    (hlat : ∀ q : RingQry B w, q.isMul = true → ℓ ≤ W.lat q) (hspan : W.span ≤ D * ℓ) :
    ∃ P : Prog (ringRO B w p) α, P.RingBounded D Q L ∧
      ∀ f : RingQry B w → Bits w, W.keys f ≤ Q → W.ops f ≤ L → P.run f = W.result f := by
  refine ⟨cap d D Q L (sim W ℓ W.span 0 [] [] (fun _ _ => false) [] 0), fun ω => cap_bounded d _ _ D Q L, ?_⟩
  intro f hQ hL
  unfold RingWindow.keys at hQ
  unfold RingWindow.ops at hL
  obtain ⟨h1, h2, h3⟩ := sim_spec (p := p) W hℓ hlat f W.span 0 (fun _ _ => false) [] 0 (by omega)
    (fun e he => by simp [RingWindow.trace] at he) (fun e he => by simp at he)
  have hd := sim_mulDepth (p := p) W hℓ f W.span 0 [] [] (fun _ _ => false) [] 0 (by omega) (fun h => absurd rfl h)
  have hD : (W.span + ℓ - 1) / ℓ ≤ D := by
    have := (Nat.div_lt_iff_lt_mul hℓ).2 (show W.span + ℓ - 1 < (D + 1) * ℓ by
      rw [Nat.add_mul, Nat.one_mul]
      generalize D * ℓ = M at hspan ⊢
      omega)
    omega
  simp only [RingWindow.trace, List.countP_nil, Nat.add_zero, List.length_nil] at h1 h2 h3
  rw [cap_run d f _ D Q L (le_trans hd (by simpa using hD)) (by omega) (by omega)]
  exact h1

theorem seqAnswers_congr {I : IdealModel} {ℓ' S B' : ℕ} (f : I.Qry → I.Ans)
    (step step' : ℕ → Bits S → Fin B' → Prog I (Bits ℓ' × Bits S))
    (h : ∀ j σ i, (step j σ i).run f = (step' j σ i).run f) :
    ∀ (is : List (Fin B')) j σ, seqAnswers f step j is σ = seqAnswers f step' j is σ := by
  intro is
  induction is with
  | nil => intro j σ; rfl
  | cons i is ih => intro j σ; simp only [seqAnswers, h j σ i, ih]

end Latency

open Pous in
theorem B2Window : Guarantees.B2Window := by
  intro B w p _ sch N D Q L ℓ Wd hℓ hlat hspan
  choose P hP hrun using fun W pp j σ i =>
    Latency.window_responder (p := p) (Wd W pp j σ i) ℓ D Q L (fun _ => false, fun _ => false) hℓ (hlat W pp j σ i)
      (hspan W pp j σ i)
  exact ⟨⟨P, hP⟩, hrun⟩

theorem B2WindowAudit : Guarantees.B2WindowAudit := by
  intro B w p _ sch N D Q L ℓ k δ ε hsec W A₁ Wd hℓ hlat hspan hbud
  obtain ⟨A₂, hA₂⟩ := B2Window sch N D Q L ℓ Wd hℓ hlat hspan
  refine (pr_mono fun x hx => ?_).trans (hsec W A₁ A₂)
  unfold Pous.seqAcceptsOn
  dsimp only
  refine Eq.trans (Latency.seqAnswers_congr ((Pous.ringRO B w p).answer x.1) (A₂.step W (sch.enc W x.1 x.2.1).1)
    (fun j σ i => Pous.Prog.ret ((Wd W (sch.enc W x.1 x.2.1).1 j σ i).result ((Pous.ringRO B w p).answer x.1)))
    (fun j σ i => hA₂ W (sch.enc W x.1 x.2.1).1 j σ i _ (hbud x.1 _ _ j σ i).1 (hbud x.1 _ _ j σ i).2)
    (List.ofFn x.2.2) 0 _) hx

theorem B2WindowP2Ring : Guarantees.B2WindowP2Ring := by
  intro D Q L ℓ hsec W A₁ Wd hℓ hlat hspan hbud
  exact B2WindowAudit _ _ _ _ _ _ _ _ _ hsec W A₁ Wd hℓ hlat hspan hbud

end Accounting.SecurityProofs
