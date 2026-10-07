import Proofs.Flock.Recursive.Defs

/-!
# Recursion: commit-and-prove composition

`commit_and_prove` bounds the recursive audit's soundness error, `Pr[the outer session accepts ∧ P's statement for the
drawn units is false]`, by four terms, given **reference openings** `ref ω p c`, one message for each commitment `c`
sent after the draw `ω` and the transcript `p`:

* `εo`, outer soundness: the outer session accepts but the witness decoded from the prover's strategy for it and its
  output (`dec`) does not make the inner verifier accept its own statement and messages on the transcript's public coins;
* `εP`, the circuit's binding: the decoded statement is not `x R ω`, the statement of the circuit registered behind
  `c_P` for the draw;
* `εb k`, stage `k`'s binding: the decoded round-`k` message is not the reference opening of `c_k`;
* `εin`, inner soundness (`InnerSound`) at the simulated inner prover (`refStrat`), which sends `ref p c_k` at the node
  where the recursive prover sends `c_k`, before the coin `ρ_k`.

Off those events the decoded messages are the reference openings, so the inner verifier accepts `x R ω` on the
relabelled transcript (`zip_eq_relabel`), which is the simulated inner prover's (`leafAvg_relabel`). The reference
openings are anything the hypotheses hold for: `Recursion.Fork` builds them, as the plurality opening at each node, and
bounds `εb k` under `cr/sha-512`.

`commit_and_prove_upto` asks less of a stage: the decoded message need only agree with the reference opening where the
inner verifier reads it (`same k` at the transcript's coins, under which the inner verdict is invariant), so that a round
whose message is an oracle, which the outer witness holds only at its queried rows, binds too. `commit_and_prove` is its
case `same k := Eq`.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit
open scoped ENNReal

theorem prob_exists_finset_le {α ι : Type} (S : Finset ι) (p : ι → α → Prop) (g : Game α) (s : Strategy g) :
    prob (fun a => ∃ i ∈ S, p i a) g s ≤ ∑ i ∈ S, prob (p i) g s := by
  classical
  induction S using Finset.induction_on with
  | empty => simp [prob_false]
  | insert j S hj ih =>
      rw [Finset.sum_insert hj]
      refine (prob_mono (q := fun a => p j a ∨ ∃ i ∈ S, p i a) (fun a ⟨i, hi, hp⟩ => ?_) g s).trans
        ((prob_or_le _ _ g s).trans (add_le_add_right ih _))
      rcases Finset.mem_insert.1 hi with rfl | hi
      · exact Or.inl hp
      · exact Or.inr ⟨i, hi, hp⟩

/-- The union bound at one leaf: an event inside `A ∪ B ∪ ⋃_{k ∈ S} C_k`, or inside a fact `Q` fixed at the leaf. -/
theorem prob_le_union {α : Type} (g : Game α) (s : Strategy g) (P A B : α → Prop) (S : Finset ℕ)
    (Ck : ℕ → α → Prop) (Q : Prop) [Decidable Q] (h : ∀ a, P a → A a ∨ B a ∨ (∃ k ∈ S, Ck k a) ∨ Q) :
    prob P g s ≤ prob A g s + prob B g s + ∑ k ∈ S, prob (Ck k) g s + if Q then 1 else 0 := by
  have hQ : prob (fun _ => Q) g s = if Q then 1 else 0 := by
    by_cases hq : Q
    · simp only [hq, ite_true]; exact prob_true g s
    · simp only [hq, ite_false]; exact prob_false g s
  calc prob P g s ≤ prob (fun a => A a ∨ B a ∨ (∃ k ∈ S, Ck k a) ∨ Q) g s := prob_mono h g s
    _ ≤ prob A g s + (prob B g s + (prob (fun a => ∃ k ∈ S, Ck k a) g s + prob (fun _ => Q) g s)) := by
        refine (prob_or_le _ _ g s).trans ?_
        gcongr
        refine (prob_or_le _ _ g s).trans ?_
        gcongr
        exact prob_or_le _ _ g s
    _ ≤ prob A g s + (prob B g s + (∑ k ∈ S, prob (Ck k) g s + prob (fun _ => Q) g s)) := by
        gcongr
        exact prob_exists_finset_le S Ck g s
    _ = _ := by rw [hQ, add_assoc, add_assoc]

theorem rel_eq_iff {α : Type} {a b : Option α} : Option.Rel (fun x y => x = y) a b ↔ a = b := by
  cases a <;> cases b <;> simp

theorem zip_congr {α γ : Type} : ∀ (ms ms' : List α) (ρs : List γ),
    (∀ k < ρs.length, ms[k]? = ms'[k]?) → ms.zip ρs = ms'.zip ρs
  | _, _, [], _ => by simp
  | [], [], _ :: _, _ => rfl
  | [], _ :: _, _ :: _, h => by simpa using h 0 (by simp)
  | _ :: _, [], _ :: _, h => by simpa using h 0 (by simp)
  | a :: ms, a' :: ms', _ :: ρs, h => by
      have h0 : a = a' := by simpa using h 0 (by simp)
      simp only [List.zip_cons_cons, h0, zip_congr ms ms' ρs fun k hk => by simpa using h (k + 1) (by simp [hk])]

/-- `commit_and_prove_upto`'s `hsame` at `same k := Eq`. -/
theorem Inner.accepts_of_rel_eq (I : Inner) (y : I.Stmt) (ms ms' : List I.Msg) (ρs : List I.Coin)
    (h : ∀ k < ρs.length, Option.Rel (fun a b => a = b) ms[k]? ms'[k]?) (hy : I.accepts y (ms.zip ρs)) :
    I.accepts y (ms'.zip ρs) :=
  zip_congr ms ms' ρs (fun k hk => rel_eq_iff.1 (h k hk)) ▸ hy

variable {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {β : Type}
  {outer : Reg → L.Ω → List (Cm × I.Coin) → Game β}

/-- **Commit-and-prove composition, up to what the inner verifier reads.** `Acc R ω t b`: the outer session accepts with
output `b` after the registration `R`, the draw `ω` and the round transcript `t`; `dec` decodes a witness from the
prover's strategy for the outer session and its output, with its statement `wstmt` and round messages `wmsg`. `same k ρs` relates two
round-`k` messages the inner verifier can't tell apart at the coins `ρs` (`hsame`: its verdict doesn't change when each
message is replaced by a related one), and stage `k`'s binding asks the decoded message only to be related to the
reference opening. -/
theorem commit_and_prove_upto (σ : Strategy (recGame L Reg I Cm outer))
    (Acc : Reg → L.Ω → List (Cm × I.Coin) → β → Prop) (x : Reg → L.Ω → I.Stmt) {W : Type}
    (dec : ∀ R ω t, Strategy (outer R ω t) → β → W) (wstmt : W → I.Stmt) (wmsg : W → List I.Msg)
    (ref : L.Ω → List (Cm × I.Coin) → Cm → I.Msg) (same : ℕ → List I.Coin → I.Msg → I.Msg → Prop)
    (hsame : ∀ y (ms ms' : List I.Msg) (ρs : List I.Coin), (∀ k < ρs.length, Option.Rel (same k ρs) ms[k]? ms'[k]?) →
      I.accepts y (ms.zip ρs) → I.accepts y (ms'.zip ρs))
    {εo εP εin : ℝ≥0∞} (εb : ℕ → ℝ≥0∞)
    (hout : probAt σ (fun ω t s b => Acc σ.1 ω t b ∧
      ¬ I.accepts (wstmt (dec σ.1 ω t s b)) ((wmsg (dec σ.1 ω t s b)).zip (t.map Prod.snd))) ≤ εo)
    (hP : probAt σ (fun ω t s b => Acc σ.1 ω t b ∧ wstmt (dec σ.1 ω t s b) ≠ x σ.1 ω) ≤ εP)
    (hb : ∀ k < I.r, probAt σ (fun ω t s b => Acc σ.1 ω t b ∧ ¬ Option.Rel (same k (t.map Prod.snd))
      (wmsg (dec σ.1 ω t s b))[k]? ((t[k]?).map fun e => ref ω (t.take k) e.1)) ≤ εb k)
    (hin : InnerSound I εin) :
    prob (fun o => Acc o.1 o.2.1 o.2.2.1 o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1)) (recGame L Reg I Cm outer) σ ≤
      εo + εP + ∑ k ∈ Finset.range I.r, εb k + εin := by
  classical
  rw [prob_recGame]
  set Z : L.Ω → List (Cm × I.Coin) → ℝ≥0∞ := fun ω t =>
    if ¬ I.holds (x σ.1 ω) ∧ I.accepts (x σ.1 ω) (relabel (ref ω) t) then 1 else 0 with hZ
  have hleaf : ∀ ω t (s : Strategy (outer σ.1 ω t)), t.length = I.r →
      prob (fun b => Acc σ.1 ω t b ∧ ¬ I.holds (x σ.1 ω)) (outer σ.1 ω t) s ≤
        prob (fun b => Acc σ.1 ω t b ∧
          ¬ I.accepts (wstmt (dec σ.1 ω t s b)) ((wmsg (dec σ.1 ω t s b)).zip (t.map Prod.snd))) (outer σ.1 ω t) s +
        prob (fun b => Acc σ.1 ω t b ∧ wstmt (dec σ.1 ω t s b) ≠ x σ.1 ω) (outer σ.1 ω t) s +
        ∑ k ∈ Finset.range I.r, prob (fun b => Acc σ.1 ω t b ∧ ¬ Option.Rel (same k (t.map Prod.snd))
          (wmsg (dec σ.1 ω t s b))[k]? ((t[k]?).map fun e => ref ω (t.take k) e.1)) (outer σ.1 ω t) s +
        Z ω t := by
    intro ω t s ht
    refine prob_le_union _ _ _ _ _ _ _ _ fun b hb => ?_
    by_contra hno
    simp only [not_or, not_and, not_not, not_exists, Finset.mem_range] at hno
    obtain ⟨h1, h2, h3, h4⟩ := hno
    have hz : (t.mapIdx fun k e => ref ω (t.take k) e.1).zip (t.map Prod.snd) = relabel (ref ω) t :=
      zip_eq_relabel (ref ω) t _ fun k _ => List.getElem?_mapIdx
    have hacc := hsame _ _ _ (t.map Prod.snd) (fun k hk => by
      rw [List.getElem?_mapIdx]; exact h3 k (by simpa [ht] using hk) hb.1) (h1 hb.1)
    exact h4 hb.2 (h2 hb.1 ▸ hz ▸ hacc)
  have hZle : ∀ ω, leafAvg I.r (fun t => (outer σ.1 ω t).map fun b => (σ.1, ω, t, b)) (fun t _ => Z ω t) (σ.2 ω) ≤
      εin := by
    intro ω
    by_cases hx : I.holds (x σ.1 ω)
    · have : (fun (t : List (Cm × I.Coin)) (_ : Strategy ((outer σ.1 ω t).map fun b => (σ.1, ω, t, b))) => Z ω t) =
          fun _ _ => 0 := funext fun t => funext fun _ => by simp [hZ, hx]
      rw [this, leafAvg_const]
      exact zero_le
    · have : (fun (t : List (Cm × I.Coin)) (_ : Strategy ((outer σ.1 ω t).map fun b => (σ.1, ω, t, b))) => Z ω t) =
          fun t _ => if I.accepts (x σ.1 ω) ((fun t => t) (relabel (ref ω) t)) then 1 else 0 :=
        funext fun t => funext fun _ => by simp [hZ, hx]
      rw [this]
      exact (leafAvg_relabel (I.accepts (x σ.1 ω)) I.r _ (ref ω) (fun t => t) (σ.2 ω)).trans_le (hin _ hx _)
  unfold probAt expAt
  calc avg (fun ω => leafAvg I.r (fun t => (outer σ.1 ω t).map fun b => (σ.1, ω, t, b))
        (fun t s => prob (fun b => Acc σ.1 ω t b ∧ ¬ I.holds (x σ.1 ω)) (outer σ.1 ω t)
          (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s)) (σ.2 ω))
      ≤ avg fun ω => leafAvg I.r (fun t => (outer σ.1 ω t).map fun b => (σ.1, ω, t, b))
          (fun t s => prob (fun b => Acc σ.1 ω t b ∧ ¬ I.accepts (wstmt (dec σ.1 ω t (Strategy.ofMap _ _ s) b))
              ((wmsg (dec σ.1 ω t (Strategy.ofMap _ _ s) b)).zip (t.map Prod.snd))) (outer σ.1 ω t)
              (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s) +
            prob (fun b => Acc σ.1 ω t b ∧ wstmt (dec σ.1 ω t (Strategy.ofMap _ _ s) b) ≠ x σ.1 ω) (outer σ.1 ω t)
              (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s) +
            ∑ k ∈ Finset.range I.r, prob (fun b => Acc σ.1 ω t b ∧ ¬ Option.Rel (same k (t.map Prod.snd))
              (wmsg (dec σ.1 ω t (Strategy.ofMap _ _ s) b))[k]? ((t[k]?).map fun e => ref ω (t.take k) e.1))
              (outer σ.1 ω t) (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s) +
            Z ω t) (σ.2 ω) :=
        avg_mono fun ω => leafAvg_mono_len (fun t s ht => hleaf ω t _ ht) _
    _ = probAt σ (fun ω t s b => Acc σ.1 ω t b ∧
          ¬ I.accepts (wstmt (dec σ.1 ω t s b)) ((wmsg (dec σ.1 ω t s b)).zip (t.map Prod.snd))) +
        probAt σ (fun ω t s b => Acc σ.1 ω t b ∧ wstmt (dec σ.1 ω t s b) ≠ x σ.1 ω) +
        ∑ k ∈ Finset.range I.r, probAt σ (fun ω t s b => Acc σ.1 ω t b ∧ ¬ Option.Rel (same k (t.map Prod.snd))
          (wmsg (dec σ.1 ω t s b))[k]? ((t[k]?).map fun e => ref ω (t.take k) e.1)) +
        avg (fun ω => leafAvg I.r (fun t => (outer σ.1 ω t).map fun b => (σ.1, ω, t, b))
          (fun t _ => Z ω t) (σ.2 ω)) := by
        simp only [leafAvg_add, leafAvg_sum, avg_add, avg_finset_sum]
        rfl
    _ ≤ εo + εP + ∑ k ∈ Finset.range I.r, εb k + εin := by
        exact add_le_add (add_le_add (add_le_add hout hP)
          (Finset.sum_le_sum fun k hk => hb k (Finset.mem_range.1 hk))) ((avg_mono hZle).trans (avg_const εin).le)

/-- **Commit-and-prove composition**: `commit_and_prove_upto` where each stage's decoded message must equal the
reference opening. -/
theorem commit_and_prove (σ : Strategy (recGame L Reg I Cm outer))
    (Acc : Reg → L.Ω → List (Cm × I.Coin) → β → Prop) (x : Reg → L.Ω → I.Stmt) {W : Type}
    (dec : ∀ R ω t, Strategy (outer R ω t) → β → W) (wstmt : W → I.Stmt) (wmsg : W → List I.Msg)
    (ref : L.Ω → List (Cm × I.Coin) → Cm → I.Msg) {εo εP εin : ℝ≥0∞} (εb : ℕ → ℝ≥0∞)
    (hout : probAt σ (fun ω t s b => Acc σ.1 ω t b ∧
      ¬ I.accepts (wstmt (dec σ.1 ω t s b)) ((wmsg (dec σ.1 ω t s b)).zip (t.map Prod.snd))) ≤ εo)
    (hP : probAt σ (fun ω t s b => Acc σ.1 ω t b ∧ wstmt (dec σ.1 ω t s b) ≠ x σ.1 ω) ≤ εP)
    (hb : ∀ k < I.r, probAt σ (fun ω t s b => Acc σ.1 ω t b ∧
      (wmsg (dec σ.1 ω t s b))[k]? ≠ (t[k]?).map fun e => ref ω (t.take k) e.1) ≤ εb k)
    (hin : InnerSound I εin) :
    prob (fun o => Acc o.1 o.2.1 o.2.2.1 o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1)) (recGame L Reg I Cm outer) σ ≤
      εo + εP + ∑ k ∈ Finset.range I.r, εb k + εin :=
  commit_and_prove_upto σ Acc x dec wstmt wmsg ref (fun _ _ a b => a = b) I.accepts_of_rel_eq εb hout hP
    (fun k hk => by simpa only [rel_eq_iff, ne_eq] using hb k hk) hin

end FlockSoundness.Discharge.Recursion
