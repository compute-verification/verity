import Proofs.Flock.Recursive.Registrants
import Proofs.Flock.Recursive.ForkRuns

/-!
# Recursion: the inner protocol's soundness in fork form, through the recursive audit

`zk_sessions_recursive_fork_inner` (`Compile`) is `zk_sessions_recursive_fork` with `InnerSoundForkOn` in place of
`InnerSound`. The chain from `commit_and_prove_upto` to `zk_sessions_recursive_fork` reads `InnerSound` once, at the
leaves of the recursive prover's own strategy (`commit_and_prove_upto`'s `hZle`), so here it is restated with that use
as its hypothesis, `InnerAtLeaves`: at every draw whose statement is false, the leaves' average of the indicator that
the relabelled transcript is accepted and that some outer outcome the leaf reaches accepts and decodes to it is at most
`εin`.

* `commit_and_prove_on`, `recursive_sound_on`, `zk_sessions_recursive_upto_on`, `zk_sessions_recursive_full_on`,
  `zk_sessions_recursive_fork_on`: the chain at `InnerAtLeavesZ` for every choice of reference openings, at a reading
  `rd` of the decoded transcript that holds wherever each round's message is `same` as the reference opening's: the
  relabelled transcript with `same` equality (`InnerAtLeaves`, as `Flock.Assumptions.VBridge` has it), or with any last
  message where the last round is not committed (`zk_sessions_recursive_fork_innerC`), on V*'s bridge off
  an event `Bad` whose probability is added to the outer term (`zk_sessions_recursive_upto_off`); the fork form takes
  `Bad := RegColl`, with `regFinder` finding a collision as one more fork (`zk_sessions_recursive_fork_read_off`).
* `innerAtLeavesV_of_forkOn`: `InnerSoundForkOn` gives `InnerAtLeaves` at V*'s outer proof unless the recursive prover
  has an inner fork (the body of `InnerForkRec`).
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit StrictCR
open scoped ENNReal

theorem option_eq_of_rel {α : Type} {R : α → α → Prop} (hR : ∀ a b, R a b → a = b) :
    ∀ {a b : Option α}, Option.Rel R a b → a = b
  | none, none, _ => rfl
  | some a, some b, .some h => congrArg some (hR a b h)
  | none, some _, h => nomatch h
  | some _, none, h => nomatch h

theorem ite_one_zero_le {p q : Prop} {dp : Decidable p} {dq : Decidable q} (h : p → q) :
    (if p then (1 : ℝ≥0∞) else 0) ≤ if q then 1 else 0 := by
  by_cases hp : p
  · exact le_of_eq (by rw [ite_eq_left hp, ite_eq_left (h hp)])
  · rw [ite_eq_right hp]
    exact zero_le

section Outer

variable {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {β : Type}
  {outer : Reg → L.Ω → List (Cm × I.Coin) → Game β}

open Classical in
/-- **The inner bound at the recursive prover's own leaves**, at the reference openings `ref`: at every draw `ω`, the
average over the round coins of the indicator that the statement `x R ω` is false, the inner verifier accepts it on the
relabelled transcript (the simulated inner prover's, `refStrat`), and some outcome of the outer session the leaf reaches
accepts (`Acc`) and decodes (`wmsg ∘ dec`) to that transcript, is at most `εin`. -/
def InnerAtLeaves (σ : Strategy (recGame L Reg I Cm outer)) (Acc : Reg → L.Ω → List (Cm × I.Coin) → β → Prop)
    (x : Reg → L.Ω → I.Stmt) {W : Type} (dec : ∀ R ω t, Strategy (outer R ω t) → β → W) (wmsg : W → List I.Msg)
    (ref : L.Ω → List (Cm × I.Coin) → Cm → I.Msg) (εin : ℝ≥0∞) : Prop :=
  ∀ ω, leafAvg I.r (fun t => (outer σ.1 ω t).map fun b => (σ.1, ω, t, b))
    (fun t s => if ¬ I.holds (x σ.1 ω) ∧ I.accepts (x σ.1 ω) (relabel (ref ω) t) ∧
        ∃ b, Game.Outcome (outer σ.1 ω t) (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s) b ∧
          Acc σ.1 ω t b ∧ (wmsg (dec σ.1 ω t (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s) b)).zip
            (t.map Prod.snd) = relabel (ref ω) t then 1 else 0) (σ.2 ω) ≤ εin

open Classical in
/-- **The inner bound at the prover's own leaves, at a reading `rd` of the decoded transcript**: at every draw, the
leaves' average of the indicator that the statement is false and some outer outcome the leaf reaches accepts, decodes to
a transcript `rd` holds of, and the inner verifier accepts that transcript, is at most `εin`. `InnerAtLeaves` is it at
`rd` := the relabelled transcript (`InnerAtLeaves.toZ`). -/
def InnerAtLeavesZ (σ : Strategy (recGame L Reg I Cm outer)) (Acc : Reg → L.Ω → List (Cm × I.Coin) → β → Prop)
    (x : Reg → L.Ω → I.Stmt) {W : Type} (dec : ∀ R ω t, Strategy (outer R ω t) → β → W) (wmsg : W → List I.Msg)
    (rd : L.Ω → List (Cm × I.Coin) → List (I.Msg × I.Coin) → Prop) (εin : ℝ≥0∞) : Prop :=
  ∀ ω, leafAvg I.r (fun t => (outer σ.1 ω t).map fun b => (σ.1, ω, t, b))
    (fun t s => if ¬ I.holds (x σ.1 ω) ∧
        ∃ b, Game.Outcome (outer σ.1 ω t) (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s) b ∧
          Acc σ.1 ω t b ∧
          rd ω t ((wmsg (dec σ.1 ω t (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s) b)).zip
            (t.map Prod.snd)) ∧
          I.accepts (x σ.1 ω) ((wmsg (dec σ.1 ω t (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s) b)).zip
            (t.map Prod.snd)) then 1 else 0) (σ.2 ω) ≤ εin

theorem InnerAtLeaves.toZ {σ : Strategy (recGame L Reg I Cm outer)}
    {Acc : Reg → L.Ω → List (Cm × I.Coin) → β → Prop} {x : Reg → L.Ω → I.Stmt} {W : Type}
    {dec : ∀ R ω t, Strategy (outer R ω t) → β → W} {wmsg : W → List I.Msg}
    {ref : L.Ω → List (Cm × I.Coin) → Cm → I.Msg} {εin : ℝ≥0∞} (h : InnerAtLeaves σ Acc x dec wmsg ref εin) :
    InnerAtLeavesZ σ Acc x dec wmsg (fun ω t u => u = relabel (ref ω) t) εin := by
  intro ω
  refine (leafAvg_mono ?_ _).trans (h ω)
  intro t s
  refine ite_one_zero_le ?_
  rintro ⟨hx, b, hb, ha, hz, hq⟩
  exact ⟨hx, by rw [← hz]; exact hq, b, hb, ha, hz⟩

/-- **Commit-and-prove composition, with the inner term at the prover's own leaves**: `commit_and_prove_upto` with
`InnerAtLeavesZ` in place of `InnerSound`, at a reading `rd` that holds of the decoded transcript wherever every round's
message is `same` as the reference opening's (`hrd`). -/
theorem commit_and_prove_on (σ : Strategy (recGame L Reg I Cm outer))
    (Acc : Reg → L.Ω → List (Cm × I.Coin) → β → Prop) (x : Reg → L.Ω → I.Stmt) {W : Type}
    (dec : ∀ R ω t, Strategy (outer R ω t) → β → W) (wstmt : W → I.Stmt) (wmsg : W → List I.Msg)
    (ref : L.Ω → List (Cm × I.Coin) → Cm → I.Msg) (same : ℕ → List I.Coin → I.Msg → I.Msg → Prop)
    (rd : L.Ω → List (Cm × I.Coin) → List (I.Msg × I.Coin) → Prop)
    (hrd : ∀ ω t (l : List I.Msg), t.length = I.r → (∀ k < I.r, Option.Rel (same k (t.map Prod.snd)) l[k]?
      ((t[k]?).map fun e => ref ω (t.take k) e.1)) → rd ω t (l.zip (t.map Prod.snd)))
    {εo εP εin : ℝ≥0∞} (εb : ℕ → ℝ≥0∞)
    (hout : probAt σ (fun ω t s b => Acc σ.1 ω t b ∧
      ¬ I.accepts (wstmt (dec σ.1 ω t s b)) ((wmsg (dec σ.1 ω t s b)).zip (t.map Prod.snd))) ≤ εo)
    (hP : probAt σ (fun ω t s b => Acc σ.1 ω t b ∧ wstmt (dec σ.1 ω t s b) ≠ x σ.1 ω) ≤ εP)
    (hb : ∀ k < I.r, probAt σ (fun ω t s b => Acc σ.1 ω t b ∧ ¬ Option.Rel (same k (t.map Prod.snd))
      (wmsg (dec σ.1 ω t s b))[k]? ((t[k]?).map fun e => ref ω (t.take k) e.1)) ≤ εb k)
    (hin : InnerAtLeavesZ σ Acc x dec wmsg rd εin) :
    prob (fun o => Acc o.1 o.2.1 o.2.2.1 o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1)) (recGame L Reg I Cm outer) σ ≤
      εo + εP + ∑ k ∈ Finset.range I.r, εb k + εin := by
  classical
  rw [prob_recGame]
  set zLeaf : (ω : L.Ω) → (t : List (Cm × I.Coin)) → Strategy (outer σ.1 ω t) → ℝ≥0∞ := fun ω t s =>
    if ¬ I.holds (x σ.1 ω) ∧ ∃ b, Game.Outcome (outer σ.1 ω t) s b ∧ Acc σ.1 ω t b ∧
        rd ω t ((wmsg (dec σ.1 ω t s b)).zip (t.map Prod.snd)) ∧
        I.accepts (x σ.1 ω) ((wmsg (dec σ.1 ω t s b)).zip (t.map Prod.snd)) then 1 else 0
  have hleaf : ∀ ω t (s : Strategy (outer σ.1 ω t)), t.length = I.r →
      prob (fun b => Acc σ.1 ω t b ∧ ¬ I.holds (x σ.1 ω)) (outer σ.1 ω t) s ≤
        prob (fun b => Acc σ.1 ω t b ∧
          ¬ I.accepts (wstmt (dec σ.1 ω t s b)) ((wmsg (dec σ.1 ω t s b)).zip (t.map Prod.snd))) (outer σ.1 ω t) s +
        prob (fun b => Acc σ.1 ω t b ∧ wstmt (dec σ.1 ω t s b) ≠ x σ.1 ω) (outer σ.1 ω t) s +
        ∑ k ∈ Finset.range I.r, prob (fun b => Acc σ.1 ω t b ∧ ¬ Option.Rel (same k (t.map Prod.snd))
          (wmsg (dec σ.1 ω t s b))[k]? ((t[k]?).map fun e => ref ω (t.take k) e.1)) (outer σ.1 ω t) s +
        zLeaf ω t s := by
    intro ω t s ht
    refine (Game.prob_mono_outcome (outer σ.1 ω t) s
      (q := fun b => (Acc σ.1 ω t b ∧
          ¬ I.accepts (wstmt (dec σ.1 ω t s b)) ((wmsg (dec σ.1 ω t s b)).zip (t.map Prod.snd))) ∨
        (Acc σ.1 ω t b ∧ wstmt (dec σ.1 ω t s b) ≠ x σ.1 ω) ∨
        (∃ k ∈ Finset.range I.r, Acc σ.1 ω t b ∧ ¬ Option.Rel (same k (t.map Prod.snd))
          (wmsg (dec σ.1 ω t s b))[k]? ((t[k]?).map fun e => ref ω (t.take k) e.1)) ∨
        (¬ I.holds (x σ.1 ω) ∧ ∃ b, Game.Outcome (outer σ.1 ω t) s b ∧ Acc σ.1 ω t b ∧
          rd ω t ((wmsg (dec σ.1 ω t s b)).zip (t.map Prod.snd)) ∧
          I.accepts (x σ.1 ω) ((wmsg (dec σ.1 ω t s b)).zip (t.map Prod.snd))))
      fun b hr hb => ?_).trans (prob_le_union _ _ _ _ _ _ _ _ fun _ h => h)
    by_cases h1 : I.accepts (wstmt (dec σ.1 ω t s b)) ((wmsg (dec σ.1 ω t s b)).zip (t.map Prod.snd))
    · by_cases h2 : wstmt (dec σ.1 ω t s b) = x σ.1 ω
      · by_cases h3 : ∃ k ∈ Finset.range I.r, Acc σ.1 ω t b ∧ ¬ Option.Rel (same k (t.map Prod.snd))
            (wmsg (dec σ.1 ω t s b))[k]? ((t[k]?).map fun e => ref ω (t.take k) e.1)
        · exact Or.inr (Or.inr (Or.inl h3))
        · refine Or.inr (Or.inr (Or.inr ⟨hb.2, b, hr, hb.1,
            hrd ω t _ ht fun k hk => not_not.1 fun hn => h3 ⟨k, Finset.mem_range.2 hk, hb.1, hn⟩, ?_⟩))
          rw [← h2]
          exact h1
      · exact Or.inr (Or.inl ⟨hb.1, h2⟩)
    · exact Or.inl ⟨hb.1, h1⟩
  have hZle : ∀ ω, leafAvg I.r (fun t => (outer σ.1 ω t).map fun b => (σ.1, ω, t, b))
      (fun t s => zLeaf ω t (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s)) (σ.2 ω) ≤ εin := hin
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
            zLeaf ω t (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s)) (σ.2 ω) :=
        avg_mono fun ω => leafAvg_mono_len (fun t s ht => hleaf ω t _ ht) _
    _ = probAt σ (fun ω t s b => Acc σ.1 ω t b ∧
          ¬ I.accepts (wstmt (dec σ.1 ω t s b)) ((wmsg (dec σ.1 ω t s b)).zip (t.map Prod.snd))) +
        probAt σ (fun ω t s b => Acc σ.1 ω t b ∧ wstmt (dec σ.1 ω t s b) ≠ x σ.1 ω) +
        ∑ k ∈ Finset.range I.r, probAt σ (fun ω t s b => Acc σ.1 ω t b ∧ ¬ Option.Rel (same k (t.map Prod.snd))
          (wmsg (dec σ.1 ω t s b))[k]? ((t[k]?).map fun e => ref ω (t.take k) e.1)) +
        avg (fun ω => leafAvg I.r (fun t => (outer σ.1 ω t).map fun b => (σ.1, ω, t, b))
          (fun t s => zLeaf ω t (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s)) (σ.2 ω)) := by
        simp only [leafAvg_add, leafAvg_sum, avg_add, avg_finset_sum]
        rfl
    _ ≤ εo + εP + ∑ k ∈ Finset.range I.r, εb k + εin := by
        exact add_le_add (add_le_add (add_le_add hout hP)
          (Finset.sum_le_sum fun k hk => hb k (Finset.mem_range.1 hk))) ((avg_mono hZle).trans (avg_const εin).le)

end Outer

section Sound

variable {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {β : Type}
  {outer : Reg → L.Ω → List (Cm × I.Coin) → Game β}
  {D : Type} {H : List UInt8 → D} {V Pos : Type} (ob : Openings H V)

/-- **Soundness of the recursive audit, with the inner term at the prover's own leaves**: `recursive_sound_upto` on
`commit_and_prove_on`, with `InnerAtLeavesZ` at every choice of reference openings in place of `InnerSound`. -/
theorem recursive_sound_on [Fintype V] [DecidableEq V] [Nonempty V] [Fintype Pos]
    (σ : Strategy (recGame L Reg I Cm outer))
    (Acc : Reg → L.Ω → List (Cm × I.Coin) → β → Prop) (x : Reg → L.Ω → I.Stmt) {W : Type}
    (dec : ∀ R ω t, Strategy (outer R ω t) → β → W) (wstmt : W → I.Stmt) (wmsg : W → List I.Msg)
    (same : ℕ → List I.Coin → I.Msg → I.Msg → Prop)
    (rd : (L.Ω → List (Cm × I.Coin) → Cm → I.Msg) → L.Ω → List (Cm × I.Coin) → List (I.Msg × I.Coin) → Prop)
    (hrd : ∀ (ref : L.Ω → List (Cm × I.Coin) → Cm → I.Msg) ω t (l : List I.Msg), t.length = I.r →
      (∀ k < I.r, Option.Rel (same k (t.map Prod.snd)) l[k]? ((t[k]?).map fun e => ref ω (t.take k) e.1)) →
      rd ref ω t (l.zip (t.map Prod.snd)))
    (cmt : Cm → Pos → ob.Dg) (msg : ℕ → (Pos → V) → I.Msg)
    (T : ℕ → ∀ R ω t, Strategy (outer R ω t) → Pos → Finder (Finset ob.Op))
    (εtr : ℕ → ∀ R ω t, Strategy (outer R ω t) → ℝ≥0∞) (κ : ℕ → Pos → ℝ≥0∞) (q : ℕ → ℝ)
    {εo εP εin : ℝ≥0∞}
    (hout : probAt σ (fun ω t s b => Acc σ.1 ω t b ∧
      ¬ I.accepts (wstmt (dec σ.1 ω t s b)) ((wmsg (dec σ.1 ω t s b)).zip (t.map Prod.snd))) ≤ εo)
    (hP : probAt σ (fun ω t s b => Acc σ.1 ω t b ∧ wstmt (dec σ.1 ω t s b) ≠ x σ.1 ω) ≤ εP)
    (htr : ∀ k < I.r, AllAt σ fun ω t s => ∀ (v : Pos → V) (e : Cm × I.Coin), t[k]? = some e →
      prob (fun b => Acc σ.1 ω t b ∧
          ¬ Option.Rel (same k (t.map Prod.snd)) (wmsg (dec σ.1 ω t s b))[k]? (some (msg k v)))
        (outer σ.1 ω t) s ≤
        εtr k σ.1 ω t s +
          ∑ pos, κ k pos * prob (OffAt ob (cmt e.1 pos) (v pos)) (T k σ.1 ω t s pos).g (T k σ.1 ω t s pos).s)
    (hCR : ∀ k < I.r, ∀ ω pos, (forkAt ob σ cmt (T k) k ω pos).CR H (q k))
    (hin : ∀ ref, InnerAtLeavesZ σ Acc x dec wmsg (rd ref) εin) :
    prob (fun o => Acc o.1 o.2.1 o.2.2.1 o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1)) (recGame L Reg I Cm outer) σ ≤
      εo + εP + ∑ k ∈ Finset.range I.r,
        (expAt σ (εtr k σ.1) + (∑ pos, κ k pos) * ENNReal.ofReal (Real.sqrt (q k ^ 2 / 2 ^ (513 : ℝ)))) + εin := by
  classical
  have hst : ∀ k, k < I.r → ∀ ω, ∃ refv : List (Cm × I.Coin) → Cm → Pos → V,
      leafAvg I.r (fun t => (outer σ.1 ω t).map fun b => (σ.1, ω, t, b))
        (fun t s => prob (fun b => Acc σ.1 ω t b ∧ ¬ Option.Rel (same k (t.map Prod.snd))
            (wmsg (dec σ.1 ω t (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s) b))[k]?
              ((t[k]?).map fun e => msg k (refv (t.take k) e.1)))
          (outer σ.1 ω t) (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s)) (σ.2 ω) ≤
        leafAvg I.r (fun t => (outer σ.1 ω t).map fun b => (σ.1, ω, t, b))
          (fun t s => εtr k σ.1 ω t (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s)) (σ.2 ω) +
        (∑ pos, κ k pos) * ENNReal.ofReal (Real.sqrt (q k ^ 2 / 2 ^ (513 : ℝ))) := fun k hk ω =>
    stage_bind ob cmt (msg k) hk _
      (fun t s => T k σ.1 ω t (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s)) (σ.2 ω)
      (fun t s m => prob (fun b => Acc σ.1 ω t b ∧ ¬ Option.Rel (same k (t.map Prod.snd))
          (wmsg (dec σ.1 ω t (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s) b))[k]? m)
        (outer σ.1 ω t) (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s))
      (fun t s => εtr k σ.1 ω t (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s)) (κ k) (htr k hk ω)
      fun pos => hCR k hk ω pos
  let refAll : ℕ → L.Ω → List (Cm × I.Coin) → Cm → Pos → V := fun k ω =>
    if hk : k < I.r then (hst k hk ω).choose else fun _ _ _ => Classical.arbitrary V
  have hrefAll : ∀ k (hk : k < I.r) ω, refAll k ω = (hst k hk ω).choose := fun k hk ω => dite_eq_left hk
  refine commit_and_prove_on σ Acc x dec wstmt wmsg (fun ω p c => msg p.length (refAll p.length ω p c)) same
    (rd fun ω p c => msg p.length (refAll p.length ω p c)) (hrd _) _ hout hP (fun k hk => ?_) (hin _)
  unfold probAt expAt
  calc avg (fun ω => leafAvg I.r (fun t => (outer σ.1 ω t).map fun b => (σ.1, ω, t, b))
        (fun t s => prob (fun b => Acc σ.1 ω t b ∧ ¬ Option.Rel (same k (t.map Prod.snd))
            (wmsg (dec σ.1 ω t (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s) b))[k]?
              ((t[k]?).map fun e => msg (t.take k).length (refAll (t.take k).length ω (t.take k) e.1)))
          (outer σ.1 ω t) (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s)) (σ.2 ω))
      = avg (fun ω => leafAvg I.r (fun t => (outer σ.1 ω t).map fun b => (σ.1, ω, t, b))
        (fun t s => prob (fun b => Acc σ.1 ω t b ∧ ¬ Option.Rel (same k (t.map Prod.snd))
            (wmsg (dec σ.1 ω t (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s) b))[k]?
              ((t[k]?).map fun e => msg k ((hst k hk ω).choose (t.take k) e.1)))
          (outer σ.1 ω t) (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s)) (σ.2 ω)) := by
        refine congrArg avg (funext fun ω => leafAvg_congr_len (fun t s ht => ?_) _)
        have hlen : (t.take k).length = k := by rw [List.length_take, ht]; exact min_eq_left hk.le
        rw [hlen, hrefAll k hk ω]
    _ ≤ avg fun ω => leafAvg I.r (fun t => (outer σ.1 ω t).map fun b => (σ.1, ω, t, b))
          (fun t s => εtr k σ.1 ω t (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s)) (σ.2 ω) +
        (∑ pos, κ k pos) * ENNReal.ofReal (Real.sqrt (q k ^ 2 / 2 ^ (513 : ℝ))) :=
        avg_mono fun ω => (hst k hk ω).choose_spec
    _ = _ := by rw [avg_add, avg_const]

end Sound

section Leaf

theorem ite_one_zero_iff {p q : Prop} {dp : Decidable p} {dq : Decidable q} (h : p ↔ q) :
    (if p then (1 : ℝ≥0∞) else 0) = if q then 1 else 0 := by
  by_cases hp : p
  · rw [if_pos hp, if_pos (h.1 hp)]
  · rw [if_neg hp, if_neg (mt h.2 hp)]

theorem outcome_of_bind {α β : Type} {f : α → Game β} : ∀ (g : Game α) (s : Strategy g) (t : (a : α) → Strategy (f a))
    {b : β}, Game.Outcome (g.bind f) (Strategy.bind g f s t) b → ∃ a, Game.Outcome g s a ∧ Game.Outcome (f a) (t a) b
  | .ret a, _, _, _, hb => ⟨a, rfl, hb⟩
  | .send _ k, s, t, _, hb => outcome_of_bind (k s.1) s.2 t hb
  | @Game.coin _ _ _ _ k, s, t, _, hb => by
      obtain ⟨c, hb⟩ := hb
      obtain ⟨a, ha, hb⟩ := outcome_of_bind (k c) (s c) t hb
      exact ⟨a, ⟨c, ha⟩, hb⟩

theorem outcome_of_map {α β : Type} (f : α → β) (g : Game α) (s : Strategy g) {b : β}
    (h : Game.Outcome (g.map f) (Strategy.map f g s) b) : ∃ a, Game.Outcome g s a ∧ f a = b :=
  outcome_of_bind (f := fun a => .ret (f a)) g s (fun _ => PUnit.unit) h

theorem outcome_oneRun {β γ : Type} (g : Game β) (s : Strategy g) (pick : β → γ) {a : β}
    (ha : Game.Outcome g s a) : Game.Outcome (StrictCR.oneRun g s pick).g (StrictCR.oneRun g s pick).s (pick a) :=
  Game.Outcome.map_of pick g s ha

theorem outcome_twoRuns {β γ : Type} (g : Game β) (s : Strategy g) (pick : β → β → γ) {a a' : β}
    (ha : Game.Outcome g s a) (ha' : Game.Outcome g s a') :
    Game.Outcome (StrictCR.twoRuns g s pick).g (StrictCR.twoRuns g s pick).s (pick a a') :=
  Game.Outcome.bind_of g (fun b => g.map (pick b)) s (fun b => Strategy.map (pick b) g s) ha
    (Game.Outcome.map_of (pick a) g s ha')

theorem outcome_of_twoRuns {β γ : Type} (g : Game β) (s : Strategy g) (pick : β → β → γ) {p : γ}
    (h : Game.Outcome (StrictCR.twoRuns g s pick).g (StrictCR.twoRuns g s pick).s p) :
    ∃ a a', Game.Outcome g s a ∧ Game.Outcome g s a' ∧ pick a a' = p := by
  obtain ⟨a, ha, h⟩ := outcome_of_bind (f := fun b => g.map (pick b)) g s (fun b => Strategy.map (pick b) g s) h
  obtain ⟨a', ha', h⟩ := outcome_of_map (pick a) g s h
  exact ⟨a, a', ha, ha', h⟩

variable {M C β : Type} [Fintype C] [Nonempty C]

/-- **`V` at the leaf of `s` that the coins of `u` reach**, read at `u`: down the rounds, the message is the one `s`
sends and the coin is `u`'s, and `V` sees the leaf's transcript, its strategy and `u`. -/
def AtLeaf {N : Type} : (r : ℕ) → (f : List (M × C) → Game β) →
    ((t : List (M × C)) → Strategy (f t) → List (N × C) → Prop) → Strategy (rounds M C r f) → List (N × C) → Prop
  | 0, _, V, s, u => V [] s u
  | _ + 1, _, _, _, [] => False
  | r + 1, f, V, s, (a, ρ) :: u =>
      AtLeaf r (fun t => f ((s.1, ρ) :: t)) (fun t s' u' => V ((s.1, ρ) :: t) s' ((a, ρ) :: u')) (s.2 ρ) u

open Classical in
/-- **The simulation, with a predicate of the leaf**: averaged over the round coins, the indicator that the relabelled
transcript satisfies `Q` and the leaf's `V` is the average, against the simulated inner prover, of the indicator that
its transcript satisfies `Q` and `V` at the leaf of `s` its coins reach (`leafAvg_relabel` with `V` true). -/
theorem leafAvg_relabel_at {N δ : Type} (Q : δ → Prop) : ∀ (r : ℕ) (f : List (M × C) → Game β)
    (ref : List (M × C) → M → N) (h : List (N × C) → δ)
    (V : (t : List (M × C)) → Strategy (f t) → List (N × C) → Prop) (s : Strategy (rounds M C r f)),
    leafAvg r f (fun t s' => if Q (h (relabel ref t)) ∧ V t s' (relabel ref t) then 1 else 0) s =
      leafAvg r (fun t => .ret (h t)) (fun u _ => if Q (h u) ∧ AtLeaf r f V s u then 1 else 0)
        (refStrat r f ref h s)
  | 0, _, _, _, _, _ => rfl
  | r + 1, f, ref, h, V, s =>
      congrArg avg (funext fun ρ => leafAvg_relabel_at Q r (fun t => f ((s.1, ρ) :: t))
        (fun p c => ref ((s.1, ρ) :: p) c) (fun t => h ((ref [] s.1, ρ) :: t))
        (fun t s' u' => V ((s.1, ρ) :: t) s' ((ref [] s.1, ρ) :: u')) (s.2 ρ))

/-- **An outcome of the simulated inner prover's run that satisfies `G` is one of the recursive prover's**, when `G` of
a transcript says `V` holds at the leaf its coins reach and `V` there says the leaf's trial reaches that transcript. -/
theorem outcome_contF_ref {N δ : Type} (G : δ → Prop) : ∀ (r : ℕ) (f : List (M × C) → Game β)
    (ref : List (M × C) → M → N) (h : List (N × C) → δ)
    (V : (t : List (M × C)) → Strategy (f t) → List (N × C) → Prop)
    (T : (t : List (M × C)) → Strategy (f t) → StrictCR.Finder δ) (s : Strategy (rounds M C r f)),
    (∀ t s' u, V t s' u → Game.Outcome (T t s').g (T t s').s (h u)) →
    (∀ u, G (h u) → AtLeaf r f V s u) → ∀ d,
    Game.Outcome (contF r (fun t => .ret (h t)) (fun t _ => ⟨.ret (h t), PUnit.unit⟩) (refStrat r f ref h s)).g
      (contF r (fun t => .ret (h t)) (fun t _ => ⟨.ret (h t), PUnit.unit⟩) (refStrat r f ref h s)).s d →
    G d → Game.Outcome (contF r f T s).g (contF r f T s).s d
  | 0, _, _, h, _, _, s, hVT, hG, d, hd, hGd => by
      have hd' : h [] = d := hd
      subst hd'
      exact hVT [] s [] (hG [] hGd)
  | r + 1, f, ref, h, V, T, s, hVT, hG, d, hd, hGd => by
      obtain ⟨ρ, hd⟩ := hd
      exact ⟨ρ, outcome_contF_ref G r (fun t => f ((s.1, ρ) :: t)) (fun p c => ref ((s.1, ρ) :: p) c)
        (fun t => h ((ref [] s.1, ρ) :: t)) (fun t s' u' => V ((s.1, ρ) :: t) s' ((ref [] s.1, ρ) :: u'))
        (fun t => T ((s.1, ρ) :: t)) (s.2 ρ) (fun t s' u hv => hVT _ s' _ hv) (fun u hu => hG _ hu) d hd hGd⟩

/-- **A fork of the simulated inner prover on transcripts satisfying `G` is a fork of the recursive prover**: at the
same round and coins, its two trials reach the two transcripts (`outcome_contF_ref`), so `pick` reads the same
collision. -/
theorem finds_forkG_ref {D N δ : Type} (H : List UInt8 → D) (pk : δ → δ → StrictCR.Pair) (G : δ → Prop) :
    ∀ (k r : ℕ) (f : List (M × C) → Game β) (ref : List (M × C) → M → N) (h : List (N × C) → δ)
    (V : (t : List (M × C)) → Strategy (f t) → List (N × C) → Prop)
    (T : (t : List (M × C)) → Strategy (f t) → StrictCR.Finder δ) (s : Strategy (rounds M C r f)),
    (∀ t s' u, V t s' u → Game.Outcome (T t s').g (T t s').s (h u)) →
    (∀ u, G (h u) → AtLeaf r f V s u) →
    (forkG (pickOn G pk) k r (fun t => .ret (h t)) (fun t _ => ⟨.ret (h t), PUnit.unit⟩)
      (refStrat r f ref h s)).Finds H →
    (forkG pk k r f T s).Finds H
  | 0, 0, _, _, _, _, _, _, _, _, hF => hF
  | _ + 1, 0, _, _, _, _, _, _, _, _, hF => hF
  | 0, r + 1, f, ref, h, V, T, s, hVT, hG, hF => by
      classical
      obtain ⟨p, hp, hc⟩ := hF
      obtain ⟨d, d', hd, hd', rfl⟩ := outcome_of_twoRuns _ _ _ hp
      have hGG : G d ∧ G d' := by
        by_contra hn
        have hnone : pickOn G pk d d' = none := if_neg hn
        rw [hnone] at hc
        obtain ⟨_, _, he, _⟩ := hc
        exact absurd he (by simp)
      have hpk : pickOn G pk d d' = pk d d' := if_pos hGG
      rw [hpk] at hc
      exact ⟨pk d d', outcome_twoRuns _ _ pk (outcome_contF_ref G (r + 1) f ref h V T s hVT hG d hd hGG.1)
        (outcome_contF_ref G (r + 1) f ref h V T s hVT hG d' hd' hGG.2), hc⟩
  | k + 1, r + 1, f, ref, h, V, T, s, hVT, hG, hF => by
      obtain ⟨p, ⟨ρ, hp⟩, hc⟩ := hF
      obtain ⟨p', hp', hc'⟩ := finds_forkG_ref H pk G k r (fun t => f ((s.1, ρ) :: t))
        (fun p c => ref ((s.1, ρ) :: p) c) (fun t => h ((ref [] s.1, ρ) :: t))
        (fun t s' u' => V ((s.1, ρ) :: t) s' ((ref [] s.1, ρ) :: u')) (fun t => T ((s.1, ρ) :: t)) (s.2 ρ)
        (fun t s' u hv => hVT _ s' _ hv) (fun u hu => hG _ hu) ⟨p, hp, hc⟩
      exact ⟨p', ⟨ρ, hp'⟩, hc'⟩

end Leaf

section Chain

open Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed DecodeZL Composed
  ZkReg

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type} {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type}

/-- **`InnerAtLeaves` at V*'s outer proof**: an outcome accepts when every session verifies (`AcceptsV`), and decodes to
the messages `wmsg` reads from the sessions' layers. -/
def InnerAtLeavesV (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (ref : L.Ω → List (Cm × I.Coin) → Cm → I.Msg) (εin : ℝ≥0∞) : Prop :=
  InnerAtLeaves σ (fun R ω t o => AcceptsV (zs R ω t) o) x (W := ((j : Fin S) → (Vs j).Bits) × I.Stmt)
    (fun R ω t _ o => (layers (zs R ω t o.1) dg N k Rw o.2, x R ω)) (fun w => wmsg w.1) ref εin

/-- **`InnerAtLeavesZ` at V*'s outer proof**, as `InnerAtLeavesV` reads its outcomes. -/
def InnerAtLeavesVZ (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (rd : L.Ω → List (Cm × I.Coin) → List (I.Msg × I.Coin) → Prop) (εin : ℝ≥0∞) : Prop :=
  InnerAtLeavesZ σ (fun R ω t o => AcceptsV (zs R ω t) o) x (W := ((j : Fin S) → (Vs j).Bits) × I.Stmt)
    (fun R ω t _ o => (layers (zs R ω t o.1) dg N k Rw o.2, x R ω)) (fun w => wmsg w.1) rd εin

/-- **`zk_sessions_recursive_upto` with the inner term at the prover's own leaves** (`recursive_sound_on`). -/
theorem zk_sessions_recursive_upto_on
    -- V*'s laws: A3
    (hA3 : ∀ j, Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    -- V*'s sessions at each history and registration
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the count curve's parameters, and the finders' budgets
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ R ω t R₂ j S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    {t' : ℝ} (ht' : 0 ≤ t') (qF qS qW qR : ℝ)
    -- the recursive prover
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    -- the hidden circuit's statement, the messages V*'s layers hold, and V*'s bridge
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (Bad : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → Prop) (hbr : VBridgeOff Bad zs dg N k Rw x wmsg) {εX : ℝ≥0∞}
    (hX : prob (fun o => Bad o.1 o.2.1 o.2.2.1 o.2.2.2.1) (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
      εX)
    -- what the inner verifier reads of each round's message
    (same : ℕ → List I.Coin → I.Msg → I.Msg → Prop)
    (rd : (L.Ω → List (Cm × I.Coin) → Cm → I.Msg) → L.Ω → List (Cm × I.Coin) → List (I.Msg × I.Coin) → Prop)
    (hrd : ∀ (ref : L.Ω → List (Cm × I.Coin) → Cm → I.Msg) ω t (l : List I.Msg), t.length = I.r →
      (∀ k < I.r, Option.Rel (same k (t.map Prod.snd)) l[k]? ((t[k]?).map fun e => ref ω (t.take k) e.1)) →
      rd ref ω t (l.zip (t.map Prod.snd)))
    -- the stages: commitments' openings, transfer, and cr/sha-512 for the fork finders
    {V Pos : Type} [Fintype V] [DecidableEq V] [Nonempty V] [Fintype Pos] (ob : Openings H512 V)
    (cmt : Cm → Pos → ob.Dg) (msg : ℕ → (Pos → V) → I.Msg)
    (T : ℕ → ∀ R ω t, Strategy (outerV (zs R ω t) dg N) → Pos → Finder (Finset ob.Op))
    (εtr : ℕ → ∀ R ω t, Strategy (outerV (zs R ω t) dg N) → ℝ≥0∞) (κ : ℕ → Pos → ℝ≥0∞) (q : ℕ → ℝ)
    (htr : ∀ i < I.r, AllAt σ fun ω t s => ∀ (v : Pos → V) (e : Cm × I.Coin), t[i]? = some e →
      prob (fun o => AcceptsV (zs σ.1 ω t) o ∧ ¬ Option.Rel (same i (t.map Prod.snd))
          (wmsg (layers (zs σ.1 ω t o.1) dg N k Rw o.2))[i]? (some (msg i v))) (outerV (zs σ.1 ω t) dg N) s ≤
        εtr i σ.1 ω t s +
          ∑ pos, κ i pos * prob (OffAt ob (cmt e.1 pos) (v pos)) (T i σ.1 ω t s pos).g (T i σ.1 ω t s pos).s)
    (hCR : ∀ i < I.r, ∀ ω pos, (forkAt ob σ cmt (T i) i ω pos).CR H512 (q i))
    -- the inner protocol
    {εin : ℝ≥0∞} (hin : ∀ ref, InnerAtLeavesVZ zs dg N k Rw σ x wmsg (rd ref) εin) :
    prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1))
        (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
      expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
          (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundIf dg N (Vs j).law.model k Rw M ρ t' qF qS qW qR τ) s) + εX +
        ∑ i ∈ Finset.range I.r,
          (expAt σ (εtr i σ.1) + (∑ pos, κ i pos) * ENNReal.ofReal (Real.sqrt (q i ^ 2 / 2 ^ (513 : ℝ)))) + εin := by
  have h := recursive_sound_on ob σ (fun R ω t o => AcceptsV (zs R ω t) o) x
    (W := ((j : Fin S) → (Vs j).Bits) × I.Stmt) (fun R ω t _ o => (layers (zs R ω t o.1) dg N k Rw o.2, x R ω))
    Prod.snd (fun w => wmsg w.1) same rd hrd cmt msg T εtr κ q
    (εo := expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
      (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundIf dg N (Vs j).law.model k Rw M ρ t' qF qS qW qR τ) s) + εX) (εP := 0)
    ?_ ?_ htr hCR hin
  · rwa [add_zero] at h
  · have hsplit : ∀ ω t (s : Strategy (outerV (zs σ.1 ω t) dg N)),
        prob (fun b => AcceptsV (zs σ.1 ω t) b ∧
            ¬ I.accepts (x σ.1 ω) ((wmsg (layers (zs σ.1 ω t b.1) dg N k Rw b.2)).zip (t.map Prod.snd)))
          (outerV (zs σ.1 ω t) dg N) s ≤
        sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundIf dg N (Vs j).law.model k Rw M ρ t' qF qS qW qR τ) s +
          prob (fun b => Bad σ.1 ω t b.1) (outerV (zs σ.1 ω t) dg N) s := by
      intro ω t s
      refine (prob_mono (q := fun b : Reg₂ × VRec Vs Reg' dg N => (AcceptsV (zs σ.1 ω t) b ∧
          ¬ (Bad σ.1 ω t b.1 ∨
            I.accepts (x σ.1 ω) ((wmsg (layers (zs σ.1 ω t b.1) dg N k Rw b.2)).zip (t.map Prod.snd)))) ∨
          Bad σ.1 ω t b.1) (fun b hb => ?_) _ _).trans ((prob_or_le _ _ _ _).trans (add_le_add ?_ le_rfl))
      · obtain ⟨ha, hn⟩ := hb
        by_cases hb : Bad σ.1 ω t b.1
        · exact Or.inr hb
        · exact Or.inl ⟨ha, fun h => h.elim hb hn⟩
      · exact outerV_le_dep hA3 (zs σ.1 ω t) dg N hk hRw hM hρ (hr σ.1 ω t) ht' qF qS qW qR
          (fun R₂ Xs => Bad σ.1 ω t R₂ ∨ I.accepts (x σ.1 ω) ((wmsg Xs).zip (t.map Prod.snd)))
          (fun R₂ τ o h => (Classical.em (Bad σ.1 ω t R₂)).elim Or.inl fun hb => Or.inr (hbr σ.1 ω t R₂ hb τ o h)) s
    refine le_trans (b := expAt σ fun ω t s => sumV (zs σ.1 ω t) dg N
        (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundIf dg N (Vs j).law.model k Rw M ρ t' qF qS qW qR τ) s +
          prob (fun b => Bad σ.1 ω t b.1) (outerV (zs σ.1 ω t) dg N) s)
      (avg_mono fun ω => leafAvg_mono (fun t s => hsplit ω t _) _) ?_
    exact (expAt_add σ _ _).trans_le (add_le_add le_rfl ((prob_recGame _ σ).symm.le.trans hX))
  · refine le_of_eq ((congrArg avg (funext fun ω => ?_)).trans (avg_const 0))
    refine (leafAvg_congr_len (G := fun _ _ => 0) (fun t s _ => ?_) _).trans (leafAvg_const 0 _)
    exact le_antisymm ((prob_mono (fun b hb => hb.2 rfl) _ _).trans (prob_false _ _).le) zero_le

/-- **`zk_sessions_recursive_read` with the inner term at the prover's own leaves.** -/
theorem zk_sessions_recursive_full_on
    -- V*'s laws: A3
    (hA3 : ∀ j, Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    -- V*'s sessions at each history and registration
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the count curve's parameters, and the finders' budgets
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ R ω t R₂ j S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    {t' : ℝ} (ht' : 0 ≤ t') (qF qS qW qR : ℝ)
    -- the recursive prover
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    -- the hidden circuit's statement, the messages V*'s layers hold, and V*'s bridge
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (Bad : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → Prop) (hbr : VBridgeOff Bad zs dg N k Rw x wmsg) {εX : ℝ≥0∞}
    (hX : prob (fun o => Bad o.1 o.2.1 o.2.2.1 o.2.2.2.1) (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
      εX)
    -- what the inner verifier reads of each round's message
    (same : ℕ → List I.Coin → I.Msg → I.Msg → Prop)
    (rd : (L.Ω → List (Cm × I.Coin) → Cm → I.Msg) → L.Ω → List (Cm × I.Coin) → List (I.Msg × I.Coin) → Prop)
    (hrd : ∀ (ref : L.Ω → List (Cm × I.Coin) → Cm → I.Msg) ω t (l : List I.Msg), t.length = I.r →
      (∀ k < I.r, Option.Rel (same k (t.map Prod.snd)) l[k]? ((t[k]?).map fun e => ref ω (t.take k) e.1)) →
      rd ref ω t (l.zip (t.map Prod.snd)))
    -- the stages: their commit strings, commitments and messages, V*'s registration and decoding of them, the units
    -- they are read from, and cr/sha-512 for the fork finders
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg)
    (hcmt : VCommits zs Sk cmt) (hdec : VDecodesR wmsg same Sk msg) (hfull : VFullR I.r Sk)
    (q : ℕ → ℝ)
    (hCR : ∀ i < I.r, ∀ ω pos,
      (forkAt (obS Vs) σ (cmtS cmt) (fun R ω t s => trialV (zs R ω t) dg N k Rw s) i ω pos).CR H512 (q i))
    -- the inner protocol
    {εin : ℝ≥0∞} (hin : ∀ ref, InnerAtLeavesVZ zs dg N k Rw σ x wmsg (rd ref) εin) :
    prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1))
        (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
      expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
          (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundIf dg N (Vs j).law.model k Rw M ρ t' qF qS qW qR τ) s) + εX +
        ∑ i ∈ Finset.range I.r,
          (expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
              (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackIf dg N k Rw M ρ t' qF qS τ) s) +
            (∑ pos, weightS Vs Sk ρ i pos) * ENNReal.ofReal (Real.sqrt (q i ^ 2 / 2 ^ (513 : ℝ)))) + εin := by
  refine zk_sessions_recursive_upto_on hA3 zs dg N hk hRw hM hρ hr ht' qF qS qW qR σ x wmsg Bad hbr hX same rd hrd
    (obS Vs) (cmtS cmt) (msgS msg) (fun _ R ω t s => trialV (zs R ω t) dg N k Rw s)
    (fun _ R ω t s => sumV (zs R ω t) dg N (fun R₂ j τ => (zs R ω t R₂ j).slackIf dg N k Rw M ρ t' qF qS τ) s)
    (weightS Vs Sk ρ) q (fun i hi ω => AllLeaves.of_forall (fun t s' v e he => ?_) _) hCR hin
  refine (outerV_stage_le_read (zs σ.1 ω t) dg N hk hRw hM hρ (hr σ.1 ω t) ht' qF qS (Sk i) (hfull i hi)
    (fun j pos => cmt j e.1 pos) (fun R₂ j hacc pos hp R' => hcmt i σ.1 ω t e he R₂ j hacc pos hp R')
    (fun Xs => Option.Rel (same i (t.map Prod.snd)) (wmsg Xs)[i]? (some (msgS msg i v)))
    (fun j p => projV j (v ⟨j, p⟩)) (fun Xt hXt => hdec i hi _ Xt (fun j p => projV j (v ⟨j, p⟩)) hXt) _).trans
    (add_le_add le_rfl ?_)
  rw [Fintype.sum_sigma]
  exact Finset.sum_le_sum fun j _ => Finset.sum_le_sum fun p _ =>
    mul_le_mul' le_rfl (trialS_le (zs σ.1 ω t _) dg N k Rw _ j p _ _)

/-- **`zk_sessions_recursive_fork_read` with the inner term at the prover's own leaves.** -/
theorem zk_sessions_recursive_fork_on
    (hA3 : ∀ j, Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ R ω t R₂ j S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (hbr : VBridgeOff (RegColl zs) zs dg N k Rw x wmsg)
    (same : ℕ → List I.Coin → I.Msg → I.Msg → Prop)
    (rd : (L.Ω → List (Cm × I.Coin) → Cm → I.Msg) → L.Ω → List (Cm × I.Coin) → List (I.Msg × I.Coin) → Prop)
    (hrd : ∀ (ref : L.Ω → List (Cm × I.Coin) → Cm → I.Msg) ω t (l : List I.Msg), t.length = I.r →
      (∀ k < I.r, Option.Rel (same k (t.map Prod.snd)) l[k]? ((t[k]?).map fun e => ref ω (t.take k) e.1)) →
      rd ref ω t (l.zip (t.map Prod.snd)))
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg)
    (hcmt : VCommits zs Sk cmt) (hdec : VDecodesR wmsg same Sk msg) (hfull : VFullR I.r Sk)
    {εin : ℝ≥0∞} (hin : ∀ ref, InnerAtLeavesVZ zs dg N k Rw σ x wmsg (rd ref) εin) :
    (regFinder zs dg N σ).Finds H512 ∨ RoundFork zs dg N k Rw σ cmt ∨ SessionFork zs dg N k Rw M σ ∨
      prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1))
          (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
        expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) +
          (I.r : ℝ≥0∞) * expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) + εin := by
  by_cases h0 : (regFinder zs dg N σ).Finds H512
  · exact Or.inl h0
  by_cases h1 : RoundFork zs dg N k Rw σ cmt
  · exact Or.inr (Or.inl h1)
  by_cases h2 : SessionFork zs dg N k Rw M σ
  · exact Or.inr (Or.inr (Or.inl h2))
  refine Or.inr (Or.inr (Or.inr ?_))
  have hno := allAt_of_not_exAt h2
  have hB : expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
        (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundIf dg N (Vs j).law.model k Rw M ρ 0 0 0 0 0 τ) s) ≤
      expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
        (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) := by
    refine expAt_mono_all (hno.mono fun ω t s hs => ?_)
    unfold sumV sumS
    exact exG_mono_outcome _ _ fun rs hrs => Finset.sum_le_sum fun j _ =>
      ZkOuter.boundIf_le _ _ _ _ _ fun hf => hs ⟨rs, hrs, j, hf⟩
  have hsl : expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
        (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackIf dg N k Rw M ρ 0 0 0 τ) s) ≤
      expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
        (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) := by
    refine expAt_mono_all (hno.mono fun ω t s hs => ?_)
    unfold sumV sumS
    exact exG_mono_outcome _ _ fun rs hrs => Finset.sum_le_sum fun j _ =>
      ZkOuter.slackIf_le _ _ _ _ fun hf => hs ⟨rs, hrs, j, hf⟩
  refine (zk_sessions_recursive_full_on hA3 zs dg N hk hRw hM hρ hr le_rfl 0 0 0 0 σ x wmsg (RegColl zs) hbr
    (prob_regColl_le zs dg N σ (Finder.cr_of_not_finds h0 0)) same rd hrd Sk cmt msg hcmt hdec hfull (fun _ => 0)
    (fun i hi ω pos => Finder.cr_of_not_finds (fun hf => h1 ⟨i, hi, ω, pos, hf⟩) 0) hin).trans
    (add_le_add (add_le_add ?_ ?_) le_rfl)
  · simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, zero_div, ENNReal.ofReal_zero, add_zero]
    exact hB
  · simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, zero_div, Real.sqrt_zero, ENNReal.ofReal_zero,
      mul_zero, add_zero]
    rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
    exact mul_le_mul' le_rfl hsl

/-- **The inner protocol's soundness in fork form bounds the inner term at the prover's own leaves**, at every choice of
reference openings, unless the recursive prover has an inner fork (the body of `InnerForkRec`: at some draw and round,
two continuations of `σ` whose decoded inner transcripts `pick` reads as a collision of `H`).

It reads `InnerSoundForkOn` at the simulated inner prover `refStrat` and at `G` := the inner transcripts `u` such that
the leaf of `σ` at `u`'s coins has an outer outcome that accepts and decodes to `u` (`AtLeaf`). The leaves' average is
then `prob (accepts ∧ G)` against `refStrat` (`leafAvg_relabel_at`), and a fork of `refStrat` whose two transcripts
satisfy `G` is a fork of `σ` at the same coins (`finds_forkG_ref`). -/
theorem innerAtLeavesV_of_forkOn (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg) {D : Type} (H : List UInt8 → D)
    (pick : I.Stmt → List (I.Msg × I.Coin) → List (I.Msg × I.Coin) → StrictCR.Pair) {εin : ℝ≥0∞}
    (hin : InnerSoundForkOn H I pick εin)
    (hno : ¬ ∃ ω, ∃ i < I.r,
      (forkG (pick (x σ.1 ω)) i I.r (fun t => (outerV (zs σ.1 ω t) dg N).map fun b => (σ.1, ω, t, b))
        (fun t s => StrictCR.oneRun (outerV (zs σ.1 ω t) dg N)
          (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outerV (zs σ.1 ω t) dg N) s)
          (fun o => (wmsg (layers (zs σ.1 ω t o.1) dg N k Rw o.2)).zip (t.map Prod.snd)))
        (σ.2 ω)).Finds H) :
    ∀ ref, InnerAtLeavesV zs dg N k Rw σ x wmsg ref εin := by
  classical
  intro ref ω
  by_cases hx : I.holds (x σ.1 ω)
  · refine le_trans (leafAvg_mono (G := fun _ _ => 0) (fun t s => ?_) _) ?_
    · exact le_of_eq (if_neg fun h => h.1 hx)
    · exact (leafAvg_const 0 _).le.trans zero_le
  let f := fun t : List (Cm × I.Coin) => (outerV (zs σ.1 ω t) dg N).map fun b => (σ.1, ω, t, b)
  let V : (t : List (Cm × I.Coin)) → Strategy (f t) → List (I.Msg × I.Coin) → Prop := fun t s u =>
    ∃ b, Game.Outcome (outerV (zs σ.1 ω t) dg N)
        (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outerV (zs σ.1 ω t) dg N) s) b ∧
      AcceptsV (zs σ.1 ω t) b ∧ (wmsg (layers (zs σ.1 ω t b.1) dg N k Rw b.2)).zip (t.map Prod.snd) = u
  let T : (t : List (Cm × I.Coin)) → Strategy (f t) → StrictCR.Finder (List (I.Msg × I.Coin)) := fun t s =>
    StrictCR.oneRun (outerV (zs σ.1 ω t) dg N) (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outerV (zs σ.1 ω t) dg N) s)
      (fun o => (wmsg (layers (zs σ.1 ω t o.1) dg N k Rw o.2)).zip (t.map Prod.snd))
  let G : List (I.Msg × I.Coin) → Prop := AtLeaf I.r f V (σ.2 ω)
  have hVT : ∀ t s' u, V t s' u → Game.Outcome (T t s').g (T t s').s (id u) := by
    rintro t s' u ⟨b, hb, -, rfl⟩
    exact outcome_oneRun (outerV (zs σ.1 ω t) dg N)
      (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outerV (zs σ.1 ω t) dg N) s')
      (fun o => (wmsg (layers (zs σ.1 ω t o.1) dg N k Rw o.2)).zip (t.map Prod.snd)) hb
  rcases hin (x σ.1 ω) hx G (refStrat I.r f (ref ω) id (σ.2 ω)) with ⟨i, hi, hF⟩ | hle
  · exact (hno ⟨ω, i, hi, finds_forkG_ref H (pick (x σ.1 ω)) G i I.r f (ref ω) id V T (σ.2 ω) hVT
      (fun _ hu => hu) hF⟩).elim
  refine Eq.trans_le ?_ hle
  refine Eq.trans (leafAvg_congr_len (G := fun t s => if I.accepts (x σ.1 ω) (id (relabel (ref ω) t)) ∧
      V t s (relabel (ref ω) t) then 1 else 0) (fun t s _ => ?_) _) ?_
  · exact ite_one_zero_iff ⟨fun h => h.2, fun h => ⟨hx, h⟩⟩
  refine Eq.trans (leafAvg_relabel_at (fun u => I.accepts (x σ.1 ω) u) I.r f (ref ω) id V (σ.2 ω)) ?_
  refine Eq.trans (leafAvg_congr_len (G := fun u s' => prob (fun u => I.accepts (x σ.1 ω) u ∧ G u) (.ret (id u)) s')
    (fun u s' _ => ?_) _) ?_
  · exact ite_one_zero_iff Iff.rfl
  exact (prob_rounds _ _ _ _).symm

end Chain

end FlockSoundness.Discharge.Recursion
