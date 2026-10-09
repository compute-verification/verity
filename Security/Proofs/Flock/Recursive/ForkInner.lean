import Proofs.Flock.Recursive.Stage

/-!
# Recursion: the inner protocol's soundness in fork form, through the recursive audit

`zk_sessions_recursive_fork_inner` (`Compile`) is `zk_sessions_recursive_fork` with `InnerSoundForkOn` in place of
`InnerSound`. The chain from `commit_and_prove_upto` to `zk_sessions_recursive_fork` reads `InnerSound` once, at the
leaves of the recursive prover's own strategy (`commit_and_prove_upto`'s `hZle`), so here it is restated with that use
as its hypothesis, `InnerAtLeaves`: at every draw whose statement is false, the leaves' average of the indicator that
the relabelled transcript is accepted and that some outer outcome the leaf reaches accepts and decodes to it is at most
`εin`.

* `commit_and_prove_on`, `recursive_sound_on`, `zk_sessions_recursive_upto_on`, `zk_sessions_recursive_full_on`,
  `zk_sessions_recursive_fork_on`: the chain at `InnerAtLeaves` for every choice of reference openings, with the inner
  verifier reading each round's message whole (`hsameEq`, as `Flock.Assumptions.VBridge` has it).
* `innerAtLeavesV_of_forkOn`: `InnerSoundForkOn` gives `InnerAtLeaves` at V*'s outer proof unless the recursive prover
  has an inner fork (the body of `InnerForkRec`).
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit StrictCR
open scoped ENNReal

/-- **A bound at every outcome the strategy reaches is a bound**: if every reached outcome with `E` has `F`, then
`Pr[E] ≤ Pr[F]`. -/
theorem prob_mono_reached {α : Type} {E F : α → Prop} :
    ∀ (g : Game α) (s : Strategy g), (∀ a, Game.Outcome g s a → E a → F a) → prob E g s ≤ prob F g s
  | .ret a, _, h => by
      classical
      simp only [prob]
      by_cases hE : E a
      · simp [hE, h a rfl hE]
      · simp [hE]
  | .send _ k, s, h => prob_mono_reached (k s.1) s.2 h
  | @Game.coin _ _ _ _ k, s, h =>
      Finset.sum_le_sum fun c _ => mul_le_mul_right (prob_mono_reached (k c) (s c) fun a ha => h a ⟨c, ha⟩) _

theorem option_eq_of_rel {α : Type} {R : α → α → Prop} (hR : ∀ a b, R a b → a = b) :
    ∀ {a b : Option α}, Option.Rel R a b → a = b
  | none, none, _ => rfl
  | some a, some b, .some h => congrArg some (hR a b h)
  | none, some _, h => nomatch h
  | some _, none, h => nomatch h

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

/-- **Commit-and-prove composition, with the inner term at the prover's own leaves**: `commit_and_prove_upto` with the
inner verifier reading each round's message whole (`hsameEq`) and `InnerAtLeaves` in place of `InnerSound`. -/
theorem commit_and_prove_on (σ : Strategy (recGame L Reg I Cm outer))
    (Acc : Reg → L.Ω → List (Cm × I.Coin) → β → Prop) (x : Reg → L.Ω → I.Stmt) {W : Type}
    (dec : ∀ R ω t, Strategy (outer R ω t) → β → W) (wstmt : W → I.Stmt) (wmsg : W → List I.Msg)
    (ref : L.Ω → List (Cm × I.Coin) → Cm → I.Msg) (same : ℕ → List I.Coin → I.Msg → I.Msg → Prop)
    (hsameEq : ∀ k ρs a b, same k ρs a b → a = b)
    {εo εP εin : ℝ≥0∞} (εb : ℕ → ℝ≥0∞)
    (hout : probAt σ (fun ω t s b => Acc σ.1 ω t b ∧
      ¬ I.accepts (wstmt (dec σ.1 ω t s b)) ((wmsg (dec σ.1 ω t s b)).zip (t.map Prod.snd))) ≤ εo)
    (hP : probAt σ (fun ω t s b => Acc σ.1 ω t b ∧ wstmt (dec σ.1 ω t s b) ≠ x σ.1 ω) ≤ εP)
    (hb : ∀ k < I.r, probAt σ (fun ω t s b => Acc σ.1 ω t b ∧ ¬ Option.Rel (same k (t.map Prod.snd))
      (wmsg (dec σ.1 ω t s b))[k]? ((t[k]?).map fun e => ref ω (t.take k) e.1)) ≤ εb k)
    (hin : InnerAtLeaves σ Acc x dec wmsg ref εin) :
    prob (fun o => Acc o.1 o.2.1 o.2.2.1 o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1)) (recGame L Reg I Cm outer) σ ≤
      εo + εP + ∑ k ∈ Finset.range I.r, εb k + εin := by
  classical
  rw [prob_recGame]
  set Z : (ω : L.Ω) → (t : List (Cm × I.Coin)) → Strategy (outer σ.1 ω t) → ℝ≥0∞ := fun ω t s =>
    if ¬ I.holds (x σ.1 ω) ∧ I.accepts (x σ.1 ω) (relabel (ref ω) t) ∧
        ∃ b, Game.Outcome (outer σ.1 ω t) s b ∧ Acc σ.1 ω t b ∧
          (wmsg (dec σ.1 ω t s b)).zip (t.map Prod.snd) = relabel (ref ω) t then 1 else 0
  have hleaf : ∀ ω t (s : Strategy (outer σ.1 ω t)), t.length = I.r →
      prob (fun b => Acc σ.1 ω t b ∧ ¬ I.holds (x σ.1 ω)) (outer σ.1 ω t) s ≤
        prob (fun b => Acc σ.1 ω t b ∧
          ¬ I.accepts (wstmt (dec σ.1 ω t s b)) ((wmsg (dec σ.1 ω t s b)).zip (t.map Prod.snd))) (outer σ.1 ω t) s +
        prob (fun b => Acc σ.1 ω t b ∧ wstmt (dec σ.1 ω t s b) ≠ x σ.1 ω) (outer σ.1 ω t) s +
        ∑ k ∈ Finset.range I.r, prob (fun b => Acc σ.1 ω t b ∧ ¬ Option.Rel (same k (t.map Prod.snd))
          (wmsg (dec σ.1 ω t s b))[k]? ((t[k]?).map fun e => ref ω (t.take k) e.1)) (outer σ.1 ω t) s +
        Z ω t s := by
    intro ω t s ht
    refine (prob_mono_reached (outer σ.1 ω t) s
      (F := fun b => (Acc σ.1 ω t b ∧
          ¬ I.accepts (wstmt (dec σ.1 ω t s b)) ((wmsg (dec σ.1 ω t s b)).zip (t.map Prod.snd))) ∨
        (Acc σ.1 ω t b ∧ wstmt (dec σ.1 ω t s b) ≠ x σ.1 ω) ∨
        (∃ k ∈ Finset.range I.r, Acc σ.1 ω t b ∧ ¬ Option.Rel (same k (t.map Prod.snd))
          (wmsg (dec σ.1 ω t s b))[k]? ((t[k]?).map fun e => ref ω (t.take k) e.1)) ∨
        (¬ I.holds (x σ.1 ω) ∧ I.accepts (x σ.1 ω) (relabel (ref ω) t) ∧
          ∃ b, Game.Outcome (outer σ.1 ω t) s b ∧ Acc σ.1 ω t b ∧
            (wmsg (dec σ.1 ω t s b)).zip (t.map Prod.snd) = relabel (ref ω) t))
      fun b hr hb => ?_).trans (prob_le_union _ _ _ _ _ _ _ _ fun _ h => h)
    by_cases h1 : I.accepts (wstmt (dec σ.1 ω t s b)) ((wmsg (dec σ.1 ω t s b)).zip (t.map Prod.snd))
    · by_cases h2 : wstmt (dec σ.1 ω t s b) = x σ.1 ω
      · by_cases h3 : ∃ k ∈ Finset.range I.r, Acc σ.1 ω t b ∧ ¬ Option.Rel (same k (t.map Prod.snd))
            (wmsg (dec σ.1 ω t s b))[k]? ((t[k]?).map fun e => ref ω (t.take k) e.1)
        · exact Or.inr (Or.inr (Or.inl h3))
        · have hz : (wmsg (dec σ.1 ω t s b)).zip (t.map Prod.snd) = relabel (ref ω) t :=
            zip_eq_relabel (ref ω) t _ fun k hk => option_eq_of_rel (hsameEq k (t.map Prod.snd))
              (not_not.1 fun hn => h3 ⟨k, Finset.mem_range.2 (by rw [← ht]; exact hk), hb.1, hn⟩)
          refine Or.inr (Or.inr (Or.inr ⟨hb.2, ?_, b, hr, hb.1, hz⟩))
          rw [← hz, ← h2]
          exact h1
      · exact Or.inr (Or.inl ⟨hb.1, h2⟩)
    · exact Or.inl ⟨hb.1, h1⟩
  have hZle : ∀ ω, leafAvg I.r (fun t => (outer σ.1 ω t).map fun b => (σ.1, ω, t, b))
      (fun t s => Z ω t (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s)) (σ.2 ω) ≤ εin := hin
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
            Z ω t (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s)) (σ.2 ω) :=
        avg_mono fun ω => leafAvg_mono_len (fun t s ht => hleaf ω t _ ht) _
    _ = probAt σ (fun ω t s b => Acc σ.1 ω t b ∧
          ¬ I.accepts (wstmt (dec σ.1 ω t s b)) ((wmsg (dec σ.1 ω t s b)).zip (t.map Prod.snd))) +
        probAt σ (fun ω t s b => Acc σ.1 ω t b ∧ wstmt (dec σ.1 ω t s b) ≠ x σ.1 ω) +
        ∑ k ∈ Finset.range I.r, probAt σ (fun ω t s b => Acc σ.1 ω t b ∧ ¬ Option.Rel (same k (t.map Prod.snd))
          (wmsg (dec σ.1 ω t s b))[k]? ((t[k]?).map fun e => ref ω (t.take k) e.1)) +
        avg (fun ω => leafAvg I.r (fun t => (outer σ.1 ω t).map fun b => (σ.1, ω, t, b))
          (fun t s => Z ω t (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s)) (σ.2 ω)) := by
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
`commit_and_prove_on`, the inner verifier reading each round's message whole and `InnerAtLeaves` at every choice of
reference openings in place of `InnerSound`. -/
theorem recursive_sound_on [Fintype V] [DecidableEq V] [Nonempty V] [Fintype Pos]
    (σ : Strategy (recGame L Reg I Cm outer))
    (Acc : Reg → L.Ω → List (Cm × I.Coin) → β → Prop) (x : Reg → L.Ω → I.Stmt) {W : Type}
    (dec : ∀ R ω t, Strategy (outer R ω t) → β → W) (wstmt : W → I.Stmt) (wmsg : W → List I.Msg)
    (same : ℕ → List I.Coin → I.Msg → I.Msg → Prop)
    (hsameEq : ∀ k ρs a b, same k ρs a b → a = b)
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
    (hin : ∀ ref, InnerAtLeaves σ Acc x dec wmsg ref εin) :
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
  refine commit_and_prove_on σ Acc x dec wstmt wmsg (fun ω p c => msg p.length (refAll p.length ω p c)) same hsameEq _
    hout hP (fun k hk => ?_) (hin _)
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

/-- **`zk_sessions_recursive_upto` with the inner term at the prover's own leaves** (`recursive_sound_on`). -/
theorem zk_sessions_recursive_upto_on
    -- V*'s laws: their strata and A3
    (hS : ∀ j, Law.ExecStrata (Vs j).law.σs (Vs j).law.ks (Vs j).law.St)
    (hA3 : ∀ j, Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    -- V*'s sessions at each history and registration
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the count curve's parameters, and the finders' budgets
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ R ω t R₂ j S' R' i, rateZ (planZAt (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    {t' : ℝ} (ht' : 0 ≤ t') (qF qS qW qR : ℝ)
    -- the recursive prover
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    -- the hidden circuit's statement, the messages V*'s layers hold, and V*'s bridge
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (hbr : VBridge zs dg N k Rw x wmsg)
    -- what the inner verifier reads of each round's message
    (same : ℕ → List I.Coin → I.Msg → I.Msg → Prop)
    (hsameEq : ∀ i ρs a b, same i ρs a b → a = b)
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
    {εin : ℝ≥0∞} (hin : ∀ ref, InnerAtLeavesV zs dg N k Rw σ x wmsg ref εin) :
    prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1))
        (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
      expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
          (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundIf dg N (Vs j).law.model k Rw M ρ t' qF qS qW qR τ) s) +
        ∑ i ∈ Finset.range I.r,
          (expAt σ (εtr i σ.1) + (∑ pos, κ i pos) * ENNReal.ofReal (Real.sqrt (q i ^ 2 / 2 ^ (513 : ℝ)))) + εin := by
  have h := recursive_sound_on ob σ (fun R ω t o => AcceptsV (zs R ω t) o) x
    (W := ((j : Fin S) → (Vs j).Bits) × I.Stmt) (fun R ω t _ o => (layers (zs R ω t o.1) dg N k Rw o.2, x R ω))
    Prod.snd (fun w => wmsg w.1) same hsameEq cmt msg T εtr κ q
    (εo := expAt σ fun ω t s => sumV (zs σ.1 ω t) dg N
      (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundIf dg N (Vs j).law.model k Rw M ρ t' qF qS qW qR τ) s) (εP := 0)
    ?_ ?_ htr hCR hin
  · rwa [add_zero] at h
  · exact avg_mono fun ω => leafAvg_mono (fun t s => outerV_le hS hA3 (zs σ.1 ω t) dg N hk hRw hM hρ (hr σ.1 ω t) ht'
      qF qS qW qR (fun Xs => I.accepts (x σ.1 ω) ((wmsg Xs).zip (t.map Prod.snd))) (hbr σ.1 ω t) _) _
  · refine le_of_eq ((congrArg avg (funext fun ω => ?_)).trans (avg_const 0))
    refine (leafAvg_congr_len (G := fun _ _ => 0) (fun t s _ => ?_) _).trans (leafAvg_const 0 _)
    exact le_antisymm ((prob_mono (fun b hb => hb.2 rfl) _ _).trans (prob_false _ _).le) zero_le

/-- **`zk_sessions_recursive_full` with the inner term at the prover's own leaves.** -/
theorem zk_sessions_recursive_full_on
    -- V*'s laws: their strata and A3
    (hS : ∀ j, Law.ExecStrata (Vs j).law.σs (Vs j).law.ks (Vs j).law.St)
    (hA3 : ∀ j, Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    -- V*'s sessions at each history and registration
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the count curve's parameters, and the finders' budgets
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ R ω t R₂ j S' R' i, rateZ (planZAt (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    {t' : ℝ} (ht' : 0 ≤ t') (qF qS qW qR : ℝ)
    -- the recursive prover
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    -- the hidden circuit's statement, the messages V*'s layers hold, and V*'s bridge
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (hbr : VBridge zs dg N k Rw x wmsg)
    -- what the inner verifier reads of each round's message
    (same : ℕ → List I.Coin → I.Msg → I.Msg → Prop)
    (hsameEq : ∀ i ρs a b, same i ρs a b → a = b)
    -- the stages: their commit strings, commitments and messages, V*'s registration and decoding of them, the strata
    -- V*'s laws draw in full, and cr/sha-512 for the fork finders
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg)
    (hcmt : VCommits zs Sk cmt) (hdec : VDecodes wmsg same Sk msg) (hfull : VFull I.r Sk)
    (q : ℕ → ℝ)
    (hCR : ∀ i < I.r, ∀ ω pos,
      (forkAt (obS Vs) σ (cmtS cmt) (fun R ω t s => trialV (zs R ω t) dg N k Rw s) i ω pos).CR H512 (q i))
    -- the inner protocol
    {εin : ℝ≥0∞} (hin : ∀ ref, InnerAtLeavesV zs dg N k Rw σ x wmsg ref εin) :
    prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1))
        (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
      expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
          (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundIf dg N (Vs j).law.model k Rw M ρ t' qF qS qW qR τ) s) +
        ∑ i ∈ Finset.range I.r,
          (expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
              (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackIf dg N k Rw M ρ t' qF qS τ) s) +
            (∑ pos, weightS Vs Sk ρ i pos) * ENNReal.ofReal (Real.sqrt (q i ^ 2 / 2 ^ (513 : ℝ)))) + εin := by
  refine zk_sessions_recursive_upto_on hS hA3 zs dg N hk hRw hM hρ hr ht' qF qS qW qR σ x wmsg hbr same hsameEq
    (obS Vs) (cmtS cmt) (msgS msg) (fun _ R ω t s => trialV (zs R ω t) dg N k Rw s)
    (fun _ R ω t s => sumV (zs R ω t) dg N (fun R₂ j τ => (zs R ω t R₂ j).slackIf dg N k Rw M ρ t' qF qS τ) s)
    (weightS Vs Sk ρ) q (fun i hi ω => AllLeaves.of_forall (fun t s' v e he => ?_) _) hCR hin
  refine (outerV_stage_le hS (zs σ.1 ω t) dg N hk hRw hM hρ (hr σ.1 ω t) ht' qF qS (Sk i) (hfull i hi)
    (fun j pos => cmt j e.1 pos) (fun R₂ j pos hp R' => hcmt i σ.1 ω t e he R₂ j pos hp R')
    (fun Xs => Option.Rel (same i (t.map Prod.snd)) (wmsg Xs)[i]? (some (msgS msg i v)))
    (fun j p => projV j (v ⟨j, p⟩)) (fun Xt hXt => hdec i hi _ Xt (fun j p => projV j (v ⟨j, p⟩)) hXt) _).trans
    (add_le_add le_rfl ?_)
  rw [Fintype.sum_sigma]
  exact Finset.sum_le_sum fun j _ => Finset.sum_le_sum fun p _ =>
    mul_le_mul' le_rfl (trialS_le (zs σ.1 ω t _) dg N k Rw _ j p _ _)

/-- **`zk_sessions_recursive_fork` with the inner term at the prover's own leaves.** -/
theorem zk_sessions_recursive_fork_on
    (hS : ∀ j, Law.ExecStrata (Vs j).law.σs (Vs j).law.ks (Vs j).law.St)
    (hA3 : ∀ j, Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ R ω t R₂ j S' R' i, rateZ (planZAt (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (hbr : VBridge zs dg N k Rw x wmsg)
    (same : ℕ → List I.Coin → I.Msg → I.Msg → Prop)
    (hsameEq : ∀ i ρs a b, same i ρs a b → a = b)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg)
    (hcmt : VCommits zs Sk cmt) (hdec : VDecodes wmsg same Sk msg) (hfull : VFull I.r Sk)
    {εin : ℝ≥0∞} (hin : ∀ ref, InnerAtLeavesV zs dg N k Rw σ x wmsg ref εin) :
    RoundFork zs dg N k Rw σ cmt ∨ SessionFork zs dg N k Rw M σ ∨
      prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1))
          (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
        expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) +
          (I.r : ℝ≥0∞) * expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) + εin := by
  by_cases h1 : RoundFork zs dg N k Rw σ cmt
  · exact Or.inl h1
  by_cases h2 : SessionFork zs dg N k Rw M σ
  · exact Or.inr (Or.inl h2)
  refine Or.inr (Or.inr ?_)
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
  refine (zk_sessions_recursive_full_on hS hA3 zs dg N hk hRw hM hρ hr le_rfl 0 0 0 0 σ x wmsg hbr same hsameEq Sk cmt
    msg hcmt hdec hfull (fun _ => 0) (fun i hi ω pos => Finder.cr_of_not_finds (fun hf => h1 ⟨i, hi, ω, pos, hf⟩) 0)
    hin).trans (add_le_add (add_le_add hB ?_) le_rfl)
  simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, zero_div, Real.sqrt_zero, ENNReal.ofReal_zero,
    mul_zero, add_zero]
  rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  exact mul_le_mul' le_rfl hsl

/-- **The inner protocol's soundness in fork form bounds the inner term at the prover's own leaves**, at every choice of
reference openings, unless the recursive prover has an inner fork (the body of `InnerForkRec`: at some draw and round,
two continuations of `σ` whose decoded inner transcripts `pick` reads as a collision of `H`).

Its proof is open (about 200 lines): read `InnerSoundForkOn` at the simulated inner prover `refStrat` and at `G` := the
inner transcripts `u` such that the leaf of `σ` at `u`'s coins has an outer outcome that accepts and decodes to `u`.
Then the leaves' average is `prob (accepts ∧ G)` against `refStrat` (`leafAvg_relabel` with the leaf's predicate
carried along the coins), and a fork of `refStrat` whose two transcripts satisfy `G` is a fork of `σ` at the same coins
whose two trials reach outcomes decoding to those transcripts (`forkG` on `refStrat` against `forkG` on `σ`, by
induction on the rounds through `contF`). -/
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
  sorry

end Chain

end FlockSoundness.Discharge.Recursion
