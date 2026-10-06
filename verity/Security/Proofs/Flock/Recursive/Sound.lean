import Proofs.Flock.Recursive.Compose
import Proofs.Flock.Recursive.Fork

/-!
# Recursion: soundness of the recursive audit under `cr/sha-512`

`recursive_sound` is `commit_and_prove` with each stage's reference openings built by `stage_bind`, at the plurality
opening of the node where the stage's commitment is sent, so that what remains are hypotheses Flock proves and the
assumptions:

* `hout` (outer soundness) and `hP` (the registered circuit's binding), as in `commit_and_prove`;
* `htr k`, stage `k`'s transfer: for every candidate value vector `v`, the decoded round-`k` message is `msg v` except
  with probability `εtr k` plus `κ k pos` times the trial's probability of reading `c_k`'s digest at `pos` off `v pos`;
* `hCR k ω pos`: `cr/sha-512` for stage `k`'s fork finder (`forkAt`) at budget `q k`;
* `hin`: `InnerSound`.

The bound is `εo + εP + Σ_{k < r} (E[εtr k] + (Σ_pos κ k pos)·√(q_k²/2^513)) + εin`, the square root being the
rewinding's loss. `recursive_sound_upto` is the same on `commit_and_prove_upto`: stage `k` opens to `msg k v`, and its
transfer bounds the decoded message not being `same k` as that.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit StrictCR
open scoped ENNReal

variable {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {β : Type}
  {outer : Reg → L.Ω → List (Cm × I.Coin) → Game β}
  {D : Type} {H : List UInt8 → D} {V Pos : Type} (ob : Openings H V)

/-- **The recursive audit's fork finder** at stage `k`, the draw `ω` and position `pos`: `forkF` on the rounds after `ω`,
with the trial `T` at the outer session's start. -/
noncomputable def forkAt (σ : Strategy (recGame L Reg I Cm outer)) (cmt : Cm → Pos → ob.Dg)
    (T : ∀ R ω t, Strategy (outer R ω t) → Pos → Finder (Finset ob.Op)) (k : ℕ) (ω : L.Ω) (pos : Pos) :
    Finder Pair :=
  forkF ob cmt pos k I.r (fun t => (outer σ.1 ω t).map fun b => (σ.1, ω, t, b))
    (fun t s => T σ.1 ω t (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s)) (σ.2 ω)

/-- **Soundness of the recursive audit, up to what the inner verifier reads**: `commit_and_prove_upto` with `stage_bind`'s
reference openings. Stage `k` opens to `msg k v` at the value vector `v`, and its transfer bounds the decoded round-`k`
message not being related (`same k`) to it. -/
theorem recursive_sound_upto [Fintype V] [DecidableEq V] [Nonempty V] [Fintype Pos]
    (σ : Strategy (recGame L Reg I Cm outer))
    (Acc : Reg → L.Ω → List (Cm × I.Coin) → β → Prop) (x : Reg → L.Ω → I.Stmt) {W : Type}
    (dec : ∀ R ω t, Strategy (outer R ω t) → β → W) (wstmt : W → I.Stmt) (wmsg : W → List I.Msg)
    (same : ℕ → List I.Coin → I.Msg → I.Msg → Prop)
    (hsame : ∀ y (ms ms' : List I.Msg) (ρs : List I.Coin), (∀ k < ρs.length, Option.Rel (same k ρs) ms[k]? ms'[k]?) →
      I.accepts y (ms.zip ρs) → I.accepts y (ms'.zip ρs))
    (cmt : Cm → Pos → ob.Dg) (msg : ℕ → (Pos → V) → I.Msg)
    (T : ℕ → ∀ R ω t, Strategy (outer R ω t) → Pos → Finder (Finset ob.Op))
    (εtr : ℕ → ∀ R ω t, Strategy (outer R ω t) → ℝ≥0∞) (κ : ℕ → Pos → ℝ≥0∞) (q : ℕ → ℝ)
    {εo εP εin : ℝ≥0∞}
    (hout : probAt σ (fun ω t s b => Acc σ.1 ω t b ∧
      ¬ I.accepts (wstmt (dec σ.1 ω t s b)) ((wmsg (dec σ.1 ω t s b)).zip (t.map Prod.snd))) ≤ εo)
    (hP : probAt σ (fun ω t s b => Acc σ.1 ω t b ∧ wstmt (dec σ.1 ω t s b) ≠ x σ.1 ω) ≤ εP)
    (htr : ∀ k < I.r, AllAt σ fun ω t s => ∀ (v : Pos → V) (e : Cm × I.Coin), t[k]? = some e →
      prob (fun b => Acc σ.1 ω t b ∧ ¬ Option.Rel (same k (t.map Prod.snd)) (wmsg (dec σ.1 ω t s b))[k]? (some (msg k v)))
        (outer σ.1 ω t) s ≤
        εtr k σ.1 ω t s +
          ∑ pos, κ k pos * prob (OffAt ob (cmt e.1 pos) (v pos)) (T k σ.1 ω t s pos).g (T k σ.1 ω t s pos).s)
    (hCR : ∀ k < I.r, ∀ ω pos, (forkAt ob σ cmt (T k) k ω pos).CR H (q k))
    (hin : InnerSound I εin) :
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
  refine commit_and_prove_upto σ Acc x dec wstmt wmsg (fun ω p c => msg p.length (refAll p.length ω p c)) same hsame _
    hout hP (fun k hk => ?_) hin
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

/-- **Soundness of the recursive audit.** `recursive_sound_upto` where each stage's decoded message must equal its
opening `msg v`. -/
theorem recursive_sound [Fintype V] [DecidableEq V] [Nonempty V] [Fintype Pos]
    (σ : Strategy (recGame L Reg I Cm outer))
    (Acc : Reg → L.Ω → List (Cm × I.Coin) → β → Prop) (x : Reg → L.Ω → I.Stmt) {W : Type}
    (dec : ∀ R ω t, Strategy (outer R ω t) → β → W) (wstmt : W → I.Stmt) (wmsg : W → List I.Msg)
    (cmt : Cm → Pos → ob.Dg) (msg : (Pos → V) → I.Msg)
    (T : ℕ → ∀ R ω t, Strategy (outer R ω t) → Pos → Finder (Finset ob.Op))
    (εtr : ℕ → ∀ R ω t, Strategy (outer R ω t) → ℝ≥0∞) (κ : ℕ → Pos → ℝ≥0∞) (q : ℕ → ℝ)
    {εo εP εin : ℝ≥0∞}
    (hout : probAt σ (fun ω t s b => Acc σ.1 ω t b ∧
      ¬ I.accepts (wstmt (dec σ.1 ω t s b)) ((wmsg (dec σ.1 ω t s b)).zip (t.map Prod.snd))) ≤ εo)
    (hP : probAt σ (fun ω t s b => Acc σ.1 ω t b ∧ wstmt (dec σ.1 ω t s b) ≠ x σ.1 ω) ≤ εP)
    (htr : ∀ k < I.r, AllAt σ fun ω t s => ∀ (v : Pos → V) (e : Cm × I.Coin), t[k]? = some e →
      prob (fun b => Acc σ.1 ω t b ∧ (wmsg (dec σ.1 ω t s b))[k]? ≠ some (msg v)) (outer σ.1 ω t) s ≤
        εtr k σ.1 ω t s +
          ∑ pos, κ k pos * prob (OffAt ob (cmt e.1 pos) (v pos)) (T k σ.1 ω t s pos).g (T k σ.1 ω t s pos).s)
    (hCR : ∀ k < I.r, ∀ ω pos, (forkAt ob σ cmt (T k) k ω pos).CR H (q k))
    (hin : InnerSound I εin) :
    prob (fun o => Acc o.1 o.2.1 o.2.2.1 o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1)) (recGame L Reg I Cm outer) σ ≤
      εo + εP + ∑ k ∈ Finset.range I.r,
        (expAt σ (εtr k σ.1) + (∑ pos, κ k pos) * ENNReal.ofReal (Real.sqrt (q k ^ 2 / 2 ^ (513 : ℝ)))) + εin :=
  recursive_sound_upto ob σ Acc x dec wstmt wmsg (fun _ _ a b => a = b) I.accepts_of_rel_eq cmt (fun _ => msg) T εtr κ q
    hout hP (fun k hk => by simpa only [rel_eq_iff, ne_eq] using htr k hk) hCR hin

end FlockSoundness.Discharge.Recursion
