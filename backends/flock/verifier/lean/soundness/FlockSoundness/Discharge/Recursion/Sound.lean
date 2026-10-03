import FlockSoundness.Discharge.Recursion.Compose
import FlockSoundness.Discharge.Recursion.Fork

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
rewinding's loss.
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

/-- **Soundness of the recursive audit.** -/
theorem recursive_sound [Fintype V] [DecidableEq V] [Nonempty V] [Fintype Pos]
    (σ : Strategy (recGame L Reg I Cm outer))
    (Acc : Reg → L.Ω → List (Cm × I.Coin) → β → Prop) (x : Reg → L.Ω → I.Stmt) {W : Type}
    (dec : ∀ R ω t, Strategy (outer R ω t) → W) (wstmt : W → I.Stmt) (wmsg : W → List I.Msg)
    (cmt : Cm → Pos → ob.Dg) (msg : (Pos → V) → I.Msg)
    (T : ℕ → ∀ R ω t, Strategy (outer R ω t) → Pos → Finder (Finset ob.Op))
    (εtr : ℕ → ∀ R ω t, Strategy (outer R ω t) → ℝ≥0∞) (κ : ℕ → Pos → ℝ≥0∞) (q : ℕ → ℝ)
    {εo εP εin : ℝ≥0∞}
    (hout : probAt σ (fun ω t s b => Acc σ.1 ω t b ∧
      ¬ I.accepts (wstmt (dec σ.1 ω t s)) ((wmsg (dec σ.1 ω t s)).zip (t.map Prod.snd))) ≤ εo)
    (hP : probAt σ (fun ω t s b => Acc σ.1 ω t b ∧ wstmt (dec σ.1 ω t s) ≠ x σ.1 ω) ≤ εP)
    (htr : ∀ k < I.r, AllAt σ fun ω t s => ∀ (v : Pos → V) (e : Cm × I.Coin), t[k]? = some e →
      prob (fun b => Acc σ.1 ω t b ∧ (wmsg (dec σ.1 ω t s))[k]? ≠ some (msg v)) (outer σ.1 ω t) s ≤
        εtr k σ.1 ω t s +
          ∑ pos, κ k pos * prob (OffAt ob (cmt e.1 pos) (v pos)) (T k σ.1 ω t s pos).g (T k σ.1 ω t s pos).s)
    (hCR : ∀ k < I.r, ∀ ω pos, (forkAt ob σ cmt (T k) k ω pos).CR H (q k))
    (hin : InnerSound I εin) :
    prob (fun o => Acc o.1 o.2.1 o.2.2.1 o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1)) (recGame L Reg I Cm outer) σ ≤
      εo + εP + ∑ k ∈ Finset.range I.r,
        (expAt σ (εtr k σ.1) + (∑ pos, κ k pos) * ENNReal.ofReal (Real.sqrt (q k ^ 2 / 2 ^ (513 : ℝ)))) + εin := by
  sorry

end FlockSoundness.Discharge.Recursion
