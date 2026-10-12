import Proofs.Flock.Soundness.Model.HoloCompiled
import Proofs.Flock.VBridge.InnerAccepts

/-!
# Piece (b), the model's half: candidate 3's table accepts from its stages

`holoSteps` is the inner transcript's steps in the recorded order, stage by stage: the level-0 cap, the link points,
the reps (rep 0's steps first, `firstSteps`, as `tableTape` takes them), rep 0's and rep 1's sparse phases, R's opening,
R's Ligerito and the openings with their salts. `holoTableCL_plays`: when each stage's steps play its game to an
accepting output, the whole plays `holoTableCL` to an accepted output, `inner_accepts_of_reps`' last step at
candidate 3.
-/

namespace FlockC3.Compose

open FlockSoundness FlockSoundness.Game FlockSoundness.Model FlockSoundness.Accounting
open FlockVBridge.InnerAccepts (firstSteps)

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}

/-- **Candidate 3's inner transcript, stage by stage**, in the recorded order. -/
def holoSteps {mPts : ℕ} (cap₀ : Cap D) (pts : Fin 2 → Fin mPts → F) (ts₀ ts₁ sp₀ sp₁ op lg : List Step)
    (os : Opens F D × Salts Sl) : List Step :=
  .inl ⟨Cap D, cap₀⟩ :: .inr ⟨Fin 2 → Fin mPts → F, pts⟩ ::
    (firstSteps ts₀ ++ ts₁ ++ sp₀ ++ sp₁ ++ op ++ lg ++ [.inl ⟨Opens F D × Salts Sl, os⟩])

/-- `playsT_bind` at `>>=`, the form a `do` block's steps take. -/
theorem playsT_bind' {α β : Type} {g : Game α} {f : α → Game β} {ts₁ ts₂ : List Step} {a : α} {b : β}
    (h₁ : PlaysT g ts₁ a) (h₂ : PlaysT (f a) ts₂ b) : PlaysT (g >>= f) (ts₁ ++ ts₂) b :=
  FlockVBridge.InnerAccepts.playsT_bind h₁ h₂

theorem playsT_recv_bind {β M : Type} {f : M → Game β} {m : M} {ts : List Step} {b : β}
    (h : PlaysT (f m) ts b) : PlaysT (recv M >>= f) (.inl ⟨M, m⟩ :: ts) b :=
  .send m h

theorem playsT_draw_bind {β C : Type} {iF : Fintype C} {iN : Nonempty C} {f : C → Game β} {x : C}
    {ts : List Step} {b : β} (h : PlaysT (f x) ts b) : PlaysT (@draw C iF iN >>= f) (.inr ⟨C, x⟩ :: ts) b :=
  .coin x h

theorem playsT_dite_pos {α : Type} {c : Prop} {d : Decidable c} {t : c → Game α} {e : ¬ c → Game α} (hc : c)
    {ts : List Step} {a : α} (h : PlaysT (t hc) ts a) : PlaysT (@dite _ c d t e) ts a := by
  rw [dite_eq_left hc]
  exact h

/-- **Candidate 3's table accepts from its stages**: both reps play `holoRepC` to the outputs their sparse phases read,
each sparse phase plays to its registered claim, R's opening of the two claims plays to `o`, Ligerito on R plays to
`pR`, and the openings accept every result and verify every path (the reps' against `cap₀`, R's against `capReg`). -/
theorem holoTableCL_plays (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (sh : SparseShape) (units : Fin (2 ^ sh.sLog) → Finset ℕ) (τ : Fin (2 ^ sh.sLog) → ℕ) (rsch : Schedule)
    (capReg cap₀ : Cap D) (pts : Fin 2 → Fin mPts → F) (ts₀ ts₁ sp₀ sp₁ op lg : List Step) (os : Opens F D × Salts Sl)
    (r₀ r₁ : ZcOut F S.m × HoloLcOut F S.m (S.kLog - 6)) (pd₀ pd₁ pR : PendL F D) (c₀ c₁ : RegClaim F sh)
    (hreg : 7 ≤ sh.mReg) (o : OpenOut F)
    (h₀ : PlaysT (holoRepC (D := D) A 0 S sch hm (extraClaims S ptLocal mPts pts)) ts₀ (some r₀, pd₀))
    (h₁ : PlaysT (holoRepC (D := D) A 1 S sch hm (extraClaims S ptLocal mPts pts)) ts₁ (some r₁, pd₁))
    (hs₀ : PlaysT (sparsePhase A sh (holoPub S sh units r₀.1 r₀.2) τ r₀.2.h) sp₀ (some c₀))
    (hs₁ : PlaysT (sparsePhase A sh (holoPub S sh units r₁.1 r₁.2) τ r₁.2.h) sp₁ (some c₁))
    (ho : PlaysT (opening A hreg [c₀.toClaim, c₁.toClaim]) op (some o))
    (hl : PlaysT (ligeritoC (D := D) A 2 rsch o.T o.b) lg pR)
    (hv : pd₀.verdict os.1 = true ∧ pd₁.verdict os.1 = true ∧ pR.verdict os.1 = true)
    (hok : OpensOKL H E Lf cap₀ 0 pd₀ os.1 os.2 ∧ OpensOKL H E Lf cap₀ 1 pd₁ os.1 os.2 ∧
      OpensOKL H E Lf capReg 2 pR os.1 os.2) :
    ∃ out, PlaysT (holoTableCL A H E Lf S sch hm ptLocal mPts sh units τ rsch capReg)
      (holoSteps cap₀ pts ts₀ ts₁ sp₀ sp₁ op lg os) out ∧ out.accepted = true := by
  obtain ⟨zc₀, lc₀⟩ := r₀
  obtain ⟨zc₁, lc₁⟩ := r₁
  have hne : ∀ b', holoRepC (D := D) A 1 S sch hm (extraClaims S ptLocal mPts pts) ≠ .ret b' := fun b' => by
    unfold holoRepC
    exact FlockVBridge.InnerAccepts.exact_bind_ne_ret (FlockSoundness.Discharge.Exec.zerocheck_exact A S.m)
      (by simp [FlockSoundness.Discharge.Exec.zcShape]) _ _
  have hint := FlockVBridge.InnerAccepts.playsT_interleave_first h₀ h₁ hne
  have hrp : PlaysT (regPhase A S sh units τ PendL.reject (fun o' => ligeritoC (D := D) A 2 rsch o'.T o'.b)
      (some (zc₀, lc₀), some (zc₁, lc₁))) (sp₀ ++ (sp₁ ++ (op ++ lg))) pR := by
    unfold regPhase
    refine playsT_bind' hs₀ ?_
    refine playsT_bind' hs₁ ?_
    refine playsT_dite_pos hreg ?_
    refine playsT_bind' ho ?_
    exact hl
  have e : holoSteps cap₀ pts ts₀ ts₁ sp₀ sp₁ op lg os = .inl ⟨Cap D, cap₀⟩ :: .inr ⟨Fin 2 → Fin mPts → F, pts⟩ ::
      (firstSteps ts₀ ++ ts₁ ++ (sp₀ ++ (sp₁ ++ (op ++ lg)) ++ [.inl ⟨Opens F D × Salts Sl, os⟩])) := by
    simp only [holoSteps, List.append_assoc]
  rw [e]
  unfold holoTableCL
  exact ⟨_, playsT_recv_bind (playsT_draw_bind (playsT_bind' hint (playsT_bind' hrp (playsT_recv_bind (.ret _))))),
    by simp only [Bool.and_eq_true, decide_eq_true_eq]; exact ⟨⟨⟨hv.1, hv.2.1⟩, hv.2.2⟩, hok⟩⟩

end FlockC3.Compose
