import Proofs.Flock.C3.PubRows.Prefix
import Proofs.Flock.C3.Compose.Honest

/-!
# n1: the tape `decodeRun` reads, and its frame

`decodeRun` takes each stage on a prefix of what the earlier stages leave, so the stages it decodes are always a
prefix of the tape after its first two steps (`decodeRun_split`), and the openings are read from the step that follows
them. The tape is `holoSteps` of its decode exactly when it is `Framed` around them (`holoSteps_iff_framed`): its first
step is a cap, its second the link points, and the stages are followed by one step, the openings. Nothing in the
stages' obligations reads a step past the openings, so at the honest family `HTape` is `HFrame` (`hTape_iff_frame`,
`PubRows/Honest.lean`), an obligation on the decoder `dec`.
-/

namespace FlockC3.PubRows

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled FlockC3.Compose
open FlockVBridge.InnerAccepts (firstSteps)

theorem stageOf_prefix {α : Type} (g : Game α) (l : List Step) (d : α) : (stageOf g l d).1 <+: l := by
  unfold stageOf
  split
  · rename_i h
    exact h.choose_spec.1
  · exact List.nil_prefix

theorem repsOf_prefix {α β : Type} (g₀ : Game α) (g₁ : Game β) (l : List Step) (a : α) (b : β) :
    firstSteps (FlockC3.Compose.repsOf g₀ g₁ l ([], a, [], b)).1 ++
      (FlockC3.Compose.repsOf g₀ g₁ l ([], a, [], b)).2.2.1 <+: l := by
  unfold FlockC3.Compose.repsOf
  split
  · rename_i h
    exact h.choose_spec.1
  · exact List.nil_prefix

theorem firstSteps_length (ts : List Step) : (firstSteps ts).length = 2 * ts.length := by
  induction ts with
  | nil => rfl
  | cons s ts ih =>
    show (Sum.inl ⟨Bool, true⟩ :: s :: firstSteps ts).length = _
    simp only [List.length_cons, ih]
    omega

theorem proverAt_some (α : Type) (a d : α) : proverAt α (some (.inl ⟨α, a⟩)) d = a := by
  unfold proverAt
  split
  · rename_i h
    have hs := h.choose_spec
    generalize h.choose = c at hs ⊢
    simp only [Option.some.injEq, Sum.inl.injEq, Sigma.mk.inj_iff, heq_eq_eq, true_and] at hs
    exact hs.symm
  · rename_i h
    exact absurd ⟨a, rfl⟩ h

theorem coinAt_some (α : Type) (a d : α) : coinAt α (some (.inr ⟨α, a⟩)) d = a := by
  unfold coinAt
  split
  · rename_i h
    have hs := h.choose_spec
    generalize h.choose = c at hs ⊢
    simp only [Option.some.injEq, Sum.inr.injEq, Sigma.mk.inj_iff, heq_eq_eq, true_and] at hs
    exact hs.symm
  · rename_i h
    exact absurd ⟨a, rfl⟩ h

/-- **A run's stages, in the chain's order**: what `holoSteps` puts between the link points and the openings. -/
def stagesOf {e : ExecHolo} (u : C3Run e) : List Step :=
  firstSteps (u.ts 0) ++ u.ts 1 ++ u.sp 0 ++ u.sp 1 ++ u.op ++ u.lg

theorem holoSteps_eq {e : ExecHolo} (u : C3Run e) :
    holoSteps u.cap₀ u.pts (u.ts 0) (u.ts 1) (u.sp 0) (u.sp 1) u.op u.lg u.os =
      .inl ⟨Cap Dig, u.cap₀⟩ :: .inr ⟨Fin 2 → Fin e.cls.mPts → GF128, u.pts⟩ ::
        (stagesOf u ++ [.inl ⟨Opens GF128 Dig × Salts (List UInt8), u.os⟩]) := rfl

theorem eq_append_drop {a l : List Step} (h : a <+: l) : l = a ++ l.drop a.length :=
  (List.prefix_iff_eq_append.1 h).symm

/-- **The decoded stages are a prefix of the tape after the cap and the link points**, and the openings are decoded
from the step after them. -/
theorem decodeRun_split (e : ExecHolo) (l : List Step) :
    l.drop 2 = stagesOf (decodeRun e l) ++ (l.drop 2).drop (stagesOf (decodeRun e l)).length ∧
      (decodeRun e l).os = proverAt (Opens GF128 Dig × Salts (List UInt8))
        ((l.drop 2).drop (stagesOf (decodeRun e l)).length)[0]? (c3Default e).os := by
  set u := decodeRun e l with hu
  have h1 : firstSteps (u.ts 0) ++ u.ts 1 <+: l.drop 2 := repsOf_prefix _ _ _ _ _
  have h2 : u.sp 0 <+: (l.drop 2).drop (2 * (u.ts 0).length + (u.ts 1).length) := stageOf_prefix _ _ _
  have h3 : u.sp 1 <+: ((l.drop 2).drop (2 * (u.ts 0).length + (u.ts 1).length)).drop (u.sp 0).length :=
    stageOf_prefix _ _ _
  have h4 : u.op <+: (((l.drop 2).drop (2 * (u.ts 0).length + (u.ts 1).length)).drop (u.sp 0).length).drop
      (u.sp 1).length := stageOf_prefix _ _ _
  have h5 : u.lg <+: ((((l.drop 2).drop (2 * (u.ts 0).length + (u.ts 1).length)).drop (u.sp 0).length).drop
      (u.sp 1).length).drop u.op.length := stageOf_prefix _ _ _
  have h6 : u.os = proverAt (Opens GF128 Dig × Salts (List UInt8))
      ((((((l.drop 2).drop (2 * (u.ts 0).length + (u.ts 1).length)).drop (u.sp 0).length).drop
        (u.sp 1).length).drop u.op.length).drop u.lg.length)[0]? (c3Default e).os := rfl
  have e1 := eq_append_drop h1
  rw [List.length_append, firstSteps_length] at e1
  generalize (l.drop 2).drop (2 * (u.ts 0).length + (u.ts 1).length) = L₂ at e1 h2 h3 h4 h5 h6
  have e2 := eq_append_drop h2
  generalize L₂.drop (u.sp 0).length = L₃ at e2 h3 h4 h5 h6
  have e3 := eq_append_drop h3
  generalize L₃.drop (u.sp 1).length = L₄ at e3 h4 h5 h6
  have e4 := eq_append_drop h4
  generalize L₄.drop u.op.length = L₅ at e4 h5 h6
  have e5 := eq_append_drop h5
  generalize L₅.drop u.lg.length = L₆ at e5 h6
  have key : l.drop 2 = stagesOf u ++ L₆ := by
    rw [e1, e2, e3, e4, e5]
    simp only [stagesOf, List.append_assoc]
  have hL : (l.drop 2).drop (stagesOf u).length = L₆ := by rw [key, List.drop_left]
  rw [hL]
  exact ⟨key, h6⟩

/-- **The tape is framed around the run's stages**: its first step is a cap, its second the link points, and the
stages are followed by exactly one step, the openings. -/
def Framed {e : ExecHolo} (u : C3Run e) (l : List Step) : Prop :=
  (∃ c : Cap Dig, l[0]? = some (.inl ⟨Cap Dig, c⟩)) ∧
    (∃ p : Fin 2 → Fin e.cls.mPts → GF128, l[1]? = some (.inr ⟨Fin 2 → Fin e.cls.mPts → GF128, p⟩)) ∧
    ∃ o : Opens GF128 Dig × Salts (List UInt8),
      (l.drop 2).drop (stagesOf u).length = [.inl ⟨Opens GF128 Dig × Salts (List UInt8), o⟩]

theorem holoSteps_of_framed {e : ExecHolo} {l : List Step} (h : Framed (decodeRun e l) l) :
    l = holoSteps (decodeRun e l).cap₀ (decodeRun e l).pts ((decodeRun e l).ts 0) ((decodeRun e l).ts 1)
      ((decodeRun e l).sp 0) ((decodeRun e l).sp 1) (decodeRun e l).op (decodeRun e l).lg (decodeRun e l).os := by
  obtain ⟨hsplit, hos⟩ := decodeRun_split e l
  have hcap : (decodeRun e l).cap₀ = proverAt (Cap Dig) l[0]? (c3Default e).cap₀ := rfl
  have hpts : (decodeRun e l).pts = coinAt (Fin 2 → Fin e.cls.mPts → GF128) l[1]? (c3Default e).pts := rfl
  generalize decodeRun e l = u at h hsplit hos hcap hpts ⊢
  obtain ⟨⟨c, hc⟩, ⟨p, hp⟩, ⟨o, ho⟩⟩ := h
  rw [ho, List.getElem?_cons_zero, proverAt_some] at hos
  rw [hc, proverAt_some] at hcap
  rw [hp, coinAt_some] at hpts
  rw [ho] at hsplit
  rw [holoSteps_eq, hcap, hpts, hos, ← hsplit]
  match l, hc, hp with
  | a :: b :: rest, hc, hp =>
    simp only [List.getElem?_cons_zero, List.getElem?_cons_succ, Option.some.injEq] at hc hp
    rw [hc, hp]
    rfl

theorem framed_of_holoSteps {e : ExecHolo} {l : List Step}
    (h : l = holoSteps (decodeRun e l).cap₀ (decodeRun e l).pts ((decodeRun e l).ts 0) ((decodeRun e l).ts 1)
      ((decodeRun e l).sp 0) ((decodeRun e l).sp 1) (decodeRun e l).op (decodeRun e l).lg (decodeRun e l).os) :
    Framed (decodeRun e l) l := by
  generalize decodeRun e l = u at h ⊢
  rw [holoSteps_eq] at h
  subst h
  refine ⟨⟨_, rfl⟩, ⟨_, rfl⟩, ⟨u.os, ?_⟩⟩
  simp only [List.drop_succ_cons, List.drop_zero, List.drop_left]

/-- **A tape is `holoSteps` of its decode exactly where it is framed around its stages.** -/
theorem holoSteps_iff_framed (e : ExecHolo) (l : List Step) :
    l = holoSteps (decodeRun e l).cap₀ (decodeRun e l).pts ((decodeRun e l).ts 0) ((decodeRun e l).ts 1)
      ((decodeRun e l).sp 0) ((decodeRun e l).sp 1) (decodeRun e l).op (decodeRun e l).lg (decodeRun e l).os ↔
      Framed (decodeRun e l) l :=
  ⟨framed_of_holoSteps, holoSteps_of_framed⟩

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

/-- **(n1) The inner transcript's steps are framed around the decoded stages**, at every accepted outcome: V*'s layers
hold the cap first, then the link points, and after the stages only the openings. An obligation on the family's
messages (`wmsg`, at the honest family the decoder `dec`'s), as G's `MsgsAre` is. -/
def HFrame (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ) : Prop :=
  F.AtAccepted dg N k Rw fun R ω t Lb =>
    Framed (F.run R ω t Lb) (schedSteps T CT 0 ((F.wmsg Lb).zip (t.map Prod.snd)))

end FlockC3.PubRows
