import Proofs.Flock.C3.Opening.Algebra
import Proofs.Flock.C3.HoloValues.Reps
import Proofs.Flock.C3.Sparse.Session
import Proofs.Flock.C3.PubRows.Prefix
import Proofs.Flock.C3.Compose.Honest
import Proofs.Flock.Soundness.OpeningPhase

/-!
# n4: R's opening at the honest family

`hOpening_honest` (stub): `HOpening` at `C3Family.honest`, from the public rows, the reps and their sparse phases (each
on the prefix `decodeRun` takes, so R's opening is decoded from what follows them, on the two registered claims). The
`rec-opening` units are opening-only, O1–O6 at `m = mReg`; `7 ≤ mReg` is `seven_le_mReg`.

* `playsT_prefix_unique`, `stageOf_of_plays`, `stageOf_some_iff`: a game plays on at most one prefix of a tape, so a
  stage's decode is the play's where one plays, and it plays to an output (`some`) iff some prefix plays to one.
* `opening_decodeRun_iff`: R's decoded opening plays to its output iff `opening` plays to an output on a prefix of the
  tape past both sparse phases (`afterSparse`).
* `plays_opening`: on two ring-switch messages, their seven coins each and the batching coins (`openSteps`), `opening`
  plays to `opResult`; it is an output iff both messages pass their claims' checks (`opResult_some`, `opResult_none`).
  So R's decoded opening plays to its output when the tape's messages pass (`opening_of_steps`), and does not when the
  first fails (`not_opening_of_check`).
* `hOpening_of_steps`: `HOpening` at the honest family from `HOpenSteps`, the tape's opening steps and their checks at
  every accepted outcome.
-/

namespace FlockC3.Opening

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockC3.Compose

/-! ## A stage's decode -/

/-- **A game plays on at most one prefix of a tape**, to one output. -/
theorem playsT_prefix_unique {α : Type} {g : Game α} {p : List Step} {a : α} (hg : PlaysT g p a) :
    ∀ {p' l : List Step} {a' : α}, PlaysT g p' a' → p <+: l → p' <+: l → p = p' ∧ a = a' := by
  induction hg with
  | ret b =>
    intro p' l a' h' _ _
    obtain ⟨rfl, rfl⟩ := h'.ret_inv
    exact ⟨rfl, rfl⟩
  | send m _ ih =>
    intro p' l a' h' hp hp'
    obtain ⟨t, rfl⟩ := hp
    rcases p' with _ | ⟨s, ts'⟩
    · cases h'.nil_inv
    · obtain ⟨t', ht'⟩ := hp'
      simp only [List.cons_append, List.cons.injEq] at ht'
      obtain ⟨rfl, e₂⟩ := ht'
      obtain ⟨rfl, rfl⟩ := ih h'.send_msg ⟨t, rfl⟩ ⟨t', e₂⟩
      exact ⟨rfl, rfl⟩
  | coin x _ ih =>
    intro p' l a' h' hp hp'
    obtain ⟨t, rfl⟩ := hp
    rcases p' with _ | ⟨s, ts'⟩
    · cases h'.nil_inv
    · obtain ⟨t', ht'⟩ := hp'
      simp only [List.cons_append, List.cons.injEq] at ht'
      obtain ⟨rfl, e₂⟩ := ht'
      obtain ⟨rfl, rfl⟩ := ih h'.coin_val ⟨t, rfl⟩ ⟨t', e₂⟩
      exact ⟨rfl, rfl⟩

/-- **A stage's decode is the play's** where the stage's game plays on a prefix of the tape. -/
theorem stageOf_of_plays {α : Type} {g : Game α} {l p : List Step} {a : α} (d : α) (hp : p <+: l)
    (hg : PlaysT g p a) : stageOf g l d = (p, a) := by
  have hex : ∃ q : List Step × α, q.1 <+: l ∧ PlaysT g q.1 q.2 := ⟨(p, a), hp, hg⟩
  rw [stageOf, dif_pos hex]
  obtain ⟨e₁, e₂⟩ := playsT_prefix_unique hex.choose_spec.2 hg hex.choose_spec.1 hp
  exact Prod.ext e₁ e₂

/-- **A stage's decode plays to an output** iff its game plays to one on some prefix of the tape. -/
theorem stageOf_some_iff {α : Type} (g : Game (Option α)) (l : List Step) (d : α) :
    PlaysT g (stageOf g l none).1 (some ((stageOf g l none).2.getD d)) ↔ ∃ p a, p <+: l ∧ PlaysT g p (some a) := by
  constructor
  · intro hg
    by_cases hex : ∃ q : List Step × Option α, q.1 <+: l ∧ PlaysT g q.1 q.2
    · have hs : stageOf g l none = hex.choose := by rw [stageOf, dif_pos hex]
      rw [hs] at hg
      exact ⟨_, _, hex.choose_spec.1, hg⟩
    · have hs : stageOf g l none = ([], none) := by rw [stageOf, dif_neg hex]
      rw [hs] at hg
      exact (hex ⟨([], some ((none : Option α).getD d)), ⟨l, rfl⟩, hg⟩).elim
  · rintro ⟨p, a, hp, hg⟩
    rw [stageOf_of_plays none hp hg]
    exact hg

/-! ## R's decoded opening -/

/-- The tape past the cap, the link points, the reps and both sparse phases of the run `u` decoded from it: where
`decodeRun` decodes R's opening. -/
def afterSparse {e : ExecHolo} (u : C3Run e) (l : List Step) : List Step :=
  (((l.drop 2).drop (2 * (u.ts 0).length + (u.ts 1).length)).drop (u.sp 0).length).drop (u.sp 1).length

/-- R's opening on the run's two registered claims. -/
abbrev openingAt {e : ExecHolo} (u : C3Run e) : Game (Option (OpenOut GF128)) :=
  opening execArith (seven_le_mReg e.cls.shape) [(u.claim 0).toClaim, (u.claim 1).toClaim]

theorem decodeRun_op (e : ExecHolo) (l : List Step) :
    (decodeRun e l).op = (stageOf (openingAt (decodeRun e l)) (afterSparse (decodeRun e l) l) none).1 := rfl

theorem decodeRun_opened (e : ExecHolo) (l : List Step) :
    (decodeRun e l).opened =
      (stageOf (openingAt (decodeRun e l)) (afterSparse (decodeRun e l) l) none).2.getD (c3Default e).opened := rfl

/-- **R's decoded opening plays to its output** iff `opening` plays to an output on a prefix of the tape past both
sparse phases. -/
theorem opening_decodeRun_iff (e : ExecHolo) (l : List Step) :
    PlaysT (openingAt (decodeRun e l)) (decodeRun e l).op (some (decodeRun e l).opened) ↔
      ∃ p a, p <+: afterSparse (decodeRun e l) l ∧ PlaysT (openingAt (decodeRun e l)) p (some a) := by
  rw [decodeRun_op, decodeRun_opened]
  exact stageOf_some_iff _ _ _

/-! ## The opening on two claims -/

/-- The opening's steps on two claims: each claim's ring-switch message (O3) and seven coins (O5), then the batching
coins (O6). -/
def openSteps (s : Fin 2 → Fin 128 → GF128) (ρ : Fin 2 → Fin 7 → GF128) (γ : Fin 2 → GF128) : List Step :=
  [.inl ⟨Fin 128 → GF128, s 0⟩, .inr ⟨Fin 7 → GF128, ρ 0⟩, .inl ⟨Fin 128 → GF128, s 1⟩, .inr ⟨Fin 7 → GF128, ρ 1⟩,
    .inr ⟨Fin 2 → GF128, γ⟩]

/-- **The opening plays on its steps**, to the verdict on the switched claims. -/
theorem plays_opening {m : ℕ} (hm : 7 ≤ m) (c : Fin 2 → Model.Claim GF128 m) (s : Fin 2 → Fin 128 → GF128)
    (ρ : Fin 2 → Fin 7 → GF128) (γ : Fin 2 → GF128) :
    PlaysT (opening execArith hm [c 0, c 1]) (openSteps s ρ γ)
      (opResult execArith hm [c 0, c 1] [(c 0, s 0, ρ 0), (c 1, s 1, ρ 1)] γ) := by
  have h₀ : PlaysT (opStep (c 0)) [.inl ⟨Fin 128 → GF128, s 0⟩, .inr ⟨Fin 7 → GF128, ρ 0⟩] (c 0, s 0, ρ 0) :=
    playsT_recv_bind (playsT_draw_bind (.ret _))
  have h₁ : PlaysT (opStep (c 1)) [.inl ⟨Fin 128 → GF128, s 1⟩, .inr ⟨Fin 7 → GF128, ρ 1⟩] (c 1, s 1, ρ 1) :=
    playsT_recv_bind (playsT_draw_bind (.ret _))
  have hm₁ : PlaysT ([c 1].mapM opStep)
      ([.inl ⟨Fin 128 → GF128, s 1⟩, .inr ⟨Fin 7 → GF128, ρ 1⟩] ++ ([] ++ [])) [(c 1, s 1, ρ 1)] := by
    rw [List.mapM_cons, List.mapM_nil]
    exact playsT_bind' h₁ (playsT_bind' (.ret []) (.ret _))
  have hm₀ : PlaysT ([c 0, c 1].mapM opStep)
      ([.inl ⟨Fin 128 → GF128, s 0⟩, .inr ⟨Fin 7 → GF128, ρ 0⟩] ++
        (([.inl ⟨Fin 128 → GF128, s 1⟩, .inr ⟨Fin 7 → GF128, ρ 1⟩] ++ ([] ++ [])) ++ []))
      [(c 0, s 0, ρ 0), (c 1, s 1, ρ 1)] := by
    rw [List.mapM_cons]
    exact playsT_bind' h₀ (playsT_bind' hm₁ (.ret _))
  have e : openSteps s ρ γ = ([.inl ⟨Fin 128 → GF128, s 0⟩, .inr ⟨Fin 7 → GF128, ρ 0⟩] ++
      (([.inl ⟨Fin 128 → GF128, s 1⟩, .inr ⟨Fin 7 → GF128, ρ 1⟩] ++ ([] ++ [])) ++ [])) ++
        [.inr ⟨Fin 2 → GF128, γ⟩] := rfl
  rw [opening_eq, e]
  refine FlockVBridge.InnerAccepts.playsT_bind hm₀ ?_
  exact .coin γ (.ret _)

theorem opResult_some {m : ℕ} (hm : 7 ≤ m) (c : Fin 2 → Model.Claim GF128 m) (s : Fin 2 → Fin 128 → GF128)
    (ρ : Fin 2 → Fin 7 → GF128) (γ : Fin 2 → GF128) (hc : ∀ i, (∑ v, packedWeight (c i) hm v * s i v) = (c i).value) :
    ∃ a, opResult execArith hm [c 0, c 1] [(c 0, s 0, ρ 0), (c 1, s 1, ρ 1)] γ = some a := by
  simp only [opResult]
  split_ifs with hok
  · exact ⟨_, rfl⟩
  · simp [hc 0, hc 1] at hok

theorem opResult_none {m : ℕ} (hm : 7 ≤ m) (c : Fin 2 → Model.Claim GF128 m) (s : Fin 2 → Fin 128 → GF128)
    (ρ : Fin 2 → Fin 7 → GF128) (γ : Fin 2 → GF128) (h0 : (∑ v, packedWeight (c 0) hm v * s 0 v) ≠ (c 0).value) :
    opResult execArith hm [c 0, c 1] [(c 0, s 0, ρ 0), (c 1, s 1, ρ 1)] γ = none := by
  simp only [opResult]
  split_ifs with hok
  · simp [h0] at hok
  · rfl

/-- **R's decoded opening plays to its output** where the tape past both sparse phases begins with the opening's steps
and both messages pass their claims' checks. -/
theorem opening_of_steps (e : ExecHolo) (l : List Step) (s : Fin 2 → Fin 128 → GF128) (ρ : Fin 2 → Fin 7 → GF128)
    (γ : Fin 2 → GF128) (hpre : openSteps s ρ γ <+: afterSparse (decodeRun e l) l)
    (hc : ∀ i, (∑ v, packedWeight ((decodeRun e l).claim i).toClaim (seven_le_mReg _) v * s i v) =
      ((decodeRun e l).claim i).toClaim.value) :
    PlaysT (openingAt (decodeRun e l)) (decodeRun e l).op (some (decodeRun e l).opened) := by
  obtain ⟨a, ha⟩ := opResult_some (seven_le_mReg e.cls.shape) (fun i => ((decodeRun e l).claim i).toClaim) s ρ γ hc
  have hpl := plays_opening (seven_le_mReg e.cls.shape) (fun i => ((decodeRun e l).claim i).toClaim) s ρ γ
  rw [ha] at hpl
  exact (opening_decodeRun_iff e l).2 ⟨_, a, hpre, hpl⟩

/-- **R's decoded opening does not play to an output** where the tape past both sparse phases begins with the
opening's steps and the first message fails its claim's check. -/
theorem not_opening_of_check (e : ExecHolo) (l : List Step) (s : Fin 2 → Fin 128 → GF128)
    (ρ : Fin 2 → Fin 7 → GF128) (γ : Fin 2 → GF128) (hpre : openSteps s ρ γ <+: afterSparse (decodeRun e l) l)
    (h0 : (∑ v, packedWeight ((decodeRun e l).claim 0).toClaim (seven_le_mReg _) v * s 0 v) ≠
      ((decodeRun e l).claim 0).toClaim.value) :
    ¬ PlaysT (openingAt (decodeRun e l)) (decodeRun e l).op (some (decodeRun e l).opened) := by
  rw [opening_decodeRun_iff]
  rintro ⟨p, a, hp, hpl⟩
  have hn := opResult_none (seven_le_mReg e.cls.shape) (fun i => ((decodeRun e l).claim i).toClaim) s ρ γ h0
  have hu := (playsT_prefix_unique
    (plays_opening (seven_le_mReg e.cls.shape) (fun i => ((decodeRun e l).claim i).toClaim) s ρ γ) hpl hpre hp).2
  rw [hn] at hu
  cases hu

/-! ## At the honest family -/

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}
  {ish : RecVStar.Inner} {ss : Array VStar.Stmt} {hss : RecVStar.statements ish = .ok ss}
  {bind : Fin (vstarCount ish) → VStar.Bindings} {Vs : Fin (vstarCount ish) → VStmt}
  {hob : VStarParses Flock.Tags.circuit ish bind Vs}
  {zr : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
    (Vs j).OuterR Reg'}
  {hCust : Flock.Premises.RecordCustody zr} {h : Typed zr}
  {own : Fin (vstarCount ish) → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
  {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
  {dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (holoInnerAt T CT st hp).Msg}
  {tr : Lean.Json} {record : String} {reg : Lean.Json}

local notation "𝓗" => C3Family.honest (xR := xR) ish hss bind hob zr hCust h own sel bc dec tr record reg

/-- **R's opening steps at every accepted outcome**: on the tape of V*'s layers' messages and the rounds' coins, the
part past both sparse phases begins with the opening's steps on the two registered claims, and both ring-switch
messages pass their claims' checks (O4). -/
def HOpenSteps (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ) : Prop :=
  F.AtAccepted dg N k Rw fun R ω t Lb =>
    let u := F.run R ω t Lb
    ∃ (s : Fin 2 → Fin 128 → GF128) (ρ : Fin 2 → Fin 7 → GF128) (γ : Fin 2 → GF128),
      openSteps s ρ γ <+: afterSparse u (schedSteps T CT 0 ((F.wmsg Lb).zip (t.map Prod.snd))) ∧
      ∀ i, (∑ v, packedWeight (u.claim i).toClaim (seven_le_mReg _) v * s i v) = (u.claim i).toClaim.value

/-- **R's opening plays at the honest family** where the tape holds the opening's steps and they pass. -/
theorem hOpening_of_steps {dg : ByteArray → ByteArray} {N k Rw : ℕ} (hst : HOpenSteps 𝓗 dg N k Rw) :
    HOpening 𝓗 dg N k Rw := by
  intro R ω t R₂ τ o hacc
  obtain ⟨s, ρ, γ, hpre, hc⟩ := hst R ω t R₂ τ o hacc
  exact ⟨seven_le_mReg _, opening_of_steps (holoAt st hp)
    (schedSteps T CT 0 (((𝓗).wmsg (FlockVBridge.Compiled.layers (𝓗).zs dg N k Rw R ω t R₂ τ)).zip
      (t.map Prod.snd))) s ρ γ hpre hc⟩

/-- **(n4) R's opening plays at the honest family**, from the public rows, the reps and their sparse phases. A stub. -/
theorem hOpening_honest {dg : ByteArray → ByteArray} {N k Rw : ℕ} (hrows : FlockC3.PubRows.HPubRows 𝓗)
    (hreps : FlockC3.HoloValues.HReps 𝓗 dg N k Rw) (hsp : FlockC3.Sparse.HSparse 𝓗 dg N k Rw) :
    HOpening 𝓗 dg N k Rw := by
  sorry

end FlockC3.Opening
