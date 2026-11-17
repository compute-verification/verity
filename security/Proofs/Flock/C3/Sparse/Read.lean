module

public import Proofs.Flock.C3.Sparse.Session
public import Proofs.Flock.C3.Compose.Honest
public import Proofs.Flock.VBridge.Sparse.Refine
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Flock.C3.Compose.Plays
import Proofs.Flock.Soundness.Discharge.PlaysCoin.Coins

/-! Each rep's sparse phase from the tape (`SparseRead`). -/

@[expose] public section

namespace FlockC3.Sparse

noncomputable section

section Unit

open FlockSoundness FlockSoundness.Game FlockSoundness.Model FlockSoundness.Refine FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.PlaysCoin (coinArr toT_exact)
open FlockVBridge.Sparse (SCoins nw nc verifierRow phaseMsgs coinTape SparseAccepts sparse_refines)

local notation "𝔽" => FlockLevel3.GF128

theorem exact_recv (M : Type) : Exact (recv M) [.msg M] := .send fun m => .ret m

theorem exact_bind_nil {α β : Type} {g : Game α} {f : α → Game β} {σ : List Kind} (h₁ : Exact g σ)
    (h₂ : ∀ a, Exact (f a) []) : Exact (g >>= f) σ := by
  have h := Exact.bind h₁ h₂
  rwa [List.append_nil] at h

theorem sampleFin_exact {F : Type} [Field F] [Fintype F] [DecidableEq F] :
    ∀ n : ℕ, Exact (sampleFin (F := F) n) (List.replicate n (.coin F))
  | 0 => by rw [sampleFin]; exact .ret _
  | n + 1 => by
      rw [sampleFin, List.replicate_succ]
      exact Exact.draw_bind fun _ => exact_bind_nil (sampleFin_exact n) fun _ => .ret _

theorem degRounds_exact {F : Type} [Field F] [Fintype F] [DecidableEq F] (k : ℕ) :
    ∀ (n : ℕ) (claim : F), Exact (degRounds k n claim) (degKinds F k n)
  | 0, _ => by rw [degRounds]; exact .ret _
  | n + 1, claim => by
      rw [degRounds, degKinds]
      refine Exact.recv_bind fun c => Exact.draw_bind fun r => exact_bind_nil (degRounds_exact k n _) fun x => ?_
      obtain ⟨rs, out⟩ := x
      exact .ret _

theorem sparsePhase_exact {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] (A : Arith F K)
    (sh : SparseShape) (pub : SparsePub F sh) (τ : Fin (2 ^ sh.sLog) → ℕ) (h : Fin 64 → F) :
    Exact (sparsePhase A sh pub τ h) (sparseKinds F sh) := by
  unfold sparsePhase sparseKinds
  refine Exact.bind (sampleFin_exact 6) fun ζc => ?_
  refine Exact.bind (degRounds_exact 1 6 _) fun x => ?_
  obtain ⟨ζr, c₂⟩ := x
  refine Exact.bind (exact_recv F) fun v₂ => ?_
  refine Exact.bind (degRounds_exact _ _ _) fun x => ?_
  obtain ⟨rk, c₃⟩ := x
  refine Exact.bind (degRounds_exact _ _ _) fun x => ?_
  obtain ⟨rs, c₄⟩ := x
  refine Exact.bind (exact_recv _) fun u => ?_
  refine Exact.bind (sampleFin_exact _) fun ζp => ?_
  dsimp only
  by_cases ht : sh.tLog = 0
  · rw [ite_eq_left ht, ite_eq_left ht]
    exact Exact.bind (.ret _) fun _ => exact_bind_nil (sampleFin_exact _) fun _ => .ret _
  · rw [ite_eq_right ht, ite_eq_right ht]
    exact Exact.bind (exact_recv _) fun _ => exact_bind_nil (sampleFin_exact _) fun _ => .ret _

def stepsOf (sh : SparseShape) (w cs : Array Flock.F128) : List Step :=
  weave (sparseKinds 𝔽 sh) (phaseMsgs sh w) ((coinTape cs 0 (nc sh)).map (coinArr fOf))

theorem playsT_stepsOf {sh : SparseShape} {pub : SparsePub 𝔽 sh} {τ : Fin (2 ^ sh.sLog) → ℕ} {h : Fin 64 → 𝔽}
    {w cs : Array Flock.F128} {a : Option (RegClaim 𝔽 sh)}
    (hp : Plays fOf (sparsePhase execArith sh pub τ h) (phaseMsgs sh w) (coinTape cs 0 (nc sh)) a) :
    PlaysT (sparsePhase execArith sh pub τ h) (stepsOf sh w cs) a := by
  obtain ⟨ts, ht, hl, hr⟩ := toT_exact fOf hp
  have e := weave_self ts
  rw [Exact.kinds ht (sparsePhase_exact execArith sh pub τ h), hl, hr] at e
  subst e
  exact ht

def SparseUnit (sh : SparseShape) (pub : SparsePub 𝔽 sh) (τ' : Fin (2 ^ sh.sLog) → ℕ) (h : Fin 64 → 𝔽)
    (w : Array Flock.F128) (τ : Array ℕ) (c : SCoins) : Prop :=
  w.size = nw sh ∧ nc sh ≤ c.cs.size ∧ FlockVBridge.Sparse.pubOf sh c = pub ∧
    (∀ σ : Fin (2 ^ sh.sLog), τ[σ.val]! = τ' σ) ∧ (∀ s : Fin 64, fOf w[s.val]! = h s) ∧
    ∃ o, FlockVBridge.Sparse.out sh w (verifierRow sh c) τ = .ok o ∧ SparseAccepts o

theorem playsT_of_unit {sh : SparseShape} {pub : SparsePub 𝔽 sh} {τ' : Fin (2 ^ sh.sLog) → ℕ} {h : Fin 64 → 𝔽}
    {w : Array Flock.F128} {τ : Array ℕ} {c : SCoins} (hu : SparseUnit sh pub τ' h w τ c) :
    ∃ x, PlaysT (sparsePhase execArith sh pub τ' h) (stepsOf sh w c.cs) (some x) := by
  obtain ⟨hw, hc, rfl, hτ, hh, o, ho, hacc⟩ := hu
  obtain rfl : (fun σ : Fin (2 ^ sh.sLog) => τ[σ.val]!) = τ' := funext hτ
  obtain rfl : (fun s : Fin 64 => fOf w[s.val]!) = h := funext hh
  obtain ⟨o', ho', hpl, hiff⟩ := sparse_refines sh w c τ hw hc
  have e : Except.ok o = (Except.ok o' : Except String _) := ho.symm.trans ho'
  cases e
  rw [(hiff.1 hacc).2] at hpl
  exact ⟨_, playsT_stepsOf hpl⟩

end Unit

section Family

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockC3.Compose
open FlockVBridge.Sparse (SCoins)

abbrev sparseAt (e : ExecHolo) (u : C3Run e) (i : Fin 2) : Game (Option (RegClaim GF128 e.cls.shape)) :=
  sparsePhase execArith e.cls.shape
    (holoPub (e.cls.pubStatement e.regs) e.cls.shape e.cls.units (u.rep i).1 (u.rep i).2) (fun σ => (e.τ σ).val)
    (u.rep i).2.h

/-- The tape `l` past what `decodeRun` reads before the sparse phases: the cap's and the points' steps, then the reps'
(`repsOf`) at the run `u`'s. -/
def afterReps {e : ExecHolo} (l : List Step) (u : C3Run e) : List Step :=
  (l.drop 2).drop (2 * (u.ts 0).length + (u.ts 1).length)

def SparseReadAt (e : ExecHolo) (l : List Step) (u : C3Run e) : Prop :=
  ∃ (w : Fin 2 → Array Flock.F128) (τ : Fin 2 → Array ℕ) (c : Fin 2 → SCoins),
    (∀ i, SparseUnit e.cls.shape
      (holoPub (e.cls.pubStatement e.regs) e.cls.shape e.cls.units (u.rep i).1 (u.rep i).2)
      (fun σ => (e.τ σ).val) (u.rep i).2.h (w i) (τ i) (c i)) ∧
    stepsOf e.cls.shape (w 0) (c 0).cs ++ stepsOf e.cls.shape (w 1) (c 1).cs <+: afterReps l u

def SparseStages (e : ExecHolo) (l : List Step) (u : C3Run e) : Prop :=
  u.sp 0 = (stageOf (sparseAt e u 0) (afterReps l u) none).1 ∧
    (∀ x, (stageOf (sparseAt e u 0) (afterReps l u) none).2 = some x → u.claim 0 = x) ∧
    u.sp 1 = (stageOf (sparseAt e u 1) ((afterReps l u).drop (u.sp 0).length) none).1 ∧
    ∀ x, (stageOf (sparseAt e u 1) ((afterReps l u).drop (u.sp 0).length) none).2 = some x → u.claim 1 = x

theorem sparseStages_decodeRun (e : ExecHolo) (l : List Step) : SparseStages e l (decodeRun e l) := by
  refine And.intro rfl (And.intro (fun x hx => ?_) (And.intro rfl fun x hx => ?_))
  · change (stageOf (sparseAt e (decodeRun e l) 0) (afterReps l (decodeRun e l)) none).2.getD
      ((c3Default e).claim 0) = x
    rw [hx, Option.getD_some]
  · change (stageOf (sparseAt e (decodeRun e l) 1)
      ((afterReps l (decodeRun e l)).drop ((decodeRun e l).sp 0).length) none).2.getD ((c3Default e).claim 1) = x
    rw [hx, Option.getD_some]

theorem sparse_of_readAt {e : ExecHolo} {l : List Step} {u : C3Run e} (hs : SparseStages e l u)
    (hr : SparseReadAt e l u) : ∀ i : Fin 2, PlaysT (sparseAt e u i) (u.sp i) (some (u.claim i)) := by
  obtain ⟨hs₀, hc₀, hs₁, hc₁⟩ := hs
  obtain ⟨w, τ, c, hu, hpre⟩ := hr
  obtain ⟨x₀, h₀⟩ := playsT_of_unit (hu 0)
  obtain ⟨x₁, h₁⟩ := playsT_of_unit (hu 1)
  have e₀ : stageOf (sparseAt e u 0) (afterReps l u) none = (stepsOf e.cls.shape (w 0) (c 0).cs, some x₀) :=
    stageOf_of_playsT none h₀ ((List.prefix_append _ _).trans hpre)
  rw [e₀] at hs₀ hc₀
  dsimp only at hs₀ hc₀
  have hpre₁ : stepsOf e.cls.shape (w 1) (c 1).cs <+: (afterReps l u).drop (u.sp 0).length := by
    obtain ⟨t, ht⟩ := hpre
    refine ⟨t, ?_⟩
    rw [hs₀, ← ht, List.append_assoc, List.drop_left]
  have e₁ : stageOf (sparseAt e u 1) ((afterReps l u).drop (u.sp 0).length) none =
      (stepsOf e.cls.shape (w 1) (c 1).cs, some x₁) := stageOf_of_playsT none h₁ hpre₁
  rw [e₁] at hs₁ hc₁
  dsimp only at hs₁ hc₁
  have p₀ : PlaysT (sparseAt e u 0) (u.sp 0) (some (u.claim 0)) := by rw [hs₀, hc₀ x₀ rfl]; exact h₀
  have p₁ : PlaysT (sparseAt e u 1) (u.sp 1) (some (u.claim 1)) := by rw [hs₁, hc₁ x₁ rfl]; exact h₁
  exact Fin.forall_fin_two.2 ⟨p₀, p₁⟩

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg' Reg₂ : Type}

/-- It reads the tape, not the decoder of V*'s messages. -/
def SparseRead (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ) : Prop :=
  F.AtAccepted dg N k Rw fun R ω t Lb =>
    SparseReadAt (holoAt st hp) (schedSteps T CT 0 ((F.wmsg Lb).zip (t.map Prod.snd))) (F.run R ω t Lb)

def HStages (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ) : Prop :=
  F.AtAccepted dg N k Rw fun R ω t Lb =>
    SparseStages (holoAt st hp) (schedSteps T CT 0 ((F.wmsg Lb).zip (t.map Prod.snd))) (F.run R ω t Lb)

theorem hSparse_of_read {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂} {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    (hst : HStages F dg N k Rw) (hread : SparseRead F dg N k Rw) : HSparse F dg N k Rw := by
  intro R ω t R₂ τ o hacc
  exact sparse_of_readAt (hst R ω t R₂ τ o hacc) (hread R ω t R₂ τ o hacc)

variable {ish : RecVStar.Inner} {ss : Array VStar.Stmt} {hss : RecVStar.statements ish = .ok ss}
  {bind : Fin (vstarCount ish) → VStar.Bindings} {Vs : Fin (vstarCount ish) → VStmt}
  {hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs}
  {zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
    (Vs j).Outer Reg'}
  {own : Fin (vstarCount ish) → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
  {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
  {dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (holoInnerAt T CT st hp).Msg}
  {wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String} {reg : Lean.Json}

local notation "𝓗" => C3Family.honest (xR := xR) ish hss bind hob zs own sel bc dec wr reg

theorem hStages_honest {dg : ByteArray → ByteArray} {N k Rw : ℕ} : HStages 𝓗 dg N k Rw := by
  intro _ _ _ _ _ _ _
  exact sparseStages_decodeRun (holoAt st hp) _

theorem hSparse_honest_of_read {dg : ByteArray → ByteArray} {N k Rw : ℕ} (hread : SparseRead 𝓗 dg N k Rw) :
    HSparse 𝓗 dg N k Rw :=
  hSparse_of_read hStages_honest hread

end Family

end

end FlockC3.Sparse
