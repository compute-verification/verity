import Proofs.Flock.C3.Sparse.Session
import Proofs.Flock.C3.Compose.Honest
import Proofs.Flock.C3.Compose.Plays
import Proofs.Flock.VBridge.Sparse.Refine
import Proofs.Flock.Soundness.Discharge.PlaysCoin.Coins
import Proofs.Flock.C3.Sparse.Kinds

/-!
# n3's read fact: each rep's sparse phase from the tape

`SparseRead`: at every accepted outcome, the inner transcript's steps, past the reps' steps the run reads, begin with
rep 0's RecSparse unit and then rep 1's: each unit's messages woven with its coins along the sparse phase's shape
(`stepsOf`), for a witness the unit's verifier row accepts on that rep's public values, the slots' types and the rep's
`h` (`SparseUnit`). It reads the tape, not the decoder of V*'s messages.

`hSparse_of_read`: `HSparse` from `SparseRead`, at any family whose run decodes its sparse stages from the tape as
`decodeRun` does (`HStages`, which the honest family has: `hStages_honest`). An accepting unit's verifier row is the
phase's run to the registered claim (`sparse_refines`), which is a play on one tape (`toT_exact`) in the order the
phase's shape fixes (`sparsePhase_exact`, so the tape is `stepsOf`); and a game plays on at most one prefix of a tape
(`stageOf_of_playsT`), so the stage the run decodes is that play.
-/

namespace FlockC3.Sparse

noncomputable section

section Unit

open FlockSoundness FlockSoundness.Game FlockSoundness.Model FlockSoundness.Refine FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.PlaysCoin (coinArr toT_exact)
open FlockVBridge.Sparse (SCoins nw nc verifierRow phaseMsgs coinTape SparseAccepts sparse_refines)

local notation "𝔽" => FlockLevel3.GF128

/-! ## The sparse phase's shape (`sparseKinds`, `Kinds.lean`) -/

theorem exact_recv (M : Type) : Exact (recv M) [.msg M] := .send fun m => .ret m

theorem exact_bind_nil {α β : Type} {g : Game α} {f : α → Game β} {σ : List Kind} (h₁ : Exact g σ)
    (h₂ : ∀ a, Exact (f a) []) : Exact (g >>= f) σ := by
  have h := Exact.bind h₁ h₂
  rwa [List.append_nil] at h

theorem drawFin_exact {F : Type} [Field F] [Fintype F] [DecidableEq F] :
    ∀ n : ℕ, Exact (drawFin (F := F) n) (List.replicate n (.coin F))
  | 0 => by rw [drawFin]; exact .ret _
  | n + 1 => by
      rw [drawFin, List.replicate_succ]
      exact Exact.draw_bind fun _ => exact_bind_nil (drawFin_exact n) fun _ => .ret _

theorem degRounds_exact {F : Type} [Field F] [Fintype F] [DecidableEq F] (k : ℕ) :
    ∀ (n : ℕ) (claim : F), Exact (degRounds k n claim) (degKinds F k n)
  | 0, _ => by rw [degRounds]; exact .ret _
  | n + 1, claim => by
      rw [degRounds, degKinds]
      refine Exact.recv_bind fun c => Exact.draw_bind fun r => exact_bind_nil (degRounds_exact k n _) fun x => ?_
      obtain ⟨rs, out⟩ := x
      exact .ret _

/-- **The sparse phase takes its shape** on every run. -/
theorem sparsePhase_exact {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] (A : Arith F K)
    (sh : SparseShape) (pub : SparsePub F sh) (τ : Fin (2 ^ sh.sLog) → ℕ) (h : Fin 64 → F) :
    Exact (sparsePhase A sh pub τ h) (sparseKinds F sh) := by
  unfold sparsePhase sparseKinds
  refine Exact.bind (drawFin_exact 6) fun ζc => ?_
  refine Exact.bind (degRounds_exact 1 6 _) fun x => ?_
  obtain ⟨ζr, c₂⟩ := x
  refine Exact.bind (exact_recv F) fun v₂ => ?_
  refine Exact.bind (degRounds_exact _ _ _) fun x => ?_
  obtain ⟨rk, c₃⟩ := x
  refine Exact.bind (degRounds_exact _ _ _) fun x => ?_
  obtain ⟨rs, c₄⟩ := x
  refine Exact.bind (exact_recv _) fun u => ?_
  refine Exact.bind (drawFin_exact _) fun ζp => ?_
  dsimp only
  by_cases ht : sh.tLog = 0
  · rw [ite_eq_left ht, ite_eq_left ht]
    exact Exact.bind (.ret _) fun _ => exact_bind_nil (drawFin_exact _) fun _ => .ret _
  · rw [ite_eq_right ht, ite_eq_right ht]
    exact Exact.bind (exact_recv _) fun _ => exact_bind_nil (drawFin_exact _) fun _ => .ret _

/-! ## A unit's steps -/

/-- **A RecSparse unit's steps on the tape**: the messages of its witness `w` (`phaseMsgs`) and its phase's coins from
`cs`, one word each (`coinTape`), woven along the sparse phase's shape. -/
def stepsOf (sh : SparseShape) (w cs : Array Flock.F128) : List Step :=
  weave (sparseKinds 𝔽 sh) (phaseMsgs sh w) ((coinTape cs 0 (nc sh)).map (coinArr fOf))

/-- **A run of the sparse phase on `w`'s messages and the coins `cs` is a play on `stepsOf`.** -/
theorem playsT_stepsOf {sh : SparseShape} {pub : SparsePub 𝔽 sh} {τ : Fin (2 ^ sh.sLog) → ℕ} {h : Fin 64 → 𝔽}
    {w cs : Array Flock.F128} {a : Option (RegClaim 𝔽 sh)}
    (hp : Plays fOf (sparsePhase execArith sh pub τ h) (phaseMsgs sh w) (coinTape cs 0 (nc sh)) a) :
    PlaysT (sparsePhase execArith sh pub τ h) (stepsOf sh w cs) a := by
  obtain ⟨ts, ht, hl, hr⟩ := toT_exact fOf hp
  have e := weave_self ts
  rw [Exact.kinds ht (sparsePhase_exact execArith sh pub τ h), hl, hr] at e
  subst e
  exact ht

/-- **A RecSparse unit at a rep's values**: a witness `w` of the message ports' size whose first ports are the rep's
`h`, the slots' types `τ` (the rep's `τ'`), and coins `c` with the phase's coins and the rep's public values `pub`, at
which the unit's verifier row accepts `w`. -/
def SparseUnit (sh : SparseShape) (pub : SparsePub 𝔽 sh) (τ' : Fin (2 ^ sh.sLog) → ℕ) (h : Fin 64 → 𝔽)
    (w : Array Flock.F128) (τ : Array ℕ) (c : SCoins) : Prop :=
  w.size = nw sh ∧ nc sh ≤ c.cs.size ∧ FlockVBridge.Sparse.pubOf sh c = pub ∧
    (∀ σ : Fin (2 ^ sh.sLog), τ[σ.val]! = τ' σ) ∧ (∀ s : Fin 64, fOf w[s.val]! = h s) ∧
    ∃ o, FlockVBridge.Sparse.out sh w (verifierRow sh c) τ = .ok o ∧ SparseAccepts o

/-- **An accepting unit plays its sparse phase on its steps to a registered claim** (`sparse_refines`). -/
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

/-! ## The read fact -/

/-- Rep `i`'s sparse phase at the run `u`, on its public values (`holoPub` of its zerocheck and lincheck outputs), the
slots' types and its `h`, as `HSparse` and `decodeRun` take it. -/
abbrev sparseAt (e : ExecHolo) (u : C3Run e) (i : Fin 2) : Game (Option (RegClaim GF128 e.cls.shape)) :=
  sparsePhase execArith e.cls.shape
    (holoPub (e.cls.pubStatement e.regs) e.cls.shape e.cls.units (u.rep i).1 (u.rep i).2) (fun σ => (e.τ σ).val)
    (u.rep i).2.h

/-- The tape `l` past what `decodeRun` reads before the sparse phases: the cap's and the points' steps, then the reps'
(`repsOf`) at the run `u`'s. -/
def afterReps {e : ExecHolo} (l : List Step) (u : C3Run e) : List Step :=
  (l.drop 2).drop (2 * (u.ts 0).length + (u.ts 1).length)

/-- **The read fact at a tape** `l` and a run `u`: past the reps' steps, `l` begins with rep 0's RecSparse unit and
then rep 1's (`stepsOf`), each for a witness its verifier row accepts at that rep's values (`SparseUnit`). -/
def SparseReadAt (e : ExecHolo) (l : List Step) (u : C3Run e) : Prop :=
  ∃ (w : Fin 2 → Array Flock.F128) (τ : Fin 2 → Array ℕ) (c : Fin 2 → SCoins),
    (∀ i, SparseUnit e.cls.shape
      (holoPub (e.cls.pubStatement e.regs) e.cls.shape e.cls.units (u.rep i).1 (u.rep i).2)
      (fun σ => (e.τ σ).val) (u.rep i).2.h (w i) (τ i) (c i)) ∧
    stepsOf e.cls.shape (w 0) (c 0).cs ++ stepsOf e.cls.shape (w 1) (c 1).cs <+: afterReps l u

/-- **The run's sparse stages are the tape's**, as `decodeRun` decodes them: rep 0's steps are the prefix its phase
plays on past the reps' steps, rep 1's the prefix its phase plays on past rep 0's, and each rep's claim is the output
its phase ends with there, where it ends with one. -/
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

/-- **Each rep's sparse phase plays to its claim** on its decoded steps, at a tape whose units the read fact gives. -/
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
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

/-- **(n3's read fact) Each rep's RecSparse unit is on the inner transcript**: at every accepted outcome, the inner
transcript's steps (V*'s layers' messages zipped with the rounds' coins) hold, past the reps' steps the run reads, rep
0's and then rep 1's RecSparse unit, each for a witness its verifier row accepts at that rep's values. -/
def SparseRead (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ) : Prop :=
  F.AtAccepted dg N k Rw fun R ω t Lb =>
    SparseReadAt (holoAt st hp) (schedSteps T CT 0 ((F.wmsg Lb).zip (t.map Prod.snd))) (F.run R ω t Lb)

/-- **The family's run decodes its sparse stages from the inner transcript's steps** as `decodeRun` does, at every
accepted outcome. -/
def HStages (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ) : Prop :=
  F.AtAccepted dg N k Rw fun R ω t Lb =>
    SparseStages (holoAt st hp) (schedSteps T CT 0 ((F.wmsg Lb).zip (t.map Prod.snd))) (F.run R ω t Lb)

/-- **(n3) Each rep's sparse phase plays to its registered claim, from the read fact**, at any family whose run
decodes its sparse stages as `decodeRun` does. -/
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

/-- The honest family's run is `decodeRun` of the inner transcript's steps, at any decoder `dec`. -/
theorem hStages_honest {dg : ByteArray → ByteArray} {N k Rw : ℕ} : HStages 𝓗 dg N k Rw := by
  intro _ _ _ _ _ _ _
  exact sparseStages_decodeRun (holoAt st hp) _

/-- **(n3) at the honest family, from the read fact.** -/
theorem hSparse_honest_of_read {dg : ByteArray → ByteArray} {N k Rw : ℕ} (hread : SparseRead 𝓗 dg N k Rw) :
    HSparse 𝓗 dg N k Rw :=
  hSparse_of_read hStages_honest hread

end Family

end

end FlockC3.Sparse
