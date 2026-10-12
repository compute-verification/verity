import Proofs.Flock.C3.Compose.Tied

/-!
# n9: `MsgsReadC3` at the normalized tied source

The last round's message is the openings step, `os`, which V* holds only in part: the openings at the reps (strata 0
and 1) and at R (stratum 2). The salts (candidate 3's leaf is plain) and the strata from 3 on are on no read wire, and
no stage reads them. `normOs` sets them to defaults, so the source is `tiedSrc (normSrc stp)`, and every tie, the
stages' plays and `SrcOK`'s other clauses carry over (`repTie_normOs`, `levelTie_normOs`, `openingReadAt_normOs`,
`srcOK_tied_norm`).

`msgsReadC3_of_parts` is `MsgsReadC3` from a read fact per part of the steps: at a committed round (`i < r`, where the
mask is the commit strings', `maskOf_skLeaf`) the round closes in the reps' part, the sparse phases', R's opening's or
R's Ligerito's (`RepsMsgsRead`, `SparseMsgsRead`, `OpenMsgsRead`, `LigMsgsRead`); the last round is the openings step
alone where R's Ligerito ends in a coin (`LigWhole`), and the openings are read from the layers stratum by stratum
(`OpensReadAt`); past it both runs have no messages. `msgsReadC3_tied` is it at the normalized tied source.
-/

namespace FlockC3.Compose

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockVBridge.Family (skOf maskOf)
open FlockVBridge.InnerAccepts (firstSteps)
open FlockC3.Reps (cutOf repPart RepsMsgsRead RepTie)

/-! ## The normalized run and the parts of its steps -/

section run

variable {e : ExecHolo}

/-- `u` with the openings no stage reads set to defaults: the strata from 3 on, and every salt. -/
def normOs (u : C3Run e) : C3Run e :=
  { u with os := (fun s => if s < 3 then u.os.1 s else fun _ _ => ([], []), fun _ _ _ => []) }

/-- **A run's openings are normal**: every salt and every stratum from 3 on is the default. -/
def OsNormal (u : C3Run e) : Prop :=
  u.os.2 = (fun _ _ _ => []) ∧ ∀ s, 3 ≤ s → u.os.1 s = fun _ _ => ([], [])

theorem normOs_os_of_lt {u : C3Run e} {s : ℕ} (hs : s < 3) : (normOs u).os.1 s = u.os.1 s := ite_eq_left hs

theorem osNormal_normOs (u : C3Run e) : OsNormal (normOs u) := ⟨rfl, fun _ hs => ite_eq_right (by omega)⟩

theorem os_eq_of_normal {u u' : C3Run e} (hn : OsNormal u) (hn' : OsNormal u') (h : ∀ s < 3, u.os.1 s = u'.os.1 s) :
    u.os = u'.os :=
  Prod.ext (funext fun s => (Nat.lt_or_ge s 3).elim (h s) fun hs => (hn.2 s hs).trans (hn'.2 s hs).symm)
    (hn.1.trans hn'.1.symm)

/-- **The sparse phases' part of a run's steps**: the reps' part, then both sparse phases. -/
def sparsePart (u : C3Run e) : List Step := (runSteps u).take 2 ++ (firstSteps (u.ts 0) ++ u.ts 1 ++ u.sp 0 ++ u.sp 1)

/-- **R's opening's part**: the sparse phases' part, then R's opening. -/
def openPart (u : C3Run e) : List Step :=
  (runSteps u).take 2 ++ (firstSteps (u.ts 0) ++ u.ts 1 ++ u.sp 0 ++ u.sp 1 ++ u.op)

/-- **R's Ligerito's part**: every step but the openings'. -/
def ligPart (u : C3Run e) : List Step := (runSteps u).take 2 ++ FlockC3.PubRows.stagesOf u

theorem runSteps_lig (u : C3Run e) :
    runSteps u = ligPart u ++ [.inl ⟨Opens GF128 Dig × Salts (List UInt8), u.os⟩] := by
  have hd : (runSteps u).drop 2 =
      FlockC3.PubRows.stagesOf u ++ [.inl ⟨Opens GF128 Dig × Salts (List UInt8), u.os⟩] := by
    rw [runSteps, FlockC3.PubRows.holoSteps_eq]
    simp only [List.drop_succ_cons, List.drop_zero]
  rw [ligPart, List.append_assoc, ← hd, List.take_append_drop]

theorem coins_runSteps (u : C3Run e) :
    ((runSteps u).filterMap Sum.getRight?).length = ((ligPart u).filterMap Sum.getRight?).length := by
  rw [runSteps_lig u, List.filterMap_append, List.length_append]
  rfl

theorem rep_sparse (u : C3Run e) : repPart u <+: sparsePart u :=
  ⟨u.sp 0 ++ u.sp 1, by simp only [repPart, sparsePart, List.append_assoc]⟩

theorem sparse_open (u : C3Run e) : sparsePart u <+: openPart u :=
  ⟨u.op, by simp only [sparsePart, openPart, List.append_assoc]⟩

theorem open_lig (u : C3Run e) : openPart u <+: ligPart u :=
  ⟨u.lg, by simp only [openPart, ligPart, FlockC3.PubRows.stagesOf, List.append_assoc]⟩

theorem lig_run (u : C3Run e) : ligPart u <+: runSteps u := ⟨_, (runSteps_lig u).symm⟩

/-! ## Messages cut at coins -/

theorem cutOf_prefix {a l : List Step} (h : a <+: l) {i : ℕ} (hi : i < (a.filterMap Sum.getRight?).length) :
    (cutOf l)[i]? = (cutOf a)[i]? := by
  obtain ⟨b, rfl⟩ := h
  exact FlockC3.Reps.cutOf_append b a i hi

/-- **After a last coin, one message is the last group.** -/
theorem cutOf_last (c m : Game.Msg) : ∀ a : List Step,
    (cutOf (a ++ [.inr c] ++ [.inl m]))[(a.filterMap Sum.getRight?).length + 1]? = some [m]
  | [] => rfl
  | .inl m' :: a => by
      have ih := cutOf_last c m a
      change (PlaysCoin.consHead m' (cutOf (a ++ [.inr c] ++ [.inl m])))[(a.filterMap Sum.getRight?).length + 1]? =
        some [m]
      obtain ⟨g, gs, hg⟩ := List.exists_cons_of_ne_nil (FlockC3.Reps.cutOf_ne_nil (a ++ [.inr c] ++ [.inl m]))
      rw [hg] at ih ⊢
      simpa only [PlaysCoin.consHead, List.getElem?_cons_succ] using ih
  | .inr c' :: a => by
      have ih := cutOf_last c m a
      change ([] :: cutOf (a ++ [.inr c] ++ [.inl m]))[(a.filterMap Sum.getRight?).length + 1 + 1]? = some [m]
      rw [List.getElem?_cons_succ]
      exact ih

/-- **The last group is the openings step** where R's Ligerito ends in a coin. -/
theorem cutOf_runSteps_last {u : C3Run e} {l : List Step} {c : Game.Msg} {i : ℕ} (hl : u.lg = l ++ [.inr c])
    (hi : ((ligPart u).filterMap Sum.getRight?).length = i) :
    (cutOf (runSteps u))[i]? = some [⟨Opens GF128 Dig × Salts (List UInt8), u.os⟩] := by
  subst hi
  have h : ligPart u =
      (runSteps u).take 2 ++ (firstSteps (u.ts 0) ++ u.ts 1 ++ u.sp 0 ++ u.sp 1 ++ u.op ++ l) ++ [.inr c] := by
    rw [ligPart, FlockC3.PubRows.stagesOf, hl]
    simp only [List.append_assoc]
  rw [runSteps_lig u, h, FlockVBridge.Family.rights_append, List.length_append]
  exact cutOf_last c _ _

theorem cutOf_runSteps_none {u : C3Run e} {i : ℕ} (hi : ((ligPart u).filterMap Sum.getRight?).length < i) :
    (cutOf (runSteps u))[i]? = none := by
  have hl : (cutOf (runSteps u)).length = ((ligPart u).filterMap Sum.getRight?).length + 1 :=
    (FlockVBridge.Family.length_cut (runSteps u)).trans (congrArg (· + 1) (coins_runSteps u))
  exact List.getElem?_eq_none_iff.2 (by omega)

/-- **A round before both runs' last coins**, from the part it closes in, part by part: each part's fact is asked
only where the round closes in that part in both runs and not in the one before it in both. -/
theorem cutOf_runSteps_eq {u u' : C3Run e} {i : ℕ} (hi : i < ((ligPart u).filterMap Sum.getRight?).length)
    (hi' : i < ((ligPart u').filterMap Sum.getRight?).length)
    (hrp : i < ((repPart u).filterMap Sum.getRight?).length → i < ((repPart u').filterMap Sum.getRight?).length →
      (cutOf (repPart u))[i]? = (cutOf (repPart u'))[i]?)
    (hsp : ¬ (i < ((repPart u).filterMap Sum.getRight?).length ∧ i < ((repPart u').filterMap Sum.getRight?).length) →
      i < ((sparsePart u).filterMap Sum.getRight?).length → i < ((sparsePart u').filterMap Sum.getRight?).length →
      (cutOf (sparsePart u))[i]? = (cutOf (sparsePart u'))[i]?)
    (hop : ¬ (i < ((sparsePart u).filterMap Sum.getRight?).length ∧
        i < ((sparsePart u').filterMap Sum.getRight?).length) →
      i < ((openPart u).filterMap Sum.getRight?).length → i < ((openPart u').filterMap Sum.getRight?).length →
      (cutOf (openPart u))[i]? = (cutOf (openPart u'))[i]?)
    (hlg : ¬ (i < ((openPart u).filterMap Sum.getRight?).length ∧ i < ((openPart u').filterMap Sum.getRight?).length) →
      (cutOf (ligPart u))[i]? = (cutOf (ligPart u'))[i]?) :
    (cutOf (runSteps u))[i]? = (cutOf (runSteps u'))[i]? := by
  have pl := lig_run u
  have pl' := lig_run u'
  have po := (open_lig u).trans pl
  have po' := (open_lig u').trans pl'
  have ps := (sparse_open u).trans po
  have ps' := (sparse_open u').trans po'
  by_cases hA : i < ((repPart u).filterMap Sum.getRight?).length ∧ i < ((repPart u').filterMap Sum.getRight?).length
  · rw [cutOf_prefix ((rep_sparse u).trans ps) hA.1, cutOf_prefix ((rep_sparse u').trans ps') hA.2]
    exact hrp hA.1 hA.2
  by_cases hS : i < ((sparsePart u).filterMap Sum.getRight?).length ∧
      i < ((sparsePart u').filterMap Sum.getRight?).length
  · rw [cutOf_prefix ps hS.1, cutOf_prefix ps' hS.2]
    exact hsp hA hS.1 hS.2
  by_cases hO : i < ((openPart u).filterMap Sum.getRight?).length ∧ i < ((openPart u').filterMap Sum.getRight?).length
  · rw [cutOf_prefix po hO.1, cutOf_prefix po' hO.2]
    exact hop hS hO.1 hO.2
  · rw [cutOf_prefix pl hi, cutOf_prefix pl' hi']
    exact hlg hO

/-! ## The ties under `normOs`: they read the openings at strata 0–2 only -/

theorem repTie_normOs {u : C3Run e} {i : Fin 2} (h : RepTie e u i) : RepTie e (normOs u) i := by
  obtain ⟨U⟩ := h
  exact ⟨{ U with opens := fun ℓ j =>
    (congrFun (congrFun (normOs_os_of_lt (u := u) (Nat.lt_trans i.isLt (by decide))) ℓ) j).trans (U.opens ℓ j) }⟩

theorem levelTie_normOs {cap₀ : Cap Dig} {u : C3Run e} (h : FlockC3.RegLevels.LevelTie cap₀ e u) :
    FlockC3.RegLevels.LevelTie cap₀ e (normOs u) := by
  obtain ⟨msId, m, sch, Tw, sws, nc, cap, p, rt, rt', h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈, h₉, h₁₀⟩ := h
  exact ⟨msId, m, sch, Tw, sws, nc, cap, p, rt, rt', h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈, h₉,
    fun ℓ j => (congrFun (congrFun (normOs_os_of_lt (u := u) (by decide : 2 < 3)) ℓ) j).trans (h₁₀ ℓ j)⟩

/-- `SparseTie` and `OpenTie` read no openings, so they hold at `normOs u` as at `u`, definitionally. R's opening's
read fact needs the opening's steps to be `u.op`, which `OpenTie` gives (`opening_plays_of`). -/
theorem openingReadAt_normOs {st : Flock.Stmt} {hp : HoloParams} {u : C3Run (holoAt st hp)} {sh : Flock.Shape}
    {w : Array F128} {ops : FlockVBridge.AccChain.Openings} (ht : FlockC3.Opening.OpenTie sh w ops u)
    (h : OpeningReadAt (runSteps u) u) : OpeningReadAt (runSteps (normOs u)) (normOs u) := by
  obtain ⟨sh', M, a, cs, k₀, k₁, res, ρ, γ, hs, hr, hz, hrel, hpre⟩ := h
  obtain ⟨ρ', γ', hu⟩ := ht
  rw [FlockC3.Opening.afterSparse_runSteps] at hpre
  have hlen : (FlockC3.Opening.openSteps (fun i => FlockSoundness.Refine.vecOf 128 M.rs[i.val]!) ρ γ).length =
      u.op.length := by
    rw [hu]
    rfl
  have he : FlockC3.Opening.openSteps (fun i => FlockSoundness.Refine.vecOf 128 M.rs[i.val]!) ρ γ = u.op :=
    (List.prefix_of_prefix_length_le hpre (List.prefix_append _ _) hlen.le).eq_of_length hlen
  refine ⟨sh', M, a, cs, k₀, k₁, res, ρ, γ, hs, hr, hz, hrel, ?_⟩
  rw [FlockC3.Opening.afterSparse_runSteps, he]
  exact List.prefix_append _ _

/-! ## The tied run under `normOs`: the same steps

Stated field by field and rewritten, not unfolded: checking `tie (normOs v)` against `tie v` by unfolding compares the
stages' plays. -/

theorem tie_normOs_steps (v : C3Run e) :
    (tie (normOs v)).cap₀ = (tie v).cap₀ ∧ (tie (normOs v)).pts = (tie v).pts ∧ (tie (normOs v)).ts = (tie v).ts ∧
      (tie (normOs v)).sp = (tie v).sp ∧ (tie (normOs v)).op = (tie v).op ∧ (tie (normOs v)).lg = (tie v).lg := by
  simp only [tie, tieLig, tieOpening, tieSparse, tieReps, normOs, and_self]

theorem tie_os (v : C3Run e) : (tie v).os = v.os := by
  simp only [tie, tieLig, tieOpening, tieSparse, tieReps]

theorem take_two_runSteps (u : C3Run e) :
    (runSteps u).take 2 = [.inl ⟨Cap Dig, u.cap₀⟩, .inr ⟨Fin 2 → Fin e.cls.mPts → GF128, u.pts⟩] := by
  rw [runSteps, FlockC3.PubRows.holoSteps_eq]
  rfl

theorem take_two_norm (v : C3Run e) : (runSteps (tie (normOs v))).take 2 = (runSteps (tie v)).take 2 := by
  obtain ⟨h₁, h₂, -⟩ := tie_normOs_steps v
  rw [take_two_runSteps, take_two_runSteps, h₁, h₂]

theorem repPart_norm (v : C3Run e) : repPart (tie (normOs v)) = repPart (tie v) := by
  obtain ⟨-, -, h₃, -⟩ := tie_normOs_steps v
  rw [repPart, repPart, take_two_norm, h₃]

theorem sparsePart_norm (v : C3Run e) : sparsePart (tie (normOs v)) = sparsePart (tie v) := by
  obtain ⟨-, -, h₃, h₄, -⟩ := tie_normOs_steps v
  rw [sparsePart, sparsePart, take_two_norm, h₃, h₄]

theorem openPart_norm (v : C3Run e) : openPart (tie (normOs v)) = openPart (tie v) := by
  obtain ⟨-, -, h₃, h₄, h₅, -⟩ := tie_normOs_steps v
  rw [openPart, openPart, take_two_norm, h₃, h₄, h₅]

theorem ligPart_norm (v : C3Run e) : ligPart (tie (normOs v)) = ligPart (tie v) := by
  obtain ⟨-, -, h₃, h₄, h₅, h₆⟩ := tie_normOs_steps v
  rw [ligPart, ligPart, FlockC3.PubRows.stagesOf, FlockC3.PubRows.stagesOf, take_two_norm, h₃, h₄, h₅, h₆]

theorem osNormal_tie_norm (v : C3Run e) : OsNormal (tie (normOs v)) := by
  unfold OsNormal
  rw [tie_os]
  exact osNormal_normOs v

theorem os_norm (v : C3Run e) {s : ℕ} (hs : s < 3) : (tie (normOs v)).os.1 s = (tie v).os.1 s := by
  rw [tie_os, tie_os]
  exact normOs_os_of_lt hs

end run

/-! ## The read facts -/

section src

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **The normalized source** of a source of steps `stp`: each run `normOs`'s. -/
def normSrc (stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)) :
    Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp) :=
  fun R ω t Lb => normOs (stp R ω t Lb)

/-- **`MsgsReadC3` on the part `part` of the steps, past the part `prev`** (`RepsMsgsRead`'s form): two accepted outer
proofs whose layers agree on the read wires of stage `i`'s commit strings have the same round-`i` messages in the
source's parts `part`, at a round that closes in `part` in both runs and not in `prev` in both. -/
def PartMsgsRead (prev part : C3Run (holoAt st hp) → List Step)
    (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop) : Prop :=
  ∀ i R ω t R₂ τ o R' ω' t' R₂' τ' o', FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
    FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R' ω' t' R₂' τ' o' →
    maskOf Vs (skOf Vs own sel) i (layers zs dg N k Rw R ω t R₂ τ) =
      maskOf Vs (skOf Vs own sel) i (layers zs dg N k Rw R' ω' t' R₂' τ') →
    ¬ (i < ((prev (src R ω t (layers zs dg N k Rw R ω t R₂ τ))).filterMap Sum.getRight?).length ∧
      i < ((prev (src R' ω' t' (layers zs dg N k Rw R' ω' t' R₂' τ'))).filterMap Sum.getRight?).length) →
    i < ((part (src R ω t (layers zs dg N k Rw R ω t R₂ τ))).filterMap Sum.getRight?).length →
    i < ((part (src R' ω' t' (layers zs dg N k Rw R' ω' t' R₂' τ'))).filterMap Sum.getRight?).length →
    (cutOf (part (src R ω t (layers zs dg N k Rw R ω t R₂ τ))))[i]? =
      (cutOf (part (src R' ω' t' (layers zs dg N k Rw R' ω' t' R₂' τ'))))[i]?

/-- **The sparse phases' committed rounds** (named; from each phase's tie, `SparseTie`). -/
abbrev SparseMsgsRead (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop) : Prop :=
  PartMsgsRead repPart sparsePart zs dg N k Rw src own sel

/-- **R's opening's committed rounds** (named; from `OpenTie`). -/
abbrev OpenMsgsRead (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop) : Prop :=
  PartMsgsRead sparsePart openPart zs dg N k Rw src own sel

/-- **R's Ligerito's committed rounds** (named; from `LevelTie`). -/
abbrev LigMsgsRead (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop) : Prop :=
  PartMsgsRead openPart ligPart zs dg N k Rw src own sel

/-- **The openings at stratum `s` are read from the layers** (named; for the reps from `RepUnit.opens`, for R from
`LevelTie`'s): two accepted outer proofs whose layers agree on every read wire have the same openings at `s`. -/
def OpensReadAt (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)) (s : ℕ) : Prop :=
  ∀ R ω t R₂ τ o R' ω' t' R₂' τ' o', FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
    FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R' ω' t' R₂' τ' o' →
    (∀ j g, ReadW Vs j g → layers zs dg N k Rw R ω t R₂ τ j g = layers zs dg N k Rw R' ω' t' R₂' τ' j g) →
    (src R ω t (layers zs dg N k Rw R ω t R₂ τ)).os.1 s = (src R' ω' t' (layers zs dg N k Rw R' ω' t' R₂' τ')).os.1 s

/-- **R's Ligerito ends in a coin** at every accepted outer proof (named; from an accepting `LevelTie`, whose run
consumes the last level's queries, `qbShape lv false`), so the last group is the openings step alone
(`cutOf_runSteps_last`). -/
def LigWhole (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)) : Prop :=
  AtSrc zs dg N k Rw src fun _ u => ∃ (l : List Step) (c : Game.Msg), u.lg = l ++ [.inr c]

/-! ## `MsgsReadC3` from the parts -/

theorem skLeaf_of_not_lt {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop} {m i : ℕ}
    (hi : ¬ i + 1 < m) (j : Fin S) : skLeaf Vs own sel m i j = Finset.univ := ite_eq_right hi

/-- **At the last round the mask is every read wire.** -/
theorem read_of_maskLeaf {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop} {m i : ℕ}
    (hi : ¬ i + 1 < m) {Xs Xs' : (j : Fin S) → (Vs j).Bits}
    (h : maskOf Vs (skLeaf Vs own sel m) i Xs = maskOf Vs (skLeaf Vs own sel m) i Xs') :
    ∀ j g, ReadW Vs j g → Xs j g = Xs' j g := by
  intro j g hg
  have hj := congrFun (congrFun h j) g
  simp only [maskOf] at hj
  rw [skLeaf_of_not_lt hi j] at hj
  simpa [hg] using hj

/-- **A run has `r` coins** at an accepted outer proof, from the whole history and each stage's coins. -/
theorem coins_src {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    (hleaf : HLeafC3 zs dg N k Rw) (hcoins : AtSrc zs dg N k Rw src fun t u => StageCoins (CT := CT) (t.map Prod.snd) u)
    {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (hA : FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o) :
    ((ligPart (src R ω t (layers zs dg N k Rw R ω t R₂ τ))).filterMap Sum.getRight?).length = r := by
  have hr : r ≤ (t.map Prod.snd).length :=
    Nat.le_trans (Nat.le_add_right r 1) (Nat.le_trans (hleaf R ω t R₂ τ o hA) (Nat.le_of_eq (List.length_map _).symm))
  rw [← coins_runSteps, coinClause_of_stageCoins hr (hcoins R ω t R₂ τ o hA)]
  exact FlockVBridge.Family.length_coinSteps CT hr

/-- **`MsgsReadC3` from the parts' read facts**, at a source whose runs have `r` coins (`HLeafC3`, `StageCoins`), normal
openings (`OsNormal`), openings read from the layers at strata 0–2 (`OpensReadAt`) and a Ligerito ending in a coin
(`LigWhole`). -/
theorem msgsReadC3_of_parts {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
    (hleaf : HLeafC3 zs dg N k Rw) (hcoins : AtSrc zs dg N k Rw src fun t u => StageCoins (CT := CT) (t.map Prod.snd) u)
    (hrp : RepsMsgsRead zs dg N k Rw src own sel) (hsp : SparseMsgsRead zs dg N k Rw src own sel)
    (hop : OpenMsgsRead zs dg N k Rw src own sel) (hlg : LigMsgsRead zs dg N k Rw src own sel)
    (hn : AtSrc zs dg N k Rw src fun _ u => OsNormal u) (hos : ∀ s < 3, OpensReadAt zs dg N k Rw src s)
    (hw : LigWhole zs dg N k Rw src) : MsgsReadC3 zs dg N k Rw src own sel := by
  intro i R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX
  have hR : (ℐ).r = r + 1 := rfl
  have hc := coins_src hleaf hcoins hA
  have hc' := coins_src hleaf hcoins hA'
  show (cutOf (runSteps (src R ω t (layers zs dg N k Rw R ω t R₂ τ))))[i]? =
    (cutOf (runSteps (src R' ω' t' (layers zs dg N k Rw R' ω' t' R₂' τ'))))[i]?
  rcases Nat.lt_trichotomy i r with hi | hi | hi
  · rw [maskOf_skLeaf (by omega : i + 1 < (ℐ).r)] at hX
    exact cutOf_runSteps_eq (by omega) (by omega) (hrp i R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX)
      (hsp i R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX) (hop i R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX)
      fun h => hlg i R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX h (by omega) (by omega)
  · obtain ⟨l, c, hl⟩ := hw R ω t R₂ τ o hA
    obtain ⟨l', c', hl'⟩ := hw R' ω' t' R₂' τ' o' hA'
    have hrd := read_of_maskLeaf (by omega : ¬ i + 1 < (ℐ).r) hX
    rw [cutOf_runSteps_last hl (hc.trans hi.symm), cutOf_runSteps_last hl' (hc'.trans hi.symm),
      os_eq_of_normal (hn R ω t R₂ τ o hA) (hn R' ω' t' R₂' τ' o' hA')
        fun s hs => hos s hs R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hrd]
  · rw [cutOf_runSteps_none (u := src R ω t (layers zs dg N k Rw R ω t R₂ τ)) (i := i) (by omega),
      cutOf_runSteps_none (u := src R' ω' t' (layers zs dg N k Rw R' ω' t' R₂' τ')) (i := i) (by omega)]

/-! ## At the normalized tied source -/

theorem part_norm {part : C3Run (holoAt st hp) → List Step} (hpart : ∀ v, part (tie (normOs v)) = part (tie v))
    (stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)) (R : Reg)
    (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (Lb : (j : Fin S) → (Vs j).Bits) :
    part (tiedSrc (normSrc stp) R ω t Lb) = part (tiedSrc stp R ω t Lb) :=
  hpart (stp R ω t Lb)

theorem partMsgsRead_norm {prev part : C3Run (holoAt st hp) → List Step}
    (hprev : ∀ v, prev (tie (normOs v)) = prev (tie v)) (hpart : ∀ v, part (tie (normOs v)) = part (tie v))
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
    (h : PartMsgsRead prev part zs dg N k Rw (tiedSrc stp) own sel) :
    PartMsgsRead prev part zs dg N k Rw (tiedSrc (normSrc stp)) own sel := by
  intro i R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX
  rw [part_norm (part := prev) hprev stp R ω t, part_norm (part := prev) hprev stp R' ω' t',
    part_norm (part := part) hpart stp R ω t, part_norm (part := part) hpart stp R' ω' t']
  exact h i R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX

theorem repsMsgsRead_norm {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
    (h : RepsMsgsRead zs dg N k Rw (tiedSrc stp) own sel) :
    RepsMsgsRead zs dg N k Rw (tiedSrc (normSrc stp)) own sel := by
  intro i R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX
  rw [part_norm (part := repPart) repPart_norm stp R ω t, part_norm (part := repPart) repPart_norm stp R' ω' t']
  exact h i R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX

theorem ligWhole_norm {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    (h : LigWhole zs dg N k Rw (tiedSrc stp)) : LigWhole zs dg N k Rw (tiedSrc (normSrc stp)) := by
  intro R ω t R₂ τ o hA
  obtain ⟨l, c, hl⟩ := h R ω t R₂ τ o hA
  exact ⟨l, c, (tie_normOs_steps (stp R ω t (layers zs dg N k Rw R ω t R₂ τ))).2.2.2.2.2.trans hl⟩

/-- **`MsgsReadC3` at the normalized tied source**, from the read facts at the tied source `tiedSrc stp`. -/
theorem msgsReadC3_tied {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
    (hleaf : HLeafC3 zs dg N k Rw)
    (hcoins : AtSrc zs dg N k Rw (tiedSrc stp) fun t u => StageCoins (CT := CT) (t.map Prod.snd) u)
    (hrp : RepsMsgsRead zs dg N k Rw (tiedSrc stp) own sel) (hsp : SparseMsgsRead zs dg N k Rw (tiedSrc stp) own sel)
    (hop : OpenMsgsRead zs dg N k Rw (tiedSrc stp) own sel) (hlg : LigMsgsRead zs dg N k Rw (tiedSrc stp) own sel)
    (hos : ∀ s < 3, OpensReadAt zs dg N k Rw (tiedSrc stp) s) (hw : LigWhole zs dg N k Rw (tiedSrc stp)) :
    MsgsReadC3 zs dg N k Rw (tiedSrc (normSrc stp)) own sel :=
  msgsReadC3_of_parts (src := tiedSrc (normSrc stp)) hleaf hcoins (repsMsgsRead_norm hrp)
    (partMsgsRead_norm (prev := repPart) (part := sparsePart) repPart_norm sparsePart_norm hsp)
    (partMsgsRead_norm (prev := sparsePart) (part := openPart) sparsePart_norm openPart_norm hop)
    (partMsgsRead_norm (prev := openPart) (part := ligPart) openPart_norm ligPart_norm hlg)
    (fun R ω t R₂ τ o _ => osNormal_tie_norm (stp R ω t (layers zs dg N k Rw R ω t R₂ τ)))
    (fun s hs R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX =>
      (os_norm (stp R ω t (layers zs dg N k Rw R ω t R₂ τ)) hs).trans
        ((hos s hs R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX).trans
          (os_norm (stp R' ω' t' (layers zs dg N k Rw R' ω' t' R₂' τ')) hs).symm))
    (ligWhole_norm hw)

/-- **`SrcOK` at the normalized tied source**: `srcOK_tied_of_shapes` at `normSrc stp`, its reads `msgsReadC3_tied`,
and its other clauses as at `tiedSrc stp`, since no stage reads what `normOs` changes. -/
theorem srcOK_tied_norm {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
    (hleaf : HLeafC3 zs dg N k Rw)
    (hcoins : AtSrc zs dg N k Rw (tiedSrc stp) fun t u => StageCoins (CT := CT) (t.map Prod.snd) u)
    (hrp : RepsMsgsRead zs dg N k Rw (tiedSrc stp) own sel) (hsp : SparseMsgsRead zs dg N k Rw (tiedSrc stp) own sel)
    (hop : OpenMsgsRead zs dg N k Rw (tiedSrc stp) own sel) (hlg : LigMsgsRead zs dg N k Rw (tiedSrc stp) own sel)
    (hos : ∀ s < 3, OpensReadAt zs dg N k Rw (tiedSrc stp) s) (hw : LigWhole zs dg N k Rw (tiedSrc stp))
    (hcap : ∃ j, T j = Cap Dig) (hbool : ∃ j, T j = Bool) (hty : ∃ j, T j = (Opens GF128 Dig × Salts (List UInt8)))
    (hT : AtSrc zs dg N k Rw (tiedSrc stp) fun _ u => ShapeTys T (holoAt st hp) u)
    (hplay : AtSrc zs dg N k Rw (tiedSrc stp) fun _ u => StagesPlay (holoAt st hp) u) :
    SrcOK zs dg N k Rw (tiedSrc (normSrc stp)) own sel :=
  srcOK_tied_of_shapes (stp := normSrc stp) hleaf hcoins (msgsReadC3_tied hleaf hcoins hrp hsp hop hlg hos hw) hcap
    hbool hty hT hplay

/-- **The reps' tie at the normalized tied source** (`repTie_normOs`). -/
theorem srcRepTie_norm {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    (h : FlockC3.Reps.SrcRepTie zs dg N k Rw (tiedSrc stp)) :
    FlockC3.Reps.SrcRepTie zs dg N k Rw (tiedSrc (normSrc stp)) :=
  fun R ω t R₂ τ o hA i => repTie_normOs (h R ω t R₂ τ o hA i)

end src

end

end FlockC3.Compose
