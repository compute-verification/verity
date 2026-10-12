import Proofs.Flock.C3.Compose.SrcClauses
import Proofs.Flock.Recursive.ForkRuns
import Proofs.Flock.C3.Reps.Tie
import Proofs.Flock.C3.Sparse.SrcTie
import Proofs.Flock.C3.Opening.SrcOpening
import Proofs.Flock.C3.RegLevels.SrcTie

/-!
# n7: the tied source, each stage's output its play's

`tie u` keeps a run's steps, cap, link points and openings, and sets each stage's output to the one its game plays to
on the run's steps for it (`playOut`), stage by stage, since each game reads the outputs before it: `repAt` reads the
link points, `sparseAt` its rep's output, `openingAt` the claims and `ligAt` the opening's output. Where each stage
plays on its steps to some output (`StagesPlay`), `RunPlays` holds at `tie u` (`runPlays_tie`), so `srcOK_of_plays`
applies at the tied source `tiedSrc stp` of any source of steps `stp` (`srcOK_tied`).

The stages play where the source is tied (`stagesPlay_of_ties`): each rep by its unit (`RepTie`, `rep_of_unit`), each
sparse phase by its RecSparse unit (`SparseTie`, `plays_of_tie`), R's opening on w's ring switches (`OpenTie`) where
its read fact's checks pass (`OpeningReadAt`), and R's Ligerito by its accepting run (`LevelTie`). A stage's message
types follow from its kinds (`tysAt_of_kinds`) where the shape's are among `T`.
-/

namespace FlockC3.Compose

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.HoloValues (repAt)
open FlockC3.Sparse (sparseAt)
open FlockC3.Opening (openingAt)
open FlockC3.RegLevels (ligAt)

/-! ## A stage's output -/

open Classical in
/-- A game's output on the steps `ts`: the one it plays to there, where it plays to one, else `d`. -/
def playOut {α : Type} (g : Game α) (ts : List Step) (d : α) : α :=
  if h : ∃ a, PlaysT g ts a then h.choose else d

theorem playOut_eq {α : Type} {g : Game α} {ts : List Step} {a : α} (d : α) (h : PlaysT g ts a) :
    playOut g ts d = a := by
  have hx : ∃ a, PlaysT g ts a := ⟨a, h⟩
  unfold playOut
  rw [dif_pos hx]
  exact PlaysT.det hx.choose_spec h

/-! ## The tied run -/

variable {e : ExecHolo}

/-- `u` with each rep's output and pending result its play's on `u.ts i`. -/
def tieReps (u : C3Run e) : C3Run e :=
  { u with
    rep := fun i => ((playOut (repAt e u i) (u.ts i) (some (u.rep i), u.pend i)).1).getD (u.rep i)
    pend := fun i => (playOut (repAt e u i) (u.ts i) (some (u.rep i), u.pend i)).2 }

/-- `u` with each rep's registered claim its sparse phase's on `u.sp i`. -/
def tieSparse (u : C3Run e) : C3Run e :=
  { u with claim := fun i => (playOut (sparseAt e u i) (u.sp i) (some (u.claim i))).getD (u.claim i) }

/-- `u` with R's opening's output its play's on `u.op`. -/
def tieOpening (u : C3Run e) : C3Run e :=
  { u with opened := (playOut (openingAt u) u.op (some u.opened)).getD u.opened }

/-- `u` with R's Ligerito's result its play's on `u.lg`. -/
def tieLig (u : C3Run e) : C3Run e :=
  { u with pendR := playOut (ligAt e u) u.lg u.pendR }

/-- **The tied run**: `u`'s steps, each stage's output its play's, in the stages' order. -/
def tie (u : C3Run e) : C3Run e := tieLig (tieOpening (tieSparse (tieReps u)))

/-- **Each stage plays on the run's steps for it to some output**, at the run's own earlier outputs. -/
def StagesPlay (e : ExecHolo) (u : C3Run e) : Prop :=
  (∀ i : Fin 2, ∃ a p, PlaysT (repAt e u i) (u.ts i) (some a, p)) ∧
    (∀ i : Fin 2, ∃ a, PlaysT (sparseAt e u i) (u.sp i) (some a)) ∧
    (∃ a, PlaysT (openingAt u) u.op (some a)) ∧ ∃ p, PlaysT (ligAt e u) u.lg p

/-- **`RunPlays` at a tied run** where each stage plays on its steps to some output. -/
theorem runPlays_tie {u : C3Run e} (h : StagesPlay e (tie u)) : RunPlays e (tie u) := by
  obtain ⟨h₀, h₁, h₂, h₃⟩ := h
  refine ⟨fun i => ?_, fun i => ?_, ?_, ?_⟩
  · obtain ⟨a, p, ha⟩ := h₀ i
    have ha' : PlaysT (repAt e u i) (u.ts i) (some a, p) := ha
    have e₁ : playOut (repAt e u i) (u.ts i) (some (u.rep i), u.pend i) = (some a, p) := playOut_eq _ ha'
    show PlaysT (repAt e u i) (u.ts i)
      (some (((playOut (repAt e u i) (u.ts i) (some (u.rep i), u.pend i)).1).getD (u.rep i)),
        (playOut (repAt e u i) (u.ts i) (some (u.rep i), u.pend i)).2)
    rw [e₁]
    exact ha'
  · obtain ⟨a, ha⟩ := h₁ i
    have ha' : PlaysT (sparseAt e (tieReps u) i) (u.sp i) (some a) := ha
    have e₁ : playOut (sparseAt e (tieReps u) i) (u.sp i) (some (u.claim i)) = some a := playOut_eq _ ha'
    show PlaysT (sparseAt e (tieReps u) i) (u.sp i)
      (some ((playOut (sparseAt e (tieReps u) i) (u.sp i) (some (u.claim i))).getD (u.claim i)))
    rw [e₁]
    exact ha'
  · obtain ⟨a, ha⟩ := h₂
    have ha' : PlaysT (openingAt (tieSparse (tieReps u))) u.op (some a) := ha
    have e₁ : playOut (openingAt (tieSparse (tieReps u))) u.op (some u.opened) = some a := playOut_eq _ ha'
    show PlaysT (openingAt (tieSparse (tieReps u))) u.op
      (some ((playOut (openingAt (tieSparse (tieReps u))) u.op (some u.opened)).getD u.opened))
    rw [e₁]
    exact ha'
  · obtain ⟨p, hp⟩ := h₃
    have hp' : PlaysT (ligAt e (tieOpening (tieSparse (tieReps u)))) u.lg p := hp
    have e₁ : playOut (ligAt e (tieOpening (tieSparse (tieReps u)))) u.lg u.pendR = p := playOut_eq _ hp'
    show PlaysT (ligAt e (tieOpening (tieSparse (tieReps u)))) u.lg
      (playOut (ligAt e (tieOpening (tieSparse (tieReps u)))) u.lg u.pendR)
    rw [e₁]
    exact hp'

/-! ## The stages play where the source is tied -/

/-- **A rep tied to a rep unit plays on its steps to an output.** -/
theorem reps_plays_of_tie {u : C3Run e} {i : Fin 2} (h : FlockC3.Reps.RepTie e u i) :
    ∃ a p, PlaysT (repAt e u i) (u.ts i) (some a, p) := by
  obtain ⟨U⟩ := h
  obtain ⟨zc, lc, pend, hpl, -⟩ := FlockC3.HoloValues.rep_of_unit e.cls.hm U
  exact ⟨(zc, lc), pend, hpl⟩

/-- **R's Ligerito tied to an accepting run plays on its steps** (`regLevels_of_readAt`'s play). -/
theorem lig_plays_of_levelTie {cap₀ : Cap Dig} {u : C3Run e} (h : FlockC3.RegLevels.LevelTie cap₀ e u) :
    ∃ p, PlaysT (ligAt e u) u.lg p := by
  obtain ⟨msId, m, sch, Tw, sws, nc, cap, p, rt, rt', hf, hsch, -, hop, hwf, hlig, hk, hl, hr, ho⟩ := h
  have hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default) := fun i hi =>
    fast100_level m sch hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  obtain ⟨pend, -, -, hpl, -⟩ := ligerito_accepts_opening (Flock.MerkleScheme.sha512 msId) sch Tw sws
    nc u.opened hop cap p 2 capDig u.os.1 (fast100_dims m sch hf).1 hlv
    (fun i hi => fast100_alpha m sch hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi))
    (fun ℓ _ j _ => show (u.os.1 2 ℓ j).1 = (openRow p sch ℓ j).1 from congrArg Prod.fst (ho ℓ j)) hwf hlig
  rw [hsch] at hpl
  exact ⟨pend, FlockC3.RegLevels.Read.playsT_lig_of_tape hpl hk hl hr⟩

/-- **R's opening on w's ring switches plays on its steps to an output** where its read fact's checks pass: the read
fact's opening steps are a prefix of the run's past both sparse phases, of `u.op`'s length, so they are `u.op`. -/
theorem opening_plays_of {st : Flock.Stmt} {hp : HoloParams} {u : C3Run (holoAt st hp)} {sh : Flock.Shape}
    {w : Array F128} {ops : FlockVBridge.AccChain.Openings} (ht : FlockC3.Opening.OpenTie sh w ops u)
    (hr : OpeningReadAt (runSteps u) u) : ∃ a, PlaysT (openingAt u) u.op (some a) := by
  obtain ⟨_, M, a, _, _, _, _, ρ, γ, -, hop, hz, hrel, hpre⟩ := hr
  obtain ⟨ρ', γ', hu⟩ := ht
  have hm := seven_le_mReg (holoAt st hp).cls.shape
  obtain ⟨x, hx⟩ := FlockC3.Opening.opResult_some hm (fun i => (u.claim i).toClaim)
    (fun i => FlockSoundness.Refine.vecOf 128 M.rs[i.val]!) ρ γ
    fun i => FlockC3.Opening.packed_check hm (hrel i) (FlockC3.Opening.checks_of_run hop hz i)
  have hpl := FlockC3.Opening.plays_opening hm (fun i => (u.claim i).toClaim)
    (fun i => FlockSoundness.Refine.vecOf 128 M.rs[i.val]!) ρ γ
  rw [hx] at hpl
  rw [FlockC3.Opening.afterSparse_runSteps] at hpre
  have hlen : (FlockC3.Opening.openSteps (fun i => FlockSoundness.Refine.vecOf 128 M.rs[i.val]!) ρ γ).length =
      u.op.length := by
    rw [hu]
    rfl
  have he : FlockC3.Opening.openSteps (fun i => FlockSoundness.Refine.vecOf 128 M.rs[i.val]!) ρ γ = u.op :=
    (List.prefix_of_prefix_length_le hpre (List.prefix_append _ _) hlen.le).eq_of_length hlen
  rw [← he]
  exact ⟨x, hpl⟩

/-- **Each stage plays at a tied run**: the reps by their units, the sparse phases by their RecSparse units, R's
opening on w's ring switches where its read fact holds, and R's Ligerito by its accepting run. -/
theorem stagesPlay_of_ties {st : Flock.Stmt} {hp : HoloParams} {u : C3Run (holoAt st hp)} {cap₀ : Cap Dig}
    {sh : Flock.Shape} {w : Array F128} {ops : FlockVBridge.AccChain.Openings}
    (hr : ∀ i, FlockC3.Reps.RepTie (holoAt st hp) u i) (hs : ∀ i, FlockC3.Sparse.SparseTie (holoAt st hp) u i)
    (ho : FlockC3.Opening.OpenTie sh w ops u) (hor : OpeningReadAt (runSteps u) u)
    (hl : FlockC3.RegLevels.LevelTie cap₀ (holoAt st hp) u) : StagesPlay (holoAt st hp) u :=
  ⟨fun i => reps_plays_of_tie (hr i), fun i => FlockC3.Sparse.plays_of_tie (hs i), opening_plays_of ho hor,
    lig_plays_of_levelTie hl⟩

/-- **A stage's messages are typed by `T`** where its kinds are a prefix of a shape whose message types are. -/
theorem tysAt_of_kinds {nT : ℕ} {T : Fin nT → Type} {l : List Step} {σ : List Kind} (hk : l.map kindOf <+: σ)
    (hT : ∀ M, Kind.msg M ∈ σ → ∃ j, T j = M) : TysAt T l := fun m hm =>
  hT m.1 (hk.subset (List.mem_map.2 ⟨.inl m, hm, rfl⟩))

/-! ## The tied source -/

section src

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **The tied source** of a source of steps `stp`: each run's steps are `stp`'s, and its outputs its stages' plays'. -/
def tiedSrc (stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)) :
    Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp) :=
  fun R ω t Lb => tie (stp R ω t Lb)

/-- **`SrcOK` at the tied source**: `srcOK_of_plays`, with `RunPlays` from each stage playing at the tied runs. -/
theorem srcOK_tied {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
    (hleaf : HLeafC3 zs dg N k Rw)
    (hcoins : AtSrc zs dg N k Rw (tiedSrc stp) fun t u => StageCoins (CT := CT) (t.map Prod.snd) u)
    (hread : MsgsReadC3 zs dg N k Rw (tiedSrc stp) own sel)
    (hcap : ∃ j, T j = Cap Dig) (hbool : ∃ j, T j = Bool) (hos : ∃ j, T j = (Opens GF128 Dig × Salts (List UInt8)))
    (htys : AtSrc zs dg N k Rw (tiedSrc stp) fun _ u => TysAt T (u.ts 0) ∧ TysAt T (u.ts 1) ∧ TysAt T (u.sp 0) ∧
      TysAt T (u.sp 1) ∧ TysAt T u.op ∧ TysAt T u.lg)
    (hplay : AtSrc zs dg N k Rw (tiedSrc stp) fun _ u => StagesPlay (holoAt st hp) u) :
    SrcOK zs dg N k Rw (tiedSrc stp) own sel :=
  srcOK_of_plays hleaf hcoins hread hcap hbool hos htys fun R ω t R₂ τ o h =>
    runPlays_tie (u := stp R ω t (layers zs dg N k Rw R ω t R₂ τ)) (hplay R ω t R₂ τ o h)

end src

end

end FlockC3.Compose
