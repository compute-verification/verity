import Proofs.Flock.C3.Compose.Decoder

/-!
# The reps' tie: each rep's steps are a rep unit V*'s layers check

`RepTie e u i`: rep `i`'s steps in the run `u` (`u.ts i`) are an accepting run of the executable's holo rep
(`RepUnit`) at the run's statement, schedule, extra claims at its points, level-0 cap and openings: the values, proof,
record and claims V*'s rep-`i` units check on the layers. `SrcRepTie` is n7's tie for the reps beside `SrcOK`: the
source's runs are tied so at every accepted outer proof.

* `reps_prefix`, `repsReadAt_of_tie`: past the cap's and the points' steps, `runSteps u` begins with rep 0's steps in
  the reps' interleaving and then rep 1's, so the two ties are the reps' read fact on the run's own steps
  (`RepsReadAt`). `repsRead_of_tie` and `hReps_of_tie` are `RepsRead` and `HReps` at the fixed decoder.
* `cutOf_append`, `c3Msgs_repPart`, `msgsRead_repPart`: a round that closes in the reps' part of the steps (the cap's
  and the points' steps, then the reps': `repPart`) has the messages of that part. So `MsgsReadC3` at those rounds,
  which are committed ones, is a fact of the reps' parts alone (`RepsMsgsRead`).
* `srcRepTie_of_alg`: the tie from G's items 1–4 at a rep decoder of V*'s layers (`AlgebraDecodes`, `AccChains`, which
  give a unit's `alg` and `acc`) and `RepUnitOfAlg`, the rest of the unit at those values, which is not proved here.
-/

namespace FlockC3.Reps

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockVBridge.Family (skOf maskOf)
open FlockVBridge.InnerAccepts (firstSteps)
open FlockC3.Compose

/-! ## The tie and the read fact -/

/-- **Rep `i`'s steps are a rep unit**: an accepting run of the executable's holo rep at the run's statement, schedule,
extra claims at the run's points, level-0 cap and openings, on rep `i`'s steps. -/
def RepTie (e : ExecHolo) (u : C3Run e) (i : Fin 2) : Prop :=
  Nonempty (FlockC3.HoloValues.RepUnit (e.cls.pubStatement e.regs) e.sch
    (extraClaims (e.cls.pubStatement e.regs) e.cls.ptLocal e.cls.mPts u.pts) u.cap₀ u.os.1 i (u.ts i))

/-- **`runSteps`' reps' prefix**: past the cap's and the points' steps, rep 0's steps in the reps' interleaving, then
rep 1's. -/
theorem reps_prefix {e : ExecHolo} (u : C3Run e) : firstSteps (u.ts 0) ++ u.ts 1 <+: (runSteps u).drop 2 :=
  List.prefix_append_of_prefix (List.prefix_append_of_prefix (List.prefix_append_of_prefix
    (List.prefix_append_of_prefix (List.prefix_append _ _))))

/-- **The reps' read fact on the run's own steps, from the two ties.** -/
theorem repsReadAt_of_tie {e : ExecHolo} {u : C3Run e} (h₀ : RepTie e u 0) (h₁ : RepTie e u 1) :
    FlockC3.HoloValues.RepsReadAt e (runSteps u) u :=
  ⟨u.ts 0, u.ts 1, reps_prefix u, h₀, h₁⟩

/-! ## The messages of the reps' part -/

/-- A list of steps' messages cut at its coins (`PlaysCoin.cut`). -/
abbrev cutOf (l : List Step) : List (List Game.Msg) := PlaysCoin.cut (l.map Exec.kindOf) (l.filterMap Sum.getLeft?)

theorem cutOf_ne_nil (l : List Step) : cutOf l ≠ [] := FlockVBridge.Family.cut_ne_nil _ _

/-- **The rounds that close in a prefix are the prefix's**: steps `a ++ b` have `a`'s messages at each round before
`a`'s last coin. -/
theorem cutOf_append (b : List Step) : ∀ (a : List Step) (i : ℕ), i < (a.filterMap Sum.getRight?).length →
    (cutOf (a ++ b))[i]? = (cutOf a)[i]?
  | [], _, hi => absurd hi (Nat.not_lt_zero _)
  | .inl m :: a, i, hi => by
      have hl : ((Sum.inl m :: a : List Step).filterMap Sum.getRight?).length =
          (a.filterMap Sum.getRight?).length := rfl
      have ih := cutOf_append b a i (by omega)
      change (PlaysCoin.consHead m (cutOf (a ++ b)))[i]? = (PlaysCoin.consHead m (cutOf a))[i]?
      obtain ⟨g, gs, hg⟩ := List.exists_cons_of_ne_nil (cutOf_ne_nil (a ++ b))
      obtain ⟨g', gs', hg'⟩ := List.exists_cons_of_ne_nil (cutOf_ne_nil a)
      rw [hg, hg'] at ih ⊢
      cases i with
      | zero =>
          simp only [List.getElem?_cons_zero, Option.some.injEq] at ih
          simp only [PlaysCoin.consHead, List.getElem?_cons_zero, ih]
      | succ j => simpa only [PlaysCoin.consHead, List.getElem?_cons_succ] using ih
  | .inr c :: a, i, hi => by
      have hl : ((Sum.inr c :: a : List Step).filterMap Sum.getRight?).length =
          (a.filterMap Sum.getRight?).length + 1 := rfl
      change ([] :: cutOf (a ++ b))[i]? = ([] :: cutOf a)[i]?
      cases i with
      | zero => rfl
      | succ j =>
          rw [List.getElem?_cons_succ, List.getElem?_cons_succ]
          exact cutOf_append b a j (by omega)

/-- **The reps' part of a run's steps**: the cap's and the points' steps, then the reps' in their interleaving. -/
def repPart {e : ExecHolo} (u : C3Run e) : List Step := (runSteps u).take 2 ++ (firstSteps (u.ts 0) ++ u.ts 1)

theorem runSteps_split {e : ExecHolo} (u : C3Run e) : ∃ rest, runSteps u = repPart u ++ rest := by
  obtain ⟨rest, h⟩ := reps_prefix u
  exact ⟨rest, by rw [repPart, List.append_assoc ((runSteps u).take 2) (firstSteps (u.ts 0) ++ u.ts 1) rest, h,
    List.take_append_drop]⟩

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt T CT st hp

section source

variable {S : ℕ} {Vs : Fin S → VStmt}

/-- **The source's round-`i` messages, at a round that closes in the reps' part, are that part's.** -/
theorem c3Msgs_repPart (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (Lb : (j : Fin S) → (Vs j).Bits) (i : ℕ)
    (hi : i < ((repPart (src R ω t Lb)).filterMap Sum.getRight?).length) :
    (c3Msgs src R ω t Lb)[i]? = (cutOf (repPart (src R ω t Lb)))[i]? := by
  obtain ⟨rest, h⟩ := runSteps_split (src R ω t Lb)
  show (cutOf (runSteps (src R ω t Lb)))[i]? = _
  rw [h]
  exact cutOf_append rest _ i hi

/-- **`MsgsReadC3`'s reps' part**: two accepted outer proofs whose layers agree on the read wires of stage `i`'s
commit strings have the same messages at round `i` of the source's reps' parts, where it closes in both. -/
def RepsMsgsRead (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop) : Prop :=
  ∀ i R ω t R₂ τ o R' ω' t' R₂' τ' o', FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
    FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R' ω' t' R₂' τ' o' →
    maskOf Vs (skOf Vs own sel) i (layers zs dg N k Rw R ω t R₂ τ) =
      maskOf Vs (skOf Vs own sel) i (layers zs dg N k Rw R' ω' t' R₂' τ') →
    i < ((repPart (src R ω t (layers zs dg N k Rw R ω t R₂ τ))).filterMap Sum.getRight?).length →
    i < ((repPart (src R' ω' t' (layers zs dg N k Rw R' ω' t' R₂' τ'))).filterMap Sum.getRight?).length →
    (cutOf (repPart (src R ω t (layers zs dg N k Rw R ω t R₂ τ))))[i]? =
      (cutOf (repPart (src R' ω' t' (layers zs dg N k Rw R' ω' t' R₂' τ'))))[i]?

/-- **`MsgsReadC3` at the rounds that close in the reps' parts**, from those parts' read fact. -/
theorem msgsRead_repPart {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
    (h : RepsMsgsRead zs dg N k Rw src own sel) {i : ℕ}
    {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    {R' : Reg} {ω' : L.Ω} {t' : List (Cm × (ℐ).Coin)} {R₂' : Reg₂} {τ' o'}
    (hA : FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o)
    (hA' : FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R' ω' t' R₂' τ' o')
    (hX : maskOf Vs (skOf Vs own sel) i (layers zs dg N k Rw R ω t R₂ τ) =
      maskOf Vs (skOf Vs own sel) i (layers zs dg N k Rw R' ω' t' R₂' τ'))
    (hi : i < ((repPart (src R ω t (layers zs dg N k Rw R ω t R₂ τ))).filterMap Sum.getRight?).length)
    (hi' : i < ((repPart (src R' ω' t' (layers zs dg N k Rw R' ω' t' R₂' τ'))).filterMap Sum.getRight?).length) :
    (c3Msgs src R ω t (layers zs dg N k Rw R ω t R₂ τ))[i]? =
      (c3Msgs src R' ω' t' (layers zs dg N k Rw R' ω' t' R₂' τ'))[i]? := by
  rw [c3Msgs_repPart src R ω t _ i hi, c3Msgs_repPart src R' ω' t' _ i hi']
  exact h i R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX hi hi'

/-! ## The source tied to V*'s rep units -/

/-- **The source tied to V*'s rep units** (n7's tie for the reps): at every accepted outer proof, each rep's steps in
the source's run are a rep unit (`RepTie`). -/
def SrcRepTie (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)) : Prop :=
  AtSrc zs dg N k Rw src fun _ u => ∀ i : Fin 2, RepTie (holoAt st hp) u i

/-- **The reps' read fact on the source's own steps**, at a source tied to V*'s rep units: `repsRead_c3DecOf`'s
`hrp`. -/
theorem srcRepsReadAt_of_tie {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    (h : SrcRepTie zs dg N k Rw src) :
    AtSrc zs dg N k Rw src fun _ u => FlockC3.HoloValues.RepsReadAt (holoAt st hp) (runSteps u) u :=
  fun R ω t R₂ τ o hacc => repsReadAt_of_tie (h R ω t R₂ τ o hacc 0) (h R ω t R₂ τ o hacc 1)

/-- **What is left of a rep unit once its values meet E's preconditions** (not proved here): at every accepted outer
proof, where rep `i`'s values `rep Lb i` in V*'s layers meet them at `sh` (`alg`, `acc`), the source's rep-`i` steps
are a rep unit at those values. Its parts, with the lines each should take:
* the proof `p` from the values (`zp`, `lp`, `foldPts`, the ring switches `rs`, `rsShape`, `rsVal`) and the record's
  coins (`coins`), as G's `RunsFrom` gives them at the table files' decode: 80–120;
* the setup and schedule of the run's statement (`m`, `kL`, `mS`, `kS`, `fold`, `fast`, `sched`, `merkle`, the extra
  claims): 40–60;
* the record's frames at the holo shape (`k0`, `wf`, the counts and answers `n₀`–`ar`, `lc`, `cl`, `rsRounds`, `lg`):
  `framed_frames` and `hlg_of_frames` at holo, 150–250 beside c1's generalization (600);
* the source's rep steps as `holoRepMsgs` of `p` woven with the record's coins (`kinds`, `msgs`, `coinSteps`), its cap
  `capDig p.cap` and openings `openRow p.ligerito sch` (`cap`, `opens`): about 60. -/
def RepUnitOfAlg (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (sh : Flock.Shape) (rep : ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData) : Prop :=
  ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ i : Fin 2,
    AlgebraHolds sh (rep (layers zs dg N k Rw R ω t R₂ τ) i) → AccHolds sh (rep (layers zs dg N k Rw R ω t R₂ τ) i) →
    ∃ U : FlockC3.HoloValues.RepUnit ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch
        (extraClaims ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).cls.ptLocal
          (holoAt st hp).cls.mPts (src R ω t (layers zs dg N k Rw R ω t R₂ τ)).pts)
        (src R ω t (layers zs dg N k Rw R ω t R₂ τ)).cap₀ (src R ω t (layers zs dg N k Rw R ω t R₂ τ)).os.1 i
        ((src R ω t (layers zs dg N k Rw R ω t R₂ τ)).ts i),
      U.sh = sh ∧ U.d = rep (layers zs dg N k Rw R ω t R₂ τ) i

/-- **The reps' tie from G's items 1–4** at a rep decoder `rep` of V*'s layers and the rest of each unit at its
values (`RepUnitOfAlg`). -/
theorem srcRepTie_of_alg {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape} {rep : ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    (halg : AlgebraDecodes (I := ℐ) zs dg N k Rw sh rep) (hacc : AccChains (I := ℐ) zs dg N k Rw sh rep)
    (h : RepUnitOfAlg zs dg N k Rw src sh rep) : SrcRepTie zs dg N k Rw src := by
  intro R ω t R₂ τ o hA i
  obtain ⟨U, -, -⟩ := h R ω t R₂ τ o hA i (halg R ω t R₂ τ o hA i) (hacc R ω t R₂ τ o hA i)
  exact ⟨U⟩

end source

/-! ## At the fixed decoder -/

section honest

variable {ish : RecVStar.Inner} {ss : Array VStar.Stmt} {hss : RecVStar.statements ish = .ok ss}
  {bind : Fin (vstarCount ish) → VStar.Bindings} {Vs : Fin (vstarCount ish) → VStmt}
  {hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs}
  {zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
    (Vs j).Outer Reg'}
  {own : Fin (vstarCount ish) → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
  {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
  {tr : Lean.Json} {record : String} {reg : Lean.Json} {dg : ByteArray → ByteArray} {N k Rw : ℕ}
  {src : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) →
    C3Run (holoAt st hp)}

local notation "𝒵" => zs
local notation "𝓓" => C3Family.honest (xR := xR) ish hss bind hob 𝒵 own sel bc
  (c3DecOf 𝒵 dg N k Rw src own sel) tr record reg

/-- **`RepsRead` at the fixed decoder, from the reps' tie.** -/
theorem repsRead_of_tie (hs : SrcOK 𝒵 dg N k Rw src own sel) (ht : SrcRepTie 𝒵 dg N k Rw src) :
    FlockC3.HoloValues.RepsRead 𝓓 dg N k Rw :=
  repsRead_c3DecOf hs (srcRepsReadAt_of_tie ht)

/-- **(n2) at the fixed decoder, from the reps' tie**: each rep plays `holoRepC` to its decoded output. -/
theorem hReps_of_tie (hs : SrcOK 𝒵 dg N k Rw src own sel) (ht : SrcRepTie 𝒵 dg N k Rw src) :
    FlockC3.HoloValues.HReps 𝓓 dg N k Rw :=
  FlockC3.HoloValues.hReps_honest_of_read (repsRead_of_tie hs ht)

end honest

end

end FlockC3.Reps
