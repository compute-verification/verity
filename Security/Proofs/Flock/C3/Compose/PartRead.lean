import Proofs.Flock.C3.Compose.LastRead
import Proofs.Flock.C3.Opening.SrcLayer

/-!
# n9: the parts' committed rounds from their stages

Each part of a run's steps is the part before it and a stage: the sparse phases after the reps' part, R's opening after
the sparse phases', R's Ligerito after R's opening's. Where the part before has the same coins at every accepted outer
proof and ends in a coin (`PartEnds`), the part's round `c₀ + j` is the stage's round `j` (`cutOf_shift_of`), so its
read fact is the stage's alone (`StageMsgsRead`, `partMsgsRead_of_stage`). R's opening's part ends in a coin, three
past the sparse phases' (`partEnds_open`). R's opening holds the ring switches V*'s layers hold (`OpenTieL`, `RsAt`) at
its first two rounds and nothing at its third, so `OpenMsgsRead` is that the ring switches are read at their rounds
(`RsRead`, `openMsgsRead_of_layer`).
-/

namespace FlockC3.Compose

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockVBridge.Family (skOf maskOf)
open FlockC3.Reps (cutOf repPart)
open FlockC3.Opening (openSteps OpenTieL RsAt)

/-! ## Past a coin, the rounds are the rest's -/

theorem coins_append (a b : List Step) : ((a ++ b).filterMap Sum.getRight?).length =
    (a.filterMap Sum.getRight?).length + (b.filterMap Sum.getRight?).length := by
  rw [FlockVBridge.Family.rights_append, List.length_append]

theorem cutOf_shift (c : Game.Msg) (b : List Step) : ∀ (a : List Step) (j : ℕ),
    (cutOf (a ++ [.inr c] ++ b))[(a.filterMap Sum.getRight?).length + j + 1]? = (cutOf b)[j]?
  | [], j => by
      change ([] :: cutOf b)[0 + j + 1]? = (cutOf b)[j]?
      rw [List.getElem?_cons_succ, Nat.zero_add]
  | .inl m :: a, j => by
      have ih := cutOf_shift c b a j
      change (PlaysCoin.consHead m (cutOf (a ++ [.inr c] ++ b)))[(a.filterMap Sum.getRight?).length + j + 1]? = _
      obtain ⟨g, gs, hg⟩ := List.exists_cons_of_ne_nil (FlockC3.Reps.cutOf_ne_nil (a ++ [.inr c] ++ b))
      rw [hg] at ih ⊢
      simpa only [PlaysCoin.consHead, List.getElem?_cons_succ] using ih
  | .inr _ :: a, j => by
      have ih := cutOf_shift c b a j
      change ([] :: cutOf (a ++ [.inr c] ++ b))[(a.filterMap Sum.getRight?).length + 1 + j + 1]? = _
      rw [List.getElem?_cons_succ, Nat.add_right_comm]
      exact ih

/-- **Round `c₀ + j` of `p ++ b` is round `j` of `b`** where `p` has `c₀` coins and ends in one. -/
theorem cutOf_shift_of {p b : List Step} {c₀ : ℕ} (hc : (p.filterMap Sum.getRight?).length = c₀)
    (he : ∃ (l : List Step) (c : Game.Msg), p = l ++ [.inr c]) (j : ℕ) :
    (cutOf (p ++ b))[c₀ + j]? = (cutOf b)[j]? := by
  obtain ⟨l, c, rfl⟩ := he
  have hl : (l.filterMap Sum.getRight?).length + 1 = c₀ := (coins_append l [.inr c]).symm.trans hc
  rw [show c₀ + j = (l.filterMap Sum.getRight?).length + j + 1 by omega]
  exact cutOf_shift c b l j

/-! ## The parts as stages -/

section run

variable {e : ExecHolo}

theorem sparsePart_split (u : C3Run e) : sparsePart u = repPart u ++ (u.sp 0 ++ u.sp 1) := by
  simp only [sparsePart, repPart, List.append_assoc]

theorem openPart_split (u : C3Run e) : openPart u = sparsePart u ++ u.op := by
  simp only [openPart, sparsePart, List.append_assoc]

theorem ligPart_split (u : C3Run e) : ligPart u = openPart u ++ u.lg := by
  simp only [ligPart, openPart, FlockC3.PubRows.stagesOf, List.append_assoc]

theorem coins_openSteps (s : Fin 2 → Fin 128 → GF128) (ρ : Fin 2 → Fin 7 → GF128) (γ : Fin 2 → GF128) :
    ((openSteps s ρ γ).filterMap Sum.getRight?).length = 3 := rfl

theorem cutOf_openSteps (s : Fin 2 → Fin 128 → GF128) (ρ : Fin 2 → Fin 7 → GF128) (γ : Fin 2 → GF128) :
    cutOf (openSteps s ρ γ) = [[⟨Fin 128 → GF128, s 0⟩], [⟨Fin 128 → GF128, s 1⟩], [], []] := rfl

/-- **R's opening's part ends in a coin** (the batching coins `γ`), three coins past the sparse phases' part. -/
theorem openPart_ends {u : C3Run e} {c₀ : ℕ} (hc : ((sparsePart u).filterMap Sum.getRight?).length = c₀)
    {s : Fin 2 → Fin 128 → GF128} {ρ : Fin 2 → Fin 7 → GF128} {γ : Fin 2 → GF128} (hop : u.op = openSteps s ρ γ) :
    ((openPart u).filterMap Sum.getRight?).length = c₀ + 3 ∧
      ∃ (l : List Step) (c : Game.Msg), openPart u = l ++ [.inr c] := by
  refine ⟨by rw [openPart_split, coins_append, hc, hop, coins_openSteps],
    sparsePart u ++ (openSteps s ρ γ).take 4, ⟨Fin 2 → GF128, γ⟩, ?_⟩
  rw [openPart_split, hop, List.append_assoc]
  rfl

end run

/-! ## The read facts from the stages' -/

section src

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **A part has `c₀` coins and ends in one** at every accepted outer proof. -/
def PartEnds (prev : C3Run (holoAt st hp) → List Step)
    (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)) (c₀ : ℕ) : Prop :=
  AtSrc zs dg N k Rw src fun _ u => ((prev u).filterMap Sum.getRight?).length = c₀ ∧
    ∃ (l : List Step) (c : Game.Msg), prev u = l ++ [.inr c]

/-- **A stage's committed rounds, past `c₀` coins** (`PartMsgsRead`'s form on the stage's steps): two accepted outer
proofs whose layers agree on the read wires of stage `c₀ + j`'s commit strings have the same round-`j` messages in the
stage, at a round that closes in it in both runs. -/
def StageMsgsRead (c₀ : ℕ) (stage : C3Run (holoAt st hp) → List Step)
    (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop) : Prop :=
  ∀ j R ω t R₂ τ o R' ω' t' R₂' τ' o', FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
    FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R' ω' t' R₂' τ' o' →
    maskOf Vs (skOf Vs own sel) (c₀ + j) (layers zs dg N k Rw R ω t R₂ τ) =
      maskOf Vs (skOf Vs own sel) (c₀ + j) (layers zs dg N k Rw R' ω' t' R₂' τ') →
    j < ((stage (src R ω t (layers zs dg N k Rw R ω t R₂ τ))).filterMap Sum.getRight?).length →
    j < ((stage (src R' ω' t' (layers zs dg N k Rw R' ω' t' R₂' τ'))).filterMap Sum.getRight?).length →
    (cutOf (stage (src R ω t (layers zs dg N k Rw R ω t R₂ τ))))[j]? =
      (cutOf (stage (src R' ω' t' (layers zs dg N k Rw R' ω' t' R₂' τ'))))[j]?

/-- **R's ring switches are read at their rounds** (named): two accepted outer proofs whose layers agree on the read
wires of stage `c₀ + i`'s commit strings hold the same ring switch `i < 2` at V*'s layers (`RsAt`), at the groups of
the opening's structure at `sh`. -/
def RsRead (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Flock.Shape)
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128)
    (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop) (c₀ : ℕ) : Prop :=
  ∀ i : ℕ, i < 2 → ∀ R ω t R₂ τ o R' ω' t' R₂' τ' o',
    FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
    FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R' ω' t' R₂' τ' o' →
    maskOf Vs (skOf Vs own sel) (c₀ + i) (layers zs dg N k Rw R ω t R₂ τ) =
      maskOf Vs (skOf Vs own sel) (c₀ + i) (layers zs dg N k Rw R' ω' t' R₂' τ') →
    ∀ (S0 : RecAlgebra.GfS) gs x x', RecOpening.structureOf sh = .ok S0 →
      Flock.RecResiduals.groups S0 (RecOpening.portSizes sh) = .ok gs →
      RsAt (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) sh gs (w R ω t (layers zs dg N k Rw R ω t R₂ τ)) i x →
      RsAt (zs R' ω' t' R₂') (layers zs dg N k Rw R' ω' t' R₂' τ') sh gs
        (w R' ω' t' (layers zs dg N k Rw R' ω' t' R₂' τ')) i x' → x = x'

/-- **A part's committed rounds are its stage's** where the part before it has `c₀` coins and ends in one. -/
theorem partMsgsRead_of_stage {prev part stage : C3Run (holoAt st hp) → List Step}
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
    (hsplit : ∀ u, part u = prev u ++ stage u) {c₀ : ℕ} (he : PartEnds prev zs dg N k Rw src c₀)
    (h : StageMsgsRead c₀ stage zs dg N k Rw src own sel) : PartMsgsRead prev part zs dg N k Rw src own sel := by
  intro i R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX hn hi hi'
  have hc : ((prev (src R ω t (layers zs dg N k Rw R ω t R₂ τ))).filterMap Sum.getRight?).length = c₀ :=
    (he R ω t R₂ τ o hA).1
  have hc' : ((prev (src R' ω' t' (layers zs dg N k Rw R' ω' t' R₂' τ'))).filterMap Sum.getRight?).length = c₀ :=
    (he R' ω' t' R₂' τ' o' hA').1
  have hle : c₀ ≤ i := Nat.le_of_not_lt fun hlt => hn ⟨by rw [hc]; exact hlt, by rw [hc']; exact hlt⟩
  obtain ⟨j, rfl⟩ : ∃ j, i = c₀ + j := ⟨i - c₀, by omega⟩
  rw [hsplit, coins_append, hc] at hi
  rw [hsplit, coins_append, hc'] at hi'
  rw [hsplit, hsplit, cutOf_shift_of hc (he R ω t R₂ τ o hA).2, cutOf_shift_of hc' (he R' ω' t' R₂' τ' o' hA').2]
  exact h j R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX (by omega) (by omega)

theorem sparseMsgsRead_of_stage
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop} {c₀ : ℕ}
    (he : PartEnds repPart zs dg N k Rw src c₀)
    (h : StageMsgsRead c₀ (fun u => u.sp 0 ++ u.sp 1) zs dg N k Rw src own sel) :
    SparseMsgsRead zs dg N k Rw src own sel :=
  partMsgsRead_of_stage sparsePart_split he h

theorem openMsgsRead_of_stage
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop} {c₀ : ℕ}
    (he : PartEnds sparsePart zs dg N k Rw src c₀)
    (h : StageMsgsRead c₀ C3Run.op zs dg N k Rw src own sel) : OpenMsgsRead zs dg N k Rw src own sel :=
  partMsgsRead_of_stage openPart_split he h

theorem ligMsgsRead_of_stage
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop} {c₀ : ℕ}
    (he : PartEnds openPart zs dg N k Rw src c₀)
    (h : StageMsgsRead c₀ C3Run.lg zs dg N k Rw src own sel) : LigMsgsRead zs dg N k Rw src own sel :=
  partMsgsRead_of_stage ligPart_split he h

/-- **R's opening's part ends in a coin at the source**, three coins past the sparse phases'. -/
theorem partEnds_open
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {c₀ : ℕ} (he : PartEnds sparsePart zs dg N k Rw src c₀)
    (hop : AtSrc zs dg N k Rw src fun _ u => ∃ s ρ γ, u.op = openSteps s ρ γ) :
    PartEnds openPart zs dg N k Rw src (c₀ + 3) := by
  intro R ω t R₂ τ o hA
  obtain ⟨s, ρ, γ, h⟩ := hop R ω t R₂ τ o hA
  exact openPart_ends (he R ω t R₂ τ o hA).1 h

/-- **R's opening's committed rounds from V*'s layers**: R's opening tied to V*'s layers (`OpenTieL`) holds its ring
switches at rounds `c₀` and `c₀ + 1` past the sparse phases' coins and nothing at `c₀ + 2`, so the ring switches being
read at their rounds (`RsRead`) is `OpenMsgsRead`. -/
theorem openMsgsRead_of_layer
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop} {sh : Flock.Shape}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128} {c₀ : ℕ}
    (he : PartEnds sparsePart zs dg N k Rw src c₀)
    (htie : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      OpenTieL (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) sh (w R ω t (layers zs dg N k Rw R ω t R₂ τ))
        (src R ω t (layers zs dg N k Rw R ω t R₂ τ)))
    (hrs : RsRead zs dg N k Rw sh w own sel c₀) : OpenMsgsRead zs dg N k Rw src own sel := by
  refine openMsgsRead_of_stage he ?_
  intro j R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX hj _
  obtain ⟨S0, gs, x, _, _, hS, hg, hx, hop⟩ := htie R ω t R₂ τ o hA
  obtain ⟨S0', gs', x', _, _, hS', hg', hx', hop'⟩ := htie R' ω' t' R₂' τ' o' hA'
  obtain rfl : S0' = S0 := Except.ok.inj (hS'.symm.trans hS)
  obtain rfl : gs' = gs := Except.ok.inj (hg'.symm.trans hg)
  rw [hop] at hj ⊢
  rw [hop', cutOf_openSteps, cutOf_openSteps]
  have hj3 : j < 3 := hj
  rcases j with _ | _ | _ | j
  · exact congrArg (fun y : Fin 128 → F128 => some [(⟨Fin 128 → GF128, fun t => fOf (y t)⟩ : Game.Msg)])
      (hrs 0 (by decide) R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX S0 gs (x 0) (x' 0) hS hg (hx 0) (hx' 0))
  · exact congrArg (fun y : Fin 128 → F128 => some [(⟨Fin 128 → GF128, fun t => fOf (y t)⟩ : Game.Msg)])
      (hrs 1 (by decide) R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX S0 gs (x 1) (x' 1) hS hg (hx 1) (hx' 1))
  · rfl
  · omega

end src

end

end FlockC3.Compose
