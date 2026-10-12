import Proofs.Flock.C3.PubRows.Frame
import Proofs.Flock.C3.Sparse.Read
import Proofs.Flock.C3.Opening.Read
import Proofs.Flock.C3.RegLevels.Read
import Proofs.Flock.VBridge.Family.Transcript

/-!
# n6: the fixed decoder, and the read facts at it

`c3DecOf` is G's `decOf` at candidate 3's table: round `i`'s message at the read wires of stage `i`'s commit strings is
round `i` of the messages the source's run holds (`c3Msgs`: its steps' messages cut at their coins, as G's `tableMsgs`
is), at any accepted outer proof whose layers read those wires there. The source `src` is the run V*'s layers hold,
stage by stage, a parameter as G's `cap` and `lig` are. Where round `i`'s messages are read from stage `i`'s commit
strings (`MsgsReadC3`, G's `MsgsRead`) and typed by `T` (`MsgTysC3`), the decoder reads them (`c3DecOf_spec`).

At the honest family at that decoder (`𝓓`) the inner transcript is the source's steps (`tape_c3DecOf`), where the
source's obligations hold (`SrcOK`): the history is whole (`HLeafC3`), the source's coins are the history's, its
messages are read and typed as above, and its run is the decode of its steps. So each read fact at `𝓓` is that read
fact of the source's run on its own steps (`at_c3DecOf`): `hFrame_c3DecOf` (which needs nothing more),
`sparseRead_c3DecOf`, `openingRead_c3DecOf`, `regLevelsRead_c3DecOf`.

No decoder meets a read fact or `HFrame` at a history shorter than the inner protocol's rounds: `zip` stops at the
last coin, so the tape is a prefix of the whole one (at `t = []`, empty), and V*'s sessions can still accept there.
That is G's `HLeaf`, false as `VBridge` is quantified; `HLeafC3` holds once `VBridge` is read at whole histories only,
as `recGame` reads it, or the family's sessions refuse a short history.
-/

namespace FlockC3.Compose

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockVBridge.Family (skOf wmsgOf maskOf coinSteps roundMsgs toSs roundMsgs_toSs)
open FlockC3.Sparse FlockC3.Opening FlockC3.RegLevels FlockC3.PubRows

/-- **A run's steps**, in the chain's order (`holoSteps`). -/
def runSteps {e : ExecHolo} (u : C3Run e) : List Step :=
  holoSteps u.cap₀ u.pts (u.ts 0) (u.ts 1) (u.sp 0) (u.sp 1) u.op u.lg u.os

/-- **A run is framed on its own steps.** -/
theorem framed_runSteps {e : ExecHolo} (u : C3Run e) : Framed u (runSteps u) := by
  rw [runSteps, holoSteps_eq]
  refine ⟨⟨_, rfl⟩, ⟨_, rfl⟩, ⟨u.os, ?_⟩⟩
  simp only [List.drop_succ_cons, List.drop_zero, List.drop_left]

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt T CT st hp

section decoder

variable {S : ℕ} {Vs : Fin S → VStmt}
  (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
  (dg : ByteArray → ByteArray) (N k Rw : ℕ)
  (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
  (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)

/-- **The inner protocol's messages, round by round**, at a history and V*'s layers: the source's run's steps'
messages cut at their coins (`PlaysCoin.cut`). -/
def c3Msgs (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (l : (j : Fin S) → (Vs j).Bits) : List (List Game.Msg) :=
  PlaysCoin.cut ((runSteps (src R ω t l)).map Exec.kindOf) ((runSteps (src R ω t l)).filterMap Sum.getLeft?)

/-- **At every accepted outer proof**, at the history and the source's run there. -/
def AtSrc (P : List (Cm × (ℐ).Coin) → C3Run (holoAt st hp) → Prop) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → P t (src R ω t (layers zs dg N k Rw R ω t R₂ τ))

/-- Obligation: false as `VBridge` is quantified (G's `HLeaf`); discharged once `VBridge` is stated at whole histories,
as `recGame` reads it, or the family's sessions refuse a short history.
**V*'s sessions accept only at a whole history**: at every accepted outer proof, the history has a coin for each of
the inner protocol's rounds. -/
def HLeafC3 : Prop := ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → (ℐ).r ≤ t.length

/-- **The table's round-`i` messages are read from stage `i`'s commit strings** (G's `MsgsRead`): two accepted outer
proofs whose layers agree on the read wires of stage `i`'s commit strings (`maskOf` at `skOf own sel`) have the same
round-`i` messages (`c3Msgs`). -/
def MsgsReadC3 : Prop :=
  ∀ i R ω t R₂ τ o R' ω' t' R₂' τ' o', Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
    Accepted (I := ℐ) zs dg N k Rw R' ω' t' R₂' τ' o' →
    maskOf Vs (skOf Vs own sel) i (layers zs dg N k Rw R ω t R₂ τ) =
      maskOf Vs (skOf Vs own sel) i (layers zs dg N k Rw R' ω' t' R₂' τ') →
    (c3Msgs src R ω t (layers zs dg N k Rw R ω t R₂ τ))[i]? =
      (c3Msgs src R' ω' t' (layers zs dg N k Rw R' ω' t' R₂' τ'))[i]?

/-- **The inner protocol's message types are V*'s**: at every accepted outer proof, each of the source's messages has
its type among the schedule's message types `T`. -/
def MsgTysC3 : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
    ∀ g ∈ c3Msgs src R ω t (layers zs dg N k Rw R ω t R₂ τ), ∀ m ∈ g, ∃ j, T j = m.1

open Classical in
/-- **The decoder**: round `i`'s message at the read wires `Xs` of stage `i`'s commit strings is the source's round-`i`
messages at any accepted outer proof whose layers read `Xs` there, typed by `T` (`toSs`), and none elsewhere. -/
def c3DecOf (i : ℕ) (Xs : (j : Fin S) → (Vs j).Bits) : (ℐ).Msg :=
  if h : ∃ g : List Game.Msg, ∃ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o ∧
      maskOf Vs (skOf Vs own sel) i (layers zs dg N k Rw R ω t R₂ τ) = Xs ∧
      (c3Msgs src R ω t (layers zs dg N k Rw R ω t R₂ τ))[i]?.getD [] = g
  then toSs T (Classical.choose h) else []

/-- **The decoder reads round `i`**, at an accepted outer proof: its message at the layers' stage-`i` commit strings
is the source's round `i` (`MsgsReadC3`, `MsgTysC3`). -/
theorem c3DecOf_spec (hread : MsgsReadC3 zs dg N k Rw src own sel) (htys : MsgTysC3 zs dg N k Rw src)
    {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (h : Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o) (i : ℕ) :
    roundMsgs T (c3DecOf zs dg N k Rw src own sel i (maskOf Vs (skOf Vs own sel) i (layers zs dg N k Rw R ω t R₂ τ))) =
      (c3Msgs src R ω t (layers zs dg N k Rw R ω t R₂ τ))[i]?.getD [] := by
  have hex : ∃ g : List Game.Msg, ∃ R' ω' t' R₂' τ' o', Accepted (I := ℐ) zs dg N k Rw R' ω' t' R₂' τ' o' ∧
      maskOf Vs (skOf Vs own sel) i (layers zs dg N k Rw R' ω' t' R₂' τ') =
        maskOf Vs (skOf Vs own sel) i (layers zs dg N k Rw R ω t R₂ τ) ∧
      (c3Msgs src R' ω' t' (layers zs dg N k Rw R' ω' t' R₂' τ'))[i]?.getD [] = g :=
    ⟨_, R, ω, t, R₂, τ, o, h, rfl, rfl⟩
  have hd : c3DecOf zs dg N k Rw src own sel i (maskOf Vs (skOf Vs own sel) i (layers zs dg N k Rw R ω t R₂ τ)) =
      toSs T (Classical.choose hex) :=
    dif_pos hex
  rw [hd]
  obtain ⟨R', ω', t', R₂', τ', o', h', hX', hg'⟩ := Classical.choose_spec hex
  have e := hread i R ω t R₂ τ o R' ω' t' R₂' τ' o' h h' hX'.symm
  have hc : Classical.choose hex = (c3Msgs src R ω t (layers zs dg N k Rw R ω t R₂ τ))[i]?.getD [] := by
    rw [← hg', ← e]
  rw [hc]
  apply roundMsgs_toSs
  intro m hm
  cases hti : (c3Msgs src R ω t (layers zs dg N k Rw R ω t R₂ τ))[i]? with
  | none => rw [hti] at hm; simp at hm
  | some g =>
      rw [hti, Option.getD_some] at hm
      exact htys R ω t R₂ τ o h g (List.mem_of_getElem? hti) m hm

/-- **The source's obligations**: the history is whole (`HLeafC3`), the source's coin steps are the history's
(`coinSteps`), its round-`i` messages are read from stage `i`'s commit strings and typed by `T`, and its run is the
decode of its steps (`decodeRun`, as it is where each stage plays on its steps: `stageOf_of_playsT`). -/
def SrcOK : Prop :=
  HLeafC3 zs dg N k Rw ∧
    AtSrc zs dg N k Rw src (fun t u => (runSteps u).filterMap Sum.getRight? = coinSteps CT 0 (t.map Prod.snd)) ∧
    MsgsReadC3 zs dg N k Rw src own sel ∧ MsgTysC3 zs dg N k Rw src ∧
    AtSrc zs dg N k Rw src fun _ u => decodeRun (holoAt st hp) (runSteps u) = u

/-- Open (n6's shared piece).
**At the fixed decoder the inner transcript is the source's run's steps**, at every accepted outer proof where the
source's obligations hold: the decoder's round `i` is the source's round `i` (`c3DecOf_spec`), the history has a coin
for each round but the last and they are the source's coin steps, so the schedule weaves the source's messages with
its coins as its steps do (`schedSteps`, `PlaysCoin.cut`). -/
theorem tape_c3DecOf (hs : SrcOK zs dg N k Rw src own sel) {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)}
    {R₂ : Reg₂} {τ o} (h : Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o) :
    schedSteps T CT 0 ((wmsgOf Vs (skOf Vs own sel) (ℐ).r (c3DecOf zs dg N k Rw src own sel)
      (layers zs dg N k Rw R ω t R₂ τ)).zip (t.map Prod.snd)) =
      runSteps (src R ω t (layers zs dg N k Rw R ω t R₂ τ)) := by
  sorry

end decoder

/-- R's opening's read fact at a tape `l` and a run `u`, as `OpeningRead` reads them. -/
def OpeningReadAt (l : List Step) (u : C3Run (holoAt st hp)) : Prop :=
  ∃ (sh : Flock.Shape) (M : Flock.RecAlgebra.Messages F128) (a : Fin 2 → Flock.RecAlgebra.AClaim F128)
    (cs : Flock.RecAlgebra.Coins) (k₀ k₁ : ℕ) (res : Array F128) (ρ : Fin 2 → Fin 7 → GF128) (γ : Fin 2 → GF128),
    Flock.RecOpening.shape? (holoAt st hp).cls.shape.mReg 2 = some sh ∧
    Flock.RecOpening.run sh M #[a 0, a 1] k₀ cs = .ok (res, k₁) ∧ (∀ x ∈ res, x = 0) ∧
    (∀ i, FlockSoundness.Refine.ClaimRel (holoAt st hp).cls.shape.mReg (a i).toClaim (u.claim i).toClaim) ∧
    openSteps (fun i => FlockSoundness.Refine.vecOf 128 M.rs[i.val]!) ρ γ <+: afterSparse u l

section honest

variable {ish : RecVStar.Inner} {ss : Array VStar.Stmt} {hss : RecVStar.statements ish = .ok ss}
  {bind : Fin (vstarCount ish) → VStar.Bindings} {Vs : Fin (vstarCount ish) → VStmt}
  {hob : VStarParses Flock.Tags.circuit ish bind Vs}
  {zr : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin (vstarCount ish)) → (Vs j).OuterR Reg'}
  {hCust : Flock.Premises.RecordCustody zr} {h : Typed zr}
  {own : Fin (vstarCount ish) → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
  {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
  {tr : Lean.Json} {record : String} {reg : Lean.Json} {dg : ByteArray → ByteArray} {N k Rw : ℕ}
  {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → C3Run (holoAt st hp)}

local notation "𝒵" => Flock.Guarantees.custodied zr hCust h
local notation "𝓓" => C3Family.honest (xR := xR) ish hss bind hob zr hCust h own sel bc
  (c3DecOf 𝒵 dg N k Rw src own sel) tr record reg

/-- **A fact of the tape and the run, at the fixed decoder**, from that fact of the source's run on its own steps: the
tape is the source's steps (`tape_c3DecOf`) and the honest family's run their decode, which is the source's run. -/
theorem at_c3DecOf {P : List Step → C3Run (holoAt st hp) → Prop} (hs : SrcOK 𝒵 dg N k Rw src own sel)
    (hP : AtSrc 𝒵 dg N k Rw src fun _ u => P (runSteps u) u) :
    (𝓓).AtAccepted dg N k Rw fun R ω t Lb =>
      P (schedSteps T CT 0 (((𝓓).wmsg Lb).zip (t.map Prod.snd))) ((𝓓).run R ω t Lb) := by
  intro R ω t R₂ τ o hacc
  have e := tape_c3DecOf 𝒵 dg N k Rw src own sel hs hacc
  obtain ⟨-, -, -, -, hdec⟩ := hs
  have hd : decodeRun (holoAt st hp) (runSteps (src R ω t (layers 𝒵 dg N k Rw R ω t R₂ τ))) =
      src R ω t (layers 𝒵 dg N k Rw R ω t R₂ τ) := hdec R ω t R₂ τ o hacc
  show P (schedSteps T CT 0 ((wmsgOf Vs (skOf Vs own sel) (ℐ).r (c3DecOf 𝒵 dg N k Rw src own sel)
      (layers 𝒵 dg N k Rw R ω t R₂ τ)).zip (t.map Prod.snd)))
    (decodeRun (holoAt st hp) (schedSteps T CT 0 ((wmsgOf Vs (skOf Vs own sel) (ℐ).r
      (c3DecOf 𝒵 dg N k Rw src own sel) (layers 𝒵 dg N k Rw R ω t R₂ τ)).zip (t.map Prod.snd))))
  rw [e, hd]
  exact hP R ω t R₂ τ o hacc

/-- **`HFrame` at the fixed decoder.** -/
theorem hFrame_c3DecOf (hs : SrcOK 𝒵 dg N k Rw src own sel) : HFrame 𝓓 dg N k Rw :=
  at_c3DecOf (P := fun l u => Framed u l) hs fun _ _ _ _ _ _ _ => framed_runSteps _

/-- **`SparseRead` at the fixed decoder**, from the source's sparse phases on its own steps. -/
theorem sparseRead_c3DecOf (hs : SrcOK 𝒵 dg N k Rw src own sel)
    (hsp : AtSrc 𝒵 dg N k Rw src fun _ u => SparseReadAt (holoAt st hp) (runSteps u) u) :
    SparseRead 𝓓 dg N k Rw :=
  at_c3DecOf (P := SparseReadAt (holoAt st hp)) hs hsp

/-- **`OpeningRead` at the fixed decoder**, from the source's opening on its own steps. -/
theorem openingRead_c3DecOf (hs : SrcOK 𝒵 dg N k Rw src own sel)
    (hop : AtSrc 𝒵 dg N k Rw src fun _ u => OpeningReadAt (runSteps u) u) : OpeningRead 𝓓 dg N k Rw :=
  at_c3DecOf (P := OpeningReadAt) hs hop

/-- **`RegLevelsRead` at the fixed decoder**, from the source's Ligerito on its own steps, against `xR`'s cap: where
the source is read from V*'s layers, that cap is the sessions' registered top, which `_htop` and `HPubRows` give. -/
theorem regLevelsRead_c3DecOf (hs : SrcOK 𝒵 dg N k Rw src own sel)
    (hrl : AtSrc 𝒵 dg N k Rw src fun _ u =>
      RegTape xR.cap (holoAt st hp).rsch.sch u.opened (tapeR (runSteps u) u)) :
    RegLevelsRead 𝓓 dg N k Rw :=
  at_c3DecOf (P := fun l u => RegTape xR.cap (holoAt st hp).rsch.sch u.opened (tapeR l u)) hs hrl

end honest

end

end FlockC3.Compose
