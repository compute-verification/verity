module

public import Proofs.Flock.C3.PubRows.Frame
public import Proofs.Flock.C3.Sparse.Read
public import Proofs.Flock.C3.Opening.Read
public import Proofs.Flock.C3.RegLevels.Read
public import Proofs.Flock.C3.HoloValues.Read
public import Proofs.Flock.VBridge.Family.Transcript
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

/-! The fixed decoder (`c3DecOf`), and the read facts at it. -/

@[expose] public section

namespace FlockC3.Compose

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockVBridge.Family (skOf wmsgOf maskOf coinSteps roundMsgs toSs roundMsgs_toSs)
open FlockC3.Sparse FlockC3.Opening FlockC3.RegLevels FlockC3.PubRows

def runSteps {e : ExecHolo} (u : C3Run e) : List Step :=
  holoSteps u.cap₀ u.pts (u.ts 0) (u.ts 1) (u.sp 0) (u.sp 1) u.op u.lg u.os

theorem framed_runSteps {e : ExecHolo} (u : C3Run e) : Framed u (runSteps u) := by
  rw [runSteps, holoSteps_eq]
  refine ⟨⟨_, rfl⟩, ⟨_, rfl⟩, ⟨u.os, ?_⟩⟩
  simp only [List.drop_succ_cons, List.drop_zero, List.drop_left]

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt T CT st hp

section decoder

variable {S : ℕ} {Vs : Fin S → VStmt}

def c3Msgs (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (l : (j : Fin S) → (Vs j).Bits) : List (List Game.Msg) :=
  PlaysCoin.cut ((runSteps (src R ω t l)).map Exec.kindOf) ((runSteps (src R ω t l)).filterMap Sum.getLeft?)

def AtSrc (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (P : List (Cm × (ℐ).Coin) → C3Run (holoAt st hp) → Prop) : Prop :=
  ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
    P t (src R ω t (layers zs dg N k Rw R ω t R₂ τ))

/-- V*'s sessions accept only at a whole history (G's `HLeaf`): false at a family whose sessions accept at a
short history, which the checked sessions refuse (`hLeafC3_checked`). -/
def HLeafC3 (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) : Prop :=
  ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → (ℐ).r ≤ t.length

open Classical in
theorem not_acceptsZKR_keepOwn {Q : VStmt} {z : Q.Outer Reg'}
    {P : FlockSoundness.Discharge.Exec.Inputs → Array Flock.Registered.Port →
      FlockSoundness.Discharge.Exec.Transcript → Prop} (hP : ∀ J own tr, ¬ P J own tr)
    (tr : FlockSoundness.Discharge.Exec.Transcript) :
    ¬ AcceptsZKR (Flock.Guarantees.keepOwn z P).I (Flock.Guarantees.keepOwn z P).own tr := by
  intro h
  have hf : (Flock.Guarantees.keepOwn z P).own = freshOwn Q.c := ite_eq_right fun ⟨_, _, _, _, hp⟩ => hP _ _ _ hp
  rw [hf] at h
  exact not_acceptsZKR_fresh z.I z.ht z.hTags z.hcH tr h

theorem hLeafC3_checked {ish : RecVStar.Inner} {cs : List (Cm × (ℐ).Coin) → Option Flock.RecVStar.InnerCommits}
    {coins : List (Cm × (ℐ).Coin) → Option Flock.RecCoins.Source} {bind : Fin S → VStar.Bindings}
    {rows : Option (Array Flock.Registered.Port)}
    {zr : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).OuterR Reg'}
    {hC : Flock.Premises.RecordCustody zr} {h : Typed zr} (hS : 0 < S) (dg : ByteArray → ByteArray) (N k Rw : ℕ) :
    HLeafC3 (Flock.Guarantees.checkedC3 ish cs coins bind rows zr hC h) dg N k Rw := by
  intro R ω t R₂ τ o hacc
  by_contra hlt
  exact not_acceptsZKR_keepOwn (z := Flock.Guarantees.custodied zr hC h R ω t R₂ ⟨0, hS⟩)
    (P := fun J own tr => (ℐ).r ≤ t.length ∧
      Flock.Guarantees.ChecksC3 ish (cs t) (coins t) (⟨0, hS⟩ : Fin S) (bind ⟨0, hS⟩) rows J own tr)
    (fun _ _ _ hp => hlt hp.1) _ (hacc.1 ⟨0, hS⟩).1

def MsgsReadC3 (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop) : Prop :=
  ∀ i R ω t R₂ τ o R' ω' t' R₂' τ' o', FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
    FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R' ω' t' R₂' τ' o' →
    maskOf Vs (skLeaf Vs own sel (ℐ).r) i (layers zs dg N k Rw R ω t R₂ τ) =
      maskOf Vs (skLeaf Vs own sel (ℐ).r) i (layers zs dg N k Rw R' ω' t' R₂' τ') →
    (c3Msgs src R ω t (layers zs dg N k Rw R ω t R₂ τ))[i]? =
      (c3Msgs src R' ω' t' (layers zs dg N k Rw R' ω' t' R₂' τ'))[i]?

def MsgTysC3 (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)) : Prop :=
  ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
    ∀ g ∈ c3Msgs src R ω t (layers zs dg N k Rw R ω t R₂ τ), ∀ m ∈ g, ∃ j, T j = m.1

open Classical in
def c3DecOf (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)
    (i : ℕ) (Xs : (j : Fin S) → (Vs j).Bits) : (ℐ).Msg :=
  if h : ∃ g : List Game.Msg, ∃ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o ∧
      maskOf Vs (skLeaf Vs own sel (ℐ).r) i (layers zs dg N k Rw R ω t R₂ τ) = Xs ∧
      (c3Msgs src R ω t (layers zs dg N k Rw R ω t R₂ τ))[i]?.getD [] = g
  then toSs T (Classical.choose h) else []

theorem c3DecOf_spec {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
    (hread : MsgsReadC3 zs dg N k Rw src own sel) (htys : MsgTysC3 zs dg N k Rw src)
    {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (h : FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o) (i : ℕ) :
    roundMsgs T
        (c3DecOf zs dg N k Rw src own sel i (maskOf Vs (skLeaf Vs own sel (ℐ).r) i (layers zs dg N k Rw R ω t R₂ τ))) =
      (c3Msgs src R ω t (layers zs dg N k Rw R ω t R₂ τ))[i]?.getD [] := by
  have hex : ∃ g : List Game.Msg, ∃ R' ω' t' R₂' τ' o',
      FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R' ω' t' R₂' τ' o' ∧
      maskOf Vs (skLeaf Vs own sel (ℐ).r) i (layers zs dg N k Rw R' ω' t' R₂' τ') =
        maskOf Vs (skLeaf Vs own sel (ℐ).r) i (layers zs dg N k Rw R ω t R₂ τ) ∧
      (c3Msgs src R' ω' t' (layers zs dg N k Rw R' ω' t' R₂' τ'))[i]?.getD [] = g :=
    ⟨_, R, ω, t, R₂, τ, o, h, rfl, rfl⟩
  have hd :
      c3DecOf zs dg N k Rw src own sel i (maskOf Vs (skLeaf Vs own sel (ℐ).r) i (layers zs dg N k Rw R ω t R₂ τ)) =
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

def SrcOK (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop) : Prop :=
  HLeafC3 zs dg N k Rw ∧
    AtSrc zs dg N k Rw src (fun t u => (runSteps u).filterMap Sum.getRight? = coinSteps CT 0 (t.map Prod.snd)) ∧
    MsgsReadC3 zs dg N k Rw src own sel ∧ MsgTysC3 zs dg N k Rw src ∧
    AtSrc zs dg N k Rw src fun _ u => decodeRun (holoAt st hp) (runSteps u) = u

theorem tape_c3DecOf {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
    (hs : SrcOK zs dg N k Rw src own sel) {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (h : FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o) :
    schedSteps T CT 0 ((wmsgOf Vs (skLeaf Vs own sel (ℐ).r) (ℐ).r (c3DecOf zs dg N k Rw src own sel)
      (layers zs dg N k Rw R ω t R₂ τ)).zip (t.map Prod.snd)) =
      runSteps (src R ω t (layers zs dg N k Rw R ω t R₂ τ)) := by
  obtain ⟨hleaf, hcoin, hread, htys, -⟩ := hs
  have hr : (ℐ).r ≤ t.length := hleaf R ω t R₂ τ o h
  have hc : (runSteps (src R ω t (layers zs dg N k Rw R ω t R₂ τ))).filterMap Sum.getRight? =
      coinSteps CT 0 (t.map Prod.snd) := hcoin R ω t R₂ τ o h
  have key : ∀ (ts : List Step) (dec : ℕ → List (SMsg T)) (cs : List ((k : Fin r) → CT k)), r + 1 ≤ cs.length →
      (∀ i, roundMsgs T (dec i) = (PlaysCoin.cut (ts.map Exec.kindOf) (ts.filterMap Sum.getLeft?))[i]?.getD []) →
      coinSteps CT 0 cs = ts.filterMap Sum.getRight? →
      schedSteps T CT 0 (((List.range (r + 1)).map dec).zip cs) = ts := by
    intro ts dec cs hcs hm hc'
    have hlen : (PlaysCoin.cut (ts.map Exec.kindOf) (ts.filterMap Sum.getLeft?)).length = r + 1 := by
      rw [FlockVBridge.Family.length_cut, ← hc', FlockVBridge.Family.length_coinSteps CT (Nat.le_of_succ_le hcs)]
    refine FlockVBridge.Family.schedSteps_zip T CT ts _ cs (by rw [List.length_map, List.length_range]; exact hcs)
      ?_ hc'
    refine List.ext_getElem (by rw [List.length_map, List.length_map, List.length_range, hlen]) fun i _ hi => ?_
    rw [List.getElem_map, List.getElem_map, List.getElem_range, hm i, List.getElem?_eq_getElem hi, Option.getD_some]
  exact key (runSteps (src R ω t (layers zs dg N k Rw R ω t R₂ τ)))
    (fun i =>
      c3DecOf zs dg N k Rw src own sel i (maskOf Vs (skLeaf Vs own sel (ℐ).r) i (layers zs dg N k Rw R ω t R₂ τ)))
    (t.map Prod.snd) (Nat.le_trans hr (Nat.le_of_eq (List.length_map _).symm))
    (fun i => c3DecOf_spec hread htys h i) hc.symm

end decoder

def OpeningReadAt (l : List Step) (u : C3Run (holoAt st hp)) : Prop :=
  ∃ (sh : Flock.Shape) (M : Flock.RecAlgebra.Messages F128) (a : Fin 2 → Flock.RecAlgebra.AClaim F128)
    (cs : Flock.RecAlgebra.RoundCoins) (k₀ k₁ : ℕ) (res : Array F128) (ρ : Fin 2 → Fin 7 → GF128) (γ : Fin 2 → GF128),
    Flock.RecOpening.shape? (holoAt st hp).cls.shape.mReg 2 = some sh ∧
    Flock.RecOpening.run sh M #[a 0, a 1] k₀ cs = .ok (res, k₁) ∧ (∀ x ∈ res, x = 0) ∧
    (∀ i, FlockSoundness.Refine.ClaimRel (holoAt st hp).cls.shape.mReg (a i).toClaim (u.claim i).toClaim) ∧
    openSteps (fun i => FlockSoundness.Refine.vecOf 128 M.rs[i.val]!) ρ γ <+: afterSparse u l

section honest

variable {ish : RecVStar.Inner} {ss : Array VStar.Stmt} {hss : RecVStar.statements ish = .ok ss}
  {bind : Fin (vstarCount ish) → VStar.Bindings} {Vs : Fin (vstarCount ish) → VStmt}
  {hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs}
  {zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
    (Vs j).Outer Reg'}
  {own : Fin (vstarCount ish) → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
  {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
  {wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String} {reg : Lean.Json}
  {dg : ByteArray → ByteArray} {N k Rw : ℕ}
  {src : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) →
    C3Run (holoAt st hp)}

local notation "𝒵" => zs
local notation "𝓓" => C3Family.honest (xR := xR) ish hss bind hob 𝒵 own sel bc
  (c3DecOf 𝒵 dg N k Rw src own sel) wr reg

theorem at_c3DecOf {P : List Step → C3Run (holoAt st hp) → Prop} (hs : SrcOK 𝒵 dg N k Rw src own sel)
    (hP : AtSrc 𝒵 dg N k Rw src fun _ u => P (runSteps u) u) :
    (𝓓).AtAccepted dg N k Rw fun R ω t Lb =>
      P (schedSteps T CT 0 (((𝓓).wmsg Lb).zip (t.map Prod.snd))) ((𝓓).run R ω t Lb) := by
  intro R ω t R₂ τ o hacc
  have e := tape_c3DecOf hs hacc
  obtain ⟨-, -, -, -, hdec⟩ := hs
  have hd : decodeRun (holoAt st hp) (runSteps (src R ω t (layers 𝒵 dg N k Rw R ω t R₂ τ))) =
      src R ω t (layers 𝒵 dg N k Rw R ω t R₂ τ) := hdec R ω t R₂ τ o hacc
  show P (schedSteps T CT 0 ((wmsgOf Vs (skLeaf Vs own sel (ℐ).r) (ℐ).r (c3DecOf 𝒵 dg N k Rw src own sel)
      (layers 𝒵 dg N k Rw R ω t R₂ τ)).zip (t.map Prod.snd)))
    (decodeRun (holoAt st hp) (schedSteps T CT 0 ((wmsgOf Vs (skLeaf Vs own sel (ℐ).r) (ℐ).r
      (c3DecOf 𝒵 dg N k Rw src own sel) (layers 𝒵 dg N k Rw R ω t R₂ τ)).zip (t.map Prod.snd))))
  rw [e, hd]
  exact hP R ω t R₂ τ o hacc

theorem hFrame_c3DecOf (hs : SrcOK 𝒵 dg N k Rw src own sel) : HFrame 𝓓 dg N k Rw :=
  at_c3DecOf (P := fun l u => Framed u l) hs fun _ _ _ _ _ _ _ => framed_runSteps _

theorem repsRead_c3DecOf (hs : SrcOK 𝒵 dg N k Rw src own sel)
    (hrp : AtSrc 𝒵 dg N k Rw src fun _ u => FlockC3.HoloValues.RepsReadAt (holoAt st hp) (runSteps u) u) :
    FlockC3.HoloValues.RepsRead 𝓓 dg N k Rw :=
  at_c3DecOf (P := FlockC3.HoloValues.RepsReadAt (holoAt st hp)) hs hrp

theorem sparseRead_c3DecOf (hs : SrcOK 𝒵 dg N k Rw src own sel)
    (hsp : AtSrc 𝒵 dg N k Rw src fun _ u => SparseReadAt (holoAt st hp) (runSteps u) u) :
    SparseRead 𝓓 dg N k Rw :=
  at_c3DecOf (P := SparseReadAt (holoAt st hp)) hs hsp

theorem openingRead_c3DecOf (hs : SrcOK 𝒵 dg N k Rw src own sel)
    (hop : AtSrc 𝒵 dg N k Rw src fun _ u => OpeningReadAt (runSteps u) u) : OpeningRead 𝓓 dg N k Rw :=
  at_c3DecOf (P := OpeningReadAt) hs hop

theorem regLevelsRead_c3DecOf (hs : SrcOK 𝒵 dg N k Rw src own sel)
    (hrl : AtSrc 𝒵 dg N k Rw src fun _ u =>
      RegTape xR.cap (holoAt st hp).rsch.sch u.opened (tapeR (runSteps u) u)) :
    RegLevelsRead 𝓓 dg N k Rw :=
  at_c3DecOf (P := fun l u => RegTape xR.cap (holoAt st hp).rsch.sch u.opened (tapeR l u)) hs hrl

end honest

end

end FlockC3.Compose
