module

public import Proofs.Flock.VBridge.Family.Openings
public import Proofs.Flock.VBridge.Family.ReadTransfer
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
import Proofs.Flock.VBridge.Algebra.BitsOk

/-! The family's commit strings, digests and messages (`skOf`, `cmtOf`, `wmsgOf`, `msgOf`) and the obligations
`HCommits`, `VDecodesRead` and `VFullRead` on them. -/

@[expose] public section

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages

theorem vb_cs_inl {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit}
    {hscope : Layout.scopeOk c = true} {Lv : SamplingDesign nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv) (R' : Reg')
    (u : Fin nv) (q : Fin c.ports.size) :
    zo.vb.cs R' (Sum.inl (u, q) : PosK c hU nv) =
      ((Sum.inl (u, q) : PosK c hU nv),
        Refine.H512 (Refine.K512.leafPrefix ++ (zo.pub.digest u.val q.val c.ports.size).data.toList)) :=
  rfl

theorem unit_of_io {c : Flock.Circuit} {hU : Built c.unit} {nv : ℕ}
    {g : Fin ((Placed.keyProg c hU nv).circuit (Layout.outsOf c hU nv)).N} {u u' : Fin nv} {q : Fin c.ports.size}
    (hg : posK c hU nv g = .inl (u, q))
    (hio : g ∈ ((Placed.keyProg c hU nv).partition (Layout.outsOf c hU nv)).io u') : u' = u := by
  unfold posK at hg
  rcases h : HiddenExec.rowOfH c hU nv (Layout.outsOf c hU nv) g with _ | ⟨u₀, q₀, k⟩ <;> rw [h] at hg
  · simp at hg
  · simp only [Sum.inl.injEq, Prod.mk.injEq] at hg
    obtain ⟨rfl, -⟩ := hg
    obtain ⟨x, rfl⟩ :=
      (Const.Prog.instC (Placed.keyProg c hU nv) (Layout.outsOf c hU nv) u').exists_wire_of_mem_io hio
    exact (HiddenExec.rowOfH_wire_unit (Layout.outsOf c hU nv) x h).symm

section defs

variable {S : ℕ} (Vs : Fin S → VStmt)

open Classical in
noncomputable def skOf (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop) (i : ℕ) (j : Fin S) :
    Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv) :=
  Finset.univ.filter fun
    | .inl (u, q) => ∃ p ∈ own j, p.name = (Vs j).c.ports[q].name ∧
        ∃ x, p.reads[u.val]? = some (some x) ∧ sel i p.value x
    | .inr _ => False

noncomputable def cmtOf {Cm : Type} (bc : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8)
    (j : Fin S) (e : Cm) (pos : PosK (Vs j).c (Vs j).hU (Vs j).nv) : (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg :=
  (pos, Refine.H512 (Refine.K512.leafPrefix ++ bc j e pos))

variable (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))

open Classical in
noncomputable def maskOf (i : ℕ) (Xs : (j : Fin S) → (Vs j).Bits) : (j : Fin S) → (Vs j).Bits :=
  fun j g => if posK (Vs j).c (Vs j).hU (Vs j).nv g ∈ Sk i j ∧ ReadW Vs j g then Xs j g else false

open Classical in
noncomputable def layerOf (i : ℕ) (v : (j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) :
    (j : Fin S) → (Vs j).Bits :=
  fun j g => if posK (Vs j).c (Vs j).hU (Vs j).nv g ∈ Sk i j ∧ ReadW Vs j g then
    bitK (Vs j).c (Vs j).hU (Vs j).nv (v j (posK (Vs j).c (Vs j).hU (Vs j).nv g)) g else false

noncomputable def wmsgOf {Msg : Type} (r : ℕ) (dec : ℕ → ((j : Fin S) → (Vs j).Bits) → Msg)
    (Xs : (j : Fin S) → (Vs j).Bits) : List Msg :=
  (List.range r).map fun i => dec i (maskOf Vs Sk i Xs)

noncomputable def msgOf {Msg : Type} (dec : ℕ → ((j : Fin S) → (Vs j).Bits) → Msg) (i : ℕ)
    (v : (j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) : Msg :=
  dec i (layerOf Vs Sk i v)

variable {Vs}

theorem mem_skOf_inl {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop} {i : ℕ} {j : Fin S}
    {u : Fin (Vs j).nv} {q : Fin (Vs j).c.ports.size} :
    Sum.inl (u, q) ∈ skOf Vs own sel i j ↔
      ∃ p ∈ own j, p.name = (Vs j).c.ports[q].name ∧ ∃ x, p.reads[u.val]? = some (some x) ∧ sel i p.value x := by
  simp only [skOf, Finset.mem_filter, Finset.mem_univ, true_and]

theorem inr_not_mem_skOf {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop} {i : ℕ}
    {j : Fin S} (g : Fin ((Placed.keyProg (Vs j).c (Vs j).hU (Vs j).nv).circuit
      (Layout.outsOf (Vs j).c (Vs j).hU (Vs j).nv)).N) :
    Sum.inr g ∉ skOf Vs own sel i j := by
  simp only [skOf, Finset.mem_filter, Finset.mem_univ, true_and, not_false_eq_true]

end defs

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂)

/-- The verifier's `--registered` check ties those rows to the entry `rec_residuals.commits_entry` rebuilds only on
an accepted run, which is where `VCommits` reads it. -/
def PubCommits (bc : (j : Fin F.stmts.size) → Cm → PosK (F.Vs j).c (F.Vs j).hU (F.Vs j).nv → List UInt8) : Prop :=
  ∀ i R ω (t : List (Cm × (ℐ).Coin)) (e : Cm × (ℐ).Coin), t[i]? = some e → ∀ R₂ j,
    (∃ tr, ZkReg.AcceptsZKR (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own tr) → ∀ (u : Fin (F.Vs j).nv)
    (q : Fin (F.Vs j).c.ports.size), Sum.inl (u, q) ∈ F.Sk i j →
      ((F.zs R ω t R₂ j).pub.digest u.val q.val (F.Vs j).c.ports.size).data.toList = bc j e.1 (.inl (u, q))

theorem hCommits_of (own : Fin F.stmts.size → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)
    (bc : (j : Fin F.stmts.size) → Cm → PosK (F.Vs j).c (F.Vs j).hU (F.Vs j).nv → List UInt8)
    (hSk : F.Sk = skOf F.Vs own sel) (hcmt : F.cmt = cmtOf F.Vs bc) (hpub : F.PubCommits bc) : F.HCommits := by
  intro i R ω t e he R₂ j R' tr _ hacc pos hpos
  rcases pos with ⟨u, q⟩ | g
  · rw [hcmt]
    exact (vb_cs_inl (F.zs R ω t R₂ j) R' u q).trans
      (congrArg (fun l => ((Sum.inl (u, q) : PosK (F.Vs j).c (F.Vs j).hU (F.Vs j).nv),
        Refine.H512 (Refine.K512.leafPrefix ++ l))) (hpub i R ω t e he R₂ j ⟨tr, hacc⟩ u q hpos))
  · rw [hSk] at hpos
    exact absurd hpos (inr_not_mem_skOf g)

def VDecodesRead : Prop := VDecodesR F.wmsg (fun _ _ a b => a = b) F.Sk F.msg

theorem vDecodesRead_of (dec : ℕ → ((j : Fin F.stmts.size) → (F.Vs j).Bits) → (ℐ).Msg)
    (hw : F.wmsg = wmsgOf F.Vs F.Sk (ℐ).r dec) (hm : F.msg = msgOf F.Vs F.Sk dec) : F.VDecodesRead := by
  intro i hi _ Xt v hX
  rw [hw, hm]
  simp only [wmsgOf, List.getElem?_map, List.getElem?_range hi, Option.map_some]
  refine Option.Rel.some ?_
  show dec i (maskOf F.Vs F.Sk i Xt) = dec i (layerOf F.Vs F.Sk i v)
  refine congrArg (dec i) (funext fun j => funext fun g => ?_)
  simp only [maskOf, layerOf]
  split_ifs with h
  · exact hX j g h.1 h.2
  · rfl

theorem hDecodes_of (dec : ℕ → ((j : Fin F.stmts.size) → (F.Vs j).Bits) → (ℐ).Msg)
    (hw : F.wmsg = wmsgOf F.Vs F.Sk (ℐ).r dec) (hm : F.msg = msgOf F.Vs F.Sk dec) : F.HDecodes :=
  F.vDecodesRead_of dec hw hm

def SelRounds (sel : ℕ → String → ℕ → Prop) : Prop :=
  ∀ i v x, sel i v x → Flock.Registered.roundCommitment v = true

/-- A circuit fact no check makes; it fails at V*'s message rows, whose port words past the first `128 n` bits no
unit reads. -/
def RowsRead : Prop := ∀ i < (ℐ).r, ∀ j g, posK (F.Vs j).c (F.Vs j).hU (F.Vs j).nv g ∈ F.Sk i j → ReadW F.Vs j g

theorem reads_of_skOf {own : Fin F.stmts.size → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
    (hSk : F.Sk = skOf F.Vs own sel) (hsel : SelRounds sel) {i : ℕ} {j : Fin F.stmts.size}
    {g : Fin ((Placed.keyProg (F.Vs j).c (F.Vs j).hU (F.Vs j).nv).circuit
      (Layout.outsOf (F.Vs j).c (F.Vs j).hU (F.Vs j).nv)).N} {u : Fin (F.Vs j).nv}
    (hg : posK (F.Vs j).c (F.Vs j).hU (F.Vs j).nv g ∈ F.Sk i j)
    (hio : g ∈ ((Placed.keyProg (F.Vs j).c (F.Vs j).hU (F.Vs j).nv).partition
      (Layout.outsOf (F.Vs j).c (F.Vs j).hU (F.Vs j).nv)).io u) :
    ∃ p ∈ own j, Flock.Registered.roundCommitment p.value = true ∧ ∃ x, p.reads[u.val]? = some (some x) := by
  rw [hSk] at hg
  rcases hpos : posK (F.Vs j).c (F.Vs j).hU (F.Vs j).nv g with ⟨u₀, q⟩ | g'
  · rw [hpos, mem_skOf_inl] at hg
    obtain ⟨p, hp, -, x, hx, hs⟩ := hg
    obtain rfl := unit_of_io hpos hio
    exact ⟨p, hp, hsel i p.value x hs, x, hx⟩
  · rw [hpos] at hg
    exact absurd hg (inr_not_mem_skOf g')

theorem hFull_of (hrd : F.RowsRead) : F.HFull := hrd

def VFullRead : Prop := VFullR (ℐ).r F.Sk

theorem vFullRead_of : F.VFullRead := vFullR_of_read _ _

end VStarFamily

theorem vbridgeRead_of_family (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray)
    (N k Rw : ℕ) {GS : Flock.RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hst : F.HStruct GS gs)
    (hsh : F.HShape dg N k Rw GS) (hres : F.HResL dg N k Rw gs) (hu : F.HUL dg N k Rw) (hrd : F.HReads dg N k Rw)
    (hsch : F.HSch) (hfiles : F.HFiles dg N k Rw) (hruns : F.HRuns dg N k Rw) (hcm : F.HCommits)
    (hdec : F.VDecodesRead) (hfull : F.VFullRead) :
    VBridgeRead (I := ℐ) F.zs dg N k Rw F.x F.wmsg F.Sk F.cmt F.msg :=
  ⟨fun R ω t R₂ hc τ o hout h₀ =>
    have h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o := ⟨h₀, hc, hout⟩
    (F.files R ω t (layers F.zs dg N k Rw R ω t R₂ τ)).accepts regs (F.x R ω) T CT sch (hsch R ω t _) _
      (hfiles R ω t R₂ τ o h) (hruns R ω t R₂ τ o h fun r => by
        obtain ⟨⟨v, info, kappa, hv, hvs⟩, hw, hc⟩ := hsh R ω t R₂ τ o h r
        exact ⟨FlockVBridge.OfVStar.algebraHolds_of_decodeL (F.zs R ω t R₂) (layers F.zs dg N k Rw R ω t R₂ τ)
            (fun s => (h.1 s).2) hst.1 (Flock.RecAlgebra.structureOf_bitsOk hst.1) hv hw hc hvs hst.2 F.sess
            (fun p => F.un p r) (fun p => cp p r) (hres R ω t R₂ τ o h r),
          FlockVBridge.AccChain.accHolds_of_units
            (unitSums_of_opensAtL (F.zs R ω t R₂) (layers F.zs dg N k Rw R ω t R₂ τ) F.sh _ _ F.sessO
              (fun ℓ j => F.unO ℓ r j) (fun ℓ j => cpO ℓ r j) (hu R ω t R₂ τ o h r))
            (F.coefsAre R ω t _ r) (F.consAre R ω t _ r) (hrd R ω t R₂ τ o h r)⟩),
    hcm, hdec, hfull⟩

end family

end FlockVBridge.Family
