module

public import Proofs.Flock.C3.Opening.SrcOpening
public import Proofs.Flock.C3.Reps.ReadWhole
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

/-! The opening tie at V*'s layer (`OpenTieL`). -/

@[expose] public section

namespace FlockVBridge.OfVStar

open Flock.HmNets hiding and
open Flock FlockSoundness.Discharge.Hm
open Flock.RecResiduals (groups)
open FlockVBridge.PartS (wIdx)

section Copy

open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open FlockVBridge.Decode

variable {mPts nU J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : SamplingDesign nU} (zo : ZkOuter mPts nU J Reg' c hU hscope Lv)

theorem fill_rowCol {bits : Fin (Layout.keyN c hU nU) → Bool} {u : Fin nU} {j : ℕ} {sh : Shape}
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} {p : ℕ} {w v : ℕ → F128} (h : OpeningZeroAtL zo bits u j sh gs p w v)
    (hj : j < c.upv) {S0 : RecAlgebra.GfS} (hS : RecOpening.structureOf sh = .ok S0)
    (hg : groups S0 (RecOpening.portSizes sh) = .ok gs) (hpg : p < gs.size) {S : RecOpen.GfS} {ns : List ℕ}
    {nv : ℕ} (hpart : RecOpening.part sh p = .ok (S, ns, nv)) {k b : ℕ}
    (hk : k < (wIdx (RecOpening.portSizes sh) gs[p].2.1).size) (hb : b < 128)
    (hrd : ReadCol (Rows.ofNet hU) (rowCol (partRep ns nv).ports 0 (128 * k + b))) :
    copyVal (keyInst hU nU (Layout.outsOf c hU nU) u) (fill zo bits) hj
        (rowCol (partRep ns nv).ports 0 (128 * k + b)) =
      bitOf (w (wIdx (RecOpening.portSizes sh) gs[p].2.1)[k]) b := by
  obtain ⟨-, hns, -⟩ := part_eqOpening hS hg hpg hpart
  have hq : k < ns.sum := by
    rw [hns, ← wIdx_size]
    exact hk
  obtain ⟨_, _, _, _, _, _, _, _, _, S', ns', nv', hpart', hw, -, -⟩ := h
  rw [hpart] at hpart'
  cases hpart'
  refine (copyVal_regBit zo bits hj (hw _ (by omega) hrd)).trans ?_
  unfold wbit
  rw [show (128 * k + b) / 128 = k by omega, show (128 * k + b) % 128 = b by omega]
  exact congrArg (bitOf · b) (sub_lt _ w k hk)

theorem readCol_rsAt {bits : Fin (Layout.keyN c hU nU) → Bool} {u : Fin nU} {j : ℕ} {sh : Shape}
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} {p : ℕ} {w v : ℕ → F128} (h : OpeningZeroAtL zo bits u j sh gs p w v)
    {S0 : RecAlgebra.GfS} (hS : RecOpening.structureOf sh = .ok S0)
    (hg : groups S0 (RecOpening.portSizes sh) = .ok gs) (hpg : p < gs.size) {S : RecOpen.GfS} {ns : List ℕ}
    {nv : ℕ} (hpart : RecOpening.part sh p = .ok (S, ns, nv)) {k e i t b : ℕ}
    (he : (wIdx (RecOpening.portSizes sh) gs[p].2.1)[k]? = some e) (hwi : RecOpening.wi sh (.rs i t) = some e)
    (hb : b < 128) : ReadCol (Rows.ofNet hU) (rowCol (partRep ns nv).ports 0 (128 * k + b)) := by
  obtain ⟨_, _, _, hc, hu, m, cl, hpm, hsh, -⟩ := h
  exact FlockC3.Reps.ReadWhole.readCol_rs hU hc hu hpm hsh hS hg hpg hpart he hwi hb

end Copy

end FlockVBridge.OfVStar

namespace FlockC3.Opening

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled FlockC3.Compose
open FlockVBridge (copyVal keyInst bitOf)
open FlockVBridge.Decode (fill)
open FlockVBridge.OfVStar (OpeningZeroAtL rowCol partRep fill_rowCol readCol_rsAt)
open FlockVBridge.PartS (wIdx)

theorem decodedO_rs {sh : Flock.Shape} (w : Array F128) (cons : ℕ → ℕ → F128) {i t e : ℕ} (hi : i < sh.claims)
    (ht : t < 128) (hwi : RecOpening.wi sh (.rs i t) = some e) : ((decodedO sh w cons).rs[i]!)[t]! = w[e]! := by
  have hrs : (decodedO sh w cons).rs =
      (Array.range sh.claims).map fun k => (Array.range 128).map fun u => valO sh w cons (.rs k u) := by
    simp only [decodedO, RecAlgebra.Messages.map, RecAlgebra.symbolic, Array.map_map, Function.comp_def,
      RecAlgebra.Lin.eval_var]
  have hi' :
      i < ((Array.range sh.claims).map fun k => (Array.range 128).map fun u => valO sh w cons (.rs k u)).size := by
    simp only [Array.size_map, Array.size_range]
    exact hi
  have ht' : t < ((Array.range 128).map fun u => valO sh w cons (.rs i u)).size := by
    simp only [Array.size_map, Array.size_range]
    exact ht
  rw [hrs, getElem!_pos _ i hi', Array.getElem_map, Array.getElem_range, getElem!_pos _ t ht', Array.getElem_map,
    Array.getElem_range]
  simp only [valO, hwi]

section Sessions

variable {nS : ℕ} {Vs : Fin nS → VStmt} {Reg' : Type}

def RsAt (zs : (s : Fin nS) → (Vs s).Outer Reg') (Lb : (s : Fin nS) → (Vs s).Bits) (sh : Flock.Shape)
    (gs : Array (Array ℕ × Array ℕ × Array ℕ)) (w : Array F128) (i : ℕ) (x : Fin 128 → F128) : Prop :=
  ∃ (s : Fin nS) (un : Fin (Vs s).nv) (j p k₀ : ℕ) (v : ℕ → F128) (hj : j < (Vs s).c.upv) (hpg : p < gs.size)
    (S : RecOpen.GfS) (ns : List ℕ) (nv : ℕ),
    OpeningZeroAtL (zs s) (Lb s) un j sh gs p (RecAlgebra.el w) v ∧ RecOpening.part sh p = .ok (S, ns, nv) ∧
    ∀ t : Fin 128, ∃ e, (wIdx (RecOpening.portSizes sh) gs[p].2.1)[k₀ + t.val]? = some e ∧
      RecOpening.wi sh (.rs i t.val) = some e ∧ ∀ b < 128,
        copyVal (keyInst (Vs s).hU (Vs s).nv (Layout.outsOf (Vs s).c (Vs s).hU (Vs s).nv) un) (fill (zs s) (Lb s)) hj
          (rowCol (partRep ns nv).ports 0 (128 * (k₀ + t.val) + b)) = bitOf (x t) b

theorem eq_of_rsAt {zs : (s : Fin nS) → (Vs s).Outer Reg'} {Lb : (s : Fin nS) → (Vs s).Bits} {sh : Flock.Shape}
    {S0 : RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hS : RecOpening.structureOf sh = .ok S0)
    (hg : Flock.RecResiduals.groups S0 (RecOpening.portSizes sh) = .ok gs) {w : Array F128} {i : ℕ} {x : Fin 128 → F128}
    (h : RsAt zs Lb sh gs w i x) (t : Fin 128) : ∃ e, RecOpening.wi sh (.rs i t.val) = some e ∧ x t = w[e]! := by
  obtain ⟨s, un, j, p, k₀, v, hj, hpg, S, ns, nv, hz, hpart, hx⟩ := h
  obtain ⟨e, he, hwi, hb⟩ := hx t
  obtain ⟨hk, hke⟩ := Array.getElem?_eq_some_iff.mp he
  refine ⟨e, hwi, FlockVBridge.AccChain.eq_of_bitOf fun b hb' => ?_⟩
  rw [← hb b hb', fill_rowCol (zs s) hz hj hS hg hpg hpart hk hb' (readCol_rsAt (zs s) hz hS hg hpg hpart he hwi hb'),
    hke, RecAlgebra.getElem!_el]

def OpenTieL (zs : (s : Fin nS) → (Vs s).Outer Reg') (Lb : (s : Fin nS) → (Vs s).Bits) (sh : Flock.Shape)
    (w : Array F128) {e : ExecHolo} (u : C3Run e) : Prop :=
  ∃ (S0 : RecAlgebra.GfS) (gs : Array (Array ℕ × Array ℕ × Array ℕ)) (x : Fin 2 → Fin 128 → F128)
    (ρ : Fin 2 → Fin 7 → GF128) (γ : Fin 2 → GF128),
    RecOpening.structureOf sh = .ok S0 ∧ Flock.RecResiduals.groups S0 (RecOpening.portSizes sh) = .ok gs ∧
    (∀ i : Fin 2, RsAt zs Lb sh gs w i.val (x i)) ∧ u.op = openSteps (fun i t => fOf (x i t)) ρ γ

theorem openTie_of_layer {zs : (s : Fin nS) → (Vs s).Outer Reg'} {Lb : (s : Fin nS) → (Vs s).Bits} {m : ℕ}
    {sh : Flock.Shape} (hsh : RecOpening.shape? m 2 = some sh) {w : Array F128} (ops : FlockVBridge.AccChain.Openings)
    {e : ExecHolo} {u : C3Run e} (h : OpenTieL zs Lb sh w u) : OpenTie sh w ops u := by
  obtain ⟨S0, gs, x, ρ, γ, hS, hg, hx, hop⟩ := h
  refine ⟨ρ, γ, hop.trans (congrArg (openSteps · ρ γ) (funext fun i => funext fun t => ?_))⟩
  obtain ⟨e, hwi, hxe⟩ := eq_of_rsAt hS hg (hx i) t
  show fOf (x i t) = fOf (((decodedO sh w (consO ops)).rs[i.val]!)[t.val]!)
  rw [hxe, decodedO_rs w (consO ops) (by rw [shape_claims hsh]; exact i.isLt) t.isLt hwi]

end Sessions

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

theorem openingReadAt_srcL
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape} (hsh : RecOpening.shape? (holoAt st hp).cls.shape.mReg 2 = some sh) (unit : String)
    (c3 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → RecCoins.C3)
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128)
    (ops : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → FlockVBridge.AccChain.Openings)
    (hsess : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      SessionAt (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) sh unit (c3 R ω t (layers zs dg N k Rw R ω t R₂ τ))
        (w R ω t (layers zs dg N k Rw R ω t R₂ τ)) (ops R ω t (layers zs dg N k Rw R ω t R₂ τ)))
    (hclaim : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      ∀ a b, coinClaim (c3 R ω t (layers zs dg N k Rw R ω t R₂ τ)) unit sh 0 = .ok a →
        coinClaim (c3 R ω t (layers zs dg N k Rw R ω t R₂ τ)) unit sh 1 = .ok b →
        ∀ i, FlockSoundness.Refine.ClaimRel (holoAt st hp).cls.shape.mReg
          (claimO sh (w R ω t (layers zs dg N k Rw R ω t R₂ τ)) ![a, b] i).toClaim
          ((src R ω t (layers zs dg N k Rw R ω t R₂ τ)).claim i).toClaim)
    (htie : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      OpenTieL (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) sh (w R ω t (layers zs dg N k Rw R ω t R₂ τ))
        (src R ω t (layers zs dg N k Rw R ω t R₂ τ))) :
    AtSrc zs dg N k Rw src fun _ u => OpeningReadAt (runSteps u) u :=
  openingReadAt_src hsh unit c3 w ops hsess hclaim fun R ω t R₂ τ o hacc =>
    openTie_of_layer hsh _ (htie R ω t R₂ τ o hacc)

end FlockC3.Opening
