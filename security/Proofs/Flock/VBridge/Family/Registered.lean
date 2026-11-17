module

public import Proofs.Flock.VBridge.Family.Openings
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

/-! What a session's acceptance says of its registered reads: each is a row of a port `own` names at a sampled proof
block, and never that a row is read. -/

@[expose] public section

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages
open Flock.RecResiduals (portSizes groups)
open Flock.RecAlgebra (structureOf verifierRows el)

section Reads

variable {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : SamplingDesign nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv)

theorem rd_isLeft {p : PosK c hU nv} (h : zo.rd.isRead p) : p.isLeft = true :=
  zo.rd.read_row p h

end Reads

theorem oneAt_keyProg (c : Flock.Circuit) (hU : Built c.unit) (nv : ℕ) (u : Fin nv) :
    Const.Prog.oneAt (Placed.keyProg c hU nv) u = Layout.oneG (c := c) (hU := hU) (n := nv) :=
  Fin.ext (Layout.oneAt_uniform_val (k := Placed.inGates nv (Layout.keyB c hU)) (R := Placed.circuitRows c hU)
    (ins := Placed.insOf nv (Layout.keyB c hU) (Aliased.keyIdx c (Layout.keyB c hU)))
    (one := Placed.oneGate nv (Layout.keyB c hU))
    (hone := Placed.insOf_ne_one nv (Layout.keyB c hU) (Aliased.keyIdx c (Layout.keyB c hU))) nv u)

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)

/-- Discharged by a check of the layer at `oneG` in `ZkOuter.Good`, because V* registers no row holding the constant.
Acceptance does not give it: the public-file layer `Xpub` pins it, the `--zk` layer does not. -/
def OneInLayer : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ j,
    layers F.zs dg N k Rw R ω t R₂ τ j (Layout.oneG (c := (F.Vs j).c) (hU := (F.Vs j).hU) (n := (F.Vs j).nv)) = true

theorem hOne_of (h : F.OneInLayer dg N k Rw) : F.HOne dg N k Rw := by
  intro R ω t R₂ τ o ha r p _
  rw [oneAt_keyProg]
  exact h R ω t R₂ τ o ha (F.sess p)

theorem hOneOpen_of (h : F.OneInLayer dg N k Rw) : F.HOneOpen dg N k Rw := by
  intro R ω t R₂ τ o ha r ℓ _ j _
  rw [oneAt_keyProg]
  exact h R ω t R₂ τ o ha (F.sessO ℓ)

/-- Discharged by extraction at the committed trace τ*, else by record custody; acceptance does not give it, because
`rd` is the family's field. -/
def RowsRegistered : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ j (u : Fin (F.Vs j).nv)
    (q : Fin (F.Vs j).c.ports.size), (∃ pt ∈ (F.zs R ω t R₂ j).own, pt.name = (F.Vs j).c.ports[q].name) →
      (F.zs R ω t R₂ j).rd.isRead (.inl (u, q))

def RowColKeys : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ p, p < (partsOf F.sh).size →
    ∀ q < 128 * (nsOf F.sh p).sum, rowColOf F.sh p q < (Rows.ofNet (F.Vs (F.sess p)).hU).nIn ∧
      ∃ (q' : Fin (F.Vs (F.sess p)).c.ports.size) (k' : ℕ) (_ : k' < Layout.portRowW (F.Vs (F.sess p)).c q'),
        Aliased.srcKey (F.Vs (F.sess p)).c (0 * (Rows.ofNet (F.Vs (F.sess p)).hU).nIn + rowColOf F.sh p q) =
          .msg q'.val (k' / 8) (k' % 8) ∧
        ∃ pt ∈ (F.zs R ω t R₂ (F.sess p)).own, pt.name = (F.Vs (F.sess p)).c.ports[q'].name

theorem hRowKeys_of (hreg : F.RowsRegistered dg N k Rw) (hcol : F.RowColKeys dg N k Rw) : F.HRowKeys dg N k Rw := by
  intro R ω t R₂ τ o ha r p hp q hq
  obtain ⟨hlt, q', k', hk, hkey, hown⟩ := hcol R ω t R₂ τ o ha p hp q hq
  exact ⟨_, hlt, q', k', hk, hkey, hreg R ω t R₂ τ o ha (F.sess p) (F.un p r) q' hown, rfl⟩

def SameRow : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ (r : Fin 2) (p p' a a' e : ℕ),
    p < (partsOf F.sh).size → p' < (partsOf F.sh).size → ∀ (ha : a < (wIOf F.sh p).size)
    (ha' : a' < (wIOf F.sh p').size), (wIOf F.sh p)[a] = (wIOf F.sh p')[a'] → e < 128 →
    ∀ (q : Fin (F.Vs (F.sess p)).c.ports.size) (b : ℕ) (q' : Fin (F.Vs (F.sess p')).c.ports.size) (b' : ℕ),
      Aliased.srcKey (F.Vs (F.sess p)).c (0 * (Rows.ofNet (F.Vs (F.sess p)).hU).nIn + rowColOf F.sh p (128 * a + e)) =
        .msg q.val (b / 8) (b % 8) →
      Aliased.srcKey (F.Vs (F.sess p')).c
          (0 * (Rows.ofNet (F.Vs (F.sess p')).hU).nIn + rowColOf F.sh p' (128 * a' + e)) =
        .msg q'.val (b' / 8) (b' % 8) →
      b < Layout.portRowW (F.Vs (F.sess p)).c q → b' < Layout.portRowW (F.Vs (F.sess p')).c q' →
      (F.zs R ω t R₂ (F.sess p)).rd.isRead (.inl (F.un p r, q)) →
      (F.zs R ω t R₂ (F.sess p')).rd.isRead (.inl (F.un p' r, q')) →
      b = b' ∧ (F.zs R ω t R₂ (F.sess p)).rd.port (.inl (F.un p r, q)) =
          (F.zs R ω t R₂ (F.sess p')).rd.port (.inl (F.un p' r, q')) ∧
        (F.zs R ω t R₂ (F.sess p)).rd.readAt (.inl (F.un p r, q)) =
          (F.zs R ω t R₂ (F.sess p')).rd.readAt (.inl (F.un p' r, q'))

/-- Discharged by `SharedRows` where no read is segmented (`rowsShared`); the transfer's `cr/sha-512` term does not
price it, because each of its finders sees one session's record. -/
def RowsShared : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ (s s' : Fin F.stmts.size)
    (u : Fin (F.Vs s).nv) (u' : Fin (F.Vs s').nv) (q : Fin (F.Vs s).c.ports.size) (q' : Fin (F.Vs s').c.ports.size),
    (F.zs R ω t R₂ s).rd.isRead (.inl (u, q)) → (F.zs R ω t R₂ s').rd.isRead (.inl (u', q')) →
    (F.zs R ω t R₂ s).rd.port (.inl (u, q)) = (F.zs R ω t R₂ s').rd.port (.inl (u', q')) →
    (F.zs R ω t R₂ s).rd.readAt (.inl (u, q)) = (F.zs R ω t R₂ s').rd.readAt (.inl (u', q')) →
    ∀ (b : ℕ) (hb : b < Layout.portRowW (F.Vs s).c q) (hb' : b < Layout.portRowW (F.Vs s').c q'),
      (Decode.regVal (F.zs R ω t R₂ s) (.inl (u, q))).2
          ⟨b, lt_of_lt_of_le hb ((Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _))⟩ =
        (Decode.regVal (F.zs R ω t R₂ s') (.inl (u', q'))).2
          ⟨b, lt_of_lt_of_le hb' ((Finset.le_sup (Finset.mem_univ q')).trans (Nat.le_succ _))⟩

theorem hRowsAgree_of (hreg : F.RowsRegistered dg N k Rw) (hcol : F.RowColKeys dg N k Rw) (hsame : F.SameRow dg N k Rw)
    (hshared : F.RowsShared dg N k Rw) : F.HRowsAgree dg N k Rw := by
  intro R ω t R₂ τ o ha r p p' a a' e bit hp hp' ha₁ ha₁' heq he hrb
  obtain ⟨-, q, b, hb, hkey, hread, hval⟩ := hrb
  have hsz : (wIOf F.sh p').size = (nsOf F.sh p').sum := by rw [wIOf, nsOf, OfVStar.wIdx_size]
  obtain ⟨hlt', q', b', hb', hkey', hown'⟩ := hcol R ω t R₂ τ o ha p' hp' (128 * a' + e) (by omega)
  have hread' := hreg R ω t R₂ τ o ha (F.sess p') (F.un p' r) q' hown'
  obtain ⟨rfl, hport, hat⟩ :=
    hsame R ω t R₂ τ o ha r p p' a a' e hp hp' ha₁ ha₁' heq he q b q' b' hkey hkey' hb hb' hread hread'
  refine ⟨hlt', q', b, hb', hkey', hread', ?_⟩
  rw [← hval]
  exact (hshared R ω t R₂ τ o ha _ _ _ _ _ _ hread hread' hport hat b hb hb').symm

end VStarFamily

end family

end FlockVBridge.Family
