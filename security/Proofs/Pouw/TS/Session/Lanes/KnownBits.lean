module

public import Proofs.Pouw.TS.Session.Lanes.PortBits
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.RingTheory.WittVector.IsPoly
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
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Pouw.TS.HeldSound
import Proofs.Flock.Soundness.Discharge.Placed.Wiring
import Proofs.Flock.Soundness.Lowering

/-! The row bits a lane's circuit reads, and its known rows' bits, at the opened rows. -/

@[expose] public section

namespace ResourceAccounting.TS.Lanes

open Flock.SoundAtBridge ResourceAccounting.TS.Checked FlockSoundness.Discharge
open FlockSoundness.Discharge.Layout (keyN keyB keyRowOf portRowW)

/-- Bit `k` is message bit `k % 8` of byte `k / 8` of row port `q`; a fact of the parsed circuit alone. -/
def ColRead (c : _root_.Flock.Circuit) (hU : FlockSoundness.Built c.unit) (q : Fin c.ports.size) (k : ℕ) : Prop :=
  ∃ (i : Fin (Placed.circuitRows c hU).nComp) (x : Fin (Placed.circuitRows c hU).w),
    x ∈ (Placed.circuitRows c hU).a i ++ (Placed.circuitRows c hU).b i ∧ x.val < (Placed.circuitRows c hU).nIn ∧
      Aliased.srcKey c x.val = .msg q.val (k / 8) (k % 8)

variable {s : RowsSession}

theorem rowBitRead_of_col (u : Fin s.n) {q : Fin s.c.ports.size} {k : ℕ} (hk : k < portRowW s.c q)
    (h : ColRead s.c s.hU q k) : RowBitRead s u q k := by
  obtain ⟨i, x, hx, hxn, hkey⟩ := h
  have key : ∀ (R : FlockSoundness.Rows) (_ : (Placed.keyProg s.c s.hU s.n).rows u = R) (i : Fin R.nComp)
      (x : Fin R.w), x ∈ R.a i ++ R.b i → x.val < R.nIn → R.nIn = keyB s.c s.hU →
      Aliased.srcKey s.c x.val = .msg q.val (k / 8) (k % 8) → RowBitRead s u q k := by
    intro R hR i x hx hxn hB hkey
    subst hR
    have hxB : x.val < keyB s.c s.hU := lt_of_lt_of_eq hxn hB
    have hr := FlockVBridge.Decode.keyRowOf_instC (Layout.outsOf s.c s.hU s.n) u x hxB hk hkey
    obtain ⟨-, hlt, -, -⟩ := Layout.keyRowOf_spec hr
    have hnone : (Placed.keyProg s.c s.hU s.n).unit
        ((Const.Prog.instC (Placed.keyProg s.c s.hU s.n) (Layout.outsOf s.c s.hU s.n) u).wire x) = none :=
      Layout.unit_uniform_input (k := Placed.inGates s.n (keyB s.c s.hU)) (R := Placed.circuitRows s.c s.hU)
        (ins := Placed.insOf s.n (keyB s.c s.hU) (Aliased.keyIdx s.c (keyB s.c s.hU)))
        (one := Placed.oneGate s.n (keyB s.c s.hU))
        (hone := Placed.insOf_ne_one s.n (keyB s.c s.hU) (Aliased.keyIdx s.c (keyB s.c s.hU))) s.n _
        (by show _ < 2 + s.n * keyB s.c s.hU; exact hlt)
    exact ⟨_, hk, HiddenExec.rowOfH_of_key _ hr,
      (Const.Prog.instC (Placed.keyProg s.c s.hU s.n) (Layout.outsOf s.c s.hU s.n) u).mem_io_of_read
        hx fun he => by cases hnone.symm.trans he⟩
  exact key _ (Placed.rows_circuitProg s.c s.hU s.n (keyB s.c s.hU) (Aliased.keyIdx s.c (keyB s.c s.hU)) u) i x hx
    hxn rfl hkey

theorem known_held_at {known : Known} (hK : KnownHeld s known) {u : Fin s.n} (hS : s.ServedRegistered u)
    (hcomp : _root_.Flock.Registered.checkComplete s.c s.own = .ok ()) {q : Fin s.c.ports.size}
    {ks : Array (Array ℕ × ByteArray)} (hm : (s.c.ports[q].name, ks) ∈ known) :
    (s.rdU u).readAt (Sum.inl (u, q)) < ks.size ∧ (rsOf s).enc (Sum.inl (u, q)) (rowVal s u q) =
      (_root_.Flock.Registered.rowBytesV2 ks[(s.rdU u).readAt (Sum.inl (u, q))]!.1).data.toList := by
  have hn := _root_.Flock.Registered.checkComplete_ok hcomp _ (Array.getElem_mem q.isLt)
  have hread : (s.rdU u).isRead (Sum.inl (u, q)) := (hS.1 _).2 ⟨q, rfl, hn⟩
  have hname : ((s.rdU u).port (Sum.inl (u, q))).name = s.c.ports[q].name := by
    rw [(hS.2 q hn).1]
    obtain ⟨pt, hpt, hpn⟩ := hn
    exact Held.find_name_eq (l := s.own.toList) (Array.mem_toList_iff.2 hpt) hpn
  exact hK _ _ hm u _ hread hname

theorem known_bytes {known : Known} (hK : KnownHeld s known) {u : Fin s.n} (hS : s.ServedRegistered u)
    (hcomp : _root_.Flock.Registered.checkComplete s.c s.own = .ok ()) {q : Fin s.c.ports.size}
    {ks : Array (Array ℕ × ByteArray)} (hm : (s.c.ports[q].name, ks) ∈ known) :
    (rsOf s).enc (Sum.inl (u, q)) (rowVal s u q) =
      (_root_.Flock.Registered.rowBytesV2 ks[(s.rdU u).readAt (Sum.inl (u, q))]!.1).data.toList :=
  (known_held_at hK hS hcomp hm).2

theorem known_readAt_zero {known : Known} (hK : KnownHeld s known) {u : Fin s.n} (hS : s.ServedRegistered u)
    (hcomp : _root_.Flock.Registered.checkComplete s.c s.own = .ok ()) {q : Fin s.c.ports.size}
    {r : Array ℕ × ByteArray} (hm : (s.c.ports[q].name, #[r]) ∈ known) :
    (s.rdU u).readAt (Sum.inl (u, q)) = 0 := by
  have h := (known_held_at hK hS hcomp hm).1
  simpa using h

theorem portBit_known (hN : ∀ q q' : Fin s.c.ports.size, s.c.ports[q].name = s.c.ports[q'].name → q = q')
    {known : Known} (hK : KnownHeld s known) {u : Fin s.n} (hu : u.val = 0) (hS : s.ServedRegistered u)
    (hcomp : _root_.Flock.Registered.checkComplete s.c s.own = .ok ()) {q : Fin s.c.ports.size}
    (hseg : s.c.ports[q.val]!.segs = none) (hv2 : s.c.ports[q.val]!.v2 = true)
    {ks : Array (Array ℕ × ByteArray)} (hm : (s.c.ports[q].name, ks) ∈ known) {k : ℕ}
    (hread : RowBitRead s u q k) :
    portBit s (openedReg s) s.c.ports[q].name k =
      some ((ks[(s.rdU u).readAt (Sum.inl (u, q))]!.1[k / 16]!).testBit (k % 16)) :=
  portBit_bytes hN hu hS hcomp hseg hv2 (known_bytes hK hS hcomp hm) hread

theorem holds_known (hN : ∀ q q' : Fin s.c.ports.size, s.c.ports[q].name = s.c.ports[q'].name → q = q')
    {known : Known} (hK : KnownHeld s known) {u : Fin s.n} (hu : u.val = 0) (hS : s.ServedRegistered u)
    (hcomp : _root_.Flock.Registered.checkComplete s.c s.own = .ok ()) {q : Fin s.c.ports.size}
    (hseg : s.c.ports[q.val]!.segs = none) (hv2 : s.c.ports[q.val]!.v2 = true)
    {ks : Array (Array ℕ × ByteArray)} (hm : (s.c.ports[q].name, ks) ∈ known) {lo n x : ℕ}
    (hread : ∀ i < n, RowBitRead s u q (lo + i))
    (hx : ∀ i < n, (ks[(s.rdU u).readAt (Sum.inl (u, q))]!.1[(lo + i) / 16]!).testBit ((lo + i) % 16) = x.testBit i) :
    Holds (portBit s (openedReg s) s.c.ports[q].name) lo n x :=
  holds_of_bytes hN hu hS hcomp hseg hv2 (known_bytes hK hS hcomp hm) hread hx

end ResourceAccounting.TS.Lanes
