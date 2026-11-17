module

public import Proofs.Flock.Claims.KS.LiveMix
public import Proofs.Flock.Claims.KS.ZkDecodeW
public import Proofs.Flock.Claims.KS.ZkEvent
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
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.KS.ConflictRun
public import Proofs.Flock.Claims.KS.PlainFirst
public import Proofs.Flock.Claims.KS.PlainThird

/-! The masked mixture's first and third bounds, from a simulation, which unlike the plain bound read no sample in
the record. -/

@[expose] public section

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Tables Claims.Bridge.Custody Claims.Bridge.Live
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

def ZkSimRun {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (J : Claims.Msg → ℕ) (mPts : ℕ) {nMax : ℕ}
    (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) (w : (x : Claims.Msg) → Wire nMax (J x) mPts) {τ : Type}
    (P' : Claims.Prover (Ω × τ)) (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V true).k x) → Coin) : Prop :=
  ∀ u r', (Claims.run sha512 P (V true) x ω (Φ (u, r'))).out = .acc →
    ∀ s spec sts link c₁ mids f,
      act true x ((Claims.run sha512 P (V true) x ω (Φ (u, r'))).msgs.map Prod.fst)
          ((List.range ((Claims.run sha512 P (V true) x ω (Φ (u, r'))).msgs.map Prod.fst).length).map
            (Claims.coinSeq (Φ (u, r')))) = .decide s spec sts link c₁ mids f →
        liveOut J (fun _ => mPts) N dg w sha512 P' x (ω, u) r' = viewOf (J x) mPts spec link c₁ f

namespace Gen

variable [S : ZkSetup]

def ZkSimRun {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (J : Claims.Msg → ℕ) (mPts : ℕ) {nMax : ℕ}
    (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) (w : (x : Claims.Msg) → Wire nMax (J x) mPts) {τ : Type}
    (P' : Claims.Prover (Ω × τ)) (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V true).k x) → Coin) : Prop :=
  ∀ u r', (Claims.run sha512 P (V true) x ω (Φ (u, r'))).out = .acc →
    ∀ s spec sts link c₁ mids f,
      act true x ((Claims.run sha512 P (V true) x ω (Φ (u, r'))).msgs.map Prod.fst)
          ((List.range ((Claims.run sha512 P (V true) x ω (Φ (u, r'))).msgs.map Prod.fst).length).map
            (Claims.coinSeq (Φ (u, r')))) = .decide s spec sts link c₁ mids f →
        liveOut J (fun _ => mPts) N dg w sha512 P' x (ω, u) r' = viewOf (J x) mPts spec link c₁ f

end Gen

def ZkSim {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (J : Claims.Msg → ℕ) (mPts : ℕ) {nMax : ℕ}
    (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) (w : (x : Claims.Msg) → Wire nMax (J x) mPts) {τ : Type}
    [Fintype τ] [Nonempty τ] (P' : Claims.Prover (Ω × τ))
    (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V true).k x) → Coin) : Prop :=
  (∀ E : (Fin ((V true).k x) → Coin) → Prop, Claims.Pr (fun p => E (Φ p)) = Claims.Pr E) ∧
    ZkSimRun P x ω J mPts N dg w P' Φ

namespace Gen

variable [S : ZkSetup]

def ZkSim {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (J : Claims.Msg → ℕ) (mPts : ℕ) {nMax : ℕ}
    (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) (w : (x : Claims.Msg) → Wire nMax (J x) mPts) {τ : Type}
    [Fintype τ] [Nonempty τ] (P' : Claims.Prover (Ω × τ))
    (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V true).k x) → Coin) : Prop :=
  (∀ E : (Fin ((V true).k x) → Coin) → Prop, Claims.Pr (fun p => E (Φ p)) = Claims.Pr E) ∧
    ZkSimRun P x ω J mPts N dg w P' Φ

end Gen

def zkCoins {x : Claims.Msg} {N : ℕ} (hk : N + 2 = (V true).k x) (n : Fin ((V true).k x) → ℕ)
    (p : (Fin ((V true).k x) → Coin) × (Fin (N + 2) → LCoin 65536)) : Fin ((V true).k x) → Coin :=
  @mergeAt _ _ _ 𝔽 hk n p.1 p.2

theorem zkSim_of_run {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {J : Claims.Msg → ℕ} {mPts : ℕ}
    {N : Claims.Msg → ℕ} {dg : ByteArray → ByteArray} {w : (x : Claims.Msg) → Wire 65536 (J x) mPts}
    {P' : Claims.Prover (Ω × (Fin ((V true).k x) → Coin))} (hk : N x + 2 = (V true).k x)
    (n : Fin ((V true).k x) → ℕ) (h : ZkSimRun P x ω J mPts N dg w P' (zkCoins hk n)) :
    ZkSim P x ω J mPts N dg w P' (zkCoins hk n) :=
  ⟨fun E => @Pr_mergeAt _ _ _ 𝔽 _ _ hk n E, h⟩

namespace Gen

variable [S : ZkSetup]

open Classical in
theorem zk_first_of_sim {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) {M : ℕ}
    (C : Fin M → List 𝔽) (j : ℕ) (J : Claims.Msg → ℕ) (mPts : ℕ) {s₀ : Ins} {spec₀ : SessionSpec}
    {sts₀ : Array Stmt} {d : Json} (hset : setupOf true x = some (s₀, spec₀, sts₀)) (hd : s₀.read.draw = some d)
    (hJ : J x ≤ sts₀.size) (hj : j < J x) {nMax : ℕ} (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Claims.Msg) → Wire nMax (J x) mPts) {τ : Type} [Fintype τ] [Nonempty τ] (P' : Claims.Prover (Ω × τ))
    (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V true).k x) → Coin) (hsim : ZkSim P x ω J mPts N dg w P' Φ) :
    Claims.Pr (fun r : Fin ((V true).k x) → Coin => (Claims.run sha512 P (V true) x ω r).out = .acc ∧
        ¬ ∃ (p : Fin M) (v : List 𝔽),
          Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen j) p.val = some v ∧
            v ≠ C p) ≤
      (∑ u, (prob (AccTxJ s₀.read.I d j ((linkOf ((P.next x [] ω).eval sha512).1).getD default).1.toArray C)
        (liveSession dg (J x) mPts (N x)) (liveStrat J (fun _ => mPts) N dg w sha512 P' x (ω, u))).toReal) /
        Fintype.card τ := by
  set E : (Fin ((V true).k x) → Coin) → Prop := fun r => (Claims.run sha512 P (V true) x ω r).out = .acc ∧
    ¬ ∃ (p : Fin M) (v : List 𝔽),
      Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen j) p.val = some v ∧ v ≠ C p
    with hEdef
  obtain ⟨hΦ, hsim⟩ := hsim
  refine Pr_le_liveMix J (fun _ => mPts) N dg w sha512 P' x ω
    (AccTxJ s₀.read.I d j ((linkOf ((P.next x [] ω).eval sha512).1).getD default).1.toArray C) E Φ (hΦ E)
    fun u r' hE => ?_
  obtain ⟨hacc, hno⟩ := hE
  obtain ⟨s, spec, sts, link, c₁, mids, f, hact, hv⟩ := accTxJ_of_run mPts C j hacc
  obtain ⟨rfl, -, rfl⟩ : s = s₀ ∧ spec = spec₀ ∧ sts = sts₀ := by
    have e := (act_decide hact).symm.trans hset
    simp only [Option.some.injEq, Prod.mk.injEq] at e
    exact e
  have hroot : ((linkOf ((P.next x [] ω).eval sha512).1).getD default).1.toArray = link.1.toArray := by
    obtain ⟨rest, hhead⟩ := run_head sha512 P (V true) x ω (Φ (u, r')) hacc
    obtain ⟨rest', -, hlink⟩ := act_decide_link hact
    rw [hhead] at hlink
    simp only [List.headD_cons] at hlink
    rw [hlink]
    rfl
  rw [hroot, hsim u r' hacc s spec sts link c₁ mids f hact]
  exact hv d hd hJ hj hno

end Gen

open Classical in
theorem zk_first_of_sim {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) {M : ℕ}
    (C : Fin M → List 𝔽) (j : ℕ) (J : Claims.Msg → ℕ) (mPts : ℕ) {s₀ : Ins} {spec₀ : SessionSpec}
    {sts₀ : Array Stmt} {d : Json} (hset : setupOf true x = some (s₀, spec₀, sts₀)) (hd : s₀.read.draw = some d)
    (hJ : J x ≤ sts₀.size) (hj : j < J x) {nMax : ℕ} (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Claims.Msg) → Wire nMax (J x) mPts) {τ : Type} [Fintype τ] [Nonempty τ] (P' : Claims.Prover (Ω × τ))
    (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V true).k x) → Coin) (hsim : ZkSim P x ω J mPts N dg w P' Φ) :
    Claims.Pr (fun r : Fin ((V true).k x) → Coin => (Claims.run sha512 P (V true) x ω r).out = .acc ∧
        ¬ ∃ (p : Fin M) (v : List 𝔽),
          Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen j) p.val = some v ∧
            v ≠ C p) ≤
      (∑ u, (prob (AccTxJ s₀.read.I d j ((linkOf ((P.next x [] ω).eval sha512).1).getD default).1.toArray C)
        (liveSession dg (J x) mPts (N x)) (liveStrat J (fun _ => mPts) N dg w sha512 P' x (ω, u))).toReal) /
        Fintype.card τ :=
  Gen.zk_first_of_sim (S := S₀) (Ω := Ω) (P := P) (x := x) (ω := ω) (M := M) (C := C) (j := j) (J := J) (mPts := mPts)
    (s₀ := s₀) (spec₀ := spec₀) (sts₀ := sts₀) (d := d) (hset := hset) (hd := hd) (hJ := hJ) (hj := hj) (nMax := nMax)
    (N := N) (dg := dg) (w := w) (τ := τ) (P' := P') (Φ := Φ) (hsim := hsim)

open Classical in
theorem zk_first_of_run {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) {M : ℕ}
    (C : Fin M → List 𝔽) (j : ℕ) (J : Claims.Msg → ℕ) (mPts : ℕ) {s₀ : Ins} {spec₀ : SessionSpec}
    {sts₀ : Array Stmt} {d : Json} (hset : setupOf true x = some (s₀, spec₀, sts₀)) (hd : s₀.read.draw = some d)
    (hJ : J x ≤ sts₀.size) (hj : j < J x) (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Claims.Msg) → Wire 65536 (J x) mPts) (P' : Claims.Prover (Ω × (Fin ((V true).k x) → Coin)))
    (hk : N x + 2 = (V true).k x) (n : Fin ((V true).k x) → ℕ) (hrun : ZkSimRun P x ω J mPts N dg w P' (zkCoins hk n)) :
    Claims.Pr (fun r : Fin ((V true).k x) → Coin => (Claims.run sha512 P (V true) x ω r).out = .acc ∧
        ¬ ∃ (p : Fin M) (v : List 𝔽),
          Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen j) p.val = some v ∧
            v ≠ C p) ≤
      (∑ u, (prob (AccTxJ s₀.read.I d j ((linkOf ((P.next x [] ω).eval sha512).1).getD default).1.toArray C)
        (liveSession dg (J x) mPts (N x)) (liveStrat J (fun _ => mPts) N dg w sha512 P' x (ω, u))).toReal) /
        Fintype.card (Fin ((V true).k x) → Coin) :=
  zk_first_of_sim P x ω C j J mPts hset hd hJ hj N dg w P' (zkCoins hk n) (zkSim_of_run hk n hrun)

open Classical in
theorem zkMix_le_of_run {Ω : Type} (x : Claims.Msg) (ω : Ω) (J : Claims.Msg → ℕ) (mPts : ℕ) (N : Claims.Msg → ℕ)
    (dg : ByteArray → ByteArray) (w : (x : Claims.Msg) → Wire 65536 (J x) mPts)
    (P' : Claims.Prover (Ω × (Fin ((V true).k x) → Coin))) (hk : N x + 2 = (V true).k x)
    (n : Fin ((V true).k x) → ℕ) (pL : LiveOutJ (J x) mPts → Prop) (E : (Fin ((V true).k x) → Coin) → Prop)
    (h : ∀ u r', pL (liveOut J (fun _ => mPts) N dg w sha512 P' x (ω, u) r') → E (zkCoins hk n (u, r'))) :
    (∑ u, (prob pL (liveSession dg (J x) mPts (N x))
      (liveStrat J (fun _ => mPts) N dg w sha512 P' x (ω, u))).toReal) /
        Fintype.card (Fin ((V true).k x) → Coin) ≤ Claims.Pr E :=
  liveMix_le_Pr J (fun _ => mPts) N dg w sha512 P' x ω pL E (zkCoins hk n) (@Pr_mergeAt _ _ _ 𝔽 _ _ hk n E) h

end FlockDeploy.KS
