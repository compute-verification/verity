module

public import Proofs.Flock.Claims.ZK.Os
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

/-! The frame's samples as the masked prover's tape. -/

@[expose] public section

namespace Claims.ZKServer

open _root_.Flock.ZkServer.Frame (Draws sampleBytes codeLen SALT)

theorem fitBytes_of_size (n : ℕ) (b : ByteArray) (h : b.size = n) : FlockComplete.fitBytes n b = b := by
  subst h
  unfold FlockComplete.fitBytes
  rw [Nat.min_self, Nat.sub_self, ByteArray.extract_zero_size]
  apply ByteArray.ext
  simp [ByteArray.data_append]

/-- A rep's samples as the masked prover's pads (`zk_veil::Pads`): all but the pads tree's leaf salts. -/
def samplesPads (d : Draws) : FlockComplete.PadsZk := ⟨d.h, d.mu, d.padH, d.padMu, d.tauSalt⟩

def wordAt (b : ByteArray) (o k : ℕ) : _root_.Flock.F128 :=
  ⟨_root_.Flock.u64OfLE b (o + 16 * k), _root_.Flock.u64OfLE b (o + 16 * k + 8)⟩

def tauAt (l : _root_.Flock.Zk.PadLayout) : ℕ := 32 * l.len + 32 * _root_.Flock.Zk.K_PAD

def tapeOfSamples (t : FlockComplete.TapeZk) (lay : ℕ → _root_.Flock.Zk.PadLayout) (B : ℕ → ByteArray)
    (rep : ℕ → ℕ → ℕ) : FlockComplete.TapeZk :=
  { t with
    h := fun i r => wordAt (B (rep i r)) 0
    mu := fun i r => wordAt (B (rep i r)) (16 * (lay (rep i r)).len)
    padH := fun i r => wordAt (B (rep i r)) (32 * (lay (rep i r)).len)
    padMu := fun i r => wordAt (B (rep i r)) (32 * (lay (rep i r)).len + 16 * _root_.Flock.Zk.K_PAD)
    tauSalt := fun i r => (Draws.ofBytes (lay (rep i r)) (B (rep i r))).tauSalt
    innerSalt := fun i r j => (Draws.ofBytes (lay (rep i r)) (B (rep i r))).salts.extract (SALT * j) (SALT * j + SALT) }

theorem tauSalt_size (l : _root_.Flock.Zk.PadLayout) (b : ByteArray)
    (hb : tauAt l + _root_.Flock.Zk.TAU_SALT_BYTES ≤ b.size) :
    (Draws.ofBytes l b).tauSalt.size = _root_.Flock.Zk.TAU_SALT_BYTES := by
  show (b.extract (tauAt l) (tauAt l + _root_.Flock.Zk.TAU_SALT_BYTES)).size = _
  simp only [ByteArray.size_extract]
  omega

theorem tau_le_sampleBytes (l : _root_.Flock.Zk.PadLayout) :
    tauAt l + _root_.Flock.Zk.TAU_SALT_BYTES ≤ sampleBytes l := by
  unfold tauAt sampleBytes
  omega

theorem padsOf_tapeOfSamples (t : FlockComplete.TapeZk) (lay : ℕ → _root_.Flock.Zk.PadLayout) (B : ℕ → ByteArray)
    (rep : ℕ → ℕ → ℕ) (i r : ℕ) (hb : tauAt (lay (rep i r)) + _root_.Flock.Zk.TAU_SALT_BYTES ≤ (B (rep i r)).size) :
    FlockComplete.padsOf (tapeOfSamples t lay B rep) i r (lay (rep i r)) =
      samplesPads (Draws.ofBytes (lay (rep i r)) (B (rep i r))) := by
  have ht : FlockComplete.fitBytes _root_.Flock.Zk.TAU_SALT_BYTES ((tapeOfSamples t lay B rep).tauSalt i r) =
      (Draws.ofBytes (lay (rep i r)) (B (rep i r))).tauSalt :=
    fitBytes_of_size _ _ (tauSalt_size _ _ hb)
  simp only [FlockComplete.padsOf, ht]
  rfl

theorem innerSalt_fit (t : FlockComplete.TapeZk) (lay : ℕ → _root_.Flock.Zk.PadLayout) (B : ℕ → ByteArray)
    (rep : ℕ → ℕ → ℕ) (i r j : ℕ) (hs : (B (rep i r)).size = sampleBytes (lay (rep i r)))
    (hj : j < codeLen (lay (rep i r))) :
    FlockComplete.fitBytes 192 ((tapeOfSamples t lay B rep).innerSalt i r j) =
      (Draws.ofBytes (lay (rep i r)) (B (rep i r))).salts.extract (SALT * j) (SALT * j + SALT) := by
  apply fitBytes_of_size
  show ((B (rep i r)).extract _ _ |>.extract _ _).size = _
  simp only [ByteArray.size_extract, hs, sampleBytes, SALT]
  omega

theorem padsOf_samplesAt {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (t : FlockComplete.TapeZk)
    (rep : ℕ → ℕ → ℕ) (i r : ℕ) (hs : (L.draws ρ (rep i r)).size = sampleBytes (L.lay x (rep i r))) :
    FlockComplete.padsOf (tapeOfSamples t (L.lay x) (L.draws ρ) rep) i r (L.lay x (rep i r)) =
      samplesPads (L.samplesAt x ρ (rep i r)) :=
  padsOf_tapeOfSamples t (L.lay x) (L.draws ρ) rep i r (by rw [hs]; exact tau_le_sampleBytes _)

theorem cfg_tapeOfSamples {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (hs : List (Msg × Msg))
    (t : FlockComplete.TapeZk) (rep : ℕ → ℕ → ℕ) (i r : ℕ) (hr : L.rep x hs.length = rep i r)
    (hz : (L.draws ρ (rep i r)).size = sampleBytes (L.lay x (rep i r))) :
    (L.cfg x ρ hs).h = (FlockComplete.padsOf (tapeOfSamples t (L.lay x) (L.draws ρ) rep) i r (L.lay x (rep i r))).h ∧
    (L.cfg x ρ hs).rowH =
      (FlockComplete.padsOf (tapeOfSamples t (L.lay x) (L.draws ρ) rep) i r (L.lay x (rep i r))).h ++
        (FlockComplete.padsOf (tapeOfSamples t (L.lay x) (L.draws ρ) rep) i r (L.lay x (rep i r))).padH ∧
    (L.cfg x ρ hs).rowMu =
      (FlockComplete.padsOf (tapeOfSamples t (L.lay x) (L.draws ρ) rep) i r (L.lay x (rep i r))).mu ++
        (FlockComplete.padsOf (tapeOfSamples t (L.lay x) (L.draws ρ) rep) i r (L.lay x (rep i r))).padMu := by
  rw [padsOf_samplesAt L x ρ t rep i r hz]
  unfold LeanRelease.cfg
  rw [hr]
  exact ⟨rfl, rfl, rfl⟩

theorem padsOf_os {W R : Type} (L : LeanRelease W R) (x : Msg) (O : OsTape R (L.lay x)) (ρ : R)
    (t : FlockComplete.TapeZk) (rep : ℕ → ℕ → ℕ) (i r : ℕ) (hk : rep i r < O.reps) :
    FlockComplete.padsOf (tapeOfSamples t (L.lay x) (O.frameBytes ρ) rep) i r (L.lay x (rep i r)) =
      samplesPads ((O.release L).samplesAt x ρ (rep i r)) :=
  padsOf_samplesAt (O.release L) x ρ t rep i r (O.frameBytes_size ρ ⟨rep i r, hk⟩)

end Claims.ZKServer
