module

public import Proofs.Flock.VBridge.Family.Opened
public import Proofs.Flock.Claims.Recursion.VStarParts
public import Flock.RecCoins
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
import Proofs.Flock.VBridge.Algebra.Holo
import Std.Tactic.BVDecide.Normalize.BitVec

/-! A rep's Ligerito openings at V*'s scheme `plain`, assembled from what one query must give; nothing here reads the verdict's pieces. -/

@[expose] public section

namespace RecDeploy.OpenedPlain

open Flock Flock.RecAlgebra FlockSoundness.Refine FlockVBridge.Family

def QueryOpens (sch : Flock.Schedule) (cap : Array ByteArray) (p : LigeritoProof) (ℓ j pos : ℕ) : Prop :=
  ((openAt p (sch.levels.size - 1) ℓ).1[j]!).size = 2 ^ (sch.levels[ℓ]!).lanes ∧
    climb plain (Sha512.hash (MerkleScheme.rowBytes (openAt p (sch.levels.size - 1) ℓ).1[j]!)) pos
        (sibsOf (openAt p (sch.levels.size - 1) ℓ).2.1 ((sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap) j) =
      (capAt cap p ℓ).getD (pos >>> ((sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap)) ByteArray.empty

def LevelShaped (sch : Flock.Schedule) (p : LigeritoProof) (ℓ n : ℕ) : Prop :=
  (openAt p (sch.levels.size - 1) ℓ).1.size = n ∧
    (openAt p (sch.levels.size - 1) ℓ).2.1.size = n * ((sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap) ∧
    (∀ x ∈ (openAt p (sch.levels.size - 1) ℓ).2.1, x.size = 64) ∧ (openAt p (sch.levels.size - 1) ℓ).2.2 = #[]

theorem opened_plain_of {sch : Flock.Schedule} {cap : Array ByteArray} {p : LigeritoProof} {M : Messages F128}
    {cs : Flock.ConsCtx} (hℓ : cs.start < sch.levels.size) (hcaps : CapsAt sch cap p)
    (hsh : LevelShaped sch p cs.start cs.pos.size)
    (hq : ∀ j < cs.pos.size, QueryOpens sch cap p cs.start j cs.pos[j]!)
    (hcons : M.cons cs.start ((eqTable cs.alpha).extract 0 (sch.levels[cs.start]!).q) =
      (Array.range (2 ^ (sch.levels[cs.start]!).lanes)).map
        (consAt (openAt p (sch.levels.size - 1) cs.start).1 (sch.levels[cs.start]!).q cs.alpha)) :
    Opened plain sch cap p M cs := by
  obtain ⟨hc1, hc2⟩ := hcaps cs.start hℓ
  obtain ⟨h2, h3, h6, h4⟩ := hsh
  unfold Opened
  rw [Conss.getD_default]
  refine ⟨?_, hcons⟩
  rw [h4]
  refine Conss.merkleCheck_of hc1 h2 h3 rfl hc2 h6 fun j hj => ?_
  obtain ⟨hs, hc⟩ := hq j hj
  unfold opens
  simp only [Bool.and_eq_true, decide_eq_true_eq]
  refine ⟨⟨⟨hs, ?_⟩, by simp; rfl⟩, hc⟩
  have hle : j * ((sch.levels[cs.start]!).d - (sch.levels[cs.start]!).cap) +
      ((sch.levels[cs.start]!).d - (sch.levels[cs.start]!).cap) ≤
        cs.pos.size * ((sch.levels[cs.start]!).d - (sch.levels[cs.start]!).cap) := by
    have := Nat.mul_le_mul_right ((sch.levels[cs.start]!).d - (sch.levels[cs.start]!).cap) hj
    rwa [Nat.succ_mul] at this
  simp only [sibsOf, Array.length_toList, Array.size_extract]
  omega

theorem openedAll_of_levels {ms : MerkleScheme} {sch : Flock.Schedule} {sh : Shape} {cap : Array ByteArray}
    {p : LigeritoProof} {M : Messages F128} {rounds : Array (Array F128)} {extras : Array Claim}
    {comb v : Array F128} {info : Info Lin} {kappa : Array (Array (Array F128))}
    (hv : verifierRows sh rounds extras comb = .ok (v, info, kappa)) (h0 : 0 < sh.sched.levels.size)
    (hq : ∀ i < sh.sched.levels.size, ∀ c,
      (Flock.positions (sh.sched.levels[i]!) c).size = (sh.sched.levels[i]!).q)
    (hlev : ∀ ℓ < sh.sched.levels.size, (info.lg.conss[ℓ]!).start = ℓ →
      (info.lg.conss[ℓ]!).pos.size = (sh.sched.levels[ℓ]!).q → Opened ms sch cap p M (info.lg.conss[ℓ]!)) :
    ∀ cs ∈ info.lg.conss.toList, Opened ms sch cap p M cs := by
  intro cs hcs
  obtain ⟨hℓ, he, hpsz⟩ := Conss.conss_mem (rows_conss hv h0 hq) hcs
  have h := hlev cs.start hℓ (by rw [he]) (by rw [he]; exact hpsz)
  rwa [he] at h

theorem verifierRows_info_comb {sh : Shape} {rounds : Array (Array F128)} {extras : Array Claim}
    {comb comb' v v' : Array F128} {info info' : Info Lin} {kappa kappa' : Array (Array (Array F128))}
    (h : verifierRows sh rounds extras comb = .ok (v, info, kappa))
    (h' : verifierRows sh rounds extras comb' = .ok (v', info', kappa')) : info' = info ∧ kappa' = kappa := by
  obtain ⟨V, -, hall⟩ := verifierRows_comb h
  rw [hall comb'] at h'
  simp only [Except.ok.injEq, Prod.mk.injEq] at h'
  exact ⟨h'.2.1.symm, h'.2.2.symm⟩

theorem queryOpens_of_climb {sch : Flock.Schedule} {cap : Array ByteArray} {p : LigeritoProof} {ℓ j pos idx : ℕ}
    {r : Array F128} {ss : List ByteArray} (hr : (openAt p (sch.levels.size - 1) ℓ).1[j]! = r)
    (hss : sibsOf (openAt p (sch.levels.size - 1) ℓ).2.1 ((sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap) j = ss)
    (hsize : r.size = 2 ^ (sch.levels[ℓ]!).lanes) (hlen : ss.length = (sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap)
    (hbits : ∀ h < (sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap, idx.testBit h = pos.testBit h)
    (hclimb : climb plain (Sha512.hash (MerkleScheme.rowBytes r)) idx ss =
      (capAt cap p ℓ).getD (pos >>> ((sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap)) ByteArray.empty) :
    QueryOpens sch cap p ℓ j pos := by
  unfold QueryOpens
  rw [hr, hss]
  exact ⟨hsize, (Conss.climb_congr plain ss _ pos idx fun h hh => (hbits h (hlen ▸ hh)).symm).trans hclimb⟩

theorem climb_of_outs {v : ℕ → Bool} {LANES H : ℕ} {r a cf : Array F128} {rest : ByteArray} {idx : ℕ}
    {ss : List ByteArray} (h : Readable.OpenOuts v plain LANES H r a cf rest idx ss) :
    bytesAt v (outCol LANES H) 64 = climb plain (Sha512.hash (MerkleScheme.rowBytes r)) idx ss := by
  unfold Readable.OpenOuts at h
  have hX := bytesAt_of_cols (colsHold_left (colsHold_left h))
  rw [FlockVBridge.AccChain.climb_size_of plain (fun _ _ => rfl) _ _ _ (sha512_size _)] at hX
  exact hX

theorem queryOpens_of_outs {sch : Flock.Schedule} {cap : Array ByteArray} {p : LigeritoProof} {ℓ j pos : ℕ}
    {v : ℕ → Bool} {LANES H : ℕ} {r a cf : Array F128} {rest : ByteArray} {idx : ℕ} {ss : List ByteArray}
    (h : Readable.OpenOuts v plain LANES H r a cf rest idx ss) (hr : (openAt p (sch.levels.size - 1) ℓ).1[j]! = r)
    (hss : sibsOf (openAt p (sch.levels.size - 1) ℓ).2.1 ((sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap) j = ss)
    (hsize : r.size = 2 ^ (sch.levels[ℓ]!).lanes) (hlen : ss.length = (sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap)
    (hbits : ∀ h < (sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap, idx.testBit h = pos.testBit h)
    (htop : bytesAt v (outCol LANES H) 64 =
      (capAt cap p ℓ).getD (pos >>> ((sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap)) ByteArray.empty) :
    QueryOpens sch cap p ℓ j pos :=
  queryOpens_of_climb hr hss hsize hlen hbits ((climb_of_outs h).symm.trans htop)

theorem dirsRow_testBit {pos H h : ℕ} (hH : H ≤ 32) (hh : h < H) :
    ((RecCoins.dirsRow pos H)[h / 16]!).testBit (h % 16) = pos.testBit h := by
  by_cases h16 : h < 16
  · rw [Nat.div_eq_of_lt h16, Nat.mod_eq_of_lt h16]
    have e : (RecCoins.dirsRow pos H)[0]! = pos % 2 ^ H % 2 ^ 16 := by
      rw [getElem!_pos _ 0 (by simp [RecCoins.dirsRow, RecCoins.DIRS_WORDS])]
      simp [RecCoins.dirsRow]
    rw [e, Nat.testBit_mod_two_pow, Nat.testBit_mod_two_pow]
    simp [hh, h16]
  · rw [show h / 16 = 1 by omega]
    have e : (RecCoins.dirsRow pos H)[1]! = pos % 2 ^ H / 2 ^ 16 := by
      rw [getElem!_pos _ 1 (by simp [RecCoins.dirsRow, RecCoins.DIRS_WORDS])]
      simp [RecCoins.dirsRow]
    rw [e, Nat.testBit_div_two_pow, Nat.testBit_mod_two_pow, show h % 16 + 16 = h by omega]
    simp [hh]

end RecDeploy.OpenedPlain
