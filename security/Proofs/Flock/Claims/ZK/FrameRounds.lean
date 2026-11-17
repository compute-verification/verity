module

public import Proofs.Flock.Claims.Complete.Replay.Steps
public import Proofs.Flock.Claims.ZK.CutSession
public import Proofs.Flock.Claims.ZK.FrameMask
public import Proofs.Flock.Claims.ZK.Refine
public import Proofs.Flock.Claims.ZK.SimZerocheck
import all Init.Data.Array.Basic
import all Init.Data.String.Defs
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
import Std.Tactic.BVDecide.Normalize.Prop

/-! The cut's zerocheck and lincheck, round by round, and the frame's release of the device's. -/

@[expose] public section

namespace Claims.ZKServer

open _root_.Flock (F128 Transcript TMW)
open _root_.Flock.Transcript (framed)
open _root_.Flock.ZkServer.Frame (Cfg Masker roundFree LABEL_ZC)
open FlockSoundness.Refine (Frame encs label_enc absorbBytes_enc absorbSlice_enc absorbF_enc framed_enc)
open FlockComplete (PS asksCut zerocheckCut lincheckCut sampleVecP squeezeP_cut_ok sampleP_cut_ok)
open FlockComplete.Replay (pF pushes afterSq bind_go liftTM_ok pure_go)

def zcBind (digest cap root : ByteArray) : List Frame :=
  [.label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap, .label "flock-zerocheck-v0".toUTF8,
    .label _root_.Flock.Zk.LABEL_PADS.toUTF8, .bytes root]

def zcBodies (m : ℕ) (digest cap root : ByteArray) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128)
    (P : List UInt8) : List Claims.Msg :=
  (P ++ encs (zcBind digest cap root) ++ (Frame.sq 2 6).enc) :: (Frame.sq 2 (m - 13)).enc ::
    (encs [.vec ((List.range 64).map fun i => mk (lay.r1ab i)), .vec ((List.range 64).map fun i => mk (lay.r1c i))] ++
      (Frame.sq 1 1).enc) ::
    (List.range (m - 6)).map fun i => encs [.one (mk (lay.ml i 0)), .one (mk (lay.ml i 1))] ++ (Frame.sq 1 1).enc

theorem sampleVecP_cut_ok (coin : ℕ → ℕ → F128) {n k : ℕ} (hk : k ≠ 0) (t : Flock.Transcript) {l : Array Claims.Msg}
    (hl : l.size < n) :
    ((sampleVecP (asksCut coin n) k).run t).go l =
      (.ok ((Array.range k).map (coin l.size), afterSq t 2 k (coin l.size)),
        l.push (Claims.toMsg (framed t 2 k))) :=
  squeezeP_cut_ok coin 2 hk t hl

theorem forIn_go_lt {σ : Type} (body : ℕ → σ → TMW PS (ForInStep σ)) (S : ℕ → σ) (T : ℕ → Flock.Transcript)
    (B : ℕ → Claims.Msg) (L n : ℕ)
    (hbody : ∀ i < n, ∀ (l : Array Claims.Msg), l.size = L + i →
      ((body i (S i)).run (T i)).go l = (.ok (.yield (S (i + 1)), T (i + 1)), l.push (B i)))
    (l : Array Claims.Msg) (hl : l.size = L) :
    ((forIn [:n] (S 0) body).run (T 0)).go l = (.ok (S n, T n), l ++ ((List.range n).map B).toArray) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices h : ∀ k a (l : Array Claims.Msg), a + k ≤ n → l.size = L + a →
      ((forIn (List.range' a k) (S a) body).run (T a)).go l =
        (.ok (S (a + k), T (a + k)), l ++ ((List.range' a k).map B).toArray) by
    simpa [List.range_eq_range'] using h n 0 l (by omega) (by simpa using hl)
  intro k
  induction k with
  | zero =>
    intro a l _ _
    show (Except.ok (S a, T a), l) = _
    simp
  | succ k ih =>
    intro a l hk hl
    rw [List.range'_succ, List.forIn_cons, bind_go (hbody a (by omega) l hl)]
    dsimp only
    rw [ih (a + 1) (l.push (B a)) (by omega) (by simp [hl]; omega), show a + 1 + k = a + (k + 1) by omega]
    simp

theorem forIn_go_from {σ : Type} (body : ℕ → σ → TMW PS (ForInStep σ)) (S : ℕ → σ)
    (T : Flock.Transcript → ℕ → Flock.Transcript) (B : Flock.Transcript → ℕ → Claims.Msg) (L n : ℕ) (hT : ∀ t, T t 0 = t)
    (hbody : ∀ t, ∀ i < n, ∀ (l : Array Claims.Msg), l.size = L + i →
      ((body i (S i)).run (T t i)).go l = (.ok (.yield (S (i + 1)), T t (i + 1)), l.push (B t i)))
    (t : Flock.Transcript) (l : Array Claims.Msg) (hl : l.size = L) :
    ((forIn [:n] (S 0) body).run t).go l = (.ok (S n, T t n), l ++ ((List.range n).map (B t)).toArray) := by
  have h := forIn_go_lt body S (T t) (B t) L n (hbody t) l hl
  rwa [hT] at h

def pairPre (g : ℕ → ℕ → F128) (i : ℕ) (t : Flock.Transcript) : Flock.Transcript :=
  ⟨t.rounds, t.k, pF (g i 1) (pF (g i 0) t.pending)⟩

def pairLoopT (g : ℕ → ℕ → F128) (coin : ℕ → ℕ → F128) (L : ℕ) (t₀ : Flock.Transcript) : ℕ → Flock.Transcript
  | 0 => t₀
  | i + 1 => afterSq (pairPre g i (pairLoopT g coin L t₀ i)) 1 1 (coin (L + i))

def pairLoopB (g : ℕ → ℕ → F128) (coin : ℕ → ℕ → F128) (L : ℕ) (t₀ : Flock.Transcript) (i : ℕ) : Claims.Msg :=
  Claims.toMsg (framed (pairPre g i (pairLoopT g coin L t₀ i)) 1 1)

theorem pairLoopT_pending (g : ℕ → ℕ → F128) (coin : ℕ → ℕ → F128) (L : ℕ) {t₀ : Flock.Transcript}
    (hz : t₀.pending = ByteArray.empty) : ∀ i, (pairLoopT g coin L t₀ i).pending = ByteArray.empty
  | 0 => hz
  | _ + 1 => rfl

theorem empty_toList : ByteArray.empty.data.toList = [] := rfl

theorem pairLoopB_enc (g : ℕ → ℕ → F128) (coin : ℕ → ℕ → F128) (L : ℕ) {t₀ : Flock.Transcript}
    (hz : t₀.pending = ByteArray.empty) :
    pairLoopB g coin L t₀ = fun i => encs [.one (g i 0), .one (g i 1)] ++ (Frame.sq 1 1).enc := by
  funext i
  simp only [pairLoopB, Claims.toMsg, framed_enc, pairPre, pF, absorbF_enc, pairLoopT_pending g coin L hz i,
    empty_toList, encs, List.flatMap_cons, List.flatMap_nil, List.append_nil, List.nil_append, List.append_assoc]

theorem zerocheckCut_bodies (coin : ℕ → ℕ → F128) (n : ℕ) {m : ℕ} (hm : 14 ≤ m) (digest cap root : ByteArray)
    (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128) (t : Flock.Transcript) (l : Array Claims.Msg)
    (hl : l.size + 3 + (m - 6) ≤ n) :
    ∃ res t' L', ((zerocheckCut (asksCut coin n) m digest cap root lay mk).run t).go l = (.ok (res, t'), L') ∧
      L'.toList = l.toList ++ zcBodies m digest cap root lay mk t.pending.data.toList ∧
      t'.pending.data.toList = encs [.one (mk lay.fa), .one (mk lay.fb)] := by
  unfold zerocheckCut
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.label_eq _ _) _)]
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbBytes_eq _ _) _)]
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbBytes_eq _ _) _)]
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.label_eq _ _) _)]
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.label_eq _ _) _)]
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbBytes_eq _ _) _)]
  rw [bind_go (sampleVecP_cut_ok coin (by decide) _ ?h1)]
  rw [bind_go (sampleVecP_cut_ok coin (show m - 13 ≠ 0 by omega) _ ?h2)]
  dsimp only
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbSlice_eq _ _) _)]
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbSlice_eq _ _) _)]
  rw [bind_go (sampleP_cut_ok coin _ ?h3)]
  case h1 => omega
  case h2 => simp only [Array.size_push]; omega
  case h3 => simp only [Array.size_push]; omega
  erw [bind_go (forIn_go_from _
      (fun i => (pushes (fun j => (mk (lay.ml j 0), mk (lay.ml j 1))) i, pushes (fun j => coin (l.size + 3 + j) 0) i))
      (pairLoopT (fun i t => mk (lay.ml i t)) coin (l.size + 3))
      (pairLoopB (fun i t => mk (lay.ml i t)) coin (l.size + 3)) (l.size + 3) (m - 6)
      (fun _ => rfl) ?hbody _ _ ?hl0)]
  case hbody =>
    intro t₀ i hi l' hl'
    dsimp only
    rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbF_eq _ _) _)]
    rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbF_eq _ _) _)]
    rw [bind_go (sampleP_cut_ok coin _ (by omega))]
    rw [hl']
    rfl
  case hl0 => simp only [Array.size_push]
  dsimp only
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbF_eq _ _) _)]
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbF_eq _ _) _)]
  rw [pure_go]
  refine ⟨_, _, _, rfl, ?_, ?_⟩
  · simp only [Array.toList_append, Array.toList_push, List.append_assoc, List.cons_append, List.nil_append,
      zcBodies, zcBind, encs, List.flatMap_cons, List.flatMap_nil, List.append_nil, Claims.toMsg, framed_enc,
      label_enc, absorbBytes_enc, absorbSlice_enc, afterSq, Array.toList_map, Array.toList_range, empty_toList,
      pairLoopB_enc]
  · simp only [absorbF_enc, pairLoopT_pending _ coin (l.size + 3), afterSq, empty_toList, encs, List.flatMap_cons,
      List.flatMap_nil, List.append_nil, List.nil_append]

def zcR1 (dv : ℕ → F128) : List Frame :=
  [.vec ((List.range' 0 64).map dv), .vec ((List.range' 64 64).map dv)]

def zcMl (dv : ℕ → F128) (i : ℕ) : List Frame :=
  [.one (dv (128 + 2 * i)), .one (dv (128 + 2 * i + 1))]

theorem roundFree_zcBind (c : Cfg) (s : Masker) (digest cap : ByteArray)
    (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) (hon : s.on = false) (hd : s.done = false)
    (hi : s.inner = none) (hdg : digest.size < 2 ^ 64) (hcap : cap.size < 2 ^ 64) (hb : s.pos ≤ c.order.size) :
    ∃ out fr, roundFree c s
        (ByteArray.mk (encs [.label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap, .label LABEL_ZC.toUTF8] ++
          (Frame.sq 2 6).enc).toArray) 6 =
        .ok (out, { s with on := true, owed := some 6 }, fr) ∧
      out.data.toList = encs (zcBind digest cap c.root) ++ (Frame.sq 2 6).enc := by
  obtain ⟨out, fr, hr, ho⟩ := roundFree_zc c s [.label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap] [] 2 6 hc
    hon hd hi
    (by
      intro f hf
      simp only [List.cons_append, List.nil_append, List.mem_cons, List.not_mem_nil, or_false] at hf
      rcases hf with rfl | rfl | rfl | rfl
      · show "flock-r1cs-v0".toUTF8.size < 2 ^ 64; decide
      · exact hdg
      · exact hcap
      · show LABEL_ZC.toUTF8.size < 2 ^ 64; decide)
    (by
      intro f hf
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hf
      rcases hf with rfl | rfl | rfl
      · show plainLabel "flock-r1cs-v0".toUTF8; unfold plainLabel; decide
      · trivial
      · trivial)
    (by simp) (.inl rfl) (by decide) (by decide) hb
  refine ⟨out, fr, hr, ?_⟩
  rw [ho]
  simp only [maskFrames, zcBind, encs, LABEL_ZC, List.flatMap_cons, List.flatMap_nil, List.append_nil,
    List.append_assoc]

theorem roundFree_zcOuter (c : Cfg) (s : Masker) (k : ℕ) (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size)
    (hon : s.on = true) (hi : s.inner = none) (hk : k ≠ 0) (hn : k < 2 ^ 64) (hb : s.pos ≤ c.order.size) :
    ∃ out fr, roundFree c s (ByteArray.mk (encs [] ++ (Frame.sq 2 k).enc).toArray) k =
        .ok (out, { s with owed := some k }, fr) ∧ out.data.toList = (Frame.sq 2 k).enc := by
  obtain ⟨out, fr, hr, ho⟩ := roundFree_on c s [] 2 k hc hon hi (by simp) (by simp) (.inl rfl) (by omega) hn hb
  exact ⟨out, fr, hr, by rw [ho]; rfl⟩

theorem maskVals_range' (g : ℕ → F128 → F128) (f : ℕ → F128) :
    ∀ (k s : ℕ), maskVals g s ((List.range' s k).map f) = (List.range' s k).map fun q => g q (f q)
  | 0, _ => rfl
  | k + 1, s => by
    rw [List.range'_succ, List.map_cons, List.map_cons, maskVals, maskVals_range' g f k (s + 1)]

theorem maskVal_false (c : Cfg) (p : ℕ) (v : F128) : maskVal c false p v = v + padAt c p := by
  simp [maskVal]

theorem roundFree_zcR1 (c : Cfg) (s : Masker) (dv : ℕ → F128)
    (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) (hon : s.on = true) (hi : s.inner = none)
    (hpos : s.pos = 0) (hb : 128 ≤ c.order.size) :
    ∃ out fr, roundFree c s (ByteArray.mk (encs (zcR1 dv) ++ (Frame.sq 1 1).enc).toArray) 1 =
        .ok (out, { s with pos := 128, owed := some 1 }, fr) ∧
      out.data.toList = encs (maskFrames (maskVal c s.stopped) 0 (zcR1 dv)) ++ (Frame.sq 1 1).enc := by
  have hm : ∀ f ∈ zcR1 dv, midFrame f := by
    intro f hf
    simp only [zcR1, List.mem_cons, List.not_mem_nil, or_false] at hf
    rcases hf with rfl | rfl <;> simp [midFrame]
  have hp : ∀ f ∈ zcR1 dv, plainFrame f := by
    intro f hf
    simp only [zcR1, List.mem_cons, List.not_mem_nil, or_false] at hf
    rcases hf with rfl | rfl <;> trivial
  have hN : nAbs (zcR1 dv) = 128 := by simp [zcR1, nAbs]
  obtain ⟨out, fr, hr, ho⟩ := roundFree_on c s (zcR1 dv) 1 1 hc hon hi hm hp (.inr ⟨rfl, rfl⟩) (by decide)
    (by decide) (by rw [hN, hpos]; omega)
  rw [hN, hpos] at hr
  rw [hpos] at ho
  exact ⟨out, fr, hr, ho⟩

theorem roundFree_zcMl (c : Cfg) (s : Masker) (dv : ℕ → F128) (i : ℕ)
    (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) (hon : s.on = true) (hi : s.inner = none)
    (hpos : s.pos = 128 + 2 * i) (hb : 128 + 2 * i + 2 ≤ c.order.size) :
    ∃ out fr, roundFree c s (ByteArray.mk (encs (zcMl dv i) ++ (Frame.sq 1 1).enc).toArray) 1 =
        .ok (out, { s with pos := 128 + 2 * i + 2, owed := some 1 }, fr) ∧
      out.data.toList = encs (maskFrames (maskVal c s.stopped) (128 + 2 * i) (zcMl dv i)) ++ (Frame.sq 1 1).enc := by
  have hm : ∀ f ∈ zcMl dv i, midFrame f := by
    intro f hf
    simp only [zcMl, List.mem_cons, List.not_mem_nil, or_false] at hf
    rcases hf with rfl | rfl <;> trivial
  have hp : ∀ f ∈ zcMl dv i, plainFrame f := by
    intro f hf
    simp only [zcMl, List.mem_cons, List.not_mem_nil, or_false] at hf
    rcases hf with rfl | rfl <;> trivial
  have hN : nAbs (zcMl dv i) = 2 := rfl
  obtain ⟨out, fr, hr, ho⟩ := roundFree_on c s (zcMl dv i) 1 1 hc hon hi hm hp (.inr ⟨rfl, rfl⟩) (by decide)
    (by decide) (by rw [hN, hpos]; omega)
  rw [hN, hpos] at hr
  rw [hpos] at ho
  exact ⟨out, fr, hr, ho⟩

theorem zcBodies_frame (m : ℕ) (digest cap : ByteArray) (c : Cfg) (lay : _root_.Flock.Zk.PadLayout)
    (mk dv : ℕ → F128) (hn : lay.nMlv = m - 6) (hmk : ∀ p < lay.fa + 2, mk p = dv p + padAt c p) :
    zcBodies m digest cap c.root lay mk [] =
      (encs (zcBind digest cap c.root) ++ (Frame.sq 2 6).enc) :: (Frame.sq 2 (m - 13)).enc ::
        (encs (maskFrames (maskVal c false) 0 (zcR1 dv)) ++ (Frame.sq 1 1).enc) ::
        (List.range (m - 6)).map fun i =>
          encs (maskFrames (maskVal c false) (128 + 2 * i) (zcMl dv i)) ++ (Frame.sq 1 1).enc := by
  have hfa : lay.fa = 128 + 2 * (m - 6) := by rw [← hn]; rfl
  have hA : (List.range 64).map (fun i => mk (lay.r1ab i)) = (List.range' 0 64).map fun q => dv q + padAt c q := by
    rw [List.range_eq_range']
    refine List.map_congr_left fun q hq => ?_
    rw [List.mem_range'] at hq
    exact hmk q (by omega)
  have hC : (List.range 64).map (fun i => mk (lay.r1c i)) = (List.range' 64 64).map fun q => dv q + padAt c q := by
    rw [List.range'_eq_map_range, List.map_map]
    refine List.map_congr_left fun q hq => ?_
    rw [List.mem_range] at hq
    exact hmk (64 + q) (by omega)
  unfold zcBodies
  simp only [List.nil_append, zcR1, maskFrames, List.length_map, List.length_range', maskVals_range', maskVal_false,
    hA, hC, List.cons.injEq, true_and]
  refine List.map_congr_left fun i hi => ?_
  rw [List.mem_range] at hi
  simp only [zcMl, maskFrames, maskVal_false]
  rw [hmk (lay.ml i 0) (by unfold _root_.Flock.Zk.PadLayout.ml; omega),
    hmk (lay.ml i 1) (by unfold _root_.Flock.Zk.PadLayout.ml; omega)]
  rfl

def lcBodies (kLog : ℕ) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128) (P : List UInt8) : List Claims.Msg :=
  (P ++ (Frame.label "flock-lincheck-v0".toUTF8).enc ++ (Frame.sq 1 1).enc) :: (Frame.sq 1 1).enc ::
    ((List.range (kLog - 6)).map fun j =>
      encs [.one (mk (lay.lc j 0)), .one (mk (lay.lc j 1))] ++ (Frame.sq 1 1).enc) ++
    [encs [.vec ((List.range 64).map fun i => mk (lay.zp i))] ++ (Frame.sq 1 1).enc]

theorem lincheckCut_bodies (coin : ℕ → ℕ → F128) (n : ℕ) (c : _root_.Flock.CircuitFold) (m : ℕ)
    (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128) (zc : _root_.Flock.ZcOut) (t : Flock.Transcript)
    (l : Array Claims.Msg) (hl : l.size + 3 + (c.kLog - 6) ≤ n)
    (hf : ∃ v, c.fold (coin l.size 0) (_root_.Flock.skipWeights zc.z) (zc.rho.extract 0 (c.kLog - 6)) = .ok v) :
    ∃ res t' L', ((lincheckCut (asksCut coin n) c m lay mk zc).run t).go l = (.ok (res, t'), L') ∧
      L'.toList = l.toList ++ lcBodies c.kLog lay mk t.pending.data.toList ∧ t'.pending = ByteArray.empty := by
  obtain ⟨v, hv⟩ := hf
  unfold lincheckCut
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.label_eq _ _) _)]
  rw [bind_go (sampleP_cut_ok coin _ ?h1)]
  rw [bind_go (sampleP_cut_ok coin _ ?h2)]
  case h1 => omega
  case h2 => simp only [Array.size_push]; omega
  dsimp only
  rw [hv]
  dsimp only
  rw [bind_go (pure_go _ _ _)]
  erw [bind_go (forIn_go_from _
      (fun j => (pushes (fun j => (mk (lay.lc j 0), mk (lay.lc j 1))) j, pushes (fun j => coin (l.size + 2 + j) 0) j))
      (pairLoopT (fun j t => mk (lay.lc j t)) coin (l.size + 2))
      (pairLoopB (fun j t => mk (lay.lc j t)) coin (l.size + 2)) (l.size + 2) (c.kLog - 6)
      (fun _ => rfl) ?hbody _ _ ?hl0)]
  case hbody =>
    intro t₀ j hj l' hl'
    dsimp only
    rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbF_eq _ _) _)]
    rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbF_eq _ _) _)]
    rw [bind_go (sampleP_cut_ok coin _ (by omega))]
    rw [hl']
    rfl
  case hl0 => simp only [Array.size_push]
  dsimp only
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbSlice_eq _ _) _)]
  rw [bind_go (sampleP_cut_ok coin _ ?h3)]
  case h3 =>
    simp only [Array.size_append, Array.size_push, List.size_toArray, List.length_map, List.length_range]
    omega
  rw [pure_go]
  refine ⟨_, _, _, rfl, ?_, rfl⟩
  simp only [Array.toList_append, Array.toList_push, List.append_assoc, List.cons_append, List.nil_append,
    lcBodies, encs, List.flatMap_cons, List.flatMap_nil, List.append_nil, Claims.toMsg, framed_enc, label_enc,
    absorbSlice_enc, afterSq, Array.toList_map, Array.toList_range, empty_toList, pairLoopB_enc,
    pairLoopT_pending _ coin (l.size + 2)]

def lcFirst (dv : ℕ → F128) (p : ℕ) : List Frame :=
  [.one (dv p), .one (dv (p + 1)), .label "flock-lincheck-v0".toUTF8]

def twoAt (dv : ℕ → F128) (p : ℕ) : List Frame := [.one (dv p), .one (dv (p + 1))]

theorem absPos_lt (l : _root_.Flock.Zk.PadLayout) {j : ℕ} (h : j < l.fa + 2) : absPos l j = j := by
  simp [absPos, h]

theorem absPos_ge (l : _root_.Flock.Zk.PadLayout) {j : ℕ} (h : l.fa + 2 ≤ j) : absPos l j = j - 1 := by
  simp [absPos, show ¬ j < l.fa + 2 by omega]

theorem lcBodies_frame (kLog : ℕ) (c : Cfg) (lay : _root_.Flock.Zk.PadLayout) (mk dv : ℕ → F128)
    (hn : lay.nLc = kLog - 6)
    (hmk : ∀ j, j ≠ lay.fc → j < lay.zp 64 → mk j = dv (absPos lay j) + padAt c (absPos lay j)) :
    lcBodies kLog lay mk (encs [.one (mk lay.fa), .one (mk lay.fb)]) =
      (encs (maskFrames (maskVal c false) lay.fa (lcFirst dv lay.fa)) ++ (Frame.sq 1 1).enc) :: (Frame.sq 1 1).enc ::
        ((List.range (kLog - 6)).map fun j =>
          encs (maskFrames (maskVal c false) (lay.fa + 2 + 2 * j) (twoAt dv (lay.fa + 2 + 2 * j))) ++
            (Frame.sq 1 1).enc) ++
        [encs (maskFrames (maskVal c false) (lay.fa + 2 + 2 * (kLog - 6))
          [.vec ((List.range' (lay.fa + 2 + 2 * (kLog - 6)) 64).map dv)]) ++ (Frame.sq 1 1).enc] := by
  have hfc : lay.fc = lay.fa + 2 := rfl
  have hfb : lay.fb = lay.fa + 1 := rfl
  have hlc : ∀ j t, lay.lc j t = lay.fa + 3 + 2 * j + t := fun j t => by
    show lay.fc + 1 + 2 * j + t = _; omega
  have hzp : ∀ i, lay.zp i = lay.fa + 3 + 2 * (kLog - 6) + i := fun i => by
    show lay.lc lay.nLc 0 + i = _; rw [hlc, hn]; omega
  have hA : mk lay.fa = dv lay.fa + padAt c lay.fa := by
    rw [hmk _ (by omega) (by rw [hzp]; omega), absPos_lt lay (by omega)]
  have hB : mk lay.fb = dv (lay.fa + 1) + padAt c (lay.fa + 1) := by
    rw [hmk _ (by omega) (by rw [hzp]; omega), absPos_lt lay (by omega), hfb]
  have hL : ∀ j < kLog - 6, ∀ t < 2, mk (lay.lc j t) =
      dv (lay.fa + 2 + 2 * j + t) + padAt c (lay.fa + 2 + 2 * j + t) := by
    intro j hj t ht
    rw [hmk _ (by rw [hlc]; omega) (by rw [hlc, hzp]; omega), absPos_ge lay (by rw [hlc]; omega), hlc]
    congr 2 <;> omega
  have hZ : (List.range 64).map (fun i => mk (lay.zp i)) =
      (List.range' (lay.fa + 2 + 2 * (kLog - 6)) 64).map fun q => dv q + padAt c q := by
    rw [List.range'_eq_map_range, List.map_map]
    refine List.map_congr_left fun i hi => ?_
    rw [List.mem_range] at hi
    rw [hmk _ (by rw [hzp]; omega) (by rw [hzp, hzp]; omega), absPos_ge lay (by rw [hzp]; omega), hzp,
      show lay.fa + 3 + 2 * (kLog - 6) + i - 1 = lay.fa + 2 + 2 * (kLog - 6) + i by omega]
    rfl
  unfold lcBodies
  simp only [lcFirst, twoAt, maskFrames, maskVal_false, maskVals_range', hA, hB, hZ, encs, List.flatMap_cons,
    List.flatMap_nil, List.append_nil, List.append_assoc, List.cons.injEq, true_and, List.cons_append]
  refine congrArg (· ++ _) (List.map_congr_left fun j hj => ?_)
  rw [List.mem_range] at hj
  rw [hL j hj 0 (by omega), hL j hj 1 (by omega)]
  rfl

theorem except_bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem answer_alg (cfg : Cfg) (s : Masker) (c : Array F128) {s' : Masker} (hi : s.inner = none)
    (hl : s.l0 = false) (h : s.answer cfg c = .ok s') :
    s' = { s with owed := none, coins := if s.on then s.coins ++ c else s.coins } := by
  unfold Masker.answer at h
  cases ho : s.owed with
  | none => simp [ho] at h
  | some n =>
    simp only [ho, hi, hl] at h
    by_cases hc : c.size = n
    · have hne : (c.size != n) = false := by simp [hc]
      simp only [hne, Bool.false_eq_true, ↓reduceIte] at h
      cases hon : s.on
      · simp only [hon, Bool.not_false, ↓reduceIte, pure, Except.pure, Except.ok.injEq] at h
        subst h
        cases s
        simp only at hi hl ho hon
        subst hi hl hon
        rfl
      · simp only [hon, Bool.not_true, Bool.false_eq_true, ↓reduceIte] at h
        obtain ⟨_, _, h⟩ := except_bind_ok h
        simp only [pure, Except.pure, Except.ok.injEq] at h
        subst h
        cases s
        simp only at hi hl ho hon
        subst hi hl hon
        rfl
    · have hne : (c.size != n) = true := by simp [hc]
      simp only [hne, ↓reduceIte] at h
      cases h

namespace LeanRelease

theorem release_of_roundFree {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (st : Bool)
    (v : Msg) {m : Masker} {n : ℕ} {out : ByteArray} {s' : Masker} {fr : Array ℕ} (hA : L.answered x ρ h = .ok m)
    (hn : L.coins x h.length = n)
    (hr : roundFree (L.cfg x ρ h) { m with stopped := st } ⟨v.toArray⟩ n = .ok (out, s', fr)) :
    L.release x ρ h st v = 0 :: out.data.toList := by
  have hs : L.step x ρ h st v = .ok (out, s') := by
    unfold step
    rw [hA]
    show _root_.Flock.ZkServer.Frame.round _ _ _ _ = _
    unfold _root_.Flock.ZkServer.Frame.round
    rw [hn, hr]
    rfl
  unfold release
  rw [hs]

theorem release_zcBind {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (st : Bool)
    (digest cap : ByteArray) {m : Masker} (hA : L.answered x ρ h = .ok m)
    (hc : ∀ p, p < (L.cfg x ρ h).order.size → (L.cfg x ρ h).order[p]! < (L.cfg x ρ h).h.size)
    (hon : m.on = false) (hd : m.done = false) (hi : m.inner = none) (hdg : digest.size < 2 ^ 64)
    (hcap : cap.size < 2 ^ 64) (hb : m.pos ≤ (L.cfg x ρ h).order.size) (hn : L.coins x h.length = 6) :
    L.release x ρ h st
        (encs [.label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap, .label LABEL_ZC.toUTF8] ++
          (Frame.sq 2 6).enc) =
      0 :: (encs (zcBind digest cap (L.cfg x ρ h).root) ++ (Frame.sq 2 6).enc) := by
  obtain ⟨out, fr, hr, ho⟩ := roundFree_zcBind (L.cfg x ρ h) { m with stopped := st } digest cap hc hon hd hi hdg
    hcap hb
  rw [release_of_roundFree L x ρ h st _ hA hn hr, ho]

theorem release_zcOuter {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (st : Bool)
    {m : Masker} (hA : L.answered x ρ h = .ok m)
    (hc : ∀ p, p < (L.cfg x ρ h).order.size → (L.cfg x ρ h).order[p]! < (L.cfg x ρ h).h.size)
    (hon : m.on = true) (hi : m.inner = none) (hk : L.coins x h.length ≠ 0) (hn : L.coins x h.length < 2 ^ 64)
    (hb : m.pos ≤ (L.cfg x ρ h).order.size) :
    L.release x ρ h st (encs [] ++ (Frame.sq 2 (L.coins x h.length)).enc) =
      0 :: (Frame.sq 2 (L.coins x h.length)).enc := by
  obtain ⟨out, fr, hr, ho⟩ := roundFree_zcOuter (L.cfg x ρ h) { m with stopped := st } _ hc hon hi hk hn hb
  rw [release_of_roundFree L x ρ h st _ hA rfl hr, ho]

theorem release_zcR1 {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (st : Bool)
    (dv : ℕ → F128) {m : Masker} (hA : L.answered x ρ h = .ok m)
    (hc : ∀ p, p < (L.cfg x ρ h).order.size → (L.cfg x ρ h).order[p]! < (L.cfg x ρ h).h.size)
    (hon : m.on = true) (hi : m.inner = none) (hpos : m.pos = 0) (hb : 128 ≤ (L.cfg x ρ h).order.size)
    (hn : L.coins x h.length = 1) :
    L.release x ρ h st (encs (zcR1 dv) ++ (Frame.sq 1 1).enc) =
      0 :: (encs (maskFrames (maskVal (L.cfg x ρ h) st) 0 (zcR1 dv)) ++ (Frame.sq 1 1).enc) := by
  obtain ⟨out, fr, hr, ho⟩ := roundFree_zcR1 (L.cfg x ρ h) { m with stopped := st } dv hc hon hi hpos hb
  rw [release_of_roundFree L x ρ h st _ hA hn hr, ho]

theorem release_zcMl {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (st : Bool)
    (dv : ℕ → F128) (i : ℕ) {m : Masker} (hA : L.answered x ρ h = .ok m)
    (hc : ∀ p, p < (L.cfg x ρ h).order.size → (L.cfg x ρ h).order[p]! < (L.cfg x ρ h).h.size)
    (hon : m.on = true) (hi : m.inner = none) (hpos : m.pos = 128 + 2 * i)
    (hb : 128 + 2 * i + 2 ≤ (L.cfg x ρ h).order.size) (hn : L.coins x h.length = 1) :
    L.release x ρ h st (encs (zcMl dv i) ++ (Frame.sq 1 1).enc) =
      0 :: (encs (maskFrames (maskVal (L.cfg x ρ h) st) (128 + 2 * i) (zcMl dv i)) ++ (Frame.sq 1 1).enc) := by
  obtain ⟨out, fr, hr, ho⟩ := roundFree_zcMl (L.cfg x ρ h) { m with stopped := st } dv i hc hon hi hpos hb
  rw [release_of_roundFree L x ρ h st _ hA hn hr, ho]

theorem release_on {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (st : Bool)
    (fs : List Frame) (k : UInt8) {m : Masker} (hA : L.answered x ρ h = .ok m)
    (hc : ∀ p, p < (L.cfg x ρ h).order.size → (L.cfg x ρ h).order[p]! < (L.cfg x ρ h).h.size)
    (hon : m.on = true) (hi : m.inner = none) (hm : ∀ f ∈ fs, midFrame f) (hp : ∀ f ∈ fs, plainFrame f)
    (hk : k = 2 ∨ (k = 1 ∧ L.coins x h.length = 1)) (h0 : ¬ (L.coins x h.length = 0 ∧ k = 2))
    (hn : L.coins x h.length < 2 ^ 64) (hb : m.pos + nAbs fs ≤ (L.cfg x ρ h).order.size) :
    L.release x ρ h st (encs fs ++ (Frame.sq k (L.coins x h.length)).enc) =
      0 :: (encs (maskFrames (maskVal (L.cfg x ρ h) st) m.pos fs) ++ (Frame.sq k (L.coins x h.length)).enc) := by
  obtain ⟨out, fr, hr, ho⟩ := roundFree_on (L.cfg x ρ h) { m with stopped := st } fs k _ hc hon hi hm hp hk h0 hn hb
  rw [release_of_roundFree L x ρ h st _ hA rfl hr, ho]

end LeanRelease


open Claims.ZKMal (dropDom domLen RefSession refK allOk)
open FlockSoundness.Refine (header_data)

def noDom : Frame → Prop
  | .dom _ => False
  | _ => True

theorem maskFrames_noDom (g : ℕ → F128 → F128) : ∀ (p : ℕ) (G : List Frame), ∀ f ∈ maskFrames g p G, noDom f
  | _, [] => by simp [maskFrames]
  | p, f :: G => by
    cases f with
    | dom b => exact maskFrames_noDom g p G
    | label b =>
      intro y hy
      simp only [maskFrames, List.mem_cons] at hy
      rcases hy with rfl | hy
      · trivial
      · exact maskFrames_noDom g p G y hy
    | bytes b =>
      intro y hy
      simp only [maskFrames, List.mem_cons] at hy
      rcases hy with rfl | hy
      · trivial
      · exact maskFrames_noDom g p G y hy
    | one v =>
      intro y hy
      simp only [maskFrames, List.mem_cons] at hy
      rcases hy with rfl | hy
      · trivial
      · exact maskFrames_noDom g (p + 1) G y hy
    | vec vs =>
      intro y hy
      simp only [maskFrames, List.mem_cons] at hy
      rcases hy with rfl | hy
      · trivial
      · exact maskFrames_noDom g (p + vs.length) G y hy
    | sq k n =>
      intro y hy
      simp only [maskFrames, List.mem_cons] at hy
      rcases hy with rfl | hy
      · trivial
      · exact maskFrames_noDom g p G y hy

theorem dropDoms_noDom : ∀ (F : List Frame), (∀ f ∈ F, noDom f) → dropDoms F = F
  | [], _ => rfl
  | f :: F, h => by
    have ih := dropDoms_noDom F fun g hg => h g (List.mem_cons_of_mem _ hg)
    have hf := h f List.mem_cons_self
    cases f with
    | dom b => exact hf.elim
    | label b | bytes b | one v | vec vs | sq k n => exact congrArg (List.cons _) ih

theorem enc_head (f : Frame) (rest : List UInt8) : (f.enc ++ rest)[0]! = f.op := by
  simp [Frame.enc, header_data]

theorem dropDom_noDom (M : List Frame) (k : UInt8) (n : ℕ) (hM : ∀ f ∈ M, noDom f) :
    dropDom (0 :: (encs M ++ (Frame.sq k n).enc)) = 0 :: (encs M ++ (Frame.sq k n).enc) := by
  have hop : (encs M ++ (Frame.sq k n).enc)[0]! ≠ 1 := by
    cases M with
    | nil =>
      rw [show encs [] ++ (Frame.sq k n).enc = (Frame.sq k n).enc ++ [] by simp [encs], enc_head]
      show (4 : UInt8) ≠ 1
      decide
    | cons f M =>
      rw [show encs (f :: M) ++ (Frame.sq k n).enc = f.enc ++ (encs M ++ (Frame.sq k n).enc) by
        simp [encs, List.append_assoc], enc_head]
      have hf := hM f List.mem_cons_self
      cases f with
      | dom b => exact hf.elim
      | label b => show (2 : UInt8) ≠ 1; decide
      | one v => show (3 : UInt8) ≠ 1; decide
      | vec vs => show (3 : UInt8) ≠ 1; decide
      | bytes b => show (5 : UInt8) ≠ 1; decide
      | sq k' n' => show (4 : UInt8) ≠ 1; decide
  show (if (0 : UInt8) = 0 ∧ _ then _ else _) = _
  exact ite_eq_right fun h => hop h.2

theorem dropDom_dom (d : ByteArray) (F : List Frame) (r : List UInt8) (hd : d.size < 2 ^ 64) :
    dropDom (0 :: (encs (.dom d :: F) ++ r)) = 0 :: (encs F ++ r) := by
  have he : encs (.dom d :: F) ++ r = (Frame.dom d).enc ++ (encs F ++ r) := by
    simp [encs, List.append_assoc]
  obtain ⟨h0, -, hl⟩ := header_at (encs (.dom d :: F) ++ r) 0 (.dom d) (encs F ++ r) (by rw [List.drop_zero, he]) hd
  rw [get!_list] at h0
  have hlen : domLen (encs (.dom d :: F) ++ r) = (Frame.dom d).enc.length := by
    rw [show (0 : ℕ) + 8 = 8 from rfl] at hl
    simp only [domLen]
    rw [hl, Frame.enc_length]
    simp only [Frame.len, Frame.payLen]
    omega
  show (if (0 : UInt8) = 0 ∧ (encs (.dom d :: F) ++ r)[0]! = 1 then
      (0 : UInt8) :: (encs (.dom d :: F) ++ r).drop (domLen (encs (.dom d :: F) ++ r)) else
      0 :: (encs (.dom d :: F) ++ r)) = _
  rw [ite_eq_left ⟨rfl, h0⟩, hlen, he, List.drop_left]

namespace LeanRelease

def RoundFrames {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (st : Bool)
    (u msg : Msg) : Prop :=
  ∃ (m : Masker) (G D : List Frame) (k : UInt8),
    L.answered x ρ h = .ok m ∧
    (∀ p, p < (L.cfg x ρ h).order.size → (L.cfg x ρ h).order[p]! < (L.cfg x ρ h).h.size) ∧
    m.on = true ∧ m.inner = none ∧ (∀ f ∈ G, midFrame f) ∧ (∀ f ∈ G, plainFrame f) ∧
    (k = 2 ∨ (k = 1 ∧ L.coins x h.length = 1)) ∧ ¬ (L.coins x h.length = 0 ∧ k = 2) ∧
    L.coins x h.length < 2 ^ 64 ∧ m.pos + nAbs G ≤ (L.cfg x ρ h).order.size ∧
    (D = [] ∨ ∃ d, D = [.dom d] ∧ d.size < 2 ^ 64) ∧
    u = encs G ++ (Frame.sq k (L.coins x h.length)).enc ∧
    msg = 0 :: (encs (D ++ maskFrames (maskVal (L.cfg x ρ h) st) m.pos G) ++ (Frame.sq k (L.coins x h.length)).enc)

theorem release_of_roundFrames {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg))
    (st : Bool) (u msg : Msg) (hR : L.RoundFrames x ρ h st u msg) : L.release x ρ h st u = dropDom msg := by
  obtain ⟨m, G, D, k, hA, hc, hon, hi, hm, hp, hk, h0, hn, hb, hD, rfl, rfl⟩ := hR
  rw [L.release_on x ρ h st G k hA hc hon hi hm hp hk h0 hn hb]
  rcases hD with rfl | ⟨d, rfl, hd⟩
  · rw [List.nil_append, dropDom_noDom _ _ _ (maskFrames_noDom _ _ _)]
  · rw [List.singleton_append, dropDom_dom d _ _ hd]

def RoundHonest {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (u msg : Msg) : Prop :=
  ∃ (m : Masker) (D F : List Frame) (k : UInt8),
    L.answered x ρ h = .ok m ∧
    (∀ p, p < (L.cfg x ρ h).order.size → (L.cfg x ρ h).order[p]! < (L.cfg x ρ h).h.size) ∧
    m.on = true ∧ m.inner = none ∧ (∀ f ∈ F, midFrame f) ∧ (∀ f ∈ F, plainFrame f) ∧ (∀ f ∈ F, noDom f) ∧
    (k = 2 ∨ (k = 1 ∧ L.coins x h.length = 1)) ∧ ¬ (L.coins x h.length = 0 ∧ k = 2) ∧
    L.coins x h.length < 2 ^ 64 ∧ m.pos + nAbs F ≤ (L.cfg x ρ h).order.size ∧
    (D = [] ∨ ∃ d, D = [.dom d] ∧ d.size < 2 ^ 64) ∧
    u = encs (unmaskFrames (L.cfg x ρ h) m.pos (D ++ F)) ++ (Frame.sq k (L.coins x h.length)).enc ∧
    msg = 0 :: (encs (D ++ F) ++ (Frame.sq k (L.coins x h.length)).enc)

theorem roundFrames_of_honest {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg))
    (u msg : Msg) (hR : L.RoundHonest x ρ h u msg) : L.RoundFrames x ρ h false u msg := by
  obtain ⟨m, D, F, k, hA, hc, hon, hi, hm, hp, hnd, hk, h0, hn, hb, hD, rfl, rfl⟩ := hR
  have hDm : ∀ f ∈ D ++ F, midFrame f := by
    rcases hD with rfl | ⟨d, rfl, hd⟩
    · simpa using hm
    · intro f hf
      simp only [List.singleton_append, List.mem_cons] at hf
      rcases hf with rfl | hf
      · exact hd
      · exact hm f hf
  have hDp : ∀ f ∈ D ++ F, plainFrame f := by
    rcases hD with rfl | ⟨d, rfl, -⟩
    · simpa using hp
    · intro f hf
      simp only [List.singleton_append, List.mem_cons] at hf
      rcases hf with rfl | hf
      · trivial
      · exact hp f hf
  have hDn : nAbs (D ++ F) = nAbs F := by
    rcases hD with rfl | ⟨d, rfl, -⟩ <;> rfl
  have hDd : dropDoms (D ++ F) = F := by
    rcases hD with rfl | ⟨d, rfl, -⟩
    · exact dropDoms_noDom F hnd
    · exact dropDoms_noDom F hnd
  refine ⟨m, unmaskFrames (L.cfg x ρ h) m.pos (D ++ F), D, k, hA, hc, hon, hi, unmask_mid _ _ _ hDm,
    unmask_plain _ _ _ hDp, hk, h0, hn, by rw [nAbs_unmask, hDn]; exact hb, hD, rfl, ?_⟩
  rw [maskFrames_unmask, hDd]

end LeanRelease

namespace LeanRelease

def LiveFrames (S : RefSession) (vs : S.SwVals) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (H : Claims.Msg → Claims.Msg) (wk : L.toSrv.Worker) (x : Claims.Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : S.Nonce) (t : S.Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Claims.Msg) (ops : List (Claims.Msg × Claims.Msg))
    (v : Claims.Msg),
    0 < ops.length → ops.length < refK x → L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = true →
    L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false) →
    L.RoundFrames x (ν, t) ((m₀, ρ₀) :: ops) false v (S.swMsgAt x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2))

theorem liveRoundEq_of_frames (S : RefSession) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Claims.Msg → Claims.Msg) (wk : L.toSrv.Worker)
    (x : Claims.Msg) (w : Flock.Wit) (hF : LiveFrames S vs L H wk x w) : LiveRoundEq S vs L H wk x w :=
  fun ν t κ m₀ ρ₀ ops v h0 hlt hl hok hr hv =>
    L.release_of_roundFrames x _ _ false v _ (hF ν t κ m₀ ρ₀ ops v h0 hlt hl hok hr hv)

def LiveHonest (S : RefSession) (vs : S.SwVals) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (H : Claims.Msg → Claims.Msg) (wk : L.toSrv.Worker) (x : Claims.Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : S.Nonce) (t : S.Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Claims.Msg) (ops : List (Claims.Msg × Claims.Msg))
    (v : Claims.Msg),
    0 < ops.length → ops.length < refK x → L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = true →
    L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false) →
    L.RoundHonest x (ν, t) ((m₀, ρ₀) :: ops) v (S.swMsgAt x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2))

theorem liveRoundEq_of_honest (S : RefSession) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Claims.Msg → Claims.Msg) (wk : L.toSrv.Worker)
    (x : Claims.Msg) (w : Flock.Wit) (hF : LiveHonest S vs L H wk x w) : LiveRoundEq S vs L H wk x w :=
  liveRoundEq_of_frames S vs L H wk x w fun ν t κ m₀ ρ₀ ops v h0 hlt hl hok hr hv =>
    L.roundFrames_of_honest x _ _ v _ (hF ν t κ m₀ ρ₀ ops v h0 hlt hl hok hr hv)

def FillFrames (S : RefSession) (vs : S.SwVals) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (H : Claims.Msg → Claims.Msg) (wk : L.toSrv.Worker) (x : Claims.Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : S.Nonce) (t : S.Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Claims.Msg) (ops : List (Claims.Msg × Claims.Msg)),
    0 < ops.length → ops.length < refK x → L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = false →
    ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = true →
    L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    L.RoundFrames x (ν, t) ((m₀, ρ₀) :: ops) true
      (L.scheduled x w (ν, t) ((m₀, ρ₀) :: ops) (computedOf (L.computed x w (ν, t) ((m₀, ρ₀) :: ops))))
      (S.swMsgAt x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2))

theorem fillRoundEq_of_frames (S : RefSession) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Claims.Msg → Claims.Msg) (wk : L.toSrv.Worker)
    (x : Claims.Msg) (w : Flock.Wit) (hF : FillFrames S vs L H wk x w) : FillRoundEq S vs L H wk x w :=
  fun ν t κ m₀ ρ₀ ops h0 hlt hl hok hr =>
    L.release_of_roundFrames x _ _ true _ _ (hF ν t κ m₀ ρ₀ ops h0 hlt hl hok hr)

end LeanRelease

end Claims.ZKServer
