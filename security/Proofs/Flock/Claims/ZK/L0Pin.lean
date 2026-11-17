module

public import Proofs.Flock.Claims.ZK.FinExe
public import Proofs.Flock.Claims.ZK.FinishSwLig
public import Proofs.Flock.Claims.ZK.FrameOpenCut
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

/-! Level 0's bodies pin the source's masked values. -/

@[expose] public section

namespace FlockComplete.LigLock

open Flock Transcript Replay FlockLevel3
open FlockSoundness.Refine (Frame encs)

theorem f128_add_left_cancel {a x y : F128} (h : a + x = a + y) : x = y := by
  apply FlockSoundness.Discharge.Composed.Zerocheck.eq_of_of
  have h' := congrArg GF128.of h
  rw [of_add, of_add] at h'
  exact add_left_cancel h'

theorem f128_eq_add_add (m c : F128) : m = c + (m + c) := by
  apply FlockSoundness.Discharge.Composed.Zerocheck.eq_of_of
  rw [of_add, of_add, add_left_comm, CharTwo.add_self_eq_zero, add_zero]

theorem oodFr_val {y y' : F128} {nb nb' : ByteArray} {r r' : List UInt8}
    (h : encs (Claims.ZKServer.oodFr y nb) ++ r = encs (Claims.ZKServer.oodFr y' nb') ++ r') : y = y' := by
  simp only [Claims.ZKServer.oodFr, encs, List.flatMap_cons, List.flatMap_nil, List.append_nil,
    List.append_assoc] at h
  have := (Frame.enc_append_inj (f := .one y) (f' := .one y') (by simp [Frame.WF, Frame.len])
    (by simp [Frame.WF, Frame.len]) h).1
  injection this

theorem pairFr_val {a b a' b' : F256} {r r' : List UInt8}
    (h : encs (Claims.ZKServer.pairFr a b) ++ r = encs (Claims.ZKServer.pairFr a' b') ++ r') : a = a' ∧ b = b' := by
  simp only [Claims.ZKServer.pairFr, encs, List.flatMap_cons, List.flatMap_nil, List.append_nil,
    List.append_assoc] at h
  obtain ⟨ha, h₂⟩ := Frame.enc_append_inj (f := .vec [a.c0, a.c1]) (f' := .vec [a'.c0, a'.c1])
    (by simp [Frame.WF, Frame.len]) (by simp [Frame.WF, Frame.len]) h
  obtain ⟨hb, -⟩ := Frame.enc_append_inj (f := .vec [b.c0, b.c1]) (f' := .vec [b'.c0, b'.c1])
    (by simp [Frame.WF, Frame.len]) (by simp [Frame.WF, Frame.len]) h₂
  simp only [Frame.vec.injEq, List.cons.injEq, and_true] at ha hb
  obtain ⟨a0, a1⟩ := a
  obtain ⟨b0, b1⟩ := b
  obtain ⟨a0', a1'⟩ := a'
  obtain ⟨b0', b1'⟩ := b'
  obtain ⟨rfl, rfl⟩ := ha
  obtain ⟨rfl, rfl⟩ := hb
  exact ⟨rfl, rfl⟩

theorem l0BodyAt_pads {coin : ℕ → ℕ → F128} {p0 : ℕ} {P P' : List UInt8} {T T' : F128} {cap cap' : ByteArray}
    {sch : Schedule} {t b : Array F128} {lay : Zk.PadLayout} {h h' : Array F128} {rep : ℕ} {extra : ℕ → ℕ → F128}
    {nonce nonce' : ℕ → ℕ → UInt64} (hood : lay.nOod0 = 1) (hlk : lay.k0 = sch.k0)
    (hb : ∀ i, 1 ≤ i → i ≤ sch.k0 + 2 →
      l0BodyAt coin p0 P T cap sch t b lay h rep extra nonce i =
        l0BodyAt coin p0 P' T' cap' sch t b lay h' rep extra nonce' i)
    {k : ℕ} (hlo : lay.zp 64 ≤ k) (hhi : k < lay.hab) : h[k]! = h'[k]! := by
  simp only [Zk.PadLayout.hab] at hhi
  rcases Nat.lt_or_ge k (lay.zp 64 + 1) with hk1 | hk1
  · obtain rfl : k = lay.lgOod 0 := by simp only [Zk.PadLayout.lgOod]; omega
    have e := hb 1 le_rfl (by omega)
    rw [l0BodyAt_one, l0BodyAt_one] at e
    exact f128_add_left_cancel (oodFr_val e)
  rcases Nat.lt_or_ge k (lay.zp 64 + 1 + 4 * sch.k0) with hk2 | hk2
  · obtain ⟨j, c, hj, hc, rfl⟩ : ∃ j c, j < sch.k0 ∧ c < 4 ∧ k = lay.lgMsg j c :=
      ⟨(k - (lay.zp 64 + 1)) / 4, (k - (lay.zp 64 + 1)) % 4, by omega, by omega, by
        simp only [Zk.PadLayout.lgMsg]; omega⟩
    have e := hb (j + 2) (by omega) (by omega)
    rw [l0BodyAt_msg hj, l0BodyAt_msg hj] at e
    obtain ⟨e1, e2⟩ := pairFr_val e
    simp only [l0Msg, maskMsg] at e1 e2
    interval_cases c
    · exact f128_add_left_cancel (congrArg F256.c0 e1)
    · exact f128_add_left_cancel (congrArg F256.c1 e1)
    · exact f128_add_left_cancel (congrArg F256.c0 e2)
    · exact f128_add_left_cancel (congrArg F256.c1 e2)
  · obtain ⟨i, c, hi, hc, rfl⟩ : ∃ i c, i < 2 ∧ c < 2 ∧ k = lay.lgE i c :=
      ⟨(k - (lay.zp 64 + 1 + 4 * sch.k0)) / 2, (k - (lay.zp 64 + 1 + 4 * sch.k0)) % 2, by omega, by omega, by
        simp only [Zk.PadLayout.lgE]; omega⟩
    have e := hb (sch.k0 + 2) (by omega) le_rfl
    rw [l0BodyAt_e, l0BodyAt_e] at e
    obtain ⟨e1, e2⟩ := pairFr_val e
    interval_cases i <;> interval_cases c
    · exact f128_add_left_cancel (congrArg F256.c0 e1)
    · exact f128_add_left_cancel (congrArg F256.c1 e1)
    · exact f128_add_left_cancel (congrArg F256.c0 e2)
    · exact f128_add_left_cancel (congrArg F256.c1 e2)

def padsL0 (coin : ℕ → ℕ → F128) (p0 : ℕ) (sch : Schedule) (t b : Array F128) (lay : Zk.PadLayout)
    (extra : ℕ → ℕ → F128) (rep : ℕ) (src : RepSrc) : Array F128 :=
  (Array.range lay.hab).map fun k => src.masked k + l0Clear coin p0 sch t b lay extra rep k

theorem ligeritoCut_cutL0 (coin : ℕ → ℕ → F128) (hs : Hashes PS) (ms : MerkleScheme) (sch : Schedule) (T : F128)
    (cm0 : Commit) (t b : Array F128) (lay : Zk.PadLayout) (rep : ℕ) (extra : ℕ → ℕ → F128) (src : RepSrc)
    (tr : Transcript) (l : Array Claims.Msg) {j : ℕ} (hj : j < sch.k0 + 4) (hk : 1 ≤ sch.k0) (hn : sch.n ≠ 0)
    (hlv : 2 ≤ sch.levels.size) (hood : lay.nOod0 = 1) (hlk : lay.k0 = sch.k0) :
    CutStop (fun L => L.toList = l.toList ++ (List.range (j + 1)).map
        (l0BodyAt coin l.size tr.pending.data.toList T (flat cm0.cap) sch t b lay
          (padsL0 coin l.size sch t b lay extra rep src) rep extra fun _ _ => 0))
      (((ligeritoCut (asksCut coin (l.size + j)) hs ms sch T (eraseC cm0) b lay src).run tr).go l) := by
  have H : LigHyp coin (l.size + j) l.size ms sch t b lay (padsL0 coin l.size sch t b lay extra rep src) rep 0 extra
      (fun _ _ => 0) (fun _ _ => ByteArray.empty) src :=
    ligHyp_of_l0 coin (l.size + j) l.size ms sch t b lay _ rep 0 extra (fun _ _ => 0) (fun _ _ => ByteArray.empty) src
      hk hn hlv hood hlk
      (fun k _ hk' => by
        rw [padsL0, FlockSoundness.Discharge.Composed.Zerocheck.get_map_range _ hk']
        exact f128_eq_add_add _ _)
      (fun h => absurd h (by omega)) (fun h => absurd h (by omega)) (fun h => absurd h (by omega))
  rw [ligeritoCut_alike coin (l.size + j) hs ms sch T cm0 t b lay _ rep 0 extra (fun _ _ => 0)
    (fun _ _ => ByteArray.empty) src tr l H]
  exact cutStop_bind (ligeritoPZk_cutL0 coin hs ms sch T cm0 t b lay _ rep 0 extra (fun _ _ => 0)
    (fun _ _ => ByteArray.empty) (fun _ _ => 0) tr l hj hk hn)

theorem masked_of_cutL0 {coin : ℕ → ℕ → F128} {p0 : ℕ} {P P' : List UInt8} {T T' : F128} {cap cap' : ByteArray}
    {sch : Schedule} {t b : Array F128} {lay : Zk.PadLayout} {h : Array F128} {rep : ℕ} {extra : ℕ → ℕ → F128}
    {nonce : ℕ → ℕ → UInt64} {src : RepSrc} (hood : lay.nOod0 = 1) (hlk : lay.k0 = sch.k0)
    (hb : ∀ i, 1 ≤ i → i ≤ sch.k0 + 2 →
      l0BodyAt coin p0 P T cap sch t b lay (padsL0 coin p0 sch t b lay extra rep src) rep extra (fun _ _ => 0) i =
        l0BodyAt coin p0 P' T' cap' sch t b lay h rep extra nonce i)
    {k : ℕ} (hlo : lay.zp 64 ≤ k) (hhi : k < lay.hab) :
    src.masked k = l0Clear coin p0 sch t b lay extra rep k + h[k]! := by
  have e := l0BodyAt_pads hood hlk hb hlo hhi
  rw [padsL0, FlockSoundness.Discharge.Composed.Zerocheck.get_map_range _ hhi] at e
  rw [← e]
  exact f128_eq_add_add _ _

theorem cutStop_of_eraseZ {Q : Array Claims.Msg → Prop} {m m' : TMW PS Zk.ZkProof} (he : eraseZ <$> m = eraseZ <$> m')
    {t : Transcript} {l : Array Claims.Msg} (h : CutStop Q ((m.run t).go l)) : CutStop Q ((m'.run t).go l) := by
  obtain ⟨L, hm, hQ⟩ := h
  refine ⟨L, ?_, hQ⟩
  have e := congrArg (fun z => (z.run t).go l) he
  simp only [StateT.run_map, go_map', hm] at e
  rcases h' : (m'.run t).go l with ⟨r, l''⟩
  rw [h'] at e
  cases r with
  | error err =>
    simp only [Except.map, Prod.mk.injEq, Except.error.injEq] at e
    obtain ⟨h1, h2⟩ := e
    subst h1 h2
    rfl
  | ok q => simp [Except.map] at e

theorem repCut_cutL0 (coin : ℕ → ℕ → F128) (hs : Hashes PS) (st : Setup) (sch : Schedule) (rep : ℕ) (dom : ByteArray)
    (cm0 : Commit) (extras : Array Claim) (ic : InnerCommit) (src : RepSrc) (prior : Array (List F128))
    (words : List ℕ) (t : Array F128) (rp : ℕ) (extra : ℕ → ℕ → F128) (tr : Transcript) (l : Array Claims.Msg)
    {j : ℕ} (hj : j < sch.k0 + 4) (hk : 1 ≤ sch.k0) (hn : sch.n ≠ 0) (hlv : 2 ≤ sch.levels.size)
    {res} {tr₁ : Transcript} {l₁ : Array Claims.Msg}
    (hp : ((piopCut (asksCut coin (l₁.size + j)) st (Zk.PadLayout.new st.m st.kLog sch.k0) src (flat cm0.cap) ic.root
      extras prior words).run tr).go l = (.ok (res, tr₁), l₁)) :
    CutStop (fun L => L.toList = l₁.toList ++ (List.range (j + 1)).map
        (l0BodyAt coin l₁.size tr₁.pending.data.toList res.2.2.2.1 (flat cm0.cap) sch t
          (basisVals res.2.2.2.2 (st.m - 7)) (Zk.PadLayout.new st.m st.kLog sch.k0)
          (padsL0 coin l₁.size sch t (basisVals res.2.2.2.2 (st.m - 7)) (Zk.PadLayout.new st.m st.kLog sch.k0) extra
            rp src) rp extra fun _ _ => 0))
      (((repCut (asksCut coin (l₁.size + j)) hs st sch rep dom cm0 extras ic src prior words).run tr).go l) := by
  have hcap : (eraseC cm0).cap = cm0.cap := rfl
  have hlay : (eraseC cm0).layers = cm0.layers := rfl
  have he := Claims.ZKMal.Masked.FinRep.repCut_eraseC (asksCut coin (l₁.size + j)) hs st sch rep dom hcap hlay extras
    ic src prior words
  refine cutStop_of_eraseZ he ?_
  obtain ⟨zp, lp, ss, T, sws⟩ := res
  unfold repCut
  dsimp only
  rw [show flat (eraseC cm0).cap = flat cm0.cap from rfl, bind_go hp]
  dsimp only
  exact cutStop_bind (ligeritoCut_cutL0 coin hs st.merkle sch T cm0 t _ _ rp extra src tr₁ l₁ hj hk hn hlv rfl rfl)

end FlockComplete.LigLock
