module

public import Proofs.Flock.Claims.Bridge.Custody
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
import Mathlib.Topology.Sheaves.Init

/-! `Deploy.V false`'s custody checks, as `RecordsLiveJ`. -/

@[expose] public section

namespace Claims.Bridge.CustodyPlain

open _root_.Flock FlockDeploy FlockSoundness FlockSoundness.Discharge Claims.Bridge.Custody
open FlockSoundness.Refine (ebind_ok ptsOf)
open Lean (Json)

theorem setupOf_plainInputs {r : Readable.Read} {spec : SessionSpec} {sts : Array Stmt} {record : String}
    (hs : r.setup = .ok (spec, sts)) (hzk : r.zk = false) (hpi : r.publicInputs = none)
    (hdraw : Exec.recordDraw record = r.draw) :
    Exec.setupOf r.I record = .ok (spec, sts) := by
  rw [← hs]
  simp only [Exec.setupOf, Readable.Read.setup, Readable.Read.tags, Readable.Read.coins, hzk, hpi, hdraw,
    Bool.false_eq_true, ↓reduceIte]

theorem verify_nil_plain (spec : SessionSpec) (tabs : Array Setup) (record : String) :
    verify spec tabs record #[] ≠ .ok () := by
  intro h
  unfold verify verifyFiles at h
  simp only [] at h
  by_cases c1 : (tabs.size != spec.tables.size || tabs.size == 0) = true
  · rw [ite_eq_left c1] at h
    obtain ⟨_, h', _⟩ := ebind_ok h
    cases h'
  rw [ite_eq_right c1] at h
  obtain ⟨sess, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨schs, -, h⟩ := ebind_ok h
  have c3 : ((#[] : Array ByteArray).size != spec.repStreams.size) = true := by
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, beq_iff_eq, not_or, Decidable.not_not] at c1
    simp only [List.size_toArray, List.length_nil, repStreams_size, bne_iff_ne, ne_eq]
    omega
  rw [ite_eq_left c3] at h
  obtain ⟨_, h', _⟩ := ebind_ok h
  cases h'

def RecordsLiveAtP (I : Exec.Inputs) (d : Json) {J mPts : ℕ} (o : Tables.LiveOutJ J mPts) (t : Exec.Transcript) :
    Prop :=
  Exec.Accepts I t →
    Exec.recordDraw t.record = some d ∧ t.proofs = o.proofs ∧
    ∀ spec sts sess, Exec.setupOf I t.record = .ok (spec, sts) →
      Record.decode spec t.record = .ok sess →
      ptsOf mPts sess.points = o.pts ∧
      ∀ j : Fin J, sess.rootB[j.val]! = o.rootB[j.val]! ∧ ∀ r : Fin 2, ∃ s : StreamRec,
        sess.streams.find? (fun x => x.name == toString spec.tables[j.val]! ++ toString "/rep" ++ toString r.val) =
          some s ∧ s.rounds.toList = o.reps j r

theorem recordsLiveJ_iff {n mPts : ℕ} {Reg : Type} (I : Exec.Inputs) (dj : Finset (Fin n) → Json)
    (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → Tables.LiveOutJ (Jf S R) mPts → Exec.Transcript) :
    DecodeZLJ.RecordsLiveJ (mPts := mPts) I dj Jf wr ↔ ∀ S R o, RecordsLiveAtP I (dj S) o (wr S R o) := Iff.rfl

theorem recordsLiveAtP_of_checks {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    {J mPts : ℕ} {d : Json} (hs : s.setup = .ok (spec, sts)) (hzk : s.read.zk = false)
    (hpi : s.read.publicInputs = none) (hdraw : Exec.recordDraw f.record = s.read.draw)
    (hd : s.read.draw = some d) (hJ : J ≤ spec.tables.size) (hc : checks false s spec x link c₁ mids f = .ok ()) :
    RecordsLiveAtP s.read.I d (viewOf J mPts spec link c₁ f) ⟨f.record, f.proofs⟩ := by
  intro _
  refine ⟨hdraw.trans hd, rfl, fun spec' sts' sess hset hdec => ?_⟩
  rw [setupOf_plainInputs (Ins.setup_read hs) hzk hpi hdraw] at hset
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hset)
  obtain ⟨-, lv, hlv, hcus⟩ := checks_ok hc
  obtain ⟨hroot, -, hpts, hrounds⟩ := custody_ok hcus
  obtain ⟨eroot, epts, esize, estr⟩ := decode_live hdec hlv
  have hD : liveD spec f.record = lv := by simp [liveD, hlv]
  refine ⟨by simp only [viewOf]; rw [epts, hpts], fun j => ⟨?_, fun r => ?_⟩⟩
  · show sess.rootB[j.val]! = link.1.toArray[j.val]!
    rw [eroot, ← hroot]
  · obtain ⟨hq, hw⟩ := repStreams_getElem spec (i := j.val) (r := r.val) (by omega) r.isLt
    obtain ⟨y, hy, hyr⟩ := estr _ hq
    rw [hw] at hy
    refine ⟨y, hy, ?_⟩
    show y.rounds.toList = ((liveD spec f.record).streams[2 * j.val + r.val]!).toList.map (withSha spec)
    rw [hD]
    have hq' : 2 * j.val + r.val < lv.streams.size := esize ▸ hq
    refine forall₂_map_eq hyr fun rd rd' hmem ⟨hn, hcn, hb, hsha⟩ => ?_
    obtain ⟨am, -, hbytes, -⟩ := forall₂_mem_left hrounds rd' (List.mem_flatMap.2
      ⟨lv.streams[2 * j.val + r.val]!, by rw [getElem!_pos lv.streams _ hq']; exact Array.getElem_mem_toList hq',
        hmem⟩)
    rcases rd with ⟨n, sha, coins, bytes⟩
    rcases rd' with ⟨n', sha', coins', bytes'⟩
    simp only at hn hcn hb hsha hbytes
    subst hn hcn hb hbytes
    simp [withSha, hsha _ rfl]

theorem accepts_of_decide_plain {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} {J mPts : ℕ} {d : Json}
    (ha : act false x as cs = .decide s spec sts link c₁ mids f)
    (hacc : ((next false x as cs).eval sha512).1 = .acc) (hpi : s.read.publicInputs = none)
    (hd : s.read.draw = some d) (hJ : J ≤ spec.tables.size) :
    Exec.Accepts s.read.I ⟨f.record, f.proofs⟩ ∧
      RecordsLiveAtP s.read.I d (viewOf J mPts spec link c₁ f) ⟨f.record, f.proofs⟩ := by
  have hdraw := recordSample_of_decide ha hacc
  obtain ⟨hv, hc⟩ := (next_decide_sha512 ha).1 hacc
  simp only [Bool.false_eq_true, ite_false] at hv
  obtain ⟨hins, hs⟩ := setupOf_some (act_decide ha)
  have hzk : s.read.zk = false := readOf_zk (x := x) (by simp [readOf, hins])
  exact ⟨⟨spec, sts, setupOf_plainInputs (Ins.setup_read hs) hzk hpi hdraw, hv⟩,
    recordsLiveAtP_of_checks hs hzk hpi hdraw hd hJ hc⟩

open Classical in
/-- Where no accepted transcript records the outcome, it writes a session without proof files, which `verify`
refuses (`verify_nil_plain`). -/
noncomputable def wrP {n mPts : ℕ} {Reg : Type} (I : Exec.Inputs) (dj : Finset (Fin n) → Json)
    (Jf : Finset (Fin n) → Reg → ℕ) (S : Finset (Fin n)) (R : Reg) (o : Tables.LiveOutJ (Jf S R) mPts) :
    Exec.Transcript :=
  if h : ∃ t, Exec.Accepts I t ∧ RecordsLiveAtP I (dj S) o t then h.choose else ⟨"", #[]⟩

theorem not_accepts_nil (I : Exec.Inputs) (record : String) : ¬ Exec.Accepts I ⟨record, #[]⟩ := by
  rintro ⟨spec, sts, -, h⟩
  exact verify_nil_plain _ _ _ h

theorem recordsLiveJ_wrP {n mPts : ℕ} {Reg : Type} (I : Exec.Inputs) (dj : Finset (Fin n) → Json)
    (Jf : Finset (Fin n) → Reg → ℕ) : DecodeZLJ.RecordsLiveJ (mPts := mPts) I dj Jf (wrP I dj Jf) := by
  refine (recordsLiveJ_iff I dj Jf _).2 fun S R o => ?_
  unfold wrP
  split
  · rename_i h
    exact (Classical.choose_spec h).2
  · exact fun h => absurd h (not_accepts_nil _ _)

theorem accepts_wrP {n mPts : ℕ} {Reg : Type} (dj : Finset (Fin n) → Json) (Jf : Finset (Fin n) → Reg → ℕ)
    {S : Finset (Fin n)} {R : Reg} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (ha : act false x as cs = .decide s spec sts link c₁ mids f)
    (hacc : ((next false x as cs).eval sha512).1 = .acc) (hpi : s.read.publicInputs = none)
    (hd : s.read.draw = some (dj S)) (hJ : Jf S R ≤ spec.tables.size) :
    Exec.Accepts s.read.I (wrP s.read.I dj Jf S R (viewOf (Jf S R) mPts spec link c₁ f)) := by
  have hex : ∃ t, Exec.Accepts s.read.I t ∧
      RecordsLiveAtP s.read.I (dj S) (viewOf (Jf S R) mPts spec link c₁ f) t :=
    ⟨_, accepts_of_decide_plain ha hacc hpi hd hJ⟩
  unfold wrP
  rw [dite_eq_left_of_eq_true (eq_true hex)]
  exact (Classical.choose_spec hex).1

end Claims.Bridge.CustodyPlain
