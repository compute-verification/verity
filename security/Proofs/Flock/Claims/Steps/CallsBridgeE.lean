module

public import Proofs.Flock.Claims.Bridge.Custody
public import Proofs.Flock.Claims.Complete.Encode
import Proofs.Flock.Claims.DeployK
import Proofs.Flock.Claims.KS.ConflictRun
import Proofs.Flock.Claims.KS.Width
import Flock.With.SimVerify
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
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init

/-! # What the plain KS proofs read of `Claims.Recursion.CallsBridgeE`, without its recursion imports. -/

@[expose] public section

namespace FlockDeploy.KS

open Flock FlockSoundness.Refine

open Claims.Bridge.Custody (RootField PointField RoundFields)

/-- A stream as `Record.decodeWithK` reads it, whatever `keep`: its name, and its rounds' fields. -/
def DecStreamK (x : Lean.Json) (y : StreamRec) : Prop :=
  Record.str x "stream" = .ok y.name ∧ ∃ rs, Record.arr x "rounds" = .ok rs ∧
    List.Forall₂ (fun r rd => RoundFields r rd.n rd.coins rd.bytes) rs.toList y.rounds.toList

/-- **What `Record.decodeWithK` reads**, whatever the hashes and `keep`: `decode_ok`'s fields, without S12's digests. -/
theorem decodeWithK_ok (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {text : String}
    {sess : RunRecord} (h : Record.decodeWithK keep hs spec text = .ok sess) :
    ∃ j l recs, Flock.JsonParse.parse text = .ok j ∧ Record.fld j "link" = .ok l ∧ Record.arr j "streams" = .ok recs ∧
      List.Forall₂ (RootField l) spec.tables.toList sess.rootB.toList ∧
      (∃ ps, Record.arr l "points" = .ok ps ∧ List.Forall₂ PointField ps.toList sess.points.toList) ∧
      List.Forall₂ DecStreamK recs.toList sess.streams.toList := by
  unfold Record.decodeWithK Record.decodeJWithK at h
  dsimp only at h
  split at h
  rotate_left
  · cases h
  rename_i j hparse
  obtain ⟨j', hj, h⟩ := ebind_ok h
  cases hj
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  all_goals
    cases h
    refine ⟨_, _, _, by assumption, by assumption, by assumption, ?_, ⟨_, by assumption, ?_⟩, ?_⟩
    · refine List.Forall₂.imp (fun t rb h => ?_) (Claims.Bridge.Custody.mapM_forall₂ (by assumption))
      beta_reduce at h
      obtain ⟨v, h1, h⟩ := ebind_ok h
      obtain ⟨s, h2, h⟩ := ebind_ok h
      obtain ⟨rb', h3, h⟩ := ebind_ok h
      repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (simp only [] at h))
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      exact ⟨v, s, h1, h2, h3⟩
    · refine List.Forall₂.imp (fun p pt h => ?_) (Claims.Bridge.Custody.mapM_forall₂ (by assumption))
      beta_reduce at h
      split at h
      · rename_i a ha
        exact ⟨a, ha, h⟩
      · cases h
    · refine List.Forall₂.imp (fun x y h => ?_) (Claims.Bridge.Custody.mapM_forall₂ (by assumption))
      beta_reduce at h
      obtain ⟨name, hname, h⟩ := ebind_ok h
      obtain ⟨rs, hrs, h⟩ := ebind_ok h
      obtain ⟨rounds, hrounds, h⟩ := ebind_ok h
      repeat' obtain ⟨_, _, h⟩ := ebind_ok h
      cases h
      refine ⟨hname, rs, hrs, forall₂_of_mapIdxM (fun k r rd hr => ?_) hrounds⟩
      obtain ⟨_, _, hr⟩ := ebind_ok hr
      obtain ⟨_, _, hr⟩ := ebind_ok hr
      obtain ⟨n, hn, hr⟩ := ebind_ok hr
      obtain ⟨_, _, hr⟩ := ebind_ok hr
      obtain ⟨_, _, hr⟩ := ebind_ok hr
      obtain ⟨cs, hcs, hr⟩ := ebind_ok hr
      obtain ⟨coins, hcoins, hr⟩ := ebind_ok hr
      repeat' (first | (obtain ⟨_, _, hr⟩ := ebind_ok hr) | (simp only [] at hr))
      by_cases hb : Record.isNull r "msg" = true
      · rw [ite_eq_left hb] at hr
        obtain ⟨w, hw, hr⟩ := ebind_ok hr
        simp only [pure, Except.pure, Except.ok.injEq] at hw hr
        subst hw hr
        exact ⟨hn, ⟨cs, hcs, hcoins⟩, fun _ => rfl, fun h' => by rw [hb] at h'; cases h'⟩
      · rw [ite_eq_right hb] at hr
        obtain ⟨s', hs', hr⟩ := ebind_ok hr
        obtain ⟨b, hb', hr⟩ := ebind_ok hr
        obtain ⟨_, _, hr⟩ := ebind_ok hr
        obtain ⟨_, _, hr⟩ := ebind_ok hr
        obtain ⟨_, _, hr⟩ := ebind_ok hr
        obtain ⟨w, hw, hr⟩ := ebind_ok hr
        simp only [pure, Except.pure, Except.ok.injEq] at hw hr
        subst hw hr
        exact ⟨hn, ⟨cs, hcs, hcoins⟩, fun h' => absurd h' hb, fun _ => ⟨s', b, hs', hb', rfl⟩⟩

/-- **The element-level decoder's live part is `liveOf`'s**, whatever the hashes and `keep`: the same `root_B` and
link points, and at each rep stream the stream the decoder finds by its name, whose rounds are `liveOf`'s up to their
recorded digests (`decode_live`'s, without S12's digests). -/
theorem decodeWithK_live (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {text : String}
    {sess : RunRecord} {lv : Live} (hd : Record.decodeWithK keep hs spec text = .ok sess)
    (hl : liveOf spec text = .ok lv) :
    sess.rootB = lv.rootB ∧ sess.points = lv.points ∧ lv.streams.size = spec.repStreams.size ∧
      ∀ q (hq : q < spec.repStreams.size), ∃ y, sess.streams.find? (·.name == spec.repStreams[q].2.2) = some y ∧
        List.Forall₂ (fun rd rd' => rd.n = rd'.n ∧ rd.coins = rd'.coins ∧ rd.bytes = rd'.bytes) y.rounds.toList
          (lv.streams[q]!).toList := by
  obtain ⟨j, l, recs, hp, hl1, hr, hroot, ⟨ps, hps, hpts⟩, hstr⟩ := decodeWithK_ok keep hs hd
  obtain ⟨j', l', recs', hp', hl1', hr', hroot', ⟨ps', hps', hpts'⟩, hstr'⟩ := Claims.Bridge.Custody.liveOf_ok hl
  rw [hp] at hp'
  cases hp'
  rw [hl1] at hl1'
  cases hl1'
  rw [hr] at hr'
  cases hr'
  rw [hps] at hps'
  cases hps'
  refine ⟨Array.ext' (Claims.Bridge.Custody.forall₂_unique (R := RootField l)
      (fun _ _ _ => Claims.Bridge.Custody.rootField_unique) hroot hroot'),
    Array.ext' (Claims.Bridge.Custody.forall₂_unique (R := PointField)
      (fun _ _ _ => Claims.Bridge.Custody.pointField_unique) hpts hpts'),
    by simpa using hstr'.length_eq.symm,
    fun q hq => ?_⟩
  obtain ⟨x, hx, rs, hrs, hrds⟩ := Claims.Bridge.Custody.forall₂_getElem hstr' hq
  rw [← Array.find?_toList] at hx
  obtain ⟨y, ⟨-, rs', hrs', hdr⟩, hy⟩ := Claims.Bridge.Custody.forall₂_find? (q := (·.name == spec.repStreams[q].2.2))
    (fun a b hab => (Claims.Bridge.Custody.stream_pred hab.1).symm) hstr hx
  rw [hrs] at hrs'
  cases hrs'
  refine ⟨y, by rw [← Array.find?_toList]; exact hy,
    (Claims.Bridge.Custody.forall₂_left hdr hrds).imp fun rd rd' h => ?_⟩
  obtain ⟨r, h1, h3, -⟩ := h
  exact Claims.Bridge.Custody.roundFields_unique h1 h3

end FlockDeploy.KS
