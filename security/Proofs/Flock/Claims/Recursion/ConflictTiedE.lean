module

public import Proofs.Flock.Claims.Recursion.FlockKSDeployE
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
import Proofs.Flock.Claims.KS.Width

/-! `ConflictTied` at the plain record at `VE` (`plainE_conflictTied`); (P1) at `VE` still rests on the table bound at
`deployPlainE rel`, which is not proved. -/

@[expose] public section

namespace FlockDeploy.KS

open Flock
open Claims (OC OCV)
open FlockSoundness.Refine (ebind_ok)

local notation "𝔽" => FlockLevel3.GF128

theorem accE_decide {H : Claims.Msg → Claims.Msg} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin (VE.k x) → Coin} (h : (Claims.run H P VE x ω r).out = .acc) :
    ∃ s spec sts link c₁ mids f,
      act false x ((Claims.run H P VE x ω r).msgs.map Prod.fst)
          ((List.range ((Claims.run H P VE x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) =
        .decide s spec sts link c₁ mids f ∧
      Runs H (decisionE s x link c₁ mids f) () ((Claims.run H P VE x ω r).vLog.map Prod.fst) := by
  have ha := FlockDeploy.KS.run_acc_next H P VE x ω r h
  have hl := run_acc_log H P VE x ω r h
  generalize (List.range _).map (Claims.coinSeq r) = cs at ha hl
  replace ha : ((nextE x ((Claims.run H P VE x ω r).msgs.map Prod.fst) cs).eval H).1 = .acc := ha
  replace hl : ∀ p ∈ ((nextE x ((Claims.run H P VE x ω r).msgs.map Prod.fst) cs).eval H).2,
      p ∈ (Claims.run H P VE x ω r).vLog := hl
  cases hact : act false x ((Claims.run H P VE x ω r).msgs.map Prod.fst) cs with
  | rej => rw [nextE_rej H hact] at ha; cases ha
  | coin c => rw [nextE_coin H hact] at ha; cases ha
  | decide s spec sts link c₁ mids f =>
    refine ⟨s, spec, sts, link, c₁, mids, f, rfl, ?_, fun p hp => List.mem_map_of_mem (hl p ?_)⟩
    · simp only [nextE, hact] at ha
      exact outOf_acc H _ ha
    · simp only [nextE, hact, outOf_log]
      exact hp

theorem decisionE_runs {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    {L : List Claims.Msg} (hs : s.setup = .ok (spec, sts)) (h : Runs sha512 (decisionE s x link c₁ mids f) () L) :
    Runs sha512 (verifyWithME Claims.Hashes.oc spec (tabsOf sts) f.record f.proofs) () L ∧
      checks false s spec x link c₁ mids f = .ok () := by
  unfold decisionE at h
  obtain ⟨⟨spec', sts'⟩, h0, h⟩ := Runs.bind h
  have e : setupWith (Claims.hOf sha512) s.read = .ok (spec', sts') := by
    rw [← (setupM_eval sha512 s.read).1]
    exact h0.1
  rw [hOf_sha512, setupWith_sha512, Ins.setup_read hs] at e
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj e)
  unfold decisionAtE at h
  obtain ⟨⟨⟩, h1, h2⟩ := Runs.bind h
  exact ⟨h1, Runs.lift h2⟩

theorem verdictE_within {spec : SessionSpec} {tabs : Array Setup} {record : String} {proofs : Array ByteArray}
    {L : List Claims.Msg} (h : Runs sha512 (verifyWithME Claims.Hashes.oc spec tabs record proofs) () L) :
    verifyWithME (hashesIn L) spec tabs record proofs = .ok () :=
  (With.verifyWithME_sim (within L) (within_hashes L) spec tabs record proofs) () h

theorem accE_within {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {r : Fin (VE.k x) → Coin}
    (h : (Claims.run sha512 P VE x ω r).out = .acc) :
    ∃ s spec sts link c₁ mids f, setupOf false x = some (s, spec, sts) ∧
      lastFinal (Claims.run sha512 P VE x ω r).seen = some f ∧
      linkOf ((P.next x [] ω).eval sha512).1 = some link ∧ checks false s spec x link c₁ mids f = .ok () ∧
      verifyWithME (hashesIn ((Claims.run sha512 P VE x ω r).vLog.map Prod.fst)) spec (tabsOf sts) f.record
        f.proofs = .ok () := by
  obtain ⟨s, spec, sts, link, c₁, mids, f, hact, hrun⟩ := accE_decide h
  obtain ⟨hv, hc⟩ := decisionE_runs (setupOf_some (act_decide hact)).2 hrun
  obtain ⟨rest, hhead⟩ := run_head sha512 P VE x ω r h
  obtain ⟨rest', has, hlink⟩ := act_decide_link hact
  rw [hhead] at hlink
  refine ⟨s, spec, sts, link, c₁, mids, f, act_decide hact, ?_, hlink, hc, verdictE_within hv⟩
  have hl := act_decide_final hact
  rw [List.getLast?_map, Option.bind_map] at hl
  exact hl

theorem plainE_opening_raw {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin (VE.k x) → Coin} {q : ℕ} {u : List 𝔽} (h : (Claims.run sha512 P VE x ω r).out = .acc)
    (hu : (q, u) ∈ plainRawCols (Claims.run sha512 P VE x ω r).seen) :
    ∃ (s : Ins) (spec : SessionSpec) (sts : Array Flock.Stmt) (st : Flock.Stmt) (sch : Flock.Schedule)
      (link : List ByteArray × List ByteArray) (row : Array F128),
      setupOf false x = some (s, spec, sts) ∧ sts[0]? = some st ∧ fast100 st.m = some sch ∧
      linkOf ((P.next x [] ω).eval sha512).1 = some link ∧
      Opening (Setup.ofCircuit st).merkle (sch.levels.getD 0 default) (2 ^ (sch.levels.getD 0 default).lanes)
        link.1.toArray[0]! (Claims.run sha512 P VE x ω r).vLog q row ∧
      u = row.toList.map FlockLevel3.GF128.of := by
  obtain ⟨s, spec, sts, link, c₁, mids, f, hs, hf, hlink, hc, hv⟩ := accE_within h
  obtain ⟨st, sch, row, hst, hsch, hop, hu⟩ :=
    plain_opening_raw_of_verdictK (fun _ => true) (L := (Claims.run sha512 P VE x ω r).vLog) hs hf hc hv hu
  exact ⟨s, spec, sts, st, sch, link, row, hs, hst, hsch, hlink, hop, hu⟩

theorem plainE_opening {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {r : Fin (VE.k x) → Coin}
    {q : ℕ} {u : List 𝔽} (h : (Claims.run sha512 P VE x ω r).out = .acc)
    (hu : Claims.Flock.lookCol (cols false (Claims.run sha512 P VE x ω r).seen 0) q = some u) :
    ∃ (s : Ins) (spec : SessionSpec) (sts : Array Flock.Stmt) (st : Flock.Stmt) (sch : Flock.Schedule)
      (link : List ByteArray × List ByteArray) (row : Array F128),
      setupOf false x = some (s, spec, sts) ∧ sts[0]? = some st ∧ fast100 st.m = some sch ∧
      linkOf ((P.next x [] ω).eval sha512).1 = some link ∧
      Opening (Setup.ofCircuit st).merkle (sch.levels.getD 0 default) (2 ^ (sch.levels.getD 0 default).lanes)
        link.1.toArray[0]! (Claims.run sha512 P VE x ω r).vLog q row ∧
      u = row.toList.map FlockLevel3.GF128.of :=
  plainE_opening_raw h (mem_plainRawCols (lookCol_mem hu))

theorem plainE_conflictTied (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ConflictTied sha512 (deployPlainE rel) (wordLen false) := by
  intro x Ω P ω r r' q u v hacc hacc' _ hu hv huv
  obtain ⟨s, spec, sts, st, sch, link, row, hs, hst, hsch, hlink, ho, hu'⟩ :=
    plainE_opening (P := P) (r := r) hacc hu
  obtain ⟨s', spec', sts', st', sch', link', row', hs', hst', hsch', hlink', ho', hv'⟩ :=
    plainE_opening (P := P) (r := r') hacc' hv
  rw [hs] at hs'
  cases hs'
  rw [hst] at hst'
  cases hst'
  rw [hsch] at hsch'
  cases hsch'
  rw [hlink] at hlink'
  cases hlink'
  exact tiedColl_of_openings ho ho' fun e => huv (by rw [hu', hv', e])

theorem plainE_KSAbort_of_table (rel : Readable.Parsed → Readable.Relation)
    (htab : Claims.Recursion.TableBoundθOnAccR sha512 (deployPlainE rel) (wordLen false) (wide false))
    (decode : List (ℕ × List 𝔽) → List Claims.Flock.PlainTrace)
    (hdec : Claims.Recursion.decOK ((deployPlainE rel).mapCols trimCol) decode) :
    (Claims.Recursion.Inner.ofFlock ((deployPlainE rel).mapCols trimCol) decode).KSAbort sha512 :=
  plainE_KSAbortR rel (plainE_conflictTied rel) htab decode hdec

end FlockDeploy.KS

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

theorem plainKSAbort_bdE (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (htab : Claims.Recursion.TableBoundθOnAccR FlockDeploy.sha512 (FlockDeploy.deployPlainE rel)
      (FlockDeploy.KS.wordLen false) (FlockDeploy.KS.wide false))
    (join : List Msg → Msg) (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) :
    (bdDeployed rel FlockDeploy.VE.next (FlockDeploy.deployPlainTrim rel).cols
      (FlockDeploy.deployPlainTrim rel).weight join rowIn final w).PlainKSAbort FlockDeploy.sha512 :=
  fun decI hdec => FlockDeploy.KS.plainE_KSAbort_of_table rel htab decI hdec

end RecDeploy
