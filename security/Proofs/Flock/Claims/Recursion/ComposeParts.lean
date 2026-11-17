module

public import Proofs.Flock.Claims.Recursion.Deploy
public import Proofs.Flock.Claims.Recursion.ComposeShared
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

/-! `rec-compose` holds the compiled verifier's parts to one shared set and one run (`composed_parts`, `composed_run`);
that the run record is the challenger's rests on `trusted-device`. -/

@[expose] public section

namespace RecDeploy

open Flock
open FlockDeploy (Raw)

def RecRaw.shared (w : RecRaw) : RecShared :=
  ⟨w.recInner, w.recRows, w.innerCircuit, w.innerPublic, w.innerPartition, w.innerCoins, w.innerSession, w.transcript,
    w.registration, w.innerStatement, w.innerTables, w.innerSample, w.recLinks, w.innerProgram⟩

def RecRaw.ofShared (S : RecShared) (base : Raw) (s : ℕ) (bindings : Option ByteArray) : RecRaw where
  base := base
  recInner := S.recInner
  recPart := some s
  recBindings := bindings
  recRows := S.recRows
  innerCircuit := S.innerCircuit
  innerPublic := S.innerPublic
  innerPartition := S.innerPartition
  innerCoins := S.innerCoins
  innerSession := S.innerSession
  transcript := S.transcript
  registration := S.registration
  innerStatement := S.innerStatement
  innerTables := S.innerTables
  innerSample := S.innerSample
  recLinks := S.links
  innerProgram := S.innerProgram

theorem RecRaw.ofShared_shared {w : RecRaw} {s : ℕ} (h : w.recPart = some s) :
    RecRaw.ofShared w.shared w.base s w.recBindings = w := by
  cases w
  cases h
  rfl

def stagedShared (r : XR) (st : Stage) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) : RecShared :=
  ⟨some r.w.recInner, some (st.rows x ms), some r.innerRaw.circuit, some r.innerRaw.publicFile, r.innerRaw.partition,
    none, some (st.record x ms), some (st.transcript x ms), r.w.registration, some r.innerRaw.statement,
    some r.innerRaw.tables, r.innerRaw.draw, r.w.links, r.innerRaw.program⟩

theorem partRaw_ofShared (r : XR) (st : Stage) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) (s : ℕ)
    (p : PartX) :
    partRaw r st x ms s p =
      RecRaw.ofShared (stagedShared r st x ms) { p.base with publicFile := st.pub x ms s } s (some p.bindings) :=
  rfl

structure PartRun where
  w : RecRaw
  accepted : Bool
  why : String
  coinKey : Option String
  identifier : Bool

def PartRun.view (p : PartRun) : PartView :=
  ⟨p.w.shared, p.w.recPart, p.w.base.zk, p.accepted, p.why, p.coinKey, p.identifier⟩

theorem composed_parts {n : ℕ} {rows inner : String} {links : Option (String × ℕ)} {ps : List PartRun}
    (h : RecCompose.Accepts n rows inner links (ps.map (·.view.verdict))) :
    Collides (ps.map PartRun.view) ∨ ∃ S : RecShared, S.Form ∧ S.recInner.map RecCompose.digest = some inner ∧
      RecCompose.linksOf S.links = links ∧ S.recRows.map RecCompose.digest = some rows ∧ ps.length = n ∧
      (∀ j, j < n → (ps.map (·.w.recPart)).count (some j) = 1) ∧ (ps.map (·.coinKey)).Nodup ∧
      ∀ p ∈ ps, (∃ s, s < n ∧ p.w = RecRaw.ofShared S p.w.base s p.w.recBindings) ∧ p.accepted = true ∧
        p.w.base.zk = true ∧ p.identifier = false ∧ p.coinKey.isSome := by
  have h' : RecCompose.Accepts n rows inner links ((ps.map PartRun.view).map PartView.verdict) := by
    rw [List.map_map]
    exact h
  rcases composed_views h' with hc | ⟨S, hS, hi, hli, hr, hl, hidx, hk, hall⟩
  · exact .inl hc
  refine .inr ⟨S, hS, hi, hli, hr, by simpa using hl, fun j hj => ?_, ?_, fun p hp => ?_⟩
  · have e := hidx j hj
    rw [List.map_map] at e
    exact e
  · rw [List.map_map] at hk
    exact hk
  · obtain ⟨hsh, ha, hz, hid, hkey, j, hj, hjn⟩ := hall _ (List.mem_map_of_mem hp)
    refine ⟨⟨j, hjn, ?_⟩, ha, hz, hid, hkey⟩
    have e := RecRaw.ofShared_shared (w := p.w) hj
    rw [show p.w.shared = S from hsh] at e
    exact e.symm

def OneRunOf (n : ℕ) (rows : String) (S : RecShared) (ps : List PartRun) (evs : List RecCompose.Event) : Prop :=
  ∃ opens : List RecCompose.Opened,
    evs = RecCompose.Event.stage rows S.digests :: (opens.map RecCompose.Event.opened ++ [RecCompose.Event.close]) ∧
    (∀ o ∈ opens, o.part < n) ∧ (∀ j, j < n → (opens.map (·.part)).count j = 1) ∧
    (∀ o ∈ opens, ∃ p ∈ ps, p.w.recPart = some o.part ∧ p.coinKey = some o.coinKey) ∧
    (∀ p ∈ ps, ∃ o ∈ opens, p.w.recPart = some o.part ∧ p.coinKey = some o.coinKey)

theorem composed_run {n : ℕ} {rows inner : String} {links : Option (String × ℕ)} {ps : List PartRun}
    {evs : List RecCompose.Event}
    (h : RecCompose.Accepts n rows inner links (ps.map (·.view.verdict)))
    (hr : RecCompose.OneRun n rows (ps.map (·.view.verdict)) evs) :
    Collides (ps.map PartRun.view) ∨ ∃ S : RecShared, S.Form ∧ S.recInner.map RecCompose.digest = some inner ∧
      RecCompose.linksOf S.links = links ∧ S.recRows.map RecCompose.digest = some rows ∧ ps.length = n ∧
      (∀ j, j < n → (ps.map (·.w.recPart)).count (some j) = 1) ∧ (ps.map (·.coinKey)).Nodup ∧
      (∀ p ∈ ps, (∃ s, s < n ∧ p.w = RecRaw.ofShared S p.w.base s p.w.recBindings) ∧ p.accepted = true ∧
        p.w.base.zk = true ∧ p.identifier = false ∧ p.coinKey.isSome) ∧
      OneRunOf n rows S ps evs := by
  rcases composed_parts h with hc | ⟨S, hS, hi, hli, hrows, hl, hidx, hk, hall⟩
  · exact .inl hc
  obtain ⟨sh, opens, hevs, hsh, hb, honce, hop, hrec⟩ := hr
  have hpos : 0 < n := h.1
  obtain _ | ⟨p0, rest⟩ := ps
  · simp at hl
    omega
  have hshared : ∀ p ∈ p0 :: rest, p.w.shared = S := by
    intro p hp
    obtain ⟨⟨s, _, e⟩, _⟩ := hall p hp
    rw [e]
    rfl
  have hsh0 := hsh p0.view.verdict
    (List.mem_map_of_mem (f := fun p : PartRun => p.view.verdict) (List.mem_cons_self (a := p0) (l := rest)))
  simp only [PartRun.view, PartView.verdict, Option.some.injEq] at hsh0
  rw [hshared p0 List.mem_cons_self] at hsh0
  subst hsh0
  refine .inr ⟨S, hS, hi, hli, hrows, hl, hidx, hk, hall, opens, hevs, hb, honce, fun o ho => ?_, fun p hp => ?_⟩
  · obtain ⟨v, hv, hpart, hkey⟩ := hop o ho
    obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hv
    exact ⟨p, hp, hpart, hkey⟩
  · exact hrec _ (List.mem_map_of_mem hp)

end RecDeploy
