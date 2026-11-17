module

public import Flock.TmplComputes
public import Proofs.Flock.Soundness.ExecCircuit
public import Proofs.Flock.Soundness.TmplValue
public import Proofs.Flock.Soundness.TmplValueUnit
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.MeasureTheory.Integral.Bochner.Basic
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
import Proofs.Flock.Soundness.TmplUnrollAt
import VCVio.Prelude.Core

/-! A templated file's unit at its committed net: what its parse accepted, and its Definitions' specs at every copy
(`TmplUnit.computes`); no verifier path reads this file. -/

@[expose] public section

open Flock Flock.Tmpl FlockSoundness.Audit

namespace FlockSoundness.TmplValue

theorem parseTmpl_cases' {tags : Flock.Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Flock.Circuit}
    (h : HmOut.parseTmpl tags bytes tables = .ok c) :
    ∃ (tu : TmplUnit) (c₀ : Flock.Circuit), HmRow.parse tags bytes tables (pre := #[("unit", tu.net)]) = .ok c₀ ∧
      HmOut.checkRows c₀ = .ok () ∧
      (c₀.nets.extract 1 c₀.nets.size).any (fun x => x.1 == "unit" || x.1 == "root") = false ∧
      c = { c₀ with tmpl := some tu } ∧ HmOut.recUnit tu.net.name = false := by
  unfold HmOut.parseTmpl at h
  dsimp only at h
  obtain ⟨_, -, h⟩ := Flock.TmplOrder.bind_ok h
  split at h
  · cases h
  obtain ⟨_, -, h⟩ := Flock.TmplOrder.bind_ok h
  obtain ⟨_, -, h⟩ := Flock.TmplOrder.bind_ok h
  split at h
  · obtain ⟨_, -, h⟩ := Flock.TmplOrder.bind_ok h
    obtain ⟨tu, -, h⟩ := Flock.TmplOrder.bind_ok h
    obtain ⟨c₀, hc₀, h⟩ := Flock.TmplOrder.bind_ok h
    split at h
    · cases h
    · rename_i hb
      split at h
      · cases h
      · rename_i hr
        obtain ⟨_, hu, h⟩ := Flock.TmplOrder.bind_ok h
        simp only [pure, Except.pure, Except.ok.injEq] at h
        exact ⟨tu, c₀, hc₀, hu, Bool.eq_false_iff.mpr hb, h.symm, Bool.eq_false_iff.mpr hr⟩
  · cases h

theorem parseTmpl_cases {tags : Flock.Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Flock.Circuit}
    (h : HmOut.parseTmpl tags bytes tables = .ok c) :
    ∃ (tu : TmplUnit) (c₀ : Flock.Circuit), HmRow.parse tags bytes tables (pre := #[("unit", tu.net)]) = .ok c₀ ∧
      HmOut.checkRows c₀ = .ok () ∧
      (c₀.nets.extract 1 c₀.nets.size).any (fun x => x.1 == "unit" || x.1 == "root") = false ∧
      c = { c₀ with tmpl := some tu } :=
  let ⟨tu, c₀, hp, hc, hn, he, _⟩ := parseTmpl_cases' h
  ⟨tu, c₀, hp, hc, hn, he⟩

theorem parseTmpl_facts {tags : Flock.Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Flock.Circuit}
    (h : HmOut.parseTmpl tags bytes tables = .ok c) :
    ∃ (tu : TmplUnit) (c₀ : Flock.Circuit), ParseFactsPre tags #[("unit", tu.net)] c₀ ∧ HmOut.checkRows c₀ = .ok () ∧
      c = { c₀ with tmpl := some tu } ∧ c₀.unitNet = 0 ∧ c₀.nets.getD 0 default = ("unit", tu.net) := by
  obtain ⟨tu, c₀, hp, hck, -, he⟩ := parseTmpl_cases h
  obtain ⟨P, ⟨more, hm, -⟩, hu, -, -⟩ := parse_factsPre_nets hp
  have hm' : c₀.nets.toList = ("unit", tu.net) :: more := hm
  have h0 : c₀.nets.findIdx? (·.1 == "unit") = some 0 := by
    rw [← Array.toArray_toList (xs := c₀.nets), List.findIdx?_toArray, hm']
    simp [List.findIdx?_cons]
  refine ⟨tu, c₀, P, hck, he, (Option.some.inj (hu.symm.trans h0)), ?_⟩
  rw [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, hm']
  rfl

theorem parseTmpl_notRec {tags : Flock.Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Flock.Circuit}
    (h : HmOut.parseTmpl tags bytes tables = .ok c) : HmOut.recUnit c.unit.name = false := by
  obtain ⟨tu, c₀, -, -, he, hu, hn⟩ := parseTmpl_facts h
  obtain ⟨tu', c₀', -, -, -, he', hr⟩ := parseTmpl_cases' h
  have h1 : c.tmpl = some tu := by rw [he]
  have h2 : c.tmpl = some tu' := by rw [he']
  obtain rfl : tu' = tu := Option.some.inj (h2.symm.trans h1)
  have hcu : c.unit = tu'.net := by
    rw [he]
    show (c₀.nets.getD c₀.unitNet default).2 = tu'.net
    rw [hu, hn]
  rw [hcu]
  exact hr

theorem unroll_of_unrollAt {t : Tmpl} {lays : Array Lay} {n' : Net} (h : Tmpl.unrollAt t lays = .ok n') :
    ∃ (acc : Acc) (n : Net), t.rows = .ok acc ∧
      Net.ofRows t.name (t.defs.getD t.root default).cols t.constPos t.inGroups t.outGroups acc.ra acc.rb = .ok n ∧
      Net.ofRowsAt t.name (t.defs.getD t.root default).cols t.constPos t.inGroups t.outGroups acc.ra acc.rb
        (posMap t lays) = .ok n' ∧ Flock.Tmpl.unroll t = .ok n := by
  unfold Flock.Tmpl.unrollAt at h
  obtain ⟨acc, hr, hat⟩ := Flock.TmplOrder.bind_ok h
  obtain ⟨n, hn, -⟩ := Net.ofRowsAt_ok hat
  refine ⟨acc, n, hr, hn, hat, ?_⟩
  unfold Flock.Tmpl.unroll
  rw [hr]
  exact hn

end FlockSoundness.TmplValue

namespace Flock

theorem TmplUnit.rowsSat (tu : TmplUnit) {n : Net} (hn : Tmpl.unroll tu.t = .ok n) (hu : n.unitLog ≤ 32)
    (hlog : tu.net.unitLog ≤ 32) {z : ℕ → Bool} (hs : FlockSoundness.NetSat tu.net (2 ^ tu.net.unitLog) z) :
    (FlockSoundness.Rows.ofNet (FlockSoundness.TmplValue.unroll_built hn)).Sat
        (fun c => z ((Tmpl.posMap tu.t tu.lays).getD c.val 0)) ∧
      ∀ p < 2 ^ tu.net.unitLog,
        (∀ x < (tu.t.defs.getD tu.t.root default).cols, (Tmpl.posMap tu.t tu.lays).getD x 0 ≠ p) → z p = false := by
  obtain ⟨_, n₀, -, hn₀, hat, hu₀⟩ := FlockSoundness.TmplValue.unroll_of_unrollAt tu.unrolled
  have e : n₀ = n := Except.ok.inj (hu₀.symm.trans hn)
  subst e
  exact FlockSoundness.ofRowsAt_rowsSat hn₀ hat hu hlog hs

theorem TmplUnit.computes (tu : TmplUnit) (hlog : tu.net.unitLog ≤ 32) {z : ℕ → Bool}
    (hs : FlockSoundness.NetSat tu.net (2 ^ tu.net.unitLog) z)
    (hone : z ((Tmpl.posMap tu.t tu.lays).getD tu.t.constPos 0) = true) {c : Tmpl.Call} {j f e o : ℕ} {inp : List ℕ}
    (hc : c ∈ (tu.t.defs.getD tu.t.root default).calls.toList) (hb : tu.t.inCols ≤ c.base) (hj : j < c.count)
    (hw : Tmpl.Within tu.t.defs tu.t.constPos tu.t.defs.size c.def_ (c.base + j * c.stride)
      (c.bound ((tu.t.defs.getD tu.t.root default).ref 0 [] tu.t.constPos) j) f e o inp)
    {outs : List ℕ} {spec : (ℕ → Bool) → List Bool} (hD : tu.t.DefComputes e outs spec)
    (hout : ∀ q < outs.length, outs.getD q 0 < (tu.t.defs.getD e default).cols) :
    ∀ q < outs.length, z ((Tmpl.posMap tu.t tu.lays).getD (o + outs.getD q 0) 0) =
      (spec fun x => z ((Tmpl.posMap tu.t tu.lays).getD (inp.getD x 0) 0)).getD q false :=
  FlockSoundness.TmplValue.computes_of_eval (t := tu.t) (f := fun x => z ((Tmpl.posMap tu.t tu.lays).getD x 0))
    (Flock.TmplValue.within_ins (Flock.TmplValue.wf_getD tu.wf) hw
      (Flock.TmplValue.call_bound_length (Flock.TmplValue.wf_getD tu.wf tu.t.root) hc _ j))
    (fun y hy => by
      have hv := tu.eval hlog hs hc hb hj hw y hy
      rwa [hone] at hv) hD hout

end Flock
