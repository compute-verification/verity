import Proofs.Flock.Soundness.TmplValueUnit
import Proofs.Flock.Soundness.ExecCircuit
import Flock.HmRow

/-!
# A templated file's unit, at its committed net

A templated statement's circuit is `HmOut.parseTmpl`'s: its template read as a unit (`TmplUnit.read`), whose committed
net `tu.net` is `HmRow.parse`'s first net, named `unit`, where a flat file's `CIRCUIT unit` section is, then
`HmOut.checkRows`, with the unit carried on the circuit (`parseTmpl_cases`); what that parse accepted is `ParseFacts`
with the unit net exempt from `Built` (`parseTmpl_facts`, `ParseFactsPre`). That net is `Tmpl.unrollAt`'s:
`Tmpl.unroll`'s logical net, its rows moved to `posMap`'s positions (`unroll_of_unrollAt`). So column values that
satisfy the committed net satisfy, through `posMap`, the audit's `Rows` of the logical net, the rows `unit_eval` reads,
and are zero off `posMap`'s image (`TmplUnit.rowsSat`). With the constant at 1, at every copy of a computing Definition placed
under one of the root's calls past the input rows, they carry its spec on its bound inputs (`TmplUnit.computes`, the
committed net's `unit_computes`). No verifier path reads this file.
-/

open Flock Flock.Tmpl FlockSoundness.Audit

namespace FlockSoundness.TmplValue

/-- **A templated file's circuit** is `HmRow.parse`'s with the read unit's committed net first, named `unit`, passing
`HmOut.checkRows`, with no net named `unit` or `root` after it, and the unit carried on it. -/
theorem parseTmpl_cases' {tags : Flock.Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Flock.Circuit}
    (h : HmOut.parseTmpl tags bytes tables = .ok c) :
    ∃ (tu : TmplUnit) (c₀ : Flock.Circuit), HmRow.parse tags bytes tables (pre := #[("unit", tu.net)]) = .ok c₀ ∧
      HmOut.checkRows c₀ = .ok () ∧
      (c₀.nets.extract 1 c₀.nets.size).any (fun x => x.1 == "unit" || x.1 == "root") = false ∧
      c = { c₀ with tmpl := some tu } ∧ HmOut.recUnit tu.net.name = false := by
  unfold HmOut.parseTmpl at h
  dsimp only at h
  obtain ⟨_, -, h⟩ := Flock.TmplOrder.bind_ok h
  obtain ⟨_, -, h⟩ := Flock.TmplOrder.bind_ok h
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

/-- **What a templated file's parse accepted**: `ParseFactsPre` at the read unit's committed net, which is the circuit's
first net, its unit: every other net is `Built` or a lookup slot, as a flat file's are. -/
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

/-- **A templated file's unit is no recursion unit** (`HmOut.recUnit`, which `parseTmpl` refuses). -/
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

/-- **The committed net is the logical net moved**: `Tmpl.unrollAt` and `Tmpl.unroll` read the same rows, and
`Net.ofRowsAt`'s net is at `posMap` the net `Net.ofRows` builds from them, `Tmpl.unroll`'s. -/
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

/-- **A read templated unit's committed net, satisfied, is its logical net's audit `Rows` satisfied through `posMap`**,
and zero at every position of its slot no column is placed at. -/
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

/-- **A read templated unit's committed net computes its Definitions' specs**: column values that satisfy it, with the
constant's position at 1, at every copy of Definition `e` placed under one of the root's calls past the input rows, at
any depth, are at the positions of `e`'s columns `outs` `spec` of the values at its bound inputs' positions, when `e`
computes `spec` on `outs`. -/
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
