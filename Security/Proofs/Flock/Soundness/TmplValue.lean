import Proofs.Flock.Soundness.ExecRowsAt
import Proofs.Flock.Soundness.TmplUnrollAt
import Flock.TmplValue

/-!
# A templated unit's committed net, satisfied, composes its Definitions

Column values `z` that satisfy the committed net `Tmpl.unrollAt` builds, its whole `2^unitLog`-row slot, read through
`posMap`'s positions satisfy the template's logical rows (`rowHolds_of_netSat`), so `Flock.TmplValue.rows_eval` applies:
at every copy placed under one of the root's calls past the input rows, at any depth, the copy's columns hold its
Definition's value (`Tmpl.evalDef`) on its bound inputs (`unrollAt_eval`). A read templated unit's net is that net
(`TmplUnit.eval`). The same holds of the audit's `Rows` of the logical net `Tmpl.unroll` builds (`rowHolds_of_rowsSat`,
`unroll_eval`), the rows a partition unit is an instance of (`TmplValueUnit`). No verifier path reads this file.
-/

open Flock Flock.Tmpl FlockSoundness.Audit FlockSoundness.SparseRows

namespace FlockSoundness.TmplValue

/-- **The committed rows, holding, are the logical rows holding through `pos`.** -/
theorem rowHolds_of_netSat {name : String} {useful constPos : ℕ} {ig og : Array Group} {acc : Acc}
    {pos : Array ℕ} {n' : Net} (h : Net.ofRowsAt name useful constPos ig og acc.ra acc.rb pos = .ok n')
    (hu : n'.unitLog ≤ 32) {z : ℕ → Bool} (hs : NetSat n' (2 ^ n'.unitLog) z) :
    ∀ x < useful, Flock.TmplValue.RowHolds (fun x => z (pos.getD x 0)) acc x := by
  obtain ⟨n₀, -, -, -, hl, ha', hb', -, -, -, -, -, hp, -, -⟩ := Net.ofRowsAt_ok h
  have hw : 2 ^ n'.unitLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by decide) hu
  have hsize : pos.getD constPos 0 + 1 ≤ 2 ^ n'.unitLog := by
    rw [hl]
    exact Nat.le_trans (le_two_pow_log2ceil _) (Nat.pow_le_pow_right (by decide) (Nat.le_max_right _ _))
  intro x hx
  have e := hs _ (Nat.lt_of_lt_of_le (Net.posOk_lt hp hx) hsize)
  rw [ofRows_row ha' hw, ofRows_row hb' hw, Net.placed_at hp _ hx, Net.placed_at hp _ hx, Array.toList_map,
    Array.toList_map, xorSum_map, xorSum_map] at e
  exact e

/-- **The eval composition on the committed net**: column values that satisfy a templated unit's committed net take,
at the position of every column of every copy placed under one of the root's calls past the input rows, at any depth,
the value of that copy's Definition on the values at its bound inputs' positions, the constant at its position's. -/
theorem unrollAt_eval {t : Tmpl} {lays : Array Lay} {net : Net} (hwf : t.wellFormed = true)
    (ha : t.apart = true) (ho : t.ordered = true) (h : Tmpl.unrollAt t lays = .ok net) (hlog : net.unitLog ≤ 32)
    {z : ℕ → Bool} (hs : NetSat net (2 ^ net.unitLog) z) {c : Call} {j f e o : ℕ} {inp : List ℕ}
    (hc : c ∈ (t.defs.getD t.root default).calls.toList) (hb : t.inCols ≤ c.base) (hj : j < c.count)
    (hw : Within t.defs t.constPos t.defs.size c.def_ (c.base + j * c.stride)
      (c.bound ((t.defs.getD t.root default).ref 0 [] t.constPos) j) f e o inp) :
    ∀ y < (t.defs.getD e default).cols, z ((posMap t lays).getD (o + y) 0) =
      t.evalDef e (inp.map fun x => z ((posMap t lays).getD x 0)) (z ((posMap t lays).getD t.constPos 0)) y := by
  unfold Flock.Tmpl.unrollAt at h
  obtain ⟨acc, hr, hat⟩ := Flock.TmplOrder.bind_ok h
  exact Flock.TmplValue.rows_eval hwf ha ho hr (fun x => z ((posMap t lays).getD x 0))
    (fun x _ hx => rowHolds_of_netSat hat hlog hs x (by have := Flock.TmplUnroll.wf_cols hwf; omega)) hc hb hj hw

/-! ## The logical net's audit `Rows` -/

/-- **A built net's audit `Rows`, holding, are its rows holding** from its first computed row to its constant: the
converse of `rowsSat_of_netSat` on the rows `Rows.ofNet` reads. -/
theorem netRow_of_rowsSat {net : Net} (hB : Built net) (hu : net.unitLog ≤ 32) {z : ℕ → Bool}
    (hs : (Rows.ofNet hB).Sat fun c => z c.val) {r : ℕ} (h1 : min (net.inWords * WORD) net.constPos ≤ r)
    (h2 : r < net.constPos) : z r = (xorSum (net.a.row r).toList z && xorSum (net.b.row r).toList z) := by
  have hn : (Rows.ofNet hB).nComp = net.constPos - min (net.inWords * WORD) net.constPos := rfl
  have hr : (Rows.ofNet hB).nIn + (r - min (net.inWords * WORD) net.constPos) = r := by
    rw [Rows.ofNet_nIn]; omega
  have e := hs ⟨r - min (net.inWords * WORD) net.constPos, by omega⟩
  change z ((Rows.ofNet hB).nIn + (r - min (net.inWords * WORD) net.constPos)) =
    (xorSum ((Rows.ofNet hB).a _) (z ∘ Fin.val) && xorSum ((Rows.ofNet hB).b _) (z ∘ Fin.val)) at e
  rw [← xorSum_map, ← xorSum_map, Rows.ofNet_a hB hu, Rows.ofNet_b hB hu] at e
  simp only [hr] at e
  exact e

/-- **The audit `Rows` of a template's logical net, holding, are its logical rows holding** past the input rows. -/
theorem rowHolds_of_rowsSat {t : Tmpl} {acc : Acc} {n : Net}
    (hn : Net.ofRows t.name (t.defs.getD t.root default).cols t.constPos t.inGroups t.outGroups acc.ra acc.rb = .ok n)
    (hu : n.unitLog ≤ 32) {z : ℕ → Bool} (hs : (Rows.ofNet ⟨_, _, _, _, _, _, _, hn⟩).Sat fun c => z c.val) :
    ∀ x, t.inCols ≤ x → x < t.constPos → Flock.TmplValue.RowHolds z acc x := by
  obtain ⟨-, ha, hb, -, hc, -, -, -, -, -, -, -, -, hw, -⟩ := Net.ofRows_ok hn
  intro x h1 h2
  have e := netRow_of_rowsSat _ hu hs (r := x) (Nat.le_trans (Nat.min_le_left _ _) (by rw [hw]; exact h1))
    (by rw [hc]; exact h2)
  have w32 : 2 ^ n.unitLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by decide) hu
  rw [ofRows_row ha w32, ofRows_row hb w32] at e
  exact e

/-- `Tmpl.unroll`'s net is built by `Net.ofRows`. -/
theorem unroll_built {t : Tmpl} {n : Net} (h : Flock.Tmpl.unroll t = .ok n) : Built n := by
  unfold Flock.Tmpl.unroll at h
  obtain ⟨_, -, hn⟩ := Flock.TmplOrder.bind_ok h
  exact ⟨_, _, _, _, _, _, _, hn⟩

/-- **The eval composition on the logical net's audit `Rows`**: column values that satisfy the `Rows` of the net
`Tmpl.unroll` builds take, at every column of every copy placed under one of the root's calls past the input rows, at any
depth, the value of that copy's Definition on the values at its bound inputs, the constant at the constant column's. -/
theorem unroll_eval {t : Tmpl} {n : Net} (hwf : t.wellFormed = true) (ha : t.apart = true) (ho : t.ordered = true)
    (h : Flock.Tmpl.unroll t = .ok n) (hu : n.unitLog ≤ 32) {z : ℕ → Bool}
    (hs : (Rows.ofNet (unroll_built h)).Sat fun c => z c.val) {c : Call} {j f e o : ℕ} {inp : List ℕ}
    (hc : c ∈ (t.defs.getD t.root default).calls.toList) (hb : t.inCols ≤ c.base) (hj : j < c.count)
    (hw : Within t.defs t.constPos t.defs.size c.def_ (c.base + j * c.stride)
      (c.bound ((t.defs.getD t.root default).ref 0 [] t.constPos) j) f e o inp) :
    ∀ y < (t.defs.getD e default).cols, z (o + y) = t.evalDef e (inp.map z) (z t.constPos) y := by
  have h' := h
  unfold Flock.Tmpl.unroll at h'
  obtain ⟨acc, hr, hn⟩ := Flock.TmplOrder.bind_ok h'
  exact Flock.TmplValue.rows_eval hwf ha ho hr z (rowHolds_of_rowsSat hn hu hs) hc hb hj hw

end FlockSoundness.TmplValue

namespace Flock

/-- **A read templated unit's net composes its Definitions** (`FlockSoundness.TmplValue.unrollAt_eval`). -/
theorem TmplUnit.eval (tu : TmplUnit) (hlog : tu.net.unitLog ≤ 32) {z : ℕ → Bool}
    (hs : FlockSoundness.NetSat tu.net (2 ^ tu.net.unitLog) z) {c : Tmpl.Call} {j f e o : ℕ} {inp : List ℕ}
    (hc : c ∈ (tu.t.defs.getD tu.t.root default).calls.toList) (hb : tu.t.inCols ≤ c.base) (hj : j < c.count)
    (hw : Tmpl.Within tu.t.defs tu.t.constPos tu.t.defs.size c.def_ (c.base + j * c.stride)
      (c.bound ((tu.t.defs.getD tu.t.root default).ref 0 [] tu.t.constPos) j) f e o inp) :
    ∀ y < (tu.t.defs.getD e default).cols, z ((Tmpl.posMap tu.t tu.lays).getD (o + y) 0) =
      tu.t.evalDef e (inp.map fun x => z ((Tmpl.posMap tu.t tu.lays).getD x 0))
        (z ((Tmpl.posMap tu.t tu.lays).getD tu.t.constPos 0)) y := by
  have h := tu.ok
  unfold Tmpl.unrollOk at h
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨hwf, ha⟩, ho⟩, -⟩, -⟩, -⟩, -⟩, -⟩ := h
  exact FlockSoundness.TmplValue.unrollAt_eval hwf ha ho tu.unrolled hlog hs hc hb hj hw

end Flock
