import Proofs.Flock.Soundness.ExecPlacement
import Proofs.Flock.Soundness.TmplUnitNet

/-!
# A templated unit is its logical net placed at `posMap`

A circuit carrying a read templated unit `tu`, whose unit net is `tu.net`: its logical net (`lunit`) is `Tmpl.unroll`'s,
a `Built` net (`built_tmpl`), and `tu.net` is that net's rows moved to `posMap`'s positions (`Net.ofRowsAt`,
`TmplUnit.unrolled`), so the unit is its logical net placed at `upos` (`UnitAt.tmpl`). The committed net itself need not
be `Built`: a laned copy's columns can sit before the columns it reads.
-/

namespace FlockSoundness

open Flock

theorem log2ceil_mono {a b : ℕ} (h : a ≤ b) : log2ceil a ≤ log2ceil b := by
  unfold log2ceil
  split
  · exact Nat.zero_le _
  · rw [if_neg (by omega)]
    have : Nat.log2 (a - 1) ≤ Nat.log2 (b - 1) := by
      rw [Nat.log2_eq_log_two, Nat.log2_eq_log_two]
      exact Nat.log_mono_right (by omega)
    omega

theorem lunit_tmpl {c : Circuit} {tu : TmplUnit} (ht : c.tmpl = some tu) {n : Net} (hn : Tmpl.unroll tu.t = .ok n) :
    lunit c = n := by
  simp [lunit, ht, hn]

theorem upos_tmpl {c : Circuit} {tu : TmplUnit} (ht : c.tmpl = some tu) :
    upos c = fun x => (Tmpl.posMap tu.t tu.lays).getD x 0 := by
  funext x
  simp [upos, ht]

/-- **A templated unit's logical net is `Built`**: `Tmpl.unroll`'s, `Net.ofRows` on its rows. -/
theorem built_tmpl {c : Circuit} {tu : TmplUnit} (ht : c.tmpl = some tu) : Built (lunit c) := by
  obtain ⟨_, n, -, -, -, hu⟩ := TmplValue.unroll_of_unrollAt tu.unrolled
  rw [lunit_tmpl ht hu]
  exact TmplValue.unroll_built hu

/-- **A templated unit net's rows are its useful rows**, which end at its constant: `Net.ofRowsAt` places one row at
each position up to the committed constant. -/
theorem tmpl_net_rows (tu : TmplUnit) :
    tu.net.a.rows = tu.net.useful ∧ tu.net.b.rows = tu.net.useful ∧ tu.net.useful = tu.net.constPos + 1 := by
  obtain ⟨_acc, _n, -, -, hat, -⟩ := TmplValue.unroll_of_unrollAt tu.unrolled
  obtain ⟨_n₀, -, hu, hcp, -, ha, hb, -, -, -, -, -, hp, -, -⟩ := Net.ofRowsAt_ok hat
  have hi := (Net.posOk_parts hp).2.1
  rw [ofRows_rows ha, ofRows_rows hb, Net.placed_size, Net.placed_size, hi, hu, hcp]
  exact ⟨rfl, rfl, rfl⟩

/-- **A templated unit is its logical net placed at `upos`**: `posMap` keeps the input rows, puts every other column
between them and the constant, the constant at the committed constant, and committed row `posMap[x]` is logical row `x`
read through `posMap`. -/
theorem UnitAt.tmpl {c : Circuit} {tu : TmplUnit} (ht : c.tmpl = some tu) (hu : c.unit = tu.net) : UnitAt c := by
  obtain ⟨acc, n, -, hn, hat, hun⟩ := TmplValue.unroll_of_unrollAt tu.unrolled
  have el := lunit_tmpl ht hun
  have ep := upos_tmpl ht
  obtain ⟨n₀, hn₀, -, hcp, hl, ha', hb', -, hig, hiw, -, -, hp, -, -⟩ := Net.ofRowsAt_ok hat
  have e : n₀ = n := by rw [hn] at hn₀; cases hn₀; rfl
  subst e
  obtain ⟨hlog0, ha, hb, -, hc0, -, -, hin, -⟩ := Net.ofRows_ok hn
  have hgr := (ofRows_groups hn).1
  obtain ⟨-, -, h1, -, hinp⟩ := Net.posOk_parts hp
  generalize hP : Tmpl.posMap tu.t tu.lays = pos at ep hcp hl ha' hb' hp h1 hinp
  generalize hU : (tu.t.defs.getD tu.t.root default).cols = U at hgr hp h1 hlog0
  generalize hK : tu.t.constPos = K at hgr hc0 hin hcp hl ha' hb' hp h1
  have hinj : ∀ {x y}, x < U → y < U → pos.getD x 0 = pos.getD y 0 → x = y := fun hx hy e =>
    Net.posOk_inj hp hx hy e
  have hUle : U ≤ pos.getD K 0 + 1 := by
    have := Fintype.card_le_of_injective
      (fun x : Fin U => (⟨pos.getD x.val 0, (h1 x.val x.isLt).1⟩ : Fin (pos.getD K 0 + 1)))
      (fun x y hxy => Fin.ext (hinj x.isLt y.isLt (congrArg Fin.val hxy)))
    simpa using this
  have hlogle : n₀.unitLog ≤ tu.net.unitLog := by
    rw [hlog0, hl]
    exact max_le_max (le_refl _) (log2ceil_mono hUle)
  have hle : ∀ x ≤ K, pos.getD x 0 ≤ pos.getD K 0 := fun x hx => Nat.lt_succ_iff.1 (h1 x (by omega)).1
  have hrow : tu.net.unitLog ≤ 32 → ∀ (ra : Array (Array ℕ)) (m m' : Sparse), Sparse.ofRows ra (2 ^ n₀.unitLog) = .ok m →
      Sparse.ofRows (Net.placed pos (Net.invOf pos (pos.getD K 0 + 1)) ra) (2 ^ tu.net.unitLog) = .ok m' →
      ∀ x < U, ∀ j, FlockLevel3.sparseEntry m' (pos.getD x 0) j =
        ((((m.row x).toList.map fun y => pos.getD y 0).count j : ℕ) : ZMod 2) := by
    intro hlog ra m m' hm hm' x hx j
    have w32 : 2 ^ n₀.unitLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by decide) (by omega)
    have w32' : 2 ^ tu.net.unitLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by decide) hlog
    rw [sparseEntry_ofRows hm', ofRows_row hm' w32', Net.placed_at hp ra hx, ofRows_row hm w32, Array.toList_map]
  refine
    { in_eq := (by rw [el, hu, hiw])
      groups := (by rw [el, hu, hig])
      in_le := (by rw [el, hc0]; exact hin)
      log_le := (by rw [el, hu]; exact hlogle)
      const_le := (by rw [el, hu, hc0, hcp]; omega)
      const := (by rw [el, ep, hu, hc0, hcp])
      le := fun x hx => (by rw [ep, hu, hcp]; rw [el, hc0] at hx; exact hle x hx)
      inputs := fun x hx => (by rw [ep]; rw [hu, hiw] at hx; exact hinp x hx)
      comp := fun x hx1 hx2 => ?_
      row_a := fun x hx2 hlog j => ?_
      row_b := fun x hx2 hlog j => ?_ }
  · rw [ep, hu, hcp, hiw]
    beta_reduce
    rw [hu, hiw] at hx1
    rw [el, hc0] at hx2
    refine ⟨Nat.not_lt.1 fun hlt => ?_, lt_of_le_of_ne (hle x hx2.le) fun e => by
      have := hinj (by omega) (by omega) e
      omega⟩
    have := hinj (x := pos.getD x 0) (y := x) (by omega) (by omega) (hinp _ hlt)
    omega
  · rw [el, hc0] at hx2
    rw [el, ep, hu]
    rw [hu] at hlog
    exact hrow hlog _ _ _ ha ha' x (by omega) j
  · rw [el, hc0] at hx2
    rw [el, ep, hu]
    rw [hu] at hlog
    exact hrow hlog _ _ _ hb hb' x (by omega) j

end FlockSoundness
