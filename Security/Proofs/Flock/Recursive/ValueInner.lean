import Proofs.Flock.Recursive.ValueSource

/-!
# The verifier's rows at the inner session's units

`valueSrcAll` at `rec-open`, `rec-residuals` and `rec-sparse` (`openRows`, `residualRows`, `sparseRows`, defined beside
`valueSrcC3` in `Recursive/ValueSrc.lean`):

* `open_of_check`, `residuals_of_check`, `sparse_of_check`: an accepted unit's `checkValue`s are at those rows, which
  are `valueRowsAll`'s (`valueRowsAll_load`, `innerRows_coef`, `innerRows_dirs`, `innerRows_v`, `innerRows_sparse`), so
  `valueSrcAll_eq` gives `value_tie_coll`'s `hsrc`;
* `openCheck_of_check`, `residualsCheck_of_check`, `sparseCheck_of_check`: `RecCoins.check` runs those checks there.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit

variable {n : ℕ} {L : Law n} {Reg Cm Reg₂ : Type} {I : Inner} {S : ℕ} {Vs : Fin S → VStmt}

theorem valueSrcAll_eq {src : List (Cm × I.Coin) → Option Flock.RecCoins.Source}
    {c3 : List (Cm × I.Coin) → Option Flock.RecCoins.C3} {R : Reg} {ω : L.Ω} {t : List (Cm × I.Coin)} {R₂ : Reg₂}
    {j : Fin S} {u : Fin (Vs j).nv} {q : Fin (Vs j).c.ports.size} {rows : Array (Array ℕ)} {value prog : String}
    (hv : valueRowsAll (src t) (c3 t) (Vs j).c ((Vs j).c.ports[q.val]!).name = some (rows, value))
    (hprog : (Vs j).c.metaJ.getObjValAs? String "program_sha512" = .ok prog) :
    valueSrcAll (L := L) (Reg := Reg) (Reg₂ := Reg₂) (Vs := Vs) src c3 R ω t R₂ ⟨j, u, q⟩ =
      some (rows, Flock.RecCoins.verifierSalt prog value) := by
  simp only [valueSrcAll, hv, hprog]

theorem valueRowsAll_stacked {src : Option Flock.RecCoins.Source} {c3 : Option Flock.RecCoins.C3} {c : Flock.Circuit}
    {port : String} {p : ℕ × ℕ} (hso : Flock.RecCoins.stackedParams c.unit.name = some p) :
    valueRowsAll src c3 c port = valueRows c3 c port := by
  simp only [valueRowsAll, hso]

theorem valueRowsAll_opening {src : Option Flock.RecCoins.Source} {c3 : Option Flock.RecCoins.C3} {c : Flock.Circuit}
    {port : String} {p : ℕ × ℕ × ℕ} (hso : Flock.RecCoins.stackedParams c.unit.name = none)
    (hop : Flock.RecOpening.params c.unit.name = some p) : valueRowsAll src c3 c port = valueRows c3 c port := by
  simp only [valueRowsAll, hso, hop]

theorem valueRowsAll_load {f : Flock.RecCoins.Source} {inn : Flock.RecCoins.Inner} {c3 : Option Flock.RecCoins.C3}
    {c : Flock.Circuit} {port : String} (hso : Flock.RecCoins.stackedParams c.unit.name = none)
    (hop : Flock.RecOpening.params c.unit.name = none) (hl : f.load c.unit.name = .ok inn) :
    valueRowsAll (some f) c3 c port = innerRows inn c3 c port := by
  simp only [valueRowsAll, loadRows, hso, hop, hl]

/-- **What an accepted `rec-open` unit checked**: its `coef` and `dirs` at `openRows`' rows. -/
theorem open_of_check {inn : Flock.RecCoins.Inner} {c : Flock.Circuit} {pub : Flock.Public}
    {own : Array Flock.Registered.Port} {LANES H : ℕ} (h : Flock.RecCoins.checkOpen inn c pub own LANES H = .ok ()) :
    ∃ cr dr, openRows inn LANES H = some (cr, dr) ∧
      (∃ what, Flock.RecCoins.checkValue c pub own "coef" Flock.RecCoins.COEF_VALUE cr what = .ok ()) ∧
      ∃ what, Flock.RecCoins.checkValue c pub own "dirs" Flock.RecCoins.DIRS_VALUE dr what = .ok () := by
  unfold Flock.RecCoins.checkOpen at h
  dsimp only at h
  obtain ⟨⟨sh, extras⟩, halg, h⟩ := FlockVBridge.PartS.bind_ok h
  try dsimp only at h
  split at h
  · rename_i l hls
    have hls' : innerLevels sh LANES H = [l] := hls
    obtain ⟨r0, h0, h⟩ := FlockVBridge.PartS.bind_ok h
    obtain ⟨r1, h1, h⟩ := FlockVBridge.PartS.bind_ok h
    obtain ⟨⟨⟩, -, h⟩ := FlockVBridge.PartS.bind_ok h
    try dsimp only at h
    obtain ⟨⟨⟩, hcv, hdv⟩ := FlockVBridge.PartS.bind_ok h
    exact ⟨openCoef sh l LANES r0 r1, openDirs sh l H r0 r1, by simp only [openRows, halg, hls', h0, h1], ⟨_, hcv⟩, _,
      hdv⟩
  · cases h

/-- **What an accepted `rec-residuals` unit checked**: its `v` at `residualRows`' rows. -/
theorem residuals_of_check {inn : Flock.RecCoins.Inner} {c : Flock.Circuit} {pub : Flock.Public}
    {own : Array Flock.Registered.Port} {m k cl j : ℕ}
    (h : Flock.RecCoins.checkResiduals inn c pub own m k cl j = .ok ()) :
    ∃ rows what, residualRows inn j = some rows ∧
      Flock.RecCoins.checkValue c pub own "v" Flock.RecCoins.V_VALUE rows what = .ok () := by
  unfold Flock.RecCoins.checkResiduals at h
  dsimp only at h
  obtain ⟨⟨sh, extras⟩, halg, h⟩ := FlockVBridge.PartS.bind_ok h
  try dsimp only at h
  split at h
  · cases h
  · split at h
    · rename_i gs hX
      have hX' : residualGroups sh = .ok gs := hX
      obtain ⟨gs', hgs, h⟩ := FlockVBridge.PartS.bind_ok h
      cases hgs
      split at h
      · rename_i hj
        obtain ⟨rows, hrows, h⟩ := FlockVBridge.PartS.bind_ok h
        have hrows' : residualRowsOf inn sh extras _ = .ok rows := hrows
        exact ⟨rows, _, by simp only [residualRows, halg, hX', hj, hrows'], h⟩
      · cases h
    · cases h

/-- **What an accepted `rec-sparse` unit checked**: its `v` at `sparseRows`' rows. -/
theorem sparse_of_check {inn : Flock.RecCoins.Inner} {c3 : Flock.RecCoins.C3} {c : Flock.Circuit} {pub : Flock.Public}
    {own : Array Flock.Registered.Port} {sh : Flock.RecSparse.Shape}
    (h : Flock.RecCoins.checkSparse inn c3 c pub own sh = .ok ()) :
    ∃ rows what, sparseRows inn c3 c.unit.name sh = some rows ∧
      Flock.RecCoins.checkValue c pub own "v" Flock.RecCoins.SPARSE_V_VALUE rows what = .ok () := by
  unfold Flock.RecCoins.checkSparse at h
  dsimp only at h
  split at h
  · cases h
  · obtain ⟨⟨ash, extras⟩, halg, h⟩ := FlockVBridge.PartS.bind_ok h
    try dsimp only at h
    obtain ⟨rng, hrng, h⟩ := FlockVBridge.PartS.bind_ok h
    obtain ⟨rows, hrows, h⟩ := FlockVBridge.PartS.bind_ok h
    have hrng' : sparseRange inn c.unit.name = .ok rng := hrng
    have hrows' : sparseRowsOf inn c3 c.unit.name sh ash extras rng = .ok rows := hrows
    exact ⟨rows, _, by simp only [sparseRows, halg, hrng', hrows'], h⟩

theorem innerRows_coef {inn : Flock.RecCoins.Inner} {c3 : Option Flock.RecCoins.C3} {c : Flock.Circuit}
    {LANES H : ℕ} {cr dr : Array (Array ℕ)} (hsp : Flock.RecSparse.params c.unit.name = none)
    (ho : Flock.RecOpen.params c.unit.name = some (LANES, H)) (hr : openRows inn LANES H = some (cr, dr)) :
    innerRows inn c3 c "coef" = some (cr, Flock.RecCoins.COEF_VALUE) := by
  simp only [innerRows, hsp, ho, hr]
  rfl

theorem innerRows_dirs {inn : Flock.RecCoins.Inner} {c3 : Option Flock.RecCoins.C3} {c : Flock.Circuit}
    {LANES H : ℕ} {cr dr : Array (Array ℕ)} (hsp : Flock.RecSparse.params c.unit.name = none)
    (ho : Flock.RecOpen.params c.unit.name = some (LANES, H)) (hr : openRows inn LANES H = some (cr, dr)) :
    innerRows inn c3 c "dirs" = some (dr, Flock.RecCoins.DIRS_VALUE) := by
  simp only [innerRows, hsp, ho, hr]
  rfl

theorem innerRows_v {inn : Flock.RecCoins.Inner} {c3 : Option Flock.RecCoins.C3} {c : Flock.Circuit}
    {m k cl j : ℕ} {rows : Array (Array ℕ)} (hsp : Flock.RecSparse.params c.unit.name = none)
    (ho : Flock.RecOpen.params c.unit.name = none) (hres : Flock.RecResiduals.params c.unit.name = some (m, k, cl, j))
    (hr : residualRows inn j = some rows) : innerRows inn c3 c "v" = some (rows, Flock.RecCoins.V_VALUE) := by
  simp only [innerRows, hsp, ho, hres, hr]
  rfl

theorem innerRows_sparse {inn : Flock.RecCoins.Inner} {c3 : Flock.RecCoins.C3} {c : Flock.Circuit}
    {sh : Flock.RecSparse.Shape} {rows : Array (Array ℕ)} (hsp : Flock.RecSparse.params c.unit.name = some sh)
    (hr : sparseRows inn c3 c.unit.name sh = some rows) :
    innerRows inn (some c3) c "v" = some (rows, Flock.RecCoins.SPARSE_V_VALUE) := by
  simp only [innerRows, hsp, hr]
  rfl

/-- **Check 5 at a `rec-open` unit runs `checkOpen`** at the inner session and its parameters. -/
theorem openCheck_of_check {st : Flock.Stmt} {own : Array Flock.Registered.Port} {f : Flock.RecCoins.Source}
    {c3 : Option Flock.RecCoins.C3} {LANES H : ℕ} (h : Flock.RecCoins.check st (some own) (some f) c3 = .ok ())
    (hso : Flock.RecCoins.stackedParams st.c.unit.name = none) (hop : Flock.RecOpening.params st.c.unit.name = none)
    (hsp : Flock.RecSparse.params st.c.unit.name = none) (ho : Flock.RecOpen.params st.c.unit.name = some (LANES, H)) :
    ∃ inn, f.load st.c.unit.name = .ok inn ∧ Flock.RecCoins.checkOpen inn st.c st.pub own LANES H = .ok () := by
  have h' : (f.load st.c.unit.name >>= fun inn => Flock.RecCoins.checkOpen inn st.c st.pub own LANES H) = .ok () := by
    revert h
    unfold Flock.RecCoins.check
    simp [hso, hop, hsp, ho]
  exact FlockVBridge.PartS.bind_ok h'

/-- **Check 5 at a `rec-residuals` unit runs `checkResiduals`** at the inner session and its parameters. -/
theorem residualsCheck_of_check {st : Flock.Stmt} {own : Array Flock.Registered.Port} {f : Flock.RecCoins.Source}
    {c3 : Option Flock.RecCoins.C3} {m k cl j : ℕ} (h : Flock.RecCoins.check st (some own) (some f) c3 = .ok ())
    (hso : Flock.RecCoins.stackedParams st.c.unit.name = none) (hop : Flock.RecOpening.params st.c.unit.name = none)
    (hsp : Flock.RecSparse.params st.c.unit.name = none) (ho : Flock.RecOpen.params st.c.unit.name = none)
    (hres : Flock.RecResiduals.params st.c.unit.name = some (m, k, cl, j)) :
    ∃ inn, f.load st.c.unit.name = .ok inn ∧ Flock.RecCoins.checkResiduals inn st.c st.pub own m k cl j = .ok () := by
  have h' : (f.load st.c.unit.name >>= fun inn => Flock.RecCoins.checkResiduals inn st.c st.pub own m k cl j) = .ok () := by
    revert h
    unfold Flock.RecCoins.check
    simp [hso, hop, hsp, ho, hres]
  exact FlockVBridge.PartS.bind_ok h'

/-- **Check 5 at a `rec-sparse` unit runs `checkSparse`** at the inner session, candidate 3's inputs and its shape. -/
theorem sparseCheck_of_check {st : Flock.Stmt} {own : Array Flock.Registered.Port} {f : Flock.RecCoins.Source}
    {c3 : Flock.RecCoins.C3} {sh : Flock.RecSparse.Shape} (h : Flock.RecCoins.check st (some own) (some f) (some c3) = .ok ())
    (hso : Flock.RecCoins.stackedParams st.c.unit.name = none) (hop : Flock.RecOpening.params st.c.unit.name = none)
    (hsp : Flock.RecSparse.params st.c.unit.name = some sh) :
    ∃ inn, f.load st.c.unit.name = .ok inn ∧ Flock.RecCoins.checkSparse inn c3 st.c st.pub own sh = .ok () := by
  have h' : (f.load st.c.unit.name >>= fun inn => Flock.RecCoins.checkSparse inn c3 st.c st.pub own sh) = .ok () := by
    revert h
    unfold Flock.RecCoins.check
    simp [hso, hop, hsp]
  exact FlockVBridge.PartS.bind_ok h'

end FlockSoundness.Discharge.Recursion
