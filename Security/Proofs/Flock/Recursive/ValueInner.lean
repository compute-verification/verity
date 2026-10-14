import Proofs.Flock.Recursive.ValueSource

/-!
# The verifier's rows at the inner session's units

`valueSrcAll` at `rec-open`, `rec-residuals` and `rec-sparse` (`openRows`, `residualRows`, `sparseRows`, defined beside
`valueSrcC3` in `Recursive/ValueSrc.lean`):

* `open_of_check`, `residuals_of_check`, `sparse_of_check`: an accepted unit's `checkValue`s are at those rows, which
  are `valueRowsAll`'s (`valueRowsAll_load`, `innerRows_coef`, `innerRows_dirs`, `innerRows_v`, `innerRows_sparse`), so
  `valueSrcAll_eq` gives `value_tie_coll`'s `hsrc`;
* `residualsBoth_of_check`, `sparse_of_check`: a part's `checkZero` at its `out`, at the zero rows `innerRows_out` and
  `innerRows_sparseOut` give, under `zeroSalt` (`valueSrcAll_zero`, `value_tie_collR`'s `hsrc`; `zeroRows_get`);
* `openCheck_of_check`, `residualsCheck_of_check`, `sparseCheck_of_check`: `RecCoins.check` runs those checks there.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit

variable {n : ℕ} {L : Law n} {Reg Cm Reg₂ : Type} {I : Inner} {S : ℕ} {Vs : Fin S → VStmt}

theorem valueSrcAll_eq {src : List (Cm × I.Coin) → Option Flock.RecCoins.Source}
    {c3 : List (Cm × I.Coin) → Option Flock.RecCoins.C3} {R : Reg} {ω : L.Ω} {t : List (Cm × I.Coin)} {R₂ : Reg₂}
    {j : Fin S} {u : Fin (Vs j).nv} {q : Fin (Vs j).c.ports.size} {rows : Array (Array ℕ)} {value prog : String}
    (hv : valueRowsAll (src t) (c3 t) (Vs j).c ((Vs j).c.ports[q.val]!).name = some (rows, value))
    (hprog : (Vs j).c.metaJ.getObjValAs? String "program_sha512" = .ok prog)
    (hz : ¬ (value = Flock.RecCoins.ZERO_VALUE ∨ value = Flock.RecCoins.SPARSE_ZERO_VALUE) := by decide) :
    valueSrcAll (L := L) (Reg := Reg) (Reg₂ := Reg₂) (Vs := Vs) src c3 R ω t R₂ ⟨j, u, q⟩ =
      some (rows, Flock.RecCoins.verifierSalt prog value) := by
  simp only [valueSrcAll, hv, hprog, if_neg hz]

theorem valueSrcAll_zero {src : List (Cm × I.Coin) → Option Flock.RecCoins.Source}
    {c3 : List (Cm × I.Coin) → Option Flock.RecCoins.C3} {R : Reg} {ω : L.Ω} {t : List (Cm × I.Coin)} {R₂ : Reg₂}
    {j : Fin S} {u : Fin (Vs j).nv} {q : Fin (Vs j).c.ports.size} {rows : Array (Array ℕ)} {value prog : String}
    (hv : valueRowsAll (src t) (c3 t) (Vs j).c ((Vs j).c.ports[q.val]!).name = some (rows, value))
    (hprog : (Vs j).c.metaJ.getObjValAs? String "program_sha512" = .ok prog)
    (hz : value = Flock.RecCoins.ZERO_VALUE ∨ value = Flock.RecCoins.SPARSE_ZERO_VALUE := by decide) :
    valueSrcAll (L := L) (Reg := Reg) (Reg₂ := Reg₂) (Vs := Vs) src c3 R ω t R₂ ⟨j, u, q⟩ =
      some (rows, Flock.RecCoins.zeroSalt value) := by
  simp only [valueSrcAll, hv, hprog, if_pos hz]

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

/-- **What an accepted `rec-residuals` unit checked**: its `v` at `residualRows`' rows, and its `out` at the zero rows
(`checkZero` at `rec-residuals-zero`). -/
theorem residualsBoth_of_check {inn : Flock.RecCoins.Inner} {c : Flock.Circuit} {pub : Flock.Public}
    {own : Array Flock.Registered.Port} {m k cl j : ℕ}
    (h : Flock.RecCoins.checkResiduals inn c pub own m k cl j = .ok ()) :
    ∃ rows what, residualRows inn j = some rows ∧
      Flock.RecCoins.checkValue c pub own "v" Flock.RecCoins.V_VALUE rows what = .ok () ∧
      Flock.RecCoins.checkZero c pub own Flock.RecCoins.ZERO_VALUE 2 Flock.RecCoins.OUT_WORDS = .ok () := by
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
        obtain ⟨⟨⟩, hcv, hz⟩ := FlockVBridge.PartS.bind_ok h
        exact ⟨rows, _, by simp only [residualRows, halg, hX', hj, hrows'], hcv, hz⟩
      · cases h
    · cases h

/-- **What an accepted `rec-residuals` unit checked of its `v`**: at `residualRows`' rows. -/
theorem residuals_of_check {inn : Flock.RecCoins.Inner} {c : Flock.Circuit} {pub : Flock.Public}
    {own : Array Flock.Registered.Port} {m k cl j : ℕ}
    (h : Flock.RecCoins.checkResiduals inn c pub own m k cl j = .ok ()) :
    ∃ rows what, residualRows inn j = some rows ∧
      Flock.RecCoins.checkValue c pub own "v" Flock.RecCoins.V_VALUE rows what = .ok () :=
  let ⟨rows, what, hr, hcv, _⟩ := residualsBoth_of_check h
  ⟨rows, what, hr, hcv⟩

/-- **What an accepted `rec-sparse` unit checked**: its `v` at `sparseRows`' rows, and its `out` at the zero rows
(`checkZero` at `rec-sparse-zero`). -/
theorem sparse_of_check {inn : Flock.RecCoins.Inner} {c3 : Flock.RecCoins.C3} {c : Flock.Circuit} {pub : Flock.Public}
    {own : Array Flock.Registered.Port} {sh : Flock.RecSparse.Shape}
    (h : Flock.RecCoins.checkSparse inn c3 c pub own sh = .ok ()) :
    ∃ rows what, sparseRows inn c3 c.unit.name sh = some rows ∧
      Flock.RecCoins.checkValue c pub own "v" Flock.RecCoins.SPARSE_V_VALUE rows what = .ok () ∧
      Flock.RecCoins.checkZero c pub own Flock.RecCoins.SPARSE_ZERO_VALUE 2 (Flock.RecSparse.outWords sh) = .ok () := by
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
    obtain ⟨⟨⟩, hcv, hz⟩ := FlockVBridge.PartS.bind_ok h
    exact ⟨rows, _, by simp only [sparseRows, halg, hrng', hrows'], hcv, hz⟩

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

theorem innerRows_out {inn : Flock.RecCoins.Inner} {c3 : Option Flock.RecCoins.C3} {c : Flock.Circuit}
    {m k cl j : ℕ} (hsp : Flock.RecSparse.params c.unit.name = none) (ho : Flock.RecOpen.params c.unit.name = none)
    (hres : Flock.RecResiduals.params c.unit.name = some (m, k, cl, j)) :
    innerRows inn c3 c "out" = some (Flock.RecCoins.zeroRows 2 Flock.RecCoins.OUT_WORDS, Flock.RecCoins.ZERO_VALUE) := by
  simp only [innerRows, hsp, ho, hres] <;> rfl

theorem innerRows_sparseOut {inn : Flock.RecCoins.Inner} {c3 : Flock.RecCoins.C3} {c : Flock.Circuit}
    {sh : Flock.RecSparse.Shape} (hsp : Flock.RecSparse.params c.unit.name = some sh) :
    innerRows inn (some c3) c "out" =
      some (Flock.RecCoins.zeroRows 2 (Flock.RecSparse.outWords sh), Flock.RecCoins.SPARSE_ZERO_VALUE) := by
  simp only [innerRows, hsp] <;> rfl

theorem zeroRows_get (n w u k : ℕ) : ((Flock.RecCoins.zeroRows n w)[u]!)[k]! = 0 := by
  have hrow : ∀ a : Array ℕ, (∀ i (h : i < a.size), a[i] = 0) → a[k]! = 0 := fun a ha => by
    by_cases hk : k < a.size
    · rw [getElem!_pos a k hk]; exact ha k hk
    · rw [getElem!_neg a k hk]; rfl
  refine hrow _ fun i h => ?_
  unfold Flock.RecCoins.zeroRows at h ⊢
  by_cases hu : u < n
  · simp only [getElem!_pos (Array.replicate n (Array.replicate w 0)) u (by simpa using hu), Array.getElem_replicate]
  · simp only [getElem!_neg (Array.replicate n (Array.replicate w 0)) u (by simpa using hu)] at h
    exact absurd h (Nat.not_lt_zero _)

/-- **A source loaded from the inner-session files has a decodable record**: `Inner.load` reaches
`Record.decode` only after setting up the statement and checking its shape and schedule. -/
theorem record_decode_of_load {unit : String} {f : Flock.RecCoins.Files} {inn : Flock.RecCoins.Inner}
    (h : Flock.RecCoins.Inner.load unit f = .ok inn) :
    ∃ spec sess, Flock.Record.decode spec f.record = .ok sess := by
  unfold Flock.RecCoins.Inner.load at h
  obtain ⟨⟨spec, st⟩, _, h⟩ := FlockVBridge.PartS.bind_ok h
  cases hi : f.inner with
  | none =>
      cases hs : Flock.Zk.schedule st.m with
      | error e => simp [hi, hs] at h
      | ok p =>
          cases hd : Flock.Record.decode spec f.record with
          | error e => simp [hi, hs, hd] at h
          | ok sess => exact ⟨spec, sess, hd⟩
  | some i =>
      cases hc : Flock.RecVStar.checkInnerStmt i st with
      | error e => simp [hi, hc] at h
      | ok u =>
          cases hs : Flock.Zk.schedule st.m with
          | error e => simp [hi, hc, hs] at h
          | ok p =>
              cases hd : Flock.Record.decode spec f.record with
              | error e => simp [hi, hc, hs, hd] at h
              | ok sess => exact ⟨spec, sess, hd⟩

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
