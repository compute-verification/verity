import Proofs.Flock.Recursive.ValueSource

/-!
# The verifier's rows at the inner session's units

`valueSrcAll`: `valueSrcC3`'s source, with the rows check 5 registers at the units it checks against the inner session
(`RecCoins.check`'s `files`, `src t` at each history): `coef` and `dirs` at `rec-open` (`openRows`, `RecCoins.checkOpen`),
`v` at `rec-residuals` (`residualRows`, `RecCoins.checkResiduals`) and at `rec-sparse` (`sparseRows`,
`RecCoins.checkSparse`). `valueRowsAll` is `valueRows` at `rec-open-stacked` and `rec-opening`, and `innerRows` at the
inner session's load elsewhere.

* `open_of_check`, `residuals_of_check`, `sparse_of_check`: an accepted unit's `checkValue`s are at those rows, which
  are `valueRowsAll`'s (`valueRowsAll_load`, `innerRows_coef`, `innerRows_dirs`, `innerRows_v`, `innerRows_sparse`), so
  `valueSrcAll_eq` gives `value_tie_coll`'s `hsrc`;
* `openCheck_of_check`, `residualsCheck_of_check`, `sparseCheck_of_check`: `RecCoins.check` runs those checks there.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit

/-- The levels of the inner schedule with `LANES` lanes and `H` siblings below their cap. -/
def innerLevels (sh : Flock.Shape) (LANES H : ℕ) : List ℕ :=
  (List.range sh.sched.levels.size).filter fun l =>
    2 ^ sh.sched.levels[l]!.lanes == LANES && sh.sched.levels[l]!.d - sh.sched.levels[l]!.cap == H

/-- Level `l`'s `coef` rows: rep `i / q`'s opening `i % q`'s `coefAt (κ[l]) LANES eq(α_l)[j]` in words. -/
def openCoef (sh : Flock.Shape) (l LANES : ℕ) (r0 r1 : Flock.RecCoins.Rep) : Array (Array ℕ) :=
  (Array.range (2 * sh.sched.levels[l]!.q)).map fun i =>
    let (rep, j) := (if i < sh.sched.levels[l]!.q then r0 else r1, i % sh.sched.levels[l]!.q)
    Flock.RecCoins.words (Flock.RecCoins.coefAt (rep.kappa[l]!) LANES (Flock.eqTable (rep.info.lg.conss[l]!).alpha)[j]!)

/-- Level `l`'s `dirs` rows: rep `i / q`'s opening `i % q`'s position's low `H` bits. -/
def openDirs (sh : Flock.Shape) (l H : ℕ) (r0 r1 : Flock.RecCoins.Rep) : Array (Array ℕ) :=
  (Array.range (2 * sh.sched.levels[l]!.q)).map fun i =>
    let (rep, j) := (if i < sh.sched.levels[l]!.q then r0 else r1, i % sh.sched.levels[l]!.q)
    Flock.RecCoins.dirsRow (rep.info.lg.conss[l]!).pos[j]! H

/-- `rec-open`'s `coef` and `dirs` rows at the inner session `inn`. -/
def openRows (inn : Flock.RecCoins.Inner) (LANES H : ℕ) : Option (Array (Array ℕ) × Array (Array ℕ)) :=
  match inn.algebra with
  | .error _ => none
  | .ok (sh, extras) => match innerLevels sh LANES H with
    | [l] => match inn.rep sh extras 0, inn.rep sh extras 1 with
      | .ok r0, .ok r1 => some (openCoef sh l LANES r0 r1, openDirs sh l H r0 r1)
      | _, _ => none
    | _ => none

/-- The parts of the inner algebra's v. -/
def residualGroups (sh : Flock.Shape) : Except String (Array (Array ℕ × Array ℕ × Array ℕ)) :=
  do Flock.RecResiduals.groups (← Flock.RecAlgebra.structureOf sh) (Flock.RecResiduals.portSizes sh)

/-- Each rep's `v` elements `vs` (the comb last), padded to the port's words. -/
def residualRowsOf (inn : Flock.RecCoins.Inner) (sh : Flock.Shape) (extras : Array Flock.Claim) (vs : Array ℕ) :
    Flock.V (Array (Array ℕ)) :=
  #[0, 1].mapM fun r => do
    let rep ← inn.rep sh extras r
    let base := rep.v.extract 0 (rep.v.size - 64)
    let comb ← if vs.any (· ≥ base.size) then inn.comb r rep else pure Flock.RecCoins.ZERO_COMB
    let full := base ++ comb
    pure (Flock.RecCoins.padTo (Flock.RecCoins.words (vs.map (full[·]!))) (Flock.RecResiduals.portWordsOf vs.size))

/-- `rec-residuals`' `v` rows of part `j` at the inner session `inn`. -/
def residualRows (inn : Flock.RecCoins.Inner) (j : ℕ) : Option (Array (Array ℕ)) :=
  match inn.algebra with
  | .error _ => none
  | .ok (sh, extras) => match residualGroups sh with
    | .error _ => none
    | .ok gs => match gs[j]? with
      | none => none
      | some (_, _, vs) => match residualRowsOf inn sh extras vs with
        | .ok rows => some rows
        | .error _ => none

/-- The inner circuit's unit range, as `checkSparse` reads it at unit `unit`. -/
def sparseRange (inn : Flock.RecCoins.Inner) (unit : String) : Flock.V Flock.Range :=
  inn.unitRange.mapError (s!"rec-coins: {unit}: " ++ ·)

/-- Each rep's RecSparse `v` (`sparseV`) at its inner points and sparse coins, padded to the port's words. -/
def sparseRowsOf (inn : Flock.RecCoins.Inner) (c3 : Flock.RecCoins.C3) (unit : String) (sh : Flock.RecSparse.Shape)
    (ash : Flock.Shape) (extras : Array Flock.Claim) (rng : Flock.Range) : Flock.V (Array (Array ℕ)) :=
  #[0, 1].mapM fun r => do
    let rep ← inn.rep ash extras r
    let sc ← Flock.RecCoins.sparseCoins c3 unit r
    let i := rep.info
    let v ← (do Flock.RecCoins.sparseV sh sc i.lc.fold.alpha i.zc.z i.lc.fold.rhoIn i.lc.rIn
      (← Flock.RecCoins.layoutFactors sh i.lc.fold.rhoIn i.lc.rIn rng)).mapError (s!"rec-coins: {unit}: rep {r}: " ++ ·)
    pure (Flock.RecCoins.padTo (Flock.RecCoins.words v) (Flock.RecSparse.portWords (Flock.RecSparse.nv sh)))

/-- `rec-sparse`'s `v` rows at the inner session `inn` and candidate 3's inputs `c3`. -/
def sparseRows (inn : Flock.RecCoins.Inner) (c3 : Flock.RecCoins.C3) (unit : String) (sh : Flock.RecSparse.Shape) :
    Option (Array (Array ℕ)) :=
  match inn.algebra, sparseRange inn unit with
  | .ok (ash, extras), .ok rng => match sparseRowsOf inn c3 unit sh ash extras rng with
    | .ok rows => some rows
    | .error _ => none
  | _, _ => none

/-- The rows and value check 5 registers at port `port` of a unit it checks against the inner session `inn`. -/
def innerRows (inn : Flock.RecCoins.Inner) (c3 : Option Flock.RecCoins.C3) (c : Flock.Circuit) (port : String) :
    Option (Array (Array ℕ) × String) :=
  match Flock.RecSparse.params c.unit.name, Flock.RecOpen.params c.unit.name, Flock.RecResiduals.params c.unit.name with
  | some sh, _, _ => match c3 with
    | some c3 => if port = "v" then (sparseRows inn c3 c.unit.name sh).map (·, Flock.RecCoins.SPARSE_V_VALUE) else none
    | none => none
  | none, some (LANES, H), _ => match openRows inn LANES H with
    | some (cr, dr) => if port = "coef" then some (cr, Flock.RecCoins.COEF_VALUE)
      else if port = "dirs" then some (dr, Flock.RecCoins.DIRS_VALUE) else none
    | none => none
  | none, none, some (_, _, _, j) => if port = "v" then (residualRows inn j).map (·, Flock.RecCoins.V_VALUE) else none
  | none, none, none => none

/-- `innerRows` at the inner session `src` loads. -/
def loadRows (src : Option Flock.RecCoins.Source) (c3 : Option Flock.RecCoins.C3) (c : Flock.Circuit) (port : String) :
    Option (Array (Array ℕ) × String) :=
  match src with
  | none => none
  | some f => match f.load c.unit.name with
    | .error _ => none
    | .ok inn => innerRows inn c3 c port

/-- **The rows and value check 5 registers at port `port` of circuit `c`**, at its inner session `src` and candidate-3
inputs `c3`. -/
def valueRowsAll (src : Option Flock.RecCoins.Source) (c3 : Option Flock.RecCoins.C3) (c : Flock.Circuit)
    (port : String) : Option (Array (Array ℕ) × String) :=
  match Flock.RecCoins.stackedParams c.unit.name, Flock.RecOpening.params c.unit.name with
  | none, none => loadRows src c3 c port
  | _, _ => valueRows c3 c port

variable {n : ℕ} {L : Law n} {Reg Cm Reg₂ : Type} {I : Inner} {S : ℕ} {Vs : Fin S → VStmt}

/-- **The verifier's rows as a value source**, at its inner session `src t` and candidate-3 inputs `c3 t`. -/
def valueSrcAll (src : List (Cm × I.Coin) → Option Flock.RecCoins.Source)
    (c3 : List (Cm × I.Coin) → Option Flock.RecCoins.C3) : ValueSrc L Reg Cm Reg₂ I Vs :=
  fun _ _ t _ x =>
    match valueRowsAll (src t) (c3 t) (Vs x.1).c ((Vs x.1).c.ports[x.2.2.val]!).name,
      (Vs x.1).c.metaJ.getObjValAs? String "program_sha512" with
    | some (rows, value), .ok prog => some (rows, Flock.RecCoins.verifierSalt prog value)
    | _, _ => none

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
    exact ⟨_, _, by simp only [openRows, halg, hls', h0, h1],
      ⟨_, (hcv : Flock.RecCoins.checkValue c pub own "coef" Flock.RecCoins.COEF_VALUE (openCoef sh l LANES r0 r1) _ = _)⟩,
      _, (hdv : Flock.RecCoins.checkValue c pub own "dirs" Flock.RecCoins.DIRS_VALUE (openDirs sh l H r0 r1) _ = _)⟩
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

#print axioms FlockSoundness.Discharge.Recursion.valueSrcAll_eq
#print axioms FlockSoundness.Discharge.Recursion.valueRowsAll_stacked
#print axioms FlockSoundness.Discharge.Recursion.valueRowsAll_opening
#print axioms FlockSoundness.Discharge.Recursion.valueRowsAll_load
#print axioms FlockSoundness.Discharge.Recursion.open_of_check
#print axioms FlockSoundness.Discharge.Recursion.residuals_of_check
#print axioms FlockSoundness.Discharge.Recursion.sparse_of_check
#print axioms FlockSoundness.Discharge.Recursion.innerRows_coef
#print axioms FlockSoundness.Discharge.Recursion.innerRows_dirs
#print axioms FlockSoundness.Discharge.Recursion.innerRows_v
#print axioms FlockSoundness.Discharge.Recursion.innerRows_sparse
#print axioms FlockSoundness.Discharge.Recursion.openCheck_of_check
#print axioms FlockSoundness.Discharge.Recursion.residualsCheck_of_check
#print axioms FlockSoundness.Discharge.Recursion.sparseCheck_of_check
