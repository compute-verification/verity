import Proofs.Flock.Recursive.ValueFinder

/-!
# The verifier's own rows as `valueFinder`'s source

`valueSrcC3`: at a read `(j, u, q)` of a history `t`, the rows check 5 registers itself at port `q` of V*'s statement
`j` (`valueRows`, at the verifier's candidate-3 inputs `c3 t`), under `RecCoins.verifierSalt` of META's
`program_sha512` and the value's name. `valueRows` computes the rows as the check does: a level of R's openings
(`rec-open-stacked`, `RecCoins.checkStackedLevel`: `coef`, `dirs`) and a part of R's opening (`rec-opening`,
`RecCoins.checkOpening`: `v`), and none elsewhere. The finder's budget counts each row's words once and each salt's
SHAKE256 once.

`valueSrcAll`: `valueSrcC3` with the rows check 5 registers at the units it checks against the inner session (`src t`,
`RecCoins.check`'s `files`): `valueRowsAll` is `valueRows` at `rec-open-stacked` and `rec-opening`, and elsewhere
`innerRows` at the session's load: `coef` and `dirs` at `rec-open` (`openRows`, `RecCoins.checkOpen`), `v` at
`rec-residuals` (`residualRows`, `RecCoins.checkResiduals`) and at `rec-sparse` (`sparseRows`, `RecCoins.checkSparse`).
Their check ties are `ValueInner.lean`'s.

* `stacked_of_check`, `opening_of_check`: an accepted unit's `checkValue`s are at `valueRows`' rows
  (`valueRows_coef`, `valueRows_dirs`, `valueRows_v`), so `valueSrcC3_eq` gives `value_tie_coll`'s `hsrc`;
* `stackedLevel_of_check`, `openingCheck_of_check`: `RecCoins.check` runs those checks at those units.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit

/-- Level `l`'s `coef` rows at R's opening `o`: opening `j`'s `coefAt (κ[l]) LANES eq(α_l)[j]` in words. -/
def stackedCoefRows (o : Flock.RecCoins.Opening) (l LANES : ℕ) : Array (Array ℕ) :=
  (Array.range (o.sh.sched.levels[l]!).q).map fun j =>
    Flock.RecCoins.words (Flock.RecCoins.coefAt (o.kappa[l]!) LANES (Flock.eqTable (o.lg.conss[l]!).alpha)[j]!)

/-- Level `l`'s `dirs` rows at R's opening `o`: opening `j`'s position's low `H` bits. -/
def stackedDirsRows (o : Flock.RecCoins.Opening) (l H : ℕ) : Array (Array ℕ) :=
  (Array.range (o.sh.sched.levels[l]!).q).map fun j => Flock.RecCoins.dirsRow (o.lg.conss[l]!).pos[j]! H

/-- The levels of R's schedule with `LANES` lanes and `H` siblings below their cap. -/
def stackedLevels (o : Flock.RecCoins.Opening) (LANES H : ℕ) : List ℕ :=
  (List.range o.sh.sched.levels.size).filter fun l =>
    2 ^ (o.sh.sched.levels[l]!).lanes == LANES && (o.sh.sched.levels[l]!).d - (o.sh.sched.levels[l]!).cap == H

/-- Part `j`'s `v` row of R's opening `o`. -/
def openingVRows (o : Flock.RecCoins.Opening) (j : ℕ) : Option (Array (Array ℕ)) :=
  match (do Flock.RecResiduals.groups (← Flock.RecOpening.structureOf o.sh) (Flock.RecOpening.portSizes o.sh)) with
  | .ok gs => match gs[j]? with
    | some (_, _, vs) => some #[Flock.RecCoins.padTo (Flock.RecCoins.words (vs.map (o.v[·]!))) (Flock.RecResiduals.portWordsOf vs.size)]
    | none => none
  | .error _ => none

/-- **The rows and value check 5 registers at port `port` of circuit `c`**, at its candidate-3 inputs `c3`. -/
def valueRows (c3 : Option Flock.RecCoins.C3) (c : Flock.Circuit) (port : String) : Option (Array (Array ℕ) × String) :=
  match c3 with
  | none => none
  | some c3 => match Flock.RecCoins.opening c3 c.unit.name with
    | .error _ => none
    | .ok o => match Flock.RecCoins.stackedParams c.unit.name, Flock.RecOpening.params c.unit.name with
      | some (LANES, H), _ => match stackedLevels o LANES H with
        | [l] => if port = "coef" then some (stackedCoefRows o l LANES, Flock.RecCoins.COEF_VALUE)
          else if port = "dirs" then some (stackedDirsRows o l H, Flock.RecCoins.DIRS_VALUE) else none
        | _ => none
      | none, some (_, _, j) => if port = "v" then (openingVRows o j).map (·, Flock.RecCoins.V_VALUE) else none
      | none, none => none

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

/-- **The verifier's rows as a value source**, at its candidate-3 inputs `c3 t` at each history. -/
def valueSrcC3 (c3 : List (Cm × I.Coin) → Option Flock.RecCoins.C3) : ValueSrc L Reg Cm Reg₂ I Vs :=
  fun _ _ t _ x =>
    match valueRows (c3 t) (Vs x.1).c ((Vs x.1).c.ports[x.2.2.val]!).name,
      (Vs x.1).c.metaJ.getObjValAs? String "program_sha512" with
    | some (rows, value), .ok prog => some (rows, Flock.RecCoins.verifierSalt prog value)
    | _, _ => none

/-- **The verifier's rows as a value source**, at its inner session `src t` and candidate-3 inputs `c3 t`. -/
def valueSrcAll (src : List (Cm × I.Coin) → Option Flock.RecCoins.Source)
    (c3 : List (Cm × I.Coin) → Option Flock.RecCoins.C3) : ValueSrc L Reg Cm Reg₂ I Vs :=
  fun _ _ t _ x =>
    match valueRowsAll (src t) (c3 t) (Vs x.1).c ((Vs x.1).c.ports[x.2.2.val]!).name,
      (Vs x.1).c.metaJ.getObjValAs? String "program_sha512" with
    | some (rows, value), .ok prog => some (rows, Flock.RecCoins.verifierSalt prog value)
    | _, _ => none

theorem valueSrcC3_eq {c3 : List (Cm × I.Coin) → Option Flock.RecCoins.C3} {R : Reg} {ω : L.Ω} {t : List (Cm × I.Coin)}
    {R₂ : Reg₂} {j : Fin S} {u : Fin (Vs j).nv} {q : Fin (Vs j).c.ports.size} {rows : Array (Array ℕ)}
    {value prog : String} (hv : valueRows (c3 t) (Vs j).c ((Vs j).c.ports[q.val]!).name = some (rows, value))
    (hprog : (Vs j).c.metaJ.getObjValAs? String "program_sha512" = .ok prog) :
    valueSrcC3 (L := L) (Reg := Reg) (Reg₂ := Reg₂) (Vs := Vs) c3 R ω t R₂ ⟨j, u, q⟩ =
      some (rows, Flock.RecCoins.verifierSalt prog value) := by
  simp only [valueSrcC3, hv, hprog]

/-- **What an accepted `rec-open-stacked` unit checked**: R's opening `o`, the one level `l` of its parameters, and
its `coef` and `dirs` at that level's rows. -/
theorem stacked_of_check {c3 : Flock.RecCoins.C3} {c : Flock.Circuit} {pub : Flock.Public} {own : Array Flock.Registered.Port}
    {LANES H : ℕ} (h : Flock.RecCoins.checkStackedLevel c3 c pub own LANES H = .ok ()) :
    ∃ o l, Flock.RecCoins.opening c3 c.unit.name = .ok o ∧ stackedLevels o LANES H = [l] ∧
      (∃ what, Flock.RecCoins.checkValue c pub own "coef" Flock.RecCoins.COEF_VALUE (stackedCoefRows o l LANES) what = .ok ()) ∧
      ∃ what, Flock.RecCoins.checkValue c pub own "dirs" Flock.RecCoins.DIRS_VALUE (stackedDirsRows o l H) what = .ok () := by
  unfold Flock.RecCoins.checkStackedLevel at h
  dsimp only at h
  obtain ⟨o, hop, h⟩ := FlockVBridge.PartS.bind_ok h
  split at h
  · rename_i l hls
    split at h
    · cases h
    · obtain ⟨_u, hcv, hdv⟩ := FlockVBridge.PartS.bind_ok h
      exact ⟨o, l, hop, hls, ⟨_, hcv⟩, _, hdv⟩
  · cases h

/-- **What an accepted `rec-opening` unit checked**: R's opening `o` and its `v` at part `j`'s row. -/
theorem opening_of_check {c3 : Flock.RecCoins.C3} {c : Flock.Circuit} {pub : Flock.Public} {own : Array Flock.Registered.Port}
    {m cl j : ℕ} (h : Flock.RecCoins.checkOpening c3 c pub own m cl j = .ok ()) :
    ∃ o rows what, Flock.RecCoins.opening c3 c.unit.name = .ok o ∧ openingVRows o j = some rows ∧
      Flock.RecCoins.checkValue c pub own "v" Flock.RecCoins.V_VALUE rows what = .ok () := by
  unfold Flock.RecCoins.checkOpening at h
  dsimp only at h
  obtain ⟨o, hop, h⟩ := FlockVBridge.PartS.bind_ok h
  split at h
  · cases h
  · split at h
    · rename_i gs hX
      obtain ⟨gs', hgs, h⟩ := FlockVBridge.PartS.bind_ok h
      cases hgs
      split at h
      · rename_i hj
        split at h
        · cases h
        · exact ⟨o, _, _, hop, by simp only [openingVRows, hX, hj], h⟩
      · cases h
    · cases h

theorem valueRows_coef {c3 : Flock.RecCoins.C3} {c : Flock.Circuit} {LANES H l : ℕ} {o : Flock.RecCoins.Opening}
    (hso : Flock.RecCoins.stackedParams c.unit.name = some (LANES, H)) (hop : Flock.RecCoins.opening c3 c.unit.name = .ok o)
    (hls : stackedLevels o LANES H = [l]) :
    valueRows (some c3) c "coef" = some (stackedCoefRows o l LANES, Flock.RecCoins.COEF_VALUE) := by
  simp only [valueRows, hop, hso, hls]
  rfl

theorem valueRows_dirs {c3 : Flock.RecCoins.C3} {c : Flock.Circuit} {LANES H l : ℕ} {o : Flock.RecCoins.Opening}
    (hso : Flock.RecCoins.stackedParams c.unit.name = some (LANES, H)) (hop : Flock.RecCoins.opening c3 c.unit.name = .ok o)
    (hls : stackedLevels o LANES H = [l]) :
    valueRows (some c3) c "dirs" = some (stackedDirsRows o l H, Flock.RecCoins.DIRS_VALUE) := by
  simp only [valueRows, hop, hso, hls]
  rfl

theorem valueRows_v {c3 : Flock.RecCoins.C3} {c : Flock.Circuit} {m cl j : ℕ} {o : Flock.RecCoins.Opening}
    {rows : Array (Array ℕ)} (hso : Flock.RecCoins.stackedParams c.unit.name = none)
    (hp : Flock.RecOpening.params c.unit.name = some (m, cl, j)) (hop : Flock.RecCoins.opening c3 c.unit.name = .ok o)
    (hr : openingVRows o j = some rows) :
    valueRows (some c3) c "v" = some (rows, Flock.RecCoins.V_VALUE) := by
  simp only [valueRows, hop, hso, hp, hr]
  rfl

/-- **Check 5 at a `rec-open-stacked` unit runs `checkStackedLevel`** at its parameters. -/
theorem stackedLevel_of_check {st : Flock.Stmt} {own : Array Flock.Registered.Port} {files : Option Flock.RecCoins.Source}
    {c3 : Flock.RecCoins.C3} {LANES H : ℕ} (h : Flock.RecCoins.check st (some own) files (some c3) = .ok ())
    (hso : Flock.RecCoins.stackedParams st.c.unit.name = some (LANES, H)) :
    Flock.RecCoins.checkStackedLevel c3 st.c st.pub own LANES H = .ok () := by
  revert h
  unfold Flock.RecCoins.check
  simp [hso]

/-- **Check 5 at a `rec-opening` unit runs `checkOpening`** at its parameters. -/
theorem openingCheck_of_check {st : Flock.Stmt} {own : Array Flock.Registered.Port} {files : Option Flock.RecCoins.Source}
    {c3 : Flock.RecCoins.C3} {m cl j : ℕ} (h : Flock.RecCoins.check st (some own) files (some c3) = .ok ())
    (hso : Flock.RecCoins.stackedParams st.c.unit.name = none) (hp : Flock.RecOpening.params st.c.unit.name = some (m, cl, j)) :
    Flock.RecCoins.checkOpening c3 st.c st.pub own m cl j = .ok () := by
  revert h
  unfold Flock.RecCoins.check
  simp [hso, hp]

end FlockSoundness.Discharge.Recursion
