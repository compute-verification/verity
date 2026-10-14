import Proofs.Flock.Recursive.ValueFinder

/-!
# The verifier's own rows as `valueFinder`'s source

`valueSrcC3`: at a read `(j, u, q)` of a history `t`, the rows check 5 registers itself at port `q` of V*'s statement
`j` (`valueRows`, at the verifier's candidate-3 inputs `c3 t`), under `RecCoins.verifierSalt` of META's
`program_sha512` and the value's name. `valueRows` computes the rows as the check does: a level of R's openings
(`rec-open-stacked`, `RecCoins.checkStackedLevel`: `coef`, `dirs`) and a part of R's opening (`rec-opening`,
`RecCoins.checkOpening`: `v`), and none elsewhere. The finder's budget counts each row's words once and each salt's
SHAKE256 once.

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

variable {n : ℕ} {L : Law n} {Reg Cm Reg₂ : Type} {I : Inner} {S : ℕ} {Vs : Fin S → VStmt}

/-- **The verifier's rows as a value source**, at its candidate-3 inputs `c3 t` at each history. -/
def valueSrcC3 (c3 : List (Cm × I.Coin) → Option Flock.RecCoins.C3) : ValueSrc L Reg Cm Reg₂ I Vs :=
  fun _ _ t _ x =>
    match valueRows (c3 t) (Vs x.1).c ((Vs x.1).c.ports[x.2.2.val]!).name,
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

#print axioms FlockSoundness.Discharge.Recursion.valueSrcC3_eq
#print axioms FlockSoundness.Discharge.Recursion.stacked_of_check
#print axioms FlockSoundness.Discharge.Recursion.opening_of_check
#print axioms FlockSoundness.Discharge.Recursion.valueRows_coef
#print axioms FlockSoundness.Discharge.Recursion.valueRows_dirs
#print axioms FlockSoundness.Discharge.Recursion.valueRows_v
#print axioms FlockSoundness.Discharge.Recursion.stackedLevel_of_check
#print axioms FlockSoundness.Discharge.Recursion.openingCheck_of_check
