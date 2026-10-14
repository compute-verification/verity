import Proofs.Flock.Recursive.TopFinder
import Proofs.Flock.C3.RegLevels.ValueTie

/-!
# A one-run finder for a registrant's row off a value the verifier registers itself

`FlockC3.RegLevels.value_tie` ties a registrant's row at a value the verifier registers itself (`coef`, `dirs`, `v`) to
the verifier's row, or `valueColl` outputs a SHA-512 collision at that read. That collision is not `RegColl`'s (two
sessions' records) nor `TopColl`'s (the held top), so it is a case of its own, priced by one more finder of the
recursive audit and union-bounded with theirs:

* `ValueSrc`: the verifier's rows and salts at each read `(j, u, q)` (V*'s session `j`, unit `u`, row port `q`) of a
  history, or none where the port holds none of the verifier's values;
* `ValueColl`, `valueFinder`, `prob_valueColl_le`: some registered read of some session where `valueColl` collides; the
  finder that plays the audit once and outputs `valueColl`'s pair at a read the collision test passes (at a read off a
  verifier's port, or where `value_tie`'s premises fail, its output is unconstrained, so the test is the filter);
  under `cr/sha-512` (`Finder.CR` at `qV`), at most `qV²/2^513` likely, once per audit;
* `prob_valueColl_fork`: the fork form, the finder finds a collision or the event never happens;
* `valueColl_of_tie`, `value_tie_coll`: `value_tie`'s collision disjunct is `ValueColl`;
* `prob_regTopValue_le`: `RegColl`, the held top's `TopColl` and `ValueColl` union-bounded. A further case (C's level-0
  salt finder) is one more `prob_or_le` around it.

An honest budget for `valueFinder` counts the audit's run and, at each read the search tests, what `valueColl`
recomputes: at most `7n + 2D + 5` SHA-512 evaluations for `n` rows and `D = log2ceil n` (`3n + 2^(D+1) + 2D + 7`, and
`2^D < 2n`), plus the `n` salts' SHAKE256 (`RecCoins.verifierSalt`), which are not SHA-512.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit Refine StrictCR
open scoped ENNReal

variable {n : ℕ} {L : Law n} {Reg Cm Reg₂ Reg' : Type} {I : Inner} {S : ℕ} {Vs : Fin S → VStmt}

/-- **A read of V*'s sessions**: the session `j`, the unit `u` and the row port `q`, read at `.inl (u, q)`. -/
abbrev VRead (Vs : Fin S → VStmt) : Type := (j : Fin S) × Fin (Vs j).nv × Fin (Vs j).c.ports.size

/-- **The verifier's rows of the values it registers itself**: at each history `(R, ω, t)`, registration `R₂` and read
`(j, u, q)`, the rows and their salts `RecCoins.checkValue` holds the port to, or none. -/
abbrev ValueSrc (L : Law n) (Reg Cm Reg₂ : Type) (I : Inner) (Vs : Fin S → VStmt) : Type :=
  Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → VRead Vs → Option (Array (Array ℕ) × (ℕ → ByteArray))

/-- **`valueColl` at a read**: at the read `x = (j, u, q)`, against the verifier's rows and salts there, the registrant's
row bytes (`encPortH` of its value), salt and path; nothing where the source has no rows. -/
noncomputable def valueAt (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (vsrc : ValueSrc L Reg Cm Reg₂ I Vs) (R : Reg) (ω : L.Ω) (t : List (Cm × I.Coin)) (R₂ : Reg₂) (x : VRead Vs) :
    Pair :=
  match vsrc R ω t R₂ x with
  | none => none
  | some (rows, salt) =>
    FlockC3.RegLevels.valueColl ((zs R ω t R₂ x.1).rd.port (.inl x.2)) rows salt x.2.1.val
      (FlockSoundness.Discharge.HiddenExec.encPortH (Vs x.1).c.ports[x.2.2.val]!
        (FlockVBridge.Decode.regVal (zs R ω t R₂ x.1) (.inl x.2)))
      ((zs R ω t R₂ x.1).rd.ws (.inl x.2)) ((zs R ω t R₂ x.1).rd.wpath (.inl x.2))

theorem valueAt_of_some {zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {vsrc : ValueSrc L Reg Cm Reg₂ I Vs} {R : Reg} {ω : L.Ω} {t : List (Cm × I.Coin)} {R₂ : Reg₂} {j : Fin S}
    {u : Fin (Vs j).nv} {q : Fin (Vs j).c.ports.size} {rows : Array (Array ℕ)} {salt : ℕ → ByteArray}
    (hsrc : vsrc R ω t R₂ ⟨j, u, q⟩ = some (rows, salt)) :
    valueAt zs vsrc R ω t R₂ ⟨j, u, q⟩ =
      FlockC3.RegLevels.valueColl ((zs R ω t R₂ j).rd.port (.inl (u, q))) rows salt u.val
        (FlockSoundness.Discharge.HiddenExec.encPortH (Vs j).c.ports[q.val]!
          (FlockVBridge.Decode.regVal (zs R ω t R₂ j) (.inl (u, q))))
        ((zs R ω t R₂ j).rd.ws (.inl (u, q))) ((zs R ω t R₂ j).rd.wpath (.inl (u, q))) := by
  first
    | (unfold valueAt; rw [hsrc]; done)
    | (unfold valueAt; rw [hsrc]; rfl)
    | simp only [valueAt, hsrc]

/-- **A registrant's row is off a value the verifier registers itself, by a collision**: at the history `(R, ω, t)`
and registration `R₂`, some registered read of some session at which `valueAt` is a collision of SHA-512. -/
def ValueColl (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (vsrc : ValueSrc L Reg Cm Reg₂ I Vs) (R : Reg) (ω : L.Ω) (t : List (Cm × I.Coin)) (R₂ : Reg₂) : Prop :=
  ∃ x : VRead Vs, (zs R ω t R₂ x.1).rd.isRead (.inl x.2) ∧ IsColl Refine.H512 (valueAt zs vsrc R ω t R₂ x)

/-- **`value_tie`'s collision is `ValueColl`**: at a registered read `(u, q)` of session `j` where the source gives
`rows` and `salt`, a collision of `valueColl` there. -/
theorem valueColl_of_tie {zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {vsrc : ValueSrc L Reg Cm Reg₂ I Vs} {R : Reg} {ω : L.Ω} {t : List (Cm × I.Coin)} {R₂ : Reg₂} {j : Fin S}
    {u : Fin (Vs j).nv} {q : Fin (Vs j).c.ports.size} {rows : Array (Array ℕ)} {salt : ℕ → ByteArray}
    (hsrc : vsrc R ω t R₂ ⟨j, u, q⟩ = some (rows, salt)) (hp : (zs R ω t R₂ j).rd.isRead (.inl (u, q)))
    (h : IsColl Refine.H512 (FlockC3.RegLevels.valueColl ((zs R ω t R₂ j).rd.port (.inl (u, q))) rows salt u.val
      (FlockSoundness.Discharge.HiddenExec.encPortH (Vs j).c.ports[q.val]!
        (FlockVBridge.Decode.regVal (zs R ω t R₂ j) (.inl (u, q))))
      ((zs R ω t R₂ j).rd.ws (.inl (u, q))) ((zs R ω t R₂ j).rd.wpath (.inl (u, q))))) :
    ValueColl zs vsrc R ω t R₂ :=
  ⟨⟨j, u, q⟩, hp, by rw [valueAt_of_some hsrc]; exact h⟩

/-- **`value_tie` in `ValueColl`'s terms**: at a registered read `(u, q)` of a port without segments of session `j`,
where `checkValue` accepted the port at the verifier's `rows` and the source gives them under `verifierSalt`, the read
is of row `u` and the registrant's bit `b` is bit `b % 16` of the row's word `b / 16`, or `ValueColl`. -/
theorem value_tie_coll {zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {vsrc : ValueSrc L Reg Cm Reg₂ I Vs} {R : Reg} {ω : L.Ω} {t : List (Cm × I.Coin)} {R₂ : Reg₂} {j : Fin S}
    {port value : String} {rows : Array (Array ℕ)} {what : ℕ → String}
    (h : Flock.RecCoins.checkValue (Vs j).c (zs R ω t R₂ j).pub (zs R ω t R₂ j).own port value rows what = .ok ())
    {prog : String} (hprog : (Vs j).c.metaJ.getObjValAs? String "program_sha512" = .ok prog)
    {u : Fin (Vs j).nv} {q : Fin (Vs j).c.ports.size}
    (hsrc : vsrc R ω t R₂ ⟨j, u, q⟩ = some (rows, Flock.RecCoins.verifierSalt prog value))
    (hp : (zs R ω t R₂ j).rd.isRead (.inl (u, q)))
    (hone : ∀ reg ∈ (zs R ω t R₂ j).own, reg.name = port → reg = (zs R ω t R₂ j).rd.port (.inl (u, q)))
    (hpos : ((zs R ω t R₂ j).rd.port (.inl (u, q))).reads[u.val]? =
      some (some ((zs R ω t R₂ j).rd.readAt (.inl (u, q)))))
    (hseg : (Vs j).c.ports[q.val]!.segs = none) :
    u.val < rows.size ∧
      ((∀ (b : ℕ) (hb : b < FlockSoundness.Discharge.Layout.portRowW (Vs j).c q),
        (FlockVBridge.Decode.regVal (zs R ω t R₂ j) (.inl (u, q))).2
            ⟨b, lt_of_lt_of_le hb ((Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _))⟩ =
          (rows[u.val]!)[b / 16]!.testBit (b % 16)) ∨
      ValueColl zs vsrc R ω t R₂) := by
  obtain ⟨hu, hor⟩ := FlockC3.RegLevels.value_tie (zs R ω t R₂ j) h hprog hp hone hpos hseg
  exact ⟨hu, hor.imp id fun hc => valueColl_of_tie hsrc hp hc⟩

open Classical in
/-- **`valueFinder`'s output at an outcome**: `valueAt` at a read where `ValueColl`'s test passes, if there is one. -/
noncomputable def valuePick (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (vsrc : ValueSrc L Reg Cm Reg₂ I Vs) {β : Type} (o : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × β)) : Pair :=
  if h : ValueColl zs vsrc o.1 o.2.1 o.2.2.1 o.2.2.2.1 then valueAt zs vsrc o.1 o.2.1 o.2.2.1 o.2.2.2.1 h.choose
  else none

theorem ind_valueColl_le (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (vsrc : ValueSrc L Reg Cm Reg₂ I Vs) {β : Type} (o : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × β)) :
    ind (fun o : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × β) => ValueColl zs vsrc o.1 o.2.1 o.2.2.1 o.2.2.2.1) o ≤
      ind (IsColl Refine.H512) (valuePick zs vsrc o) := by
  by_cases h : ValueColl zs vsrc o.1 o.2.1 o.2.2.1 o.2.2.2.1
  · have hc : IsColl Refine.H512 (valuePick zs vsrc o) := by
      unfold valuePick
      rw [dif_pos h]
      exact h.choose_spec.2
    have h1 : ind (IsColl Refine.H512) (valuePick zs vsrc o) = 1 := if_pos hc
    rw [h1]
    unfold ind
    split_ifs <;> norm_num
  · have h0 : ind (fun o : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × β) =>
        ValueColl zs vsrc o.1 o.2.1 o.2.2.1 o.2.2.2.1) o = 0 := if_neg h
    rw [h0]
    exact ind_nonneg _ _

/-- **The value finder**: one run of the recursive audit against the prover `σ`, output `valueColl`'s pair at a
registered read of the outcome's history and registration where it collides (`ValueColl`), else nothing. -/
noncomputable def valueFinder (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (vsrc : ValueSrc L Reg Cm Reg₂ I Vs) (dg : ByteArray → ByteArray) (N : ℕ)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N)) : Finder Pair :=
  oneRun _ σ (valuePick zs vsrc)

/-- **A registrant's row is off a verifier's value at most `qV²/2^513` likely** under `cr/sha-512` for `valueFinder`
at the budget `qV`: once per audit, whatever the number of sessions and reads. -/
theorem prob_valueColl_le (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (vsrc : ValueSrc L Reg Cm Reg₂ I Vs) (dg : ByteArray → ByteArray) (N : ℕ)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    {qV : ℝ} (hV : (valueFinder zs vsrc dg N σ).CR Refine.H512 qV) :
    prob (fun o => ValueColl zs vsrc o.1 o.2.1 o.2.2.1 o.2.2.2.1)
      (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
      ENNReal.ofReal (qV ^ 2 / 2 ^ (513 : ℝ)) := by
  have hev : (prob (fun o => ValueColl zs vsrc o.1 o.2.1 o.2.2.1 o.2.2.2.1)
      (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ).toReal ≤ qV ^ 2 / 2 ^ (513 : ℝ) := by
    rw [prob_toReal]
    refine le_trans ?_ hV.le
    rw [valueFinder, oneRun_ev]
    exact expect_mono (ind_valueColl_le zs vsrc) _ _
  exact (ENNReal.le_ofReal_iff_toReal_le (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _))
    (by positivity)).2 hev

/-- **The value case in fork form**: the value finder finds a collision at an outcome `σ` reaches, or no outcome has a
registered read off a verifier's value. -/
theorem prob_valueColl_fork (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (vsrc : ValueSrc L Reg Cm Reg₂ I Vs) (dg : ByteArray → ByteArray) (N : ℕ)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N)) :
    (valueFinder zs vsrc dg N σ).Finds Refine.H512 ∨
      prob (fun o => ValueColl zs vsrc o.1 o.2.1 o.2.2.1 o.2.2.2.1)
        (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ = 0 := by
  by_cases h : (valueFinder zs vsrc dg N σ).Finds Refine.H512
  · exact Or.inl h
  refine Or.inr (le_antisymm ((prob_valueColl_le zs vsrc dg N σ (Finder.cr_of_not_finds h 0)).trans ?_) zero_le)
  simp

/-- **The registrant, held-top and value cases, union-bounded**: two sessions' records clash (`regFinder` at `qX`), the
held top differs from the registered strings (`topFinder` at `qT`), or a registrant's row is off a verifier's value
(`valueFinder` at `qV`). -/
theorem prob_regTopValue_le
    (wr : List (Cm × I.Coin) → Lean.Json × String) (st : Flock.Stmt) (reg : Lean.Json)
    (bcs' : Array ByteArray) (lv : FlockSoundness.Accounting.Level) (cap : FlockSoundness.Model.Cap Refine.Dig)
    (hstk : StackedTopAt lv st reg cap bcs')
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (vsrc : ValueSrc L Reg Cm Reg₂ I Vs) (dg : ByteArray → ByteArray) (N : ℕ)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    {qX qT qV : ℝ} (hX : (regFinder zs dg N σ).CR Refine.H512 qX)
    (hT : (topFinder wr st reg bcs' lv cap hstk zs σ).CR Refine.H512 qT)
    (hV : (valueFinder zs vsrc dg N σ).CR Refine.H512 qV) :
    prob (fun o => (RegColl zs o.1 o.2.1 o.2.2.1 o.2.2.2.1 ∨
        (AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ TopColl wr st reg bcs' o.2.2.1)) ∨
        ValueColl zs vsrc o.1 o.2.1 o.2.2.1 o.2.2.2.1)
      (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
      ENNReal.ofReal (qX ^ 2 / 2 ^ (513 : ℝ)) + ENNReal.ofReal (qT ^ 2 / 2 ^ (513 : ℝ)) +
        ENNReal.ofReal (qV ^ 2 / 2 ^ (513 : ℝ)) :=
  (prob_or_le _ _ _ _).trans (add_le_add
    (prob_regOrTop_le wr st reg bcs' lv cap hstk zs σ hX hT) (prob_valueColl_le zs vsrc dg N σ hV))

end FlockSoundness.Discharge.Recursion

#print axioms FlockSoundness.Discharge.Recursion.valueAt_of_some
#print axioms FlockSoundness.Discharge.Recursion.valueColl_of_tie
#print axioms FlockSoundness.Discharge.Recursion.value_tie_coll
#print axioms FlockSoundness.Discharge.Recursion.ind_valueColl_le
#print axioms FlockSoundness.Discharge.Recursion.prob_valueColl_le
#print axioms FlockSoundness.Discharge.Recursion.prob_valueColl_fork
#print axioms FlockSoundness.Discharge.Recursion.prob_regTopValue_le
