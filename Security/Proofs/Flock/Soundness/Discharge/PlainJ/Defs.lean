import Proofs.Flock.Soundness.Discharge.Composed.DefsJ
import Proofs.Flock.Soundness.Discharge.Integrate.Headline

/-!
# One plain sampled session of `J` tables: the plan, sites, layout and rows

`flock-verify verify --plain-tables --session-tables J` (no `--zk`) accepts a public-output session of up to
`Tags.MAX_TABLES` tables at a subset draw (`ProvedScope.checkP`, `ProvedScope.drawLaw` at `plainTables`). `Stmt.setupTables` sets table
`j` up under `tagsAt` at its draw `Composed.drawJ` (the session's draw at one table, its part `j` at several). These are
`Composed.DefsJ`'s public shadow pieces at the plain inputs themselves, so the layout and rows are e2e-layout's directly,
with no transfer to compiled tables:

* `inputsPJ`, `tablePJ`, `planPJ`, `tabPJ`: table `j`'s inputs (`tagsAt`), its table (the `DrawSetup` table where
  flock-verify sets one up at table `j`'s inputs and draw, else a reference setup `x₀`'s refused table), the plan of
  `nTab I` tables, and each drawn unit's table (`partOf`).
* `sitesPJ`, `sitesPJ_parse`, `sitesPJ_bcSite`: each drawn unit's acceptance, VU and block, the one circuit they parse,
  and e2e-layout's site fact.
* `layPJ`, `rsPJ`, `rsPJ_computes`: `keyedLayout`, `keyedRowsOuts` and `hHm` at those sites.
* `DrawOkPJ`, `opnPJ`: the coin server's draw at each table, and table `j`'s openings.
-/

namespace FlockSoundness.Discharge.PlainJ

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Integrate
open FlockSoundness.Discharge.Composed (nTab nTab_pos drawJ partOf partS mem_partS parse_tagsAt loadPublic_tagsAt)
open FlockSoundness.Discharge.ZkBind (recastAcc)

local notation "𝔽" => FlockLevel3.GF128

variable {n mPts : ℕ} {Reg : Type}

/-- Table `j`'s inputs: the session's, under table `j`'s tags (`tagsAt`: the statement's at one table, under
`Tags.tableIdentity count j` at several). -/
def inputsPJ (I : Inputs) (j : ℕ) : Inputs := { I with tags := tagsAt I.tags I.count j }

theorem typedPJ {I : Inputs} (ht : I.tags.typed = false) (j : ℕ) : (inputsPJ I j).tags.typed = false :=
  (Exec.tagsAt_typed _ _ _).trans ht

section Plan

variable {I : Inputs} (ht : I.tags.typed = false) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  {d₀ : Lean.Json} (x₀ : DrawSetup (inputsPJ I 0) mPts S₀ d₀)

open Classical in
/-- **Table `j` at draw `S`**: the table flock-verify sets up at table `j`'s inputs and draw when it sets one up, and
otherwise the reference setup `x₀`'s refused table (a setup at table 0's inputs, which every session flock-verify
accepts at a draw `DrawOkPJ` admits has: `setupPJ_of_verifiesP`). -/
noncomputable def tablePJ (S : Finset (Fin n)) (j : Fin (nTab I)) : Integrate.Table mPts (partS I (dj S) S j) :=
  if h : Nonempty (DrawSetup (inputsPJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)) then
    (Classical.choice h).table (typedPJ ht j)
  else x₀.refused (typedPJ ht 0) _

/-- **The plan of `J` tables**: at draw `S`, table `j` is `tablePJ … S j`, for every registration. -/
noncomputable def planPJ : Finset (Fin n) → Reg → List (TabSpec mPts) :=
  fun S _ => List.ofFn fun j => (tablePJ ht dj x₀ S j).T

theorem planPJ_length (S : Finset (Fin n)) (R : Reg) : (planPJ (Reg := Reg) ht dj x₀ S R).length = nTab I :=
  List.length_ofFn

/-- Each drawn unit is in the table it is drawn into (`partOf`). -/
noncomputable def tabPJ : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planPJ (Reg := Reg) ht dj x₀ S R).length :=
  fun S _ u => Fin.cast List.length_ofFn.symm (partOf I (dj S) u.val)

theorem tabOfPJ_eq (S : Finset (Fin n)) (R : Reg) (u : Fin n) :
    Partition.tabOf (planPJ ht dj x₀) (tabPJ ht dj x₀) S R u = (tablePJ ht dj x₀ S (partOf I (dj S) u.val)).T := by
  simp only [Partition.tabOf, planPJ, tabPJ, List.get_ofFn]
  rfl

/-- **Table `j` where flock-verify sets it up**: at a `DrawSetup` of table `j`'s inputs and draw, `tablePJ` is the table of
one with the same statement. -/
theorem tablePJ_of {S : Finset (Fin n)} {j : Fin (nTab I)}
    (x : DrawSetup (inputsPJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)) :
    ∃ y : DrawSetup (inputsPJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j), y.st = x.st ∧
      tablePJ ht dj x₀ S j = y.table (typedPJ ht j) := by
  refine ⟨Classical.choice ⟨x⟩, Except.ok.inj ((Classical.choice ⟨x⟩).h.symm.trans x.h), ?_⟩
  unfold tablePJ
  rw [dite_eq_left_of_eq_true (eq_true ⟨x⟩)]

/-- **The plan's sites**: each drawn unit's acceptance, VU and block are its table's. -/
noncomputable def sitesPJ : Placed.Sites (planPJ (Reg := Reg) ht dj x₀) (tabPJ ht dj x₀) where
  acc S R u _ := recastAcc (tablePJ ht dj x₀ S (partOf I (dj S) u.val)).acc
    (congrArg TabSpec.S (tabOfPJ_eq ht dj x₀ S R u)).symm
  vu S _ u h := (tablePJ ht dj x₀ S (partOf I (dj S) u.val)).vu u (mem_partS h)
  hvu S _ u h := (tablePJ ht dj x₀ S (partOf I (dj S) u.val)).hvu u (mem_partS h)
  o S R u h := Fin.cast (congrArg (fun T : TabSpec mPts => 2 ^ T.S.nbl) (tabOfPJ_eq ht dj x₀ S R u)).symm
    ((tablePJ ht dj x₀ S (partOf I (dj S) u.val)).o u (mem_partS h))

variable {c : Flock.Circuit} (hc : HmRow.parse I.tags I.circuitFile I.tables = .ok c)

include hc in
/-- **One circuit per session, at every table**: every site's acceptance parses the session's circuit file to `c` (table
`j`'s tags differ from the session's in the identity alone, which the parse does not read). -/
theorem sitesPJ_parse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse ((sitesPJ ht dj x₀).acc S R u h).tags ((sitesPJ ht dj x₀).acc S R u h).cf
      ((sitesPJ ht dj x₀).acc S R u h).tables = .ok c := by
  intro S R u h
  show HmRow.parse (tablePJ ht dj x₀ S (partOf I (dj S) u.val)).acc.tags
    (tablePJ ht dj x₀ S (partOf I (dj S) u.val)).acc.cf (tablePJ ht dj x₀ S (partOf I (dj S) u.val)).acc.tables = .ok c
  unfold tablePJ
  by_cases hS : Nonempty (DrawSetup (inputsPJ I (partOf I (dj S) u.val)) mPts
    (partS I (dj S) S (partOf I (dj S) u.val)) (drawJ I (dj S) (partOf I (dj S) u.val)))
  · rw [dite_eq_left_of_eq_true (eq_true hS)]; exact (parse_tagsAt _ _ _ _ _).trans hc
  · rw [dite_eq_right_of_eq_false (eq_false hS)]; exact (parse_tagsAt _ _ _ _ _).trans hc

variable {pub : Public} (hpub : HmRow.loadPublic I.tags c I.publicFile = .ok pub) (hpt : pub.tables = none)

include hc hpub hpt in
/-- **`BcSite` at every satisfiable site** at `J` tables (`sitesAt_bcSite`): at a table flock-verify sets up by
`table_bcSite` at table `j`'s inputs; at a refused one no message satisfies the table. -/
theorem sitesPJ_bcSite (ho : c.outNet = c.unitNet) :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planPJ ht dj x₀) (tabPJ ht dj x₀) S R u),
      Binding.Sat (planPJ ht dj x₀) (tabPJ ht dj x₀) S R u M →
        Layout.BcSite ((sitesPJ ht dj x₀).acc S R u h) pub ((sitesPJ ht dj x₀).o S R u h)
          ((sitesPJ ht dj x₀).vu S R u h) u.val := by
  intro S R u h M hM
  refine Composed.bcSite_recast _ (congrArg TabSpec.S (tabOfPJ_eq ht dj x₀ S R u)).symm _ _
    (congrArg (fun T : TabSpec mPts => 2 ^ T.S.nbl) (tabOfPJ_eq ht dj x₀ S R u)).symm _ _ ?_
  show Layout.BcSite (tablePJ ht dj x₀ S (partOf I (dj S) u.val)).acc pub
    ((tablePJ ht dj x₀ S (partOf I (dj S) u.val)).o u (mem_partS h))
    ((tablePJ ht dj x₀ S (partOf I (dj S) u.val)).vu u (mem_partS h)) u.val
  by_cases hS : Nonempty (DrawSetup (inputsPJ I (partOf I (dj S) u.val)) mPts
    (partS I (dj S) S (partOf I (dj S) u.val)) (drawJ I (dj S) (partOf I (dj S) u.val)))
  · have e : tablePJ ht dj x₀ S (partOf I (dj S) u.val) = (Classical.choice hS).table (typedPJ ht _) := by
      unfold tablePJ; rw [dite_eq_left_of_eq_true (eq_true hS)]
    rw [e]
    exact table_bcSite (typedPJ ht _) (fun _ => drawJ I (dj S) (partOf I (dj S) u.val))
      ((parse_tagsAt _ _ _ _ _).trans hc) ho ((loadPublic_tagsAt _ _ _ _ _).trans hpub) hpt
      (Classical.choice hS) (mem_partS h)
  · have e : tablePJ ht dj x₀ S (partOf I (dj S) u.val) = x₀.refused (typedPJ ht 0) _ := by
      unfold tablePJ; rw [dite_eq_right_of_eq_false (eq_false hS)]
    have hns : ∀ z, ¬ (Partition.tabOf (planPJ (Reg := Reg) ht dj x₀) (tabPJ ht dj x₀) S R u).S.Satisfies z := by
      rw [tabOfPJ_eq, e]; exact x₀.refused_unsat (typedPJ ht 0) _
    exact absurd hM (hns _)

variable (hU : Built c.unit) (hscope : Layout.scopeOk c = true)

/-- **`lay` at `J` tables**: e2e-layout's `keyedLayout` at `sitesPJ`, the program's outputs the returned outputs' bits,
the rows' salts at the `hm96` slots' salt columns, the registration's bits the population's returned outputs. -/
noncomputable abbrev layPJ :
    Binding.Layout ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) (planPJ (Reg := Reg) ht dj x₀)
      (tabPJ ht dj x₀)
      (((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)).placeDecoder (planPJ ht dj x₀) (tabPJ ht dj x₀)
        (Layout.keyPlace (sitesPJ_parse ht dj x₀ hc) (Layout.outsOf c hU n))) :=
  Layout.keyedLayout (sitesPJ_parse ht dj x₀ hc) (Layout.outsOf c hU n) Hm.saltCol fun _ => pub

/-- **`rs` at `J` tables**: e2e-layout's `keyedRowsOuts` at `sitesPJ`, its circuit conditions from `scopeOk`. -/
noncomputable abbrev rsPJ : Binding.HmRows (layPJ ht dj x₀ hc hU (pub := pub) (Reg := Reg)) :=
  Layout.keyedRowsOuts (sitesPJ_parse ht dj x₀ hc) Hm.saltCol hc (Layout.scopeOk_spec hU hscope).1
    (Layout.scopeOk_spec hU hscope).2.2 (Layout.scopeOk_spec hU hscope).2.1
    (sitesPJ_bcSite ht dj x₀ hc hpub hpt (Layout.scopeOk_spec hU hscope).2.1)

/-- **`hHm` at `J` tables' rows** (`rsAt_computes`). -/
theorem rsPJ_computes : (rsPJ ht dj x₀ hc hpub hpt hU hscope (Reg := Reg)).Computes :=
  Hm.keyedRowsOuts_computes (sitesPJ_parse ht dj x₀ hc) hc (Layout.scopeOk_spec hU hscope).1
    (Layout.scopeOk_spec hU hscope).2.2 (Layout.scopeOk_spec hU hscope).2.1 (Layout.scopeOk_ports hscope)
    (sitesPJ_bcSite ht dj x₀ hc hpub hpt (Layout.scopeOk_spec hU hscope).2.1)

end Plan

/-! ## The coin server's draws, and table `j`'s openings -/

/-- **The coin server's draw at `S`, at `J` tables** (`DrawOk` at table `j`'s inputs and draw): each table's draw proves
the drawn units it is given (`partS`), and setupH sets each table up at it with `mPts` link points. -/
def DrawOkPJ (I : Inputs) (dj : Finset (Fin n) → Lean.Json) (mPts : ℕ) (S : Finset (Fin n)) : Prop :=
  ∀ j : Fin (nTab I),
    (∀ D, Draw.UnitDraw.ofJson (drawJ I (dj S) j) = .ok D → ∀ u ∈ partS I (dj S) S j, u.val ∈ D.proved) ∧
    ∀ st, Stmt.setupH (tagsAt I.tags I.count j) I.circuitFile I.publicFile I.coins I.tables
      (some (drawJ I (dj S) j)) I.partition I.program = .ok st → st.mPts = mPts

/-- **Table `j`'s openings** (`DecodeZLJ.opnJ` at `drawJ`): `opnOf` at the statement setupH sets up under table `j`'s tags
at its draw, and its `fast100` schedule, on proof files `2j` and `2j + 1`. -/
noncomputable def opnPJ (I : Inputs) (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (dj : Finset (Fin n) → Lean.Json) (S : Finset (Fin n)) (R : Reg) (proofs : Array ByteArray)
    (j : Fin (plan S R).length) : Opens 𝔽 Dig × Salts (List UInt8) :=
  match Stmt.setupH (tagsAt I.tags I.count j.val) I.circuitFile I.publicFile I.coins I.tables
      (some (drawJ I (dj S) j.val)) I.partition I.program with
  | .ok st => match Flock.fast100 st.m with
    | some sch => opnOf (Setup.ofCircuit st) sch (DecodeZLJ.nXT ((plan S R).get j))
        #[proofs[2 * j.val]!, proofs[2 * j.val + 1]!]
    | none => default
  | .error _ => default

end FlockSoundness.Discharge.PlainJ
