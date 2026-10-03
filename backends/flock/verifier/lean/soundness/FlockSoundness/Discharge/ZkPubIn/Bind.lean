import FlockSoundness.Discharge.ZkPubIn.KeyP
import FlockSoundness.Discharge.ZkHidden.SoundBind

set_option Elab.async false

/-!
# The sites, the layout and the rows at the `--zk` tables with the verifier's public inputs

ZkHidden's `tableShH`, `sitesShH`, `layShH`, `rsShH`, `placeZH`, `layZH`, `rsZH` (`ZkHidden/SoundBind.lean`) at
`tablesAtZKHP` and `keyProgP`:

* `tableShHP`: the public table (`TableHP`) at the same draw; its statement is the `--zk` table's (`tableAtZKHP_S`).
* `sitesShHP`, `layShHP`, `rsShHP`: `keyLayoutHP` and `keyRowsHP` at `tableShHP`'s sites, the registration the
  population's public file (`HmIn.loadPublic`), `hsite` from `sitesShHP_bcSite`.
* `placeZHP`, `layZHP`, `rsZHP`, `rsZHP_computes`: moved to the `--zk` tables (`LayoutZ.ofShadow`,
  `HmRowsZ.ofShadow`).

`pub.tables = none` is not needed (as ZkReg's `rsZHS`): `BcSiteHP` reads the public file with or without shared rows.
-/

namespace FlockSoundness.Discharge.ZkPubIn

open Flock FlockLevel3 Game Model Accounting Refine Audit ZkSession ZkLink ZkExec ZkBind HiddenExec Placed ZkHidden
open FlockSoundness.Discharge.Exec (Inputs)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {n : ℕ} {I : Inputs} {own : ByteArray} {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  (y₀ : DrawSetupZKHP I own mPts S₀ (dj S₀)) {Reg : Type}

open Classical in
/-- **The public table at the `--zk` table's draw**: the draw's `DrawSetupZKHP`'s table, or `y₀`'s refused one. -/
noncomputable def tableShHP (S : Finset (Fin n)) : TableHP (zkInputs I) own mPts S :=
  if h : Nonempty (DrawSetupZKHP I own mPts S (dj S)) then (Classical.choice h).x.table else y₀.x.refused S

/-- **The `--zk` table's statement is the public table's**, at every draw. -/
theorem tableAtZKHP_S (S : Finset (Fin n)) : (tableAtZKHP dj y₀ S).S = (tableShHP dj y₀ S).T.S := by
  unfold tableAtZKHP tableShHP
  split_ifs <;> rfl

/-- The public plan at the same draws. -/
noncomputable abbrev planShHP : Finset (Fin n) → Reg → List (TabSpec mPts) := fun S _ => [(tableShHP dj y₀ S).T]

abbrev tabShHP : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planShHP (Reg := Reg) dj y₀ S R).length :=
  fun _ _ _ => ⟨0, Nat.zero_lt_one⟩

theorem planAtZKHP_S : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n),
    (tabOfZ (planAtZKHP dj y₀) (tabAtZKHP dj y₀) S R u).S =
      (Partition.tabOf (planShHP dj y₀) (tabShHP dj y₀) S R u).S :=
  fun S _ _ => tableAtZKHP_S dj y₀ S

variable (ht : I.tags.typed = false) {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
  (hscope : Layout.scopeOk c = true)

/-- **The public plan's sites** (`sitesShH`'s, at `tableShHP`). -/
noncomputable def sitesShHP : SitesHP (planShHP (Reg := Reg) dj y₀) (tabShHP dj y₀) where
  acc S _ _ _ := (tableShHP dj y₀ S).acc (zkTyped ht) hc (v2_of_scope hscope)
  vu S _ u h := (tableShHP dj y₀ S).vu u h
  hvu S _ u h := (tableShHP dj y₀ S).hvu u h
  o S _ u h := (tableShHP dj y₀ S).o u h

/-- **One circuit per session**: every site's acceptance parses the session's one circuit file to `c`. -/
theorem sitesShHP_parse :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      HmRow.parse ((sitesShHP dj y₀ ht hc hscope).acc S R u h).tags ((sitesShHP dj y₀ ht hc hscope).acc S R u h).cf
        ((sitesShHP dj y₀ ht hc hscope).acc S R u h).tables = .ok c :=
  fun _ _ _ _ => (parse_untyped (zkTyped ht) hc).1

variable (hU : Built c.unit) {pi : HmIn.Ports} (hpi : HmIn.parse c = .ok pi) {pub : Public}
  (hpub : HmIn.loadPublic (zkTags I) c pi I.publicFile = .ok pub)

include hU hpi hpub in
/-- **`BcSiteHP` at every satisfiable site** of the public plan (`sitesShH_bcSite`): at a draw `verify --zk` sets up,
the unit's rank in the draw's table; at a refused draw no message satisfies the table. -/
theorem sitesShHP_bcSite :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planShHP dj y₀) (tabShHP dj y₀) S R u), Binding.Sat (planShHP dj y₀) (tabShHP dj y₀) S R u M →
        BcSiteHP ((sitesShHP dj y₀ ht hc hscope).acc S R u h) pi pub ((sitesShHP dj y₀ ht hc hscope).o S R u h)
          ((sitesShHP dj y₀ ht hc hscope).vu S R u h) u.val := by
  sorry

/-- **`lay` at the public plan**: `keyLayoutHP` at `sitesShHP`, the program's outputs the returned outputs' bits, the
registration the population's public file. -/
noncomputable abbrev layShHP :
    Binding.Layout ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planShHP (Reg := Reg) dj y₀) (tabShHP dj y₀)
      (((keyProgP c hU n pi).partition (Layout.outsOf c hU n)).placeDecoder (planShHP dj y₀) (tabShHP dj y₀)
        (keyPlaceHP (sitesShHP_parse dj y₀ ht hc hscope) pi (Layout.outsOf c hU n))) :=
  keyLayoutHP (sitesShHP_parse dj y₀ ht hc hscope) pi own (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOk_spec hU hscope).2.1 (Layout.scopeOk_ports hscope) (Layout.outsOut_outsOf c hU n)

/-- **`rs` at the public plan**: `keyRowsHP` at `sitesShHP`, with `hsite` from `sitesShHP_bcSite`. -/
noncomputable abbrev rsShHP : Binding.HmRows (layShHP dj y₀ ht hc hscope hU (own := own) (pi := pi) (pub := pub)
    (Reg := Reg)) :=
  keyRowsHP (sitesShHP_parse dj y₀ ht hc hscope) pi own (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOk_spec hU hscope).2.1 (Layout.scopeOk_ports hscope) (Layout.outsOut_outsOf c hU n)
    (sitesShHP_bcSite dj y₀ ht hc hscope hU hpi hpub)
    (Layout.keysRow_of_parse (parse_untyped (zkTyped ht) hc).1 (Layout.scopeOk_spec hU hscope).1
      (Layout.scopeOk_spec hU hscope).2.2)

theorem rsShHP_computes : (rsShHP dj y₀ ht hc hscope hU hpi hpub (Reg := Reg)).Computes :=
  keyRowsHP_computes (sitesShHP_parse dj y₀ ht hc hscope) pi own (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOk_spec hU hscope).2.1 (Layout.scopeOk_ports hscope) (Layout.outsOut_outsOf c hU n)
    (sitesShHP_bcSite dj y₀ ht hc hscope hU hpi hpub)
    (Layout.keysRow_of_parse (parse_untyped (zkTyped ht) hc).1 (Layout.scopeOk_spec hU hscope).1
      (Layout.scopeOk_spec hU hscope).2.2)

/-- **Each drawn unit's place in its `--zk` table**: `keyPlaceHP`'s place in the public table, along
`planAtZKHP_S`. -/
noncomputable def placeZHP (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    UnitPlace ((keyProgP c hU n pi).partition (Layout.outsOf c hU n))
      (tabOfZ (planAtZKHP dj y₀) (tabAtZKHP dj y₀) S R u).S u :=
  placeCast (planAtZKHP_S dj y₀ S R u)
    (keyPlaceHP (sitesShHP_parse dj y₀ ht hc hscope) pi (Layout.outsOf c hU n) S R u h)

/-- **The decoders agree**: the `--zk` places and the public places read one witness alike. -/
theorem decAgreeHP :
    DecAgree (P := (keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planAtZKHP_S (Reg := Reg) dj y₀)
      (decZ := decodeOfPlace execArith ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planAtZKHP dj y₀)
        (tabAtZKHP dj y₀) (placeZHP dj y₀ ht hc hscope hU))
      (decS := ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)).placeDecoder (planShHP dj y₀) (tabShHP dj y₀)
        (keyPlaceHP (sitesShHP_parse dj y₀ ht hc hscope) pi (Layout.outsOf c hU n))) := by
  intro S R u hu M' M hM _ g _
  simp only [decodeOfPlace, Partition.placeDecoder, hu, ↓reduceDIte, placeZHP, decode_placeCast]
  exact congrArg (fun z => UnitPlace.decode _ z g) hM.symm

/-- **`lay` at the `--zk` tables with the verifier's public inputs**: `layShHP` moved along `planAtZKHP_S`. -/
noncomputable abbrev layZHP :
    LayoutZ execArith ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planAtZKHP (Reg := Reg) dj y₀)
      (tabAtZKHP dj y₀)
      (decodeOfPlace execArith ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planAtZKHP dj y₀)
        (tabAtZKHP dj y₀) (placeZHP dj y₀ ht hc hscope hU)) :=
  LayoutZ.ofShadow (planAtZKHP_S dj y₀) (layShHP dj y₀ ht hc hscope hU (own := own) (pub := pub))
    (decAgreeHP dj y₀ ht hc hscope hU)

/-- **`rs` at the `--zk` tables with the verifier's public inputs**: `rsShHP` moved along `planAtZKHP_S`. -/
noncomputable abbrev rsZHP : HmRowsZ (layZHP dj y₀ ht hc hscope hU (pi := pi) (pub := pub) (Reg := Reg)) :=
  HmRowsZ.ofShadow (planAtZKHP_S dj y₀) (decAgreeHP dj y₀ ht hc hscope hU) (rsShHP dj y₀ ht hc hscope hU hpi hpub)

/-- **`hHm` at the `--zk` tables' rows with the verifier's public inputs** (`rsShHP_computes`, moved along). -/
theorem rsZHP_computes : (rsZHP dj y₀ ht hc hscope hU hpi hpub (Reg := Reg)).Computes :=
  HmRowsZ.ofShadow_computes (planAtZKHP_S dj y₀) (decAgreeHP dj y₀ ht hc hscope hU) _
    (rsShHP_computes dj y₀ ht hc hscope hU hpi hpub)

end FlockSoundness.Discharge.ZkPubIn
