module

public import Properties.Nci.Record
import Mathlib.Algebra.Order.Field.Basic
import Std.Data.TreeMap.Raw.Lemmas
import all Lean.Data.Json.Basic

/-!
# The circuit record's properties, from their parts

`complies_of`: the rate rule on a record's circuit (`Complies`) from four statements about the check, that a record it
reads carries its instance's fields (`Owns`) and lays out a circuit (`LaidOut`), that its summary counts every block
(`Counts`), and that its rate rule is the policy's on a summary (`Rates`).

`pinnedOpens_of`: what a pinned reader opens (`PinnedOpens`) from the closure digest's pin (`DigestPins`), the
check's provenance (`Provenance`) and a layout that reads a Definition's signature alone (`LayoutReads`).

`pin_listed`: the pin an opened row is held under is one the instance lists, from `Owns` and `Provenance`.

`ofJson_wiresOf`: a text `Spot.Wires.Wiring.ofJson` accepts whose `cuts` and `wires` are the record's has the cuts
and the wires that reader gives of the record's own piece (`wiresOf`), which it accepts too. The reader reads each of
its keys on its own, the piece's empty `templates` has no template to refuse, and it states no `order`.
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Lean (Json)
open Flock.Program (Prog)
open NciCheck.DefinitionDigest (closureDigest)
open NciCheck.RecordText (Row wiresOf valueAt)

theorem complies_of {F : Form} (hown : Owns F) (hlaid : LaidOut F) (hcount : Counts F) (hrate : Rates F) :
    Complies F := by
  intro j tr c hacc
  obtain ⟨hread, hnone⟩ := hacc
  obtain ⟨hΓ, hα⟩ := hrate _ _ _ _ hnone
  have hcar := hown j tr c hread
  have hden : (0 : ℝ) < (F.inst c).den := by
    obtain ⟨-, -, -, -, h, -⟩ := hcar
    exact_mod_cast h
  refine ⟨hcar, hlaid j tr c hread, fun d hd => ?_⟩
  have hmeets : ∀ u, Meets d d.𝓘 (F.inst c).gamma (((F.inst c).num : ℝ) / (F.inst c).den) u := by
    intro u
    obtain ⟨hfresh, hsize | hempty⟩ := hcount j tr c hread d hd u
    · unfold Meets
      rw [hsize]
      refine ⟨hΓ, ?_⟩
      have h1 : (Nci.novelReads d (d.𝓘.gates u) : ℝ) ≤ (F.summarize c).fresh := by exact_mod_cast hfresh
      have h2 : ((F.summarize c).fresh : ℝ) * (F.inst c).den ≤ (F.inst c).num * (F.summarize c).gates := by
        exact_mod_cast hα
      rw [div_mul_eq_mul_div, le_div_iff₀ hden]
      exact (mul_le_mul_of_nonneg_right h1 hden.le).trans h2
    · unfold Meets
      rw [hempty]
      simp [Nci.novelReads, Verity.Circuit.size, Verity.Circuit.into, Verity.Circuit.width]
  exact ⟨hmeets, by unfold Nci.CompliesRate; exact ⟨d.n, d.𝓘, hmeets⟩⟩

/-- One Definition has one signature. -/
theorem Same.signature {P P' : Prog} {k k' : ℕ} (h : Same P P' k k') :
    ∃ d d', P.defs[k]? = some d ∧ P'.defs[k']? = some d' ∧ d.params = d'.params ∧ d.ret = d'.ret :=
  match h with
  | .intro _ _ d d' hd hd' _ _ _ hparams hret .. => ⟨d, d', hd, hd', hparams, hret⟩

theorem pinnedOpens_of {F : Form} (hpins : DigestPins) (hprov : Provenance F) (hlay : LayoutReads F) :
    PinnedOpens F := by
  intro Opens text₀ j₀ P₀ k₀ d₀ key pin hof hdec₀ hparse₀ hdig₀ hk₀ hd₀ hopens j tr c P pj r name p hread hdec
    hparse htid hheld
  obtain ⟨log, run, -, -, hgood⟩ := hprov j tr c P pj hread hdec hparse
  obtain ⟨-, -, hdig, -⟩ :
      IsTheta log run p ∧ pin ∈ (F.inst c).pins ∧ closureDigest pj r.tid = .ok pin ∧ _ := hgood r _ hheld
  rw [htid] at hdig
  by_cases hex : ∃ k, P.index[key]? = some k
  · obtain ⟨k, hk⟩ := hex
    rcases hpins text₀ (F.inst c).program j₀ pj P₀ P key pin k₀ k hdec₀ hdec hparse₀ hparse hdig₀ hdig hk₀ hk
      with hsame | hcol
    · refine Or.inl fun k' d hk' hd => ?_
      have hkk : k' = k := Option.some.inj (hk'.symm.trans hk)
      obtain ⟨e₀, e, he₀, he, hparams, hret⟩ := Same.signature hsame
      have h1 : d₀ = e₀ := Option.some.inj (hd₀.symm.trans he₀)
      have h2 : d = e := by rw [hkk] at hd; exact Option.some.inj (hd.symm.trans he)
      rw [hkk, h1, h2]
      exact ⟨hof P₀ P k₀ k hsame hopens, hlay e e₀ hparams.symm hret.symm⟩
    · exact Or.inr hcol
  · exact Or.inl fun k d hk' _ => absurd ⟨k, hk'⟩ hex

/-- The pin an opened row is held under is one the instance lists. -/
theorem pin_listed {F : Form} (hown : Owns F) (hprov : Provenance F) {j tr : Json} {c : F.Case} {P : Prog}
    {pj : Json} {r : Row} {name p pin : String} (hread : F.read j tr = .ok c)
    (hdec : Flock.Program.decode (F.inst c).program.toUTF8 = .ok P)
    (hparse : Flock.JsonParse.parse (F.inst c).program = .ok pj) (hheld : F.held c r = some (.opened name p pin)) :
    ∃ ps : Array Json, j.getObjVal? "pins" = .ok (.arr ps) ∧ Json.str pin ∈ ps := by
  obtain ⟨log, run, -, -, hgood⟩ := hprov j tr c P pj hread hdec hparse
  obtain ⟨-, hmem, -⟩ :
      IsTheta log run p ∧ pin ∈ (F.inst c).pins ∧ closureDigest pj r.tid = .ok pin ∧ _ := hgood r _ hheld
  obtain ⟨-, -, -, -, -, hpins⟩ := hown j tr c hread
  exact ⟨_, hpins, List.mem_toArray.mpr (List.mem_map.mpr ⟨pin, hmem, rfl⟩)⟩

/-! ## The record's piece of a text SPOT's reader accepts -/

private theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : x >>= f = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error => cases h
  | ok a => exact ⟨a, rfl, h⟩

private theorem ok_bind {ε α β : Type} (a : α) (f : α → Except ε β) : (Except.ok a >>= f) = f a := rfl

private theorem getObjVal?_mkObj {l : List (String × Json)} (hd : l.Pairwise (fun a b => ¬ compare a.1 b.1 = .eq))
    {k : String} {v : Json} (hm : (k, v) ∈ l) : (Json.mkObj l).getObjVal? k = .ok v := by
  simp only [Json.mkObj, Json.getObjVal?, Std.TreeMap.Raw.get?_eq_getElem?]
  rw [Std.TreeMap.Raw.getElem?_ofList_of_mem (Std.ReflCmp.compare_self) hd hm]
  rfl

private theorem getObjVal?_mkObj_not {l : List (String × Json)} {k : String}
    (hk : (l.map Prod.fst).contains k = false) : ∃ e, (Json.mkObj l).getObjVal? k = .error e := by
  simp only [Json.mkObj, Json.getObjVal?, Std.TreeMap.Raw.get?_eq_getElem?]
  rw [Std.TreeMap.Raw.getElem?_ofList_of_contains_eq_false hk]
  exact ⟨_, rfl⟩

private theorem field_ok {j v : Json} {k : String} (h : j.getObjVal? k = .ok v) : Spot.Wires.field j k = .ok v := by
  simp only [Spot.Wires.field, h]

/-- An object of no field has no field. -/
private theorem fields_nil : Spot.Wires.fields (Json.mkObj []) = .ok [] := by
  have h : (∅ : Std.TreeMap.Raw String Json).toList = [] :=
    List.isEmpty_iff.1 (Std.TreeMap.Raw.isEmpty_toList.trans Std.TreeMap.Raw.isEmpty_emptyc)
  simp only [Json.mkObj, Spot.Wires.fields, Std.TreeMap.Raw.ofList_nil, Array.toList_map,
    Std.TreeMap.Raw.toList_toArray, h, List.map_nil]

private theorem wiresOf_templates (doc : Json) : (wiresOf doc).getObjVal? "templates" = .ok (Json.mkObj []) :=
  getObjVal?_mkObj (by simp) (by simp)

private theorem wiresOf_cuts (doc : Json) :
    (wiresOf doc).getObjVal? "cuts" = .ok ((doc.getObjVal? "cuts").toOption.getD (Json.mkObj [])) :=
  getObjVal?_mkObj (by simp) (by simp)

private theorem wiresOf_wires (doc : Json) : (wiresOf doc).getObjVal? "wires" = .ok (valueAt doc "wires") :=
  getObjVal?_mkObj (by simp) (by simp)

private theorem wiresOf_order (doc : Json) : ∃ e, (wiresOf doc).getObjVal? "order" = .error e :=
  getObjVal?_mkObj_not (by simp)

/-- **A text with the record's cuts and wires is read as the record's piece is.** Where `Spot.Wires.Wiring.ofJson`
accepts the text `j`, and `j`'s `cuts` and `wires` are the ones `wiresOf` gives of the record `doc`, it accepts
`wiresOf doc` too, with the same cuts and the same wires, whatever templates and order `j` states beside them: what
`Served` holds a wiring to, for a check whose wiring is the reader's of such a text. -/
theorem ofJson_wiresOf {j doc : Json} {W : Spot.Wires.Wiring} (h : Spot.Wires.Wiring.ofJson j = .ok W)
    (hc : j.getObjVal? "cuts" = .ok ((doc.getObjVal? "cuts").toOption.getD (Json.mkObj [])))
    (hw : j.getObjVal? "wires" = .ok (valueAt doc "wires")) :
    ∃ W', Spot.Wires.Wiring.ofJson (wiresOf doc) = .ok W' ∧ W'.cuts = W.cuts ∧ W'.wires = W.wires := by
  obtain ⟨e, ho⟩ := wiresOf_order doc
  unfold Spot.Wires.Wiring.ofJson at h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨c, h4, h⟩ := bind_ok h
  obtain rfl := Except.ok.inj ((field_ok hc).symm.trans h4)
  obtain ⟨_, h5, h⟩ := bind_ok h
  obtain ⟨cs, h6, h⟩ := bind_ok h
  obtain ⟨w, h7, h⟩ := bind_ok h
  obtain rfl := Except.ok.inj ((field_ok hw).symm.trans h7)
  obtain ⟨_, h8, h⟩ := bind_ok h
  obtain ⟨ws, h9, h⟩ := bind_ok h
  refine ⟨{ templates := [], cuts := cs, wires := ws, order := none }, ?_, ?_⟩
  · unfold Spot.Wires.Wiring.ofJson
    simp only [field_ok (wiresOf_templates doc), field_ok (wiresOf_cuts doc), field_ok (wiresOf_wires doc), ho,
      fields_nil, h5, h6, h8, h9, List.mapM_nil, ok_bind, pure_bind]
    rfl
  · cases hj : j.getObjVal? "order"
    all_goals
      simp only [hj] at h
      repeat (obtain ⟨_, -, h⟩ := bind_ok h)
      obtain rfl := Except.ok.inj h
      exact ⟨rfl, rfl⟩

end NciCheck.Properties.Record.SecurityProofs
