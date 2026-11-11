module

public import Properties.Nci.Record
import Mathlib.Algebra.Order.Field.Basic

/-!
# The circuit record's properties, from their parts

`complies_of`: the rate rule on a record's circuit (`Complies`) from four statements about the check, that a record it
reads carries its instance's fields (`Owns`) and lays out a circuit (`LaidOut`), that its summary counts every block
(`Counts`), and that its rate rule is the policy's on a summary (`Rates`). `NciCheck.SecurityProofs.ServedComplies` is
that theorem as the guarantee `NciCheck.Guarantees.ServedComplies` states it.

`pinnedOpens_of`: what a pinned reader opens (`PinnedOpens`) from the closure digest's pin (`DigestPins`), the
check's provenance (`Provenance`) and a layout that reads a Definition's signature alone (`LayoutReads`).

`pin_listed`: the pin an opened row is held under is one the instance lists, from `Owns` and `Provenance`.

`unitAt_iff`: the two numberings of a template's units agree. Where the record's text gives its templates
(`NciCheck.RecordText.templatesOf`), no two of the root's ranges have one template (`range_unique`), so unit `u` of a
template (`UnitAt`) is member `u` of its one range, and the range's count is the units the text's template has. The
root's ranges are the same whatever templates the template query is asked for (`evaluate_ranges`).
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Lean (Json)
open Flock.Program (Prog Scope)
open Flock.TemplateQuery (TRange)
open NciCheck.DefinitionDigest (closureDigest)
open NciCheck.RecordText (Row Tmpl tmplOf templatesOf)

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

/-! ## The two numberings of a template's units

`UnitAt` numbers a template's units over the root's ranges: member `m` of a range, after the units of the template's
earlier ranges. The record's text numbers them from 0 to the template's `units` (`templatesOf`). The text's readers
accept only where no two ranges have one template, and there the two numberings are one. -/

private theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : x >>= f = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error => cases h
  | ok a => exact ⟨a, rfl, h⟩

/-- `eraseDups` keeps a list's length only where the list has no duplicates. -/
private theorem eraseDups_nodup {α : Type} [BEq α] [LawfulBEq α] : ∀ (n : ℕ) (l : List α), l.length ≤ n →
    l.eraseDups.length ≤ l.length ∧ (l.eraseDups.length = l.length → l.Nodup)
  | _, [], _ => by simp
  | 0, _ :: _, h => by simp at h
  | n + 1, a :: as, h => by
    have hf : (as.filter fun b => !b == a).length ≤ as.length := List.length_filter_le ..
    obtain ⟨ih1, ih2⟩ := eraseDups_nodup n (as.filter fun b => !b == a) (by simp at h; omega)
    rw [List.eraseDups_cons, List.length_cons, List.length_cons]
    refine ⟨by omega, fun he => ?_⟩
    have hfl : (as.filter fun b => !b == a).length = as.length := by omega
    have hall := List.length_filter_eq_length_iff.1 hfl
    have hfe : (as.filter fun b => !b == a) = as := List.filter_eq_self.2 hall
    rw [hfe] at ih2 he
    refine List.nodup_cons.2 ⟨fun hm => ?_, ih2 (by omega)⟩
    have := hall a hm
    simp at this

/-- Two loops over one list from one state end in one state where both accept, if their bodies give one step
wherever both accept. -/
private theorem forIn_agree {σ : Type} {f g : ℕ → σ → Except String (ForInStep σ)}
    (hfg : ∀ k s x y, f k s = .ok x → g k s = .ok y → x = y) :
    ∀ (l : List ℕ) (s x y : σ), forIn l s f = .ok x → forIn l s g = .ok y → x = y
  | [], s, x, y, hx, hy => by
    rw [List.forIn_nil] at hx hy
    cases hx
    cases hy
    rfl
  | k :: l, s, x, y, hx, hy => by
    rw [List.forIn_cons] at hx hy
    obtain ⟨a, ha, hx⟩ := bind_ok hx
    obtain ⟨b, hb, hy⟩ := bind_ok hy
    rw [← hfg k s a b ha hb] at hy
    cases a with
    | done s' =>
      have hx' : (pure s' : Except String σ) = .ok x := hx
      have hy' : (pure s' : Except String σ) = .ok y := hy
      cases hx'
      cases hy'
      rfl
    | yield s' => exact forIn_agree hfg l s' x y hx hy

/-- **The root's ranges do not depend on the templates asked for**: where the template query accepts two lists of
templates, it gives the same ranges. -/
theorem evaluate_ranges {d : Flock.TemplateQuery.Desc} {a b : Array String} {rs rs' : Array TRange}
    (h : Flock.TemplateQuery.evaluate d a = .ok rs) (h' : Flock.TemplateQuery.evaluate d b = .ok rs') :
    rs = rs' := by
  unfold Flock.TemplateQuery.evaluate at h h'
  cases hroot : d.def? d.root with
  | none =>
    simp only [hroot] at h
    cases h
  | some root =>
    simp only [hroot] at h h'
    obtain ⟨r, hloop, h⟩ := bind_ok h
    obtain ⟨r', hloop', h'⟩ := bind_ok h'
    rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hloop hloop'
    have hr : r = r' := by
      refine forIn_agree ?_ _ _ _ _ hloop hloop'
      intro k s x y hx hy
      obtain ⟨rg, at0⟩ := s
      simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at hx hy
      repeat' (first
        | (cases hx; cases hy; rfl)
        | (cases hx; done)
        | (cases hy; done)
        | contradiction
        | split at hx
        | split at hy)
    rw [← hr] at h'
    obtain ⟨rg, at0⟩ := r
    simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h h'
    repeat' (first
      | (cases h; cases h'; rfl)
      | contradiction
      | split at h
      | split at h')

/-- A template the text's reader gives of a range has the range's template and its count for units. -/
private theorem tmplOf_ok {P : Prog} {r : TRange} {t : Tmpl} (h : tmplOf P r = .ok t) :
    t.tid = r.template ∧ t.units = r.count := by
  unfold tmplOf at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  repeat (split at h <;> try cases h)
  exact ⟨rfl, rfl⟩

private theorem mapM_tmplOf {P : Prog} : ∀ (l : List TRange) (ts : List Tmpl), l.mapM (tmplOf P) = .ok ts →
    ts.map (fun t => (t.tid, t.units)) = l.map (fun r => (r.template, r.count))
  | [], ts, h => by
    rw [List.mapM_nil] at h
    cases h
    rfl
  | r :: l, ts, h => by
    rw [List.mapM_cons] at h
    obtain ⟨t, ht, h⟩ := bind_ok h
    obtain ⟨ts', hts', h⟩ := bind_ok h
    cases h
    obtain ⟨h1, h2⟩ := tmplOf_ok ht
    simp only [List.map_cons, h1, h2, mapM_tmplOf l ts' hts']

/-- What the text's reader has checked where it gives the templates: the text's template ids are distinct, the
template query accepts them, its ranges are the text's templates with their units, in order, and the templates are
the ranges', one each. The reader's `match` on a template's text has no equations for `simp` to rewrite by, so the
binds are opened without them. -/
private theorem templatesOf_ok {P : Prog} {doc : Json} {ts : List Tmpl} (h : templatesOf P doc = .ok ts) :
    ∃ (tsp : Array (String × ℕ)) (rs : Array TRange),
      (tsp.map (·.1)).toList.eraseDups.length = (tsp.map (·.1)).size ∧
      Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) (tsp.map (·.1)) = .ok rs ∧
      rs.map (fun r => (r.template, r.count)) = tsp ∧ rs.toList.mapM (tmplOf P) = .ok ts := by
  unfold templatesOf at h
  simp (config := { iota := false }) only [bind, Except.bind, pure, Except.pure, throw, throwThe,
    MonadExceptOf.throw] at h
  repeat (split at h <;> try cases h)
  rename_i tsp _ hdup _ rs₀ hev _ rs hrs hne
  have hrs' : rs₀ = rs := Except.ok.inj hrs
  exact ⟨tsp, rs, Classical.not_not.1 fun hc => hdup (bne_iff_ne.2 hc), by rw [← hrs']; exact hev,
    Classical.not_not.1 fun hc => hne (bne_iff_ne.2 hc), h⟩

/-- **The text's templates are the root's ranges.** Where the text's reader gives the templates `ts` and the template
query gives the ranges `rs`, whatever templates it is asked for: no two ranges have one template, and `ts` are the
ranges in order, each with its range's template and its count for units. -/
theorem templatesOf_ranges {P : Prog} {doc : Json} {ts : List Tmpl} {tids : Array String} {rs : Array TRange}
    (hts : templatesOf P doc = .ok ts)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) tids = .ok rs) :
    (rs.toList.map (·.template)).Nodup ∧
      ts.map (fun t => (t.tid, t.units)) = rs.toList.map (fun r => (r.template, r.count)) := by
  obtain ⟨tsp, rs₀, hdup, hev, hmap, hmapM⟩ := templatesOf_ok hts
  obtain rfl := evaluate_ranges hev hrs
  refine ⟨?_, mapM_tmplOf _ _ hmapM⟩
  have hnd : (tsp.map (·.1)).toList.Nodup :=
    (eraseDups_nodup _ (tsp.map (·.1)).toList (Nat.le_refl _)).2 (hdup.trans Array.length_toList.symm)
  subst hmap
  simpa [Function.comp_def] using hnd

/-- **One range has a template**, where the text's reader gives its templates. -/
theorem range_unique {P : Prog} {doc : Json} {ts : List Tmpl} {tids : Array String} {rs : Array TRange}
    (hts : templatesOf P doc = .ok ts)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) tids = .ok rs) {i j : ℕ}
    (hi : i < rs.size) (hj : j < rs.size) (h : rs[i]!.template = rs[j]!.template) : i = j := by
  have hnd := (templatesOf_ranges hts hrs).1
  have hi' : i < (rs.toList.map (·.template)).length := by simpa using hi
  have hj' : j < (rs.toList.map (·.template)).length := by simpa using hj
  refine (hnd.getElem_inj (hi := hi') (hj := hj')).1 ?_
  rw [getElem!_pos rs i hi, getElem!_pos rs j hj] at h
  simpa using h

/-- The text's template at a range's place has the range's template and its count for units. -/
private theorem tmpl_at {ts : List Tmpl} {rs : Array TRange}
    (hmap : ts.map (fun t => (t.tid, t.units)) = rs.toList.map (fun r => (r.template, r.count))) {i : ℕ}
    (hi : i < rs.size) : ∃ t, ts[i]? = some t ∧ t.tid = rs[i]!.template ∧ t.units = rs[i]!.count := by
  have hlen : ts.length = rs.size := by simpa using congrArg List.length hmap
  have hi' : i < ts.length := by omega
  have h : (ts.map fun t => (t.tid, t.units))[i]? = (rs.toList.map fun r => (r.template, r.count))[i]? := by
    rw [hmap]
  refine ⟨ts[i], List.getElem?_eq_getElem hi', ?_⟩
  rw [getElem!_pos rs i hi]
  simpa [hi, hi'] using h

/-- **The two numberings of a template's units agree.** Where the text's reader gives the templates `ts` and the
template query gives the ranges `rs`, whatever templates it is asked for: `sc` is unit `u` of the template `tid`
(`UnitAt`) exactly when a range of `rs`, the one of that template (`range_unique`), has `u` below its count, which is
the units of the text's template at its place, and `sc` is member `u` of the range's root node. -/
theorem unitAt_iff {P : Prog} {doc : Json} {ts : List Tmpl} {tids : Array String} {rs : Array TRange}
    (hts : templatesOf P doc = .ok ts)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) tids = .ok rs) (tid : String) (u : ℕ)
    (sc : Scope) :
    UnitAt P rs tid u sc ↔ ∃ i t, i < rs.size ∧ rs[i]!.template = tid ∧ ts[i]? = some t ∧ t.tid = tid ∧
      t.units = rs[i]!.count ∧ u < t.units ∧ sc = P.child P.rootScope rs[i]!.node u := by
  obtain ⟨-, hmap⟩ := templatesOf_ranges hts hrs
  have hzero : ∀ i, i < rs.size → rs[i]!.template = tid →
      ((rs.extract 0 i).filter fun r => r.template == tid).foldl (fun a r => a + r.count) 0 = 0 := by
    intro i hi hti
    have hemp : ((rs.extract 0 i).filter fun r => r.template == tid) = #[] := by
      rw [Array.filter_eq_empty_iff]
      intro r hr hb
      obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 hr
      have hsz : (rs.extract 0 i).size = min i rs.size - 0 := Array.size_extract
      have hji : j < i := by omega
      have hjs : j < rs.size := by omega
      have hjt : rs[j]!.template = tid := by
        rw [getElem!_pos rs j hjs]
        simpa using hb
      exact absurd (range_unique hts hrs hjs hi (hjt.trans hti.symm)) (Nat.ne_of_lt hji)
    rw [hemp]
    rfl
  unfold UnitAt
  constructor
  · rintro ⟨i, m, hi, hti, hm, hu, hsc⟩
    rw [hzero i hi hti, Nat.zero_add] at hu
    subst hu
    obtain ⟨t, ht, htt, htu⟩ := tmpl_at hmap hi
    exact ⟨i, t, hi, hti, ht, htt.trans hti, htu, by rw [htu]; exact hm, hsc⟩
  · rintro ⟨i, t, hi, hti, -, -, htu, hu, hsc⟩
    exact ⟨i, u, hi, hti, by rw [← htu]; exact hu, by rw [hzero i hi hti, Nat.zero_add], hsc⟩

end NciCheck.Properties.Record.SecurityProofs

namespace NciCheck.SecurityProofs

open NciCheck.Properties.Record

theorem ServedComplies : NciCheck.Guarantees.ServedComplies := by
  intro F hown hlaid hcount hrate
  exact NciCheck.Properties.Record.SecurityProofs.complies_of hown hlaid hcount hrate

end NciCheck.SecurityProofs
