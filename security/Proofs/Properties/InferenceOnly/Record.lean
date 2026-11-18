module

public import Properties.InferenceOnly.Record
import Mathlib.Algebra.Order.Field.Basic

/-!
# The circuit record's properties, from their parts

`complies_of`: the rate rule on a record's circuit (`Complies`) from four statements about the check, that a record it
reads carries its instance's fields (`Owns`) and lays out a circuit (`LaidOut`), that its summary counts every block
(`Counts`), and that its rate rule is the policy's on a summary (`Rates`). `InferenceOnlyCheck.SecurityProofs.ServedComplies` is
that theorem as the guarantee `InferenceOnlyCheck.Guarantees.ServedComplies` states it.

`pinnedOpens_of`: what a pinned reader opens (`PinnedOpens`) from the closure digest's pin (`DigestPins`), the
check's provenance (`Provenance`) and a layout that reads a Definition's signature alone (`LayoutReads`).

`pin_listed`: the pin an opened row is held under is one the instance lists, from `Owns` and `Provenance`.

`unitAt_iff`: the two numberings of a template's units agree. Where the record's text gives its templates
(`Spot.RecordText.templatesOf`), no two of the root's ranges have one template (`range_unique`), so unit `u` of a
template (`UnitAt`) is member `u` of its one range, and the range's count is the units the text's template has. The
root's ranges are the same whatever templates the template query is asked for (`evaluate_ranges`).
-/

@[expose] public section

namespace InferenceOnlyVerifier.Properties.Record.SecurityProofs

open Lean (Json)
open Flock.Program (Prog Scope)
open Flock.TemplateQuery (TRange)
open InferenceOnlyVerifier.DefinitionDigest (closureDigest)
open Spot.RecordText (Row Tmpl tmplOf templatesOf)

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
      have h1 : (InferenceOnly.freshInput d (d.𝓘.gates u) : ℝ) ≤ (F.summarize c).fresh := by exact_mod_cast hfresh
      have h2 : ((F.summarize c).fresh : ℝ) * (F.inst c).den ≤ (F.inst c).num * (F.summarize c).gates := by
        exact_mod_cast hα
      rw [div_mul_eq_mul_div, le_div_iff₀ hden]
      exact (mul_le_mul_of_nonneg_right h1 hden.le).trans h2
    · unfold Meets
      rw [hempty]
      simp [InferenceOnly.freshInput, Verity.Circuit.size, Verity.Circuit.into, Verity.Circuit.width]
  exact ⟨hmeets, by unfold InferenceOnly.CompliesRate; exact ⟨d.n, d.𝓘, hmeets⟩⟩

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
earlier ranges. The record's text numbers them from 0 to the template's `units` (`templatesOf`), which its reader
holds to the counts of the template's ranges summed (`unitsOf`). The two numberings cover the same units
(`unitAt_lt_iff`, `unitAt_iff`). -/

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

/-- The text's reading of a template off its first range has that range's template and the units asked for. -/
theorem tmplAt_ok {P : Prog} {rs : Array TRange} {x : String × ℕ} {t : Tmpl} (h : Spot.RecordText.tmplAt P rs x = .ok t) :
    t.tid = x.1 ∧ t.units = x.2 ∧ ∃ r ∈ rs, r.template = x.1 ∧ tmplOf P { r with count := x.2 } = .ok t := by
  unfold Spot.RecordText.tmplAt at h
  split at h
  · rename_i r hr
    obtain ⟨h1, h2⟩ := tmplOf_ok h
    have hb : r.template = x.1 := by simpa using Array.find?_some hr
    exact ⟨h1.trans hb, h2, r, Array.mem_of_find?_eq_some hr, hb, h⟩
  · cases h

private theorem mapM_tmplAt {P : Prog} {rs : Array TRange} :
    ∀ (l : List (String × ℕ)) (ts : List Tmpl), l.mapM (Spot.RecordText.tmplAt P rs) = .ok ts →
      ts.map (fun t => (t.tid, t.units)) = l
  | [], ts, h => by
    rw [List.mapM_nil] at h
    cases h
    rfl
  | x :: l, ts, h => by
    rw [List.mapM_cons] at h
    obtain ⟨t, ht, h⟩ := bind_ok h
    obtain ⟨ts', hts', h⟩ := bind_ok h
    cases h
    obtain ⟨h1, h2, -⟩ := tmplAt_ok ht
    simp only [List.map_cons, h1, h2, mapM_tmplAt l ts' hts', Prod.mk.eta]

/-- **The text's templates are read off the root's ranges**: where the text's reader gives the templates `ts`, the
template query accepts the text's template ids, which are distinct and are the templates `unitsOf` gives of its
ranges, with the units it gives them, and `ts` are those templates, each read off its first range. The reader's
`match` on a template's text has no equations for `simp` to rewrite by, so the binds are opened without them. -/
theorem templatesOf_spec {P : Prog} {doc : Json} {ts : List Tmpl} (h : templatesOf P doc = .ok ts) :
    ∃ (tids : Array String) (rs : Array TRange),
      Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) tids = .ok rs ∧ tids.toList.Nodup ∧
        tids.toList = (Spot.RecordText.unitsOf rs).map (·.1) ∧
        (Spot.RecordText.unitsOf rs).mapM (Spot.RecordText.tmplAt P rs) = .ok ts := by
  unfold templatesOf at h
  simp (config := { iota := false }) only [bind, Except.bind, pure, Except.pure, throw, throwThe,
    MonadExceptOf.throw] at h
  repeat (split at h <;> try cases h)
  rename_i tsp _ hdup _ rs₀ hev _ rs hrs hne
  have hrs' : rs₀ = rs := Except.ok.inj hrs
  have hu : Spot.RecordText.unitsOf rs = tsp.toList := Classical.not_not.1 fun hc => hne (bne_iff_ne.2 hc)
  refine ⟨tsp.map (·.1), rs, by rw [← hrs']; exact hev, ?_, by rw [hu, Array.toList_map], by rw [hu]; exact h⟩
  exact (eraseDups_nodup _ (tsp.map (·.1)).toList (Nat.le_refl _)).2
    ((Classical.not_not.1 fun hc => hdup (bne_iff_ne.2 hc)).trans Array.length_toList.symm)

/-- **The text's templates are the root's templates.** Where the text's reader gives the templates `ts` and the
template query gives the ranges `rs`, whatever templates it is asked for: `ts` have distinct ids, and are the
templates `unitsOf` gives of `rs`, each with the units it gives. -/
theorem templatesOf_ranges {P : Prog} {doc : Json} {ts : List Tmpl} {tids : Array String} {rs : Array TRange}
    (hts : templatesOf P doc = .ok ts)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) tids = .ok rs) :
    (ts.map (·.tid)).Nodup ∧ ts.map (fun t => (t.tid, t.units)) = Spot.RecordText.unitsOf rs := by
  obtain ⟨tids₀, rs₀, hev, hnd, htids, hmapM⟩ := templatesOf_spec hts
  obtain rfl := evaluate_ranges hev hrs
  have hmap := mapM_tmplAt _ _ hmapM
  refine ⟨?_, hmap⟩
  have he : ts.map (·.tid) = tids₀.toList := by
    rw [htids, ← hmap, List.map_map]
    rfl
  rw [he]
  exact hnd

/-- A template `unitsOf` gives is one of a range, with its ranges' counts summed. -/
theorem mem_unitsOf {rs : Array TRange} {x : String × ℕ} :
    x ∈ Spot.RecordText.unitsOf rs ↔ (∃ r ∈ rs, r.template = x.1) ∧
      x.2 = (Spot.RecordText.rangesAt rs x.1).foldl (fun a r => a + r.count) 0 := by
  obtain ⟨tid, n⟩ := x
  simp only [Spot.RecordText.unitsOf, List.mem_map, List.mem_eraseDups, Array.mem_toList_iff, Prod.mk.injEq]
  constructor
  · rintro ⟨a, ⟨r, hr, rfl⟩, rfl, rfl⟩
    exact ⟨⟨r, hr, rfl⟩, rfl⟩
  · rintro ⟨⟨r, hr, rfl⟩, rfl⟩
    exact ⟨_, ⟨r, hr, rfl⟩, rfl, rfl⟩

private theorem before_succ (rs : Array TRange) (tid : String) {n : ℕ} (hn : n < rs.size) :
    ((rs.extract 0 (n + 1)).filter fun r => r.template == tid).foldl (fun a r => a + r.count) 0 =
      ((rs.extract 0 n).filter fun r => r.template == tid).foldl (fun a r => a + r.count) 0 +
        if rs[n]!.template = tid then rs[n]!.count else 0 := by
  rw [Array.extract_succ_right (Nat.succ_pos n) hn, Array.filter_push, getElem!_pos rs n hn]
  by_cases h : rs[n].template = tid
  · simp [h]
  · simp [h]

private theorem below_iff (rs : Array TRange) (tid : String) (u : ℕ) : ∀ n, n ≤ rs.size →
    ((∃ i m, i < n ∧ rs[i]!.template = tid ∧ m < rs[i]!.count ∧
        u = ((rs.extract 0 i).filter fun r => r.template == tid).foldl (fun a r => a + r.count) 0 + m) ↔
      u < ((rs.extract 0 n).filter fun r => r.template == tid).foldl (fun a r => a + r.count) 0)
  | 0, _ => by
    have h0 : ((rs.extract 0 0).filter fun r => r.template == tid).foldl (fun a r => a + r.count) 0 = 0 := by
      rw [Array.extract_zero]
      simp
    rw [h0]
    simp
  | n + 1, hn => by
    have ih := below_iff rs tid u n (by omega)
    rw [before_succ rs tid (by omega : n < rs.size)]
    constructor
    · rintro ⟨i, m, hi, hti, hm, hu⟩
      rcases Nat.lt_succ_iff_lt_or_eq.1 hi with hi | rfl
      · have := ih.1 ⟨i, m, hi, hti, hm, hu⟩
        omega
      · rw [if_pos hti]
        omega
    · intro hu
      by_cases hlt : u < ((rs.extract 0 n).filter fun r => r.template == tid).foldl (fun a r => a + r.count) 0
      · obtain ⟨i, m, hi, h⟩ := ih.2 hlt
        exact ⟨i, m, by omega, h⟩
      · by_cases ht : rs[n]!.template = tid
        · rw [if_pos ht] at hu
          exact ⟨n, u - ((rs.extract 0 n).filter fun r => r.template == tid).foldl (fun a r => a + r.count) 0,
            by omega, ht, by omega, by omega⟩
        · rw [if_neg ht] at hu
          omega

/-- **A template's units are its ranges' members**: some scope is unit `u` of the template `tid` (`UnitAt`) exactly
when `u` is below the counts of `tid`'s ranges summed (`rangesAt`), the units `unitsOf` gives it. -/
theorem unitAt_lt_iff (P : Prog) (rs : Array TRange) (tid : String) (u : ℕ) :
    (∃ sc, UnitAt P rs tid u sc) ↔ u < (Spot.RecordText.rangesAt rs tid).foldl (fun a r => a + r.count) 0 := by
  have h := below_iff rs tid u rs.size (Nat.le_refl _)
  rw [Array.extract_size] at h
  rw [show Spot.RecordText.rangesAt rs tid = rs.filter fun r => r.template == tid from rfl, ← h]
  unfold UnitAt
  constructor
  · rintro ⟨sc, i, m, hi, hti, hm, hu, -⟩
    exact ⟨i, m, hi, hti, hm, hu⟩
  · rintro ⟨i, m, hi, hti, hm, hu⟩
    exact ⟨_, i, m, hi, hti, hm, hu, rfl⟩

private theorem before_mono (rs : Array TRange) (tid : String) (a : ℕ) : ∀ b, a ≤ b → b ≤ rs.size →
    ((rs.extract 0 a).filter fun r => r.template == tid).foldl (fun a r => a + r.count) 0 ≤ ((rs.extract 0 b).filter fun r => r.template == tid).foldl (fun a r => a + r.count) 0
  | 0, hab, _ => by rw [Nat.le_zero.1 hab]
  | b + 1, hab, hb => by
    rcases Nat.lt_or_ge a (b + 1) with h | h
    · have := before_mono rs tid a b (by omega) (by omega)
      rw [before_succ rs tid (by omega : b < rs.size)]
      omega
    · rw [Nat.le_antisymm hab h]

/-- **A unit of a template is one scope**: member `m` of one range, the template's units before it and `m` being
`u`. -/
theorem unitAt_unique {P : Prog} {rs : Array TRange} {tid : String} {u : ℕ} {sc sc' : Scope}
    (h : UnitAt P rs tid u sc) (h' : UnitAt P rs tid u sc') : sc = sc' := by
  unfold UnitAt at h h'
  obtain ⟨i, m, hi, hti, hm, hu, rfl⟩ := h
  obtain ⟨i', m', hi', hti', hm', hu', rfl⟩ := h'
  have key : ∀ {a b ma mb : ℕ}, a < b → b < rs.size → rs[a]!.template = tid → ma < rs[a]!.count →
      ((rs.extract 0 a).filter fun r => r.template == tid).foldl (fun a r => a + r.count) 0 + ma ≠ ((rs.extract 0 b).filter fun r => r.template == tid).foldl (fun a r => a + r.count) 0 + mb := by
    intro a b ma mb hab hb hta hma
    have h1 := before_succ rs tid (by omega : a < rs.size)
    rw [if_pos hta] at h1
    have h2 := before_mono rs tid (a + 1) b hab hb.le
    omega
  rcases Nat.lt_trichotomy i i' with hlt | rfl | hgt
  · exact absurd (hu.symm.trans hu') (key hlt hi' hti hm)
  · have hmm : m = m' := by omega
    subst hmm
    rfl
  · exact absurd (hu'.symm.trans hu) (key hgt hi hti' hm')

/-- **The two numberings of a template's units agree.** Where the text's reader gives the templates `ts` and the
template query gives the ranges `rs`, whatever templates it is asked for: some scope is unit `u` of the template
`tid` (`UnitAt`) exactly when the text's template of that id has more units than `u`. -/
theorem unitAt_iff {P : Prog} {doc : Json} {ts : List Tmpl} {tids : Array String} {rs : Array TRange}
    (hts : templatesOf P doc = .ok ts)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) tids = .ok rs) (tid : String) (u : ℕ) :
    (∃ sc, UnitAt P rs tid u sc) ↔ ∃ t ∈ ts, t.tid = tid ∧ u < t.units := by
  obtain ⟨-, hmap⟩ := templatesOf_ranges hts hrs
  rw [unitAt_lt_iff]
  constructor
  · intro hu
    have hex : ∃ r ∈ rs, r.template = tid := by
      refine Classical.not_not.1 fun hne => ?_
      have he : Spot.RecordText.rangesAt rs tid = #[] := by
        rw [Spot.RecordText.rangesAt, Array.filter_eq_empty_iff]
        intro r hr hb
        exact hne ⟨r, hr, by simpa using hb⟩
      rw [he] at hu
      simp at hu
    have hx := (mem_unitsOf (rs := rs) (x := (tid, (Spot.RecordText.rangesAt rs tid).foldl
      (fun a r => a + r.count) 0))).2 ⟨hex, rfl⟩
    rw [← hmap] at hx
    obtain ⟨t, ht, he⟩ := List.mem_map.1 hx
    simp only [Prod.mk.injEq] at he
    exact ⟨t, ht, he.1, by rw [he.2]; exact hu⟩
  · rintro ⟨t, ht, rfl, hu⟩
    have hx : (t.tid, t.units) ∈ Spot.RecordText.unitsOf rs := by
      rw [← hmap]
      exact List.mem_map_of_mem ht
    rw [← (mem_unitsOf.1 hx).2]
    exact hu

end InferenceOnlyVerifier.Properties.Record.SecurityProofs

namespace InferenceOnlyVerifier.SecurityProofs

open InferenceOnlyVerifier.Properties.Record

theorem ServedComplies : InferenceOnlyVerifier.Guarantees.ServedComplies := by
  intro F hown hlaid hcount hrate
  exact InferenceOnlyVerifier.Properties.Record.SecurityProofs.complies_of hown hlaid hcount hrate

end InferenceOnlyVerifier.SecurityProofs
