module

public import Proofs.Properties.Nci.RecordBlocks
public import Proofs.Properties.Nci.RecordCircuit
public import Proofs.Properties.Nci.RecordForm
public import Proofs.Properties.Nci.RecordFreshRows
public import Proofs.Properties.Nci.RecordGates
public import Proofs.Properties.Nci.RecordInputs
public import Proofs.Properties.Nci.RecordJson
public import Proofs.Properties.Nci.RecordKinds
public import Proofs.Properties.Nci.RecordLanding
public import Proofs.Properties.Nci.RecordLeafWide
public import Proofs.Properties.Nci.RecordProgram
public import Proofs.Properties.Nci.RecordRead
public import Proofs.Properties.Nci.RecordReads
public import Proofs.Properties.Nci.RecordRowBits
public import Proofs.Properties.Nci.RecordRows
public import Proofs.Properties.Nci.RecordTemplates
public import Proofs.Properties.Nci.RecordWide
public import Proofs.Properties.Nci.RecordWire

/-!
# A block reads at most the summary's fresh bits

Of a record the check reads and a circuit the record lays out (`Served`), a block reads from outside itself, θ
aside, gates no wider together than the summary's fresh bits (`block_fresh`), where the fresh rows of a slot are within
the summary's fresh (`FreshRows`):

* a gate the block reads from outside itself that θ does not hold is one that a row of the block's slot lands on, a
  row the check counts neither as wired inside its slot nor as θ (`reads_land`, with `inside_writer` and
  `theta_held` for the check's two kinds);
* the gates one row's reads land on are together no wider than the row's bits in the record, its port's bits over the
  port's cut (`landing_bits`): the landing bound (`landing_le`), the cut dividing the port's outer length
  (`cut_outer`), and the record's template of the unit having the Definition's parameters for rows (`unit_tmpl`);
* so the gates are charged each to one such row, and the rows' bits are within the summary's fresh.
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Flock.Program (Prog Scope)
open Flock.TemplateQuery (TRange)
open NciCheck.RecordText (Row outerOf)
open NciCheck.Record (Case read Kind)
open NciCheck.Record.SecurityProofs (form read_front front_text finish_text mem_of_lookup rowBits rowBits_eq FreshRows
  tmplOf_spec bitsOf_eq program_of_read)
open Spot.Bits (bitsOf)
open Lean (Json)

/-- A key with a value in a list has one by `lookup`. -/
theorem lookup_of_mem {α β : Type} [BEq α] [LawfulBEq α] : ∀ (l : List (α × β)) (a : α) (b : β), (a, b) ∈ l →
    ∃ b', l.lookup a = some b'
  | [], _, _, h => by cases h
  | (a', b') :: l, a, b, h => by
    by_cases he : a = a'
    · subst he
      exact ⟨b', by simp [List.lookup]⟩
    · rcases List.mem_cons.1 h with h | h
      · cases h
        exact absurd rfl he
      · obtain ⟨b'', hb⟩ := lookup_of_mem l a b h
        have hne : (a == a') = false := beq_eq_false_iff_ne.2 he
        exact ⟨b'', by simp [List.lookup, hne, hb]⟩

/-- **The cut of a unit's port divides the parameter's outer length**, in a wiring `Spot.Wires.check` accepts whose
templates are the Program's. -/
theorem cut_outer {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary}
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) (W.templates.map (·.1)).toArray = .ok rs)
    (hT : Templates P rs W) (hS : Spot.Wires.check W = .ok S) {tid : String} {u : ℕ} {sc : Scope}
    (hu : UnitAt P rs tid u sc) {i : ℕ} (hi : i < (P.defs[sc.d]!).params.size) :
    0 < W.cutsOf tid (P.defs[sc.d]!).params[i]!.1 ∧
      outerOf (P.defs[sc.d]!).params[i]!.2 % W.cutsOf tid (P.defs[sc.d]!).params[i]!.1 = 0 := by
  obtain ⟨t0, ht0, -⟩ := unit_template hrs hT hu
  obtain ⟨t, hlt⟩ := lookup_of_mem _ _ _ ht0
  obtain ⟨hnames, -, hrows, -⟩ := (hT tid t (mem_of_lookup _ _ _ hlt)).2 u sc hu
  have hq0 : ((P.defs[sc.d]!).params[i]!.1, (⟨bitsOf (P.defs[sc.d]!).params[i]!.2 / 16,
      outerOf (P.defs[sc.d]!).params[i]!.2⟩ : Spot.Wires.Port)) ∈ t.rows := (hrows _ _).2 ⟨i, hi, rfl, rfl, rfl⟩
  obtain ⟨q, hlq⟩ := lookup_of_mem _ _ _ hq0
  obtain ⟨i', hi', hpi', -, houter⟩ := (hrows _ q).1 (mem_of_lookup _ _ _ hlq)
  obtain rfl : i' = i := hnames i' i hi' hi hpi'.symm
  obtain ⟨hk0, hko, -⟩ := cut_divides hS hlt hlq
  exact ⟨hk0, by rw [← houter]; exact hko⟩

/-- **A row's reads land on no more bits than the record gives the row.** A row some read lands through is a row of
a template of the record, inside its port's cut, and the gates its reads land on are together no wider than the row's
bits in the record: the port's bits over its cut. -/
theorem landing_bits {j tr : Json} {c : Case} (h : read j tr = .ok c) {rs : Array TRange} {S : Spot.Wires.Summary}
    {D : Verity.Circuit} {at_ : Fin D.N → ℕ}
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg c.f.P)
      (c.W.templates.map (·.1)).toArray = .ok rs)
    (hT : Templates c.f.P rs c.W) (hOwn : OwnRows c.f.P rs c.W) (hAl : Aligned c.f.P rs c.W S)
    (hL : LaysOut c.f.P rs c.W S D at_) (hS : Spot.Wires.check c.W = .ok S) {r : Row} {a : Fin D.N}
    (ha : a ∈ landing c.f.P rs c.W S at_ r) :
    (∃ t ∈ c.f.ts, t.tid = r.tid ∧ ∃ row ∈ t.rows, row.1 = r.port) ∧ r.row < c.W.cutsOf r.tid r.port ∧
      D.width (landing c.f.P rs c.W S at_ r) ≤ rowBits c.f.ts c.W r.tid r.port := by
  obtain ⟨f, pin, hf, hc⟩ := read_front h
  obtain ⟨rfl, -⟩ := finish_text hc
  have hts := front_text hf
  have hP := readProg_of_read h
  obtain ⟨-, -, -, -, hO, -⟩ := program_of_read h
  obtain ⟨hnd, hnames⟩ := tmpls_named hts
  obtain ⟨l, o, lo, hi, ⟨sc, hu, i, hi', hp, hrow, -⟩, -⟩ := mem_landing.1 ha
  obtain ⟨t, ht, htid, -, hdefn, r0, hr0⟩ := unit_tmpl hP hts hrs hu
  obtain ⟨-, -, -, hrows, hbits, -, -⟩ := tmplOf_spec hr0
  rw [hdefn] at hrows hbits
  have hmem : (c.f.P.defs[sc.d]!).params[i]! ∈ (c.f.P.defs[sc.d]!).params.toList := by
    rw [getElem!_pos (c.f.P.defs[sc.d]!).params i hi']
    exact Array.getElem_mem_toList hi'
  have hrowm : ((c.f.P.defs[sc.d]!).params[i]!.1, Flock.Sigma.bits (c.f.P.defs[sc.d]!).params[i]!.2 / 16,
      outerOf (c.f.P.defs[sc.d]!).params[i]!.2) ∈ t.rows := by
    rw [hrows]
    exact List.mem_map.2 ⟨_, hmem, rfl⟩
  obtain ⟨hk0, hko⟩ := cut_outer hrs hT hS hu hi'
  rw [hp] at hko
  refine ⟨⟨t, ht, htid, _, hrowm, hp⟩, hrow, ?_⟩
  obtain ⟨ρ, hρ⟩ := ranked hS
  have hle := landing_le c.f.P rs c.W S D at_ hOwn hAl hL (leafWide hP hO hrs) (returnsOwn hP hrs hT hS)
    (unitsComputed hP hrs) (portsNamed hrs hT) (outFits hT hS) ρ hρ r sc hu i hi' hp
  have hrb := rowBits_eq c.W hnd ht (List.Nodup.of_append_left (hnames t ht)) hrowm
  dsimp only at hrb
  rw [htid, hp] at hrb
  rw [inRow_bits _ _ _ _ hko hrow, bitsOf_eq] at hle
  have h16 := (hbits _ hmem).2
  have he : 16 * (Flock.Sigma.bits (c.f.P.defs[sc.d]!).params[i]!.2 / 16) =
      Flock.Sigma.bits (c.f.P.defs[sc.d]!).params[i]!.2 := by omega
  rw [hrb, he]
  exact hle

/-- **A block reads at most the summary's fresh bits**, where the fresh rows of a slot are within the summary's
fresh (`FreshRows`). A gate the block reads from outside itself that θ does not hold is one a row of the block's slot
lands on, a row that is neither wired inside the slot nor held (`reads_land`); the gates one row's reads land on are
no wider than the row's bits (`landing_bits`); and those rows' bits are within the summary's fresh. -/
theorem block_fresh (hFR : FreshRows) {j tr : Json} {c : Case} (h : read j tr = .ok c)
    {d : Verity.RegisteredCircuit} (hd : Served form c d) (u : Fin d.n) :
    Nci.novelReads d (d.𝓘.gates u) ≤ (form.summarize c).fresh := by
  classical
  obtain ⟨P, doc, Wt, S, rs, at_, hdec, -, -, -, -, -, -, hS, hrs, hT, hOwn, hAl, -, -, hL, hNm, hD⟩ := hd
  obtain rfl : c.f.P = P := Except.ok.inj ((decode_of_read h).symm.trans hdec)
  obtain rfl : c.wired = S := Except.ok.inj ((wired_of_read h).symm.trans hS)
  have hS : Spot.Wires.check c.W = .ok c.wired := hS
  have hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg c.f.P)
      (c.W.templates.map (·.1)).toArray = .ok rs := hrs
  have hT : Templates c.f.P rs c.W := hT
  have hOwn : OwnRows c.f.P rs c.W := hOwn
  have hAl : Aligned c.f.P rs c.W c.wired := hAl
  have hL : LaysOut c.f.P rs c.W c.wired d.D at_ := hL
  have hNm : Names c.f.P rs c.W (fun r => (c.heldAt r).isSome) at_ d.θ := hNm
  have hD : Divides c.f.P rs c.slot at_ d.𝓘 := hD
  have hP := readProg_of_read h
  obtain ⟨-, -, -, -, hO, -⟩ := program_of_read h
  obtain ⟨ρ, hρ⟩ := ranked hS
  obtain ⟨f, pin, hf, hc⟩ := read_front h
  have hland : ∀ a ∈ d.D.into (d.𝓘.gates u) \ d.θ, ∃ r : Row, c.slot r.tid r.unit = some u.val ∧
      c.rowKind r ≠ Kind.inside ∧ c.rowKind r ≠ Kind.theta ∧ a ∈ landing c.f.P rs c.W c.wired at_ r := by
    intro a ha
    obtain ⟨ha1, ha2⟩ := Finset.mem_sdiff.1 ha
    obtain ⟨r, hs, hni, hh, hl⟩ := reads_land hOwn hL (unitsComputed hP hrs) (operandsOwn hP hO hrs hT hS)
      (returnsOwn hP hrs hT hS) (portsNamed hrs hT) hρ
      (fun tid u sc tid' u' sc' g hu hu' h1 h2 h1' h2' => unit_of_gate hP hrs hu hu' h1 h2 h1' h2') hD
      (fun tid u sc hu => (slot_of_unit h hrs hu).imp fun s hs => hs.1) hNm
      (fun r hr => held_unwired h r hr) (fun r => c.rowKind r = Kind.inside) (fun r hr => inside_writer h r hr)
      u a ha1 ha2
    refine ⟨r, hs, hni, fun hk => ?_, hl⟩
    have hth := theta_held hc r hk
    rw [hth] at hh
    cases hh
  choose! row hrow using hland
  rcases (d.D.into (d.𝓘.gates u) \ d.θ).eq_empty_or_nonempty with hAe | ⟨a0, ha0⟩
  · show d.D.width (d.D.into (d.𝓘.gates u) \ d.θ) ≤ _
    rw [hAe]
    exact Nat.zero_le _
  have hsS : u.val < c.S := by
    obtain ⟨hs0, -, -, hl⟩ := hrow a0 ha0
    obtain ⟨l, o, lo, hi, ⟨sc, hu, -⟩, -⟩ := mem_landing.1 hl
    obtain ⟨s, hs, hlt⟩ := slot_of_unit h hrs hu
    rw [hs] at hs0
    cases hs0
    exact hlt
  let φ : Fin d.D.N → String × ℕ × String × ℕ := fun a => ((row a).tid, (row a).unit, (row a).port, (row a).row)
  have hbound := hFR j tr c h u.val hsS ((d.D.into (d.𝓘.gates u) \ d.θ).image φ) (by
    intro x hx
    obtain ⟨a, ha, rfl⟩ := Finset.mem_image.1 hx
    obtain ⟨hs, hni, hnt, hl⟩ := hrow a ha
    obtain ⟨h1, h2, -⟩ := landing_bits h hrs hT hOwn hAl hL hS hl
    exact ⟨h1, h2, hs, hni, hnt⟩)
  refine le_trans ?_ hbound
  show d.D.width (d.D.into (d.𝓘.gates u) \ d.θ) ≤ _
  unfold Verity.Circuit.width
  rw [← Finset.sum_fiberwise_of_maps_to (g := φ) (t := (d.D.into (d.𝓘.gates u) \ d.θ).image φ)
    (fun a ha => Finset.mem_image_of_mem φ ha)]
  refine Finset.sum_le_sum fun x hx => ?_
  obtain ⟨a, ha, rfl⟩ := Finset.mem_image.1 hx
  obtain ⟨-, -, -, hl⟩ := hrow a ha
  obtain ⟨-, -, hle⟩ := landing_bits h hrs hT hOwn hAl hL hS hl
  refine le_trans (Finset.sum_le_sum_of_subset ?_) hle
  intro b hb
  obtain ⟨hbA, hbφ⟩ := Finset.mem_filter.1 hb
  have hrb : row b = row a := by
    obtain ⟨h1, h2, h3, h4⟩ : (row b).tid = (row a).tid ∧ (row b).unit = (row a).unit ∧
        (row b).port = (row a).port ∧ (row b).row = (row a).row := by
      simpa only [φ, Prod.mk.injEq] using hbφ
    cases hr : row b
    cases hr' : row a
    rw [hr, hr'] at h1 h2 h3 h4
    simp only at h1 h2 h3 h4
    subst h1 h2 h3 h4
    rfl
  rw [← hrb]
  exact (hrow b hbA).2.2.2

end NciCheck.Properties.Record.SecurityProofs
