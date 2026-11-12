module

public import Proofs.Flock.SpotFlock.AddressFacts.Row

/-!
# The read's kind is the auditor's

Round 2 tells a read's kind (`AddressRow.kindOf`) from the run's interior record: whether the record registers the
value (`SecondStage.registeredEntry`) or has a port of its name (`Record.byPort`). `recordsOk` holds that record to
name exactly the interior registration's names (`Held.interiorNames`), so the kind is the one the auditor's copies
give (`kindHeld`): `kindFact`, which is `AddressRow.KindFact`.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Lean (Json)

namespace AddressRow

/-- **Names read off a list are its entries' strings**: where `g` reads each entry's `key` as a string, and `mapM g`
reads names `ns` off list `l`, `v` is one of them exactly when some entry's `key` is the string `v`. -/
theorem mem_mapM {g : Json → Option String} {key : String} (hg : ∀ e s, g e = some s ↔ Spot.getD e key = .str s)
    (v : String) : ∀ {l : List Json} {ns : List String}, l.mapM g = some ns → (v ∈ ns ↔ ∃ e ∈ l, Spot.getD e key = .str v)
  | [], ns, h => by simp at h; subst h; simp
  | a :: l, ns, h => by
    rw [List.mapM_cons] at h
    cases hga : g a with
    | none => simp [hga] at h
    | some s =>
      cases hl : l.mapM g with
      | none => simp [hga, hl] at h
      | some ns' =>
        simp [hga, hl] at h
        subst h
        have ha := (hg a s).1 hga
        simp only [List.mem_cons, mem_mapM hg v hl]
        constructor
        · rintro (rfl | ⟨e, he, hk⟩)
          · exact ⟨a, Or.inl rfl, by rw [ha]⟩
          · exact ⟨e, Or.inr he, hk⟩
        · rintro ⟨e, (rfl | he), hk⟩
          · rw [ha] at hk; cases hk; exact Or.inl rfl
          · exact Or.inr ⟨e, he, hk⟩

/-- **The names `namesAt` lists are its entries' strings**: where `rec`'s `field` is the array `es` and `namesAt` reads
names `ns` off it, `v` is one of them exactly when some entry's `key` is the string `v`. -/
theorem mem_namesAt {rec : Json} {field key : String} {es : Array Json} {ns : List String}
    (hes : Spot.getD rec field = .arr es) (hn : namesAt rec field key = some ns) (v : String) :
    v ∈ ns ↔ ∃ e ∈ es, Spot.getD e key = .str v := by
  unfold namesAt at hn
  rw [hes] at hn
  simp only at hn
  have hm := mem_mapM (key := key) (fun e s => by cases Spot.getD e key <;> simp) v hn
  simp only [Array.mem_toList_iff] at hm
  exact hm

/-- **`byPort` names a port exactly when a root does**: where `rec`'s roots are the array `rs`, some port `byPort` lists
is named `v` exactly when some root's port is the string `v`. -/
theorem mem_byPort {rec : Json} {rs : Array Json} (hrs : Spot.getD rec "roots" = .arr rs) (v : String) :
    (∃ x ∈ Spot.Record.byPort rec, x.1 = v) ↔ ∃ r ∈ rs, Spot.getD r "port" = .str v := by
  unfold Spot.Record.byPort
  rw [hrs]
  simp only
  rw [← Array.foldl_toList]
  have hrs' : (∃ r ∈ rs, Spot.getD r "port" = .str v) ↔
      (∃ x ∈ (#[] : Array (String × Json)), x.1 = v) ∨ ∃ r ∈ rs.toList, Spot.getD r "port" = .str v := by
    simp
  rw [hrs']
  generalize (#[] : Array (String × Json)) = acc
  generalize rs.toList = l
  induction l generalizing acc with
  | nil => simp
  | cons a l ih =>
    rw [List.foldl_cons]
    refine (ih _).trans ?_
    cases ha : Spot.getD a "port" with
    | str p =>
      simp only
      constructor
      · rintro (⟨x, hx, rfl⟩ | h)
        · rw [Array.mem_push, Array.mem_filter] at hx
          rcases hx with ⟨hx, -⟩ | rfl
          · exact Or.inl ⟨x, hx, rfl⟩
          · exact Or.inr ⟨a, List.mem_cons_self, ha⟩
        · rcases h with ⟨r, hr, he⟩
          exact Or.inr ⟨r, List.mem_cons_of_mem _ hr, he⟩
      · rintro (⟨x, hx, rfl⟩ | ⟨r, hr, he⟩)
        · by_cases hp : x.1 = p
          · exact Or.inl ⟨(p, a), Array.mem_push_self, hp.symm⟩
          · exact Or.inl ⟨x, Array.mem_push.2 (Or.inl (Array.mem_filter.2 ⟨hx, by simp [hp]⟩)), rfl⟩
        · rcases List.mem_cons.1 hr with rfl | hr
          · rw [ha] at he
            cases he
            exact Or.inl ⟨_, Array.mem_push_self, rfl⟩
          · exact Or.inr ⟨r, hr, he⟩
    | null => simp [ha]
    | bool _ => simp [ha]
    | num _ => simp [ha]
    | arr _ => simp [ha]
    | obj _ => simp [ha]

/-- **`byPort` finds a port exactly when `namesAt` lists it.** -/
theorem byPort_isSome_iff {rec : Json} {ns : List String} (hn : namesAt rec "roots" "port" = some ns) (v : String) :
    ((Spot.Record.byPort rec).find? (·.1 == v)).isSome = true ↔ v ∈ ns := by
  rw [Array.find?_isSome]
  simp only [beq_iff_eq]
  cases hr : Spot.getD rec "roots" with
  | arr rs => rw [mem_byPort hr, mem_namesAt hr hn]
  | null | bool _ | num _ | str _ | obj _ =>
    unfold Spot.Record.byPort namesAt at *
    rw [hr] at hn ⊢
    simp at hn
    subst hn
    simp

/-- **`registeredEntry` finds a value exactly when `namesAt` lists it.** -/
theorem registeredEntry_isSome_iff {rec : Json} {ns : List String} (hn : namesAt rec "registered" "value" = some ns)
    (v : String) : (Spot.SecondStage.registeredEntry rec v).isSome = true ↔ v ∈ ns := by
  unfold Spot.SecondStage.registeredEntry
  rw [Array.find?_isSome]
  have hs : ∀ e, Spot.jsonStrIs (Spot.getD e "value") v = true ↔ Spot.getD e "value" = .str v := fun e => by
    cases Spot.getD e "value" <;> simp [Spot.jsonStrIs]
  simp only [hs]
  cases hr : Spot.getD rec "registered" with
  | arr es => rw [mem_namesAt hr hn]
  | null | bool _ | num _ | str _ | obj _ =>
    unfold namesAt at hn
    rw [hr] at hn
    simp at hn
    subst hn
    simp

variable (h : Held)

/-- **An accepted run's statement check draws the second stage**: where `stmtOk` passes and there is a template, the
second stage's draw reads (`secondDraw`), which each template's `readsOk` asks. -/
theorem draw_of_stmtOk {x m₁ m₂ : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h}
    (hok : stmtOk h x m₁ m₂ c₀ c₁ = true) {t : ℕ} (ht : t < h.ms.length) :
    ∃ draw, secondDraw h c₀ c₁ = some draw := by
  cases hd : secondDraw h c₀ c₁ with
  | some d => exact ⟨d, rfl⟩
  | none =>
    exfalso
    unfold stmtOk at hok
    cases hp : parse h x with
    | none => simp [hp] at hok
    | some s =>
      cases hx : stmtsOf m₂ with
      | none => simp [hp, hx] at hok
      | some xs =>
        simp only [hp, hx, Bool.and_eq_true, List.all_eq_true, List.mem_range] at hok
        have hr := (hok.2 t ht).2
        unfold readsOk at hr
        rw [hd] at hr
        cases decode x <;> cases jsonMsg m₁ <;> simp at hr

/-- **The read's kind is the auditor's** (`KindFact`). -/
theorem kindFact : KindFact h := by
  intro r hr t ht port v
  obtain ⟨Pt, d, spec, -, -, -, -, hok, -⟩ := hr
  obtain ⟨st, rec, hst, hrec, -, hrn⟩ := recordsOk_spec h (recordsOk_of_stmtOk h hok)
  obtain ⟨draw, hdraw⟩ := draw_of_stmtOk h hok ht
  obtain ⟨⟨ps, hps, -, hpsm⟩, vs, hvs, -, hvsm⟩ := recordOk_spec hrn
  have hp : ((Spot.Record.byPort rec).find? (·.1 == v)).isSome = h.interiorNames.ports.contains v := by
    rw [Bool.eq_iff_iff, byPort_isSome_iff hps, List.contains_iff_mem]
    exact hpsm v
  have hv : (Spot.SecondStage.registeredEntry rec v).isSome = h.interiorNames.values.contains v := by
    rw [Bool.eq_iff_iff, registeredEntry_isSome_iff hvs, List.contains_iff_mem]
    exact hvsm v
  have hl : Spot.SecondStage.linkOf (ownOf h st rec draw r.c₀) (h.names.getD t "") port =
      Spot.SecondStage.linkOf (heldOwn h) (h.names.getD t "") port := rfl
  have hw : Spot.SecondStage.wireOf (ownOf h st rec draw r.c₀) (h.names.getD t "") port =
      Spot.SecondStage.wireOf (heldOwn h) (h.names.getD t "") port := rfl
  have hi : (ownOf h st rec draw r.c₀).interior = rec := rfl
  simp only [kindAt, hst, hrec, hdraw, kindOf, kindHeld, hl, hw, hi, hp, hv]
  cases Spot.SecondStage.linkOf (heldOwn h) (h.names.getD t "") port <;>
    cases Spot.SecondStage.wireOf (heldOwn h) (h.names.getD t "") port <;> rfl

end AddressRow

end SpotFlock
