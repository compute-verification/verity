import Definitions.NetworkCertifier.SampledAudit

/-!
# A gate's value from what it depends on, and the wires' resolution

A gate's value under held gates reads only the inputs of the gates it depends on (`valOver_congr_inp`). The circuit's
Program resolves the wires in as many rounds as `C` has gates (`resolveWires`). Under a wiring that fits the program,
the resolution is stable past the wired gates the wires' rank puts below a wired gate (`resolveWires_stable`,
`wiredBelow`), so it is a fixed point: each wired gate reads what its writer computes on the resolved inputs
(`resolveWires_fix`). A gate's resolution reads only the inputs it depends on through reads and wires
(`resolveWires_congr_inp`), so two ingresses that agree there resolve it alike. With no wires the wiring fits every
program (`wiresFit_none`), and depending through reads and wires is depending (`dependsWired_none`).
-/

namespace NetTiming

open Claims.Sampling

/-! ### A gate's value reads only what it depends on -/

theorem mapM_congr_option {ws : List ℕ} {rd rd' : ℕ → Option Bool} (h : ∀ i ∈ ws, rd i = rd' i) :
    ws.mapM rd = ws.mapM rd' := by
  induction ws with
  | nil => rfl
  | cons w ws ih =>
    simp only [List.mapM_cons]
    rw [h w List.mem_cons_self, ih fun i hi => h i (List.mem_cons_of_mem w hi)]

theorem bgate_val_congr (x : BGate) {rd rd' : ℕ → Option Bool} (own : Option Bool)
    (h : ∀ i ∈ x.reads, rd i = rd' i) : x.val rd own = x.val rd' own := by
  cases x with
  | input => rfl
  | const b => rfl
  | not i => simp only [BGate.val, h i (by simp [BGate.reads])]
  | and ws => simp only [BGate.val, mapM_congr_option (ws := ws) fun i hi => h i (by simpa [BGate.reads] using hi)]
  | xor ws => simp only [BGate.val, mapM_congr_option (ws := ws) fun i hi => h i (by simpa [BGate.reads] using hi)]

theorem depends_of_read {C : Prog} {g : ℕ} {x : BGate} (hx : C.gates[g]? = some x) {i : ℕ} (hi : i ∈ x.reads) :
    Depends C i g :=
  Relation.ReflTransGen.single (by simp [hx, hi])

theorem dependsWired_of_depends {C : Prog} {wires : ℕ → Option ℕ} {i g : ℕ} (h : Depends C i g) :
    DependsWired C wires i g := by
  unfold Depends at h
  unfold DependsWired
  induction h with
  | refl => exact Relation.ReflTransGen.refl
  | tail _ hab ih => exact ih.tail (Or.inl hab)

/-- With no wires, depending through reads and wires is depending. -/
theorem dependsWired_none {C : Prog} {wires : ℕ → Option ℕ} (hw : wires = fun _ => none) {i g : ℕ}
    (h : DependsWired C wires i g) : Depends C i g := by
  unfold DependsWired at h
  unfold Depends
  induction h with
  | refl => exact Relation.ReflTransGen.refl
  | tail _ hab ih => exact Or.elim hab (fun h1 => ih.tail h1) (fun h2 => by simp [hw] at h2)

theorem valOver_of_some (C : Prog) (inp : ℕ → Bool) {free : ℕ → Option Bool} {g : ℕ} {b : Bool}
    (hf : free g = some b) : valOver C inp free g = b := by
  rw [valOver]
  simp only [hf]

theorem valOver_of_none_none (C : Prog) (inp : ℕ → Bool) {free : ℕ → Option Bool} {g : ℕ}
    (hf : free g = none) (hx : C.gates[g]? = none) : valOver C inp free g = false := by
  rw [valOver]
  simp only [hf, hx]

theorem valOver_of_none_some (C : Prog) (inp : ℕ → Bool) {free : ℕ → Option Bool} {g : ℕ} {x : BGate}
    (hf : free g = none) (hx : C.gates[g]? = some x) :
    valOver C inp free g =
      (x.val (fun i => if _h : i < g then some (valOver C inp free i) else some false) (some (inp g))).getD false := by
  rw [valOver]
  simp only [hf, hx]

/-- Inputs that agree on every gate `g` depends on give `g` one value. -/
theorem valOver_congr_inp (C : Prog) {inp inp' : ℕ → Bool} (free : ℕ → Option Bool) (g : ℕ)
    (h : ∀ i, Depends C i g → inp i = inp' i) : valOver C inp free g = valOver C inp' free g := by
  induction g using Nat.strong_induction_on with
  | _ g ih =>
    rcases hf : free g with _ | b
    · rcases hx : C.gates[g]? with _ | x
      · rw [valOver_of_none_none C inp hf hx, valOver_of_none_none C inp' hf hx]
      · rw [valOver_of_none_some C inp hf hx, valOver_of_none_some C inp' hf hx, h g Relation.ReflTransGen.refl]
        congr 1
        apply bgate_val_congr x
        intro i hi
        split_ifs with hig
        · rw [ih i hig fun j hj => h j (hj.trans (depends_of_read hx hi))]
        · rfl
    · rw [valOver_of_some C inp hf, valOver_of_some C inp' hf]

/-! ### The wires' resolution -/

theorem resolveWires_of_none {C : Prog} {wires : ℕ → Option ℕ} {inp : ℕ → Bool} {free : ℕ → Option Bool} {g : ℕ}
    (h : wires g = none) : ∀ n, resolveWires C wires inp free n g = inp g
  | 0 => by simp only [resolveWires, h]
  | n + 1 => by simp only [resolveWires, h]

/-- With no wires, the resolution is the inputs. -/
theorem resolveWires_nil {C : Prog} {wires : ℕ → Option ℕ} (h : wires = fun _ => none) (inp : ℕ → Bool)
    (free : ℕ → Option Bool) (n : ℕ) : resolveWires C wires inp free n = inp :=
  funext fun g => resolveWires_of_none (by rw [h]) n

theorem resolveWires_succ {C : Prog} {wires : ℕ → Option ℕ} {inp : ℕ → Bool} {free : ℕ → Option Bool} {g w : ℕ}
    (h : wires g = some w) (n : ℕ) :
    resolveWires C wires inp free (n + 1) g = valOver C (resolveWires C wires inp free n) free w := by
  simp only [resolveWires, h]

open Classical in
/-- The wired gates of `C` that the rank `r` puts below `g`. -/
noncomputable def wiredBelow (C : Prog) (wires : ℕ → Option ℕ) (r : ℕ → ℕ) (g : ℕ) : Finset ℕ :=
  (Finset.range C.gates.length).filter fun g₂ => wires g₂ ≠ none ∧ r g₂ < r g

theorem wiredBelow_ssubset {C : Prog} {wires : ℕ → Option ℕ} {r : ℕ → ℕ} {g g' : ℕ} (hg' : g' < C.gates.length)
    (hw : wires g' ≠ none) (hr : r g' < r g) : wiredBelow C wires r g' ⊂ wiredBelow C wires r g := by
  refine (Finset.ssubset_iff_of_subset fun x hx => ?_).2 ⟨g', ?_, ?_⟩
  · simp only [wiredBelow, Finset.mem_filter] at hx ⊢
    exact ⟨hx.1, hx.2.1, hx.2.2.trans hr⟩
  · simp only [wiredBelow, Finset.mem_filter, Finset.mem_range]
    exact ⟨hg', hw, hr⟩
  · simp [wiredBelow]

theorem card_wiredBelow_lt {C : Prog} {wires : ℕ → Option ℕ} {r : ℕ → ℕ} {g : ℕ} (hg : g < C.gates.length) :
    (wiredBelow C wires r g).card < C.gates.length := by
  have h : wiredBelow C wires r g ⊂ Finset.range C.gates.length :=
    (Finset.ssubset_iff_of_subset (Finset.filter_subset _ _)).2 ⟨g, Finset.mem_range.2 hg, by simp [wiredBelow]⟩
  simpa using Finset.card_lt_card h

/-- With no wires, the wiring fits every program. -/
theorem wiresFit_none {L S : ℕ} {lay : BoundaryGates L S} (h : lay.wires = fun _ => none) (C : Prog) (k : ℕ) :
    lay.WiresFit C k :=
  ⟨fun g w hw => by simp [h] at hw, fun _ => 0, fun g w g' hw => by simp [h] at hw⟩

section Fit

variable {L S : ℕ} {C : Prog} {lay : BoundaryGates L S} {k : ℕ}

theorem wired_lt (hwf : lay.WiresFit C k) {g w : ℕ} (hw : lay.wires g = some w) : g < C.gates.length :=
  (List.getElem?_eq_some_iff.1 (hwf.1 g w hw).1).1

/-- A wired gate that a writer depends on ranks below that writer's wired gate, so fewer wired gates rank below it. -/
theorem card_wiredBelow_lt_of_depends (hwf : lay.WiresFit C k) {r : ℕ → ℕ}
    (hr : ∀ g w g', lay.wires g = some w → lay.wires g' ≠ none → Depends C g' w → r g' < r g) {g w g' w' : ℕ}
    (hw : lay.wires g = some w) (hw' : lay.wires g' = some w') (hdep : Depends C g' w) :
    (wiredBelow C lay.wires r g').card < (wiredBelow C lay.wires r g).card :=
  Finset.card_lt_card (wiredBelow_ssubset (wired_lt hwf hw') (by simp [hw']) (hr g w g' hw (by simp [hw']) hdep))

/-- **The resolution is stable** past the wired gates the rank puts below a wired gate: one more round leaves it. -/
theorem resolveWires_stable (hwf : lay.WiresFit C k) (r : ℕ → ℕ)
    (hr : ∀ g w g', lay.wires g = some w → lay.wires g' ≠ none → Depends C g' w → r g' < r g)
    (inp : ℕ → Bool) (free : ℕ → Option Bool) :
    ∀ m g, lay.wires g ≠ none → (wiredBelow C lay.wires r g).card < m →
      resolveWires C lay.wires inp free m g = resolveWires C lay.wires inp free (m + 1) g := by
  intro m
  induction m with
  | zero => intro g _ hc; exact absurd hc (Nat.not_lt_zero _)
  | succ m ih =>
  intro g hg hc
  obtain ⟨w, hw⟩ := Option.ne_none_iff_exists'.1 hg
  rw [resolveWires_succ hw (m + 1), resolveWires_succ hw m]
  refine valOver_congr_inp C free w fun g' hdep => ?_
  cases hw' : lay.wires g' with
  | none => rw [resolveWires_of_none hw', resolveWires_of_none hw']
  | some w' =>
    have hlt := card_wiredBelow_lt_of_depends hwf hr hw hw' hdep
    exact ih g' (by simp [hw']) (by omega)

/-- **The resolution is a fixed point** in as many rounds as `C` has gates: a wired gate reads what its writer computes
on the resolved inputs. -/
theorem resolveWires_fix (hwf : lay.WiresFit C k) (r : ℕ → ℕ)
    (hr : ∀ g w g', lay.wires g = some w → lay.wires g' ≠ none → Depends C g' w → r g' < r g)
    (inp : ℕ → Bool) (free : ℕ → Option Bool) {g w : ℕ} (hw : lay.wires g = some w) :
    resolveWires C lay.wires inp free C.gates.length g =
      valOver C (resolveWires C lay.wires inp free C.gates.length) free w := by
  have hc := card_wiredBelow_lt (wires := lay.wires) (r := r) (wired_lt hwf hw)
  obtain ⟨N, hN⟩ : ∃ N, C.gates.length = N + 1 := ⟨C.gates.length - 1, by omega⟩
  rw [hN, resolveWires_succ hw]
  refine valOver_congr_inp C free w fun g' hdep => ?_
  cases hw' : lay.wires g' with
  | none => rw [resolveWires_of_none hw', resolveWires_of_none hw']
  | some w' =>
    have hlt := card_wiredBelow_lt_of_depends hwf hr hw hw' hdep
    exact resolveWires_stable hwf r hr inp free N g' (by simp [hw']) (by omega)

end Fit

/-- **A gate's resolution reads only what it depends on through reads and wires**: inputs that agree at every gate
`g` depends on through reads and wires resolve `g` alike, in any number of rounds. -/
theorem resolveWires_congr_inp (C : Prog) (wires : ℕ → Option ℕ) {inp inp' : ℕ → Bool} (free : ℕ → Option Bool) :
    ∀ m g, (∀ i, DependsWired C wires i g → inp i = inp' i) →
      resolveWires C wires inp free m g = resolveWires C wires inp' free m g := by
  intro m
  induction m with
  | zero =>
    intro g h
    cases hw : wires g with
    | none => rw [resolveWires_of_none hw, resolveWires_of_none hw, h g Relation.ReflTransGen.refl]
    | some w => simp only [resolveWires, hw]
  | succ m ih =>
    intro g h
    cases hw : wires g with
    | none => rw [resolveWires_of_none hw, resolveWires_of_none hw, h g Relation.ReflTransGen.refl]
    | some w =>
      rw [resolveWires_succ hw, resolveWires_succ hw]
      exact valOver_congr_inp C free w fun g' hdep =>
        ih g' fun i hi => h i ((hi.trans (dependsWired_of_depends hdep)).tail (Or.inr hw))

end NetTiming
