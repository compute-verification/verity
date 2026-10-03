import FlockSoundness.Discharge.PrivateCircuit.InstanceSound

/-!
# Protocol 2 at the universal unit, with the registered check at the drawn units only

`private_circuit_sound` takes the registered check at every program site on every accepted outcome (`hchk`). A sampled
`--zk --registered` session checks the registered reads at the outcome's drawn instances only
(`ZkReg.checked_of_accepts`), so here the check is per outcome, at the reads `A o` it covers (`readsOn`), and the
binding steps are read where it holds:

* **(i) the leaves at the checked sites.** The session's opening of the program commitment at the checked sites, the
  registrant's elsewhere (`openingOn`), opens the registered root, so leaves other than the registrant's at a checked
  site are a SHA-512 collision (`progCommit_binding`).
* **The program at the drawn units.** Where those leaves are the registrant's, the committed values hold `pad C c`'s
  program at every drawn unit, or SHA-512 collides (`LeafBindsOn`).
* **The units of `C`.** A unit of `C` wrong at the committed values' claimed inputs and outputs, whose committed program is
  `pad C c`'s (`wrongAgree`), is a wrong unit of the universal unit's partition (`UnitWrong`, `pad_correct`), so
  protocol 1's headline (`ZkSessionSound`) bounds them.

`private_circuit_sound_drawn` bounds accepting with at least `K₀` such units, with a drawn unit's committed program other
than `pad C c`'s, or with a checked leaf other than the registrant's. What the drawn units' check cannot bound is a
committed program other than `pad C c`'s at undrawn units (`forged`): every wrong unit of `C` is one of the first or one
of those (`wrong_le_wrongAgree_add`). Its second conjunct counts `C`'s wrong units directly, given the law's miss at
`forged` (`ForgedMiss`).
-/

namespace FlockSoundness.Discharge.PrivateCircuit

open Flock FlockSoundness.Game FlockSoundness.Registered Refine Binding Audit
open scoped ENNReal

/-! ## The units of `C` the committed program runs as `pad C c` -/

/-- Unit `i` of the program `p` is `q`'s (vacuous past the class's units). -/
def agreesAt {c : Universal.Cls} (p q : c.Prog) (i : ℕ) : Prop := ∀ h : i < c.N, p ⟨i, h⟩ = q ⟨i, h⟩

open Classical in
/-- **`C`'s wrong units where the program is `pad C c`'s**: the units `i` of `C` whose output on the claimed inputs `x i`
is not the claimed output `y i`, and at which the program `p` is `pad C c`'s. -/
noncomputable def wrongAgree (C : Universal.Circ) (c : Universal.Cls) (p : c.Prog) (x : ℕ → ℕ → Bool)
    (y : ℕ → List Bool) : ℕ :=
  ((Finset.range (Universal.size C)).filter fun i =>
    Universal.eval C i (x i) ≠ y i ∧ agreesAt p (Universal.pad C c) i).card

open Classical in
/-- **The forged units**: the class's units at which the program `p` is not `pad C c`'s. -/
noncomputable def forged (C : Universal.Circ) (c : Universal.Cls) (p : c.Prog) : Finset (Fin (Universal.units c)) :=
  Finset.univ.filter fun u => p u ≠ Universal.pad C c u

/-- `C`'s wrong units where the program is `pad C c`'s are wrong units of the program. -/
theorem wrongAgree_le (C : Universal.Circ) (c : Universal.Cls) (hC : Universal.fits C c) (p : c.Prog)
    (x : ℕ → ℕ → Bool) (y : ℕ → List Bool) :
    wrongAgree C c p x y ≤ wrong (Universal.run c p) (Universal.units c) x y := by
  have hpc := pad_correct C c hC
  unfold wrongAgree wrong
  apply Finset.card_le_card
  intro i hi
  simp only [Finset.mem_filter, Finset.mem_range] at hi ⊢
  obtain ⟨hi, hne, hag⟩ := hi
  have hiN : i < c.N := lt_of_lt_of_le hi hpc.1
  refine ⟨hiN, ?_⟩
  have hrun : Universal.run c p i (x i) = Universal.run c (Universal.pad C c) i (x i) := by
    rw [Universal.run, Universal.run, dite_eq_left_of_eq_true (eq_true hiN), dite_eq_left_of_eq_true (eq_true hiN), hag hiN]
  rw [hrun, hpc.2.1 i hi]
  exact hne

/-- **Every wrong unit of `C` is one where the program is `pad C c`'s, or a forged unit.** -/
theorem wrong_le_wrongAgree_add (C : Universal.Circ) (c : Universal.Cls) (p : c.Prog)
    (x : ℕ → ℕ → Bool) (y : ℕ → List Bool) :
    wrong (Universal.eval C) (Universal.size C) x y ≤ wrongAgree C c p x y + (forged C c p).card := by
  classical
  unfold wrong wrongAgree
  have h1 : (Finset.range (Universal.size C)).filter (fun i => Universal.eval C i (x i) ≠ y i) ⊆
      (Finset.range (Universal.size C)).filter
        (fun i => Universal.eval C i (x i) ≠ y i ∧ agreesAt p (Universal.pad C c) i) ∪
      (forged C c p).map Fin.valEmbedding := by
    intro i hi
    simp only [Finset.mem_filter, Finset.mem_range] at hi
    by_cases hag : agreesAt p (Universal.pad C c) i
    · exact Finset.mem_union_left _ (Finset.mem_filter.2 ⟨Finset.mem_range.2 hi.1, hi.2, hag⟩)
    · apply Finset.mem_union_right
      simp only [agreesAt, not_forall] at hag
      obtain ⟨h, hne⟩ := hag
      exact Finset.mem_map.2 ⟨⟨i, h⟩, by unfold forged; exact Finset.mem_filter.2 ⟨Finset.mem_univ _, hne⟩, rfl⟩
  calc _ ≤ _ := Finset.card_le_card h1
    _ ≤ _ := Finset.card_union_le _ _
    _ = _ := by rw [Finset.card_map]

/-! ## The registered check at some reads -/

section Sites

variable {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}
  {lay : Binding.Layout P plan tab decode} {rs : Binding.HmRows lay}
  {rd : Registered.Reads lay rs} {rt : Flock.Registered.Port} {L : ℕ}

/-- The registered reads `rd` at the commit strings `A` only. -/
def readsOn (rd : Registered.Reads lay rs) (A : lay.Pos → Prop) : Registered.Reads lay rs where
  isRead p := rd.isRead p ∧ A p
  read_row p h := rd.read_row p h.1
  port := rd.port
  readAt := rd.readAt
  wv := rd.wv
  ws := rd.ws
  wpath := rd.wpath
  committed p h := rd.committed p h.1

open Classical in
/-- **The program's opening checked at `A`**: the session's leaves and paths at the program's sites in `A`, the
registrant's elsewhere. -/
noncomputable def openingOn (ps : ProgSites rd rt L) (A : lay.Pos → Prop) (R : Reg) (path : lay.Pos → ByteArray) :
    ProgOpening L :=
  ⟨rt, fun i => if A (ps.site i) then ps.leaves R i else ps.regLeaves i,
    fun i => if A (ps.site i) then path (ps.site i) else rd.wpath (ps.site i)⟩

/-- Where the registered check holds at `A`, `openingOn` opens the registered root. -/
theorem opens_on (ps : ProgSites rd rt L) {A : lay.Pos → Prop} {R : Reg} {path : lay.Pos → ByteArray}
    (hc : (readsOn rd A).Checked R path) :
    (progCommit L).Opens rt (openingOn ps A R path).leaf (openingOn ps A R path) := by
  refine ⟨rfl, rfl, fun i => ?_⟩
  by_cases hA : A (ps.site i)
  · have h := hc _ ⟨ps.isRead i, hA⟩
    change Flock.Registered.opensLeaf (rd.port (ps.site i)) _ (rd.readAt (ps.site i)) _ = true at h
    rw [ps.port i, ps.readAt i] at h
    simp only [openingOn, hA, ↓reduceIte]
    exact h
  · have h := rd.committed _ (ps.isRead i)
    rw [ps.port i, ps.readAt i] at h
    simp only [openingOn, hA, ↓reduceIte]
    exact h

/-- **The step to the program at the drawn units**: on every accepted outcome `o` of the session `s` against `τ` whose
leaves at the program's sites in `A o` are the registrant's, the program the committed values hold at the registered
sites (`progOf (s.vals τ o)`) is `p` at every drawn unit (`drawn o`), or `valPair o` is a SHA-512 collision. -/
def LeafBindsOn (ps : ProgSites rd rt L) (s : UnitSession P Reg) (τ : Strategy s.G) {c : Universal.Cls}
    (progOf : (Fin C.N → Bool) → c.Prog) (p : c.Prog) (A : s.Out → lay.Pos → Prop)
    (drawn : s.Out → Finset (Fin (Universal.units c))) (valPair : s.Out → List UInt8 × List UInt8) : Prop :=
  ∀ o, s.Accepts o → (∀ i, A o (ps.site i) → ps.leaves (s.regOf o) i = ps.regLeaves i) →
    (∀ u ∈ drawn o, progOf (s.vals τ o) u = p u) ∨ Binding.Collides (valPair o)

open Classical in
/-- **The binding finder's pair, checked at `A`**: the step to the program's (`valPair o`) where the session's leaves at
the checked sites are the registrant's, else the program commitment's extractor on `openingOn` and the registrant's
opening. -/
noncomputable def progPairOn (ps : ProgSites rd rt L) (s : UnitSession P Reg) (A : s.Out → lay.Pos → Prop)
    (path : s.Out → lay.Pos → ByteArray) (valPair : s.Out → List UInt8 × List UInt8) (o : s.Out) :
    List UInt8 × List UInt8 :=
  if ∀ i, A o (ps.site i) → ps.leaves (s.regOf o) i = ps.regLeaves i then valPair o
  else (progCommit L).ext (openingOn ps (A o) (s.regOf o) (path o)) ps.regOpening

end Sites

/-- **The law misses the forged units**: accepting while no drawn unit is in `F o` and `F o` has at least `K` units has
probability at most `miss(K)`. For `F` read off the committed values, which the prover fixes before the draw, this is
the law's miss at a set fixed before the draw. -/
def ForgedMiss {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} (s : UnitSession P Reg)
    (τ : Strategy s.G) (drawn : s.Out → Finset (Fin n)) (F : s.Out → Finset (Fin n)) (K : ℕ) : Prop :=
  prob (fun o => s.Accepts o ∧ (∀ u ∈ drawn o, u ∉ F o) ∧ K ≤ (F o).card) s.G τ ≤ s.Lm.miss K

/-- **Protocol 2 is sound at the universal unit, with the registered check at the drawn units.** As
`private_circuit_sound`, except that on every accepted outcome `o` the registered check holds at the reads `A o` only
(`hchk`, at `readsOn rd (A o)`: `ZkReg.checked_of_accepts` at the outcome's drawn instances), and the step to the program
holds at the drawn units `drawn o` where the leaves at `A o` are the registrant's (`LeafBindsOn`).

Then for every live prover `τ` whose knowledge finders meet `cr/sha-512` (`hCR`):

* accepting with at least `K₀` units of `C` wrong at the committed values' claimed inputs and outputs where the committed
  program is `pad C c`'s (`wrongAgree`), or with a drawn unit's committed program other than `pad C c`'s, or with a
  program leaf at a checked site other than `pad C c`'s registered one, has probability at most
  `miss(K₀) + ε_ks + δ_link + q²/2^513`, the last under `SHA512CRStrict` for the finder that outputs `progPairOn` (`hBind`);
* where the law misses the forged units at `K₁` (`ForgedMiss` at `forged`), accepting with at least `K₀ + K₁` units of
  `C` wrong at the claimed inputs and outputs has probability at most `miss(K₀) + miss(K₁) + ε_ks + δ_link + q²/2^513`. -/
theorem private_circuit_sound_drawn
    {UCc : Audit.Circuit} {c : Universal.Cls} (P : Audit.Partition UCc (Universal.units c))
    (C : Universal.Circ) (hC : Universal.fits C c)
    (progOf : (Fin UCc.N → Bool) → c.Prog) (xOf : (Fin UCc.N → Bool) → ℕ → ℕ → Bool)
    (yOf : (Fin UCc.N → Bool) → ℕ → List Bool) (hunit : UnitWrong P progOf xOf yOf)
    {Reg : Type} {mPts : ℕ} {plan : Finset (Fin (Universal.units c)) → Reg → List (TabSpec mPts)}
    {tab : (S : Finset (Fin (Universal.units c))) → (R : Reg) → Fin (Universal.units c) → Fin (plan S R).length}
    {decode : Audit.Partition.DecoderB (C := UCc) (Reg := Reg) plan tab}
    {lay : Binding.Layout P plan tab decode} {rs : Binding.HmRows lay} (rd : Registered.Reads lay rs)
    {rt : Flock.Registered.Port} {L : ℕ} (ps : ProgSites rd rt L)
    (row : c.Prog → Fin L → lay.Val) (hreg : ∀ i, rd.wv (ps.site i) = row (Universal.pad C c) i)
    (s : UnitSession P Reg) (hS : ZkSessionSound s)
    (hA3 : Assumptions.UniformRandomBytes s.Lb s.src) (hA6 : s.A6)
    (A : s.Out → lay.Pos → Prop) (path : s.Out → lay.Pos → ByteArray)
    (hchk : ∀ o, s.Accepts o → (readsOn rd (A o)).Checked (s.regOf o) (path o))
    (τ : Strategy s.G) (hCR : s.CR τ) (drawn : s.Out → Finset (Fin (Universal.units c)))
    (valPair : s.Out → List UInt8 × List UInt8)
    (hstep : LeafBindsOn ps s τ progOf (Universal.pad C c) A drawn valPair)
    (K₀ : ℕ) (cost : s.Out → ℝ) (q : ℝ) (hcost : ∀ o, cost o ≤ q)
    (hBind : Assumptions.SHA512CRStrict H512 s.G τ (fun o => some (progPairOn ps s A path valPair o)) cost q) :
    prob (fun o => s.Accepts o ∧
        (K₀ ≤ wrongAgree C c (progOf (s.vals τ o)) (xOf (s.vals τ o)) (yOf (s.vals τ o)) ∨
          (∃ u ∈ drawn o, progOf (s.vals τ o) u ≠ Universal.pad C c u) ∨
          ∃ i, A o (ps.site i) ∧ lay.regLeaf (s.regOf o) (ps.site i) ≠
            (ExplicitLeaf.hm96 rs.enc rs.enc_inj).leaf (row (Universal.pad C c) i) (rd.ws (ps.site i)))) s.G τ
      ≤ s.Lm.miss K₀ + s.εks τ + s.δlink τ + ENNReal.ofReal (q ^ 2 / 2 ^ (513 : ℝ)) ∧
    ∀ K₁, ForgedMiss s τ drawn (fun o => forged C c (progOf (s.vals τ o))) K₁ →
      prob (fun o => s.Accepts o ∧
          K₀ + K₁ ≤ wrong (Universal.eval C) (Universal.size C) (xOf (s.vals τ o)) (yOf (s.vals τ o))) s.G τ
        ≤ s.Lm.miss K₀ + s.Lm.miss K₁ + s.εks τ + s.δlink τ + ENNReal.ofReal (q ^ 2 / 2 ^ (513 : ℝ)) := by
  classical
  have hcoll : ∀ o, s.Accepts o → ((∃ i, A o (ps.site i) ∧ ps.leaves (s.regOf o) i ≠ ps.regLeaves i) ∨
      ∃ u ∈ drawn o, progOf (s.vals τ o) u ≠ Universal.pad C c u) →
      Binding.Collides (progPairOn ps s A path valPair o) := by
    intro o ho h
    by_cases hl : ∀ i, A o (ps.site i) → ps.leaves (s.regOf o) i = ps.regLeaves i
    · rw [progPairOn, ite_eq_left_of_eq_true _ _ (eq_true hl)]
      rcases h with ⟨i, hi, hne⟩ | ⟨u, hu, hne⟩
      · exact absurd (hl i hi) hne
      · exact (hstep o ho hl).resolve_left fun h' => hne (h' u hu)
    · rw [progPairOn, ite_eq_right_of_eq_false _ _ (eq_false hl)]
      refine progCommit_binding L rt _ _ _ _ (opens_on ps (hchk o ho)) ps.opens_reg fun heq => hl fun i hi => ?_
      have h' := congrFun heq i
      simp only [openingOn, hi, ↓reduceIte] at h'
      exact h'
  have hc : ∀ o, Binding.Collides (progPairOn ps s A path valPair o) →
      ∃ x y, some (progPairOn ps s A path valPair o) = some (x, y) ∧ x ≠ y ∧ H512 x = H512 y :=
    fun o h => ⟨_, _, congrArg some Prod.mk.eta.symm, h.1, h.2⟩
  have hfin : prob (fun o => ∃ x y, some (progPairOn ps s A path valPair o) = some (x, y) ∧ x ≠ y ∧
      H512 x = H512 y) s.G τ ≠ ⊤ := ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _)
  have hCRb := (ENNReal.le_ofReal_iff_toReal_le hfin (by positivity)).2 (hBind hcost)
  have hW : ∀ o, wrongAgree C c (progOf (s.vals τ o)) (xOf (s.vals τ o)) (yOf (s.vals τ o)) ≤
      (P.wrong (s.vals τ o)).card := fun o => (wrongAgree_le C c hC _ _ _).trans (hunit.card _).symm.le
  have hmain : prob (fun o => s.Accepts o ∧
        (K₀ ≤ wrongAgree C c (progOf (s.vals τ o)) (xOf (s.vals τ o)) (yOf (s.vals τ o)) ∨
          (∃ u ∈ drawn o, progOf (s.vals τ o) u ≠ Universal.pad C c u) ∨
          ∃ i, A o (ps.site i) ∧ lay.regLeaf (s.regOf o) (ps.site i) ≠
            (ExplicitLeaf.hm96 rs.enc rs.enc_inj).leaf (row (Universal.pad C c) i) (rd.ws (ps.site i)))) s.G τ
      ≤ s.Lm.miss K₀ + s.εks τ + s.δlink τ + ENNReal.ofReal (q ^ 2 / 2 ^ (513 : ℝ)) := by
    calc _ ≤ prob (fun o => (s.Accepts o ∧ K₀ ≤ (P.wrong (s.vals τ o)).card) ∨
          ∃ x y, some (progPairOn ps s A path valPair o) = some (x, y) ∧ x ≠ y ∧ H512 x = H512 y) s.G τ := by
          refine prob_mono (fun o ⟨ha, hq⟩ => ?_) s.G τ
          rcases hq with hw | hu | ⟨i, hi, hne⟩
          · exact Or.inl ⟨ha, hw.trans (hW o)⟩
          · exact Or.inr (hc o (hcoll o ha (Or.inr hu)))
          · refine Or.inr (hc o (hcoll o ha (Or.inl ⟨i, hi, fun hl => hne ?_⟩)))
            simp only [ProgSites.leaves, ProgSites.regLeaves, Reads.leaf, hreg] at hl
            exact hl
      _ ≤ _ := prob_or_le _ _ s.G τ
      _ ≤ _ := add_le_add (hS hA3 hA6 τ hCR K₀) hCRb
  refine ⟨hmain, fun K₁ hF => ?_⟩
  calc _ ≤ prob (fun o => (s.Accepts o ∧
        (K₀ ≤ wrongAgree C c (progOf (s.vals τ o)) (xOf (s.vals τ o)) (yOf (s.vals τ o)) ∨
          (∃ u ∈ drawn o, progOf (s.vals τ o) u ≠ Universal.pad C c u) ∨
          ∃ i, A o (ps.site i) ∧ lay.regLeaf (s.regOf o) (ps.site i) ≠
            (ExplicitLeaf.hm96 rs.enc rs.enc_inj).leaf (row (Universal.pad C c) i) (rd.ws (ps.site i)))) ∨
        (s.Accepts o ∧ (∀ u ∈ drawn o, u ∉ forged C c (progOf (s.vals τ o))) ∧
          K₁ ≤ (forged C c (progOf (s.vals τ o))).card)) s.G τ := by
        refine prob_mono (fun o ⟨ha, hw⟩ => ?_) s.G τ
        by_cases hd : ∃ u ∈ drawn o, progOf (s.vals τ o) u ≠ Universal.pad C c u
        · exact Or.inl ⟨ha, Or.inr (Or.inl hd)⟩
        by_cases hK : K₀ ≤ wrongAgree C c (progOf (s.vals τ o)) (xOf (s.vals τ o)) (yOf (s.vals τ o))
        · exact Or.inl ⟨ha, Or.inl hK⟩
        refine Or.inr ⟨ha, fun u hu hmem => hd ⟨u, hu, by simpa [forged] using hmem⟩, ?_⟩
        have := wrong_le_wrongAgree_add C c (progOf (s.vals τ o)) (xOf (s.vals τ o)) (yOf (s.vals τ o))
        omega
    _ ≤ _ := prob_or_le _ _ s.G τ
    _ ≤ _ := add_le_add hmain hF
    _ = _ := by ring

end FlockSoundness.Discharge.PrivateCircuit
