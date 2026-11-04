import Mathlib.Data.Finset.Card
import Proofs.Properties.NetworkScan.Lowered

/-!
# A circuit file's source columns

What the circuit count's proof reads off a circuit file that Bits lays out as its template
(`SampledProofs.Bits.check`, `fileOk`): one template instance a VU, the unit's input columns its input words, each
leaf port one 16-bit word, each cut port wired by a leaf cut, and as many ports as its Definition's parameters' words
(`FileOk`). An input gate of the key program a unit reads is a source gate `2 + u·B + b` whose key `b` is a column's
(`input_src`), and that column is a bit below 16 of a port, the input bit `inGate` names, or a column no port holds
(`col_slot`, `unheld`).
-/

namespace Properties.NetworkCertifier.Counts

open SampledProofs.Bits
open Properties.NetworkScan.Lowered

/-! ## The unit's ports -/

/-- Ports of widths `ws` laid from column `a`, each after the last. -/
def colsFrom : ℕ → List ℕ → List (ℕ × ℕ)
  | _, [] => []
  | a, w :: ws => (a, w) :: colsFrom (a + w) ws

theorem colsFrom_snd : ∀ (a : ℕ) (ws : List ℕ), (colsFrom a ws).map Prod.snd = ws
  | _, [] => rfl
  | a, w :: ws => by simp only [colsFrom, List.map_cons, colsFrom_snd (a + w) ws]

theorem colsFrom_length (a : ℕ) (ws : List ℕ) : (colsFrom a ws).length = ws.length := by
  have := congrArg List.length (colsFrom_snd a ws)
  rwa [List.length_map] at this

theorem colsFrom_getElem? : ∀ (a : ℕ) (ws : List ℕ) (p : ℕ),
    (colsFrom a ws)[p]? = ws[p]?.map fun w => (a + (ws.take p).sum, w)
  | _, [], _ => rfl
  | a, w :: ws, 0 => by simp [colsFrom]
  | a, w :: ws, p + 1 => by
    simp only [colsFrom, List.getElem?_cons_succ, colsFrom_getElem? (a + w) ws p, List.take_succ_cons,
      List.sum_cons, Nat.add_assoc]

theorem inner_foldl : ∀ (ws : List ℕ) (a : ℕ) (out : Array (ℕ × ℕ)),
    (ws.foldl (fun (s : Array (ℕ × ℕ) × ℕ) w => (s.1.push (s.2, w), s.2 + w)) (out, a)).1 =
      out ++ (colsFrom a ws).toArray
  | [], a, out => by simp [colsFrom]
  | w :: ws, a, out => by
    simp only [List.foldl_cons]
    rw [inner_foldl ws (a + w) (out.push (a, w))]
    apply Array.toList_inj.1
    simp [colsFrom]

theorem outer_foldl : ∀ (gs : List Flock.Group) (out : Array (ℕ × ℕ)),
    gs.foldl (fun (out : Array (ℕ × ℕ)) (g : Flock.Group) =>
        (g.bits.toList.foldl (fun (s : Array (ℕ × ℕ) × ℕ) w => (s.1.push (s.2, w), s.2 + w))
          (out, g.col * Flock.WORD)).1) out =
      out ++ (gs.flatMap fun g => colsFrom (g.col * Flock.WORD) g.bits.toList).toArray
  | [], out => by simp
  | g :: gs, out => by
    simp only [List.foldl_cons]
    rw [inner_foldl, outer_foldl gs]
    apply Array.toList_inj.1
    simp

/-- **The unit's ports**: each group's ports' widths laid from its first word's column, group by group. -/
theorem portCols_eq (gs : Array Flock.Group) :
    Flock.portCols gs = (gs.toList.flatMap fun g => colsFrom (g.col * Flock.WORD) g.bits.toList).toArray := by
  have h : Flock.portCols gs = gs.foldl (fun (out : Array (ℕ × ℕ)) (g : Flock.Group) =>
      (g.bits.foldl (fun (s : Array (ℕ × ℕ) × ℕ) w => (s.1.push (s.2, w), s.2 + w))
        (out, g.col * Flock.WORD)).1) #[] := by
    simp [Flock.portCols]
  rw [h]
  simp only [← Array.foldl_toList]
  rw [outer_foldl]
  simp

theorem toList_cons_of_getD {gs : Array Flock.Group} {p : ℕ} (hp : p < (gs.getD 0 default).bits.size) :
    ∃ rest, gs.toList = gs.getD 0 default :: rest := by
  rcases h : gs.toList with _ | ⟨g, rest⟩
  · have e : gs.getD 0 default = default := by
      rw [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, h]
      rfl
    rw [e] at hp
    exact absurd hp (Nat.not_lt_zero _)
  · refine ⟨rest, ?_⟩
    rw [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, h]
    rfl

/-- **A port of the first group**: its column and its width, the group's bit `p`. -/
theorem portCols_getElem? {gs : Array Flock.Group} {g : Flock.Group} {rest : List Flock.Group}
    (hg : gs.toList = g :: rest) {p : ℕ} (hp : p < g.bits.size) :
    (Flock.portCols gs)[p]? = some (g.col * Flock.WORD + (g.bits.toList.take p).sum, g.bits[p]) := by
  rw [portCols_eq, hg, List.flatMap_cons, List.getElem?_toArray,
    List.getElem?_append_left (by rw [colsFrom_length]; simpa using hp), colsFrom_getElem?]
  simp [hp]

/-- **A leaf port is one word** where the first group's ports are each 16 bits. -/
theorem leaf_width {gs : Array Flock.Group} (h : (gs.getD 0 default).bits.any (· != 16) = false) {p : ℕ}
    (hp : p < (gs.getD 0 default).bits.size) : ((Flock.portCols gs).getD p (0, 0)).2 = 16 := by
  obtain ⟨rest, hg⟩ := toList_cons_of_getD hp
  rw [Array.getD_eq_getD_getElem?, portCols_getElem? hg hp, Option.getD_some]
  have := Array.any_eq_false.1 h p hp
  simpa using this

/-! ## The leaf cuts -/

theorem foldl_getD_of_ne {α : Type} (f : Std.HashMap (ℕ × ℕ) ℕ → α → Std.HashMap (ℕ × ℕ) ℕ) (keyOf : α → ℕ × ℕ)
    (K : ℕ × ℕ) (hf : ∀ m a, keyOf a ≠ K → (f m a).getD K 0 = m.getD K 0) :
    ∀ (l : List α) (m : Std.HashMap (ℕ × ℕ) ℕ), (∀ a ∈ l, keyOf a ≠ K) → (l.foldl f m).getD K 0 = m.getD K 0
  | [], _, _ => rfl
  | a :: l, m, h => by
    rw [List.foldl_cons, foldl_getD_of_ne f keyOf K hf l (f m a) (fun b hb => h b (List.mem_cons_of_mem a hb)),
      hf m a (h a List.mem_cons_self)]

theorem foldl_contains {α : Type} (f : Std.HashMap (ℕ × ℕ) ℕ → α → Std.HashMap (ℕ × ℕ) ℕ) (keyOf : α → ℕ × ℕ)
    (K : ℕ × ℕ) (hf : ∀ m a, (m.contains K = true ∨ keyOf a = K) → (f m a).contains K = true) :
    ∀ (l : List α) (m : Std.HashMap (ℕ × ℕ) ℕ), (m.contains K = true ∨ ∃ a ∈ l, keyOf a = K) →
      (l.foldl f m).contains K = true
  | [], _, h => by
    rcases h with h | ⟨a, ha, -⟩
    · exact h
    · cases ha
  | a :: l, m, h => by
    rw [List.foldl_cons]
    refine foldl_contains f keyOf K hf l (f m a) ?_
    rcases h with h | ⟨b, hb, hbK⟩
    · exact .inl (hf m a (.inl h))
    · rcases List.mem_cons.1 hb with hb | hb
      · subst hb
        exact .inl (hf m _ (.inr hbK))
      · exact .inr ⟨b, hb, hbK⟩

/-- **A wired cut port has its first leaf cut**: where a leaf cut names copy `k`'s port `p`, the index holds one. -/
theorem cuts_some {c : Flock.Circuit} {K : ℕ × ℕ} (h : ∃ e ∈ c.leafCuts.toList, (e.1, e.2.1) = K) :
    ∃ w, (Index.of c).cuts[K]? = some w := by
  have hc : (Index.of c).cuts.contains K = true := by
    unfold Index.of
    dsimp only
    rw [← Array.foldl_toList]
    refine foldl_contains _ (fun e => (e.1, e.2.1)) K ?_ _ _ (.inr h)
    rintro m ⟨k', p', w'⟩ hm
    dsimp only at hm ⊢
    split
    · rename_i hc
      rcases hm with hm | rfl
      · exact hm
      · exact hc
    · rw [Std.HashMap.contains_insert]
      rcases hm with hm | rfl
      · simp [hm]
      · simp
  rw [Std.HashMap.contains_eq_isSome_getElem?] at hc
  exact Option.isSome_iff_exists.1 hc

/-! ## A file Bits lays out -/

/-- **What Bits' acceptance gives a circuit file** at its template's Definition `d`: one template instance a VU, the
unit net's input columns its input words, each leaf port one 16-bit word, each cut port of the instance its leaf cut,
and as many ports as `d`'s parameters' packed words. -/
structure FileOk (c : Flock.Circuit) (d : Flock.Program.Defn) : Prop where
  upv : c.upv = 1
  nIn : min (c.unit.inWords * Flock.WORD) c.unit.constPos = c.unit.inWords * Flock.WORD
  leaf : ∀ p < (c.unit.inGroups.getD 0 default).bits.size, ((Flock.portCols c.unit.inGroups).getD p (0, 0)).2 = 16
  cut : ∀ p, (c.unit.inGroups.getD 0 default).bits.size ≤ p → p < (Flock.portCols c.unit.inGroups).size →
    ∃ w, (Index.of c).cuts[(0, p)]? = some w
  ports : (Flock.portCols c.unit.inGroups).size = (d.params.toList.map fun q => wordsOf q.2).sum

theorem ite_some_eq_none {α : Type} {c : Prop} [Decidable c] {x : α} {y : Option α}
    (h : (if c then some x else y) = none) : ¬ c ∧ y = none := by
  by_cases hc : c
  · rw [ite_eq_left hc] at h
    cases h
  · rw [ite_eq_right hc] at h
    exact ⟨hc, h⟩

theorem orElse_eq_none {α : Type} {a b : Option α} (h : (a <|> b) = none) : a = none ∧ b = none := by
  cases a with
  | none => exact ⟨rfl, h⟩
  | some x => injection h

theorem fileOk {c : Flock.Circuit} {P : Flock.Program.Prog} {key : String}
    (h : SampledProofs.Bits.check c P key = none) :
    ∃ d, P.defs.find? (·.key == key) = some d ∧ FileOk c d := by
  unfold SampledProofs.Bits.check at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true })
    nIn nComp gi go outs wired cut pc unwired at h
  obtain ⟨h1, h⟩ := ite_some_eq_none h
  obtain ⟨-, h⟩ := ite_some_eq_none h
  obtain ⟨h3, h⟩ := ite_some_eq_none h
  obtain ⟨-, h⟩ := ite_some_eq_none h
  obtain ⟨-, h⟩ := ite_some_eq_none h
  obtain ⟨-, h⟩ := ite_some_eq_none h
  obtain ⟨-, h⟩ := ite_some_eq_none h
  rcases hun : unwired with _ | ⟨k, p⟩
  swap
  · rw [hun] at h
    injection h
  rw [hun] at h
  dsimp only at h
  obtain ⟨-, h⟩ := ite_some_eq_none h
  obtain ⟨h10, h⟩ := ite_some_eq_none h
  obtain ⟨-, h⟩ := ite_some_eq_none h
  obtain ⟨-, h⟩ := ite_some_eq_none h
  obtain ⟨hX, hY⟩ := orElse_eq_none h
  have hupv : c.upv = 1 := by
    have hne := (ite_some_eq_none hY).1
    simpa using hne
  have hgi : (c.unit.inGroups.getD 0 default).bits.any (· != 16) = false := by
    simp only [Bool.or_eq_true, not_or, Bool.not_eq_true] at h3
    exact h3.1
  have hnIn : min (c.unit.inWords * Flock.WORD) c.unit.constPos = c.unit.inWords * Flock.WORD := by
    have h10' : nIn = c.unit.inWords * Flock.WORD := by simpa using h10
    exact h10'
  have hun' : ((List.range c.upv).flatMap fun k => (List.range (pc.size - cut)).map fun i => (k, cut + i)).find?
      (fun kp => wired.getD kp 0 != 1) = none := hun
  have e1 : cut = (c.unit.inGroups.getD 0 default).bits.size := rfl
  have e2 : pc.size = (Flock.portCols c.unit.inGroups).size := rfl
  have hcut : ∀ p, (c.unit.inGroups.getD 0 default).bits.size ≤ p → p < (Flock.portCols c.unit.inGroups).size →
      ∃ e ∈ c.leafCuts.toList, (e.1, e.2.1) = (0, p) := by
    intro p hp1 hp2
    have hmem : (0, p) ∈ (List.range c.upv).flatMap fun k =>
        (List.range (pc.size - cut)).map fun i => (k, cut + i) := by
      rw [List.mem_flatMap]
      refine ⟨0, List.mem_range.2 (by omega), List.mem_map.2 ⟨p - cut, List.mem_range.2 (by omega), ?_⟩⟩
      exact Prod.ext rfl (show cut + (p - cut) = p by omega)
    have hne := List.find?_eq_none.1 hun' (0, p) hmem
    have hw1 : wired.getD (0, p) 0 = 1 := by simpa using hne
    by_contra hno
    have hno' : ∀ e ∈ c.leafCuts.toList, (fun (e : ℕ × ℕ × ℕ) => (e.1, e.2.1)) e ≠ (0, p) :=
      fun e he heq => hno ⟨e, he, heq⟩
    have hw0 : wired.getD (0, p) 0 = 0 := by
      simp only [wired]
      rw [← Array.foldl_toList, foldl_getD_of_ne _ (fun (e : ℕ × ℕ × ℕ) => (e.1, e.2.1)) (0, p) ?_ _ _ hno',
        Std.HashMap.getD_empty]
      rintro m ⟨k', p', w'⟩ hk
      simp only [Std.HashMap.getD_insert]
      split
      · rename_i heq
        exact absurd (beq_iff_eq.1 heq) hk
      · rfl
    omega
  have hany : P.defs.any (·.key == key) = true := by simpa using h1
  rcases hf : P.defs.find? (·.key == key) with _ | d
  · exfalso
    obtain ⟨x, hx, hpx⟩ := Array.any_eq_true'.1 hany
    exact Array.find?_eq_none.1 hf x hx hpx
  refine ⟨d, hf, ?_⟩
  rw [hf] at hX
  dsimp only at hX
  obtain ⟨-, hX⟩ := ite_some_eq_none hX
  obtain ⟨-, hX⟩ := ite_some_eq_none hX
  obtain ⟨-, hX⟩ := ite_some_eq_none hX
  obtain ⟨-, hX⟩ := ite_some_eq_none hX
  obtain ⟨h5, -⟩ := ite_some_eq_none hX
  have hports : (Flock.portCols c.unit.inGroups).size = (d.params.toList.map fun q => wordsOf q.2).sum := by
    have h5' : pc.size = (d.params.toList.map fun q => wordsOf q.2).sum := by simpa using h5
    exact h5'
  exact ⟨hupv, hnIn, fun p hp => leaf_width hgi hp, fun p hp1 hp2 => cuts_some (hcut p hp1 hp2), hports⟩

/-- A file's unit sources: its input words' columns. -/
theorem srcs_eq {c : Flock.Circuit} {d : Flock.Program.Defn} (hF : FileOk c d) :
    srcs c = c.unit.inWords * Flock.WORD := by
  show c.upv * min (c.unit.inWords * Flock.WORD) c.unit.constPos = _
  rw [hF.upv, Nat.one_mul, hF.nIn]

/-! ## An input gate's column -/

/-- **A row's reads are its unit's columns** (`wireOf`). -/
theorem row_cols (c : Flock.Circuit) {n g : ℕ} {a b : List ℕ} (h : (keyProg c n).row? g = some (a, b)) :
    (g - (keyProg c n).inputs) / (keyProg c n).perUnit < n ∧
      ∀ x ∈ a ++ b, ∃ col, x = wireOf c n ((g - (keyProg c n).inputs) / (keyProg c n).perUnit) col := by
  have h1 := inputs_le_of_row? _ h
  have h2 := lt_size_of_row? _ h
  have hs := keyProg_size_rows c n
  unfold KeyProg.size at h2
  have hr : g - (keyProg c n).inputs < n * (keyProg c n).perUnit := by omega
  have hP : 0 < (keyProg c n).perUnit :=
    Nat.pos_of_ne_zero fun h0 => by rw [h0, Nat.mul_zero] at hr; omega
  have hu : (g - (keyProg c n).inputs) / (keyProg c n).perUnit < n := (Nat.div_lt_iff_lt_mul hP).2 hr
  have hi := Nat.mod_lt (g - (keyProg c n).inputs) hP
  have hdm : (g - (keyProg c n).inputs) / (keyProg c n).perUnit * (keyProg c n).perUnit +
      (g - (keyProg c n).inputs) % (keyProg c n).perUnit = g - (keyProg c n).inputs := by
    rw [Nat.mul_comm]; exact Nat.div_add_mod _ _
  have hrow := keyProg_row? c hu hi
  rw [show (keyProg c n).inputs + ((g - (keyProg c n).inputs) / (keyProg c n).perUnit * (keyProg c n).perUnit +
      (g - (keyProg c n).inputs) % (keyProg c n).perUnit) = g by omega, h] at hrow
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hrow)
  refine ⟨hu, fun x hx => ?_⟩
  rw [List.mem_append, List.mem_map, List.mem_map] at hx
  obtain ⟨col, -, rfl⟩ | ⟨col, -, rfl⟩ := hx
  · exact ⟨col, rfl⟩
  · exact ⟨col, rfl⟩

/-- **A source gate's column**: where a unit's column is wired to a source gate, the column is a source column with a
key, and the gate is the unit's at that key. -/
theorem wireOf_src {c : Flock.Circuit} {n u col : ℕ} (h2 : 2 ≤ wireOf c n u col)
    (hx : wireOf c n u col < 2 + n * srcs c) :
    col < srcs c ∧ ∃ b, keyIdx c (srcs c) col = some b ∧ wireOf c n u col = 2 + u * srcs c + b := by
  unfold srcs at hx ⊢
  unfold wireOf at h2 hx ⊢
  by_cases h1 : col < ((Rows.ofNet c.unit).stack c.upv).nIn
  · have hkey : ((Array.range ((Rows.ofNet c.unit).stack c.upv).nIn).map
        (keyIdx c ((Rows.ofNet c.unit).stack c.upv).nIn)).getD col none =
        keyIdx c ((Rows.ofNet c.unit).stack c.upv).nIn col := by
      simp [h1]
    rw [ite_eq_left h1, hkey] at h2 ⊢
    refine ⟨h1, ?_⟩
    rcases hk : keyIdx c ((Rows.ofNet c.unit).stack c.upv).nIn col with _ | b
    · exfalso
      simp only [hk] at h2
      omega
    · exact ⟨b, rfl, rfl⟩
  · exfalso
    rw [ite_eq_right h1] at h2 hx
    by_cases h3 : col < ((Rows.ofNet c.unit).stack c.upv).nIn + ((Rows.ofNet c.unit).stack c.upv).nComp
    · rw [ite_eq_left h3] at hx
      omega
    · rw [ite_eq_right h3] at h2
      omega

/-- **An input gate a unit reads is a source gate at a column's key**: of `keyProg c n` at its constants, a lowered
`.input` is unit `u`'s source gate `2 + u·B + b`, `b` the key of a source column `col`. -/
theorem input_src {c : Flock.Circuit} {n x : ℕ} (hx : (prog (keyProg c n) (consts c n)).1[x]? = some .input) :
    ∃ u < n, ∃ col < srcs c, ∃ b, keyIdx c (srcs c) col = some b ∧ x = 2 + u * srcs c + b := by
  obtain ⟨hxi, hκ⟩ := (prog_input_iff (keyProg c n) (consts c n) x).1 hx
  have hx0 : x ≠ 0 := by
    rintro rfl
    simp [consts] at hκ
  have hx1 : x ≠ 1 := by
    rintro rfl
    simp [consts] at hκ
  have hown : owner (keyProg c n) x ≠ none := by
    intro hn
    simp [consts, hn, hxi, hx0] at hκ
  obtain ⟨u0, hu0⟩ := Option.ne_none_iff_exists'.1 hown
  have hp := List.find?_some hu0
  rw [List.any_eq_true] at hp
  obtain ⟨g, -, hpg⟩ := hp
  simp only [Bool.and_eq_true] at hpg
  obtain ⟨-, hcont⟩ := hpg
  unfold args at hcont
  split at hcont
  · rename_i a b hrow
    have hmem : x ∈ a ++ b := by simpa using hcont
    obtain ⟨hu, hcols⟩ := row_cols c hrow
    generalize (g - (keyProg c n).inputs) / (keyProg c n).perUnit = u at hu hcols
    obtain ⟨col, rfl⟩ := hcols x hmem
    rw [Properties.NetworkScan.Lowered.keyProg_inputs] at hxi
    obtain ⟨hcol, b', hb', he⟩ := wireOf_src (by omega) hxi
    exact ⟨u, hu, col, hcol, b', hb', he⟩
  · simp at hcont

/-- **Input padding**: the unit's input columns no port holds, searched as `srcKey` searches them. -/
def unheld (c : Flock.Circuit) : Finset ℕ :=
  ((List.range (min (c.unit.inWords * Flock.WORD) c.unit.constPos)).filter fun col =>
    ((List.range (Flock.portCols c.unit.inGroups).size).find? fun p =>
      decide (((Flock.portCols c.unit.inGroups).getD p (0, 0)).1 ≤ col ∧
        col < ((Flock.portCols c.unit.inGroups).getD p (0, 0)).1 +
          ((Flock.portCols c.unit.inGroups).getD p (0, 0)).2)).isNone).toFinset

/-- **A keyed column is a port bit or padding**: in a file Bits lays out, a source column with a key is bit `t < 16` of
an input port `p`, whose input bit at unit `u` is the unit's source gate at the key (`inGate`), or no port's
(`unheld`). -/
theorem col_slot {c : Flock.Circuit} {d : Flock.Program.Defn} (hF : FileOk c d) {n u col b : ℕ} (hu : u < n)
    (hcol : col < srcs c) (hb : keyIdx c (srcs c) col = some b) :
    (∃ p < (Flock.portCols c.unit.inGroups).size, ∃ t < 16, inGate c n u 0 p t = some (2 + u * srcs c + b)) ∨
      col ∈ unheld c := by
  have hlt : col < c.unit.inWords * Flock.WORD := srcs_eq hF ▸ hcol
  have hz : srcKey c (Index.of c) col ≠ .zero := by
    intro h0
    unfold keyIdx at hb
    dsimp only at hb
    rw [ite_eq_left h0] at hb
    cases hb
  unfold srcKey at hz
  dsimp only at hz
  rw [Nat.mod_eq_of_lt hlt, Nat.div_eq_of_lt hlt] at hz
  split at hz
  · rename_i p hp
    have hpq := List.find?_some hp
    have hpr := List.mem_range.1 (List.mem_of_find?_eq_some hp)
    simp only [decide_eq_true_eq] at hpq
    obtain ⟨hp1, hp2⟩ := hpq
    have ht : col - ((Flock.portCols c.unit.inGroups).getD p (0, 0)).1 < 16 := by
      by_cases hgi : p < (c.unit.inGroups.getD 0 default).bits.size
      · have := hF.leaf p hgi
        omega
      · obtain ⟨w, hw⟩ := hF.cut p (Nat.le_of_not_lt hgi) hpr
        by_contra h16
        apply hz
        unfold srcAt
        rw [ite_eq_right hgi, hw]
        unfold msgAt
        dsimp only
        rw [ite_eq_right]
        simp only [Bool.false_or, Bool.and_eq_true, decide_eq_true_eq, not_and]
        intro h
        omega
    left
    refine ⟨p, hpr, _, ht, ?_⟩
    unfold inGate
    dsimp only
    rw [ite_eq_left]
    · have e : inCol c 0 p (col - ((Flock.portCols c.unit.inGroups).getD p (0, 0)).1) = col := by
        unfold inCol
        show 0 * _ + (((Flock.portCols c.unit.inGroups).getD p (0, 0)).1 + _) = col
        omega
      rw [e]
      show Option.map _ (keyIdx c (srcs c) col) = _
      rw [hb]
      rfl
    · simp only [Bool.and_eq_true, decide_eq_true_eq]
      refine ⟨⟨⟨hu, by rw [hF.upv]; exact Nat.one_pos⟩, hpr⟩, ?_⟩
      show _ < ((Flock.portCols c.unit.inGroups).getD p (0, 0)).2
      omega
  · rename_i hnone
    right
    exact List.mem_toFinset.2 (List.mem_filter.2
      ⟨List.mem_range.2 (by rw [hF.nIn]; exact hlt), Option.isNone_iff_eq_none.2 hnone⟩)

end Properties.NetworkCertifier.Counts
