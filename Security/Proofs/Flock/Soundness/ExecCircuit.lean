import Proofs.Flock.Soundness.ExecParse

/-!
# What `HmRow.parse` accepted

`parse_facts`: a circuit file `HmRow.parse` accepted has only nets `Net.ofRows` built (its text nets, and a typed flat
class's unit net given first as `pre`, `parse_facts_pre`) and lookup slots (`Lookup.build`), `sha512/row/v1` ports of
whole SHA-512 blocks (`rowPorts_words`), the unit, `sha512x3` and `hm96` nets and every range's net among its nets, passed
`HmRow.check`, and its `sha512x3` and `hm96` nets are the verifier's (`HmNets.check`), entered only by the rows' wires
(`wires_prefix`). The walk lifts each level's join points into local definitions (`extract_lets`), so a branch point is one
goal: the branches meet at the join point.
-/

open Flock FlockLevel3

namespace FlockSoundness

/-! ## Loops and maps in `Except` -/

/-- A `while` loop in `Except` keeps an invariant when each step keeps it and every continuing step lowers a measure. -/
theorem loop_inv {β : Type} (P : β → Prop) (μ : β → ℕ) {f : Unit → β → Except String (ForInStep β)}
    (hf : ∀ b, P b → ∀ s, f () b = .ok s →
      (∃ b', s = .done b' ∧ P b') ∨ ∃ b', s = .yield b' ∧ P b' ∧ μ b' < μ b) :
    ∀ n b v, μ b ≤ n → P b → forIn Lean.Loop.mk b f = .ok v → P v := by
  intro n
  induction n with
  | zero =>
    intro b v hμ hb h
    change Lean.Loop.forIn Lean.Loop.mk b f = .ok v at h
    rw [Lean.Loop.forIn_eq_of_monadTail] at h
    obtain ⟨s, hs, h⟩ := except_bind_ok h
    rcases hf b hb s hs with ⟨b', rfl, hb'⟩ | ⟨b', rfl, -, hlt⟩
    · cases h; exact hb'
    · omega
  | succ n ih =>
    intro b v hμ hb h
    change Lean.Loop.forIn Lean.Loop.mk b f = .ok v at h
    rw [Lean.Loop.forIn_eq_of_monadTail] at h
    obtain ⟨s, hs, h⟩ := except_bind_ok h
    rcases hf b hb s hs with ⟨b', rfl, hb'⟩ | ⟨b', rfl, hb', hlt⟩
    · cases h; exact hb'
    · exact ih b' v (by omega) hb' h

/-- A `for` loop in `Except` keeps an invariant its body keeps. -/
theorem forIn_except_inv {α β : Type} (P : β → Prop) {f : α → β → Except String (ForInStep β)} :
    ∀ (l : List α), (∀ a ∈ l, ∀ b, P b → ∀ s, f a b = .ok s → ∃ b', (s = .yield b' ∨ s = .done b') ∧ P b') →
      ∀ b v, P b → forIn l b f = .ok v → P v
  | [], _, b, v, hb, h => by rw [List.forIn_nil] at h; cases h; exact hb
  | a :: l, hf, b, v, hb, h => by
    rw [List.forIn_cons] at h
    obtain ⟨s, hs, h⟩ := except_bind_ok h
    obtain ⟨b', hs', hb'⟩ := hf a List.mem_cons_self b hb s hs
    rcases hs' with rfl | rfl
    · exact forIn_except_inv P l (fun a ha => hf a (List.mem_cons_of_mem _ ha)) b' v hb' h
    · cases h; exact hb'

theorem list_mapM_ok {α β : Type} {f : α → Except String β} :
    ∀ {l : List α} {ys : List β}, l.mapM f = .ok ys → ∀ y ∈ ys, ∃ x ∈ l, f x = .ok y
  | [], ys, h => by
    rw [List.mapM_nil] at h
    cases h
    simp
  | a :: l, ys, h => by
    rw [List.mapM_cons] at h
    obtain ⟨b, hb, h⟩ := except_bind_ok h
    obtain ⟨bs, hbs, h⟩ := except_bind_ok h
    cases h
    intro y hy
    rcases List.mem_cons.1 hy with rfl | hy
    · exact ⟨a, List.mem_cons_self, hb⟩
    · obtain ⟨x, hx, hfx⟩ := list_mapM_ok hbs y hy
      exact ⟨x, List.mem_cons_of_mem _ hx, hfx⟩

theorem array_mapM_ok {α β : Type} {f : α → Except String β} {xs : Array α} {ys : Array β}
    (h : xs.mapM f = .ok ys) : ∀ y ∈ ys.toList, ∃ x ∈ xs.toList, f x = .ok y := by
  rw [Array.mapM_eq_mapM_toList] at h
  rcases hm : xs.toList.mapM f with e | zs
  · rw [hm] at h; cases h
  · rw [hm] at h
    cases h
    simpa using list_mapM_ok hm

theorem findIdx?_lt {α : Type} {xs : Array α} {p : α → Bool} {i : ℕ} (h : xs.findIdx? p = some i) : i < xs.size :=
  (Array.findIdx?_eq_some_iff_findIdx_eq.1 h).1

theorem wire_eq {a b : Wire} (h1 : a.dst = b.dst) (h2 : a.src = b.src) : a = b := by
  cases a; cases b; simp_all

/-- `HmRow.parse`'s wire check: the file's wires begin with `derived`, and no later wire's ends satisfy `p`. -/
theorem wires_prefix {wires derived : Array Wire} {p : ℕ → Bool}
    (h : ¬ ((wires.size < derived.size || (List.range derived.size).any (fun i => wires[i]!.dst != derived[i]!.dst ||
      wires[i]!.src != derived[i]!.src) || (wires.extract derived.size wires.size).any (fun w => p w.dst.1 || p w.src.1)) = true)) :
    ∃ rest, wires.toList = derived.toList ++ rest ∧ ∀ w ∈ rest, p w.dst.1 = false ∧ p w.src.1 = false := by
  simp only [Bool.or_eq_true, decide_eq_true_eq, not_or, List.any_eq_true, Array.any_eq_true, List.mem_range,
    not_exists, not_and, Bool.not_eq_true, bne_eq_false_iff_eq] at h
  obtain ⟨⟨hsz, hpre⟩, hrest⟩ := h
  refine ⟨(wires.extract derived.size wires.size).toList, ?_, ?_⟩
  · have hsplit : wires.toList = (wires.extract 0 derived.size).toList ++ (wires.extract derived.size wires.size).toList := by
      rw [← Array.toList_append, Array.extract_append_extract]
      simp
    rw [hsplit]
    congr 1
    apply List.ext_getElem
    · simp; omega
    · intro i h1 h2
      simp only [Array.getElem_toList, Array.getElem_extract, Nat.zero_add]
      have := hpre i (by simpa using h2)
      rw [getElem!_pos wires i (by simp at h1; omega), getElem!_pos derived i (by simpa using h2)] at this
      exact wire_eq this.1 this.2
  · intro w hw
    rw [Array.mem_toList_iff] at hw
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hw
    simpa using hrest i hi

set_option maxHeartbeats 1000000 in
/-- `HmRow.rowPorts`' `sha512/row/v1` ports are whole SHA-512 blocks. -/
theorem rowPorts_words {mj : Lean.Json} {typed : Bool} {ports : Array Port} (h : HmRow.rowPorts mj typed = .ok ports) :
    ∀ p ∈ ports.toList, p.v2 = false → 2 * p.words = 128 * p.chunks := by
  unfold HmRow.rowPorts at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) v2 one at h
  obtain ⟨s, -, h⟩ := except_bind_ok h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp0 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp0] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp1 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp1] at h
  obtain ⟨listedPorts, -, h⟩ := except_bind_ok h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp2 at h
  obtain ⟨lengths, h⟩ : ∃ l, jp2 l = .ok ports := by
    split at h
    · obtain ⟨s, -, h⟩ := except_bind_ok h
      extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpv at h
      replace h := ite_throw_ok h
      simp (config := { zeta := false }) only [jpv] at h
      obtain ⟨l, -, h⟩ := except_bind_ok h
      exact ⟨l, h⟩
    · exact ⟨_, h⟩
  simp (config := { zeta := false }) only [jp2] at h
  obtain ⟨ports', hmap, h⟩ := except_bind_ok h
  obtain ⟨listed, -, h⟩ := except_bind_ok h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) bad jp3 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp3] at h
  obtain ⟨u, -, h⟩ := except_bind_ok h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp4 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp4] at h
  cases h
  intro p hp hv
  obtain ⟨i, -, hf⟩ := array_mapM_ok hmap p hp
  obtain ⟨a, -, hf⟩ := except_bind_ok hf
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpa at hf
  replace hf := ite_throw_ok hf
  simp (config := { zeta := false }) only [jpa] at hf
  obtain ⟨name, -, hf⟩ := except_bind_ok hf
  obtain ⟨words, -, hf⟩ := except_bind_ok hf
  obtain ⟨comps, -, hf⟩ := except_bind_ok hf
  split at hf
  · extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpn at hf
    obtain ⟨hbad, hf⟩ := ite_throw_ok' hf
    simp (config := { zeta := false }) only [jpn] at hf
    cases hf
    simp only [Bool.or_eq_true, beq_iff_eq, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hbad
    show 2 * words = 128 * comps
    omega
  · extract_lets (config := { underBinder := false, onlyGivenNames := true }) want jpw at hf
    replace hf := ite_throw_ok hf
    simp (config := { zeta := false }) only [jpw] at hf
    cases hf
    cases hv

set_option maxHeartbeats 1000000 in
/-- `HmRow.rowPorts`' `sha512/row/v1` ports are not segmented (a segmented row is a bit row). -/
theorem rowPorts_v1 {mj : Lean.Json} {typed : Bool} {ports : Array Port} (h : HmRow.rowPorts mj typed = .ok ports) :
    ∀ p ∈ ports.toList, p.v2 = false → p.segs = none := by
  unfold HmRow.rowPorts at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) v2 one at h
  obtain ⟨s, -, h⟩ := except_bind_ok h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp0 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp0] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp1 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp1] at h
  obtain ⟨listedPorts, -, h⟩ := except_bind_ok h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp2 at h
  obtain ⟨lengths, h⟩ : ∃ l, jp2 l = .ok ports := by
    split at h
    · obtain ⟨s, -, h⟩ := except_bind_ok h
      extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpv at h
      replace h := ite_throw_ok h
      simp (config := { zeta := false }) only [jpv] at h
      obtain ⟨l, -, h⟩ := except_bind_ok h
      exact ⟨l, h⟩
    · exact ⟨_, h⟩
  simp (config := { zeta := false }) only [jp2] at h
  obtain ⟨ports', hmap, h⟩ := except_bind_ok h
  obtain ⟨listed, -, h⟩ := except_bind_ok h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) bad jp3 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp3] at h
  obtain ⟨u, -, h⟩ := except_bind_ok h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp4 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp4] at h
  cases h
  intro p hp hv
  obtain ⟨i, -, hf⟩ := array_mapM_ok hmap p hp
  obtain ⟨a, -, hf⟩ := except_bind_ok hf
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpa at hf
  replace hf := ite_throw_ok hf
  simp (config := { zeta := false }) only [jpa] at hf
  obtain ⟨name, -, hf⟩ := except_bind_ok hf
  obtain ⟨words, -, hf⟩ := except_bind_ok hf
  obtain ⟨comps, -, hf⟩ := except_bind_ok hf
  split at hf
  · extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpn at hf
    obtain ⟨-, hf⟩ := ite_throw_ok' hf
    simp (config := { zeta := false }) only [jpn] at hf
    cases hf
    rfl
  · extract_lets (config := { underBinder := false, onlyGivenNames := true }) want jpw at hf
    replace hf := ite_throw_ok hf
    simp (config := { zeta := false }) only [jpw] at hf
    cases hf
    cases hv

/-! ## What `HmRow.parse` accepted -/

/-- **What `HmRow.parse` accepted.** -/
structure ParseFacts (tags : Tags) (c : Circuit) : Prop where
  nets : ∀ x ∈ c.nets.toList, Built x.2 ∨ ∃ nm tb n lo bits, Lookup.build nm tb n lo bits = .ok x.2
  /-- a `sha512/row/v1` port's row is whole SHA-512 blocks -/
  portWords : ∀ p ∈ c.ports.toList, p.v2 = false → 2 * p.words = 128 * p.chunks
  /-- a `sha512/row/v1` port is not segmented -/
  v1Segs : ∀ p ∈ c.ports.toList, p.v2 = false → p.segs = none
  /-- the rows' chains and hm96 inputs come first, and no other wire enters the `sha512x3` or `hm96` net -/
  rowWires : ∃ rest, c.wires.toList = (HmRow.rowWires c.ports c.shaNet c.hmNet).toList ++ rest ∧
    ∀ w ∈ rest, w.dst.1 ≠ c.shaNet ∧ w.dst.1 ≠ c.hmNet
  unitNet : c.unitNet < c.nets.size
  shaNet : c.shaNet < c.nets.size
  hmNet : c.hmNet < c.nets.size
  kinds : ∀ r ∈ c.ranges.toList, ∀ i, r.kind = .net i → i < c.nets.size
  check : ∃ ls, HmRow.check c tags ls = .ok ()
  /-- the `sha512x3` and `hm96` nets are the verifier's (`HmNets.check_ok`) -/
  rowNets : HmNets.check c = .ok ()
  /-- a `rec-open/l<L>-h<H>` unit is the verifier's (`RecOpen.check_ok`) -/
  recNet : RecOpen.check c = .ok ()
  /-- a `rec-residuals/m<m>-k<k>-c<c>-p<j>` unit is the verifier's -/
  recResiduals : RecResiduals.check c = .ok ()
  /-- a `rec-sparse/n<NU>-e<ELL>-s<S_LOG>-t<T_LOG>` unit is the verifier's -/
  recSparse : RecSparse.check c = .ok ()
  /-- a `rec-opening/m<m>-c<claims>-p<j>` unit is the verifier's -/
  recOpening : RecOpening.check c = .ok ()
  /-- not a typed template: `parse`'s `tmpl` is `none` -/
  typed : c.typed = none
  /-- no templated unit: only `HmOut.parseTmpl` carries one -/
  tmpl : c.tmpl = none

/-- **What `HmRow.parse` accepted, with nets `pre` first, of any kind**: `ParseFacts` but with each net one of `pre`'s,
`Built` or a lookup slot (a templated file's unit net, `Tmpl.unitNet`, is not `Built`). -/
structure ParseFactsPre (tags : Tags) (pre : Array (String × Net)) (c : Circuit) : Prop where
  nets : ∀ x ∈ c.nets.toList, x ∈ pre.toList ∨ Built x.2 ∨ ∃ nm tb n lo bits, Lookup.build nm tb n lo bits = .ok x.2
  portWords : ∀ p ∈ c.ports.toList, p.v2 = false → 2 * p.words = 128 * p.chunks
  v1Segs : ∀ p ∈ c.ports.toList, p.v2 = false → p.segs = none
  rowWires : ∃ rest, c.wires.toList = (HmRow.rowWires c.ports c.shaNet c.hmNet).toList ++ rest ∧
    ∀ w ∈ rest, w.dst.1 ≠ c.shaNet ∧ w.dst.1 ≠ c.hmNet
  unitNet : c.unitNet < c.nets.size
  shaNet : c.shaNet < c.nets.size
  hmNet : c.hmNet < c.nets.size
  kinds : ∀ r ∈ c.ranges.toList, ∀ i, r.kind = .net i → i < c.nets.size
  check : ∃ ls, HmRow.check c tags ls = .ok ()
  rowNets : HmNets.check c = .ok ()
  recNet : RecOpen.check c = .ok ()
  recResiduals : RecResiduals.check c = .ok ()
  recSparse : RecSparse.check c = .ok ()
  recOpening : RecOpening.check c = .ok ()
  typed : c.typed = none
  tmpl : c.tmpl = none

/-- `ParseFactsPre` with `pre`'s nets `Built` is `ParseFacts`. -/
theorem ParseFactsPre.facts {tags : Tags} {pre : Array (String × Net)} {c : Circuit} (P : ParseFactsPre tags pre c)
    (hpre : ∀ x ∈ pre.toList, Built x.2) : ParseFacts tags c :=
  { nets := fun x hx => (P.nets x hx).elim (fun h => .inl (hpre x h)) id, portWords := P.portWords,
    v1Segs := P.v1Segs, rowWires := P.rowWires, unitNet := P.unitNet, shaNet := P.shaNet, hmNet := P.hmNet,
    kinds := P.kinds, check := P.check, rowNets := P.rowNets, recNet := P.recNet, recResiduals := P.recResiduals,
    recSparse := P.recSparse, recOpening := P.recOpening, typed := P.typed, tmpl := P.tmpl }

set_option maxHeartbeats 4000000 in
/-- **What `HmRow.parse` accepted, with nets `pre` first, of any kind** (a typed flat class's unit net, or a
templated file's committed unit net): its nets are `pre`'s, then the file's own, each `Net.parse`'s or a lookup slot,
and its unit is the first named `unit`. -/
theorem parse_factsPre_nets {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    {pre : Array (String × Net)} {c : Circuit} (h : HmRow.parse tags bytes tables (pre := pre) = .ok c) :
    ParseFactsPre tags pre c ∧ (∃ more, c.nets.toList = pre.toList ++ more ∧ ∀ x ∈ more,
      (∃ lines, Net.parse lines = .ok x.2) ∨ ∃ nm tb n lo bits, Lookup.build nm tb n lo bits = .ok x.2) ∧
      c.nets.findIdx? (·.1 == "unit") = some c.unitNet ∧
      (c.nets.getD c.shaNet default).1 = HmRow.SHA_NET ∧ (c.nets.getD c.hmNet default).1 = HmRow.HM_NET := by
  unfold HmRow.parse at h
  simp (config := { zeta := false }) only [Option.isSome_none, Bool.false_eq_true, ↓reduceIte, Option.mapM_none] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) sha jp0 at h)
  split at h
  swap
  · obtain ⟨_, hte, -⟩ := except_bind_ok h; cases hte
  (have hh := except_bind_ok h; clear h; obtain ⟨text, hte, h⟩ := hh)
  cases hte
  simp (config := { zeta := false }) only [jp0] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) lines0 lines head jp1 at h)
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp1] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) name metaLine jp2 at h)
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp2] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp3 at h)
  split at h
  swap
  · obtain ⟨_, hte, -⟩ := except_bind_ok h; cases hte
  (have hh := except_bind_ok h; clear h; obtain ⟨mj, hte, h⟩ := hh)
  cases hte
  simp (config := { zeta := false }) only [jp3] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨f1, hf1, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨f2, hf2, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp4 at h)
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp4] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp5 jp6 at h)
  obtain ⟨classSha, h⟩ : ∃ classSha, jp5 classSha = .ok c := by
    split at h
    · replace h := ite_throw_ok h
      simp (config := { zeta := false }) only [jp6] at h
      ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) part rule jp7 at h)
      replace h := ite_throw_ok h
      simp (config := { zeta := false }) only [jp7] at h
      obtain ⟨cs, -, h⟩ := except_bind_ok h
      exact ⟨cs, h⟩
    · obtain ⟨cs, -, h⟩ := except_bind_ok h
      exact ⟨cs, h⟩
  simp (config := { zeta := false }) only [jp5] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets0 at0 at h)
  (have hh := except_bind_ok h; clear h; obtain ⟨s1, hwhile, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets1 at h)
  (have hh := except_bind_ok h; clear h; obtain ⟨u1, -, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpT at h)
  simp (config := { zeta := false }) only [jpT] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpL at h)
  obtain ⟨lks, h⟩ : ∃ lks, jpL lks = .ok c := by
    split at h <;> (obtain ⟨l, -, h⟩ := except_bind_ok h; exact ⟨l, h⟩)
  simp (config := { zeta := false }) only [jpL] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨s2, hlook, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets2 at h)
  (have hh := except_bind_ok h; clear h; obtain ⟨ports, hrp, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpK at h)
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jpK] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨g, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨upv, -, h⟩ := hh)
  split at h
  swap
  · cases h
  rename_i unitNet hunit
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) netOf at h)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨ranges, hranges, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨leavesIn, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨leavesOut, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨dummy, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨dummyDigests, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨dummyOutputs, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨outPorts, -, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpR at h)
  obtain ⟨outRowPorts, h⟩ : ∃ r, jpR r = .ok c := by
    split at h
    · exact ⟨_, h⟩
    · obtain ⟨_, -, h⟩ := except_bind_ok h
      obtain ⟨r, -, h⟩ := except_bind_ok h
      exact ⟨r, ite_throw_ok h⟩
  simp (config := { zeta := false }) only [jpR] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) perVu0 jpP at h)
  obtain ⟨pv, h⟩ : ∃ pv, jpP () pv = .ok c := by
    split at h
    · obtain ⟨p', -, h⟩ := except_bind_ok h
      exact ⟨_, h⟩
    · exact ⟨_, h⟩
  simp (config := { zeta := false }) only [jpP] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨shaNet, hsha, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨hmNet, hhm, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨outNet, -, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpO at h)
  obtain ⟨onp, h⟩ : ∃ x, jpO x = .ok c := by
    split at h <;> (obtain ⟨l, -, h⟩ := except_bind_ok h; exact ⟨l, h⟩)
  simp (config := { zeta := false }) only [jpO] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpC at h)
  obtain ⟨lc, h⟩ : ∃ x, jpC x = .ok c := by
    split at h <;> (obtain ⟨l, -, h⟩ := except_bind_ok h; exact ⟨l, h⟩)
  simp (config := { zeta := false }) only [jpC] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpW at h)
  obtain ⟨wires, h⟩ : ∃ x, jpW x = .ok c := by
    split at h
    · obtain ⟨_, -, h⟩ := except_bind_ok h
      obtain ⟨l, -, h⟩ := except_bind_ok h
      exact ⟨l, h⟩
    · obtain ⟨l, -, h⟩ := except_bind_ok h
      exact ⟨l, h⟩
  simp (config := { zeta := false }) only [jpW] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) derived wiresT rowNet at h)
  obtain ⟨hwires, h⟩ := ite_throw_ok' h
  simp (config := { zeta := false }) only [rowNet] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨ls, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨leafId, -, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpF at h)
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jpF] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨kLog, -, h⟩ := hh)
  simp (config := { zeta := false }) only [pure_bind] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) c0 at h)
  (have hh := except_bind_ok h; clear h; obtain ⟨w, hcheck, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨w', hrows, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨w'', hrec, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨w''', hres, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨w'''', hsparse, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨w5, hopening, h⟩ := hh)
  have hc : c0 = c := Except.ok.inj h
  subst hc
  have hnetOf : ∀ nm i, netOf nm = .ok i → i < nets2.size := by
    intro nm i hi
    simp only [netOf] at hi
    split at hi
    · rename_i i' hi'; cases hi; exact findIdx?_lt hi'
    · cases hi
  -- the text nets (the `while` loop over the `NET` blocks)
  have htext : ∃ more, nets1.toList = nets0.toList ++ more ∧ ∀ x ∈ more, ∃ lines, Net.parse lines = .ok x.2 := by
    refine loop_inv (fun b : Array (String × Net) × ℕ =>
        ∃ more, b.1.toList = nets0.toList ++ more ∧ ∀ x ∈ more, ∃ lines, Net.parse lines = .ok x.2)
      (fun b => lines.size - b.2) ?_ _ (nets0, at0) s1 le_rfl ⟨[], by simp, by simp⟩ hwhile
    intro b hb s h
    ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets at_ at h)
    split at h
    · rename_i hlt
      ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) hh jp at h)
      replace h := ite_throw_ok h
      simp (config := { zeta := false }) only [jp] at h
      split at h
      · rename_i n hn
        ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp' at h)
        obtain ⟨hnot, h⟩ := ite_throw_ok' h
        simp (config := { zeta := false }) only [jp'] at h
        ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp'' at h)
        split at h
        · rename_i v hv
          (have hh := except_bind_ok h; clear h; obtain ⟨net, hnet, h⟩ := hh)
          cases hnet
          simp (config := { zeta := false }) only [jp''] at h
          ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets' at' at h)
          have hs := Except.ok.inj h
          subst hs
          refine Or.inr ⟨_, rfl, ?_, ?_⟩
          · obtain ⟨more, hm, hmore⟩ := hb
            refine ⟨more ++ [_], by simp [nets', nets, hm]; rfl, fun x hx => ?_⟩
            simp only [List.mem_append, List.mem_singleton] at hx
            rcases hx with hx | rfl
            · exact hmore x hx
            · exact ⟨_, hv⟩
          · simp only [at', at_]; omega
        · (have hh := except_bind_ok h; clear h; obtain ⟨net, hnet, h⟩ := hh)
          cases hnet
      · (have hh := except_bind_ok h; clear h; obtain ⟨w, hw, h⟩ := hh)
        cases hw
    · have hs := Except.ok.inj h
      subst hs
      exact Or.inl ⟨_, rfl, hb⟩
  -- the lookup slots
  have hnets : ∃ more, nets2.toList = nets0.toList ++ more ∧ ∀ x ∈ more,
      (∃ lines, Net.parse lines = .ok x.2) ∨ ∃ nm tb n lo bits, Lookup.build nm tb n lo bits = .ok x.2 := by
    rw [← Array.forIn_toList] at hlook
    obtain ⟨more0, hm0, hmore0⟩ := htext
    refine forIn_except_inv (fun nets : Array (String × Net) => ∃ more, nets.toList = nets0.toList ++ more ∧ ∀ x ∈ more,
      (∃ lines, Net.parse lines = .ok x.2) ∨ ∃ nm tb n lo bits, Lookup.build nm tb n lo bits = .ok x.2)
      _ ?_ nets1 s2 ⟨more0, hm0, fun x hx => Or.inl (hmore0 x hx)⟩ hlook
    intro a _ b hb s h
    obtain ⟨lname, spec⟩ := a
    ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets at h)
    split at h
    · ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) size at h)
      (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
      (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
      (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
      ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpA at h)
      replace h := ite_throw_ok h
      simp (config := { zeta := false }) only [jpA] at h
      (have hh := except_bind_ok h; clear h; obtain ⟨n, -, h⟩ := hh)
      (have hh := except_bind_ok h; clear h; obtain ⟨lo, -, h⟩ := hh)
      ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpB at h)
      replace h := ite_throw_ok h
      simp (config := { zeta := false }) only [jpB] at h
      split at h
      · ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpC at h)
        replace h := ite_throw_ok h
        simp (config := { zeta := false }) only [jpC] at h
        (have hh := except_bind_ok h; clear h; obtain ⟨net, hbuild, h⟩ := hh)
        ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets' at h)
        have hs := Except.ok.inj h
        subst hs
        refine ⟨_, Or.inl rfl, ?_⟩
        obtain ⟨more, hm, hmore⟩ := hb
        refine ⟨more ++ [_], by simp [nets', nets, hm]; rfl, fun x hx => ?_⟩
        simp only [List.mem_append, List.mem_singleton] at hx
        rcases hx with hx | rfl
        · exact hmore x hx
        · exact Or.inr ⟨_, _, _, _, _, hbuild⟩
      · (have hh := except_bind_ok h; clear h; obtain ⟨w, hw, h⟩ := hh)
        cases hw
    · (have hh := except_bind_ok h; clear h; obtain ⟨w, hw, h⟩ := hh)
      cases hw
  -- the ranges' nets
  have hkinds : ∀ r ∈ ranges.toList, ∀ i, r.kind = .net i → i < nets2.size := by
    intro r hr i hi
    obtain ⟨x, -, h⟩ := array_mapM_ok hranges r hr
    (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
    ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpK at h)
    split at h
    · (have hh := except_bind_ok h; clear h; obtain ⟨kind, hk, h⟩ := hh)
      cases hk
      simp (config := { zeta := false }) only [jpK] at h
      ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) packed at h)
      (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
      (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
      (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
      have hs := Except.ok.inj h
      subst hs
      cases hi
    · (have hh := except_bind_ok h; clear h; obtain ⟨nm, -, h⟩ := hh)
      (have hh := except_bind_ok h; clear h; obtain ⟨i', hi', h⟩ := hh)
      (have hh := except_bind_ok h; clear h; obtain ⟨kind, hk, h⟩ := hh)
      cases hk
      simp (config := { zeta := false }) only [jpK] at h
      ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) packed at h)
      (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
      (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
      (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
      have hs := Except.ok.inj h
      subst hs
      cases hi
      exact hnetOf _ _ hi'
    · (have hh := except_bind_ok h; clear h; obtain ⟨kind, hk, h⟩ := hh)
      cases hk
  obtain ⟨more, hm, hmore⟩ := hnets
  have hbuilt : ∀ x ∈ nets2.toList,
      x ∈ pre.toList ∨ Built x.2 ∨ ∃ nm tb n lo bits, Lookup.build nm tb n lo bits = .ok x.2 := by
    intro x hx
    rw [hm, List.mem_append] at hx
    rcases hx with hx | hx
    · exact Or.inl hx
    · rcases hmore x hx with ⟨_, hp⟩ | hb
      · exact Or.inr (Or.inl (parse_built hp))
      · exact Or.inr (Or.inr hb)
  have hname : ∀ nm i, netOf nm = .ok i → (nets2.getD i default).1 = nm := by
    intro nm i hi
    simp only [netOf] at hi
    split at hi
    · rename_i i' hi'
      cases hi
      obtain ⟨hlt, hp, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hi'
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hlt, Option.getD_some]
      simpa using hp
    · cases hi
  refine ⟨?_, ⟨more, hm, hmore⟩, hunit, hname _ _ hsha, hname _ _ hhm⟩
  have hrowWires := wires_prefix (p := rowNet) hwires
  exact { nets := hbuilt, portWords := rowPorts_words hrp, v1Segs := rowPorts_v1 hrp,
          rowWires := by
            obtain ⟨rest, hl, hr⟩ := hrowWires
            refine ⟨rest, hl, fun w hw => ?_⟩
            have := (hr w hw).1
            simp only [rowNet, Bool.or_eq_false_iff, beq_eq_false_iff_ne] at this
            exact this,
          unitNet := findIdx?_lt hunit, shaNet := hnetOf _ _ hsha,
          hmNet := hnetOf _ _ hhm, kinds := hkinds, check := ⟨ls, by cases w; exact hcheck⟩,
          rowNets := by cases w'; exact hrows, recNet := by cases w''; exact hrec,
          recResiduals := by cases w'''; exact hres, recSparse := by cases w''''; exact hsparse,
          recOpening := by cases w5; exact hopening, typed := rfl, tmpl := rfl }

/-- **What `HmRow.parse` accepted, with nets `pre` first** (a typed flat class's unit net): its nets are `pre`'s,
then the file's own, each `Net.parse`'s or a lookup slot, and its unit is the first named `unit`. -/
theorem parse_facts_nets {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {pre : Array (String × Net)}
    {c : Circuit} (hpre : ∀ x ∈ pre.toList, Built x.2) (h : HmRow.parse tags bytes tables (pre := pre) = .ok c) :
    ParseFacts tags c ∧ (∃ more, c.nets.toList = pre.toList ++ more ∧ ∀ x ∈ more,
      (∃ lines, Net.parse lines = .ok x.2) ∨ ∃ nm tb n lo bits, Lookup.build nm tb n lo bits = .ok x.2) ∧
      c.nets.findIdx? (·.1 == "unit") = some c.unitNet ∧
      (c.nets.getD c.shaNet default).1 = HmRow.SHA_NET ∧ (c.nets.getD c.hmNet default).1 = HmRow.HM_NET :=
  let ⟨P, hm, hu, hs, hh⟩ := parse_factsPre_nets h
  ⟨P.facts hpre, hm, hu, hs, hh⟩

/-- **What `HmRow.parse` accepted, with nets `pre` first**: its nets are `pre`'s, then the file's own, and its unit is
the first named `unit`. -/
theorem parse_facts_pre {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {pre : Array (String × Net)}
    {c : Circuit} (hpre : ∀ x ∈ pre.toList, Built x.2) (h : HmRow.parse tags bytes tables (pre := pre) = .ok c) :
    ParseFacts tags c ∧ (∃ more, c.nets.toList = pre.toList ++ more) ∧ c.nets.findIdx? (·.1 == "unit") = some c.unitNet ∧
      (c.nets.getD c.shaNet default).1 = HmRow.SHA_NET ∧ (c.nets.getD c.hmNet default).1 = HmRow.HM_NET :=
  let ⟨P, ⟨more, hm, _⟩, hu, hs, hh⟩ := parse_facts_nets hpre h
  ⟨P, ⟨more, hm⟩, hu, hs, hh⟩

theorem parse_facts {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (h : HmRow.parse tags bytes tables = .ok c) : ParseFacts tags c :=
  (parse_facts_pre (fun _ hx => by simp at hx) h).1

end FlockSoundness
