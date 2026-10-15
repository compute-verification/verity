import Proofs.Flock.Soundness.ExecCircuit
import Proofs.Flock.Soundness.ExecCheck
import Proofs.Flock.Soundness.ExecTyped
import Proofs.Flock.Soundness.ExecSetup

/-!
# A typed template's statement, as `HmRow.parse` and `Stmt.setupH` accept it (1e for templates, T2)

`HmRow.parse … (tmpl := some t)` is the path `parseTyped` takes for a typed template (#277, #307):
- `parse_facts_tmpl`: its nets are the template's root and placed layouts' circuits, then the file's own text nets and
  lookup slots; its unit is the root (`unitNet = 0`), one per template instance; its block is `blockOf t` over those nets; its wires are
  the rows' own chains (`rowWires`), with no leaf map or cut; and `HmRow.check` passed. A typed statement's rows are
  `sha512/row/v1`'s (`rowPorts_typed`).
-/

open Flock FlockLevel3

namespace FlockSoundness

set_option maxHeartbeats 1000000 in
/-- `HmRow.rowPorts`' ports for a typed statement are `sha512/row/v1`'s (it refuses `row_schemas`), each whole SHA-512
blocks of `16 · words` bits, with a compression at least (it refuses a row of no words). -/
theorem rowPorts_typed {mj : Lean.Json} {ports : Array Port} (h : HmRow.rowPorts mj true = .ok ports) :
    ∀ p ∈ ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks := by
  unfold HmRow.rowPorts at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) v2 one at h
  obtain ⟨s, -, h⟩ := except_bind_ok h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp0 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp0] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp1 at h
  obtain ⟨hv2, h⟩ := ite_throw_ok' h
  simp (config := { zeta := false }) only [jp1] at h
  replace hv2 : v2 = false := by simpa using hv2
  obtain ⟨listedPorts, -, h⟩ := except_bind_ok h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp2 at h
  simp (config := { zeta := false }) only [hv2, Bool.false_eq_true, ↓reduceIte] at h
  change jp2 (Array.replicate listedPorts.size none) = .ok ports at h
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
  intro p hp
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
    refine ⟨rfl, rfl, ?_⟩
    show 0 < comps
    omega
  · rename_i heq
    have hn : (Array.replicate listedPorts.size (none : Option (ℕ × Bool)))[i]! = none := by
      by_cases hi : i < listedPorts.size
      · rw [getElem!_pos _ i (by simpa using hi)]; simp
      · rw [getElem!_neg _ i (by simpa using hi)]; rfl
    rw [hn] at heq
    cases heq

/-- **What `HmRow.parse` accepted for a typed template's block `t`.** -/
structure ParseFactsT (tags : Tags) (t : Typed.Template) (c : Circuit) : Prop where
  pre : ∃ more, c.nets.toList = (#[("root", t.root)] ++ t.layouts.map fun (e : String × Net × ℕ) => (e.1, e.2.1)).toList ++ more ∧
    ∀ x ∈ more, (∃ lines, Net.parse lines = .ok x.2) ∨ ∃ nm tb n lo bits, Lookup.build nm tb n lo bits = .ok x.2
  /-- a `sha512/row/v1` port's row is whole SHA-512 blocks -/
  portWords : ∀ p ∈ c.ports.toList, p.v2 = false → 2 * p.words = 128 * p.chunks
  /-- a `sha512/row/v1` port is not segmented -/
  v1Segs : ∀ p ∈ c.ports.toList, p.v2 = false → p.segs = none
  unitNet : c.unitNet < c.nets.size
  root : c.unitNet = 0
  shaNet : c.shaNet < c.nets.size
  hmNet : c.hmNet < c.nets.size
  kinds : ∀ r ∈ c.ranges.toList, ∀ i, r.kind = .net i → i < c.nets.size
  check : ∃ ls, HmRow.check c tags ls = .ok ()
  /-- the `sha512x3` and `hm96` nets are the verifier's (`HmNets.check_ok`) -/
  rowNets : HmNets.check c = .ok ()
  typed : c.typed = some (HmRow.blockOf t)
  upv : c.upv = 1
  leavesIn : c.leavesIn = #[#[]]
  wires : c.wires = HmRow.rowWires c.ports c.shaNet c.hmNet
  leafCuts : c.leafCuts = #[]
  /-- a typed statement's rows are `sha512/row/v1`'s, each of `16 · words` bits in at least one compression -/
  v1 : tags.typed = true → ∀ p ∈ c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks

set_option maxHeartbeats 4000000 in
theorem parse_facts_tmpl {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {t : Typed.Template}
    {c : Circuit} (h : HmRow.parse tags bytes tables (tmpl := some t) = .ok c) : ParseFactsT tags t c := by
  unfold HmRow.parse at h
  simp (config := { zeta := false }) only [Option.isSome_some, ↓reduceIte, Option.mapM_some] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) sha jp0 at h
  split at h
  swap
  · obtain ⟨_, hte, -⟩ := except_bind_ok h; cases hte
  (have hh := except_bind_ok h; clear h; obtain ⟨text, hte, h⟩ := hh)
  cases hte
  simp (config := { zeta := false }) only [jp0] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) lines0 lines head jp1 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp1] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) name metaLine jp2 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp2] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp3 at h
  split at h
  swap
  · obtain ⟨_, hte, -⟩ := except_bind_ok h; cases hte
  rename_i jv hjv
  (have hh := except_bind_ok h; clear h; obtain ⟨mj, hte, h⟩ := hh)
  cases hte
  simp (config := { zeta := false }) only [jp3] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨f1, hf1, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨f2, hf2, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp4 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp4] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp5 jp6 at h
  obtain ⟨classSha, h⟩ : ∃ classSha, jp5 classSha = .ok c := by
    split at h
    · replace h := ite_throw_ok h
      simp (config := { zeta := false }) only [jp6] at h
      extract_lets (config := { underBinder := false, onlyGivenNames := true }) part rule jp7 at h
      replace h := ite_throw_ok h
      simp (config := { zeta := false }) only [jp7] at h
      obtain ⟨cs, -, h⟩ := except_bind_ok h
      exact ⟨cs, h⟩
    · obtain ⟨cs, -, h⟩ := except_bind_ok h
      exact ⟨cs, h⟩
  simp (config := { zeta := false }) only [jp5] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets0 at0 at h
  (have hh := except_bind_ok h; clear h; obtain ⟨s1, hwhile, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets1 at h
  (have hh := except_bind_ok h; clear h; obtain ⟨u1, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpT at h
  (have hh := except_bind_ok h; clear h; obtain ⟨uT, hderived, h⟩ := hh)
  simp (config := { zeta := false }) only [jpT] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpL at h
  obtain ⟨lks, h⟩ : ∃ lks, jpL lks = .ok c := by
    split at h <;> (obtain ⟨l, -, h⟩ := except_bind_ok h; exact ⟨l, h⟩)
  simp (config := { zeta := false }) only [jpL] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨s2, hlook, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets2 at h
  (have hh := except_bind_ok h; clear h; obtain ⟨ports, hrp, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpK at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jpK] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨g, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨upv, hupv, h⟩ := hh)
  split at h
  swap
  · cases h
  rename_i unitNet hunit
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) netOf at h)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨ranges, hranges, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨leavesIn, hleavesIn, h⟩ := hh)
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
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) perVu0 jpP at h
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
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpC at h
  obtain ⟨lc, h, hlc⟩ : ∃ x, jpC x = .ok c ∧ (J.has jv "leaf_cuts" = false → x = #[]) := by
    split at h
    · rename_i a ha
      obtain ⟨l, -, h⟩ := except_bind_ok h
      exact ⟨l, h, fun hn => by simp [J.has, ha] at hn⟩
    · obtain ⟨l, hl, h⟩ := except_bind_ok h
      cases hl
      exact ⟨_, h, fun _ => rfl⟩
  simp (config := { zeta := false }) only [jpC] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpW at h
  obtain ⟨wires, h⟩ : ∃ x, jpW x = .ok c := by
    split at h
    · obtain ⟨_, -, h⟩ := except_bind_ok h
      obtain ⟨l, -, h⟩ := except_bind_ok h
      exact ⟨l, h⟩
    · obtain ⟨l, -, h⟩ := except_bind_ok h
      exact ⟨l, h⟩
  simp (config := { zeta := false }) only [jpW] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) derived wiresT rowNet jpD at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jpD] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨ls, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨leafId, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpF at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jpF] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨kLog, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨tyd, htyd, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) c0 at h
  (have hh := except_bind_ok h; clear h; obtain ⟨w, hcheck, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨w', hrows, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
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
    extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets at_ at h
    split at h
    · rename_i hlt
      extract_lets (config := { underBinder := false, onlyGivenNames := true }) hh jp at h
      replace h := ite_throw_ok h
      simp (config := { zeta := false }) only [jp] at h
      split at h
      · rename_i n hn
        extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp' at h
        obtain ⟨hnot, h⟩ := ite_throw_ok' h
        simp (config := { zeta := false }) only [jp'] at h
        extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp'' at h
        split at h
        · rename_i v hv
          (have hh := except_bind_ok h; clear h; obtain ⟨net, hnet, h⟩ := hh)
          cases hnet
          simp (config := { zeta := false }) only [jp''] at h
          extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets' at' at h
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
    extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets at h
    split at h
    · extract_lets (config := { underBinder := false, onlyGivenNames := true }) size at h
      (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
      (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
      (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
      extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpA at h
      replace h := ite_throw_ok h
      simp (config := { zeta := false }) only [jpA] at h
      (have hh := except_bind_ok h; clear h; obtain ⟨n, -, h⟩ := hh)
      (have hh := except_bind_ok h; clear h; obtain ⟨lo, -, h⟩ := hh)
      extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpB at h
      replace h := ite_throw_ok h
      simp (config := { zeta := false }) only [jpB] at h
      split at h
      · extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpC at h
        replace h := ite_throw_ok h
        simp (config := { zeta := false }) only [jpC] at h
        (have hh := except_bind_ok h; clear h; obtain ⟨net, hbuild, h⟩ := hh)
        extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets' at h
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
    extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpK at h
    split at h
    · (have hh := except_bind_ok h; clear h; obtain ⟨kind, hk, h⟩ := hh)
      cases hk
      simp (config := { zeta := false }) only [jpK] at h
      extract_lets (config := { underBinder := false, onlyGivenNames := true }) packed at h
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
      extract_lets (config := { underBinder := false, onlyGivenNames := true }) packed at h
      (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
      (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
      (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
      have hs := Except.ok.inj h
      subst hs
      cases hi
      exact hnetOf _ _ hi'
    · (have hh := except_bind_ok h; clear h; obtain ⟨kind, hk, h⟩ := hh)
      cases hk
  -- the template's own facts
  have htyped : tyd = some (HmRow.blockOf t) := by
    simpa [Functor.map, Except.map, pure, Except.pure] using htyd.symm
  have hkeys := forIn_list_forall (P := fun k => J.has jv k = false) (fun k _ s t hs => by
    split at hs
    · obtain ⟨_, hx, -⟩ := except_bind_ok hs; cases hx
    · rename_i hk; cases hs; exact ⟨⟨_, rfl⟩, by simpa using hk⟩) hderived
  exact { pre := by obtain ⟨more, hm, hmore⟩ := hnets; exact ⟨more, hm, hmore⟩
          portWords := rowPorts_words hrp, v1Segs := rowPorts_v1 hrp, unitNet := findIdx?_lt hunit
          root := by
            obtain ⟨more, hm, -⟩ := hnets
            obtain ⟨hlt, -, hmin⟩ := Array.findIdx?_eq_some_iff_getElem.1 hunit
            by_contra hne
            have h0 := hmin 0 (Nat.pos_of_ne_zero hne)
            have h1 : nets2[0]? = some ("root", t.root) := by
              rw [← Array.getElem?_toList, hm]; simp [nets0]
            have h2 : nets2[0] = ("root", t.root) := by
              rw [Array.getElem?_eq_getElem (by omega)] at h1; exact Option.some.inj h1
            simp [h2] at h0
          shaNet := hnetOf _ _ hsha,
          hmNet := hnetOf _ _ hhm, kinds := hkinds, check := ⟨ls, by cases w; exact hcheck⟩
          rowNets := by cases w'; exact hrows
          typed := htyped
          upv := (Except.ok.inj hupv).symm
          leavesIn := (Except.ok.inj hleavesIn).symm
          wires := rfl
          leafCuts := hlc (hkeys "leaf_cuts" (by simp [HmRow.TEMPLATE_DERIVED]))
          v1 := fun ht => rowPorts_typed (ht ▸ hrp) }

/-- **What `parseTyped` accepted** (#307): `Typed.read`'s class `k` and block, then `parse` on the block, a template's
(`tmpl`) or a flat class's unit net (`pre`), with the class recorded as `cls`. -/
theorem parseTyped_spec {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (h : HmRow.parseTyped tags bytes tables = .ok c) :
    ∃ k r, Typed.read tags bytes tables = .ok (k, r) ∧
      ((∃ t c', r = .inr t ∧ HmRow.parse tags bytes tables (tmpl := some t) = .ok c' ∧ c = { c' with cls := some k }) ∨
        ∃ net c', r = .inl net ∧ HmRow.parse tags bytes tables (pre := #[("unit", net)]) = .ok c' ∧
          c = { c' with cls := some k }) := by
  unfold HmRow.parseTyped at h
  obtain ⟨⟨k, r⟩, hr, h⟩ := except_bind_ok h
  refine ⟨k, r, hr, ?_⟩
  cases r with
  | inr t =>
    obtain ⟨c', hc', h⟩ := except_bind_ok h
    exact Or.inl ⟨t, c', rfl, hc', (Except.ok.inj h).symm⟩
  | inl net =>
    obtain ⟨c', hc', h⟩ := except_bind_ok h
    exact Or.inr ⟨net, c', rfl, hc', (Except.ok.inj h).symm⟩

/-- **What `Stmt.setupH` accepted, for a typed statement** (`tags.typed = true`): the circuit `parseTyped` accepted, the
pin `HmRow.pin` gave, Δ from `HmRow.delta`, `m = k_log + nbl` and `k_log ≤ 27`. -/
theorem setupH_spec_typed {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (ht : tags.typed = true) (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) :
    HmRow.parseTyped tags cf tables = .ok st.c ∧ HmRow.pin st.c = .ok st.pin ∧
      st.da = (HmRow.delta st.c st.pin).1 ∧ st.db = (HmRow.delta st.c st.pin).2 ∧
      st.m = st.c.kLog + st.nbl ∧ st.c.kLog ≤ 27 := by
  unfold Stmt.setupH at h
  simp (config := { zeta := false }) only [ht, ↓reduceIte] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨c, hc, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpA at h)
  obtain ⟨f0, h⟩ : ∃ f0, jpA f0 = .ok st := by
    split at h
    · obtain ⟨f, -, h⟩ := except_bind_ok h; exact ⟨f, h⟩
    · split at h <;> (obtain ⟨f, -, h⟩ := except_bind_ok h; exact ⟨f, h⟩)
  simp (config := { zeta := false }) only [jpA] at h
  split at h
  swap
  · cases h
  (have hh := except_bind_ok h; clear h; obtain ⟨pop, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpB at h
  obtain ⟨pub, h⟩ : ∃ pub, jpB pub = .ok st := by
    split at h <;> (obtain ⟨p, -, h⟩ := except_bind_ok h; exact ⟨p, h⟩)
  simp (config := { zeta := false }) only [jpB] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpU at h
  obtain ⟨u0, h⟩ : ∃ u, jpU u = .ok st := by
    split at h
    · split at h
      · obtain ⟨_, -, h⟩ := except_bind_ok h
        exact ⟨(), ite_throw_ok h⟩
      · exact ⟨(), h⟩
    · exact ⟨(), h⟩
  simp (config := { zeta := false }) only [jpU] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpC at h
  obtain ⟨finding, h⟩ : ∃ f, jpC f = .ok st := by
    split at h
    · extract_lets (config := { underBinder := false, onlyGivenNames := true }) q at h
      split at h
      · obtain ⟨_, -, h⟩ := except_bind_ok h
        obtain ⟨_, -, h⟩ := except_bind_ok h
        obtain ⟨_, -, h⟩ := except_bind_ok h
        obtain ⟨_, -, h⟩ := except_bind_ok h
        obtain ⟨f, -, h⟩ := except_bind_ok h
        exact ⟨f, h⟩
      · obtain ⟨f, -, h⟩ := except_bind_ok h; exact ⟨f, h⟩
    · obtain ⟨f, -, h⟩ := except_bind_ok h; exact ⟨f, h⟩
  simp (config := { zeta := false }) only [jpC] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) n blocksReal nbl m jpD at h
  obtain ⟨hk, h⟩ := ite_throw_ok' h
  simp (config := { zeta := false }) only [jpD] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨pin, hpin, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) h0 at h
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) h1 at h
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) h2 digest sigma at h
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  have hs := Except.ok.inj h
  subst hs
  refine ⟨hc, hpin, rfl, rfl, rfl, ?_⟩
  simp only [Bool.or_eq_true, decide_eq_true_eq, not_or] at hk
  exact Nat.le_of_not_lt hk.1


end FlockSoundness
