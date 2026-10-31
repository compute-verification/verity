import Proofs.Flock.Claims.Recursion.VStarTrace
import Proofs.Flock.Claims.Recursion.VStarLinks
import Proofs.Flock.Claims.Recursion.SlotsCheck
import Proofs.Flock.Claims.Scope.InScope
import Proofs.Flock.Recursive.VStarCheck
import Proofs.Flock.Soundness.Discharge.Hm.Place
import Proofs.Flock.Soundness.Discharge.Hm.Copy
import Proofs.Flock.Soundness.Discharge.Aliased.Rows
import Proofs.Flock.Soundness.Discharge.Layout.KeyProg

/-!
# `TraceOk` at the deployed statements (`traceOk_deployed`)

`VStarTrace.TraceOk c pin` is what the port lemma reads of a part's circuit and its port inputs. At a staged part
(`StagedPart`), with part `s`'s statement and circuit `partAt x s = some (sv, c)` and the stager's port inputs
`partIn src (progOf c) sv = .ok pin`, every field but `rows` follows from the verifier's own checks, given that the
circuit has three nets:

* the circuit's fields from what the verifier parsed and checked of the staged table (`traceOk_of`): `HmRow.parse`'s
  `ParseFacts` and `HmRow.check`'s `CheckFacts` (ranges apart and inside the block, slots their nets', one unit range of
  `g · upv` slots, `nextPow2` row slots per VU, the unit `Net.ofRows`'s with 16-bit leaf ports, the `sha512x3` and
  `hm96` nets the verifier's), `ProvedScope.scopeOk` (one input group, `unitLog ≤ 32`, unsegmented `16 · words`-bit
  rows), V\*'s parse check (`checkParsed`: one unit per VU, no leaf cuts) and the setup's `kLog ≤ 27`;
* `refs` from the stager's builders (`partIn_refs`): each of a level's, algebra part's or cap's ports has one ref per
  opening, rep or top node.

Two facts the checks don't give, the hypotheses here:

* `c.nets.size = 3`: V\*'s file writes exactly the unit, `sha512x3` and `hm96` (`VStar.emitSections`,
  `VStar/Compose.lean:234`), and `checkFile` holds the bytes to it, but no proof reads through `Lean.Json.parse`, and
  `HmRow.check` accepts further nets. With it, `supported`'s lookups and wires follow: the three nets are `Net.ofRows`'s,
  and a wire into the unit would reach its one (leaf) input group, which `HmRow.check` refuses.
* `RowsOk c pin` (`TraceOk.rows`): `partIn` checks each opening's lane and sibling counts but not the rows' words or
  salts. The running sums' salts, the fresh salts, κ, eq(α) and v come from `InnerRec.rest` and the message rounds'
  words from `InnerRec.rows`, which are arbitrary: an `InnerRec` whose `rest` has `accSalts = #[]` gives a level's `acc`
  rows the empty salt, `partIn` still succeeds, and `rows` fails. It is a fact of the honest inner prover and of
  V\*'s port widths (`vstarRows`), not of the verifier.
-/

namespace RecDeploy.VStarTraceOk

open _root_.Flock
open Claims (Msg)
open RecDeploy.VStarBuild (partAt progOf portRow)
open RecDeploy.VStarTrace (TraceOk hmN InR)

/-! ## The stager's refs -/

theorem array_mapM_size {α β : Type} {f : α → Option β} {a : Array α} {b : Array β}
    (h : a.mapM f = some b) : b.size = a.size := by
  have key : ∀ {l : List α} {l' : List β}, l.mapM f = some l' → l'.length = l.length := by
    intro l
    induction l with
    | nil => intro l' h; simp at h; subst h; rfl
    | cons a l ih =>
      intro l' h
      simp only [List.mapM_cons, bind, Option.bind_eq_some_iff, pure, Option.some.injEq] at h
      obtain ⟨b, -, l'', hl, rfl⟩ := h
      simp [ih hl]
  rw [Array.mapM_eq_mapM_toList] at h
  simp only [Functor.map, Option.map_eq_some_iff] at h
  obtain ⟨l, hl, rfl⟩ := h
  simpa using key hl

theorem forIn_inv {α σ : Type} (P : σ → Prop) {f : α → σ → Except String (ForInStep σ)}
    (hf : ∀ a b s, P b → f a b = .ok s → ∃ b', s = .yield b' ∧ P b') :
    ∀ (l : List α) (b : σ) (v : σ), P b → forIn l b f = .ok v → P v
  | [], b, v, hb, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h; exact hb
  | a :: l, b, v, hb, h => by
    rw [List.forIn_cons] at h
    simp only [bind, Except.bind] at h
    split at h
    · cases h
    · rename_i s hs
      obtain ⟨b', rfl, hb'⟩ := hf a b s hb hs
      exact forIn_inv P hf l b' v hb' h

open _root_.VStar.In in
/-- **A level's port inputs**: one ref per opening of the level in every port. -/
theorem levelIn_refs {src : Src} {prog : String} {level LANES H n : ℕ} {pin : PartIn}
    (h : levelIn src prog level LANES H n = .ok pin) : 0 < pin.size ∧ ∀ x ∈ pin.toList, x.2.refs.size = n := by
  unfold levelIn at h
  simp only [bind, Except.bind] at h
  split at h
  · rename_i hn
    split at h
    · split at h
      · rename_i ps hps
        split at h
        · cases h
        · split at h
          · simp only [pure, Except.pure, Except.ok.injEq] at h
            subst h
            simp only [beq_iff_eq] at hn
            have hp := array_mapM_size hps
            refine ⟨by simp, ?_⟩
            intro x hx
            simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
            rcases hx with rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> simp [ownRows, hn, hp]
          · cases h
      · cases h
    · cases h
  · cases h

open _root_.VStar.In in
/-- **An algebra part's port inputs**: one ref per rep in every port. -/
theorem algebraIn_refs {src : Src} {prog : String} {ports : List (String × ℕ)} {v : List ℕ} {pin : PartIn}
    (h : algebraIn src prog ports v = .ok pin) : 0 < pin.size ∧ ∀ x ∈ pin.toList, x.2.refs.size = 2 := by
  unfold algebraIn at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  · split at h
    · cases h
    · split at h
      · cases h
      · rename_i out hout
        have hinv := forIn_inv (fun (o : PartIn) => ∀ x ∈ o.toList, x.2.refs.size = 2) (fun a b s hb hs => by
          split at hs
          · simp only [pure, Except.pure, Except.ok.injEq] at hs
            subst hs
            refine ⟨_, rfl, fun x hx => ?_⟩
            simp only [Array.toList_push, List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at hx
            rcases hx with hx | rfl
            · exact hb x hx
            · simp [ownRows]
          · split at hs
            · simp only [pure, Except.pure, Except.ok.injEq] at hs
              subst hs
              refine ⟨_, rfl, fun x hx => ?_⟩
              simp only [Array.toList_push, List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at hx
              rcases hx with hx | rfl
              · exact hb x hx
              · simp [ownRows]
            · split at hs
              · split at hs
                · split at hs
                  · simp only [pure, Except.pure, Except.ok.injEq] at hs
                    subst hs
                    refine ⟨_, rfl, fun x hx => ?_⟩
                    simp only [Array.toList_push, List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at hx
                    rcases hx with hx | rfl
                    · exact hb x hx
                    · rfl
                  · cases hs
                · cases hs
              · cases hs) ports #[] out (by simp) hout
        simp only [pure, Except.pure, Except.ok.injEq] at h
        subst h
        refine ⟨by simp, fun x hx => ?_⟩
        simp only [Array.toList_push, List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at hx
        rcases hx with (hx | rfl) | rfl
        · exact hinv x hx
        · simp [ownRows]
        · simp [ownRows, Flock.RecCoins.zeroRows]

open _root_.VStar.In in
/-- **A cap's port inputs**: one ref per top node in every port. -/
theorem capIn_refs {src : Src} {level N : ℕ} {pin : PartIn} (h : capIn src level N = .ok pin) :
    0 < pin.size ∧ ∀ x ∈ pin.toList, x.2.refs.size = N := by
  unfold capIn at h
  cases ht : topsAt src level with
  | error e => rw [ht] at h; cases h
  | ok tops =>
    rw [ht] at h
    simp only [bind, Except.bind] at h
    split at h
    · rename_i hN
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      simp only [beq_iff_eq] at hN
      refine ⟨by simp, ?_⟩
      intro x hx
      simp at hx
      rcases hx with rfl | rfl <;> simp [ownRows, hN, Flock.RecCoins.zeroRows]
    · cases h

/-- **The stager's port inputs have one ref per instance in each port** (`TraceOk.refs`). -/
theorem partIn_refs {src : _root_.VStar.In.Src} {prog : String} {st : _root_.VStar.Stmt}
    {pin : _root_.VStar.In.PartIn} (h : _root_.VStar.In.partIn src prog st = .ok pin) :
    ∀ x ∈ pin.toList, x.2.refs.size = _root_.VStar.Trace.count pin := by
  have key : ∃ N, 0 < pin.size ∧ ∀ x ∈ pin.toList, x.2.refs.size = N := by
    unfold _root_.VStar.In.partIn at h
    split at h
    · exact ⟨_, levelIn_refs h⟩
    · exact ⟨_, algebraIn_refs h⟩
    · exact ⟨_, capIn_refs h⟩
    · cases h
  obtain ⟨N, hpos, hN⟩ := key
  have hc : _root_.VStar.Trace.count pin = N := by
    unfold _root_.VStar.Trace.count
    rw [Array.getElem?_eq_getElem hpos]
    simp only [Option.map_some, Option.getD_some]
    exact hN _ (Array.mem_toList_iff.mpr (Array.getElem_mem hpos))
  intro x hx
  rw [hc]
  exact hN x hx

/-! ## The circuit's fields, from the verifier's checks -/

/-- `TraceOk.rows`: every port's row at every instance is the port's words, under a 192-byte salt. -/
def RowsOk (c : Circuit) (pin : _root_.VStar.In.PartIn) : Prop :=
  ∀ q (hq : q < c.ports.size) n r, portRow pin c.ports[q].name n = some r →
    r.words.size = c.ports[q].words ∧ r.salt.size = _root_.Flock.HmRow.SALT_BYTES

theorem eq_single {α : Type} [Inhabited α] : ∀ {a : Array α}, a.size = 1 → a = #[a.getD 0 default]
  | ⟨[x]⟩, _ => by simp [Array.getD_eq_getD_getElem?]
  | ⟨[]⟩, h => by simp at h
  | ⟨_ :: _ :: _⟩, h => by simp at h

theorem portCols_single_size (g : Group) : (portCols #[g]).size = g.bits.size := by
  have h := FlockSoundness.inner_snd g.bits.toList #[] (g.col * WORD)
  rw [← FlockSoundness.Discharge.Aliased.portCols_one g] at h
  have hl := congrArg List.length h
  simp only [List.length_map, Array.length_toList, Array.toList_empty, List.map_nil, List.nil_append] at hl
  exact hl

theorem sum_of_all16 : ∀ l : List ℕ, (∀ b ∈ l, b = 16) → l.sum = 16 * l.length
  | [], _ => rfl
  | b :: l, h => by
    rw [List.sum_cons, List.length_cons, h b List.mem_cons_self,
      sum_of_all16 l fun x hx => h x (List.mem_cons_of_mem _ hx)]
    omega

variable {tags : Tags} {c : Circuit}

theorem scope_unit (h : ProvedScope.scopeOk c = true) :
    c.unit.inGroups.size = 1 ∧ c.outNet = c.unitNet ∧ c.unit.unitLog ≤ 32 := by
  unfold ProvedScope.scopeOk at h
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
  exact ⟨h.1.1.1.1, h.1.1.1.2, h.1.1.2⟩

theorem scope_ports (h : ProvedScope.scopeOk c = true) {q : ℕ} (hq : q < c.ports.size) :
    c.ports[q].segs = none ∧ c.ports[q].bits = 16 * c.ports[q].words := by
  unfold ProvedScope.scopeOk ProvedScope.v1Ports ProvedScope.v1Rows at h
  rw [Bool.and_eq_true] at h
  have := Array.all_eq_true.mp ((Bool.and_eq_true _ _).mp h.2).1 q hq
  simp only [Bool.and_eq_true, Bool.not_eq_true', Option.isNone_iff_eq_none, beq_iff_eq, decide_eq_true_eq] at this
  exact ⟨this.1.1.2, this.1.2⟩

theorem unit_ne_hm (P : FlockSoundness.ParseFacts tags c) (C : FlockSoundness.CheckFacts c) :
    c.unitNet ≠ c.hmNet := by
  intro he
  have h16 := C.unit_groups.2
  have hg := (FlockSoundness.Discharge.Hm.hmNet_groups P.rowNets).1
  rw [Circuit.unit, he, hg] at h16
  exact absurd (h16 512 (by decide)) (by decide)

theorem unit_ne_sha (P : FlockSoundness.ParseFacts tags c) (C : FlockSoundness.CheckFacts c) :
    c.unitNet ≠ c.shaNet := by
  intro he
  have h16 := C.unit_groups.2
  have hg := (FlockSoundness.Discharge.Hm.shaNet_groups P.rowNets).1
  rw [Circuit.unit, he, hg] at h16
  exact absurd (h16 512 (by decide)) (by decide)

/-- **`TraceOk` from the verifier's checks of a staged table**: `HmRow.parse`'s and `HmRow.check`'s facts, `scopeOk`,
V\*'s parse check and the setup's `kLog ≤ 27`, at a circuit of three nets, with the port inputs' refs and rows. -/
theorem traceOk_of {sv : _root_.VStar.Stmt} {pin : _root_.VStar.In.PartIn} (P : FlockSoundness.ParseFacts tags c)
    (hsc : ProvedScope.scopeOk c = true) (hcp : RecVStar.checkParsed sv c = .ok ()) (hk : c.kLog ≤ 27)
    (h3 : c.nets.size = 3) (hrefs : ∀ x ∈ pin.toList, x.2.refs.size = _root_.VStar.Trace.count pin)
    (hR : RowsOk c pin) : TraceOk c pin := by
  obtain ⟨ls, hls⟩ := P.check
  have C := FlockSoundness.check_facts P.typed hls
  obtain ⟨u, -, hupv, hcuts, -, -⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok hcp
  obtain ⟨hig1, ho, hul⟩ := scope_unit hsc
  have hu := FlockSoundness.unit_built P C
  have hhb : FlockSoundness.Built (hmN c) :=
    ⟨_, _, _, _, _, _, _, FlockSoundness.Discharge.Hm.hmNet_ofRows P.rowNets⟩
  have hsb : FlockSoundness.Built (c.nets.getD c.shaNet default).2 :=
    ⟨_, _, _, _, _, _, _, FlockSoundness.Discharge.Hm.shaNet_ofRows P.rowNets⟩
  have hus := unit_ne_sha P C
  have huh := unit_ne_hm P C
  have hhs : c.hmNet ≠ c.shaNet := fun he => FlockSoundness.Discharge.Hm.sha_ne_hm P.rowNets he.symm
  have hun := P.unitNet
  have hsn := P.shaNet
  have hhn := P.hmNet
  obtain ⟨mu, ku, cu⟩ := FlockSoundness.unit_range_facts_of P.unitNet C
  obtain ⟨mh, kh, -⟩ := FlockSoundness.rangeOf_spec (C.one_range _ P.hmNet)
  have hg : 0 < c.g := by
    have hp := FlockSoundness.Discharge.Layout.unit_count_pow2 P.typed hls ho
    rw [C.unit_range.2] at hp
    simp only [isPow2, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hp
    rcases Nat.eq_zero_or_pos c.g with h0 | h0
    · rw [h0, Nat.zero_mul] at hp; omega
    · exact h0
  have hpv : 0 < c.perVu.getD c.hmNet 0 := by
    rw [C.perVu_hm]; unfold nextPow2; exact Nat.two_pow_pos _
  have hlsz : (c.leavesIn.getD 0 #[]).size = (c.unit.inGroups.getD 0 default).bits.size := by
    have h0 : 0 < c.leavesIn.size := C.leaves.1 ▸ C.upv_pos
    have hs := C.leaves.2 0 C.upv_pos
    rw [getElem!_pos c.leavesIn 0 h0] at hs
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem h0, Option.getD_some]
    exact hs
  have hg0 : c.unit.inGroups.getD 0 default ∈ c.unit.inGroups.toList := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (by omega : 0 < c.unit.inGroups.size),
      Option.getD_some]
    exact Array.mem_toList_iff.mpr (Array.getElem_mem _)
  have h16 : ∀ b ∈ (c.unit.inGroups.getD 0 default).bits.toList, b = 16 := C.unit_groups.2
  have hwh : 2 ^ (c.nets.getD c.hmNet default).2.unitLog ≤ 2 ^ 32 := by
    have hsl : (c.rangeOf c.hmNet).slotLog = (c.nets.getD c.hmNet default).2.unitLog :=
      ((C.bound_slotLog _ mh).2 _ kh).symm
    have hb := (C.bound_slotLog _ mh).1
    have hsp := FlockSoundness.count_mul_le_spanH (c.rangeOf c.hmNet)
    have hc1 : 0 < (c.rangeOf c.hmNet).count := by rw [C.count _ mh _ kh]; exact Nat.mul_pos hg hpv
    have h1 : 2 ^ (c.rangeOf c.hmNet).slotLog ≤ (c.rangeOf c.hmNet).count * 2 ^ (c.rangeOf c.hmNet).slotLog :=
      Nat.le_mul_of_pos_left _ hc1
    have h2 : 2 ^ c.kLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by norm_num) (by omega)
    rw [← hsl]
    omega
  have hmIn2 : RecDeploy.hmIn c 2 = 1536 := by
    rw [VStarTrace.hmIn_eq (c := c), (FlockSoundness.Discharge.Hm.hmNet_groups P.rowNets).1]
    decide
  have hW : WORD = 128 := rfl
  exact {
    supported := by
      unfold _root_.VStar.Trace.supported
      simp only [Bool.and_eq_true]
      refine ⟨⟨⟨by rw [P.typed]; rfl, ?_⟩, ?_⟩, ?_⟩
      · rw [Array.all_eq_true]
        intro n hn
        have hl : (c.nets.getD n default).2.lookup = none := by
          rcases (by omega : n = c.unitNet ∨ n = c.shaNet ∨ n = c.hmNet) with rfl | rfl | rfl
          · exact (FlockSoundness.built_spec hu).1
          · exact (FlockSoundness.built_spec hsb).1
          · exact (FlockSoundness.built_spec hhb).1
        rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hn, Option.getD_some] at hl
        simp [hl]
      · rw [Array.all_eq_true]
        intro q hq
        simp [(scope_ports hsc hq).1]
      · rw [Array.all_eq_true]
        intro i hi
        have hw : c.wires[i] ∈ c.wires.toList := Array.mem_toList_iff.mpr (Array.getElem_mem hi)
        obtain ⟨hd, -, hp⟩ := C.wires _ hw
        have hne : c.wires[i].dst.1 ≠ c.unitNet := by
          intro he
          have hl := C.wire_leaf _ hw he
          rw [he] at hp
          have hpc : (portCols c.unit.inGroups).size = (c.unit.inGroups.getD 0 default).bits.size := by
            conv_lhs => rw [eq_single hig1]
            exact portCols_single_size _
          have : (portCols (c.nets.getD c.unitNet default).2.inGroups).size = (portCols c.unit.inGroups).size := rfl
          omega
        simp only [Bool.or_eq_true, beq_iff_eq]
        omega
    disjoint := by
      intro n₁ n₂ col h1 h2 hne hr1 hr2
      obtain ⟨m1, k1, -⟩ := FlockSoundness.rangeOf_spec (C.one_range n₁ h1)
      obtain ⟨m2, k2, -⟩ := FlockSoundness.rangeOf_spec (C.one_range n₂ h2)
      have s1 : FlockSoundness.InSpan (c.rangeOf n₁) col :=
        ⟨hr1.1, lt_of_lt_of_le hr1.2 (Nat.add_le_add_left (FlockSoundness.count_mul_le_spanH _) _)⟩
      have s2 : FlockSoundness.InSpan (c.rangeOf n₂) col :=
        ⟨hr2.1, lt_of_lt_of_le hr2.2 (Nat.add_le_add_left (FlockSoundness.count_mul_le_spanH _) _)⟩
      have he := FlockSoundness.eq_of_inSpan C.disj m1 m2 s1 s2
      rw [he, k2] at k1
      exact hne (Kind.net.inj k1).symm
    unit_lt := hun
    hm_lt := hhn
    unit_sha := hus
    unit_hm := huh
    hm_sha := hhs
    g_pos := hg
    upv_pos := by omega
    unit_count := le_of_eq cu.symm
    hm_count := le_of_eq (C.count _ mh _ kh).symm
    ports_hm := by rw [C.perVu_hm]; exact FlockSoundness.le_nextPow2 _
    unit_slot := ((C.bound_slotLog _ mu).2 _ ku).symm
    hm_slot := ((C.bound_slotLog _ mh).2 _ kh).symm
    unit_block := by
      have := (C.bound_slotLog _ mu).1
      have := FlockSoundness.count_mul_le_spanH c.unitRange
      omega
    hm_block := by
      have := (C.bound_slotLog _ mh).1
      have := FlockSoundness.count_mul_le_spanH (c.rangeOf c.hmNet)
      omega
    unit_const := (FlockSoundness.built_spec hu).2.2.2.1
    unit_log := (FlockSoundness.built_fit hu).1
    hm_const := (FlockSoundness.built_spec hhb).2.2.2.1
    hm_log := (FlockSoundness.built_fit hhb).1
    leaf_bits := fun j hj => by
      rw [hlsz] at hj
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hj, Option.getD_some]
      exact h16 _ (Array.mem_toList_iff.mpr (Array.getElem_mem hj))
    leaf_fit := by
      obtain ⟨h1, h2⟩ := (FlockSoundness.built_spec hu).2.2.2.2.1 _ hg0
      rw [← Array.foldl_toList, FlockSoundness.foldl_add_eq_sum, sum_of_all16 _ h16, Array.length_toList] at h1
      rw [hlsz]
      rw [hW] at h1 ⊢
      omega
    leaf_self := by
      intro k hk t ht
      obtain ⟨_, _, _, ig, _, ra, rb, hor⟩ := hu
      obtain ⟨-, ha, -, -, -, hig, -⟩ := Net.ofRows_ok hor
      have hin := (Net.ofRows_inputs hor).2 (c.unit.inGroups.getD 0 default) (by rw [← hig]; exact hg0)
        (16 * k + t) (by
          rw [← Array.foldl_toList, FlockSoundness.foldl_add_eq_sum, sum_of_all16 _ h16, Array.length_toList]
          omega)
      have hw : 2 ^ c.unit.unitLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by norm_num) hul
      rw [FlockSoundness.ofRows_row ha hw,
        show (c.unit.inGroups.getD 0 default).col * WORD + 16 * k + t =
          (c.unit.inGroups.getD 0 default).col * WORD + (16 * k + t) by omega, hin.1]
      rfl
    cuts := by
      intro x hx
      rw [hcuts] at hx
      simp at hx
    salt_fit := by
      rw [hmIn2]
      show 1536 + 8 * HmRow.SALT_BYTES ≤ (c.nets.getD c.hmNet default).2.inWords * WORD
      rw [FlockSoundness.Discharge.Hm.hmNet_inRows P.rowNets]
      decide
    salt_self := by
      intro j hj
      rw [hmIn2]
      obtain ⟨-, ha, -⟩ := Net.ofRows_ok (FlockSoundness.Discharge.Hm.hmNet_ofRows P.rowNets)
      show ((c.nets.getD c.hmNet default).2.a.row (1536 + j)).size = 1
      have hs : HmRow.SALT_BYTES = 192 := rfl
      rw [FlockSoundness.ofRows_row ha hwh, (FlockSoundness.Discharge.Hm.hmLaid_self (by omega)).1]
      rfl
    refs := hrefs
    rows := hR
    full := by
      unfold _root_.VStar.Trace.fullRows
      rw [Array.all_eq_true]
      intro q hq
      simp [(scope_ports hsc hq).2] }

/-! ## At the deployed statements -/

theorem partAt_some {x : Msg} {s : ℕ} {sv : _root_.VStar.Stmt} {c : Circuit} (hp : partAt x s = some (sv, c)) :
    ∃ r, readX x = some r ∧ (RecVStar.statements r.shape).toOption.bind (·[s]?) = some sv ∧
      r.w.parts[s]?.bind partCircuit = some c := by
  unfold partAt at hp
  cases hr : readX x with
  | none => rw [hr] at hp; simp at hp
  | some r =>
    rw [hr] at hp
    cases hst : (RecVStar.statements r.shape).toOption.bind (·[s]?) with
    | none => simp [hst] at hp
    | some st' =>
      cases hc' : r.w.parts[s]?.bind partCircuit with
      | none => simp [hst, hc'] at hp
      | some c' =>
        simp [hst, hc'] at hp
        obtain ⟨rfl, rfl⟩ := hp
        exact ⟨r, rfl, hst, hc'⟩

/-- **`TraceOk` at a staged part**: part `s`'s circuit at the compiled statement `x` (`partAt`) is the staged table's
(`partCircuit_staged`), which the verifier parsed (`parseFacts_of_parse`), scoped (`staged_scope`), held to V\*'s parse
(`stagedTables`) and set up with `kLog ≤ 27` (`table0`). Its three nets, its port inputs' refs and their rows are the
hypotheses. -/
theorem traceOk_refs {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {sv : _root_.VStar.Stmt}
    {c : Circuit} (hp : partAt x s = some (sv, c)) {pin : _root_.VStar.In.PartIn} (h3 : c.nets.size = 3)
    (hrefs : ∀ x ∈ pin.toList, x.2.refs.size = _root_.VStar.Trace.count pin) (hR : RowsOk c pin) :
    TraceOk c pin := by
  obtain ⟨r, hr, hsv, hc⟩ := partAt_some hp
  obtain ⟨p, r', sv', hx, ht, hpi, hr', ⟨ss, hss, hs⟩, h0, htab⟩ := stagedTables hS
  rw [hr] at hr'
  cases hr'
  have hsv' : ss[s]? = some sv := by
    rw [hss] at hsv
    exact hsv
  rw [hs] at hsv'
  cases hsv'
  have hpc := partCircuit_staged hS hx ht hpi hr hc h0
  obtain ⟨tags, P⟩ := Flock.Readable.parseFacts_of_parse hx ht hpi h0
  have hsc := (staged_scope hS hx h0).1
  have hcp := (htab 0 h0).2.2
  have hk : p.sts[0].c.kLog ≤ 27 := by
    have hx' := hx
    unfold Flock.Readable.parseBytes at hx'
    obtain ⟨rd, -, hrd⟩ := Option.bind_eq_some_iff.1 hx'
    obtain ⟨st0, h0', -, -, -, -, hk, -⟩ :=
      Flock.Readable.table0 (Flock.Readable.tablesOf_of_parses (Flock.Readable.parses_of_parse hrd)) ht hpi
    rw [Array.getElem?_eq_getElem h0] at h0'
    cases h0'
    exact hk
  rw [hpc] at P hsc hcp hk
  exact traceOk_of P hsc hcp hk h3 hrefs hR

/-- `traceOk_refs` at the stager's port inputs (`partIn_refs`). -/
theorem traceOk_deployed {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {sv : _root_.VStar.Stmt}
    {c : Circuit} (hp : partAt x s = some (sv, c)) {src : _root_.VStar.In.Src} {pin : _root_.VStar.In.PartIn}
    (hi : _root_.VStar.In.partIn src (progOf c) sv = .ok pin) (h3 : c.nets.size = 3) (hR : RowsOk c pin) :
    TraceOk c pin :=
  traceOk_refs hS hp h3 (partIn_refs hi) hR

end RecDeploy.VStarTraceOk
