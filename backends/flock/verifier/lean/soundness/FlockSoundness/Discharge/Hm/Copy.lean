/-
Δ's copies at the rows' inputs, in a satisfying witness. Each `cv`, chain and `pad` input bit of a row's slots is a self
row of its net (`HmNets.layout`), so its block row is `[x]` plus Δ's pairs there. Δ (`delta_rows`) writes there exactly
one copy `(x, x), (x, src)`: the rows' slots are distinct (`at_inj`, `slot_inj`) and the `sha512x3` and `hm96` nets are
two (`sha_ne_hm`). So the block row at `x` is `[src]`, and the block holds the copy (`inBit_copy`): the first `cv` is the
row prefix's midstate, each later `cv` the previous output, the `hm96` slot's `cv` the last output and its `pad` the
padding block (`row_copies_of_sat`), the constant bits read from the pin and the forced-zero row.
-/
import FlockSoundness.Discharge.Hm.Copies

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hm

open Compose Flock.DeriveAll Types.Dag

theorem atOf_succ {ports : Array Port} {p : ℕ} (hp : p < ports.size) :
    atOf ports (p + 1) = atOf ports p + ports[p]!.chunks := by
  unfold atOf
  rw [List.take_add_one, List.getElem?_eq_getElem (by simpa using hp), List.map_append, List.sum_append]
  simp [getElem!_pos ports p hp]

theorem atOf_mono {ports : Array Port} {p q : ℕ} (hpq : p ≤ q) (hq : q ≤ ports.size) :
    atOf ports p ≤ atOf ports q := by
  induction q with
  | zero => rw [Nat.le_zero.1 hpq]
  | succ q ih =>
      rcases Nat.lt_or_eq_of_le hpq with h | h
      · rw [atOf_succ (by omega)]
        exact le_trans (ih (by omega) (by omega)) (Nat.le_add_right _ _)
      · rw [h]

/-- Distinct compressions of the row are at distinct places `at_p + k`. -/
theorem at_inj {ports : Array Port} {p p' k k' : ℕ} (hp : p < ports.size) (hp' : p' < ports.size)
    (hk : k < ports[p]!.chunks) (hk' : k' < ports[p']!.chunks) (h : atOf ports p + k = atOf ports p' + k') :
    p = p' ∧ k = k' := by
  rcases lt_trichotomy p p' with hlt | rfl | hlt
  · have h1 := atOf_succ hp
    have h2 := atOf_mono (ports := ports) (show p + 1 ≤ p' by omega) hp'.le
    omega
  · exact ⟨rfl, by omega⟩
  · have h1 := atOf_succ hp'
    have h2 := atOf_mono (ports := ports) (show p' + 1 ≤ p by omega) hp.le
    omega

theorem slot_inj {m a b i j : ℕ} (hi : i < m) (hj : j < m) (h : a * m + i = b * m + j) : a = b ∧ i = j := by
  have hm : 0 < m := by omega
  have h1 : (a * m + i) / m = a := by rw [Nat.mul_comm, Nat.mul_add_div hm, Nat.div_eq_of_lt hi, Nat.add_zero]
  have h2 : (b * m + j) / m = b := by rw [Nat.mul_comm, Nat.mul_add_div hm, Nat.div_eq_of_lt hj, Nat.add_zero]
  have hab : a = b := by rw [← h1, h, h2]
  subst hab
  exact ⟨rfl, by omega⟩

theorem filter_flatMap2_single {β : Type} (G N : ℕ) (F : ℕ → ℕ → List (ℕ × β)) (x g₀ p₀ : ℕ) (hg : g₀ < G)
    (hp : p₀ < N) (hne : ∀ g < G, ∀ p < N, ¬ (g = g₀ ∧ p = p₀) → ∀ e ∈ F g p, e.1 ≠ x) :
    ((List.range G).flatMap fun g => (List.range N).flatMap fun p => F g p).filter (fun e => e.1 = x) =
      (F g₀ p₀).filter (fun e => e.1 = x) := by
  have hnil : ∀ g < G, ∀ p < N, ¬ (g = g₀ ∧ p = p₀) → (F g p).filter (fun e => e.1 = x) = [] :=
    fun g hg' p hp' hne' => List.filter_eq_nil_iff.2 fun e he => by simpa using hne g hg' p hp' hne' e he
  rw [filter_flatMap_single _ List.nodup_range _ _ g₀ (List.mem_range.2 hg) fun g hg' hne' =>
    filter_flatMap_none _ _ _ fun p hp' => hnil g (List.mem_range.1 hg') p (List.mem_range.1 hp') fun h => hne' h.1]
  exact filter_flatMap_single _ List.nodup_range _ _ p₀ (List.mem_range.2 hp) fun p hp' hne' =>
    hnil g₀ hg p (List.mem_range.1 hp') fun h => hne' h.2

theorem filter_flatMap2_none {β : Type} (G N : ℕ) (F : ℕ → ℕ → List (ℕ × β)) (x : ℕ)
    (hne : ∀ g < G, ∀ p < N, ∀ e ∈ F g p, e.1 ≠ x) :
    ((List.range G).flatMap fun g => (List.range N).flatMap fun p => F g p).filter (fun e => e.1 = x) = [] :=
  List.filter_eq_nil_iff.2 fun e he => by
    simp only [List.mem_flatMap, List.mem_range] at he
    obtain ⟨g, hg, p, hp, he⟩ := he
    simpa using hne g hg p hp e he

theorem filter_nil_of {β : Type} (L : List (ℕ × β)) (x : ℕ) (h : ∀ e ∈ L, e.1 ≠ x) :
    L.filter (fun e => e.1 = x) = [] :=
  List.filter_eq_nil_iff.2 fun e he => by simpa using h e he

/-! ## The rows' nets' inputs -/

theorem shaIn_cols {c : Circuit} (h : HmNets.check c = .ok ()) {s : ℕ} (hs : s < 3) :
    (portCols (c.nets.getD c.shaNet default).2.inGroups).getD (2 * s) default = (1536 * s, 512) ∧
      (portCols (c.nets.getD c.shaNet default).2.inGroups)[2 * s]! = (1536 * s, 512) := by
  rw [(shaNet_groups h).1]
  obtain _ | _ | _ | s := s
  · decide
  · decide
  · decide
  · omega

theorem hmIn_cols {c : Circuit} (h : HmNets.check c = .ok ()) :
    (portCols (c.nets.getD c.hmNet default).2.inGroups).getD 0 default = (0, 512) ∧
      (portCols (c.nets.getD c.hmNet default).2.inGroups)[1]! = (512, 1024) := by
  rw [(hmNet_groups h).1]
  decide

theorem sha_ne_hm {c : Circuit} (h : HmNets.check c = .ok ()) : c.shaNet ≠ c.hmNet := by
  intro he
  have h1 := shaNet_inRows h
  rw [he, hmNet_inRows h] at h1
  exact absurd h1 (by decide)

/-- The `sha512x3` net's input bits are self rows. -/
theorem shaLaid_self {off : ℕ} (ho : off < 4608) :
    (HmNets.shaLaid ()).ra.getD off #[] = #[off] ∧ (HmNets.shaLaid ()).rb.getD off #[] = #[off] := by
  rw [HmNets.shaLaid_eq]
  exact (HmNets.layout_rows _ _ _ _).2.1 off (by rw [sum_SHA_IN]; exact ho)

/-- The `hm96` net's input bits are self rows. -/
theorem hmLaid_self {off : ℕ} (ho : off < 3072) :
    (HmNets.hmLaid ()).ra.getD off #[] = #[off] ∧ (HmNets.hmLaid ()).rb.getD off #[] = #[off] := by
  rw [HmNets.hmLaid_eq]
  exact (HmNets.layout_rows _ _ _ _).2.1 off (by rw [sum_HM_IN]; exact ho)

theorem padCol_eq {c : Circuit} (h : HmNets.check c = .ok ()) (g p : ℕ) :
    padCol c g p = c.slotCol c.hmNet g p + 512 := by
  unfold padCol
  rw [(hmIn_cols h).2]

theorem chainDst_eq {c : Circuit} (h : HmNets.check c = .ok ()) {g p k : ℕ} (hk : shaSub c p k < 3) :
    Flat.wireDst c g (chainWire c.ports c.shaNet p k) = shaBase c g p k + shaIn c (2 * shaSub c p k) := by
  unfold Flat.wireDst chainWire shaBase shaIn shaSub at *
  simp only
  rw [(shaIn_cols h hk).1, (shaIn_cols h hk).2]

theorem chainWidth_eq {c : Circuit} (h : HmNets.check c = .ok ()) {p k : ℕ} (hk : shaSub c p k < 3) :
    Flat.wireWidth c (chainWire c.ports c.shaNet p k) = 512 := by
  unfold Flat.wireWidth chainWire shaSub at *
  simp only
  rw [(shaIn_cols h hk).1]

theorem hmWidth_eq {c : Circuit} (h : HmNets.check c = .ok ()) (p : ℕ) :
    Flat.wireWidth c (hmWire c.ports c.shaNet c.hmNet p) = 512 := by
  unfold Flat.wireWidth hmWire
  simp only
  rw [(hmIn_cols h).1]

theorem rowF_mem {c : Circuit} {pin g p : ℕ} {e : ℕ × ℕ}
    (he : e ∈ (List.range 512).map (fun t => (cvCol c g p + t, constSrc c pin (HmRow.midstate c.ports[p]!.words) t)) ++
      (List.range 1024).map fun t => (padCol c g p + t, constSrc c pin (HmRow.padWords c.ports[p]!.words) t)) :
    (∃ t < 512, e.1 = cvCol c g p + t) ∨ ∃ t < 1024, e.1 = padCol c g p + t := by
  simp only [List.mem_append, List.mem_map, List.mem_range] at he
  rcases he with ⟨t, ht, rfl⟩ | ⟨t, ht, rfl⟩
  · exact Or.inl ⟨t, ht, rfl⟩
  · exact Or.inr ⟨t, ht, rfl⟩

/-- The wires' copies, VU by VU and port by port. -/
def wireF (c : Circuit) (g p : ℕ) : List (ℕ × ℕ) :=
  ((List.range' 1 (((HmRow.rowSlots c.ports).2[p]!).size - 1)).map (chainWire c.ports c.shaNet p) ++
    [hmWire c.ports c.shaNet c.hmNet p]).flatMap fun w =>
      (List.range (Flat.wireWidth c w)).map fun t => (Flat.wireDst c g w + t, Flat.wireSrc c g w + t)

theorem wireCopies_eq {c : Circuit} (hw : c.wires = HmRow.rowWires c.ports c.shaNet c.hmNet) :
    wireCopies c = (List.range c.g).flatMap fun g =>
      (List.range (HmRow.rowSlots c.ports).2.size).flatMap fun p => wireF c g p := by
  unfold wireCopies wireF
  rw [hw, rowWires_eq]
  simp only [List.flatMap_assoc]

theorem wireF_mem {c : Circuit} {g p : ℕ} {e : ℕ × ℕ} (he : e ∈ wireF c g p) :
    (∃ k, 1 ≤ k ∧ k < ((HmRow.rowSlots c.ports).2[p]!).size ∧
      ∃ t < Flat.wireWidth c (chainWire c.ports c.shaNet p k),
        e.1 = Flat.wireDst c g (chainWire c.ports c.shaNet p k) + t) ∨
      ∃ t < Flat.wireWidth c (hmWire c.ports c.shaNet c.hmNet p),
        e.1 = Flat.wireDst c g (hmWire c.ports c.shaNet c.hmNet p) + t := by
  unfold wireF at he
  simp only [List.flatMap_append, List.flatMap_map, List.mem_append, List.mem_flatMap, List.mem_range',
    List.mem_map, List.mem_range, List.flatMap_cons, List.flatMap_nil, List.append_nil] at he
  rcases he with ⟨k, ⟨i, hi, rfl⟩, t, ht, rfl⟩ | ⟨t, ht, rfl⟩
  · exact Or.inl ⟨1 + 1 * i, by omega, by omega, t, ht, rfl⟩
  · exact Or.inr ⟨t, ht, rfl⟩

section Loc

variable {types : String → Option CircuitType} {ls : Array Flock.Layout} {info : String → Option TableInfo}
  {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D} {st : Stmt} {tb : TypedBlock}

set_option hygiene false in
local notation "𝒖" => done.getD unit default

theorem net_off_lt (T : TemplateLayout st tb words done 𝒖) {n : ℕ} (hn : n = st.c.shaNet ∨ n = st.c.hmNet)
    {l : Flock.HmNets.Laid} {name : String}
    (hl : Net.ofRows name l.useful l.constPos l.inGroups l.outGroups l.ra l.rb = .ok (st.c.nets.getD n default).2)
    {off : ℕ} (ho : off < (st.c.nets.getD n default).2.inWords * Flock.WORD) : off < 2 ^ (st.c.rangeOf n).slotLog := by
  obtain ⟨-, -, -, -, hcp, -, -, hin, -⟩ := Net.ofRows_ok hl
  rw [(T.range (rowNet_lt T hn)).2.2.2]
  have := T.fit n
  omega

/-- Bit `t` of port `p`'s compression `k`'s `cv` input, at VU `g`: in the `sha512x3` slot `(at_p + k) / 3`, at
`1536 · ((at_p + k) % 3) + t`. -/
theorem shaIn_at (T : TemplateLayout st tb words done 𝒖) (C : CheckFactsT st.c tb) (hck : HmNets.check st.c = .ok ())
    {g p k t : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size) (hk : k < st.c.ports[p]!.chunks) (ht : t < 512) :
    AtSlot st.c st.c.shaNet (g * st.c.perVu.getD st.c.shaNet 0 + ((HmRow.rowSlots st.c.ports).2[p]!)[k]!.1)
      (1536 * shaSub st.c p k + t) (shaBase st.c g p k + shaIn st.c (2 * shaSub st.c p k) + t) := by
  have hs := rowSlot_lt hp hk
  refine ⟨slot_lt_count T (Or.inl rfl) hg (by rw [C.perVu_sha]; exact hs.1),
    net_off_lt T (Or.inl rfl) (shaNet_ofRows hck) (by rw [shaNet_inRows hck]; unfold shaSub; omega), ?_⟩
  rw [(shaIn_eq hck (shaSub st.c p k) hs.2).1, shaBase, Circuit.slotCol, Nat.add_assoc]

/-- Bit `off` of port `p`'s `hm96` slot's inputs, at VU `g`. -/
theorem hmIn_at (T : TemplateLayout st tb words done 𝒖) (C : CheckFactsT st.c tb) (hck : HmNets.check st.c = .ok ())
    {g p off : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size) (ho : off < 4096) :
    AtSlot st.c st.c.hmNet (g * st.c.perVu.getD st.c.hmNet 0 + p) off (st.c.slotCol st.c.hmNet g p + off) :=
  ⟨slot_lt_count T (Or.inr rfl) hg (by rw [C.perVu_hm]; exact lt_of_lt_of_le hp (le_nextPow2 _)),
    net_off_lt T (Or.inr rfl) (hmNet_ofRows hck) (by rw [hmNet_inRows hck]; exact ho), rfl⟩

theorem sha_dec (T : TemplateLayout st tb words done 𝒖) (C : CheckFactsT st.c tb) (hck : HmNets.check st.c = .ok ())
    {g p k t g' p' k' t' : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size) (hk : k < st.c.ports[p]!.chunks)
    (ht : t < 512) (hg' : g' < st.c.g) (hp' : p' < st.c.ports.size) (hk' : k' < st.c.ports[p']!.chunks)
    (ht' : t' < 512)
    (h : shaBase st.c g p k + shaIn st.c (2 * shaSub st.c p k) + t =
      shaBase st.c g' p' k' + shaIn st.c (2 * shaSub st.c p' k') + t') :
    g = g' ∧ p = p' ∧ k = k' ∧ t = t' := by
  have hn := T.range T.shaNet.1
  obtain ⟨-, hs, ho⟩ := (shaIn_at T C hck hg hp hk ht).inj T.disj hn.1 hn.2.1 hn.1 hn.2.1
    (by rw [h]; exact shaIn_at T C hck hg' hp' hk' ht')
  have h1 := rowSlot_lt hp hk
  have h2 := rowSlot_lt hp' hk'
  obtain ⟨rfl, hs⟩ := slot_inj (by rw [C.perVu_sha]; exact h1.1) (by rw [C.perVu_sha]; exact h2.1) hs
  unfold shaSub at ho
  rw [rowSlots_entry hp hk, rowSlots_entry hp' hk'] at hs ho
  simp only at hs ho
  have : atOf st.c.ports p + k = atOf st.c.ports p' + k' := by omega
  obtain ⟨rfl, rfl⟩ := at_inj hp hp' hk hk' this
  exact ⟨rfl, rfl, rfl, by omega⟩

theorem hm_dec (T : TemplateLayout st tb words done 𝒖) (C : CheckFactsT st.c tb) (hck : HmNets.check st.c = .ok ())
    {g p off g' p' off' : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size) (ho : off < 4096) (hg' : g' < st.c.g)
    (hp' : p' < st.c.ports.size) (ho' : off' < 4096)
    (h : st.c.slotCol st.c.hmNet g p + off = st.c.slotCol st.c.hmNet g' p' + off') :
    g = g' ∧ p = p' ∧ off = off' := by
  have hn := T.range T.hmNet.1
  obtain ⟨-, hs, rfl⟩ := (hmIn_at T C hck hg hp ho).inj T.disj hn.1 hn.2.1 hn.1 hn.2.1
    (by rw [h]; exact hmIn_at T C hck hg' hp' ho')
  have hpv : ∀ q, q < st.c.ports.size → q < st.c.perVu.getD st.c.hmNet 0 := fun q hq => by
    rw [C.perVu_hm]; exact lt_of_lt_of_le hq (le_nextPow2 _)
  obtain ⟨rfl, rfl⟩ := slot_inj (hpv p hp) (hpv p' hp') hs
  exact ⟨rfl, rfl, rfl⟩

theorem sha_ne_hm_col (T : TemplateLayout st tb words done 𝒖) (C : CheckFactsT st.c tb)
    (hck : HmNets.check st.c = .ok ()) {g p k t g' p' off' : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size)
    (hk : k < st.c.ports[p]!.chunks) (ht : t < 512) (hg' : g' < st.c.g) (hp' : p' < st.c.ports.size)
    (ho' : off' < 4096) :
    shaBase st.c g p k + shaIn st.c (2 * shaSub st.c p k) + t ≠ st.c.slotCol st.c.hmNet g' p' + off' := by
  intro h
  have hn := T.range T.shaNet.1
  have hm := T.range T.hmNet.1
  exact sha_ne_hm hck ((shaIn_at T C hck hg hp hk ht).inj T.disj hn.1 hn.2.1 hm.1 hm.2.1
    (by rw [h]; exact hmIn_at T C hck hg' hp' ho')).1

end Loc

end FlockSoundness.Discharge.Hm
