/-
Δ's copies at the rows' inputs, in a satisfying witness. Each `cv`, chain and `pad` input bit of a row's slots is a self
row of its net (`HmNets.layout`), so its block row is `[x]` plus Δ's pairs there. Δ (`delta_rows`) writes there exactly
one copy `(x, x), (x, src)`: the rows' slots are distinct (`at_inj`, `slot_inj`) and the `sha512x3` and `hm96` nets are
two (`sha_ne_hm`). So the block row at `x` is `[src]`, and the block holds the copy (`inBit_copy`): the first `cv` is the
row prefix's midstate, each later `cv` the previous output, the `hm96` slot's `cv` the last output and its `pad` the
padding block (`row_copies_of_sat`), the constant bits read from the pin and the forced-zero row.
-/
import Proofs.Flock.Soundness.Discharge.Hm.Copies

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

/-- The `hm96` wire's destination column. -/
def hmWireDst (c : Circuit) (g p : ℕ) : ℕ := Flat.wireDst c g (hmWire c.ports c.shaNet c.hmNet p)

theorem hmWireDst_eq {c : Circuit} (h : HmNets.check c = .ok ()) (g p : ℕ) :
    hmWireDst c g p = c.slotCol c.hmNet g p := by
  unfold hmWireDst Flat.wireDst hmWire
  simp only
  rw [(hmIn_cols h).1, Nat.add_zero]

/-- The wires' copies, template instance by template instance and port by port. -/
def wireF (c : Circuit) (g p : ℕ) : List (ℕ × ℕ) :=
  ((List.range' 1 (((HmRow.rowSlots c.ports).2[p]!).size - 1)).map (chainWire c.ports c.shaNet p) ++
    [hmWire c.ports c.shaNet c.hmNet p]).flatMap fun w =>
      (List.range (Flat.wireWidth c w)).map fun t => (Flat.wireDst c g w + t, Flat.wireSrc c g w + t)

theorem wireCopies_eq {c : Circuit} (hw : c.wires = HmRow.rowWires c.ports c.shaNet c.hmNet)
    (hpos : ∀ q < c.ports.size, 0 < c.ports[q]!.chunks)
    (hs : ∀ q < c.ports.size, c.ports[q]!.segs = none) :
    wireCopies c = (List.range c.g).flatMap fun g =>
      (List.range (HmRow.rowSlots c.ports).2.size).flatMap fun p => wireF c g p := by
  unfold wireCopies wireF
  rw [hw, rowWires_eq _ _ _ hpos hs]
  simp only [List.flatMap_assoc]

/-- Δ's copies along wires `ws`. -/
def wireCopiesOf (c : Circuit) (ws : List Wire) : List (ℕ × ℕ) :=
  (List.range c.g).flatMap fun g => ws.flatMap fun w =>
    (List.range (Flat.wireWidth c w)).map fun t => (Flat.wireDst c g w + t, Flat.wireSrc c g w + t)

theorem wireCopiesOf_rows (c : Circuit) (hpos : ∀ q < c.ports.size, 0 < c.ports[q]!.chunks)
    (hs : ∀ q < c.ports.size, c.ports[q]!.segs = none) :
    wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList = (List.range c.g).flatMap fun g =>
      (List.range (HmRow.rowSlots c.ports).2.size).flatMap fun p => wireF c g p := by
  unfold wireCopiesOf wireF
  rw [rowWires_eq _ _ _ hpos hs]
  simp only [List.flatMap_assoc]

theorem rowSlots_size' (ports : Array Port) : (HmRow.rowSlots ports).2.size = ports.size := by
  rw [← Array.length_toList, rowSlots_eq, List.length_map, List.length_range]

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

/-- What the rows' slots need of an accepted statement's circuit, a typed template's (`ofT`) or an untyped one's
(`ofL`): its ranges (disjoint, in the block, of their nets' slots, `G · per_vu` slots each, one per net), its nets'
constants inside their slots, the rows' nets (the verifier's, `HmNets.check`) and their `per_vu`. -/
structure RowRanges (c : Circuit) : Prop where
  disj : c.ranges.toList.Pairwise Apart
  bound : ∀ r ∈ c.ranges.toList, r.start + spanH r ≤ 2 ^ c.kLog
  slotLog : ∀ r ∈ c.ranges.toList, ∀ i, r.kind = .net i → r.slotLog = (c.nets.getD i default).2.unitLog
  count : ∀ r ∈ c.ranges.toList, ∀ n, r.kind = .net n → r.count = c.g * c.perVu.getD n 0
  one_range : ∀ n < c.nets.size, (c.ranges.filter (·.kind == Kind.net n)).size = 1
  fit : ∀ i, (c.nets.getD i default).2.constPos < 2 ^ (c.nets.getD i default).2.unitLog
  shaNet : c.shaNet < c.nets.size
  hmNet : c.hmNet < c.nets.size
  perVu_sha : c.perVu.getD c.shaNet 0 = (HmRow.rowSlots c.ports).1
  perVu_hm : c.perVu.getD c.hmNet 0 = nextPow2 c.ports.size
  rowNets : HmNets.check c = .ok ()
  kLog : c.kLog ≤ 32

theorem RowRanges.ofT {st : Stmt} {tb : TypedBlock} {words : String → Option (Array ℕ)} {done : Array D} {u : D}
    {tags : Tags} {t : Typed.Template} (T : TemplateLayout st tb words done u) (C : CheckFactsT st.c tb)
    (P : ParseFactsT tags t st.c) : RowRanges st.c :=
  ⟨T.disj, T.bound, T.slotLog, T.count, T.one_range, T.fit, T.shaNet.1, T.hmNet.1, C.perVu_sha, C.perVu_hm, P.rowNets,
    T.kLog⟩

theorem RowRanges.ofL {st : Stmt} {tags : Tags} (L : Layout st) (P : ParseFacts tags st.c) (C : CheckFacts st.c) :
    RowRanges st.c :=
  ⟨L.disj, L.bound, L.slotLog, C.count, C.one_range, fun i => (L.fit i).1, P.shaNet, P.hmNet, C.perVu_sha, C.perVu_hm,
    P.rowNets, L.kLog⟩

namespace RowRanges

variable {c : Circuit} (R : RowRanges c)
include R

theorem range {n : ℕ} (hn : n < c.nets.size) :
    c.rangeOf n ∈ c.ranges.toList ∧ (c.rangeOf n).kind = .net n ∧ (c.rangeOf n).count = c.g * c.perVu.getD n 0 ∧
      (c.rangeOf n).slotLog = (c.nets.getD n default).2.unitLog := by
  obtain ⟨hm, hk, -⟩ := rangeOf_spec (R.one_range n hn)
  exact ⟨hm, hk, R.count _ hm _ hk, R.slotLog _ hm _ hk⟩

theorem rowNet_lt {n : ℕ} (hn : n = c.shaNet ∨ n = c.hmNet) : n < c.nets.size := by
  rcases hn with rfl | rfl
  · exact R.shaNet
  · exact R.hmNet

theorem slot_lt_count {n g i : ℕ} (hn : n = c.shaNet ∨ n = c.hmNet) (hg : g < c.g) (hi : i < c.perVu.getD n 0) :
    g * c.perVu.getD n 0 + i < (c.rangeOf n).count := by
  rw [(R.range (R.rowNet_lt hn)).2.2.1]
  calc g * c.perVu.getD n 0 + i < g * c.perVu.getD n 0 + c.perVu.getD n 0 := by omega
    _ = (g + 1) * c.perVu.getD n 0 := (Nat.succ_mul _ _).symm
    _ ≤ c.g * c.perVu.getD n 0 := Nat.mul_le_mul_right _ hg

theorem net_off_lt {n : ℕ} (hn : n = c.shaNet ∨ n = c.hmNet) {l : Flock.HmNets.Laid} {name : String}
    (hl : Net.ofRows name l.useful l.constPos l.inGroups l.outGroups l.ra l.rb = .ok (c.nets.getD n default).2)
    {off : ℕ} (ho : off < (c.nets.getD n default).2.inWords * Flock.WORD) : off < 2 ^ (c.rangeOf n).slotLog := by
  obtain ⟨-, -, -, -, hcp, -, -, hin, -⟩ := Net.ofRows_ok hl
  rw [(R.range (R.rowNet_lt hn)).2.2.2]
  have := R.fit n
  omega

/-- Bit `t` of port `p`'s compression `k`'s `cv` input, at template instance `g`: in the `sha512x3` slot
`(at_p + k) / 3`, at `1536 · ((at_p + k) % 3) + t`. -/
theorem shaIn_at {g p k t : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hk : k < c.ports[p]!.chunks) (ht : t < 512) :
    AtSlot c c.shaNet (g * c.perVu.getD c.shaNet 0 + ((HmRow.rowSlots c.ports).2[p]!)[k]!.1)
      (1536 * shaSub c p k + t) (shaBase c g p k + shaIn c (2 * shaSub c p k) + t) := by
  have hs := rowSlot_lt hp hk
  refine ⟨R.slot_lt_count (Or.inl rfl) hg (by rw [R.perVu_sha]; exact hs.1),
    R.net_off_lt (Or.inl rfl) (shaNet_ofRows R.rowNets) (by rw [shaNet_inRows R.rowNets]; unfold shaSub; omega), ?_⟩
  rw [(shaIn_eq R.rowNets (shaSub c p k) hs.2).1, shaBase, Circuit.slotCol, Nat.add_assoc]

/-- Bit `off` of port `p`'s `hm96` slot's inputs, at template instance `g`. -/
theorem hmIn_at {g p off : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (ho : off < 4096) :
    AtSlot c c.hmNet (g * c.perVu.getD c.hmNet 0 + p) off (c.slotCol c.hmNet g p + off) :=
  ⟨R.slot_lt_count (Or.inr rfl) hg (by rw [R.perVu_hm]; exact lt_of_lt_of_le hp (le_nextPow2 _)),
    R.net_off_lt (Or.inr rfl) (hmNet_ofRows R.rowNets) (by rw [hmNet_inRows R.rowNets]; exact ho), rfl⟩

theorem sha_dec {g p k t g' p' k' t' : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hk : k < c.ports[p]!.chunks)
    (ht : t < 512) (hg' : g' < c.g) (hp' : p' < c.ports.size) (hk' : k' < c.ports[p']!.chunks) (ht' : t' < 512)
    (h : shaBase c g p k + shaIn c (2 * shaSub c p k) + t = shaBase c g' p' k' + shaIn c (2 * shaSub c p' k') + t') :
    g = g' ∧ p = p' ∧ k = k' ∧ t = t' := by
  have hn := R.range R.shaNet
  obtain ⟨-, hs, ho⟩ := (R.shaIn_at hg hp hk ht).inj R.disj hn.1 hn.2.1 hn.1 hn.2.1
    (by rw [h]; exact R.shaIn_at hg' hp' hk' ht')
  have h1 := rowSlot_lt hp hk
  have h2 := rowSlot_lt hp' hk'
  obtain ⟨rfl, hs⟩ := slot_inj (by rw [R.perVu_sha]; exact h1.1) (by rw [R.perVu_sha]; exact h2.1) hs
  unfold shaSub at ho
  rw [rowSlots_entry hp hk, rowSlots_entry hp' hk'] at hs ho
  simp only at hs ho
  have : atOf c.ports p + k = atOf c.ports p' + k' := by omega
  obtain ⟨rfl, rfl⟩ := at_inj hp hp' hk hk' this
  exact ⟨rfl, rfl, rfl, by omega⟩

theorem hm_dec {g p off g' p' off' : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (ho : off < 4096) (hg' : g' < c.g)
    (hp' : p' < c.ports.size) (ho' : off' < 4096) (h : c.slotCol c.hmNet g p + off = c.slotCol c.hmNet g' p' + off') :
    g = g' ∧ p = p' ∧ off = off' := by
  have hn := R.range R.hmNet
  obtain ⟨-, hs, rfl⟩ := (R.hmIn_at hg hp ho).inj R.disj hn.1 hn.2.1 hn.1 hn.2.1
    (by rw [h]; exact R.hmIn_at hg' hp' ho')
  have hpv : ∀ q, q < c.ports.size → q < c.perVu.getD c.hmNet 0 := fun q hq => by
    rw [R.perVu_hm]; exact lt_of_lt_of_le hq (le_nextPow2 _)
  obtain ⟨rfl, rfl⟩ := slot_inj (hpv p hp) (hpv p' hp') hs
  exact ⟨rfl, rfl, rfl⟩

theorem sha_ne_hm_col {g p k t g' p' off' : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hk : k < c.ports[p]!.chunks)
    (ht : t < 512) (hg' : g' < c.g) (hp' : p' < c.ports.size) (ho' : off' < 4096) :
    shaBase c g p k + shaIn c (2 * shaSub c p k) + t ≠ c.slotCol c.hmNet g' p' + off' := by
  intro h
  have hn := R.range R.shaNet
  have hm := R.range R.hmNet
  exact sha_ne_hm R.rowNets ((R.shaIn_at hg hp hk ht).inj R.disj hn.1 hn.2.1 hm.1 hm.2.1
    (by rw [h]; exact R.hmIn_at hg' hp' ho')).1

/-- A row copy at a `cv` input bit of compression `k` is the first compression's, at its bit. -/
theorem rowF_ne_sha (hpos : ∀ q < c.ports.size, 0 < c.ports[q]!.chunks)
    {pin g' p' g p k t₀ : ℕ} (hg' : g' < c.g) (hp' : p' < c.ports.size) (hg : g < c.g)
    (hp : p < c.ports.size) (hk : k < c.ports[p]!.chunks) (ht₀ : t₀ < 512) {e : ℕ × ℕ}
    (he : e ∈ (List.range 512).map (fun t => (cvCol c g' p' + t, constSrc c pin (HmRow.midstate c.ports[p']!.words) t)) ++
      (List.range 1024).map fun t => (padCol c g' p' + t, constSrc c pin (HmRow.padWords c.ports[p']!.words) t))
    (hne : ¬ (g' = g ∧ p' = p ∧ k = 0)) : e.1 ≠ shaBase c g p k + shaIn c (2 * shaSub c p k) + t₀ := by
  rcases rowF_mem he with ⟨t, ht, he1⟩ | ⟨t, ht, he1⟩
  · rw [he1]
    intro h
    obtain ⟨h1, h2, h3, -⟩ := R.sha_dec hg' hp' (hpos p' hp') ht hg hp hk ht₀ h
    exact hne ⟨h1, h2, h3.symm⟩
  · rw [he1, padCol_eq R.rowNets, Nat.add_assoc]
    exact fun h => R.sha_ne_hm_col hg hp hk ht₀ hg' hp' (by omega) h.symm

/-- A wire copy at a `cv` input bit of compression `k` is the chain into it. -/
theorem wireF_ne_sha {g' p' g p k t₀ : ℕ} (hg' : g' < c.g) (hp' : p' < c.ports.size) (hg : g < c.g)
    (hp : p < c.ports.size) (hk : k < c.ports[p]!.chunks) (ht₀ : t₀ < 512) {e : ℕ × ℕ} (he : e ∈ wireF c g' p')
    (hne : ¬ (g' = g ∧ p' = p ∧ 1 ≤ k)) : e.1 ≠ shaBase c g p k + shaIn c (2 * shaSub c p k) + t₀ := by
  rcases wireF_mem he with ⟨k', hk1, hk2, t, ht, he1⟩ | ⟨t, ht, he1⟩
  · rw [rowSlots_size hp'] at hk2
    have hs := (rowSlot_lt hp' hk2).2
    rw [chainWidth_eq R.rowNets hs] at ht
    rw [he1, chainDst_eq R.rowNets hs]
    intro h
    obtain ⟨h1, h2, h3, -⟩ := R.sha_dec hg' hp' hk2 ht hg hp hk ht₀ h
    exact hne ⟨h1, h2, h3 ▸ hk1⟩
  · rw [hmWidth_eq R.rowNets] at ht
    rw [he1]
    show hmWireDst c g' p' + t ≠ _
    rw [hmWireDst_eq R.rowNets]
    exact fun h => R.sha_ne_hm_col hg hp hk ht₀ hg' hp' (by omega) h.symm

/-- A row copy at an `hm96` input bit at `off` is the `pad` copy at `off − 512`. -/
theorem rowF_ne_hm (hpos : ∀ q < c.ports.size, 0 < c.ports[q]!.chunks)
    {pin g' p' g p off : ℕ} (hg' : g' < c.g) (hp' : p' < c.ports.size) (hg : g < c.g)
    (hp : p < c.ports.size) (ho : off < 1536) {e : ℕ × ℕ}
    (he : e ∈ (List.range 512).map (fun t => (cvCol c g' p' + t, constSrc c pin (HmRow.midstate c.ports[p']!.words) t)) ++
      (List.range 1024).map fun t => (padCol c g' p' + t, constSrc c pin (HmRow.padWords c.ports[p']!.words) t))
    (hne : ¬ (g' = g ∧ p' = p ∧ 512 ≤ off)) : e.1 ≠ c.slotCol c.hmNet g p + off := by
  rcases rowF_mem he with ⟨t, ht, he1⟩ | ⟨t, ht, he1⟩
  · rw [he1]
    exact R.sha_ne_hm_col hg' hp' (hpos p' hp') ht hg hp (by omega)
  · rw [he1, padCol_eq R.rowNets, Nat.add_assoc]
    intro h
    obtain ⟨h1, h2, h3⟩ := R.hm_dec hg' hp' (by omega) hg hp (by omega) h
    exact hne ⟨h1, h2, by omega⟩

/-- A wire copy at an `hm96` input bit at `off` is the `hm96` wire, at `off`. -/
theorem wireF_ne_hm {g' p' g p off : ℕ} (hg' : g' < c.g) (hp' : p' < c.ports.size) (hg : g < c.g)
    (hp : p < c.ports.size) (ho : off < 1536) {e : ℕ × ℕ} (he : e ∈ wireF c g' p')
    (hne : ¬ (g' = g ∧ p' = p ∧ off < 512)) : e.1 ≠ c.slotCol c.hmNet g p + off := by
  rcases wireF_mem he with ⟨k', hk1, hk2, t, ht, he1⟩ | ⟨t, ht, he1⟩
  · rw [rowSlots_size hp'] at hk2
    have hs := (rowSlot_lt hp' hk2).2
    rw [chainWidth_eq R.rowNets hs] at ht
    rw [he1, chainDst_eq R.rowNets hs]
    exact R.sha_ne_hm_col hg' hp' hk2 ht hg hp (by omega)
  · rw [hmWidth_eq R.rowNets] at ht
    rw [he1]
    show hmWireDst c g' p' + t ≠ _
    rw [hmWireDst_eq R.rowNets]
    intro h
    obtain ⟨h1, h2, h3⟩ := R.hm_dec hg' hp' (by omega) hg hp (by omega) h
    exact hne ⟨h1, h2, by omega⟩

/-- **Δ's one copy into the first compression's `cv` bit `t₀`**: the row prefix's midstate bit, from the pin or the
forced-zero row. -/
theorem cv0_filter (hpos : ∀ q < c.ports.size, 0 < c.ports[q]!.chunks)
    (hseg : ∀ q < c.ports.size, c.ports[q]!.segs = none)
    {pin g p t₀ : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (ht₀ : t₀ < 512) :
    (rowCopies c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
        (fun e => e.1 = cvCol c g p + t₀) =
      [(cvCol c g p + t₀, constSrc c pin (HmRow.midstate c.ports[p]!.words) t₀)] := by
  have hsz := rowSlots_size' c.ports
  have h0 := hpos p hp
  have hw := filter_flatMap2_none _ _ (fun g p => wireF c g p) (cvCol c g p + t₀) fun g' hg' p' hp' e he =>
    R.wireF_ne_sha (k := 0) hg' (hsz ▸ hp') hg hp h0 ht₀ he fun h => absurd h.2.2 (by omega)
  have hr := filter_flatMap2_single _ _ _ (cvCol c g p + t₀) g p hg (hsz ▸ hp) fun g' hg' p' hp' hne e he =>
    R.rowF_ne_sha hpos (pin := pin) (k := 0) hg' (hsz ▸ hp') hg hp h0 ht₀ he fun h => hne ⟨h.1, h.2.1⟩
  have hpad := filter_map_none (List.range 1024) (padCol c g p)
    (constSrc c pin (HmRow.padWords c.ports[p]!.words)) (cvCol c g p + t₀) fun t ht => by
      have := List.mem_range.1 ht
      rw [padCol_eq R.rowNets, Nat.add_assoc]
      exact fun h => R.sha_ne_hm_col (k := 0) hg hp h0 ht₀ hg hp (by omega) h.symm
  rw [List.filter_append, wireCopiesOf_rows _ hpos hseg, hw, List.append_nil]
  unfold rowCopies
  rw [hr, List.filter_append, filter_map_single _ _ _ _ ht₀, hpad, List.append_nil]

/-- **Δ's one copy into compression `k ≥ 1`'s `cv` bit `t₀`**: the chain from compression `k − 1`'s output. -/
theorem chain_filter (hpos : ∀ q < c.ports.size, 0 < c.ports[q]!.chunks)
    (hseg : ∀ q < c.ports.size, c.ports[q]!.segs = none)
    {pin g p k t₀ : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hk1 : 1 ≤ k)
    (hk : k < c.ports[p]!.chunks) (ht₀ : t₀ < 512) :
    (rowCopies c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
        (fun e => e.1 = Flat.wireDst c g (chainWire c.ports c.shaNet p k) + t₀) =
      [(Flat.wireDst c g (chainWire c.ports c.shaNet p k) + t₀,
        Flat.wireSrc c g (chainWire c.ports c.shaNet p k) + t₀)] := by
  have hsz := rowSlots_size' c.ports
  have hs := (rowSlot_lt hp hk).2
  have hx := chainDst_eq (g := g) R.rowNets hs
  have hr := filter_flatMap2_none _ _ (fun g' p' =>
      (List.range 512).map (fun t => (cvCol c g' p' + t, constSrc c pin (HmRow.midstate c.ports[p']!.words) t)) ++
      (List.range 1024).map fun t => (padCol c g' p' + t, constSrc c pin (HmRow.padWords c.ports[p']!.words) t))
    (Flat.wireDst c g (chainWire c.ports c.shaNet p k) + t₀) fun g' hg' p' hp' e he => by
      rw [hx]; exact R.rowF_ne_sha hpos hg' (hsz ▸ hp') hg hp hk ht₀ he fun h => by omega
  have hw := filter_flatMap2_single _ _ (fun g p => wireF c g p)
    (Flat.wireDst c g (chainWire c.ports c.shaNet p k) + t₀) g p hg (hsz ▸ hp) fun g' hg' p' hp' hne e he => by
      rw [hx]; exact R.wireF_ne_sha hg' (hsz ▸ hp') hg hp hk ht₀ he fun h => hne ⟨h.1, h.2.1⟩
  have hkm : k ∈ List.range' 1 (((HmRow.rowSlots c.ports).2[p]!).size - 1) := by
    rw [List.mem_range'_1, rowSlots_size hp]; omega
  have hc := filter_flatMap_single _ List.nodup_range' (fun k => (List.range (Flat.wireWidth c
      (chainWire c.ports c.shaNet p k))).map fun t => (Flat.wireDst c g (chainWire c.ports c.shaNet p k) + t,
        Flat.wireSrc c g (chainWire c.ports c.shaNet p k) + t))
    (fun e => decide (e.1 = Flat.wireDst c g (chainWire c.ports c.shaNet p k) + t₀)) k hkm
    fun k' hk' hne => filter_nil_of _ _ fun e he => by
      rw [List.mem_range'_1, rowSlots_size hp] at hk'
      have hs' := (rowSlot_lt hp (show k' < c.ports[p]!.chunks by omega)).2
      simp only [List.mem_map, List.mem_range] at he
      obtain ⟨t, ht, rfl⟩ := he
      rw [chainWidth_eq R.rowNets hs'] at ht
      rw [chainDst_eq R.rowNets hs', hx]
      intro h
      exact hne (R.sha_dec hg hp (show k' < c.ports[p]!.chunks by omega) ht hg hp hk ht₀ h).2.2.1
  have hh := filter_nil_of ((List.range (Flat.wireWidth c (hmWire c.ports c.shaNet c.hmNet p))).map fun t =>
      (Flat.wireDst c g (hmWire c.ports c.shaNet c.hmNet p) + t, Flat.wireSrc c g (hmWire c.ports c.shaNet c.hmNet p) + t))
    (Flat.wireDst c g (chainWire c.ports c.shaNet p k) + t₀) fun e he => by
      simp only [List.mem_map, List.mem_range] at he
      obtain ⟨t, ht, rfl⟩ := he
      rw [hmWidth_eq R.rowNets] at ht
      show hmWireDst c g p + t ≠ _
      rw [hmWireDst_eq R.rowNets, hx]
      exact fun h => R.sha_ne_hm_col hg hp hk ht₀ hg hp (by omega) h.symm
  rw [List.filter_append]
  unfold rowCopies
  rw [hr, List.nil_append, wireCopiesOf_rows _ hpos hseg, hw]
  unfold wireF
  rw [List.flatMap_append, List.filter_append, List.flatMap_map, hc, List.flatMap_cons, List.flatMap_nil,
    List.append_nil, hh, List.append_nil, chainWidth_eq R.rowNets hs, filter_map_single _ _ _ _ ht₀]

/-- **Δ's one copy into the `hm96` slot's `cv` bit `t₀`**: the wire from the last compression's output. -/
theorem hmcv_filter (hpos : ∀ q < c.ports.size, 0 < c.ports[q]!.chunks)
    (hseg : ∀ q < c.ports.size, c.ports[q]!.segs = none)
    {pin g p t₀ : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (ht₀ : t₀ < 512) :
    (rowCopies c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
        (fun e => e.1 = Flat.wireDst c g (hmWire c.ports c.shaNet c.hmNet p) + t₀) =
      [(Flat.wireDst c g (hmWire c.ports c.shaNet c.hmNet p) + t₀,
        Flat.wireSrc c g (hmWire c.ports c.shaNet c.hmNet p) + t₀)] := by
  have hsz := rowSlots_size' c.ports
  have hx : Flat.wireDst c g (hmWire c.ports c.shaNet c.hmNet p) = c.slotCol c.hmNet g p :=
    hmWireDst_eq R.rowNets g p
  have hr := filter_flatMap2_none _ _ (fun g' p' =>
      (List.range 512).map (fun t => (cvCol c g' p' + t, constSrc c pin (HmRow.midstate c.ports[p']!.words) t)) ++
      (List.range 1024).map fun t => (padCol c g' p' + t, constSrc c pin (HmRow.padWords c.ports[p']!.words) t))
    (Flat.wireDst c g (hmWire c.ports c.shaNet c.hmNet p) + t₀) fun g' hg' p' hp' e he => by
      rw [hx]; exact R.rowF_ne_hm hpos hg' (hsz ▸ hp') hg hp (by omega) he fun h => by omega
  have hw := filter_flatMap2_single _ _ (fun g p => wireF c g p)
    (Flat.wireDst c g (hmWire c.ports c.shaNet c.hmNet p) + t₀) g p hg (hsz ▸ hp) fun g' hg' p' hp' hne e he => by
      rw [hx]; exact R.wireF_ne_hm hg' (hsz ▸ hp') hg hp (by omega) he fun h => hne ⟨h.1, h.2.1⟩
  have hc := filter_nil_of (((List.range' 1 (((HmRow.rowSlots c.ports).2[p]!).size - 1)).map
      (chainWire c.ports c.shaNet p)).flatMap fun w =>
        (List.range (Flat.wireWidth c w)).map fun t => (Flat.wireDst c g w + t, Flat.wireSrc c g w + t))
    (Flat.wireDst c g (hmWire c.ports c.shaNet c.hmNet p) + t₀) fun e he => by
      simp only [List.flatMap_map, List.mem_flatMap, List.mem_range'_1, List.mem_map, List.mem_range] at he
      obtain ⟨k', hk', t, ht, rfl⟩ := he
      rw [rowSlots_size hp] at hk'
      have hk2 : k' < c.ports[p]!.chunks := by omega
      have hs' := (rowSlot_lt hp hk2).2
      rw [chainWidth_eq R.rowNets hs'] at ht
      rw [chainDst_eq R.rowNets hs', hx]
      exact R.sha_ne_hm_col hg hp hk2 ht hg hp (by omega)
  rw [List.filter_append]
  unfold rowCopies
  rw [hr, List.nil_append, wireCopiesOf_rows _ hpos hseg, hw]
  unfold wireF
  rw [List.flatMap_append, List.filter_append, hc, List.nil_append, List.flatMap_cons, List.flatMap_nil,
    List.append_nil, hmWidth_eq R.rowNets, filter_map_single _ _ _ _ ht₀]

/-- **Δ's one copy into the `hm96` slot's `pad` bit `t₀`**: the padding block's bit, from the pin or the forced-zero
row. -/
theorem pad_filter (hpos : ∀ q < c.ports.size, 0 < c.ports[q]!.chunks)
    (hseg : ∀ q < c.ports.size, c.ports[q]!.segs = none)
    {pin g p t₀ : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (ht₀ : t₀ < 1024) :
    (rowCopies c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
        (fun e => e.1 = padCol c g p + t₀) =
      [(padCol c g p + t₀, constSrc c pin (HmRow.padWords c.ports[p]!.words) t₀)] := by
  have hsz := rowSlots_size' c.ports
  have h0 := hpos p hp
  have hx : padCol c g p + t₀ = c.slotCol c.hmNet g p + (512 + t₀) := by
    rw [padCol_eq R.rowNets, Nat.add_assoc]
  have hw := filter_flatMap2_none _ _ (fun g p => wireF c g p) (padCol c g p + t₀) fun g' hg' p' hp' e he => by
    rw [hx]; exact R.wireF_ne_hm hg' (hsz ▸ hp') hg hp (by omega) he fun h => by omega
  have hr := filter_flatMap2_single _ _ _ (padCol c g p + t₀) g p hg (hsz ▸ hp) fun g' hg' p' hp' hne e he => by
    rw [hx]; exact R.rowF_ne_hm hpos (pin := pin) hg' (hsz ▸ hp') hg hp (by omega) he fun h => hne ⟨h.1, h.2.1⟩
  have hcv := filter_map_none (List.range 512) (cvCol c g p)
    (constSrc c pin (HmRow.midstate c.ports[p]!.words)) (padCol c g p + t₀) fun t ht => by
      rw [hx]
      exact R.sha_ne_hm_col (k := 0) hg hp h0 (List.mem_range.1 ht) hg hp (by omega)
  rw [List.filter_append, wireCopiesOf_rows _ hpos hseg, hw, List.append_nil]
  unfold rowCopies
  rw [hr, List.filter_append, hcv, List.nil_append, filter_map_single _ _ _ _ ht₀]

end RowRanges

theorem blockBit_of_lt (S : Model.Statement) (z : Model.Witness S.m) (o : Fin (2 ^ S.nbl)) {y : ℕ}
    (hy : y < 2 ^ S.kLog) : blockBit S z o y = FlockSoundness.bit (S.block z o ⟨y, hy⟩) := by
  unfold blockBit
  split
  · rfl
  · exact absurd hy ‹_›

/-- **The block rows at an input bit of a row net's slot** with one copy `(x, src)`: at a self row `x` of a
`sha512x3` or `hm96` slot below the net's input rows, where Δ_A and Δ_B write `(x, x)` and `(x, src)` and nothing else,
the level-0 rows are `[src]`, in A and in B (zero when `src` is past the block). -/
theorem inBit_rows {st : Stmt} (R : RowRanges st.c) {n s off x src : ℕ} (hn : n = st.c.shaNet ∨ n = st.c.hmNet)
    (hat : AtSlot st.c n s off x) {l : Flock.HmNets.Laid} {name : String}
    (hl : Net.ofRows name l.useful l.constPos l.inGroups l.outGroups l.ra l.rb = .ok (st.c.nets.getD n default).2)
    (hself : l.ra.getD off #[] = #[off] ∧ l.rb.getD off #[] = #[off]) (hx : x < 2 ^ st.c.kLog)
    (hδ : ∀ j, deltaEntry st.da x j = (((if j = x then 1 else 0) + (if j = src then 1 else 0) : ℕ) : ZMod 2) ∧
      deltaEntry st.db x j = (((if j = x then 1 else 0) + (if j = src then 1 else 0) : ℕ) : ZMod 2))
    (j : Fin (2 ^ st.c.kLog)) :
    placedA st ⟨x, hx⟩ j = (if j.val = src then 1 else 0) ∧ placedB st ⟨x, hx⟩ j = (if j.val = src then 1 else 0) := by
  have hnn := R.rowNet_lt hn
  obtain ⟨hm, hk, -, hsl⟩ := R.range hnn
  have h32 : (st.c.nets.getD n default).2.unitLog ≤ 32 := by
    rw [← hsl]; exact le_trans (hat.slotLog_le R.bound hm) R.kLog
  have hw : 2 ^ (st.c.nets.getD n default).2.unitLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by norm_num) h32
  obtain ⟨-, ha, hb, -, -, -, -, -, -, -, -, -, hlk, -⟩ := Net.ofRows_ok hl
  have hxe := hat.2.2
  have hE : ∀ j, rangesEntry st (slotEntryA st) x j + deltaEntry st.da x j =
        (([x].count j + ((if j = x then 1 else 0) + (if j = src then 1 else 0)) : ℕ) : ZMod 2) ∧
      rangesEntry st (slotEntryB st) x j + deltaEntry st.db x j =
        (([x].count j + ((if j = x then 1 else 0) + (if j = src then 1 else 0)) : ℕ) : ZMod 2) := by
    intro j
    have hre := fun e => rangesEntry_at R.disj hm hk hat e j
    have hla := ofRows_cols_lt ha off
    have hlb := ofRows_cols_lt hb off
    have hxa : [x] = (l.ra.getD off #[]).toList.map ((st.c.rangeOf n).slot s + ·) := by rw [hself.1, hxe]; rfl
    have hxb : [x] = (l.rb.getD off #[]).toList.map ((st.c.rangeOf n).slot s + ·) := by rw [hself.2, hxe]; rfl
    refine ⟨?_, ?_⟩
    · rw [hre, (hδ j).1, hxa]
      show (if _ then sparseEntry (st.c.nets.getD n default).2.a off _ else 0) + _ = _
      rw [sparseEntry_ofRows ha, ofRows_row ha hw, count_slot _ (by rw [hsl]; exact hla) j]
      push_cast; ring
    · rw [hre, (hδ j).2, hxb]
      show (if _ then sparseEntry (st.c.nets.getD n default).2.b off _ +
        lookupEntry (st.c.nets.getD n default).2.lookup off _ else 0) + _ = _
      rw [hlk]
      show (if _ then sparseEntry (st.c.nets.getD n default).2.b off _ + 0 else 0) + _ = _
      rw [add_zero, sparseEntry_ofRows hb, ofRows_row hb hw, count_slot _ (by rw [hsl]; exact hlb) j]
      push_cast; ring
  have hval : (([x].count j.val + ((if j.val = x then 1 else 0) + (if j.val = src then 1 else 0)) : ℕ) : ZMod 2) =
      if j.val = src then 1 else 0 := by
    by_cases h1 : j.val = x <;> by_cases h2 : j.val = src
    · rw [ite_eq_left h1, ite_eq_left h2, ite_eq_left h2, h1, List.count_singleton_self]; decide
    · rw [ite_eq_left h1, ite_eq_right h2, ite_eq_right h2, h1, List.count_singleton_self]; decide
    · rw [ite_eq_right h1, ite_eq_left h2, ite_eq_left h2, List.count_eq_zero.2 (by simpa using h1)]; decide
    · rw [ite_eq_right h1, ite_eq_right h2, ite_eq_right h2, List.count_eq_zero.2 (by simpa using h1)]; decide
  exact ⟨(hE j.val).1.trans hval, (hE j.val).2.trans hval⟩

/-- A row `[src]` in A and in B (zero when `src` is past the block) holds what `src` holds, in every block of a
satisfying witness. -/
theorem block_of_rows {S : Model.Statement} {z : Model.Witness S.m} (hz : S.Satisfies z) (o : Fin (2 ^ S.nbl))
    {x src : ℕ} (hx : x < 2 ^ S.kLog)
    (h : ∀ j : Fin (2 ^ S.kLog), S.A₀ ⟨x, hx⟩ j = (if j.val = src then 1 else 0) ∧
      S.B₀ ⟨x, hx⟩ j = (if j.val = src then 1 else 0)) :
    blockBit S z o x = blockBit S z o src := by
  by_cases hsrc : src < 2 ^ S.kLog
  · have he : ∀ j : Fin (2 ^ S.kLog), (if j.val = src then (1 : ZMod 2) else 0) =
        if j = (⟨src, hsrc⟩ : Fin (2 ^ S.kLog)) then 1 else 0 := fun j => by simp [Fin.ext_iff]
    have hcopy : CopyRow S ⟨x, hx⟩ ⟨src, hsrc⟩ := ⟨fun j => (h j).1.trans (he j), fun j => (h j).2.trans (he j)⟩
    rw [blockBit_of_lt _ z o hx, blockBit_of_lt _ z o hsrc, hcopy.block_eq hz o]
  · have hne : ∀ j : Fin (2 ^ S.kLog), (if j.val = src then (1 : ZMod 2) else 0) = 0 :=
      fun j => ite_eq_right fun (e : j.val = src) => hsrc (e ▸ j.isLt)
    have hzr : ZeroRow S ⟨x, hx⟩ := ⟨fun j => (h j).1.trans (hne j), fun j => (h j).2.trans (hne j)⟩
    rw [blockBit_of_lt _ z o hx, hzr.block_eq hz o]
    unfold blockBit
    simp only [hsrc, dite_false]

theorem blockBit_constSrc {S : Model.Statement} {z : Model.Witness S.m} {o : Fin (2 ^ S.nbl)} {c : Circuit}
    {pin : ℕ} (ws : Array UInt64) (t : ℕ) (hpin : blockBit S z o pin = true)
    (hzero : blockBit S z o (zeroCol c) = false) : blockBit S z o (constSrc c pin ws t) = wordsBit ws t := by
  unfold constSrc
  cases h : wordsBit ws t <;> simp [hpin, hzero]

/-- Δ's flat pairs at an input bit of a row net's slot with one copy `(x, src)`: `(x, x)` and `(x, src)`, when the slot
constants' pairs, the leaf pairs and the cut pairs write nothing there. -/
theorem flatPairs_count {c : Circuit} {pin x src : ℕ} (hC : ∀ j, (constPairs c pin).count (x, j) = 0)
    (hleaf : ∀ j, (Flat.leafPairs c).count (x, j) = 0) (hcut : ∀ j, (Flat.cutPairs c).count (x, j) = 0)
    (hf : (rowCopies c pin ++ wireCopies c).filter (fun e => e.1 = x) = [(x, src)]) (j : ℕ) :
    (flatPairs c pin).count (x, j) = (if j = x then 1 else 0) + (if j = src then 1 else 0) := by
  simp only [flatPairs, rowPairs_eq, wirePairs_eq, List.count_append, hC, hleaf, hcut, Nat.zero_add, Nat.add_zero]
  rw [← List.count_append, ← List.flatMap_append, count_pair2, hf]
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero]

section Loc

variable {types : String → Option CircuitType} {ls : Array Flock.Layout} {info : String → Option TableInfo}
  {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D} {st : Stmt} {tb : TypedBlock}

set_option hygiene false in
local notation "𝒖" => done.getD unit default

/-- **An input bit of a row net's slot holds its one copy**, in a typed template's statement. At a self row `x` of a
`sha512x3` or `hm96` slot below the net's input rows, where Δ's copies (`rowCopies`, `wireCopies`) write exactly one
`(x, src)`, every block of a satisfying witness holds at `x` what it holds at `src`. -/
theorem inBit_copy {tags : Tags} {t : Typed.Template} (hd : deriveChecked types ls info words unit = .ok done)
    (hul : unit < ls.size) (T : TemplateLayout st tb words done 𝒖) (U : UnitShape done 𝒖)
    (P : ParseFactsT tags t st.c) (C : CheckFactsT st.c tb)
    (hv1 : ∀ p ∈ st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    (hda : st.da = (HmRow.delta st.c st.pin).1) (hdb : st.db = (HmRow.delta st.c st.pin).2)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (T.model regs).m}
    (hz : (T.model regs).Satisfies z) (o : Fin (2 ^ (T.model regs).nbl)) {n s off x src : ℕ}
    (hn : n = st.c.shaNet ∨ n = st.c.hmNet) (hat : AtSlot st.c n s off x) {l : Flock.HmNets.Laid} {name : String}
    (hl : Net.ofRows name l.useful l.constPos l.inGroups l.outGroups l.ra l.rb = .ok (st.c.nets.getD n default).2)
    (hoff : off < (st.c.nets.getD n default).2.inWords * Flock.WORD)
    (hself : l.ra.getD off #[] = #[off] ∧ l.rb.getD off #[] = #[off])
    (hf : (rowCopies st.c st.pin ++ wireCopies st.c).filter (fun e => e.1 = x) = [(x, src)]) :
    blockBit (T.model regs) z o x = blockBit (T.model regs) z o src := by
  have R := RowRanges.ofT T C P
  have hnn := R.rowNet_lt hn
  obtain ⟨hm, hk, -, -⟩ := R.range hnn
  obtain ⟨-, -, -, -, hcp, -, -, hin, -⟩ := Net.ofRows_ok hl
  have hy : ∀ g' < st.c.g, ∀ u, InUnit done 𝒖 u → NotConst 𝒖 u → Typed.col st.c tb g' u ≠ x := by
    intro g' hg' u hu _ hux
    rcases T.at_inUnit hd hg' hu with ⟨-, hat'⟩ | ⟨i, hi, hiu, -, -, -, hat'⟩
    · obtain ⟨h1, -⟩ := hat'.inj T.disj (T.range T.unitNet).1 (T.range T.unitNet).2.1 hm hk (hux ▸ hat)
      rcases hn with rfl | rfl
      · exact T.shaNet.2 h1.symm
      · exact T.hmNet.2 h1.symm
    · obtain ⟨-, hpn, -, hps, hph, -⟩ := T.part i hi hiu
      obtain ⟨h1, -⟩ := hat'.inj T.disj (T.range hpn).1 (T.range hpn).2.1 hm hk (hux ▸ hat)
      rcases hn with rfl | rfl
      · exact hps h1
      · exact hph h1
  have hC : ∀ j, (constPairs st.c st.pin).count (x, j) = 0 := fun j => by
    rw [constPairs_count T.disj T.slotLog T.fit hm hk hat j, ite_eq_right (fun h => by have := h.1; omega)]
  have hF := flatPairs_count hC (fun j => by rw [leafPairs_nil P.leavesIn P.upv]; rfl)
    (fun j => by rw [cutPairs_nil P.leafCuts]; rfl) hf
  have hδ : ∀ j, deltaEntry st.da x j = (((if j = x then 1 else 0) + (if j = src then 1 else 0) : ℕ) : ZMod 2) ∧
      deltaEntry st.db x j = (((if j = x then 1 else 0) + (if j = src then 1 else 0) : ℕ) : ZMod 2) := by
    intro j
    obtain ⟨h1, h2⟩ := delta_rows st.c st.pin T.typed hv1 P.portWords P.v1Segs
    rw [deltaEntry_eq_count, deltaEntry_eq_count, hda, hdb, h1, h2, List.count_append, List.count_append, hF,
      T.tail_zero hd hul U hy true, T.tail_zero hd hul U hy false, Nat.add_zero]
    exact ⟨rfl, rfl⟩
  have hx : x < 2 ^ (T.model regs).kLog := hat.lt T.bound hm
  exact block_of_rows hz o hx fun j => inBit_rows R hn hat hl hself hx hδ j

end Loc

end FlockSoundness.Discharge.Hm
