import Proofs.Flock.Soundness.Discharge.Layout.Region
import Proofs.Flock.Soundness.ExecSetup

/-!
# The `b ‖ c` that the statement pins at the `hm96` slots' outputs

`HmRow.regions` pins, for each input port `p`, every template instance's `hm96` output `b ‖ c` (128 bytes) to the public
digest of the template instance's input (the pinned dummy past the population): region "digest p", whose free bits are a
byte string's 10 bit-index bits and the template instance's `log₂ g` bits, at the fixed address `hmFixed c p`
(`regions_digest`). So in a satisfying witness of the model's statement at the executable's regions, the 128 bytes at
template instance `g`'s `hm96` output of port `p` are that digest (`bc_pinned`). The address is the `hm96` slot's
(`bcAddr_eq_slotCol`), and setupH's statement's regions are `HmRow.regions` (`setupH_regions`).
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.Layout

open Flock Model Refine Binding

/-! ### Region "digest p" -/

/-- Template instance `g`'s `b ‖ c` of port `p` at block `b` that region "digest p" pins: the digest of instance
`b·g + g`, or the pinned dummy past the population. -/
def bcChunk (c : Circuit) (pub : Public) (b g p : ℕ) : ByteArray :=
  match c.instOf pub.n b g with
  | some i => pub.digest i p c.ports.size
  | none => c.dummyDigests.getD p ByteArray.empty

/-- Region "digest p"'s free bits: the 10 bits of a bit's index in a template instance's 128 bytes, then the template
instance's bits, above the port's slot bits. -/
def hmFree (c : Circuit) : Array ℕ :=
  Array.map (fun x => 0 + x) (Array.range 10) ++
    Array.map (fun x => (c.rangeOf c.hmNet).slotLog + (c.perVu.getD c.hmNet 1).log2 + x) (Array.range c.g.log2)

/-- Region "digest p"'s fixed address: port `p`'s `hm96` slot of template instance 0, at the `hm96` net's first output
column. -/
def hmFixed (c : Circuit) (p : ℕ) : ℕ :=
  (c.rangeOf c.hmNet).start + p * 2 ^ (c.rangeOf c.hmNet).slotLog +
    ((portCols (c.nets.getD c.hmNet default).2.outGroups).getD 0 default).1

/-- Region "digest p"'s bytes at block `b`: every template instance's `b ‖ c`, template instance by template instance.
-/
def hmBytes (c : Circuit) (pub : Public) (p : ℕ) : ℕ → ByteArray :=
  fun b => Array.foldl (fun acc g => acc ++ bcChunk c pub b g p) ByteArray.empty (Array.range c.g)

theorem forall₂_mem {α β : Type} {R : α → β → Prop} : ∀ {L : List α} {es : List β}, List.Forall₂ R L es →
    ∀ a ∈ L, ∃ e ∈ es, R a e
  | _, _, .nil, _, h => absurd h List.not_mem_nil
  | _, _, .cons hab hrest, a, h => by
      rcases List.mem_cons.1 h with rfl | h
      · exact ⟨_, List.mem_cons_self, hab⟩
      · obtain ⟨e, he, hr⟩ := forall₂_mem hrest a h
        exact ⟨e, List.mem_cons_of_mem _ he, hr⟩

set_option maxHeartbeats 2000000 in
/-- **`HmRow.regions` has region "digest p" for every input port `p`**, as `mkRegion` built it. -/
theorem regions_digest {c : Circuit} {pub : Public} {regs : Array Region} (h : HmRow.regions c pub = .ok regs)
    {p : ℕ} (hp : p < c.ports.size) :
    ∃ name, ∃ r ∈ regs.toList, mkRegion c.kLog name (hmFree c) (hmFixed c p) (hmBytes c pub p) = .ok r := by
  unfold HmRow.regions at h
  dsimp only at h
  try simp only [throw_bind', pure_bind] at h
  obtain ⟨out, hloop, h⟩ := ebind_ok h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hloop
  obtain ⟨es, hes, heq⟩ := list_forIn_push _ (fun _ v => v) _ _ _ hloop
  have hpL : p ∈ List.range' [:c.ports.size].start [:c.ports.size].size [:c.ports.size].step := by
    simp only [Std.Legacy.Range.size, List.mem_range'_1]
    omega
  obtain ⟨e, he, v, hv, hev⟩ := forall₂_mem hes p hpL
  have hout : e ∈ out.toList := by rw [heq]; simpa using he
  have hsub : ∀ x ∈ out.toList, x ∈ regs.toList := by
    split at h
    all_goals
      obtain ⟨r, -, h⟩ := ebind_ok h
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      intro x hx
      simp [hx]
  exact ⟨_, e, hsub e hout, hev ▸ hv⟩

/-! ### Where region "digest p" pins byte `j`, bit `t` of template instance `g` -/

/-- The in-block address region "digest p" pins byte `j`'s bit `t` of template instance `g`'s `b ‖ c` at. -/
def bcAddr (c : Circuit) (g p j t : ℕ) : ℕ :=
  hmFixed c p + (8 * j + t) + g * 2 ^ ((c.rangeOf c.hmNet).slotLog + (c.perVu.getD c.hmNet 1).log2)

/-- **It is the `hm96` slot's output**: template instance `g`'s port-`p` slot of the `hm96` net, at the net's first
output column plus `8j + t`, when the net has `2^lp` slots per template instance. -/
theorem bcAddr_eq_slotCol (c : Circuit) (g p j t : ℕ)
    (hpv : c.perVu.getD c.hmNet 0 = 2 ^ (c.perVu.getD c.hmNet 1).log2) :
    bcAddr c g p j t =
      c.slotCol c.hmNet g p + ((portCols (c.nets.getD c.hmNet default).2.outGroups).getD 0 default).1 + 8 * j + t := by
  simp only [bcAddr, hmFixed, Circuit.slotCol, Range.slot, Range.start, hpv, pow_add]
  ring

theorem hmFree_toList (c : Circuit) : (hmFree c).toList =
    (List.range 10).map (0 + ·) ++
      (List.range c.g.log2).map ((c.rangeOf c.hmNet).slotLog + (c.perVu.getD c.hmNet 1).log2 + ·) := by
  simp [hmFree]

/-- Region bit `g·1024 + 8j + t` (byte `j`'s bit `t` of template instance `g`) of a region `mkRegion` built as "digest
p" sits at `bcAddr`. -/
theorem inBlock_digest {c : Circuit} {pub : Public} {p : ℕ} {name : String} {r : Region} (nbl : ℕ)
    (h : mkRegion c.kLog name (hmFree c) (hmFixed c p) (hmBytes c pub p) = .ok r) {g j t : ℕ}
    (hg : g < 2 ^ c.g.log2) (hj : j < 128) (ht : t < 8) :
    (Refine.regionOf c.kLog nbl r).inBlock (g * 1024 + (8 * j + t)) = bcAddr c g p j t := by
  rw [inBlock_mkRegion h, hmFree_toList, spread_append, spread_range, spread_range]
  simp only [List.length_map, List.length_range, pow_zero, mul_one, bcAddr]
  have h1 : (g * 1024 + (8 * j + t)) % 2 ^ 10 = 8 * j + t := by
    rw [show (2 : ℕ) ^ 10 = 1024 by norm_num]; omega
  have h2 : (g * 1024 + (8 * j + t)) / 2 ^ 10 % 2 ^ c.g.log2 = g := by
    rw [show (2 : ℕ) ^ 10 = 1024 by norm_num,
      show (g * 1024 + (8 * j + t)) / 1024 = g by omega]
    exact Nat.mod_eq_of_lt hg
  rw [h1, h2]
  ring

/-! ### The bytes there -/

theorem get!_eq_getD (a : ByteArray) (i : ℕ) : a.get! i = a.data.toList.getD i default := by
  cases a with
  | mk d =>
      simp only [ByteArray.get!, List.getD_eq_getElem?_getD, Array.getElem?_toList]
      rw [getElem!_def]; cases d[i]? <;> rfl

theorem data_foldl_append (f : ℕ → ByteArray) : ∀ G : ℕ,
    (Array.foldl (fun acc g => acc ++ f g) ByteArray.empty (Array.range G)).data.toList =
      ((List.range G).map fun g => (f g).data.toList).flatten
  | 0 => rfl
  | G + 1 => by
      rw [Array.range_succ, Array.foldl_append]
      have : ∀ A : ByteArray, Array.foldl (fun acc g => acc ++ f g) A #[G] = A ++ f G := fun A => rfl
      rw [this, ByteArray.data_append, Array.toList_append, data_foldl_append f G, List.range_succ, List.map_append,
        List.flatten_append]
      simp

theorem getD_flatten_uniform {L : List (List UInt8)} :
    ∀ (g j : ℕ), (∀ i ≤ g, (L.getD i []).length = 128) → g < L.length → j < 128 →
      L.flatten.getD (g * 128 + j) default = (L.getD g []).getD j default := by
  induction L with
  | nil => intro g j _ hg; simp at hg
  | cons l L ih =>
      intro g j hL hg hj
      have hl : l.length = 128 := hL 0 (Nat.zero_le _)
      rw [List.flatten_cons]
      cases g with
      | zero =>
          simp only [zero_mul, zero_add, List.getD_cons_zero]
          rw [List.getD_eq_getElem?_getD, List.getElem?_append_left (by omega), ← List.getD_eq_getElem?_getD]
      | succ g =>
          rw [List.getD_eq_getElem?_getD, List.getElem?_append_right (by rw [hl]; nlinarith), hl,
            show (g + 1) * 128 + j - 128 = g * 128 + j by ring_nf; omega, ← List.getD_eq_getElem?_getD,
            List.getD_cons_succ]
          exact ih g j (fun i hi => by simpa using hL (i + 1) (by omega)) (by simp at hg; omega) hj

/-- Byte `g·128 + j` of region "digest p"'s bytes at block `b` is byte `j` of template instance `g`'s `b ‖ c`, when
every template instance's up to `g` is 128 bytes. -/
theorem hmBytes_get {c : Circuit} {pub : Public} {p b g j : ℕ} (hg : g < c.g) (hj : j < 128)
    (hsz : ∀ g' ≤ g, (bcChunk c pub b g' p).size = 128) :
    (hmBytes c pub p b).get! (g * 128 + j) = (bcChunk c pub b g p).get! j := by
  have hm : ∀ g' < c.g, ((List.range c.g).map fun g => (bcChunk c pub b g p).data.toList).getD g' [] =
      (bcChunk c pub b g' p).data.toList := by
    intro g' hg'
    rw [List.getD_eq_getElem _ _ (by simpa using hg')]
    simp
  rw [get!_eq_getD, get!_eq_getD, hmBytes, data_foldl_append,
    getD_flatten_uniform g j (fun i hi => by
      rw [hm i (by omega), Array.length_toList, ByteArray.size_data, hsz i hi]) (by simpa using hg) hj, hm g hg]

/-! ### What a satisfying witness holds there -/

/-- An in-block address, or `0` past the block: a fixed reader that never reads past the block. -/
def inBlockOr (K x : ℕ) : Fin (2 ^ K) := if h : x < 2 ^ K then ⟨x, h⟩ else ⟨0, Nat.two_pow_pos K⟩

/-- **The `b ‖ c` a satisfying witness holds at an `hm96` slot's output** is the one the statement pins: at block `o`,
template instance `g` (below `2^⌊log₂ g⌋`, all of them when the block's template instance count is a power of two) and
input port `p`, the 128 bytes at `bcAddr` are `bcChunk`, the public digest of the template instance's input (the pinned
dummy past the population). It holds in the model's statement at setupH's layout whose regions contain the executable's
(`hregs`), once every template instance's `b ‖ c` in the block up to `g` is 128 bytes (`hsz`). -/
theorem bc_pinned {st : Stmt} (L : FlockSoundness.Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
    (hregs : ∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs)
    (hR : HmRow.regions st.c st.pub = .ok st.regions)
    {z : Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl))
    {g p : ℕ} (hg : g < st.c.g) (hg2 : g < 2 ^ st.c.g.log2) (hp : p < st.c.ports.size)
    (hsz : ∀ g' ≤ g, (bcChunk st.c st.pub o.val g' p).size = 128) :
    bytesAt z (fun (j : Fin 128) (t : Fin 8) =>
        (L.model regs).addr o (inBlockOr st.c.kLog (bcAddr st.c g p j.val t.val))) =
      (bcChunk st.c st.pub o.val g p).data.toList := by
  obtain ⟨name, r, hr, hmk⟩ := regions_digest hR hp
  have hmem : Refine.regionOf (L.model regs).kLog ((L.model regs).m - (L.model regs).kLog) r ∈
      (L.model regs).regions := hregs r hr
  obtain ⟨hrf, -, -, hnd, -, hlt⟩ := mkRegion_spec hmk
  have hfree : r.free = hmFree st.c := by rw [hrf]
  have hbytes : r.bytes = hmBytes st.c st.pub p := by rw [hrf]
  refine bytesAt_eq z _ _ (by rw [Array.length_toList, ByteArray.size_data, hsz g le_rfl]) fun j t => ?_
  have hlen : (Refine.regionOf st.c.kLog (st.m - st.c.kLog) r).free.length = 10 + st.c.g.log2 := by
    rw [← List.length_map (f := Fin.val), regionOf_free_map _ _ _ (fun b hb => hlt b (by
      rw [hfree] at hb; exact Array.mem_toList_iff.1 hb)), hfree, hmFree_toList]
    simp
  have hk : g * 1024 + (8 * j.val + t.val) < 2 ^ (Refine.regionOf st.c.kLog (st.m - st.c.kLog) r).free.length := by
    rw [hlen, pow_add, show (2 : ℕ) ^ 10 = 1024 by norm_num]
    have := j.isLt; have := t.isLt
    nlinarith
  have hib := inBlock_digest (st.m - st.c.kLog) hmk hg2 j.isLt t.isLt
  have hnd' : (Refine.regionOf st.c.kLog (st.m - st.c.kLog) r).free.Nodup := by
    unfold Refine.regionOf
    rw [hfree]
    exact filterMap_fin_nodup _ _ hnd
  have ha : (Refine.regionOf st.c.kLog (st.m - st.c.kLog) r).inBlock (g * 1024 + (8 * j.val + t.val)) <
      2 ^ st.c.kLog :=
    inBlock_lt _ hnd' _
  have hread := region_read hz hmem o _ hk ha
  have hfin : inBlockOr st.c.kLog (bcAddr st.c g p j.val t.val) = ⟨_, ha⟩ := by
    simp only [inBlockOr, ← hib, ha, ↓reduceDIte]
  show bit ((L.model regs).block z o (inBlockOr st.c.kLog (bcAddr st.c g p j.val t.val))) = _
  rw [hfin]
  refine hread.trans ?_
  rw [hbytes, show (g * 1024 + (8 * j.val + t.val)) / 8 = g * 128 + j.val by omega,
    show (g * 1024 + (8 * j.val + t.val)) % 8 = t.val by omega, hmBytes_get hg j.isLt hsz, get!_eq_getD]
  have hjl : j.val < (bcChunk st.c st.pub o.val g p).data.toList.length := by
    rw [Array.length_toList, ByteArray.size_data, hsz g le_rfl]; exact j.isLt
  rw [List.getD_eq_getElem _ _ hjl]
  rfl

/-! ### setupH's regions -/

set_option maxHeartbeats 2000000 in
/-- **setupH's statement's regions are `HmRow.regions` of its circuit and public file.** -/
theorem setupH_regions {tags : Tags} {circuitFile publicFile : ByteArray} {coins : Coins}
    {tables : List (String × ByteArray)} {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (h : Stmt.setupH tags circuitFile publicFile coins tables draw partition program = .ok st) :
    HmRow.regions st.c st.pub = .ok st.regions := by
  unfold Stmt.setupH at h
  dsimp only at h
  simp only [throw_bind', pure_bind] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    assumption

end FlockSoundness.Discharge.Layout
