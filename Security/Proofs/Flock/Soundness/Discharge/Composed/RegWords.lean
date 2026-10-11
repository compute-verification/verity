import Proofs.Flock.Soundness.Discharge.Composed.Coined
import Proofs.Flock.Soundness.Discharge.Composed.Inner.Holds
import Proofs.Flock.Soundness.Discharge.Hidden.Public
import Proofs.Flock.Soundness.Discharge.Layout.HmRegion
import Proofs.Flock.Soundness.LinkSound

/-!
# A valid witness's registered values are the public ones

A region is whole-word when its free bits include the in-word bits 0–6 (`WholeWord`). Then each word `x` of a region
claim either lies on the region, where a valid witness's bits are the region's public ones (`wordBit_eq_regionBit`), or
off it, where the region fixes a word bit that `x` lacks and the claim's `eq` weight vanishes (`eqWord_off`). So at a
valid witness every region claim's message is the public one (`coinedOf_reg`). Every region of a `setupHidden`
statement is whole-word (`setupHidden_whole`), and of a `setupH` statement whose output is the unit's and whose circuit
is untyped (`setupH_whole`).
-/

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock Model ZkSession ZkLink ZkExec DecodeZL

local notation "𝔽" => FlockLevel3.GF128

/-- A region whose free bits include every in-word bit (0 to 6). -/
def WholeWord (reg : Flock.Region) : Prop := ∀ b < 7, b ∈ reg.free.toList

/-- Every region of the statement is whole-word. -/
def WholeWords (st : Stmt) : Prop := ∀ reg ∈ st.regions.toList, WholeWord reg

/-! ### The regions the setups build are whole-word -/

theorem mkRegion_whole {kl : ℕ} {name : String} {free : Array ℕ} {fixed : ℕ} {bytes : ℕ → ByteArray}
    {r : Flock.Region} (h : mkRegion kl name free fixed bytes = .ok r) (hf : ∀ b < 7, b ∈ free.toList) :
    WholeWord r := by
  rw [WholeWord, (mkRegion_ok h).1]; exact hf

theorem low_mem {n : ℕ} (hn : 7 ≤ n) (A : Array ℕ) : ∀ b < 7, b ∈ ((Array.range n).map (0 + ·) ++ A).toList := by
  intro b hb
  simp only [Array.toList_append, Array.toList_map, Array.toList_range, List.mem_append, List.mem_map, List.mem_range]
  exact Or.inl ⟨b, by omega, Nat.zero_add b⟩

theorem hmFree_low (c : Flock.Circuit) : ∀ b < 7, b ∈ (Layout.hmFree c).toList := low_mem (n := 10) (by omega) _

theorem outRegion_whole {c : Flock.Circuit} {pub : Flock.Public} {r : Flock.Region} (h : c.outRegion pub = .ok r) : WholeWord r := by
  unfold Flock.Circuit.outRegion at h
  exact mkRegion_whole h (low_mem (by omega) _)

theorem whole_push {out : Array Flock.Region} {r : Flock.Region} (h1 : ∀ x ∈ out.toList, WholeWord x)
    (h2 : WholeWord r) : ∀ x ∈ (out.push r).toList, WholeWord x := by
  intro x hx
  rw [Array.toList_push, List.mem_append, List.mem_singleton] at hx
  rcases hx with hx | rfl
  · exact h1 x hx
  · exact h2

set_option maxHeartbeats 2000000 in
/-- **`HmRow.regions`' regions are whole-word** once the output is the unit's and the circuit untyped: each port's
"digest p" (`Layout.hmFree`) and the unit's "out" (`Circuit.outRegion`). -/
theorem hm_regions_whole {c : Flock.Circuit} {pub : Flock.Public} {regs : Array Flock.Region} (h : HmRow.regions c pub = .ok regs)
    (ho : c.outNet = c.unitNet) (ht : c.typed = none) : ∀ x ∈ regs.toList, WholeWord x := by
  unfold HmRow.regions at h
  dsimp only at h
  try simp only [throw_bind', pure_bind] at h
  obtain ⟨out, hloop, h⟩ := ebind_ok h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hloop
  obtain ⟨es, hes, heq⟩ := list_forIn_push _ (fun _ v => v) _ _ _ hloop
  have hI : ∀ x ∈ out.toList, WholeWord x := by
    intro x hx
    have hx' : x ∈ es := by simpa [heq] using hx
    obtain ⟨a, -, v, hv, rfl⟩ := Layout.forall₂_mem hes.flip x hx'
    exact mkRegion_whole hv (hmFree_low c)
  split at h
  · rename_i hc
    simp [ho, ht] at hc
  · obtain ⟨r, hr, h⟩ := ebind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact whole_push hI (outRegion_whole hr)

/-- **A `setupH` statement's regions are whole-word** once its circuit's output is the unit's and it is untyped. -/
theorem setupH_whole {tags : Tags} {circuitFile publicFile : ByteArray} {coins : Coins}
    {tables : List (String × ByteArray)} {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (h : Stmt.setupH tags circuitFile publicFile coins tables draw partition program = .ok st)
    (ho : st.c.outNet = st.c.unitNet) (ht : st.c.typed = none) : WholeWords st :=
  hm_regions_whole (Layout.setupH_regions h) ho ht

/-- **A `setupHidden` statement's regions are whole-word**: each is a port's "digest p" (`Hidden.regions_get`). -/
theorem setupHidden_whole {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) : WholeWords st := by
  obtain ⟨-, hall⟩ := Hidden.regions_get (Hidden.setupHidden_spec' h).regions
  intro reg hreg
  obtain ⟨p, hp, rfl⟩ := List.getElem_of_mem hreg
  have hp' : p < st.regions.size := by simpa using hp
  simp only [Array.getElem_toList]
  exact mkRegion_whole (hall p hp') (hmFree_low _)

/-! ### At a valid witness, a region claim's message is the public one -/

theorem foldl_add_range (f : ℕ → ℕ) : ∀ (N a : ℕ),
    (List.range N).foldl (fun s t => s + f t) a = a + ∑ t ∈ Finset.range N, f t
  | 0, a => by simp
  | N + 1, a => by
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, foldl_add_range f N a,
        Finset.sum_range_succ, Nat.add_assoc]

/-- The bits of the sum of `2^u` over the `u < N` where `P u`, as `regionBit` adds them up. -/
theorem testBit_foldl (P : ℕ → Bool) (N t : ℕ) :
    ((List.range N).foldl (fun s u => s + if P u then 2 ^ u else 0) 0).testBit t = (decide (t < N) && P t) := by
  have h : (List.range N).foldl (fun s u => s + if P u then 2 ^ u else 0) 0 =
      ∑ u ∈ (Finset.range N).filter (fun u => P u = true), 2 ^ u := by
    rw [Finset.sum_filter]
    exact (foldl_add_range (fun u => if P u then 2 ^ u else 0) N 0).trans (Nat.zero_add _)
  rw [h, testBit_sum_two_pow]
  simp [Finset.mem_filter, Finset.mem_range]

theorem zmod2_of_bit {a : ZMod 2} {b : Bool} (h : FlockSoundness.bit a = b) : a = if b = true then 1 else 0 := by
  subst h; revert a; decide

/-- **On a whole-word region, a valid witness's bit is the region's**: word `x`'s bit `v`, where `x` has the region's
fixed bits above the word, is bit `128·x + v` of the region's bytes (`regionBit`), by `Satisfies`' region conjunct
(`Layout.region_read`) at the block `(128·x + v) / 2^k_log` and region bit `k` whose in-block address is the rest. -/
theorem wordBit_eq_regionBit {st : Stmt} (hwf : StmtWF st) (hm : st.m = st.c.kLog + st.nbl)
    {reg : Flock.Region} (hreg : reg ∈ st.regions.toList) (hw : RegionWF st.c.kLog reg) (hW : WholeWord reg)
    {z : Witness st.m} (hz : (stmtOf st hwf (regionsOf st)).Satisfies z) {x v : ℕ} (hx : x < 2 ^ (st.m - 7))
    (hv : v < 128)
    (hfix : ∀ i, i + 7 < st.c.kLog → i + 7 ∉ reg.free.toList →
      x.testBit i = ((reg.mask >>> (i + 7)) % 2 == 1 && (reg.fixed >>> (i + 7)) % 2 == 1)) :
    wordBit z x v = regionBit st reg (128 * x + v) := by
  have h7 : 7 ≤ st.c.kLog := by have := hw.free_lt 6 (hW 6 (by omega)); omega
  have hkm : st.c.kLog ≤ st.m := hwf.hk
  have hidx : 128 * x + v < 2 ^ st.m := by
    have : 2 ^ st.m = 128 * 2 ^ (st.m - 7) := by
      rw [show (128 : ℕ) = 2 ^ 7 by norm_num, ← pow_add]; congr 1; omega
    rw [this]; omega
  unfold wordBit
  rw [dite_eq_left hidx]
  unfold regionBit
  dsimp only
  have hb : (List.range st.nbl).foldl (fun a k => a + if (128 * x + v).testBit (st.c.kLog + k) then 2 ^ k else 0) 0
      = (128 * x + v) / 2 ^ st.c.kLog := by
    refine Nat.eq_of_testBit_eq fun t => ?_
    rw [testBit_foldl (fun k => (128 * x + v).testBit (st.c.kLog + k)), Nat.testBit_div_two_pow]
    by_cases ht : t < st.nbl
    · rw [decide_eq_true ht, Bool.true_and, Nat.add_comm st.c.kLog t]
    · have hlt : 128 * x + v < 2 ^ (t + st.c.kLog) :=
        lt_of_lt_of_le hidx (Nat.pow_le_pow_right (by norm_num) (by omega))
      rw [decide_eq_false ht, Bool.false_and, Nat.testBit_lt_two_pow hlt]
  rw [hb]
  have hkb := fun t => testBit_foldl (fun t => (128 * x + v).testBit reg.free[t]!) reg.free.size t
  generalize (List.range reg.free.size).foldl
    (fun a t => a + if (128 * x + v).testBit reg.free[t]! then 2 ^ t else 0) 0 = k at hkb ⊢
  have hRnd := filterMap_fin_nodup st.c.kLog _ hw.nodup
  have hRlen : (regionOf st.c.kLog (st.m - st.c.kLog) reg).free.length = reg.free.size :=
    (filterMap_fin_length _ _ hw.free_lt).trans Array.length_toList
  have hklt : k < 2 ^ (regionOf st.c.kLog (st.m - st.c.kLog) reg).free.length := by
    rw [hRlen]
    refine Nat.lt_pow_two_of_testBit k fun t ht => ?_
    rw [hkb, decide_eq_false (by omega : ¬ t < reg.free.size), Bool.false_and]
  have hin : (regionOf st.c.kLog (st.m - st.c.kLog) reg).inBlock k = (128 * x + v) % 2 ^ st.c.kLog := by
    refine Nat.eq_of_testBit_eq fun n => ?_
    rw [Nat.testBit_mod_two_pow]
    by_cases hn : n < st.c.kLog
    · rw [decide_eq_true hn, Bool.true_and,
        testBit_inBlock (regionOf st.c.kLog (st.m - st.c.kLog) reg) hRnd k ⟨n, hn⟩]
      have hI := idxOf?_filterMap_fin st.c.kLog n hn reg.free.toList hw.free_lt
      split
      · rename_i t ht
        obtain ⟨htl, hget, -⟩ := List.idxOf?_eq_some_iff.1 (hI.symm.trans ht)
        have hsz : t < reg.free.size := by simpa using htl
        have hft : reg.free[t]! = n := by rw [getElem!_pos reg.free t hsz]; simpa using hget
        rw [hkb, decide_eq_true hsz, Bool.true_and, hft]
      · rename_i hnone
        have hnf : n ∉ reg.free.toList := List.idxOf?_eq_none_iff.1 (hI.symm.trans hnone)
        have h7n : 7 ≤ n := by
          by_contra h
          exact hnf (hW n (by omega))
        obtain ⟨i, rfl⟩ : ∃ i, n = i + 7 := ⟨n - 7, by omega⟩
        rw [show (128 : ℕ) = 2 ^ 7 by norm_num, Nat.testBit_two_pow_mul_add x (by omega : v < 2 ^ 7),
          ite_eq_right (show ¬ (i + 7 < 7) by omega), Nat.add_sub_cancel]
        exact (hfix i hn hnf).symm
    · rw [decide_eq_false hn, Bool.false_and]
      exact testBit_inBlock_ge (regionOf st.c.kLog (st.m - st.c.kLog) reg) hRnd k n (by omega)
  have ha : (regionOf st.c.kLog (st.m - st.c.kLog) reg).inBlock k < 2 ^ st.c.kLog := by
    rw [hin]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have hdiv : (128 * x + v) / 2 ^ st.c.kLog < 2 ^ (st.m - st.c.kLog) := by
    rw [Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _), ← pow_add, Nat.sub_add_cancel hkm]; exact hidx
  have hread := Layout.region_read hz (reg := reg) (List.mem_map_of_mem hreg) ⟨_, hdiv⟩ k hklt ha
  have hz_eq : (stmtOf st hwf (regionsOf st)).block z ⟨_, hdiv⟩ ⟨_, ha⟩ = z ⟨128 * x + v, hidx⟩ := by
    refine congrArg z (Fin.ext ?_)
    show (128 * x + v) / 2 ^ st.c.kLog * 2 ^ st.c.kLog + (regionOf st.c.kLog (st.m - st.c.kLog) reg).inBlock k =
      128 * x + v
    rw [hin, Nat.div_add_mod']
  simp only [Layout.uint8_bit]
  exact hz_eq.symm.trans (zmod2_of_bit hread)

/-- **Off a region, its claim's weight vanishes**: at a word `x` whose bit `i` differs from the bit `i + 7` the region
fixes, `eq(p_j, x) = 0` (`regionPoint_fixed`). -/
theorem eqWord_off (st : Stmt) (link : Fin 2 → Fin st.mPts → F128) (j : Fin (2 * st.regions.size)) {x i : ℕ}
    (hk : i + 7 < st.c.kLog) (hm : i + 7 < st.m) (hf : i + 7 ∉ (st.regions[j.val / 2]!).free.toList)
    (hne : x.testBit i ≠ (((st.regions[j.val / 2]!).mask >>> (i + 7)) % 2 == 1 &&
      ((st.regions[j.val / 2]!).fixed >>> (i + 7)) % 2 == 1)) :
    eqWord (regionPtOf st link j) x = 0 := by
  have hfix := regionPoint_fixed st (st.regions[j.val / 2]!)
    (Array.ofFn (link (if j.val % 2 = 0 then 0 else 1))) hk hm hf
  generalize (((st.regions[j.val / 2]!).mask >>> (i + 7)) % 2 == 1 &&
    ((st.regions[j.val / 2]!).fixed >>> (i + 7)) % 2 == 1) = bv at hfix hne
  unfold eqWord regionPtOf
  generalize hq : (st.regionPoint st.regions[j.val / 2]! (Array.ofFn (link (if j.val % 2 = 0 then 0 else 1)))).toList.drop 7 = q
  have hqi : q[i]? = some (if bv = true then 1 else 0) := by
    rw [← hq, List.getElem?_drop, Array.getElem?_toList, Nat.add_comm]; exact hfix
  have hlen : i < q.length := (List.getElem?_eq_some_iff.1 hqi).1
  have hv : q[i]! = (if bv = true then 1 else 0) := by
    rw [getElem!_pos q i hlen]; exact Option.some.inj ((List.getElem?_eq_getElem hlen).symm.trans hqi)
  refine foldl_mul_eq_zero _ _ _ ⟨i, List.mem_range.2 hlen, ?_⟩
  try dsimp only
  rw [hv]
  revert hne
  cases bv <;> cases x.testBit i <;> simp [one_add_one']

/-- **A region claim's message at a valid witness is the public one**, word by word: on the region by
`wordBit_eq_regionBit`, off it by `eqWord_off`. -/
theorem sum_reg {S : Statement} {st : Stmt} (hwf : StmtWF st) (hS : S = stmtOf st hwf (regionsOf st))
    (hr : RegionsWF st) (hW : WholeWords st) {z : Witness S.m} (hz : S.Satisfies z)
    (link : Fin 2 → Fin st.mPts → F128) (j : Fin (2 * st.regions.size)) (v : Fin 128) :
    ∑ x : Fin (2 ^ (S.m - 7)), fOf (eqWord (regionPtOf st link j) x.val) *
      ZK.Table.castF (F := 𝔽) (wordBit z x.val v.val) = pubRegOf st link j v := by
  subst hS
  have hj : j.val / 2 < st.regions.size := by omega
  have hmem : st.regions[j.val / 2]! ∈ st.regions.toList := by
    rw [getElem!_pos st.regions _ hj]; exact Array.getElem_mem_toList _
  unfold pubRegOf
  refine Finset.sum_congr rfl fun x _ => ?_
  by_cases hfix : ∀ i, i + 7 < st.c.kLog → i + 7 ∉ (st.regions[j.val / 2]!).free.toList →
      x.val.testBit i = (((st.regions[j.val / 2]!).mask >>> (i + 7)) % 2 == 1 &&
        ((st.regions[j.val / 2]!).fixed >>> (i + 7)) % 2 == 1)
  · exact congrArg (fun b => fOf (eqWord (regionPtOf st link j) x.val) * ZK.Table.castF (F := 𝔽) b)
      (wordBit_eq_regionBit hwf hr.hm hmem (hr.regions _ hmem) (hW _ hmem) hz x.isLt v.isLt hfix)
  · push Not at hfix
    obtain ⟨i, hi, hnf, hne⟩ := hfix
    rw [eqWord_off st link j hi (by have := hwf.hk; omega) hnf hne, fOf_zero, zero_mul, zero_mul]

variable {mPts : ℕ} (P : LigProver)

/-- **`H_reg` at a valid witness** of `coinedOf`'s tables, once the statement's regions are whole-word: every region
claim's message is the public one. -/
theorem coinedOf_reg (Tz : TabZK 𝔽 mPts) (st : Stmt) (hwf : StmtWF st) (hr : RegionsWF st)
    (hS : Tz.S = stmtOf st hwf (regionsOf st)) (hW : WholeWords st) :
    ∀ c w, (coinedOf P Tz st).Valid w → ((coinedOf P Tz st).T c).tab.sReg w = ((coinedOf P Tz st).T c).tab.pubReg :=
  fun c _ hv => funext fun j => funext fun v => sum_reg hwf hS hr hW hv c.rest.link j v

end FlockSoundness.Discharge.Composed
