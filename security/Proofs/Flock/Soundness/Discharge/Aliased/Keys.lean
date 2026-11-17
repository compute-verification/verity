module

public import Proofs.Flock.Soundness.Discharge.Aliased.Copies
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.Discharge.Aliased.Rows

/-! Δ's source as a key the same at every template instance (`srcKey`), from which `Copies` follows for an instance that joins inputs only with one key. -/

@[expose] public section

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Aliased

/-- A key's column at template instance `g` (a forced-zero row's and a padding row's are not read). -/
def realize (c : Circuit) (g : ℕ) : Src → ℕ
  | .msg p byte bit => Typed.msgCol c g p byte bit
  | .wire n i col => c.slotCol n g i + col
  | .lit col => col
  | _ => 0

theorem srcAt_ne_pad (c : Circuit) (k p t k' off : ℕ) : srcAt c k p t ≠ .pad k' off := by
  unfold srcAt
  split
  · split
    · simp
    · simp only
      split <;> simp
  · split
    · simp only
      split <;> simp
    · split <;> simp

theorem flatSrc_key (c : Circuit) (s₀ p t : ℕ) :
    (srcAt c (s₀ % c.upv) p t = .zero ∧ (Flat.flatSrc c s₀ p t = c.unitRange.slot s₀ + c.unit.useful ∨
      Flat.flatSrc c s₀ p t = c.unitRange.slot 0 + c.unit.useful)) ∨
    (srcAt c (s₀ % c.upv) p t ≠ .zero ∧
      Flat.flatSrc c s₀ p t = realize c (s₀ / c.upv) (srcAt c (s₀ % c.upv) p t)) := by
  unfold Flat.flatSrc srcAt
  by_cases hp : p < (c.unit.inGroups.getD 0 default).bits.size
  · simp only [hp, ↓reduceIte]
    unfold Flat.leafSrc
    by_cases hl : (c.leavesIn.getD (s₀ % c.upv) #[])[p]! < 0
    · refine Or.inl ⟨?_, Or.inl ?_⟩ <;> simp only [hl, ↓reduceIte]
    · simp only [hl, ↓reduceIte]
      split
      · exact Or.inr ⟨Src.noConfusion, rfl⟩
      · exact Or.inl ⟨rfl, Or.inl rfl⟩
  · simp only [hp, ↓reduceIte]
    rcases hf : c.leafCuts.toList.filter (fun e => decide (e.1 = s₀ % c.upv ∧ e.2.1 = p)) with _ | ⟨e, l⟩
    · rcases hw : c.wires.toList.filter (fun w => decide (w.dst = (c.unitNet, s₀ % c.upv, p))) with _ | ⟨w, l'⟩
      · refine Or.inr ⟨?_, ?_⟩ <;> simp only [hf, hw, realize]
        exact Src.noConfusion
      · refine Or.inr ⟨?_, ?_⟩ <;> simp only [hf, hw, realize, Flat.wireSrc]
        · exact Src.noConfusion
        · omega
    · unfold Flat.cutSrc
      simp only [hf]
      split
      · exact Or.inr ⟨Src.noConfusion, rfl⟩
      · exact Or.inl ⟨rfl, Or.inr rfl⟩

theorem srcAt_leaf (c : Circuit) (k : ℕ) {p : ℕ} (t : ℕ) (hp : p < (c.unit.inGroups.getD 0 default).bits.size) :
    srcAt c k p t = .zero ∨ ∃ q byte bit, srcAt c k p t = .msg q byte bit ∧ bit < 8 := by
  unfold srcAt
  simp only [hp, ↓reduceIte]
  split
  · exact Or.inl rfl
  · split
    · exact Or.inr ⟨_, _, _, rfl, Nat.mod_lt _ (by decide)⟩
    · exact Or.inl rfl

theorem srcKey_leaf (c : Circuit) (h1 : c.unit.inGroups.size = 1) (hnd : (portBits c.unit.inGroups).Nodup) {x : ℕ}
    (hm : x % (c.unit.inWords * WORD) ∈ portBits c.unit.inGroups) :
    srcKey c x = .zero ∨ ∃ q byte bit, srcKey c x = .msg q byte bit ∧ bit < 8 := by
  unfold srcKey
  rw [ite_eq_left hm]
  obtain ⟨p, t, hp, ht, hpt⟩ := mem_portBits hm
  rw [hpt, portOf_eq hnd hp ht]
  refine srcAt_leaf c _ _ ?_
  obtain ⟨g, hg⟩ := List.length_eq_one_iff.1 (show c.unit.inGroups.toList.length = 1 by simpa using h1)
  have hgs : c.unit.inGroups = #[g] := Array.toList_inj.1 (by simpa using hg)
  have hsz : (portCols #[g]).size = g.bits.size := by
    simpa using congrArg List.length (portCols_single g)
  rw [hgs] at hp ⊢
  simpa [hsz] using hp

theorem srcKey_pad {c : Circuit} {x k off : ℕ} (h : srcKey c x = .pad k off) :
    k = x / (c.unit.inWords * WORD) ∧ off = x % (c.unit.inWords * WORD) := by
  unfold srcKey at h
  split at h
  · exact absurd h (srcAt_ne_pad _ _ _ _ _ _)
  · cases h
    exact ⟨rfl, rfl⟩

theorem keyIdx_eq_none {c : Circuit} {n : ℕ} (x : Fin n) : keyIdx c n x = none ↔ srcKey c x.val = .zero := by
  unfold keyIdx
  split
  · exact iff_of_true rfl ‹_›
  · refine iff_of_false (fun h => ?_) ‹_›
    rw [List.find?_eq_none] at h
    exact h x (List.mem_finRange x) (by simp)

theorem keyIdx_eq {c : Circuit} {n : ℕ} {x x' : Fin n} (h : keyIdx c n x = keyIdx c n x') :
    srcKey c x.val = srcKey c x'.val := by
  by_cases hz : srcKey c x.val = .zero
  · have := (keyIdx_eq_none x).2 hz
    rw [h] at this
    rw [hz, (keyIdx_eq_none x').1 this]
  · have hz' : srcKey c x'.val ≠ .zero := fun hz' => hz ((keyIdx_eq_none x).1 (h.trans ((keyIdx_eq_none x').2 hz')))
    unfold keyIdx at h
    rw [ite_eq_right hz, ite_eq_right hz'] at h
    rcases hf : (List.finRange n).find? (fun i => srcKey c i.val = srcKey c x.val) with _ | i
    · rw [List.find?_eq_none] at hf
      exact absurd (by simp) (hf x (List.mem_finRange x))
    · rw [hf] at h
      have h1 := List.find?_some hf
      have h2 := List.find?_some h.symm
      simp only [decide_eq_true_eq] at h1 h2
      exact h1.symm.trans h2

section keys

variable {tags : Tags} {st : Stmt} (L : Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
  (F : UnitFacts tags st)

theorem stack_slot {g x : ℕ} (hx : x < st.c.upv * (st.c.unit.inWords * WORD)) :
    (g * st.c.upv + x / (st.c.unit.inWords * WORD)) % st.c.upv = x / (st.c.unit.inWords * WORD) ∧
      (g * st.c.upv + x / (st.c.unit.inWords * WORD)) / st.c.upv = g := by
  have hn : 0 < st.c.unit.inWords * WORD := Nat.pos_of_ne_zero fun h => by rw [h, Nat.mul_zero] at hx; omega
  have hk : x / (st.c.unit.inWords * WORD) < st.c.upv := (Nat.div_lt_iff_lt_mul hn).2 hx
  have hu : 0 < st.c.upv := Nat.pos_of_ne_zero fun h => by rw [h, Nat.zero_mul] at hx; omega
  refine ⟨?_, ?_⟩
  · rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hk]
  · rw [Nat.add_comm, Nat.add_mul_div_right _ _ hu, Nat.div_eq_of_lt hk, Nat.zero_add]

theorem flatSrcAt_key {g x : ℕ} (hx : x < st.c.upv * (st.c.unit.inWords * WORD))
    (hm : x % (st.c.unit.inWords * WORD) ∈ portBits st.c.unit.inGroups) :
    (srcKey st.c x = .zero ∧
      (flatSrcAt st.c (g * st.c.upv + x / (st.c.unit.inWords * WORD)) (x % (st.c.unit.inWords * WORD)) =
          st.c.unitRange.slot (g * st.c.upv + x / (st.c.unit.inWords * WORD)) + st.c.unit.useful ∨
        flatSrcAt st.c (g * st.c.upv + x / (st.c.unit.inWords * WORD)) (x % (st.c.unit.inWords * WORD)) =
          st.c.unitRange.slot 0 + st.c.unit.useful)) ∨
    (srcKey st.c x ≠ .zero ∧
      flatSrcAt st.c (g * st.c.upv + x / (st.c.unit.inWords * WORD)) (x % (st.c.unit.inWords * WORD)) =
        realize st.c g (srcKey st.c x)) := by
  obtain ⟨hmod, hdiv⟩ := stack_slot (g := g) hx
  have hk : srcKey st.c x = srcAt st.c ((g * st.c.upv + x / (st.c.unit.inWords * WORD)) % st.c.upv)
      (portOf (portCols st.c.unit.inGroups) (x % (st.c.unit.inWords * WORD)))
      (x % (st.c.unit.inWords * WORD) - (portCols st.c.unit.inGroups)[portOf (portCols st.c.unit.inGroups)
        (x % (st.c.unit.inWords * WORD))]!.1) := by
    rw [hmod]
    unfold srcKey
    rw [ite_eq_left hm]
  have := flatSrc_key st.c (g * st.c.upv + x / (st.c.unit.inWords * WORD))
    (portOf (portCols st.c.unit.inGroups) (x % (st.c.unit.inWords * WORD)))
    (x % (st.c.unit.inWords * WORD) - (portCols st.c.unit.inGroups)[portOf (portCols st.c.unit.inGroups)
      (x % (st.c.unit.inWords * WORD))]!.1)
  rw [← hk, hdiv] at this
  exact this

theorem copyPos_zero {g x : ℕ} (hg : g < st.c.g) (hx : x < st.c.upv * (st.c.unit.inWords * WORD))
    (hz : srcKey st.c x = .zero) : IsZeroPos L regs F (copyPos L regs F hg x) := by
  have hm : x % (st.c.unit.inWords * WORD) ∈ portBits st.c.unit.inGroups := by
    by_contra hm
    unfold srcKey at hz
    rw [ite_eq_right hm] at hz
    cases hz
  have hs₀ := stack_copy_lt L hg hx
  unfold copyPos copyAt
  rw [dite_eq_left hx, ite_eq_left hm]
  split
  · rename_i hlt
    rcases flatSrcAt_key (g := g) hx hm with ⟨-, he | he⟩ | ⟨hne, -⟩
    · exact ⟨_, hs₀, Fin.ext he⟩
    · exact ⟨0, Nat.lt_of_le_of_lt (Nat.zero_le _) hs₀, Fin.ext he⟩
    · exact absurd hz hne
  · exact ⟨_, hs₀, rfl⟩

theorem copyPos_key {g x : ℕ} (hg : g < st.c.g) (hx : x < st.c.upv * (st.c.unit.inWords * WORD))
    (hz : srcKey st.c x ≠ .zero) (hp : ∀ k off, srcKey st.c x ≠ .pad k off) :
    (copyPos L regs F hg x).val = realize st.c g (srcKey st.c x) ∨
      (¬ realize st.c g (srcKey st.c x) < 2 ^ st.c.kLog ∧ IsZeroPos L regs F (copyPos L regs F hg x)) := by
  have hm : x % (st.c.unit.inWords * WORD) ∈ portBits st.c.unit.inGroups := by
    by_contra hm
    exact hp (x / (st.c.unit.inWords * WORD)) (x % (st.c.unit.inWords * WORD))
      (by unfold srcKey; rw [ite_eq_right hm])
  have hs₀ := stack_copy_lt L hg hx
  have he : flatSrcAt st.c (g * st.c.upv + x / (st.c.unit.inWords * WORD)) (x % (st.c.unit.inWords * WORD)) =
      realize st.c g (srcKey st.c x) := by
    rcases flatSrcAt_key (g := g) hx hm with ⟨hz', -⟩ | ⟨-, he⟩
    · exact absurd hz' hz
    · exact he
  unfold copyPos copyAt
  rw [dite_eq_left hx, ite_eq_left hm]
  split
  · exact Or.inl he
  · rename_i hlt
    exact Or.inr ⟨he ▸ hlt, _, hs₀, rfl⟩

theorem copyPos_of_key {g x x' : ℕ} (hg : g < st.c.g) (hx : x < st.c.upv * (st.c.unit.inWords * WORD))
    (hx' : x' < st.c.upv * (st.c.unit.inWords * WORD)) (hkey : srcKey st.c x = srcKey st.c x') :
    copyPos L regs F hg x = copyPos L regs F hg x' ∨
      (IsZeroPos L regs F (copyPos L regs F hg x) ∧ IsZeroPos L regs F (copyPos L regs F hg x')) := by
  by_cases hz : srcKey st.c x = .zero
  · exact Or.inr ⟨copyPos_zero L regs F hg hx hz, copyPos_zero L regs F hg hx' (hkey ▸ hz)⟩
  by_cases hp : ∃ k off, srcKey st.c x = .pad k off
  · obtain ⟨k, off, hpad⟩ := hp
    obtain ⟨hk1, ho1⟩ := srcKey_pad hpad
    obtain ⟨hk2, ho2⟩ := srcKey_pad (hkey ▸ hpad)
    left
    have : x = x' := by
      rw [← Nat.div_add_mod x (st.c.unit.inWords * WORD), ← Nat.div_add_mod x' (st.c.unit.inWords * WORD),
        ← hk1, ← ho1, ← hk2, ← ho2]
    rw [this]
  simp only [not_exists] at hp
  rcases copyPos_key L regs F hg hx hz hp with h1 | ⟨hlt, h1⟩ <;>
    rcases copyPos_key L regs F hg hx' (hkey ▸ hz) (fun k off => hkey ▸ hp k off) with h2 | ⟨hlt', h2⟩
  · exact Or.inl (Fin.ext (h1.trans (hkey ▸ h2.symm)))
  · exact absurd (h1 ▸ (copyPos L regs F hg x).isLt) (hkey ▸ hlt')
  · exact absurd (h2 ▸ (copyPos L regs F hg x').isLt) (hkey ▸ hlt)
  · exact Or.inr ⟨h1, h2⟩

theorem copies_of_key {C : Verity.Boolean.Circuit} {n : ℕ} {P : Audit.Partition C n} {u : Fin n}
    (inst : P.IsRowsUnit u ((unitRows st L.unit_built).stack st.c.upv)) {g : ℕ} (hg : g < st.c.g)
    (hk : ∀ c c', inst.wire c = inst.wire c' → c.val < ((unitRows st L.unit_built).stack st.c.upv).nIn →
      c'.val < ((unitRows st L.unit_built).stack st.c.upv).nIn → srcKey st.c c.val = srcKey st.c c'.val) :
    Copies L regs F inst hg := by
  intro c c' hw hc hc'
  have hn := L.unitRows_nIn
  exact copyPos_of_key L regs F hg (by rw [← hn]; exact hc) (by rw [← hn]; exact hc') (hk c c' hw hc hc')

end keys

theorem setupH_aliased_of_key {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt} (ht : tags.typed = false)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {C : Verity.Boolean.Circuit} {n : ℕ}
    {P : Audit.Partition C n} {u : Fin n}
    (inst : P.IsRowsUnit u ((unitRows st (setupH_layout ht h).unit_built).stack st.c.upv))
    (o : Fin (2 ^ ((setupH_layout ht h).model regs).nbl)) {g : ℕ} (hg : g < st.c.g)
    (hk : ∀ c c', inst.wire c = inst.wire c' →
      c.val < ((unitRows st (setupH_layout ht h).unit_built).stack st.c.upv).nIn →
      c'.val < ((unitRows st (setupH_layout ht h).unit_built).stack st.c.upv).nIn →
        srcKey st.c c.val = srcKey st.c c'.val) :
    (setupH_layout ht h).Aliased regs inst o hg :=
  setupH_aliased ht h regs inst o hg (copies_of_key _ regs _ inst hg hk)

/-- A flat class's unit input `i` at copy index `k`: the key of the port bit it is (`inCols[i]`). -/
def flatKey (c : Circuit) (inCols : Array ℕ) (k i : ℕ) : Src :=
  srcAt c k (portOf (portCols c.unit.inGroups) (inCols.getD i 0))
    (inCols.getD i 0 - (portCols c.unit.inGroups)[portOf (portCols c.unit.inGroups) (inCols.getD i 0)]!.1)

section flat

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt} (ht : tags.typed = true)
  (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) {k : Typed.Class} {net : Net}
  (hr : Typed.read tags cf tables = .ok (k, .inl net)) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))

theorem flatCopySrc_key (g : Fin st.c.unitRange.count) (i : ℕ) :
    (flatKey st.c (k.done.getD k.unit default).inCols (g.val % st.c.upv) i = .zero ∧
      (flatSrcAt st.c g ((k.done.getD k.unit default).inCols.getD i 0) = st.c.unitRange.slot g + st.c.unit.useful ∨
        flatSrcAt st.c g ((k.done.getD k.unit default).inCols.getD i 0) = st.c.unitRange.slot 0 + st.c.unit.useful)) ∨
    (flatKey st.c (k.done.getD k.unit default).inCols (g.val % st.c.upv) i ≠ .zero ∧
      flatSrcAt st.c g ((k.done.getD k.unit default).inCols.getD i 0) =
        realize st.c (g.val / st.c.upv) (flatKey st.c (k.done.getD k.unit default).inCols (g.val % st.c.upv) i)) :=
  flatSrc_key st.c g.val _ _

theorem flatCopyPos_zero (g : Fin st.c.unitRange.count) {i : ℕ}
    (hz : flatKey st.c (k.done.getD k.unit default).inCols (g.val % st.c.upv) i = .zero) :
    flatCopyPos ht h hr regs g i ∈ Finset.univ.image (flatZeroPos ht h hr regs) := by
  unfold flatCopyPos
  split
  · rcases flatCopySrc_key (k := k) g i with ⟨-, he | he⟩ | ⟨hne, -⟩
    · exact Finset.mem_image.2 ⟨g, Finset.mem_univ _, Fin.ext he.symm⟩
    · exact Finset.mem_image.2 ⟨⟨0, Nat.lt_of_le_of_lt (Nat.zero_le _) g.isLt⟩, Finset.mem_univ _, Fin.ext he.symm⟩
    · exact absurd hz hne
  · exact Finset.mem_image.2 ⟨g, Finset.mem_univ _, rfl⟩

theorem flatCopies_of_key (g : Fin st.c.unitRange.count) (src : ℕ → FlockSoundness.Src)
    (hk : ∀ i < (setupH_flatTableClass ht h hr regs).rows.nIn, ∀ j < (setupH_flatTableClass ht h hr regs).rows.nIn,
      src i = src j → flatKey st.c (k.done.getD k.unit default).inCols (g.val % st.c.upv) i =
        flatKey st.c (k.done.getD k.unit default).inCols (g.val % st.c.upv) j) :
    (setupH_flatTableClass ht h hr regs).Copies g src := by
  intro i hi j hj hs
  have hkey := hk i hi j hj hs
  show flatCopyPos ht h hr regs g i = flatCopyPos ht h hr regs g j ∨
    flatCopyPos ht h hr regs g i ∈ Finset.univ.image (flatZeroPos ht h hr regs) ∧
      flatCopyPos ht h hr regs g j ∈ Finset.univ.image (flatZeroPos ht h hr regs)
  by_cases hz : flatKey st.c (k.done.getD k.unit default).inCols (g.val % st.c.upv) i = .zero
  · exact Or.inr ⟨flatCopyPos_zero ht h hr regs g hz, flatCopyPos_zero ht h hr regs g (hkey ▸ hz)⟩
  · have e : ∀ i', flatKey st.c (k.done.getD k.unit default).inCols (g.val % st.c.upv) i' ≠ .zero →
        flatSrcAt st.c g ((k.done.getD k.unit default).inCols.getD i' 0) =
          realize st.c (g.val / st.c.upv) (flatKey st.c (k.done.getD k.unit default).inCols (g.val % st.c.upv) i') :=
      fun i' hne => by
        rcases flatCopySrc_key (k := k) g i' with ⟨hz', -⟩ | ⟨-, he⟩
        · exact absurd hz' hne
        · exact he
    have he : flatSrcAt st.c g ((k.done.getD k.unit default).inCols.getD i 0) =
        flatSrcAt st.c g ((k.done.getD k.unit default).inCols.getD j 0) := by
      rw [e i hz, e j (hkey ▸ hz), hkey]
    left
    unfold flatCopyPos
    simp only [he]

end flat

end FlockSoundness.Discharge.Aliased
