import Proofs.Flock.Soundness.Discharge.Aliased.Keys
import Proofs.Flock.Soundness.Discharge.Zero.Stack
import Proofs.Flock.Soundness.Discharge.Layout.OutShape
import Proofs.Flock.Soundness.Discharge.Hidden.Commit
import Proofs.Flock.Soundness.Discharge.Layout.KeyProg
import Proofs.Flock.Soundness.Discharge.Layout.KeysRow

/-!
# The committed circuit's keys from the statement circuit's

A templated session commits `cs`, its unit net `Tmpl.unitNet`'s, and its statement circuit is `HmOut.unrolled cs`, the
same circuit with the logical unit net. The two read the same input ports, leaves, cuts and wires; only a wire out of a
unit slot reads an output column, which the committed unit holds at `upos` of the logical one: its output groups are
the logical groups placed at `posMap` (`Net.Group.at`, `outsOk`), so each output port is as wide, bit `t` at `upos` of the
logical bit (`portCols_at`).

So Δ's source key in the committed circuit is the statement circuit's with a unit output column moved to `upos`
(`srcKey_commit`, `Src.atUnit`): inputs with one statement key have one committed key (`srcKey_eq_commit`), and an input
reads the zero in one exactly when it does in the other (`zeroIn_commit`). A flat circuit is its own statement circuit,
at `upos = id`.
-/

namespace FlockSoundness.Discharge.Hidden

open Flock
open Aliased (Src srcKey srcAt)
open Zero (ZeroIn)

/-! ## An output group placed at `pos` -/

theorem ports_closed : ∀ (L : List ℕ) (out : Array (ℕ × ℕ)) (s : ℕ),
    ((forIn L (out, s) (fun w r => pure (ForInStep.yield (r.1.push (r.2, w), r.2 + w))) : Id _).1.toList) =
      out.toList ++ (List.range L.length).map (fun i => (s + (L.take i).sum, L.getD i 0))
  | [], out, s => by
    simp only [List.forIn_nil, List.length_nil, List.range_zero, List.map_nil, List.append_nil]
    rfl
  | w :: L, out, s => by
    rw [List.forIn_cons]
    show ((forIn L (out.push (s, w), s + w) _ : Id _).1.toList) = _
    rw [ports_closed L (out.push (s, w)) (s + w)]
    simp only [Array.toList_push, List.append_assoc, List.length_cons, List.range_succ_eq_map, List.map_cons,
      List.map_map, List.take_zero, List.sum_nil, Nat.add_zero, List.getD_cons_zero, List.cons_append]
    congr 2
    apply List.map_congr_left
    intro i _
    simp only [Function.comp_apply, List.take_succ_cons, List.sum_cons, List.getD_cons_succ, Prod.mk.injEq]
    exact ⟨by omega, trivial⟩

/-- One group's ports: bit `i` of its widths starts past the widths before it. -/
theorem portCols_one_closed (g : Group) :
    (portCols #[g]).toList =
      (List.range g.bits.size).map (fun i => (g.col * WORD + (g.bits.toList.take i).sum, g.bits.toList.getD i 0)) := by
  unfold portCols
  simp only
  rw [← Array.forIn_toList]
  simp only [List.forIn_cons, List.forIn_nil]
  rw [← Array.forIn_toList]
  have := ports_closed g.bits.toList #[] (g.col * WORD)
  simp only [List.nil_append, Array.length_toList] at this
  exact this

theorem take_sum_getD_le : ∀ (L : List ℕ) (i : ℕ), i < L.length → (L.take i).sum + L.getD i 0 ≤ L.sum
  | [], _, h => absurd h (Nat.not_lt_zero _)
  | w :: L, 0, _ => by simp
  | w :: L, i + 1, h => by
    have := take_sum_getD_le L i (by simp at h; omega)
    simp only [List.take_succ_cons, List.sum_cons, List.getD_cons_succ]
    omega

/-- **Port `q'` is logical port `q` placed at `pos`**: as wide, its bit `t` at `pos` of `q`'s. -/
def PortAt (pos : Array ℕ) (q q' : ℕ × ℕ) : Prop := q'.2 = q.2 ∧ ∀ t < q.2, q'.1 + t = pos.getD (q.1 + t) 0

theorem portCols_one_at {pos : Array ℕ} {g : Group}
    (h0 : pos.getD (g.col * WORD) 0 % (g.words * WORD) = 0)
    (hin : ∀ i < g.words * WORD, pos.getD (g.col * WORD + i) 0 = pos.getD (g.col * WORD) 0 + i)
    (hfit : g.bits.toList.sum ≤ g.words * WORD) :
    (portCols #[Net.Group.at pos g]).toList.length = (portCols #[g]).toList.length ∧
      ∀ j (h : j < (portCols #[g]).toList.length) (h' : j < (portCols #[Net.Group.at pos g]).toList.length),
        PortAt pos ((portCols #[g]).toList[j]) ((portCols #[Net.Group.at pos g]).toList[j]) := by
  have hd : pos.getD (g.col * WORD) 0 / WORD * WORD = pos.getD (g.col * WORD) 0 := by
    have : WORD ∣ pos.getD (g.col * WORD) 0 :=
      Nat.dvd_trans (Nat.dvd_mul_left WORD g.words) (Nat.dvd_of_mod_eq_zero h0)
    exact Nat.div_mul_cancel this
  rw [portCols_one_closed, portCols_one_closed]
  refine ⟨by simp [Net.Group.at], fun j h h' => ?_⟩
  simp only [List.length_map, List.length_range] at h
  simp only [List.getElem_map, List.getElem_range, Net.Group.at]
  refine ⟨rfl, fun t ht => ?_⟩
  have hle := take_sum_getD_le g.bits.toList j (by simpa using h)
  dsimp only at ht ⊢
  rw [hd, Nat.add_assoc, Nat.add_assoc, hin _ (by omega)]

/-- A group `outsOk` places at `pos`, which its widths fit. -/
def AtOk (pos : Array ℕ) (g : Group) : Prop :=
  pos.getD (g.col * WORD) 0 % (g.words * WORD) = 0 ∧
    (∀ i < g.words * WORD, pos.getD (g.col * WORD + i) 0 = pos.getD (g.col * WORD) 0 + i) ∧
    g.bits.toList.sum ≤ g.words * WORD

theorem flatMap_at {pos : Array ℕ} : ∀ (A : List Group), (∀ g ∈ A, AtOk pos g) →
    (A.flatMap fun g => (portCols #[Net.Group.at pos g]).toList).length = (A.flatMap fun g => (portCols #[g]).toList).length ∧
      ∀ j (h : j < (A.flatMap fun g => (portCols #[g]).toList).length)
        (h' : j < (A.flatMap fun g => (portCols #[Net.Group.at pos g]).toList).length),
        PortAt pos ((A.flatMap fun g => (portCols #[g]).toList)[j]) ((A.flatMap fun g => (portCols #[Net.Group.at pos g]).toList)[j])
  | [], _ => ⟨rfl, fun j h => absurd h (by simp)⟩
  | g :: A, hA => by
    obtain ⟨h0, hin, hfit⟩ := hA g (List.mem_cons_self ..)
    obtain ⟨hl1, hp1⟩ := portCols_one_at h0 hin hfit
    obtain ⟨hl2, hp2⟩ := flatMap_at A fun g' hg' => hA g' (List.mem_cons_of_mem _ hg')
    simp only [List.flatMap_cons, List.length_append]
    refine ⟨by rw [hl1, hl2], fun j h h' => ?_⟩
    by_cases hj : j < (portCols #[g]).toList.length
    · rw [List.getElem_append_left hj, List.getElem_append_left (by rw [hl1]; exact hj)]
      exact hp1 j hj _
    · have hj' : (portCols #[g]).toList.length ≤ j := Nat.le_of_not_lt hj
      rw [List.getElem_append_right hj', List.getElem_append_right (by rw [hl1]; exact hj'),
        getElem_congr_idx (show j - (portCols #[Net.Group.at pos g]).toList.length = j - (portCols #[g]).toList.length by
          rw [hl1])]
      exact hp2 _ (by omega) _

/-- **Output groups placed at `pos`** (`outsOk`, as `Net.ofRowsAt` places them): as many ports, each the logical port
placed at `pos`. -/
theorem portCols_at {pos : Array ℕ} {gs : Array Group} (hout : Net.outsOk pos gs = true)
    (hfit : ∀ g ∈ gs.toList, g.bits.foldl (· + ·) 0 ≤ g.words * WORD) :
    (portCols (gs.map (Net.Group.at pos))).size = (portCols gs).size ∧
      ∀ j < (portCols gs).size,
        PortAt pos ((portCols gs).getD j default) ((portCols (gs.map (Net.Group.at pos))).getD j default) := by
  have hok : ∀ g ∈ gs.toList, AtOk pos g := by
    intro g hg
    unfold Net.outsOk at hout
    simp only [Array.all_eq_true', Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.mem_range] at hout
    obtain ⟨h0, hin⟩ := hout g (Array.mem_toList_iff.1 hg)
    refine ⟨h0, hin, ?_⟩
    rw [List.sum_eq_foldl, Array.foldl_toList]
    exact hfit g hg
  obtain ⟨hl, hp⟩ := flatMap_at gs.toList hok
  have e1 := portCols_toList gs
  have e2 := portCols_toList (gs.map (Net.Group.at pos))
  rw [Array.toList_map, List.flatMap_map] at e2
  have hs : (portCols (gs.map (Net.Group.at pos))).size = (portCols gs).size := by
    rw [← Array.length_toList, ← Array.length_toList, e1, e2, hl]
  refine ⟨hs, fun j hj => ?_⟩
  have hj1 : j < (gs.toList.flatMap fun g => (portCols #[g]).toList).length := by
    rw [← e1, Array.length_toList]; exact hj
  have hj2 : j < (gs.toList.flatMap fun g => (portCols #[Net.Group.at pos g]).toList).length := by
    rw [hl]; exact hj1
  have g1 : (portCols gs).getD j default = (gs.toList.flatMap fun g => (portCols #[g]).toList)[j] := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hj, Option.getD_some, ← Array.getElem_toList]
    exact List.getElem_of_eq e1 _
  have g2 : (portCols (gs.map (Net.Group.at pos))).getD j default =
      (gs.toList.flatMap fun g => (portCols #[Net.Group.at pos g]).toList)[j] := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (by rw [hs]; exact hj), Option.getD_some,
      ← Array.getElem_toList]
    exact List.getElem_of_eq e2 _
  rw [g1, g2]
  exact hp j hj1 hj2

/-! ## The committed unit and the logical one -/

theorem getD_set_unit {α : Type} [Inhabited α] (xs : Array α) (u i : ℕ) (x : α) (hu : u < xs.size) :
    (xs.set! u x).getD i default = if i = u then x else xs.getD i default := by
  simp only [Array.set!_eq_setIfInBounds, Array.getD_eq_getD_getElem?, Array.getElem?_setIfInBounds]
  by_cases h : u = i
  · subst h; simp [hu]
  · simp [h, Ne.symm h]

theorem arr_getElem!_getD {α : Type} [Inhabited α] (a : Array α) (i : ℕ) : a[i]! = a.getD i default := by
  by_cases h : i < a.size
  · rw [getElem!_pos a i h, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem h, Option.getD_some]
  · rw [getElem!_neg a i h, Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega), Option.getD_none]

/-- **A templated committed circuit and its statement circuit**: the statement circuit is `cs` with the unit net the
logical net `n`, which reads `cs`'s unit's inputs; the committed unit's output groups are `n`'s placed at `pos`, which
`upos` reads. -/
theorem tmpl_unit_at {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {cs : Circuit}
    (hT : HmOut.parseTmpl tags cf tables = .ok cs) :
    ∃ (n : Net) (pos : Array ℕ), cs.unitNet < cs.nets.size ∧
      HmOut.unrolled cs = { cs with nets := cs.nets.set! cs.unitNet ("unit", n), tmpl := none } ∧
      upos cs = (fun x => pos.getD x 0) ∧ n.inGroups = cs.unit.inGroups ∧ n.inWords = cs.unit.inWords ∧
      Built n ∧ cs.unit.outGroups = n.outGroups.map (Net.Group.at pos) ∧
      (portCols (n.outGroups.map (Net.Group.at pos))).size = (portCols n.outGroups).size ∧
      (∀ j < (portCols n.outGroups).size,
        PortAt pos ((portCols n.outGroups).getD j default) ((portCols (n.outGroups.map (Net.Group.at pos))).getD j default)) ∧
      Net.outsOk pos n.outGroups = true := by
  obtain ⟨tu, n, ht, hn, hl, -, hsz, hu, he⟩ := unrolled_of_parseTmpl hT
  have A := UnitAt.tmpl ht hu
  have hB : Built n := hl ▸ built_tmpl ht
  obtain ⟨acc, n', -, hn', hat, hun⟩ := TmplValue.unroll_of_unrollAt tu.unrolled
  have e1 : n' = n := Except.ok.inj (hun.symm.trans hn)
  obtain ⟨n₀, -, -, -, -, -, -, -, -, -, -, hog, -, hok, -⟩ := Net.ofRowsAt_ok hat
  have hgo : n.outGroups = tu.t.outGroups := by rw [← e1]; exact (Net.ofRows_ok hn').2.2.2.2.2.2.1
  have hP := portCols_at hok fun g hg =>
    ((FlockSoundness.Discharge.Layout.built_out hB).2 g (by rw [hgo]; exact hg)).2
  rw [← hgo] at hog hP hok
  refine ⟨n, Tmpl.posMap tu.t tu.lays, hsz, he, upos_tmpl ht, by rw [← hl]; exact A.groups,
    by rw [← hl]; exact A.in_eq, hB, by rw [hu]; exact hog, hP.1, hP.2, hok⟩

/-- **A templated unit's output column is `upos` of its logical one**: the committed output group is the logical one
placed at `posMap` (`outsOk`), each of its bits at `posMap` of the logical bit. -/
theorem outCol_tmpl {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {cs : Circuit}
    (hT : HmOut.parseTmpl tags cf tables = .ok cs) {j t : ℕ}
    (hi : 16 * j + t < ((HmOut.unrolled cs).unit.outGroups.getD 0 default).words * WORD) :
    Layout.outCol cs j t = upos cs (Layout.outCol (HmOut.unrolled cs) j t) := by
  obtain ⟨n, pos, hsz, he, hup, -, -, hB, hog, -, -, hok⟩ := tmpl_unit_at hT
  have hU : ({ cs with nets := cs.nets.set! cs.unitNet ("unit", n), tmpl := none } : Circuit).unit = n := by
    show ((cs.nets.set! cs.unitNet ("unit", n)).getD cs.unitNet default).2 = n
    rw [getD_set_unit _ _ _ _ hsz, ite_eq_left rfl]
  rw [he, hU] at hi
  unfold Layout.outCol
  rw [he, hU, hog, hup]
  have hne : 0 < n.outGroups.size := Nat.pos_of_ne_zero (FlockSoundness.Discharge.Layout.built_out hB).1
  have e0 : n.outGroups.getD 0 default = n.outGroups[0] := by simp [Array.getD_eq_getD_getElem?, hne]
  have e1 : (n.outGroups.map (Net.Group.at pos)).getD 0 default = Net.Group.at pos n.outGroups[0] := by
    simp [Array.getD_eq_getD_getElem?, hne]
  rw [e1, e0]
  rw [e0] at hi
  unfold Net.outsOk at hok
  simp only [Array.all_eq_true', Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.mem_range] at hok
  obtain ⟨h0, hin⟩ := hok _ (Array.getElem_mem hne)
  have hd : pos.getD (n.outGroups[0].col * WORD) 0 / WORD * WORD = pos.getD (n.outGroups[0].col * WORD) 0 :=
    Nat.div_mul_cancel (Nat.dvd_trans (Nat.dvd_mul_left WORD _) (Nat.dvd_of_mod_eq_zero h0))
  simp only [Net.Group.at]
  rw [hd, hin _ hi]

/-- **Δ's committed key with a unit output column moved by `f`**: every other key is the same. -/
def _root_.FlockSoundness.Discharge.Aliased.Src.atUnit (u : ℕ) (f : ℕ → ℕ) : Aliased.Src → Aliased.Src
  | .wire m i col => .wire m i (if m = u then f col else col)
  | s => s

theorem atUnit_id (u : ℕ) (s : Aliased.Src) : s.atUnit u id = s := by
  cases s <;> simp [Aliased.Src.atUnit]

theorem atUnit_zero {u : ℕ} {f : ℕ → ℕ} {s : Aliased.Src} : s.atUnit u f = .zero ↔ s = .zero := by
  cases s <;> simp [Aliased.Src.atUnit]

theorem atUnit_msg {u : ℕ} {f : ℕ → ℕ} {s : Aliased.Src} {q byte bit : ℕ} :
    s.atUnit u f = .msg q byte bit ↔ s = .msg q byte bit := by
  cases s <;> simp [Aliased.Src.atUnit]

theorem srcAt_unrolled {cs : Circuit} {n : Net} {pos : Array ℕ} {k p t : ℕ}
    (hsz : cs.unitNet < cs.nets.size) (hg : n.inGroups = cs.unit.inGroups)
    (hog : cs.unit.outGroups = n.outGroups.map (Net.Group.at pos))
    (hat : (portCols (n.outGroups.map (Net.Group.at pos))).size = (portCols n.outGroups).size ∧
      ∀ j < (portCols n.outGroups).size,
        PortAt pos ((portCols n.outGroups).getD j default) ((portCols (n.outGroups.map (Net.Group.at pos))).getD j default))
    (hwire : ∀ w ∈ cs.wires.toList, w.dst = (cs.unitNet, k, p) →
      w.src.2.2 < (portCols (cs.nets.getD w.src.1 default).2.outGroups).size ∧
        t < ((portCols (cs.nets.getD w.src.1 default).2.outGroups).getD w.src.2.2 default).2) :
    srcAt cs k p t = (srcAt { cs with nets := cs.nets.set! cs.unitNet ("unit", n), tmpl := none } k p t).atUnit
      cs.unitNet (fun x => pos.getD x 0) := by
  have hU : ({ cs with nets := cs.nets.set! cs.unitNet ("unit", n), tmpl := none } : Circuit).unit = n := by
    show ((cs.nets.set! cs.unitNet ("unit", n)).getD cs.unitNet default).2 = n
    rw [getD_set_unit _ _ _ _ hsz, ite_eq_left rfl]
  have hst : Typed.starts { cs with nets := cs.nets.set! cs.unitNet ("unit", n), tmpl := none } =
      Typed.starts cs := rfl
  unfold srcAt
  rw [hU, hg, hst]
  dsimp only
  by_cases hp0 : p < (cs.unit.inGroups.getD 0 default).bits.size
  · simp only [hp0, ↓reduceIte]
    split
    · rfl
    · split <;> rfl
  · simp only [hp0, ↓reduceIte]
    split
    · split <;> rfl
    · rename_i hf
      split
      · rename_i w l' hw
        have hm : w ∈ cs.wires.toList.filter (fun w => decide (w.dst = (cs.unitNet, k, p))) := by
          rw [hw]; exact List.mem_cons_self ..
        obtain ⟨hmem, hd⟩ := List.mem_filter.1 hm
        obtain ⟨hj, htw⟩ := hwire w hmem (of_decide_eq_true hd)
        simp only [hw, Aliased.Src.atUnit, getD_set_unit _ _ _ _ hsz]
        by_cases hu : w.src.1 = cs.unitNet
        · simp only [hu, ↓reduceIte, Aliased.Src.wire.injEq, true_and]
          have eu : (cs.nets.getD cs.unitNet default).2 = cs.unit := rfl
          rw [hu, eu] at hj htw
          show ((portCols cs.unit.outGroups).getD w.src.2.2 default).1 + t = _
          rw [hog] at hj htw ⊢
          rw [hat.1] at hj
          obtain ⟨e2, e1⟩ := hat.2 _ hj
          exact e1 t (by rw [← e2]; exact htw)
        · simp only [hu, ↓reduceIte]
      · rename_i hw
        simp only [hw]
        rfl

/-- **A templated session's committed key is its statement key with a unit output column moved to `upos`.** -/
theorem srcKey_tmpl {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {cs : Circuit}
    (hT : HmOut.parseTmpl tags cf tables = .ok cs) (C : CheckFacts cs) (x : ℕ) :
    srcKey cs x = (srcKey (HmOut.unrolled cs) x).atUnit cs.unitNet (upos cs) := by
  obtain ⟨n, pos, hsz, he, hup, hg, hw, hB, hog, hat1, hat2, -⟩ := tmpl_unit_at hT
  have hat := And.intro hat1 hat2
  have hnd : (portBits cs.unit.inGroups).Nodup := by rw [← hg]; exact Aliased.portBits_nodup hB
  have hU : ({ cs with nets := cs.nets.set! cs.unitNet ("unit", n), tmpl := none } : Circuit).unit = n := by
    show ((cs.nets.set! cs.unitNet ("unit", n)).getD cs.unitNet default).2 = n
    rw [getD_set_unit _ _ _ _ hsz, ite_eq_left rfl]
  rw [he, hup]
  unfold srcKey
  rw [hU, hg, hw]
  by_cases hm : x % (cs.unit.inWords * WORD) ∈ portBits cs.unit.inGroups
  · rw [ite_eq_left hm, ite_eq_left hm]
    obtain ⟨p, t, hp, ht, hpt⟩ := Aliased.mem_portBits hm
    rw [hpt, portOf_eq hnd hp ht, Nat.add_sub_cancel_left]
    refine srcAt_unrolled hsz hg hog hat fun w hw hd => ?_
    obtain ⟨-, -, hj, hwid⟩ := C.wire_src w hw
    refine ⟨hj, ?_⟩
    rw [hd] at hwid
    rw [← arr_getElem!_getD, ← hwid]
    exact ht
  · rw [ite_eq_right hm, ite_eq_right hm]
    rfl

/-- A templated session's input reads the zero in its committed circuit exactly when it does in its statement
circuit: the two read the same input ports, leaves and cuts. -/
theorem zeroIn_tmpl {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {cs : Circuit}
    (hT : HmOut.parseTmpl tags cf tables = .ok cs) (x : ℕ) : ZeroIn (HmOut.unrolled cs) x ↔ ZeroIn cs x := by
  obtain ⟨n, pos, hsz, he, -, hg, hw, -⟩ := tmpl_unit_at hT
  have hU : ({ cs with nets := cs.nets.set! cs.unitNet ("unit", n), tmpl := none } : Circuit).unit = n := by
    show ((cs.nets.set! cs.unitNet ("unit", n)).getD cs.unitNet default).2 = n
    rw [getD_set_unit _ _ _ _ hsz, ite_eq_left rfl]
  rw [he]
  unfold ZeroIn Zero.InPort Zero.ZeroAt Zero.ZeroPort
  rw [hU, hg, hw]
  exact Iff.rfl

theorem srcKey_withTmpl (c : Circuit) (t : Option TmplUnit) (x : ℕ) : srcKey { c with tmpl := t } x = srcKey c x :=
  rfl

/-- **`KeysRow` at a templated session's statement circuit**: its unit has only its leaf group, and every input a unit
row reads is a port bit (`PortReads`). A row key is the zero or a message bit, which the committed circuit's key is too
(`Layout.rowKey_of_cutsPre`), and `srcKey_tmpl` moves only wire keys. -/
theorem keysRow_of_tmpl {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {cs : Circuit}
    (hty : tags.typed = false) (hT : HmOut.parseTmpl tags cf tables = .ok cs) (h1 : cs.unit.inGroups.size = 1)
    {hU : Built (HmOut.unrolled cs).unit} (hpr : Layout.PortReads (HmOut.unrolled cs) hU) :
    Layout.KeysRow (HmOut.unrolled cs) hU (Layout.portRowW (HmOut.unrolled cs)) := by
  have C : CheckFacts cs := by
    obtain ⟨hty', ⟨ls, hls⟩, -⟩ := Committed.check hty (Or.inr hT)
    exact check_facts hty' hls
  obtain ⟨n, pos, hsz, heu, -, hg, hw, hB, -⟩ := tmpl_unit_at hT
  have hU' : (HmOut.unrolled cs).unit = n := by
    rw [heu]
    show ((cs.nets.set! cs.unitNet ("unit", n)).getD cs.unitNet default).2 = n
    rw [getD_set_unit _ _ _ _ hsz, ite_eq_left rfl]
  refine Layout.keysRow_of_rowKey hpr fun x hm hu => ?_
  rw [hU', hg, hw] at hm
  rw [hU', hw, heu] at hu
  have hkc := srcKey_tmpl hT C x
  unfold Layout.RowKey
  rw [heu] at hkc ⊢
  obtain ⟨tu, c₀, hp0, -, -, he⟩ := TmplValue.parseTmpl_cases hT
  subst he
  rw [srcKey_withTmpl c₀ (some tu) x] at hkc
  have hnd : (portBits c₀.unit.inGroups).Nodup := by
    have := Aliased.portBits_nodup hB
    rw [hg] at this
    exact this
  rcases Layout.rowKey_of_cutsPre hp0 (Layout.cutsFromRows_of_singlePre hp0 h1) hnd hm hu with
    h0 | ⟨q, byte, bit, hq0, hq, hlt, hb⟩
  · exact Or.inl (atUnit_zero.1 (hkc.symm.trans h0))
  · exact Or.inr ⟨q, byte, bit, atUnit_msg.1 (hkc.symm.trans hq0), hq, hlt, hb⟩

/-- **`KeysRow` at a session's statement circuit**, flat (`Layout.keysRow_of_parse`) or templated (`keysRow_of_tmpl`). -/
theorem keysRow_commit {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {cs : Circuit}
    (hty : tags.typed = false) (h : Committed tags cf tables cs) (h1 : cs.unit.inGroups.size = 1)
    {hU : Built (HmOut.unrolled cs).unit} (hpr : Layout.PortReads (HmOut.unrolled cs) hU) :
    Layout.KeysRow (HmOut.unrolled cs) hU (Layout.portRowW (HmOut.unrolled cs)) := by
  rcases h with h | h
  · have hf := parse_tmpl_none hty h
    revert hU
    rw [unrolled_flat hf]
    intro hU hpr
    exact Layout.keysRow_of_parse (parse_untyped hty h).1 h1 hpr
  · exact keysRow_of_tmpl hty h h1 hpr

/-- **The statement circuit's unit has the committed unit's input groups.** -/
theorem Committed.unrolled_inGroups {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {cs : Circuit}
    (hty : tags.typed = false) (h : Committed tags cf tables cs) :
    (HmOut.unrolled cs).unit.inGroups = cs.unit.inGroups := by
  rcases h with h | h
  · rw [unrolled_flat (parse_tmpl_none hty h)]
  · obtain ⟨n, _pos, hsz, heu, -, hg, -⟩ := tmpl_unit_at h
    rw [heu]
    show ((cs.nets.set! cs.unitNet ("unit", n)).getD cs.unitNet default).2.inGroups = _
    rw [getD_set_unit _ _ _ _ hsz, ite_eq_left rfl, hg]

/-- **`KeysRow` at a session's statement circuit** (`HmOut.parseStmt`, untyped tags): its committed circuit's
(`keysRow_commit`), when the statement's unit has only its leaf group. -/
theorem keysRow_stmt {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (hty : tags.typed = false) (h : HmOut.parseStmt tags cf tables = .ok c) (h1 : c.unit.inGroups.size = 1)
    {hU : Built c.unit} (hpr : Layout.PortReads c hU) : Layout.KeysRow c hU (Layout.portRowW c) := by
  obtain ⟨cs, hcs, rfl⟩ := commit_of_parseStmt hty h
  have hC := committed_of_commit hcs
  exact keysRow_commit hty hC (by rw [← hC.unrolled_inGroups hty]; exact h1) hpr

/-! ## On the committed circuit, flat or templated -/

variable {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {cs : Circuit}

/-- **A session's committed key is its statement key with a unit output column moved to `upos`**, flat (`upos = id`)
or templated (`srcKey_tmpl`). -/
theorem srcKey_commit (hty : tags.typed = false) (h : Committed tags cf tables cs) (C : CheckFacts cs) (x : ℕ) :
    srcKey cs x = (srcKey (HmOut.unrolled cs) x).atUnit cs.unitNet (upos cs) := by
  rcases h with h | h
  · have hf := parse_tmpl_none hty h
    rw [unrolled_flat hf, show upos cs = id from funext (upos_flat hf), atUnit_id]
  · exact srcKey_tmpl h C x

/-- **Inputs with one statement key have one committed key.** -/
theorem srcKey_eq_commit (hty : tags.typed = false) (h : Committed tags cf tables cs) (C : CheckFacts cs) {x x' : ℕ}
    (hk : srcKey (HmOut.unrolled cs) x = srcKey (HmOut.unrolled cs) x') : srcKey cs x = srcKey cs x' := by
  rw [srcKey_commit hty h C, srcKey_commit hty h C, hk]

/-- **An input reads the zero in the committed circuit exactly when it does in the statement circuit.** -/
theorem zeroIn_commit (hty : tags.typed = false) (h : Committed tags cf tables cs) (x : ℕ) :
    ZeroIn (HmOut.unrolled cs) x ↔ ZeroIn cs x := by
  rcases h with h | h
  · rw [unrolled_flat (parse_tmpl_none hty h)]
  · exact zeroIn_tmpl h x

/-- **A unit's committed output column is `upos` of its statement one**, flat or templated (`outCol_tmpl`). -/
theorem outCol_commit (hty : tags.typed = false) (h : Committed tags cf tables cs) {j t : ℕ}
    (hi : 16 * j + t < ((HmOut.unrolled cs).unit.outGroups.getD 0 default).words * WORD) :
    Layout.outCol cs j t = upos cs (Layout.outCol (HmOut.unrolled cs) j t) := by
  rcases h with h | h
  · have hf := parse_tmpl_none hty h
    rw [unrolled_flat hf, upos_flat hf]
  · exact outCol_tmpl h hi

/-- **A committed circuit's message columns are in the block** (`Layout.msgCol_lt_of_parsePre`). -/
theorem Committed.msgCol_lt (hty : tags.typed = false) (h : Committed tags cf tables cs) {g : ℕ} (hg : g < cs.g)
    {p : ℕ} (hp : p < cs.ports.size) (byte b : ℕ) (hb : b < 8) : Typed.msgCol cs g p byte b < 2 ^ cs.kLog := by
  rcases h with h | h
  · exact Layout.msgCol_lt_of_parse (parse_untyped hty h).1 hg hp byte b hb
  · obtain ⟨tu, c₀, h0, -, -, he⟩ := TmplValue.parseTmpl_cases h
    rw [he] at hg hp ⊢
    exact Layout.msgCol_lt_of_parsePre h0 (fun x hx => by
      obtain rfl : x = ("unit", tu.net) := by simpa using hx
      exact tu.netOK) hg hp byte b hb


theorem foldl_add16 : ∀ (m s : ℕ), (List.replicate m 16).foldl (· + ·) s = s + 16 * m
  | 0, s => by simp
  | m + 1, s => by rw [List.replicate_succ, List.foldl_cons, foldl_add16 m (s + 16)]; omega

/-- **A unit's committed output column is `upos` of its statement one**, at an output of 16-bit outputs. -/
theorem outCol_commit16 (hty : tags.typed = false) (h : Committed tags cf tables cs) {j t : ℕ}
    (h16 : ∀ b ∈ (cs.unit.outGroups.getD 0 default).bits.toList, b = 16)
    (hj : j < (cs.unit.outGroups.getD 0 default).bits.size) (ht : t < 16) :
    Layout.outCol cs j t = upos cs (Layout.outCol (HmOut.unrolled cs) j t) := by
  rcases h with h | h
  · have hf := parse_tmpl_none hty h
    rw [unrolled_flat hf, upos_flat hf]
  · refine outCol_tmpl h ?_
    obtain ⟨n, pos, hsz, he, -, -, -, hB, hog, -, -, -⟩ := tmpl_unit_at h
    have hU : ({ cs with nets := cs.nets.set! cs.unitNet ("unit", n), tmpl := none } : Circuit).unit = n := by
      show ((cs.nets.set! cs.unitNet ("unit", n)).getD cs.unitNet default).2 = n
      rw [getD_set_unit _ _ _ _ hsz, ite_eq_left rfl]
    rw [he, hU]
    have hne : 0 < n.outGroups.size := Nat.pos_of_ne_zero (FlockSoundness.Discharge.Layout.built_out hB).1
    have e0 : n.outGroups.getD 0 default = n.outGroups[0] := by simp [Array.getD_eq_getD_getElem?, hne]
    have e1 : cs.unit.outGroups.getD 0 default = Net.Group.at pos n.outGroups[0] := by
      rw [hog]; simp [Array.getD_eq_getD_getElem?, hne]
    rw [e1] at h16 hj
    rw [e0]
    have hfit := ((FlockSoundness.Discharge.Layout.built_out hB).2 _ (Array.getElem_mem_toList hne)).2
    have hr : n.outGroups[0].bits.toList = List.replicate n.outGroups[0].bits.size 16 :=
      List.eq_replicate_iff.2 ⟨Array.length_toList, h16⟩
    have hs : n.outGroups[0].bits.foldl (· + ·) 0 = 16 * n.outGroups[0].bits.size := by
      rw [← Array.foldl_toList, hr, foldl_add16]; omega
    change j < n.outGroups[0].bits.size at hj
    omega

end FlockSoundness.Discharge.Hidden
