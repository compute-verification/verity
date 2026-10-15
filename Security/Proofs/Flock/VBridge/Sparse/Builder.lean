import Proofs.Flock.VBridge.Sparse.Steps
import Proofs.Flock.VBridge.Sparse.Residuals
import Proofs.Flock.VBridge.Pinned

/-!
# The sparse-phase unit's residuals are `Defs`' structure

`Flock.RecSparse.resList` builds the residuals `steps` reads in a `StateM` over the operand keys. Each one comes out
named `str nm` for `nm` in `names`, in order, and is `put` on keys whose terms are `spec nm`'s (`resList_spec`): the
message ports' offsets are `Defs`' `w` offsets (`mp_offsets`) and the verifier row's parts' are its `v` offsets
(`vp_all`). So the names `steps` reads are the builder's (`used_mem`, `sound_steps`' hypothesis), and residual `nm`'s
value at `w`, `v` (`rval`) is `Defs`' `rget` of `resArr` (`rval_rget`).
-/

namespace FlockVBridge.Sparse

open Flock (F128)
open Flock.RecSparse (messagePorts vParts offset Key op put resList structureOf)
open FlockSoundness.Model (SparseShape)
open FlockSoundness.Refine (fOf fOf_add fOf_mul fOf_zero fOf_injective)

/-! ## The shape -/

/-- The model's shape at the builder's. -/
def shOf (sh : Flock.RecSparse.Shape) : SparseShape := ⟨sh.nu, sh.ell, sh.sLog, sh.tLog⟩

section Shape

variable (sh : Flock.RecSparse.Shape)

@[simp] theorem shOf_nu : (shOf sh).nu = sh.nu := rfl
@[simp] theorem shOf_ell : (shOf sh).ell = sh.ell := rfl
@[simp] theorem shOf_sLog : (shOf sh).sLog = sh.sLog := rfl
@[simp] theorem shOf_tLog : (shOf sh).tLog = sh.tLog := rfl

theorem pLog_shOf : (shOf sh).pLog = sh.pLog := by
  simp only [SparseShape.pLog, Flock.RecSparse.Shape.pLog, Flock.RecSparse.Shape.P, shOf]
  rw [show 2 * sh.ell + 2 - 1 = 2 * sh.ell + 1 by omega]

theorem tLog_ne (hs : sh.stacked = true) : (shOf sh).tLog ≠ 0 := by
  simp only [Flock.RecSparse.Shape.stacked, decide_eq_true_eq] at hs
  simp only [shOf]; omega

theorem tLog_eq (hs : sh.stacked = false) : (shOf sh).tLog = 0 := by
  simp only [Flock.RecSparse.Shape.stacked, decide_eq_false_iff_not] at hs
  simp only [shOf]; omega

theorem degs_length : sh.degrees.length = nR (shOf sh) := by
  simp [Flock.RecSparse.Shape.degrees, nR, shOf]

theorem degs_getD {j : Nat} (hj : j < nR (shOf sh)) : sh.degrees.getD j 0 = deg (shOf sh) j := by
  simp only [nR, shOf] at hj
  by_cases h : j < sh.nu
  · simp [Flock.RecSparse.Shape.degrees, deg, shOf, Flock.RecSparse.Shape.d, List.getD_eq_getElem?_getD,
      List.getElem?_append, h]
  · simp [Flock.RecSparse.Shape.degrees, deg, shOf, Flock.RecSparse.Shape.d, List.getD_eq_getElem?_getD,
      List.getElem?_append, h, show j - sh.nu < sh.sLog by omega]

theorem degs_take : ∀ j, j ≤ nR (shOf sh) → (sh.degrees.take j).foldl (· + ·) 0 = degSum (shOf sh) j
  | 0, _ => by simp [degSum]
  | j + 1, hj => by
      have hl : j < sh.degrees.length := by rw [degs_length]; omega
      rw [List.take_add_one, List.getElem?_eq_getElem hl, Option.toList_some, List.foldl_append, List.foldl_cons,
        List.foldl_nil, degs_take j (by omega), degSum_succ, ← degs_getD sh (show j < nR (shOf sh) by omega),
        List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hl, Option.getD_some]

theorem sum_degs : sh.degrees.foldl (· + ·) 0 = degSum (shOf sh) (nR (shOf sh)) := by
  rw [← degs_take sh _ (Nat.le_refl _), ← degs_length sh, List.take_of_length_le (Nat.le_refl _)]

theorem sum_getD : ∀ j, j ≤ nR (shOf sh) → ((List.range j).map (sh.degrees.getD · 0)).sum = degSum (shOf sh) j
  | 0, _ => by simp [degSum]
  | j + 1, hj => by
      rw [List.range_succ, List.map_append, List.sum_append, sum_getD j (by omega), degSum_succ]
      simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero,
        degs_getD sh (show j < nR (shOf sh) by omega)]

end Shape

/-! ## Names -/

theorem head_idx (p : String) {c : Char} {cs : List Char} (hp : p.toList = c :: cs) (j : Nat) :
    (toString p ++ toString j).toList.head? = some c := by
  show (p ++ toString j).toList.head? = _
  rw [String.toList_append, hp]; rfl

theorem ne_idx (p : String) {c : Char} {cs : List Char} (hp : p.toList = c :: cs) (j : Nat) {nm : String}
    (h : nm.toList.head? ≠ some c) : toString p ++ toString j ≠ nm := fun e => h (by rw [← e, head_idx p hp j])

theorem idx_inj (p : String) {i j : Nat} (h : toString p ++ toString i = toString p ++ toString j) : i = j := by
  have h := congrArg String.toList h
  simp only [String.toList_append, toString, Nat.toList_repr] at h
  exact eq_of_toDigits (List.append_cancel_left h)

theorem take2_idx (p : String) (c d : Char) (cs : List Char) (hp : p.toList = c :: d :: cs) (j : Nat) :
    (toString p ++ toString j).toList.take 2 = [c, d] := by
  show (p ++ toString j).toList.take 2 = _
  rw [String.toList_append, hp]; rfl

/-- A residual's name in the builder (`products_structure`'s `names`). -/
def str : RName → String
  | .h1 => "h1"
  | .pre j => s!"pre{j}"
  | .wv => "wv"
  | .sc j => s!"sc{j}"
  | .f0 => "f0"
  | .fx j => s!"fx{j}"
  | .fy j => s!"fy{j}"
  | .bp => "bp"
  | .claim => "claim"

/-- A name's first two characters. -/
def tag : RName → List Char
  | .h1 => ['h', '1'] | .pre _ => ['p', 'r'] | .wv => ['w', 'v'] | .sc _ => ['s', 'c'] | .f0 => ['f', '0']
  | .fx _ => ['f', 'x'] | .fy _ => ['f', 'y'] | .bp => ['b', 'p'] | .claim => ['c', 'l']

theorem take2_str : ∀ r, (str r).toList.take 2 = tag r
  | .h1 => by decide
  | .pre j => take2_idx "pre" 'p' 'r' ['e'] (by decide) j
  | .wv => by decide
  | .sc j => take2_idx "sc" 's' 'c' [] (by decide) j
  | .f0 => by decide
  | .fx j => take2_idx "fx" 'f' 'x' [] (by decide) j
  | .fy j => take2_idx "fy" 'f' 'y' [] (by decide) j
  | .bp => by decide
  | .claim => by decide

theorem str_inj {a b : RName} (h : str a = str b) : a = b := by
  have ht : tag a = tag b := by rw [← take2_str, ← take2_str, h]
  cases a <;> cases b <;> simp only [str, tag] at h ht ⊢ <;>
    first | rfl | exact absurd ht (by decide) | exact congrArg _ (idx_inj _ h)

theorem idxOf_map_inj {α β : Type} [DecidableEq α] [DecidableEq β] (f : α → β) (hf : ∀ a b, f a = f b → a = b)
    (a : α) : ∀ l : List α, (l.map f).idxOf (f a) = l.idxOf a
  | [] => rfl
  | x :: l => by
      rw [List.map_cons, List.idxOf_cons, List.idxOf_cons, idxOf_map_inj f hf a l]
      by_cases h : x = a
      · subst h; simp
      · have h' : f x ≠ f a := fun e => h (hf _ _ e)
        simp [h, h']

/-! ## The ports' offsets -/

theorem offset_append (l₁ l₂ : List (String × Nat)) (nm : String) (h : ∀ p ∈ l₁, p.1 ≠ nm) :
    offset (l₁ ++ l₂) nm = (l₁.map (·.2)).sum + offset l₂ nm := by
  induction l₁ with
  | nil => simp
  | cons p l ih =>
    obtain ⟨n, k⟩ := p
    have hn : n ≠ nm := h _ List.mem_cons_self
    simp only [List.cons_append, offset, hn, ↓reduceIte, ih fun q hq => h q (List.mem_cons_of_mem _ hq),
      List.map_cons, List.sum_cons]
    omega

theorem offset_cons_ne {n nm : String} (k : Nat) (l : List (String × Nat)) (h : n ≠ nm) :
    offset ((n, k) :: l) nm = k + offset l nm := by simp [offset, h]

theorem offset_cons_self (nm : String) (k : Nat) (l : List (String × Nat)) : offset ((nm, k) :: l) nm = 0 := by
  simp [offset]

theorem offset_map_ne (f : Nat → String) (g : Nat → Nat) (n : Nat) (l : List (String × Nat)) (nm : String)
    (h : ∀ i, f i ≠ nm) : offset ((List.range n).map (fun i => (f i, g i)) ++ l) nm = ((List.range n).map g).sum + offset l nm := by
  rw [offset_append _ _ _ fun p hp => ?_]
  · simp [List.map_map, Function.comp_def]
  · obtain ⟨i, -, rfl⟩ := List.mem_map.1 hp; exact h i

theorem offset_map_range (f : Nat → String) (g : Nat → Nat) (hf : ∀ i j, f i = f j → i = j) {n j : Nat} (hj : j < n)
    (rest : List (String × Nat)) :
    offset ((List.range n).map (fun i => (f i, g i)) ++ rest) (f j) = ((List.range j).map g).sum := by
  obtain ⟨m, rfl⟩ : ∃ m, n = j + (m + 1) := ⟨n - j - 1, by omega⟩
  rw [List.range_add, List.range_succ_eq_map, List.map_append, List.append_assoc,
    offset_append _ _ _ fun p hp => ?_]
  · simp [offset, List.map_map, Function.comp_def]
  · simp only [List.mem_map, List.mem_range] at hp
    obtain ⟨i, hi, rfl⟩ := hp
    exact fun e => by have := hf _ _ e; omega

theorem sum_two : ∀ j, ((List.range j).map fun _ => 2).sum = 2 * j
  | 0 => rfl
  | j + 1 => by rw [List.range_succ, List.map_append, List.sum_append, sum_two j]; simp; omega

theorem map_eq_of_forall₂ {α β γ : Type} {f : α → γ} {g : β → γ} {l : List α} {m : List β}
    (h : List.Forall₂ (fun a b => f a = g b) l m) : l.map f = m.map g := by
  induction h with
  | nil => rfl
  | cons hab _ ih => rw [List.map_cons, List.map_cons, hab, ih]

section Ports

variable (sh : Flock.RecSparse.Shape)

theorem mp_eq : messagePorts sh = ("h", 64) :: ((List.range 6).map (fun j => (s!"pre{j}", 2)) ++ (("v2", 1) ::
    ((List.range sh.degrees.length).map (fun j => (s!"sc{j}", sh.degrees.getD j 0)) ++ (("u", sh.PP) ::
      ((if sh.stacked then [("vt", sh.T)] else []) ++ [("claim", 1)]))))) := by
  have hz : sh.degrees.zipIdx.map (fun (deg, j) => (s!"sc{j}", deg)) =
      (List.range sh.degrees.length).map (fun j => (s!"sc{j}", sh.degrees.getD j 0)) :=
    List.ext_getElem (by simp) fun i h1 h2 => by
      simp [List.getElem_zipIdx, List.getElem?_eq_getElem (show i < sh.degrees.length by simpa using h1)]
  simp only [Flock.RecSparse.messagePorts, hz, List.append_assoc, List.cons_append, List.nil_append,
    Flock.RecSparse.SKIP, Flock.RecSparse.PRE_ROUNDS]

theorem pre_ne (j : Nat) {nm : String} (h : nm.toList.head? ≠ some 'p') : s!"pre{j}" ≠ nm :=
  ne_idx "pre" (cs := ['r', 'e']) (by decide) j h

theorem sc_ne (j : Nat) {nm : String} (h : nm.toList.head? ≠ some 's') : s!"sc{j}" ≠ nm :=
  ne_idx "sc" (cs := ['c']) (by decide) j h

/-- The message ports' offsets are `Defs`' `w` offsets. -/
theorem mp_offsets : offset (messagePorts sh) "h" = 0 ∧
    (∀ j < 6, offset (messagePorts sh) s!"pre{j}" = 64 + 2 * j) ∧ offset (messagePorts sh) "v2" = 76 ∧
    (∀ j < nR (shOf sh), offset (messagePorts sh) s!"sc{j}" = wSc (shOf sh) j) ∧
    offset (messagePorts sh) "u" = wU (shOf sh) ∧
    (sh.stacked = true → offset (messagePorts sh) "vt" = wVt (shOf sh)) ∧
    offset (messagePorts sh) "claim" = wClaim (shOf sh) := by
  have hl := degs_length sh
  have pre12 : ((List.range 6).map fun _ => 2).sum = 12 := by decide
  have hsc : ((List.range sh.degrees.length).map (sh.degrees.getD · 0)).sum = degSum (shOf sh) (nR (shOf sh)) := by
    rw [hl]; exact sum_getD sh _ (Nat.le_refl _)
  have pre_sc : ∀ i j, s!"pre{i}" ≠ s!"sc{j}" := fun i j => pre_ne i (by rw [head_idx "sc" (c := 's') (cs := ['c']) (by decide)]; decide)
  rw [mp_eq]
  refine ⟨offset_cons_self _ _ _, fun j hj => ?_, ?_, fun j hj => ?_, ?_, fun hs => ?_, ?_⟩
  · rw [offset_cons_ne _ _ (pre_ne j (by decide)).symm,
      offset_map_range (fun j => s!"pre{j}") (fun _ => 2) (fun i j h => idx_inj "pre" h) hj]
    rw [sum_two]
  · rw [offset_cons_ne _ _ (by decide), offset_map_ne _ _ _ _ "v2" fun i => pre_ne i (by decide), offset_cons_self,
      pre12]
  · rw [offset_cons_ne _ _ (sc_ne j (by decide)).symm, offset_map_ne _ _ _ _ _ fun i => pre_sc i j,
      offset_cons_ne _ _ (sc_ne j (by decide)).symm,
      offset_map_range (fun j => s!"sc{j}") (sh.degrees.getD · 0) (fun i j h => idx_inj "sc" h) (by omega),
      pre12, sum_getD sh j (by omega), wSc]
    omega
  · rw [offset_cons_ne _ _ (by decide), offset_map_ne _ _ _ _ "u" fun i => pre_ne i (by decide),
      offset_cons_ne _ _ (by decide), offset_map_ne _ _ _ _ "u" fun i => sc_ne i (by decide), offset_cons_self,
      pre12, hsc, wU]
    omega
  · simp only [hs, ↓reduceIte, List.cons_append, List.nil_append]
    rw [offset_cons_ne _ _ (by decide), offset_map_ne _ _ _ _ "vt" fun i => pre_ne i (by decide),
      offset_cons_ne _ _ (by decide), offset_map_ne _ _ _ _ "vt" fun i => sc_ne i (by decide),
      offset_cons_ne _ _ (by decide), offset_cons_self, pre12, hsc]
    simp only [wVt, wU, Flock.RecSparse.Shape.PP, pLog_shOf]
    omega
  · rw [offset_cons_ne _ _ (by decide), offset_map_ne _ _ _ _ "claim" fun i => pre_ne i (by decide),
      offset_cons_ne _ _ (by decide), offset_map_ne _ _ _ _ "claim" fun i => sc_ne i (by decide),
      offset_cons_ne _ _ (by decide), pre12, hsc]
    simp only [wClaim, wVt, wU, tw, Flock.RecSparse.Shape.PP, Flock.RecSparse.Shape.T, pLog_shOf]
    cases hst : sh.stacked
    · simp only [tLog_eq sh hst, Bool.false_eq_true, ↓reduceIte, List.nil_append, offset_cons_self]
      omega
    · have ht : sh.tLog ≠ 0 := tLog_ne sh hst
      simp only [ht, ↓reduceIte, List.cons_append, List.nil_append, shOf_tLog]
      rw [offset_cons_ne _ _ (by decide), offset_cons_self]
      omega

/-- The verifier row's parts' offsets are `Defs`' `v` offsets. -/
theorem vp_all : offset (vParts sh) "eqc" = 0 ∧ offset (vParts sh) "pre" = 64 ∧ offset (vParts sh) "what" = vWhat ∧
    offset (vParts sh) "sc" = 77 ∧ offset (vParts sh) "alpha" = vAlpha (shOf sh) ∧
    offset (vParts sh) "ax" = vAx (shOf sh) ∧ offset (vParts sh) "ay" = vAy (shOf sh) ∧
    offset (vParts sh) "eqp" = vEqp (shOf sh) ∧ (sh.stacked = true → offset (vParts sh) "eqt" = vEqt (shOf sh)) := by
  have hs := sum_degs sh
  simp only [vAlpha, vAx, vAy, vUr, vEqp, vEqs, vEqt, vWhat, shOf_ell, shOf_sLog, pLog_shOf]
  cases h : sh.stacked <;>
    simp [vParts, offset, Flock.RecSparse.SKIP, Flock.RecSparse.PRE_ROUNDS, Flock.RecSparse.Shape.PP,
      Flock.RecSparse.Shape.S, h, hs] <;> omega

end Ports

/-! ## The builder's residuals -/

/-- `ops` extends `s`. -/
def Ext (s ops : Array Key) : Prop := s.toList <+: ops.toList

/-- `m` extends the operands and `Q` holds of its result at every extension of its final operands. -/
def Spec {α : Type} (m : StateM (Array Key) α) (Q : α → Array Key → Prop) : Prop :=
  ∀ s, Ext s (m.run s).2 ∧ ∀ F, Ext (m.run s).2 F → Q (m.run s).1 F

theorem Spec.pure' {α : Type} {a : α} {Q : α → Array Key → Prop} (h : ∀ F, Q a F) : Spec (pure a) Q :=
  fun _ => ⟨List.prefix_refl _, fun F _ => h F⟩

theorem Spec.bind' {α β : Type} {m : StateM (Array Key) α} {f : α → StateM (Array Key) β}
    {Q : α → Array Key → Prop} {R : β → Array Key → Prop} (hm : Spec m Q)
    (hf : ∀ a, Spec (f a) fun b F => Q a F → R b F) : Spec (m >>= f) R := fun s =>
  have h1 := hm s
  have h2 := hf (m.run s).1 (m.run s).2
  ⟨h1.1.trans h2.1, fun F hF => h2.2 F hF (h1.2 F (h2.1.trans hF))⟩

theorem Spec.mono {α : Type} {m : StateM (Array Key) α} {Q R : α → Array Key → Prop} (h : Spec m Q)
    (hQR : ∀ a F, Q a F → R a F) : Spec m R := fun s => ⟨(h s).1, fun F hF => hQR _ _ ((h s).2 F hF)⟩

theorem Spec.mapM {α β : Type} (f : β → StateM (Array Key) α) (Q : β → α → Array Key → Prop) :
    ∀ l : List β, (∀ x ∈ l, Spec (f x) (Q x)) → Spec (l.mapM f) fun ys F => List.Forall₂ (fun x y => Q x y F) l ys
  | [], _ => Spec.pure' fun _ => .nil
  | x :: l, h => by
      rw [List.mapM_cons]
      exact Spec.bind' (h x List.mem_cons_self) fun y => Spec.bind' (Spec.mapM f Q l fun z hz =>
        h z (List.mem_cons_of_mem _ hz)) fun ys => Spec.pure' fun _ hl hy => .cons hy hl

theorem getElem?_of_ext {s F : Array Key} (h : Ext s F) {i : Nat} {k : Key} (hi : s[i]? = some k) :
    F[i]? = some k := by
  obtain ⟨t, ht⟩ := h
  have hl : i < s.toList.length := by
    rw [← Array.getElem?_toList] at hi; exact (List.getElem?_eq_some_iff.1 hi).1
  rw [← Array.getElem?_toList, ← ht, List.getElem?_append_left hl, Array.getElem?_toList, hi]

theorem op_spec (k : Key) : Spec (op k) fun i F => F[i]? = some k := by
  intro s
  cases h : s.findIdx? (· == k) with
  | some i =>
    have hr : (op k).run s = (i, s) := by simp [Flock.RecSparse.op, h]; try rfl
    rw [hr]
    obtain ⟨hi, hk, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 h
    refine ⟨List.prefix_refl _, fun F hF => getElem?_of_ext hF ?_⟩
    rw [Array.getElem?_eq_getElem hi, beq_iff_eq.1 hk]
  | none =>
    have hr : (op k).run s = (s.size, s.push k) := by simp [Flock.RecSparse.op, h]; try rfl
    rw [hr]
    exact ⟨by simp [Ext], fun F hF => getElem?_of_ext hF (by simp)⟩

/-- Pair `ij` indexes `ab`'s keys in `F`. -/
def Ix (F : Array Key) (ij : Nat × Nat) (ab : Key × Key) : Prop := F[ij.1]? = some ab.1 ∧ F[ij.2]? = some ab.2

theorem put_spec (mul : List (Key × Key)) (add : List Key) :
    Spec (put mul add) fun r F => r.add = add.map Flock.RecSparse.term ∧ List.Forall₂ (fun ab ij => Ix F ij ab) mul r.mul := by
  unfold Flock.RecSparse.put
  refine Spec.bind' (Spec.mapM _ (fun ab ij F => Ix F ij ab) mul fun ab _ => ?_) fun m => Spec.pure' fun F hm => ⟨rfl, hm⟩
  obtain ⟨a, b⟩ := ab
  exact Spec.bind' (op_spec a) fun i => Spec.bind' (op_spec b) fun j => Spec.pure' fun F hj hi => ⟨hi, hj⟩

/-- A key as `Defs`' term: `(0, i)` is `w i`, any other `(_, i)` is `v i`. -/
def kTerm (k : Key) : Flock.RecAlgebra.Term := if k.1 = 0 then .w k.2 else .v k.2

/-- `put mul add`'s products and added terms as `Defs`' terms. -/
def kSpec (mul : List (Key × Key)) (add : List Key) :
    List (Flock.RecAlgebra.Term × Flock.RecAlgebra.Term) × List Flock.RecAlgebra.Term :=
  (mul.map fun ab => (kTerm ab.1, kTerm ab.2), add.map kTerm)

/-- Residual `r` is `put`'s on keys whose terms are `s`, its pairs indexing `F`. -/
def Den (F : Array Key) (r : Flock.RecOpen.Res)
    (s : List (Flock.RecAlgebra.Term × Flock.RecAlgebra.Term) × List Flock.RecAlgebra.Term) : Prop :=
  ∃ mul add, s = kSpec mul add ∧ r.add = add.map Flock.RecSparse.term ∧
    List.Forall₂ (fun ab ij => Ix F ij ab) mul r.mul

theorem put_den {mul : List (Key × Key)} {add : List Key} {s} (hs : s = kSpec mul add) :
    Spec (put mul add) fun r F => Den F r s :=
  Spec.mono (put_spec mul add) fun _ _ h => ⟨mul, add, hs, h⟩

section Builder

variable (sh : Flock.RecSparse.Shape)

/-- Residual `p` is named `str nm` and is `put`'s on `spec nm`'s terms. -/
def Named (F : Array Key) (nm : RName) (p : String × Flock.RecOpen.Res) : Prop :=
  p.1 = str nm ∧ Den F p.2 (spec (shOf sh) nm)

theorem names_rel {F : Array Key} {h1 wv f0 bp : Flock.RecOpen.Res} {pre sc cl : List (String × Flock.RecOpen.Res)}
    {fs : List (List (String × Flock.RecOpen.Res))} (hh1 : Den F h1 (spec (shOf sh) .h1))
    (hpre : List.Forall₂ (fun j => Named sh F (.pre j)) (List.range 6) pre) (hwv : Den F wv (spec (shOf sh) .wv))
    (hsc : List.Forall₂ (fun j => Named sh F (.sc j)) (List.range sh.degrees.length) sc)
    (hf0 : Den F f0 (spec (shOf sh) .f0))
    (hfs : List.Forall₂ (fun j => List.Forall₂ (Named sh F) [.fx j, .fy j]) (List.range sh.ell) fs)
    (hbp : Den F bp (spec (shOf sh) .bp)) (hcl : List.Forall₂ (Named sh F) (if sh.tLog = 0 then [] else [.claim]) cl) :
    List.Forall₂ (Named sh F) (names (shOf sh))
      ([("h1", h1)] ++ pre ++ [("wv", wv)] ++ sc ++ [("f0", f0)] ++ fs.flatten ++ [("bp", bp)] ++ cl) := by
  rw [degs_length] at hsc
  exact List.rel_append (List.rel_append (List.rel_append (List.rel_append (List.rel_append (List.rel_append
    (List.rel_append (.cons ⟨rfl, hh1⟩ .nil) (List.forall₂_map_left_iff.2 hpre)) (.cons ⟨rfl, hwv⟩ .nil))
    (List.forall₂_map_left_iff.2 hsc)) (.cons ⟨rfl, hf0⟩ .nil)) (forall₂_flatten (List.forall₂_map_left_iff.2 hfs)))
    (.cons ⟨rfl, hbp⟩ .nil)) hcl

/-- **`resList` is `Defs`' structure**: its residuals are named `str` of `names`, each one `put`'s on `spec`'s terms. -/
theorem resList_spec : Spec (resList sh) fun rs F => List.Forall₂ (Named sh F) (names (shOf sh)) rs := by
  obtain ⟨oh, opre, ov2, osc, ou, ovt, ocl⟩ := mp_offsets sh
  obtain ⟨veqc, vpre, vwhat, vsc, valpha, vax, vay, veqp, veqt⟩ := vp_all sh
  have hl := degs_length sh
  unfold Flock.RecSparse.resList
  dsimp only
  refine Spec.bind' (put_den (s := spec (shOf sh) .h1) ?_) fun h1 => ?_
  · simp [spec, kSpec, kTerm, oh, veqc, Flock.RecSparse.SKIP, List.map_map, Function.comp_def]
  refine Spec.bind' (Spec.mapM _ (fun j p F => Named sh F (.pre j) p) _ fun j hj =>
    Spec.bind' (put_den (s := spec (shOf sh) (.pre j)) ?_) fun r => Spec.pure' fun _ h => ⟨rfl, h⟩) fun pre => ?_
  · have hj : j < 6 := by simpa [Flock.RecSparse.PRE_ROUNDS] using hj
    simp only [opre j hj, vpre]
    simp [spec, kSpec, kTerm, wPre, vPre]
    omega
  refine Spec.bind' (put_den (s := spec (shOf sh) .wv) ?_) fun wv => ?_
  · simp [spec, kSpec, kTerm, ov2, vwhat, wV2]
  refine Spec.bind' (Spec.mapM _ (fun j p F => Named sh F (.sc j) p) _ fun j hj =>
    Spec.bind' (put_den (s := spec (shOf sh) (.sc j)) ?_) fun r => Spec.pure' fun _ h => ⟨rfl, h⟩) fun sc => ?_
  · have hj : j < nR (shOf sh) := by rw [← hl]; simpa using hj
    simp only [osc j hj, vsc, degs_getD sh hj, degs_take sh j (Nat.le_of_lt hj)]
    simp only [spec, kSpec, List.range'_eq_map_range, List.map_map, List.map_cons, List.map_nil, Prod.mk.injEq]
    refine ⟨List.map_congr_left fun i _ => ?_, ?_⟩ <;> simp [kTerm, wSc, vSc]
    omega
  refine Spec.bind' (put_den (s := spec (shOf sh) .f0) ?_) fun f0 => ?_
  · simp [spec, kSpec, kTerm, valpha, ou]
  refine Spec.bind' (Spec.mapM _ (fun j l F => List.Forall₂ (Named sh F) [.fx j, .fy j] l) _ fun j _ =>
    Spec.bind' (put_den (s := spec (shOf sh) (.fx j)) ?_) fun fx =>
      Spec.bind' (put_den (s := spec (shOf sh) (.fy j)) ?_) fun fy =>
        Spec.pure' fun _ hy hx => .cons ⟨rfl, hx⟩ (.cons ⟨rfl, hy⟩ .nil)) fun fs => ?_
  · simp [spec, kSpec, kTerm, vax, ou]
  · simp [spec, kSpec, kTerm, vay, ou]
  refine Spec.bind' (put_den (s := spec (shOf sh) .bp) ?_) fun bp => ?_
  · cases hst : sh.stacked
    · have ht : sh.tLog = 0 := tLog_eq sh hst
      simp [spec, kSpec, kTerm, veqp, ou, ocl, ht, Flock.RecSparse.Shape.PP, pLog_shOf, List.map_map,
        Function.comp_def]
    · have ht : sh.tLog ≠ 0 := tLog_ne sh hst
      simp [spec, kSpec, kTerm, veqp, ou, ht, Flock.RecSparse.Shape.PP, pLog_shOf, List.map_map,
        Function.comp_def]
  split
  · rename_i hst
    have ht : sh.tLog ≠ 0 := tLog_ne sh hst
    refine Spec.bind' (put_den (s := spec (shOf sh) .claim) ?_) fun r => ?_
    · simp [spec, kSpec, kTerm, veqt hst, ovt hst, ocl, Flock.RecSparse.Shape.T, List.map_map, Function.comp_def]
    simp only [pure_bind]
    exact Spec.pure' fun F hr hbp hfs hf0 hsc hwv hpre hh1 =>
      names_rel sh hh1 hpre hwv hsc hf0 hfs hbp (by simp only [ht, ↓reduceIte]; exact .cons ⟨rfl, hr⟩ .nil)
  · rename_i hst
    have ht : sh.tLog = 0 := tLog_eq sh (by simpa using hst)
    simp only [pure_bind]
    exact Spec.pure' fun F hbp hfs hf0 hsc hwv hpre hh1 =>
      names_rel sh hh1 hpre hwv hsc hf0 hfs hbp (by simp only [ht, ↓reduceIte]; exact .nil)

theorem structureOf_spec : ∃ (F : Array Key) (rs : List (String × Flock.RecOpen.Res)),
    structureOf sh = (⟨F.toList.map fun k => [Flock.RecSparse.term k], rs.map (·.2)⟩, rs.map (·.1)) ∧
    List.Forall₂ (Named sh F) (names (shOf sh)) rs := by
  have h := (resList_spec sh #[]).2 _ (List.prefix_refl _)
  unfold Flock.RecSparse.structureOf
  generalize (resList sh).run #[] = p at h ⊢
  obtain ⟨rs, F⟩ := p
  exact ⟨F, rs, rfl, h⟩

/-- The names `steps` reads are the builder's. -/
theorem used_mem {nm : String} (hu : Used sh nm) : nm ∈ (structureOf sh).2 := by
  obtain ⟨F, rs, hs, h⟩ := structureOf_spec sh
  rw [hs]
  show nm ∈ rs.map (·.1)
  rw [← map_eq_of_forall₂ (h.imp fun _ _ hp => hp.1.symm)]
  refine List.mem_map.2 ?_
  rcases hu with rfl | rfl | rfl | rfl | ⟨hst, rfl⟩ | ⟨j, hj, rfl⟩ | ⟨j, hj, rfl⟩ | ⟨j, hj, rfl | rfl⟩
  · exact ⟨.h1, mem_h1 _, rfl⟩
  · exact ⟨.wv, mem_wv _, rfl⟩
  · exact ⟨.f0, mem_f0 _, rfl⟩
  · exact ⟨.bp, mem_bp _, rfl⟩
  · exact ⟨.claim, mem_claim _ (tLog_ne sh hst), rfl⟩
  · exact ⟨.pre j, mem_pre _ hj, rfl⟩
  · exact ⟨.sc j, mem_sc _ (by rwa [← degs_length]), rfl⟩
  · exact ⟨.fx j, mem_fx _ hj, rfl⟩
  · exact ⟨.fy j, mem_fy _ hj, rfl⟩

/-! ## Residual values -/

theorem termVal_term (wA vA : Array F128) (k : Key) :
    termVal (wA[·]!) (vA[·]!) (termOf (Flock.RecSparse.term k)) = Flock.RecAlgebra.termValue wA vA (kTerm k) := by
  unfold Flock.RecSparse.term kTerm
  by_cases h : k.1 = 0 <;> simp only [h, ↓reduceIte] <;> rfl

/-- A residual `put` builds, over operands that are its keys' terms, is `specVal` of its terms. -/
theorem fOf_resVal (F : Array Key) (res : List Flock.RecOpen.Res) (wA vA : Array F128) (r : Flock.RecOpen.Res)
    {mul : List (Key × Key)} {add : List Key} (hadd : r.add = add.map Flock.RecSparse.term)
    (hmul : List.Forall₂ (fun ab ij => Ix F ij ab) mul r.mul) :
    fOf (resVal (gfsOf ⟨F.toList.map fun k => [Flock.RecSparse.term k], res⟩) (termVal (wA[·]!) (vA[·]!))
      (FlockVBridge.resOf r)) = specVal wA vA (kSpec mul add) := by
  have hop : ∀ i k, F[i]? = some k → fOf (opVal (gfsOf ⟨F.toList.map fun k => [Flock.RecSparse.term k], res⟩)
      (termVal (wA[·]!) (vA[·]!)) i) = fOf (Flock.RecAlgebra.termValue wA vA (kTerm k)) := by
    intro i k hk
    have hg : (gfsOf ⟨F.toList.map fun k => [Flock.RecSparse.term k], res⟩).ops.getD i [] =
        [termOf (Flock.RecSparse.term k)] := by
      simp only [gfsOf, List.map_map, List.getD_eq_getElem?_getD, List.getElem?_map, Array.getElem?_toList, hk,
        Option.map_some, Option.getD_some, Function.comp_apply, List.map_cons, List.map_nil]
    rw [opVal, hg, List.map_cons, List.map_nil, List.foldl_cons, List.foldl_nil, fOf_add, fOf_zero, zero_add,
      termVal_term]
  simp only [resVal, FlockVBridge.resOf, hadd, List.map_map, List.foldl_map]
  rw [fOf_foldl_sum, fOf_foldl_sum, fOf_zero, zero_add]
  simp only [specVal, kSpec, List.map_map]
  congr 1
  · refine congrArg List.sum (List.map_congr_left fun k _ => ?_)
    simp only [Function.comp_apply, termVal_term]
  · symm
    refine congrArg List.sum (map_eq_of_forall₂ (hmul.imp fun ab ij hx => ?_))
    simp only [Function.comp_apply, fOf_mul, hop _ _ hx.1, hop _ _ hx.2]

/-- Residual `r`'s value at `w`, `v` is `spec r` evaluated. -/
theorem fOf_rval (wA vA : Array F128) {r : RName} (hr : r ∈ names (shOf sh)) :
    fOf (rval sh (wA[·]!) (vA[·]!) (str r)) = specVal wA vA (spec (shOf sh) r) := by
  obtain ⟨F, rs, hs, h⟩ := structureOf_spec sh
  have hl : (names (shOf sh)).idxOf r < (names (shOf sh)).length := List.idxOf_lt_length_of_mem hr
  have hrs : (names (shOf sh)).idxOf r < rs.length := h.length_eq ▸ hl
  have hN := forall₂_getD (d := RName.h1) (e := (("" : String), (⟨[], []⟩ : Flock.RecOpen.Res))) h hrs
  rw [List.getD_eq_getElem _ _ hl, List.getElem_idxOf hl, List.getD_eq_getElem _ _ hrs] at hN
  obtain ⟨-, mul, add, hsp, hadd, hmul⟩ := hN
  have hidx : (structureOf sh).2.idxOf (str r) = (names (shOf sh)).idxOf r := by
    rw [hs]
    dsimp only
    rw [← map_eq_of_forall₂ (h.imp fun _ _ hp => hp.1.symm), idxOf_map_inj str (fun _ _ => str_inj) r]
  have hres : (gfsOf (structureOf sh).1).res.getD ((names (shOf sh)).idxOf r) ⟨[], []⟩ =
      FlockVBridge.resOf rs[(names (shOf sh)).idxOf r].2 := by
    rw [hs]
    dsimp only [gfsOf]
    rw [List.getD_eq_getElem _ _ (by simpa using hrs), List.getElem_map, List.getElem_map]
  unfold rval
  rw [hidx, hres, hs, hsp]
  exact fOf_resVal F (rs.map (·.2)) wA vA _ hadd hmul

/-- **The builder's residuals are `Defs`'**: residual `r`'s value at `w`, `v` is `rget` of `resArr`. -/
theorem rval_rget (wA vA : Array F128) {r : RName} (hr : r ∈ names (shOf sh)) :
    rval sh (wA[·]!) (vA[·]!) (str r) = rget (shOf sh) (resArr (shOf sh) wA vA) r :=
  fOf_injective ((fOf_rval sh wA vA hr).trans (rget_eq _ wA vA hr).symm)

end Builder

end FlockVBridge.Sparse
