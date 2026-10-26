import Proofs.Flock.Claims.Recursion.RestElemR
import Proofs.Flock.Claims.Recursion.SlotsPorted
import Proofs.Flock.Claims.Recursion.OpenSlots
import Proofs.Flock.Claims.Recursion.OpensReads
import Proofs.Flock.Claims.Recursion.RestLens
import Proofs.Flock.Claims.Recursion.NextRFits
import Proofs.Flock.Claims.Recursion.NextRSetup
import Proofs.Flock.Claims.Recursion.OpenLowerDeployed
import Proofs.Flock.Claims.Recursion.RestFitsLift
import Proofs.Flock.Recursive.HPortTie
import Proofs.Flock.C3.Reps.RoundLinks
import Proofs.Flock.VBridge.Family.Structure
import Proofs.Flock.SessionLoop.FinPoints
import Proofs.Flock.C3.VStarList.Shape
import Proofs.Flock.Claims.DeployK
import VStar.Stage

/-!
# `OpenedSized` at the deployed last message `elemR`

`readsAt_elemR'` takes `OpenedSized elemR`: every rep's block of the opened rounds after the link has the byte sizes
of an honest rep's rows. This file proves it, so the last step reads at the deployed readers with no hypothesis
(`readsAt_elemR_closed`), at every witness and every decision `rdR` admits.

* **An honest rep's row sizes** (`FlockDeploy.Cnt`): the honest proof `openP` (`sizedProof` at the rep's shape, with
  the context's cap) is an `HonestRep`, and round `t` of its rows is its tops' 128-byte node rows, then, where the
  round absorbs, one padded row of the values the round's chunk of `repFrames` absorbs. That count is
  `schedV`'s `t`-th entry (`cntPat_repFrames`), a row of `2 · portWordsOf` of it bytes (`open_sized`).
* **V\*'s rounds** (`VStarSizes`): the inner statement's `m{i}` ports carry `rounds`' sizes, which are `schedV`'s
  nonzero entries in order (`rounds_sizes`, `schedV_filter`), and round `t`'s is its `commitRounds` index's
  (`Cnt.rounds_at`).
* **The opened rows** (`OpenSlots`): `elemR` opens, at each round, its committed tops' cap rows (128 bytes each,
  `capSlot_len`) and, where the round absorbs, the row its `m{i}` port reads (`rowSlot_len`), so its sizes are an
  honest rep's (`roundSlots_sizes`, `openedSized_of`). Nothing reads a witness's values, only `elemR`'s layout.
-/

namespace VStarSizes

open VStar.Algebra

theorem msgNames_size (n : Nat) : (msgNames n).size = 4 := rfl

theorem oodMsg_size (o n : Nat) : (#[Name.ood o] ++ msgNames n).size = 5 := rfl

theorem fp_sizes (g : Nat → Array Name) : ∀ (l : List Nat) (init : Array (Array Name)),
    (l.foldl (fun b a => b.push (g a)) init).toList.map Array.size =
      init.toList.map Array.size ++ l.map fun a => (g a).size
  | [], init => by simp
  | a :: l, init => by
    rw [List.foldl_cons, fp_sizes g l]
    simp

theorem f1_fst : ∀ (l : List Nat) (init : Array (Array Name) × Nat),
    (l.foldl (fun b _ => (b.fst.push (msgNames b.snd), b.snd + 1)) init).fst.toList.map Array.size =
      init.fst.toList.map Array.size ++ List.replicate l.length 4
  | [], init => by simp
  | a :: l, init => by
    rw [List.foldl_cons, f1_fst l]
    simp [msgNames_size, List.replicate_succ]

theorem f1_snd : ∀ (l : List Nat) (init : Array (Array Name) × Nat),
    (l.foldl (fun b _ => (b.fst.push (msgNames b.snd), b.snd + 1)) init).snd = init.snd + l.length
  | [], init => by simp
  | a :: l, init => by
    rw [List.foldl_cons, f1_snd l]
    simp
    omega

theorem f2_fst : ∀ (l : List Nat) (init : Array (Array Name) × Nat × Nat),
    (l.foldl (fun b _ => (b.fst.push (#[Name.ood b.snd.snd] ++ msgNames b.snd.fst), b.snd.fst + 1, b.snd.snd + 1))
      init).fst.toList.map Array.size = init.fst.toList.map Array.size ++ List.replicate l.length 5
  | [], init => by simp
  | a :: l, init => by
    rw [List.foldl_cons, f2_fst l]
    simp [msgNames_size, List.replicate_succ]

theorem forIn_sizes (body : Nat → Array (Array Name) × Nat × Nat → Id (ForInStep (Array (Array Name) × Nat × Nat)))
    (g : Nat → List Nat)
    (hb : ∀ i s, ∃ s', body i s = pure (.yield s') ∧
      s'.1.toList.map Array.size = s.1.toList.map Array.size ++ g i) :
    ∀ (l : List Nat) (init : Array (Array Name) × Nat × Nat),
      (Prod.fst <$> forIn l init body : Id (Array (Array Name))).toList.map Array.size =
        init.1.toList.map Array.size ++ l.flatMap g
  | [], init => by simp; rfl
  | a :: l, init => by
    obtain ⟨s', h1, h2⟩ := hb a init
    rw [List.forIn_cons, h1]
    simp only [pure_bind]
    rw [forIn_sizes body g hb l s', h2, List.flatMap_cons, List.append_assoc]

/-- The sizes of `rounds`' rounds. -/
def sizesV (sh : Shape) (s : Flock.Schedule) : List Nat :=
  [128] ++ List.replicate (sh.m - 6) 2 ++ [2] ++ List.replicate (sh.kLog - 6) 2 ++ [64] ++
    List.replicate sh.claims 128 ++ [1, 1] ++ List.replicate (s.k0 + 1) 4 ++ [5, 5, 4] ++
    (List.range' 0 (s.levels.size - 1)).flatMap fun i =>
      if i + 1 < s.levels.size - 1 then [4, 4, 4, 4, 5, 5, 4] else [4, 4, 4, 4 + (2 <<< s.ext)]

/-- **`rounds`' sizes are `sizesV`'s.** -/
theorem rounds_sizes (sh : Shape) (s : Flock.Schedule) : (rounds sh s).toList.map Array.size = sizesV sh s := by
  unfold rounds
  simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, List.forIn_pure_yield_eq_foldl,
    bind_pure_comp, map_pure, pure_bind, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  rw [forIn_sizes (g := fun i => if i + 1 < s.levels.size - 1 then [4, 4, 4, 4, 5, 5, 4] else [4, 4, 4, 4 + (2 <<< s.ext)])]
  · simp only [Array.toList_push, List.map_append, f2_fst, f1_fst, fp_sizes]
    simp [sizesV, List.map_const', List.length_range', msgNames_size]
  · intro i st
    split
    · refine ⟨_, rfl, ?_⟩
      simp only [Array.toList_push, List.map_append, f2_fst, f1_fst]
      simp [msgNames_size, *]
    · refine ⟨_, rfl, ?_⟩
      simp only [Array.toList_push, List.map_append, f1_fst]
      simp [msgNames_size, *]

theorem ebind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases hx : x with
  | error e => rw [hx] at h; cases h
  | ok a => rw [hx] at h; exact ⟨a, rfl, h⟩

theorem foldl_push_map {α β : Type} (f : β → α) : ∀ (l : List β) (init : Array α),
    l.foldl (fun out b => out.push (f b)) init = init ++ (l.map f).toArray
  | [], init => by simp
  | b :: l, init => by
    rw [List.foldl_cons, foldl_push_map f l]
    rw [← Array.toList_inj]
    simp

/-- **The algebra's ports**: `m{i}` of round `i`'s size, then the two accumulators. -/
theorem spec_ports {sh : VStar.Algebra.Shape} {sp : VStar.RepS} (h : VStar.Algebra.spec sh = .ok sp) :
    ∃ s, sh.sched = .ok s ∧ sp.ports = (List.range (VStar.Algebra.rounds sh s).size).map
      (fun i => (s!"m{i}", (VStar.Algebra.rounds sh s)[i]!.size)) ++ [("acc0", 2), ("acc1", 2)] := by
  unfold VStar.Algebra.spec at h
  obtain ⟨cx, hcx, h⟩ := ebind_ok h
  obtain ⟨syms, -, h⟩ := ebind_ok h
  obtain ⟨gf, -, h⟩ := ebind_ok h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, List.forIn_pure_yield_eq_foldl,
    Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one, bind_pure_comp, map_pure, pure_bind] at h
  obtain ⟨⟨⟩, -, h⟩ := ebind_ok h
  cases h
  unfold VStar.Algebra.Ctx.make at hcx
  obtain ⟨s, hs, hcx⟩ := ebind_ok hcx
  obtain ⟨idx, -, hcx⟩ := ebind_ok hcx
  simp only [pure, Except.pure, Except.ok.injEq] at hcx
  subst hcx
  refine ⟨s, hs, ?_⟩
  simp only [foldl_push_map, List.range_eq_range']
  simp

theorem mem_map_getElem! {α : Type} [Inhabited α] (a : Array α) (l : List Nat) :
    ∀ x ∈ l.map (a[·]!), x = default ∨ x ∈ a.toList := by
  intro x hx
  obtain ⟨i, -, rfl⟩ := List.mem_map.1 hx
  by_cases hi : i < a.size
  · exact .inr (by rw [getElem!_pos a i hi]; exact Array.getElem_mem_toList hi)
  · exact .inl (getElem!_neg a i hi)

theorem forIn_pushes_all {ε α β : Type} (P : β → Prop) (f : α → Array β → Except ε (ForInStep (Array β)))
    (hf : ∀ a r y, f a r = .ok y → ∃ x, y = .yield (r.push x) ∧ P x) :
    ∀ (L : List α) (init v : Array β), (∀ x ∈ init, P x) → forIn L init f = .ok v → ∀ x ∈ v, P x
  | [], init, v, hi, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact hi
  | a :: L, init, v, hi, h => by
    rw [List.forIn_cons] at h
    obtain ⟨y, hy, h⟩ := ebind_ok h
    obtain ⟨x, rfl, hx⟩ := hf a init y hy
    refine forIn_pushes_all P f hf L _ v (fun z hz => ?_) h
    rcases Array.mem_push.1 hz with hz | rfl
    · exact hi z hz
    · exact hx

/-- **Each part's ports are its whole S's** (or the default, past its ports). -/
theorem parts_ports {S : VStar.RepS} {ps : Array VStar.Part} (h : VStar.parts S = .ok ps) :
    ∀ p ∈ ps, ∀ x ∈ p.S.ports, x = default ∨ x ∈ S.ports := by
  unfold VStar.parts at h
  dsimp only at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨G, -, h⟩ := ebind_ok h
  obtain ⟨out, hout, h⟩ := ebind_ok h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  rw [← Array.forIn_toList] at hout
  refine forIn_pushes_all (fun p : VStar.Part => ∀ x ∈ p.S.ports, x = default ∨ x ∈ S.ports) _
    (fun a r y hy => ?_) _ _ _ (by simp) hout
  repeat' (first | (obtain ⟨_, _, hy⟩ := ebind_ok hy) | (split at hy))
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    refine ⟨_, hy.symm, ?_⟩
    intro x hx
    rcases mem_map_getElem! _ _ x hx with h | h
    · exact .inl h
    · exact .inr (by simpa using h)

/-- **An inner unit's input ports**: one per port of its S, of the port's words, then the folded value's. -/
theorem innerUnit_inPorts {S : VStar.RepS} {name : String} {u : VStar.UnitC} (h : VStar.innerUnit S name = .ok u) :
    u.inPorts = S.ports.map (fun p => VStar.portWordsOf p.2) ++ [VStar.portWordsOf S.gf.nv] := by
  unfold VStar.innerUnit at h
  obtain ⟨⟨⟩, -, h⟩ := ebind_ok h
  obtain ⟨⟨⟩, -, h⟩ := ebind_ok h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  rfl

end VStarSizes

namespace FlockDeploy.Cnt

open Flock
open FlockSoundness.Refine (Frame colBytes)
open FlockSoundness.Discharge.DecodeZL (ProofSizes)
open FlockSoundness.Discharge.Exec (chunks closes glue chunks_append repFrames zcFrames lcFrames rsFrames ligFrames
  levelFr introFr queryFr chFr kFr capFr)

/-! ## Which round holds which value count -/

theorem filter_idx : ∀ (L : List ℕ) (k t : ℕ),
    ((List.range L.length).filter (fun i => L.getD i 0 != 0))[k]? = some t →
      (L.filter (· != 0))[k]? = some (L.getD t 0)
  | [], k, t, h => by simp at h
  | a :: L, k, t, h => by
    rw [List.length_cons, List.range_succ_eq_map, List.filter_cons, List.filter_map] at h
    have e : (List.filter ((fun i => (a :: L).getD i 0 != 0) ∘ Nat.succ) (List.range L.length)) =
        (List.range L.length).filter (fun i => L.getD i 0 != 0) := by
      congr 1
    rw [e] at h
    rw [List.filter_cons]
    by_cases ha : (a != 0) = true
    · simp only [List.getD_cons_zero, ha, ↓reduceIte] at h ⊢
      cases k with
      | zero =>
        simp at h
        subst h
        simp
      | succ k =>
        simp only [List.getElem?_cons_succ, List.getElem?_map] at h ⊢
        obtain ⟨t', ht', rfl⟩ := Option.map_eq_some_iff.1 h
        rw [filter_idx L k t' ht']
        simp
    · simp only [List.getD_cons_zero, ha, Bool.false_eq_true, ↓reduceIte] at h ⊢
      rw [List.getElem?_map] at h
      obtain ⟨t', ht', rfl⟩ := Option.map_eq_some_iff.1 h
      rw [filter_idx L k t' ht']
      simp

/-! ## Value counts -/

/-- A frame's value count. -/
def cv : Frame → ℕ
  | .one _ => 1
  | .vec vs => vs.length
  | .dom _ => 0
  | .label _ => 0
  | .bytes _ => 0
  | .sq _ _ => 0

/-- Frames' value count. -/
def nV (fs : List Frame) : ℕ := (vals fs).length

theorem nV_cons (f : Frame) (fs : List Frame) : nV (f :: fs) = cv f + nV fs := by
  cases f <;> simp [nV, vals, cv] <;> omega

theorem nV_append (a b : List Frame) : nV (a ++ b) = nV a + nV b := by
  simp [nV, vals_append]

/-- Rounds' value counts, and the pending frames'. -/
def cntOf (x : List (List Frame) × List Frame) : List ℕ × ℕ := (x.1.map nV, nV x.2)

/-- `glue` at value counts. -/
def glueN (a b : List ℕ × ℕ) : List ℕ × ℕ :=
  match b.1 with
  | [] => (a.1, a.2 + b.2)
  | c :: cs => (a.1 ++ (a.2 + c) :: cs, b.2)

/-- **Frames' rounds' value counts**, and the pending frames'. -/
def cntPat (F : List Frame) : List ℕ × ℕ := cntOf (chunks F)

theorem cntOf_glue (a b : List (List Frame) × List Frame) : cntOf (glue a b) = glueN (cntOf a) (cntOf b) := by
  rcases b with ⟨_ | ⟨c, cs⟩, t⟩ <;> simp [glue, cntOf, glueN, nV_append]

theorem cntPat_append (F G : List Frame) : cntPat (F ++ G) = glueN (cntPat F) (cntPat G) := by
  unfold cntPat
  rw [chunks_append, cntOf_glue]

theorem cntPat_nil : cntPat [] = ([], 0) := rfl

theorem glueN_closed (la lb : List ℕ) (pb : ℕ) : glueN (la, 0) (lb, pb) = (la ++ lb, pb) := by
  cases lb <;> simp [glueN]

theorem glueN_cons (a : List ℕ) (x c : ℕ) (cs : List ℕ) (y : ℕ) :
    glueN (a, x) (c :: cs, y) = (a ++ (x + c) :: cs, y) := rfl

theorem glueN_nil (a : List ℕ) (x y : ℕ) : glueN (a, x) ([], y) = (a, x + y) := rfl

theorem glueN_zero_nil (x : List ℕ × ℕ) : glueN ([], 0) x = x := by
  rcases x with ⟨_ | _, _⟩ <;> simp [glueN]

theorem cntPat_open {f : Frame} (hf : closes f = false) (F : List Frame) :
    cntPat (f :: F) = glueN ([], cv f) (cntPat F) := by
  unfold cntPat
  simp only [chunks, hf, Bool.false_eq_true, ↓reduceIte]
  generalize chunks F = x
  rcases x with ⟨_ | ⟨c, cs⟩, t⟩ <;> simp [cntOf, glueN, nV_cons]

theorem cntPat_sq (k : UInt8) {n : ℕ} (hn : n ≠ 0) (F : List Frame) :
    cntPat (.sq k n :: F) = (0 :: (cntPat F).1, (cntPat F).2) := by
  have hc : closes (.sq k n) = true := by simp [closes, hn]
  unfold cntPat
  simp only [chunks, hc, ↓reduceIte]
  rfl

theorem cntPat_dom (b : ByteArray) (F : List Frame) : cntPat (.dom b :: F) = cntPat F :=
  (cntPat_open (f := .dom b) rfl F).trans (glueN_zero_nil _)

theorem cntPat_label (b : ByteArray) (F : List Frame) : cntPat (.label b :: F) = cntPat F :=
  (cntPat_open (f := .label b) rfl F).trans (glueN_zero_nil _)

theorem cntPat_bytes (b : ByteArray) (F : List Frame) : cntPat (.bytes b :: F) = cntPat F :=
  (cntPat_open (f := .bytes b) rfl F).trans (glueN_zero_nil _)

theorem cntPat_capFr (cap : Array ByteArray) (F : List Frame) : cntPat (capFr cap :: F) = cntPat F :=
  cntPat_bytes _ F

theorem cntPat_one (v : F128) (F : List Frame) : cntPat (.one v :: F) = glueN ([], 1) (cntPat F) :=
  cntPat_open rfl F

theorem cntPat_vec (vs : List F128) (F : List Frame) :
    cntPat (.vec vs :: F) = glueN ([], vs.length) (cntPat F) :=
  cntPat_open rfl F

theorem cntPat_kFr (m : F256 × F256) : cntPat (kFr m) = ([], 4) := rfl

theorem cntPat_ones : ∀ l : List F128, cntPat (l.map Frame.one) = ([], l.length)
  | [] => rfl
  | v :: l => by
    rw [List.map_cons, cntPat_one, cntPat_ones l, glueN_nil, List.length_cons, Nat.add_comm]

theorem cntPat_blocks {F : ℕ → List Frame} {c : ℕ} (hF : ∀ i, cntPat (F i) = ([c], 0)) :
    ∀ a k, cntPat ((List.range' a k).flatMap F) = (List.replicate k c, 0)
  | _, 0 => rfl
  | a, k + 1 => by
    rw [List.range'_succ, List.flatMap_cons, cntPat_append, hF, cntPat_blocks hF (a + 1) k, glueN_closed]
    rfl

theorem cntPat_flatMap {F : ℕ → List Frame} {G : ℕ → List ℕ} :
    ∀ a k, (∀ i, a ≤ i → i < a + k → cntPat (F i) = (G i, 0)) →
      cntPat ((List.range' a k).flatMap F) = ((List.range' a k).flatMap G, 0)
  | _, 0, _ => rfl
  | a, k + 1, h => by
    rw [List.range'_succ, List.flatMap_cons, List.flatMap_cons, cntPat_append, h a (Nat.le_refl a) (by omega),
      cntPat_flatMap (a + 1) k fun i h1 h2 => h i (by omega) (by omega), glueN_closed]

theorem cntPat_chFr (p : LigeritoProof) :
    ∀ k msg, cntPat (chFr p msg (k + 1)) = (0 :: List.replicate k 4, 4)
  | 0, msg => by
    simp only [chFr, List.append_nil, cntPat_sq 2 (by decide : (2 : ℕ) ≠ 0), cntPat_kFr, List.replicate_zero]
  | k + 1, msg => by
    rw [chFr, cntPat_append, cntPat_chFr p k (msg + 1)]
    simp only [cntPat_sq 2 (by decide : (2 : ℕ) ≠ 0), cntPat_kFr, glueN_cons, Nat.add_zero, List.nil_append,
      List.cons_append, List.replicate_succ]

theorem cntPat_introFr (p : LigeritoProof) {dim : ℕ} (hd : dim ≠ 0) (ood msg cbn : ℕ) :
    cntPat (introFr p dim ood msg cbn) = ([0, 5], 0) := by
  unfold introFr
  simp only [cntPat_append, cntPat_sq _ hd, cntPat_one, cntPat_kFr, cntPat_bytes, cntPat_nil,
    cntPat_sq _ Nat.one_ne_zero, glueN_cons, glueN_nil, List.nil_append, List.cons_append]

theorem cntPat_queryFr (p : LigeritoProof) {lv : Flock.Level} (hq : lv.q ≠ 0) (qn cn : ℕ) :
    cntPat (queryFr p lv qn cn) = ([0, 0], 0) := by
  unfold queryFr
  simp only [cntPat_bytes, cntPat_sq _ hq, cntPat_sq _ (Nat.add_one_ne_zero _), cntPat_nil]

theorem cntPat_zc {m : ℕ} (hm : 14 ≤ m) (digest cap : ByteArray) (p : ZerocheckProof) :
    cntPat (zcFrames m digest cap p) =
      (0 :: 0 :: (p.round1ab.size + p.round1c.size) :: List.replicate (m - 6) 2, 2) := by
  have hB := cntPat_blocks (F := fun i => [Frame.one (p.rounds[i]!).1, .one (p.rounds[i]!).2, .sq 1 1]) (c := 2)
    (fun i => by simp only [cntPat_one, cntPat_sq _ Nat.one_ne_zero, cntPat_nil, glueN_cons, glueN_nil,
      List.nil_append]) 0 (m - 6)
  have h13 : m - 13 ≠ 0 := by omega
  unfold zcFrames
  simp only [cntPat_append, hB, cntPat_one, cntPat_nil, cntPat_label, cntPat_bytes,
    cntPat_sq _ (by decide : (6 : ℕ) ≠ 0), cntPat_sq _ h13, cntPat_vec, cntPat_sq _ Nat.one_ne_zero,
    glueN_nil, glueN_cons, glueN_closed, Nat.add_zero, Nat.zero_add, List.nil_append, List.cons_append,
    Array.length_toList, List.append_nil]

theorem cntPat_lc (k : ℕ) (p : LincheckProof) :
    cntPat (lcFrames k p) = (0 :: 0 :: (List.replicate (k - 6) 2 ++ [p.zPartial.size]), 0) := by
  have hB := cntPat_blocks (F := fun j => [Frame.one (p.rounds[j]!).1, .one (p.rounds[j]!).2, .sq 1 1]) (c := 2)
    (fun i => by simp only [cntPat_one, cntPat_sq _ Nat.one_ne_zero, cntPat_nil, glueN_cons, glueN_nil,
      List.nil_append]) 0 (k - 6)
  unfold lcFrames
  simp only [cntPat_append, hB, cntPat_vec, cntPat_nil, cntPat_label, cntPat_sq _ Nat.one_ne_zero,
    glueN_cons, glueN_nil, glueN_closed, Nat.add_zero, Nat.zero_add, List.nil_append, List.cons_append,
    Array.length_toList]

theorem flatMap_range'_rep (g : ℕ → ℕ) (c : ℕ) :
    ∀ a k, (∀ i, a ≤ i → i < a + k → g i = c) → (List.range' a k).flatMap (fun i => [g i]) = List.replicate k c
  | _, 0, _ => rfl
  | a, k + 1, h => by
    rw [List.range'_succ, List.flatMap_cons, h a (Nat.le_refl a) (by omega),
      flatMap_range'_rep g c (a + 1) k fun i h1 h2 => h i (by omega) (by omega)]
    rfl

theorem cntPat_rs {n : ℕ} (hn : n ≠ 0) (rs : Array (Array F128)) (hrs : ∀ k < n, (rs[k]!).size = 128) :
    cntPat (rsFrames n rs) = (List.replicate n 128 ++ [0], 0) := by
  have hB := cntPat_flatMap
    (F := fun k => [Frame.label "flock-ring-switch-v0".toUTF8, .vec (rs[k]!).toList, .sq 2 7])
    (G := fun k => [(rs[k]!).size]) 0 n
    (fun i _ _ => by simp only [cntPat_label, cntPat_vec, cntPat_sq _ (by decide : (7 : ℕ) ≠ 0), cntPat_nil,
      glueN_cons, Nat.add_zero, List.nil_append, Array.length_toList])
  rw [flatMap_range'_rep (fun k => (rs[k]!).size) 128 0 n (fun i _ h => hrs i (by omega))] at hB
  unfold rsFrames
  simp only [cntPat_label, cntPat_append, hB, cntPat_sq _ hn, cntPat_nil, glueN_closed]

/-- Level `i`'s seven rounds' value counts. -/
def lvV : List ℕ := [4, 5, 0, 5, 0, 0, 4]

/-- Pass `i` of the levels' loop's value counts, of `r` passes, at `yr` final values. -/
def passV (r yr i : ℕ) : List ℕ := [0, 4, 4, 4] ++ if i + 1 < r then lvV else [4 + yr, 0, 0]

/-- **A rep's rounds' value counts.** -/
def schedV (sch : Flock.Schedule) (m kLog n yr : ℕ) : List ℕ :=
  [0, 0, 128] ++ List.replicate (m - 6) 2 ++ [2, 0] ++ List.replicate (kLog - 6) 2 ++ [64] ++
    List.replicate n 128 ++ [0, 1, 1] ++ List.replicate sch.k0 4 ++ lvV ++
    (List.range' 0 (sch.levels.size - 1)).flatMap (passV (sch.levels.size - 1) yr)

theorem cntPat_levelFr {sch : Flock.Schedule} (p : LigeritoProof)
    (hq : ∀ i < sch.levels.size, (sch.levels.getD i default).q ≠ 0) {i : ℕ} (hi : i < sch.levels.size - 1) :
    cntPat (levelFr sch p i) = (passV (sch.levels.size - 1) p.finalProof.yr.size i, 0) := by
  unfold levelFr passV
  by_cases hc : i + 1 < sch.levels.size - 1
  · simp only [hc, ↓reduceIte, cntPat_append, cntPat_chFr p 3, cntPat_capFr,
      cntPat_introFr p (Nat.add_one_ne_zero _), cntPat_queryFr p (hq (i + 1) (by omega)), cntPat_kFr,
      cntPat_bytes, cntPat_sq _ Nat.one_ne_zero, cntPat_nil, glueN_closed, glueN_cons, glueN_nil, Nat.add_zero,
      List.nil_append, List.cons_append, List.append_nil, lvV]
    rfl
  · simp only [hc, ↓reduceIte, cntPat_append, cntPat_chFr p 3, cntPat_ones,
      cntPat_queryFr p (hq (sch.levels.size - 1) (by omega)), cntPat_bytes, cntPat_sq _ Nat.one_ne_zero,
      cntPat_nil, glueN_cons, glueN_nil, glueN_closed, Nat.add_zero, List.nil_append, List.cons_append,
      Array.length_toList]
    rfl

theorem cntPat_lig {sch : Flock.Schedule} (T : F128) (cap : Array ByteArray) (p : LigeritoProof)
    (hk : 0 < sch.k0) (hq : ∀ i < sch.levels.size, (sch.levels.getD i default).q ≠ 0) (hn : sch.n ≠ 0)
    (hL : 1 ≤ sch.levels.size) :
    cntPat (ligFrames sch T cap p) = (1 :: 1 :: (List.replicate sch.k0 4 ++ (lvV ++
      (List.range' 0 (sch.levels.size - 1)).flatMap (passV (sch.levels.size - 1) p.finalProof.yr.size))), 0) := by
  have hloop := cntPat_flatMap (F := levelFr sch p)
    (G := passV (sch.levels.size - 1) p.finalProof.yr.size) 0 (sch.levels.size - 1)
    (fun i _ h => cntPat_levelFr p hq (by omega))
  obtain ⟨j, hj⟩ : ∃ j, sch.k0 = j + 1 := ⟨sch.k0 - 1, by omega⟩
  unfold ligFrames
  rw [hj]
  simp only [cntPat_append, hloop, cntPat_label, cntPat_one, cntPat_capFr, cntPat_sq _ hn, cntPat_bytes,
    cntPat_sq _ Nat.one_ne_zero, cntPat_kFr, cntPat_chFr, cntPat_introFr p (Nat.add_one_ne_zero _),
    cntPat_queryFr p (hq 0 (by omega)), cntPat_nil, glueN_cons, glueN_nil, glueN_closed, Nat.add_zero,
    Nat.zero_add, List.nil_append, List.cons_append, List.append_nil, List.append_assoc, List.replicate_succ, lvV]

/-- **A rep's frames' rounds' value counts are `schedV`'s**, with nothing pending. -/
theorem cntPat_repFrames {st : Flock.Setup} {sch : Flock.Schedule} (d : ByteArray) (e : ℕ) (p : Flock.Proof)
    (T : F128) (hm : 14 ≤ st.m) (hk : 0 < sch.k0) (hq : ∀ i < sch.levels.size, (sch.levels.getD i default).q ≠ 0)
    (hn : sch.n ≠ 0) (hL : 1 ≤ sch.levels.size) (hab : p.zerocheck.round1ab.size = 64)
    (hc : p.zerocheck.round1c.size = 64) (hz : p.lincheck.zPartial.size = 64)
    (hrs : ∀ k < 2 + e, (p.ringSwitches[k]!).size = 128) :
    cntPat (repFrames st sch d (2 + e) p T) =
      (schedV sch st.m st.fold.kLog (2 + e) p.ligerito.finalProof.yr.size, 0) := by
  unfold repFrames schedV
  simp only [cntPat_dom, cntPat_append, cntPat_zc hm, cntPat_lc, cntPat_rs (by omega) _ hrs,
    cntPat_lig T p.cap p.ligerito hk hq hn hL, hab, hc, hz, glueN_cons, glueN_closed, glueN_nil, Nat.add_zero,
    List.nil_append, List.cons_append, List.append_assoc]

/-! ## Flags and V\*'s round sizes -/

theorem schedPat_flags (sch : Flock.Schedule) {m : ℕ} (hm : 6 ≤ m) (kLog n yr : ℕ) :
    (schedPat sch m kLog n).map Prod.fst = (schedV sch m kLog n yr).map (· != 0) := by
  have h5 : m - 5 = m - 6 + 1 := by omega
  have hl : ∀ r, (List.range' 0 (sch.levels.size - 1)).flatMap (fun i => (passT sch r i).map Prod.fst) =
      (List.range' 0 (sch.levels.size - 1)).flatMap (fun i => (passV r yr i).map (· != 0)) := fun r => by
    congr 1
    funext i
    unfold passT passV lvT lvV
    split <;> simp
  unfold schedPat schedV
  simp only [List.map_append, List.map_replicate, List.map_flatMap, hl, h5, List.replicate_succ']
  simp [lvT, lvV, List.append_assoc]

/-- **`schedV`'s nonzero entries are V\*'s rounds' sizes.** -/
theorem schedV_filter (sch : Flock.Schedule) (m kLog n : ℕ) :
    (schedV sch m kLog n (2 <<< sch.ext)).filter (· != 0) = VStarSizes.sizesV ⟨m, kLog, n⟩ sch := by
  have hl : ((List.range' 0 (sch.levels.size - 1)).flatMap (passV (sch.levels.size - 1) (2 <<< sch.ext))).filter
      (· != 0) = (List.range' 0 (sch.levels.size - 1)).flatMap fun i =>
        if i + 1 < sch.levels.size - 1 then [4, 4, 4, 4, 5, 5, 4] else [4, 4, 4, 4 + (2 <<< sch.ext)] := by
    rw [List.filter_flatMap]
    congr 1
    funext i
    unfold passV lvV
    split <;> simp
  unfold schedV VStarSizes.sizesV
  simp only [List.filter_append, hl, List.filter_replicate]
  simp [lvV, List.append_assoc, List.replicate_succ']

theorem commitRounds_eq {rs : Array RecVStar.SchedRound} {L : List ℕ}
    (h : rs.toList.map (·.2.2.2) = L.map (· != 0)) :
    RecVStar.commitRounds rs = (List.range L.length).filter fun i => L.getD i 0 != 0 := by
  have hlen : rs.size = L.length := by
    have := congrArg List.length h
    simpa using this
  unfold RecVStar.commitRounds
  rw [hlen]
  refine List.filter_congr fun i hi => ?_
  have hi' : i < L.length := List.mem_range.1 hi
  have hr : i < rs.size := hlen ▸ hi'
  have := congrArg (·[i]?) h
  simp only [List.getElem?_map, Array.getElem?_toList, Array.getElem?_eq_getElem hr, List.getElem?_eq_getElem hi',
    Option.map_some, Option.some.injEq] at this
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hr, Option.getD_some, this,
    List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi', Option.getD_some]

/-- **The `kk`-th committing round's V\* size is its value count**: round `t`, the `kk`-th of `commitRounds`, has
`schedV`'s `t`-th entry as `rounds`' `kk`-th size. -/
theorem rounds_at {m kLog claims kk t : ℕ} {sch : Flock.Schedule} {rs : Array RecVStar.SchedRound}
    (hs : fast100 m = some sch) (h : RecVStar.repRounds m kLog claims = some rs)
    (ht : (RecVStar.commitRounds rs)[kk]? = some t) :
    (_root_.VStar.Algebra.rounds ⟨m, kLog, claims⟩ sch)[kk]!.size =
      (schedV sch m kLog claims (2 <<< sch.ext)).getD t 0 := by
  obtain ⟨h22, -⟩ := RecDeploy.fast100_sizes hs
  have hfl : rs.toList.map (·.2.2.2) = (schedV sch m kLog claims (2 <<< sch.ext)).map (· != 0) := by
    rw [map_flag, repRounds_flagTop hs h, schedPat_flags sch (show 6 ≤ m by omega) kLog claims (2 <<< sch.ext)]
  rw [commitRounds_eq hfl] at ht
  have h2 := filter_idx _ kk t ht
  rw [schedV_filter, ← VStarSizes.rounds_sizes, List.getElem?_map, Array.getElem?_toList] at h2
  obtain ⟨a, ha, hae⟩ := Option.map_eq_some_iff.1 h2
  obtain ⟨hlt, rfl⟩ := Array.getElem?_eq_some_iff.1 ha
  rw [getElem!_pos _ kk hlt]
  exact hae

/-! ## An honest rep's rows' sizes -/

theorem vals_length_chunk {F : List Frame} {V : List ℕ} (h : cntPat F = (V, 0)) (t : ℕ) :
    (vals ((chunks F).1.getD t [])).length = V.getD t 0 := by
  have hV : (chunks F).1.map nV = V := congrArg Prod.fst h
  rw [← hV, List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map]
  cases (chunks F).1[t]? <;> rfl

theorem padRow_len (vs : List F128) : (padRow (colBytes vs)).length = 2 * _root_.VStar.portWordsOf vs.length := by
  simp only [padRow, List.length_append, List.length_replicate, FlockSoundness.Refine.Frame.colBytes_length,
    _root_.VStar.portWordsOf]
  omega

/-- **An honest round's rows' sizes**: its tops' node rows, 128 bytes each at 64-byte nodes, then its value row. -/
theorem honestRows_getD : ∀ (sched : List RecVStar.SchedRound) (chs : List (List Frame))
    (caps : List (Array ByteArray)), (∀ a ∈ caps, ∀ b ∈ a, b.size = 64) → ∀ t, t < sched.length → t < chs.length →
    ∃ n, ((honestRows sched chs caps).getD t []).map ByteArray.size =
      List.replicate n 128 ++
        (if (sched.getD t default).2.2.2 then [(padRow (colBytes (vals (chs.getD t [])))).length] else [])
  | [], _, _, _, t, ht, _ => absurd ht (Nat.not_lt_zero t)
  | _ :: _, [], _, _, t, _, hc => absurd hc (Nat.not_lt_zero t)
  | s :: ss, ch :: chs, caps, hcap, 0, _, _ => by
    refine ⟨(((caps.take s.2.1.size).flatMap Array.toList).map nodeRow).length, ?_⟩
    simp only [honestRows, List.getD_cons_zero, List.map_append]
    congr 1
    · refine List.eq_replicate_iff.2 ⟨by simp, fun b hb => ?_⟩
      obtain ⟨r, hr, rfl⟩ := List.mem_map.1 hb
      obtain ⟨nd, hnd, rfl⟩ := List.mem_map.1 hr
      obtain ⟨a, ha, hb'⟩ := List.mem_flatMap.1 hnd
      rw [size_nodeRow, hcap a (List.mem_of_mem_take ha) nd (Array.mem_toList_iff.1 hb')]
    · by_cases hb : s.2.2.2 = true
      · simp only [if_pos hb, List.map_cons, List.map_nil]
        exact congrArg (fun n => [n]) List.size_toArray
      · simp only [if_neg hb, List.map_nil]
  | s :: ss, ch :: chs, caps, hcap, t + 1, ht, hc => by
    obtain ⟨n, hn⟩ := honestRows_getD ss chs (caps.drop s.2.1.size)
      (fun a ha => hcap a (List.mem_of_mem_drop ha)) t (by simpa using ht) (by simpa using hc)
    exact ⟨n, by simpa only [honestRows, List.getD_cons_succ] using hn⟩

/-- **The honest proof at a rep**: `sizedProof` at the rep's shape, with the context's cap. -/
def openP (c : RepCtx) (e : ℕ) : Flock.Proof :=
  { sizedProof c.st c.sch c.z (FlockSoundness.Refine.shapeOf c.st c.sch e) with
    cap := c.cap0.toArray
    ligerito := { (sizedProof c.st c.sch c.z (FlockSoundness.Refine.shapeOf c.st c.sch e)).ligerito with
      initialCap := c.cap0.toArray } }

theorem openP_sizes (c : RepCtx) (e : ℕ) (hcap : c.cap0.length = 2 ^ (c.sch.levels.getD 0 default).cap)
    (hcapE : ∀ b ∈ c.cap0, b.size = c.st.merkle.digestLen) :
    ProofSizes (FlockSoundness.Refine.shapeOf c.st c.sch e) (openP c e) :=
  let h := sizedProof_sizes c.st c.sch c.z (FlockSoundness.Refine.shapeOf c.st c.sch e)
  { r1ab := h.r1ab, r1c := h.r1c, zcRounds := h.zcRounds, lcRounds := h.lcRounds, zPartial := h.zPartial
    rs := h.rs, rsEach := h.rsEach
    cap := by show c.cap0.toArray.size = _; rw [List.size_toArray]; exact hcap
    capEach := fun b hb => hcapE b (List.mem_toArray.1 hb)
    rcaps := h.rcaps, rcapsEach := h.rcapsEach, yr := h.yr, messages := h.messages, queryNonces := h.queryNonces
    oodValues := h.oodValues, claimNonces := h.claimNonces, consistencyNonces := h.consistencyNonces }

theorem openP_rep (c : RepCtx) {e : ℕ} (hn : c.n = 2 + e) (hL : 2 ≤ c.sch.levels.size)
    (hcap : c.cap0.length = 2 ^ (c.sch.levels.getD 0 default).cap)
    (hcapE : ∀ b ∈ c.cap0, b.size = c.st.merkle.digestLen) : HonestRep c (openP c e) :=
  ⟨⟨e, hn, openP_sizes c e hcap hcapE⟩, hL,
    sizedProof_nonces c.st c.sch c.z (FlockSoundness.Refine.shapeOf c.st c.sch e), rfl, rfl, rfl⟩

/-- **An honest rep's rows' byte sizes**: at a rep whose schedule is its frames' and whose cap has its level-0 top's
64-byte nodes, `openP` is honest at the cap, and round `t` of its rows is its tops' 128-byte rows, then, where it
absorbs, a row of `2 · portWordsOf` of `schedV`'s `t`-th entry bytes. -/
theorem open_sized {c : RepCtx} {e : ℕ} (hsc : SchedOf c) (hn : c.n = 2 + e)
    (hcap : c.cap0.length = 2 ^ (c.sch.levels.getD 0 default).cap) (hcapE : ∀ b ∈ c.cap0, b.size = 64)
    (hdig : c.st.merkle.digestLen = 64) (T : F128) :
    HonestRep c (openP c e) ∧ (openP c e).cap.toList = c.cap0 ∧
      (repRows c (openP c e) T).length = c.sched.length ∧
      ∀ t < c.sched.length, ((repRows c (openP c e) T).getD t []).map ByteArray.size =
        List.replicate (c.sched.getD t default).2.1.toList.sum 128 ++
          (if (c.sched.getD t default).2.2.2 then
            [2 * _root_.VStar.portWordsOf ((schedV c.sch c.st.m c.st.fold.kLog (2 + e) (2 <<< c.sch.ext)).getD t 0)]
          else []) := by
  obtain ⟨hs, -, rs, hrs, hsc'⟩ := id hsc
  obtain ⟨h22, -, hn7, hk0, hlv⟩ := RecDeploy.fast100_sizes hs
  have hk : 0 < c.sch.k0 := by rw [hk0]; split_ifs <;> omega
  have hL : 2 ≤ c.sch.levels.size := by rw [hlv]; split_ifs <;> omega
  have hcapE' : ∀ b ∈ c.cap0, b.size = c.st.merkle.digestLen := fun b hb => (hcapE b hb).trans hdig.symm
  have hp := openP_rep c hn hL hcap hcapE'
  have hcp : (openP c e).cap.toList = c.cap0 := by
    show c.cap0.toArray.toList = c.cap0
    exact List.toList_toArray
  have hF := repFits c (openP c e) T hsc hp hcp
  have hsz := openP_sizes c e hcap hcapE'
  have hyr : (openP c e).ligerito.finalProof.yr.size = 2 <<< c.sch.ext := by
    rw [hsz.yr]
    show 2 ^ (c.sch.ext + 1) = 2 <<< c.sch.ext
    rw [Nat.shiftLeft_eq, Nat.pow_succ, Nat.mul_comm]
  have hrs128 : ∀ k < 2 + e, ((openP c e).ringSwitches[k]!).size = 128 := fun k hk => by
    have hk' : k < (openP c e).ringSwitches.size := by rw [hsz.rs]; exact hk
    rw [getElem!_pos _ k hk']
    exact hsz.rsEach _ (Array.getElem_mem hk')
  have hcnt := cntPat_repFrames (st := c.st) (sch := c.sch) c.dom e (openP c e) T (by omega) hk (q_ne_zero hs)
    (by omega) (by omega) hsz.r1ab hsz.r1c hsz.zPartial hrs128
  rw [hyr] at hcnt
  have hch : chunksOf c (openP c e) T = (chunks (repFrames c.st c.sch c.dom (2 + e) (openP c e) T)).1 := by
    unfold chunksOf
    rw [hn]
  have hvl : ∀ t, (vals ((chunksOf c (openP c e) T).getD t [])).length =
      (schedV c.sch c.st.m c.st.fold.kLog (2 + e) (2 <<< c.sch.ext)).getD t 0 := fun t => by
    rw [hch]
    exact vals_length_chunk hcnt t
  have hrc : ∀ a ∈ (openP c e).ligerito.recursiveCaps.toList, ∀ b ∈ a, b.size = 64 := by
    intro a ha b hb
    obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem ha
    rw [Array.getElem_toList] at hb
    exact ((hsz.rcapsEach i (by simpa using hi)).2 b hb).trans hdig
  have hlen : (repRows c (openP c e) T).length = c.sched.length := hF.sched.symm
  refine ⟨hp, hcp, hlen, fun t ht => ?_⟩
  have htc : t < (chunksOf c (openP c e) T).length := by rw [hF.rounds, hlen]; exact ht
  obtain ⟨n, hrow⟩ := honestRows_getD c.sched (chunksOf c (openP c e) T) (openP c e).ligerito.recursiveCaps.toList
    hrc t ht htc
  have hE := hF.elems t (by rw [hlen]; exact ht)
  have h1 := congrArg List.length hrow
  have h2 : ((honestRows c.sched (chunksOf c (openP c e) T) (openP c e).ligerito.recursiveCaps.toList).getD t
      []).length = (c.sched.getD t default).2.1.toList.sum + (if (c.sched.getD t default).2.2.2 then 1 else 0) :=
    hE.1
  have hn' : n = (c.sched.getD t default).2.1.toList.sum := by
    simp only [List.length_map, List.length_append, List.length_replicate] at h1
    rw [h1] at h2
    split_ifs at h2 <;> simp only [List.length_cons, List.length_nil] at h2 <;> omega
  rw [show repRows c (openP c e) T =
    honestRows c.sched (chunksOf c (openP c e) T) (openP c e).ligerito.recursiveCaps.toList from rfl, hrow, hn']
  congr 1
  by_cases hb : (c.sched.getD t default).2.2.2 = true
  · rw [if_pos hb, if_pos hb, padRow_len, hvl]
  · rw [if_neg hb, if_neg hb]

end FlockDeploy.Cnt
