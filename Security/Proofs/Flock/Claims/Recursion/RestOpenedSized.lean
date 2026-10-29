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
import Proofs.Flock.Claims.Recursion.SchedCount
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

namespace RecDeploy.RestRead

open Flock
open Claims (Msg)
open Claims.Recursion.VStarB

/-- **`schedBridge` with its setup's coins and the shape's counts**: the table's `m`, `k_log` and claim count are the
shape's, its rep schedule is `repRounds` there, and the inner statement is set up at `RecCoins.innerTags` with no
public inputs, at a dense shape. -/
theorem schedBridge' {x : Msg} (hx : DenseX x) :
    ∃ r rs s spec st, readX x = some r ∧ repSched r.shape = some rs ∧
      FlockDeploy.setupOf false (xIOf x) = some (s, spec, #[st]) ∧ st.m = r.shape.m ∧
      r.shape.kLog = st.c.kLog ∧ r.shape.claims = 2 + spec.points * st.regions.size ∧
      RecVStar.repRounds r.shape.m r.shape.kLog r.shape.claims = some rs ∧
      repSchedAt spec st = rs.toList ∧ kI x = 2 + 2 * rs.size ∧
      s.read.tags = RecCoins.innerTags ∧ s.read.publicInputs = none ∧
      r.shape.holo = false ∧ r.shape.sparse = none := by
  obtain ⟨st0, hst, hf⟩ := hx
  obtain ⟨r, hr, ha, rfl⟩ := parse_read hst
  have hh : r.shape.holo = false := hf.1
  have hsp : r.shape.sparse = none := by
    have h2 : (r.shape.sparse.map fun sp => (sp.nu, sp.ell, sp.sLog, sp.tLog)) = none := hf.2
    cases hs : r.shape.sparse with
    | none => rfl
    | some _ => simp [hs] at h2
  obtain ⟨f, spec', st', s, spec, st, hfs, hstm, hset, hm, hk, hreg, hpts, htags, hpi⟩ := innerSetup_files hr ha
  unfold RecScope.accepts at ha
  simp only [Bool.and_eq_true, RecScope.isOk_iff] at ha
  obtain ⟨⟨⟨⟨ss, hss⟩, -⟩, ⟨u, hsh⟩⟩, -⟩ := ha
  have hss : RecVStar.statements r.shape = .ok ss := hss
  have hsh : RecSetup.checkShape (some r.shape) (filesOf r) = .ok u := hsh
  rw [hfs] at hsh
  obtain ⟨spec'', st'', hstm', e1, e2, -, e4, -⟩ := checkShape_files hsh
  rw [hstm] at hstm'
  cases hstm'
  obtain ⟨-, -, hk6, -, -⟩ := RecScope.statements_ok hss
  obtain ⟨rs, hrs⟩ := repSched_of_statements hss
  have hrs' : RecVStar.repRounds r.shape.m r.shape.kLog r.shape.claims = some rs := by
    unfold repSched at hrs
    rw [hh] at hrs
    exact hrs
  have hrr := hrs'
  have hx : xIOf x = r.w.inner := by unfold xIOf; rw [hst]; rfl
  have hm' : r.shape.m = st.m := e1.trans hm.symm
  have hk' : r.shape.kLog = st.c.kLog := e2.trans hk.symm
  have hc' : r.shape.claims = 2 + spec.points * st.regions.size := by rw [e4, hpts, hreg]
  have hset' : FlockDeploy.setupOf false (xIOf x) = some (s, spec, #[st]) := by rw [hx]; exact hset
  refine ⟨r, rs, s, spec, st, hr, hrs, hset', hm'.symm, hk', hc', hrr, ?_, ?_, htags, hpi, hh, hsp⟩
  · rw [hm', hk', hc'] at hrs'
    simp only [repSchedAt, hrs', Option.map_some, Option.getD_some]
  · show (FlockDeploy.V false).k (xIOf x) = _
    rw [hx]
    exact k_plain hset hm' hk' hc' (by omega) hrs'

end RecDeploy.RestRead

namespace RecDeploy.RestElem

open Flock
open Claims (Msg)

/-- **A dense inner statement's one table has its regions' bytes**: `setupHidden` checks them. -/
theorem dense_regionBytes {x : Msg} {s : FlockDeploy.Ins} {spec : SessionSpec} {st : Stmt}
    (hset : FlockDeploy.setupOf false x = some (s, spec, #[st])) (htags : s.read.tags = RecCoins.innerTags)
    (hpi : s.read.publicInputs = none) : FlockVBridge.Family.RegionBytes st := by
  have h := FlockDeploy.Ins.setup_read (FlockDeploy.setupOf_some hset).2
  unfold Readable.Read.setup at h
  have htags' : s.read.tags = RecVStar.innerTags := htags
  rw [htags', hpi] at h
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okH
      FlockSoundness.Discharge.Recursion.innerTags_hiddenOutputs h with
    ⟨-, st', hst, -, hsts⟩ | ⟨hc, hsz, -, -, -⟩
  · have e : st = st' := by simpa using hsts
    subst e
    exact _root_.Claims.ZKMal.Masked.setupHidden_regionBytes hst
  · have : (#[st] : Array Stmt).size = 1 := rfl
    omega

end RecDeploy.RestElem

namespace RecDeploy.OpenSlots

open Flock
open Claims (Msg)
open Claims.Recursion.VStarB
open RecDeploy.DeployedR
open RecDeploy.RestRead

/-! ## Names and lists -/

theorem mIndex_v : RecVStar.mIndex "v" = none := by
  unfold Flock.RecVStar.mIndex; exact ite_eq_right_iff.2 fun h => absurd h (by decide +kernel)

theorem mIndex_empty : RecVStar.mIndex "" = none := by
  unfold Flock.RecVStar.mIndex; exact ite_eq_right_iff.2 fun h => absurd h (by decide +kernel)

theorem mIndex_acc0 : RecVStar.mIndex "acc0" = none := by
  unfold Flock.RecVStar.mIndex; exact ite_eq_right_iff.2 fun h => absurd h (by decide +kernel)

theorem mIndex_acc1 : RecVStar.mIndex "acc1" = none := by
  unfold Flock.RecVStar.mIndex; exact ite_eq_right_iff.2 fun h => absurd h (by decide +kernel)

theorem mIndex_default : RecVStar.mIndex (default : String × ℕ).1 = none := mIndex_empty

theorem mem_zip_map {α β γ : Type} (f : α → β) (g : α → γ) :
    ∀ (l : List α) {b : β} {c : γ}, (b, c) ∈ (l.map f).zip (l.map g) → ∃ a ∈ l, b = f a ∧ c = g a
  | [], _, _, h => by simp at h
  | a :: l, b, c, h => by
    simp only [List.map_cons, List.zip_cons_cons, List.mem_cons, Prod.mk.injEq] at h
    rcases h with ⟨rfl, rfl⟩ | h
    · exact ⟨a, List.mem_cons_self, rfl, rfl⟩
    · obtain ⟨a', ha', h1, h2⟩ := mem_zip_map f g l h
      exact ⟨a', List.mem_cons_of_mem _ ha', h1, h2⟩

theorem flatMap_rep {α : Type} (f : α → ℕ) (b : ℕ) :
    ∀ l : List α, l.flatMap (fun a => List.replicate (f a) b) = List.replicate (l.map f).sum b
  | [] => rfl
  | a :: l => by
    rw [List.flatMap_cons, flatMap_rep f b l, List.map_cons, List.sum_cons, List.replicate_append_replicate]

theorem cut_two {α : Type} (n : ℕ) (l : List α) {k : ℕ} (hk : k < 2) :
    (FlockDeploy.cutLens [n, n] l).getD k [] = (l.drop (k * n)).take n := by
  rcases (show k = 0 ∨ k = 1 by omega) with rfl | rfl <;> simp [FlockDeploy.cutLens]

theorem getD_block {α : Type} (g : ℕ → List α) {k n t : ℕ} (hk : k < 2) (ht : t < n) :
    ((((List.range (2 * n)).map g).drop (k * n)).take n).getD t [] = g (k * n + t) := by
  have hkn : k * n + t < 2 * n := by
    rcases (show k = 0 ∨ k = 1 by omega) with rfl | rfl <;> omega
  rw [List.getD_eq_getElem?_getD, List.getElem?_take_of_lt ht, List.getElem?_drop, List.getElem?_map,
    List.getElem?_range hkn]
  rfl

theorem arr_toList_getD {α : Type} (a : Array α) (i : ℕ) (d : α) : a.toList.getD i d = a.getD i d := by
  rw [List.getD_eq_getElem?_getD, Array.getElem?_toList, Array.getD_eq_getD_getElem?]

theorem getD_sizes (L : List (List ByteArray)) (t : ℕ) :
    (L.map (·.map ByteArray.size)).getD t [] = (L.getD t []).map ByteArray.size := by
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map]
  cases L[t]? <;> rfl

theorem ext_getD {α : Type} {l₁ l₂ : List α} (d : α) (hl : l₁.length = l₂.length)
    (h : ∀ t < l₁.length, l₁.getD t d = l₂.getD t d) : l₁ = l₂ := by
  refine List.ext_getElem hl fun t h1 h2 => ?_
  have := h t h1
  rwa [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h1,
    List.getElem?_eq_getElem h2, Option.getD_some, Option.getD_some] at this

theorem getElem_single {α : Type} (a : α) {i : ℕ} (h : i < (#[a] : Array α).size) : (#[a] : Array α)[i] = a := by
  obtain rfl : i = 0 := by
    have : i < 1 := h
    omega
  rfl

theorem nodeOf_size {b : ByteArray} (h : b.size = 128) : (FlockDeploy.nodeOf b).size = 64 := by
  have h' : b.data.size = 128 := h
  show ((b.data.toList.take (b.data.size - 64)).toArray).size = 64
  rw [List.size_toArray, List.length_take, Array.length_toList, h']
  all_goals omega

/-! ## The slots' rows' byte lengths -/

/-- **A slot's port is one of V\*'s input rows**, of some `w` words, and its row has `16 · w` bits. -/
theorem slot_row {σ : Slot} {sub : _root_.VStar.Sub} {u : _root_.VStar.UnitC}
    (hl : σ.c.leavesIn.getD 0 #[] = (Array.range u.inWords).map Int.ofNat)
    (hrows : σ.c.ports.toList.map (fun p => (p.name, p.words, p.bits)) = RecVStar.vstarRows sub u)
    (hnm : σ.nm ∉ sub.outputs) (hq : σ.c.ports.findIdx? (·.name == σ.nm) = some σ.q) :
    ∃ w, (σ.nm, w) ∈ sub.inputs.zip u.inPorts ∧ σ.row.length = 16 * w := by
  obtain ⟨hqs, hname, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hq
  have hname' : σ.c.ports[σ.q].name = σ.nm := by
    first
    | exact beq_iff_eq.1 hname
    | simpa using hname
  have hN := in_block hrows hnm hq
  rw [getElem!_pos σ.c.ports σ.q hqs] at hN
  have hbw : σ.c.ports[σ.q].bits = 16 * σ.c.ports[σ.q].words :=
    bits_words hrows _ (Array.getElem_mem_toList hqs)
  have hrow : σ.row.length = 16 * σ.c.ports[σ.q].words := by
    unfold Slot.row
    rw [rowHidden_eq σ.n hl hq hqs hN, Option.getD_some, List.length_map, List.length_map, List.length_range, hbw]
  refine ⟨σ.c.ports[σ.q].words, ?_, hrow⟩
  have hm : (σ.c.ports[σ.q].name, σ.c.ports[σ.q].words, σ.c.ports[σ.q].bits) ∈ RecVStar.vstarRows sub u := by
    rw [← hrows]
    exact List.mem_map.2 ⟨_, Array.getElem_mem_toList hqs, rfl⟩
  rw [hname'] at hm
  unfold RecVStar.vstarRows at hm
  rcases List.mem_append.1 hm with hm | hm
  · obtain ⟨⟨a, b⟩, hab, he⟩ := List.mem_map.1 hm
    simp only [Prod.mk.injEq] at he
    obtain ⟨rfl, rfl, -⟩ := he
    exact hab
  · obtain ⟨⟨a, b⟩, hab, he⟩ := List.mem_map.1 hm
    simp only [Prod.mk.injEq] at he
    obtain ⟨rfl, -, -⟩ := he
    exact absurd (List.of_mem_zip hab).1 hnm

/-- **A cap's slot opens a 128-byte row**: port `top`, `CAP_TOP_WORDS = 64` words. -/
theorem capSlot_len {x : Msg} {r : XR} (hr : readX x = some r) {s lv N : ℕ}
    (hst : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (_root_.VStar.Stmt.cap lv N))
    {ns : List ℕ} {σ : Slot} (hσ : σ ∈ capSlots r s ns) : (σ.row.length + 7) / 8 = 128 := by
  obtain ⟨c, hc, u, hu, hl, hrows⟩ := part_parsed hr hst
  obtain ⟨q, hq⟩ := Option.isSome_iff_exists.1 (capPort_isSome hu hrows)
  have hcO : cOf r s = c := by simp [cOf, hc]
  have hnm : "top" ∉ (_root_.VStar.Stmt.cap lv N).sub.outputs := by
    show "top" ∉ ["out"]
    decide
  have hu' : (Except.ok (_root_.VStar.capUnit lv N) : Except String _root_.VStar.UnitC) = .ok u := hu
  cases hu'
  obtain ⟨n, -, rfl⟩ := List.mem_map.1 hσ
  rw [hcO, hq, Option.getD_some]
  show ((⟨s, c, "top", q, n⟩ : Slot).row.length + 7) / 8 = 128
  obtain ⟨w, hw, hrow⟩ := slot_row (σ := ⟨s, c, "top", q, n⟩) hl hrows hnm hq
  have hw' : ("top", w) ∈ [("top", _root_.VStar.CAP_TOP_WORDS)] := hw
  have hw'' : w = _root_.VStar.CAP_TOP_WORDS := by simpa using hw'
  rw [hrow, hw'']
  all_goals first | decide | simp [_root_.VStar.CAP_TOP_WORDS]

/-- **A statement whose own ports hold a row is an inner part's**, at a dense shape: one of `parts` of the algebra's
`spec`. -/
theorem inner_of_ownPorts {sh : RecVStar.Inner} {vs : Array _root_.VStar.Stmt}
    (hvs : RecVStar.statements sh = .ok vs) (hsp : sh.sparse = none) (hh : sh.holo = false)
    {sv : _root_.VStar.Stmt} (hsv : sv ∈ vs) {q : String × String × Array (Option ℕ)}
    (hq : q ∈ RecVStar.ownPorts sv) :
    ∃ sch sp ps p jj name, (⟨sh.m, sh.kLog, sh.claims⟩ : _root_.VStar.Algebra.Shape).sched = .ok sch ∧
      _root_.VStar.Algebra.spec ⟨sh.m, sh.kLog, sh.claims⟩ = .ok sp ∧ _root_.VStar.parts sp = .ok ps ∧ p ∈ ps ∧
      sv = .inner jj name p := by
  obtain ⟨A, K, n, hA, hK, rfl⟩ := OpenLower.statements_dense hvs hsp
  rw [hh] at hA
  obtain ⟨sch, sp, ps, hsch, hspec, hps, rfl⟩ := FlockVBridge.Family.statementsOf_eq hA
  have hspec' : _root_.VStar.Algebra.spec ⟨sh.m, sh.kLog, sh.claims⟩ = .ok sp := hspec
  rcases Array.mem_append.1 hsv with hsv | hsv
  · rcases Array.mem_append.1 hsv with hsv | hsv
    · obtain ⟨l, -, rfl⟩ := List.mem_map.1 (List.mem_toArray.1 hsv)
      simp [RecVStar.ownPorts] at hq
    · obtain ⟨j, hj, rfl⟩ := List.mem_map.1 (List.mem_toArray.1 hsv)
      refine ⟨sch, sp, ps, ps[j]!, j, _, hsch, hspec', hps, ?_, rfl⟩
      rw [getElem!_pos ps j (List.mem_range.1 hj)]
      exact Array.getElem_mem _
  · obtain ⟨l, N, rfl⟩ := FlockC3.VStarList.caps_cap hK n sv hsv
    simp [RecVStar.ownPorts] at hq

/-- **A held row's slot opens `2 · portWordsOf` of its round's V\* size bytes**: its port is `m{kk}`, the `kk`-th
committing round's, of `rounds`' `kk`-th size. -/
theorem rowSlot_len {x : Msg} {r : XR} (hr : readX x = some r) (hh : r.shape.holo = false)
    (hsp : r.shape.sparse = none) {rs : Array RecVStar.SchedRound} (hrs : repSched r.shape = some rs) {j : ℕ}
    {σ : Slot} (h : rowSlot r j = some σ) :
    ∃ (sch : _root_.Flock.Schedule) (kk : ℕ), fast100 r.shape.m = some sch ∧
      (RecVStar.commitRounds rs)[kk]? = some (j % rs.size) ∧
      (σ.row.length + 7) / 8 = 2 * _root_.VStar.portWordsOf
        ((_root_.VStar.Algebra.rounds ⟨r.shape.m, r.shape.kLog, r.shape.claims⟩ sch)[kk]!.size) := by
  obtain ⟨s, -, hs⟩ := List.exists_of_findSome?_eq_some h
  unfold rowSlotAt at hs
  obtain ⟨hd, hf, hs⟩ := Option.bind_eq_some_iff.1 hs
  obtain ⟨c, hc, rfl⟩ := Option.map_eq_some_iff.1 hs
  obtain ⟨vs, hvs, hlen⟩ := readX_statements hr
  have hsl : s < r.w.parts.length := by
    refine Nat.lt_of_not_le fun hge => ?_
    rw [List.getElem?_eq_none hge] at hc
    cases hc
  have hst : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some vs[s] := by
    rw [hvs]
    simp only [Except.toOption, Option.bind_some]
    exact Array.getElem?_eq_getElem (by omega)
  have hdj : hd.1 = j := by
    have h1 := List.find?_some hf
    exact beq_iff_eq.1 h1
  obtain ⟨q, hq, k', rs', hck, -, hname, hfi, hrs', hmr⟩ :=
    mem_heldRows_round hst hc (List.mem_of_find?_eq_some hf)
  rw [hrs] at hrs'
  obtain rfl := Option.some.inj hrs'
  unfold msgRound at hmr
  obtain ⟨t, ht, htj⟩ := Option.map_eq_some_iff.1 hmr
  have htj' : hd.2.2.2 * rs.size + t = hd.1 := htj
  have htl : t < rs.size := by
    have hmem := List.mem_of_getElem? ht
    simp only [RecVStar.commitRounds, List.mem_filter, List.mem_range] at hmem
    exact hmem.1
  have hjm : j % rs.size = t := by
    rw [← hdj, ← htj', Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt htl]
  obtain ⟨c', hc', u, hu, hl, hrows⟩ := part_parsed hr hst
  rw [hc] at hc'
  cases hc'
  have hnm : hd.2.1 ∉ vs[s].sub.outputs := by
    rw [hname]
    exact (ownPorts_held hq).1
  have hfi' : c.ports.findIdx? (·.name == hd.2.1) = some hd.2.2.1 := by
    rw [hname]
    exact hfi
  obtain ⟨w, hw, hrowl⟩ := slot_row (σ := ⟨s, c, hd.2.1, hd.2.2.1, hd.2.2.2⟩) hl hrows hnm hfi'
  obtain ⟨sch, sp, ps, p, jj, name, hsch, hspec, hps, hp, hsvi⟩ :=
    inner_of_ownPorts hvs hsp hh (Array.getElem_mem (by omega : s < vs.size)) hq
  have hw1 : (hd.2.1, w) ∈ (_root_.VStar.Stmt.inner jj name p).sub.inputs.zip u.inPorts := by
    rw [← hsvi]
    exact hw
  have hu' : _root_.VStar.innerUnit p.S name = .ok u := by
    have := hu
    rw [hsvi] at this
    exact this
  have hq' : q ∈ RecVStar.ownPorts (_root_.VStar.Stmt.inner jj name p) := by
    rw [← hsvi]
    exact hq
  have hsub : (_root_.VStar.Stmt.inner jj name p).sub.inputs = p.S.ports.map (·.1) ++ ["v"] := rfl
  rw [hsub, VStarSizes.innerUnit_inPorts hu',
    List.zip_append (l₁ := p.S.ports.map (·.1)) (l₂ := p.S.ports.map fun p => _root_.VStar.portWordsOf p.2)
      (by simp)] at hw1
  simp only [RecVStar.ownPorts] at hq'
  obtain ⟨⟨nm0, w0⟩, -, hqe⟩ := List.mem_filterMap.1 hq'
  obtain ⟨k0, hk0, rfl⟩ := Option.map_eq_some_iff.1 hqe
  have hck' : RecVStar.cIndex s!"rec-c{k0}" = some k' := hck
  rw [FlockC3.Reps.cIndex_c] at hck'
  obtain rfl := Option.some.inj hck'
  have hmk : RecVStar.mIndex hd.2.1 = some k0 := by
    rw [hname]
    exact hk0
  have hwk : w = _root_.VStar.portWordsOf
      ((_root_.VStar.Algebra.rounds ⟨r.shape.m, r.shape.kLog, r.shape.claims⟩ sch)[k0]!.size) := by
    rcases List.mem_append.1 hw1 with hw1 | hw1
    · obtain ⟨a, ha, h1, hw2⟩ := mem_zip_map (·.1) (fun p => _root_.VStar.portWordsOf p.2) p.S.ports hw1
      have h1' : hd.2.1 = a.1 := h1
      rcases VStarSizes.parts_ports hps p hp a ha with rfl | ha2
      · rw [h1', mIndex_default] at hmk
        cases hmk
      · obtain ⟨s', hs', hports⟩ := VStarSizes.spec_ports hspec
        have hss : sch = s' := by
          rw [hsch] at hs'
          exact Except.ok.inj hs'
        subst hss
        rw [hports] at ha2
        rcases List.mem_append.1 ha2 with ha3 | ha3
        · obtain ⟨i, -, rfl⟩ := List.mem_map.1 ha3
          have h1'' : hd.2.1 = s!"m{i}" := h1'
          rw [h1'', FlockC3.Reps.mIndex_m, Option.some.injEq] at hmk
          subst hmk
          exact hw2
        · have ha4 : a = ("acc0", 2) ∨ a = ("acc1", 2) := by simpa using ha3
          rcases ha4 with rfl | rfl
          · have h1'' : hd.2.1 = "acc0" := h1'
            rw [h1'', mIndex_acc0] at hmk
            cases hmk
          · have h1'' : hd.2.1 = "acc1" := h1'
            rw [h1'', mIndex_acc1] at hmk
            cases hmk
    · have hv : hd.2.1 = "v" := by
        simp at hw1
        exact hw1.1
      rw [hv, mIndex_v] at hmk
      cases hmk
  refine ⟨sch, k0, RestRead.sched_fast100' hsch, by rw [hjm]; exact ht, ?_⟩
  show ((⟨s, c, hd.2.1, hd.2.2.1, hd.2.2.2⟩ : Slot).row.length + 7) / 8 = _
  rw [hrowl, hwk]
  omega

/-- **Stream round `j`'s slots' byte lengths**: its tops' 128-byte rows, then, where it absorbs, one row of
`2 · portWordsOf` of `schedV`'s `j % R`-th entry. -/
theorem roundSlots_sizes {x : Msg} {r : XR} (hr : readX x = some r) (hh : r.shape.holo = false)
    (hsp : r.shape.sparse = none) {rs : Array RecVStar.SchedRound} (hrs : repSched r.shape = some rs)
    {sch : _root_.Flock.Schedule} (hsch : fast100 r.shape.m = some sch)
    (hrr : RecVStar.repRounds r.shape.m r.shape.kLog r.shape.claims = some rs) {j : ℕ} (hj : j < 2 * rs.size) :
    (roundSlots r rs j).map (fun σ => (σ.row.length + 7) / 8) =
      List.replicate (rs.getD (j % rs.size) default).2.1.toList.sum 128 ++
        (if (rs.getD (j % rs.size) default).2.2.2 then
          [2 * _root_.VStar.portWordsOf ((FlockDeploy.Cnt.schedV sch r.shape.m r.shape.kLog r.shape.claims
            (2 <<< sch.ext)).getD (j % rs.size) 0)] else []) := by
  have hpos : 0 < rs.size := by omega
  unfold roundSlots
  rw [List.map_append, List.map_flatMap]
  congr 1
  · have hcap : ∀ e ∈ List.range ((rs.getD (j % rs.size) default).2.1).size,
        (capSlotsAt r rs j e).map (fun σ => (σ.row.length + 7) / 8) =
          List.replicate (((rs.getD (j % rs.size) default).2.1).getD e 0) 128 := by
      intro e he
      have hst := capStmt_at hr hrs (topSizes_at (Nat.mod_lt j hpos) (List.mem_range.1 he))
      rw [← Nat.add_assoc] at hst
      refine List.eq_replicate_iff.2 ⟨by simp [capSlotsAt, capSlots], fun b hb => ?_⟩
      obtain ⟨σ, hσ, rfl⟩ := List.mem_map.1 hb
      unfold capSlotsAt at hσ
      exact capSlot_len hr hst hσ
    refine (flatMap_congr_mem _ hcap).trans ?_
    rw [flatMap_rep, RestRead.range_map_getD]
  · cases hσ : rowSlot r j with
    | none =>
      have hA : (rs.getD (j % rs.size) default).2.2.2 = false := by
        by_contra hA
        have hA' : (rs.getD (j % rs.size) default).2.2.2 = true := by simpa using hA
        have hsr : streamRound x j = some (rs.getD (j % rs.size) default) := by
          unfold streamRound
          rw [hr, Option.bind_some, hrs, Option.bind_some, if_pos hj]
        obtain ⟨s, hs, hd, hhd, hdj⟩ := rowsCoverAll hr hsr hA'
        obtain ⟨c, hc⟩ := RestRead.heldRows_circuit hhd
        have hn : nS x = r.w.parts.length := by simp [nS, hr]
        have hfs := List.findSome?_eq_none_iff.1 hσ s (List.mem_range.2 (by omega))
        have hf : ((heldRows r s).find? (·.1 == j)).isSome := List.find?_isSome.2 ⟨hd, hhd, by simp [hdj]⟩
        obtain ⟨h', hh'⟩ := Option.isSome_iff_exists.1 hf
        simp [rowSlotAt, hh', hc] at hfs
      rw [hA]
      simp
    | some σ =>
      obtain ⟨sch', kk, hsch', hkk, hlen⟩ := rowSlot_len hr hh hsp hrs hσ
      rw [hsch] at hsch'
      obtain rfl := Option.some.inj hsch'
      have hA : (rs.getD (j % rs.size) default).2.2.2 = true := by
        have hmem := List.mem_of_getElem? hkk
        simp only [RecVStar.commitRounds, List.mem_filter, List.mem_range] at hmem
        exact hmem.2
      simp only [if_pos hA, Option.toList_some, List.map_cons, List.map_nil, hlen,
        FlockDeploy.Cnt.rounds_at hsch hrr hkk]

/-! ## The opened rows' sizes -/

/-- **The opened rows' byte sizes are the slots' rows'**, round by round, at every witness `finalR` reads. -/
theorem rows_os (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) {x : Msg}
    (hx : DenseX x) {ws : List Claims.Flock.Wit} (hl : ws.length = nS x) {os last}
    (hf : finalR (lastAt (restR elem) x ws) = some (os, last)) :
    ∃ r, readX x = some r ∧ os.map (fun o => (rowsR o).map ByteArray.size) =
      (slotRounds r).map fun l => l.map fun σ => (σ.row.length + 7) / 8 := by
  obtain ⟨r, hr, hlast, -⟩ := lastAt_parts elem hx hl
  have hn : nS x = r.w.parts.length := by simp [nS, hr]
  obtain ⟨N, -, hpos⟩ := capAt_readX hr
  have h0 : 0 < ws.length := by omega
  obtain ⟨d⟩ : Nonempty Claims.Flock.Wit := ⟨ws[0]⟩
  obtain ⟨L, hL⟩ : ∃ L : List (List (Msg × Msg)),
      L = (slotRounds r).map fun l => l.map fun σ => σ.opening (ws.getD σ.s d) := ⟨_, rfl⟩
  have hpl : ∀ p ∈ openPieces r, p.1 < ws.length := fun p hp => by
    rw [hl, hn]
    exact openPieces_lt hr p hp
  have hlens : openLens r = L.map fun l => l.map (·.1.length) := by
    rw [openLens_eq, hL, List.map_map]
    refine List.map_congr_left fun l _ => ?_
    simp only [Function.comp_apply, List.map_map]
    refine List.map_congr_left fun σ _ => ?_
    simp [Slot.opening, readForms_length, pieceLen]
  have hbytes : readPieces (openPieces r) ws = L.flatMap (·.flatMap fun p => p.1 ++ p.2) := by
    rw [openBytes r ws d hpl, hL, List.flatMap_map]
    refine flatMap_congr_mem _ fun l _ => ?_
    simp only [List.flatMap_map]
    exact flatMap_congr_mem _ fun σ _ => readSegs_segs _ σ
  have hsalt : ∀ l ∈ L, ∀ p ∈ l, p.2.length = 192 := by
    intro l hl p hp
    rw [hL] at hl
    obtain ⟨l', -, rfl⟩ := List.mem_map.1 hl
    obtain ⟨σ, -, rfl⟩ := List.mem_map.1 hp
    exact HmOpen.length_salt _ _ _ _
  have hf' := hf
  rw [hlast, finalR_trailer, hlens, hbytes, CutOpens.cutRounds_groups _ L hsalt] at hf'
  obtain ⟨rfl, -⟩ := Prod.mk.inj (Option.some.inj hf')
  refine ⟨r, hr, ?_⟩
  rw [hL, List.map_map, List.map_map]
  refine List.map_congr_left fun l _ => ?_
  simp only [Function.comp_apply, rowsR, List.map_map]
  refine List.map_congr_left fun σ _ => ?_
  simp only [Function.comp_apply]
  first
  | exact (List.size_toArray _).trans (readForms_length _ _)
  | simp [Slot.opening, readForms_length, ByteArray.size]

/-- **An honest rep at the opened rows' sizes**: where `B` has, round by round, the sizes `open_sized` gives an honest
rep's rows, `openP` is an honest rep at the context's cap whose rows have `B`'s sizes. -/
theorem rep_sized {C : FlockDeploy.RepCtx} {e : ℕ} (hsc : FlockDeploy.SchedOf C) (hn : C.n = 2 + e)
    (hcap : C.cap0.length = 2 ^ (C.sch.levels.getD 0 default).cap) (hcapE : ∀ b ∈ C.cap0, b.size = 64)
    (hdig : C.st.merkle.digestLen = 64) (B : List (List ℕ)) (hlen : B.length = C.sched.length)
    (hB : ∀ t < C.sched.length, B.getD t [] = List.replicate (C.sched.getD t default).2.1.toList.sum 128 ++
      (if (C.sched.getD t default).2.2.2 then
        [2 * _root_.VStar.portWordsOf ((FlockDeploy.Cnt.schedV C.sch C.st.m C.st.fold.kLog (2 + e)
          (2 <<< C.sch.ext)).getD t 0)] else [])) :
    ∃ (p : Flock.Proof) (T : F128), FlockDeploy.HonestRep C p ∧ p.cap.toList = C.cap0 ∧
      B = (FlockDeploy.repRows C p T).map (·.map ByteArray.size) := by
  obtain ⟨hp, hcp, hl, hsz⟩ := FlockDeploy.Cnt.open_sized hsc hn hcap hcapE hdig default
  refine ⟨_, default, hp, hcp, ext_getD [] (by rw [List.length_map, hl, hlen]) fun t ht => ?_⟩
  rw [hlen] at ht
  rw [getD_sizes, hsz t ht, hB t ht]

/-- **`OpenedSized` at every `elem`**: the opened rows are `elem`'s slots' (`rows_os`), whose sizes at each rep's
block are an honest rep's (`roundSlots_sizes`, `open_sized`), at the rep's context, whose schedule is its frames'
(`schedOf_repCtx`) and whose cap is round 0's link nodes, 64 bytes each (`capSlot_len`). -/
theorem openedSized_of (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) :
    RestElem.OpenedSized elem := by
  intro x hx c ws hl _ os last hf hos v hv k hk
  obtain ⟨r, rs, s, spec, st, hr, hrs, hset, hm, hk', hc', hrr, hsc, hkI, htags, hpi, hh, hsp⟩ := schedBridge' hx
  rw [hset] at hv
  obtain rfl := Option.some.inj hv
  dsimp only at hk ⊢
  have hk2 : k < 2 := by
    first
    | exact hk
    | simpa using hk
  have hk01 : k / 2 = 0 := by omega
  obtain ⟨sch, hsch', -⟩ := RecDeploy.repRounds_size hrr
  have hsch : fast100 st.m = some sch := by rw [hm]; exact hsch'
  obtain ⟨extras, he⟩ := FlockVBridge.Family.extraClaims_exists st (FlockDeploy.pointsOf spec (c 0))
    (RestElem.dense_regionBytes hset htags hpi)
  have hpts : (FlockDeploy.pointsOf spec (c 0)).size = spec.points := by
    unfold FlockDeploy.pointsOf
    rw [Array.size_map, Array.size_range]
  obtain ⟨r', hr', hos'⟩ := rows_os elem hx hl hf
  rw [hr] at hr'
  obtain rfl := Option.some.inj hr'
  have hc0 := capAt_level0 hr hm
  have hSR : slotRounds r = capSlots r (nS x - 1) (List.range (level0Nodes st)) ::
      (List.range (2 * rs.size)).map (roundSlots r rs) := by
    simp [slotRounds, hc0, hrs]
  rw [hSR] at hos'
  rcases os with _ | ⟨o0, os'⟩
  · simp at hos'
  simp only [List.map_cons, List.cons.injEq] at hos'
  obtain ⟨h0, h1⟩ := hos'
  simp only [List.headD_cons, List.tail_cons]
  have hlen0 : (rowsR o0).length = level0Nodes st := by
    simpa [capSlots] using congrArg List.length h0
  have hcs := capAt_stmt hr hc0
  have hrow0 : ∀ b ∈ rowsR o0, b.size = 128 := by
    intro b hb
    have hm0 : b.size ∈ (rowsR o0).map ByteArray.size := List.mem_map.2 ⟨b, hb, rfl⟩
    rw [h0] at hm0
    obtain ⟨σ, hσ, hσe⟩ := List.mem_map.1 hm0
    have := capSlot_len hr hcs hσ
    rw [← hσe]
    exact this
  have hlenN : (nodesR o0).length = level0Nodes st := by
    show ((rowsR o0).map FlockDeploy.nodeOf).length = _
    rw [List.length_map]
    exact hlen0
  have hk1 : k / 2 < (#[st]).size := by
    show k / 2 < 1
    omega
  have hst1 : (#[st])[k / 2]'hk1 = st := getElem_single st hk1
  generalize hC : FlockDeploy.repCtx spec #[st] (c 0) (nodesR o0) ((List.range (kI x)).map c).tail.dropLast k = C
  have hsc' : FlockDeploy.SchedOf C := by
    rw [← hC]
    exact FlockDeploy.schedOf_repCtx hk1 (by rw [hst1, hsch]; all_goals rfl) (by rw [hst1]; exact he)
  have hCst : C.st = Setup.ofCircuit st := by
    rw [← hC]
    show (FlockDeploy.tabsOf #[st])[k / 2]! = _
    rw [hk01]
    exact FlockDeploy.tabsOf_get! (sts := #[st]) (i := 0) (by show 0 < 1; omega)
  have hCsch : C.sch = sch := by
    rw [← hC]
    show (fast100 (FlockDeploy.tabsOf #[st])[k / 2]!.m).getD default = sch
    rw [hk01, FlockDeploy.tabsOf_get! (sts := #[st]) (i := 0) (by show 0 < 1; omega)]
    show (fast100 st.m).getD default = sch
    rw [hsch, Option.getD_some]
  have hCn : C.n = 2 + spec.points * st.regions.size := by
    rw [← hC]
    show 2 + (match (FlockDeploy.tabsOf #[st])[k / 2]!.extra (FlockDeploy.pointsOf spec (c 0)) with
      | .ok e => e.size
      | .error _ => 0) = _
    rw [hk01, FlockDeploy.tabsOf_get! (sts := #[st]) (i := 0) (by show 0 < 1; omega)]
    show 2 + (match st.extraClaims (FlockDeploy.pointsOf spec (c 0)) with
      | .ok e => e.size
      | .error _ => 0) = _
    have hsz : extras.size = spec.points * st.regions.size := by
      rw [FlockDeploy.extraClaims_size he, hpts, Nat.mul_comm]
    rw [he]
    exact congrArg (2 + ·) hsz
  have hCsched : C.sched = rs.toList := by
    rw [← hC]
    show RecDeploy.repSchedAt spec (#[st])[k / 2]! = _
    rw [hk01, show (#[st])[0]! = st from rfl]
    exact hsc
  have hC0 : C.cap0 = (nodesR o0).take (level0Nodes st) := by
    rw [← hC]
    show (FlockDeploy.cutLens (FlockDeploy.level0Lens #[st]) (nodesR o0)).getD (k / 2) [] = _
    rw [hk01]
    first
    | rfl
    | simp [FlockDeploy.cutLens, FlockDeploy.level0Lens]
  have hC0' : C.cap0 = nodesR o0 := by
    rw [hC0, List.take_of_length_le hlenN.le]
  have hN : level0Nodes st = 2 ^ (sch.levels.getD 0 default).cap := by
    simp [level0Nodes, hsch]
  have hcap : C.cap0.length = 2 ^ (C.sch.levels.getD 0 default).cap := by
    rw [hC0', hCsch, hlenN, hN]
  have hcapE : ∀ b ∈ C.cap0, b.size = 64 := by
    rw [hC0']
    intro b hb
    obtain ⟨row, hrow, rfl⟩ := List.mem_map.1 hb
    exact nodeOf_size (hrow0 row hrow)
  have hdig : C.st.merkle.digestLen = 64 := by
    rw [hCst]
    show st.merkle.digestLen = 64
    rw [RestElem.dense_merkle hset htags hpi]
    all_goals rfl
  obtain ⟨-, hfk, -⟩ := id hsc'
  have hCm : C.st.m = r.shape.m := by
    rw [hCst]
    exact hm
  have hCk : C.st.fold.kLog = r.shape.kLog := by
    rw [hfk, hCst]
    exact hk'.symm
  have he2 : 2 + spec.points * st.regions.size = r.shape.claims := hc'.symm
  have hRL : FlockDeploy.repLens spec #[st] = [rs.size, rs.size] := by
    have e0 : (#[st])[0]! = st := rfl
    unfold FlockDeploy.repLens
    rw [show 2 * (#[st]).size = 2 from rfl, show List.range 2 = [0, 1] from rfl]
    simp only [List.map_cons, List.map_nil, Nat.zero_div, show (1 : ℕ) / 2 = 0 from rfl, e0, hsc,
      Array.length_toList]
  have hBeq : ((FlockDeploy.cutLens (FlockDeploy.repLens spec #[st]) (os'.map rowsR)).getD k []).map
      (·.map ByteArray.size) = (((List.range (2 * rs.size)).map fun j =>
        (roundSlots r rs j).map fun σ => (σ.row.length + 7) / 8).drop (k * rs.size)).take rs.size := by
    rw [hRL, cut_two _ _ hk2, List.map_take, List.map_drop, List.map_map]
    refine congrArg (fun L => (L.drop (k * rs.size)).take rs.size) (h1.trans ?_)
    rw [List.map_map]
    all_goals rfl
  have hlenB : ((((List.range (2 * rs.size)).map fun j =>
      (roundSlots r rs j).map fun σ => (σ.row.length + 7) / 8).drop (k * rs.size)).take rs.size).length =
        C.sched.length := by
    rw [hCsched, List.length_take, List.length_drop, List.length_map, List.length_range, Array.length_toList]
    rcases (show k = 0 ∨ k = 1 by omega) with rfl | rfl <;> omega
  have hblk : ∀ t < C.sched.length, ((((List.range (2 * rs.size)).map fun j =>
      (roundSlots r rs j).map fun σ => (σ.row.length + 7) / 8).drop (k * rs.size)).take rs.size).getD t [] =
        List.replicate (C.sched.getD t default).2.1.toList.sum 128 ++
          (if (C.sched.getD t default).2.2.2 then
            [2 * _root_.VStar.portWordsOf ((FlockDeploy.Cnt.schedV C.sch C.st.m C.st.fold.kLog
              (2 + spec.points * st.regions.size) (2 <<< C.sch.ext)).getD t 0)] else []) := by
    intro t ht
    rw [hCsched, Array.length_toList] at ht
    have hj : k * rs.size + t < 2 * rs.size := by
      rcases (show k = 0 ∨ k = 1 by omega) with rfl | rfl <;> omega
    have hmod : (k * rs.size + t) % rs.size = t := by
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt ht]
    rw [getD_block _ hk2 ht]
    try dsimp only
    rw [roundSlots_sizes hr hh hsp hrs hsch' hrr hj, hmod, hCsched, arr_toList_getD rs, hCsch, hCm, hCk, he2]
  refine ⟨hsc', ?_⟩
  obtain ⟨p, T, hp, hcp, hpB⟩ := rep_sized hsc' hCn hcap hcapE hdig _ hlenB hblk
  exact ⟨p, T, hp, hcp, hBeq.trans hpB⟩

end RecDeploy.OpenSlots

namespace RecDeploy.RestElem

open RecDeploy.DeployedR
open RecDeploy.OpenRows

/-- **`OpenedSized` at the deployed last message `elemR`.** -/
theorem openedSized_elemR : OpenedSized elemR := OpenSlots.openedSized_of elemR

/-- **The last step reads at the deployed readers at `elemR`**, with no hypothesis. -/
theorem readsAt_elemR_closed : RestRead.ReadsAt rdR finalR (restR elemR) := readsAt_elemR' openedSized_elemR

end RecDeploy.RestElem
