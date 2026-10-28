import Proofs.Flock.Claims.Recursion.NextR
import Proofs.Flock.C3.Reps.LigNames

/-!
# A rep's frames absorb its proof's values in w's order

`repFrames` absorbs a decoded proof's values phase by phase: the zerocheck's (`round1_ab`, `round1_c`, each round's
pair, `(final_a, final_b)`), the lincheck's (each round's pair, `z_partial`), the ring switches', then Ligerito's (`T`,
the out-of-domain values and the sumcheck messages level by level, the final `yr`). V*'s `rounds` lists the same
elements, by name, in the same order. So at a plain shape whose dimensions are the setup's, and a proof of its sizes,
the values the frames absorb are the rounds' names read in the proof (`vals_repFrames_rounds`): `keyVal` reads a name's
value off the proof, and `T`.

* **Phase by phase** (`vals_zcFrames`, `vals_lcFrames`, `vals_rsFrames`, `vals_ligFrames`, `vals_repFrames`): each
  phase's frames absorb `keyVal` at that phase's element list (`zcEls`, `lcEls`, `rsEls`, `ligEls`).
* **The rounds' names** (`ligRounds_els`, `rounds_els`): Ligerito's rounds, through `ligRounds_eq`'s folds and level
  loop, flatten to `ligEls`; at a plain shape, all the rounds flatten to the four lists in turn.
-/

namespace FlockDecodeTies

open Flock
open FlockDeploy (vals)
open Flock.RecAlgebra (Key msgNames ligRounds rounds hRound)
open FlockSoundness.Refine (Frame)
open FlockSoundness.Discharge.Exec
  (repFrames zcFrames lcFrames rsFrames ligFrames levelFr introFr queryFr chFr kFr capFr)
open FlockC3.Reps (ligF1 ligF2 lig0 ligMid ligLast ligLevels ligRounds_eq)

/-! ## Values -/

/-- **A message element's value in a decoded proof** `p`, with Ligerito's batched value `T`; `0` at an element no frame
absorbs (`h`, the running sums, a transpose, the comb or a consistency sum). -/
def keyVal (p : Proof) (T : F128) : Key → F128
  | .ab i => p.zerocheck.round1ab[i]!
  | .c i => p.zerocheck.round1c[i]!
  | .g1 i => (p.zerocheck.rounds[i]!).1
  | .ginf i => (p.zerocheck.rounds[i]!).2
  | .fa => p.zerocheck.finalA
  | .fb => p.zerocheck.finalB
  | .q1 j => (p.lincheck.rounds[j]!).1
  | .qinf j => (p.lincheck.rounds[j]!).2
  | .zp l => p.lincheck.zPartial[l]!
  | .rs k u => (p.ringSwitches[k]!)[u]!
  | .T => T
  | .ood i => p.ligerito.oodValues[i]!
  | .msg i h c =>
    if c = 0 then (if h = 0 then (p.ligerito.messages[i]!).1 else (p.ligerito.messages[i]!).2).c0
    else (if h = 0 then (p.ligerito.messages[i]!).1 else (p.ligerito.messages[i]!).2).c1
  | .yr j => p.ligerito.finalProof.yr[j]!
  | _ => 0

/-! ## The elements, phase by phase -/

/-- Sumcheck message `i`'s four elements. -/
def msgEls (i : ℕ) : List Key := [Key.msg i 0 0, Key.msg i 0 1, Key.msg i 1 0, Key.msg i 1 1]

/-- The zerocheck's elements at `m` variables: `round1_ab`, `round1_c`, each round's pair, `(final_a, final_b)`. -/
def zcEls (m : ℕ) : List Key :=
  (List.range' 0 64).map Key.ab ++ (List.range' 0 64).map Key.c ++
    (List.range' 0 (m - 6)).flatMap (fun i => [Key.g1 i, Key.ginf i]) ++ [Key.fa, Key.fb]

/-- The lincheck's elements at `k_log`: each round's pair, then `z_partial`. -/
def lcEls (kLog : ℕ) : List Key :=
  (List.range' 0 (kLog - 6)).flatMap (fun j => [Key.q1 j, Key.qinf j]) ++ (List.range' 0 64).map Key.zp

/-- The ring switches' elements, for `n` claims. -/
def rsEls (n : ℕ) : List Key := (List.range' 0 n).flatMap fun k => (List.range' 0 128).map (Key.rs k)

/-- Level `i` of Ligerito's loop: four folding messages; then, below the last level, two out-of-domain values with
their messages and the query message, and at the last the final `yr`. -/
def lvEls (sch : Schedule) (i : ℕ) : List Key :=
  (List.range' (sch.k0 + 4 + 7 * i) 4).flatMap msgEls ++
    if i + 1 < sch.levels.size - 1 then
      Key.ood (3 + 2 * i) :: msgEls (sch.k0 + 8 + 7 * i) ++ Key.ood (4 + 2 * i) :: msgEls (sch.k0 + 9 + 7 * i) ++
        msgEls (sch.k0 + 10 + 7 * i)
    else (List.range' 0 (2 <<< sch.ext)).map Key.yr

/-- Ligerito's elements: `T`, level 0's out-of-domain value, its `k₀ + 1` folding messages, two out-of-domain values
with their messages, the query message, then the level loop's. -/
def ligEls (sch : Schedule) : List Key :=
  Key.T :: Key.ood 0 :: (msgEls 0 ++ (List.range' 1 sch.k0).flatMap msgEls ++
    (Key.ood 1 :: msgEls (sch.k0 + 1) ++ Key.ood 2 :: msgEls (sch.k0 + 2) ++ msgEls (sch.k0 + 3))) ++
    (List.range' 0 (sch.levels.size - 1)).flatMap (lvEls sch)

/-! ## The frames' values -/

theorem vals_nil : vals [] = [] := rfl
theorem vals_dom (b : ByteArray) (fs : List Frame) : vals (.dom b :: fs) = vals fs := rfl
theorem vals_label (b : ByteArray) (fs : List Frame) : vals (.label b :: fs) = vals fs := rfl
theorem vals_one (v : F128) (fs : List Frame) : vals (.one v :: fs) = v :: vals fs := rfl
theorem vals_vec (vs : List F128) (fs : List Frame) : vals (.vec vs :: fs) = vs ++ vals fs := rfl
theorem vals_bytes (b : ByteArray) (fs : List Frame) : vals (.bytes b :: fs) = vals fs := rfl
theorem vals_sq (k : UInt8) (n : ℕ) (fs : List Frame) : vals (.sq k n :: fs) = vals fs := rfl
theorem vals_capFr (cap : Array ByteArray) (fs : List Frame) : vals (capFr cap :: fs) = vals fs := rfl

theorem vals_append (a b : List Frame) : vals (a ++ b) = vals a ++ vals b := by
  induction a with
  | nil => rfl
  | cons f a ih =>
    cases f <;> simp only [List.cons_append, vals_dom, vals_label, vals_one, vals_vec, vals_bytes, vals_sq, ih,
      List.append_assoc]

theorem vals_flatMap {β : Type} (F : β → List Frame) :
    ∀ l : List β, vals (l.flatMap F) = l.flatMap fun b => vals (F b)
  | [] => rfl
  | b :: l => by rw [List.flatMap_cons, List.flatMap_cons, vals_append, vals_flatMap F l]

theorem vals_map_one : ∀ l : List F128, vals (l.map Frame.one) = l
  | [] => rfl
  | v :: l => by rw [List.map_cons, vals_one, vals_map_one l]

theorem flatMap_congr' {α β : Type} {F G : α → List β} :
    ∀ {l : List α}, (∀ a ∈ l, F a = G a) → l.flatMap F = l.flatMap G
  | [], _ => rfl
  | a :: l, h => by
    rw [List.flatMap_cons, List.flatMap_cons, h a List.mem_cons_self,
      flatMap_congr' fun b hb => h b (List.mem_cons_of_mem _ hb)]

/-- An array of `n` entries, as its entries in order. -/
theorem toList_range' {α : Type} [Inhabited α] {a : Array α} {n : ℕ} (h : a.size = n) :
    a.toList = (List.range' 0 n).map (a[·]!) := by
  rw [← List.range_eq_range']
  apply List.ext_getElem (by simp [h])
  intro i h1 h2
  simp only [List.getElem_map, List.getElem_range, Array.getElem_toList]
  rw [getElem!_pos a i (by simpa using h1)]

section frames

variable (p : Proof) (T : F128)

theorem vals_kFr (i : ℕ) : vals (kFr p.ligerito.messages[i]!) = (msgEls i).map (keyVal p T) := rfl

theorem vals_chFr : ∀ (k i : ℕ), vals (chFr p.ligerito i k) = ((List.range' i k).flatMap msgEls).map (keyVal p T)
  | 0, _ => rfl
  | k + 1, i => by
    rw [List.range'_succ, List.flatMap_cons, List.map_append, ← vals_kFr p T, ← vals_chFr k (i + 1),
      ← vals_append]
    rfl

theorem vals_introFr (dim ood i cbn : ℕ) :
    vals (introFr p.ligerito dim ood i cbn) = (Key.ood ood :: msgEls i).map (keyVal p T) := rfl

theorem vals_queryFr (lv : Level) (qn cn : ℕ) : vals (queryFr p.ligerito lv qn cn) = [] := rfl

theorem vals_levelFr (sch : Schedule) (hyr : p.ligerito.finalProof.yr.size = 2 <<< sch.ext) (i : ℕ) :
    vals (levelFr sch p.ligerito i) = (lvEls sch i).map (keyVal p T) := by
  unfold levelFr lvEls
  rw [vals_append, List.map_append, vals_chFr p T]
  by_cases h : i + 1 < sch.levels.size - 1
  · simp only [h, ↓reduceIte, vals_capFr, vals_append, vals_introFr p T, vals_queryFr p, vals_kFr p T, vals_bytes,
      vals_sq, vals_nil, List.append_nil, List.map_append, List.map_cons]
  · simp only [h, ↓reduceIte, vals_append, vals_map_one, vals_queryFr p, vals_bytes, vals_sq, vals_nil,
      List.append_nil, List.map_map]
    rw [toList_range' hyr]
    rfl

theorem vals_ligFrames (sch : Schedule) (hyr : p.ligerito.finalProof.yr.size = 2 <<< sch.ext)
    (cap : Array ByteArray) : vals (ligFrames sch T cap p.ligerito) = (ligEls sch).map (keyVal p T) := by
  simp only [ligFrames, ligEls, vals_append, vals_label, vals_one, vals_capFr, vals_sq, vals_bytes,
    vals_kFr p T, vals_chFr p T, vals_introFr p T, vals_queryFr p, vals_flatMap, vals_levelFr p T sch hyr,
    List.map_cons, List.map_append, List.map_flatMap, List.append_assoc, List.cons_append, List.nil_append]
  all_goals rfl

theorem vals_zcFrames (m : ℕ) (d c : ByteArray) (hab : p.zerocheck.round1ab.size = 64)
    (hc : p.zerocheck.round1c.size = 64) : vals (zcFrames m d c p.zerocheck) = (zcEls m).map (keyVal p T) := by
  simp only [zcFrames, zcEls, vals_append, vals_label, vals_bytes, vals_sq, vals_vec, vals_one, vals_nil,
    vals_flatMap, List.append_nil, List.map_append, List.map_map, List.map_flatMap, List.map_cons, List.map_nil]
  rw [toList_range' hab, toList_range' hc]
  rfl

theorem vals_lcFrames (k : ℕ) (hz : p.lincheck.zPartial.size = 64) :
    vals (lcFrames k p.lincheck) = (lcEls k).map (keyVal p T) := by
  simp only [lcFrames, lcEls, vals_append, vals_label, vals_sq, vals_vec, vals_one, vals_nil, vals_flatMap,
    List.append_nil, List.map_append, List.map_map, List.map_flatMap, List.map_cons, List.map_nil]
  rw [toList_range' hz]
  rfl

theorem vals_rsFrames (n : ℕ) (hrs : ∀ k < n, (p.ringSwitches[k]!).size = 128) :
    vals (rsFrames n p.ringSwitches) = (rsEls n).map (keyVal p T) := by
  simp only [rsFrames, rsEls, vals_label, vals_append, vals_flatMap, vals_vec, vals_sq, vals_nil, List.append_nil,
    List.map_flatMap, List.map_map]
  apply flatMap_congr'
  intro k hk
  obtain ⟨-, hk'⟩ := List.mem_range'_1.1 hk
  rw [toList_range' (hrs k (by omega))]
  rfl

/-- **A rep's frames absorb its proof's elements, phase by phase**, at sizes of the decoded proof's shape. -/
theorem vals_repFrames (st : Setup) (sch : Schedule) (dom : ByteArray) (n : ℕ)
    (hab : p.zerocheck.round1ab.size = 64) (hc : p.zerocheck.round1c.size = 64)
    (hz : p.lincheck.zPartial.size = 64) (hrs : ∀ k < n, (p.ringSwitches[k]!).size = 128)
    (hyr : p.ligerito.finalProof.yr.size = 2 <<< sch.ext) :
    vals (repFrames st sch dom n p T) =
      (zcEls st.m ++ lcEls st.fold.kLog ++ rsEls n ++ ligEls sch).map (keyVal p T) := by
  simp only [repFrames, vals_dom, vals_append, vals_zcFrames p T _ _ _ hab hc, vals_lcFrames p T _ hz,
    vals_rsFrames p T n hrs, vals_ligFrames p T sch hyr, List.map_append]

end frames

/-! ## The rounds' elements -/

theorem flat_push (out : Array (Array Key)) (r : Array Key) :
    (out.push r).flatten.toList = out.flatten.toList ++ r.toList := by
  simp [Array.toList_flatten]

theorem ligF1_els : ∀ (l : List ℕ) (mi : ℕ) (out : Array (Array Key)),
    (l.foldl ligF1 (mi, out)).1 = mi + l.length ∧
      (l.foldl ligF1 (mi, out)).2.flatten.toList = out.flatten.toList ++ (List.range' mi l.length).flatMap msgEls
  | [], mi, out => ⟨rfl, by simp⟩
  | _ :: l, mi, out => by
    obtain ⟨h1, h2⟩ := ligF1_els l (mi + 1) (out.push (msgNames mi))
    simp only [List.foldl_cons, List.length_cons, ligF1]
    refine ⟨by rw [h1]; omega, ?_⟩
    rw [h2, flat_push, List.range'_succ, List.flatMap_cons, List.append_assoc]
    rfl

theorem lig0_els (sch : Schedule) : (lig0 sch).1 = sch.k0 + 4 ∧ (lig0 sch).2.1 = 3 ∧
    (lig0 sch).2.2.flatten.toList = Key.T :: Key.ood 0 :: (msgEls 0 ++ (List.range' 1 sch.k0).flatMap msgEls ++
      (Key.ood 1 :: msgEls (sch.k0 + 1) ++ Key.ood 2 :: msgEls (sch.k0 + 2) ++ msgEls (sch.k0 + 3))) := by
  obtain ⟨a1, a2⟩ := ligF1_els (List.range' 0 (sch.k0 + 1)) 0 #[#[Key.T], #[Key.ood 0]]
  simp only [lig0]
  generalize List.foldl ligF1 (0, #[#[Key.T], #[Key.ood 0]]) (List.range' 0 (sch.k0 + 1)) = A at a1 a2 ⊢
  simp only [List.length_range', Nat.zero_add] at a1 a2
  rw [show List.range' 0 2 = [0, 1] from rfl]
  simp only [List.foldl_cons, List.foldl_nil, ligF2]
  refine ⟨?_, ?_, ?_⟩
  · first | trivial | omega | rfl
  · first | trivial | omega | rfl
  rw [flat_push, flat_push, flat_push, a2, a1, List.range'_succ, List.flatMap_cons]
  simp only [Array.toList_append, List.append_assoc]
  rfl

theorem ligMid_els (mi oi : ℕ) (out : Array (Array Key)) : (ligMid (mi, oi, out)).1 = mi + 7 ∧
    (ligMid (mi, oi, out)).2.1 = oi + 2 ∧
    (ligMid (mi, oi, out)).2.2.flatten.toList = out.flatten.toList ++ ((List.range' mi 4).flatMap msgEls ++
      (Key.ood oi :: msgEls (mi + 4) ++ Key.ood (oi + 1) :: msgEls (mi + 5) ++ msgEls (mi + 6))) := by
  simp only [ligMid]
  rw [show List.range' 0 4 = [0, 1, 2, 3] from rfl, show List.range' 0 2 = [0, 1] from rfl]
  simp only [List.foldl_cons, List.foldl_nil, ligF1, ligF2]
  refine ⟨?_, ?_, ?_⟩
  · first | trivial | omega | rfl
  · first | trivial | omega | rfl
  simp only [flat_push, Array.toList_append, List.append_assoc]
  rfl

theorem ligLast_els (ext mi oi : ℕ) (out : Array (Array Key)) :
    (ligLast ext (mi, oi, out)).2.2.flatten.toList =
      out.flatten.toList ++ ((List.range' mi 4).flatMap msgEls ++ (List.range' 0 (2 <<< ext)).map Key.yr) := by
  simp only [ligLast]
  rw [show List.range' 0 3 = [0, 1, 2] from rfl]
  simp only [List.foldl_cons, List.foldl_nil, ligF1]
  simp only [flat_push, Array.toList_append, Array.toList_map, Array.toList_range, List.range_eq_range',
    List.append_assoc]
  rfl

theorem ligLevels_nil (R : ℕ) (G1 G2 : ℕ × ℕ × Array (Array Key) → ℕ × ℕ × Array (Array Key))
    (init : ℕ × ℕ × Array (Array Key)) : ligLevels R G1 G2 [] init = init := rfl

theorem ligLevels_cons (R : ℕ) (G1 G2 : ℕ × ℕ × Array (Array Key) → ℕ × ℕ × Array (Array Key)) (i : ℕ)
    (l : List ℕ) (init : ℕ × ℕ × Array (Array Key)) :
    ligLevels R G1 G2 (i :: l) init = ligLevels R G1 G2 l (if i + 1 < R then G1 init else G2 init) := by
  unfold ligLevels
  rw [List.forIn_cons]
  by_cases h : i + 1 < R <;> simp only [h, ↓reduceIte, pure_bind] <;> rfl

theorem ligLevels_els (sch : Schedule) : ∀ (n a : ℕ) (init : ℕ × ℕ × Array (Array Key)),
    a + n = sch.levels.size - 1 → init.1 = sch.k0 + 4 + 7 * a → init.2.1 = 3 + 2 * a →
    (ligLevels (sch.levels.size - 1) ligMid (ligLast sch.ext) (List.range' a n) init).2.2.flatten.toList =
      init.2.2.flatten.toList ++ (List.range' a n).flatMap (lvEls sch)
  | 0, a, init, _, _, _ => by
    rw [List.range'_zero, ligLevels_nil, List.flatMap_nil, List.append_nil]
  | n + 1, a, (mi, oi, out), hR, h1, h2 => by
    simp only at h1 h2
    subst h1 h2
    rw [List.range'_succ, ligLevels_cons, List.flatMap_cons]
    by_cases ha : a + 1 < sch.levels.size - 1
    · obtain ⟨c1, c2, c3⟩ := ligMid_els (sch.k0 + 4 + 7 * a) (3 + 2 * a) out
      simp only [ha, ↓reduceIte]
      rw [ligLevels_els sch n (a + 1) _ (by omega) (by rw [c1]; omega) (by rw [c2]; omega), c3]
      simp only [lvEls, ha, ↓reduceIte]
      rw [show sch.k0 + 4 + 7 * a + 4 = sch.k0 + 8 + 7 * a by omega,
        show sch.k0 + 4 + 7 * a + 5 = sch.k0 + 9 + 7 * a by omega,
        show sch.k0 + 4 + 7 * a + 6 = sch.k0 + 10 + 7 * a by omega, show 3 + 2 * a + 1 = 4 + 2 * a by omega]
      simp only [List.append_assoc]
    · obtain rfl : n = 0 := by omega
      simp only [ha, ↓reduceIte]
      rw [List.range'_zero, ligLevels_nil, List.flatMap_nil, List.append_nil, ligLast_els]
      simp only [lvEls, ha, ↓reduceIte]

/-- **Ligerito's rounds' elements**, in order. -/
theorem ligRounds_els (sch : Schedule) : (ligRounds sch).flatten.toList = ligEls sch := by
  rw [ligRounds_eq]
  obtain ⟨h1, h2, h3⟩ := lig0_els sch
  rw [ligLevels_els sch (sch.levels.size - 1) 0 (lig0 sch) (by omega) (by omega) (by omega), h3]
  rfl

theorem flat_range_map (f : ℕ → Array Key) (n : ℕ) :
    ((Array.range n).map f).flatten.toList = (List.range' 0 n).flatMap fun i => (f i).toList := by
  rw [Array.toList_flatten, Array.toList_map, Array.toList_range, List.map_map, List.range_eq_range']
  rfl

/-- **The rounds' elements at a plain shape**, phase by phase: w's names but the running sums. -/
theorem rounds_els {sh : Shape} (hh : sh.holo = false) :
    (rounds sh).flatten.toList = zcEls sh.m ++ lcEls sh.kLog ++ rsEls sh.claims ++ ligEls sh.sched := by
  simp only [rounds, hRound, hh, Bool.false_eq_true, ↓reduceIte, Array.flatten_append, Array.toList_append,
    flat_range_map, ligRounds_els, Array.toList_map, Array.toList_range, List.range_eq_range']
  simp only [zcEls, lcEls, rsEls, Array.toList_flatten, List.map_cons, List.map_nil,
    List.flatten_cons, List.flatten_nil, List.append_nil, Array.toList_append, Array.toList_map, Array.toList_range,
    List.range_eq_range', List.append_assoc]

/-- **A rep's frames absorb its proof's values in w's order**: at a plain shape of the setup's dimensions and the
decoded proof's sizes, the frames' values are `keyVal` at the rounds' elements. -/
theorem vals_repFrames_rounds {sh : Shape} {st : Setup} {sch : Schedule} {n : ℕ} (dom : ByteArray) (p : Proof)
    (T : F128) (hm : st.m = sh.m) (hk : st.fold.kLog = sh.kLog) (hn : n = sh.claims) (hs : sch = sh.sched)
    (hh : sh.holo = false) (hab : p.zerocheck.round1ab.size = 64) (hc : p.zerocheck.round1c.size = 64)
    (hz : p.lincheck.zPartial.size = 64) (hrs : ∀ k < n, (p.ringSwitches[k]!).size = 128)
    (hyr : p.ligerito.finalProof.yr.size = 2 <<< sch.ext) :
    vals (repFrames st sch dom n p T) = (rounds sh).flatten.toList.map (keyVal p T) := by
  rw [vals_repFrames p T st sch dom n hab hc hz hrs hyr, rounds_els hh, hm, hk, hn, hs]

end FlockDecodeTies
