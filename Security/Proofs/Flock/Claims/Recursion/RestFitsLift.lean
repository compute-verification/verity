import Proofs.Flock.Claims.Recursion.RestReadR
import Proofs.Flock.Claims.Recursion.NextRHonest

/-!
# A fit lifts to rows of its rows' sizes (`fitsLift`)

A rep's fit (`FitsRep`) reads its rows' contents in two places only: the recursive caps are the rounds' tops, and each
round's value row holds the values its chunk absorbs. Everything else it asks of the proof is a size. So rows of the
same byte sizes as fitting ones have a fit (`RestRead.FitsLift`): take the tops of the new rows as the recursive caps,
and the leading bytes of each new value row as that round's values.

The values are realizable because a rep's frames read every absorbed value slot of a proof of its shape exactly once.
`repFrames_fields` and `ligFrames_fields` show the frames fix those slots, so the map from the slots to the frames'
values is injective; both sides have `|F128|` to the same power (`vals_repFrames_length`), so it is onto
(`realize`). Any 16 bytes are an `F128`'s by the same count (`toBytes_surj`).
-/

namespace FlockDeploy

open Flock
open FlockSoundness.Refine (Frame colBytes shapeOf)
open FlockSoundness.Discharge.DecodeZL (ProofSizes)
open FlockSoundness.Discharge.Exec (chunks closes chunks_flatten repFrames repFrames_chunks_snd repFrames_fields
  zcFrames lcFrames rsFrames ligFrames ligFrames_fields levelFr introFr queryFr chFr kFr capFr rs_size)

/-! ## Bytes -/

theorem card_UInt8 : Fintype.card UInt8 = 2 ^ 8 :=
  (FinEnum.card_eq_fintypeCard (α := UInt8)).symm.trans rfl

theorem card_F128 : Fintype.card F128 = 2 ^ 128 := by
  rw [Fintype.card_congr FlockSoundness.Discharge.ZkEveryCoin.f128Equiv, Fintype.card_prod, Fintype.card_fin,
    ← pow_add]

theorem byteArray_ext {x y : ByteArray} (h : x.data = y.data) : x = y := by
  cases x
  cases y
  cases h
  rfl

/-- An element's 16 bytes. -/
def toVec (a : F128) : List.Vector UInt8 16 := ⟨a.toBytes.data.toList, FlockSoundness.Refine.toBytes_length a⟩

theorem toVec_inj : Function.Injective toVec := fun a b h => by
  simp only [toVec] at h
  exact FlockSoundness.Refine.toBytes_inj (byteArray_ext (Array.toList_inj.1 (Subtype.mk.inj h)))

theorem card_F128_vec : Fintype.card F128 = Fintype.card (List.Vector UInt8 16) := by
  rw [card_vector, card_F128, card_UInt8]
  all_goals norm_num

/-- **Any 16 bytes are an element's.** -/
theorem toBytes_surj (l : List UInt8) (hl : l.length = 16) : ∃ a : F128, a.toBytes.data.toList = l := by
  obtain ⟨a, ha⟩ := ((Fintype.bijective_iff_injective_and_card toVec).2 ⟨toVec_inj, card_F128_vec⟩).2 ⟨l, hl⟩
  simp only [toVec] at ha
  exact ⟨a, Subtype.mk.inj ha⟩

/-- **Any whole 16-byte blocks are some elements' bytes.** -/
theorem colBytes_surj : ∀ (n : ℕ) (l : List UInt8), l.length = 16 * n →
    ∃ vs : List F128, vs.length = n ∧ colBytes vs = l
  | 0, l, h => ⟨[], rfl, (List.eq_nil_of_length_eq_zero (by omega)).symm⟩
  | n + 1, l, h => by
    obtain ⟨a, ha⟩ := toBytes_surj (l.take 16) (by rw [List.length_take]; omega)
    obtain ⟨vs, hl, hv⟩ := colBytes_surj n (l.drop 16) (by rw [List.length_drop]; omega)
    refine ⟨a :: vs, by rw [List.length_cons, hl], ?_⟩
    show a.toBytes.data.toList ++ colBytes vs = l
    rw [ha, hv, List.take_append_drop]

/-! ## Lists -/

theorem getD_map_nil {α β : Type} (f : List α → List β) (hf : f [] = []) :
    ∀ (L : List (List α)) (j : ℕ), (L.map f).getD j [] = f (L.getD j [])
  | [], _ => by simp [hf]
  | _ :: _, 0 => by simp
  | _ :: L, j + 1 => by simpa using getD_map_nil f hf L j

theorem eq_of_flatten {α : Type} : ∀ {A B : List (List α)}, A.map List.length = B.map List.length →
    A.flatten = B.flatten → A = B
  | [], [], _, _ => rfl
  | [], _ :: _, h, _ => by simp at h
  | _ :: _, [], h, _ => by simp at h
  | a :: A, b :: B, h, h' => by
    rw [List.map_cons, List.map_cons, List.cons.injEq] at h
    rw [List.flatten_cons, List.flatten_cons] at h'
    obtain ⟨e, e'⟩ := List.append_inj h' h.1
    rw [e, eq_of_flatten h.2 e']

theorem length_flatMap_eq {α β : Type} {F : α → List β} {k : ℕ} :
    ∀ {l : List α}, (∀ a ∈ l, (F a).length = k) → (l.flatMap F).length = l.length * k
  | [], _ => by simp
  | a :: l, h => by
    rw [List.flatMap_cons, List.length_append, h a List.mem_cons_self,
      length_flatMap_eq fun b hb => h b (List.mem_cons_of_mem a hb), List.length_cons]
    ring

theorem cutLens_map {α β : Type} (f : α → β) : ∀ (ns : List ℕ) (l : List α),
    cutLens ns (l.map f) = (cutLens ns l).map (List.map f)
  | [], _ => rfl
  | n :: ns, l => by
    have ih := cutLens_map f ns (l.drop n)
    rw [List.map_drop] at ih
    simp only [cutLens, List.map_cons, List.map_take, ih]

/-! ## The values a rep's frames absorb, by count -/

theorem vals_dom (b : ByteArray) (fs : List Frame) : vals (.dom b :: fs) = vals fs := rfl
theorem vals_label (b : ByteArray) (fs : List Frame) : vals (.label b :: fs) = vals fs := rfl
theorem vals_one (v : F128) (fs : List Frame) : vals (.one v :: fs) = v :: vals fs := rfl
theorem vals_vec (vs : List F128) (fs : List Frame) : vals (.vec vs :: fs) = vs ++ vals fs := rfl
theorem vals_bytes (b : ByteArray) (fs : List Frame) : vals (.bytes b :: fs) = vals fs := rfl
theorem vals_sq (k : UInt8) (n : ℕ) (fs : List Frame) : vals (.sq k n :: fs) = vals fs := rfl
theorem vals_capFr (cap : Array ByteArray) (fs : List Frame) : vals (capFr cap :: fs) = vals fs := rfl
theorem vals_nilR : vals [] = [] := rfl

theorem vals_flatMap {α : Type} (F : α → List Frame) :
    ∀ l : List α, vals (l.flatMap F) = l.flatMap fun a => vals (F a)
  | [] => rfl
  | a :: l => by rw [List.flatMap_cons, List.flatMap_cons, vals_append, vals_flatMap F l]

theorem vals_map_one : ∀ l : List F128, vals (l.map Frame.one) = l
  | [] => rfl
  | v :: l => by rw [List.map_cons, vals_one, vals_map_one l]

theorem vals_kFr_length (m : F256 × F256) : (vals (kFr m)).length = 4 := rfl

theorem vals_chFr_length (p : LigeritoProof) : ∀ (msg k : ℕ), (vals (chFr p msg k)).length = 4 * k
  | _, 0 => rfl
  | msg, k + 1 => by
    simp only [chFr, vals_sq, vals_append, List.length_append, vals_kFr_length, vals_chFr_length p (msg + 1) k]
    ring

theorem vals_introFr_length (p : LigeritoProof) (dim ood msg cbn : ℕ) :
    (vals (introFr p dim ood msg cbn)).length = 5 := by
  simp only [introFr, vals_sq, vals_one, vals_append, vals_bytes, vals_nilR, List.length_cons, List.length_append,
    vals_kFr_length, List.length_nil]

theorem vals_queryFr (p : LigeritoProof) (lv : Flock.Level) (qn cn : ℕ) : vals (queryFr p lv qn cn) = [] := rfl

theorem vals_levelFr_length (sch : Flock.Schedule) (p : LigeritoProof) (i : ℕ) :
    (vals (levelFr sch p i)).length =
      16 + if i + 1 < sch.levels.size - 1 then 14 else p.finalProof.yr.size := by
  unfold levelFr
  rw [vals_append, List.length_append, vals_chFr_length]
  by_cases hc : i + 1 < sch.levels.size - 1
  · rw [if_pos hc, if_pos hc]
    simp only [vals_capFr, vals_append, vals_bytes, vals_sq, vals_nilR, vals_queryFr, List.length_append,
      vals_introFr_length, vals_kFr_length, List.length_nil]
    all_goals omega
  · rw [if_neg hc, if_neg hc]
    simp only [vals_append, vals_bytes, vals_sq, vals_nilR, vals_queryFr, vals_map_one, List.length_append,
      Array.length_toList, List.length_nil]
    all_goals omega

theorem vals_ligFrames_length (sch : Flock.Schedule) (T : F128) (cap : Array ByteArray) (p : LigeritoProof)
    (hL : 2 ≤ sch.levels.size) :
    (vals (ligFrames sch T cap p)).length = 4 * sch.k0 + 36 + 30 * (sch.levels.size - 2) + p.finalProof.yr.size := by
  have hlv : (vals ((List.range' 0 (sch.levels.size - 1)).flatMap (levelFr sch p))).length =
      30 * (sch.levels.size - 2) + 16 + p.finalProof.yr.size := by
    rw [vals_flatMap, show sch.levels.size - 1 = sch.levels.size - 2 + 1 by omega, List.range'_concat,
      List.flatMap_append, List.length_append, List.flatMap_cons, List.flatMap_nil, List.append_nil,
      length_flatMap_eq (k := 30) (l := List.range' 0 (sch.levels.size - 2)) (fun i hi => by
        have hi' : i + 1 < sch.levels.size - 1 := by rw [List.mem_range'_1] at hi; omega
        show (vals (levelFr sch p i)).length = 30
        rw [vals_levelFr_length, if_pos hi']), List.length_range', vals_levelFr_length]
    split <;> omega
  simp only [ligFrames, vals_label, vals_one, vals_capFr, vals_sq, vals_bytes, vals_append, vals_nilR, vals_queryFr,
    List.length_cons, List.length_append, vals_kFr_length, vals_chFr_length, vals_introFr_length, List.length_nil,
    hlv]
  all_goals omega

theorem vals_zcFrames_length (m : ℕ) (digest cap : ByteArray) (z : ZerocheckProof) :
    (vals (zcFrames m digest cap z)).length = z.round1ab.size + z.round1c.size + 2 * (m - 6) + 2 := by
  have h : (vals ((List.range' 0 (m - 6)).flatMap fun i =>
      [Frame.one (z.rounds[i]!).1, .one (z.rounds[i]!).2, .sq 1 1])).length = (m - 6) * 2 := by
    rw [vals_flatMap, length_flatMap_eq (k := 2) ?_, List.length_range']
    intro _ _
    rfl
  simp only [zcFrames, vals_label, vals_bytes, vals_sq, vals_vec, vals_one, vals_append, vals_nilR,
    List.length_append, List.length_cons, List.length_nil, Array.length_toList, h]
  all_goals omega

theorem vals_lcFrames_length (k : ℕ) (l : LincheckProof) :
    (vals (lcFrames k l)).length = 2 * (k - 6) + l.zPartial.size := by
  have h : (vals ((List.range' 0 (k - 6)).flatMap fun j =>
      [Frame.one (l.rounds[j]!).1, .one (l.rounds[j]!).2, .sq 1 1])).length = (k - 6) * 2 := by
    rw [vals_flatMap, length_flatMap_eq (k := 2) ?_, List.length_range']
    intro _ _
    rfl
  simp only [lcFrames, vals_label, vals_sq, vals_vec, vals_append, vals_nilR, List.length_append,
    List.length_nil, Array.length_toList, h]
  all_goals omega

theorem vals_rsFrames_length {sh : Shape} {p : Proof} (hp : ProofSizes sh p) :
    (vals (rsFrames sh.claims p.ringSwitches)).length = 128 * sh.claims := by
  simp only [rsFrames, vals_label, vals_append, vals_sq, vals_nilR, List.append_nil]
  rw [vals_flatMap, length_flatMap_eq (k := 128) ?_, List.length_range', Nat.mul_comm]
  intro k hk
  rw [List.mem_range'_1] at hk
  simp only [vals_label, vals_vec, vals_sq, vals_nilR, List.append_nil, Array.length_toList, rs_size hp,
    if_pos (show k < sh.claims by omega)]

/-- **How many values a rep's frames absorb**, at a proof of the rep's shape. -/
theorem vals_repFrames_length {st : Setup} {sch : Flock.Schedule} {e : ℕ} {p : Proof}
    (hp : ProofSizes (shapeOf st sch e) p) (hk : st.fold.kLog = st.kLog) (hL : 2 ≤ sch.levels.size)
    (d : ByteArray) (T : F128) :
    (vals (repFrames st sch d (2 + e) p T)).length =
      64 + 64 + 2 * (st.m - 6) + 2 + 2 * (st.kLog - 6) + 64 + 128 * (2 + e) +
        (4 * sch.k0 + 36 + 30 * (sch.levels.size - 2) + 2 ^ (sch.ext + 1)) := by
  have h1 : p.zerocheck.round1ab.size = 64 := hp.r1ab
  have h2 : p.zerocheck.round1c.size = 64 := hp.r1c
  have h3 : p.lincheck.zPartial.size = 64 := hp.zPartial
  have h4 : p.ligerito.finalProof.yr.size = 2 ^ (sch.ext + 1) := hp.yr
  have h5 : (vals (rsFrames (2 + e) p.ringSwitches)).length = 128 * (2 + e) := vals_rsFrames_length hp
  simp only [repFrames, vals_dom, vals_append, List.length_append, vals_zcFrames_length, vals_lcFrames_length,
    vals_ligFrames_length sch T _ _ hL, h5, h1, h2, h3, h4, hk]
  all_goals omega

/-! ## Every absorbed value is some proof's -/

/-- The absorbed value slots of a proof the size of `q`: `T`, the zerocheck's (`round1ab`, `round1c`, its rounds,
`finalA`, `finalB`), the lincheck's (its rounds and `zPartial`), the ring switches', and Ligerito's (its messages, its
out-of-domain values and `yr`). -/
abbrev Slots (q : Proof) : Type :=
  F128 × (Fin q.zerocheck.round1ab.size → F128) × (Fin q.zerocheck.round1c.size → F128) ×
    (Fin q.zerocheck.rounds.size → F128 × F128) × F128 × F128 ×
    (Fin q.lincheck.rounds.size → F128 × F128) × (Fin q.lincheck.zPartial.size → F128) ×
    (Fin q.ringSwitches.size → Fin 128 → F128) ×
    (Fin q.ligerito.messages.size → (F128 × F128) × (F128 × F128)) ×
    (Fin q.ligerito.oodValues.size → F128) × (Fin q.ligerito.finalProof.yr.size → F128)

/-- `q` with its absorbed values at the slots `d`, and the slots' `T`. -/
def fill (q : Proof) (d : Slots q) : Proof × F128 :=
  ({ q with
      zerocheck := { q.zerocheck with
        round1ab := Array.ofFn d.2.1, round1c := Array.ofFn d.2.2.1, rounds := Array.ofFn d.2.2.2.1,
        finalA := d.2.2.2.2.1, finalB := d.2.2.2.2.2.1 }
      lincheck := ⟨Array.ofFn d.2.2.2.2.2.2.1, Array.ofFn d.2.2.2.2.2.2.2.1⟩
      ringSwitches := Array.ofFn fun k => Array.ofFn (d.2.2.2.2.2.2.2.2.1 k)
      ligerito := { q.ligerito with
        messages := Array.ofFn fun i =>
          (⟨(d.2.2.2.2.2.2.2.2.2.1 i).1.1, (d.2.2.2.2.2.2.2.2.2.1 i).1.2⟩,
            ⟨(d.2.2.2.2.2.2.2.2.2.1 i).2.1, (d.2.2.2.2.2.2.2.2.2.1 i).2.2⟩)
        oodValues := Array.ofFn d.2.2.2.2.2.2.2.2.2.2.1
        finalProof := { q.ligerito.finalProof with yr := Array.ofFn d.2.2.2.2.2.2.2.2.2.2.2 } } },
    d.1)

theorem fill_sizes {sh : Shape} {q : Proof} (hq : ProofSizes sh q) (d : Slots q) : ProofSizes sh (fill q d).1 where
  r1ab := (Array.size_ofFn ..).trans hq.r1ab
  r1c := (Array.size_ofFn ..).trans hq.r1c
  zcRounds := (Array.size_ofFn ..).trans hq.zcRounds
  lcRounds := (Array.size_ofFn ..).trans hq.lcRounds
  zPartial := (Array.size_ofFn ..).trans hq.zPartial
  rs := (Array.size_ofFn ..).trans hq.rs
  rsEach := fun v hv => by
    obtain ⟨k, rfl⟩ := Array.mem_ofFn.1 hv
    exact Array.size_ofFn ..
  cap := hq.cap
  capEach := hq.capEach
  rcaps := hq.rcaps
  rcapsEach := hq.rcapsEach
  yr := (Array.size_ofFn ..).trans hq.yr
  messages := (Array.size_ofFn ..).trans hq.messages
  queryNonces := hq.queryNonces
  oodValues := (Array.size_ofFn ..).trans hq.oodValues
  claimNonces := hq.claimNonces
  consistencyNonces := hq.consistencyNonces

theorem ofFn_inj {α : Type} {n : ℕ} {f g : Fin n → α} (h : Array.ofFn f = Array.ofFn g) : f = g :=
  List.ofFn_injective (by simpa only [Array.toList_ofFn] using congrArg Array.toList h)

/-- **The slots are fixed by the values the frames absorb**, at a setup that folds at its own `k_log`. -/
theorem fill_inj {st : Setup} {sch : Flock.Schedule} {d₀ : ByteArray} {e : ℕ} {q : Proof}
    (hq : ProofSizes (shapeOf st sch e) q) (hk : st.fold.kLog = st.kLog) (hL : 2 ≤ sch.levels.size)
    {a b : Slots q}
    (h : vals (repFrames st sch d₀ (2 + e) (fill q a).1 (fill q a).2) =
      vals (repFrames st sch d₀ (2 + e) (fill q b).1 (fill q b).2)) : a = b := by
  have ha := fill_sizes hq a
  have hb := fill_sizes hq b
  have hF := eq_of_skel_vals (repFrames_skel ha hb rfl rfl rfl rfl rfl) h
  obtain ⟨-, e1, e2, e3, e4, e5, e6, e7, hlig⟩ := repFrames_fields ha hb hk hF
  obtain ⟨-, e8, -, e9, -, -, e10⟩ := ligFrames_fields ha hb hL hlig
  have hT : (fill q a).2 = (fill q b).2 := by
    unfold ligFrames at hlig
    exact Frame.one.inj (List.cons.inj (List.cons.inj (List.append_inj hlig rfl).1).2).1
  obtain ⟨T, ab, c1, zr, fA, fB, lr, zp, rs, ms, ood, yr⟩ := a
  obtain ⟨T', ab', c1', zr', fA', fB', lr', zp', rs', ms', ood', yr'⟩ := b
  simp only [fill, LincheckProof.mk.injEq] at e1 e2 e3 e4 e5 e6 e7 e8 e9 e10 hT
  have h7 := ofFn_inj e7
  have h8 := ofFn_inj e8
  simp only [Prod.mk.injEq]
  refine ⟨hT, ofFn_inj e1, ofFn_inj e2, ofFn_inj e3, e4, e5, ofFn_inj e6.1, ofFn_inj e6.2,
    funext fun k => ofFn_inj (congrFun h7 k), funext fun i => ?_, ofFn_inj e9, ofFn_inj e10⟩
  have hi := congrFun h8 i
  simp only [Prod.mk.injEq, F256.mk.injEq] at hi
  exact Prod.ext (Prod.ext hi.1.1 hi.1.2) (Prod.ext hi.2.1 hi.2.2)

theorem pow_count (N m k e k0 L x : ℕ) :
    N * (N ^ 64 * (N ^ 64 * ((N * N) ^ (m - 6) * (N * (N * ((N * N) ^ (k - 6) * (N ^ 64 *
      ((N ^ 128) ^ (2 + e) * ((N * N * (N * N)) ^ (k0 + 8 + 7 * L) * (N ^ (1 + (2 * L + 2)) *
        N ^ 2 ^ (x + 1))))))))))) =
    N ^ (64 + 64 + 2 * (m - 6) + 2 + 2 * (k - 6) + 64 + 128 * (2 + e) + (4 * k0 + 36 + 30 * L + 2 ^ (x + 1))) := by
  ring

/-- **As many slots as absorbed values**, counted as `F128`s. -/
theorem card_slots {st : Setup} {sch : Flock.Schedule} {e : ℕ} {q : Proof} (hq : ProofSizes (shapeOf st sch e) q)
    (hL : 2 ≤ sch.levels.size) :
    Fintype.card (Slots q) = Fintype.card F128 ^ (64 + 64 + 2 * (st.m - 6) + 2 + 2 * (st.kLog - 6) + 64 +
      128 * (2 + e) + (4 * sch.k0 + 36 + 30 * (sch.levels.size - 2) + 2 ^ (sch.ext + 1))) := by
  simp only [Fintype.card_prod, Fintype.card_fun, Fintype.card_fin, hq.r1ab, hq.r1c, hq.zcRounds, hq.lcRounds,
    hq.zPartial, hq.rs, hq.messages, hq.oodValues, hq.yr]
  have h2 : 2 * (sch.levels.size - 1) = 2 * (sch.levels.size - 2) + 2 := by omega
  have h7 : sch.levels.size - 1 - 1 = sch.levels.size - 2 := by omega
  simp only [shapeOf, h7, h2]
  generalize Fintype.card F128 = N
  exact pow_count N st.m st.kLog e sch.k0 (sch.levels.size - 2) sch.ext

/-- The absorbed values at the slots `d`, as a vector of their count. -/
def slotVals (st : Setup) (sch : Flock.Schedule) (d₀ : ByteArray) (e : ℕ) (q : Proof) (n : ℕ)
    (hn : ∀ d : Slots q, (vals (repFrames st sch d₀ (2 + e) (fill q d).1 (fill q d).2)).length = n) (d : Slots q) :
    List.Vector F128 n :=
  ⟨vals (repFrames st sch d₀ (2 + e) (fill q d).1 (fill q d).2), hn d⟩

theorem slotVals_val {st : Setup} {sch : Flock.Schedule} {d₀ : ByteArray} {e : ℕ} {q : Proof} {n : ℕ}
    (hn : ∀ d : Slots q, (vals (repFrames st sch d₀ (2 + e) (fill q d).1 (fill q d).2)).length = n) (d : Slots q) :
    (slotVals st sch d₀ e q n hn d).val = vals (repFrames st sch d₀ (2 + e) (fill q d).1 (fill q d).2) := rfl

/-- **Every value list of the frames' count is some proof's**: a proof of `q`'s shape with `q`'s cap, recursive caps
and nonces, and a `T`, whose rep frames absorb exactly `W`. -/
theorem realize {st : Setup} {sch : Flock.Schedule} {d₀ : ByteArray} {e : ℕ} {q : Proof}
    (hq : ProofSizes (shapeOf st sch e) q) (hk : st.fold.kLog = st.kLog) (hL : 2 ≤ sch.levels.size) (T₀ : F128)
    (W : List F128) (hW : W.length = (vals (repFrames st sch d₀ (2 + e) q T₀)).length) :
    ∃ (p : Proof) (T : F128), ProofSizes (shapeOf st sch e) p ∧ p.cap = q.cap ∧
      p.ligerito.recursiveCaps = q.ligerito.recursiveCaps ∧ p.ligerito.queryNonces = q.ligerito.queryNonces ∧
      p.ligerito.claimNonces = q.ligerito.claimNonces ∧
      p.ligerito.consistencyNonces = q.ligerito.consistencyNonces ∧
      vals (repFrames st sch d₀ (2 + e) p T) = W := by
  have hlen : ∀ d : Slots q, (vals (repFrames st sch d₀ (2 + e) (fill q d).1 (fill q d).2)).length = W.length :=
    fun d => by rw [vals_repFrames_length (fill_sizes hq d) hk hL, hW, vals_repFrames_length hq hk hL]
  have hf : Function.Injective (slotVals st sch d₀ e q W.length hlen) := fun a b h => by
    have h' := congrArg Subtype.val h
    rw [slotVals_val, slotVals_val] at h'
    exact fill_inj hq hk hL h'
  have hc : Fintype.card (Slots q) = Fintype.card (List.Vector F128 W.length) := by
    rw [card_vector, hW, vals_repFrames_length hq hk hL]
    exact card_slots hq hL
  obtain ⟨d, hd⟩ :=
    ((Fintype.bijective_iff_injective_and_card (slotVals st sch d₀ e q W.length hlen)).2 ⟨hf, hc⟩).2 ⟨W, rfl⟩
  have hv := congrArg Subtype.val hd
  rw [slotVals_val] at hv
  have hs := fill_sizes hq d
  have h1 : (fill q d).1.cap = q.cap := rfl
  have h2 : (fill q d).1.ligerito.recursiveCaps = q.ligerito.recursiveCaps := rfl
  have h3 : (fill q d).1.ligerito.queryNonces = q.ligerito.queryNonces := rfl
  have h4 : (fill q d).1.ligerito.claimNonces = q.ligerito.claimNonces := rfl
  have h5 : (fill q d).1.ligerito.consistencyNonces = q.ligerito.consistencyNonces := rfl
  exact ⟨(fill q d).1, (fill q d).2, hs, h1, h2, h3, h4, h5, hv⟩

/-! ## The recursive caps change bytes frames only -/

/-- A frame with its bytes blanked. -/
def blank : Frame → Frame
  | .dom b => .dom b
  | .label b => .label b
  | .one v => .one v
  | .vec vs => .vec vs
  | .bytes _ => .bytes ByteArray.empty
  | .sq k n => .sq k n

theorem closes_blank (f : Frame) : closes (blank f) = closes f := by
  cases f <;> rfl

theorem vals_blank : ∀ fs : List Frame, vals (fs.map blank) = vals fs
  | [] => rfl
  | .dom _ :: fs => vals_blank fs
  | .label _ :: fs => vals_blank fs
  | .one v :: fs => congrArg (v :: ·) (vals_blank fs)
  | .vec vs :: fs => congrArg (vs ++ ·) (vals_blank fs)
  | .bytes _ :: fs => vals_blank fs
  | .sq _ _ :: fs => vals_blank fs

theorem chunks_blank : ∀ fs : List Frame, (chunks (fs.map blank)).1 = (chunks fs).1.map (List.map blank)
  | [] => rfl
  | f :: fs => by
    have ih := chunks_blank fs
    rw [List.map_cons]
    simp only [chunks, closes_blank, ih]
    cases closes f <;> rcases (chunks fs).1 with _ | ⟨c, cs⟩ <;> simp

theorem levelFr_blank (sch : Flock.Schedule) (l : LigeritoProof) (R : Array (Array ByteArray)) (i : ℕ) :
    (levelFr sch { l with recursiveCaps := R } i).map blank = (levelFr sch l i).map blank := by
  unfold levelFr
  rw [chFr_congr (p := { l with recursiveCaps := R }) (p' := l) rfl]
  by_cases hc : i + 1 < sch.levels.size - 1
  · rw [if_pos hc, if_pos hc]
    simp [blank, capFr, introFr, queryFr]
  · rw [if_neg hc, if_neg hc]
    simp [blank, capFr, introFr, queryFr]

theorem ligFrames_blank (sch : Flock.Schedule) (T : F128) (cap : Array ByteArray) (l : LigeritoProof)
    (R : Array (Array ByteArray)) :
    (ligFrames sch T cap { l with recursiveCaps := R }).map blank = (ligFrames sch T cap l).map blank := by
  have hlv : ((List.range' 0 (sch.levels.size - 1)).flatMap (levelFr sch { l with recursiveCaps := R })).map blank =
      ((List.range' 0 (sch.levels.size - 1)).flatMap (levelFr sch l)).map blank := by
    simp only [List.map_flatMap, levelFr_blank]
  unfold ligFrames
  rw [chFr_congr (p := { l with recursiveCaps := R }) (p' := l) rfl]
  simp only [List.map_append, List.map_cons, hlv]
  simp [blank, capFr, introFr, queryFr]

/-- `p` with the recursive caps `R`. -/
def withCaps (p : Proof) (R : Array (Array ByteArray)) : Proof :=
  { p with ligerito := { p.ligerito with recursiveCaps := R } }

theorem repFrames_withCaps (st : Setup) (sch : Flock.Schedule) (d : ByteArray) (n : ℕ) (p : Proof)
    (R : Array (Array ByteArray)) (T : F128) :
    (repFrames st sch d n (withCaps p R) T).map blank = (repFrames st sch d n p T).map blank := by
  unfold repFrames withCaps
  simp only [List.map_cons, List.map_append, ligFrames_blank]

theorem chunksOf_withCaps (c : RepCtx) (p : Proof) (R : Array (Array ByteArray)) (T : F128) :
    (chunksOf c (withCaps p R) T).map (List.map blank) = (chunksOf c p T).map (List.map blank) := by
  unfold chunksOf
  rw [← chunks_blank, ← chunks_blank, repFrames_withCaps]

theorem withCaps_sizes {sh : Shape} {p : Proof} (hp : ProofSizes sh p) {R : Array (Array ByteArray)}
    (h : R.toList.map (fun a => a.toList.map ByteArray.size) =
      p.ligerito.recursiveCaps.toList.map (fun a => a.toList.map ByteArray.size)) :
    ProofSizes sh (withCaps p R) := by
  have hs : R.size = p.ligerito.recursiveCaps.size := by
    simpa using congrArg List.length h
  have hi : ∀ (i : ℕ) (h1 : i < R.size) (h2 : i < p.ligerito.recursiveCaps.size),
      R[i].toList.map ByteArray.size = p.ligerito.recursiveCaps[i].toList.map ByteArray.size := fun i h1 h2 => by
    have := congrArg (·[i]?) h
    simpa [h1, h2] using this
  exact {
    r1ab := hp.r1ab, r1c := hp.r1c, zcRounds := hp.zcRounds, lcRounds := hp.lcRounds, zPartial := hp.zPartial,
    rs := hp.rs, rsEach := hp.rsEach, cap := hp.cap, capEach := hp.capEach
    rcaps := hs.trans hp.rcaps
    rcapsEach := fun i h1 => by
      have h2 : i < p.ligerito.recursiveCaps.size := hs ▸ h1
      have e := hi i h1 h2
      obtain ⟨hz, he⟩ := hp.rcapsEach i h2
      refine ⟨?_, fun x hx => ?_⟩
      · have := congrArg List.length e
        simp only [List.length_map, Array.length_toList] at this
        exact this.trans hz
      · have hm : x.size ∈ p.ligerito.recursiveCaps[i].toList.map ByteArray.size :=
          e ▸ List.mem_map_of_mem (Array.mem_toList_iff.2 hx)
        obtain ⟨y, hy, hxy⟩ := List.mem_map.1 hm
        exact hxy ▸ he y (Array.mem_toList_iff.1 hy)
    yr := hp.yr, messages := hp.messages, queryNonces := hp.queryNonces, oodValues := hp.oodValues,
    claimNonces := hp.claimNonces, consistencyNonces := hp.consistencyNonces }

/-! ## The rows' tops and value rows, by size -/

theorem size_nodeOf (r : ByteArray) : (nodeOf r).size = r.size - 64 := by
  show (r.data.toList.take (r.data.size - 64)).toArray.size = r.data.size - 64
  rw [List.size_toArray, List.length_take, Array.length_toList]
  omega

theorem topGroups_sizes : ∀ (sched : List RecVStar.SchedRound) (rows rows' : List (List ByteArray)),
    rows.map (·.map ByteArray.size) = rows'.map (·.map ByteArray.size) →
      (topGroups sched rows).map (·.map ByteArray.size) = (topGroups sched rows').map (·.map ByteArray.size)
  | [], _, _, _ => by simp [topGroups]
  | _ :: _, [], [], _ => by simp [topGroups]
  | _ :: _, [], _ :: _, h => by simp at h
  | _ :: _, _ :: _, [], h => by simp at h
  | s :: ss, r :: rs, r' :: rs', h => by
    rw [List.map_cons, List.map_cons, List.cons.injEq] at h
    have ih := topGroups_sizes ss rs rs' h.2
    unfold topGroups at ih ⊢
    rw [List.zip_cons_cons, List.zip_cons_cons, List.flatMap_cons, List.flatMap_cons, List.map_append,
      List.map_append, ih]
    congr 1
    have e : ∀ x : List ByteArray, ((cutLens s.2.1.toList x).map (List.map nodeOf)).map (·.map ByteArray.size) =
        (cutLens s.2.1.toList (x.map ByteArray.size)).map (List.map (· - 64)) := fun x => by
      rw [cutLens_map, List.map_map, List.map_map]
      refine List.map_congr_left fun y _ => ?_
      simp only [Function.comp_apply, List.map_map]
      exact List.map_congr_left fun z _ => size_nodeOf z
    rw [e, e, h.1]

/-- **A round's values lift to rows of its rows' sizes**: some values of the same count, read off the leading bytes of
the new value row. -/
theorem rowsAt_lift {s : RecVStar.SchedRound} {rs rs' : List ByteArray} {vs' : List F128} (h : RowsAt s rs' vs')
    (hs : rs.map ByteArray.size = rs'.map ByteArray.size) : ∃ vs, vs.length = vs'.length ∧ RowsAt s rs vs := by
  obtain ⟨h1, ⟨pad', hpad', h2⟩, h3⟩ := h
  have hl : rs.length = rs'.length := by simpa using congrArg List.length hs
  have hd : (rs.drop s.2.1.toList.sum).map ByteArray.size = (rs'.drop s.2.1.toList.sum).map ByteArray.size := by
    rw [List.map_drop, List.map_drop, hs]
  cases hb : s.2.2.2
  · rw [h3 hb]
    refine ⟨[], rfl, by rw [hl, h1], ⟨[], rfl, ?_⟩, fun _ => rfl⟩
    simp only [hb, Bool.false_eq_true, ↓reduceIte] at h1 ⊢
    exact List.drop_of_length_le (by rw [hl]; omega)
  · rw [hb, if_pos rfl] at h2
    rw [h2] at hd
    obtain ⟨r, hr⟩ : ∃ r, rs.drop s.2.1.toList.sum = [r] := by
      rcases hx : rs.drop s.2.1.toList.sum with _ | ⟨r, _ | ⟨_, _⟩⟩
      · rw [hx] at hd; simp at hd
      · exact ⟨r, rfl⟩
      · rw [hx] at hd; simp at hd
    rw [hr] at hd
    have hrs : r.data.toList.length = 16 * vs'.length + pad'.length := by
      simp only [List.map_cons, List.map_nil, List.cons.injEq, and_true] at hd
      rw [Array.length_toList]
      show r.size = _
      rw [hd]
      show (colBytes vs' ++ pad').toArray.size = _
      rw [List.size_toArray, List.length_append, FlockSoundness.Refine.Frame.colBytes_length]
    obtain ⟨vs, hvl, hv⟩ := colBytes_surj vs'.length (r.data.toList.take (16 * vs'.length))
      (by rw [List.length_take]; omega)
    refine ⟨vs, hvl, by rw [hl, h1], ⟨r.data.toList.drop (16 * vs'.length), ?_, ?_⟩, fun h => by
      rw [hb] at h; cases h⟩
    · rw [FlockSoundness.Refine.Frame.colBytes_length] at hpad'
      rw [List.length_drop, FlockSoundness.Refine.Frame.colBytes_length, hvl]
      omega
    · rw [hb, if_pos rfl, hr, hv, List.take_append_drop, Array.toArray_toList]

end FlockDeploy

namespace RecDeploy.RestRead

open Flock
open FlockDeploy (RepCtx FitsRep vals skel topGroups RowsAt chunksOf withCaps blank)
open FlockSoundness.Discharge.Exec (chunks chunks_flatten repFrames repFrames_chunks_snd)

/-- **A fit lifts to every rows of its rows' byte sizes** (`FitsLift`): the new rows' tops as the recursive caps, and
the values each new value row leads with (`rowsAt_lift`), which some proof of the fit's shape absorbs (`realize`). -/
theorem fitsLift : FitsLift := by
  intro c hk rows rows' p' T' h hsz
  have hlen : rows.length = rows'.length := by simpa using congrArg List.length hsz
  have hrow : ∀ j, (rows.getD j []).map ByteArray.size = (rows'.getD j []).map ByteArray.size := fun j => by
    rw [← FlockDeploy.getD_map_nil (List.map ByteArray.size) rfl,
      ← FlockDeploy.getD_map_nil (List.map ByteArray.size) rfl, hsz]
  have hround : ∀ j, ∃ vs : List F128, vs.length = (vals ((chunksOf c p' T').getD j [])).length ∧
      (j < rows.length → RowsAt (c.sched.getD j default) (rows.getD j []) vs) := fun j => by
    by_cases hj : j < rows.length
    · obtain ⟨vs, hl, hr⟩ := FlockDeploy.rowsAt_lift (h.elems j (hlen ▸ hj)) (hrow j)
      exact ⟨vs, hl, fun _ => hr⟩
    · exact ⟨_, rfl, fun h => absurd h hj⟩
  choose g hg using hround
  obtain ⟨e, hn, hp'⟩ := h.sizes
  have hfl : ∀ (p : Flock.Proof) (T : F128),
      vals (repFrames c.st c.sch c.dom c.n p T) = ((chunksOf c p T).map vals).flatten := fun p T => by
    rw [← FlockDeploy.vals_flatten]
    unfold chunksOf
    conv_lhs => rw [← chunks_flatten (repFrames c.st c.sch c.dom c.n p T), repFrames_chunks_snd, List.append_nil]
  have hgl : ((List.range (chunksOf c p' T').length).map g).map List.length =
      ((chunksOf c p' T').map vals).map List.length := by
    refine List.ext_getElem (by simp) fun j h1 h2 => ?_
    simp only [List.length_map, List.length_range] at h1
    simp only [List.getElem_map, List.getElem_range]
    rw [(hg j).1, List.getD_eq_getElem _ _ h1]
  have hW : ((List.range (chunksOf c p' T').length).map g).flatten.length =
      (vals (repFrames c.st c.sch c.dom (2 + e) p' T')).length := by
    rw [← hn, hfl, List.length_flatten, List.length_flatten, hgl]
  obtain ⟨p₀, T, hp₀, hcap, hrc, hnq, hnc, hns, hv⟩ := FlockDeploy.realize hp' hk h.levels T' _ hW
  rw [← hn] at hv
  have hsk : (chunksOf c p₀ T).map (List.map skel) = (chunksOf c p' T').map (List.map skel) := by
    unfold chunksOf
    rw [← FlockDeploy.chunks_skel, ← FlockDeploy.chunks_skel, FlockDeploy.repFrames_skel hp₀ hp' hcap hrc hnc hnq hns]
  have hsj : ∀ j, ((chunksOf c p₀ T).getD j []).map skel = ((chunksOf c p' T').getD j []).map skel := fun j => by
    rw [← FlockDeploy.getD_map_nil (List.map skel) rfl, ← FlockDeploy.getD_map_nil (List.map skel) rfl, hsk]
  have hl₀ : (chunksOf c p₀ T).length = (chunksOf c p' T').length := by
    simpa using congrArg List.length hsk
  have hv₀ : (chunksOf c p₀ T).map vals = (List.range (chunksOf c p' T').length).map g := by
    refine FlockDeploy.eq_of_flatten ?_ ?_
    · rw [hgl]
      refine List.ext_getElem (by simp [hl₀]) fun j h1 h2 => ?_
      simp only [List.length_map] at h1 h2
      simp only [List.getElem_map]
      rw [← List.getD_eq_getElem _ [] h1, ← List.getD_eq_getElem _ [] h2, ← FlockDeploy.vals_length_skel,
        hsj, FlockDeploy.vals_length_skel]
    · rw [← hfl, hv]
  obtain ⟨R, hR⟩ : ∃ R : Array (Array ByteArray), R.toList.map Array.toList = topGroups c.sched rows :=
    ⟨((topGroups c.sched rows).map List.toArray).toArray, by simp [List.map_map, Function.comp_def]⟩
  have hb := FlockDeploy.chunksOf_withCaps c p₀ R T
  have hlb : (chunksOf c (withCaps p₀ R) T).length = (chunksOf c p₀ T).length := by
    simpa using congrArg List.length hb
  have hvb : ∀ j, vals ((chunksOf c (withCaps p₀ R) T).getD j []) = vals ((chunksOf c p₀ T).getD j []) := fun j => by
    rw [← FlockDeploy.vals_blank ((chunksOf c (withCaps p₀ R) T).getD j []),
      ← FlockDeploy.vals_blank ((chunksOf c p₀ T).getD j []),
      ← FlockDeploy.getD_map_nil (List.map blank) rfl, ← FlockDeploy.getD_map_nil (List.map blank) rfl, hb]
  refine ⟨(withCaps p₀ R, T), ⟨?_, h.sched.trans hlen.symm, ?_, hR, ?_, ⟨e, hn, ?_⟩, h.levels, ?_⟩⟩
  · show (chunksOf c (withCaps p₀ R) T).length = rows.length
    rw [hlb, hl₀, h.rounds, hlen]
  · show p₀.cap.toList = c.cap0
    rw [hcap]
    exact h.cap
  · show (∀ n ∈ p₀.ligerito.queryNonces, n = 0) ∧ (∀ n ∈ p₀.ligerito.claimNonces, n = 0) ∧
      ∀ n ∈ p₀.ligerito.consistencyNonces, n = 0
    rw [hnq, hnc, hns]
    exact h.nonces
  · refine FlockDeploy.withCaps_sizes hp₀ ?_
    have e1 : R.toList.map (fun a => a.toList.map ByteArray.size) =
        (topGroups c.sched rows).map (·.map ByteArray.size) := by
      rw [← hR, List.map_map]
      rfl
    have e2 : p'.ligerito.recursiveCaps.toList.map (fun a => a.toList.map ByteArray.size) =
        (topGroups c.sched rows').map (·.map ByteArray.size) := by
      rw [← h.tops, List.map_map]
      rfl
    rw [e1, hrc, e2]
    exact FlockDeploy.topGroups_sizes c.sched rows rows' hsz
  · intro j hj
    have hj' : j < (chunksOf c p' T').length := by rw [h.rounds, ← hlen]; exact hj
    show RowsAt (c.sched.getD j default) (rows.getD j []) (vals ((chunksOf c (withCaps p₀ R) T).getD j []))
    rw [hvb j, ← FlockDeploy.getD_map_nil vals rfl, hv₀,
      List.getD_eq_getElem ((List.range (chunksOf c p' T').length).map g) [] (by simpa using hj'),
      List.getElem_map, List.getElem_range]
    exact (hg j).2 hj

end RecDeploy.RestRead
