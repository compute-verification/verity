import Proofs.Flock.Claims.Recursion.NextRSession
import Proofs.Flock.Claims.DeployTables

/-!
# The honest rows fit a message

`sessionRun_decides` takes that the honest prover's rows fit a message (`RowsFit`): round 0's (`capRows`) and every
rep's round's (`repRows`). They do, at every statement that sets up, so the honest session decides with no row premise
(`sessionRun_decides_fit`, which still takes the files' `finalC` at the reader's `z`, `FilesFinalC`, which
`NextRCoins` proves at coins the replies are):

* **Tables** (`setupTables_le`, `setup_tables`): `Stmt.setupTables` refuses more than `Tags.MAX_TABLES = 64` tables, and
  its spec names one table per statement, so `sessionRun_decides`' `sts.size = spec.tables.size` is derived too.
* **Schedules** (`fast100_bound`, by `decide`): every `fast100` schedule is of `m ≤ 35`, at most 7 levels, every cap of
  at most `2^7` nodes, `ext ≤ 28` and `k0 ≤ 6`.
* **Values** (`vals_repFrames_le`, `vals_chunk_le`): a rep's frames absorb the 64-value vectors and the `2^(ext+1)`
  residual `ProofSizes` fixes, `128` values a claim, and a bounded number of single values, so at an honest rep of a
  `TableOk` table with at most 2,050 claims every round's value row is shorter than `2^44` bytes.
* **Rows** (`honestRows_rows`, `repRows_fit`, `capRows_fit`): each row is a cap node's row (`nodeRow`), at most 128 bytes
  (a node is at most 64, `digestLen` of a scheme in `MerkleScheme.all`), or a round's value row; a round has at most its rep's recursive caps' nodes and one
  value row, and round 0 at most 64 tables' level-0 caps.

The bounds' hypotheses hold together at the concrete honest rep `honest₀` (`rowsFit₀`).
-/

namespace FlockDeploy

open Flock

section Walk

open FlockSoundness.Refine FlockSoundness.Discharge.Exec

set_option maxHeartbeats 4000000 in
/-- **A session has at most `Tags.MAX_TABLES` tables, and its spec names one per statement**, at any public input
words. -/
theorem setupTables_le {tags : Tags} {cf pf : ByteArray} {coins : Coins} {count : ℕ}
    {tables : List (String × ByteArray)} {draw partition : Option Lean.Json} {program : Option ByteArray}
    {work : Option (ℕ × Array (String × ℕ × ℕ))} {stratified : Option ℕ}
    {closure : Option (Std.HashMap ℕ (Array ℕ))} {publicInputs : Option ByteArray} {spec : SessionSpec}
    {sts : Array Stmt}
    (h : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure publicInputs =
      .ok (spec, sts)) :
    sts.size ≤ Tags.MAX_TABLES ∧ spec.tables.size = sts.size := by
  unfold Stmt.setupTables at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp0 at h
  replace h : jp0 () = .ok (spec, sts) := by
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp0] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp1 at h
  replace h : jp1 () = .ok (spec, sts) := by
    try dsimp only at h
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp1] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp2 at h
  replace h : jp2 () = .ok (spec, sts) := by
    try dsimp only at h
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp2] at h
  dsimp only at h
  simp only [throw_bind', pure_bind] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
  all_goals
    first
    | (simp [Stmt.spec, Tags.MAX_TABLES]; done)
    | (obtain ⟨hsz, -⟩ := range_mapM_ok (by assumption)
       refine ⟨by omega, ?_⟩
       simp [Tags.tableNames, show ¬ count ≤ 1 by omega, hsz])

end Walk

/-- **A statement that sets up has at most 64 tables, and its spec names one per statement.** -/
theorem setup_tables {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hx : setupOf false x = some (s, spec, sts)) : sts.size ≤ 64 ∧ spec.tables.size = sts.size := by
  have h := setupTables_le (Ins.setup_read (setupOf_some hx).2)
  have hmax : Tags.MAX_TABLES = 64 := rfl
  exact ⟨by omega, h.2⟩

namespace Rows

open FlockSoundness.Refine (Frame colBytes)
open FlockSoundness.Discharge.DecodeZL (ProofSizes)
open FlockSoundness.Discharge.Exec (chunks chunks_flatten repFrames zcFrames lcFrames rsFrames ligFrames levelFr
  introFr queryFr chFr kFr capFr)

/-! ## Schedules -/

/-- **Every `fast100` schedule is of `m ≤ 35`.** -/
theorem fast100_m {m : ℕ} {s : Flock.Schedule} (hs : fast100 m = some s) : m ≤ 35 := by
  by_contra h
  rw [FlockSoundness.Refine.exec_fast100_none m (by omega)] at hs
  cases hs

/-- **Every `fast100` schedule** has at most 7 levels, each of a cap of at most `2^7` nodes, `ext ≤ 28` and
`k0 ≤ 6`. -/
theorem fast100_bound {m : ℕ} {s : Flock.Schedule} (hs : fast100 m = some s) :
    s.levels.size ≤ 7 ∧ s.ext ≤ 28 ∧ s.k0 ≤ 6 ∧ ∀ i < 7, (s.levels.getD i default).cap ≤ 7 := by
  have key : ∀ m < 40, (match fast100 m with
      | some s => decide (s.levels.size ≤ 7 ∧ s.ext ≤ 28 ∧ s.k0 ≤ 6 ∧
          ∀ i : ℕ, i < 7 → (s.levels.getD i default).cap ≤ 7)
      | none => true) = true := by
    decide +kernel
  have h := key m (by have := fast100_m hs; omega)
  rw [hs] at h
  simpa using h

/-! ## Values -/

theorem vals_one (v : F128) (fs : List Frame) : vals (.one v :: fs) = v :: vals fs := rfl
theorem vals_vec (vs : List F128) (fs : List Frame) : vals (.vec vs :: fs) = vs ++ vals fs := rfl
theorem vals_dom (b : ByteArray) (fs : List Frame) : vals (.dom b :: fs) = vals fs := rfl
theorem vals_label (b : ByteArray) (fs : List Frame) : vals (.label b :: fs) = vals fs := rfl
theorem vals_bytes (b : ByteArray) (fs : List Frame) : vals (.bytes b :: fs) = vals fs := rfl
theorem vals_sq (k : UInt8) (n : ℕ) (fs : List Frame) : vals (.sq k n :: fs) = vals fs := rfl
theorem vals_nil : vals [] = [] := rfl

theorem vals_map_one (ws : List F128) : vals (ws.map .one) = ws := by
  induction ws with
  | nil => rfl
  | cons w ws ih => exact congrArg (w :: ·) ih

theorem vals_flatMap_le {α : Type} (f : α → List Frame) (B : ℕ) :
    ∀ (L : List α), (∀ a ∈ L, (vals (f a)).length ≤ B) → (vals (L.flatMap f)).length ≤ L.length * B
  | [], _ => by simp [vals_nil]
  | a :: L, h => by
    rw [List.flatMap_cons, vals_append, List.length_append, List.length_cons, Nat.add_mul, Nat.one_mul]
    have := h a (by simp)
    have := vals_flatMap_le f B L fun b hb => h b (by simp [hb])
    omega

theorem vals_chFr (p : Flock.LigeritoProof) : ∀ (msg k : ℕ), (vals (chFr p msg k)).length = 4 * k
  | _, 0 => rfl
  | msg, k + 1 => by
    simp only [chFr, kFr, List.cons_append, List.nil_append, vals_sq, vals_vec, List.length_cons,
      vals_chFr p (msg + 1) k]
    omega

theorem vals_introFr (p : Flock.LigeritoProof) (dim ood msg cbn : ℕ) :
    (vals (introFr p dim ood msg cbn)).length = 5 := by
  simp [introFr, kFr, vals_sq, vals_one, vals_vec, vals_bytes, vals_nil]

theorem vals_queryFr (p : Flock.LigeritoProof) (lv : Flock.Level) (qn cn : ℕ) :
    vals (queryFr p lv qn cn) = [] := rfl

theorem vals_levelFr_le (sch : Flock.Schedule) (p : Flock.LigeritoProof) (i : ℕ) :
    (vals (levelFr sch p i)).length ≤ 30 + p.finalProof.yr.size := by
  unfold levelFr
  rw [vals_append, List.length_append, vals_chFr]
  split
  · simp only [capFr, vals_bytes, vals_append, List.length_append, vals_introFr, kFr, vals_queryFr, vals_vec,
      vals_sq, vals_nil, List.length_cons, List.length_nil, List.cons_append, List.nil_append]
    omega
  · simp only [vals_append, vals_map_one, vals_queryFr, vals_bytes, vals_sq, vals_nil, List.length_append,
      Array.length_toList, List.length_nil]
    omega

theorem vals_ligFrames_le (sch : Flock.Schedule) (T : F128) (cap : Array ByteArray) (p : Flock.LigeritoProof) :
    (vals (ligFrames sch T cap p)).length ≤ 20 + 4 * sch.k0 + (sch.levels.size - 1) * (30 + p.finalProof.yr.size) := by
  have hL := vals_flatMap_le (levelFr sch p) _ (List.range' 0 (sch.levels.size - 1)) fun i _ => vals_levelFr_le sch p i
  rw [List.length_range'] at hL
  simp only [ligFrames, capFr, kFr, List.cons_append, List.nil_append, List.append_assoc, vals_label, vals_one,
    vals_bytes, vals_sq, vals_vec, vals_append, vals_introFr, vals_queryFr, List.length_append, List.length_cons,
    vals_chFr] at hL ⊢
  omega

theorem vals_zcFrames_le (m : ℕ) (d c : ByteArray) (p : Flock.ZerocheckProof) (h1 : p.round1ab.size = 64)
    (h2 : p.round1c.size = 64) : (vals (zcFrames m d c p)).length ≤ 130 + 2 * (m - 6) := by
  have hL := vals_flatMap_le (fun i => [.one (p.rounds[i]!).1, .one (p.rounds[i]!).2, .sq 1 1]) 2
    (List.range' 0 (m - 6)) fun i _ => by simp [vals_one, vals_sq, vals_nil]
  rw [List.length_range'] at hL
  simp only [zcFrames, List.cons_append, List.nil_append, vals_label, vals_one, vals_bytes, vals_sq, vals_vec,
    vals_append, List.length_append, List.length_cons, List.length_nil, vals_nil, Array.length_toList, h1, h2] at hL ⊢
  omega

theorem vals_lcFrames_le (kLog : ℕ) (p : Flock.LincheckProof) (h : p.zPartial.size = 64) :
    (vals (lcFrames kLog p)).length ≤ 64 + 2 * (kLog - 6) := by
  have hL := vals_flatMap_le (fun j => [.one (p.rounds[j]!).1, .one (p.rounds[j]!).2, .sq 1 1]) 2
    (List.range' 0 (kLog - 6)) fun i _ => by simp [vals_one, vals_sq, vals_nil]
  rw [List.length_range'] at hL
  simp only [lcFrames, List.cons_append, List.nil_append, vals_label, vals_sq, vals_vec, vals_append,
    List.length_append, List.length_nil, vals_nil, Array.length_toList, h] at hL ⊢
  omega

theorem rs_get_le {rs : Array (Array F128)} (h : ∀ v ∈ rs, v.size = 128) (k : ℕ) : (rs[k]!).size ≤ 128 := by
  by_cases hk : k < rs.size
  · rw [getElem!_pos rs k hk]
    exact Nat.le_of_eq (h _ (Array.getElem_mem hk))
  · rw [getElem!_neg rs k hk]
    exact Nat.zero_le _

theorem vals_rsFrames_le (n : ℕ) (rs : Array (Array F128)) (h : ∀ v ∈ rs, v.size = 128) :
    (vals (rsFrames n rs)).length ≤ n * 128 := by
  have hL := vals_flatMap_le (fun k => [.label "flock-ring-switch-v0".toUTF8, .vec (rs[k]!).toList, .sq 2 7]) 128
    (List.range' 0 n) fun k _ => by simpa [vals_label, vals_vec, vals_sq, vals_nil] using rs_get_le h k
  rw [List.length_range'] at hL
  simp only [rsFrames, vals_label, vals_append, vals_sq, vals_nil, List.length_append, List.length_nil] at hL ⊢
  omega

/-- **A rep's frames absorb** the zerocheck's and lincheck's vectors and round values, `128` values a claim, and
Ligerito's messages, samples and residual. -/
theorem vals_repFrames_le (st : Flock.Setup) (sch : Flock.Schedule) (dom : ByteArray) (n : ℕ) (p : Flock.Proof)
    (T : F128)
    (h1 : p.zerocheck.round1ab.size = 64) (h2 : p.zerocheck.round1c.size = 64) (h3 : p.lincheck.zPartial.size = 64)
    (h4 : ∀ v ∈ p.ringSwitches, v.size = 128) :
    (vals (repFrames st sch dom n p T)).length ≤ 130 + 2 * (st.m - 6) + (64 + 2 * (st.fold.kLog - 6)) + n * 128 +
      (20 + 4 * sch.k0 + (sch.levels.size - 1) * (30 + p.ligerito.finalProof.yr.size)) := by
  have := vals_zcFrames_le st.m st.digest (p.cap.foldl (· ++ ·) ByteArray.empty) p.zerocheck h1 h2
  have := vals_lcFrames_le st.fold.kLog p.lincheck h3
  have := vals_rsFrames_le n p.ringSwitches h4
  have := vals_ligFrames_le sch T p.cap p.ligerito
  simp only [repFrames, vals_dom, vals_append, List.length_append]
  omega

theorem length_le_flatten {α : Type} {x : List α} : ∀ {L : List (List α)}, x ∈ L → x.length ≤ L.flatten.length
  | a :: L, h => by
    rw [List.flatten_cons, List.length_append]
    rcases List.mem_cons.1 h with rfl | h
    · omega
    · have := length_le_flatten h
      omega

/-- **A round absorbs at most its rep's values.** -/
theorem vals_chunk_le {fs ch : List Frame} (h : ch ∈ (chunks fs).1) : (vals ch).length ≤ (vals fs).length := by
  have e := congrArg (fun l => (vals l).length) (chunks_flatten fs)
  simp only [vals_append, vals_flatten, List.length_append] at e
  have := length_le_flatten (List.mem_map.2 ⟨ch, h, rfl⟩ : vals ch ∈ (chunks fs).1.map vals)
  omega

/-! ## Rows -/

theorem flatMap_take_drop (l : List (Array ByteArray)) (n : ℕ) :
    ((l.take n).flatMap Array.toList).length + ((l.drop n).flatMap Array.toList).length =
      (l.flatMap Array.toList).length := by
  calc _ = ((l.take n ++ l.drop n).flatMap Array.toList).length := by rw [List.flatMap_append, List.length_append]
    _ = _ := by rw [List.take_append_drop]

theorem flatMap_length_le {B : ℕ} : ∀ {l : List (Array ByteArray)}, (∀ a ∈ l, a.size ≤ B) →
    (l.flatMap Array.toList).length ≤ l.length * B
  | [], _ => by simp
  | a :: l, h => by
    rw [List.flatMap_cons, List.length_append, Array.length_toList, List.length_cons, Nat.add_mul, Nat.one_mul]
    have := h a (by simp)
    have := flatMap_length_le (l := l) fun b hb => h b (by simp [hb])
    omega

/-- **Each honest round's rows** are at most the caps' nodes and one value row, each a cap node's row or a chunk's
value row. -/
theorem honestRows_rows : ∀ (ss : List RecVStar.SchedRound) (chs : List (List Frame)) (caps : List (Array ByteArray)),
    ∀ rs ∈ honestRows ss chs caps, rs.length ≤ (caps.flatMap Array.toList).length + 1 ∧
      ∀ r ∈ rs, (∃ a ∈ caps, ∃ b ∈ a, r = nodeRow b) ∨ ∃ ch ∈ chs, r = ⟨(padRow (colBytes (vals ch))).toArray⟩
  | s :: ss, ch :: chs, caps, rs, hrs => by
    simp only [honestRows, List.mem_cons] at hrs
    rcases hrs with rfl | hrs
    · refine ⟨?_, fun r hr => ?_⟩
      · have := flatMap_take_drop caps s.2.1.size
        rw [List.length_append, List.length_map]
        split <;> simp only [List.length_singleton, List.length_nil] <;> omega
      · rcases List.mem_append.1 hr with hr | hr
        · obtain ⟨b, hb, rfl⟩ := List.mem_map.1 hr
          obtain ⟨a, ha, hba⟩ := List.mem_flatMap.1 hb
          exact Or.inl ⟨a, List.mem_of_mem_take ha, b, Array.mem_toList_iff.1 hba, rfl⟩
        · split at hr
          · exact Or.inr ⟨ch, List.mem_cons_self .., List.mem_singleton.1 hr⟩
          · cases hr
    · obtain ⟨h1, h2⟩ := honestRows_rows ss chs (caps.drop s.2.1.size) rs hrs
      refine ⟨by have := flatMap_take_drop caps s.2.1.size; omega, fun r hr => ?_⟩
      rcases h2 r hr with ⟨a, ha, hra⟩ | ⟨c', hc', e⟩
      · exact Or.inl ⟨a, List.mem_of_mem_drop ha, hra⟩
      · exact Or.inr ⟨c', List.mem_cons_of_mem _ hc', e⟩
  | [], _, _, rs, hrs => by simp [honestRows] at hrs
  | _ :: _, [], _, rs, hrs => by simp [honestRows] at hrs

/-- **An honest rep's cap**: at most `2^7` nodes, each of at most 64 bytes. -/
theorem rep_cap_le {c : RepCtx} {p : Flock.Proof} (hs : SchedOf c) (hp : HonestRep c p)
    (hd : c.st.merkle.digestLen ≤ 64) : p.cap.size ≤ 128 ∧ ∀ b ∈ p.cap, b.size ≤ 64 := by
  obtain ⟨-, -, -, hcap⟩ := fast100_bound hs.1
  obtain ⟨e, -, hps⟩ := hp.sizes
  have h1 : p.cap.size = 2 ^ (c.sch.levels.getD 0 default).cap := hps.cap
  have := Nat.pow_le_pow_right (show 0 < 2 by decide) (hcap 0 (by decide))
  refine ⟨by omega, fun b hb => ?_⟩
  have : b.size = c.st.merkle.digestLen := hps.capEach b hb
  omega

/-- **An honest rep's rows fit a message**, at a scheme of at most 64-byte digests, `k_log ≤ 35` and at most 2,050
claims. -/
theorem repRows_fit {c : RepCtx} {p : Flock.Proof} (T : F128) (hs : SchedOf c) (hp : HonestRep c p)
    (hd : c.st.merkle.digestLen ≤ 64) (hk : c.st.kLog ≤ 35) (hn : c.n ≤ 2050) :
    ∀ rs ∈ repRows c p T, RowsFit rs := by
  obtain ⟨hf, hfold, -⟩ := hs
  have hm := fast100_m hf
  obtain ⟨hL, hext, hk0, hcap⟩ := fast100_bound hf
  obtain ⟨e, -, hps⟩ := hp.sizes
  have hyr : p.ligerito.finalProof.yr.size = 2 ^ (c.sch.ext + 1) := hps.yr
  have hrc : p.ligerito.recursiveCaps.size = c.sch.levels.size - 1 := hps.rcaps
  have hfr : (vals (repFrames c.st c.sch c.dom c.n p T)).length < 2 ^ 40 := by
    have h1 := vals_repFrames_le c.st c.sch c.dom c.n p T hps.r1ab hps.r1c hps.zPartial hps.rsEach
    rw [hyr, hfold] at h1
    have h2 : 2 ^ (c.sch.ext + 1) ≤ 2 ^ 29 := Nat.pow_le_pow_right (by decide) (by omega)
    have h3 : (c.sch.levels.size - 1) * (30 + 2 ^ (c.sch.ext + 1)) ≤ 6 * (30 + 2 ^ 29) :=
      Nat.mul_le_mul (by omega) (by omega)
    omega
  have hcaps : (p.ligerito.recursiveCaps.toList.flatMap Array.toList).length ≤ 6 * 128 := by
    have h := flatMap_length_le (B := 128) (l := p.ligerito.recursiveCaps.toList) fun a ha => by
      obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 ha)
      have h1 : p.ligerito.recursiveCaps[i].size = 2 ^ (c.sch.levels.getD (i + 1) default).cap :=
        (hps.rcapsEach i hi).1
      have := Nat.pow_le_pow_right (show 0 < 2 by decide) (hcap (i + 1) (by omega))
      omega
    have e : p.ligerito.recursiveCaps.toList.length = p.ligerito.recursiveCaps.size := Array.length_toList
    omega
  intro rs hrs
  unfold repRows at hrs
  obtain ⟨hlen, hmem⟩ := honestRows_rows _ _ _ rs hrs
  refine ⟨by omega, fun r hr => ?_⟩
  rcases hmem r hr with ⟨a, ha, b, hb, rfl⟩ | ⟨ch, hch, rfl⟩
  · obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 ha)
    have : b.size = c.st.merkle.digestLen := (hps.rcapsEach i hi).2 b hb
    rw [size_nodeRow]
    omega
  · show (padRow (colBytes (vals ch))).toArray.size < 2 ^ 64
    rw [List.size_toArray]
    have hp := padRow_length_le (colBytes (vals ch))
    rw [Frame.colBytes_length] at hp
    have := vals_chunk_le (fs := repFrames c.st c.sch c.dom c.n p T) hch
    omega

/-- **Round 0's rows fit a message**: at most 64 caps, each of at most `2^7` nodes of at most 64 bytes. -/
theorem capRows_fit {caps : Array (Array ByteArray)} (hn : caps.size ≤ 64)
    (hc : ∀ a ∈ caps, a.size ≤ 128 ∧ ∀ b ∈ a, b.size ≤ 64) : RowsFit ((capRows caps).map nodeRow) := by
  refine ⟨?_, fun r hr => ?_⟩
  · have h := flatMap_length_le (B := 128) (l := caps.toList) fun a ha => (hc a (Array.mem_toList_iff.1 ha)).1
    have e : caps.toList.length = caps.size := Array.length_toList
    rw [List.length_map]
    show (caps.toList.flatMap Array.toList).length < 2 ^ 64
    omega
  · obtain ⟨b, hb, rfl⟩ := List.mem_map.1 hr
    obtain ⟨l, hl, hb⟩ := List.mem_flatten.1 hb
    obtain ⟨a, ha, rfl⟩ := List.mem_map.1 hl
    have := (hc a (Array.mem_toList_iff.1 ha)).2 b (Array.mem_toList_iff.1 hb)
    rw [size_nodeRow]
    omega

/-- **Rep `k`'s context at a `TableOk` table**: a scheme of at most 64-byte digests, `k_log ≤ 35`, and at most 2,050
claims where the table has at most 2,048 extra claims. -/
theorem repCtx_bounds {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin} {r₀ : List ByteArray} {cs : List Coin} {k : ℕ}
    (hk : k / 2 < sts.size) (hok : FlockComplete.JsonRT.TableOk sts[k / 2])
    (he : ∀ e, sts[k / 2].extraClaims (pointsOf spec c₁) = .ok e → e.size ≤ 2048) :
    (repCtx spec sts c₁ r₀ cs k).st.merkle.digestLen ≤ 64 ∧ (repCtx spec sts c₁ r₀ cs k).st.kLog ≤ 35 ∧
      (repCtx spec sts c₁ r₀ cs k).n ≤ 2050 := by
  have ht := tabsOf_get! hk
  have hst : (repCtx spec sts c₁ r₀ cs k).st = Flock.Setup.ofCircuit sts[k / 2] := ht
  refine ⟨?_, ?_, ?_⟩
  · rw [hst]
    exact (FlockComplete.JsonRT.scheme_sizes hok.scheme).2.1
  · rw [hst]
    show sts[k / 2].c.kLog ≤ 35
    have := hok.m
    have := hok.m35
    omega
  · show 2 + (match (tabsOf sts)[k / 2]!.extra (pointsOf spec c₁) with
      | .ok e => e.size
      | .error _ => 0) ≤ 2050
    rw [ht]
    show 2 + (match sts[k / 2].extraClaims (pointsOf spec c₁) with
      | .ok e => e.size
      | .error _ => 0) ≤ 2050
    cases hex : sts[k / 2].extraClaims (pointsOf spec c₁) with
    | error _ => dsimp only; omega
    | ok e => have := he e hex; dsimp only; omega

end Rows

/-- **`repRows_fit`'s and `capRows_fit`'s hypotheses hold together** at the honest rep `c₀`, `p₀` (`honest₀`): its rows
fit a message, and so does a round 0 of its cap. -/
theorem rowsFit₀ (T : F128) : (∀ rs ∈ repRows c₀ p₀ T, RowsFit rs) ∧ RowsFit ((capRows #[p₀.cap]).map nodeRow) :=
  ⟨Rows.repRows_fit T honest₀.1 honest₀.2.1 (by decide +kernel) (by decide +kernel) (by decide +kernel),
    Rows.capRows_fit (by simp) fun a ha => by
      obtain rfl : a = p₀.cap := by simpa using ha
      exact Rows.rep_cap_le honest₀.1 honest₀.2.1 (by decide +kernel)⟩

open FlockComplete Rows in
/-- **The honest run decides at its own proofs, with no row premise**: `sessionRun_decides`, whose rows fit a message
at every statement that sets up (`setup_tables`, `capRows_fit`, `repRows_fit`), and whose spec names one table per
statement. The files' `finalC` at the reader's `z` (`FilesFinalC`) stays a hypothesis here; `sessionRun_decides_coins`
discharges it. -/
theorem sessionRun_decides_fit {H : Claims.Msg → Claims.Msg} (h64 : Claims.Out64 H) {rd : Claims.Msg → Option FinalP}
    {x last : Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin}
    {resp : ℕ → Claims.Msg} {w : Claims.Flock.Wit} {ω : Tape} {Ts : ℕ → F128} {fl : FinalP}
    (hx : setupOf false x = some (s, spec, sts)) (hcs : cs.head? = some c₁) (hc : resp 0 = c₁.bytes)
    (htab : ∀ (i : ℕ) (hi : i < sts.size), JsonRT.TableOk sts[i] ∧
      ∀ e, sts[i].extraClaims (pointsOf spec c₁) = .ok e → e.size ≤ 2048)
    {rec : String} {pfs : Array ByteArray} {bs : Array Claims.Msg}
    (hS : FlockComplete.sessionS H resp spec sts w ω = (.ok (rec, pfs), bs))
    (hfc : FilesFinalC spec sts c₁ cs.tail.dropLast pfs) (hrd : rd last = some fl)
    (hps : fl.ps = filePs spec sts c₁ pfs) (hst : fl.stmt = x) :
    ∃ d, decR rd x (honestMsgs (fileCaps spec sts c₁ pfs) ((List.range (filePs spec sts c₁ pfs).size).map fun k =>
      repRows (repCtx spec sts c₁ (capRows (fileCaps spec sts c₁ pfs)) cs.tail.dropLast k) (filePs spec sts c₁ pfs)[k]!
        (Ts k)) last) cs = some d := by
  obtain ⟨h64t, hts⟩ := setup_tables hx
  obtain ⟨hsz, hall⟩ := sessionS_reps h64 hc hts.symm htab hS (capRows (fileCaps spec sts c₁ pfs)) cs.tail.dropLast
  refine sessionRun_decides h64 hx hcs hc hts.symm htab hS hfc ?_ ?_ hrd hps hst
  · refine capRows_fit (by rw [fileCaps_size]; exact h64t) fun a ha => ?_
    obtain ⟨i, hi, rfl⟩ := Array.mem_map.1 ha
    have hi' := Array.mem_range.1 hi
    have hk2 : 2 * i / 2 < sts.size := by omega
    obtain ⟨-, hsc, hon, -, -, -⟩ := hall (2 * i) (by omega)
    have hok := htab (2 * i / 2) hk2
    exact rep_cap_le hsc (hon (hfc (2 * i) (by rw [hsz]; omega))) (repCtx_bounds hk2 hok.1 hok.2).1
  · intro k hk rs hrs
    rw [filePs_size, hsz] at hk
    have hk2 : k / 2 < sts.size := by omega
    obtain ⟨-, hsc, hon, -, -, -⟩ := hall k hk
    have hok := htab (k / 2) hk2
    obtain ⟨hd, hkl, hn⟩ := repCtx_bounds (r₀ := capRows (fileCaps spec sts c₁ pfs)) (cs := cs.tail.dropLast) hk2
      hok.1 hok.2
    exact repRows_fit (Ts k) hsc (hon (hfc k (by rw [hsz]; omega))) hd hkl hn rs hrs

end FlockDeploy
