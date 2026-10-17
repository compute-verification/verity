import Proofs.Flock.Claims.Complete.Zk.Level0Pad
import Proofs.Flock.Claims.Complete.Replay.LigeritoRun
import Proofs.Flock.Soundness.Discharge.ZkExec.RunsLig

/-!
# The verifier's masked Ligerito returns on the model's rounds

`ligeritoWithZ_of`: plain's `Lg.ligeritoWith_of` for the masked verifier (`Zk.ligeritoWith`). From a transcript that has
recorded its rounds, on rounds that carry the masked model's (`outZ`), the masked verifier returns the lane phase's
record where the model's transcript ends, given the proof's list sizes, the level-0 hiding messages' shape, independent
pair coins, each level's Merkle check at its query coins (`MerkleOkZ`, level 0's rows `EXTRA_LANES` wider) and the final
identity at the model's values.

* **The model** (`outZ`): `LABEL_OPEN` after plain's start (`startZ`), plain's head, the lane rounds (`laneRun`: plain's
  challenge, then at the last lane round the pair's sums, its coins `ρ`, `LABEL_CLOSE`, the blinded claim and `ybar`,
  `chalZ`), level 0's cap, intros and query, its value the padded `consE0 + extraE` (`lv0Z`), then plain's level loop.
* **Its cursors** (`inv_x₀Z`): plain's `Inv` from round `k + 2` (the pair's two rounds); the level loop's invariant from
  any start (`inv_loop_of`, `done_of`).
-/

namespace FlockComplete.L0

open Flock Transcript FlockComplete Replay Replay.Lg

/-! ## The model -/

/-- The pending bytes after `absorbK v`. -/
def pKv (v : F256) (P : ByteArray) : ByteArray := pSlice #[v.c0, v.c1] P

/-- `ybar`'s coordinates, as the verifier absorbs them. -/
def ybarFlat (ybar : Array F256) : Array F128 := ybar.foldl (fun acc v => (acc.push v.c0).push v.c1) #[]

/-- The transcript once the masked verifier has absorbed its label, `T`, the level-0 cap and `LABEL_OPEN`. -/
def startZ (cap : Array ByteArray) (T : F128) (t : Transcript) : Transcript :=
  ap (pLabel Zk.LABEL_OPEN) (start cap T t)

@[simp] theorem startZ_k (cap : Array ByteArray) (T : F128) (t : Transcript) : (startZ cap T t).k = t.k := rfl

/-- **The last lane round** from `(st, t)`: the coin `r`, `s ← q(r)`, the pair's sums, the pair's coins `ρ`,
`LABEL_CLOSE`, the blinded claim `T'` (which `s` becomes) and `ybar`, then the next message `m` and `q ← (m, T')`. -/
def chalZ (p : LigeritoProof) (cf : ℕ → ℕ → F128) (zm : Zk.Level0Msgs) (st : LState) (t : Transcript) :
    F256 × (F256 × F256) × LState × Transcript :=
  let r : F256 := ⟨cf t.k 0, cf t.k 1⟩
  let t₁ := ap (pKv zm.e1) (ap (pKv zm.e0) (afterSq t 2 2 (cf t.k)))
  let t₂ := afterSq t₁ 2 2 (cf t₁.k)
  let t₃ := afterSq t₂ 2 2 (cf t₂.k)
  let t₄ := ap (pSlice (ybarFlat zm.ybar)) (ap (pKv zm.tPrime) (ap (pLabel Zk.LABEL_CLOSE) t₃))
  let m := p.messages[st.msg]!
  (r, (⟨cf t₁.k 0, cf t₁.k 1⟩, ⟨cf t₂.k 0, cf t₂.k 1⟩),
    { { { { st with s := st.q.eval r } with s := zm.tPrime } with msg := st.msg + 1 } with
      q := Poly.ofMsg m zm.tPrime },
    ap (pK m) t₄)

/-- Lane round `j` of `k0`: plain's challenge, or the last lane round's (`chalZ`), on the loop's challenges and pair. -/
def laneStep (p : LigeritoProof) (cf : ℕ → ℕ → F128) (k0 : ℕ) (zm : Zk.Level0Msgs) (j : ℕ)
    (x : (Array F256 × F256 × F256) × LState × Transcript) : (Array F256 × F256 × F256) × LState × Transcript :=
  if j + 1 = k0 then
    let y := chalZ p cf zm x.2.1 x.2.2
    ((x.1.1.push y.1, y.2.1), y.2.2.1, y.2.2.2)
  else
    let y := chal p cf x.2.1 x.2.2
    ((x.1.1.push y.1, x.1.2), y.2.1, y.2.2)

/-- The lane rounds after `j` steps from `(st, t)`. -/
def laneRun (p : LigeritoProof) (cf : ℕ → ℕ → F128) (k0 : ℕ) (zm : Zk.Level0Msgs) (st : LState) (t : Transcript) :
    ℕ → (Array F256 × F256 × F256) × LState × Transcript
  | 0 => ((#[], 0, 0), st, t)
  | j + 1 => laneStep p cf k0 zm j (laneRun p cf k0 zm st t j)

/-- Level 0 after the lane rounds, on their challenges `initCh` and pair `rho`: level 1's cap and two out-of-domain
intros, level 0's query, its value the padded one, closed. -/
def lv0Z (p : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (zl : Zk.Level0) (initCh : Array F256)
    (rho : F256 × F256) (st : LState) (t : Transcript) : LState × Transcript :=
  let x₁ := ood p cf (sch.n - sch.k0 + 1) 0 st (ap (pBytes (capBytes p.recursiveCaps[0]!)) t)
  let x₂ := ood p cf (sch.n - sch.k0 + 1) 0 x₁.1 x₁.2
  let lv := sch.levels.getD 0 default
  let qr := query p cf lv p.initialProof.rows initCh x₂.1 x₂.2
  let e := FlockSoundness.Discharge.ZkExec.consE0 p.initialProof.rows lv.q (2 ^ lv.lanes) qr.1.2.1 initCh +
    FlockSoundness.Discharge.ZkExec.extraE p.initialProof.rows qr.1.1 (2 ^ lv.lanes) zl.rep lv.cols rho qr.1.2.1
      zl.msgs.ybar
  close p cf (sch.n - sch.k0) 0 (qr.1.1, qr.1.2.1, e) qr.2.1 qr.2.2

/-- Up to the level loop: the lane rounds' challenges and pair, and the state and transcript after level 0. -/
def preZ (p : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (T : F128) (cap : Array ByteArray)
    (zl : Zk.Level0) (t : Transcript) : (Array F256 × F256 × F256) × LState × Transcript :=
  let h := head p cf sch.n (init T) (startZ cap T t)
  let c := laneRun p cf sch.k0 zl.msgs h.1 h.2 sch.k0
  let y := lv0Z p cf sch zl c.1.1 c.1.2 c.2.1 c.2.2
  (c.1, y.1, y.2)

/-- The level loop's start. -/
def x₀Z (p : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (T : F128) (cap : Array ByteArray)
    (zl : Zk.Level0) (t : Transcript) : Vars × LState × Transcript :=
  ((sch.n - sch.k0 + 1, p.recursiveCaps[0]!, #[], 0), (preZ p cf sch T cap zl t).2.1, (preZ p cf sch T cap zl t).2.2)

/-- **The masked verifier's Ligerito, in closed form**: the level loop's variables, state and transcript at its end. -/
def outZ (p : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (T : F128) (cap : Array ByteArray)
    (zl : Zk.Level0) (t : Transcript) : Vars × LState × Transcript :=
  loop p cf sch (x₀Z p cf sch T cap zl t) (sch.levels.size - 1)

/-- The record the masked verifier's Ligerito returns, at the state `st` it ends in and the lane phase's `(ch, rho)`. -/
def recZ (p : LigeritoProof) (T : F128) (st : LState) (ch : Array F256 × F256 × F256) : Zk.L0Rec :=
  { target := T, ood := #[((st.oods[0]!).beta, p.oodValues[0]!)], challenges := ch.1, rho := ch.2 }

/-! ## Its cursors -/

theorem laneRun_fields (p : LigeritoProof) (cf : ℕ → ℕ → F128) (k0 : ℕ) (zm : Zk.Level0Msgs) (st : LState)
    (t : Transcript) : ∀ j, j ≤ k0 →
    (laneRun p cf k0 zm st t j).2.1.msg = st.msg + j ∧ (laneRun p cf k0 zm st t j).2.1.ood = st.ood ∧
      (laneRun p cf k0 zm st t j).2.1.cbn = st.cbn ∧ (laneRun p cf k0 zm st t j).2.1.qn = st.qn ∧
      (laneRun p cf k0 zm st t j).2.1.cn = st.cn ∧
      (laneRun p cf k0 zm st t j).2.2.k = t.k + j + (if j = k0 ∧ 0 < k0 then 2 else 0) ∧
      (laneRun p cf k0 zm st t j).1.1.size = j ∧ (laneRun p cf k0 zm st t j).2.1.oods = st.oods ∧
      (laneRun p cf k0 zm st t j).2.1.conss = st.conss
  | 0, _ => by
    refine ⟨rfl, rfl, rfl, rfl, rfl, ?_, rfl, rfl, rfl⟩
    rw [ite_eq_right (by omega)]
    rfl
  | j + 1, hj => by
    obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9⟩ := laneRun_fields p cf k0 zm st t j (by omega)
    rw [ite_eq_right (by omega), Nat.add_zero] at h6
    by_cases hl : j + 1 = k0
    · simp only [laneRun, laneStep, ite_eq_left hl, chalZ, ap, afterSq, Array.size_push, h1, h2, h3, h4, h5, h6, h7, h8,
        h9, true_and, and_true, ite_eq_left (show j + 1 = k0 ∧ 0 < k0 from ⟨hl, by omega⟩)]
      omega
    · simp only [laneRun, laneStep, ite_eq_right hl, chal, ap, afterSq, Array.size_push, h1, h2, h3, h4, h5, h6, h7, h8,
        h9, true_and, and_true, ite_eq_right (show ¬ (j + 1 = k0 ∧ 0 < k0) by omega), Nat.add_zero]
      omega

theorem laneRun_ch (p : LigeritoProof) (cf : ℕ → ℕ → F128) (k0 : ℕ) (zm : Zk.Level0Msgs) (st : LState)
    (t : Transcript) : ∀ j, j ≤ k0 → ∀ i < j,
    (laneRun p cf k0 zm st t j).1.1[i]! = ⟨cf (t.k + i) 0, cf (t.k + i) 1⟩
  | 0, _, i, hi => absurd hi (Nat.not_lt_zero i)
  | j + 1, hj, i, hi => by
    have hf := laneRun_fields p cf k0 zm st t j (by omega)
    have hk : (laneRun p cf k0 zm st t j).2.2.k = t.k + j := by rw [hf.2.2.2.2.2.1, ite_eq_right (by omega)]; rfl
    have hsz := hf.2.2.2.2.2.2.1
    have ih := laneRun_ch p cf k0 zm st t j (by omega)
    have key : (laneRun p cf k0 zm st t (j + 1)).1.1 = (laneRun p cf k0 zm st t j).1.1.push
        ⟨cf (t.k + j) 0, cf (t.k + j) 1⟩ := by
      show (laneStep p cf k0 zm j (laneRun p cf k0 zm st t j)).1.1 = _
      unfold laneStep
      split <;> simp only [chalZ, chal, hk]
    rw [key]
    rcases Nat.lt_succ_iff_lt_or_eq.1 hi with hi | hi
    · rw [push_get_lt _ (by rw [hsz]; exact hi)]
      exact ih i hi
    · rw [push_get_eq' _ (hsz.trans hi.symm), hi]

theorem laneRun_rho (p : LigeritoProof) (cf : ℕ → ℕ → F128) (k0 : ℕ) (zm : Zk.Level0Msgs) (st : LState)
    (t : Transcript) (hk0 : 1 ≤ k0) :
    (laneRun p cf k0 zm st t k0).1.2 = (⟨cf (t.k + k0) 0, cf (t.k + k0) 1⟩, ⟨cf (t.k + k0 + 1) 0, cf (t.k + k0 + 1) 1⟩) := by
  obtain ⟨j, rfl⟩ : ∃ j, k0 = j + 1 := ⟨k0 - 1, by omega⟩
  have hf := laneRun_fields p cf (j + 1) zm st t j (by omega)
  have hk : (laneRun p cf (j + 1) zm st t j).2.2.k = t.k + j := by rw [hf.2.2.2.2.2.1, ite_eq_right (by omega)]; rfl
  show (laneStep p cf (j + 1) zm j (laneRun p cf (j + 1) zm st t j)).1.2 = _
  rw [laneStep, ite_eq_left rfl]
  simp only [chalZ, ap, afterSq, hk]
  rw [show t.k + j + 1 = t.k + (j + 1) by omega]

theorem inv_x₀Z (p : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (T : F128) (cap : Array ByteArray)
    (zl : Zk.Level0) (t : Transcript) (hL : 2 ≤ sch.levels.size) (hk0 : 1 ≤ sch.k0) :
    Inv p sch (t.k + 2) 0 (x₀Z p cf sch T cap zl t) := by
  have hf := laneRun_fields p cf sch.k0 zl.msgs (head p cf sch.n (init T) (startZ cap T t)).1
    (head p cf sch.n (init T) (startZ cap T t)).2 sch.k0 le_rfl
  obtain ⟨h1, h2, h3, h4, h5, h6, -, h8, h9⟩ := hf
  rw [ite_eq_left ⟨rfl, by omega⟩] at h6
  refine
    { sd := rfl
      cap := rfl
      lvs := rfl
      lv4 := by intro l hl; simp [x₀Z] at hl
      msg := ?_
      ood := ?_
      cbn := ?_
      qn := ?_
      cn := ?_
      k := ?_
      oods := ?_
      conss := ?_ }
  · simp only [x₀Z, preZ, lv0Z, close_fields, query_fields, ood_fields, h1, head_fields, init_fields]; omega
  · simp only [x₀Z, preZ, lv0Z, close_fields, query_fields, ood_fields, h2, head_fields, init_fields]
  · simp only [x₀Z, preZ, lv0Z, close_fields, query_fields, ood_fields, h3, head_fields, init_fields]
  · simp only [x₀Z, preZ, lv0Z, close_fields, query_fields, ood_fields, h4, head_fields, init_fields]
  · simp only [x₀Z, preZ, lv0Z, close_fields, query_fields, ood_fields, h5, head_fields, init_fields]
  · simp only [x₀Z, preZ, lv0Z, close_fields, query_fields, ood_fields, h6, ap_k, head_fields, startZ_k]
  · simp only [x₀Z, preZ, lv0Z, close_fields, query_fields, ood_fields, h8, head_fields, init_fields,
      Array.toList_push, List.nil_append, List.mem_append, List.mem_cons, List.not_mem_nil, or_false]
    rintro o ((rfl | rfl) | rfl)
    · simp [OodDim]
    · simp [OodDim]; omega
    · simp [OodDim]; omega
  · simp only [x₀Z, preZ, lv0Z, close_fields, query_fields, ood_fields, h9, head_fields, init_fields,
      Array.toList_push, List.nil_append, List.mem_singleton]
    rintro cs rfl
    simp [ConsDim]

theorem inv_loop_of (p : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) {k : ℕ}
    {x : Vars × LState × Transcript} (h0 : Inv p sch k 0 x) (hsd : sch.k0 + 3 * (sch.levels.size - 1) ≤ sch.n) :
    ∀ j, j < sch.levels.size - 1 → Inv p sch k j (loop p cf sch x j)
  | 0, _ => h0
  | j + 1, hj => by
    rw [loop_stepN p cf sch _ hj]
    exact (inv_loop_of p cf sch h0 hsd j (by omega)).next hj hsd

theorem done_of (p : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) {k : ℕ} {x : Vars × LState × Transcript}
    (h0 : Inv p sch k 0 x) (hL : 2 ≤ sch.levels.size) (hsd : sch.k0 + 3 * (sch.levels.size - 1) ≤ sch.n) :
    Done p sch k (loop p cf sch x (sch.levels.size - 1)) := by
  obtain ⟨j, hj⟩ : ∃ j, sch.levels.size - 1 = j + 1 := ⟨sch.levels.size - 2, by omega⟩
  rw [hj, loop_stepL p cf sch _ (by omega)]
  exact (inv_loop_of p cf sch h0 hsd j (by omega)).last (by omega) hsd

/-! ## The Merkle openings -/

/-- **Every level's Merkle check passes** at its query coins, the coins of round `k + qOff sch j`: level 0's rows
`2^lanes + EXTRA_LANES` wide, the others as plain's. -/
def MerkleOkZ (hs : Hashes V) (ms : MerkleScheme) (sch : Schedule) (cap : Array ByteArray) (p : LigeritoProof)
    (cf : ℕ → ℕ → F128) (k : ℕ) : Prop :=
  merkleCheckWith (M := V) (ms.opsWith hs) ms cap (sch.levels.getD 0 default).cap (sch.levels.getD 0 default).d
      (positions (sch.levels.getD 0 default) ((Array.range (sch.levels.getD 0 default).q).map (cf (k + qOff sch 0))))
      p.initialProof.rows p.initialProof.siblings p.initialProof.salts
      (2 ^ (sch.levels.getD 0 default).lanes + Zk.EXTRA_LANES) = .ok () ∧
  ∀ j < sch.levels.size, 0 < j →
    merkleCheckWith (M := V) (ms.opsWith hs) ms (opening p cap (sch.levels.size - 1) j).1
      (sch.levels.getD j default).cap (sch.levels.getD j default).d
      (positions (sch.levels.getD j default) ((Array.range (sch.levels.getD j default).q).map (cf (k + qOff sch j))))
      (opening p cap (sch.levels.size - 1) j).2.1 (opening p cap (sch.levels.size - 1) j).2.2.1
      (opening p cap (sch.levels.size - 1) j).2.2.2 (2 ^ (sch.levels.getD j default).lanes) = .ok ()

theorem MerkleOkZ.at {hs : Hashes V} {ms : MerkleScheme} {sch : Schedule} {cap : Array ByteArray}
    {p : LigeritoProof} {cf : ℕ → ℕ → F128} {k : ℕ} (h : MerkleOkZ hs ms sch cap p cf k) {j : ℕ}
    (hj : j < sch.levels.size) (hj0 : 0 < j) {k' : ℕ} (hk : k' = k + qOff sch j) {c : Array ByteArray}
    {rows : Array (Array F128)} {sibs salts : Array ByteArray}
    (ho : opening p cap (sch.levels.size - 1) j = (c, rows, sibs, salts)) :
    merkleCheckWith (M := V) (ms.opsWith hs) ms c (sch.levels.getD j default).cap (sch.levels.getD j default).d
      (positions (sch.levels.getD j default) ((Array.range (sch.levels.getD j default).q).map (cf k'))) rows sibs
      salts (2 ^ (sch.levels.getD j default).lanes) = .ok () := by
  have := h.2 j hj hj0
  rw [ho] at this
  rw [hk]
  exact this

/-! ## The masked verifier's own steps -/

variable {F : Array Round} {k₀ : ℕ}

theorem Runs.absorbK (v : F256) (st : LState) (t : Transcript) :
    Runs F k₀ (lift (absorbK v)) st t () st (ap (pKv v) t) :=
  Runs.stay rfl rfl fun _ => lift_of (Flock.RecAlgebra.absorbSlice_eq _ _)

theorem Runs.absorbSlice (vs : Array F128) (st : LState) (t : Transcript) :
    Runs F k₀ (lift (absorbSlice vs)) st t () st (ap (pSlice vs) t) :=
  Runs.stay rfl rfl fun _ => lift_of (Flock.RecAlgebra.absorbSlice_eq _ _)

theorem Runs.label (s : String) (st : LState) (t : Transcript) :
    Runs F k₀ (lift (label s)) st t () st (ap (pLabel s) t) :=
  Runs.stay rfl rfl fun _ => lift_of (Flock.RecAlgebra.label_eq _ _)

/-- **Level 0's masked query** (`Zk.queryLevel0With`): plain's query, its rows `EXTRA_LANES` wider in the Merkle check,
its value the first `2^lanes` lanes folded (`consE0`). -/
theorem Runs.queryLevel0 (hs : Hashes V) (ms : MerkleScheme) (p : LigeritoProof) (cf : ℕ → ℕ → F128)
    (lv : Flock.Level) (cap : Array ByteArray) (fold : Array F256) (st : LState) (t : Transcript) (hq : lv.q ≠ 0)
    (hqn : st.qn < p.queryNonces.size) (hcn : st.cn < p.consistencyNonces.size)
    (hmc : merkleCheckWith (M := V) (ms.opsWith hs) ms cap lv.cap lv.d (positions lv (queryCoins p cf lv st t))
      p.initialProof.rows p.initialProof.siblings p.initialProof.salts (2 ^ lv.lanes + Zk.EXTRA_LANES) = .ok ()) :
    Runs F k₀ (Zk.queryLevel0With hs ms p lv cap fold) st t
      ((query p cf lv p.initialProof.rows fold st t).1.1, (query p cf lv p.initialProof.rows fold st t).1.2.1,
        FlockSoundness.Discharge.ZkExec.consE0 p.initialProof.rows lv.q (2 ^ lv.lanes)
          (query p cf lv p.initialProof.rows fold st t).1.2.1 fold)
      (query p cf lv p.initialProof.rows fold st t).2.1 (query p cf lv p.initialProof.rows fold st t).2.2 := by
  unfold Zk.queryLevel0With
  simp only [Flock.With.liftLM_V]
  refine (Runs.get st t).bind ((Runs.nonceFrom _ _ _ st t hqn).bind ((Runs.nonceFrom _ _ _ st t (by exact hcn)).bind
    ((Runs.set _ st t).bind ((Runs.pow _ _ t).bind ((Runs.sampleVec hs cf hq _ _).bind ((Runs.pow _ _ _).bind
    ((Runs.sampleVec hs cf (Nat.succ_ne_zero _) _ _).bind ?_)))))))
  have hmc' : merkleCheckWith (M := V) (ms.opsWith hs) ms cap lv.cap lv.d
      (positions lv (vec cf (ap (pN p.queryNonces[st.qn]!) t) lv.q)) p.initialProof.rows p.initialProof.siblings
      p.initialProof.salts (2 ^ lv.lanes + Zk.EXTRA_LANES) = .ok () := hmc
  simp only [hmc', List.forIn_pure_yield_eq_foldl, Std.Legacy.Range.forIn_eq_forIn_range', pure_bind,
    FlockSoundness.Refine.range_size, ← List.range_eq_range']
  exact Runs.pure _ _ _

/-- A loop over `[0:n]` whose every step yields a function of the step and the state. -/
theorem Runs.forInPure {σ : Type} (n : ℕ) (s : σ) (g : ℕ → σ → σ) (st : LState) (t : Transcript) :
    Runs F k₀ (forIn [:n] s fun j a => (Pure.pure (ForInStep.yield (g j a)) : LM (ForInStep σ))) st t
      ((List.range' 0 n).foldl (fun a j => g j a) s) st t := by
  rw [FlockSoundness.Refine.lforIn_range_eq, List.forIn_pure_yield_eq_foldl]
  exact Runs.pure _ _ _

theorem MerkleOkZ.at0 {hs : Hashes V} {ms : MerkleScheme} {sch : Schedule} {cap : Array ByteArray}
    {p : LigeritoProof} {cf : ℕ → ℕ → F128} {k : ℕ} (h : MerkleOkZ hs ms sch cap p cf k) {k' : ℕ}
    (hk : k' = k + qOff sch 0) :
    merkleCheckWith (M := V) (ms.opsWith hs) ms cap (sch.levels.getD 0 default).cap (sch.levels.getD 0 default).d
      (positions (sch.levels.getD 0 default) ((Array.range (sch.levels.getD 0 default).q).map (cf k')))
      p.initialProof.rows p.initialProof.siblings p.initialProof.salts
      (2 ^ (sch.levels.getD 0 default).lanes + Zk.EXTRA_LANES) = .ok () := by
  rw [hk]
  exact h.1

/-! ## The final check -/

open FlockSoundness.Refine in
/-- **The masked verifier's final check passes** (`Flock.RecAlgebra.finalProg_of`, returning the lane phase's record):
on a state that meets its shape checks, its value check and its leftover-list check. -/
theorem finalProgZ_of {p : LigeritoProof} {basisAt : Array F256 → ℕ → Array F256} {n : ℕ} {T : F128}
    {initCh : Array F256} {rho : F256 × F256} {levels : Array (Array F256)} {ext : ℕ} {st : LState}
    {tr : Transcript}
    (h1 : (origE initCh levels).size + ext = n) (h2 : (basisAt (origE initCh levels) ext).size = 2 ^ ext)
    (hood : ∀ o ∈ st.oods.toList, o.z.size = (oodFixed levels (origE initCh levels) o).1.size + ext)
    (hcons : ∀ cs ∈ st.conss.toList, (consFixed levels cs.start).size + ext = cs.cols)
    (hlhs : lhsE p.finalProof.yr ext (resE basisAt n initCh levels ext st) = st.s)
    (hleft : st.ood = p.oodValues.size ∧ st.qn = p.queryNonces.size ∧ st.cn = p.consistencyNonces.size ∧
      st.cbn = p.claimNonces.size) :
    FlockSoundness.Discharge.ZkExec.finalProgZ p basisAt n T initCh rho levels ext st tr =
      .ok (({ target := T, ood := #[((st.oods[0]!).beta, p.oodValues[0]!)], challenges := initCh, rho }, st), tr) := by
  unfold FlockSoundness.Discharge.ZkExec.finalProgZ
  rw [Flock.RecAlgebra.lget_bind_eq]
  dsimp only
  change ((if ((origE initCh levels).size + ext != n) = true then _ else _ : Flock.LM Zk.L0Rec) st tr) = _
  rw [ite_eq_right (by simp [h1])]
  change ((if ((basisAt (origE initCh levels) ext).size != 2 ^ ext) = true then _ else _ : Flock.LM Zk.L0Rec) st tr) = _
  rw [ite_eq_right (by simp [h2]), ← Array.forIn_toList]
  rw [Flock.RecAlgebra.lbind_forIn (fun res o => oodStep ext (oodFixed levels (origE initCh levels) o) o res)
    (fun o => o.z.size = (oodFixed levels (origE initCh levels) o).1.size + ext) ?hb1 hood]
  case hb1 =>
    intro o s st tr hz
    change ((if (o.z.size != (oodFixed levels (origE initCh levels) o).1.size + ext) = true then _ else _ :
      Flock.LM (ForInStep (Array F256))) st tr) = _
    rw [ite_eq_right (by simp [hz])]
    rfl
  rw [← Array.forIn_toList]
  rw [Flock.RecAlgebra.lbind_forIn (consStep (svTable (n + 1)) ext levels)
    (fun cs => (consFixed levels cs.start).size + ext = cs.cols) ?hb2 hcons]
  case hb2 =>
    intro cs s st tr hz
    change ((if ((consFixed levels cs.start).size + ext != cs.cols) = true then _ else _ :
      Flock.LM (ForInStep (Array F256))) st tr) = _
    rw [ite_eq_right (by simp [hz]), lforIn_range_eq, List.forIn_pure_yield_eq_foldl]
    rfl
  change ((if (lhsE p.finalProof.yr ext (resE basisAt n initCh levels ext st) != st.s) = true then _ else _ :
    Flock.LM Zk.L0Rec) st tr) = _
  obtain ⟨e1, e2, e3, e4⟩ := hleft
  rw [ite_eq_right (by simp [hlhs]), ite_eq_right (by simp [e1, e2, e3, e4])]
  rfl

/-! ## The walk -/

set_option maxHeartbeats 4000000 in
/-- **The masked verifier's Ligerito returns** on rounds `R` that carry the masked model's from `t.k` on: from a
transcript that has recorded its rounds, at a proof with the lengths fast100's `R + 1 ≥ 2` levels fix, whose level-0 cap
is the commitment's, whose hiding messages have their shape (`hzl`), whose pair coins are independent at the model's
(`hrho`), whose Merkle openings pass at their query coins (`hmo`) and whose final identity holds at the model's values
(`hfin`), it returns the lane phase's record at the model's position and pending bytes, and that position is the
model's round count. -/
theorem ligeritoWithZ_of (hs : Hashes V) (ms : MerkleScheme) (sch : Schedule) (T : F128) (cap : Array ByteArray)
    (p : LigeritoProof) (basisAt : Array F256 → ℕ → Array F256) (zl : Zk.Level0) (cf : ℕ → ℕ → F128)
    (t : Transcript) (hk : t.k = t.rounds.size) {R : Array Round}
    (hR : Agrees R (outZ p cf sch T cap zl t).2.2.rounds t.k) (hcap : (p.initialCap == cap) = true)
    (hzl : (zl.msgs.ybar.size == zl.tPad && decide (zl.rep < Zk.EXTRA_LANES / 2)) = true)
    (hrho : Zk.rhoIndependent (preZ p cf sch T cap zl t).1.2 = true) (hn : sch.n ≠ 0) (hk0 : 1 ≤ sch.k0)
    (hL : 2 ≤ sch.levels.size) (hsd : sch.k0 + 3 * (sch.levels.size - 1) ≤ sch.n)
    (hq : ∀ j < sch.levels.size, (sch.levels.getD j default).q ≠ 0)
    (hmsg : p.messages.size = sch.k0 + 7 * (sch.levels.size - 1) + 1)
    (hood : p.oodValues.size = 2 * (sch.levels.size - 1) + 1)
    (hqn : p.queryNonces.size = sch.levels.size) (hcn : p.consistencyNonces.size = sch.levels.size)
    (hcbn : p.claimNonces.size = 3 * (sch.levels.size - 1) + 2)
    (hyr : p.finalProof.yr.size = 2 ^ (sch.n - sch.k0 - 3 * (sch.levels.size - 1) + 1))
    (hb : ∀ q e, (basisAt q e).size = 2 ^ e) (hmo : MerkleOkZ hs ms sch cap p cf (t.k + 2))
    (hfin : FlockSoundness.Refine.lhsE p.finalProof.yr (outZ p cf sch T cap zl t).1.2.2.2
      (FlockSoundness.Refine.resE basisAt sch.n (preZ p cf sch T cap zl t).1.1 (outZ p cf sch T cap zl t).1.2.2.1
        (outZ p cf sch T cap zl t).1.2.2.2 (outZ p cf sch T cap zl t).2.1) = (outZ p cf sch T cap zl t).2.1.s) :
    (Zk.ligeritoWith hs ms sch T cap p basisAt zl).run ⟨R, t.k, t.pending⟩ =
      .ok (recZ p T (outZ p cf sch T cap zl t).2.1 (preZ p cf sch T cap zl t).1,
        ⟨R, (outZ p cf sch T cap zl t).2.2.k, (outZ p cf sch T cap zl t).2.2.pending⟩) ∧
    (outZ p cf sch T cap zl t).2.2.k = (outZ p cf sch T cap zl t).2.2.rounds.size := by
  show Zk.ligeritoWith hs ms sch T cap p basisAt zl ⟨R, t.k, t.pending⟩ = _ ∧ _
  unfold Zk.ligeritoWith
  simp only [Flock.With.liftTM_V, Flock.With.liftLM_V]
  rw [Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.check_eq hcap _ _),
    Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.check_eq hzl _ _),
    Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.label_eq _ _),
    Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.absorbF_eq _ _),
    Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.absorbBytes_eq _ _),
    Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.label_eq _ _)]
  refine (fun h : Runs (outZ p cf sch T cap zl t).2.2.rounds t.k _ (init T) (startZ cap T t)
      (recZ p T (outZ p cf sch T cap zl t).2.1 (preZ p cf sch T cap zl t).1)
      (outZ p cf sch T cap zl t).2.1 (outZ p cf sch T cap zl t).2.2 =>
    ⟨(Flock.RecAlgebra.bind_of_ok (t₁ := on R (outZ p cf sch T cap zl t).2.2)
      (h.2 le_rfl hk (Prefix.refl _) R hR)).trans rfl, h.1.2.2 hk⟩) ?_
  refine Runs.run ?_
  refine (Runs.sampleVec hs cf hn _ _).bind ((Runs.nextOod p _ _ (by simp; omega)).bind
    ((Runs.absorbF _ _ _).bind ((Runs.claimPow p _ _ (by simp; omega)).bind ((Runs.sample hs cf _ _).bind
    ((Runs.modify _ _ _).bind ((Runs.nextMsg p _ _ (by simp; omega)).bind ((Runs.modify _ _ _).bind ?_)))))))
  refine Runs.start (st₀ := (head p cf sch.n (init T) (startZ cap T t)).1)
    (t₀ := (head p cf sch.n (init T) (startZ cap T t)).2) rfl rfl ?_
  rw [FlockSoundness.Refine.lforIn_range_eq]
  have hf0 := laneRun_fields p cf sch.k0 zl.msgs (head p cf sch.n (init T) (startZ cap T t)).1
    (head p cf sch.n (init T) (startZ cap T t)).2 sch.k0 le_rfl
  rw [ite_eq_left ⟨rfl, by omega⟩] at hf0
  simp only [head_fields, init_fields] at hf0
  refine Runs.bind (Runs.loop _
    (fun j => (laneRun p cf sch.k0 zl.msgs (head p cf sch.n (init T) (startZ cap T t)).1
      (head p cf sch.n (init T) (startZ cap T t)).2 j).1)
    (fun j => (laneRun p cf sch.k0 zl.msgs (head p cf sch.n (init T) (startZ cap T t)).1
      (head p cf sch.n (init T) (startZ cap T t)).2 j).2.1)
    (fun j => (laneRun p cf sch.k0 zl.msgs (head p cf sch.n (init T) (startZ cap T t)).1
      (head p cf sch.n (init T) (startZ cap T t)).2 j).2.2) sch.k0 0 fun j hj => ?_) ?_
  · rw [Nat.zero_add]
    have hf := laneRun_fields p cf sch.k0 zl.msgs (head p cf sch.n (init T) (startZ cap T t)).1
      (head p cf sch.n (init T) (startZ cap T t)).2 j (by omega)
    simp only [head_fields, init_fields] at hf
    refine (Runs.sampleK hs cf _ _).bind ((Runs.modify _ _ _).bind ?_)
    by_cases hl : j + 1 = sch.k0
    · refine Runs.ite_pos (by simp [hl]) ?_
      have hr : Zk.rhoIndependent (laneRun p cf sch.k0 zl.msgs (head p cf sch.n (init T) (startZ cap T t)).1
          (head p cf sch.n (init T) (startZ cap T t)).2 (j + 1)).1.2 = true := by rw [hl]; exact hrho
      rw [show laneRun p cf sch.k0 zl.msgs (head p cf sch.n (init T) (startZ cap T t)).1
          (head p cf sch.n (init T) (startZ cap T t)).2 (j + 1) = laneStep p cf sch.k0 zl.msgs j
          (laneRun p cf sch.k0 zl.msgs (head p cf sch.n (init T) (startZ cap T t)).1
            (head p cf sch.n (init T) (startZ cap T t)).2 j) from rfl, laneStep, ite_eq_left hl] at hr
      refine (Runs.absorbK _ _ _).bind ((Runs.absorbK _ _ _).bind ((Runs.sampleK hs cf _ _).bind
        ((Runs.sampleK hs cf _ _).bind ?_)))
      refine Runs.ite_neg (by simp only [Bool.not_eq_true', Bool.not_eq_false]; exact hr) ?_
      refine (Runs.label _ _ _).bind ((Runs.absorbK _ _ _).bind ((Runs.absorbSlice _ _ _).bind
        ((Runs.modify _ _ _).bind ((Runs.nextMsg p _ _ (by simp only [hf.1]; omega)).bind
        ((Runs.modify _ _ _).bind ?_)))))
      show Runs _ _ _ _ _ (ForInStep.yield (laneStep p cf sch.k0 zl.msgs j
          (laneRun p cf sch.k0 zl.msgs (head p cf sch.n (init T) (startZ cap T t)).1
            (head p cf sch.n (init T) (startZ cap T t)).2 j)).1)
        (laneStep p cf sch.k0 zl.msgs j (laneRun p cf sch.k0 zl.msgs (head p cf sch.n (init T) (startZ cap T t)).1
            (head p cf sch.n (init T) (startZ cap T t)).2 j)).2.1
        (laneStep p cf sch.k0 zl.msgs j (laneRun p cf sch.k0 zl.msgs (head p cf sch.n (init T) (startZ cap T t)).1
            (head p cf sch.n (init T) (startZ cap T t)).2 j)).2.2
      rw [laneStep, ite_eq_left hl]
      exact Runs.pure _ _ _
    · refine Runs.ite_neg (by simp [hl]) ?_
      refine (Runs.nextMsg p _ _ (by simp only [hf.1]; omega)).bind ((Runs.modify _ _ _).bind ?_)
      show Runs _ _ _ _ _ (ForInStep.yield (laneStep p cf sch.k0 zl.msgs j
          (laneRun p cf sch.k0 zl.msgs (head p cf sch.n (init T) (startZ cap T t)).1
            (head p cf sch.n (init T) (startZ cap T t)).2 j)).1)
        (laneStep p cf sch.k0 zl.msgs j (laneRun p cf sch.k0 zl.msgs (head p cf sch.n (init T) (startZ cap T t)).1
            (head p cf sch.n (init T) (startZ cap T t)).2 j)).2.1
        (laneStep p cf sch.k0 zl.msgs j (laneRun p cf sch.k0 zl.msgs (head p cf sch.n (init T) (startZ cap T t)).1
            (head p cf sch.n (init T) (startZ cap T t)).2 j)).2.2
      rw [laneStep, ite_eq_right hl]
      exact Runs.pure _ _ _
  refine (Runs.absorbBytes _ _ _).bind ((Runs.oodIntro hs p cf (Nat.succ_ne_zero _) 0 _ _ (by simp [hf0]; omega)
    (by simp [hf0]; omega) (by simp [hf0]; omega)).bind ((Runs.oodIntro hs p cf (Nat.succ_ne_zero _) 0 _ _
    (by simp [hf0]; omega) (by simp [hf0]; omega) (by simp [hf0]; omega)).bind
    ((Runs.queryLevel0 hs ms p cf _ cap _ _ _ (hq 0 (by omega)) (by simp [hf0]; omega) (by simp [hf0]; omega)
    (hmo.at0 (by rw [qOff, ite_eq_left (by omega)]; simp [hf0]; omega))).bind ?_)))
  refine (Runs.forInPure _ _ _ _ _).bind ?_
  refine (Runs.nextMsg p _ _ (by simp [hf0]; omega)).bind ((Runs.claimPow p _ _ (by simp [hf0]; omega)).bind
    ((Runs.sample hs cf _ _).bind ((Runs.modify _ _ _).bind ?_)))
  refine Runs.start (st₀ := (x₀Z p cf sch T cap zl t).2.1) (t₀ := (x₀Z p cf sch T cap zl t).2.2) rfl rfl ?_
  rw [FlockSoundness.Refine.lforIn_range_eq]
  have hI0 := inv_x₀Z p cf sch T cap zl t hL hk0
  refine Runs.bind (Runs.loop _ (fun j => (loop p cf sch (x₀Z p cf sch T cap zl t) j).1)
    (fun j => (loop p cf sch (x₀Z p cf sch T cap zl t) j).2.1)
    (fun j => (loop p cf sch (x₀Z p cf sch T cap zl t) j).2.2) (sch.levels.size - 1) 0 fun j hj => ?_) ?_
  · have hI := inv_loop_of p cf sch hI0 hsd j hj
    rw [Nat.zero_add]
    refine Runs.ite_neg (by rw [hI.sd]; omega) ?_
    rw [FlockSoundness.Refine.lforIn_range_eq]
    refine (Runs.challenges hs p cf #[] _ _ 4 (by rw [hI.msg]; omega)).bind ?_
    by_cases hj1 : j + 1 < sch.levels.size - 1
    · rw [loop_stepN p cf sch _ hj1]
      refine Runs.ite_pos hj1 ?_
      refine (Runs.absorbBytes _ _ _).bind ((Runs.oodIntro hs p cf (Nat.succ_ne_zero _) (j + 1) _ _
        (by simp [hI.ood]; omega) (by simp [hI.msg]; omega) (by simp [hI.cbn]; omega)).bind
        ((Runs.oodIntro hs p cf (Nat.succ_ne_zero _) (j + 1) _ _ (by simp [hI.ood]; omega) (by simp [hI.msg]; omega)
        (by simp [hI.cbn]; omega)).bind ((Runs.queryLevel hs ms p cf _ _ _ _ _ _ _ _ (hq (j + 1) (by omega))
        (by simp [hI.qn]; omega) (by simp [hI.cn]; omega)
        (hmo.at (j := j + 1) (by omega) (by omega) (by rw [qOff, ite_eq_left hj1]; simp [hI.k]; omega)
          (by rw [opening, ite_eq_right (Nat.succ_ne_zero j), ite_eq_left hj1, hI.cap]; rfl))).bind
        ((Runs.nextMsg p _ _ (by simp [hI.msg]; omega)).bind ((Runs.claimPow p _ _ (by simp [hI.cbn]; omega)).bind
        ((Runs.sample hs cf _ _).bind ((Runs.modify _ _ _).bind
        (Runs.ite_neg (by rw [hI.sd]; omega) (Runs.pure _ _ _)))))))))
    · have hjR : j + 1 = sch.levels.size - 1 := by omega
      rw [loop_stepL p cf sch _ hj1]
      refine Runs.ite_neg hj1 ((Runs.get _ _).bind (Runs.ite_neg (by simp [hI.msg]; omega)
        (Runs.ite_neg (by simp [hI.sd, hyr]; omega) ?_)))
      refine (Runs.absorbAll _ _ _).bind ((Runs.queryLevel hs ms p cf _ _ _ _ _ _ _ _ (hq _ (by omega))
        (by simp [hI.qn]; omega) (by simp [hI.cn]; omega)
        (hmo.at (j := sch.levels.size - 1) (by omega) (by omega)
          (by rw [qOff, ite_eq_right (by omega)]; simp [hI.k]; omega)
          (by rw [opening, ite_eq_right (by omega), ite_eq_right (by omega), hI.cap, ← hjR]; rfl))).bind
        ((Runs.claimPow p _ _ (by simp [hI.cbn]; omega)).bind ((Runs.sample hs cf _ _).bind
        ((Runs.modify _ _ _).bind (Runs.pure _ _ _)))))
  · have hD : Done p sch (t.k + 2) (outZ p cf sch T cap zl t) := done_of p cf sch hI0 hL hsd
    have hpre : (preZ p cf sch T cap zl t).1.1.size = sch.k0 := hf0.2.2.2.2.2.2.1
    refine Runs.stay rfl rfl fun R' => ?_
    have hfin' := finalProgZ_of (p := p) (basisAt := basisAt) (n := sch.n) (T := T)
      (initCh := (preZ p cf sch T cap zl t).1.1) (rho := (preZ p cf sch T cap zl t).1.2)
      (levels := (outZ p cf sch T cap zl t).1.2.2.1)
      (ext := (outZ p cf sch T cap zl t).1.2.2.2) (st := (outZ p cf sch T cap zl t).2.1)
      (tr := on R' (outZ p cf sch T cap zl t).2.2)
      (by rw [origE_size _ _ hD.lv4, hD.lvs, hD.ext, hpre]; omega) (hb _ _) ?_ ?_ hfin
      ⟨by rw [hD.ood, hood], by rw [hD.qn, hqn], by rw [hD.cn, hcn], by rw [hD.cbn, hcbn]⟩
    · exact hfin'
    · intro o ho
      have := hD.oods o ho
      unfold FlockSoundness.Refine.oodFixed OodDim at *
      split at this
      · rename_i hs
        rw [hs, origE_size _ _ hD.lv4, hD.lvs, hD.ext, hpre]
        omega
      · rename_i s hs
        rw [hs, oodFixed_size _ hD.lv4 (by rw [hD.lvs]; exact this.1), hD.lvs, hD.ext]
        omega
    · intro cs hc
      have := hD.conss cs hc
      unfold ConsDim at this
      rw [consFixed_size _ hD.lv4, hD.lvs, hD.ext]
      omega

end FlockComplete.L0
