import Proofs.Flock.Claims.Complete.Replay.LigeritoModel

/-!
# The verifier's Ligerito returns on the model's rounds

`ligeritoWith_of`: from a transcript that has recorded its rounds, on rounds that carry the model's (`out`), the
verifier's Ligerito (`ligeritoWith`) returns where the model's transcript ends, given the proof's list sizes, fast100's
dimensions, each level's Merkle check at its query coins (`MerkleOk`), and the final identity at the model's values.

* **The model** (`out`): level 0's out-of-domain sample and first message (`head`), the `k0` challenges, level 0's cap,
  intros and query (`lv0`), then the level loop (`loop`): an inner level's step (`stepN`) or the last's (`stepL`).
* **Its cursors** (`Inv`, `inv_loop`): before step `j` the split dimension is `n − k0 + 1 − 3j`, the previous cap is
  `recursiveCaps[j]`, the cursors are `k0 + 4 + 7j` messages, `3 + 2j` out-of-domain values, `4 + 3j` claim nonces and
  `1 + j` query and consistency nonces, the transcript is `k0 + 9 + 11j` rounds on, and every context has its level's
  dimensions. At the end (`Done`, `done_out`) every list is read to its end and the transcript is `k0 + 5 + 11R` rounds
  on, `R + 1` levels.
* **The Merkle openings** (`MerkleOk`): level `j`'s query coins are those of round `k + qOff sch j`, its cap and opened
  rows `opening`'s.
* **The final check** (`finalProg_of`): its dimension checks from the contexts' (`origE_size`, `oodFixed_size`,
  `consFixed_size`), its value check the hypothesis `hfin`, and its leftover check from `Done`.
-/

namespace FlockComplete.Replay

open Flock Transcript

namespace Lg

/-- `cap`'s digests, concatenated: the bytes the verifier absorbs for a cap. -/
def capBytes (cap : Array ByteArray) : ByteArray := cap.foldl (· ++ ·) ByteArray.empty

/-- After a level's query `r`: its message `m`, the claim nonce and the batching coin `b`, folded into the claim, and the
level's consistency context of `cols` columns from level `start` recorded. -/
def close (p : LigeritoProof) (cf : ℕ → ℕ → F128) (cols start : ℕ) (r : Array ℕ × Array F128 × F256) (st : LState)
    (t : Transcript) : LState × Transcript :=
  let m := p.messages[st.msg]!
  let t₁ := ap (pN p.claimNonces[st.cbn]!) (ap (pK m) t)
  let b := cf t₁.k 0
  ({ { { st with msg := st.msg + 1 } with cbn := st.cbn + 1 } with
      q := st.q.addScaled (Poly.ofMsg m r.2.2) b, s := st.s + r.2.2.smul b,
      conss := st.conss.push ⟨cols, r.1, r.2.1, b, start⟩ },
    afterSq t₁ 1 1 (cf t₁.k))

/-- After the last level's query `r`: the claim nonce and the batching coin `b`, folded into the sum. -/
def closeLast (p : LigeritoProof) (cf : ℕ → ℕ → F128) (cols start : ℕ) (r : Array ℕ × Array F128 × F256)
    (st : LState) (t : Transcript) : LState × Transcript :=
  let t₁ := ap (pN p.claimNonces[st.cbn]!) t
  let b := cf t₁.k 0
  ({ { st with cbn := st.cbn + 1 } with
      s := st.s + r.2.2.smul b, conss := st.conss.push ⟨cols, r.1, r.2.1, b, start⟩ },
    afterSq t₁ 1 1 (cf t₁.k))

/-- Level 0's out-of-domain sample (no message) and the first sumcheck message, from `(st, t)`. -/
def head (p : LigeritoProof) (cf : ℕ → ℕ → F128) (n : ℕ) (st : LState) (t : Transcript) : LState × Transcript :=
  let z := vec cf t n
  let t₁ := afterSq t 2 n (cf t.k)
  let y := p.oodValues[st.ood]!
  let t₂ := ap (pN p.claimNonces[st.cbn]!) (ap (pF y) t₁)
  let b := cf t₂.k 0
  let s := st.s + (F256.ofF y).smul b
  let m := p.messages[st.msg]!
  ({ { { { { st with ood := st.ood + 1 } with cbn := st.cbn + 1 } with
      s := s, oods := st.oods.push ⟨z, b, none⟩ } with msg := st.msg + 1 } with q := Poly.ofMsg m s },
    ap (pK m) (afterSq t₂ 1 1 (cf t₂.k)))

/-- Level 0 after its `k0` challenges `initCh`: level 1's cap and two out-of-domain intros, level 0's query, closed. -/
def lv0 (p : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (initCh : Array F256) (st : LState)
    (t : Transcript) : LState × Transcript :=
  let x₁ := ood p cf (sch.n - sch.k0 + 1) 0 st (ap (pBytes (capBytes p.recursiveCaps[0]!)) t)
  let x₂ := ood p cf (sch.n - sch.k0 + 1) 0 x₁.1 x₁.2
  let qr := query p cf (sch.levels.getD 0 default) p.initialProof.rows initCh x₂.1 x₂.2
  close p cf (sch.n - sch.k0) 0 qr.1 qr.2.1 qr.2.2

/-- The level loop's own variables: the split dimension, the previous cap, the levels' challenges and `ext`. -/
abbrev Vars := ℕ × Array ByteArray × Array (Array F256) × ℕ

/-- **An inner level's step `i`**: four challenges, the next level's cap and two out-of-domain intros, and its query
against the previous cap, closed. -/
def stepN (p : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (i : ℕ) (x : Vars × LState × Transcript) :
    Vars × LState × Transcript :=
  let c := chals p cf #[] x.2.1 x.2.2 4
  let ext := x.1.1 - 4
  let capN := p.recursiveCaps[i + 1]!
  let x₁ := ood p cf (ext + 1) (i + 1) c.2.1 (ap (pBytes (capBytes capN)) c.2.2)
  let x₂ := ood p cf (ext + 1) (i + 1) x₁.1 x₁.2
  let qr := query p cf (sch.levels.getD (i + 1) default) p.recursiveProofs[i]!.rows c.1 x₂.1 x₂.2
  let y := close p cf ext (i + 1) qr.1 qr.2.1 qr.2.2
  ((ext + 1, capN, x.1.2.2.1.push c.1, ext), y.1, y.2)

/-- **The last level's step**: four challenges, `y_r` absorbed, and the last query against the previous cap, closed. -/
def stepL (p : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (x : Vars × LState × Transcript) :
    Vars × LState × Transcript :=
  let c := chals p cf #[] x.2.1 x.2.2 4
  let ext := x.1.1 - 4
  let qr := query p cf (sch.levels.getD (sch.levels.size - 1) default) p.finalProof.rows c.1 c.2.1
    (ap (fun P => p.finalProof.yr.foldl (fun P w => pF w P) P) c.2.2)
  let y := closeLast p cf ext (sch.levels.size - 1) qr.1 qr.2.1 qr.2.2
  ((x.1.1, x.1.2.1, x.1.2.2.1.push c.1, ext), y.1, y.2)

/-- **The level loop's step `i`**: an inner level's, or the last level's. -/
def step (p : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (i : ℕ) (x : Vars × LState × Transcript) :
    Vars × LState × Transcript :=
  if i + 1 < sch.levels.size - 1 then stepN p cf sch i x else stepL p cf sch x

/-- The level loop after `j` steps from `x`. -/
def loop (p : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (x : Vars × LState × Transcript) :
    ℕ → Vars × LState × Transcript
  | 0 => x
  | j + 1 => step p cf sch j (loop p cf sch x j)

/-- The verifier's Ligerito state at the start, on the batched claim `T`. -/
def init (T : F128) : LState := ⟨F256.ofF T, ⟨0, 0, 0⟩, 0, 0, 0, 0, 0, #[], #[]⟩

/-- The transcript once the verifier has absorbed its label, `T` and the level-0 cap. -/
def start (cap : Array ByteArray) (T : F128) (t : Transcript) : Transcript :=
  ap (pBytes (capBytes cap)) (ap (pF T) (ap (pLabel "flock-ligerito-basis-f256-split-v0") t))

/-- Up to the level loop: the `k0` challenges `initCh`, and the state and transcript after level 0. -/
def pre (p : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (T : F128) (cap : Array ByteArray)
    (t : Transcript) : Array F256 × LState × Transcript :=
  let h := head p cf sch.n (init T) (start cap T t)
  let c := chals p cf #[] h.1 h.2 sch.k0
  let y := lv0 p cf sch c.1 c.2.1 c.2.2
  (c.1, y.1, y.2)

/-- The level loop's start: its variables, and the state and transcript after level 0. -/
def x₀ (p : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (T : F128) (cap : Array ByteArray)
    (t : Transcript) : Vars × LState × Transcript :=
  ((sch.n - sch.k0 + 1, p.recursiveCaps[0]!, #[], 0), (pre p cf sch T cap t).2.1, (pre p cf sch T cap t).2.2)

/-- **The verifier's Ligerito, in closed form**: the level loop's variables, the state and the transcript at its end. -/
def out (p : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (T : F128) (cap : Array ByteArray)
    (t : Transcript) : Vars × LState × Transcript :=
  loop p cf sch (x₀ p cf sch T cap t) (sch.levels.size - 1)

/-! ## The model's cursors, positions and contexts, read off each step -/

@[simp] theorem ap_k (f : ByteArray → ByteArray) (t : Transcript) : (ap f t).k = t.k := rfl

@[simp] theorem afterSq_k (t : Transcript) (kind : UInt8) (n : ℕ) (coin : ℕ → F128) :
    (afterSq t kind n coin).k = t.k + 1 := rfl

@[simp] theorem chals_fields (p : LigeritoProof) (cf : ℕ → ℕ → F128) (acc : Array F256) (st : LState)
    (t : Transcript) (j : ℕ) :
    (chals p cf acc st t j).2.1.msg = st.msg + j ∧ (chals p cf acc st t j).2.1.ood = st.ood ∧
      (chals p cf acc st t j).2.1.cbn = st.cbn ∧ (chals p cf acc st t j).2.1.qn = st.qn ∧
      (chals p cf acc st t j).2.1.cn = st.cn ∧ (chals p cf acc st t j).2.2.k = t.k + j ∧
      (chals p cf acc st t j).1.size = acc.size + j ∧ (chals p cf acc st t j).2.1.oods = st.oods ∧
      (chals p cf acc st t j).2.1.conss = st.conss := by
  induction j with
  | zero => exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩
  | succ j ih =>
    obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9⟩ := ih
    simp only [chals, chal, ap, afterSq, Array.size_push, h1, h2, h3, h4, h5, h6, h7, h8, h9, true_and, and_true]
    omega

@[simp] theorem ood_fields (p : LigeritoProof) (cf : ℕ → ℕ → F128) (dim split : ℕ) (st : LState) (t : Transcript) :
    (ood p cf dim split st t).1.msg = st.msg + 1 ∧ (ood p cf dim split st t).1.ood = st.ood + 1 ∧
      (ood p cf dim split st t).1.cbn = st.cbn + 1 ∧ (ood p cf dim split st t).1.qn = st.qn ∧
      (ood p cf dim split st t).1.cn = st.cn ∧ (ood p cf dim split st t).2.k = t.k + 2 ∧
      (ood p cf dim split st t).1.conss = st.conss ∧
      (ood p cf dim split st t).1.oods = st.oods.push ⟨vec cf t dim, cf (t.k + 1) 0, some split⟩ :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

@[simp] theorem query_fields (p : LigeritoProof) (cf : ℕ → ℕ → F128) (lv : Flock.Level) (rows : Array (Array F128))
    (fold : Array F256) (st : LState) (t : Transcript) :
    (query p cf lv rows fold st t).2.1.msg = st.msg ∧ (query p cf lv rows fold st t).2.1.ood = st.ood ∧
      (query p cf lv rows fold st t).2.1.cbn = st.cbn ∧ (query p cf lv rows fold st t).2.1.qn = st.qn + 1 ∧
      (query p cf lv rows fold st t).2.1.cn = st.cn + 1 ∧ (query p cf lv rows fold st t).2.2.k = t.k + 2 ∧
      (query p cf lv rows fold st t).2.1.conss = st.conss ∧ (query p cf lv rows fold st t).2.1.oods = st.oods :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

@[simp] theorem close_fields (p : LigeritoProof) (cf : ℕ → ℕ → F128) (cols start : ℕ)
    (r : Array ℕ × Array F128 × F256) (st : LState) (t : Transcript) :
    (close p cf cols start r st t).1.msg = st.msg + 1 ∧ (close p cf cols start r st t).1.ood = st.ood ∧
      (close p cf cols start r st t).1.cbn = st.cbn + 1 ∧ (close p cf cols start r st t).1.qn = st.qn ∧
      (close p cf cols start r st t).1.cn = st.cn ∧ (close p cf cols start r st t).2.k = t.k + 1 ∧
      (close p cf cols start r st t).1.oods = st.oods ∧
      (close p cf cols start r st t).1.conss = st.conss.push ⟨cols, r.1, r.2.1, cf t.k 0, start⟩ :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

@[simp] theorem closeLast_fields (p : LigeritoProof) (cf : ℕ → ℕ → F128) (cols start : ℕ)
    (r : Array ℕ × Array F128 × F256) (st : LState) (t : Transcript) :
    (closeLast p cf cols start r st t).1.msg = st.msg ∧ (closeLast p cf cols start r st t).1.ood = st.ood ∧
      (closeLast p cf cols start r st t).1.cbn = st.cbn + 1 ∧ (closeLast p cf cols start r st t).1.qn = st.qn ∧
      (closeLast p cf cols start r st t).1.cn = st.cn ∧ (closeLast p cf cols start r st t).2.k = t.k + 1 ∧
      (closeLast p cf cols start r st t).1.oods = st.oods ∧
      (closeLast p cf cols start r st t).1.conss = st.conss.push ⟨cols, r.1, r.2.1, cf t.k 0, start⟩ :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

@[simp] theorem head_fields (p : LigeritoProof) (cf : ℕ → ℕ → F128) (n : ℕ) (st : LState) (t : Transcript) :
    (head p cf n st t).1.msg = st.msg + 1 ∧ (head p cf n st t).1.ood = st.ood + 1 ∧
      (head p cf n st t).1.cbn = st.cbn + 1 ∧ (head p cf n st t).1.qn = st.qn ∧ (head p cf n st t).1.cn = st.cn ∧
      (head p cf n st t).2.k = t.k + 2 ∧ (head p cf n st t).1.conss = st.conss ∧
      (head p cf n st t).1.oods = st.oods.push ⟨vec cf t n, cf (t.k + 1) 0, none⟩ :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

@[simp] theorem init_fields (T : F128) :
    (init T).msg = 0 ∧ (init T).ood = 0 ∧ (init T).cbn = 0 ∧ (init T).qn = 0 ∧ (init T).cn = 0 ∧
      (init T).oods = #[] ∧ (init T).conss = #[] :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

@[simp] theorem start_k (cap : Array ByteArray) (T : F128) (t : Transcript) : (start cap T t).k = t.k := rfl

@[simp] theorem vec_size (cf : ℕ → ℕ → F128) (t : Transcript) (n : ℕ) : (vec cf t n).size = n := by
  simp [vec]

/-! ## The level loop's invariant -/

/-- An out-of-domain sample's dimension: level 0's first, all `n` coordinates; one at level `s < R`, the split
dimension there. -/
def OodDim (sch : Schedule) (o : OodCtx) : Prop :=
  match o.split with
  | none => o.z.size = sch.n
  | some s => s < sch.levels.size - 1 ∧ o.z.size = sch.n - sch.k0 + 1 - 3 * s

/-- A consistency context's columns: level `start ≤ R`'s, `3` fewer per level. -/
def ConsDim (sch : Schedule) (cs : ConsCtx) : Prop :=
  cs.start ≤ sch.levels.size - 1 ∧ cs.cols = sch.n - sch.k0 - 3 * cs.start

/-- **The level loop before step `j`**, from round `k`: the split dimension and the previous cap, `j` levels of four
challenges, the cursors and the position, and every context's dimensions. -/
structure Inv (p : LigeritoProof) (sch : Schedule) (k j : ℕ) (x : Vars × LState × Transcript) : Prop where
  sd : x.1.1 = sch.n - sch.k0 + 1 - 3 * j
  cap : x.1.2.1 = p.recursiveCaps[j]!
  lvs : x.1.2.2.1.size = j
  lv4 : ∀ l ∈ x.1.2.2.1.toList, l.size = 4
  msg : x.2.1.msg = sch.k0 + 4 + 7 * j
  ood : x.2.1.ood = 3 + 2 * j
  cbn : x.2.1.cbn = 4 + 3 * j
  qn : x.2.1.qn = 1 + j
  cn : x.2.1.cn = 1 + j
  k : x.2.2.k = k + sch.k0 + 9 + 11 * j
  oods : ∀ o ∈ x.2.1.oods.toList, OodDim sch o
  conss : ∀ cs ∈ x.2.1.conss.toList, ConsDim sch cs

/-- **The level loop done**, from round `k`: `R` levels of four challenges, the last `ext`, the cursors at the lists'
ends, the position, and every context's dimensions. -/
structure Done (p : LigeritoProof) (sch : Schedule) (k : ℕ) (x : Vars × LState × Transcript) : Prop where
  lvs : x.1.2.2.1.size = sch.levels.size - 1
  lv4 : ∀ l ∈ x.1.2.2.1.toList, l.size = 4
  ext : x.1.2.2.2 = sch.n - sch.k0 - 3 * (sch.levels.size - 1)
  ood : x.2.1.ood = 2 * (sch.levels.size - 1) + 1
  cbn : x.2.1.cbn = 3 * (sch.levels.size - 1) + 2
  qn : x.2.1.qn = sch.levels.size
  cn : x.2.1.cn = sch.levels.size
  k : x.2.2.k = k + sch.k0 + 5 + 11 * (sch.levels.size - 1)
  oods : ∀ o ∈ x.2.1.oods.toList, OodDim sch o
  conss : ∀ cs ∈ x.2.1.conss.toList, ConsDim sch cs

theorem inv_x₀ (p : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (T : F128) (cap : Array ByteArray)
    (t : Transcript) (hL : 2 ≤ sch.levels.size) : Inv p sch t.k 0 (x₀ p cf sch T cap t) where
  sd := rfl
  cap := rfl
  lvs := rfl
  lv4 := by simp [x₀]
  msg := by simp [x₀, pre, lv0]; omega
  ood := by simp [x₀, pre, lv0]
  cbn := by simp [x₀, pre, lv0]
  qn := by simp [x₀, pre, lv0]
  cn := by simp [x₀, pre, lv0]
  k := by simp [x₀, pre, lv0]; omega
  oods := by
    simp only [x₀, pre, lv0, close_fields, query_fields, ood_fields, chals_fields, head_fields, init_fields,
      Array.toList_push, List.nil_append, List.mem_append, List.mem_cons, List.not_mem_nil, or_false]
    rintro o ((rfl | rfl) | rfl)
    · simp [OodDim]
    · simp [OodDim]; omega
    · simp [OodDim]; omega
  conss := by
    simp only [x₀, pre, lv0, close_fields, query_fields, ood_fields, chals_fields, head_fields, init_fields,
      Array.toList_push, List.nil_append, List.mem_singleton]
    rintro cs rfl
    simp [ConsDim]

theorem Inv.next {p : LigeritoProof} {cf : ℕ → ℕ → F128} {sch : Schedule} {k j : ℕ}
    {x : Vars × LState × Transcript} (h : Inv p sch k j x) (hj : j + 1 < sch.levels.size - 1)
    (hsd : sch.k0 + 3 * (sch.levels.size - 1) ≤ sch.n) : Inv p sch k (j + 1) (stepN p cf sch j x) where
  sd := by simp only [stepN, h.sd]; omega
  cap := rfl
  lvs := by simp [stepN, h.lvs]
  lv4 := by
    simp only [stepN, Array.toList_push, List.mem_append, List.mem_singleton]
    rintro l (hl | rfl)
    · exact h.lv4 l hl
    · simp
  msg := by simp [stepN, h.msg]; omega
  ood := by simp [stepN, h.ood]; omega
  cbn := by simp [stepN, h.cbn]; omega
  qn := by simp [stepN, h.qn]; omega
  cn := by simp [stepN, h.cn]; omega
  k := by simp [stepN, h.k]; omega
  oods := by
    simp only [stepN, close_fields, query_fields, ood_fields, chals_fields, Array.toList_push, List.mem_append,
      List.mem_singleton]
    rintro o ((ho | rfl) | rfl)
    · exact h.oods o ho
    · simp only [OodDim, vec_size, h.sd]; omega
    · simp only [OodDim, vec_size, h.sd]; omega
  conss := by
    simp only [stepN, close_fields, query_fields, ood_fields, chals_fields, Array.toList_push, List.mem_append,
      List.mem_singleton]
    rintro cs (hc | rfl)
    · exact h.conss cs hc
    · simp only [ConsDim, h.sd]; omega

theorem Inv.last {p : LigeritoProof} {cf : ℕ → ℕ → F128} {sch : Schedule} {k j : ℕ}
    {x : Vars × LState × Transcript} (h : Inv p sch k j x) (hj : j + 1 = sch.levels.size - 1)
    (hsd : sch.k0 + 3 * (sch.levels.size - 1) ≤ sch.n) : Done p sch k (stepL p cf sch x) where
  lvs := by simp [stepL, h.lvs]; omega
  lv4 := by
    simp only [stepL, Array.toList_push, List.mem_append, List.mem_singleton]
    rintro l (hl | rfl)
    · exact h.lv4 l hl
    · simp
  ext := by simp only [stepL, h.sd]; omega
  ood := by simp [stepL, h.ood]; omega
  cbn := by simp [stepL, h.cbn]; omega
  qn := by simp [stepL, h.qn]; omega
  cn := by simp [stepL, h.cn]; omega
  k := by simp [stepL, h.k]; omega
  oods := by
    simp only [stepL, closeLast_fields, query_fields, chals_fields]
    exact h.oods
  conss := by
    simp only [stepL, closeLast_fields, query_fields, chals_fields, Array.toList_push, List.mem_append,
      List.mem_singleton]
    rintro cs (hc | rfl)
    · exact h.conss cs hc
    · simp only [ConsDim, h.sd]; omega

theorem loop_stepN (p : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (x : Vars × LState × Transcript) {j : ℕ}
    (hj : j + 1 < sch.levels.size - 1) : loop p cf sch x (j + 1) = stepN p cf sch j (loop p cf sch x j) := by
  simp only [loop, step, ite_eq_left hj]

theorem loop_stepL (p : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (x : Vars × LState × Transcript) {j : ℕ}
    (hj : ¬ j + 1 < sch.levels.size - 1) : loop p cf sch x (j + 1) = stepL p cf sch (loop p cf sch x j) := by
  simp only [loop, step, ite_eq_right hj]

theorem inv_loop (p : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (T : F128) (cap : Array ByteArray)
    (t : Transcript) (hL : 2 ≤ sch.levels.size) (hsd : sch.k0 + 3 * (sch.levels.size - 1) ≤ sch.n) :
    ∀ j, j < sch.levels.size - 1 → Inv p sch t.k j (loop p cf sch (x₀ p cf sch T cap t) j)
  | 0, _ => inv_x₀ p cf sch T cap t hL
  | j + 1, hj => by
    rw [loop_stepN p cf sch _ hj]
    exact (inv_loop p cf sch T cap t hL hsd j (by omega)).next hj hsd

theorem done_out (p : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (T : F128) (cap : Array ByteArray)
    (t : Transcript) (hL : 2 ≤ sch.levels.size) (hsd : sch.k0 + 3 * (sch.levels.size - 1) ≤ sch.n) :
    Done p sch t.k (out p cf sch T cap t) := by
  unfold out
  obtain ⟨j, hj⟩ : ∃ j, sch.levels.size - 1 = j + 1 := ⟨sch.levels.size - 2, by omega⟩
  rw [hj, loop_stepL p cf sch _ (by omega)]
  exact (inv_loop p cf sch T cap t hL hsd j (by omega)).last (by omega) hsd

/-! ## The final check's dimensions -/

theorem tails_length (L : List (Array F256)) (h : ∀ l ∈ L, l.size = 4) :
    (L.map fun l => l.toList.tail).flatten.length = 3 * L.length := by
  induction L with
  | nil => rfl
  | cons l L ih =>
    simp only [List.map_cons, List.flatten_cons, List.length_append, List.length_tail, Array.length_toList,
      List.length_cons, h l (List.mem_cons_self ..), ih fun l hl => h l (List.mem_cons_of_mem _ hl)]
    omega

theorem origE_size (initCh : Array F256) (levels : Array (Array F256)) (h : ∀ l ∈ levels.toList, l.size = 4) :
    (FlockSoundness.Refine.origE initCh levels).size = initCh.size + 3 * levels.size := by
  rw [← Array.length_toList, FlockSoundness.Refine.origE, ← Array.foldl_toList, FlockSoundness.Refine.foldl_tails,
    List.length_append, tails_length _ h, Array.length_toList, Array.length_toList]

theorem consFixed_size (levels : Array (Array F256)) (h : ∀ l ∈ levels.toList, l.size = 4) (s : ℕ) :
    (FlockSoundness.Refine.consFixed levels s).size = 3 * (levels.size - s) := by
  rw [← Array.length_toList, FlockSoundness.Refine.consFixed, ← Array.foldl_toList,
    FlockSoundness.Refine.foldl_tails, FlockSoundness.Refine.extract_toList, List.length_append,
    tails_length _ fun l hl => h l (List.mem_of_mem_drop hl)]
  simp

theorem oodFixed_size (levels : Array (Array F256)) (h : ∀ l ∈ levels.toList, l.size = 4) {s : ℕ}
    (hs : s < levels.size) :
    ((levels.extract s levels.size).foldl (fun acc l => if acc.isEmpty then l else acc ++ l.extract 1 l.size)
      #[]).size = 3 * (levels.size - s) + 1 := by
  have hs' : s < levels.toList.length := by simpa using hs
  have h4 : levels[s].size = 4 := by rw [← Array.getElem_toList]; exact h _ (List.getElem_mem hs')
  rw [← Array.length_toList, ← Array.foldl_toList, FlockSoundness.Refine.extract_toList,
    List.drop_eq_getElem_cons hs', List.foldl_cons]
  simp only [Array.isEmpty_empty, ite_true]
  rw [FlockSoundness.Refine.foldl_tails' _ _ (by unfold Array.isEmpty; simp [h4]), List.length_append,
    tails_length _ fun l hl => h l (List.mem_of_mem_drop hl), List.length_drop, Array.length_toList,
    Array.getElem_toList, h4]
  simp only [Array.length_toList]
  omega

/-! ## The Merkle openings -/

/-- Where level `j`'s query coins are drawn, counted from Ligerito's first round: level 0's after `k0 + 6` rounds, an
inner level's eleven later per level, and the last level's four rounds sooner than an inner level's would be. -/
def qOff (sch : Schedule) (j : ℕ) : ℕ :=
  if j < sch.levels.size - 1 then sch.k0 + 6 + 11 * j else sch.k0 + 2 + 11 * j

/-- The cap level `j`'s query opens against, and its opened rows, siblings and salts, of `R + 1` levels. -/
def opening (p : LigeritoProof) (cap : Array ByteArray) (R j : ℕ) :
    Array ByteArray × Array (Array F128) × Array ByteArray × Array ByteArray :=
  if j = 0 then (cap, p.initialProof.rows, p.initialProof.siblings, p.initialProof.salts)
  else if j < R then
    (p.recursiveCaps[j - 1]!, p.recursiveProofs[j - 1]!.rows, p.recursiveProofs[j - 1]!.siblings,
      p.recursiveProofs[j - 1]!.salts)
  else (p.recursiveCaps[j - 1]!, p.finalProof.rows, p.finalProof.siblings, p.finalProof.salts)

/-- **Every level's Merkle check passes** at its query coins, the coins of the round `k + qOff sch j`. -/
def MerkleOk (hs : Hashes V) (ms : MerkleScheme) (sch : Schedule) (cap : Array ByteArray) (p : LigeritoProof)
    (cf : ℕ → ℕ → F128) (k : ℕ) : Prop :=
  ∀ j < sch.levels.size,
    merkleCheckWith (M := V) (ms.opsWith hs) ms (opening p cap (sch.levels.size - 1) j).1
      (sch.levels.getD j default).cap (sch.levels.getD j default).d
      (positions (sch.levels.getD j default) ((Array.range (sch.levels.getD j default).q).map (cf (k + qOff sch j))))
      (opening p cap (sch.levels.size - 1) j).2.1 (opening p cap (sch.levels.size - 1) j).2.2.1
      (opening p cap (sch.levels.size - 1) j).2.2.2 (2 ^ (sch.levels.getD j default).lanes) = .ok ()

/-- Level `j`'s Merkle check, at a query drawn at round `k + qOff sch j`. -/
theorem MerkleOk.at {hs : Hashes V} {ms : MerkleScheme} {sch : Schedule} {cap : Array ByteArray}
    {p : LigeritoProof} {cf : ℕ → ℕ → F128} {k : ℕ} (h : MerkleOk hs ms sch cap p cf k) {j : ℕ}
    (hj : j < sch.levels.size) {k' : ℕ} (hk : k' = k + qOff sch j) {c : Array ByteArray}
    {rows : Array (Array F128)} {sibs salts : Array ByteArray}
    (ho : opening p cap (sch.levels.size - 1) j = (c, rows, sibs, salts)) :
    merkleCheckWith (M := V) (ms.opsWith hs) ms c (sch.levels.getD j default).cap (sch.levels.getD j default).d
      (positions (sch.levels.getD j default) ((Array.range (sch.levels.getD j default).q).map (cf k'))) rows sibs
      salts (2 ^ (sch.levels.getD j default).lanes) = .ok () := by
  have := h j hj
  rw [ho] at this
  rw [hk]
  exact this

/-- **The verifier's Ligerito returns** on rounds `R` that carry the model's from `t.k` on: from a transcript that has
recorded its rounds, at a proof with the lengths fast100's `R + 1 ≥ 2` levels fix, whose level-0 cap is the
commitment's, whose Merkle openings pass at their query coins (`hmo`) and whose final identity holds at the model's
values (`hfin`), it ends at the model's position and pending bytes, and that position is the model's round count. -/
theorem ligeritoWith_of (hs : Hashes V) (ms : MerkleScheme) (sch : Schedule) (T : F128) (cap : Array ByteArray)
    (p : LigeritoProof) (basisAt : Array F256 → ℕ → Array F256) (cf : ℕ → ℕ → F128) (t : Transcript)
    (hk : t.k = t.rounds.size) {R : Array Round} (hR : Agrees R (out p cf sch T cap t).2.2.rounds t.k)
    (hcap : (p.initialCap == cap) = true) (hn : sch.n ≠ 0) (hL : 2 ≤ sch.levels.size)
    (hsd : sch.k0 + 3 * (sch.levels.size - 1) ≤ sch.n) (hq : ∀ j < sch.levels.size, (sch.levels.getD j default).q ≠ 0)
    (hmsg : p.messages.size = sch.k0 + 7 * (sch.levels.size - 1) + 1)
    (hood : p.oodValues.size = 2 * (sch.levels.size - 1) + 1)
    (hqn : p.queryNonces.size = sch.levels.size) (hcn : p.consistencyNonces.size = sch.levels.size)
    (hcbn : p.claimNonces.size = 3 * (sch.levels.size - 1) + 2)
    (hyr : p.finalProof.yr.size = 2 ^ (sch.n - sch.k0 - 3 * (sch.levels.size - 1) + 1))
    (hb : ∀ q e, (basisAt q e).size = 2 ^ e) (hmo : MerkleOk hs ms sch cap p cf t.k)
    (hfin : FlockSoundness.Refine.lhsE p.finalProof.yr (out p cf sch T cap t).1.2.2.2
      (FlockSoundness.Refine.resE basisAt sch.n (pre p cf sch T cap t).1 (out p cf sch T cap t).1.2.2.1
        (out p cf sch T cap t).1.2.2.2 (out p cf sch T cap t).2.1) = (out p cf sch T cap t).2.1.s) :
    (ligeritoWith hs ms sch T cap p basisAt).run ⟨R, t.k, t.pending⟩ =
      .ok ((), ⟨R, (out p cf sch T cap t).2.2.k, (out p cf sch T cap t).2.2.pending⟩) ∧
    (out p cf sch T cap t).2.2.k = (out p cf sch T cap t).2.2.rounds.size := by
  show ligeritoWith hs ms sch T cap p basisAt ⟨R, t.k, t.pending⟩ = _ ∧ _
  unfold ligeritoWith
  simp only [Flock.With.liftTM_V, Flock.With.liftLM_V]
  rw [Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.check_eq hcap _ _),
    Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.label_eq _ _),
    Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.absorbF_eq _ _),
    Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.absorbBytes_eq _ _)]
  refine (fun h : Runs (out p cf sch T cap t).2.2.rounds t.k _ (init T) (start cap T t) () (out p cf sch T cap t).2.1
      (out p cf sch T cap t).2.2 =>
    ⟨(Flock.RecAlgebra.bind_of_ok (a := ((), (out p cf sch T cap t).2.1)) (t₁ := on R (out p cf sch T cap t).2.2)
      (h.2 le_rfl hk (Prefix.refl _) R hR)).trans rfl, h.1.2.2 hk⟩) ?_
  refine Runs.run ?_
  refine (Runs.sampleVec hs cf hn _ _).bind ((Runs.nextOod p _ _ (by simp; omega)).bind
    ((Runs.absorbF _ _ _).bind ((Runs.claimPow p _ _ (by simp; omega)).bind ((Runs.sample hs cf _ _).bind
    ((Runs.modify _ _ _).bind ((Runs.nextMsg p _ _ (by simp; omega)).bind ((Runs.modify _ _ _).bind ?_)))))))
  refine Runs.start (st₀ := (head p cf sch.n (init T) (start cap T t)).1)
    (t₀ := (head p cf sch.n (init T) (start cap T t)).2) rfl rfl ?_
  rw [FlockSoundness.Refine.lforIn_range_eq]
  refine (Runs.challenges hs p cf #[] _ _ sch.k0 (by simp; omega)).bind ?_
  refine (Runs.absorbBytes _ _ _).bind ((Runs.oodIntro hs p cf (Nat.succ_ne_zero _) 0 _ _ (by simp; omega)
    (by simp; omega) (by simp; omega)).bind ((Runs.oodIntro hs p cf (Nat.succ_ne_zero _) 0 _ _ (by simp; omega)
    (by simp; omega) (by simp; omega)).bind ((Runs.queryLevel hs ms p cf _ cap _ _ _ _ _ _ (hq 0 (by omega))
    (by simp; omega) (by simp; omega) (hmo.at (j := 0) (by omega) (by rw [qOff, ite_eq_left (by omega)]; simp; omega)
    (by simp [opening]))).bind
    ((Runs.nextMsg p _ _ (by simp; omega)).bind ((Runs.claimPow p _ _ (by simp; omega)).bind
    ((Runs.sample hs cf _ _).bind ((Runs.modify _ _ _).bind ?_)))))))
  refine Runs.start (st₀ := (x₀ p cf sch T cap t).2.1) (t₀ := (x₀ p cf sch T cap t).2.2) rfl rfl ?_
  rw [FlockSoundness.Refine.lforIn_range_eq]
  refine Runs.bind (Runs.loop _ (fun j => (loop p cf sch (x₀ p cf sch T cap t) j).1)
    (fun j => (loop p cf sch (x₀ p cf sch T cap t) j).2.1) (fun j => (loop p cf sch (x₀ p cf sch T cap t) j).2.2)
    (sch.levels.size - 1) 0 fun j hj => ?_) ?_
  · have hI := inv_loop p cf sch T cap t hL hsd j hj
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
        (hmo.at (j := j + 1) (by omega) (by rw [qOff, ite_eq_left hj1]; simp [hI.k]; omega)
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
        (hmo.at (j := sch.levels.size - 1) (by omega) (by rw [qOff, ite_eq_right (by omega)]; simp [hI.k]; omega)
          (by rw [opening, ite_eq_right (by omega), ite_eq_right (by omega), hI.cap, ← hjR]; rfl))).bind
        ((Runs.claimPow p _ _ (by simp [hI.cbn]; omega)).bind ((Runs.sample hs cf _ _).bind
        ((Runs.modify _ _ _).bind (Runs.pure _ _ _)))))
  · have hD := done_out p cf sch T cap t hL hsd
    refine Runs.stay rfl rfl fun R' => ?_
    have hfin' := Flock.RecAlgebra.finalProg_of (p := p) (basisAt := basisAt) (n := sch.n)
      (initCh := (pre p cf sch T cap t).1) (levels := (out p cf sch T cap t).1.2.2.1)
      (ext := (out p cf sch T cap t).1.2.2.2) (st := (out p cf sch T cap t).2.1) (tr := on R' (out p cf sch T cap t).2.2)
      (by rw [origE_size _ _ hD.lv4, hD.lvs, hD.ext]; simp [pre]; omega) (hb _ _) ?_ ?_ hfin
      ⟨by rw [hD.ood, hood], by rw [hD.qn, hqn], by rw [hD.cn, hcn], by rw [hD.cbn, hcbn]⟩
    · exact hfin'
    · intro o ho
      have := hD.oods o ho
      unfold FlockSoundness.Refine.oodFixed OodDim at *
      split at this
      · rename_i hs
        rw [hs, origE_size _ _ hD.lv4, hD.lvs, hD.ext]
        simp [pre]
        omega
      · rename_i s hs
        rw [hs, oodFixed_size _ hD.lv4 (by rw [hD.lvs]; exact this.1), hD.lvs, hD.ext]
        omega
    · intro cs hc
      have := hD.conss cs hc
      unfold ConsDim at this
      rw [consFixed_size _ hD.lv4, hD.lvs, hD.ext]
      omega

end Lg

end FlockComplete.Replay
