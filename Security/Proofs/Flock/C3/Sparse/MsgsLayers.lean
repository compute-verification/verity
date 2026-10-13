import Proofs.Flock.C3.Sparse.MsgsRead
import Proofs.Flock.C3.Sparse.SessionAt

/-!
# The sparse phases' messages from V*'s sessions

`SparseMsgsOfLayers` from the sparse sessions at a fixed copy per rep: copy 0 of unit `un i` of session `sess i`.

* `wStart`, `phaseMsgs_congr`: message `m` of a sparse phase reads only port `m + 1`'s words, `wStart m` to
  `wStart (m + 1)` (the message ports' offsets, `mp_offsets`), so witnesses that agree there have the same message `m`.
* `msgWord`, `rdOf`: rep `i`'s message `m` read off the layers, `phaseMsgs` of the words whose bits copy 0 holds at the
  message columns.
* `bitAt_mask`: a registered bit is the same in the layer masked to a stage whose commit strings hold its row, the row
  `(u, q)` of the port `q` its key reads.
* `SparseSessionsAt`, `SparseCols`, `SparseSel` (named): `SparseSessionAt`'s copy (`SparseCopyAt`) and `SparseSrcAt` at
  the map; every message column an input column a row reads (`OpenCols`' form; the support question rec-lean decides);
  stage `c₀ + i · nS + roundOf m` selects, in `own (sess i)`, the port each column of message `m`'s words reads
  (`skOf`'s form). `sparseMsgsOfLayers_of_sessions`: the three give `SparseMsgsOfLayers` at `rdOf`.
-/

namespace FlockC3.Sparse

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled FlockC3.Compose
open FlockVBridge (wbit wbit_elem ofBits bitOf bitOf_ofBits copyVal_input)
open FlockVBridge.Decode (RegBit keyRowOf_keyInst)
open FlockVBridge.OfVStar (ReadCol rowCol)
open FlockVBridge.AccChain (eq_of_bitOf)
open FlockVBridge.Family (skOf maskOf mem_skOf_inl bitAt IoColAt bitAt_of_regBit ioColAt_of_readCol)
open FlockVBridge.Sparse (phaseMsgs vtTape msgTape msg2 msg3 uOf nR degSum tw wPre wV2 wSc wU wVt wClaim nw)

local notation "𝔽" => FlockLevel3.GF128

/-! ## The words each message reads -/

/-- Message `m`'s first word: port `m + 1`'s offset (`mp_offsets`), so message `m` reads `wStart m … wStart (m + 1)`. -/
def wStart (sh : SparseShape) (m : ℕ) : ℕ :=
  if m ≤ 6 then 64 + 2 * m else if m ≤ 7 + nR sh then wSc sh (m - 7) else if m = 8 + nR sh then wVt sh else wClaim sh

theorem degSum_le (sh : SparseShape) {a b : ℕ} (h : a ≤ b) : degSum sh a ≤ degSum sh b := by
  have := Nat.mul_le_mul_right (2 * sh.ell + 1) h
  unfold degSum
  omega

theorem lt_nw {sh : SparseShape} {m e : ℕ} (h : e < wStart sh m) : e < nw sh := by
  have hu : wU sh ≤ wVt sh := Nat.le_add_right _ _
  have hv : wVt sh ≤ wClaim sh := Nat.le_add_right _ _
  have h0 : 77 ≤ wU sh := Nat.le_add_right _ _
  refine Nat.lt_succ_of_lt (Nat.lt_of_lt_of_le h ?_)
  unfold wStart
  by_cases h1 : m ≤ 6
  · rw [ite_eq_left h1]; omega
  · rw [ite_eq_right h1]
    by_cases h2 : m ≤ 7 + nR sh
    · have : wSc sh (m - 7) ≤ wU sh := Nat.add_le_add_left (degSum_le sh (by omega)) 77
      rw [ite_eq_left h2]; omega
    · rw [ite_eq_right h2]; split <;> omega

theorem wStart_sc {sh : SparseShape} {m : ℕ} (h1 : 7 ≤ m) (h2 : m < 7 + nR sh) : wStart sh m = wSc sh (m - 7) ∧
    wStart sh (m + 1) = wSc sh (m - 7) + (2 * sh.ell + 1) + ((m - 7 + 1 - sh.nu) - (m - 7 - sh.nu)) := by
  unfold wStart
  rw [ite_eq_right (by omega), ite_eq_left (by omega), ite_eq_right (by omega), ite_eq_left (by omega),
    show m + 1 - 7 = m - 7 + 1 by omega]
  refine ⟨rfl, ?_⟩
  unfold wSc degSum
  rw [Nat.add_mul, Nat.one_mul]
  omega

theorem wStart_u (sh : SparseShape) : wStart sh (7 + nR sh) = wU sh ∧ wStart sh (7 + nR sh + 1) = wVt sh := by
  unfold wStart
  rw [ite_eq_right (by omega), ite_eq_left (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    ite_eq_left (by omega), Nat.add_sub_cancel_left]
  exact ⟨rfl, rfl⟩

theorem wStart_vt (sh : SparseShape) : wStart sh (8 + nR sh) = wVt sh ∧ wStart sh (8 + nR sh + 1) = wClaim sh := by
  unfold wStart
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left rfl, ite_eq_right (by omega),
    ite_eq_right (by omega), ite_eq_right (by omega)]
  exact ⟨rfl, rfl⟩

theorem getElem?_app {α : Type _} {l₁ l₂ l₁' l₂' : List α} {n m : ℕ} (hl : l₁.length = n) (hl' : l₁'.length = n)
    (h₁ : m < n → l₁[m]? = l₁'[m]?) (h₂ : n ≤ m → l₂[m - n]? = l₂'[m - n]?) : (l₁ ++ l₂)[m]? = (l₁' ++ l₂')[m]? := by
  rw [List.getElem?_append, List.getElem?_append, hl, hl']
  by_cases h : m < n
  · rw [ite_eq_left h, ite_eq_left h]; exact h₁ h
  · rw [ite_eq_right h, ite_eq_right h]; exact h₂ (Nat.le_of_not_lt h)

theorem length_msgTape {k : ℕ} (msg : ℕ → 𝔽 × (Fin k → 𝔽)) (a n : ℕ) : (msgTape msg a n).length = n := by
  simp only [msgTape, List.length_map, List.length_range']

theorem getElem?_msgTape {k : ℕ} {msg msg' : ℕ → 𝔽 × (Fin k → 𝔽)} {a n j : ℕ}
    (h : j < n → msg (a + j) = msg' (a + j)) : (msgTape msg a n)[j]? = (msgTape msg' a n)[j]? := by
  simp only [msgTape, List.getElem?_map]
  by_cases hj : j < n
  · rw [List.getElem?_range' hj, Option.map_some, Option.map_some, Nat.one_mul, h hj]
  · rw [List.getElem?_eq_none (by rw [List.length_range']; omega)]
    rfl

/-- **Message `m` reads only port `m + 1`'s words**: witnesses agreeing on `wStart m … wStart (m + 1)` have the same
message `m`. -/
theorem phaseMsgs_congr {sh : SparseShape} {w w' : Array Flock.F128} {m : ℕ}
    (h : ∀ e, wStart sh m ≤ e → e < wStart sh (m + 1) → w[e]! = w'[e]!) :
    (phaseMsgs sh w)[m]? = (phaseMsgs sh w')[m]? := by
  have hn : nR sh = sh.nu + sh.sLog := rfl
  unfold phaseMsgs
  refine getElem?_app (length_msgTape _ _ _) (length_msgTape _ _ _) (fun _ => getElem?_msgTape fun hm => ?_)
    fun h6 => ?_
  · have e : ∀ m', m' ≤ 6 → wStart sh m' = 64 + 2 * m' := fun m' h' => ite_eq_left h'
    unfold msg2 wPre
    rw [h _ (by rw [e m (by omega)]; omega) (by rw [e (m + 1) (by omega)]; omega),
      h _ (by rw [e m (by omega)]; omega) (by rw [e (m + 1) (by omega)]; omega)]
  refine getElem?_app (n := 1) rfl rfl (fun _ => ?_) fun h7 => ?_
  · obtain rfl : m = 6 := by omega
    rw [h wV2 (by unfold wStart wV2; rw [ite_eq_left (by omega)])
      (by unfold wStart; rw [ite_eq_right (by omega), ite_eq_left (by omega)]; unfold wSc degSum wV2; omega)]
  refine getElem?_app (length_msgTape _ _ _) (length_msgTape _ _ _) (fun _ => getElem?_msgTape fun hm => ?_)
    fun h8 => ?_
  · obtain ⟨e1, e2⟩ := wStart_sc (sh := sh) (m := m) (by omega) (by omega)
    rw [show 0 + (m - 6 - 1) = m - 7 by omega]
    unfold msg3
    congr 1
    · rw [h _ (by omega) (by omega)]
    · funext i
      have := i.isLt
      rw [h _ (by omega) (by omega)]
  refine getElem?_app (length_msgTape _ _ _) (length_msgTape _ _ _) (fun _ => getElem?_msgTape fun hm => ?_)
    fun h9 => ?_
  · obtain ⟨e1, e2⟩ := wStart_sc (sh := sh) (m := m) (by omega) (by omega)
    rw [show sh.nu + (m - 6 - 1 - sh.nu) = m - 7 by omega]
    unfold msg3
    congr 1
    · rw [h _ (by omega) (by omega)]
    · funext i
      have := i.isLt
      rw [h _ (by omega) (by omega)]
  refine getElem?_app (n := 1) rfl rfl (fun _ => ?_) fun h10 => ?_
  · obtain rfl : m = 7 + nR sh := by omega
    obtain ⟨e1, e2⟩ := wStart_u sh
    have hu : uOf sh w = uOf sh w' := funext fun q => by
      have := q.isLt
      unfold uOf
      rw [h _ (by omega) (by unfold wVt at e2; omega)]
    rw [hu]
  · unfold vtTape
    by_cases ht : sh.tLog = 0
    · rw [ite_eq_left ht, ite_eq_left ht]
    · rw [ite_eq_right ht, ite_eq_right ht]
      rcases hx : m - 6 - 1 - sh.nu - sh.sLog - 1 with _ | x
      · obtain rfl : m = 8 + nR sh := by omega
        obtain ⟨e1, e2⟩ := wStart_vt sh
        have hv : (fun t : Fin (2 ^ sh.tLog) => fOf w[wVt sh + t.val]!) = fun t => fOf w'[wVt sh + t.val]! :=
          funext fun t => by
            have := t.isLt
            rw [h _ (by omega) (by unfold wClaim tw at e2; rw [ite_eq_right ht] at e2; omega)]
        rw [hv]
      · simp only [List.getElem?_cons_succ, List.getElem?_nil]

/-! ## The words in V*'s layers -/

section cols

variable {S : ℕ} (Vs : Fin S → VStmt)

/-- Word `e` at copy 0 of unit `u` of session `s` in the layers `Lb`: its bit `t` at message column `128 e + t`. -/
def msgWord (mp : List (String × ℕ)) (Lb : (j : Fin S) → (Vs j).Bits) (s : Fin S) (u : Fin (Vs s).nv) (e : ℕ) :
    Flock.F128 :=
  ofBits fun t => bitAt (Vs s) (Lb s) u (rowCol mp 0 (128 * e + t))

/-- **Rep `i`'s message `m` read off the layers**: `phaseMsgs` of the words at copy 0 of unit `un i` of `sess i`. -/
def rdOf (sh : SparseShape) (mp : List (String × ℕ)) (sess : Fin 2 → Fin S) (un : (i : Fin 2) → Fin (Vs (sess i)).nv)
    (i : Fin 2) (m : ℕ) (Lb : (j : Fin S) → (Vs j).Bits) : Option Msg :=
  (phaseMsgs sh (Array.ofFn fun x : Fin (nw sh) => msgWord Vs mp Lb (sess i) (un i) x.val))[m]?

/-- **Hypothesis: every message column is an input column a row reads** (`OpenCols`' form), at each rep's session. -/
def SparseCols (mp : List (String × ℕ)) (sess : Fin 2 → Fin S) : Prop :=
  ∀ i : Fin 2, ∀ q < 128 * (mp.map (·.2)).sum,
    rowCol mp 0 q < (Rows.ofNet (Vs (sess i)).hU).nIn ∧ ReadCol (Rows.ofNet (Vs (sess i)).hU) (rowCol mp 0 q)

/-- **Hypothesis: stage `c₀ + i · nS + roundOf m` selects port `m + 1` in `own (sess i)`** (`skOf`'s form): at each
column of message `m`'s words, the reads file registers a port of the name of the port `q` its key reads, under a value
whose row unit `un i` reads holds that stage's message. -/
def SparseSel (sh : SparseShape) (mp : List (String × ℕ)) (own : Fin S → Array Flock.Registered.Port)
    (sel : ℕ → String → ℕ → Prop) (c₀ : ℕ) (sess : Fin 2 → Fin S) (un : (i : Fin 2) → Fin (Vs (sess i)).nv) : Prop :=
  ∀ (i : Fin 2) (m e t : ℕ), wStart sh m ≤ e → e < wStart sh (m + 1) → t < 128 →
    ∀ (q : Fin (Vs (sess i)).c.ports.size) (byte bit : ℕ),
      Aliased.srcKey (Vs (sess i)).c (rowCol mp 0 (128 * e + t)) = .msg q.val byte bit →
      ∃ p ∈ own (sess i), p.name = (Vs (sess i)).c.ports[q].name ∧ ∃ x, p.reads[(un i).val]? = some (some x) ∧
        sel (c₀ + i.val * nS sh + roundOf (sparseKinds 𝔽 sh) m) p.value x

variable {Vs}

theorem getElem!_rd {sh : SparseShape} {mp : List (String × ℕ)} {Lb : (j : Fin S) → (Vs j).Bits} {s : Fin S}
    {u : Fin (Vs s).nv} {e : ℕ} (h : e < nw sh) :
    (Array.ofFn fun x : Fin (nw sh) => msgWord Vs mp Lb s u x.val)[e]! = msgWord Vs mp Lb s u e :=
  arr_ofFn_get _ h

theorem bitOf_msgWord {mp : List (String × ℕ)} {Lb : (j : Fin S) → (Vs j).Bits} {s : Fin S} {u : Fin (Vs s).nv}
    {e t : ℕ} (ht : t < 128) : bitOf (msgWord Vs mp Lb s u e) t = bitAt (Vs s) (Lb s) u (rowCol mp 0 (128 * e + t)) :=
  bitOf_ofBits _ ht

/-- **A registered bit survives a mask selecting its row**: copy 0's input column `col` is the same in the layer masked
to stage `i`, where stage `i`'s commit strings hold the row `(u, q)` of the port `q` its key reads. -/
theorem bitAt_mask {Reg' : Type} (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)) (i : ℕ)
    (Xs : (j : Fin S) → (Vs j).Bits) {j : Fin S} (zo : (Vs j).Outer Reg') {u : Fin (Vs j).nv} {col : ℕ} {b : Bool}
    (hb : RegBit zo u 0 col b) (hio : IoColAt (Vs j) u col)
    (hsk : ∀ (q : Fin (Vs j).c.ports.size) (byte bit : ℕ), Aliased.srcKey (Vs j).c col = .msg q.val byte bit →
      Sum.inl (u, q) ∈ Sk i j) :
    bitAt (Vs j) (maskOf Vs Sk i Xs j) u col = bitAt (Vs j) (Xs j) u col := by
  unfold bitAt
  by_cases h : 0 < (Vs j).c.upv
  · rw [dite_eq_left h, dite_eq_left h]
    obtain ⟨hcol, q, k, hk, hkey, -, -⟩ := hb
    have hx : ((Rows.ofNet (Vs j).hU).stackCol (Vs j).c.upv h ⟨col, by simp only [Rows.w]; omega⟩).val =
        0 * (Rows.ofNet (Vs j).hU).nIn + col := by
      rw [Rows.stackCol, dite_eq_left hcol]
    have hr := keyRowOf_keyInst (Layout.outsOf (Vs j).c (Vs j).hU (Vs j).nv) u
      ((Rows.ofNet (Vs j).hU).stackCol (Vs j).c.upv h ⟨col, by simp only [Rows.w]; omega⟩)
      (by rw [hx]; exact Rows.mul_add_lt h hcol) hk (by rw [hx]; exact hkey)
    have hr' := HiddenExec.rowOfH_of_key (Layout.outsOf (Vs j).c (Vs j).hU (Vs j).nv) hr
    refine (copyVal_input _ _ h hcol).trans (Eq.trans ?_ (copyVal_input _ _ h hcol).symm)
    simp only [maskOf]
    refine ite_eq_left ⟨?_, u, hio h _⟩
    unfold posK
    rw [hr']
    exact hsk q (k / 8) (k % 8) (by rw [Nat.zero_mul, Nat.zero_add] at hkey; exact hkey)
  · rw [dite_eq_right h, dite_eq_right h]

end cols

/-! ## From the sessions -/

section src

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **Hypothesis: rep `i`'s sparse phase at copy 0 of unit `un i` of session `sess i`** (`SparseSessionAt`'s copy,
`SparseCopyAt`, at the map, with `SparseSrcAt`): at every accepted outer proof, rep `i`'s sparse steps are `stepsOf` at
w and coins `c`, and that copy holds the phase at w and `c`. -/
def SparseSessionsAt (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (sess : Fin 2 → Fin S) (un : (i : Fin 2) → Fin (Vs (sess i)).nv) : Prop :=
  ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ i : Fin 2,
    ∃ (w : Array Flock.F128) (c : FlockVBridge.Sparse.SCoins) (hj : 0 < (Vs (sess i)).c.upv),
      SparseSrcAt (holoAt st hp) (src R ω t (layers zs dg N k Rw R ω t R₂ τ)) i w c ∧
        SparseCopyAt (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) (holoAt st hp) w c (sess i) (un i) 0 hj

/-- **`SparseMsgsOfLayers` from V*'s sessions**: each word message `m` reads (`phaseMsgs_congr`) has its bits at the
message columns, which a row reads (`SparseCols`), registered there (`SparseSessionsAt`) and so the layer's
(`bitAt_of_regBit`), and the same in the layer masked to message `m`'s stage, which selects their rows (`SparseSel`,
`bitAt_mask`). -/
theorem sparseMsgsOfLayers_of_sessions
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop} {c₀ : ℕ} {sess : Fin 2 → Fin S}
    {un : (i : Fin 2) → Fin (Vs (sess i)).nv} (hS : SparseSessionsAt zs dg N k Rw src sess un)
    (hC : SparseCols Vs (Flock.RecSparse.messagePorts (unitShape (holoAt st hp))) sess)
    (hK : SparseSel Vs (holoAt st hp).cls.shape (Flock.RecSparse.messagePorts (unitShape (holoAt st hp))) own sel c₀
      sess un) :
    SparseMsgsOfLayers zs dg N k Rw src own sel c₀
      (rdOf Vs (holoAt st hp).cls.shape (Flock.RecSparse.messagePorts (unitShape (holoAt st hp))) sess un) := by
  intro R ω t R₂ τ o hA i
  obtain ⟨w, c, hj, ⟨-, -, hst⟩, -, -, -, -, hrb, -⟩ := hS R ω t R₂ τ o hA i
  refine ⟨w, c.cs, hst, fun m => ?_⟩
  unfold rdOf
  refine phaseMsgs_congr fun e he he' => ?_
  have hnw := lt_nw he'
  have hm : nw (holoAt st hp).cls.shape ≤
      ((Flock.RecSparse.messagePorts (unitShape (holoAt st hp))).map (·.2)).sum :=
    FlockVBridge.Sparse.nw_le_msum (unitShape (holoAt st hp))
  rw [getElem!_rd hnw]
  refine eq_of_bitOf fun b hb => ?_
  have hq : 128 * e + b < 128 * ((Flock.RecSparse.messagePorts (unitShape (holoAt st hp))).map (·.2)).sum := by
    omega
  obtain ⟨hlt, hrd⟩ := hC i _ hq
  have hio := ioColAt_of_readCol (un i) hlt hrd
  have hr := hrb _ hq hrd
  rw [bitOf_msgWord hb, bitAt_mask (skOf Vs own sel) _ _ (zs R ω t R₂ (sess i)) hr hio
      fun q byte bit hkey => mem_skOf_inl.2 (hK i m e b he he' hb q byte bit hkey),
    bitAt_of_regBit (zs R ω t R₂ (sess i)) (bits := layers zs dg N k Rw R ω t R₂ τ (sess i)) (hA.1 (sess i)).2.2 hr hio
      hj, wbit_elem _ _ hb]

end src

end

end FlockC3.Sparse
