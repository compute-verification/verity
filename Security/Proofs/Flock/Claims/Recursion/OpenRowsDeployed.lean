import Proofs.Flock.Claims.Recursion.OpenLowerDeployed
import Proofs.Flock.Claims.Recursion.RestFinal
import Proofs.Flock.Claims.Recursion.PartsVStar
import Proofs.Flock.Claims.Recursion.BoundAccepts
import Proofs.Flock.Claims.Recursion.SlotsCommit
import Proofs.Flock.VBridge.RecOpen
import Proofs.Flock.Soundness.Refine.Paths

/-!
# The element-level last message at the `rec-open` copies (`elemR`), and `CopyOpensR` there (U8)

`restR elem` leaves one choice open, `elem`: the pieces of the element-level last message, which `rdR` reads as the
statement and each rep's proof's openings (`encLast`). `elemR` lays every opening at the columns of the copy that opens
it, so a `rec-open` copy reads exactly the row and siblings the last message carries:

* **The pieces** (`elemR`): the tag, the statement and the proofs' count, constant bytes at the last part; then, for
  each rep `k`, its proof's openings level by level (`proofPieces`): level `ℓ`'s, at part `ℓ` (`levelPiece`), are its
  `q` opened rows, query `j`'s the `128 LANES` columns of instance `k q + j`'s copy from column 0 (`rowSeg`), then its
  `q · HH` siblings, sibling `h` of query `j` the 512 columns from `128 LANES + 512 h` (`sibSeg`), then no salts
  (`levelSegs`). A column is the copy's (`colForm`: `copyCol`'s hidden value up to the unit's constant, zero past it),
  at the part's circuit (`circOf`) and table 0's schedule (`schR`).
* **They read as `encLast`** (`readPieces_elemR`, `encLevel_levelSegs`): a row's bits are `elems`' (`colsHold_elems`,
  `encF_toBytes`), a sibling's `bytesAt`'s (`colsHold_bytesAt`). Every length fits its word (`fast100_small`), so `rdR`
  reads the openings back (`rdR_elemR`).
* **A part's pieces are its witness's** (`readSegs_piece`): `Agrees`' second conjunct holds part `s`'s pieces of the last
  round to its staged witness `w`, and the round is assembled off `ws`, so both read the same bytes there; `encLevel`
  reads back (`rdLevel_encLevel`), so the level's rows and siblings are `w`'s.
* `copyOpensR_elemR`: `CopyOpensR` at any layout with `layR elemR`'s messages and pieces (`layR` itself, or
  `ResLower.layVR`), the deployed `final`, `opened` and `k`, and `sideAtR rdR`, for pieces in range.
-/

namespace RecDeploy.OpenRows

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP encNat encBytes encList rowsOf joinR rowsOf_joinR')
open Claims.Recursion.VStarB
open RecDeploy.DeployedR
open RecDeploy.RestRead (DenseX kI lastAt layAt schedBridge finalR_last)
open RecDeploy.OpenLower
open FlockSoundness.Refine (openAt)
open FlockVBridge.Family (elems bytesAt)
open Lean (Json)

noncomputable section

/-! ## The pieces -/

/-- The inner statement's bytes at a compiled statement's reading: `xIOf`'s wherever the setup's checks accept it. -/
def stmtOf (r : XR) : Msg := if _root_.Flock.RecScope.accepts (readOf r) = true then r.w.inner else []

/-- The statement the last message carries: `stmtOf`, wherever its length fits its word. -/
def stmtR (r : XR) : Msg := if (stmtOf r).length < 2 ^ 64 then stmtOf r else []

/-- Table 0's schedule at the inner statement (`schAt`). -/
def schR (r : XR) : Flock.Schedule := schAt (stmtOf r) 0

/-- The inner statement's tables, none where it doesn't set up. -/
def tablesR (r : XR) : ℕ :=
  match FlockDeploy.setupOf false (stmtOf r) with
  | some v => v.2.2.size
  | none => 0

/-- The proofs the last message carries: two a table, wherever their count fits its word. -/
def nProofs (r : XR) : ℕ := if 2 * tablesR r < 2 ^ 64 then 2 * tablesR r else 0

/-- Part `ℓ`'s circuit, as the setup parses it. -/
def circOf (r : XR) (ℓ : ℕ) : Circuit := (r.w.parts[ℓ]?.bind partCircuit).getD default

/-- The part a level's openings sit at: part `ℓ`, else the last. -/
def partIdx (r : XR) (ℓ : ℕ) : ℕ := if ℓ < r.w.parts.length then ℓ else r.w.parts.length - 1

def partOf (r : XR) (h : 0 < r.w.parts.length) (ℓ : ℕ) : Fin r.w.parts.length :=
  ⟨partIdx r ℓ, by unfold partIdx; split_ifs <;> omega⟩

/-- **Column `t` of instance `n`'s copy** of circuit `c`'s unit, at a one-table statement (`copyCol`'s copy `0`): the
hidden value at block `n / g`, VU `n % g` up to the unit's constant, zero past it. -/
def colForm (c : Circuit) (n t : ℕ) : HmNets.Lin :=
  if t ≤ c.unit.constPos then HmNets.Lin.var (n / c.g * 2 ^ c.kLog + (c.unitRange.slot (n % c.g * c.upv) + t))
  else HmNets.Lin.const false

/-- Instance `n`'s copy's columns off a witness `u`. -/
def colBits (u : Claims.Flock.Wit) (c : Circuit) (n : ℕ) : ℕ → Bool := fun t => formBit u (colForm c n t)

/-- An opened row: its length, then instance `n`'s copy's `128 LANES` columns from column 0. -/
def rowSeg (c : Circuit) (LANES n : ℕ) : Msg × List HmNets.Lin :=
  (encNat LANES, (List.range (128 * LANES)).map (colForm c n))

/-- Sibling `h` of an opening: its length, then instance `n`'s copy's 512 columns from `128 LANES + 512 h`. -/
def sibSeg (c : Circuit) (LANES n h : ℕ) : Msg × List HmNets.Lin :=
  (encNat 64, (List.range 512).map fun t => colForm c n (128 * LANES + 512 * h + t))

/-- **A level's openings at rep `k`** (`encLevel`'s layout): its `q` rows, query `j`'s at instance `k q + j`, its
`q · HH` siblings, query by query, and no salts. -/
def levelSegs (c : Circuit) (q LANES HH k : ℕ) : List (Msg × List HmNets.Lin) :=
  [(encNat q, [])] ++ (List.range q).map (fun j => rowSeg c LANES (k * q + j)) ++ [(encNat (q * HH), [])] ++
    ((List.range q).flatMap fun j => (List.range HH).map (sibSeg c LANES (k * q + j))) ++ [(encNat 0, [])]

/-- Level `ℓ`'s openings at rep `k`, at part `ℓ`, at table 0's schedule. -/
def levelPiece (r : XR) (h : 0 < r.w.parts.length) (k ℓ : ℕ) :
    Fin r.w.parts.length × List (Msg × List HmNets.Lin) :=
  (partOf r h ℓ, levelSegs (circOf r ℓ) ((schR r).levels[ℓ]!).q (2 ^ ((schR r).levels[ℓ]!).lanes)
    (((schR r).levels[ℓ]!).d - ((schR r).levels[ℓ]!).cap) k)

/-- Constant bytes, at the last part. -/
def constPiece (r : XR) (h : 0 < r.w.parts.length) (m : Msg) : Fin r.w.parts.length × List (Msg × List HmNets.Lin) :=
  (⟨r.w.parts.length - 1, by omega⟩, [(m, [])])

/-- **Rep `k`'s proof's openings** (`encOpens`' layout): level 0's, the recursive levels' count and each's, then the
last level's. -/
def proofPieces (r : XR) (h : 0 < r.w.parts.length) (k : ℕ) :
    List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)) :=
  [levelPiece r h k 0, constPiece r h (encNat ((schR r).levels.size - 2))] ++
    (List.range ((schR r).levels.size - 2)).map (fun i => levelPiece r h k (i + 1)) ++
    [levelPiece r h k ((schR r).levels.size - 1)]

/-- **The element-level last message's pieces** (`encLast`'s layout): the tag, the statement and the proofs' count, then
each proof's openings. -/
def elemR (r : XR) : List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)) :=
  if h : 0 < r.w.parts.length then
    constPiece r h (1 :: (encNat (stmtR r).length ++ stmtR r ++ encNat (nProofs r))) ::
      (List.range (nProofs r)).flatMap (proofPieces r h)
  else []

/-! ## What they read -/

/-- A level's rows off a witness `u`: query `j`'s, instance `k q + j`'s copy's `LANES` elements. -/
def lvlRows (u : Claims.Flock.Wit) (c : Circuit) (q LANES k : ℕ) : Array (Array F128) :=
  ((List.range q).map fun j => elems (colBits u c (k * q + j)) 0 LANES).toArray

/-- A level's siblings off a witness `u`: query `j`'s `HH`, sibling `h` 64 bytes from column `128 LANES + 512 h`. -/
def lvlSibs (u : Claims.Flock.Wit) (c : Circuit) (q LANES HH k : ℕ) : Array ByteArray :=
  ((List.range q).flatMap fun j => (List.range HH).map fun h =>
    bytesAt (colBits u c (k * q + j)) (128 * LANES + 512 * h) 64).toArray

/-- The witness of part `i`. -/
def wAt (ws : List Claims.Flock.Wit) (i : ℕ) : Claims.Flock.Wit := ws.getD i ⟨[]⟩

/-- Level `ℓ`'s openings at rep `k`, off its part's witness. -/
def levelProof (r : XR) (ws : List Claims.Flock.Wit) (k ℓ : ℕ) : LevelProof :=
  { rows := lvlRows (wAt ws (partIdx r ℓ)) (circOf r ℓ) ((schR r).levels[ℓ]!).q (2 ^ ((schR r).levels[ℓ]!).lanes) k,
    siblings := lvlSibs (wAt ws (partIdx r ℓ)) (circOf r ℓ) ((schR r).levels[ℓ]!).q
      (2 ^ ((schR r).levels[ℓ]!).lanes) (((schR r).levels[ℓ]!).d - ((schR r).levels[ℓ]!).cap) k,
    salts := #[] }

/-- **Rep `k`'s proof as the last message carries it**: its levels' openings, every other field default. -/
def proofR (r : XR) (ws : List Claims.Flock.Wit) (k : ℕ) : Flock.Proof :=
  opensOnly (levelProof r ws k 0)
    ((List.range ((schR r).levels.size - 2)).map fun i => levelProof r ws k (i + 1)).toArray
    (levelProof r ws k ((schR r).levels.size - 1)).rows (levelProof r ws k ((schR r).levels.size - 1)).siblings #[]

/-! ## Bytes -/

theorem encF_toBytes (a : F128) : encF a = a.toBytes.data.toList := by
  apply List.ext_getElem
  · simp [encF, encNat, FlockSoundness.Refine.toBytes_length]
  · intro i h1 h2
    have hi : i < 16 := by simpa [encF, encNat] using h1
    have e := FlockSoundness.Refine.toBytes_get a i hi
    rw [getElem!_pos a.toBytes.data i (by simpa using hi)] at e
    rw [Array.getElem_toList, e]
    apply UInt8.toNat_inj.1
    by_cases h8 : i < 8
    · rw [if_pos h8, List.getElem_append_left (by simpa [encNat] using h8)]
      simp only [encNat, List.getElem_map, List.getElem_range]
      simp only [Nat.toUInt8, UInt8.toNat_ofNat, UInt64.toNat_toUInt8, UInt64.toNat_shiftRight,
        Nat.shiftRight_eq_div_pow, UInt64.toNat_ofNat]
      rw [Nat.mod_eq_of_lt (show 8 * i < 2 ^ 64 by omega), Nat.mod_eq_of_lt (show 8 * i < 64 by omega), pow_mul]
      rfl
    · rw [if_neg h8, List.getElem_append_right (by simp [encNat]; omega)]
      simp only [encNat, List.getElem_map, List.getElem_range, List.length_map, List.length_range]
      simp only [Nat.toUInt8, UInt8.toNat_ofNat, UInt64.toNat_toUInt8, UInt64.toNat_shiftRight,
        Nat.shiftRight_eq_div_pow, UInt64.toNat_ofNat]
      rw [Nat.mod_eq_of_lt (show 8 * (i - 8) < 2 ^ 64 by omega),
        Nat.mod_eq_of_lt (show 8 * (i - 8) < 64 by omega), pow_mul]
      rfl

theorem rowBytes_encF (row : Array F128) : (MerkleScheme.rowBytes row).data.toList = row.toList.flatMap encF := by
  rw [FlockSoundness.Refine.rowBytes_data, FlockSoundness.Refine.colBytes]
  congr 1
  funext a
  exact (encF_toBytes a).symm

theorem readForms_nil (u : Claims.Flock.Wit) : readForms u [] = [] :=
  List.eq_nil_of_length_eq_zero (by rw [readForms_length]; rfl)

/-- **Forms read the bytes their columns hold**: forms whose bits off `u` are columns `base …` of `f`, as many as `b`'s
bits, read `b` where those columns hold it. -/
theorem readForms_cols {u : Claims.Flock.Wit} {fs : List HmNets.Lin} {f : ℕ → Bool} {base : ℕ} {b : ByteArray}
    (hfs : fs.map (formBit u) = (List.range (8 * b.size)).map fun t => f (base + t)) (hb : FlockVBridge.ColsHold f base b) :
    readForms u fs = b.data.toList := by
  unfold readForms
  rw [hfs]
  conv_rhs => rw [← Claims.Recursion.VStar.bytesOf_bitL b.data.toList]
  have hl : b.data.toList.length = b.size := Array.length_toList
  rw [hl]
  congr 1
  apply List.map_congr_left
  intro t ht
  exact hb t (List.mem_range.1 ht)

theorem formBit_const (u : Claims.Flock.Wit) (b : Bool) : formBit u (HmNets.Lin.const b) = b := by
  simp [formBit, HmNets.Lin.const]

theorem readSeg_row (u : Claims.Flock.Wit) (c : Circuit) (LANES n : ℕ) :
    (rowSeg c LANES n).1 ++ readForms u (rowSeg c LANES n).2 =
      encList encF (elems (colBits u c n) 0 LANES).toList := by
  have hb : (MerkleScheme.rowBytes (elems (colBits u c n) 0 LANES)).size = 16 * LANES := by
    rw [FlockVBridge.size_rowBytes, FlockVBridge.Family.size_elems]
  have hr : readForms u (rowSeg c LANES n).2 = (MerkleScheme.rowBytes (elems (colBits u c n) 0 LANES)).data.toList :=
    readForms_cols (f := colBits u c n) (base := 0)
      (by rw [hb, rowSeg, List.map_map, show 8 * (16 * LANES) = 128 * LANES by ring]
          apply List.map_congr_left
          intro t _
          simp [colBits])
      (FlockVBridge.Family.colsHold_elems _ 0 LANES)
  rw [hr, rowBytes_encF, encList, Array.length_toList, FlockVBridge.Family.size_elems]
  rfl

theorem readSeg_sib (u : Claims.Flock.Wit) (c : Circuit) (LANES n h : ℕ) :
    (sibSeg c LANES n h).1 ++ readForms u (sibSeg c LANES n h).2 =
      encBytes (bytesAt (colBits u c n) (128 * LANES + 512 * h) 64) := by
  have hs := FlockVBridge.Family.size_bytesAt (colBits u c n) (128 * LANES + 512 * h) 64
  have hr : readForms u (sibSeg c LANES n h).2 = (bytesAt (colBits u c n) (128 * LANES + 512 * h) 64).data.toList :=
    readForms_cols (f := colBits u c n) (base := 128 * LANES + 512 * h)
      (by rw [hs, sibSeg, List.map_map]
          apply List.map_congr_left
          intro t _
          simp [colBits, Nat.add_assoc])
      (FlockVBridge.Family.colsHold_bytesAt _ _ 64)
  rw [hr, encBytes, hs]
  rfl

theorem length_flatMap_range {α β : Type} (q H : ℕ) (g : ℕ → ℕ → β) :
    ((List.range q).flatMap fun j => (List.range H).map (g j)).length = q * H := by
  induction q with
  | zero => rfl
  | succ q ih =>
    rw [List.range_succ, List.flatMap_append, List.length_append, ih]
    simp [Nat.succ_mul]

/-- **A level's openings read as `encLevel`** of its rows and siblings off the witness, and no salts. -/
theorem encLevel_levelSegs (u : Claims.Flock.Wit) (c : Circuit) (q LANES HH k : ℕ) :
    readSegs u (levelSegs c q LANES HH k) = encLevel (lvlRows u c q LANES k) (lvlSibs u c q LANES HH k) #[] := by
  unfold readSegs levelSegs encLevel lvlRows lvlSibs
  simp only [List.flatMap_append, List.flatMap_cons, List.flatMap_nil, List.append_nil, readForms_nil,
    List.flatMap_map, List.toList_toArray, List.map_map, encList, List.length_map, List.length_range,
    List.flatMap_assoc, List.toArray_toList, List.length_nil]
  rw [length_flatMap_range q HH (fun j h => bytesAt (colBits u c (k * q + j)) (128 * LANES + 512 * h) 64)]
  have e1 : ((List.range q).flatMap fun j => (rowSeg c LANES (k * q + j)).1 ++ readForms u (rowSeg c LANES (k * q + j)).2) =
      (List.range q).flatMap fun j => encList encF (elems (colBits u c (k * q + j)) 0 LANES).toList := by
    congr 1
    funext j
    exact readSeg_row u c LANES (k * q + j)
  have e2 : ((List.range q).flatMap fun j => (List.range HH).flatMap fun h =>
      (sibSeg c LANES (k * q + j) h).1 ++ readForms u (sibSeg c LANES (k * q + j) h).2) =
      (List.range q).flatMap fun j => (List.range HH).flatMap fun h =>
        encBytes (bytesAt (colBits u c (k * q + j)) (128 * LANES + 512 * h) 64) := by
    congr 1
    funext j
    congr 1
    funext h
    exact readSeg_sib u c LANES (k * q + j) h
  simp only [Function.comp_def] at e1 e2 ⊢
  rw [e1, e2]
  simp [encList, List.append_assoc]

/-! ## The last message's openings -/

theorem mapM_map_some {α β γ : Type} {f : β → Option γ} {g : α → β} {h : α → γ} (hf : ∀ a, f (g a) = some (h a)) :
    ∀ l : List α, (l.map g).mapM f = some (l.map h)
  | [] => rfl
  | a :: l => by simp [List.mapM_cons, hf, mapM_map_some hf l]

theorem readPieces_cons (p : ℕ × List (Msg × List HmNets.Lin)) (ps : List (ℕ × List (Msg × List HmNets.Lin)))
    (ws : List Claims.Flock.Wit) :
    readPieces (p :: ps) ws =
      (ws[p.1]?.map fun w => readSegs w p.2).getD (List.replicate (pieceLen p.2) 0) ++ readPieces ps ws := rfl

theorem readPieces_append (a b : List (ℕ × List (Msg × List HmNets.Lin))) (ws : List Claims.Flock.Wit) :
    readPieces (a ++ b) ws = readPieces a ws ++ readPieces b ws := by
  simp [readPieces, List.flatMap_append]

theorem piece_at {ws : List Claims.Flock.Wit} {i : ℕ} (hi : i < ws.length) (segs : List (Msg × List HmNets.Lin)) :
    (ws[i]?.map fun w => readSegs w segs).getD (List.replicate (pieceLen segs) 0) = readSegs (wAt ws i) segs := by
  rw [List.getElem?_eq_getElem hi, Option.map_some, Option.getD_some, wAt, List.getD_eq_getElem _ _ hi]

theorem readSegs_const (u : Claims.Flock.Wit) (m : Msg) : readSegs u [(m, [])] = m := by
  simp [readSegs, readForms_nil]

theorem partIdx_lt (r : XR) (h : 0 < r.w.parts.length) (ℓ : ℕ) : partIdx r ℓ < r.w.parts.length := by
  unfold partIdx; split_ifs <;> omega

theorem readPieces_proofPieces (r : XR) (h : 0 < r.w.parts.length) {ws : List Claims.Flock.Wit}
    (hl : ws.length = r.w.parts.length) (k : ℕ) :
    readPieces ((proofPieces r h k).map fun p => (p.1.val, p.2)) ws = encOpens (proofR r ws k) := by
  have hL : ∀ ℓ, readPieces [((levelPiece r h k ℓ).1.val, (levelPiece r h k ℓ).2)] ws =
      encLevel (levelProof r ws k ℓ).rows (levelProof r ws k ℓ).siblings (levelProof r ws k ℓ).salts := by
    intro ℓ
    rw [readPieces_cons, piece_at (by rw [hl]; exact partIdx_lt r h ℓ)]
    simp only [readPieces, List.flatMap_nil, List.append_nil]
    exact encLevel_levelSegs _ _ _ _ _ _
  have hC : ∀ m, readPieces [((constPiece r h m).1.val, (constPiece r h m).2)] ws = m := by
    intro m
    rw [readPieces_cons, piece_at (by rw [hl]; show r.w.parts.length - 1 < r.w.parts.length; omega)]
    simp only [readPieces, List.flatMap_nil, List.append_nil]
    exact readSegs_const _ m
  have hM : readPieces (((List.range ((schR r).levels.size - 2)).map fun i => levelPiece r h k (i + 1)).map
      fun p => (p.1.val, p.2)) ws = ((List.range ((schR r).levels.size - 2)).map fun i => levelProof r ws k (i + 1)).flatMap
        fun q => encLevel q.rows q.siblings q.salts := by
    induction (List.range ((schR r).levels.size - 2)) with
    | nil => rfl
    | cons i l ih =>
      rw [List.map_cons, List.map_cons, show ∀ (a : ℕ × List (Msg × List HmNets.Lin)) l, a :: l = [a] ++ l from
        fun _ _ => rfl, readPieces_append, hL, ih]
      rfl
  unfold proofPieces
  simp only [List.map_append, List.map_cons, List.map_nil]
  rw [show ∀ (a b : ℕ × List (Msg × List HmNets.Lin)) l c, [a, b] ++ l ++ [c] = [a] ++ [b] ++ l ++ [c] from
    fun _ _ _ _ => rfl, readPieces_append, readPieces_append, readPieces_append, hL, hC, hM, hL]
  simp only [encOpens, proofR, opensOnly, encList, List.toList_toArray, List.length_map, List.length_range,
    List.append_assoc]

/-- **`elemR`'s pieces read as `encLast`** of the statement and each rep's proof off the witnesses. -/
theorem readPieces_elemR (r : XR) {ws : List Claims.Flock.Wit} (hl : ws.length = r.w.parts.length)
    (h : 0 < r.w.parts.length) :
    readPieces ((elemR r).map fun p => (p.1.val, p.2)) ws =
      encLast (stmtR r) ((List.range (nProofs r)).map (proofR r ws)) := by
  unfold elemR
  rw [dif_pos h, List.map_cons, show ∀ (a : ℕ × List (Msg × List HmNets.Lin)) l, a :: l = [a] ++ l from
    fun _ _ => rfl, readPieces_append]
  rw [readPieces_cons, piece_at (by rw [hl]; show r.w.parts.length - 1 < r.w.parts.length; omega)]
  simp only [readPieces, List.flatMap_nil, List.append_nil, readSegs_const]
  have hF : ((List.range (nProofs r)).flatMap (proofPieces r h)).map (fun p => (p.1.val, p.2)) =
      (List.range (nProofs r)).flatMap fun k => (proofPieces r h k).map fun p => (p.1.val, p.2) := by
    rw [List.map_flatMap]
  have hR : ((List.range (nProofs r)).flatMap fun k => (proofPieces r h k).map fun p => (p.1.val, p.2)).flatMap
      (fun p => (ws[p.1]?.map fun w => readSegs w p.2).getD (List.replicate (pieceLen p.2) 0)) =
      ((List.range (nProofs r)).map (proofR r ws)).flatMap encOpens := by
    rw [List.flatMap_assoc, List.flatMap_map]
    congr 1
    funext k
    exact readPieces_proofPieces r h hl k
  rw [hF, hR]
  simp [encLast, encList, List.append_assoc]

/-! ## Every length fits its word -/

theorem fast100_small {m : ℕ} {s : Flock.Schedule} (h : Flock.fast100 m = some s) :
    s.levels.size ≤ 7 ∧ ∀ l ∈ s.levels, l.q < 2 ^ 16 ∧ l.lanes < 16 ∧ l.d < 2 ^ 16 := by
  have hm : m < 40 := by
    have := FlockVBridge.Family.fast100_range h
    omega
  have key : ∀ m : Fin 40, ∀ s ∈ Flock.fast100 m.val, (decide (s.levels.size ≤ 7) &&
      s.levels.all fun l => decide (l.q < 2 ^ 16) && decide (l.lanes < 16) && decide (l.d < 2 ^ 16)) = true := by
    decide +kernel
  have hk := key ⟨m, hm⟩ s h
  simp only [Bool.and_eq_true, decide_eq_true_eq, Array.all_eq_true] at hk
  refine ⟨hk.1, fun l hl => ?_⟩
  obtain ⟨i, hi, rfl⟩ := Array.getElem_of_mem hl
  exact ⟨(hk.2 i hi).1.1, (hk.2 i hi).1.2, (hk.2 i hi).2⟩

theorem schAt_small (x : Msg) (t : ℕ) :
    (schAt x t).levels.size ≤ 7 ∧ ∀ ℓ, ((schAt x t).levels[ℓ]!).q < 2 ^ 16 ∧ ((schAt x t).levels[ℓ]!).lanes < 16 ∧
      ((schAt x t).levels[ℓ]!).d < 2 ^ 16 := by
  have hS : ∀ s : Flock.Schedule, s.levels.size ≤ 7 → (∀ l ∈ s.levels, l.q < 2 ^ 16 ∧ l.lanes < 16 ∧ l.d < 2 ^ 16) →
      s.levels.size ≤ 7 ∧ ∀ ℓ, (s.levels[ℓ]!).q < 2 ^ 16 ∧ (s.levels[ℓ]!).lanes < 16 ∧ (s.levels[ℓ]!).d < 2 ^ 16 := by
    intro s h7 hl
    refine ⟨h7, fun ℓ => ?_⟩
    by_cases hℓ : ℓ < s.levels.size
    · rw [getElem!_pos s.levels ℓ hℓ]
      exact hl _ (Array.getElem_mem hℓ)
    · rw [getElem!_neg s.levels ℓ hℓ]
      exact ⟨by decide, by decide, by decide⟩
  have hD : (default : Flock.Schedule).levels.size ≤ 7 ∧
      ∀ l ∈ (default : Flock.Schedule).levels, l.q < 2 ^ 16 ∧ l.lanes < 16 ∧ l.d < 2 ^ 16 :=
    ⟨by decide, fun l hl => by simp at hl⟩
  unfold schAt
  split
  · cases hf : Flock.fast100 _ with
    | none => exact hS _ hD.1 hD.2
    | some s =>
      obtain ⟨h7, hl⟩ := fast100_small hf
      exact hS s h7 hl
  · exact hS _ hD.1 hD.2

theorem levelFits (u : Claims.Flock.Wit) (c : Circuit) {q LANES HH : ℕ} (k : ℕ) (hq : q < 2 ^ 32)
    (hL : LANES < 2 ^ 32) (hH : HH < 2 ^ 32) :
    LevelFits (lvlRows u c q LANES k) (lvlSibs u c q LANES HH k) #[] := by
  have hqH : q * HH < 2 ^ 64 := by
    calc q * HH < 2 ^ 32 * 2 ^ 32 := Nat.mul_lt_mul'' hq hH
      _ = 2 ^ 64 := by norm_num
  refine ⟨?_, ?_, ?_, ?_, by simp, by simp⟩
  · simp only [lvlRows, List.size_toArray, List.length_map, List.length_range]
    omega
  · intro a ha
    simp only [lvlRows, List.toList_toArray, List.mem_map, List.mem_range] at ha
    obtain ⟨j, -, rfl⟩ := ha
    rw [FlockVBridge.Family.size_elems]
    omega
  · simp only [lvlSibs, List.size_toArray]
    rw [length_flatMap_range]
    exact hqH
  · intro b hb
    simp only [lvlSibs, List.toList_toArray, List.mem_flatMap, List.mem_map, List.mem_range] at hb
    obtain ⟨j, -, h, -, rfl⟩ := hb
    rw [FlockVBridge.Family.size_bytesAt]
    norm_num

theorem levelProof_fits (r : XR) (ws : List Claims.Flock.Wit) (k ℓ : ℕ) :
    LevelFits (levelProof r ws k ℓ).rows (levelProof r ws k ℓ).siblings (levelProof r ws k ℓ).salts := by
  obtain ⟨-, hs⟩ := schAt_small (stmtOf r) 0
  obtain ⟨hq, hl, hd⟩ := hs ℓ
  have hL : 2 ^ ((schR r).levels[ℓ]!).lanes < 2 ^ 32 :=
    lt_of_lt_of_le (Nat.pow_lt_pow_right (by norm_num) hl) (by norm_num)
  exact levelFits _ _ k (by unfold schR; omega) hL (by unfold schR; omega)

theorem proofR_fits (r : XR) (ws : List Claims.Flock.Wit) (k : ℕ) : OpensFit (proofR r ws k) := by
  obtain ⟨h7, -⟩ := schAt_small (stmtOf r) 0
  refine ⟨levelProof_fits r ws k 0, ?_, ?_, ?_⟩
  · simp only [proofR, opensOnly, List.size_toArray, List.length_map, List.length_range]
    unfold schR
    omega
  · intro q hq
    simp only [proofR, opensOnly, List.toList_toArray, List.mem_map, List.mem_range] at hq
    obtain ⟨i, -, rfl⟩ := hq
    exact levelProof_fits r ws k (i + 1)
  · exact ⟨(levelProof_fits r ws k _).1, (levelProof_fits r ws k _).2.1, (levelProof_fits r ws k _).2.2.1,
      (levelProof_fits r ws k _).2.2.2.1, by simp, by simp⟩

/-- **`rdR` reads `elemR`'s pieces back**: the statement, and each rep's proof's openings. -/
theorem rdR_elemR (r : XR) {ws : List Claims.Flock.Wit} (hl : ws.length = r.w.parts.length)
    (h : 0 < r.w.parts.length) :
    rdR (readPieces ((elemR r).map fun p => (p.1.val, p.2)) ws) =
      some ⟨stmtR r, Json.null, (((List.range (nProofs r)).map (proofR r ws)).map opensOf).toArray⟩ := by
  rw [readPieces_elemR r hl h]
  refine rdR_encLast ?_ ?_ ?_
  · unfold stmtR
    split_ifs with hs
    · exact hs
    · simp
  · simp only [List.length_map, List.length_range, nProofs]
    split_ifs with hs
    · exact hs
    · norm_num
  · intro p hp
    obtain ⟨k, -, rfl⟩ := List.mem_map.1 hp
    exact proofR_fits r ws k

/-! ## The openings a level opens -/

theorem openAt_opensOf (p : Flock.Proof) (R ℓ : ℕ) :
    (openAt (opensOf p).ligerito R ℓ).1 = (openAt p.ligerito R ℓ).1 ∧
      (openAt (opensOf p).ligerito R ℓ).2.1 = (openAt p.ligerito R ℓ).2.1 := by
  unfold openAt opensOf opensOnly
  split_ifs <;> exact ⟨rfl, rfl⟩

/-- **Level `ℓ`'s openings in rep `k`'s proof** are `levelProof`'s, at every level of the schedule. -/
theorem openAt_proofR (r : XR) (ws : List Claims.Flock.Wit) (k : ℕ) {ℓ : ℕ} (hℓ : ℓ < (schR r).levels.size) :
    (openAt (proofR r ws k).ligerito ((schR r).levels.size - 1) ℓ).1 = (levelProof r ws k ℓ).rows ∧
      (openAt (proofR r ws k).ligerito ((schR r).levels.size - 1) ℓ).2.1 = (levelProof r ws k ℓ).siblings := by
  unfold openAt
  split_ifs with h0 h1
  · subst h0
    exact ⟨rfl, rfl⟩
  · have hi : ℓ - 1 < ((List.range ((schR r).levels.size - 2)).map fun i => levelProof r ws k (i + 1)).toArray.size := by
      simp only [List.size_toArray, List.length_map, List.length_range]
      omega
    have e : (((List.range ((schR r).levels.size - 2)).map fun i => levelProof r ws k (i + 1)).toArray)[ℓ - 1]! =
        levelProof r ws k ℓ := by
      rw [getElem!_pos _ _ hi]
      simp only [List.getElem_toArray, List.getElem_map, List.getElem_range]
      rw [Nat.sub_add_cancel (by omega)]
    show ((((List.range ((schR r).levels.size - 2)).map fun i => levelProof r ws k (i + 1)).toArray)[ℓ - 1]!).rows = _ ∧
      ((((List.range ((schR r).levels.size - 2)).map fun i => levelProof r ws k (i + 1)).toArray)[ℓ - 1]!).siblings = _
    rw [e]
    exact ⟨rfl, rfl⟩
  · have e : ℓ = (schR r).levels.size - 1 := by omega
    subst e
    exact ⟨rfl, rfl⟩

/-- The rows' `j`-th, at a query of the level. -/
theorem lvlRows_get (u : Claims.Flock.Wit) (c : Circuit) {q j : ℕ} (LANES k : ℕ) (hj : j < q) :
    (lvlRows u c q LANES k)[j]! = elems (colBits u c (k * q + j)) 0 LANES := by
  unfold lvlRows
  rw [getElem!_pos _ _ (by simpa using hj)]
  simp

theorem take_drop_blocks {β : Type} (g : ℕ → List β) {H : ℕ} (hg : ∀ j, (g j).length = H) :
    ∀ q j, j < q → (((List.range q).flatMap g).drop (j * H)).take H = g j
  | 0, j, h => absurd h (Nat.not_lt_zero j)
  | q + 1, j, h => by
    have hlen : ((List.range q).flatMap g).length = q * H := by
      rw [List.length_flatMap]
      simp [hg]
    rw [List.range_succ, List.flatMap_append, List.flatMap_singleton]
    rcases Nat.lt_succ_iff_lt_or_eq.1 h with h | rfl
    · rw [List.drop_append_of_le_length (by rw [hlen]; exact Nat.mul_le_mul_right H (by omega)),
        List.take_append_of_le_length (by
          rw [List.length_drop, hlen]
          have : (j + 1) * H ≤ q * H := Nat.mul_le_mul_right H h
          rw [Nat.succ_mul] at this
          omega)]
      exact take_drop_blocks g hg q j h
    · rw [List.drop_append_of_le_length (le_of_eq hlen.symm), List.drop_of_length_le (le_of_eq hlen), List.nil_append,
        List.take_of_length_le (le_of_eq (hg _))]
      · rw [List.drop_left' hlen]
      · simp [hlen]

/-- The siblings' `j`-th block, at a query of the level. -/
theorem sibsOf_lvlSibs (u : Claims.Flock.Wit) (c : Circuit) {q j : ℕ} (LANES HH k : ℕ) (hj : j < q) :
    Flock.sibsOf (lvlSibs u c q LANES HH k) HH j =
      (List.range HH).map fun h => bytesAt (colBits u c (k * q + j)) (128 * LANES + 512 * h) 64 := by
  unfold Flock.sibsOf lvlSibs
  rw [Array.toList_extract, List.extract_eq_drop_take, List.toList_toArray, show j * HH + HH - j * HH = HH by omega]
  exact take_drop_blocks (fun j => (List.range HH).map fun h =>
    bytesAt (colBits u c (k * q + j)) (128 * LANES + 512 * h) 64) (fun _ => by simp) q j hj

/-! ## A part's pieces are its witness's -/

theorem foldl_keep (w : Claims.Flock.Wit) (s : ℕ) :
    ∀ (l : List (ℕ × List (Msg × List HmNets.Lin))) (o : ℕ) (acc : List (ℕ × Msg)) (e : ℕ × Msg), e ∈ acc →
      e ∈ (l.foldl (fun (acc : ℕ × List (ℕ × Msg)) p =>
        (acc.1 + pieceLen p.2, if p.1 = s then acc.2 ++ [(acc.1, readSegs w p.2)] else acc.2)) (o, acc)).2
  | [], _, _, _, h => h
  | q :: l, o, acc, e, h => by
    rw [List.foldl_cons]
    refine foldl_keep w s l _ _ e ?_
    dsimp only
    split_ifs
    · exact List.mem_append_left _ h
    · exact h

theorem foldl_at (w : Claims.Flock.Wit) (s : ℕ) (p : ℕ × List (Msg × List HmNets.Lin)) (hp : p.1 = s)
    (post : List (ℕ × List (Msg × List HmNets.Lin))) :
    ∀ (pre : List (ℕ × List (Msg × List HmNets.Lin))) (o : ℕ) (acc : List (ℕ × Msg)),
      (o + (pre.map fun q => pieceLen q.2).sum, readSegs w p.2) ∈ ((pre ++ p :: post).foldl
        (fun (acc : ℕ × List (ℕ × Msg)) p =>
          (acc.1 + pieceLen p.2, if p.1 = s then acc.2 ++ [(acc.1, readSegs w p.2)] else acc.2)) (o, acc)).2
  | [], o, acc => by
    rw [List.nil_append, List.foldl_cons]
    refine foldl_keep w s post _ _ _ ?_
    dsimp only
    rw [if_pos hp]
    simp
  | q :: pre, o, acc => by
    rw [List.cons_append, List.foldl_cons]
    have := foldl_at w s p hp post pre (o + pieceLen q.2)
      (if q.1 = s then acc ++ [(o, readSegs w q.2)] else acc)
    simpa [Nat.add_assoc] using this

/-- **Each of a round's pieces at part `s` is among part `s`'s**, at its offset (`LayoutB.mem_piecesOf`'s converse). -/
theorem piecesOf_mem {L : LayoutB} {x : Msg} {s : ℕ} {w : Claims.Flock.Wit} {i : ℕ}
    {pre post : List (ℕ × List (Msg × List HmNets.Lin))} {p : ℕ × List (Msg × List HmNets.Lin)}
    (hl : L.pieceAt x i = pre ++ p :: post) (hp : p.1 = s) :
    ((pre.map fun q => pieceLen q.2).sum, readSegs w p.2) ∈ L.piecesOf x s w i := by
  unfold LayoutB.piecesOf
  rw [hl]
  simpa using foldl_at w s p hp post pre 0 []

/-- **A part's piece of an assembled round reads the same off its staged witness as off its own**: the round agrees
with part `s`'s pieces at `w` (`Agrees`' second conjunct) and is assembled off `ws`. -/
theorem readSegs_piece {L : LayoutB} {x : Msg} {s : ℕ} {w : Claims.Flock.Wit} {ws : List Claims.Flock.Wit} {i : ℕ}
    (hA : ∀ i < L.rounds x, ∀ o m, (o, m) ∈ L.piecesOf x s w i →
      (((L.msgs x ws).getD i []).drop o).take m.length = m)
    (hi : i < L.rounds x) (hf : L.fixedAt x i = none) (hs : s < ws.length) {segs : List (Msg × List HmNets.Lin)}
    (hp : (s, segs) ∈ L.pieceAt x i) : readSegs w segs = readSegs ws[s] segs := by
  obtain ⟨pre, post, hl⟩ := List.append_of_mem hp
  have hne : L.pieceAt x i ≠ [] := List.ne_nil_of_mem hp
  have hmsg : (L.msgs x ws).getD i [] = L.assembled x ws i := by
    rw [L.msgs_getD x ws hi, LayoutB.msg_of_pieces hf hne]
  have h1 := hA i hi _ _ (piecesOf_mem (w := w) hl rfl)
  have h2 := LayoutB.assembled_piece hs hl rfl
  rw [hmsg] at h1
  have hlen : (readSegs w segs).length = (readSegs ws[s] segs).length := by rw [readSegs_length, readSegs_length]
  rw [hlen] at h1
  exact h1.symm.trans h2

/-- **The last round at a dense compiled statement**: it is in range, the statement doesn't fix it, and its pieces are
the openings then `restR elem`'s (`lastAt_parts`'). -/
theorem lastPieces (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) {x : Msg}
    (hx : DenseX x) :
    ∃ r, readX x = some r ∧ kI x - 1 < (layAt (restR elem)).rounds x ∧
      (layAt (restR elem)).fixedAt x (kI x - 1) = none ∧
      (layAt (restR elem)).pieceAt x (kI x - 1) = openPieces r ++ (restR elem r).map fun p => (p.1.val, p.2) := by
  obtain ⟨r, rs, s0, spec, st, hr, hrs, hset, hm, hsc, hk⟩ := schedBridge hx
  have hR : layoutOf.rounds x = 2 + 2 * rs.size := hk
  have hk1 : kI x - 1 = 2 * rs.size + 1 := by omega
  have hsr : streamRound x (2 * rs.size) = none := by
    unfold streamRound
    rw [hr, Option.bind_some, hrs, Option.bind_some, if_neg (by omega)]
  have hf : (layAt (restR elem)).fixedAt x (2 * rs.size + 1) = none := by
    show (streamRound x (2 * rs.size)).bind _ = none
    rw [hsr]
    rfl
  have hp : (layAt (restR elem)).pieceAt x (2 * rs.size + 1) = finOpen (restR elem) x := by
    show stepPieces (finOpen (restR elem)) x (2 * rs.size + 1) = _
    unfold stepPieces
    rw [hR, if_pos (by omega)]
  have hfo : finOpen (restR elem) x = openPieces r ++ (restR elem r).map fun p => (p.1.val, p.2) := by
    simp [finOpen, hr]
  refine ⟨r, hr, by show kI x - 1 < kI x; omega, ?_, ?_⟩
  · rw [hk1]; exact hf
  · rw [hk1, hp, hfo]

theorem mem_proofPieces (r : XR) (h : 0 < r.w.parts.length) (k : ℕ) {ℓ : ℕ} (hℓ : ℓ < (schR r).levels.size) :
    levelPiece r h k ℓ ∈ proofPieces r h k := by
  unfold proofPieces
  by_cases h0 : ℓ = 0
  · subst h0
    simp
  · by_cases h1 : ℓ < (schR r).levels.size - 1
    · refine List.mem_append_left _ (List.mem_append_right _ (List.mem_map.2 ⟨ℓ - 1, List.mem_range.2 (by omega), ?_⟩))
      rw [Nat.sub_add_cancel (by omega)]
    · have e : ℓ = (schR r).levels.size - 1 := by omega
      subst e
      simp

/-! ## `CopyOpensR` at `elemR` -/

theorem stmtOf_xIOf {x : Msg} {st : Claims.Recursion.Stmt} (hst : parse x = some st) {r : XR}
    (hr : readX x = some r) : xIOf x = stmtOf r := by
  obtain ⟨r', hr', ha, rfl⟩ := parse_read hst
  rw [hr] at hr'
  cases hr'
  have hx : xIOf x = r.w.inner := by unfold xIOf; rw [hst]; rfl
  rw [hx, stmtOf, if_pos ha]

/-- **A copy's columns are `colForm`'s**, at its part's circuit, at a one-table statement. -/
theorem copyOf_colBits {p : _root_.Flock.Readable.Parsed} (ht : 0 < p.sts.size) (hsz : p.sts.size = 1)
    {c : Circuit} (hc : p.sts[0].c = c) (w : Claims.Flock.Wit) (n : ℕ) :
    _root_.Flock.Readable.copyOf p 0 ht w (n / p.sts[0]!.c.g) (n % p.sts[0]!.c.g) 0 = colBits w c n := by
  rw [getElem!_pos p.sts 0 ht]
  funext t
  unfold _root_.Flock.Readable.copyOf _root_.Flock.Readable.copyCol colBits colForm
  rw [hc, hsz]
  split_ifs with h
  · rw [formBit_var]
    simp
  · rw [formBit_const]

variable (H : Msg → Msg) (D : Claims.Recursion.Deploy Coin) (shape : Claims.Recursion.Form → Prop)
  (untr : TrReader) (b : Brings) (cmt : Msg → ℕ → Msg → Msg) (L : LayoutB)
  (Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace)

/-- **`CopyOpensR` at `elemR`**: at a layout with `layR elemR`'s rounds, messages and pieces (`layR elemR`, or
`ResLower.layVR elemR`), at a bound record reading the last message with `finalR`, its opened rounds joined by
`joinR` and the plain verifier's rounds, at `sideAtR rdR` and pieces in range, each piece's copy reads the row and
siblings the last message opens. The last message is `elemR`'s pieces (`finalR_last`), which `rdR` reads back as each
rep's openings off the witnesses (`rdR_elemR`); part `s`'s are its staged witness's (`readSegs_piece`), and those
are its copy's columns (`copyOf_colBits`). -/
theorem copyOpensR_elemR (pieces : Msg → List OpenPiece) (hP : ∀ x, D.parse x = parse x)
    (hS : ∀ f, shape f → Claims.Recursion.Form.Dense f) (hN : ∀ x, D.nS x = nS x)
    (hLr : ∀ x, L.rounds x = (layAt (restR elemR)).rounds x)
    (hLm : ∀ x ws, L.msgs x ws = (layAt (restR elemR)).msgs x ws)
    (hLf : ∀ x i, L.fixedAt x i = (layAt (restR elemR)).fixedAt x i)
    (hLp : ∀ x i, L.pieceAt x i = (layAt (restR elemR)).pieceAt x i)
    (hk : ∀ y, Bd.DE.V.k y = (FlockDeploy.V false).k y) (hfin : Bd.final = finalR)
    (hop : ∀ os last, Bd.opened os last = os.map (fun o => joinR (o.map Prod.fst)) ++ [last])
    (hrange : PiecesInRangeR pieces) :
    CopyOpensR H D shape untr b cmt L Bd (sideAtR rdR) pieces := by
  intro x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hst q hq ho ht os last hfn hos
  have hx' : parse x = some st := (hP x).symm.trans hx
  have hD : DenseX x := ⟨st, hx', hS _ hf⟩
  have hxI : Claims.Recursion.xI D x = xIOf x := by
    unfold Claims.Recursion.xI xIOf
    rw [hP x]
  have hxs : xIOf x = stmtOf r := stmtOf_xIOf hx' hr
  have hn : nS x = r.w.parts.length := by simp [nS, hr]
  have hl : ws.length = r.w.parts.length := by rw [hJ.1, hN, hn]
  -- the piece's part and level
  obtain ⟨h0, r', ss', hr', hss', L', H', n', hsv'⟩ := ownQR_some ho
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  rw [hsv] at hsv'
  cases hsv'
  obtain ⟨hls, hL, hH, -⟩ := levelMatch hx' (hS _ hf) hr hss hsv
  obtain ⟨vs, hvs, hlen, hparts⟩ := readX_partVStar hr
  rw [hss] at hvs
  cases hvs
  have hsl : s < ss.size := (Array.getElem?_eq_some_iff.1 hsv).1
  obtain ⟨-, pt, cc, -, hpt, hpc, -, -, -⟩ := hparts s hsl
  have hpc' : r.w.parts[s]?.bind partCircuit = some cc := by rw [hpt, Option.bind_some, hpc]
  have hsp : s < r.w.parts.length := by omega
  have hpos : 0 < r.w.parts.length := by omega
  -- the staged copy
  obtain ⟨hxO, -, -, hAg, -, -, -, -, hSP⟩ := hst
  obtain ⟨p', r'', sv, hxO', htyp, hpi, -, -, h0p, -⟩ := stagedTables hSP
  rw [hxO] at hxO'
  cases hxO'
  have hc0 : p.sts[0].c = cc := partCircuit_staged hSP hxO htyp hpi hr hpc' h0p
  have hSP' := hSP
  obtain ⟨-, i0, spec0, sts0, -, hsu, -⟩ := hSP'
  have hpe : p = { i0.base.read with sts := sts0 } := Option.some.inj (hxO.symm.trans (parse_of_setupO hsu))
  have hsz : p.sts.size = 1 := by rw [hpe]; exact (oneFull_staged hSP hsu).1
  have hcirc : circOf r s = cc := by unfold circOf; rw [hpc']; rfl
  -- the last message
  have hk' : Bd.DE.V.k (Claims.Recursion.xI D x) = kI x := by rw [hk, hxI]
  rw [hk', hfin, hLm] at hfn
  rw [hk'] at hos
  obtain ⟨r1, hr1, hlast⟩ := finalR_last elemR hD (hl.trans hn.symm) hfn
  rw [hr] at hr1
  cases hr1
  have hrd := rdR_elemR r hl hpos
  rw [← hlast] at hrd
  -- the inner statement sets up at one table: two proofs
  obtain ⟨r2, rs, s0, spec, st0, hr2, -, hset, -, -, hkI⟩ := schedBridge hD
  rw [hr] at hr2
  cases hr2
  have hT : nProofs r = 2 := by
    unfold nProofs tablesR
    rw [← hxs, hset]
    rfl
  -- the side's proof
  obtain ⟨o₀, ot, rfl⟩ : ∃ o₀ ot, os = o₀ :: ot := by
    cases os with
    | nil => simp at hos; omega
    | cons o₀ ot => exact ⟨o₀, ot, rfl⟩
  have hsetI : FlockDeploy.setupOf false (Claims.Recursion.xI D x) = some (s0, spec, #[st0]) := by rw [hxI]; exact hset
  have hlig : ∀ R ℓ, openAt (sideAtR rdR (Claims.Recursion.xI D x) (Bd.opened (o₀ :: ot) last) q).lig R ℓ =
      openAt ((((List.range (nProofs r)).map (proofR r ws)).map opensOf).toArray[2 * q.table + q.rep]!).ligerito
        R ℓ := by
    intro R ℓ
    have hgl : ((Bd.opened (o₀ :: ot) last).getLast?).bind rdR =
        some ⟨stmtR r, Json.null, (((List.range (nProofs r)).map (proofR r ws)).map opensOf).toArray⟩ := by
      rw [hop, List.getLast?_concat, Option.bind_some, hrd]
    have hh : rowsOf ((Bd.opened (o₀ :: ot) last).headD []) = some _ := by
      rw [hop]
      exact rowsOf_joinR' _
    have ht' : ((Bd.opened (o₀ :: ot) last).tail.dropLast).mapM rowsOf = some _ := by
      rw [hop]
      simp only [List.map_cons, List.cons_append, List.tail_cons, List.dropLast_concat]
      exact mapM_map_some (fun o => rowsOf_joinR' _) ot
    show openAt (ligR rdR (Claims.Recursion.xI D x) (Bd.opened (o₀ :: ot) last) q) R ℓ = _
    unfold ligR
    rw [hsetI, hgl, hh, ht']
    rfl
  -- the level's openings
  have hq0 : q.table = 0 := h0
  have hlv : q.level = s := by rw [← hls]
  have hsch : (sideAtR rdR (Claims.Recursion.xI D x) (Bd.opened (o₀ :: ot) last) q).sch = schR r := by
    show schAt (Claims.Recursion.xI D x) q.table = schAt (stmtOf r) 0
    rw [hxI, hxs, hq0]
  have hrep : q.rep < 2 := (hrange _ q hq).1
  have hqq : q.query < ((schR r).levels[q.level]!).q := by
    have := (hrange _ q hq).2
    rwa [hq0, hxI, hxs] at this
  have hℓ : q.level < (schR r).levels.size := by
    by_contra hc
    rw [getElem!_neg _ _ hc] at hqq
    exact Nat.not_lt_zero _ hqq
  have hps : (((List.range (nProofs r)).map (proofR r ws)).map opensOf).toArray[2 * q.table + q.rep]! =
      opensOf (proofR r ws q.rep) := by
    rw [hq0, Nat.mul_zero, Nat.zero_add, getElem!_pos _ _ (by simp [hT, hrep])]
    simp
  obtain ⟨hO1, hO2⟩ := openAt_opensOf (proofR r ws q.rep) ((schR r).levels.size - 1) q.level
  obtain ⟨hP1, hP2⟩ := openAt_proofR r ws q.rep hℓ
  -- part `s`'s pieces read off `w`
  obtain ⟨r3, hr3, hiR, hfx, hpcs⟩ := lastPieces elemR hD
  rw [hr] at hr3
  cases hr3
  have hmem : ((levelPiece r hpos q.rep q.level).1.val, (levelPiece r hpos q.rep q.level).2) ∈
      (layAt (restR elemR)).pieceAt x (kI x - 1) := by
    rw [hpcs, restR, List.map_append]
    refine List.mem_append_right _ (List.mem_append_left _ (List.mem_map.2 ⟨_, ?_, rfl⟩))
    unfold elemR
    rw [dif_pos hpos]
    exact List.mem_cons_of_mem _ (List.mem_flatMap.2 ⟨q.rep, List.mem_range.2 (by omega),
      mem_proofPieces r hpos q.rep hℓ⟩)
  have hpi1 : (levelPiece r hpos q.rep q.level).1.val = s := by
    show partIdx r q.level = s
    unfold partIdx
    rw [if_pos (by omega), hlv]
  rw [hpi1] at hmem
  have hseg := readSegs_piece (L := L) (ws := ws) hAg.2.1 (by rw [hLr]; exact hiR) (by rw [hLf]; exact hfx)
    (by omega) (segs := (levelPiece r hpos q.rep q.level).2) (by rw [hLp]; exact hmem)
  simp only [levelPiece] at hseg
  rw [encLevel_levelSegs, encLevel_levelSegs] at hseg
  have hwA : ws[s] = wAt ws (partIdx r q.level) := by
    rw [wAt, List.getD_eq_getElem _ _ (by omega)]
    unfold partIdx
    rw [if_pos (by omega), hlv]
  have hd1 := rdLevel_encLevel (levelFits w (circOf r q.level) q.rep
    (by have := (schAt_small (stmtOf r) 0).2 q.level; unfold schR; omega)
    (lt_of_lt_of_le (Nat.pow_lt_pow_right (by norm_num) ((schAt_small (stmtOf r) 0).2 q.level).2.1) (by norm_num))
    (by have := (schAt_small (stmtOf r) 0).2 q.level; unfold schR; omega)) []
  rw [hseg, hwA] at hd1
  have hd2 := rdLevel_encLevel (levelProof_fits r ws q.rep q.level) []
  simp only [levelProof] at hd2
  rw [hd2] at hd1
  obtain ⟨hRw, hSw, -⟩ := Prod.mk.inj (Prod.mk.inj (Option.some.inj hd1)).1
  -- the copy's columns
  have hcopy : copyR (Claims.Recursion.xI D x) w p ht q =
      colBits w (circOf r q.level) (q.rep * ((schR r).levels[q.level]!).q + q.query) := by
    unfold copyR copyAtR qAt
    rw [hxI, hxs, hlv, hcirc]
    exact copyOf_colBits ht hsz hc0 w _
  rw [hsch, hlig, hps, hO1, hO2, hP1, hP2, hcopy]
  simp only [levelProof]
  rw [hRw, hSw, hL, hH, hxs, ← hlv]
  refine ⟨?_, ?_⟩
  · rw [lvlRows_get _ _ _ _ hqq]
    rfl
  · rw [sibsOf_lvlSibs _ _ _ _ _ hqq]
    rfl

end

end RecDeploy.OpenRows
