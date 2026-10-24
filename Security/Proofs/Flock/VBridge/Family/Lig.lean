import Proofs.Flock.VBridge.Family.Encode
import Proofs.Flock.VBridge.Family.Openings
import Proofs.Flock.Soundness.Refine.Setup
import Proofs.Flock.Soundness.Discharge.Hidden.Defs

/-!
# The proof files' cap and Ligerito openings, read off V*'s layers and the inner messages

`proofOf` (Files.lean) takes the level-0 cap and each rep's `LigeritoProof` as parameters (`cap`, `lig`), and `Encodes`
asks their shapes (`CapShaped`, `LigShaped`) and the statement's region count (`RegionsFit`). Here they are defined, each
array at the length `decodeProof` reads (Proof.lean:94-151) by construction, so that the shapes are theorems.

* `capOf ms`: the `2^cap` entries of the first cap among the inner messages `ms` (the table's level-0 cap, `Steps`' first
  step), each its 64 digest bytes.
* `ligOf ms l r`: rep `r`'s openings. Level `ℓ`'s opening `j` is copy 0 of unit `unO ℓ r j` of the level's session
  (`openCol`): its row the first `2^lanes` elements of its `row` columns (`rowsOf`'s), its `h`-th sibling the 64 bytes
  from column `128 · 2^lanes + 512 h` (its `aux`, `RecOpen.sibForms`); no salts, the inner statement's leaves being
  unsalted (`flock-leaf/sha512-unsalted`, `Tags.withPlainLeaves`). Rep `r`'s recursive caps are the caps among `ms` after
  the level-0 one, rep 0's first (`tableTape`: rep 0's tape, then rep 1's), in level order (`ligMsgs`). The nonces are zero: the verifier absorbs them
  and checks nothing of them (Ligerito.lean:143, 146, `Transcript.pow`). The messages, the out-of-domain values and `yr`
  are empty: `proofOf` sets them from the rep's values.
* `capShaped_capOf`, `ligShaped_ligOf`: the shapes at any setup on unsalted SHA-512 (`SetupRealizes`' scheme), and
  `hCapS_of`, `hLigS_of` at the family: `vbridgeRead_assembled`'s `hcapS` and `hligS`. `capOf_of_capDig`: when the first cap message
  is `capDig a` for `2^cap` digests `a` (`Steps`' first step), `capOf` is `a`.
* `setupH_regions_le`: `Stmt.setupH` refuses more than 1,024 regions (`checkInRange`, Statement.lean:114,
  HmRow.lean:1339), as `Stmt.setupHidden` does (`Hidden.Spec.inRange`), so `RegionsFit` holds at a statement with a
  setup's regions (`regionsFit_of_setupH`, `regionsFit_of_spec`; `regionsOf`'s, `length_regionsOf`), and `hRegs_of`,
  `hRegs_of_spec` are `vbridgeRead_assembled`'s `hregs`.
-/

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.AccChain

local notation "𝔽" => FlockLevel3.GF128

/-! ## The region count -/

set_option maxHeartbeats 2000000 in
/-- **`Stmt.setupH` checks the statement range** (HmRow.lean:1339): every statement it returns passed `checkInRange`
at its `k_log`, its regions and its link points. -/
theorem setupH_inRange {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) :
    checkInRange st.c.kLog st.regions.size (PT_LOCAL + st.nbl) = .ok () := by
  unfold Stmt.setupH at h
  dsimp only at h
  simp only [throw_bind', pure_bind] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    assumption

/-- **`Stmt.setupH` refuses more than 1,024 regions** (`checkInRange_ok`). -/
theorem setupH_regions_le {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) : st.regions.size ≤ 1024 :=
  ((checkInRange_ok _ _ _).1 (setupH_inRange h)).2.1

/-- **What `Stmt.setupHidden` accepts has at most 1,024 regions** (its `checkInRange`, `Hidden.Spec.inRange`). -/
theorem spec_regions_le {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw : Option Lean.Json} {st : Stmt} (h : FlockSoundness.Discharge.Hidden.Spec tags cf pf coins tables draw st) :
    st.regions.size ≤ 1024 :=
  ((checkInRange_ok _ _ _).1 h.inRange).2.1

/-- **`RegionsFit` from the verifier's setup**: a statement with as many regions as one `Stmt.setupH` returned has
fewer than `2^48`. -/
theorem regionsFit_of_setupH {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {stH : Stmt}
    (hH : Stmt.setupH tags cf pf coins tables draw partition program = .ok stH) {S : Model.Statement}
    (hS : S.regions.length = stH.regions.size) : S.regions.length < 2 ^ 48 := by
  have := setupH_regions_le hH
  have h48 : (1024 : ℕ) < 2 ^ 48 := by norm_num
  omega

/-- **`RegionsFit` from the verifier's hidden setup** (`Stmt.setupHidden`'s `checkInRange`). -/
theorem regionsFit_of_spec {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw : Option Lean.Json} {stH : Stmt} (h : FlockSoundness.Discharge.Hidden.Spec tags cf pf coins tables draw stH)
    {S : Model.Statement} (hS : S.regions.length = stH.regions.size) : S.regions.length < 2 ^ 48 := by
  have := spec_regions_le h
  have h48 : (1024 : ℕ) < 2 ^ 48 := by norm_num
  omega

/-- A statement's model regions are as many as its regions. -/
theorem length_regionsOf (st : Stmt) : (regionsOf st).length = st.regions.size := by
  simp [regionsOf]

theorem TableSrc.regionsFit_of_setupH (s : TableSrc) {tags : Tags} {cf pf : ByteArray} {coins}
    {tables : List (String × ByteArray)} {draw partition : Option Lean.Json} {program : Option ByteArray} {stH : Stmt}
    (hH : Stmt.setupH tags cf pf coins tables draw partition program = .ok stH)
    (hS : s.S.regions.length = stH.regions.size) : s.RegionsFit :=
  Family.regionsFit_of_setupH hH hS

/-! ## The inner statement's path check -/

/-- **A query opens at the inner statement's scheme** (`opens`, Merkle.lean:59-62) exactly when its row has the lane
count, it has `d − c` siblings, its salt is empty, and SHA-512 of the row's bytes climbs to its cap entry: the climb
`sound_climbRow` (Open.lean) computes from the row's bytes. -/
theorem opens_plain {cap : Array ByteArray} {c d lanes pos : ℕ} {row : Array F128} {salt : ByteArray}
    {sibs : List ByteArray} :
    opens (MerkleScheme.sha512 "flock-leaf/sha512-unsalted") cap c d lanes pos row salt sibs = true ↔
      row.size = lanes ∧ sibs.length = d - c ∧ salt.size = 0 ∧
        climb (MerkleScheme.sha512 "flock-leaf/sha512-unsalted") (Sha512.hash (MerkleScheme.rowBytes row)) pos sibs =
          cap.getD (pos >>> (d - c)) ByteArray.empty := by
  unfold opens
  simp only [Bool.and_eq_true, decide_eq_true_eq, and_assoc]
  exact Iff.rfl

/-! ## Arrays at their lengths -/

/-- `n` bytes of `b` from byte 0, zero past its end. -/
def fitB (n : ℕ) (b : ByteArray) : ByteArray := ⟨Array.ofFn fun i : Fin n => b.get! i.val⟩

theorem size_fitB (n : ℕ) (b : ByteArray) : (fitB n b).size = n :=
  show (Array.ofFn fun i : Fin n => b.get! i.val).size = n from Array.size_ofFn

theorem fitB_of_size {n : ℕ} {b : ByteArray} (h : b.size = n) : fitB n b = b := by
  subst h
  cases b with
  | mk d =>
    show (⟨Array.ofFn fun i : Fin d.size => (ByteArray.mk d).get! i.val⟩ : ByteArray) = ⟨d⟩
    congr 1
    apply Array.ext Array.size_ofFn
    intro i _ h2
    rw [Array.getElem_ofFn]
    show d[i]! = d[i]
    exact getElem!_pos d i h2

/-- `w` elements of `a` from element 0, zero past its end. -/
def fitRow (w : ℕ) (a : Array F128) : Array F128 := Array.ofFn fun i : Fin w => a[i.val]!

theorem size_fitRow (w : ℕ) (a : Array F128) : (fitRow w a).size = w := Array.size_ofFn

theorem fitRow_of_size {w : ℕ} {a : Array F128} (h : a.size = w) : fitRow w a = a := by
  subst h
  apply Array.ext (size_fitRow _ _)
  intro i h1 _
  simp only [fitRow, Array.getElem_ofFn]
  exact getElem!_pos a i (by simpa [size_fitRow] using h1)

/-- A cap of `2^c` digests of `H` bytes, entry `i` from `e i`. -/
def capArr (H c : ℕ) (e : ℕ → ByteArray) : Array ByteArray := Array.ofFn fun i : Fin (2 ^ c) => fitB H (e i.val)

/-- `q` salts of `s` bytes for a layout that salts (`sl = some s`, `s > 0`), none else. -/
def saltsArr (sl : Option ℕ) (q : ℕ) (salt : ℕ → ByteArray) : Array ByteArray :=
  match sl with
  | some (s + 1) => Array.ofFn fun j : Fin q => fitB (s + 1) (salt j.val)
  | _ => #[]

theorem saltsOk_saltsArr {sh : Shape} (q : ℕ) (salt : ℕ → ByteArray) :
    ZkView.SaltsOk sh q (saltsArr sh.saltLen q salt) := by
  unfold ZkView.SaltsOk
  generalize sh.saltLen = sl
  rcases sl with _ | _ | s
  · exact rfl
  · exact rfl
  · refine ⟨Array.size_ofFn, fun x hx => ?_⟩
    have hx' : x ∈ Array.ofFn fun j : Fin q => fitB (s + 1) (salt j.val) := hx
    obtain ⟨j, rfl⟩ := Array.mem_ofFn.1 hx'
    exact size_fitB _ _

/-- A level's openings: `q` rows of `w` elements, `n` siblings of `H` bytes a row (row `j`'s `h`-th at `j · n + h`,
`sibsOf`), and the salts (`saltsArr`). -/
def levelOf (H : ℕ) (sl : Option ℕ) (q w n : ℕ) (row : ℕ → Array F128) (sib : ℕ → ℕ → ByteArray)
    (salt : ℕ → ByteArray) : LevelProof where
  rows := Array.ofFn fun j : Fin q => fitRow w (row j.val)
  siblings := Array.ofFn fun k : Fin (q * n) => fitB H (sib (k.val / n) (k.val % n))
  salts := saltsArr sl q salt

theorem capOk_capArr {sh : Shape} {l : Level} {H : ℕ} (hH : sh.hashLen = H) (e : ℕ → ByteArray) :
    ZkView.CapOk sh l (capArr H l.cap e) := by
  refine ⟨Array.size_ofFn, fun s hs => ?_⟩
  obtain ⟨i, rfl⟩ := Array.mem_ofFn.1 hs
  rw [size_fitB, hH]

theorem levelOk_levelOf {sh : Shape} {l : Level} {H : ℕ} (hH : sh.hashLen = H) (w : ℕ) (row : ℕ → Array F128)
    (sib : ℕ → ℕ → ByteArray) (salt : ℕ → ByteArray) :
    ZkView.LevelOk sh l w (levelOf H sh.saltLen l.q w (l.d - l.cap) row sib salt) := by
  refine ⟨Array.size_ofFn, fun r hr => ?_, Array.size_ofFn, fun s hs => ?_, ?_⟩
  · obtain ⟨j, rfl⟩ := Array.mem_ofFn.1 hr
    exact size_fitRow _ _
  · obtain ⟨k, rfl⟩ := Array.mem_ofFn.1 hs
    rw [size_fitB, hH]
  · exact saltsOk_saltsArr _ _

/-! ## A rep's Ligerito openings -/

/-- What a rep's Ligerito openings are read from: level `ℓ`'s opening `j`'s row, its `h`-th sibling and its salt, and
the `i`-th recursive cap's entry `e`. -/
structure LigRead where
  row : ℕ → ℕ → Array F128
  sib : ℕ → ℕ → ℕ → ByteArray
  salt : ℕ → ℕ → ByteArray
  cap : ℕ → ℕ → ByteArray

/-- **A rep's `LigeritoProof` from its reads**, at digests of `H` bytes, the salt layout `sl` and the schedule `sch`:
each level's openings (`levelOf`: level 0's rows of `2^lanes` elements, the others' of 16), the `R` recursive caps of
levels `1 … R` (`capArr`), and zero nonces, `R + 1` query, `3R + 2` claim and `R + 1` consistency ones. The cap, the
messages, the out-of-domain values and `yr` are left empty for `proofOf` to set. -/
def ligOfRead (H : ℕ) (sl : Option ℕ) (sch : Flock.Schedule) (x : LigRead) : LigeritoProof :=
  let R := sch.levels.size - 1
  let lvl := fun (i : ℕ) => sch.levels.getD i default
  let lev := fun (ℓ w : ℕ) => levelOf H sl (lvl ℓ).q w ((lvl ℓ).d - (lvl ℓ).cap) (x.row ℓ) (x.sib ℓ) (x.salt ℓ)
  { initialCap := #[]
    initialProof := lev 0 (2 ^ (lvl 0).lanes)
    recursiveCaps := Array.ofFn fun i : Fin R => capArr H (lvl (i.val + 1)).cap (x.cap i.val)
    recursiveProofs := Array.ofFn fun i : Fin (R - 1) => lev (i.val + 1) 16
    finalProof := { yr := #[], rows := (lev R 16).rows, siblings := (lev R 16).siblings, salts := (lev R 16).salts }
    messages := #[]
    queryNonces := Array.replicate (R + 1) 0
    oodValues := #[]
    claimNonces := Array.replicate (3 * R + 2) 0
    consistencyNonces := Array.replicate (R + 1) 0 }

/-- **The openings have the schedule's shape** at a setup whose digests are `H` bytes, at its salt layout. -/
theorem ligShaped_ligOfRead {st : Setup} {sch : Flock.Schedule} {H : ℕ} {sl : Option ℕ}
    (hH : (shapeOf st sch 0).hashLen = H) (hS : (shapeOf st sch 0).saltLen = sl) (x : LigRead) :
    LigShaped st sch (ligOfRead H sl sch x) := by
  subst hS
  refine ⟨levelOk_levelOf hH _ _ _ _, Array.size_ofFn, fun i hi => ?_, Array.size_ofFn, fun i hi => ?_,
    levelOk_levelOf hH _ _ _ _, Array.size_replicate, Array.size_replicate, Array.size_replicate,
    ⟨fun _ hx => (Array.mem_replicate.1 hx).2, fun _ hx => (Array.mem_replicate.1 hx).2,
      fun _ hx => (Array.mem_replicate.1 hx).2⟩⟩
  · simp only [ligOfRead, Array.getElem_ofFn]
    exact capOk_capArr hH _
  · simp only [ligOfRead, Array.getElem_ofFn]
    exact levelOk_levelOf hH _ _ _ _

/-- **The level-0 cap has the schedule's shape** at a setup whose digests are `H` bytes. -/
theorem capShaped_capArr {st : Setup} {sch : Flock.Schedule} {H : ℕ} (hH : (shapeOf st sch 0).hashLen = H)
    (e : ℕ → ByteArray) : CapShaped st sch (capArr H (sch.levels.getD 0 default).cap e) :=
  capOk_capArr hH e

/-! ## The inner messages -/

section msgs

variable {nT : ℕ} (T : Fin nT → Type)

open Classical in
/-- An inner message as a value of `α`: its value where the schedule's type at its slot is `α`. -/
noncomputable def msgAs (α : Type) (e : SMsg T) : Option α := if h : T e.1 = α then some (cast h e.2) else none

/-- The caps among the inner messages `ms`, in order. -/
noncomputable def capsIn (ms : List (List (SMsg T))) : List (Cap Dig) := ms.flatten.filterMap (msgAs T (Cap Dig))

/-- Entry `e` of the `k`-th cap among the inner messages, as bytes. -/
noncomputable def capEntry (ms : List (List (SMsg T))) (k e : ℕ) : ByteArray :=
  match (capsIn T ms)[k]? with
  | some c => (c e).1
  | none => ByteArray.empty

/-- **The level-0 cap** (`proofOf`'s `cap`): the `2^cap` entries of the first cap among the inner messages, each its
64 bytes. -/
noncomputable def capOf (sch : Flock.Schedule) (ms : List (List (SMsg T))) : Array ByteArray :=
  capArr 64 (sch.levels.getD 0 default).cap (capEntry T ms 0)

/-- **`CapShaped`** at a setup on the inner statement's unsalted SHA-512. -/
theorem capShaped_capOf {st : Setup} {sch : Flock.Schedule}
    (hms : st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted") (ms : List (List (SMsg T))) :
    CapShaped st sch (capOf T sch ms) :=
  capShaped_capArr (shapeOf_plain (n := 0) hms).1 _

/-- **`capOf` is the cap the first cap message holds**: when that message is `capDig a` (`Steps`' first step) for `2^cap`
digests `a`, `capOf` is `a`. -/
theorem capOf_of_capDig {sch : Flock.Schedule} {ms : List (List (SMsg T))} {a : Array ByteArray}
    (h0 : (capsIn T ms)[0]? = some (capDig a)) (ha : a.size = 2 ^ (sch.levels.getD 0 default).cap)
    (h64 : ∀ x ∈ a, x.size = 64) : capOf T sch ms = a := by
  apply Array.ext (by rw [capOf, capArr, Array.size_ofFn, ha])
  intro i h1 h2
  simp only [capOf, capArr, Array.getElem_ofFn, capEntry, h0, capDig]
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem h2, Option.getD_some,
    toDig_val (h64 _ (Array.getElem_mem h2)), fitB_of_size (h64 _ (Array.getElem_mem h2))]

end msgs

/-! ## At the family -/

section family

open FlockSoundness.Game FlockSoundness.Discharge.Exec FlockSoundness.Audit FlockVBridge.InnerAccepts
open FlockVBridge.Stages

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂)

/-- **Rep `r`'s reads in V*'s layers `l` and the inner messages `ms`**: level `ℓ`'s opening `j` is copy 0 of unit
`unO ℓ r j` of the level's session (`openCol`); its row the first `2^lanes` elements of its columns (`rowsOf`), its
`h`-th sibling the 64 bytes of `aux` from column `128 · 2^lanes + 512 h` (`RecOpen.ports`, `RecOpen.sibForms`); no salt
(the inner statement's leaves are unsalted, and V* holds none); rep `r`'s `i`-th recursive cap the `(1 + r R + i)`-th cap
among `ms`. -/
noncomputable def ligRead (ms : List (List (SMsg T))) (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (r : Fin 2) :
    LigRead where
  row ℓ j := elems (F.openCol l r ℓ j) 0 (2 ^ (sch.levels.getD ℓ default).lanes)
  sib ℓ j h := bytesAt (F.openCol l r ℓ j) (128 * 2 ^ (sch.levels.getD ℓ default).lanes + 512 * h) 64
  salt _ _ := ByteArray.empty
  cap i e := capEntry T ms (1 + r.val * (sch.levels.size - 1) + i) e

/-- **Rep `r`'s Ligerito openings** (`proofOf`'s `lig`), at the inner statement's 64-byte digests, without salts. -/
noncomputable def ligOf (ms : List (List (SMsg T))) (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (r : Fin 2) :
    LigeritoProof :=
  ligOfRead 64 none sch (F.ligRead ms l r)

/-- **`LigShaped`** at a setup on the inner statement's unsalted SHA-512, whether it opens salts (`some 0`) or not:
`saltsArr` gives none either way. -/
theorem ligShaped_ligOf {st : Setup} (hms : st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted")
    (ms : List (List (SMsg T))) (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (r : Fin 2) :
    LigShaped st sch (F.ligOf ms l r) := by
  obtain ⟨hH, hsl⟩ := shapeOf_plain (sch := sch) (n := 0) hms
  rcases hsl with hsl | hsl
  · exact ligShaped_ligOfRead hH hsl _
  · show LigShaped st sch (ligOfRead 64 (some 0) sch (F.ligRead ms l r))
    exact ligShaped_ligOfRead hH hsl _

variable (dg : ByteArray → ByteArray) (N k Rw : ℕ) (st : Reg → L.Ω → Setup)

/-- **`vbridgeRead_assembled`'s `hcapS`** at `cap l := capOf (w l)`, for any inner messages `w l`, from the setup's
scheme (`SetupRealizes`). -/
theorem hCapS_of (hset : ∀ R ω, SetupRealizes (st R ω) ((F.x R ω).toStatement regs) sch cls.ptLocal cls.mPts)
    (w : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → List (List (SMsg T))) :
    F.AtAccepted dg N k Rw fun R ω _ l => CapShaped (st R ω) sch (capOf T sch (w l)) :=
  fun R ω _ _ _ _ _ => capShaped_capOf T (hset R ω).2.2.2.1 _

/-- **`vbridgeRead_assembled`'s `hligS`** at `lig l := ligOf (w l) l`, for any inner messages `w l`, from the setup's
scheme (`SetupRealizes`). -/
theorem hLigS_of (hset : ∀ R ω, SetupRealizes (st R ω) ((F.x R ω).toStatement regs) sch cls.ptLocal cls.mPts)
    (w : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → List (List (SMsg T))) :
    F.AtAccepted dg N k Rw fun R ω _ l => ∀ r, LigShaped (st R ω) sch (F.ligOf (w l) l r) :=
  fun R ω _ _ _ _ _ r => F.ligShaped_ligOf (hset R ω).2.2.2.1 _ _ r

/-- **`vbridgeRead_assembled`'s `hregs`**: the inner statement has `regs`' regions, as many as a statement `Stmt.setupH`
returned (`regionsFit_of_setupH`). -/
theorem hRegs_of {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {stH : Stmt}
    (hH : Stmt.setupH tags cf pf coins tables draw partition program = .ok stH) (hlen : regs.length = stH.regions.size) :
    ∀ R ω, ((F.x R ω).toStatement regs).regions.length < 2 ^ 48 :=
  fun _ _ => regionsFit_of_setupH hH (S := (F.x _ _).toStatement regs) hlen

/-- **`hregs` from the verifier's hidden setup** (`Stmt.setupHidden`, `Hidden.Spec.inRange`). -/
theorem hRegs_of_spec {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw : Option Lean.Json} {stH : Stmt} (h : FlockSoundness.Discharge.Hidden.Spec tags cf pf coins tables draw stH)
    (hlen : regs.length = stH.regions.size) : ∀ R ω, ((F.x R ω).toStatement regs).regions.length < 2 ^ 48 :=
  fun _ _ => regionsFit_of_spec h (S := (F.x _ _).toStatement regs) hlen

end VStarFamily

end family

end FlockVBridge.Family
