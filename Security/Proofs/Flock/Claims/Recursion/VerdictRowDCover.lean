import Proofs.Flock.Claims.Recursion.VerdictAssembly
import Proofs.Flock.Claims.Complete.Replay.LigeritoRun
import Proofs.Flock.Verifier.Merkle

/-!
# Row D's `rec-open` pieces at a decision: which pieces, their cover, and their holding

`DecAt.runs_assembled` takes row D's pieces, and that they hold (`hholds`) and cover every query (`hcover`), as
hypotheses. This file supplies the pieces and those two.

* `rowDPiecesOf`, `rowDPieces`: one piece per table, rep, level and query of the statements `setupOf` reads off the
  inner statement, at a copy `key` the layout names (which copy of a level's `rec-open` statement computes which
  query). The pieces read the inner statement only. `mem_rowDPiecesOf`: a piece is in the list exactly where its
  indices are in range and its key is `key`'s.
* `DecAt.rowDPieces_eq`, `DecAt.rowDPieces_cover`: at a decision, the pieces are the decided statements', and they
  cover every query of every level of every rep of every table: `runs_assembled`'s `hcover`, outright.
* `levelCheck`: level `ℓ`'s Merkle check under `plain` at positions `pos`, against the level's cap (`capAt`) on what
  the proof opens there (`openAt`). `queryOpens_of_check`: a passing check opens each query (`merkleCheck_opens`),
  which is the converse of `opened_plain_of`'s use of `merkleCheck_of`.
* `levelCheck_of_merkleOk`, `levelPass_of_merkleOk`: the honest run's checks (`FlockComplete.Replay.Lg.MerkleOk`, at
  hashes whose `plain` operations are the scheme's, e.g. `With.std` by `plain_named`) are `levelCheck`s at the
  positions its query coins draw, with the level's `q` of them.
* `DecAt.hholds_rowDPieces`: every piece holds (`hholds`), at the layout tie (`hside`) and coin tie (`hpos`) in
  `runs_assembled`'s form at each table, where at each rep the verifier rows run and every level's check passes at
  their positions (`hrun`).
* `DecAt.runs_open`: `runs_assembled` at these pieces, with `hcover` discharged and `hholds` from `hrun`.
* At one table: `rowDPiecesAt key i x` (table `i`'s pieces), `DecAt.rowDPiecesAt_cover`, and
  `DecAt.hholds_rowDPiecesAt` at `runs_assembled`'s own `hside` and `hpos`, where the rows run (`verifierRows_rep`) and
  every run's level checks pass (`hlev`; at the honest run, `DecAt.levels_of_merkleOk` from `MerkleOk` and the coin
  tie). `DecAt.rowD_target`: row D's four hypotheses of `runs_assembled` at table `i` (`TargetD` body), with
  the rows' run from row R's structure and sizes (`DecAt.rowsRun_of`). `DecAt.runs_openAt`: `runs_assembled` at them,
  with `hcover` and `hholds` discharged but for `hlev`.
-/

namespace RecDeploy.Rows

open Flock
open FlockDeploy (Coin FinalP tabsOf pointsOf shapeAt)
open FlockVBridge.Family (FramedAt ProofShaped roundSizes plain)
open FlockVBridge.AccChain (UnitSums ReadsEnds)
open Flock.RecAlgebra (verifierRows valOf NoEmpty structureOf)
open Flock.RecResiduals (portSizes groups)
open FlockSoundness.Discharge.Exec (repFrames)
open FlockSoundness.Discharge.Recursion (HClass)
open FlockSoundness.Refine (openAt)
open RecDeploy.OpenedPlain (QueryOpens)

/-! ## The pieces -/

/-- Table `t`'s schedule at the statements `sts`: `fast100` at its `m`, as `DecAt.files` reads it. -/
def schAt (sts : Array Stmt) (t : ℕ) : Flock.Schedule := (fast100 (tabsOf sts)[t]!.m).getD default

/-- **The `rec-open` pieces of the statements `sts`**: one per table `t`, rep `r < 2`, level `ℓ` of `t`'s schedule
and query `j < q` of the level, at the copy `key t r ℓ j`. -/
def rowDPiecesOf (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (sts : Array Stmt) : List OpenPiece :=
  (List.range sts.size).flatMap fun t => (List.range 2).flatMap fun r =>
    (List.range (schAt sts t).levels.size).flatMap fun ℓ =>
      (List.range ((schAt sts t).levels[ℓ]!).q).map fun j => ⟨t, r, ℓ, j, key t r ℓ j⟩

/-- **The pieces at the inner statement `x`**: those of the statements `setupOf` reads off it, none where it doesn't
set up. -/
def rowDPieces (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (x : Claims.Msg) : List OpenPiece :=
  match FlockDeploy.setupOf false x with
  | some v => rowDPiecesOf key v.2.2
  | none => []

theorem mem_rowDPiecesOf {key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ} {sts : Array Stmt} {q : OpenPiece} :
    q ∈ rowDPiecesOf key sts ↔ q.table < sts.size ∧ q.rep < 2 ∧ q.level < (schAt sts q.table).levels.size ∧
      q.query < ((schAt sts q.table).levels[q.level]!).q ∧ q.key = key q.table q.rep q.level q.query := by
  obtain ⟨t, r, ℓ, j, k⟩ := q
  simp only [rowDPiecesOf, List.mem_flatMap, List.mem_range, List.mem_map, OpenPiece.mk.injEq]
  constructor
  · rintro ⟨t', ht, r', hr, ℓ', hℓ, j', hj, rfl, rfl, rfl, rfl, rfl⟩
    exact ⟨ht, hr, hℓ, hj, rfl⟩
  · rintro ⟨ht, hr, hℓ, hj, rfl⟩
    exact ⟨t, ht, r, hr, ℓ, hℓ, j, hj, rfl, rfl, rfl, rfl, rfl⟩

/-! ## A passing level check opens its queries -/

/-- **Level `ℓ`'s Merkle check** of the proof `p` under `plain` at the positions `pos`: against the level's cap, on
the rows, siblings and salts `p` opens at the level, at the level's lanes. -/
def levelCheck (sch : Flock.Schedule) (cap : Array ByteArray) (p : LigeritoProof) (ℓ : ℕ) (pos : Array ℕ) :
    Flock.V Unit :=
  merkleCheck plain (FlockSoundness.Refine.capAt cap p ℓ) (sch.levels[ℓ]!).cap (sch.levels[ℓ]!).d pos
    (openAt p (sch.levels.size - 1) ℓ).1 (openAt p (sch.levels.size - 1) ℓ).2.1
    (openAt p (sch.levels.size - 1) ℓ).2.2 (2 ^ (sch.levels[ℓ]!).lanes)

/-- **A passing level check opens each of its queries** at its position (`merkleCheck_opens`). -/
theorem queryOpens_of_check {sch : Flock.Schedule} {cap : Array ByteArray} {p : LigeritoProof} {ℓ : ℕ}
    {pos : Array ℕ} (h : levelCheck sch cap p ℓ pos = .ok ()) {j : ℕ} (hj : j < pos.size) :
    QueryOpens sch cap p ℓ j pos[j]! := by
  unfold levelCheck at h
  have ho := merkleCheck_opens plain _ _ _ pos _ _ _ _ h j hj
  simp only [opens, Bool.and_eq_true, decide_eq_true_eq] at ho
  exact ⟨ho.1.1.1, ho.2⟩

theorem plain_named : plain.Named := fun s hs => by
  have e : MerkleScheme.find? (MerkleScheme.sha512 "flock-leaf/sha512-unsalted").id =
      some (MerkleScheme.sha512 "flock-leaf/sha512-unsalted") := rfl
  rw [e] at hs
  exact (Option.some.inj hs).symm

/-- `plain`'s operations at the standard hashes are the scheme's. -/
theorem plain_opsWith_std : plain.opsWith With.std = MerkleOps.ofScheme plain :=
  With.opsWith_std plain plain_named

/-- FlockComplete's level opening is the executable's cap and openings at the level. -/
theorem opening_eq (p : LigeritoProof) (cap : Array ByteArray) (R ℓ : ℕ) :
    FlockComplete.Replay.Lg.opening p cap R ℓ =
      (FlockSoundness.Refine.capAt cap p ℓ, (openAt p R ℓ).1, (openAt p R ℓ).2.1, (openAt p R ℓ).2.2) := by
  unfold FlockComplete.Replay.Lg.opening FlockSoundness.Refine.capAt openAt
  by_cases h0 : ℓ = 0 <;> by_cases h1 : ℓ < R <;> simp [h0, h1]

/-- **The honest run's Merkle check at level `ℓ` is `levelCheck`** at the positions the level's query coins draw
(`MerkleOk`, at hashes whose `plain` operations are the scheme's). -/
theorem levelCheck_of_merkleOk {hs : Hashes Flock.V} (hops : plain.opsWith hs = MerkleOps.ofScheme plain)
    {sch : Flock.Schedule} {cap : Array ByteArray} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {k : ℕ}
    (h : FlockComplete.Replay.Lg.MerkleOk hs plain sch cap p cf k) {ℓ : ℕ} (hℓ : ℓ < sch.levels.size) :
    levelCheck sch cap p ℓ (positions (sch.levels[ℓ]!)
      ((Array.range (sch.levels[ℓ]!).q).map (cf (k + FlockComplete.Replay.Lg.qOff sch ℓ)))) = .ok () := by
  have h' := h ℓ hℓ
  rw [hops, With.ofScheme_merkleCheck, opening_eq] at h'
  simpa only [levelCheck, FlockVBridge.Family.Conss.getD_default] using h'

/-- **Every level passes at positions `pos ℓ`** where they are the ones the honest run's query coins draw: `q` of
them (`positions_size`, at `fast100`'s seamed levels), and the level's check passes there. -/
theorem levelPass_of_merkleOk {hs : Hashes Flock.V} (hops : plain.opsWith hs = MerkleOps.ofScheme plain)
    {sch : Flock.Schedule} (hseam : ∀ lv ∈ sch.levels, FlockSoundness.Refine.LevelSeam lv)
    {cap : Array ByteArray} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {k : ℕ}
    (h : FlockComplete.Replay.Lg.MerkleOk hs plain sch cap p cf k) {pos : ℕ → Array ℕ}
    (hpos : ∀ ℓ < sch.levels.size, pos ℓ = positions (sch.levels[ℓ]!)
      ((Array.range (sch.levels[ℓ]!).q).map (cf (k + FlockComplete.Replay.Lg.qOff sch ℓ)))) :
    ∀ ℓ < sch.levels.size, (sch.levels[ℓ]!).q ≤ (pos ℓ).size ∧ levelCheck sch cap p ℓ (pos ℓ) = .ok () := by
  intro ℓ hℓ
  rw [hpos ℓ hℓ]
  have hmem : sch.levels[ℓ]! ∈ sch.levels := by
    rw [getElem!_pos sch.levels ℓ hℓ]
    exact Array.getElem_mem hℓ
  exact ⟨(FlockSoundness.Refine.positions_size _ (hseam _ hmem) _).ge, levelCheck_of_merkleOk hops h hℓ⟩

namespace DecAt

variable {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
  (D : DecAt rd x ms cs) (cls : HClass) (acc : ℕ → Array F128)

/-! ## The cover -/

/-- At a decision, the pieces are the decided statements'. -/
theorem rowDPieces_eq (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) : rowDPieces key x = rowDPiecesOf key D.sts := by
  obtain ⟨v, c₁, r₀, rss, fl, link, pf, hv, -, -, -, -, -, hd⟩ := FlockDeploy.decR_some D.dec
  have hs : D.d.sts = v.2.2 := congrArg (·.sts) hd
  rw [D.setup_eq.2, hs]
  simp only [rowDPieces, hv]

/-- **The pieces cover every query** of every level of rep `r` of table `i` (`runs_assembled`'s `hcover`). -/
theorem rowDPieces_cover (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) {i : ℕ} (hi : i < D.sts.size) (r : Fin 2) :
    ∀ ℓ < (D.files cls i).sch.levels.size, ∀ j < ((D.files cls i).sch.levels[ℓ]!).q,
      ∃ q ∈ rowDPieces key x, q.table = i ∧ q.rep = r.val ∧ q.level = ℓ ∧ q.query = j := by
  intro ℓ hℓ j hj
  refine ⟨⟨i, r.val, ℓ, j, key i r.val ℓ j⟩, ?_, rfl, rfl, rfl, rfl⟩
  rw [D.rowDPieces_eq key, mem_rowDPiecesOf]
  exact ⟨hi, r.isLt, hℓ, hj, rfl⟩

/-! ## The holding -/

/-- **Every piece holds** (`runs_assembled`'s `hholds`), at the layout tie `hside` and the coin tie `hpos` in
`runs_assembled`'s form at each table, where at each rep of each table the verifier rows run, and at their positions
every level has at least its `q` positions and its check passes on the rep's decoded proof (`hrun`; at the honest
run, `levelPass_of_merkleOk`). -/
theorem hholds_rowDPieces (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
    (sideAt : Claims.Msg → List Claims.Msg → OpenPiece → OpenSide) (posAt : Claims.Msg → List Coin → OpenPiece → ℕ)
    (hside : ∀ t < D.sts.size, ∀ r : Fin 2, ∀ q ∈ rowDPieces key x, q.table = t → q.rep = r.val →
      sideAt x ms q = D.side cls t r)
    (hpos : ∀ t < D.sts.size, ∀ r : Fin 2, ∀ v info kappa,
      verifierRows (D.sh cls t) (D.rep cls acc t r).rounds (D.rep cls acc t r).extras (D.rep cls acc t r).comb =
        .ok (v, info, kappa) →
      ∀ q ∈ rowDPieces key x, q.table = t → q.rep = r.val → posAt x cs q = (info.lg.conss[q.level]!).pos[q.query]!)
    (hrun : ∀ t < D.sts.size, ∀ r : Fin 2, ∃ v info kappa,
      verifierRows (D.sh cls t) (D.rep cls acc t r).rounds (D.rep cls acc t r).extras (D.rep cls acc t r).comb =
        .ok (v, info, kappa) ∧ ∀ ℓ < (D.files cls t).sch.levels.size,
        ((D.files cls t).sch.levels[ℓ]!).q ≤ (info.lg.conss[ℓ]!).pos.size ∧
        levelCheck (D.files cls t).sch ((D.files cls t).proof r).cap ((D.files cls t).proof r).ligerito ℓ
          (info.lg.conss[ℓ]!).pos = .ok ()) :
    ∀ q ∈ rowDPieces key x, openHolds sideAt posAt q x ms cs := by
  intro q hq
  have hq' := hq
  rw [D.rowDPieces_eq key, mem_rowDPiecesOf] at hq'
  obtain ⟨ht, hr, hℓ, hj, -⟩ := hq'
  obtain ⟨v, info, kappa, hv, hlev⟩ := hrun q.table ht ⟨q.rep, hr⟩
  obtain ⟨hle, hc⟩ := hlev q.level hℓ
  unfold openHolds
  rw [hside q.table ht ⟨q.rep, hr⟩ q hq rfl rfl, hpos q.table ht ⟨q.rep, hr⟩ v info kappa hv q hq rfl rfl]
  exact queryOpens_of_check hc (Nat.lt_of_lt_of_le hj hle)

/-! ## The assembly at these pieces -/

/-- **Table `i`'s reps run at the decision** (`runs_assembled`) at the pieces `rowDPieces key x`: their cover is
`rowDPieces_cover`, and they hold by `hholds_rowDPieces` at the layout and coin ties and the verifier rows' passing
level checks at every table. -/
theorem runs_open {xo : Claims.Msg} {sto : Claims.Recursion.CompiledStatement} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : x = xIOf xo) {i : ℕ} (hi : i < D.sts.size)
    (hti : i < D.spec.tables.size)
    (hframed : ∀ r : Fin 2, FramedAt ((D.files cls i).stream r).rounds 0 (repFrames (D.files cls i).st
      (D.files cls i).sch ((D.files cls i).spec.domain (D.files cls i).spec.tables[(D.files cls i).i]! r).toUTF8
      (2 + (D.files cls i).extras.size) ((D.files cls i).proof r)
      (valOf (D.sh cls i) (D.rep cls acc i r).w (D.rep cls acc i r).comb (D.rep cls acc i r).cons .T)))
    (hpts : D.sess.points = pointsOf D.spec D.d.c₁)
    (hk : ∀ r : Fin 2, 2 * i + r < D.d.f.ps.size)
    (hsh : ∀ r : Fin 2,
      ProofShaped (shapeAt (tabsOf D.sts) (pointsOf D.spec D.d.c₁) (2 * i + r)) (D.d.f.ps[2 * i + r]'(hk r)))
    (hL : ∀ r, D.RowL cls acc i r)
    (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
    (sideAt : Claims.Msg → List Claims.Msg → OpenPiece → OpenSide) (posAt : Claims.Msg → List Coin → OpenPiece → ℕ)
    (hside : ∀ t < D.sts.size, ∀ r : Fin 2, ∀ q ∈ rowDPieces key x, q.table = t → q.rep = r.val →
      sideAt x ms q = D.side cls t r)
    (hpos : ∀ t < D.sts.size, ∀ r : Fin 2, ∀ v info kappa,
      verifierRows (D.sh cls t) (D.rep cls acc t r).rounds (D.rep cls acc t r).extras (D.rep cls acc t r).comb =
        .ok (v, info, kappa) →
      ∀ q ∈ rowDPieces key x, q.table = t → q.rep = r.val → posAt x cs q = (info.lg.conss[q.level]!).pos[q.query]!)
    (hrun : ∀ t < D.sts.size, ∀ r : Fin 2, ∃ v info kappa,
      verifierRows (D.sh cls t) (D.rep cls acc t r).rounds (D.rep cls acc t r).extras (D.rep cls acc t r).comb =
        .ok (v, info, kappa) ∧ ∀ ℓ < (D.files cls t).sch.levels.size,
        ((D.files cls t).sch.levels[ℓ]!).q ≤ (info.lg.conss[ℓ]!).pos.size ∧
        levelCheck (D.files cls t).sch ((D.files cls t).proof r).cap ((D.files cls t).proof r).ligerito ℓ
          (info.lg.conss[ℓ]!).pos = .ok ())
    {S : RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hS : structureOf (D.sh cls i) = .ok S) (hg : groups S (portSizes (D.sh cls i)) = .ok gs)
    (hsz : ∀ r : Fin 2, (D.rep cls acc i r).rounds.toList.map Array.size = roundSizes (D.sh cls i))
    (hw : ∀ r : Fin 2, (D.rep cls acc i r).w.size = S.nw) (hparts : ∀ r : Fin 2, D.PartsZero cls acc i r gs)
    (accC : Fin 2 → ℕ → Array F128) (p0 : Fin 2 → ℕ)
    (hu : ∀ r : Fin 2, UnitSums (D.sh cls i) (accC r) (D.opens cls acc i r (p0 r)))
    (hrd : ∀ r : Fin 2, ReadsEnds (D.sh cls i) (D.rep cls acc i r) (accC r) (D.opens cls acc i r (p0 r))) :
    (D.files cls i).Runs :=
  D.runs_assembled cls acc hst hf hx hi hti hframed hpts hk hsh hL (rowDPieces key x) sideAt posAt
    (D.hholds_rowDPieces cls acc key sideAt posAt hside hpos hrun) (hside i hi) (hpos i hi)
    (fun r => D.rowDPieces_cover cls key hi r) hS hg hsz hw hparts accC p0 hu hrd

/-! ## At one table -/

/-- **Table `i`'s pieces** at the inner statement `x`. -/
def rowDPiecesAt (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (i : ℕ) (x : Claims.Msg) : List OpenPiece :=
  (rowDPieces key x).filter fun q => q.table == i

theorem mem_rowDPiecesAt {key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ} {i : ℕ} {q : OpenPiece} :
    q ∈ rowDPiecesAt key i x ↔ q ∈ rowDPieces key x ∧ q.table = i := by
  simp [rowDPiecesAt]

/-- **Table `i`'s pieces cover every query** of every level of its rep `r` (`runs_assembled`'s `hcover`). -/
theorem rowDPiecesAt_cover (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) {i : ℕ} (hi : i < D.sts.size) (r : Fin 2) :
    ∀ ℓ < (D.files cls i).sch.levels.size, ∀ j < ((D.files cls i).sch.levels[ℓ]!).q,
      ∃ q ∈ rowDPiecesAt key i x, q.table = i ∧ q.rep = r.val ∧ q.level = ℓ ∧ q.query = j := by
  intro ℓ hℓ j hj
  obtain ⟨q, hq, ht, hr, hl, hq'⟩ := D.rowDPieces_cover cls key hi r ℓ hℓ j hj
  exact ⟨q, mem_rowDPiecesAt.2 ⟨hq, ht⟩, ht, hr, hl, hq'⟩

/-- **Table `i`'s pieces hold** (`runs_assembled`'s `hholds`), at `runs_assembled`'s `hside` and `hpos`, where each
rep's verifier rows run (`hrun`; `verifierRows_rep`) and, at any run of them, every level has at least its `q`
positions and its check passes on the rep's decoded proof (`hlev`; at the honest run, `levels_of_merkleOk`). -/
theorem hholds_rowDPiecesAt (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) {i : ℕ}
    (sideAt : Claims.Msg → List Claims.Msg → OpenPiece → OpenSide) (posAt : Claims.Msg → List Coin → OpenPiece → ℕ)
    (hside : ∀ r : Fin 2, ∀ q ∈ rowDPiecesAt key i x, q.table = i → q.rep = r.val → sideAt x ms q = D.side cls i r)
    (hpos : ∀ r : Fin 2, ∀ v info kappa,
      verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
        .ok (v, info, kappa) →
      ∀ q ∈ rowDPiecesAt key i x, q.table = i → q.rep = r.val →
        posAt x cs q = (info.lg.conss[q.level]!).pos[q.query]!)
    (hrun : ∀ r : Fin 2, ∃ y,
      verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
        .ok y)
    (hlev : ∀ r : Fin 2, ∀ v info kappa,
      verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
        .ok (v, info, kappa) →
      ∀ ℓ < (D.files cls i).sch.levels.size, ((D.files cls i).sch.levels[ℓ]!).q ≤ (info.lg.conss[ℓ]!).pos.size ∧
        levelCheck (D.files cls i).sch ((D.files cls i).proof r).cap ((D.files cls i).proof r).ligerito ℓ
          (info.lg.conss[ℓ]!).pos = .ok ()) :
    ∀ q ∈ rowDPiecesAt key i x, openHolds sideAt posAt q x ms cs := by
  intro q hq
  obtain ⟨hq0, hti⟩ := mem_rowDPiecesAt.1 hq
  rw [D.rowDPieces_eq key, mem_rowDPiecesOf] at hq0
  obtain ⟨-, hr, hℓ, hj, -⟩ := hq0
  rw [hti] at hℓ hj
  obtain ⟨⟨v, info, kappa⟩, hv⟩ := hrun ⟨q.rep, hr⟩
  obtain ⟨hle, hc⟩ := hlev ⟨q.rep, hr⟩ v info kappa hv q.level hℓ
  unfold openHolds
  rw [hside ⟨q.rep, hr⟩ q hq hti rfl, hpos ⟨q.rep, hr⟩ v info kappa hv q hq hti rfl]
  exact queryOpens_of_check hc (Nat.lt_of_lt_of_le hj hle)

/-- **`hlev` at the honest run**: where rep `r`'s proof passes FlockComplete's Merkle checks (`MerkleOk`) at the query
coins `cf` from round `k`, and any run of the verifier rows draws its positions from those coins (`htie`, the coin
tie), every level passes at the rows' positions. -/
theorem levels_of_merkleOk {i : ℕ} (r : Fin 2) {hs : Hashes Flock.V}
    (hops : plain.opsWith hs = MerkleOps.ofScheme plain)
    (hfast : Flock.fast100 (D.files cls i).st.m = some (D.files cls i).sch) {cf : ℕ → ℕ → F128} {k : ℕ}
    (h : FlockComplete.Replay.Lg.MerkleOk hs plain (D.files cls i).sch ((D.files cls i).proof r).cap
      ((D.files cls i).proof r).ligerito cf k)
    (htie : ∀ v info kappa,
      verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
        .ok (v, info, kappa) →
      ∀ ℓ < (D.files cls i).sch.levels.size, (info.lg.conss[ℓ]!).pos = positions ((D.files cls i).sch.levels[ℓ]!)
        ((Array.range ((D.files cls i).sch.levels[ℓ]!).q).map
          (cf (k + FlockComplete.Replay.Lg.qOff (D.files cls i).sch ℓ)))) :
    ∀ v info kappa,
      verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
        .ok (v, info, kappa) →
      ∀ ℓ < (D.files cls i).sch.levels.size, ((D.files cls i).sch.levels[ℓ]!).q ≤ (info.lg.conss[ℓ]!).pos.size ∧
        levelCheck (D.files cls i).sch ((D.files cls i).proof r).cap ((D.files cls i).proof r).ligerito ℓ
          (info.lg.conss[ℓ]!).pos = .ok () := fun v info kappa hv =>
  levelPass_of_merkleOk hops (FlockSoundness.Refine.fast100_level _ _ hfast) h
    (pos := fun ℓ => (info.lg.conss[ℓ]!).pos) (htie v info kappa hv)

/-- **Row D's four hypotheses of `runs_assembled`** at table `i` (`TargetD` body): table `i`'s pieces, at
`sideAt` and `posAt` with their ties `hside` and `hpos`, hold where the rows run (`hrun`) and every run's level checks
pass (`hlev`), and cover every query. -/
theorem rowD_target (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) {i : ℕ} (hi : i < D.sts.size)
    (sideAt : Claims.Msg → List Claims.Msg → OpenPiece → OpenSide) (posAt : Claims.Msg → List Coin → OpenPiece → ℕ)
    (hside : ∀ r : Fin 2, ∀ q ∈ rowDPiecesAt key i x, q.table = i → q.rep = r.val → sideAt x ms q = D.side cls i r)
    (hpos : ∀ r : Fin 2, ∀ v info kappa,
      verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
        .ok (v, info, kappa) →
      ∀ q ∈ rowDPiecesAt key i x, q.table = i → q.rep = r.val →
        posAt x cs q = (info.lg.conss[q.level]!).pos[q.query]!)
    (hrun : ∀ r : Fin 2, ∃ y,
      verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
        .ok y)
    (hlev : ∀ r : Fin 2, ∀ v info kappa,
      verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
        .ok (v, info, kappa) →
      ∀ ℓ < (D.files cls i).sch.levels.size, ((D.files cls i).sch.levels[ℓ]!).q ≤ (info.lg.conss[ℓ]!).pos.size ∧
        levelCheck (D.files cls i).sch ((D.files cls i).proof r).cap ((D.files cls i).proof r).ligerito ℓ
          (info.lg.conss[ℓ]!).pos = .ok ()) :
    ∃ (pieces : List OpenPiece) (sideAt : Claims.Msg → List Claims.Msg → OpenPiece → OpenSide)
      (posAt : Claims.Msg → List Coin → OpenPiece → ℕ),
      (∀ q ∈ pieces, openHolds sideAt posAt q x ms cs) ∧
      (∀ r : Fin 2, ∀ q ∈ pieces, q.table = i → q.rep = r.val → sideAt x ms q = D.side cls i r) ∧
      (∀ r : Fin 2, ∀ v info kappa,
        verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
          .ok (v, info, kappa) →
        ∀ q ∈ pieces, q.table = i → q.rep = r.val → posAt x cs q = (info.lg.conss[q.level]!).pos[q.query]!) ∧
      (∀ r : Fin 2, ∀ ℓ < (D.files cls i).sch.levels.size, ∀ j < ((D.files cls i).sch.levels[ℓ]!).q,
        ∃ q ∈ pieces, q.table = i ∧ q.rep = r.val ∧ q.level = ℓ ∧ q.query = j) :=
  ⟨rowDPiecesAt key i x, sideAt, posAt, D.hholds_rowDPiecesAt cls acc key sideAt posAt hside hpos hrun hlev, hside,
    hpos, fun r => D.rowDPiecesAt_cover cls key hi r⟩

/-- **Each rep's verifier rows run** at row R's structure and round sizes (`verifierRows_rep`, with `noEmpty`). -/
theorem rowsRun_of {i : ℕ} (hti : i < D.spec.tables.size) {S : RecAlgebra.GfS}
    (hS : structureOf (D.sh cls i) = .ok S)
    (hsz : ∀ r : Fin 2, (D.rep cls acc i r).rounds.toList.map Array.size = roundSizes (D.sh cls i)) :
    ∀ r : Fin 2, ∃ y,
      verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
        .ok y :=
  fun r => D.verifierRows_rep cls acc i r hS (D.noEmpty cls acc hti r) (hsz r)

/-- **Table `i`'s reps run at the decision** (`runs_assembled`) at table `i`'s pieces, with `runs_assembled`'s own
`hside` and `hpos`: the cover is `rowDPiecesAt_cover`, and the pieces hold by `hholds_rowDPiecesAt` at the verifier
rows' run (`verifierRows_rep`) and their passing level checks (`hlev`). -/
theorem runs_openAt {xo : Claims.Msg} {sto : Claims.Recursion.CompiledStatement} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : x = xIOf xo) {i : ℕ} (hi : i < D.sts.size)
    (hti : i < D.spec.tables.size)
    (hframed : ∀ r : Fin 2, FramedAt ((D.files cls i).stream r).rounds 0 (repFrames (D.files cls i).st
      (D.files cls i).sch ((D.files cls i).spec.domain (D.files cls i).spec.tables[(D.files cls i).i]! r).toUTF8
      (2 + (D.files cls i).extras.size) ((D.files cls i).proof r)
      (valOf (D.sh cls i) (D.rep cls acc i r).w (D.rep cls acc i r).comb (D.rep cls acc i r).cons .T)))
    (hpts : D.sess.points = pointsOf D.spec D.d.c₁)
    (hk : ∀ r : Fin 2, 2 * i + r < D.d.f.ps.size)
    (hsh : ∀ r : Fin 2,
      ProofShaped (shapeAt (tabsOf D.sts) (pointsOf D.spec D.d.c₁) (2 * i + r)) (D.d.f.ps[2 * i + r]'(hk r)))
    (hL : ∀ r, D.RowL cls acc i r)
    (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
    (sideAt : Claims.Msg → List Claims.Msg → OpenPiece → OpenSide) (posAt : Claims.Msg → List Coin → OpenPiece → ℕ)
    (hside : ∀ r : Fin 2, ∀ q ∈ rowDPiecesAt key i x, q.table = i → q.rep = r.val → sideAt x ms q = D.side cls i r)
    (hpos : ∀ r : Fin 2, ∀ v info kappa,
      verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
        .ok (v, info, kappa) →
      ∀ q ∈ rowDPiecesAt key i x, q.table = i → q.rep = r.val →
        posAt x cs q = (info.lg.conss[q.level]!).pos[q.query]!)
    (hlev : ∀ r : Fin 2, ∀ v info kappa,
      verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
        .ok (v, info, kappa) →
      ∀ ℓ < (D.files cls i).sch.levels.size, ((D.files cls i).sch.levels[ℓ]!).q ≤ (info.lg.conss[ℓ]!).pos.size ∧
        levelCheck (D.files cls i).sch ((D.files cls i).proof r).cap ((D.files cls i).proof r).ligerito ℓ
          (info.lg.conss[ℓ]!).pos = .ok ())
    {S : RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hS : structureOf (D.sh cls i) = .ok S) (hg : groups S (portSizes (D.sh cls i)) = .ok gs)
    (hsz : ∀ r : Fin 2, (D.rep cls acc i r).rounds.toList.map Array.size = roundSizes (D.sh cls i))
    (hw : ∀ r : Fin 2, (D.rep cls acc i r).w.size = S.nw) (hparts : ∀ r : Fin 2, D.PartsZero cls acc i r gs)
    (accC : Fin 2 → ℕ → Array F128) (p0 : Fin 2 → ℕ)
    (hu : ∀ r : Fin 2, UnitSums (D.sh cls i) (accC r) (D.opens cls acc i r (p0 r)))
    (hrd : ∀ r : Fin 2, ReadsEnds (D.sh cls i) (D.rep cls acc i r) (accC r) (D.opens cls acc i r (p0 r))) :
    (D.files cls i).Runs :=
  D.runs_assembled cls acc hst hf hx hi hti hframed hpts hk hsh hL (rowDPiecesAt key i x) sideAt posAt
    (D.hholds_rowDPiecesAt cls acc key sideAt posAt hside hpos (D.rowsRun_of cls acc hti hS hsz) hlev)
    hside hpos (fun r => D.rowDPiecesAt_cover cls key hi r) hS hg hsz hw hparts accC p0 hu hrd

end DecAt

end RecDeploy.Rows
