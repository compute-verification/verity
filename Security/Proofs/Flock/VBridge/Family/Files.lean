import Proofs.Flock.VBridge.Family
import Proofs.Flock.VBridge.Runs.Rep

/-!
# The family's table files: `HSch`, `HFiles`, `HRuns`

`files` is a proof device: `vbridge_of_family` reads it only through `TableFiles.accepts`, which routes the inner
transcript through the verifier of record (`verifyRep`, then `table_refines_plain`). So the files are made from what the
family already reads, at each history and registration:

* the proof file of rep `r` (`proofOf`) holds the messages w holds (`decoded`, at `F.rep`), the verifier's `final_c`
  at the rep's coins, the PCS parameters the verifier pins (`pcsParams`) and the level-0 cap and Ligerito openings
  V*'s layers hold (`cap`, `lig`, parameters of the definition);
* rep `r`'s stream (`streamRounds`) has one round per coin array of `F.rep`'s rounds (the verifier's coins at `t`), each
  issuing its coins and retaining the frames `verifyRep` absorbs before it (`frameAt`, `run_of_frames`' frames);
* the link points are the inner transcript's first coin (`ptsOf`), and both reps' tapes the inner transcript's steps
  after them (`Steps`).

`hSch_of`, `hFiles_of` and `hRuns_of` prove the three obligations for every family whose `files` is `filesOf`, from
named `Prop`s, one per fact no check here establishes: the inner statement's setup (`SetupRealizes`), V*'s shape
(`ShapeIs`), the proof files' encodings (`TableSrc.Encodes`), the inner transcript V*'s layers hold
(`TableSrc.TranscriptIs`), the rounds' coin counts (`TableSrc.Coins`), the reps' extra claims and combs
(`TableSrc.ExtrasAre`, `CombFolds`), and the Ligerito openings (`OpenedAll`).
-/

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages
open Flock.RecAlgebra (decoded valOf verifierRows)

local notation "𝔽" => FlockLevel3.GF128

local instance lawfulBEqByteArray : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := by show (a.data == a.data) = true; exact beq_self_eq_true _

/-! ## A rep's stream -/

/-- **The bytes round `j` of a rep's stream retains** (`run_of_frames`' frames): the zerocheck's rounds from the
stream's domain (Piop.lean), the lincheck's from the zerocheck's last messages, one ring switch per claim, the
batching coins', then the Ligerito rounds' chunks, for `n` extra claims and the batched value `T`. -/
def frameAt (st : Setup) (sch : Flock.Schedule) (dom : ByteArray) (n : ℕ) (p : Proof) (T : F128) (j : ℕ) :
    List UInt8 :=
  if j = 0 then (Transcript.header 1 0 dom.size ++ Transcript.padded dom).data.toList ++
      encs [.label "flock-r1cs-v0".toUTF8, .bytes st.digest, .bytes (p.cap.foldl (· ++ ·) ByteArray.empty),
        .label "flock-zerocheck-v0".toUTF8, .sq 2 6]
  else if j = 1 then encs [.sq 2 (st.m - 13)]
  else if j = 2 then encs [.vec p.zerocheck.round1ab.toList, .vec p.zerocheck.round1c.toList, .sq 1 1]
  else if j < 3 + (st.m - 6) then
    encs [.one (p.zerocheck.rounds[j - 3]!).1, .one (p.zerocheck.rounds[j - 3]!).2, .sq 1 1]
  else if j = 3 + (st.m - 6) then
    encs [.one p.zerocheck.finalA, .one p.zerocheck.finalB] ++ encs [.label "flock-lincheck-v0".toUTF8, .sq 1 1]
  else if j = 3 + (st.m - 6) + 1 then encs [.sq 1 1]
  else if j < 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) then
    encs [.one (p.lincheck.rounds[j - (3 + (st.m - 6) + 2)]!).1, .one (p.lincheck.rounds[j - (3 + (st.m - 6) + 2)]!).2,
      .sq 1 1]
  else if j = 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) then encs [.vec p.lincheck.zPartial.toList, .sq 1 1]
  else if j < 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + (2 + n) then
    (rsFramed (rsPending ByteArray.empty (j - (3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1)))
      p.ringSwitches[j - (3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1)]!).data.toList
  else if j = 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + (2 + n) then
    (ByteArray.empty ++ Transcript.header 4 2 (2 + n)).data.toList
  else encs ((chunks (ligFrames sch T p.cap p.ligerito)).1.getD
    (j - (3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + (2 + n) + 1)) [])

/-- **A rep's recorded rounds**: round `j` issues the coins `coins[j]` and retains the bytes `fr j`. -/
def streamRounds (coins : Array (Array F128)) (fr : ℕ → List UInt8) : Array Round :=
  coins.mapIdx fun j c => { n := c.size, sha := ByteArray.empty, coins := c, bytes := some ⟨(fr j).toArray⟩ }

theorem streamRounds_get {coins : Array (Array F128)} {fr : ℕ → List UInt8} {j : ℕ} (h : j < coins.size) :
    (streamRounds coins fr)[j]! =
      { n := coins[j].size, sha := ByteArray.empty, coins := coins[j], bytes := some ⟨(fr j).toArray⟩ } := by
  unfold streamRounds
  rw [getElem!_pos _ j (by simpa using h), Array.getElem_mapIdx]

theorem streamRounds_coins (coins : Array (Array F128)) (fr : ℕ → List UInt8) :
    (streamRounds coins fr).map (·.coins) = coins := by
  apply Array.ext (by simp [streamRounds])
  intro i h₁ h₂
  simp [streamRounds]

theorem streamRounds_wf (coins : Array (Array F128)) (fr : ℕ → List UInt8) : RoundsWF (streamRounds coins fr) := by
  intro i hi
  rw [streamRounds_get (by simpa [streamRounds] using hi)]

theorem streamRounds_retains {coins : Array (Array F128)} {fr : ℕ → List UInt8} {j : ℕ} {L : List UInt8}
    (h : j < coins.size) (hL : fr j = L) : Retains (streamRounds coins fr)[j]! L := by
  rw [streamRounds_get h]
  simp [Retains, hL]

theorem streamRounds_n {coins : Array (Array F128)} {fr : ℕ → List UInt8} {j c : ℕ} (h : j < coins.size)
    (hc : (coins[j]!).size = c) : ((streamRounds coins fr)[j]!).n = c := by
  rw [streamRounds_get h]
  rw [getElem!_pos coins j h] at hc
  exact hc

theorem answers_of_retains' {r : Round} {B : ByteArray} (h : Retains r B.data.toList) : r.answers B := by
  have e := answers_of_retains h
  have hB : (⟨B.data.toList.toArray⟩ : ByteArray) = B := by cases B; simp
  rwa [hB] at e

/-! ### `frameAt`'s cases -/

section frames

variable (st : Setup) (sch : Flock.Schedule) (dom : ByteArray) (n : ℕ) (p : Proof) (T : F128)

theorem frameAt_zero : frameAt st sch dom n p T 0 =
    (Transcript.header 1 0 dom.size ++ Transcript.padded dom).data.toList ++
      encs [.label "flock-r1cs-v0".toUTF8, .bytes st.digest, .bytes (p.cap.foldl (· ++ ·) ByteArray.empty),
        .label "flock-zerocheck-v0".toUTF8, .sq 2 6] := by
  unfold frameAt; rw [ite_eq_left rfl]

theorem frameAt_one : frameAt st sch dom n p T 1 = encs [.sq 2 (st.m - 13)] := by
  unfold frameAt; rw [ite_eq_right (by omega), ite_eq_left rfl]

theorem frameAt_two : frameAt st sch dom n p T 2 =
    encs [.vec p.zerocheck.round1ab.toList, .vec p.zerocheck.round1c.toList, .sq 1 1] := by
  unfold frameAt; rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left rfl]

theorem frameAt_zc {j i : ℕ} (hj : j = 3 + i) (hi : i < st.m - 6) : frameAt st sch dom n p T j =
    encs [.one (p.zerocheck.rounds[i]!).1, .one (p.zerocheck.rounds[i]!).2, .sq 1 1] := by
  unfold frameAt
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left (by omega), show j - 3 = i by omega]

theorem frameAt_alpha {j : ℕ} (hj : j = 3 + (st.m - 6)) : frameAt st sch dom n p T j =
    encs [.one p.zerocheck.finalA, .one p.zerocheck.finalB] ++ encs [.label "flock-lincheck-v0".toUTF8, .sq 1 1] := by
  unfold frameAt
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left hj]

theorem frameAt_beta {j : ℕ} (hj : j = 3 + (st.m - 6) + 1) : frameAt st sch dom n p T j = encs [.sq 1 1] := by
  unfold frameAt
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left hj]

theorem frameAt_lc {j i : ℕ} (hj : j = 3 + (st.m - 6) + 2 + i) (hi : i < st.fold.kLog - 6) :
    frameAt st sch dom n p T j =
      encs [.one (p.lincheck.rounds[i]!).1, .one (p.lincheck.rounds[i]!).2, .sq 1 1] := by
  unfold frameAt
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    ite_eq_right (by omega), ite_eq_left (by omega), show j - (3 + (st.m - 6) + 2) = i by omega]

theorem frameAt_zp {j : ℕ} (hj : j = 3 + (st.m - 6) + 2 + (st.fold.kLog - 6)) :
    frameAt st sch dom n p T j = encs [.vec p.lincheck.zPartial.toList, .sq 1 1] := by
  unfold frameAt
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left hj]

theorem frameAt_rs {j k : ℕ} (hj : j = 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + k) (hk : k < 2 + n) :
    frameAt st sch dom n p T j = (rsFramed (rsPending ByteArray.empty k) p.ringSwitches[k]!).data.toList := by
  unfold frameAt
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left (by omega),
    show j - (3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1) = k by omega]

theorem frameAt_gamma {j : ℕ} (hj : j = 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + (2 + n)) :
    frameAt st sch dom n p T j = (ByteArray.empty ++ Transcript.header 4 2 (2 + n)).data.toList := by
  unfold frameAt
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left hj]

theorem frameAt_lig {j i : ℕ} (hj : j = 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + (2 + n) + 1 + i) :
    frameAt st sch dom n p T j = encs ((chunks (ligFrames sch T p.cap p.ligerito)).1.getD i []) := by
  unfold frameAt
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    show j - (3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + (2 + n) + 1) = i by omega]

end frames

/-! ## The proof files -/

/-- The verifier's `final_c` at a rep (`ZcProofOf.fc`): the zerocheck's `vc` at the rows `verifierRows` gives at the
rep's coins, evaluated at the values w holds. -/
def fcOf (sh : Shape) (d : RepData) : F128 :=
  match verifierRows sh d.rounds d.extras d.comb with
  | .ok (_, info, _) => info.zc.vc.eval (valOf sh d.w d.comb d.cons)
  | .error _ => 0

/-- **A rep's proof file**: the messages w holds (`decoded`), the verifier's `final_c`, the PCS parameters `params`,
the cap, and the Ligerito openings `lig` with w's sumcheck messages, out-of-domain values and `yr`. -/
def proofOf (sh : Shape) (params : PcsParams) (cap : Array ByteArray) (lig : LigeritoProof) (d : RepData) : Proof :=
  { cap, params,
    zerocheck := ⟨(decoded sh d.w d.comb d.cons).ab, (decoded sh d.w d.comb d.cons).c, (decoded sh d.w d.comb d.cons).zc,
      (decoded sh d.w d.comb d.cons).fa, (decoded sh d.w d.comb d.cons).fb, fcOf sh d⟩,
    lincheck := ⟨(decoded sh d.w d.comb d.cons).lc, (decoded sh d.w d.comb d.cons).zp⟩,
    ringSwitches := (decoded sh d.w d.comb d.cons).rs,
    ligerito := { lig with
      initialCap := cap,
      messages := (decoded sh d.w d.comb d.cons).msgs.map fun mm => (RecAlgebra.kv mm.1, RecAlgebra.kv mm.2),
      oodValues := (decoded sh d.w d.comb d.cons).ood,
      finalProof := { lig.finalProof with yr := (decoded sh d.w d.comb d.cons).yr } } }

/-! ## The table files -/

/-- Rep `r`'s index. -/
def fin2 (r : ℕ) : Fin 2 := if r = 0 then 0 else 1

/-- Rep `r`'s stream's name, as `verifyRep` looks it up. -/
def streamName (spec : SessionSpec) (r : ℕ) : String := toString spec.tables[0]! ++ toString "/rep" ++ toString r

/-- The link points `P` of the verifier: arrays whose extra claims (§5) are well formed and realize the model's at
`pts`. -/
def PointsOK (st : Setup) (S : Model.Statement) (ptLocal mPts : ℕ) (pts : Fin 2 → Fin mPts → 𝔽)
    (P : Array (Array F128)) : Prop :=
  ∃ e, st.extra P = .ok e ∧ ExtraRealizes st S P (extraClaims S ptLocal mPts pts) ∧
    ∀ j < e.size, e[j]!.skip.size = 64 ∧ e[j]!.x.size = st.m - 6

/-- What the table files are made from: the session's spec, the setup and schedule, V*'s shape, the hidden statement,
the level-0 cap, each rep's Ligerito openings and each rep's values. -/
structure TableSrc where
  spec : SessionSpec
  st : Setup
  sch : Flock.Schedule
  sh : Shape
  S : Model.Statement
  cap : Array ByteArray
  lig : Fin 2 → LigeritoProof
  d : Fin 2 → RepData

namespace TableSrc

variable (s : TableSrc)

/-- The link points as the verifier's arrays. -/
noncomputable def points (ptLocal mPts : ℕ) (pts : Fin 2 → Fin mPts → 𝔽) : Array (Array F128) :=
  Classical.epsilon (PointsOK s.st s.S ptLocal mPts pts)

/-- The setup's extra claims at the link points. -/
def extrasAt (P : Array (Array F128)) : Array Claim :=
  match s.st.extra P with
  | .ok e => e
  | .error _ => #[]

/-- Rep `r`'s proof file. -/
def proofAt (r : Fin 2) : Proof := proofOf s.sh (pcsParams s.st.merkle s.sch) s.cap (s.lig r) (s.d r)

/-- Rep `r`'s proof file, as `TableFiles` indexes it. -/
def proof (r : ℕ) : Proof := s.proofAt (fin2 r)

/-- Rep `r`'s stream, for `n` extra claims. -/
def stream (n r : ℕ) : StreamRec :=
  { name := streamName s.spec r, parent := "", label := "", at_ := 0, rootSha := none,
    rounds := streamRounds (s.d (fin2 r)).rounds (frameAt s.st s.sch (s.spec.domain s.spec.tables[0]! r).toUTF8 n
      (s.proof r) (valOf s.sh (s.d (fin2 r)).w (s.d (fin2 r)).comb (s.d (fin2 r)).cons .T)) }

/-- The session record at the link points `P`, for `n` extra claims: the level-0 cap's bytes as the table's `root_B`, and
both reps' streams. -/
def sessAt (P : Array (Array F128)) (n : ℕ) : Session where
  rootF := ByteArray.empty
  rootB := #[s.cap.foldl (· ++ ·) ByteArray.empty]
  publics := #[]
  points := P
  y := #[]
  streams := #[s.stream n 0, s.stream n 1]
  proofSha := #[]

/-- **The table files at the link points `pts`**, their tapes empty. -/
noncomputable def filesAt (cls : HClass) (pts : Fin 2 → Fin cls.mPts → 𝔽) : TableFiles cls :=
  { spec := s.spec, st := s.st, sch := s.sch, pts, i := 0,
    extras := s.extrasAt (s.points cls.ptLocal cls.mPts pts),
    sess := s.sessAt (s.points cls.ptLocal cls.mPts pts) (s.extrasAt (s.points cls.ptLocal cls.mPts pts)).size,
    bytes := fun r => Classical.epsilon fun b =>
      decodeProof (shapeOf s.st s.sch (s.extrasAt (s.points cls.ptLocal cls.mPts pts)).size) b = .ok (s.proof r),
    proof := s.proof, stream := s.stream (s.extrasAt (s.points cls.ptLocal cls.mPts pts)).size,
    tape := fun _ => [] }

end TableSrc

section steps

variable {nT nR : ℕ} (T : Fin nT → Type) (CT : Fin nR → Type) {cls : HClass}

/-- **Both reps' tapes in the inner transcript `tr`** (`TableFiles.Valid`'s last two conjuncts): each a tape of its
rep's spine with its proof file's messages and its stream's coins, and `tr`'s steps the level-0 cap, the link points,
the tapes in `tableTape`'s order and the openings. -/
def Steps (b : TableFiles cls) (S : Model.Statement) (tr : List (List (SMsg T) × ((k : Fin nR) → CT k)))
    (ts : List Game.Step × List Game.Step) : Prop :=
  RepTape b.st S b.sch (extraClaims S cls.ptLocal cls.mPts b.pts).length (b.proof 0) (b.stream 0) ts.1 ∧
    RepTape b.st S b.sch (extraClaims S cls.ptLocal cls.mPts b.pts).length (b.proof 1) (b.stream 1) ts.2 ∧
    schedSteps T CT 0 tr = .inl ⟨Cap Dig, capDig (b.proof 0).cap⟩ :: .inr ⟨Fin 2 → Fin cls.mPts → 𝔽, b.pts⟩ ::
      tableTape ts.1 ts.2 (opensOf (b.proof 0) (b.proof 1) b.sch, saltsOf (b.proof 0) (b.proof 1) b.sch)

/-- **The link points of the inner transcript `tr`**: its first coin. -/
noncomputable def ptsOf (tr : List (List (SMsg T) × ((k : Fin nR) → CT k))) : Fin 2 → Fin cls.mPts → 𝔽 :=
  Classical.epsilon fun pts => ∃ a rest,
    schedSteps T CT 0 tr = a :: .inr ⟨Fin 2 → Fin cls.mPts → 𝔽, pts⟩ :: rest

theorem ptsOf_eq {tr : List (List (SMsg T) × ((k : Fin nR) → CT k))} {pts : Fin 2 → Fin cls.mPts → 𝔽}
    {a : Game.Step} {rest : List Game.Step}
    (h : schedSteps T CT 0 tr = a :: .inr ⟨Fin 2 → Fin cls.mPts → 𝔽, pts⟩ :: rest) : ptsOf T CT tr = pts := by
  obtain ⟨a', rest', h'⟩ := Classical.epsilon_spec (p := fun pts => ∃ a rest,
    schedSteps T CT 0 tr = a :: .inr ⟨Fin 2 → Fin cls.mPts → 𝔽, pts⟩ :: rest) ⟨pts, a, rest, h⟩
  have e := (List.cons.inj (List.cons.inj (h.symm.trans h')).2).1
  exact (eq_of_heq (Sigma.mk.inj (Sum.inr.inj e)).2).symm

/-- **The table files of the inner transcript `tr`**: at its link points, with the tapes `Steps` names. -/
noncomputable def TableSrc.files (s : TableSrc) (tr : List (List (SMsg T) × ((k : Fin nR) → CT k))) :
    TableFiles cls :=
  { s.filesAt cls (ptsOf T CT tr) with
    tape := fun r => if r = 0 then (Classical.epsilon (Steps T CT (s.filesAt cls (ptsOf T CT tr)) s.S tr)).1
      else (Classical.epsilon (Steps T CT (s.filesAt cls (ptsOf T CT tr)) s.S tr)).2 }

end steps

/-! ## What the files read -/

theorem pcs_beq_self (x : PcsParams) : (x == x) = true := by
  obtain ⟨a, b, c, d, e, f⟩ := x
  show (a == a && (b == b && (c == c && (d == d && (e == e && f == f))))) = true
  simp

theorem pcs_bne_self (x : PcsParams) : (x != x) = false := by rw [bne, pcs_beq_self]; rfl

theorem streamName_ne (spec : SessionSpec) : (streamName spec 0 == streamName spec 1) = false := by
  rw [beq_eq_false_iff_ne]
  intro h
  have h' := congrArg String.toList h
  simp only [streamName, String.toList_append, List.append_assoc] at h'
  have h'' := List.append_cancel_left h'
  revert h''
  decide

theorem proofOf_zc {sh : Shape} {params : PcsParams} {cap : Array ByteArray} {lig : LigeritoProof} {d : RepData}
    {v info kappa} (h : verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa)) :
    RecAlgebra.ZcProofOf (decoded sh d.w d.comb d.cons) (info.zc.vc.eval (valOf sh d.w d.comb d.cons))
      (proofOf sh params cap lig d).zerocheck :=
  ⟨rfl, rfl, rfl, rfl, rfl, by show fcOf sh d = _; unfold fcOf; rw [h]⟩

/-- **The inner statement's setup realizes the hidden statement `S`** (`TableFiles.Valid`'s first five conjuncts and
`run_of_frames`' dimensions), and at all link points `pts` some point arrays are well formed and realize `S`'s extra
claims at `pts` (§5). The verifier's setup of the inner circuit establishes it (`Setup.ofCircuit`, Verify.lean:29-32;
`Refine.Regions`' `extraRealizes` gives the points' half at `ptsOf`'s arrays). -/
def SetupRealizes (st : Setup) (S : Model.Statement) (sch : Flock.Schedule) (ptLocal mPts : ℕ) : Prop :=
  st.m = S.m ∧ st.kLog = S.kLog ∧ FoldRealizes st.fold S ∧
    st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" ∧
    Flock.fast100 st.m = some sch ∧ 14 ≤ st.m ∧ st.kLog ≤ st.m ∧ st.fold.kLog = st.kLog ∧
    ∀ pts, ∃ P, PointsOK st S ptLocal mPts pts P

/-- **`SetupRealizes`' scheme is the one the recursion's inner statements name**: the executable's plain leaf
(`Tags.withPlainLeaves`, `leafSchemePlain`), as `Setup.ofCircuit` finds it (`MerkleScheme.find?`). A change to the
executable's plain leaf fails this theorem rather than leaving G at another leaf. -/
theorem setupRealizes_scheme_of_tags (t : Tags) :
    MerkleScheme.find? (Tags.withPlainLeaves t).merkleLeaf = some (MerkleScheme.sha512 "flock-leaf/sha512-unsalted") :=
  rfl

/-- **V*'s shape is the setup's** (the verifier's gap `vsh`): its `m`, `k_log` and schedule. -/
def ShapeIs (sh : Shape) (st : Setup) (sch : Flock.Schedule) : Prop :=
  sh.m = st.m ∧ sh.kLog = st.kLog ∧ sh.sched = sch

/-- **A rep's coin counts** (`run_of_frames`' counts, of the rounds `verifyRep`'s program draws, Verify.lean:43-48):
the zerocheck's rounds draw 6,
`m - 13`, 1 and then one coin each; the lincheck's, one each; each of the `2 + n` ring switches 7, the batching
`2 + n`; and the `L` Ligerito rounds follow. -/
def CoinsShaped (m kLog n L : ℕ) (coins : Array (Array F128)) : Prop :=
  3 + (m - 6) + 2 + (kLog - 6) + 1 + (2 + n) + 1 + L ≤ coins.size ∧
    (coins[0]!).size = 6 ∧ (coins[1]!).size = m - 13 ∧ (coins[2]!).size = 1 ∧
    (∀ i < m - 6, (coins[3 + i]!).size = 1) ∧
    (∀ j < 2 + (kLog - 6) + 1, (coins[3 + (m - 6) + j]!).size = 1) ∧
    (∀ k < 2 + n, (coins[3 + (m - 6) + 2 + (kLog - 6) + 1 + k]!).size = 7) ∧
    (coins[3 + (m - 6) + 2 + (kLog - 6) + 1 + (2 + n)]!).size = 2 + n

/-- **The comb port is the lincheck's fold** at the verifier rows' `α`, `β`, weights, `ρ` and points (`hfoldPts`). -/
def CombFolds (st : Setup) (sh : Shape) (d : RepData) : Prop :=
  ∀ v info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa) →
    st.fold.folded info.lc.fold.alpha info.lc.fold.beta info.lc.fold.lw info.lc.fold.rhoIn info.lc.fold.ts = .ok d.comb

/-- **Every level's openings pass** (`hopen`; Ligerito.lean:148-158) in the proof file `p`. -/
def OpenedAll (st : Setup) (sch : Flock.Schedule) (sh : Shape) (p : Proof) (d : RepData) : Prop :=
  ∀ v info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa) →
    ∀ cs ∈ info.lg.conss.toList, RecAlgebra.Opened st.merkle sch p.cap p.ligerito (decoded sh d.w d.comb d.cons) cs

namespace TableSrc

variable (s : TableSrc)

/-- **Each rep's proof file has bytes** that decode to it at the shape `verifyRep` decodes with (Verify.lean:40-41). -/
def Encodes : Prop :=
  ∀ r, ∃ b, decodeProof (shapeOf s.st s.sch (2 * s.S.regions.length)) b = .ok (s.proof r)

/-- **Each rep's coin counts** (`CoinsShaped`), for the statement's `2 · |regions|` extra claims and the rep's Ligerito
rounds, and none of its rounds empty. -/
def Coins : Prop :=
  ∀ r : Fin 2, CoinsShaped s.st.m s.st.fold.kLog (2 * s.S.regions.length)
    (chunks (ligFrames s.sch (valOf s.sh (s.d r).w (s.d r).comb (s.d r).cons .T) (s.proofAt r).cap
      (s.proofAt r).ligerito)).1.length (s.d r).rounds ∧ Flock.RecAlgebra.NoEmpty (s.d r).rounds

theorem extrasAt_of {P : Array (Array F128)} {e : Array Claim} (h : s.st.extra P = .ok e) : s.extrasAt P = e := by
  unfold extrasAt; rw [h]

theorem points_ok {ptLocal mPts : ℕ} (hset : SetupRealizes s.st s.S s.sch ptLocal mPts)
    (pts : Fin 2 → Fin mPts → 𝔽) : PointsOK s.st s.S ptLocal mPts pts (s.points ptLocal mPts pts) :=
  Classical.epsilon_spec (hset.2.2.2.2.2.2.2.2 pts)

theorem stream_wf (n r : ℕ) : RoundsWF (s.stream n r).rounds := streamRounds_wf _ _

/-- **Both reps' runs of the table files** (`TableFiles.Runs`), by `run_of_frames`: the stream's rounds issue `F.rep`'s
coins and retain `frameAt`'s bytes. -/
theorem runs_filesAt {cls : HClass} (pts : Fin 2 → Fin cls.mPts → 𝔽)
    (hset : SetupRealizes s.st s.S s.sch cls.ptLocal cls.mPts) (hsh : ShapeIs s.sh s.st s.sch)
    (hpl : s.sh.holo = false) (halg : ∀ r, AlgebraHolds s.sh (s.d r) ∧ AccHolds s.sh (s.d r))
    (hxs : ∀ (r : Fin 2) P, PointsOK s.st s.S cls.ptLocal cls.mPts pts P → s.st.extra P = .ok (s.d r).extras)
    (henc : s.Encodes) (hcoins : s.Coins) (hcomb : ∀ r, CombFolds s.st s.sh (s.d r))
    (hopen : ∀ r, OpenedAll s.st s.sch s.sh (s.proofAt r) (s.d r)) :
    (s.filesAt cls pts).Runs := by
  obtain ⟨e, he, hext, hes⟩ := s.points_ok hset pts
  obtain ⟨-, -, -, -, hf, hm14, hkm, hkL, -⟩ := hset
  obtain ⟨hshm, hshk, hshs⟩ := hsh
  have hX : s.extrasAt (s.points cls.ptLocal cls.mPts pts) = e := s.extrasAt_of he
  have hsz : e.size = 2 * s.S.regions.length := by
    rw [(hext e he).1, FlockSoundness.extraClaims_length]
  intro r _
  show ∃ t₄, repProg s.st s.sch (s.proof r) (s.extrasAt (s.points cls.ptLocal cls.mPts pts))
      (Transcript.init (s.stream (s.extrasAt (s.points cls.ptLocal cls.mPts pts)).size r).rounds
        (s.spec.domain s.spec.tables[0]! r).toUTF8) = .ok ((), t₄) ∧
      t₄.k = (s.stream (s.extrasAt (s.points cls.ptLocal cls.mPts pts)).size r).rounds.size
  rw [hX]
  have hxe : (s.d (fin2 r)).extras = e := by
    have h := hxs (fin2 r) (s.points cls.ptLocal cls.mPts pts) ⟨e, he, hext, hes⟩
    rw [he] at h
    exact (Except.ok.inj h).symm
  obtain ⟨b, hb⟩ := henc r
  rw [← hsz] at hb
  have hco : CoinsShaped s.st.m s.st.fold.kLog (2 * s.S.regions.length)
      (chunks (ligFrames s.sch (valOf s.sh (s.d (fin2 r)).w (s.d (fin2 r)).comb (s.d (fin2 r)).cons .T)
        (s.proof r).cap (s.proof r).ligerito)).1.length (s.d (fin2 r)).rounds := (hcoins (fin2 r)).1
  obtain ⟨hcs, hc0, hc1, hc2, hcz, hcl, hcr, hcg⟩ := hco
  rw [← hsz] at hcs hcr hcg
  obtain ⟨t, htdef⟩ : ∃ t, t = Transcript.init (s.stream e.size r).rounds (s.spec.domain s.spec.tables[0]! r).toUTF8 :=
    ⟨_, rfl⟩
  rw [← htdef]
  have htk : t.k = 0 := by subst htdef; rfl
  have htr : t.rounds = streamRounds (s.d (fin2 r)).rounds (frameAt s.st s.sch (s.spec.domain s.spec.tables[0]! r).toUTF8
      e.size (s.proof r) (valOf s.sh (s.d (fin2 r)).w (s.d (fin2 r)).comb (s.d (fin2 r)).cons .T)) := by
    subst htdef; rfl
  have htp : t.pending = Transcript.header 1 0 (s.spec.domain s.spec.tables[0]! r).toUTF8.size ++
      Transcript.padded (s.spec.domain s.spec.tables[0]! r).toUTF8 := by
    subst htdef; rfl
  have hsize : t.rounds.size = (s.stream e.size r).rounds.size := by subst htdef; rfl
  have hR : ∀ j, j < (s.d (fin2 r)).rounds.size →
      Retains t.rounds[j]! (frameAt s.st s.sch (s.spec.domain s.spec.tables[0]! r).toUTF8 e.size (s.proof r)
        (valOf s.sh (s.d (fin2 r)).w (s.d (fin2 r)).comb (s.d (fin2 r)).cons .T) j) :=
    fun j hj => by rw [htr]; exact streamRounds_retains hj rfl
  have hN : ∀ j c, j < (s.d (fin2 r)).rounds.size → ((s.d (fin2 r)).rounds[j]!).size = c → (t.rounds[j]!).n = c :=
    fun j c hj hc => by rw [htr]; exact streamRounds_n hj hc
  have hcl' : ∀ j, 3 + (s.st.m - 6) ≤ j → j < 3 + (s.st.m - 6) + 2 + (s.st.fold.kLog - 6) + 1 →
      ((s.d (fin2 r)).rounds[j]!).size = 1 := fun j h₁ h₂ => by
    have h := hcl (j - (3 + (s.st.m - 6))) (by omega)
    rwa [show 3 + (s.st.m - 6) + (j - (3 + (s.st.m - 6))) = j by omega] at h
  have hOne : ∀ j L, 3 + (s.st.m - 6) ≤ j → j < 3 + (s.st.m - 6) + 2 + (s.st.fold.kLog - 6) + 1 →
      frameAt s.st s.sch (s.spec.domain s.spec.tables[0]! r).toUTF8 e.size (s.proof r)
        (valOf s.sh (s.d (fin2 r)).w (s.d (fin2 r)).comb (s.d (fin2 r)).cons .T) j = L →
      OneCoin t.rounds[j]! L := fun j L h₁ h₂ hL =>
    ⟨hN j 1 (by omega) (hcl' j h₁ h₂), answers_of_retains (hL ▸ hR j (by omega))⟩
  obtain ⟨t₄, hrun, hk⟩ := FlockVBridge.Runs.run_of_frames s.sh (s.d (fin2 r)) (halg (fin2 r)).1 (halg (fin2 r)).2
    s.st s.sch
    (s.proof r) b e t htk (by rw [htr, streamRounds_coins]) (hcoins (fin2 r)).2
    (by rw [htr]; exact streamRounds_wf _ _) hshm.symm hm14
    (hkL.trans hshk.symm) hshk.symm hkm hxe hes hshs hf hb (fun _ _ _ h => proofOf_zc h) ⟨rfl, rfl⟩ hpl
    rfl rfl rfl rfl (beq_self_eq_true _) (hcomb (fin2 r)) (hopen (fin2 r))
    (by rw [htk]; exact hN 0 6 (by omega) hc0)
    (by rw [htk, Nat.zero_add]; exact hN 1 _ (by omega) hc1)
    (by rw [htk, Nat.zero_add]; exact hN 2 _ (by omega) hc2)
    (fun i hi => by rw [htk, Nat.zero_add]; exact hN _ _ (by omega) (hcz i hi))
    (by
      rw [htk, htp]
      have h := hR 0 (by omega)
      rw [frameAt_zero] at h
      exact RecAlgebra.answersL_of_retains h)
    (by
      rw [htk, Nat.zero_add]
      have h := hR 1 (by omega)
      rw [frameAt_one] at h
      exact RecAlgebra.answersL_of_retains h)
    (by
      rw [htk, Nat.zero_add]
      have h := hR 2 (by omega)
      rw [frameAt_two] at h
      exact RecAlgebra.answersL_of_retains h)
    (fun i hi => by
      rw [htk, Nat.zero_add]
      have h := hR (3 + i) (by omega)
      rw [frameAt_zc _ _ _ _ _ _ rfl hi] at h
      exact RecAlgebra.answersL_of_retains h)
    (by
      rw [htk, Nat.zero_add]
      exact ⟨hOne (3 + (s.st.m - 6)) _ le_rfl (by omega) (frameAt_alpha _ _ _ _ _ _ rfl),
        hOne (3 + (s.st.m - 6) + 1) _ (by omega) (by omega) (frameAt_beta _ _ _ _ _ _ rfl),
        fun j hj => hOne (3 + (s.st.m - 6) + 2 + j) _ (by omega) (by omega) (frameAt_lc _ _ _ _ _ _ rfl hj),
        hOne (3 + (s.st.m - 6) + 2 + (s.st.fold.kLog - 6)) _ (by omega) (by omega)
          (frameAt_zp _ _ _ _ _ _ rfl)⟩)
    (fun k hk => by rw [htk, Nat.zero_add]; exact hN _ _ (by omega) (hcr k hk))
    (fun k hk => by
      rw [htk, Nat.zero_add]
      have h := hR (3 + (s.st.m - 6) + 2 + (s.st.fold.kLog - 6) + 1 + k) (by omega)
      rw [frameAt_rs _ _ _ _ _ _ rfl hk] at h
      exact answers_of_retains' h)
    (by rw [htk, Nat.zero_add]; exact hN _ _ (by omega) hcg)
    (by
      rw [htk, Nat.zero_add]
      have h := hR (3 + (s.st.m - 6) + 2 + (s.st.fold.kLog - 6) + 1 + (2 + e.size)) (by omega)
      rw [frameAt_gamma _ _ _ _ _ _ rfl] at h
      exact answers_of_retains' h)
    (fun i hi => by
      rw [htk, Nat.zero_add]
      have h := hR (3 + (s.st.m - 6) + 2 + (s.st.fold.kLog - 6) + 1 + (2 + e.size) + 1 + i) (by omega)
      rwa [frameAt_lig _ _ _ _ _ _ rfl] at h)
  exact ⟨t₄, hrun, hk.trans hsize⟩

end TableSrc

section valid

variable {nT nR : ℕ} (T : Fin nT → Type) (CT : Fin nR → Type) {cls : HClass}

/-- **The inner transcript `tr` is the table files' at some link points** (`Steps`): its first steps the level-0 cap and
the link points, then both reps' tapes, each of its spine's kinds with the proof file's messages and the stream's
coins (`F.rep`'s rounds), then the openings. -/
def TableSrc.TranscriptIs (s : TableSrc) (cls : HClass) (tr : List (List (SMsg T) × ((k : Fin nR) → CT k))) : Prop :=
  ∃ pts ts, Steps T CT (s.filesAt cls pts) s.S tr ts

/-- **Each rep's extra claims are the setup's at `tr`'s link points** (`verifyRep`'s `st.extra sess.points`,
Verify.lean:38): every well-formed point arrays realizing the statement's claims at `ptsOf tr` give the rep's. -/
def TableSrc.ExtrasAre (s : TableSrc) (cls : HClass) (tr : List (List (SMsg T) × ((k : Fin nR) → CT k))) : Prop :=
  ∀ (r : Fin 2) P, PointsOK s.st s.S cls.ptLocal cls.mPts (ptsOf T CT tr) P → s.st.extra P = .ok (s.d r).extras

theorem TableSrc.runs (s : TableSrc) (tr : List (List (SMsg T) × ((k : Fin nR) → CT k)))
    (hset : SetupRealizes s.st s.S s.sch cls.ptLocal cls.mPts) (hsh : ShapeIs s.sh s.st s.sch)
    (hpl : s.sh.holo = false)
    (halg : ∀ r, AlgebraHolds s.sh (s.d r) ∧ AccHolds s.sh (s.d r)) (hxs : s.ExtrasAre T CT cls tr)
    (henc : s.Encodes) (hcoins : s.Coins) (hcomb : ∀ r, CombFolds s.st s.sh (s.d r))
    (hopen : ∀ r, OpenedAll s.st s.sch s.sh (s.proofAt r) (s.d r)) :
    (s.files (cls := cls) T CT tr).Runs :=
  s.runs_filesAt (ptsOf T CT tr) hset hsh hpl halg hxs henc hcoins hcomb hopen

/-- **The table files are valid for the inner transcript `tr`** (`TableFiles.Valid`) once `tr` is theirs at some link
points (`TranscriptIs`). -/
theorem TableSrc.valid (s : TableSrc) (regs : List (Region cls.kLog (cls.m - cls.kLog))) (M : Hidden cls)
    (tr : List (List (SMsg T) × ((k : Fin nR) → CT k))) (hS : M.toStatement regs = s.S)
    (hset : SetupRealizes s.st s.S s.sch cls.ptLocal cls.mPts) (henc : s.Encodes) (htr : s.TranscriptIs T CT cls tr) :
    (s.files (cls := cls) T CT tr).Valid regs M T CT tr := by
  obtain ⟨pts, ts, hst⟩ := htr
  have hpts : ptsOf T CT tr = pts := ptsOf_eq T CT hst.2.2
  have hex : ∃ ts, Steps T CT (s.filesAt cls (ptsOf T CT tr)) s.S tr ts := by rw [hpts]; exact ⟨ts, hst⟩
  have hspec := Classical.epsilon_spec hex
  obtain ⟨e, he, hext, -⟩ := s.points_ok hset (ptsOf T CT tr)
  have hX : s.extrasAt (s.points cls.ptLocal cls.mPts (ptsOf T CT tr)) = e := s.extrasAt_of he
  have hsz : e.size = 2 * s.S.regions.length := by
    rw [(hext e he).1, FlockSoundness.extraClaims_length]
  obtain ⟨hmS, hkS, hfold, hms, hf, -⟩ := hset
  unfold TableFiles.Valid
  rw [hS]
  refine ⟨hmS, hkS, hfold, hms, hf, hext, ?_,
    by rw [show (s.files (cls := cls) T CT tr).extras = e from hX]; exact he, ?_, hspec.2.2⟩
  · intro x hx
    have hx' : x ∈ [s.stream _ 0, s.stream _ 1] := hx
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hx'
    rcases hx' with rfl | rfl <;> exact s.stream_wf _ _
  · intro r hr
    have hdec : decodeProof (shapeOf s.st s.sch (s.extrasAt (s.points cls.ptLocal cls.mPts (ptsOf T CT tr))).size)
        ((s.files (cls := cls) T CT tr).bytes r) = .ok (s.proof r) := by
      obtain ⟨b, hb⟩ := henc r
      have hb' : decodeProof (shapeOf s.st s.sch
          (s.extrasAt (s.points cls.ptLocal cls.mPts (ptsOf T CT tr))).size) b = .ok (s.proof r) := by
        rw [hX, hsz]; exact hb
      exact Classical.epsilon_spec (p := fun b => decodeProof (shapeOf s.st s.sch
        (s.extrasAt (s.points cls.ptLocal cls.mPts (ptsOf T CT tr))).size) b = .ok (s.proof r)) ⟨b, hb'⟩
    have hcap : ((s.proof r).cap.foldl (· ++ ·) ByteArray.empty ==
        #[s.cap.foldl (· ++ ·) ByteArray.empty][0]!) = true := by
      show (s.cap.foldl (· ++ ·) ByteArray.empty == #[s.cap.foldl (· ++ ·) ByteArray.empty][0]!) = true
      simp
    rcases (by omega : r = 0 ∨ r = 1) with rfl | rfl
    · refine ⟨?_, hdec, pcs_bne_self _, hcap, hspec.1⟩
      show Array.find? _ #[s.stream _ 0, s.stream _ 1] = _
      rw [← Array.find?_toList]
      exact List.find?_cons_of_pos (beq_self_eq_true _)
    · refine ⟨?_, hdec, pcs_bne_self _, hcap, hspec.2.1⟩
      show Array.find? _ #[s.stream _ 0, s.stream _ 1] = _
      rw [← Array.find?_toList]
      refine (List.find?_cons_of_neg ?_).trans (List.find?_cons_of_pos (beq_self_eq_true _))
      rw [Bool.not_eq_true]
      exact streamName_ne s.spec

end valid

/-! ## At the family -/

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂)

/-- What the family's table files are made from at a history and V*'s layers: the session's `spec`, the inner
statement's setup `st`, the level-0 cap `cap` and the Ligerito openings `lig` V*'s layers hold, and `F.rep`. -/
def tableSrc (spec : SessionSpec) (st : Reg → L.Ω → Setup)
    (cap : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Array ByteArray)
    (lig : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Fin 2 → LigeritoProof)
    (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) : TableSrc :=
  ⟨spec, st R ω, sch, F.sh, (F.x R ω).toStatement regs, cap l, lig l, F.rep R ω t l⟩

/-- **The family's table files**: `tableSrc`'s, at the inner transcript V*'s layers hold at the rounds' coins. -/
noncomputable def filesOf (spec : SessionSpec) (st : Reg → L.Ω → Setup)
    (cap : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Array ByteArray)
    (lig : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Fin 2 → LigeritoProof)
    (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) : TableFiles cls :=
  (F.tableSrc spec st cap lig R ω t l).files T CT ((F.wmsg l).zip (t.map Prod.snd))

end VStarFamily

end family

/-! ## The obligations -/

section obligations

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)
variable (spec : SessionSpec) (st : Reg → L.Ω → Setup)
    (cap : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Array ByteArray)
    (lig : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Fin 2 → LigeritoProof)

/-- `P` at every accepted outer proof's history and layers. -/
def AtAccepted (P : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Prop) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → P R ω t (layers F.zs dg N k Rw R ω t R₂ τ)

/-- **`hsch`** for a family whose files are `filesOf`'s. -/
theorem hSch_of (hF : F.files = F.filesOf spec st cap lig) : F.HSch :=
  fun R ω t l => (congrArg (fun f => (f R ω t l).sch) hF).trans rfl

/-- **`hfiles`** for a family whose files are `filesOf`'s: at every accepted outer proof, from the inner statement's
setup (`SetupRealizes`), the proof files' encodings (`Encodes`) and the inner transcript V*'s layers hold being the
files' steps at some link points (`Steps`). -/
theorem hFiles_of (hF : F.files = F.filesOf spec st cap lig)
    (hset : ∀ R ω, SetupRealizes (st R ω) ((F.x R ω).toStatement regs) sch cls.ptLocal cls.mPts)
    (henc : F.AtAccepted dg N k Rw fun R ω t l => (F.tableSrc spec st cap lig R ω t l).Encodes)
    (htr : F.AtAccepted dg N k Rw fun R ω t l =>
      (F.tableSrc spec st cap lig R ω t l).TranscriptIs T CT cls ((F.wmsg l).zip (t.map Prod.snd))) :
    F.HFiles dg N k Rw := by
  intro R ω t R₂ τ o hacc
  rw [hF]
  exact TableSrc.valid T CT _ regs (F.x R ω) _ rfl (hset R ω) (henc R ω t R₂ τ o hacc) (htr R ω t R₂ τ o hacc)

/-- **`hruns`** for a family whose files are `filesOf`'s: at every accepted outer proof whose reps meet E's
preconditions, `run_of_frames` at each rep, from the setup (`SetupRealizes`), V*'s shape (`ShapeIs`), the proof files'
encodings (`Encodes`), the reps' extra claims at the inner transcript's link points (`ExtrasAre`), their coin counts
(`Coins`), combs (`CombFolds`) and openings (`OpenedAll`). -/
theorem hRuns_of (hF : F.files = F.filesOf spec st cap lig)
    (hset : ∀ R ω, SetupRealizes (st R ω) ((F.x R ω).toStatement regs) sch cls.ptLocal cls.mPts)
    (hsh : ∀ R ω, ShapeIs F.sh (st R ω) sch)
    (henc : F.AtAccepted dg N k Rw fun R ω t l => (F.tableSrc spec st cap lig R ω t l).Encodes)
    (hxs : F.AtAccepted dg N k Rw fun R ω t l =>
      (F.tableSrc spec st cap lig R ω t l).ExtrasAre T CT cls ((F.wmsg l).zip (t.map Prod.snd)))
    (hcoins : F.AtAccepted dg N k Rw fun R ω t l => (F.tableSrc spec st cap lig R ω t l).Coins)
    (hcomb : F.AtAccepted dg N k Rw fun R ω t l => ∀ r, CombFolds (st R ω) F.sh (F.rep R ω t l r))
    (hopen : F.AtAccepted dg N k Rw fun R ω t l => ∀ r,
      OpenedAll (st R ω) sch F.sh ((F.tableSrc spec st cap lig R ω t l).proofAt r) (F.rep R ω t l r)) :
    F.HRuns dg N k Rw := by
  intro R ω t R₂ τ o hacc halg
  rw [hF]
  exact TableSrc.runs T CT (F.tableSrc spec st cap lig R ω t (layers F.zs dg N k Rw R ω t R₂ τ))
    ((F.wmsg (layers F.zs dg N k Rw R ω t R₂ τ)).zip (t.map Prod.snd)) (hset R ω) (hsh R ω)
    (FlockVBridge.OfVStar.shape?_holo F.hsh) halg (hxs R ω t R₂ τ o hacc) (henc R ω t R₂ τ o hacc)
    (hcoins R ω t R₂ τ o hacc) (hcomb R ω t R₂ τ o hacc) (hopen R ω t R₂ τ o hacc)

end VStarFamily

end obligations

end FlockVBridge.Family
