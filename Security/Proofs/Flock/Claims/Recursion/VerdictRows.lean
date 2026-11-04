import Proofs.Flock.Claims.Recursion.NextR
import Proofs.Flock.VBridge.Family.Runs

/-!
# The verdict's rows: what `hpieces` is assembled from

`hpieces` (`VerdictParts`) asks the verdict on the decoded proofs (`verifyJPWith`) to accept at a decision of the last
step, at SHA-512. Each table's reps run through `verifyRepDWith`, and a rep's run is VBridge's `runs_of_files` at
`RunsFrom`. This file fixes the objects every piece of that argument is stated at, and splits `RunsFrom` into rows, one
per owner:

* **The decision** (`DecAt`): `decR`'s reading, the setup at SHA-512 and the session the record decodes to.
* **A table's files** (`DecAt.files`): the verifier's setup, schedule and extra claims for table `i`, the session, each
  rep's decided proof, its file (`filesAt`) and its stream (S11's `find?`).
* **A rep's values** (`DecAt.rep`): V\*'s `RepData` at the decision. The shape is the one the proof file decodes at
  (`Refine.shapeOf`, dense); `w` is the values the rep's frames absorb, in order, at the `T` its opened rounds fit
  (`DecAt.T`), then the rep's four accumulators `acc`, which the layout supplies; the rounds are the stream's coins, the
  comb the lincheck's fold at them (`combOf`), and each level's consistency sums the opened rows' at the level's `α`
  (`consOf`, what `Opened` compares the messages with).
* **The rows** (`RowG`, `RowB`, `RowC`, `RowL`, `RowD`, `RowR`, `RowE`): `RunsFrom`'s fields, `AlgebraHolds`,
  `AccHolds` and the verdict's other per-rep checks, grouped by what proves them: the setup (G), custody's frames (B),
  the decoded proof's ties to the opened rounds (C), the fields the reader fixes (L), the Ligerito openings (D), the
  residuals (R) and the accumulators (E). `RunsFrom`'s `m`, `kLog`, `sched`, `extras` and `coins` hold by definition.
* **The assembly at a table** (`runsFrom_of_rows`, `runs_of_rows`): the rows make `RunsFrom`, and with R and E,
  `TableFiles.Runs`.
-/

namespace RecDeploy.Rows

open Flock
open FlockDeploy (Coin FinalP DecR decR jAt RepCtx repCtx repLens cutLens rowsOf chunksOf FitsRep vals tabsOf
  filesAt pointsOf setupWith)
open FlockVBridge.Stages (TableFiles)
open FlockVBridge.Compiled (RepData AlgebraHolds AccHolds)
open FlockVBridge.Family (RunsFrom FramedAt runs_of_files combOf)
open Flock.RecAlgebra (verifierRows decoded valOf NoEmpty ZcProofOf LcProofOf Opened consAt)
open FlockSoundness.Refine (RoundsWF openAt shapeOf)
open FlockSoundness.Discharge.Exec (repFrames)
open FlockSoundness.Discharge.Recursion (HClass)

noncomputable section

/-- **A decision of the last step at SHA-512**: `decR`'s reading `d` of the messages, the setup `(spec, sts)` the
statement's read gives at SHA-512, and the session `sess` the record value the verdict reads decodes to. -/
structure DecAt (rd : Claims.Msg → Option FinalP) (x : Claims.Msg) (ms : List Claims.Msg) (cs : List Coin) where
  d : DecR
  dec : decR rd x ms cs = some d
  spec : SessionSpec
  sts : Array Stmt
  setup : setupWith (Claims.hOf FlockDeploy.sha512) d.s.read = .ok (spec, sts)
  sess : RunRecord
  decode : Record.decodeJWith (Flock.Hashes.ofSha512 (Claims.hOf FlockDeploy.sha512) : Flock.Hashes Flock.V) spec
    (jAt FlockDeploy.sha512 spec sts d.c₁ d.mids d.f) = .ok sess

namespace DecAt

variable {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
  (D : DecAt rd x ms cs) (cls : HClass)

/-! ## A table's files -/

/-- **Table `i`'s files at the decision**: the verifier's setup and `fast100` schedule, the session, the statement's
extra claims at the session's points, rep `r`'s decided proof `ps[2 i + r]`, its file, and its stream as S11 finds
it. -/
def files (i : ℕ) : TableFiles cls where
  spec := D.spec
  st := (tabsOf D.sts)[i]!
  sess := D.sess
  sch := (fast100 (tabsOf D.sts)[i]!.m).getD default
  pts := fun _ _ => (0 : F128)
  i := i
  extras := ((tabsOf D.sts)[i]!.extra D.sess.points).toOption.getD #[]
  bytes r := (filesAt (tabsOf D.sts) (pointsOf D.spec D.d.c₁) D.d.f.ps)[2 * i + r]!
  proof r := D.d.f.ps[2 * i + r]!
  stream r := (D.sess.streams.find? fun s => s.name == toString D.spec.tables[i]! ++ toString "/rep" ++ toString r).getD
    default
  tape _ := []

/-- The shape table `i`'s proof files decode at, which V\*'s algebra reads them at: dense, with the statement's extra
claims. -/
def sh (i : ℕ) : Shape := shapeOf (D.files cls i).st (D.files cls i).sch (D.files cls i).extras.size

/-! ## A rep's values -/

/-- Round 0's rows' nodes, as the decision read them. -/
def r₀ (_ : DecAt rd x ms cs) : List ByteArray := ((rowsOf (ms.headD [])).map (List.map FlockDeploy.nodeOf)).getD []

/-- The later rounds' rows, as the decision read them. -/
def rss (_ : DecAt rd x ms cs) : List (List ByteArray) := (ms.tail.dropLast.mapM rowsOf).getD []

/-- Rep `k`'s context, as `decR` reads its proof (`decR_fits`). -/
def ctx (k : ℕ) : RepCtx := repCtx D.d.spec D.d.sts D.d.c₁ D.r₀ cs.tail.dropLast k

/-- Rep `k`'s opened rounds' rows. -/
def rows (k : ℕ) : List (List ByteArray) := (cutLens (repLens D.d.spec D.d.sts) D.rss).getD k []

open Classical in
/-- The batched value `T` at which rep `k`'s decided proof fits its opened rounds (`decR_fits`), `0` where none
does. -/
def T (k : ℕ) : F128 :=
  if h : ∃ T, FitsRep (D.ctx k) (D.rows k) D.d.f.ps[k]! T then Classical.choose h else 0

/-- Rep `k`'s `w`: the values its frames absorb at `T`, round by round, then its accumulators `acc`. -/
def w (k : ℕ) (acc : Array F128) : Array F128 :=
  ((chunksOf (D.ctx k) D.d.f.ps[k]! (D.T k)).flatMap vals).toArray ++ acc

end DecAt

/-- **Each level's consistency sums**: at level `level`, lane `l`, the sum the level's opened rows give at the `α` the
verifier rows draw for it, the value `Opened` compares the messages' with; `0` where the rows don't run. -/
def consOf (sh : Shape) (sch : Flock.Schedule) (p : Proof) (rounds : Array (Array F128)) (extras : Array Claim)
    (comb : Array F128) (level l : ℕ) : F128 :=
  match verifierRows sh rounds extras comb with
  | .ok (_, info, _) =>
    match info.lg.conss.find? (·.start == level) with
    | some c => consAt (openAt p.ligerito (sch.levels.size - 1) level).1 (sch.levels.getD level default).q c.alpha l
    | none => 0
  | .error _ => 0

namespace DecAt

variable {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
  (D : DecAt rd x ms cs) (cls : HClass)

/-- **Rep `r` of table `i`'s values in V\*'s layers**, with rep `k`'s accumulators `acc k`: `w` (`DecAt.w`), the
stream's coins, the comb at them, the opened rows' consistency sums and the statement's extra claims. -/
def rep (acc : ℕ → Array F128) (i : ℕ) (r : Fin 2) : RepData where
  w := D.w (2 * i + r) (acc (2 * i + r))
  comb := combOf (D.sh cls i) (D.files cls i) (FlockVBridge.Family.roundsOf (D.files cls i) r)
  cons := consOf (D.sh cls i) (D.files cls i).sch ((D.files cls i).proof r)
    (FlockVBridge.Family.roundsOf (D.files cls i) r) (D.files cls i).extras (combOf (D.sh cls i) (D.files cls i) (FlockVBridge.Family.roundsOf (D.files cls i) r))
  rounds := FlockVBridge.Family.roundsOf (D.files cls i) r
  extras := (D.files cls i).extras

/-! ## The rows -/

variable (acc : ℕ → Array F128) (i : ℕ)

/-- **Row G, the setup**: the verifier's setup is the reader's, table `i`'s Merkle scheme is the registry's plain
SHA-512 one, its dimensions are its fold's and its `fast100` schedule's, its extra claims are the session's and of
the claims' shapes, and the lincheck's fold at each rep's coins is the comb. -/
structure RowG : Prop where
  spec : D.spec = D.d.spec
  sts : D.sts = D.d.sts
  merkle : (D.files cls i).st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted"
  named : (D.files cls i).st.merkle.Named
  foldKLog : (D.files cls i).st.fold.kLog = (D.sh cls i).kLog
  kLog_le : (D.files cls i).st.kLog ≤ (D.files cls i).st.m
  fast : Flock.fast100 (D.files cls i).st.m = some (D.files cls i).sch
  extra : (D.files cls i).st.extra D.sess.points = .ok (D.files cls i).extras
  extras_wf : ∀ j < (D.files cls i).extras.size,
    (D.files cls i).extras[j]!.skip.size = 64 ∧ (D.files cls i).extras[j]!.x.size = (D.files cls i).st.m - 6
  foldPts : ∀ r : Fin 2, ∀ v info kappa,
    verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
      .ok (v, info, kappa) →
    (D.files cls i).st.fold.folded info.lc.fold.alpha info.lc.fold.beta info.lc.fold.lw info.lc.fold.rhoIn
      info.lc.fold.ts = .ok (D.rep cls acc i r).comb

variable (r : Fin 2)

/-- **Row B, custody's frames**: rep `r`'s stream is in the session (S11), its rounds are well formed and none of its
coin rounds is empty, and they hold the rep's frames at `w`'s batched value `T`. -/
structure RowB : Prop where
  finds : (D.files cls i).Finds r
  noEmpty : NoEmpty (D.rep cls acc i r).rounds
  roundsWF : RoundsWF ((D.files cls i).stream r).rounds
  framed : FramedAt ((D.files cls i).stream r).rounds 0 (repFrames (D.files cls i).st (D.files cls i).sch
    ((D.files cls i).spec.domain (D.files cls i).spec.tables[(D.files cls i).i]! r).toUTF8
    (2 + (D.files cls i).extras.size) ((D.files cls i).proof r)
    (valOf (D.sh cls i) (D.rep cls acc i r).w (D.rep cls acc i r).comb (D.rep cls acc i r).cons .T))

/-- **Row C, the decoded proof's ties**: rep `r`'s file decodes to its proof, and every field its frames absorb is
the one `w` holds (`decoded`), but the zerocheck's `finalC`. -/
structure RowC : Prop where
  decode : decodeProof (shapeOf (D.files cls i).st (D.files cls i).sch (D.files cls i).extras.size)
    ((D.files cls i).bytes r) = .ok ((D.files cls i).proof r)
  ab : ((D.files cls i).proof r).zerocheck.round1ab =
    (decoded (D.sh cls i) (D.rep cls acc i r).w (D.rep cls acc i r).comb (D.rep cls acc i r).cons).ab
  c : ((D.files cls i).proof r).zerocheck.round1c =
    (decoded (D.sh cls i) (D.rep cls acc i r).w (D.rep cls acc i r).comb (D.rep cls acc i r).cons).c
  rounds : ((D.files cls i).proof r).zerocheck.rounds =
    (decoded (D.sh cls i) (D.rep cls acc i r).w (D.rep cls acc i r).comb (D.rep cls acc i r).cons).zc
  fa : ((D.files cls i).proof r).zerocheck.finalA =
    (decoded (D.sh cls i) (D.rep cls acc i r).w (D.rep cls acc i r).comb (D.rep cls acc i r).cons).fa
  fb : ((D.files cls i).proof r).zerocheck.finalB =
    (decoded (D.sh cls i) (D.rep cls acc i r).w (D.rep cls acc i r).comb (D.rep cls acc i r).cons).fb
  lincheck : LcProofOf (decoded (D.sh cls i) (D.rep cls acc i r).w (D.rep cls acc i r).comb (D.rep cls acc i r).cons)
    ((D.files cls i).proof r).lincheck
  ringSwitches : ((D.files cls i).proof r).ringSwitches =
    (decoded (D.sh cls i) (D.rep cls acc i r).w (D.rep cls acc i r).comb (D.rep cls acc i r).cons).rs
  messages : ((D.files cls i).proof r).ligerito.messages =
    (decoded (D.sh cls i) (D.rep cls acc i r).w (D.rep cls acc i r).comb (D.rep cls acc i r).cons).msgs.map
      fun mm => (RecAlgebra.kv mm.1, RecAlgebra.kv mm.2)
  oodValues : ((D.files cls i).proof r).ligerito.oodValues =
    (decoded (D.sh cls i) (D.rep cls acc i r).w (D.rep cls acc i r).comb (D.rep cls acc i r).cons).ood
  yr : ((D.files cls i).proof r).ligerito.finalProof.yr =
    (decoded (D.sh cls i) (D.rep cls acc i r).w (D.rep cls acc i r).comb (D.rep cls acc i r).cons).yr

/-- **Row L, the fields the reader fixes**: rep `r`'s PCS parameters are the pinned ones, its Ligerito initial cap is
its cap, its zerocheck's `finalC` is V\*'s `v_C` at the verifier rows, and its cap is the committed `root_B`. -/
structure RowL : Prop where
  params : (((D.files cls i).proof r).params != pcsParams (D.files cls i).st.merkle (D.files cls i).sch) = false
  initialCap : (((D.files cls i).proof r).ligerito.initialCap == ((D.files cls i).proof r).cap) = true
  fc : ∀ v info kappa,
    verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
      .ok (v, info, kappa) →
    ((D.files cls i).proof r).zerocheck.finalC =
      info.zc.vc.eval (valOf (D.sh cls i) (D.rep cls acc i r).w (D.rep cls acc i r).comb (D.rep cls acc i r).cons)
  cap : (((D.files cls i).proof r).cap.foldl (· ++ ·) ByteArray.empty == D.sess.rootB[i]!) = true

/-- **Row D, the Ligerito openings**: at the verifier rows, every level's queries pass on rep `r`'s proof. -/
def RowD : Prop :=
  ∀ v info kappa,
    verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
      .ok (v, info, kappa) →
    ∀ c ∈ info.lg.conss.toList, Opened (D.files cls i).st.merkle (D.files cls i).sch ((D.files cls i).proof r).cap
      ((D.files cls i).proof r).ligerito
      (decoded (D.sh cls i) (D.rep cls acc i r).w (D.rep cls acc i r).comb (D.rep cls acc i r).cons) c

/-- **Row R, the residuals**: E's preconditions at rep `r` but the running sums. -/
def RowR : Prop := AlgebraHolds (D.sh cls i) (D.rep cls acc i r)

/-- **Row E, the accumulators**: E's running-sums precondition at rep `r`. -/
def RowE : Prop := AccHolds (D.sh cls i) (D.rep cls acc i r)

/-! ## The assembly at a table -/

/-- **The rows make `RunsFrom`** at table `i`'s files, its shape and its reps' values. -/
theorem runsFrom_of_rows (hG : D.RowG cls acc i) (hB : ∀ r, D.RowB cls acc i r) (hC : ∀ r, D.RowC cls acc i r)
    (hL : ∀ r, D.RowL cls acc i r) (hD : ∀ r, D.RowD cls acc i r) :
    RunsFrom (D.files cls i) (D.sh cls i) (D.rep cls acc i) where
  m := rfl
  foldKLog := hG.foldKLog
  kLog := rfl
  kLog_le := hG.kLog_le
  sched := rfl
  fast := hG.fast
  extras := fun _ => rfl
  extras_wf := hG.extras_wf
  coins := fun _ => rfl
  noEmpty := fun r => (hB r).noEmpty
  roundsWF := fun r => (hB r).roundsWF
  decode := fun r => (hC r).decode
  zerocheck := fun r v info kappa h =>
    ⟨(hC r).ab, (hC r).c, (hC r).rounds, (hC r).fa, (hC r).fb, (hL r).fc v info kappa h⟩
  lincheck := fun r => (hC r).lincheck
  ringSwitches := fun r => (hC r).ringSwitches
  messages := fun r => (hC r).messages
  oodValues := fun r => (hC r).oodValues
  yr := fun r => (hC r).yr
  initialCap := fun r => (hL r).initialCap
  foldPts := hG.foldPts
  opened := hD
  framed := fun r => (hB r).framed

/-- **The rows make table `i`'s reps run**: `runs_of_files` at `runsFrom_of_rows`, the shape dense, with R and E at
each rep. -/
theorem runs_of_rows (hG : D.RowG cls acc i) (hB : ∀ r, D.RowB cls acc i r) (hC : ∀ r, D.RowC cls acc i r)
    (hL : ∀ r, D.RowL cls acc i r) (hD : ∀ r, D.RowD cls acc i r) (hR : ∀ r, D.RowR cls acc i r)
    (hE : ∀ r, D.RowE cls acc i r) : (D.files cls i).Runs :=
  runs_of_files _ _ _ (D.runsFrom_of_rows cls acc i hG hB hC hL hD) rfl fun r => ⟨hR r, hE r⟩

end DecAt

end

end RecDeploy.Rows
