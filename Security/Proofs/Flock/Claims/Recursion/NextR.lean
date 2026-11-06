import Proofs.Flock.Claims.Recursion.NextS
import Proofs.Flock.Claims.Complete.Encode
import Proofs.Flock.Claims.Recursion.RepFramesBind
import Proofs.Flock.Claims.Recursion.VarLen

/-!
# `IVStar`'s last step at the deployed reader (`nextR`)

`nextS` reads each rep's whole proof from the last message, which `IVStar` doesn't open: no commit string binds it, so
any value the transcript absorbs would reach the verdict from bytes no part holds. `nextR` is `nextS` with the reader
fixed: every absorbed value of a decided proof is read from an opened round, the last message supplies only the
openings, and the fields the transcript never absorbs are what V\* takes for them (`params` the pinned ones,
`ligerito.initialCap` the cap, `zerocheck.finalC` V\*'s `v_C`).

* **How a round's rows join** (`joinR`, `rowsOf`, `rowsOf_joinR'`): tag `0`, then the rows, so each element-level
  message reads back as its opened rows.
* **Node rows** (`nodeRow`, `nodeOf`, `isNodeRow`): a cap node is opened as a 128-byte row, the node then 64 zero
  bytes, as the deployed verifier registers it (`rec_live.top_rows`, RecCap's `top`); the reader takes each round-0
  and tops row's node (`nodeOf`), and `rec-cap-zero` holds the padding.
* **The link** (`linkR`), from round 0's nodes, each table's level-0 top in turn: a table's `root_B` is its nodes
  joined, and its public words are empty, read from no message (`linkR_publics`, `linkR_s8`); every word round 0
  carries is a `root_B` node (`linkR_rootB`), which `rec-cap` holds.
* **A rep's proof** (`FitsRep`, `readRep`): rep `k`'s rounds are the schedule's (`RecDeploy.repSchedAt`); round `j`'s
  rows are its committed tops' node rows, then, where it absorbs, one row of the element values its squeeze answers (`vals`
  of the `j`-th chunk of `repFrames`, `T` among them), padded to whole 128-byte blocks by bytes no check reads (the
  deployed prover writes zeros, `padRow`; the frames' skeleton fixes how many values the row holds). The proof's cap is round 0's nodes for its table, its recursive
  caps the rounds' tops in order, its nonces `0`, and its fields of the lengths a proof decoded at the rep's shape has
  (`DecodeZL.ProofSizes`), so no field carries an entry no frame reads. `readRep` takes a proof and `T` that fit, with
  the last message's openings and the unabsorbed fields as V\* takes them (`absorbedFrom`, `repFrames_absorbedFrom`):
  what it reads of the last message changes no absorbed value (`readRep_last`). The nonces are absorbed
  (`.bytes (le64 n)`) but no opened row carries one, so the fit fixes them at `0`, the only nonce a deployed prover
  sends: every grinder ignores its bits (`LiveChallenger`'s `pow_nonce` is `0` outside flock-live's R4 test, and the
  firewall device's is `0`), and the recorded inner proofs carry only
  zeros. A proof with another nonce is decided by
  no fit. An honest prover's rounds read back (some fit exists, and it is the prover's proof), which `RecursiveKS` at
  `nextR` needs to be non-vacuous: `NextRHonest.lean` states it (`RepReadsBack`, `ReadsBack`) and `NextRFits.lean`
  proves it (`readsBack`).
* **The round messages custody and the verdict read** (`framesOf`): each round's framed bytes, the chunks of the
  decided proof's `repFrames`, with the round's coin.
* **What the reading refuses** (`decR_stmt`, `decR_shaped`): a last message whose statement is not the statement, as
  the checks after the verdict do (`checksAtJ`), and proofs not shaped for their files (`ShapedAt`), as decoding each
  file at its shape gives.
* **The record custody and the verdict read** (`recJ`, `recStreams`): the reader's, not the prover's: the honest
  record's form at SHA-512, whose rep streams retain the round messages, each round counting its schedule's coins
  (`RecDeploy.repSchedAt`) and holding its coin's first that many. This is what the shipped recursion enforces of the
  inner record: check 5 holds its coins to the coin server's round for round (`RecVStar.innerCommits`, `CoinTie`), and
  the coin server's rounds to the schedule (`RecVStar.checkSchedule`). At any hash but SHA-512 the record's digests
  disagree with the verdict's, which then refuses.
* **(a)** (`decR_fits`): at a decision, every rep's proof fits its opened rounds. **(b)** (`decR_publics`, `decR_s8`,
  `decR_rootB`): round 0 carries no public words; the link's are empty, and S8 passes on each.
* **The step, the split and the composition** (`actR`, `nextR`, `lastSplitR`, `RecDeploy.decompB_nextR`): as `nextS`,
  with the checks the decision needs (`VerdictR`): the setup at `hOf H`, the proofs' shapes, the rep streams' names, the
  verdict on the decoded proofs and the checks after it.
-/

namespace FlockDeploy

open Flock
open Lean (Json)
open FlockSoundness.Refine (Frame encs colBytes)
open FlockSoundness.Discharge.Exec (chunks repFrames zcFrames ligFrames levelFr introFr queryFr chFr)

noncomputable section

/-! ## Rows -/

/-- **How a round's rows join**: tag `0`, then the rows, every length in variable-length bytes (`encV`). -/
def joinR (rs : List Claims.Msg) : Claims.Msg := 0 :: encVL encVB (rs.map fun r => (⟨r.toArray⟩ : ByteArray))

theorem joinR_head (rs : List Claims.Msg) : ∃ a, joinR rs = 0 :: a := ⟨_, rfl⟩

/-- An element-level message read back as its rows. -/
def rowsOf : Claims.Msg → Option (List ByteArray)
  | 0 :: body =>
    match (rdVLs rdVB).run body with
    | some (v, []) => some v
    | _ => none
  | _ => none

/-- **A joined round reads back as its rows**, at every length. -/
theorem rowsOf_joinR' (rs : List Claims.Msg) :
    rowsOf (joinR rs) = some (rs.map fun r => (⟨r.toArray⟩ : ByteArray)) := by
  have h := rdVLs_encVL (r := rdVB) (e := encVB) (l := rs.map fun r => (⟨r.toArray⟩ : ByteArray))
    (fun b _ rest => rdVB_encVB b rest) []
  rw [List.append_nil] at h
  unfold rowsOf joinR
  simp only [h]

/-! ## Node rows -/

/-- **A cap node's row** (`rec_live.top_rows`): its bytes, then 64 zero bytes. -/
def nodeRow (node : ByteArray) : ByteArray := ⟨(node.data.toList ++ List.replicate 64 0).toArray⟩

/-- **The node a cap row holds**: the row less its last 64 bytes. A deployed cap row is 128 bytes (RecCap's `top`;
RecOpen's `out`, the climbed node then 512 zero bits, reads the same registration, `RecVStar.checkShared`), so this
is its first 64, the node. -/
def nodeOf (row : ByteArray) : ByteArray := ⟨(row.data.toList.take (row.data.size - 64)).toArray⟩

/-- **A row is a node's row**: 128 bytes, the last 64 zero. -/
def isNodeRow (row : ByteArray) : Bool :=
  row.data.size == 128 && row.data.toList.drop 64 == List.replicate 64 0

theorem nodeOf_nodeRow (node : ByteArray) : nodeOf (nodeRow node) = node := by
  show (⟨((node.data.toList ++ List.replicate 64 0).toArray.toList.take
    ((node.data.toList ++ List.replicate 64 0).toArray.size - 64)).toArray⟩ : ByteArray) = node
  rw [List.size_toArray, List.length_append, List.length_replicate, Nat.add_sub_cancel, List.toList_toArray,
    List.take_left, Array.toArray_toList]

theorem size_nodeRow (node : ByteArray) : (nodeRow node).size = node.size + 64 := by
  show (node.data.toList ++ List.replicate 64 0).toArray.size = node.data.size + 64
  rw [List.size_toArray, List.length_append, Array.length_toList, List.length_replicate]

theorem map_nodeOf_nodeRow (l : List ByteArray) : (l.map nodeRow).map nodeOf = l := by
  rw [List.map_map]
  exact (List.map_congr_left fun b _ => nodeOf_nodeRow b).trans (List.map_id l)

theorem nodeRow_nodeOf {row : ByteArray} (h : isNodeRow row = true) : nodeRow (nodeOf row) = row := by
  unfold isNodeRow at h
  simp only [Bool.and_eq_true, beq_iff_eq] at h
  show (⟨((row.data.toList.take (row.data.size - 64)).toArray.toList ++ List.replicate 64 0).toArray⟩ : ByteArray) =
    row
  rw [h.1, List.toList_toArray, ← h.2, show (128 - 64 : ℕ) = 64 from rfl, List.take_append_drop, Array.toArray_toList]

theorem isNodeRow_nodeRow {node : ByteArray} (h : node.data.size = 64) : isNodeRow (nodeRow node) = true := by
  have hl : node.data.toList.length = 64 := by rw [Array.length_toList]; exact h
  unfold isNodeRow nodeRow
  simp only [Bool.and_eq_true, beq_iff_eq]
  refine ⟨?_, ?_⟩
  · show (node.data.toList ++ List.replicate 64 0).toArray.size = 128
    rw [List.size_toArray, List.length_append, hl, List.length_replicate]
  · show (node.data.toList ++ List.replicate 64 0).toArray.toList.drop 64 = List.replicate 64 0
    rw [List.toList_toArray, ← hl, List.drop_left]

/-- `rowsOf_joinR'` at its earlier bounds. -/
theorem rowsOf_joinR {rs : List Claims.Msg} (_hl : rs.length < 2 ^ 64) (_hb : ∀ r ∈ rs, r.length < 2 ^ 64) :
    rowsOf (joinR rs) = some (rs.map fun r => (⟨r.toArray⟩ : ByteArray)) :=
  rowsOf_joinR' rs

/-- Consecutive blocks of the given lengths. -/
def cutLens {α : Type} : List ℕ → List α → List (List α)
  | [], _ => []
  | n :: ns, l => l.take n :: cutLens ns (l.drop n)

theorem cutLens_flatten {α : Type} : ∀ (ns : List ℕ) (l : List α), l.length ≤ ns.sum → (cutLens ns l).flatten = l
  | [], l, h => by
    cases l with
    | nil => rfl
    | cons a l => simp at h
  | n :: ns, l, h => by
    simp only [cutLens, List.flatten_cons]
    rw [cutLens_flatten ns (l.drop n) (by simp only [List.length_drop, List.sum_cons] at h ⊢; omega),
      List.take_append_drop]

/-! ## The link (b) -/

/-- Each table's level-0 top's node count, in table order: round 0's rows. -/
def level0Lens (sts : Array Stmt) : List ℕ := sts.toList.map RecDeploy.level0Nodes

/-- **The link from round 0's rows**: each table's `root_B` its level-0 top's nodes joined, and no public words. -/
def linkR (sts : Array Stmt) (r₀ : List ByteArray) : Option (List ByteArray × List ByteArray) :=
  if (level0Lens sts).sum = r₀.length then
    some ((cutLens (level0Lens sts) r₀).map (·.foldl (· ++ ·) ByteArray.empty),
      sts.toList.map fun _ => ByteArray.empty)
  else none

/-- **(b) Round 0 carries no public words**: the link's are each table's empty word, read from no message. -/
theorem linkR_publics {sts : Array Stmt} {r₀ : List ByteArray} {l : List ByteArray × List ByteArray}
    (h : linkR sts r₀ = some l) : l.2 = sts.toList.map fun _ => ByteArray.empty := by
  unfold linkR at h
  by_cases hc : (level0Lens sts).sum = r₀.length
  · rw [if_pos hc] at h
    cases h
    rfl
  · rw [if_neg hc] at h
    cases h

/-- **S8 passes on the link's public words**: each is empty, which a circuit statement's setup admits. -/
theorem linkR_s8 {sts : Array Stmt} {r₀ : List ByteArray} {l : List ByteArray × List ByteArray}
    (h : linkR sts r₀ = some l) (st : Stmt) : ∀ p ∈ l.2, (Setup.ofCircuit st).publics p = pure () := by
  intro p hp
  rw [linkR_publics h] at hp
  obtain ⟨_, _, rfl⟩ := List.mem_map.1 hp
  first | rfl | simp [Flock.Setup.ofCircuit] | decide

/-- **Every word round 0 carries is a `root_B` node**: the link's `root_B` are round 0's rows, table by table, each
table's joined, and no row is left over. -/
theorem linkR_rootB {sts : Array Stmt} {r₀ : List ByteArray} {l : List ByteArray × List ByteArray}
    (h : linkR sts r₀ = some l) :
    l.1 = (cutLens (level0Lens sts) r₀).map (·.foldl (· ++ ·) ByteArray.empty) ∧
      (cutLens (level0Lens sts) r₀).flatten = r₀ := by
  unfold linkR at h
  by_cases hc : (level0Lens sts).sum = r₀.length
  · rw [if_pos hc] at h
    cases h
    exact ⟨rfl, cutLens_flatten _ _ (by omega)⟩
  · rw [if_neg hc] at h
    cases h

/-! ## A rep's proof (a) -/

/-- The element values a round's frames absorb, in order. -/
def vals : List Frame → List F128
  | [] => []
  | .one v :: fs => v :: vals fs
  | .vec vs :: fs => vs ++ vals fs
  | _ :: fs => vals fs

/-- **A round's value row as the deployed prover writes it**: the values' bytes, then zero bytes up to whole 128-byte
blocks (`rec_live.message_rows`, `Firewall.messageRows`: 64 u16 words a block). -/
def padRow (l : List UInt8) : List UInt8 := l ++ List.replicate ((128 - l.length % 128) % 128) 0

theorem padRow_length_le (l : List UInt8) : (padRow l).length ≤ l.length + 127 := by
  simp only [padRow, List.length_append, List.length_replicate]
  omega

/-- **A padded row fixes its bytes at their length.** -/
theorem padRow_inj {l l' : List UInt8} (h : padRow l = padRow l') (hl : l.length = l'.length) : l = l' := by
  have e := congrArg (List.take l.length) h
  unfold padRow at e
  rwa [List.take_left' rfl, List.take_left' hl.symm] at e

/-- **Round `s`'s opened rows hold the values `vs` its squeeze answers**: its committed tops' nodes, then, where it
absorbs, one row of `vs` padded to whole blocks by any bytes (`rec_residuals`: a row port's words after its elements
are unread; the honest prover's zeros are `padRow`, `rowsAt_padRow`); a round that absorbs nothing answers no
value. -/
def RowsAt (s : RecVStar.SchedRound) (rs : List ByteArray) (vs : List F128) : Prop :=
  rs.length = s.2.1.toList.sum + (if s.2.2.2 then 1 else 0) ∧
    (∃ pad : List UInt8, pad.length = (128 - (colBytes vs).length % 128) % 128 ∧
      rs.drop s.2.1.toList.sum = (if s.2.2.2 then [(⟨(colBytes vs ++ pad).toArray⟩ : ByteArray)] else [])) ∧
    (s.2.2.2 = false → vs = [])

/-- **The honest prover's rows hold the values**: rows whose value row is `padRow`'s. -/
theorem rowsAt_padRow {s : RecVStar.SchedRound} {rs : List ByteArray} {vs : List F128}
    (h1 : rs.length = s.2.1.toList.sum + (if s.2.2.2 then 1 else 0))
    (h2 : rs.drop s.2.1.toList.sum = (if s.2.2.2 then [(⟨(padRow (colBytes vs)).toArray⟩ : ByteArray)] else []))
    (h3 : s.2.2.2 = false → vs = []) : RowsAt s rs vs :=
  ⟨h1, ⟨_, List.length_replicate .., h2⟩, h3⟩

/-- **A round's rows fix the values it absorbs**, at their count: the value row holds them first. -/
theorem RowsAt.vals_eq {s : RecVStar.SchedRound} {rs : List ByteArray} {vs vs' : List F128} (h : RowsAt s rs vs)
    (h' : RowsAt s rs vs') (hl : vs.length = vs'.length) : vs = vs' := by
  cases hb : s.2.2.2
  · rw [h.2.2 hb, h'.2.2 hb]
  · obtain ⟨pad, -, hd⟩ := h.2.1
    obtain ⟨pad', -, hd'⟩ := h'.2.1
    have e := hd.symm.trans hd'
    rw [ite_eq_left hb, ite_eq_left hb] at e
    have e' : colBytes vs ++ pad = colBytes vs' ++ pad' := by
      simpa using congrArg Array.toList (ByteArray.mk.inj (List.cons.inj e).1)
    exact FlockSoundness.Refine.colBytes_inj (List.append_inj e'
      (by rw [FlockSoundness.Refine.Frame.colBytes_length, FlockSoundness.Refine.Frame.colBytes_length, hl])).1

/-- The tops the rounds commit, in order: each round's first rows, cut at its tops' node counts, each row's node. -/
def topGroups (sched : List RecVStar.SchedRound) (rows : List (List ByteArray)) : List (List ByteArray) :=
  (sched.zip rows).flatMap fun q => (cutLens q.1.2.1.toList q.2).map (List.map nodeOf)

/-- What a rep's frames are read at: its table's setup and schedule, its stream's domain, its claim count, its rounds'
schedule, its table's level-0 top's nodes (round 0's rows for the table), and the zerocheck's point `z`. -/
structure RepCtx where
  st : Flock.Setup
  sch : Flock.Schedule
  dom : ByteArray
  n : ℕ
  sched : List RecVStar.SchedRound
  cap0 : List ByteArray
  z : F128

/-- Each rep's round count, rep `k` table `k / 2`'s schedule's, in `spec.repStreams`' order. -/
def repLens (spec : SessionSpec) (sts : Array Stmt) : List ℕ :=
  (List.range (2 * sts.size)).map fun k => (RecDeploy.repSchedAt spec sts[k / 2]!).length

/-- Rep `k`'s context: table `k / 2`'s, rep `k % 2`, at the link points of the first coin. `cs` are the coins of the
rounds after the link, rep by rep; the zerocheck's `z` is the first value of rep `k`'s round-2 coin, its third squeeze
(`bindAndZerocheck`: `sampleVec 6`, `sampleVec (m - 13)`, then `sample`). -/
def repCtx (spec : SessionSpec) (sts : Array Stmt) (c₁ : Coin) (r₀ : List ByteArray) (cs : List Coin) (k : ℕ) :
    RepCtx where
  st := (tabsOf sts)[k / 2]!
  sch := (fast100 (tabsOf sts)[k / 2]!.m).getD default
  dom := (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8
  n := 2 + (match (tabsOf sts)[k / 2]!.extra (pointsOf spec c₁) with
    | .ok e => e.size
    | .error _ => 0)
  sched := RecDeploy.repSchedAt spec sts[k / 2]!
  cap0 := (cutLens (level0Lens sts) r₀).getD (k / 2) []
  z := (((cutLens (repLens spec sts) cs).getD k []).getD 2 default).at 0

/-- The rep's rounds, as frames: the chunks of its proof's `repFrames`. -/
def chunksOf (c : RepCtx) (p : Flock.Proof) (T : F128) : List (List Frame) :=
  (chunks (repFrames c.st c.sch c.dom c.n p T)).1

/-- **A proof and `T` fit a rep's opened rounds**: one chunk per round, as the schedule has; the cap is its table's
level-0 top's nodes (round 0's); its recursive caps the rounds' tops, in order; every nonce `0`; the lengths a proof
decoded at the rep's shape has (`DecodeZL.ProofSizes`: no field carries an entry no frame reads, and two vectors
absorbed back to back can't split one row two ways), at a schedule of at least two levels; and round `j`'s rows hold
the values the `j`-th chunk absorbs. -/
structure FitsRep (c : RepCtx) (rows : List (List ByteArray)) (p : Flock.Proof) (T : F128) : Prop where
  rounds : (chunksOf c p T).length = rows.length
  sched : c.sched.length = rows.length
  cap : p.cap.toList = c.cap0
  tops : p.ligerito.recursiveCaps.toList.map Array.toList = topGroups c.sched rows
  nonces : (∀ n ∈ p.ligerito.queryNonces, n = 0) ∧ (∀ n ∈ p.ligerito.claimNonces, n = 0) ∧
    ∀ n ∈ p.ligerito.consistencyNonces, n = 0
  sizes : ∃ e, c.n = 2 + e ∧
    FlockSoundness.Discharge.DecodeZL.ProofSizes (FlockSoundness.Refine.shapeOf c.st c.sch e) p
  levels : 2 ≤ c.sch.levels.size
  elems : ∀ j < rows.length, RowsAt (c.sched.getD j default) (rows.getD j []) (vals ((chunksOf c p T).getD j []))

/-- **The absorbed fields of `a`, the openings of `q`, the rest as V\* takes them**: every field a rep's frames read
is `a`'s, and the openings are `q`'s. V\* reads no other field of an inner proof (`RecAlgebra`), and this reader follows
it: `params` is the pinned `pcsParams` of the table's scheme and schedule (V\*'s units are compiled at them),
`ligerito.initialCap` the cap (V\* absorbs only the cap), and `zerocheck.finalC` the value V\* takes for `v_C`,
`round1c` at the Lagrange weights of `z` (`RecAlgebra.zerocheck`'s `pc`). -/
def absorbedFrom (c : RepCtx) (a q : Flock.Proof) : Flock.Proof :=
  { q with
    cap := a.cap
    params := pcsParams c.st.merkle c.sch
    zerocheck := { q.zerocheck with
      round1ab := a.zerocheck.round1ab, round1c := a.zerocheck.round1c, rounds := a.zerocheck.rounds,
      finalA := a.zerocheck.finalA, finalB := a.zerocheck.finalB,
      finalC := dot (lagrange nodesL den6Inv c.z) a.zerocheck.round1c }
    lincheck := a.lincheck
    ringSwitches := a.ringSwitches
    ligerito := { q.ligerito with
      initialCap := a.cap,
      recursiveCaps := a.ligerito.recursiveCaps, messages := a.ligerito.messages,
      queryNonces := a.ligerito.queryNonces, oodValues := a.ligerito.oodValues,
      claimNonces := a.ligerito.claimNonces, consistencyNonces := a.ligerito.consistencyNonces,
      finalProof := { q.ligerito.finalProof with yr := a.ligerito.finalProof.yr } } }

theorem chFr_congr {p p' : Flock.LigeritoProof} (h : p.messages = p'.messages) :
    ∀ (msg k : ℕ), chFr p msg k = chFr p' msg k
  | _, 0 => rfl
  | msg, k + 1 => by simp only [chFr, h, chFr_congr h (msg + 1) k]

theorem ligFrames_congr (sch : Flock.Schedule) (T : F128) (cap : Array ByteArray) {p p' : Flock.LigeritoProof}
    (hm : p.messages = p'.messages) (ho : p.oodValues = p'.oodValues) (hc : p.claimNonces = p'.claimNonces)
    (hq : p.queryNonces = p'.queryNonces) (hs : p.consistencyNonces = p'.consistencyNonces)
    (hr : p.recursiveCaps = p'.recursiveCaps) (hy : p.finalProof.yr = p'.finalProof.yr) :
    ligFrames sch T cap p = ligFrames sch T cap p' := by
  have hch := chFr_congr hm
  have hlv : levelFr sch p = levelFr sch p' := funext fun i => by
    simp only [levelFr, introFr, queryFr, hch, hm, ho, hc, hq, hs, hr, hy]
  simp only [ligFrames, introFr, queryFr, hch, hlv, hm, ho, hc, hq, hs, hr]

theorem zcFrames_congr (m : ℕ) (d c : ByteArray) {z z' : Flock.ZerocheckProof} (h1 : z.round1ab = z'.round1ab)
    (h2 : z.round1c = z'.round1c) (h3 : z.rounds = z'.rounds) (h4 : z.finalA = z'.finalA) (h5 : z.finalB = z'.finalB) :
    zcFrames m d c z = zcFrames m d c z' := by
  simp only [zcFrames, h1, h2, h3, h4, h5]

theorem repFrames_congr (st : Flock.Setup) (sch : Flock.Schedule) (d : ByteArray) (n : ℕ) (T : F128) {p p' : Flock.Proof}
    (hc : p.cap = p'.cap)
    (hz : zcFrames st.m st.digest (p'.cap.foldl (· ++ ·) ByteArray.empty) p.zerocheck =
      zcFrames st.m st.digest (p'.cap.foldl (· ++ ·) ByteArray.empty) p'.zerocheck)
    (hl : p.lincheck = p'.lincheck) (hr : p.ringSwitches = p'.ringSwitches)
    (hg : ligFrames sch T p'.cap p.ligerito = ligFrames sch T p'.cap p'.ligerito) :
    repFrames st sch d n p T = repFrames st sch d n p' T := by
  unfold repFrames
  rw [hc, hz, hl, hr, hg]

/-- **A rep's frames read only the absorbed fields.** -/
theorem repFrames_absorbedFrom (st : Flock.Setup) (sch : Flock.Schedule) (d : ByteArray) (n : ℕ) (c : RepCtx)
    (a q : Flock.Proof) (T : F128) : repFrames st sch d n (absorbedFrom c a q) T = repFrames st sch d n a T :=
  repFrames_congr st sch d n T rfl (zcFrames_congr _ _ _ rfl rfl rfl rfl rfl) rfl rfl
    (ligFrames_congr _ _ _ rfl rfl rfl rfl rfl rfl rfl)

theorem fitsRep_absorbedFrom {c : RepCtx} {rows : List (List ByteArray)} {a : Flock.Proof} {T : F128}
    (h : FitsRep c rows a T) (q : Flock.Proof) : FitsRep c rows (absorbedFrom c a q) T := by
  have e : chunksOf c (absorbedFrom c a q) T = chunksOf c a T := by
    unfold chunksOf
    rw [repFrames_absorbedFrom]
  obtain ⟨x, hn, hs⟩ := h.sizes
  exact ⟨by rw [e]; exact h.rounds, h.sched, h.cap, h.tops, h.nonces,
    ⟨x, hn, hs.r1ab, hs.r1c, hs.zcRounds, hs.lcRounds, hs.zPartial, hs.rs, hs.rsEach, hs.cap, hs.capEach, hs.rcaps,
      hs.rcapsEach, hs.yr, hs.messages, hs.queryNonces, hs.oodValues, hs.claimNonces, hs.consistencyNonces⟩,
    h.levels, by rw [e]; exact h.elems⟩

/-- The rep's round messages: each round's framed bytes. -/
def framesOf (c : RepCtx) (p : Flock.Proof) (T : F128) : List Claims.Msg := (chunksOf c p T).map encs

open Classical in
/-- **A rep's proof read from its opened rounds**: a proof and `T` that fit them, with the last message's proof `q`
where no frame reads; and the rep's round messages, the chosen proof's framed bytes. None where nothing fits. -/
def readRep (c : RepCtx) (rows : List (List ByteArray)) (q : Flock.Proof) : Option (Flock.Proof × List Claims.Msg) :=
  if h : ∃ pT : Flock.Proof × F128, FitsRep c rows pT.1 pT.2 then
    some (absorbedFrom c (Classical.choose h).1 q,
      framesOf c (absorbedFrom c (Classical.choose h).1 q) (Classical.choose h).2)
  else none

/-- **(a) at a rep**: the proof `readRep` reads fits the rep's opened rounds, and its round messages are its framed
bytes. -/
theorem readRep_fits {c : RepCtx} {rows : List (List ByteArray)} {q p : Flock.Proof} {ms : List Claims.Msg}
    (h : readRep c rows q = some (p, ms)) : ∃ T, FitsRep c rows p T ∧ ms = framesOf c p T := by
  by_cases hx : ∃ pT : Flock.Proof × F128, FitsRep c rows pT.1 pT.2
  · rw [readRep, dif_pos hx] at h
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj h)
    exact ⟨_, fitsRep_absorbedFrom (Classical.choose_spec hx) q, rfl⟩
  · rw [readRep, dif_neg hx] at h
    cases h

/-- **What `readRep` reads of the last message changes no absorbed value**: read with another last-message proof `q'`,
the proof is the first's absorbed fields with `q'`'s others, and the round messages are the same. -/
theorem readRep_last {c : RepCtx} {rows : List (List ByteArray)} {q q' p p' : Flock.Proof} {ms ms' : List Claims.Msg}
    (h : readRep c rows q = some (p, ms)) (h' : readRep c rows q' = some (p', ms')) :
    p' = absorbedFrom c p q' ∧ ms' = ms := by
  by_cases hx : ∃ pT : Flock.Proof × F128, FitsRep c rows pT.1 pT.2
  · rw [readRep, dif_pos hx] at h h'
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj h)
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj h')
    refine ⟨rfl, ?_⟩
    simp only [framesOf, chunksOf, repFrames_absorbedFrom]
  · rw [readRep, dif_neg hx] at h
    cases h

/-! ## Every rep -/

/-- **Every rep's proof, read from its opened rounds**, rep `k` from the `k`-th block of the rounds after the link,
with the last message's `k`-th proof where no frame reads; and the round messages, rep by rep. -/
def readPs (spec : SessionSpec) (sts : Array Stmt) (c₁ : Coin) (r₀ : List ByteArray) (rss : List (List ByteArray))
    (cs : List Coin) (fl : FinalP) : Option (Array Flock.Proof × List Claims.Msg) :=
  if fl.ps.size = 2 * sts.size ∧ (repLens spec sts).sum = rss.length then
    ((List.range (2 * sts.size)).mapM fun k =>
      readRep (repCtx spec sts c₁ r₀ cs k) ((cutLens (repLens spec sts) rss).getD k []) fl.ps[k]!).map
        fun rs => ((rs.map Prod.fst).toArray, (rs.map Prod.snd).flatten)
  else none

theorem mapM_some_get {α β : Type} {f : α → Option β} : ∀ {l : List α} {r : List β}, l.mapM f = some r →
    r.length = l.length ∧ ∀ (i : ℕ) (hi : i < l.length) (hr : i < r.length), f l[i] = some r[i]
  | [], r, h => by
    rw [List.mapM_nil] at h
    cases h
    exact ⟨rfl, fun i hi => absurd hi (Nat.not_lt_zero _)⟩
  | a :: l, r, h => by
    simp only [List.mapM_cons, Option.bind_eq_bind, Option.pure_def] at h
    obtain ⟨b, hb, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨r', hr', h⟩ := Option.bind_eq_some_iff.1 h
    have e : r = b :: r' := (Option.some.inj h).symm
    subst e
    obtain ⟨hl, hg⟩ := mapM_some_get hr'
    refine ⟨by simp [hl], fun i hi hr => ?_⟩
    cases i with
    | zero => exact hb
    | succ i => exact hg i (by simpa using hi) (by simpa using hr)

/-- **(a) at every rep**: each proof `readPs` reads fits its rep's opened rounds. -/
theorem readPs_fits {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin} {r₀ : List ByteArray}
    {rss : List (List ByteArray)} {cs : List Coin} {fl : FinalP} {pf : Array Flock.Proof × List Claims.Msg}
    (h : readPs spec sts c₁ r₀ rss cs fl = some pf) :
    pf.1.size = 2 * sts.size ∧ ∀ (k : ℕ) (hk : k < pf.1.size), ∃ T,
      FitsRep (repCtx spec sts c₁ r₀ cs k) ((cutLens (repLens spec sts) rss).getD k []) pf.1[k] T := by
  unfold readPs at h
  by_cases hc : fl.ps.size = 2 * sts.size ∧ (repLens spec sts).sum = rss.length
  · rw [if_pos hc] at h
    obtain ⟨rs, hrs, rfl⟩ := Option.map_eq_some_iff.1 h
    obtain ⟨hlen, hget⟩ := mapM_some_get hrs
    rw [List.length_range] at hlen
    refine ⟨by simp [hlen], fun k hk => ?_⟩
    have hk' : k < rs.length := by simpa using hk
    have hg := hget k (by rw [List.length_range]; omega) hk'
    rw [List.getElem_range] at hg
    obtain ⟨T, hT, -⟩ := readRep_fits hg
    exact ⟨T, by simpa using hT⟩
  · rw [if_neg hc] at h
    cases h

/-! ## The decision -/

/-- What `nextR` decides on. -/
structure DecR where
  s : Ins
  spec : SessionSpec
  sts : Array Stmt
  link : List ByteArray × List ByteArray
  c₁ : Coin
  mids : List (Claims.Msg × Coin)
  f : FinalP

/-- **The rep streams' rounds as the reader fixes them**: rep `k`, in `spec.repStreams`' order, takes the `k`-th block
of the round messages `ms`; its round `j` counts its schedule's coins `n`, holds the first `n` values of its coin among
`cs`, and retains its message. -/
def recStreams (spec : SessionSpec) (sts : Array Stmt) (ms : List Claims.Msg) (cs : List Coin) :
    Array (Array Flock.Round) :=
  ((List.range (2 * sts.size)).map fun k =>
    (((cutLens (repLens spec sts) ms).getD k []).mapIdx fun j m =>
      let n := ((RecDeploy.repSchedAt spec sts[k / 2]!).getD j default).1
      (⟨n, ByteArray.empty, (cs.getD (((repLens spec sts).take k).sum + j) default).vals n,
        some ⟨m.toArray⟩⟩ : Flock.Round)).toArray).toArray

/-- **The record the reader hands over**: the honest record's form at SHA-512, at the link's `root_B`, the first
coin's link points, the reader's streams (`recStreams`) and the proofs' files. -/
def recJ (spec : SessionSpec) (sts : Array Stmt) (link : List ByteArray × List ByteArray) (c₁ : Coin)
    (ms : List Claims.Msg) (cs : List Coin) (ps : Array Flock.Proof) : Json :=
  FlockComplete.JsonRT.recordObj spec (Claims.hOf sha512) link.1.toArray (pointsOf spec c₁)
    (recStreams spec sts ms cs) (filesAt (tabsOf sts) (pointsOf spec c₁) ps)

/-- **The proofs are shaped for their files**: each at the shape its file is read at (`shapeAt`), as decoding a file
at its shape gives. -/
def ShapedAt (spec : SessionSpec) (sts : Array Stmt) (c₁ : Coin) (ps : Array Flock.Proof) : Prop :=
  ∀ (k : ℕ) (hk : k < ps.size), FlockVBridge.Family.ProofShaped (shapeAt (tabsOf sts) (pointsOf spec c₁) k) ps[k]

/-- **The reading at a decision**: at a statement that sets up, the last message's proof (`rd`, for what no frame
reads) where its statement is the statement, round 0's rows' nodes as the link (`linkR`), the rounds after it as the
reps' proofs (`readPs`) where they are shaped for their files (`ShapedAt`), the round messages their framed bytes,
each with its round's coin, and the record the reader's (`recJ`) at the statement. -/
def decR (rd : Claims.Msg → Option FinalP) (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin) : Option DecR := do
  let v ← setupOf false x
  let a ← as.getLast?
  let a₁ ← as.head?
  let c₁ ← cs.head?
  let fl ← (rd a).filter (·.stmt == x)
  let r₀ ← (rowsOf a₁).map (List.map nodeOf)
  let rss ← as.tail.dropLast.mapM rowsOf
  let link ← linkR v.2.2 r₀
  let pf ← (readPs v.2.1 v.2.2 c₁ r₀ rss cs.tail.dropLast fl).filter fun pf =>
    @decide (ShapedAt v.2.1 v.2.2 c₁ pf.1) (Classical.propDecidable _)
  pure ⟨v.1, v.2.1, v.2.2, link, c₁, pf.2.zip cs.tail.dropLast,
    ⟨x, recJ v.2.1 v.2.2 link c₁ pf.2 cs.tail.dropLast pf.1, pf.1⟩⟩

theorem decR_some {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {d : DecR} (h : decR rd x as cs = some d) :
    ∃ (v : Ins × SessionSpec × Array Stmt) (c₁ : Coin) (r₀ : List ByteArray) (rss : List (List ByteArray))
      (fl : FinalP) (link : List ByteArray × List ByteArray) (pf : Array Flock.Proof × List Claims.Msg),
      setupOf false x = some v ∧ cs.head? = some c₁ ∧ (rowsOf (as.headD [])).map (List.map nodeOf) = some r₀ ∧
      as.tail.dropLast.mapM rowsOf = some rss ∧ linkR v.2.2 r₀ = some link ∧
      readPs v.2.1 v.2.2 c₁ r₀ rss cs.tail.dropLast fl = some pf ∧
      d = ⟨v.1, v.2.1, v.2.2, link, c₁, pf.2.zip cs.tail.dropLast,
        ⟨x, recJ v.2.1 v.2.2 link c₁ pf.2 cs.tail.dropLast pf.1, pf.1⟩⟩ := by
  simp only [decR, Option.bind_eq_bind, Option.pure_def] at h
  obtain ⟨v, h1, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨a, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨a₁, h3, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨c₁, h4, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨fl, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨r₀, h6, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨rss, h7, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨link, h8, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨pf, h9, h⟩ := Option.bind_eq_some_iff.1 h
  have ha : as.headD [] = a₁ := by
    cases as with
    | nil => cases h3
    | cons b l => exact Option.some.inj h3
  exact ⟨v, c₁, r₀, rss, fl, link, pf, h1, h4, (by rw [ha]; exact h6), h7, h8,
    (Option.filter_eq_some_iff.1 h9).1, (Option.some.inj h).symm⟩

/-- **The reading fixes the last message's statement**: a decision's statement is the statement. -/
theorem decR_stmt {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {d : DecR} (h : decR rd x as cs = some d) : d.f.stmt = x := by
  obtain ⟨v, c₁, r₀, rss, fl, link, pf, -, -, -, -, -, -, rfl⟩ := decR_some h
  rfl

/-- **The reading refuses proofs not shaped for their files**: a decision's proofs are each shaped at the shape its
file is read at. -/
theorem decR_shaped {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {d : DecR} (h : decR rd x as cs = some d) : ShapedAt d.spec d.sts d.c₁ d.f.ps := by
  simp only [decR, Option.bind_eq_bind, Option.pure_def] at h
  obtain ⟨v, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨c₁, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨pf, h9, h⟩ := Option.bind_eq_some_iff.1 h
  obtain rfl := Option.some.inj h
  exact @of_decide_eq_true _ (Classical.propDecidable _) (Option.filter_eq_some_iff.1 h9).2

/-- **(a) Every absorbed value of a decided proof is read from an opened round**: at a decision, round 0's rows and
the later rounds' rows read back from the element-level messages, and rep `k`'s proof fits the `k`-th block of the
later rounds' rows, at its table's level-0 nodes among round 0's (`FitsRep`). -/
theorem decR_fits {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {d : DecR} (h : decR rd x as cs = some d) :
    ∃ (r₀ : List ByteArray) (rss : List (List ByteArray)),
      (rowsOf (as.headD [])).map (List.map nodeOf) = some r₀ ∧ as.tail.dropLast.mapM rowsOf = some rss ∧
      d.f.ps.size = 2 * d.sts.size ∧
      ∀ (k : ℕ) (hk : k < d.f.ps.size), ∃ T,
        FitsRep (repCtx d.spec d.sts d.c₁ r₀ cs.tail.dropLast k) ((cutLens (repLens d.spec d.sts) rss).getD k [])
          d.f.ps[k] T := by
  obtain ⟨v, c₁, r₀, rss, fl, link, pf, -, -, h6, h7, -, h9, rfl⟩ := decR_some h
  exact ⟨r₀, rss, h6, h7, readPs_fits h9⟩

/-- **(b) At a decision, round 0 carries no public words**: the link's are each table's empty word. -/
theorem decR_publics {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {d : DecR} (h : decR rd x as cs = some d) : d.link.2 = d.sts.toList.map fun _ => ByteArray.empty := by
  obtain ⟨v, c₁, r₀, rss, fl, link, pf, -, -, -, -, h8, -, rfl⟩ := decR_some h
  exact linkR_publics h8

/-- **(b) S8 passes at a decision**, on each of the link's public words, at every table. -/
theorem decR_s8 {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {d : DecR} (h : decR rd x as cs = some d) (st : Stmt) :
    ∀ p ∈ d.link.2, (Setup.ofCircuit st).publics p = pure () := by
  obtain ⟨v, c₁, r₀, rss, fl, link, pf, -, -, -, -, h8, -, rfl⟩ := decR_some h
  exact linkR_s8 h8 st

/-- **(b) Every word round 0 carries is a `root_B` node** (`rec-cap`'s): at a decision, the link's `root_B` are round
0's rows' nodes, each table's joined, with no node left over. -/
theorem decR_rootB {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {d : DecR} (h : decR rd x as cs = some d) :
    ∃ r₀, (rowsOf (as.headD [])).map (List.map nodeOf) = some r₀ ∧
      d.link.1 = (cutLens (level0Lens d.sts) r₀).map (·.foldl (· ++ ·) ByteArray.empty) ∧
      (cutLens (level0Lens d.sts) r₀).flatten = r₀ := by
  obtain ⟨v, c₁, r₀, rss, fl, link, pf, -, -, h6, -, h8, -, rfl⟩ := decR_some h
  exact ⟨r₀, h6, linkR_rootB h8⟩

/-! ## The step -/

/-- **`nextR`'s step**: a coin where the deployed plain verifier answers one; else decide where `decR` reads; else
refuse. -/
def actR (rd : Claims.Msg → Option FinalP) (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin) : ActS :=
  match act false x as cs with
  | .coin c => .coin c
  | _ =>
    match decR rd x as cs with
    | some d => .decide d.s d.spec d.sts d.link d.c₁ d.mids d.f
    | none => .rej

/-- **The element-level step at the deployed reader.** -/
def nextR (rd : Claims.Msg → Option FinalP) (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin) :
    Claims.OC Claims.Out :=
  match actR rd x as cs with
  | .rej => .ret .rej
  | .coin c => .ret (.msg c.bytes)
  | .decide s _ _ link c₁ mids f => outOf (decisionSF s x link c₁ mids f)

/-- **A coin is answered as the plain verifier answers it** (`deploySplit`'s `hcoin`). -/
theorem nextR_coin (rd : Claims.Msg → Option FinalP) (H : Claims.Msg → Claims.Msg) : ∀ (x : Claims.Msg)
    (as : List Claims.Msg) (cs : List Coin) (c : Coin), act false x as cs = .coin c →
    ((nextR rd x as cs).eval H).1 = .msg c.bytes := by
  intro x as cs c h
  simp only [nextR, actR, h, Claims.OC.eval]

/-- **A decision is `decR`'s.** -/
theorem actR_decide {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FinalP} (h : actR rd x as cs = .decide s spec sts link c₁ mids f) :
    decR rd x as cs = some ⟨s, spec, sts, link, c₁, mids, f⟩ := by
  unfold actR at h
  split at h
  · cases h
  · revert h
    cases hd : decR rd x as cs with
    | none => intro h; cases h
    | some d => intro h; cases h; rfl

theorem nextR_decide {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FinalP} (h : actR rd x as cs = .decide s spec sts link c₁ mids f) :
    nextR rd x as cs = outOf (decisionSF s x link c₁ mids f) := by
  simp only [nextR, h]

/-- **A decision of `nextR` accepts exactly where the setup at `hOf H` returns a session at which `decisionAtS`
returns.** -/
theorem nextR_acc_iff (H : Claims.Msg → Claims.Msg) {rd : Claims.Msg → Option FinalP} {x : Claims.Msg}
    {as : List Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : FinalP}
    (h : actR rd x as cs = .decide s spec sts link c₁ mids f) :
    ((nextR rd x as cs).eval H).1 = .acc ↔ ∃ spec' sts', setupWith (Claims.hOf H) s.read = .ok (spec', sts') ∧
      ((decisionAtS s spec' sts' x link c₁ mids f).run.eval H).1 = .ok () := by
  rw [nextR_decide h, outOf_acc_iff, decisionSF_eval]
  constructor
  · intro hd
    obtain ⟨⟨spec', sts'⟩, hs, hd⟩ := FlockSoundness.Refine.ebind_ok hd
    exact ⟨spec', sts', hs, hd⟩
  · rintro ⟨spec', sts', hs, hd⟩
    rw [hs]
    exact hd

/-- The record's value the verdict reads at a session: the coins substituted, the proof digests the files'. -/
def jAt (H : Claims.Msg → Claims.Msg) (spec : SessionSpec) (sts : Array Stmt) (c₁ : Coin)
    (mids : List (Claims.Msg × Coin)) (f : FinalP) : Json :=
  coinSubJ spec c₁ mids (shaSubJ spec ((filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).map
    (FlockComplete.dgOf spec (Claims.hOf H))) f.j)

/-- **The verdict on the decoded proofs and the checks after it make a decision of `nextR` accept**, at the session
the setup at `hOf H` returns, where each proof is shaped for its file and the rep streams' names are distinct. -/
theorem nextR_acc_of (H : Claims.Msg → Claims.Msg) {rd : Claims.Msg → Option FinalP} {x : Claims.Msg}
    {as : List Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : FinalP}
    (h : actR rd x as cs = .decide s spec sts link c₁ mids f) {spec' : SessionSpec} {sts' : Array Stmt}
    (hs : setupWith (Claims.hOf H) s.read = .ok (spec', sts'))
    (hsh : ∀ (k : ℕ) (hk : k < f.ps.size),
      FlockVBridge.Family.ProofShaped (shapeAt (tabsOf sts') (pointsOf spec' c₁) k) f.ps[k])
    (hnd : NamesDistinct spec')
    (hv : verifyJPWith (Flock.Hashes.ofSha512 (Claims.hOf H) : Flock.Hashes Flock.V) spec' (tabsOf sts')
      (jAt H spec' sts' c₁ mids f) f.ps = .ok ())
    (hc : checksAtJ s spec' x link c₁ mids ⟨f.stmt, f.j, #[]⟩ (jAt H spec' sts' c₁ mids f) = .ok ()) :
    ((nextR rd x as cs).eval H).1 = .acc := by
  refine (nextR_acc_iff H h).2 ⟨spec', sts', hs,
    (decisionAtS_eval H s spec' sts' x link c₁ mids (stmt := f.stmt) (j := f.j) (ps := f.ps) hsh hnd).trans ?_⟩
  unfold jAt at hv hc
  rw [hv]
  exact hc

/-- **The reading at the last step**: `decR` reads (`actR` decides), and the record value's fields custody reads are
there with their shapes. -/
def ReadOkR (rd : Claims.Msg → Option FinalP) (x : Claims.Msg) (ms : List Claims.Msg) (cs : List Coin) : Prop :=
  match actR rd x ms cs with
  | .decide _ spec _ _ c₁ mids f => ∃ lv, liveOfJ spec (coinSubJ spec c₁ mids f.j) = .ok lv
  | _ => False

/-- **The checks a decision of `nextR` needs**: the setup at `hOf H` returns; at its session, each proof is shaped for
its file, the rep streams' names are distinct, the verdict on the decoded proofs passes, and the checks after it. -/
inductive VerdictR where
  | setup
  | shaped
  | names
  | verdict
  | after
  deriving DecidableEq

def VerdictR.all : List VerdictR := [.setup, .shaped, .names, .verdict, .after]

theorem VerdictR.mem_all (q : VerdictR) : q ∈ VerdictR.all := by
  cases q <;> decide

/-- What check `q` asks of a decision. -/
def VerdictR.At (H : Claims.Msg → Claims.Msg) (q : VerdictR) (s : Ins) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (f : FinalP) : Prop :=
  match q with
  | .setup => ∃ v, setupWith (Claims.hOf H) s.read = .ok v
  | .shaped => ∀ spec' sts', setupWith (Claims.hOf H) s.read = .ok (spec', sts') → ∀ (k : ℕ) (hk : k < f.ps.size),
      FlockVBridge.Family.ProofShaped (shapeAt (tabsOf sts') (pointsOf spec' c₁) k) f.ps[k]
  | .names => ∀ spec' sts', setupWith (Claims.hOf H) s.read = .ok (spec', sts') → NamesDistinct spec'
  | .verdict => ∀ spec' sts', setupWith (Claims.hOf H) s.read = .ok (spec', sts') →
      verifyJPWith (Flock.Hashes.ofSha512 (Claims.hOf H) : Flock.Hashes Flock.V) spec' (tabsOf sts')
        (jAt H spec' sts' c₁ mids f) f.ps = .ok ()
  | .after => ∀ spec' sts', setupWith (Claims.hOf H) s.read = .ok (spec', sts') →
      checksAtJ s spec' x link c₁ mids ⟨f.stmt, f.j, #[]⟩ (jAt H spec' sts' c₁ mids f) = .ok ()

/-- Check `q` at the prover's messages: at a decision, what it asks of it; elsewhere nothing (the reading fails). -/
def holdsR (H : Claims.Msg → Claims.Msg) (rd : Claims.Msg → Option FinalP) (q : VerdictR) (x : Claims.Msg)
    (ms : List Claims.Msg) (cs : List Coin) : Prop :=
  match actR rd x ms cs with
  | .decide s _ _ link c₁ mids f => q.At H s x link c₁ mids f
  | _ => True

/-- **The reading and the five checks make the last step accept.** -/
theorem nextR_sound (H : Claims.Msg → Claims.Msg) (rd : Claims.Msg → Option FinalP) (x : Claims.Msg)
    (ms : List Claims.Msg) (cs : List Coin) (hr : ReadOkR rd x ms cs) (hc : ∀ q : VerdictR, holdsR H rd q x ms cs) :
    ((nextR rd x ms cs).eval H).1 = .acc := by
  unfold ReadOkR at hr
  unfold holdsR at hc
  revert hr hc
  cases h : actR rd x ms cs with
  | rej => intro hr; exact (hr : False).elim
  | coin c => intro hr; exact (hr : False).elim
  | decide s spec sts link c₁ mids f =>
    intro _ hc
    have hq : ∀ q : VerdictR, q.At H s x link c₁ mids f := hc
    obtain ⟨⟨spec', sts'⟩, hs⟩ := (hq .setup : ∃ v, setupWith (Claims.hOf H) s.read = .ok v)
    exact nextR_acc_of H h hs (hq .shaped spec' sts' hs) (hq .names spec' sts' hs) (hq .verdict spec' sts' hs)
      (hq .after spec' sts' hs)

end

end FlockDeploy

namespace RecDeploy

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin FinalP nextR nextR_coin ReadOkR VerdictR holdsR)
open Claims.Recursion.VStarB

noncomputable section

/-- **A split of `nextR`'s last step that may assume the reading**: checks the inner statement fixes, and the proof that
where the reading holds (`ReadOkR`) and every check does, the last step accepts. -/
structure LastSplitR (H : Msg → Msg) (rd : Msg → Option FinalP) (R : Type) where
  checks : Msg → List R
  holds : R → Msg → List Msg → List Coin → Prop
  sound : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), ms.length = (FlockDeploy.V false).k x →
    cs.length = ms.length → ReadOkR rd x ms cs → (∀ r ∈ checks x, holds r x ms cs) → ((nextR rd x ms cs).eval H).1 = .acc

/-- **The deployed split**: the five checks of `VerdictR`, at every statement. -/
def lastSplitR (H : Msg → Msg) (rd : Msg → Option FinalP) : LastSplitR H rd VerdictR where
  checks _ := VerdictR.all
  holds := holdsR H rd
  sound x ms cs _ _ hr hc := FlockDeploy.nextR_sound H rd x ms cs hr fun q => hc q (VerdictR.mem_all q)

/-- **The split with the reading as its own check.** -/
def LastSplitR.toLast {H : Msg → Msg} {rd : Msg → Option FinalP} {R : Type} (LS : LastSplitR H rd R) :
    LastSplit H (nextR rd) (ReadCheck R) where
  checks x := .read :: (LS.checks x).map .verdict
  holds q x ms cs := match q with
    | .read => ReadOkR rd x ms cs
    | .verdict r => LS.holds r x ms cs
  sound x ms cs hk hc h := LS.sound x ms cs hk hc (h .read (List.mem_cons_self ..)) fun r hr =>
    h (.verdict r) (List.mem_cons_of_mem _ (List.mem_map.2 ⟨r, hr, rfl⟩))

/-- `ownJ`'s owners are among the statement's parts wherever `ownR`'s are. -/
theorem ownR_lt {H : Msg → Msg} {rd : Msg → Option FinalP} {R : Type} (LS : LastSplitR H rd R)
    (ownR : Msg → R → Option ℕ) (D : Claims.Recursion.Deploy Coin) (L : LayoutB) (shape : Claims.Recursion.Form → Prop)
    (hR : ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), D.parse x = some st → shape st.form →
      ∀ r ∈ LS.checks (Claims.Recursion.xI D x), ∀ s, ownR x r = some s → s < D.nS x) :
    ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), D.parse x = some st → shape st.form →
      ∀ r ∈ LS.toLast.checks (Claims.Recursion.xI D x), ∀ s, ownJ D L ownR x r = some s → s < D.nS x := by
  intro x st hx hf r hr s ho
  cases r with
  | read => exact firstHolder_lt ho
  | verdict r =>
    refine hR x st hx hf r ?_ s ho
    rcases List.mem_cons.1 hr with h | h
    · cases h
    · obtain ⟨r', hr', h⟩ := List.mem_map.1 h
      cases h
      exact hr'

/-- **The composition at `nextR`**: `bdDeployed` with `nextR rd` as its steps, split by `deploySplit` at the reading's
split (`LastSplitR.toLast`), the reading given to the reader part of the last round (`ownJ`). Where the verdict's owners
are the statement's parts (`hR`), the checks no part holds hold off any witnesses (`hrest`), and each part's circuit
accepting implies its share (`hlow`), `DecompB` holds at what each part's circuit accepts. -/
theorem decompB_nextR (H : Msg → Msg) (rd : Msg → Option FinalP)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) {join : List Msg → Msg}
    (hj : ∀ rs, ∃ a, join rs = 0 :: a)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    {R : Type} (LS : LastSplitR H rd R) (ownR : Msg → R → Option ℕ)
    (rel' : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings) (L : LayoutB)
    (IV : Claims.Verifier Coin) (shape : Claims.Recursion.Form → Prop) {Col' T' : Type}
    (DO : Claims.Flock.Deploy Coin Col' T') (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg)
    (hR : ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (recBound rel' b L IV).parse x = some st → shape st.form →
      ∀ r ∈ LS.checks (Claims.Recursion.xI (recBound rel' b L IV) x), ∀ s, ownR x r = some s →
        s < (recBound rel' b L IV).nS x)
    (hrest : ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (recBound rel' b L IV).parse x = some st → shape st.form →
      ∀ q ∈ ivChecks (bdDeployed rel (nextR rd) colsE wE join rowIn final w)
          (deploySplit H rel (nextR rd) colsE wE join rowIn final w (nextR_coin rd H) LS.toLast)
          (Claims.Recursion.xI (recBound rel' b L IV) x),
        ownHolders (recBound rel' b L IV) L (ownDeploy (ownJ (recBound rel' b L IV) L ownR)) x q = none →
        (∀ r, q = .elem r → ∃ r', r = .last r') →
        ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit), ws.length = (recBound rel' b L IV).nS x →
          (∀ e ∈ L.allCalls x ws, H e.1 = e.2) →
          q.Holds H (bdDeployed rel (nextR rd) colsE wE join rowIn final w)
            (deploySplit H rel (nextR rd) colsE wE join rowIn final w (nextR_coin rd H) LS.toLast)
            (Claims.Recursion.xI (recBound rel' b L IV) x) c (L.msgs x ws))
    (hlow : ∀ s x c ms, PartAccepts H (recBound rel' b L IV) DO L shape shows cmt s x c ms →
      PartComputes H (recBound rel' b L IV) (bdDeployed rel (nextR rd) colsE wE join rowIn final w)
        (deploySplit H rel (nextR rd) colsE wE join rowIn final w (nextR_coin rd H) LS.toLast)
        (ownHolders (recBound rel' b L IV) L (ownDeploy (ownJ (recBound rel' b L IV) L ownR))) s x c ms) :
    DecompB H (recBound rel' b L IV) (bdDeployed rel (nextR rd) colsE wE join rowIn final w).IVStar L shape
      (PartAccepts H (recBound rel' b L IV) DO L shape shows cmt) :=
  decompB_partAccepts_of_parts H (recBound rel' b L IV) DO _ L shape shows cmt
    (deploySplit H rel (nextR rd) colsE wE join rowIn final w (nextR_coin rd H) LS.toLast) _ _ hlow
    (ownedBy_deploy H rel (nextR rd) colsE wE join rowIn final w (nextR_coin rd H) LS.toLast
      (ownJ (recBound rel' b L IV) L ownR) _ L shape (ownR_lt LS ownR _ L shape hR))
    (stmtFixes_deploy H rel (nextR rd) colsE wE hj rowIn final w (nextR_coin rd H) LS.toLast
      (ownJ (recBound rel' b L IV) L ownR) rel' b L IV shape hrest)

/-- **The composition at the deployed reader**: `decompB_nextR` at `lastSplitR`, its rows joined by `joinR`. -/
theorem decompB_lastSplitR (H : Msg → Msg) (rd : Msg → Option FinalP)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (ownR : Msg → VerdictR → Option ℕ)
    (rel' : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings) (L : LayoutB)
    (IV : Claims.Verifier Coin) (shape : Claims.Recursion.Form → Prop) {Col' T' : Type}
    (DO : Claims.Flock.Deploy Coin Col' T') (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg)
    (hR : ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (recBound rel' b L IV).parse x = some st → shape st.form →
      ∀ r ∈ VerdictR.all, ∀ s, ownR x r = some s → s < (recBound rel' b L IV).nS x)
    (hrest : ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (recBound rel' b L IV).parse x = some st → shape st.form →
      ∀ q ∈ ivChecks (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w)
          (deploySplit H rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w (nextR_coin rd H)
            (lastSplitR H rd).toLast)
          (Claims.Recursion.xI (recBound rel' b L IV) x),
        ownHolders (recBound rel' b L IV) L (ownDeploy (ownJ (recBound rel' b L IV) L ownR)) x q = none →
        (∀ r, q = .elem r → ∃ r', r = .last r') →
        ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit), ws.length = (recBound rel' b L IV).nS x →
          (∀ e ∈ L.allCalls x ws, H e.1 = e.2) →
          q.Holds H (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w)
            (deploySplit H rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w (nextR_coin rd H)
              (lastSplitR H rd).toLast)
            (Claims.Recursion.xI (recBound rel' b L IV) x) c (L.msgs x ws))
    (hlow : ∀ s x c ms, PartAccepts H (recBound rel' b L IV) DO L shape shows cmt s x c ms →
      PartComputes H (recBound rel' b L IV) (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w)
        (deploySplit H rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w (nextR_coin rd H)
          (lastSplitR H rd).toLast)
        (ownHolders (recBound rel' b L IV) L (ownDeploy (ownJ (recBound rel' b L IV) L ownR))) s x c ms) :
    DecompB H (recBound rel' b L IV) (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w).IVStar L
      shape (PartAccepts H (recBound rel' b L IV) DO L shape shows cmt) :=
  decompB_nextR H rd rel colsE wE FlockDeploy.joinR_head rowIn final w (lastSplitR H rd) ownR rel' b L IV shape DO
    shows cmt hR hrest hlow

end

end RecDeploy
