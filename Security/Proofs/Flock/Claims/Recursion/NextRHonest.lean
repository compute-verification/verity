import Proofs.Flock.Claims.Recursion.NextR
import Proofs.Flock.Claims.Recursion.RepFramesLig

/-!
# An honest prover's rounds read back at `nextR`

`nextR` reads each rep's proof from its opened rounds (`readRep`, `FitsRep`), so `RecursiveKS` at `nextR` is
non-vacuous only if an honest prover's rounds read back: on the messages the honest recursive prover sends, some proof
fits every rep, and the proof `readPs` reads is the prover's. This file defines what the honest prover sends and states
that.

* **What the honest prover sends** (`honestRows`, `repRows`, `capRows`, `honestMsgs`): round 0 carries each table's
  level-0 cap nodes, in table order, each as its 128-byte row (`nodeRow`, `rec_live.top_rows`). Round `j` of rep `k`,
  along the rep's schedule (`RecDeploy.repSchedAt`, `rec_live.rep_rounds`), carries the node rows of the tops the
  round commits, its proof's next recursive caps in order (`rec_live.top_rows`), then, where the round absorbs, one row of the values the `j`-th chunk of the rep's frames
  absorbs (`message_rows`, `vals`). Each round's rows are joined (`joinR`). The last message is any that the reader `rd`
  reads as the proofs.
* **The honest proofs** (`HonestRep`, `SchedOf`): each rep's proof has the lengths a proof decoded at the rep's shape
  has (`DecodeZL.ProofSizes`, which `ProofShaped` at that shape gives: `FlockComplete`'s `repShaped`), and every nonce
  is `0` (the deployed provers', D20). Both reps of a table carry its level-0 cap, as `sessionP` commits each table
  once. The rep's schedule is `RecVStar.repRounds` at its frames' `m`, `k_log` and claims.
* **Rows fix the fit** (`fitUnique`): two fits of one rep's rows have one cap, recursive caps and nonces, so frames of
  one skeleton (`skel`, `repFrames_skel`), which fixes each round's value count (`chunks_skel`, `vals_length_skel`), and
  in each round one value row (`RowsAt.vals_eq`), so one rep's frames
  (`eq_of_skel_vals`, `fits_frames`), which fix every absorbed field and `T` (`repFrames_fields`, `ligFrames_fields`).
* **The statements**: `RepFits` (the honest proof fits the honest rows) and `fitUnique` give `RepReadsBack`, that
  `readRep` reads the honest proof back (`repReadsBack_of`). That gives `ReadsBack`, that `decR` decides on the honest
  messages and reads exactly the honest proofs (`readsBack_of_rep`).

`RepFits`, which matches `RecVStar.repRounds` with `repFrames`' chunks round by round, is `NextRFits`' `repFits`.
-/

namespace FlockDeploy

open Flock
open FlockSoundness.Refine (Frame colBytes colBytes_inj)
open FlockSoundness.Discharge.DecodeZL (ProofSizes)
open FlockSoundness.Discharge.Exec (chunks chunks_flatten repFrames repFrames_chunks_snd repFrames_fields zcFrames
  lcFrames rsFrames ligFrames ligFrames_fields levelFr introFr queryFr chFr kFr rs_size)

/-! ## What the honest prover sends -/

/-- A row's bytes, as a message. -/
def rowMsg (b : ByteArray) : Claims.Msg := b.data.toList

/-- **The honest prover's rows at a rep's rounds**, along its schedule: each round's top nodes' rows (the next of
`caps`, in order, each `nodeRow`), then, where the round absorbs, one row of the values its frames `ch` absorb, padded
to whole blocks (`padRow`). -/
def honestRows : List RecVStar.SchedRound → List (List Frame) → List (Array ByteArray) → List (List ByteArray)
  | s :: ss, ch :: chs, caps =>
    (((caps.take s.2.1.size).flatMap Array.toList).map nodeRow ++
        (if s.2.2.2 then [(⟨(padRow (colBytes (vals ch))).toArray⟩ : ByteArray)] else [])) ::
      honestRows ss chs (caps.drop s.2.1.size)
  | _, _, _ => []

/-- **What the honest prover commits at a rep's rounds**, for its proof `p` at the batched value `T`. -/
def repRows (c : RepCtx) (p : Flock.Proof) (T : F128) : List (List ByteArray) :=
  honestRows c.sched (chunksOf c p T) p.ligerito.recursiveCaps.toList

/-- Round 0's rows at the honest prover: each table's level-0 cap nodes, in table order. -/
def capRows (caps : Array (Array ByteArray)) : List ByteArray := (caps.toList.map Array.toList).flatten

/-- **The honest prover's element-level messages**: round 0's rows (`capRows`, each `nodeRow`), then each rep's
rounds' rows, rep by rep, each round joined (`joinR`); then the last message. -/
def honestMsgs (caps : Array (Array ByteArray)) (reps : List (List (List ByteArray))) (last : Claims.Msg) :
    List Claims.Msg :=
  (joinR (((capRows caps).map nodeRow).map rowMsg) :: reps.flatten.map fun rs => joinR (rs.map rowMsg)) ++ [last]

/-- **An honest rep's proof**: the lengths a proof decoded at the rep's shape has (`DecodeZL.ProofSizes`, which
`ProofShaped` gives), at a schedule of at least two levels, every nonce `0`, and the fields the transcript never absorbs
as V\* takes them (`absorbedFrom`): the pinned `params`, the cap as `initialCap`, and `finalC` the value of `round1c`
at the Lagrange weights of the rep's `z`. -/
structure HonestRep (c : RepCtx) (p : Flock.Proof) : Prop where
  sizes : ∃ e, c.n = 2 + e ∧
    FlockSoundness.Discharge.DecodeZL.ProofSizes (FlockSoundness.Refine.shapeOf c.st c.sch e) p
  levels : 2 ≤ c.sch.levels.size
  nonces : (∀ n ∈ p.ligerito.queryNonces, n = 0) ∧ (∀ n ∈ p.ligerito.claimNonces, n = 0) ∧
    ∀ n ∈ p.ligerito.consistencyNonces, n = 0
  params : p.params = pcsParams c.st.merkle c.sch
  initialCap : p.ligerito.initialCap = p.cap
  finalC : p.zerocheck.finalC = dot (lagrange nodesL den6Inv c.z) p.zerocheck.round1c

/-- **A rep's schedule is its frames'**: `RecVStar.repRounds` at the frames' `m`, `k_log` and claim count, at the
frames' `fast100` schedule, of a setup that folds at its own `k_log`. -/
def SchedOf (c : RepCtx) : Prop :=
  fast100 c.st.m = some c.sch ∧ c.st.fold.kLog = c.st.kLog ∧
    ∃ rs, RecVStar.repRounds c.st.m c.st.fold.kLog c.n = some rs ∧ c.sched = rs.toList

/-- Rows one message can join (`rowsOf_joinR`): fewer than `2^64`, each shorter than `2^64` bytes. -/
def RowsFit (rs : List ByteArray) : Prop := rs.length < 2 ^ 64 ∧ ∀ r ∈ rs, r.size < 2 ^ 64

/-! ## The statements -/

/-- **The honest proof fits the honest rows**, at any `T`. -/
def RepFits : Prop :=
  ∀ (c : RepCtx) (p : Flock.Proof) (T : F128), SchedOf c → HonestRep c p → p.cap.toList = c.cap0 →
    FitsRep c (repRows c p T) p T

/-- **Rows fix a fitting proof's absorbed fields and its `T`**, at a setup that folds at its own `k_log`. -/
def FitUnique : Prop :=
  ∀ (c : RepCtx) (rows : List (List ByteArray)) (p p' : Flock.Proof) (T T' : F128), c.st.fold.kLog = c.st.kLog →
    FitsRep c rows p T → FitsRep c rows p' T' → ∀ q, absorbedFrom c p' q = absorbedFrom c p q ∧ T' = T

/-- **One honest rep reads back**: from the honest rows, with the honest proof as the last message's, `readRep` reads
the honest proof and its round messages. -/
def RepReadsBack : Prop :=
  ∀ (c : RepCtx) (p : Flock.Proof) (T : F128), SchedOf c → HonestRep c p → p.cap.toList = c.cap0 →
    readRep c (repRows c p T) p = some (p, framesOf c p T)

/-- **An honest prover's rounds read back.** Take a statement that sets up as `(s, spec, sts)` and coins whose first is
`c₁`. The honest prover sends round 0 from each table's level-0 cap `caps[i]` (of its `2^cap₀` nodes), and each rep's
rounds from its proof `ps[k]` (honest, at its table's cap) at its batched value `Ts k`, with a last message the reader
reads as `fl`, whose statement is the statement and whose proofs are `ps`, each shaped for its file (`ShapedAt`). On
those messages `decR` decides. It reads round 0 as the link of the caps, each table's `root_B` its cap's nodes joined,
with no public words; every rep's proof as `ps[k]`; the round messages as the honest proofs' framed bytes; and the
record as the reader's (`recJ`) at the statement. -/
def ReadsBack : Prop :=
  ∀ (rd : Claims.Msg → Option FinalP) (x last : Claims.Msg) (cs : List Coin) (s : Ins) (spec : SessionSpec)
    (sts : Array Stmt) (c₁ : Coin) (caps : Array (Array ByteArray)) (ps : Array Flock.Proof) (Ts : ℕ → F128)
    (fl : FinalP),
    setupOf false x = some (s, spec, sts) → cs.head? = some c₁ →
    caps.size = sts.size → (∀ i < caps.size, caps[i]!.size = RecDeploy.level0Nodes sts[i]!) →
    ps.size = 2 * sts.size →
    (∀ k < ps.size, SchedOf (repCtx spec sts c₁ (capRows caps) cs.tail.dropLast k) ∧
      HonestRep (repCtx spec sts c₁ (capRows caps) cs.tail.dropLast k) ps[k]! ∧ ps[k]!.cap = caps[k / 2]!) →
    RowsFit ((capRows caps).map nodeRow) →
    (∀ k < ps.size, ∀ rs ∈ repRows (repCtx spec sts c₁ (capRows caps) cs.tail.dropLast k) ps[k]! (Ts k), RowsFit rs) →
    rd last = some fl → fl.ps = ps → fl.stmt = x → ShapedAt spec sts c₁ ps →
    decR rd x (honestMsgs caps ((List.range ps.size).map fun k =>
        repRows (repCtx spec sts c₁ (capRows caps) cs.tail.dropLast k) ps[k]! (Ts k)) last) cs =
      some ⟨s, spec, sts,
        ((caps.toList.map Array.toList).map (·.foldl (· ++ ·) ByteArray.empty),
          sts.toList.map fun _ => ByteArray.empty),
        c₁, ((List.range ps.size).map fun k =>
          framesOf (repCtx spec sts c₁ (capRows caps) cs.tail.dropLast k) ps[k]! (Ts k)).flatten.zip
          cs.tail.dropLast,
        ⟨x, recJ spec sts
          ((caps.toList.map Array.toList).map (·.foldl (· ++ ·) ByteArray.empty),
            sts.toList.map fun _ => ByteArray.empty) c₁
          ((List.range ps.size).map fun k =>
            framesOf (repCtx spec sts c₁ (capRows caps) cs.tail.dropLast k) ps[k]! (Ts k)).flatten
          cs.tail.dropLast ps, ps⟩⟩

/-! ## Rows fix the fit -/

/-- A frame with its element values blanked. -/
def skel : Frame → Frame
  | .dom b => .dom b
  | .label b => .label b
  | .one _ => .one default
  | .vec vs => .vec (List.replicate vs.length default)
  | .bytes b => .bytes b
  | .sq k n => .sq k n

theorem skel_dom (b : ByteArray) : skel (.dom b) = .dom b := rfl
theorem skel_label (b : ByteArray) : skel (.label b) = .label b := rfl
theorem skel_one (v : F128) : skel (.one v) = .one default := rfl
theorem skel_vec (vs : List F128) : skel (.vec vs) = .vec (List.replicate vs.length default) := rfl
theorem skel_bytes (b : ByteArray) : skel (.bytes b) = .bytes b := rfl
theorem skel_sq (k : UInt8) (n : ℕ) : skel (.sq k n) = .sq k n := rfl

theorem vals_append : ∀ (a b : List Frame), vals (a ++ b) = vals a ++ vals b
  | [], _ => rfl
  | .dom _ :: a, b => vals_append a b
  | .label _ :: a, b => vals_append a b
  | .one v :: a, b => congrArg (v :: ·) (vals_append a b)
  | .vec vs :: a, b => by
    show vs ++ vals (a ++ b) = vs ++ vals a ++ vals b
    rw [vals_append a b, List.append_assoc]
  | .bytes _ :: a, b => vals_append a b
  | .sq _ _ :: a, b => vals_append a b

theorem vals_flatten : ∀ L : List (List Frame), vals L.flatten = (L.map vals).flatten
  | [] => rfl
  | l :: L => by rw [List.flatten_cons, vals_append, vals_flatten L, List.map_cons, List.flatten_cons]

/-- **Frames are their skeleton and their values.** -/
theorem eq_of_skel_vals : ∀ {fs fs' : List Frame}, fs.map skel = fs'.map skel → vals fs = vals fs' → fs = fs'
  | [], [], _, _ => rfl
  | [], _ :: _, h, _ => by simp at h
  | _ :: _, [], h, _ => by simp at h
  | f :: fs, f' :: fs', h, hv => by
    rw [List.map_cons, List.map_cons, List.cons.injEq] at h
    obtain ⟨h1, h2⟩ := h
    cases f <;> cases f' <;> try (simp only [skel, reduceCtorEq] at h1; done)
    · rw [Frame.dom.inj h1, eq_of_skel_vals h2 hv]
    · rw [Frame.label.inj h1, eq_of_skel_vals h2 hv]
    · rename_i v v'
      obtain ⟨e, hv⟩ := List.cons.inj (show v :: vals fs = v' :: vals fs' from hv)
      rw [e, eq_of_skel_vals h2 hv]
    · rename_i vs vs'
      have hl : vs.length = vs'.length := by simpa using congrArg List.length (Frame.vec.inj h1)
      obtain ⟨e, hv⟩ := List.append_inj (show vs ++ vals fs = vs' ++ vals fs' from hv) hl
      rw [e, eq_of_skel_vals h2 hv]
    · rw [Frame.bytes.inj h1, eq_of_skel_vals h2 hv]
    · obtain ⟨rfl, rfl⟩ := Frame.sq.inj h1
      rw [eq_of_skel_vals h2 hv]

theorem closes_skel (f : Frame) : FlockSoundness.Discharge.Exec.closes (skel f) =
    FlockSoundness.Discharge.Exec.closes f := by
  cases f <;> rfl

/-- **A frame list's rounds, skeleton by skeleton**: the skeleton's rounds are the rounds' skeletons. -/
theorem chunks_skel : ∀ fs : List Frame, (chunks (fs.map skel)).1 = (chunks fs).1.map (List.map skel)
  | [] => rfl
  | f :: fs => by
    have ih := chunks_skel fs
    rw [List.map_cons]
    simp only [chunks, closes_skel, ih]
    cases FlockSoundness.Discharge.Exec.closes f <;> rcases (chunks fs).1 with _ | ⟨c, cs⟩ <;> simp

/-- **The skeleton keeps the values' count.** -/
theorem vals_length_skel : ∀ fs : List Frame, (vals (fs.map skel)).length = (vals fs).length
  | [] => rfl
  | .dom _ :: fs => vals_length_skel fs
  | .label _ :: fs => vals_length_skel fs
  | .one v :: fs => by
    show (default :: vals (fs.map skel)).length = (v :: vals fs).length
    simp only [List.length_cons, vals_length_skel fs]
  | .vec vs :: fs => by
    show (List.replicate vs.length default ++ vals (fs.map skel)).length = (vs ++ vals fs).length
    rw [List.length_append, List.length_append, List.length_replicate, vals_length_skel fs]
  | .bytes _ :: fs => vals_length_skel fs
  | .sq _ _ :: fs => vals_length_skel fs

theorem getD_map_skel : ∀ (L : List (List Frame)) (j : ℕ),
    (L.getD j []).map skel = (L.map (List.map skel)).getD j []
  | [], _ => by simp
  | _ :: _, 0 => by simp
  | _ :: L, j + 1 => by simpa using getD_map_skel L j

/-- **Two fits of one rep's rows with one skeleton have one rep's frames.** -/
theorem fits_frames {c : RepCtx} {rows : List (List ByteArray)} {p p' : Flock.Proof} {T T' : F128}
    (h : FitsRep c rows p T) (h' : FitsRep c rows p' T')
    (hs : (repFrames c.st c.sch c.dom c.n p T).map skel = (repFrames c.st c.sch c.dom c.n p' T').map skel) :
    repFrames c.st c.sch c.dom c.n p T = repFrames c.st c.sch c.dom c.n p' T' := by
  refine eq_of_skel_vals hs ?_
  have hc : (chunksOf c p T).map (List.map skel) = (chunksOf c p' T').map (List.map skel) := by
    unfold chunksOf
    rw [← chunks_skel, ← chunks_skel, hs]
  have hv : (chunksOf c p T).map vals = (chunksOf c p' T').map vals := by
    refine List.ext_getElem (by rw [List.length_map, List.length_map, h.rounds, h'.rounds]) fun j hj hj' => ?_
    rw [List.length_map, h.rounds] at hj
    have hl : (vals ((chunksOf c p T).getD j [])).length = (vals ((chunksOf c p' T').getD j [])).length := by
      rw [← vals_length_skel, getD_map_skel, hc, ← getD_map_skel, vals_length_skel]
    rw [List.getElem_map, List.getElem_map, ← List.getD_eq_getElem _ [], ← List.getD_eq_getElem _ []]
    exact (h.elems j hj).vals_eq (h'.elems j hj) hl
  rw [← chunks_flatten (repFrames c.st c.sch c.dom c.n p T), ← chunks_flatten (repFrames c.st c.sch c.dom c.n p' T'),
    repFrames_chunks_snd, repFrames_chunks_snd, List.append_nil, List.append_nil, vals_flatten, vals_flatten]
  exact congrArg List.flatten hv

theorem kFr_skel (m : Flock.F256 × Flock.F256) :
    (kFr m).map skel = [.vec [default, default], .vec [default, default]] := rfl

theorem chFr_skel (p p' : Flock.LigeritoProof) :
    ∀ (k msg msg' : ℕ), (chFr p msg k).map skel = (chFr p' msg' k).map skel
  | 0, _, _ => rfl
  | k + 1, msg, msg' => by
    simp only [chFr, List.map_cons, List.map_append, kFr_skel, chFr_skel p p' k (msg + 1) (msg' + 1)]

theorem introFr_skel {p p' : Flock.LigeritoProof} (hc : p.claimNonces = p'.claimNonces) (dim ood msg cbn : ℕ) :
    (introFr p dim ood msg cbn).map skel = (introFr p' dim ood msg cbn).map skel := by
  simp only [introFr, List.map_cons, List.map_append, kFr_skel, skel_one, hc]

theorem queryFr_skel {p p' : Flock.LigeritoProof} (hq : p.queryNonces = p'.queryNonces)
    (hs : p.consistencyNonces = p'.consistencyNonces) (lv : Flock.Level) (qn cn : ℕ) :
    (queryFr p lv qn cn).map skel = (queryFr p' lv qn cn).map skel := by
  simp only [queryFr, hq, hs]

theorem levelFr_skel {sch : Flock.Schedule} {p p' : Flock.LigeritoProof}
    (hr : p.recursiveCaps = p'.recursiveCaps) (hc : p.claimNonces = p'.claimNonces)
    (hq : p.queryNonces = p'.queryNonces)
    (hs : p.consistencyNonces = p'.consistencyNonces) (hy : p.finalProof.yr.size = p'.finalProof.yr.size) (i : ℕ) :
    (levelFr sch p i).map skel = (levelFr sch p' i).map skel := by
  unfold levelFr
  rw [List.map_append, List.map_append, chFr_skel p p']
  congr 1
  split
  · simp only [List.map_cons, List.map_append, kFr_skel, hr, hc, introFr_skel hc, queryFr_skel hq hs]
  · have e : ∀ q : Flock.LigeritoProof, (q.finalProof.yr.toList.map Frame.one).map skel =
        List.replicate q.finalProof.yr.size (.one default) := fun q => by
      rw [List.map_map, show skel ∘ Frame.one = fun _ => Frame.one default from rfl, List.map_const',
        Array.length_toList]
    simp only [List.map_append, List.map_cons, List.map_nil, e, hy, hc, queryFr_skel hq hs]

theorem ligFrames_skel {sch : Flock.Schedule} {cap : Array ByteArray} {T T' : F128} {p p' : Flock.LigeritoProof}
    (hr : p.recursiveCaps = p'.recursiveCaps) (hc : p.claimNonces = p'.claimNonces)
    (hq : p.queryNonces = p'.queryNonces) (hs : p.consistencyNonces = p'.consistencyNonces)
    (hy : p.finalProof.yr.size = p'.finalProof.yr.size) :
    (ligFrames sch T cap p).map skel = (ligFrames sch T' cap p').map skel := by
  have hlv : (List.range' 0 (sch.levels.size - 1)).flatMap (fun i => (levelFr sch p i).map skel) =
      (List.range' 0 (sch.levels.size - 1)).flatMap (fun i => (levelFr sch p' i).map skel) := by
    congr 1
    funext i
    exact levelFr_skel hr hc hq hs hy i
  have hch := chFr_skel p p' sch.k0 1 1
  simp only [ligFrames, List.map_cons, List.map_append, List.map_flatMap, List.map_nil, skel_one, kFr_skel, hch, hr,
    hc, hlv, introFr_skel hc, queryFr_skel hq hs]

/-- **Two proofs of one shape with one cap, recursive caps and nonces have frames of one skeleton**, whatever their
values and batched values. -/
theorem repFrames_skel {st : Flock.Setup} {sch : Flock.Schedule} {d : ByteArray} {n : ℕ} {sh : Flock.Shape}
    {p p' : Flock.Proof} {T T' : F128} (hp : ProofSizes sh p) (hp' : ProofSizes sh p') (hcap : p.cap = p'.cap)
    (hr : p.ligerito.recursiveCaps = p'.ligerito.recursiveCaps) (hc : p.ligerito.claimNonces = p'.ligerito.claimNonces)
    (hq : p.ligerito.queryNonces = p'.ligerito.queryNonces)
    (hs : p.ligerito.consistencyNonces = p'.ligerito.consistencyNonces) :
    (repFrames st sch d n p T).map skel = (repFrames st sch d n p' T').map skel := by
  have hrs : ∀ k : ℕ, (p.ringSwitches[k]!).size = (p'.ringSwitches[k]!).size := fun k => by
    rw [rs_size hp, rs_size hp']
  have hlg := ligFrames_skel (sch := sch) (cap := p'.cap) (T := T) (T' := T') hr hc hq hs (hp.yr.trans hp'.yr.symm)
  simp only [repFrames, zcFrames, lcFrames, rsFrames, List.map_cons, List.map_append, List.map_flatMap, List.map_nil,
    skel_dom, skel_label, skel_one, skel_vec, skel_bytes, skel_sq, Array.length_toList, hp.r1ab, hp'.r1ab, hp.r1c,
    hp'.r1c, hp.zPartial, hp'.zPartial, hcap, hrs, hlg]

theorem zeros_eq {a b : Array UInt64} (hs : a.size = b.size) (ha : ∀ n ∈ a, n = 0) (hb : ∀ n ∈ b, n = 0) : a = b :=
  Array.ext hs fun _ h1 h2 => by rw [ha _ (Array.getElem_mem h1), hb _ (Array.getElem_mem h2)]

/-- **Rows fix a fitting proof's absorbed fields and its `T`**: two fits have one cap (round 0's), recursive caps (the
rounds' tops) and nonces (`0`), so frames of one skeleton (`repFrames_skel`) and, round by round, of one value row
(`RowsAt.vals_eq`): one rep's frames (`fits_frames`), which fix every absorbed field (`repFrames_fields`,
`ligFrames_fields`) and `T`. -/
theorem fitUnique : FitUnique := by
  intro c rows p p' T T' hk h h' q
  obtain ⟨e, hn, hp⟩ := h.sizes
  obtain ⟨e', hn', hp'⟩ := h'.sizes
  obtain rfl : e' = e := by omega
  have hcap : p.cap = p'.cap := Array.toList_inj.1 (h.cap.trans h'.cap.symm)
  have hinj : Function.Injective (Array.toList : Array ByteArray → List ByteArray) :=
    fun _ _ he => Array.toList_inj.1 he
  have hr : p.ligerito.recursiveCaps = p'.ligerito.recursiveCaps :=
    Array.toList_inj.1 (List.map_injective_iff.2 hinj (h.tops.trans h'.tops.symm))
  have hq := zeros_eq (hp.queryNonces.trans hp'.queryNonces.symm) h.nonces.1 h'.nonces.1
  have hc := zeros_eq (hp.claimNonces.trans hp'.claimNonces.symm) h.nonces.2.1 h'.nonces.2.1
  have hs := zeros_eq (hp.consistencyNonces.trans hp'.consistencyNonces.symm) h.nonces.2.2 h'.nonces.2.2
  have hF := fits_frames h h' (repFrames_skel hp hp' hcap hr hc hq hs)
  rw [hn] at hF
  obtain ⟨-, ha, hb, hzr, hA, hB, hl, hrs, hlig⟩ := repFrames_fields hp hp' hk hF
  obtain ⟨-, hm, -, ho, -, -, hy⟩ := ligFrames_fields hp hp' h.levels hlig
  have hT : T = T' := by
    unfold ligFrames at hlig
    exact Frame.one.inj (List.cons.inj (List.cons.inj (List.append_inj hlig rfl).1).2).1
  refine ⟨?_, hT.symm⟩
  simp only [absorbedFrom, hcap, ha, hb, hzr, hA, hB, hl, hrs, hr, hm, hq, ho, hc, hs, hy]

/-! ## From one rep to the session -/

/-- **An honest proof is its own read-back**: its unabsorbed fields are already what V\* takes for them. -/
theorem absorbedFrom_self {c : RepCtx} {p : Flock.Proof} (hp : HonestRep c p) : absorbedFrom c p p = p := by
  have h1 := hp.params
  have h2 := hp.initialCap
  have h3 := hp.finalC
  rcases p with ⟨cap, params, ⟨a, b, r, fa, fb, fc⟩, lc, rs, ⟨ic, ip, rc, rp, ⟨yr, rows, sib, salts⟩, ms, qn, ood, cbn,
    cn⟩⟩
  dsimp only at h1 h2 h3
  subst h1 h2 h3
  rfl

/-- **One rep reads back where the honest proof fits**, as fits are unique (`fitUnique`). -/
theorem repReadsBack_of (hE : RepFits) : RepReadsBack := by
  intro c p T hs hp hc
  have hf := hE c p T hs hp hc
  have hx : ∃ pT : Flock.Proof × F128, FitsRep c (repRows c p T) pT.1 pT.2 := ⟨(p, T), hf⟩
  obtain ⟨h1, h2⟩ := fitUnique c _ p _ T _ hs.2.1 hf (Classical.choose_spec hx) p
  unfold readRep
  rw [dite_eq_left hx, h1, absorbedFrom_self hp, h2]

theorem rowsOf_rows {rs : List ByteArray} (h : RowsFit rs) : rowsOf (joinR (rs.map rowMsg)) = some rs := by
  rw [rowsOf_joinR (by rw [List.length_map]; exact h.1) (by
    intro r hr
    obtain ⟨b, hb, rfl⟩ := List.mem_map.1 hr
    show b.data.toList.length < 2 ^ 64
    rw [Array.length_toList]
    exact h.2 b hb)]
  have e : ((fun r : Claims.Msg => (⟨r.toArray⟩ : ByteArray)) ∘ rowMsg) = id := funext fun _ => rfl
  rw [List.map_map, e, List.map_id]

theorem mapM_rowsOf : ∀ {l : List (List ByteArray)}, (∀ rs ∈ l, RowsFit rs) →
    (l.map fun rs => joinR (rs.map rowMsg)).mapM rowsOf = some l
  | [], _ => rfl
  | rs :: l, h => by
    rw [List.map_cons, List.mapM_cons, rowsOf_rows (h rs (List.mem_cons_self ..)),
      mapM_rowsOf fun r hr => h r (List.mem_cons_of_mem _ hr)]
    rfl

theorem mapM_eq_some {α β : Type} {f : α → Option β} {g : α → β} :
    ∀ {l : List α}, (∀ a ∈ l, f a = some (g a)) → l.mapM f = some (l.map g)
  | [], _ => rfl
  | a :: l, h => by
    rw [List.mapM_cons, h a (List.mem_cons_self ..), mapM_eq_some fun b hb => h b (List.mem_cons_of_mem _ hb)]
    rfl

/-- **Blocks cut at their own lengths are the blocks.** -/
theorem cutLens_flatten_eq {α : Type} : ∀ (ls : List (List α)), cutLens (ls.map List.length) ls.flatten = ls
  | [] => rfl
  | l :: ls => by
    simp only [List.map_cons, List.flatten_cons, cutLens, List.take_left, List.drop_left, cutLens_flatten_eq ls]

theorem capLens {sts : Array Stmt} {caps : Array (Array ByteArray)} (hs : caps.size = sts.size)
    (hc : ∀ i < caps.size, caps[i]!.size = RecDeploy.level0Nodes sts[i]!) :
    (caps.toList.map Array.toList).map List.length = level0Lens sts := by
  refine List.ext_getElem (by simp [level0Lens, hs]) fun i h₁ h₂ => ?_
  have hi : i < caps.size := by simpa using h₁
  have hi' : i < sts.size := hs ▸ hi
  simp only [List.map_map, List.getElem_map, Array.getElem_toList, level0Lens, Function.comp_apply,
    Array.length_toList]
  rw [← getElem!_pos caps i hi, hc i hi, getElem!_pos sts i hi']

theorem cutLens_capRows {sts : Array Stmt} {caps : Array (Array ByteArray)} (hs : caps.size = sts.size)
    (hc : ∀ i < caps.size, caps[i]!.size = RecDeploy.level0Nodes sts[i]!) :
    cutLens (level0Lens sts) (capRows caps) = caps.toList.map Array.toList := by
  rw [← capLens hs hc, capRows, cutLens_flatten_eq]

/-- **The honest link**: each table's `root_B` its cap's nodes joined, and no public words. -/
theorem linkR_capRows {sts : Array Stmt} {caps : Array (Array ByteArray)} (hs : caps.size = sts.size)
    (hc : ∀ i < caps.size, caps[i]!.size = RecDeploy.level0Nodes sts[i]!) :
    linkR sts (capRows caps) =
      some ((caps.toList.map Array.toList).map (·.foldl (· ++ ·) ByteArray.empty),
        sts.toList.map fun _ => ByteArray.empty) := by
  have hsum : (level0Lens sts).sum = (capRows caps).length := by
    rw [← capLens hs hc, capRows, List.length_flatten]
  unfold linkR
  rw [ite_eq_left hsum, cutLens_capRows hs hc]

/-- **Rep `k` reads its table's level-0 cap among the honest round 0's rows.** -/
theorem cap0_capRows {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin} {caps : Array (Array ByteArray)}
    {cs : List Coin} (hs : caps.size = sts.size) (hc : ∀ i < caps.size, caps[i]!.size = RecDeploy.level0Nodes sts[i]!)
    {k : ℕ} (hk : k / 2 < caps.size) : (repCtx spec sts c₁ (capRows caps) cs k).cap0 = caps[k / 2]!.toList := by
  show (cutLens (level0Lens sts) (capRows caps)).getD (k / 2) [] = _
  rw [cutLens_capRows hs hc]
  simp only [List.getD_eq_getElem?_getD, List.getElem?_map, Array.getElem?_toList, Array.getElem?_eq_getElem hk,
    getElem!_pos caps (k / 2) hk, Option.map_some, Option.getD_some]

theorem range_map_getElem! {α : Type} [Inhabited α] (a : Array α) :
    ((List.range a.size).map fun k => a[k]!).toArray = a := by
  refine Array.ext (by simp) fun i h₁ h₂ => ?_
  simp only [List.getElem_toArray, List.getElem_map, List.getElem_range]
  exact getElem!_pos a i h₂

/-- **The session reads back where each rep does.** -/
theorem readsBack_of_rep (hR : RepReadsBack) : ReadsBack := by
  intro rd x last cs s spec sts c₁ caps ps Ts fl hx hc1 hs hcs hps hrep h0 hrows hrd hfl hst hsh
  set reps := (List.range ps.size).map fun k =>
    repRows (repCtx spec sts c₁ (capRows caps) cs.tail.dropLast k) ps[k]! (Ts k)
    with hreps
  have hk2 : ∀ k < ps.size, k / 2 < caps.size := fun k hk => by omega
  have hone : ∀ k < ps.size, readRep (repCtx spec sts c₁ (capRows caps) cs.tail.dropLast k)
      (repRows (repCtx spec sts c₁ (capRows caps) cs.tail.dropLast k) ps[k]! (Ts k)) ps[k]! =
      some (ps[k]!, framesOf (repCtx spec sts c₁ (capRows caps) cs.tail.dropLast k) ps[k]! (Ts k)) := fun k hk => by
    obtain ⟨h1, h2, h3⟩ := hrep k hk
    refine hR _ _ _ h1 h2 ?_
    rw [cap0_capRows hs hcs (hk2 k hk), h3]
  have hlen : ∀ k < ps.size, (repRows (repCtx spec sts c₁ (capRows caps) cs.tail.dropLast k) ps[k]! (Ts k)).length =
      (RecDeploy.repSchedAt spec sts[k / 2]!).length := fun k hk => by
    obtain ⟨T, hT, -⟩ := readRep_fits (hone k hk)
    exact hT.sched.symm
  have hlens : reps.map List.length = repLens spec sts := by
    rw [hreps, List.map_map, repLens, ← hps]
    exact List.map_congr_left fun k hk => hlen k (List.mem_range.1 hk)
  have hm : ((List.range ps.size).mapM fun k =>
      readRep (repCtx spec sts c₁ (capRows caps) cs.tail.dropLast k) (reps.getD k []) fl.ps[k]!) =
      some ((List.range ps.size).map fun k =>
        (ps[k]!, framesOf (repCtx spec sts c₁ (capRows caps) cs.tail.dropLast k) ps[k]! (Ts k))) := by
    refine mapM_eq_some fun k hk => ?_
    have hk' := List.mem_range.1 hk
    rw [hreps, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hk', Option.map_some,
      Option.getD_some, hfl]
    exact hone k hk'
  have hif : fl.ps.size = 2 * sts.size ∧ (repLens spec sts).sum = reps.flatten.length :=
    ⟨by rw [hfl, hps], by rw [← hlens, List.length_flatten]⟩
  have hpf : readPs spec sts c₁ (capRows caps) reps.flatten cs.tail.dropLast fl =
      some (ps, ((List.range ps.size).map fun k =>
        framesOf (repCtx spec sts c₁ (capRows caps) cs.tail.dropLast k) ps[k]! (Ts k)).flatten) := by
    unfold readPs
    rw [ite_eq_left hif, ← hlens, cutLens_flatten_eq, ← hps, hm]
    simp only [Option.map_some, List.map_map, Function.comp_def, range_map_getElem!]
  have hhead : (honestMsgs caps reps last).head? = some (joinR (((capRows caps).map nodeRow).map rowMsg)) := rfl
  have hlast : (honestMsgs caps reps last).getLast? = some last := List.getLast?_concat
  have hmid : (honestMsgs caps reps last).tail.dropLast.mapM rowsOf = some reps.flatten := by
    rw [show (honestMsgs caps reps last).tail.dropLast = reps.flatten.map fun rs => joinR (rs.map rowMsg) from
      List.dropLast_concat]
    refine mapM_rowsOf fun rs hr => ?_
    obtain ⟨l, hl, hrl⟩ := List.mem_flatten.1 hr
    rw [hreps] at hl
    obtain ⟨k, hk, rfl⟩ := List.mem_map.1 hl
    exact hrows k (List.mem_range.1 hk) rs hrl
  have hrd' : (rd last).filter (·.stmt == x) = some fl := Option.filter_eq_some_iff.2 ⟨hrd, beq_iff_eq.2 hst⟩
  have hpf' : (readPs spec sts c₁ (capRows caps) reps.flatten cs.tail.dropLast fl).filter (fun pf =>
      @decide (ShapedAt spec sts c₁ pf.1) (Classical.propDecidable _)) =
      some (ps, ((List.range ps.size).map fun k =>
        framesOf (repCtx spec sts c₁ (capRows caps) cs.tail.dropLast k) ps[k]! (Ts k)).flatten) :=
    Option.filter_eq_some_iff.2 ⟨hpf, @decide_eq_true _ (Classical.propDecidable _) hsh⟩
  simp only [decR, hx, hlast, hhead, hc1, hrd', rowsOf_rows h0, Option.map_some, map_nodeOf_nodeRow, hmid,
    linkR_capRows hs hcs, hpf', Option.bind_eq_bind, Option.bind_some, Option.pure_def]

end FlockDeploy
