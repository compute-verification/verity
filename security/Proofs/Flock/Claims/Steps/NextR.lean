module

public import Proofs.Flock.C3.Reps.AtList
public import Proofs.Flock.C3.Reps.RoundLinks
public import Proofs.Flock.C3.VStarList.Accepts
public import Proofs.Flock.Claims.Complete.Json.UnitSample
public import Proofs.Flock.Claims.Complete.Walk
public import Proofs.Flock.Claims.Complete.ZkTables.Scope
public import Proofs.Flock.Claims.KS.PlainKSLin
public import Proofs.Flock.Claims.KS.ZkOuter
public import Proofs.Flock.Claims.Sampling.Bext
public import Proofs.Flock.Claims.Steps.BoundDeploy
public import Proofs.Flock.Claims.Steps.NextCs
public import Proofs.Flock.Claims.Steps.NextP
public import Proofs.Flock.Claims.Steps.NextS
public import Proofs.Flock.Claims.Steps.NextSha
public import Proofs.Flock.Claims.Steps.VarLen
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-! # What the plain KS proofs read of `Claims.Recursion.NextR`, without its recursion imports. -/

@[expose] public section

namespace FlockDeploy

open Flock

open Lean (Json)

open FlockSoundness.Refine (Frame encs colBytes)

open FlockSoundness.Discharge.Exec (chunks repFrames zcFrames ligFrames levelFr introFr queryFr chFr)

noncomputable section

/-- An element-level message read back as its rows. -/
def rowsOf : Claims.Msg → Option (List ByteArray)
  | 0 :: body =>
    match (rdVLs rdVB).run body with
    | some (v, []) => some v
    | _ => none
  | _ => none

/-- **The node a cap row holds**: the row less its last 64 bytes. A deployed cap row is 128 bytes (RecCap's `top`;
RecOpen's `out`, the climbed node then 512 zero bits, reads the same registration, `RecVStar.checkShared`), so this
is its first 64, the node. -/
def nodeOf (row : ByteArray) : ByteArray := ⟨(row.data.toList.take (row.data.size - 64)).toArray⟩

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

/-- Each table's level-0 top's node count, in table order: round 0's rows. -/
def level0Lens (sts : Array Stmt) : List ℕ := sts.toList.map RecDeploy.level0Nodes

/-- **The link from round 0's rows**: each table's `root_B` its level-0 top's nodes joined, and no public words. -/
def linkR (sts : Array Stmt) (r₀ : List ByteArray) : Option (List ByteArray × List ByteArray) :=
  if (level0Lens sts).sum = r₀.length then
    some ((cutLens (level0Lens sts) r₀).map (·.foldl (· ++ ·) ByteArray.empty),
      sts.toList.map fun _ => ByteArray.empty)
  else none

/-- The element values a round's frames absorb, in order. -/
def vals : List Frame → List F128
  | [] => []
  | .one v :: fs => v :: vals fs
  | .vec vs :: fs => vs ++ vals fs
  | _ :: fs => vals fs

/-- **Round `s`'s opened rows hold the values `vs` its squeeze answers**: its committed tops' nodes, then, where it
absorbs, one row of `vs` padded to whole blocks by any bytes (`rec_residuals`: a row port's words after its elements
are unread; the honest prover's zeros are `padRow`, `rowsAt_padRow`); a round that absorbs nothing answers no
value. -/
def RowsAt (s : RecVStar.SchedRound) (rs : List ByteArray) (vs : List F128) : Prop :=
  rs.length = s.2.1.toList.sum + (if s.2.2.2 then 1 else 0) ∧
    (∃ pad : List UInt8, pad.length = (128 - (colBytes vs).length % 128) % 128 ∧
      rs.drop s.2.1.toList.sum = (if s.2.2.2 then [(⟨(colBytes vs ++ pad).toArray⟩ : ByteArray)] else [])) ∧
    (s.2.2.2 = false → vs = [])

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

end

end FlockDeploy
