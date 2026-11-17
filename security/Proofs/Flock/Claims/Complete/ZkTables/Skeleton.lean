module

public import Proofs.Flock.Claims.Complete.Skeleton
public import Proofs.Flock.Claims.Complete.ZkTables.Ready
public import Proofs.Flock.Claims.Complete.ZkTables.Run
public import Proofs.Flock.Claims.Complete.Zk.Verdict
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
import Proofs.Flock.Claims.Complete.Encode
import Proofs.Flock.Claims.Complete.Zk.Sim
import Proofs.Flock.Claims.Scope.Sizes

/-! The masked completeness proof's skeleton at any number of tables (`complete_of_goals`). -/

@[expose] public section

open Flock FlockDeploy Transcript

namespace FlockComplete.ZkTables

structure Case (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) where
  x : Claims.Msg
  w : Claims.Flock.Wit
  ω : TapeZk
  r : Fin ((V true).k x) → Coin
  s : Ins
  spec : SessionSpec
  sts : Array Flock.Stmt
  scope : Claims.Flock.Scope (deployZk rel) x
  rel_w : Claims.Flock.R (deployZk rel).parse x w
  ready : ready H x = true
  good : ¬ BadZk x r
  setup : setupOf true x = some (s, spec, sts)

def Case.resp {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation} (c : Case H rel) :
    ℕ → Claims.Msg :=
  fun j => (Claims.coinSeq c.r j).bytes

structure Run (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) extends Case H rel where
  record : String
  proofs : Array ByteArray
  bodies : Array Claims.Msg
  ran : sessionS H (fun j => (Claims.coinSeq r j).bytes) spec sts w ω = (.ok (record, proofs), bodies)

namespace Run

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

def hs (_ : Run H rel) : Flock.Hashes V := Flock.Hashes.ofSha512 (Claims.hOf H)

def tabs (o : Run H rel) : Array Setup := tabsOf o.sts

def final (o : Run H rel) : Final := ⟨o.x, o.record, o.proofs⟩

def sends (o : Run H rel) : List Claims.Msg := o.bodies.toList.map (List.cons 0)

def link (o : Run H rel) : List ByteArray × List ByteArray := (linkOf (o.sends.headD [])).getD ([], [])

def mids (o : Run H rel) : List (Claims.Msg × Coin) :=
  midsOf (o.sends ++ [o.final.encode]) ((List.range (o.bodies.size + 1)).map (Claims.coinSeq o.r))

def live (o : Run H rel) : Live :=
  match liveOf o.spec o.record with
  | .ok lv => lv
  | .error _ => ⟨#[], #[], #[], #[]⟩

def sess (o : Run H rel) : RunRecord := val (Record.decodeWith o.hs o.spec o.record)

def sch (o : Run H rel) (i : ℕ) : Schedule := (zkSchedsOf o.tabs)[i]!.1

def tPad (o : Run H rel) (i : ℕ) : ℕ := (zkSchedsOf o.tabs)[i]!.2

def stream (o : Run H rel) (i r : ℕ) : StreamRec :=
  (o.sess.streams.find? (·.name == s!"{o.spec.tables[i]!}/rep{r}")).getD default

def extras (o : Run H rel) (i : ℕ) : Array Claim := val (o.tabs[i]!.extra o.sess.points)

def bytes (o : Run H rel) (i r : ℕ) : ByteArray := o.proofs[2 * i + r]!

def zp (o : Run H rel) (i r : ℕ) : Zk.ZkProof :=
  val (Zk.decode (shapeZk o.tabs[i]! (o.sch i) (o.extras i).size) (o.bytes i r))

def lay (o : Run H rel) (i : ℕ) : Zk.PadLayout := Zk.PadLayout.new o.tabs[i]!.m o.tabs[i]!.kLog (o.sch i).k0

def t₀ (o : Run H rel) (i r : ℕ) : Transcript :=
  Transcript.init (o.stream i r).rounds (o.spec.domain o.spec.tables[i]! r).toUTF8

def rp (o : Run H rel) (i r : ℕ) : Zk.Replayed × Transcript :=
  val ((Zk.replayWith o.hs o.tabs[i]! (o.lay i) (o.zp i r).proof (o.zp i r).inner.root).run (o.t₀ i r))

def sw (o : Run H rel) (i r : ℕ) : (F128 × Array Switched) × Transcript :=
  val ((ringSwitchWith o.hs o.tabs[i]!.m (ringClaimsZk (o.rp i r).1 (o.zp i r).proof ++ o.extras i)
    (o.zp i r).proof.ringSwitches).run (o.rp i r).2)

def lg (o : Run H rel) (i r : ℕ) : Zk.L0Rec × Transcript :=
  val ((Zk.ligeritoWith o.hs o.tabs[i]!.merkle (o.sch i) (o.sw i r).1.1 (o.zp i r).proof.cap
    (o.zp i r).proof.ligerito (basis (o.sw i r).1.2) ⟨o.tPad i, r, (o.zp i r).level0⟩).run (o.sw i r).2)

def lanes (o : Run H rel) (i r : ℕ) : Array (String × Zk.Lin) :=
  val (Zk.ligeritoConstraints (o.lay i) (o.lg i r).1 (o.zp i r).proof.ligerito (o.zp i r).level0)

def inner (o : Run H rel) (i r : ℕ) : Unit × Transcript :=
  val ((Zk.verifyInnerWith o.hs (o.lay i) (o.zp i r).inner (o.zp i r).proof.zerocheck.finalC
    (repConsZk (o.rp i r).1 (o.zp i r).proof ++ o.lanes i r)).run (o.lg i r).2)

end Run

section Goals

variable (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)

def SessionRunsZk : Prop :=
  ∀ c : Case H rel, ∃ record proofs bodies, sessionS H c.resp c.spec c.sts c.w c.ω = (.ok (record, proofs), bodies)

def SendCountZk : Prop := ∀ o : Run H rel, o.bodies.size ≤ 1 + rounds true o.spec o.sts

def LinkFirstZk : Prop :=
  ∀ o : Run H rel, ∃ rootB publics : List ByteArray, o.bodies.toList.head? = some (linkBody rootB publics) ∧
    rootB.length < 2 ^ 64 ∧ publics.length < 2 ^ 64 ∧ (∀ b ∈ rootB, b.size < 2 ^ 64) ∧ ∀ b ∈ publics, b.size < 2 ^ 64

def FinalFitsZk : Prop :=
  ∀ o : Run H rel, o.record.toUTF8.size < 2 ^ 64 ∧ o.proofs.size < 2 ^ 64 ∧ ∀ b ∈ o.proofs, b.size < 2 ^ 64

def SampleMatchesZk : Prop :=
  ∀ o : Run H rel, sampleEq (FlockSoundness.Discharge.Exec.recordDraw o.record) o.s.read.draw = true

def LiveReadsZk : Prop := ∀ o : Run H rel, liveOf o.spec o.record = .ok o.live

def RecordCanonicalZk : Prop := ∀ o : Run H rel, Flock.RecordText.check o.record = .ok ()

def CustodyRootBZk : Prop := ∀ o : Run H rel, (o.live.rootB.toList == o.link.1) = true

def CustodyPublicsZk : Prop := ∀ o : Run H rel, (o.live.publics.toList == o.link.2) = true

def CustodyPointsZk : Prop := ∀ o : Run H rel, (o.live.points == pointsOf o.spec (Claims.coinSeq o.r 0)) = true

def CustodyCountZk : Prop :=
  ∀ o : Run H rel, ((o.live.streams.toList.flatMap (·.toList)).length == o.mids.length) = true

def CustodyRoundsZk : Prop :=
  ∀ o : Run H rel, ((o.live.streams.toList.flatMap (·.toList)).zip o.mids).all
    (fun (r, a, c) => r.bytes == some ⟨a.toArray⟩ && r.coins == c.vals r.n) = true

def TablesZk : Prop := ∀ o : Run H rel, (o.tabs.size != o.spec.tables.size || o.tabs.size == 0) = false

def CoinTree : Prop := ∀ o : Run H rel, o.spec.coinTree.isNone = false

def DecodesZk : Prop := ∀ o : Run H rel, Record.decodeWith o.hs o.spec o.record = .ok o.sess

def PublicsZk : Prop := ∀ o : Run H rel, ∀ q ∈ (o.tabs.zip o.sess.publics).toList, q.1.publics q.2 = .ok ()

def SchedulesZk : Prop := ∀ o : Run H rel, ∀ st ∈ o.tabs.toList, ∃ v, Zk.schedule st.wideLeaf st.m = .ok v

def ProofCountZk : Prop := ∀ o : Run H rel, (o.proofs.size != o.spec.repStreams.size) = false

def ProofListZk : Prop :=
  ∀ o : Run H rel, (o.sess.proofSha.size != o.spec.repStreams.size ||
    !o.spec.repStreams.all fun a => o.sess.proofSha.any fun x => x.1 == a.2.2) = false

def ProofDigestsZk : Prop :=
  ∀ o : Run H rel, ∀ q ∈ (o.spec.repStreams.zip o.proofs).toList,
    ((o.sess.proofSha.find? (·.1 == q.1.2.2)).map (·.2) != some (hexOf (dgOf o.spec (Claims.hOf H) q.2))) = false

def RepStreamZk : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2,
    (o.sess.streams.find? (·.name == s!"{o.spec.tables[i]!}/rep{r}")).isSome = true

def RepExtrasZk : Prop := ∀ o : Run H rel, ∀ i < o.tabs.size, o.tabs[i]!.extra o.sess.points = .ok (o.extras i)

def RepDecodesZk : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2,
    Zk.decode (shapeZk o.tabs[i]! (o.sch i) (o.extras i).size) (o.bytes i r) = .ok (o.zp i r)

def RepParamsZk : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2,
    ((o.zp i r).proof.params != pcsParams o.tabs[i]!.merkle (o.sch i)) = false

def PadsRoot : Prop := ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, ((o.zp i r).inner.root.size != 64) = false

def Replay : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2,
    (Zk.replayWith o.hs o.tabs[i]! (o.lay i) (o.zp i r).proof (o.zp i r).inner.root).run (o.t₀ i r) = .ok (o.rp i r)

def RingSwitchZk : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2,
    (ringSwitchWith o.hs o.tabs[i]!.m (ringClaimsZk (o.rp i r).1 (o.zp i r).proof ++ o.extras i)
      (o.zp i r).proof.ringSwitches).run (o.rp i r).2 = .ok (o.sw i r)

def LigeritoZk : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2,
    (Zk.ligeritoWith o.hs o.tabs[i]!.merkle (o.sch i) (o.sw i r).1.1 (o.zp i r).proof.cap (o.zp i r).proof.ligerito
      (basis (o.sw i r).1.2) ⟨o.tPad i, r, (o.zp i r).level0⟩).run (o.sw i r).2 = .ok (o.lg i r)

def LaneConstraints : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2,
    Zk.ligeritoConstraints (o.lay i) (o.lg i r).1 (o.zp i r).proof.ligerito (o.zp i r).level0 = .ok (o.lanes i r)

def Inner : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2,
    (Zk.verifyInnerWith o.hs (o.lay i) (o.zp i r).inner (o.zp i r).proof.zerocheck.finalC
      (repConsZk (o.rp i r).1 (o.zp i r).proof ++ o.lanes i r)).run (o.lg i r).2 = .ok (o.inner i r)

def ConsumedZk : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, ((o.inner i r).2.k != (o.stream i r).rounds.size) = false

def CapCommittedZk : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, (flat (o.zp i r).proof.cap != o.sess.rootB[i]!) = false

/-- **The masked goals** at any number of tables, every check but the six proved here; `GoalsZk`'s fields. -/
structure Goals : Prop where
  session : SessionRunsZk H rel
  sends : SendCountZk H rel
  linkFirst : LinkFirstZk H rel
  finalFits : FinalFitsZk H rel
  draw : SampleMatchesZk H rel
  liveReads : LiveReadsZk H rel
  rootB : CustodyRootBZk H rel
  publicsLink : CustodyPublicsZk H rel
  points : CustodyPointsZk H rel
  count : CustodyCountZk H rel
  rounds : CustodyRoundsZk H rel
  tables : TablesZk H rel
  decodes : DecodesZk H rel
  publics : PublicsZk H rel
  schedules : SchedulesZk H rel
  proofCount : ProofCountZk H rel
  proofList : ProofListZk H rel
  proofDigests : ProofDigestsZk H rel
  repStream : RepStreamZk H rel
  repExtras : RepExtrasZk H rel
  repDecodes : RepDecodesZk H rel
  repParams : RepParamsZk H rel
  padsRoot : PadsRoot H rel
  replay : Replay H rel
  ringSwitch : RingSwitchZk H rel
  ligerito : LigeritoZk H rel
  lanes : LaneConstraints H rel
  inner : Inner H rel
  consumed : ConsumedZk H rel
  capCommitted : CapCommittedZk H rel
  canonical : RecordCanonicalZk H rel

end Goals

section Proved

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

theorem Case.ready_all (c : Case H rel) :
    c.s.read.draw.isSome = true ∧ c.x.length < 2 ^ 64 ∧ (∀ st ∈ c.sts, RegionsFit st) ∧
      setupWith (Claims.hOf H) c.s.read = .ok (c.spec, c.sts) ∧ c.spec.coinTree.isSome = true ∧
      RecordFitsAt H c.x c.spec c.sts := by
  have h := c.ready
  unfold ZkTables.ready at h
  rw [c.setup] at h
  simpa [and_assoc] using h

theorem Case.ready_parts (c : Case H rel) :
    c.s.read.draw.isSome = true ∧ c.x.length < 2 ^ 64 ∧ (∀ st ∈ c.sts, RegionsFit st) ∧
      setupWith (Claims.hOf H) c.s.read = .ok (c.spec, c.sts) ∧ c.spec.coinTree.isSome = true :=
  have h := c.ready_all
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1⟩

theorem Run.recordFits (o : Run H rel) : o.record.toUTF8.size < 2 ^ 64 := by
  obtain ⟨bs, h1, -⟩ := sessionPZk_sim H (fun j => (Claims.coinSeq o.r j).bytes) o.spec (tabsOf o.sts)
    (MaskRank.masksOf o.sts) (traces o.sts o.w) o.ω #[]
  have hran := o.ran
  unfold sessionS at hran
  rw [hran] at h1
  exact o.ready_all.2.2.2.2.2 o.w o.ω o.r o.record o.proofs (by simpa using congrArg Prod.snd h1)

theorem coinTree_goal : CoinTree H rel := fun o => by
  have := o.ready_parts.2.2.2.2
  cases h : o.spec.coinTree
  · rw [h] at this; cases this
  · rfl

theorem Case.drawn (c : Case H rel) : c.s.read.draw.isNone = false := by
  have := c.ready_parts.1
  cases h : c.s.read.draw
  · rw [h] at this; cases this
  · rfl

theorem Case.scope_ok (c : Case H rel) : c.s.read.check = .ok () := by
  obtain ⟨p, hp, hsc⟩ := (Readable.scope_iff (deployZk rel) rel (readOf true) rfl
    (fun _ _ h => readOf_zk h) c.x).1 c.scope
  have hi := (setupOf_some c.setup).1
  unfold Readable.parseBytes readOf at hp
  rw [hi] at hp
  have ht := Readable.toRead_of_parse (r := c.s.read) hp
  rw [← ht]
  exact ((Readable.scope_ok_iff p).1 hsc).1

theorem Run.stmt_ok (o : Run H rel) : (o.final.stmt == o.x) = true := beq_self_eq_true _

theorem Run.link_ok (hL : LinkFirstZk H rel) (o : Run H rel) : linkOf (o.sends.headD []) = some o.link := by
  obtain ⟨rootB, publics, hh, h1, h2, h3, h4⟩ := hL o
  have e : o.sends.headD [] = 0 :: linkBody rootB publics := by
    unfold Run.sends
    cases hb : o.bodies.toList with
    | nil => rw [hb] at hh; cases hh
    | cons b bs =>
      rw [hb] at hh
      simp only [List.head?_cons, Option.some.injEq] at hh
      simp [hh]
  have hl := linkOf_linkBody h1 h2 h3 h4
  unfold Run.link
  rw [e, hl]
  rfl

theorem Run.final_ok (hF : FinalFitsZk H rel) (o : Run H rel) : finalOf o.final.encode = some o.final := by
  obtain ⟨h1, h2, h3⟩ := hF o
  exact finalOf_encode o.ready_parts.2.1 h1 h2 h3

theorem Run.rep_ok (g : Goals H rel) (o : Run H rel) {i : ℕ} (hi : i < o.tabs.size) {r : ℕ} (hr : r < 2) :
    Zk.verifyRepWith o.hs o.spec o.tabs[i]! (o.sch i) (o.tPad i) o.sess i r (o.bytes i r) = .ok o.sess.rootB[i]! := by
  have hS := g.repStream o i hi r hr
  have hfind : o.sess.streams.find? (·.name == s!"{o.spec.tables[i]!}/rep{r}") = some (o.stream i r) := by
    unfold Run.stream
    cases hf : o.sess.streams.find? (·.name == s!"{o.spec.tables[i]!}/rep{r}") with
    | none => rw [hf] at hS; cases hS
    | some st => rfl
  rw [← bytes_eq_of_bne (g.capCommitted o i hi r hr)]
  exact zk_verifyRepWith_ok o.hs o.spec o.tabs[i]! (o.sch i) (o.tPad i) o.sess i r (o.bytes i r) hfind
    (g.repExtras o i hi) (g.repDecodes o i hi r hr) (g.repParams o i hi r hr) (g.padsRoot o i hi r hr)
    (g.replay o i hi r hr) (g.ringSwitch o i hi r hr) (g.ligerito o i hi r hr) (g.lanes o i hi r hr)
    (g.inner o i hi r hr) (g.consumed o i hi r hr) (g.capCommitted o i hi r hr)

theorem Run.verdict_ok (g : Goals H rel) (o : Run H rel) :
    verdictWith true (Claims.hOf H) o.spec (tabsOf o.sts) o.record o.proofs = .ok () :=
  zk_verifyWith_ok (Claims.hOf H) o.spec o.tabs o.record o.proofs (g.tables o) (coinTree_goal o) (g.decodes o)
    (g.publics o) (g.schedules o) (g.proofCount o) (g.proofList o) (g.proofDigests o)
    fun i hi => ⟨o.sess.rootB[i]!, o.rep_ok g hi (by decide), o.rep_ok g hi (by decide)⟩

theorem Run.checks_ok' (g : Goals H rel) (o : Run H rel) :
    checks true o.s o.spec o.x o.link (Claims.coinSeq o.r 0) o.mids o.final = .ok () :=
  checks_ok o.stmt_ok (g.draw o) (g.liveReads o)
    (custody_ok (g.rootB o) (g.publicsLink o) (g.points o) (g.count o) (g.rounds o)) o.scope_ok o.drawn
    (g.canonical o)

theorem Run.acc (g : Goals H rel) (o : Run H rel) :
    (Claims.run H (honestP o.w) (V true) o.x o.ω o.r).out = .acc := by
  have he := honestRun_eq H o.x o.w o.ω o.r o.setup o.ran
  have hk : (V true).k o.x = 2 + rounds true o.spec o.sts := by
    show (match setupOf true o.x with
      | some (_, spec, sts) => 2 + rounds true spec sts
      | none => 1) = _
    rw [o.setup]
  have hlen : (honestRun H o.x o.w o.ω o.r).1.length = o.bodies.size := by
    rw [he]; simp
  refine run_acc_of H o.w o.x o.ω o.r (link := o.link) (F := o.final) o.setup o.ready_parts.2.2.2.1 ?_ ?_ ?_ ?_ ?_
    (o.verdict_ok g) ?_
  · rw [hlen, hk]
    have := g.sends o
    omega
  · rw [he]
    intro m hm
    obtain ⟨b, -, rfl⟩ := List.mem_map.1 hm
    exact ⟨b, rfl⟩
  · rw [he]
    obtain ⟨rootB, publics, hh, -⟩ := g.linkFirst o
    intro h0
    simp only [List.map_eq_nil_iff, Array.toList_eq_nil_iff] at h0
    rw [Array.toList_eq_nil_iff.2 h0] at hh
    cases hh
  · rw [he]
    exact o.link_ok g.linkFirst
  · rw [he]
    exact o.final_ok g.finalFits
  · have hc := o.checks_ok' g
    rw [he]
    unfold Run.mids Run.sends at hc
    simp only [List.length_map, Array.length_toList]
    exact hc

end Proved

theorem complete_of_goals (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (hMR : MaskRank.MaskRankBound) (hMF : MaskFree rel) (g : Goals H rel) : Claims.Flock.Complete H (deployZk rel) := by
  refine complete_of H rel hMR hMF fun x w hsc hr hready ω r hgood => ?_
  have hset : ∃ s spec sts, setupOf true x = some (s, spec, sts) := by
    unfold ZkTables.ready at hready
    split at hready
    · rename_i s spec sts hs
      exact ⟨s, spec, sts, hs⟩
    · cases hready
  obtain ⟨s, spec, sts, hs⟩ := hset
  let c : Case H rel := ⟨x, w, ω, r, s, spec, sts, hsc, hr, hready, hgood, hs⟩
  obtain ⟨record, proofs, bodies, hran⟩ := g.session c
  exact Run.acc g ⟨c, record, proofs, bodies, hran⟩

end FlockComplete.ZkTables
