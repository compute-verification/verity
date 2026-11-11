module

public import Proofs.Properties.Verity.Sampled.CouplingAt
public import Proofs.Properties.Verity.Sampled.Counts
import Proofs.Flock.Claims.Complete.Json.Stmt

/-!
# The count match on the moves' reach, at the session's layout

At the readings and the decoders at the session's layout (`Live.layout`, `Live.tables`) and the sample-coupled coin map
under `fieldEnc`, every outcome the joint law reaches off the empty classes at their counts (`linkEmptyAt`,
`roundEmptyAt`), beside an accepted play that misses nowhere and whose log v2 read is the device's, it reaches off the
empty classes (`CountsReach`): each stop's opening asks for the count of coins the stop releases. It needs no verifier
fact beyond those `MovesReach` reads: v2 reads each opening frame's count off the line that opens its round, and
`sessionTreeReason` holds that line's count to the record's.

* **The count a line opening a round shows** (`opening_count`): the line of an accepted release log opening a round a
  tree shows opened has that opening's `n` (`opened_after`).
* **The count the record gives a decoded round** (`nat?_of_recordNat`, `recordCount_decode`): `Trees.recordCount` reads
  of a stream's round `k`, not the session stream's, the `n` `Record.decode` decodes there.
* **The link's count** (`link_count`, `SetsUp.mPts_eq`): the session stream's round 0 gives a coin a 16 bytes of the
  two link points of `spec.mPts` coordinates each (`link_coins`, `decode_points`), and the session `--zk` sets up at
  the record's sample has `A.mP` of them (`drawOkZKHJP_of_size`).
* **A stop's frame** (`RoundsReach.stop_frame`, `firstOpened_noPick`): a stop where `StopOf` says and not every cut is
  past releases, on the tape a trace from it ends on, at an opening frame its fixed rounds pick, after a log none of its
  live rounds picks in, its cuts at the bytes that frame answers, and every description of its opening asks for that
  frame's count (`OpeningAt.n_eq`). The first stop's log opens no round, since the link's slot picks the log's first
  opening (`link_picks`).
* **A trace off another hold** (`RoundsReach.weaken`): a trace off `B` is one off `B'` wherever every stop it answers
  is off `B'`.
* **An answered stop's count** (`stop_count`): at a stop that fixes the cut of round `k` of the first table's rep `r`
  stream at the message `w`, under `stop_round`'s premises, the opening frame there asks for `tailCount w` coins: the
  frame is v2's read of the line opening the round (`readLog_round`), whose count is the opening the record's tree shows
  (`opening_count`), whose count is the record's (`Trees.round_counted`), which is `tailCount` of the round's message
  (`rep_tailCount_accepted`), which is `w` (`RoundsReach.stop_modes_reveal`).
* **The link's frame count** (`link_frameCount`): the log's first opening, of the session stream, asks for
  `A.mP + A.mP` coins (`Trees.session_counted`, `link_count`, `SetsUp.mPts_eq` at the record's sample).
* **`CountsReach`** (`ReachesOff.counts`, `countsReach_layout`, `countsReachAt`): at the first stop by
  `link_frameCount`, and at each answered stop by `stop_count`, so the coupling at the deployment off the empty classes
  at their counts (`couplingTargetAt_counts`) takes `couplingTargetAt`'s premises but its bound on the stops' term.
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open Lean (Json)
open Spot.Challenger
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Tables (LiveOutJ SubmitJ addRoundJ)
open FlockSoundness.Refine (roundOf ebind_ok)
open Claims.Bridge.Custody (decode_ok DecStream forall₂_find? forall₂_mem_left forall₂_getElem)

local notation "𝔽" => FlockLevel3.GF128

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-! ## The counts the release log and the record give -/

/-- **The line opening a round a tree shows opened counts the opening's coins**: where `coinTreesOf` accepts the
release log `lines`, a line of it opening round `k` of stream `name` of a tree `t` of its trees (`OpensRound`) is the
one line opening that round (`opensRound_unique`), so its `n` is that of the opening `t` shows (`opened_after`). -/
theorem opening_count {lines : List String} {trees : Array LoggedTree} (hT : coinTreesOf lines = .ok trees)
    {t : LoggedTree} (ht : t ∈ trees) {name : String} {k : ℕ} {o : TreeOpening} (ho : t.opening name k = some o)
    {l : String} (hl : l ∈ lines) (hr : OpensRound l t.root name k) {d : Json} (hd : event l = .ok d) :
    _root_.Spot.nat? (_root_.Spot.getD d "n") = some o.n := by
  have hp := Array.find?_some ho
  simp only [Bool.and_eq_true, beq_iff_eq] at hp
  obtain ⟨l', hl', hr', d', hd', -, hn⟩ := opened_after hT t ht o (Array.mem_of_find?_eq_some ho)
  rw [hp.1, hp.2] at hr'
  obtain rfl := opensRound_unique hT hl hl' hr hr'
  rw [hd] at hd'
  cases hd'
  exact hn

/-- A natural number `Record.nat` reads is the one `nat?` reads of the field. -/
theorem nat?_of_recordNat {r : Json} {key : String} {n : ℕ} (h : Flock.Record.nat r key = .ok n) :
    _root_.Spot.nat? (_root_.Spot.getD r key) = some n := by
  unfold Flock.Record.nat at h
  obtain ⟨v, hv, h⟩ := ebind_ok h
  rw [fld_getD hv]
  cases hg : v.getNat? with
  | error e => rw [hg] at h; cases h
  | ok a =>
    rw [hg] at h
    obtain rfl : a = n := Except.ok.inj h
    rw [FlockComplete.JsonRT.getNat_ok hg]
    simp [_root_.Spot.nat?, _root_.Spot.int?]

/-- **The count the record gives a decoded stream's round is the decoded round's**: where `record` decodes to `sess`
and parses as `J`, `Trees.recordCount` reads of round `k` of `J`'s stream `name`, not the session stream, the `n` of
round `k` of `sess`'s stream so named. -/
theorem recordCount_decode {spec : Flock.SessionSpec} {record : String} {sess : Flock.RunRecord} {J : Json}
    {name : String} {y : Flock.StreamRec} {k : ℕ} (hd : Flock.Record.decode spec record = .ok sess)
    (hJ : Flock.JsonParse.parse record = .ok J) (hn : (name == TREE_SESSION) = false)
    (hy : sess.streams.find? (·.name == name) = some y) (hk : k < y.rounds.size) :
    Trees.recordCount J name k = some y.rounds[k].n := by
  obtain ⟨j, l, recs, hp, -, hrecs, -, -, hstr⟩ := decode_ok hd
  rw [hJ] at hp
  obtain rfl := Except.ok.inj hp
  have hy' : sess.streams.toList.find? (·.name == name) = some y := by rw [Array.find?_toList]; exact hy
  obtain ⟨x, ⟨-, rs, hrs, hrds⟩, hx⟩ := forall₂_find?
    (R := fun y x => DecStream spec x y) (p := fun y => y.name == name)
    (q := fun x => (Flock.Record.str x "stream").toOption == some name)
    (fun a b hab => by rw [hab.1]; rfl) (forall₂_swap hstr) hy'
  have hlen : rs.size = y.rounds.size := by simpa using hrds.length_eq
  have hrk : k < rs.size := hlen ▸ hk
  have hfield := forall₂_getElem hrds hrk
  rw [getElem!_pos y.rounds k hk] at hfield
  obtain ⟨⟨hnat, -⟩, -⟩ := hfield
  have hx' : recs.find? (fun x => (Flock.Record.str x "stream").toOption == some name) = some x := by
    rw [← Array.find?_toList]; exact hx
  simp only [Trees.recordCount, hn, Bool.false_and, Bool.false_eq_true, ↓reduceIte, arr_getD hrecs]
  rw [find?_congr (q' := fun x => (Flock.Record.str x "stream").toOption == some name) ?_, hx']
  · simp only [Option.bind_some, arr_getD hrs, Array.getElem?_eq_getElem hrk]
    exact nat?_of_recordNat hnat
  · intro a ha
    obtain ⟨y', -, hy''⟩ := forall₂_mem_left hstr a (Array.mem_toList_iff.2 ha)
    rw [str_getD hy''.1, hy''.1]
    rfl

/-- **The link's count**: the coins the record gives the session stream's round 0, a coin a 16 bytes, are the two
decoded link points' coordinates (`link_coins`), `spec.mPts` each (`decode_points`). -/
theorem link_count {spec : Flock.SessionSpec} {record : String} {sess : Flock.RunRecord} {J : Json} {b : ByteArray}
    (hd : Flock.Record.decode spec record = .ok sess) (hJ : Flock.JsonParse.parse record = .ok J)
    (hp2 : spec.points = 2) (hb : recordRound J TREE_SESSION 0 = some b) : b.size / 16 = spec.mPts + spec.mPts := by
  obtain ⟨cs, hrr, hf⟩ := link_coins hd hJ hp2
  rw [hrr] at hb
  have hpts := FlockSoundness.Refine.decode_points hd
  have h1 : 1 < sess.points.size := by have := hpts.1; omega
  have hs0 : sess.points[0]!.size = spec.mPts :=
    hpts.2 _ (by rw [getElem!_pos sess.points 0 (by omega)]; exact Array.getElem_mem_toList _)
  have hs1 : sess.points[1]!.size = spec.mPts :=
    hpts.2 _ (by rw [getElem!_pos sess.points 1 h1]; exact Array.getElem_mem_toList h1)
  have hl := congrArg List.length (bits_of_coins hf hb)
  rw [Lines.bitsOfBytes_length, length_chunks fieldBits, Array.length_toList, Array.size_append, hs0, hs1] at hl
  omega

/-- **The session `--zk` sets up at the record's sample has `A.mP` points' coordinates**: at one table, its one
statement is set up at the record's sample, the session's sample file at a sample, where every statement has
`mPtsOf A.D.unroll.g (A.kd / nTab A.I)` of them (`drawOkZKHJP_of_size`). -/
theorem SetsUp.mPts_eq {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {A : Live} (hA : SetsUp V θ F A)
    {record : String} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (hs : Exec.setupOfP (ZkExec.zkInputs A.I) A.own record = .ok (spec, sts)) {S : Finset (Fin A.n)}
    (hdr : Exec.recordDraw record = some (A.dj S)) : spec.mPts = A.mP := by
  have hpin := ZkHidden.drawOkZKHJP_of_size A.I A.own A.ht A.hcS (.subset A.kd) A.kd A.n A.hdiv A.dj
    (drawFileZK_printedDraw A.kd) S (A.hsz S)
  have h1 : A.I.count ≤ 1 := hA.count.le
  unfold Exec.setupOfP at hs
  rw [hdr] at hs
  rcases HiddenExec.setupTables_okHP ((ZkExec.zkTags_hiddenOutputs A.I).trans A.hTags.1) hs with
    ⟨-, st, hst, rfl, -⟩ | ⟨h1', -⟩
  · refine (hpin ⟨0, Composed.nTab_pos A.I⟩).2 st ?_
    simp only [Exec.tagsAt, Composed.drawJ, ite_eq_left h1]
    exact hst
  · exact absurd h1' (by show ¬ 1 < A.I.count; omega)

/-! ## A stop's frame -/

section Frames

variable {g : Game Exchange} {σ : Strategy g} {Ω Reg : Type} {R mPts : ℕ} {D : Decoders R Reg}
  {M : CoinMap g σ Ω R mPts} {dg : ByteArray → ByteArray} {J : ℕ} {B : RoundHold}

/-- **A stop's opening frame in the log**: at a position `q` that keeps the invariant at the cuts `Ks₀`, where `StopOf`
says from the log `pre₀` and not every cut is past, where each round's reading picks only openings the device serves,
the play on the tape any trace from `q`, any rewriting of its tape consumed, ends on has after `pre₀` and a log `L`
none of the live rounds picks in an opening frame `.opening i s n` that some round whose cut `q` fixes picks, as every
such round does; the cuts `q` fixes are at the bytes it answers (`CutsAt`), and every description of `q`'s opening asks
for its `n` coins (`OpeningAt.n_eq`). -/
theorem RoundsReach.stop_frame {P : Fin R → List Release → Frame → Prop}
    {rv : Fin R → Exchange → Option Claims.Msg} {sg : Signer} {θ : ByteArray} {F : Array ByteArray}
    {c rq : ByteArray} (hPS : ∀ j pre i s n, P j pre (.opening i s n) → Served pre i s n)
    {Ks₀ : Fin R → DigestCut Claims.Msg g} {q : Pos g σ Claims.Msg (Fin R)} (hi : q.Inv Ks₀)
    {pre₀ : List Release} {f : List Release → Game Exchange} {a : Fin R → Prop}
    (hf : ∀ log, f log = finish sg θ F c rq (pre₀ ++ log)) (hS : StopOf P rv pre₀ f a q)
    (hK : ¬ ∀ j, q.Ks j = .skip) {φ : CoinTape q.g' q.s' → CoinTape q.g' q.s'} {β : Type}
    {fin : LiveOutJ J mPts → β} {rootB : Array ByteArray} {pts : Fin 2 → Fin mPts → 𝔽} {N : ℕ}
    {rs : Fin J → Fin 2 → List Flock.Round} {o : β × CoinTape g σ}
    (hr : RoundsReach D M dg J B fin rootB pts N rs (q.rewrite φ).consume o) :
    ∃ (L : List Release) (i s n : ℕ) (b : ByteArray) (bits : List Bool) (L' : List Release),
      NoPick P a pre₀ L ∧ (∃ j, (q.Ks j).isHere = true ∧ P j (pre₀ ++ L) (.opening i s n)) ∧
      (∀ j, (q.Ks j).isHere = true → P j (pre₀ ++ L) (.opening i s n)) ∧
      (CoinTape.run g σ o.2).log = (pre₀ ++ L) ++ ⟨.opening i s n, b, bits⟩ :: L' ∧
      CutsAt rv q (pre₀ ++ L) (CoinTape.run g σ o.2).log ∧
      ∀ o' : OpeningAt (finish sg θ F c rq) q.g', o'.n = n := by
  obtain ⟨L, hNP, h⟩ := hS
  rcases h with ⟨hK', -⟩ | ⟨k, fs, ts, f', i, s, n, b, hf', hT, ⟨j, haj, hj⟩, s'', Ks, x'', back'', hq, hKs⟩
  · exact absurd hK' hK
  have hjK : (q.Ks j).isHere = true := by
    rw [hq]
    exact (isHere_openCuts hKs j).2 ⟨haj, hj⟩
  have hall : ∀ j', (q.Ks j').isHere = true → P j' (pre₀ ++ L) (.opening i s n) := fun j' h' => by
    rw [hq] at h'
    exact ((isHere_openCuts hKs j').1 h').2
  have hf'' : ∀ log, f' log = finish sg θ F c rq ((pre₀ ++ L) ++ log) := fun log => by
    rw [hf', hf, List.append_assoc]
  have hSv := hPS j _ i s n hj
  subst hq
  obtain ⟨T, hTi, _, _, hp, -⟩ := Served.next hT hSv
  have hc := Served.cap hT hSv
  obtain ⟨z, hz, -⟩ := hr.head (OpeningAt.ofNext (k := k) (b := b) hf'' hc hTi hp).eq (Pos.isHere_consume _)
  have hX : CoinTape.run g σ o.2 = CoinTape.run (optOpenK k fs ts f' i s n b ts[i]?) s'' z := by
    rw [hz]
    exact hi.run z
  obtain ⟨L', hL'⟩ := log_of_opening hf'' hc hTi hp s'' z
  rw [← hX] at hL'
  refine ⟨L, i, s, n, b, _, L', hNP, ⟨j, hjK, hj⟩, hall, hL', cutsAt_openCuts rfl hKs hL', fun o' => ?_⟩
  exact OpeningAt.n_eq (finish_recordsLog sg θ F c rq) o' (OpeningAt.ofNext (k := k) (b := b) hf'' hc hTi hp)

/-- **The first stop's log opens no round**: where round `j₀`'s reading picks what the link's slot picks, the device
serves every opening of the log `L ++ L''`, and no round picks in `L` after the empty log, `L` opens no round, since
the link's slot picks the first opening of a log that has one (`link_picks`). -/
theorem firstOpened_noPick {P : Fin R → List Release → Frame → Prop} (tables : Array String) {j₀ : Fin R}
    (hj₀ : ∀ pre fr, slotRound tables .link pre fr → P j₀ pre fr) {log L L'' : List Release} (hS : LogServed log)
    (hlog : log = L ++ L'') (hNP : NoPick P (fun _ => True) [] L) : firstOpened L = none := by
  cases hfo : firstOpened L with
  | none => rfl
  | some i₁ =>
    exfalso
    obtain ⟨r, hr, hri⟩ := List.exists_of_findSome?_eq_some hfo
    obtain ⟨A₁, C, he⟩ := List.append_of_mem hr
    have ho : ∃ s n, r.frame = .opening i₁ s n := by
      try dsimp only at hri
      cases hfr : r.frame <;> rw [hfr] at hri <;> first | (cases hri; exact ⟨_, _, rfl⟩) | cases hri
    obtain ⟨s₁, n₁, ho⟩ := ho
    have hSL : LogServed L := fun L₁ r' L₂ i s n he' hf =>
      hS L₁ r' (L₂ ++ L'') i s n (by rw [hlog, he']; simp) hf
    obtain ⟨L₁, r', L₂, he', hl⟩ := link_picks tables hSL _ A₁ (Nat.lt_succ_self _) r C i₁ s₁ n₁ he ho
    exact hNP j₀ trivial L₁ r' L₂ he' (hj₀ _ _ hl)

/-- **A trace off `B` is one off `B'`** wherever, at every position `Q` holds of, each stop the trace answers there is
off `B'`, and `Q` holds of each answered stop's next position. -/
theorem RoundsReach.weaken {B' : RoundHold} {β : Type} {fin : LiveOutJ J mPts → β} {rootB : Array ByteArray}
    {pts : Fin 2 → Fin mPts → 𝔽} {o : β × CoinTape g σ} (Q : Pos g σ Claims.Msg (Fin R) → Prop)
    (hstep : ∀ q (φ : CoinTape q.toStop.g' q.toStop.s' → CoinTape q.toStop.g' q.toStop.s'), Q q →
      Q (q.toStop.rewrite φ).consume)
    (hB : ∀ q (s : SubmitJ J) (c : Fin s.n → 𝔽) (N : ℕ) (rs : Fin J → Fin 2 → List Flock.Round), Q q →
      D.sub J q.toStop.modes = some s → ¬ B q.toStop.g' J s c →
      RoundsReach D M dg J B fin rootB pts N rs (q.toStop.rewrite (M.round q.toStop J s c)).consume o →
      ¬ B' q.toStop.g' J s c)
    {N : ℕ} {rs : Fin J → Fin 2 → List Flock.Round} {p : Pos g σ Claims.Msg (Fin R)} (hp : Q p)
    (hr : RoundsReach D M dg J B fin rootB pts N rs p o) : RoundsReach D M dg J B' fin rootB pts N rs p o := by
  revert hB hp
  induction hr with
  | zero rs p => exact fun _ _ => .zero rs p
  | stop N rs p h => exact fun _ _ => .stop N rs p h
  | round N rs p s c o h hB₀ ho ih =>
    exact fun hB hp => .round N rs p s c o h (hB p s c N _ hp h hB₀ ho) (ih hB (hstep p _ hp))

end Frames

/-! ## The counts at the stops -/

/-- **An answered stop's frame asks for its message's count**, at the readings at a layout `L` whose round `j` is
round `k` of the first table's rep `r` stream: where the stop `q`, keeping the invariant at the rounds' cuts, fixes
round `j`'s cut at the most likely message `w`, the rest of the trace runs from it off every miss, and the device's log
has the opening frame `.opening i s n` after the log `m` at which the cuts the stop fixes are at the bytes that frame
answers and which opened that stream `k` times, under `stop_round`'s premises on v2's read and the record, `n` is
`tailCount w`: v2 read the frame off the line opening round `k` of that stream (`readLog_round`), whose `n` is the one
the record's tree shows opened (`opening_count`), which `sessionTreeReason` holds to the record's count
(`Trees.round_counted`, `recordCount_decode`), `tailCount` of the round's message (`rep_tailCount_accepted`), which is
`w` (`RoundsReach.stop_modes_reveal`). -/
theorem stop_count {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {A : Live}
    {tables : Array String} {L : Fin A.rounds → Slot} {record : Exchange → Option Claims.Msg}
    {recordDigest : Claims.Msg} {rd : Reading A.rounds} (hrd : rd = readingAt tables L record recordDigest)
    {σ : Strategy (V.game θ F c)} {D : Decoders A.rounds A.Reg} {M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP}
    {B : RoundHold} {J N : ℕ} {β : Type} {fin : LiveOutJ J A.mP → β} {rootB : Array ByteArray}
    {pts : Fin 2 → Fin A.mP → 𝔽} {rs : Fin J → Fin 2 → List Flock.Round}
    {q : Pos (V.game θ F c) σ Claims.Msg (Fin A.rounds)} (hq : q.Inv (roundCuts V θ F c rd))
    {φ : CoinTape q.toStop.g' q.toStop.s' → CoinTape q.toStop.g' q.toStop.s'} {o : β × CoinTape (V.game θ F c) σ}
    (hr : RoundsReach D M A.dg J B fin rootB pts N rs (q.toStop.rewrite φ).consume o)
    (hn : NoMiss V θ F c rd σ o.2) {j : Fin A.rounds} {r : Fin 2} {k : ℕ} (hLj : L j = .rep 0 r k)
    (hj : (q.toStop.Ks j).isHere = true) {w : Claims.Msg} (hw : q.toStop.modes j = some w)
    {m L' : List Release} {i s n : ℕ} {b : ByteArray} {bits : List Bool}
    (hlog : (CoinTape.run (V.game θ F c) σ o.2).log = m ++ ⟨.opening i s n, b, bits⟩ :: L')
    (hcut : CutsAt rd.reveal q.toStop m (CoinTape.run (V.game θ F c) σ o.2).log) (hk : openedOf m i s = k)
    (hname : ((helloSpecs m)[i]?.bind fun sp => sp.streams[s]?) = tables[0]?.map (repStreamName · r.val))
    {label : String} {after ξ : ByteArray} {lines : List String} {lr : List Release} {trees : Array LoggedTree}
    {Jr : Json}
    (hF : List.Forall₂ Binding.Read.Of (lr.map Binding.Read.ofV2) (CoinTape.run (V.game θ F c) σ o.2).log)
    (hread : readLog label after [Jr] ξ lines = some lr) (hct : coinTreesOf lines = .ok trees)
    (h1 : trees.size = 1) {tabsB : Array (String × Option ℕ)}
    (hJt : sessionTreeReason trees tabsB Jr = none) {text : String} {spec : Flock.SessionSpec}
    {tabs : Array Flock.Setup} {proofs : Array ByteArray} {sess : Flock.RunRecord}
    (hrec : sentRecord (CoinTape.run (V.game θ F c) σ o.2).sent = some text)
    (hJ : Flock.JsonParse.parse text = .ok Jr) (hd : Flock.Record.decode spec text = .ok sess)
    (hv : Flock.Zk.verify spec tabs text proofs = .ok ()) (hret : spec.retainRounds = true)
    (h512 : spec.sha512Digests = true) (htab : tables = spec.tables)
    (h0 : 0 < spec.tables.size) (hO : OpenedRecorded Jr (CoinTape.run (V.game θ F c) σ o.2).log)
    (hsome : ∀ name, ∀ k < recordRounds Jr name, (recordRound Jr name k).isSome)
    (hans : ∀ {pre post : List Release} {y : Release} {i s n : ℕ} {name : String} {st : Flock.StreamRec},
      lr = pre ++ y :: post → y.frame = .opening i s n →
      ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = some name →
      (name == TREE_SESSION) = false →
      sess.streams.find? (·.name == name) = some st → openedOf pre i s < st.rounds.size →
      y.answers = st.rounds[openedOf pre i s]!.sha) :
    n = tailCount w := by
  subst hrd htab
  have ht0 : spec.tables[0]? = some spec.tables[0]! := by
    rw [Array.getElem?_eq_getElem h0, getElem!_pos spec.tables 0 h0]
  have hname' : ((helloSpecs m)[i]?.bind fun sp => sp.streams[s]?) =
      some (repStreamName spec.tables[0]! r.val) :=
    hname.trans (congrArg (Option.map (repStreamName · r.val)) ht0)
  have hnT := repStreamName_ne spec.tables[0]! r.val
  obtain ⟨h0t, sch, tPad, cap, -, hrep⟩ := verifyRep_of_accepted hv hd r
  obtain ⟨st, _, _, _, hst, -⟩ := FlockSoundness.Discharge.ZkExec.zk_verifyRep_ok _ _ _ _ _ _ _ _ _ hrep
  have hst' : sess.streams.find? (·.name == repStreamName spec.tables[0]! r.val) = some st := hst
  have hrr : recordRounds Jr (repStreamName spec.tables[0]! r.val) = st.rounds.size :=
    recordRounds_decode hd hJ hnT hst'
  have hkR : k < st.rounds.size := by
    have h := hO m _ L' hlog i s n rfl _ hname'
    rwa [hk, hrr] at h
  rw [hlog] at hF
  obtain ⟨pre, y, post, hlr, hpre, hy, -⟩ := forall₂_split (List.forall₂_map_left_iff.1 hF)
  have hy' : (Binding.Read.ofV2 y).Of ⟨.opening i s n, b, bits⟩ := hy
  have hyf : y.frame = .opening i s n := hy'.1
  have hya : y.answers = b := hy'.2.1
  have hpre' : List.Forall₂ (fun x y => x.frame = y.frame ∧ x.answers = y.answers) pre m :=
    hpre.imp fun _ _ h => ⟨h.1, h.2.1⟩
  have hHe : helloSpecs pre = helloSpecs m := helloSpecs_congr hpre'
  have hOp : openedOf pre i s = k := (openedOf_congr hpre').trans hk
  obtain ⟨l, ρ, name, k', hlm, hro, hkey, hnm, hk'⟩ := readLog_round hct hread hlr hyf
  rw [hHe, hname'] at hnm
  obtain rfl : repStreamName spec.tables[0]! r.val = name := Option.some.inj hnm
  rw [hOp] at hk'
  rw [hk'] at hro
  obtain ⟨b', hb'⟩ := Option.isSome_iff_exists.1
    (hsome (repStreamName spec.tables[0]! r.val) k (by rw [hrr]; exact hkR))
  obtain ⟨t₁, ht₁, -, o₁, ho₁, hon₁, -⟩ := Trees.round_counted hJt hb' (recordCount_decode hd hJ hnT hst' hkR)
  obtain rfl := root_of_keyAt h1 ht₁ hkey
  obtain ⟨hor, -, dl, hev, hnl, -⟩ := hro
  have hcnt := opening_count hct ht₁ ho₁ hlm hor hev
  have hnn := hnl i s n hyf
  obtain ⟨cs', -, -, hsha⟩ := recordRound_decode hd hJ hnT hst' hkR
  obtain ⟨bb, hbb, hmsg⟩ := roundMsg_decode hd hret hst' hkR
  have hdg512 : spec.dg bb = Flock.Sha512.hash bb := by simp [Flock.SessionSpec.dg, h512]
  have hyans : y.answers = st.rounds[k].sha := by
    have h := hans hlr hyf (by rw [hHe]; exact hname') hnT hst' (by rw [hOp]; exact hkR)
    rwa [hOp, getElem!_pos st.rounds k hkR] at h
  have hrev : (readingAt spec.tables L record recordDigest).reveal j (CoinTape.run (V.game θ F c) σ o.2) =
      some bb.data.toList := by
    change slotReveal spec.tables (L j) _ = _
    rw [hLj]
    simp only [slotReveal, roundIn, ht0, Option.bind_some, hrec, hmsg]
  have hmode := RoundsReach.stop_modes_reveal hq hr hn hcut hlog hj hrev (by
    change sha512Msg bb.data.toList = b.data.toList
    rw [← hya, hyans, hsha bb hbb, hdg512]
    all_goals rfl)
  have hwb : w = bb.data.toList := Option.some.inj (hw.symm.trans hmode)
  subst hwb
  rw [rep_tailCount_accepted hv hd hret r hst' hkR hmsg, ← hon₁]
  exact Option.some.inj (hnn.symm.trans hcnt)

/-- **The link's frame asks for both points' coordinates**: where `A` is the session v2 sets up on the parameters, the
sampled verdict accepts, the log v2's check read is the device's, and the device's first opening `r`, after releases
`L₀` that open none, opens the stream `TREE_SESSION` of the tree it opens with the frame `.opening i s n`, `n` is
`A.mP + A.mP`: v2 read the frame off the line opening that stream's round 0 (`readLog_round`), whose `n` is the one
the record's tree shows opened (`opening_count`), which `sessionTreeReason` holds to a coin a 16 bytes of the record's
link coins (`Trees.session_counted`), both points' coordinates (`link_count`), `A.mP` each at the record's sample
(`SetsUp.mPts_eq`). -/
theorem link_frameCount {V : Vocab d α} {k : Verity.Registration.Composite} {θ : ByteArray} {F : Array ByteArray}
    {A : Live} {x : Exchange} (hA : SetsUp V θ F A) (hθ : x.params = θ) (hv : V.Verdict k .sampled x)
    (hl : Binding.Logged Binding.readsV2 x) {L₀ L : List Release} {r : Release} {i s n : ℕ}
    (hlog : x.log = L₀ ++ r :: L) (hr : r.frame = .opening i s n)
    (hL₀ : ∀ q ∈ L₀, ∀ i' s' n', q.frame ≠ .opening i' s' n')
    (hO : ((helloSpecs L₀)[i]?.bind fun sp => sp.streams[s]?) = some TREE_SESSION) {text : String}
    {spec : Flock.SessionSpec} {sts : Array Flock.Stmt} {sess : Flock.RunRecord}
    (hrec : sentRecord x.sent = some text)
    (hs : Exec.setupOfP (ZkExec.zkInputs A.I) A.own text = .ok (spec, sts))
    (hd : Flock.Record.decode spec text = .ok sess) {S : Finset (Fin A.n)}
    (hdr : Exec.recordDraw text = some (A.dj S)) : n = A.mP + A.mP := by
  obtain ⟨_, _, _, lines, lr, trees, Jr, _, hV, hread, hct, h1, hJt, hJ⟩ := verdict_read hA hθ hv hl hrec
  rw [hlog] at hV
  obtain ⟨pre, y, post, hlr, hpre, hy, -⟩ := forall₂_split (List.forall₂_map_left_iff.1 hV)
  have hyf : y.frame = .opening i s n := hy.1.trans hr
  have hfr : List.Forall₂ (fun a b => a.frame = b.frame) pre L₀ := hpre.imp fun _ _ h => h.1
  obtain ⟨l, ρ, name, k', hlm, hro, hkey, hnm, hk'⟩ := readLog_round hct hread hlr hyf
  rw [helloSpecs_frames hfr, hO] at hnm
  obtain rfl : name = TREE_SESSION := (Option.some.inj hnm).symm
  rw [openedOf_eq_zero (noOpening_frames hfr hL₀) i s] at hk'
  subst hk'
  obtain ⟨cs, hcs⟩ := LinkTrees.link_some hJt
  obtain ⟨t, ht, -, hlink, -⟩ := Trees.session_counted hJt
  obtain ⟨o, ho, hon, -⟩ := hlink TREE_SESSION 0 cs (by simp) hcs
  obtain rfl := root_of_keyAt h1 ht hkey
  obtain ⟨hor, -, dl, hev, hnl, -⟩ := hro
  have hcnt := opening_count hct ht ho hlm hor hev
  have hnn := hnl i s n hyf
  rw [Option.some.inj (hnn.symm.trans hcnt), hon, link_count hd hJ (hA.setup_one hs).2.2.2.1 hcs,
    SetsUp.mPts_eq hA hs hdr]

/-! ## `CountsReach` at the session's layout -/

-- `decodersAt`'s roots search `List.finRange A.rounds`, which unification would otherwise unfold at `A.rounds`'s value
attribute [local irreducible] Live.rounds Live.N in
/-- **The count match at a witness of the joint law's support, at the session's layout**: at the readings and the
decoders at `Live.layout` and `Live.tables`, and at the sample-coupled coin map under `fieldEnc`, an outcome off the
empty classes at their counts, beside a play that the sampled verdict accepts, that misses nowhere and whose log v2's
check read is the device's, is off the empty classes, under `ReachesOff.movesOf`'s premises. The first stop's frame is
the log's first opening (`RoundsReach.stop_frame`, `firstOpened_noPick`), of the session stream (`linkFirst_verdict`),
which asks for `A.mP + A.mP` coins (`link_frameCount`, at the record's sample, `recordDraw_firstStop`); each answered
stop's frame is round `k` of the first table's rep stream at its fixed round's slot (`stop_slot`), which asks for its
message's count (`stop_count`). So no stop off the first events is in the second (`RoundsReach.weaken`). -/
theorem ReachesOff.counts {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {k : Verity.Registration.Composite} {A : Live} (hA : SetsUp V θ F A) (hDg : DigestsAt V θ)
    (reg : Option Claims.Msg → A.Reg)
    (record : Exchange → Option Claims.Msg) (recordDigest : Claims.Msg) {rd : Reading A.rounds}
    (hrd : rd = readingAt A.tables A.layout record recordDigest) {σ : Strategy (V.game θ F c)}
    {D : Decoders A.rounds A.Reg} (hD : D = decodersAt reg A.layout) {label : String} {after : ByteArray} {m : ℕ}
    {κ : A.Ω → Fin m → Fin 512 → Bool} (hm : _root_.Spot.Challenger.MAX_COINS ≤ m)
    (hsrc : ∀ ω, _root_.Properties.Spot.bytesOf (A.src ω 0) =
      _root_.Properties.Spot.bytesOf (coinBytes m 64 (κ ω) 0))
    {M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP} (hM : M = sampleMap fieldEnc V θ F c A rd σ label after κ)
    {o : A.Out × CoinTape (V.game θ F c) σ}
    (h : ReachesOff V θ F c A rd σ D M (linkEmptyAt fieldEnc (finish (V.signer θ) θ F c σ.1))
      (roundEmptyAt fieldEnc (finish (V.signer θ) θ F c σ.1)) o)
    (hv : V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ o.2)) (hn : NoMiss V θ F c rd σ o.2)
    (hl : Binding.Logged Binding.readsV2 (CoinTape.run (V.game θ F c) σ o.2))
    (hN : SampleNamed k (CoinTape.run (V.game θ F c) σ o.2).sent label after)
    (hO : ∀ text Jr, sentRecord (CoinTape.run (V.game θ F c) σ o.2).sent = some text →
      Flock.JsonParse.parse text = .ok Jr → OpenedRecorded Jr (CoinTape.run (V.game θ F c) σ o.2).log) :
    ReachesOff V θ F c A rd σ D M (linkEmpty fieldEnc (finish (V.signer θ) θ F c σ.1))
      (roundEmpty fieldEnc (finish (V.signer θ) θ F c σ.1)) o := by
  subst hrd hD hM
  obtain ⟨ω, pts, hL0, hr⟩ := h
  have hθ := (play_ends V θ F c σ o.2).1
  have hfirst := linkFirst_verdict hA hθ hv hl
  have hP := freshBefore_verdict hA hθ hv hl hN
  have h0r := A.rounds_pos
  obtain ⟨text₀, proofs₀, spec₀, sts₀, hrec₀, -, hs₀, spec', sts', hs', -, hver⟩ :=
    accepts_sent hA hθ hv (filesOn_play hA hDg hv)
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj (hs₀.symm.trans hs'))
  obtain ⟨sess₀, hd₀⟩ := zkVerify_decode hver
  obtain ⟨label', after', ξ, lines, lr, trees, Jr, tabsB, hV, hread, hct, h1, hJt, hJ⟩ :=
    verdict_read hA hθ hv hl hrec₀
  obtain ⟨htab, -, hret, -, -⟩ := hA.setup_one hs₀
  have h512 := setup_sha512Digests A.hTags.1 (statement_hm96Rows hA.tags) hA.count hs₀
  have h0 : 0 < spec₀.tables.size := by
    rw [htab]
    exact Nat.one_pos
  have hPk := readingAt_picks A.tables A.layout record recordDigest
  have hPS := readingAt_served A.tables A.layout record recordDigest
  have hdr := recordDraw_firstStop fieldEnc hA hA.popTied (drawReadAll_coinsRead k) (drawHeld k) hPk hm hsrc ω
    (firstStop_playReaches V θ F c A _ σ _ ω hr) hv hl hN hP hrec₀
  have hjl := A.layout_link h0r
  have hj₀ : ∀ pre fr, slotRound A.tables .link pre fr →
      (readingAt A.tables A.layout record recordDigest).round ⟨0, h0r⟩ pre fr := fun pre fr h => by
    show slotRound A.tables (A.layout ⟨0, h0r⟩) pre fr
    rw [hjl]
    exact h
  have hK := firstStop_fixes V θ F c A (readingAt A.tables A.layout record recordDigest) σ
    (sampleMap fieldEnc V θ F c A (readingAt A.tables A.layout record recordDigest) σ label after κ)
    hPk A.tables ⟨0, h0r⟩ hj₀ ω _ hr (verdict_opens hA hθ hv hl hrec₀)
  have hi₀ := (Pos.toStop_spec (Pos.inv_root (roundCuts V θ F c (readingAt A.tables A.layout record recordDigest))
    ((sampleMap fieldEnc V θ F c A (readingAt A.tables A.layout record recordDigest) σ label after κ).start ω))).2.2
  obtain ⟨L₀, i, s, n, b, bits, L', hNP, -, -, hlog, -, hcnt⟩ :=
    RoundsReach.stop_frame (pre₀ := []) (f := finish (V.signer θ) θ F c σ.1) hPS hi₀ (fun _ => rfl)
      (stopOf_firstStop V θ F c A _ σ _ ω hPk) hK hr
  rw [List.nil_append] at hlog
  have h₀ := firstOpened_noPick A.tables hj₀ (logServed_run V θ F c σ o.2) hlog hNP
  have hno := noOpening_of_firstOpened h₀
  have hnL := link_frameCount hA hθ hv hl hlog rfl hno (hfirst L₀ _ L' hlog i s n rfl hno) hrec₀ hs₀ hd₀ hdr
  refine ⟨ω, pts, fun hE => ?_, ?_⟩
  · obtain ⟨o', ho', hne⟩ := hE
    exact hL0 ⟨o', ho', (hcnt o').trans hnL, hne⟩
  refine RoundsReach.weaken (fun q => q.Inv (roundCuts V θ F c (readingAt A.tables A.layout record recordDigest)) ∧
      StopsAt (readingAt A.tables A.layout record recordDigest).round
        (readingAt A.tables A.layout record recordDigest).reveal (V.signer θ) θ F c σ.1 q.toStop)
    (fun q φ hq => ⟨Pos.inv_consume (Pos.inv_rewrite (Pos.toStop_spec hq.1).2.2 φ), StopsAt.next hPk hq.2 φ⟩)
    ?_ ⟨Pos.inv_consume (Pos.inv_rewrite hi₀ _), StopsAt.next hPk (stopsAt_firstStop V θ F c A _ σ _ ω hPk) _⟩ hr
  intro q sub x N' rs' hq hs hBq ho hE
  obtain ⟨o', ho', hne⟩ := hE
  obtain ⟨pre₀, f, a, hf, hSO⟩ := hq.2
  have hKq : ¬ ∀ j, q.toStop.Ks j = .skip := fun hK' => by
    have hm : q.toStop.modes = fun _ => none := funext fun j =>
      (congrArg (fun K => DigestCut.modeHere K q.toStop.s') (hK' j)).trans rfl
    rw [hm, sub_eq_none reg A.layout _ (fun _ _ => rfl)] at hs
    cases hs
  obtain ⟨Lq, i', s', n', b', bits', L'', -, ⟨j, hjK, hj⟩, hall, hlog', hcut, hcnt'⟩ :=
    RoundsReach.stop_frame hPS (Pos.toStop_spec hq.1).2.2 hf hSO hKq ho
  obtain ⟨t, ht', r, kk, w, hLj, hw, rfl⟩ :=
    stop_slot reg A.layout_injective A.tables_nodup record recordDigest hs hall hj
  have ht1 : t < 1 := lt_of_lt_of_eq ht' (hA.J_eq _ _)
  obtain rfl : t = 0 := by omega
  have hP' : slotRound A.tables (A.layout j) (pre₀ ++ Lq) (.opening i' s' n') := hj
  rw [hLj] at hP'
  obtain ⟨i'', s'', n'', hfr, -, -, hname, hk⟩ := hP'
  obtain ⟨rfl, rfl, rfl⟩ := Frame.opening.inj hfr
  refine hBq ⟨o', ho', ?_, hne⟩
  rw [hcnt' o']
  exact stop_count rfl hq.1 ho hn hLj hjK hw hlog' hcut hk hname hV hread hct h1 hJt hrec₀ hJ hd₀ hver hret h512
    htab.symm h0 (hO _ _ hrec₀ hJ) (recordRound_isSome hJt)
    (fun hlr hy hO' _ hst hk => rep_answers (hlr ▸ hread) hct h1 hJt hy hO' hd₀ hJ h512 hst hk)

/-- **`CountsReach` at the session's layout**: `ReachesOff.counts` at every outcome, at the cut that v2's check read
the device's log and that v2 samples at `after` (`SampledDigest`), with what it takes of each accepted play the cut holds
of stated for every tape, as `movesReach_layout` takes it. -/
theorem countsReach_layout {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {k : Verity.Registration.Composite} {A : Live} (hA : SetsUp V θ F A) (hDg : DigestsAt V θ)
    (reg : Option Claims.Msg → A.Reg)
    (record : Exchange → Option Claims.Msg) (recordDigest : Claims.Msg) {rd : Reading A.rounds}
    (hrd : rd = readingAt A.tables A.layout record recordDigest) {σ : Strategy (V.game θ F c)}
    {D : Decoders A.rounds A.Reg} (hD : D = decodersAt reg A.layout) {label : String} {after : ByteArray} {m : ℕ}
    {κ : A.Ω → Fin m → Fin 512 → Bool} (hm : _root_.Spot.Challenger.MAX_COINS ≤ m)
    (hsrc : ∀ ω, _root_.Properties.Spot.bytesOf (A.src ω 0) =
      _root_.Properties.Spot.bytesOf (coinBytes m 64 (κ ω) 0))
    {M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP} (hM : M = sampleMap fieldEnc V θ F c A rd σ label after κ)
    (hN : ∀ x : CoinTape (V.game θ F c) σ, V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) →
      Binding.Logged Binding.readsV2 (CoinTape.run (V.game θ F c) σ x) →
      SampledDigest k (CoinTape.run (V.game θ F c) σ x).sent after →
        SampleNamed k (CoinTape.run (V.game θ F c) σ x).sent label after)
    (hO : ∀ x : CoinTape (V.game θ F c) σ, V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) →
      Binding.Logged Binding.readsV2 (CoinTape.run (V.game θ F c) σ x) →
      ∀ text Jr, sentRecord (CoinTape.run (V.game θ F c) σ x).sent = some text →
        Flock.JsonParse.parse text = .ok Jr → OpenedRecorded Jr (CoinTape.run (V.game θ F c) σ x).log) :
    CountsReach fieldEnc V θ F c k (fun o => Binding.Logged Binding.readsV2 o ∧ SampledDigest k o.sent after) A rd σ
      D M :=
  fun _ h ⟨hv, hn, hl, hd⟩ => h.counts hA hDg reg record recordDigest hrd hD hm hsrc hM hv hn hl
    (hN _ hv hl hd) (hO _ hv hl)

/-- **`CountsReach` at the deployment**: `countsReach_layout` at `movesReachAt`'s readings, decoders, family, digest
and coins, its premises on the sample closed as `movesReachAt` closes them. -/
theorem countsReachAt {V : Vocab d α} {k : Verity.Registration.Composite} {θ : ByteArray} {F : Array ByteArray}
    {A : Live} (reg : Option Claims.Msg → A.Reg) (record : ByteArray → Exchange → Option Claims.Msg)
    (recordDigest : ByteArray → Claims.Msg) (hA : SetsUp V θ F A) (hDg : DigestsAt V θ)
    (hL : A.Lb 0 = _root_.Spot.Challenger.MAX_COINS * 64)
    (hO : ∀ c (σ : Strategy (V.game θ F c)) (x : CoinTape (V.game θ F c) σ),
      V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) →
      Binding.Logged Binding.readsV2 (CoinTape.run (V.game θ F c) σ x) →
      ∀ text Jr, sentRecord (CoinTape.run (V.game θ F c) σ x).sent = some text →
        Flock.JsonParse.parse text = .ok Jr → OpenedRecorded Jr (CoinTape.run (V.game θ F c) σ x).log)
    (c : ByteArray) (σ : Strategy (V.game θ F c)) :
    CountsReach fieldEnc V θ F c k
      (fun o => Binding.Logged Binding.readsV2 o ∧ SampledDigest k o.sent (afterAt V k θ F c σ)) A
      (readingAt A.tables A.layout (record c) (recordDigest c)) σ (decodersAt reg A.layout)
      (sampleMap fieldEnc V θ F c A (readingAt A.tables A.layout (record c) (recordDigest c)) σ (labelAt θ)
        (afterAt V k θ F c σ) (srcCoinsAt _root_.Spot.Challenger.MAX_COINS A)) :=
  countsReach_layout (label := labelAt θ) (after := afterAt V k θ F c σ)
    (κ := srcCoinsAt _root_.Spot.Challenger.MAX_COINS A) hA hDg reg (record c) (recordDigest c)
    rfl rfl le_rfl (bytesOf_srcCoinsAt hL) rfl (sampleNamed_layout hA) (hO c σ)

/-- **The coupling at the deployment, off the empty classes at their counts**: `couplingTarget_assembled_at` at the
deployed device's history and `movesReachAt`'s readings, decoders, family, digest and coins, with its `hM` by
`movesReachAt` and the count match (`countsReachAt`, `movesReach_at`). It takes `couplingTargetAt`'s premises but its
bound on the stops' term. -/
theorem couplingTargetAt_counts {V : Vocab d α} {k : Verity.Registration.Composite} (K : Constructions V)
    (P : Device.Deployment) {θ : ByteArray} {F : Array ByteArray} {A : Live}
    (reg : (A : Live) → Option Claims.Msg → A.Reg)
    (record : Live → ByteArray → Array ByteArray → ByteArray → Exchange → Option Claims.Msg)
    (recordDigest : Live → ByteArray → Array ByteArray → ByteArray → Claims.Msg)
    (hD : V.signer θ = P.signer V.families θ)
    (hA : SetsUp V θ F A) (hDg : DigestsAt V θ) (hL : A.Lb 0 = _root_.Spot.Challenger.MAX_COINS * 64)
    (hO : ∀ c (σ : Strategy (V.game θ F c)) (x : CoinTape (V.game θ F c) σ),
      V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) →
      Binding.Logged Binding.readsV2 (CoinTape.run (V.game θ F c) σ x) →
      ∀ text Jr, sentRecord (CoinTape.run (V.game θ F c) σ x).sent = some text →
        Flock.JsonParse.parse text = .ok Jr → OpenedRecorded Jr (CoinTape.run (V.game θ F c) σ x).log) :
    CouplingTarget V k (K.assembled fieldEnc (Device.dev (P.setupOf V.families)) k
      (fun A θ F c => readingAt A.tables A.layout (record A θ F c) (recordDigest A θ F c))
      (fun A _ _ _ _ => decodersAt (reg A) A.layout) (fun θ _ _ _ => labelAt θ) (afterAt V k)
      (srcCoinsAt _root_.Spot.Challenger.MAX_COINS)) A θ F :=
  couplingTarget_assembled_at fieldEnc V k K (Device.dev (P.setupOf V.families))
    (fun A θ F c => readingAt A.tables A.layout (record A θ F c) (recordDigest A θ F c))
    (fun A _ _ _ _ => decodersAt (reg A) A.layout) (fun θ _ _ _ => labelAt θ)
    (srcCoinsAt _root_.Spot.Challenger.MAX_COINS) A θ F (fun o _ => Device.printsLog_dev _ o)
    (fun c => Device.signsPlay_dev V P θ F c hD) (avg_srcCoinsAt hL) hA hDg (writesView_wrZP A rfl)
    (fun c σ => movesReach_at (movesReachAt (reg A) (record A θ F) (recordDigest A θ F) hA hDg hL hO c σ)
      (countsReachAt (reg A) (record A θ F) (recordDigest A θ F) hA hDg hL hO c σ))

end Verity.Sampled
