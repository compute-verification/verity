import Proofs.Flock.Claims.Recursion.RecRecord
import Proofs.Flock.Claims.Recursion.DeployHonest
import Proofs.Flock.C3.VStarList.Accepts
import Proofs.Flock.C3.PubRows.Check
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Shared
import Proofs.Flock.Soundness.Discharge.Layout.SharedRows
import Proofs.Flock.Soundness.Discharge.Layout.QSort
import Proofs.Flock.Claims.Recursion.SlotsCheck

/-!
# What a staged part's transcript shows is the record the verifier read in its own format

`PartOpensB` at `cmtTr recRead` asks the commit strings a part holds to be those its V\* statement's transcript shows,
read by `recRead`: the coin server's record in its own format (`Flock.RecRecord.readFull`). Every staged part carries
the coin server's transcript (`insO_staged_link`), and its recursive setup holds it at the verifier's inner shape
(`RecSetup.commits`), which reads it in that same format and holds what it reads there to what its JSON readers read
(`RecSetup.checkReads`, `FlockC3.VStarList.checkReads_ok`). So at a staged part (`staged_reads`) the rounds
`cmtTr recRead` reads are the record's (`Record.msgs`), and the record's link, rounds, tops and `c_k` are the JSON's the
setup pins: Hello's table, `root_B`, the link points, the schedule's rounds, and each rep's salted tops and `c_k`.

This is the transcript's half of `PartOpensB`. The other half is that a part's hm96 slot outputs are the public rows
its statement pins, and those rows are the JSON's (`RecVStar.checkPubRows`, `RecVStar.checkTopRows`). Its first steps
are here: the setup checks every table's rows and registered reads (`RecIns.setup_pubRows`,
`RecIns.setup_registered`), which give each registered port's rows as `fixedRow`'s at the positions its instances read
(`rowsAt`), and a drawn table's slot holds its instance's public `b ‖ c` (`bc_setupOne`), a table row
(`digest_tables`). So port `q`'s hm96 slot at instance `n`, read off the witness (`hmCommit`), is that instance's row
digest (`slotPub_table`). On the record's side, rep `x`'s `kc`-th `c_k` in the JSON is at the commit round `k` its
schedule lists `kc`-th, and at a round with no salted tops it is the record's message `x · R + k + 1`
(`cmt_of_rep`): the rounds open with the two reps' schedules (`scheduleOf_reps`), whose streams are apart (`rep_ne`).
-/

namespace RecDeploy

open Flock
open Claims (Msg Out)
open Lean (Json)

theorem byteArray_toList_loop (b : ByteArray) (i : ℕ) (r : List UInt8) :
    ByteArray.toList.loop b i r = r.reverse ++ b.data.toList.drop i := by
  fun_induction ByteArray.toList.loop b i r with
  | case1 i r h ih =>
    rw [ih, List.reverse_cons, List.append_assoc, List.singleton_append]
    congr 1
    conv_rhs => rw [List.drop_eq_getElem_cons (by simpa using h)]
    simp [ByteArray.get!, h]
  | case2 i r h =>
    rw [List.drop_eq_nil_of_le (by simp; simpa using h), List.append_nil]

theorem byteArray_toList_eq (b : ByteArray) : b.toList = b.data.toList := by
  unfold ByteArray.toList
  rw [byteArray_toList_loop]
  simp

theorem toUTF8_of_fromUTF8? {b : ByteArray} {s : String} (h : String.fromUTF8? b = some s) : s.toUTF8 = b := by
  unfold String.fromUTF8? at h
  split at h
  · cases h; rfl
  · cases h

/-- **The commits the reader reads** are `commitsOf` at the statement's source, binding and inner shape. -/
theorem RecRaw.ins_commits {w : RecRaw} {s : RecIns} (h : w.ins = .ok s) :
    commitsOf w s.src s.bound s.inner = .ok s.commits := by
  unfold RecRaw.ins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; assumption)

theorem commits_parse_error {sh : RecVStar.Inner} {tt : String} {e : String} {r : Except String (Option String)}
    {bound : Flock.V (Option String × Option RecVStar.StackedShape)} {record : String} {cs : RecVStar.InnerCommits}
    (hp : Json.parse tt = .error e) : RecSetup.commits (some sh) (.ok tt) r bound record ≠ .ok cs := by
  intro h
  unfold RecSetup.commits at h
  simp only [bind, Except.bind, pure, Except.pure, hp] at h
  repeat' split at h
  all_goals cases h

theorem commits_text_error {sh : RecVStar.Inner} {e : String} {r : Except String (Option String)}
    {bound : Flock.V (Option String × Option RecVStar.StackedShape)} {record : String} {cs : RecVStar.InnerCommits} :
    RecSetup.commits (some sh) (.error e) r bound record ≠ .ok cs := by
  intro h
  unfold RecSetup.commits at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals cases h

/-- **A transcript's commits at the verifier's inner shape read it in the coin server's own format**: where
`commitsOf` holds the transcript `t`, it is UTF-8 and JSON, the source has a record, and `checkReads` passed. -/
theorem commitsOf_reads {w : RecRaw} {src : Option RecCoins.Source}
    {bound : Flock.V (Option String × Option RecVStar.StackedShape)} {sh : RecVStar.Inner}
    {c : Option RecVStar.InnerCommits} {t : ByteArray} (ht : w.transcript = some t)
    (h : commitsOf w src bound (some sh) = .ok c) :
    ∃ cs tt tr record, c = some cs ∧ String.fromUTF8? t = some tt ∧ Json.parse tt = .ok tr ∧
      src.map (·.record) = some record ∧ RecSetup.checkReads sh tt tr record cs = .ok () := by
  unfold commitsOf at h
  rw [ht] at h
  cases hsrc : src.map (·.record) with
  | none => rw [hsrc] at h; cases h
  | some record =>
    rw [hsrc] at h
    dsimp only at h
    split at h
    · cases h
    cases hX : RecSetup.commits (some sh) (textOf "rec inner transcript" t)
        (w.registration.mapM (textOf "rec inner registration")) bound record with
    | error e => rw [hX] at h; cases h
    | ok cs =>
      rw [hX] at h
      cases h
      cases htx : String.fromUTF8? t with
      | none =>
        rw [textOf, htx] at hX
        exact absurd hX commits_text_error
      | some tt =>
        rw [textOf, htx] at hX
        cases hp : Json.parse tt with
        | error e => exact absurd hX (commits_parse_error hp)
        | ok tr => exact ⟨cs, tt, tr, record, rfl, rfl, hp, rfl, FlockC3.VStarList.reads_of_commits hp hX⟩

theorem ins_of_insO {x : Msg} {i : RecIns} (h : insO x = some i) : ∃ w, decodeRec x = some w ∧ w.ins = .ok i := by
  unfold insO at h
  obtain ⟨w, hw, h⟩ := Option.bind_eq_some_iff.1 h
  split_ifs at h
  · cases h
  exact ⟨w, hw, toOption_eq_some h⟩

/-- **A staged part's transcript, as the verifier read it in its own format**: at a part staged at `x`, the recursive
setup holds the coin server's transcript at the verifier's inner shape `sh` (`i.commits = some cs`), `checkReads`
passed on it, and the rounds `recRead` reads off the V\* statement are the record's (`Record.msgs`), the record
`readFull` reads. So `checkReads_ok` gives that record's link, rounds, tops and `c_k` as the JSON's. -/
theorem staged_reads {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) :
    ∃ (i : RecIns) (sh : RecVStar.Inner) (cs : RecVStar.InnerCommits) (tt : String) (tr : Json) (record : String)
      (r : Flock.RecRecord.Record),
      insO xO = some i ∧ i.inner = some sh ∧ i.commits = some cs ∧
      RecSetup.checkReads sh tt tr record cs = .ok () ∧ Flock.RecRecord.readFull tt.toUTF8.toList = some r ∧
      shownOf recRead x xO = some (r.msgs.map fun mc => (mc.1, mc.2.toArray)) := by
  obtain ⟨ms, i, _spec, _sts, rfl, hsu, -⟩ := hS
  obtain ⟨hins, -⟩ := setupO_some hsu
  obtain ⟨xr, -, hinner, -⟩ := insO_staged hins
  obtain ⟨wr, hd, htr, -, -⟩ := insO_staged_link hins
  obtain ⟨wr', hd', hw⟩ := ins_of_insO hins
  rw [hd] at hd'
  cases hd'
  have hc := RecRaw.ins_commits hw
  rw [hinner] at hc
  obtain ⟨cs, tt, tr, record, hcs, htx, -, -, hR⟩ := commitsOf_reads htr hc
  obtain ⟨_lk, r, _want, -, hr, -⟩ := FlockC3.VStarList.checkReads_ok hR
  refine ⟨i, xr.shape, cs, tt, tr, record, r, hins, hinner, hcs, hR, hr, ?_⟩
  have hb : (st.transcript x ms).data.toList = tt.toUTF8.toList := by
    rw [byteArray_toList_eq, toUTF8_of_fromUTF8? htx]
  simp only [shownOf, hd, htr, Option.bind_some, recRead, hb, hr, Option.map_some]

/-! ## What the setup pins of a part's rows -/

theorem RecIns.setup_pubRows {s : RecIns} {spec : SessionSpec} {sts : Array Stmt} (h : s.setup = .ok (spec, sts)) :
    sts.forM (RecVStar.checkPubRows · s.base.registered s.commits (s.inner.bind (·.sparse))) = .ok () := by
  unfold RecIns.setup at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; assumption)

theorem forIn_ok_of {α : Type} {xs : Array α} {f : α → Flock.V Unit}
    {b : α → PUnit → Flock.V (ForInStep PUnit)} {u : PUnit}
    (hb : ∀ a v r, b a v = .ok r → f a = .ok () ∧ r = .yield PUnit.unit) (h : forIn xs PUnit.unit b = .ok u) :
    ∀ a ∈ xs, f a = .ok () := by
  rcases xs with ⟨l⟩
  simp only [List.forIn_toArray] at h
  intro a ha
  simp only [List.mem_toArray] at ha
  induction l generalizing u with
  | nil => cases ha
  | cons c l ih =>
    simp only [List.forIn_cons, bind, Except.bind] at h
    cases hc : b c PUnit.unit with
    | error e => rw [hc] at h; cases h
    | ok r =>
      rw [hc] at h
      rcases List.mem_cons.1 ha with rfl | ha
      · exact (hb _ _ _ hc).1
      · rw [(hb _ _ _ hc).2] at h
        exact ih h ha

/-- The recursive setup checks each table's registered reads (`checkRegistered`). -/
theorem RecIns.setup_registered {s : RecIns} {spec : SessionSpec} {sts : Array Stmt}
    (h : s.setup = .ok (spec, sts)) : ∀ st ∈ sts, checkRegistered st s.base.registered = .ok () := by
  unfold RecIns.setup at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  all_goals
    obtain ⟨-, h2⟩ := Prod.mk.inj (Except.ok.inj h)
    subst h2
    refine forIn_ok_of ?_ (by assumption)
    intro a v r hr
    try dsimp only at hr
    split at hr
    · cases hr
    · rename_i u hu
      split at hr
      · cases hr
      · cases hr
        exact ⟨hu, rfl⟩

/-- A shared-row file's `b ‖ c` of instance `i`'s row `p` is its table row at the instance's ref. -/
theorem digest_tables {pub : Public} {t : Tables} (ht : pub.tables = some t) (i p np : ℕ) :
    pub.digest i p np = Registered.tableRow pub t p (Registered.ref t np p i) := by
  unfold Public.digest Registered.tableRow Registered.ref
  rw [ht]

/-- **A registered port's rows at a checked table**: where the table passed `checkPubRows` and `checkRegistered`
against the verifier's own reads `own`, a port `reg` of `own` registering a value `fixedRow` knows has a table `t`, a
circuit port `p` and an entry's positions, with every instance `i`'s ref a row of that table read at the position its
unit reads, and every row the value's `fixedRow` row at its position. -/
theorem rowsAt {st : Stmt} {own : Array Registered.Port} {cs : RecVStar.InnerCommits}
    {sparse : Option RecSparse.Shape} (hP : RecVStar.checkPubRows st (some own) (some cs) sparse = .ok ())
    (hR : checkRegistered st (some own) = .ok ()) {reg : Registered.Port} (hmem : reg ∈ own)
    (hfix : (RecVStar.fixedRow { reps := #[] } reg.value 0).isSome = true) :
    ∃ t idx p positions, st.pub.tables = some t ∧ Registered.unitIndices st.pub = .ok idx ∧
      (List.range st.c.ports.size).find? (fun p => st.c.ports[p]!.name == reg.name) = some p ∧
      positions.size = t.counts.getD p 0 ∧
      (∀ i < st.pub.n, Registered.ref t st.c.ports.size p i < positions.size ∧
        reg.reads[idx.getD i 0]? = some (some (positions.getD (Registered.ref t st.c.ports.size p i) 0))) ∧
      ∀ r < positions.size, ∃ w c, RecVStar.fixedRow cs reg.value (positions.getD r 0)
        ((RecSparse.params st.c.unit.name).or sparse) = some (.ok (w, c)) ∧ Registered.tableRow st.pub t p r = c := by
  obtain ⟨t, ht, p, positions, paths, hp, hent, hrows⟩ := FlockC3.PubRows.checkPubRows_rows hP hmem hfix
  obtain ⟨t', idx, ht', hidx, hports⟩ :=
    FlockSoundness.Discharge.ZkReg.check_tables (FlockSoundness.Discharge.ZkReg.checkRegistered_ok hR)
  rw [ht] at ht'
  cases ht'
  obtain ⟨p', positions', paths', hp', hent', hall⟩ := FlockSoundness.Discharge.ZkReg.checkPort_at (hports _ hmem)
  obtain ⟨p'', positions'', paths'', hp'', hent'', hsize⟩ := FlockC3.PubRows.checkPort_size (hports _ hmem)
  rw [hp] at hp' hp''
  cases hp'
  cases hp''
  rw [hent] at hent' hent''
  cases hent'
  cases hent''
  refine ⟨t, idx, p, positions, ht, hidx, hp, hsize, fun i hi => ?_, hrows⟩
  obtain ⟨hlt, hreads, -⟩ := hall i hi
  exact ⟨hsize ▸ hlt, hreads⟩

theorem regionsOf_mem (st : Stmt) :
    ∀ r ∈ st.regions.toList,
      FlockSoundness.Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ FlockSoundness.Refine.regionsOf st :=
  fun _ hr => List.mem_map_of_mem hr

open FlockSoundness.Discharge in
/-- **A drawn table's hm96 slot holds its instance's public `b ‖ c`**: at a statement `setupOne` sets up at a draw, in
either branch, a witness satisfying the table's model (at the regions the soundness proof reads) holds at instance
`o · g + g`'s port-`p` slot the statement's `b ‖ c` of that instance's row `p` (`Layout.bc_drawnS`,
`Hidden.bc_drawnW`, with `Hidden.drawn_digestW`). -/
theorem bc_setupOne {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {d : Json} {partition : Option Json} {program : Option ByteArray} {st : Stmt}
    (h : HiddenExec.setupOne tags cf pf coins tables (some d) partition program = .ok st)
    (L : FlockSoundness.Layout st) {z : FlockSoundness.Model.Witness (L.model (FlockSoundness.Refine.regionsOf st)).m}
    (hz : (L.model (FlockSoundness.Refine.regionsOf st)).Satisfies z)
    (o : Fin (2 ^ (L.model (FlockSoundness.Refine.regionsOf st)).nbl)) {g p : ℕ} (hg : g < st.c.g)
    (hg2 : g < 2 ^ st.c.g.log2) (hp : p < st.c.ports.size) (hr : o.val * st.c.g + g < st.pub.n) :
    FlockSoundness.Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) =>
        (L.model (FlockSoundness.Refine.regionsOf st)).addr o
          (Layout.inBlockOr st.c.kLog (Layout.bcAddr st.c g p j.val t.val))) =
      (st.pub.digest (o.val * st.c.g + g) p st.c.ports.size).data.toList := by
  unfold HiddenExec.setupOne at h
  split at h
  · have s := Hidden.setupHidden_spec' h
    obtain ⟨pop, hpop, -, hdr⟩ := s.pub
    have hdrawn : HmRow.drawn { st.c with outPorts := #[] } pop d = .ok st.pub := hdr d rfl
    obtain ⟨D, hD, -, hn, -⟩ := Layout.drawn_spec hdrawn
    have hlt := Layout.proved_lt_population hD
    have hr' : o.val * st.c.g + g < D.proved.size := by omega
    rw [Hidden.drawn_digestW (Hidden.loadPublic_nil hpop) hdrawn hD hlt hr' hp]
    exact Hidden.bc_drawnW h hpop hD hlt L _ (regionsOf_mem st) hz o hg hg2 hp hr'
  · obtain ⟨pop, hpop, hdr⟩ := Layout.setupH_drawn h
    obtain ⟨D, hD, -, hn, -⟩ := Layout.drawn_spec hdr
    have hlt := Layout.proved_lt_population hD
    have hr' : o.val * st.c.g + g < D.proved.size := by omega
    rw [Hidden.drawn_digestW hpop hdr hD hlt hr' hp]
    exact Layout.bc_drawnS h hpop hD hlt L _ (regionsOf_mem st) hz o hg hg2 hp hr'

/-! ## Where a rep's commit string sits in the record -/

/-- A round with nothing in it. -/
def noRound : RecRecord.Round := ⟨[], 0, [], 0, [], 0, []⟩

theorem filterMap_ite_eq {α β : Type} (p : α → Bool) (h : α → β) (l : List α) :
    l.filterMap (fun a => if p a then some (h a) else none) = (l.filter p).map h := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    by_cases hp : p a
    · simp [hp, ih]
    · simp [hp, ih]

theorem filterMap_congr' {α β : Type} {f g : α → Option β} {l : List α} (h : ∀ a ∈ l, f a = g a) :
    l.filterMap f = l.filterMap g := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    rw [List.filterMap_cons, List.filterMap_cons, h a List.mem_cons_self,
      ih fun b hb => h b (List.mem_cons_of_mem a hb)]

/-- A stream's rounds read off its schedule: as many as the schedule's, each the stream's, at its index, with the
schedule's round. -/
theorem sched_at {A : List RecRecord.Round} {s : List UInt8} {rs : Array RecVStar.SchedRound}
    (hA : A.map (fun x => (x.stream, x.k, x.n, RecSetup.schedOfRound x)) =
      rs.toList.mapIdx fun k w => (s, k, w.1, w)) :
    A.length = rs.size ∧ ∀ k (hk : k < A.length),
      A[k].stream = s ∧ A[k].k = k ∧ RecSetup.schedOfRound A[k] = rs.getD k default := by
  have hlen : A.length = rs.size := by
    simpa using congrArg List.length hA
  refine ⟨hlen, fun k hk => ?_⟩
  have e := congrArg (·[k]?) hA
  simp only [List.getElem?_map, List.getElem?_mapIdx, List.getElem?_eq_getElem hk, Option.map_some] at e
  rw [Array.getElem?_toList, Array.getElem?_eq_getElem (by omega), Option.map_some] at e
  simp only [Option.some.injEq, Prod.mk.injEq] at e
  obtain ⟨h0, h1, -, h2⟩ := e
  refine ⟨h0, h1, ?_⟩
  rw [h2, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (by omega), Option.getD_some]

/-- A stream's rounds, read off the schedule, list its commit rounds in the schedule's order. -/
theorem commits_of_sched {A : List RecRecord.Round} {s : List UInt8} {rs : Array RecVStar.SchedRound}
    (hA : A.map (fun x => (x.stream, x.k, x.n, RecSetup.schedOfRound x)) =
      rs.toList.mapIdx fun k w => (s, k, w.1, w)) :
    A.filterMap (fun x => if x.commits.isEmpty then none else some (x.k, x.commits)) =
      (RecVStar.commitRounds rs).map fun k => (k, (A.getD k noRound).commits) := by
  obtain ⟨hlen, hat⟩ := sched_at hA
  have hA' : A = (List.range rs.size).map fun k => A.getD k noRound := by
    apply List.ext_getElem (by simp [hlen])
    intro k h1 h2
    simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h1]
  have e := congrArg (List.filterMap fun x => if x.commits.isEmpty then none else some (x.k, x.commits)) hA'
  rw [e, List.filterMap_map]
  unfold RecVStar.commitRounds
  rw [← filterMap_ite_eq]
  apply filterMap_congr'
  intro k hk
  have hk' : k < A.length := by rw [hlen]; exact List.mem_range.1 hk
  obtain ⟨-, h1, h2⟩ := hat k hk'
  have hc : (rs.getD k default).2.2.2 = !A[k].commits.isEmpty := by
    rw [← h2]; rfl
  simp only [Function.comp, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hk', Option.getD_some, hc, h1]
  cases A[k].commits.isEmpty <;> rfl

/-- **One stream's segment of the record**: where the record's rounds are `P ++ A ++ Q`, with `A` the stream's
schedule `rs` and `P` none of the stream's, the `kc`-th of the stream's commit strings is its commit round `k`'s, and
where that round has no salted tops the inner message `P.length + k + 1` (round 0 being the link's) is that string. -/
theorem msg_of_segment {r : RecRecord.Record} {P A Q : List RecRecord.Round} {s : List UInt8}
    {rs : Array RecVStar.SchedRound} (hr : r.rounds = P ++ A ++ Q) (hP : P.filter (·.stream == s) = [])
    (hA : A.map (fun x => (x.stream, x.k, x.n, RecSetup.schedOfRound x)) =
      rs.toList.mapIdx fun k w => (s, k, w.1, w))
    {kc k kk : ℕ} {c : List UInt8} (hk : (RecVStar.commitRounds rs)[kc]? = some k)
    (htops : (rs.getD k default).2.1 = #[]) (hrc : (RecSetup.repCommits r s)[kc]? = some (kk, [c])) :
    kk = k ∧ (r.msgs[P.length + k + 1]?).map Prod.fst = some c := by
  obtain ⟨hlen, hat⟩ := sched_at hA
  have hA1 : A.filter (·.stream == s) = A :=
    List.filter_eq_self.2 fun a ha => by
      obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem ha
      simp [(hat i hi).1]
  have hkc : kc < (RecVStar.commitRounds rs).length := by
    rcases Nat.lt_or_ge kc (RecVStar.commitRounds rs).length with h | h
    · exact h
    · rw [List.getElem?_eq_none h] at hk; cases hk
  have hkm : k ∈ RecVStar.commitRounds rs := List.mem_of_getElem? hk
  have hkA : k < A.length := by
    unfold RecVStar.commitRounds at hkm; rw [hlen]; exact List.mem_range.1 (List.mem_filter.1 hkm).1
  unfold RecSetup.repCommits at hrc
  rw [hr, List.filter_append, List.filter_append, hP, hA1, List.nil_append, List.filterMap_append,
    commits_of_sched hA, List.getElem?_append_left (by simpa using hkc), List.getElem?_map, hk,
    Option.map_some, Option.some.injEq, Prod.mk.injEq] at hrc
  obtain ⟨hkk, hc⟩ := hrc
  refine ⟨hkk.symm, ?_⟩
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hkA, Option.getD_some] at hc
  have ht : A[k].tops = [] := by
    have e := (hat k hkA).2.2
    rw [← e] at htops
    unfold RecSetup.schedOfRound at htops
    simpa using htops
  unfold RecRecord.Record.msgs
  rw [List.getElem?_cons_succ, List.getElem?_map, hr, List.append_assoc,
    List.getElem?_append_right (by omega), Nat.add_sub_cancel_left, List.getElem?_append_left hkA,
    List.getElem?_eq_getElem hkA]
  simp [ht, hc]

theorem toUTF8_toList_inj {s t : String} (h : s.toUTF8.toList = t.toUTF8.toList) : s = t := by
  rw [byteArray_toList_eq, byteArray_toList_eq] at h
  have hd : s.toUTF8.data = t.toUTF8.data := Array.toList_inj.mp h
  have hb : s.toUTF8 = t.toUTF8 := by
    cases hs : s.toUTF8; cases ht : t.toUTF8
    rw [hs, ht] at hd
    cases hd; rfl
  exact String.toByteArray_inj.mp hb

/-- The two reps' streams are apart, whatever the table. -/
theorem rep_ne (t : String) : (s!"{t}/rep{(0 : ℕ)}").toUTF8.toList ≠ (s!"{t}/rep{(1 : ℕ)}").toUTF8.toList := by
  intro h
  have := (String.append_right_inj _).mp (toUTF8_toList_inj h)
  revert this
  decide

/-- **Rep `x`'s commit round `k`, with no salted tops, is inner message `x · R + k + 1`**, R its schedule's length:
the record's rounds open with rep 0's schedule, then rep 1's, and that message's bytes are rep `x`'s `kc`-th commit
string in the record's own format. -/
theorem msg_of_commit {r : RecRecord.Record} {t : String} {rs : Array RecVStar.SchedRound}
    {E : List (String × String × ℕ × Array RecVStar.SchedRound)}
    (h : r.rounds.map (fun x => (x.stream, x.k, x.n, RecSetup.schedOfRound x)) =
      ((s!"{t}/rep{(0 : ℕ)}", "", 0, rs) :: (s!"{t}/rep{(1 : ℕ)}", "", 0, rs) :: E).flatMap
        (fun (s, _, _, rs) => rs.toList.mapIdx fun k w => (s.toUTF8.toList, k, w.1, w)))
    {x kc k kk : ℕ} {c : List UInt8} (hx : x < 2) (hk : (RecVStar.commitRounds rs)[kc]? = some k)
    (htops : (rs.getD k default).2.1 = #[])
    (hrc : (RecSetup.repCommits r (s!"{t}/rep{x}").toUTF8.toList)[kc]? = some (kk, [c])) :
    kk = k ∧ (r.msgs[x * rs.size + k + 1]?).map Prod.fst = some c := by
  simp only [List.flatMap_cons] at h
  obtain ⟨A0, R1, hr, h0, h1'⟩ := List.map_eq_append_iff.1 h
  obtain ⟨A1, AE, rfl, h1, -⟩ := List.map_eq_append_iff.1 h1'
  obtain ⟨l0, a0⟩ := sched_at h0
  rcases (by omega : x = 0 ∨ x = 1) with rfl | rfl
  · have := msg_of_segment (P := []) (by rw [hr]; rfl) rfl h0 hk htops hrc
    simpa using this
  · have hP : A0.filter (·.stream == (s!"{t}/rep{(1 : ℕ)}").toUTF8.toList) = [] :=
      List.filter_eq_nil_iff.2 fun a ha => by
        obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem ha
        rw [(a0 i hi).1]
        simpa using rep_ne t
    have := msg_of_segment (by rw [hr, List.append_assoc]) hP h1 hk htops hrc
    rw [l0] at this
    simpa using this

/-- The schedule `checkReads` holds the record to opens with the two reps', each the inner shape's rep schedule. -/
theorem scheduleOf_reps {i : RecVStar.Inner} {t : String}
    {want : Array (String × String × ℕ × Array RecVStar.SchedRound)} (h : RecVStar.scheduleOf i t = some want) :
    ∃ rs E, (if i.holo then RecVStar.holoRepRounds else RecVStar.repRounds) i.m i.kLog i.claims = some rs ∧
      want.toList = (s!"{t}/rep{(0 : ℕ)}", "", 0, rs) :: (s!"{t}/rep{(1 : ℕ)}", "", 0, rs) :: E := by
  unfold RecVStar.scheduleOf at h
  cases hr : (if i.holo then RecVStar.holoRepRounds else RecVStar.repRounds) i.m i.kLog i.claims with
  | none => rw [hr] at h; cases h
  | some rs =>
    rw [hr] at h
    simp only [bind, Option.bind, pure] at h
    have fin : ∀ X : Array (String × String × ℕ × Array RecVStar.SchedRound),
        want = #[0, 1].map (fun x : ℕ => (s!"{t}/rep{x}", "", 0, rs)) ++ X →
        ∃ E, want.toList = (s!"{t}/rep{(0 : ℕ)}", "", 0, rs) :: (s!"{t}/rep{(1 : ℕ)}", "", 0, rs) :: E := by
      rintro X rfl
      exact ⟨X.toList, by simp only [Array.toList_append, Array.toList_map, List.map_cons, List.map_nil,
        List.cons_append, List.nil_append]⟩
    split at h
    · rename_i sh _
      cases hs : RecVStar.stackedRounds sh with
      | none => rw [hs] at h; cases h
      | some a =>
        rw [hs] at h; cases h
        obtain ⟨E, hE⟩ := fin _ Array.append_assoc
        exact ⟨rs, E, rfl, hE⟩
    · cases h
      obtain ⟨E, hE⟩ := fin _ Array.append_assoc
      exact ⟨rs, E, rfl, hE⟩

/-- **The JSON's `c_k` of rep `x` at a commit round with no salted tops is the coin server's inner message there**:
`checkReads` reads the record's rounds as the schedule's and rep `x`'s commit strings as the JSON's
(`checkReads_ok`), so the `kc`-th entry of rep `x`'s `c_k` is at the commit round `k` the schedule lists `kc`-th, and
the record's message `x · R + k + 1` (round 0 being the link's) is that commit string. -/
theorem cmt_of_rep {i : RecVStar.Inner} {t : String} {want : Array (String × String × ℕ × Array RecVStar.SchedRound)}
    {r : RecRecord.Record} {cs : RecVStar.InnerCommits} (hw : RecVStar.scheduleOf i t = some want)
    (h : r.rounds.map (fun x => (x.stream, x.k, x.n, RecSetup.schedOfRound x)) =
      want.toList.flatMap (fun (s, _, _, rs) => rs.toList.mapIdx fun k w => (s.toUTF8.toList, k, w.1, w)))
    {x : ℕ} (hx : x < 2)
    (hc : RecSetup.repCommits r s!"{t}/rep{x}".toUTF8.toList =
      (cs.reps.getD x #[]).toList.map fun (k, c) => (k, [c.toList]))
    {rs : Array RecVStar.SchedRound}
    (hrs : (if i.holo then RecVStar.holoRepRounds else RecVStar.repRounds) i.m i.kLog i.claims = some rs)
    {kc k kk : ℕ} {c : ByteArray} (hk : (RecVStar.commitRounds rs)[kc]? = some k)
    (htops : (rs.getD k default).2.1 = #[]) (hcs : (cs.reps.getD x #[])[kc]? = some (kk, c)) :
    kk = k ∧ (r.msgs.map Prod.fst).getD (x * rs.size + k + 1) [] = c.toList := by
  obtain ⟨rs', E, hrs', hE⟩ := scheduleOf_reps hw
  rw [hrs] at hrs'; cases hrs'
  rw [hE] at h
  have hrc : (RecSetup.repCommits r (s!"{t}/rep{x}").toUTF8.toList)[kc]? = some (kk, [c.toList]) := by
    rw [hc, List.getElem?_map, Array.getElem?_toList, hcs]; rfl
  obtain ⟨hkk, hm⟩ := msg_of_commit h hx hk htops hrc
  refine ⟨hkk, ?_⟩
  rw [List.getD_eq_getElem?_getD, List.getElem?_map, hm, Option.getD_some]

/-! ## A slot's bytes are its public row -/

/-- `Readable.layout_of_check` at the regions the soundness proof reads (`relS_layout`'s, `Refine.regionsOf`). -/
theorem layout_of_check_regs {decode : Msg → Option Flock.Readable.Read} {x : Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.check (Flock.Readable.parseOf Flock.Readable.relation decode) x w = true)
    {p : Flock.Readable.Parsed} (hx : Flock.Readable.parseBytes decode x = some p) (ht : p.tags.typed = false)
    (hpi : p.publicInputs = none) {t : ℕ} (htl : t < p.sts.size) :
    ∃ L : FlockSoundness.Layout p.sts[t], (L.model (FlockSoundness.Refine.regionsOf p.sts[t])).Satisfies
      (Flock.Readable.zOf p.sts.size w t p.sts[t].m) := by
  obtain ⟨p', hx', hp', hR⟩ := Flock.Readable.relS_of_check h
  rw [hx] at hx'
  cases hx'
  exact (Flock.Readable.relS_layout hp' ht hpi w).1 hR t htl

open FlockSoundness.Discharge in
/-- **A table `setupOne` sets up holds a power of two of template instances a block**, in either branch: its circuit
passed `HmRow.check` (`CheckFacts.g_pow2`, `Layout.vu_lt_pow2_of`). -/
theorem vu_lt_pow2_setupOne {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Json} {program : Option ByteArray} {st : Stmt} (ht : tags.typed = false)
    (h : HiddenExec.setupOne tags cf pf coins tables draw partition program = .ok st) {g : ℕ} (hg : g < st.c.g) :
    g < 2 ^ st.c.g.log2 := by
  unfold HiddenExec.setupOne at h
  split at h
  · have P := FlockSoundness.parse_facts (HiddenExec.parse_untyped ht (HiddenExec.setupHidden_parseC h)).1
    obtain ⟨ls, hls⟩ := P.check
    exact Layout.vu_lt_pow2_of (FlockSoundness.check_facts P.typed hls) hg
  · exact Layout.vu_lt_pow2_of (Aliased.setupH_unitFacts ht h).check hg

open FlockSoundness.Discharge in
/-- **Port `q`'s hm96 slot at a drawn table's instance `n` holds that instance's public `b ‖ c`**: at a table
`setupOne` sets up at a draw, with the witness's one-table share satisfying its model, the slot's out columns
(`hmCommit`) read as the model's `b ‖ c` bytes (`bc_bit`, `Hidden.bytesAt_bc`), which are the statement's digest of
instance `n`'s row `q` (`bc_setupOne`). -/
theorem slotPub_table {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {d : Json} {partition : Option Json} {program : Option ByteArray} {st : Stmt} (ht : tags.typed = false)
    (hd : HiddenExec.setupOne tags cf pf coins tables (some d) partition program = .ok st)
    (hupv : 0 < st.c.upv) (hv1 : ∀ p ∈ st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    (L : FlockSoundness.Layout st) {w : Claims.Flock.Wit}
    (hz : (L.model (FlockSoundness.Refine.regionsOf st)).Satisfies (Flock.Readable.zOf 1 w 0 st.m)) {q n : ℕ}
    (hq : q < st.c.ports.size) (hn : n < st.pub.n) (hb : n / st.c.g < 2 ^ st.nbl) :
    Claims.Recursion.VStarB.readForms w (hmCommit st.c q n) = (st.pub.digest n q st.c.ports.size).data.toList := by
  obtain ⟨R, hg0, -, -, -⟩ := table_rows ht hd hupv hv1 L _ hz
  have hgn : n % st.c.g < st.c.g := Nat.mod_lt _ hg0
  replace hb : n / st.c.g < 2 ^ (L.model (FlockSoundness.Refine.regionsOf st)).nbl := by
    show n / st.c.g < 2 ^ (st.m - st.c.kLog)
    rw [L.m, Nat.add_sub_cancel_left]
    exact hb
  let o : Fin (2 ^ (L.model (FlockSoundness.Refine.regionsOf st)).nbl) := ⟨n / st.c.g, hb⟩
  have e : o.val * st.c.g + n % st.c.g = n := Nat.div_add_mod' n st.c.g
  have hpub := bc_setupOne hd L hz o hgn (vu_lt_pow2_setupOne ht hd hgn) hq (by rw [e]; exact hn)
  rw [Hidden.bytesAt_bc L R _ _ o hgn hq, e] at hpub
  rw [← hpub]
  refine readForms_eq_of_bits (by simp [hmCommit, Hm.length_bytesOf]) fun t ht' => ?_
  have ht1 : t < 1024 := by simpa [hmCommit] using ht'
  have hy : hmBase st.c q n + (hmOutCol st.c + t) < 2 ^ st.c.kLog := by
    have := Hm.bcCol_lt R hgn hq (j := t / 8) (t := t % 8) (by omega) (by omega)
    unfold Hm.bcCol at this
    rw [hmOut_eq] at this
    unfold hmBase
    omega
  simp only [hmCommit, List.getElem_map, List.getElem_range, formBit_var]
  exact (bc_bit (L.model _) w st.c rfl hb hy ht1).symm
