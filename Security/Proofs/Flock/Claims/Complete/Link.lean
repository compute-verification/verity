import Proofs.Flock.Claims.Complete.Shape

/-!
# The first send is the link message

* **Quiet runs** (`quietS`): a pure-session run related to a verdict `x` returns `x` and leaves the bodies as they were.
  The schedules and the level-0 commitments are quiet (`schedsP_quiet`, `commitsP_quiet`), with the verdict-monad runs'
  values.
* **The session in three parts** (`sessionP_eq`): the schedules (`schedsP`), the level-0 commitments (`commitsP`), then
  the link message and what follows its coin (`afterLinkP`), whose sends only append to the bodies (`afterLinkP_sim`).
* `sessionS_link`: a pure session that returns sent the link message of its level-0 caps, with no public words, first.
-/

open Flock FlockDeploy

namespace FlockComplete

open With

/-- **Quiet runs**: the pure session's run returns `x` and leaves the bodies as they were. -/
def quietS : With.Sim V PS where
  R x y := ∀ l, y.go l = (x, l)
  pure _ _ := rfl
  bind {α β x y f g} hxy hfg l := by
    rw [go_bind, hxy l]
    cases x with
    | error _ => rfl
    | ok a => exact hfg a l
  throw _ _ := rfl
  lift v _ := by cases v <;> rfl

theorem quietS_hashes (h : ByteArray → ByteArray) :
    quietS.Hashes (Hashes.ofSha512 h) (Hashes.ofSha512 h) where
  sha512 _ _ := rfl
  sha256 _ _ := rfl
  blake3 _ _ := rfl
  blake3Keyed _ _ _ := rfl

/-! ## The session in three parts -/

/-- The session's schedules: each table's fast100 schedule. -/
def schedsP {M : Type → Type} [Monad M] [MonadExceptOf String M] (tabs : Array Setup) : M (Array Schedule) :=
  tabs.mapM fun st => match fast100 st.m with
    | some sch => pure sch
    | none => throw s!"m = {st.m}: no fast100 schedule"

/-- Each table's packed table. -/
def tablesP (tabs : Array Setup) (trs : Array Trace) : Array (Array F128) :=
  (Array.range tabs.size).map fun i => packed trs[i]! tabs[i]!.m

/-- Each table's level-0 commitment, salted from the tape's level `0`. -/
def commitsP {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (hs : Hashes M)
    (tabs : Array Setup) (schs : Array Schedule) (tbls : Array (Array F128)) (tape : Tape) : M (Array Commit) :=
  (Array.range tabs.size).mapM fun i =>
    commitP hs tabs[i]!.merkle (schs[i]!.levels.getD 0 default) tbls[i]! (tape.salt i 0 0)

/-- The session from the link message's coin `c1` on: the rep streams, the record and the proof files. -/
def afterLinkP {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (io : Asks M) (hs : Hashes M)
    (spec : SessionSpec) (tabs : Array Setup) (trs : Array Trace) (tape : Tape) (schs : Array Schedule)
    (tbls : Array (Array F128)) (cms : Array Commit) (c1 : ℕ → F128) : M (String × Array ByteArray) := do
  let rootB := cms.map fun c => flat c.cap
  let points := (Array.range spec.points).map fun p => (Array.range spec.mPts).map fun j => c1 (p * spec.mPts + j)
  let mut streams : Array (Array Round) := #[]
  let mut proofs : Array ByteArray := #[]
  for (i, r, _) in spec.repStreams do
    let st := tabs[i]!
    let extras ← (monadLift (st.extra points) : M _)
    let (p, t) ← (repP io hs st schs[i]! trs[i]! tbls[i]! cms[i]! extras (tape.salt i r) (tape.nonce i r)).run
      (Transcript.init #[] (spec.domain spec.tables[i]! r).toUTF8)
    streams := streams.push t.rounds
    proofs := proofs.push (FlockVBridge.Family.encodeProof (shapeOf st schs[i]! extras.size) p)
  let record ← recordJson hs spec rootB points streams proofs
  return (record, proofs)

/-- **The session in three parts**: the schedules, the level-0 commitments, the link message, and the rest. -/
theorem sessionP_eq {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (io : Asks M)
    (hs : Hashes M) (spec : SessionSpec) (tabs : Array Setup) (trs : Array Trace) (tape : Tape) :
    sessionP io hs spec tabs trs tape = (do
      let schs ← schedsP tabs
      let cms ← commitsP hs tabs schs (tablesP tabs trs) tape
      let c1 ← io.link (cms.map fun c => flat c.cap) (tabs.map fun _ => ByteArray.empty)
      afterLinkP io hs spec tabs trs tape schs (tablesP tabs trs) cms c1) := rfl

theorem schedsP_quiet (tabs : Array Setup) : quietS.R (schedsP tabs : V _) (schedsP tabs : PS _) := by
  unfold schedsP
  sim_walk

theorem commitP_quiet (h : ByteArray → ByteArray) (ms : MerkleScheme) (lv : Level) (g : Array F128)
    (salt : ℕ → ByteArray) :
    quietS.R (commitP (Hashes.ofSha512 h) ms lv g salt : V _) (commitP (Hashes.ofSha512 h) ms lv g salt : PS _) := by
  have ho := opsWith_sim quietS (quietS_hashes h) ms
  unfold commitP
  sim_walk [ho.leaf, ho.node]

theorem commitsP_quiet (h : ByteArray → ByteArray) (tabs : Array Setup) (schs : Array Schedule)
    (tbls : Array (Array F128)) (tape : Tape) :
    quietS.R (commitsP (Hashes.ofSha512 h) tabs schs tbls tape : V _)
      (commitsP (Hashes.ofSha512 h) tabs schs tbls tape : PS _) := by
  unfold commitsP
  sim_walk [commitP_quiet h]

theorem afterLinkP_sim (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (spec : SessionSpec)
    (tabs : Array Setup) (trs : Array Trace) (tape : Tape) (schs : Array Schedule) (tbls : Array (Array F128))
    (cms : Array Commit) (c1 : ℕ → F128) :
    (simS H resp).R (afterLinkP asks Hashes.pi spec tabs trs tape schs tbls cms c1)
      (afterLinkP (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) spec tabs trs tape schs tbls cms c1) := by
  unfold afterLinkP
  sim_walk [repP_sim H resp, recordJson_sim H resp]

theorem asksS_link_go (resp : ℕ → Claims.Msg) (a b : Array ByteArray) (l : Array Claims.Msg) :
    ((asksS resp).link a b).go l = (.ok (coinOf (resp l.size)), l.push (linkBody a.toList b.toList)) := rfl

/-- **The first send is the link message**: a pure session that returns ran its schedules and level-0 commitments
(their verdict-monad runs return), then sent the link message of the caps, with no public words, first. -/
theorem sessionS_link {H : Claims.Msg → Claims.Msg} {resp : ℕ → Claims.Msg} {spec : SessionSpec}
    {sts : Array Flock.Stmt} {w : Claims.Flock.Wit} {ω : Tape} {rec : String} {pfs : Array ByteArray}
    {bodies : Array Claims.Msg} (h : sessionS H resp spec sts w ω = (.ok (rec, pfs), bodies)) :
    ∃ schs cms, (schedsP (tabsOf sts) : V _) = .ok schs ∧
      (commitsP (Hashes.ofSha512 (Claims.hOf H)) (tabsOf sts) schs (tablesP (tabsOf sts) (tracesOf sts w)) ω : V _) =
        .ok cms ∧
      bodies.toList.head? = some (linkBody (cms.map fun c => flat c.cap).toList
        ((tabsOf sts).map fun _ => ByteArray.empty).toList) := by
  unfold sessionS at h
  rw [sessionP_eq, go_bind, schedsP_quiet] at h
  cases hs : (schedsP (tabsOf sts) : V _) with
  | error e => rw [hs] at h; cases h
  | ok schs =>
    rw [hs] at h
    dsimp only at h
    rw [go_bind, commitsP_quiet] at h
    cases hc : (commitsP (Hashes.ofSha512 (Claims.hOf H)) (tabsOf sts) schs (tablesP (tabsOf sts) (tracesOf sts w)) ω :
      V _) with
    | error e => rw [hc] at h; cases h
    | ok cms =>
      rw [hc] at h
      dsimp only at h
      rw [go_bind, asksS_link_go] at h
      dsimp only at h
      rw [Array.size_empty] at h
      refine ⟨schs, cms, rfl, hc, ?_⟩
      obtain ⟨bs, -, hl⟩ := afterLinkP_sim H resp spec (tabsOf sts) (tracesOf sts w) ω schs
        (tablesP (tabsOf sts) (tracesOf sts w)) cms (coinOf (resp 0)) (#[].push (linkBody
          (cms.map fun c => flat c.cap).toList ((tabsOf sts).map fun _ => ByteArray.empty).toList))
      rw [h] at hl
      dsimp only at hl
      rw [hl]
      simp

/-! ## The caps' sizes -/

theorem except_ok_bind {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {v : β} (h : (x >>= f) = .ok v) :
    ∃ a, x = .ok a ∧ f a = .ok v := by
  cases x with
  | error _ => cases h
  | ok a => exact ⟨a, rfl, h⟩

/-- An invariant of every step holds of a loop's result. -/
theorem except_forIn_inv {ε α β : Type} (P : β → Prop) {f : α → β → Except ε (ForInStep β)}
    (hf : ∀ a b r, P b → f a b = .ok r → P r.value) :
    ∀ (l : List α) (b r : β), P b → forIn l b f = .ok r → P r := by
  intro l
  induction l with
  | nil =>
    intro b r hb h
    rw [List.forIn_nil] at h
    cases h
    exact hb
  | cons a as ih =>
    intro b r hb h
    rw [List.forIn_cons] at h
    obtain ⟨s, hs, h⟩ := except_ok_bind h
    cases s with
    | done b' => cases h; exact hf a b _ hb hs
    | yield b' => exact ih b' r (hf a b _ hb hs) h

/-- A traversal whose every step returns 64 bytes returns one 64-byte value per element. -/
theorem mapM_sized {α : Type} {xs : Array α} {f : α → V ByteArray} {ys : Array ByteArray}
    (hf : ∀ a r, f a = .ok r → r.size = 64) (h : xs.mapM f = .ok ys) : ys.size = xs.size ∧ ∀ b ∈ ys, b.size = 64 := by
  obtain ⟨hsz, hall⟩ := FlockSoundness.Discharge.Exec.array_mapM_ok_getElem h
  refine ⟨hsz, fun b hb => ?_⟩
  obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 hb
  exact hf _ _ (hall j (by omega) hj)

theorem hm96_ops (h : ByteArray → ByteArray) :
    MerkleScheme.hm96Sha512.opsWith (Hashes.ofSha512 h : Hashes V) = MerkleOps.hm96With (Hashes.ofSha512 h) := rfl

theorem hm96_leaf {h : ByteArray → ByteArray} (h64 : ∀ b, (h b).size = 64) (row : Array F128) (salt : ByteArray)
    {r : ByteArray} (hr : (MerkleOps.hm96With (Hashes.ofSha512 h : Hashes V)).leaf row salt = .ok r) : r.size = 64 := by
  obtain ⟨d, hd⟩ : ∃ d, (MerkleOps.hm96With (Hashes.ofSha512 h : Hashes V)).leaf row salt = .ok (h d) := ⟨_, rfl⟩
  rw [hd] at hr
  cases hr
  exact h64 d

theorem hm96_node {h : ByteArray → ByteArray} (h64 : ∀ b, (h b).size = 64) (a b : ByteArray) {r : ByteArray}
    (hr : (MerkleOps.hm96With (Hashes.ofSha512 h : Hashes V)).node a b = .ok r) : r.size = 64 := by
  cases hr
  exact h64 _

theorem plain_ops (h : ByteArray → ByteArray) :
    (MerkleScheme.sha512 "flock-leaf/sha512-unsalted").opsWith (Hashes.ofSha512 h : Hashes V) =
      MerkleOps.sha512With (Hashes.ofSha512 h) := rfl

theorem plain_leaf {h : ByteArray → ByteArray} (h64 : ∀ b, (h b).size = 64) (row : Array F128) (salt : ByteArray)
    {r : ByteArray} (hr : (MerkleOps.sha512With (Hashes.ofSha512 h : Hashes V)).leaf row salt = .ok r) :
    r.size = 64 := by
  cases hr
  exact h64 _

theorem plain_node {h : ByteArray → ByteArray} (h64 : ∀ b, (h b).size = 64) (a b : ByteArray) {r : ByteArray}
    (hr : (MerkleOps.sha512With (Hashes.ofSha512 h : Hashes V)).node a b = .ok r) : r.size = 64 := by
  cases hr
  exact h64 _

/-- **A level's cap under `hm96-sha512/v1` or plain SHA-512 leaves**, for a SHA-512 of 64-byte outputs: at most `2^d`
digests of 64 bytes. -/
theorem commitP_cap {h : ByteArray → ByteArray} (h64 : ∀ b, (h b).size = 64) (lv : Level) (g : Array F128)
    (salt : ℕ → ByteArray) {c : Commit} {ms : MerkleScheme}
    (hms : ms = MerkleScheme.hm96Sha512 ∨ ms = MerkleScheme.sha512 "flock-leaf/sha512-unsalted")
    (hc : (commitP (Hashes.ofSha512 h) ms lv g salt : V Commit) = .ok c) :
    c.cap.size ≤ 2 ^ lv.d ∧ ∀ b ∈ c.cap, b.size = 64 := by
  have hw : (∀ row y r, (ms.opsWith (Hashes.ofSha512 h : Hashes V)).leaf row y = .ok r → r.size = 64) ∧
      ∀ a b r, (ms.opsWith (Hashes.ofSha512 h : Hashes V)).node a b = .ok r → r.size = 64 := by
    rcases hms with rfl | rfl
    · rw [hm96_ops]
      exact ⟨fun row y _ hr => hm96_leaf h64 row y hr, fun a b _ hr => hm96_node h64 a b hr⟩
    · rw [plain_ops]
      exact ⟨fun row y _ hr => plain_leaf h64 row y hr, fun a b _ hr => plain_node h64 a b hr⟩
  unfold commitP at hc
  obtain ⟨leaves, hl, hc⟩ := except_ok_bind hc
  obtain ⟨s, hs, hc⟩ := except_ok_bind hc
  cases hc
  have hL := mapM_sized (fun _ _ hr => hw.1 _ _ _ hr) hl
  rw [Array.size_range] at hL
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hs
  refine except_forIn_inv (fun s : Array (Array ByteArray) × Array ByteArray => s.2.size ≤ 2 ^ lv.d ∧
    ∀ b ∈ s.2, b.size = 64) ?_ _ _ _ ⟨hL.1.le, hL.2⟩ hs
  rintro - ⟨layers, cur⟩ r ⟨h1, -⟩ hr
  obtain ⟨cur', hc', hr⟩ := except_ok_bind hr
  cases hr
  have hN := mapM_sized (fun _ _ hr => hw.2 _ _ _ hr) hc'
  rw [Array.size_range] at hN
  exact ⟨hN.1.trans_le ((Nat.div_le_self _ _).trans h1), hN.2⟩

/-- 64 bytes per digest: `flat` of `n` of them is `64 n` bytes. -/
theorem flat_size {cap : Array ByteArray} (h : ∀ b ∈ cap, b.size = 64) : (flat cap).size = 64 * cap.size := by
  unfold flat
  rw [← Array.foldl_toList]
  have h' : ∀ b ∈ cap.toList, b.size = 64 := fun b hb => h b (Array.mem_toList_iff.1 hb)
  rw [← Array.length_toList]
  suffices ∀ (l : List ByteArray) (acc : ByteArray), (∀ b ∈ l, b.size = 64) →
      (l.foldl (· ++ ·) acc).size = acc.size + 64 * l.length by
    simpa using this cap.toList ByteArray.empty h'
  intro l
  induction l with
  | nil => intro acc _; simp
  | cons b l ih =>
    intro acc hl
    rw [List.foldl_cons, ih _ fun x hx => hl x (List.mem_cons_of_mem _ hx), ByteArray.size_append,
      hl b List.mem_cons_self, List.length_cons]
    ring

/-- fast100's level 0 has `d ≤ 56` at every `m`. -/
theorem fast100_d0 {m : ℕ} {sch : Schedule} (h : fast100 m = some sch) : (sch.levels.getD 0 default).d ≤ 56 := by
  by_cases hm : m < 40
  · have key : ∀ m : Fin 40, (fast100 m.val).all (fun s => decide ((s.levels.getD 0 default).d ≤ 56)) = true := by
      decide +kernel
    have := key ⟨m, hm⟩
    simp only [h, Option.all_some, decide_eq_true_eq] at this
    exact this
  · rw [FlockSoundness.Refine.exec_fast100_none m (by omega)] at h
    cases h

/-- **The case's statement is one table, and commits with the scope check's leaf scheme**: `hm96-sha512/v1`, or plain
SHA-512 leaves under hidden outputs. -/
theorem Case.leaf {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation} (c : Case H rel) :
    ∃ st, c.sts = #[st] ∧ c.spec = st.spec ∧
      (st.merkle = MerkleScheme.hm96Sha512 ∨ st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted") := by
  obtain ⟨p, hp, hsc⟩ := (Readable.scope_iff (deployPlain rel) rel (readOf false) rfl
    (fun _ _ h => readOf_zk h) c.x).1 c.scope
  have hi := (setupOf_some c.setup).1
  have hzk : p.zk = false := readOf_zk (x := c.x) (by
    unfold Readable.parseBytes at hp
    obtain ⟨r, hr, hrp⟩ := Option.bind_eq_some_iff.1 hp
    rw [hr, ← Readable.toRead_of_parse hrp])
  unfold Readable.parseBytes readOf at hp
  rw [hi] at hp
  have ht := Readable.toRead_of_parse (r := c.s.read) hp
  obtain ⟨hc, -⟩ := (Readable.scope_ok_iff p).1 hsc
  have hR := ((Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hc).1
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hR.2.2.2.2.2])
  have hc' : ProvedScope.check p.tags p.zk p.I.circuitFile p.I.publicFile p.I.tables p.I.count false p.registered =
      .ok () := by
    have := hc
    unfold Readable.Read.check at this
    rw [hpi] at this
    exact this
  have hl := (FlockSoundness.Discharge.FailClosed.check_ok hc').1.2.2.2.1
  have hcount : p.I.count ≤ 1 := by simpa [hzk] using hR.2.2.2.2.1
  have hset := Ins.setup_read (setupOf_some c.setup).2
  rw [← ht] at hset
  unfold Readable.Read.setup at hset
  rw [hpi] at hset
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okAny hset with ⟨-, st, hst, hsp, hsts⟩ | ⟨hc1, -⟩
  · refine ⟨st, hsts, hsp, ?_⟩
    rcases hl with hl | ⟨-, hh, hl⟩
    · left
      cases hh : p.tags.hiddenOutputs
      · exact FlockSoundness.Discharge.Exec.setupH_hm96 hl
          ((FlockSoundness.Discharge.HiddenExec.setupOne_public hh).symm.trans hst)
      · exact FlockSoundness.Discharge.HiddenExec.setupHidden_hm96 hl
          ((FlockSoundness.Discharge.HiddenExec.setupOne_hidden hh).symm.trans hst)
    · exact Or.inr (FlockSoundness.Discharge.HiddenExec.setupHidden_plain hl
        ((FlockSoundness.Discharge.HiddenExec.setupOne_hidden hh).symm.trans hst))
  · omega

/-- One table's schedules: its fast100 schedule. -/
theorem schedsP_one {st : Stmt} {schs : Array Schedule} (hs : (schedsP #[Setup.ofCircuit st] : V _) = .ok schs) :
    ∃ sch, fast100 st.m = some sch ∧ schs = #[sch] := by
  unfold schedsP at hs
  obtain ⟨h1, h2⟩ := FlockSoundness.Discharge.Exec.array_mapM_ok_getElem hs
  have h0 := h2 0 (by simp) (by rw [h1]; simp)
  simp only [List.getElem_toArray, List.getElem_cons_zero, Setup.ofCircuit] at h0
  cases hf : fast100 st.m with
  | none => rw [hf] at h0; cases h0
  | some sch =>
    rw [hf] at h0
    refine ⟨sch, rfl, ?_⟩
    apply Array.ext (by simpa using h1)
    intro j hj _
    have : j = 0 := by simp at h1; omega
    subst this
    exact (Except.ok.inj h0).symm

/-! ## The goal -/

/-- **The first send is the link message**, each length in its word: the goal `LinkFirst`, for a SHA-512 of 64-byte
outputs. -/
theorem linkFirst_goal (H : Claims.Msg → Claims.Msg) (hH : ∀ m, (H m).length = 64)
    (rel : Readable.Parsed → Readable.Relation) : LinkFirst H rel := by
  intro o
  have h64 : ∀ b, (Claims.hOf H b).size = 64 := fun b => by
    simp [Claims.hOf, Claims.ofMsg, ByteArray.size, hH]
  obtain ⟨schs, cms, hs, hc, hh⟩ := sessionS_link o.ran
  obtain ⟨st, hsts, -, hms⟩ := o.leaf
  have htabs : tabsOf o.sts = #[Setup.ofCircuit st] := by rw [hsts]; simp [tabsOf]
  rw [htabs] at hs
  obtain ⟨sch, hf, rfl⟩ := schedsP_one hs
  unfold commitsP at hc
  obtain ⟨hcs, hcall⟩ := FlockSoundness.Discharge.Exec.range_mapM_ok hc
  rw [htabs] at hcs hcall
  simp only [List.size_toArray, List.length_cons, List.length_nil, zero_add] at hcs
  refine ⟨_, _, hh, ?_, ?_, ?_, ?_⟩
  · simp [hcs]
  · simp [htabs]
  · intro b hb
    obtain ⟨c, hcm, rfl⟩ : ∃ c ∈ cms, flat c.cap = b := by simpa using hb
    obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 hcm
    have hj0 : j = 0 := by omega
    have hcj := hcall j hj
    subst hj0
    simp only [List.getElem!_toArray, List.getElem!_cons_zero, Setup.ofCircuit] at hcj
    obtain ⟨hcap, hcap64⟩ := commitP_cap h64 _ _ _ hms hcj
    rw [flat_size hcap64]
    have hd := fast100_d0 hf
    calc 64 * cms[0].cap.size ≤ 64 * 2 ^ 56 :=
          Nat.mul_le_mul_left _ (hcap.trans (Nat.pow_le_pow_right (by norm_num) hd))
      _ < 2 ^ 64 := by norm_num
  · simp

end FlockComplete
