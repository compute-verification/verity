import Proofs.Flock.Claims.Recursion.VerdictLog
import Proofs.Flock.Claims.Recursion.LastColsDeployed
import Proofs.Flock.Claims.Complete.ZkTables.Scope
import Proofs.Flock.Claims.DeployK

/-!
# Two accepted plain decisions on one set of proof files make one set of calls

* `tables_nodup`, `repNames_nodup`: a session the statement sets up names its tables apart (`Tags.tableNames`, at most
  `Tags.MAX_TABLES`), so its rep streams too.
* `streams_pinned`: S11's count and names then leave a decoded session no stream but the one it finds at each rep
  stream's name.
* `decode_publics`: the public words `Record.decode` reads (S7) are `liveOf`'s.
* `verdict_calls_sub`: every SHA-512 input of an accepting plain verdict (`verdict_calls`: the decoder's, S17's, the
  reps') is one of another's on the same proof files, where the decoded sessions have the same `root_B`, points, public
  words and rep rounds. S10 pins `y`, and a public word one record hashes and the other leaves empty is `ε`, which each
  stream's first round hashes (`eps_mem`).
* `plain_calls_agree`: two accepted plain decisions on one setup, link, coin and round messages, with the same proof
  files and the same rep rounds, make one set of calls: a decision logs the setup's calls, then the verdict's
  (`next_decide_calls`), and custody pins the rest. `plain_decide_shared` gives the shared decision.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness.Refine

/-! ## The tables' names are distinct -/

/-- A table index's digits read back. -/
def digitsVal (l : List Char) : ℕ := l.foldl (fun n c => 10 * n + (c.toNat - 48)) 0

theorem digitsVal_pad : ∀ j < 64, digitsVal ((if j < 10 then ['0'] else []) ++ Nat.toDigits 10 j) = j := by
  decide +kernel

theorem pad_toList (j : ℕ) : ((if j < 10 then "0" else "") ++ toString j).toList =
    (if j < 10 then ['0'] else []) ++ Nat.toDigits 10 j := by
  rw [String.toList_append, Nat.toString_eq_repr, Nat.toList_repr]
  split <;> simp

theorem tableNames_nodup (t : Tags) {count : ℕ} (h : count ≤ Tags.MAX_TABLES) :
    (t.tableNames count).toList.Nodup := by
  unfold Tags.tableNames
  split
  · simp
  · rw [Array.toList_map, Array.toList_range]
    refine List.Nodup.map_on (fun i hi j hj e => ?_) List.nodup_range
    rw [List.mem_range] at hi hj
    unfold Tags.MAX_TABLES at h
    have e' := congrArg String.toList e
    simp only [String.toList_append, List.append_assoc, List.append_cancel_left_eq] at e'
    have e2 : ((if i < 10 then "0" else "") ++ toString i).toList =
        ((if j < 10 then "0" else "") ++ toString j).toList := by
      rw [String.toList_append, String.toList_append]
      exact e'
    rw [pad_toList, pad_toList] at e2
    rw [← digitsVal_pad i (by omega), ← digitsVal_pad j (by omega), e2]

/-- **A session the statement sets up names its tables apart.** -/
theorem tables_nodup {s : Ins} {spec : SessionSpec} {sts : Array Stmt} (hs : s.setup = .ok (spec, sts))
    (hpi : s.read.publicInputs = none) : spec.tables.toList.Nodup := by
  have h := Ins.setup_read hs
  unfold Readable.Read.setup at h
  rw [hpi] at h
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okAny h with ⟨-, st, -, rfl, -⟩ | ⟨hc, -, -, -, -⟩
  · simp [Stmt.spec]
  · obtain ⟨hn, hmax⟩ := FlockComplete.ZkTables.setupTables_names hc h
    rw [hn]
    exact tableNames_nodup _ hmax

theorem toDigits_two_inj : ∀ r < 2, ∀ r' < 2, Nat.toDigits 10 r = Nat.toDigits 10 r' → r = r' := by decide

/-- **Tables named apart name their rep streams apart.** -/
theorem repNames_nodup {spec : SessionSpec} (h : spec.tables.toList.Nodup) :
    (spec.repStreams.toList.map (·.2.2)).Nodup := by
  rw [repStreams_toList, List.map_map]
  refine List.Nodup.map_on (fun k hk k' hk' e => ?_) List.nodup_range
  rw [List.mem_range] at hk hk'
  have e' := congrArg String.toList e
  simp only [Function.comp, String.toList_append, Nat.toString_eq_repr, Nat.toList_repr] at e'
  have hlen : ∀ r < 2, (Nat.toDigits 10 r).length = 1 := by decide
  obtain ⟨e1, e2⟩ :=
    List.append_inj' e' (by rw [hlen _ (Nat.mod_lt _ (by omega)), hlen _ (Nat.mod_lt _ (by omega))])
  have et : spec.tables[k / 2]! = spec.tables[k' / 2]! := String.toList_injective (List.append_cancel_right e1)
  have hi : k / 2 < spec.tables.size := by have := repStreams_size spec; omega
  have hi' : k' / 2 < spec.tables.size := by have := repStreams_size spec; omega
  rw [getElem!_pos spec.tables _ hi, getElem!_pos spec.tables _ hi', ← Array.getElem_toList,
    ← Array.getElem_toList, h.getElem_inj] at et
  have := toDigits_two_inj _ (Nat.mod_lt _ (by omega)) _ (Nat.mod_lt _ (by omega)) e2
  omega

/-! ## S11 leaves no stream but the rep streams -/

theorem find?_self_of_nodup {α β : Type} [BEq β] [LawfulBEq β] (key : α → β) :
    ∀ {l : List α}, (l.map key).Nodup → ∀ {s : α}, s ∈ l → l.find? (fun x => key x == key s) = some s
  | [], _, _, h => by cases h
  | a :: l, hn, s, h => by
    rw [List.map_cons, List.nodup_cons] at hn
    rcases List.mem_cons.1 h with rfl | h
    · simp
    · have hne : ¬(key a == key s) = true := by
        rw [beq_iff_eq]
        intro e
        exact hn.1 (e ▸ List.mem_map_of_mem h)
      rw [List.find?_cons_of_neg (p := fun x => key x == key s) hne]
      exact find?_self_of_nodup key hn.2 h

/-- **S11 pins a decoded session's streams**: with the rep streams named apart, as many streams as rep streams, and
one found at each rep stream's name, every stream is the one found at some rep stream's name. -/
theorem streams_pinned {spec : SessionSpec} {sess : RunRecord} (hn : (spec.repStreams.toList.map (·.2.2)).Nodup)
    (hsz : sess.streams.size = spec.repStreams.size)
    (hf : ∀ x ∈ spec.repStreams.toList, ∃ s, sess.streams.find? (·.name == x.2.2) = some s) :
    ∀ s ∈ sess.streams.toList, ∃ q, ∃ hq : q < spec.repStreams.size,
      sess.streams.find? (·.name == spec.repStreams[q].2.2) = some s := by
  classical
  have hsub : (spec.repStreams.toList.map (·.2.2)).toFinset ⊆ (sess.streams.toList.map StreamRec.name).toFinset := by
    intro w hw
    rw [List.mem_toFinset, List.mem_map] at hw ⊢
    obtain ⟨x, hx, rfl⟩ := hw
    obtain ⟨s, hs⟩ := hf x hx
    rw [← Array.find?_toList] at hs
    exact ⟨s, List.mem_of_find?_eq_some hs, by simpa using List.find?_some hs⟩
  have hcW : (spec.repStreams.toList.map (·.2.2)).toFinset.card = (sess.streams.toList.map StreamRec.name).length := by
    rw [List.toFinset_card_of_nodup hn]
    simp [hsz]
  have hnN : (sess.streams.toList.map StreamRec.name).Nodup := by
    have hc : (sess.streams.toList.map StreamRec.name).toFinset.card =
        (sess.streams.toList.map StreamRec.name).length :=
      le_antisymm (List.toFinset_card_le _) (hcW ▸ Finset.card_le_card hsub)
    exact Multiset.coe_nodup.1 (Multiset.toFinset_card_eq_card_iff_nodup.1 hc)
  have heq := Finset.eq_of_subset_of_card_le hsub (by rw [hcW]; exact List.toFinset_card_le _)
  intro s hs
  have hsW : s.name ∈ spec.repStreams.toList.map (·.2.2) := by
    rw [← List.mem_toFinset, heq, List.mem_toFinset]
    exact List.mem_map_of_mem hs
  obtain ⟨x, hx, hxs⟩ := List.mem_map.1 hsW
  obtain ⟨q, hq, rfl⟩ := List.mem_iff_getElem.1 hx
  refine ⟨q, by simpa using hq, ?_⟩
  rw [← Array.find?_toList]
  simp only [Array.getElem_toList] at hxs
  rw [hxs]
  exact find?_self_of_nodup StreamRec.name hnN hs

/-! ## The decoded public words are `liveOf`'s -/

section Publics

open Lean (Json)
open Claims.Bridge.Custody

/-- Table `t`'s public words as the record's link has them: empty where absent. -/
def PubField (pubs : Json) (t : String) (p : ByteArray) : Prop :=
  (Record.isNull pubs t = true → p = ByteArray.empty) ∧
    (Record.isNull pubs t = false → ∃ s, Record.str pubs t = .ok s ∧ unhex s = .ok p)

theorem pubField_unique {pubs : Json} {t : String} {a b : ByteArray} (ha : PubField pubs t a)
    (hb : PubField pubs t b) : a = b := by
  cases hn : Record.isNull pubs t
  · obtain ⟨s, h1, h2⟩ := ha.2 hn
    obtain ⟨s', h1', h2'⟩ := hb.2 hn
    rw [h1] at h1'
    cases h1'
    rw [h2] at h2'
    cases h2'
    rfl
  · rw [ha.1 hn, hb.1 hn]

theorem pubField_of {pubs : Json} {t : String} {p : ByteArray}
    (h : (if Record.isNull pubs t = true then pure ByteArray.empty else do unhex (← Record.str pubs t) :
      Flock.V ByteArray) = .ok p) : PubField pubs t p := by
  refine ⟨fun hn => ?_, fun hn => ?_⟩
  · rw [ite_eq_left hn] at h
    exact (Except.ok.inj h).symm
  · rw [ite_eq_right (by simp [hn])] at h
    obtain ⟨s, hs, h⟩ := ebind_ok h
    exact ⟨s, hs, h⟩

/-- **What `Record.decode` reads of the public words.** -/
theorem decode_pubs {spec : SessionSpec} {text : String} {sess : RunRecord} (h : Record.decode spec text = .ok sess) :
    ∃ j l pubs, Json.parse text = .ok j ∧ Record.fld j "link" = .ok l ∧ Record.fld l "publics" = .ok pubs ∧
      List.Forall₂ (PubField pubs) spec.tables.toList sess.publics.toList := by
  revert sess
  show Ret (Record.decode spec text) _
  rw [Record.decode.eq_def]
  cases hp : Json.parse text with
  | error e => intro sess h; cases h
  | ok j =>
    simp only []
    rw [pure_bind]
    ret_walk [Record.fld, Array.mapM]
    all_goals
      refine ⟨j, _, _, rfl, by assumption, by assumption, ?_⟩
      refine List.Forall₂.imp (fun t p h => ?_) (mapM_forall₂ (by assumption))
      beta_reduce at h
      split at h
      · rename_i hn
        rw [pure_bind] at h
        refine ⟨fun _ => ?_, fun hn' => by simp [hn] at hn'⟩
        split at h
        · obtain ⟨_, _, h⟩ := ebind_ok h
          obtain ⟨_, _, h⟩ := ebind_ok h
          exact (Except.ok.inj h).symm
        · exact (Except.ok.inj h).symm
      · rename_i hn
        obtain ⟨s, hs, h⟩ := ebind_ok h
        obtain ⟨p', hp', h⟩ := ebind_ok h
        refine ⟨fun hn' => absurd hn' hn, fun _ => ⟨s, hs, ?_⟩⟩
        split at h
        · obtain ⟨_, _, h⟩ := ebind_ok h
          obtain ⟨_, _, h⟩ := ebind_ok h
          rw [← Except.ok.inj h]
          exact hp'
        · rw [← Except.ok.inj h]
          exact hp'

/-- **What `liveOf` reads of the public words.** -/
theorem liveOf_pubs {spec : SessionSpec} {record : String} {lv : Live} (h : liveOf spec record = .ok lv) :
    ∃ j l pubs, Json.parse record = .ok j ∧ Record.fld j "link" = .ok l ∧ Record.fld l "publics" = .ok pubs ∧
      List.Forall₂ (PubField pubs) spec.tables.toList lv.publics.toList := by
  revert lv
  show Ret (liveOf spec record) _
  rw [liveOf.eq_def]
  cases hp : Json.parse record with
  | error e => intro lv h; cases h
  | ok j =>
    dsimp only
    rw [pure_bind]
    ret_walk [Record.fld, Array.mapM]
    refine ⟨j, _, _, rfl, by assumption, by assumption, ?_⟩
    exact List.Forall₂.imp (fun t p h => pubField_of h) (mapM_forall₂ (by assumption))

/-- **The decoded session's public words are `liveOf`'s.** -/
theorem decode_publics {spec : SessionSpec} {text : String} {sess : RunRecord} {lv : Live}
    (hd : Record.decode spec text = .ok sess) (hl : liveOf spec text = .ok lv) : sess.publics = lv.publics := by
  obtain ⟨j, l, pubs, hp, hl1, hp1, hf⟩ := decode_pubs hd
  obtain ⟨j', l', pubs', hp', hl1', hp1', hf'⟩ := liveOf_pubs hl
  rw [hp] at hp'
  cases hp'
  rw [hl1] at hl1'
  cases hl1'
  rw [hp1] at hp1'
  cases hp1'
  exact Array.ext' (forall₂_unique (R := PubField pubs) (fun _ _ _ => pubField_unique) hf hf')

end Publics

/-! ## The plain verdict's calls at a decoded session -/

theorem forall₂_mem_right {α β : Type} {R : α → β → Prop} :
    ∀ {l : List α} {l' : List β}, List.Forall₂ R l l' → ∀ b ∈ l', ∃ a ∈ l, R a b
  | _, _, .nil, _, h => by cases h
  | _, _, .cons hab hl, b, h => by
    rcases List.mem_cons.1 h with rfl | h
    · exact ⟨_, List.mem_cons_self .., hab⟩
    · obtain ⟨a, ha, hr⟩ := forall₂_mem_right hl b h
      exact ⟨a, List.mem_cons_of_mem _ ha, hr⟩

theorem f128_zero {c : F128} (h : c.isZero = true) : c = ⟨0, 0⟩ := by
  obtain ⟨lo, hi⟩ := c
  simp only [F128.isZero, Bool.and_eq_true, beq_iff_eq] at h
  rw [h.1, h.2]

/-- S10 pins `y`: zero, of the configured size. -/
theorem y_pinned {a b : Array F128} (hs : a.size = b.size) (ha : ∀ c ∈ a, c.isZero = true)
    (hb : ∀ c ∈ b, c.isZero = true) : a = b :=
  Array.ext hs fun _ h1 h2 => by rw [f128_zero (ha _ (Array.getElem_mem h1)), f128_zero (hb _ (Array.getElem_mem h2))]

theorem lbOf_congr {spec : SessionSpec} {sess sess' : RunRecord} (hy : sess.y = sess'.y)
    (hP : sess.points = sess'.points) (hB : sess.rootB = sess'.rootB) (hF : sess.rootF = sess'.rootF) :
    lbOf spec sess = lbOf spec sess' := by
  unfold lbOf
  rw [hy, hP, hB, hF]

theorem mem_dgl {spec : SessionSpec} {b p : ByteArray} : b ∈ dgl spec p ↔ spec.sha512Digests = true ∧ b = p := by
  unfold dgl
  split <;> simp_all

/-- `H(ε)` is among a session's round calls where the digests are SHA-512 and there is a table. -/
theorem eps_mem {spec : SessionSpec} {sess : RunRecord} (hd : spec.sha512Digests = true) (hne : 0 < spec.tables.size)
    (hf : ∀ x ∈ spec.repStreams.toList,
      ∃ s, sess.streams.find? (·.name == x.2.2) = some s ∧ 0 < s.rounds.size) :
    ByteArray.empty ∈ sess.streams.toList.flatMap (streamLog spec) := by
  have hq : 0 < spec.repStreams.size := by
    rw [FlockDeploy.repStreams_size]
    omega
  obtain ⟨s, hs, hr⟩ := hf _ (Array.getElem_mem_toList hq)
  refine List.mem_flatMap.2 ⟨s, Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hs), ?_⟩
  refine List.mem_flatMap.2 ⟨s.rounds[0], Array.getElem_mem_toList hr, ?_⟩
  simp [roundLog, dgl, hd]

/-- **The plain verdict's SHA-512 calls, where it accepts**, at the session it decodes: the decoder's (`decode_log`),
S17's digests of the files, and each table's two reps' own. -/
theorem verdict_calls {h : ByteArray → ByteArray} {spec : SessionSpec} {tabs : Array Setup} {record : String}
    {proofs : Array ByteArray} {L : Array ByteArray} (hv : verifyLogged h spec tabs record proofs = (.ok (), L)) :
    ∃ (sess : RunRecord) (schs : Array Schedule) (ds : List (List ByteArray)),
      Record.decodeWith (M := Flock.V) (Hashes.ofSha512 h) spec record = .ok sess ∧
      List.Forall₂ (fun st sch => fast100 st.m = some sch) tabs.toList schs.toList ∧
      tabs.size = spec.tables.size ∧
      List.Forall₂ (fun p d => d = dgl spec p ∨ d = [] ∧ p.size = 0) sess.publics.toList ds ∧
      sess.rootF = spec.sigma ∧ sess.y.size = spec.points * spec.tables.size ∧
      (∀ c ∈ sess.y, c.isZero = true) ∧ sess.streams.size = spec.repStreams.size ∧
      (∀ x ∈ spec.repStreams.toList,
        ∃ s, sess.streams.find? (·.name == x.2.2) = some s ∧ 0 < s.rounds.size) ∧
      L.toList = sess.rootB.toList.flatMap (dgl spec) ++ dgl spec sess.rootF ++ ds.flatten ++
        sess.streams.toList.flatMap (streamLog spec) ++
        spec.repStreams.toList.flatMap (fun x => dgl spec sess.rootB[x.1]!) ++ dgl spec (lbOf spec sess) ++
        (spec.repStreams.zip proofs).toList.flatMap (fun x => dgl spec x.2) ++
        (List.range tabs.size).flatMap fun i =>
          repLog h spec tabs[i]! schs[i]! sess i 0 proofs[2 * i]! ++
            repLog h spec tabs[i]! schs[i]! sess i 1 proofs[2 * i + 1]! := by
  obtain ⟨sess, schs, l₁, hd, hsch, ht, -, hL⟩ := verdict_log (l₀ := #[]) () L hv
  obtain ⟨ds, hl₁, hds, -, hF, hy, hy0, hss, hf⟩ := decode_log (l₀ := #[]) sess l₁ hd
  have hdec := (With.decodeWith_sim (With.Sim.forget _) (With.Sim.forget_logged h) spec record) #[]
  rw [hd] at hdec
  refine ⟨sess, schs, ds, hdec.symm, hsch, ht, hds, hF, hy, hy0, hss, hf, ?_⟩
  rw [hL, hl₁]
  simp only [List.nil_append, List.append_assoc]

/-- **Two accepting plain verdicts on one set of proof files call SHA-512 on the same inputs**, where the sessions they
decode have the same `root_B`, points and public words, and the same rounds at each rep stream: every call of the
first is a call of the second. -/
theorem verdict_calls_sub {spec : SessionSpec} {tabs : Array Setup} {record record' : String}
    {proofs : Array ByteArray} {L L' : Array ByteArray} {sess sess' : RunRecord}
    (hv : verifyLogged Sha512.hash spec tabs record proofs = (.ok (), L))
    (hv' : verifyLogged Sha512.hash spec tabs record' proofs = (.ok (), L'))
    (hd : Record.decode spec record = .ok sess) (hd' : Record.decode spec record' = .ok sess')
    (hn : spec.tables.toList.Nodup) (hB : sess.rootB = sess'.rootB) (hP : sess.points = sess'.points)
    (hpub : sess.publics = sess'.publics)
    (hR : ∀ q (hq : q < spec.repStreams.size) (y y' : StreamRec),
      sess.streams.find? (·.name == spec.repStreams[q].2.2) = some y →
        sess'.streams.find? (·.name == spec.repStreams[q].2.2) = some y' → y.rounds = y'.rounds) :
    ∀ b ∈ L.toList, b ∈ L'.toList := by
  obtain ⟨s₁, schs, ds, hd₁, hsch, ht, hds, hF, hy, hy0, hss, hf, hL⟩ := verdict_calls hv
  obtain ⟨s₂, schs', ds', hd₂, hsch', -, hds', hF', hy', hy0', -, hf', hL'⟩ := verdict_calls hv'
  obtain rfl := Except.ok.inj (hd₁.symm.trans ((With.decodeWith_std spec record).trans hd))
  obtain rfl := Except.ok.inj (hd₂.symm.trans ((With.decodeWith_std spec record').trans hd'))
  have hsc : schs = schs' := Array.ext' (Claims.Bridge.Custody.forall₂_unique
    (R := fun (st : Setup) sch => fast100 st.m = some sch) (fun _ _ _ h1 h2 => Option.some.inj (h1.symm.trans h2))
    hsch hsch')
  subst hsc
  have hlb : lbOf spec s₁ = lbOf spec s₂ :=
    lbOf_congr (y_pinned (hy.trans hy'.symm) hy0 hy0') hP hB (hF.trans hF'.symm)
  have hrep : ∀ i < tabs.size, ∀ r < 2, ∀ bytes, repLog Sha512.hash spec tabs[i]! schs[i]! s₁ i r bytes =
      repLog Sha512.hash spec tabs[i]! schs[i]! s₂ i r bytes := by
    intro i hi r hr bytes
    obtain ⟨hq, e⟩ := Claims.Bridge.Custody.repStreams_getElem spec (ht ▸ hi) hr
    obtain ⟨y, hy₁, -⟩ := hf _ (Array.getElem_mem_toList hq)
    obtain ⟨y', hy₂, -⟩ := hf' _ (Array.getElem_mem_toList hq)
    have hyr := hR _ hq y y' hy₁ hy₂
    rw [e] at hy₁ hy₂
    unfold repLog
    rw [verifyRepWith_congr (Hashes.logged Sha512.hash) _ hy₁ hy₂ hyr hP (by rw [hB])]
  have hreps : ((List.range tabs.size).flatMap fun i =>
      repLog Sha512.hash spec tabs[i]! schs[i]! s₁ i 0 proofs[2 * i]! ++
        repLog Sha512.hash spec tabs[i]! schs[i]! s₁ i 1 proofs[2 * i + 1]!) =
      (List.range tabs.size).flatMap fun i =>
        repLog Sha512.hash spec tabs[i]! schs[i]! s₂ i 0 proofs[2 * i]! ++
          repLog Sha512.hash spec tabs[i]! schs[i]! s₂ i 1 proofs[2 * i + 1]! := by
    refine List.flatMap_congr fun i hi => ?_
    rw [List.mem_range] at hi
    rw [hrep i hi 0 (by omega), hrep i hi 1 (by omega)]
  have hstr : ∀ s ∈ s₁.streams.toList, ∃ s' ∈ s₂.streams.toList, streamLog spec s = streamLog spec s' := by
    intro s hs
    obtain ⟨q, hq, hsq⟩ := streams_pinned (repNames_nodup hn) hss (fun x hx => (hf x hx).imp fun _ h => h.1) s hs
    obtain ⟨s', hs', -⟩ := hf' _ (Array.getElem_mem_toList hq)
    refine ⟨s', Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hs'), ?_⟩
    unfold streamLog
    rw [hR q hq s s' hsq hs']
  rw [hL, hL', ← hB, ← hlb, ← hreps, hF, hF']
  intro b hb
  simp only [List.mem_append] at hb ⊢
  rcases hb with ((((((hb | hb) | hb) | hb) | hb) | hb) | hb) | hb
  · exact .inl (.inl (.inl (.inl (.inl (.inl (.inl hb))))))
  · exact .inl (.inl (.inl (.inl (.inl (.inl (.inr hb))))))
  · obtain ⟨d, hd, hbd⟩ := List.mem_flatten.1 hb
    obtain ⟨p, hp, hpd⟩ := forall₂_mem_right hds d hd
    rcases hpd with rfl | ⟨rfl, -⟩
    · obtain ⟨hsha, rfl⟩ := mem_dgl.1 hbd
      rw [hpub] at hp
      obtain ⟨d', hd'm, hpd'⟩ := Claims.Bridge.Custody.forall₂_mem_left hds' b hp
      rcases hpd' with rfl | ⟨-, hz⟩
      · exact .inl (.inl (.inl (.inl (.inl (.inr (List.mem_flatten.2 ⟨_, hd'm, hbd⟩))))))
      · have he : b = ByteArray.empty := ByteArray.ext (Array.eq_empty_of_size_eq_zero hz)
        subst he
        refine .inl (.inl (.inl (.inl (.inr (eps_mem hsha ?_ hf')))))
        obtain ⟨-, -, pubs, -, -, -, hpf⟩ := decode_pubs hd'
        have := hpf.length_eq
        rw [Array.length_toList, Array.length_toList] at this
        have := List.length_pos_of_mem hp
        rw [Array.length_toList] at this
        omega
    · cases hbd
  · obtain ⟨s, hs, hbs⟩ := List.mem_flatMap.1 hb
    obtain ⟨s', hs', e⟩ := hstr s hs
    exact .inl (.inl (.inl (.inl (.inr (List.mem_flatMap.2 ⟨s', hs', e ▸ hbs⟩)))))
  · exact .inl (.inl (.inl (.inr hb)))
  · exact .inl (.inl (.inr hb))
  · exact .inl (.inr hb)
  · exact .inr hb

/-! ## The decision's calls -/

/-- Every entry of an oracle program's log is a query with `H`'s answer. -/
theorem eval_log_ans (H : Claims.Msg → Claims.Msg) {α : Type} (x : Claims.OC α) :
    ∀ p ∈ (x.eval H).2, p.2 = H p.1 := by
  induction x with
  | ret a => simp [Claims.OC.eval]
  | query q k ih =>
    intro p hp
    simp only [Claims.OC.eval, List.mem_cons] at hp
    rcases hp with rfl | hp
    · rfl
    · exact ih _ p hp

theorem ofMsg_inj {a b : Claims.Msg} (h : Claims.ofMsg a = Claims.ofMsg b) : a = b := by
  have := congrArg Claims.toMsg h
  simpa [Claims.toMsg, Claims.ofMsg] using this

/-- Two logs of `H`'s answers with the same queries are one set. -/
theorem toFinset_eq_of_queries {H : Claims.Msg → Claims.Msg} {L L' : List (Claims.Msg × Claims.Msg)}
    (hL : ∀ p ∈ L, p.2 = H p.1) (hL' : ∀ p ∈ L', p.2 = H p.1) (h : ∀ p ∈ L, ∃ p' ∈ L', p'.1 = p.1)
    (h' : ∀ p ∈ L', ∃ p' ∈ L, p'.1 = p.1) : L.toFinset = L'.toFinset := by
  have key : ∀ {A B : List (Claims.Msg × Claims.Msg)}, (∀ p ∈ A, p.2 = H p.1) → (∀ p ∈ B, p.2 = H p.1) →
      (∀ p ∈ A, ∃ p' ∈ B, p'.1 = p.1) → ∀ p ∈ A, p ∈ B := by
    intro A B hA hB hAB p hp
    obtain ⟨p', hp', e⟩ := hAB p hp
    have : p' = p := Prod.ext e (by rw [hB p' hp', hA p hp, e])
    exact this ▸ hp'
  ext p
  simp only [List.mem_toFinset]
  exact ⟨key hL hL' h p, key hL' hL h' p⟩

/-- The decision at a session logs its verdict's calls: the checks call nothing. -/
theorem decisionAt_log (H : Claims.Msg → Claims.Msg) (zk : Bool) (s : Ins) (spec : SessionSpec) (sts : Array Stmt)
    (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin))
    (f : Final) :
    ((decisionAt zk s spec sts x link c₁ mids f).run.eval H).2 =
      ((verdictM zk Claims.Hashes.oc spec (tabsOf sts) f.record f.proofs).run.eval H).2 := by
  simp only [decisionAt, ExceptT.run_bind, Claims.OC.bind_eq, Claims.OC.eval_bind]
  rcases hv : (verdictM zk Claims.Hashes.oc spec (tabsOf sts) f.record f.proofs).run.eval H with ⟨r, l⟩
  cases r <;> exact List.append_nil l

/-- **A step that decides logs the setup's calls, then the verdict's**, where the setup is `.ok`. -/
theorem next_decide_calls (H : Claims.Msg → Claims.Msg) {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg}
    {cs : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray}
    {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (h : act zk x as cs = .decide s spec sts link c₁ mids f)
    (hs : ((setupM Claims.Hashes.oc s.read).run.eval H).1 = .ok (spec, sts)) :
    ((next zk x as cs).eval H).2 = ((setupM Claims.Hashes.oc s.read).run.eval H).2 ++
      ((verdictM zk Claims.Hashes.oc spec (tabsOf sts) f.record f.proofs).run.eval H).2 := by
  simp only [next, h, outOf_log, decision, ExceptT.run_bind, Claims.OC.bind_eq, Claims.OC.eval_bind]
  rcases hv : (setupM Claims.Hashes.oc s.read).run.eval H with ⟨r, l⟩
  rw [hv] at hs
  simp only at hs
  subst hs
  show l ++ ((decisionAt zk s spec sts x link c₁ mids f).run.eval H).2 = _
  rw [decisionAt_log]

/-- **An accepting plain step at SHA-512** has a logged verdict that accepts, and its checks pass. -/
theorem acc_verifyLogged {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)}
    {f : Final} (h : act false x as cs = .decide s spec sts link c₁ mids f)
    (hacc : ((next false x as cs).eval sha512).1 = .acc) :
    verifyLogged Sha512.hash spec (tabsOf sts) f.record f.proofs =
        (.ok (), (verifyLogged Sha512.hash spec (tabsOf sts) f.record f.proofs).2) ∧
      checks false s spec x link c₁ mids f = .ok () := by
  obtain ⟨spec', sts', he, hv, hc⟩ := (next_decide sha512 h).1 hacc
  rw [act_decide_setupWith h] at he
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj he)
  rw [hOf_sha512] at hv
  exact ⟨Prod.ext ((verifyLogged_fst _ _ _ _ _ _).trans hv) rfl, hc⟩

/-- **Two plain steps that decide on one setup call SHA-512 on one set**, where each verdict's calls are among the
other's: the decision's calls are the setup's, then the verdict's. -/
theorem next_calls_toFinset {x : Claims.Msg} {as as' : List Claims.Msg} {cs cs' : List Coin} {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link link' : List ByteArray × List ByteArray} {c₁ c₁' : Coin}
    {mids mids' : List (Claims.Msg × Coin)} {f f' : Final}
    (h : act false x as cs = .decide s spec sts link c₁ mids f)
    (h' : act false x as' cs' = .decide s spec sts link' c₁' mids' f')
    (hsub : ∀ b ∈ (verifyLogged Sha512.hash spec (tabsOf sts) f.record f.proofs).2.toList,
      b ∈ (verifyLogged Sha512.hash spec (tabsOf sts) f'.record f'.proofs).2.toList)
    (hsub' : ∀ b ∈ (verifyLogged Sha512.hash spec (tabsOf sts) f'.record f'.proofs).2.toList,
      b ∈ (verifyLogged Sha512.hash spec (tabsOf sts) f.record f.proofs).2.toList) :
    ((next false x as cs).eval sha512).2.toFinset = ((next false x as' cs').eval sha512).2.toFinset := by
  have hs := (setupM_eval sha512 s.read).1.trans (act_decide_setupWith h)
  rw [next_decide_calls sha512 h hs, next_decide_calls sha512 h' hs, List.toFinset_append, List.toFinset_append]
  congr 1
  have hl := Claims.verifyOC_log sha512 spec (tabsOf sts) f.record f.proofs
  have hl' := Claims.verifyOC_log sha512 spec (tabsOf sts) f'.record f'.proofs
  rw [hOf_sha512] at hl hl'
  have hq : ∀ {A B : List (Claims.Msg × Claims.Msg)} {LA LB : Array ByteArray},
      A.map (fun p => Claims.ofMsg p.1) = LA.toList → B.map (fun p => Claims.ofMsg p.1) = LB.toList →
      (∀ b ∈ LA.toList, b ∈ LB.toList) → ∀ p ∈ A, ∃ p' ∈ B, p'.1 = p.1 := by
    intro A B LA LB hA hB hAB p hp
    have h1 : Claims.ofMsg p.1 ∈ LA.toList := by
      rw [← hA]
      exact List.mem_map_of_mem hp
    have h2 := hAB _ h1
    rw [← hB] at h2
    obtain ⟨p', hp', e⟩ := List.mem_map.1 h2
    exact ⟨p', hp', ofMsg_inj e⟩
  exact toFinset_eq_of_queries (eval_log_ans _ _) (eval_log_ans _ _) (hq hl hl' hsub) (hq hl' hl hsub')

/-- **Two accepted plain last messages after the same rounds and coins decide on one setup, link, coin and round
messages.** -/
theorem plain_decide_shared {x : Claims.Msg} {ms : List Claims.Msg} {l l' : Claims.Msg} {cs : List Coin}
    {f f' : Final} (hc : cs.length = ms.length + 1)
    (hacc : ((next false x (ms ++ [l]) cs).eval sha512).1 = .acc)
    (hacc' : ((next false x (ms ++ [l']) cs).eval sha512).1 = .acc)
    (hfl : finalOf l = some f) (hfl' : finalOf l' = some f') :
    ∃ s spec sts link c₁ mids, act false x (ms ++ [l]) cs = .decide s spec sts link c₁ mids f ∧
      act false x (ms ++ [l']) cs = .decide s spec sts link c₁ mids f' := by
  obtain ⟨s, spec, sts, link, c₁, mids, f₀, hact, -⟩ := next_acc_decide hacc
  obtain ⟨s', spec', sts', link', c₁', mids', f₀', hact', -⟩ := next_acc_decide hacc'
  have hs := act_decide hact
  have hs' := act_decide hact'
  rw [hs] at hs'
  cases hs'
  have hlen := act_decide_length (by simp [hc]) hact
  obtain ⟨b, ms'', rfl⟩ : ∃ b ms'', ms = b :: ms'' := by
    cases ms with
    | nil =>
      simp only [List.nil_append, List.length_singleton] at hlen
      omega
    | cons b ms'' => exact ⟨b, ms'', rfl⟩
  obtain ⟨a₁, rest, crest, has, hcs, hlink, hmids⟩ := act_decide_parts hact
  obtain ⟨a₁', rest', crest', has', hcs', hlink', hmids'⟩ := act_decide_parts hact'
  simp only [List.cons_append, List.cons.injEq] at has has'
  obtain ⟨rfl, rfl⟩ := has
  obtain ⟨rfl, rfl⟩ := has'
  obtain ⟨rfl, rfl⟩ : c₁' = c₁ ∧ crest' = crest := by
    rw [hcs] at hcs'
    simp only [List.cons.injEq] at hcs'
    exact ⟨hcs'.1.symm, hcs'.2.symm⟩
  rw [hlink] at hlink'
  cases hlink'
  have hm : mids' = mids := by rw [hmids, hmids', List.dropLast_concat, List.dropLast_concat]
  subst hm
  have e : finalOf l = some f₀ := by
    have e := act_decide_final hact
    rw [List.getLast?_concat] at e
    exact e
  have e' : finalOf l' = some f₀' := by
    have e := act_decide_final hact'
    rw [List.getLast?_concat] at e
    exact e
  obtain rfl := Option.some.inj (e.symm.trans hfl)
  obtain rfl := Option.some.inj (e'.symm.trans hfl')
  exact ⟨_, _, _, _, _, _, hact, hact'⟩

/-- **Two accepted plain decisions on one setup, link, coin and round messages, with the same proof files, call
SHA-512 on one set**, where the sessions their records decode have the same rounds at each rep stream. Custody pins
both records' `root_B`, points and public words to the link and the first coin; S10 pins `y`; the statement's tables
are named apart (`tables_nodup`), so S11 leaves no stream but the rep streams (`verdict_calls_sub`). -/
theorem plain_calls_agree {x : Claims.Msg} {as as' : List Claims.Msg} {cs cs' : List Coin} {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f f' : Final}
    (h : act false x as cs = .decide s spec sts link c₁ mids f)
    (h' : act false x as' cs' = .decide s spec sts link c₁ mids f')
    (hacc : ((next false x as cs).eval sha512).1 = .acc) (hacc' : ((next false x as' cs').eval sha512).1 = .acc)
    (hpf : f.proofs = f'.proofs)
    (hR : ∀ sess sess', Record.decode spec f.record = .ok sess → Record.decode spec f'.record = .ok sess' →
      ∀ q (hq : q < spec.repStreams.size) (y y' : StreamRec),
        sess.streams.find? (·.name == spec.repStreams[q].2.2) = some y →
          sess'.streams.find? (·.name == spec.repStreams[q].2.2) = some y' → y.rounds = y'.rounds) :
    ((next false x as cs).eval sha512).2.toFinset = ((next false x as' cs').eval sha512).2.toFinset := by
  obtain ⟨hV, hck⟩ := acc_verifyLogged h hacc
  obtain ⟨hV', hck'⟩ := acc_verifyLogged h' hacc'
  rw [hpf] at hV
  obtain ⟨sess, -, -, hd₀, -⟩ := verdict_calls hV
  obtain ⟨sess', -, -, hd₀', -⟩ := verdict_calls hV'
  have hd : Record.decode spec f.record = .ok sess := (With.decodeWith_std spec f.record).symm.trans hd₀
  have hd' : Record.decode spec f'.record = .ok sess' := (With.decodeWith_std spec f'.record).symm.trans hd₀'
  obtain ⟨-, lv, hlv, hcu⟩ := Claims.Bridge.Custody.checks_ok hck
  obtain ⟨-, lv', hlv', hcu'⟩ := Claims.Bridge.Custody.checks_ok hck'
  obtain ⟨hrB, hpts, -⟩ := Claims.Bridge.Custody.decode_live hd hlv
  obtain ⟨hrB', hpts', -⟩ := Claims.Bridge.Custody.decode_live hd' hlv'
  obtain ⟨hlB, hlU, hlP, -⟩ := Claims.Bridge.Custody.custody_ok hcu
  obtain ⟨hlB', hlU', hlP', -⟩ := Claims.Bridge.Custody.custody_ok hcu'
  have hP : sess.points = sess'.points := by rw [hpts, hpts', hlP, hlP']
  have hB : sess.rootB = sess'.rootB := by
    rw [hrB, hrB']
    exact Array.ext' (hlB.trans hlB'.symm)
  have hU : sess.publics = sess'.publics := by
    rw [decode_publics hd hlv, decode_publics hd' hlv']
    exact Array.ext' (hlU.trans hlU'.symm)
  have hn := tables_nodup (setupOf_some (act_decide h)).2 (Claims.Bridge.Custody.publicInputs_none_of_checks hck)
  have hR' := hR sess sess' hd hd'
  refine next_calls_toFinset h h' ?_ ?_ <;> rw [hpf]
  · exact verdict_calls_sub hV hV' hd hd' hn hB hP hU hR'
  · exact verdict_calls_sub hV' hV hd' hd hn hB.symm hP.symm hU.symm
      fun q hq y y' hy hy' => (hR' q hq y' y hy' hy).symm

end FlockDeploy.KS
