import Proofs.Flock.Claims.Recursion.LastColsDeployed
import Proofs.Flock.Claims.Recursion.RepBindProof

/-!
# The calls half where a proof file differs

`RecDeploy.PlainCallsBindOrSlack` compares the calls of two decisions the deployed plain verifier accepts after the
same rounds and coins. They split at their proof files:

* where the files agree, the two decisions' calls are one set (`RecDeploy.PlainCallsOfProofs`, proved at SHA-512 in
  `CallsOfProofs`);
* where a file differs, custody pins both records' rounds to the same round messages at the counts the statement fixes,
  so the two sessions have the same rounds on every rep stream, the same points and the same `root_B`
  (`sess_rounds_eq`). The two files are then accepted as one rep on one session (`verifyRep_congr`), and
  `rep_bind_proof` gives one proof, which is `RecDeploy.ProofSlack`, or two of their openings that collide. Every
  SHA-512 input of each opening is among its decision's calls (`rep_level_logs`), so the collision is among the calls.

`RecDeploy.plainCallsBindOrSlack_of_proofs` reduces `PlainCallsBindOrSlack` to `PlainCallsOfProofs`.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness.Refine

/-! ## Two records under one custody have one session's rounds -/

/-- A round's coin count, coins and retained bytes. -/
def roundKey (rd : Round) : ℕ × Array F128 × Option ByteArray := (rd.n, rd.coins, rd.bytes)

/-- Streams of the same sizes whose rounds, read in order, agree under `g` agree under `g` stream by stream. -/
theorem streams_eq_of_flat {β : Type} (g : Round → β) : ∀ {S S' : List (Array Round)},
    S.map (fun s => s.size) = S'.map (fun s => s.size) →
    (S.flatMap (·.toList)).map g = (S'.flatMap (·.toList)).map g →
    S.map (fun s => s.toList.map g) = S'.map (fun s => s.toList.map g)
  | [], [], _, _ => rfl
  | [], _ :: _, hl, _ => by simp at hl
  | _ :: _, [], hl, _ => by simp at hl
  | a :: S, a' :: S', hl, hf => by
    simp only [List.map_cons, List.cons.injEq] at hl
    simp only [List.flatMap_cons, List.map_append] at hf
    have hlen : (a.toList.map g).length = (a'.toList.map g).length := by simp [hl.1]
    obtain ⟨h₁, h₂⟩ := List.append_inj hf hlen
    simp only [List.map_cons, h₁, streams_eq_of_flat g hl.2 h₂]

theorem keys_of_forall₂ : ∀ {rs : List Round} {ms : List (Claims.Msg × Coin)},
    List.Forall₂ (fun rd am => rd.bytes = some ⟨am.1.toArray⟩ ∧ rd.coins = am.2.vals rd.n) rs ms →
      rs.map roundKey =
        List.zipWith (fun n (am : Claims.Msg × Coin) => (n, am.2.vals n, some (⟨am.1.toArray⟩ : ByteArray)))
          (rs.map Round.n) ms
  | [], [], .nil => rfl
  | rd :: _, am :: _, .cons hab h => by
    rw [List.map_cons, List.map_cons, List.zipWith_cons_cons, keys_of_forall₂ h]
    show (rd.n, rd.coins, rd.bytes) :: _ = _
    rw [hab.1, hab.2]

/-- **Two accepted plain records under custody of the same round messages carry the same rounds' counts, coins and
bytes, stream by stream**, whatever hashes each verdict ran with. -/
theorem live_keys_eq (hs hs' : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt} {record record' : String}
    {proofs proofs' : Array ByteArray} {lv lv' : Live} {link link' : List ByteArray × List ByteArray} {c₁ c₁' : Coin}
    {mids : List (Claims.Msg × Coin)}
    (hv : verifyWithM hs spec (tabsOf sts) record proofs = .ok ()) (hl : liveOf spec record = .ok lv)
    (hv' : verifyWithM hs' spec (tabsOf sts) record' proofs' = .ok ()) (hl' : liveOf spec record' = .ok lv')
    (hc : custody spec lv link c₁ mids = .ok ()) (hc' : custody spec lv' link' c₁' mids = .ok ()) :
    lv.streams.toList.map (fun s => s.toList.map roundKey) =
      lv'.streams.toList.map (fun s => s.toList.map roundKey) := by
  refine streams_eq_of_flat roundKey
    ((streams_sizes_plain hs hv hl).trans (streams_sizes_plain hs' hv' hl').symm) ?_
  rw [keys_of_forall₂ (Claims.Bridge.Custody.custody_ok hc).2.2.2,
    keys_of_forall₂ (Claims.Bridge.Custody.custody_ok hc').2.2.2, rounds_ns_plain hs hv hl,
    rounds_ns_plain hs' hv' hl']

theorem keys_at {S S' : Array (Array Round)}
    (hS : S.toList.map (fun s => s.toList.map roundKey) = S'.toList.map (fun s => s.toList.map roundKey))
    (q : ℕ) : (S[q]!).toList.map roundKey = (S'[q]!).toList.map roundKey := by
  have hsz : S.size = S'.size := by simpa using congrArg List.length hS
  by_cases hq : q < S.size
  · have hq' : q < S'.size := by omega
    have e := congrArg (fun L => L[q]?) hS
    simp only [List.getElem?_map, Array.getElem?_toList, Array.getElem?_eq_getElem hq,
      Array.getElem?_eq_getElem hq', Option.map_some] at e
    rw [getElem!_pos S q hq, getElem!_pos S' q hq']
    exact Option.some.inj e
  · rw [getElem!_neg S q hq, getElem!_neg S' q (by omega)]

theorem forall₂_left {α β : Type} {R : α → β → Prop} :
    ∀ {l₁ : List α} {l₂ : List β}, List.Forall₂ R l₁ l₂ → ∀ x ∈ l₁, ∃ y, R x y
  | _, _, .nil, _, h => by simp at h
  | _, _, .cons hab h, x, hx => by
    rcases List.mem_cons.1 hx with rfl | hx
    · exact ⟨_, hab⟩
    · exact forall₂_left h x hx

/-- Every round of an accepted record's live session retained its bytes. -/
theorem live_retained {spec : SessionSpec} {lv : Live} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} (hc : custody spec lv link c₁ mids = .ok ()) (q : ℕ) :
    ∀ rd ∈ (lv.streams[q]!).toList, ∃ B, rd.bytes = some B := by
  intro rd hrd
  have hflat : rd ∈ lv.streams.toList.flatMap (·.toList) := by
    by_cases hq : q < lv.streams.size
    · rw [getElem!_pos lv.streams q hq] at hrd
      exact List.mem_flatMap.2 ⟨lv.streams[q], Array.mem_toList_iff.2 (Array.getElem_mem hq), hrd⟩
    · rw [getElem!_neg lv.streams q hq] at hrd
      cases hrd
  obtain ⟨am, hb, -⟩ := forall₂_left (Claims.Bridge.Custody.custody_ok hc).2.2.2 rd hflat
  exact ⟨_, hb⟩

/-- How the verifier's decoder reads a round that `liveOf` reads (`Claims.Bridge.Custody.decode_live`). -/
abbrev DecRound (spec : SessionSpec) (rd rd' : Round) : Prop :=
  rd.n = rd'.n ∧ rd.coins = rd'.coins ∧ rd.bytes = rd'.bytes ∧ ∀ b, rd.bytes = some b → rd.sha = spec.dg b

theorem retained_of {spec : SessionSpec} : ∀ {a c : List Round}, List.Forall₂ (DecRound spec) a c →
    (∀ rd ∈ c, ∃ B, rd.bytes = some B) → ∀ rd ∈ a, ∃ B, rd.bytes = some B
  | _, _, .nil, _, _, h => by simp at h
  | _, _, .cons hab h, hr, rd, hm => by
    rcases List.mem_cons.1 hm with rfl | hm
    · obtain ⟨B, hB⟩ := hr _ (List.mem_cons_self ..)
      exact ⟨B, hab.2.2.1.trans hB⟩
    · exact retained_of h (fun x hx => hr x (List.mem_cons_of_mem _ hx)) rd hm

/-- Two decoded round lists over retained rounds with the same counts, coins and bytes are one list: a retained round's
digest is its bytes'. -/
theorem rounds_eq_of_keys {spec : SessionSpec} : ∀ {a b c d : List Round},
    List.Forall₂ (DecRound spec) a c → List.Forall₂ (DecRound spec) b d → c.map roundKey = d.map roundKey →
    (∀ rd ∈ c, ∃ B, rd.bytes = some B) → a = b
  | _, _, _, _, .nil, .nil, _, _ => rfl
  | _, _, _, _, .nil, .cons _ _, hk, _ => by simp at hk
  | _, _, _, _, .cons _ _, .nil, hk, _ => by simp at hk
  | x :: _, y :: _, u :: _, v :: _, .cons hx ha, .cons hy hb, hk, hret => by
    simp only [List.map_cons, List.cons.injEq] at hk
    obtain ⟨hk1, hk2⟩ := hk
    obtain ⟨B, hB⟩ := hret u (List.mem_cons_self ..)
    have e : x = y := by
      obtain ⟨hn₁, hc₁, hb₁, hs₁⟩ := hx
      obtain ⟨hn₂, hc₂, hb₂, hs₂⟩ := hy
      simp only [roundKey, Prod.mk.injEq] at hk1
      obtain ⟨kn, kc, kb⟩ := hk1
      have hxb : x.bytes = some B := hb₁.trans hB
      have hyb : y.bytes = some B := hb₂.trans (kb.symm.trans hB)
      have hxs := hs₁ B hxb
      have hys := hs₂ B hyb
      cases x
      cases y
      simp only at hn₁ hc₁ hxb hn₂ hc₂ hyb hxs hys ⊢
      rw [hn₁, hc₁, hxb, hxs, hn₂, hc₂, hyb, hys, kn, kc]
    rw [e, rounds_eq_of_keys ha hb hk2 (fun rd h => hret rd (List.mem_cons_of_mem _ h))]

/-- **Two accepted plain records under custody of the same link and round messages decode to sessions with the same
rounds on every rep stream**, all retained. -/
theorem sess_rounds_eq (hs hs' : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt} {record record' : String}
    {proofs proofs' : Array ByteArray} {lv lv' : Live} {link link' : List ByteArray × List ByteArray} {c₁ c₁' : Coin}
    {mids : List (Claims.Msg × Coin)} {sess sess' : RunRecord}
    (hv : verifyWithM hs spec (tabsOf sts) record proofs = .ok ()) (hl : liveOf spec record = .ok lv)
    (hv' : verifyWithM hs' spec (tabsOf sts) record' proofs' = .ok ()) (hl' : liveOf spec record' = .ok lv')
    (hc : custody spec lv link c₁ mids = .ok ()) (hc' : custody spec lv' link' c₁' mids = .ok ())
    (hd : Record.decode spec record = .ok sess) (hd' : Record.decode spec record' = .ok sess')
    {q : ℕ} (hq : q < spec.repStreams.size) :
    ∃ y y' : StreamRec, sess.streams.find? (·.name == spec.repStreams[q].2.2) = some y ∧
      sess'.streams.find? (·.name == spec.repStreams[q].2.2) = some y' ∧ y.rounds = y'.rounds ∧
      AllRetained y.rounds := by
  obtain ⟨-, -, -, hst⟩ := Claims.Bridge.Custody.decode_live hd hl
  obtain ⟨-, -, -, hst'⟩ := Claims.Bridge.Custody.decode_live hd' hl'
  obtain ⟨y, hy, hfa⟩ := hst q hq
  obtain ⟨y', hy', hfa'⟩ := hst' q hq
  have hk := keys_at (live_keys_eq hs hs' hv hl hv' hl' hc hc') q
  have hr := live_retained hc q
  have e : y.rounds.toList = y'.rounds.toList := rounds_eq_of_keys hfa hfa' hk hr
  refine ⟨y, y', hy, hy', Array.ext' e, fun j hj => ?_⟩
  rw [getElem!_pos y.rounds j hj]
  exact retained_of hfa hr _ (Array.mem_toList_iff.2 (Array.getElem_mem hj))

/-! ## One rep on two sessions with one stream -/

/-- **A rep reads its session only through its stream's rounds, the points and its table's `root_B`.** -/
theorem verifyRep_congr {spec : SessionSpec} {st : Setup} {sch : Flock.Schedule} {sess sess' : RunRecord} {i r : ℕ}
    (bytes : ByteArray) {y y' : StreamRec}
    (hy : sess.streams.find? (fun x => x.name == s!"{spec.tables[i]!}/rep{r}") = some y)
    (hy' : sess'.streams.find? (fun x => x.name == s!"{spec.tables[i]!}/rep{r}") = some y')
    (hr : y.rounds = y'.rounds) (hP : sess.points = sess'.points) (hB : sess.rootB[i]! = sess'.rootB[i]!) :
    verifyRep spec st sch sess i r bytes = verifyRep spec st sch sess' i r bytes := by
  unfold verifyRep
  dsimp only
  rw [hy, hy']
  dsimp only
  rw [hP, hB, hr]

/-! ## Every opening's inputs are among the calls -/

/-- **An accepted plain rep passed every level's Merkle check through the oracle `hs`**: `rep_levels_check` before
the oracle is replaced. -/
theorem rep_levels_checkWith (hs : Hashes Flock.V) {spec : SessionSpec} {st : Setup} {sch : Flock.Schedule}
    {sess : RunRecord} {i r : ℕ} {bytes cap : ByteArray} (hsch : fast100 st.m = some sch)
    (hf : ∀ a b lw rho ts v, st.fold.folded a b lw rho ts = .ok v → rho.size + 6 = st.fold.kLog)
    (hkl : st.fold.kLog = st.kLog) (h : verifyRepWith hs spec st sch sess i r bytes = .ok cap) :
    ∃ (stream : StreamRec) (extras : Array Flock.Claim) (p : Proof),
      sess.streams.find? (fun x => x.name == s!"{spec.tables[i]!}/rep{r}") = some stream ∧
      st.extra sess.points = .ok extras ∧ decodeProof (shapeOf st sch extras.size) bytes = .ok p ∧
      ∀ ℓ < sch.levels.size, merkleCheckWith (st.merkle.opsWith hs) st.merkle (capAt p.cap p.ligerito ℓ)
        (sch.levels.getD ℓ default).cap (sch.levels.getD ℓ default).d
        (positions (sch.levels.getD ℓ default) (stream.rounds[repRound st sch extras.size ℓ]!).coins)
        (openAt p.ligerito (sch.levels.size - 1) ℓ).1 (openAt p.ligerito (sch.levels.size - 1) ℓ).2.1
        (openAt p.ligerito (sch.levels.size - 1) ℓ).2.2 (2 ^ (sch.levels.getD ℓ default).lanes) = .ok () := by
  have h2 : 2 ≤ sch.levels.size := (fast100_dims _ _ hsch).1.2.1
  obtain ⟨stream, extras, p, hstr, hex, hp, -, -, hmc0, -⟩ :=
    plainRepColsOf_verifyRepWith hs spec st sch sess i r bytes cap hsch hf hkl h
  obtain ⟨stream₁, extras₁, p₁, hstr₁, hex₁, hp₁, hlev⟩ :=
    plainLevels_verifyRepWith hs spec st sch sess i r bytes cap hsch hf hkl h
  rw [hstr] at hstr₁
  cases hstr₁
  rw [hex] at hex₁
  cases hex₁
  rw [hp] at hp₁
  cases hp₁
  refine ⟨stream, extras, p, hstr, hex, hp, fun ℓ hℓ => ?_⟩
  rcases Nat.eq_zero_or_pos ℓ with rfl | h1
  · have e : repRound st sch extras.size 0 = Claims.Flock.plainQueryRound st.m st.kLog sch.k0 (2 + extras.size) := by
      simp only [repRound, plainLevelRound, levRound, Claims.Flock.plainQueryRound,
        show 0 < sch.levels.size - 1 by omega, ↓reduceIte]
    rw [e]
    exact hmc0
  · have hmc := (hlev ℓ h1 hℓ).2
    have e : capAt p.cap p.ligerito ℓ = p.ligerito.recursiveCaps[ℓ - 1]! :=
      ite_eq_right_iff.2 fun h0 => absurd h0 (by omega)
    rw [e]
    exact hmc

/-- **Every SHA-512 input of an accepted plain rep's openings is among the calls `L`**, when the rep ran with SHA-512
answered only within `L`: at every level `ℓ` and query `j`, the inputs of the proof's opening (`levelIns`). -/
theorem rep_level_logs {L : List Claims.Msg} {spec : SessionSpec} {st : Setup} {sch : Flock.Schedule}
    {sess : RunRecord} {i r : ℕ} {bytes cap : ByteArray} (hsch : fast100 st.m = some sch)
    (hf : ∀ a b lw rho ts v, st.fold.folded a b lw rho ts = .ok v → rho.size + 6 = st.fold.kLog)
    (hkl : st.fold.kLog = st.kLog)
    (hms : st.merkle = MerkleScheme.hm96Sha512 ∨ st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted")
    (h : verifyRepWith (hashesIn L) spec st sch sess i r bytes = .ok cap) :
    ∃ (stream : StreamRec) (extras : Array Flock.Claim) (p : Proof),
      sess.streams.find? (fun x => x.name == s!"{spec.tables[i]!}/rep{r}") = some stream ∧
      st.extra sess.points = .ok extras ∧ decodeProof (shapeOf st sch extras.size) bytes = .ok p ∧
      ∀ ℓ < sch.levels.size, ∀ j < (positions (sch.levels.getD ℓ default)
          (stream.rounds[repRound st sch extras.size ℓ]!).coins).size,
        ∀ u ∈ levelIns st.merkle sch stream.rounds (repRound st sch extras.size) p ℓ j, Claims.toMsg u ∈ L := by
  obtain ⟨stream, extras, p, hstr, hex, hp, hmc⟩ := rep_levels_checkWith (hashesIn L) hsch hf hkl h
  refine ⟨stream, extras, p, hstr, hex, hp, fun ℓ hℓ j hj u hu => ?_⟩
  have hruns := With.merkleCheckWith_sim (back L) (With.opsWith_sim (back L) (back_hashes L) st.merkle)
    st.merkle _ _ _ _ _ _ _ _ () (hmc ℓ hℓ)
  exact (merkleCheck_logsOf hms hruns).2 j hj u hu

/-! ## The verdict's reps -/

theorem verifyWithM_proofs_size (hs : Hashes Flock.V) {spec : SessionSpec} {tabs : Array Setup} {record : String}
    {proofs : Array ByteArray} (h : verifyWithM hs spec tabs record proofs = .ok ()) :
    proofs.size = spec.repStreams.size := by
  unfold verifyWithM at h
  dsimp only at h
  by_cases c0 : (tabs.size != spec.tables.size || tabs.size == 0) = true
  · rw [ite_eq_left c0] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c0] at h
  obtain ⟨sess, hsess, h⟩ := ebind_ok h
  obtain ⟨_, _, h⟩ := ebind_ok h
  obtain ⟨schs, hschs, h⟩ := ebind_ok h
  split at h
  · obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rename_i hne
  simpa [bne_iff_ne] using hne

/-- **Rep `k` of an accepted plain verdict**: table `k / 2`'s rep `k % 2` passes on proof file `k`. -/
theorem verifyWithM_rep (hs : Hashes Flock.V) {spec : SessionSpec} {tabs : Array Setup} {record : String}
    {proofs : Array ByteArray} (h : verifyWithM hs spec tabs record proofs = .ok ()) :
    ∃ (sess : RunRecord) (schs : Array Schedule), Record.decodeWith hs spec record = .ok sess ∧
      tabs.size = spec.tables.size ∧
      ∀ (k : ℕ) (hk : k / 2 < tabs.size), fast100 tabs[k / 2].m = some schs[k / 2]! ∧
        ∃ c, verifyRepWith hs spec tabs[k / 2] schs[k / 2]! sess (k / 2) (k % 2) proofs[k]! = .ok c := by
  obtain ⟨sess, schs, hdec, hsz, hss, hall⟩ := verifyWithM_tables hs h
  refine ⟨sess, schs, hdec, hsz, fun k hk => ?_⟩
  obtain ⟨hk', c₀, c₁, hf, h₀, h₁⟩ := hall (k / 2) hk
  rw [getElem!_pos schs (k / 2) hk']
  refine ⟨hf, ?_⟩
  rcases Nat.mod_two_eq_zero_or_one k with e | e
  · have e' : proofs[k]! = proofs[2 * (k / 2)]! := by rw [show 2 * (k / 2) = k by omega]
    rw [e, e']
    exact ⟨c₀, h₀⟩
  · have e' : proofs[k]! = proofs[2 * (k / 2) + 1]! := by rw [show 2 * (k / 2) + 1 = k by omega]
    rw [e, e']
    exact ⟨c₁, h₁⟩

/-! ## The calls half where a proof file differs -/

/-- **Two last messages the deployed plain verifier accepts after the same rounds and coins** have the same proof
files, or two of their openings collide among the two decisions' calls, or two of their proof files at one position
differ and decode alike. -/
theorem plain_calls_tied {x : Claims.Msg} {ms : List Claims.Msg} {l l' : Claims.Msg} {cs : List Coin}
    (hc : cs.length = ms.length + 1)
    (hacc : ((next false x (ms ++ [l]) cs).eval sha512).1 = .acc)
    (hacc' : ((next false x (ms ++ [l']) cs).eval sha512).1 = .acc) :
    ∃ f f' : Final, finalOf l = some f ∧ finalOf l' = some f' ∧
      (f.proofs = f'.proofs ∨
        Claims.TiedColl sha512 (((next false x (ms ++ [l]) cs).eval sha512).2 ++
          ((next false x (ms ++ [l']) cs).eval sha512).2) ∨
        RecDeploy.ProofSlack l l') := by
  obtain ⟨s, spec, sts, link, c₁, mids, f, hact, hrun⟩ := next_acc_decide hacc
  obtain ⟨s', spec', sts', link', c₁', mids', f', hact', hrun'⟩ := next_acc_decide hacc'
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
  have hfl : finalOf l = some f := by
    have e := act_decide_final hact
    rw [List.getLast?_concat] at e
    exact e
  have hfl' : finalOf l' = some f' := by
    have e := act_decide_final hact'
    rw [List.getLast?_concat] at e
    exact e
  refine ⟨f, f', hfl, hfl', ?_⟩
  by_cases hpf : f.proofs = f'.proofs
  · exact Or.inl hpf
  right
  have hsu := (setupOf_some hs).2
  obtain ⟨hvr, hck⟩ := decision_runs hsu hrun
  obtain ⟨hvr', hck'⟩ := decision_runs hsu hrun'
  generalize hL₁ : ((next false x (b :: ms'' ++ [l]) cs).eval sha512).2 = L₁ at hvr ⊢
  generalize hL₂ : ((next false x (b :: ms'' ++ [l']) cs).eval sha512).2 = L₂ at hvr' ⊢
  have hV : verifyWithM (hashesIn (L₁.map Prod.fst)) spec (tabsOf sts) f.record f.proofs = .ok () :=
    verdict_within hvr
  have hV' : verifyWithM (hashesIn (L₂.map Prod.fst)) spec (tabsOf sts) f'.record f'.proofs = .ok () :=
    verdict_within hvr'
  obtain ⟨-, lv, hlv, hcu⟩ := Claims.Bridge.Custody.checks_ok hck
  obtain ⟨-, lv', hlv', hcu'⟩ := Claims.Bridge.Custody.checks_ok hck'
  have hsz := verifyWithM_proofs_size _ hV
  have hsz' := verifyWithM_proofs_size _ hV'
  obtain ⟨k, hk, hk', hne⟩ : ∃ (k : ℕ) (hk : k < f.proofs.size) (hk' : k < f'.proofs.size),
      f.proofs[k] ≠ f'.proofs[k] := by
    by_contra hcon
    push Not at hcon
    exact hpf (Array.ext (hsz.trans hsz'.symm) hcon)
  obtain ⟨sess, schs, hdec, htsz, hrep⟩ := verifyWithM_rep _ hV
  obtain ⟨sess', schs', hdec', -, hrep'⟩ := verifyWithM_rep _ hV'
  have hq : k < spec.repStreams.size := hsz ▸ hk
  have hi : k / 2 < (tabsOf sts).size := by
    have := repStreams_size spec
    omega
  obtain ⟨hsch, c, hr₁⟩ := hrep k hi
  obtain ⟨hsch', c', hr₂⟩ := hrep' k hi
  rw [hsch] at hsch'
  have hss : schs'[k / 2]! = schs[k / 2]! := (Option.some.inj hsch').symm
  rw [hss] at hr₂
  generalize hst : (tabsOf sts)[k / 2] = st at hsch hr₁ hr₂
  generalize hsc : schs[k / 2]! = sch at hsch hr₁ hr₂
  have hfold := tabsOf_fold sts st (hst ▸ Array.getElem_mem hi)
  have hsts : k / 2 < sts.size := by simpa [tabsOf] using hi
  have hms : st.merkle = MerkleScheme.hm96Sha512 ∨
      st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" := by
    rw [← hst, tabsOf_getElem hi]
    rcases checks_leaf hck hsu (k / 2) hsts with e | ⟨-, e⟩
    · exact Or.inl e
    · exact Or.inr e
  have named := named_of hms
  have hd : Record.decode spec f.record = .ok sess := by
    rw [← With.decodeWith_std]
    exact (With.decodeWith_sim sub (sub_hashes _) spec f.record) sess hdec
  have hd' : Record.decode spec f'.record = .ok sess' := by
    rw [← With.decodeWith_std]
    exact (With.decodeWith_sim sub (sub_hashes _) spec f'.record) sess' hdec'
  obtain ⟨y, y', hy, hy', hyy, hret⟩ := sess_rounds_eq _ _ hV hlv hV' hlv' hcu hcu' hd hd' hq
  simp only [repStreams_getElem spec hq] at hy hy'
  obtain ⟨hrB, hpts, -⟩ := Claims.Bridge.Custody.decode_live hd hlv
  obtain ⟨hrB', hpts', -⟩ := Claims.Bridge.Custody.decode_live hd' hlv'
  obtain ⟨hlB, -, hlP, -⟩ := Claims.Bridge.Custody.custody_ok hcu
  obtain ⟨hlB', -, hlP', -⟩ := Claims.Bridge.Custody.custody_ok hcu'
  have hP : sess.points = sess'.points := by rw [hpts, hpts', hlP, hlP']
  have hB : sess.rootB = sess'.rootB := by
    rw [hrB, hrB']
    exact Array.ext' (hlB.trans hlB'.symm)
  have hpure : ∀ {L : List Claims.Msg} {sess₀ : RunRecord} {bs c₀ : ByteArray},
      verifyRepWith (hashesIn L) spec st sch sess₀ (k / 2) (k % 2) bs = .ok c₀ →
        verifyRep spec st sch sess₀ (k / 2) (k % 2) bs = .ok c₀ := by
    intro L sess₀ bs c₀ h
    rw [← With.verifyRepWith_std _ _ named]
    exact (With.verifyRepWith_sim sub (sub_hashes L) spec st sch sess₀ _ _ bs) c₀ h
  have hv₁ := hpure hr₁
  have hv₂ : verifyRep spec st sch sess (k / 2) (k % 2) f'.proofs[k]! = .ok c' := by
    rw [verifyRep_congr _ hy hy' hyy hP (by rw [hB])]
    exact hpure hr₂
  have hret' : ∀ stream : StreamRec,
      sess.streams.find? (fun x => x.name == toString spec.tables[k / 2]! ++ toString "/rep" ++ toString (k % 2)) =
        some stream → AllRetained stream.rounds := by
    intro stream hs₀
    have e : stream = y := Option.some.inj (hs₀.symm.trans hy)
    rw [e]
    exact hret
  obtain ⟨stream, extras, p, p', hstr, hex, hp, hp', hor⟩ :=
    rep_bind_proof spec st sch sess (k / 2) (k % 2) _ _ c c' hsch hfold.1 hfold.2 hms hv₁ hv₂ hret'
  have hb : f.proofs[k]! = f.proofs[k] := getElem!_pos f.proofs k hk
  have hb' : f'.proofs[k]! = f'.proofs[k] := getElem!_pos f'.proofs k hk'
  rcases hor with rfl | ⟨ℓ, hℓ, j, hj, hcoll⟩
  · right
    refine ⟨f, f', k, f.proofs[k], f'.proofs[k], hfl, hfl', Array.getElem?_eq_getElem hk,
      Array.getElem?_eq_getElem hk', hne, shapeOf st sch extras.size, p, ?_, ?_⟩
    · rw [← hb]; exact hp
    · rw [← hb']; exact hp'
  · left
    have ey : y = stream := Option.some.inj (hy.symm.trans hstr)
    subst y
    obtain ⟨s₁, e₁, q₁, hs₁, he₁, hq₁, hlog₁⟩ := rep_level_logs hsch hfold.1 hfold.2 hms hr₁
    obtain ⟨s₂, e₂, q₂, hs₂, he₂, hq₂, hlog₂⟩ := rep_level_logs hsch hfold.1 hfold.2 hms hr₂
    have hex₂ : st.extra sess'.points = .ok extras := by rw [← hP]; exact hex
    have es₁ : s₁ = stream := Option.some.inj (hs₁.symm.trans hstr)
    have es₂ : s₂ = y' := Option.some.inj (hs₂.symm.trans hy')
    have ee₁ : e₁ = extras := Except.ok.inj (he₁.symm.trans hex)
    have ee₂ : e₂ = extras := Except.ok.inj (he₂.symm.trans hex₂)
    subst s₁ s₂ e₁ e₂
    have eq₁ : q₁ = p := Except.ok.inj (hq₁.symm.trans hp)
    have eq₂ : q₂ = p' := Except.ok.inj (hq₂.symm.trans hp')
    subst q₁ q₂
    rw [← hyy] at hlog₂
    refine CollIn.tied hcoll (fun u hu => ?_) (fun u hu => ?_)
    · rw [List.map_append]
      exact List.mem_append_left _ (hlog₁ ℓ hℓ j hj u hu)
    · rw [List.map_append]
      exact List.mem_append_right _ (hlog₂ ℓ hℓ j hj u hu)

end FlockDeploy.KS

namespace RecDeploy

open Claims (Msg Out OC)
open FlockDeploy (Coin)

/-- **The calls half where the proof files agree**: two last messages the deployed plain verifier accepts after the
same rounds and coins, where it answered each earlier round with that round's coin, with the same proof files, make
the same set of calls. Proved at SHA-512 (`RecDeploy.plainCallsOfProofs_sha512`): every call `Record.decodeWith`
makes hashes a value custody or the statement pins (`root_B`, `root_F = Σ`, the publics, `H(ε)`, the rounds' bytes,
S13's roots, S14's exchange), the S17 digests are of the files, each rep reads its session only through what
`sess_rounds_eq` pins (`verifyRep_congr`), and S11 leaves no stream outside the rep streams, whose names are distinct
(`FlockDeploy.KS.plain_calls_agree`). -/
def PlainCallsOfProofs (H : Msg → Msg) : Prop :=
  ∀ (x : Msg) (ms : List Msg) (l l' : Msg) (cs : List Coin) (f f' : FlockDeploy.Final),
    ms.length + 1 = (FlockDeploy.V false).k x → cs.length = ms.length + 1 →
    (∀ j < ms.length, ((FlockDeploy.next false x (ms.take (j + 1)) (cs.take (j + 1))).eval H).1 =
      .msg (cs.getD j default).bytes) →
    ((FlockDeploy.next false x (ms ++ [l]) cs).eval H).1 = .acc →
    ((FlockDeploy.next false x (ms ++ [l']) cs).eval H).1 = .acc →
    FlockDeploy.finalOf l = some f → FlockDeploy.finalOf l' = some f' → f.proofs = f'.proofs →
    ((FlockDeploy.next false x (ms ++ [l]) cs).eval H).2.toFinset =
      ((FlockDeploy.next false x (ms ++ [l']) cs).eval H).2.toFinset

/-- **The calls half from the case where the proof files agree**, at SHA-512: where a proof file differs, two
openings collide among the calls (`FlockDeploy.KS.plain_calls_tied`), or the files decode alike. -/
theorem plainCallsBindOrSlack_of_proofs (hR : PlainCallsOfProofs FlockDeploy.sha512) :
    PlainCallsBindOrSlack FlockDeploy.sha512 := by
  intro x ms l l' cs hk hc hearly hacc hacc' hne
  obtain ⟨f, f', hfl, hfl', hpf | ⟨u, v, hcol, hu, hv⟩ | hsl⟩ := FlockDeploy.KS.plain_calls_tied hc hacc hacc'
  · exact absurd (hR x ms l l' cs f f' hk hc hearly hacc hacc' hfl hfl' hpf) hne
  · exact Or.inl ⟨u, hu, v, hv, hcol⟩
  · exact Or.inr hsl

end RecDeploy
