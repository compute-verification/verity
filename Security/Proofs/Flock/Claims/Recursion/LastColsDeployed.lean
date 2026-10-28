import Proofs.Flock.Claims.Recursion.LastBindsDeployed
import Proofs.Flock.Claims.KS.ConflictTied
import Proofs.Flock.Claims.KS.RoundsN

/-!
# The deployed plain verifier binds its last message's columns

`RecDeploy.PlainLastBindsOrSlack` splits at its last hypothesis: two last messages that differ in table 0's opened
columns (`RecDeploy.PlainColsBind`), or in the decision's calls (`RecDeploy.PlainCallsBindOrSlack`)
(`RecDeploy.plainLastBindsOrSlack_of`). The columns half holds at SHA-512 (`RecDeploy.plainColsBind_sha512`):

* both decisions read one link message, its coin and the same round messages with their coins (`act_decide_parts`),
  so custody pins both records' rounds to those coins, at the counts the statement fixes (`rounds_ns_plain`): the two
  records' rep streams carry the same coins round by round (`streams_coins_eq`);
* so table 0's reps query level 0 at the same positions in both, and each passes its Merkle check there with one row
  per position (`merkleCheckWith_rows`): their columns sit at the same positions (`rawCols_keys`);
* columns that differ then differ at one position (`exists_mem_snd_ne`), where both are openings against the link's
  `root_B[0]`, every SHA-512 input of each in its decision's calls (`plain_opening_at`), which collide
  (`tiedColl_of_openings`).

The calls half compares the two decisions' calls as sets: `Record.decodeWith` hashes the rounds in the
record's own stream order, which only S11's count and names constrain, so two records that list the streams in two
orders are accepted with permuted calls and no collision (`note:proofs/20261020T0132Z-finding-rec2-calls-as-lists`).
It holds at SHA-512 (`RecDeploy.plainCallsBindOrSlack_sha512`, in `CallsOfProofs`).
`RecDeploy.plainLastBinds_sha512_of_calls` reduces `PlainLastBinds` to it and `CanonicalProofFiles`, and
`RecDeploy.lastBindsEC_deployed_sha512` gives (P4) at the deployed bound record from the same two.
-/

namespace FlockDeploy.KS

open Flock
open FlockSoundness.Refine (ebind_ok entries_eq_of_flat flat_data merkleCheck_cap)

local notation "𝔽" => FlockLevel3.GF128

/-! ## A step that accepts decides -/

/-- **A step that accepts decides**, within its own calls. -/
theorem next_acc_decide {zk : Bool} {H : Claims.Msg → Claims.Msg} {x : Claims.Msg} {as : List Claims.Msg}
    {cs : List Coin} (h : ((next zk x as cs).eval H).1 = .acc) :
    ∃ s spec sts link c₁ mids f, act zk x as cs = .decide s spec sts link c₁ mids f ∧
      Runs H (decision zk s x link c₁ mids f) () (((next zk x as cs).eval H).2.map Prod.fst) := by
  cases hact : act zk x as cs with
  | rej => rw [next_rej H hact] at h; cases h
  | coin c => rw [next_coin H hact] at h; cases h
  | decide s spec sts link c₁ mids f =>
    refine ⟨s, spec, sts, link, c₁, mids, f, rfl, ?_⟩
    have ha : ((outOf (decision zk s x link c₁ mids f)).eval H).1 = .acc := by
      simp only [next, hact] at h
      exact h
    have hl : ((next zk x as cs).eval H).2 = ((decision zk s x link c₁ mids f).run.eval H).2 := by
      simp only [next, hact, outOf_log]
    rw [hl]
    exact Runs.self H (outOf_acc H _ ha)

/-- **A decision reads the link from the first message, its coin from the first coin, and pairs the messages between
the first and the last with their coins.** -/
theorem act_decide_parts {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (h : act zk x as cs = .decide s spec sts link c₁ mids f) :
    ∃ a₁ rest crest, as = a₁ :: rest ∧ cs = c₁ :: crest ∧ linkOf a₁ = some link ∧
      mids = (rest.dropLast.map List.tail).zip crest.dropLast := by
  unfold act at h
  split at h
  · cases h
  · repeat' split at h
    all_goals first
      | (cases h; done)
      | (cases h; exact ⟨_, _, _, rfl, rfl, by assumption, rfl⟩)

/-! ## Lists -/

theorem map_fst_zip_congr {α β γ : Type} : ∀ (l₁ : List α) (l₂ : List β) (l₃ : List γ), l₂.length = l₃.length →
    (l₁.zip l₂).map Prod.fst = (l₁.zip l₃).map Prod.fst
  | [], _, _, _ => by simp
  | _ :: _, [], [], _ => rfl
  | _ :: _, [], _ :: _, h => by simp at h
  | _ :: _, _ :: _, [], h => by simp at h
  | _ :: l₁, _ :: l₂, _ :: l₃, h => by
    simp only [List.zip_cons_cons, List.map_cons]
    rw [map_fst_zip_congr l₁ l₂ l₃ (by simpa using h)]

theorem map_fst_map_pair {α β γ : Type} (g : α × β → γ) : ∀ L : List (α × β),
    (L.map fun p => (p.1, g p)).map Prod.fst = L.map Prod.fst
  | [] => rfl
  | p :: L => by simp only [List.map_cons, map_fst_map_pair g L]

/-- Two lists of pairs with the same keys that differ differ at one key. -/
theorem exists_mem_snd_ne {α β : Type} : ∀ {L₁ L₂ : List (α × β)}, L₁.map Prod.fst = L₂.map Prod.fst → L₁ ≠ L₂ →
    ∃ a b b', (a, b) ∈ L₁ ∧ (a, b') ∈ L₂ ∧ b ≠ b'
  | [], [], _, hne => absurd rfl hne
  | [], _ :: _, h, _ => by simp at h
  | _ :: _, [], h, _ => by simp at h
  | (a, b) :: L₁, (a', b') :: L₂, h, hne => by
    simp only [List.map_cons, List.cons.injEq] at h
    obtain ⟨rfl, h⟩ := h
    by_cases hb : b = b'
    · subst hb
      obtain ⟨c, d, d', h₁, h₂, hd⟩ := exists_mem_snd_ne h (fun e => hne (by rw [e]))
      exact ⟨c, d, d', List.mem_cons_of_mem _ h₁, List.mem_cons_of_mem _ h₂, hd⟩
    · exact ⟨a, b, b', List.mem_cons_self .., List.mem_cons_self .., hb⟩

/-! ## The coins of two records under one custody -/

theorem coins_of_forall₂ {R : Round → Claims.Msg × Coin → Prop} :
    ∀ {rs : List Round} {ms : List (Claims.Msg × Coin)}, List.Forall₂ R rs ms →
      (∀ rd am, R rd am → rd.coins = am.2.vals rd.n) →
      rs.map Round.coins = List.zipWith (fun n (am : Claims.Msg × Coin) => am.2.vals n) (rs.map Round.n) ms
  | [], [], .nil, _ => rfl
  | rd :: _, am :: _, .cons hab h, hR => by
    simp only [List.map_cons, List.zipWith_cons_cons, hR rd am hab, coins_of_forall₂ h hR]

/-- Streams of the same sizes whose rounds, read in order, carry the same coins carry them stream by stream. -/
theorem coins_streams_eq : ∀ {S S' : List (Array Round)}, S.map (fun s => s.size) = S'.map (fun s => s.size) →
    (S.flatMap (·.toList)).map Round.coins = (S'.flatMap (·.toList)).map Round.coins →
    S.map (fun s => s.toList.map Round.coins) = S'.map (fun s => s.toList.map Round.coins)
  | [], [], _, _ => rfl
  | [], _ :: _, hl, _ => by simp at hl
  | _ :: _, [], hl, _ => by simp at hl
  | a :: S, a' :: S', hl, hf => by
    simp only [List.map_cons, List.cons.injEq] at hl
    simp only [List.flatMap_cons, List.map_append] at hf
    have hlen : (a.toList.map Round.coins).length = (a'.toList.map Round.coins).length := by simp [hl.1]
    obtain ⟨h₁, h₂⟩ := List.append_inj hf hlen
    simp only [List.map_cons, h₁, coins_streams_eq hl.2 h₂]

/-- **An accepted plain record's rep streams have the sizes its statement fixes.** -/
theorem streams_sizes_plain (hs : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt} {record : String}
    {proofs : Array ByteArray} {lv : Live} (hv : verifyWithM hs spec (tabsOf sts) record proofs = .ok ())
    (hl : liveOf spec record = .ok lv) :
    lv.streams.toList.map (fun s => s.size) =
      (List.range (2 * sts.size)).map fun k => (plainRepNs spec sts[k / 2]!).length := by
  obtain ⟨hsz, hall⟩ := streams_ns_plain hs hv hl
  refine List.ext_getElem (by simp [hsz]) fun i h₁ h₂ => ?_
  have hi : i < lv.streams.size := by simpa using h₁
  simp only [List.getElem_map, List.getElem_range, Array.getElem_toList]
  rw [← hall i hi]
  simp only [List.length_map, Array.length_toList]

/-- **Two accepted plain records of one session under custody of the same round messages carry the same coins, stream
by stream**, whatever hashes each verdict ran with. -/
theorem streams_coins_eq (hs hs' : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt} {record record' : String}
    {proofs proofs' : Array ByteArray} {lv lv' : Live} {link link' : List ByteArray × List ByteArray} {c₁ c₁' : Coin}
    {mids : List (Claims.Msg × Coin)}
    (hv : verifyWithM hs spec (tabsOf sts) record proofs = .ok ()) (hl : liveOf spec record = .ok lv)
    (hv' : verifyWithM hs' spec (tabsOf sts) record' proofs' = .ok ()) (hl' : liveOf spec record' = .ok lv')
    (hc : custody spec lv link c₁ mids = .ok ()) (hc' : custody spec lv' link' c₁' mids = .ok ()) :
    lv.streams.toList.map (fun s => s.toList.map Round.coins) =
      lv'.streams.toList.map (fun s => s.toList.map Round.coins) := by
  refine coins_streams_eq ((streams_sizes_plain hs hv hl).trans (streams_sizes_plain hs' hv' hl').symm) ?_
  rw [coins_of_forall₂ (Claims.Bridge.Custody.custody_ok hc).2.2.2 (fun _ _ h => h.2),
    coins_of_forall₂ (Claims.Bridge.Custody.custody_ok hc').2.2.2 (fun _ _ h => h.2),
    rounds_ns_plain hs hv hl, rounds_ns_plain hs' hv' hl']

theorem coins_at {S S' : Array (Array Round)}
    (hS : S.toList.map (fun s => s.toList.map Round.coins) = S'.toList.map (fun s => s.toList.map Round.coins))
    {j i : ℕ} {rd rd' : Round} (h : (S.getD j #[])[i]? = some rd) (h' : (S'.getD j #[])[i]? = some rd') :
    rd.coins = rd'.coins := by
  have hsz : S.size = S'.size := by simpa using congrArg List.length hS
  by_cases hj : j < S.size
  · have hj' : j < S'.size := by omega
    have e₁ : S.getD j #[] = S[j] := by simp [Array.getD, hj]
    have e₂ : S'.getD j #[] = S'[j] := by simp [Array.getD, hj']
    rw [e₁] at h
    rw [e₂] at h'
    have hrow : S[j].toList.map Round.coins = S'[j].toList.map Round.coins := by
      have e := congrArg (fun L => L[j]?) hS
      simp only [List.getElem?_map, Array.getElem?_toList, Array.getElem?_eq_getElem hj,
        Array.getElem?_eq_getElem hj'] at e
      exact Option.some.inj e
    have hi : (S[j].toList.map Round.coins)[i]? = some rd.coins := by
      rw [List.getElem?_map, Array.getElem?_toList, h]; rfl
    have hi' : (S'[j].toList.map Round.coins)[i]? = some rd'.coins := by
      rw [List.getElem?_map, Array.getElem?_toList, h']; rfl
    rw [hrow, hi'] at hi
    exact (Option.some.inj hi).symm
  · have e₁ : S.getD j #[] = #[] := by simp [Array.getD, hj]
    rw [e₁] at h
    simp at h

/-! ## Table 0's columns at two records -/

theorem merkleCheckWith_rows {ops : MerkleOps Flock.V} {ms : MerkleScheme} {cap : Array ByteArray} {c d : ℕ}
    {pos : Array ℕ} {rows : Array (Array F128)} {sibs salts : Array ByteArray} {lanes : ℕ}
    (h : merkleCheckWith ops ms cap c d pos rows sibs salts lanes = .ok ()) : rows.size = pos.size := by
  unfold merkleCheckWith at h
  split at h
  · cases h
  rename_i h₁
  simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, not_not] at h₁
  exact h₁.1.1.2

theorem zkRepCols_keys (lv₀ : Flock.Level) (c : Array F128) {rows rows' : Array (Array F128)}
    (h : rows.size = rows'.size) :
    (Claims.Flock.zkRepCols lv₀ c rows).map Prod.fst = (Claims.Flock.zkRepCols lv₀ c rows').map Prod.fst := by
  have e : ∀ L : List (ℕ × Array F128),
      (L.map fun p => (p.1, p.2.toList.map FlockLevel3.GF128.of)).map Prod.fst = L.map Prod.fst :=
    fun L => map_fst_map_pair (fun p : ℕ × Array F128 => p.2.toList.map FlockLevel3.GF128.of) L
  unfold Claims.Flock.zkRepCols
  rw [e, e]
  exact map_fst_zip_congr _ _ _ (by simp [h])

/-- What `runOf` takes of table 0 at a plain record whose statement's first table is `st`. -/
def plainRun0 (st : Stmt) (extras : Array Flock.Claim) (lv : Live) (f : Final) : Claims.Flock.ZkRun :=
  { m := (Setup.ofCircuit st).m, kLog := (Setup.ofCircuit st).kLog, claims := 2 + extras.size,
    hashLen := (Setup.ofCircuit st).merkle.digestLen, saltLen := saltLenOf false (Setup.ofCircuit st),
    reps := [(lv.streams.getD 0 #[], f.proofs.getD 0 ByteArray.empty),
             (lv.streams.getD 1 #[], f.proofs.getD 1 ByteArray.empty)] }

theorem runOf_plain_of {seen : Claims.Seen} {f : Final} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {lv : Live} {st : Stmt} {extras : Array Flock.Claim} (hf : lastFinal seen = some f)
    (hset : setupOf false f.stmt = some (s, spec, sts)) (hlv : liveOf spec f.record = .ok lv)
    (hst : sts[0]? = some st) (hex : (Setup.ofCircuit st).extra lv.points = .ok extras) :
    runOf false seen 0 = some (plainRun0 st extras lv f) := by
  unfold runOf
  simp [hf, hset, hst, hlv, hex, Except.toOption, plainRun0]

theorem runOf_none_of_tab {seen : Claims.Seen} {f : Final} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hf : lastFinal seen = some f) (hset : setupOf false f.stmt = some (s, spec, sts)) (hst : sts[0]? = none) :
    runOf false seen 0 = none := by
  unfold runOf
  simp [hf, hset, hst]

theorem runOf_none_of_extra {seen : Claims.Seen} {f : Final} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {lv : Live} {st : Stmt} {e : String} (hf : lastFinal seen = some f)
    (hset : setupOf false f.stmt = some (s, spec, sts)) (hlv : liveOf spec f.record = .ok lv)
    (hst : sts[0]? = some st) (hex : (Setup.ofCircuit st).extra lv.points = .error e) :
    runOf false seen 0 = none := by
  unfold runOf
  simp [hf, hset, hst, hlv, hex, Except.toOption]

theorem cols_eq_raw {seen : Claims.Seen} {st : Stmt} {extras : Array Flock.Claim} {lv : Live} {f : Final}
    (h : runOf false seen 0 = some (plainRun0 st extras lv f)) :
    cols false seen 0 =
      Claims.Flock.distinctBelow (Claims.Flock.plainLevel0 (Setup.ofCircuit st).m).d (plainRawCols seen) := by
  unfold cols plainRawCols
  rw [h]
  simp only [Bool.false_eq_true, ↓reduceIte, Claims.Flock.plainRunCols, plainRun0]

theorem rawCols_eq {seen : Claims.Seen} {st : Stmt} {extras : Array Flock.Claim} {lv : Live} {f : Final}
    (h : runOf false seen 0 = some (plainRun0 st extras lv f)) :
    plainRawCols seen =
      Claims.Flock.plainRepColsOf (plainRun0 st extras lv f).plainShape (lv.streams.getD 0 #[])
          (f.proofs.getD 0 ByteArray.empty) ++
        Claims.Flock.plainRepColsOf (plainRun0 st extras lv f).plainShape (lv.streams.getD 1 #[])
          (f.proofs.getD 1 ByteArray.empty) := by
  unfold plainRawCols
  rw [h]
  simp only [plainRun0, List.flatMap_cons, List.flatMap_nil, List.append_nil]

/-- **Two accepted plain records whose rep streams carry the same coins open table 0 at the same positions**: each
rep queries level 0 at its query round's coins, and its Merkle check passes there with one row per position. -/
theorem rawCols_keys {hs hs' : Hashes Flock.V} {seen seen' : Claims.Seen} {f f' : Final} {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {st : Stmt} {extras : Array Flock.Claim} {lv lv' : Live}
    (hf : lastFinal seen = some f) (hf' : lastFinal seen' = some f')
    (hset : setupOf false f.stmt = some (ins, spec, sts)) (hset' : setupOf false f'.stmt = some (ins, spec, sts))
    (hv : verifyWithM hs spec (tabsOf sts) f.record f.proofs = .ok ())
    (hv' : verifyWithM hs' spec (tabsOf sts) f'.record f'.proofs = .ok ())
    (hr : runOf false seen 0 = some (plainRun0 st extras lv f))
    (hr' : runOf false seen' 0 = some (plainRun0 st extras lv' f'))
    (hS : lv.streams.toList.map (fun s => s.toList.map Round.coins) =
      lv'.streams.toList.map (fun s => s.toList.map Round.coins)) :
    (plainRawCols seen).map Prod.fst = (plainRawCols seen').map Prod.fst := by
  have hmem : ∀ (lv₀ : Live) (f₀ : Final),
      (lv₀.streams.getD 0 #[], f₀.proofs.getD 0 ByteArray.empty) ∈ (plainRun0 st extras lv₀ f₀).reps ∧
        (lv₀.streams.getD 1 #[], f₀.proofs.getD 1 ByteArray.empty) ∈ (plainRun0 st extras lv₀ f₀).reps := by
    intro lv₀ f₀
    simp [plainRun0]
  have key : ∀ j : ℕ,
      (lv.streams.getD j #[], f.proofs.getD j ByteArray.empty) ∈ (plainRun0 st extras lv f).reps →
      (lv'.streams.getD j #[], f'.proofs.getD j ByteArray.empty) ∈ (plainRun0 st extras lv' f').reps →
      (Claims.Flock.plainRepColsOf (plainRun0 st extras lv f).plainShape (lv.streams.getD j #[])
          (f.proofs.getD j ByteArray.empty)).map Prod.fst =
        (Claims.Flock.plainRepColsOf (plainRun0 st extras lv' f').plainShape (lv'.streams.getD j #[])
          (f'.proofs.getD j ByteArray.empty)).map Prod.fst := by
    intro j h₁ h₂
    obtain ⟨p, rd, -, hrd, -, hmc, hc⟩ := runOf_queryRound_plain hs hf hset hv hr _ h₁
    obtain ⟨p', rd', -, hrd', -, hmc', hc'⟩ := runOf_queryRound_plain hs' hf' hset' hv' hr' _ h₂
    have hrd₁ : (lv.streams.getD j #[])[Claims.Flock.plainQueryRound (Setup.ofCircuit st).m
        (Setup.ofCircuit st).kLog (Claims.Flock.plainSched (Setup.ofCircuit st).m).k0 (2 + extras.size)]? =
        some rd := hrd
    have hrd₂ : (lv'.streams.getD j #[])[Claims.Flock.plainQueryRound (Setup.ofCircuit st).m
        (Setup.ofCircuit st).kLog (Claims.Flock.plainSched (Setup.ofCircuit st).m).k0 (2 + extras.size)]? =
        some rd' := hrd'
    have hco : rd.coins = rd'.coins := coins_at hS hrd₁ hrd₂
    have hm₁ : p.ligerito.initialProof.rows.size =
        (positions (Claims.Flock.plainLevel0 (Setup.ofCircuit st).m) rd.coins).size :=
      merkleCheckWith_rows hmc
    have hm₂ : p'.ligerito.initialProof.rows.size =
        (positions (Claims.Flock.plainLevel0 (Setup.ofCircuit st).m) rd'.coins).size :=
      merkleCheckWith_rows hmc'
    have hc₁ : Claims.Flock.plainRepColsOf (plainRun0 st extras lv f).plainShape (lv.streams.getD j #[])
        (f.proofs.getD j ByteArray.empty) =
        Claims.Flock.zkRepCols (Claims.Flock.plainLevel0 (Setup.ofCircuit st).m) rd.coins
          p.ligerito.initialProof.rows := hc
    have hc₂ : Claims.Flock.plainRepColsOf (plainRun0 st extras lv' f').plainShape (lv'.streams.getD j #[])
        (f'.proofs.getD j ByteArray.empty) =
        Claims.Flock.zkRepCols (Claims.Flock.plainLevel0 (Setup.ofCircuit st).m) rd'.coins
          p'.ligerito.initialProof.rows := hc'
    rw [hc₁, hc₂, ← hco]
    exact zkRepCols_keys _ _ (by rw [hm₁, hm₂, hco])
  rw [rawCols_eq hr, rawCols_eq hr', List.map_append, List.map_append, key 0 (hmem lv f).1 (hmem lv' f').1,
    key 1 (hmem lv f).2 (hmem lv' f').2]

/-- **A plain decision's reps open a column only as a Merkle opening** of table 0's level 0 within the calls `L` its
verdict ran in: `plain_opening_raw` at a decision, before the run that made it. -/
theorem plain_opening_at {x : Claims.Msg} {seen : Claims.Seen} {L : List (Claims.Msg × Claims.Msg)} {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (hs : setupOf false x = some (s, spec, sts))
    (hf : lastFinal seen = some f) (hc : checks false s spec x link c₁ mids f = .ok ())
    (hv : verdictM false (hashesIn (L.map Prod.fst)) spec (tabsOf sts) f.record f.proofs = .ok ())
    {q : ℕ} {u : List 𝔽} (hu : (q, u) ∈ plainRawCols seen) :
    ∃ (st : Stmt) (sch : Flock.Schedule) (row : Array F128), sts[0]? = some st ∧ fast100 st.m = some sch ∧
      Opening (Setup.ofCircuit st).merkle (sch.levels.getD 0 default) (2 ^ (sch.levels.getD 0 default).lanes)
        link.1.toArray[0]! L q row ∧ u = row.toList.map FlockLevel3.GF128.of := by
  obtain ⟨hx, lv, hlv, hcu⟩ := Claims.Bridge.Custody.checks_ok hc
  obtain ⟨hroot, -⟩ := Claims.Bridge.Custody.custody_ok hcu
  unfold plainRawCols at hu
  split at hu
  swap
  · simp at hu
  rename_i rr hrr
  obtain ⟨st, extras, hst, hext, rfl⟩ := runOf_eq hf hx hs hlv hrr
  obtain ⟨ρ, hρ, hp⟩ := List.mem_flatMap.1 hu
  obtain ⟨k, hk, rfl⟩ : ∃ k < 2, ρ = (lv.streams.getD k #[], f.proofs.getD k ByteArray.empty) := by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hρ
    rcases hρ with rfl | rfl
    · exact ⟨0, by omega, rfl⟩
    · exact ⟨1, by omega, rfl⟩
  have hsts0 : 0 < sts.size := by
    rcases Array.getElem?_eq_some_iff.1 hst with ⟨h0, -⟩
    exact h0
  have hst0 : sts[0] = st := by
    rcases Array.getElem?_eq_some_iff.1 hst with ⟨h0, e⟩
    exact e
  have h0 : 0 < (tabsOf sts).size := by simpa [tabsOf] using hsts0
  have ht0 : (tabsOf sts)[0] = Setup.ofCircuit st := by rw [tabsOf_getElem h0]; simp only [hst0]
  replace hv : verifyWithM (hashesIn (L.map Prod.fst)) spec (tabsOf sts) f.record f.proofs = .ok () := hv
  obtain ⟨sess, hdec, hsz, hreps⟩ :=
    plainQueryRound_verifyWithM (hashesIn (L.map Prod.fst)) spec (tabsOf sts) f.record f.proofs (tabsOf_fold sts) hv
  obtain ⟨sch, hsch, hrep⟩ := hreps 0 h0
  obtain ⟨stream, extras', p, hstream, hext', hp', hlt, hcols, hmc, hcap⟩ := hrep k hk
  simp only [ht0] at hsch hext' hp' hlt hcols hmc
  have hdec' : Record.decode spec f.record = .ok sess := by
    rw [← With.decodeWith_std]
    exact (With.decodeWith_sim sub (sub_hashes (L.map Prod.fst)) spec f.record) sess hdec
  obtain ⟨hrB, hpts, hlsz, hstreams⟩ := Claims.Bridge.Custody.decode_live hdec' hlv
  rw [hpts, hext] at hext'
  cases hext'
  have htab : 0 < spec.tables.size := by rw [← hsz]; exact h0
  obtain ⟨hk', hrs⟩ := Claims.Bridge.Custody.repStreams_getElem spec htab hk
  obtain ⟨y, hy, hfa⟩ := hstreams (2 * 0 + k) hk'
  rw [hrs] at hy
  have hys : y = stream := Option.some.inj (hy.symm.trans hstream)
  subst hys
  have hcoins := forall₂_coins (fun _ _ hab => hab.2.1) hfa
  have hcolsEq : Claims.Flock.plainRepColsOf
      (Claims.Flock.plainProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog (2 + extras.size)
        (Setup.ofCircuit st).merkle.digestLen (saltLenOf false (Setup.ofCircuit st)))
      (lv.streams.getD k #[]) (f.proofs.getD k ByteArray.empty) =
      Claims.Flock.zkRepCols (sch.levels.getD 0 default)
        (y.rounds[Claims.Flock.plainQueryRound (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog sch.k0
          (2 + extras.size)]!).coins p.ligerito.initialProof.rows := by
    rw [← hcols]
    have e1 : lv.streams.getD k #[] = lv.streams[2 * 0 + k]! := by
      rw [show (#[] : Array Flock.Round) = default from rfl, getD_eq_getElem!]; simp
    have e2 : f.proofs.getD k ByteArray.empty = f.proofs[2 * 0 + k]! := by
      rw [show ByteArray.empty = default from rfl, getD_eq_getElem!]; simp
    rw [e1, e2]
    refine plainRepColsOf_congr _ _ fun j => ?_
    rw [← Array.getElem?_toList, ← Array.getElem?_toList]
    exact (hcoins j).symm
  have hp2 : (q, u) ∈ Claims.Flock.zkRepCols (sch.levels.getD 0 default)
      (y.rounds[Claims.Flock.plainQueryRound (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog sch.k0
        (2 + extras.size)]!).coins p.ligerito.initialProof.rows := by
    rw [← hcolsEq]
    exact hp
  obtain ⟨j, hj, -, hjq, hju⟩ := mem_zkRepCols hp2
  have hms : (Setup.ofCircuit st).merkle = MerkleScheme.hm96Sha512 ∨
      (Setup.ofCircuit st).merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" := by
    rcases checks_leaf hc (setupOf_some hs).2 0 hsts0 with h | ⟨-, h⟩
    · rw [hst0] at h
      exact Or.inl h
    · rw [hst0] at h
      exact Or.inr h
  have hdl : (Setup.ofCircuit st).merkle.digestLen = 64 := by
    rcases hms with e | e <;> rw [e] <;> rfl
  have hruns := With.merkleCheckWith_sim (back (L.map Prod.fst))
    (With.opsWith_sim (back (L.map Prod.fst)) (back_hashes (L.map Prod.fst)) (Setup.ofCircuit st).merkle)
    (Setup.ofCircuit st).merkle _ _ _ _ _ _ _ _ () hmc
  obtain ⟨hpure, hlog⟩ := merkleCheck_logsOf hms hruns
  obtain ⟨hcs, hcw⟩ := merkleCheck_cap hpure
  refine ⟨st, sch, p.ligerito.initialProof.rows[j]!, hst, hsch,
    ⟨hms, p.cap, p.ligerito.initialProof.salts.getD j ByteArray.empty,
      sibsOf p.ligerito.initialProof.siblings ((sch.levels.getD 0 default).d - (sch.levels.getD 0 default).cap) j,
      hcs, fun b hb => (hcw b hb).trans hdl, ?_, ?_,
      fun b hb => (merkleCheck_sibs _ _ _ _ _ _ _ _ _ hpure j b hb).trans hdl, ?_⟩, hju⟩
  · rw [hcap, hrB, ← hroot, Array.toArray_toList]
  · rw [← hjq]
    exact merkleCheck_opens _ _ _ _ _ _ _ _ _ hpure j hj
  · rw [← hjq]
    exact hlog j hj

/-- **Two last messages the deployed plain verifier accepts after the same rounds and coins, that differ in table 0's
opened columns, collide among the two decisions' calls**, at SHA-512. -/
theorem plain_cols_tied {x : Claims.Msg} {ms : List Claims.Msg} {l l' : Claims.Msg} {cs : List Coin}
    (hc : cs.length = ms.length + 1)
    (hacc : ((next false x (ms ++ [l]) cs).eval sha512).1 = .acc)
    (hacc' : ((next false x (ms ++ [l']) cs).eval sha512).1 = .acc)
    (hne : cols false (RecDeploy.plainView (ms ++ [l]) cs) 0 ≠ cols false (RecDeploy.plainView (ms ++ [l']) cs) 0) :
    Claims.TiedColl sha512 (((next false x (ms ++ [l]) cs).eval sha512).2 ++
      ((next false x (ms ++ [l']) cs).eval sha512).2) := by
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
  obtain ⟨hvr, hck⟩ := decision_runs (setupOf_some hs).2 hrun
  obtain ⟨hvr', hck'⟩ := decision_runs (setupOf_some hs).2 hrun'
  have hw := verdict_within hvr
  have hw' := verdict_within hvr'
  generalize hL₁ : ((next false x (b :: ms'' ++ [l]) cs).eval sha512).2 = L₁ at hw ⊢
  generalize hL₂ : ((next false x (b :: ms'' ++ [l']) cs).eval sha512).2 = L₂ at hw' ⊢
  have hmsg : ∀ l₀ : Claims.Msg,
      (RecDeploy.plainView (b :: ms'' ++ [l₀]) cs).msgs.map Prod.fst = b :: ms'' ++ [l₀] := by
    intro l₀
    unfold RecDeploy.plainView
    exact List.map_fst_zip (by
      simp only [List.length_append, List.length_cons, List.length_map, List.length_dropLast, hc]
      omega)
  have hf₁ : lastFinal (RecDeploy.plainView (b :: ms'' ++ [l]) cs) = some f := by
    have h := act_decide_final hact
    rw [← hmsg l, List.getLast?_map, Option.bind_map] at h
    exact h
  have hf₂ : lastFinal (RecDeploy.plainView (b :: ms'' ++ [l']) cs) = some f' := by
    have h := act_decide_final hact'
    rw [← hmsg l', List.getLast?_map, Option.bind_map] at h
    exact h
  obtain ⟨hx, lv, hlv, hcu⟩ := Claims.Bridge.Custody.checks_ok hck
  obtain ⟨hx', lv', hlv', hcu'⟩ := Claims.Bridge.Custody.checks_ok hck'
  obtain ⟨-, -, hpts, -⟩ := Claims.Bridge.Custody.custody_ok hcu
  obtain ⟨-, -, hpts', -⟩ := Claims.Bridge.Custody.custody_ok hcu'
  have hpp : lv'.points = lv.points := hpts'.trans hpts.symm
  have hV : verifyWithM (hashesIn (L₁.map Prod.fst)) spec (tabsOf sts) f.record f.proofs = .ok () := hw
  have hV' : verifyWithM (hashesIn (L₂.map Prod.fst)) spec (tabsOf sts) f'.record f'.proofs = .ok () := hw'
  have hset₁ : setupOf false f.stmt = some (s, spec, sts) := by rw [hx]; exact hs
  have hset₂ : setupOf false f'.stmt = some (s, spec, sts) := by rw [hx']; exact hs
  cases hst : sts[0]? with
  | none =>
    exfalso
    apply hne
    unfold cols
    rw [runOf_none_of_tab hf₁ hset₁ hst, runOf_none_of_tab hf₂ hset₂ hst]
  | some st =>
    cases hex : (Setup.ofCircuit st).extra lv.points with
    | error e =>
      exfalso
      apply hne
      unfold cols
      rw [runOf_none_of_extra hf₁ hset₁ hlv hst hex,
        runOf_none_of_extra hf₂ hset₂ hlv' hst (by rw [hpp]; exact hex)]
    | ok extras =>
      have hr₁ := runOf_plain_of hf₁ hset₁ hlv hst hex
      have hr₂ := runOf_plain_of hf₂ hset₂ hlv' hst (by rw [hpp]; exact hex)
      have hraw : plainRawCols (RecDeploy.plainView (b :: ms'' ++ [l]) cs) ≠
          plainRawCols (RecDeploy.plainView (b :: ms'' ++ [l']) cs) :=
        fun e => hne (by rw [cols_eq_raw hr₁, cols_eq_raw hr₂, e])
      have hS := streams_coins_eq _ _ hV hlv hV' hlv' hcu hcu'
      obtain ⟨q, u, v, hu, hv, huv⟩ :=
        exists_mem_snd_ne (rawCols_keys hf₁ hf₂ hset₁ hset₂ hV hV' hr₁ hr₂ hS) hraw
      obtain ⟨st₁, sch, row, hst₁, hsch, ho, hu'⟩ := plain_opening_at hs hf₁ hck hw hu
      obtain ⟨st₂, sch', row', hst₂, hsch', ho', hv'⟩ := plain_opening_at hs hf₂ hck' hw' hv
      rw [hst] at hst₁ hst₂
      cases hst₁
      cases hst₂
      rw [hsch] at hsch'
      cases hsch'
      exact tiedColl_of_openings ho ho' fun e => huv (by rw [hu', hv', e])

end FlockDeploy.KS

namespace RecDeploy

open Claims (Msg Out OC)
open FlockDeploy (Coin)

/-- **The deployed plain verifier binds its last message's columns**: two last messages it accepts after the same
rounds and coins, that differ in table 0's opened columns, show a SHA-512 collision among the two decisions' calls. -/
def PlainColsBind (H : Msg → Msg) : Prop :=
  ∀ (x : Msg) (ms : List Msg) (l l' : Msg) (cs : List Coin), ms.length + 1 = (FlockDeploy.V false).k x →
    cs.length = ms.length + 1 →
    ((FlockDeploy.next false x (ms ++ [l]) cs).eval H).1 = .acc →
    ((FlockDeploy.next false x (ms ++ [l']) cs).eval H).1 = .acc →
    FlockDeploy.cols false (plainView (ms ++ [l]) cs) 0 ≠ FlockDeploy.cols false (plainView (ms ++ [l']) cs) 0 →
    ∃ q ∈ (((FlockDeploy.next false x (ms ++ [l]) cs).eval H).2 ++
        ((FlockDeploy.next false x (ms ++ [l']) cs).eval H).2).map Prod.fst,
      ∃ q' ∈ (((FlockDeploy.next false x (ms ++ [l]) cs).eval H).2 ++
        ((FlockDeploy.next false x (ms ++ [l']) cs).eval H).2).map Prod.fst, Claims.Collision H q q'

/-- **The deployed plain verifier binds its last message's calls, up to the decoder's slack**: two last messages it
accepts after the same rounds and coins, where it answered each earlier round with that round's coin, that differ in
the set of the decision's calls, show a collision among those calls, or two proof files at one position that differ in
their bytes and decode alike (`ProofSlack`). The calls are compared as sets: two records that list the same
streams in two orders give permuted calls, and a table's empty publics may come with their digest `H(ε)`, which S12
already calls, or without. Proved at SHA-512 (`plainCallsBindOrSlack_sha512`): where a proof file differs, two reps
accepted on one stream bind (`FlockDeploy.KS.plain_calls_tied`); where the files agree, the calls are one set
(`plainCallsOfProofs_sha512`). -/
def PlainCallsBindOrSlack (H : Msg → Msg) : Prop :=
  ∀ (x : Msg) (ms : List Msg) (l l' : Msg) (cs : List Coin), ms.length + 1 = (FlockDeploy.V false).k x →
    cs.length = ms.length + 1 →
    (∀ j < ms.length, ((FlockDeploy.next false x (ms.take (j + 1)) (cs.take (j + 1))).eval H).1 =
      .msg (cs.getD j default).bytes) →
    ((FlockDeploy.next false x (ms ++ [l]) cs).eval H).1 = .acc →
    ((FlockDeploy.next false x (ms ++ [l']) cs).eval H).1 = .acc →
    ((FlockDeploy.next false x (ms ++ [l]) cs).eval H).2.toFinset ≠
      ((FlockDeploy.next false x (ms ++ [l']) cs).eval H).2.toFinset →
    (∃ q ∈ (((FlockDeploy.next false x (ms ++ [l]) cs).eval H).2 ++
        ((FlockDeploy.next false x (ms ++ [l']) cs).eval H).2).map Prod.fst,
      ∃ q' ∈ (((FlockDeploy.next false x (ms ++ [l]) cs).eval H).2 ++
        ((FlockDeploy.next false x (ms ++ [l']) cs).eval H).2).map Prod.fst, Claims.Collision H q q') ∨
    ProofSlack l l'

/-- `PlainLastBindsOrSlack` is its columns half and its calls half. -/
theorem plainLastBindsOrSlack_of {H : Msg → Msg} (hC : PlainColsBind H) (hK : PlainCallsBindOrSlack H) :
    PlainLastBindsOrSlack H := by
  intro x ms l l' cs hk hc hearly hacc hacc' hne
  rcases hne with hne | hne
  · exact Or.inl (hC x ms l l' cs hk hc hacc hacc' hne)
  · exact hK x ms l l' cs hk hc hearly hacc hacc' hne

/-- **The columns half at SHA-512.** -/
theorem plainColsBind_sha512 : PlainColsBind FlockDeploy.sha512 := by
  intro x ms l l' cs _ hc hacc hacc' hne
  obtain ⟨u, v, hcol, hu, hv⟩ := FlockDeploy.KS.plain_cols_tied hc hacc hacc' hne
  exact ⟨u, hu, v, hv, hcol⟩

/-- **`PlainLastBinds` at SHA-512 from the decoder's canonicity and the calls half** (`PlainCallsBindOrSlack`, which
holds at SHA-512: `plainLastBinds_sha512_of_canonical`). -/
theorem plainLastBinds_sha512_of_calls (hC : CanonicalProofFiles)
    (hK : PlainCallsBindOrSlack FlockDeploy.sha512) : PlainLastBinds FlockDeploy.sha512 :=
  plainLastBinds_of_canonical hC (plainLastBindsOrSlack_of plainColsBind_sha512 hK)

/-- **(P4) modulo what `E*` reads at the deployed bound record, at SHA-512**, from the decoder's canonicity and the
calls half; the columns half is `plainColsBind_sha512`. -/
theorem lastBindsEC_deployed_sha512 (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (wE : Msg → ℝ) (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hC : CanonicalProofFiles) (hK : PlainCallsBindOrSlack FlockDeploy.sha512) :
    (bdDeployed rel (FlockDeploy.V false).next (FlockDeploy.deployPlainTrim rel).cols wE join rowInDeployed final
      w).LastBindsEC FlockDeploy.sha512 :=
  lastBindsEC_deployed rel wE join final w FlockDeploy.sha512 (plainLastBinds_sha512_of_calls hC hK)

end RecDeploy
