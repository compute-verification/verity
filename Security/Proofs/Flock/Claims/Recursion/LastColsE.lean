import Proofs.Flock.Claims.Recursion.CallsAgreeE
import Proofs.Flock.Claims.Recursion.LastColsDeployed
import Proofs.Flock.Claims.Recursion.LastBindsClosed

/-!
# (P4) at the element-level inner verifier

The columns half at `VE` (`FlockDeploy.KS.plainE_cols_tied`), by `plain_cols_tied`'s proof: it reads the verdict only
through the rep streams' round counts (`streams_sizes_plainK`, `rounds_ns_plainK`), table 0's query round
(`runOf_queryRound_plainK`), the reps' checks (`verifyWithMK_tables`) and the decoder's live part (`decodeWithK_live`),
all at any `keep`, and none of them reads a digest of an encoding. With the calls half at `VE`
(`RecDeploy.plainCallsBindOrSlackE_sha512`) and V*'s canonical proof-file decoder, `VE` binds its last message
(`RecDeploy.plainLastBindsE_sha512`), and the deployed bound record at `VE`'s steps meets (P4) modulo what `E*` reads,
with no premise (`RecDeploy.lastBindsEC_deployedE_sha512`).
-/

namespace FlockDeploy.KS

open Flock
open FlockSoundness.Refine (ebind_ok merkleCheck_cap)

local notation "𝔽" => FlockLevel3.GF128

/-- `streams_coins_eq` at `verifyWithMK keep` and `keep'`. -/
theorem streams_coins_eqK (keep keep' : Bool → Bool) (hs hs' : Hashes Flock.V) {spec : SessionSpec}
    {sts : Array Stmt} {record record' : String} {proofs proofs' : Array ByteArray} {lv lv' : Live}
    {link link' : List ByteArray × List ByteArray} {c₁ c₁' : Coin} {mids : List (Claims.Msg × Coin)}
    (hv : verifyWithMK keep hs spec (tabsOf sts) record proofs = .ok ()) (hl : liveOf spec record = .ok lv)
    (hv' : verifyWithMK keep' hs' spec (tabsOf sts) record' proofs' = .ok ()) (hl' : liveOf spec record' = .ok lv')
    (hc : custody spec lv link c₁ mids = .ok ()) (hc' : custody spec lv' link' c₁' mids = .ok ()) :
    lv.streams.toList.map (fun s => s.toList.map Round.coins) =
      lv'.streams.toList.map (fun s => s.toList.map Round.coins) := by
  refine coins_streams_eq
    ((streams_sizes_plainK keep hs hv hl).trans (streams_sizes_plainK keep' hs' hv' hl').symm) ?_
  rw [coins_of_forall₂ (Claims.Bridge.Custody.custody_ok hc).2.2.2 (fun _ _ h => h.2),
    coins_of_forall₂ (Claims.Bridge.Custody.custody_ok hc').2.2.2 (fun _ _ h => h.2),
    rounds_ns_plainK keep hs hv hl, rounds_ns_plainK keep' hs' hv' hl']

/-- `rawCols_keys` at `verifyWithMK keep` and `keep'`. -/
theorem rawCols_keysK (keep keep' : Bool → Bool) {hs hs' : Hashes Flock.V} {seen seen' : Claims.Seen}
    {f f' : Final} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt} {st : Stmt} {extras : Array Flock.Claim}
    {lv lv' : Live} (hf : lastFinal seen = some f) (hf' : lastFinal seen' = some f')
    (hset : setupOf false f.stmt = some (ins, spec, sts)) (hset' : setupOf false f'.stmt = some (ins, spec, sts))
    (hv : verifyWithMK keep hs spec (tabsOf sts) f.record f.proofs = .ok ())
    (hv' : verifyWithMK keep' hs' spec (tabsOf sts) f'.record f'.proofs = .ok ())
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
    obtain ⟨p, rd, -, hrd, -, hmc, hc⟩ := runOf_queryRound_plainK keep hs hf hset hv hr _ h₁
    obtain ⟨p', rd', -, hrd', -, hmc', hc'⟩ := runOf_queryRound_plainK keep' hs' hf' hset' hv' hr' _ h₂
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

/-- `plain_opening_at` at the element-level verdict: an element-level decision's reps open a column only as a Merkle
opening of table 0's level 0 within the calls `L` its verdict ran in. -/
theorem plainE_opening_at {x : Claims.Msg} {seen : Claims.Seen} {L : List (Claims.Msg × Claims.Msg)} {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (hs : setupOf false x = some (s, spec, sts))
    (hf : lastFinal seen = some f) (hc : checks false s spec x link c₁ mids f = .ok ())
    (hv : verifyWithME (hashesIn (L.map Prod.fst)) spec (tabsOf sts) f.record f.proofs = .ok ())
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
  obtain ⟨sess, schs, hdec, hsz, -, hall⟩ := verifyWithMK_tables (fun _ => true) _ hv
  obtain ⟨hs0, d₀, d₁, hsch, h₀, h₁⟩ := hall 0 h0
  obtain ⟨d, hrk⟩ : ∃ d, verifyRepWith (hashesIn (L.map Prod.fst)) spec (tabsOf sts)[0] schs[0] sess 0 k
      f.proofs[2 * 0 + k]! = .ok d := by
    interval_cases k
    · exact ⟨d₀, by simpa using h₀⟩
    · exact ⟨d₁, by simpa using h₁⟩
  have hfold := tabsOf_fold sts _ (Array.getElem_mem h0)
  obtain ⟨stream, extras', p, hstream, hext', hp', hlt, hcols, hmc, hcap⟩ :=
    plainRepColsOf_verifyRepWith _ spec _ _ sess 0 k _ d hsch hfold.1 hfold.2 hrk
  generalize schs[0] = sch at hsch hp' hlt hcols hmc
  simp only [ht0] at hsch hext' hp' hlt hcols hmc
  obtain ⟨hrB, hpts, -, hstreams⟩ := decodeWithK_live (fun _ => true) _ hdec hlv
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

/-- **Two last messages the element-level verifier accepts after the same rounds and coins, that differ in table 0's
opened columns, collide among the two decisions' calls**, at SHA-512 (`plain_cols_tied` at `nextE`). -/
theorem plainE_cols_tied {x : Claims.Msg} {ms : List Claims.Msg} {l l' : Claims.Msg} {cs : List Coin}
    (hc : cs.length = ms.length + 1)
    (hacc : ((nextE x (ms ++ [l]) cs).eval sha512).1 = .acc)
    (hacc' : ((nextE x (ms ++ [l']) cs).eval sha512).1 = .acc)
    (hne : cols false (RecDeploy.plainView (ms ++ [l]) cs) 0 ≠ cols false (RecDeploy.plainView (ms ++ [l']) cs) 0) :
    Claims.TiedColl sha512 (((nextE x (ms ++ [l]) cs).eval sha512).2 ++
      ((nextE x (ms ++ [l']) cs).eval sha512).2) := by
  obtain ⟨s, spec, sts, link, c₁, mids, f, hact, hrun⟩ := nextE_acc_decide hacc
  obtain ⟨s', spec', sts', link', c₁', mids', f', hact', hrun'⟩ := nextE_acc_decide hacc'
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
  obtain ⟨hvr, hck⟩ := decisionE_runs (setupOf_some hs).2 hrun
  obtain ⟨hvr', hck'⟩ := decisionE_runs (setupOf_some hs).2 hrun'
  have hw := verdictE_within hvr
  have hw' := verdictE_within hvr'
  generalize hL₁ : ((nextE x (b :: ms'' ++ [l]) cs).eval sha512).2 = L₁ at hw ⊢
  generalize hL₂ : ((nextE x (b :: ms'' ++ [l']) cs).eval sha512).2 = L₂ at hw' ⊢
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
  have hV : verifyWithMK (fun _ => true) (hashesIn (L₁.map Prod.fst)) spec (tabsOf sts) f.record f.proofs =
      .ok () := hw
  have hV' : verifyWithMK (fun _ => true) (hashesIn (L₂.map Prod.fst)) spec (tabsOf sts) f'.record f'.proofs =
      .ok () := hw'
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
      have hS := streams_coins_eqK _ _ _ _ hV hlv hV' hlv' hcu hcu'
      obtain ⟨q, u, v, hu, hv, huv⟩ :=
        exists_mem_snd_ne (rawCols_keysK _ _ hf₁ hf₂ hset₁ hset₂ hV hV' hr₁ hr₂ hS) hraw
      obtain ⟨st₁, sch, row, hst₁, hsch, ho, hu'⟩ := plainE_opening_at hs hf₁ hck hw hu
      obtain ⟨st₂, sch', row', hst₂, hsch', ho', hv'⟩ := plainE_opening_at hs hf₂ hck' hw' hv
      rw [hst] at hst₁ hst₂
      cases hst₁
      cases hst₂
      rw [hsch] at hsch'
      cases hsch'
      exact tiedColl_of_openings ho ho' fun e => huv (by rw [hu', hv', e])

end FlockDeploy.KS

namespace RecDeploy

open Claims (Msg Out OC)
open Claims.Recursion (Bound bindOC)
open FlockDeploy (Coin)

/-- `PlainColsBind` at the element-level verifier. -/
def PlainColsBindE (H : Msg → Msg) : Prop :=
  ∀ (x : Msg) (ms : List Msg) (l l' : Msg) (cs : List Coin), ms.length + 1 = FlockDeploy.VE.k x →
    cs.length = ms.length + 1 →
    ((FlockDeploy.nextE x (ms ++ [l]) cs).eval H).1 = .acc →
    ((FlockDeploy.nextE x (ms ++ [l']) cs).eval H).1 = .acc →
    FlockDeploy.cols false (plainView (ms ++ [l]) cs) 0 ≠ FlockDeploy.cols false (plainView (ms ++ [l']) cs) 0 →
    ∃ q ∈ (((FlockDeploy.nextE x (ms ++ [l]) cs).eval H).2 ++
        ((FlockDeploy.nextE x (ms ++ [l']) cs).eval H).2).map Prod.fst,
      ∃ q' ∈ (((FlockDeploy.nextE x (ms ++ [l]) cs).eval H).2 ++
        ((FlockDeploy.nextE x (ms ++ [l']) cs).eval H).2).map Prod.fst, Claims.Collision H q q'

/-- **The columns half at `VE`, at SHA-512.** -/
theorem plainColsBindE_sha512 : PlainColsBindE FlockDeploy.sha512 := by
  intro x ms l l' cs _ hc hacc hacc' hne
  obtain ⟨u, v, hcol, hu, hv⟩ := FlockDeploy.KS.plainE_cols_tied hc hacc hacc' hne
  exact ⟨u, hu, v, hv, hcol⟩

/-- `PlainLastBinds` at the element-level verifier. -/
def PlainLastBindsE (H : Msg → Msg) : Prop :=
  ∀ (x : Msg) (ms : List Msg) (l l' : Msg) (cs : List Coin), ms.length + 1 = FlockDeploy.VE.k x →
    cs.length = ms.length + 1 →
    (∀ j < ms.length, ((FlockDeploy.nextE x (ms.take (j + 1)) (cs.take (j + 1))).eval H).1 =
      .msg (cs.getD j default).bytes) →
    ((FlockDeploy.nextE x (ms ++ [l]) cs).eval H).1 = .acc →
    ((FlockDeploy.nextE x (ms ++ [l']) cs).eval H).1 = .acc →
    (FlockDeploy.cols false (plainView (ms ++ [l]) cs) 0 ≠ FlockDeploy.cols false (plainView (ms ++ [l']) cs) 0 ∨
      ((FlockDeploy.nextE x (ms ++ [l]) cs).eval H).2.toFinset ≠
        ((FlockDeploy.nextE x (ms ++ [l']) cs).eval H).2.toFinset) →
    ∃ q ∈ (((FlockDeploy.nextE x (ms ++ [l]) cs).eval H).2 ++
        ((FlockDeploy.nextE x (ms ++ [l']) cs).eval H).2).map Prod.fst,
      ∃ q' ∈ (((FlockDeploy.nextE x (ms ++ [l]) cs).eval H).2 ++
        ((FlockDeploy.nextE x (ms ++ [l']) cs).eval H).2).map Prod.fst, Claims.Collision H q q'

/-- **`PlainLastBindsE` from a canonical decoder and its two halves** (`plainLastBinds_of_canonical` with
`plainLastBindsOrSlack_of`, at `VE`). -/
theorem plainLastBindsE_of {H : Msg → Msg} (hC : CanonicalProofFiles) (hcols : PlainColsBindE H)
    (hK : PlainCallsBindOrSlackE H) : PlainLastBindsE H := by
  intro x ms l l' cs hk hc hearly hacc hacc' hne
  rcases hne with hne | hne
  · exact hcols x ms l l' cs hk hc hacc hacc' hne
  · rcases hK x ms l l' cs hk hc hearly hacc hacc' hne with hcol | ⟨_, _, _, b, b', -, -, -, -, hbb, sh, p, hd, hd'⟩
    · exact hcol
    · exact absurd (hC sh b b' p hd hd') hbb

/-- **`VE` binds its last message, at SHA-512, with no premise.** -/
theorem plainLastBindsE_sha512 : PlainLastBindsE FlockDeploy.sha512 :=
  plainLastBindsE_of canonicalProofFiles plainColsBindE_sha512 plainCallsBindOrSlackE_sha512

/-- `next_lastStep` at `nextE`: `VE`'s last step reads no last coin. -/
theorem nextE_lastStep (H : Msg → Msg) (x : Msg) (ms : List Msg) (L : List Coin) (c c' : Coin)
    (hk : ms.length = FlockDeploy.VE.k x) (hc : L.length + 1 = ms.length) :
    (bindOC (FlockDeploy.nextE x ms (L ++ [c])) fun b => OC.ret (decide (b = Out.acc))).eval H =
      (bindOC (FlockDeploy.nextE x ms (L ++ [c'])) fun b => OC.ret (decide (b = Out.acc))).eval H := by
  by_cases hz : ∃ t, ms.getLast? = some (0 :: t)
  · obtain ⟨t, ht⟩ := hz
    have hfalse : ∀ c₀ : Coin,
        (bindOC (FlockDeploy.nextE x ms (L ++ [c₀])) fun b => OC.ret (decide (b = Out.acc))).eval H =
          (false, []) := by
      intro c₀
      rw [Claims.Recursion.LastGap.eval_bind]
      rcases FlockDeploy.act_snoc_zero (zk := false) (x := x) (cs := L) (c := c₀) ht with h | h
      · rw [FlockDeploy.nextE_rej H h]
        rfl
      · rw [FlockDeploy.nextE_coin H h]
        rfl
    rw [hfalse, hfalse]
  · have ha : ∀ t, ms.getLast? ≠ some (0 :: t) := fun t h => hz ⟨t, h⟩
    rcases L with _ | ⟨c₀, l⟩
    · have hk1 : (FlockDeploy.V false).k x = 1 := by
        have hk' : ms.length = (FlockDeploy.V false).k x := hk
        simp only [List.length_nil, Nat.zero_add] at hc
        omega
      have hsu : FlockDeploy.setupOf false x = none := by
        rcases h : FlockDeploy.setupOf false x with _ | ⟨s, spec, sts⟩
        · rfl
        · simp only [FlockDeploy.V, h] at hk1
          omega
      have hr : ∀ c₀ : Coin, FlockDeploy.act false x ms ([] ++ [c₀]) = .rej := fun c₀ => by
        unfold FlockDeploy.act
        rw [hsu]
      rw [Claims.Recursion.LastGap.eval_bind, Claims.Recursion.LastGap.eval_bind, FlockDeploy.nextE_rej H (hr c),
        FlockDeploy.nextE_rej H (hr c')]
    · have heq : FlockDeploy.nextE x ms ((c₀ :: l) ++ [c']) = FlockDeploy.nextE x ms ((c₀ :: l) ++ [c]) := by
        simp only [FlockDeploy.nextE, FlockDeploy.act_cons_snoc false x ms c₀ l c c' ha]
      rw [heq]

/-- **`LastStepFree` at every bound record whose `DE` steps as `VE`.** -/
theorem lastStepFree_of_VE {Col T : Type} (H : Msg → Msg) (Bd : Bound Coin Col T) (hV : Bd.DE.V = FlockDeploy.VE) :
    Bd.LastStepFree H := by
  intro x ms cs cs' hk hc hc' hd
  rw [hV] at hk
  have hne : ∀ l : List Coin, l.length = ms.length → l ≠ [] := fun l hl h => by
    have h0 := V_k_pos_flock false x
    subst h
    simp only [List.length_nil] at hl
    change ms.length = (FlockDeploy.V false).k x at hk
    omega
  obtain ⟨L, c, rfl⟩ : ∃ L c, cs = L ++ [c] :=
    ⟨cs.dropLast, cs.getLast (hne cs hc), (List.dropLast_concat_getLast (hne cs hc)).symm⟩
  obtain ⟨L', c', rfl⟩ : ∃ L c, cs' = L ++ [c] :=
    ⟨cs'.dropLast, cs'.getLast (hne cs' hc'), (List.dropLast_concat_getLast (hne cs' hc')).symm⟩
  rw [List.dropLast_concat, List.dropLast_concat] at hd
  subst hd
  rw [hV]
  exact nextE_lastStep H x ms L c c' hk (by simp only [List.length_append, List.length_singleton] at hc; exact hc)

/-- **`LastStepBinds` at every bound record whose `DE` steps as `VE`**, answers a round with its coin's bytes and reads
a view's columns only through table 0's, from `PlainLastBindsE` (`lastStepBinds_of_V` at `VE`). -/
theorem lastStepBinds_of_VE {T : Type} (H : Msg → Msg) (Bd : Bound Coin (List FlockLevel3.GF128) T)
    (hV : Bd.DE.V = FlockDeploy.VE) (hans : ∀ c, Bd.ans c = c.bytes)
    (hcols : ∀ v v' : Claims.Seen, FlockDeploy.cols false v 0 = FlockDeploy.cols false v' 0 →
      Bd.DE.cols v = Bd.DE.cols v') (h : PlainLastBindsE H) : Bd.LastStepBinds H := by
  intro x ms l l' cs hk hc hearly hacc hacc' hne
  have hview : ∀ ms' : List Msg, Bd.lastView ms' cs = plainView ms' cs := fun ms' => by
    unfold Bound.lastView plainView
    simp only [hans]
  rw [hV] at hk hacc hacc' hne ⊢
  simp only [hV, hans] at hearly
  rw [hview, hview] at hne
  exact h x ms l l' cs hk hc hearly hacc hacc' (hne.imp_left fun hc0 heq => hc0 (hcols _ _ heq))

/-- **(P4) modulo what `E*` reads, at the deployed bound record at `VE`'s steps, at SHA-512, with no premise**
(`lastBindsEC_deployed_sha512_closed` at `VE`): at `VE`'s steps, the deployed plain record's trimmed columns and the
deployed rows' digest input, whatever the code side's weights, join and `final`. -/
theorem lastBindsEC_deployedE_sha512 (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (wE : Msg → ℝ) (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    (bdDeployed rel FlockDeploy.VE.next (FlockDeploy.deployPlainTrim rel).cols wE join rowInDeployed final
      w).LastBindsEC FlockDeploy.sha512 :=
  Bound.lastBindsEC_of_lastStep (binds_deployed rel _ _ wE join final w) (lastStepFree_of_VE _ _ rfl)
    (lastStepBinds_of_VE _ _ rfl (bdDeployed_ans rel _ _ wE join rowInDeployed final w)
      (fun _ _ heq => by rw [bdDeployed_cols]; exact plainTrim_cols_congr rel heq) plainLastBindsE_sha512)

end RecDeploy
