import Proofs.Flock.Claims.Complete.Json.Live
import Proofs.Flock.Claims.Complete.Zk.Link
import Proofs.Flock.Claims.Complete.Zk.Json.Decode

/-!
# What the one-table honest masked session returns and sends

`sessionP_facts` at the masked session (`sessionPZk_facts`): at one table, with the keyed coin tree, a run of the pure
masked session from no sends returns the printed honest masked record (`printJ` of `recordObjZk`) of a non-empty
`root_B` under `2^64` bytes, the link's points off the first reply's coin, and one well-formed stream per rep stream;
the table's extra claims evaluate at those points; and its sends are the link of that `root_B` and no public words,
then the streams' rounds in order (`SessionFactsZk`).

* **The rounds**: every masked prover function from the zerocheck to the rep adds rounds that are the sends it adds
  (`zerocheckPZk_rec`, …, `repPZk_rec`), as `Session`'s plain ones; the verifier's own replay and level 0's lane phase
  inside the rep (`StateT.lift` of a `V` computation) and the read of the transcript (`get`) add none.
* **The quiet parts**: the masked commitment and the pads' commitment send nothing (`commitPZk_pureS`,
  `innerCommitP_pureS`).
-/

namespace FlockComplete.JsonRT

open Flock Transcript FlockComplete FlockDeploy With
open Lean (Json JsonNumber)

/-! ## The rounds -/

theorem get_rec (resp : ℕ → Claims.Msg) : (recS resp).R (get : TMW PS Transcript) get := by
  intro t l a t' l' h
  simp only [StateT.run_get, go_pure, Prod.mk.injEq, Except.ok.injEq] at h
  obtain ⟨⟨-, rfl⟩, rfl⟩ := h
  exact Rec.same rfl rfl

theorem liftV_rec (resp : ℕ → Claims.Msg) {α : Type} (v : V α) :
    (recS resp).R (StateT.lift (monadLift v : PS α) : TMW PS α) (StateT.lift (monadLift v : PS α)) := by
  intro t l a t' l' e
  change ((monadLift v : PS _) >>= fun a => Pure.pure (a, t)).go l = _ at e
  obtain ⟨b, l₁, e1, e2⟩ := go_bind_ok e
  rw [monadLift_go] at e1
  simp only [Prod.mk.injEq] at e1
  obtain ⟨-, rfl⟩ := e1
  cases e2
  exact Rec.same rfl rfl

/-- A pure computation of `V`, run in `PS` and lifted to the transcript, adds no rounds and no sends. -/
theorem liftPure_rec (resp : ℕ → Claims.Msg) {α : Type} {x : V α} {y : PS α} (hxy : pureS.R x y) :
    (recS resp).R (StateT.lift y : TMW PS α) (StateT.lift y) := by
  intro t l a t' l' e
  change (y >>= fun a => Pure.pure (a, t)).go l = _ at e
  obtain ⟨b, l₁, e1, e2⟩ := go_bind_ok e
  have := Still.go (Still.of_pureS hxy) e1
  subst this
  cases e2
  exact Rec.same rfl rfl

theorem zerocheckPZk_rec (resp : ℕ → Claims.Msg) (m : ℕ) (digest cap root : ByteArray) (lay : Zk.PadLayout)
    (h : Array F128) (tr : Trace) :
    (recS resp).R (zerocheckPZk (asksS resp) m digest cap root lay h tr)
      (zerocheckPZk (asksS resp) m digest cap root lay h tr) := by
  unfold zerocheckPZk sampleVecP
  sim_walk [squeezeP_rec resp, sampleP_rec resp, label_rec resp, absorbBytes_rec resp, absorbSlice_rec resp,
    absorbF_rec resp]

theorem lincheckPZk_rec (resp : ℕ → Claims.Msg) (c : CircuitFold) (m : ℕ) (lay : Zk.PadLayout) (h : Array F128)
    (tr : Trace) (zc : ZcOut) :
    (recS resp).R (lincheckPZk (asksS resp) c m lay h tr zc) (lincheckPZk (asksS resp) c m lay h tr zc) := by
  unfold lincheckPZk
  sim_walk [sampleP_rec resp, label_rec resp, absorbSlice_rec resp, absorbF_rec resp]

theorem commitRowsP_pureS (h : ByteArray → ByteArray) (ms : MerkleScheme) (lv : Level) (rows : Array (Array F128))
    (salt : ℕ → ByteArray) :
    pureS.R (commitRowsP (Hashes.ofSha512 h : Hashes V) ms lv rows salt)
      (commitRowsP (Hashes.ofSha512 h : Hashes PS) ms lv rows salt) := by
  have ho := With.opsWith_sim pureS (pureS_hashes h) ms
  unfold commitRowsP
  sim_walk [ho.leaf, ho.node]

theorem commitPZk_pureS (h : ByteArray → ByteArray) (ms : MerkleScheme) (lv : Level) (tPad : ℕ) (g : Array F128)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ByteArray) :
    pureS.R (commitPZk (Hashes.ofSha512 h : Hashes V) ms lv tPad g extra lanePad salt)
      (commitPZk (Hashes.ofSha512 h : Hashes PS) ms lv tPad g extra lanePad salt) :=
  commitRowsP_pureS h ms lv _ salt

theorem innerCommitP_pureS (h : ByteArray → ByteArray) (p : PadsZk) (salt : ℕ → ByteArray) :
    pureS.R (innerCommitP (Hashes.ofSha512 h : Hashes V) p salt)
      (innerCommitP (Hashes.ofSha512 h : Hashes PS) p salt) := by
  have hh := pureS_hashes h
  unfold innerCommitP MerkleOps.hm96With
  sim_walk [leafWith_sim pureS hh]

theorem ligeritoPZk_rec (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (ms : MerkleScheme) (sch : Schedule)
    (T : F128) (cm0 : Commit) (t b : Array F128) (lay : Zk.PadLayout) (hp : Array F128) (rep tPad : ℕ)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) :
    (recS resp).R
      (ligeritoPZk (asksS resp) (Hashes.ofSha512 h) ms sch T cm0 t b lay hp rep tPad extra lanePad salt nonce)
      (ligeritoPZk (asksS resp) (Hashes.ofSha512 h) ms sch T cm0 t b lay hp rep tPad extra lanePad salt nonce) := by
  unfold ligeritoPZk sampleVecP
  sim_walk [liftT_rec resp, squeezeP_rec resp, sampleP_rec resp, sampleKP_rec resp, sendMsg_rec resp, powP_rec resp,
    challengeP_rec resp, oodIntroP_rec resp, queryP_rec resp, commitP_rec resp h, label_rec resp, absorbF_rec resp,
    absorbBytes_rec resp, absorbK_rec resp, absorbSlice_rec resp]

theorem tau_rec (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (tau : F128) (salt : ByteArray) :
    (recS resp).R (StateT.lift (Zk.tauCommitmentWith (Hashes.ofSha512 h : Hashes PS) tau salt) : TMW PS ByteArray)
      (StateT.lift (Zk.tauCommitmentWith (Hashes.ofSha512 h : Hashes PS) tau salt)) :=
  liftPure_rec resp (zk_tauCommitmentWith_sim pureS (pureS_hashes h) tau salt)

theorem innerP_rec (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (lay : Zk.PadLayout) (p : PadsZk)
    (ic : InnerCommit) (finalC : F128) (cons : Array (String × Zk.Lin)) :
    (recS resp).R (innerP (asksS resp) (Hashes.ofSha512 h) lay p ic finalC cons)
      (innerP (asksS resp) (Hashes.ofSha512 h) lay p ic finalC cons) := by
  unfold innerP sampleVecP
  sim_walk [sampleP_rec resp, squeezeP_rec resp, label_rec resp, absorbF_rec resp, absorbSlice_rec resp,
    absorbBytes_rec resp, tau_rec resp h]

/-- **A masked rep's new rounds are its new sends.** -/
theorem repPZk_rec (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (st : Setup) (sch : Schedule) (tPad rep : ℕ)
    (dom : ByteArray) (tr : Trace) (t : Array F128) (cm0 : Commit) (extras : Array Claim) (pads : PadsZk)
    (ic : InnerCommit) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (prior : Array (List F128)) (words : List ℕ) :
    (recS resp).R
      (repPZk (asksS resp) (Hashes.ofSha512 h) st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce
        prior words)
      (repPZk (asksS resp) (Hashes.ofSha512 h) st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt
        nonce prior words) := by
  unfold repPZk rankCheckP
  sim_walk [zerocheckPZk_rec resp, lincheckPZk_rec resp, ringSwitchP_rec resp, ligeritoPZk_rec resp h,
    innerP_rec resp h, get_rec resp, liftV_rec resp]

/-! ## The session -/

/-- **What the one-table masked session returns and sends**: its record is the honest masked record of a non-empty
`root_B`, the link's points off the first reply's coin, and one well-formed stream per rep stream; the table's extra
claims evaluate at those points; and its sends are the link of that `root_B` and no public words, then the streams'
rounds in order, each from index `1` on (`Recs`). -/
def SessionFactsZk (h : ByteArray → ByteArray) (resp : ℕ → Claims.Msg) (spec : SessionSpec) (st : Setup)
    (hz : String) (ct : CoinTree.Spec) (r : String × Array ByteArray) (bodies : Array Claims.Msg) : Prop :=
  ∃ rb streams, r.1 = printJ (recordObjZk spec h hz ct #[rb] (ptsOf spec (coinOf (resp 0))) streams r.2) ∧
    0 < rb.size ∧ rb.size < 2 ^ 64 ∧ streams.size = spec.repStreams.size ∧ (∀ rs ∈ streams, RoundsOk rs) ∧
    (∃ e, st.extra (ptsOf spec (coinOf (resp 0))) = .ok e) ∧
    ∃ bs, bodies.toList = linkBody [rb] [ByteArray.empty] :: bs ∧ Recs resp 1 (streams.toList.flatMap Array.toList) bs

/-- **The one-table masked session's facts**, given a non-empty masked level-0 cap under `2^64` bytes and well-formed
rep rounds on the table's extra claims at the link's points. -/
theorem sessionPZk_facts (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) {spec : SessionSpec} {t : String}
    (ht : spec.tables = #[t]) {ct : CoinTree.Spec} (hct : spec.coinTree = some ct) {own : Json}
    (hown : Json.parse spec.hello = .ok own) {hz : String} (hc : canon (helloJ own) = .ok hz) (st : Setup)
    (words : Array (List ℕ)) (trs : Array Trace) (tape : TapeZk)
    (hC : ∀ v, Zk.schedule st.m = .ok v → ∀ g extra lanePad salt cm,
      commitPZk (Hashes.ofSha512 (Claims.hOf H) : Hashes V) st.merkle (v.1.levels.getD 0 default) v.2 g extra lanePad
        salt = .ok cm → 0 < (flat cm.cap).size ∧ (flat cm.cap).size < 2 ^ 64)
    (hR : ∀ v, Zk.schedule st.m = .ok v → ∀ rep dom tr tb cm₀ extras pads ic extra lanePad salt nonce prior ws,
      st.extra (ptsOf spec (coinOf (resp 0))) = .ok extras →
      Ret ((repPZk (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) st v.1 v.2 rep dom tr tb cm₀ extras pads ic extra
        lanePad salt nonce prior ws).run (Transcript.init #[] dom)) fun pt => RepOk pt.2.rounds)
    {r : String × Array ByteArray} {bodies : Array Claims.Msg}
    (hrun : (sessionPZk (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) spec #[st] words trs tape).go #[] =
      (.ok r, bodies)) :
    SessionFactsZk (Claims.hOf H) resp spec st hz ct r bodies := by
  unfold sessionPZk at hrun
  obtain ⟨schs, l1, hA, hrun⟩ := go_bind_ok hrun
  have hschs := Ret.go (Ret.mapM_array _ (fun st' v => Zk.schedule st'.m = .ok v) _ fun a _ =>
    Ret.monadLift (Zk.schedule a.m)) hA
  have hl1 := Still.go (Still.mapM_array (fun a l => by rw [monadLift_go]) _) hA
  subst hl1
  obtain ⟨v, rfl, hv⟩ : ∃ v, schs = #[v] ∧ Zk.schedule st.m = .ok v := by
    obtain ⟨hsz, hf⟩ := hschs
    have h1 : schs.size = 1 := by simpa using hsz
    exact ⟨schs[0], eq_singleton_of h1, by simpa using hf 0 (by simp) (by omega)⟩
  dsimp only at hrun
  obtain ⟨cms, l2, hB, hrun⟩ := go_bind_ok hrun
  have hcms := Ret.go (Ret.mapM_array _ (fun _ cm => 0 < (flat cm.cap).size ∧ (flat cm.cap).size < 2 ^ 64) _
    fun i hi => by
      have : i = 0 := by simpa using hi
      subst this
      exact Ret.of_pureS (commitPZk_pureS _ _ _ _ _ _ _ _) fun cm e => hC v hv _ _ _ _ cm (by simpa using e)) hB
  have hl2 := Still.go (Still.mapM_array (fun i => by
    exact Still.of_pureS (commitPZk_pureS _ _ _ _ _ _ _ _)) _) hB
  subst hl2
  obtain ⟨cm, rfl, hcm⟩ : ∃ cm, cms = #[cm] ∧ 0 < (flat cm.cap).size ∧ (flat cm.cap).size < 2 ^ 64 := by
    obtain ⟨hsz, hf⟩ := hcms
    have h1 : cms.size = 1 := by simpa using hsz
    exact ⟨cms[0], eq_singleton_of h1, hf 0 (by simp) (by omega)⟩
  obtain ⟨c₁, l₃, hL, hrun⟩ := go_bind_ok hrun
  rw [link_go] at hL
  simp only [Prod.mk.injEq, Except.ok.injEq, Array.size_empty] at hL
  obtain ⟨rfl, rfl⟩ := hL
  have hrs1 := repStreams_one ht
  obtain ⟨⟨ss, ps⟩, l₄, hF, hrun⟩ := go_bind_ok hrun
  have hinv := forIn_array_inv _ _ (fun k (b : Array (Array Round) × Array ByteArray) (l : Array Claims.Msg) =>
      b.1.size = k ∧ b.2.size = k ∧ (∀ rs ∈ b.1, RoundsOk rs) ∧
        (0 < k → ∃ e, st.extra (ptsOf spec (coinOf (resp 0))) = .ok e) ∧
        ∃ bs, l.toList = linkBody [flat cm.cap] [ByteArray.empty] :: bs ∧
          Recs resp 1 (b.1.toList.flatMap Array.toList) bs) ?_
    (by exact ⟨rfl, rfl, by simp, by simp, [], by simp, trivial⟩) hF
  · obtain ⟨h₁, h₂, h₃, h₄, bs, hbs, hrec⟩ := hinv
    dsimp only at hrun
    rw [recordJson_eqZk _ hct hown hc] at hrun
    obtain ⟨rec, l₅, hJ, hrun⟩ := go_bind_ok hrun
    simp only [go_pure, Prod.mk.injEq, Except.ok.injEq] at hJ hrun
    obtain ⟨rfl, rfl⟩ := hJ
    obtain ⟨rfl, rfl⟩ := hrun
    refine ⟨flat cm.cap, ss, ?_, hcm.1, hcm.2, h₁, h₃, h₄ (by rw [hrs1]; simp), bs, hbs, hrec⟩
    simp only [List.map_toArray, List.map_cons, List.map_nil]
    rfl
  · rintro k ⟨i, r, name⟩ ⟨ss, ps⟩ l res l' hmem ⟨h₁, h₂, h₃, -, bs, hl, hrec⟩ e
    have hi : i = 0 := by
      rw [hrs1] at hmem
      simp at hmem
      omega
    subst hi
    obtain ⟨extras, l₁, e1, e⟩ := go_bind_ok e
    rw [monadLift_go] at e1
    simp only [Prod.mk.injEq] at e1
    obtain ⟨hex, rfl⟩ := e1
    obtain ⟨ic, l₀, e0, e3⟩ := go_bind_ok e
    rw [Still.go (Still.of_pureS (innerCommitP_pureS _ _ _)) e0] at e3
    obtain ⟨⟨pf, tr'⟩, l₂, e2, e⟩ := go_bind_ok e3
    have hex' : st.extra (ptsOf spec (coinOf (resp 0))) = .ok extras := by simpa [ptsOf] using hex
    have hok := Ret.go (hR v hv _ _ _ _ _ _ _ _ _ _ _ _ _ _ hex') e2
    obtain ⟨rs, bs', hrs, hl₂, hrec'⟩ := Rec.go (repPZk_rec resp (Claims.hOf H) _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _) e2
    simp only [go_pure, Prod.mk.injEq, Except.ok.injEq] at e
    obtain ⟨rfl, rfl⟩ := e
    have hsz : l.size = 1 + (ss.toList.flatMap Array.toList).length := by
      rw [← Array.length_toList, hl, Recs.length_eq hrec]
      simp only [List.length_cons]
      omega
    refine ⟨_, rfl, by simp [h₁], by simp [h₂], fun q hq => ?_, fun _ => ⟨extras, hex'⟩,
      bs ++ bs', by rw [hl₂, hl]; rfl, ?_⟩
    · rcases Array.mem_push.1 hq with hq | rfl
      · exact h₃ q hq
      · have hrs' : tr'.rounds.toList = rs := by simpa [Transcript.init] using hrs
        refine ⟨hok.1, fun x hx => ?_⟩
        have := Recs.mem hrec' x (by rw [← hrs']; exact Array.mem_toList_iff.2 hx)
        exact ⟨this.1, hok.2 x hx, this.2⟩
    · simp only [Transcript.init, List.nil_append] at hrs
      simp only [Array.toList_push, List.flatMap_append, List.flatMap_cons, List.flatMap_nil, List.append_nil, hrs]
      exact Recs.append hrec (by rw [← hsz]; exact hrec')

end FlockComplete.JsonRT
