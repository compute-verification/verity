import Proofs.Flock.Claims.Complete.Zk.Json.Files
import Proofs.Flock.Claims.Complete.Walk

/-!
# The masked session's reps, by index

`sessionPZk_facts` and `sessionPZk_files` together, with each stream and proof file tied to the rep that made it
(`sessionPZk_walk`): the session's `j`th stream is the rounds rep `j` ends at, and its `j`th proof file encodes the
proof rep `j` returns, on the table's trace, its masked commitment, the extra claims at the link's points, and rep
`j`'s pads committed (`RepRunZk`). The loop is walked with its index (`forIn_array_idx`).

`RunZk.walk`: at the run, the record decodes to those streams, every recorded round retained, and each rep's decoded
masked proof is the proof that rep returned (`ZkView.decode_encode`, given `RepShapedZk`).
-/

namespace FlockComplete.JsonRT

open Flock Transcript FlockComplete FlockDeploy With
open Lean (Json)

/-- A `for` loop over a list, from position `k` of `s`, with an invariant of the position, the loop's state and the
sends; each step sees the element at its position. -/
theorem forIn_list_idx {α β : Type} (f : α → β → PS (ForInStep β)) (inv : ℕ → β → Array Claims.Msg → Prop)
    (s : List α) (hs : ∀ k a b l r l', s[k]? = some a → inv k b l → (f a b).go l = (.ok r, l') →
      ∃ b', r = .yield b' ∧ inv (k + 1) b' l') :
    ∀ (xs : List α) (k : ℕ), s.drop k = xs → ∀ b l b' l', inv k b l → (forIn xs b f).go l = (.ok b', l') →
      inv (k + xs.length) b' l'
  | [], k, _, b, l, b', l', h, e => by
    simp only [List.forIn_nil, go_pure, Prod.mk.injEq, Except.ok.injEq] at e
    obtain ⟨rfl, rfl⟩ := e
    simpa using h
  | a :: xs, k, hd, b, l, b', l', h, e => by
    rw [List.forIn_cons] at e
    obtain ⟨r, l₁, e1, e2⟩ := go_bind_ok e
    have ha : s[k]? = some a := by
      have := List.getElem?_drop (xs := s) (i := k) (j := 0)
      rw [hd, Nat.add_zero] at this
      rw [← this]
      rfl
    obtain ⟨b₁, rfl, h₁⟩ := hs k a b l r l₁ ha h e1
    have hd' : s.drop (k + 1) = xs := by
      rw [← List.drop_drop, hd]
      rfl
    have := forIn_list_idx f inv s hs xs (k + 1) hd' b₁ l₁ b' l' h₁ e2
    rwa [List.length_cons, show k + (xs.length + 1) = k + 1 + xs.length by omega]

/-- A `for` loop over an array, with an invariant of the steps taken, the loop's state and the sends; each step sees
the element at its index. -/
theorem forIn_array_idx {α β : Type} (xs : Array α) (f : α → β → PS (ForInStep β))
    (inv : ℕ → β → Array Claims.Msg → Prop) (hs : ∀ k a b l r l', xs[k]? = some a → inv k b l →
      (f a b).go l = (.ok r, l') → ∃ b', r = .yield b' ∧ inv (k + 1) b' l')
    {b : β} {l : Array Claims.Msg} {b' : β} {l' : Array Claims.Msg} (h0 : inv 0 b l)
    (e : (forIn xs b f).go l = (.ok b', l')) : inv xs.size b' l' := by
  rw [← Array.forIn_toList] at e
  have := forIn_list_idx f inv xs.toList (fun k a b l r l' ha => hs k a b l r l' (by simpa using ha))
    xs.toList 0 (by simp) b l b' l' h0 e
  simpa using this

theorem push_getElem!_lt {α : Type} [Inhabited α] (xs : Array α) (x : α) {j : ℕ} (hj : j < xs.size) :
    (xs.push x)[j]! = xs[j]! := by
  rw [getElem!_pos (xs.push x) j (by simp; omega), getElem!_pos xs j hj, Array.getElem_push_lt]

theorem push_getElem!_size {α : Type} [Inhabited α] (xs : Array α) (x : α) : (xs.push x)[xs.size]! = x := by
  rw [getElem!_pos (xs.push x) xs.size (by simp), Array.getElem_push_eq]

/-- **Rep `rep`'s run**: the rep, on the table's trace `tr`, its packed table, the masked level-0 commitment `cm`, the
extra claims `e`, and rep `rep`'s pads committed as `ic`, returns a proof with `q`, from the empty transcript at its
domain, ending at the rounds `rs`, its rank check at some earlier claim points and mask words. -/
def RepRunZk (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (spec : SessionSpec) (st : Setup)
    (v : Schedule × ℕ) (tr : Trace) (cm : Commit) (e : Array Claim) (tape : TapeZk) (rep : ℕ) (rs : Array Round)
    (q : Zk.ZkProof → Prop) : Prop :=
  ∃ p ic l t' l' prior ws, innerCommitP (Hashes.ofSha512 h : Hashes V)
      (padsOf tape 0 rep (Zk.PadLayout.new st.m st.kLog v.1.k0)) (tape.innerSalt 0 rep) = .ok ic ∧ q p ∧
    rs = t'.rounds ∧
    ((repPZk (asksS resp) (Hashes.ofSha512 h) st v.1 v.2 rep (spec.domain spec.tables[0]! rep).toUTF8 tr
      (packed tr st.m) cm e (padsOf tape 0 rep (Zk.PadLayout.new st.m st.kLog v.1.k0)) ic (tape.extra 0)
      (tape.lanePad 0) (tape.salt 0 rep) (tape.nonce 0 rep) prior ws).run
        (Transcript.init #[] (spec.domain spec.tables[0]! rep).toUTF8)).go l = (.ok (p, t'), l')

/-- **The one-table masked session, rep by rep**: `sessionPZk_facts` and `sessionPZk_files` on the same streams, the
`j`th stream and proof file being rep `j`'s rounds and proof (`RepRunZk`). -/
theorem sessionPZk_walk (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) {spec : SessionSpec} {t : String}
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
    ∃ v cm e streams, Zk.schedule st.m = .ok v ∧
      commitPZk (Hashes.ofSha512 (Claims.hOf H) : Hashes V) st.merkle (v.1.levels.getD 0 default) v.2
        (packed trs[0]! st.m) (tape.extra 0) (tape.lanePad 0) (tape.salt 0 0 0) = .ok cm ∧
      st.extra (ptsOf spec (coinOf (resp 0))) = .ok e ∧
      r.1 = printJ (recordObjZk spec (Claims.hOf H) hz ct #[flat cm.cap] (ptsOf spec (coinOf (resp 0))) streams
        r.2) ∧
      0 < (flat cm.cap).size ∧ (flat cm.cap).size < 2 ^ 64 ∧ streams.size = 2 ∧ r.2.size = 2 ∧
      (∀ rs ∈ streams, RoundsOk rs) ∧
      (∃ bs, bodies.toList = linkBody [flat cm.cap] [ByteArray.empty] :: bs ∧
        Recs resp 1 (streams.toList.flatMap Array.toList) bs) ∧
      ∀ j < 2, RepRunZk resp (Claims.hOf H) spec st v trs[0]! cm e tape j streams[j]!
        fun p => r.2[j]! = FlockSoundness.Discharge.ZkView.encZk (shapeZk st v.1 e.size) p := by
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
  have hcms := Ret.go (Ret.mapM_array _ (fun _ cm => commitPZk (Hashes.ofSha512 (Claims.hOf H) : Hashes V) st.merkle
      (v.1.levels.getD 0 default) v.2 (packed trs[0]! st.m) (tape.extra 0) (tape.lanePad 0) (tape.salt 0 0 0) = .ok cm)
    _ fun i hi => by
      have : i = 0 := by simpa using hi
      subst this
      exact Ret.of_pureS (commitPZk_pureS _ _ _ _ _ _ _ _) fun cm e => by simpa using e) hB
  have hl2 := Still.go (Still.mapM_array (fun i => by
    exact Still.of_pureS (commitPZk_pureS _ _ _ _ _ _ _ _)) _) hB
  subst hl2
  obtain ⟨cm, rfl, hcm⟩ : ∃ cm, cms = #[cm] ∧ commitPZk (Hashes.ofSha512 (Claims.hOf H) : Hashes V) st.merkle
      (v.1.levels.getD 0 default) v.2 (packed trs[0]! st.m) (tape.extra 0) (tape.lanePad 0) (tape.salt 0 0 0) =
      .ok cm := by
    obtain ⟨hsz, hf⟩ := hcms
    have h1 : cms.size = 1 := by simpa using hsz
    exact ⟨cms[0], eq_singleton_of h1, hf 0 (by simp) (by omega)⟩
  have hcap := hC v hv _ _ _ _ cm hcm
  obtain ⟨c₁, l₃, hL, hrun⟩ := go_bind_ok hrun
  rw [link_go] at hL
  simp only [Prod.mk.injEq, Except.ok.injEq, Array.size_empty] at hL
  obtain ⟨rfl, rfl⟩ := hL
  have hrs1 := repStreams_one ht
  obtain ⟨⟨ss, ps⟩, l₄, hF, hrun⟩ := go_bind_ok hrun
  have hinv := forIn_array_idx _ _ (fun k (b : Array (Array Round) × Array ByteArray) (l : Array Claims.Msg) =>
      b.1.size = k ∧ b.2.size = k ∧ (∀ rs ∈ b.1, RoundsOk rs) ∧
        (0 < k → ∃ e, st.extra (ptsOf spec (coinOf (resp 0))) = .ok e) ∧
        (∀ e, st.extra (ptsOf spec (coinOf (resp 0))) = .ok e → ∀ j < k,
          RepRunZk resp (Claims.hOf H) spec st v trs[0]! cm e tape j b.1[j]!
            fun p => b.2[j]! = FlockSoundness.Discharge.ZkView.encZk (shapeZk st v.1 e.size) p) ∧
        ∃ bs, l.toList = linkBody [flat cm.cap] [ByteArray.empty] :: bs ∧
          Recs resp 1 (b.1.toList.flatMap Array.toList) bs) ?_
    (by exact ⟨rfl, rfl, by simp, by simp, by simp, [], by simp, trivial⟩) hF
  · obtain ⟨h₁, h₂, h₃, h₄, h₅, bs, hbs, hrec⟩ := hinv
    dsimp only at hrun
    rw [recordJson_eqZk _ hct hown hc] at hrun
    obtain ⟨rec, l₅, hJ, hrun⟩ := go_bind_ok hrun
    simp only [go_pure, Prod.mk.injEq, Except.ok.injEq] at hJ hrun
    obtain ⟨rfl, rfl⟩ := hJ
    obtain ⟨rfl, rfl⟩ := hrun
    rw [hrs1] at h₁ h₂ h₄ h₅
    obtain ⟨e, he⟩ := h₄ (by simp)
    refine ⟨v, cm, e, ss, hv, hcm, he, ?_, hcap.1, hcap.2, h₁, h₂, h₃, ⟨bs, hbs, hrec⟩, h₅ e he⟩
    simp only [List.map_toArray, List.map_cons, List.map_nil]
    rfl
  · rintro k ⟨i, rp, name⟩ ⟨ss, ps⟩ l res l' hk ⟨h₁, h₂, h₃, -, h₅, bs, hl, hrec⟩ e
    have hk' : i = 0 ∧ k = rp := by
      rw [hrs1] at hk
      rcases k with _ | _ | k <;> simp at hk
      · exact ⟨hk.1.symm, hk.2.1⟩
      · exact ⟨hk.1.symm, hk.2.1⟩
    obtain ⟨rfl, hkr⟩ := hk'
    subst hkr
    dsimp only at h₁ h₂ h₅
    obtain ⟨extras, l₁, e1, e⟩ := go_bind_ok e
    rw [monadLift_go] at e1
    simp only [Prod.mk.injEq] at e1
    obtain ⟨hex, rfl⟩ := e1
    obtain ⟨ic, l₀, e0, e3⟩ := go_bind_ok e
    have hic := Ret.go (Ret.of_pureS (innerCommitP_pureS _ _ _) (P := fun ic => innerCommitP
      (Hashes.ofSha512 (Claims.hOf H) : Hashes V) (padsOf tape 0 k (Zk.PadLayout.new st.m st.kLog v.1.k0))
      (tape.innerSalt 0 k) = .ok ic) fun _ e => by simpa using e) e0
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
    refine ⟨_, rfl, by simp [h₁], by simp [h₂], fun q hq => ?_, fun _ => ⟨extras, hex'⟩, fun e' he' j hj => ?_,
      bs ++ bs', by rw [hl₂, hl]; rfl, ?_⟩
    · rcases Array.mem_push.1 hq with hq | rfl
      · exact h₃ q hq
      · have hrs' : tr'.rounds.toList = rs := by simpa [Transcript.init] using hrs
        refine ⟨hok.1, fun x hx => ?_⟩
        have := Recs.mem hrec' x (by rw [← hrs']; exact Array.mem_toList_iff.2 hx)
        exact ⟨this.1, hok.2 x hx, this.2⟩
    · rw [hex'] at he'
      cases he'
      rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | rfl
      · rw [push_getElem!_lt _ _ (by omega), push_getElem!_lt _ _ (by omega)]
        exact h₅ extras hex' j hj
      · refine ⟨pf, ic, l, tr', l₂, priorOf st ss j, words[0]!, hic, ?_, ?_, by simpa using e2⟩
        · rw [← h₂, push_getElem!_size]
          simp
        · rw [← h₁, push_getElem!_size]
    · simp only [Transcript.init, List.nil_append] at hrs
      simp only [Array.toList_push, List.flatMap_append, List.flatMap_cons, List.flatMap_nil, List.append_nil, hrs]
      exact Recs.append hrec (by rw [← hsz]; exact hrec')

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- **The honest masked run, rep by rep**: at the run's one table, with `--zk` schedule `v`, the record decodes to two
streams whose every round is retained, rep `r`'s recorded rounds are its stream's as the decoder reads them back, the
verifier's extra claims are the prover's, the table has `25 ≤ m`, and rep `r`'s decoded masked proof is the proof it
returned, ending at that stream. -/
theorem RunZk.walk (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hS : StmtOkZk H rel)
    (hP : ProverOkZk H rel) (hR : RepShapedZk H rel) (o : RunZk H rel) :
    ∃ (st : Flock.Stmt) (v : Schedule × ℕ) (cm : Commit) (e : Array Claim) (streams : Array (Array Round)),
      o.tabs = #[Setup.ofCircuit st] ∧ o.sch 0 = v.1 ∧ streams.size = 2 ∧
      (∀ rs ∈ streams, ∀ x ∈ rs, ∃ c, x.bytes = some c) ∧
      (∀ r < 2, (o.stream 0 r).rounds = streams[r]!.map (roundRec o.spec (Claims.hOf H))) ∧ o.extras 0 = e ∧ 25 ≤ st.m ∧
      ∀ r < 2, RepRunZk o.resp (Claims.hOf H) o.spec (Setup.ofCircuit st) v (tracesOf o.sts o.w)[0]! cm e o.ω r
        streams[r]! fun p => o.zp 0 r = p := by
  obtain ⟨st, hst, hsp, h25, -⟩ := o.table
  obtain ⟨hsig, hdraw, -, m, kvs, hown, hm, hkv, hco, hg, ha, hT⟩ := hS o.toCaseZk
  obtain ⟨hz, hc⟩ := S.canon_ok _ hg ha
  have hH : HelloOk o.spec hz := ⟨m, kvs, hown, hm, hkv, hco, hg, hc⟩
  obtain ⟨ct, hct⟩ : ∃ ct, o.spec.coinTree = some ct := Option.isSome_iff_exists.1 o.ready_parts.2.2.2.2
  obtain ⟨hC, hRk⟩ := hP o.toCaseZk st (by rw [hst]; simp)
  have ht : o.spec.tables = #[st.tags.table] := by rw [hsp]; rfl
  have htabs : tabsOf #[st] = #[Setup.ofCircuit st] := by simp [tabsOf]
  have hrun := o.ran
  unfold sessionSZk at hrun
  rw [hst, htabs] at hrun
  obtain ⟨v, cm, e, streams, hv, hcm, he, hrec, hrb, -, hss, -, hrs, ⟨bs, -, hrecs⟩, hreps⟩ :=
    sessionPZk_walk H _ ht hct hown hc (Setup.ofCircuit st) _ _ _ hC hRk hrun
  rw [show o.resp 0 = (Claims.coinSeq o.r 0).bytes from rfl, ptsOf_coin] at hrec he
  have hpts : (pointsOf o.spec (Claims.coinSeq o.r 0)).size = o.spec.points := by simp [pointsOf]
  have hpm : ∀ p ∈ pointsOf o.spec (Claims.coinSeq o.r 0), p.size = o.spec.mPts := by
    intro p hp
    obtain ⟨i, -, rfl⟩ := Array.mem_map.1 hp
    simp
  dsimp only at hrec hreps
  have hrs1 := repStreams_one ht
  have hdec : Record.decodeWith o.hs o.spec o.record = .ok (sessOf o.spec (Claims.hOf H) #[flat cm.cap]
      (pointsOf o.spec (Claims.coinSeq o.r 0)) streams o.proofs) := by
    rw [hrec]
    exact decode_recordObjZk S (hOf_size h64) ht hct (hT ct hct) hH hdraw hsig hrb hpts hpm (by rw [hss, hrs1]; rfl) hrs
      o.proofs
  have hsess : o.sess = sessOf o.spec (Claims.hOf H) #[flat cm.cap] (pointsOf o.spec (Claims.coinSeq o.r 0)) streams
      o.proofs := by
    unfold RunZk.sess
    rw [hdec]
    rfl
  have hotabs : o.tabs = #[Setup.ofCircuit st] := by simp [RunZk.tabs, tabsOf, hst]
  have hsch0 : o.sch 0 = v.1 := by simp [RunZk.sch, zkSchedsOf, schedZkOf, hotabs, hv]
  have hext0 : o.extras 0 = e := by
    unfold RunZk.extras
    rw [hotabs, hsess, show (#[Setup.ofCircuit st])[0]! = Setup.ofCircuit st from rfl]
    simp only [sessOf, he]
    rfl
  obtain ⟨s0, s1, rfl⟩ := two_of hss
  have hstream : ∀ r < 2, (o.stream 0 r).rounds = (#[s0, s1])[r]!.map (roundRec o.spec (Claims.hOf H)) := by
    intro r hr
    unfold RunZk.stream
    rw [hsess]
    simp only [sessOf, hrs1, ht, List.zip_toArray, List.zip_cons_cons, List.zip_nil_right, List.map_toArray,
      List.map_cons, List.map_nil, List.find?_toArray, streamRec]
    rcases (show r = 0 ∨ r = 1 by omega) with rfl | rfl <;> simp
  refine ⟨st, v, cm, e, #[s0, s1], hotabs, hsch0, rfl, fun rs hrs x hx => ?_, hstream, hext0, h25, fun r hr => ?_⟩
  · exact Recs.bytes hrecs x (List.mem_flatMap.2 ⟨rs, Array.mem_toList_iff.2 hrs, Array.mem_toList_iff.2 hx⟩)
  · obtain ⟨p, ic, l, t', l', prior, ws, hic, hb, hrr, hp⟩ := hreps r hr
    have hR' := hR o.toCaseZk st v cm e hst hv (by rw [hst]; exact hcm) he r ic hic
      (o.spec.domain o.spec.tables[0]! r).toUTF8 prior ws
    rw [hst] at hR'
    have hps := Ret.go hR' hp
    refine ⟨p, ic, l, t', l', prior, ws, hic, ?_, hrr, by rw [hst]; exact hp⟩
    have hdec' : Zk.decode (shapeZk o.tabs[0]! (o.sch 0) (o.extras 0).size) (o.bytes 0 r) = .ok p := by
      rw [show o.bytes 0 r = o.proofs[r]! by simp [RunZk.bytes], hb, hsch0, hext0,
        show o.tabs[0]! = Setup.ofCircuit st by rw [hotabs]; rfl]
      exact FlockSoundness.Discharge.ZkView.decode_encode _ _ hps
    unfold RunZk.zp
    rw [hdec']
    rfl

end FlockComplete.JsonRT
