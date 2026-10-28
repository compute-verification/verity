import Proofs.Flock.Claims.Complete.Zk.Json.Goals
import Proofs.Flock.Claims.Complete.Files

/-!
# The honest masked run's proof files

`Files.lean` at the masked run: the rep goals on the masked proof files, `RepDecodesZk`, `RepParamsZk`, `PadsRoot` and
`CapCommittedZk`, and `FinalFitsZk` (the record's length from `readyZk`, `RunZk.recordFits`), at every honest masked
run, under the masked record goals' hypotheses (`JsonSpec`, `Out64`, `StmtOkZk`, `ProverOkZk`) and one more:

* `RepShapedZk`: each masked proof a rep returns, on the case's trace, its masked level-0 commitment under the case's
  tape, the table's extra claims at the link's points, and the tape's pads committed, is `Shaped` at the shape the
  masked verifier decodes it at, so `ZkView.decode_encode` reads it back.

`sessionPZk_files`: each proof file of the one-table masked session encodes a masked proof its rep returned
(`RepFileZk`), and the first send is the link of the masked commitment's cap. `RunZk.rep`: at the run, the masked
verifier's decoded proof is that rep's proof, whose cap is the commitment's (the record's `root_B`), whose PCS
parameters are the table's, and whose pads' root is their commitment's (`repPZk_fields`, `innerP_root`), a 64-byte
digest at 64-byte oracle outputs (`innerCommitP_root`).
-/

namespace FlockComplete.JsonRT

open Flock Transcript FlockComplete FlockDeploy

/-- A `throw` in the transcript monad never returns. -/
theorem Ret.run_throw {α : Type} {P : α × Transcript → Prop} (e : String) (t : Transcript) :
    Ret ((throw e : TMW PS α).run t) P := by
  intro l a l' h
  change ((throw e : PS _) >>= fun a => Pure.pure (a, t)).go l = _ at h
  rw [go_bind] at h
  cases h

/-- Nor does anything after it. -/
theorem Ret.run_throw_bind {α β : Type} {P : β × Transcript → Prop} (e : String) (f : α → TMW PS β)
    (t : Transcript) : Ret ((throw e >>= f : TMW PS β).run t) P := by
  rw [StateT.run_bind]
  exact Ret.bind (Q := fun _ => False) (Ret.run_throw e t) fun _ h => h.elim

/-- **The inner proof carries its commitment's root.** -/
theorem innerP_root (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (lay : Zk.PadLayout) (p : PadsZk)
    (ic : InnerCommit) (finalC : F128) (cons : Array (String × Zk.Lin)) (t : Transcript) :
    Ret ((innerP (asksS resp) (Hashes.ofSha512 h) lay p ic finalC cons).run t) fun pt => pt.1.root = ic.root := by
  unfold innerP
  simp only [StateT.run_bind]
  refine Ret.bind (Ret.true _) fun a _ => ?_
  refine Ret.bind (Ret.true _) fun b _ => ?_
  refine Ret.bind (Ret.true _) fun c _ => ?_
  refine Ret.bind (Ret.true _) fun d _ => ?_
  refine Ret.bind (Ret.true _) fun e _ => ?_
  split
  · exact Ret.run_throw_bind _ _ _
  simp only [StateT.run_bind]
  refine Ret.bind (Ret.true _) fun f _ => ?_
  refine Ret.bind (Ret.true _) fun g _ => ?_
  refine Ret.bind (Ret.true _) fun k _ => ?_
  split
  · exact Ret.run_throw_bind _ _ _
  simp only [StateT.run_bind]
  refine Ret.bind (Ret.true _) fun m _ => ?_
  refine Ret.bind (Ret.true _) fun n _ => ?_
  exact Ret.pure rfl

/-- **A masked rep's proof carries the commitment's cap, the table's PCS parameters and its pads' root.** -/
theorem repPZk_fields (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (st : Setup) (sch : Schedule)
    (tPad rep : ℕ) (dom : ByteArray) (tr : Trace) (t : Array F128) (cm0 : Commit) (extras : Array Claim)
    (pads : PadsZk) (ic : InnerCommit) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray)
    (nonce : ℕ → ℕ → UInt64) (prior : Array (List F128)) (words : List ℕ) (tr₀ : Transcript) :
    Ret ((repPZk (asksS resp) (Hashes.ofSha512 h) st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt
      nonce prior words).run tr₀) fun pt => pt.1.proof.cap = cm0.cap ∧ pt.1.proof.params = pcsParams st.merkle sch ∧
        pt.1.inner.root = ic.root := by
  unfold repPZk
  simp only [StateT.run_bind]
  refine Ret.bind (Ret.true _) fun a _ => ?_
  refine Ret.bind (Ret.true _) fun b _ => ?_
  refine Ret.bind (Ret.true _) fun _ _ => ?_
  refine Ret.bind (Ret.true _) fun c _ => ?_
  refine Ret.bind (Ret.true _) fun d _ => ?_
  refine Ret.bind (Ret.true _) fun e _ => ?_
  refine Ret.bind (Ret.true _) fun f _ => ?_
  refine Ret.bind (Ret.true _) fun g _ => ?_
  refine Ret.bind (innerP_root resp h _ _ _ _ _ _) fun i hi => ?_
  exact Ret.pure ⟨rfl, rfl, hi⟩


/-- A masked proof file of a rep: it encodes, at the rep's shape, a masked proof the rep returned on the table's trace
`tr`, its packed table, the masked level-0 commitment `cm`, the extra claims `e`, and the tape's pads, committed, at
some rank check. -/
def RepFileZk (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (st : Setup) (v : Schedule × ℕ) (tr : Trace)
    (cm : Commit) (e : Array Claim) (tape : TapeZk) (b : ByteArray) : Prop :=
  ∃ p rep ic d l t' l' prior ws, innerCommitP (Hashes.ofSha512 h : Hashes V)
      (padsOf tape 0 rep (Zk.PadLayout.new st.m st.kLog v.1.k0)) (tape.innerSalt 0 rep) = .ok ic ∧
    b = FlockSoundness.Discharge.ZkView.encZk (shapeZk st v.1 e.size) p ∧
    ((repPZk (asksS resp) (Hashes.ofSha512 h) st v.1 v.2 rep d tr (packed tr st.m) cm e
      (padsOf tape 0 rep (Zk.PadLayout.new st.m st.kLog v.1.k0)) ic (tape.extra 0) (tape.lanePad 0) (tape.salt 0 rep)
      (tape.nonce 0 rep) prior ws).run (Transcript.init #[] d)).go l = (.ok (p, t'), l')

/-- **The one-table masked session's proof files**: the table's `--zk` schedule, its masked level-0 commitment `cm` and
its extra claims `e` at the link's points; the first send is the link of `cm`'s cap; one proof file per rep stream,
each a `RepFileZk`. -/
theorem sessionPZk_files (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) {spec : SessionSpec} {t : String}
    (ht : spec.tables = #[t]) {ct : CoinTree.Spec} (hct : spec.coinTree = some ct) {own : Lean.Json}
    (hown : Lean.Json.parse spec.hello = .ok own) {hz : String} (hc : canon (helloJ own) = .ok hz) (st : Setup)
    (words : Array (List ℕ)) (trs : Array Trace) (tape : TapeZk) {r : String × Array ByteArray}
    {bodies : Array Claims.Msg}
    (hrun : (sessionPZk (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) spec #[st] words trs tape).go #[] =
      (.ok r, bodies)) :
    ∃ v cm e, Zk.schedule st.m = .ok v ∧
      commitPZk (Hashes.ofSha512 (Claims.hOf H) : Hashes V) st.merkle (v.1.levels.getD 0 default) v.2
        (packed trs[0]! st.m) (tape.extra 0) (tape.lanePad 0) (tape.salt 0 0 0) = .ok cm ∧
      st.extra (ptsOf spec (coinOf (resp 0))) = .ok e ∧
      (∃ bs, bodies.toList = linkBody [flat cm.cap] [ByteArray.empty] :: bs) ∧
      r.2.size = spec.repStreams.size ∧ ∀ b ∈ r.2, RepFileZk resp (Claims.hOf H) st v trs[0]! cm e tape b := by
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
  obtain ⟨c₁, l₃, hL, hrun⟩ := go_bind_ok hrun
  rw [link_go] at hL
  simp only [Prod.mk.injEq, Except.ok.injEq, Array.size_empty] at hL
  obtain ⟨rfl, rfl⟩ := hL
  have hrs1 := repStreams_one ht
  obtain ⟨⟨ss, ps⟩, l₄, hF, hrun⟩ := go_bind_ok hrun
  have hinv := forIn_array_inv _ _ (fun k (b : Array (Array Round) × Array ByteArray) (l : Array Claims.Msg) =>
      b.2.size = k ∧ (0 < k → ∃ e, st.extra (ptsOf spec (coinOf (resp 0))) = .ok e) ∧
        (∀ e, st.extra (ptsOf spec (coinOf (resp 0))) = .ok e →
          ∀ q ∈ b.2, RepFileZk resp (Claims.hOf H) st v trs[0]! cm e tape q) ∧
        ∃ bs, l.toList = linkBody [flat cm.cap] [ByteArray.empty] :: bs) ?_
    (by exact ⟨rfl, by simp, by simp, [], by simp⟩) hF
  · obtain ⟨h₁, h₂, h₃, bs, hbs⟩ := hinv
    dsimp only at hrun
    rw [recordJson_eqZk _ hct hown hc] at hrun
    obtain ⟨rec, l₅, hJ, hrun⟩ := go_bind_ok hrun
    simp only [go_pure, Prod.mk.injEq, Except.ok.injEq] at hJ hrun
    obtain ⟨rfl, rfl⟩ := hJ
    obtain ⟨rfl, rfl⟩ := hrun
    obtain ⟨e, he⟩ := h₂ (by rw [hrs1]; simp)
    exact ⟨v, cm, e, hv, hcm, he, ⟨bs, hbs⟩, h₁, h₃ e he⟩
  · rintro k ⟨i, r, name⟩ ⟨ss, ps⟩ l res l' hmem ⟨h₁, h₂, h₃, bs, hl⟩ e
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
    have hic := Ret.go (Ret.of_pureS (innerCommitP_pureS _ _ _) (P := fun ic => innerCommitP
      (Hashes.ofSha512 (Claims.hOf H) : Hashes V) (padsOf tape 0 r (Zk.PadLayout.new st.m st.kLog v.1.k0))
      (tape.innerSalt 0 r) = .ok ic) fun _ e => by simpa using e) e0
    rw [Still.go (Still.of_pureS (innerCommitP_pureS _ _ _)) e0] at e3
    obtain ⟨⟨pf, tr'⟩, l₂, e2, e⟩ := go_bind_ok e3
    obtain ⟨rs, bs', hrs, hl₂, hrec'⟩ := Rec.go (repPZk_rec resp (Claims.hOf H) _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _) e2
    simp only [go_pure, Prod.mk.injEq, Except.ok.injEq] at e
    obtain ⟨rfl, rfl⟩ := e
    have hex' : st.extra (ptsOf spec (coinOf (resp 0))) = .ok extras := by simpa [ptsOf] using hex
    refine ⟨_, rfl, by simp [h₁], fun _ => ⟨extras, hex'⟩, fun e he q hq => ?_, bs ++ bs', by rw [hl₂, hl]; rfl⟩
    rcases Array.mem_push.1 hq with hq | rfl
    · exact h₃ e he q hq
    · rw [hex'] at he
      cases he
      exact ⟨pf, r, ic, (spec.domain spec.tables[0]! r).toUTF8, l, tr', l₂, priorOf st ss r, words[0]!, hic,
        by simp, by simpa using e2⟩


variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- **The honest masked rep proof fits its shape**: at every good masked case, on its table's trace, the trace's packed
table committed at the `--zk` schedule's first level under the case's tape (`cm`), the table's extra claims at the
link's points (`e`), and the tape's pads committed (`ic`), each masked proof a rep returns is `Shaped` at the shape the
masked verifier decodes it at. -/
def RepShapedZk (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ c : CaseZk H rel, ∀ st v cm e, c.sts = #[st] → Zk.schedule (Setup.ofCircuit st).m = .ok v →
    commitPZk (Hashes.ofSha512 (Claims.hOf H) : Hashes V) (Setup.ofCircuit st).merkle (v.1.levels.getD 0 default) v.2
      (packed (tracesOf c.sts c.w)[0]! (Setup.ofCircuit st).m) (c.ω.extra 0) (c.ω.lanePad 0) (c.ω.salt 0 0 0) =
      .ok cm →
    (Setup.ofCircuit st).extra (pointsOf c.spec (Claims.coinSeq c.r 0)) = .ok e →
    ∀ rep ic, innerCommitP (Hashes.ofSha512 (Claims.hOf H) : Hashes V)
      (padsOf c.ω 0 rep (Zk.PadLayout.new (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog v.1.k0))
      (c.ω.innerSalt 0 rep) = .ok ic →
    ∀ d prior ws, Ret ((repPZk (asksS c.resp) (Hashes.ofSha512 (Claims.hOf H)) (Setup.ofCircuit st) v.1 v.2 rep d
      (tracesOf c.sts c.w)[0]! (packed (tracesOf c.sts c.w)[0]! (Setup.ofCircuit st).m) cm e
      (padsOf c.ω 0 rep (Zk.PadLayout.new (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog v.1.k0)) ic
      (c.ω.extra 0) (c.ω.lanePad 0) (c.ω.salt 0 rep) (c.ω.nonce 0 rep) prior ws).run (Transcript.init #[] d)) fun pt =>
      FlockSoundness.Discharge.ZkView.Shaped (shapeZk (Setup.ofCircuit st) v.1 e.size) pt.1

/-- **Each rep's decoded masked proof is the honest one**: at the run's one table, with `--zk` schedule `v` and masked
level-0 commitment `cm`, the record's `root_B` is `cm`'s cap, two proof files, each under `2^64` bytes, and each rep's
file decodes at the masked verifier's shape to a proof with `cm`'s cap, the table's PCS parameters, and the root of its
pads' commitment. -/
theorem RunZk.rep (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hS : StmtOkZk H rel)
    (hP : ProverOkZk H rel) (hR : RepShapedZk H rel) (o : RunZk H rel) :
    ∃ (st : Flock.Stmt) (v : Schedule × ℕ) (cm : Commit), o.tabs = #[Setup.ofCircuit st] ∧ o.sch 0 = v.1 ∧
      o.sess.rootB = #[flat cm.cap] ∧ o.proofs.size = 2 ∧ (∀ b ∈ o.proofs, b.size < 2 ^ 64) ∧ ∀ r < 2,
        Zk.decode (shapeZk o.tabs[0]! (o.sch 0) (o.extras 0).size) (o.bytes 0 r) = .ok (o.zp 0 r) ∧
        (o.zp 0 r).proof.cap = cm.cap ∧ (o.zp 0 r).proof.params = pcsParams (Setup.ofCircuit st).merkle v.1 ∧
        ∃ pads isalt ic, innerCommitP (Hashes.ofSha512 (Claims.hOf H) : Hashes V) pads isalt = .ok ic ∧
          (o.zp 0 r).inner.root = ic.root := by
  obtain ⟨st, rb, streams, hz, ct, hst, ht, htabs, hss, hpf, hex, hlink, -, hrec, hdec, hsess, hlv, hlive⟩ :=
    RunZk.json S h64 hS hP o
  obtain ⟨-, -, -, m, kvs, hown, -, -, -, hg, ha, -⟩ := hS o.toCaseZk
  obtain ⟨hz', hc⟩ := S.canon_ok _ hg ha
  obtain ⟨ct', hct⟩ : ∃ ct, o.spec.coinTree = some ct := Option.isSome_iff_exists.1 o.ready_parts.2.2.2.2
  obtain ⟨hC, -⟩ := hP o.toCaseZk st (by rw [hst]; simp)
  have hrun := o.ran
  unfold sessionSZk at hrun
  have htabs' : tabsOf #[st] = #[Setup.ofCircuit st] := by simp [tabsOf]
  rw [hst, htabs'] at hrun
  obtain ⟨v, cm, e, hv, hcm, he, ⟨bs, hbs⟩, hsz, hfiles⟩ :=
    sessionPZk_files H o.resp ht hct hown hc (Setup.ofCircuit st) _ _ _ hrun
  rw [show o.resp 0 = (Claims.coinSeq o.r 0).bytes from rfl, ptsOf_coin] at he
  have hcap := hC v hv _ _ _ _ cm hcm
  have hl : o.link = ([flat cm.cap], [ByteArray.empty]) := by
    unfold RunZk.link RunZk.sends
    rw [hbs]
    show (linkOf (0 :: linkBody [flat cm.cap] [ByteArray.empty])).getD ([], []) = _
    rw [linkOf_linkBody (by simp) (by simp) (by simpa using hcap.2) (by simp)]
    rfl
  rw [hlink] at hl
  obtain rfl : rb = flat cm.cap := by simpa using hl
  have hsch0 : o.sch 0 = v.1 := by simp [RunZk.sch, zkSchedsOf, schedZkOf, htabs, hv]
  have hext0 : o.extras 0 = e := by
    unfold RunZk.extras
    rw [htabs, hsess, show (#[Setup.ofCircuit st])[0]! = Setup.ofCircuit st from rfl]
    simp only [sessOf, he]
    rfl
  have hrs := repStreams_one ht
  have hsz2 : o.proofs.size = 2 := by rw [hsz, hrs]; rfl
  have hshaped : ∀ b ∈ o.proofs, ∃ p, b = FlockSoundness.Discharge.ZkView.encZk
      (shapeZk (Setup.ofCircuit st) v.1 e.size) p ∧
      FlockSoundness.Discharge.ZkView.Shaped (shapeZk (Setup.ofCircuit st) v.1 e.size) p ∧
      p.proof.cap = cm.cap ∧ p.proof.params = pcsParams (Setup.ofCircuit st).merkle v.1 ∧
      ∃ pads isalt ic, innerCommitP (Hashes.ofSha512 (Claims.hOf H) : Hashes V) pads isalt = .ok ic ∧
        p.inner.root = ic.root := by
    intro b hb
    obtain ⟨p, rep, ic, d, l, t', l', prior, ws, hic, rfl, hp⟩ := hfiles b hb
    have hR' := hR o.toCaseZk st v cm e hst hv (by rw [hst]; exact hcm) he rep ic hic d prior ws
    rw [hst] at hR'
    have h1 := Ret.go hR' hp
    have h2 := Ret.go (repPZk_fields _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _) hp
    exact ⟨p, rfl, h1, h2.1, h2.2.1, _, _, ic, hic, h2.2.2⟩
  refine ⟨st, v, cm, htabs, hsch0, by rw [hsess]; rfl, hsz2, fun b hb => ?_, fun r hr => ?_⟩
  · obtain ⟨p, rfl, hps, -⟩ := hshaped b hb
    exact hps.2.2.1
  · have hmem : o.bytes 0 r ∈ o.proofs := by
      unfold RunZk.bytes
      rw [Nat.mul_zero, Nat.zero_add, getElem!_pos o.proofs r (by omega)]
      exact Array.getElem_mem _
    obtain ⟨p, hb, hps, hc, hpar, hin⟩ := hshaped _ hmem
    have hdec : Zk.decode (shapeZk o.tabs[0]! (o.sch 0) (o.extras 0).size) (o.bytes 0 r) = .ok p := by
      rw [hb, hsch0, hext0, show o.tabs[0]! = Setup.ofCircuit st by rw [htabs]; rfl]
      exact FlockSoundness.Discharge.ZkView.decode_encode _ _ hps
    have hp : o.zp 0 r = p := by
      unfold RunZk.zp
      rw [hdec]
      rfl
    rw [hp]
    exact ⟨hdec, hc, hpar, hin⟩


/-- A loop whose every step yields, and takes an invariant at `n + 1` remaining steps to it at `n`, ends at `0`. -/
theorem except_forIn_count {ε α β : Type} (P : ℕ → β → Prop) {f : α → β → Except ε (ForInStep β)}
    (hf : ∀ a b r n, P (n + 1) b → f a b = .ok r → ∃ v, r = .yield v ∧ P n v) :
    ∀ (l : List α) (b r : β), P l.length b → forIn l b f = .ok r → P 0 r := by
  intro l
  induction l with
  | nil =>
    intro b r hb h
    rw [List.forIn_nil] at h
    cases h
    exact hb
  | cons a l ih =>
    intro b r hb h
    rw [List.forIn_cons] at h
    obtain ⟨s, hs, h⟩ := except_ok_bind h
    obtain ⟨v, rfl, hv⟩ := hf a b s l.length hb hs
    exact ih v r hv h

/-- **The pads' root is a digest**, for a SHA-512 of 64-byte outputs: the tree has `2^k` leaves, each level halves it,
and the root is the one node left. -/
theorem innerCommitP_root {h : ByteArray → ByteArray} (h64 : ∀ b, (h b).size = 64) (p : PadsZk)
    (salt : ℕ → ByteArray) {ic : InnerCommit}
    (hc : (innerCommitP (Hashes.ofSha512 h) p salt : V InnerCommit) = .ok ic) : ic.root.size = 64 := by
  unfold innerCommitP at hc
  obtain ⟨leaves, hl, hc⟩ := except_ok_bind hc
  obtain ⟨s, hs, hc⟩ := except_ok_bind hc
  cases hc
  have hL := mapM_sized (fun _ _ hr => by cases hr; exact h64 _) hl
  rw [Array.size_range] at hL
  obtain ⟨k, hk⟩ := Nat.isPowerOfTwo_nextPowerOfTwo (p.h ++ p.padH).size
  rw [hk, ← Nat.pow_add] at hL hs
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hs
  have hfin := except_forIn_count (fun n (b : Array (Array ByteArray) × Array ByteArray) =>
    b.2.size = 2 ^ n ∧ ∀ x ∈ b.2, x.size = 64) ?_ _ _ _ ?_ hs
  · obtain ⟨h1, h2⟩ := hfin
    show s.2[0]!.size = 64
    exact h2 _ (by rw [getElem!_pos s.2 0 (by rw [h1]; simp)]; exact Array.getElem_mem _)
  · rintro - ⟨layers, cur⟩ r n ⟨h1, -⟩ hr
    obtain ⟨cur', hc', hr⟩ := except_ok_bind hr
    cases hr
    have hN := mapM_sized (fun _ _ hr => hm96_node h64 _ _ hr) hc'
    rw [Array.size_range, h1, Nat.pow_succ, Nat.mul_div_cancel _ (by norm_num)] at hN
    exact ⟨_, rfl, hN⟩
  · simpa [List.length_range', Std.Legacy.Range.size, Nat.log2_two_pow] using hL


/-- **The rep's masked proof file decodes** at its shape: the goal `RepDecodesZk`. -/
theorem repDecodesZk_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hS : StmtOkZk H rel)
    (hP : ProverOkZk H rel) (hR : RepShapedZk H rel) : RepDecodesZk H rel := by
  intro o i hi r hr
  obtain ⟨st, v, cm, htabs, -, -, -, -, hrep⟩ := RunZk.rep S h64 hS hP hR o
  rw [htabs] at hi
  obtain rfl : i = 0 := by simpa using hi
  exact (hrep r hr).1

/-- **The rep's PCS parameters are the `--zk` schedule's** (R7/F2): the goal `RepParamsZk`. -/
theorem repParamsZk_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hS : StmtOkZk H rel)
    (hP : ProverOkZk H rel) (hR : RepShapedZk H rel) : RepParamsZk H rel := by
  intro o i hi r hr
  obtain ⟨st, v, cm, htabs, hsch0, -, -, -, hrep⟩ := RunZk.rep S h64 hS hP hR o
  rw [htabs] at hi
  obtain rfl : i = 0 := by simpa using hi
  rw [(hrep r hr).2.2.1, hsch0, show o.tabs[0]! = Setup.ofCircuit st by rw [htabs]; rfl]
  exact FlockVBridge.Family.pcs_bne_self _

/-- **The pads' root is 64 bytes**: the goal `PadsRoot`. -/
theorem padsRoot_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hS : StmtOkZk H rel)
    (hP : ProverOkZk H rel) (hR : RepShapedZk H rel) : PadsRoot H rel := by
  intro o i hi r hr
  obtain ⟨st, v, cm, htabs, -, -, -, -, hrep⟩ := RunZk.rep S h64 hS hP hR o
  rw [htabs] at hi
  obtain rfl : i = 0 := by simpa using hi
  obtain ⟨-, -, -, pads, isalt, ic, hic, hroot⟩ := hrep r hr
  rw [hroot, innerCommitP_root (hOf_size h64) _ _ hic]
  rfl

/-- **The rep's cap is the committed `root_B`** (S16): the goal `CapCommittedZk`. -/
theorem capCommittedZk_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hS : StmtOkZk H rel)
    (hP : ProverOkZk H rel) (hR : RepShapedZk H rel) : CapCommittedZk H rel := by
  intro o i hi r hr
  obtain ⟨st, v, cm, htabs, -, hroot, -, -, hrep⟩ := RunZk.rep S h64 hS hP hR o
  rw [htabs] at hi
  obtain rfl : i = 0 := by simpa using hi
  rw [(hrep r hr).2.1, hroot]
  show (flat cm.cap != flat cm.cap) = false
  rw [bne, bytes_beq_self]
  rfl

/-- **The last message's lengths fit their words**: the goal `FinalFitsZk`, the printed record's length from `readyZk`
(`RunZk.recordFits`). -/
theorem finalFitsZk_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hS : StmtOkZk H rel)
    (hP : ProverOkZk H rel) (hR : RepShapedZk H rel) : FinalFitsZk H rel := by
  intro o
  obtain ⟨st, v, cm, -, -, -, hsz, hb, -⟩ := RunZk.rep S h64 hS hP hR o
  exact ⟨o.recordFits, by omega, hb⟩

end FlockComplete.JsonRT
