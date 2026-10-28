import Proofs.Flock.Claims.Complete.Shaped
import Proofs.Flock.VBridge.Family.Files

/-!
# The honest run's proof files

The rep goals on the proof files, `RepDecodes`, `RepParams` and `CapCommitted`, and `FinalFits` (the record's length
from `ready`, `Run.recordFits`), at every honest run, under the record goals' hypotheses (`JsonSpec`, `Claims.Out64`,
`ProverOk`). Each proof a
rep returns is `ProofShaped` at the shape the verifier decodes it at (`Shaped`'s `repShaped`), so
`decodeProof_encodeProof` reads it back.

`sessionP_files`: each proof file of the one-table session encodes a proof its rep returned, on the table's level-0
commitment and its extra claims, and the first send is the link of that commitment's cap. `Run.rep`: at the run, the
verifier's decoded proof is that rep's proof, whose cap is the commitment's (the record's `root_B`) and whose PCS
parameters are the table's (`repP_fields`).
-/

namespace FlockComplete.JsonRT

open Flock Transcript FlockComplete FlockDeploy

/-- **A rep's proof carries the commitment's cap and the table's PCS parameters.** -/
theorem repP_fields (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (st : Setup) (sch : Schedule) (tr : Trace)
    (t : Array F128) (cm0 : Commit) (extras : Array Claim) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (tr₀ : Transcript) :
    Ret ((repP (asksS resp) (Hashes.ofSha512 h) st sch tr t cm0 extras salt nonce).run tr₀)
      fun pt => pt.1.cap = cm0.cap ∧ pt.1.params = pcsParams st.merkle sch := by
  unfold repP
  simp only [StateT.run_bind]
  refine Ret.bind (Ret.true _) fun a _ => ?_
  refine Ret.bind (Ret.true _) fun b _ => ?_
  refine Ret.bind (Ret.true _) fun c _ => ?_
  refine Ret.bind (Ret.true _) fun d _ => ?_
  exact Ret.pure ⟨rfl, rfl⟩

/-- A proof file of a rep: it encodes, at the rep's shape, a proof the rep returned on the table's trace `tr`, its packed
table, the level-0 commitment `cm` and the extra claims `e`. -/
def RepFile (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (st : Setup) (sch : Schedule) (tr : Trace) (cm : Commit)
    (e : Array Claim) (b : ByteArray) : Prop :=
  ∃ p salt nonce d l t' l', b = FlockVBridge.Family.encodeProof (shapeOf st sch e.size) p ∧
    ((repP (asksS resp) (Hashes.ofSha512 h) st sch tr (packed tr st.m) cm e salt nonce).run
      (Transcript.init #[] d)).go l = (.ok (p, t'), l')

/-- **The one-table session's proof files**: the table's schedule, its level-0 commitment `cm` and its extra claims `e`
at the link's points; the first send is the link of `cm`'s cap; one proof file per rep stream, each a `RepFile`. -/
theorem sessionP_files (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) {spec : SessionSpec} {t : String}
    (ht : spec.tables = #[t]) (hct : spec.coinTree = none) (st : Setup) (trs : Array Trace) (tape : Tape)
    {r : String × Array ByteArray} {bodies : Array Claims.Msg}
    (hrun : (sessionP (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) spec #[st] trs tape).go #[] = (.ok r, bodies)) :
    ∃ sch cm e, fast100 st.m = some sch ∧
      commitP (Hashes.ofSha512 (Claims.hOf H) : Hashes V) st.merkle (sch.levels.getD 0 default)
        (packed trs[0]! st.m) (tape.salt 0 0 0) = .ok cm ∧
      st.extra (ptsOf spec (coinOf (resp 0))) = .ok e ∧
      (∃ bs, bodies.toList = linkBody [flat cm.cap] [ByteArray.empty] :: bs) ∧
      r.2.size = spec.repStreams.size ∧ ∀ b ∈ r.2, RepFile resp (Claims.hOf H) st sch trs[0]! cm e b := by
  unfold sessionP at hrun
  obtain ⟨schs, l1, hA, hrun⟩ := go_bind_ok hrun
  have hschs := Ret.go (Ret.mapM_array _ (fun st' sch => fast100 st'.m = some sch) _ fun a _ => by
    cases hf : fast100 a.m with
    | none => exact Ret.throw' _
    | some sch => exact Ret.pure rfl) hA
  have hl1 := Still.go (Still.mapM_array (fun a => by
    split
    · exact Still.pure _
    · exact Still.throw _) _) hA
  subst hl1
  obtain ⟨sch, rfl, hsch⟩ : ∃ sch, schs = #[sch] ∧ fast100 st.m = some sch := by
    obtain ⟨hsz, hf⟩ := hschs
    have h1 : schs.size = 1 := by simpa using hsz
    exact ⟨schs[0], eq_singleton_of h1, by simpa using hf 0 (by simp) (by omega)⟩
  dsimp only at hrun
  obtain ⟨cms, l2, hB, hrun⟩ := go_bind_ok hrun
  have hcms := Ret.go (Ret.mapM_array _ (fun _ cm => commitP (Hashes.ofSha512 (Claims.hOf H) : Hashes V) st.merkle
      (sch.levels.getD 0 default) (packed trs[0]! st.m) (tape.salt 0 0 0) = .ok cm) _
    fun i hi => by
      have : i = 0 := by simpa using hi
      subst this
      exact Ret.of_pureS (commitP_pureS _ _ _ _ _) fun cm e => by simpa using e) hB
  have hl2 := Still.go (Still.mapM_array (fun i => by
    exact Still.of_pureS (commitP_pureS _ _ _ _ _)) _) hB
  subst hl2
  obtain ⟨cm, rfl, hcm⟩ : ∃ cm, cms = #[cm] ∧ commitP (Hashes.ofSha512 (Claims.hOf H) : Hashes V) st.merkle
      (sch.levels.getD 0 default) (packed trs[0]! st.m) (tape.salt 0 0 0) = .ok cm := by
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
          ∀ q ∈ b.2, RepFile resp (Claims.hOf H) st sch trs[0]! cm e q) ∧
        ∃ bs, l.toList = linkBody [flat cm.cap] [ByteArray.empty] :: bs) ?_
    (by exact ⟨rfl, by simp, by simp, [], by simp⟩) hF
  · obtain ⟨h₁, h₂, h₃, bs, hbs⟩ := hinv
    dsimp only at hrun
    rw [recordJson_eq _ hct] at hrun
    obtain ⟨rec, l₅, hJ, hrun⟩ := go_bind_ok hrun
    simp only [go_pure, Prod.mk.injEq, Except.ok.injEq] at hJ hrun
    obtain ⟨rfl, rfl⟩ := hJ
    obtain ⟨rfl, rfl⟩ := hrun
    obtain ⟨e, he⟩ := h₂ (by rw [hrs1]; simp)
    exact ⟨sch, cm, e, hsch, hcm, he, ⟨bs, hbs⟩, h₁, h₃ e he⟩
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
    obtain ⟨⟨pf, tr'⟩, l₂, e2, e⟩ := go_bind_ok e
    obtain ⟨rs, bs', hrs, hl₂, hrec'⟩ := Rec.go (repP_rec resp (Claims.hOf H) _ _ _ _ _ _ _ _) e2
    simp only [go_pure, Prod.mk.injEq, Except.ok.injEq] at e
    obtain ⟨rfl, rfl⟩ := e
    have hex' : st.extra (ptsOf spec (coinOf (resp 0))) = .ok extras := by simpa [ptsOf] using hex
    refine ⟨_, rfl, by simp [h₁], fun _ => ⟨extras, hex'⟩, fun e he q hq => ?_, bs ++ bs', by rw [hl₂, hl]; rfl⟩
    rcases Array.mem_push.1 hq with hq | rfl
    · exact h₃ e he q hq
    · rw [hex'] at he
      cases he
      exact ⟨pf, tape.salt 0 r, tape.nonce 0 r, (spec.domain spec.tables[0]! r).toUTF8, l, tr', l₂, by simp,
        by simpa using e2⟩

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- **Each rep's decoded proof is the honest one**: at the run's one table, with schedule `sch` and level-0
commitment `cm`, the record's `root_B` is `cm`'s cap, two proof files, each under `2^64` bytes, and each rep's file
decodes at the verifier's shape to a proof with `cm`'s cap and the table's PCS parameters. -/
theorem Run.rep (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hP : ProverOk H rel) (o : Run H rel) :
    ∃ (st : Flock.Stmt) (sch : Schedule) (cm : Commit), o.tabs = #[Setup.ofCircuit st] ∧ o.sch 0 = sch ∧
      o.sess.rootB = #[flat cm.cap] ∧
      o.proofs.size = 2 ∧ (∀ b ∈ o.proofs, b.size < 2 ^ 64) ∧ ∀ r < 2,
        decodeProof (shapeOf o.tabs[0]! (o.sch 0) (o.extras 0).size) (o.bytes 0 r) = .ok (o.proof 0 r) ∧
        (o.proof 0 r).cap = cm.cap ∧ (o.proof 0 r).params = pcsParams (Setup.ofCircuit st).merkle sch := by
  obtain ⟨st, rb, streams, hst, ht, htabs, hss, hpf, hex, hlink, -, hrec, hdec, hsess, hlv, hlive⟩ :=
    Run.json S h64 hP o
  obtain ⟨hct, -, -, -⟩ := stmtOk o.toCase
  obtain ⟨hC, -⟩ := hP o.toCase st (by rw [hst]; simp)
  have hrun := o.ran
  unfold sessionS at hrun
  have htabs' : tabsOf #[st] = #[Setup.ofCircuit st] := by simp [tabsOf]
  rw [hst, htabs'] at hrun
  obtain ⟨sch, cm, e, hsch, hcm, he, ⟨bs, hbs⟩, hsz, hfiles⟩ :=
    sessionP_files H o.resp ht hct (Setup.ofCircuit st) _ _ hrun
  rw [show o.resp 0 = (Claims.coinSeq o.r 0).bytes from rfl, ptsOf_coin] at he
  have hcap := hC sch hsch _ _ cm hcm
  have hl : o.link = ([flat cm.cap], [ByteArray.empty]) := by
    unfold Run.link Run.sends
    rw [hbs]
    show (linkOf (0 :: linkBody [flat cm.cap] [ByteArray.empty])).getD ([], []) = _
    rw [linkOf_linkBody (by simp) (by simp) (by simpa using hcap.2) (by simp)]
    rfl
  rw [hlink] at hl
  obtain rfl : rb = flat cm.cap := by simpa using hl
  have hsch0 : o.sch 0 = sch := by simp [Run.sch, schedsOf, htabs, hsch]
  have hext0 : o.extras 0 = e := by
    unfold Run.extras
    rw [htabs, hsess, show (#[Setup.ofCircuit st])[0]! = Setup.ofCircuit st from rfl]
    simp only [sessOf, he]
    rfl
  have hrs := repStreams_one ht
  have hsz2 : o.proofs.size = 2 := by rw [hsz, hrs]; rfl
  have hshaped : ∀ b ∈ o.proofs, ∃ p, b = FlockVBridge.Family.encodeProof (shapeOf (Setup.ofCircuit st) sch e.size) p ∧
      FlockVBridge.Family.ProofShaped (shapeOf (Setup.ofCircuit st) sch e.size) p ∧
      p.cap = cm.cap ∧ p.params = pcsParams (Setup.ofCircuit st).merkle sch := by
    intro b hb
    obtain ⟨p, salt, nonce, d, l, t', l', rfl, hp⟩ := hfiles b hb
    have hR' := repShaped h64 o.toCase st sch cm e hst hsch _ (by rw [hst]; exact hcm) he salt nonce d
    rw [hst] at hR'
    have h1 := Ret.go hR' hp
    have h2 := Ret.go (repP_fields _ _ _ _ _ _ _ _ _ _ _) hp
    exact ⟨p, rfl, h1, h2⟩
  refine ⟨st, sch, cm, htabs, hsch0, by rw [hsess]; rfl, hsz2, fun b hb => ?_, fun r hr => ?_⟩
  · obtain ⟨p, rfl, hps, -⟩ := hshaped b hb
    exact hps.2.2.1
  · have hmem : o.bytes 0 r ∈ o.proofs := by
      unfold Run.bytes
      rw [Nat.mul_zero, Nat.zero_add, getElem!_pos o.proofs r (by omega)]
      exact Array.getElem_mem _
    obtain ⟨p, hb, hps, hc, hpar⟩ := hshaped _ hmem
    have hdec : decodeProof (shapeOf o.tabs[0]! (o.sch 0) (o.extras 0).size) (o.bytes 0 r) = .ok p := by
      rw [hb, hsch0, hext0, show o.tabs[0]! = Setup.ofCircuit st by rw [htabs]; rfl]
      exact FlockVBridge.Family.decodeProof_encodeProof _ _ hps
    have hp : o.proof 0 r = p := by
      unfold Run.proof
      rw [hdec]
      rfl
    rw [hp]
    exact ⟨hdec, hc, hpar⟩

/-- **The rep's proof file decodes** at its shape: the goal `RepDecodes`. -/
theorem repDecodes_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hP : ProverOk H rel) :
    RepDecodes H rel := by
  intro o i hi r hr
  obtain ⟨st, sch, cm, htabs, -, -, -, -, hrep⟩ := Run.rep S h64 hP o
  rw [htabs] at hi
  obtain rfl : i = 0 := by simpa using hi
  exact (hrep r hr).1

/-- **The rep's PCS parameters are fast100's** (R7/F2): the goal `RepParams`. -/
theorem repParams_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hP : ProverOk H rel) :
    RepParams H rel := by
  intro o i hi r hr
  obtain ⟨st, sch, cm, htabs, hsch0, -, -, -, hrep⟩ := Run.rep S h64 hP o
  rw [htabs] at hi
  obtain rfl : i = 0 := by simpa using hi
  rw [(hrep r hr).2.2, hsch0, show o.tabs[0]! = Setup.ofCircuit st by rw [htabs]; rfl]
  exact FlockVBridge.Family.pcs_bne_self _

/-- **The rep's cap is the committed `root_B`** (S16): the goal `CapCommitted`. -/
theorem capCommitted_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hP : ProverOk H rel) :
    CapCommitted H rel := by
  intro o i hi r hr
  obtain ⟨st, sch, cm, htabs, -, hroot, -, -, hrep⟩ := Run.rep S h64 hP o
  rw [htabs] at hi
  obtain rfl : i = 0 := by simpa using hi
  rw [(hrep r hr).2.1, hroot]
  show (flat cm.cap != flat cm.cap) = false
  rw [bne, bytes_beq_self]
  rfl

/-- **The last message's lengths fit their words**: the goal `FinalFits`. -/
theorem finalFits_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hP : ProverOk H rel) :
    FinalFits H rel := by
  intro o
  obtain ⟨st, sch, cm, -, -, -, hsz, hb, -⟩ := Run.rep S h64 hP o
  exact ⟨o.recordFits, by omega, hb⟩

end FlockComplete.JsonRT
