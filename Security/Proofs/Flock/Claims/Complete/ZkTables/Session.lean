import Proofs.Flock.Claims.Complete.Zk.Json.Session

/-!
# What the honest masked session returns and sends, at any number of tables

`Zk/Json/Session.lean`'s `sessionPZk_facts` at every number of tables (`sessionPZk_factsJ`): with the keyed coin tree,
a run of the pure masked session from no sends returns the printed honest masked record (`printJ` of `recordObjZk`)
of one non-empty `root_B` under `2^64` bytes per table, the link's points off the first reply's coin, and one
well-formed stream per rep stream; and its sends are the link of those `root_B`s and no public words, then the
streams' rounds in order (`SessionFactsJ`). Each table's masked commitment and reps are asked for what `ProverOkZk`
asks of one table's.
-/

namespace FlockComplete.ZkTables

open Flock Transcript FlockComplete FlockDeploy JsonRT
open Lean (Json JsonNumber)

/-- **What the masked session returns and sends**, at any number of tables: its record is the honest masked record of
one non-empty `root_B` per table, the link's points off the first reply's coin, and one well-formed stream per rep
stream; and its sends are the link of those `root_B`s and no public words, then the streams' rounds in order, each from
index `1` on (`Recs`). -/
def SessionFactsJ (h : ByteArray → ByteArray) (resp : ℕ → Claims.Msg) (spec : SessionSpec) (tabs : Array Setup)
    (hz : String) (ct : CoinTree.Spec) (r : String × Array ByteArray) (bodies : Array Claims.Msg) : Prop :=
  ∃ rbs streams, rbs.size = tabs.size ∧
    r.1 = (printJ (recordObjZk spec h hz ct rbs (ptsOf spec (coinOf (resp 0))) streams r.2)) ∧
    (∀ rb ∈ rbs, 0 < rb.size ∧ rb.size < 2 ^ 64) ∧ streams.size = spec.repStreams.size ∧
    (∀ rs ∈ streams, RoundsOk rs) ∧
    ∃ bs, bodies.toList = linkBody rbs.toList (tabs.map fun _ => ByteArray.empty).toList :: bs ∧
      Recs resp 1 (streams.toList.flatMap Array.toList) bs

/-- **The masked session's facts** at any number of tables, given each table's non-empty masked level-0 cap under
`2^64` bytes and well-formed rep rounds on its extra claims at the link's points, every rep stream naming a table. -/
theorem sessionPZk_factsJ (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) {spec : SessionSpec}
    {ct : CoinTree.Spec} (hct : spec.coinTree = some ct) {own : Json} (hown : Flock.JsonParse.parse spec.hello = .ok own)
    {hz : String} (hc : canon (helloJ own) = .ok hz) (tabs : Array Setup) (words : Array (List ℕ))
    (trs : Array Trace) (tape : TapeZk) (hC : ∀ st ∈ tabs, ∀ v, Zk.schedule st.m = .ok v → ∀ g extra lanePad salt cm,
      commitPZk (Hashes.ofSha512 (Claims.hOf H) : Hashes V) st.merkle (v.1.levels.getD 0 default) v.2 g extra lanePad
        salt = .ok cm → 0 < (flat cm.cap).size ∧ (flat cm.cap).size < 2 ^ 64)
    (hR : ∀ st ∈ tabs, ∀ v, Zk.schedule st.m = .ok v →
      ∀ rep dom tr tb cm₀ extras pads ic extra lanePad salt nonce prior ws,
      st.extra (ptsOf spec (coinOf (resp 0))) = .ok extras →
      Ret ((repPZk (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) st v.1 v.2 rep dom tr tb cm₀ extras pads ic extra
        lanePad salt nonce prior ws).run (Transcript.init #[] dom)) fun pt => RepOk pt.2.rounds)
    (hst : ∀ a ∈ spec.repStreams, a.1 < tabs.size)
    {r : String × Array ByteArray} {bodies : Array Claims.Msg}
    (hrun : (sessionPZk (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) spec tabs words trs tape).go #[] =
      (.ok r, bodies)) :
    SessionFactsJ (Claims.hOf H) resp spec tabs hz ct r bodies := by
  unfold sessionPZk at hrun
  obtain ⟨schs, l1, hA, hrun⟩ := go_bind_ok hrun
  obtain ⟨hssz, hsf⟩ := Ret.go (Ret.mapM_array _ (fun st' v => Zk.schedule st'.m = .ok v) _ fun a _ =>
    Ret.monadLift (Zk.schedule a.m)) hA
  have hl1 := Still.go (Still.mapM_array (fun a l => by rw [monadLift_go]) _) hA
  subst hl1
  have hv : ∀ i (hi : i < tabs.size), Zk.schedule tabs[i]!.m = .ok schs[i]! := fun i hi => by
    rw [getElem!_pos tabs i hi, getElem!_pos schs i (by omega)]
    exact hsf i hi (by omega)
  dsimp only at hrun
  obtain ⟨cms, l2, hB, hrun⟩ := go_bind_ok hrun
  obtain ⟨hcsz, hcf⟩ := Ret.go (Ret.mapM_array _ (fun _ cm => 0 < (flat cm.cap).size ∧ (flat cm.cap).size < 2 ^ 64) _
    fun i hi => by
      have hi' : i < tabs.size := by simpa using hi
      exact Ret.of_pureS (commitPZk_pureS _ _ _ _ _ _ _ _) fun cm e =>
        hC tabs[i]! (by rw [getElem!_pos tabs i hi']; exact Array.getElem_mem hi') _ (hv i hi') _ _ _ _ cm e) hB
  have hl2 := Still.go (Still.mapM_array (fun i => by
    exact Still.of_pureS (commitPZk_pureS _ _ _ _ _ _ _ _)) _) hB
  subst hl2
  have hcms : ∀ cm ∈ cms, 0 < (flat cm.cap).size ∧ (flat cm.cap).size < 2 ^ 64 := by
    intro cm hm
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hm
    exact hcf i (by rw [← hcsz]; exact hi) hi
  obtain ⟨c₁, l₃, hL, hrun⟩ := go_bind_ok hrun
  rw [link_go] at hL
  simp only [Prod.mk.injEq, Except.ok.injEq, Array.size_empty] at hL
  obtain ⟨rfl, rfl⟩ := hL
  obtain ⟨⟨ss, ps⟩, l₄, hF, hrun⟩ := go_bind_ok hrun
  have hinv := forIn_array_inv _ _ (fun k (b : Array (Array Round) × Array ByteArray) (l : Array Claims.Msg) =>
      b.1.size = k ∧ b.2.size = k ∧ (∀ rs ∈ b.1, RoundsOk rs) ∧
        ∃ bs, l.toList = linkBody (cms.map fun c => flat c.cap).toList (tabs.map fun _ => ByteArray.empty).toList ::
          bs ∧ Recs resp 1 (b.1.toList.flatMap Array.toList) bs) ?_
    (by exact ⟨rfl, rfl, by simp, [], by simp, trivial⟩) hF
  · obtain ⟨h₁, h₂, h₃, bs, hbs, hrec⟩ := hinv
    dsimp only at hrun
    rw [recordJson_eqZk _ hct hown hc] at hrun
    obtain ⟨rec, l₅, hJ, hrun⟩ := go_bind_ok hrun
    simp only [go_pure, Prod.mk.injEq, Except.ok.injEq] at hJ hrun
    obtain ⟨rfl, rfl⟩ := hJ
    obtain ⟨rfl, rfl⟩ := hrun
    refine ⟨cms.map fun c => flat c.cap, ss, by simp [hcsz], by dsimp only; rfl, fun rb hrb => ?_, h₁, h₃, bs, ?_,
      hrec⟩
    · obtain ⟨cm, hcm, rfl⟩ := Array.mem_map.1 hrb
      exact hcms cm hcm
    · exact hbs
  · rintro k ⟨i, r, name⟩ ⟨ss, ps⟩ l res l' hmem ⟨h₁, h₂, h₃, bs, hl, hrec⟩ e
    have hi : i < tabs.size := hst _ hmem
    obtain ⟨extras, l₁, e1, e⟩ := go_bind_ok e
    rw [monadLift_go] at e1
    simp only [Prod.mk.injEq] at e1
    obtain ⟨hex, rfl⟩ := e1
    obtain ⟨ic, l₀, e0, e3⟩ := go_bind_ok e
    rw [Still.go (Still.of_pureS (innerCommitP_pureS _ _ _)) e0] at e3
    obtain ⟨⟨pf, tr'⟩, l₂, e2, e⟩ := go_bind_ok e3
    have hex' : tabs[i]!.extra (ptsOf spec (coinOf (resp 0))) = .ok extras := by simpa [ptsOf] using hex
    have hok := Ret.go (hR tabs[i]! (by rw [getElem!_pos tabs i hi]; exact Array.getElem_mem hi) _ (hv i hi)
      _ _ _ _ _ _ _ _ _ _ _ _ _ _ hex') e2
    obtain ⟨rs, bs', hrs, hl₂, hrec'⟩ := Rec.go (repPZk_rec resp (Claims.hOf H) _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _) e2
    simp only [go_pure, Prod.mk.injEq, Except.ok.injEq] at e
    obtain ⟨rfl, rfl⟩ := e
    have hsz : l.size = 1 + (ss.toList.flatMap Array.toList).length := by
      rw [← Array.length_toList, hl, Recs.length_eq hrec]
      simp only [List.length_cons]
      omega
    refine ⟨_, rfl, by simp [h₁], by simp [h₂], fun q hq => ?_, bs ++ bs', by rw [hl₂, hl]; rfl, ?_⟩
    · rcases Array.mem_push.1 hq with hq | rfl
      · exact h₃ q hq
      · have hrs' : tr'.rounds.toList = rs := by simpa [Transcript.init] using hrs
        refine ⟨hok.1, fun x hx => ?_⟩
        have := Recs.mem hrec' x (by rw [← hrs']; exact Array.mem_toList_iff.2 hx)
        exact ⟨this.1, hok.2 x hx, this.2⟩
    · simp only [Transcript.init, List.nil_append] at hrs
      simp only [Array.toList_push, List.flatMap_append, List.flatMap_cons, List.flatMap_nil, List.append_nil, hrs]
      exact Recs.append hrec (by rw [← hsz]; exact hrec')

end FlockComplete.ZkTables
