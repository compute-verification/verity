import Proofs.Flock.Claims.Recursion.CallsBridgeE
import Proofs.Flock.Claims.Recursion.CallsAgree

/-!
# The calls half where the proof files agree, at the element-level verifier

`plain_calls_agree` at `nextE`: two accepted element-level decisions on one setup, link, coin and round messages, with
the same proof files, call SHA-512 on one set, where the sessions their records decode have the same rounds at each
rep stream as replay reads them (`Round.erase`). The element-level verdict computes every digest the plain one does
(`verdictK_log`, `decodeK_log`), a round's calls read only its bytes (`streamLog_erase`), and a rep reads its rounds
only up to erasure (`Flock.With.verifyRepWith_erase`, at the logging oracle). With `sess_rounds_eqK`,
`RecDeploy.PlainCallsOfProofsE` holds at SHA-512, and so does `RecDeploy.PlainCallsBindOrSlackE`.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness.Refine

/-! ## The public words, as the element-level decoder reads them -/

/-- `decode_pubs` at `Record.decodeWithK keep`, whatever the hashes. -/
theorem decodeWithK_pubs (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {text : String}
    {sess : RunRecord} (h : Record.decodeWithK keep hs spec text = .ok sess) :
    ∃ j l pubs, Lean.Json.parse text = .ok j ∧ Record.fld j "link" = .ok l ∧ Record.fld l "publics" = .ok pubs ∧
      List.Forall₂ (PubField pubs) spec.tables.toList sess.publics.toList := by
  unfold Record.decodeWithK Record.decodeJWithK at h
  dsimp only at h
  split at h
  rotate_left
  · cases h
  rename_i j hparse
  obtain ⟨j', hj, h⟩ := ebind_ok h
  cases hj
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  all_goals
    cases h
    refine ⟨_, _, _, by assumption, by assumption, by assumption, ?_⟩
    refine List.Forall₂.imp (fun t p h => ?_) (Claims.Bridge.Custody.mapM_forall₂ (by assumption))
    try dsimp only at h
    refine ⟨fun hn => ?_, fun hn => ?_⟩
    · rw [ite_eq_left hn] at h
      obtain ⟨_, hw, h⟩ := ebind_ok h
      simp only [pure, Except.pure, Except.ok.injEq] at hw
      subst hw
      split at h
      · repeat' obtain ⟨_, _, h⟩ := ebind_ok h
        exact (Except.ok.inj h).symm
      · exact (Except.ok.inj h).symm
    · rw [ite_eq_right (by simp [hn])] at h
      obtain ⟨s, hs', h⟩ := ebind_ok h
      obtain ⟨p', hp', h⟩ := ebind_ok h
      refine ⟨s, hs', ?_⟩
      split at h
      · repeat' obtain ⟨_, _, h⟩ := ebind_ok h
        rw [← Except.ok.inj h]
        exact hp'
      · rw [← Except.ok.inj h]
        exact hp'

/-- `decode_publics` at `Record.decodeWithK keep`. -/
theorem decodeWithK_publics (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {text : String}
    {sess : RunRecord} {lv : Live} (hd : Record.decodeWithK keep hs spec text = .ok sess)
    (hl : liveOf spec text = .ok lv) : sess.publics = lv.publics := by
  obtain ⟨j, l, pubs, hp, hl1, hp1, hf⟩ := decodeWithK_pubs keep hs hd
  obtain ⟨j', l', pubs', hp', hl1', hp1', hf'⟩ := liveOf_pubs hl
  rw [hp] at hp'
  cases hp'
  rw [hl1] at hl1'
  cases hl1'
  rw [hp1] at hp1'
  cases hp1'
  exact Array.ext' (Claims.Bridge.Custody.forall₂_unique (R := PubField pubs) (fun _ _ _ => pubField_unique) hf hf')

/-! ## The element-level verdict's calls -/

theorem roundLog_erase (spec : SessionSpec) (rd : Round) : roundLog spec rd.erase = roundLog spec rd := by
  simp only [roundLog, Round.erase_bytes]

/-- A stream's calls read its rounds only as replay does. -/
theorem streamLog_erase {spec : SessionSpec} {s s' : StreamRec}
    (h : s.rounds.map Round.erase = s'.rounds.map Round.erase) : streamLog spec s = streamLog spec s' := by
  have e : ∀ l : List Round, l.flatMap (roundLog spec) = (l.map Round.erase).flatMap (roundLog spec) := by
    intro l
    induction l with
    | nil => rfl
    | cons a l ih => simp only [List.flatMap_cons, List.map_cons, roundLog_erase, ← ih]
  unfold streamLog
  rw [e s.rounds.toList, e s'.rounds.toList, ← Array.toList_map, ← Array.toList_map, h]

/-- `verdict_calls` at `verifyLoggedE`. -/
theorem verdictE_calls {h : ByteArray → ByteArray} {spec : SessionSpec} {tabs : Array Setup} {record : String}
    {proofs : Array ByteArray} {L : Array ByteArray} (hv : verifyLoggedE h spec tabs record proofs = (.ok (), L)) :
    ∃ (sess : RunRecord) (schs : Array Schedule) (ds : List (List ByteArray)),
      Record.decodeWithK (M := Flock.V) (fun _ => true) (Hashes.ofSha512 h) spec record = .ok sess ∧
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
  obtain ⟨sess, schs, l₁, hd, hsch, ht, -, hL⟩ := verdictK_log (fun _ => true) (l₀ := #[]) () L hv
  obtain ⟨ds, hl₁, hds, -, hF, hy, hy0, hss, hf⟩ := decodeK_log (fun _ => true) (l₀ := #[]) sess l₁ hd
  have hdec := (With.decodeWithK_sim (With.Sim.forget _) (With.Sim.forget_logged h) (fun _ => true) spec record) #[]
  rw [hd] at hdec
  refine ⟨sess, schs, ds, hdec.symm, hsch, ht, hds, hF, hy, hy0, hss, hf, ?_⟩
  rw [hL, hl₁]
  simp only [List.nil_append, List.append_assoc]

/-- `verdict_calls_sub` at `verifyLoggedE`, with the rep streams' rounds compared after `Round.erase`. -/
theorem verdictE_calls_sub {spec : SessionSpec} {tabs : Array Setup} {record record' : String}
    {proofs : Array ByteArray} {L L' : Array ByteArray} {sess sess' : RunRecord}
    (hv : verifyLoggedE Sha512.hash spec tabs record proofs = (.ok (), L))
    (hv' : verifyLoggedE Sha512.hash spec tabs record' proofs = (.ok (), L'))
    (hd : Record.decodeWithK (M := Flock.V) (fun _ => true) (Hashes.ofSha512 Sha512.hash) spec record = .ok sess)
    (hd' : Record.decodeWithK (M := Flock.V) (fun _ => true) (Hashes.ofSha512 Sha512.hash) spec record' = .ok sess')
    (hn : spec.tables.toList.Nodup) (hB : sess.rootB = sess'.rootB) (hP : sess.points = sess'.points)
    (hpub : sess.publics = sess'.publics)
    (hR : ∀ q (hq : q < spec.repStreams.size) (y y' : StreamRec),
      sess.streams.find? (·.name == spec.repStreams[q].2.2) = some y →
        sess'.streams.find? (·.name == spec.repStreams[q].2.2) = some y' →
          y.rounds.map Round.erase = y'.rounds.map Round.erase) :
    ∀ b ∈ L.toList, b ∈ L'.toList := by
  obtain ⟨s₁, schs, ds, hd₁, hsch, ht, hds, hF, hy, hy0, hss, hf, hL⟩ := verdictE_calls hv
  obtain ⟨s₂, schs', ds', hd₂, hsch', -, hds', hF', hy', hy0', -, hf', hL'⟩ := verdictE_calls hv'
  obtain rfl := Except.ok.inj (hd₁.symm.trans hd)
  obtain rfl := Except.ok.inj (hd₂.symm.trans hd')
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
    rw [With.verifyRepWith_erase (With.Sim.liftMap_StV _) (Hashes.logged Sha512.hash) spec _ _ _ hy₁ hy₂ hyr hP
      (by rw [hB])]
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
    exact ⟨s', Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hs'), streamLog_erase (hR q hq s s' hsq hs')⟩
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
        obtain ⟨-, -, pubs, -, -, -, hpf⟩ := decodeWithK_pubs _ _ hd'
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

/-! ## The element-level decision's calls -/

/-- `decisionAt_log` at `decisionAtE`. -/
theorem decisionAtE_log (H : Claims.Msg → Claims.Msg) (s : Ins) (spec : SessionSpec) (sts : Array Stmt)
    (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin))
    (f : Final) :
    ((decisionAtE s spec sts x link c₁ mids f).run.eval H).2 =
      ((verifyWithME Claims.Hashes.oc spec (tabsOf sts) f.record f.proofs).run.eval H).2 := by
  simp only [decisionAtE, ExceptT.run_bind, Claims.OC.bind_eq, Claims.OC.eval_bind]
  rcases hv : (verifyWithME Claims.Hashes.oc spec (tabsOf sts) f.record f.proofs).run.eval H with ⟨r, l⟩
  cases r <;> exact List.append_nil l

/-- `next_decide_calls` at `nextE`. -/
theorem nextE_decide_calls (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {as : List Claims.Msg}
    {cs : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray}
    {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (h : act false x as cs = .decide s spec sts link c₁ mids f)
    (hs : ((setupM Claims.Hashes.oc s.read).run.eval H).1 = .ok (spec, sts)) :
    ((nextE x as cs).eval H).2 = ((setupM Claims.Hashes.oc s.read).run.eval H).2 ++
      ((verifyWithME Claims.Hashes.oc spec (tabsOf sts) f.record f.proofs).run.eval H).2 := by
  simp only [nextE, h, outOf_log, decisionE, ExceptT.run_bind, Claims.OC.bind_eq, Claims.OC.eval_bind]
  rcases hv : (setupM Claims.Hashes.oc s.read).run.eval H with ⟨r, l⟩
  rw [hv] at hs
  simp only at hs
  subst hs
  show l ++ ((decisionAtE s spec sts x link c₁ mids f).run.eval H).2 = _
  rw [decisionAtE_log]

/-- `acc_verifyLogged` at `nextE`. -/
theorem accE_verifyLogged {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)}
    {f : Final} (h : act false x as cs = .decide s spec sts link c₁ mids f)
    (hacc : ((nextE x as cs).eval sha512).1 = .acc) :
    verifyLoggedE Sha512.hash spec (tabsOf sts) f.record f.proofs =
        (.ok (), (verifyLoggedE Sha512.hash spec (tabsOf sts) f.record f.proofs).2) ∧
      checks false s spec x link c₁ mids f = .ok () := by
  obtain ⟨spec', sts', he, hv, hc⟩ := (nextE_decide sha512 h).1 hacc
  rw [act_decide_setupWith h] at he
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj he)
  rw [hOf_sha512] at hv
  exact ⟨Prod.ext ((verifyLoggedE_fst _ _ _ _ _ _).trans hv) rfl, hc⟩

/-- `next_calls_toFinset` at `nextE`. -/
theorem nextE_calls_toFinset {x : Claims.Msg} {as as' : List Claims.Msg} {cs cs' : List Coin} {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link link' : List ByteArray × List ByteArray} {c₁ c₁' : Coin}
    {mids mids' : List (Claims.Msg × Coin)} {f f' : Final}
    (h : act false x as cs = .decide s spec sts link c₁ mids f)
    (h' : act false x as' cs' = .decide s spec sts link' c₁' mids' f')
    (hsub : ∀ b ∈ (verifyLoggedE Sha512.hash spec (tabsOf sts) f.record f.proofs).2.toList,
      b ∈ (verifyLoggedE Sha512.hash spec (tabsOf sts) f'.record f'.proofs).2.toList)
    (hsub' : ∀ b ∈ (verifyLoggedE Sha512.hash spec (tabsOf sts) f'.record f'.proofs).2.toList,
      b ∈ (verifyLoggedE Sha512.hash spec (tabsOf sts) f.record f.proofs).2.toList) :
    ((nextE x as cs).eval sha512).2.toFinset = ((nextE x as' cs').eval sha512).2.toFinset := by
  have hs := (setupM_eval sha512 s.read).1.trans (act_decide_setupWith h)
  rw [nextE_decide_calls sha512 h hs, nextE_decide_calls sha512 h' hs, List.toFinset_append, List.toFinset_append]
  congr 1
  have hl := Claims.verifyOCE_log sha512 spec (tabsOf sts) f.record f.proofs
  have hl' := Claims.verifyOCE_log sha512 spec (tabsOf sts) f'.record f'.proofs
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

/-- `plain_decide_shared` at `nextE`. -/
theorem plainE_decide_shared {x : Claims.Msg} {ms : List Claims.Msg} {l l' : Claims.Msg} {cs : List Coin}
    {f f' : Final} (hc : cs.length = ms.length + 1)
    (hacc : ((nextE x (ms ++ [l]) cs).eval sha512).1 = .acc)
    (hacc' : ((nextE x (ms ++ [l']) cs).eval sha512).1 = .acc)
    (hfl : finalOf l = some f) (hfl' : finalOf l' = some f') :
    ∃ s spec sts link c₁ mids, act false x (ms ++ [l]) cs = .decide s spec sts link c₁ mids f ∧
      act false x (ms ++ [l']) cs = .decide s spec sts link c₁ mids f' := by
  obtain ⟨s, spec, sts, link, c₁, mids, f₀, hact, -⟩ := nextE_acc_decide hacc
  obtain ⟨s', spec', sts', link', c₁', mids', f₀', hact', -⟩ := nextE_acc_decide hacc'
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

/-- **Two accepted element-level decisions on one setup, link, coin and round messages, with the same proof files,
call SHA-512 on one set**, where the sessions their records decode have the same rounds at each rep stream as replay
reads them (`plain_calls_agree`'s, at `nextE`). -/
theorem plainE_calls_agree {x : Claims.Msg} {as as' : List Claims.Msg} {cs cs' : List Coin} {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f f' : Final}
    (h : act false x as cs = .decide s spec sts link c₁ mids f)
    (h' : act false x as' cs' = .decide s spec sts link c₁ mids f')
    (hacc : ((nextE x as cs).eval sha512).1 = .acc) (hacc' : ((nextE x as' cs').eval sha512).1 = .acc)
    (hpf : f.proofs = f'.proofs)
    (hR : ∀ sess sess',
      Record.decodeWithK (M := Flock.V) (fun _ => true) (Hashes.ofSha512 Sha512.hash) spec f.record = .ok sess →
      Record.decodeWithK (M := Flock.V) (fun _ => true) (Hashes.ofSha512 Sha512.hash) spec f'.record = .ok sess' →
      ∀ q (hq : q < spec.repStreams.size) (y y' : StreamRec),
        sess.streams.find? (·.name == spec.repStreams[q].2.2) = some y →
          sess'.streams.find? (·.name == spec.repStreams[q].2.2) = some y' →
            y.rounds.map Round.erase = y'.rounds.map Round.erase) :
    ((nextE x as cs).eval sha512).2.toFinset = ((nextE x as' cs').eval sha512).2.toFinset := by
  obtain ⟨hV, hck⟩ := accE_verifyLogged h hacc
  obtain ⟨hV', hck'⟩ := accE_verifyLogged h' hacc'
  rw [hpf] at hV
  obtain ⟨sess, -, -, hd, -⟩ := verdictE_calls hV
  obtain ⟨sess', -, -, hd', -⟩ := verdictE_calls hV'
  obtain ⟨-, lv, hlv, hcu⟩ := Claims.Bridge.Custody.checks_ok hck
  obtain ⟨-, lv', hlv', hcu'⟩ := Claims.Bridge.Custody.checks_ok hck'
  obtain ⟨hrB, hpts, -⟩ := decodeWithK_live _ _ hd hlv
  obtain ⟨hrB', hpts', -⟩ := decodeWithK_live _ _ hd' hlv'
  obtain ⟨hlB, hlU, hlP, -⟩ := Claims.Bridge.Custody.custody_ok hcu
  obtain ⟨hlB', hlU', hlP', -⟩ := Claims.Bridge.Custody.custody_ok hcu'
  have hP : sess.points = sess'.points := by rw [hpts, hpts', hlP, hlP']
  have hB : sess.rootB = sess'.rootB := by
    rw [hrB, hrB']
    exact Array.ext' (hlB.trans hlB'.symm)
  have hU : sess.publics = sess'.publics := by
    rw [decodeWithK_publics _ _ hd hlv, decodeWithK_publics _ _ hd' hlv']
    exact Array.ext' (hlU.trans hlU'.symm)
  have hn := tables_nodup (setupOf_some (act_decide h)).2 (Claims.Bridge.Custody.publicInputs_none_of_checks hck)
  have hR' := hR sess sess' hd hd'
  refine nextE_calls_toFinset h h' ?_ ?_ <;> rw [hpf]
  · exact verdictE_calls_sub hV hV' hd hd' hn hB hP hU hR'
  · exact verdictE_calls_sub hV' hV hd' hd hn hB.symm hP.symm hU.symm
      fun q hq y y' hy hy' => (hR' q hq y' y hy' hy).symm

end FlockDeploy.KS

namespace RecDeploy

open Claims (Msg)

/-- **The calls half at `VE` where the proof files agree, at SHA-512** (`plainCallsOfProofs_sha512`'s). -/
theorem plainCallsOfProofsE_sha512 : PlainCallsOfProofsE FlockDeploy.sha512 := by
  intro x ms l l' cs f f' _ hc _ hacc hacc' hfl hfl' hpf
  obtain ⟨s, spec, sts, link, c₁, mids, hact, hact'⟩ := FlockDeploy.KS.plainE_decide_shared hc hacc hacc' hfl hfl'
  refine FlockDeploy.KS.plainE_calls_agree hact hact' hacc hacc' hpf fun sess sess' hd hd' q hq y y' hy hy' => ?_
  obtain ⟨hV, hck⟩ := FlockDeploy.KS.accE_verifyLogged hact hacc
  obtain ⟨hV', hck'⟩ := FlockDeploy.KS.accE_verifyLogged hact' hacc'
  obtain ⟨-, lv, hlv, hcu⟩ := Claims.Bridge.Custody.checks_ok hck
  obtain ⟨-, lv', hlv', hcu'⟩ := Claims.Bridge.Custody.checks_ok hck'
  have hv : Flock.verifyWithMK (M := Flock.V) (fun _ => true) (Flock.Hashes.ofSha512 Flock.Sha512.hash) spec
      (FlockDeploy.tabsOf sts) f.record f.proofs = .ok () :=
    (Flock.verifyLoggedE_fst _ _ _ _ _ _).symm.trans (congrArg Prod.fst hV)
  have hv' : Flock.verifyWithMK (M := Flock.V) (fun _ => true) (Flock.Hashes.ofSha512 Flock.Sha512.hash) spec
      (FlockDeploy.tabsOf sts) f'.record f'.proofs = .ok () :=
    (Flock.verifyLoggedE_fst _ _ _ _ _ _).symm.trans (congrArg Prod.fst hV')
  obtain ⟨z, z', hz, hz', hzz, -⟩ := FlockDeploy.KS.sess_rounds_eqK _ _ _ _ hv hlv hv' hlv' hcu hcu' hd hd' hq
  rw [hy] at hz
  rw [hy'] at hz'
  cases hz
  cases hz'
  exact hzz

/-- **The calls half at `VE`, at SHA-512.** -/
theorem plainCallsBindOrSlackE_sha512 : PlainCallsBindOrSlackE FlockDeploy.sha512 :=
  plainCallsBindOrSlackE_of_proofs plainCallsOfProofsE_sha512

end RecDeploy
