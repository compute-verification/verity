import Proofs.Flock.Claims.Recursion.LastFilesR
import Proofs.Flock.Claims.Recursion.CallsAgreeE

/-!
# The calls half of the deployed reader's last-message binding (`RecDeploy.RCallsBind`)

A decision of `nextR rd` calls SHA-512 in the statement's setup, in the digests of the proofs' files, and in the plain
verdict at the record's value `jAt sha512` and those files (`nextR_calls`, `decisionAtS_log`); the checks after it call
nothing. Two accepted decisions on the same rounds and coins share the setup, the first coin and the round messages
(`nextR_shared`). On different files two openings collide (`filesR_tied`). On the same files the setup's and the
digests' calls are one, and the verdicts' calls are one set (`verdictJ_calls_sub`, `plainE_calls_agree`'s at a parsed
value and `keep := id`): the verdict calls the decoder's digests (`decodeJK_log`), S17's digests of the files and the
reps' own calls (`verdictJK_log`), and custody pins every input they read (`sess_rounds_eqJK`, `decodeJWithK_publics`).

So `RCallsBind` holds at SHA-512 with no premise (`RecDeploy.rCallsBind_sha512`), and with the columns half the deployed
bound record at the deployed reader meets `LastBindsEC` and `LastCoinFree` (`RecDeploy.lastR_sha512`).
-/

namespace FlockDeploy.KS

open Flock FlockSoundness.Refine

local instance lawfulBEqByteArrayLC : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := by show (a.data == a.data) = true; exact beq_self_eq_true _

/-! ## The public words, at a parsed value -/

section Publics

open Lean (Json)
open Claims.Bridge.Custody

/-- `liveOf_pubs` at a parsed value. -/
theorem liveOfJ_pubs {spec : SessionSpec} {j : Json} {lv : Live} (h : FlockDeploy.liveOfJ spec j = .ok lv) :
    ∃ l pubs, Record.fld j "link" = .ok l ∧ Record.fld l "publics" = .ok pubs ∧
      List.Forall₂ (PubField pubs) spec.tables.toList lv.publics.toList := by
  unfold FlockDeploy.liveOfJ at h
  obtain ⟨l, hl, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨pubs, hpubs, h⟩ := ebind_ok h
  obtain ⟨publics, hpublics, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  cases h
  exact ⟨l, pubs, hl, hpubs, List.Forall₂.imp (fun t p h => pubField_of h) (mapM_forall₂ hpublics)⟩

/-- `decodeWithK_pubs` at a parsed value. -/
theorem decodeJWithK_pubs (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {j : Json}
    {sess : Session} (h : Record.decodeJWithK keep hs spec j = .ok sess) :
    ∃ l pubs, Record.fld j "link" = .ok l ∧ Record.fld l "publics" = .ok pubs ∧
      List.Forall₂ (PubField pubs) spec.tables.toList sess.publics.toList := by
  unfold Record.decodeJWithK at h
  dsimp only at h
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  all_goals
    cases h
    refine ⟨_, _, by assumption, by assumption, ?_⟩
    refine List.Forall₂.imp (fun t p h => ?_) (mapM_forall₂ (by assumption))
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

/-- `decodeWithK_publics` at a parsed value. -/
theorem decodeJWithK_publics (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {j : Json}
    {sess : Session} {lv : Live} (hd : Record.decodeJWithK keep hs spec j = .ok sess)
    (hl : FlockDeploy.liveOfJ spec j = .ok lv) : sess.publics = lv.publics := by
  obtain ⟨l, pubs, hl1, hp1, hf⟩ := decodeJWithK_pubs keep hs hd
  obtain ⟨l', pubs', hl1', hp1', hf'⟩ := liveOfJ_pubs hl
  rw [hl1] at hl1'
  cases hl1'
  rw [hp1] at hp1'
  cases hp1'
  exact Array.ext' (forall₂_unique (R := PubField pubs) (fun _ _ _ => pubField_unique) hf hf')

end Publics

/-! ## The plain verdict's calls, at a parsed value -/

open Lean (Json) in
/-- `decodeK_log` at a parsed value. -/
theorem decodeJK_log (keep : Bool → Bool) {h : ByteArray → ByteArray} {spec : SessionSpec} {j : Json}
    {l₀ : Array ByteArray} :
    RetL (Record.decodeJWithK keep (Hashes.logged h) spec j) l₀ fun sess l' =>
      ∃ ds : List (List ByteArray), l'.toList = l₀.toList ++ (sess.rootB.toList.flatMap (dgl spec) ++
        dgl spec sess.rootF ++ ds.flatten ++ sess.streams.toList.flatMap (streamLog spec) ++
        spec.repStreams.toList.flatMap (fun x => dgl spec sess.rootB[x.1]!) ++ dgl spec (lbOf spec sess)) ∧
      List.Forall₂ (fun p d => d = dgl spec p ∨ d = [] ∧ p.size = 0) sess.publics.toList ds ∧
      sess.rootB.size = spec.tables.size ∧ sess.rootF = spec.sigma ∧
      sess.y.size = spec.points * spec.tables.size ∧ (∀ c ∈ sess.y, c.isZero = true) ∧
      sess.streams.size = spec.repStreams.size ∧
      ∀ x ∈ spec.repStreams.toList,
        ∃ s, sess.streams.find? (·.name == x.2.2) = some s ∧ 0 < s.rounds.size := by
  unfold Record.decodeJWithK
  retl_walk [Record.check]
  all_goals
    subst_vars
    obtain ⟨y5, e5, f5, g5⟩ := mapM_log (R := fun _ rb d => True ∧ d = dgl spec rb) (fun t l => by
      retl_walk []
      exact ⟨_, by simp [dgL_toList], trivial, rfl⟩) h_rootB
    obtain ⟨y7, e7, f7, g7⟩ := mapM_log (R := fun _ p d => d = dgl spec p ∨ d = [] ∧ p.size = 0) (fun t l => by
      retl_walk []
      all_goals first
        | (refine ⟨_, ?_, .inl rfl⟩; simp [dgL_toList]; done)
        | exact ⟨[], by simp, .inr ⟨rfl, size_zero_of_not hc⟩⟩) h_publics
    obtain ⟨y12, e12, f12, g12⟩ := mapM_log (R := fun _ s d => True ∧ d = streamLog spec s) (fun s l => by
      retl_walk []
      obtain ⟨yr, er, fr, gr⟩ := mapIdxM_log (R := fun _ rd d => True ∧ d = roundLog spec rd) (fun k r l => by
        retl_walk []
        all_goals exact ⟨_, by simp [dgL_toList, roundLog], trivial, rfl⟩) h_rounds
      refine ⟨_, ?_, trivial, rfl⟩
      rw [gr, flatten_snd (R := fun _ _ => True) (F := roundLog spec) fr, ← er]; rfl) h_streams
    have h13 := ‹StV.go (forIn spec.repStreams _ _) _ = _›
    obtain ⟨d13, f13, g13⟩ := forIn_log (R := fun x d => d = dgl spec rootB[x.1]! ∧
        ∃ s, streams.find? (·.name == x.2.2) = some s ∧ 0 < s.rounds.size) (fun x _ l => by
      obtain ⟨i, r, name⟩ := x
      dsimp only
      retl_walk [Record.check]
      have hr := check_ok_iff.1 ‹Record.check (_ && decide (_ > 0)) _ = _›
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hr
      exact ⟨rfl, _, by simp [dgL_toList], rfl, _, by assumption, hr.2⟩) h13
    have h14a := ‹StV.go (forIn ((spec.tables.zip rootB).qsort _) _ _) _ = _›
    obtain ⟨rfl, rfl⟩ := go_forIn_pure_inv (f := fun (x : String × ByteArray) b => b ++ x.2) h14a
    have h14b := ‹StV.go (forIn pts _ _) _ = _›
    simp only [Array.forIn_pure_yield_eq_foldl, pure_bind] at h14b
    obtain ⟨rfl, rfl⟩ := go_pure_inv h14b
    have h14c := ‹StV.go (forIn y _ _) _ = _›
    obtain ⟨rfl, rfl⟩ := go_forIn_pure_inv (f := fun c b => F128.pushBytes b c) h14c
    obtain ⟨yp, -, fp, gp⟩ := mapM_log (R := fun _ _ d => True ∧ d = []) (fun p l => by
      retl_walk []
      exact ⟨[], by simp, trivial, rfl⟩) h_proofSha
    have h10 := check_ok_iff.1 ‹Record.check (y.size == _ && _) _ = _›
    have h11 := check_ok_iff.1 ‹Record.check (streams.size == _ && _) _ = _›
    simp only [Bool.and_eq_true, beq_iff_eq, Array.all_eq_true_iff_forall_mem] at h10 h11
    refine ⟨y7.map Prod.snd, ?_, ?_, ?_, beq_iff_eq.1 (check_ok_iff.1 ‹Record.check (_ == spec.sigma) _ = _›),
      h10.1, h10.2, h11.1, fun x hx => (forall₂_exists_right f13 x hx).elim fun _ h => h.2⟩
    · rw [gp, flatten_snd (R := fun _ _ => True) (F := fun _ => []) fp]
      simp only [Array.toList_append, dgL_toList, g13, g12, g7, g5, flatMap_nil', List.append_nil,
        flatten_snd (R := fun _ _ => True) (F := dgl spec) f5, flatten_snd (R := fun _ _ => True)
          (F := streamLog spec) f12, ← e5, ← e12, flatten_eq_flatMap (fun _ _ h => h.1) f13, List.append_assoc]
      rfl
    · rw [e7]
      exact forall₂_fst_snd f7
    · show rootB.size = _
      rw [← Array.length_toList, e5, List.length_map, ← f5.length_eq, Array.length_toList]

/-- `verdictK_log` at a parsed value. -/
theorem verdictJK_log (keep : Bool → Bool) {h : ByteArray → ByteArray} {spec : SessionSpec} {tabs : Array Setup}
    {j : Lean.Json} {proofs : Array ByteArray} {l₀ : Array ByteArray} :
    RetL (verifyJWithMK keep (Hashes.logged h) spec tabs j proofs) l₀ fun _ l' =>
      ∃ (sess : Session) (schs : Array Schedule) (l₁ : Array ByteArray),
        (Record.decodeJWithK keep (Hashes.logged h) spec j).go l₀ = (.ok sess, l₁) ∧
        List.Forall₂ (fun st sch => fast100 st.m = some sch) tabs.toList schs.toList ∧
        tabs.size = spec.tables.size ∧ proofs.size = spec.repStreams.size ∧
        l'.toList = l₁.toList ++ (spec.repStreams.zip proofs).toList.flatMap (fun x => dgl spec x.2) ++
          (List.range tabs.size).flatMap fun i =>
            repLog h spec tabs[i]! schs[i]! sess i 0 proofs[2 * i]! ++
              repLog h spec tabs[i]! schs[i]! sess i 1 proofs[2 * i + 1]! := by
  unfold verifyJWithMK
  retl_walk []
  all_goals
    subst_vars
    have hp := ‹StV.go (forIn (tabs.zip sess.publics) _ _) _ = _›
    have h17 := ‹StV.go (forIn (spec.repStreams.zip proofs) _ _) _ = _›
    have hr := ‹StV.go (forIn [:tabs.size] _ _) _ = _›
    obtain ⟨dp, fp, gp⟩ := forIn_log (R := fun _ d => d = []) (fun x _ l => by
      obtain ⟨st, p⟩ := x
      dsimp only
      retl_walk []
      exact ⟨rfl, [], by simp, rfl⟩) hp
    obtain ⟨ys, es, fs, gs⟩ := mapM_log (R := fun st sch d => fast100 st.m = some sch ∧ d = []) (fun st l => by
      retl_walk []
      all_goals exact ⟨[], by simp, by assumption, rfl⟩) h_schs
    obtain ⟨d17, f17, g17⟩ := forIn_log (R := fun x d => d = dgl spec x.2) (fun x _ l => by
      obtain ⟨⟨_, _, w⟩, b⟩ := x
      dsimp only
      retl_walk []
      all_goals exact ⟨rfl, _, by simp [dgL_toList], rfl⟩) h17
    rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hr
    simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one, ← List.range_eq_range'] at hr
    obtain ⟨dr, fr, gr⟩ := list_forIn_log (R := fun i d =>
        d = repLog h spec tabs[i]! schs[i]! sess i 0 proofs[2 * i]! ++
          repLog h spec tabs[i]! schs[i]! sess i 1 proofs[2 * i + 1]!) _ (fun i _ l => by
      retl_walk []
      all_goals
        have e0 := (Prod.mk.inj ((rep_shift ..).symm.trans h_c0)).2
        have e1 := (Prod.mk.inj ((rep_shift ..).symm.trans h_c1)).2
        refine ⟨rfl, _, ?_, rfl⟩
        rw [← e1, ← e0]
        simp [repLog]) _ _ _ hr
    have hc1 := ‹¬(tabs.size != spec.tables.size || tabs.size == 0) = true›
    have hc2 := ‹¬(proofs.size != spec.repStreams.size) = true›
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, beq_iff_eq, not_or, Decidable.not_not] at hc1 hc2
    refine ⟨sess, schs, l_sess, h_sess, ?_, hc1.1, hc2, ?_⟩
    · rw [es]
      exact List.forall₂_map_right_iff.2 (fs.imp fun _ _ h => h.1)
    · rw [gr, g17, gs, gp, flatten_eq_flatMap (fun _ _ h => h) fp, flatMap_nil',
        flatten_snd (R := fun (x : Setup) y => fast100 x.m = some y) (F := fun _ => []) fs, flatMap_nil',
        flatten_eq_flatMap (fun _ _ h => h) f17, flatten_eq_flatMap (fun _ _ h => h) fr]
      simp only [List.append_nil, List.append_assoc]

/-- `verdictE_calls` at a parsed value and `keep := id`. -/
theorem verdictJ_calls {h : ByteArray → ByteArray} {spec : SessionSpec} {tabs : Array Setup} {j : Lean.Json}
    {proofs : Array ByteArray} {L : Array ByteArray}
    (hv : (verifyJWithMK id (Hashes.logged h) spec tabs j proofs).go #[] = (.ok (), L)) :
    ∃ (sess : Session) (schs : Array Schedule) (ds : List (List ByteArray)),
      Record.decodeJWithK (M := Flock.V) id (Hashes.ofSha512 h) spec j = .ok sess ∧
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
  obtain ⟨sess, schs, l₁, hd, hsch, ht, -, hL⟩ := verdictJK_log id (l₀ := #[]) () L hv
  obtain ⟨ds, hl₁, hds, -, hF, hy, hy0, hss, hf⟩ := decodeJK_log id (l₀ := #[]) sess l₁ hd
  have hdec := (With.decodeJWithK_sim (With.Sim.forget _) (With.Sim.forget_logged h) id spec j) #[]
  rw [hd] at hdec
  refine ⟨sess, schs, ds, hdec.symm, hsch, ht, hds, hF, hy, hy0, hss, hf, ?_⟩
  rw [hL, hl₁]
  simp only [List.nil_append, List.append_assoc]

/-- `verdictE_calls_sub` at parsed values and `keep := id`. -/
theorem verdictJ_calls_sub {spec : SessionSpec} {tabs : Array Setup} {j j' : Lean.Json}
    {proofs : Array ByteArray} {L L' : Array ByteArray} {sess sess' : Session}
    (hv : (verifyJWithMK id (Hashes.logged Sha512.hash) spec tabs j proofs).go #[] = (.ok (), L))
    (hv' : (verifyJWithMK id (Hashes.logged Sha512.hash) spec tabs j' proofs).go #[] = (.ok (), L'))
    (hd : Record.decodeJWithK (M := Flock.V) id (Hashes.ofSha512 Sha512.hash) spec j = .ok sess)
    (hd' : Record.decodeJWithK (M := Flock.V) id (Hashes.ofSha512 Sha512.hash) spec j' = .ok sess')
    (hn : spec.tables.toList.Nodup) (hB : sess.rootB = sess'.rootB) (hP : sess.points = sess'.points)
    (hpub : sess.publics = sess'.publics)
    (hR : ∀ q (hq : q < spec.repStreams.size) (y y' : StreamRec),
      sess.streams.find? (·.name == spec.repStreams[q].2.2) = some y →
        sess'.streams.find? (·.name == spec.repStreams[q].2.2) = some y' →
          y.rounds.map Round.erase = y'.rounds.map Round.erase) :
    ∀ b ∈ L.toList, b ∈ L'.toList := by
  obtain ⟨s₁, schs, ds, hd₁, hsch, ht, hds, hF, hy, hy0, hss, hf, hL⟩ := verdictJ_calls hv
  obtain ⟨s₂, schs', ds', hd₂, hsch', -, hds', hF', hy', hy0', -, hf', hL'⟩ := verdictJ_calls hv'
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
        obtain ⟨-, pubs, -, -, hpf⟩ := decodeJWithK_pubs _ _ hd'
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

/-- **The plain verdict's oracle log at a parsed value is its logged run's**, and the two runs agree. -/
theorem verifyJ_log (H : Claims.Msg → Claims.Msg) (spec : SessionSpec) (tabs : Array Setup) (j : Lean.Json)
    (proofs : Array ByteArray) :
    ((verifyJWithM Claims.Hashes.oc spec tabs j proofs).run.eval H).2.map (fun p => Claims.ofMsg p.1) =
        ((verifyJWithMK id (Hashes.logged (Claims.hOf H)) spec tabs j proofs).go #[]).2.toList ∧
      ((verifyJWithMK id (Hashes.logged (Claims.hOf H)) spec tabs j proofs).go #[]).1 =
        ((verifyJWithM Claims.Hashes.oc spec tabs j proofs).run.eval H).1 := by
  have h := With.verifyJWithMK_sim (Claims.ocLog H) (Claims.ocLog_hashes H) id spec tabs j proofs #[]
  rw [verifyJWithM_eq_verifyJWithMK]
  refine ⟨?_, h.1⟩
  rw [h.2]
  simp

/-- An accepting plain verdict at a parsed value, over `OC` at SHA-512, is an accepting logged run. -/
theorem verifyJ_go {spec : SessionSpec} {tabs : Array Setup} {j : Lean.Json} {proofs : Array ByteArray}
    (h : ((verifyJWithM Claims.Hashes.oc spec tabs j proofs).run.eval sha512).1 = .ok ()) :
    (verifyJWithMK id (Hashes.logged Sha512.hash) spec tabs j proofs).go #[] =
      (.ok (), ((verifyJWithMK id (Hashes.logged Sha512.hash) spec tabs j proofs).go #[]).2) := by
  have e := (verifyJ_log sha512 spec tabs j proofs).2
  rw [hOf_sha512] at e
  exact Prod.ext (e.trans h) rfl

/-- `nextE_calls_toFinset`'s verdict part at parsed values: two plain verdicts over `OC` at SHA-512, each of whose
logged runs' inputs is one of the other's, log one set. -/
theorem verdictJ_toFinset {spec : SessionSpec} {tabs : Array Setup} {j j' : Lean.Json} {proofs : Array ByteArray}
    (hsub : ∀ b ∈ ((verifyJWithMK id (Hashes.logged Sha512.hash) spec tabs j proofs).go #[]).2.toList,
      b ∈ ((verifyJWithMK id (Hashes.logged Sha512.hash) spec tabs j' proofs).go #[]).2.toList)
    (hsub' : ∀ b ∈ ((verifyJWithMK id (Hashes.logged Sha512.hash) spec tabs j' proofs).go #[]).2.toList,
      b ∈ ((verifyJWithMK id (Hashes.logged Sha512.hash) spec tabs j proofs).go #[]).2.toList) :
    ((verifyJWithM Claims.Hashes.oc spec tabs j proofs).run.eval sha512).2.toFinset =
      ((verifyJWithM Claims.Hashes.oc spec tabs j' proofs).run.eval sha512).2.toFinset := by
  have hl := (verifyJ_log sha512 spec tabs j proofs).1
  have hl' := (verifyJ_log sha512 spec tabs j' proofs).1
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

/-! ## A decision's calls -/

/-- `decisionAtE_log` at `decisionAtJ`. -/
theorem decisionAtJ_log (H : Claims.Msg → Claims.Msg) (s : Ins) (spec : SessionSpec) (sts : Array Stmt)
    (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin))
    (f : FinalJ) :
    ((decisionAtJ s spec sts x link c₁ mids f).run.eval H).2 =
      ((verifyJWithM Claims.Hashes.oc spec (tabsOf sts) (coinSubJ spec c₁ mids f.j) f.proofs).run.eval H).2 := by
  simp only [decisionAtJ, ExceptT.run_bind, Claims.OC.bind_eq, Claims.OC.eval_bind]
  rcases hv : (verifyJWithM Claims.Hashes.oc spec (tabsOf sts) (coinSubJ spec c₁ mids f.j) f.proofs).run.eval H
    with ⟨r, l⟩
  cases r <;> exact List.append_nil l

/-- **`decisionAtS` calls the files' digests, then the verdict at `jAt`.** -/
theorem decisionAtS_log (H : Claims.Msg → Claims.Msg) (s : Ins) (spec : SessionSpec) (sts : Array Stmt)
    (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin))
    (f : FinalP) :
    ((decisionAtS s spec sts x link c₁ mids f).run.eval H).2 =
      (((filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).mapM (spec.dgWith Claims.Hashes.oc)).run.eval H).2 ++
        ((verifyJWithM Claims.Hashes.oc spec (tabsOf sts) (jAt H spec sts c₁ mids f)
          (filesAt (tabsOf sts) (pointsOf spec c₁) f.ps)).run.eval H).2 := by
  simp only [decisionAtS, ExceptT.run_bind, Claims.OC.bind_eq, Claims.OC.eval_bind]
  rcases hv : (((filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).mapM (spec.dgWith Claims.Hashes.oc)).run.eval H)
    with ⟨r, l⟩
  have hr : r = .ok ((filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).map
      (FlockComplete.dgOf spec (Claims.hOf H))) := by
    have e : EvOk H _ _ := evOk_mapM_array (evOk_dg H spec) (filesAt (tabsOf sts) (pointsOf spec c₁) f.ps)
    unfold EvOk at e
    rw [hv] at e
    exact e
  subst hr
  show l ++ ((decisionAtP s spec sts x link c₁ mids ⟨f.stmt, shaSubJ spec ((filesAt (tabsOf sts)
    (pointsOf spec c₁) f.ps).map (FlockComplete.dgOf spec (Claims.hOf H))) f.j, f.ps⟩).run.eval H).2 = _
  unfold decisionAtP
  rw [decisionAtJ_log]
  rfl

/-- **A decision of `nextR` calls the setup's digests, then `decisionAtS`'s** (`nextE_decide_calls` at `nextR`). -/
theorem nextR_calls (H : Claims.Msg → Claims.Msg) {rd : Claims.Msg → Option FinalP} {x : Claims.Msg}
    {as : List Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : FinalP}
    (h : actR rd x as cs = .decide s spec sts link c₁ mids f)
    (hs : ((setupM Claims.Hashes.oc s.read).run.eval H).1 = .ok (spec, sts)) :
    ((nextR rd x as cs).eval H).2 = ((setupM Claims.Hashes.oc s.read).run.eval H).2 ++
      ((decisionAtS s spec sts x link c₁ mids f).run.eval H).2 := by
  simp only [nextR, h, outOf_log, decisionSF, ExceptT.run_bind, Claims.OC.bind_eq, Claims.OC.eval_bind]
  rcases hv : (setupM Claims.Hashes.oc s.read).run.eval H with ⟨r, l⟩
  rw [hv] at hs
  simp only at hs
  subst hs
  rfl

/-! ## The same files make one set of calls -/

/-- **Two accepted decisions of `nextR` at one setup, first coin and round messages, on the same proof files, call
SHA-512 on one set**: the setup's and the files' digests are one, and the verdicts' calls are one set
(`plainE_calls_agree`'s at `nextR`). -/
theorem callsR_of_files {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as as' : List Claims.Msg}
    {cs cs' : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f f' : FinalP}
    (ha : actR rd x as cs = .decide s spec sts link c₁ mids f)
    (ha' : actR rd x as' cs' = .decide s spec sts link c₁ mids f')
    (hacc : ((nextR rd x as cs).eval sha512).1 = .acc) (hacc' : ((nextR rd x as' cs').eval sha512).1 = .acc)
    (hF : filesAt (tabsOf sts) (pointsOf spec c₁) f.ps = filesAt (tabsOf sts) (pointsOf spec c₁) f'.ps) :
    ((nextR rd x as cs).eval sha512).2.toFinset = ((nextR rd x as' cs').eval sha512).2.toFinset := by
  obtain ⟨hv, hck, hsu, -, hpi⟩ := Claims.Bridge.PlainDraw.decideR_sha512 ha hacc
  obtain ⟨hv', hck', -, -, -⟩ := Claims.Bridge.PlainDraw.decideR_sha512 ha' hacc'
  have hr := (decideR_runs ha hacc).1
  have hr' := (decideR_runs ha' hacc').1
  have hs : ((setupM Claims.Hashes.oc s.read).run.eval sha512).1 = .ok (spec, sts) := by
    rw [(setupM_eval sha512 s.read).1, hOf_sha512, setupWith_sha512, Ins.setup_read hsu]
  rw [← hF] at hv' hr'
  generalize hF₀ : filesAt (tabsOf sts) (pointsOf spec c₁) f.ps = F at hv hv' hr hr'
  generalize hJ : jAt sha512 spec sts c₁ mids f = J at hv hck hr
  generalize hJ' : jAt sha512 spec sts c₁ mids f' = J' at hv' hck' hr'
  obtain ⟨-, lv, hlv, hcu⟩ := Claims.Bridge.Custody.checksAtJ_ok hck
  obtain ⟨-, lv', hlv', hcu'⟩ := Claims.Bridge.Custody.checksAtJ_ok hck'
  have hg := verifyJ_go hr
  have hg' := verifyJ_go hr'
  obtain ⟨sess, -, -, hd, -⟩ := verdictJ_calls hg
  obtain ⟨sess', -, -, hd', -⟩ := verdictJ_calls hg'
  obtain ⟨hrB, hpts, -⟩ := decodeJWithK_live _ _ hd hlv
  obtain ⟨hrB', hpts', -⟩ := decodeJWithK_live _ _ hd' hlv'
  obtain ⟨hlB, hlU, hlP, -⟩ := Claims.Bridge.Custody.custody_ok hcu
  obtain ⟨hlB', hlU', hlP', -⟩ := Claims.Bridge.Custody.custody_ok hcu'
  have hP : sess.points = sess'.points := by rw [hpts, hpts', hlP, hlP']
  have hB : sess.rootB = sess'.rootB := by
    rw [hrB, hrB']
    exact Array.ext' (hlB.trans hlB'.symm)
  have hU : sess.publics = sess'.publics := by
    rw [decodeJWithK_publics _ _ hd hlv, decodeJWithK_publics _ _ hd' hlv']
    exact Array.ext' (hlU.trans hlU'.symm)
  have hn := tables_nodup hsu hpi
  have hR : ∀ q (hq : q < spec.repStreams.size) (y y' : StreamRec),
      sess.streams.find? (·.name == spec.repStreams[q].2.2) = some y →
        sess'.streams.find? (·.name == spec.repStreams[q].2.2) = some y' →
          y.rounds.map Round.erase = y'.rounds.map Round.erase := by
    intro q hq y y' hy hy'
    obtain ⟨z, z', hz, hz', hzz, -⟩ := sess_rounds_eqJK _ _ _ _ hv hlv hv' hlv' hcu hcu' hd hd' hq
    rw [hy] at hz
    rw [hy'] at hz'
    cases hz
    cases hz'
    exact hzz
  have hVF : ((verifyJWithM Claims.Hashes.oc spec (tabsOf sts) J F).run.eval sha512).2.toFinset =
      ((verifyJWithM Claims.Hashes.oc spec (tabsOf sts) J' F).run.eval sha512).2.toFinset :=
    verdictJ_toFinset (verdictJ_calls_sub hg hg' hd hd' hn hB hP hU hR)
      (verdictJ_calls_sub hg' hg hd' hd hn hB.symm hP.symm hU.symm
        fun q hq y y' hy hy' => (hR q hq y' y hy' hy).symm)
  rw [nextR_calls sha512 ha hs, nextR_calls sha512 ha' hs, decisionAtS_log, decisionAtS_log, ← hF, hF₀, hJ, hJ']
  simp only [List.toFinset_append]
  rw [hVF]

end FlockDeploy.KS

namespace RecDeploy

open FlockDeploy (Coin FinalP nextR actR)

/-- **The deployed reader binds its last message's calls, at SHA-512**: two accepted last steps on the same rounds and
coins share the setup, the first coin and the round messages (`nextR_shared`); on the same proof files they call
SHA-512 on one set (`callsR_of_files`), and on different ones two openings collide among their calls
(`filesR_tied`). -/
theorem rCallsBind_sha512 (rd : Claims.Msg → Option FinalP) : RCallsBind FlockDeploy.sha512 rd := by
  intro x ms l l' cs hk _ _ hacc hacc' hne
  obtain ⟨s, spec, sts, link, c₁, mids, f, ha⟩ := FlockDeploy.KS.actR_of_acc hacc
  obtain ⟨s', spec', sts', link', c₁', mids', f', ha'⟩ := FlockDeploy.KS.actR_of_acc hacc'
  obtain ⟨e₁, e₂, e₃, e₄, e₅, e₆⟩ :=
    FlockDeploy.KS.nextR_shared (FlockDeploy.KS.ms_ne_nil_of_actR hk ha) ha ha'
  subst s' spec' sts' link' c₁' mids'
  by_cases hF : FlockDeploy.filesAt (FlockDeploy.tabsOf sts) (FlockDeploy.pointsOf spec c₁) f.ps =
      FlockDeploy.filesAt (FlockDeploy.tabsOf sts) (FlockDeploy.pointsOf spec c₁) f'.ps
  · exact absurd (FlockDeploy.KS.callsR_of_files ha ha' hacc hacc' hF) hne
  · obtain ⟨u, v, hcol, hu, hv⟩ := FlockDeploy.KS.filesR_tied ha ha' hacc hacc' hF
    exact ⟨u, hu, v, hv, hcol⟩

/-- **What `StepsBase` reads of the last step at the deployed reader, at SHA-512, with no premise**: `LastBindsEC` and
`LastCoinFree` at `bdR`, whatever the code side's join, `final` and weights. -/
theorem lastR_sha512 (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (rd : Claims.Msg → Option FinalP) (wE : Claims.Msg → ℝ) (join : List Claims.Msg → Claims.Msg)
    (final : Claims.Msg → Option (List (List (Claims.Msg × Salt)) × Claims.Msg)) (w : Claims.Msg → ℝ) :
    (FlockDeploy.KS.bdR rel rd wE join rowInDeployed final w).LastBindsEC FlockDeploy.sha512 ∧
      (FlockDeploy.KS.bdR rel rd wE join rowInDeployed final w).LastCoinFree FlockDeploy.sha512 :=
  lastR_of rel rd wE join final w (rColsBind_sha512 rd) (rCallsBind_sha512 rd)

end RecDeploy
