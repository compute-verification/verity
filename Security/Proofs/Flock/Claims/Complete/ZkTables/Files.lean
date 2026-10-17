import Proofs.Flock.Claims.Complete.ZkTables.Goals
import Proofs.Flock.Claims.Complete.Zk.Json.Files

/-!
# The honest masked run's proof files, at several tables

`Zk/Json/Files.lean` at a run of several tables: what the record's checks ask of the proof files (`FilesJ`) from the
per-table forms of `RepShapedZk` and `RecordFitsZk` (`RepShapedJ`, `RecordFitsJ`).

* `sessionPZk_filesJ`: the masked session's `j`th proof file encodes a masked proof that table `j / 2`'s rep `j % 2`
  returned on its share's trace, its masked level-0 commitment and its extra claims at the link's points, with the
  tape's pads committed (`RepFileJ`); the first send is the link of the commitments' caps. The session's loop is walked
  by index (`forIn_array_idx`).
* `Run.repJ`: at a run of several tables, each table's `root_B` is its commitment's cap, and each rep's file decodes at
  the masked verifier's shape to that rep's proof, of the commitment's cap, the table's PCS parameters, and its pads'
  root (`repPZk_fields`), a 64-byte digest (`innerCommitP_root`).
* `filesJ`: `FilesJ` from `RepShapedJ` and `RecordFitsJ`; so `complete` holds with them for `RecordJ` (`completeJR`).
-/

namespace FlockComplete.ZkTables

open Flock Transcript FlockComplete FlockDeploy JsonRT
open Lean (Json JsonNumber)

/-- A `for` loop over a list from index `k`, with an invariant of the index, the loop's state and the sends, each step
taking the list's element at its index. -/
theorem forIn_list_idx {α β : Type} (L : List α) (f : α → β → PS (ForInStep β))
    (inv : ℕ → β → Array Claims.Msg → Prop)
    (hs : ∀ k (hk : k < L.length) b l r l', inv k b l → (f L[k] b).go l = (.ok r, l') →
      ∃ b', r = .yield b' ∧ inv (k + 1) b' l') :
    ∀ k, k ≤ L.length → ∀ b l b' l', inv k b l → (forIn (L.drop k) b f).go l = (.ok b', l') → inv L.length b' l'
  | k, hk, b, l, b', l', h, e => by
    by_cases hlt : k < L.length
    · rw [List.drop_eq_getElem_cons hlt, List.forIn_cons] at e
      obtain ⟨r, l₁, e1, e2⟩ := go_bind_ok e
      obtain ⟨b₁, rfl, h₁⟩ := hs k hlt b l r l₁ h e1
      exact forIn_list_idx L f inv hs (k + 1) hlt b₁ l₁ b' l' h₁ e2
    · obtain rfl : k = L.length := by omega
      simp only [List.drop_length, List.forIn_nil, go_pure, Prod.mk.injEq, Except.ok.injEq] at e
      obtain ⟨rfl, rfl⟩ := e
      exact h
termination_by k => L.length - k

/-- A `for` loop over an array, with an invariant of the index, the loop's state and the sends, each step taking the
array's element at its index. -/
theorem forIn_array_idx {α β : Type} (xs : Array α) (f : α → β → PS (ForInStep β))
    (inv : ℕ → β → Array Claims.Msg → Prop)
    (hs : ∀ k (hk : k < xs.size) b l r l', inv k b l → (f xs[k] b).go l = (.ok r, l') →
      ∃ b', r = .yield b' ∧ inv (k + 1) b' l')
    {b : β} {l : Array Claims.Msg} {b' : β} {l' : Array Claims.Msg} (h0 : inv 0 b l)
    (e : (forIn xs b f).go l = (.ok b', l')) : inv xs.size b' l' := by
  rw [← Array.forIn_toList] at e
  have := forIn_list_idx xs.toList f inv (fun k hk b l r l' h e => hs k (by simpa using hk) b l r l' h
    (by simpa using e)) 0 (Nat.zero_le _) b l b' l' h0 (by simpa using e)
  simpa using this

/-- Table `i`'s rep `rep`'s masked proof file: at the table's extra claims `e` at the points `pts`, it encodes, at the
rep's shape, a masked proof the rep returned on the table's trace, its packed table, its masked level-0 commitment, and
the tape's pads, committed. -/
def RepFileJ (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (spec : SessionSpec) (tabs : Array Setup)
    (schs : Array (Schedule × ℕ)) (cms : Array Commit) (trs : Array Trace) (tape : TapeZk) (pts : Array (Array F128))
    (i rep : ℕ) (b : ByteArray) : Prop :=
  ∃ e p ic l t' l', tabs[i]!.extra pts = .ok e ∧
    innerCommitP (Hashes.ofSha512 h : Hashes V)
      (padsOf tape i rep (Zk.PadLayout.new tabs[i]!.m tabs[i]!.kLog schs[i]!.1.k0)) (tape.innerSalt i rep) = .ok ic ∧
    b = FlockSoundness.Discharge.ZkView.encZk (shapeZk tabs[i]! schs[i]!.1 e.size) p ∧
    ((repPZk (asksS resp) (Hashes.ofSha512 h) tabs[i]! schs[i]!.1 schs[i]!.2 rep
      (spec.domain spec.tables[i]! rep).toUTF8 trs[i]! (packed trs[i]! tabs[i]!.m) cms[i]! e
      (padsOf tape i rep (Zk.PadLayout.new tabs[i]!.m tabs[i]!.kLog schs[i]!.1.k0)) ic (tape.extra i)
      (tape.lanePad i) (tape.salt i rep) (tape.nonce i rep)).run
      (Transcript.init #[] (spec.domain spec.tables[i]! rep).toUTF8)).go l = (.ok (p, t'), l')

/-- **The masked session's proof files**, at any number of tables: each table's `--zk` schedule and masked level-0
commitment; the first send is the link of the commitments' caps; one proof file per rep stream, the `j`th table
`j / 2`'s rep `j % 2`'s (`RepFileJ`). -/
theorem sessionPZk_filesJ (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) {spec : SessionSpec}
    {ct : CoinTree.Spec} (hct : spec.coinTree = some ct) {own : Json} (hown : Json.parse spec.hello = .ok own)
    {hz : String} (hc : canon (helloJ own) = .ok hz) (tabs : Array Setup) (trs : Array Trace) (tape : TapeZk)
    (hst : ∀ a ∈ spec.repStreams, a.1 < tabs.size) {r : String × Array ByteArray} {bodies : Array Claims.Msg}
    (hrun : (sessionPZk (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) spec tabs trs tape).go #[] = (.ok r, bodies)) :
    ∃ schs cms, (∀ i < tabs.size, Zk.schedule tabs[i]!.m = .ok schs[i]! ∧
        commitPZk (Hashes.ofSha512 (Claims.hOf H) : Hashes V) tabs[i]!.merkle (schs[i]!.1.levels.getD 0 default)
          schs[i]!.2 (packed trs[i]! tabs[i]!.m) (tape.extra i) (tape.lanePad i) (tape.salt i 0 0) = .ok cms[i]!) ∧
      cms.size = tabs.size ∧
      (∃ bs, bodies.toList = linkBody (cms.map fun c => flat c.cap).toList (tabs.map fun _ => ByteArray.empty).toList ::
        bs) ∧
      r.2.size = spec.repStreams.size ∧ ∀ j (hj : j < r.2.size),
        RepFileJ resp (Claims.hOf H) spec tabs schs cms trs tape (ptsOf spec (coinOf (resp 0))) (j / 2) (j % 2)
          r.2[j] := by
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
  obtain ⟨hcsz, hcf⟩ := Ret.go (Ret.mapM_array _ (fun i cm => commitPZk (Hashes.ofSha512 (Claims.hOf H) : Hashes V)
      tabs[i]!.merkle (schs[i]!.1.levels.getD 0 default) schs[i]!.2
      ((Array.range tabs.size).map fun i => packed trs[i]! tabs[i]!.m)[i]! (tape.extra i) (tape.lanePad i)
      (tape.salt i 0 0) = .ok cm) _
    fun i hi => Ret.of_pureS (commitPZk_pureS _ _ _ _ _ _ _ _) fun cm e => e) hB
  have hl2 := Still.go (Still.mapM_array (fun i => by
    exact Still.of_pureS (commitPZk_pureS _ _ _ _ _ _ _ _)) _) hB
  subst hl2
  rw [Array.size_range] at hcsz
  have htb : ∀ i < tabs.size, ((Array.range tabs.size).map fun i => packed trs[i]! tabs[i]!.m)[i]! =
      packed trs[i]! tabs[i]!.m := fun i hi => by
    rw [getElem!_pos ((Array.range tabs.size).map fun i => packed trs[i]! tabs[i]!.m) i (by simpa using hi)]
    simp
  have hcm : ∀ i < tabs.size, commitPZk (Hashes.ofSha512 (Claims.hOf H) : Hashes V) tabs[i]!.merkle
      (schs[i]!.1.levels.getD 0 default) schs[i]!.2 (packed trs[i]! tabs[i]!.m) (tape.extra i) (tape.lanePad i)
      (tape.salt i 0 0) = .ok cms[i]! := by
    intro i hi
    have := hcf i (by simpa using hi) (by omega)
    simp only [Array.getElem_range] at this
    rw [htb i hi] at this
    rw [getElem!_pos cms i (by omega)]
    exact this
  obtain ⟨c₁, l₃, hL, hrun⟩ := go_bind_ok hrun
  rw [link_go] at hL
  simp only [Prod.mk.injEq, Except.ok.injEq, Array.size_empty] at hL
  obtain ⟨rfl, rfl⟩ := hL
  obtain ⟨⟨ss, ps⟩, l₄, hF, hrun⟩ := go_bind_ok hrun
  have hinv := forIn_array_idx _ _ (fun k (b : Array (Array Round) × Array ByteArray) (l : Array Claims.Msg) =>
      b.2.size = k ∧ (∀ j (hj : j < b.2.size), RepFileJ resp (Claims.hOf H) spec tabs schs cms trs tape
        (ptsOf spec (coinOf (resp 0))) (j / 2) (j % 2) b.2[j]) ∧
      ∃ bs, l.toList = linkBody (cms.map fun c => flat c.cap).toList (tabs.map fun _ => ByteArray.empty).toList :: bs)
    ?_ (by exact ⟨rfl, by simp, [], by simp⟩) hF
  · obtain ⟨h₁, h₂, bs, hbs⟩ := hinv
    dsimp only at hrun
    rw [recordJson_eqZk _ hct hown hc] at hrun
    obtain ⟨rec, l₅, hJ, hrun⟩ := go_bind_ok hrun
    simp only [go_pure, Prod.mk.injEq, Except.ok.injEq] at hJ hrun
    obtain ⟨rfl, rfl⟩ := hJ
    obtain ⟨rfl, rfl⟩ := hrun
    exact ⟨schs, cms, fun i hi => ⟨hv i hi, hcm i hi⟩, by omega, ⟨bs, hbs⟩, h₁, h₂⟩
  · rintro k hk ⟨ss, ps⟩ l res l' ⟨h₁, h₂, bs, hl⟩ e
    have hi : k / 2 < tabs.size := by
      have := hst _ (Array.getElem_mem hk)
      rwa [repStreams_getElem _ hk] at this
    rw [repStreams_getElem _ hk] at e
    obtain ⟨extras, l₁, e1, e⟩ := go_bind_ok e
    rw [monadLift_go] at e1
    simp only [Prod.mk.injEq] at e1
    obtain ⟨hex, rfl⟩ := e1
    obtain ⟨ic, l₀, e0, e3⟩ := go_bind_ok e
    have hic := Ret.go (Ret.of_pureS (innerCommitP_pureS _ _ _) (P := fun ic => innerCommitP
      (Hashes.ofSha512 (Claims.hOf H) : Hashes V)
      (padsOf tape (k / 2) (k % 2) (Zk.PadLayout.new tabs[k / 2]!.m tabs[k / 2]!.kLog schs[k / 2]!.1.k0))
      (tape.innerSalt (k / 2) (k % 2)) = .ok ic) fun _ e => by simpa using e) e0
    rw [Still.go (Still.of_pureS (innerCommitP_pureS _ _ _)) e0] at e3
    obtain ⟨⟨pf, tr'⟩, l₂, e2, e⟩ := go_bind_ok e3
    simp only [go_pure, Prod.mk.injEq, Except.ok.injEq] at e
    obtain ⟨rfl, rfl⟩ := e
    have hex' : tabs[k / 2]!.extra (ptsOf spec (coinOf (resp 0))) = .ok extras := by simpa [ptsOf] using hex
    obtain ⟨-, bs', -, hl₂, -⟩ := Rec.go (repPZk_rec resp (Claims.hOf H) _ _ _ _ _ _ _ _ _ _ _ _ _ _ _) e2
    dsimp only at h₁ h₂
    subst h₁
    refine ⟨_, rfl, by simp, fun j hj => ?_, bs ++ bs', by rw [hl₂, hl]; rfl⟩
    by_cases hjk : j < ps.size
    · rw [Array.getElem_push_lt hjk]
      exact h₂ j hjk
    · obtain rfl : j = ps.size := by simp only [Array.size_push] at hj; omega
      rw [Array.getElem_push_eq]
      rw [htb _ hi] at e2
      exact ⟨extras, pf, ic, _, tr', l₂, hex', hic, rfl, e2⟩

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

section Several

variable (H rel)

/-- **The honest masked rep proof fits its shape** (`RepShapedZk`), at each table of every good case of several tables:
on table `i`'s `--zk` schedule `v`, its masked level-0 commitment `cm` of its share's packed table, its extra claims `e`
at the link's points and its pads' commitment `ic`, each masked proof a rep returns (`repRunJ`) is `Shaped` at the shape
the masked verifier decodes it at. -/
def RepShapedJ : Prop :=
  ∀ c : Case H rel, 1 < c.sts.size → ∀ i < c.sts.size, ∀ v cm e,
    Zk.schedule (Setup.ofCircuit c.sts[i]!).m = .ok v →
    commitPZk (Hashes.ofSha512 (Claims.hOf H) : Hashes V) (Setup.ofCircuit c.sts[i]!).merkle
      (v.1.levels.getD 0 default) v.2 (packed (traces c.sts c.w)[i]! (Setup.ofCircuit c.sts[i]!).m) (c.ω.extra i)
      (c.ω.lanePad i) (c.ω.salt i 0 0) = .ok cm →
    (Setup.ofCircuit c.sts[i]!).extra (pointsOf c.spec (Claims.coinSeq c.r 0)) = .ok e →
    ∀ rep ic, innerCommitP (Hashes.ofSha512 (Claims.hOf H) : Hashes V)
      (padsOf c.ω i rep (Zk.PadLayout.new (Setup.ofCircuit c.sts[i]!).m (Setup.ofCircuit c.sts[i]!).kLog v.1.k0))
      (c.ω.innerSalt i rep) = .ok ic →
    Ret (repRunJ H rel c i v cm e rep ic) fun pt =>
      FlockSoundness.Discharge.ZkView.Shaped (shapeZk (Setup.ofCircuit c.sts[i]!) v.1 e.size) pt.1

/-- **The printed honest masked record is under `2^64` bytes** (`RecordFitsZk`), at every run of several tables. -/
def RecordFitsJ : Prop := ∀ o : Run H rel, 1 < o.sts.size → o.record.toUTF8.size < 2 ^ 64

end Several

/-- **Each rep's decoded masked proof is the honest one**, at a run of several tables: two proof files per table, each
under `2^64` bytes, and each rep's file decodes at the masked verifier's shape to a proof of the table's PCS
parameters, with the table's `root_B` as its cap and a 64-byte pads' root. -/
theorem Run.repJ (S : JsonSpec) (h64 : Out64 H) (hS : StmtOkJ H rel) (hR : RepShapedJ H rel) (o : Run H rel)
    (hc : 1 < o.sts.size) :
    o.proofs.size = 2 * o.tabs.size ∧ (∀ b ∈ o.proofs, b.size < 2 ^ 64) ∧ ∀ i < o.tabs.size, ∀ r < 2,
      Zk.decode (shapeZk o.tabs[i]! (o.sch i) (o.extras i).size) (o.bytes i r) = .ok (o.zp i r) ∧
      (o.zp i r).proof.params = pcsParams o.tabs[i]!.merkle (o.sch i) ∧
      flat (o.zp i r).proof.cap = o.sess.rootB[i]! ∧ (o.zp i r).inner.root.size = 64 := by
  obtain ⟨-, -, -, m, kvs, hown, -, -, -, hg⟩ := hS o.toCase hc
  obtain ⟨hz, hcz⟩ := S.canon_ok _ hg
  obtain ⟨ct, hct⟩ : ∃ ct, o.spec.coinTree = some ct := Option.isSome_iff_exists.1 o.ready_parts.2.2.2.2
  obtain ⟨-, hmax, hsz, -⟩ := o.toCase.tables
  unfold Tags.MAX_TABLES at hmax
  have hP := proverOkJ (rel := rel) h64 o.toCase
  have htsz : (tabsOf o.sts).size = o.sts.size := tabsOf_size _
  have htab : ∀ i (hi : i < o.sts.size), (tabsOf o.sts)[i]! = Setup.ofCircuit o.sts[i]! := fun i hi => by
    rw [getElem!_pos (tabsOf o.sts) i (by rw [htsz]; exact hi), getElem!_pos o.sts i hi]
    simp [tabsOf]
  have hst : ∀ a ∈ o.spec.repStreams, a.1 < (tabsOf o.sts).size := by
    intro a ha
    obtain ⟨k, hk, rfl⟩ := Array.mem_iff_getElem.1 ha
    have hk' := hk
    rw [repStreams_size] at hk'
    rw [repStreams_getElem _ hk, tabsOf_size]
    dsimp only
    omega
  have hrun := o.ran
  unfold sessionS at hrun
  obtain ⟨schs, cms, hsc, hcsz, ⟨bs, hbs⟩, hpsz, hfiles⟩ :=
    sessionPZk_filesJ H _ hct hown hcz (tabsOf o.sts) _ _ hst hrun
  dsimp only at hpsz hfiles
  rw [htsz] at hcsz
  obtain ⟨rbs, streams, -, -, -, -, -, hlink, -, -, hsess, -, -, -⟩ := Run.jsonJ S h64 hS o hc
  have hsc' : ∀ i (hi : i < o.sts.size), Zk.schedule (Setup.ofCircuit o.sts[i]!).m = .ok schs[i]! ∧
      commitPZk (Hashes.ofSha512 (Claims.hOf H) : Hashes V) (Setup.ofCircuit o.sts[i]!).merkle
        (schs[i]!.1.levels.getD 0 default) schs[i]!.2 (packed (traces o.sts o.w)[i]! (Setup.ofCircuit o.sts[i]!).m)
        (o.ω.extra i) (o.ω.lanePad i) (o.ω.salt i 0 0) = .ok cms[i]! := fun i hi => by
    rw [← htab i hi]
    exact hsc i (by rw [htsz]; exact hi)
  have hcapk : ∀ k (hk : k < o.sts.size), (flat cms[k]!.cap).size < 2 ^ 64 := by
    intro k hk
    obtain ⟨hv, hcm⟩ := hsc' k hk
    rw [getElem!_pos o.sts k hk] at hv hcm
    exact ((hP _ (Array.getElem_mem hk)).1 _ hv _ _ _ _ _ hcm).2
  have hcap : ∀ b ∈ (cms.map fun c => flat c.cap).toList, b.size < 2 ^ 64 := by
    intro b hb
    obtain ⟨k, hk, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 hb)
    rw [Array.size_map] at hk
    rw [Array.getElem_map, ← getElem!_pos cms k hk]
    exact hcapk k (by omega)
  have hl : o.link = ((cms.map fun c => flat c.cap).toList, (o.tabs.map fun _ => ByteArray.empty).toList) := by
    unfold Run.link Run.sends
    rw [hbs]
    show (linkOf (0 :: linkBody _ _)).getD ([], []) = _
    rw [linkOf_linkBody (by simp; omega) (by simp [htsz]; omega) hcap (by simp)]
    rfl
  rw [hlink, Prod.mk.injEq] at hl
  have hrbs : rbs = cms.map fun c => flat c.cap := Array.toList_inj.1 hl.1
  have hroot : ∀ i (hi : i < o.sts.size), o.sess.rootB[i]! = flat cms[i]!.cap := fun i hi => by
    rw [hsess]
    show rbs[i]! = _
    rw [hrbs, getElem!_pos _ i (by simp; omega), Array.getElem_map, getElem!_pos cms i (by omega)]
  have hpts : o.sess.points = pointsOf o.spec (Claims.coinSeq o.r 0) := by rw [hsess]; rfl
  have hsch : ∀ i (hi : i < o.sts.size), o.sch i = schs[i]!.1 := fun i hi => by
    unfold Run.sch zkSchedsOf Run.tabs
    rw [getElem!_pos _ i (by simp [htsz]; exact hi), Array.getElem_map,
      ← getElem!_pos (tabsOf o.sts) i (by rw [htsz]; exact hi), htab i hi]
    unfold schedZkOf
    rw [(hsc' i hi).1]
  have hfile : ∀ j (hj : j < o.proofs.size), ∃ e p,
      (Setup.ofCircuit o.sts[j / 2]!).extra (pointsOf o.spec (Claims.coinSeq o.r 0)) = .ok e ∧
      o.proofs[j] = FlockSoundness.Discharge.ZkView.encZk (shapeZk (Setup.ofCircuit o.sts[j / 2]!) schs[j / 2]!.1
        e.size) p ∧
      FlockSoundness.Discharge.ZkView.Shaped (shapeZk (Setup.ofCircuit o.sts[j / 2]!) schs[j / 2]!.1 e.size) p ∧
      p.proof.cap = cms[j / 2]!.cap ∧ p.proof.params = pcsParams (Setup.ofCircuit o.sts[j / 2]!).merkle schs[j / 2]!.1 ∧
      p.inner.root.size = 64 := by
    intro j hj
    have hi : j / 2 < o.sts.size := by
      rw [hpsz, repStreams_size, hsz] at hj
      omega
    obtain ⟨e, p, ic, l, t', l', hex, hic, hb, hp⟩ := hfiles j hj
    rw [htab _ hi] at hex hic hb hp
    simp only [ptsOf_coin] at hex
    obtain ⟨hv, hcm⟩ := hsc' _ hi
    have h1 := Ret.go (hR o.toCase hc (j / 2) hi _ _ e hv hcm hex (j % 2) ic hic) hp
    have h2 := Ret.go (repPZk_fields _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _) hp
    exact ⟨e, p, hex, hb, h1, h2.1, h2.2.1, by rw [h2.2.2]; exact innerCommitP_root (hOf_size h64) _ _ hic⟩
  have hpsz2 : o.proofs.size = 2 * o.tabs.size := by
    rw [hpsz, repStreams_size, hsz]
    exact congrArg (2 * ·) htsz.symm
  refine ⟨hpsz2, fun b hb => ?_, fun i hi r hr => ?_⟩
  · obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 hb
    obtain ⟨e, p, -, hb, hps, -⟩ := hfile j hj
    rw [hb]
    exact hps.2.2.1
  · have hi' : i < o.sts.size := by rw [← htsz]; exact hi
    have hj : 2 * i + r < o.proofs.size := by omega
    obtain ⟨e, p, hex, hb, hps, hcp, hpar, hrt⟩ := hfile (2 * i + r) hj
    have hji : (2 * i + r) / 2 = i := by omega
    rw [hji] at hex hb hps hcp hpar
    have htabi : o.tabs[i]! = Setup.ofCircuit o.sts[i]! := htab i hi'
    have hext : o.extras i = e := by
      unfold Run.extras
      rw [hpts, htabi, hex]
      rfl
    have hdec : Zk.decode (shapeZk o.tabs[i]! (o.sch i) (o.extras i).size) (o.bytes i r) = .ok p := by
      unfold Run.bytes
      rw [getElem!_pos o.proofs _ hj, hb, htabi, hsch i hi', hext]
      exact FlockSoundness.Discharge.ZkView.decode_encode _ _ hps
    have hzp : o.zp i r = p := by
      unfold Run.zp
      rw [hdec]
      rfl
    rw [hzp, htabi, hsch i hi', hroot i hi']
    exact ⟨by rw [← htabi, ← hsch i hi', hdec], hpar, by rw [hcp], hrt⟩

/-- **What the record's checks ask of the proof files**, at every run of several tables, from `RepShapedJ` and
`RecordFitsJ`. -/
theorem filesJ (S : JsonSpec) (h64 : Out64 H) (hS : StmtOkJ H rel) (hR : RepShapedJ H rel)
    (hF : RecordFitsJ H rel) : FilesJ H rel := by
  intro o hc
  obtain ⟨hsz, hb, hrep⟩ := Run.repJ S h64 hS hR o hc
  obtain ⟨-, hmax, -, -⟩ := o.toCase.tables
  unfold Tags.MAX_TABLES at hmax
  have ht : o.tabs.size = o.sts.size := tabsOf_size _
  refine ⟨⟨hF o hc, by omega, hb⟩, fun i hi r hr => (hrep i hi r hr).1, fun i hi r hr => ?_,
    fun i hi r hr => ?_, fun i hi r hr => ?_⟩
  · rw [(hrep i hi r hr).2.1]
    exact FlockVBridge.Family.pcs_bne_self _
  · rw [(hrep i hi r hr).2.2.2]
    rfl
  · rw [(hrep i hi r hr).2.2.1, bne, bytes_beq_self]
    rfl

/-- **Flock's masked verifier is complete** for the honest masked prover on each table's share, with error `δc H`, at
any number of tables: `completeJ` with what the record's checks ask of the proof files from `RepShapedJ` and
`RecordFitsJ` (`filesJ`). -/
theorem completeJR (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (S : JsonRT.JsonSpec) (h64 : JsonRT.Out64 H) (hS : JsonRT.StmtOkZk H rel) (hR : JsonRT.RepShapedZk H rel)
    (hF : JsonRT.RecordFitsZk H rel) (hPR : JsonRT.ProverRunsZk H rel)
    (hRp : FlockComplete.Replay H rel) (hRS : FlockComplete.RingSwitchZk H rel)
    (hLg : FlockComplete.LigeritoZk H rel) (hLC : FlockComplete.LaneConstraints H rel)
    (hIn : FlockComplete.Inner H rel) (hCo : FlockComplete.ConsumedZk H rel) (hSJ : StmtOkJ H rel)
    (hPRJ : ProverRunsJ H rel) (hRJ : RepShapedJ H rel) (hFJ : RecordFitsJ H rel) (hOp : OpenJ H rel) :
    Claims.Flock.Complete H (deployZk rel) honestP (δc H) :=
  completeJ H rel S h64 hS hR hF hPR hRp hRS hLg hLC hIn hCo hSJ hPRJ (filesJ S h64 hSJ hRJ hFJ) hOp

end FlockComplete.ZkTables
