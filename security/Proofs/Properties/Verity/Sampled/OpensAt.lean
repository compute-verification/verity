module

public import Proofs.Properties.Verity.Sampled.Instance
public import Proofs.Properties.Verity.Sampled.OpensRecorded
public import Proofs.Properties.Verity.Sampled.Place
public import Proofs.Properties.Verity.Sampled.SessionAt
public import Proofs.Flock.Bridge.V5.Accepted

/-! The deployed commitment's openings are `Hash.RowLeaf`'s. -/

@[expose] public section

namespace Verity.Sampled

open Properties.Certifier.Bytes (bs bs_length)

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

theorem sha512_hashOf : Flock.Bridge.Spec.HashOf Verity.sha512 := fun x => by
  have hx : (⟨(bs x).toArray⟩ : ByteArray) = x := Claims.Sampling.bytes_bs x
  show (fun i : Fin 64 => (Flock.Sha512.hash ⟨(bs x).toArray⟩).data.getD i.val 0) = _
  rw [hx]
  rfl

theorem sha512Msg_length (m : Claims.Msg) : (Claims.Sampling.sha512Msg m).length = 64 := by
  rw [Claims.Sampling.sha512Msg, bs_length, Flock.Sha512.hash_size]

theorem rowMsg_digest (p : Flock.Port) (bits : List Bool) :
    Verity.sha512 (Flock.Bridge.rowMsg p bits) =
      (Flock.Bridge.Spec.formatOf (Flock.Registered.portSchema p)).digest Verity.sha512 bits := by
  unfold Flock.Bridge.rowMsg Flock.Registered.portSchema
  by_cases hs : p.segs.isSome = true
  · rw [ite_eq_left hs, ite_eq_left hs, Flock.Bridge.Spec.formatOf_seg, Hash.RowLeaf.Format.digest,
      Hash.RowSeg.digest, Flock.Bridge.Spec.segEnc_eq sha512_hashOf]
  · by_cases hv : p.v2 = true
    · rw [ite_eq_right hs, ite_eq_right hs, ite_eq_left hv, ite_eq_left hv, Flock.Bridge.Spec.formatOf_v2,
        Hash.RowLeaf.Format.digest, Hash.RowV2.digest, Flock.Bridge.Spec.enc_eq]
    · rw [ite_eq_right hs, ite_eq_right hs, ite_eq_right hv, ite_eq_right hv, Flock.Bridge.Spec.formatOf_v1,
        Hash.RowLeaf.Format.digest, Hash.RowV1.digest, Flock.Bridge.Spec.encV1_eq]

theorem portSchema_cases (p : Flock.Port) :
    Flock.Registered.portSchema p = Flock.HmRow.ROW_SCHEMA_SEG ∨
      Flock.Registered.portSchema p = Flock.HmRow.ROW_SCHEMA_V2 ∨
      Flock.Registered.portSchema p = Flock.HmRow.ROW_SCHEMA := by
  unfold Flock.Registered.portSchema
  by_cases hs : p.segs.isSome = true
  · exact Or.inl (ite_eq_left hs)
  · by_cases hv : p.v2 = true
    · exact Or.inr (Or.inl ((ite_eq_right hs).trans (ite_eq_left hv)))
    · exact Or.inr (Or.inr ((ite_eq_right hs).trans (ite_eq_right hv)))

theorem fixes_words_of_checksStmt {tags : Flock.Tags} {cf pf : ByteArray} {tables : List (String × ByteArray)}
    {D : Flock.Desc} {pi : Flock.HmIn.Ports} {pub : Flock.Public} {n : ℕ} (hty : tags.typed = false)
    (hh : tags.hiddenOutputs = true) (hD : Flock.Desc.parseStmt tags cf tables = .ok D)
    (hpub : Flock.HmIn.loadPublic tags D.unroll pi pf = .ok pub) (hn : pub.n = n)
    (hck : Flock.PortRoots.checkOf (Flock.ProvedScope.circuitOf tags cf tables) tags pf = .ok ()) :
    Flock.Bridge.RootFixesDomainL D.unroll (Flock.Bridge.LeafMap.ofPublic D.unroll pub n) ∧
      Flock.PortRoots.wordsOk D.unroll = true := by
  obtain ⟨cs, hcs, he⟩ := FlockSoundness.commit_of_parseStmt hty (Flock.Desc.parseStmt_stmt hD)
  have hC := FlockSoundness.Discharge.Hidden.committed_of_commit hcs
  rw [Flock.ProvedScope.circuitOf_hidden hh, hcs] at hck
  rw [he, FlockSoundness.Discharge.Hidden.loadPublicIn_commit hty hC] at hpub
  obtain ⟨pubO, hO⟩ := Flock.PortRoots.checkOf_load hck
  have hw : Flock.PortRoots.wordsOk cs = true :=
    (Flock.PortRoots.loadedWhy_none (Flock.PortRoots.checkOf_loaded hck hO).2.2).2.1
  have hf := (Flock.Bridge.v5Files_of_checks hpub hn hck).2
  obtain ⟨nets, hu⟩ := hC.unrolled_eq hty
  rw [he, hu]
  exact ⟨hf, hw⟩

theorem w_ofAudit {K : Verity.Circuit} {C : Verity.Boolean.Circuit} (h : K = Verity.Circuit.ofAudit C)
    (g : Fin K.N) : K.w g = 1 := by
  subst h
  rfl

namespace Commit

open FlockSoundness FlockSoundness.Discharge.Placed

variable (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (n : ℕ) (pi : Flock.HmIn.Ports)
  (pub : Flock.Public)

noncomputable abbrev Cc :=
  (keyProgP D.unroll pi hU n).circuit (FlockSoundness.Discharge.Layout.outsOfP D.unroll hU n pi)

noncomputable def Lm := Flock.Bridge.LeafMap.ofPublic D.unroll pub n

noncomputable def rows : Fin (Cc D hU n pi).N → Option Flock.Bridge.RowAt := fun g =>
  (FlockSoundness.Discharge.HiddenExec.rowOfHP D.unroll hU n pi
      (FlockSoundness.Discharge.Layout.outsOfP D.unroll hU n pi) g).map fun r =>
    ⟨r.2.1.val, (Lm D n pub).ref r.2.1.val r.1.val, FlockSoundness.Discharge.Layout.portRowW D.unroll r.2.1,
      r.2.2.val⟩

noncomputable def roots := Flock.Bridge.rootsOfL D.unroll (Lm D n pub)

noncomputable def Cd :=
  Flock.Bridge.commitDepL (Cc D hU n pi) (rows D hU n pi pub) D.unroll (Lm D n pub) (roots D n pub)

noncomputable def placeAt (i : ℕ) : Option Place :=
  placeFrom (Cc D hU n pi) (rows D hU n pi pub) D.unroll (Lm D n pub) (roots D n pub) i

structure Loads (I : FlockSoundness.Discharge.Exec.Inputs) : Prop where
  typed : I.tags.typed = false
  tags : FlockSoundness.Discharge.ZkHidden.TagsOkZHJ I
  parse : Flock.Desc.parse (FlockSoundness.Discharge.ZkExec.zkTags I) I.circuitFile I.tables = .ok D
  load : Flock.HmIn.loadPublic (FlockSoundness.Discharge.ZkExec.zkTags I) D.unroll pi I.publicFile = .ok pub
  pop : pub.n = n
  distinct : Flock.PortRoots.dupId D.unroll pub = false

structure Ready (I : FlockSoundness.Discharge.Exec.Inputs) : Prop where
  ports : Flock.PortRoots.checkOf
    (Flock.ProvedScope.circuitOf (FlockSoundness.Discharge.ZkExec.zkTags I) I.circuitFile I.tables)
    (FlockSoundness.Discharge.ZkExec.zkTags I) I.publicFile = .ok ()

variable {D n pi pub} {I : FlockSoundness.Discharge.Exec.Inputs}

theorem Ready.domain (hA : Ready I) (hL : Loads D n pi pub I) :
    ∀ q, q < D.unroll.ports.size → (Flock.Bridge.domOf D.unroll pub q).size = 64 :=
  Flock.Bridge.idsSized_of_checksStmt ((FlockSoundness.Discharge.ZkExec.zkTags_typed I).trans hL.typed)
    ((FlockSoundness.Discharge.ZkExec.zkTags_hiddenOutputs I).trans hL.tags.1) (Flock.Desc.parseStmt_of_parse hL.parse)
    hL.load hA.ports

theorem Ready.fixes_words (hA : Ready I) (hL : Loads D n pi pub I) :
    Flock.Bridge.RootFixesDomainL D.unroll (Lm D n pub) ∧ Flock.PortRoots.wordsOk D.unroll = true :=
  fixes_words_of_checksStmt ((FlockSoundness.Discharge.ZkExec.zkTags_typed I).trans hL.typed)
    ((FlockSoundness.Discharge.ZkExec.zkTags_hiddenOutputs I).trans hL.tags.1) (Flock.Desc.parseStmt_of_parse hL.parse)
    hL.load hL.pop hA.ports

theorem Ready.roots (hA : Ready I) (hL : Loads D n pi pub I) :
    ∀ q q', q' < q → q < D.unroll.ports.size →
      (Commit.roots D n pub).getD q' [] ≠ (Commit.roots D n pub).getD q [] :=
  Flock.Bridge.rootsDistinctL_of_fixes (L := Flock.Bridge.LeafMap.ofPublic D.unroll pub n)
    (Flock.Bridge.LeafMap.ofPublic_dom _ _ _) hL.distinct (hA.fixes_words hL).1

theorem Ready.words (hA : Ready I) (hL : Loads D n pi pub I) :
    Flock.PortRoots.wordsOk D.unroll = true :=
  (hA.fixes_words hL).2

variable {hU} in
theorem placeOf_words {i : ℕ} {r : Flock.Bridge.RowAt}
    (hr : Flock.Bridge.rowOfPos (Cc D hU n pi) (rows D hU n pi pub) i = some r) :
    ∃ (q : ℕ) (hq : q < D.unroll.ports.size),
      (Flock.Bridge.placeOf (Cc D hU n pi) (rows D hU n pi pub) i).1 = 16 * D.unroll.ports[q].words := by
  obtain ⟨g, -, hg⟩ := Option.bind_eq_some_iff.mp
    (show (Flock.Bridge.wireOf (Cc D hU n pi) i).bind (rows D hU n pi pub) = some r from hr)
  simp only [Commit.rows, Option.map_eq_some_iff] at hg
  obtain ⟨w, -, rfl⟩ := hg
  exact ⟨w.2.1.val, w.2.1.isLt, congrArg Sigma.fst (Flock.Bridge.placeOf_eq hr w.2.2.isLt)⟩

variable {hU} in
theorem placeOfRead_eq {r : Verity.FileRead} (hD : Flock.Desc.parse r.tags r.circuitFile r.tables = .ok D)
    (hpub : Flock.HmIn.loadPublic r.tags D.unroll {} r.publicFile = .ok pub) (hn : pub.n = n)
    (hN : r.circuit.N = (Cc D hU n {}).N) (g : Fin r.circuit.N) :
    placeOfRead r g = placeAt D hU n {} pub (Flock.Bridge.posOf (Cc D hU n {}) (Fin.cast hN g)) := by
  unfold placeOfRead
  simp only [hD, hpub, dite_eq_left hU]
  rw [hn, dite_eq_left (show g.val < (Cc D hU n {}).N from hN ▸ g.isLt)]
  rfl

variable {hU} in
theorem opens_rowLeaf (hL : Loads D n pi pub I) (hA : Ready I) {i : ℕ} {b : Bool}
    {m : Claims.Msg} (h : (Cd D hU n pi pub).Opens Claims.Sampling.sha512Msg (roots D n pub) ⟨i, b, m⟩ = true) :
    ∃ p, placeAt D hU n pi pub i = some p ∧ ∃ o : Hash.RowLeaf.Opening,
      Hash.RowLeaf.accepts Verity.sha512 Verity.sha512 p.verifier p.root p.rank o ∧ o.row[p.offset]? = some b := by
  have hR := hA.roots hL
  unfold Commit.Cd Flock.Bridge.commitDepL at h
  simp only [Claims.Sampling.CommitmentScheme.Opens, Claims.Sampling.pathCommit] at h
  cases hat : Flock.Bridge.atOf (Cc D hU n pi) (rows D hU n pi pub) (roots D n pub) i with
  | none => simp [hat, Claims.OC.eval] at h
  | some ri =>
    rw [hat] at h
    dsimp only at h
    rw [Claims.Sampling.ocBind_eval, Claims.Sampling.hm96Leaf_fst,
      ite_eq_left (And.intro (sha512Msg_length _) (sha512Msg_length _))] at h
    dsimp only at h
    obtain ⟨reg, hfind, hop⟩ := (Claims.Sampling.regPath_sha512 _ _ _ _ _).mp h
    have hmem := Array.mem_of_find?_eq_some hfind
    simp only [Flock.Bridge.portsOfL, Array.mem_map, Array.mem_range] at hmem
    obtain ⟨q, hq, hreg⟩ := hmem
    obtain ⟨r, hr, hri⟩ := Option.bind_eq_some_iff.mp
      (show (Flock.Bridge.rowOfPos (Cc D hU n pi) (rows D hU n pi pub) i).bind
          (fun r => (roots D n pub)[r.port]?.map fun rt => (rt, r.leaf)) = some ri from hat)
    obtain ⟨rt, hrt, hrti⟩ := Option.map_eq_some_iff.mp hri
    have hrq : r.port < D.unroll.ports.size := by
      simpa [Commit.roots, Flock.Bridge.rootsOfL] using (List.getElem?_eq_some_iff.mp hrt).1
    have hqr : q = r.port := by
      have e1 : bs reg.root = (roots D n pub).getD q [] := by rw [← hreg]; exact Claims.Sampling.bs_bytes _
      have e2 : bs reg.root = ri.1 := by
        have h2 := Array.find?_some hfind
        exact of_decide_eq_true h2
      have e3 : (roots D n pub).getD r.port [] = ri.1 := by simp [List.getD_eq_getElem?_getD, hrt, ← hrti]
      have he : (roots D n pub).getD q [] = (roots D n pub).getD r.port [] := e1.symm.trans (e2.trans e3.symm)
      rcases Nat.lt_trichotomy q r.port with hlt | heq | hgt
      · exact absurd he (hR r.port q hlt hrq)
      · exact heq
      · exact absurd he.symm (hR q r.port hgt hq)
    subst hqr
    have hschema : reg.schema = Flock.Registered.portSchema D.unroll.ports[r.port] := by
      rw [← hreg]; simp [Flock.Bridge.schemaOf, hq]
    have hrow : Flock.Bridge.rowDep (Cc D hU n pi) (rows D hU n pi pub) D.unroll
        (Flock.Bridge.placeOf (Cc D hU n pi) (rows D hU n pi pub)) (Flock.Bridge.bitsOf (Lm D n pub).d) ⟨i, b, m⟩ =
          Flock.Bridge.rowMsg D.unroll.ports[r.port]
            (SamplingDeploy.rowBits (Flock.Bridge.placeOf (Cc D hU n pi) (rows D hU n pi pub))
              (Flock.Bridge.bitsOf (Lm D n pub).d) ⟨i, b, m⟩) := by
      have hpo : Flock.Bridge.portOfPos (Cc D hU n pi) (rows D hU n pi pub) D.unroll i =
          some D.unroll.ports[r.port] := by
        simp [Flock.Bridge.portOfPos, hr, hq]
      simp only [Flock.Bridge.rowDep, hpo]
    have hdom : reg.domain.size = 64 := by
      rw [← hreg]
      show ((Lm D n pub).dom r.port).size = 64
      rw [Commit.Lm, Flock.Bridge.LeafMap.ofPublic_dom]
      exact hA.domain hL r.port hq
    have hsalt : (Flock.Bridge.saltOf ⟨i, b, m⟩).length = Hash.Hm96.SALT_BYTES := by
      have hop' := hop
      simp only [Flock.Registered.opensLeaf, Bool.and_eq_true, decide_eq_true_eq] at hop'
      obtain ⟨⟨-, hl⟩, -⟩ := hop'
      rw [Claims.Sampling.size_bytes] at hl
      exact Flock.Bridge.saltOf_length
        (d := Flock.log2ceil ((Lm D n pub).cnt (Flock.Bridge.portIdxOf (Cc D hU n pi) (rows D hU n pi pub) i))) hl
    have hacc := Flock.Bridge.Spec.opensLeaf_accepts sha512_hashOf hdom hop
    rw [Claims.Sampling.bs_bytes, Flock.Bridge.Spec.sha512Msg_hm96Outer sha512_hashOf, hrow, rowMsg_digest,
      ← hschema] at hacc
    have hfs : (Flock.Bridge.Spec.formatOf reg.schema).schema = reg.schema :=
      Flock.Bridge.Spec.formatOf_schema (by rw [hschema]; exact portSchema_cases _)
    have hfit : (Flock.Bridge.Spec.formatOf reg.schema).Fits
        (SamplingDeploy.rowBits (Flock.Bridge.placeOf (Cc D hU n pi) (rows D hU n pi pub))
          (Flock.Bridge.bitsOf (Lm D n pub).d) ⟨i, b, m⟩).length := by
      obtain ⟨qw, hqw, hwid⟩ := placeOf_words hr
      have h1 : (Flock.Bridge.placeOf (Cc D hU n pi) (rows D hU n pi pub) i).1 < 2 ^ 64 := by
        rw [hwid]
        exact Flock.PortRoots.words_lt_of_wordsOk (hA.words hL) hqw
      have h2 : (Flock.Bridge.placeOf (Cc D hU n pi) (rows D hU n pi pub) i).1 % 16 = 0 := by
        rw [hwid]
        exact Nat.mul_mod_right 16 _
      have h3 := Flock.Bridge.Spec.formatOf_fits reg.schema h1 h2
      simpa only [SamplingDeploy.rowBits, List.length_ofFn] using h3
    generalize hf : Flock.Bridge.Spec.formatOf reg.schema = f at hacc hfs hfit
    rw [← hfs] at hacc
    refine ⟨{ verifier := { domain := Flock.Bridge.Spec.domainOf reg, key := bs Flock.Hm96.Default512.key,
                            format := f }
              root := Flock.Bridge.Spec.dig reg.root, rank := ri.2,
              offset := (Flock.Bridge.placeOf (Cc D hU n pi) (rows D hU n pi pub) i).2.val }, ?_,
      ⟨SamplingDeploy.rowBits (Flock.Bridge.placeOf (Cc D hU n pi) (rows D hU n pi pub))
          (Flock.Bridge.bitsOf (Lm D n pub).d) ⟨i, b, m⟩,
        Flock.Bridge.saltOf ⟨i, b, m⟩, _⟩, ⟨hfit, hsalt, hacc⟩, ?_⟩
    · simp [Commit.placeAt, placeFrom, hat, hfind, hf]
    · simp [SamplingDeploy.rowBits]

end Commit

namespace Live

variable (A : Live)

noncomputable def placeAt (i : ℕ) : Option Place := placeFrom A.Cc A.rows A.D.unroll A.Lm A.roots i

theorem placeOfRead_eq {r : Verity.FileRead} (hr : r = A.fileRead) (hN : r.circuit.N = A.Cc.N)
    (g : Fin r.circuit.N) : placeOfRead r g = A.placeAt (Flock.Bridge.posOf A.Cc (Fin.cast hN g)) := by
  subst hr
  exact Commit.placeOfRead_eq A.hD A.hpub A.hn hN g

theorem placeOfFiles_eq {R : Verity.ExecutionPhaseFiles} (hr : R = Verity.ExecutionPhaseFiles.one A.fileRead)
    (hN : R.circuit.N = A.Cc.N) (g : Fin R.circuit.N) :
    placeOfFiles R g = A.placeAt (Flock.Bridge.posOf A.Cc (Fin.cast hN g)) := by
  subst hr
  exact A.placeOfRead_eq rfl hN g

theorem loads : Commit.Loads A.D A.n {} A.pub A.I :=
  ⟨A.ht, A.hTags, A.hD, A.hpub, A.hn, A.hdup⟩

theorem OpensReady.commit {A : Live} (hA : A.OpensReady) : Commit.Ready A.I := ⟨hA.ports⟩

theorem OpensReady.domain {A : Live} (hA : A.OpensReady) :
    ∀ q, q < A.D.unroll.ports.size → (Flock.Bridge.domOf A.D.unroll A.pub q).size = 64 :=
  hA.commit.domain A.loads

theorem OpensReady.fixes_words {A : Live} (hA : A.OpensReady) :
    Flock.Bridge.RootFixesDomainL A.D.unroll (Flock.Bridge.LeafMap.ofPublic A.D.unroll A.pub A.n) ∧
      Flock.PortRoots.wordsOk A.D.unroll = true :=
  hA.commit.fixes_words A.loads

theorem OpensReady.roots {A : Live} (hA : A.OpensReady) :
    ∀ q q', q' < q → q < A.D.unroll.ports.size → A.roots.getD q' [] ≠ A.roots.getD q [] :=
  hA.commit.roots A.loads

theorem OpensReady.words {A : Live} (hA : A.OpensReady) : Flock.PortRoots.wordsOk A.D.unroll = true :=
  hA.commit.words A.loads

theorem placeOf_words {i : ℕ} {r : Flock.Bridge.RowAt} (hr : Flock.Bridge.rowOfPos A.Cc A.rows i = some r) :
    ∃ (q : ℕ) (hq : q < A.D.unroll.ports.size),
      (Flock.Bridge.placeOf A.Cc A.rows i).1 = 16 * A.D.unroll.ports[q].words :=
  Commit.placeOf_words (D := A.D) (hU := A.hU) (n := A.n) (pi := {}) (pub := A.pub) hr

theorem opens_rowLeaf {A : Live} (hA : A.OpensReady) {i : ℕ} {b : Bool} {m : Claims.Msg}
    (h : A.Cd.Opens Claims.Sampling.sha512Msg A.roots ⟨i, b, m⟩ = true) :
    ∃ p, A.placeAt i = some p ∧ ∃ o : Hash.RowLeaf.Opening,
      Hash.RowLeaf.accepts Verity.sha512 Verity.sha512 p.verifier p.root p.rank o ∧ o.row[p.offset]? = some b :=
  Commit.opens_rowLeaf (D := A.D) (hU := A.hU) (n := A.n) (pi := {}) (pub := A.pub) A.loads hA.commit h

end Live

namespace Instance

theorem opensAt_of (I : Instance d α)
    (h : ∀ θ F A, I.sessions θ F = some A →
      ∀ c (hN : (I.base.read θ F).circuit.N = A.Cc.N) (g : Fin (I.base.read θ F).circuit.N),
        (I.base.read θ F).consts g = none →
          I.base.place θ F c g = A.placeAt (Flock.Bridge.posOf A.Cc (Fin.cast hN g))) :
    I.OpensAt := by
  intro θ F A hs c hN g b m hconst hop v hv
  obtain ⟨-, hsu, hready⟩ := I.sessions_some hs
  have hplace := h θ F A hs
  obtain ⟨p, hp, o, hacc, hrow⟩ := Live.opens_rowLeaf hready hop
  have hw : (I.base.read θ F).circuit.w g = 1 :=
    w_ofAudit ((congrArg Verity.ExecutionPhaseFiles.circuit hsu.read).trans
      (A.circuit_eq.trans (Flock.Bridge.circuitOf_eq A.D A.hU A.n))) g
  unfold Vocab.Opens
  rw [hconst]
  refine ⟨p, (hplace c hN g hconst).trans hp, o, hacc, fun k => ?_⟩
  have hk : k.val = 0 := by
    have h1 := k.isLt
    have h2 : (I.base.circuitOf θ F).w g = 1 := hw
    omega
  rw [hk, Nat.add_zero, hrow]
  exact congrArg some hv.symm

theorem opensAt (I : Instance d α) : I.OpensAt :=
  I.opensAt_of fun θ F A hs c hN g _ =>
    (I.place θ F c g).trans (A.placeOfFiles_eq (I.sessions_some hs).2.1.read hN g)

end Instance

/-- **`remote_soundness_deployedAt` with its openings and registered files discharged**: its openings premise
`OpensAt` holds at every instance (`Instance.opensAt`), as does the registered files' file check (`RegisteredFiles`,
`Instance.registeredFiles`); its session premise `SessionAt` is `sessionAt_of_registered`'s from `SessionRegistered`,
the ports and roots check being v2's own on the accepted play (`opensReady_of_accepted`), and `SessionRegistered` is
`Instance.sessionRegistered`'s from the session's construction (`Instance.sessionOf`) and `FamiliesPinned`, which owes
only that the coin commitment's families hold the family v2's parameters name; the coupling's, the device's log opening
only the record's rounds (`LogOpensRecorded`), is v2's record check's, which refuses a session whose coin tree opens a
round its record doesn't give (`Instance.logOpensRecorded`). The two statements on the committed files
(`Vocab.WiresSound` at the deployed vocabulary, `GateARow`) are taken as they stand: no clause of the conclusion reads
either, and neither is proved. -/
theorem remote_soundness_deployedReady (I : Instance d α) (K₀ : Constructions I.base)
    (reg : (A : Live) → Option Claims.Msg → A.Reg)
    (record : Live → ByteArray → Array ByteArray → ByteArray → Exchange → Option Claims.Msg)
    (recordDigest : Live → ByteArray → Array ByteArray → ByteArray → Claims.Msg)
    (hFP : I.FamiliesPinned .remote)
    (hW : I.base.PouwSound .remote)
    (hWi : (I.vocab (I.constructionsAt K₀ .remote reg record recordDigest)).WiresSound .remote)
    (hG : Verity.GateARow) :
    (I.vocab (I.constructionsAt K₀ .remote reg record recordDigest)).SoundnessAt .remote :=
  remote_soundness_deployedAt I K₀ reg record recordDigest I.opensAt I.registeredFiles
    (I.sessionAt_of_registered .remote (I.sessionRegistered hFP)) (I.logOpensRecorded .remote) hW hWi hG

open scoped ENNReal in
/-- **`remote_soundness_deployedReady`'s honest lemma**: its premises hold at the idle deployment's instance at v2's
reading (`Instance.Honest.inst`), whose coin commitment holds the family v2's parameters name
(`Instance.Honest.familiesPinned`) and whose base publishes `⊤` for every subprotocol, where the base's PoUW target
holds. It concludes neither statement on the committed files: the instance runs sessions, at which SPOT's error
is finite, so the wires' statement bounds something there and holds only by its proof, and `GateARow` reads no
deployment, so no instance concludes it. -/
theorem remote_soundness_deployedReady_honest :
    (Instance.Honest.inst ⊤ Instance.Honest.bound).FamiliesPinned .remote ∧
      (Instance.Honest.inst ⊤ Instance.Honest.bound).base.PouwSound .remote :=
  ⟨Instance.Honest.familiesPinned ⊤ Instance.Honest.bound .remote, Instance.Honest.target_top .remote .pouw⟩

/-- **`remote_soundness_deployedReady`'s nontrivial lemma**: its conclusion, soundness at remote's compiled verifier,
fails for a verifier that accepts everything at the idle deployment whose errors are all `0`. -/
theorem remote_soundness_deployedReady_nontrivial : ¬ Verity.Idle.exact.SoundnessFor .remote fun _ => True :=
  Verity.remote_soundness_of_subprotocols_nontrivial

theorem spotSound_deployedReady (I : Instance d α) (K₀ : Constructions I.base) (k : Verity.Registration.Composite)
    (reg : (A : Live) → Option Claims.Msg → A.Reg)
    (record : Live → ByteArray → Array ByteArray → ByteArray → Exchange → Option Claims.Msg)
    (recordDigest : Live → ByteArray → Array ByteArray → ByteArray → Claims.Msg)
    (hSr : I.SessionRegistered k) :
    (I.vocab (I.constructionsAt K₀ k reg record recordDigest)).SpotSound k :=
  spotSound_of_targets (I.vocab (I.constructionsAt K₀ k reg record recordDigest)) k
    (I.deployed (I.constructionsAt K₀ k reg record recordDigest) (fun _ _ _ _ => rfl) (fun _ _ _ _ => rfl)
      I.opensAt)
    (I.covers (I.constructionsAt K₀ k reg record recordDigest) k I.registeredFiles (I.sessionAt_of_registered k hSr))
    fun θ F A h => I.couplingTarget_lift (I.constructionsAt K₀ k reg record recordDigest) k
      (I.couplingTarget_at K₀ k reg record recordDigest (I.logOpensRecorded k) θ F A h)

theorem Instance.target_of_base (I : Instance d α) (K : Constructions I.base) (k : Verity.Registration.Composite) :
    ∀ a : Subprotocol, a ≠ .sampled → I.base.Target k a → (I.vocab K).Target k a
  | .sampled, ha, _ => absurd rfl ha
  | .pouw, _, h => h
  | .inferenceOnly, _, h => h
  | .pous, _, h => h
  | .poub, _, h => h
  | .network, _, h => h
  | .wipe, _, h => h

/-- **`Verity.Soundness` at the deployed vocabulary, SPOT's target discharged**: onsite's compiled verifier accepts
files that passed the commit phase's check while one of the period's seven claims fails with probability at most the
seven errors at that strategy summed, at the vocabulary `Instance.vocab` builds from the deployment and the
constructions `Instance.constructionsAt` assembles at onsite. It is `soundness_of_subprotocols` with SPOT's target
proved (`spotSound_deployedReady`, from Flock's live chain through `spotSound_of_targets`) from the one premise about
the deployment's sessions that `remote_soundness_deployedReady` takes, here at onsite's verdict (`FamiliesPinned`,
which gives `SessionRegistered` by `Instance.sessionRegistered`).

The other six targets are the base's (`Instance.target_of_base`), each a premise no proof here discharges: PoUW's
(`PouwSound`), the inference-only policy's (`InferenceOnlySound`), PoUS's (`PousSound`), PoUB-P2's
(`BandwidthAuditSound`), the network attester's (`NetworkAttesterSound`) and the wipe's (`WipeSound`). Of the
network attester's: its period check takes each of SPOT's answers in its evidence as given
(`NetworkCheck`'s `answered` holds an answer to one-stage's id and its registration's digest, then reads its verdict
and profile), and verifies no proof behind one, so the `K` it publishes rests on answers no check of the
composite has verified; the premise stands for that until the period check reads verified outcomes. Of the wipe's: the
bits that leave the device during the wipe are the vocabulary's (`Verity.Wipe.egress`), which no statement yet ties
to the network attester's count over the wipe's span. -/
theorem onsite_soundness_deployedReady (I : Instance d α) (K₀ : Constructions I.base)
    (reg : (A : Live) → Option Claims.Msg → A.Reg)
    (record : Live → ByteArray → Array ByteArray → ByteArray → Exchange → Option Claims.Msg)
    (recordDigest : Live → ByteArray → Array ByteArray → ByteArray → Claims.Msg)
    (hFP : I.FamiliesPinned .onsite)
    (hW : I.base.PouwSound .onsite) (hI : I.base.InferenceOnlySound .onsite)
    (hP : I.base.PousSound .onsite) (hB : I.base.BandwidthAuditSound .onsite)
    (hN : I.base.NetworkAttesterSound .onsite) (hX : I.base.WipeSound .onsite) :
    (I.vocab (I.constructionsAt K₀ .onsite reg record recordDigest)).SoundnessAt .onsite :=
  Verity.soundness_of_subprotocols ι α d (I.vocab (I.constructionsAt K₀ .onsite reg record recordDigest))
    (spotSound_deployedReady I K₀ .onsite reg record recordDigest (I.sessionRegistered hFP))
    (I.target_of_base _ .onsite .pouw (by decide) hW) (I.target_of_base _ .onsite .inferenceOnly (by decide) hI)
    (I.target_of_base _ .onsite .pous (by decide) hP) (I.target_of_base _ .onsite .poub (by decide) hB)
    (I.target_of_base _ .onsite .network (by decide) hN) (I.target_of_base _ .onsite .wipe (by decide) hX)

open scoped ENNReal in
/-- **`onsite_soundness_deployedReady`'s honest lemma**: its premises hold at the idle deployment's instance at v2's
reading (`Instance.Honest.inst`), whose coin commitment holds the family v2's parameters name
(`Instance.Honest.familiesPinned`) and whose base publishes `⊤` for every subprotocol, where the base's six targets
hold. -/
theorem onsite_soundness_deployedReady_honest :
    (Instance.Honest.inst ⊤ Instance.Honest.bound).FamiliesPinned .onsite ∧
      (Instance.Honest.inst ⊤ Instance.Honest.bound).base.PouwSound .onsite ∧
      (Instance.Honest.inst ⊤ Instance.Honest.bound).base.InferenceOnlySound .onsite ∧
      (Instance.Honest.inst ⊤ Instance.Honest.bound).base.PousSound .onsite ∧
      (Instance.Honest.inst ⊤ Instance.Honest.bound).base.BandwidthAuditSound .onsite ∧
      (Instance.Honest.inst ⊤ Instance.Honest.bound).base.NetworkAttesterSound .onsite ∧
      (Instance.Honest.inst ⊤ Instance.Honest.bound).base.WipeSound .onsite :=
  ⟨Instance.Honest.familiesPinned ⊤ Instance.Honest.bound .onsite,
    Instance.Honest.target_top .onsite .pouw, Instance.Honest.target_top .onsite .inferenceOnly,
    Instance.Honest.target_top .onsite .pous, Instance.Honest.target_top .onsite .poub,
    Instance.Honest.target_top .onsite .network, Instance.Honest.target_top .onsite .wipe⟩

/-- **`onsite_soundness_deployedReady`'s nontrivial lemma**: its conclusion, soundness at onsite's compiled verifier,
fails for a verifier that accepts everything at the idle deployment whose errors are all `0`. -/
theorem onsite_soundness_deployedReady_nontrivial : ¬ Verity.Idle.exact.SoundnessFor .onsite fun _ => True :=
  Verity.soundness_of_subprotocols_nontrivial

end Verity.Sampled
