module

public import Proofs.Properties.Verity.Sampled.Instance
public import Proofs.Properties.Verity.Sampled.OpensRecorded
public import Proofs.Properties.Verity.Sampled.Place
public import Proofs.Properties.Verity.Sampled.SessionAt
public import Proofs.Flock.Bridge.V5.Accepted

/-!
# The deployed commitment's openings are `Hash.RowLeaf`'s

`Instance.OpensAt` asks that wherever a session's deployed commitment (`Live.Cd`, Flock's `commitDepL`) opens a
gate, the base's openings hold of it (`Vocab.Opens`: a `Hash.RowLeaf` opening under SHA-512 at the base's place). Here
it is from the code's side (`Flock.Bridge.Spec`), at:

* `Live.placeAt`: **the place the session registers for a claims position**, `placeFrom` (`Sampled/Place.lean`) at
  the session's commitment (`Live.Cd`'s layout). `Live.placeOfRead_eq`: wherever v2 reads a session's files, the
  committed files' place of a gate (`placeOfRead`) is the session's `placeAt` of its claims position.
* `Live.OpensReady`: what the session's files must give: they pass `verify --zk`'s ports and roots check
  (`Flock.PortRoots.checkOf`), which v2 runs on every session it accepts (`Accepted.ports`). What `opens_rowLeaf` reads
  of the files is a property of that check:
  * `Live.OpensReady.domain`: **every port's frame-v3 domain is 64 bytes**, the width of `Hash.Tree.Domain.id`: the
    domain is the one the public file publishes (`Flock.Bridge.domOf`), and the check refuses a file whose published
    domain is not 64 bytes (`Flock.Bridge.idsSized_of_checksStmt`).
  * `Live.OpensReady.roots`: **the ports' roots are pairwise distinct**, so the port the deployed path check finds by
    its root is the row's own, of the row's format: the check refuses two ports' published roots that are one root
    under different published domains, and the session's published domains are pairwise distinct (`Live.hdup`).
  * `Live.OpensReady.words`: **every row is shorter than `2^64` bits**, `RowLeaf.accepts`'s `u64` length: the check
    refuses a port of `2^64` bits or more (`Flock.PortRoots.wordsOk`).
  Every port's row is one `Hash.RowLeaf` has a format for, whichever of Flock's three schemas it is under
  (`rowMsg_digest`, `portSchema_cases`), and its bits are whole 16-bit words (`Live.placeOf_words`), as the
  `sha512/row/v1` format takes. An accepted opening's salt is `saltLen` bytes at every depth: the deployed commitment
  refuses data shorter than the salt (`Flock.Bridge.saltOf_length`).
* `rowMsg_digest`: a port's row message (`Flock.Bridge.rowMsg`), `sha512/row-seg/v1`'s, `sha512/row/v2`'s or
  `sha512/row/v1`'s, hashes to its registered schema's `Hash.RowLeaf` digest (`Flock.Bridge.Spec.formatOf`).
* `Live.opens_rowLeaf`: **an opening the deployed commitment accepts is a `Hash.RowLeaf` opening at `placeAt`** under
  SHA-512, whose row holds the opened bit at the place's offset.
* `Instance.opensAt_of`: **`OpensAt`**, wherever each session v2 sets up is `OpensReady` and the base places each
  non-constant gate at the session's `placeAt` of its claims position. The committed files' circuit is Boolean
  (`Circuit.ofAudit`), so the opened bit is the gate's whole value. `Instance.opensAt_of_ready`: the same, for a base
  whose place is `placeOfRead` at its reading of the files, at sessions `OpensReady` alone; `Instance.opensAt_of_sessions`
  at every instance, whose base's place is `placeOfRead` by construction (`Instance.place`).
* `Instance.SessionsReady`: every session v2 sets up is `OpensReady`: its files pass the ports and roots check.
* `remote_soundness_deployedReady_honest`, `remote_soundness_deployedReady_nontrivial`: its premises hold at the idle
  deployment's instance that runs no session, and its conclusion fails for a verifier that accepts everything.
* `remote_soundness_deployedReady`: **`remote_soundness_deployedAt` with `OpensAt` discharged** at `SessionsReady`,
  `SessionAt` by `sessionAt_of_registered` at the deployment's session on the registered files, and the device's log
  opening only the record's rounds by v2's record check (`Instance.logOpensRecorded`).
-/

@[expose] public section

namespace Verity.Sampled

open Properties.NetworkCertifier.Bytes (bs bs_length)

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- `Verity.sha512` reads a byte string as the code's SHA-512 does. -/
theorem sha512_hashOf : Flock.Bridge.Spec.HashOf Verity.sha512 := fun x => by
  have hx : (⟨(bs x).toArray⟩ : ByteArray) = x := Claims.Sampling.bytes_bs x
  show (fun i : Fin 64 => (Flock.Sha512.hash ⟨(bs x).toArray⟩).data.getD i.val 0) = _
  rw [hx]
  rfl

theorem sha512Msg_length (m : Claims.Msg) : (Claims.Sampling.sha512Msg m).length = 64 := by
  rw [Claims.Sampling.sha512Msg, bs_length, Flock.Sha512.hash_size]

/-- **A port's row message hashes to its registered schema's `Hash.RowLeaf` digest** under SHA-512, whichever of
Flock's formats the port's rows are in: `sha512/row-seg/v1`, `sha512/row/v2` or `sha512/row/v1`. -/
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

/-- **A port's schema is one of Flock's three**, each one `Hash.RowLeaf` has a format for. -/
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

/-- **`verify --zk`'s files check at the statement circuit refuses colliding roots and rows of `2^64` bits**, flat or
templated: at a descriptor `Flock.Desc.parseStmt` reads under hidden outputs, when `Flock.PortRoots.checkOf` passes at
the circuit `Flock.ProvedScope.circuitOf` reads, equal roots of the statement circuit `D.unroll` at the public file's
leaf map have one published domain (`Flock.Bridge.RootFixesDomainL`, from `Flock.PortRoots.collidesPub`) and each of
its ports is under `2^64` bits (`Flock.PortRoots.wordsOk`), as at the committed circuit
(`Flock.Bridge.v5Files_of_checks`), whose ports and public file the statement circuit has. -/
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

namespace Live

variable (A : Live)

/-- **The place the session registers for claims position `i`**: `placeFrom` at the deployed commitment's layout
(`Live.Cd`'s). -/
noncomputable def placeAt (i : ℕ) : Option Place := placeFrom A.Cc A.rows A.D.unroll A.Lm A.roots i

/-- **The committed files' place for a gate is the session's at its claims position**, wherever v2 reads the
session's files. -/
theorem placeOfRead_eq {r : Verity.FileRead} (hr : r = A.fileRead) (hN : r.circuit.N = A.Cc.N)
    (g : Fin r.circuit.N) : placeOfRead r g = A.placeAt (Flock.Bridge.posOf A.Cc (Fin.cast hN g)) := by
  subst hr
  have hD : Flock.Desc.parse A.fileRead.tags A.fileRead.circuitFile A.fileRead.tables = .ok A.D := A.hD
  have hpub : Flock.HmIn.loadPublic A.fileRead.tags A.D.unroll {} A.fileRead.publicFile = .ok A.pub := A.hpub
  unfold placeOfRead
  simp only [hD, hpub, dite_eq_left A.hU]
  rw [A.hn, dite_eq_left (show g.val < A.Cc.N from hN ▸ g.isLt)]
  rfl

/-- **What the session's files must give for its openings to be `Hash.RowLeaf`'s**: they pass `verify --zk`'s ports
and roots check (`Flock.PortRoots.checkOf` at the circuit `Flock.ProvedScope.circuitOf` reads). The check refuses a
published frame-v3 domain that is not 64 bytes (`OpensReady.domain`), two ports' published roots that are one root
under different domains (`OpensReady.roots`: the deployed path check opens a row at the first port of the row's root,
which is then the row's own) and a row of `2^64` bits or more (`OpensReady.words`). -/
structure OpensReady : Prop where
  ports : Flock.PortRoots.checkOf
    (Flock.ProvedScope.circuitOf (FlockSoundness.Discharge.ZkExec.zkTags A.I) A.I.circuitFile A.I.tables)
    (FlockSoundness.Discharge.ZkExec.zkTags A.I) A.I.publicFile = .ok ()

/-- **Every port's frame-v3 domain is 64 bytes at files that pass the ports and roots check**: the domain is the one
the public file publishes for the port (`Flock.Bridge.domOf`), `Flock.PortRoots.checkOf` refuses a public file with a
published domain that is not 64 bytes (`Flock.PortRoots.idsSized`), and the session's statement circuit has the ports
and the public file of the circuit the check reads (`Flock.Bridge.idsSized_of_checksStmt`). -/
theorem OpensReady.domain {A : Live} (hA : A.OpensReady) :
    ∀ q, q < A.D.unroll.ports.size → (Flock.Bridge.domOf A.D.unroll A.pub q).size = 64 :=
  Flock.Bridge.idsSized_of_checksStmt ((FlockSoundness.Discharge.ZkExec.zkTags_typed A.I).trans A.ht)
    ((FlockSoundness.Discharge.ZkExec.zkTags_hiddenOutputs A.I).trans A.hTags.1) (Flock.Desc.parseStmt_of_parse A.hD)
    A.hpub hA.ports

/-- **What the ports and roots check refuses of the session's statement circuit**: `fixes_words_of_checksStmt` at the
session's files. -/
theorem OpensReady.fixes_words {A : Live} (hA : A.OpensReady) :
    Flock.Bridge.RootFixesDomainL A.D.unroll (Flock.Bridge.LeafMap.ofPublic A.D.unroll A.pub A.n) ∧
      Flock.PortRoots.wordsOk A.D.unroll = true :=
  fixes_words_of_checksStmt ((FlockSoundness.Discharge.ZkExec.zkTags_typed A.I).trans A.ht)
    ((FlockSoundness.Discharge.ZkExec.zkTags_hiddenOutputs A.I).trans A.hTags.1) (Flock.Desc.parseStmt_of_parse A.hD)
    A.hpub A.hn hA.ports

/-- **The ports' roots are pairwise distinct at files that pass the ports and roots check**: the check refuses two
ports' published roots that are one root under different published domains (`Flock.PortRoots.collidesPub`), and the
session's published domains are pairwise distinct (`Live.hdup`), so no two ports have one root
(`Flock.Bridge.rootsDistinctL_of_fixes`). -/
theorem OpensReady.roots {A : Live} (hA : A.OpensReady) :
    ∀ q q', q' < q → q < A.D.unroll.ports.size → A.roots.getD q' [] ≠ A.roots.getD q [] :=
  Flock.Bridge.rootsDistinctL_of_fixes (L := Flock.Bridge.LeafMap.ofPublic A.D.unroll A.pub A.n)
    (Flock.Bridge.LeafMap.ofPublic_dom _ _ _) A.hdup hA.fixes_words.1

/-- **Every port is under `2^64` bits at files that pass the ports and roots check** (`Flock.PortRoots.wordsOk`). -/
theorem OpensReady.words {A : Live} (hA : A.OpensReady) : Flock.PortRoots.wordsOk A.D.unroll = true :=
  hA.fixes_words.2

/-- **A position's row is its port's whole 16-bit words**: where claims position `i` has a row, the width the session
registers for it (`Flock.Bridge.placeOf`'s) is `16 · words` of a port of the statement circuit, by the session's row
map's construction (`Live.rows`, at `Layout.portRowW`). -/
theorem placeOf_words {i : ℕ} {r : Flock.Bridge.RowAt} (hr : Flock.Bridge.rowOfPos A.Cc A.rows i = some r) :
    ∃ (q : ℕ) (hq : q < A.D.unroll.ports.size),
      (Flock.Bridge.placeOf A.Cc A.rows i).1 = 16 * A.D.unroll.ports[q].words := by
  obtain ⟨g, -, hg⟩ := Option.bind_eq_some_iff.mp
    (show (Flock.Bridge.wireOf A.Cc i).bind A.rows = some r from hr)
  simp only [Live.rows, Option.map_eq_some_iff] at hg
  obtain ⟨w, -, rfl⟩ := hg
  exact ⟨w.2.1.val, w.2.1.isLt, congrArg Sigma.fst (Flock.Bridge.placeOf_eq hr w.2.2.isLt)⟩

/-- **An opening the deployed commitment accepts is a `Hash.RowLeaf` opening at the registered place**: under
SHA-512, at `placeAt`'s verifier, root and rank, its row holding the opened bit at the place's offset. -/
theorem opens_rowLeaf {A : Live} (hA : A.OpensReady) {i : ℕ} {b : Bool} {m : Claims.Msg}
    (h : A.Cd.Opens Claims.Sampling.sha512Msg A.roots ⟨i, b, m⟩ = true) :
    ∃ p, A.placeAt i = some p ∧ ∃ o : Hash.RowLeaf.Opening,
      Hash.RowLeaf.accepts Verity.sha512 Verity.sha512 p.verifier p.root p.rank o ∧ o.row[p.offset]? = some b := by
  unfold Live.Cd Flock.Bridge.commitDepL at h
  simp only [Claims.Sampling.CommitmentScheme.Opens, Claims.Sampling.pathCommit] at h
  cases hat : Flock.Bridge.atOf A.Cc A.rows A.roots i with
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
      (show (Flock.Bridge.rowOfPos A.Cc A.rows i).bind (fun r => A.roots[r.port]?.map fun rt => (rt, r.leaf)) =
        some ri from hat)
    obtain ⟨rt, hrt, hrti⟩ := Option.map_eq_some_iff.mp hri
    have hrq : r.port < A.D.unroll.ports.size := by
      simpa [Live.roots, Flock.Bridge.rootsOfL] using (List.getElem?_eq_some_iff.mp hrt).1
    have hqr : q = r.port := by
      have e1 : bs reg.root = A.roots.getD q [] := by rw [← hreg]; exact Claims.Sampling.bs_bytes _
      have e2 : bs reg.root = ri.1 := by
        have h2 := Array.find?_some hfind
        exact of_decide_eq_true h2
      have e3 : A.roots.getD r.port [] = ri.1 := by simp [List.getD_eq_getElem?_getD, hrt, ← hrti]
      have he : A.roots.getD q [] = A.roots.getD r.port [] := e1.symm.trans (e2.trans e3.symm)
      rcases Nat.lt_trichotomy q r.port with hlt | heq | hgt
      · exact absurd he (hA.roots r.port q hlt hrq)
      · exact heq
      · exact absurd he.symm (hA.roots q r.port hgt hq)
    subst hqr
    have hschema : reg.schema = Flock.Registered.portSchema A.D.unroll.ports[r.port] := by
      rw [← hreg]; simp [Flock.Bridge.schemaOf, hq]
    have hrow : Flock.Bridge.rowDep A.Cc A.rows A.D.unroll (Flock.Bridge.placeOf A.Cc A.rows)
        (Flock.Bridge.bitsOf A.Lm.d) ⟨i, b, m⟩ = Flock.Bridge.rowMsg A.D.unroll.ports[r.port]
          (SamplingDeploy.rowBits (Flock.Bridge.placeOf A.Cc A.rows) (Flock.Bridge.bitsOf A.Lm.d) ⟨i, b, m⟩) := by
      have hpo : Flock.Bridge.portOfPos A.Cc A.rows A.D.unroll i = some A.D.unroll.ports[r.port] := by
        simp [Flock.Bridge.portOfPos, hr, hq]
      simp only [Flock.Bridge.rowDep, hpo]
    have hdom : reg.domain.size = 64 := by
      rw [← hreg]
      show (A.Lm.dom r.port).size = 64
      rw [Live.Lm, Flock.Bridge.LeafMap.ofPublic_dom]
      exact hA.domain r.port hq
    have hsalt : (Flock.Bridge.saltOf ⟨i, b, m⟩).length = Hash.Hm96.SALT_BYTES := by
      have hop' := hop
      simp only [Flock.Registered.opensLeaf, Bool.and_eq_true, decide_eq_true_eq] at hop'
      obtain ⟨⟨-, hl⟩, -⟩ := hop'
      rw [Claims.Sampling.size_bytes] at hl
      exact Flock.Bridge.saltOf_length (d := Flock.log2ceil (A.Lm.cnt (Flock.Bridge.portIdxOf A.Cc A.rows i))) hl
    have hacc := Flock.Bridge.Spec.opensLeaf_accepts sha512_hashOf hdom hop
    rw [Claims.Sampling.bs_bytes, Flock.Bridge.Spec.sha512Msg_hm96Outer sha512_hashOf, hrow, rowMsg_digest,
      ← hschema] at hacc
    have hfs : (Flock.Bridge.Spec.formatOf reg.schema).schema = reg.schema :=
      Flock.Bridge.Spec.formatOf_schema (by rw [hschema]; exact portSchema_cases _)
    have hfit : (Flock.Bridge.Spec.formatOf reg.schema).Fits
        (SamplingDeploy.rowBits (Flock.Bridge.placeOf A.Cc A.rows) (Flock.Bridge.bitsOf A.Lm.d) ⟨i, b, m⟩).length := by
      obtain ⟨qw, hqw, hwid⟩ := A.placeOf_words hr
      have h1 : (Flock.Bridge.placeOf A.Cc A.rows i).1 < 2 ^ 64 := by
        rw [hwid]
        exact Flock.PortRoots.words_lt_of_wordsOk hA.words hqw
      have h2 : (Flock.Bridge.placeOf A.Cc A.rows i).1 % 16 = 0 := by
        rw [hwid]
        exact Nat.mul_mod_right 16 _
      have h3 := Flock.Bridge.Spec.formatOf_fits reg.schema h1 h2
      simpa only [SamplingDeploy.rowBits, List.length_ofFn] using h3
    generalize hf : Flock.Bridge.Spec.formatOf reg.schema = f at hacc hfs hfit
    rw [← hfs] at hacc
    refine ⟨{ verifier := { domain := Flock.Bridge.Spec.domainOf reg, key := bs Flock.Hm96.Default512.key,
                            format := f }
              root := Flock.Bridge.Spec.dig reg.root, rank := ri.2,
              offset := (Flock.Bridge.placeOf A.Cc A.rows i).2.val }, ?_,
      ⟨SamplingDeploy.rowBits (Flock.Bridge.placeOf A.Cc A.rows) (Flock.Bridge.bitsOf A.Lm.d) ⟨i, b, m⟩,
        Flock.Bridge.saltOf ⟨i, b, m⟩, _⟩, ⟨hfit, hsalt, hacc⟩, ?_⟩
    · simp [placeAt, placeFrom, hat, hfind, hf]
    · simp [SamplingDeploy.rowBits]

end Live

namespace Instance

/-- **Every session v2 sets up is `OpensReady`**: its files pass `verify --zk`'s ports and roots check, which gives
what `Live.opens_rowLeaf` reads of them. -/
def SessionsReady (I : Instance d α) : Prop :=
  ∀ θ F A, I.session θ = some A → SetsUp I.base θ F A → A.OpensReady

/-- **`OpensAt`, at sessions ready for it**: wherever each session v2 sets up on the committed files is `OpensReady`
and the base places each non-constant gate at the session's registered place of its claims position (`placeAt`), the
deployed commitment's openings are the base's. -/
theorem opensAt_of (I : Instance d α)
    (h : ∀ θ F A, I.session θ = some A → SetsUp I.base θ F A → A.OpensReady ∧
      ∀ c (hN : (I.base.read θ F).circuit.N = A.Cc.N) (g : Fin (I.base.read θ F).circuit.N),
        (I.base.read θ F).consts g = none →
          I.base.place θ F c g = A.placeAt (Flock.Bridge.posOf A.Cc (Fin.cast hN g))) :
    I.OpensAt := by
  intro θ F A hs hsu c hN g b m hconst hop v hv
  obtain ⟨hready, hplace⟩ := h θ F A hs hsu
  obtain ⟨p, hp, o, hacc, hrow⟩ := Live.opens_rowLeaf hready hop
  have hw : (I.base.read θ F).circuit.w g = 1 :=
    w_ofAudit ((congrArg Verity.FileRead.circuit hsu.read).trans
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

/-- **`OpensAt` at the committed files' places**: wherever the base places each gate at the committed files' place
(`placeOfRead`, by definition at the deployment) and each session v2 sets up is `OpensReady`. -/
theorem opensAt_of_ready (I : Instance d α)
    (hplace : ∀ θ F c g, I.base.place θ F c g = placeOfRead (I.base.read θ F) g)
    (h : I.SessionsReady) : I.OpensAt :=
  I.opensAt_of fun θ F A hs hsu => ⟨h θ F A hs hsu, fun c hN g _ =>
    (hplace θ F c g).trans (A.placeOfRead_eq hsu.read hN g)⟩

/-- **`OpensAt` at sessions `OpensReady`**: an instance's base places each gate at the committed files' place by its
construction (`Instance.place`), so `OpensAt` needs only that each session v2 sets up is `OpensReady`. -/
theorem opensAt_of_sessions (I : Instance d α) (h : I.SessionsReady) : I.OpensAt :=
  I.opensAt_of_ready I.place h

end Instance

/-- **`remote_soundness_deployedAt` at sessions `OpensReady`**: its openings premise `OpensAt` discharged by
`opensAt_of_sessions`, its session premise `SessionAt` by `sessionAt_of_registered` from `SessionRegistered`, and the
coupling's, the device's log opening only the record's rounds (`LogOpensRecorded`), by v2's record check, which
refuses a session whose coin tree opens a round its record doesn't give (`Instance.logOpensRecorded`). -/
theorem remote_soundness_deployedReady (I : Instance d α) (K₀ : Constructions I.base)
    (reg : (A : Live) → Option Claims.Msg → A.Reg)
    (record : Live → ByteArray → Array ByteArray → ByteArray → Exchange → Option Claims.Msg)
    (recordDigest : Live → ByteArray → Array ByteArray → ByteArray → Claims.Msg)
    (hready : I.SessionsReady) (hRg : I.RegisteredFiles) (hSr : I.SessionRegistered .remote)
    (hW : I.base.PouwSound .remote) (hI : I.base.InferenceOnlySound .remote) :
    (I.vocab (I.constructionsAt K₀ .remote reg record recordDigest)).SoundnessAt .remote :=
  remote_soundness_deployedAt I K₀ reg record recordDigest (I.opensAt_of_sessions hready) hRg
    (I.sessionAt_of_registered .remote hSr) (I.logOpensRecorded .remote) hW hI

open scoped ENNReal in
/-- **`remote_soundness_deployedReady`'s honest lemma**: its premises hold at the idle deployment's instance at v2's
reading (`Instance.Honest.inst`) that runs no session, whose base publishes `⊤` for every subprotocol: each deployment
premise speaks of the sessions the deployment runs, and the base's PoUW and inference-only targets hold at `⊤`. -/
theorem remote_soundness_deployedReady_honest :
    (Instance.Honest.inst ⊤ fun _ => none).SessionsReady ∧ (Instance.Honest.inst ⊤ fun _ => none).RegisteredFiles ∧
      (Instance.Honest.inst ⊤ fun _ => none).SessionRegistered .remote ∧
      (Instance.Honest.inst ⊤ fun _ => none).base.PouwSound .remote ∧
      (Instance.Honest.inst ⊤ fun _ => none).base.InferenceOnlySound .remote :=
  ⟨fun _ _ _ h => by simp [Instance.Honest.inst] at h, Instance.Honest.registeredFiles ⊤ _,
    Instance.Honest.sessionRegistered_none ⊤ .remote,
    Instance.Honest.target_top .remote .pouw, Instance.Honest.target_top .remote .inferenceOnly⟩

/-- **`remote_soundness_deployedReady`'s nontrivial lemma**: its conclusion, soundness at remote's compiled verifier,
fails for a verifier that accepts everything at the idle deployment whose errors are all `0`. -/
theorem remote_soundness_deployedReady_nontrivial : ¬ Verity.Idle.exact.SoundnessFor .remote fun _ => True :=
  Verity.remote_soundness_of_subprotocols_nontrivial

end Verity.Sampled
