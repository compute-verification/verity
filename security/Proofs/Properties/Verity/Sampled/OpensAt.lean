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

* `Commit`: **the same over the commitment's data** (the descriptor, the population, the public input ports and the
  public file), of which a session's (`Live`, no public input port) and a window's member's (`MemberV5`) are
  instances: its circuit, leaf map, rows, roots and commitment (`Commit.Cc`, `Commit.Lm`, `Commit.rows`,
  `Commit.roots`, `Commit.Cd`, the formulas `Live`'s and `MemberV5`'s are) and its place (`Commit.placeAt`).
  `Commit.Loads`: the files load to that data. `Commit.Ready`: what the files must give, that they pass `verify --zk`'s
  ports and roots check (`Flock.PortRoots.checkOf`), which v2 runs on every session it accepts (`Accepted.ports`).
  What `Commit.opens_rowLeaf` reads of the files is a property of that check:
  * `Commit.Ready.domain`: **every port's frame-v3 domain is 64 bytes**, the width of `Hash.Tree.Domain.id`: the
    domain is the one the public file publishes (`Flock.Bridge.domOf`), and the check refuses a file whose published
    domain is not 64 bytes (`Flock.Bridge.idsSized_of_checksStmt`).
  * `Commit.Ready.roots`: **the ports' roots are pairwise distinct**, so the port the deployed path check finds by
    its root is the row's own, of the row's format: the check refuses two ports' published roots that are one root
    under different published domains (`Commit.Ready.fixes_words`), and the published domains are pairwise distinct
    (`Commit.Loads.distinct`).
  * `Commit.Ready.words`: **every row is shorter than `2^64` bits**, `RowLeaf.accepts`'s `u64` length: the check
    refuses a port of `2^64` bits or more (`Flock.PortRoots.wordsOk`).
  Every port's row is one `Hash.RowLeaf` has a format for, whichever of Flock's three schemas it is under
  (`rowMsg_digest`, `portSchema_cases`), and its bits are whole 16-bit words (`Commit.placeOf_words`), as the
  `sha512/row/v1` format takes. An accepted opening's salt is `saltLen` bytes at every depth: the deployed commitment
  refuses data shorter than the salt (`Flock.Bridge.saltOf_length`). `Commit.placeOfRead_eq`: wherever the committed
  files parse and load to the commitment's data, their place of a gate (`placeOfRead`) is the commitment's `placeAt`
  of its claims position.
* `rowMsg_digest`: a port's row message (`Flock.Bridge.rowMsg`), `sha512/row-seg/v1`'s, `sha512/row/v2`'s or
  `sha512/row/v1`'s, hashes to its registered schema's `Hash.RowLeaf` digest (`Flock.Bridge.Spec.formatOf`).
* `Commit.opens_rowLeaf`: **an opening the deployed commitment accepts is a `Hash.RowLeaf` opening at `placeAt`**
  under SHA-512, whose row holds the opened bit at the place's offset.
* `Live`: the same at a session, each an instance of `Commit`'s: `Live.placeAt`, `Live.placeOfRead_eq`, `Live.loads`,
  and at `Live.OpensReady` (`Sampled/Instance.lean`), the session's files passing the ports and roots check, which each
  of the deployment's sessions does (`Instance.sessions`; `Live.OpensReady.commit`), its `domain`, `roots`, `words` and
  `opens_rowLeaf`.
* `Instance.opensAt_of`: **`OpensAt`**, wherever the base places each non-constant gate of a session's files at the
  session's `placeAt` of its claims position. The committed files' circuit is Boolean (`Circuit.ofAudit`), so the
  opened bit is the gate's whole value. `Instance.opensAt`: at every instance, whose base's place is `placeOfFiles` by
  construction (`Instance.place`).
* `remote_soundness_deployedReady_honest`, `remote_soundness_deployedReady_nontrivial`: its premises hold at the idle
  deployment's instance that runs no session, and its conclusion fails for a verifier that accepts everything.
* `remote_soundness_deployedReady`: **`remote_soundness_deployedAt` with `OpensAt` and `RegisteredFiles` discharged**
  at every instance (`Instance.opensAt`, `Instance.registeredFiles`), `SessionAt` by `sessionAt_of_registered` at the
  deployment's session on the registered files, and the device's log opening only the record's rounds by v2's record
  check (`Instance.logOpensRecorded`).
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

namespace Commit

open FlockSoundness FlockSoundness.Discharge.Placed

variable (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (n : ℕ) (pi : Flock.HmIn.Ports)
  (pub : Flock.Public)

/-- **The deployed commitment's circuit** at the descriptor `D`, the population `n` and the public input ports `pi`:
the key program's at those ports, with its outputs (`Live.Cc` at no public input port, `MemberV5.Cc`). -/
noncomputable abbrev Cc :=
  (keyProgP D.unroll pi hU n).circuit (FlockSoundness.Discharge.Layout.outsOfP D.unroll hU n pi)

/-- The public file `pub`'s leaf map at the population `n`. -/
noncomputable def Lm := Flock.Bridge.LeafMap.ofPublic D.unroll pub n

/-- The deployed commitment's rows: each committed gate's port, leaf, width and place, read off the descriptor. -/
noncomputable def rows : Fin (Cc D hU n pi).N → Option Flock.Bridge.RowAt := fun g =>
  (FlockSoundness.Discharge.HiddenExec.rowOfHP D.unroll hU n pi
      (FlockSoundness.Discharge.Layout.outsOfP D.unroll hU n pi) g).map fun r =>
    ⟨r.2.1.val, (Lm D n pub).ref r.2.1.val r.1.val, FlockSoundness.Discharge.Layout.portRowW D.unroll r.2.1,
      r.2.2.val⟩

/-- The deployed commitment's roots, one a port. -/
noncomputable def roots := Flock.Bridge.rootsOfL D.unroll (Lm D n pub)

/-- **The deployed commitment** at the public file's leaf map (`Flock.Bridge.commitDepL`). -/
noncomputable def Cd :=
  Flock.Bridge.commitDepL (Cc D hU n pi) (rows D hU n pi pub) D.unroll (Lm D n pub) (roots D n pub)

/-- **The place the commitment registers for claims position `i`**: `placeFrom` at its layout. -/
noncomputable def placeAt (i : ℕ) : Option Place :=
  placeFrom (Cc D hU n pi) (rows D hU n pi pub) D.unroll (Lm D n pub) (roots D n pub) i

/-- **The files `I` load to the commitment's data**: untyped, under the zero-knowledge path's tags, they parse to the
descriptor `D` and load, at the public input ports `pi`, to the public file `pub` of the population `n`, whose
published frame-v3 domains are pairwise distinct. -/
structure Loads (I : FlockSoundness.Discharge.Exec.Inputs) : Prop where
  typed : I.tags.typed = false
  tags : FlockSoundness.Discharge.ZkHidden.TagsOkZHJ I
  parse : Flock.Desc.parse (FlockSoundness.Discharge.ZkExec.zkTags I) I.circuitFile I.tables = .ok D
  load : Flock.HmIn.loadPublic (FlockSoundness.Discharge.ZkExec.zkTags I) D.unroll pi I.publicFile = .ok pub
  pop : pub.n = n
  distinct : Flock.PortRoots.dupId D.unroll pub = false

/-- **What the commitment's files must give for its openings to be `Hash.RowLeaf`'s**: they pass `verify --zk`'s
ports and roots check (`Flock.PortRoots.checkOf` at the circuit `Flock.ProvedScope.circuitOf` reads). At files that
load to the commitment's data, the check refuses a published frame-v3 domain that is not 64 bytes (`Ready.domain`),
two ports' published roots that are one root under different domains (`Ready.roots`: the deployed path check opens a
row at the first port of the row's root, which is then the row's own) and a row of `2^64` bits or more
(`Ready.words`). -/
structure Ready (I : FlockSoundness.Discharge.Exec.Inputs) : Prop where
  ports : Flock.PortRoots.checkOf
    (Flock.ProvedScope.circuitOf (FlockSoundness.Discharge.ZkExec.zkTags I) I.circuitFile I.tables)
    (FlockSoundness.Discharge.ZkExec.zkTags I) I.publicFile = .ok ()

variable {D n pi pub} {I : FlockSoundness.Discharge.Exec.Inputs}

/-- **Every port's frame-v3 domain is 64 bytes at files that pass the ports and roots check**: the domain is the one
the public file publishes for the port (`Flock.Bridge.domOf`), `Flock.PortRoots.checkOf` refuses a public file with a
published domain that is not 64 bytes (`Flock.PortRoots.idsSized`), and the statement circuit has the ports and the
public file of the circuit the check reads (`Flock.Bridge.idsSized_of_checksStmt`). -/
theorem Ready.domain (hA : Ready I) (hL : Loads D n pi pub I) :
    ∀ q, q < D.unroll.ports.size → (Flock.Bridge.domOf D.unroll pub q).size = 64 :=
  Flock.Bridge.idsSized_of_checksStmt ((FlockSoundness.Discharge.ZkExec.zkTags_typed I).trans hL.typed)
    ((FlockSoundness.Discharge.ZkExec.zkTags_hiddenOutputs I).trans hL.tags.1) (Flock.Desc.parseStmt_of_parse hL.parse)
    hL.load hA.ports

/-- **What the ports and roots check refuses of the commitment's statement circuit**: `fixes_words_of_checksStmt` at
files that load to the commitment's data. -/
theorem Ready.fixes_words (hA : Ready I) (hL : Loads D n pi pub I) :
    Flock.Bridge.RootFixesDomainL D.unroll (Lm D n pub) ∧ Flock.PortRoots.wordsOk D.unroll = true :=
  fixes_words_of_checksStmt ((FlockSoundness.Discharge.ZkExec.zkTags_typed I).trans hL.typed)
    ((FlockSoundness.Discharge.ZkExec.zkTags_hiddenOutputs I).trans hL.tags.1) (Flock.Desc.parseStmt_of_parse hL.parse)
    hL.load hL.pop hA.ports

/-- **The ports' roots are pairwise distinct at files that pass the ports and roots check**: the check refuses two
ports' published roots that are one root under different published domains (`Flock.PortRoots.collidesPub`), and the
published domains are pairwise distinct (`Loads.distinct`), so no two ports have one root
(`Flock.Bridge.rootsDistinctL_of_fixes`). -/
theorem Ready.roots (hA : Ready I) (hL : Loads D n pi pub I) :
    ∀ q q', q' < q → q < D.unroll.ports.size →
      (Commit.roots D n pub).getD q' [] ≠ (Commit.roots D n pub).getD q [] :=
  Flock.Bridge.rootsDistinctL_of_fixes (L := Flock.Bridge.LeafMap.ofPublic D.unroll pub n)
    (Flock.Bridge.LeafMap.ofPublic_dom _ _ _) hL.distinct (hA.fixes_words hL).1

/-- **Every port is under `2^64` bits at files that pass the ports and roots check** (`Flock.PortRoots.wordsOk`). -/
theorem Ready.words (hA : Ready I) (hL : Loads D n pi pub I) :
    Flock.PortRoots.wordsOk D.unroll = true :=
  (hA.fixes_words hL).2

variable {hU} in
/-- **A position's row is its port's whole 16-bit words**: where claims position `i` has a row, the width the
commitment registers for it (`Flock.Bridge.placeOf`'s) is `16 · words` of a port of the statement circuit, by the
commitment's row map's construction (`Commit.rows`, at `Layout.portRowW`). -/
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
/-- **The committed files' place for a gate is the commitment's at its claims position**, wherever the files parse
to the commitment's descriptor and load to its public file, of its population, with no public input port. -/
theorem placeOfRead_eq {r : Verity.FileRead} (hD : Flock.Desc.parse r.tags r.circuitFile r.tables = .ok D)
    (hpub : Flock.HmIn.loadPublic r.tags D.unroll {} r.publicFile = .ok pub) (hn : pub.n = n)
    (hN : r.circuit.N = (Cc D hU n {}).N) (g : Fin r.circuit.N) :
    placeOfRead r g = placeAt D hU n {} pub (Flock.Bridge.posOf (Cc D hU n {}) (Fin.cast hN g)) := by
  unfold placeOfRead
  simp only [hD, hpub, dite_eq_left hU]
  rw [hn, dite_eq_left (show g.val < (Cc D hU n {}).N from hN ▸ g.isLt)]
  rfl

variable {hU} in
/-- **An opening the deployed commitment accepts is a `Hash.RowLeaf` opening at the registered place**: under
SHA-512, at `placeAt`'s verifier, root and rank, its row holding the opened bit at the place's offset, at files that
load to the commitment's data and pass the ports and roots check. -/
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

/-- **The place the session registers for claims position `i`**: `placeFrom` at the deployed commitment's layout
(`Live.Cd`'s). -/
noncomputable def placeAt (i : ℕ) : Option Place := placeFrom A.Cc A.rows A.D.unroll A.Lm A.roots i

/-- **The committed files' place for a gate is the session's at its claims position**, wherever v2 reads the
session's files. -/
theorem placeOfRead_eq {r : Verity.FileRead} (hr : r = A.fileRead) (hN : r.circuit.N = A.Cc.N)
    (g : Fin r.circuit.N) : placeOfRead r g = A.placeAt (Flock.Bridge.posOf A.Cc (Fin.cast hN g)) := by
  subst hr
  exact Commit.placeOfRead_eq A.hD A.hpub A.hn hN g

/-- **The window's committed files' place for a gate is the session's at its claims position**, wherever v2 reads
the session's files as a window of one member (`placeOfFiles`, there the member's `placeOfRead`). -/
theorem placeOfFiles_eq {R : Verity.WindowFiles} (hr : R = Verity.WindowFiles.one A.fileRead)
    (hN : R.circuit.N = A.Cc.N) (g : Fin R.circuit.N) :
    placeOfFiles R g = A.placeAt (Flock.Bridge.posOf A.Cc (Fin.cast hN g)) := by
  subst hr
  exact A.placeOfRead_eq rfl hN g

/-- **The session's files load to its commitment's data** (`Commit.Loads`), with no public input port. -/
theorem loads : Commit.Loads A.D A.n {} A.pub A.I :=
  ⟨A.ht, A.hTags, A.hD, A.hpub, A.hn, A.hdup⟩

/-- **A session ready for its openings has files ready for them** (`Commit.Ready`). -/
theorem OpensReady.commit {A : Live} (hA : A.OpensReady) : Commit.Ready A.I := ⟨hA.ports⟩

/-- **Every port's frame-v3 domain is 64 bytes at files that pass the ports and roots check**
(`Commit.Ready.domain`). -/
theorem OpensReady.domain {A : Live} (hA : A.OpensReady) :
    ∀ q, q < A.D.unroll.ports.size → (Flock.Bridge.domOf A.D.unroll A.pub q).size = 64 :=
  hA.commit.domain A.loads

/-- **What the ports and roots check refuses of the session's statement circuit** (`Commit.Ready.fixes_words`). -/
theorem OpensReady.fixes_words {A : Live} (hA : A.OpensReady) :
    Flock.Bridge.RootFixesDomainL A.D.unroll (Flock.Bridge.LeafMap.ofPublic A.D.unroll A.pub A.n) ∧
      Flock.PortRoots.wordsOk A.D.unroll = true :=
  hA.commit.fixes_words A.loads

/-- **The ports' roots are pairwise distinct at files that pass the ports and roots check**
(`Commit.Ready.roots`). -/
theorem OpensReady.roots {A : Live} (hA : A.OpensReady) :
    ∀ q q', q' < q → q < A.D.unroll.ports.size → A.roots.getD q' [] ≠ A.roots.getD q [] :=
  hA.commit.roots A.loads

/-- **Every port is under `2^64` bits at files that pass the ports and roots check** (`Commit.Ready.words`). -/
theorem OpensReady.words {A : Live} (hA : A.OpensReady) : Flock.PortRoots.wordsOk A.D.unroll = true :=
  hA.commit.words A.loads

/-- **A position's row is its port's whole 16-bit words** (`Commit.placeOf_words`). -/
theorem placeOf_words {i : ℕ} {r : Flock.Bridge.RowAt} (hr : Flock.Bridge.rowOfPos A.Cc A.rows i = some r) :
    ∃ (q : ℕ) (hq : q < A.D.unroll.ports.size),
      (Flock.Bridge.placeOf A.Cc A.rows i).1 = 16 * A.D.unroll.ports[q].words :=
  Commit.placeOf_words (D := A.D) (hU := A.hU) (n := A.n) (pi := {}) (pub := A.pub) hr

/-- **An opening the deployed commitment accepts is a `Hash.RowLeaf` opening at the registered place**: under
SHA-512, at `placeAt`'s verifier, root and rank, its row holding the opened bit at the place's offset. -/
theorem opens_rowLeaf {A : Live} (hA : A.OpensReady) {i : ℕ} {b : Bool} {m : Claims.Msg}
    (h : A.Cd.Opens Claims.Sampling.sha512Msg A.roots ⟨i, b, m⟩ = true) :
    ∃ p, A.placeAt i = some p ∧ ∃ o : Hash.RowLeaf.Opening,
      Hash.RowLeaf.accepts Verity.sha512 Verity.sha512 p.verifier p.root p.rank o ∧ o.row[p.offset]? = some b :=
  Commit.opens_rowLeaf (D := A.D) (hU := A.hU) (n := A.n) (pi := {}) (pub := A.pub) A.loads hA.commit h

end Live

namespace Instance

/-- **`OpensAt`, at the base's places**: wherever the base places each non-constant gate of a session's committed files
at the session's registered place of its claims position (`placeAt`), the deployed commitment's openings are the
base's: each of the deployment's sessions is set up on the files (`SetsUp`) and passes the ports and roots check
(`Live.OpensReady`, `Instance.sessions`). -/
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
    w_ofAudit ((congrArg Verity.WindowFiles.circuit hsu.read).trans
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

/-- **`OpensAt` at every instance**: an instance's base places each gate at the committed files' place by its
construction (`Instance.place`, `placeOfFiles`), which is the session's `placeAt` wherever v2 reads the session's files
(`Live.placeOfFiles_eq`), and each of its sessions passes the ports and roots check (`Instance.sessions`). -/
theorem opensAt (I : Instance d α) : I.OpensAt :=
  I.opensAt_of fun θ F A hs c hN g _ =>
    (I.place θ F c g).trans (A.placeOfFiles_eq (I.sessions_some hs).2.1.read hN g)

end Instance

/-- **`remote_soundness_deployedAt` with its openings and registered files discharged**: its openings premise
`OpensAt` holds at every instance (`Instance.opensAt`), as does the registered files' file check (`RegisteredFiles`,
`Instance.registeredFiles`); its session premise `SessionAt` is `sessionAt_of_registered`'s from `SessionRegistered`,
and the coupling's, the device's log opening only the record's rounds (`LogOpensRecorded`), is v2's record check's,
which refuses a session whose coin tree opens a round its record doesn't give (`Instance.logOpensRecorded`). -/
theorem remote_soundness_deployedReady (I : Instance d α) (K₀ : Constructions I.base)
    (reg : (A : Live) → Option Claims.Msg → A.Reg)
    (record : Live → ByteArray → Array ByteArray → ByteArray → Exchange → Option Claims.Msg)
    (recordDigest : Live → ByteArray → Array ByteArray → ByteArray → Claims.Msg)
    (hSr : I.SessionRegistered .remote)
    (hW : I.base.PouwSound .remote) (hI : I.base.InferenceOnlySound .remote) :
    (I.vocab (I.constructionsAt K₀ .remote reg record recordDigest)).SoundnessAt .remote :=
  remote_soundness_deployedAt I K₀ reg record recordDigest I.opensAt I.registeredFiles
    (I.sessionAt_of_registered .remote hSr) (I.logOpensRecorded .remote) hW hI

open scoped ENNReal in
/-- **`remote_soundness_deployedReady`'s honest lemma**: its premises hold at the idle deployment's instance at v2's
reading (`Instance.Honest.inst`) that runs no session, whose base publishes `⊤` for every subprotocol: the session
premise speaks of the sessions the deployment runs, and the base's PoUW and inference-only targets hold at `⊤`. -/
theorem remote_soundness_deployedReady_honest :
    (Instance.Honest.inst ⊤ fun _ => none).SessionRegistered .remote ∧
      (Instance.Honest.inst ⊤ fun _ => none).base.PouwSound .remote ∧
      (Instance.Honest.inst ⊤ fun _ => none).base.InferenceOnlySound .remote :=
  ⟨Instance.Honest.sessionRegistered_none ⊤ .remote,
    Instance.Honest.target_top .remote .pouw, Instance.Honest.target_top .remote .inferenceOnly⟩

/-- **`remote_soundness_deployedReady`'s nontrivial lemma**: its conclusion, soundness at remote's compiled verifier,
fails for a verifier that accepts everything at the idle deployment whose errors are all `0`. -/
theorem remote_soundness_deployedReady_nontrivial : ¬ Verity.Idle.exact.SoundnessFor .remote fun _ => True :=
  Verity.remote_soundness_of_subprotocols_nontrivial

end Verity.Sampled
