module

public import Proofs.Properties.Verity.Sampled.Instance
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
* `Live.OpensReady`: what the session's files must give: every port a `sha512/row/v2` or `sha512/row-seg/v1` row,
  formats `Hash.RowLeaf` takes from the code's schemas (`Flock.Bridge.Spec.formatOf`); the files pass `verify --zk`'s
  ports and roots check (`Flock.PortRoots.checkOf`), which v2 runs on every session it accepts (`Accepted.ports`);
  every port at least two leaves, so an accepted opening's path has a level and its salt is `saltLen` bytes
  (`Flock.Bridge.saltOf_length`); the ports' roots distinct, so the port the deployed path check finds by its root is
  the row's own, of the row's format; and every row shorter than `2^64` bits, `RowLeaf.accepts`'s `u64` length.
* `Live.OpensReady.domain`: **every port's frame-v3 domain is 64 bytes**, the width of `Hash.Tree.Domain.id`, at files
  that pass the ports and roots check: the domain is the one the public file publishes (`Flock.Bridge.domOf`), and the
  check refuses a file whose published domain is not 64 bytes (`Flock.Bridge.idsSized_of_checksStmt`).
* `rowMsg_digest`: a `sha512/row/v2` or `sha512/row-seg/v1` port's row message (`Flock.Bridge.rowMsg`) hashes to its
  registered schema's `Hash.RowLeaf` digest (`Flock.Bridge.Spec.formatOf`).
* `Live.opens_rowLeaf`: **an opening the deployed commitment accepts is a `Hash.RowLeaf` opening at `placeAt`** under
  SHA-512, whose row holds the opened bit at the place's offset.
* `Instance.opensAt_of`: **`OpensAt`**, wherever each session v2 sets up is `OpensReady` and the base places each
  non-constant gate at the session's `placeAt` of its claims position. The committed files' circuit is Boolean
  (`Circuit.ofAudit`), so the opened bit is the gate's whole value. `Instance.opensAt_of_ready`: the same, for a base
  whose place is `placeOfRead` at its reading of the files, at sessions `OpensReady` alone; `Instance.opensAt_of_sessions`
  at every instance, whose base's place is `placeOfRead` by construction (`Instance.place`).
* `Instance.SessionsReady`: every session v2 sets up is `OpensReady`.
* `remote_soundness_deployedReady_honest`, `remote_soundness_deployedReady_nontrivial`: its premises hold at the idle
  deployment's instance that runs no session, and its conclusion fails for a verifier that accepts everything.
* `remote_soundness_deployedReady`: **`remote_soundness_deployedAt` with `OpensAt` discharged** at `SessionsReady`,
  and `SessionAt` by `sessionAt_of_registered` at the deployment's session on the registered files.
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

/-- **A `sha512/row/v2` or `sha512/row-seg/v1` port's row message hashes to its registered schema's `Hash.RowLeaf`
digest** under SHA-512. -/
theorem rowMsg_digest {p : Flock.Port} (hp : (p.v2 || p.segs.isSome) = true) (bits : List Bool) :
    Verity.sha512 (Flock.Bridge.rowMsg p bits) =
      (Flock.Bridge.Spec.formatOf (Flock.Registered.portSchema p)).digest Verity.sha512 bits := by
  unfold Flock.Bridge.rowMsg Flock.Registered.portSchema
  by_cases hs : p.segs.isSome = true
  · rw [ite_eq_left hs, ite_eq_left hs, Flock.Bridge.Spec.formatOf_seg, Hash.RowLeaf.Format.digest,
      Hash.RowSeg.digest, Flock.Bridge.Spec.segEnc_eq sha512_hashOf]
  · have hv : p.v2 = true := by simpa [hs] using hp
    rw [ite_eq_right hs, ite_eq_right hs, ite_eq_left hv, ite_eq_left hv, Flock.Bridge.Spec.formatOf_v2,
      Hash.RowLeaf.Format.digest, Hash.RowV2.digest, Flock.Bridge.Spec.enc_eq]

/-- **A `sha512/row/v2` or `sha512/row-seg/v1` port's schema is one `Hash.RowLeaf` has.** -/
theorem portSchema_bits {p : Flock.Port} (hp : (p.v2 || p.segs.isSome) = true) :
    Flock.Registered.portSchema p = Flock.HmRow.ROW_SCHEMA_SEG ∨
      Flock.Registered.portSchema p = Flock.HmRow.ROW_SCHEMA_V2 := by
  unfold Flock.Registered.portSchema
  by_cases hs : p.segs.isSome = true
  · exact Or.inl (ite_eq_left hs)
  · have hv : p.v2 = true := by simpa [hs] using hp
    exact Or.inr ((ite_eq_right hs).trans (ite_eq_left hv))

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

/-- **What the session's files must give for its openings to be `Hash.RowLeaf`'s**: `sha512/row/v2` or
`sha512/row-seg/v1` rows, files that pass `verify --zk`'s ports and roots check (`Flock.PortRoots.checkOf` at the
circuit `Flock.ProvedScope.circuitOf` reads, which refuses a published frame-v3 domain that is not 64 bytes:
`OpensReady.domain`), at least two leaves a port (an accepted path then has a level, which the salt precedes), distinct
roots (the deployed path check opens a row at the first port of the row's root, which is then the row's own), and rows
shorter than `2^64` bits. -/
structure OpensReady : Prop where
  rows : A.D.unroll.ports.all (fun p => p.v2 || p.segs.isSome) = true
  ports : Flock.PortRoots.checkOf
    (Flock.ProvedScope.circuitOf (FlockSoundness.Discharge.ZkExec.zkTags A.I) A.I.circuitFile A.I.tables)
    (FlockSoundness.Discharge.ZkExec.zkTags A.I) A.I.publicFile = .ok ()
  leaves : ∀ q < A.D.unroll.ports.size, 2 ≤ A.Lm.cnt q
  roots : ∀ q q', q' < q → q < A.D.unroll.ports.size → A.roots.getD q' [] ≠ A.roots.getD q []
  width : ∀ i, (Flock.Bridge.placeOf A.Cc A.rows i).1 < 2 ^ 64

/-- **Every port's frame-v3 domain is 64 bytes at files that pass the ports and roots check**: the domain is the one
the public file publishes for the port (`Flock.Bridge.domOf`), `Flock.PortRoots.checkOf` refuses a public file with a
published domain that is not 64 bytes (`Flock.PortRoots.idsSized`), and the session's statement circuit has the ports
and the public file of the circuit the check reads (`Flock.Bridge.idsSized_of_checksStmt`). -/
theorem OpensReady.domain {A : Live} (hA : A.OpensReady) :
    ∀ q, q < A.D.unroll.ports.size → (Flock.Bridge.domOf A.D.unroll A.pub q).size = 64 :=
  Flock.Bridge.idsSized_of_checksStmt ((FlockSoundness.Discharge.ZkExec.zkTags_typed A.I).trans A.ht)
    ((FlockSoundness.Discharge.ZkExec.zkTags_hiddenOutputs A.I).trans A.hTags.1) (Flock.Desc.parseStmt_of_parse A.hD)
    A.hpub hA.ports

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
    have hpr : (A.D.unroll.ports[r.port].v2 || A.D.unroll.ports[r.port].segs.isSome) = true :=
      (Array.all_eq_true.mp hA.rows) r.port hq
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
      have hleaves : reg.leaves = A.Lm.cnt r.port := by rw [← hreg]
      have hidx : Flock.Bridge.portIdxOf A.Cc A.rows i = r.port := by simp [Flock.Bridge.portIdxOf, hr]
      rw [Claims.Sampling.size_bytes, hleaves] at hl
      exact Flock.Bridge.saltOf_length (by simpa [Flock.Bridge.pathOfL, hidx] using hl)
        (Flock.Bridge.PortRoots.one_le_log2ceil (hA.leaves r.port hq))
    have hacc := Flock.Bridge.Spec.opensLeaf_accepts sha512_hashOf hdom hop
    rw [Claims.Sampling.bs_bytes, Flock.Bridge.Spec.sha512Msg_hm96Outer sha512_hashOf, hrow, rowMsg_digest hpr,
      ← hschema] at hacc
    have hfs : (Flock.Bridge.Spec.formatOf reg.schema).schema = reg.schema :=
      Flock.Bridge.Spec.formatOf_schema (by rw [hschema]; exact portSchema_bits hpr)
    have hfit : ∀ ℓ, ℓ < 2 ^ 64 → (Flock.Bridge.Spec.formatOf reg.schema).Fits ℓ := fun ℓ h => by
      unfold Flock.Bridge.Spec.formatOf
      split <;> exact h
    generalize hf : Flock.Bridge.Spec.formatOf reg.schema = f at hacc hfs hfit
    rw [← hfs] at hacc
    refine ⟨{ verifier := { domain := Flock.Bridge.Spec.domainOf reg, key := bs Flock.Hm96.Default512.key,
                            format := f }
              root := Flock.Bridge.Spec.dig reg.root, rank := ri.2,
              offset := (Flock.Bridge.placeOf A.Cc A.rows i).2.val }, ?_,
      ⟨SamplingDeploy.rowBits (Flock.Bridge.placeOf A.Cc A.rows) (Flock.Bridge.bitsOf A.Lm.d) ⟨i, b, m⟩,
        Flock.Bridge.saltOf ⟨i, b, m⟩, _⟩, ⟨?_, hsalt, hacc⟩, ?_⟩
    · simp [placeAt, placeFrom, hat, hfind, hf]
    · refine hfit _ ?_
      simp only [SamplingDeploy.rowBits, List.length_ofFn]
      exact hA.width i
    · simp [SamplingDeploy.rowBits]

end Live

namespace Instance

/-- **Every session v2 sets up is `OpensReady`**: the deployment's files give what `Live.opens_rowLeaf` reads. -/
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
`opensAt_of_sessions`, and its session premise `SessionAt` by `sessionAt_of_registered` from `SessionRegistered`; of
the coupling, `couplingTarget_at`'s, what is left is the device's log opening only the record's rounds
(`LogOpensRecorded`). -/
theorem remote_soundness_deployedReady (I : Instance d α) (K₀ : Constructions I.base)
    (reg : (A : Live) → Option Claims.Msg → A.Reg)
    (record : Live → ByteArray → Array ByteArray → ByteArray → Exchange → Option Claims.Msg)
    (recordDigest : Live → ByteArray → Array ByteArray → ByteArray → Claims.Msg)
    (hready : I.SessionsReady) (hRg : I.RegisteredFiles) (hSr : I.SessionRegistered .remote)
    (hOR : I.LogOpensRecorded .remote) (hW : I.base.PouwSound .remote) (hI : I.base.InferenceOnlySound .remote) :
    (I.vocab (I.constructionsAt K₀ .remote reg record recordDigest)).SoundnessAt .remote :=
  remote_soundness_deployedAt I K₀ reg record recordDigest (I.opensAt_of_sessions hready) hRg
    (I.sessionAt_of_registered .remote hSr) hOR hW hI

open scoped ENNReal in
/-- **`remote_soundness_deployedReady`'s honest lemma**: its premises hold at the idle deployment's instance at v2's
reading (`Instance.Honest.inst`) that runs no session, whose base publishes `⊤` for every subprotocol: each deployment
premise speaks of the sessions the deployment runs, and the base's PoUW and inference-only targets hold at `⊤`. -/
theorem remote_soundness_deployedReady_honest :
    (Instance.Honest.inst ⊤ fun _ => none).SessionsReady ∧ (Instance.Honest.inst ⊤ fun _ => none).RegisteredFiles ∧
      (Instance.Honest.inst ⊤ fun _ => none).SessionRegistered .remote ∧
      (Instance.Honest.inst ⊤ fun _ => none).LogOpensRecorded .remote ∧
      (Instance.Honest.inst ⊤ fun _ => none).base.PouwSound .remote ∧
      (Instance.Honest.inst ⊤ fun _ => none).base.InferenceOnlySound .remote :=
  ⟨fun _ _ _ h => by simp [Instance.Honest.inst] at h, Instance.Honest.registeredFiles ⊤ _,
    Instance.Honest.sessionRegistered_none ⊤ .remote, Instance.Honest.logOpensRecorded_none ⊤ .remote,
    Instance.Honest.target_top .remote .pouw, Instance.Honest.target_top .remote .inferenceOnly⟩

/-- **`remote_soundness_deployedReady`'s nontrivial lemma**: its conclusion, soundness at remote's compiled verifier,
fails for a verifier that accepts everything at the idle deployment whose errors are all `0`. -/
theorem remote_soundness_deployedReady_nontrivial : ¬ Verity.Idle.exact.SoundnessFor .remote fun _ => True :=
  Verity.remote_soundness_of_subprotocols_nontrivial

end Verity.Sampled
