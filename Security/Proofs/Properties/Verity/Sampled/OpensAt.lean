module

public import Proofs.Properties.Verity.Sampled.Instance
public import Proofs.Properties.Verity.Sampled.Place

/-!
# The deployed commitment's openings are `Hash.RowLeaf`'s

`Instance.OpensAt` asks that wherever a session's deployed commitment (`Live.Cd`, C-Flock's `commitDepL`) opens a
gate, the base's openings hold of it (`Vocab.Opens`: a `Hash.RowLeaf` opening under SHA-512 at the base's place). Here
it is from the code's side (`Flock.Bridge.Spec`), at:

* `Live.placeAt`: **the place the session registers for a claims position**, `placeFrom` (`Sampled/Place.lean`) at
  the session's commitment (`Live.Cd`'s layout). `Live.placeOfRead_eq`: wherever v2 reads a session's files, the
  committed files' place of a gate (`placeOfRead`) is the session's `placeAt` of its claims position.
* `Live.OpensReady`: what the session's files must give: every port a `sha512/row/v2` or `sha512/row-seg/v1` row,
  the two formats `Hash.RowLeaf` has; every port's frame-v3 domain 64 bytes, the width of `Hash.Tree.Domain.id`; the
  ports' roots distinct, so the port the deployed path check finds by its root is the row's own, of the row's format;
  and every row shorter than `2^64` bits, `RowLeaf.accepts`'s `u64` length. The salt is `saltLen` bytes at every
  port, a one-leaf port's included (`Flock.Bridge.saltOf_length`).
* `rowMsg_digest`: a `sha512/row/v2` or `sha512/row-seg/v1` port's row message (`Flock.Bridge.rowMsg`) hashes to its
  registered schema's `Hash.RowLeaf` digest (`Flock.Bridge.Spec.formatOf`).
* `Live.opens_rowLeaf`: **an opening the deployed commitment accepts is a `Hash.RowLeaf` opening at `placeAt`** under
  SHA-512, whose row holds the opened bit at the place's offset.
* `Instance.opensAt_of`: **`OpensAt`**, wherever each session v2 sets up is `OpensReady` and the base places each
  non-constant gate at the session's `placeAt` of its claims position. The committed files' circuit is Boolean
  (`Circuit.ofAudit`), so the opened bit is the gate's whole value. `Instance.opensAt_of_ready`: the same, for a base
  whose place is `placeOfRead` at its reading of the files, at sessions `OpensReady` alone; `Instance.opensAt_of_sessions`
  at every instance, whose base's place is `placeOfRead` by construction (`Instance.place`).
* `remote_soundness_deployedReady`: **`remote_soundness_deployedAt` with `OpensAt` discharged** at sessions
  `OpensReady`.
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

theorem w_ofAudit {K : Verity.Circuit} {C : FlockSoundness.Audit.Circuit} (h : K = Verity.Circuit.ofAudit C)
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
`sha512/row-seg/v1` rows, 64-byte frame-v3 domains, distinct roots (the deployed path check opens a row at the first
port of the row's root, which is then the row's own), and rows shorter than `2^64` bits. -/
structure OpensReady : Prop where
  rows : A.D.unroll.ports.all (fun p => p.v2 || p.segs.isSome) = true
  domain : ∀ q < A.D.unroll.ports.size, (Flock.Bridge.domOf A.D.unroll A.pub q).size = 64
  roots : ∀ q q', q' < q → q < A.D.unroll.ports.size → A.roots.getD q' [] ≠ A.roots.getD q []
  width : ∀ i, (Flock.Bridge.placeOf A.Cc A.rows i).1 < 2 ^ 64

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
    have hacc := Flock.Bridge.Spec.opensLeaf_accepts sha512_hashOf hdom hop
    rw [Claims.Sampling.bs_bytes, Flock.Bridge.Spec.sha512Msg_hm96Outer sha512_hashOf, hrow, rowMsg_digest hpr,
      ← hschema] at hacc
    have hfs : (Flock.Bridge.Spec.formatOf reg.schema).schema = reg.schema :=
      Flock.Bridge.Spec.formatOf_schema (by rw [hschema]; exact portSchema_bits hpr)
    generalize hf : Flock.Bridge.Spec.formatOf reg.schema = f at hacc hfs
    rw [← hfs] at hacc
    have hsalt : (Flock.Bridge.saltOf ⟨i, b, m⟩).length = Hash.Hm96.SALT_BYTES := Flock.Bridge.saltOf_length _
    refine ⟨{ verifier := { domain := Flock.Bridge.Spec.domainOf reg, key := bs Flock.Hm96.Default512.key,
                            format := f }
              root := Flock.Bridge.Spec.dig reg.root, rank := ri.2,
              offset := (Flock.Bridge.placeOf A.Cc A.rows i).2.val }, ?_,
      ⟨SamplingDeploy.rowBits (Flock.Bridge.placeOf A.Cc A.rows) (Flock.Bridge.bitsOf A.Lm.d) ⟨i, b, m⟩,
        Flock.Bridge.saltOf ⟨i, b, m⟩, _⟩, ⟨?_, hsalt, hacc⟩, ?_⟩
    · simp [placeAt, placeFrom, hat, hfind, hf]
    · simp only [SamplingDeploy.rowBits, List.length_ofFn]
      exact hA.width i
    · simp [SamplingDeploy.rowBits]

end Live

namespace Instance

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
    (h : ∀ θ F A, I.session θ = some A → SetsUp I.base θ F A → A.OpensReady) : I.OpensAt :=
  I.opensAt_of fun θ F A hs hsu => ⟨h θ F A hs hsu, fun c hN g _ =>
    (hplace θ F c g).trans (A.placeOfRead_eq hsu.read hN g)⟩

/-- **`OpensAt` at sessions `OpensReady`**: an instance's base places each gate at the committed files' place by its
construction (`Instance.place`), so `OpensAt` needs only that each session v2 sets up is `OpensReady`. -/
theorem opensAt_of_sessions (I : Instance d α)
    (h : ∀ θ F A, I.session θ = some A → SetsUp I.base θ F A → A.OpensReady) : I.OpensAt :=
  I.opensAt_of_ready I.place h

end Instance

/-- **`remote_soundness_deployedAt` at sessions `OpensReady`**: its openings premise `OpensAt` discharged by
`opensAt_of_sessions`. -/
theorem remote_soundness_deployedReady (I : Instance d α) (K₀ : Constructions I.base)
    (reg : (A : Live) → Option Claims.Msg → A.Reg)
    (record : Live → ByteArray → Array ByteArray → ByteArray → Exchange → Option Claims.Msg)
    (recordDigest : Live → ByteArray → Array ByteArray → ByteArray → Claims.Msg)
    (hready : ∀ θ F A, I.session θ = some A → SetsUp I.base θ F A → A.OpensReady)
    (hRg : I.RegisteredFiles) (hSa : I.SessionAt .remote) (hε : I.StopsWithin .remote reg record recordDigest)
    (hL : ∀ θ F A, I.sessions θ F = some A → A.Lb 0 = _root_.SampledProofs.Challenger.MAX_COINS * 64)
    (hord : ∀ θ F c (σ : Verity.Game.Strategy (I.base.game θ F c)) (x : Verity.Game.CoinTape (I.base.game θ F c) σ),
      I.base.Verdict .remote .sampled (Verity.Game.CoinTape.run (I.base.game θ F c) σ x) →
      OrderHeld (Verity.Game.CoinTape.run (I.base.game θ F c) σ x).sent)
    (hOR : ∀ θ F c (σ : Verity.Game.Strategy (I.base.game θ F c)) (x : Verity.Game.CoinTape (I.base.game θ F c) σ),
      I.base.Verdict .remote .sampled (Verity.Game.CoinTape.run (I.base.game θ F c) σ x) →
      Binding.Logged Binding.readsV2 (Verity.Game.CoinTape.run (I.base.game θ F c) σ x) →
      ∀ text Jr, sentRecord (Verity.Game.CoinTape.run (I.base.game θ F c) σ x).sent = some text →
        Flock.JsonParse.parse text = .ok Jr →
        OpenedRecorded Jr (Verity.Game.CoinTape.run (I.base.game θ F c) σ x).log)
    (hB : UnboundFaulty) (hR : RecursiveForm (I.vocab (I.constructionsAt K₀ .remote reg record recordDigest)) .remote)
    (hW : I.base.PouwSound .remote) (hI : I.base.InferenceOnlySound .remote) :
    (I.vocab (I.constructionsAt K₀ .remote reg record recordDigest)).SoundnessAt .remote :=
  remote_soundness_deployedAt I K₀ reg record recordDigest (I.opensAt_of_sessions hready) hRg hSa hε hL hord hOR hB
    hR hW hI

end Verity.Sampled
