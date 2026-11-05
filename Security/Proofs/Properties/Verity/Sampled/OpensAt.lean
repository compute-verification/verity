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
* `Live.OpensReady`: what the session's files must give: every port a `sha512/row/v2` row (`RowsV2`); every port's
  frame-v3 domain 64 bytes, the width of `Hash.Tree.Domain.id`; every port at least two leaves, so an accepted path
  reaches past the salt and the salt is `saltLen` bytes, as `Hash.RowLeaf.accepts` requires; and every row shorter than
  `2^64` bits, `RowLeaf.accepts`'s `u64` length.
* `Live.opens_rowLeaf`: **an opening the deployed commitment accepts is a `Hash.RowLeaf` opening at `placeAt`** under
  SHA-512, whose row holds the opened bit at the place's offset.
* `Instance.opensAt_of`: **`OpensAt`**, wherever each session v2 sets up is `OpensReady` and the base places each
  non-constant gate at the session's `placeAt` of its claims position. The committed files' circuit is Boolean
  (`Circuit.ofAudit`), so the opened bit is the gate's whole value. `Instance.opensAt_of_ready`: the same, for a base
  whose place is `placeOfRead` at its reading of the files, at sessions `OpensReady` alone.
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

theorem log2ceil_pos {n : ℕ} (h : 2 ≤ n) : 1 ≤ Flock.log2ceil n := by
  unfold Flock.log2ceil
  rw [ite_eq_right (show ¬ n ≤ 1 by omega)]
  omega

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

/-- **What the session's files must give for its openings to be `Hash.RowLeaf`'s**: `sha512/row/v2` rows, 64-byte
frame-v3 domains, at least two leaves a port (an accepted path then reaches past the salt), and rows shorter than
`2^64` bits. -/
structure OpensReady : Prop where
  v2 : Flock.Bridge.RowsV2 A.D.unroll
  domain : ∀ q < A.D.unroll.ports.size, (Flock.Bridge.domOf A.D.unroll q).size = 64
  leaves : ∀ q < A.D.unroll.ports.size, 2 ≤ A.Lm.cnt q
  width : ∀ i, (Flock.Bridge.placeOf A.Cc A.rows i).1 < 2 ^ 64

/-- **An opening the deployed commitment accepts is a `Hash.RowLeaf` opening at the registered place**: under
SHA-512, at `placeAt`'s verifier, root and rank, its row holding the opened bit at the place's offset. -/
theorem opens_rowLeaf {A : Live} (hA : A.OpensReady) {i : ℕ} {b : Bool} {m : Claims.Msg}
    (h : A.Cd.Opens Claims.Sampling.sha512Msg A.roots ⟨i, b, m⟩ = true) :
    ∃ p, A.placeAt i = some p ∧ ∃ o : Hash.RowLeaf.Opening,
      Hash.RowLeaf.accepts Verity.sha512 Verity.sha512 p.verifier p.root p.rank o ∧ o.row[p.offset]? = some b := by
  unfold Live.Cd at h
  rw [Flock.Bridge.commitDepL_v2 _ _ hA.v2] at h
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
    have hdom : reg.domain.size = 64 := by rw [← hreg]; exact hA.domain q hq
    have hleaves : reg.leaves = A.Lm.cnt q := by rw [← hreg]
    have hschema : reg.schema = Flock.HmRow.ROW_SCHEMA_V2 := by
      rw [← hreg]; exact Flock.Bridge.schemaOf_v2 hA.v2 hq
    have hacc := Flock.Bridge.Spec.opensLeaf_accepts sha512_hashOf hdom hop
    rw [Claims.Sampling.bs_bytes, Flock.Bridge.Spec.sha512Msg_hm96Outer sha512_hashOf, hschema,
      show SamplingDeploy.row (Flock.Bridge.placeOf A.Cc A.rows) (Flock.Bridge.bitsOf A.Lm.d) ⟨i, b, m⟩ =
        Hash.RowV2.msg (SamplingDeploy.rowBits (Flock.Bridge.placeOf A.Cc A.rows) (Flock.Bridge.bitsOf A.Lm.d)
          ⟨i, b, m⟩) from Flock.Bridge.Spec.enc_eq _] at hacc
    simp only [Flock.Registered.opensLeaf, Bool.and_eq_true, decide_eq_true_eq, Claims.Sampling.size_bytes] at hop
    obtain ⟨⟨-, hlen⟩, -⟩ := hop
    have hsalt : (Flock.Bridge.saltOf ⟨i, b, m⟩).length = Hash.Hm96.SALT_BYTES := by
      have h1 : 1 ≤ Flock.log2ceil reg.leaves := by rw [hleaves]; exact log2ceil_pos (hA.leaves q hq)
      simp only [Flock.Bridge.pathOfL, Flock.Bridge.pathOf, Flock.Bridge.saltLen, List.length_take,
        List.length_drop] at hlen
      simp only [Flock.Bridge.saltOf, List.length_take, Flock.Bridge.saltLen, Hash.Hm96.SALT_BYTES]
      omega
    refine ⟨{ verifier := { domain := Flock.Bridge.Spec.domainOf reg, key := bs Flock.Hm96.Default512.key,
                            format := .v2 }
              root := Flock.Bridge.Spec.dig reg.root, rank := ri.2,
              offset := (Flock.Bridge.placeOf A.Cc A.rows i).2.val }, ?_,
      ⟨SamplingDeploy.rowBits (Flock.Bridge.placeOf A.Cc A.rows) (Flock.Bridge.bitsOf A.Lm.d) ⟨i, b, m⟩,
        Flock.Bridge.saltOf ⟨i, b, m⟩, _⟩, ⟨?_, hsalt, hacc⟩, ?_⟩
    · simp [placeAt, placeFrom, hat, hfind]
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

end Instance

end Verity.Sampled
