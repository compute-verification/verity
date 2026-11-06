module

public import Properties.SampledProofs.Tables

/-! # Proofs of v2's refusal of blobs holding a lookup table (`Properties.SampledProofs.Tables`)

`tablesHeld` matches on the blobs' tables and throws on any, so its acceptance is their emptiness (`tablesHeld_ok`).
`run` binds it first (`run_tables`), and `checkBlobsAt` accepts only where its chain of refusals ends in `run`'s
acceptance (`checkBlobsAt_run`).

`tablesRefused_honest` is its honest lemma: blobs holding a circuit file and a public file under digests that are no
pinned table's, as the deployment hands v2 (`honestBlobs`), pass the refusal. `tablesRefused_nontrivial` is its
nontrivial lemma: blobs holding a file under the pinned SHA-512 of `rcp` (`tableBlobs`) hold a table, so a check that
accepts every input, as one without the refusal does there, does not have `TablesRefused`'s conclusion.
-/

@[expose] public section

open SampledProofs

namespace Properties.SampledProofs.SecurityProofs

private theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem tablesHeld_ok {bs : Core.Blobs} (h : OneStageZk.tablesHeld bs = .ok ()) : HoldsNoTable bs := by
  unfold OneStageZk.tablesHeld at h
  unfold HoldsNoTable
  split at h
  · assumption
  · cases h

theorem run_tables {p : OneStageZk.Own} {ss : Array OneStageZk.RunFiles} {pu : Option (Array Nat)} {bs : Core.Blobs}
    {r : OneStageZk.Run} (h : OneStageZk.run p ss pu bs = .ok r) : HoldsNoTable bs := by
  unfold OneStageZk.run at h
  obtain ⟨u, hu, -⟩ := bind_ok h
  exact tablesHeld_ok hu

theorem checkBlobsAt_run {p : OneStageZk.Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : OneStageZk.Evidence}
    {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity}
    (h : OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) :
    ∃ r, OneStageZk.run p.own e.sessions (some e.base.provedUnits) bs = .ok r := by
  unfold OneStageZk.checkBlobsAt at h
  dsimp only at h
  split at h
  · cases h
  · rename_i r hr
    obtain ⟨-, -, h1⟩ := bind_ok hr
    obtain ⟨t, -, h2⟩ := bind_ok h1
    obtain ⟨-, -, h3⟩ := bind_ok h2
    obtain ⟨-, -, h4⟩ := bind_ok h3
    obtain ⟨f, -, h5⟩ := bind_ok h4
    obtain ⟨-, -, h6⟩ := bind_ok h5
    exact ⟨r, h6⟩

theorem tablesRefused : TablesRefused :=
  ⟨fun _ => tablesHeld_ok, fun _ _ _ _ _ _ _ h => (checkBlobsAt_run h).elim fun _ hr => run_tables hr⟩

/-- Blobs as the deployment hands v2: a circuit file and a public file, each under a SHA-512 that is no pinned table's. -/
def honestBlobs : Core.Blobs :=
  ⟨#[(⟨Array.replicate 64 0x11⟩, "circuit file".toUTF8), (⟨Array.replicate 64 0x22⟩, "public file".toUTF8)]⟩

/-- **`TablesRefused`'s honest lemma**: blobs holding a circuit file and a public file pass v2's refusal. -/
theorem tablesRefused_honest : OneStageZk.tablesHeld honestBlobs = .ok () := by
  have h : (OneStageZk.tablesOf honestBlobs).isEmpty = true := by decide +kernel
  unfold OneStageZk.tablesHeld
  rw [List.isEmpty_iff.1 h]
  rfl

/-- `rcp`'s pinned SHA-512 (`Flock.MUFU_TABLES`), as bytes. -/
def rcpDigest : ByteArray :=
  ⟨#[0x8a, 0x31, 0x7a, 0xdb, 0xcc, 0x07, 0x8b, 0xf7, 0x0a, 0xe0, 0x19, 0x59, 0xb6, 0x79, 0xd4, 0x9f,
     0x97, 0xcd, 0x9c, 0x8a, 0x93, 0xd6, 0xf3, 0x9b, 0x73, 0x25, 0xc9, 0x3d, 0x0f, 0xd5, 0x50, 0xfa,
     0x43, 0x96, 0xa4, 0xc3, 0xa0, 0x56, 0x8d, 0x23, 0x87, 0xc2, 0xff, 0x2f, 0x33, 0xeb, 0x76, 0x45,
     0x3b, 0x33, 0xa7, 0xb8, 0x89, 0x0d, 0x97, 0x75, 0x2d, 0x86, 0xcc, 0x1d, 0x97, 0x1a, 0xea, 0xcc]⟩

/-- Blobs holding a file under `rcp`'s pinned SHA-512. -/
def tableBlobs : Core.Blobs := ⟨#[(rcpDigest, ByteArray.empty)]⟩

/-- **`TablesRefused`'s nontrivial lemma**: blobs holding a file under a pinned table's SHA-512 hold a table, so a check
that accepts them, as one that accepts every input does, does not have `TablesRefused`'s conclusion. -/
theorem tablesRefused_nontrivial : ¬ HoldsNoTable tableBlobs := by
  have h : (OneStageZk.tablesOf tableBlobs).isEmpty = false := by decide +kernel
  intro hn
  unfold HoldsNoTable at hn
  rw [hn] at h
  cases h

end Properties.SampledProofs.SecurityProofs
