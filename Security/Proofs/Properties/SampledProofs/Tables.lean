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

/-- **`TablesRefused`'s honest lemma**: blobs holding a circuit file and a public file pass v2's refusal. -/
theorem tablesRefused_honest : OneStageZk.tablesHeld honestBlobs = .ok () := by
  have h : (OneStageZk.tablesOf honestBlobs).isEmpty = true := by decide +kernel
  unfold OneStageZk.tablesHeld
  rw [List.isEmpty_iff.1 h]
  rfl

/-- **`TablesRefused`'s nontrivial lemma**: blobs holding a file under a pinned table's SHA-512 hold a table, so a check
that accepts them, as one that accepts every input does, does not have `TablesRefused`'s conclusion. -/
theorem tablesRefused_nontrivial : ¬ HoldsNoTable tableBlobs := by
  have h : (OneStageZk.tablesOf tableBlobs).isEmpty = false := by decide +kernel
  intro hn
  unfold HoldsNoTable at hn
  rw [hn] at h
  cases h

end Properties.SampledProofs.SecurityProofs
