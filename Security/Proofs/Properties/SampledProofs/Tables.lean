module

public import Properties.SampledProofs.Tables

/-! # Proofs of v2's holding the blobs' lookup tables to the registered circuit's (`Properties.SampledProofs.Tables`)

`tablesHeld` reads the circuit file, its lookups and the blobs' tables, and throws where the two lists of names differ,
so its acceptance is their equality (`tablesHeld_ok`). `run` binds it first (`run_tables`), and `checkBlobsAt` accepts
only where its chain of refusals ends in `run`'s acceptance (`checkBlobsAt_run`).

`tablesRegistered_honest` is its honest lemma: blobs holding a circuit file that looks up `rcp` and `rcp`'s table, as
the served path hands v2 (`honestBlobs`), pass the refusal. `tablesRegistered_nontrivial` is its nontrivial lemma: blobs
holding a circuit file that looks up no table and `rcp`'s table (`tableBlobs`) hold a table the circuit doesn't look
up, so a check that accepts every input, as one without the refusal does there, does not have `TablesRegistered`'s
conclusion.
-/

@[expose] public section

open SampledProofs

namespace Properties.SampledProofs.SecurityProofs

private theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem tablesHeld_ok {p : OneStageZk.Own} {bs : Core.Blobs} (h : OneStageZk.tablesHeld p bs = .ok ()) :
    HoldsRegisteredTables p bs := by
  unfold OneStageZk.tablesHeld at h
  obtain ⟨c, hc, h1⟩ := bind_ok h
  obtain ⟨w, hw, h2⟩ := bind_ok h1
  refine ⟨c, hc, ?_⟩
  rw [hw]
  dsimp only at h2
  split at h2
  · cases h2
  · rename_i hne
    simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hne
    rw [hne]

theorem run_tables {p : OneStageZk.Own} {ss : Array OneStageZk.RunFiles} {pu : Option (Array Nat)} {bs : Core.Blobs}
    {r : OneStageZk.Run} (h : OneStageZk.run p ss pu bs = .ok r) : HoldsRegisteredTables p bs := by
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

theorem tablesRegistered : TablesRegistered :=
  ⟨fun _ _ => tablesHeld_ok, fun _ _ _ _ _ _ _ h => (checkBlobsAt_run h).elim fun _ hr => run_tables hr⟩

/-- **`TablesRegistered`'s honest lemma**: blobs holding a circuit file that looks up `rcp` and `rcp`'s table pass v2's
refusal. -/
theorem tablesRegistered_honest : OneStageZk.tablesHeld witnessOwn honestBlobs = .ok () := by
  have h : (OneStageZk.tablesHeld witnessOwn honestBlobs).toBool = true := by decide +kernel
  generalize OneStageZk.tablesHeld witnessOwn honestBlobs = x at h ⊢
  cases x with
  | error e => cases h
  | ok u => rfl

/-- `HoldsRegisteredTables` as a computation. -/
private def holdsB (p : OneStageZk.Own) (bs : Core.Blobs) : Bool :=
  match bs.get p.circuit "circuit" with
  | .ok c => match OneStageZk.lookupsOf c with
    | .ok w => w == (OneStageZk.tablesOf bs).map (·.1)
    | .error _ => false
  | .error _ => false

private theorem holdsB_of {p : OneStageZk.Own} {bs : Core.Blobs} (h : HoldsRegisteredTables p bs) : holdsB p bs = true := by
  obtain ⟨c, hc, hl⟩ := h
  unfold holdsB
  simp only [hc, hl, beq_self_eq_true]

/-- **`TablesRegistered`'s nontrivial lemma**: blobs holding a circuit file that looks up no table and `rcp`'s table
hold a table the circuit doesn't look up, so a check that accepts them, as one that accepts every input does, does not
have `TablesRegistered`'s conclusion. -/
theorem tablesRegistered_nontrivial : ¬ HoldsRegisteredTables witnessOwn tableBlobs := by
  intro h
  have hf : holdsB witnessOwn tableBlobs = false := by decide +kernel
  rw [holdsB_of h] at hf
  cases hf

end Properties.SampledProofs.SecurityProofs
