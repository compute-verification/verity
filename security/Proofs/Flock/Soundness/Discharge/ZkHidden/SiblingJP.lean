module

public import Proofs.Flock.Claims.ZK.MaskedTables
public import Proofs.Flock.Soundness.Discharge.Hidden.Sibling
public import Proofs.Flock.Soundness.Discharge.ZkHidden.DefsJP
public import Proofs.Flock.Soundness.Discharge.ZkHidden.Sched
public import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundDrawOk

/-!
# The reference setup of a statement with no session

A `--zk` table's reference setup (`DrawSetupZHJP`) is built from a setup `verify --zk` made. A statement no session
samples has none of its own. `y₀_of_sibling` builds one from a setup at the sample of no unit on another public file
of the same circuit file, tags, coins, tables, partition and program (`Hidden.setupHidden_noUnit`): at one table, the
statement's public file loaded and held to its roots, its header naming no `public_inputs`, sets up a `--zk` table at
the sample of no unit, at `mPtsOf g 0` points.

A reference setup is the one value of its type (`drawSetupZHJP_subsingleton`): its statement is what
`Stmt.setupHidden` returns, its sample what `Draw.UnitDraw.ofJson` decodes, its schedules what `fast100` and
`Zk.schedule` return, and the rest are proofs.
-/

@[expose] public section

namespace FlockSoundness.Discharge.HiddenExec

open Flock
open FlockSoundness.Discharge.Exec (Inputs)

/-- **Two setups of one sample at one statement's inputs are one**: the statement, the decoded sample and the
schedule are each a function's value, and every other field is a proof. -/
theorem drawSetupHP_eq {n : ℕ} {I : Inputs} {own : Option ByteArray} {mPts : ℕ} {S : Finset (Fin n)}
    {d : Lean.Json} : ∀ x x' : DrawSetupHP I own mPts S d, x = x'
  | ⟨st, h, D, hD, _, _, _, m, k0, l₀, l₁, rest, ext, hsch, _, _, _⟩,
    ⟨st', h', D', hD', _, _, _, m', k0', l₀', l₁', rest', ext', hsch', _, _, _⟩ => by
    obtain rfl : st = st' := Except.ok.inj (h.symm.trans h')
    obtain rfl : D = D' := Except.ok.inj (hD.symm.trans hD')
    cases Option.some.inj (hsch.symm.trans hsch')
    rfl

end FlockSoundness.Discharge.HiddenExec

namespace FlockSoundness.Discharge.ZkHidden

open Flock Exec ZkExec Composed HiddenExec
open Lean (Json)

/-- **A reference setup is the one value of its type**: at given inputs, public inputs of the verifier's own, link
points, sampled proof blocks and sample. -/
theorem drawSetupZHJP_subsingleton {n : ℕ} (I' : Inputs) (own : Option ByteArray) (mPts : ℕ) (S : Finset (Fin n))
    (d : Json) : Subsingleton (DrawSetupZHJP I' own mPts S d) :=
  ⟨fun y y' => by
    obtain ⟨x, sch, tPad, hz, _, _, _, _, _, _, _⟩ := y
    obtain ⟨x', sch', tPad', hz', _, _, _, _, _, _, _⟩ := y'
    obtain rfl : x = x' := drawSetupHP_eq x x'
    cases Except.ok.inj (hz.symm.trans hz')
    rfl⟩

/-- **The reference setup of a statement with no session**: at one table, a setup `verify --zk` makes at a sample
of no unit on the public file `pf₁` (`hst₁`, `hn₁`), under the statement's own tags, circuit file, coins, tables,
partition and program, that passes the `--zk` shape check, gives the statement a `--zk` table at the sample of no unit
(`Hidden.emptyDraw`) and `mPtsOf g 0` points, when the statement's public file loads for that setup's circuit, is held
to its roots (`HmOut.checkPublic`) and names no `public_inputs` in its header. -/
theorem y₀_of_sibling (n : ℕ) {I : Inputs} {D : Flock.Desc} {pf₁ : ByteArray} {d₁ : Json} {st₁ : Flock.Stmt}
    {pop : Flock.Public} (hc1 : I.count = 1) (ht : I.tags.typed = false)
    (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D)
    (hst₁ : Flock.Stmt.setupHidden (inputsJ I 0).tags (inputsJ I 0).circuitFile pf₁ (inputsJ I 0).coins
      (inputsJ I 0).tables (some d₁) (inputsJ I 0).partition (inputsJ I 0).program none = .ok st₁)
    (hn₁ : st₁.n = 0) (hshape₁ : Flock.Zk.shapeOk st₁ = .ok ())
    (hpop : Flock.HmOut.loadPublic (zkTags I) st₁.c I.publicFile = .ok pop)
    (hkey : (pop.header.getObjVal? "public_inputs").toOption = none)
    (hpub : Flock.HmOut.checkPublic st₁.c pop = .ok ()) :
    ∃ (S₀ : Finset (Fin n)) (d₀ : Json),
      Nonempty (DrawSetupZHJP (inputsJ I 0) none (Exec.mPtsOf D.unroll.g (0 / nTab I)) S₀ d₀) := by
  have htJ : (inputsJ I 0).tags = zkTags I := by
    show Exec.tagsAt _ I.count 0 = _
    unfold Exec.tagsAt
    exact ite_eq_left (by omega)
  obtain ⟨st₂, hst₂, -, hsh⟩ := Hidden.setupHidden_noUnit (pf₂ := I.publicFile) hst₁ hn₁
    (by rw [htJ]; exact hpop) hkey hpub
  have hshape : Flock.Zk.shapeOk st₂ = .ok () := hsh.trans hshape₁
  obtain ⟨⟨sch, tPad⟩, hz⟩ := schedule_ok_of_shapeOk hshape
  obtain ⟨sch₀, hf⟩ := fast100_of_schedule hz
  have hst' := hst₂
  rw [htJ] at hst'
  have hmPts : st₂.mPts = Exec.mPtsOf D.unroll.g (0 / nTab I) := by
    rw [Nat.zero_div]
    exact setupHidden_mPts_of_proved (Flock.Desc.parse_flat hD)
      (fun D' hD' => by
        obtain rfl := Except.ok.inj ((Hidden.ofJson_emptyDraw pop.n).symm.trans hD')
        rfl) hst'
  obtain ⟨xx, hxx⟩ := drawSetupHP_of_setupHiddenI (I := inputsJ I 0) (own := none) (S := (∅ : Finset (Fin n)))
    (typedJ ht 0) hst₂ hf (Hidden.ofJson_emptyDraw pop.n) (fun v hv => by simp at hv) hmPts
  obtain ⟨y, -, -, -⟩ := drawSetupZHJP_of xx (by rw [hxx]; exact hshape) (sch := sch) (tPad := tPad)
    (by rw [hxx]; exact hz)
  exact ⟨∅, _, ⟨y⟩⟩

end FlockSoundness.Discharge.ZkHidden
