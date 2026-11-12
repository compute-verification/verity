module

public import Proofs.Properties.Verity.Sampled.SessionOf
public import Proofs.Properties.Verity.SampledRead
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
public import Proofs.Flock.Soundness.Discharge.Hidden.PortsP
public import Proofs.Flock.Soundness.Discharge.Hidden.Spec
public import Proofs.Flock.Soundness.Discharge.ZkHidden.NonVacuousJ
public import Proofs.Properties.Verity.Sampled.Deployed
public import Proofs.Properties.Verity.Sampled.Held
public import Proofs.Properties.Verity.Sampled.Moves
import Proofs.Flock.Soundness.Refine.Walk
import Proofs.Flock.Soundness.Refine.Setup

/-!
# An accepted play on the registered files passes the checks `sessionOf` reads (`AcceptedChecks`)

`sessionOf` builds the session from the parameters and the registered files under `SessionChecks`. Where v2's
sampled verdict accepts a play whose blobs are the registered files, then the evidence's, `SessionChecks` holds of those
files (`sessionChecks_of_verdict`), but for two facts about the parameters, which a play does not check:

* the descriptor of the verifier's program (`hdesc`);
* the registration's law gives a plain subset sampling design wherever `lawAt` reads one (`hsub`).

The reference table is the play's own first table: the record's sample file is the subset print of the units v2
drew (`drawHeld`), and `verify --zk` sets every table up there (`drawHypsZHJ`).

The pieces:

* **The form check parses the circuit** (`ksCheck_parsed`): `KsScope.checkOf` refuses a circuit that does not parse
  and a public file that does not load.
* **Without public input ports, `HmIn.parse` reads nothing** (`parse_empty`): its one return without ports is `{}`.
* **An accepted session names no public input port** (`noInputs_of_accepts`): at one table, `setupTables` is one
  `setupHidden` (`HiddenExec.setupTables_okH`), which at no recomputation of public input words passes `HmIn.checkOwn`
  only without ports (`Hidden.checkOwn_none`), so `HmIn.parse` reads `{}` (`parse_empty`).
* **The tables the blobs hold are the registered ones** (`pinnedTables_eq`): where the circuit file looks up exactly
  the pinned tables the blobs hold and each is its registered file, the two lists are equal, not only one extending
  the other.
* **v2's inputs on the blobs are `inputsAt`'s** (`inputsOf_eq`) at those tables.
* **A `DrawSetupZHJ` is a `DrawSetupZHJP` at `none`** (`setupZHJP`).
* **`Accepted` passes the checks** (`Accepted.checksWitness`).
-/

@[expose] public section

namespace Verity.Sampled

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Exec FlockSoundness.Discharge.ZkExec
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.ZkLink
open FlockSoundness.Discharge.Recursion.Witness (toExec)
open FlockSoundness.Refine (throw_bind' mufu_nodup)

/-- **The form check parses the circuit and loads the public file**: `KsScope.checkOf` refuses a circuit that does
not parse and a public file that does not load. -/
theorem ksCheck_parsed {parsed : Flock.V Flock.Circuit} {tags : Flock.Tags} {pf : ByteArray} {count : ℕ}
    {draw : Option Lean.Json} (h : Flock.KsScope.checkOf parsed tags pf count draw = .ok ()) :
    ∃ c, parsed = .ok c ∧ ∃ pub, Flock.HmOut.loadPublic tags c pf = .ok pub := by
  unfold Flock.KsScope.checkOf at h
  split at h
  · cases h
  · cases parsed with
    | error e => cases h
    | ok c =>
      cases hl : Flock.HmOut.loadPublic tags c pf with
      | error e =>
        dsimp only at h
        rw [hl] at h
        cases h
      | ok pub => exact ⟨c, rfl, pub, rfl⟩

-- `walk_step` down `HmIn.parse`'s unfolded do-block, a goal per branch
set_option maxHeartbeats 4000000 in
/-- **Without public input ports, `HmIn.parse` reads nothing**: it returns ports only where META names
`public_inputs`, and then at least one. -/
theorem parse_empty {c : Flock.Circuit} {pi : Flock.HmIn.Ports} (h : Flock.HmIn.parse c = .ok pi)
    (he : pi.ports.isEmpty = true) : pi = {} := by
  unfold Flock.HmIn.parse at h
  dsimp only at h
  cases hj : c.metaJ.getObjVal? "public_inputs" with
  | error e =>
    rw [hj] at h
    try dsimp only at h
    try simp only [throw_bind'] at h
    repeat' walk_step
    all_goals
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      rfl
  | ok pj =>
    rw [hj] at h
    try dsimp only at h
    try simp only [throw_bind'] at h
    repeat' walk_step
    all_goals
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      exfalso
      have hps := ‹¬ (Array.isEmpty (α := Lean.Json) _) = true›
      have hm := Hidden.mapM_size ‹Array.mapM _ _ = Except.ok _›
      try dsimp only at he
      apply hps
      rw [Array.isEmpty_iff_size_eq_zero] at he ⊢
      omega

/-- **An accepted session at one table names no public input port**: flock-verify `--zk` sets its one table up with
`setupHidden` at no recomputation of public input words, which passes `HmIn.checkOwn` only without public input ports;
and the public file loads at the circuit's empty input ports. -/
theorem noInputs_of_accepts {I : Exec.Inputs} {t : Exec.Transcript} (hA : AcceptsZK I t)
    (hflat : (zkTags I).hiddenOutputs = true ∧ (zkTags I).isTmpl = false) (hcount : I.count = 1)
    {c : Flock.Circuit} (hc : Flock.HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) :
    Flock.HmIn.parse c = .ok {} ∧ ∃ pub, Flock.HmIn.loadPublic (zkTags I) c {} I.publicFile = .ok pub := by
  obtain ⟨spec, sts, hs, -, -⟩ := hA
  unfold Exec.setupOf at hs
  rcases HiddenExec.setupTables_okH hflat.1 hs with ⟨-, st, hst, -, -⟩ | ⟨h1, -⟩
  · obtain ⟨pi, s⟩ := Hidden.setupHidden_specP hst
    have hpc : Flock.HmOut.parse (zkTags I) I.circuitFile I.tables = .ok st.c := s.parseF hflat.2
    obtain rfl : st.c = c := Except.ok.inj (hpc.symm.trans hc)
    obtain ⟨pop, hl, hown, -, -⟩ := s.pub
    have hpi := s.ports
    obtain rfl : pi = {} := parse_empty hpi (Hidden.checkOwn_none hown)
    exact ⟨hpi, pop, hl⟩
  · have h1' : (zkInputs I).count = 1 := hcount
    omega

/-- **The tables the blobs sent hold are the ones the circuit file registers**: where it looks up exactly the pinned
tables the blobs hold (`Spot.OneStageZk.lookupsOf`) and each the blobs hold is its registered file (`TablesSent`). -/
theorem pinnedTables_eq {s : Verity.Sent} {circuitB : ByteArray} {names : List String}
    (hl : _root_.Spot.OneStageZk.lookupsOf circuitB = .ok names)
    (hn : names = (_root_.Spot.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray s.blobs)).map (·.1))
    (ht : TablesSent s) :
    pinnedTables circuitB = _root_.Spot.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray s.blobs) := by
  classical
  simp only [pinnedTables, hl]
  rw [tablesOf_ofArray]
  refine List.filterMap_congr fun t htm => ?_
  cases hf : tableFile t.sha512 s.blobs with
  | none =>
    have hc : ¬ names.contains t.name = true := by
      intro hc
      have hm : t.name ∈ names := by simpa using hc
      rw [hn, tablesOf_ofArray] at hm
      obtain ⟨y, hy, hyn⟩ := List.mem_map.1 hm
      obtain ⟨t', ht', hft'⟩ := List.mem_filterMap.1 hy
      obtain ⟨a, ha, rfl⟩ := Option.map_eq_some_iff.1 hft'
      have htt : t' = t := List.inj_on_of_nodup_map mufu_nodup ht' htm hyn
      subst htt
      rw [hf] at ha
      cases ha
    rw [if_neg hc] <;> rfl
  | some a =>
    have hm : t.name ∈ names := by
      rw [hn, tablesOf_ofArray]
      exact List.mem_map.2 ⟨(t.name, a), List.mem_filterMap.2 ⟨t, htm, by simp [hf]⟩, rfl⟩
    have hc : names.contains t.name = true := by simpa using hm
    obtain ⟨b, hb⟩ : ∃ b, pinnedFile t.sha512 = some b := by
      unfold pinnedFile
      rw [dif_pos ⟨a, tableFile_hash hf⟩]
      exact ⟨_, rfl⟩
    rw [if_pos hc, hb, ← ht t htm a b hf hb] <;> rfl

/-- **v2's inputs to `verify --zk` on the blobs are `inputsAt`'s** where the tables the blobs hold are the ones the
circuit file registers. -/
theorem inputsOf_eq {p : _root_.Spot.OneStageZk.Params} {cf pf : ByteArray} {bs : _root_.Core.Blobs}
    (ht : _root_.Spot.OneStageZk.tablesOf bs = pinnedTables cf) :
    toExec (_root_.Spot.OneStageZk.inputsOf Flock.Tags.circuit p.own cf pf bs) = inputsAt p cf pf := by
  unfold _root_.Spot.OneStageZk.inputsOf inputsAt
  rw [ht]

theorem statement_circuit : Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some Flock.Tags.circuit := rfl

/-- **A table `verify --zk` sets up is one it sets up at no recomputed public input words**: `DrawSetupZHJ` is
`DrawSetupZHJP` at `own = none`, `Stmt.setupHidden`'s default. -/
def setupZHJP {n mPts : ℕ} {I' : Exec.Inputs} {S : Finset (Fin n)} {d : Lean.Json}
    (y : DrawSetupZHJ I' mPts S d) : DrawSetupZHJP I' none mPts S d where
  x :=
    { st := y.x.st, h := y.x.h, D := y.x.D, hD := y.x.hD, hS := y.x.hS, hg := y.x.hg, hblk := y.x.hblk, m := y.x.m,
      k0 := y.x.k0, l₀ := y.x.l₀, l₁ := y.x.l₁, rest := y.x.rest, ext := y.x.ext, hsch := y.x.hsch, hm := y.x.hm,
      hmPts := y.x.hmPts, wf := y.x.wf }
  sch := y.sch
  tPad := y.tPad
  hz := y.hz
  hshape := y.hshape
  hmsch := y.hmsch
  hpad := y.hpad
  hN := y.hN
  hL := y.hL
  hpos := y.hpos
  hzk := y.hzk

/-- **An accepted session on `inputsAt`'s inputs passes the checks** at its parse, its public file at the circuit's
empty input ports and at its outputs, its population, and its first table, set up at the record's sample
(`drawHypsZHJ`), given what a play does not check of the parameters: a descriptor of the verifier's program, and the
record's sample file being the verifier's `subset kd` sample at the law `lawAt` reads. -/
theorem Accepted.checksWitness {θ : ByteArray} {p : _root_.Spot.OneStageZk.Params} {cf pf : ByteArray}
    {t : Exec.Transcript} {n : ℕ} (hA : Accepted (inputsAt p cf pf) t p.base.verifier n) (hθ : ParamsAt θ p)
    (hch : ∃ key label, p.base.challenger = some (key, label)) {desc : Claims.Msg}
    (hdesc : _root_.Spot.Program.parse ⟨desc.toArray⟩ = .ok p.base.verifier.program) {kd : ℕ}
    (hlaw : lawAt p.base.verifier n = .ok (.subset kd)) {S : Finset (Fin n)}
    (hd : Exec.recordDraw t.record = some (printedDraw (.subset kd) S)) :
    Nonempty (ChecksWitness θ cf pf) := by
  have hflat : (zkTags (inputsAt p cf pf)).hiddenOutputs = true ∧ (zkTags (inputsAt p cf pf)).isTmpl = false :=
    statement_zkTags statement_circuit _ _
  obtain ⟨c, hc, pubO, hO⟩ := ksCheck_parsed hA.form
  rw [circuitOf_parse hflat.1 hflat.2] at hc
  have hD := Flock.Desc.parse_single hc
  obtain ⟨hpi, pub, hpub⟩ := noInputs_of_accepts hA.accepts hflat rfl hc
  have hnO : pubO.n = n := hA.loads hO
  have hn : pub.n = n := (Flock.Bridge.loadPublic_inOut hpub hO).1.symm.trans hnO
  have hform : Flock.KsScope.check (zkTags (inputsAt p cf pf)) (inputsAt p cf pf).circuitFile
      (inputsAt p cf pf).publicFile (inputsAt p cf pf).tables (inputsAt p cf pf).count
      (Exec.recordDraw t.record) = .ok () := by
    show Flock.KsScope.checkOf (Flock.HmOut.parse (zkTags (inputsAt p cf pf)) (inputsAt p cf pf).circuitFile
      (inputsAt p cf pf).tables) (zkTags (inputsAt p cf pf)) (inputsAt p cf pf).publicFile (inputsAt p cf pf).count
      (Exec.recordDraw t.record) = .ok ()
    rw [← circuitOf_parse hflat.1 hflat.2]
    exact hA.form
  obtain ⟨ht, hU, hscope, hTags, -⟩ := Flock.Bridge.ksScoped_files hD hO hform
  have hdiv : kd % nTab (inputsAt p cf pf) = 0 := by
    show kd % 1 = 0
    exact Nat.mod_one kd
  obtain ⟨-, hy⟩ := drawHypsZHJ (inputsAt p cf pf) ht hTags hc hU hscope kd hdiv (printedDraw (.subset kd))
    (drawFileZK_printedDraw kd) hA.accepts hd
  obtain ⟨y⟩ := hy ⟨0, nTab_pos _⟩
  have hk : Flock.Bridge.lawK
      (Flock.Bridge.designAt desc p.base.verifier.query p.base.verifier.law p.base.verifier.workTable n) = kd := by
    rw [designAt_of_lawAt hdesc hlaw] <;> rfl
  have hy' : Nonempty (DrawSetupZHJP (inputsJ (inputsAt p cf pf) 0) none
      (mPtsOf c.g (Flock.Bridge.lawK
        (Flock.Bridge.designAt desc p.base.verifier.query p.base.verifier.law p.base.verifier.workTable n) /
          nTab (inputsAt p cf pf)))
      (partS (inputsAt p cf pf) (printedDraw (.subset kd) S) S ⟨0, nTab_pos _⟩)
      (drawJ (inputsAt p cf pf) (printedDraw (.subset kd) S) ⟨0, nTab_pos _⟩)) := by
    rw [hk]
    exact ⟨setupZHJP y⟩
  exact ⟨{ p := p
           D := .single c
           pub := pub
           pubO := pubO
           n := n
           desc := desc
           S₀ := partS (inputsAt p cf pf) (printedDraw (.subset kd) S) S ⟨0, nTab_pos _⟩
           d₀ := drawJ (inputsAt p cf pf) (printedDraw (.subset kd) S) ⟨0, nTab_pos _⟩
           checks :=
            { params := hθ
              challenger := hch
              parse := hD
              noInputs := hpi
              loadIn := hpub
              loadOut := hO
              ports := hA.ports
              form := ⟨_, hA.form⟩
              popIn := hn
              popOut := hnO
              pop := hA.pop
              program := hdesc
              subset := ⟨kd, hlaw⟩
              setup := hy' } }⟩

variable {ι : Type} {dv : Accounting.Device ι} {α : Type} [Fintype α]

/-- **An accepted play on the registered files passes the checks `sessionOf` reads**: where v2's sampled verdict
accepts a play on the parameters `θ` whose blobs are the registered files `F`, then the evidence's `B`, at a
vocabulary whose digests are v2's (`programDigestOf`, `publicDigestOf`) and whose parameters name them (`DigestsAt`),
each pinned table the blobs hold being its registered file (`TablesSent`), `SessionChecks θ F` holds; given, of v2's
parameters `p₀` in `θ`, a descriptor of the verifier's program and that the registration's law gives a plain subset
sampling design wherever `lawAt` reads one. The record's sample file is then the subset print of the units v2 drew
(`drawHeld`, `drawOf_filter`), at which `verify --zk` sets the first table up (`Accepted.checksWitness`). -/
theorem sessionChecks_of_verdict {V : Vocab dv α} {k : Verity.Registration.Composite} {o : Verity.Exchange}
    {θ : ByteArray} {F B : Array ByteArray}
    (hc : ∀ θ, V.programDigest θ = programDigestOf θ) (hp : ∀ θ, V.publicDigest θ = publicDigestOf θ)
    (hD : DigestsAt V θ) (hreg : V.Registered θ F) (hv : V.Verdict k .sampled o) (hθ : o.params = θ)
    (hs : o.sent.blobs = F ++ B) (ht : TablesSent o.sent) {p₀ : _root_.Spot.OneStageZk.Params}
    (hp₀ : ParamsAt θ p₀) {desc : Claims.Msg}
    (hdesc : _root_.Spot.Program.parse ⟨desc.toArray⟩ = .ok p₀.base.verifier.program)
    (hsub : ∀ n L, lawAt p₀.base.verifier n = .ok L → ∃ kd, L = .subset kd) :
    SessionChecks θ F := by
  subst hθ
  obtain ⟨a, b, e, p, c, ev, ξ, dr, hcase, hacc⟩ := case_drew hv.2.1 hp₀ (verdict_case hv)
  have hpp : p₀ = p := paramsAt_case hv.2.1 hcase hacc hp₀
  subst hpp
  obtain ⟨-, -, -, -, hsrc, pub, prof, hck⟩ := id hacc
  obtain ⟨key, hkey, -⟩ := source_of hsrc
  obtain ⟨tags0, circuitB, pubB, record, proofs, n, htags, hcirc, hpubB, hrec, -, hn, hA⟩ := checkBlobsAt_event hck
  obtain rfl : tags0 = Flock.Tags.circuit := Option.some.inj (htags.symm.trans statement_circuit)
  obtain ⟨out, pub', prof', hat⟩ := checkBlobsAt_checkAt hck
  obtain ⟨law, n', read, hd, hn', hdf⟩ := checkAt_drew hat
  obtain ⟨derived, fl, hder, hfl, hdu⟩ := drawnFrom_ok hdf
  have hnn : n' = n := Option.some.inj (hn'.symm.trans hn)
  rw [hnn] at hfl hdu
  have hlaw : lawAt p₀.base.verifier n = .ok (derived.getD fl) :=
    lawAt_of_lawOf (c := c) ⟨law, derived, fl, hd, hn, hder, hfl, rfl⟩
  obtain ⟨kd, hkd⟩ := hsub _ _ hlaw
  rw [hkd] at hlaw hdu
  have hsz : (drawnUnits ev.base.draw).size = kd := Flock.Bridge.drawUnits_size (law := .subset kd) (ns := [kd]) (kd := kd)
    ⟨rfl, by simp⟩ hdu
  obtain ⟨-, -, -, -, -, hcs, -, -, -⟩ := sent_verified hv.2.1 hcase hacc
  have hsr : sentRecord o.sent = some record := by
    unfold sentRecord
    rw [hcs]
    exact hrec
  obtain ⟨hasc, hrd⟩ := drawHeld k o.sent dr (.subset kd) n ξ (drawnUnits ev.base.draw) record hv.2.1
    ⟨a, b, e, p₀, c, ev, hcase, hacc, ⟨law, derived, fl, hd, hn, hder, hfl, hkd.symm⟩, rfl, read, hdu⟩ hsr
  have hrd' : Exec.recordDraw record =
      some (printedDraw (n := n) (.subset kd) (Finset.univ.filter fun u : Fin n => u.val ∈ drawnUnits ev.base.draw)) := by
    rw [exec_recordDraw, hrd]
    show _ = some (ZkLink.drawOf (.subset kd) n _).toJson
    rw [drawOf_filter hasc, hsz]
  obtain ⟨cB', hg', hl⟩ := tables_registered hp₀ hv
  obtain rfl : circuitB = cB' := Except.ok.inj (hcirc.symm.trans hg')
  have htab := pinnedTables_eq hl rfl ht
  rw [inputsOf_eq htab.symm] at hA
  have hcf : circuitB = circuitAt o.params F := by
    rw [hs] at hcirc
    have h := get_named (hD.circuit p₀ hp₀) hreg.1 hcirc
    rwa [hc] at h
  have hpf : pubB = publicAt o.params F := by
    rw [hs] at hpubB
    have h := get_named (hD.publicFile p₀ hp₀) hreg.2 hpubB
    rwa [hp] at h
  subst hcf hpf
  exact hA.checksWitness hp₀ ⟨key, dr.label, hkey⟩ hdesc hlaw hrd'

end Verity.Sampled
