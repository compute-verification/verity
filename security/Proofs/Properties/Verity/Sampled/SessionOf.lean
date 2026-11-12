module

public import Proofs.Properties.Verity.Sampled.OpensAt
public import Proofs.Properties.Verity.Sampled.Event
public import Proofs.Flock.Bridge.Refused
public import Proofs.Flock.Soundness.Discharge.FailClosed.ZkHiddenM1

/-!
# The session v2 sets up, built from the parameters and the files (`sessionOf`)

`Live` is a session's binders and facts; nothing in `Sampled` builds one from the parameters and the files. This file
does, under one `Prop` of what v2's check computes on them (`SessionChecks`), so a deployment's session at `θ` can be
`sessionOf` at the registered files, and v2 sets it up there (`SetsUp`) and it is ready for openings (`OpensReady`) by
construction.

* **v2's inputs to `verify --zk`** (`inputsAt`): at v2's parameters `p`, a circuit file and a public file, the
  statement's tags, one table, the lookup tables that circuit file registers (`pinnedTables`), the parameters' partition
  object and program, no work, stratified or closure input (`Spot.OneStageZk.inputsOf`'s shape, at the registered
  tables). On the files `F` they are the ones the parameters' digests name (`circuitAt`, `publicAt`), so files naming
  the same two pass the same checks (`sessionChecks_files`).
* **What v2's check computes on them** (`ChecksAt`, `ChecksWitness`, `SessionChecks`): v2's parameters in `θ` and their
  challenger; the circuit file's parse `D`, which names no public input port; the public file loaded at the circuit's
  empty input ports (`pub`) and at its outputs (`pubO`), both of `n` proof blocks; the ports and roots check and the
  form check at some sample (`PortRoots.checkOf`, `KsScope.checkOf`, as `Accepted` states them); the registration's
  population `n` under the verifier's program and query (`Slots.popOk`); a descriptor `desc` that parses to the
  verifier's program; the sampling design `lawAt` reads, `designAt`'s; and a reference sample at which `verify --zk`
  sets a table up (`DrawSetupZHJP`). One clause is not a check v2 makes: the sampling design `lawAt` reads is a plain
  subset (`ChecksAt.subset`, giving `Live.hsub`), the theorem's limit.
* **The bound's parameters** (`SessionBound`): `k`, `Rw`, `M`, the claim's share `ε` and the costs `v`, `w`, `h`, with
  `rateM1 k < 1`.
* **The tables' rate** (`RateM1`): every table of the hidden-output plan at public inputs `own` has rate at most
  `rateM1 k`. It is `planZHJ_M1W` and `rateZ_le_m1w` for `planZHJP`, not yet proved; `sessionOf` takes it as a
  hypothesis.
* **The session** (`ChecksWitness.live`, `sessionOf`): each binder of `Live` the value the checks compute, its facts
  read off them (`Flock.Bridge.ksScoped_files`: untyped tags, the built unit, the scope, the `--zk` tags;
  `Flock.Bridge.v5Files_of_checksStmt`: no two ports one domain; `Flock.Bridge.popMatches_of_popOk`: the population in
  order), at the device's bytes as deployed (`deployedSrc`, A3 with no premise) and the bound's parameters.
* **v2 sets it up** (`ChecksWitness.live_setsUp`, `sessionOf_setsUp`): at a vocabulary whose digests and reading are
  v2's, whose families give the parameters' challenger `MAX_COINS` coins and whose certificate's share is the bound's.
* **It is ready for openings** (`ChecksWitness.live_opensReady`, `sessionOf_opensReady`): its files pass the ports and
  roots check.
-/

@[expose] public section

namespace Verity.Sampled

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Exec FlockSoundness.Discharge.ZkExec
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.ZkLink

/-- **The circuit file the parameters `θ` name in the files `F`**: the one with the digest v2's parameters hold
(`programDigestOf`), as `readOf` reads it. -/
noncomputable abbrev circuitAt (θ : ByteArray) (F : Array ByteArray) : ByteArray :=
  Verity.namedFile (programDigestOf θ) F

/-- **The public file the parameters `θ` name in the files `F`** (`publicDigestOf`), as `readOf` reads it. -/
noncomputable abbrev publicAt (θ : ByteArray) (F : Array ByteArray) : ByteArray :=
  Verity.namedFile (publicDigestOf θ) F

/-- **The inputs v2 hands `verify --zk`** at its parameters `p`, the circuit file `cf` and the public file `pf`: the
statement's tags, one table, the lookup tables the circuit file registers, and the parameters' partition object and
program (`Spot.OneStageZk.inputsOf`'s, at the registered tables), with no work, stratified or closure input
(`Recursion.Witness.toExec`). -/
noncomputable def inputsAt (p : _root_.Spot.OneStageZk.Params) (cf pf : ByteArray) : Exec.Inputs :=
  Recursion.Witness.toExec
    { tags := Flock.Tags.circuit
      circuitFile := cf
      publicFile := pf
      count := 1
      tables := pinnedTables cf
      partition := some p.own.partition
      program := some p.own.program }

/-- **What v2's check computes on the parameters `θ`, the circuit file `cf` and the public file `pf`**, at its
parameters `p`, the circuit file's parse `D`, the public file loaded at the circuit's empty input ports (`pub`) and at
its outputs (`pubO`), the population `n`, a descriptor `desc` of the verifier's program and a reference sample `S₀`
printed as `d₀`: each clause a check v2 makes before it accepts, as `Accepted` and v2's parameters' read state it, but
`subset`, which v2 does not check and the theorem for sampled proofs covers alone. -/
structure ChecksAt (θ : ByteArray) (cf pf : ByteArray) (p : _root_.Spot.OneStageZk.Params) (D : Flock.Desc)
    (pub pubO : Flock.Public) (n : ℕ) (desc : Claims.Msg) (S₀ : Finset (Fin n)) (d₀ : Lean.Json) : Prop where
  /-- v2's parameters in `θ` -/
  params : ParamsAt θ p
  /-- the parameters name the challenger whose coins the sample reads -/
  challenger : ∃ key label, p.base.challenger = some (key, label)
  /-- the circuit file parses -/
  parse : Flock.Desc.parse (zkTags (inputsAt p cf pf)) (inputsAt p cf pf).circuitFile (inputsAt p cf pf).tables = .ok D
  /-- the circuit names no public input port -/
  noInputs : Flock.HmIn.parse D.unroll = .ok {}
  /-- the public file loads at the circuit's empty input ports -/
  loadIn : Flock.HmIn.loadPublic (zkTags (inputsAt p cf pf)) D.unroll {} (inputsAt p cf pf).publicFile = .ok pub
  /-- the public file loads at the circuit's outputs -/
  loadOut : Flock.HmOut.loadPublic (zkTags (inputsAt p cf pf)) D.unroll (inputsAt p cf pf).publicFile = .ok pubO
  /-- the ports and roots check (`Accepted.ports`) -/
  ports : Flock.PortRoots.checkOf
    (Flock.ProvedScope.circuitOf (zkTags (inputsAt p cf pf)) (inputsAt p cf pf).circuitFile (inputsAt p cf pf).tables)
    (zkTags (inputsAt p cf pf)) (inputsAt p cf pf).publicFile = .ok ()
  /-- the form check at some sample (`Accepted.form`) -/
  form : ∃ draw, Flock.KsScope.checkOf
    (Flock.ProvedScope.circuitOf (zkTags (inputsAt p cf pf)) (inputsAt p cf pf).circuitFile (inputsAt p cf pf).tables)
    (zkTags (inputsAt p cf pf)) (inputsAt p cf pf).publicFile (inputsAt p cf pf).count draw = .ok ()
  /-- the public file holds the population's proof blocks (`Accepted.loads`) -/
  popIn : pub.n = n
  popOut : pubO.n = n
  /-- the registration's population is `n` proof blocks of the verifier's program under its query, in order
  (`Accepted.pop`) -/
  pop : _root_.Spot.Slots.popOk p.base.verifier.program p.base.verifier.query n = true
  /-- the descriptor parses to the verifier's program -/
  program : _root_.Spot.Program.parse ⟨desc.toArray⟩ = .ok p.base.verifier.program
  /-- the sampling design v2 samples under over the population is the registration's (`lawAt`, `designAt`) -/
  law : lawAt p.base.verifier n =
    .ok (Flock.Bridge.designAt desc p.base.verifier.query p.base.verifier.law p.base.verifier.workTable n)
  /-- **not a check v2 makes**: the sampling design the registration's law gives over the population is a plain
  subset, the one the theorem covers (`Live.hsub`) -/
  subset : ∃ kd, lawAt p.base.verifier n = .ok (.subset kd)
  /-- `verify --zk` sets a table up at the reference sample -/
  setup : Nonempty (DrawSetupZHJP (inputsJ (inputsAt p cf pf) 0) none
    (mPtsOf D.unroll.g (Flock.Bridge.lawK
      (Flock.Bridge.designAt desc p.base.verifier.query p.base.verifier.law p.base.verifier.workTable n) /
        nTab (inputsAt p cf pf))) S₀ d₀)

/-- **The values v2's check computes on `θ`, `cf` and `pf`, with its checks passing** (`ChecksAt`). -/
structure ChecksWitness (θ : ByteArray) (cf pf : ByteArray) where
  p : _root_.Spot.OneStageZk.Params
  D : Flock.Desc
  pub : Flock.Public
  pubO : Flock.Public
  n : ℕ
  desc : Claims.Msg
  S₀ : Finset (Fin n)
  d₀ : Lean.Json
  checks : ChecksAt θ cf pf p D pub pubO n desc S₀ d₀

/-- **v2's checks pass on the parameters `θ` and the files `F`**, at the circuit and public files the parameters name
in them: the `Prop` a session built from them needs. -/
def SessionChecks (θ : ByteArray) (F : Array ByteArray) : Prop :=
  Nonempty (ChecksWitness θ (circuitAt θ F) (publicAt θ F))

/-- **Files naming the same circuit and public files pass the same checks.** -/
theorem sessionChecks_files {θ : ByteArray} {F F' : Array ByteArray} (hc : circuitAt θ F = circuitAt θ F')
    (hp : publicAt θ F = publicAt θ F') : SessionChecks θ F ↔ SessionChecks θ F' := by
  unfold SessionChecks
  rw [hc, hp]

/-- **The bound's parameters**: `k`, `Rw`, `M`, the claim's share `ε`, and the costs of one opening's verification, one
acceptance check and one hashed message, in SHA-512 calls; at `rateM1 k < 1`. -/
structure SessionBound where
  k : ℕ
  Rw : ℕ
  M : ℕ
  hk : 1 ≤ k
  hRw : 1 ≤ Rw
  hM : 1 ≤ M
  hρ : rateM1 k < 1
  ε : ℝ
  v : ℕ
  w : ℕ
  h : ℕ

/-- **Every table of the hidden-output plan at public inputs `own` has rate at most `rateM1 k`**: `planZHJ_M1W` and
`rateZ_le_m1w` at `planZHJP`. A statement to prove (`tableZHJP_M1W`, `planZHJP_M1W` beside `planZHJ_M1W` in
`FailClosed/ZkHiddenM1.lean`); until then `sessionOf` takes it as a hypothesis. -/
def RateM1 : Prop :=
  ∀ (I : Exec.Inputs) (own : Option ByteArray) (n mPts : ℕ) (dj : Finset (Fin n) → Lean.Json)
    (S₀ : Finset (Fin n)) (d₀ : Lean.Json) (y₀ : DrawSetupZHJP (inputsJ I 0) own mPts S₀ d₀)
    (S : Finset (Fin n)) (R : ByteArray) (j : Fin (planZHJP (Reg := ByteArray) dj y₀ S R).length) (k : ℕ),
    rateZ (planZHJP (Reg := ByteArray) dj y₀ S R) j k ≤ rateM1 k

namespace ChecksWitness

variable {θ : ByteArray} {cf pf : ByteArray} (W : ChecksWitness θ cf pf)

/-- The inputs v2 hands `verify --zk` at the witness's parameters. -/
noncomputable abbrev I : Exec.Inputs := inputsAt W.p cf pf

theorem count_I : W.I.count = 1 := rfl

theorem nTab_I : nTab W.I = 1 := rfl

theorem statement_tags : Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some W.I.tags := rfl

theorem flat : (zkTags W.I).hiddenOutputs = true ∧ (zkTags W.I).isTmpl = false :=
  statement_zkTags W.statement_tags _ _

/-- The form check on `HmOut.parse`'s circuit (`KsScope.check`), the one `ksScoped_files` reads. -/
theorem formCheck : ∃ draw, Flock.KsScope.check (zkTags W.I) W.I.circuitFile W.I.publicFile W.I.tables W.I.count draw
    = .ok () := by
  obtain ⟨draw, h⟩ := W.checks.form
  refine ⟨draw, ?_⟩
  show Flock.KsScope.checkOf (Flock.HmOut.parse (zkTags W.I) W.I.circuitFile W.I.tables) (zkTags W.I)
    W.I.publicFile W.I.count draw = .ok ()
  rw [← circuitOf_parse W.flat.1 W.flat.2]
  exact h

/-- What `ksScoped_files` reads off the form check. -/
theorem scoped : W.I.tags.typed = false ∧ FlockSoundness.Built W.D.unroll.unit ∧
    FlockSoundness.Discharge.Layout.scopeOkH W.D.unroll = true ∧ TagsOkZHJ W.I := by
  obtain ⟨draw, h⟩ := W.formCheck
  obtain ⟨ht, hU, hs, hT, -⟩ := Flock.Bridge.ksScoped_files W.checks.parse W.checks.loadOut h
  exact ⟨ht, hU, hs, hT⟩

/-- No two of the circuit's ports have one published domain (`v5Files_of_checksStmt`). -/
theorem dup : Flock.PortRoots.dupId W.D.unroll W.pub = false := by
  obtain ⟨draw, h⟩ := W.checks.form
  exact (Flock.Bridge.v5Files_of_checksStmt ((zkTags_typed W.I).trans W.scoped.1)
    (Flock.Desc.parseStmt_of_parse W.checks.parse) W.checks.loadIn W.checks.popIn W.checks.ports h).1

/-- The sampling design v2 samples under from the registration. -/
abbrev design : Flock.Draw.Law :=
  Flock.Bridge.designAt W.desc W.p.base.verifier.query W.p.base.verifier.law W.p.base.verifier.workTable W.n

/-- The registration's sampling design is the subset one `lawAt` reads (`ChecksAt.law`, `ChecksAt.subset`). -/
theorem hsub : W.design = .subset (Flock.Bridge.lawK W.design) := by
  obtain ⟨kd, hk⟩ := W.checks.subset
  have h : W.design = .subset kd := Except.ok.inj (W.checks.law.symm.trans hk)
  rw [h]
  rfl

/-- **The session built from the witness**, at the bound's parameters `B`, wherever the tables' rate is
`rateM1`'s (`RateM1`): each binder the value v2's check computes, at the device's bytes as deployed. -/
noncomputable def live (B : SessionBound) (hrate : RateM1) : Live where
  n := W.n
  I := W.I
  own := none
  ht := W.scoped.1
  D := W.D
  hD := W.checks.parse
  hU := W.scoped.2.1
  hpi := W.checks.noInputs
  pub := W.pub
  hpub := W.checks.loadIn
  pubO := W.pubO
  hpubO := W.checks.loadOut
  hscope := W.scoped.2.2.1
  hdup := W.dup
  hn := W.checks.popIn
  hnO := W.checks.popOut
  desc := W.desc
  qr := W.p.base.verifier.query
  hpop := Flock.Bridge.popMatches_of_popOk W.checks.program W.checks.pop
  lj := W.p.base.verifier.law
  tb := W.p.base.verifier.workTable
  hsub := W.hsub
  hdiv := by show _ % nTab W.I = 0; rw [W.nTab_I, Nat.mod_one]
  S₀ := W.S₀
  d₀ := W.d₀
  y₀ := Classical.choice W.checks.setup
  Ω := DeployedCoins
  Lb := deployedLb
  src := deployedSrc
  hA3 := uniformRandomBytes_deployed
  hTags := W.scoped.2.2.2
  k := B.k
  Rw := B.Rw
  M := B.M
  hk := B.hk
  hRw := B.hRw
  hM := B.hM
  ρ := rateM1 B.k
  hρ := B.hρ
  hr S R j := hrate _ _ _ _ _ _ _ _ S R j B.k
  ε := B.ε
  v := B.v
  w := B.w
  h := B.h

/-- **The witness's session is ready for openings**: its files pass the ports and roots check. -/
theorem live_opensReady (B : SessionBound) (hrate : RateM1) : (W.live B hrate).OpensReady :=
  ⟨W.checks.ports⟩

end ChecksWitness

variable {ι : Type} {dv : Accounting.Device ι} {α : Type} [Fintype α]

/-- **v2 sets the witness's session up on `θ` and `F`**, at the circuit and public files the parameters name in `F`, at
a vocabulary whose digests and reading of the files are v2's (`programDigestOf`, `publicDigestOf`, `readOf`: an
`Instance`'s `programDigest`, `publicDigest` and `read`), whose coin families give the parameters' challenger
`MAX_COINS` coins (an `Instance`'s `coins`, at the family the deployment names) and whose published certificate's share
is the bound's. -/
theorem ChecksWitness.live_setsUp {θ : ByteArray} {F : Array ByteArray}
    (W : ChecksWitness θ (circuitAt θ F) (publicAt θ F)) (V : Vocab dv α)
    (hc : ∀ θ, V.programDigest θ = programDigestOf θ) (hp : ∀ θ, V.publicDigest θ = publicDigestOf θ)
    (hr : ∀ θ F, V.read θ F = readOf θ F)
    (hfam : ∀ key label, W.p.base.challenger = some (key, label) →
      (label, _root_.Spot.Challenger.MAX_COINS) ∈ V.families θ)
    (B : SessionBound) (hε : (V.certOf θ).ε = B.ε) (hrate : RateM1) : SetsUp V θ F (W.live B hrate) where
  params := by
    obtain ⟨key, label, hch⟩ := W.checks.challenger
    exact ⟨W.p, W.checks.params, rfl, rfl, W.checks.law, W.checks.program, rfl,
      key, label, _root_.Spot.Challenger.MAX_COINS, hch, hfam key label hch, rfl⟩
  tags := W.statement_tags
  count := rfl
  work := rfl
  stratified := rfl
  closure := rfl
  own := rfl
  circuit := by rw [Vocab.circuitFile, hc]; rfl
  publicFile := by rw [Vocab.publicFile, hp]; rfl
  read := by
    rw [hr]
    exact readOf_eq V hc hp W.statement_tags rfl (by rw [Vocab.circuitFile, hc]; rfl)
      (by rw [Vocab.publicFile, hp]; rfl) (by rw [Vocab.circuitFile, hc]; rfl)
  share := hε

open Classical in
/-- **The session v2 sets up on the parameters `θ` and the files `F`**, where its checks pass (`SessionChecks`), at the
bound's parameters `B` and the tables' rate (`RateM1`); `none` elsewhere. The values the checks compute are unique
but for the reference sample, so the choice of witness is the choice of a reference sample at which `verify --zk`
sets a table up, which `Live`'s plan reads only where no witness satisfies the refused table. -/
noncomputable def sessionOf (θ : ByteArray) (F : Array ByteArray) (B : SessionBound) (hrate : RateM1) : Option Live :=
  if h : SessionChecks θ F then some ((Classical.choice h).live B hrate) else none

theorem sessionOf_some {θ : ByteArray} {F : Array ByteArray} {B : SessionBound} {hrate : RateM1} {A : Live}
    (h : sessionOf θ F B hrate = some A) :
    ∃ W : ChecksWitness θ (circuitAt θ F) (publicAt θ F), A = W.live B hrate := by
  unfold sessionOf at h
  split at h
  · exact ⟨_, (Option.some.inj h).symm⟩
  · cases h

theorem sessionOf_isSome {θ : ByteArray} {F : Array ByteArray} (B : SessionBound) (hrate : RateM1)
    (h : SessionChecks θ F) : (sessionOf θ F B hrate).isSome := by
  unfold sessionOf
  rw [dif_pos h]
  rfl

/-- **v2 sets up the session `sessionOf` builds** on `θ` and `F`, at a vocabulary as `ChecksWitness.live_setsUp` asks,
whose families give the challenger any of `θ`'s parameters name `MAX_COINS` coins. -/
theorem sessionOf_setsUp (V : Vocab dv α) (hc : ∀ θ, V.programDigest θ = programDigestOf θ)
    (hp : ∀ θ, V.publicDigest θ = publicDigestOf θ) (hr : ∀ θ F, V.read θ F = readOf θ F) {θ : ByteArray}
    {F : Array ByteArray}
    (hfam : ∀ p key label, ParamsAt θ p → p.base.challenger = some (key, label) →
      (label, _root_.Spot.Challenger.MAX_COINS) ∈ V.families θ)
    {B : SessionBound} (hε : (V.certOf θ).ε = B.ε) {hrate : RateM1} {A : Live}
    (h : sessionOf θ F B hrate = some A) : SetsUp V θ F A := by
  obtain ⟨W, rfl⟩ := sessionOf_some h
  exact W.live_setsUp V hc hp hr (fun key label => hfam W.p key label W.checks.params) B hε hrate

/-- **The session `sessionOf` builds is ready for openings.** -/
theorem sessionOf_opensReady {θ : ByteArray} {F : Array ByteArray} {B : SessionBound} {hrate : RateM1} {A : Live}
    (h : sessionOf θ F B hrate = some A) : A.OpensReady := by
  obtain ⟨W, rfl⟩ := sessionOf_some h
  exact W.live_opensReady B hrate

end Verity.Sampled
