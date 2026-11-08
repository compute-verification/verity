module

public import Proofs.Flock.Bridge.V5.KSLive
public import Proofs.Flock.Bridge.V5.Refused
public import Proofs.Flock.Bridge.Refused
public import Proofs.Flock.Claims.Recursion.SetupValues
public import Proofs.Flock.Soundness.Discharge.Hidden.Public
public import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SharedP
public import Flock.With.SetupEq

/-!
# `SampledKSLiveV5`'s file preconditions, read off acceptance (`SampledKSLiveV5Accepted`)

`SampledKSLiveV5R`, V5's bound under the roots guard, takes four things of the files and the registration as
hypotheses: `Bridge.RowsOk`, `Registered.dupName = none`, `RootFixesDomainL` at the public file's leaf map, and
`Bridge.PopMatches`. The verifiers refuse every file and registration that fails them: `verify --zk` runs `Flock.PortRoots.checkOf` and then
`Flock.KsScope.checkOf` last in its verdict (`Main.verifyCmd`), and the sampled-proofs run refuses the registration
(`SampledProofs.Run.run`, `population-order`, `SampledProofs.Slots.popOk`).

`PortRoots` reads the public file as `HmOut.loadPublic` loads it, without the circuit's public input words, while V5
reads it over them (`HmIn.loadPublic`). Both are `HmRow.loadPublic` at one circuit's ports with other output ports, and
the population, its `b ‖ c` and its tables don't read the output ports (`loadPublic_agree`), so the roots `PortRoots`
checks are V5's.

* `loadPublic_core`: what a load returns of the population, its `b ‖ c` and its tables, as functions of the file and the
  circuit's ports; `loadPublic_agree`, `loadPublic_inOut`: two loads at one circuit's ports agree on them.
* `v5Files_of_checks`: `PortRoots.check` and `KsScope.check` passing give `dupName = none` and `RootFixesDomainL`
  (at a file without shared rows, at the public file's population: `pub.n = n`).
* `SampledKSLiveV5Accepted`: `SampledKSLiveV5R` with its event `LiveAcceptsKSP ∧ RunPopOk` in place of `LiveAcceptsZKP`
  and without the four hypotheses, under the sampling design the verifier samples under from the registration (`Bridge.DrawsUnder`),
  no Bernoulli law, and at `δ` the sampling design's own miss of `B`, as V5 has them; `sampledKSLiveV5Accepted_of` derives it from
  `SampledKSLiveV5R` as `sampledKSLiveV3_of` derives V3 from V2, and `sampledKSLiveV5Accepted` proves it. The session's
  simulator is V5's: its sample files decode to `kd` units whatever the population (`Bridge.lawSized_drawsUnder`,
  `Bridge.lawSized_size`), so it reads no population check.
-/

@[expose] public section

namespace Flock.Bridge

open Flock RecDeploy.SetupValues

/-- A file's `b ‖ c` as `HmRow.loadPublic` reads them: `nBcs` pieces of `BC_BYTES` bytes after its header line. -/
def bcsOf (f : ByteArray) (nl nBcs : ℕ) : Array ByteArray :=
  (Array.range nBcs).map fun i => (f.extract (nl + 1) f.size).extract (HmRow.BC_BYTES * i) (HmRow.BC_BYTES * i + HmRow.BC_BYTES)

/-- A shared-row file's tables as `HmRow.loadPublic` reads them, at their `counts`. -/
def tablesOf (f : ByteArray) (nl nBcs n np : ℕ) (counts : Array ℕ) : Flock.Tables :=
  { counts, starts := (counts.foldl (fun (acc : Array ℕ × ℕ) r => (acc.1.push acc.2, acc.2 + r)) (#[], 0)).1,
    refs := (Array.range (n * np)).map fun k => (u32OfLE (f.extract (nl + 1) f.size) (HmRow.BC_BYTES * nBcs + 4 * k)).toNat }

/-- **What a load returns of the population, its `b ‖ c` and its tables**: the header's `instances`, the pieces after
the header line, and the tables at the header's `shared_rows` over the circuit's ports, whatever its output ports. -/
theorem loadPublic_core (h : ByteArray → ByteArray) (tags : Tags) (c : Circuit) (f : ByteArray) (pop : Public)
    (hp : HmRow.loadPublicWith (M := V) (Hashes.ofSha512 h) tags c f = .ok pop) :
    ∃ nl hd counts, findByte f 10 = some nl ∧ headerOf f = some hd ∧ J.nat hd "instances" "public file" = .ok pop.n ∧
      HmRow.sharedRows tags c hd = .ok counts ∧
      pop.digests = bcsOf f nl ((counts.map (·.foldl (· + ·) 0)).getD (pop.n * c.ports.size)) ∧
      pop.tables = counts.map
        (tablesOf f nl ((counts.map (·.foldl (· + ·) 0)).getD (pop.n * c.ports.size)) pop.n c.ports.size) := by
  unfold HmRow.loadPublicWith at hp
  simp (config := { zeta := false }) only [Hashes.ofSha512, monadLift_self] at hp
  cases hnl : findByte f 10 with
  | none => simp (config := { zeta := false }) only [hnl] at hp; cases hp
  | some nl =>
  simp (config := { zeta := false }) only [hnl] at hp
  cases htx : String.fromUTF8? (f.extract 0 nl) with
  | none =>
    simp (config := { zeta := false }) only [htx] at hp
    obtain ⟨_, hte, -⟩ := except_bind_ok hp; cases hte
  | some text =>
  simp (config := { zeta := false }) only [htx, pure_bind] at hp
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpA at hp
  simp (config := { zeta := false }) only [jpA] at hp
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpB at hp
  cases hj : Flock.JsonParse.parse text with
  | error e =>
    simp (config := { zeta := false }) only [hj] at hp
    obtain ⟨_, hte, -⟩ := except_bind_ok hp; cases hte
  | ok hd0 =>
  simp (config := { zeta := false }) only [hj, jpB] at hp
  have hhd : headerOf f = some (Flock.norm hd0) := by
    simp only [headerOf, hnl, htx, hj, Except.toOption, Option.map_some]
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) pctx at hp
  (have hh := except_bind_ok hp; clear hp; obtain ⟨_, -, hp⟩ := hh)
  (have hh := except_bind_ok hp; clear hp; obtain ⟨_, -, hp⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp1 at hp
  replace hp := ite_throw_ok hp
  simp (config := { zeta := false }) only [jp1] at hp
  (have hh := except_bind_ok hp; clear hp; obtain ⟨_, -, hp⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp2 at hp
  replace hp := ite_throw_ok hp
  simp (config := { zeta := false }) only [jp2] at hp
  (have hh := except_bind_ok hp; clear hp; obtain ⟨n, hn, hp⟩ := hh)
  (have hh := except_bind_ok hp; clear hp; obtain ⟨_, -, hp⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp3 at hp
  replace hp := ite_throw_ok hp
  simp (config := { zeta := false }) only [jp3] at hp
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) np nOut at hp
  (have hh := except_bind_ok hp; clear hp; obtain ⟨counts, hcounts, hp⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) nBcs refsLen body jp4 at hp
  replace hp := ite_throw_ok hp
  simp (config := { zeta := false }) only [jp4] at hp
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) bcs o outputs jpT at hp
  obtain ⟨tables, htab, hp⟩ : ∃ tables, tables = counts.map (tablesOf f nl nBcs n np) ∧ jpT tables = .ok pop := by
    cases counts with
    | none => exact ⟨none, rfl, by simpa only [pure_bind] using hp⟩
    | some cs =>
      simp (config := { zeta := false }) only at hp
      extract_lets (config := { underBinder := false, onlyGivenNames := true }) refs jpS at hp
      replace hp := ite_throw_ok hp
      simp (config := { zeta := false }) only [jpS] at hp
      exact ⟨_, rfl, hp⟩
  simp (config := { zeta := false }) only [jpT] at hp
  (have hh := except_bind_ok hp; clear hp; obtain ⟨_, -, hp⟩ := hh)
  (have hh := except_bind_ok hp; clear hp; obtain ⟨_, -, hp⟩ := hh)
  (have hh := except_bind_ok hp; clear hp; obtain ⟨indices, -, hp⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp5 at hp
  obtain ⟨u5, hp⟩ : ∃ u, jp5 u = .ok pop := by
    split at hp
    · (have hh := except_bind_ok hp; clear hp; obtain ⟨_, -, hp⟩ := hh)
      (have hh := except_bind_ok hp; clear hp; obtain ⟨_, -, hp⟩ := hh)
      exact ⟨(), ite_throw_ok hp⟩
    · exact ⟨(), hp⟩
  simp (config := { zeta := false }) only [jp5] at hp
  (have hh := except_bind_ok hp; clear hp; obtain ⟨_, -, hp⟩ := hh)
  (have hh := except_bind_ok hp; clear hp; obtain ⟨_, -, hp⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp6 at hp
  replace hp := ite_throw_ok hp
  simp (config := { zeta := false }) only [jp6] at hp
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp7 at hp
  replace hp := ite_throw_ok hp
  simp (config := { zeta := false }) only [jp7] at hp
  (have hh := except_bind_ok hp; clear hp; obtain ⟨_, -, hp⟩ := hh)
  cases hp
  exact ⟨nl, _, counts, rfl, hhd, hn, hcounts, rfl, htab⟩

theorem sharedRows_ports {tags : Tags} {c₁ c₂ : Circuit} (hports : c₁.ports = c₂.ports) (hd : Lean.Json) :
    HmRow.sharedRows tags c₁ hd = HmRow.sharedRows tags c₂ hd := by
  unfold HmRow.sharedRows
  rw [hports]

/-- **Two loads at one circuit's ports agree on the population, its `b ‖ c` and its tables**, whatever their output
ports. -/
theorem loadPublic_agree {tags : Tags} {c₁ c₂ : Circuit} {f : ByteArray} {p₁ p₂ : Public}
    (hports : c₁.ports = c₂.ports) (h₁ : HmRow.loadPublic tags c₁ f = .ok p₁)
    (h₂ : HmRow.loadPublic tags c₂ f = .ok p₂) :
    p₁.n = p₂.n ∧ p₁.digests = p₂.digests ∧ p₁.tables = p₂.tables := by
  rw [← Flock.With.loadPublicWith_std] at h₁ h₂
  obtain ⟨nl₁, hd₁, cs₁, hnl₁, hhd₁, hn₁, hc₁, hb₁, ht₁⟩ := loadPublic_core _ _ _ _ _ h₁
  obtain ⟨nl₂, hd₂, cs₂, hnl₂, hhd₂, hn₂, hc₂, hb₂, ht₂⟩ := loadPublic_core _ _ _ _ _ h₂
  rw [hnl₁] at hnl₂
  cases hnl₂
  rw [hhd₁] at hhd₂
  cases hhd₂
  rw [hn₁] at hn₂
  have hn : p₁.n = p₂.n := Except.ok.inj hn₂
  rw [sharedRows_ports hports.symm, hc₁] at hc₂
  cases hc₂
  refine ⟨hn, ?_, ?_⟩
  · rw [hb₁, hb₂, hn, hports]
  · rw [ht₁, ht₂, hn, hports]

/-- **`HmIn.loadPublic` and `HmOut.loadPublic` agree on the population, its `b ‖ c` and its tables.** -/
theorem loadPublic_inOut {tags : Tags} {c : Circuit} {pi : HmIn.Ports} {pf : ByteArray} {pub pubO : Public}
    (h : HmIn.loadPublic tags c pi pf = .ok pub) (hO : HmOut.loadPublic tags c pf = .ok pubO) :
    pubO.n = pub.n ∧ pubO.digests = pub.digests ∧ pubO.tables = pub.tables :=
  loadPublic_agree (c₁ := { c with outPorts := #[] }) (c₂ := FlockSoundness.Discharge.Hidden.inRows c pi) rfl
    (FlockSoundness.Discharge.Hidden.loadPublic_nil hO) (FlockSoundness.Discharge.Hidden.loadPublic_inRows h)

/-- **The leaf map reads the public file's `b ‖ c` and tables only.** -/
theorem LeafMap.ofPublic_congr {c : Flock.Circuit} {p p' : Flock.Public} (nl : ℕ) (hd : p.digests = p'.digests)
    (ht : p.tables = p'.tables) : LeafMap.ofPublic c p nl = LeafMap.ofPublic c p' nl := by
  have hdig : ∀ u q np, p.digest u q np = p'.digest u q np := by
    intro u q np
    simp only [Flock.Public.digest, hd, ht]
  cases h : p'.tables with
  | none =>
    rw [LeafMap.ofPublic_none nl (ht.trans h), LeafMap.ofPublic_none nl h]
    simp only [LeafMap.ident, hdig]
  | some t =>
    rw [LeafMap.ofPublic_some nl (ht.trans h), LeafMap.ofPublic_some nl h]
    simp only [LeafMap.ofTables, hd]

/-- **`PortRoots.check` passing loads the public file** (`HmOut.loadPublic`). -/
theorem portRoots_load {tags : Tags} {cf pf : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (h : Flock.PortRoots.check tags cf pf tables = .ok ()) (hc : HmOut.parse tags cf tables = .ok c) :
    ∃ pub, HmOut.loadPublic tags c pf = .ok pub := by
  unfold Flock.PortRoots.check Flock.PortRoots.checkOf Flock.PortRoots.checkWith at h
  rw [hc] at h
  cases hl : HmOut.loadPublic tags c pf with
  | ok pub => exact ⟨pub, rfl⟩
  | error e =>
    simp only [hl] at h
    split at h
    · cases h
    · split at h <;> cases h

/-- **`verify --zk`'s files checks give V5's file hypotheses**: when `PortRoots.check` and `KsScope.check` pass, the
circuit names each port once and `RootFixesDomainL` holds at the public file's leaf map over its public input ports, at
its population `n`. -/
theorem v5Files_of_checks {tags : Tags} {cf pf : ByteArray} {tables : List (String × ByteArray)} {count : ℕ}
    {draw : Option Lean.Json} {c : Circuit} {pi : HmIn.Ports} {pub : Public} {n : ℕ}
    (hc : HmOut.parse tags cf tables = .ok c) (hpi : HmIn.parse c = .ok pi)
    (hpub : HmIn.loadPublic tags c pi pf = .ok pub) (hn : pub.n = n)
    (hck : Flock.PortRoots.check tags cf pf tables = .ok ())
    (hks : Flock.KsScope.check tags cf pf tables count draw = .ok ()) :
    Flock.Registered.dupName c = none ∧ RootFixesDomainL c (LeafMap.ofPublic c pub n) := by
  obtain ⟨pubO, hO⟩ := portRoots_load hck hc
  obtain ⟨_, hdup, hcoll⟩ := Flock.PortRoots.check_ok hck hc hO
  obtain ⟨hnO, hdO, htO⟩ := loadPublic_inOut hpub hO
  refine ⟨hdup, ?_⟩
  cases hpt : pub.tables with
  | none =>
    rw [LeafMap.ofPublic_congr n hdO.symm htO.symm, LeafMap.ofPublic_none n (htO.trans hpt),
      rootFixesDomainL_ident, ← hn, ← hnO]
    exact fixes_of_collides_false hcoll
  | some t =>
    obtain ⟨_, pi', pub', _, hpi', hpub', _, hall⟩ := v5Files_of_check hks hc hO (htO.trans hpt)
    rw [hpi] at hpi'
    cases hpi'
    rw [hpub] at hpub'
    cases hpub'
    exact hall n

open FlockSoundness.Discharge.ZkHidden in
/-- **`verify --zk` accepts at `own`, files and transcript**: the live audit's transcript (`LiveAcceptsZKP`), the files'
rows, names and roots (`Flock.PortRoots.check`), and their form (`Flock.KsScope.check`), the last two refusals of
`verify --zk`'s verdict. -/
def LiveAcceptsKSP {n : ℕ} {Reg : Type} (I : FlockSoundness.Discharge.Exec.Inputs) (own : Option ByteArray)
    (o : Reg × Finset (Fin n) × FlockSoundness.Discharge.Exec.Transcript) : Prop :=
  LiveAcceptsZKP I own o ∧
    Flock.PortRoots.check (FlockSoundness.Discharge.ZkExec.zkTags I) I.circuitFile I.publicFile I.tables = .ok () ∧
    ∃ count draw,
      Flock.KsScope.check (FlockSoundness.Discharge.ZkExec.zkTags I) I.circuitFile I.publicFile I.tables count draw = .ok ()

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

open Classical in
/-- **`SampledKSLiveV5R` with its file preconditions read off acceptance**: its bound on the event that
`verify --zk` at `own` accepts the transcript and the files (`LiveAcceptsKSP`: their ports row/v1, row/v2 or row-seg/v1
rows of distinct names, no two registered roots one root under different domains, and the form, shared rows included,
`KsScope` admits), the sampled-proofs run passes the registration's population (`RunPopOk`: the descriptor `desc` and
the query `qr` give the game's `n` units in order), and the claims relation fails, under the sampling design the verifier samples
under from the registration (`Bridge.DrawsUnder`), no Bernoulli law (`Bridge.NotBernoulli`), at `δ` that sampling design's own miss
of `B`, as in `SampledKSLiveV5`. It has none of `SampledKSLiveV5R`'s hypotheses `RowsOk`, `dupName = none`, `RootFixesDomainL` and
`PopMatches`; the game's `n` is the public file's population (`pub.n = n`). It reads the circuit file flat
(`Flock.Desc.parse`), as `PortRoots` does, and `SampledKSLiveV5R` at it through `Flock.Desc.parseStmt_of_parse`, which
implies it (`sampledKSLiveV5Accepted_of`). Since acceptance gives the roots guard, its bound has no `ecr/sha-512` term
at the roots finder, which V5's has. -/
def SampledKSLiveV5Accepted : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    (I : Inputs) (own : Option ByteArray) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pi : HmIn.Ports) (hpi : HmIn.parse D.unroll = .ok pi)
    (pub : Public) (hpub : HmIn.loadPublic (zkTags I) D.unroll pi I.publicFile = .ok pub)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    -- the game's units are the public file's population
    (_hn : pub.n = n)
    -- the statement's descriptor, and the registration's query
    (desc : Claims.Msg) (qr : Lean.Json)
    -- the registration's law object and work table, the sampling design the verifier samples under from them, sampling kd proof
    -- blocks, and that it is no Bernoulli law
    (lj : Lean.Json) (tb : Option Lean.Json) (law : Flock.Draw.Law) (kd : ℕ)
    (hlaw : Bridge.DrawsUnder desc qr lj tb n law kd) (hb : Bridge.NotBernoulli law) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJP (inputsJ I 0) own (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJP (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the verifier's operating-system bytes, and A3
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (_hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the verify command's tags, and live-verifier at `own` and the sampling design
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJPLaw I own dj (fun S R => (planZHJP (Reg := Reg) dj y₀ S R).length) wr
      fun _ => law)
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hr : ∀ S R j, rateZ (planZHJP (Reg := Reg) dj y₀ S R) j k ≤ ρ) (t' : ℝ) (_ht' : 0 ≤ t')
    -- the proof blocks whose miss is `δ`
    (B : Finset ℕ)
    (τ : Strategy (auditLive (Bridge.execSel (n := n) src law n) Reg (liveOfZJ dg (planZHJP dj y₀) N wr)))
    (qF qS : ℝ)
    -- what the sampling statement carries and `RelS` doesn't read
    (q : Claims.Sampling.Query),
    let L := Bridge.execSel (n := n) src law n
    let P := (keyProgP D.unroll pi hU n).partition (Discharge.Layout.outsOfP D.unroll hU n pi)
    let vb := (rsZHJP dj y₀ ht (Flock.Desc.parseStmt_stmt (Flock.Desc.parseStmt_of_parse hD)) hscope hU hpi
        hpub).bindingZ
      (rsZHJP_computes dj y₀ ht (Flock.Desc.parseStmt_stmt (Flock.Desc.parseStmt_of_parse hD)) hscope hU hpi hpub)
    let σ := simZJA execArith hs512 (planZHJP dj y₀) (AcceptsZKP I own) wr (opnAtZHJP (planZHJP dj y₀) I own dj) dg
      (decAtZJ (planZHJP dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesZCJH_custodyP I own ht (Flock.Desc.parseStmt_stmt (Flock.Desc.parseStmt_of_parse hD)) law kd hdiv dj y₀ wr
        _ hTags hCust (Bridge.lawSized_size (Bridge.lawSized_drawsUnder hlaw hb) hCust.2)) τ
    let Xr := XplurZC execArith hs512 P (planZHJP dj y₀) (tabZHJP dj y₀) vb (reg σ) (cont σ) k Rw
    -- the deployed commitment's layout: the game's rows, read off the circuit file, at the public file's leaf map
    let Cc := (keyProgP D.unroll pi hU n).circuit (Discharge.Layout.outsOfP D.unroll hU n pi)
    let Lm := Bridge.LeafMap.ofPublic D.unroll pub n
    let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
      (Discharge.HiddenExec.rowOfHP D.unroll hU n pi (Discharge.Layout.outsOfP D.unroll hU n pi) g).map fun r =>
        ⟨r.2.1.val, Lm.ref r.2.1.val r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
    let roots := Bridge.rootsOfL D.unroll Lm
    let Cd := Bridge.commitDepL Cc rows D.unroll Lm roots
    -- the circuit's constants and every proof block's public input bits, public
    let κ : Fin Cc.N → Option Bool := fun g => Bridge.constsOfP D.unroll hU n pi pub g
    -- the sampling design's design, the registration's strata, and the sampling design's own miss of `B`
    let r := Properties.SampledProofs.oneStageOf law
    let ns := Bridge.nsOf desc qr
    let δ : ℝ := (Bridge.designOf r).Pa (Bridge.partOf P κ ns) B
    -- cr/sha-512 for the knowledge finders of each table the prover's session runs
    (∀ ω j, TableCRZ execArith hs512 (planZHJP dj y₀ (L.draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJP dj y₀) (cont σ ω)) qF qS) →
    -- ecr/sha-512 for the prover's link finders
    LinkCRZC execArith hs512 P (planZHJP dj y₀) (tabZHJP dj y₀) vb (reg σ) (cont σ) k Rw M t' →
    prob (fun o => (Bridge.LiveAcceptsKSP I own o ∧ Bridge.RunPopOk desc qr n) ∧
        ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ r desc q roots) (Bridge.partOf P κ ns)
          (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr))
        (auditLive L Reg (liveOfZJ dg (planZHJP dj y₀) N wr)) τ ≤
      ENNReal.ofReal δ +
        ksAvgStrictZ (planZHJP dj y₀) L k qF qS (reg σ) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZHJP dj y₀) (tabZHJP dj y₀) vb L p) : ℝ) / Fintype.card L.Ω) *
          (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ))

/-- **V5's bound under the roots guard gives `SampledKSLiveV5Accepted`**: at files and a registration that pass the
verifiers' checks they give `SampledKSLiveV5R`'s hypotheses (`rowsOk_of_scopeOkH`, `v5Files_of_checks` at `pub.n = n`,
`popMatches_of_popOk`), and the accepted event is inside its; at files that fail, it is empty. -/
theorem sampledKSLiveV5Accepted_of (h : SampledKSLiveV5R) : SampledKSLiveV5Accepted := by
  intro n Reg I own ht D hD hU pi hpi pub hpub hscope hn desc qr lj tb law kd hlaw hb hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb
    src hA3 hTags hCust k Rw M hk hRw hM ρ hρ hr t' ht' B τ qF qS q L P vb σ Xr Cc Lm rows roots Cd κ r ns δ hT hL
  by_cases hc : (Flock.PortRoots.check (zkTags I) I.circuitFile I.publicFile I.tables = .ok () ∧
      ∃ count draw, Flock.KsScope.check (zkTags I) I.circuitFile I.publicFile I.tables count draw = .ok ()) ∧
      Bridge.RunPopOk desc qr n
  · obtain ⟨⟨hck, _, _, hks⟩, p, hp, hpo⟩ := hc
    obtain ⟨hdup, hroots⟩ := v5Files_of_checks (Flock.Desc.parse_flat hD) hpi hpub hn hck hks
    exact (prob_mono (fun o ho => ⟨ho.1.1.1, ho.2⟩) _ τ).trans
      (h n Reg I own ht D (Flock.Desc.parseStmt_of_parse hD) hU pi hpi pub hpub hscope
        (Bridge.rowsOk_of_scopeOkH hscope) hdup hroots desc qr (Bridge.popMatches_of_popOk hp hpo) lj tb law kd hlaw hb
        hdiv dj S₀ d₀ y₀ wr dg N Ω Lb src hA3 hTags hCust k Rw M hk hRw hM ρ hρ hr t' ht' B τ qF qS q hT hL)
  · exact (prob_mono (q := fun _ => False) (fun o ho => hc ⟨⟨ho.1.1.2.1, ho.1.1.2.2⟩, ho.1.2⟩) _ τ).trans
      ((prob_false _ τ).le.trans zero_le)

/-- **The sampled audit's knowledge soundness at public inputs, on the event that the verifiers accept the files**:
no hypothesis on the files' rows, names or roots, or on the registration's population order. -/
theorem sampledKSLiveV5Accepted : SampledKSLiveV5Accepted :=
  sampledKSLiveV5Accepted_of sampledKSLiveV5R

end Flock.Bridge
