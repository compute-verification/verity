import Proofs.Flock.Bridge.Refused
import Proofs.Flock.Soundness.Discharge.FailClosed.Scope
import Proofs.Flock.Recursive.Defs
import Flock.KsScope

/-!
# `SampledKSLiveV3`'s file preconditions, refused by `verify --zk` (`SampledKSLiveV4`)

`SampledKSLiveV3` still takes six things of the files and the draw as hypotheses: an untyped statement, the unit a built
net, no shared rows, the circuit in `Layout.scopeOkH`, the draw's `k` a multiple of the tables, and the `--zk` tags'
hidden outputs, retained rounds and `hm96-sha512/v1` leaves (`TagsOkZHJ`). `verify --zk` now refuses every session
outside them (`Flock.KsScope.check`, after `Flock.PortRoots.check` in its verdict), so the guarantee reads them off
acceptance.

* `Bridge.KsScoped I D pub kd`: the six.
* `Bridge.AcceptsScoped I t`: `verify --zk`'s verdict on one session's transcript: `AcceptsZK`, and the files pass
  `Flock.PortRoots.check`, and the files and the record's draw pass `Flock.KsScope.check`. `Bridge.LiveAcceptsScoped I o`
  is it at the live audit's transcript.
* `Bridge.typed_of_check`: the check refuses a typed statement. `Bridge.built_of_parse`: the circuit file of an untyped
  statement that parses has a built unit.
* `Bridge.ksScoped_files`, `Bridge.div_of_drawWhy`: the check gives five of the six from the files, and the sixth, the
  draw's `k` a multiple of the tables, from the record's draw, which custody (`RecordCustodyZKJ`) makes the verifier's.
  `Bridge.ksScoped_of_accepts` puts them together: an accepted transcript the prover's session wrote is in `KsScoped`.
* `SampledKSLiveV4`: at files and a draw in `KsScoped`, `SampledKSLiveV3`'s bound on `LiveAcceptsScoped`'s event;
  outside them, `verify --zk` accepts with probability 0. It takes none of the six as a hypothesis.
  `sampledKSLiveV4_of` derives it from `SampledKSLiveV3`.
-/

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Discharge.Exec FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Composed
  FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)

/-- **`SampledKSLiveV3`'s file preconditions**: an untyped statement, the unit a built net, no shared rows, the circuit in
`Layout.scopeOkH`, the draw's `k` a multiple of the tables (`nTab`), and the `--zk` tags `TagsOkZHJ`. -/
structure KsScoped (I : Inputs) (D : Flock.Desc) (pub : Flock.Public) (kd : ℕ) : Prop where
  typed : I.tags.typed = false
  built : Built D.unroll.unit
  tables : pub.tables = none
  scope : FlockSoundness.Discharge.Layout.scopeOkH D.unroll = true
  div : kd % nTab I = 0
  tags : TagsOkZHJ I

/-- **`verify --zk` accepts one session's transcript**: `AcceptsZK`, the files' rows, names and roots
(`Flock.PortRoots.check`), and the tags, files and the record's draw (`Flock.KsScope.check`, the last refusal of
`verify --zk`'s verdict outside `--rec-inner`). -/
def AcceptsScoped (I : Inputs) (t : Transcript) : Prop :=
  AcceptsZK I t ∧ Flock.PortRoots.check (zkTags I) I.circuitFile I.publicFile I.tables = .ok () ∧
    Flock.KsScope.check (zkTags I) I.circuitFile I.publicFile I.tables I.count (recordDraw t.record) = .ok ()

/-- **`verify --zk` accepts the live audit's transcript** (`AcceptsScoped` at it). -/
def LiveAcceptsScoped {n : ℕ} {Reg : Type} (I : Inputs) (o : Reg × Finset (Fin n) × Transcript) : Prop :=
  AcceptsScoped I o.2.2

theorem LiveAcceptsScoped.ks {n : ℕ} {Reg : Type} {I : Inputs} {o : Reg × Finset (Fin n) × Transcript}
    (h : LiveAcceptsScoped I o) : LiveAcceptsKS I o :=
  ⟨h.1, h.2.1⟩

/-- **`Flock.KsScope.check` passing passes its tags check**, whatever the files. -/
theorem tagsWhy_of_check {tags : Flock.Tags} {cf pf : ByteArray} {tables : List (String × ByteArray)} {count : ℕ}
    {draw : Option Lean.Json} (h : Flock.KsScope.check tags cf pf tables count draw = .ok ()) :
    Flock.KsScope.tagsWhy tags = none := by
  unfold Flock.KsScope.check Flock.KsScope.checkOf at h
  cases ht : Flock.KsScope.tagsWhy tags with
  | none => rfl
  | some w => simp [ht] at h

/-- **`verify --zk` refuses a typed statement**: files that pass `Flock.KsScope.check` are of an untyped one. -/
theorem typed_of_check {I : Inputs} {draw : Option Lean.Json}
    (h : Flock.KsScope.check (zkTags I) I.circuitFile I.publicFile I.tables I.count draw = .ok ()) :
    I.tags.typed = false :=
  (zkTags_typed I).symm.trans (Flock.KsScope.tagsWhy_none (tagsWhy_of_check h)).1

/-- **An untyped statement's circuit file that parses has a built unit**: the parse builds every net it reads from
lines (`parse_facts`), and the unit is one (`unit_built`). -/
theorem built_of_parse {tags : Flock.Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {D : Flock.Desc}
    (ht : tags.typed = false) (hD : Flock.Desc.parse tags cf tables = .ok D) : Built D.unroll.unit := by
  have P := parse_facts (FlockSoundness.Discharge.Hidden.parse_untyped ht (Flock.Desc.parse_flat hD)).1
  obtain ⟨ls, hls⟩ := P.check
  exact unit_built P (check_facts P.typed hls)

/-- **The files that pass `Flock.KsScope.check` are in five of `KsScoped`'s six**, and the draw passes `drawWhy`. -/
theorem ksScoped_files {I : Inputs} {D : Flock.Desc} {pub : Flock.Public} {draw : Option Lean.Json}
    (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D)
    (hpub : Flock.HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub)
    (h : Flock.KsScope.check (zkTags I) I.circuitFile I.publicFile I.tables I.count draw = .ok ()) :
    I.tags.typed = false ∧ Built D.unroll.unit ∧ pub.tables = none ∧
      FlockSoundness.Discharge.Layout.scopeOkH D.unroll = true ∧ TagsOkZHJ I ∧
      Flock.KsScope.drawWhy I.count draw = none := by
  obtain ⟨ht, hs, hp, hd⟩ := Flock.KsScope.check_ok h (Flock.Desc.parse_flat hD) hpub
  obtain ⟨ht1, ht2, ht3, ht4⟩ := Flock.KsScope.tagsWhy_none ht
  exact ⟨typed_of_check h, built_of_parse ht1 hD, hp,
    (FlockSoundness.Discharge.FailClosed.scopeOkH_eq _).symm.trans hs,
    ⟨(zkTags_hiddenOutputs I).symm.trans ht2, (zkTags_retainRounds I).symm.trans ht3,
      (zkTags_merkleLeaf I).symm.trans ht4⟩, hd⟩

/-- **The record's draw passing `drawWhy` gives the draw's `k` a multiple of the tables**: custody makes an accepted
record's draw `dj S`, and a `dj S` that parses the verifier's own subset draw of `kd` units. -/
theorem div_of_drawWhy {n mPts : ℕ} {Reg : Type} {I : Inputs} {dj : Finset (Fin n) → Lean.Json}
    {Jf : Finset (Fin n) → Reg → ℕ} {wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (Jf S R) mPts → Transcript}
    {kd : ℕ} (hCust : FlockSoundness.Assumptions.Zk.RecordCustodyZKJ I dj Jf wr kd) {S : Finset (Fin n)} {R : Reg}
    {lo : LiveOutJ (Jf S R) mPts} (hacc : AcceptsZK I (wr S R lo))
    (hd : Flock.KsScope.drawWhy I.count (recordDraw (wr S R lo).record) = none) : kd % nTab I = 0 := by
  unfold nTab
  split
  · exact Nat.mod_one kd
  · next hc =>
    obtain ⟨j, d, hj, hjd, hk⟩ := Flock.KsScope.drawWhy_none hd hc
    rw [(hCust.1 S R lo hacc).1] at hj
    obtain rfl := Option.some.inj hj
    have hdS := hCust.2 S d hjd
    exact hk kd (by rw [hdS]; rfl)

/-- **An accepted transcript the prover's session wrote is in `KsScoped`**: its files pass `Flock.KsScope.check`
(`ksScoped_files`), and custody makes its record's draw the verifier's (`div_of_drawWhy`). -/
theorem ksScoped_of_accepts {n mPts : ℕ} {Reg : Type} {I : Inputs} {D : Flock.Desc} {pub : Flock.Public}
    {dj : Finset (Fin n) → Lean.Json} {Jf : Finset (Fin n) → Reg → ℕ}
    {wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (Jf S R) mPts → Transcript} {kd : ℕ}
    (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D)
    (hpub : Flock.HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub)
    (hCust : FlockSoundness.Assumptions.Zk.RecordCustodyZKJ I dj Jf wr kd) {S : Finset (Fin n)} {R : Reg}
    {lo : LiveOutJ (Jf S R) mPts} (h : AcceptsScoped I (wr S R lo)) : KsScoped I D pub kd := by
  obtain ⟨hty, hU, hpt, hsc, hTags, hd⟩ := ksScoped_files hD hpub h.2.2
  exact ⟨hty, hU, hpt, hsc, div_of_drawWhy hCust h.1 hd, hTags⟩

end Flock.Bridge

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

open Classical in
/-- **`SampledKSLiveV3` with its file preconditions read off acceptance**: on the event that `verify --zk` accepts the
transcript, the files and the record's draw (`Bridge.LiveAcceptsScoped`: `AcceptsZK`, `Flock.PortRoots.check` and
`Flock.KsScope.check`; v2's one-stage check is to run the same three), the sampled-proofs run passes the registration's population (`Bridge.RunPopOk`), and the claims
relation fails, at files and a draw in `Bridge.KsScoped` the bound is `SampledKSLiveV3`'s, and outside them the event
has probability 0. It has none of V3's hypotheses `I.tags.typed = false`, `Built D.unroll.unit`, `pub.tables = none`,
`scopeOkH D.unroll = true`, `kd % nTab I = 0` and `TagsOkZHJ I`. `SampledKSLiveV3` implies it
(`sampledKSLiveV4_of`). -/
def SampledKSLiveV4 : Prop :=
  ∀ (n : ℕ) (Reg : Type) (I : Inputs)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub)
    -- the game's units are the public file's population
    (_hn : pub.n = n)
    -- the statement's descriptor, and the registration's query
    (desc : Claims.Msg) (qr : Lean.Json)
    (kd : ℕ) (_hkd : kd ≤ n)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the verifier's operating-system bytes, and A3
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (_hA3 : Assumptions.UniformRandomBytes Lb src)
    -- live-verifier
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) (t' : ℝ) (_ht' : 0 ≤ t') (K₀ : ℕ) (_hK₀ : K₀ ≤ n)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    (qF qS : ℝ)
    -- what the sampling statement carries and `RelS` doesn't read
    (q : Claims.Sampling.Query),
    if hs : Bridge.KsScoped I D pub kd then
      let L := Law.execOS (n := n) src (.subset kd) n
      let P := (keyProg D.unroll hs.built n).partition (Discharge.Layout.outsOf D.unroll hs.built n)
      let vb := (rsZHJ dj y₀ hs.typed (Flock.Desc.parse_flat hD) hs.scope hs.built hpub hs.tables).bindingZ
        (rsZHJ_computes dj y₀ hs.typed (Flock.Desc.parse_flat hD) hs.scope hs.built hpub hs.tables)
      let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
        (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
        (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_live I hs.typed (Flock.Desc.parse_flat hD)
            (Discharge.Layout.scopeOkH_spec hs.built hs.scope).2.1 dj y₀ wr _ hs.tags hCust.1
            fun _ => drawOkZKHJ_of_file I hs.typed (Flock.Desc.parse_flat hD) kd n hs.div dj hCust.2 _)) τ
      let Xr := XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw
      let δ : ℝ := ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)
      -- the deployed commitment's layout: the game's rows, read off the circuit file
      let Cc := (keyProg D.unroll hs.built n).circuit (Discharge.Layout.outsOf D.unroll hs.built n)
      let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
        (FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hs.built n
          (Discharge.Layout.outsOf D.unroll hs.built n) g).map fun r =>
          ⟨r.2.1.val, r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
      -- the registration's frame-v3 roots, over the public file's rows
      let roots := Bridge.rootsOf D.unroll pub n
      let Cd := Bridge.commitDep Cc rows D.unroll n roots
      -- the circuit's constants, public
      let κ : Fin Cc.N → Option Bool := fun g => Bridge.constsOf D.unroll hs.built n g
      -- cr/sha-512 for the knowledge finders of each table the prover's session runs
      (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ (L.draw ω) (reg σ)) j
        (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
      -- ecr/sha-512 for the prover's link finders
      LinkCRZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw M t' →
      prob (fun o => (Bridge.LiveAcceptsScoped I o ∧ Bridge.RunPopOk desc qr n) ∧
          ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P)
            (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr))
          (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
        ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) +
          ksAvgStrictZ (planZHJ dj y₀) L k qF qS (reg σ) +
          ENNReal.ofReal ((∑ p : vb.Pos,
              (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω) *
            (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ))
    else
      prob (fun o => Bridge.LiveAcceptsScoped I o ∧ Bridge.RunPopOk desc qr n)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ = 0

/-- **`SampledKSLiveV3` gives `SampledKSLiveV4`**: at files and a draw in `KsScoped`, V4's event is inside V3's at
`KsScoped`'s six; outside them no outcome the prover reaches is accepted (`Bridge.ksScoped_of_accepts`). -/
theorem sampledKSLiveV4_of (h : SampledKSLiveV3) : SampledKSLiveV4 := by
  intro n Reg I D hD pub hpub hn desc qr kd hkd dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src hA3 hCust k Rw M hk hRw hM ρ hρ hr
    t' ht' K₀ hK₀ τ qF qS q
  split
  · next hs =>
    intro L P vb σ Xr δ Cc rows roots Cd κ hT hL
    exact (prob_mono (fun o ho => ⟨⟨ho.1.1.ks, ho.1.2⟩, ho.2⟩) _ τ).trans
      (h n Reg I hs.typed D hD hs.built pub hpub hs.tables hs.scope hn desc qr kd hkd hs.div dj S₀ d₀ y₀ wr dg N Ω Lb
        src hA3 hs.tags hCust k Rw M hk hRw hM ρ hρ hr t' ht' K₀ hK₀ τ qF qS q hT hL)
  · next hs =>
    refine nonpos_iff_eq_zero.1 (((FlockSoundness.Game.prob_mono_outcome (q := fun _ => False) _ τ
      fun o ho he => hs ?_).trans (prob_false _ τ).le))
    unfold auditLive at ho
    obtain ⟨ω, hω⟩ := ho
    obtain ⟨t, ht, rfl⟩ := Game.Outcome.of_map _ _ _ hω
    unfold liveOfZJ at ht
    obtain ⟨lo, -, rfl⟩ := Game.Outcome.of_map _ _ _ ht
    exact Bridge.ksScoped_of_accepts hD hpub hCust he.1

end Flock.Guarantees
