module

public import Proofs.Flock.Bridge.Draw
public import Proofs.Flock.Bridge.PopMatches
public import Proofs.Flock.Claims.Sampling.Select.Design
public import Spot.OneStage
import Proofs.Flock.Claims.Sampling.Select.Misses
import Proofs.Flock.Claims.Sampling.Select.Take

/-!
<<<<<<< ours
# The sampling design the audit samples under (`Bridge.drawnSamplingDesign`)
||||||| base
# The sampling design the audit samples under (`Bridge.drawnLaw`)
=======
# The sampling design the audit samples under (`Bridge.sampledLaw`)
>>>>>>> theirs

<<<<<<< ours
The deployed sample (`Spot.OneStage.drawnFrom`) samples under the sampling design the registration's sampling design object expands to
over its strata (`Draw.deriveSamplingDesign` over `Partition.strata`), else the sampling design `flockSamplingDesign` parses from the sampling design object,
||||||| base
The deployed sample (`Spot.OneStage.drawnFrom`) samples under the sampling design the registration's sampling design object expands to
over its strata (`Draw.deriveLaw` over `Partition.strata`), else the sampling design `flockLaw` parses from the sampling design object,
=======
The deployed sample (`Spot.OneStage.sampledFrom`) samples under the sampling design the registration's sampling design object expands to
over its strata (`Draw.deriveLaw` over `Partition.strata`), else the sampling design `flockLaw` parses from the sampling design object,
>>>>>>> theirs
and refuses a Bernoulli one (`OneStage.coveredDesign`).
<<<<<<< ours
`drawnSamplingDesign desc q lj tb n` is that sampling design, from the registration's descriptor bytes `desc`, query `q`, sampling design object `lj` and
||||||| base
`drawnLaw desc q lj tb n` is that sampling design, from the registration's descriptor bytes `desc`, query `q`, sampling design object `lj` and
=======
`sampledLaw desc q lj tb n` is that sampling design, from the registration's descriptor bytes `desc`, query `q`, sampling design object `lj` and
>>>>>>> theirs
work table `tb`, at the population `n`:

<<<<<<< ours
* `drawnFrom_drawnSamplingDesign`: the deployed sample is `drawUnits` under `drawnSamplingDesign`, at a verifier whose program `desc` parses to;
* `notBernoulli_sampledDesign`, `notBernoulli_drawsUnder`: it is no Bernoulli sampling design (`NotBernoulli`);
* `samplingDesignK samplingDesign`: the units a sampling design samples, a subset sampling design's `k` or its strata's `k`s summed;
* `DrawsUnder desc q lj tb n samplingDesign kd`: the verifier samples `kd` units under `samplingDesign`: `samplingDesign` is `drawnSamplingDesign`'s, and `kd` its
  `samplingDesignK`; it holds at every sample the one-stage verifier samples (`drawsUnder_of_drawnFrom`);
* `drawnSamplingDesign_cases`: `drawnSamplingDesign` is a subset sampling design within the population, a Bernoulli sampling design, or the stratified or work sampling design
||||||| base
* `drawnFrom_drawnLaw`: the deployed sample is `drawUnits` under `drawnLaw`, at a verifier whose program `desc` parses to;
* `notBernoulli_sampledDesign`, `notBernoulli_drawsUnder`: it is no Bernoulli sampling design (`NotBernoulli`);
* `lawK law`: the units a sampling design samples, a subset sampling design's `k` or its strata's `k`s summed;
* `DrawsUnder desc q lj tb n law kd`: the verifier samples `kd` units under `law`: `law` is `drawnLaw`'s, and `kd` its
  `lawK`; it holds at every sample the one-stage verifier samples (`drawsUnder_of_drawnFrom`);
* `drawnLaw_cases`: `drawnLaw` is a subset sampling design within the population, a Bernoulli sampling design, or the stratified or work sampling design
=======
* `sampledFrom_sampledLaw`: the deployed sample is `sampleUnits` under `sampledLaw`, at a verifier whose program `desc` parses to;
* `notBernoulli_sampledDesign`, `notBernoulli_samplesUnder`: it is no Bernoulli sampling design (`NotBernoulli`);
* `lawK law`: the units a sampling design samples, a subset sampling design's `k` or its strata's `k`s summed;
* `SamplesUnder desc q lj tb n law kd`: the verifier samples `kd` units under `law`: `law` is `sampledLaw`'s, and `kd` its
  `lawK`; it holds at every sample the one-stage verifier samples (`samplesUnder_of_sampledFrom`);
* `sampledLaw_cases`: `sampledLaw` is a subset sampling design within the population, a Bernoulli sampling design, or the stratified or work sampling design
>>>>>>> theirs
  `Flock.Draw.derive` or `deriveWork` expands over the registration's strata;
* `strata_pos`, `strata_nodup`: those strata each hold a unit, and their slots hold each unit once;
* `popMatches_strata`: at `PopMatches`, `nsOf` is those strata's sizes and their slots are the units `0, …, n − 1`;
<<<<<<< ours
* `samplingDesignSized_drawnSamplingDesign`, `samplingDesignSized_drawsUnder`: a `drawnSamplingDesign` that is no Bernoulli sampling design (`NotBernoulli`) is sized
  (`SamplingDesignSized`) at `samplingDesignK`, at any population;
* `samplingDesignOn_drawnSamplingDesign`, `samplingDesignOn_drawsUnder`: at `PopMatches`, it is on the registration's strata (`SamplingDesignOn` at `nsOf`), sampling
  `samplingDesignK` units;
* `designAt desc q lj tb n`: `drawnSamplingDesign`'s sampling design, or `subset 0` where it errs (`designAt_cases`), under which every sample
  `drawnFrom` samples is sampled (`drawnFrom_designAt`), no Bernoulli sampling design (`notBernoulli_designAt`) and sized (`sized_designAt`).
||||||| base
* `lawSized_drawnLaw`, `lawSized_drawsUnder`: a `drawnLaw` that is no Bernoulli sampling design (`NotBernoulli`) is sized
  (`LawSized`) at `lawK`, at any population;
* `lawOn_drawnLaw`, `lawOn_drawsUnder`: at `PopMatches`, it is on the registration's strata (`LawOn` at `nsOf`), sampling
  `lawK` units;
* `designAt desc q lj tb n`: `drawnLaw`'s sampling design, or `subset 0` where it errs (`designAt_cases`), under which every sample
  `drawnFrom` samples is sampled (`drawnFrom_designAt`), no Bernoulli sampling design (`notBernoulli_designAt`) and sized (`sized_designAt`).
=======
* `lawSized_sampledLaw`, `lawSized_samplesUnder`: a `sampledLaw` that is no Bernoulli sampling design (`NotBernoulli`) is sized
  (`LawSized`) at `lawK`, at any population;
* `lawOn_sampledLaw`, `lawOn_samplesUnder`: at `PopMatches`, it is on the registration's strata (`LawOn` at `nsOf`), sampling
  `lawK` units;
* `designAt desc q lj tb n`: `sampledLaw`'s sampling design, or `subset 0` where it errs (`designAt_cases`), under which every sample
  `sampledFrom` samples is sampled (`sampledFrom_designAt`), no Bernoulli sampling design (`notBernoulli_designAt`) and sized (`sized_designAt`).
>>>>>>> theirs
-/

@[expose] public section

namespace Flock.Bridge

open Properties.Spot

/-- **The sampling design the audit samples under**: the registration's sampling design object `lj` expanded over the strata of the descriptor
<<<<<<< ours
`desc` under the query `q` (`Draw.deriveSamplingDesign`, with the work table `tb`), else the sampling design `flockSamplingDesign` parses from it at the
population `n`, a Bernoulli one refused, as `OneStage.drawnFrom` samples (`drawnFrom_drawnSamplingDesign`). -/
def drawnSamplingDesign (desc : Claims.Msg) (q lj : Lean.Json) (tb : Option Lean.Json) (n : ℕ) : Except String Flock.Draw.SamplingDesign := do
||||||| base
`desc` under the query `q` (`Draw.deriveLaw`, with the work table `tb`), else the sampling design `flockLaw` parses from it at the
population `n`, a Bernoulli one refused, as `OneStage.drawnFrom` samples (`drawnFrom_drawnLaw`). -/
def drawnLaw (desc : Claims.Msg) (q lj : Lean.Json) (tb : Option Lean.Json) (n : ℕ) : Except String Flock.Draw.Law := do
=======
`desc` under the query `q` (`Draw.deriveLaw`, with the work table `tb`), else the sampling design `flockLaw` parses from it at the
population `n`, a Bernoulli one refused, as `OneStage.sampledFrom` samples (`sampledFrom_sampledLaw`). -/
def sampledLaw (desc : Claims.Msg) (q lj : Lean.Json) (tb : Option Lean.Json) (n : ℕ) : Except String Flock.Draw.Law := do
>>>>>>> theirs
  let p ← Spot.Program.parse ⟨desc.toArray⟩
  let derived ← Spot.Draw.deriveSamplingDesign p q lj tb
  let parsed ← Spot.Draw.flockSamplingDesign (SecurityProofs.samplingDesignObj lj derived) n
  Spot.OneStage.coveredDesign (derived.getD parsed)

/-- `OneStage.coveredDesign` refuses a Bernoulli sampling design. -/
theorem coveredDesign_bernoulli (a b : ℕ) :
    Spot.OneStage.coveredDesign (.bernoulli a b) =
      .error "the law is bernoulli, which the sampled audit's soundness statement doesn't cover" :=
  rfl

/-- `OneStage.coveredDesign` passes exactly the sampling designs that are no Bernoulli sampling design, as they are. -/
theorem coveredDesign_ok {l samplingDesign : Flock.Draw.SamplingDesign} (h : Spot.OneStage.coveredDesign l = .ok samplingDesign) :
    samplingDesign = l ∧ NotBernoulli l := by
  cases l
  case bernoulli a b =>
    rw [coveredDesign_bernoulli] at h
    cases h
  all_goals exact ⟨(Except.ok.inj h).symm, fun _ _ h' => by cases h'⟩

/-- **The units a sampling design samples**: a subset sampling design's `k`, a stratified or work sampling design's strata's `k`s summed; none for a Bernoulli
sampling design, whose count the coins set. -/
def samplingDesignK : Flock.Draw.SamplingDesign → ℕ
  | .subset k => k
  | .bernoulli _ _ => 0
  | .stratified _ St => (St.toList.map (·.k)).sum
  | .work _ ws => (ws.toList.map (·.k)).sum

<<<<<<< ours
/-- **The deployed sample is `drawUnits` under `drawnSamplingDesign`**: at a verifier whose program the descriptor `desc` parses to,
under the sampling design object `Draw.derive` writes from its own, `drawnFrom` samples under the sampling design `drawnSamplingDesign` reads off its query,
||||||| base
/-- **The deployed sample is `drawUnits` under `drawnLaw`**: at a verifier whose program the descriptor `desc` parses to,
under the sampling design object `Draw.derive` writes from its own, `drawnFrom` samples under the sampling design `drawnLaw` reads off its query,
=======
/-- **The deployed sample is `sampleUnits` under `sampledLaw`**: at a verifier whose program the descriptor `desc` parses to,
under the sampling design object `Draw.derive` writes from its own, `sampledFrom` samples under the sampling design `sampledLaw` reads off its query,
>>>>>>> theirs
sampling design object and work table. -/
<<<<<<< ours
theorem drawnFrom_drawnSamplingDesign {desc : Claims.Msg} {v : Spot.Registration.Verifier} {lo : Lean.Json}
||||||| base
theorem drawnFrom_drawnLaw {desc : Claims.Msg} {v : Spot.Registration.Verifier} {lo : Lean.Json}
=======
theorem sampledFrom_sampledLaw {desc : Claims.Msg} {v : Spot.Registration.Verifier} {lo : Lean.Json}
>>>>>>> theirs
    (hp : Spot.Program.parse ⟨desc.toArray⟩ = .ok v.program)
<<<<<<< ours
    (hlo : Spot.Draw.derive v.program v.query v.samplingDesign v.workTable = .ok lo) (n : ℕ) (src : ByteArray) :
    Spot.OneStage.drawnFrom v lo n src =
      (drawnSamplingDesign desc v.query v.samplingDesign v.workTable n >>= fun samplingDesign => Spot.Draw.drawUnits samplingDesign n src) := by
||||||| base
    (hlo : Spot.Draw.derive v.program v.query v.law v.workTable = .ok lo) (n : ℕ) (src : ByteArray) :
    Spot.OneStage.drawnFrom v lo n src =
      (drawnLaw desc v.query v.law v.workTable n >>= fun law => Spot.Draw.drawUnits law n src) := by
=======
    (hlo : Spot.Draw.derive v.program v.query v.law v.workTable = .ok lo) (n : ℕ) (src : ByteArray) :
    Spot.OneStage.sampledFrom v lo n src =
      (sampledLaw desc v.query v.law v.workTable n >>= fun law => Spot.Draw.sampleUnits law n src) := by
>>>>>>> theirs
  unfold Spot.Draw.derive at hlo
  obtain ⟨derived, hd, hlo⟩ := SecurityProofs.bind_eq_ok.1 hlo
<<<<<<< ours
  obtain rfl : SecurityProofs.samplingDesignObj v.samplingDesign derived = lo := Except.ok.inj hlo
  unfold Spot.OneStage.drawnFrom drawnSamplingDesign
||||||| base
  obtain rfl : SecurityProofs.lawObj v.law derived = lo := Except.ok.inj hlo
  unfold Spot.OneStage.drawnFrom drawnLaw
=======
  obtain rfl : SecurityProofs.lawObj v.law derived = lo := Except.ok.inj hlo
  unfold Spot.OneStage.sampledFrom sampledLaw
>>>>>>> theirs
  rw [hp, SecurityProofs.bind_ok, hd, SecurityProofs.bind_ok, SecurityProofs.bind_ok]
  cases Spot.Draw.flockSamplingDesign (SecurityProofs.samplingDesignObj v.samplingDesign derived) n <;> rfl

<<<<<<< ours
/-- **The verifier samples `kd` units under `samplingDesign`**: `samplingDesign` is the sampling design it samples under from the registration's descriptor
`desc`, query `q`, sampling design object `lj` and work table `tb` at the population `n` (`drawnSamplingDesign`), and `kd` the units it samples
(`samplingDesignK`). -/
def DrawsUnder (desc : Claims.Msg) (q lj : Lean.Json) (tb : Option Lean.Json) (n : ℕ) (samplingDesign : Flock.Draw.SamplingDesign) (kd : ℕ) :
||||||| base
/-- **The verifier samples `kd` units under `law`**: `law` is the sampling design it samples under from the registration's descriptor
`desc`, query `q`, sampling design object `lj` and work table `tb` at the population `n` (`drawnLaw`), and `kd` the units it samples
(`lawK`). -/
def DrawsUnder (desc : Claims.Msg) (q lj : Lean.Json) (tb : Option Lean.Json) (n : ℕ) (law : Flock.Draw.Law) (kd : ℕ) :
=======
/-- **The verifier samples `kd` units under `law`**: `law` is the sampling design it samples under from the registration's descriptor
`desc`, query `q`, sampling design object `lj` and work table `tb` at the population `n` (`sampledLaw`), and `kd` the units it samples
(`lawK`). -/
def SamplesUnder (desc : Claims.Msg) (q lj : Lean.Json) (tb : Option Lean.Json) (n : ℕ) (law : Flock.Draw.Law) (kd : ℕ) :
>>>>>>> theirs
    Prop :=
<<<<<<< ours
  drawnSamplingDesign desc q lj tb n = .ok samplingDesign ∧ samplingDesignK samplingDesign = kd
||||||| base
  drawnLaw desc q lj tb n = .ok law ∧ lawK law = kd
=======
  sampledLaw desc q lj tb n = .ok law ∧ lawK law = kd
>>>>>>> theirs

/-- **Every sample the one-stage verifier samples is under a sampling design `SamplesUnder` names**: where `sampledFrom` samples,
at a verifier whose program the descriptor `desc` parses to and under the sampling design object `Draw.derive` writes,
<<<<<<< ours
its sample is `drawUnits` under a `samplingDesign` with `DrawsUnder desc v.query v.samplingDesign v.workTable n samplingDesign (samplingDesignK samplingDesign)`. -/
theorem drawsUnder_of_drawnFrom {desc : Claims.Msg} {v : Spot.Registration.Verifier} {lo : Lean.Json}
||||||| base
its sample is `drawUnits` under a `law` with `DrawsUnder desc v.query v.law v.workTable n law (lawK law)`. -/
theorem drawsUnder_of_drawnFrom {desc : Claims.Msg} {v : Spot.Registration.Verifier} {lo : Lean.Json}
=======
its sample is `sampleUnits` under a `law` with `SamplesUnder desc v.query v.law v.workTable n law (lawK law)`. -/
theorem samplesUnder_of_sampledFrom {desc : Claims.Msg} {v : Spot.Registration.Verifier} {lo : Lean.Json}
>>>>>>> theirs
    (hp : Spot.Program.parse ⟨desc.toArray⟩ = .ok v.program)
<<<<<<< ours
    (hlo : Spot.Draw.derive v.program v.query v.samplingDesign v.workTable = .ok lo) {n : ℕ} {src : ByteArray}
    {r : Array ℕ × ℕ} (h : Spot.OneStage.drawnFrom v lo n src = .ok r) :
    ∃ samplingDesign, DrawsUnder desc v.query v.samplingDesign v.workTable n samplingDesign (samplingDesignK samplingDesign) ∧
      Spot.Draw.drawUnits samplingDesign n src = .ok r := by
  rw [drawnFrom_drawnSamplingDesign hp hlo] at h
  obtain ⟨samplingDesign, hl, hd⟩ := SecurityProofs.bind_eq_ok.1 h
  exact ⟨samplingDesign, ⟨hl, rfl⟩, hd⟩

/-- `drawnSamplingDesign`'s steps: the descriptor's program, the sampling design `deriveSamplingDesign` expands over its strata, if any, and the sampling design
`flockSamplingDesign` parses from the sampling design object `Draw.derive` writes. -/
theorem drawnSamplingDesign_ok {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ} {samplingDesign : Flock.Draw.SamplingDesign}
    (h : drawnSamplingDesign desc q lj tb n = .ok samplingDesign) :
||||||| base
    (hlo : Spot.Draw.derive v.program v.query v.law v.workTable = .ok lo) {n : ℕ} {src : ByteArray}
    {r : Array ℕ × ℕ} (h : Spot.OneStage.drawnFrom v lo n src = .ok r) :
    ∃ law, DrawsUnder desc v.query v.law v.workTable n law (lawK law) ∧
      Spot.Draw.drawUnits law n src = .ok r := by
  rw [drawnFrom_drawnLaw hp hlo] at h
  obtain ⟨law, hl, hd⟩ := SecurityProofs.bind_eq_ok.1 h
  exact ⟨law, ⟨hl, rfl⟩, hd⟩

/-- `drawnLaw`'s steps: the descriptor's program, the sampling design `deriveLaw` expands over its strata, if any, and the sampling design
`flockLaw` parses from the sampling design object `Draw.derive` writes. -/
theorem drawnLaw_ok {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ} {law : Flock.Draw.Law}
    (h : drawnLaw desc q lj tb n = .ok law) :
=======
    (hlo : Spot.Draw.derive v.program v.query v.law v.workTable = .ok lo) {n : ℕ} {src : ByteArray}
    {r : Array ℕ × ℕ} (h : Spot.OneStage.sampledFrom v lo n src = .ok r) :
    ∃ law, SamplesUnder desc v.query v.law v.workTable n law (lawK law) ∧
      Spot.Draw.sampleUnits law n src = .ok r := by
  rw [sampledFrom_sampledLaw hp hlo] at h
  obtain ⟨law, hl, hd⟩ := SecurityProofs.bind_eq_ok.1 h
  exact ⟨law, ⟨hl, rfl⟩, hd⟩

/-- `sampledLaw`'s steps: the descriptor's program, the sampling design `deriveLaw` expands over its strata, if any, and the sampling design
`flockLaw` parses from the sampling design object `Draw.derive` writes. -/
theorem sampledLaw_ok {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ} {law : Flock.Draw.Law}
    (h : sampledLaw desc q lj tb n = .ok law) :
>>>>>>> theirs
    ∃ p derived parsed, Spot.Program.parse ⟨desc.toArray⟩ = .ok p ∧
<<<<<<< ours
      Spot.Draw.deriveSamplingDesign p q lj tb = .ok derived ∧
      Spot.Draw.flockSamplingDesign (SecurityProofs.samplingDesignObj lj derived) n = .ok parsed ∧ samplingDesign = derived.getD parsed := by
  unfold drawnSamplingDesign at h
||||||| base
      Spot.Draw.deriveLaw p q lj tb = .ok derived ∧
      Spot.Draw.flockLaw (SecurityProofs.lawObj lj derived) n = .ok parsed ∧ law = derived.getD parsed := by
  unfold drawnLaw at h
=======
      Spot.Draw.deriveLaw p q lj tb = .ok derived ∧
      Spot.Draw.flockLaw (SecurityProofs.lawObj lj derived) n = .ok parsed ∧ law = derived.getD parsed := by
  unfold sampledLaw at h
>>>>>>> theirs
  obtain ⟨p, hp, h⟩ := SecurityProofs.bind_eq_ok.1 h
  obtain ⟨derived, hd, h⟩ := SecurityProofs.bind_eq_ok.1 h
  obtain ⟨parsed, hpar, h⟩ := SecurityProofs.bind_eq_ok.1 h
  exact ⟨p, derived, parsed, hp, hd, hpar, (coveredDesign_ok h).1⟩

<<<<<<< ours
/-- **The audit samples under no Bernoulli sampling design**: `drawnSamplingDesign` refuses one (`OneStage.coveredDesign`). -/
||||||| base
/-- **The audit samples under no Bernoulli sampling design**: `drawnLaw` refuses one (`OneStage.coveredDesign`). -/
=======
/-- **The audit samples under no Bernoulli sampling design**: `sampledLaw` refuses one (`OneStage.coveredDesign`). -/
>>>>>>> theirs
theorem notBernoulli_sampledDesign {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ}
<<<<<<< ours
    {samplingDesign : Flock.Draw.SamplingDesign} (h : drawnSamplingDesign desc q lj tb n = .ok samplingDesign) : NotBernoulli samplingDesign := by
  unfold drawnSamplingDesign at h
||||||| base
    {law : Flock.Draw.Law} (h : drawnLaw desc q lj tb n = .ok law) : NotBernoulli law := by
  unfold drawnLaw at h
=======
    {law : Flock.Draw.Law} (h : sampledLaw desc q lj tb n = .ok law) : NotBernoulli law := by
  unfold sampledLaw at h
>>>>>>> theirs
  obtain ⟨_, -, h⟩ := SecurityProofs.bind_eq_ok.1 h
  obtain ⟨_, -, h⟩ := SecurityProofs.bind_eq_ok.1 h
  obtain ⟨_, -, h⟩ := SecurityProofs.bind_eq_ok.1 h
  obtain ⟨rfl, hb⟩ := coveredDesign_ok h
  exact hb

<<<<<<< ours
/-- **`drawnSamplingDesign`'s cases**: a subset sampling design within the population, a Bernoulli sampling design, or the sampling design `Flock.Draw.derive` or
||||||| base
/-- **`drawnLaw`'s cases**: a subset sampling design within the population, a Bernoulli sampling design, or the sampling design `Flock.Draw.derive` or
=======
/-- **`sampledLaw`'s cases**: a subset sampling design within the population, a Bernoulli sampling design, or the sampling design `Flock.Draw.derive` or
>>>>>>> theirs
`deriveWork` expands over the registration's strata (`Partition.strata`). -/
<<<<<<< ours
theorem drawnSamplingDesign_cases {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ} {samplingDesign : Flock.Draw.SamplingDesign}
    (h : drawnSamplingDesign desc q lj tb n = .ok samplingDesign) :
    (∃ k, samplingDesign = .subset k ∧ k ≤ n) ∨ (∃ a b, samplingDesign = .bernoulli a b) ∨
||||||| base
theorem drawnLaw_cases {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ} {law : Flock.Draw.Law}
    (h : drawnLaw desc q lj tb n = .ok law) :
    (∃ k, law = .subset k ∧ k ≤ n) ∨ (∃ a b, law = .bernoulli a b) ∨
=======
theorem sampledLaw_cases {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ} {law : Flock.Draw.Law}
    (h : sampledLaw desc q lj tb n = .ok law) :
    (∃ k, law = .subset k ∧ k ≤ n) ∨ (∃ a b, law = .bernoulli a b) ∨
>>>>>>> theirs
      ∃ p S k, Spot.Program.parse ⟨desc.toArray⟩ = .ok p ∧ Spot.Partition.strata p q = .ok S ∧
<<<<<<< ours
        (Flock.Draw.derive k S = .ok samplingDesign ∨ ∃ tab, Flock.Draw.deriveWork k S tab = .ok samplingDesign) := by
  obtain ⟨p, derived, parsed, hp, hd, hpar, rfl⟩ := drawnSamplingDesign_ok h
  rcases SecurityProofs.deriveSamplingDesign_cases p q lj tb derived hd with ⟨rfl, hk1, hk2⟩ | ⟨S, k, samplingDesign', hS, rfl, hlaw⟩
  · rcases SecurityProofs.flockSamplingDesign_cases _ _ _ hpar with ⟨k, rfl, hk, -⟩ | ⟨a, b, rfl, -⟩ | hk | hk
||||||| base
        (Flock.Draw.derive k S = .ok law ∨ ∃ tab, Flock.Draw.deriveWork k S tab = .ok law) := by
  obtain ⟨p, derived, parsed, hp, hd, hpar, rfl⟩ := drawnLaw_ok h
  rcases SecurityProofs.deriveLaw_cases p q lj tb derived hd with ⟨rfl, hk1, hk2⟩ | ⟨S, k, law', hS, rfl, hlaw⟩
  · rcases SecurityProofs.flockLaw_cases _ _ _ hpar with ⟨k, rfl, hk, -⟩ | ⟨a, b, rfl, -⟩ | hk | hk
=======
        (Flock.Draw.derive k S = .ok law ∨ ∃ tab, Flock.Draw.deriveWork k S tab = .ok law) := by
  obtain ⟨p, derived, parsed, hp, hd, hpar, rfl⟩ := sampledLaw_ok h
  rcases SecurityProofs.deriveLaw_cases p q lj tb derived hd with ⟨rfl, hk1, hk2⟩ | ⟨S, k, law', hS, rfl, hlaw⟩
  · rcases SecurityProofs.flockLaw_cases _ _ _ hpar with ⟨k, rfl, hk, -⟩ | ⟨a, b, rfl, -⟩ | hk | hk
>>>>>>> theirs
    · exact Or.inl ⟨k, rfl, hk⟩
    · exact Or.inr (Or.inl ⟨a, b, rfl⟩)
    · exact absurd hk hk1
    · exact absurd hk hk2
  · refine Or.inr (Or.inr ⟨p, S, k, hp, hS, ?_⟩)
    rcases hlaw with hder | ⟨-, tab, -, hwork⟩
    · exact Or.inl hder
    · exact Or.inr ⟨tab, hwork⟩

/-- The registration's strata are the evaluated query's ranges, by template (`Partition.strata`). -/
theorem strata_evaluate {p : Spot.Program} {q : Lean.Json} {S : Array (String × Array (ℕ × ℕ))}
    (hS : Spot.Partition.strata p q = .ok S) :
    ∃ rs, Spot.Partition.evaluate p q = .ok rs ∧
      S = Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count)) := by
  unfold Spot.Partition.strata at hS
  obtain ⟨rs, he, h⟩ := SecurityProofs.bind_eq_ok.1 hS
  exact ⟨rs, he, (Except.ok.inj h).symm⟩

/-- **The registration's strata each hold a unit**: `TemplateQuery.evaluate` refuses a template with no instance
(`strataOf_pos`, `partition_covered`). -/
theorem strata_pos {p : Spot.Program} {q : Lean.Json} {S : Array (String × Array (ℕ × ℕ))}
    (hS : Spot.Partition.strata p q = .ok S) :
    ∀ s ∈ S.toList, 1 ≤ (SecurityProofs.expand s.2.toList).length := by
  obtain ⟨rs, he, rfl⟩ := strata_evaluate hS
  exact SecurityProofs.strataOf_pos _ (SecurityProofs.partition_covered p q rs he)

/-- **The registration's strata's slots hold each unit once**: they are the evaluated ranges, which tile the population
(`strataOf_perm`, `partition_evaluate_tiles`). -/
theorem strata_nodup {p : Spot.Program} {q : Lean.Json} {S : Array (String × Array (ℕ × ℕ))}
    (hS : Spot.Partition.strata p q = .ok S) :
    (S.toList.flatMap fun s => SecurityProofs.expand s.2.toList).Nodup := by
  obtain ⟨rs, he, rfl⟩ := strata_evaluate hS
  exact (SecurityProofs.strataOf_perm _ _ (SecurityProofs.partition_evaluate_tiles p q rs he)).nodup_iff.2
    List.nodup_range'

/-- Strata whose ranges are another list's, in order, have its slots. -/
theorem flatMap_expand_eq {α : Type} {f : α → Array (ℕ × ℕ)} {l : List α} {S : List (String × Array (ℕ × ℕ))}
    (h : l.map f = S.map (·.2)) :
    l.flatMap (fun a => SecurityProofs.expand (f a).toList) = S.flatMap fun s => SecurityProofs.expand s.2.toList := by
  have h1 := congrArg (fun m => m.flatMap fun u : Array (ℕ × ℕ) => SecurityProofs.expand u.toList) h
  simpa only [List.flatMap_map] using h1

/-- **At `PopMatches`, the registration's strata are its population**: where `desc` parses to `p`, whose strata under
the query `q` are `S` (`Partition.strata`), `nsOf` is their sizes and their slots, stratum by stratum, are the units
`0, …, n − 1`, since the layout `PopMatches` reads lays each stratum's units in order (`layout_ok`, `laysOut_units`). -/
theorem popMatches_strata {desc : Claims.Msg} {q : Lean.Json} {n : ℕ} {p : Spot.Program}
    {S : Array (String × Array (ℕ × ℕ))} (hpop : PopMatches desc q n)
    (hp : Spot.Program.parse ⟨desc.toArray⟩ = .ok p) (hS : Spot.Partition.strata p q = .ok S) :
    nsOf desc q = (S.toList.map fun s => (SecurityProofs.expand s.2.toList).length) ∧
      (S.toList.flatMap fun s => SecurityProofs.expand s.2.toList) = List.range n := by
  obtain ⟨p', l, hp', hl, hu⟩ := popMatches_iff.1 hpop
  have epp : p' = p := by rw [hp] at hp'; exact (Except.ok.inj hp').symm
  rw [epp] at hl
  obtain ⟨rs, he, pop, rfl, hlays⟩ := SecurityProofs.layout_ok p q l hl
  obtain ⟨rs', he', rfl⟩ := strata_evaluate hS
  obtain rfl : rs' = rs := by rw [he] at he'; exact (Except.ok.inj he').symm
  obtain ⟨h1, h2⟩ := SecurityProofs.laysOut_units _ _ hlays
  have hpo : popOf { classes := #[pop, SecurityProofs.inputClass p] } = pop.toList.flatMap (·.toList) := by
    simp [popOf, List.flatMap_def]
  refine ⟨?_, List.map_injective_iff.2 (Option.some_injective _) ?_⟩
  · rw [nsOf_eq hp hl]
    simpa [SecurityProofs.rsSize] using h2
  · rw [← h1, ← hu, hpo]

<<<<<<< ours
/-- **The sampling design the audit samples under is on the registration's strata**: at `PopMatches`, a `drawnSamplingDesign` that is not a
Bernoulli sampling design is `SamplingDesignOn` the strata's sizes `nsOf`, sampling `samplingDesignK` units: a subset sampling design within the `n` units they sum to,
||||||| base
/-- **The sampling design the audit samples under is on the registration's strata**: at `PopMatches`, a `drawnLaw` that is not a
Bernoulli sampling design is `LawOn` the strata's sizes `nsOf`, sampling `lawK` units: a subset sampling design within the `n` units they sum to,
=======
/-- **The sampling design the audit samples under is on the registration's strata**: at `PopMatches`, a `sampledLaw` that is not a
Bernoulli sampling design is `LawOn` the strata's sizes `nsOf`, sampling `lawK` units: a subset sampling design within the `n` units they sum to,
>>>>>>> theirs
or the stratified or work sampling design over the strata, whose slots are the units `0, …, n − 1` in order, each stratum's `k` the
rule's. -/
<<<<<<< ours
theorem samplingDesignOn_drawnSamplingDesign {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ} {samplingDesign : Flock.Draw.SamplingDesign}
    (hpop : PopMatches desc q n) (h : drawnSamplingDesign desc q lj tb n = .ok samplingDesign) (hb : NotBernoulli samplingDesign) :
    SamplingDesignOn samplingDesign (nsOf desc q) (samplingDesignK samplingDesign) := by
||||||| base
theorem lawOn_drawnLaw {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ} {law : Flock.Draw.Law}
    (hpop : PopMatches desc q n) (h : drawnLaw desc q lj tb n = .ok law) (hb : NotBernoulli law) :
    LawOn law (nsOf desc q) (lawK law) := by
=======
theorem lawOn_sampledLaw {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ} {law : Flock.Draw.Law}
    (hpop : PopMatches desc q n) (h : sampledLaw desc q lj tb n = .ok law) (hb : NotBernoulli law) :
    LawOn law (nsOf desc q) (lawK law) := by
>>>>>>> theirs
  have hsum : (nsOf desc q).sum = n := nsOf_sum hpop
<<<<<<< ours
  rcases drawnSamplingDesign_cases h with ⟨k, rfl, hk⟩ | ⟨a, b, rfl⟩ | ⟨p, S, k, hp, hS, hlaw⟩
||||||| base
  rcases drawnLaw_cases h with ⟨k, rfl, hk⟩ | ⟨a, b, rfl⟩ | ⟨p, S, k, hp, hS, hlaw⟩
=======
  rcases sampledLaw_cases h with ⟨k, rfl, hk⟩ | ⟨a, b, rfl⟩ | ⟨p, S, k, hp, hS, hlaw⟩
>>>>>>> theirs
  · exact ⟨rfl, by rw [hsum]; exact hk⟩
  · exact (hb a b rfl).elim
  obtain ⟨hns, hn⟩ := popMatches_strata hpop hp hS
  have hns' : (nsOf desc q).sum = SecurityProofs.strataSize S := by rw [hns]; rfl
  have hslots : (S.toList.flatMap fun s => SecurityProofs.expand s.2.toList) = List.range (nsOf desc q).sum := by
    rw [hsum, hn]
  rcases hlaw with hder | ⟨tab, hwork⟩
  · obtain ⟨-, hkn⟩ := SecurityProofs.derive_bounds k S samplingDesign hder
    obtain rfl := SecurityProofs.derive_ok k S samplingDesign hder
    refine ⟨⟨?_, ?_⟩, by rw [hns']; exact hkn, ?_, ?_, rfl⟩
    · rw [Array.toList_map, List.flatMap_map]; exact hslots
    · rw [hns, Array.toList_map]
      simp only [List.map_map, Function.comp_def, SecurityProofs.stratum_size, SecurityProofs.rsSize]
    · rw [hns]; intro x hx; obtain ⟨s, hs, rfl⟩ := List.mem_map.1 hx; exact strata_pos hS s hs
    · intro s hs; rw [Array.toList_map] at hs; obtain ⟨s₀, -, rfl⟩ := List.mem_map.1 hs
      simp only [SecurityProofs.stratum_size, hns']
  · obtain ⟨ws, hF, rfl⟩ := SecurityProofs.deriveWork_ok k S tab samplingDesign hwork
    have hunits := SecurityProofs.units_of_forall₂ _ _ hF
    have htot : Flock.Draw.totalWork (ws.map fun s => { s with
        k := Flock.Draw.workK k s.work s.size (Flock.Draw.totalWork ws) s.floor }) = Flock.Draw.totalWork ws := by
      rw [SecurityProofs.totalWork_eq, SecurityProofs.totalWork_eq, Array.toList_map, List.map_map]
      rfl
    refine ⟨⟨?_, ?_⟩, ?_, rfl⟩
    · rw [Array.toList_map, ← hslots]
      simpa only [List.map_map, List.flatMap_map, Function.comp_def] using flatMap_expand_eq hunits
    · rw [hns, Array.toList_map]
      have h2 := congrArg (List.map fun u : Array (ℕ × ℕ) => (SecurityProofs.expand u.toList).length) hunits
      simpa only [List.map_map, Function.comp_def, SecurityProofs.stratum_size, SecurityProofs.rsSize] using h2
    · intro s hs; rw [Array.toList_map] at hs; obtain ⟨w, -, rfl⟩ := List.mem_map.1 hs
      exact (congrArg (fun t => Flock.Draw.workK k w.work w.size t w.floor) htot).symm

<<<<<<< ours
/-- **The sampling design the audit samples under is sized**, at any population: a `drawnSamplingDesign` that is not a Bernoulli sampling design is a
subset sampling design, or a stratified or work sampling design whose strata's slots hold each unit once, at `samplingDesignK` (`SamplingDesignSized`), so its sample
files hold `samplingDesignK` units (`samplingDesignSized_size`). -/
theorem samplingDesignSized_drawnSamplingDesign {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ}
    {samplingDesign : Flock.Draw.SamplingDesign} (h : drawnSamplingDesign desc q lj tb n = .ok samplingDesign) (hb : NotBernoulli samplingDesign) :
    SamplingDesignSized samplingDesign (samplingDesignK samplingDesign) := by
  rcases drawnSamplingDesign_cases h with ⟨k, rfl, -⟩ | ⟨a, b, rfl⟩ | ⟨p, S, k, -, hS, hder | ⟨tab, hwork⟩⟩
||||||| base
/-- **The sampling design the audit samples under is sized**, at any population: a `drawnLaw` that is not a Bernoulli sampling design is a
subset sampling design, or a stratified or work sampling design whose strata's slots hold each unit once, at `lawK` (`LawSized`), so its sample
files hold `lawK` units (`lawSized_size`). -/
theorem lawSized_drawnLaw {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ}
    {law : Flock.Draw.Law} (h : drawnLaw desc q lj tb n = .ok law) (hb : NotBernoulli law) :
    LawSized law (lawK law) := by
  rcases drawnLaw_cases h with ⟨k, rfl, -⟩ | ⟨a, b, rfl⟩ | ⟨p, S, k, -, hS, hder | ⟨tab, hwork⟩⟩
=======
/-- **The sampling design the audit samples under is sized**, at any population: a `sampledLaw` that is not a Bernoulli sampling design is a
subset sampling design, or a stratified or work sampling design whose strata's slots hold each unit once, at `lawK` (`LawSized`), so its sample
files hold `lawK` units (`lawSized_size`). -/
theorem lawSized_sampledLaw {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ}
    {law : Flock.Draw.Law} (h : sampledLaw desc q lj tb n = .ok law) (hb : NotBernoulli law) :
    LawSized law (lawK law) := by
  rcases sampledLaw_cases h with ⟨k, rfl, -⟩ | ⟨a, b, rfl⟩ | ⟨p, S, k, -, hS, hder | ⟨tab, hwork⟩⟩
>>>>>>> theirs
  · exact rfl
  · exact (hb a b rfl).elim
  · obtain rfl := SecurityProofs.derive_ok k S samplingDesign hder
    refine ⟨?_, rfl⟩
    rw [Array.toList_map, List.flatMap_map]
    exact strata_nodup hS
  · obtain ⟨ws, hF, rfl⟩ := SecurityProofs.deriveWork_ok k S tab samplingDesign hwork
    refine ⟨?_, rfl⟩
    have h1 := flatMap_expand_eq (SecurityProofs.units_of_forall₂ _ _ hF)
    have h2 : (ws.toList.flatMap fun a => SecurityProofs.expand a.toStratum.units.toList).Nodup := by
      rw [h1]; exact strata_nodup hS
    simpa only [Array.toList_map, List.map_map, List.flatMap_map, Function.comp_def] using h2

<<<<<<< ours
/-- **The sampling design the verifier samples under is sized**, at any population: `SamplingDesignSized` at the `kd` units it samples, when it is
no Bernoulli sampling design (`samplingDesignSized_drawnSamplingDesign`). -/
theorem samplingDesignSized_drawsUnder {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ}
    {samplingDesign : Flock.Draw.SamplingDesign} {kd : ℕ} (h : DrawsUnder desc q lj tb n samplingDesign kd) (hb : NotBernoulli samplingDesign) :
    SamplingDesignSized samplingDesign kd := by
||||||| base
/-- **The sampling design the verifier samples under is sized**, at any population: `LawSized` at the `kd` units it samples, when it is
no Bernoulli sampling design (`lawSized_drawnLaw`). -/
theorem lawSized_drawsUnder {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ}
    {law : Flock.Draw.Law} {kd : ℕ} (h : DrawsUnder desc q lj tb n law kd) (hb : NotBernoulli law) :
    LawSized law kd := by
=======
/-- **The sampling design the verifier samples under is sized**, at any population: `LawSized` at the `kd` units it samples, when it is
no Bernoulli sampling design (`lawSized_sampledLaw`). -/
theorem lawSized_samplesUnder {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ}
    {law : Flock.Draw.Law} {kd : ℕ} (h : SamplesUnder desc q lj tb n law kd) (hb : NotBernoulli law) :
    LawSized law kd := by
>>>>>>> theirs
  obtain ⟨h, rfl⟩ := h
<<<<<<< ours
  exact samplingDesignSized_drawnSamplingDesign h hb
||||||| base
  exact lawSized_drawnLaw h hb
=======
  exact lawSized_sampledLaw h hb
>>>>>>> theirs

<<<<<<< ours
/-- **The sampling design the verifier samples under is on the registration's strata**: at `PopMatches`, `SamplingDesignOn` at `nsOf` and the
`kd` units it samples, when it is no Bernoulli sampling design (`samplingDesignOn_drawnSamplingDesign`). -/
theorem samplingDesignOn_drawsUnder {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ}
    {samplingDesign : Flock.Draw.SamplingDesign} {kd : ℕ} (hpop : PopMatches desc q n) (h : DrawsUnder desc q lj tb n samplingDesign kd)
    (hb : NotBernoulli samplingDesign) : SamplingDesignOn samplingDesign (nsOf desc q) kd := by
||||||| base
/-- **The sampling design the verifier samples under is on the registration's strata**: at `PopMatches`, `LawOn` at `nsOf` and the
`kd` units it samples, when it is no Bernoulli sampling design (`lawOn_drawnLaw`). -/
theorem lawOn_drawsUnder {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ}
    {law : Flock.Draw.Law} {kd : ℕ} (hpop : PopMatches desc q n) (h : DrawsUnder desc q lj tb n law kd)
    (hb : NotBernoulli law) : LawOn law (nsOf desc q) kd := by
=======
/-- **The sampling design the verifier samples under is on the registration's strata**: at `PopMatches`, `LawOn` at `nsOf` and the
`kd` units it samples, when it is no Bernoulli sampling design (`lawOn_sampledLaw`). -/
theorem lawOn_samplesUnder {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ}
    {law : Flock.Draw.Law} {kd : ℕ} (hpop : PopMatches desc q n) (h : SamplesUnder desc q lj tb n law kd)
    (hb : NotBernoulli law) : LawOn law (nsOf desc q) kd := by
>>>>>>> theirs
  obtain ⟨h, rfl⟩ := h
<<<<<<< ours
  exact samplingDesignOn_drawnSamplingDesign hpop h hb

/-- **The sampling design the verifier samples under at a registration**: `drawnSamplingDesign`'s, and `subset 0`, which samples no
proof block, at a registration whose sampling design `drawnSamplingDesign` refuses, where `drawnFrom` refuses every sample
(`drawnFrom_drawnSamplingDesign`). -/
def designAt (desc : Claims.Msg) (q lj : Lean.Json) (tb : Option Lean.Json) (n : ℕ) : Flock.Draw.SamplingDesign :=
  match drawnSamplingDesign desc q lj tb n with
  | .ok samplingDesign => samplingDesign
||||||| base
  exact lawOn_drawnLaw hpop h hb

/-- **The sampling design the verifier samples under at a registration**: `drawnLaw`'s, and `subset 0`, which samples no
proof block, at a registration whose sampling design `drawnLaw` refuses, where `drawnFrom` refuses every sample
(`drawnFrom_drawnLaw`). -/
def designAt (desc : Claims.Msg) (q lj : Lean.Json) (tb : Option Lean.Json) (n : ℕ) : Flock.Draw.Law :=
  match drawnLaw desc q lj tb n with
  | .ok law => law
=======
  exact lawOn_sampledLaw hpop h hb

/-- **The sampling design the verifier samples under at a registration**: `sampledLaw`'s, and `subset 0`, which samples no
proof block, at a registration whose sampling design `sampledLaw` refuses, where `sampledFrom` refuses every sample
(`sampledFrom_sampledLaw`). -/
def designAt (desc : Claims.Msg) (q lj : Lean.Json) (tb : Option Lean.Json) (n : ℕ) : Flock.Draw.Law :=
  match sampledLaw desc q lj tb n with
  | .ok law => law
>>>>>>> theirs
  | .error _ => .subset 0

<<<<<<< ours
/-- `designAt` is the sampling design the verifier samples under (`DrawsUnder`, at its `samplingDesignK`), or `subset 0`. -/
||||||| base
/-- `designAt` is the sampling design the verifier samples under (`DrawsUnder`, at its `lawK`), or `subset 0`. -/
=======
/-- `designAt` is the sampling design the verifier samples under (`SamplesUnder`, at its `lawK`), or `subset 0`. -/
>>>>>>> theirs
theorem designAt_cases (desc : Claims.Msg) (q lj : Lean.Json) (tb : Option Lean.Json) (n : ℕ) :
<<<<<<< ours
    DrawsUnder desc q lj tb n (designAt desc q lj tb n) (samplingDesignK (designAt desc q lj tb n)) ∨
||||||| base
    DrawsUnder desc q lj tb n (designAt desc q lj tb n) (lawK (designAt desc q lj tb n)) ∨
=======
    SamplesUnder desc q lj tb n (designAt desc q lj tb n) (lawK (designAt desc q lj tb n)) ∨
>>>>>>> theirs
      designAt desc q lj tb n = .subset 0 := by
  unfold designAt
<<<<<<< ours
  cases h : drawnSamplingDesign desc q lj tb n with
  | ok samplingDesign => exact Or.inl ⟨h, rfl⟩
||||||| base
  cases h : drawnLaw desc q lj tb n with
  | ok law => exact Or.inl ⟨h, rfl⟩
=======
  cases h : sampledLaw desc q lj tb n with
  | ok law => exact Or.inl ⟨h, rfl⟩
>>>>>>> theirs
  | error _ => exact Or.inr rfl

<<<<<<< ours
/-- **Every sample the one-stage verifier samples is under `designAt`**: where `drawnFrom` samples, at a verifier whose program
the descriptor `desc` parses to and under the sampling design object `Draw.derive` writes, its sample is `drawUnits` under
`designAt desc v.query v.samplingDesign v.workTable n`. -/
theorem drawnFrom_designAt {desc : Claims.Msg} {v : Spot.Registration.Verifier} {lo : Lean.Json}
||||||| base
/-- **Every sample the one-stage verifier samples is under `designAt`**: where `drawnFrom` samples, at a verifier whose program
the descriptor `desc` parses to and under the sampling design object `Draw.derive` writes, its sample is `drawUnits` under
`designAt desc v.query v.law v.workTable n`. -/
theorem drawnFrom_designAt {desc : Claims.Msg} {v : Spot.Registration.Verifier} {lo : Lean.Json}
=======
/-- **Every sample the one-stage verifier samples is under `designAt`**: where `sampledFrom` samples, at a verifier whose program
the descriptor `desc` parses to and under the sampling design object `Draw.derive` writes, its sample is `sampleUnits` under
`designAt desc v.query v.law v.workTable n`. -/
theorem sampledFrom_designAt {desc : Claims.Msg} {v : Spot.Registration.Verifier} {lo : Lean.Json}
>>>>>>> theirs
    (hp : Spot.Program.parse ⟨desc.toArray⟩ = .ok v.program)
<<<<<<< ours
    (hlo : Spot.Draw.derive v.program v.query v.samplingDesign v.workTable = .ok lo) {n : ℕ} {src : ByteArray}
    {r : Array ℕ × ℕ} (h : Spot.OneStage.drawnFrom v lo n src = .ok r) :
    Spot.Draw.drawUnits (designAt desc v.query v.samplingDesign v.workTable n) n src = .ok r := by
  obtain ⟨samplingDesign, ⟨hl, -⟩, hd⟩ := drawsUnder_of_drawnFrom hp hlo h
||||||| base
    (hlo : Spot.Draw.derive v.program v.query v.law v.workTable = .ok lo) {n : ℕ} {src : ByteArray}
    {r : Array ℕ × ℕ} (h : Spot.OneStage.drawnFrom v lo n src = .ok r) :
    Spot.Draw.drawUnits (designAt desc v.query v.law v.workTable n) n src = .ok r := by
  obtain ⟨law, ⟨hl, -⟩, hd⟩ := drawsUnder_of_drawnFrom hp hlo h
=======
    (hlo : Spot.Draw.derive v.program v.query v.law v.workTable = .ok lo) {n : ℕ} {src : ByteArray}
    {r : Array ℕ × ℕ} (h : Spot.OneStage.sampledFrom v lo n src = .ok r) :
    Spot.Draw.sampleUnits (designAt desc v.query v.law v.workTable n) n src = .ok r := by
  obtain ⟨law, ⟨hl, -⟩, hd⟩ := samplesUnder_of_sampledFrom hp hlo h
>>>>>>> theirs
  unfold designAt
  rw [hl]
  exact hd

/-- **The verifier samples under no Bernoulli sampling design** (`notBernoulli_sampledDesign`). -/
<<<<<<< ours
theorem notBernoulli_drawsUnder {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ}
    {samplingDesign : Flock.Draw.SamplingDesign} {kd : ℕ} (h : DrawsUnder desc q lj tb n samplingDesign kd) : NotBernoulli samplingDesign :=
||||||| base
theorem notBernoulli_drawsUnder {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ}
    {law : Flock.Draw.Law} {kd : ℕ} (h : DrawsUnder desc q lj tb n law kd) : NotBernoulli law :=
=======
theorem notBernoulli_samplesUnder {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ}
    {law : Flock.Draw.Law} {kd : ℕ} (h : SamplesUnder desc q lj tb n law kd) : NotBernoulli law :=
>>>>>>> theirs
  notBernoulli_sampledDesign h.1

/-- **`designAt` is no Bernoulli sampling design** (`notBernoulli_samplesUnder`; `subset 0` is none). -/
theorem notBernoulli_designAt (desc : Claims.Msg) (q lj : Lean.Json) (tb : Option Lean.Json) (n : ℕ) :
    NotBernoulli (designAt desc q lj tb n) := by
  rcases designAt_cases desc q lj tb n with h | h
  · exact notBernoulli_samplesUnder h
  · rw [h]
    intro _ _ h'
    cases h'

<<<<<<< ours
/-- **`designAt` is sized** at its `samplingDesignK` (`samplingDesignSized_drawsUnder`; `subset 0` at `0`). -/
||||||| base
/-- **`designAt` is sized** at its `lawK` (`lawSized_drawsUnder`; `subset 0` at `0`). -/
=======
/-- **`designAt` is sized** at its `lawK` (`lawSized_samplesUnder`; `subset 0` at `0`). -/
>>>>>>> theirs
theorem sized_designAt (desc : Claims.Msg) (q lj : Lean.Json) (tb : Option Lean.Json) (n : ℕ) :
    SamplingDesignSized (designAt desc q lj tb n) (samplingDesignK (designAt desc q lj tb n)) := by
  rcases designAt_cases desc q lj tb n with h | h
<<<<<<< ours
  · exact samplingDesignSized_drawsUnder h (notBernoulli_drawsUnder h)
||||||| base
  · exact lawSized_drawsUnder h (notBernoulli_drawsUnder h)
=======
  · exact lawSized_samplesUnder h (notBernoulli_samplesUnder h)
>>>>>>> theirs
  · rw [h]
    exact rfl

end Flock.Bridge
