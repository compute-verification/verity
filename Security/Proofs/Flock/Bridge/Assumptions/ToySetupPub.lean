import Proofs.Flock.Bridge.ToyPubReads
import Proofs.Flock.Bridge.PublicWitnesses
import Proofs.Flock.Bridge.Honest
import Flock.Test.ToyPubFiles

/-!
# The toy session's setup, at its registered rows (`toy-setup`)

`SampledKSLivePub`'s nontrivial witness is at the toy `verify --zk` session of `Flock.Test.ToyPubFiles`, with the
registrant `Bridge.ToyPub.lieReads`. As in `ToySetupHJ`, some of its fields can't be built in Lean: the reference setup
`y₀` is a statement `Stmt.setupHidden` accepts on the toy's files, the session's transcripts `wr` and their custody
`hCust` are what the live verifier writes and records, and a prover's finder's collision probability is SHA-512's.
`ToySetupPub` is the premise that they exist, with every transcript's record carrying the toy's draw.
-/

namespace Flock.Assumptions

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open Flock.Test.ToyPubFiles (circuit circuitFile publicFile pub)

/-- The toy session's inputs: `verify --zk`'s for one session of the toy's files. -/
abbrev toyPubI : Inputs := Bridge.servedI circuitFile publicFile

theorem toyPubI_typed : toyPubI.tags.typed = false := rfl

theorem toyPubI_tags : TagsOkZHJ toyPubI := ⟨rfl, rfl, rfl⟩

theorem toyPubI_div : 1 % nTab toyPubI = 0 := rfl

/-- The toy's random bytes: one stream of none. -/
def toyLb : Fin 1 → ℕ := fun _ => 0

/-- The toy's coin space: the bytes themselves. -/
abbrev ToyΩ : Type := (i : Fin 1) → Fin (toyLb i) → Fin 256

/-- The toy's source: its coin. -/
def toySrc : ToyΩ → (i : Fin 1) → Fin (toyLb i) → Fin 256 := id

theorem toySrc_uniform : FlockSoundness.Assumptions.UniformRandomBytes toyLb toySrc := fun _ => rfl

open Classical in
/-- Premise (`toy-setup`, interim) of `SampledKSLivePub`'s nontrivial witness. **The toy session has a reference setup,
transcripts that carry its draw, and a prover whose finder rarely collides.** At the toy's files parsed as
`Bridge.Assumptions.ToyFilesParse` states, one unit drawn by the law of the subset draw of size 1: there are a draw
file `dj`, a hidden-output table `verify --zk` sets up at table 0 (`DrawSetupZHJ`), transcripts `wr` whose every
record carries the draw `draw` and whose custody is the live verifier's (`RecordCustodyZKJ`), and a prover `τ` whose
finder `B^τ` against the registrant `Bridge.ToyPub.lieReads` (at a per-run time `t'`, a verification `v` and a
strict-path time `qR`) has `J²·K·Pr[B^τ] ≤ 2^-200`, `K` the number of C-Flock's finders and `J` the plan's tables.
Every draw flock-verify `--zk` accepts gives the setup, so a run of flock-verify on the toy's files checks it; the bound
is `cr/sha-512`'s at a prover that finds no collision. -/
def ToySetupPub : Prop :=
  ∀ (hD : Flock.Desc.parse (zkTags toyPubI) toyPubI.circuitFile toyPubI.tables = .ok (.single circuit))
    (hU : Built circuit.unit) (hpub : HmOut.loadPublic (zkTags toyPubI) circuit toyPubI.publicFile = .ok pub)
    (hpt : pub.tables = none) (hscope : Discharge.Layout.scopeOkH circuit = true),
  ∃ (dj : Finset (Fin 1) → Lean.Json) (S₀ : Finset (Fin 1)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ toyPubI 0) (mPtsOf circuit.g (1 / nTab toyPubI)) S₀ d₀)
    (wr : (S : Finset (Fin 1)) → (R : Unit) →
      LiveOutJ (planZHJ (Reg := Unit) dj y₀ S R).length (mPtsOf circuit.g (1 / nTab toyPubI)) → Discharge.Exec.Transcript),
    (∀ S R o, recordDraw (wr S R o).record = some Flock.Test.ToyPubFiles.draw) ∧
    ∃ (hCust : FlockSoundness.Assumptions.Zk.RecordCustodyZKJ toyPubI dj
        (fun S R => (planZHJ (Reg := Unit) dj y₀ S R).length) wr 1)
      (dg : ByteArray → ByteArray) (N : ℕ)
      (τ : Strategy (auditLive (Law.execOS (n := 1) toySrc (.subset 1) 1) Unit (liveOfZJ dg (planZHJ dj y₀) N wr)))
      (t' v qR : ℝ),
      let L := Law.execOS (n := 1) toySrc (.subset 1) 1
      let G := auditLive L Unit (liveOfZJ dg (planZHJ dj y₀) N wr)
      let P := (keyProg circuit hU 1).partition (Discharge.Layout.outsOf circuit hU 1)
      let vb := (rsZHJ (Reg := Unit) dj y₀ toyPubI_typed (Flock.Desc.parse_flat hD) hscope hU hpub hpt).bindingZ
        (rsZHJ_computes dj y₀ toyPubI_typed (Flock.Desc.parse_flat hD) hscope hU hpub hpt)
      let rd := Bridge.ToyPub.lieReads (Reg := Unit) dj y₀ toyPubI_typed (Flock.Desc.parse_flat hD) hscope hU hpub
      let wo : vb.Pos → vb.Opening := fun p => ((p, rd.wv p, rd.ws p) : vb.Opening)
      let σ := simZJ execArith hs512 (planZHJ dj y₀) toyPubI wr (opnAtZHJ (planZHJ dj y₀) toyPubI dj) dg
        (decAtZJ (planZHJ dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
        (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ toyPubI wr _ _ _ _ _
          (runsZCJH_live toyPubI toyPubI_typed (Flock.Desc.parse_flat hD)
            (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ toyPubI_tags hCust.1
            fun _ => drawOkZKHJ_of_file toyPubI toyPubI_typed (Flock.Desc.parse_flat hD) 1 1 toyPubI_div dj hCust.2 _))
        τ
      (nTab toyPubI : ℝ) ^ 2 * Fintype.card (Bridge.PubIxR dj y₀ P vb L) *
          (Bridge.pubFinderR dj y₀ P vb wo Bridge.kDep (2 ^ 192) (2 ^ 192) (2 * (t' + v)) (t' + v) t' (reg σ) (cont σ)
            (Bridge.strictComp (FlockSoundness.StrictCR.oneRun G τ fun o =>
              rd.pick o.1 (Discharge.ZkReg.regPathsHJ toyPubI circuit pub o.2.2 rd.port)) qR)).Pr
            Claims.Sampling.sha512Msg ≤ 1 / 2 ^ 200

end Flock.Assumptions
