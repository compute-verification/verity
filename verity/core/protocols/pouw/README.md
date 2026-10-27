# PoUW (`verity.core.protocols.pouw`)

Proof of useful work: certifies that a served model's declared matmuls were computed, by sampled proofs over committed,
noised checked values. One protocol carries several named schemes behind one interface (`protocol.PoUWScheme`,
`schemes.get`). The proof service registers an audit window, draws its proof blocks with live coins, proves and verifies
them; PoUW states what each window commits, the Program each drawn proof block must pass, and what the outcome credits.
What the draw established is ε at δ; PoUW carries no policy parameter.

Lean is the source of truth. The verifier of record is the three Lake packages under `lean/`; the Python modules of the
same names are their references, held to them by the difftests. The guarantees are in `Security/`, and every name below
is listed in `Security/lean-audit.json`.

## Outline

A window goes through these steps; each names the Lean that decides it.

1. **Weights.** Each weight's rows are committed, and the weights' root binds every weight root
   (`PouwCommit.Rows`, `PouwCommit.Bindings`; `PouwCheck.weightsRoot`).
2. **Salt.** The window's salt is a live coin of the challenger, opened once after the root it follows: a live window's
   after its open root (`PouwCheck.Rounds`' `saltCheck`, `PouwCheck.WindowSalt`), a Pearl-C pass's from a salt-only
   session on its salt epoch's root (`PouwCheck.Pass`' `liveSaltCheck`), a certified window's from the network
   certifier's reveal (`PouwCheck.Pass`' `certifiedSalt`, `pouwSalt`).
3. **Commitments.** The activation rows, the entries of the window root and their side trees (`PouwCommit`;
   `PouwWindow.Entry`, `PouwWindow.Side`, `PouwWindow.Record`), and the window statement on the entry rows
   (`PouwWindow.Statement`). A workload of record serves at a public fixed length (`PouwWindow.Length`).
4. **Registration.** The window registers its level on the grid and its sampling design subset:K′ before the draw
   (`PouwWindow.Grid`, `PouwWindow.Draw`, `PouwWindow.Public`); the proof service refuses a registration after the
   deadline (`SampledProofs.Stream`).
5. **Draw.** The draw coin is opened once, after the registration's receipt, under the challenger key the deployment
   pins (`PouwCheck.Pass`' `drawCheck`), and the units are one uniform K′-subset of the window's N tiles
   (`Flock.Draw.subset`). A live window's round coins open after their round roots (`PouwCheck.Rounds`'
   `roundsCheck`, `PouwWindow.Rounds`).
6. **Proofs.** Each drawn unit is proved in zero knowledge in sessions of the verifier's own public inputs
   (`PouwWindow.Inputs`, `PouwWindow.Lanes`, `PouwWindow.KnownRows`), under a template the verifier pins
   (`PouwCheck.Tile`), and opened against the window's entries (`PouwCheck.Opening`).
7. **Verdict and credit.** `PouwCheck.check` is the window's verdict over the registration and the service's answer
   (`PouwCheck.Check`, `PouwCheck.Inputs`, `PouwCheck.Protocol`). A served window's credit and its period's are
   `PouwCheck.Credit`, on each drawn unit's verdict (`PouwCheck.Verified`, `PouwCheck.LiveVerified`) over the circuit
   files its period's open committed (`PouwCheck.CommittedFiles`).

The packages, and their module tables: `lean/check/README.md` (`pouw_check`), `lean/window/README.md` (`pouw_window`),
and `lean/commit` (`pouw_commit`: tagged bindings, frames, rows, coins and draws). Their properties are
`Properties.Pouw` (`Security/Properties/Pouw.lean` and `Security/Properties/Pouw/`).

## Guarantees

Each guarantee G is stated in `Security/Definitions/Pouw` and proved in `Security/Proofs/Pouw` as
`Pouw.SecurityProofs.G : Pouw.Guarantees.G`.

- **The audit game:** `Pouw.SecurityProofs.Theorem1`, `Pouw.SecurityProofs.WorkWeightedSampling` and
  `Pouw.SecurityProofs.EndToEnd` (a scheme with no γ_0 gets only the sampling half); `Pouw.SecurityProofs.GammaFromTT`.
- **The deadline:** `Pouw.SecurityProofs.WorkBetweenSaltAndDeadline`, `Pouw.SecurityProofs.SpareCapacityAtDeadline`,
  `Pouw.SecurityProofs.PearlCSampledByDeadline` and its sm_120 instances.
- **The schemes' γ:** NCP's `Pouw.SecurityProofs.NCP.GammaFromTTNCP_U_v1` and the other NCP guarantees;
  Pearl-C's γ and sampled guarantees, one per device record, cast and shape (the served point is
  `Pouw.SecurityProofs.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_8192`, with its sampled twin
  `Pouw.SecurityProofs.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_8192`), and the row seed's reduction
  `Pouw.SecurityProofs.PearlC.TTOutRowSeedOfTTOut`; the barrier `Pouw.SecurityProofs.Barrier.Generic`.
- **The relative security of the served credit:** `Pouw.SecurityProofs.RelativeSecurity.TheoremSStaged`,
  `Pouw.SecurityProofs.RelativeSecurity.TheoremSOverlap` and their floors.
- **Compute accounting reads PoUW** through `Accounting.SecurityProofs.ComputeOfPouw`,
  `Accounting.SecurityProofs.ComputeOfPouwPerRun`, `Accounting.SecurityProofs.PouwEndToEndPerRun` and
  `Accounting.SecurityProofs.PouwLocalOff`.

The named premises they rest on include the schemes' conjectures (`Pouw.NCP.Assumptions.TTNCP_U`,
`Pouw.PearlC.Assumptions.TTOutPearlCDevRev1W1` and Pearl-C's other TT_OUT forms), the seeded noise's forms
(`Pouw.PearlC.RowSeedNoise`, `Pouw.PearlC.CodeSeedNoise`), the receipt clock (`Pouw.Assumptions.HonestClock`), the
drawn tile's proof (`Pouw.PearlC.Assumptions.TileProofSound`), and the claims `trusted-host`, `live-verifier`,
`cr/sha-512` and `prf/sha-512` (`verity.core.claims`). Dither, PoUW's no-fold protocol, is experimental and has its own
spec, `dither/PROTOCOL.md`.

## Core API

~~~python
from verity.core.protocols.pouw import core_check, credit, window, verifier
from verity.core.protocols.pouw import schemes

verdict = core_check.check(params, commitment, evidence)   # one audit window, as `pouw-check` answers (Lean of record)
outcome = credit.window(key, ledger, pins, main, ...)      # a served window's credited set and its drawn units' outcome
period = credit.period_credit(key, ledger, pins, main, name, windows, verify, ...)   # a period, closed and complete
passes = credit.pearl_c_period_credit(key, ledger, pins, main, name, passes, earlier, ...)   # a Pearl-C period
work = window.bucket(n_real)                               # the window's level: credited tiles g_ℓ and padded N
k = window.k_prime(eps, delta, work.credited, work.n, tile_work)   # K′, the subset draw that establishes ε at δ
item = window.sampling(eps, delta, work, tile_work)        # the registered sampling design
scheme = schemes.get("pearl-c-sm120-v1-h3")                # a configured PoUWScheme
~~~

`verifier.lean(fn, value)` runs the Lean verifier of record (`RECORD_VERDICT`), and `verifier.coins(n)` is the
challenger's entropy, read by the Lean from the operating system.

## Where things are

- `protocol`, `schemes/`: the `PoUWScheme` interface and the named schemes (`ncp-v1`, `pearl-fp8-v4`, the Pearl-C family,
  `pearl-c-nvfp4-v0`); each scheme module's docstring states its steps and roundings, with pinned vectors under
  `tests/vectors`.
- `audit`, `coins`, `salt`, `window`, `core_check`, `credit`, `identifier`, `registration`, `weights`, `operands`: the
  references the Lean is held to, and the reference rules not yet ported.
- `circuit/`: PoUW as `verity.ml.circuits` Programs: the tile and lane Definitions, the compile pass over a served Program
  (`compile`) and the draw plan (`plan`). Each Definition has its `circuit-check` sidecar.
- `lean/check`, `lean/commit`, `lean/window`: the Lean verifier of record, each with its difftest.
- `dither/`: Dither, experimental, with its `PROTOCOL.md`.
- `tests/`: the package's suite. `benchmarks/pouw/` is the measuring side and the clear reference lifecycle
  (`benchmarks/pouw/clear`); nothing here imports it.
