module
public import Proofs.Properties.Verity.Sampled.HonestSession.PublicFileProof
import Flock.HmRow
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic
import all Std.Data.TreeMap.Raw.Basic
import all Std.Data.DTreeMap.Raw.Basic
import all Std.Data.DTreeMap.Internal.Operations
import all Std.Data.DTreeMap.Internal.Queries
import all Std.Data.DTreeMap.Internal.Balancing
import all Std.Data.DTreeMap.Internal.Def
import all Init.Data.Ord.String
import all Init.Data.Ord.Basic
import all Init.Data.String.TakeDrop
import all Init.Data.String.Slice
import all Init.Data.String.Pattern.Basic
import all Init.Data.String.Pattern.String
import all Init.Data.String.Pattern.Char
import all Init.Data.String.Search
import all Init.Data.String.Iterate
import all Init.Data.String.Decode
import all Init.Data.String.Termination
import all Init.Data.String.Defs
import all Init.Data.String.Basic
import all Init.Data.String.Extra
import all Init.Data.Array.QSort.Basic
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control
import all Init.Data.Range.Basic
import all Init.Data.Int.Repr
import all Init.Data.Repr
import all Init.Data.Nat.Fold
import all Init.Data.Array.Attach
import all Init.WF
@[expose] public section
open Flock
namespace Verity.Sampled.Instance.HonestSession

/-! # The setup's reads of the loaded and drawn public values

The honest-session witness (integration `HonestSession`) and the setup/program pieces (`SetupReads`, `ProgramSetup`)
take, as hypotheses, four reads this file discharges from the public-file proof (which owns the loaded value `pubI = Vout`
and the drawn value `pubD = Wdrawn`): the drawn population (`pubD.n = 1`), the one-port row digest size
(`(pubD.digest 0 0 1).size = 128`), and that each header's `units` entry names the one drawn unit, index 0
(`hidx` for `pubD`, `hidxI` for `pubI`), plus `pubI = Vout` for the links check. The statements match the
hypotheses of `SetupReads.setupFacts_of` and `ProgramSetup.links_ok` exactly.

The `units`/`indices` reads go through `Json`'s `Std.TreeMap.Raw` object, whose `insert`/`ofList` and `get?`
the kernel reduces only over concrete keys with its implementation exposed; hence the `import all` block above (the same
the parameters piece uses). No hash is forced: the reads touch only keys and the (concrete) index array. -/

/-- `pubD`'s population is one. -/
theorem pubD_n (hC : CircuitFileFacts) : pubD.n = 1 := by
  rw [pubD_eq hC]; kernel_rfl'

/-- `pubD`'s one instance's one-port row digest is 128 bytes: the drawn digest is the population's row, the concrete
`rowBc`'s first `BC_BYTES` bytes, whose size is `128` (no hash forced -- `rowBc` is the concrete zero row). -/
theorem pubD_digest_size (hC : CircuitFileFacts) : (pubD.digest 0 0 1).size = 128 := by
  rw [pubD_eq hC, show Wdrawn.digest 0 0 1 = rowBc.extract 0 128 from by
    simp [Flock.Public.digest, Wdrawn, drawnPF, pubPF, Draw.UnitSample.proved, HmRow.BC_BYTES]]
  kernel_rfl'

/-- The program-side links check reads `pubI`; it is the loaded public value `Vout` (restated from the public-file
proof for citation by the program piece, which takes it as a named hypothesis). -/
theorem pubI_val (hC : CircuitFileFacts) : pubI = Vout := pubI_eq hC

/-- **`hidxI`**: the loaded header's `units` entry names the one unit, index 0. `Vout.header`'s `units` is the
loaded `unitsF`, whose `indices` is `#[0]`. -/
theorem pubI_units_indices (hC : CircuitFileFacts) :
    ∃ U, J.get pubI.header "units" "public file" = .ok U ∧
      J.arr U "indices" "public file" = .ok #[Lean.toJson (0 : Nat)] :=
  ⟨unitsF (hexOf classSha), by rw [pubI_eq hC]; kernel_rfl', by kernel_rfl'⟩

/-- **`hidx`**: the drawn header's `units` entry names the one drawn unit, index 0. The draw stores
`HmRow.unitsEntry { cH with outPorts := #[] } #[0]` under `units`; reading `indices` off it (`ue_eq` to the
concrete `UEJmine`) is `#[0]`. -/
theorem pubD_units_indices (hC : CircuitFileFacts) :
    ∃ U, J.get pubD.header "units" "public file" = .ok U ∧
      J.arr U "indices" "public file" = .ok #[Lean.toJson (0 : Nat)] := by
  refine ⟨HmRow.unitsEntry { cH with outPorts := #[] } #[0], by rw [pubD_eq hC]; kernel_rfl', ?_⟩
  rw [ue_eq { cH with outPorts := #[] } hC.cH_g hC.cH_upv hC.cH_classSha]; kernel_rfl'

end Verity.Sampled.Instance.HonestSession
