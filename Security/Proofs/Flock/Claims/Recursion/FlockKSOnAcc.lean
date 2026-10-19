import Proofs.Flock.Claims.Recursion.FlockKSRecord
import Proofs.Flock.Claims.KS.PlainAcc

/-!
# The recursion's table bound from Flock's table bound with escapes on accepted runs

`TableBoundθOnAcc`, the table bound `KSAbort_of_tableBoundOnAcc` reads, is `Flock.TableBoundθAccAt` at every statement
(`tableBoundθOnAcc_of_accAt`). At the deployed plain record that is `FlockDeploy.KS.plain_tableBoundθAccAt` from the
live sessions' bound with escapes on accepted runs at the statements some run accepts
(`FlockDeploy.KS.plain_tableBoundθOnAcc`), the full-width bound `FlockDeploy.KS.plain_KSAbort` takes, so the trimmed
plain record meets `Inner.KSAbort` from that bound alone (`FlockDeploy.KS.plain_KSAbort_of_liveAcc`).
-/

namespace Claims.Recursion

/-- **`TableBoundθOnAcc` from `TableBoundθAccAt` at every statement.** -/
theorem tableBoundθOnAcc_of_accAt {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] [Zero Col] {H : Msg → Msg} {D : Flock.Deploy Coin Col T} {N : Msg → ℕ}
    {Ok : Msg → Col → Prop} (h : ∀ x, Flock.TableBoundθAccAt H D N Ok x) : TableBoundθOnAcc H D N Ok :=
  fun x Ω _ _ P => h x Ω P

end Claims.Recursion

namespace FlockDeploy.KS

open Flock
open Claims.Recursion

/-- **The recursion's table bound at the deployed plain record**, at full-width columns, from the live sessions' bound
with escapes on accepted runs at the statements some run accepts. -/
theorem plain_tableBoundθOnAcc (hl : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V false) x → PlainLiveAcc x) :
    TableBoundθOnAcc sha512 (deployPlain Readable.relation) (wordLen false) (wide false) :=
  tableBoundθOnAcc_of_accAt (plain_tableBoundθAccAt hl)

/-- **The plain record the claims read meets `Inner.KSAbort`** at the deployed relation, at SHA-512, at every decoder
within the radius, from the live sessions' bound with escapes on accepted runs at the statements some run accepts. -/
theorem plain_KSAbort_of_liveAcc (hl : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V false) x → PlainLiveAcc x)
    (decode : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    (hdec : decOK (deployPlainTrim Readable.relation) decode) :
    (Inner.ofFlock (deployPlainTrim Readable.relation) decode).KSAbort sha512 :=
  plain_KSAbort Readable.relation (plain_tableBoundθOnAcc hl) decode hdec

end FlockDeploy.KS
