module

public import Proofs.Flock.SpotFlock.RunsAs.Top
public import Proofs.Flock.SpotFlock.RunsAs.RoundTwoOf

/-! A two-stage run `Run.run` accepts is a play `SpotFlock.V h` accepts, with the premise `RoundTwo` discharged
(`runAcceptsPlay`). -/

@[expose] public section

namespace SpotFlock.RunsAs

open Claims
open Lean (Json)
open Round2 (NamesDistinct TwoHolds RecordsHeld RegistersNone)

theorem runAcceptsPlay (i : Spot.Run.Input) (it : Spot.Run.Interior) (j : Json) (h : Held) (H : Msg → Msg)
    (x m₁ m₂ : Msg) (P' : ℕ → Prover Unit)
    (r' : ∀ t, Fin ((FlockDeploy.V true).k (stmtAt h m₂ t)) → FlockDeploy.Coin) (c₀ : FlockDeploy.Coin)
    (c₁ : Stage h) (ha : AcceptedTwo i it j) (hm : HeldMatches i it h) (htwo : TwoHolds h)
    (hnd : NamesDistinct h) (hrec : RecordsHeld h x m₁) (hnone : RegistersNone it.record)
    (hx : jsonMsg x = some i.record) (hm₁ : jsonMsg m₁ = some it.record)
    (hsa : SessionsAccept h H ((it.statements).getD #[]) m₂ P' r') (hpc : PlayCoins h i it c₀ c₁) :
    (Claims.run H (proverSeq h [] m₁ m₂ P') (V h) x (fun _ => ()) (seqCoins h (x := x) c₀ c₁ r')).out = .acc :=
  spotRunAcceptsPlay i it j h H x m₁ m₂ P' r' c₀ c₁ ha hm htwo hnd hrec hnone hx hm₁ (stmtOk_of_check h) hsa hpc

end SpotFlock.RunsAs
