module

public import Proofs.Flock.RecGeneric.AtFlock
public import Proofs.Flock.Claims.Recursion.BoundBridge
public import Proofs.Flock.Claims.Recursion.HonestIVStarR

/-! The Flock instance's server, hm96 at the bound key (`Server.ofBound`), opens and binds as the bound record does. -/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

theorem ivStarOfBound {Coin Col T : Type} [Inhabited Coin] (Bd : Bound Coin Col T) : IVStarOfBound Bd := by
  unfold IVStarOfBound Server.IVStar Bound.IVStar
  dsimp only [Server.ofBound]
  congr 1
  funext x as cs
  rcases Bd.final (as.getLastD []) with _ | ⟨os, last⟩ <;> rfl

theorem openViewOfBound {Coin Col T : Type} (Bd : Bound Coin Col T) : OpenViewOfBound Bd := by
  unfold OpenViewOfBound
  funext v
  unfold Server.openView Bound.openView
  dsimp only [Server.ofBound]
  rcases Bd.final (v.msgs.getLastD ([], Out.rej)).1 with _ | ⟨os, last⟩ <;> rfl

theorem opens_openView_ofBound {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T) (Cp : Caps Coin) (v : Seen) :
    (Proto.ofFlock Bd decI Cp).opens ((Server.ofBound Bd).openView v) = (Inner.ofFlock Bd.deploy decI).opens v :=
  congrArg (Proto.ofFlock Bd decI Cp).opens (congrFun (openViewOfBound Bd) v)

theorem commit_ofBound_eval {Coin Col T : Type} (Bd : Bound Coin Col T) (H : Msg → Msg) (row : Msg)
    (y : Fin Bd.saltLen → UInt8) : (((Server.ofBound Bd).commit row y).eval H).1 = Bd.commitRow H row y :=
  rfl

theorem bindsOfBound {Coin Col T : Type} (Bd : Bound Coin Col T) : BindsOfBound Bd := by
  intro hB H a row row' y y' hne h h'
  obtain ⟨u, hu, v, hv, hc⟩ := Bd.opens_coll H hB h h' fun he => hne (Prod.mk.inj he).1
  exact ⟨u, List.mem_append_left _ hu, v, List.mem_append_right _ hv, hc⟩

theorem opensOfBound {Coin Col T : Type} (Bd : Bound Coin Col T) : OpensOfBound Bd := fun _ h64 row y =>
  Bd.opens_commitRow h64 row y

end Claims.RecGeneric
