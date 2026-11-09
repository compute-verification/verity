module

public import Proofs.Flock.RecGeneric.AtFlock
public import Proofs.Flock.Claims.Recursion.BoundBridge
public import Proofs.Flock.Claims.Recursion.HonestIVStarR

/-!
# The Flock instance's server: hm96 at the bound key (F1, F2)

`Server.ofBound Bd` is the bound record's server:

* `Open(DE, hm96)` is the bound record's `IVStar`, and its opened views are the bound record's, both by unfolding,
  case by case on how the last message reads (`ivStarOfBound`, `openViewOfBound`), so what `E^rep` reads through hm96's opening at the element-level Flock is
  what it reads at the bound record (`opens_openView_ofBound`);
* it binds where a row's digest input is injective: two passing openings of one commit string to different rows collide
  on the salts' hashes where the salts differ and on the rows' digests where they share it (`bindsOfBound`, from
  `Bound.opens_coll`);
* its commit strings open at a hash of 64-byte outputs (`opensOfBound`): hm96's commit string is the bound record's
  honest one (`commit_ofBound_eval`, `Bound.commitRow`), which opens (`Bound.opens_commitRow`).
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

/-- **F1: `Open(DE, hm96)` is the bound record's `IVStar`**: `Server.IVStar` at `Server.ofBound` unfolds to
`Bound.IVStar`, case by case on how the last message reads (`final`). -/
theorem ivStarOfBound {Coin Col T : Type} [Inhabited Coin] (Bd : Bound Coin Col T) : IVStarOfBound Bd := by
  unfold IVStarOfBound Server.IVStar Bound.IVStar
  dsimp only [Server.ofBound]
  congr 1
  funext x as cs
  rcases Bd.final (as.getLastD []) with _ | ⟨os, last⟩ <;> rfl

/-- **F1: hm96's opened views are the bound record's**: `Server.openView` at `Server.ofBound` unfolds to
`Bound.openView`, case by case on how the last message reads (`final`). -/
theorem openViewOfBound {Coin Col T : Type} (Bd : Bound Coin Col T) : OpenViewOfBound Bd := by
  unfold OpenViewOfBound
  funext v
  unfold Server.openView Bound.openView
  dsimp only [Server.ofBound]
  rcases Bd.final (v.msgs.getLastD ([], Out.rej)).1 with _ | ⟨os, last⟩ <;> rfl

/-- **The contract's reader at hm96 is the bound record's**: what `E^rep` reads of a view at the element-level Flock
through hm96's opening (`RecoversLinkOn`'s reader) is `DE`'s columns of the bound record's opened view, the reading of
`Inner.ofFlock` at `Bound.deploy` (`openViewOfBound`). -/
theorem opens_openView_ofBound {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T) (Cp : Caps Coin) (v : Seen) :
    (Proto.ofFlock Bd decI Cp).opens ((Server.ofBound Bd).openView v) = (Inner.ofFlock Bd.deploy decI).opens v :=
  congrArg (Proto.ofFlock Bd decI Cp).opens (congrFun (openViewOfBound Bd) v)

/-- **hm96's commit string is the bound record's honest one**: `b ‖ c` with `b = H(rowIn row) ⊕ Φy` and
`c = H(p_salt ‖ y)` (`Bound.commitRow`). -/
theorem commit_ofBound_eval {Coin Col T : Type} (Bd : Bound Coin Col T) (H : Msg → Msg) (row : Msg)
    (y : Fin Bd.saltLen → UInt8) : (((Server.ofBound Bd).commit row y).eval H).1 = Bd.commitRow H row y :=
  rfl

/-- **F2: hm96 binds where a row's digest input is injective**: two passing openings of one commit string to different
rows show a collision among their calls, on the salts' hashes where the salts differ and on the rows' digests where
they share it (`Bound.opens_coll`). -/
theorem bindsOfBound {Coin Col T : Type} (Bd : Bound Coin Col T) : BindsOfBound Bd := by
  intro hB H a row row' y y' hne h h'
  obtain ⟨u, hu, v, hv, hc⟩ := Bd.opens_coll H hB h h' fun he => hne (Prod.mk.inj he).1
  exact ⟨u, List.mem_append_left _ hu, v, List.mem_append_right _ hv, hc⟩

/-- **F2: hm96's commit strings open**, at a hash whose outputs are 64 bytes: the commit string is
`Bound.commitRow`'s, and `b ⊕ Φy` unmasks to the row's digest (`Bound.opens_commitRow`). -/
theorem opensOfBound {Coin Col T : Type} (Bd : Bound Coin Col T) : OpensOfBound Bd := fun _ h64 row y =>
  Bd.opens_commitRow h64 row y

end Claims.RecGeneric
