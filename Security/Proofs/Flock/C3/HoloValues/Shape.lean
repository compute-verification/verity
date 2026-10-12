import Proofs.Flock.Soundness.Discharge.Exec.Shape
import Proofs.Flock.Soundness.Discharge.PlaysCoin.Coins
import Proofs.Flock.Soundness.Model.HoloCompiled
import Proofs.Flock.VBridge.InnerAccepts

/-!
# A rep with hidden units has a spine

`repC_spine` (`Discharge/Exec/Shape.lean`) at `holoRepC`: the hidden units' lincheck (`holoLincheck`) receives `h` after
`z_partial`, so its shape (`holoLcShape`) is `lcShape` with one more message, and a rep's (`holoRepShape`) is
`repShape` with it. `playsT_of_spine` is `InnerAccepts.playsT_of_tape` at any spine: a run on a game's messages and
coins (`Plays`) is a run on any tape of the spine's kinds with those messages and coin steps (`PlaysT`).
-/

namespace FlockC3.HoloValues

open FlockSoundness FlockSoundness.Game FlockSoundness.Model FlockSoundness.Accounting FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.PlaysCoin (coinArr toT_exact)

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D : Type}

/-- The lincheck's shape with hidden units (L1–L5 with L4′): `lcShape` with `h` after `z_partial`. -/
def holoLcShape (F : Type) (kLog : ℕ) : List Kind :=
  .coin F :: .coin F :: (pairRounds F (kLog - 6) ++ [.msg (Fin 64 → F), .msg (Fin 64 → F), .coin F])

omit [Fintype K] [DecidableEq K] in
theorem holoLincheck_exact (A : Arith F K) (S : Statement) (zc : ZcOut F S.m) :
    Exact (holoLincheck A S zc) (holoLcShape F S.kLog) := by
  unfold holoLincheck holoLcShape
  refine Exact.draw_bind fun α => Exact.draw_bind fun β => ?_
  exact Exact.bind (lincheckRounds_exact _ _) fun r =>
    Exact.recv_bind fun zP => Exact.recv_bind fun h => Exact.draw_bind fun r' => .ret _

/-- **A rep's shape with hidden units**: `repShape` with `holoLcShape`. -/
def holoRepShape (F K D : Type) (m kLog nExtra : ℕ) (sch : Schedule) : List Kind :=
  zcShape F m ++ (holoLcShape F kLog ++ (opShape F (nExtra + 2) ++ ligShape F K D sch))

theorem spine_bind_pure {α β : Type} {g : Game α} {σ : List Kind} (hs : Spine g σ) (f : α → β) :
    Spine (g >>= fun a => pure (f a)) σ := by
  induction hs with
  | ret a σ => exact .ret _ σ
  | send _ ih => exact .send ih
  | coin _ ih => exact .coin ih

omit [Fintype K] in
/-- **A rep with hidden units has a spine**: every run of rep `r` takes its steps in the order of `holoRepShape`. -/
theorem holoRepC_spine (A : Arith F K) (r : ℕ) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m)
    (extra : List (Claim F S.m)) :
    Spine (holoRepC (D := D) A r S sch hm extra) (holoRepShape F K D S.m S.kLog extra.length sch) := by
  unfold holoRepC holoRepShape
  refine Exact.bindS (zerocheck_exact A S.m) fun z => ?_
  rcases z with _ | zc
  · exact .ret _ _
  · refine Exact.bindS (holoLincheck_exact A S zc) fun l => ?_
    rcases l with _ | lc
    · exact .ret _ _
    · refine Exact.bindS (opening_exact A _ (lc.ab :: cClaim A zc :: extra)) fun o => ?_
      rcases o with _ | op
      · exact .ret _ _
      · exact spine_bind_pure (ligeritoC_spine A r sch op.T op.b) _

/-- **A run on a game's messages and coins is a run on one tape** that takes steps of the kinds of the game's spine,
with those messages and coin steps: the spine places the coins among the messages. -/
theorem playsT_of_spine {α : Type} {g : Game α} {σ : List Kind} (hs : Spine g σ) {ms : List Game.Msg}
    {cs : List (Array Flock.F128)} {a : α} {ts : List Step} (h : Plays Refine.fOf g ms cs a)
    (hk : ts.map kindOf <+: σ) (hl : ts.filterMap Sum.getLeft? = ms)
    (hr : ts.filterMap Sum.getRight? = cs.map (coinArr Refine.fOf)) : PlaysT g ts a := by
  obtain ⟨ts', hp, h1, h2⟩ := toT_exact Refine.fOf h
  have hlen : ts.length = ts'.length := by
    rw [FlockVBridge.InnerAccepts.length_eq_lefts_rights ts, FlockVBridge.InnerAccepts.length_eq_lefts_rights ts',
      hl, hr, h1, h2]
  have hw : ts = weave (σ.take ts.length) (ts.filterMap Sum.getLeft?) (ts.filterMap Sum.getRight?) := by
    have k := List.prefix_iff_eq_take.mp hk
    rw [List.length_map] at k
    calc ts = weave (ts.map kindOf) (ts.filterMap Sum.getLeft?) (ts.filterMap Sum.getRight?) := (weave_self ts).symm
      _ = _ := by rw [k]
  have e : ts = ts' := by
    rw [hw, Spine.tape hp hs, hlen, hl, hr, h1, h2]
  rw [e]
  exact hp

end FlockC3.HoloValues
