import Proofs.Flock.Soundness.ZK.Table

/-!
# Zero knowledge: what a prover that stops has sent

The prover refuses three kinds of coins (`backends/flock/live/PROTOCOL.md` §7 "Refused coins", `flock-circuit.rs`'s
`refused_coins`), each checked just before the messages it guards and each stopping the whole session, after which no
proof is sent (the barrier: every proof follows the session's last coin):

* **the opening's rank check** (§6, `zk_masking`'s `at_open`): before rep `r`'s `ŝ`, the mask slot's weights must span
  over every masked claim point of the table so far, rep `r`'s and the earlier reps' (`all.extend(…)` then
  `mask_rank_ok(&all)`, one list per table);
* **the restart's `ρ` check** (§7 item 3): `ρ^r₀, ρ^r₁` independent over `F`, before `T'^r`, `ȳ^r` and levels ≥ 1;
* **the inner proof's `β` check** (§5 step 3): `β^r ≠ 0`, before `Y`.

The prover sends tables in order, each table's rep 0 before its rep 1 (`for rep in 0..2`: rep 1's stream opens after rep
0's last round), and in each rep the opening, then the restart, then the inner proof (§1). So the session stops at the
first failing check in that order, and a guarded message is sent exactly when every check up to its own passes.

* `Table.Sent`: which guarded messages one table sent. `Table.RankOn`: the rank check over a set of claims.
  `Table.Admits`: the checks that guard the sent messages passed.
* `Table.Cut` and `Table.cut`: the pre-final view in world `W₀` with the guarded messages a table didn't send removed.
  It keeps everything else whole, even where the prover stopped before sending it: a stopped prover's view is a fixed
  function of the cut (at fixed coins), so a statement about the cut is at least as strong.
* `Session.pos`, `Session.guard`, `Session.passes`, `Session.sent`: the session's order, its checks, and what each table
  sent when the session stops at the first failing check. `Refusal.Session.sent_admits` proves that it is admitted.
-/

namespace FlockSoundness.ZK

open Finset

namespace Table

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] (T : Table F K W Salt Seed D Hi VL)

/-- **Which guarded messages a table sent**: masked claim `k`'s `ŝ` (`cl`), rep `r`'s restarted table `y₁'^r` with levels
≥ 1 (`y1`), and rep `r`'s `Y` (`fin`). -/
structure Sent where
  cl : Fin T.nC → Prop
  y1 : Fin 2 → Prop
  fin : Fin 2 → Prop

variable [CharP F 2]

/-- **The rank check over the claims `P`** (§6): their weights on the mask slot span, so a GF(2) vector over the mask-slot
words reaches every vector of their values. `Rank` is the check over every claim. -/
def RankOn (P : Fin T.nC → Prop) : Prop :=
  Function.Surjective fun c : Fin T.nMask → ZMod 2 => fun k : {k // P k} => ∑ x, T.eqw k.1 x * castF (c x)

/-- **The sent messages' checks passed**: the rank check over the claims whose `ŝ` was sent, `W^r` invertible for each rep
whose restarted table was sent, `β^r ≠ 0` for each rep whose `Y` was sent. -/
def Admits (S : T.Sent) : Prop :=
  T.RankOn S.cl ∧ (∀ r, S.y1 r → Function.Bijective fun p : F × F => T.Wr r p.1 p.2) ∧ (∀ r, S.fin r → T.β r ≠ 0)

/-- **The pre-final view of a prover that may have stopped**, in world `W₀`: `PreView` with each guarded message optional. -/
@[ext] structure Cut where
  sHat : Fin T.nC → Option (Fin 128 → F)
  sReg : Fin T.nR → Fin 128 → F
  masked : Fin 2 → Fin T.nE → F
  y1 : Fin 2 → Option (Fin T.L → K)
  ybar : Fin 2 → Fin T.tpad → K
  lig : Fin 2 → Option VL
  inner : Fin 2 → F × F
  Y1 : Fin 2 → Option (PadIdx T.nE → F)
  Y2 : Fin 2 → Fin T.nPad → F
  leaves0 : (Fin T.nS → D) × (Fin T.nU → D)
  tauCom : Fin 2 → D
  leavesP : Fin 2 → (Fin T.nQ → D) × (Fin T.nPU → D)

open Classical in
/-- **The pre-final view cut to the sent messages**: a guarded message the table didn't send is `none`. -/
noncomputable def cut (S : T.Sent) (v : T.PreView) : T.Cut where
  sHat k := if S.cl k then some (v.sHat k) else none
  sReg := v.sReg
  masked := v.masked
  y1 r := if S.y1 r then some (v.y1 r) else none
  ybar := v.ybar
  lig r := if S.y1 r then some (v.lig r) else none
  inner := v.inner
  Y1 r := if S.fin r then some (v.Y1 r) else none
  Y2 := v.Y2
  leaves0 := v.leaves0
  tauCom := v.tauCom
  leavesP := v.leavesP

end Table

namespace Session

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] {J : ℕ} (T : Fin J → Table F K W Salt Seed D Hi VL)

/-- The three checks of a rep, in the order the prover makes them. -/
inductive Phase
  | opening
  | restart
  | inner

/-- A phase's place in its rep. -/
def Phase.idx : Phase → ℕ
  | .opening => 0
  | .restart => 1
  | .inner => 2

/-- **The session's order**: table `j`'s rep `r`'s phase `p` comes after every earlier table's, and after rep `r − 1`'s. -/
def pos (j : Fin J) (r : Fin 2) (p : Phase) : ℕ := (2 * j.val + r.val) * 3 + p.idx

variable [CharP F 2]

/-- **The check at table `j`, rep `r`, phase `p`.** The opening's is cumulative over the table's reps: the rank check over
every masked claim of reps `≤ r` (`rep j k` is claim `k`'s rep). -/
def guard (rep : ∀ j, Fin (T j).nC → Fin 2) (j : Fin J) (r : Fin 2) : Phase → Prop
  | .opening => (T j).RankOn fun k => rep j k ≤ r
  | .restart => Function.Bijective fun p : F × F => (T j).Wr r p.1 p.2
  | .inner => (T j).β r ≠ 0

/-- **The session reaches past table `j`, rep `r`, phase `p`**: every check up to it in the session's order passes. -/
def passes (rep : ∀ j, Fin (T j).nC → Fin 2) (j : Fin J) (r : Fin 2) (p : Phase) : Prop :=
  ∀ j' r' p', pos j' r' p' ≤ pos j r p → guard T rep j' r' p'

/-- **What table `j` sent** when the session stops at its first failing check: `ŝ` of the claims whose rep's opening
passed, the restarted tables of the reps whose restart passed, `Y` of the reps whose inner proof's check passed. -/
def sent (rep : ∀ j, Fin (T j).nC → Fin 2) (j : Fin J) : (T j).Sent where
  cl k := passes T rep j (rep j k) .opening
  y1 r := passes T rep j r .restart
  fin r := passes T rep j r .inner

end Session

end FlockSoundness.ZK
