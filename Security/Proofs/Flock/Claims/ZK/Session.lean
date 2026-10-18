import Proofs.Flock.Claims.ZK.Capacity

/-!
# The session's premises at the deployed masked record, round by round

`Flock.zk_deployZk_ref_on` rests on the reference session's table data (`TableData`) through the session's Props on a
class `A` of statements and `RefSession.HonestMsgsOn A`: within its rounds, at a statement in `A`, `refP`'s message is
the record's reference prover's (`FlockComplete.ZkTables.honestP`, `(FlockDeploy.deployZk rel).P`). `refP`'s message
after the openings `ops` is one of three (`RefSession.msgAt`), so `HonestMsgsOn A` is three Props, one per kind of round:

* `HonestPreOn A`: at a pre-final round (`ops.length < refK x`), the round encoding `enc` of the cut pre-final view;
* `HonestFinOn A`: at the last round, where every table admits the coins, `fin` of the tables' views;
* `HonestOffOn A`: at the last round, where a table refuses the coins, `off` of the cut view;

each equal to `honestP`'s next message at the history with each opening read as its coin (`coinOf`), at the tape
`tapeOf (ν, t)`. `RefSession.honestMsgsOn_of` assembles them, and `Flock.zk_deployZk_ref_on_rounds` is
`Flock.zk_deployZk_ref_on` with the three in place of `HonestMsgsOn A`.
-/

namespace Claims.ZKMal

open Claims FlockSoundness.Merkle FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

/-- **`refP`'s pre-final messages are the reference prover's at the statements in `A`**: after `ops`, fewer than
`refK x` openings, `enc` of the cut pre-final view at the coins they open is `honestP`'s next message. -/
def RefSession.HonestPreOn (S : RefSession) (A : Msg → Prop) (H : Msg → Msg) (coinOf : Msg → Msg)
    (tapeOf : S.refP.Nonce × S.refP.Rand → FlockComplete.TapeZk) : Prop :=
  ∀ x, A x → ∀ w ν t (ops : List (Msg × Msg)), ops.length < refK x →
    S.enc x (S.pad (ops.map fun p => S.decOf p.2)) ops.length
        (S.cutOf x (S.pad (ops.map fun p => S.decOf p.2)) (S.wt x w) t) =
      (((FlockComplete.ZkTables.honestP w).next x (ops.map fun p => (p.1, coinOf p.2)) (tapeOf (ν, t))).eval H).1

/-- **`refP`'s last message is the reference prover's where the tables admit the coins**, at the statements in `A`:
after `refK x` openings whose coins every table admits, `fin` of the tables' views is `honestP`'s next message. -/
def RefSession.HonestFinOn (S : RefSession) (A : Msg → Prop) (H : Msg → Msg) (coinOf : Msg → Msg)
    (tapeOf : S.refP.Nonce × S.refP.Rand → FlockComplete.TapeZk) : Prop :=
  ∀ x, A x → ∀ w ν t (ops : List (Msg × Msg)), ops.length = refK x →
    S.Admits x (S.pad (ops.map fun p => S.decOf p.2)) →
    S.fin x (S.pad (ops.map fun p => S.decOf p.2))
        (S.viewsAt x (S.pad (ops.map fun p => S.decOf p.2)) (S.wt x w) t) =
      (((FlockComplete.ZkTables.honestP w).next x (ops.map fun p => (p.1, coinOf p.2)) (tapeOf (ν, t))).eval H).1

/-- **`refP`'s last message is the reference prover's where a table refuses the coins**, at the statements in `A`:
after `refK x` openings whose coins some table refuses, `off` of the cut view is `honestP`'s next message. -/
def RefSession.HonestOffOn (S : RefSession) (A : Msg → Prop) (H : Msg → Msg) (coinOf : Msg → Msg)
    (tapeOf : S.refP.Nonce × S.refP.Rand → FlockComplete.TapeZk) : Prop :=
  ∀ x, A x → ∀ w ν t (ops : List (Msg × Msg)), ops.length = refK x →
    ¬ S.Admits x (S.pad (ops.map fun p => S.decOf p.2)) →
    S.off x (S.pad (ops.map fun p => S.decOf p.2))
        (S.cutOf x (S.pad (ops.map fun p => S.decOf p.2)) (S.wt x w) t) =
      (((FlockComplete.ZkTables.honestP w).next x (ops.map fun p => (p.1, coinOf p.2)) (tapeOf (ν, t))).eval H).1

/-- **`HonestMsgsOn A` from its three rounds**: `msgAt` is `enc` before the last coin, then `fin` or `off`. -/
theorem RefSession.honestMsgsOn_of (S : RefSession) (A : Msg → Prop) (H : Msg → Msg) (coinOf : Msg → Msg)
    (tapeOf : S.refP.Nonce × S.refP.Rand → FlockComplete.TapeZk) (hpre : S.HonestPreOn A H coinOf tapeOf)
    (hfin : S.HonestFinOn A H coinOf tapeOf) (hoff : S.HonestOffOn A H coinOf tapeOf) :
    S.HonestMsgsOn A H coinOf tapeOf := by
  intro x hx w ν t ops hops
  have e : (ops.map fun p => S.decOf p.2).length = ops.length := by simp
  unfold RefSession.msgAt
  split_ifs with h1 h2
  · rw [e]
    exact hpre x hx w ν t ops (by simpa using h1)
  · exact hfin x hx w ν t ops (by rw [e] at h1; omega) h2
  · exact hoff x hx w ν t ops (by rw [e] at h1; omega) h2

/-- **`Flock.zk_deployZk_ref_on` round by round**: `Claims.Flock.ZK` at the deployed masked record, at `ε_zk` on the
statements in `A` and `1` outside them, with `HonestMsgsOn A` as its three rounds (`HonestPreOn`, `HonestFinOn`,
`HonestOffOn`). -/
theorem Flock.zk_deployZk_ref_on_rounds (H : Msg → Msg)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (nr : Msg → ℕ → ℕ) (T : TableData)
    (A : Msg → Prop) [DecidablePred A] (t : ℕ) (ht : 0 < t) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hS : (session nr T).Tables)
    (hV : (session nr T).Complete fun x w => A x ∧ Flock.R (FlockDeploy.deployZk rel).parse x w)
    (hD : (session nr T).DummyReg) (hc : (session nr T).Causal) (hT : ∀ x, A x → FibersEq (T.tape x))
    (hpre : (session nr T).HonestPreOn A H coinOfD fun ω => T.tapeZk ω.2)
    (hfin : (session nr T).HonestFinOn A H coinOfD fun ω => T.tapeZk ω.2)
    (hoff : (session nr T).HonestOffOn A H coinOfD fun ω => T.tapeZk ω.2) :
    Claims.Flock.ZK H (zkDeployed rel nr T t)
      (fun x => if A x then 8 * (session nr T).nLeaves x * δ₁.toReal +
        8 * t * (session nr T).nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t else 1)
      fun _ => 4 * t * 3 ^ t :=
  Flock.zk_deployZk_ref_on H rel nr T A t ht hT1 hδ₁ hS hV hD hc hT
    ((session nr T).honestMsgsOn_of A H coinOfD _ hpre hfin hoff)

end Claims.ZKMal
