module

public import Definitions.Claims.ZKOut
public import Definitions.Claims.Recursion
public import Proofs.Flock.Recursive.Statements
public import Proofs.Flock.Claims.ZK.Statement

/-!
# The ZK server, as the recursion's zero knowledge reads it

Zero knowledge is the claims core's one notion, per public outcome (`Claims.ZKOut`): of the compiled prover the ZK
server runs (`ZkRelease.prover`), against every malicious verifier of its rounds, at a simulator fixed before the
witness and the verifier. It reads the server through what it sends, never through Flock's message formats:

* **hiding leaves** (`Leaves`): the server commits to every inner message under a fresh salt, and the verifier sees
  only the commitment (`leaf`). `Leaves.Hides δ` is hiding against adaptive messages: message `i` may read the
  commitments before it, since each inner coin is live, sampled after the commitment before it, and a malicious
  verifier's coin is a function of its tape and what it has seen; swapping every commitment for an ideal one (`ideal`,
  reading no message) raises the probability of any event by at most `n·δ`.
* **the release** (`ZkRelease`): at a compiled statement `x`, a first part of `n_h(x)` rounds, round `i` carrying
  `n_ℓ(x, i)` leaves (both protocol structure, public): the leaves of the workers' values for the round, each under a
  fresh salt, inside a public frame (`frame`, which reads the statement, the transcript so far and the leaves, as the
  deployed record's staging message `Claims.Recursion.Deploy.stage` does; a bare leaf where `n_ℓ = 1`); then the outer
  phase, `k_O(x)` rounds at the statement `x_O` the first part's transcript fixes, run by the outer prover at the
  witness the workers' values and the salts give (`wo`) where it is in the outer relation (`ZkRelease.ok`), each
  message tagged (`tagOuter`); and otherwise the refusal (`refusal`) every round left, which reads nothing. The outer
  phase is any prover; at Flock it is the V\* sessions, each the ZK server's release of a session
  (`Claims.ZKServer.ZkServer`).
* **the outcome** (`ZkRelease.outcome`), read off the transcript, so public: `none` where no outer view is released
  (the verifier stopped in the first part, or the server refused), and otherwise the outer phase's (`outO`: its
  verdict and where it stopped).
* **the simulator** (`ZkRelease.sim`): ideal leaves for the first part, then, given the outcome, the refusal or the
  outer simulator (`ZkRelease.simO`) at `x_O`. It reads the verifier, public data and the outcome only.

`GenericRecursiveZK` (Z1): where the leaves hide within `δ` and the outer phase is zero knowledge per outcome
(`ZkRelease.OuterZK`: `ZKOut` at each compiled statement, on every witness in the outer relation), the compiled
prover is zero knowledge per outcome at every witness (`ZKOut`), within the outer error at the released outcome,
averaged over the first part (`ZkRelease.err`), plus `δ` per leaf. A misbehaving worker costs only the refusal's
outcome: the refused view reads no value. Without the refusal the outer prover would run on a non-witness, where its
zero knowledge says nothing. Summed over the outcomes it is the leak form (`Claims.zkOutLeak`).

At hm96 (`Leaves.hm96`, through any encoding of a commit string's two halves) `Hides` holds at the key's hiding gap
(`Hm96Hides`, Z2, from `ZK.ideal_leaves_adaptive`).

Z1 takes the outer simulator as any program, so its simulator is counted only as the outer one is:
`GenericRecursiveZKCost` (Z1-cost) bounds the compiled simulator's calls by the frames' and the outer simulator's
(`Claims.ZKMal.SimCostLe`).

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion FlockSoundness.Discharge
open scoped NNReal ENNReal

/-! ## Hiding leaves -/

/-- **Hiding commitments as the verifier sees them**: salts (`Salt`) and ideal randomness (`Rand`), both finite and
nonempty, the commitment to a message under a salt (`leaf`), and an ideal commitment, which reads no message
(`ideal`). -/
structure Leaves where
  Salt : Type
  [saltFin : Fintype Salt]
  [saltNe : Nonempty Salt]
  Rand : Type
  [randFin : Fintype Rand]
  [randNe : Nonempty Rand]
  leaf : Msg → Salt → Msg
  ideal : Rand → Msg

attribute [instance] Leaves.saltFin Leaves.saltNe Leaves.randFin Leaves.randNe

/-- **The commitments to adaptive messages**: commitment `i` is `leaf` of message `i`, which reads the commitments
before it, under the salt `y i`. -/
def Leaves.commits (Lv : Leaves) {n : ℕ} (msgs : (i : Fin n) → (Fin i → Msg) → Msg) (y : Fin n → Lv.Salt)
    (i : Fin n) : Msg :=
  Lv.leaf (msgs i fun j => Lv.commits msgs y (Fin.castLE i.isLt.le j)) (y i)
termination_by i.val
decreasing_by exact j.isLt

/-- **The leaves hide, against adaptive messages, within `δ` each**: in an experiment on a uniform `ω` and salts, whose
message `i` reads `ω` and the commitments before it, swapping every commitment for an ideal one on fresh uniform `Rand`s
raises the probability of any event by at most `n·δ`. -/
def Leaves.Hides (Lv : Leaves) (δ : ℝ) : Prop :=
  ∀ (n : ℕ) (Ω : Type) [Fintype Ω] [Nonempty Ω] (msgs : Ω → (i : Fin n) → (Fin i → Msg) → Msg)
    (G : Ω → (Fin n → Msg) → Prop),
    Pr (fun p : Ω × (Fin n → Lv.Salt) => G p.1 (Lv.commits (msgs p.1) p.2)) ≤
      Pr (fun p : Ω × (Fin n → Lv.Rand) => G p.1 fun i => Lv.ideal (p.2 i)) + n * δ

/-- **hm96 leaves through an encoding `enc` of a commit string's two halves**: the commitment to `m` under the salt `y`
is `enc(SHA-512(m) ⊕ M·y, c(y))`, the ideal one `enc(U, c(y))` (`OuterZK.idealStrs`'s). -/
def Leaves.hm96 (enc : ZkView.Bits × ByteArray → Msg) : Leaves where
  Salt := ZkView.Salt
  Rand := ZkView.Bits × ZkView.Salt
  leaf m y := enc (ZkView.bitsOf (Flock.Sha512.hash ⟨m.toArray⟩) + ZkView.My y, ZkView.saltHash y)
  ideal u := enc (u.1, ZkView.saltHash u.2)

/-! ## The ZK server -/

/-- An outer phase's message as the server sends it, tagged `0`, so that no outer message reads as the refusal. -/
def tagOuter (m : Msg) : Msg := 0 :: m

/-- The outer message a tagged one carries. -/
def untagOuter : Msg → Msg
  | [] => []
  | _ :: m => m

/-- **The server's refusal**, sent every round after the first part where the workers' values are no witness. -/
def refusal : Msg := [1]

/-- **The ZK server's release at the leaves `Lv`, for the developer's witnesses `W`**, at each compiled statement `x`:

* the first part: `nh x` rounds, round `i` carrying `nl x i` leaves, on the workers' tape `Ωp x` and one salt a leaf;
  round `i`'s message is `frame` at the transcript so far and the round's leaves, leaf `j` the leaf of entry `j` of the
  workers' values for the round (`msg`, from the history so far; `[]` past its end) under its salt;
* the outer phase: `ko x` rounds at the statement `xo x pre` that the first part's transcript `pre` fixes, by the
  outer prover `PO` on its tape `NO x × ΩO x`, at the witness `wo` (the workers' values and the salts), where that
  witness is in the outer relation `RO x`; its outcome `outO x` (its verdict and where it stopped) at its statement
  and transcript. The tape, the relation and the outcome are `x`'s, so the outer phase's layout (its sessions) may be
  fixed by the compiled statement;
* the outer simulator (`progO`, on coins `CO x`), given the outer statement and outcome, fixed before the witness and
  the verifier. -/
structure ZkRelease (W : Type) (Lv : Leaves) where
  nh : Msg → ℕ
  nl : Msg → ℕ → ℕ
  frame : Msg → List (Msg × Out) → List Msg → OC Msg
  Ωp : Msg → Type
  [pFin : ∀ x, Fintype (Ωp x)]
  [pNe : ∀ x, Nonempty (Ωp x)]
  msg : W → (x : Msg) → Ωp x → List (Msg × Msg) → OC (List Msg)
  ko : Msg → ℕ
  xo : Msg → List (Msg × Out) → Msg
  WO : Type
  wo : W → (x : Msg) → Ωp x → (Fin (∑ i ∈ Finset.range (nh x), nl x i) → Lv.Salt) → List (Msg × Out) → WO
  RO : Msg → Rel WO
  NO : Msg → Type
  ΩO : Msg → Type
  [nFin : ∀ x, Fintype (NO x)]
  [nNe : ∀ x, Nonempty (NO x)]
  [oFin : ∀ x, Fintype (ΩO x)]
  [oNe : ∀ x, Nonempty (ΩO x)]
  PO : WO → (x : Msg) → Prover (NO x × ΩO x)
  St : Type
  outO : Msg → Msg → List (Msg × Out) → St
  CO : Msg → Type
  [cFin : ∀ x, Fintype (CO x)]
  [cNe : ∀ x, Nonempty (CO x)]
  progO : (x : Msg) → Msg → St → CO x → SP (List (Msg × Out))

attribute [instance] ZkRelease.pFin ZkRelease.pNe ZkRelease.nFin ZkRelease.nNe ZkRelease.oFin ZkRelease.oNe
  ZkRelease.cFin ZkRelease.cNe

namespace ZkRelease

variable {W : Type} {Lv : Leaves}

/-- The first part's leaves at `x`, every round's. -/
abbrev nLeaves (Zs : ZkRelease W Lv) (x : Msg) : ℕ := ∑ i ∈ Finset.range (Zs.nh x), Zs.nl x i

/-- The leaves before first-part round `i`. -/
def off (Zs : ZkRelease W Lv) (x : Msg) (i : ℕ) : ℕ := ∑ i' ∈ Finset.range i, Zs.nl x i'

/-- **Leaf `j` of first-part round `i`**, among the first part's leaves: the `off x i + j`-th. -/
def leafAt (Zs : ZkRelease W Lv) (x : Msg) {i : ℕ} (hi : i < Zs.nh x) (j : Fin (Zs.nl x i)) : Fin (Zs.nLeaves x) :=
  ⟨Zs.off x i + j, by
    have h1 : Zs.off x i + Zs.nl x i = ∑ i' ∈ Finset.range (i + 1), Zs.nl x i' := (Finset.sum_range_succ _ _).symm
    have h2 : ∑ i' ∈ Finset.range (i + 1), Zs.nl x i' ≤ Zs.nLeaves x :=
      Finset.sum_le_sum_of_subset (Finset.range_subset_range.2 hi)
    omega⟩

/-- The compiled prover's nonce at `x`: the workers' tape and the first part's salts, one a leaf. -/
abbrev N (Zs : ZkRelease W Lv) (x : Msg) : Type := Zs.Ωp x × (Fin (Zs.nLeaves x) → Lv.Salt)

/-- The rest of the compiled prover's tape at `x`: the outer prover's. -/
abbrev Ω (Zs : ZkRelease W Lv) (x : Msg) : Type := Zs.NO x × Zs.ΩO x

/-- The compiled verifier's rounds: the first part's, then the outer phase's. -/
def k (Zs : ZkRelease W Lv) (x : Msg) : ℕ := Zs.nh x + Zs.ko x

/-- **The server releases**: the workers' values and the salts give a witness in the outer relation at the statement
the first part's transcript `pre` fixes. -/
def ok (Zs : ZkRelease W Lv) (w : W) (x : Msg) (t : Zs.N x) (pre : List (Msg × Out)) : Prop :=
  Zs.RO x (Zs.xo x pre) (Zs.wo w x t.1 t.2 pre)

open Classical in
/-- **The compiled prover** at the witness `w` and the statement `x`: the first part's rounds, each its frame at its
leaves, then the outer prover's messages, tagged, where the server releases, and the refusal otherwise. -/
noncomputable def prover (Zs : ZkRelease W Lv) (w : W) (x : Msg) : Prover (Zs.N x × Zs.Ω x) where
  next _ hist t :=
    if h : hist.length < Zs.nh x then
      OC.bind (Zs.msg w x t.1.1 hist) fun vs =>
        Zs.frame x (hist.map fun p => (p.1, Out.msg p.2))
          (List.ofFn fun j => Lv.leaf (vs.getD j []) (t.1.2 (Zs.leafAt x h j)))
    else
      let pre := (hist.take (Zs.nh x)).map fun p => (p.1, Out.msg p.2)
      if Zs.ok w x t.1 pre then
        OC.bind ((Zs.PO (Zs.wo w x t.1.1 t.1.2 pre) x).next (Zs.xo x pre)
          ((hist.drop (Zs.nh x)).map fun p => (untagOuter p.1, p.2)) t.2) fun m => .ret (tagOuter m)
      else .ret refusal

/-- **The released outcome**, off the transcript: `none` where the run ends in the first part or the server refuses,
and otherwise the outer phase's outcome at its statement and its untagged transcript. -/
def outcome (Zs : ZkRelease W Lv) (x : Msg) (ms : List (Msg × Out)) : Option Zs.St :=
  if ms.length ≤ Zs.nh x ∨ (ms.drop (Zs.nh x)).head?.map Prod.fst = some refusal then none
  else some (Zs.outO x (Zs.xo x (ms.take (Zs.nh x))) ((ms.drop (Zs.nh x)).map fun p => (untagOuter p.1, p.2)))

/-- The outer simulator at the compiled statement `x`, on the coins `CO x` at every outer statement. -/
def simO (Zs : ZkRelease W Lv) (x : Msg) : SimOut Zs.St where
  C _ := Zs.CO x
  prog xo o c := Zs.progO x xo o c

/-- **The outer phase is zero knowledge per outcome** at each compiled statement `x`: `ZKOut` of the outer prover on
the tape `NO x × ΩO x`, against every verifier of the outer phase's `ko x` rounds, on every witness in the outer
relation, at the outer simulator `simO x` and the error `εO x`. -/
def OuterZK (Zs : ZkRelease W Lv) (H : Msg → Msg)
    (εO : (x : Msg) → Msg → Zs.WO → (Θ : Type) → VStar Θ → Θ → Zs.St → ℝ) : Prop :=
  ∀ x, ZKOut (fun _ => Zs.NO x) (fun _ => Zs.ΩO x) H (fun _ => Zs.ko x) (fun wo _ => Zs.PO wo x) (Zs.RO x)
    (Zs.outO x) (Zs.simO x) (εO x)

/-- The first part's transcript against `V*` at `θ`, on the nonce `t`: the compiled run cut at `nh x` rounds. -/
noncomputable def preRun (Zs : ZkRelease W Lv) (H : Msg → Msg) (w : W) (x : Msg) {Θ : Type} (Vs : VStar Θ) (θ : Θ)
    (t : Zs.N x) : List (Msg × Out) :=
  (stepWith H (Zs.prover w x) x (t, Classical.arbitrary _) (fun as => Vs.next x as θ) (Zs.nh x) [] []
    ⟨[], [], [], .rej⟩).msgs

end ZkRelease

/-- **`V*` after the first part's prover messages `pre`**, as the outer phase's verifier of `ko` rounds: its answer to
the outer messages, tagged, after `pre`. -/
def vAfter {Θ : Type} (Vs : VStar Θ) (x : Msg) (ko : ℕ) (pre : List Msg) : VStar Θ where
  k _ := ko
  next _ as θ := Vs.next x (pre ++ as.map tagOuter) θ

/-- A simulator program whose questions to the verifier follow the prover messages `pre`, its own tagged. -/
def spTag {α : Type} (pre : List Msg) : SP α → SP α
  | .ret a => .ret a
  | .hash q k => .hash q fun a => spTag pre (k a)
  | .ask as k => .ask (pre ++ as.map tagOuter) fun b => spTag pre (k b)

namespace ZkRelease

variable {W : Type} {Lv : Leaves}

open Classical in
/-- **The compiled prover's zero-knowledge error** at the outcome `s`: at an outer outcome, the outer error at `V*`
after the first part, averaged over the first part's nonce where the server releases; and `δ` for each of the first
part's leaves at every outcome, the refusal's included. -/
noncomputable def err (Zs : ZkRelease W Lv) (H : Msg → Msg)
    (εO : (x : Msg) → Msg → Zs.WO → (Θ : Type) → VStar Θ → Θ → Zs.St → ℝ) (δ : ℝ) (x : Msg) (w : W) (Θ : Type)
    (Vs : VStar Θ) (θ : Θ) (s : Option Zs.St) : ℝ :=
  s.elim 0 (fun o => Ex fun t : Zs.N x =>
      if goesOn (Zs.preRun H w x Vs θ t) ∧ Zs.ok w x t (Zs.preRun H w x Vs θ t) then
        εO x (Zs.xo x (Zs.preRun H w x Vs θ t)) (Zs.wo w x t.1 t.2 (Zs.preRun H w x Vs θ t)) Θ
          (vAfter Vs x (Zs.ko x) ((Zs.preRun H w x Vs θ t).map Prod.fst)) θ o
      else 0) + Zs.nLeaves x * δ

/-- **The compiled prover's simulator, given the outcome**: the first part's rounds, each its frame at ideal leaves,
then, where the run goes on, the refusal every round at `none`, or the outer simulator given `o` at the outer
statement the simulated first part fixes, its questions and messages tagged, at `some o`. -/
def sim (Zs : ZkRelease W Lv) : SimOut (Option Zs.St) where
  C x := (Fin (Zs.nLeaves x) → Lv.Rand) × Zs.CO x
  prog x s c :=
    spBind (simRounds (fun ms => if h : ms.length < Zs.nh x then
          spOf (Zs.frame x ms (List.ofFn fun j => Lv.ideal (c.1 (Zs.leafAt x h j))))
        else .ret []) (Zs.nh x) []) fun ms =>
      if goesOn ms then
        match s with
        | none => simRounds (fun _ => .ret refusal) (Zs.ko x) ms
        | some o => spBind (spTag (ms.map Prod.fst) (Zs.progO x (Zs.xo x ms) o c.2)) fun ts =>
            .ret (ms ++ ts.map fun p => (tagOuter p.1, p.2))
      else .ret ms

end ZkRelease

/-- **Z2: hm96 leaves hide at the key's hiding bound**, through every encoding of the commit string (from
`ZK.ideal_leaves_adaptive`). -/
def Hm96Hides : Prop :=
  ∀ (enc : ZkView.Bits × ByteArray → Msg) (δ : ℝ≥0), Claims.Hm96Hiding ZkView.My ZkView.saltHash (δ : ℝ≥0∞) →
    (Leaves.hm96 enc).Hides δ

end Claims.RecGeneric

namespace Flock.Guarantees

open Claims Claims.RecGeneric

/-- **Z1: the recursion is zero knowledge per outcome (`Claims.ZKOut`), generic in the server's leaves and the outer
phase.** For every release `Zs` of the ZK server (its outer simulator fixed before the witness and the verifier) and
outer error `ε_O`: where the leaves hide within `δ` and the outer phase is zero knowledge per outcome at its simulator
within `ε_O` on every witness in the outer relation (`ZkRelease.OuterZK`), the compiled prover is zero knowledge per
outcome at every witness, at the simulator `ZkRelease.sim` given the released outcome (the outer phase's `some o`, or
`none` where no outer view is released), within `ZkRelease.err`: the outer error averaged over the first part where
the server releases, at `some o` only, plus `δ` per leaf of the first part. The outer simulator `progO` is any
program, and Z1 transfers whatever `OuterZK` gives it: a simulator that picks an outer witness by choice can meet
`OuterZK`, so the result is zero knowledge in the usual sense only at a counted simulator (Flock's at the instance). -/
def GenericRecursiveZK : Prop :=
  ∀ (H : Msg → Msg) {W : Type} (Lv : Leaves) (Zs : ZkRelease W Lv)
    (εO : (x : Msg) → Msg → Zs.WO → (Θ : Type) → VStar Θ → Θ → Zs.St → ℝ) (δ : ℝ),
    Lv.Hides δ → Zs.OuterZK H εO →
      ZKOut Zs.N Zs.Ω H Zs.k Zs.prover (fun _ _ => True) Zs.outcome Zs.sim (Zs.err H εO δ)

/-- **Z1-cost: the compiled simulator is as counted as the outer one.** Where each frame makes at most `c_F(x)` calls
and the outer simulator at most `c_O(x)`, on every coin, at every outer statement and outcome and against every
verifier, the compiled simulator given the outcome makes at most `n_h(x)·(c_F(x) + 1)` calls in the first part (each
round its frame and one question to `V*`), then at most `k_O(x)` at the refusal (one question a round) or `c_O(x)`
at an outer outcome. -/
def GenericRecursiveZKCost : Prop :=
  ∀ (H : Msg → Msg) {W : Type} (Lv : Leaves) (Zs : ZkRelease W Lv) (cF cO : Msg → ℕ),
    (∀ x pre ls, ((Zs.frame x pre ls).eval H).2.length ≤ cF x) →
    (∀ x xO o (c : Zs.CO x) ask, ((Zs.progO x xO o c).eval H ask).2 ≤ cO x) →
    ZKMal.SimCostLe H (Zs.sim.given none) (fun x => Zs.nh x * (cF x + 1) + Zs.ko x) ∧
      ∀ o, ZKMal.SimCostLe H (Zs.sim.given (some o)) (fun x => Zs.nh x * (cF x + 1) + cO x)

end Flock.Guarantees
