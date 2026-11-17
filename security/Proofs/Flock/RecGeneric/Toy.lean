module

public import Proofs.Flock.RecGeneric.Outer

/-! A toy outer argument `echoArg` and its compiler `Compiled.echo`, at which L2's hypotheses hold together (`EchoHolds`); definitions, no proofs. -/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion
open Claims.SamplingGeneric (Arg)

/-- **The echo verifier**: one round; it accepts where the prover's message is the statement or its coin is
`default`. -/
def echoV (Coin : Type) [Inhabited Coin] [DecidableEq Coin] : Verifier Coin :=
  ⟨fun _ => 1, fun x as cs => .ret (if as.headD [] = x ∨ cs.headD default = default then .acc else .rej)⟩

/-- **The echo argument**: `echoV`, the witness of `x` being `x`, and `E^rep` at one wait of one fresh rerun, its
candidate the accepted rerun's first message in one step, unchecked, at `κ = 1/|Coin|` and `γ = 0`. -/
noncomputable def echoArg (Coin : Type) [Fintype Coin] [Inhabited Coin] [DecidableEq Coin] : Arg Coin where
  Sh := Unit
  shape _ := ()
  V := echoV Coin
  rounds _ := 1
  wf _ := rfl
  W := Msg
  R x w := w = x
  Op := Msg
  opens v := v.msgs.map Prod.fst
  assemble _ os := (os.head?, 1)
  check _ _ := true
  waits _ := 1
  k _ := 1
  cap _ := 1
  κ _ := 1 / (Fintype.card Coin : ℝ)
  γ _ := 0

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Coin]

/-- **The echo compiler's verifier** at the inner verifier `IV`, with `k = k_IV(x)` and `2k + 4` rounds: rounds
`0..k−1` answer with `IV`'s messages (`[]` where `IV` decides); rounds `k..k+2` stage (`[]`); round `k+3+s` is session
`s`, which rejects unless its message is round `s`'s or its coin is `default`; the last round accepts where `IV`'s run
on the first `k` messages at the first `k` coins (`Comp.replayOC`) accepts. It rejects where `k = 0`. -/
def echoCV (IV : Verifier Coin) : Verifier Coin where
  k x := 2 * IV.k x + 4
  next x as cs :=
    if IV.k x = 0 then .ret .rej
    else if as.length ≤ IV.k x then
      (IV.next x as cs).bind fun b => .ret (match b with
        | .msg m => .msg m
        | _ => .msg [])
    else if as.length ≤ IV.k x + 3 then .ret (.msg [])
    else if as.length ≤ 2 * IV.k x + 3 then
      .ret (if as.getLastD [] = as.getD (as.length - (IV.k x + 4)) [] ∨ cs.getLastD default = default then .msg []
        else .rej)
    else (Comp.replayOC IV x (as.take (IV.k x)) (fun i => cs.getD i default) (IV.k x) [] []).bind fun v =>
      .ret (if v.out = .acc then .acc else .rej)

/-- **The echo compiler**: at the inner verifier `IV`, the verifier `echoCV IV`; every statement parses and is its own
inner statement; `k` sessions after `k + 3` rounds, each within `echoArg`'s errors, session `s`'s statement the inner
round `s`'s message and its view the message of round `k+3+s`, accepted where the run accepts or that round did not
reject; the inner messages are the witnesses; a rerun's searched evaluations are `IV`'s calls in its run on them, and the
compiled verifier makes none. -/
noncomputable def Compiled.echo : Compiler Coin (echoArg Coin) := fun IV =>
  { V := echoCV IV
    parses := fun _ => True
    xI := id
    pre := fun x => IV.k x + 3
    nS := fun x => IV.k x
    sh := fun _ _ => ()
    κb := fun _ _ => 1 / (Fintype.card Coin : ℝ)
    γb := fun _ _ => 0
    stmts := fun x ms => (ms.take (IV.k x)).map Prod.fst
    session := fun x s v =>
      let e := v.msgs.getD (IV.k x + 3 + s) ([], .rej)
      let o : Out := if v.out = .acc ∨ e.2 ≠ .rej then .acc else .rej
      ⟨[(e.1, o)], o⟩
    msgs := fun _ ws => ws.map Prod.snd
    calls := fun H x r ws => (run H (replay (ws.map Prod.snd)) IV x () fun i => coinSeq r i.val).vLog
    own := fun _ _ _ => [] }

/-- The echo argument's outer extractor may give a non-witness on an accepted run. -/
def EchoHolds : Prop :=
  ∀ (Coin W : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Coin] (H : Msg → Msg)
    (I : Proto Coin W) (S : Server),
    (echoArg Coin).KS H ∧ (compile Compiled.echo I S).Sessions H ∧
      ∀ x, (compile Compiled.echo I S).toComp.Contract H (S.IVStar I.V I.ans) (fun v => I.opens (S.openView v)) x

end Claims.RecGeneric
