import Proofs.Flock.Claims.Recursion.VStar

/-!
# V*'s parts at the representation V* binds (option (d)): the layout and part facts, restated

`VStar.lean` has two gaps at rounds no part holds:

* `PartChecks` quantifies a part's inner check over every list of inner messages that agrees with the rounds the part
  holds.
* `Decomp` asks the inner verifier to accept every list on which all the parts' checks hold.

At a round no part holds, the checks can say nothing. Round 0 is public there and the last round is never held, so
`Decomp` asks the inner verifier to accept whatever sits at both, which Flock refuses (custody and S16 read both).

At option (d)'s inner verifier (`Claims.Recursion.Bound.IVStar`) every round is read off V*'s parts:

* **Round 0 and each later round before the last:** the inner session's commit strings. Each is the `b ‖ c` a part
  computes from a row and a salt it holds (its salt-hash gadget's output, and its row-digest gadget's output masked at
  the salt), held whole by the parts that open it (`LayoutB.msgAt`). A round with no commit string carries the empty
  message the statement fixes (`LayoutB.fixedAt`).
* **The last round:** each earlier round's rows and salts, then the element-level last message (the proof files' rest),
  assembled from pieces across the parts that hold them (`LayoutB.pieceAt`). Each piece sits at a byte offset the
  statement fixes (`pieceLen`).

So the restated facts constrain every round:

* `HeldB`: the layout spans exactly the inner verifier's rounds, and each round, round 0 and the last included, is
  fixed by the statement, held whole by some part, or assembled from parts' pieces. The bound layout has no public
  round.
* `PartChecksB`: `PartChecks`, where the messages it quantifies over also agree with the part's pieces of the
  assembled rounds, at their offsets. A part's check can read its own pieces of the last message.
* `DecompB`: the inner verifier accepts the readers' messages (`LayoutB.msgs`, every round read off the witnesses)
  wherever every part's check holds on them, or two of the parts' recorded calls collide. Where several parts hold one
  commit string, the readers take the first holder's; a holder that opens it otherwise shows a collision among the
  calls they record, so that is the escape.

`FinalOpens` gives way to `Claims.Recursion.Bound.LastBinds`: the view's last message isn't the run's, and two accepted
views that differ only there collide among `IVStar`'s calls.

`LayoutB` is its own record here, the deployed layout's fields with the pieces added (segments of constant bytes and
hidden values, the statement's fixed rounds), so the statement builds on `VStar.Layout` as it stands.

Statements only. The proofs that V*'s parts meet them, and the route from them to `KSBound`, are to come.
-/

namespace Claims.Recursion.VStarB

open Claims
open Claims.Recursion.VStar (readAt)

/-- A message off a witness from its segments: each segment's constant bytes, then the bytes at its hidden values. -/
def readSegs (w : Flock.Wit) (segs : List (Msg × List ℕ)) : Msg := segs.flatMap fun p => p.1 ++ readAt w p.2

/-- A piece's byte length: each segment's constant bytes, then its hidden values' bytes, eight bits a byte. -/
def pieceLen (segs : List (Msg × List ℕ)) : ℕ := (segs.map fun p => p.1.length + (p.2.length + 7) / 8).sum

/-- **Where V\*'s parts hold the bound view** (option (d)), fixed by the compiled statement `x`: the inner rounds' count
(`rounds x`); where part `s` holds inner round `i`'s message whole (`msgAt x s i`, `none` where it holds none), as
segments, each constant bytes the statement fixes then the hidden values of the next bits, in order; each SHA-512
evaluation part `s` computes (`shaAt x s`): its input's constant prefix, then the hidden values of the rest of its
input's bits, and those of its output's; the rounds whose message the statement fixes (`fixedAt x i`), which no part
holds; and each round that no part holds whole, assembled from pieces (`pieceAt x i`, empty at a round held whole or
fixed), each a part and its segments, in order. A piece's bytes have the length its segments fix (`pieceLen`), so each
piece sits at an offset the statement fixes. -/
structure LayoutB where
  rounds : Msg → ℕ
  msgAt : Msg → ℕ → ℕ → Option (List (Msg × List ℕ))
  shaAt : Msg → ℕ → List (Msg × List ℕ × List ℕ)
  fixedAt : Msg → ℕ → Option Msg := fun _ _ => none
  pieceAt : Msg → ℕ → List (ℕ × List (Msg × List ℕ)) := fun _ _ => []

/-- Inner round `i`'s message as part `s` holds it whole, off its witness `w`: none at a round the statement fixes. -/
def LayoutB.held (L : LayoutB) (x : Msg) (s : ℕ) (w : Flock.Wit) (i : ℕ) : Option Msg :=
  if (L.fixedAt x i).isSome then none else (L.msgAt x s i).map (readSegs w)

/-- Round `i` assembled off the witnesses: each piece read off its part's witness, in order. Zero bytes fill a piece
whose part has no witness, so the later pieces keep their offsets. -/
def LayoutB.assembled (L : LayoutB) (x : Msg) (ws : List Flock.Wit) (i : ℕ) : Msg :=
  (L.pieceAt x i).flatMap fun p => (ws[p.1]?.map fun w => readSegs w p.2).getD (List.replicate (pieceLen p.2) 0)

/-- Inner round `i`'s message off the witnesses: the statement's where it fixes it, its assembled message where it has
pieces, else as the first part that holds it whole holds it, empty where none does. -/
def LayoutB.msg (L : LayoutB) (x : Msg) (ws : List Flock.Wit) (i : ℕ) : Msg :=
  match L.fixedAt x i with
  | some m => m
  | none =>
    if (L.pieceAt x i).isEmpty then
      ((List.range ws.length).findSome? fun s => ws[s]?.bind fun w => L.held x s w i).getD []
    else L.assembled x ws i

/-- **The bound readers' messages**: the inner session's `rounds x` messages, each off the witnesses (`LayoutB.msg`). -/
def LayoutB.msgs (L : LayoutB) (x : Msg) (ws : List Flock.Wit) : List Msg :=
  (List.range (L.rounds x)).map (L.msg x ws)

/-- **The bound readers' calls**: part `s`'s SHA-512 evaluations off its witness `w`, each its input (its constant
prefix, then the bytes at its hidden values) and its output read at their hidden values. -/
def LayoutB.calls (L : LayoutB) (x : Msg) (s : ℕ) (w : Flock.Wit) : List (Msg × Msg) :=
  (L.shaAt x s).map fun p => (p.1 ++ readAt w p.2.1, readAt w p.2.2)

/-- Every part's recorded calls, off the witnesses `ws`, part by part. -/
def LayoutB.allCalls (L : LayoutB) (x : Msg) (ws : List Flock.Wit) : List (Msg × Msg) :=
  (ws.mapIdx fun s w => L.calls x s w).flatten

/-- **Part `s`'s pieces of round `i`**, off its witness: each piece it holds, with its byte offset in the round. -/
def LayoutB.piecesOf (L : LayoutB) (x : Msg) (s : ℕ) (w : Flock.Wit) (i : ℕ) : List (ℕ × Msg) :=
  ((L.pieceAt x i).foldl (fun (acc : ℕ × List (ℕ × Msg)) p =>
    (acc.1 + pieceLen p.2, if p.1 = s then acc.2 ++ [(acc.1, readSegs w p.2)] else acc.2)) (0, [])).2

section Facts

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **Every inner round is fixed, held or assembled** (a code fact about the layout at the shape, restating `HeldPub` for
option (d)). At a compiled statement of the shape, the layout spans exactly the inner verifier's rounds. Each round,
round 0 and the last included, is one of three kinds:

* fixed by the statement;
* held whole by some part;
* assembled from pieces of the parts.

The bound layout has no public round. -/
def HeldB (D : Deploy Coin) (I : Inner Coin W) (L : LayoutB) (shape : Form → Prop) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form →
    I.V.k (xI D x) = L.rounds x ∧ ∀ i < I.V.k (xI D x),
      (L.fixedAt x i).isSome ∨ (∃ s < D.nS x, (L.msgAt x s i).isSome) ∨
        (L.pieceAt x i ≠ [] ∧ ∀ p ∈ L.pieceAt x i, p.1 < D.nS x)

/-- **V\* part `s`'s circuit checks the inner check `chk s` at every round it reads** (restating `PartChecks` for option
(d)). It is `PartChecks`, except that the messages it quantifies over agree with the part in two ways: with its
holdings at the rounds it holds whole, and with its pieces at the assembled rounds, each at its offset. So a part's
check can read its own pieces of the last message, which no part held whole before. Or two of its recorded calls
collide. -/
def PartChecksB (H : Msg → Msg) (D : Deploy Coin) (DO : Flock.Deploy Coin Col T) (L : LayoutB)
    (shape : Form → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ (xO : Msg) (w : Flock.Wit),
    Flock.check DO.parse xO w = true → (∀ e ∈ L.calls x s w, H e.1 = e.2) →
    (∀ (c : ℕ → Coin), shows s x xO c → ∀ ms : List Msg,
      (∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m) →
      (∀ i < L.rounds x, ∀ o m, (o, m) ∈ L.piecesOf x s w i → ((ms.getD i []).drop o).take m.length = m) →
      chk s x c ms) ∨
      (findColl (L.calls x s w)).isSome

/-- **The parts' inner checks are the inner verifier, at the readers' messages** (restating `Decomp` for option (d)).
At a compiled statement of the shape, take any witnesses, one a part, whose recorded calls are true evaluations, such
that every part's inner check holds at the coins `c` on the readers' messages (`LayoutB.msgs`, every round off the
witnesses under `HeldB`). Then `IV` accepts those messages at those coins, or two of the parts' recorded calls collide
(`LayoutB.allCalls`). The escape is for a commit string several parts hold: the readers take the first holder's
message, and a holder that opens the string otherwise shows a collision among the calls they record. Round 0 and the
last round are the parts', not free. -/
def DecompB (H : Msg → Msg) (D : Deploy Coin) (IV : Verifier Coin) (L : LayoutB) (shape : Form → Prop)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ (c : ℕ → Coin) (ws : List Flock.Wit),
    ws.length = D.nS x → (∀ e ∈ L.allCalls x ws, H e.1 = e.2) → (∀ s < D.nS x, chk s x c (L.msgs x ws)) →
      (run H (replay (L.msgs x ws)) IV (xI D x) () fun i => c i.val).out = .acc ∨
        (findColl (L.allCalls x ws)).isSome

end Facts

end Claims.Recursion.VStarB
