import Proofs.Flock.Claims.Recursion.BoundDeploy

/-!
# (P4) restated modulo what `E*` reads: a candidate statement

`Bound.LastBinds` asks for a collision among `IVStar`'s calls wherever two accepted decisions, after the same rounds and
coins, have last messages that `final` reads differently. At the deployed record it is false
(`RecDeploy.lastBinds_deployed_rejects`): the deployed plain verifier reads the record's text only through `Json.parse`,
so the same last message with a trailing space in the record is decided alike, with the same calls.

`Bound.LastBindsE` asks for the collision only where the two decisions differ in what `E*` reads of them. `E*` reads
an inner view only through its decision and its openings (`served_eval_congr`), and the bound record's openings are
`DE`'s columns of the opened view (`Bound.deploy`'s `cols`). Two accepted decisions share their decision, so what
`E*` reads differs exactly where `DE.cols (openView ·)` differs. `Bound.viewAt` is the view `IVStar`'s run shows
(`Bound.run_ivstar`'s closed form, in list form).

Comparing the opened views themselves (`openView`) wouldn't help: the opened view carries `DE`'s last message as it is,
so the trailing-space pair differs there too. At the deployed record `cols` reads the record only through `liveOf`
(`Json.parse`) and the proof files, so the trailing-space pair is not a counterexample to `LastBindsE`.

A statement for review, with no proof. The coordinator's card on (P4) decides whether it replaces `LastBinds`. -/

namespace Claims.Recursion

open Claims

variable {Coin Col T : Type} [Inhabited Coin]

/-- **The view of `IVStar`'s decision** on the messages `as` at the coins `cs`: each message before the last answered
by its coin (`ans`), the last by the decision. -/
def Bound.viewAt (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) (as : List Msg) (cs : List Coin) : Seen :=
  ⟨as.zip (cs.dropLast.map (fun c => Out.msg (Bd.ans c)) ++ [((Bd.IVStar.next x as cs).eval H).1]),
    ((Bd.IVStar.next x as cs).eval H).1⟩

/-- **(P4), modulo what `E*` reads.** Take two decisions `IVStar` accepts at one statement, on messages and coins that
agree before the last round, whose opened views `DE` opens to different columns (`Bound.deploy`'s `cols`). They show a
SHA-512 collision among `IVStar`'s calls in the two decisions. -/
def Bound.LastBindsE (H : Msg → Msg) (Bd : Bound Coin Col T) : Prop :=
  ∀ (x : Msg) (as as' : List Msg) (cs cs' : List Coin), as.length = Bd.DE.V.k x → as'.length = as.length →
    cs.length = as.length → cs'.length = as.length → as.dropLast = as'.dropLast → cs.dropLast = cs'.dropLast →
    Bd.DE.cols (Bd.openView (Bd.viewAt H x as cs)) ≠ Bd.DE.cols (Bd.openView (Bd.viewAt H x as' cs')) →
    ((Bd.IVStar.next x as cs).eval H).1 = .acc → ((Bd.IVStar.next x as' cs').eval H).1 = .acc →
    ∃ q ∈ (((Bd.IVStar.next x as cs).eval H).2 ++ ((Bd.IVStar.next x as' cs').eval H).2).map Prod.fst,
      ∃ q' ∈ (((Bd.IVStar.next x as cs).eval H).2 ++ ((Bd.IVStar.next x as' cs').eval H).2).map Prod.fst,
        Collision H q q'

end Claims.Recursion
