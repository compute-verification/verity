module

public import Definitions.Claims.Recursion

/-!
# The ZK server, as the recursion's knowledge soundness reads it

The ZK server commits to every inner message before the auditor samples the coin that answers it, and the compiled
verifier opens those commitments inside the outer proofs. Knowledge soundness reads the server only through how a
commitment is opened (`Server`): a round's commit strings, each `width` bytes, open to that round's rows under salts,
and the inner verifier's last message carries the openings (`final`). `Server.IVStar` is the inner verifier the outer
sessions prove: it accepts the inner verifier's messages opened from the commit strings, at the run's coins, with its
answers the coins' (`ans`). `Server.Binds` is what the server must give soundness: two passing openings of one commit
string to different rows show a SHA-512 collision among their calls.

`Claims.Recursion.Bound`'s `IVStar`, `agrees`, `opens*` and `openView` are these at the hm96 server (`Server.ofBound`,
`Proofs.Flock.RecGeneric.AtFlock`), with the element-level Flock's verifier for `V`.

What zero knowledge reads of the server (hiding leaves, live coins, what it releases) is `Proofs.Flock.RecGeneric.ZK`.

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

/-- **A ZK server's commitments, as the compiled verifier opens them**: its salts; the bytes of one commit string
(`width`); the count of commit strings at each inner round before the last (`nAt x j`); how a round's rows join into
the inner verifier's message (`join`); how the inner verifier's last message reads (`final`: per earlier round its rows
with their salts, in commit order, then the inner verifier's own last message); how the server commits to a row under a
salt (`commit`); and whether a commit string opens to a row under a salt (`opens`), each through the oracle. -/
structure Server where
  Salt : Type
  width : ℕ
  nAt : Msg → ℕ → ℕ
  join : List Msg → Msg
  final : Msg → Option (List (List (Msg × Salt)) × Msg)
  commit : Msg → Salt → OC Msg
  opens : Msg → Msg → Salt → OC Bool

namespace Server

/-- **A round message opens to rows under salts**: it is as many commit strings as there are rows, in order, each
opening to its row under its salt. -/
def opensRound (S : Server) (a : Msg) (os : List (Msg × S.Salt)) : OC Bool :=
  if a.length = S.width * os.length then
    allOC (os.mapIdx fun i o => S.opens ((a.drop (S.width * i)).take S.width) o.1 o.2)
  else .ret false

/-- The inner verifier's messages a last message opens: each earlier round's rows joined, then its own last message. -/
def opened (S : Server) (os : List (List (Msg × S.Salt))) (last : Msg) : List Msg :=
  os.map (fun o => S.join (o.map Prod.fst)) ++ [last]

/-- **A view, opened**: its prover messages are the inner verifier's messages its last message opens, its answers and
decision kept; unchanged where its last message doesn't read. -/
def openView (S : Server) (v : Seen) : Seen :=
  match S.final (v.msgs.getLastD ([], .rej)).1 with
  | none => v
  | some (os, last) => ⟨(S.opened os last).zip (v.msgs.map Prod.snd), v.out⟩

variable {Coin : Type} [Inhabited Coin]

/-- **The inner verifier accepts the messages `ms` at the coins `cs`**: it answers each round before the last with that
round's coin (`ans`) and accepts at the last. -/
def agrees (V : Verifier Coin) (ans : Coin → Msg) (x : Msg) (ms : List Msg) (cs : List Coin) : OC Bool :=
  allOC ((List.range ms.length).map fun j =>
    bindOC (V.next x (ms.take (j + 1)) (cs.take (j + 1))) fun b =>
      .ret (decide (b = if j + 1 < ms.length then .msg (ans (cs.getD j default)) else .acc)))

/-- **`Open(V, S)`, the inner verifier over the server's commitments**: `V`'s rounds. At each round before its last it
answers the round's coin and decides nothing. At its last round it reads the last message (`final`) and accepts exactly
when each earlier round's message opens to that round's rows under their salts, as many as the schedule fixes (`nAt`),
and `V` accepts the opened messages at the run's coins, answering each earlier round with its coin (`agrees`). -/
def IVStar (S : Server) (V : Verifier Coin) (ans : Coin → Msg) : Verifier Coin where
  k := V.k
  next x as cs :=
    if as.length < V.k x then .ret (.msg (ans (cs.getLastD default)))
    else match S.final (as.getLastD []) with
      | none => .ret .rej
      | some (os, last) =>
        if os.length + 1 = as.length ∧ ∀ j < os.length, (os.getD j []).length = S.nAt x j then
          bindOC (allOC ((as.dropLast.zip os).map fun p => S.opensRound p.1 p.2)) fun ok =>
            if ok then bindOC (agrees V ans x (S.opened os last) cs) fun acc => .ret (if acc then .acc else .rej)
            else .ret .rej
        else .ret .rej

end Server

/-- The SHA-512 inputs an oracle program queries, at `H`. -/
def queries {α : Type} (H : Msg → Msg) (p : OC α) : List Msg := (p.eval H).2.map Prod.fst

/-- **The server's commitments bind** (soundness's one requirement on the server), at every oracle: two openings of
one commit string to different rows that both pass show a SHA-512 collision among the two openings' calls. At hm96
it holds wherever a row's digest input is injective (`Claims.Recursion.Bound.Binds`): different salts collide on their
salts' hashes, one salt on the rows' digests. -/
def Server.Binds (S : Server) : Prop :=
  ∀ (H : Msg → Msg) (a row row' : Msg) (y y' : S.Salt), row ≠ row' →
    ((S.opens a row y).eval H).1 = true → ((S.opens a row' y').eval H).1 = true →
    ∃ q ∈ queries H (S.opens a row y) ++ queries H (S.opens a row' y'),
      ∃ q' ∈ queries H (S.opens a row y) ++ queries H (S.opens a row' y'), Collision H q q'

end Claims.RecGeneric
