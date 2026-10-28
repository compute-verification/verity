/-
# The coin tree's openings, checked in Lean (`coin-tree/hm96-sha512/v2`, docs/coin-tree-v2.md, live PROTOCOL.md §9)

The prover's side of the keyed coin tree. The verifier commits every coin of the session at `Hello` and opens them round
by round; before the prover uses a coin, the opening it came with must walk the committed tree to the root `Hello`'s
answer named (`check`). Every SHA-512 input is prefixed by the prover's nonce `ν` (`hk`): the chain
`T_i = H_ν(CHAIN ‖ c_i ‖ T_{i+1})` from the opening's `T_n` up through its `n` coins (`chain`), the round digest
`x = H_ν(ROUND ‖ u32be(stream) ‖ u32be(round) ‖ T_0)`, the keyed hm96 leaf `H_ν(pad(LEAF_TAG ‖ SHA-512(K)) ‖ x ⊕ M(K)·y ‖
H_ν(pad(SALT_TAG) ‖ y))` under the answer's key `K` and the opening's salt `y` (`leaf`), then the path, the round's
subtree and then the streams' tree, each node `H_ν(left ‖ right)` (`walk`). Rust's `coin_tree::check` is the speed
reference: its refusals are these, word for word (`live/src/coin_open_vectors.json` holds both to the same cases).

A message is as the wire frames it, the form the zero-knowledge analysis reads (`Claims.ZKMal.CoinLayer`): `Resp::Coins`
of `W` words is `2 ‖ u32le(W) ‖ 16·W bytes` (`frame`).

`flock-firewall coins` serves one session's checks (`cmd`), the verdict the prover's checked transport takes for every
coin it hands on: the session's `Hello` (the spec, the link points' count and the nonce), `Hello`'s answer (the root and
the key), then each opening in the order the prover asks for them. The first refusal ends the session: nothing after it
is checked, so no coin after a refused opening is handed on.
-/
import Flock.CoinTree
import Flock.Hm96

namespace Flock.Firewall.Coins

open Lean (Json)

/-- The prefix of a refusal: the prover stops there, before it uses any coin of the round (`coin_tree::Checked`). -/
def STOP : String := "PROVER-STOPPED: "

/-- The stream of the session's own round: the link points, drawn at `Commit` (`coin_tree::SESSION`). -/
def SESSION : String := "session"

/-- `SHA-512(ν ‖ m)`. -/
def hk (ν m : List UInt8) : List UInt8 := (Sha512.hash ⟨(ν ++ m).toArray⟩).data.toList

/-- v2's domain tag `verity/coin-tree/{name}/v2\0`. -/
def tag (name : String) : List UInt8 := (s!"verity/coin-tree/{name}/v2").toUTF8.data.toList ++ [0]

/-- The first `n` bytes of `b`, zeros past its end. -/
def fix (n : Nat) (b : List UInt8) : List UInt8 := (b ++ List.replicate n 0).take n

/-- `u32be(v)`. -/
def u32be (v : Nat) : List UInt8 :=
  [(v / 2 ^ 24 % 256).toUInt8, (v / 2 ^ 16 % 256).toUInt8, (v / 2 ^ 8 % 256).toUInt8, (v % 256).toUInt8]

/-- `Resp::Coins`'s frame of `ws` words: tag `2`, `u32le(ws)`, the words' bytes. -/
def frame (ws : Nat) (b : List UInt8) : List UInt8 := 2 :: ((u32be ws).reverse ++ b)

/-- The path's length: the round's subtree, then the streams' tree (`Spec::opening_words`). -/
def depth (s : CoinTree.Spec) : Nat := s.roundsLog + (Nat.nextPowerOfTwo s.streams.size).log2

/-- An opening of `n` coins in words: the coins, `T_n` and the salt (16), the path (4 a node). -/
def openingWords (s : CoinTree.Spec) (n : Nat) : Nat := n + 16 + 4 * depth s

/-- `T_0`: the chain from `T_n` up through the coins, `T_i = H_ν(CHAIN ‖ c_i ‖ T_{i+1})`. -/
def chain (ν : List UInt8) : List (List UInt8) → List UInt8 → List UInt8
  | [], tn => tn
  | c :: r, tn => hk ν (tag "chain" ++ c ++ chain ν r tn)

/-- The node the path from `a` reaches: `H_ν(acc ‖ sib)` at an even index, `H_ν(sib ‖ acc)` at an odd one. -/
def walk (ν : List UInt8) : Nat → List UInt8 → List (List UInt8) → List UInt8
  | _, a, [] => a
  | idx, a, s :: r => walk ν (idx / 2) (hk ν (if idx % 2 = 0 then a ++ s else s ++ a)) r

/-- The keyed hm96 mix `x ⊕ M(K)·y` (`Hm96.sha512.mask`). -/
def mix (key : ByteArray) (x y : List UInt8) : List UInt8 :=
  fix 64 (List.zipWith (· ^^^ ·) x (fix 64 (Hm96.sha512.mask key ⟨y.toArray⟩).data.toList))

/-- The hm96 leaf of the position `pos` whose chain reaches `t0`, under the key `key` and the salt `y`. -/
def leaf (ν : List UInt8) (key : ByteArray) (pos t0 y : List UInt8) : List UInt8 :=
  hk ν ((Hm96.sha512.leafPrefix key).data.toList ++ mix key (hk ν (tag "round" ++ pos ++ t0)) y ++
    hk ν ((Hm96.pad Hm96.sha512.saltTag Hm96.sha512.block).data.toList ++ y))

/-- An opening's parts in the bytes `b`, at `n` coins and `d` path nodes: the coins, `T_n`, the salt, the path. -/
structure Parts where
  cs : List (List UInt8)
  tn : List UInt8
  y : List UInt8
  sibs : List (List UInt8)

def parts (n d : Nat) (b : List UInt8) : Parts where
  cs := (List.range n).map fun i => fix 16 (b.drop (16 * i))
  tn := fix 64 (b.drop (16 * n))
  y := fix 192 (b.drop (16 * n + 64))
  sibs := (List.range d).map fun i => fix 64 (b.drop (16 * n + 256 + 64 * i))

/-- The parts' bytes, in order. -/
def Parts.join (p : Parts) : List UInt8 := p.cs.flatten ++ p.tn ++ p.y ++ p.sibs.flatten

/-- Whether `m` is an opening of `n` coins of stream `s`'s round `r`: the round in the schedule, `n` in the block, the
stream in the tree, and `m` exactly `Resp::Coins` of the opening's words. -/
def parses (spec : CoinTree.Spec) (s r n : Nat) (m : List UInt8) : Bool :=
  decide (r < 2 ^ spec.roundsLog ∧ n ≤ spec.block ∧ s < spec.streams.size) &&
    decide (m = frame (openingWords spec n) (parts n (depth spec) (m.drop 5)).join)

/-- The node the opening `m` of `n` coins of stream `s`'s round `r` walks to: its leaf, then its path. -/
def reach (spec : CoinTree.Spec) (ν : List UInt8) (key : ByteArray) (s r n : Nat) (m : List UInt8) : List UInt8 :=
  let p := parts n (depth spec) (m.drop 5)
  walk ν (s * 2 ^ spec.roundsLog + r) (leaf ν key (u32be s ++ u32be r) (chain ν p.cs p.tn) p.y) p.sibs

/-- **The opening check** (`coin_tree::check`): `m` opens `n` coins of stream `s`'s round `r` under the nonce `ν` and the
answer's `key` and `root`; the coins' bytes. Refused, with Rust's reason, when it does not parse (`parses`) or its walk
does not reach the root (`reach`). -/
def check (spec : CoinTree.Spec) (ν : List UInt8) (key : ByteArray) (root : List UInt8) (s r n : Nat) (m : List UInt8) :
    Except String (List UInt8) :=
  let name := spec.streams[s]?.getD "?"
  if !parses spec s r n m then
    .error s!"coin opening of {name} round {r}: {(m.length - 5) / 16} words, want {openingWords spec n}"
  else if reach spec ν key s r n m = root then .ok (parts n (depth spec) (m.drop 5)).cs.flatten
  else .error s!"the verifier's coins for {name} round {r} do not open its step-0 commitment"

/-- **`Hello`'s answer** `a` (`Resp::Coins`' frame): the coin tree's root and the session's hm96 key, 20 words, the key's
bit 2047 clear (`coin_tree::Checked`); the key and the root. -/
def answer (a : List UInt8) : Except String (ByteArray × List UInt8) :=
  if !(a.length == 325 && a.take 5 == frame 20 []) then
    .error s!"{STOP}Hello's answer has {(a.length - 5) / 16} words, not the coin tree's root and session key (20)"
  else if a.getD 324 0 &&& 0x80 != 0 then .error s!"{STOP}the verifier's session key has bit 2047 set"
  else .ok (⟨((a.drop 69).take 256).toArray⟩, fix 64 (a.drop 5))

/-- What the session's `Hello` fixes of its coins: the tree's spec, the session stream's coin count (the link points,
`link.points · link.m`) and the prover's nonce. -/
structure Hello where
  spec : CoinTree.Spec
  links : Nat
  nonce : List UInt8

/-- **The session's `Hello`**, as the prover sent it: its nonce (`coins.nonce`, 64 lowercase hex digits, `coin_tree::hello_nonce`),
the tree it names (`CoinTree.ofHello`) and its link points. -/
def Hello.parse (h : String) : Except String Hello := do
  let j ← (Json.parse h).mapError (s!"{STOP}Hello is not JSON: {·}")
  let some n := (j.getObjVal? "coins").toOption.bind (·.getObjValAs? String "nonce" |>.toOption)
    | throw s!"{STOP}Hello carries no coin-tree nonce (coins.nonce)"
  if !CoinTree.isHex 64 n then throw s!"{STOP}Hello's coin-tree nonce is not 64 lowercase hex digits"
  let some spec := CoinTree.ofHello h | throw s!"{STOP}Hello's coins are not {CoinTree.SCHEME}"
  let link := fun (k : String) => ((j.getObjVal? "link").toOption.bind (·.getObjValAs? Nat k |>.toOption)).getD 0
  let nonce ← (unhex n).mapError (s!"{STOP}Hello's coin-tree nonce: {·}")
  return { spec, links := link "points" * link "m", nonce := nonce.data.toList }

/-- A session past `Hello`'s answer: its `Hello`, the answer's key and root, and the streams of the rounds opened so far,
in order (a stream's next round is how often it appears). -/
structure Release where
  hello : Hello
  key : ByteArray
  root : List UInt8
  opened : List Nat

/-- The session once `Hello`'s answer `a` checks. -/
def Release.start (h : Hello) (a : List UInt8) : Except String Release :=
  match answer a with
  | .ok (key, root) => .ok { hello := h, key, root, opened := [] }
  | .error e => .error e

/-- **The next opening** `m`, of `n` coins of `stream`: the stream's next round must open (`check`). The session stream's
count is `Hello`'s link points, whatever the request says. The coins, and the session with the round counted. -/
def Release.open (S : Release) (stream : String) (n : Nat) (m : List UInt8) : Except String (Release × List UInt8) :=
  let s := S.hello.spec.streams.toList.idxOf stream
  if s < S.hello.spec.streams.size then
    match check S.hello.spec S.hello.nonce S.key S.root s (S.opened.count s)
        (if stream == SESSION then S.hello.links else n) m with
    | .ok c => .ok ({ S with opened := S.opened ++ [s] }, c)
    | .error e => .error s!"{STOP}{e}"
  else .error s!"{STOP}{stream} is outside the committed schedule"

/-- **A session's openings** `os`, each `(stream, n, m)` as `open` takes it, in order: the session after them all, or the
first refusal. `cmd` answers them one at a time, so it hands on a round's coins only when this passes up to it. -/
def Release.openAll (S : Release) : List (String × Nat × List UInt8) → Except String Release
  | [] => .ok S
  | (stream, n, m) :: os =>
    match S.open stream n m with
    | .ok (S', _) => S'.openAll os
    | .error e => .error e

/-- `n` bytes of `h`, for `what`'s messages. Each read that returns nothing ends the input, so `n` reads that each return
a byte or more suffice. -/
def readExact (what : String) (h : IO.FS.Stream) (n : Nat) : IO ByteArray := go n .empty
where
  go : Nat → ByteArray → IO ByteArray
    | 0, acc =>
      if acc.size ≥ n then pure acc
      else throw (IO.userError s!"{what}: {acc.size} bytes after {n} reads of a {n}-byte message")
    | fuel + 1, acc => do
      if acc.size ≥ n then return acc
      let b ← h.read (n - acc.size).toUSize
      if b.isEmpty then throw (IO.userError s!"{what}: input closed {acc.size} bytes into a {n}-byte message")
      go fuel (acc ++ b)

/-- `flock-firewall coins`: one session's opening checks, served on stdin and stdout. `hello LEN` followed by the `LEN`
bytes of the prover's `Hello` (`Hello.parse`), then `answer LEN` followed by the `LEN` bytes of `Hello`'s answer, the
`Resp::Coins` frame (`Release.start`), each answered `ok`. Then `open STREAM N LEN` followed by the `LEN` bytes of the
answer to the prover's request for `N` coins of `STREAM` (`Resp::Coins`' frame) is answered `ok N'` followed by the
`16 N'` bytes of the coins it opens (`Release.open`). A refusal is answered `stop WHY` and ends the session: nothing after
it is checked. Anything else (a request out of order, a malformed line) is answered `err WHY` and ends it too. -/
def cmd (_ : List String) : IO UInt32 := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  let say := fun (line : String) => do
    stdout.putStrLn (line.replace "\n" " ")
    stdout.flush
  let refuse := fun (why : String) => do
    say s!"err {why}"
    return (1 : UInt32)
  let stop := fun (why : String) => do
    say s!"stop {why}"
    return (0 : UInt32)
  let mut hello : Option Hello := none
  let mut session : Option Release := none
  repeat
    let line := (← stdin.getLine).replace "\n" ""
    if line.isEmpty then break
    match line.splitOn " " with
    | ["hello", len] =>
      let some len := len.toNat? | return (← refuse s!"a request `{line}`")
      let b ← readExact "coins" stdin len
      if hello.isSome then return (← refuse "a second Hello")
      let some h := String.fromUTF8? b | return (← refuse "a Hello that is not UTF-8")
      match Hello.parse h with
      | .ok h =>
        hello := some h
        say "ok"
      | .error e => return (← stop e)
    | ["answer", len] =>
      let some len := len.toNat? | return (← refuse s!"a request `{line}`")
      let b ← readExact "coins" stdin len
      let some h := hello | return (← refuse "Hello's answer before the Hello")
      if session.isSome then return (← refuse "a second answer to Hello")
      match Release.start h b.data.toList with
      | .ok S =>
        session := some S
        say "ok"
      | .error e => return (← stop e)
    | ["open", stream, n, len] =>
      let (some n, some len) := (n.toNat?, len.toNat?) | return (← refuse s!"a request `{line}`")
      let b ← readExact "coins" stdin len
      let some S := session | return (← stop s!"{STOP}coins before the step-0 coin commitment")
      match S.open stream n b.data.toList with
      | .ok (S', c) =>
        session := some S'
        say s!"ok {c.length / 16}"
        stdout.write ⟨c.toArray⟩
        stdout.flush
      | .error e => return (← stop e)
    | _ => return (← refuse s!"a request `{line}`")
  return 0

end Flock.Firewall.Coins
