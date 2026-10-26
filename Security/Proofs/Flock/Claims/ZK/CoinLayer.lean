import Proofs.Flock.Claims.ZK.Reference
import Flock.Hm96
import Flock.Canon
import Flock.Firewall.SessionPlan

/-!
# The deployed coin layer of `flock-circuit --zk`

`RefSession.zkD` is the masked record at a reference session whose coin layer (`Hello`, the opening check, the coin it
hands on) is the session's data. `zkDeployed` is the record at the deployed coin layer, `coin-tree/hm96-sha512/v2` as the
masked prover's transport runs it (`verity/ml/flock/live/src/coin_tree.rs`, `Checked`) and the masked verifier states it
(`Flock.CoinTree`, `Flock.Zk.coinSpec`):

* **Nonce.** 32 bytes (`Nonce`), the prefix of every hash of the tree (`nb`).
* **`Hello`** (`helloD`): `Req::Hello`'s frame (tag `1`, `u32le` length, the bytes) of the verifier's own `Hello` with the
  nonce's 64 lowercase hex digits in `coins.nonce`, canonical (`helloJson`, what `CoinTree.checkHello` compares).
* **`Hello`'s answer** (`answerOK`, `rootD`, `keyD`): `Resp::Coins` of 20 words, the root's 64 bytes then the session's
  256-byte hm96 key, its bit 2047 clear.
* **The tree** (`treeD`): v2's tags, the keyed hm96 leaf `H_ν(pad(LEAF_TAG ‖ SHA-512(K)) ‖ x ⊕ M(K)·y ‖ c)` under the
  answer's key (`Hm96.sha512`'s `mask` and `leafPrefix`), nodes `H_ν(l ‖ r)`.
* **The schedule** (`slot`, `place`): coin `0` is the link's, the `session` stream's round `0`, with the statement's
  `points · m_pts` coins; coin `j + 1` is round `j` of the rep streams' rounds in `spec.repStreams` order, each stream's
  in order (`FlockDeploy.V`'s rounds), its stream's index in the tree's leaf order, `nr x (j + 1)` coins. The leaf index
  is `stream · 2^roundsLog + round`, the path `roundsLog + log₂(streams)` nodes, and a round past the schedule, a count
  past the block or a stream outside it fails. A tree whose streams are not the session's (`session`, then the rep
  streams, distinct, two a table: `Flock.Firewall.Session.streamsOk`), which the transport refuses, places no coin.
* **An opening** (`parseD`): `Resp::Coins` of `n + 16 + 4·depth` words, the coins, `T_n`, the salt and the path, read at
  the round's place; it passes where the walk reaches the answer's root (`treeChk`).
* **The coin handed on** (`coinOfD`): the opening's words, which the reference prover reads as coins (`coinOf`, the first
  `n`); the coin `refP` reads (`rd`) is the same reading.

`Flock.zk_deployZk_ref` is `Flock.zk_zkD_ref` at `zkDeployed`. Its rounds' coin counts `nr` are the prover's own round
requests, not the statement's, so the record takes them as a parameter, and the theorem holds at every one.
`Claims.Witness.Flock.ZK_meets` states the record's coin layer; `ZK_coins_fails` that the degenerate coin layer (an
opening check that passes everything, a coin that is the opening) is not it.
-/

namespace Claims.ZKMal

open Claims FlockSoundness.Merkle FlockSoundness.ZK
open FlockSoundness.Discharge
open Lean (Json)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace CoinLayer

/-! ## Bytes -/

/-- The first `n` bytes of `b`, zeros past its end. -/
def fix (n : ℕ) (b : List UInt8) : {b : List UInt8 // b.length = n} :=
  ⟨(b ++ List.replicate n 0).take n, by rw [List.length_take, List.length_append, List.length_replicate]; omega⟩

theorem fix_of_len {n : ℕ} {b : List UInt8} (h : b.length = n) : (fix n b).val = b := by
  simp [fix, ← h]

/-- `u32be(v)`. -/
def u32be (v : ℕ) : List UInt8 :=
  [(v / 2 ^ 24 % 256).toUInt8, (v / 2 ^ 16 % 256).toUInt8, (v / 2 ^ 8 % 256).toUInt8, (v % 256).toUInt8]

/-- `u32le(v)`. -/
def u32le (v : ℕ) : List UInt8 := (u32be v).reverse

/-- A tree word: 16 bytes, `le64(lo) ‖ le64(hi)` of one GF(2^128) element. -/
abbrev Word : Type := {b : List UInt8 // b.length = 16}

/-- An hm96 salt: 192 bytes. -/
abbrev Salt : Type := {b : List UInt8 // b.length = 192}

/-- `Resp::Coins`'s frame of `ws` words: tag `2`, `u32le(ws)`, the words' bytes. -/
def frame (ws : ℕ) (b : List UInt8) : Msg := 2 :: (u32le ws ++ b)

/-! ## The tree -/

/-- v2's domain tag `verity/coin-tree/{name}/v2\0`. -/
def tagD (name : String) : List UInt8 := (s!"verity/coin-tree/{name}/v2").toUTF8.data.toList ++ [0]

/-- The prover's nonce: 32 bytes. -/
abbrev Nonce : Type := Fin 32 → Fin 256

/-- The nonce's bytes, the prefix of every hash of the tree. -/
def nb (ν : Nonce) : List UInt8 := List.ofFn fun i => (ν i).val.toUInt8

/-- `Hello`'s answer is the coin tree's: `Resp::Coins` of 20 words, the root's 4 then the key's 16, its bit 2047 clear. -/
def answerOK (ρ : Msg) : Bool :=
  ρ.length == 325 && ρ.take 5 == frame 20 [] && ρ.getD 324 0 &&& 0x80 == 0

/-- The root `Hello`'s answer commits: its words 0 to 3. -/
def rootD (ρ : Msg) : Dg := fix 64 (ρ.drop 5)

/-- The session's hm96 key `Hello`'s answer reveals: its words 4 to 19. -/
def keyD (ρ : Msg) : ByteArray := ⟨((ρ.drop 69).take 256).toArray⟩

/-- `M(K)·y` (`Hm96.sha512.mask`): bit `i` is `⊕_j K_{i+j} y_j`. -/
def maskD (key : ByteArray) (y : Salt) : List UInt8 := (fix 64 (Flock.Hm96.sha512.mask key ⟨y.val.toArray⟩).data.toList).val

theorem maskD_length (key : ByteArray) (y : Salt) : (maskD key y).length = 64 := (fix 64 _).property

theorem zipWith_xor_inj : ∀ (a a' m : List UInt8), a.length = a'.length → a.length ≤ m.length →
    List.zipWith (· ^^^ ·) a m = List.zipWith (· ^^^ ·) a' m → a = a'
  | [], [], _, _, _, _ => rfl
  | [], _ :: _, _, h, _, _ => by simp at h
  | _ :: _, [], _, h, _, _ => by simp at h
  | _ :: _, _ :: _, [], _, h, _ => by simp at h
  | b :: r, b' :: r', c :: m, h1, h2, h3 => by
      simp only [List.zipWith_cons_cons, List.cons.injEq, UInt8.xor_left_inj] at h3
      obtain ⟨rfl, e⟩ := h3
      rw [zipWith_xor_inj r r' m (by simpa using h1) (by simpa using h2) e]

/-- The keyed hm96 mix `x ⊕ M(K)·y`. -/
def mixD (key : ByteArray) (x : Dg) (y : Salt) : Dg := fix 64 (List.zipWith (· ^^^ ·) x.val (maskD key y))

theorem mixD_inj (key : ByteArray) (y : Salt) : Function.Injective fun x => mixD key x y := by
  intro x x' h
  have hl : ∀ z : Dg, (List.zipWith (· ^^^ ·) z.val (maskD key y)).length = 64 := fun z => by
    rw [List.length_zipWith, z.property, maskD_length]; rfl
  have h' := congrArg Subtype.val h
  simp only [mixD, fix_of_len (hl x), fix_of_len (hl x')] at h'
  exact Subtype.ext (zipWith_xor_inj _ _ _ (x.property.trans x'.property.symm)
    (by rw [x.property, maskD_length]) h')

/-- The tree's digests, coins and salts: their bytes. -/
def encD : FlockSoundness.ZK.CoinEnc Dg Word Salt where
  dig := Subtype.val
  coin := Subtype.val
  salt := Subtype.val
  dig_inj := Subtype.val_injective
  coin_inj := Subtype.val_injective
  salt_inj := Subtype.val_injective
  dig_len a b := a.property.trans b.property.symm
  coin_len a b := a.property.trans b.property.symm

/-- The tree's nodes: a digest's bytes. -/
def nodeD : Enc Unit Dg where
  col _ := []
  dig := Subtype.val
  col_inj _ _ _ := rfl
  dig_inj := Subtype.val_injective
  dig_len a b := a.property.trans b.property.symm

/-- **The coin tree `Hello`'s answer `ρ` keys** (`coin-tree/hm96-sha512/v2`): v2's chain and round tags, hm96-sha512's salt
prefix, and the leaf prefix and mix under the answer's key. -/
def treeD (ρ : Msg) : GK.CoinTree Dg Word Salt where
  E := encD
  ME := nodeD
  mix := mixD (keyD ρ)
  mix_inj := mixD_inj (keyD ρ)
  chainTag := tagD "chain"
  roundTag := tagD "round"
  lp := (Flock.Hm96.sha512.leafPrefix (keyD ρ)).data.toList
  sp := (Flock.Hm96.pad Flock.Hm96.sha512.saltTag Flock.Hm96.sha512.block).data.toList

/-! ## The schedule -/

/-- The rep stream and round of rep round `r`, where stream `k`'s rounds are `cs`'s `k`th count. -/
def locate : List ℕ → ℕ → ℕ → ℕ × ℕ
  | [], k, r => (k, r)
  | c :: cs, k, r => if r < c then (k, r) else locate cs (k + 1) (r - c)

/-- **Coin `j`'s place in the committed schedule** at statement `x`: the session's spec, the coin tree's, the stream's
index in the tree's leaf order and the round. Coin `0` is the `session` stream's round `0`; coin `j + 1` is rep round
`j`, the rep streams' rounds in `spec.repStreams` order (`FlockDeploy.V`'s rounds). None where the tree's streams are not
the session's (`Flock.Firewall.Session.streamsOk`). -/
def slot (x : Msg) (j : ℕ) : Option (Flock.SessionSpec × Flock.CoinTree.Spec × ℕ × ℕ) :=
  match FlockDeploy.setupOf true x with
  | none => none
  | some (_, spec, sts) =>
    match spec.coinTree with
    | none => none
    | some ct =>
      if _root_.Flock.Firewall.Session.streamsOk spec sts ct then
        if j = 0 then some (spec, ct, 0, 0)
        else
          let rs := (List.range spec.repStreams.size).map fun k => FlockDeploy.repRounds true spec sts[k / 2]!
          some (spec, ct, ct.streams.toList.idxOf (_root_.Flock.Firewall.Session.repName spec (locate rs 0 (j - 1)).1),
            (locate rs 0 (j - 1)).2)
      else none

/-- What the check reads of a round's place: whether the place is in the schedule, its position bytes
`u32be(stream) ‖ u32be(round)`, its leaf index, its coin count and its path's length. -/
structure Place where
  ok : Bool
  pos : List UInt8
  idx : ℕ
  n : ℕ
  depth : ℕ

/-- **Coin `j`'s place** at statement `x`, rep rounds of `nr x j` coins: the link's `points · m_pts`. -/
def place (nr : Msg → ℕ → ℕ) (x : Msg) (j : ℕ) : Place :=
  match slot x j with
  | some (spec, ct, s, r) =>
    let n := if j = 0 then spec.points * spec.mPts else nr x j
    { ok := decide (r < 2 ^ ct.roundsLog ∧ n ≤ ct.block ∧ s < ct.streams.size)
      pos := u32be s ++ u32be r
      idx := s * 2 ^ ct.roundsLog + r
      n
      depth := ct.roundsLog + (Nat.nextPowerOfTwo ct.streams.size).log2 }
  | none => { ok := false, pos := [], idx := 0, n := 0, depth := 0 }

/-! ## Openings -/

/-- The opening `b`'s words at place `p`: `n` coins, `T_n`, the salt, then `depth` path nodes. -/
def openOf (p : Place) (b : List UInt8) : GK.CoinOpening Dg Word Salt where
  pos := p.pos
  idx := p.idx
  cs := (List.range p.n).map fun i => fix 16 (b.drop (16 * i))
  Tn := fix 64 (b.drop (16 * p.n))
  y := fix 192 (b.drop (16 * p.n + 64))
  sibs := (List.range p.depth).map fun i => fix 64 (b.drop (16 * p.n + 256 + 64 * i))

/-- An opening's words, in order. -/
def join (op : GK.CoinOpening Dg Word Salt) : List UInt8 :=
  op.cs.flatMap Subtype.val ++ op.Tn.val ++ op.y.val ++ op.sibs.flatMap Subtype.val

/-- **The message `m` read as the opening of place `p`** under `Hello`'s answer `ρ`: the place is in the schedule, the
answer is the tree's, and `m` is `Resp::Coins` of exactly the opening's `n + 16 + 4·depth` words. -/
def parseD (p : Place) (ρ m : Msg) : Option (GK.CoinOpening Dg Word Salt) :=
  if p.ok && answerOK ρ && decide (m = frame (p.n + 16 + 4 * p.depth) (join (openOf p (m.drop 5)))) then
    some (openOf p (m.drop 5))
  else none

theorem parseD_some {p : Place} {ρ m : Msg} {op : GK.CoinOpening Dg Word Salt} (h : parseD p ρ m = some op) :
    m = frame (p.n + 16 + 4 * p.depth) (join op) := by
  unfold parseD at h
  split_ifs at h with hc
  obtain rfl := Option.some.inj h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
  exact hc.2

theorem parseD_nil (p : Place) (ρ : Msg) : parseD p ρ [] = none := by
  simp [parseD, frame]

/-- **The coin the prover hands on**: the opening's words, as the reference prover reads a coin (`coinOf`). -/
def coinOfD (m : Msg) : Msg := m.drop 5

/-- The coin an opening reads as: element `i` the opening's word `i`. -/
def rd (m : Msg) : FlockDeploy.Coin := fun i => FlockComplete.coinOf (coinOfD m) i.val

/-! ## `Hello` -/

/-- The verifier's own `Hello` with `nonce` in `coins.nonce`, canonical (what `CoinTree.checkHello` compares). -/
def helloJson (spec : Flock.SessionSpec) (nonce : String) : Flock.V String :=
  match spec.coinTree with
  | none => pure spec.hello
  | some _ => do
    let some own := (Json.parse spec.hello).toOption | throw "S2/R7: the verifier's Hello is not JSON"
    let coins := (own.getObjVal? "coins").toOption.getD Json.null
    Flock.canon (own.setObjVal! "coins" (coins.setObjVal! "nonce" (Json.str nonce)))

/-- **The prover's `Hello`** at statement `x` and nonce `ν`: `Req::Hello`'s frame (tag `1`, `u32le` length, the bytes)
of the statement's `Hello` with the nonce's hex. -/
def helloD (x : Msg) (ν : Nonce) : Msg :=
  match FlockDeploy.setupOf true x with
  | some (_, spec, _) =>
    match helloJson spec (Flock.hexOf ⟨(nb ν).toArray⟩) with
    | .ok h => 1 :: (u32le h.utf8ByteSize ++ h.toUTF8.data.toList)
    | .error _ => []
  | none => []

/-! ## The session at the deployed coin layer -/

/-- **What the reference session reads besides the coin layer**, at the deployed coins: the prover's tape and its
reading as the masked prover's (`tapeZk`), each statement's tables, their coins, the witness's and the dummy's table
witnesses, the round encoding, the last message where the prover goes on and where it refuses, and the tables'
randomness on the tape. -/
structure TableData where
  Tape : Type
  [finTape : Fintype Tape]
  [neTape : Nonempty Tape]
  tapeZk : Tape → FlockComplete.TapeZk
  J : Msg → ℕ
  C : (x : Msg) → Fin (J x) → Composed.CoinedSetup
  cz : (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (C x j).Rest
  wt : (x : Msg) → Flock.Wit → ∀ j, FlockSoundness.Model.Witness (C x j).S.m
  wt₀ : (x : Msg) → ∀ j, FlockSoundness.Model.Witness (C x j).S.m
  enc : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → ℕ → (∀ j, ((C x j).T (cz x c j)).tab.Cut) → Msg
  fin : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → (∀ j, ((C x j).T (cz x c j)).tab.View) → Msg
  off : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → (∀ j, ((C x j).T (cz x c j)).tab.Cut) → Msg
  tape : (x : Msg) → Tape → (∀ j, ZkView.Base (C x j).z (C x j).Hi) ×
    (ZkView.Leaves (fun j => (C x j).z) → ZkView.Salt) × (ZkView.Leaves (fun j => (C x j).z) → ZkView.Bits × ZkView.Salt)

attribute [instance] TableData.finTape TableData.neTape

instance instDecidableEqCoin : DecidableEq FlockDeploy.Coin := inferInstanceAs (DecidableEq (Fin 65536 → Flock.F128))

/-- **The reference session at the deployed coin layer**, rep rounds of `nr` coins, at the tables `T`. -/
def session (nr : Msg → ℕ → ℕ) (T : TableData) : RefSession where
  Coin := FlockDeploy.Coin
  Nonce := Nonce
  Tape := T.Tape
  CoinT := Word
  CSalt := Salt
  Tr _ ρ := treeD ρ
  nb := nb
  root _ ρ := rootD ρ
  parse x ρ j m := parseD (place nr x j) ρ m
  pos x _ j := (place nr x j).pos
  idx x _ j := (place nr x j).idx
  len x _ j := (place nr x j).n
  depth x _ j := (place nr x j).depth
  rd := rd
  rd_eq _ _ _ m m' _ h h' := by rw [(parseD_some h).trans (parseD_some h').symm]
  hello := helloD
  abort := []
  J := T.J
  C := T.C
  cz := T.cz
  wt := T.wt
  wt₀ := T.wt₀
  enc := T.enc
  fin := T.fin
  off := T.off
  tape := T.tape

end CoinLayer

open CoinLayer

/-- **The deployed masked record** (`coin-tree/hm96-sha512/v2`): `RefSession.zkD` at the deployed coin layer, rep rounds of
`nr` coins and the tables `T`: `Hello` the statement's with the prover's nonce (`helloD`), the opening check the keyed
coin-tree walk at the round's place under `Hello`'s answer (`treeD`, `parseD`), the coin handed on the opening's words
(`coinOfD`), and the tape the masked prover's reading of the rest (`T.tapeZk`). -/
noncomputable def zkDeployed (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nr : Msg → ℕ → ℕ) (T : TableData) (t : ℕ) :
    Claims.Flock.ZkDeploy FlockDeploy.Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace :=
  (session nr T).zkD rel coinOfD (fun ω => T.tapeZk ω.2) t

/-- **Flock's zero knowledge against a malicious verifier at the deployed masked record**: `Claims.Flock.ZK` at
`zkDeployed`, at every rep rounds' coin counts `nr`, at `ε_zk = 8·n_L·δ₁ + 8t·n_L·δ₁ + 2·3^{-t}` with `n_L` the session's
pre-final leaves, under `Claims.Hm96Hiding` at the pinned key, the tables' five Props (`Tables`, `Complete`, `DummyReg`,
`Causal`, `TapeOK`) and `HonestMsgs`: `refP`'s messages are `ZkTables.honestPR`'s. -/
theorem Flock.zk_deployZk_ref (H : Msg → Msg) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nr : Msg → ℕ → ℕ) (T : TableData) (t : ℕ) (ht : 0 < t) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hS : (session nr T).Tables)
    (hV : (session nr T).Complete (Flock.R (FlockDeploy.deployZk rel).parse)) (hD : (session nr T).DummyReg)
    (hc : (session nr T).Causal) (hT : (session nr T).TapeOK)
    (hH : (session nr T).HonestMsgs H coinOfD fun ω => T.tapeZk ω.2) :
    Claims.Flock.ZK H (zkDeployed rel nr T t)
      (fun x => 8 * (session nr T).nLeaves x * δ₁.toReal + 8 * t * (session nr T).nLeaves x * δ₁.toReal +
        2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t :=
  Flock.zk_zkD_ref H rel (session nr T) t ht hT1 hδ₁ hS hV hD hc hT coinOfD _ hH

/-- **The same at the pinned key's hiding gap** (`CROnly.hm96Hiding_gap`), with `Claims.Hm96Hiding` discharged. -/
theorem Flock.zk_deployZk_ref_gap (H : Msg → Msg)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (nr : Msg → ℕ → ℕ) (T : TableData) (t : ℕ)
    (ht : 0 < t) (hS : (session nr T).Tables)
    (hV : (session nr T).Complete (Flock.R (FlockDeploy.deployZk rel).parse)) (hD : (session nr T).DummyReg)
    (hc : (session nr T).Causal) (hT : (session nr T).TapeOK)
    (hH : (session nr T).HonestMsgs H coinOfD fun ω => T.tapeZk ω.2) :
    Claims.Flock.ZK H (zkDeployed rel nr T t)
      (fun x => 8 * (session nr T).nLeaves x *
          (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash)).toReal +
        8 * t * (session nr T).nLeaves x *
          (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash)).toReal +
        2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t :=
  Flock.zk_zkD_ref_gap H rel (session nr T) t ht hS hV hD hc hT coinOfD _ hH

end Claims.ZKMal

namespace Claims.Witness.Flock

open Claims Claims.ZKMal Claims.ZKMal.CoinLayer

/-- **`ZK`'s checkable antecedents at the deployed masked record**: it is masked, and its coin layer is the deployed one:
`Hello` is `helloD`, an opening is checked by the keyed coin-tree walk at its round's place under `Hello`'s answer, the coin
handed on is the opening's words, and the prover aborts with the empty message. -/
theorem ZK_meets (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (nr : Msg → ℕ → ℕ)
    (T : TableData) (t : ℕ) :
    (zkDeployed rel nr T t).zk = true ∧ (zkDeployed rel nr T t).hello = helloD ∧
      (zkDeployed rel nr T t).chk = (fun x ν ρ j m => treeChk (treeD ρ) (nb ν) (rootD ρ) (place nr x j).pos
        (place nr x j).idx (place nr x j).n (place nr x j).depth (parseD (place nr x j) ρ m)) ∧
      (zkDeployed rel nr T t).coinOf = coinOfD ∧ (zkDeployed rel nr T t).abort = [] :=
  ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- **The degenerate coin layer is not the deployed record's**: at every rep rounds' coin counts, tables and threshold,
the deployed record's opening check refuses the empty message at every round, so it is not the check that passes
everything (`ZKMal.toyZD`'s), and the coin it hands on is not the opening itself. -/
theorem ZK_coins_fails (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (nr : Msg → ℕ → ℕ)
    (T : TableData) (t : ℕ) :
    (zkDeployed rel nr T t).chk ≠ (fun _ _ _ _ _ => .ret true) ∧ (zkDeployed rel nr T t).coinOf ≠ id := by
  refine ⟨fun h => ?_, fun h => ?_⟩
  · have e := congrFun (congrFun (congrFun (congrFun (congrFun h []) (fun _ => 0)) []) 0) []
    change treeChk (treeD []) (nb fun _ => 0) (rootD []) (place nr [] 0).pos (place nr [] 0).idx (place nr [] 0).n
      (place nr [] 0).depth (parseD (place nr [] 0) [] []) = _ at e
    rw [parseD_nil] at e
    exact absurd e (by simp [treeChk])
  · have e := congrFun h [0, 0, 0, 0, 0, 0]
    change coinOfD [0, 0, 0, 0, 0, 0] = [0, 0, 0, 0, 0, 0] at e
    exact absurd e (by decide)

end Claims.Witness.Flock
