import Proofs.Flock.Claims.ZK.Deployed
import Proofs.Flock.Claims.ZK.CoinBind

/-!
# The reference prover of `flock-circuit --zk`'s outer layer

`Flock.zk_deployZk_of_tables_bind` takes a committed-coin protocol `G` and three Props of it. `RefSession.refP` is that `G`:
the Lean reference prover, defined from the session's data (`RefSession`), what the ZK server releases on the witness and
its own randomness. Its messages are the session's by definition:

* `Hello`, then before the last coin the round encoding `enc` of the session's pre-final view cut where it stops
  (`cutAt`), at the executable's leaves (`exeOf`), at the coins opened so far padded to a full vector (`RefSession.pad`);
* after the last coin, `fin` of the tables' real views (`Coined.viewExe`) where every table admits the coins
  (`CoinedSetup.Admits`), else `off` of the cut view: the prover refuses by construction where the self-check fails;
* the opening check is the coin-tree walk `treeChk` on the message read as an opening at its round (`parse`), and the
  coin it hands on a reading `rd` of the message, one coin for two messages that read as one opening (`rd_eq`).

The tables' randomness at a coin vector (`RefSession.randAt`) reads the same tape as the pre-final leaves: each salt the
final message reveals is its leaf's (`baseWith`), so the real views' pre-final part is the cut view the round messages
encode (`viewPre_viewExe`).

What `zk_deployZk_of_tables_bind` asks of `G` is then a theorem about `refP`, from five Props of the session's data:
`RefSession.binding` (none: `Proto.binding_of_open`), `RefSession.tablesOK` (`Tables`: the tables are tables the
session sets up, which `setupTable_planZHJ` proves of the hidden-output plan's), and `RefSession.msgLink` (`Complete`: the witness's table
witnesses valid, the lowering's completeness; `DummyReg`: the dummy's carry the public region data; `Causal`: round `i`'s
encoding reads the coins before `i` only; `TapeOK`: the tape read with fibers of one size). `Flock.zkAt_deployZk_ref` is
`Claims.Flock.ZKAt` at `refP`'s prover; `Flock.zk_zkD_ref` is `Claims.Flock.ZK` at the masked record
`RefSession.zkD`, whose reference prover is `ZkTables.honestPR` behind `refP`'s `Hello` and coin-tree checks, where `refP`'s
messages are `ZkTables.honestPR`'s (`RefSession.HonestMsgs`, a named Prop until it is proved). `Flock.zk_deployZk_ref` is its case
at the deployed coin layer (`ZK.CoinLayer`).
-/

namespace Claims.ZKMal

open Claims
open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-! ## A table's real view before the last coin -/

/-- **The pre-final messages of a table's view**: every message but the final one's. -/
def viewPre {K W Hi VL : Type} [Field K] {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (v : T.tab.View) :
    T.tab.PreView where
  sHat := v.sHat
  sReg := v.sReg
  masked := v.masked
  y1 := v.y1
  ybar := v.ybar
  lig := v.lig
  inner := v.inner
  Y1 := v.Y1
  Y2 := v.Y2
  leaves0 := v.leaves0
  tauCom := v.tauCom
  leavesP := v.leavesP

/-- **A base with the final message's salts read from the pre-final leaves'** (`y`, by tree position): level 0's opened
leaves', each rep's opened pads leaves' and `y_τ`, so that each salt the final message reveals is its leaf's. -/
def baseWith {K W Hi VL : Type} [Field K] {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (b : ZkView.Base z Hi)
    (y : ZkView.LeafPos z → ZkView.Salt) : ZkView.Base z Hi :=
  (b.1, b.2.1, b.2.2.1, b.2.2.2.1, b.2.2.2.2.1, b.2.2.2.2.2.1, fun q => y (.inl (T.s0 (.inl q))),
    fun r k => y (.inr (.inl (r, T.sP r (.inl k)))), fun r => y (.inr (.inr r)), b.2.2.2.2.2.2.2.2.2)

/-- The hidden leaves' salts, read from the pre-final leaves' by tree position. -/
def hidOf {K W Hi VL : Type} [Field K] {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z)
    (y : ZkView.LeafPos z → ZkView.Salt) : T.tab.Hid → ZkView.Salt
  | .inl q => y (.inl (T.s0 (.inr q)))
  | .inr (r, k) => y (.inr (.inl (r, T.sP r (.inr k))))

/-- **The real view's pre-final part is the pre-final view at the executable's leaves**: at the table randomness whose
revealed salts are the leaves' (`baseWith`) and whose hidden leaves' salts are theirs (`hidOf`), whatever the ideal
seeds `u`. -/
theorem viewPre_viewExe {K W Hi VL : Type} [Field K] {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W)
    (b : ZkView.Base z Hi) (y : ZkView.LeafPos z → ZkView.Salt) (u : ZkView.LeafPos z → ZkView.Bits × ZkView.Salt) :
    viewPre T (T.viewExe w (T.randOf (baseWith T b y) u) (hidOf T y)) =
      T.preWith w b fun p => T.exeLeaf w b p (y p) := by
  refine FlockSoundness.ZK.Table.PreView.ext rfl rfl rfl rfl rfl rfl rfl rfl rfl ?_ ?_ ?_
  · refine Prod.ext (funext fun q => ?_) (funext fun q => ?_)
    · show ZkView.leaf0 _ _ = _root_.Flock.Hm96.Default512.leaf (T.content w b (.inl (T.s0 (.inl q)))) _
      simp only [ZkView.Coined.content, ZkView.Coined.col0At, Equiv.symm_apply_apply]
      rfl
    · show ZkView.leaf0 _ _ = _root_.Flock.Hm96.Default512.leaf (T.content w b (.inl (T.s0 (.inr q)))) _
      simp only [ZkView.Coined.content, ZkView.Coined.col0At, Equiv.symm_apply_apply]
      rfl
  · exact funext fun r => rfl
  · refine funext fun r => Prod.ext (funext fun k => ?_) (funext fun k => ?_)
    · show ZkView.leafP _ _ = _root_.Flock.Hm96.Default512.leaf (T.content w b (.inr (.inl (r, T.sP r (.inl k))))) _
      simp only [ZkView.Coined.content, ZkView.Coined.colPAt, Equiv.symm_apply_apply]
      rfl
    · show ZkView.leafP _ _ = _root_.Flock.Hm96.Default512.leaf (T.content w b (.inr (.inl (r, T.sP r (.inr k))))) _
      simp only [ZkView.Coined.content, ZkView.Coined.colPAt, Equiv.symm_apply_apply]
      rfl

/-! ## The reference session -/

/-- The protocol's coin rounds at a statement: the deployed verifier's rounds but the last message. -/
def refK (x : Msg) : ℕ := (FlockDeploy.V true).k x - 1

/-- **The reference session of `flock-circuit --zk`'s outer layer**: what the reference prover reads. The coins, the
coin tree each `Hello` commits, how a message is read as an opening at its round and as a coin (`Binding`'s data: two
messages that read as one opening read as one coin); the nonce and the tape; at each
statement, its `J` tables (`CoinedSetup`), their coins at the protocol's coins, the witness's table witnesses and the
dummy's; the round encoding `enc` of the session's cut pre-final view, the last message `fin` of the tables' views and
`off` of the cut view where the prover refuses; and the tape read as each table's base, every pre-final leaf's salt and
every pre-final leaf's ideal seed. -/
structure RefSession where
  Coin : Type
  [decCoin : DecidableEq Coin]
  [finCoin : Fintype Coin]
  [inhCoin : Inhabited Coin]
  Nonce : Type
  [finNonce : Fintype Nonce]
  [neNonce : Nonempty Nonce]
  Tape : Type
  [finTape : Fintype Tape]
  [neTape : Nonempty Tape]
  CoinT : Type
  CSalt : Type
  Tr : Msg → Msg → FlockSoundness.ZK.GK.CoinTree Dg CoinT CSalt
  nb : Nonce → List UInt8
  root : Msg → Msg → Dg
  parse : Msg → Msg → ℕ → Msg → Option (FlockSoundness.ZK.GK.CoinOpening Dg CoinT CSalt)
  pos : Msg → Msg → ℕ → List UInt8
  idx : Msg → Msg → ℕ → ℕ
  len : Msg → Msg → ℕ → ℕ
  depth : Msg → Msg → ℕ → ℕ
  rd : Msg → Coin
  rd_eq : ∀ x ρ j m m' op, parse x ρ j m = some op → parse x ρ j m' = some op → rd m = rd m'
  hello : Msg → Nonce → Msg
  abort : Msg
  J : Msg → ℕ
  C : (x : Msg) → Fin (J x) → Composed.CoinedSetup
  cz : (x : Msg) → (Fin (refK x) → Coin) → ∀ j, Composed.CoinsZK (C x j).Rest
  wt : (x : Msg) → Flock.Wit → ∀ j, FlockSoundness.Model.Witness (C x j).S.m
  wt₀ : (x : Msg) → ∀ j, FlockSoundness.Model.Witness (C x j).S.m
  enc : (x : Msg) → (c : Fin (refK x) → Coin) → ℕ → (∀ j, ((C x j).T (cz x c j)).tab.Cut) → Msg
  fin : (x : Msg) → (c : Fin (refK x) → Coin) → (∀ j, ((C x j).T (cz x c j)).tab.View) → Msg
  off : (x : Msg) → (c : Fin (refK x) → Coin) → (∀ j, ((C x j).T (cz x c j)).tab.Cut) → Msg
  tape : (x : Msg) → Tape → (∀ j, ZkView.Base (C x j).z (C x j).Hi) ×
    (ZkView.Leaves (fun j => (C x j).z) → ZkView.Salt) × (ZkView.Leaves (fun j => (C x j).z) → ZkView.Bits × ZkView.Salt)

attribute [instance] RefSession.decCoin RefSession.finCoin RefSession.inhCoin RefSession.finNonce RefSession.neNonce
  RefSession.finTape RefSession.neTape

namespace RefSession

/-- The coins opened so far, padded to a full vector. -/
def pad (S : RefSession) {K : ℕ} (cs : List S.Coin) : Fin K → S.Coin := fun j => cs.getD j.val default

/-- What an opening reads as. -/
def decOf (S : RefSession) (m : Msg) : S.Coin := S.rd m

/-- The tape before the last coin: each table's base and every pre-final leaf's salt. -/
def eP (S : RefSession) (x : Msg) (t : S.Tape) :
    (∀ j, ZkView.Base (S.C x j).z (S.C x j).Hi) × (ZkView.Leaves (fun j => (S.C x j).z) → ZkView.Salt) :=
  ((S.tape x t).1, (S.tape x t).2.1)

/-- **The session's cut pre-final view at coins `c`**, on table witnesses `wt` and tape `t`, every pre-final leaf the
executable's under its own salt. -/
noncomputable def cutOf (S : RefSession) (x : Msg) (c : Fin (refK x) → S.Coin)
    (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m) (t : S.Tape) :
    ∀ j, ((S.C x j).T (S.cz x c j)).tab.Cut :=
  cutAt (S.C x) (S.cz x c) wt (S.eP x t).1 (exeOf (S.C x) (S.cz x c) wt (S.eP x t).1 (S.eP x t).2)

/-- **The tables' randomness and hidden leaves' salts at coins `c`**, from tape `t`: each table's base with the leaves'
revealed salts (`baseWith`), the ideal seeds, and the hidden leaves' salts (`hidOf`). -/
noncomputable def randAt (S : RefSession) (x : Msg) (c : Fin (refK x) → S.Coin) (t : S.Tape) :
    ∀ j, ((S.C x j).T (S.cz x c j)).tab.Rand × (((S.C x j).T (S.cz x c j)).tab.Hid → ZkView.Salt) :=
  fun j => (((S.C x j).T (S.cz x c j)).randOf (baseWith ((S.C x j).T (S.cz x c j)) ((S.tape x t).1 j)
      fun p => (S.tape x t).2.1 ⟨j, p⟩) fun p => (S.tape x t).2.2 ⟨j, p⟩,
    hidOf ((S.C x j).T (S.cz x c j)) fun p => (S.tape x t).2.1 ⟨j, p⟩)

/-- The tables' real views at coins `c`. -/
noncomputable def viewsAt (S : RefSession) (x : Msg) (c : Fin (refK x) → S.Coin)
    (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m) (t : S.Tape) : ∀ j, ((S.C x j).T (S.cz x c j)).tab.View :=
  fun j => ((S.C x j).T (S.cz x c j)).viewExe (wt j) (S.randAt x c t j).1 (S.randAt x c t j).2

/-- The cut of the tables' views' pre-final part, where the session stops. -/
noncomputable def cutV (S : RefSession) (x : Msg) (c : Fin (refK x) → S.Coin)
    (V : ∀ j, ((S.C x j).T (S.cz x c j)).tab.View) : ∀ j, ((S.C x j).T (S.cz x c j)).tab.Cut :=
  fun j => ((S.C x j).T (S.cz x c j)).tab.cut
    (famSent (fun j => ((S.C x j).T (S.cz x c j)).tab) (fun j => ((S.C x j).T (S.cz x c j)).rep) j)
    (viewPre _ (V j))

/-- Every table admits the coins. -/
def Admits (S : RefSession) (x : Msg) (c : Fin (refK x) → S.Coin) : Prop := ∀ j, (S.C x j).Admits (S.cz x c j)

open Classical in
/-- **The reference prover's message after the coins `cs`**, on table witnesses `wt` and tape `t`: before the last coin
the round encoding of the cut view at the padded coins; after it, `fin` of the tables' views where they admit the coins,
else `off` of the cut view. -/
noncomputable def msgAt (S : RefSession) (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (t : S.Tape) (cs : List S.Coin) : Msg :=
  if cs.length < refK x then S.enc x (S.pad cs) cs.length (S.cutOf x (S.pad cs) wt t)
  else if S.Admits x (S.pad cs) then S.fin x (S.pad cs) (S.viewsAt x (S.pad cs) wt t)
  else S.off x (S.pad cs) (S.cutOf x (S.pad cs) wt t)

/-- **The reference prover** `P` of the outer `--zk` layer, as a committed-coin protocol: the session's coins, nonce and
tape; the coin-tree walk as its opening check; its messages `msgAt` on the witness's table witnesses, and on the dummy's
for the dummy-witness prover. Its SHVZK simulator is the models' (`Proto.withSim`). -/
noncomputable def refP (S : RefSession) : Proto Flock.Wit where
  Coin := S.Coin
  Nonce := S.Nonce
  Rand := S.Tape
  SimRand _ := Unit
  K := refK
  hello := S.hello
  abort := S.abort
  dec := S.decOf
  chk x ν ρ j m :=
    treeChk (S.Tr x ρ) (S.nb ν) (S.root x ρ) (S.pos x ρ j) (S.idx x ρ j) (S.len x ρ j) (S.depth x ρ j)
      (S.parse x ρ j m)
  msg x w _ t cs := .ret (S.msgAt x (S.wt x w) t cs)
  dmsg x _ t cs := .ret (S.msgAt x (S.wt₀ x) t cs)
  sim _ _ _ _ := .ret []

noncomputable instance instFintypeCoinWitRefP (S : RefSession) : Fintype S.refP.Coin := S.finCoin

/-! ## Its models -/

/-- The coins the answers `a` open: coin `j` is answer `j + 1`'s. -/
def coinsOf (S : RefSession) {K : ℕ} (a : Fin (K + 1) → Out) : Fin K → S.Coin := fun j => S.decOf (msgOf (a j.succ))

/-- The fiber size of the tape before the last coin. -/
noncomputable def kP (S : RefSession) (x : Msg) : ℕ :=
  Nat.card {t : S.Tape // S.eP x t = S.eP x (Classical.arbitrary _)}

/-- The fiber size of the tape at coins `c`. -/
noncomputable def kT (S : RefSession) (x : Msg) (c : Fin (refK x) → S.Coin) : ℕ :=
  Nat.card {t : S.Tape // S.randAt x c t = S.randAt x c (Classical.arbitrary _)}

/-- **The model before the last coin**: the session's tables at the coins the answers open, and the round encoding of the
cut view as the committed-coin prover sends it (`Hello`, then `enc`, or `abort` from the first opening that fails). -/
noncomputable def preModel (S : RefSession) (H : Msg → Msg) (x : Msg) (ν : S.Nonce) : S.refP.TPreModel x where
  J := S.J x
  C := S.C x
  cz a := S.cz x (S.coinsOf a)
  out a σ i := S.refP.pmG H x ν (fun _ => .ret (S.enc x (S.coinsOf a) (i.val - 1) σ)) (ansHist a i)
  wt := S.wt x
  wt₀ := S.wt₀ x
  eR := S.eP x
  kR := S.kP x

open Classical in
/-- **The model at a coin vector**: where every table admits the coins, the tables' views with `enc` of their cut
pre-final part and `fin` (`TabModel`); else the cut view with `enc` and `off` (`TOffModel`). -/
noncomputable def coinModel (S : RefSession) (x : Msg) (c : Fin (refK x) → S.Coin) : S.refP.TCoinModel x :=
  if S.Admits x c then
    .inl { J := S.J x, C := S.C x, cz := S.cz x c, wt := S.wt x, wt₀ := S.wt₀ x
           lift := fun V => (List.range (refK x + 1)).map fun j =>
             if j < refK x then S.enc x c j (S.cutV x c V) else S.fin x c V
           eR := S.randAt x c, kR := S.kT x c }
  else
    .inr { J := S.J x, C := S.C x, cz := S.cz x c, wt := S.wt x, wt₀ := S.wt₀ x
           out := fun σ => (List.range (refK x + 1)).map fun j => if j < refK x then S.enc x c j σ else S.off x c σ
           eR := S.eP x, kR := S.kP x }

/-! ## What remains of the session's data -/

/-- **The tables are tables the session sets up** (`setupTable_planZHJ` at the hidden-output plan's). -/
def Tables (S : RefSession) : Prop := ∀ x j, SetupTable (S.C x j)

/-- **The lowering's completeness**: a witness in the relation has valid table witnesses (`CoinedSetup.Valid`). -/
def Complete (S : RefSession) (R : Rel Flock.Wit) : Prop := ∀ x w, R x w → ∀ j, (S.C x j).Valid (S.wt x w j)

/-- **The dummy's table witnesses carry the public region data**, at every coin vector. -/
def DummyReg (S : RefSession) : Prop :=
  ∀ x c j, ((S.C x j).T (S.cz x c j)).tab.sReg (S.wt₀ x j) = ((S.C x j).T (S.cz x c j)).tab.pubReg

/-- **Round `i`'s encoding reads the coins before `i` only**, whatever each pre-final leaf's pair `(M·y, c(y))`. -/
def Causal (S : RefSession) : Prop :=
  ∀ (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m) (ω : ∀ j, ZkView.Base (S.C x j).z (S.C x j).Hi)
    (p : ZkView.Leaves (fun j => (S.C x j).z) → ZkView.Bits × ByteArray) (c c' : Fin (refK x) → S.Coin) (i : ℕ),
    (∀ j : Fin (refK x), j.val < i → c j = c' j) →
    S.enc x c i (cutAt (S.C x) (S.cz x c) wt ω fun l =>
        ZkView.leafH (digOf (S.C x) (S.cz x c) wt ω l + (p l).1, (p l).2)) =
      S.enc x c' i (cutAt (S.C x) (S.cz x c') wt ω fun l =>
        ZkView.leafH (digOf (S.C x) (S.cz x c') wt ω l + (p l).1, (p l).2))

/-- **The tape is read with fibers of one size**: before the last coin as the bases and the leaves' salts, and at each
coin vector as the tables' randomness and hidden salts (`randAt`). Both hold where the tape reads the bases, the salts
and the seeds uniformly (`tape` with fibers of one size). -/
def TapeOK (S : RefSession) : Prop :=
  ∀ x, (∀ p p', Nat.card {t // S.eP x t = p} = Nat.card {t // S.eP x t = p'}) ∧
    ∀ c p p', Nat.card {t // S.randAt x c t = p} = Nat.card {t // S.randAt x c t = p'}

/-! ## The three Props, as theorems about `refP` -/

/-- **`Binding` holds of `refP`**: its check is the coin-tree walk on the message read at its round, and two messages
that read as one opening read as one coin (`Proto.binding_of_open`). -/
theorem binding (S : RefSession) (H : Msg → Msg) : S.refP.Binding H :=
  S.refP.binding_of_open H S.Tr S.nb S.root S.parse S.pos S.idx S.len S.depth (fun _ _ _ _ _ => rfl) S.rd_eq

/-- **`TablesOK` holds of `refP`'s models** at tables the session sets up: where the prover goes on, every table admits
the coins, by `refP`'s definition. -/
theorem tablesOK (S : RefSession) (H : Msg → Msg) (hT : S.Tables) :
    S.refP.TablesOK (fun x ν => S.preModel H x ν) fun x _ c => S.coinModel x c := by
  refine ⟨fun x _ => hT x, fun x _ c => ?_⟩
  show (S.coinModel x c).OK
  unfold coinModel
  split_ifs with h
  · exact ⟨hT x, h⟩
  · exact hT x

/-! ## The message link -/

/-- The padded coins at a prefix of `c` agree with `c` before it. -/
theorem pad_take (S : RefSession) {K : ℕ} (c : Fin K → S.Coin) (j : ℕ) (k : Fin K) (hk : k.val < j) :
    S.pad ((List.ofFn c).take j) k = c k := by
  simp [pad, hk]

theorem pad_ofFn (S : RefSession) {K : ℕ} (c : Fin K → S.Coin) : S.pad ((List.ofFn c).take K) = c :=
  funext fun k => S.pad_take c K k k.isLt

/-- **The real views' pre-final part, cut, is the cut view the round messages encode** (`viewPre_viewExe`). -/
theorem cutV_viewsAt (S : RefSession) (x : Msg) (c : Fin (refK x) → S.Coin)
    (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m) (t : S.Tape) :
    S.cutV x c (S.viewsAt x c wt t) = S.cutOf x c wt t := by
  funext j
  simp only [cutV, viewsAt, randAt, viewPre_viewExe]
  rfl

/-- **Round `i`'s encoding of the cut view reads the coins before `i` only** (`Causal` at the executable's leaves). -/
theorem enc_cutOf (S : RefSession) (x : Msg) (hc : S.Causal) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (t : S.Tape) (c c' : Fin (refK x) → S.Coin) (i : ℕ) (h : ∀ k : Fin (refK x), k.val < i → c k = c' k) :
    S.enc x c i (S.cutOf x c wt t) = S.enc x c' i (S.cutOf x c' wt t) := by
  have e := hc x wt (S.eP x t).1 (fun l => (ZkView.My ((S.eP x t).2 l), ZkView.saltHash ((S.eP x t).2 l))) c c' i h
  unfold cutOf
  rw [exeOf_eq, exeOf_eq]
  convert e using 4

open Classical in
/-- **The reference prover's messages at the prefixes of a coin vector** (`Causal`): round `j`'s encoding at `c` before
the last coin, then the last message at `c`. -/
theorem msgAt_take (S : RefSession) (x : Msg) (hc : S.Causal) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (t : S.Tape) (c : Fin (refK x) → S.Coin) (j : ℕ) (hj : j < refK x + 1) :
    S.msgAt x wt t ((List.ofFn c).take j) =
      if j < refK x then S.enc x c j (S.cutOf x c wt t)
      else if S.Admits x c then S.fin x c (S.viewsAt x c wt t) else S.off x c (S.cutOf x c wt t) := by
  have hl : ((List.ofFn c).take j).length = j := by simp only [List.length_take, List.length_ofFn]; omega
  unfold msgAt
  rw [hl]
  by_cases h : j < refK x
  · rw [ite_eq_left h, ite_eq_left h]
    exact S.enc_cutOf x hc wt t _ c j fun k hk => S.pad_take c j k hk
  · rw [ite_eq_right h, ite_eq_right h]
    obtain rfl : j = refK x := by omega
    rw [S.pad_ofFn c]

theorem strat_congr_gen {Coin : Type} (hello abort : Msg) (dec : Msg → Coin) (acc : Msg → ℕ → Msg → OC Bool)
    (gen gen' : List Coin → OC Msg) :
    ∀ h : List (Msg × Msg), (h ≠ [] → gen (h.tail.map fun p => dec p.2) = gen' (h.tail.map fun p => dec p.2)) →
      strat hello abort dec acc gen h = strat hello abort dec acc gen' h
  | [], _ => rfl
  | (a, ρ) :: ops, e => by
      have e' : gen (ops.map fun p => dec p.2) = gen' (ops.map fun p => dec p.2) := e (List.cons_ne_nil _ _)
      simp only [strat, e']

theorem pad_ansHist (S : RefSession) {K : ℕ} (a : Fin (K + 1) → Out) (i : Fin (K + 1)) (k : Fin K)
    (hk : k.val < i.val - 1) : S.pad ((ansHist a i).tail.map fun p => S.decOf p.2) k = S.coinsOf a k := by
  have hk' : k.val + 1 < i.val := by omega
  simp only [pad, coinsOf, ansHist, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_tail,
    List.getElem?_ofFn, hk', dite_true, Option.map_some, Option.getD_some]
  rfl

/-- **Before the last coin, the reference prover sends the round encoding at the coins the answers open**
(`Causal`). -/
theorem pre_msg (S : RefSession) (H : Msg → Msg) (x : Msg) (ν : S.Nonce) (hc : S.Causal)
    (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m) (t : S.Tape) (a : Fin (refK x + 1) → Out)
    (i : Fin (refK x + 1)) :
    S.refP.pmG H x ν (fun cs => .ret (S.msgAt x wt t cs)) (ansHist a i) =
      S.refP.pmG H x ν (fun _ => .ret (S.enc x (S.coinsOf a) (i.val - 1) (S.cutOf x (S.coinsOf a) wt t)))
        (ansHist a i) := by
  unfold Proto.pmG
  congr 2
  refine strat_congr_gen _ _ _ _ _ _ _ fun hne => ?_
  have hi : 0 < i.val := by
    rcases Nat.eq_zero_or_pos i.val with h0 | h0
    · exact absurd (by simp [ansHist, h0]) hne
    · exact h0
  have hl : ((ansHist a i).tail.map fun p => S.decOf p.2).length = i.val - 1 := by
    simp only [List.length_map, List.length_tail, ansHist, List.length_ofFn]
  show OC.ret (S.msgAt x wt t ((ansHist a i).tail.map fun p => S.decOf p.2)) = OC.ret _
  unfold msgAt
  rw [hl, ite_eq_left (by omega), S.enc_cutOf x hc wt t _ (S.coinsOf a) (i.val - 1) fun k hk => S.pad_ansHist a i k hk]

theorem preModel_link (S : RefSession) (H : Msg → Msg) (R : Rel Flock.Wit) (hV : S.Complete R) (hD : S.DummyReg)
    (hc : S.Causal) (hT : S.TapeOK) (x : Msg) (w : Flock.Wit) (hR : R x w) (ν : S.Nonce) :
    (S.preModel H x ν).Link H ν w := by
  refine ⟨hV x w hR, fun a j => hD x (S.coinsOf a) j, ?_, fun p => (hT x).1 p _, fun t a => funext fun i =>
    S.pre_msg H x ν hc (S.wt x w) t a i, fun t a => funext fun i => S.pre_msg H x ν hc (S.wt₀ x) t a i⟩
  intro wt ω p a a' i h
  show S.refP.pmG H x ν _ (ansHist a i) = S.refP.pmG H x ν _ (ansHist a' i)
  rw [ansHist_congr a a' i h]
  refine congrArg (fun g => S.refP.pmG H x ν g (ansHist a' i)) (funext fun _ => congrArg OC.ret ?_)
  exact hc x wt ω p (S.coinsOf a) (S.coinsOf a') (i.val - 1) fun k hk =>
    congrArg (fun o => S.decOf (msgOf o)) (h k.succ (Fin.lt_def.mpr (by simp only [Fin.val_succ]; omega)))

theorem coinModel_link (S : RefSession) (H : Msg → Msg) (R : Rel Flock.Wit) (hV : S.Complete R) (hD : S.DummyReg)
    (hc : S.Causal) (hT : S.TapeOK) (x : Msg) (w : Flock.Wit) (hR : R x w) (ν : S.Nonce)
    (c : Fin (refK x) → S.Coin) : (S.coinModel x c).Link H ν c w := by
  unfold coinModel
  split_ifs with hA
  · refine ⟨hV x w hR, fun p => (hT x).2 c p _, fun (t : S.Tape) => ?_⟩
    show (List.range (refK x + 1)).map (fun j => S.msgAt x (S.wt x w) t ((List.ofFn c).take j)) =
      (List.range (refK x + 1)).map fun j =>
        if j < refK x then S.enc x c j (S.cutV x c (S.viewsAt x c (S.wt x w) t)) else S.fin x c (S.viewsAt x c (S.wt x w) t)
    refine List.map_congr_left fun j hj => ?_
    rw [S.msgAt_take x hc _ t c j (List.mem_range.1 hj), S.cutV_viewsAt, ite_eq_left hA]
  · refine ⟨hV x w hR, fun j => hD x c j, fun p => (hT x).1 p _, fun (t : S.Tape) => ?_,
      fun (t : S.Tape) => ?_⟩
    · show (List.range (refK x + 1)).map (fun j => S.msgAt x (S.wt x w) t ((List.ofFn c).take j)) =
        (List.range (refK x + 1)).map fun j =>
          if j < refK x then S.enc x c j (S.cutOf x c (S.wt x w) t) else S.off x c (S.cutOf x c (S.wt x w) t)
      refine List.map_congr_left fun j hj => ?_
      rw [S.msgAt_take x hc _ t c j (List.mem_range.1 hj), ite_eq_right hA]
    · show (List.range (refK x + 1)).map (fun j => S.msgAt x (S.wt₀ x) t ((List.ofFn c).take j)) =
        (List.range (refK x + 1)).map fun j =>
          if j < refK x then S.enc x c j (S.cutOf x c (S.wt₀ x) t) else S.off x c (S.cutOf x c (S.wt₀ x) t)
      refine List.map_congr_left fun j hj => ?_
      rw [S.msgAt_take x hc _ t c j (List.mem_range.1 hj), ite_eq_right hA]

/-- **`TMsgLink` holds of `refP`'s models**, from the lowering's completeness, the dummy's region data, the round
encoding's causality and the tape. -/
theorem msgLink (S : RefSession) (H : Msg → Msg) (R : Rel Flock.Wit) (hV : S.Complete R) (hD : S.DummyReg)
    (hc : S.Causal) (hT : S.TapeOK) :
    S.refP.TMsgLink H R (fun x ν => S.preModel H x ν) fun x _ c => S.coinModel x c := fun x w hR ν =>
  ⟨S.preModel_link H R hV hD hc hT x w hR ν, fun c => S.coinModel_link H R hV hD hc hT x w hR ν c⟩

/-! ## Flock's zero knowledge at the reference prover -/

/-- The session's pre-final leaves at a statement. -/
noncomputable def nLeaves (S : RefSession) (x : Msg) : ℕ := Fintype.card (ZkView.Leaves fun j => (S.C x j).z)

/-- A hidden leaf's tree position. -/
def hidPos {K W Hi VL : Type} [Field K] {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) :
    T.tab.Hid → ZkView.LeafPos z
  | .inl q => .inl (T.s0 (.inr q))
  | .inr (r, k) => .inr (.inl (r, T.sP r (.inr k)))

theorem hidPos_injective {K W Hi VL : Type} [Field K] {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) :
    Function.Injective (hidPos T) := by
  rintro (q | ⟨r, k⟩) (q' | ⟨r', k'⟩) h <;> simp only [hidPos, Sum.inl.injEq, Sum.inr.injEq, reduceCtorEq,
    Prod.mk.injEq] at h
  · exact congrArg Sum.inl (Sum.inr_injective (T.s0.injective h))
  · obtain ⟨rfl, h⟩ := h
    exact congrArg Sum.inr (Prod.ext rfl (Sum.inr_injective ((T.sP r).injective h)))

theorem coinModel_size (S : RefSession) (x : Msg) (c : Fin (refK x) → S.Coin) :
    (S.coinModel x c).size ≤ S.nLeaves x := by
  unfold coinModel
  split_ifs
  · show ∑ j, Fintype.card ((S.C x j).T (S.cz x c j)).tab.Hid ≤ Fintype.card (ZkView.Leaves fun j => (S.C x j).z)
    rw [Fintype.card_sigma]
    exact Finset.sum_le_sum fun j _ => Fintype.card_le_of_injective _ (hidPos_injective ((S.C x j).T (S.cz x c j)))
  · exact le_rfl

end RefSession

/-- **Flock's zero knowledge against a malicious verifier at `FlockDeploy.deployZk`, at the reference prover**:
`Claims.Flock.ZKAt` at `P_zk = refP`'s prover and its models' Goldreich–Kahan simulator, at
`ε_zk = 8·n_L·δ₁ + 8t·n_L·δ₁ + 2·3^{-t}` with `n_L` the session's pre-final leaves, under `Claims.Hm96Hiding` at the
pinned key and five Props of the session's data: `Tables`, `Complete`, `DummyReg`, `Causal`, `TapeOK`. -/
theorem Flock.zkAt_deployZk_ref (H : Msg → Msg) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (S : RefSession) (t : ℕ) (ht : 0 < t) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁)
    (hδ₁ : δ₁ ≠ ⊤) (hS : S.Tables) (hV : S.Complete (Flock.R (FlockDeploy.deployZk rel).parse)) (hD : S.DummyReg)
    (hc : S.Causal) (hT : S.TapeOK) :
    Claims.Flock.ZKAt H (FlockDeploy.deployZk rel) S.refP.prover
      ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t))
      (fun x => 8 * S.nLeaves x * δ₁.toReal + 8 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t)
      fun _ => 4 * t * 3 ^ t :=
  Flock.zk_deployZk_of_tables_bind H rel S.refP t ht (fun _ => rfl) hT1 hδ₁ (fun x ν => S.preModel H x ν)
    (fun x _ c => S.coinModel x c) S.nLeaves S.nLeaves (fun _ _ => le_rfl) (fun x _ c => S.coinModel_size x c)
    (S.msgLink H _ hV hD hc hT) (S.tablesOK H hS) (S.binding H)

/-! ## At the masked record -/

/-- **`refP`'s messages are the reference prover's** within its rounds, at `H`: after the openings `ops`, at most
`refK x` of them, `refP`'s message (`msgAt` at the coins they open) is `ZkTables.honestPR`'s at the history with each opening
read as its coin (`coinOf`), at the tape `tapeOf (ν, t)`. `refP` makes no call, so this is `Proto.Honest`
(`RefSession.honest_of_msgs`). -/
def RefSession.HonestMsgs (S : RefSession) (H : Msg → Msg) (coinOf : Msg → Msg)
    (tapeOf : S.refP.Nonce × S.refP.Rand → FlockComplete.TapeZk) : Prop :=
  ∀ x w ν t (ops : List (Msg × Msg)), ops.length ≤ refK x →
    S.msgAt x (S.wt x w) t (ops.map fun p => S.decOf p.2) =
      (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOf p.2)) (tapeOf (ν, t))).eval H).1

theorem RefSession.honest_of_msgs (S : RefSession) (H : Msg → Msg) (coinOf : Msg → Msg)
    (tapeOf : S.refP.Nonce × S.refP.Rand → FlockComplete.TapeZk) (h : S.HonestMsgs H coinOf tapeOf) :
    S.refP.Honest H FlockComplete.ZkTables.honestPR coinOf tapeOf :=
  fun x w ν t ops hops => ⟨h x w ν t ops hops, List.nil_subset _⟩

/-- **The masked record at the reference session**: `zkOf` at `refP`, the record's reference prover `ZkTables.honestPR` behind
`refP`'s `Hello` and coin-tree checks, its simulator the models' Goldreich–Kahan simulator at threshold `t`. -/
noncomputable def RefSession.zkD (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (S : RefSession) (coinOf : Msg → Msg) (tapeOf : S.refP.Nonce × S.refP.Rand → FlockComplete.TapeZk) (t : ℕ) :
    Claims.Flock.ZkDeploy FlockDeploy.Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace :=
  zkOf rel S.refP coinOf tapeOf ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t))

/-- **Flock's zero knowledge against a malicious verifier at the reference session's masked record**:
`Claims.Flock.ZK` at `RefSession.zkD`, whose reference prover is `ZkTables.honestPR`, at
`ε_zk = 8·n_L·δ₁ + 8t·n_L·δ₁ + 2·3^{-t}` with `n_L` the session's pre-final leaves, under `Claims.Hm96Hiding` at the
pinned key, the session's five Props (`Tables`, `Complete`, `DummyReg`, `Causal`, `TapeOK`) and `HonestMsgs`: `refP`'s
messages are `ZkTables.honestPR`'s. -/
theorem Flock.zk_zkD_ref (H : Msg → Msg) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (S : RefSession) (t : ℕ) (ht : 0 < t) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁)
    (hδ₁ : δ₁ ≠ ⊤) (hS : S.Tables) (hV : S.Complete (Flock.R (FlockDeploy.deployZk rel).parse)) (hD : S.DummyReg)
    (hc : S.Causal) (hT : S.TapeOK) (coinOf : Msg → Msg)
    (tapeOf : S.refP.Nonce × S.refP.Rand → FlockComplete.TapeZk) (hH : S.HonestMsgs H coinOf tapeOf) :
    Claims.Flock.ZK H (S.zkD rel coinOf tapeOf t)
      (fun x => 8 * S.nLeaves x * δ₁.toReal + 8 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t)
      fun _ => 4 * t * 3 ^ t :=
  zk_zkOf rel S.refP coinOf tapeOf _ (fun _ => by positivity) (deployZk_proto_k rel S.refP fun _ => rfl)
    (S.honest_of_msgs H coinOf tapeOf hH) (Flock.zkAt_deployZk_ref H rel S t ht hT1 hδ₁ hS hV hD hc hT)

/-- **The same at the pinned key's hiding gap** (`CROnly.hm96Hiding_gap`), with `Claims.Hm96Hiding` discharged. -/
theorem Flock.zk_zkD_ref_gap (H : Msg → Msg)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (S : RefSession) (t : ℕ) (ht : 0 < t)
    (hS : S.Tables) (hV : S.Complete (Flock.R (FlockDeploy.deployZk rel).parse)) (hD : S.DummyReg) (hc : S.Causal)
    (hT : S.TapeOK) (coinOf : Msg → Msg) (tapeOf : S.refP.Nonce × S.refP.Rand → FlockComplete.TapeZk)
    (hH : S.HonestMsgs H coinOf tapeOf) :
    Claims.Flock.ZK H (S.zkD rel coinOf tapeOf t)
      (fun x => 8 * S.nLeaves x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash)).toReal +
        8 * t * S.nLeaves x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash)).toReal +
        2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t :=
  Flock.zk_zkD_ref H rel S t ht (FlockSoundness.CROnly.hm96Hiding_gap ZkView.My ZkView.saltHash)
    ENNReal.ofReal_ne_top hS hV hD hc hT coinOf tapeOf hH

end Claims.ZKMal
