module

public import Proofs.Properties.Verity.SampledTwo.Projection
public import Proofs.Properties.Verity.SampledTwo.Reading
public import Proofs.Properties.Verity.Sampled.CoinMap

/-!
# The two-stage leg's coin map

`CoinMapLaw` (`Proofs.Properties.Verity.SampledTwo.Projection`) reads a play of the period's game as a run of the two-stage
verifier whose coins are written into the play's tape where the device releases them. This module builds that map and
states its sampling design.

* **Where a release goes** (`Locate`, `locAt`, `widthOf`, `usedOf`): a locator names, from the device's log before a
  frame, the frame and the bytes it answers, the run's round whose coin the frame's release continues. A family's fresh
  coin continues its round by 4 elements of GF(2^128) (512 bits), and an opening of `n` coins by `n` elements; a `Hello`
  continues none. Each round's releases continue its coin in log order, from its first element (`usedOf`). The locator
  reads only what the play fixed before the release's coin, so a release's place is a stopping order by construction:
  the tape is a tree, and each coin node's place in the play is the values before it.
* **The coin map** (`slot`, `freshVal`, `openBits`, `openW`, `placeK`, `place`, `frame`): at a fresh coin the locator puts
  on round `i`, the node's 512 bits become the round's coin's next 4 elements; at an opening it puts there, the node's
  round becomes one of those whose first `n` coins are the round's coin's next `n` elements (`pickIn` of the class,
  as the one-stage leg's `writeOpening` picks), the node's own where there is none; every other node keeps its value.
  `placeK` writes along the releases' continuation form `relK`, and `place` at the period's game (`game_eq`).
* **What the device released for a round** (`devBits`, `elemOfBits`, `dev`): the coin whose elements are the bits of
  the releases the locator puts on round `i`, in log order, padded with zeros; none where it puts none or more than a
  coin's 65536 elements.
* **The hiding term** (`bitsOf`, `gapK`, `gapOf`): along the frame's play, at each written opening, the written round's
  shortfall below uniform (`Verity.Game.shortfall`, the one-stage leg's `openGap` at each node) plus the chance its class
  at the run's uniform elements is empty. A fresh coin's write costs nothing: its 512 bits are the coin's 4 elements one
  for one.
* **The sampling design** (`LawAt`, `ElemAgrees`, `CoinMapHolds`): the period's play is at most as likely, at every
  event, as the frame's play at a uniform tape and uniform run coins plus the hiding term (`LawAt`); and `CoinMapLaw`
  at the coin map's agreement (`ElemAgrees`): on the frame's plays counted, the coin `dev` reads for each read round has
  the run's coin's elements below what its releases that fit carry (`DevAgrees.lean`'s `ElemsBelow` at `fitOf`,
  stated by its body), which its `FirstElems` and `SecondElems` read. The agreement holds only in the event: at an opening whose
  class at the run's elements is empty the device releases the tree's own round, whose bits are not the run's.
Statements and definitions; `LawAt` and `CoinMapHolds` are proved in `CoinMapLaw.lean` and `CoinMapAgree.lean`.
-/

@[expose] public section

namespace Verity.SampledTwo.CoinMap

open Verity.Game
open Claims
open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-! ## Where a release goes -/

/-- **A locator**: from the device's log before a frame, the frame and the bytes the prover sent with it, the run's
round whose coin the frame's release continues, or none. -/
abbrev Locate : Type := List Verity.Release → Verity.Frame → ByteArray → Option ℕ

/-- **The locator at a frame that releases coins**: a family's fresh coin or an opening; a `Hello` continues no round. -/
def locAt (L : Locate) (pre : List Verity.Release) : Verity.Frame → ByteArray → Option ℕ
  | .hello _, _ => none
  | fr, b => L pre fr b

/-- **The elements of GF(2^128) a frame's release continues its round by**: a fresh coin's 512 bits are 4, an opening
of `n` coins `n`. -/
def widthOf : Verity.Frame → ℕ
  | .fresh _ => 4
  | .hello _ => 0
  | .opening _ _ n => n

/-- **The elements the log `log` gives round `i`**, after the releases `pre`: the widths of the frames the locator puts
on round `i`, each located from the log before it. -/
def usedOf (L : Locate) (i : ℕ) : List Verity.Release → List Verity.Release → ℕ
  | _, [] => 0
  | pre, rl :: log =>
    (if locAt L pre rl.frame rl.answers = some i then widthOf rl.frame else 0) + usedOf L i (pre ++ [rl]) log

/-! ## The coin map -/

/-- **Bit `t` of a run's coin**: bit `t % 128` of its element `t / 128` (`F128.toBytes`' order: bit `j` of byte `b` is
the element's bit `8b + j`). -/
def coinBit (c : FlockDeploy.Coin) (t : ℕ) : Bool := (c.at (t / 128)).bit (t % 128)

/-- **Where a frame's release is written**, at `K` rounds: the round `i < K` the locator puts it on, and the element
`u` its round's coin continues from, where its `m` elements fit in the coin's 65536. -/
def slot (L : Locate) (K : ℕ) (pre : List Verity.Release) (fr : Verity.Frame) (b : ByteArray) (m : ℕ) :
    Option (Fin K × ℕ) :=
  match locAt L pre fr b with
  | some i =>
    if h : i < K then
      if usedOf L i [] pre + m ≤ 65536 then some (⟨i, h⟩, usedOf L i [] pre) else none
    else none
  | none => none

/-- **A fresh coin's written bits**: the coin `c`'s 4 elements from `u`. -/
def freshVal (c : FlockDeploy.Coin) (u : ℕ) : Fin 512 → Bool := fun j => coinBit c (128 * u + j.val)

/-- **An opening's written bits**: the coin `c`'s `n` elements from `u`, as the device releases a round's first `n`
coins (`TreeDraw.opened`). -/
def openBits (c : FlockDeploy.Coin) (u n : ℕ) : List Bool := List.ofFn fun t : Fin (128 * n) => coinBit c (128 * u + t.val)

open Classical in
/-- **An opening's write**: the round picked among those released as `bits` (`pickIn`, as the one-stage leg's
`writeOpening`), the node's own where none is. -/
noncomputable def openW {ν : ByteArray} {spec : Flock.CoinTree.Spec} {t : TreeSample spec} {s : Fin spec.streams.size}
    {r : Fin (2 ^ spec.roundsLog)} (n : ℕ) (bits : List Bool) (dr : TreeOpenable ν spec t s r) :
    TreeOpenable ν spec t s r :=
  pickIn (Finset.univ.filter fun e : TreeOpenable ν spec t s r => TreeDraw.opened e.1 n = bits) dr

/-- At an `ite` node: the tape placed by `fX` or `fY`, as the condition decides. -/
noncomputable def placeIte {β : Type} {c : Prop} :
    (dc : Decidable c) → (G₁ G₂ : Game β) → (s : Strategy (@ite _ c dc G₁ G₂)) →
      ((s : Strategy G₁) → Tape G₁ s → Tape G₁ s) → ((s : Strategy G₂) → Tape G₂ s → Tape G₂ s) →
        Tape (@ite _ c dc G₁ G₂) s → Tape (@ite _ c dc G₁ G₂) s
  | .isTrue _, _, _, s, fX, _ => fX s
  | .isFalse _, _, _, s, _, fY => fY s

/-- At an `optGame` node: the tape placed by `fn` at none, by `fy a` at `some a`. -/
noncomputable def placeOpt {β γ : Type} (n : Game β) (y : γ → Game β) (fn : (s : Strategy n) → Tape n s → Tape n s)
    (fy : (a : γ) → (s : Strategy (y a)) → Tape (y a) s → Tape (y a) s) :
    (o : Option γ) → (s : Strategy (optGame o n y)) → Tape (optGame o n y) s → Tape (optGame o n y) s
  | none, s => fn s
  | some a, s => fy a s

/-- At a coin node: the node's value written by `w`, and the tape after each value placed by `f`. -/
noncomputable def placeCoin {β C : Type} {iF : Fintype C} {iN : Nonempty C} (k : C → Game β) (w : C → C)
    (f : (c : C) → (s : Strategy (k c)) → Tape (k c) s → Tape (k c) s) (s : Strategy (@Game.coin β C iF iN k))
    (t : Tape (@Game.coin β C iF iN k) s) : Tape (@Game.coin β C iF iN k) s :=
  let t' : C × ((c : C) → Tape (k c) (s c)) := t
  ((w t'.1, fun c => f c (s c) (t'.2 c)) : C × ((c : C) → Tape (k c) (s c)))

/-! ### The releases' subgames, as `relK` writes them -/

/-- After a fresh coin of family `l` is taken from the layout, leaving `fs'`: its 512 bits, then the releases. -/
def freshY {β : Type} (k : ℕ) (ts : List OpenTree) (f : List Verity.Release → Game β) (l : String) (b : ByteArray)
    (fs' : List (String × ℕ)) : Game β :=
  .coin (Fin 512 → Bool) fun x => relK k fs' ts fun log => f (⟨.fresh l, b, List.ofFn x⟩ :: log)

/-- After a frame asking a fresh coin of family `l`. -/
def freshG {β : Type} (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree) (f : List Verity.Release → Game β)
    (l : String) (b : ByteArray) : Game β :=
  optGame (takeCoin l fs) (f []) (freshY k ts f l b)

/-- After a `Hello` the device accepts: the tree's sample, then the releases. -/
def helloX {β : Type} (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree) (f : List Verity.Release → Game β)
    (spec : Flock.CoinTree.Spec) (ν : ByteArray) : Game β :=
  .coin (TreeSample spec) fun t => relK k fs (ts ++ [⟨ν, spec, t, fun _ => 0⟩]) fun log =>
    f (⟨.hello spec, ν, bitsOfBytes (treeRoot ν spec t ++ treeKey t.1)⟩ :: log)

/-- After a `Hello` frame. -/
def helloG {β : Type} (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree) (f : List Verity.Release → Game β)
    (spec : Flock.CoinTree.Spec) (ν : ByteArray) : Game β :=
  if treeLines ts < maxTreeLines ∧ TreeSpecOk spec then helloX k fs ts f spec ν else f []

/-- After an opening of the tree `T`'s round `p`: the round, then the releases. -/
def openP {β : Type} (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree) (f : List Verity.Release → Game β)
    (i sn n : ℕ) (b : ByteArray) (T : OpenTree) (p : Fin T.spec.streams.size × Fin (2 ^ T.spec.roundsLog)) :
    Game β :=
  haveI : Nonempty (TreeOpenable T.nonce T.spec T.sample p.1 p.2) := ⟨⟨T.sample.2 p.1 p.2, rfl⟩⟩
  .coin (TreeOpenable T.nonce T.spec T.sample p.1 p.2) fun d =>
    relK k fs (ts.set i (T.bump p.1)) fun log => f (⟨.opening i sn n, b, TreeDraw.opened d.1 n⟩ :: log)

/-- After an opening frame finds the tree `T`. -/
def openT {β : Type} (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree) (f : List Verity.Release → Game β)
    (i sn n : ℕ) (b : ByteArray) (T : OpenTree) : Game β :=
  optGame (T.next sn n) (f []) (openP k fs ts f i sn n b T)

/-- After an opening frame. -/
def openG {β : Type} (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree) (f : List Verity.Release → Game β)
    (i sn n : ℕ) (b : ByteArray) : Game β :=
  if treeLines ts < maxTreeLines then optGame ts[i]? (f []) (openT k fs ts f i sn n b) else f []

/-- What the coin map does along `relK k`, from any layout, trees, releases before and continuation. -/
abbrev PlaceRec (β : Type) (k : ℕ) : Type 1 :=
  (fs : List (String × ℕ)) → (ts : List OpenTree) → (pre : List Verity.Release) → (f : List Verity.Release → Game β) →
    (s : Strategy (relK k fs ts f)) → Tape (relK k fs ts f) s → Tape (relK k fs ts f) s

/-! ### The coin map, frame by frame -/

/-- **At a fresh coin**: written with the round's coin's next 4 elements where the locator puts it and they fit. -/
noncomputable def placeFresh {β : Type} (L : Locate) {K : ℕ} (r : Fin K → FlockDeploy.Coin) (k : ℕ)
    (rec : PlaceRec β k) (fs : List (String × ℕ)) (ts : List OpenTree) (pre : List Verity.Release)
    (f : List Verity.Release → Game β) (l : String) (b : ByteArray) (s : Strategy (freshG k fs ts f l b))
    (t : Tape (freshG k fs ts f l b) s) : Tape (freshG k fs ts f l b) s :=
  placeOpt (f []) (freshY k ts f l b) (fun _ t => t)
    (fun fs' s' t' => placeCoin _
      (fun v => match slot L K pre (.fresh l) b 4 with
        | some (i, u) => freshVal (r i) u
        | none => v)
      (fun x s₃ t₃ => rec fs' ts (pre ++ [⟨.fresh l, b, List.ofFn x⟩])
        (fun log => f (⟨.fresh l, b, List.ofFn x⟩ :: log)) s₃ t₃) s' t')
    (takeCoin l fs) s t

/-- **At a `Hello`**: the sample unchanged. -/
noncomputable def placeHello {β : Type} (k : ℕ) (rec : PlaceRec β k) (fs : List (String × ℕ)) (ts : List OpenTree)
    (pre : List Verity.Release) (f : List Verity.Release → Game β) (spec : Flock.CoinTree.Spec) (ν : ByteArray)
    (s : Strategy (helloG k fs ts f spec ν)) (t : Tape (helloG k fs ts f spec ν) s) :
    Tape (helloG k fs ts f spec ν) s :=
  placeIte _ (helloX k fs ts f spec ν) (f []) s
    (fun s' t' => placeCoin _ id
      (fun tr s₃ t₃ => rec fs (ts ++ [⟨ν, spec, tr, fun _ => 0⟩])
        (pre ++ [⟨.hello spec, ν, bitsOfBytes (treeRoot ν spec tr ++ treeKey tr.1)⟩])
        (fun log => f (⟨.hello spec, ν, bitsOfBytes (treeRoot ν spec tr ++ treeKey tr.1)⟩ :: log)) s₃ t₃) s' t')
    (fun _ t => t) t

/-- **At an opening's round**: the round picked among those whose first `n` coins are the round's coin's next `n`
elements, where the locator puts it and they fit. -/
noncomputable def placeRound {β : Type} (L : Locate) {K : ℕ} (r : Fin K → FlockDeploy.Coin) (k : ℕ)
    (rec : PlaceRec β k) (fs : List (String × ℕ)) (ts : List OpenTree) (pre : List Verity.Release)
    (f : List Verity.Release → Game β) (i sn n : ℕ) (b : ByteArray) (T : OpenTree)
    (p : Fin T.spec.streams.size × Fin (2 ^ T.spec.roundsLog)) (s : Strategy (openP k fs ts f i sn n b T p))
    (t : Tape (openP k fs ts f i sn n b T p) s) : Tape (openP k fs ts f i sn n b T p) s :=
  placeCoin _
    (fun dr => match slot L K pre (.opening i sn n) b n with
      | some (j, u) => openW n (openBits (r j) u n) dr
      | none => dr)
    (fun dr s₅ t₅ => rec fs (ts.set i (T.bump p.1)) (pre ++ [⟨.opening i sn n, b, TreeDraw.opened dr.1 n⟩])
      (fun log => f (⟨.opening i sn n, b, TreeDraw.opened dr.1 n⟩ :: log)) s₅ t₅) s t

/-- **At an opening**: the round written by `placeRound`, where the device opens one. -/
noncomputable def placeOpen {β : Type} (L : Locate) {K : ℕ} (r : Fin K → FlockDeploy.Coin) (k : ℕ)
    (rec : PlaceRec β k) (fs : List (String × ℕ)) (ts : List OpenTree) (pre : List Verity.Release)
    (f : List Verity.Release → Game β) (i sn n : ℕ) (b : ByteArray) (s : Strategy (openG k fs ts f i sn n b))
    (t : Tape (openG k fs ts f i sn n b) s) : Tape (openG k fs ts f i sn n b) s :=
  placeIte _ (optGame ts[i]? (f []) (openT k fs ts f i sn n b)) (f []) s
    (fun s' t' => placeOpt (f []) (openT k fs ts f i sn n b) (fun _ t => t)
      (fun T s₃ t₃ => placeOpt (f []) (openP k fs ts f i sn n b T) (fun _ t => t)
        (fun p s₄ t₄ => placeRound L r k rec fs ts pre f i sn n b T p s₄ t₄) (T.next sn n) s₃ t₃)
      ts[i]? s' t')
    (fun _ t => t) t

/-- **The coin map along the device's releases**, from the releases `pre`, with the run's coins `r` at `K` rounds:
each fresh coin and each opening the locator puts on a round, written with the round's coin's next elements where they
fit (`slot`); every other node, and the play after the releases (`f`), unchanged. -/
noncomputable def placeK {β : Type} (L : Locate) {K : ℕ} (r : Fin K → FlockDeploy.Coin) :
    (k : ℕ) → (fs : List (String × ℕ)) → (ts : List OpenTree) → (pre : List Verity.Release) →
      (f : List Verity.Release → Game β) → (s : Strategy (relK k fs ts f)) →
        Tape (relK k fs ts f) s → Tape (relK k fs ts f) s
  | 0, _, _, _, _, _, t => t
  | k + 1, fs, ts, pre, f, ⟨m, s'⟩, t =>
    match m, s', t with
    | none, _, t => t
    | some (.fresh l, b), s', t =>
      placeFresh L r k (fun fs ts pre f s t => placeK L r k fs ts pre f s t) fs ts pre f l b s' t
    | some (.hello spec, ν), s', t =>
      placeHello k (fun fs ts pre f s t => placeK L r k fs ts pre f s t) fs ts pre f spec ν s' t
    | some (.opening i sn n, b), s', t =>
      placeOpen L r k (fun fs ts pre f s t => placeK L r k fs ts pre f s t) fs ts pre f i sn n b s' t

/-- **The period's game in the releases' continuation form**: the prover's request, then `relK` (`releases_bind`). -/
theorem game_eq (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) :
    V.game θ F c = .send ByteArray fun a =>
      relK (frames (V.families θ)) (V.families θ) [] (finish (V.signer θ) θ F c a) := by
  show play (V.families θ) (V.signer θ) θ F c = _
  unfold play
  congr 1
  funext a
  exact releases_bind _ _ _ _

/-- A map of the tapes of `g'` read at `g`, for `g = g'`. -/
noncomputable def along {β : Type} {g g' : Game β} (h : g = g') (P : (s : Strategy g') → Tape g' s → Tape g' s)
    (s : Strategy g) : Tape g s → Tape g s := by
  subst h
  exact P s

/-- A value of the strategies of `g'` read at `g`, for `g = g'`. -/
noncomputable def alongVal {β γ : Type} {g g' : Game β} (h : g = g') (G : Strategy g' → γ) (s : Strategy g) : γ := by
  subst h
  exact G s

/-- **The coin map at the period's game** (`Frame.place`): the run's coins `r` written into the tape `ω` at the releases
the locator `L` puts on their rounds, from the device's first release. -/
noncomputable def place (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Strategy (V.game θ F c)) (L : Locate) {K : ℕ} (r : Fin K → FlockDeploy.Coin) :
    Tape (V.game θ F c) σ → Tape (V.game θ F c) σ :=
  along (game_eq V θ F c)
    (fun s t => placeK L r (frames (V.families θ)) (V.families θ) [] [] (finish (V.signer θ) θ F c s.1) s.2 t) σ

/-- **The frame at the locator `L`**: the coin map `place`, and a prover `P` the strategy is read as. -/
noncomputable def frame (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Strategy (V.game θ F c)) (L : Locate) (P : Claims.Prover (TapeOf V θ F c σ)) : Frame V A x θ F c σ :=
  ⟨fun ω r => place V θ F c σ L r ω, P⟩

/-! ## What the device released for a round -/

/-- **The bits the log `log` released for round `i`**, after the releases `pre`: each release the locator puts on round
`i`, in order. -/
def devBits (L : Locate) (i : ℕ) : List Verity.Release → List Verity.Release → List Bool
  | _, [] => []
  | pre, rl :: log =>
    (if locAt L pre rl.frame rl.answers = some i then rl.coins else []) ++ devBits L i (pre ++ [rl]) log

/-- The number below `2^n` whose bit `j < n` is `g j`. -/
def natOfBits : ℕ → (ℕ → Bool) → ℕ
  | 0, _ => 0
  | n + 1, g => 2 * natOfBits n (fun j => g (j + 1)) + (g 0).toNat

/-- The element of GF(2^128) whose bit `j < 128` is `g j` (`F128.bit`'s order: `lo`'s bits, then `hi`'s). -/
def elemOfBits (g : ℕ → Bool) : Flock.F128 :=
  ⟨UInt64.ofNat (natOfBits 64 g), UInt64.ofNat (natOfBits 64 fun j => g (64 + j))⟩

/-- **The device's release for round `i` in a log** (`Reading.dev`): the coin whose element `j` is the bits
`128 j` to `128 j + 127` the releases the locator puts on round `i` carry, padded with zeros, and past a coin's 65536
elements cut off; none where it puts none. -/
def dev (L : Locate) (log : List Verity.Release) (i : ℕ) : Option FlockDeploy.Coin :=
  if usedOf L i [] log = 0 then none
  else some fun j => elemOfBits fun b => (devBits L i [] log).getD (128 * j.val + b) false

/-- **The elements of the releases the log `log` gives round `i` that fit in its coin**, after the releases `pre`: the
widths of the frames the locator puts on round `i` whose elements end within the coin's 65536. Once one does not fit,
none after it does, so they are the round's first. -/
def fitOf (L : Locate) (i : ℕ) : List Verity.Release → List Verity.Release → ℕ
  | _, [] => 0
  | pre, rl :: log =>
    (if locAt L pre rl.frame rl.answers = some i ∧ usedOf L i [] pre + widthOf rl.frame ≤ 65536 then
      widthOf rl.frame else 0) + fitOf L i (pre ++ [rl]) log

/-! ## The hiding term -/

/-- `n` elements' bits laid end to end, bit `t % 128` of element `t / 128` (`openBits`' order). -/
def bitsOf {n : ℕ} (e : Fin n → Fin 128 → Bool) : List Bool :=
  List.ofFn fun t : Fin (128 * n) => e ⟨t.val / 128, by have := t.isLt; omega⟩ ⟨t.val % 128, Nat.mod_lt _ (by omega)⟩

/-- The hiding term at an `ite` node. -/
noncomputable def gapIte {β : Type} {c : Prop} :
    (dc : Decidable c) → (G₁ G₂ : Game β) → (s : Strategy (@ite _ c dc G₁ G₂)) →
      ((s : Strategy G₁) → ℝ≥0∞) → ((s : Strategy G₂) → ℝ≥0∞) → ℝ≥0∞
  | .isTrue _, _, _, s, gX, _ => gX s
  | .isFalse _, _, _, s, _, gY => gY s

/-- The hiding term at an `optGame` node. -/
noncomputable def gapOpt {β γ : Type} (n : Game β) (y : γ → Game β) (gn : Strategy n → ℝ≥0∞)
    (gy : (a : γ) → Strategy (y a) → ℝ≥0∞) : (o : Option γ) → Strategy (optGame o n y) → ℝ≥0∞
  | none, s => gn s
  | some a, s => gy a s

/-- What the hiding term is along `relK k`, from any layout, trees, releases before and continuation. -/
abbrev GapRec (β : Type) (k : ℕ) : Type 1 :=
  (fs : List (String × ℕ)) → (ts : List OpenTree) → (pre : List Verity.Release) → (f : List Verity.Release → Game β) →
    Strategy (relK k fs ts f) → ℝ≥0∞

/-- The hiding term at a fresh coin: the term after it, averaged over its bits. -/
noncomputable def gapFresh {β : Type} (k : ℕ) (rec : GapRec β k) (fs : List (String × ℕ)) (ts : List OpenTree)
    (pre : List Verity.Release) (f : List Verity.Release → Game β) (l : String) (b : ByteArray)
    (s : Strategy (freshG k fs ts f l b)) : ℝ≥0∞ :=
  gapOpt (f []) (freshY k ts f l b) (fun _ => 0)
    (fun fs' s' => avg fun x : Fin 512 → Bool => rec fs' ts (pre ++ [⟨.fresh l, b, List.ofFn x⟩])
      (fun log => f (⟨.fresh l, b, List.ofFn x⟩ :: log)) (s' x))
    (takeCoin l fs) s

/-- The hiding term at a `Hello`: the term after it, averaged over the sample. -/
noncomputable def gapHello {β : Type} (k : ℕ) (rec : GapRec β k) (fs : List (String × ℕ)) (ts : List OpenTree)
    (pre : List Verity.Release) (f : List Verity.Release → Game β) (spec : Flock.CoinTree.Spec) (ν : ByteArray)
    (s : Strategy (helloG k fs ts f spec ν)) : ℝ≥0∞ :=
  gapIte _ (helloX k fs ts f spec ν) (f []) s
    (fun s' => avg fun tr : TreeSample spec => rec fs (ts ++ [⟨ν, spec, tr, fun _ => 0⟩])
      (pre ++ [⟨.hello spec, ν, bitsOfBytes (treeRoot ν spec tr ++ treeKey tr.1)⟩])
      (fun log => f (⟨.hello spec, ν, bitsOfBytes (treeRoot ν spec tr ++ treeKey tr.1)⟩ :: log)) (s' tr))
    (fun _ => 0)

open Classical in
/-- **The hiding term at an opening's round**: where the locator puts it and its elements fit, the written round's
shortfall below uniform over the node's rounds, plus the chance its class at the run's uniform `n` elements is empty,
plus the term after the written round averaged over the elements and the node's round; elsewhere the term after it,
averaged. -/
noncomputable def gapRound {β : Type} (L : Locate) (K : ℕ) (k : ℕ) (rec : GapRec β k) (fs : List (String × ℕ))
    (ts : List OpenTree) (pre : List Verity.Release) (f : List Verity.Release → Game β) (i sn n : ℕ) (b : ByteArray)
    (T : OpenTree) (p : Fin T.spec.streams.size × Fin (2 ^ T.spec.roundsLog))
    (s : Strategy (openP k fs ts f i sn n b T p)) : ℝ≥0∞ :=
  haveI : Nonempty (TreeOpenable T.nonce T.spec T.sample p.1 p.2) := ⟨⟨T.sample.2 p.1 p.2, rfl⟩⟩
  let next := fun dr : TreeOpenable T.nonce T.spec T.sample p.1 p.2 =>
    rec fs (ts.set i (T.bump p.1)) (pre ++ [⟨.opening i sn n, b, TreeDraw.opened dr.1 n⟩])
      (fun log => f (⟨.opening i sn n, b, TreeDraw.opened dr.1 n⟩ :: log)) (s dr)
  match slot L K pre (.opening i sn n) b n with
  | some _ =>
    shortfall (fun e : Fin n → Fin 128 → Bool =>
        openW (t := T.sample) (s := p.1) (r := p.2) (ν := T.nonce) n (bitsOf e)) +
      avg (fun e : Fin n → Fin 128 → Bool =>
        if (Finset.univ.filter fun e' : TreeOpenable T.nonce T.spec T.sample p.1 p.2 =>
            TreeDraw.opened e'.1 n = bitsOf e).Nonempty then 0 else 1) +
      avg fun e : Fin n → Fin 128 → Bool => avg fun dr => next (openW n (bitsOf e) dr)
  | none => avg next

/-- The hiding term at an opening: `gapRound` where the device opens a round. -/
noncomputable def gapOpen {β : Type} (L : Locate) (K : ℕ) (k : ℕ) (rec : GapRec β k) (fs : List (String × ℕ))
    (ts : List OpenTree) (pre : List Verity.Release) (f : List Verity.Release → Game β) (i sn n : ℕ) (b : ByteArray)
    (s : Strategy (openG k fs ts f i sn n b)) : ℝ≥0∞ :=
  gapIte _ (optGame ts[i]? (f []) (openT k fs ts f i sn n b)) (f []) s
    (fun s' => gapOpt (f []) (openT k fs ts f i sn n b) (fun _ => 0)
      (fun T s₃ => gapOpt (f []) (openP k fs ts f i sn n b T) (fun _ => 0)
        (fun p s₄ => gapRound L K k rec fs ts pre f i sn n b T p s₄) (T.next sn n) s₃)
      ts[i]? s')
    (fun _ => 0)

/-- **The hiding term along the device's releases**, from the releases `pre`, along the frame's play: the sum, over the
written openings, of each one's `gapRound` term; fresh coins and `Hello`s add nothing of their own. -/
noncomputable def gapK {β : Type} (L : Locate) (K : ℕ) :
    (k : ℕ) → (fs : List (String × ℕ)) → (ts : List OpenTree) → (pre : List Verity.Release) →
      (f : List Verity.Release → Game β) → Strategy (relK k fs ts f) → ℝ≥0∞
  | 0, _, _, _, _, _ => 0
  | k + 1, fs, ts, pre, f, ⟨m, s'⟩ =>
    match m, s' with
    | none, _ => 0
    | some (.fresh l, b), s' => gapFresh k (fun fs ts pre f s => gapK L K k fs ts pre f s) fs ts pre f l b s'
    | some (.hello spec, ν), s' => gapHello k (fun fs ts pre f s => gapK L K k fs ts pre f s) fs ts pre f spec ν s'
    | some (.opening i sn n, b), s' =>
      gapOpen L K k (fun fs ts pre f s => gapK L K k fs ts pre f s) fs ts pre f i sn n b s'

/-- **The coin trees' hiding term at the period's game** (`hide`): `gapK` from the device's first release, at the
strategy's request. -/
noncomputable def gapOf (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Strategy (V.game θ F c)) (L : Locate) (K : ℕ) : ℝ≥0∞ :=
  alongVal (game_eq V θ F c)
    (fun s => gapK L K (frames (V.families θ)) (V.families θ) [] [] (finish (V.signer θ) θ F c s.1) s.2) σ

/-! ## The sampling design -/

/-- **The sampling design at the locator `L`** (`CoinMapLaw`'s first clause): the period's play is at most as likely, at every event,
as the frame's play at a uniform tape and uniform run coins, plus the hiding term. -/
def LawAt (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Strategy (V.game θ F c)) (L : Locate) (P : Claims.Prover (TapeOf V θ F c σ)) : Prop :=
  ∀ E : Verity.Exchange → Prop, Verity.Game.prob E (V.game θ F c) σ ≤
    ENNReal.ofReal (Claims.Pr fun s : TapeOf V θ F c σ × Coins A x => E ((frame V A x θ F c σ L P).out s.1 s.2)) +
      gapOf V θ F c σ L (A.L.V.k x)

/-- **The releases carry the run's coins' elements** (the coin map's agreement): at the play `o` and the run's coins `r`,
the coin `dev L` reads for each read round has the run's coin's elements below what the locator's releases for the round
that fit carry (`fitOf`). -/
def ElemAgrees (L : Locate) (A : Live₂) (x : Msg) (reads : ℕ → Prop) (o : Verity.Exchange) (r : Coins A x) : Prop :=
  ∀ i : Fin (A.L.V.k x), reads i.val → ∀ c, dev L o.log i.val = some c →
    ∀ j : Fin 65536, j.val < fitOf L i.val [] o.log → c j = r i j

/-- **The coin map's law holds** at every vocabulary, statement, strategy, locator, prover and set of read rounds:
`CoinMapLaw` at the frame of `L` and the agreement `ElemAgrees`, with `hide` the hiding term (`gapOf`: the written
openings' shortfalls and empty classes along the frame's play). -/
def CoinMapHolds : Prop :=
  ∀ (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Strategy (V.game θ F c)) (L : Locate) (P : Claims.Prover (TapeOf V θ F c σ)) (reads : ℕ → Prop),
    CoinMapLaw V A x θ F c σ (frame V A x θ F c σ L P) (ElemAgrees L A x reads) (gapOf V θ F c σ L (A.L.V.k x))

/-! ## Probes -/

theorem placeK_zero {β : Type} (L : Locate) {K : ℕ} (r : Fin K → FlockDeploy.Coin) (fs : List (String × ℕ))
    (ts : List OpenTree) (pre : List Verity.Release) (f : List Verity.Release → Game β) (s : Strategy (relK 0 fs ts f))
    (t : Tape (relK 0 fs ts f) s) : placeK L r 0 fs ts pre f s t = t := rfl

theorem placeK_close {β : Type} (L : Locate) {K : ℕ} (r : Fin K → FlockDeploy.Coin) (k : ℕ)
    (fs : List (String × ℕ)) (ts : List OpenTree) (pre : List Verity.Release) (f : List Verity.Release → Game β)
    (s' : Strategy (f [])) (t : Tape (f []) s') :
    placeK L r (k + 1) fs ts pre f ⟨none, s'⟩ t = t := rfl

theorem placeK_fresh {β : Type} (L : Locate) {K : ℕ} (r : Fin K → FlockDeploy.Coin) (k : ℕ)
    (fs : List (String × ℕ)) (ts : List OpenTree) (pre : List Verity.Release) (f : List Verity.Release → Game β)
    (l : String) (b : ByteArray) (s' : Strategy (freshG k fs ts f l b)) (t : Tape (freshG k fs ts f l b) s') :
    placeK L r (k + 1) fs ts pre f ⟨some (.fresh l, b), s'⟩ t =
      placeFresh L r k (fun fs ts pre f s t => placeK L r k fs ts pre f s t) fs ts pre f l b s' t := rfl

theorem placeK_hello {β : Type} (L : Locate) {K : ℕ} (r : Fin K → FlockDeploy.Coin) (k : ℕ)
    (fs : List (String × ℕ)) (ts : List OpenTree) (pre : List Verity.Release) (f : List Verity.Release → Game β)
    (spec : Flock.CoinTree.Spec) (ν : ByteArray) (s' : Strategy (helloG k fs ts f spec ν))
    (t : Tape (helloG k fs ts f spec ν) s') :
    placeK L r (k + 1) fs ts pre f ⟨some (.hello spec, ν), s'⟩ t =
      placeHello k (fun fs ts pre f s t => placeK L r k fs ts pre f s t) fs ts pre f spec ν s' t := rfl

theorem placeK_open {β : Type} (L : Locate) {K : ℕ} (r : Fin K → FlockDeploy.Coin) (k : ℕ)
    (fs : List (String × ℕ)) (ts : List OpenTree) (pre : List Verity.Release) (f : List Verity.Release → Game β)
    (i sn n : ℕ) (b : ByteArray) (s' : Strategy (openG k fs ts f i sn n b)) (t : Tape (openG k fs ts f i sn n b) s') :
    placeK L r (k + 1) fs ts pre f ⟨some (.opening i sn n, b), s'⟩ t =
      placeOpen L r k (fun fs ts pre f s t => placeK L r k fs ts pre f s t) fs ts pre f i sn n b s' t := rfl

theorem gapK_open {β : Type} (L : Locate) (K k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree)
    (pre : List Verity.Release) (f : List Verity.Release → Game β) (i sn n : ℕ) (b : ByteArray)
    (s' : Strategy (openG k fs ts f i sn n b)) :
    gapK L K (k + 1) fs ts pre f ⟨some (.opening i sn n, b), s'⟩ =
      gapOpen L K k (fun fs ts pre f s => gapK L K k fs ts pre f s) fs ts pre f i sn n b s' := rfl

theorem dev_nil (L : Locate) (i : ℕ) : dev L [] i = none := by
  simp [dev, usedOf]

end Verity.SampledTwo.CoinMap
