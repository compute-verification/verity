import Proofs.Flock.Claims.ZK.CutSession
import Proofs.Flock.Claims.ZK.FirstRefusal
import Proofs.Flock.Claims.ZK.Round0
import Proofs.Flock.Claims.ZK.SaltProver

/-!
# The deployed masked session's round plan

The coin plan, the round encodings and the last message of `flock-circuit --zk` at the executable's rounds, as
`RoundPlan.stop` takes them, with levels ≥ 1's prover `ofSalts PcExe`:

* `restAt`, `czRaw`, `czExe`: each table's coins, every field read at the session's round for it (rep `q`'s PIOP rounds
  from `repBase x q`, its Ligerito and inner rounds from `repSlot x q`, the link points at round `0`); `czExe` is
  `withPoints czRaw`, so `ZKR1`'s `OuterReads`, `CoinReads` and `PointReads` hold at it.
* `repSrcCut`, `tabSrcCut`: what `FlockComplete.sessionCut` reads from a cut: each rep's masked values by pad index, its
  switched vectors (`ŝ` of `ab`, `c`, then the region claims'), `y₁'`, `ȳ`, levels ≥ 1's salts read back from the
  rep's levels ≥ 1 value (`blobSalt`), the triple's openings, `τ`'s commitment, `Y` in the pads layout's order, and
  level 0's and the pads' leaves by tree position (`leafAt`, `padLeafCut`).
* `viewCut`, `tabSrcView`: a view as a cut with every guarded message sent, and the source of the last message, which
  also reads level 0's opened rows and salts, `τ`, its salt and the pads' opened columns and salts by tree position.
* `encExe`: round `0` is `enc0`; round `n ≥ 1` is body `n` of `sessionCut` stopped at round `n`, tagged `0`, or `[]`
  where it stops before. `finExe`: `Final.encode` of `sessionCut`'s record and proof files on the view, `[]` where it
  stops.
-/

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

namespace Masked

/-! ## The coins -/

/-- The link points' coordinate count of `x`'s session. -/
def mPtsOf (x : Msg) : ℕ :=
  match FlockDeploy.setupOf true x with
  | some (_, spec, _) => spec.mPts
  | none => 0

/-- **Element `i` of the coin at rep `q`'s round `f sch` after its PIOP rounds** (`repSlot`). -/
def ligAt {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (q : ℕ) (f : _root_.Flock.Schedule → ℕ) (i : ℕ) :
    _root_.Flock.F128 :=
  slotAt c (repSlot x q f) i

/-- **The `F256` coin at round `k`**: its elements `0` and `1` (`sampleKP`). -/
def kAt {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (k : ℕ) : _root_.Flock.F256 := ⟨slotAt c k 0, slotAt c k 1⟩

/-- **Ligerito's fold round `i` round `e` of rep `q`**, after its PIOP rounds: level 0's `k₀ + 11` rounds, then eleven a
fold round. -/
def foldOff (i e : ℕ) (sch : _root_.Flock.Schedule) : ℕ := sch.k0 + 11 + 11 * i + e

/-- **Fold round `i`'s query, consistency and batching rounds**: `4`, `5`, `6` after its challenges at the last fold round,
`8`, `9`, `10` (after the next level's two out-of-domain samples) before. -/
def lastOff (i a b : ℕ) (sch : _root_.Flock.Schedule) : ℕ :=
  foldOff i (if i + 1 = sch.levels.size - 1 then a else b) sch

/-- **The inner proof's round `e` of rep `q`**: after Ligerito's `k₀ + 7 + 11R` rounds. -/
def innerOff (e : ℕ) (sch : _root_.Flock.Schedule) : ℕ := sch.k0 + 7 + 11 * (sch.levels.size - 1) + e

/-- **Table `j`'s rep `r`'s level-0 query positions** at the opened coins: `positions` of level 0 at the query coins
(`queryP`'s first round, Ligerito's `k₀ + 8`). -/
noncomputable def q0At {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (j : Fin (jOf x)) (r : Fin 2) : Array ℕ :=
  _root_.Flock.positions (lv0 x j)
    ((Array.range (lv0 x j).q).map fun e => ligAt c (2 * j.val + r.val) (fun sch => sch.k0 + 8) e)

/-- **Table `j`'s other coins at the opened coins `c`**, each at the session's round for it. -/
noncomputable def restAt (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (j : Fin (jOf x)) :
    Composed.RestOf (zkTab x j).table (stAt x j) where
  link p i := slotAt c 0 (p.val * mPtsOf x + i.val)
  q0s r i := ⟨(q0At c j r)[i.val]! % 2 ^ (zkTab x j).table.sch.level₀.logLen, Nat.mod_lt _ (Nat.two_pow_pos _)⟩
  rSkip r i := slotAt c (repBase x (2 * j.val + r.val)) i.val
  rOuter r i := outerAt c j.val r.val i.val
  zc r := slotAt c (repBase x (2 * j.val + r.val) + 2) 0
  rho r i := slotAt c (repBase x (2 * j.val + r.val) + (3 + i.val)) 0
  lcAlpha r := slotAt c (repBase x (2 * j.val + r.val) + (3 + ((stAt x j).m - 6))) 0
  lcBeta r := slotAt c (repBase x (2 * j.val + r.val) + (4 + ((stAt x j).m - 6))) 0
  ts r i := slotAt c (repBase x (2 * j.val + r.val) + (5 + ((stAt x j).m - 6) + i.val)) 0
  rPrime r := slotAt c (repBase x (2 * j.val + r.val) + (5 + ((stAt x j).m - 6) + ((stAt x j).c.kLog - 6))) 0
  rpp r k i := slotAt c (repBase x (2 * j.val + r.val) + (6 + ((stAt x j).m - 6) + ((stAt x j).c.kLog - 6) + k.val)) i.val
  rsGamma r k := slotAt c (repSlot x (2 * j.val + r.val) (fun _ => 0) - 1) k.val
  oodPt r _ i := ligAt c (2 * j.val + r.val) (fun _ => 0) i.val
  ood r _ := ligAt c (2 * j.val + r.val) (fun _ => 1) 0
  lane r l := kAt c (repSlot x (2 * j.val + r.val) fun _ => 2 + l.val)
  ligZ r l t i := ligAt c (2 * j.val + r.val) (fun sch => sch.k0 + 4 + 11 * l.val + 2 * t.val) i.val
  ligZB r l t := ligAt c (2 * j.val + r.val) (fun sch => sch.k0 + 5 + 11 * l.val + 2 * t.val) 0
  ligAlpha0 r i := ligAt c (2 * j.val + r.val) (fun sch => sch.k0 + 9) i.val
  ligB0 r := ligAt c (2 * j.val + r.val) (fun sch => sch.k0 + 10) 0
  ligFold r i f := kAt c (repSlot x (2 * j.val + r.val) (foldOff i.val f.val))
  ligQ r i e := ligAt c (2 * j.val + r.val) (lastOff i.val 4 8) e.val
  ligAlpha r i e := ligAt c (2 * j.val + r.val) (lastOff i.val 5 9) e.val
  ligB r i := ligAt c (2 * j.val + r.val) (lastOff i.val 6 10) 0
  zeta r := ligAt c (2 * j.val + r.val) (innerOff 0) 0
  gammas r g := ligAt c (2 * j.val + r.val) (innerOff 1) g.val
  qP r e := ligAt c (2 * j.val + r.val) (innerOff 3) e.val

/-- **Each table's coins at the opened coins**, every field at its round. -/
noncomputable def czRaw (P : Composed.LigProver) (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (j : Fin (jOf x)) :
    Composed.CoinsZK (tabC P x j).Rest where
  beta r := betaAt c j.val r.val
  rho r := rhoAt c j.val r.val
  rest := (restAt x c j : Composed.RestOf (zkTab x j).table (stAt x j))

/-- **The deployed session's coin plan**: `czRaw` through `withPoints`. -/
noncomputable def czExe (P : Composed.LigProver) :
    (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (tabC P x j).Rest :=
  withPoints (czRaw P)

/-! ## The sources -/

/-- A `GF256` element as the executable's value. -/
def kVal (a : 𝕂) : _root_.Flock.F256 := a

/-- **`Y` at index `i` of the pads layout** (`innerP`'s `y`): the exposed pads' `Y₁` first, then `h_ab`, `h_x`, `h_y`,
`h_w` (`PadIdx`'s `inr 2, 0, 1, 3`, `Composed.padIdx`), then the padding's `Y₂`. -/
def yAt {nE nPad : ℕ} (Y1 : PadIdx nE → 𝔽) (Y2 : Fin nPad → 𝔽) (i : Fin (nE + 4 + nPad)) : _root_.Flock.F128 :=
  if h : i.val < nE then FlockLevel3.GF128.val (Y1 (.inl ⟨i.val, h⟩))
  else if h' : i.val < nE + 4 then
    FlockLevel3.GF128.val (Y1 (.inr ((![2, 0, 1, 3] : Fin 4 → Fin 4) ⟨i.val - nE, by omega⟩)))
  else FlockLevel3.GF128.val (Y2 ⟨i.val - (nE + 4), by have := i.isLt; omega⟩)

/-- **Table `j`'s rep `r` pads leaf at tree position `q`** of a cut at the coins `cz`: the cut's opened or hidden pads
leaf the coins' tree places at `q` (`Coined.sP`), and the empty string past the tree. -/
noncomputable def padLeafCut (P : Composed.LigProver) (x : Msg) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (cut : ∀ j, ((tabC P x j).T (cz j)).tab.Cut) (j : Fin (jOf x)) (r : Fin 2) (q : ℕ) : ByteArray :=
  if hq : q < (tabC P x j).z.nQ + (tabC P x j).z.nPU then
    match (((tabC P x j).T (cz j)).sP r).symm ⟨q, hq⟩ with
    | .inl a => ((cut j).leavesP r).1 a
    | .inr u => ((cut j).leavesP r).2 u
  else ByteArray.empty

/-- **What table `j`'s rep `r` messages read from a cut**: claim `k`'s switched vector is masked claim `2r + k`'s `ŝ`
for `k < 2` and region claim `k − 2`'s after; levels ≥ 1's salts are read back from the rep's levels ≥ 1 value; a
guarded message not sent reads as empty. The final message's fields are left empty. -/
noncomputable def repSrcCut (P : Composed.LigProver) (x : Msg) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (cut : ∀ j, ((tabC P x j).T (cz j)).tab.Cut) (j : Fin (jOf x)) (r : Fin 2) : FlockComplete.RepSrc where
  masked n := if h : n < ((tabC P x j).T (cz j)).tab.nE then FlockLevel3.GF128.val ((cut j).masked r ⟨n, h⟩) else 0
  sv k :=
    if k < 2 then
      if h : 2 * r.val + k < ((tabC P x j).T (cz j)).tab.nC then
        match (cut j).sHat ⟨2 * r.val + k, h⟩ with
        | some s => Array.ofFn fun v => FlockLevel3.GF128.val (s v)
        | none => #[]
      else #[]
    else if h : k - 2 < ((tabC P x j).T (cz j)).tab.nR then
      Array.ofFn fun v => FlockLevel3.GF128.val ((cut j).sReg ⟨k - 2, h⟩ v)
    else #[]
  y1 := match (cut j).y1 r with
    | some f => Array.ofFn fun c => kVal (f c)
    | none => #[]
  ybar := Array.ofFn fun i => kVal ((cut j).ybar r i)
  hiSalt := blobSalt (zkTab x j).table.sch.levels
    ((((cut j).lig r : Option Composed.VLOf)).map Prod.snd |>.getD ByteArray.empty)
  rhoIn := FlockLevel3.GF128.val ((cut j).inner r).1
  sigmaIn := FlockLevel3.GF128.val ((cut j).inner r).2
  tauCom := (cut j).tauCom r
  y := match (cut j).Y1 r with
    | some Y1 => Array.ofFn (yAt Y1 ((cut j).Y2 r))
    | none => #[]
  padLeaf := padLeafCut P x cz cut j r
  tau := 0
  tauSalt := ByteArray.empty
  ch _ := 0
  cm _ := 0
  padSalt _ := ByteArray.empty

/-- **What table `i`'s messages read from a cut**: level 0's leaves by tree position and each rep's source; the default
past the session's tables. -/
noncomputable def tabSrcCut (P : Composed.LigProver) (x : Msg) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (cut : ∀ j, ((tabC P x j).T (cz j)).tab.Cut) (i : ℕ) : FlockComplete.TabSrc :=
  if h : i < jOf x then
    { leaf0 := leafAt P x cz cut ⟨i, h⟩
      row0 := fun _ => #[]
      salt0 := fun _ => ByteArray.empty
      rep := fun r => if hr : r < 2 then repSrcCut P x cz cut ⟨i, h⟩ ⟨r, hr⟩ else default }
  else default

/-- **A view as a cut** with every guarded message sent. -/
def viewCut {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] {T : Table F K W Salt Seed D Hi VL} (v : T.View) :
    T.Cut where
  sHat k := some (v.sHat k)
  sReg := v.sReg
  masked := v.masked
  y1 r := some (v.y1 r)
  ybar := v.ybar
  lig r := some (v.lig r)
  inner := v.inner
  Y1 r := some (v.Y1 r)
  Y2 := v.Y2
  leaves0 := v.leaves0
  tauCom := v.tauCom
  leavesP := v.leavesP

/-- **A value of level 0's opened positions at tree position `q`**, `d` elsewhere (`Coined.s0`). -/
noncomputable def at0 (P : Composed.LigProver) (x : Msg) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (j : Fin (jOf x)) {α : Type} (d : α) (f : Fin (tabC P x j).z.nS → α) (q : ℕ) : α :=
  if hq : q < (tabC P x j).z.nS + (tabC P x j).z.nU then
    match ((tabC P x j).T (cz j)).s0.symm ⟨q, hq⟩ with
    | .inl a => f a
    | .inr _ => d
  else d

/-- **A value of rep `r`'s opened pads positions at tree position `q`**, `d` elsewhere (`Coined.sP`). -/
noncomputable def atP (P : Composed.LigProver) (x : Msg) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (j : Fin (jOf x)) (r : Fin 2) {α : Type} (d : α) (f : Fin (tabC P x j).z.nQ → α) (q : ℕ) : α :=
  if hq : q < (tabC P x j).z.nQ + (tabC P x j).z.nPU then
    match (((tabC P x j).T (cz j)).sP r).symm ⟨q, hq⟩ with
    | .inl a => f a
    | .inr _ => d
  else d

/-- **What table `i`'s messages read from the tables' views**: `tabSrcCut` of the views as cuts, with level 0's opened
rows (`ZkView.row0`) and salts by tree position, and each rep's `τ`, its salt and the pads' opened columns and salts by
tree position; empty at the positions nobody opens. -/
noncomputable def tabSrcView (P : Composed.LigProver) (x : Msg) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (v : ∀ j, ((tabC P x j).T (cz j)).tab.View) (i : ℕ) : FlockComplete.TabSrc :=
  if h : i < jOf x then
    { tabSrcCut P x cz (fun j => viewCut (v j)) i with
      row0 := at0 P x cz ⟨i, h⟩ #[] fun a => ZkView.row0 ((v ⟨i, h⟩).rows a, (v ⟨i, h⟩).xrows a)
      salt0 := at0 P x cz ⟨i, h⟩ ByteArray.empty fun a => ZkView.saltBytes ((v ⟨i, h⟩).salts0 a)
      rep := fun r => if hr : r < 2 then
        { (tabSrcCut P x cz (fun j => viewCut (v j)) i).rep r with
          tau := FlockLevel3.GF128.val ((v ⟨i, h⟩).tau ⟨r, hr⟩)
          tauSalt := ZkView.saltBytes ((v ⟨i, h⟩).ytau ⟨r, hr⟩)
          ch := atP P x cz ⟨i, h⟩ ⟨r, hr⟩ 0 fun a => FlockLevel3.GF128.val ((v ⟨i, h⟩).ch ⟨r, hr⟩ a)
          cm := atP P x cz ⟨i, h⟩ ⟨r, hr⟩ 0 fun a => FlockLevel3.GF128.val ((v ⟨i, h⟩).cm ⟨r, hr⟩ a)
          padSalt := atP P x cz ⟨i, h⟩ ⟨r, hr⟩ ByteArray.empty fun a => ZkView.saltBytes ((v ⟨i, h⟩).saltsP ⟨r, hr⟩ a) }
        else default }
  else default

/-! ## The round encodings and the last message -/

/-- **`sessionCut` at `x`'s session**, answered by the opened coins `c` and stopped right after round `n`'s send, on the
sources `src`. -/
noncomputable def runCut (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ) (spec : _root_.Flock.SessionSpec)
    (src : ℕ → FlockComplete.TabSrc) : Except String (String × Array ByteArray) × Array Msg :=
  (FlockComplete.sessionCut (FlockComplete.asksCut (fun k i => slotAt c k i) n)
    (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) spec (FlockDeploy.tabsOf (stsOf x))
    (FlockComplete.MaskRank.masksOf (stsOf x)) src).go #[]

/-- **The deployed session's round encoding**: round `0` is `enc0`; round `n ≥ 1` is `sessionCut`'s body `n` on the cut,
stopped right after it, tagged `0`, or `[]` where the session stops before it. -/
noncomputable def encExe (P : Composed.LigProver) (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ)
    (cut : ∀ j, ((tabC P x j).T (czExe P x c j)).tab.Cut) : Msg :=
  if n = 0 then enc0 P x (czExe P x c) cut else
  match FlockDeploy.setupOf true x with
  | none => []
  | some (_, spec, _) =>
    if n < (runCut x c n spec (tabSrcCut P x (czExe P x c) cut)).2.size then
      0 :: (runCut x c n spec (tabSrcCut P x (czExe P x c) cut)).2[n]!
    else []

/-- **The deployed session's last message**: `Final.encode` of `sessionCut`'s record and proof files on the tables'
views, answered by every opened coin; `[]` where it stops (`ZkTables.prog`). -/
noncomputable def finExe (P : Composed.LigProver) (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin)
    (v : ∀ j, ((tabC P x j).T (czExe P x c j)).tab.View) : Msg :=
  match FlockDeploy.setupOf true x with
  | none => []
  | some (_, spec, _) =>
    match (runCut x c (refK x) spec (tabSrcView P x (czExe P x c) v)).1 with
    | .ok (record, proofs) => FlockDeploy.Final.encode ⟨x, record, proofs⟩
    | .error _ => []

/-! ## What holds by construction -/

theorem encExe_zero (P : Composed.LigProver) (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin)
    (cut : ∀ j, ((tabC P x j).T (czExe P x c j)).tab.Cut) : encExe P x c 0 cut = enc0 P x (czExe P x c) cut :=
  ite_eq_left rfl

theorem outerReads_czExe (P : Composed.LigProver) : OuterReads (czExe P) := outerReads_withPoints _

theorem coinReads_czExe (P : Composed.LigProver) : CoinReads (czExe P) := coinReads_withPoints _

theorem pointReads_czExe (P : Composed.LigProver) : PointReads (czExe P) := pointReads_withPoints _

end Masked

end Claims.ZKMal
