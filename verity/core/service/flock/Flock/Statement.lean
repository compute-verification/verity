/-
PROTOCOL.md §16.1, §16.2, §16.5: the circuit statement, set up from its `Tags`, its row leaf (`RowLeaf`), the pinned
circuit file and the public file: Δ, the regions and their claims, the lincheck fold, the statement digest, Σ and `Hello`.
Nothing here evaluates the computation: the region values are the layout's constants, the public digests and outputs, and
hashes of pinned padding constants.
-/
import Flock.RowLeaf
import Flock.Piop
import Flock.Merkle
import Flock.Record
import Flock.TmplSplit

namespace Flock

open Lean (Json)

namespace Circuit

/-- The pinned constant column: the constant of the first slot of the first range that is not the mask, a column of the
block. -/
def pin (c : Circuit) (leaf : RowLeaf) : V Nat := do
  let p ← match c.ranges.find? (·.kind != Kind.mask) with
    | some r => match r.kind with
      | .comp => pure (r.start + leaf.comp.spec.zConst)
      | .net i => pure (r.start + (c.nets.getD i default).2.constPos)
      | .mask => throw "no pin"
    | none => throw "circuit: no range with a constant"
  if p ≥ 2 ^ c.kLog then throw "the pinned constant column is outside the block"
  return p

/-- Δ (§16.1), `(row, col)` pairs XORed into A and into B, in the order the statement digest binds: per range and slot,
the slot's constant wired to the pin, then the row leaf's own entries for its slots; then every unit's leaf bits copied from
the message bits that hash them (a zero leaf from a forced-zero row); then every META wire. -/
def delta (c : Circuit) (leaf : RowLeaf) (pin : Nat) : V (Array (Nat × Nat) × Array (Nat × Nat)) := do
  let mut da : Array (Nat × Nat) := #[]
  let mut db : Array (Nat × Nat) := #[]
  for r in c.ranges do
    for s in [0:r.count] do
      let base := r.slot s
      let konst := match r.kind with
        | .comp => some (base + leaf.comp.spec.zConst)
        | .net i => some (base + (c.nets.getD i default).2.constPos)
        | .mask => none
      if let some k := konst then
        if k != pin then
          da := da.push (k, k) |>.push (k, pin)
          db := db.push (k, k) |>.push (k, pin)
      if r.kind == Kind.comp then
        let (la, lb) ← leaf.slotDelta c r s
        da := da ++ la
        db := db ++ lb
  let unit := c.unit
  let ur := c.unitRange
  let leafCols := (portCols #[unit.inGroups.getD 0 default]).map (·.1)
  let starts := (c.ports.foldl (fun (acc : Array Nat × Nat) p => (acc.1.push acc.2, acc.2 + p.words)) (#[], 0)).1
  for k in [0:ur.count] do
    let (g, u) := (k / c.upv, k % c.upv)
    let base := ur.slot k
    let li := c.leavesIn.getD u #[]
    for j in [0:li.size] do
      let lf := li[j]!
      for t in [0:16] do
        let src ← if lf < 0 then pure (base + unit.useful) else do
          let x := lf.toNat
          let some p := (List.range starts.size).reverse.find? (fun q => starts[q]! ≤ x) | throw "a leaf outside the ports"
          pure ((← leaf.messageCol c g p (2 * (x - starts[p]!))) + t)
        let i := base + leafCols.getD j 0 + t
        da := da.push (i, i) |>.push (i, src)
        db := db.push (i, i) |>.push (i, src)
  for g in [0:c.g] do
    for w in c.wires do
      let dc := (portCols (c.nets.getD w.dst.1 default).2.inGroups).getD w.dst.2.2 default
      let sc := (portCols (c.nets.getD w.src.1 default).2.outGroups).getD w.src.2.2 default
      let dst := c.slotCol w.dst.1 g w.dst.2.1 + dc.1
      let src := c.slotCol w.src.1 g w.src.2.1 + sc.1
      for t in [0:dc.2] do
        da := da.push (dst + t, dst + t) |>.push (dst + t, src + t)
        db := db.push (dst + t, dst + t) |>.push (dst + t, src + t)
  return (da, db)

/-- The output region: every unit slot's returned-output words (the public outputs; pinned padding outputs for padding
instances). -/
def outRegion (c : Circuit) (pub : Public) : V Region := do
  let unit := c.unit
  let ur := c.unitRange
  let g0 := unit.outGroups.getD 0 default
  let nOut := c.outWords
  let bits := fun (lo n : Nat) => (Array.range n).map (lo + ·)
  mkRegion c.kLog "out" (bits 0 (7 + Nat.log2 g0.words) ++ bits ur.slotLog (Nat.log2 ur.count)) (ur.start + g0.col * WORD) fun b => Id.run do
    let mut out := ByteArray.empty
    for k in [0:ur.count] do
      let (g, u) := (k / c.upv, k % c.upv)
      let mut w : Array F128 := Array.replicate g0.words 0
      let mut col := 0
      let lo := c.leavesOut.getD u #[]
      for j in [0:lo.size] do
        let o := lo[j]!
        let v := match c.instOf pub.n b g with
          | some i => pub.outputs.getD (i * nOut + o) 0
          | none => c.dummyOutputs.getD o 0
        let wi := col / WORD
        let sh := col % WORD
        let x := w[wi]!
        w := w.set! wi (if sh < 64 then { x with lo := x.lo ||| (v.toUInt64 <<< sh.toUInt64) }
                        else { x with hi := x.hi ||| (v.toUInt64 <<< (sh - 64).toUInt64) })
        col := col + g0.bits.getD j 0
      for x in w do out := F128.pushBytes out x
    return out

end Circuit

/-- **The statement range the soundness bound covers** (the soundness package's `Stmt.InRange`): at most `2^27` witness
bits per block, 1024 public regions, and 64 coordinates per link point. `setup` and `setupH` refuse a statement outside
it, so the verifier accepts only statements the proved bound covers. -/
def checkInRange (kLog nRegions mPts : Nat) : V Unit :=
  if kLog ≤ 27 && nRegions ≤ 1024 && mPts ≤ 64 then pure ()
  else throw s!"statement range: k_log {kLog}, {nRegions} regions, m_pts {mPts}; the soundness bound covers k_log ≤ 27, at most 1024 regions and m_pts ≤ 64"

/-- `checkInRange` accepts exactly the soundness package's `Stmt.InRange`. -/
theorem checkInRange_ok (kLog nRegions mPts : Nat) :
    checkInRange kLog nRegions mPts = .ok () ↔ kLog ≤ 27 ∧ nRegions ≤ 1024 ∧ mPts ≤ 64 := by
  unfold checkInRange
  by_cases h : kLog ≤ 27 ∧ nRegions ≤ 1024 ∧ mPts ≤ 64
  · simp [h, pure, Except.pure]
  · simp only [h, iff_false]
    have : (decide (kLog ≤ 27) && decide (nRegions ≤ 1024) && decide (mPts ≤ 64)) = false := by
      simp only [Bool.and_eq_false_iff, decide_eq_false_iff_not]; omega
    simp [this, throw, throwThe, MonadExceptOf.throw]

/-- **Each region's bytes are its free bits, in every block**: the extra claims read block `b`'s bytes as the region's
`2^|free|` bits (`Stmt.regionValue`), so `setupH` and `setupHidden` refuse a statement whose regions carry bytes of
another size in any of the `2^nbl` blocks, before any proof is read. One pass over each region's bytes, as each
`regionValue` makes. -/
def checkRegionBytes (nbl : Nat) (regions : Array Region) : V Unit :=
  if regions.all fun reg => Nat.all (2 ^ nbl) fun b _ => (reg.bytes b).size * 8 == 2 ^ reg.free.size then pure ()
  else throw "a region's bytes are not its free bits in every block: the extra claims read 2^|free| bits per block"

theorem nat_all_iff (n : Nat) (p : Nat → Bool) : (Nat.all n fun b _ => p b) = true ↔ ∀ b < n, p b = true := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Nat.all_succ, Bool.and_eq_true, ih]
    constructor
    · intro h b hb
      rcases Nat.lt_succ_iff_lt_or_eq.1 hb with hb | hb
      · exact h.1 b hb
      · exact hb ▸ h.2
    · intro h
      exact ⟨fun b hb => h b (Nat.lt_succ_of_lt hb), h n (Nat.lt_succ_self n)⟩

/-- `checkRegionBytes` accepts exactly where every region's bytes are its free bits in every block. -/
theorem checkRegionBytes_ok (nbl : Nat) (regions : Array Region) :
    checkRegionBytes nbl regions = .ok () ↔
      ∀ reg ∈ regions.toList, ∀ b < 2 ^ nbl, (reg.bytes b).size * 8 = 2 ^ reg.free.size := by
  have hall : (regions.all fun reg => Nat.all (2 ^ nbl) fun b _ => (reg.bytes b).size * 8 == 2 ^ reg.free.size) = true ↔
      ∀ reg ∈ regions.toList, ∀ b < 2 ^ nbl, (reg.bytes b).size * 8 = 2 ^ reg.free.size := by
    rw [← Array.all_toList, List.all_eq_true]
    refine forall_congr' fun reg => imp_congr_right fun _ => ?_
    rw [nat_all_iff]
    simp only [beq_iff_eq]
  unfold checkRegionBytes
  split
  · rename_i h
    simp only [pure, Except.pure, true_iff]
    exact hall.1 h
  · rename_i h
    simp only [throw, throwThe, MonadExceptOf.throw, reduceCtorEq, false_iff]
    exact fun h' => h (hall.2 h')

/-- The coin source named in `Hello` (§7). -/
inductive Coins where
  | os
  | seed
  /-- the verifier's keyed coin tree (`Flock.CoinTree`): `Hello`'s `coins` is its spec, with the prover's nonce -/
  | tree (spec : CoinTree.Spec)
  deriving BEq, Inhabited

/-- The keyed coin tree the coins name, which `Hello`'s `coins` carries (`helloOf`); none for the OS's coins or a seed's. -/
def Coins.tree? : Coins → Option CoinTree.Spec
  | .tree s => some s
  | _ => none

/-- The tags a session on `coins` is verified under: on `os` coins, a statement whose prover names its coin mode
(`Tags.liveOs`) binds the live-OS identity (`Tags.liveOsIdentity`) in its statement digest; otherwise the tags as they are. -/
def Tags.withCoins (t : Tags) (coins : Coins) : Tags :=
  if coins == .os && t.liveOs then { t with identity := Tags.liveOsIdentity t.identity } else t

/-- Everything a session needs about the statement. -/
structure Stmt where
  tags : Tags
  leaf : RowLeaf
  c : Circuit
  pub : Public
  n : Nat
  blocksReal : Nat
  nbl : Nat
  m : Nat
  mPts : Nat
  pin : Nat
  regions : Array Region
  da : Array (Nat × Nat)
  db : Array (Nat × Nat)
  digest : ByteArray
  sigma : ByteArray
  hello : String
  /-- the keyed coin tree `hello` names, from the setup's coins (`Coins.tree?`) -/
  coinTree : Option CoinTree.Spec
  merkle : MerkleScheme
  /-- what the statement check established about the partition (`HmRow.PartitionFinding`), for the verdict; null before e51e2b86 -/
  partition : Json := Json.null
  deriving Inhabited

/-- `Hello`, the session parameters S2 compares. A keyed coin tree's `coins` has an empty nonce here: R7 puts the prover's
in (`CoinTree.checkHello`). -/
def helloOf (tags : Tags) (sigma : ByteArray) (mPts : Nat) (coins : Coins) (tables : Array String := #[tags.table]) : V String :=
  let base := [("profile", Json.str "fast100"), ("reps", Lean.toJson (2 : Nat)), ("flavor", Json.str "rs"),
    ("tables", Json.arr (tables.map Json.str)),
    ("link", Json.mkObj [("sigma", Json.str (hexOf sigma)), ("m", Lean.toJson mPts), ("points", Lean.toJson (2 : Nat))])]
  let base := match coins with
    | .seed => base ++ [("coins", Json.str tags.coinScheme)]
    | .tree s => base ++ [("coins", s.coins "")]
    | .os => base
  let base := if tags.hm96Rows then base ++ [("digests", Json.str "sha512")] else base
  canon (Json.mkObj (if tags.retainRounds then base ++ [("rounds", Json.str "bytes retained")] else base))

/-- `setup` (§3 step 1): parse and check everything, then derive the statement. -/
def Stmt.setup (tags : Tags) (leaf : RowLeaf) (circuitFile publicFile : ByteArray) (coins : Coins)
    (tables : List (String × ByteArray) := []) : V Stmt := do
  let c ← Circuit.parse tags circuitFile tables
  let some merkle := MerkleScheme.find? c.leafScheme | throw s!"leaf-commitment scheme {c.leafScheme}: not one this verifier implements"
  leaf.checkLayout c
  let pub ← Public.load tags c publicFile
  leaf.checkPublic c pub
  let n := pub.n
  let blocksReal := (n + c.g - 1) / c.g
  let nbl := max 3 (log2ceil (max blocksReal 1))
  let m := c.kLog + nbl
  if c.kLog > 27 || m > 35 then throw s!"m = {m}: no fast100 schedule (or k_log > 27)"
  let pin ← c.pin leaf
  let (da, db) ← c.delta leaf pin
  let regions := (← leaf.regions c pub).push (← c.outRegion pub)
  let mut h := tags.statement.toUTF8 ++ leaf.comp.digest ++ c.sha ++ leaf.key c ++ (← canon tags.identity).toUTF8
  for x in [n, blocksReal, nbl, m, c.kLog, pin, regions.size, c.g] do h := pushLe64 h x
  for d in [da, db] do
    h := pushLe64 h d.size
    for (i, cc) in d do h := pushLe32 (pushLe32 h i) cc
  let digest := Sha256.hash h
  let sigma := Sha256.hash (tags.sigmaTag.toUTF8 ++ c.sha ++ digest ++ pub.sha)
  let hello ← helloOf tags sigma (PT_LOCAL + nbl) coins
  return { tags, leaf, c, pub, n, blocksReal, nbl, m, mPts := PT_LOCAL + nbl, pin, regions, da, db, digest, sigma, hello,
           coinTree := coins.tree?, merkle }

namespace Stmt

/-- A one-table session of the statement. -/
def spec (st : Stmt) : SessionSpec :=
  { tables := #[st.tags.table], domain := fun _ r => st.tags.domain r, sigma := st.sigma, hello := st.hello, mPts := st.mPts,
    retainRounds := st.tags.retainRounds, unitDraw := (st.pub.header.getObjVal? "unit_draw").toOption, instances := st.n,
    sha512Digests := st.tags.hm96Rows, coinTree := st.coinTree }

/-- `e[s + 64·u] = lw[s] · eq(ρ, u)` for `u < 2^|ρ|`. -/
def tensor (lw : Array F128) (rho : Array F128) : Array F128 :=
  let eIn := eqTable rho
  Array.ofFn (n := lw.size * eIn.size) fun k => lw[k.val % lw.size]! * eIn[k.val / lw.size]!

private theorem getElem!_map_of_lt {α β} [Inhabited α] [Inhabited β] (f : α → β) (xs : Array α) (i : Nat)
    (h : i < xs.size) : (xs.map f)[i]! = f xs[i]! := by
  rw [getElem!_pos (xs.map f) i (by simpa using h), getElem!_pos xs i h, Array.getElem_map]

/-- `tensor` from each `lw[s]`'s table (`F128.mulBy_mulTableL`) once `eq(ρ)` has 1024 entries. -/
def tensorFast (lw : Array F128) (rho : Array F128) : Array F128 :=
  let eIn := eqTable rho
  if eIn.size < 1024 then Array.ofFn (n := lw.size * eIn.size) fun k => lw[k.val % lw.size]! * eIn[k.val / lw.size]!
  else
    let tabs := lw.map F128.mulTableL
    Array.ofFn (n := lw.size * eIn.size) fun k => F128.mulBy tabs[k.val % lw.size]! eIn[k.val / lw.size]!

@[csimp] theorem tensor_eq_fast : @tensor = @tensorFast := by
  funext lw rho
  simp only [tensor, tensorFast]
  split
  · rfl
  · congr 1
    funext k
    have hk := k.isLt
    have hs : 0 < lw.size := Nat.pos_of_ne_zero fun h => by simp [h] at hk
    rw [getElem!_map_of_lt _ _ _ (Nat.mod_lt _ hs), F128.mulBy_mulTableL]

/-- `α·Aᵀe_lo + Bᵀe_lo` on a slot of `2^sl` bits, the product rows' B side read from the table for a lookup slot. -/
def baseOf (alpha : F128) (eLo : Array F128) (a b : Sparse) (lk : Option LookupTable) (sl : Nat) : Array F128 :=
  let fa := a.foldT eLo
  let fb := match lk with
    | some t => Lookup.foldB t eLo (b.foldT eLo)
    | none => b.foldT eLo
  (Array.range (2 ^ sl)).map fun i => alpha * fa[i]! + fb[i]!

/-- `baseOf` from `α`'s table once the slot has 2048 bits. -/
def baseOfFast (alpha : F128) (eLo : Array F128) (a b : Sparse) (lk : Option LookupTable) (sl : Nat) : Array F128 :=
  let fa := a.foldT eLo
  let fb := match lk with
    | some t => Lookup.foldB t eLo (b.foldT eLo)
    | none => b.foldT eLo
  if 2 ^ sl < 2048 then (Array.range (2 ^ sl)).map fun i => alpha * fa[i]! + fb[i]!
  else
    let ta := F128.mulTableL alpha
    (Array.range (2 ^ sl)).map fun i => F128.mulBy ta fa[i]! + fb[i]!

@[csimp] theorem baseOf_eq_fast : @baseOf = @baseOfFast := by
  funext alpha eLo a b lk sl
  simp only [baseOf, baseOfFast]
  by_cases h : 2 ^ sl < 2048 <;> simp only [h, ↓reduceIte, F128.mulBy_mulTableL]

/-- One slot type's fold on its local tensor `e_lo = lw ⊗ eq(ρ[.. sl − 6])`; a mask slot's is `(α + 1)·e_lo`. -/
def slotBase (st : Stmt) (alpha : F128) (lw rho : Array F128) (key : Kind × Nat) : V (Array F128) := do
  let sl := key.2
  let eLo := tensor lw (rho.extract 0 (sl - 6))
  match key.1 with
  | .comp =>
    if sl != st.leaf.comp.spec.kLog then throw "lincheck fold: a compression slot of the wrong size"
    pure (baseOf alpha eLo st.leaf.comp.a st.leaf.comp.b none sl)
  | .net i =>
    let net := (st.c.nets.getD i default).2
    if net.unitLog != sl then throw "lincheck fold: a unit slot of the wrong size"
    pure (baseOf alpha eLo net.a net.b net.lookup sl)
  | .mask => pure (eLo.map fun x => (alpha + 1) * x)

/-- Offsets `i, …, i + fuel − 1` of the slot at `at_`: `out[at_ + i] = s · base[i]`. -/
def placeSlot (s : F128) (base : Array F128) (at_ : Nat) : Nat → Nat → Array F128 → Array F128
  | 0, _, out => out
  | fuel + 1, i, out => placeSlot s base at_ fuel (i + 1) (out.set! (at_ + i) (s * base[i]!))

/-- `placeSlot` with `s`'s products read from a table `t` (`s`'s `F128.mulTableL`). -/
def placeSlotT (t : Array F128) (base : Array F128) (at_ : Nat) : Nat → Nat → Array F128 → Array F128
  | 0, _, out => out
  | fuel + 1, i, out => placeSlotT t base at_ fuel (i + 1) (out.set! (at_ + i) (F128.mulBy t base[i]!))

theorem placeSlotT_eq (s : F128) (base : Array F128) (at_ : Nat) :
    ∀ fuel i out, placeSlotT (F128.mulTableL s) base at_ fuel i out = placeSlot s base at_ fuel i out
  | 0, _, _ => rfl
  | fuel + 1, i, out => by
    rw [placeSlotT, placeSlot, F128.mulBy_mulTableL, placeSlotT_eq s base at_ fuel]

/-- `placeSlot` from `s`'s table once the slot has 2048 offsets. -/
def placeSlotFast (s : F128) (base : Array F128) (at_ fuel i : Nat) (out : Array F128) : Array F128 :=
  if fuel < 2048 then placeSlot s base at_ fuel i out else placeSlotT (F128.mulTableL s) base at_ fuel i out

@[csimp] theorem placeSlot_eq_fast : @placeSlot = @placeSlotFast := by
  funext s base at_ fuel i out
  simp only [placeSlotFast]
  split
  · rfl
  · rw [placeSlotT_eq]

/-- Slots `k, …, k + fuel − 1` of range `r`, each scaled by `eq(ρ[sl − 6 ..], q)`. -/
def placeRange (r : Range) (eHi base : Array F128) : Nat → Nat → Array F128 → Array F128
  | 0, _, out => out
  | fuel + 1, k, out =>
    let q := r.pos0 + k
    placeRange r eHi base fuel (k + 1) (placeSlot eHi[q]! base (q * 2 ^ r.slotLog) (2 ^ r.slotLog) 0 out)

/-- Each range's slots, every slot type folded once (`bases` caches them). -/
def foldRanges (st : Stmt) (alpha : F128) (lw rho : Array F128) :
    List Range → Array ((Kind × Nat) × Array F128) → Array F128 → V (Array F128)
  | [], _, out => pure out
  | r :: rs, bases, out => do
    let key := (r.kind, r.slotLog)
    let (base, bases') ← match bases.find? (fun kb => decide (kb.1 = key)) with
      | some (_, b) => pure (b, bases)
      | none => do
        let b ← st.slotBase alpha lw rho key
        pure (b, bases.push (key, b))
    let eHi := eqTable (rho.extract (r.slotLog - 6) rho.size)
    foldRanges st alpha lw rho rs bases' (placeRange r eHi base r.count 0 out)

/-- Δ's entries: `out[cc] += w(i)` for each `(i, cc)`. -/
def addDelta (w : Nat → F128) (pairs : List (Nat × Nat)) (out : Array F128) : Array F128 :=
  pairs.foldl (fun o ic => o.modify ic.2 (· + w ic.1)) out

/-- The lincheck fold (§10, §16.1): `α·A_0ᵀe + B_0ᵀe` for `e = lw ⊗ eq(ρ)`. A slot of `2^sl` bits at position `q` sees
`e[(q << sl) | c] = e_lo[c] · eq(ρ[sl−6 ..], q)` with `e_lo = lw ⊗ eq(ρ[.. sl−6])`, so each slot type is folded once on
`e_lo` and scaled per slot; Δ's entries are added on `e`. -/
def fold (st : Stmt) (alpha : F128) (lw : Array F128) (rho : Array F128) : V (Array F128) := do
  let kl := st.c.kLog
  if rho.size + 6 != kl || lw.size != 64 then throw "lincheck fold: the point is not the block's"
  let out ← st.foldRanges alpha lw rho st.c.ranges.toList #[] (Array.replicate (2 ^ kl) 0)
  let eIn := eqTable rho
  let e := fun (i : Nat) => lw[i % 64]! * eIn[i / 64]!
  return addDelta e st.db.toList (addDelta (fun i => alpha * e i) st.da.toList out)

/-- Whether `folded` can read the block's structure: `k_log ≥ 6`, every range inside the block on slots of at least 64
columns, and no two ranges sharing a column (`Circuit.checkLayout` establishes all three). -/
def structured (st : Stmt) : Bool :=
  let kl := st.c.kLog
  let rs := st.c.ranges.toList
  decide (6 ≤ kl) && rs.all (fun r => decide (6 ≤ r.slotLog ∧ r.slotLog ≤ kl ∧ (r.pos0 + r.count) * 2 ^ r.slotLog ≤ 2 ^ kl)) &&
    decide (rs.Pairwise fun a b => (a.pos0 + a.count) * 2 ^ a.slotLog ≤ b.pos0 * 2 ^ b.slotLog ∨
      (b.pos0 + b.count) * 2 ^ b.slotLog ≤ a.pos0 * 2 ^ a.slotLog)

/-- `Σ_{i < n} f(i)`. -/
def sumUpTo (n : Nat) (f : Nat → F128) : F128 := Nat.fold n (fun i _ acc => acc + f i) 0

/-- A slot type's share of the 64 entries: `Σ_{y mod 64 = s} base[y] · eLo[⌊y/64⌋]` at `s`. -/
def project (base eLo : Array F128) : Array F128 :=
  Array.ofFn (n := 64) fun s => sumUpTo eLo.size fun u => base[s.val + 64 * u]! * eLo[u]!

/-- `folded`'s ranges: each range's slot type folded once (`slotBase`, cached as `foldRanges` caches it), its share of
entry `s` the base projected on `eq(r[.. sl − 6], ·)`, times `Σ_q eq(ρ[sl − 6 ..], q) · eq(r[sl − 6 ..], q)` over the
range's slots `q`. -/
def partialRanges (st : Stmt) (alpha : F128) (lw rho rr : Array F128) :
    List Range → Array ((Kind × Nat) × Array F128) → Array F128 → V (Array F128)
  | [], _, out => pure out
  | r :: rs, bases, out => do
    let key := (r.kind, r.slotLog)
    let (base, bases') ← match bases.find? (fun kb => decide (kb.1 = key)) with
      | some (_, b) => pure (b, bases)
      | none => do
        let b ← st.slotBase alpha lw rho key
        pure (b, bases.push (key, b))
    let h := r.slotLog - 6
    let eHi := eqTable (rho.extract h rho.size)
    let rHi := eqTable (rr.extract h rr.size)
    let wt := sumUpTo r.count fun k => eHi[r.pos0 + k]! * rHi[r.pos0 + k]!
    let p := project base (eqTable (rr.extract 0 h))
    partialRanges st alpha lw rho rr rs bases' (Array.ofFn (n := 64) fun s => out[s.val]! + wt * p[s.val]!)

/-- A templated unit's share of entry `s` before `wt` (`Tmpl.foldUnit`, from the template's description, never the slot's
`2^sl` base): where the circuit carries a template whose committed net (`Tmpl.unitNet`) is the range's (by pointer, else
compared), of the slot's size and with no lookup. `none` elsewhere, and where `foldUnit` refuses, and the range is folded
as any other, from the net's rows. -/
def unitShare (st : Stmt) (alpha : F128) (lw rho rr : Array F128) (r : Range) : Option (Array F128) :=
  match st.c.tmpl, r.kind with
  | some tu, .net i =>
    let net := (st.c.nets.getD i default).2
    if net.unitLog == r.slotLog && net.lookup.isNone &&
        withPtrEq net tu.net (fun _ => decide (net = tu.net)) (fun h => by simp [h]) then
      let h := r.slotLog - 6
      (Tmpl.foldUnit tu.t tu.lays r.slotLog alpha lw (rho.extract 0 h) (rr.extract 0 h)).toOption
    else none
  | _, _ => none

/-- `partialRanges`, with a templated unit's range folded from its template (`unitShare`). -/
def partialRangesT (st : Stmt) (alpha : F128) (lw rho rr : Array F128) :
    List Range → Array ((Kind × Nat) × Array F128) → Array F128 → V (Array F128)
  | [], _, out => pure out
  | r :: rs, bases, out => do
    let h := r.slotLog - 6
    let (p, bases') ← match st.unitShare alpha lw rho rr r with
      | some p => pure (p, bases)
      | none => do
        let key := (r.kind, r.slotLog)
        let (base, bases') ← match bases.find? (fun kb => decide (kb.1 = key)) with
          | some (_, b) => pure (b, bases)
          | none => do
            let b ← st.slotBase alpha lw rho key
            pure (b, bases.push (key, b))
        pure (project base (eqTable (rr.extract 0 h)), bases')
    let eHi := eqTable (rho.extract h rho.size)
    let rHi := eqTable (rr.extract h rr.size)
    let wt := sumUpTo r.count fun k => eHi[r.pos0 + k]! * rHi[r.pos0 + k]!
    partialRangesT st alpha lw rho rr rs bases' (Array.ofFn (n := 64) fun s => out[s.val]! + wt * p[s.val]!)

/-- `folded` from the block's structure (`structured`), never a `2^k_log` vector: the ranges' shares (`partialRangesT`),
then each Δ entry `(i, c)`'s `α·e(i)` (B's: `e(i)`) times `eq(r, ⌊c/64⌋)` at `c mod 64`, and `β·eq(r, ⌊pin/64⌋)`, for
`r = ts` reversed. -/
def foldedPartial (st : Stmt) (alpha beta : F128) (lw rho ts : Array F128) : V (Array F128) := do
  let kl := st.c.kLog
  if rho.size + 6 != kl || lw.size != 64 then throw "lincheck fold: the point is not the block's"
  let rr := ts.reverse
  let out ← st.partialRangesT alpha lw rho rr st.c.ranges.toList #[] (Array.replicate 64 0)
  let n := 2 ^ kl
  let eIn := eqTable rho
  let eR := eqTable rr
  let e := fun (i : Nat) => lw[i % 64]! * eIn[i / 64]!
  let add := fun (w : Nat → F128) (o : Array F128) (ic : Nat × Nat) =>
    if ic.2 < n then o.modify (ic.2 % 64) (· + w ic.1 * eR[ic.2 / 64]!) else o
  let out := st.db.toList.foldl (add e) (st.da.toList.foldl (add fun i => alpha * e i) out)
  return if st.pin < n then out.modify (st.pin % 64) (· + beta * eR[st.pin / 64]!) else out

/-- The lincheck's `CircuitFold.folded`: `foldedPartial` on a structured block, else `fold` with `β` at the pin, halved. -/
def folded (st : Stmt) (alpha beta : F128) (lw rho ts : Array F128) : V (Array F128) :=
  if st.structured && ts.size + 6 == st.c.kLog then st.foldedPartial alpha beta lw rho ts
  else do
    let v ← st.fold alpha lw rho
    pure (ts.foldl (fun c t => halve t c) (v.modify st.pin (· + beta)))

def circuitFold (st : Stmt) : CircuitFold := { kLog := st.c.kLog, pin := st.pin, fold := st.fold, folded := st.folded }

/-- The region's point in `F^m` for a link point `r ∈ F^{m_pts}` (§16.2). -/
def regionPoint (st : Stmt) (reg : Region) (r : Array F128) : Array F128 := Id.run do
  let kl := st.c.kLog
  let mut p : Array F128 := (Array.range st.m).map fun j =>
    if j < kl && (reg.mask >>> j) % 2 == 1 && (reg.fixed >>> j) % 2 == 1 then 1 else 0
  for t in [0:reg.free.size] do p := p.set! reg.free[t]! r[t]!
  for i in [0:st.nbl] do p := p.set! (kl + i) r[PT_LOCAL + i]!
  return p

/-- `Σ_b eq(r[PT_LOCAL ..], b) · Σ_i eq(r[.. |free|], i) · bit_i(bytes(region, b))`. -/
def regionValue (st : Stmt) (reg : Region) (r : Array F128) : V F128 := do
  let nf := reg.free.size
  let wt := eqTable (r.extract 0 nf)
  let eb := eqTable (r.extract PT_LOCAL (PT_LOCAL + st.nbl))
  let mut acc : F128 := 0
  for b in [0:2 ^ st.nbl] do
    let bytes := reg.bytes b
    if bytes.size * 8 != 2 ^ nf then throw s!"region {reg.name}: its bytes are not its free bits"
    let mut s : F128 := 0
    for i in [0:bytes.size] do
      let x := bytes.get! i
      for t in [0:8] do
        if (x >>> t.toUInt8) % 2 == 1 then s := s + wt[8 * i + t]!
    acc := acc + eb[b]! * s
  return acc

/-- The extra claims (§16.2): region-major, then point 0 before point 1. -/
def extraClaims (st : Stmt) (pts : Array (Array F128)) : V (Array Claim) := do
  let mut out := #[]
  for reg in st.regions do
    for r in pts do
      let p := st.regionPoint reg r
      out := out.push { value := ← st.regionValue reg r, skip := eqTable (p.extract 0 6), x := p.extract 6 p.size }
  return out

end Stmt

end Flock
