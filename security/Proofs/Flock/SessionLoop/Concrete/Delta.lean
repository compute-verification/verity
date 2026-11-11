module

public import Proofs.Flock.SessionLoop.Concrete.Fwd
public import Flock
import all Init.Data.String.TakeDrop
import all Init.Data.String.Slice
import all Init.Data.String.Pattern.Basic
import all Init.Data.String.Pattern.String
import all Init.Data.String.Pattern.Char
import all Init.Data.String.Search
import all Init.Data.String.Iterate
import all Init.Data.String.Decode
import all Init.Data.String.Termination
import all Init.Data.Array.Attach
import all Init.WF
import Flock.HmRow
import Flock.HmNets
import Flock.Canon
import Flock.JsonParse
import Flock.Net
import Flock.Firewall
import Flock.Norm
import Flock.Public
import Flock.Statement
import Flock.Comp
import Flock.Bytes
import Flock.FV3
import Flock.Hm96
import Flock.Hash
import Flock.RowLeaf
import Flock.Circuit
import Flock.Tags
import Flock.Zk
import Flock.CoinTree
import Flock.Merkle
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic
import all Std.Data.TreeMap.Raw.Basic
import all Std.Data.DTreeMap.Raw.Basic
import all Std.Data.DTreeMap.Internal.Operations
import all Std.Data.DTreeMap.Internal.Queries
import all Std.Data.DTreeMap.Internal.Balancing
import all Std.Data.DTreeMap.Internal.Def
import all Init.Data.Ord.String
import all Init.Data.Ord.Basic
import all Init.Data.String.Defs
import all Init.Data.String.Basic
import all Init.Data.String.Extra
import all Init.Data.Array.QSort.Basic
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control
import all Init.Data.Range.Basic
import all Init.Data.Int.Repr
import all Init.Data.Repr
import all Init.Data.Nat.Fold

/-! # `HmRow.delta` as lists

`HmRow.delta` builds a statement's two dummy-wire arrays by `Array.push`, which the kernel evaluates by copying the
array's list on every push, so a kernel check over `delta`'s arrays costs the square of their length. `HmRow.deltaL` is
the same computation consing onto lists, and `deltaL_eq` proves, for every circuit and pin, that `delta`'s arrays are
`deltaL`'s lists reversed (`Φ`). `all_dA` and `all_dB` then move a check over `delta`'s arrays to `deltaL`'s.
-/

@[expose] public section
open Flock
open Lean (Json)

set_option maxRecDepth 200000

namespace Flock.HmRow
def deltaL (c : Circuit) (pin : Nat) : List (Nat × Nat) × List (Nat × Nat) := Id.run do
  let mut d : List (Nat × Nat) := []
  for r in c.ranges do
    if let .net i := r.kind then
      for s in [0:r.count] do
        let k := r.slot s + (c.nets.getD i default).2.constPos
        if k != pin then d := ((k, pin)) :: ((k, k)) :: d
  let (_, where_) := rowSlots c.ports
  let sin := portCols (c.nets.getD c.shaNet default).2.inGroups
  let hin := portCols (c.nets.getD c.hmNet default).2.inGroups
  let unit := c.unit
  let ur := c.unitRange
  let zero := ur.slot 0 + unit.useful
  let sout := portCols (c.nets.getD c.shaNet default).2.outGroups
  for g in [0:c.g] do
    for p in [0:where_.size] do
      let w := where_[p]!
      let port := c.ports[p]!
      -- a segmented row: each chain start's midstate (a segment's, the outer message's), every constant message bit, and the
      -- outer message's digest bits copies of the segments' last outputs
      if let some segs := port.segs then
        for k in [0:w.size + 1] do
          let (cv, msg) := if k < w.size then
              (c.slotCol c.shaNet g w[k]!.1 + sin[2 * w[k]!.2]!.1, c.slotCol c.shaNet g w[k]!.1 + sin[2 * w[k]!.2 + 1]!.1)
            else (c.slotCol c.hmNet g p + hin[0]!.1, c.slotCol c.hmNet g p + hin[1]!.1)
          if let some j := chainStart port k then
            let ms := if j < segs.size then segMidstate port j else midstateOf port
            for t in [0:512] do
              let src := if (ms[t / 64]! >>> (t % 64).toUInt64) &&& 1 == 1 then pin else zero
              d := ((cv + t, src)) :: ((cv + t, cv + t)) :: d
          for t in [0:1024] do
            match msgBit port k t with
            | .const v => d := ((msg + t, if v then pin else zero)) :: ((msg + t, msg + t)) :: d
            | .digest kk col =>
              let src := c.slotCol c.shaNet g w[kk]!.1 + sout[w[kk]!.2]!.1 + col
              d := ((msg + t, src)) :: ((msg + t, msg + t)) :: d
            | .row _ => pure ()
        continue
      let ms := midstateOf port
      let tl := tailOf port
      let cv := if 0 < w.size then c.slotCol c.shaNet g w[0]!.1 + sin[2 * w[0]!.2]!.1 else c.slotCol c.hmNet g p + hin[0]!.1
      for t in [0:512] do
        let src := if (ms[t / 64]! >>> (t % 64).toUInt64) &&& 1 == 1 then pin else zero
        d := ((cv + t, src)) :: ((cv + t, cv + t)) :: d
      -- the last block's constant bits (a sha512/row/v1 row's: all of it), then those of the blocks before it from the
      -- first that holds a bit at or past the row's length
      let pad := c.slotCol c.hmNet g p + hin[1]!.1
      for t in [0:1024] do
        if let some v := constBit port tl w.size t then
          d := ((pad + t, if v then pin else zero)) :: ((pad + t, pad + t)) :: d
      for k in [port.bits / 1024:w.size] do
        let msg := c.slotCol c.shaNet g w[k]!.1 + sin[2 * w[k]!.2 + 1]!.1
        for t in [0:1024] do
          if let some v := constBit port tl k t then
            d := ((msg + t, if v then pin else zero)) :: ((msg + t, msg + t)) :: d
  let msgCol := fun (g p byte t : Nat) =>
    let (k, o) := byteAt c.ports[p]! byte
    if k < where_[p]!.size then
      let (slot, sub) := where_[p]![k]!
      c.slotCol c.shaNet g slot + sin[2 * sub + 1]!.1 + 64 * (o / 8) + 56 - 8 * (o % 8) + t
    else c.slotCol c.hmNet g p + hin[1]!.1 + 64 * (o / 8) + 56 - 8 * (o % 8) + t
  let leafCols := (portCols #[unit.inGroups.getD 0 default]).map (·.1)
  let starts := (c.ports.foldl (fun (acc : Array Nat × Nat) p => (acc.1.push acc.2, acc.2 + p.words)) (#[], 0)).1
  for k in [0:ur.count] do
    let (g, u) := (k / c.upv, k % c.upv)
    let base := ur.slot k
    let li := c.leavesIn.getD u #[]
    for j in [0:li.size] do
      let lf := li[j]!
      for t in [0:16] do
        -- a zero leaf, or a bit row's leaf bit past its length: the slot's forced-zero row
        let src := if lf < 0 then base + unit.useful else
          let x := lf.toNat
          let p := ((List.range starts.size).reverse.find? (fun q => starts[q]! ≤ x)).getD 0
          if 16 * (x - starts[p]!) + t < c.ports[p]!.bits then msgCol g p (2 * (x - starts[p]!) + t / 8) (t % 8)
          else base + unit.useful
        let i := base + leafCols.getD j 0 + t
        d := ((i, src)) :: ((i, i)) :: d
  let pc := portCols unit.inGroups
  for g in [0:c.g] do
    for (u, j, leaf) in c.leafCuts do
      let p := ((List.range starts.size).reverse.find? (fun q => starts[q]! ≤ leaf)).getD 0
      let dst := c.slotCol c.unitNet g u + pc[j]!.1
      for t in [0:pc[j]!.2] do
        let src := if t < 16 && 16 * (leaf - starts[p]!) + t < c.ports[p]!.bits then
          msgCol g p (2 * (leaf - starts[p]!) + t / 8) (t % 8) else zero
        d := ((dst + t, src)) :: ((dst + t, dst + t)) :: d
  for g in [0:c.g] do
    for w in c.wires do
      let dc := (portCols (c.nets.getD w.dst.1 default).2.inGroups).getD w.dst.2.2 default
      let sc := (portCols (c.nets.getD w.src.1 default).2.outGroups).getD w.src.2.2 default
      let dst := c.slotCol w.dst.1 g w.dst.2.1 + dc.1
      let src := c.slotCol w.src.1 g w.src.2.1 + sc.1
      for t in [0:dc.2] do d := ((dst + t, src + t)) :: ((dst + t, dst + t)) :: d
  -- a typed template, per VU: each input of the instance a copy of its message bit (input w: bit w % 16 of row word
  -- w / 16); each part's bound input rows their forms over the unit's columns, in A and B; the root's rows' entries at
  -- its parts' columns, A's in A and B's in B
  let some tb := c.typed | return (d, d)
  let col := fun (g u : Nat) =>
    if u < tb.own then c.slotCol c.unitNet g 0 + u
    else match tb.parts.toList.reverse.find? (·.2.2 ≤ u) with
      | some (n, q, b) => c.slotCol n g q + (u - b)
      | none => 0
  let mut da := d
  let mut db := d
  for g in [0:c.g] do
    for w in [0:tb.inCols.size] do
      let (leaf, bit) := (w / 16, w % 16)
      let p := ((List.range starts.size).reverse.find? (fun q => starts[q]! ≤ leaf)).getD 0
      let i := col g tb.inCols[w]!
      let src := msgCol g p (2 * (leaf - starts[p]!) + bit / 8) (bit % 8)
      da := ((i, src)) :: ((i, i)) :: da
      db := ((i, src)) :: ((i, i)) :: db
    for (r, src) in tb.bind do
      let i := col g r
      da := (src.map (fun x => (i, col g x))).toList.reverse ++ (i, i) :: da
      db := (src.map (fun x => (i, col g x))).toList.reverse ++ (i, i) :: db
    for (r, xa, xb) in tb.cross do
      let i := col g r
      da := (xa.map (fun x => (i, col g x))).toList.reverse ++ da
      db := (xb.map (fun x => (i, col g x))).toList.reverse ++ db
  return (da, db)
end Flock.HmRow

namespace FlockDeploy.Concrete

def dA (c : Circuit) (p : Nat) : Array (Nat × Nat) := (HmRow.delta c p).1
def dB (c : Circuit) (p : Nat) : Array (Nat × Nat) := (HmRow.delta c p).2
theorem delta_ab (c : Circuit) (p : Nat) : HmRow.delta c p = (dA c p, dB c p) := rfl
theorem crw_nil (c : Circuit) (da db : Array (Nat × Nat)) (regions : Array Region)
    (h : HmRow.sharedColumns regions = #[]) : HmRow.checkRegionWords c da db regions = .ok () := by
  unfold HmRow.checkRegionWords; rw [h]; rfl

abbrev WirePair := Nat × Nat
def φ (a : Array WirePair) : List WirePair := a.toList.reverse
def Φ (p : Array WirePair × Array WirePair) : List WirePair × List WirePair := (φ p.1, φ p.2)
theorem φ_push (a : Array WirePair) (x : WirePair) : φ (a.push x) = x :: φ a := by simp [φ]
theorem φ_append (a b : Array WirePair) : φ (a ++ b) = b.toList.reverse ++ φ a := by simp [φ]
theorem φ_empty : φ #[] = [] := rfl

def stepMap {β₁ β₂ : Type} (f : β₁ → β₂) : ForInStep β₁ → ForInStep β₂
  | .yield b => .yield (f b)
  | .done b => .done (f b)

theorem forIn_list_rel {α β₁ β₂ : Type} (f : β₁ → β₂) (l : List α) (b₁ : β₁) (b₂ : β₂)
    (g₁ : α → β₁ → Id (ForInStep β₁)) (g₂ : α → β₂ → Id (ForInStep β₂)) (hb : f b₁ = b₂)
    (hg : ∀ a x, stepMap f (g₁ a x) = g₂ a (f x)) : f (forIn l b₁ g₁) = forIn l b₂ g₂ := by
  subst hb
  induction l generalizing b₁ with
  | nil => rfl
  | cons a t ih =>
    simp only [List.forIn_cons]
    have h := hg a b₁
    revert h
    cases g₁ a b₁ with
    | done b => intro h; rw [← h]; rfl
    | yield b => intro h; rw [← h]; exact ih b

theorem forIn_range_rel {β₁ β₂ : Type} (f : β₁ → β₂) (r : Std.Legacy.Range) (b₁ : β₁) (b₂ : β₂)
    (g₁ : Nat → β₁ → Id (ForInStep β₁)) (g₂ : Nat → β₂ → Id (ForInStep β₂)) (hb : f b₁ = b₂)
    (hg : ∀ a x, stepMap f (g₁ a x) = g₂ a (f x)) : f (forIn r b₁ g₁) = forIn r b₂ g₂ := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.forIn_eq_forIn_range']
  exact forIn_list_rel f _ b₁ b₂ g₁ g₂ hb hg

theorem forIn_array_rel {α β₁ β₂ : Type} (f : β₁ → β₂) (xs : Array α) (b₁ : β₁) (b₂ : β₂)
    (g₁ : α → β₁ → Id (ForInStep β₁)) (g₂ : α → β₂ → Id (ForInStep β₂)) (hb : f b₁ = b₂)
    (hg : ∀ a x, stepMap f (g₁ a x) = g₂ a (f x)) : f (forIn xs b₁ g₁) = forIn xs b₂ g₂ := by
  rw [← Array.forIn_toList, ← Array.forIn_toList]
  exact forIn_list_rel f _ b₁ b₂ g₁ g₂ hb hg

theorem bind_rel {α₁ α₂ γ₁ γ₂ : Type} (f : α₁ → α₂) (ψ : γ₁ → γ₂) (x₁ : Id α₁) (x₂ : Id α₂)
    (k₁ : α₁ → Id γ₁) (k₂ : α₂ → Id γ₂) (hx : f x₁ = x₂) (hk : ∀ a, ψ (k₁ a) = k₂ (f a)) :
    ψ (x₁ >>= k₁) = x₂ >>= k₂ := by
  subst hx; exact hk x₁

theorem yield_rel {β₁ β₂ : Type} (f : β₁ → β₂) (x : β₁) (y : β₂) (h : f x = y) :
    stepMap f (pure (ForInStep.yield x) : Id _) = (pure (ForInStep.yield y) : Id _) := by
  subst h; rfl

theorem pure_rel {γ₁ γ₂ : Type} (ψ : γ₁ → γ₂) (x : γ₁) (y : γ₂) (h : ψ x = y) :
    ψ (pure x : Id γ₁) = (pure y : Id γ₂) := h

def φ2 (p : Array WirePair × Array WirePair) : List WirePair × List WirePair := (φ p.1, φ p.2)

theorem yield_rel' {β₁ β₂ : Type} (f : β₁ → β₂) (x : β₁) :
    stepMap f (pure (ForInStep.yield x) : Id _) = (pure (ForInStep.yield (f x)) : Id _) := rfl

theorem φ2_mk (a b : Array WirePair) : φ2 (a, b) = (φ a, φ b) := rfl
theorem φ2_fst (x : Array WirePair × Array WirePair) : (φ2 x).fst = φ x.fst := rfl
theorem φ2_snd (x : Array WirePair × Array WirePair) : (φ2 x).snd = φ x.snd := rfl
theorem Φ_pure (a b : Array WirePair) : Φ (pure (a, b) : Id _) = (pure (φ a, φ b) : Id _) := rfl

theorem deltaL_eq (c : Circuit) (pin : Nat) : Φ (HmRow.delta c pin) = HmRow.deltaL c pin := by
  unfold HmRow.delta HmRow.deltaL
  dsimp only [Id.run]
  iterate 80 (try (any_goals (first
    | (with_reducible refine forIn_range_rel φ _ _ _ _ _ ?_ ?_)
    | (with_reducible refine forIn_array_rel φ _ _ _ _ _ ?_ ?_)
    | (with_reducible refine forIn_range_rel φ2 _ _ _ _ _ ?_ ?_)
    | (with_reducible refine forIn_array_rel φ2 _ _ _ _ _ ?_ ?_)
    | (with_reducible refine bind_rel φ _ _ _ _ _ ?_ ?_)
    | (with_reducible refine bind_rel φ2 _ _ _ _ _ ?_ ?_)
    | (with_reducible exact yield_rel' φ _)
    | (with_reducible exact yield_rel' φ2 _)
    | (with_reducible refine yield_rel φ _ _ ?_)
    | (with_reducible refine yield_rel φ2 _ _ ?_)
    | (intro _)
    | (with_reducible exact φ_empty)
    | (with_reducible exact Φ_pure _ _)
    | (simp only [φ_push, φ_append, φ_empty, φ2_mk, φ2_fst, φ2_snd, Φ]; (try rfl); done)
    | case_both
    | dsimp only)))

theorem all_dA (c : Circuit) (p : Nat) (q : Nat × Nat → Bool) : (dA c p).all q = (HmRow.deltaL c p).1.all q := by
  rw [← (congrArg Prod.fst (deltaL_eq c p) : (Φ (HmRow.delta c p)).1 = _)]
  simp only [Φ, φ, dA, List.all_reverse, Array.all_toList]

theorem all_dB (c : Circuit) (p : Nat) (q : Nat × Nat → Bool) : (dB c p).all q = (HmRow.deltaL c p).2.all q := by
  rw [← (congrArg Prod.snd (deltaL_eq c p) : (Φ (HmRow.delta c p)).2 = _)]
  simp only [Φ, φ, dB, List.all_reverse, Array.all_toList]

end FlockDeploy.Concrete
