import Proofs.Flock.C3.Opening.SessionAcc
import Proofs.Flock.VBridge.Family.Opened

/-!
# n7's levels: `Opened` at R's consistency checks, from the `rec-open-stacked` units

`levelTie_of_run` asks that every consistency check of R's opening run pass against the proof's cap (`Opened`). G
proves it at V*'s plain openings (`openedAllAt_plain`) by climbing each `rec-open` unit, read with `keyProg_parse` at
the name `RecOpen.params` parses. R's openings are `rec-open-stacked/l<LANES>-h<H>` units, a name `RecOpen.params` does
not parse, so the climb here is `decode_recOpen`'s, from the unit facts `OpensAt` bundles (`SessionAt` (c)).

* `OpensTopAt`: `OpensAt` at a given depth `H`, position `pos` (the `dirs` registered, `RecCoins.checkStackedLevel`'s
  `dirsRow`) and siblings `ss`, with the climb's 512 output columns registered as the node `top` (a
  `rec-top-stacked-L<ℓ>` read). It gives `OpensAt` (`opensAt_of_opensTop`), and the climb from the row's plain leaf
  is `top` (`climb_of_opensTop`).
* `opened_of_climbs`: `Opened` at every context of a run whose contexts are its levels in order (`ConssAt`), from the
  proof's openings and caps, each query's climb to its cap entry, and the messages' consistency sums.
* `opened_of_units`: the same from R's units in V*'s sessions (`OpensTopAt`), at the messages w holds (`decodedO`) with
  the opened rows' consistency sums (`consO`).
-/

namespace FlockC3.RegLevels

open Flock
open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open FlockVBridge FlockVBridge.AccChain
open FlockSoundness.Discharge.Hm (bitL)
open FlockVBridge.Decode (RegBit)
open FlockSoundness.Refine (capAt openAt)

/-- A climb reads its scheme's nodes only. -/
theorem climb_node {ms ms' : MerkleScheme} (h : ∀ l r, ms.node l r = ms'.node l r) :
    ∀ (ss : List ByteArray) (acc : ByteArray) (idx : ℕ), climb ms acc idx ss = climb ms' acc idx ss
  | [], _, _ => rfl
  | s :: rest, acc, idx => by
    simp only [climb, h]
    exact climb_node h rest _ _

theorem sibsOf_length {s : Array ByteArray} {q w j : ℕ} (hs : s.size = q * w) (hj : j < q) :
    (sibsOf s w j).length = w := by
  have hle : j * w + w ≤ q * w := by
    have := Nat.mul_le_mul_right w hj
    rwa [Nat.succ_mul] at this
  simp [sibsOf, hs]
  omega

/-- **A run's consistency contexts are its levels, in order**, each with the level's `q` positions: the conclusion of
G's `Conss.ligerito_conss`, which G proves of the symbolic run (`LM Lin`); R's opening runs Ligerito over F. -/
def ConssAt (sch : Flock.Schedule) (lg : RecAlgebra.LgPts) : Prop :=
  lg.conss.toList.map (fun c => (c.start, c.pos.size)) =
    (List.range sch.levels.size).map fun k => (k, (sch.levels[k]!).q)

/-! ## One opening's climb, from its unit's registered columns -/

section Copy

variable {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv)

/-- **Copy `j` of unit `u` opens row `r` at position `pos` through the siblings `ss` to the node `top`**: `OpensAt`'s
facts at the depth `H`, with the `dirs` columns registered as `pos`'s bits (`RecCoins.checkStackedLevel` registers
`dirsRow pos[j] H`) and the `aux` columns as `ss`, and the climb's 512 output columns registered as `top`'s bits (the
`rec-top-stacked-L<ℓ>` read of R's level node). -/
def OpensTopAt (bits : Fin (Layout.keyN c hU nv) → Bool) (u : Fin nv) (j LANES : ℕ) (r a cf A : Array F128)
    (H pos : ℕ) (ss : List ByteArray) (top : ByteArray) : Prop :=
  ∃ (hj : j < c.upv), bits (Const.Prog.oneAt (Placed.keyProg c hU nv) u) = true ∧
  ∃ (ms : MerkleScheme), (∀ l r, ms.node l r = Sha512.hash (l ++ r)) ∧
  ∃ (p : Ports) (outBits : Array ℕ) (name : String) (rest : ByteArray),
    H ≤ 32 ∧ Flock.HmNets.netOf name (recOpenLaid LANES H p outBits) = .ok c.unit ∧ c.unit.unitLog ≤ 32 ∧
    (recOpenOuts LANES H p).length ≤ outBits.foldl (· + ·) 0 ∧
    r.size = LANES ∧ a.size = 2 ∧ ss.length = H ∧
    p.row = 8 * (MerkleScheme.rowBytes r).size ∧ 512 * H ≤ p.aux ∧
    p.acc = 8 * (MerkleScheme.rowBytes a ++ rest).size ∧ p.coef = 8 * (MerkleScheme.rowBytes cf).size ∧ H ≤ p.dirs ∧
    (∀ t < 8 * (MerkleScheme.rowBytes r).size,
      RegBit zo u j (0 + t) (bitL (MerkleScheme.rowBytes r).data.toList t)) ∧
    (∀ i (hi : i < ss.length), ss[i].size = 64 ∧
      ∀ t < 8 * ss[i].size, RegBit zo u j (p.row + 512 * i + t) (bitL ss[i].data.toList t)) ∧
    (∀ t < 8 * (MerkleScheme.rowBytes a ++ rest).size,
      RegBit zo u j (p.row + p.aux + t) (bitL (MerkleScheme.rowBytes a ++ rest).data.toList t)) ∧
    (∀ t < 8 * (MerkleScheme.rowBytes cf).size,
      RegBit zo u j (p.row + p.aux + p.acc + t) (bitL (MerkleScheme.rowBytes cf).data.toList t)) ∧
    (∀ h < H, RegBit zo u j (p.row + p.aux + p.acc + p.coef + h) (pos.testBit h)) ∧
    (∀ t < 256, RegOut zo u hj (Hm.outStart ((recOpenNet LANES H p).run Flock.HmNets.St.empty).2 p.bits outBits +
      1024 + t) (bitL (MerkleScheme.rowBytes A).data.toList t)) ∧
    top.size = 64 ∧ ∀ t < 512, RegOut zo u hj
      (Hm.outStart ((recOpenNet LANES H p).run Flock.HmNets.St.empty).2 p.bits outBits + t) (bitL top.data.toList t)

theorem opensAt_of_opensTop {bits : Fin (Layout.keyN c hU nv) → Bool} {u : Fin nv} {j LANES H pos : ℕ}
    {r a cf A : Array F128} {ss : List ByteArray} {top : ByteArray}
    (h : OpensTopAt zo bits u j LANES r a cf A H pos ss top) : OpensAt zo bits u j LANES r a cf A := by
  obtain ⟨hj, hone, ms, hms, p, outBits, name, rest, hH, hl, hu, hout, hr, ha, hss, hrowN, hauxN, haccN, hcoefN,
    hdirsN, hrow, haux, hacc, hcoef, hdir, haout, -⟩ := h
  exact ⟨hj, hone, ms, hms, H, p, outBits, name, rest, pos, ss, hH, hl, hu, hout, hr, ha, hss, hrowN, hauxN, haccN,
    hcoefN, hdirsN, hrow, haux, hacc, hcoef, hdir, haout⟩

/-- **An opening's climb is its registered node** (`decode_recOpen`): in a `Good` layer, the copy's first 512 output
columns hold the climb under SHA-512 from the plain leaf of its row through `ss` at `pos`, and they are registered as
`top`'s bits. -/
theorem climb_of_opensTop {bits : Fin (Layout.keyN c hU nv) → Bool} (hG : zo.Good bits) {u : Fin nv}
    {j LANES H pos : ℕ} {r a cf A : Array F128} {ss : List ByteArray} {top : ByteArray}
    (h : OpensTopAt zo bits u j LANES r a cf A H pos ss top) (msId : String) :
    climb (MerkleScheme.sha512 msId) (Sha512.hash (MerkleScheme.rowBytes r)) pos ss = top := by
  obtain ⟨hj, hone, ms, hms, p, outBits, name, rest, hH, hl, hu, hout, hr, ha, hss, hrowN, hauxN, haccN, hcoefN,
    hdirsN, hrow, haux, hacc, hcoef, hdir, -, htop, hT⟩ := h
  have hO := Decode.decode_recOpen zo hG.2 u hj hone ms hms LANES H hH p outBits hl hu hout r a cf rest pos ss hr ha
    hss hrowN hauxN haccN hcoefN hdirsN hrow haux hacc hcoef hdir
  have hnIn : (Rows.ofNet hU).nIn ≤
      Hm.outStart ((recOpenNet LANES H p).run Flock.HmNets.St.empty).2 p.bits outBits := by
    rw [Layout.ofNet_nIn_eq, inWords_recOpen hl]
    exact Decode.inRows_le_outStart _ _ _
  have hX := Family.bytesAt_of_cols (Family.colsHold_left (Family.colsHold_left hO))
  rw [climb_size_of ms hms ss pos _ (FlockSoundness.Refine.sha512_size _)] at hX
  have hT' : ColsHold (copyVal (keyInst hU nv (Layout.outsOf c hU nv) u) bits hj)
      (Hm.outStart ((recOpenNet LANES H p).run Flock.HmNets.St.empty).2 p.bits outBits) top := fun t ht =>
    copyVal_regOut zo hG.2 (keyProg_correct hU nv hG.1 u) hj (by omega) (hT t (by omega))
  have hY := Family.bytesAt_of_cols hT'
  rw [htop] at hY
  exact (climb_node (ms := MerkleScheme.sha512 msId) (ms' := ms) (fun l r => (hms l r).symm) ss _ pos).trans
    (hX.symm.trans hY)

end Copy

/-! ## Every context of the run -/

/-- **Every context of a run whose contexts are its levels passes** (`Opened`, at an unsalted SHA-512 scheme): the
proof opens level `ℓ`'s rows `rows ℓ` with the siblings `sibs ℓ` and no salts, at the level's shapes, its caps have the
schedule's shape (`CapsAt`), each query's climb from its row's plain leaf at the run's position reaches its cap entry,
and the messages' consistency sums are the rows' at the run's `α`. -/
theorem opened_of_climbs {msId : String} {sch : Flock.Schedule} {cap : Array ByteArray} {p : LigeritoProof}
    {M : RecAlgebra.Messages F128} {lg : RecAlgebra.LgPts} (hcs : ConssAt sch lg) (hcaps : Family.CapsAt sch cap p)
    (rows : ℕ → Array (Array F128)) (sibs : ℕ → Array ByteArray)
    (hO : ∀ ℓ < sch.levels.size, openAt p (sch.levels.size - 1) ℓ = (rows ℓ, sibs ℓ, #[]))
    (hrs : ∀ ℓ < sch.levels.size, (rows ℓ).size = (sch.levels[ℓ]!).q)
    (hss : ∀ ℓ < sch.levels.size,
      (sibs ℓ).size = (sch.levels[ℓ]!).q * ((sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap))
    (h64 : ∀ ℓ < sch.levels.size, ∀ x ∈ sibs ℓ, x.size = 64)
    (hrow : ∀ ℓ < sch.levels.size, ∀ j < (sch.levels[ℓ]!).q, ((rows ℓ)[j]!).size = 2 ^ (sch.levels[ℓ]!).lanes)
    (hclimb : ∀ ℓ < sch.levels.size, ∀ j < (sch.levels[ℓ]!).q,
      climb (MerkleScheme.sha512 msId) (Sha512.hash (MerkleScheme.rowBytes (rows ℓ)[j]!)) ((lg.conss[ℓ]!).pos[j]!)
          (sibsOf (sibs ℓ) ((sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap) j) =
        (capAt cap p ℓ).getD ((lg.conss[ℓ]!).pos[j]! >>> ((sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap))
          ByteArray.empty)
    (hcons : ∀ ℓ < sch.levels.size, M.cons ℓ ((eqTable (lg.conss[ℓ]!).alpha).extract 0 (sch.levels[ℓ]!).q) =
      (Array.range (2 ^ (sch.levels[ℓ]!).lanes)).map
        (RecAlgebra.consAt (rows ℓ) (sch.levels[ℓ]!).q (lg.conss[ℓ]!).alpha)) :
    ∀ cs ∈ lg.conss.toList, RecAlgebra.Opened (MerkleScheme.sha512 msId) sch cap p M cs := by
  intro cs hmem
  obtain ⟨hℓ, hcs0, hpsz⟩ := Family.Conss.conss_mem hcs hmem
  have hpsz' : cs.pos.size = (sch.levels[cs.start]!).q := hpsz
  obtain ⟨hc1, hc2⟩ := hcaps cs.start hℓ
  have hcl := hclimb cs.start hℓ
  have hco := hcons cs.start hℓ
  rw [hcs0] at hcl hco
  unfold RecAlgebra.Opened
  rw [Family.Conss.getD_default, hO cs.start hℓ]
  refine ⟨Family.Conss.merkleCheck_of (rows := rows cs.start) (sibs := sibs cs.start) (salts := #[]) hc1
    (by rw [hrs cs.start hℓ, hpsz']) (by rw [hss cs.start hℓ, hpsz']) rfl hc2 (h64 cs.start hℓ) fun j hj => ?_, hco⟩
  have hj' : j < (sch.levels[cs.start]!).q := by
    rw [← hpsz']
    exact hj
  unfold opens
  simp only [Bool.and_eq_true, decide_eq_true_eq]
  exact ⟨⟨⟨hrow cs.start hℓ j hj', sibsOf_length (hss cs.start hℓ) hj'⟩, by simp; rfl⟩, hcl j hj'⟩

/-- `decodedO`'s consistency sums are `cons`' (G's `decoded_cons`, at `valO`). -/
theorem decodedO_cons (sh : Flock.Shape) (w : Array F128) (cons : ℕ → ℕ → F128) (ℓ : ℕ) (ea : Array F128) :
    (Opening.decodedO sh w cons).cons ℓ ea = (Array.range (2 ^ (sh.sched.levels[ℓ]!).lanes)).map (cons ℓ) := by
  show ((Array.range (2 ^ (sh.sched.levels[ℓ]!).lanes)).map (RecAlgebra.Lin.var ∘ RecAlgebra.Key.cons ℓ)).map
    (RecAlgebra.Lin.eval (Opening.valO sh w cons)) = _
  rw [Array.map_map]
  congr 1
  funext i
  exact RecAlgebra.Lin.eval_var _ _

section Units

variable {nS : ℕ} {Vs : Fin nS → VStmt} {Reg' : Type}

/-- **(n7's levels) `Opened` at R's consistency checks, from R's units in V*'s sessions.** Let the sessions `zs` have
`Good` layers `Lb`, R's run have its levels for contexts (`ConssAt`), and the proof open each level's rows `ops.rows ℓ`
(R's openings') with the siblings `sibs ℓ` and no salts, at the level's shapes, its caps of the schedule's shape. If
opening `j` of level `ℓ` is copy `cp ℓ j` of unit `un ℓ j` in session `sess ℓ`, opening its row at the run's position
through its siblings in the proof to its cap entry (`OpensTopAt`), and R's openings have the level's `q` and the run's
`α`, then every consistency check passes at the messages w holds with the opened rows' consistency sums. -/
theorem opened_of_units (zs : (s : Fin nS) → (Vs s).Outer Reg') (Lb : (s : Fin nS) → (Vs s).Bits)
    (hG : ∀ s, (zs s).Good (Lb s)) {sh : Flock.Shape} {w : Array F128} {ops : FlockVBridge.AccChain.Openings}
    {msId : String} {cap : Array ByteArray} {p : LigeritoProof} {lg : RecAlgebra.LgPts} (hcs : ConssAt sh.sched lg)
    (hcaps : Family.CapsAt sh.sched cap p) (sibs : ℕ → Array ByteArray)
    (hO : ∀ ℓ < sh.sched.levels.size, openAt p (sh.sched.levels.size - 1) ℓ = (ops.rows ℓ, sibs ℓ, #[]))
    (hrs : ∀ ℓ < sh.sched.levels.size, (ops.rows ℓ).size = (sh.sched.levels[ℓ]!).q)
    (hss : ∀ ℓ < sh.sched.levels.size,
      (sibs ℓ).size = (sh.sched.levels[ℓ]!).q * ((sh.sched.levels[ℓ]!).d - (sh.sched.levels[ℓ]!).cap))
    (h64 : ∀ ℓ < sh.sched.levels.size, ∀ x ∈ sibs ℓ, x.size = 64)
    (hq : ∀ ℓ < sh.sched.levels.size, ops.q ℓ = (sh.sched.levels[ℓ]!).q)
    (halpha : ∀ ℓ < sh.sched.levels.size, ops.alpha ℓ = (lg.conss[ℓ]!).alpha)
    (sess : ℕ → Fin nS) (un : (ℓ : ℕ) → ℕ → Fin (Vs (sess ℓ)).nv) (cp : ℕ → ℕ → ℕ)
    (hu : ∀ ℓ < sh.sched.levels.size, ∀ j < (sh.sched.levels[ℓ]!).q, ∃ a cf A,
      OpensTopAt (zs (sess ℓ)) (Lb (sess ℓ)) (un ℓ j) (cp ℓ j) (levelLanes sh ℓ) (ops.rows ℓ)[j]! a cf A
        ((sh.sched.levels[ℓ]!).d - (sh.sched.levels[ℓ]!).cap) ((lg.conss[ℓ]!).pos[j]!)
        (sibsOf (sibs ℓ) ((sh.sched.levels[ℓ]!).d - (sh.sched.levels[ℓ]!).cap) j)
        ((capAt cap p ℓ).getD ((lg.conss[ℓ]!).pos[j]! >>> ((sh.sched.levels[ℓ]!).d - (sh.sched.levels[ℓ]!).cap))
          ByteArray.empty)) :
    ∀ cs ∈ lg.conss.toList,
      RecAlgebra.Opened (MerkleScheme.sha512 msId) sh.sched cap p (Opening.decodedO sh w (Opening.consO ops)) cs := by
  refine opened_of_climbs hcs hcaps ops.rows sibs hO hrs hss h64 (fun ℓ hℓ j hj => ?_) (fun ℓ hℓ j hj => ?_)
    fun ℓ hℓ => ?_
  · obtain ⟨a, cf, A, h⟩ := hu ℓ hℓ j hj
    obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, hr, _⟩ := h
    exact hr
  · obtain ⟨a, cf, A, h⟩ := hu ℓ hℓ j hj
    exact climb_of_opensTop (zs (sess ℓ)) (hG (sess ℓ)) h msId
  · rw [decodedO_cons, ← hq ℓ hℓ, ← halpha ℓ hℓ]
    rfl

end Units

end FlockC3.RegLevels
