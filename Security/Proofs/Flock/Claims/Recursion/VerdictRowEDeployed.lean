import Proofs.Flock.Claims.Recursion.VerdictAssembly

/-!
# Row E at the decision, from the `rec-open` pieces' running sums

`runs_assembled` (`VerdictAssembly`) takes row E as a chain `accC r` and a position `p0 r` per rep with `UnitSums` and
`ReadsEnds` at `D.opens cls acc i r (p0 r)`, and leaves the reps' accumulators `acc` free. This file fixes all three and
proves both facts from the pieces.

* **The chain** (`chainAt`, `nOpen`, `chainPos`): one chain per table, V\*'s `rec_outer.Chain`: both reps' openings,
  rep-major, then by level and query. `chainAt x ms i p` is row `p` of table `i`'s `rec-acc` as the messages carry it,
  a parameter. Rep `r` starts at `r · nOpen`, so rep 1's first row is rep 0's last.
* **The accumulators** (`DecAt.accE`): rep `k = 2i + r`'s four are the chain's rows at its ends, in `accNames`' order
  (`acc0 0, acc0 1, acc1 0, acc1 1`).
* **`ReadsEnds`** (`readsEnds_accE`): at `accE`, V\*'s algebra reads `acc0`, `acc1` at w's tail (`wi_acc0`, `wi_acc1`:
  the running sums follow the rounds' elements, `notMem_acc`), and w's frames part has the rounds' elements' count
  (`frames_length`, from row C's hypotheses), so w's tail is `accE`.
* **`UnitSums`** (`accHolds`, `unitSums_of_pieces`): the pieces are row D's `OpenPiece`s, since the `rec-open` copy that
  climbs also writes `acc_out`. A piece holds where its `acc_out` row is `consTerm` of its opened row, the chain's row at
  its position and its `coef` (`rec_open.acc_ref`), all read off the messages and coins. With D's `hside` and `hcover`
  and the coin tie `hcoef` (the piece's `coef` is the verifier's), the pieces give `UnitSums`.
* **Row E and the table's run** (`rowE_of_pieces`, `runs_assembledE`): `rowE_of_units` at these, and `runs_assembled`
  with E's hypotheses replaced by the pieces' and `acc := accE`.
-/

namespace RecDeploy.Rows

open Flock
open FlockDeploy (Coin FinalP tabsOf pointsOf shapeAt chunksOf vals)
open FlockVBridge.AccChain (UnitSums ReadsEnds levelLanes)
open FlockVBridge.Family (FramedAt ProofShaped roundSizes)
open Flock.RecAlgebra (Key rounds wNames wi accNames valOf verifierRows NoEmpty structureOf)
open Flock.RecResiduals (portSizes groups)
open FlockSoundness.Refine (openAt)
open FlockSoundness.Discharge.Exec (repFrames)
open FlockSoundness.Discharge.Recursion (HClass)
open FlockSoundness.Discharge.DecodeZL (ProofSizes decodeProof_sizes)

/-! ## The chain's positions -/

/-- One rep's openings at a schedule: every level's queries. -/
def nOpen (sch : Schedule) : ℕ := ((List.range sch.levels.size).map fun ℓ => (sch.levels.getD ℓ default).q).sum

/-- **Query `j` of level `ℓ` of rep `r`'s position on the chain**: rep-major, then by level, then by query
(`rec_outer.chain_order`). -/
def chainPos (sch : Schedule) (r ℓ j : ℕ) : ℕ :=
  r * nOpen sch + ((List.range ℓ).map fun ℓ' => (sch.levels.getD ℓ' default).q).sum + j

/-! ## The running sums' slots in w -/

theorem notMem_acc {sh : Shape} (hh : sh.holo = false) (r : ℕ) :
    Key.acc0 r ∉ (rounds sh).flatten ∧ Key.acc1 r ∉ (rounds sh).flatten := by
  rw [← Array.mem_toList_iff, ← Array.mem_toList_iff, FlockDecodeTies.rounds_els hh]
  have hlv : ∀ i, Key.acc0 r ∉ FlockDecodeTies.lvEls sh.sched i ∧ Key.acc1 r ∉ FlockDecodeTies.lvEls sh.sched i := by
    intro i
    unfold FlockDecodeTies.lvEls
    split <;> simp [FlockDecodeTies.msgEls]
  have h0 := fun i => (hlv i).1
  have h1 := fun i => (hlv i).2
  simp [FlockDecodeTies.zcEls, FlockDecodeTies.lcEls, FlockDecodeTies.rsEls, FlockDecodeTies.ligEls,
    FlockDecodeTies.msgEls, h0, h1]

/-- **A tail key's first slot**: a key that `xs` doesn't hold, at slot `j` of `ys` and at none before, is first at
`xs.size + j` in `xs ++ ys`. -/
theorem idxOf?_append_of {α : Type} [DecidableEq α] {xs ys : Array α} {a : α} (ha : a ∉ xs) {j : ℕ}
    (hy : ys[j]? = some a) (hfirst : ∀ j' < j, ys[j']? ≠ some a) :
    (xs ++ ys).idxOf? a = some (xs.size + j) := by
  have hj : j < ys.size := by
    rcases Nat.lt_or_ge j ys.size with h | h
    · exact h
    · rw [Array.getElem?_eq_none h] at hy; cases hy
  have hlt : xs.size + j < (xs ++ ys).size := by rw [Array.size_append]; omega
  have hat : (xs ++ ys)[xs.size + j]? = some a := by
    rw [Array.getElem?_append_right (Nat.le_add_right _ _), Nat.add_sub_cancel_left]
    exact hy
  have h : (xs ++ ys).finIdxOf? a = some ⟨xs.size + j, hlt⟩ := by
    rw [Array.finIdxOf?_eq_some_iff]
    refine ⟨?_, fun j' hj' => ?_⟩
    · show (xs ++ ys)[xs.size + j]'hlt = a
      exact Option.some.inj ((Array.getElem?_eq_getElem hlt).symm.trans hat)
    · have hm : j'.val < xs.size + j := Fin.lt_def.1 hj'
      intro e
      have e' : (xs ++ ys)[j'.val]? = some a := by
        rw [Array.getElem?_eq_getElem j'.isLt]
        exact congrArg some e
      by_cases hl : j'.val < xs.size
      · rw [Array.getElem?_append_left hl] at e'
        obtain ⟨p, hp⟩ := Array.getElem?_eq_some_iff.1 e'
        exact ha (hp ▸ Array.getElem_mem p)
      · rw [Array.getElem?_append_right (by omega)] at e'
        exact hfirst (j'.val - xs.size) (by omega) e'
  rw [Array.idxOf?_eq_map_finIdxOf?_val, h]
  rfl

theorem wi_acc0 {sh : Shape} (hh : sh.holo = false) {r : ℕ} (hr : r < 2) :
    wi sh (.acc0 r) = some ((rounds sh).flatten.size + r) := by
  show ((rounds sh).flatten ++ accNames).idxOf? (.acc0 r) = _
  rcases (by omega : r = 0 ∨ r = 1) with rfl | rfl
  · exact idxOf?_append_of (notMem_acc hh 0).1 (by decide) fun j' h => absurd h (by omega)
  · exact idxOf?_append_of (notMem_acc hh 1).1 (by decide) fun j' h => by
      obtain rfl : j' = 0 := by omega
      decide

theorem wi_acc1 {sh : Shape} (hh : sh.holo = false) {r : ℕ} (hr : r < 2) :
    wi sh (.acc1 r) = some ((rounds sh).flatten.size + (2 + r)) := by
  show ((rounds sh).flatten ++ accNames).idxOf? (.acc1 r) = _
  rcases (by omega : r = 0 ∨ r = 1) with rfl | rfl
  · exact idxOf?_append_of (notMem_acc hh 0).2 (by decide) fun j' h => by
      rcases (by omega : j' = 0 ∨ j' = 1) with rfl | rfl <;> decide
  · exact idxOf?_append_of (notMem_acc hh 1).2 (by decide) fun j' h => by
      rcases (by omega : j' = 0 ∨ j' = 1 ∨ j' = 2) with rfl | rfl | rfl <;> decide

theorem valOf_acc0 {sh : Shape} {w comb : Array F128} {cons : ℕ → ℕ → F128} {r j : ℕ}
    (h : wi sh (.acc0 r) = some j) : valOf sh w comb cons (.acc0 r) = w[j]! := by
  simp only [valOf, h]

theorem valOf_acc1 {sh : Shape} {w comb : Array F128} {cons : ℕ → ℕ → F128} {r j : ℕ}
    (h : wi sh (.acc1 r) = some j) : valOf sh w comb cons (.acc1 r) = w[j]! := by
  simp only [valOf, h]

theorem getElem!_append_length {L : List F128} {a : Array F128} (t : ℕ) :
    (L.toArray ++ a)[L.length + t]! = a[t]! := by
  rw [Array.getElem!_eq_getD, Array.getElem!_eq_getD, Array.getD_eq_getD_getElem?, Array.getD_eq_getD_getElem?,
    Array.getElem?_append_right (by simp)]
  simp

/-! ## What a piece's running sum says -/

/-- **What a `rec-open` piece's running sum says** at the inner statement `x`, the opened messages `ms` and the coins
`cs`: at its position on its table's chain (`chainPos` at the schedule `sideAt` reads), the chain's next row's first two
elements are its opening's consistency terms (`consTerm`) of the row it opens (as `sideAt`'s Ligerito proof opens it),
the chain's row at its position, and the `coef` `coefAt` reads off the coins. -/
def accHolds (sideAt : Claims.Msg → List Claims.Msg → OpenPiece → OpenSide)
    (coefAt : Claims.Msg → List Coin → OpenPiece → Array F128)
    (chainAt : Claims.Msg → List Claims.Msg → ℕ → ℕ → Array F128) (q : OpenPiece) (x : Claims.Msg)
    (ms : List Claims.Msg) (cs : List Coin) : Prop :=
  ∀ k < 2, (chainAt x ms q.table (chainPos (sideAt x ms q).sch q.rep q.level q.query + 1))[k]! =
    FlockVBridge.consTerm (2 ^ ((sideAt x ms q).sch.levels[q.level]!).lanes)
      (openAt (sideAt x ms q).lig ((sideAt x ms q).sch.levels.size - 1) q.level).1[q.query]!
      (chainAt x ms q.table (chainPos (sideAt x ms q).sch q.rep q.level q.query)) (coefAt x cs q) k

namespace DecAt

variable {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
  (D : DecAt rd x ms cs) (cls : HClass)

/-! ## The accumulators -/

/-- **Rep `k`'s accumulators**: for `k = 2i + r`, the rows of table `i`'s chain at rep `r`'s ends, `r · N` and
`r · N + N` (`N` the rep's openings at the table's schedule), in `accNames`' order: `acc0 0, acc0 1, acc1 0, acc1 1`. -/
def accE (chainAt : Claims.Msg → List Claims.Msg → ℕ → ℕ → Array F128) (k : ℕ) : Array F128 :=
  #[(chainAt x ms (k / 2) (k % 2 * nOpen ((fast100 (tabsOf D.sts)[k / 2]!.m).getD default)))[0]!,
    (chainAt x ms (k / 2) (k % 2 * nOpen ((fast100 (tabsOf D.sts)[k / 2]!.m).getD default)))[1]!,
    (chainAt x ms (k / 2) (k % 2 * nOpen ((fast100 (tabsOf D.sts)[k / 2]!.m).getD default) +
      nOpen ((fast100 (tabsOf D.sts)[k / 2]!.m).getD default)))[0]!,
    (chainAt x ms (k / 2) (k % 2 * nOpen ((fast100 (tabsOf D.sts)[k / 2]!.m).getD default) +
      nOpen ((fast100 (tabsOf D.sts)[k / 2]!.m).getD default)))[1]!]

theorem accE_size (chainAt : Claims.Msg → List Claims.Msg → ℕ → ℕ → Array F128) (k : ℕ) :
    (D.accE chainAt k).size = 4 := rfl

theorem accE_rep (chainAt : Claims.Msg → List Claims.Msg → ℕ → ℕ → Array F128) (i : ℕ) (r : Fin 2) :
    D.accE chainAt (2 * i + r) =
      #[(chainAt x ms i (r * nOpen (D.files cls i).sch))[0]!, (chainAt x ms i (r * nOpen (D.files cls i).sch))[1]!,
        (chainAt x ms i (r * nOpen (D.files cls i).sch + nOpen (D.files cls i).sch))[0]!,
        (chainAt x ms i (r * nOpen (D.files cls i).sch + nOpen (D.files cls i).sch))[1]!] := by
  have h1 : (2 * i + r.val) / 2 = i := rep_div_two i r
  have h2 : (2 * i + r.val) % 2 = r.val := by omega
  simp only [accE, h1, h2]
  rfl

variable (acc : ℕ → Array F128) (i : ℕ) (r : Fin 2)

/-! ## w's tail -/

/-- **Rep `r`'s frames part has the rounds' elements' count**: the frames absorb the decoded proof's value at each
of the rounds' elements (`vals_repFrames_rounds`), at row C's hypotheses. -/
theorem frames_length (hG : D.RowG cls acc i) (hpts : D.sess.points = pointsOf D.spec D.d.c₁)
    (hk : 2 * i + r < D.d.f.ps.size)
    (hsh : ProofShaped (shapeAt (tabsOf D.sts) (pointsOf D.spec D.d.c₁) (2 * i + r)) D.d.f.ps[2 * i + r]) :
    ((chunksOf (D.ctx (2 * i + r)) D.d.f.ps[2 * i + r]! (D.T (2 * i + r))).flatMap vals).length =
      (rounds (D.sh cls i)).flatten.size := by
  have hext : (tabsOf D.sts)[i]!.extra (pointsOf D.spec D.d.c₁) = .ok (D.files cls i).extras := by
    rw [← hpts]
    exact hG.extra
  have hdec := D.decode_rep cls i r hext hk hsh
  have hp : ProofSizes (D.sh cls i) ((D.files cls i).proof r) := decodeProof_sizes _ _ _ hdec
  have hv := FlockDecodeTies.vals_repFrames_rounds (sh := D.sh cls i) (st := (D.ctx (2 * i + r)).st)
    (sch := (D.ctx (2 * i + r)).sch) (n := (D.ctx (2 * i + r)).n) (D.ctx (2 * i + r)).dom ((D.files cls i).proof r)
    (D.T (2 * i + r)) (by rw [D.ctx_st cls acc i r hG]; rfl)
    (by rw [D.ctx_st cls acc i r hG]; exact hG.foldKLog) (D.ctx_n cls acc i r hG hext) (D.ctx_sch cls acc i r hG)
    rfl hp.r1ab hp.r1c hp.zPartial
    (fun k hk => FlockDecodeTies.rsEach_get hp (by rw [D.ctx_n cls acc i r hG hext] at hk; exact hk))
    (by rw [D.ctx_sch cls acc i r hG]; exact FlockDecodeTies.yr_size hp)
  have hc := FlockDecodeTies.chunks_vals (D.ctx (2 * i + r)).st (D.ctx (2 * i + r)).sch (D.ctx (2 * i + r)).dom
    (D.ctx (2 * i + r)).n ((D.files cls i).proof r) (D.T (2 * i + r))
  have e : (chunksOf (D.ctx (2 * i + r)) D.d.f.ps[2 * i + r]! (D.T (2 * i + r))).flatMap vals =
      (rounds (D.sh cls i)).flatten.toList.map (FlockDecodeTies.keyVal ((D.files cls i).proof r) (D.T (2 * i + r))) :=
    hc.trans hv
  rw [e, List.length_map, Array.length_toList]

/-- **w's tail is rep `r`'s accumulators**: past the rounds' elements, `D.rep`'s w reads `acc (2 i + r)`. -/
theorem w_tail (hG : D.RowG cls acc i) (hpts : D.sess.points = pointsOf D.spec D.d.c₁)
    (hk : 2 * i + r < D.d.f.ps.size)
    (hsh : ProofShaped (shapeAt (tabsOf D.sts) (pointsOf D.spec D.d.c₁) (2 * i + r)) D.d.f.ps[2 * i + r]) (t : ℕ) :
    (D.rep cls acc i r).w[(rounds (D.sh cls i)).flatten.size + t]! = (acc (2 * i + r))[t]! := by
  show (((chunksOf (D.ctx (2 * i + r)) D.d.f.ps[2 * i + r]! (D.T (2 * i + r))).flatMap vals).toArray ++
    acc (2 * i + r))[(rounds (D.sh cls i)).flatten.size + t]! = (acc (2 * i + r))[t]!
  rw [← D.frames_length cls acc i r hG hpts hk hsh]
  exact getElem!_append_length t

/-! ## `ReadsEnds` -/

/-- **V\*'s algebra reads the chain at rep `r`'s ends**, at `acc := accE`: its `acc0`, `acc1` are w's tail
(`wi_acc0`, `wi_acc1`, `w_tail`), which `accE` fills with the chain's rows at `r · N` and `r · N + N`, the openings'
first position and the one past their last (`Openings.start` at the schedule's levels). -/
theorem readsEnds_accE (chainAt : Claims.Msg → List Claims.Msg → ℕ → ℕ → Array F128)
    (hG : D.RowG cls (D.accE chainAt) i) (hpts : D.sess.points = pointsOf D.spec D.d.c₁)
    (hk : 2 * i + r < D.d.f.ps.size)
    (hsh : ProofShaped (shapeAt (tabsOf D.sts) (pointsOf D.spec D.d.c₁) (2 * i + r)) D.d.f.ps[2 * i + r]) :
    ReadsEnds (D.sh cls i) (D.rep cls (D.accE chainAt) i r) (chainAt x ms i)
      (D.opens cls acc i r (r * nOpen (D.files cls i).sch)) := by
  have hh : (D.sh cls i).holo = false := rfl
  have ht := D.w_tail cls (D.accE chainAt) i r hG hpts hk hsh
  have he := D.accE_rep cls chainAt i r
  intro r' hr'
  refine ⟨?_, ?_⟩
  · rw [valOf_acc0 (wi_acc0 hh hr'), ht, he]
    rcases (by omega : r' = 0 ∨ r' = 1) with rfl | rfl <;> rfl
  · rw [valOf_acc1 (wi_acc1 hh hr'), ht, he]
    rcases (by omega : r' = 0 ∨ r' = 1) with rfl | rfl <;> rfl

/-! ## `UnitSums` -/

theorem getD_default {α : Type} [Inhabited α] (a : Array α) (n : ℕ) : a.getD n default = a[n]! :=
  Array.getElem!_eq_getD.symm

/-- **The pieces' running sums chain at rep `r`**: where every query of every level of rep `r` is a piece's
(`hcover`), the pieces of rep `r` read its side (`hside`) and the verifier's `coef` (`hcoef`, the coin tie), and every
piece holds (`accHolds`), the chain `chainAt x ms i` from `r · N` adds each opening's consistency terms (`UnitSums`). -/
theorem unitSums_of_pieces (pieces : List OpenPiece) (sideAt : Claims.Msg → List Claims.Msg → OpenPiece → OpenSide)
    (coefAt : Claims.Msg → List Coin → OpenPiece → Array F128)
    (chainAt : Claims.Msg → List Claims.Msg → ℕ → ℕ → Array F128)
    (hholds : ∀ q ∈ pieces, accHolds sideAt coefAt chainAt q x ms cs)
    (hside : ∀ q ∈ pieces, q.table = i → q.rep = r.val → sideAt x ms q = D.side cls i r)
    (hcoef : ∀ q ∈ pieces, q.table = i → q.rep = r.val → coefAt x cs q = D.coefAt cls acc i r q.level q.query)
    (hcover : ∀ ℓ < (D.files cls i).sch.levels.size, ∀ j < ((D.files cls i).sch.levels[ℓ]!).q,
      ∃ q ∈ pieces, q.table = i ∧ q.rep = r.val ∧ q.level = ℓ ∧ q.query = j) :
    UnitSums (D.sh cls i) (chainAt x ms i) (D.opens cls acc i r (r * nOpen (D.files cls i).sch)) := by
  intro ℓ hℓ j hj k hk
  have hj' : j < ((D.files cls i).sch.levels[ℓ]!).q := by
    have e : (D.opens cls acc i r (r * nOpen (D.files cls i).sch)).q ℓ = ((D.files cls i).sch.levels[ℓ]!).q := by
      show ((D.files cls i).sch.levels.getD ℓ default).q = _
      rw [getD_default]
    exact e ▸ hj
  obtain ⟨q, hq, hi, hr, hl, hqj⟩ := hcover ℓ hℓ j hj'
  have h := hholds q hq k hk
  rw [hside q hq hi hr, hcoef q hq hi hr, hi, hr, hl, hqj] at h
  exact h

/-! ## Row E and the table's run -/

/-- **Row E from the `rec-open` pieces' running sums**, at `acc := accE`: `rowE_of_units` at the chain `chainAt x ms i`
from `r · N`, its `UnitSums` from the pieces (`unitSums_of_pieces`) and its `ReadsEnds` from `accE`
(`readsEnds_accE`). -/
theorem rowE_of_pieces (chainAt : Claims.Msg → List Claims.Msg → ℕ → ℕ → Array F128)
    (hG : D.RowG cls (D.accE chainAt) i) (hpts : D.sess.points = pointsOf D.spec D.d.c₁)
    (hk : 2 * i + r < D.d.f.ps.size)
    (hsh : ProofShaped (shapeAt (tabsOf D.sts) (pointsOf D.spec D.d.c₁) (2 * i + r)) D.d.f.ps[2 * i + r])
    (hfit : ∃ y, verifierRows (D.sh cls i) (D.rep cls (D.accE chainAt) i r).rounds
      (D.rep cls (D.accE chainAt) i r).extras (D.rep cls (D.accE chainAt) i r).comb = .ok y)
    (pieces : List OpenPiece) (sideAt : Claims.Msg → List Claims.Msg → OpenPiece → OpenSide)
    (coefAt : Claims.Msg → List Coin → OpenPiece → Array F128)
    (hholds : ∀ q ∈ pieces, accHolds sideAt coefAt chainAt q x ms cs)
    (hside : ∀ q ∈ pieces, q.table = i → q.rep = r.val → sideAt x ms q = D.side cls i r)
    (hcoef : ∀ q ∈ pieces, q.table = i → q.rep = r.val →
      coefAt x cs q = D.coefAt cls (D.accE chainAt) i r q.level q.query)
    (hcover : ∀ ℓ < (D.files cls i).sch.levels.size, ∀ j < ((D.files cls i).sch.levels[ℓ]!).q,
      ∃ q ∈ pieces, q.table = i ∧ q.rep = r.val ∧ q.level = ℓ ∧ q.query = j) :
    D.RowE cls (D.accE chainAt) i r :=
  D.rowE_of_units cls (D.accE chainAt) i r hG hfit
    (D.unitSums_of_pieces cls (D.accE chainAt) i r pieces sideAt coefAt chainAt hholds hside hcoef hcover)
    (D.readsEnds_accE cls (D.accE chainAt) i r chainAt hG hpts hk hsh)

/-- **Table `i`'s reps run at the decision, with row E from the pieces**: `runs_assembled` at `acc := accE chainAt`,
the chain `chainAt x ms i` from `r · N` for rep `r`, its `UnitSums` from row D's pieces holding `accHolds` (at D's
`sideAt`, `hside` and `hcover`, and the coin tie `hcoef`) and its `ReadsEnds` from `accE`. Every other hypothesis is
`runs_assembled`'s. -/
theorem runs_assembledE {xo : Claims.Msg} {sto : Claims.Recursion.CompiledStatement} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : x = xIOf xo) {i : ℕ} (hi : i < D.sts.size)
    (hti : i < D.spec.tables.size) (chainAt : Claims.Msg → List Claims.Msg → ℕ → ℕ → Array F128)
    (hframed : ∀ r : Fin 2, FramedAt ((D.files cls i).stream r).rounds 0 (repFrames (D.files cls i).st
      (D.files cls i).sch ((D.files cls i).spec.domain (D.files cls i).spec.tables[(D.files cls i).i]! r).toUTF8
      (2 + (D.files cls i).extras.size) ((D.files cls i).proof r)
      (valOf (D.sh cls i) (D.rep cls (D.accE chainAt) i r).w (D.rep cls (D.accE chainAt) i r).comb
        (D.rep cls (D.accE chainAt) i r).cons .T)))
    (hpts : D.sess.points = pointsOf D.spec D.d.c₁)
    (hk : ∀ r : Fin 2, 2 * i + r < D.d.f.ps.size)
    (hsh : ∀ r : Fin 2,
      ProofShaped (shapeAt (tabsOf D.sts) (pointsOf D.spec D.d.c₁) (2 * i + r)) (D.d.f.ps[2 * i + r]'(hk r)))
    (hL : ∀ r, D.RowL cls (D.accE chainAt) i r)
    (pieces : List OpenPiece) (sideAt : Claims.Msg → List Claims.Msg → OpenPiece → OpenSide)
    (posAt : Claims.Msg → List Coin → OpenPiece → ℕ)
    (hholds : ∀ q ∈ pieces, openHolds sideAt posAt q x ms cs)
    (hside : ∀ r : Fin 2, ∀ q ∈ pieces, q.table = i → q.rep = r.val → sideAt x ms q = D.side cls i r)
    (hpos : ∀ r : Fin 2, ∀ v info kappa,
      verifierRows (D.sh cls i) (D.rep cls (D.accE chainAt) i r).rounds (D.rep cls (D.accE chainAt) i r).extras
        (D.rep cls (D.accE chainAt) i r).comb = .ok (v, info, kappa) →
      ∀ q ∈ pieces, q.table = i → q.rep = r.val → posAt x cs q = (info.lg.conss[q.level]!).pos[q.query]!)
    (hcover : ∀ r : Fin 2, ∀ ℓ < (D.files cls i).sch.levels.size, ∀ j < ((D.files cls i).sch.levels[ℓ]!).q,
      ∃ q ∈ pieces, q.table = i ∧ q.rep = r.val ∧ q.level = ℓ ∧ q.query = j)
    {S : RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hS : structureOf (D.sh cls i) = .ok S) (hg : groups S (portSizes (D.sh cls i)) = .ok gs)
    (hsz : ∀ r : Fin 2, (D.rep cls (D.accE chainAt) i r).rounds.toList.map Array.size = roundSizes (D.sh cls i))
    (hw : ∀ r : Fin 2, (D.rep cls (D.accE chainAt) i r).w.size = S.nw)
    (hparts : ∀ r : Fin 2, D.PartsZero cls (D.accE chainAt) i r gs)
    (coefAt : Claims.Msg → List Coin → OpenPiece → Array F128)
    (hholdsE : ∀ q ∈ pieces, accHolds sideAt coefAt chainAt q x ms cs)
    (hcoef : ∀ r : Fin 2, ∀ q ∈ pieces, q.table = i → q.rep = r.val →
      coefAt x cs q = D.coefAt cls (D.accE chainAt) i r q.level q.query) :
    (D.files cls i).Runs :=
  have hG := D.rowG cls (D.accE chainAt) hst hf hx hi
  D.runs_assembled cls (D.accE chainAt) hst hf hx hi hti hframed hpts hk hsh hL pieces sideAt posAt hholds hside
    hpos hcover hS hg hsz hw hparts (fun _ => chainAt x ms i) (fun r => r * nOpen (D.files cls i).sch)
    (fun r => D.unitSums_of_pieces cls (D.accE chainAt) i r pieces sideAt coefAt chainAt hholdsE (hside r) (hcoef r)
      (hcover r))
    (fun r => D.readsEnds_accE cls (D.accE chainAt) i r chainAt hG hpts (hk r) (hsh r))

end DecAt

end RecDeploy.Rows
