import Proofs.Flock.Soundness.Discharge.ZkSession.Session
import Proofs.Flock.Soundness.CompiledSound
import Proofs.Flock.Soundness.ExtractAcc

/-!
# The compiled `--zk` table and session (live/PROTOCOL.md §1, §3, §5, §7; PROTOCOL.md §5, §14)

`sessionZK` (`Discharge/ZkSession/Session.lean`) is the protocol as Lean defines it at the interactive-oracle layer:
`Commit` sends each table's level-0 oracle, each rep receives its pads as two words, level 1's oracle as a table, and
the inner proof's `τ` in the clear. Here is the protocol as run, with every oracle a commitment:

* `Commit` sends each table's **witness cap**, the 68-lane level-0 tree (`2^k₀` witness lanes and the reps' `2e` extra
  lanes, one tree);
* each rep (`repZC`) starts with the **pads root** (`recv D`, §3), runs the masked algebraic phase as in `repMaskedZK`,
  and level 0 masked with Ligerito's levels compiled (`ligeritoPadZC`: level 1 onward commit by caps, and level 0's
  consistency value is computed from the opened 68-lane rows, `padColumnO`), then `final_c'`;
* the **inner proof** (`innerGameC`, §5) sends `τ`'s commitment (`recv D`) where the interactive-oracle layer sends `τ`;
* after the last coin of both reps, one message (`OpensZ`) carries every opening: the queried rows of every level with
  their salts and paths, and per rep the queried pads pairs `(C_h[j], C_mu[j])` with their salts and paths, and `τ`
  with its salt.

The table accepts (`tabAfterZC`) when both reps' Ligerito verdicts accept on the opened rows, both inner checks hold on
the opened pads and `τ` (`innerCheckC`: `τ` opens its commitment, the batched constraint with the opened `τ`, the code
identity at every query, and every pads path to the pads root), and every row path verifies from its row's leaf under
its salt, level 0's rows being 68 lanes wide (`OpensOKZ`).

`sessionZG` is `Exec.sessionG` over `--zk` tables: `sessionZK` is its instance at `TabZK.after` (`sessionZK_eq`), and the
compiled session `sessionZC` its instance at `tabAfterZC`. `batchedSessionZC` is the compiled session on a draw, as
`batchedSessionZK` is the interactive-oracle one.

**Extraction** (as `SessionBatchAcc.lean` at M0): table `j`'s extractor fills its level-0 oracle from the verifying
level-0 openings of `k` reruns of everything after `Commit` in which the session accepts (`extractTableZ`), and
`extractProbAccZ` is the probability of an event of that oracle over those reruns.
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting ZkSession
open scoped ENNReal NNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type}

/-! ## Level 0, compiled -/

/-- **The padded level-0 column from an opened row**: `Model.padColumn` with the row's lanes for the oracle's. -/
def padColumnO (A : Arith F K) (d cols t : ℕ) (c₀ : F) (ch : List K) {e : ℕ} (ρ : Fin e → K) (xl : Fin e → ℕ)
    (ybar : Fin t → K) (row : List F) (p : ℕ) : K :=
  foldColumn A ch (fun ℓ => row.getD ℓ 0) + ∑ i, ρ i * A.embed (row.getD (xl i) 0) + padCorr A d cols t c₀ ybar p

/-- **Rep `r`'s level-0 queries, compiled** (`queryAndBatchV` with caps): the consistency value is the batched padded
column of the rows the openings give, and the queries are recorded. -/
def queryAndBatchPadC (A : Arith F K) (r : ℕ) (lv : Level) (col : List F → ℕ → K) (start : ℕ)
    (st : LigStateC F K D) : Game (LigStateC F K D) := do
  let w ← draw (Fin lv.queries → F)
  let α ← draw (Fin (ceilLog2 lv.queries) → F)
  let pos := positions A lv.queries lv.logLen w
  let e : Opens F D → K := fun o =>
    ((pos.zipIdx).map fun pj => A.embed (eqList (List.ofFn α) pj.2) * col (o r st.level pj.2).1 pj.1).sum
  let msg ← recv (K × K)
  let β ← draw F
  let ctx : ConsCtx F := ⟨lv.logCols, pos, List.ofFn α, β, start⟩
  pure { st with
    s := fun o => st.s o + A.embed β * e o
    q := fun o => (st.q o).addScaled (fromMsg msg (e o)) (A.embed β)
    cons := st.cons ++ [ctx]
    reqs := st.reqs ++ [⟨st.level, pos, lv.lanes, pathLen lv⟩] }

/-- **Levels 1+ of the masked level 0, compiled** (`ZkSession.ligTail` with caps): the code-switch message, level 1's
cap, its OOD samples, level 0's padded queries on the opened rows, and the recursive rounds. -/
def ligTailC (A : Arith F K) (r : ℕ) (sch : Schedule) (t : ℕ) (c₀ : F) (e : ℕ) (b : ℕ → F) (lv₀ : Level)
    (rest : List Level) (z₀ : Fin (sch.m - 7) → F) (β₀ : F) (initial : List K) (ρ : Fin e → K) (ybar : Fin t → K)
    (s : K) : Game (PendL F D) := do
  let n := sch.m - 7
  let msg' ← recv (K × K)
  let q := fromMsg msg' s
  match rest with
  | [] => pure PendL.reject
  | lv₁ :: _ => do
      let C₁ ← recv (Cap D)
      let dim := n - sch.k0 + 1
      let (s, q, oods) ← twoOodC (D := D) A dim 0 (fun _ => s) (fun _ => q)
      let st : LigStateC F K D := ⟨s, q, dim, [], ⟨List.ofFn z₀, β₀, none⟩ :: oods, [], 0, [], [], lv₀⟩
      let st ← queryAndBatchPadC A r lv₀
        (padColumnO A lv₀.logLen lv₀.logCols t c₀ initial ρ (extraLane sch.k0 e r) ybar) 0 st
      recursiveRoundsC A r n b initial sch.ext rest { st with level := 1, caps := [C₁], prevLevel := lv₁ }

/-- **M1's masked level 0, compiled** (`ligeritoPadZK` with `ligTailC`): the lane-phase row and the pending verdict of
levels 1+. -/
def ligeritoPadZC (A : Arith F K) {np e : ℕ} (P : Pads0 np e) (sch : Schedule) (t : ℕ) (c₀ : F) (r : ℕ) (T : F)
    (b : ℕ → F) : Game (LinK K np × PendL F D) := do
  let n := sch.m - 7
  match sch.levels with
  | [] => pure (0, PendL.reject)
  | lv₀ :: rest => do
      let z₀ ← draw (Fin n → F)
      let y₀ ← recv F
      let β₀ ← draw F
      let s : LinK K np := cK (A.embed T) + A.embed β₀ • (cK (A.embed y₀) + padF P.ood)
      let msg ← recv (K × K)
      let (initial, s) ← foldRoundsZK A P sch.k0 0 s (fromMsgL (unmask A P 0 msg) s)
      let ev ← recv (Fin e → K)
      let ρ ← drawKs A e
      let T' ← recv K
      let ybar ← recv (Fin t → K)
      let row := s + ∑ i, ρ i • (cK (ev i) + padK A (P.ev i 0) (P.ev i 1)) - cK T'
      let pend ← ligTailC A r sch t c₀ e b lv₀ rest z₀ β₀ initial ρ ybar T'
      pure (row, pend)

/-- A phase's rows with a row in front, whatever the phase's verdict. -/
def consRowC {np n : ℕ} {β : Type} (row : Lin F np) (o : (Fin n → Lin F np) × β) : (Fin (n + 1) → Lin F np) × β :=
  (Fin.cons row o.1, o.2)

/-- **The rest of a masked rep, compiled** (`restZK` with `ligeritoPadZC`). -/
noncomputable def restZC (A : Arith F K) {np e : ℕ} (P₀ : Pads0 np e) (PA : PadsA np) (S : Statement)
    (sch : Schedule) (hm : 13 ≤ S.m) (t : ℕ) (c₀ : F) (r : ℕ) (extra : List (Claim F S.m))
    (zc : ZcZK F np S.m) (ab : ClaimZK F np S.m) : Game ((Fin 5 → Lin F np) × PendL F D) := do
  let o ← openingZK A (by omega) ab.pub ⟨0, A.lagS zc.z, zc.r⟩ extra
  match o.1 with
  | none => pure (fun _ => ⟨0, 0⟩, PendL.reject)
  | some op => do
      let lv ← ligeritoPadZC (D := D) A P₀ sch t c₀ r op.T op.b
      let fc ← recv F
      pure (![(ab.value + cF o.2.1).toLin, (zc.vc + mskF PA.fc fc).toLin, (mskF PA.fc fc + cF o.2.2).toLin,
        limbRows A lv.1 0, limbRows A lv.1 1], lv.2)

/-- **flock-verify's order of a masked rep's level-0 rows**: `c_eval` (row `3` of `repMaskedZK`'s) first, then `a·b`,
the lincheck final and `rs ab`, the rest kept. -/
def rowOrder : Equiv.Perm (Fin 7) where
  toFun := ![3, 0, 1, 2, 4, 5, 6]
  invFun := ![1, 2, 3, 0, 4, 5, 6]
  left_inv := by decide
  right_inv := by decide

/-- **A masked rep, compiled** (`repMaskedZK` with `restZC`): its seven rows, in flock-verify's batching order (`c_eval`
first, then `a·b`, the lincheck final and `rs ab`), and its pending Ligerito verdict. -/
noncomputable def repMaskedZC (A : Arith F K) {np e : ℕ} (P₀ : Pads0 np e) (PA : PadsA np) (tr : Triple np)
    (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (t : ℕ) (c₀ : F) (r : ℕ) (extra : List (Claim F S.m)) :
    Game ((Fin 7 → Lin F np) × PendL F D) :=
  ((zerocheckZK A S.m PA tr).bind fun o =>
    ((lincheckZK A S PA o.1).bind fun o' =>
      (restZC (D := D) A P₀ PA S sch hm t c₀ r extra o.1 o'.1).map (consRowC o'.2.toLin)).map (consRowC o.2.toLin)).map
    fun o => (o.1 ∘ ![3, 0, 1, 2, 4, 5, 6], o.2)

/-! ## The inner proof, compiled -/

/-- **A rep's inner-proof openings**: per query, the pads pair `(C_h[j], C_mu[j])` with its salt and its path to the
pads root; and `τ` with its salt (§5, step 4). -/
structure InnerOpens (F D Sl : Type) where
  pads : ℕ → (F × F) × Sl × List D
  tau : F × Sl

/-- **The inner proof's final check, compiled** (§5, step 5): `τ` opens its commitment `cτ`, the batched constraint holds
with the opened `τ`, and at every query the code identity holds on the opened pads pair, whose path verifies from its
leaf under its salt to the pads root. -/
def innerCheckC (H : List UInt8 → D) (Ep : Merkle.Enc (F × F) D) (Lp : Merkle.Leaf (F × F) Sl D H)
    (Lτ : Merkle.Leaf F Sl D H) {N : ℕ} (dom : Fin N ↪ F) {s nP q : ℕ} (pos : F → Fin N) {nA : ℕ}
    (rows : Fin nA → Lin F s) (tr : Triple s) (root cτ : D) (ε : F) (ρσ : F × F) (γ : Fin (nA + 3) → F) (β : F)
    (y : Fin (s + nP) → F) (w : Fin q → F) (io : InnerOpens F D Sl) : Prop :=
  let R := Fin.append rows (tripleRows tr ε ρσ.1 ρσ.2)
  Lτ.leaf io.tau.1 io.tau.2 = cτ ∧
    (∑ i, batchCoef γ R i * head y i = batchT γ R + β * io.tau.1) ∧
    ∀ j : Fin q, enc dom y (pos (w j)) = (io.pads j).1.1 + β * (io.pads j).1.2 ∧
      Merkle.VerifiesL H Ep Lp (fun _ => root) (Nat.log 2 N) (pos (w j)).val (io.pads j).1 (io.pads j).2.1
        (io.pads j).2.2

/-- **The inner proof, compiled** (`innerGame` with `τ` committed, §5): `ε`, `[ρ, σ]`, `γ`, `τ`'s commitment, `β`, `Y`
and the queries. Its verdict is a function of the openings, sent after the last coin. -/
def innerGameC (H : List UInt8 → D) (Ep : Merkle.Enc (F × F) D) (Lp : Merkle.Leaf (F × F) Sl D H)
    (Lτ : Merkle.Leaf F Sl D H) {N : ℕ} (dom : Fin N ↪ F) (s nP q : ℕ) (pos : F → Fin N) {nA : ℕ}
    (rows : Fin nA → Lin F s) (tr : Triple s) (root : D) : Game (InnerOpens F D Sl → Prop) := do
  let ε ← draw F
  let ρσ ← recv (F × F)
  let γ ← draw (Fin (nA + 3) → F)
  let cτ ← recv D
  let β ← draw F
  let y ← recv (Fin (s + nP) → F)
  let w ← draw (Fin q → F)
  pure (innerCheckC H Ep Lp Lτ dom pos rows tr root cτ ε ρσ γ β y w)

/-! ## The table -/

/-- **The commitments' hash and leaf schemes**: `H` and its node encoding `E` for every tree, the row leaf `Lf` (the
68-lane level-0 rows and every level's rows), the pads-pair leaf `Lp` (with `Ep`, its tree's encoding) and `τ`'s leaf
`Lτ`. Under `--zk` all three leaves are `hm96-sha512/v1` (`Merkle.Leaf.hm96`). -/
structure Hashes (F D Sl : Type) where
  H : List UInt8 → D
  E : Merkle.Enc (List F) D
  Lf : Merkle.Leaf (List F) Sl D H
  Ep : Merkle.Enc (F × F) D
  Lp : Merkle.Leaf (F × F) Sl D H
  Lτ : Merkle.Leaf F Sl D H

/-- **A compiled `--zk` rep's result**, pending the openings: the Ligerito levels' verdict, caps and requests, and the
inner proof's check. -/
structure PendZ (F D Sl : Type) where
  lig : PendL F D
  inner : InnerOpens F D Sl → Prop

/-- A rep that rejected. -/
def PendZ.reject : PendZ F D Sl := ⟨PendL.reject, fun _ => False⟩

/-- **A compiled `--zk` rep** (`repZK` compiled): the pads root, the masked phase compiled, and the compiled inner proof
on the phase's rows. -/
noncomputable def repZC (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts) (r : ℕ)
    (extra : List (Claim F T.S.m)) : Game (PendZ F D Sl) := do
  let root ← recv D
  let o ← repMaskedZC (D := D) A T.pads T.padsA T.tr T.S T.sch T.hm T.t T.c₀ r extra
  let v ← innerGameC Hs.H Hs.Ep Hs.Lp Hs.Lτ T.dom T.np T.nP T.q T.pos o.1 T.tr root
  pure ⟨o.2, v⟩

/-- **The final message of a compiled `--zk` table**, after the last coin of both reps: the rows of every level with
their salts and paths (`Opens`, `Salts`, per rep, level and query), and each rep's inner-proof openings. -/
structure OpensZ (F D Sl : Type) where
  rows : Opens F D
  salts : Salts Sl
  inner : ℕ → InnerOpens F D Sl

open Classical in
/-- **Every requested row of a rep verifies**, as `OpensOKL`, level 0's rows `w₀` lanes wide (the witness lanes and every
rep's extra lanes, one tree) and level `ℓ ≥ 1`'s `2^lanes`. -/
def OpensOKZ (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (w₀ : ℕ)
    (cap₀ : Cap D) (r : ℕ) (p : PendL F D) (o : Opens F D) (sl : Salts Sl) : Prop :=
  ∀ req ∈ p.reqs, ∃ cap, capOf cap₀ p.caps req.level = some cap ∧
    ∀ j < req.positions.length, (o r req.level j).1.length = (if req.level = 0 then w₀ else 2 ^ req.lanes) ∧
      Merkle.VerifiesL H E Lf cap req.depth (req.positions.getD j 0) (o r req.level j).1 (sl r req.level j)
        (o r req.level j).2

/-- Level 0's row width under `--zk`: `2^k₀` witness lanes and both reps' `e` extra lanes. -/
def width₀ (T : TabZK F mPts) : ℕ := 2 ^ T.sch.k0 + 2 * T.e

/-- **The compiled `--zk` table's output**: the verdict, the witness cap, both reps' results, and the openings. -/
structure OutZC (F D Sl : Type) where
  accepted : Bool
  cap₀ : Cap D
  pend : PendZ F D Sl × PendZ F D Sl
  opens : OpensZ F D Sl

/-- **Rep `r` of a compiled `--zk` table at the link points `pts`** (`Refine.modelTable`'s `G`). -/
noncomputable def repZCAt (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts) (pts : Fin 2 → Fin mPts → F)
    (r : Fin 2) : Game (PendZ F D Sl) :=
  repZC A Hs T r.val (extraClaims T.S T.ptLocal mPts pts)

open Classical in
/-- **A compiled `--zk` table's verdict and output** (`Refine.modelTable`'s `out`): accepted iff both reps' Ligerito
verdicts accept on the opened rows, both inner checks hold, and every row path verifies (`OpensOKZ`). -/
noncomputable def outZC (Hs : Hashes F D Sl) (T : TabZK F mPts) (cap₀ : Cap D) (pend : PendZ F D Sl × PendZ F D Sl)
    (os : OpensZ F D Sl) : OutZC F D Sl :=
  ⟨pend.1.lig.verdict os.rows && pend.2.lig.verdict os.rows &&
      decide (pend.1.inner (os.inner 0) ∧ pend.2.inner (os.inner 1) ∧
        OpensOKZ Hs.H Hs.E Hs.Lf (width₀ T) cap₀ 0 pend.1.lig os.rows os.salts ∧
        OpensOKZ Hs.H Hs.E Hs.Lf (width₀ T) cap₀ 1 pend.2.lig os.rows os.salts),
    cap₀, pend, os⟩

/-- **A compiled `--zk` table after its witness cap `cap₀` and the link points `pts`**: both reps concurrently, then the
openings, then the verdict (`outZC`). -/
noncomputable def tabAfterZC (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts) (cap₀ : Cap D)
    (pts : Fin 2 → Fin mPts → F) : Game (OutZC F D Sl) :=
  (interleave (repZCAt A Hs T pts 0) (repZCAt A Hs T pts 1)).bind fun pend =>
    Game.send (OpensZ F D Sl) fun os => Game.ret (outZC Hs T cap₀ pend os)

instance instInhabitedOpensZ [Inhabited D] [Inhabited Sl] : Inhabited (OpensZ F D Sl) :=
  ⟨⟨fun _ _ _ => ([], []), fun _ _ _ => default, fun _ => ⟨fun _ => ((0, 0), default, []), (0, default)⟩⟩⟩

/-- A default output, for positions outside a batch. -/
def dfltOutZC [Inhabited D] [Inhabited Sl] : OutZC F D Sl :=
  ⟨false, fun _ => default, (PendZ.reject, PendZ.reject),
    ⟨fun _ _ _ => ([], []), fun _ _ _ => default, fun _ => ⟨fun _ => ((0, 0), default, []), (0, default)⟩⟩⟩

/-! ## The session -/

/-- **A batched session over `--zk` tables and any table run** (PROTOCOL.md §5; `Exec.sessionG` over `TabZK`): `Commit`
sends every table's commitment `Cm`, then one link-point draw, then every table's run `tab` after them, concurrently. -/
def sessionZG {Cm Out : Type} (tab : TabZK F mPts → Cm → (Fin 2 → Fin mPts → F) → Game Out)
    (ts : List (TabZK F mPts)) : Game (List Out) :=
  Game.send (Fin ts.length → Cm) fun cms =>
    Game.coin (Fin 2 → Fin mPts → F) fun pts => batch ((List.ofFn fun i => tab (ts.get i) (cms i)).map (· pts))

omit [Fintype K] in
/-- The interactive-oracle session is `sessionZG` at `TabZK.after`. -/
theorem sessionZK_eq (A : Arith F K) (ts : List (TabZK F mPts)) :
    sessionZK A ts = sessionZG (fun T C pts => T.after A C pts) ts := by
  unfold sessionZK sessionZG
  simp only [List.map_ofFn]
  rfl

/-- The compiled tables' runs after `Commit`, with the caps it sent. -/
noncomputable def gamesZC (A : Arith F K) (Hs : Hashes F D Sl) (ts : List (TabZK F mPts)) (caps : Fin ts.length → Cap D) :
    List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl)) :=
  List.ofFn fun i => tabAfterZC A Hs (ts.get i) (caps i)

/-- After `Commit`: one link-point draw, then the games concurrently. -/
noncomputable def sessAfterZC (fs : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl))) :
    Game (List (OutZC F D Sl)) :=
  Game.coin (Fin 2 → Fin mPts → F) fun pts => batch (fs.map (· pts))

/-- **The compiled batched `--zk` session**: `Commit` sends every table's witness cap, then one link-point draw, then
every table's compiled reps and openings concurrently. -/
noncomputable def sessionZC (A : Arith F K) (Hs : Hashes F D Sl) (ts : List (TabZK F mPts)) :
    Game (List (OutZC F D Sl)) :=
  Game.send (Fin ts.length → Cap D) fun caps => sessAfterZC (gamesZC A Hs ts caps)

omit [Fintype K] in
/-- The compiled session is `sessionZG` at `tabAfterZC`, as `sessionZK` is at `TabZK.after`. -/
theorem sessionZC_eq (A : Arith F K) (Hs : Hashes F D Sl) (ts : List (TabZK F mPts)) :
    sessionZC A Hs ts = sessionZG (tabAfterZC A Hs) ts :=
  rfl

/-- The compiled session accepts when every table accepts. -/
def acceptedZC [Inhabited D] [Inhabited Sl] (ts : List (TabZK F mPts)) (outs : List (OutZC F D Sl)) : Prop :=
  ∀ i : Fin ts.length, (outs.getD i dfltOutZC).accepted = true

section Batched

variable [Inhabited D] [Inhabited Sl] (A : Arith F K) (Hs : Hashes F D Sl) {n : ℕ} {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))

open Classical in
/-- **The compiled batched `--zk` session on a draw**: the plan's tables in one compiled session; it accepts when every
table accepts. -/
noncomputable def batchedSessionZC (S : Finset (Fin n)) (R : Reg) : Game Bool :=
  (sessionZC A Hs (planZ S R)).map fun outs => decide (acceptedZC (planZ S R) outs)

/-- The strategy in the compiled session inside a verdict strategy. -/
noncomputable def zcStrat {S : Finset (Fin n)} {R : Reg} (τ : Strategy (batchedSessionZC A Hs planZ S R)) :
    Strategy (sessionZC A Hs (planZ S R)) :=
  Strategy.ofMap _ _ τ

end Batched

/-! ## Extraction from accepted reruns -/

/-- Rep `r`'s first level-`ℓ` request's openings, `(position, row, salt, path)` for its first `Q` queries. -/
def slotOpensZ (out : OutZC F D Sl) (r ℓ Q : ℕ) : List (ℕ × List F × Sl × List D) :=
  match ((if r = 0 then out.pend.1 else out.pend.2).lig.reqs.find? fun req => req.level == ℓ) with
  | none => []
  | some req => ((req.positions.take Q).zipIdx).map fun pj =>
      (pj.1, (out.opens.rows r ℓ pj.2).1, out.opens.salts r ℓ pj.2, (out.opens.rows r ℓ pj.2).2)

/-- Both reps' level-0 openings. -/
def opens₀Z (sch : Schedule) (out : OutZC F D Sl) : List (ℕ × List F × Sl × List D) :=
  slotOpensZ out 0 0 (lvOf sch 0).queries ++ slotOpensZ out 1 0 (lvOf sch 0).queries

/-- An opening verifies against `cap`: a row of `w` elements whose path verifies from its leaf under its salt. -/
def VerifiedZ (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (cap : Cap D)
    (w depth : ℕ) (x : ℕ × List F × Sl × List D) : Prop :=
  x.2.1.length = w ∧ Merkle.VerifiesL H E Lf cap depth x.1 x.2.1 x.2.2.1 x.2.2.2

open Classical in
/-- **The witness cap's openings, by position**: at each level-0 position, the first verifying 68-lane row a run opened
there. -/
noncomputable def op₀Z (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (T : TabZK F mPts) (cap : Cap D) (out : OutZC F D Sl) :
    Fin (2 ^ (lvOf T.sch 0).logLen) → Option (Fin (width₀ T) → F) := fun p =>
  (((opens₀Z T.sch out).filter fun x =>
      decide (x.1 = p.val ∧ VerifiedZ H E Lf cap (width₀ T) (pathLen (lvOf T.sch 0)) x)).head?).map
    fun x lane => x.2.1.getD lane.val 0

/-- The oracle with the given `w`-lane rows on a level's domain, zero elsewhere. -/
def tabOfW (w d : ℕ) (v : Fin (2 ^ d) → Fin w → F) : Oracle F :=
  fun lane q => if h : q < 2 ^ d ∧ lane < w then v ⟨q, h.1⟩ ⟨lane, h.2⟩ else 0

/-- **The extractor's level-0 oracle of table `n`** from runs `bs` after `Commit` with witness cap `cap₀`: at each
level-0 position, the first verifying opening one of the runs made there, and zero elsewhere. -/
noncomputable def extractTableZ [Inhabited D] [Inhabited Sl] (Hs : Hashes F D Sl) (T : TabZK F mPts) (n : ℕ)
    (cap₀ : Cap D) (bs : List (List (OutZC F D Sl))) : Oracle F :=
  tabOfW (width₀ T) (lvOf T.sch 0).logLen
    (Extract.firstTable (fun outs : List (OutZC F D Sl) => op₀Z Hs.H Hs.E Hs.Lf T cap₀ (outs.getD n dfltOutZC))
      (fun _ _ => 0) bs)

section Acc

variable [Inhabited D] [Inhabited Sl] (A : Arith F K) (Hs : Hashes F D Sl) (ts : List (TabZK F mPts))

/-- The probability that the compiled session accepts, after `Commit`. -/
noncomputable def accProbZC (τ : Strategy (sessionZC A Hs ts)) : ℝ :=
  expect (ind (acceptedZC ts)) (sessAfterZC (gamesZC A Hs ts τ.1)) τ.2

open Classical in
/-- **An event of table `j`'s extraction, from accepted reruns**: the probability, over `k` reruns of everything after
`Commit` in which the compiled session accepts, that the level-0 oracle the extractor fills from table `j`'s openings
satisfies `G`. -/
noncomputable def extractProbAccZ (j : Fin ts.length) (k : ℕ) (τ : Strategy (sessionZC A Hs ts))
    (G : Oracle F → Prop) : ℝ :=
  Extract.expectAcc (sessAfterZC (gamesZC A Hs ts τ.1)) τ.2 (acceptedZC ts) k
    (ind fun bs => G (extractTableZ Hs (ts.get j) j (τ.1 j) bs))

/-- **Table `j`'s extraction from accepted reruns fails**: no padded witness near the extracted oracle satisfies the
table's statement. -/
noncomputable def failProbAccZ (j : Fin ts.length) (k : ℕ) (τ : Strategy (sessionZC A Hs ts)) : ℝ :=
  extractProbAccZ A Hs ts j k τ fun C => ¬ (ts.get j).Committed A C

/-- **The rate of table `j`'s relative term**, `N₀/(ek)`, `N₀` its level-0 positions. -/
noncomputable def rateZ (j : Fin ts.length) (k : ℕ) : ℝ :=
  2 ^ (lvOf (ts.get j).sch 0).logLen / (Real.exp 1 * k)

end Acc

end FlockSoundness.Discharge.ZkLink
