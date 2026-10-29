import Proofs.Flock.Claims.Recursion.VerdictAssembly
import Proofs.Flock.Claims.KS.PlainDrawJRun
import Proofs.Flock.VBridge.Family.Tape

/-!
# Row L at a decision: the fields the reader fixes

`RowL` asks four things of table `i`'s rep `r` at a decision of the last step. The decision's proofs are `nextR`'s
reader's (`readPs`), which takes every field the frames don't absorb as V\* takes it (`absorbedFrom`), so three of the
four hold by the reader's definition:

* **What the reader pins** (`Pinned`, `readRep_pinned`, `readPs_pinned`, `decR_pinned`): each decided proof's `params`
  is the pinned `pcsParams` of its table's scheme and schedule, its Ligerito `initialCap` is its cap, and its
  zerocheck's `finalC` is `round1c` at the Lagrange weights of its context's `z` (the live coin of the rep's round 2).
* **`params`** (`rowL_params`) and **`initialCap`** (`rowL_initialCap`): from the pin, at row G's setup.
* **`fc`** (`rowL_fc`): V\*'s `v_C` is `round1c`'s symbolic messages at the Lagrange weights of the zerocheck's `z`
  (`zerocheck_vc`), so evaluated at `w` it is `decoded`'s `c` at those weights (`vc_eval`), and its `z` is the first
  coin of the rep's round 2 (`zc_z`). With row C's `c` (the proof's `round1c` is `decoded`'s) it is the pinned
  `finalC` exactly where the stream's round-2 coin is the live one the reader read `z` from (`hz`).
* **`cap`** (`rowL_cap`): the proof's cap is its table's level-0 nodes among round 0's rows (`FitsRep.cap`), which the
  link joins into the table's `root_B` (`decR_rootB`), so the cap joined is the session's `root_B` exactly where the
  session's `root_B` is the link's (`hroot`). The `after` check gives that (`rootB_link`): its custody compares
  `liveOfJ`'s `root_B` with the link's, and the decoder reads `liveOfJ`'s (`decodeJWithK_live`).

`rowL_of` is the row; `rowL_at` is it at the checks where the rows are asked, with `hz` the one hypothesis left;
`runs_assembled_L` is `runs_assembled` with `hL` replaced by the `after` check and `hz`.
-/

namespace FlockDeploy

open Flock

/-- **What the reader pins of a rep's proof outside its frames**: the pinned PCS parameters of the context's scheme
and schedule, the cap as the Ligerito initial cap, and `finalC` as V\* takes `v_C`, at the context's `z`. -/
structure Pinned (c : RepCtx) (p : Flock.Proof) : Prop where
  params : p.params = pcsParams c.st.merkle c.sch
  initialCap : p.ligerito.initialCap = p.cap
  finalC : p.zerocheck.finalC = dot (lagrange nodesL den6Inv c.z) p.zerocheck.round1c

theorem pinned_absorbedFrom (c : RepCtx) (a q : Flock.Proof) : Pinned c (absorbedFrom c a q) :=
  ⟨rfl, rfl, rfl⟩

theorem readRep_pinned {c : RepCtx} {rows : List (List ByteArray)} {q p : Flock.Proof} {ms : List Claims.Msg}
    (h : readRep c rows q = some (p, ms)) : Pinned c p := by
  by_cases hx : ∃ pT : Flock.Proof × F128, FitsRep c rows pT.1 pT.2
  · rw [readRep, dif_pos hx] at h
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj h)
    exact pinned_absorbedFrom c _ q
  · rw [readRep, dif_neg hx] at h
    cases h

/-- **Every proof `readPs` reads is pinned** at its rep's context. -/
theorem readPs_pinned {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin} {r₀ : List ByteArray}
    {rss : List (List ByteArray)} {cs : List Coin} {fl : FinalP} {pf : Array Flock.Proof × List Claims.Msg}
    (h : readPs spec sts c₁ r₀ rss cs fl = some pf) :
    ∀ (k : ℕ) (hk : k < pf.1.size), Pinned (repCtx spec sts c₁ r₀ cs k) pf.1[k] := by
  unfold readPs at h
  by_cases hc : fl.ps.size = 2 * sts.size ∧ (repLens spec sts).sum = rss.length
  · rw [if_pos hc] at h
    obtain ⟨rs, hrs, rfl⟩ := Option.map_eq_some_iff.1 h
    obtain ⟨hlen, hget⟩ := mapM_some_get hrs
    rw [List.length_range] at hlen
    intro k hk
    have hk' : k < rs.length := by simpa using hk
    have hg := hget k (by rw [List.length_range]; omega) hk'
    rw [List.getElem_range] at hg
    have hp := readRep_pinned (p := rs[k].1) (ms := rs[k].2) hg
    simpa using hp
  · rw [if_neg hc] at h
    cases h

/-- **At a decision, every rep's proof is pinned** at its context, whatever round 0's rows the context reads (the pin
reads only its setup, schedule and `z`). -/
theorem decR_pinned {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {d : DecR} (h : decR rd x as cs = some d) (r₀ : List ByteArray) :
    ∀ (k : ℕ) (hk : k < d.f.ps.size), Pinned (repCtx d.spec d.sts d.c₁ r₀ cs.tail.dropLast k) d.f.ps[k] := by
  obtain ⟨v, c₁, r₀', rss, fl, link, pf, -, -, -, -, -, h9, rfl⟩ := decR_some h
  intro k hk
  have hp := readPs_pinned h9 k hk
  exact ⟨hp.params, hp.initialCap, hp.finalC⟩

end FlockDeploy

namespace Flock.RecAlgebra

open Flock

/-- **The zerocheck's `v_C`** is `round1c`'s messages at the Lagrange weights of its `z`. -/
theorem zerocheck_vc {α : Type} [Val α] {m : Nat} {M : Messages α} {k k' : Nat} {src : Coins} {r0 : α}
    {zc : ZcPts α} (h : zerocheck m M k src = .ok ((r0, zc), k')) :
    zc.vc = dotV (lagrange nodesL den6Inv zc.z) M.c := by
  unfold zerocheck at h
  obtain ⟨_, k₁, -, h⟩ := cm_bind h
  obtain ⟨rOut, k₂, -, h⟩ := cm_bind h
  obtain ⟨z, k₃, -, h⟩ := cm_bind h
  obtain ⟨⟨claim, rho⟩, k₄, -, h⟩ := cm_bind h
  obtain ⟨ab, k₅, -, h⟩ := cm_bind h
  cases cm_pure h
  rfl

/-- **V\*'s `v_C` at `w`** is the `c` messages `w` holds (`decoded`) at the Lagrange weights of the zerocheck's `z`. -/
theorem vc_eval {sh : Shape} {rounds : Array (Array F128)} {extras : Array Claim} {comb v : Array F128}
    {info : Info Lin} {kappa : Array (Array (Array F128))}
    (hrows : verifierRows sh rounds extras comb = .ok (v, info, kappa)) (w comb' : Array F128)
    (cons : Nat → Nat → F128) :
    info.zc.vc.eval (valOf sh w comb' cons) = dot (lagrange nodesL den6Inv info.zc.z) (decoded sh w comb' cons).c := by
  obtain ⟨lins, k, hrun⟩ := FlockVBridge.Runs.run_of_rows hrows
  obtain ⟨-, _, _, _, _, _, _, _, h0, -⟩ := run_stages hrun
  rw [zerocheck_vc h0, (Lin.eval_hom (valOf sh w comb' cons)).dotV, ← dotV_eq_dot]
  rfl

/-- **The zerocheck's `z` is the first coin of round 2**, at rounds none of which is empty, where `m ≥ 14`. -/
theorem zc_z {sh : Shape} {R : Array Flock.Round} {extras : Array Claim} {comb v : Array F128}
    {info : Info Lin} {kappa : Array (Array (Array F128))}
    (hrows : verifierRows sh (R.map (·.coins)) extras comb = .ok (v, info, kappa)) (hm : 14 ≤ sh.m)
    (hne : NoEmpty (R.map (·.coins))) : info.zc.z = (R[2]!).coins.getD 0 0 := by
  obtain ⟨lins, k, hrun⟩ := FlockVBridge.Runs.run_of_rows hrows
  obtain ⟨-, _, _, _, _, _, _, _, h0, -⟩ := run_stages hrun
  obtain ⟨_, _, c₂, -, -, hc₂, -, hz, -⟩ := zerocheck_coins hm h0
  rw [hz, FlockVBridge.Runs.coin_one hne hc₂ rfl]

end Flock.RecAlgebra

namespace RecDeploy.Rows.DecAt

open Flock
open FlockDeploy (FinalP Coin tabsOf repCtx Pinned decR_pinned decR_fits decR_rootB cutLens level0Lens pointsOf
  shapeAt)
open Flock.RecAlgebra (verifierRows decoded valOf NoEmpty)
open FlockVBridge.Family (ProofShaped FramedAt roundSizes)
open FlockVBridge.AccChain (UnitSums ReadsEnds)
open Flock.RecResiduals (portSizes groups)
open FlockSoundness.Discharge.Exec (repFrames)
open FlockSoundness.Discharge.Recursion (HClass)

local instance lawfulBEqByteArrayL : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := by show (a.data == a.data) = true; exact beq_self_eq_true _

variable {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
  (D : DecAt rd x ms cs) (cls : HClass) (acc : ℕ → Array F128) (i : ℕ) (r : Fin 2)

/-- **Rep `r` of table `i`'s proof is pinned** at its context. -/
theorem pinned (hk : 2 * i + r < D.d.f.ps.size) : Pinned (D.ctx (2 * i + r)) ((D.files cls i).proof r) := by
  have h := decR_pinned D.dec D.r₀ (2 * i + r) hk
  show Pinned _ D.d.f.ps[2 * i + r]!
  rw [getElem!_pos D.d.f.ps _ hk]
  exact h

theorem ctx_sch_files (hG : D.RowG cls acc i) : (D.ctx (2 * i + r)).sch = (D.files cls i).sch := by
  simp only [DecAt.ctx, repCtx, rep_div_two i r, ← hG.sts] <;> rfl

/-- **Row L's `params`**: the proof's PCS parameters are the pinned ones of table `i`'s scheme and schedule. -/
theorem rowL_params (hG : D.RowG cls acc i) (hk : 2 * i + r < D.d.f.ps.size) :
    (((D.files cls i).proof r).params != pcsParams (D.files cls i).st.merkle (D.files cls i).sch) = false := by
  have hst : (D.ctx (2 * i + r)).st = (D.files cls i).st := D.ctx_st cls acc i r hG
  rw [(D.pinned cls i r hk).params, hst, D.ctx_sch_files cls acc i r hG]
  exact FlockVBridge.Family.pcsParams_bne_self _

/-- **Row L's `initialCap`**: the proof's Ligerito initial cap is its cap. -/
theorem rowL_initialCap (hk : 2 * i + r < D.d.f.ps.size) :
    (((D.files cls i).proof r).ligerito.initialCap == ((D.files cls i).proof r).cap) = true := by
  rw [(D.pinned cls i r hk).initialCap]
  exact beq_self_eq_true _

/-- **Row L's `fc`**: at the verifier rows, the proof's `finalC` is V\*'s `v_C` at `w`, from row C's `c` (`hc`), where
the stream's round-2 coin is the live one the reader takes `z` from (`hz`). -/
theorem rowL_fc (hG : D.RowG cls acc i) (hk : 2 * i + r < D.d.f.ps.size)
    (hc : ((D.files cls i).proof r).zerocheck.round1c =
      (decoded (D.sh cls i) (D.rep cls acc i r).w (D.rep cls acc i r).comb (D.rep cls acc i r).cons).c)
    (hne : NoEmpty (D.rep cls acc i r).rounds)
    (hz : (((D.files cls i).stream r).rounds[2]!).coins.getD 0 0 = (D.ctx (2 * i + r)).z) :
    ∀ v info kappa,
      verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
        .ok (v, info, kappa) →
      ((D.files cls i).proof r).zerocheck.finalC =
        info.zc.vc.eval (valOf (D.sh cls i) (D.rep cls acc i r).w (D.rep cls acc i r).comb (D.rep cls acc i r).cons) := by
  intro v info kappa hrows
  have hm : 14 ≤ (D.sh cls i).m := by
    have := FlockVBridge.Family.le_of_fast100 hG.fast
    show 14 ≤ (D.files cls i).st.m
    omega
  have hzz := Flock.RecAlgebra.zc_z (R := ((D.files cls i).stream r).rounds) hrows hm hne
  rw [Flock.RecAlgebra.vc_eval hrows, ← hc, (D.pinned cls i r hk).finalC, hzz, hz]

theorem getElem!_map_getD (l : List (List ByteArray)) (j : ℕ) :
    (l.map (·.foldl (· ++ ·) ByteArray.empty))[j]! = (l.getD j []).foldl (· ++ ·) ByteArray.empty := by
  rw [List.getElem!_eq_getElem?_getD, List.getElem?_map, List.getD_eq_getElem?_getD]
  cases l[j]? <;> rfl

/-- **Row L's `cap`**: the proof's cap, joined, is the session's `root_B` of table `i`, where the session's `root_B`
is the link's (`hroot`). -/
theorem rowL_cap (hk : 2 * i + r < D.d.f.ps.size) (hroot : D.sess.rootB.toList = D.d.link.1) :
    (((D.files cls i).proof r).cap.foldl (· ++ ·) ByteArray.empty == D.sess.rootB[i]!) = true := by
  obtain ⟨r₀, rss, h₀, -, -, hfit⟩ := decR_fits D.dec
  obtain ⟨T, hT⟩ := hfit (2 * i + r) hk
  obtain ⟨r₀', h₀', hlink, -⟩ := decR_rootB D.dec
  rw [h₀] at h₀'
  cases Option.some.inj h₀'
  have hcap := hT.cap
  have e : ((D.files cls i).proof r).cap.foldl (· ++ ·) ByteArray.empty = D.sess.rootB[i]! := by
    show D.d.f.ps[2 * i + r]!.cap.foldl (· ++ ·) ByteArray.empty = _
    rw [getElem!_pos D.d.f.ps _ hk, ← Array.foldl_toList, hcap, ← Array.getElem!_toList, hroot, hlink,
      getElem!_map_getD]
    simp only [repCtx, rep_div_two i r]
  rw [e]
  exact beq_self_eq_true _

/-- **Row L** at rep `r` of table `i`: from row G's setup, row C's `c`, the rounds' coins being non-empty, the
session's `root_B` being the link's (`hroot`) and the stream's round-2 coin being the live one (`hz`). -/
theorem rowL_of (hG : D.RowG cls acc i) (hC : D.RowC cls acc i r) (hk : 2 * i + r < D.d.f.ps.size)
    (hne : NoEmpty (D.rep cls acc i r).rounds) (hroot : D.sess.rootB.toList = D.d.link.1)
    (hz : (((D.files cls i).stream r).rounds[2]!).coins.getD 0 0 = (D.ctx (2 * i + r)).z) :
    D.RowL cls acc i r where
  params := D.rowL_params cls acc i r hG hk
  initialCap := D.rowL_initialCap cls i r hk
  fc := D.rowL_fc cls acc i r hG hk hC.c hne hz
  cap := D.rowL_cap cls i r hk hroot

/-- **The session's `root_B` is the link's** where the `after` check passes at the decision: custody compares
`liveOfJ`'s `root_B` at the record value with the link's, and the decoder reads `liveOfJ`'s. -/
theorem rootB_link
    (ha : FlockDeploy.VerdictR.At FlockDeploy.sha512 .after D.d.s x D.d.link D.d.c₁ D.d.mids D.d.f) :
    D.sess.rootB.toList = D.d.link.1 := by
  obtain ⟨-, lv, hlv, hcu⟩ := Claims.Bridge.Custody.checksAtJ_ok (ha D.spec D.sts D.setup)
  have hd := D.decode
  rw [Flock.Record.decodeJWith_eq_decodeJWithK] at hd
  rw [(FlockDeploy.KS.decodeJWithK_live id _ hd hlv).1]
  exact (Claims.Bridge.Custody.custody_ok hcu).1

/-- **Row L where the rows are asked**: at a dense inner statement's table, from the decision's points, the proof's
index and shape, the `after` check and the stream's round-2 coin being the live one (`hz`). -/
theorem rowL_at {xo : Claims.Msg} {sto : Claims.Recursion.Stmt} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : x = xIOf xo) (hi : i < D.sts.size)
    (hti : i < D.spec.tables.size) (hpts : D.sess.points = pointsOf D.spec D.d.c₁) (hk : 2 * i + r < D.d.f.ps.size)
    (hsh : ProofShaped (shapeAt (tabsOf D.sts) (pointsOf D.spec D.d.c₁) (2 * i + r)) (D.d.f.ps[2 * i + r]'hk))
    (ha : FlockDeploy.VerdictR.At FlockDeploy.sha512 .after D.d.s x D.d.link D.d.c₁ D.d.mids D.d.f)
    (hz : (((D.files cls i).stream r).rounds[2]!).coins.getD 0 0 = (D.ctx (2 * i + r)).z) :
    D.RowL cls acc i r :=
  have hG := D.rowG cls acc hst hf hx hi
  D.rowL_of cls acc i r hG (D.rowC_of cls acc i r hG hpts hk hsh) hk (D.noEmpty cls acc hti r) (D.rootB_link ha) hz

/-- **Table `i`'s reps run at the decision**, as `runs_assembled`, with row L from `rowL_at`: in place of `hL`, the
`after` check passes (`ha`) and each rep stream's round-2 coin is the live one the reader takes `z` from (`hz`). -/
theorem runs_assembled_L {xo : Claims.Msg} {sto : Claims.Recursion.Stmt} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : x = xIOf xo) {i : ℕ} (hi : i < D.sts.size)
    (hti : i < D.spec.tables.size)
    (hframed : ∀ r : Fin 2, FramedAt ((D.files cls i).stream r).rounds 0 (repFrames (D.files cls i).st
      (D.files cls i).sch ((D.files cls i).spec.domain (D.files cls i).spec.tables[(D.files cls i).i]! r).toUTF8
      (2 + (D.files cls i).extras.size) ((D.files cls i).proof r)
      (valOf (D.sh cls i) (D.rep cls acc i r).w (D.rep cls acc i r).comb (D.rep cls acc i r).cons .T)))
    (hpts : D.sess.points = pointsOf D.spec D.d.c₁)
    (hk : ∀ r : Fin 2, 2 * i + r < D.d.f.ps.size)
    (hsh : ∀ r : Fin 2,
      ProofShaped (shapeAt (tabsOf D.sts) (pointsOf D.spec D.d.c₁) (2 * i + r)) (D.d.f.ps[2 * i + r]'(hk r)))
    (ha : FlockDeploy.VerdictR.At FlockDeploy.sha512 .after D.d.s x D.d.link D.d.c₁ D.d.mids D.d.f)
    (hz : ∀ r : Fin 2, (((D.files cls i).stream r).rounds[2]!).coins.getD 0 0 = (D.ctx (2 * i + r)).z)
    (pieces : List OpenPiece) (sideAt : Claims.Msg → List Claims.Msg → OpenPiece → OpenSide)
    (posAt : Claims.Msg → List Coin → OpenPiece → ℕ)
    (hholds : ∀ q ∈ pieces, openHolds sideAt posAt q x ms cs)
    (hside : ∀ r : Fin 2, ∀ q ∈ pieces, q.table = i → q.rep = r.val → sideAt x ms q = D.side cls i r)
    (hpos : ∀ r : Fin 2, ∀ v info kappa,
      verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
        .ok (v, info, kappa) →
      ∀ q ∈ pieces, q.table = i → q.rep = r.val → posAt x cs q = (info.lg.conss[q.level]!).pos[q.query]!)
    (hcover : ∀ r : Fin 2, ∀ ℓ < (D.files cls i).sch.levels.size, ∀ j < ((D.files cls i).sch.levels[ℓ]!).q,
      ∃ q ∈ pieces, q.table = i ∧ q.rep = r.val ∧ q.level = ℓ ∧ q.query = j)
    {S : RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hS : Flock.RecAlgebra.structureOf (D.sh cls i) = .ok S) (hg : groups S (portSizes (D.sh cls i)) = .ok gs)
    (hsz : ∀ r : Fin 2, (D.rep cls acc i r).rounds.toList.map Array.size = roundSizes (D.sh cls i))
    (hw : ∀ r : Fin 2, (D.rep cls acc i r).w.size = S.nw) (hparts : ∀ r : Fin 2, D.PartsZero cls acc i r gs)
    (accC : Fin 2 → ℕ → Array F128) (p0 : Fin 2 → ℕ)
    (hu : ∀ r : Fin 2, UnitSums (D.sh cls i) (accC r) (D.opens cls acc i r (p0 r)))
    (hrd : ∀ r : Fin 2, ReadsEnds (D.sh cls i) (D.rep cls acc i r) (accC r) (D.opens cls acc i r (p0 r))) :
    (D.files cls i).Runs := by
  have hL : ∀ r, D.RowL cls acc i r := fun r =>
    D.rowL_at cls acc i r hst hf hx hi hti hpts (hk r) (hsh r) ha (hz r)
  exact D.runs_assembled cls acc hst hf hx hi hti hframed hpts hk hsh hL pieces sideAt posAt hholds hside hpos hcover
    hS hg hsz hw hparts accC p0 hu hrd

end RecDeploy.Rows.DecAt
