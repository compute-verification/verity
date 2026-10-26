import Proofs.Flock.Claims.Recursion.VerdictPieces
import Proofs.Flock.Claims.Recursion.NextShaDec
import Proofs.Flock.Claims.KS.PlainDrawRCols

/-!
# `hrest`'s base checks at `nextR`

`decompB_lastSplitP_shares` asks of every check no part owns that it holds off any witnesses (`hrest`). Among them are
`lastSplitP`'s base checks (`VerdictP.bases`: `.setup`, `.shaped`, `.names`, `.after`), each `holdsR` at the opened
messages and the coins. This file says which of them the inner statement fixes and proves those.

* **`.setup` and `.names` are the statement's** (`holdsR_setup`, `holdsR_names`): at every message and coin, the setup
  at SHA-512 returns the statement's session (`setupWith_sha512_of`), and at an inner statement the recursive reader
  parses (`parse_innerOk`: the scope check passes) the rep streams' names are distinct (`namesDistinct_of_setup`).
* **`.after` is the statement's where `decR` fixes its statement and draw** (`holdsR_after`): `checksAtJ` at `jAt` is
  the statement check, the draw check, custody, the scope check and the draw law. The scope check and the draw law
  are the statement's (`parse_innerOk`); custody is read off the reader's own record (`recJ`). The statement check
  reads `f.stmt` and the draw check `recJ`'s `unit_draw` (the setup's `spec.unitDraw`): taken as `hstmt` and `hdraw`,
  facts about `decR` (see the note `proofs/…-report-restverdict`).
* **`.shaped` is not the statement's** (`holdsR_shaped_of`): `ProofShaped` reads the Ligerito openings, which
  `absorbedFrom` takes from the last message's proof. Every length a rep's frames read is `FitsRep`'s (`decR_fits`),
  and the initial cap is the cap (`decR_initialCap`); the openings' levels and the file's size are `hopen`, which the
  part that reads the last message gives.
* **At `hrest`'s form** (`baseElem_holds`, `restBases_fixed`): a base check that holds at every message list as long
  as the coins is `IVStar`'s `elem` check at every message and coin.
-/

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP FinalJ Ins DecR VerdictR holdsR decR actR nextR nextR_coin checksAtJ drawOfJ liveOfJ
  custody jAt drawEq shapeAt tabsOf pointsOf Live)
open Claims.Recursion.VStarB

/-! ## The checks after the verdict -/

/-- **The checks after the verdict pass** when each does: the statement, the draw, custody at what `liveOfJ` reads,
the scope, and a drawn statement. -/
theorem checksAtJ_ok {s : Ins} {spec : SessionSpec} {x : Msg} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Msg × Coin)} {f : FinalJ} {j' : Lean.Json} {lv : Live} (hS : (f.stmt == x) = true)
    (hD : drawEq (drawOfJ f.j) s.read.draw = true) (hL : liveOfJ spec j' = .ok lv)
    (hC : custody spec lv link c₁ mids = .ok ()) (hQ : s.read.check = .ok ()) (hN : s.read.draw.isNone = false) :
    checksAtJ s spec x link c₁ mids f j' = .ok () := by
  unfold FlockDeploy.checksAtJ
  simp only [hS, hD, FlockComplete.check_true, FlockComplete.ok_bind, hL, hC, hQ, hN, Bool.false_eq_true, ite_false]
  try rfl

/-! ## The initial cap of a decided proof -/

theorem readRep_initialCap {c : FlockDeploy.RepCtx} {rows : List (List ByteArray)} {q : Flock.Proof}
    {r : Flock.Proof × List Msg} (h : FlockDeploy.readRep c rows q = some r) :
    r.1.ligerito.initialCap = r.1.cap := by
  unfold FlockDeploy.readRep at h
  split at h
  · rw [← Option.some.inj h]
    rfl
  · cases h

theorem readPs_initialCap {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin} {r₀ : List ByteArray}
    {rss : List (List ByteArray)} {cs : List Coin} {fl : FinalP} {pf : Array Flock.Proof × List Msg}
    (h : FlockDeploy.readPs spec sts c₁ r₀ rss cs fl = some pf) (k : ℕ) (hk : k < pf.1.size) :
    pf.1[k].ligerito.initialCap = pf.1[k].cap := by
  unfold FlockDeploy.readPs at h
  by_cases hc : fl.ps.size = 2 * sts.size ∧ (FlockDeploy.repLens spec sts).sum = rss.length
  · rw [if_pos hc] at h
    obtain ⟨rs, hrs, rfl⟩ := Option.map_eq_some_iff.1 h
    obtain ⟨hlen, hget⟩ := FlockDeploy.mapM_some_get hrs
    rw [List.length_range] at hlen
    have hk' : k < rs.length := by simpa using hk
    have hg := hget k (by rw [List.length_range]; omega) hk'
    rw [List.getElem_range] at hg
    simpa using readRep_initialCap hg
  · rw [if_neg hc] at h
    cases h

/-- **A decided proof's initial cap is its cap** (`absorbedFrom`'s). -/
theorem decR_initialCap {rd : Msg → Option FinalP} {x : Msg} {as : List Msg} {cs : List Coin} {d : DecR}
    (h : decR rd x as cs = some d) (k : ℕ) (hk : k < d.f.ps.size) :
    d.f.ps[k].ligerito.initialCap = d.f.ps[k].cap := by
  obtain ⟨v, c₁, r₀, rss, fl, link, pf, -, -, -, -, -, h9, rfl⟩ := FlockDeploy.decR_some h
  exact readPs_initialCap h9 k hk

section Bases

variable (rd : Msg → Option FinalP)

/-! ## `.setup` and `.names`: the statement's -/

/-- **The setup at SHA-512 returns** at every decision of `nextR`, at every statement, message and coin. -/
theorem holdsR_setup (x : Msg) (ms : List Msg) (cs : List Coin) : holdsR FlockDeploy.sha512 rd .setup x ms cs := by
  unfold FlockDeploy.holdsR
  cases h : actR rd x ms cs with
  | rej => trivial
  | coin c => trivial
  | decide s spec sts link c₁ mids f => exact ⟨(spec, sts), setupWith_sha512_of (FlockDeploy.actR_setupOf h)⟩

/-- **The rep streams' names are distinct** at every decision of `nextR` at an inner statement the recursive reader
parses, at every message and coin: its scope check passes (`parse_innerOk`), so it carries no public inputs. -/
theorem holdsR_names {x : Msg} {st : Claims.Recursion.Stmt} (hx : parse x = some st) (ms : List Msg)
    (cs : List Coin) : holdsR FlockDeploy.sha512 rd .names (xIOf x) ms cs := by
  obtain ⟨s₀, spec₀, sts₀, h₀, hck, -⟩ := parse_innerOk hx
  unfold FlockDeploy.holdsR
  cases h : actR rd (xIOf x) ms cs with
  | rej => trivial
  | coin c => trivial
  | decide s spec sts link c₁ mids f =>
    intro spec' sts' hs
    have hv := FlockDeploy.actR_setupOf h
    rw [h₀] at hv
    simp only [Option.some.injEq, Prod.mk.injEq] at hv
    obtain ⟨rfl, rfl, rfl⟩ := hv
    rw [setupWith_sha512_of h₀] at hs
    simp only [Except.ok.injEq, Prod.mk.injEq] at hs
    obtain ⟨rfl, rfl⟩ := hs
    exact FlockDeploy.namesDistinct_of_setup (FlockDeploy.setupOf_some h₀).2
      (Claims.Bridge.Custody.publicInputs_none_of_check hck)

/-! ## `.after`: the statement's where `decR` fixes its statement and draw -/

/-- **The checks after the verdict hold** at every decision of `nextR` at an inner statement the recursive reader
parses, at message and coin lists of one length, where every decision's statement is the statement (`hstmt`), its
record's draw the statement's (`hdraw`), and custody passes at the reader's record (`hcust`). -/
theorem holdsR_after
    (hstmt : ∀ x as cs (d : DecR), decR rd x as cs = some d → d.f.stmt = x)
    (hdraw : ∀ x as cs (d : DecR), decR rd x as cs = some d → drawEq (drawOfJ d.f.j) d.s.read.draw = true)
    (hcust : ∀ x as cs (d : DecR), decR rd x as cs = some d → as.length = cs.length →
      ∃ lv, liveOfJ d.spec (jAt FlockDeploy.sha512 d.spec d.sts d.c₁ d.mids d.f) = .ok lv ∧
        custody d.spec lv d.link d.c₁ d.mids = .ok ())
    {x : Msg} {st : Claims.Recursion.Stmt} (hx : parse x = some st) (ms : List Msg) (cs : List Coin)
    (hl : ms.length = cs.length) : holdsR FlockDeploy.sha512 rd .after (xIOf x) ms cs := by
  obtain ⟨s₀, spec₀, sts₀, h₀, hck, hdr⟩ := parse_innerOk hx
  unfold FlockDeploy.holdsR
  cases h : actR rd (xIOf x) ms cs with
  | rej => trivial
  | coin c => trivial
  | decide s spec sts link c₁ mids f =>
    intro spec' sts' hs
    have hv := FlockDeploy.actR_setupOf h
    rw [h₀] at hv
    simp only [Option.some.injEq, Prod.mk.injEq] at hv
    obtain ⟨rfl, rfl, rfl⟩ := hv
    rw [setupWith_sha512_of h₀] at hs
    simp only [Except.ok.injEq, Prod.mk.injEq] at hs
    obtain ⟨rfl, rfl⟩ := hs
    have hd := FlockDeploy.actR_decide h
    have e : f.stmt = xIOf x := hstmt _ _ _ _ hd
    have hD : drawEq (drawOfJ f.j) s₀.read.draw = true := hdraw _ _ _ _ hd
    obtain ⟨lv, hlv, hc⟩ := hcust _ _ _ _ hd hl
    refine checksAtJ_ok (lv := lv) ?_ hD hlv hc hck ?_
    · show (f.stmt == xIOf x) = true
      rw [e]
      exact beq_self_eq_true _
    · revert hdr
      cases s₀.read.draw <;> simp

/-! ## `.shaped`: the part's that reads the last message -/

/-- **Each decided proof is shaped for its file** where its openings fit their levels and its file a `u64` (`hopen`,
what the last message's reader gives) and its digests have positive width (`hhash`): every length a rep's frames read
is `FitsRep`'s, the initial cap the cap, every nonce `0`. -/
theorem holdsR_shaped_of {x : Msg} {ms : List Msg} {cs : List Coin}
    (hhash : ∀ d : DecR, decR rd x ms cs = some d → ∀ k < d.f.ps.size,
      0 < (shapeAt (tabsOf d.sts) (pointsOf d.spec d.c₁) k).hashLen)
    (hopen : ∀ d : DecR, decR rd x ms cs = some d → ∀ (k : ℕ) (hk : k < d.f.ps.size),
      let sh := shapeAt (tabsOf d.sts) (pointsOf d.spec d.c₁) k
      let lvl := fun (i : ℕ) => sh.sched.levels.getD i default
      let lg := d.f.ps[k].ligerito
      (FlockVBridge.Family.encodeProof sh d.f.ps[k]).size < 2 ^ 64 ∧
        FlockSoundness.Discharge.ZkView.LevelOk sh (lvl 0) (2 ^ (lvl 0).lanes) lg.initialProof ∧
        lg.recursiveProofs.size = sh.sched.levels.size - 1 - 1 ∧
        (∀ (i : ℕ) (h : i < lg.recursiveProofs.size),
          FlockSoundness.Discharge.ZkView.LevelOk sh (lvl (i + 1)) 16 lg.recursiveProofs[i]) ∧
        FlockSoundness.Discharge.ZkView.LevelOk sh (lvl (sh.sched.levels.size - 1)) 16
          ⟨lg.finalProof.rows, lg.finalProof.siblings, lg.finalProof.salts⟩) :
    holdsR FlockDeploy.sha512 rd .shaped x ms cs := by
  unfold FlockDeploy.holdsR
  cases h : actR rd x ms cs with
  | rej => trivial
  | coin c => trivial
  | decide s spec sts link c₁ mids f =>
    intro spec' sts' hs k hk
    rw [setupWith_sha512_of (FlockDeploy.actR_setupOf h)] at hs
    simp only [Except.ok.injEq, Prod.mk.injEq] at hs
    obtain ⟨rfl, rfl⟩ := hs
    have hd := FlockDeploy.actR_decide h
    obtain ⟨r₀, rss, -, -, -, hfit⟩ := FlockDeploy.decR_fits hd
    obtain ⟨T, hT⟩ := hfit k hk
    obtain ⟨e, hn, hsz⟩ := hT.sizes
    have hinit : f.ps[k].ligerito.initialCap = f.ps[k].cap := decR_initialCap hd k hk
    have hpos := hhash _ hd k hk
    obtain ⟨hfile, hl0, hrps, hrp, hfin⟩ := hopen _ hd k hk
    have hlev := hT.levels
    have he : (match (tabsOf sts)[k / 2]!.extra (pointsOf spec c₁) with
        | .ok e => e.size
        | .error _ => 0) = e := Nat.add_left_cancel hn
    subst he
    unfold FlockVBridge.Family.ProofShaped FlockVBridge.Family.Lengths
    refine ⟨hpos, Nat.le_sub_of_add_le hlev, hfile, ⟨hsz.cap, hsz.capEach⟩, hsz.r1ab, hsz.r1c, hsz.zcRounds,
      hsz.lcRounds, hsz.zPartial, hsz.rs, hsz.rsEach, ?_, hl0, hsz.rcaps, fun i hi => hsz.rcapsEach i hi, hrps, hrp,
      hsz.yr, hfin, hsz.messages, hsz.queryNonces, hsz.oodValues, hsz.claimNonces, hsz.consistencyNonces, hT.nonces⟩
    rw [hinit]
    exact ⟨hsz.cap, hsz.capEach⟩

end Bases

/-! ## At `hrest`'s form -/

section Rest

variable (rd : Msg → Option FinalP) {Q : Type} (pieces : Msg → List Q)
  (pholds : Q → Msg → List Msg → List Coin → Prop)
  (hpieces : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), FlockDeploy.ReadOkR rd x ms cs →
    (∀ b ∈ VerdictP.bases, holdsR FlockDeploy.sha512 rd b x ms cs) → (∀ q ∈ pieces x, pholds q x ms cs) →
      holdsR FlockDeploy.sha512 rd .verdict x ms cs)
  (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
  (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ)
  (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)

/-- **A base check that holds at every message list as long as the coins is `IVStar`'s `elem` check**, at every
message and coin: the opened messages are one per round, as the coins are. -/
theorem baseElem_holds {q : VerdictR} {x : Msg}
    (hq : ∀ (ms : List Msg) (cs : List Coin), ms.length = cs.length → holdsR FlockDeploy.sha512 rd q x ms cs)
    (c : ℕ → Coin) (ms : List Msg) :
    (IVCheck.elem (DeployCheck.last (ReadCheck.verdict (VerdictP.base q))) :
        IVCheck (DeployCheck (ReadCheck (VerdictP Q)))).Holds FlockDeploy.sha512
      (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w)
      (deploySplit FlockDeploy.sha512 rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w
        (nextR_coin rd FlockDeploy.sha512) (lastSplitP FlockDeploy.sha512 rd pieces pholds hpieces).toLast) x c ms := by
  intro os last _ hos
  refine hq _ _ ?_
  simp only [Claims.Recursion.Bound.opened, List.length_append, List.length_map, List.length_singleton,
    List.length_range]
  exact hos

/-- **`.setup`, `.names` and `.after` hold off any witnesses** at every compiled statement the recursive reader
parses (`hrest`'s case of these three), where `decR` fixes its statement and draw and custody passes at the reader's
record (`holdsR_after`'s hypotheses). -/
theorem restBases_fixed
    (hstmt : ∀ x as cs (d : DecR), decR rd x as cs = some d → d.f.stmt = x)
    (hdraw : ∀ x as cs (d : DecR), decR rd x as cs = some d → drawEq (drawOfJ d.f.j) d.s.read.draw = true)
    (hcust : ∀ x as cs (d : DecR), decR rd x as cs = some d → as.length = cs.length →
      ∃ lv, liveOfJ d.spec (jAt FlockDeploy.sha512 d.spec d.sts d.c₁ d.mids d.f) = .ok lv ∧
        custody d.spec lv d.link d.c₁ d.mids = .ok ())
    (rel' : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings) (L : LayoutB)
    (IV : Claims.Verifier Coin) {q : VerdictR} (hq : q = .setup ∨ q = .names ∨ q = .after) {x : Msg}
    {st : Claims.Recursion.Stmt} (hx : (recBound rel' b L IV).parse x = some st) (c : ℕ → Coin) (ms : List Msg) :
    (IVCheck.elem (DeployCheck.last (ReadCheck.verdict (VerdictP.base q))) :
        IVCheck (DeployCheck (ReadCheck (VerdictP Q)))).Holds FlockDeploy.sha512
      (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w)
      (deploySplit FlockDeploy.sha512 rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w
        (nextR_coin rd FlockDeploy.sha512) (lastSplitP FlockDeploy.sha512 rd pieces pholds hpieces).toLast)
      (Claims.Recursion.xI (recBound rel' b L IV) x) c ms := by
  have hx' : parse x = some st := hx
  refine baseElem_holds rd pieces pholds hpieces rel colsE wE rowIn final w (x := xIOf x) (fun ms' cs' hl => ?_) c ms
  rcases hq with rfl | rfl | rfl
  · exact holdsR_setup rd _ ms' cs'
  · exact holdsR_names rd hx' ms' cs'
  · exact holdsR_after rd hstmt hdraw hcust hx' ms' cs' hl

end Rest

end RecDeploy
