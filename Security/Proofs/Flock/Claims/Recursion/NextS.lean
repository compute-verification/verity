import Proofs.Flock.Claims.Recursion.NextShaDec

/-!
# `IVStar`'s last step with the proofs decoded (`nextS`)

`nextJ`'s last message carries the proof files, which its decision decodes inside `verifyRepWith`, a check no V\* part
computes (`NextJ`'s open gap). `nextS` is the element-level step whose last message carries each rep's decoded proof
(`FinalP`, read by the layout's reader `rd`), and whose decision is `decisionSF`: the setup over `OC`, the files'
digests through the oracle, then the verdict at the record's value with its proof digests the files' (`decisionAtS`).
Run against `H`, that decision is the verdict on the decoded proofs (`verifyJPWith`) then the checks after it
(`decisionAtS_eval`), so no proof file is decoded and S17 is no check of `IVStar`.

* **The step** (`actS`, `nextS`): a coin is answered as the deployed plain verifier answers it (`nextS_coin`, so
  `deploySplit` applies); at a decision it is `decisionSF` (`nextS_decide`).
* **When it accepts** (`nextS_acc_iff`, `nextS_acc_of`): run against `H`, a decision accepts exactly where the setup at
  `hOf H` returns a session at which `decisionAtS` returns; there, where each proof is shaped for its file and the rep
  streams' names are distinct, `verifyJPWith` and `checksAtJ` passing make it accept.
* **The split and the composition** (`LastSplitS`, `LastSplitS.toLast`, `ownS_lt`, `RecDeploy.decompB_nextS`): as at
  `nextJ`, the reading (`ReadOkS`) is one check given to the first part holding the last round whole (`ownJ`), and the
  verdict's checks keep their owners. This replaces `decompB_nextJ` as the route: its verdict's checks are
  `verifyJPWith`'s and `checksAtJ`'s, none of them a proof file's decode.
-/

namespace FlockDeploy

open Flock
open Lean (Json)

/-- What `nextS` does at the prover's messages. -/
inductive ActS where
  | rej
  | coin (c : Coin)
  | decide (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (link : List ByteArray × List ByteArray) (c₁ : Coin)
      (mids : List (Claims.Msg × Coin)) (f : FinalP)

/-- **`nextS`'s step**: a coin where the deployed plain verifier answers one; else, at a statement that sets up, decide
where `rd` reads the last message and the link message reads; anything else, refuse. -/
def actS (rd : Claims.Msg → Option FinalP) (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin) : ActS :=
  match act false x as cs with
  | .coin c => .coin c
  | _ =>
    match setupOf false x with
    | none => .rej
    | some (s, spec, sts) =>
      match as.getLast?, as, cs with
      | some a, a₁ :: rest, c₁ :: crest =>
        match rd a, linkOf a₁ with
        | some f, some link => .decide s spec sts link c₁ ((rest.dropLast.map List.tail).zip crest.dropLast) f
        | _, _ => .rej
      | _, _, _ => .rej

/-- **The element-level step with the proofs decoded.** -/
def nextS (rd : Claims.Msg → Option FinalP) (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin) :
    Claims.OC Claims.Out :=
  match actS rd x as cs with
  | .rej => .ret .rej
  | .coin c => .ret (.msg c.bytes)
  | .decide s _ _ link c₁ mids f => outOf (decisionSF s x link c₁ mids f)

/-- **A coin is answered as the plain verifier answers it** (`deploySplit`'s `hcoin`). -/
theorem nextS_coin (rd : Claims.Msg → Option FinalP) (H : Claims.Msg → Claims.Msg) : ∀ (x : Claims.Msg)
    (as : List Claims.Msg) (cs : List Coin) (c : Coin), act false x as cs = .coin c →
    ((nextS rd x as cs).eval H).1 = .msg c.bytes := by
  intro x as cs c h
  simp only [nextS, actS, h, Claims.OC.eval]

theorem nextS_decide {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FinalP} (h : actS rd x as cs = .decide s spec sts link c₁ mids f) :
    nextS rd x as cs = outOf (decisionSF s x link c₁ mids f) := by
  simp only [nextS, h]

theorem outOf_acc_iff (H : Claims.Msg → Claims.Msg) (d : Claims.OCV Unit) :
    ((outOf d).eval H).1 = .acc ↔ (d.run.eval H).1 = .ok () := by
  simp only [outOf, Claims.OC.eval_bind, Claims.OC.eval]
  cases hr : (d.run.eval H).1 with
  | ok u =>
    cases u
    exact ⟨fun _ => rfl, fun _ => rfl⟩
  | error e => exact ⟨fun h => by first | cases h | simp at h, fun h => by cases h⟩

/-- `decisionSF` run against `H`: the setup at `hOf H`, then `decisionAtS` at its session. -/
theorem decisionSF_eval (H : Claims.Msg → Claims.Msg) (s : Ins) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (f : FinalP) :
    ((decisionSF s x link c₁ mids f).run.eval H).1 =
      (setupWith (Claims.hOf H) s.read >>= fun v => ((decisionAtS s v.1 v.2 x link c₁ mids f).run.eval H).1) := by
  unfold decisionSF
  rw [evRun_bind, (setupM_eval H s.read).1]

/-- **A decision of `nextS` accepts exactly where the setup at `hOf H` returns a session at which `decisionAtS`
returns.** -/
theorem nextS_acc_iff (H : Claims.Msg → Claims.Msg) {rd : Claims.Msg → Option FinalP} {x : Claims.Msg}
    {as : List Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : FinalP}
    (h : actS rd x as cs = .decide s spec sts link c₁ mids f) :
    ((nextS rd x as cs).eval H).1 = .acc ↔ ∃ spec' sts', setupWith (Claims.hOf H) s.read = .ok (spec', sts') ∧
      ((decisionAtS s spec' sts' x link c₁ mids f).run.eval H).1 = .ok () := by
  rw [nextS_decide h, outOf_acc_iff, decisionSF_eval]
  constructor
  · intro hd
    obtain ⟨⟨spec', sts'⟩, hs, hd⟩ := FlockSoundness.Refine.ebind_ok hd
    exact ⟨spec', sts', hs, hd⟩
  · rintro ⟨spec', sts', hs, hd⟩
    rw [hs]
    exact hd

/-- **The verdict on the decoded proofs and the checks after it make a decision of `nextS` accept**, at the session the
setup at `hOf H` returns, where each proof is shaped for its file and the rep streams' names are distinct. -/
theorem nextS_acc_of (H : Claims.Msg → Claims.Msg) {rd : Claims.Msg → Option FinalP} {x : Claims.Msg}
    {as : List Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : FinalP}
    (h : actS rd x as cs = .decide s spec sts link c₁ mids f) {spec' : SessionSpec} {sts' : Array Stmt}
    (hs : setupWith (Claims.hOf H) s.read = .ok (spec', sts'))
    (hsh : ∀ (k : ℕ) (hk : k < f.ps.size),
      FlockVBridge.Family.ProofShaped (shapeAt (tabsOf sts') (pointsOf spec' c₁) k) f.ps[k])
    (hnd : NamesDistinct spec')
    (hv : verifyJPWith (Flock.Hashes.ofSha512 (Claims.hOf H) : Flock.Hashes Flock.V) spec' (tabsOf sts')
      (coinSubJ spec' c₁ mids (shaSubJ spec' ((filesAt (tabsOf sts') (pointsOf spec' c₁) f.ps).map
        (FlockComplete.dgOf spec' (Claims.hOf H))) f.j)) f.ps = .ok ())
    (hc : checksAtJ s spec' x link c₁ mids ⟨f.stmt, f.j, #[]⟩
      (coinSubJ spec' c₁ mids (shaSubJ spec' ((filesAt (tabsOf sts') (pointsOf spec' c₁) f.ps).map
        (FlockComplete.dgOf spec' (Claims.hOf H))) f.j)) = .ok ()) :
    ((nextS rd x as cs).eval H).1 = .acc := by
  refine (nextS_acc_iff H h).2 ⟨spec', sts', hs,
    (decisionAtS_eval H s spec' sts' x link c₁ mids (stmt := f.stmt) (j := f.j) (ps := f.ps) hsh hnd).trans ?_⟩
  rw [hv]
  exact hc

/-- **The reading at the last step**: `rd` reads the last message and the link message reads (`actS` decides), and the
record value's fields custody reads are there with their shapes (`liveOfJ` at the coin-substituted value is `.ok`). -/
def ReadOkS (rd : Claims.Msg → Option FinalP) (x : Claims.Msg) (ms : List Claims.Msg) (cs : List Coin) : Prop :=
  match actS rd x ms cs with
  | .decide _ spec _ _ c₁ mids f => ∃ lv, liveOfJ spec (coinSubJ spec c₁ mids f.j) = .ok lv
  | _ => False

end FlockDeploy

namespace RecDeploy

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin FinalP nextS nextS_coin ReadOkS)
open Claims.Recursion.VStarB

/-- **A split of `nextS`'s last step that may assume the reading**: checks the inner statement fixes, and the proof that
where the reading holds (`ReadOkS`) and every check does, the last step accepts. -/
structure LastSplitS (H : Msg → Msg) (rd : Msg → Option FinalP) (R : Type) where
  checks : Msg → List R
  holds : R → Msg → List Msg → List Coin → Prop
  sound : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), ms.length = (FlockDeploy.V false).k x →
    cs.length = ms.length → ReadOkS rd x ms cs → (∀ r ∈ checks x, holds r x ms cs) → ((nextS rd x ms cs).eval H).1 = .acc

/-- **The split with the reading as its own check.** -/
def LastSplitS.toLast {H : Msg → Msg} {rd : Msg → Option FinalP} {R : Type} (LS : LastSplitS H rd R) :
    LastSplit H (nextS rd) (ReadCheck R) where
  checks x := .read :: (LS.checks x).map .verdict
  holds q x ms cs := match q with
    | .read => ReadOkS rd x ms cs
    | .verdict r => LS.holds r x ms cs
  sound x ms cs hk hc h := LS.sound x ms cs hk hc (h .read (List.mem_cons_self ..)) fun r hr =>
    h (.verdict r) (List.mem_cons_of_mem _ (List.mem_map.2 ⟨r, hr, rfl⟩))

/-- `ownJ`'s owners are among the statement's parts wherever `ownR`'s are. -/
theorem ownS_lt {H : Msg → Msg} {rd : Msg → Option FinalP} {R : Type} (LS : LastSplitS H rd R)
    (ownR : Msg → R → Option ℕ) (D : Claims.Recursion.Deploy Coin) (L : LayoutB) (shape : Claims.Recursion.Form → Prop)
    (hR : ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), D.parse x = some st → shape st.form →
      ∀ r ∈ LS.checks (Claims.Recursion.xI D x), ∀ s, ownR x r = some s → s < D.nS x) :
    ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), D.parse x = some st → shape st.form →
      ∀ r ∈ LS.toLast.checks (Claims.Recursion.xI D x), ∀ s, ownJ D L ownR x r = some s → s < D.nS x := by
  intro x st hx hf r hr s ho
  cases r with
  | read => exact firstHolder_lt ho
  | verdict r =>
    refine hR x st hx hf r ?_ s ho
    rcases List.mem_cons.1 hr with h | h
    · cases h
    · obtain ⟨r', hr', h⟩ := List.mem_map.1 h
      cases h
      exact hr'

/-- **The composition at `nextS`**: `bdDeployed` with `nextS rd` as its steps, split by `deploySplit` at the reading's
split (`LastSplitS.toLast`), the reading given to the reader part of the last round (`ownJ`). Where the verdict's owners
are the statement's parts (`hR`), the checks no part holds hold off any witnesses (`hrest`), and each part's circuit
accepting implies its share (`hlow`), `DecompB` holds at what each part's circuit accepts. No proof file is decoded
among the verdict's checks (`FlockDeploy.nextS_acc_of`). -/
theorem decompB_nextS (H : Msg → Msg) (rd : Msg → Option FinalP)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) {join : List Msg → Msg}
    (hj : ∀ rs, ∃ a, join rs = 0 :: a)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    {R : Type} (LS : LastSplitS H rd R) (ownR : Msg → R → Option ℕ)
    (rel' : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings) (L : LayoutB)
    (IV : Claims.Verifier Coin) (shape : Claims.Recursion.Form → Prop) {Col' T' : Type}
    (DO : Claims.Flock.Deploy Coin Col' T') (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg)
    (hR : ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (recBound rel' b L IV).parse x = some st → shape st.form →
      ∀ r ∈ LS.checks (Claims.Recursion.xI (recBound rel' b L IV) x), ∀ s, ownR x r = some s →
        s < (recBound rel' b L IV).nS x)
    (hrest : ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (recBound rel' b L IV).parse x = some st → shape st.form →
      ∀ q ∈ ivChecks (bdDeployed rel (nextS rd) colsE wE join rowIn final w)
          (deploySplit H rel (nextS rd) colsE wE join rowIn final w (nextS_coin rd H) LS.toLast)
          (Claims.Recursion.xI (recBound rel' b L IV) x),
        ownHolders (recBound rel' b L IV) L (ownDeploy (ownJ (recBound rel' b L IV) L ownR)) x q = none →
        (∀ r, q = .elem r → ∃ r', r = .last r') →
        ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit), ws.length = (recBound rel' b L IV).nS x →
          (∀ e ∈ L.allCalls x ws, H e.1 = e.2) →
          q.Holds H (bdDeployed rel (nextS rd) colsE wE join rowIn final w)
            (deploySplit H rel (nextS rd) colsE wE join rowIn final w (nextS_coin rd H) LS.toLast)
            (Claims.Recursion.xI (recBound rel' b L IV) x) c (L.msgs x ws))
    (hlow : ∀ s x c ms, PartAccepts H (recBound rel' b L IV) DO L shape shows cmt s x c ms →
      PartComputes H (recBound rel' b L IV) (bdDeployed rel (nextS rd) colsE wE join rowIn final w)
        (deploySplit H rel (nextS rd) colsE wE join rowIn final w (nextS_coin rd H) LS.toLast)
        (ownHolders (recBound rel' b L IV) L (ownDeploy (ownJ (recBound rel' b L IV) L ownR))) s x c ms) :
    DecompB H (recBound rel' b L IV) (bdDeployed rel (nextS rd) colsE wE join rowIn final w).IVStar L shape
      (PartAccepts H (recBound rel' b L IV) DO L shape shows cmt) :=
  decompB_partAccepts_of_parts H (recBound rel' b L IV) DO _ L shape shows cmt
    (deploySplit H rel (nextS rd) colsE wE join rowIn final w (nextS_coin rd H) LS.toLast) _ _ hlow
    (ownedBy_deploy H rel (nextS rd) colsE wE join rowIn final w (nextS_coin rd H) LS.toLast
      (ownJ (recBound rel' b L IV) L ownR) _ L shape (ownS_lt LS ownR _ L shape hR))
    (stmtFixes_deploy H rel (nextS rd) colsE wE hj rowIn final w (nextS_coin rd H) LS.toLast
      (ownJ (recBound rel' b L IV) L ownR) rel' b L IV shape hrest)

end RecDeploy
