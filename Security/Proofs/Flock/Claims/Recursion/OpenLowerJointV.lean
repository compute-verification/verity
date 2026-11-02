import Proofs.Flock.Claims.Recursion.PartAcceptsSetup

/-!
# The composition at the parts' own messages, landing on `DecompBV` (`decompBV_lastSplitP_sharesJV`)

`decompB_lastSplitP_sharesJ` asks the pieces' share under `Joint`, whose no-collision conjunct searches the parts'
recorded calls only, and lands on `DecompB`. A share that ties two parts' commit strings through the compiled
verifier's own climbs (row E's chain slots, `Top.RowEJoint`, through `accCommit_eq`) needs the escape to search those
evaluations too. `decompBV_lastSplitP_sharesJV` asks `hlowP` under `JointV` at `PartAcceptsV` and lands on
`DecompBV` at `PartAcceptsV` (`decompBV_of_jointV`), the form `RecDeploy.ksLin2_deployedR_layRV` takes. `JointV` gives
`Joint` at `PartAccepts` (`joint_of_jointP`, `jointP_of_jointV`), so a share proved under `Joint` is one under `JointV`.
-/

namespace RecDeploy.OpenLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP nextR nextR_coin ReadOkR VerdictR holdsR)
open Claims.Recursion.VStarB

section Compose

variable (H : Msg → Msg) (rd : Msg → Option FinalP) {Q : Type} (pieces : Msg → List Q)
  (pholds : Q → Msg → List Msg → List Coin → Prop)
  (hpieces : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), ReadOkR rd x ms cs →
    (∀ b ∈ VerdictP.bases, holdsR H rd b x ms cs) → (∀ q ∈ pieces x, pholds q x ms cs) →
      holdsR H rd .verdict x ms cs)
  (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
  (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ)
  (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)

/-- **The composition with the pieces' share under `JointV`**: `decompB_lastSplitP_sharesJ` with `hlowP` asked only
where every part accepts at its V\* statement and witness and no recorded call collides with another or with the
compiled verifier's own evaluations there (`JointV`), landing on `DecompBV` at `PartAcceptsV`. -/
theorem decompBV_lastSplitP_sharesJV (ownB : Msg → VerdictR → Option ℕ) (ownQ : Msg → Q → Option ℕ)
    (rel' : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings) (L : LayoutB)
    (IV : Claims.Verifier Coin) (shape : Claims.Recursion.Form → Prop) {Col' T' : Type}
    (DO : Claims.Flock.Deploy Coin Col' T') (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg)
    (hR : ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (recBound rel' b L IV).parse x = some st → shape st.form →
      ∀ r ∈ (lastSplitP H rd pieces pholds hpieces).checks (Claims.Recursion.xI (recBound rel' b L IV) x), ∀ s,
        ownP ownB ownQ x r = some s → s < (recBound rel' b L IV).nS x)
    (hrest : ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (recBound rel' b L IV).parse x = some st → shape st.form →
      ∀ q ∈ ivChecks (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w)
          (deploySplit H rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w (nextR_coin rd H)
            (lastSplitP H rd pieces pholds hpieces).toLast)
          (Claims.Recursion.xI (recBound rel' b L IV) x),
        ownHolders (recBound rel' b L IV) L (ownDeploy (ownJ (recBound rel' b L IV) L (ownP ownB ownQ))) x q = none →
        (∀ r, q = .elem r → ∃ r', r = .last r') →
        ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit), ws.length = (recBound rel' b L IV).nS x →
          (∀ e ∈ L.allCalls x ws, H e.1 = e.2) →
          q.Holds H (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w)
            (deploySplit H rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w (nextR_coin rd H)
              (lastSplitP H rd pieces pholds hpieces).toLast)
            (Claims.Recursion.xI (recBound rel' b L IV) x) c (L.msgs x ws))
    (hlowR : ∀ s x c ms, PartAccepts H (recBound rel' b L IV) DO L shape shows cmt s x c ms →
      RestComputes H (recBound rel' b L IV) (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w)
        (deploySplit H rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w (nextR_coin rd H)
          (lastSplitP H rd pieces pholds hpieces).toLast)
        (ownHolders (recBound rel' b L IV) L (ownDeploy (ownJ (recBound rel' b L IV) L (ownP ownB ownQ)))) s x c ms)
    (hlowP : ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (recBound rel' b L IV).parse x = some st → shape st.form →
      ∀ (c : ℕ → Coin) (xs : List Msg) (ws : List Claims.Flock.Wit),
        JointV H (recBound rel' b L IV) L (PartAcceptsV H (recBound rel' b L IV) DO L shape shows cmt) x c xs ws →
        ∀ s < (recBound rel' b L IV).nS x,
          PiecesComputes (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w) (recBound rel' b L IV)
            pieces pholds ownQ s x c (L.msgs x ws)) :
    DecompBV H (recBound rel' b L IV) (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w).IVStar L
      shape (PartAcceptsV H (recBound rel' b L IV) DO L shape shows cmt) := by
  have hD := ivstar_of_parts H (recBound rel' b L IV) _ L shape
    (deploySplit H rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w (nextR_coin rd H)
      (lastSplitP H rd pieces pholds hpieces).toLast) _ _
    (ownedBy_deploy H rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w (nextR_coin rd H)
      (lastSplitP H rd pieces pholds hpieces).toLast (ownJ (recBound rel' b L IV) L (ownP ownB ownQ)) _ L shape
      (ownR_lt (lastSplitP H rd pieces pholds hpieces) (ownP ownB ownQ) _ L shape hR))
    (stmtFixes_deploy H rel (nextR rd) colsE wE FlockDeploy.joinR_head rowIn final w (nextR_coin rd H)
      (lastSplitP H rd pieces pholds hpieces).toLast (ownJ (recBound rel' b L IV) L (ownP ownB ownQ)) rel' b L IV
      shape hrest)
  refine decompBV_of_jointV ?_ hD
  intro x st hx hf c xs ws hJ s hs
  exact partComputes_of_shares H rd pieces pholds hpieces rel colsE wE rowIn final w ownB ownQ _ L
    (hlowR s x c _ ((joint_of_jointP (jointP_of_jointV hJ)).2.2.2 s hs)) (hlowP x st hx hf c xs ws hJ s hs)

end Compose

end RecDeploy.OpenLower
