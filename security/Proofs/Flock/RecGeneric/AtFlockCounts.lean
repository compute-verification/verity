module

public import Proofs.Flock.RecGeneric.AtFlockServer
public import Proofs.Flock.Claims.Recursion.BoundDeploy
public import Proofs.Flock.Claims.ERep

/-! A rerun's extra calls at Flock's deployment: `n_V ≤ n` and `w ≤ v`, the two facts about the deployed code `ksLinkOn_bits` takes. -/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

theorem collect_cons_eval {R α : Type} (H : Msg → Msg) (run : R → Transcript) (p : RP R (Option α))
    (ps : List (RP R (Option α))) :
    ((collect (p :: ps)).eval H run).1 =
      (p.eval H run).1.bind fun a => ((collect ps).eval H run).1.map fun l => a :: l := by
  rw [collect, (RP.eval_bind H run p _).1]
  cases (p.eval H run).1 with
  | none => rfl
  | some a =>
    show (((collect ps).bind fun os => RP.ret (os.map fun l => a :: l)).eval H run).1 = _
    exact (RP.eval_bind H run _ _).1

theorem collect_length {R α : Type} (H : Msg → Msg) (run : R → Transcript) :
    ∀ (ps : List (RP R (Option α))) (l : List α), ((collect ps).eval H run).1 = some l → l.length = ps.length
  | [], l, h => by
    have h' : some ([] : List α) = some l := h
    cases h'
    rfl
  | p :: ps, l, h => by
    rw [collect_cons_eval] at h
    generalize (p.eval H run).1 = op at h
    generalize hc : ((collect ps).eval H run).1 = oc at h
    cases op with
    | none =>
      have h' : (none : Option (List α)) = some l := h
      cases h'
    | some a =>
      cases oc with
      | none =>
        have h' : (none : Option (List α)) = some l := h
        cases h'
      | some os =>
        have h' : some (a :: os) = some l := h
        cases h'
        exact congrArg (· + 1) (collect_length H run ps os hc)

theorem length_flatten_mapIdx {α β : Type} (f : ℕ → α → List β) (c : ℕ → ℕ) (hc : ∀ s a, (f s a).length = c s)
    (ws : List α) : (ws.mapIdx f).flatten.length = ((List.range ws.length).map c).sum := by
  induction ws generalizing f c with
  | nil => simp
  | cons a ws ih =>
    rw [List.mapIdx_cons, List.flatten_cons, List.length_append, hc,
      ih (fun i => f (i + 1)) (fun i => c (i + 1)) (fun s a => hc (s + 1) a), List.length_cons,
      List.range_succ_eq_map, List.map_cons, List.sum_cons, List.map_map]
    all_goals rfl

section Instance

variable {Coin Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
  [DecidableEq Col']

theorem wits_length (D : RecAt Coin) (Bd : Bound Coin Col T) (DO : Claims.Flock.Deploy Coin Col' T')
    (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (u : (Fin ((compDep D Bd DO).V.k x) → Coin) × (compDep D Bd DO).EO.C x) (ws : List Claims.Flock.Wit)
    (h : (compDep D Bd DO).wits H P x ω u = some ws) : ws.length = (D.record Bd.IVStar).nS x := by
  unfold Comp.wits Extractor.eval at h
  dsimp only [compDep, compAt, Comp.ofRec, Claims.Recursion.EO, RP.eval] at h
  split at h
  · rw [collect_length H _ _ ws h, List.length_map, List.length_finRange]
  · have h' : (none : Option (List Claims.Flock.Wit)) = some ws := h
    cases h'

end Instance

section Deployed

open FlockDeploy (Coin)

variable {Col T Col' T' : Type} [DecidableEq Col] [DecidableEq Col']

theorem innerLog_length_recBoundAt (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : RecDeploy.Brings) (L : VStarB.LayoutB) (Bd : Bound Coin Col T) (DO : Claims.Flock.Deploy Coin Col' T')
    (H : Msg → Msg) (x : Msg) (r : Fin ((compDep (RecDeploy.recBoundAt rel b L) Bd DO).V.k x) → Coin)
    (ws : List Claims.Flock.Wit) :
    ((compDep (RecDeploy.recBoundAt rel b L) Bd DO).innerLog ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H x r
        ws).length =
      (((RecDeploy.recBoundAt rel b L).record Bd.IVStar).innerCalls H x (coinSeq r) ws).length := by
  have e : (Server.ofBound Bd).IVStar Bd.DE.V Bd.ans = Bd.IVStar := ivStarOfBound Bd
  rw [e]
  exact (List.length_map ..).symm

theorem innerLog_le_recBoundAt (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : RecDeploy.Brings) (L : VStarB.LayoutB) (Bd : Bound Coin Col T) (DO : Claims.Flock.Deploy Coin Col' T')
    (H : Msg → Msg) (x : Msg) (n : ℕ)
    (hn : ∀ cs ws, (((RecDeploy.recBoundAt rel b L).record Bd.IVStar).innerCalls H x cs ws).length ≤ n)
    (r : Fin ((compDep (RecDeploy.recBoundAt rel b L) Bd DO).V.k x) → Coin) (ws : List Claims.Flock.Wit) :
    ((compDep (RecDeploy.recBoundAt rel b L) Bd DO).innerLog ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H x r
      ws).length ≤ n := by
  rw [innerLog_length_recBoundAt]
  exact hn _ ws

theorem calls_length_recBoundAt (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : RecDeploy.Brings) (L : VStarB.LayoutB) (IV : Verifier Coin) (x : Msg) (ws : List Claims.Flock.Wit) :
    (((RecDeploy.recBoundAt rel b L).record IV).calls x ws).length =
      ((List.range ws.length).map fun s => (L.shaAt x s).length).sum := by
  show (ws.mapIdx fun s w => L.calls x s w).flatten.length = _
  exact length_flatten_mapIdx (fun s w => L.calls x s w) (fun s => (L.shaAt x s).length)
    (fun _ _ => List.length_map ..) ws

theorem calls_le_recBoundAt (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : RecDeploy.Brings) (L : VStarB.LayoutB) (Bd : Bound Coin Col T) (DO : Claims.Flock.Deploy Coin Col' T')
    (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (v : ℕ)
    (hv : ((List.range (RecDeploy.nS x)).map fun s => (L.shaAt x s).length).sum ≤ v) (ω : Ω)
    (u : (Fin ((compDep (RecDeploy.recBoundAt rel b L) Bd DO).V.k x) → Coin) ×
      (compDep (RecDeploy.recBoundAt rel b L) Bd DO).EO.C x) (ws : List Claims.Flock.Wit)
    (h : (compDep (RecDeploy.recBoundAt rel b L) Bd DO).wits H P x ω u = some ws) :
    (((RecDeploy.recBoundAt rel b L).record Bd.IVStar).calls x ws).length ≤ v := by
  rw [calls_length_recBoundAt, wits_length _ _ _ H P x ω u ws h]
  exact hv

end Deployed

end Claims.RecGeneric
