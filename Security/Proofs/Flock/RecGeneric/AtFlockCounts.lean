module

public import Proofs.Flock.RecGeneric.AtFlockServer
public import Proofs.Flock.Claims.Recursion.BoundDeploy
public import Proofs.Flock.Claims.ERep

/-!
# A rerun's extra calls at Flock's deployment

`ksLinkOn_bits` takes two facts about the deployed code beside route 2C's call bounds `n` (the inner verifier's calls
on the inner messages) and `v` (the compiled verifier's own at the V\* statements): a rerun's inner view's run makes at
most `n` calls (`n_V ≤ n`), and the calls `E_O`'s witnesses record number at most `v` (`w ≤ v`). At the compiled record
whose readers are V\*'s bound layout's (`RecDeploy.recBoundAt`):

* the inner view's run is the run `n` counts: `Open(DE, hm96)` is the bound record's `IVStar` (`ivStarOfBound`), run
  on the layout's messages at the inner statement and the rerun's coins, so `n_V` is the length of the record's
  `innerCalls` there (`innerLog_length_recBoundAt`) and `n_V ≤ n` wherever route 2C's `n` bounds them
  (`innerLog_le_recBoundAt`);
* `E_O` gives one witness per V\* session (`wits_length`), and part `s`'s witness records one call per SHA-512
  evaluation the layout places in part `s` at the statement (`VStarB.LayoutB.calls`), whatever the witness, so `w` is
  `∑_{s < nS(x)} |shaAt x s|`, a function of the statement alone (`calls_length_recBoundAt`), and `w ≤ v` wherever `v`
  is at least that (`calls_le_recBoundAt`).
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

/-- **`collect`'s result at a program list's head**: the head's result, then the rest's, the head's prepended; none
where either gives up. -/
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

/-- **`collect` gives one result per program**, where it gives any. -/
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

/-- **A flattened indexed map's length** where each index's list has a length fixed by the index alone. -/
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

/-- **`E_O` gives one witness per V\* session**: where it gives witnesses at a rerun, they are `collect`'s results over
the sessions. -/
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

/-- **`n_V` at the deployment is the record's inner calls' count**: a rerun's inner view's run is `Open(DE, hm96)`'s,
the bound record's `IVStar` (`ivStarOfBound`), on the layout's messages at the inner statement and the rerun's coins,
which is the run whose calls the record's `innerCalls` lists (`RecDeploy.innerCallsOf`). -/
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

/-- **`n_V ≤ n` at the deployment**, wherever route 2C's `n` bounds the record's inner calls. -/
theorem innerLog_le_recBoundAt (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : RecDeploy.Brings) (L : VStarB.LayoutB) (Bd : Bound Coin Col T) (DO : Claims.Flock.Deploy Coin Col' T')
    (H : Msg → Msg) (x : Msg) (n : ℕ)
    (hn : ∀ cs ws, (((RecDeploy.recBoundAt rel b L).record Bd.IVStar).innerCalls H x cs ws).length ≤ n)
    (r : Fin ((compDep (RecDeploy.recBoundAt rel b L) Bd DO).V.k x) → Coin) (ws : List Claims.Flock.Wit) :
    ((compDep (RecDeploy.recBoundAt rel b L) Bd DO).innerLog ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H x r
      ws).length ≤ n := by
  rw [innerLog_length_recBoundAt]
  exact hn _ ws

/-- **The calls the witnesses record at the deployment**: part `s`'s witness records one per SHA-512 evaluation the
layout places in part `s` (`VStarB.LayoutB.calls`), whatever the witness. -/
theorem calls_length_recBoundAt (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : RecDeploy.Brings) (L : VStarB.LayoutB) (IV : Verifier Coin) (x : Msg) (ws : List Claims.Flock.Wit) :
    (((RecDeploy.recBoundAt rel b L).record IV).calls x ws).length =
      ((List.range ws.length).map fun s => (L.shaAt x s).length).sum := by
  show (ws.mapIdx fun s w => L.calls x s w).flatten.length = _
  exact length_flatten_mapIdx (fun s w => L.calls x s w) (fun s => (L.shaAt x s).length)
    (fun _ _ => List.length_map ..) ws

/-- **`w ≤ v` at the deployment**, wherever `v` is at least the SHA-512 evaluations the layout places in the
statement's V\* parts: `E_O` gives one witness per part (`wits_length`), each recording its part's
(`calls_length_recBoundAt`). -/
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
