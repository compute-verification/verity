import Proofs.Flock.Claims.Recursion.BoundRoute
import Proofs.Flock.Claims.Recursion.LastBindsEC

/-!
# `KSBoundRoute` from the bridge, the last message bound modulo what `E*` reads (R1′)

`ksBoundRoute_of_bridge` reads (P4, `Bound.LastBinds`) once, in `forksBind_nf`, and (P4) was false at the deployed
record before S0 (a trailing space in the record's text, `BindsDeployed`). This file proves the route with (P4) replaced by `Bound.LastBindsEC` and
`Bound.LastCoinFree`, through REC1's chain as it stands (`KS_of_flockKS_binds`):

* The compiled record's inner view reads the last message as a representative of its reading (`RecDeploy.deployRecC`):
  `IVStar`'s decision with the set of its calls, and `DE`'s columns of the opened view (`Bound.readL`), at the coins
  before the last round (`Bound.repL`). Where `IVStar`'s last step reads no last coin (`LastCoinFree`), the
  representative keeps the decision, the calls as a set and the columns (`Bound.run_nfC`). `E*` reads an inner view
  only through its decision and its openings (`served_eval_congr`), so `E*`'s output and `KSA` are the deployed
  record's (`RecDeploy.ksA_of_C`), and since `CallsLogColl` reads the calls only by membership, the inner view's calls
  are still among the witnesses' (`RecDeploy.callsLogColl_C`).
* Two inner views that differ at the last round then differ in their readings, at coins that agree before it
  (`Bound.readL_ne_of_repL`). Both accept, so their call sets or their columns differ, and `LastBindsEC` gives a
  collision among `IVStar`'s calls, which the witnesses record (`RecDeploy.forksBind_C`).

The reading takes the calls as a set because `LastBindsEC` compares them so: the record's stream order and
its optional digest of a table's empty publics change the list of calls the deployed verifier makes, not the set
(`note:proofs/20261020T0132Z-finding-rec2-calls-as-lists`).

`RecDeploy.KSBoundRouteCols` is `KSBoundRoute` with `LastBindsEC` and `LastCoinFree` for `LastBinds`;
`RecDeploy.ksBoundRoute_of_bridge_cols` proves it from the bridge. At the deployed plain verifier's own steps
`LastCoinFree` holds (`RecDeploy.lastCoinFree_deployed`).
-/

namespace Claims.Recursion

open Claims

/-! ## A representative for each reading -/

/-- **A representative of `m` under `f`**: one message for each value `f` takes. -/
noncomputable def repBy {β : Type} (f : Msg → β) (m : Msg) : Msg :=
  Classical.epsilon fun m' => f m' = f m

theorem repBy_spec {β : Type} (f : Msg → β) (m : Msg) : f (repBy f m) = f m :=
  Classical.epsilon_spec (p := fun m' => f m' = f m) ⟨m, rfl⟩

theorem repBy_congr {β : Type} (f : Msg → β) {m m' : Msg} (h : f m = f m') : repBy f m = repBy f m' := by
  unfold repBy
  rw [h]

theorem range_map_getD_succ (ms : List Msg) (n : ℕ) :
    (List.range (n + 1)).map (fun i => ms.getD i []) = (List.range n).map (fun i => ms.getD i []) ++ [ms.getD n []] := by
  rw [List.range_succ, List.map_append, List.map_singleton]

theorem range_map_getD_pred (ms : List Msg) {k : ℕ} (hk : 0 < k) :
    (List.range k).map (fun i => ms.getD i []) =
      (List.range (k - 1)).map (fun i => ms.getD i []) ++ [ms.getD (k - 1) []] := by
  obtain ⟨n, rfl⟩ : ∃ n, k = n + 1 := ⟨k - 1, by omega⟩
  rw [Nat.add_sub_cancel]
  exact range_map_getD_succ ms n

theorem range_map_dropLast {α : Type} (f : ℕ → α) :
    ∀ k : ℕ, ((List.range k).map f).dropLast = (List.range (k - 1)).map f
  | 0 => rfl
  | n + 1 => by rw [List.range_succ, List.map_append, List.map_singleton, List.dropLast_concat, Nat.add_sub_cancel]

theorem zip_map_range {α β : Type} (g : ℕ → α) (h : ℕ → β) :
    ∀ n : ℕ, ((List.range n).map g).zip ((List.range n).map h) = (List.range n).map fun i => (g i, h i)
  | 0 => rfl
  | n + 1 => by
    simp only [List.range_succ, List.map_append, List.map_singleton]
    rw [List.zip_append (by simp only [List.length_map, List.length_range] :
      ((List.range n).map g).length = ((List.range n).map h).length)]
    exact congrArg (· ++ [(g n, h n)]) (zip_map_range g h n)

section Reading

variable {Coin Col T : Type} [Inhabited Coin]

/-- **What `E*` reads of `IVStar`'s decision, with `IVStar`'s calls as a set**: the decision on the messages `as` at
the coins `cs` with the set of its calls, and `DE`'s columns of the opened view. -/
def Bound.readL (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) (as : List Msg) (cs : List Coin) :=
  ((((Bd.IVStar.next x as cs).eval H).1, ((Bd.IVStar.next x as cs).eval H).2.toFinset),
    Bd.DE.cols (Bd.openView (Bd.viewAt H x as cs)))

/-- **The reading doesn't read the last coin** where `IVStar`'s last step doesn't. -/
theorem Bound.readL_coins {H : Msg → Msg} {Bd : Bound Coin Col T} (hCF : Bd.LastCoinFree H) {x : Msg}
    {as : List Msg} {cs cs' : List Coin} (hk : as.length = Bd.DE.V.k x) (hc : cs.length = as.length)
    (hc' : cs'.length = as.length) (hd : cs.dropLast = cs'.dropLast) :
    Bd.readL H x as cs = Bd.readL H x as cs' := by
  unfold Bound.readL Bound.viewAt
  rw [hCF x as cs cs' hk hc hc' hd, hd]

theorem Bound.readL_default {H : Msg → Msg} {Bd : Bound Coin Col T} (hCF : Bd.LastCoinFree H) (x : Msg)
    (as : List Msg) (cs : List Coin) (hk : as.length = Bd.DE.V.k x) (hc : cs.length = as.length)
    (h0 : 0 < as.length) : Bd.readL H x as cs = Bd.readL H x as (cs.dropLast ++ [default]) :=
  Bd.readL_coins hCF hk hc
    (by simp only [List.length_append, List.length_dropLast, List.length_singleton]; omega)
    (by rw [List.dropLast_concat])

/-- Two readings of accepted decisions that differ, differ in the columns or in the call sets. -/
theorem Bound.readL_ne {H : Msg → Msg} {Bd : Bound Coin Col T} {x : Msg} {as as' : List Msg} {cs cs' : List Coin}
    (h : Bd.readL H x as cs ≠ Bd.readL H x as' cs')
    (ha : ((Bd.IVStar.next x as cs).eval H).1 = ((Bd.IVStar.next x as' cs').eval H).1) :
    Bd.DE.cols (Bd.openView (Bd.viewAt H x as cs)) ≠ Bd.DE.cols (Bd.openView (Bd.viewAt H x as' cs')) ∨
      ((Bd.IVStar.next x as cs).eval H).2.toFinset ≠ ((Bd.IVStar.next x as' cs').eval H).2.toFinset := by
  by_cases hc : Bd.DE.cols (Bd.openView (Bd.viewAt H x as cs)) = Bd.DE.cols (Bd.openView (Bd.viewAt H x as' cs'))
  · exact Or.inr fun hl => h (Prod.ext (Prod.ext ha hl) hc)
  · exact Or.inl hc

/-- **The last message's representative modulo `readL`**: at the messages before the last that `ms` holds and the
coins before the last of `cs`, one message for each reading. -/
noncomputable def Bound.repL (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) (ms : List Msg) (cs : List Coin) :
    Msg :=
  repBy (fun m => Bd.readL H x ((List.range (Bd.DE.V.k x - 1)).map (fun i => ms.getD i []) ++ [m])
    (cs.dropLast ++ [default])) (ms.getD (Bd.DE.V.k x - 1) [])

/-- **Messages with the last read as its representative modulo `readL`**: `IVStar`'s rounds, each the message `ms`
holds there, the last one's representative at the coins `cs`. -/
noncomputable def Bound.nfMsgsC (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) (cs : List Coin) (ms : List Msg) :
    List Msg :=
  (List.range (Bd.DE.V.k x)).map fun i => if i + 1 = Bd.DE.V.k x then Bd.repL H x ms cs else ms.getD i []

theorem Bound.nfMsgsC_getD_lt (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) (cs : List Coin) (ms : List Msg)
    {i : ℕ} (hi : i + 1 < Bd.DE.V.k x) : (Bd.nfMsgsC H x cs ms).getD i [] = ms.getD i [] := by
  have h1 : i < Bd.DE.V.k x := by omega
  have h2 : i + 1 ≠ Bd.DE.V.k x := by omega
  simp [Bound.nfMsgsC, List.getD_eq_getElem?_getD, h1, h2]

theorem Bound.nfMsgsC_getD_eq (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) (cs : List Coin) (ms : List Msg)
    {i : ℕ} (hi : i + 1 = Bd.DE.V.k x) : (Bd.nfMsgsC H x cs ms).getD i [] = Bd.repL H x ms cs := by
  have h1 : i < Bd.DE.V.k x := by omega
  simp [Bound.nfMsgsC, List.getD_eq_getElem?_getD, h1, hi]

/-- **The representative keeps the reading**, where `IVStar`'s last step reads no last coin. -/
theorem Bound.readL_nf {H : Msg → Msg} {Bd : Bound Coin Col T} (hCF : Bd.LastCoinFree H) (x : Msg) (ms : List Msg)
    (cs : List Coin) (hc : cs.length = Bd.DE.V.k x) :
    Bd.readL H x ((List.range (Bd.DE.V.k x)).map fun i => (Bd.nfMsgsC H x cs ms).getD i []) cs =
      Bd.readL H x ((List.range (Bd.DE.V.k x)).map fun i => ms.getD i []) cs := by
  rcases Nat.eq_zero_or_pos (Bd.DE.V.k x) with hk | hk
  · simp only [hk, List.range_zero, List.map_nil]
  have hpre : (List.range (Bd.DE.V.k x - 1)).map (fun i => (Bd.nfMsgsC H x cs ms).getD i []) =
      (List.range (Bd.DE.V.k x - 1)).map (fun i => ms.getD i []) :=
    List.map_congr_left fun i hi => Bd.nfMsgsC_getD_lt H x cs ms (by rw [List.mem_range] at hi; omega)
  have hlen : ∀ m : Msg, ((List.range (Bd.DE.V.k x - 1)).map (fun i => ms.getD i []) ++ [m]).length =
      Bd.DE.V.k x := fun m => by
    simp only [List.length_append, List.length_map, List.length_range, List.length_singleton]
    omega
  rw [range_map_getD_pred _ hk, range_map_getD_pred _ hk, hpre,
    Bd.nfMsgsC_getD_eq H x cs ms (by omega : Bd.DE.V.k x - 1 + 1 = Bd.DE.V.k x),
    Bd.readL_default hCF x ((List.range (Bd.DE.V.k x - 1)).map (fun i => ms.getD i []) ++ [Bd.repL H x ms cs]) cs
      (hlen _) (hc.trans (hlen _).symm) (by rw [hlen]; exact hk),
    Bd.readL_default hCF x ((List.range (Bd.DE.V.k x - 1)).map (fun i => ms.getD i []) ++
      [ms.getD (Bd.DE.V.k x - 1) []]) cs (hlen _) (hc.trans (hlen _).symm) (by rw [hlen]; exact hk)]
  unfold Bound.repL
  exact repBy_spec (fun m => Bd.readL H x ((List.range (Bd.DE.V.k x - 1)).map (fun i => ms.getD i []) ++ [m])
    (cs.dropLast ++ [default])) (ms.getD (Bd.DE.V.k x - 1) [])

/-- **Two representatives that differ have readings that differ**, at messages and coins that agree before the last
round, where `IVStar`'s last step reads no last coin. -/
theorem Bound.readL_ne_of_repL {H : Msg → Msg} {Bd : Bound Coin Col T} (hCF : Bd.LastCoinFree H) {x : Msg}
    {ms ms' : List Msg} {cs cs' : List Coin} (hk : 0 < Bd.DE.V.k x) (hc : cs.length = Bd.DE.V.k x)
    (hc' : cs'.length = Bd.DE.V.k x) (hm : ∀ i, i + 1 < Bd.DE.V.k x → ms.getD i [] = ms'.getD i [])
    (hd : cs.dropLast = cs'.dropLast) (hne : Bd.repL H x ms cs ≠ Bd.repL H x ms' cs') :
    Bd.readL H x ((List.range (Bd.DE.V.k x)).map fun i => ms.getD i []) cs ≠
      Bd.readL H x ((List.range (Bd.DE.V.k x)).map fun i => ms'.getD i []) cs' := by
  intro h
  apply hne
  have e : ∀ (l : List Msg) (c : List Coin), c.length = Bd.DE.V.k x →
      (∀ i, i + 1 < Bd.DE.V.k x → l.getD i [] = ms'.getD i []) → c.dropLast = cs'.dropLast →
      Bd.readL H x ((List.range (Bd.DE.V.k x)).map fun i => l.getD i []) c =
        Bd.readL H x ((List.range (Bd.DE.V.k x - 1)).map (fun i => ms'.getD i []) ++ [l.getD (Bd.DE.V.k x - 1) []])
          (cs'.dropLast ++ [default]) := by
    intro l c hcl hl hcd
    have hp : (List.range (Bd.DE.V.k x - 1)).map (fun i => l.getD i []) =
        (List.range (Bd.DE.V.k x - 1)).map (fun i => ms'.getD i []) :=
      List.map_congr_left fun i hi => hl i (by rw [List.mem_range] at hi; omega)
    have hlen : ((List.range (Bd.DE.V.k x - 1)).map (fun i => ms'.getD i []) ++
        [l.getD (Bd.DE.V.k x - 1) []]).length = Bd.DE.V.k x := by
      simp only [List.length_append, List.length_map, List.length_range, List.length_singleton]
      omega
    rw [range_map_getD_pred l hk, hp, ← hcd]
    exact Bd.readL_default hCF x _ c hlen (hcl.trans hlen.symm) (by rw [hlen]; exact hk)
  rw [e ms cs hc hm hd, e ms' cs' hc' (fun _ _ => rfl) rfl] at h
  have hpre : (List.range (Bd.DE.V.k x - 1)).map (fun i => ms.getD i []) =
      (List.range (Bd.DE.V.k x - 1)).map (fun i => ms'.getD i []) :=
    List.map_congr_left fun i hi => hm i (by rw [List.mem_range] at hi; omega)
  unfold Bound.repL
  rw [hpre, hd]
  exact repBy_congr (fun m => Bd.readL H x ((List.range (Bd.DE.V.k x - 1)).map (fun i => ms'.getD i []) ++ [m])
    (cs'.dropLast ++ [default])) h

end Reading

section Run

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- **`IVStar`'s run on replayed messages shows `viewAt`'s view**, at its first `k` messages and its coins. -/
theorem Bound.seen_viewAt (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) (ms : List Msg)
    (r : Fin (Bd.IVStar.k x) → Coin) (hk : 0 < Bd.DE.V.k x) :
    (run H (replay ms) Bd.IVStar x () r).seen =
      Bd.viewAt H x ((List.range (Bd.DE.V.k x)).map fun i => ms.getD i [])
        ((List.range (Bd.DE.V.k x)).map (coinSeq r)) := by
  rw [Bd.run_ivstar H x ms r hk]
  obtain ⟨n, hn⟩ : ∃ n, Bd.DE.V.k x = n + 1 := ⟨_, (Nat.succ_pred_eq_of_pos hk).symm⟩
  unfold Bound.viewAt Bound.decAt Trace.seen
  generalize (Bd.IVStar.next x ((List.range (Bd.DE.V.k x)).map fun i => ms.getD i [])
    ((List.range (Bd.DE.V.k x)).map (coinSeq r))).eval H = d
  rw [hn, Nat.add_sub_cancel, range_map_getD_succ, List.range_succ, List.map_append, List.map_singleton,
    List.dropLast_concat, List.map_map]
  simp only [List.zip_append, List.length_map, List.length_range, zip_map_range]
  rfl

/-- **`IVStar`'s run on the representative**: the same decision, the same calls as a set and the same columns of the
opened view, where its last step reads no last coin. -/
theorem Bound.run_nfC (Bd : Bound Coin Col T) (H : Msg → Msg) (hCF : Bd.LastCoinFree H) (x : Msg) (ms : List Msg)
    (r : Fin (Bd.IVStar.k x) → Coin) :
    (run H (replay (Bd.nfMsgsC H x ((List.range (Bd.DE.V.k x)).map (coinSeq r)) ms)) Bd.IVStar x () r).out =
        (run H (replay ms) Bd.IVStar x () r).out ∧
      (run H (replay (Bd.nfMsgsC H x ((List.range (Bd.DE.V.k x)).map (coinSeq r)) ms)) Bd.IVStar x () r).vLog.toFinset =
        (run H (replay ms) Bd.IVStar x () r).vLog.toFinset ∧
      Bd.DE.cols (Bd.openView (run H (replay (Bd.nfMsgsC H x ((List.range (Bd.DE.V.k x)).map (coinSeq r)) ms))
        Bd.IVStar x () r).seen) = Bd.DE.cols (Bd.openView (run H (replay ms) Bd.IVStar x () r).seen) := by
  rcases Nat.eq_zero_or_pos (Bd.DE.V.k x) with hk | hk
  · rw [Bd.run_ivstar_zero H x _ r hk, Bd.run_ivstar_zero H x _ r hk]
    exact ⟨rfl, rfl, rfl⟩
  have h := Bd.readL_nf hCF x ms ((List.range (Bd.DE.V.k x)).map (coinSeq r)) (by simp)
  have hdec : ((Bd.decAt H x (Bd.nfMsgsC H x ((List.range (Bd.DE.V.k x)).map (coinSeq r)) ms) r).1,
      (Bd.decAt H x (Bd.nfMsgsC H x ((List.range (Bd.DE.V.k x)).map (coinSeq r)) ms) r).2.toFinset) =
      ((Bd.decAt H x ms r).1, (Bd.decAt H x ms r).2.toFinset) := congrArg Prod.fst h
  obtain ⟨h1, h2⟩ := Prod.mk.inj hdec
  refine ⟨?_, ?_, ?_⟩
  · rw [Bd.run_ivstar H x _ r hk, Bd.run_ivstar H x _ r hk]
    exact h1
  · rw [Bd.run_ivstar H x _ r hk, Bd.run_ivstar H x _ r hk]
    exact h2
  · rw [Bd.seen_viewAt H x _ r hk, Bd.seen_viewAt H x _ r hk]
    exact congrArg Prod.snd h

end Run

end Claims.Recursion

namespace RecDeploy

open Claims (Msg Out Prover run coinSeq replay findColl Collision)
open FlockDeploy (Coin)
open Claims.Recursion (Bound Inner)

section Cols

variable {Col T : Type}

/-- The run's inner coins at `r`, as `IVStar`'s list of coins at the inner statement. -/
def coinsC (Bd : Bound Coin Col T) (x : Msg) {k : ℕ} (r : Fin k → Coin) : List Coin :=
  (List.range (Bd.DE.V.k (xIOf x))).map (coinSeq fun i : Fin (Bd.IVStar.k (xIOf x)) => coinSeq r i.val)

/-- The readers' messages with the last read as its representative modulo `readL`, at the run's inner coins. -/
noncomputable def msgsC (Bd : Bound Coin Col T) (b : Brings) (H : Msg → Msg) (x : Msg) {k : ℕ} (r : Fin k → Coin)
    (ws : List Claims.Flock.Wit) : List Msg :=
  Bd.nfMsgsC H (xIOf x) (coinsC Bd x r) (b.rd.msgs x ws)

/-- **The compiled record, its inner view reading the last message as its representative modulo what `E*` reads**
(`Bound.readL`, at the run's coins before the last round): `deployRec` at `IVStar` otherwise. -/
noncomputable def deployRecC (Bd : Bound Coin Col T) (ivScope : Msg → Bool) (b : Brings) :
    Claims.Recursion.Deploy Coin :=
  { deployRec Bd.IVStar ivScope b with
    innerView := fun H x r ws =>
      (run H (replay (msgsC Bd b H x r ws)) Bd.IVStar (xIOf x) () fun i => coinSeq r i.val).seen }

/-- The inner view's run, at `deployRecC`, is `IVStar`'s run of the representative messages. -/
theorem innerRun_C (H : Msg → Msg) (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T) (ivScope : Msg → Bool)
    (b : Brings) (x : Msg) (r : Fin ((deployRecC Bd ivScope b).V.k x) → Coin) (ws : List Claims.Flock.Wit) :
    Claims.Recursion.innerRun H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) x r ws =
      run H (replay (msgsC Bd b H x r ws)) Bd.IVStar (xIOf x) () fun i => coinSeq r i.val := by
  show run H (replay ((run H (replay (msgsC Bd b H x r ws)) Bd.IVStar (xIOf x) () fun i => coinSeq r i.val).seen.msgs.map
    Prod.fst)) Bd.IVStar (xIOf x) () (fun i => coinSeq r i.val) = _
  rw [run_replay_seen]

theorem innerViewRun_C {Col' T' : Type} (H : Msg → Msg) (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T)
    (ivScope : Msg → Bool) (b : Brings) (DO : Claims.Flock.Deploy Coin Col' T') :
    Claims.Recursion.InnerViewRun H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO := by
  intro decO x Ω P ω r c ws _
  exact (congrArg Claims.Trace.seen (innerRun_C H Bd decI ivScope b x r ws)).symm

theorem innerView_C [DecidableEq Col] (H : Msg → Msg) (Bd : Bound Coin Col T) (ivScope : Msg → Bool) (b : Brings)
    (hCF : Bd.LastCoinFree H) (x : Msg) (r : Fin ((deployRec Bd.IVStar ivScope b).V.k x) → Coin)
    (ws : List Claims.Flock.Wit) :
    ((deployRecC Bd ivScope b).innerView H x r ws).out = ((deployRec Bd.IVStar ivScope b).innerView H x r ws).out ∧
      Bd.DE.cols (Bd.openView ((deployRecC Bd ivScope b).innerView H x r ws)) =
        Bd.DE.cols (Bd.openView ((deployRec Bd.IVStar ivScope b).innerView H x r ws)) :=
  let h := Bd.run_nfC H hCF (xIOf x) (b.rd.msgs x ws) fun i => coinSeq r i.val
  ⟨h.1, h.2.2⟩

/-- **The inner view's calls are still among the witnesses'**: the representative's calls are the deployed record's as
a set, and `CallsLogColl` reads them only by membership. -/
theorem callsLogColl_C {Col' T' : Type} [DecidableEq Col] (H : Msg → Msg) (Bd : Bound Coin Col T)
    (decI : List (ℕ × Col) → List T) (ivScope : Msg → Bool) (b : Brings) (DO : Claims.Flock.Deploy Coin Col' T')
    (hCF : Bd.LastCoinFree H)
    (hL : Claims.Recursion.CallsLogColl H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO) :
    Claims.Recursion.CallsLogColl H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO := by
  intro decO x Ω P ω r c ws hg
  rcases hL decO x Ω P ω r c ws hg with h | h
  · left
    intro e he
    rw [innerRun_C] at he
    apply h
    have hv : (run H (replay (msgsC Bd b H x r ws)) Bd.IVStar (xIOf x) () fun i => coinSeq r i.val).vLog.toFinset =
        (Claims.Recursion.innerRun H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) x r
          ws).vLog.toFinset :=
      (Bd.run_nfC H hCF (xIOf x) (b.rd.msgs x ws) fun i => coinSeq r i.val).2.1.trans
        (congrArg (fun t => t.vLog.toFinset) (innerRun_bound H Bd decI ivScope b x r ws)).symm
    exact List.mem_toFinset.1 (hv ▸ List.mem_toFinset.2 he)
  · exact Or.inr h

theorem innerAcceptsCollAt_C {Col' T' : Type} [DecidableEq Col] (H : Msg → Msg) (Bd : Bound Coin Col T)
    (decI : List (ℕ × Col) → List T) (ivScope : Msg → Bool) (b : Brings) (DO : Claims.Flock.Deploy Coin Col' T')
    (hCF : Bd.LastCoinFree H) (x : Msg)
    (hA : Claims.Recursion.InnerAcceptsCollAt H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO
      x) :
    Claims.Recursion.InnerAcceptsCollAt H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO x := by
  intro decO Ω P ω r c ws hg
  rcases hA decO Ω P ω r c ws hg with h | h
  · exact Or.inl ((innerView_C H Bd ivScope b hCF x r ws).1.trans h)
  · exact Or.inr h

theorem innerAcceptsColl_C {Col' T' : Type} [DecidableEq Col] (H : Msg → Msg) (Bd : Bound Coin Col T)
    (decI : List (ℕ × Col) → List T) (ivScope : Msg → Bool) (b : Brings) (DO : Claims.Flock.Deploy Coin Col' T')
    (hCF : Bd.LastCoinFree H)
    (hA : Claims.Recursion.InnerAcceptsColl H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO) :
    Claims.Recursion.InnerAcceptsColl H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO :=
  fun decO x => innerAcceptsCollAt_C H Bd decI ivScope b DO hCF x (hA.toAt x) decO

/-- **`E*` gives one output at the deployed record and at `deployRecC`.** -/
theorem E_eval_C {Col' T' : Type} [DecidableEq Col] (H : Msg → Msg) (Bd : Bound Coin Col T)
    (decI : List (ℕ × Col) → List T) (ivScope : Msg → Bool) (b : Brings) (DO : Claims.Flock.Deploy Coin Col' T')
    (decO : List (ℕ × Col') → List T') (hCF : Bd.LastCoinFree H) (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω)
    (r : Fin ((deployRec Bd.IVStar ivScope b).V.k x) → Coin)
    (c : (Claims.Recursion.E H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO).C x) :
    ((Claims.Recursion.E H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO).eval H P x ω r
        c).1 =
      ((Claims.Recursion.E H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO).eval H P x ω r
        c).1 := by
  refine Claims.Recursion.served_eval_congr H P _ _ _
    (Claims.Recursion.dec H (deployRecC Bd ivScope b)) _ _ _ _ _ _ x (fun r o => ?_) ω r c
  cases o with
  | none => exact ⟨rfl, rfl⟩
  | some ws =>
    obtain ⟨h1, h2⟩ := innerView_C H Bd ivScope b hCF x r ws
    exact ⟨h1.symm, congrArg (fun v => [v]) h2.symm⟩

/-- **`KSA` at `deployRecC` is `KSA` at the deployed record**: `E*`'s output is the same (`E_eval_C`), and the bound
reads no inner view. -/
theorem ksA_of_C {Col' T' : Type} [DecidableEq Col] [DecidableEq Col'] [Zero Col'] (H : Msg → Msg)
    (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T) (ivScope : Msg → Bool) (b : Brings)
    (DO : Claims.Flock.Deploy Coin Col' T') (hCF : Bd.LastCoinFree H)
    (h : Claims.Recursion.KSA H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO) :
    Claims.Recursion.KSA H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO := by
  intro decO hdec x Ω _ _ P
  have hE : Claims.PrBad H (deployRec Bd.IVStar ivScope b).V
      (Claims.Recursion.R (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI))
      (Claims.Recursion.E H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO) x P =
    Claims.PrBad H (deployRecC Bd ivScope b).V
      (Claims.Recursion.R (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI))
      (Claims.Recursion.E H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO) x P := by
    unfold Claims.PrBad
    exact Claims.Pr_congr fun s => by
      rw [E_eval_C H Bd decI ivScope b DO decO hCF x P s.1 s.2.1 s.2.2]
      try exact Iff.rfl
  exact hE.trans_le (h decO hdec x Ω P)

/-- **The inner rounds' forks bind at `deployRecC`, at the statement `x`, with the readers' rounds before the last
read through a choice `pk` of the runs'** (`ForksBindAt`): two inner views at `Be`'s fork at round `j` that differ at
message `j` differ before the last round, where the readers' messages are `pk` of the runs' (`hP`) and the runs share
their coins before `j`, so their messages before `j`; or at the last, where the two representatives differ, so the
readings do (`Bound.readL_ne_of_repL`). Both views accept (`hA`), so the calls or the columns differ, and
`LastBindsEC` gives a collision among `IVStar`'s calls, which the witnesses record (`hL`). -/
theorem forksBindAt_C_pick {Col' T' : Type} [DecidableEq Col] (H : Msg → Msg) (Bd : Bound Coin Col T)
    (decI : List (ℕ × Col) → List T) (ivScope : Msg → Bool) (b : Brings) (DO : Claims.Flock.Deploy Coin Col' T')
    (pk : Msg → ℕ → Msg → Msg) (hEC : Bd.LastBindsEC H) (hCF : Bd.LastCoinFree H)
    (hT : Claims.Recursion.CallsTrue H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO)
    (hL : Claims.Recursion.CallsLogColl H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO)
    (x : Msg) (hP : ∀ (decO : List (ℕ × Col') → List T') (Ω : Type) (P : Prover Ω) (ω : Ω)
      (r : Fin ((deployRec Bd.IVStar ivScope b).V.k x) → Coin)
      (c : (Claims.Recursion.EO (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO).C x)
      (ws : List Claims.Flock.Wit),
      Claims.Recursion.EOGives H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO P x ω r c
        ws →
      (∀ j, j + 1 < Bd.DE.V.k (xIOf x) → (b.rd.msgs x ws).getD j [] =
          pk x j (((run H P (deployRec Bd.IVStar ivScope b).V x ω r).msgs.map Prod.fst).getD j [])) ∨
        ∃ u v, findColl ((deployRec Bd.IVStar ivScope b).calls x ws) = some (u, v) ∧ Collision H u v)
    (hA : Claims.Recursion.InnerAcceptsCollAt H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO
      x) :
    Claims.Recursion.ForksBindAt H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO x := by
  intro decO Ω P j t hod
  obtain ⟨ws, ws', hg₁, hg₂, hne⟩ := hod
  have hg₁' : Claims.Recursion.EOGives H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO P x
      t.1 t.2.1 t.2.2.2.1 ws := hg₁
  have hg₂' : Claims.Recursion.EOGives H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO P x
      t.1 (Claims.Recursion.fork j t.2.1 t.2.2.1) t.2.2.2.2 ws' := hg₂
  have htrue : ∀ p ∈ (deployRec Bd.IVStar ivScope b).calls x ws ++ (deployRec Bd.IVStar ivScope b).calls x ws',
      H p.1 = p.2 := fun p hp =>
    (List.mem_append.1 hp).elim (hT decO x Ω P _ _ _ ws hg₁' p) (hT decO x Ω P _ _ _ ws' hg₂' p)
  have hlift : ∀ l : List (Msg × Msg),
      (∀ p ∈ l, p ∈ (deployRec Bd.IVStar ivScope b).calls x ws ++ (deployRec Bd.IVStar ivScope b).calls x ws') →
      (∃ u v, findColl l = some (u, v) ∧ Collision H u v) →
      Claims.Recursion.forkColl H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO P x j t := by
    rintro l hsub ⟨u, v, huv, -⟩
    unfold Claims.Recursion.forkColl
    rw [hg₁.2, hg₂.2]
    exact Claims.Recursion.VStar.coll_of_sub htrue hsub (by simp [huv])
  have hleft := hlift ((deployRec Bd.IVStar ivScope b).calls x ws) fun p hp => List.mem_append_left _ hp
  have hright := hlift ((deployRec Bd.IVStar ivScope b).calls x ws') fun p hp => List.mem_append_right _ hp
  have hv : ∀ (r₀ : Fin ((deployRecC Bd ivScope b).V.k x) → Coin) (ws₀ : List Claims.Flock.Wit),
      (((deployRecC Bd ivScope b).innerView H x r₀ ws₀).msgs.map Prod.fst).getD j [] =
        if j < Bd.DE.V.k (xIOf x) then (msgsC Bd b H x r₀ ws₀).getD j [] else [] :=
    fun r₀ ws₀ => Bd.run_ivstar_fst H (xIOf x) (msgsC Bd b H x r₀ ws₀) (fun i => coinSeq r₀ i.val) j
  rw [hv, hv] at hne
  by_cases hjK : j < Bd.DE.V.k (xIOf x)
  swap
  · exact absurd (by simp [hjK]) hne
  simp only [hjK, ↓reduceIte] at hne
  rcases hP decO Ω P t.1 t.2.1 t.2.2.2.1 ws hg₁' with h₁ | h₁
  swap
  · exact hleft h₁
  rcases hP decO Ω P t.1 (Claims.Recursion.fork j t.2.1 t.2.2.1) t.2.2.2.2 ws' hg₂' with h₂ | h₂
  swap
  · exact hright h₂
  have hag : ∀ i, i + 1 < Bd.DE.V.k (xIOf x) → i ≤ j → (b.rd.msgs x ws).getD i [] = (b.rd.msgs x ws').getD i [] :=
    fun i hi hij => (h₁ i hi).trans ((congrArg (pk x i) (Claims.Recursion.run_fst_getD H P _ x t.1 i fun i' hi' =>
      Claims.Recursion.fork_agree j t.2.1 t.2.2.1 i' (by omega))).trans (h₂ i hi).symm)
  unfold msgsC at hne
  by_cases hj1 : j + 1 < Bd.DE.V.k (xIOf x)
  · rw [Bd.nfMsgsC_getD_lt _ _ _ _ hj1, Bd.nfMsgsC_getD_lt _ _ _ _ hj1] at hne
    exact absurd (hag j hj1 le_rfl) hne
  -- The last round.
  have hk0 : 0 < Bd.DE.V.k (xIOf x) := by omega
  have hj : j + 1 = Bd.DE.V.k (xIOf x) := by omega
  rw [Bd.nfMsgsC_getD_eq H (xIOf x) _ _ hj, Bd.nfMsgsC_getD_eq H (xIOf x) _ _ hj] at hne
  have hcs : ∀ i, i < j → coinSeq (fun i : Fin (Bd.IVStar.k (xIOf x)) => coinSeq t.2.1 i.val) i =
      coinSeq (fun i : Fin (Bd.IVStar.k (xIOf x)) => coinSeq (Claims.Recursion.fork j t.2.1 t.2.2.1) i.val) i :=
    fun i hi => Claims.Recursion.coinSeq_congr fun _ =>
      Claims.Recursion.coinSeq_congr fun h => Claims.Recursion.fork_agree j t.2.1 t.2.2.1 ⟨i, h⟩ hi
  have hcsd : (coinsC Bd x t.2.1).dropLast = (coinsC Bd x (Claims.Recursion.fork j t.2.1 t.2.2.1)).dropLast := by
    unfold coinsC
    rw [Claims.Recursion.range_map_dropLast, Claims.Recursion.range_map_dropLast]
    exact List.map_congr_left fun i hi => hcs i (by rw [List.mem_range] at hi; omega)
  have hR := Bd.readL_ne_of_repL hCF hk0 (by simp [coinsC]) (by simp [coinsC]) (fun i hi => hag i hi (by omega))
    hcsd hne
  rcases hA decO Ω P t.1 t.2.1 t.2.2.2.1 ws hg₁' with ha₁ | ha₁
  swap
  · exact hleft ha₁
  rcases hA decO Ω P t.1 (Claims.Recursion.fork j t.2.1 t.2.2.1) t.2.2.2.2 ws' hg₂' with ha₂ | ha₂
  swap
  · exact hright ha₂
  rcases hL decO x Ω P t.1 t.2.1 t.2.2.2.1 ws hg₁' with hl₁ | hl₁
  swap
  · exact hleft hl₁
  rcases hL decO x Ω P t.1 (Claims.Recursion.fork j t.2.1 t.2.2.1) t.2.2.2.2 ws' hg₂' with hl₂ | hl₂
  swap
  · exact hright hl₂
  have hacc : ∀ (r₀ : Fin ((deployRec Bd.IVStar ivScope b).V.k x) → Coin) (ws₀ : List Claims.Flock.Wit),
      ((deployRec Bd.IVStar ivScope b).innerView H x r₀ ws₀).out = .acc →
      (Bd.decAt H (xIOf x) (b.rd.msgs x ws₀) fun i => coinSeq r₀ i.val).1 = .acc := by
    intro r₀ ws₀ h
    change (run H (replay (b.rd.msgs x ws₀)) Bd.IVStar (xIOf x) () fun i => coinSeq r₀ i.val).out = .acc at h
    rw [Bd.run_ivstar H (xIOf x) _ _ hk0] at h
    exact h
  have hlog : ∀ (r₀ : Fin ((deployRec Bd.IVStar ivScope b).V.k x) → Coin) (ws₀ : List Claims.Flock.Wit),
      (∀ e ∈ (Claims.Recursion.innerRun H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) x r₀
        ws₀).vLog, e ∈ (deployRec Bd.IVStar ivScope b).calls x ws₀) →
      ∀ e ∈ (Bd.decAt H (xIOf x) (b.rd.msgs x ws₀) fun i => coinSeq r₀ i.val).2,
        e ∈ (deployRec Bd.IVStar ivScope b).calls x ws₀ := by
    intro r₀ ws₀ h e he
    apply h
    rw [innerRun_bound, Bd.run_ivstar H (xIOf x) _ _ hk0]
    exact he
  have hacc₁ := hacc _ _ ha₁
  have hacc₂ := hacc _ _ ha₂
  have hlog₁ := hlog _ _ hl₁
  have hlog₂ := hlog _ _ hl₂
  obtain ⟨q, hq, q', hq', hc⟩ := hEC (xIOf x)
    ((List.range (Bd.DE.V.k (xIOf x))).map fun i => (b.rd.msgs x ws).getD i [])
    ((List.range (Bd.DE.V.k (xIOf x))).map fun i => (b.rd.msgs x ws').getD i [])
    (coinsC Bd x t.2.1) (coinsC Bd x (Claims.Recursion.fork j t.2.1 t.2.2.1))
    (by simp) (by simp) (by simp [coinsC]) (by simp [coinsC])
    (by
      rw [Claims.Recursion.range_map_dropLast, Claims.Recursion.range_map_dropLast]
      exact List.map_congr_left fun i hi => by
        rw [List.mem_range] at hi
        exact hag i (by omega) (by omega))
    hcsd (Bd.readL_ne hR (hacc₁.trans hacc₂.symm)) hacc₁ hacc₂
  have hsub : ∀ e ∈ (Bd.decAt H (xIOf x) (b.rd.msgs x ws) fun i => coinSeq t.2.1 i.val).2 ++
      (Bd.decAt H (xIOf x) (b.rd.msgs x ws') fun i => coinSeq (Claims.Recursion.fork j t.2.1 t.2.2.1) i.val).2,
      e ∈ (deployRec Bd.IVStar ivScope b).calls x ws ++ (deployRec Bd.IVStar ivScope b).calls x ws' := fun e he =>
    (List.mem_append.1 he).elim (fun h => List.mem_append_left _ (hlog₁ e h))
      (fun h => List.mem_append_right _ (hlog₂ e h))
  have hmem : ∀ y ∈ ((Bd.decAt H (xIOf x) (b.rd.msgs x ws) fun i => coinSeq t.2.1 i.val).2 ++
      (Bd.decAt H (xIOf x) (b.rd.msgs x ws') fun i =>
        coinSeq (Claims.Recursion.fork j t.2.1 t.2.2.1) i.val).2).map Prod.fst,
      y ∈ ((deployRec Bd.IVStar ivScope b).calls x ws ++ (deployRec Bd.IVStar ivScope b).calls x ws').map
        Prod.fst := fun y hy => by
    obtain ⟨e, he, rfl⟩ := List.mem_map.1 hy
    exact List.mem_map.2 ⟨e, hsub e he, rfl⟩
  exact hlift _ (fun p hp => hp) (Claims.Recursion.findColl_of_coll H htrue (hmem q hq) (hmem q' hq') hc)

/-- `forksBindAt_C_pick` with the readers' rounds before the last the runs' own (`pk x j m = m`). -/
theorem forksBindAt_C {Col' T' : Type} [DecidableEq Col] (H : Msg → Msg) (Bd : Bound Coin Col T)
    (decI : List (ℕ × Col) → List T) (ivScope : Msg → Bool) (b : Brings) (DO : Claims.Flock.Deploy Coin Col' T')
    (hEC : Bd.LastBindsEC H) (hCF : Bd.LastCoinFree H)
    (hT : Claims.Recursion.CallsTrue H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO)
    (hL : Claims.Recursion.CallsLogColl H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO)
    (hP : ∀ (decO : List (ℕ × Col') → List T') (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω)
      (r : Fin ((deployRec Bd.IVStar ivScope b).V.k x) → Coin)
      (c : (Claims.Recursion.EO (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO).C x)
      (ws : List Claims.Flock.Wit),
      Claims.Recursion.EOGives H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO P x ω r c
        ws →
      (∀ j, j + 1 < Bd.DE.V.k (xIOf x) → (b.rd.msgs x ws).getD j [] =
          ((run H P (deployRec Bd.IVStar ivScope b).V x ω r).msgs.map Prod.fst).getD j []) ∨
        ∃ u v, findColl ((deployRec Bd.IVStar ivScope b).calls x ws) = some (u, v) ∧ Collision H u v)
    (x : Msg)
    (hA : Claims.Recursion.InnerAcceptsCollAt H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO
      x) :
    Claims.Recursion.ForksBindAt H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO x :=
  forksBindAt_C_pick H Bd decI ivScope b DO (fun _ _ m => m) hEC hCF hT hL x (fun decO => hP decO x) hA

theorem forksBind_C {Col' T' : Type} [DecidableEq Col] (H : Msg → Msg) (Bd : Bound Coin Col T)
    (decI : List (ℕ × Col) → List T) (ivScope : Msg → Bool) (b : Brings) (DO : Claims.Flock.Deploy Coin Col' T')
    (hEC : Bd.LastBindsEC H) (hCF : Bd.LastCoinFree H)
    (hT : Claims.Recursion.CallsTrue H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO)
    (hL : Claims.Recursion.CallsLogColl H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO)
    (hA : Claims.Recursion.InnerAcceptsColl H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO)
    (hP : ∀ (decO : List (ℕ × Col') → List T') (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω)
      (r : Fin ((deployRec Bd.IVStar ivScope b).V.k x) → Coin)
      (c : (Claims.Recursion.EO (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO).C x)
      (ws : List Claims.Flock.Wit),
      Claims.Recursion.EOGives H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO P x ω r c
        ws →
      (∀ j, j + 1 < Bd.DE.V.k (xIOf x) → (b.rd.msgs x ws).getD j [] =
          ((run H P (deployRec Bd.IVStar ivScope b).V x ω r).msgs.map Prod.fst).getD j []) ∨
        ∃ u v, findColl ((deployRec Bd.IVStar ivScope b).calls x ws) = some (u, v) ∧ Collision H u v) :
    Claims.Recursion.ForksBind H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO :=
  fun decO x => forksBindAt_C H Bd decI ivScope b DO hEC hCF hT hL hP x (hA.toAt x) decO

end Cols

/-! ## The route -/

/-- **`KSBoundRoute` with the last message bound modulo what `E*` reads**: its premises with (P4, `Bound.LastBinds`)
replaced by `Bound.LastBindsEC` (a collision where two accepted decisions differ in `DE`'s columns or in the set of
`IVStar`'s calls) and `Bound.LastCoinFree` (`IVStar`'s last step reads no last coin, which holds at the deployed plain
verifier's own steps, `lastCoinFree_deployed`). -/
def KSBoundRouteCols (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) : Prop :=
  ∀ (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ),
    let Bd := bdDeployed rel nextE colsE wE join rowIn final w
    let D := recBound rel b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    Bd.PlainKSAbort FlockDeploy.sha512 → Bd.Binds → Bd.WeightCovers → Bd.LastBindsEC FlockDeploy.sha512 →
    Bd.LastCoinFree FlockDeploy.sha512 →
    Claims.Flock.KS FlockDeploy.sha512 (deployZkRec relO) →
    (∃ decC : Msg → Msg → Coin,
      (∀ decI, Claims.Recursion.VStar.RunPairs FlockDeploy.sha512 D
        (Claims.Recursion.Inner.ofFlock Bd.deploy decI) decC) ∧
      ∀ shape ∈ [Claims.Recursion.Form.Dense, Claims.Recursion.Form.Holo, Claims.Recursion.Form.Sparse],
        ∃ (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
          (cmt : Msg → ℕ → Msg → Msg),
          (∀ decI, Claims.Recursion.VStarB.HeldB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape) ∧
          (∀ decI, Claims.Recursion.VStarB.VstarShowsB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape
            decC shows cmt) ∧
          (∀ s, Claims.Recursion.VStarB.PartChecksB FlockDeploy.sha512 D (deployZkRec relO) L shape shows cmt chk s) ∧
          (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec relO) L shape shows cmt s) ∧
          Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
    ReadersLogColl FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
    ReadersRunPlayed FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
    KSBoundDeployed rel relO b L nextE colsE wE join rowIn final w

/-- **`KSBoundRouteCols` from the bridge**: `ksBoundRoute_of_bridge` with `KS_of_flockKS_binds` at `deployRecC`, whose
inner view reads the last message as its representative modulo what `E*` reads, carried back (`ksA_of_C`).
`ForksBind` is `forksBind_C`, from `LastBindsEC` and `LastCoinFree`; `CallsLogColl` and `InnerAcceptsColl` are the
deployed record's (`callsLogColl_C`, `innerAcceptsColl_C`); every other fact of the compiled record is `deployRec`'s,
which `deployRecC` shares. -/
theorem ksBoundRoute_of_bridge_cols (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (hRows : L.ReadsRows FlockDeploy.sha512)
    (hB : ∀ (nextE : Msg → List Msg → List Coin → Claims.OC Out)
      (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
      (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ),
      Claims.Recursion.Bridge FlockDeploy.sha512 (bdDeployed rel nextE colsE wE join rowIn final w)) :
    KSBoundRouteCols rel relO b L := by
  intro nextE colsE wE join rowIn final w Bd D bL hP1 hBinds hWC hEC hCF hF hst hT hL hRRP decI hdec
  obtain ⟨decC, hRP, hshapes⟩ := hst
  let I := Claims.Recursion.Inner.ofFlock Bd.deploy decI
  let ivScope := (FlockDeploy.deployPlain rel).scopeCheck
  have hplay : ∀ x ms (c : Fin (I.V.k (xIOf x)) → Coin),
      (run FlockDeploy.sha512 (replay ms) I.V (xIOf x) () c).msgs.length = I.V.k (xIOf x) :=
    fun x ms c => Bd.run_ivstar_length FlockDeploy.sha512 (xIOf x) ms c
  have hacc : ∀ shape ∈ [Claims.Recursion.Form.Dense, Claims.Recursion.Form.Holo, Claims.Recursion.Form.Sparse],
      Claims.Recursion.VStarAccepts FlockDeploy.sha512 (deployRec I.V ivScope bL) I (deployZkRec relO) shape := by
    intro shape hs
    obtain ⟨chk, shows, cmt, hHe, hSh, hCk, hOp, hDc⟩ := hshapes shape hs
    exact vstarAccepts_bound FlockDeploy.sha512 I ivScope b L (deployZkRec relO) shape decC chk shows cmt hplay
      (hRP decI) (hHe decI) (hSh decI) hCk hOp hDc hRows hRRP
  have hA := innerAcceptsColl_bound FlockDeploy.sha512 I ivScope b L (deployZkRec relO) hT
    (hacc _ (by simp)) (hacc _ (by simp)) (hacc _ (by simp))
  have hTc := callsTrue FlockDeploy.sha512 I ivScope bL (deployZkRec relO) hT
  have hLc := callsLogColl FlockDeploy.sha512 I ivScope bL (deployZkRec relO) hT hL
  have hFk := forksBind_C FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) hEC hCF hTc hLc hA
    fun decO x Ω P ω r c ws hg => readersRun_played FlockDeploy.sha512 I ivScope bL (deployZkRec relO) hplay hT hRRP hg
  have hKS : I.KSAbort FlockDeploy.sha512 := hB nextE colsE wE join rowIn final w decI hdec hBinds hWC (hP1 decI hdec)
  have hdecC : ∀ x c, FlockDeploy.decCoin x (Bd.ans c) = c := fun _ c =>
    (congrArg (FlockDeploy.decCoin _) (bdDeployed_ans rel nextE colsE wE join rowIn final w c)).trans
      (Function.leftInverse_invFun Claims.Bridge.coin_bytes_injective c)
  have hpc := Bd.publicCoin FlockDeploy.sha512 decI FlockDeploy.decCoin hdecC
  have hM := roundsMargin_deployRec FlockDeploy.sha512 I ivScope bL fun x => V_k_pos_flock false x
  have hLn := callsLogColl_C FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) hCF hLc
  have hAn := innerAcceptsColl_C FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) hCF hA
  have hKSn := Claims.Recursion.KS_of_flockKS_binds FlockDeploy.sha512 (deployRecC Bd ivScope bL) I
    (deployZkRec relO) FlockDeploy.decCoin hF hKS (refuses I.V ivScope bL FlockDeploy.sha512) (innerRounds I ivScope bL)
    (sessionNums I.V ivScope bL relO) (tableErrNonneg I.V ivScope bL) (sessionRounds I ivScope bL relO)
    (sessionAccepts I.V ivScope bL) (sessionRuns_sha512 I ivScope bL relO)
    (innerViewRun_C FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO)) hTc hLn hFk hAn hM hpc
    (callsNil I.V ivScope bL FlockDeploy.sha512)
  exact ksA_of_C FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) hCF hKSn

end RecDeploy
