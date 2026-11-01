import Proofs.Flock.Claims.Recursion.ResLowerDA
import Proofs.Flock.Claims.Recursion.ResSite
import Proofs.Flock.Claims.Recursion.LabelsFull

/-!
# `ResSite` from the verifier's checks, at part `s`'s decision

`ResSiteDec` is the site facts `resTie_resVRead` takes, at a piece's context and part `s`'s own decision (`PartDec`:
its staged statement's session decided, its checks and last refusals passed), and `resSite_of_dec` proves it at every
brings `b`:

* the piece's key is `(p, r, 0, 0)` with `r < 2` (`mem_resPD'`), so its instance `r` is one of the two the staged part's
  table holds, in a block (`resSite_staged`), at unit `0 < upv = 1`;
* part `s`'s circuit (`staged_circuit`) is table 0's, `v` and `out` where V\*'s rows put them, `v` an input row and `out`
  an output row, of `portWordsOf nv` and 128 words (`resSite_staged`, `checkPart_ns`);
* check 5's rows are two of `portWordsOf nv` words (`resRows_dense`), so as wide as `v`'s port;
* table 0 labels its instances `0..n-1` (`idx_of_dec`): setup checks the labels against the statement's draw
  (`checkLabels`), and the decision's checks and last refusals make that draw every unit (`labels_full`). Setup alone
  doesn't: without a draw `checkLabels` passes, and only the decision refuses that (`drawLaw`).

The decision is part `s`'s, at its staged statement `xO`, not the inner statement's (`Rows.DecAt`, which reads the inner
session `xI x` and says nothing of `xO`'s draw). An accepted run gives it at each part it stages (`partDec_run`).
-/

namespace RecDeploy.ResLower

open Flock
open Claims (Msg Out Prover run coinSeq)
open FlockDeploy (Coin FinalP)
open Claims.Recursion.VStarB

/-- **At part `s`'s decision, table 0 labels its instances `0..n-1`** (`labels_full`, at the staged statement's inner
shape, `insO_staged`). -/
theorem idx_of_dec {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {i : RecIns} {spec : SessionSpec}
    {sts : Array Stmt} (hsu : setupO xO = some (i, spec, sts)) (hD : PartDec xO i) (h0 : 0 < sts.size) :
    Registered.unitIndices sts[0].pub = .ok (Array.range sts[0].pub.n) := by
  obtain ⟨ms, -, -, -, rfl, -, -⟩ := hS
  obtain ⟨hins, -⟩ := setupO_some hsu
  obtain ⟨_, -, hinner, -⟩ := insO_staged hins
  obtain ⟨_, _, _, _, _, hc, hf⟩ := hD
  exact labels_full hsu (by simp [hinner]) h0 hc hf

/-- **An accepted run decides every part it stages** (`run_decides`, `nextO_decide`, as `labels_run` reads them). -/
theorem partDec_run (H : Msg → Msg) (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) {Ω : Type} (P : Prover Ω)
    (ω : Ω) (r : Fin ((RecDeploy.V IV st).k x) → Coin) (hacc : (run H P (RecDeploy.V IV st) x ω r).out = .acc)
    {s : ℕ} (hs : s < nS x) {i : RecIns} {spec : SessionSpec} {sts : Array Stmt}
    (hsu : setupO (staged st x ((run H P (RecDeploy.V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s) =
      some (i, spec, sts)) :
    PartDec (staged st x ((run H P (RecDeploy.V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s) i := by
  obtain ⟨j, f, hj, ⟨j', xO, as', cs', ins, spec₀, sts₀, link, c₁, mids, hr, hO⟩, hn⟩ :=
    run_decides H IV st x P ω r hacc hs
  rw [next_sess H hr] at hn
  have hacc' : ((nextO xO as' cs').eval H).1 = .acc := by
    rcases nextO_decide_out H hO with ho | ho
    · exact ho
    · rw [ho] at hn
      exact absurd rfl hn
  obtain ⟨⟨spec', sts', -, -, hc⟩, hf⟩ := (nextO_decide H hO).1 hacc'
  obtain ⟨i', hi', hib⟩ := actO_decide hO
  obtain ⟨hk, -, -⟩ := ract_sess_locate hr
  obtain ⟨-, -, hxO⟩ := ract_sess hr
  have hK := run_acc_length H IV st x P ω r hacc
  have hlen : (((run H P (RecDeploy.V IV st) x ω r).msgs.take (j + 1)).map Prod.fst).length = j + 1 := by
    simp only [List.length_map, List.length_take]
    omega
  rw [hlen] at hk
  have hip := innerPairs_run H IV st x P ω r hacc (off := j + 1) hk.le (by omega) []
    (range_map_take (coinSeq r) hk.le)
  rw [List.append_nil] at hip
  rw [hip] at hxO
  subst hxO
  have e := hi'.symm.trans hsu
  simp only [Option.some.injEq, Prod.mk.injEq] at e
  obtain ⟨rfl, rfl, rfl⟩ := e
  rw [← hib] at hc
  exact ⟨_, _, _, _, _, hc, hf⟩

/-- **`ResSite` at part `s`'s decision**: `ResSite`'s context, with `PartDec xO i` after the staged statement's setup. -/
def ResSiteDec (rd : Msg → Option FinalP) (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) :
    Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (DRL rd wE final w b elem).parse x = some st →
    Claims.Recursion.Form.Dense st.form → ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit),
    RecDeploy.OpenLower.JointP FlockDeploy.sha512 (DRL rd wE final w b elem) (DeployedR.layR elem)
      (AccP rd wE final w b elem Claims.Recursion.Form.Dense recRead (cmtTr recRead)) x c ws →
    ∀ s (wt : Claims.Flock.Wit), ws[s]? = some wt →
    ∀ (r : XR), readX x = some r → ∀ (ss : Array VStar.Stmt), RecVStar.statements r.shape = .ok ss →
    ∀ (j : ℕ) (nm : String) (pt : VStar.Part), ss[s]? = some (.inner j nm pt) →
    ∀ (xO : Msg) (p : _root_.Flock.Readable.Parsed),
      _root_.Flock.Readable.parseBytes readOfO xO = some p →
      Claims.Flock.check (_root_.Flock.Readable.parseOf _root_.Flock.Readable.relation readOfO) xO wt = true →
      showsPart recRead b s x xO c →
      (DeployedR.layR elem).Agrees (cmtTr recRead) x s wt xO ((DeployedR.layR elem).msgs x ws) →
      (∀ e ∈ (DeployedR.layR elem).calls x s wt, FlockDeploy.sha512 e.1 = e.2) →
      (Claims.findColl ((DeployedR.layR elem).calls x s wt ++
        vcallsAt FlockDeploy.sha512 (DRL rd wE final w b elem) xO)).isSome = false →
    ∀ q ∈ DecompBR.resPD (Claims.Recursion.xI (DRL rd wE final w b elem) x), DecompBR.ownRD x q = some s →
    ∀ (i : RecIns) (spec : SessionSpec), setupO xO = some (i, spec, p.sts) → PartDec xO i →
    ∀ (ht : 0 < p.sts.size),
      (0 < p.sts[0].c.g ∧ (q.key.2.1 + q.key.2.2.1) / p.sts[0].c.g < 2 ^ (p.sts[0].m - p.sts[0].c.kLog) ∧
        q.key.2.2.2 < p.sts[0].c.upv) ∧
      ∀ (m k cl jj : ℕ) (sh : _root_.Flock.Shape) (holo : Bool) (S : _root_.Flock.RecOpen.GfS) (ns : List ℕ)
        (nv : ℕ), RecResiduals.params p.sts[0].c.unit.name = some (m, k, cl, jj) →
        RecResiduals.shape? m k cl = some sh → RecResiduals.holoOf p.sts[0].c = .ok holo →
        RecResiduals.part { sh with holo } jj = .ok (S, ns, nv) →
        ∃ cC : Circuit, r.w.parts[s]?.bind partCircuit = some cC ∧
          Registered.unitIndices p.sts[0].pub = .ok (Array.range p.sts[0].pub.n) ∧
          q.key.2.1 + q.key.2.2.1 < p.sts[0].pub.n ∧
          ∃ (qv qo : ℕ) (ptv pto : Port), cC.ports.findIdx? (·.name == "v") = some qv ∧
            cC.ports.findIdx? (·.name == "out") = some qo ∧ qv < HmOut.nIn cC ∧ ¬ qo < HmOut.nIn cC ∧
            cC.ports[qv]? = some ptv ∧ cC.ports[qo]? = some pto ∧
            (∀ (f : RecCoins.Source), RecCoins.Source.atC3 i.src i.inner i.commits = some f →
              ∀ (inn : RecCoins.Inner) (rows : Array (Array ℕ)), f.load p.sts[0].c.unit.name = .ok inn →
              FlockSoundness.Discharge.Recursion.residualRows inn jj = some rows →
              ptv.words = (RecCoins.padTo rows[q.key.2.1 + q.key.2.2.1]!
                (Registered.rowWords rows[q.key.2.1 + q.key.2.2.1]!.size 16)).size ∧
                q.key.2.1 + q.key.2.2.1 < rows.size ∧ 8 * nv ≤ rows[q.key.2.1 + q.key.2.2.1]!.size) ∧
            pto.words = (RecCoins.padTo (RecCoins.zeroRows 2 RecCoins.OUT_WORDS)[q.key.2.1 + q.key.2.2.1]!
              (Registered.rowWords (RecCoins.zeroRows 2 RecCoins.OUT_WORDS)[q.key.2.1 + q.key.2.2.1]!.size
                16)).size

/-- **`ResSite` at part `s`'s decision, at every brings.** -/
theorem resSite_of_dec (rd : Msg → Option FinalP) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (b : Brings)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) :
    ResSiteDec rd wE final w b elem := by
  intro x st hx hf c ws hJ s wt hw r hr ss hss j nm pt hsv xO p hxO hck hsh hAg hT hcl q hq ho i spec hsu hD ht
  have hck' : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO wt = true := hck
  have hidx := idx_of_dec hsh.2 hsu hD ht
  obtain ⟨cC, hpc⟩ := staged_circuit hsh.2 hr
  obtain ⟨h0', hc0, hcp, hpn, hg0, hupv, hblk, hqv, hqo, hvin, hoout, ptv, pto, hptv, hpto, hwv, hpw⟩ :=
    resSite_staged hsh.2 hck' hr hss hsv hpc hxO
  subst hc0
  obtain ⟨_v, -, p0, r0, S0, ns0, nv0, hr0, -, hk, -⟩ := DecompBR.mem_resPD' hq
  have hn : q.key.2.1 + q.key.2.2.1 < 2 := by simp only [hk]; omega
  have hu : q.key.2.2.2 = 0 := by simp only [hk]
  have hpn' : 2 ≤ p.sts[0].pub.n := hpn
  refine ⟨⟨hg0, hblk _ hn, by rw [hu, hupv]; exact Nat.one_pos⟩, ?_⟩
  intro m k cl jj sh holo S ns nv hpar hshp hholo hpart
  obtain ⟨-, hnv⟩ := checkPart_ns hcp hpar hshp hholo hpart
  refine ⟨_, hpc, hidx, lt_of_lt_of_le hn hpn', _, _, ptv, pto, hqv, hqo, hvin, hoout, hptv, hpto, ?_, ?_⟩
  · intro f hf' inn rows hload hrows
    obtain ⟨hsz, hrw⟩ := resRows_dense hx hf hsh hsu ht hpar hshp hholo hpart hf' hload hrows
    have h1 := hrw _ hn
    have e2 : ptv.words = (8 * nv + 63) / 64 * 64 := by rw [hwv, hnv]; rfl
    rw [rs_padTo_size, h1]
    unfold Registered.rowWords RecResiduals.portWordsOf
    refine ⟨by omega, by omega, by omega⟩
  · rw [rs_padTo_size, rs_zeroRow_size hn, hpw]
    all_goals (unfold Registered.rowWords; omega)

end RecDeploy.ResLower
