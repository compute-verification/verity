import Proofs.Flock.Claims.Recursion.ResLowerDA
import Proofs.Flock.Claims.Recursion.ResSite

/-!
# `ResSite` from the verifier's checks, given table 0's labels

`ResLower.ResSite` (ResLowerD), the site facts `resLowered_D` consumes, at every brings `b`:

* the piece's key is `(p, r, 0, 0)` with `r < 2` (`mem_resPD'`), so its instance `r` is one of the two the staged part's
  table holds, in a block (`resSite_staged`), at unit `0 < upv = 1`;
* part `s`'s circuit (`staged_circuit`) is table 0's, `v` and `out` where V\*'s rows put them, `v` an input row and `out`
  an output row, of `portWordsOf nv` and 128 words (`resSite_staged`, `checkPart_ns`);
* check 5's rows are two of `portWordsOf nv` words (`resRows_dense`), so as wide as `v`'s port.

Table 0's labels (`ResIdx`) are a hypothesis: setup checks them against the statement's draw only where it has one
(`RecVStar.checkLabels none = pure ()`), and only the decision refuses a statement without a draw (`drawLaw`) or short of
its population (`fullDrawO`), which `labels_full` reads. `ResSite`'s context holds no decision.
-/

namespace RecDeploy.ResLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP)
open Claims.Recursion.VStarB

/-- **Table 0 labels its instances `0..n-1`** at each staged part of a dense statement (`hidx`, which `labels_full` gives
at a decision). -/
def ResIdx (b : Brings) : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.Stmt), RecDeploy.parse x = some st → Claims.Recursion.Form.Dense st.form →
    ∀ (s : ℕ) (xO : Msg) (c : ℕ → Coin), showsPart recRead b s x xO c →
    ∀ (i : RecIns) (spec : SessionSpec) (sts : Array Stmt), setupO xO = some (i, spec, sts) → ∀ h0 : 0 < sts.size,
      Registered.unitIndices sts[0].pub = .ok (Array.range sts[0].pub.n)

/-- **`ResSite` at every brings, given table 0's labels.** -/
theorem resSite_of_idx (rd : Msg → Option FinalP) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (b : Brings)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (hidx : ResIdx b) :
    ResSite rd wE final w b elem := by
  intro x st hx hf c ws hJ s wt hw r hr ss hss j nm pt hsv xO p hxO hck hsh hAg hT hcl q hq ho i spec hsu ht
  have hck' : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO wt = true := hck
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
  refine ⟨_, hpc, hidx x st hx hf s xO c hsh i spec p.sts hsu ht, lt_of_lt_of_le hn hpn', _, _, ptv, pto, hqv, hqo,
    hvin, hoout, hptv, hpto, ?_, ?_⟩
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
