import Proofs.Flock.Claims.KS.PlainOffLive

/-!
# A plain slot's off mass at the deployed coins, at read sets

`plainOffU_le` bounds a plain slot's term by the off mass of a single-valued deployed opening `opD`, which must equal
every decoded live opening. One deployed run can list a position twice in a level's columns with different rows, and
the model's opening is the first verified entry, so no single-valued `opD` is that in general. Here the deployed side
reads a set of values per position (`rdD`), which only has to contain every decoded live opening, and the off event
is "the read set is nonempty and isn't the singleton `{w q}`". That is `Pr_rdOff_le_capHit`'s off event
(`Claims/Recursion/CapLinkRd.lean`).
-/

namespace FlockDeploy.KS

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.Live
open Flock FlockSoundness.Discharge
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

open Classical in
/-- **The slot's term averaged over the tape is at most the chance over the deployed coins of reading anything but a
table that reads their first `j`**, at deployed read sets `rdD` that hold every decoded live opening: `plainOffU_le`
with a read set in place of a single opening. -/
theorem plainOffU_le_rd {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : PlainHDecU L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn) (hw : 2 * spec.mPts ≤ 65536) {Ω : Type}
    (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽) (T : Model.Oracle 𝔽) (r : Fin 2)
    (ℓ : Fin (l₀ :: l₁ :: rest).length) (j : ℕ) (hL : plainN x + 2 = (V false).k x)
    (hslot : ∀ u, PlainSlotQU L m k0 l₀ l₁ rest ext hsch hk0 hm hlay I d x opn hdec hw P ω C r ℓ j u)
    (rdD : (Fin ((V false).k x) → Coin) → Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
      Finset (Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽))
    (w : (Fin ((V false).k x) → Coin) → Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
      Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽)
    (hwj : ∀ c c' : Fin ((V false).k x) → Coin, (∀ i : Fin ((V false).k x), i.val < j → c i = c' i) → w c = w c')
    (hrel : ∀ (u : plainTape x) (rr : Fin (plainN x + 2) → LCoin 65536) a,
      Claims.Bridge.RetOK (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leafPlain cap pend os)
          (plainVerdictAt I d C (plainRoot P x ω)) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
          (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
          (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
          (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
            (plainSim P x spec st) x (ω, u) rr) a →
        ∀ q v, opCapL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r.val ℓ.val a q = some v →
          v ∈ rdD (plainΦ x spec st (u, rr)) q) :
    (∑ u, plainOffAtU L m k0 l₀ l₁ rest ext hsch hk0 hm hlay I d x opn hdec hw P ω C T r ℓ u) /
        Fintype.card (plainTape x) ≤
      ∑ q, Claims.Pr fun c : Fin ((V false).k x) → Coin => (rdD c q).Nonempty ∧ rdD c q ≠ {w c q} := by
  have hR : (∑ q, Claims.Pr fun c : Fin ((V false).k x) → Coin => (rdD c q).Nonempty ∧ rdD c q ≠ {w c q}) =
      Claims.Ex fun p : plainTape x × (Fin (plainN x + 2) → LCoin 65536) =>
        ∑ q, ind (fun c => (rdD c q).Nonempty ∧ rdD c q ≠ {w c q}) (plainΦ x spec st p) := by
    rw [Claims.Bridge.Ex_sum_univ]
    refine Finset.sum_congr rfl fun q _ => ?_
    rw [← Pr_plainΦ (spec := spec) (st := st) hL, Claims.Bridge.Pr_eq_Ex]
    rfl
  rw [hR, Claims.Bridge.Ex_prod]
  refine div_le_div_of_nonneg_right (Finset.sum_le_sum fun u _ => ?_) (Nat.cast_nonneg _)
  refine (plainOffAtU_le_live L m k0 l₀ l₁ rest ext hsch hk0 hm hlay I d x opn hdec hw P ω C T r ℓ j u (hslot u)
    (fun rr => w (plainΦ x spec st (u, rr)))
    (fun rr rr' h => hwj _ _ (plainΦ_prefix hL u h))).trans
    (Claims.Ex_mono fun rr => Finset.sum_le_sum fun q _ => Claims.Bridge.ind_le_ind_imp fun hoff => ?_)
  obtain ⟨hs, hne⟩ := hoff
  obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 hs
  obtain ⟨a, ha, hav⟩ := Claims.Bridge.liveOp_some hv
  have hd := hrel u rr a ha q v hav
  refine ⟨⟨v, hd⟩, fun he => hne ?_⟩
  rw [he, Finset.mem_singleton] at hd
  rw [hv]
  exact congrArg some hd

end FlockDeploy.KS
