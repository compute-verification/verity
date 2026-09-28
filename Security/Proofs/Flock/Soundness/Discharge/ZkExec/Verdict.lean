import Proofs.Flock.Soundness.Discharge.ZkExec.VerdictStatements
import Proofs.Flock.Soundness.Discharge.ZkExec.PlaysDet
import Proofs.Flock.Soundness.Discharge.ZkExec.Verify
import Proofs.Flock.Soundness.Discharge.ZkExec.PlaysRep
import Proofs.Flock.Soundness.Discharge.ZkExec.RunsSizes
import Proofs.Flock.Soundness.Discharge.ZkExec.VerdictLig
import Proofs.Flock.Soundness.Discharge.ZkExec.VerdictOpens
import Proofs.Flock.Soundness.Discharge.ZkExec.VerdictRows

/-!
# `RunsZC`'s verdict half: the compiled `--zk` table accepts at the two reps' results

* `zk_table_verdict_of`: `ZkTableVerdict` from its per-rep half `ZkRepAccepts`, through `Zk.verifyRep`'s acceptance
  (`zk_verifyRep_ok`): both reps share the statement's extra claims, their proofs decode at one shape, their caps are
  the one `root_B` (`capOfRoot_eq`), `opnZ` reads each rep's rows, salts and inner openings off its own proof, and the
  results exist (`zk_rep_plays`).
* `zk_rep_accepts`: `ZkRepAccepts`'s three conclusions, each its own lemma: the Ligerito verdict (`zk_lig_accepts`,
  `VerdictLig.lean`), the inner check (`zk_inner_of_rep`, `VerdictRows.lean`, through `zk_inner_accepts`) and the
  requested rows (`zk_opens_ok`, `VerdictOpens.lean`).
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL

local notation "𝔽" => FlockLevel3.GF128

theorem zk_rep_accepts : ZkRepAccepts := by
  intro mPts T pts r st sch tPad extras bytes zp rounds dom tr cap₀ os hsch hms hseam hlay hfold hlen hext hdec hwf
    hrun hk hcap ho hs hin pend hp
  exact ⟨zk_lig_accepts T pts r st sch tPad extras bytes zp rounds dom tr cap₀ os hsch hms hseam hlay hfold hlen hext
      hdec hwf hrun hk hcap ho hs hin pend hp,
    zk_inner_of_rep T pts r st sch tPad extras bytes zp rounds dom tr cap₀ os hsch hms hseam hlay hfold hlen hext hdec
      hwf hrun hk hcap ho hs hin pend hp,
    zk_opens_ok T pts r st sch tPad extras zp rounds dom tr cap₀ os hsch hms hseam hlen hext hwf hrun hk hcap ho hs
      pend hp⟩

section Opn

variable {st : Setup} {sch : Flock.Schedule} {n : ℕ} {b₀ b₁ : ByteArray} {zp₀ zp₁ : Zk.ZkProof}

theorem opnZ_pair (h₀ : Zk.decode (shapeZKn st sch n) b₀ = .ok zp₀) (h₁ : Zk.decode (shapeZKn st sch n) b₁ = .ok zp₁) :
    (opnZ st sch n #[b₀, b₁]).rows = opensOf zp₀.proof zp₁.proof sch ∧
      (opnZ st sch n #[b₀, b₁]).salts = saltsOf zp₀.proof zp₁.proof sch ∧
      (opnZ st sch n #[b₀, b₁]).inner 0 = innerOpensOf zp₀.inner ∧
      (opnZ st sch n #[b₀, b₁]).inner 1 = innerOpensOf zp₁.inner := by
  have e₀ : (#[b₀, b₁] : Array ByteArray)[0]! = b₀ := rfl
  have e₁ : (#[b₀, b₁] : Array ByteArray)[1]! = b₁ := rfl
  simp only [opnZ, e₀, e₁, h₀, h₁]
  exact ⟨rfl, rfl, rfl, rfl⟩

end Opn

open Classical in
/-- **The compiled `--zk` table accepts** when both reps' results accept, at one level-0 cap. -/
theorem accZ_of {n mPts : ℕ} {Reg : Type} {T : Finset (Fin n) → Reg → ZkSession.TabZK 𝔽 mPts} {S : Finset (Fin n)}
    {R : Reg} {cap₀ : Cap Dig} {pend : ZkLink.PendZ 𝔽 Dig (List UInt8) × ZkLink.PendZ 𝔽 Dig (List UInt8)}
    {os : ZkLink.OpensZ 𝔽 Dig (List UInt8)}
    (h₀ : pend.1.lig.verdict os.rows = true ∧ pend.1.inner (os.inner 0) ∧
      ZkLink.OpensOKZ H512 enc512 leaf512 (ZkLink.width₀ (T S R)) cap₀ 0 pend.1.lig os.rows os.salts)
    (h₁ : pend.2.lig.verdict os.rows = true ∧ pend.2.inner (os.inner 1) ∧
      ZkLink.OpensOKZ H512 enc512 leaf512 (ZkLink.width₀ (T S R)) cap₀ 1 pend.2.lig os.rows os.salts) :
    accZ T S R (outZ hs512 T S R (fun _ => cap₀) pend os) = true := by
  unfold accZ
  refine decide_eq_true fun ⟨j, hj⟩ => ?_
  obtain rfl : j = 0 := by simpa using hj
  show (ZkLink.outZC hs512 (T S R) cap₀ pend os).accepted = true
  simp only [ZkLink.outZC, h₀.1, h₁.1, Bool.true_and, decide_eq_true_eq]
  exact ⟨h₀.2.1, h₁.2.1, h₀.2.2, h₁.2.2⟩

/-- **`ZkTableVerdict` from its per-rep half.** -/
theorem zk_table_verdict_of (hrep : ZkRepAccepts) : ZkTableVerdict := by
  intro n mPts Reg T S R pts spec st sch tPad sess i b₀ b₁ c hz hms hseam hlay hfold hext hwf h0 h1
  obtain ⟨s₀, ex, zp₀, tr₀, hs₀, hx₀, hd₀, -, -, hrun₀, hk₀, hc₀, hb₀⟩ :=
    zk_verifyRep_ok spec st sch tPad sess i 0 b₀ c h0
  obtain ⟨s₁, ex₁, zp₁, tr₁, hs₁, hx₁, hd₁, -, -, hrun₁, hk₁, hc₁, hb₁⟩ :=
    zk_verifyRep_ok spec st sch tPad sess i 1 b₁ c h1
  obtain rfl : ex = ex₁ := Except.ok.inj (hx₀.symm.trans hx₁)
  obtain ⟨hsz, hrel⟩ := hext ex hx₀
  have hsh : shapeZK st sch ex = shapeZKn st sch (Model.extraClaims (T S R).S (T S R).ptLocal mPts pts).length := by
    unfold shapeZK shapeZKn; rw [hsz]
  have hd₀' := hd₀
  have hd₁' := hd₁
  rw [hsh] at hd₀' hd₁'
  have hwf₀ : RoundsWF s₀.rounds := hwf s₀ (Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hs₀))
  have hwf₁ : RoundsWF s₁.rounds := hwf s₁ (Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hs₁))
  have hcap : ∀ {zp : Zk.ZkProof} {b : ByteArray}, Zk.decode (shapeZK st sch ex) b = .ok zp →
      c = zp.proof.cap.foldl (· ++ ·) ByteArray.empty → capOfRoot sess.rootB[i]! = capDig zp.proof.cap := by
    intro zp b hd hc
    have hw := (zk_decode_sizes _ _ _ hd).1.capEach
    refine capOfRoot_eq (fun x hx => ?_) (hc.symm.trans hb₀)
    have := hw x (Array.mem_toList_iff.1 hx)
    rw [this]
    show st.merkle.digestLen = 64
    rw [hms]
    rfl
  obtain ⟨hro, hsa, hi₀, hi₁⟩ := opnZ_pair hd₀' hd₁'
  refine ⟨s₀, s₁, zp₀, zp₁, hs₀, hs₁, hd₀', hd₁', ?_, fun pend hp₀ hp₁ => ?_⟩
  · obtain ⟨p₀, hp₀, -⟩ := zk_rep_plays hs512 (T S R) pts 0 st sch tPad ex zp₀ s₀.rounds _ tr₀ hz hseam hsz hrel
      hwf₀ hrun₀ hk₀
    obtain ⟨p₁, hp₁, -⟩ := zk_rep_plays hs512 (T S R) pts 1 st sch tPad ex zp₁ s₁.rounds _ tr₁ hz hseam hsz hrel
      hwf₁ hrun₁ hk₁
    rw [hsz] at hp₀ hp₁
    exact ⟨(p₀, p₁), hp₀, hp₁⟩
  · rw [← hsz] at hp₀ hp₁
    apply accZ_of
    · exact hrep (T S R) pts 0 st sch tPad ex b₀ zp₀ s₀.rounds _ tr₀ _ _ hz hms hseam hlay hfold hsz hrel hd₀ hwf₀
        hrun₀ hk₀ (hcap hd₀ hc₀) (fun ℓ j => by rw [hro]; simp [opensOf])
        (fun ℓ j => by rw [hsa]; simp [saltsOf]) hi₀ pend.1 hp₀
    · exact hrep (T S R) pts 1 st sch tPad ex b₁ zp₁ s₁.rounds _ tr₁ _ _ hz hms hseam hlay hfold hsz hrel hd₁ hwf₁
        hrun₁ hk₁ (hcap hd₁ hc₁) (fun ℓ j => by rw [hro]; simp [opensOf])
        (fun ℓ j => by rw [hsa]; simp [saltsOf]) hi₁ pend.2 hp₁

/-- **The compiled `--zk` table accepts at the two reps' results.** -/
theorem zk_table_verdict : ZkTableVerdict := zk_table_verdict_of zk_rep_accepts

end FlockSoundness.Discharge.ZkExec
