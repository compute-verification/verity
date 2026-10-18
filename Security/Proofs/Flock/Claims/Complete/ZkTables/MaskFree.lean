import Proofs.Flock.Claims.ZK.MaskedTables
import Proofs.Flock.Claims.Complete.ZkTables.Ready

/-!
# The deployed relation does not read the mask slots

`maskFree_relation`: at the deployed relation (`Readable.relation`), a witness with every table's mask slot set to any
words (`maskWOf`) is in the relation wherever the witness is (`MaskFree`). At a statement of the masked scope each
table is a `MaskSetup` that avoids the mask (`Readable.tableAt_zk`, `avoidsMask_of_shapeOk`), so its mask slot is
isolated (`maskIsolated_of_layout`): its rows hold at any bits, and no other row, the pin or a region reads it. Table
`c`'s share of the mask witness is `Composed.zWith` at the share of `w` and the words' bits (`zOf_maskW`), which
satisfies the table's lowered statement where `w`'s share does (`satisfies_zWith`).
-/

namespace FlockComplete.ZkTables

open Flock FlockDeploy FlockSoundness FlockSoundness.Discharge

/-- **Table `c`'s share of the mask witness is `zWith`'s**: the share of `w` with the mask slot's word `k` set to
`μ c k`'s bits. -/
theorem zOf_maskW (sts : Array Flock.Stmt) (w : Claims.Flock.Wit) (μ : ℕ → ℕ → F128) {c : ℕ} (hc : c < sts.size) :
    Readable.zOf sts.size (maskW sts w μ) c sts[c].m =
      Composed.zWith sts[c] (Readable.zOf sts.size w c sts[c].m)
        (fun (k : Fin (Composed.maskWordsOf sts[c]).length) v => Readable.toZ ((μ c k.val).bit v.val)) := by
  funext a
  unfold Composed.zWith
  split
  · rename_i k hk
    simp only [Readable.zOf]
    rw [maskW_table sts w μ hc a.isLt hk]
  · rename_i hk
    simp only [Readable.zOf]
    rw [maskW_table_off sts w μ hc hk]

/-- **The deployed relation does not read the mask slots.** -/
theorem maskFree_relation : MaskFree Readable.relation := by
  intro x w μ hsc hR
  unfold maskWOf
  split
  · rename_i s spec sts hset
    obtain ⟨p, hp, hz, ht, hpi, hpe, hparse, hRp, cc, hc, ho⟩ := Readable.parsed_zk hsc hR hset
    subst hpe
    refine ⟨Readable.toReadable Readable.relation p, hparse, ?_⟩
    rw [Readable.relS_relation] at hRp ⊢
    intro j hj
    have hs := Readable.tablesOf_of_parses hp
    obtain ⟨hwf, hrw, hmat⟩ := Readable.tableAt_model hs ht hpi hj
    obtain ⟨hzk, M, -⟩ := Readable.tableAt_zk hs hz ht hpi hc ho hj
    have hI := Composed.maskIsolated_of_layout p.sts[j] hwf
      (Composed.maskLayout_of M (FailClosed.avoidsMask_of_shapeOk hrw hzk))
    have e : p.sts[j]! = p.sts[j] := getElem!_pos p.sts j hj
    obtain ⟨hf, hsat⟩ := hRp j hj
    revert hf hsat
    rw [e]
    rintro hf ⟨h1, h2, h3⟩
    have hA : ((Readable.Tab.ofStmt p.sts[j]).toModel hf.2.2).A₀ =
        (Refine.stmtOf p.sts[j] hwf (Refine.regionsOf p.sts[j])).A₀ :=
      funext fun a => funext fun b => (hmat a b).1
    have hB : ((Readable.Tab.ofStmt p.sts[j]).toModel hf.2.2).B₀ =
        (Refine.stmtOf p.sts[j] hwf (Refine.regionsOf p.sts[j])).B₀ :=
      funext fun a => funext fun b => (hmat a b).2
    have hsat : (Refine.stmtOf p.sts[j] hwf (Refine.regionsOf p.sts[j])).Satisfies
        (Readable.zOf p.sts.size w j p.sts[j].m) := by
      refine ⟨fun o i => ?_, h2, h3⟩
      have := h1 o i
      rw [hA, hB] at this
      exact this
    have hz' := Composed.satisfies_zWith p.sts[j] hwf (Refine.regionsOf p.sts[j]) M.m M.mask hI _ hsat
      (fun (k : Fin (Composed.maskWordsOf p.sts[j]).length) v => Readable.toZ ((μ j k.val).bit v.val)) rfl
    refine ⟨hf, ?_⟩
    rw [zOf_maskW p.sts w μ hj]
    obtain ⟨g1, g2, g3⟩ := hz'
    refine ⟨fun o i => ?_, g2, g3⟩
    have := g1 o i
    rw [← hA, ← hB] at this
    exact this
  · exact hR

end FlockComplete.ZkTables
