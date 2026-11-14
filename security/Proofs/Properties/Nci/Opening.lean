module

public import Properties.Nci.Opening
public import Proofs.Accounting.ReadUnit

/-!
# A pinned reader's unit returns its table's row

`NciCheck.SecurityProofs.PinnedReadBinds` proves `NciCheck.Guarantees.PinnedReadBinds`. The tie moves what the unit's
key, root, domain and verdict ports hold onto the ports' gates (`holds_eq`: two lists of one length that a unit holds at
the same leaves are one list), which gives the read's registered rows in the transcript. Inside the table the row
returned is the table's by `Accounting.SecurityProofs.readUnitBinds`; at or past the count it is the fill row by the
gates' refinement alone.
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Flock.Program (Prog Scope)
open Flock.TemplateQuery (TRange)

/-- Two lists of one length that a unit holds at the same leaves are one list. -/
theorem holds_eq {P : Prog} {rs : Array TRange} {val : ℕ → ℕ → Bool} {tid : String} {un : ℕ}
    {Leaf : Scope → ℕ → ℕ → ℕ → ℕ → Prop} {bs bs' : List Bool} (h : Holds P rs val tid un Leaf bs)
    (h' : Holds P rs val tid un Leaf bs') (hl : bs'.length = bs.length) : bs' = bs := by
  refine List.ext_getElem hl fun b hb' hb => ?_
  obtain ⟨⟨sc, l, g, lo, hi, hu, hleaf, hlo, hhi⟩, hall⟩ := h b hb
  rw [hall sc l g lo hi hu hleaf hlo hhi, (h' b hb').2 sc l g lo hi hu hleaf hlo hhi]

end NciCheck.Properties.Record.SecurityProofs

namespace NciCheck.SecurityProofs

open NciCheck.Properties.Record NciCheck.Properties.Record.SecurityProofs
open Accounting.ReadUnit

/-- **A pinned reader's unit returns its table's row** (`NciCheck.Guarantees.PinnedReadBinds`). -/
theorem PinnedReadBinds : NciCheck.Guarantees.PinnedReadBinds := by
  intro F c j tr P rs r name p pin k dfn ro tags bytes tables ct hB n consts u q val Yc table kb Rb db
    _ _ _ _ _ _ _ hgates htie hcorrect hconsts hkb hkey hRb hroot hdb hdomain hverdict htable
  have hk : kb = bits ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u) q.tok :=
    holds_eq htie.key hkey (by rw [hkb]; simp [bits])
  have hR : Rb = bits Yc q.root := holds_eq htie.root hroot (by rw [hRb]; simp [bits])
  have hd : db = bits Yc q.domain := holds_eq htie.domain hdomain (by rw [hdb]; simp [bits])
  have hv : [true] = [bit Yc q.ok] := holds_eq htie.verdict hverdict rfl
  subst hk hR hd
  have hrows : RegisteredRows consts (Flock.Bridge.partsOf (.tmpl ct) hB n) u q (digestOf (bits Yc q.root))
      (digestOf (bits Yc q.domain)) Yc :=
    ⟨rfl, rfl, (List.cons.inj hv).1.symm, hconsts⟩
  have hb := Accounting.SecurityProofs.readUnitBinds Verity.sha512 Verity.sha512
    Flock.Hm96.Default512.key.data.toList consts (Flock.Bridge.partsOf (.tmpl ct) hB n) u q _ _ Yc hgates hcorrect
    hrows
  obtain ⟨hh, -⟩ := hgates.computes Yc hcorrect hconsts
  by_cases hlt : q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u) < q.count
  · rcases hb (table _) (htable _ hlt) with h | h | h
    · left
      rw [h]
      unfold Hash.EmbeddingRead.h
      exact (if_pos hlt).symm
    · exact .inr (.inl h)
    · exact .inr (.inr h)
  · left
    rw [hh]
    unfold Hash.EmbeddingRead.h
    exact (if_neg hlt).trans (if_neg hlt).symm

end NciCheck.SecurityProofs
