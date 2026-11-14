module

public import Properties.Nci.Opening
public import Properties.Verity.Sha512
public import Flock.Hm96
public import Proofs.Accounting.ReadUnit

/-!
# A pinned reader's unit returns its table's row

`PinnedReadBinds` is a lemma, proved by `pinnedReadBinds`, with its statement beside its proof: no lock lists it and
nothing reads it. Let a check
hold a record's row as an opening under a pin, and let its reader's circuit file parse to a templated statement, be
laid out as the reader's template with the file's unit the record's (`laidOutAs`), and have the read unit's gates `q`
at the reader's roles (`portsOf`). Where the read's gates compute its reference at `q`
(`Accounting.ReadUnit.GatesRefine`), the unit is correct in a transcript, its verdict bit is 1 there, and a table is
committed under the root and the domain id its root and domain gates hold: the row the unit returns in its local
evaluation is that table's row at the key its key gates hold, or the fill row where the key is at or past the table's
count, or SHA-512 collides on what the unit's check and the table's opening hashed.

The record's names reach the circuit's gates by `Properties.Nci.Opening`'s map alone, three equations of code, and
`portsAtGates_spec` says which gates `q`'s are. Inside the table the row is the table's by
`Accounting.SecurityProofs.readUnitBinds`; at or past the count it is the fill row by the gates' refinement alone.

**What it does not say.** It is on the row returned, not on the opened ports: at a key at or past the count the
reference accepts any witness, and the opened ports hold what the prover likes. The output row returned is a name the
statement takes: no opening declares it, so no row of the record is named as the row returned. That the gates refine
the reference at `q` is a hypothesis, proved here for no template. It gives no file, record and transcript that meet
its hypotheses.
-/

@[expose] public section

namespace NciCheck.Properties.Record

open Lean (Json)
open Flock.Program (Prog Defn)
open NciCheck.RecordText (Row Held)
open Accounting

/-- **A pinned reader's unit returns its table's row.** Let a check read a record and hold its row `r` as an opening
under a pin, the reader's Definition having the roles `ro`. Let the circuit file `bytes` parse to the templated
statement `ct`, as the verifier reads it; let it be laid out as `r`'s template, its unit `u` being `r`'s unit; and let
`q` be the read unit's gates there at the roles and the output row `out`. Let the read's gates compute its reference
at `u` and `q` (`Accounting.ReadUnit.GatesRefine`, under SHA-512 and the served hm96 key). Let `u` be correct in the
transcript `Yc` and the registered constants hold there; let the verdict bit be 1; and let `table` be committed under
the root and the domain id the root and domain gates hold. Then the row `u` returns in its local evaluation is
`table`'s row at the key its key gates hold there, or the fill row where the key is at or past the table's count
(`Hash.EmbeddingRead.h`), or SHA-512 collides on what the unit's check and `table`'s opening at the key hashed. -/
def PinnedReadBinds : Prop :=
  ∀ (F : Form) (c : F.Case) (j tr : Json), F.read j tr = .ok c →
  ∀ (P : Prog) (r : Row) (name p pin : String) (k : ℕ) (dfn : Defn) (ro : Roles) (out : String) (tags : Flock.Tags)
    (bytes : ByteArray) (tables : List (String × ByteArray)) (ct : Flock.Circuit)
    (hB : FlockSoundness.Built (Flock.Desc.tmpl ct).unroll.unit) (n : ℕ)
    (consts : Fin (Flock.Bridge.circuitOf (.tmpl ct) hB n).N → Option Bool) (u : Fin n)
    (q : ReadUnit.Ports (Flock.Bridge.circuitOf (.tmpl ct) hB n))
    (Yc : (Flock.Bridge.circuitOf (.tmpl ct) hB n).Trace) (table : ℕ → Hash.RowLeaf.Opening),
    Flock.Program.decode (F.inst c).program.toUTF8 = .ok P →
    F.held c r = some (.opened name p pin) → P.index[r.tid]? = some k → P.defs[k]? = some dfn →
    F.layout dfn = some ro →
    Flock.Desc.parseStmt tags bytes tables = .ok (.tmpl ct) →
    laidOutAs ct P r.tid u.val r.unit = true →
    portsOf ct hB n dfn r.unit ro out q.count q.words q.position = some q →
    ReadUnit.GatesRefine Verity.sha512 Verity.sha512 Flock.Hm96.Default512.key.data.toList consts
      (Flock.Bridge.partsOf (.tmpl ct) hB n) u q →
    (Flock.Bridge.partsOf (.tmpl ct) hB n).Correct Yc u →
    (∀ g ∈ (Flock.Bridge.partsOf (.tmpl ct) hB n).io u, ∀ b, consts g = some b → ReadUnit.bit Yc g = b) →
    q.okOf Yc = true →
    (∀ i < q.count, Hash.RowLeaf.accepts Verity.sha512 Verity.sha512
      (q.verifier Flock.Hm96.Default512.key.data.toList (q.domainOf Yc)) (q.rootOf Yc) i (table i)) →
    q.hOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u) =
        Hash.EmbeddingRead.h q.count q.words (q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u))
          (table (q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u))) ∨
      Hash.CollisionIn Verity.sha512
        (Hash.RowLeaf.outerHashed Verity.sha512 Verity.sha512
            (q.verifier Flock.Hm96.Default512.key.data.toList (q.domainOf Yc))
            (q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u))
            (q.opening ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u)) ++
          Hash.RowLeaf.outerHashed Verity.sha512 Verity.sha512
            (q.verifier Flock.Hm96.Default512.key.data.toList (q.domainOf Yc))
            (q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u))
            (table (q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u)))) ∨
      Hash.CollisionIn Verity.sha512
        (Hash.RowLeaf.innerHashed Verity.sha512
            (q.verifier Flock.Hm96.Default512.key.data.toList (q.domainOf Yc))
            (q.opening ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u)) ++
          Hash.RowLeaf.innerHashed Verity.sha512
            (q.verifier Flock.Hm96.Default512.key.data.toList (q.domainOf Yc))
            (table (q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u))))

end NciCheck.Properties.Record

namespace NciCheck.Properties.Record.SecurityProofs

open Accounting

/-- **The gates `gatesOf` gives are `gate`'s**, each at its index. -/
theorem gatesOf_some {N k : ℕ} {gate : ℕ → Option ℕ} {f : Fin k → Fin N} (h : gatesOf N k gate = some f)
    (i : Fin k) : gate i.val = some (f i).val := by
  unfold gatesOf at h
  by_cases hg : ∀ i : Fin k, ∃ g : Fin N, gate i.val = some g.val
  · rw [dite_eq_left hg] at h
    obtain rfl := Option.some.inj h
    exact Classical.choose_spec (hg i)
  · rw [dite_eq_right hg] at h
    cases h

/-- `gatesOf_some` at an index and its bound. -/
theorem gatesOf_at {N k : ℕ} {gate : ℕ → Option ℕ} {f : Fin k → Fin N} (h : gatesOf N k gate = some f) (i : ℕ)
    (hi : i < k) : gate i = some (f ⟨i, hi⟩).val :=
  gatesOf_some h ⟨i, hi⟩

/-- **The gates `portsAtGates` gives are the roles' gates**, bit by bit: the index's are the key parameter's, the
root's and the domain id's the root and domain parameters', the row returned the output row `out`'s and the verdict
the first bit of the verdict's row; the opened parameters are three, whose bits are the witness row's, its salt's and
its path's, a digest a level. -/
theorem portsAtGates_spec {C : Verity.Circuit} {inG outG : String → ℕ → Option ℕ} {ro : Roles} {out : String}
    {count words : ℕ} {position : ℕ → ℕ} {q : ReadUnit.Ports C}
    (h : portsAtGates C inG outG ro out count words position = some q) :
    (∀ i : Fin 32, inG ro.key i.val = some (q.tok i).val) ∧
      (∀ i : Fin 512, inG ro.root i.val = some (q.root i).val) ∧
      (∀ i : Fin 512, inG ro.domain i.val = some (q.domain i).val) ∧
      (∀ i : Fin (16 * q.words), outG out i.val = some (q.h i).val) ∧
      outG ro.verdict 0 = some q.ok.val ∧
      ro.opened.length = 3 ∧
      (∀ i : Fin (16 * q.words), inG (ro.opened.getD 0 "") i.val = some (q.row i).val) ∧
      (∀ i : Fin (8 * Hash.Hm96.SALT_BYTES), inG (ro.opened.getD 1 "") i.val = some (q.salt i).val) ∧
      ∀ (l : Fin (Hash.Tree.depth q.count)) (i : Fin 512),
        inG (ro.opened.getD 2 "") (512 * l.val + i.val) = some (q.path l i).val := by
  unfold portsAtGates at h
  by_cases hlen : ro.opened.length = 3
  · rw [ite_eq_left hlen] at h
    obtain ⟨tok, htok, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨root, hroot, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨domain, hdomain, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨rowG, hrow, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨saltG, hsalt, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨pathG, hpath, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨hG, hh, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨okG, hok, h⟩ := Option.map_eq_some_iff.1 h
    subst h
    exact ⟨gatesOf_some htok, gatesOf_some hroot, gatesOf_some hdomain, gatesOf_some hh,
      gatesOf_at hok 0 Nat.one_pos, hlen, gatesOf_some hrow, gatesOf_some hsalt,
      fun l i => gatesOf_at hpath (512 * l.val + i.val) _⟩
  · rw [ite_eq_right hlen] at h
    cases h

/-- **A pinned reader's unit returns its table's row** (`PinnedReadBinds`). -/
theorem pinnedReadBinds : PinnedReadBinds := by
  intro F c j tr _ P r name p pin k dfn ro out tags bytes tables ct hB n consts u q Yc table
    _ _ _ _ _ _ _ _ hgates hcorrect hconsts hok htable
  have hrows : ReadUnit.RegisteredRows consts (Flock.Bridge.partsOf (.tmpl ct) hB n) u q (q.rootOf Yc)
      (q.domainOf Yc) Yc :=
    ⟨rfl, rfl, hok, hconsts⟩
  have hb := Accounting.SecurityProofs.readUnitBinds Verity.sha512 Verity.sha512
    Flock.Hm96.Default512.key.data.toList consts (Flock.Bridge.partsOf (.tmpl ct) hB n) u q _ _ Yc hgates hcorrect
    hrows
  obtain ⟨hh, -⟩ := hgates.computes Yc hcorrect hconsts
  by_cases hlt : q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u) < q.count
  · rcases hb (table _) (htable _ hlt) with h | h | h
    · left
      rw [h]
      unfold Hash.EmbeddingRead.h
      exact (ite_eq_left hlt).symm
    · exact .inr (.inl h)
    · exact .inr (.inr h)
  · left
    rw [hh]
    unfold Hash.EmbeddingRead.h
    exact (ite_eq_right hlt).trans (ite_eq_right hlt).symm

end NciCheck.Properties.Record.SecurityProofs
